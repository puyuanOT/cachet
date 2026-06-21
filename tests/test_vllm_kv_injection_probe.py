import json
from textwrap import dedent

import pytest

from document_kv_cache.engine_adapters import (
    ServingBackend,
    build_engine_adapter_request,
    engine_kv_connector_probe_result_to_record,
    read_engine_adapter_request_json,
    validate_engine_kv_connector_probe_record,
    vllm_adapter_spec,
    write_engine_adapter_request_json,
)
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_probe import (
    ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND,
    ENGINE_KV_PROBE_METADATA_PROBE_FACTORY,
    EngineKVProbeConfig,
    run_engine_kv_connector_probe,
)
from document_kv_cache.model_profiles import layout_for_model
from document_kv_cache.native_probe_factories import (
    NATIVE_PROBE_DELEGATE_CONTRACT_ATTR,
    _inspect_delegate_adapter_contract,
    native_probe_adapter_contract_to_record,
)
from vllm_kv_injection.probe import (
    NativeVLLMConnectorFactoryResult,
    VLLM_DOCUMENT_KV_NATIVE_PROBE_CONNECTOR_FACTORY,
    VLLM_NATIVE_PROBE_CONTRACT,
    VLLM_PROBE_METADATA_CONNECTOR_CLASS,
    VLLM_PROBE_METADATA_CONNECTOR_FACTORY,
    VLLM_PROBE_METADATA_NATIVE_RUNTIME,
    VLLM_PROBE_METADATA_PROBE,
    VLLM_PROBE_METADATA_PROBE_KIND,
    VLLM_PROBE_METADATA_PROVIDER_FACTORY,
    VLLM_PROBE_METADATA_REQUEST_ID,
    VLLM_PROBE_METADATA_RUNTIME_CONTRACT,
    build_in_memory_debug_probe,
    build_native_connector_probe,
)
from vllm_kv_injection.vllm_native_provider import DOCUMENT_KV_NATIVE_PROVIDER_FACTORY
from vllm_kv_injection.protocol import KVCacheHandle, KVLayout, KVSegment
from vllm_kv_injection.vllm_runtime_contract import VLLM_KV_CONNECTOR_V1_RUNTIME
from vllm_kv_injection.vllm_runtime_contract import vllm_kv_connector_v1_contract_to_record


def handle() -> KVCacheHandle:
    return KVCacheHandle(
        request_id="req-1",
        handle_uri="document-kv://req-1",
        layout=KVLayout(
            model_id="test-model",
            lora_id="base",
            layout_version="test-v1",
            dtype="int8",
            num_layers=1,
            block_size=2,
            bytes_per_token=4,
        ),
        segments=(
            KVSegment("doc-a", "document_static", "static", 0, 2, 0, 8),
            KVSegment("doc-a", "document_chunk", "section-1", 2, 3, 8, 12),
        ),
        total_tokens=5,
        total_bytes=20,
    )


def qwen3_gqa_handle() -> KVCacheHandle:
    layout = layout_for_model("qwen3:4b-instruct", dtype="int8")
    return KVCacheHandle(
        request_id="req-qwen3-gqa",
        handle_uri="document-kv://req-qwen3-gqa",
        layout=layout,
        segments=(
            KVSegment("doc-qwen3", "document_static", "static", 0, 1, 0, layout.bytes_per_token),
            KVSegment(
                "doc-qwen3",
                "document_chunk",
                "section-1",
                1,
                1,
                layout.bytes_per_token,
                layout.bytes_per_token,
            ),
        ),
        total_tokens=2,
        total_bytes=2 * layout.bytes_per_token,
    )


def write_debug_handoff(tmp_path, ready: EngineReadyRequest):
    payload_path = tmp_path / f"{ready.handle.request_id}.kv"
    payload_path.write_bytes(b"".join(ready.payload) if isinstance(ready.payload, tuple) else ready.payload)
    return write_engine_adapter_request_json(
        build_engine_adapter_request(ready, spec=vllm_adapter_spec()),
        tmp_path / f"{ready.handle.request_id}-handoff.json",
        payload_uri=f"disk:{payload_path}",
    )


def run_debug_probe(tmp_path, ready: EngineReadyRequest):
    handoff_path = write_debug_handoff(tmp_path, ready)
    return run_engine_kv_connector_probe(
        EngineKVProbeConfig(
            handoff_json=handoff_path,
            probe_factory="vllm_kv_injection.probe:build_in_memory_debug_probe",
            expected_backend=ServingBackend.VLLM,
        )
    )


def write_native_connector_factory_module(tmp_path, monkeypatch, *, module_name: str) -> str:
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        dedent(
            """
            from vllm_kv_injection.block_mapping import plan_token_blocks
            from vllm_kv_injection.probe import NativeVLLMConnectorFactoryResult
            from vllm_kv_injection.vllm_dynamic_connector import VLLMSupportsHMA

            class NativeConnector(VLLMSupportsHMA):
                def reserve(self, handle):
                    handle.validate()
                    return plan_token_blocks(
                        total_tokens=handle.total_tokens,
                        block_size=handle.layout.block_size,
                        starting_block_id=17,
                    )

                def inject(self, handle, blocks, *, payload=None):
                    if payload is None:
                        raise AssertionError("native probe requires copied payload")
                    if not blocks:
                        raise AssertionError("native probe expected reserved blocks")

                def release(self, request_id):
                    if not request_id:
                        raise AssertionError("native probe expected request_id")

                def get_num_new_matched_tokens(self, request, num_computed_tokens):
                    return 0, False

                def update_state_after_alloc(self, request, blocks, num_external_tokens):
                    return None

                def build_connector_meta(self, scheduler_output):
                    return {}

                def register_kv_caches(self, kv_caches):
                    return None

                def start_load_kv(self, forward_context, **kwargs):
                    return None

                def wait_for_layer_load(self, layer_name):
                    return None

                def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
                    return None

                def wait_for_save(self):
                    return None

                def request_finished(self, request, block_ids):
                    return False, None

                def request_finished_all_groups(self, request, block_ids):
                    return False, None

            def build_connector(context):
                return NativeVLLMConnectorFactoryResult(
                    connector=NativeConnector(),
                    engine_version="vllm-native-test",
                    metadata={"runtime.owner": context.backend.value},
                )
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return f"{module_name}:build_connector"


def write_non_hma_native_connector_factory_module(tmp_path, monkeypatch, *, module_name: str) -> str:
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        dedent(
            """
            from vllm_kv_injection.block_mapping import plan_token_blocks
            from vllm_kv_injection.probe import NativeVLLMConnectorFactoryResult

            class NativeConnector:
                def reserve(self, handle):
                    return plan_token_blocks(
                        total_tokens=handle.total_tokens,
                        block_size=handle.layout.block_size,
                    )

                def inject(self, handle, blocks, *, payload=None):
                    return None

                def release(self, request_id):
                    return None

                def get_num_new_matched_tokens(self, request, num_computed_tokens):
                    return 0, False

                def update_state_after_alloc(self, request, blocks, num_external_tokens):
                    return None

                def build_connector_meta(self, scheduler_output):
                    return {}

                def register_kv_caches(self, kv_caches):
                    return None

                def start_load_kv(self, forward_context, **kwargs):
                    return None

                def wait_for_layer_load(self, layer_name):
                    return None

                def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
                    return None

                def wait_for_save(self):
                    return None

                def request_finished(self, request, block_ids):
                    return False, None

                def request_finished_all_groups(self, request, block_ids):
                    return False, None

            def build_connector(context):
                return NativeVLLMConnectorFactoryResult(
                    connector=NativeConnector(),
                    engine_version="vllm-native-test",
                )
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return f"{module_name}:build_connector"


def write_handoff_only_native_connector_factory_module(tmp_path, monkeypatch, *, module_name: str) -> str:
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        dedent(
            """
            from vllm_kv_injection.block_mapping import plan_token_blocks
            from vllm_kv_injection.probe import NativeVLLMConnectorFactoryResult

            class NativeConnector:
                def reserve(self, handle):
                    return plan_token_blocks(
                        total_tokens=handle.total_tokens,
                        block_size=handle.layout.block_size,
                    )

                def inject(self, handle, blocks, *, payload=None):
                    return None

                def release(self, request_id):
                    return None

            def build_connector(context):
                return NativeVLLMConnectorFactoryResult(
                    connector=NativeConnector(),
                    engine_version="vllm-native-test",
                )
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return f"{module_name}:build_connector"


def write_in_memory_native_connector_factory_module(tmp_path, monkeypatch, *, module_name: str) -> str:
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        dedent(
            """
            from vllm_kv_injection.connector import InMemoryKVConnector
            from vllm_kv_injection.probe import NativeVLLMConnectorFactoryResult

            def build_connector(context):
                return NativeVLLMConnectorFactoryResult(
                    connector=InMemoryKVConnector(),
                    engine_version="vllm-native-test",
                )
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return f"{module_name}:build_connector"


def write_invalid_native_connector_factory_module(tmp_path, monkeypatch, *, module_name: str) -> str:
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        dedent(
            """
            from vllm_kv_injection.probe import NativeVLLMConnectorFactoryResult

            def build_connector(context):
                return NativeVLLMConnectorFactoryResult(
                    connector=object(),
                    engine_version="vllm-native-test",
                )
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return f"{module_name}:build_connector"


def test_build_in_memory_debug_probe_for_engine_probe_runner(tmp_path):
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    result = run_debug_probe(tmp_path, ready)

    record = engine_kv_connector_probe_result_to_record(result)
    assert record["backend"] == "vllm"
    assert record["native_probe"] is False
    assert record["engine_version"] == "vllm-in-memory-debug"
    assert record["copied_segments"] == 2
    assert record["copied_tokens"] == 5
    assert record["copied_bytes"] == 20
    assert record["metadata"][ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND] == "vllm"
    assert record["metadata"][ENGINE_KV_PROBE_METADATA_PROBE_FACTORY] == (
        "vllm_kv_injection.probe:build_in_memory_debug_probe"
    )
    assert record["metadata"][VLLM_PROBE_METADATA_CONNECTOR_CLASS] == "InMemoryKVConnector"
    assert record["metadata"][VLLM_PROBE_METADATA_NATIVE_RUNTIME] == "false"
    assert record["metadata"][VLLM_PROBE_METADATA_PROBE] == "in_memory_debug"
    assert record["metadata"][VLLM_PROBE_METADATA_PROBE_KIND] == "debug_in_memory"
    assert record["metadata"][VLLM_PROBE_METADATA_REQUEST_ID] == "req-1"
    assert record["metadata"][VLLM_PROBE_METADATA_RUNTIME_CONTRACT] == "document-kv-debug-connector"
    try:
        validate_engine_kv_connector_probe_record(record)
    except ValueError as exc:
        assert "native_probe=true" in str(exc)
    else:  # pragma: no cover
        raise AssertionError(json.dumps(record, sort_keys=True))


def test_build_in_memory_debug_probe_accepts_qwen3_gqa_release_layout(tmp_path):
    kv_handle = qwen3_gqa_handle()
    ready = EngineReadyRequest(
        handle=kv_handle,
        payload=(b"s" * kv_handle.layout.bytes_per_token, b"c" * kv_handle.layout.bytes_per_token),
        estimated_gpu_bytes=kv_handle.total_bytes * 2,
    )

    handoff_path = write_debug_handoff(tmp_path, ready)
    handoff_record = read_engine_adapter_request_json(handoff_path, expected_backend=ServingBackend.VLLM)
    handoff_layout = handoff_record["handle"]["layout"]
    assert handoff_layout["num_query_heads"] == 32
    assert handoff_layout["num_kv_heads"] == 8
    assert handoff_layout["attention_mechanism"] == "gqa"
    assert handoff_layout["shares_kv_storage"] is True
    result = run_engine_kv_connector_probe(
        EngineKVProbeConfig(
            handoff_json=handoff_path,
            probe_factory="vllm_kv_injection.probe:build_in_memory_debug_probe",
            expected_backend=ServingBackend.VLLM,
        )
    )

    record = engine_kv_connector_probe_result_to_record(result)
    assert record["backend"] == "vllm"
    assert record["model_id"] == "qwen3:4b-instruct"
    assert record["layout_version"] == "qwen3-v1"
    assert record["copied_tokens"] == 2
    assert record["copied_bytes"] == kv_handle.total_bytes
    assert record["native_probe"] is False
    with pytest.raises(ValueError, match="native_probe=true"):
        validate_engine_kv_connector_probe_record(record)


def test_build_native_connector_probe_for_engine_probe_runner(tmp_path):
    pytest.importorskip("torch")
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    handoff_path = write_debug_handoff(tmp_path, ready)

    result = run_engine_kv_connector_probe(
        EngineKVProbeConfig(
            handoff_json=handoff_path,
            probe_factory="vllm_kv_injection.probe:build_native_connector_probe",
            expected_backend=ServingBackend.VLLM,
            metadata={VLLM_PROBE_METADATA_CONNECTOR_FACTORY: VLLM_DOCUMENT_KV_NATIVE_PROBE_CONNECTOR_FACTORY},
        )
    )

    record = engine_kv_connector_probe_result_to_record(result)
    assert record["backend"] == "vllm"
    assert record["native_probe"] is True
    assert record["engine_version"].startswith("vllm-")
    assert record["metadata"][VLLM_PROBE_METADATA_CONNECTOR_CLASS] == "DocumentKVNativeProbeConnector"
    assert record["metadata"][VLLM_PROBE_METADATA_CONNECTOR_FACTORY] == VLLM_DOCUMENT_KV_NATIVE_PROBE_CONNECTOR_FACTORY
    assert record["metadata"][VLLM_PROBE_METADATA_NATIVE_RUNTIME] == "true"
    assert record["metadata"][VLLM_PROBE_METADATA_PROBE] == "native_connector"
    assert record["metadata"][VLLM_PROBE_METADATA_PROBE_KIND] == "native_runtime"
    assert record["metadata"][VLLM_PROBE_METADATA_PROVIDER_FACTORY] == DOCUMENT_KV_NATIVE_PROVIDER_FACTORY
    assert record["metadata"][VLLM_PROBE_METADATA_REQUEST_ID] == "req-1"
    assert record["metadata"][VLLM_PROBE_METADATA_RUNTIME_CONTRACT] == VLLM_KV_CONNECTOR_V1_RUNTIME
    assert record["metadata"]["runtime.owner"] == "vllm"
    validate_engine_kv_connector_probe_record(record)


def test_build_native_connector_probe_requires_connector_factory_metadata(tmp_path):
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    handoff_path = write_debug_handoff(tmp_path, ready)

    with pytest.raises(ValueError, match=VLLM_PROBE_METADATA_CONNECTOR_FACTORY):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="vllm_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.VLLM,
            )
        )


@pytest.mark.parametrize(
    "wrapper_key",
    [
        VLLM_PROBE_METADATA_CONNECTOR_CLASS,
        VLLM_PROBE_METADATA_NATIVE_RUNTIME,
        VLLM_PROBE_METADATA_PROBE,
        VLLM_PROBE_METADATA_PROBE_KIND,
        VLLM_PROBE_METADATA_REQUEST_ID,
        VLLM_PROBE_METADATA_RUNTIME_CONTRACT,
    ],
)
def test_build_native_connector_probe_rejects_wrapper_metadata_overrides(
    tmp_path,
    monkeypatch,
    wrapper_key,
):
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    handoff_path = write_debug_handoff(tmp_path, ready)
    connector_factory = write_native_connector_factory_module(
        tmp_path,
        monkeypatch,
        module_name=f"native_vllm_spoof_{wrapper_key.rsplit('.', 1)[-1]}",
    )

    with pytest.raises(ValueError, match="wrapper-owned"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="vllm_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.VLLM,
                metadata={
                    VLLM_PROBE_METADATA_CONNECTOR_FACTORY: connector_factory,
                    wrapper_key: "caller-owned",
                },
            )
        )


def test_build_native_connector_probe_rejects_in_memory_connector(tmp_path, monkeypatch):
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    handoff_path = write_debug_handoff(tmp_path, ready)
    connector_factory = write_in_memory_native_connector_factory_module(
        tmp_path,
        monkeypatch,
        module_name="native_vllm_in_memory_probe_factory",
    )

    with pytest.raises(ValueError, match="InMemoryKVConnector"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="vllm_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.VLLM,
                metadata={VLLM_PROBE_METADATA_CONNECTOR_FACTORY: connector_factory},
            )
        )


def test_build_native_connector_probe_rejects_v1_connector_without_provider_wiring(tmp_path, monkeypatch):
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    handoff_path = write_debug_handoff(tmp_path, ready)
    connector_factory = write_native_connector_factory_module(
        tmp_path,
        monkeypatch,
        module_name="native_vllm_no_provider_probe_factory",
    )

    with pytest.raises(TypeError, match="DocumentKVNativeProvider"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="vllm_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.VLLM,
                metadata={VLLM_PROBE_METADATA_CONNECTOR_FACTORY: connector_factory},
            )
        )


def test_build_native_connector_probe_rejects_connector_without_required_methods(tmp_path, monkeypatch):
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    handoff_path = write_debug_handoff(tmp_path, ready)
    connector_factory = write_invalid_native_connector_factory_module(
        tmp_path,
        monkeypatch,
        module_name="native_vllm_invalid_probe_factory",
    )

    with pytest.raises(TypeError, match="reserve, inject, release"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="vllm_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.VLLM,
                metadata={VLLM_PROBE_METADATA_CONNECTOR_FACTORY: connector_factory},
            )
        )


def test_build_native_connector_probe_rejects_handoff_only_connector(tmp_path, monkeypatch):
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    handoff_path = write_debug_handoff(tmp_path, ready)
    connector_factory = write_handoff_only_native_connector_factory_module(
        tmp_path,
        monkeypatch,
        module_name="native_vllm_handoff_only_probe_factory",
    )

    with pytest.raises(TypeError, match="get_num_new_matched_tokens"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="vllm_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.VLLM,
                metadata={VLLM_PROBE_METADATA_CONNECTOR_FACTORY: connector_factory},
            )
        )


def test_build_native_connector_probe_rejects_non_hma_connector(tmp_path, monkeypatch):
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    handoff_path = write_debug_handoff(tmp_path, ready)
    connector_factory = write_non_hma_native_connector_factory_module(
        tmp_path,
        monkeypatch,
        module_name="native_vllm_non_hma_probe_factory",
    )

    with pytest.raises(TypeError, match="SupportsHMA"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="vllm_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.VLLM,
                metadata={VLLM_PROBE_METADATA_CONNECTOR_FACTORY: connector_factory},
            )
        )


def test_build_native_connector_probe_rejects_broken_vllm_runtime(tmp_path, monkeypatch):
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    handoff_path = write_debug_handoff(tmp_path, ready)
    connector_factory = write_native_connector_factory_module(
        tmp_path,
        monkeypatch,
        module_name="native_vllm_broken_runtime_probe_factory",
    )
    monkeypatch.setattr(
        "vllm_kv_injection.probe.vllm_runtime_import_error",
        lambda: RuntimeError("torch inductor mismatch"),
    )

    with pytest.raises(RuntimeError, match="vLLM runtime import failed"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="vllm_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.VLLM,
                metadata={VLLM_PROBE_METADATA_CONNECTOR_FACTORY: connector_factory},
            )
        )


def test_build_native_connector_probe_rejects_transitive_missing_vllm_dependency(
    tmp_path,
    monkeypatch,
):
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    handoff_path = write_debug_handoff(tmp_path, ready)
    connector_factory = write_native_connector_factory_module(
        tmp_path,
        monkeypatch,
        module_name="native_vllm_missing_dependency_probe_factory",
    )
    monkeypatch.setattr(
        "vllm_kv_injection.probe.vllm_runtime_import_error",
        lambda: ModuleNotFoundError("No module named 'cuda_runtime'", name="cuda_runtime"),
    )

    with pytest.raises(RuntimeError, match="vLLM runtime import failed"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="vllm_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.VLLM,
                metadata={VLLM_PROBE_METADATA_CONNECTOR_FACTORY: connector_factory},
            )
        )


def test_build_in_memory_debug_probe_is_exported_debug_factory():
    assert build_in_memory_debug_probe.__name__ == "build_in_memory_debug_probe"
    assert build_native_connector_probe.__name__ == "build_native_connector_probe"
    assert NativeVLLMConnectorFactoryResult.__name__ == "NativeVLLMConnectorFactoryResult"
    import vllm_kv_injection

    assert vllm_kv_injection.VLLM_PROBE_METADATA_PROBE_KIND == VLLM_PROBE_METADATA_PROBE_KIND
    assert vllm_kv_injection.VLLM_PROBE_METADATA_CONNECTOR_FACTORY == VLLM_PROBE_METADATA_CONNECTOR_FACTORY
    assert vllm_kv_injection.VLLM_PROBE_METADATA_RUNTIME_CONTRACT == VLLM_PROBE_METADATA_RUNTIME_CONTRACT
    assert vllm_kv_injection.build_native_connector_probe is build_native_connector_probe


def test_native_connector_probe_declares_document_kv_contract():
    assert VLLM_NATIVE_PROBE_CONTRACT == native_probe_adapter_contract_to_record()
    assert getattr(build_native_connector_probe, NATIVE_PROBE_DELEGATE_CONTRACT_ATTR) == (
        native_probe_adapter_contract_to_record()
    )
    assert not hasattr(build_in_memory_debug_probe, NATIVE_PROBE_DELEGATE_CONTRACT_ATTR)
    _contract, native_valid, _reason = _inspect_delegate_adapter_contract(build_native_connector_probe)
    _debug_contract, debug_valid, _debug_reason = _inspect_delegate_adapter_contract(build_in_memory_debug_probe)
    assert native_valid is True
    assert debug_valid is False


def test_native_connector_probe_declares_vllm_runtime_contract():
    assert getattr(build_native_connector_probe, "document_kv_native_probe_runtime_contract") == (
        vllm_kv_connector_v1_contract_to_record(
            handoff_contract=native_probe_adapter_contract_to_record(),
        )
    )
    assert not hasattr(build_in_memory_debug_probe, "document_kv_native_probe_runtime_contract")
