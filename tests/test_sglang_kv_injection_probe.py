import json
from textwrap import dedent

import pytest

from document_kv_cache.engine_adapters import (
    ServingBackend,
    build_engine_adapter_request,
    engine_kv_connector_probe_result_to_record,
    read_engine_adapter_request_json,
    sglang_adapter_spec,
    validate_engine_kv_connector_probe_record,
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
from sglang_kv_injection.probe import (
    NativeSGLangConnectorFactoryResult,
    SGLANG_NATIVE_PROBE_CONTRACT,
    SGLANG_PROBE_METADATA_CONNECTOR_CLASS,
    SGLANG_PROBE_METADATA_CONNECTOR_FACTORY,
    SGLANG_PROBE_METADATA_NATIVE_RUNTIME,
    SGLANG_PROBE_METADATA_PROBE,
    SGLANG_PROBE_METADATA_PROBE_KIND,
    SGLANG_PROBE_METADATA_REQUEST_ID,
    SGLANG_PROBE_METADATA_RUNTIME_CONTRACT,
    build_in_memory_debug_probe,
    build_native_connector_probe,
)
from sglang_kv_injection.protocol import KVCacheHandle, KVLayout, KVSegment
from sglang_kv_injection.sglang_runtime_contract import (
    SGLANG_RUNTIME_CACHE_RUNTIME,
    sglang_runtime_cache_contract_to_record,
)


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
        build_engine_adapter_request(ready, spec=sglang_adapter_spec()),
        tmp_path / f"{ready.handle.request_id}-handoff.json",
        payload_uri=f"disk:{payload_path}",
    )


def run_debug_probe(tmp_path, ready: EngineReadyRequest):
    handoff_path = write_debug_handoff(tmp_path, ready)
    return run_engine_kv_connector_probe(
        EngineKVProbeConfig(
            handoff_json=handoff_path,
            probe_factory="sglang_kv_injection.probe:build_in_memory_debug_probe",
            expected_backend=ServingBackend.SGLANG,
        )
    )


def write_native_connector_factory_module(tmp_path, monkeypatch, *, module_name: str) -> str:
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        dedent(
            """
            from sglang_kv_injection.probe import NativeSGLangConnectorFactoryResult

            class NativeConnector:
                def __init__(self):
                    self.staged = {}

                def stage(self, record, *, payload=None):
                    if payload is None:
                        raise AssertionError("native probe requires copied payload")
                    self.staged[record.handle_uri] = record

                def attach(self, *, request_id, record):
                    if record.handle_uri not in self.staged:
                        raise AssertionError("native probe expected staged record")

                def release(self, request_id):
                    if not request_id:
                        raise AssertionError("native probe expected request_id")

            def build_connector(context):
                return NativeSGLangConnectorFactoryResult(
                    connector=NativeConnector(),
                    engine_version="sglang-native-test",
                    metadata={"runtime.owner": context.backend.value},
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
            from sglang_kv_injection.connector import InMemorySGLangKVConnector
            from sglang_kv_injection.probe import NativeSGLangConnectorFactoryResult

            def build_connector(context):
                return NativeSGLangConnectorFactoryResult(
                    connector=InMemorySGLangKVConnector(),
                    engine_version="sglang-native-test",
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
            from sglang_kv_injection.probe import NativeSGLangConnectorFactoryResult

            def build_connector(context):
                return NativeSGLangConnectorFactoryResult(
                    connector=object(),
                    engine_version="sglang-native-test",
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
    assert record["backend"] == "sglang"
    assert record["native_probe"] is False
    assert record["engine_version"] == "sglang-in-memory-debug"
    assert record["copied_segments"] == 2
    assert record["copied_tokens"] == 5
    assert record["copied_bytes"] == 20
    assert record["metadata"][ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND] == "sglang"
    assert record["metadata"][ENGINE_KV_PROBE_METADATA_PROBE_FACTORY] == (
        "sglang_kv_injection.probe:build_in_memory_debug_probe"
    )
    assert record["metadata"][SGLANG_PROBE_METADATA_CONNECTOR_CLASS] == "InMemorySGLangKVConnector"
    assert record["metadata"][SGLANG_PROBE_METADATA_NATIVE_RUNTIME] == "false"
    assert record["metadata"][SGLANG_PROBE_METADATA_PROBE] == "in_memory_debug"
    assert record["metadata"][SGLANG_PROBE_METADATA_PROBE_KIND] == "debug_in_memory"
    assert record["metadata"][SGLANG_PROBE_METADATA_REQUEST_ID] == "req-1"
    assert record["metadata"][SGLANG_PROBE_METADATA_RUNTIME_CONTRACT] == "document-kv-debug-connector"
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
    handoff_record = read_engine_adapter_request_json(handoff_path, expected_backend=ServingBackend.SGLANG)
    handoff_layout = handoff_record["handle"]["layout"]
    assert handoff_layout["num_query_heads"] == 32
    assert handoff_layout["num_kv_heads"] == 8
    assert handoff_layout["attention_mechanism"] == "gqa"
    assert handoff_layout["shares_kv_storage"] is True
    result = run_engine_kv_connector_probe(
        EngineKVProbeConfig(
            handoff_json=handoff_path,
            probe_factory="sglang_kv_injection.probe:build_in_memory_debug_probe",
            expected_backend=ServingBackend.SGLANG,
        )
    )

    record = engine_kv_connector_probe_result_to_record(result)
    assert record["backend"] == "sglang"
    assert record["model_id"] == "qwen3:4b-instruct"
    assert record["layout_version"] == "qwen3-v1"
    assert record["copied_tokens"] == 2
    assert record["copied_bytes"] == kv_handle.total_bytes
    assert record["native_probe"] is False
    with pytest.raises(ValueError, match="native_probe=true"):
        validate_engine_kv_connector_probe_record(record)


def test_build_native_connector_probe_for_engine_probe_runner(tmp_path, monkeypatch):
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    handoff_path = write_debug_handoff(tmp_path, ready)
    connector_factory = write_native_connector_factory_module(
        tmp_path,
        monkeypatch,
        module_name="native_sglang_probe_factory",
    )

    result = run_engine_kv_connector_probe(
        EngineKVProbeConfig(
            handoff_json=handoff_path,
            probe_factory="sglang_kv_injection.probe:build_native_connector_probe",
            expected_backend=ServingBackend.SGLANG,
            metadata={SGLANG_PROBE_METADATA_CONNECTOR_FACTORY: connector_factory},
        )
    )

    record = engine_kv_connector_probe_result_to_record(result)
    assert record["backend"] == "sglang"
    assert record["native_probe"] is True
    assert record["engine_version"] == "sglang-native-test"
    assert record["metadata"][SGLANG_PROBE_METADATA_CONNECTOR_CLASS] == "NativeConnector"
    assert record["metadata"][SGLANG_PROBE_METADATA_CONNECTOR_FACTORY] == connector_factory
    assert record["metadata"][SGLANG_PROBE_METADATA_NATIVE_RUNTIME] == "true"
    assert record["metadata"][SGLANG_PROBE_METADATA_PROBE] == "native_connector"
    assert record["metadata"][SGLANG_PROBE_METADATA_PROBE_KIND] == "native_runtime"
    assert record["metadata"][SGLANG_PROBE_METADATA_REQUEST_ID] == "req-1"
    assert record["metadata"][SGLANG_PROBE_METADATA_RUNTIME_CONTRACT] == SGLANG_RUNTIME_CACHE_RUNTIME
    assert record["metadata"]["runtime.owner"] == "sglang"
    validate_engine_kv_connector_probe_record(record)


def test_build_native_connector_probe_requires_connector_factory_metadata(tmp_path):
    ready = EngineReadyRequest(
        handle=handle(),
        payload=(b"s" * 8, b"c" * 12),
        estimated_gpu_bytes=40,
    )
    handoff_path = write_debug_handoff(tmp_path, ready)

    with pytest.raises(ValueError, match=SGLANG_PROBE_METADATA_CONNECTOR_FACTORY):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="sglang_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.SGLANG,
            )
        )


@pytest.mark.parametrize(
    "wrapper_key",
    [
        SGLANG_PROBE_METADATA_CONNECTOR_CLASS,
        SGLANG_PROBE_METADATA_NATIVE_RUNTIME,
        SGLANG_PROBE_METADATA_PROBE,
        SGLANG_PROBE_METADATA_PROBE_KIND,
        SGLANG_PROBE_METADATA_REQUEST_ID,
        SGLANG_PROBE_METADATA_RUNTIME_CONTRACT,
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
        module_name=f"native_sglang_spoof_{wrapper_key.rsplit('.', 1)[-1]}",
    )

    with pytest.raises(ValueError, match="wrapper-owned"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="sglang_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.SGLANG,
                metadata={
                    SGLANG_PROBE_METADATA_CONNECTOR_FACTORY: connector_factory,
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
        module_name="native_sglang_in_memory_probe_factory",
    )

    with pytest.raises(ValueError, match="InMemorySGLangKVConnector"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="sglang_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.SGLANG,
                metadata={SGLANG_PROBE_METADATA_CONNECTOR_FACTORY: connector_factory},
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
        module_name="native_sglang_invalid_probe_factory",
    )

    with pytest.raises(TypeError, match="stage, attach, release"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="sglang_kv_injection.probe:build_native_connector_probe",
                expected_backend=ServingBackend.SGLANG,
                metadata={SGLANG_PROBE_METADATA_CONNECTOR_FACTORY: connector_factory},
            )
        )


def test_build_in_memory_debug_probe_is_exported_debug_factory():
    assert build_in_memory_debug_probe.__name__ == "build_in_memory_debug_probe"
    assert build_native_connector_probe.__name__ == "build_native_connector_probe"
    assert NativeSGLangConnectorFactoryResult.__name__ == "NativeSGLangConnectorFactoryResult"
    import sglang_kv_injection

    assert sglang_kv_injection.SGLANG_PROBE_METADATA_PROBE_KIND == SGLANG_PROBE_METADATA_PROBE_KIND
    assert sglang_kv_injection.SGLANG_PROBE_METADATA_CONNECTOR_FACTORY == SGLANG_PROBE_METADATA_CONNECTOR_FACTORY
    assert sglang_kv_injection.SGLANG_PROBE_METADATA_RUNTIME_CONTRACT == SGLANG_PROBE_METADATA_RUNTIME_CONTRACT
    assert sglang_kv_injection.build_native_connector_probe is build_native_connector_probe


def test_native_connector_probe_declares_document_kv_contract():
    assert SGLANG_NATIVE_PROBE_CONTRACT == native_probe_adapter_contract_to_record()
    assert getattr(build_native_connector_probe, NATIVE_PROBE_DELEGATE_CONTRACT_ATTR) == (
        native_probe_adapter_contract_to_record()
    )
    assert not hasattr(build_in_memory_debug_probe, NATIVE_PROBE_DELEGATE_CONTRACT_ATTR)
    _contract, native_valid, _reason = _inspect_delegate_adapter_contract(build_native_connector_probe)
    _debug_contract, debug_valid, _debug_reason = _inspect_delegate_adapter_contract(build_in_memory_debug_probe)
    assert native_valid is True
    assert debug_valid is False


def test_native_connector_probe_declares_sglang_runtime_contract():
    assert getattr(build_native_connector_probe, "document_kv_native_probe_runtime_contract") == (
        sglang_runtime_cache_contract_to_record(
            handoff_contract=native_probe_adapter_contract_to_record(),
        )
    )
    assert not hasattr(build_in_memory_debug_probe, "document_kv_native_probe_runtime_contract")
