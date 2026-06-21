import json
import os
import pickle
import subprocess
import sys
from textwrap import dedent, indent

import pytest

import document_kv_cache.engine_probe as public_engine_probe
import restaurant_kv_serving.engine_probe as legacy_engine_probe
from document_kv_cache.admission import AdmissionQueue
from document_kv_cache.cache import ChunkCache
from document_kv_cache.engine_adapters import (
    ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE,
    EngineKVConnectorProbeResult,
    PayloadMode,
    ServingBackend,
    build_engine_adapter_request,
    engine_kv_connector_actions_from_record,
    engine_kv_connector_probe_result_to_record,
    read_engine_adapter_request_json,
    validate_engine_kv_connector_probe_record,
    vllm_adapter_spec,
    write_engine_adapter_request_json,
)
from document_kv_cache.engine_protocol import KVLayout
from document_kv_cache.engine_probe import (
    ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND,
    ENGINE_KV_PROBE_METADATA_HANDOFF_JSON,
    ENGINE_KV_PROBE_METADATA_PAYLOAD_URI,
    ENGINE_KV_PROBE_METADATA_PROBE_FACTORY,
    ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE,
    ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION,
    EngineKVProbeConfig,
    read_engine_adapter_payload,
    run_engine_kv_connector_probe,
    write_engine_adapter_handoff_bundle,
    write_engine_adapter_payload,
)
from document_kv_cache.kvpack import PackChunk, write_kvpack
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.materializer import KVMaterializer
from document_kv_cache.models import DocumentChunkType, DocumentKVRequest, KVCacheKey
from document_kv_cache.planner import CachePlanner
from document_kv_cache.service import DocumentKVService
from document_kv_cache.serving_env import serving_environment_profile
from document_kv_cache.storage import DiskRangeReader


BYTES_PER_TOKEN = 4
LAYOUT_VERSION = "toy-engine-probe-v1"
STATIC_TOKEN_COUNT = 2
CHUNK_TOKEN_COUNT = 3
STATIC_PAYLOAD = b"s" * (STATIC_TOKEN_COUNT * BYTES_PER_TOKEN)
CHUNK_PAYLOAD = b"c" * (CHUNK_TOKEN_COUNT * BYTES_PER_TOKEN)


def key(chunk_type: DocumentChunkType, chunk_id: str) -> KVCacheKey:
    return KVCacheKey.for_document(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_id="doc-a",
        chunk_type=chunk_type,
        chunk_id=chunk_id,
    )


def layout() -> KVLayout:
    return KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version=LAYOUT_VERSION,
        dtype="int8",
        num_layers=1,
        block_size=2,
        bytes_per_token=BYTES_PER_TOKEN,
    )


def service(tmp_path) -> DocumentKVService:
    refs = write_kvpack(
        tmp_path / "engine-probe.kvpack",
        [
            PackChunk(
                key(DocumentChunkType.DOCUMENT_STATIC, "static"),
                STATIC_PAYLOAD,
                STATIC_TOKEN_COUNT,
                "int8",
                LAYOUT_VERSION,
            ),
            PackChunk(
                key(DocumentChunkType.DOCUMENT_CHUNK, "section-1"),
                CHUNK_PAYLOAD,
                CHUNK_TOKEN_COUNT,
                "int8",
                LAYOUT_VERSION,
            ),
        ],
        align_bytes=1,
    )
    return DocumentKVService(
        planner=CachePlanner(InMemoryManifestStore(refs)),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=1024), reader=DiskRangeReader()),
        admission_queue=AdmissionQueue(max_pending_gpu_bytes=4096),
    )


def request() -> DocumentKVRequest:
    return DocumentKVRequest(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_chunks={"doc-a": ["section-1"]},
    )


def write_handoff_and_payload(tmp_path, *, segmented: bool = True) -> tuple[object, object]:
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=segmented)
    payload_path = tmp_path / "req-1.kv"
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    write_engine_adapter_payload(adapter_request, f"disk:{payload_path}")
    handoff_path = write_engine_adapter_request_json(
        adapter_request,
        tmp_path / "handoff.json",
        payload_uri=f"disk:{payload_path}",
    )
    return handoff_path, payload_path


def write_probe_factory_module(
    tmp_path,
    monkeypatch,
    *,
    module_name: str,
    engine_version: str | None,
    native_probe: bool = True,
    result_module: str = "document_kv_cache.engine_probe",
) -> str:
    module_path = tmp_path / f"{module_name}.py"
    if engine_version is None:
        factory_body = "return probe"
    else:
        factory_body = dedent(
            f"""
            return EngineKVProbeFactoryResult(
                probe=probe,
                engine_version={engine_version!r},
                native_probe={native_probe!r},
                metadata={{"probe.request_id": context.plan.request_id}},
            )
            """
        ).strip()
    module_text = (
        dedent(
            """
            from RESULT_MODULE import EngineKVProbeFactoryResult

            class Probe:
                def reserve_kv_blocks(self, action):
                    return {"request_id": action.request_id, "blocks": action.total_blocks}

                def import_kv_segment(self, reservation, action, payload):
                    if payload.nbytes != action.source_byte_length:
                        raise AssertionError("wrong slice length")

                def bind_kv_handle(self, reservation, action):
                    if reservation["request_id"] != action.request_id:
                        raise AssertionError("wrong reservation")

                def release_kv_blocks(self, reservation, action):
                    if reservation["request_id"] != action.request_id:
                        raise AssertionError("wrong release")

            def build_probe(context):
                probe = Probe()
            """
        ).replace("RESULT_MODULE", result_module)
        + indent(factory_body, "    ")
        + "\n"
    )
    module_path.write_text(module_text, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return f"{module_name}:build_probe"


def test_run_engine_kv_connector_probe_writes_native_release_record(tmp_path, monkeypatch):
    handoff_path, payload_path = write_handoff_and_payload(tmp_path, segmented=True)
    output_path = tmp_path / "probe-record.json"
    actions_output_path = tmp_path / "probe-actions.json"
    probe_factory = write_probe_factory_module(
        tmp_path,
        monkeypatch,
        module_name="native_probe_factory_success",
        engine_version="vllm-native-test",
    )

    result = run_engine_kv_connector_probe(
        EngineKVProbeConfig(
            handoff_json=handoff_path,
            probe_factory=probe_factory,
            output_json=output_path,
            actions_output_json=actions_output_path,
            expected_backend=ServingBackend.VLLM,
            metadata={
                "caller": "release-ci",
                ENGINE_KV_PROBE_METADATA_PROBE_FACTORY: "spoofed.factory:path",
            },
        )
    )

    record = engine_kv_connector_probe_result_to_record(result)
    profile = serving_environment_profile(ServingBackend.VLLM)
    assert record["backend"] == "vllm"
    assert record["engine_version"] == "vllm-native-test"
    assert record["payload_mode"] == "segmented"
    assert record["copied_segments"] == 2
    assert record["copied_tokens"] == STATIC_TOKEN_COUNT + CHUNK_TOKEN_COUNT
    assert record["copied_bytes"] == len(STATIC_PAYLOAD) + len(CHUNK_PAYLOAD)
    assert record["metadata"] == {
        ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND: "vllm",
        ENGINE_KV_PROBE_METADATA_HANDOFF_JSON: str(handoff_path),
        ENGINE_KV_PROBE_METADATA_PAYLOAD_URI: f"disk:{payload_path}",
        ENGINE_KV_PROBE_METADATA_PROBE_FACTORY: probe_factory,
        ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE: profile.engine_package,
        ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION: profile.engine_version,
        "caller": "release-ci",
        "probe.request_id": "req-1",
    }
    validate_engine_kv_connector_probe_record(record, expected_backend="vllm")
    assert json.loads(output_path.read_text(encoding="utf-8")) == record
    actions_record = json.loads(actions_output_path.read_text(encoding="utf-8"))
    actions = engine_kv_connector_actions_from_record(actions_record, expected_backend="vllm")
    assert actions_record["record_type"] == ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE
    assert actions_record["request_id"] == "req-1"
    assert actions.reservation.total_tokens == STATIC_TOKEN_COUNT + CHUNK_TOKEN_COUNT
    assert [copy["payload_index"] for copy in actions_record["copies"]] == [0, 1]


def test_engine_probe_module_execution_accepts_named_module_factory_result(tmp_path, monkeypatch):
    handoff_path, _ = write_handoff_and_payload(tmp_path, segmented=True)
    output_path = tmp_path / "probe-record.json"
    actions_output_path = tmp_path / "probe-actions.json"
    probe_factory = write_probe_factory_module(
        tmp_path,
        monkeypatch,
        module_name="module_execution_factory_result",
        engine_version="vllm-native-test",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.engine_probe",
            "--handoff-json",
            str(handoff_path),
            "--probe-factory",
            probe_factory,
            "--output-json",
            str(output_path),
            "--actions-output-json",
            str(actions_output_path),
            "--expected-backend",
            "vllm",
        ],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": f"src{os.pathsep}{tmp_path}"},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["engine_version"] == "vllm-native-test"
    assert record["native_probe"] is True
    assert actions_output_path.exists()


def test_run_engine_kv_connector_probe_does_not_write_actions_sidecar_on_probe_failure(
    tmp_path,
    monkeypatch,
):
    handoff_path, _ = write_handoff_and_payload(tmp_path, segmented=True)
    actions_output_path = tmp_path / "probe-actions.json"
    module_path = tmp_path / "failing_native_probe_factory.py"
    module_path.write_text(
        dedent(
            """
            from document_kv_cache.engine_probe import EngineKVProbeFactoryResult

            class Probe:
                def reserve_kv_blocks(self, action):
                    return {"request_id": action.request_id}

                def import_kv_segment(self, reservation, action, payload):
                    raise RuntimeError("native import failed")

                def bind_kv_handle(self, reservation, action):
                    raise AssertionError("bind should not run after import failure")

                def release_kv_blocks(self, reservation, action):
                    return None

            def build_probe(context):
                return EngineKVProbeFactoryResult(
                    probe=Probe(),
                    engine_version="vllm-native-test",
                )
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(RuntimeError, match="native import failed"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory="failing_native_probe_factory:build_probe",
                actions_output_json=actions_output_path,
                expected_backend=ServingBackend.VLLM,
            )
        )

    assert not actions_output_path.exists()


def test_run_engine_kv_connector_probe_rejects_native_engine_version_override(
    tmp_path,
    monkeypatch,
):
    handoff_path, _ = write_handoff_and_payload(tmp_path, segmented=True)
    probe_factory = write_probe_factory_module(
        tmp_path,
        monkeypatch,
        module_name="native_probe_factory_engine_version_spoof",
        engine_version="vllm-native-test",
    )

    with pytest.raises(ValueError, match="engine_version override"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory=probe_factory,
                expected_backend=ServingBackend.VLLM,
                engine_version="caller-spoofed-vllm",
            )
        )


def test_run_engine_kv_connector_probe_owns_expected_backend_metadata_without_config(
    tmp_path,
    monkeypatch,
):
    handoff_path, _ = write_handoff_and_payload(tmp_path, segmented=True)
    probe_factory = write_probe_factory_module(
        tmp_path,
        monkeypatch,
        module_name="native_probe_factory_backend_spoof",
        engine_version="vllm-native-test",
    )

    result = run_engine_kv_connector_probe(
        EngineKVProbeConfig(
            handoff_json=handoff_path,
            probe_factory=probe_factory,
            metadata={
                ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND: "sglang",
            },
        )
    )

    record = engine_kv_connector_probe_result_to_record(result)
    profile = serving_environment_profile(ServingBackend.VLLM)
    assert record["backend"] == "vllm"
    assert record["metadata"][ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND] == "vllm"
    assert (
        record["metadata"][ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE]
        == profile.engine_package
    )
    assert (
        record["metadata"][ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION]
        == profile.engine_version
    )


@pytest.mark.parametrize("module", [public_engine_probe, legacy_engine_probe])
def test_engine_probe_native_probe_flags_must_be_boolean(module, tmp_path):
    with pytest.raises(TypeError, match="native_probe must be boolean"):
        module.EngineKVProbeConfig(
            handoff_json=tmp_path / "handoff.json",
            probe_factory="some.module:factory",
            native_probe="false",
        )

    with pytest.raises(TypeError, match="native_probe must be boolean"):
        module.EngineKVProbeFactoryResult(
            probe=object(),
            engine_version="vllm-native-test",
            native_probe=1,
        )


def test_run_engine_kv_connector_probe_requires_engine_version_for_native_probe(tmp_path, monkeypatch):
    handoff_path, _ = write_handoff_and_payload(tmp_path, segmented=True)
    probe_factory = write_probe_factory_module(
        tmp_path,
        monkeypatch,
        module_name="native_probe_factory_missing_version",
        engine_version=None,
    )

    with pytest.raises(ValueError, match="engine_version"):
        run_engine_kv_connector_probe(
            EngineKVProbeConfig(
                handoff_json=handoff_path,
                probe_factory=probe_factory,
                expected_backend="vllm",
            )
        )


def test_run_engine_kv_connector_probe_allows_debug_non_native_probe_with_explicit_version(tmp_path, monkeypatch):
    handoff_path, _ = write_handoff_and_payload(tmp_path, segmented=False)
    probe_factory = write_probe_factory_module(
        tmp_path,
        monkeypatch,
        module_name="debug_probe_factory",
        engine_version=None,
    )

    result = run_engine_kv_connector_probe(
        EngineKVProbeConfig(
            handoff_json=handoff_path,
            probe_factory=probe_factory,
            expected_backend="vllm",
            engine_version="debug-adapter",
            native_probe=False,
        )
    )

    record = engine_kv_connector_probe_result_to_record(result)
    assert record["native_probe"] is False
    assert record["payload_mode"] == PayloadMode.MERGED.value
    assert record["metadata"][ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND] == "vllm"
    assert record["metadata"][ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE] == "vllm"
    assert (
        record["metadata"][ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION]
        == serving_environment_profile(ServingBackend.VLLM).engine_version
    )
    with pytest.raises(ValueError, match="native_probe=true"):
        validate_engine_kv_connector_probe_record(record)


def test_factory_result_can_force_non_native_probe_record(tmp_path, monkeypatch):
    handoff_path, _ = write_handoff_and_payload(tmp_path, segmented=False)
    probe_factory = write_probe_factory_module(
        tmp_path,
        monkeypatch,
        module_name="factory_forced_debug_probe",
        engine_version="debug-adapter",
        native_probe=False,
    )

    result = run_engine_kv_connector_probe(
        EngineKVProbeConfig(
            handoff_json=handoff_path,
            probe_factory=probe_factory,
            expected_backend="vllm",
        )
    )

    record = engine_kv_connector_probe_result_to_record(result)
    assert record["native_probe"] is False
    with pytest.raises(ValueError, match="native_probe=true"):
        validate_engine_kv_connector_probe_record(record)


def test_read_engine_adapter_payload_validates_local_uri_and_size(tmp_path):
    payload_path = tmp_path / "payload.kv"
    payload_path.write_bytes(b"payload")

    assert read_engine_adapter_payload(f"disk:{payload_path}", expected_bytes=7) == b"payload"
    assert read_engine_adapter_payload(str(payload_path), expected_bytes=7) == b"payload"
    with pytest.raises(ValueError, match="expected 8"):
        read_engine_adapter_payload(str(payload_path), expected_bytes=8)
    with pytest.raises(ValueError, match="absolute path"):
        read_engine_adapter_payload("relative.kv")
    with pytest.raises(ValueError, match="absolute paths"):
        read_engine_adapter_payload("disk:relative.kv")
    with pytest.raises(ValueError, match="absolute paths"):
        read_engine_adapter_payload("file:relative.kv")
    with pytest.raises(ValueError, match="dbfs:/"):
        read_engine_adapter_payload("dbfs:relative.kv")
    with pytest.raises(ValueError, match="disk:"):
        read_engine_adapter_payload("s3://bucket/payload.kv")
    with pytest.raises(ValueError, match="cannot contain"):
        read_engine_adapter_payload("dbfs:/../etc/passwd")
    with pytest.raises(ValueError, match="cannot contain"):
        read_engine_adapter_payload("uc-volume:/../../etc/passwd")
    with pytest.raises(ValueError, match="cannot contain"):
        read_engine_adapter_payload("/Volumes/catalog/schema/volume/../secret.kv")


@pytest.mark.parametrize("segmented", (False, True))
def test_write_engine_adapter_payload_writes_validated_request_payload(tmp_path, segmented):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=segmented)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    payload_path = tmp_path / "payloads" / f"{'segmented' if segmented else 'merged'}.kv"
    expected_payload = b"".join(ready.payload) if isinstance(ready.payload, tuple) else ready.payload

    output_path = write_engine_adapter_payload(adapter_request, f"disk:{payload_path}")

    assert output_path == payload_path
    assert payload_path.read_bytes() == expected_payload
    assert (
        read_engine_adapter_payload(f"disk:{payload_path}", expected_bytes=ready.handle.total_bytes)
        == expected_payload
    )


def test_write_engine_adapter_payload_rejects_unsupported_or_relative_uri(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())

    with pytest.raises(ValueError, match="absolute path"):
        write_engine_adapter_payload(adapter_request, "relative.kv")
    with pytest.raises(ValueError, match="absolute paths"):
        write_engine_adapter_payload(adapter_request, "disk:relative.kv")
    with pytest.raises(ValueError, match="disk:"):
        write_engine_adapter_payload(adapter_request, "s3://bucket/payload.kv")
    with pytest.raises(TypeError, match="EngineAdapterRequest"):
        write_engine_adapter_payload(ready, f"disk:{tmp_path / 'payload.kv'}")  # type: ignore[arg-type]


@pytest.mark.parametrize("segmented", (False, True))
def test_write_engine_adapter_handoff_bundle_writes_payload_and_record(tmp_path, segmented):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=segmented)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    handoff_path = tmp_path / "handoffs" / "req-1.json"
    payload_path = tmp_path / "payloads" / "req-1.kv"
    payload_uri = f"disk:{payload_path}"
    expected_payload = b"".join(ready.payload) if isinstance(ready.payload, tuple) else ready.payload

    written_handoff_path, written_payload_path = write_engine_adapter_handoff_bundle(
        adapter_request,
        f"disk:{handoff_path}",
        payload_uri=payload_uri,
    )

    assert written_handoff_path == handoff_path
    assert written_payload_path == payload_path
    assert payload_path.read_bytes() == expected_payload
    record = read_engine_adapter_request_json(handoff_path, expected_backend=ServingBackend.VLLM)
    assert record["payload_source"]["uri"] == payload_uri
    assert record["payload_source"]["total_bytes"] == ready.handle.total_bytes
    assert record["payload_mode"] == ("segmented" if segmented else "merged")
    assert read_engine_adapter_payload(payload_uri, expected_bytes=ready.handle.total_bytes) == expected_payload


def test_write_engine_adapter_handoff_bundle_rejects_non_adapter_request(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())

    with pytest.raises(TypeError, match="EngineAdapterRequest"):
        write_engine_adapter_handoff_bundle(  # type: ignore[arg-type]
            ready,
            tmp_path / "handoff.json",
            payload_uri=f"disk:{tmp_path / 'payload.kv'}",
        )


@pytest.mark.parametrize(
    ("handoff_value", "payload_value"),
    (
        ("absolute", "disk"),
        ("disk", "absolute"),
        ("file", "disk"),
    ),
)
def test_write_engine_adapter_handoff_bundle_rejects_same_resolved_destination(
    tmp_path,
    handoff_value,
    payload_value,
):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    shared_path = tmp_path / "shared-output"
    values = {
        "absolute": str(shared_path),
        "disk": f"disk:{shared_path}",
        "file": f"file:{shared_path}",
    }

    with pytest.raises(ValueError, match="different files"):
        write_engine_adapter_handoff_bundle(
            adapter_request,
            values[handoff_value],
            payload_uri=values[payload_value],
        )

    assert not shared_path.exists()


def test_write_engine_adapter_handoff_bundle_rejects_existing_hardlink_collision(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    payload_path = tmp_path / "payload.kv"
    handoff_path = tmp_path / "handoff.json"
    payload_path.write_bytes(b"existing")
    try:
        handoff_path.hardlink_to(payload_path)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable in this filesystem: {exc}")

    with pytest.raises(ValueError, match="different files"):
        write_engine_adapter_handoff_bundle(
            adapter_request,
            handoff_path,
            payload_uri=f"disk:{payload_path}",
        )

    assert payload_path.read_bytes() == b"existing"
    assert handoff_path.read_bytes() == b"existing"


def test_public_engine_probe_main_respects_document_namespace_hooks(monkeypatch, tmp_path):
    output_path = tmp_path / "probe.json"
    calls = {}

    def fake_run(config):
        calls["config"] = config
        return EngineKVConnectorProbeResult(
            backend=ServingBackend.VLLM,
            request_id="req-main",
            total_blocks=1,
            copied_segments=1,
            copied_tokens=1,
            copied_bytes=4,
            bound=True,
            released=True,
            model_id="qwen3:4b-instruct",
                layout_version=LAYOUT_VERSION,
            layout=layout(),
            payload_mode=PayloadMode.MERGED,
            connector_package="vllm",
            engine_version="vllm-native-test",
            metadata={"source": "public-hook"},
        )

    def fake_write(result, path):
        calls["record"] = engine_kv_connector_probe_result_to_record(result)
        calls["path"] = str(path)

    monkeypatch.setattr(public_engine_probe, "run_engine_kv_connector_probe", fake_run)
    monkeypatch.setattr(public_engine_probe, "write_engine_kv_connector_probe_result_json", fake_write)

    exit_code = public_engine_probe.main(
        [
            "--handoff-json",
            str(tmp_path / "handoff.json"),
            "--probe-factory",
            "some.module:factory",
            "--output-json",
            str(output_path),
            "--actions-output-json",
            str(tmp_path / "actions.json"),
            "--expected-backend",
            "vllm",
            "--metadata",
            "caller=test",
        ]
    )

    assert exit_code == 0
    assert calls["config"].handoff_json == tmp_path / "handoff.json"
    assert calls["config"].probe_factory == "some.module:factory"
    assert calls["config"].actions_output_json == tmp_path / "actions.json"
    assert calls["config"].metadata == {"caller": "test"}
    assert calls["path"] == str(output_path)
    assert calls["record"]["metadata"] == {"source": "public-hook"}


def test_legacy_engine_probe_main_respects_legacy_namespace_hooks(monkeypatch, tmp_path):
    output_path = tmp_path / "probe.json"
    calls = {}

    def fake_run(config):
        calls["config"] = config
        return EngineKVConnectorProbeResult(
            backend=ServingBackend.VLLM,
            request_id="req-main",
            total_blocks=1,
            copied_segments=1,
            copied_tokens=1,
            copied_bytes=4,
            bound=True,
            released=True,
            model_id="qwen3:4b-instruct",
            layout_version=LAYOUT_VERSION,
            layout=layout(),
            payload_mode=PayloadMode.MERGED,
            connector_package="vllm",
            engine_version="vllm-native-test",
            metadata={"source": "legacy-hook"},
        )

    def fake_write(result, path):
        calls["record"] = engine_kv_connector_probe_result_to_record(result)
        calls["path"] = str(path)

    monkeypatch.setattr(legacy_engine_probe, "run_engine_kv_connector_probe", fake_run)
    monkeypatch.setattr(legacy_engine_probe, "write_engine_kv_connector_probe_result_json", fake_write)

    exit_code = legacy_engine_probe.main(
        [
            "--handoff-json",
            str(tmp_path / "handoff.json"),
            "--probe-factory",
            "some.module:factory",
            "--output-json",
            str(output_path),
            "--actions-output-json",
            str(tmp_path / "legacy-actions.json"),
            "--expected-backend",
            "vllm",
            "--metadata",
            "caller=test",
        ]
    )

    assert exit_code == 0
    assert type(calls["config"]) is legacy_engine_probe.EngineKVProbeConfig
    assert calls["config"].handoff_json == tmp_path / "handoff.json"
    assert calls["config"].actions_output_json == tmp_path / "legacy-actions.json"
    assert calls["config"].metadata == {"caller": "test"}
    assert calls["path"] == str(output_path)
    assert calls["record"]["metadata"] == {"source": "legacy-hook"}


def test_legacy_engine_probe_ignores_document_namespace_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "probe.json"
    calls = {}

    def public_run_should_not_run(config):  # pragma: no cover - defensive assertion
        raise AssertionError("legacy main should not use document namespace monkeypatches")

    monkeypatch.setattr(public_engine_probe, "run_engine_kv_connector_probe", public_run_should_not_run)
    monkeypatch.setattr(
        legacy_engine_probe,
        "run_engine_kv_connector_probe",
        lambda config: EngineKVConnectorProbeResult(
            backend=ServingBackend.VLLM,
            request_id="req-main",
            total_blocks=1,
            copied_segments=1,
            copied_tokens=1,
            copied_bytes=4,
            bound=True,
            released=True,
            model_id="qwen3:4b-instruct",
            layout_version=LAYOUT_VERSION,
            layout=layout(),
            payload_mode=PayloadMode.MERGED,
            connector_package="vllm",
            engine_version="vllm-native-test",
            metadata={"source": "legacy"},
        ),
    )
    monkeypatch.setattr(
        legacy_engine_probe,
        "write_engine_kv_connector_probe_result_json",
        lambda result, path: calls.update(path=str(path), record=engine_kv_connector_probe_result_to_record(result)),
    )

    exit_code = legacy_engine_probe.main(
        [
            "--handoff-json",
            str(tmp_path / "handoff.json"),
            "--probe-factory",
            "some.module:factory",
            "--output-json",
            str(output_path),
            "--expected-backend",
            "vllm",
        ]
    )

    assert exit_code == 0
    assert calls["record"]["metadata"] == {"source": "legacy"}


def test_legacy_engine_probe_import_order_does_not_capture_public_monkeypatch():
    script = dedent(
        """
        import document_kv_cache.engine_probe as public_engine_probe

        def public_parse_should_not_run(items):
            raise AssertionError("legacy imported a public monkeypatch as its default")

        public_engine_probe._parse_metadata_items = public_parse_should_not_run

        import restaurant_kv_serving.engine_probe as legacy_engine_probe

        assert legacy_engine_probe._parse_metadata_items(("caller=test",)) == {"caller": "test"}
        try:
            public_engine_probe._parse_metadata_items(("caller=test",))
        except AssertionError:
            pass
        else:
            raise AssertionError("public monkeypatch was not installed")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_engine_probe_import_order_does_not_capture_public_constant_patch():
    script = dedent(
        """
        import document_kv_cache.engine_probe as public_engine_probe

        public_engine_probe.ENGINE_KV_PROBE_METADATA_PAYLOAD_URI = "patched.payload_uri"

        import restaurant_kv_serving.engine_probe as legacy_engine_probe

        config = legacy_engine_probe.EngineKVProbeConfig(
            handoff_json="handoff.json",
            probe_factory="some.module:factory",
        )
        metadata = legacy_engine_probe._probe_trace_metadata(
            config,
            payload_uri="disk:/tmp/payload.kv",
            backend=legacy_engine_probe.ServingBackend.VLLM,
        )
        assert metadata["document_kv.payload_uri"] == "disk:/tmp/payload.kv"
        assert "patched.payload_uri" not in metadata
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_engine_probe_uses_source_config_base_when_public_class_is_replaced_before_import():
    script = dedent(
        """
        import document_kv_cache.engine_probe as public_engine_probe

        public_engine_probe.EngineKVProbeConfig = object

        import restaurant_kv_serving.engine_probe as legacy_engine_probe

        config = legacy_engine_probe.EngineKVProbeConfig(
            handoff_json="handoff.json",
            probe_factory="some.module:factory",
        )
        assert type(config) is legacy_engine_probe.EngineKVProbeConfig
        assert config.handoff_json.name == "handoff.json"
        assert legacy_engine_probe.EngineKVProbeConfig.__module__ == "restaurant_kv_serving.engine_probe"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_engine_probe_uses_source_config_base_when_public_class_is_mutated_before_import():
    script = dedent(
        """
        import document_kv_cache.engine_probe as public_engine_probe

        public_engine_probe.EngineKVProbeConfig.probe_factory = property(lambda self: "")

        import restaurant_kv_serving.engine_probe as legacy_engine_probe

        config = legacy_engine_probe.EngineKVProbeConfig(
            handoff_json="handoff.json",
            probe_factory="some.module:factory",
        )
        assert config.probe_factory == "some.module:factory"
        assert legacy_engine_probe.EngineKVProbeConfig.__module__ == "restaurant_kv_serving.engine_probe"
        try:
            public_engine_probe.EngineKVProbeConfig(
                handoff_json="handoff.json",
                probe_factory="some.module:factory",
            )
        except (AttributeError, ValueError):
            pass
        else:
            raise AssertionError("public class mutation did not affect validation")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_engine_probe_private_hooks_are_isolated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    payload_path = tmp_path / "relative.kv"
    payload_path.write_bytes(b"payload")

    monkeypatch.setattr(legacy_engine_probe, "_validate_local_payload_uri", lambda payload_uri: None)

    assert legacy_engine_probe.read_engine_adapter_payload("relative.kv", expected_bytes=7) == b"payload"
    with pytest.raises(ValueError, match="absolute path"):
        public_engine_probe.read_engine_adapter_payload("relative.kv")


def test_legacy_engine_probe_accepts_document_factory_result(tmp_path, monkeypatch):
    handoff_path, _ = write_handoff_and_payload(tmp_path, segmented=True)
    probe_factory = write_probe_factory_module(
        tmp_path,
        monkeypatch,
        module_name="document_factory_result_for_legacy_probe",
        engine_version="vllm-native-test",
    )

    result = legacy_engine_probe.run_engine_kv_connector_probe(
        legacy_engine_probe.EngineKVProbeConfig(
            handoff_json=handoff_path,
            probe_factory=probe_factory,
            expected_backend="vllm",
        )
    )

    record = engine_kv_connector_probe_result_to_record(result)
    assert record["engine_version"] == "vllm-native-test"
    assert record["native_probe"] is True


def test_legacy_engine_probe_accepts_legacy_factory_result_after_public_result_class_mutation(
    tmp_path,
    monkeypatch,
):
    handoff_path, _ = write_handoff_and_payload(tmp_path, segmented=True)
    probe_factory = write_probe_factory_module(
        tmp_path,
        monkeypatch,
        module_name="legacy_factory_result_after_public_mutation",
        engine_version="vllm-native-test",
        result_module="restaurant_kv_serving.engine_probe",
    )
    public_engine_probe.EngineKVProbeFactoryResult.extra_marker = "mutated-after-legacy-import"

    result = legacy_engine_probe.run_engine_kv_connector_probe(
        legacy_engine_probe.EngineKVProbeConfig(
            handoff_json=handoff_path,
            probe_factory=probe_factory,
            expected_backend="vllm",
        )
    )

    record = engine_kv_connector_probe_result_to_record(result)
    assert record["engine_version"] == "vllm-native-test"
    assert record["native_probe"] is True
    assert record["metadata"]["probe.request_id"] == "req-1"


def test_engine_probe_reexports_document_owned_api_with_legacy_subclasses():
    assert public_engine_probe.EngineKVProbeConfig.__module__ == "document_kv_cache.engine_probe"
    assert legacy_engine_probe.EngineKVProbeConfig.__module__ == "restaurant_kv_serving.engine_probe"
    assert issubclass(legacy_engine_probe.EngineKVProbeConfig, public_engine_probe.EngineKVProbeConfig)
    assert issubclass(
        legacy_engine_probe.EngineKVProbeFactoryContext,
        public_engine_probe.EngineKVProbeFactoryContext,
    )
    assert issubclass(
        legacy_engine_probe.EngineKVProbeFactoryResult,
        public_engine_probe.EngineKVProbeFactoryResult,
    )
    assert public_engine_probe.run_engine_kv_connector_probe.__module__ == "document_kv_cache.engine_probe"
    assert legacy_engine_probe.run_engine_kv_connector_probe.__module__ == "restaurant_kv_serving.engine_probe"
    assert set(public_engine_probe.__all__) < set(legacy_engine_probe.__all__)
    assert "local_path" not in public_engine_probe.__all__
    assert "local_path" in legacy_engine_probe.__all__


def test_legacy_engine_probe_config_is_picklable_and_slotted(tmp_path):
    config = legacy_engine_probe.EngineKVProbeConfig(
        handoff_json=str(tmp_path / "handoff.json"),
        probe_factory="some.module:factory",
        metadata={"caller": "legacy"},
        actions_output_json=str(tmp_path / "actions.json"),
    )

    round_tripped = pickle.loads(pickle.dumps(config))

    assert type(config) is legacy_engine_probe.EngineKVProbeConfig
    assert isinstance(config, public_engine_probe.EngineKVProbeConfig)
    assert type(round_tripped) is legacy_engine_probe.EngineKVProbeConfig
    assert round_tripped.handoff_json == tmp_path / "handoff.json"
    assert round_tripped.actions_output_json == tmp_path / "actions.json"
    assert round_tripped.metadata == {"caller": "legacy"}
    assert not hasattr(config, "__dict__")


def test_engine_probe_star_import_surfaces_are_stable():
    expected_legacy_exports = {
        "EngineKVProbeConfig",
        "ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND",
        "ENGINE_KV_PROBE_METADATA_HANDOFF_JSON",
        "ENGINE_KV_PROBE_METADATA_PAYLOAD_URI",
        "ENGINE_KV_PROBE_METADATA_PROBE_FACTORY",
        "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE",
        "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION",
        "EngineKVProbeFactory",
        "EngineKVProbeFactoryContext",
        "EngineKVProbeFactoryResult",
        "run_engine_kv_connector_probe",
        "read_engine_adapter_payload",
        "write_engine_adapter_handoff_bundle",
        "write_engine_adapter_payload",
        "write_engine_kv_connector_actions_record_json",
        "write_engine_kv_connector_probe_result_json",
        "load_engine_kv_probe_factory",
        "parse_args",
        "main",
        "argparse",
        "importlib",
        "json",
        "Callable",
        "Mapping",
        "Sequence",
        "dataclass",
        "field",
        "Path",
        "MappingProxyType",
        "Any",
        "EngineKVBlockManagerProbe",
        "EngineKVConnectorProbeResult",
        "EngineKVInjectionPlan",
        "ServingBackend",
        "build_engine_kv_connector_actions",
        "build_engine_kv_injection_plan",
        "engine_kv_connector_probe_result_to_record",
        "probe_engine_kv_connector_actions",
        "read_engine_adapter_request_json",
        "validate_engine_kv_connector_probe_record",
        "view_engine_adapter_payload",
        "serving_environment_profile",
        "local_path",
    }
    public_namespace: dict[str, object] = {}
    legacy_namespace: dict[str, object] = {}

    exec("from document_kv_cache.engine_probe import *", public_namespace)
    exec("from restaurant_kv_serving.engine_probe import *", legacy_namespace)

    assert set(public_engine_probe.__all__) == set(public_namespace) - {"__builtins__"}
    assert set(legacy_engine_probe.__all__) == expected_legacy_exports
    assert set(legacy_engine_probe.__all__) == set(legacy_namespace) - {"__builtins__"}
    assert "local_path" not in public_namespace
    assert "local_path" in legacy_namespace


def test_legacy_engine_probe_module_execution_help():
    result = subprocess.run(
        [sys.executable, "-m", "restaurant_kv_serving.engine_probe", "--help"],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run a native engine KV connector probe" in result.stdout
