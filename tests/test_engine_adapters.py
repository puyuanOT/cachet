import json
import importlib
from array import array
from dataclasses import replace

import pytest

from document_kv_cache.admission import AdmissionQueue
from document_kv_cache.cache import CacheTier, ChunkCache
from document_kv_cache.engine_adapters import (
    ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE,
    ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION,
    ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE,
    ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION,
    EngineAdapterRequest,
    EngineAdapterSpec,
    EngineKVConnectorActions,
    EngineKVConnectorProbeResult,
    EngineKVInjectionPlan,
    EngineKVReservationAction,
    EngineKVSegmentBinding,
    PayloadMode,
    ServingBackend,
    build_engine_adapter_request,
    build_engine_kv_connector_actions,
    build_engine_kv_injection_plan,
    engine_kv_connector_actions_from_record,
    engine_kv_connector_actions_to_record,
    engine_kv_connector_probe_result_to_record,
    engine_adapter_request_to_record,
    payload_mode_for,
    probe_engine_kv_connector_actions,
    read_engine_adapter_request_json,
    sglang_adapter_spec,
    split_engine_adapter_payload,
    validate_engine_adapter_request_record,
    validate_engine_kv_connector_actions_record,
    validate_engine_kv_connector_probe_record,
    validate_engine_kv_connector_actions,
    view_engine_adapter_payload,
    vllm_adapter_spec,
    write_engine_adapter_request_json,
)
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment, KVStorageLayout
from document_kv_cache.kvpack import PackChunk, write_kvpack
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.materializer import KVMaterializer
from document_kv_cache.models import CacheGenerationMethod, DocumentChunkType, DocumentKVRequest, KVCacheKey
from document_kv_cache.planner import CachePlanner
from document_kv_cache.service import DocumentKVService
from document_kv_cache.storage import DiskRangeReader


TEST_BYTES_PER_TOKEN = 36 * 8 * 128 * 2
STATIC_TOKEN_COUNT = 2
CHUNK_TOKEN_COUNT = 3
STATIC_PAYLOAD = b"s" * (STATIC_TOKEN_COUNT * TEST_BYTES_PER_TOKEN)
CHUNK_PAYLOAD = b"c" * (CHUNK_TOKEN_COUNT * TEST_BYTES_PER_TOKEN)


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
        layout_version="qwen3-v1",
        dtype="int8",
        num_layers=36,
        block_size=16,
        bytes_per_token=TEST_BYTES_PER_TOKEN,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        kv_stride_bytes=128,
        shares_kv_storage=True,
    )


def segment_binding(
    *,
    token_start: int = 0,
    token_count: int = 1,
    byte_start: int = 0,
    byte_length: int = TEST_BYTES_PER_TOKEN,
    first_block_index: int = 0,
    last_block_index_exclusive: int = 1,
) -> EngineKVSegmentBinding:
    return EngineKVSegmentBinding(
        document_id="doc-a",
        chunk_type="document_chunk",
        chunk_id="section-1",
        token_start=token_start,
        token_count=token_count,
        token_end=token_start + token_count,
        byte_start=byte_start,
        byte_length=byte_length,
        byte_end=byte_start + byte_length,
        first_block_index=first_block_index,
        last_block_index_exclusive=last_block_index_exclusive,
    )


def injection_plan_kwargs(**overrides):
    values = {
        "backend": ServingBackend.VLLM,
        "request_id": "req-1",
        "handle_uri": "document-kv://req-1",
        "connector_package": "vllm",
        "kv_injection_method": "native-test",
        "payload_mode": PayloadMode.MERGED,
        "payload_source_uri": None,
        "layout": layout(),
        "cache_method": "vanilla_prefill",
        "adapter_ids": (),
        "total_tokens": 1,
        "total_bytes": TEST_BYTES_PER_TOKEN,
        "total_blocks": 1,
        "estimated_gpu_bytes": TEST_BYTES_PER_TOKEN,
        "segments": (segment_binding(),),
        "metadata": {},
    }
    values.update(overrides)
    return values


def request() -> DocumentKVRequest:
    return DocumentKVRequest(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_chunks={"doc-a": ["section-1"]},
    )


def service(tmp_path) -> DocumentKVService:
    refs = write_kvpack(
        tmp_path / "engine-adapters.kvpack",
        [
            PackChunk(
                key(DocumentChunkType.DOCUMENT_STATIC, "static"),
                STATIC_PAYLOAD,
                STATIC_TOKEN_COUNT,
                "int8",
                "qwen3-v1",
                storage_layout=KVStorageLayout.SHARED_KEY_VALUE,
            ),
            PackChunk(
                key(DocumentChunkType.DOCUMENT_CHUNK, "section-1"),
                CHUNK_PAYLOAD,
                CHUNK_TOKEN_COUNT,
                "int8",
                "qwen3-v1",
                storage_layout=KVStorageLayout.SHARED_KEY_VALUE,
            ),
        ],
        align_bytes=1,
    )
    return DocumentKVService(
        planner=CachePlanner(InMemoryManifestStore(refs)),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=1024), reader=DiskRangeReader()),
        admission_queue=AdmissionQueue(max_pending_gpu_bytes=4096),
        kv_gpu_bytes_per_payload_byte=2.0,
    )


def test_engine_adapters_document_module_owns_public_api():
    public_module = importlib.import_module("document_kv_cache.engine_adapters")
    legacy_module = importlib.import_module("restaurant_kv_serving.engine_adapters")

    assert public_module.EngineAdapterRequest.__module__ == "document_kv_cache.engine_adapters"
    assert public_module.EngineAdapterSpec.__module__ == "document_kv_cache.engine_adapters"
    assert public_module.EngineKVInjectionPlan.__module__ == "document_kv_cache.engine_adapters"
    assert public_module.build_engine_adapter_request.__module__ == "document_kv_cache.engine_adapters"
    assert public_module.vllm_adapter_spec.__module__ == "document_kv_cache.engine_adapters"
    assert public_module.__all__ == [
        "EngineAdapterRequest",
        "EngineAdapterSpec",
        "ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE",
        "ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION",
        "ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE",
        "ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION",
        "EngineKVBlockManagerProbe",
        "EngineKVBindAction",
        "EngineKVConnectorActions",
        "EngineKVConnectorProbeResult",
        "EngineKVInjectionPlan",
        "EngineKVReleaseAction",
        "EngineKVReservationAction",
        "EngineKVSegmentCopyAction",
        "EngineKVSegmentBinding",
        "PayloadMode",
        "ServingBackend",
        "build_engine_adapter_request",
        "build_engine_kv_connector_actions",
        "build_engine_kv_injection_plan",
        "engine_kv_connector_actions_from_record",
        "engine_kv_connector_actions_to_record",
        "engine_kv_connector_probe_result_to_record",
        "engine_adapter_request_to_record",
        "payload_mode_for",
        "probe_engine_kv_connector_actions",
        "read_engine_adapter_request_json",
        "sglang_adapter_spec",
        "split_engine_adapter_payload",
        "validate_engine_adapter_request_record",
        "validate_engine_kv_connector_actions_record",
        "validate_engine_kv_connector_probe_record",
        "validate_engine_kv_connector_actions",
        "view_engine_adapter_payload",
        "vllm_adapter_spec",
        "write_engine_adapter_request_json",
    ]
    assert legacy_module.EngineAdapterRequest is public_module.EngineAdapterRequest
    assert legacy_module.build_engine_adapter_request is public_module.build_engine_adapter_request
    assert legacy_module.ENGINE_ADAPTER_HANDOFF_RECORD_TYPE is public_module.ENGINE_ADAPTER_HANDOFF_RECORD_TYPE
    assert legacy_module.IN_PROCESS_PAYLOAD_SOURCE is public_module.IN_PROCESS_PAYLOAD_SOURCE
    assert not hasattr(legacy_module, "__all__")


def test_legacy_engine_adapters_preserves_old_non_curated_import_surface():
    from restaurant_kv_serving.engine_adapters import ENGINE_ADAPTER_HANDOFF_RECORD_TYPE
    from restaurant_kv_serving.engine_adapters import ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION
    from restaurant_kv_serving.engine_adapters import EXTERNAL_URI_PAYLOAD_SOURCE
    from restaurant_kv_serving.engine_adapters import IN_PROCESS_PAYLOAD_SOURCE

    namespace: dict[str, object] = {}

    exec("from restaurant_kv_serving.engine_adapters import *", namespace)

    assert ENGINE_ADAPTER_HANDOFF_RECORD_TYPE == "document_kv.engine_adapter_request.v1"
    assert ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION == 2
    assert IN_PROCESS_PAYLOAD_SOURCE == "in_process"
    assert EXTERNAL_URI_PAYLOAD_SOURCE == "external_uri"
    assert namespace["ENGINE_ADAPTER_HANDOFF_RECORD_TYPE"] == ENGINE_ADAPTER_HANDOFF_RECORD_TYPE
    assert namespace["ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION"] == ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION
    assert namespace["IN_PROCESS_PAYLOAD_SOURCE"] == IN_PROCESS_PAYLOAD_SOURCE
    assert namespace["EXTERNAL_URI_PAYLOAD_SOURCE"] == EXTERNAL_URI_PAYLOAD_SOURCE
    assert "__all__" not in namespace


def test_vllm_adapter_request_carries_engine_handoff_metadata(tmp_path):
    ready = service(tmp_path).prepare_for_engine(
        request(),
        layout=layout(),
        metadata={"tenant": "qa"},
        cache_method=CacheGenerationMethod.KV_PACKET,
        adapter_ids=("selection-lora",),
        segmented=True,
    )

    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())

    assert adapter_request.backend == ServingBackend.VLLM
    assert adapter_request.request_id == "req-1"
    assert adapter_request.handle_uri == "document-kv://req-1"
    assert adapter_request.payload_mode == PayloadMode.SEGMENTED
    assert adapter_request.connector_package == "vllm"
    assert adapter_request.metadata["tenant"] == "qa"
    assert adapter_request.metadata["engine.backend"] == "vllm"
    assert adapter_request.metadata["document_kv.cache_method"] == "kv_packet"
    assert adapter_request.metadata["document_kv.payload_mode"] == "segmented"
    assert adapter_request.metadata["document_kv.total_tokens"] == "5"
    assert "schedule_decode_with_engine" in adapter_request.required_steps


def test_adapter_request_record_serializes_engine_handoff_without_payload(tmp_path):
    ready = service(tmp_path).prepare_for_engine(
        request(),
        layout=layout(),
        metadata={"tenant": "qa"},
        adapter_ids=("selection-lora",),
        segmented=True,
    )
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())

    record = engine_adapter_request_to_record(adapter_request)

    assert "payload" not in record
    assert record["record_type"] == "document_kv.engine_adapter_request.v1"
    assert record["schema_version"] == 2
    assert record["backend"] == "vllm"
    assert record["request_id"] == "req-1"
    assert record["handle_uri"] == "document-kv://req-1"
    assert record["payload_mode"] == "segmented"
    assert record["estimated_gpu_bytes"] == ready.estimated_gpu_bytes
    assert record["metadata"]["document_kv.total_bytes"] == str(ready.handle.total_bytes)
    assert record["required_steps"] == list(adapter_request.required_steps)
    assert record["payload_source"] == {
        "availability": "in_process",
        "uri": None,
        "format": "document_kv.materialized_payload.v1",
        "payload_mode": "segmented",
        "total_bytes": ready.handle.total_bytes,
        "segment_count": 2,
    }
    assert record["handle"]["adapter_ids"] == ["selection-lora"]
    assert record["handle"]["cache_method"] == "vanilla_prefill"
    assert record["handle"]["metadata"] == {"tenant": "qa"}
    assert record["handle"]["layout"] == {
        "model_id": "qwen3:4b-instruct",
        "lora_id": "base",
        "layout_version": "qwen3-v1",
        "dtype": "int8",
        "num_layers": 36,
        "block_size": 16,
        "bytes_per_token": TEST_BYTES_PER_TOKEN,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "kv_stride_bytes": 128,
        "shares_kv_storage": True,
        "storage_layout": "shared_key_value",
        "attention_mechanism": "gqa",
        "query_heads_per_kv_head": 4,
    }
    assert record["handle"]["segments"] == [
        {
            "document_id": "doc-a",
            "chunk_type": "document_static",
            "chunk_id": "static",
            "cache_tier": "cold_storage",
            "token_start": 0,
            "token_count": STATIC_TOKEN_COUNT,
            "token_end": STATIC_TOKEN_COUNT,
            "byte_start": 0,
            "byte_length": len(STATIC_PAYLOAD),
            "byte_end": len(STATIC_PAYLOAD),
            "content_hash": key(DocumentChunkType.DOCUMENT_STATIC, "static").content_hash,
        },
        {
            "document_id": "doc-a",
            "chunk_type": "document_chunk",
            "chunk_id": "section-1",
            "cache_tier": "cold_storage",
            "token_start": STATIC_TOKEN_COUNT,
            "token_count": CHUNK_TOKEN_COUNT,
            "token_end": STATIC_TOKEN_COUNT + CHUNK_TOKEN_COUNT,
            "byte_start": len(STATIC_PAYLOAD),
            "byte_length": len(CHUNK_PAYLOAD),
            "byte_end": len(STATIC_PAYLOAD) + len(CHUNK_PAYLOAD),
            "content_hash": key(DocumentChunkType.DOCUMENT_CHUNK, "section-1").content_hash,
        },
    ]


def test_write_engine_adapter_request_json_requires_external_payload_source_by_default(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())

    with pytest.raises(ValueError, match="adapter-readable payload_uri"):
        write_engine_adapter_request_json(adapter_request, tmp_path / "handoff.json")


def test_write_engine_adapter_request_json_writes_stable_record_with_payload_uri(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    payload_uri = f"disk:{tmp_path / 'req-1.kv'}"

    output_path = write_engine_adapter_request_json(
        adapter_request,
        tmp_path / "handoff.json",
        payload_uri=payload_uri,
    )

    assert output_path == tmp_path / "handoff.json"
    assert output_path.read_text(encoding="utf-8").endswith("\n")
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record == engine_adapter_request_to_record(adapter_request, payload_uri=payload_uri)
    assert record["payload_source"]["availability"] == "external_uri"
    assert record["payload_source"]["uri"] == payload_uri


def test_write_engine_adapter_request_json_accepts_external_handle_uri(tmp_path):
    handle_uri = f"file:{tmp_path / 'req-1.kv'}"
    ready = service(tmp_path).prepare_for_engine(
        request(),
        layout=layout(),
        handle_uri=handle_uri,
    )
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())

    output_path = write_engine_adapter_request_json(adapter_request, tmp_path / "handoff.json")

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["handle_uri"] == handle_uri
    assert record["payload_source"]["availability"] == "external_uri"
    assert record["payload_source"]["uri"] == handle_uri


def test_write_engine_adapter_request_json_rejects_logical_engine_handle_uri(tmp_path):
    ready = service(tmp_path).prepare_for_engine(
        request(),
        layout=layout(),
        handle_uri="sglang://req-1",
    )
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())

    with pytest.raises(ValueError, match="adapter-readable payload_uri"):
        write_engine_adapter_request_json(adapter_request, tmp_path / "handoff.json")

    record = engine_adapter_request_to_record(adapter_request)
    assert record["payload_source"]["availability"] == "in_process"
    assert record["payload_source"]["uri"] is None


def test_engine_adapter_request_record_rejects_relative_payload_uri(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())

    with pytest.raises(ValueError, match="absolute path or adapter-readable URI"):
        engine_adapter_request_to_record(adapter_request, payload_uri="relative-payload.bin")

    with pytest.raises(ValueError, match="absolute path or adapter-readable URI"):
        engine_adapter_request_to_record(adapter_request, payload_uri="vllm://payload-1")

    with pytest.raises(ValueError, match="absolute path or adapter-readable URI"):
        engine_adapter_request_to_record(adapter_request, payload_uri="disk:")

    with pytest.raises(ValueError, match="absolute path or adapter-readable URI"):
        engine_adapter_request_to_record(adapter_request, payload_uri="file:relative-payload.bin")

    with pytest.raises(ValueError, match="absolute path or adapter-readable URI"):
        engine_adapter_request_to_record(adapter_request, payload_uri="s3:")


def test_read_engine_adapter_request_json_validates_backend_and_payload_source(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    handoff_path = write_engine_adapter_request_json(
        adapter_request,
        tmp_path / "handoff.json",
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )

    record = read_engine_adapter_request_json(handoff_path, expected_backend=ServingBackend.SGLANG)

    assert record["backend"] == "sglang"
    assert record["payload_source"]["availability"] == "external_uri"
    assert read_engine_adapter_request_json(
        f"disk:{handoff_path}",
        expected_backend=ServingBackend.SGLANG,
    )["request_id"] == "req-1"
    with pytest.raises(ValueError, match="does not match expected_backend"):
        read_engine_adapter_request_json(handoff_path, expected_backend=ServingBackend.VLLM)


def test_validate_engine_adapter_request_record_uses_canonical_reserved_metadata_for_mixed_case_fields(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    mixed_case_record = {
        **record,
        "backend": "VLLM",
        "payload_mode": "MERGED",
        "payload_source": {
            **record["payload_source"],
            "payload_mode": "MERGED",
        },
    }

    validate_engine_adapter_request_record(mixed_case_record, expected_backend="vLLM")
    injection_plan = build_engine_kv_injection_plan(mixed_case_record, expected_backend="VLLM")

    assert injection_plan.backend is ServingBackend.VLLM
    assert injection_plan.payload_mode is PayloadMode.MERGED
    assert injection_plan.metadata["engine.backend"] == "vllm"
    assert injection_plan.metadata["document_kv.payload_mode"] == "merged"

    bad_metadata_record = {
        **mixed_case_record,
        "metadata": {
            **mixed_case_record["metadata"],
            "engine.backend": "VLLM",
        },
    }
    with pytest.raises(ValueError, match="Reserved metadata"):
        validate_engine_adapter_request_record(bad_metadata_record, expected_backend="vLLM")


def test_validate_engine_adapter_request_record_allows_debug_in_process_records(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    record = engine_adapter_request_to_record(adapter_request)

    with pytest.raises(ValueError, match="requires an external payload source"):
        validate_engine_adapter_request_record(record)

    validate_engine_adapter_request_record(record, require_external_payload_uri=False)


def test_validate_engine_adapter_request_record_rejects_schema_and_segment_mismatch(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )

    wrong_schema = dict(record)
    wrong_schema["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        validate_engine_adapter_request_record(wrong_schema)

    wrong_segment_count = dict(record)
    wrong_segment_count["payload_source"] = {
        **record["payload_source"],
        "segment_count": 1,
    }
    with pytest.raises(ValueError, match="segment_count"):
        validate_engine_adapter_request_record(wrong_segment_count)

    wrong_byte_end = dict(record)
    wrong_handle = {
        **record["handle"],
        "segments": [
            {**record["handle"]["segments"][0], "byte_end": record["handle"]["segments"][0]["byte_end"] + 1},
            record["handle"]["segments"][1],
        ],
    }
    wrong_byte_end["handle"] = wrong_handle
    with pytest.raises(ValueError, match="byte_end"):
        validate_engine_adapter_request_record(wrong_byte_end)

    missing_cache_tier = dict(record)
    missing_cache_tier["handle"] = {
        **record["handle"],
        "segments": [
            {key: value for key, value in record["handle"]["segments"][0].items() if key != "cache_tier"},
            record["handle"]["segments"][1],
        ],
    }
    with pytest.raises(TypeError, match="cache_tier"):
        validate_engine_adapter_request_record(missing_cache_tier)

    wrong_cache_tier = dict(record)
    wrong_cache_tier["handle"] = {
        **record["handle"],
        "segments": [
            {**record["handle"]["segments"][0], "cache_tier": "object_store"},
            record["handle"]["segments"][1],
        ],
    }
    with pytest.raises(ValueError, match="segment.cache_tier"):
        validate_engine_adapter_request_record(wrong_cache_tier)


def test_validate_engine_adapter_request_record_rejects_layout_byte_mismatch(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    shortened_first_byte_length = TEST_BYTES_PER_TOKEN
    new_total_bytes = shortened_first_byte_length + len(CHUNK_PAYLOAD)
    wrong_segments = [
        {
            **record["handle"]["segments"][0],
            "byte_length": shortened_first_byte_length,
            "byte_end": shortened_first_byte_length,
        },
        {
            **record["handle"]["segments"][1],
            "byte_start": shortened_first_byte_length,
            "byte_end": new_total_bytes,
        },
    ]
    wrong_record = {
        **record,
        "payload_source": {
            **record["payload_source"],
            "total_bytes": new_total_bytes,
        },
        "handle": {
            **record["handle"],
            "total_bytes": new_total_bytes,
            "segments": wrong_segments,
        },
    }

    with pytest.raises(ValueError, match="token_count \\* bytes_per_token"):
        validate_engine_adapter_request_record(wrong_record)


def test_validate_engine_adapter_request_record_rejects_missing_or_conflicting_storage_layout(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    missing_storage_layout = {
        **record,
        "handle": {
            **record["handle"],
            "layout": {key: value for key, value in record["handle"]["layout"].items() if key != "storage_layout"},
        },
    }
    with pytest.raises(TypeError, match="storage_layout"):
        validate_engine_adapter_request_record(missing_storage_layout)

    conflicting_storage_layout = {
        **record,
        "handle": {
            **record["handle"],
            "layout": {**record["handle"]["layout"], "storage_layout": "separate_key_value"},
        },
    }
    with pytest.raises(ValueError, match="shares_kv_storage"):
        validate_engine_adapter_request_record(conflicting_storage_layout)


def test_validate_engine_adapter_request_record_rejects_empty_segments(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    empty_first_segment = {
        **record["handle"]["segments"][0],
        "token_count": 0,
        "token_end": 0,
        "byte_length": 0,
        "byte_end": 0,
    }
    shifted_second_segment = {
        **record["handle"]["segments"][1],
        "token_start": 0,
        "token_end": CHUNK_TOKEN_COUNT,
        "byte_start": 0,
        "byte_end": len(CHUNK_PAYLOAD),
    }
    empty_segment_record = {
        **record,
        "payload_source": {
            **record["payload_source"],
            "total_bytes": len(CHUNK_PAYLOAD),
        },
        "handle": {
            **record["handle"],
            "total_tokens": CHUNK_TOKEN_COUNT,
            "total_bytes": len(CHUNK_PAYLOAD),
            "segments": [empty_first_segment, shifted_second_segment],
        },
    }

    with pytest.raises(ValueError, match="token_count must be positive"):
        validate_engine_adapter_request_record(empty_segment_record)


def test_validate_engine_adapter_request_record_rejects_reserved_metadata_mismatch(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    wrong_backend_metadata = {
        **record,
        "metadata": {
            **record["metadata"],
            "engine.backend": "sglang",
        },
    }

    with pytest.raises(ValueError, match="Reserved metadata"):
        validate_engine_adapter_request_record(wrong_backend_metadata)

    wrong_token_metadata = {
        **record,
        "metadata": {
            **record["metadata"],
            "document_kv.total_tokens": "999",
        },
    }
    with pytest.raises(ValueError, match="document_kv.total_tokens"):
        validate_engine_adapter_request_record(wrong_token_metadata)


def test_validate_engine_adapter_request_record_rejects_reserved_handle_metadata(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    wrong_handle_metadata = {
        **record,
        "handle": {
            **record["handle"],
            "metadata": {
                **record["handle"]["metadata"],
                "document_kv.total_tokens": "not-the-real-value",
            },
        },
    }

    with pytest.raises(ValueError, match="reserved adapter keys"):
        validate_engine_adapter_request_record(wrong_handle_metadata)


def test_validate_engine_adapter_request_record_requires_adapter_contract_fields(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )

    for field in ("connector_package", "kv_injection_method", "payload_contract"):
        missing_field = dict(record)
        missing_field.pop(field)
        with pytest.raises((TypeError, ValueError), match=field):
            validate_engine_adapter_request_record(missing_field)

        empty_field = {**record, field: ""}
        with pytest.raises(ValueError, match=field):
            validate_engine_adapter_request_record(empty_field)

    empty_steps = {**record, "required_steps": []}
    with pytest.raises(ValueError, match="required_steps"):
        validate_engine_adapter_request_record(empty_steps)


def test_split_engine_adapter_payload_returns_segmented_bytes(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    payload = b"".join(ready.payload)

    split_payload = split_engine_adapter_payload(record, payload)

    assert split_payload == ready.payload
    injection_plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
    actions = build_engine_kv_connector_actions(injection_plan, split_payload)
    assert [segment.cache_tier for segment in injection_plan.segments] == [
        CacheTier.COLD_STORAGE,
        CacheTier.COLD_STORAGE,
    ]
    assert [copy.cache_tier for copy in actions.copies] == [CacheTier.COLD_STORAGE, CacheTier.COLD_STORAGE]
    with pytest.raises(ValueError, match="payload length"):
        split_engine_adapter_payload(record, payload[:-1])


def test_view_engine_adapter_payload_returns_segmented_memoryviews_without_copy(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    payload = b"".join(ready.payload)

    payload_views = view_engine_adapter_payload(record, payload)

    assert isinstance(payload_views, tuple)
    assert [view.obj for view in payload_views] == [payload, payload]
    assert tuple(view.tobytes() for view in payload_views) == ready.payload


def test_view_engine_adapter_payload_returns_merged_memoryview_without_copy(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )

    payload_view = view_engine_adapter_payload(record, ready.payload)

    assert isinstance(payload_view, memoryview)
    assert payload_view.obj is ready.payload
    assert payload_view.nbytes == len(ready.payload)


def test_view_engine_adapter_payload_casts_non_byte_memoryview_to_byte_view(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    two_byte_items = array("H", [0]) * (ready.handle.total_bytes // 2)

    payload_view = view_engine_adapter_payload(record, memoryview(two_byte_items))

    assert isinstance(payload_view, memoryview)
    assert payload_view.itemsize == 1
    assert payload_view.ndim == 1
    assert payload_view.nbytes == ready.handle.total_bytes


def test_split_engine_adapter_payload_keeps_merged_bytes(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )

    assert split_engine_adapter_payload(record, ready.payload) == ready.payload


def test_build_engine_kv_injection_plan_allows_overlapping_blocks_for_unaligned_segments(tmp_path):
    plan_layout = KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version="qwen3-v1",
        dtype="int8",
        num_layers=36,
        block_size=4,
        bytes_per_token=TEST_BYTES_PER_TOKEN,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        kv_stride_bytes=128,
        shares_kv_storage=True,
    )
    ready = service(tmp_path).prepare_for_engine(request(), layout=plan_layout, segmented=True)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )

    injection_plan = build_engine_kv_injection_plan(record, expected_backend="vllm")

    assert injection_plan.total_blocks == 2
    assert [
        (segment.chunk_id, segment.first_block_index, segment.last_block_index_exclusive, segment.block_count)
        for segment in injection_plan.segments
    ] == [
        ("static", 0, 1, 1),
        ("section-1", 0, 2, 2),
    ]


def test_build_engine_kv_injection_plan_maps_segments_to_native_blocks(tmp_path):
    plan_layout = KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version="qwen3-v1",
        dtype="int8",
        num_layers=36,
        block_size=2,
        bytes_per_token=TEST_BYTES_PER_TOKEN,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        kv_stride_bytes=128,
        shares_kv_storage=True,
    )
    ready = service(tmp_path).prepare_for_engine(
        request(),
        layout=plan_layout,
        adapter_ids=("selection-lora",),
        segmented=True,
    )
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )

    injection_plan = build_engine_kv_injection_plan(record, expected_backend=ServingBackend.VLLM)

    assert injection_plan.backend == ServingBackend.VLLM
    assert injection_plan.request_id == "req-1"
    assert injection_plan.payload_mode == PayloadMode.SEGMENTED
    assert injection_plan.payload_source_uri == f"disk:{tmp_path / 'req-1.kv'}"
    assert injection_plan.layout.block_size == 2
    assert injection_plan.total_tokens == STATIC_TOKEN_COUNT + CHUNK_TOKEN_COUNT
    assert injection_plan.total_bytes == len(STATIC_PAYLOAD) + len(CHUNK_PAYLOAD)
    assert injection_plan.total_blocks == 3
    assert injection_plan.adapter_ids == ("selection-lora",)
    assert injection_plan.metadata["engine.backend"] == "vllm"
    assert [
        (
            segment.chunk_id,
            segment.token_start,
            segment.token_end,
            segment.byte_start,
            segment.byte_end,
            segment.first_block_index,
            segment.last_block_index_exclusive,
            segment.block_count,
        )
        for segment in injection_plan.segments
    ] == [
        ("static", 0, 2, 0, len(STATIC_PAYLOAD), 0, 1, 1),
        (
            "section-1",
            2,
            5,
            len(STATIC_PAYLOAD),
            len(STATIC_PAYLOAD) + len(CHUNK_PAYLOAD),
            1,
            3,
            2,
        ),
    ]


def test_engine_kv_connector_probe_result_record_validates_native_probe_evidence():
    result = EngineKVConnectorProbeResult(
        backend=ServingBackend.VLLM,
        request_id="req-1",
        total_blocks=1,
        copied_segments=2,
        copied_tokens=5,
        copied_bytes=5 * TEST_BYTES_PER_TOKEN,
        bound=True,
        released=True,
        model_id="qwen3:4b-instruct",
        layout_version="qwen3-v1",
        layout=layout(),
        payload_mode=PayloadMode.MERGED,
        connector_package="vllm",
        engine_version="vllm-test",
    )

    record = engine_kv_connector_probe_result_to_record(result)

    assert record == {
        "record_type": ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE,
        "schema_version": ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION,
        "backend": "vllm",
        "request_id": "req-1",
        "total_blocks": 1,
        "copied_segments": 2,
        "copied_tokens": 5,
        "copied_bytes": 5 * TEST_BYTES_PER_TOKEN,
        "bound": True,
        "released": True,
        "model_id": "qwen3:4b-instruct",
        "layout_version": "qwen3-v1",
        "layout": {
            "model_id": "qwen3:4b-instruct",
            "lora_id": "base",
            "layout_version": "qwen3-v1",
            "dtype": "int8",
            "num_layers": 36,
            "block_size": 16,
            "bytes_per_token": TEST_BYTES_PER_TOKEN,
            "num_query_heads": 32,
            "num_kv_heads": 8,
            "head_size": 128,
            "kv_stride_bytes": 128,
            "shares_kv_storage": True,
            "storage_layout": "shared_key_value",
            "attention_mechanism": "gqa",
            "query_heads_per_kv_head": 4,
        },
        "payload_mode": "merged",
        "connector_package": "vllm",
        "engine_version": "vllm-test",
        "native_probe": True,
        "metadata": {},
    }
    validate_engine_kv_connector_probe_record(record, expected_backend="vllm")
    with pytest.raises(ValueError, match="expected"):
        validate_engine_kv_connector_probe_record(record, expected_backend="sglang")
    invalid = {**record, "bound": False}
    with pytest.raises(ValueError, match="bind"):
        validate_engine_kv_connector_probe_record(invalid)
    invalid_counter = {**record, "total_blocks": True}
    with pytest.raises(ValueError, match="positive integer"):
        validate_engine_kv_connector_probe_record(invalid_counter)
    impossible_segments = {**record, "copied_segments": 6}
    with pytest.raises(ValueError, match="copied_segments"):
        validate_engine_kv_connector_probe_record(impossible_segments)
    invalid_metadata = {**record, "metadata": {"engine.version": 1}}
    with pytest.raises(TypeError, match="metadata"):
        validate_engine_kv_connector_probe_record(invalid_metadata)
    non_native_runtime = {**record, "metadata": {"vllm_kv_injection.native_runtime": "false"}}
    with pytest.raises(ValueError, match="non-native/debug"):
        validate_engine_kv_connector_probe_record(non_native_runtime)
    debug_probe_kind = {**record, "metadata": {"sglang_kv_injection.probe_kind": "debug_in_memory"}}
    with pytest.raises(ValueError, match="non-native/debug"):
        validate_engine_kv_connector_probe_record(debug_probe_kind)
    debug_probe_name = {**record, "metadata": {"vllm_kv_injection.probe": "in_memory_debug"}}
    with pytest.raises(ValueError, match="non-native/debug"):
        validate_engine_kv_connector_probe_record(debug_probe_name)
    non_native_probe_name = {**record, "metadata": {"vllm_kv_injection.probe": "non_native_debug"}}
    with pytest.raises(ValueError, match="non-native/debug"):
        validate_engine_kv_connector_probe_record(non_native_probe_name)
    native_runtime_metadata = {**record, "metadata": {"vllm_kv_injection.native_runtime": "true"}}
    validate_engine_kv_connector_probe_record(native_runtime_metadata)
    invalid_layout = {**record, "layout": {**record["layout"], "num_kv_heads": 16}}
    with pytest.raises(ValueError, match="bytes_per_token|layout"):
        validate_engine_kv_connector_probe_record(invalid_layout)
    invalid_storage_layout = {**record, "layout": {**record["layout"], "storage_layout": "separate_key_value"}}
    with pytest.raises(ValueError, match="shares_kv_storage"):
        validate_engine_kv_connector_probe_record(invalid_storage_layout)
    invalid_schema = {**record, "schema_version": 1}
    with pytest.raises(ValueError, match="schema_version"):
        validate_engine_kv_connector_probe_record(invalid_schema)


def test_build_engine_kv_injection_plan_can_use_debug_in_process_record(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    record = engine_adapter_request_to_record(adapter_request)

    injection_plan = build_engine_kv_injection_plan(
        record,
        expected_backend="sglang",
        require_external_payload_uri=False,
    )

    assert injection_plan.backend == ServingBackend.SGLANG
    assert injection_plan.payload_source_uri is None
    assert injection_plan.payload_mode == PayloadMode.MERGED


def test_build_engine_kv_connector_actions_describe_segmented_native_handoff(tmp_path):
    plan_layout = KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version="qwen3-v1",
        dtype="int8",
        num_layers=36,
        block_size=2,
        bytes_per_token=TEST_BYTES_PER_TOKEN,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        kv_stride_bytes=128,
        shares_kv_storage=True,
    )
    ready = service(tmp_path).prepare_for_engine(
        request(),
        layout=plan_layout,
        adapter_ids=("selection-lora",),
        segmented=True,
    )
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    payload_or_segments = view_engine_adapter_payload(record, b"".join(ready.payload))
    injection_plan = build_engine_kv_injection_plan(record, expected_backend="vllm")

    actions = build_engine_kv_connector_actions(injection_plan, payload_or_segments)
    fake_connector = FakeNativeBlockManagerProbe()
    result = probe_engine_kv_connector_actions(actions, payload_or_segments, fake_connector)

    assert actions.reservation.total_blocks == 3
    assert actions.reservation.adapter_ids == ("selection-lora",)
    assert result == EngineKVConnectorProbeResult(
        backend=ServingBackend.VLLM,
        request_id="req-1",
        total_blocks=3,
        copied_segments=2,
        copied_tokens=STATIC_TOKEN_COUNT + CHUNK_TOKEN_COUNT,
        copied_bytes=len(STATIC_PAYLOAD) + len(CHUNK_PAYLOAD),
        bound=True,
        released=True,
        model_id="qwen3:4b-instruct",
        layout_version="qwen3-v1",
        layout=plan_layout,
        payload_mode=PayloadMode.SEGMENTED,
        connector_package="vllm",
        engine_version="unknown",
        native_probe=True,
        metadata={},
    )
    assert [
        (
            copy.payload_index,
            copy.chunk_id,
            copy.source_byte_start,
            copy.source_byte_end,
            copy.global_byte_start,
            copy.global_byte_end,
            copy.token_start,
            copy.token_end,
            copy.first_block_index,
            copy.last_block_index_exclusive,
            copy.block_count,
            hasattr(copy, "payload"),
        )
        for copy in actions.copies
    ] == [
        (0, "static", 0, len(STATIC_PAYLOAD), 0, len(STATIC_PAYLOAD), 0, 2, 0, 1, 1, False),
        (
            1,
            "section-1",
            0,
            len(CHUNK_PAYLOAD),
            len(STATIC_PAYLOAD),
            len(STATIC_PAYLOAD) + len(CHUNK_PAYLOAD),
            2,
            5,
            1,
            3,
            2,
            False,
        ),
    ]
    assert fake_connector.events == [
        ("reserve", "req-1", 3),
        ("copy", "req-1:blocks", "static", "memoryview", len(STATIC_PAYLOAD), 0, 1),
        ("copy", "req-1:blocks", "section-1", "memoryview", len(CHUNK_PAYLOAD), 1, 3),
        ("bind", "req-1", "document-kv://req-1", ("selection-lora",)),
        ("release", "req-1"),
    ]


def test_engine_kv_connector_actions_record_round_trips_segmented_handoff(tmp_path):
    plan_layout = KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version="qwen3-v1",
        dtype="int8",
        num_layers=36,
        block_size=2,
        bytes_per_token=TEST_BYTES_PER_TOKEN,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        kv_stride_bytes=128,
        shares_kv_storage=True,
    )
    ready = service(tmp_path).prepare_for_engine(
        request(),
        layout=plan_layout,
        adapter_ids=("selection-lora",),
        segmented=True,
    )
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    payload_or_segments = view_engine_adapter_payload(record, b"".join(ready.payload))
    injection_plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
    actions = build_engine_kv_connector_actions(injection_plan, payload_or_segments)

    actions_record = engine_kv_connector_actions_to_record(actions)
    json_record = json.loads(json.dumps(actions_record))
    restored = engine_kv_connector_actions_from_record(json_record, expected_backend=ServingBackend.VLLM)

    assert actions_record["record_type"] == ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE
    assert actions_record["schema_version"] == ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION
    assert actions_record["backend"] == "vllm"
    assert actions_record["request_id"] == "req-1"
    assert actions_record["reservation"]["layout"]["storage_layout"] == "shared_key_value"
    assert actions_record["reservation"]["adapter_ids"] == ["selection-lora"]
    assert [copy["payload_index"] for copy in actions_record["copies"]] == [0, 1]
    assert actions_record["copies"][0]["source_byte_end"] == len(STATIC_PAYLOAD)
    assert actions_record["copies"][1]["source_byte_start"] == 0
    assert actions_record["bind"]["metadata"]["engine.connector_package"] == "vllm"
    assert restored.reservation == actions.reservation
    assert restored.copies == actions.copies
    assert restored.bind.request_id == actions.bind.request_id
    assert restored.bind.adapter_ids == actions.bind.adapter_ids
    assert dict(restored.bind.metadata) == dict(actions.bind.metadata)
    assert restored.release == actions.release
    validate_engine_kv_connector_actions_record(json_record, expected_backend="vllm")


def test_engine_kv_connector_actions_record_validation_rejects_stale_fields(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    payload_view = view_engine_adapter_payload(record, ready.payload)
    injection_plan = build_engine_kv_injection_plan(record, expected_backend="sglang")
    actions = build_engine_kv_connector_actions(injection_plan, payload_view)
    actions_record = engine_kv_connector_actions_to_record(actions)

    with pytest.raises(ValueError, match="expected"):
        validate_engine_kv_connector_actions_record(actions_record, expected_backend="vllm")

    wrong_source_end = {
        **actions_record,
        "copies": [
            {**actions_record["copies"][0], "source_byte_end": actions_record["copies"][0]["source_byte_end"] + 1},
            *actions_record["copies"][1:],
        ],
    }
    with pytest.raises(ValueError, match="source_byte_end"):
        engine_kv_connector_actions_from_record(wrong_source_end)

    missing_source_end = {
        **actions_record,
        "copies": [
            {
                key: value
                for key, value in actions_record["copies"][0].items()
                if key != "source_byte_end"
            },
            *actions_record["copies"][1:],
        ],
    }
    with pytest.raises(ValueError, match="source_byte_end"):
        engine_kv_connector_actions_from_record(missing_source_end)

    wrong_reservation_backend = {
        **actions_record,
        "reservation": {**actions_record["reservation"], "backend": "vllm"},
    }
    with pytest.raises(ValueError, match="reservation.backend"):
        engine_kv_connector_actions_from_record(wrong_reservation_backend)

    wrong_schema = {**actions_record, "schema_version": 0}
    with pytest.raises(ValueError, match="schema_version"):
        engine_kv_connector_actions_from_record(wrong_schema)

    unsupported_record = {**actions_record, "debug": {"accepted": False}}
    with pytest.raises(ValueError, match=r"connector actions record has unsupported keys: \['debug'\]"):
        engine_kv_connector_actions_from_record(unsupported_record)

    unsupported_reservation = {
        **actions_record,
        "reservation": {**actions_record["reservation"], "debug": {"accepted": False}},
    }
    with pytest.raises(ValueError, match=r"connector actions reservation has unsupported keys: \['debug'\]"):
        engine_kv_connector_actions_from_record(unsupported_reservation)

    unsupported_copy = {
        **actions_record,
        "copies": [
            {**actions_record["copies"][0], "debug": {"accepted": False}},
            *actions_record["copies"][1:],
        ],
    }
    with pytest.raises(ValueError, match=r"connector actions copies\[0\] has unsupported keys: \['debug'\]"):
        engine_kv_connector_actions_from_record(unsupported_copy)

    unsupported_bind = {
        **actions_record,
        "bind": {**actions_record["bind"], "debug": {"accepted": False}},
    }
    with pytest.raises(ValueError, match=r"connector actions bind has unsupported keys: \['debug'\]"):
        engine_kv_connector_actions_from_record(unsupported_bind)

    unsupported_release = {
        **actions_record,
        "release": {**actions_record["release"], "debug": {"accepted": False}},
    }
    with pytest.raises(ValueError, match=r"connector actions release has unsupported keys: \['debug'\]"):
        engine_kv_connector_actions_from_record(unsupported_release)


def test_build_engine_kv_connector_actions_describe_merged_payload_slices(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    injection_plan = build_engine_kv_injection_plan(record, expected_backend="sglang")

    payload_view = view_engine_adapter_payload(record, ready.payload)
    actions = build_engine_kv_connector_actions(injection_plan, payload_view)

    assert actions.reservation.backend == ServingBackend.SGLANG
    assert actions.reservation.total_blocks == 1
    assert actions.bind.cache_method == "vanilla_prefill"
    assert actions.bind.metadata["engine.backend"] == "sglang"
    assert [
        (
            copy.payload_index,
            copy.source_byte_start,
            copy.source_byte_end,
            bytes(payload_view[copy.source_byte_start : copy.source_byte_end]),
        )
        for copy in actions.copies
    ] == [
        (None, 0, len(STATIC_PAYLOAD), STATIC_PAYLOAD),
        (None, len(STATIC_PAYLOAD), len(STATIC_PAYLOAD) + len(CHUNK_PAYLOAD), CHUNK_PAYLOAD),
    ]


def test_build_engine_kv_connector_actions_rejects_wrong_payload_shape(tmp_path):
    segmented_ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    adapter_request = build_engine_adapter_request(segmented_ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    injection_plan = build_engine_kv_injection_plan(record, expected_backend="vllm")

    with pytest.raises(ValueError, match="payload mode"):
        build_engine_kv_connector_actions(injection_plan, b"".join(segmented_ready.payload))

    short_segments = (segmented_ready.payload[0][:-1], segmented_ready.payload[1])
    with pytest.raises(ValueError, match="byte length"):
        build_engine_kv_connector_actions(injection_plan, short_segments)

    with pytest.raises(TypeError, match="tuple of those"):
        build_engine_kv_connector_actions(injection_plan, ["bad"])  # type: ignore[arg-type]


def test_build_engine_kv_connector_actions_rejects_non_byte_memoryview(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    injection_plan = build_engine_kv_injection_plan(record, expected_backend="sglang")
    two_byte_items = array("H", [0]) * (ready.handle.total_bytes // 2)

    with pytest.raises(TypeError, match="byte-addressable"):
        build_engine_kv_connector_actions(injection_plan, memoryview(two_byte_items))


def test_validate_engine_kv_connector_actions_rejects_out_of_order_copies(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    payload_or_segments = view_engine_adapter_payload(record, b"".join(ready.payload))
    injection_plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
    actions = build_engine_kv_connector_actions(injection_plan, payload_or_segments)
    out_of_order_actions = EngineKVConnectorActions(
        reservation=actions.reservation,
        copies=tuple(reversed(actions.copies)),
        bind=actions.bind,
        release=actions.release,
    )

    with pytest.raises(ValueError, match="Non-contiguous token copy action"):
        validate_engine_kv_connector_actions(out_of_order_actions)


def test_validate_engine_kv_connector_actions_rejects_wrong_block_range(tmp_path):
    plan_layout = KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version="qwen3-v1",
        dtype="int8",
        num_layers=36,
        block_size=2,
        bytes_per_token=TEST_BYTES_PER_TOKEN,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        kv_stride_bytes=128,
        shares_kv_storage=True,
    )
    ready = service(tmp_path).prepare_for_engine(request(), layout=plan_layout, segmented=True)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    payload_or_segments = view_engine_adapter_payload(record, b"".join(ready.payload))
    injection_plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
    actions = build_engine_kv_connector_actions(injection_plan, payload_or_segments)
    wrong_block_actions = EngineKVConnectorActions(
        reservation=actions.reservation,
        copies=(
            actions.copies[0],
            replace(actions.copies[1], first_block_index=actions.copies[1].first_block_index - 1),
        ),
        bind=actions.bind,
        release=actions.release,
    )

    with pytest.raises(ValueError, match="first_block_index"):
        validate_engine_kv_connector_actions(wrong_block_actions)


def test_probe_engine_kv_connector_actions_rejects_bad_connector_metadata_before_probe(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    payload = view_engine_adapter_payload(record, ready.payload)
    injection_plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
    actions = build_engine_kv_connector_actions(injection_plan, payload)
    bad_actions = EngineKVConnectorActions(
        reservation=actions.reservation,
        copies=actions.copies,
        bind=replace(
            actions.bind,
            metadata={**actions.bind.metadata, "engine.connector_package": "custom_solver"},
        ),
        release=actions.release,
    )
    fake_connector = FakeNativeBlockManagerProbe()

    with pytest.raises(ValueError, match="connector_package must match backend"):
        probe_engine_kv_connector_actions(bad_actions, payload, fake_connector)

    assert fake_connector.events == []


def test_probe_engine_kv_connector_actions_rejects_segmented_payload_index_miss(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    payload_or_segments = view_engine_adapter_payload(record, b"".join(ready.payload))
    injection_plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
    actions = build_engine_kv_connector_actions(injection_plan, payload_or_segments)
    fake_connector = FakeNativeBlockManagerProbe()

    with pytest.raises(ValueError, match="payload_index is out of range"):
        probe_engine_kv_connector_actions(actions, payload_or_segments[:1], fake_connector)

    assert fake_connector.events == [
        ("reserve", "req-1", 1),
        ("copy", "req-1:blocks", "static", "memoryview", len(STATIC_PAYLOAD), 0, 1),
        ("release", "req-1"),
    ]


def test_probe_engine_kv_connector_actions_uses_in_process_segment_views_without_join(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    injection_plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
    actions = build_engine_kv_connector_actions(injection_plan, ready.payload)
    fake_connector = FakeNativeBlockManagerProbe()

    result = probe_engine_kv_connector_actions(actions, ready.payload, fake_connector)

    assert result.copied_bytes == len(STATIC_PAYLOAD) + len(CHUNK_PAYLOAD)
    assert fake_connector.payload_objects == [STATIC_PAYLOAD, CHUNK_PAYLOAD]


def test_engine_adapter_request_to_record_rejects_zero_length_segments():
    zero_handle = KVCacheHandle(
        request_id="req-empty",
        handle_uri="document-kv://req-empty",
        layout=layout(),
        segments=(
            KVSegment(
                document_id="doc-a",
                chunk_type="document_chunk",
                chunk_id="empty",
                token_start=0,
                token_count=0,
                byte_start=0,
                byte_length=0,
            ),
        ),
        total_tokens=0,
        total_bytes=0,
    )
    ready = EngineReadyRequest(handle=zero_handle, payload=b"", estimated_gpu_bytes=0)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())

    with pytest.raises(ValueError, match="token_count must be positive"):
        engine_adapter_request_to_record(adapter_request)


def test_build_engine_kv_injection_plan_rejects_backend_mismatch(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )

    with pytest.raises(ValueError, match="expected_backend"):
        build_engine_kv_injection_plan(record, expected_backend=ServingBackend.VLLM)


def test_sglang_adapter_request_accepts_merged_payload(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())

    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())

    assert payload_mode_for(ready) == PayloadMode.MERGED
    assert adapter_request.backend == ServingBackend.SGLANG
    assert adapter_request.metadata["engine.backend"] == "sglang"
    assert adapter_request.metadata["document_kv.cache_method"] == "vanilla_prefill"
    assert adapter_request.metadata["document_kv.payload_mode"] == "merged"


def test_engine_adapter_dataclasses_normalize_known_backend_strings(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    spec = EngineAdapterSpec(
        backend="vLLM",  # type: ignore[arg-type]
        connector_package="vllm",
        kv_injection_method="native-test",
        payload_contract="test adapter",
    )
    adapter_request = EngineAdapterRequest(
        backend="VLLM",  # type: ignore[arg-type]
        ready_request=ready,
        connector_package="vllm",
        kv_injection_method="native-test",
        payload_contract="test adapter",
        required_steps=("reserve", "bind"),
        metadata={},
    )
    reservation = EngineKVReservationAction(
        backend="SGLang",  # type: ignore[arg-type]
        request_id="req-1",
        total_blocks=1,
        total_tokens=1,
        estimated_gpu_bytes=TEST_BYTES_PER_TOKEN,
        layout=layout(),
        adapter_ids=(),
    )
    injection_plan = EngineKVInjectionPlan(
        backend="vLLM",  # type: ignore[arg-type]
        request_id="req-1",
        handle_uri="document-kv://req-1",
        connector_package="vllm",
        kv_injection_method="native-test",
        payload_mode="MERGED",  # type: ignore[arg-type]
        payload_source_uri=None,
        layout=layout(),
        cache_method="vanilla_prefill",
        adapter_ids=(),
        total_tokens=0,
        total_bytes=0,
        total_blocks=0,
        estimated_gpu_bytes=0,
        segments=(),
        metadata={},
    )
    probe_result = EngineKVConnectorProbeResult(
        backend="SGLang",  # type: ignore[arg-type]
        request_id="req-1",
        total_blocks=1,
        copied_segments=1,
        copied_tokens=1,
        copied_bytes=TEST_BYTES_PER_TOKEN,
        bound=True,
        released=True,
        model_id="qwen3:4b-instruct",
        layout_version="qwen3-v1",
        layout=layout(),
        payload_mode="MERGED",  # type: ignore[arg-type]
        connector_package="sglang",
        engine_version="sglang-test",
    )

    assert spec.backend is ServingBackend.VLLM
    assert adapter_request.backend is ServingBackend.VLLM
    assert reservation.backend is ServingBackend.SGLANG
    assert injection_plan.backend is ServingBackend.VLLM
    assert injection_plan.payload_mode is PayloadMode.MERGED
    assert probe_result.backend is ServingBackend.SGLANG
    assert probe_result.payload_mode is PayloadMode.MERGED


@pytest.mark.parametrize("estimated_gpu_bytes", [True, 1.0, "1", -1])
def test_engine_kv_injection_plan_rejects_invalid_estimated_gpu_bytes(estimated_gpu_bytes):
    with pytest.raises(ValueError, match="estimated_gpu_bytes must be a non-negative integer"):
        EngineKVInjectionPlan(
            backend=ServingBackend.VLLM,
            request_id="req-1",
            handle_uri="document-kv://req-1",
            connector_package="vllm",
            kv_injection_method="native-test",
            payload_mode=PayloadMode.MERGED,
            payload_source_uri=None,
            layout=layout(),
            cache_method="vanilla_prefill",
            adapter_ids=(),
            total_tokens=0,
            total_bytes=0,
            total_blocks=0,
            estimated_gpu_bytes=estimated_gpu_bytes,
            segments=(),
            metadata={},
        )


@pytest.mark.parametrize(
    ("field_name", "value", "error_match"),
    [
        ("request_id", "", "request_id"),
        ("request_id", 123, "request_id must be a non-empty string"),
        ("handle_uri", "", "handle_uri"),
        ("handle_uri", 123, "handle_uri must be a non-empty string"),
        ("kv_injection_method", "", "kv_injection_method"),
        ("kv_injection_method", 123, "kv_injection_method must be a non-empty string"),
        ("cache_method", "", "cache_method"),
        ("cache_method", 123, "cache_method must be a non-empty string"),
        ("adapter_ids", "lora", "adapter_ids must be a sequence"),
        ("adapter_ids", {"lora": "ignored"}, "adapter_ids must be a sequence"),
        ("adapter_ids", ("",), "adapter_ids"),
        ("segments", ("not-a-segment",), "segments entries"),
    ],
)
def test_engine_kv_injection_plan_rejects_invalid_identity_and_sequence_fields(
    field_name,
    value,
    error_match,
):
    with pytest.raises((TypeError, ValueError), match=error_match):
        EngineKVInjectionPlan(**injection_plan_kwargs(**{field_name: value}))


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({"total_tokens": True}, "total_tokens must be a non-negative integer"),
        ({"total_tokens": -1}, "total_tokens must be a non-negative integer"),
        ({"total_bytes": True}, "total_bytes must be a non-negative integer"),
        ({"total_bytes": TEST_BYTES_PER_TOKEN - 1}, "total_bytes does not match"),
        ({"total_blocks": True}, "total_blocks must be a non-negative integer"),
        ({"total_blocks": 2}, "total_blocks does not match"),
        (
            {"layout": replace(layout(), bytes_per_token=TEST_BYTES_PER_TOKEN - 1)},
            "bytes_per_token",
        ),
        ({"segments": ()}, "Segment bindings cover 0 tokens"),
    ],
)
def test_engine_kv_injection_plan_rejects_invalid_layout_totals(overrides, error_match):
    with pytest.raises(ValueError, match=error_match):
        EngineKVInjectionPlan(**injection_plan_kwargs(**overrides))


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({"document_id": ""}, "document_id"),
        ({"document_id": 123}, "document_id must be a non-empty string"),
        ({"chunk_type": ""}, "chunk_type"),
        ({"chunk_id": ""}, "chunk_id"),
        ({"token_start": 0.0}, "token_start must be a non-negative integer"),
        ({"token_start": False}, "token_start must be a non-negative integer"),
        ({"token_count": 0, "token_end": 0}, "token_count must be positive"),
        ({"byte_length": 0, "byte_end": 0}, "byte_length must be positive"),
        ({"token_end": 2}, "token_end does not match"),
        ({"byte_end": TEST_BYTES_PER_TOKEN + 1}, "byte_end does not match"),
        ({"first_block_index": False}, "first_block_index must be a non-negative integer"),
        ({"last_block_index_exclusive": 0}, "block range must be positive"),
        ({"content_hash": object()}, "content_hash must be a string"),
        ({"cache_tier": "gpu"}, "cache_tier"),
    ],
)
def test_engine_kv_segment_binding_rejects_invalid_public_fields(overrides, error_match):
    values = {
        "document_id": "doc-a",
        "chunk_type": "document_chunk",
        "chunk_id": "section-1",
        "token_start": 0,
        "token_count": 1,
        "token_end": 1,
        "byte_start": 0,
        "byte_length": TEST_BYTES_PER_TOKEN,
        "byte_end": TEST_BYTES_PER_TOKEN,
        "first_block_index": 0,
        "last_block_index_exclusive": 1,
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError), match=error_match):
        EngineKVSegmentBinding(**values)


@pytest.mark.parametrize(
    ("segment", "error_match"),
    [
        (
            segment_binding(token_start=1, byte_start=TEST_BYTES_PER_TOKEN),
            "Non-contiguous token segment binding",
        ),
        (
            segment_binding(byte_start=1),
            "Non-contiguous byte segment binding",
        ),
        (
            segment_binding(byte_length=TEST_BYTES_PER_TOKEN - 1),
            "token_count \\* bytes_per_token",
        ),
        (
            segment_binding(first_block_index=1, last_block_index_exclusive=2),
            "first_block_index",
        ),
        (
            segment_binding(last_block_index_exclusive=2),
            "last_block_index_exclusive",
        ),
    ],
)
def test_engine_kv_injection_plan_rejects_invalid_segment_bindings(segment, error_match):
    with pytest.raises(ValueError, match=error_match):
        EngineKVInjectionPlan(**injection_plan_kwargs(segments=(segment,)))


@pytest.mark.parametrize("estimated_gpu_bytes", [True, 1.0, "1", -1])
def test_engine_kv_reservation_action_rejects_invalid_estimated_gpu_bytes(estimated_gpu_bytes):
    with pytest.raises(ValueError, match="estimated_gpu_bytes must be a non-negative integer"):
        EngineKVReservationAction(
            backend=ServingBackend.VLLM,
            request_id="req-1",
            total_blocks=1,
            total_tokens=1,
            estimated_gpu_bytes=estimated_gpu_bytes,
            layout=layout(),
            adapter_ids=(),
        )


def test_engine_adapter_dataclasses_reject_custom_solver_backends():
    with pytest.raises(ValueError, match="Unsupported backend"):
        EngineAdapterSpec(
            backend="custom_solver",  # type: ignore[arg-type]
            connector_package="custom",
            kv_injection_method="custom",
            payload_contract="custom",
        )

    with pytest.raises(ValueError, match="connector_package must match backend"):
        EngineAdapterSpec(
            backend=ServingBackend.VLLM,
            connector_package="custom_solver",
            kv_injection_method="custom",
            payload_contract="custom",
        )

    with pytest.raises(ValueError, match="Unsupported backend"):
        EngineKVReservationAction(
            backend="custom_solver",  # type: ignore[arg-type]
            request_id="req-1",
            total_blocks=1,
            total_tokens=1,
            estimated_gpu_bytes=TEST_BYTES_PER_TOKEN,
            layout=layout(),
            adapter_ids=(),
        )

    with pytest.raises(ValueError, match="connector_package must match backend"):
        EngineKVConnectorProbeResult(
            backend=ServingBackend.VLLM,
            request_id="req-1",
            total_blocks=1,
            copied_segments=1,
            copied_tokens=1,
            copied_bytes=TEST_BYTES_PER_TOKEN,
            bound=True,
            released=True,
            model_id="qwen3:4b-instruct",
            layout_version="qwen3-v1",
            layout=layout(),
            payload_mode=PayloadMode.MERGED,
            connector_package="custom_solver",
            engine_version="vllm-test",
        )

    with pytest.raises(ValueError, match="Unsupported payload_mode"):
        EngineKVConnectorProbeResult(
            backend=ServingBackend.VLLM,
            request_id="req-1",
            total_blocks=1,
            copied_segments=1,
            copied_tokens=1,
            copied_bytes=TEST_BYTES_PER_TOKEN,
            bound=True,
            released=True,
            model_id="qwen3:4b-instruct",
            layout_version="qwen3-v1",
            layout=layout(),
            payload_mode="custom_payload",  # type: ignore[arg-type]
            connector_package="vllm",
            engine_version="vllm-test",
        )


def test_engine_adapter_request_rejects_custom_connector_package(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())

    with pytest.raises(ValueError, match="connector_package must match backend"):
        EngineAdapterRequest(
            backend="vllm",  # type: ignore[arg-type]
            ready_request=ready,
            connector_package="custom_solver",
            kv_injection_method="custom",
            payload_contract="custom",
            required_steps=("reserve", "bind"),
            metadata={},
        )

    with pytest.raises(ValueError, match="Unsupported backend"):
        EngineAdapterRequest(
            backend="custom_solver",  # type: ignore[arg-type]
            ready_request=ready,
            connector_package="custom_solver",
            kv_injection_method="custom",
            payload_contract="custom",
            required_steps=("reserve", "bind"),
            metadata={},
        )


def test_engine_handoff_and_probe_records_reject_custom_connector_package(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    handoff_record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    probe_record = engine_kv_connector_probe_result_to_record(
        EngineKVConnectorProbeResult(
            backend=ServingBackend.VLLM,
            request_id="req-1",
            total_blocks=1,
            copied_segments=1,
            copied_tokens=1,
            copied_bytes=TEST_BYTES_PER_TOKEN,
            bound=True,
            released=True,
            model_id="qwen3:4b-instruct",
            layout_version="qwen3-v1",
            layout=layout(),
            payload_mode=PayloadMode.MERGED,
            connector_package="vllm",
            engine_version="vllm-test",
        )
    )

    with pytest.raises(ValueError, match="connector_package must match backend"):
        validate_engine_adapter_request_record({**handoff_record, "connector_package": "custom_solver"})

    with pytest.raises(ValueError, match="connector_package must match backend"):
        validate_engine_kv_connector_probe_record({**probe_record, "connector_package": "custom_solver"})


def test_adapter_spec_rejects_unsupported_payload_mode(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    spec = EngineAdapterSpec(
        backend=ServingBackend.VLLM,
        connector_package="vllm",
        kv_injection_method="merged-only-test",
        payload_contract="test adapter",
        supports_segmented_payload=False,
    )

    with pytest.raises(ValueError, match="does not support segmented payloads"):
        build_engine_adapter_request(ready, spec=spec)


def test_adapter_spec_rejects_lora_ids_when_adapter_does_not_support_lora(tmp_path):
    ready = service(tmp_path).prepare_for_engine(
        request(),
        layout=layout(),
        adapter_ids=("selection-lora",),
    )
    spec = EngineAdapterSpec(
        backend=ServingBackend.SGLANG,
        connector_package="sglang",
        kv_injection_method="no-lora-test",
        payload_contract="test adapter",
        supports_lora_adapters=False,
    )

    with pytest.raises(ValueError, match="does not support LoRA"):
        build_engine_adapter_request(ready, spec=spec)


def test_adapter_spec_rejects_payload_total_byte_mismatch(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    bad_ready = type(ready)(
        handle=ready.handle,
        payload=ready.payload[:-1],
        estimated_gpu_bytes=ready.estimated_gpu_bytes,
    )

    with pytest.raises(ValueError, match="Payload byte length"):
        build_engine_adapter_request(bad_ready, spec=vllm_adapter_spec())


def test_adapter_spec_rejects_segmented_payload_count_mismatch(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    bad_ready = type(ready)(
        handle=ready.handle,
        payload=ready.payload[:1],
        estimated_gpu_bytes=ready.estimated_gpu_bytes,
    )

    with pytest.raises(ValueError, match="Segmented payload count"):
        build_engine_adapter_request(bad_ready, spec=vllm_adapter_spec())


def test_adapter_spec_rejects_non_bytes_merged_payload(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout())
    bad_ready = type(ready)(
        handle=ready.handle,
        payload=[0] * ready.handle.total_bytes,  # type: ignore[arg-type]
        estimated_gpu_bytes=ready.estimated_gpu_bytes,
    )

    with pytest.raises(TypeError, match="Payload must be bytes"):
        build_engine_adapter_request(bad_ready, spec=vllm_adapter_spec())


def test_adapter_spec_rejects_non_bytes_segment_payload(tmp_path):
    ready = service(tmp_path).prepare_for_engine(request(), layout=layout(), segmented=True)
    bad_ready = type(ready)(
        handle=ready.handle,
        payload=(bytearray(ready.payload[0]), ready.payload[1]),  # type: ignore[arg-type]
        estimated_gpu_bytes=ready.estimated_gpu_bytes,
    )

    with pytest.raises(TypeError, match="Segmented payload 0 must be bytes"):
        build_engine_adapter_request(bad_ready, spec=vllm_adapter_spec())


def test_adapter_spec_rejects_reserved_handle_metadata(tmp_path):
    ready = service(tmp_path).prepare_for_engine(
        request(),
        layout=layout(),
        metadata={"engine.scheduler": "caller-owned"},
    )

    with pytest.raises(ValueError, match="reserved adapter keys"):
        build_engine_adapter_request(ready, spec=vllm_adapter_spec())


def test_adapter_spec_rejects_non_string_metadata_keys_and_values(tmp_path):
    ready = service(tmp_path).prepare_for_engine(
        request(),
        layout=layout(),
        metadata={1: "bad"},  # type: ignore[dict-item]
    )

    with pytest.raises(TypeError, match="metadata keys and values"):
        build_engine_adapter_request(ready, spec=vllm_adapter_spec())

    with pytest.raises(TypeError, match="metadata keys and values"):
        EngineAdapterSpec(
            backend=ServingBackend.VLLM,
            connector_package="vllm",
            kv_injection_method="bad",
            payload_contract="bad",
            metadata={"ok": 1},  # type: ignore[dict-item]
        )


def test_adapter_spec_coerces_required_steps_to_immutable_tuple():
    steps = ["reserve", "bind"]
    spec = EngineAdapterSpec(
        backend=ServingBackend.VLLM,
        connector_package="vllm",
        kv_injection_method="test",
        payload_contract="test adapter",
        required_steps=steps,  # type: ignore[arg-type]
    )

    steps.append("mutated")

    assert spec.required_steps == ("reserve", "bind")


def test_adapter_spec_rejects_string_required_steps():
    with pytest.raises(ValueError, match="sequence of non-empty strings"):
        EngineAdapterSpec(
            backend=ServingBackend.VLLM,
            connector_package="vllm",
            kv_injection_method="bad",
            payload_contract="bad",
            required_steps="reserve",  # type: ignore[arg-type]
        )


def test_adapter_spec_requires_at_least_one_payload_mode():
    with pytest.raises(ValueError, match="at least one payload mode"):
        EngineAdapterSpec(
            backend=ServingBackend.VLLM,
            connector_package="vllm",
            kv_injection_method="bad",
            payload_contract="bad",
            supports_merged_payload=False,
            supports_segmented_payload=False,
        )


class FakeNativeBlockManagerProbe:
    def __init__(self) -> None:
        self.events = []
        self.payload_objects = []

    def reserve_kv_blocks(self, action):
        self.events.append(("reserve", action.request_id, action.total_blocks))
        return f"{action.request_id}:blocks"

    def import_kv_segment(self, reservation_id, action, payload: memoryview) -> None:
        if payload.nbytes != action.source_byte_length:
            raise AssertionError("fake connector received the wrong payload slice")
        self.payload_objects.append(payload.obj)
        self.events.append(
            (
                "copy",
                reservation_id,
                action.chunk_id,
                type(payload).__name__,
                payload.nbytes,
                action.first_block_index,
                action.last_block_index_exclusive,
            )
        )

    def bind_kv_handle(self, reservation_id, action) -> None:
        assert reservation_id == f"{action.request_id}:blocks"
        self.events.append(("bind", action.request_id, action.handle_uri, action.adapter_ids))

    def release_kv_blocks(self, reservation_id, action) -> None:
        assert reservation_id == f"{action.request_id}:blocks"
        self.events.append(("release", action.request_id))
