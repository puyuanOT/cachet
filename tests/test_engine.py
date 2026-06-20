from dataclasses import replace

import pytest

from document_kv_cache.admission import AdmissionQueue
from document_kv_cache.cache import CacheTier, ChunkCache
from document_kv_cache.engine import EngineReadyRequest, build_engine_ready_request, build_handle_from_materialized
from document_kv_cache.engine_protocol import AttentionMechanism, KVLayout, KVStorageLayout
from document_kv_cache.kvpack import PackChunk, write_kvpack
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.materializer import KVMaterializer
from document_kv_cache.models import CacheGenerationMethod, DocumentChunkType, DocumentKVRequest, KVCacheKey
from document_kv_cache.planner import CachePlanner
from document_kv_cache.service import DocumentKVService
from document_kv_cache.storage import DiskRangeReader


TEST_BYTES_PER_TOKEN = 36 * 8 * 128 * 2
STATIC_TOKEN_COUNT = 2
REVIEW_TOKEN_COUNT = 3
STATIC_PAYLOAD = b"s" * (STATIC_TOKEN_COUNT * TEST_BYTES_PER_TOKEN)
REVIEW_PAYLOAD = b"r" * (REVIEW_TOKEN_COUNT * TEST_BYTES_PER_TOKEN)


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
        tmp_path / "engine.kvpack",
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
                REVIEW_PAYLOAD,
                REVIEW_TOKEN_COUNT,
                "int8",
                "qwen3-v1",
                storage_layout=KVStorageLayout.SHARED_KEY_VALUE,
            ),
        ],
        align_bytes=1,
    )
    manifest = InMemoryManifestStore(refs)
    materializer = KVMaterializer(cache=ChunkCache(cpu_max_bytes=1024), reader=DiskRangeReader())
    return DocumentKVService(
        planner=CachePlanner(manifest),
        materializer=materializer,
        admission_queue=AdmissionQueue(max_pending_gpu_bytes=4096),
        kv_gpu_bytes_per_payload_byte=2.0,
    )


def test_build_handle_from_materialized_kv_segments(tmp_path):
    document_service = service(tmp_path)
    plan = document_service.planner.build_plan(request())
    materialized = document_service.materializer.materialize(plan)

    handle = build_handle_from_materialized(
        materialized,
        layout=layout(),
        metadata={"engine": "vllm"},
        cache_method=CacheGenerationMethod.KV_PACKET,
        adapter_ids=("qa-lora",),
    )

    assert handle.request_id == "req-1"
    assert handle.handle_uri == "document-kv://req-1"
    assert handle.total_tokens == 5
    assert handle.total_bytes == len(STATIC_PAYLOAD) + len(REVIEW_PAYLOAD)
    assert handle.metadata == {"engine": "vllm"}
    assert handle.cache_method == "kv_packet"
    assert handle.adapter_ids == ("qa-lora",)
    assert handle.layout.attention_mechanism == AttentionMechanism.GROUPED_QUERY
    assert handle.layout.query_heads_per_kv_head == 4
    assert handle.layout.storage_layout == KVStorageLayout.SHARED_KEY_VALUE
    assert [
        (segment.chunk_type, segment.chunk_id, segment.token_start, segment.byte_start) for segment in handle.segments
    ] == [
        ("document_static", "static", 0, 0),
        ("document_chunk", "section-1", 2, len(STATIC_PAYLOAD)),
    ]

    ready = build_engine_ready_request(
        materialized,
        layout=layout(),
        metadata={"engine": "vllm"},
    )
    assert ready.segment_tiers == materialized.segment_tiers
    assert ready.segment_tiers == (CacheTier.COLD_STORAGE, CacheTier.COLD_STORAGE)


def test_service_prepares_engine_ready_request_with_segmented_payload(tmp_path):
    prepared = service(tmp_path).prepare_for_engine(
        request(),
        layout=layout(),
        handle_uri="sglang://req-1",
        metadata={"engine": "sglang"},
        segmented=True,
    )

    assert prepared.request_id == "req-1"
    assert prepared.handle.handle_uri == "sglang://req-1"
    assert prepared.payload == (STATIC_PAYLOAD, REVIEW_PAYLOAD)
    assert prepared.estimated_gpu_bytes == 2 * (len(STATIC_PAYLOAD) + len(REVIEW_PAYLOAD))
    assert prepared.handle.metadata == {"engine": "sglang"}
    assert prepared.handle.cache_method == "vanilla_prefill"
    assert prepared.segment_tiers == (CacheTier.COLD_STORAGE, CacheTier.COLD_STORAGE)


def test_build_handle_rejects_layout_metadata_mismatches(tmp_path):
    document_service = service(tmp_path)
    materialized = document_service.materializer.materialize(document_service.planner.build_plan(request()))

    with pytest.raises(ValueError, match="model_id"):
        build_handle_from_materialized(materialized, layout=replace(layout(), model_id="qwen3.5"))
    with pytest.raises(ValueError, match="dtype"):
        build_handle_from_materialized(materialized, layout=replace(layout(), dtype="fp8"))
    with pytest.raises(ValueError, match="storage_layout"):
        build_handle_from_materialized(
            materialized,
            layout=replace(layout(), shares_kv_storage=False, storage_layout=KVStorageLayout.SEPARATE_KEY_VALUE),
        )
    with pytest.raises(ValueError, match="bytes_per_token"):
        build_handle_from_materialized(
            materialized,
            layout=replace(layout(), bytes_per_token=3),
        )


@pytest.mark.parametrize(
    ("multiplier", "error_type", "message"),
    [
        (True, TypeError, "kv_gpu_bytes_per_payload_byte must be numeric"),
        ("1.0", TypeError, "kv_gpu_bytes_per_payload_byte must be numeric"),
        (10**400, ValueError, "kv_gpu_bytes_per_payload_byte must be finite"),
        (float("nan"), ValueError, "kv_gpu_bytes_per_payload_byte must be finite"),
        (float("inf"), ValueError, "kv_gpu_bytes_per_payload_byte must be finite"),
        (-1.0, ValueError, "kv_gpu_bytes_per_payload_byte must be non-negative"),
    ],
)
def test_build_engine_ready_request_rejects_invalid_gpu_multiplier(tmp_path, multiplier, error_type, message):
    document_service = service(tmp_path)
    materialized = document_service.materializer.materialize(document_service.planner.build_plan(request()))

    with pytest.raises(error_type, match=message):
        build_engine_ready_request(
            materialized,
            layout=layout(),
            kv_gpu_bytes_per_payload_byte=multiplier,
        )


def test_build_engine_ready_request_rejects_nonfinite_gpu_estimate(tmp_path):
    document_service = service(tmp_path)
    materialized = document_service.materializer.materialize(document_service.planner.build_plan(request()))

    with pytest.raises(ValueError, match="estimated_gpu_bytes must be finite"):
        build_engine_ready_request(
            materialized,
            layout=layout(),
            kv_gpu_bytes_per_payload_byte=1e308,
        )


@pytest.mark.parametrize(
    ("estimated_gpu_bytes", "message"),
    [
        (True, "estimated_gpu_bytes must be an integer"),
        (1.0, "estimated_gpu_bytes must be an integer"),
        ("1", "estimated_gpu_bytes must be an integer"),
        (-1, "estimated_gpu_bytes must be non-negative"),
    ],
)
def test_engine_ready_request_validates_estimated_gpu_bytes(tmp_path, estimated_gpu_bytes, message):
    document_service = service(tmp_path)
    materialized = document_service.materializer.materialize(document_service.planner.build_plan(request()))
    handle = build_handle_from_materialized(materialized, layout=layout())
    ready = EngineReadyRequest(
        handle=handle,
        payload=materialized.payload,
        estimated_gpu_bytes=estimated_gpu_bytes,
        segment_tiers=materialized.segment_tiers,
    )

    with pytest.raises(ValueError, match=message):
        ready.validate()


@pytest.mark.parametrize(
    ("multiplier", "error_type", "message"),
    [
        (True, TypeError, "kv_gpu_bytes_per_payload_byte must be numeric"),
        ("1.0", TypeError, "kv_gpu_bytes_per_payload_byte must be numeric"),
        (10**400, ValueError, "kv_gpu_bytes_per_payload_byte must be finite"),
        (float("nan"), ValueError, "kv_gpu_bytes_per_payload_byte must be finite"),
        (float("inf"), ValueError, "kv_gpu_bytes_per_payload_byte must be finite"),
        (-1.0, ValueError, "kv_gpu_bytes_per_payload_byte must be non-negative"),
    ],
)
def test_service_rejects_invalid_gpu_multiplier(tmp_path, multiplier, error_type, message):
    refs = write_kvpack(
        tmp_path / "engine.kvpack",
        [
            PackChunk(
                key(DocumentChunkType.DOCUMENT_STATIC, "static"),
                STATIC_PAYLOAD,
                STATIC_TOKEN_COUNT,
                "int8",
                "qwen3-v1",
                storage_layout=KVStorageLayout.SHARED_KEY_VALUE,
            )
        ],
        align_bytes=1,
    )

    with pytest.raises(error_type, match=message):
        DocumentKVService(
            planner=CachePlanner(InMemoryManifestStore(refs)),
            materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=1024), reader=DiskRangeReader()),
            admission_queue=AdmissionQueue(max_pending_gpu_bytes=4096),
            kv_gpu_bytes_per_payload_byte=multiplier,
        )


def test_layout_validation_rejects_invalid_gqa_shape():
    bad_layout = KVLayout(
        model_id="qwen3",
        lora_id="base",
        layout_version="v1",
        dtype="int8",
        num_layers=32,
        block_size=16,
        bytes_per_token=32 * 8 * 128 * 2,
        num_query_heads=30,
        num_kv_heads=8,
        head_size=128,
        kv_stride_bytes=128,
    )

    with pytest.raises(ValueError, match="divisible"):
        bad_layout.validate()


def test_layout_validation_rejects_inconsistent_geometry_byte_math():
    with pytest.raises(ValueError, match="does not match layout geometry"):
        replace(layout(), bytes_per_token=TEST_BYTES_PER_TOKEN - 1).validate()


def test_layout_validation_rejects_inconsistent_kv_stride():
    with pytest.raises(ValueError, match="kv_stride_bytes"):
        replace(layout(), kv_stride_bytes=256).validate()


def test_layout_validation_rejects_conflicting_storage_layout():
    with pytest.raises(ValueError, match="shares_kv_storage"):
        replace(layout(), storage_layout=KVStorageLayout.SEPARATE_KEY_VALUE).validate()
    with pytest.raises(ValueError, match="storage_layout"):
        replace(layout(), shares_kv_storage=False, storage_layout=KVStorageLayout.SHARED_KEY_VALUE).validate()
    with pytest.raises(ValueError, match="qwen3-v1"):
        replace(layout(), shares_kv_storage=False, storage_layout=KVStorageLayout.SEPARATE_KEY_VALUE).validate()
    with pytest.raises(ValueError, match="Unsupported storage_layout"):
        replace(layout(), storage_layout="packed").validate()


def test_layout_validation_rejects_incomplete_shared_kv_shape():
    bad_layout = KVLayout(
        model_id="qwen3",
        lora_id="base",
        layout_version="v1",
        dtype="int8",
        num_layers=32,
        block_size=16,
        bytes_per_token=32 * 8 * 128 * 2,
        num_query_heads=32,
        shares_kv_storage=True,
    )

    with pytest.raises(ValueError, match="required together"):
        bad_layout.validate()
