import importlib
import copy
import json
import pickle
from dataclasses import MISSING, asdict, fields, replace

import pytest

from document_kv_cache.cache import CacheTier, ChunkCache
from document_kv_cache.kvpack import PackChunk, write_kvpack
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.materializer import KVMaterializer, MaterializedKV, SegmentedMaterializedKV
from document_kv_cache.models import (
    CacheChunkType,
    ChunkType,
    DOCUMENT_CHUNK_TYPES,
    DocumentChunkType,
    DocumentChunkRole,
    DocumentKVRequest,
    FrozenDocumentChunkMap,
    KVCacheKey,
    LEGACY_RESTAURANT_CHUNK_TYPES,
    MaterializationPlan,
    PlanSegment,
    RestaurantKVRequest,
    chunk_type_role,
    chunk_type_sort_order,
    chunk_types_for_request,
)
from document_kv_cache.planner import CachePlanner
from document_kv_cache.storage import DiskRangeReader


def document_plan(tmp_path, payloads: tuple[bytes, ...] = (b"alpha", b"beta")):
    chunks = [
        PackChunk(
            make_key("doc-a", DocumentChunkType.DOCUMENT_CHUNK, f"section-{index}"),
            payload,
            max(1, len(payload)),
            "fp8",
            "v1",
        )
        for index, payload in enumerate(payloads, start=1)
    ]
    refs = write_kvpack(tmp_path / "document-plan.kvpack", chunks, align_bytes=1)
    request = DocumentKVRequest(
        request_id="req-1",
        task_id="qa",
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        document_chunks={"doc-a": [f"section-{index}" for index in range(1, len(payloads) + 1)]},
        include_static=False,
    )
    return CachePlanner(InMemoryManifestStore(refs)).build_plan(request)


def make_key(document_id: str, chunk_type: CacheChunkType, chunk_id: str) -> KVCacheKey:
    return KVCacheKey(
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        document_id=document_id,
        chunk_type=chunk_type,
        chunk_id=chunk_id,
    )


class BatchCountingReader:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.read_calls = 0
        self.read_many_calls = 0

    def read(self, ref) -> bytes:
        self.read_calls += 1
        return self.payloads[ref.key.chunk_id]

    def read_many(self, refs) -> tuple[bytes, ...]:
        self.read_many_calls += 1
        return tuple(self.payloads[ref.key.chunk_id] for ref in refs)


def test_kv_cache_key_uses_document_id_with_restaurant_alias_compatibility():
    key = make_key("doc-a", DocumentChunkType.DOCUMENT_CHUNK, "section-1")
    legacy_key = KVCacheKey(
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        restaurant_id="doc-a",
        chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
        chunk_id="section-1",
    )

    assert key.document_id == "doc-a"
    assert key.restaurant_id == "doc-a"
    assert key == legacy_key
    assert "|doc-a|document_chunk|section-1|" in key.storage_key()
    assert replace(key, document_id="doc-b").document_id == "doc-b"
    assert replace(key, restaurant_id="legacy-b").document_id == "legacy-b"


def test_kv_cache_key_requires_a_document_or_legacy_restaurant_id():
    with pytest.raises(TypeError, match="document_id"):
        KVCacheKey(
            model_id="qwen35-4b-w8a8",
            lora_id="selection",
            prompt_template_version="v1",
            chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
            chunk_id="section-1",
        )


def test_kv_cache_key_validates_storage_key_fields():
    key = make_key("doc-a", DocumentChunkType.DOCUMENT_CHUNK, "section-1")

    assert replace(key, content_hash="short-hash").content_hash == "short-hash"
    with pytest.raises(ValueError, match="model_id"):
        replace(key, model_id="")
    with pytest.raises(ValueError, match="lora_id"):
        replace(key, lora_id=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="document_id"):
        replace(key, document_id="")
    with pytest.raises(ValueError, match="prompt_template_version"):
        replace(key, prompt_template_version="v|1")
    with pytest.raises(TypeError, match="chunk_type"):
        replace(key, chunk_type="document_chunk")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="chunk_id"):
        replace(key, chunk_id="")
    with pytest.raises(ValueError, match="chunk_id"):
        replace(key, chunk_id="section|1")
    with pytest.raises(TypeError, match="content_hash"):
        replace(key, content_hash=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="content_hash"):
        replace(key, content_hash="hash|with-delimiter")


def test_chunk_roles_unify_document_and_legacy_aliases():
    document_request = DocumentKVRequest(
        request_id="doc-req",
        task_id="qa",
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        document_chunks={"doc-a": ["section-1"]},
    )
    restaurant_request = RestaurantKVRequest(
        request_id="legacy-req",
        task_id="selection",
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        restaurant_reviews={"r1": ["rev1"]},
    )

    assert chunk_types_for_request(document_request) == DOCUMENT_CHUNK_TYPES
    assert chunk_types_for_request(restaurant_request) == LEGACY_RESTAURANT_CHUNK_TYPES
    assert chunk_type_role(DocumentChunkType.TASK_PREFIX) == DocumentChunkRole.TASK_PREFIX
    assert chunk_type_role(ChunkType.TASK_PREFIX) == DocumentChunkRole.TASK_PREFIX
    assert chunk_type_role(DocumentChunkType.DOCUMENT_STATIC) == DocumentChunkRole.STATIC
    assert chunk_type_role(ChunkType.RESTAURANT_STATIC) == DocumentChunkRole.STATIC
    assert chunk_type_role(DocumentChunkType.DOCUMENT_CHUNK) == DocumentChunkRole.CONTENT
    assert chunk_type_role(ChunkType.REVIEW) == DocumentChunkRole.CONTENT
    assert chunk_type_role("custom_chunk") == DocumentChunkRole.OTHER
    assert chunk_type_sort_order(DocumentChunkType.TASK_PREFIX) < chunk_type_sort_order(DocumentChunkType.DOCUMENT_STATIC)
    assert chunk_type_sort_order(ChunkType.REVIEW) == chunk_type_sort_order(DocumentChunkType.DOCUMENT_CHUNK)

    with pytest.raises(TypeError, match="chunk_type"):
        chunk_type_role(object())  # type: ignore[arg-type]


def test_document_kv_request_validates_metadata_and_chunk_map():
    chunk_ids = ["section-1", 2]
    document_chunks = {"doc-a": chunk_ids}
    request = DocumentKVRequest(
        request_id="doc-req",
        task_id="qa",
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        document_chunks=document_chunks,
        task_prefix_id="prefix",
    )

    chunk_ids.append("late-section")
    document_chunks["doc-b"] = ("late-section",)

    assert request.document_chunks == {"doc-a": ("section-1", 2)}
    assert request.document_chunks["doc-a"] == ("section-1", 2)
    assert request.selected_document_ids == ("doc-a",)
    assert json.loads(json.dumps(request.document_chunks)) == {"doc-a": ["section-1", 2]}
    assert asdict(request)["document_chunks"] == {"doc-a": ("section-1", 2)}
    assert copy.deepcopy(request).document_chunks == {"doc-a": ("section-1", 2)}
    assert pickle.loads(pickle.dumps(request)).document_chunks == {"doc-a": ("section-1", 2)}
    with pytest.raises(TypeError, match="immutable"):
        request.document_chunks["doc-b"] = ("late-section",)  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        request.document_chunks.update({"doc-b": ("late-section",)})
    with pytest.raises(ValueError, match="request_id"):
        replace(request, request_id="")
    with pytest.raises(ValueError, match="model_id"):
        replace(request, model_id=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="include_static"):
        replace(request, include_static=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="task_prefix_id"):
        replace(request, task_prefix_id="")
    with pytest.raises(TypeError, match="document_chunks"):
        replace(request, document_chunks=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="document_chunks keys"):
        replace(request, document_chunks={"": ["section-1"]})
    with pytest.raises(TypeError, match="document_chunks values"):
        replace(request, document_chunks={"doc-a": "section-1"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="document_chunks values"):
        replace(request, document_chunks={"doc-a": bytearray(b"rev1")})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="document_chunks values"):
        replace(request, document_chunks={"doc-a": memoryview(b"rev1")})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="document_chunks chunk ids"):
        replace(request, document_chunks={"doc-a": [""]})
    with pytest.raises(TypeError, match="document_chunks chunk ids"):
        replace(request, document_chunks={"doc-a": [True]})  # type: ignore[list-item]


def test_document_kv_request_for_text_document_matches_source_document_default_chunk():
    request = DocumentKVRequest.for_text_document(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_id="doc-a",
    )

    assert request.document_chunks == {"doc-a": ("document",)}
    assert request.include_static is False
    assert request.selected_document_ids == ("doc-a",)

    custom = DocumentKVRequest.for_text_document(
        request_id="req-2",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_id="doc-a",
        chunk_id="body",
    )

    assert custom.document_chunks == {"doc-a": ("body",)}


def test_document_kv_request_for_document_chunks_builds_single_document_selection():
    request = DocumentKVRequest.for_document_chunks(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_id="doc-a",
        chunk_ids=("section-1", 2),
        task_prefix_id="prefix",
    )

    assert request.document_chunks == {"doc-a": ("section-1", 2)}
    assert request.include_static is True
    assert request.task_prefix_id == "prefix"
    assert request.selected_document_ids == ("doc-a",)


def test_document_kv_request_for_document_chunks_allows_static_opt_out():
    request = DocumentKVRequest.for_document_chunks(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_id="doc-a",
        chunk_ids=("section-1",),
        include_static=False,
    )

    assert request.document_chunks == {"doc-a": ("section-1",)}
    assert request.include_static is False


def test_document_kv_request_for_text_document_reuses_request_validation():
    with pytest.raises(ValueError, match="document_chunks keys"):
        DocumentKVRequest.for_text_document(
            request_id="req-1",
            task_id="qa",
            model_id="qwen3:4b-instruct",
            lora_id="base",
            prompt_template_version="v1",
            document_id="",
        )
    with pytest.raises(ValueError, match="document_chunks chunk ids"):
        DocumentKVRequest.for_text_document(
            request_id="req-1",
            task_id="qa",
            model_id="qwen3:4b-instruct",
            lora_id="base",
            prompt_template_version="v1",
            document_id="doc-a",
            chunk_id="",
        )


def test_document_kv_request_for_document_chunks_reuses_request_validation():
    with pytest.raises(TypeError, match="document_chunks values"):
        DocumentKVRequest.for_document_chunks(
            request_id="req-1",
            task_id="qa",
            model_id="qwen3:4b-instruct",
            lora_id="base",
            prompt_template_version="v1",
            document_id="doc-a",
            chunk_ids="section-1",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="document_chunks chunk ids"):
        DocumentKVRequest.for_document_chunks(
            request_id="req-1",
            task_id="qa",
            model_id="qwen3:4b-instruct",
            lora_id="base",
            prompt_template_version="v1",
            document_id="doc-a",
            chunk_ids=("",),
        )


def test_frozen_document_chunk_map_normalizes_direct_construction():
    chunk_ids = ["section-1", 2]
    chunk_map = FrozenDocumentChunkMap({"doc-a": chunk_ids})

    chunk_ids.append("late-section")

    assert chunk_map == {"doc-a": ("section-1", 2)}
    assert json.loads(json.dumps(chunk_map)) == {"doc-a": ["section-1", 2]}
    with pytest.raises(TypeError, match="immutable"):
        chunk_map.setdefault("doc-b", ("late-section",))
    with pytest.raises(ValueError, match="FrozenDocumentChunkMap keys"):
        FrozenDocumentChunkMap({"": ["section-1"]})
    with pytest.raises(TypeError, match="FrozenDocumentChunkMap values"):
        FrozenDocumentChunkMap({"doc-a": "section-1"})


def test_restaurant_kv_request_validates_legacy_review_map():
    review_ids = ["rev1", 7]
    restaurant_reviews = {"r1": review_ids}
    request = RestaurantKVRequest(
        request_id="legacy-req",
        task_id="selection",
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        restaurant_reviews=restaurant_reviews,
    )

    review_ids.append("late-review")
    restaurant_reviews["r2"] = ("late-review",)

    assert request.restaurant_reviews == {"r1": ("rev1", 7)}
    assert request.document_chunks == {"r1": ("rev1", 7)}
    assert request.selected_documents == ("r1",)
    assert json.loads(json.dumps(request.restaurant_reviews)) == {"r1": ["rev1", 7]}
    assert asdict(request)["restaurant_reviews"] == {"r1": ("rev1", 7)}
    assert copy.deepcopy(request).restaurant_reviews == {"r1": ("rev1", 7)}
    assert pickle.loads(pickle.dumps(request)).restaurant_reviews == {"r1": ("rev1", 7)}
    with pytest.raises(TypeError, match="immutable"):
        request.restaurant_reviews["r2"] = ("late-review",)  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        request.restaurant_reviews.update({"r2": ("late-review",)})
    with pytest.raises(TypeError, match="restaurant_reviews"):
        replace(request, restaurant_reviews=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="restaurant_reviews keys"):
        replace(request, restaurant_reviews={None: ["rev1"]})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="restaurant_reviews values"):
        replace(request, restaurant_reviews={"r1": b"rev1"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="restaurant_reviews values"):
        replace(request, restaurant_reviews={"r1": bytearray(b"rev1")})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="restaurant_reviews values"):
        replace(request, restaurant_reviews={"r1": memoryview(b"rev1")})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="restaurant_reviews chunk ids"):
        replace(request, restaurant_reviews={"r1": [None]})  # type: ignore[list-item]


def test_manifest_orders_mixed_legacy_and_document_aliases_deterministically(tmp_path):
    chunks = [
        PackChunk(make_key("doc-a", ChunkType.REVIEW, "same-id"), b"legacy", 6, "fp8", "v1"),
        PackChunk(make_key("doc-a", DocumentChunkType.DOCUMENT_CHUNK, "same-id"), b"generic", 7, "fp8", "v1"),
        PackChunk(make_key("doc-a", DocumentChunkType.DOCUMENT_STATIC, "static"), b"static", 6, "fp8", "v1"),
    ]
    refs = write_kvpack(tmp_path / "mixed.kvpack", reversed(chunks), align_bytes=1)
    manifest = InMemoryManifestStore(refs)

    assert [key.chunk_type for key in manifest.keys_for_document("doc-a")] == [
        DocumentChunkType.DOCUMENT_STATIC,
        DocumentChunkType.DOCUMENT_CHUNK,
        ChunkType.REVIEW,
    ]


def test_plan_static_then_selected_reviews_and_materialize(tmp_path):
    chunks = [
        PackChunk(make_key("r1", ChunkType.RESTAURANT_STATIC, "static"), b"menu:", 5, "fp8", "v1"),
        PackChunk(make_key("r1", ChunkType.REVIEW, "rev2"), b"good", 4, "fp8", "v1"),
        PackChunk(make_key("r1", ChunkType.REVIEW, "rev1"), b"bad", 3, "fp8", "v1"),
        PackChunk(make_key("r2", ChunkType.RESTAURANT_STATIC, "static"), b"ramen:", 6, "fp8", "v1"),
        PackChunk(make_key("r2", ChunkType.REVIEW, "rev9"), b"great", 5, "fp8", "v1"),
    ]
    refs = write_kvpack(tmp_path / "shard.kvpack", chunks, align_bytes=1)
    manifest = InMemoryManifestStore(refs)
    planner = CachePlanner(manifest)
    request = RestaurantKVRequest(
        request_id="req-1",
        task_id="selection",
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        restaurant_reviews={"r1": ["rev1", "rev2"], "r2": ["rev9"]},
    )

    plan = planner.build_plan(request)
    materializer = KVMaterializer(
        cache=ChunkCache(cpu_max_bytes=1024, local_dir=tmp_path / "chunk-cache"),
        reader=DiskRangeReader(),
    )
    materialized = materializer.materialize(plan)

    assert plan.selected_document_ids == ("r1", "r2")
    assert plan.selected_documents == ("r1", "r2")
    assert plan.selected_restaurants == ("r1", "r2")
    assert plan.total_tokens == 23
    assert [segment.ref.key.chunk_id for segment in plan.segments] == ["static", "rev1", "rev2", "static", "rev9"]
    assert materialized.payload == b"menu:badgoodramen:great"
    assert materialized.segment_byte_offsets == (0, 5, 8, 12, 18)
    assert materialized.segment_tiers == (CacheTier.COLD_STORAGE,) * 5

    segmented = materializer.materialize_segmented(plan)
    assert segmented.payloads == (b"menu:", b"bad", b"good", b"ramen:", b"great")
    assert segmented.total_bytes == len(materialized.payload)
    assert segmented.segment_byte_offsets == materialized.segment_byte_offsets
    assert segmented.segment_tiers == (CacheTier.CPU,) * 5


def test_materializer_uses_reader_batch_loader_when_available(tmp_path):
    plan = document_plan(tmp_path, (b"abc", b"de"))
    reader = BatchCountingReader({"section-1": b"abc", "section-2": b"de"})
    materializer = KVMaterializer(cache=ChunkCache(cpu_max_bytes=1024), reader=reader)

    materialized = materializer.materialize(plan)
    segmented = materializer.materialize_segmented(plan)

    assert materialized.payload == b"abcde"
    assert materialized.segment_tiers == (CacheTier.COLD_STORAGE, CacheTier.COLD_STORAGE)
    assert segmented.payloads == (b"abc", b"de")
    assert segmented.segment_tiers == (CacheTier.CPU, CacheTier.CPU)
    assert reader.read_many_calls == 1
    assert reader.read_calls == 0


def test_materialized_kv_validates_payload_length_and_offsets(tmp_path):
    plan = document_plan(tmp_path, (b"abc", b"de"))

    materialized = MaterializedKV(
        plan=plan,
        payload=b"abcde",
        segment_byte_offsets=(0, 3),
        materialization_seconds=0.0,
    )

    assert materialized.payload == b"abcde"
    assert materialized.segment_tiers == (CacheTier.COLD_STORAGE, CacheTier.COLD_STORAGE)
    with pytest.raises(ValueError, match="payload byte length"):
        MaterializedKV(plan=plan, payload=b"abcd", segment_byte_offsets=(0, 3), materialization_seconds=0.0)
    with pytest.raises(ValueError, match="segment_byte_offsets"):
        MaterializedKV(plan=plan, payload=b"abcde", segment_byte_offsets=(0, 4), materialization_seconds=0.0)
    with pytest.raises(TypeError, match="segment_byte_offsets must be a tuple"):
        MaterializedKV(plan=plan, payload=b"abcde", segment_byte_offsets=[0, 3], materialization_seconds=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="segment_byte_offsets entries"):
        MaterializedKV(plan=plan, payload=b"abcde", segment_byte_offsets=(False, 3), materialization_seconds=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="materialization_seconds"):
        MaterializedKV(plan=plan, payload=b"abcde", segment_byte_offsets=(0, 3), materialization_seconds=-0.1)
    with pytest.raises(TypeError, match="materialization_seconds"):
        MaterializedKV(plan=plan, payload=b"abcde", segment_byte_offsets=(0, 3), materialization_seconds=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="segment_tiers count"):
        MaterializedKV(
            plan=plan,
            payload=b"abcde",
            segment_byte_offsets=(0, 3),
            materialization_seconds=0.0,
            segment_tiers=(CacheTier.CPU,),
        )
    with pytest.raises(ValueError, match="segment_tiers entries"):
        MaterializedKV(
            plan=plan,
            payload=b"abcde",
            segment_byte_offsets=(0, 3),
            materialization_seconds=0.0,
            segment_tiers=("bad-tier", "cpu"),
        )


def test_segmented_materialized_kv_validates_payload_lengths_and_totals(tmp_path):
    plan = document_plan(tmp_path, (b"abc", b"de"))

    segmented = SegmentedMaterializedKV(
        plan=plan,
        payloads=(b"abc", b"de"),
        segment_byte_offsets=(0, 3),
        total_bytes=5,
        materialization_seconds=0.0,
    )

    assert segmented.total_bytes == 5
    assert segmented.segment_tiers == (CacheTier.COLD_STORAGE, CacheTier.COLD_STORAGE)
    with pytest.raises(ValueError, match="payload count"):
        SegmentedMaterializedKV(
            plan=plan,
            payloads=(b"abc",),
            segment_byte_offsets=(0, 3),
            total_bytes=5,
            materialization_seconds=0.0,
        )
    with pytest.raises(TypeError, match="payloads must be a tuple"):
        SegmentedMaterializedKV(
            plan=plan,
            payloads=[b"abc", b"de"],  # type: ignore[arg-type]
            segment_byte_offsets=(0, 3),
            total_bytes=5,
            materialization_seconds=0.0,
        )
    with pytest.raises(ValueError, match="payload 1 byte length"):
        SegmentedMaterializedKV(
            plan=plan,
            payloads=(b"abc", b"x"),
            segment_byte_offsets=(0, 3),
            total_bytes=5,
            materialization_seconds=0.0,
        )
    with pytest.raises(ValueError, match="total_bytes"):
        SegmentedMaterializedKV(
            plan=plan,
            payloads=(b"abc", b"de"),
            segment_byte_offsets=(0, 3),
            total_bytes=4,
            materialization_seconds=0.0,
        )
    with pytest.raises(ValueError, match="segment_tiers count"):
        SegmentedMaterializedKV(
            plan=plan,
            payloads=(b"abc", b"de"),
            segment_byte_offsets=(0, 3),
            total_bytes=5,
            materialization_seconds=0.0,
            segment_tiers=(CacheTier.LOCAL_DISK,),
        )


def test_materializer_public_module_owns_implementation_and_legacy_aliases_it():
    public_materializer = importlib.import_module("document_kv_cache.materializer")
    legacy_materializer = importlib.import_module("restaurant_kv_serving.materializer")

    assert public_materializer.MaterializedKV.__module__ == "document_kv_cache.materializer"
    assert public_materializer.SegmentedMaterializedKV.__module__ == "document_kv_cache.materializer"
    assert public_materializer.KVMaterializer.__module__ == "document_kv_cache.materializer"
    assert legacy_materializer.MaterializedKV is public_materializer.MaterializedKV
    assert legacy_materializer.SegmentedMaterializedKV is public_materializer.SegmentedMaterializedKV
    assert legacy_materializer.KVMaterializer is public_materializer.KVMaterializer
    assert legacy_materializer.normalize_segment_tiers is public_materializer.normalize_segment_tiers


def test_materializer_star_import_surfaces_are_curated_for_document_and_preserved_for_legacy():
    public_namespace: dict[str, object] = {}
    legacy_namespace: dict[str, object] = {}

    exec("from document_kv_cache.materializer import *", public_namespace)
    exec("from restaurant_kv_serving.materializer import *", legacy_namespace)

    assert set(public_namespace) >= {"MaterializedKV", "SegmentedMaterializedKV", "KVMaterializer"}
    assert "normalize_segment_tiers" not in public_namespace
    assert "time" not in public_namespace
    assert set(legacy_namespace) >= {
        "time",
        "dataclass",
        "CacheTier",
        "ChunkCache",
        "MaterializationPlan",
        "RangeReader",
        "MaterializedKV",
        "SegmentedMaterializedKV",
        "KVMaterializer",
        "normalize_segment_tiers",
    }


def test_models_public_module_owns_implementation_and_legacy_aliases_it():
    public_models = importlib.import_module("document_kv_cache.models")
    legacy_models = importlib.import_module("restaurant_kv_serving.models")

    assert public_models.DocumentChunkType.__module__ == "document_kv_cache.models"
    assert public_models.DocumentChunkRole.__module__ == "document_kv_cache.models"
    assert public_models.CacheGenerationMethod.__module__ == "document_kv_cache.models"
    assert public_models.KVCacheKey.__module__ == "document_kv_cache.models"
    assert public_models.ChunkRef.__module__ == "document_kv_cache.models"
    assert public_models.DocumentKVRequest.__module__ == "document_kv_cache.models"
    assert public_models.PlanSegment.__module__ == "document_kv_cache.models"
    assert public_models.MaterializationPlan.__module__ == "document_kv_cache.models"
    assert legacy_models.DocumentChunkType is public_models.DocumentChunkType
    assert legacy_models.DocumentChunkRole is public_models.DocumentChunkRole
    assert legacy_models.CacheGenerationMethod is public_models.CacheGenerationMethod
    assert legacy_models.KVCacheKey is public_models.KVCacheKey
    assert legacy_models.ChunkRef is public_models.ChunkRef
    assert legacy_models.DocumentKVRequest is public_models.DocumentKVRequest
    assert legacy_models.PlanSegment is public_models.PlanSegment
    assert legacy_models.MaterializationPlan is public_models.MaterializationPlan
    assert legacy_models.ChunkType is public_models.ChunkType
    assert legacy_models.RestaurantKVRequest is public_models.RestaurantKVRequest


def test_models_star_import_surfaces_are_curated_for_document_and_preserved_for_legacy():
    public_namespace: dict[str, object] = {}
    legacy_namespace: dict[str, object] = {}

    exec("from document_kv_cache.models import *", public_namespace)
    exec("from restaurant_kv_serving.models import *", legacy_namespace)

    assert set(public_namespace) >= {
        "DocumentChunkType",
        "DocumentChunkRole",
        "CacheGenerationMethod",
        "KVCacheKey",
        "ChunkRef",
        "DocumentKVRequest",
        "PlanSegment",
        "MaterializationPlan",
        "chunk_type_role",
        "chunk_type_sort_order",
        "chunk_types_for_request",
    }
    assert "RestaurantKVRequest" not in public_namespace
    assert "ChunkType" not in public_namespace
    assert "dataclass" not in public_namespace
    assert set(legacy_namespace) >= {
        "Mapping",
        "Sequence",
        "dataclass",
        "field",
        "StrEnum",
        "TypeAlias",
        "KVStorageLayout",
        "kv_storage_layout_from_value",
        "ChunkType",
        "ChunkId",
        "RestaurantKVRequest",
        "DocumentKVRequest",
        "KVCacheKey",
        "ChunkRef",
    }


def test_manifest_public_module_owns_implementation_and_legacy_aliases_it(tmp_path):
    public_manifest = importlib.import_module("document_kv_cache.manifest")
    legacy_manifest = importlib.import_module("restaurant_kv_serving.manifest")
    ref = write_kvpack(
        tmp_path / "manifest-ownership.kvpack",
        [PackChunk(make_key("doc-a", DocumentChunkType.DOCUMENT_CHUNK, "section-1"), b"body", 4, "fp8", "v1")],
        align_bytes=1,
    )[0]

    manifest = public_manifest.InMemoryManifestStore([ref])

    assert public_manifest.ManifestStore.__module__ == "document_kv_cache.manifest"
    assert public_manifest.InMemoryManifestStore.__module__ == "document_kv_cache.manifest"
    assert legacy_manifest.ManifestStore is public_manifest.ManifestStore
    assert legacy_manifest.InMemoryManifestStore is public_manifest.InMemoryManifestStore
    assert manifest.keys_for_document("doc-a") == [ref.key]
    assert manifest.keys_for_restaurant("doc-a") == [ref.key]


def test_manifest_star_import_surfaces_are_curated_for_document_and_preserved_for_legacy():
    public_namespace: dict[str, object] = {}
    legacy_namespace: dict[str, object] = {}

    exec("from document_kv_cache.manifest import *", public_namespace)
    exec("from restaurant_kv_serving.manifest import *", legacy_namespace)

    assert set(public_namespace) >= {"ManifestStore", "InMemoryManifestStore"}
    assert "Iterable" not in public_namespace
    assert "Protocol" not in public_namespace
    assert set(legacy_namespace) >= {
        "Iterable",
        "Protocol",
        "CacheChunkType",
        "ChunkRef",
        "KVCacheKey",
        "chunk_type_sort_order",
        "ManifestStore",
        "InMemoryManifestStore",
    }


def test_planner_public_module_owns_implementation_and_legacy_aliases_it(tmp_path):
    public_planner = importlib.import_module("document_kv_cache.planner")
    legacy_planner = importlib.import_module("restaurant_kv_serving.planner")
    plan = document_plan(tmp_path, (b"alpha", b"beta"))

    assert public_planner.CachePlanner.__module__ == "document_kv_cache.planner"
    assert legacy_planner.CachePlanner is public_planner.CachePlanner
    assert legacy_planner.CacheRequest is public_planner.CacheRequest
    assert plan.total_tokens == 9
    assert plan.total_bytes == 9
    assert [segment.output_byte_start for segment in plan.segments] == [0, 5]


def test_planner_star_import_surfaces_are_curated_for_document_and_preserved_for_legacy():
    public_namespace: dict[str, object] = {}
    legacy_namespace: dict[str, object] = {}

    exec("from document_kv_cache.planner import *", public_namespace)
    exec("from restaurant_kv_serving.planner import *", legacy_namespace)

    assert set(public_namespace) >= {"CacheRequest", "CachePlanner"}
    assert "ManifestStore" not in public_namespace
    assert "KVCacheKey" not in public_namespace
    assert set(legacy_namespace) >= {
        "ManifestStore",
        "DocumentKVRequest",
        "KVCacheKey",
        "MaterializationPlan",
        "PlanSegment",
        "RestaurantKVRequest",
        "chunk_types_for_request",
        "CacheRequest",
        "CachePlanner",
    }


def test_plan_segment_validates_output_positions(tmp_path):
    plan = document_plan(tmp_path, (b"abc",))
    ref = plan.segments[0].ref

    segment = PlanSegment(ref=ref, output_token_start=0, output_byte_start=0)

    assert segment.output_token_start == 0
    with pytest.raises(ValueError, match="output_token_start"):
        PlanSegment(ref=ref, output_token_start=True, output_byte_start=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="output_token_start"):
        PlanSegment(ref=ref, output_token_start=-1, output_byte_start=0)
    with pytest.raises(ValueError, match="output_byte_start"):
        PlanSegment(ref=ref, output_token_start=0, output_byte_start=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="output_byte_start"):
        PlanSegment(ref=ref, output_token_start=0, output_byte_start=-1)


def test_materialization_plan_validates_segment_cursors_and_totals(tmp_path):
    plan = document_plan(tmp_path, (b"abc", b"de"))
    first_ref = plan.segments[0].ref
    second_ref = plan.segments[1].ref

    valid = MaterializationPlan(
        request=plan.request,
        segments=plan.segments,
        total_tokens=5,
        total_bytes=5,
        selected_document_ids=("doc-a",),
    )

    assert valid.total_tokens == 5
    with pytest.raises(TypeError, match="segments"):
        MaterializationPlan(
            request=plan.request,
            segments=list(plan.segments),  # type: ignore[arg-type]
            total_tokens=5,
            total_bytes=5,
        )
    with pytest.raises(ValueError, match="total_tokens"):
        MaterializationPlan(request=plan.request, segments=plan.segments, total_tokens=True, total_bytes=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="total_bytes"):
        MaterializationPlan(request=plan.request, segments=plan.segments, total_tokens=5, total_bytes=5.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="segment token counts"):
        MaterializationPlan(request=plan.request, segments=plan.segments, total_tokens=6, total_bytes=5)
    with pytest.raises(ValueError, match="segment byte lengths"):
        MaterializationPlan(request=plan.request, segments=plan.segments, total_tokens=5, total_bytes=6)
    with pytest.raises(ValueError, match="total_bytes"):
        replace(plan, total_bytes=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="output_token_start"):
        MaterializationPlan(
            request=plan.request,
            segments=(
                PlanSegment(first_ref, output_token_start=0, output_byte_start=0),
                PlanSegment(second_ref, output_token_start=4, output_byte_start=3),
            ),
            total_tokens=5,
            total_bytes=5,
        )
    with pytest.raises(ValueError, match="output_byte_start"):
        MaterializationPlan(
            request=plan.request,
            segments=(
                PlanSegment(first_ref, output_token_start=0, output_byte_start=0),
                PlanSegment(second_ref, output_token_start=3, output_byte_start=4),
            ),
            total_tokens=5,
            total_bytes=5,
        )


def test_materializer_uses_cpu_cache_on_second_read(tmp_path):
    ref = write_kvpack(
        tmp_path / "shard.kvpack",
        [PackChunk(make_key("r1", ChunkType.REVIEW, "rev1"), b"cached", 3, "fp8", "v1")],
        align_bytes=1,
    )[0]
    manifest = InMemoryManifestStore([ref])
    plan = CachePlanner(manifest).build_plan(
        RestaurantKVRequest(
            request_id="req-1",
            task_id="selection",
            model_id="qwen35-4b-w8a8",
            lora_id="selection",
            prompt_template_version="v1",
            restaurant_reviews={"r1": ["rev1"]},
            include_static=False,
        )
    )
    cache = ChunkCache(cpu_max_bytes=1024, local_dir=tmp_path / "chunk-cache")
    materializer = KVMaterializer(cache=cache, reader=DiskRangeReader())

    assert materializer.materialize(plan).segment_tiers == (CacheTier.COLD_STORAGE,)
    assert materializer.materialize(plan).segment_tiers == (CacheTier.CPU,)
    assert cache.cold_misses == 1
    assert cache.cpu_hits == 1


def test_materializer_reports_local_disk_tier_after_cpu_eviction(tmp_path):
    ref = write_kvpack(
        tmp_path / "shard.kvpack",
        [PackChunk(make_key("r1", ChunkType.REVIEW, "rev1"), b"cached", 3, "fp8", "v1")],
        align_bytes=1,
    )[0]
    manifest = InMemoryManifestStore([ref])
    plan = CachePlanner(manifest).build_plan(
        RestaurantKVRequest(
            request_id="req-1",
            task_id="selection",
            model_id="qwen35-4b-w8a8",
            lora_id="selection",
            prompt_template_version="v1",
            restaurant_reviews={"r1": ["rev1"]},
            include_static=False,
        )
    )
    cache = ChunkCache(cpu_max_bytes=0, local_dir=tmp_path / "chunk-cache")
    materializer = KVMaterializer(cache=cache, reader=DiskRangeReader())

    assert materializer.materialize(plan).segment_tiers == (CacheTier.COLD_STORAGE,)
    assert materializer.materialize(plan).segment_tiers == (CacheTier.LOCAL_DISK,)
    assert cache.cold_misses == 1
    assert cache.local_hits == 1
    assert cache.cpu_hits == 0


def test_document_request_uses_generic_chunk_aliases(tmp_path):
    chunks = [
        PackChunk(make_key("_task", DocumentChunkType.TASK_PREFIX, "prefix"), b"task:", 5, "fp8", "v1"),
        PackChunk(make_key("doc-a", DocumentChunkType.DOCUMENT_STATIC, "static"), b"title:", 6, "fp8", "v1"),
        PackChunk(make_key("doc-a", DocumentChunkType.DOCUMENT_CHUNK, "section-1"), b"body", 4, "fp8", "v1"),
    ]
    refs = write_kvpack(tmp_path / "shard.kvpack", chunks, align_bytes=1)
    manifest = InMemoryManifestStore(refs)

    request = DocumentKVRequest(
        request_id="req-1",
        task_id="qa",
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        document_chunks={"doc-a": ["section-1"]},
        task_prefix_id="prefix",
    )

    plan = CachePlanner(manifest).build_plan(request)

    assert plan.selected_documents == ("doc-a",)
    assert plan.selected_document_ids == ("doc-a",)
    assert plan.selected_restaurants == ("doc-a",)
    assert [segment.ref.key.document_id for segment in plan.segments] == ["_task", "doc-a", "doc-a"]
    assert [segment.ref.key.chunk_id for segment in plan.segments] == ["prefix", "static", "section-1"]
    assert [segment.ref.key.chunk_type for segment in plan.segments] == [
        DocumentChunkType.TASK_PREFIX,
        DocumentChunkType.DOCUMENT_STATIC,
        DocumentChunkType.DOCUMENT_CHUNK,
    ]
    assert manifest.keys_for_document("doc-a") == [segment.ref.key for segment in plan.segments[1:]]
    assert [item.name for item in DocumentChunkType] == ["TASK_PREFIX", "DOCUMENT_STATIC", "DOCUMENT_CHUNK"]
    assert [item.value for item in DocumentChunkType] == ["task_prefix", "document_static", "document_chunk"]


def test_materialization_plan_uses_document_id_metadata_with_legacy_alias_compatibility():
    request = DocumentKVRequest(
        request_id="req-1",
        task_id="qa",
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        document_chunks={},
    )
    plan = CachePlanner(InMemoryManifestStore()).build_plan(request)

    selected_document_ids_field = next(
        field for field in fields(type(plan)) if field.name == "selected_document_ids"
    )

    assert selected_document_ids_field.default_factory is not MISSING
    assert plan.selected_document_ids == ()
    assert replace(plan, selected_document_ids=("doc-a",)).selected_documents == ("doc-a",)
    assert replace(plan, selected_restaurants=("legacy-a",)).selected_document_ids == ("legacy-a",)
