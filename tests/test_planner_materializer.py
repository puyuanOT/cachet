from restaurant_kv_serving.cache import ChunkCache
from restaurant_kv_serving.kvpack import LocalRangeReader, PackChunk, write_kvpack
from restaurant_kv_serving.manifest import InMemoryManifestStore
from restaurant_kv_serving.materializer import KVMaterializer
from restaurant_kv_serving.models import ChunkType, KVCacheKey, RestaurantKVRequest
from restaurant_kv_serving.planner import CachePlanner


def make_key(restaurant_id: str, chunk_type: ChunkType, chunk_id: str) -> KVCacheKey:
    return KVCacheKey(
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        restaurant_id=restaurant_id,
        chunk_type=chunk_type,
        chunk_id=chunk_id,
    )


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
        reader=LocalRangeReader(),
    )
    materialized = materializer.materialize(plan)

    assert plan.selected_restaurants == ("r1", "r2")
    assert plan.total_tokens == 23
    assert [segment.ref.key.chunk_id for segment in plan.segments] == ["static", "rev1", "rev2", "static", "rev9"]
    assert materialized.payload == b"menu:badgoodramen:great"
    assert materialized.segment_byte_offsets == (0, 5, 8, 12, 18)


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
    materializer = KVMaterializer(cache=cache, reader=LocalRangeReader())

    assert materializer.materialize(plan).payload == b"cached"
    assert materializer.materialize(plan).payload == b"cached"
    assert cache.cold_misses == 1
    assert cache.cpu_hits == 1

