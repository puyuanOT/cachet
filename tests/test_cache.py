import importlib
from dataclasses import replace

from document_kv_cache.cache import ByteLRU, CacheTier, ChunkCache, ChunkCacheResult, ChunkCacheStats
from document_kv_cache.kvpack import PackChunk, write_kvpack
from document_kv_cache.models import DocumentChunkType, KVCacheKey
from document_kv_cache.storage import DiskRangeReader


def key(chunk_id: str) -> KVCacheKey:
    return KVCacheKey.for_document(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_id="doc-1",
        chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
        chunk_id=chunk_id,
    )


def test_cache_tier_names_are_public_stable_values():
    assert CacheTier.CPU.value == "cpu"
    assert CacheTier.LOCAL_DISK.value == "local_disk"
    assert CacheTier.COLD_STORAGE.value == "cold_storage"


def test_byte_lru_oversized_replacement_updates_byte_accounting():
    cache = ByteLRU(max_bytes=4)

    cache.put("a", b"1234")
    cache.put("a", b"12345")

    assert cache.get("a") is None
    assert cache.current_bytes == 0
    assert len(cache) == 0


def test_chunk_cache_reports_cpu_hit_stats(tmp_path):
    ref = write_kvpack(
        tmp_path / "cpu.kvpack",
        [PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    cache = ChunkCache(cpu_max_bytes=1024)
    reader = DiskRangeReader()

    assert cache.get_or_load(ref, reader.read) == b"alpha"
    assert cache.get_or_load(ref, reader.read) == b"alpha"

    assert cache.stats() == ChunkCacheStats(
        cpu_hits=1,
        local_hits=0,
        cold_misses=1,
        cpu_items=1,
        cpu_bytes=5,
        cpu_max_bytes=1024,
        local_items=0,
        local_bytes=0,
        local_max_bytes=None,
    )


def test_chunk_cache_result_reports_serving_tier_for_cold_cpu_and_local_hits(tmp_path):
    ref = write_kvpack(
        tmp_path / "tiered.kvpack",
        [PackChunk(key=key("tiered"), payload=b"tiered", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    reader = DiskRangeReader()
    cache = ChunkCache(cpu_max_bytes=1024, local_dir=tmp_path / "chunk-cache")

    cold = cache.get_or_load_with_tier(ref, reader.read)
    cpu = cache.get_or_load_with_tier(ref, reader.read)

    assert cold == ChunkCacheResult(payload=b"tiered", tier=CacheTier.COLD_STORAGE)
    assert cpu == ChunkCacheResult(payload=b"tiered", tier=CacheTier.CPU)
    assert cache.stats().cold_misses == 1
    assert cache.stats().cpu_hits == 1

    local_only_cache = ChunkCache(cpu_max_bytes=0, local_dir=tmp_path / "chunk-cache")
    local = local_only_cache.get_or_load_with_tier(ref, reader.read)

    assert local == ChunkCacheResult(payload=b"tiered", tier=CacheTier.LOCAL_DISK)
    assert local_only_cache.stats().local_hits == 1
    assert local_only_cache.stats().cold_misses == 0


def test_chunk_cache_uses_local_disk_when_payload_is_too_large_for_cpu(tmp_path):
    ref = write_kvpack(
        tmp_path / "local-hit.kvpack",
        [PackChunk(key=key("a"), payload=b"local", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    cache = ChunkCache(cpu_max_bytes=1, local_dir=tmp_path / "chunk-cache")
    reader = DiskRangeReader()

    assert cache.get_or_load(ref, reader.read) == b"local"
    assert cache.get_or_load(ref, reader.read) == b"local"

    stats = cache.stats()
    assert stats.cpu_hits == 0
    assert stats.local_hits == 1
    assert stats.cold_misses == 1
    assert stats.local_items == 1
    assert stats.local_bytes == len(b"local")


def test_chunk_cache_enforces_local_disk_budget_with_lru_eviction(tmp_path):
    refs = write_kvpack(
        tmp_path / "local-budget.kvpack",
        [
            PackChunk(key=key("small"), payload=b"aa", token_count=1, dtype="int8", layout_version="v1"),
            PackChunk(key=key("large"), payload=b"bbbb", token_count=2, dtype="int8", layout_version="v1"),
        ],
        align_bytes=1,
    )
    cache = ChunkCache(cpu_max_bytes=0, local_dir=tmp_path / "chunk-cache", local_max_bytes=4)
    reader = DiskRangeReader()

    assert cache.get_or_load(refs[0], reader.read) == b"aa"
    assert cache.get_or_load(refs[1], reader.read) == b"bbbb"

    stats = cache.stats()
    assert stats.local_items == 1
    assert stats.local_bytes == 4
    assert stats.local_max_bytes == 4
    assert len(list((tmp_path / "chunk-cache").rglob("*.chunk"))) == 1

    assert cache.get_or_load(refs[0], reader.read) == b"aa"
    assert cache.stats().cold_misses == 3


def test_chunk_cache_local_path_includes_checksum_to_avoid_stale_reuse(tmp_path):
    same_key = key("same")
    refs = write_kvpack(
        tmp_path / "same-key.kvpack",
        [
            PackChunk(key=same_key, payload=b"old", token_count=1, dtype="int8", layout_version="v1"),
            PackChunk(key=same_key, payload=b"new", token_count=1, dtype="int8", layout_version="v1"),
        ],
        align_bytes=1,
    )
    second_ref_same_logical_key = replace(refs[1], key=refs[0].key)
    cache = ChunkCache(cpu_max_bytes=1024, local_dir=tmp_path / "chunk-cache")
    reader = DiskRangeReader()

    assert cache.get_or_load(refs[0], reader.read) == b"old"
    assert cache.get_or_load(second_ref_same_logical_key, reader.read) == b"new"

    stats = cache.stats()
    assert stats.cold_misses == 2
    assert stats.cpu_hits == 0
    assert stats.local_hits == 0
    assert stats.local_items == 2


def test_chunk_cache_reloads_corrupted_local_file(tmp_path):
    ref = write_kvpack(
        tmp_path / "corrupt-local.kvpack",
        [PackChunk(key=key("a"), payload=b"valid", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    local_dir = tmp_path / "chunk-cache"
    reader = DiskRangeReader()
    first_cache = ChunkCache(cpu_max_bytes=0, local_dir=local_dir)

    assert first_cache.get_or_load(ref, reader.read) == b"valid"
    local_path = next(local_dir.rglob("*.chunk"))
    local_path.write_bytes(b"bad")

    second_cache = ChunkCache(cpu_max_bytes=0, local_dir=local_dir)

    assert second_cache.get_or_load(ref, reader.read) == b"valid"
    assert local_path.read_bytes() == b"valid"
    assert second_cache.stats().local_hits == 0
    assert second_cache.stats().cold_misses == 1


def test_cache_public_module_owns_implementation_and_legacy_aliases_it():
    public_cache = importlib.import_module("document_kv_cache.cache")
    legacy_cache = importlib.import_module("restaurant_kv_serving.cache")

    assert public_cache.CacheTier.__module__ == "document_kv_cache.cache"
    assert public_cache.ByteLRU.__module__ == "document_kv_cache.cache"
    assert public_cache.ChunkCache.__module__ == "document_kv_cache.cache"
    assert legacy_cache.CacheTier is public_cache.CacheTier
    assert legacy_cache.ChunkCacheResult is public_cache.ChunkCacheResult
    assert legacy_cache.ChunkCacheStats is public_cache.ChunkCacheStats
    assert legacy_cache.ByteLRU is public_cache.ByteLRU
    assert legacy_cache.ChunkCache is public_cache.ChunkCache


def test_cache_star_import_surfaces_are_curated_for_document_and_preserved_for_legacy():
    public_namespace: dict[str, object] = {}
    legacy_namespace: dict[str, object] = {}

    exec("from document_kv_cache.cache import *", public_namespace)
    exec("from restaurant_kv_serving.cache import *", legacy_namespace)

    assert set(public_namespace) >= {"CacheTier", "ChunkCacheResult", "ChunkCacheStats", "ByteLRU", "ChunkCache"}
    assert "hashlib" not in public_namespace
    assert set(legacy_namespace) >= {
        "hashlib",
        "OrderedDict",
        "Callable",
        "dataclass",
        "StrEnum",
        "Path",
        "ChunkRef",
        "CacheTier",
        "ChunkCacheResult",
        "ChunkCacheStats",
        "ByteLRU",
        "ChunkCache",
    }
