import importlib
from dataclasses import replace

import pytest

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


@pytest.mark.parametrize(
    ("max_bytes", "message"),
    (
        (-1, "max_bytes must be non-negative"),
        (True, "max_bytes must be a non-negative integer"),
        (4.0, "max_bytes must be a non-negative integer"),
    ),
)
def test_byte_lru_rejects_invalid_byte_budgets(max_bytes, message):
    with pytest.raises(ValueError, match=message):
        ByteLRU(max_bytes=max_bytes)


def test_byte_lru_copies_bytes_like_values_to_immutable_bytes():
    payload = bytearray(b"alpha")
    cache = ByteLRU(max_bytes=64)

    cache.put("a", payload)
    payload[:] = b"xxxxx"

    cached = cache.get("a")
    assert cached == b"alpha"
    assert type(cached) is bytes


@pytest.mark.parametrize(
    ("key_value", "payload", "message"),
    (
        ("", b"payload", "cache key must be non-empty"),
        (123, b"payload", "cache key must be non-empty"),
        ("a", 3, "cache value must be bytes-like"),
    ),
)
def test_byte_lru_rejects_invalid_keys_and_payloads(key_value, payload, message):
    cache = ByteLRU(max_bytes=64)

    with pytest.raises(ValueError, match=message):
        cache.put(key_value, payload)


@pytest.mark.parametrize("lookup", ("get", "peek"))
def test_byte_lru_rejects_invalid_lookup_keys(lookup):
    cache = ByteLRU(max_bytes=64)

    with pytest.raises(ValueError, match="cache key must be non-empty"):
        getattr(cache, lookup)("")


def test_chunk_cache_rejects_invalid_local_disk_budget(tmp_path):
    with pytest.raises(ValueError, match="local_max_bytes must be a non-negative integer"):
        ChunkCache(cpu_max_bytes=0, local_dir=tmp_path / "chunk-cache", local_max_bytes=True)


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


def test_chunk_cache_normalizes_cold_loader_payload_to_immutable_bytes(tmp_path):
    mutable_payload = bytearray(b"alpha")
    ref = write_kvpack(
        tmp_path / "loader-normalized.kvpack",
        [
            PackChunk(
                key=key("loader"),
                payload=bytes(mutable_payload),
                token_count=2,
                dtype="int8",
                layout_version="v1",
            )
        ],
        align_bytes=1,
    )[0]
    cache = ChunkCache(cpu_max_bytes=1024, local_dir=tmp_path / "chunk-cache")

    result = cache.get_or_load_with_tier(ref, lambda _: mutable_payload)
    mutable_payload[:] = b"xxxxx"
    cached = cache.get_or_load_with_tier(ref, lambda _: b"unused")

    assert result == ChunkCacheResult(payload=b"alpha", tier=CacheTier.COLD_STORAGE)
    assert type(result.payload) is bytes
    assert cached == ChunkCacheResult(payload=b"alpha", tier=CacheTier.CPU)
    assert next((tmp_path / "chunk-cache").rglob("*.chunk")).read_bytes() == b"alpha"


def test_chunk_cache_rejects_non_bytes_like_cold_loader_payload(tmp_path):
    ref = write_kvpack(
        tmp_path / "bad-loader.kvpack",
        [PackChunk(key=key("loader"), payload=b"alpha", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    cache = ChunkCache(cpu_max_bytes=1024)

    with pytest.raises(ValueError, match="loader payload must be bytes-like"):
        cache.get_or_load_with_tier(ref, lambda _: 3)


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


def test_chunk_cache_get_many_batches_cold_loads_and_reuses_cpu_hits(tmp_path):
    refs = write_kvpack(
        tmp_path / "batch-cache.kvpack",
        [
            PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="int8", layout_version="v1"),
            PackChunk(key=key("b"), payload=b"beta", token_count=2, dtype="int8", layout_version="v1"),
        ],
        align_bytes=1,
    )
    reader = DiskRangeReader()
    cache = ChunkCache(cpu_max_bytes=1024)
    single_calls: list[str] = []
    batch_calls: list[tuple[str, ...]] = []

    def loader(ref):
        single_calls.append(ref.key.chunk_id)
        return reader.read(ref)

    def batch_loader(batch):
        batch_calls.append(tuple(ref.key.chunk_id for ref in batch))
        return reader.read_many(batch)

    cold_results = cache.get_many_or_load_with_tier(refs, loader, batch_loader=batch_loader)
    cpu_results = cache.get_many_or_load_with_tier(refs, loader, batch_loader=batch_loader)

    assert [result.payload for result in cold_results] == [b"alpha", b"beta"]
    assert [result.tier for result in cold_results] == [CacheTier.COLD_STORAGE, CacheTier.COLD_STORAGE]
    assert [result.tier for result in cpu_results] == [CacheTier.CPU, CacheTier.CPU]
    assert single_calls == []
    assert batch_calls == [("a", "b")]
    assert cache.stats().cold_misses == 2
    assert cache.stats().cpu_hits == 2


def test_chunk_cache_get_many_deduplicates_cold_refs_within_one_batch(tmp_path):
    refs = write_kvpack(
        tmp_path / "batch-cache-duplicate.kvpack",
        [
            PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="int8", layout_version="v1"),
            PackChunk(key=key("b"), payload=b"beta", token_count=2, dtype="int8", layout_version="v1"),
        ],
        align_bytes=1,
    )
    reader = DiskRangeReader()
    cache = ChunkCache(cpu_max_bytes=1024)
    batch_calls: list[tuple[str, ...]] = []

    def batch_loader(batch):
        batch_calls.append(tuple(ref.key.chunk_id for ref in batch))
        return reader.read_many(batch)

    results = cache.get_many_or_load_with_tier(
        (refs[0], refs[0], refs[1], refs[0]),
        reader.read,
        batch_loader=batch_loader,
    )

    assert [result.payload for result in results] == [b"alpha", b"alpha", b"beta", b"alpha"]
    assert [result.tier for result in results] == [
        CacheTier.COLD_STORAGE,
        CacheTier.CPU,
        CacheTier.COLD_STORAGE,
        CacheTier.CPU,
    ]
    assert batch_calls == [("a", "b")]
    assert cache.stats().cold_misses == 2
    assert cache.stats().cpu_hits == 2


def test_chunk_cache_get_many_normalizes_batch_loader_payloads_to_immutable_bytes(tmp_path):
    refs = write_kvpack(
        tmp_path / "batch-normalized.kvpack",
        [
            PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="int8", layout_version="v1"),
            PackChunk(key=key("b"), payload=b"beta", token_count=2, dtype="int8", layout_version="v1"),
        ],
        align_bytes=1,
    )
    payloads = [bytearray(b"alpha"), bytearray(b"beta")]
    cache = ChunkCache(cpu_max_bytes=1024)

    results = cache.get_many_or_load_with_tier(
        refs,
        lambda ref: b"unused",
        batch_loader=lambda batch: payloads,
    )
    payloads[0][:] = b"xxxxx"
    payloads[1][:] = b"yyyy"
    cached = cache.get_many_or_load_with_tier(refs, lambda ref: b"unused")

    assert [result.payload for result in results] == [b"alpha", b"beta"]
    assert [type(result.payload) for result in results] == [bytes, bytes]
    assert [result.payload for result in cached] == [b"alpha", b"beta"]


def test_chunk_cache_get_many_rejects_non_bytes_like_batch_loader_payload(tmp_path):
    ref = write_kvpack(
        tmp_path / "bad-batch-loader.kvpack",
        [PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    cache = ChunkCache(cpu_max_bytes=1024)

    with pytest.raises(ValueError, match="batch_loader payload must be bytes-like"):
        cache.get_many_or_load_with_tier((ref,), lambda item: b"unused", batch_loader=lambda batch: (3,))


def test_chunk_cache_get_many_duplicate_tiers_follow_configured_cache_capacity(tmp_path):
    ref = write_kvpack(
        tmp_path / "batch-cache-duplicate-local.kvpack",
        [PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    reader = DiskRangeReader()
    cache = ChunkCache(cpu_max_bytes=0, local_dir=tmp_path / "chunk-cache")
    batch_calls: list[tuple[str, ...]] = []

    def batch_loader(batch):
        batch_calls.append(tuple(ref.key.chunk_id for ref in batch))
        return reader.read_many(batch)

    results = cache.get_many_or_load_with_tier((ref, ref), reader.read, batch_loader=batch_loader)

    assert [result.payload for result in results] == [b"alpha", b"alpha"]
    assert [result.tier for result in results] == [CacheTier.COLD_STORAGE, CacheTier.LOCAL_DISK]
    assert batch_calls == [("a",)]
    assert cache.stats().cold_misses == 1
    assert cache.stats().local_hits == 1
    assert cache.stats().cpu_hits == 0


def test_chunk_cache_get_many_does_not_pre_read_local_hit_while_grouping(tmp_path, monkeypatch):
    refs = write_kvpack(
        tmp_path / "batch-cache-local-boundary.kvpack",
        [
            PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="int8", layout_version="v1"),
            PackChunk(key=key("b"), payload=b"beta", token_count=2, dtype="int8", layout_version="v1"),
        ],
        align_bytes=1,
    )
    reader = DiskRangeReader()
    cache_dir = tmp_path / "chunk-cache"
    cache = ChunkCache(cpu_max_bytes=0, local_dir=cache_dir)
    assert cache.get_or_load_with_tier(refs[1], reader.read).tier == CacheTier.COLD_STORAGE
    cache = ChunkCache(cpu_max_bytes=0, local_dir=cache_dir)
    local_read_count = 0
    path_read_bytes = type(cache_dir).read_bytes

    def count_chunk_reads(path):
        nonlocal local_read_count
        if path.suffix == ".chunk":
            local_read_count += 1
        return path_read_bytes(path)

    monkeypatch.setattr(type(cache_dir), "read_bytes", count_chunk_reads)

    results = cache.get_many_or_load_with_tier((refs[0], refs[1]), reader.read, batch_loader=reader.read_many)

    assert [result.tier for result in results] == [CacheTier.COLD_STORAGE, CacheTier.LOCAL_DISK]
    assert [result.payload for result in results] == [b"alpha", b"beta"]
    assert local_read_count == 1
    assert cache.stats().local_hits == 1
    assert cache.stats().cold_misses == 1


def test_chunk_cache_get_many_reloads_corrupted_local_boundary(tmp_path):
    refs = write_kvpack(
        tmp_path / "batch-cache-corrupt-local-boundary.kvpack",
        [
            PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="int8", layout_version="v1"),
            PackChunk(key=key("b"), payload=b"beta", token_count=2, dtype="int8", layout_version="v1"),
        ],
        align_bytes=1,
    )
    reader = DiskRangeReader()
    cache_dir = tmp_path / "chunk-cache"
    first_cache = ChunkCache(cpu_max_bytes=0, local_dir=cache_dir)
    assert first_cache.get_or_load_with_tier(refs[1], reader.read).tier == CacheTier.COLD_STORAGE
    next(cache_dir.rglob("*.chunk")).write_bytes(b"bad")
    second_cache = ChunkCache(cpu_max_bytes=0, local_dir=cache_dir)
    batch_calls: list[tuple[str, ...]] = []

    def batch_loader(batch):
        batch_calls.append(tuple(ref.key.chunk_id for ref in batch))
        return reader.read_many(batch)

    results = second_cache.get_many_or_load_with_tier((refs[0], refs[1]), reader.read, batch_loader=batch_loader)

    assert [result.payload for result in results] == [b"alpha", b"beta"]
    assert [result.tier for result in results] == [CacheTier.COLD_STORAGE, CacheTier.COLD_STORAGE]
    assert batch_calls == [("a",), ("b",)]
    assert second_cache.stats().local_hits == 0
    assert second_cache.stats().cold_misses == 2


def test_chunk_cache_get_many_preserves_interleaved_local_eviction_order(tmp_path):
    refs = write_kvpack(
        tmp_path / "batch-cache-interleaved-eviction.kvpack",
        [
            PackChunk(key=key("a"), payload=b"aaa", token_count=2, dtype="int8", layout_version="v1"),
            PackChunk(key=key("b"), payload=b"bbbb", token_count=2, dtype="int8", layout_version="v1"),
        ],
        align_bytes=1,
    )
    reader = DiskRangeReader()
    sequence = (refs[0], refs[1], refs[0])

    sequential_cache = ChunkCache(cpu_max_bytes=0, local_dir=tmp_path / "sequential-cache", local_max_bytes=4)
    assert sequential_cache.get_or_load_with_tier(refs[0], reader.read).tier == CacheTier.COLD_STORAGE
    sequential_cache = ChunkCache(cpu_max_bytes=0, local_dir=tmp_path / "sequential-cache", local_max_bytes=4)
    sequential_results = [sequential_cache.get_or_load_with_tier(ref, reader.read) for ref in sequence]

    batch_cache = ChunkCache(cpu_max_bytes=0, local_dir=tmp_path / "batch-cache", local_max_bytes=4)
    assert batch_cache.get_or_load_with_tier(refs[0], reader.read).tier == CacheTier.COLD_STORAGE
    batch_cache = ChunkCache(cpu_max_bytes=0, local_dir=tmp_path / "batch-cache", local_max_bytes=4)
    batch_calls: list[tuple[str, ...]] = []

    def batch_loader(batch):
        batch_calls.append(tuple(ref.key.chunk_id for ref in batch))
        return reader.read_many(batch)

    batch_results = batch_cache.get_many_or_load_with_tier(sequence, reader.read, batch_loader=batch_loader)

    assert [result.payload for result in batch_results] == [result.payload for result in sequential_results]
    assert [result.tier for result in batch_results] == [CacheTier.LOCAL_DISK, CacheTier.COLD_STORAGE, CacheTier.COLD_STORAGE]
    assert [result.tier for result in batch_results] == [result.tier for result in sequential_results]
    assert batch_cache.stats().local_hits == sequential_cache.stats().local_hits == 1
    assert batch_cache.stats().cold_misses == sequential_cache.stats().cold_misses == 2
    assert batch_calls == [("b",), ("a",)]


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
