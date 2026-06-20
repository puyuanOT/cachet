"""Tiered byte caches for materialized KV chunks."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from document_kv_cache.models import ChunkRef

__all__ = ["CacheTier", "ChunkCacheResult", "ChunkCacheStats", "ByteLRU", "ChunkCache"]


class CacheTier(StrEnum):
    CPU = "cpu"
    LOCAL_DISK = "local_disk"
    COLD_STORAGE = "cold_storage"


@dataclass(frozen=True, slots=True)
class ChunkCacheStats:
    cpu_hits: int
    local_hits: int
    cold_misses: int
    cpu_items: int
    cpu_bytes: int
    cpu_max_bytes: int
    local_items: int
    local_bytes: int
    local_max_bytes: int | None


@dataclass(frozen=True, slots=True)
class ChunkCacheResult:
    payload: bytes
    tier: CacheTier


class ByteLRU:
    def __init__(self, max_bytes: int) -> None:
        _validate_non_negative_int("max_bytes", max_bytes)
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self._items: OrderedDict[str, bytes] = OrderedDict()

    def get(self, key: str) -> bytes | None:
        key = _validate_cache_key(key)
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value

    def peek(self, key: str) -> bytes | None:
        key = _validate_cache_key(key)
        return self._items.get(key)

    def put(self, key: str, value: bytes) -> None:
        key = _validate_cache_key(key)
        value = _coerce_cache_payload(value, label="cache value")
        if len(value) > self.max_bytes:
            old = self._items.pop(key, None)
            if old is not None:
                self.current_bytes -= len(old)
            return
        old = self._items.pop(key, None)
        if old is not None:
            self.current_bytes -= len(old)
        self._items[key] = value
        self.current_bytes += len(value)
        while self.current_bytes > self.max_bytes and self._items:
            _, evicted = self._items.popitem(last=False)
            self.current_bytes -= len(evicted)

    def __len__(self) -> int:
        return len(self._items)


class ChunkCache:
    """Two-tier byte cache: CPU RAM LRU backed by local NVMe chunk files."""

    def __init__(
        self,
        *,
        cpu_max_bytes: int,
        local_dir: str | Path | None = None,
        local_max_bytes: int | None = None,
    ) -> None:
        if local_max_bytes is not None:
            _validate_non_negative_int("local_max_bytes", local_max_bytes)
        self.cpu = ByteLRU(cpu_max_bytes)
        self.local_dir = Path(local_dir) if local_dir is not None else None
        self.local_max_bytes = local_max_bytes
        self._local_index: OrderedDict[Path, int] = OrderedDict()
        self._local_bytes = 0
        if self.local_dir is not None:
            self.local_dir.mkdir(parents=True, exist_ok=True)
            self._load_local_index()
            self._enforce_local_budget()
        self.cpu_hits = 0
        self.local_hits = 0
        self.cold_misses = 0

    def get_or_load(self, ref: ChunkRef, loader: Callable[[ChunkRef], bytes]) -> bytes:
        return self.get_or_load_with_tier(ref, loader).payload

    def get_or_load_with_tier(self, ref: ChunkRef, loader: Callable[[ChunkRef], bytes]) -> ChunkCacheResult:
        cached = self._get_cached_result(ref)
        if cached is not None:
            return cached
        key = self._cache_key(ref)
        local_path = self._local_path(ref)
        payload = _coerce_cache_payload(loader(ref), label="loader payload")
        self.cold_misses += 1
        if local_path is not None:
            self._write_local(local_path, payload)
        self.cpu.put(key, payload)
        return ChunkCacheResult(payload=payload, tier=CacheTier.COLD_STORAGE)

    def get_many_or_load_with_tier(
        self,
        refs: Sequence[ChunkRef],
        loader: Callable[[ChunkRef], bytes],
        *,
        batch_loader: Callable[[Sequence[ChunkRef]], Sequence[bytes]] | None = None,
    ) -> tuple[ChunkCacheResult, ...]:
        results: list[ChunkCacheResult] = []
        index = 0
        while index < len(refs):
            ref = refs[index]
            cached = self._get_cached_result(ref)
            if cached is not None:
                results.append(cached)
                index += 1
                continue

            cold_refs = [ref]
            index += 1
            while index < len(refs) and not self._may_have_cached_payload(refs[index]):
                cold_refs.append(refs[index])
                index += 1
            results.extend(self._load_cold_refs(cold_refs, loader, batch_loader=batch_loader))
        return tuple(results)

    def stats(self) -> ChunkCacheStats:
        return ChunkCacheStats(
            cpu_hits=self.cpu_hits,
            local_hits=self.local_hits,
            cold_misses=self.cold_misses,
            cpu_items=len(self.cpu),
            cpu_bytes=self.cpu.current_bytes,
            cpu_max_bytes=self.cpu.max_bytes,
            local_items=len(self._local_index),
            local_bytes=self._local_bytes,
            local_max_bytes=self.local_max_bytes,
        )

    def _get_cached_result(self, ref: ChunkRef) -> ChunkCacheResult | None:
        key = self._cache_key(ref)
        cached = self.cpu.get(key)
        if cached is not None:
            self.cpu_hits += 1
            return ChunkCacheResult(payload=cached, tier=CacheTier.CPU)

        local_path = self._local_path(ref)
        if local_path is not None and local_path.exists():
            payload = local_path.read_bytes()
            if self._is_valid_payload(ref, payload):
                self.local_hits += 1
                self._record_local_access(local_path)
                self.cpu.put(key, payload)
                return ChunkCacheResult(payload=payload, tier=CacheTier.LOCAL_DISK)
            self._remove_local(local_path)
        return None

    def _may_have_cached_payload(self, ref: ChunkRef) -> bool:
        key = self._cache_key(ref)
        if self.cpu.peek(key) is not None:
            return True
        local_path = self._local_path(ref)
        if local_path is None or not local_path.exists():
            return False
        return True

    def _load_cold_refs(
        self,
        refs: Sequence[ChunkRef],
        loader: Callable[[ChunkRef], bytes],
        *,
        batch_loader: Callable[[Sequence[ChunkRef]], Sequence[bytes]] | None,
    ) -> tuple[ChunkCacheResult, ...]:
        if not refs:
            return ()
        if batch_loader is None:
            return tuple(self.get_or_load_with_tier(ref, loader) for ref in refs)

        unique_refs = _deduplicate_refs_by_cache_key(refs, self._cache_key)
        payloads = tuple(
            _coerce_cache_payload(payload, label="batch_loader payload")
            for payload in batch_loader(unique_refs)
        )
        if len(payloads) != len(unique_refs):
            raise ValueError("batch_loader returned the wrong number of payloads")
        payload_by_cache_key = {
            self._cache_key(ref): payload for ref, payload in zip(unique_refs, payloads, strict=True)
        }
        results: list[ChunkCacheResult] = []
        for ref in refs:
            cached = self._get_cached_result(ref)
            if cached is not None:
                results.append(cached)
                continue
            key = self._cache_key(ref)
            payload = payload_by_cache_key[key]
            local_path = self._local_path(ref)
            self.cold_misses += 1
            if local_path is not None:
                self._write_local(local_path, payload)
            self.cpu.put(key, payload)
            results.append(ChunkCacheResult(payload=payload, tier=CacheTier.COLD_STORAGE))
        return tuple(results)

    def _local_path(self, ref: ChunkRef) -> Path | None:
        if self.local_dir is None:
            return None
        digest = hashlib.sha256(self._cache_key(ref).encode("utf-8")).hexdigest()
        return self.local_dir / digest[:2] / f"{digest}.chunk"

    @staticmethod
    def _cache_key(ref: ChunkRef) -> str:
        return f"{ref.key.storage_key()}:{ref.checksum}"

    def _load_local_index(self) -> None:
        if self.local_dir is None:
            return
        files: list[tuple[int, Path, int]] = []
        for path in self.local_dir.rglob("*.chunk"):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            files.append((stat.st_mtime_ns, path, stat.st_size))
        for _, path, size in sorted(files):
            self._local_index[path] = size
            self._local_bytes += size

    def _record_local_access(self, path: Path) -> None:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            old_size = self._local_index.pop(path, 0)
            self._local_bytes -= old_size
            return
        old_size = self._local_index.pop(path, 0)
        self._local_bytes += size - old_size
        self._local_index[path] = size
        path.touch()

    def _write_local(self, path: Path, payload: bytes) -> None:
        if self.local_max_bytes is not None and len(payload) > self.local_max_bytes:
            old_size = self._local_index.pop(path, 0)
            self._local_bytes -= old_size
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self._record_local_access(path)
        self._enforce_local_budget()

    def _remove_local(self, path: Path) -> None:
        old_size = self._local_index.pop(path, 0)
        self._local_bytes -= old_size
        path.unlink(missing_ok=True)

    def _enforce_local_budget(self) -> None:
        if self.local_max_bytes is None:
            return
        while self._local_bytes > self.local_max_bytes and self._local_index:
            path, size = self._local_index.popitem(last=False)
            self._local_bytes -= size
            path.unlink(missing_ok=True)

    @staticmethod
    def _is_valid_payload(ref: ChunkRef, payload: bytes) -> bool:
        if len(payload) != ref.byte_length:
            return False
        return hashlib.sha256(payload).hexdigest() == ref.checksum


def _deduplicate_refs_by_cache_key(
    refs: Sequence[ChunkRef],
    cache_key: Callable[[ChunkRef], str],
) -> tuple[ChunkRef, ...]:
    seen: set[str] = set()
    unique_refs: list[ChunkRef] = []
    for ref in refs:
        key = cache_key(ref)
        if key in seen:
            continue
        seen.add(key)
        unique_refs.append(ref)
    return tuple(unique_refs)


def _validate_non_negative_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_cache_key(key: object) -> str:
    if not isinstance(key, str) or not key:
        raise ValueError("cache key must be non-empty")
    return key


def _coerce_cache_payload(payload: object, *, label: str) -> bytes:
    if isinstance(payload, bytes):
        return payload
    try:
        return memoryview(payload).tobytes()
    except TypeError as exc:
        raise ValueError(f"{label} must be bytes-like") from exc
