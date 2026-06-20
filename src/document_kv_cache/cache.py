"""Tiered byte caches for materialized KV chunks."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
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
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self._items: OrderedDict[str, bytes] = OrderedDict()

    def get(self, key: str) -> bytes | None:
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key: str, value: bytes) -> None:
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
        if local_max_bytes is not None and local_max_bytes < 0:
            raise ValueError("local_max_bytes must be non-negative")
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

        payload = loader(ref)
        self.cold_misses += 1
        if local_path is not None:
            self._write_local(local_path, payload)
        self.cpu.put(key, payload)
        return ChunkCacheResult(payload=payload, tier=CacheTier.COLD_STORAGE)

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
