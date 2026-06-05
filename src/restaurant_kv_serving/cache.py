from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from restaurant_kv_serving.models import ChunkRef


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
            self._items.pop(key, None)
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

    def __init__(self, *, cpu_max_bytes: int, local_dir: str | Path | None = None) -> None:
        self.cpu = ByteLRU(cpu_max_bytes)
        self.local_dir = Path(local_dir) if local_dir is not None else None
        if self.local_dir is not None:
            self.local_dir.mkdir(parents=True, exist_ok=True)
        self.cpu_hits = 0
        self.local_hits = 0
        self.cold_misses = 0

    def get_or_load(self, ref: ChunkRef, loader: Callable[[ChunkRef], bytes]) -> bytes:
        key = ref.key.storage_key()
        cached = self.cpu.get(key)
        if cached is not None:
            self.cpu_hits += 1
            return cached

        local_path = self._local_path(ref)
        if local_path is not None and local_path.exists():
            payload = local_path.read_bytes()
            self.local_hits += 1
            self.cpu.put(key, payload)
            return payload

        payload = loader(ref)
        self.cold_misses += 1
        if local_path is not None:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(payload)
        self.cpu.put(key, payload)
        return payload

    def _local_path(self, ref: ChunkRef) -> Path | None:
        if self.local_dir is None:
            return None
        digest = hashlib.sha256(ref.key.storage_key().encode("utf-8")).hexdigest()
        return self.local_dir / digest[:2] / f"{digest}.chunk"

