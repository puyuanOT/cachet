from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from restaurant_kv_serving.models import ChunkRef, ChunkType, KVCacheKey


class ManifestStore(Protocol):
    def get(self, key: KVCacheKey) -> ChunkRef: ...

    def put_many(self, refs: Iterable[ChunkRef]) -> None: ...


class InMemoryManifestStore:
    """Small test/dev manifest implementation.

    Production should back this with a Delta table keyed by model/lora/template,
    restaurant id, chunk type, and chunk id.
    """

    def __init__(self, refs: Iterable[ChunkRef] = ()) -> None:
        self._refs: dict[KVCacheKey, ChunkRef] = {}
        self.put_many(refs)

    def get(self, key: KVCacheKey) -> ChunkRef:
        try:
            return self._refs[key]
        except KeyError as exc:
            raise KeyError(f"Missing manifest entry for {key.storage_key()}") from exc

    def put_many(self, refs: Iterable[ChunkRef]) -> None:
        for ref in refs:
            self._refs[ref.key] = ref

    def keys_for_restaurant(self, restaurant_id: str, chunk_type: ChunkType | None = None) -> list[KVCacheKey]:
        keys = [key for key in self._refs if key.restaurant_id == restaurant_id]
        if chunk_type is not None:
            keys = [key for key in keys if key.chunk_type == chunk_type]
        return sorted(keys, key=lambda item: (item.chunk_type.value, item.chunk_id))

