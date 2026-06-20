from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from document_kv_cache.models import CacheChunkType, ChunkRef, KVCacheKey, chunk_type_sort_order


__all__ = ["ManifestStore", "InMemoryManifestStore"]


class ManifestStore(Protocol):
    def get(self, key: KVCacheKey) -> ChunkRef: ...

    def put_many(self, refs: Iterable[ChunkRef]) -> None: ...

    def keys_for_document(self, document_id: str, chunk_type: CacheChunkType | None = None) -> list[KVCacheKey]: ...


class InMemoryManifestStore:
    """Small test/dev manifest implementation.

    Production should back this with a table keyed by model/lora/template,
    document id, chunk type, and chunk id.
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

    def keys_for_document(self, document_id: str, chunk_type: CacheChunkType | None = None) -> list[KVCacheKey]:
        keys = [key for key in self._refs if key.document_id == document_id]
        if chunk_type is not None:
            keys = [key for key in keys if key.chunk_type == chunk_type]
        return sorted(
            keys,
            key=lambda item: (chunk_type_sort_order(item.chunk_type), item.chunk_type.value, item.chunk_id),
        )

    def keys_for_restaurant(self, restaurant_id: str, chunk_type: CacheChunkType | None = None) -> list[KVCacheKey]:
        return self.keys_for_document(restaurant_id, chunk_type)
