from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from document_kv_cache.models import (
    CacheChunkType,
    ChunkRef,
    ChunkType,
    DocumentChunkType,
    KVCacheKey,
    chunk_type_sort_order,
)


__all__ = ["ManifestStore", "InMemoryManifestStore"]


class ManifestStore(Protocol):
    def get(self, key: KVCacheKey) -> ChunkRef: ...

    def put_many(self, refs: Iterable[ChunkRef]) -> None: ...

    def keys_for_document(self, document_id: str, chunk_type: CacheChunkType | str | None = None) -> list[KVCacheKey]: ...


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
        refs_tuple = _chunk_refs_tuple(refs)
        duplicate_keys = _duplicate_manifest_keys(refs_tuple)
        if duplicate_keys:
            raise ValueError("manifest refs contain duplicate cache keys: " + ", ".join(duplicate_keys))
        for ref in refs_tuple:
            self._refs[ref.key] = ref

    def keys_for_document(self, document_id: str, chunk_type: CacheChunkType | str | None = None) -> list[KVCacheKey]:
        _validate_document_id(document_id)
        chunk_type = _normalize_chunk_type_filter(chunk_type)
        keys = [key for key in self._refs if key.document_id == document_id]
        if chunk_type is not None:
            keys = [key for key in keys if key.chunk_type == chunk_type]
        return sorted(
            keys,
            key=lambda item: (chunk_type_sort_order(item.chunk_type), item.chunk_type.value, item.chunk_id),
        )

    def keys_for_restaurant(self, restaurant_id: str, chunk_type: CacheChunkType | str | None = None) -> list[KVCacheKey]:
        return self.keys_for_document(restaurant_id, chunk_type)


def _chunk_refs_tuple(refs: Iterable[ChunkRef]) -> tuple[ChunkRef, ...]:
    if isinstance(refs, (str, bytes, bytearray)):
        raise TypeError("refs must be an iterable of ChunkRef instances")
    refs_tuple = tuple(refs)
    for ref in refs_tuple:
        if not isinstance(ref, ChunkRef):
            raise TypeError("refs entries must be ChunkRef instances")
    return refs_tuple


def _duplicate_manifest_keys(refs: tuple[ChunkRef, ...]) -> tuple[str, ...]:
    seen: set[KVCacheKey] = set()
    duplicates: list[str] = []
    duplicate_storage_keys: set[str] = set()
    for ref in refs:
        key = ref.key
        if key not in seen:
            seen.add(key)
            continue
        storage_key = key.storage_key()
        if storage_key not in duplicate_storage_keys:
            duplicates.append(storage_key)
            duplicate_storage_keys.add(storage_key)
    return tuple(duplicates)


def _validate_document_id(document_id: object) -> None:
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be non-empty")
    if "|" in document_id:
        raise ValueError("document_id must not contain '|'")


def _normalize_chunk_type_filter(chunk_type: CacheChunkType | str | None) -> CacheChunkType | None:
    if chunk_type is None or isinstance(chunk_type, (ChunkType, DocumentChunkType)):
        return chunk_type
    if isinstance(chunk_type, str):
        for enum_type in (DocumentChunkType, ChunkType):
            try:
                return enum_type(chunk_type)
            except ValueError:
                continue
        raise ValueError("chunk_type must be one of the known document or legacy chunk types")
    raise TypeError("chunk_type must be a CacheChunkType, string, or None")
