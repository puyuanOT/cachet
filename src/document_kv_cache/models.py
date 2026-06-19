"""Public document namespace for cache keys, requests, and plans."""

from __future__ import annotations

from importlib import import_module

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.models",
    (
        "DocumentChunkType",
        "DocumentChunkRole",
        "CacheGenerationMethod",
        "DocumentChunkMap",
        "CacheChunkType",
        "CacheChunkTypeSet",
        "DOCUMENT_CHUNK_TYPES",
        "LEGACY_RESTAURANT_CHUNK_TYPES",
        "KVCacheKey",
        "ChunkRef",
        "DocumentKVRequest",
        "PlanSegment",
        "MaterializationPlan",
        "chunk_type_role",
        "chunk_type_sort_order",
        "chunk_types_for_request",
    ),
    globals(),
)

_legacy_models = import_module("restaurant_kv_serving.models")
ChunkType = _legacy_models.ChunkType
RestaurantKVRequest = _legacy_models.RestaurantKVRequest

del _legacy_models
del import_module
del reexport_public
