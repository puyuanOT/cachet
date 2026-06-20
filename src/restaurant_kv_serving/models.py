"""Compatibility wrapper for :mod:`document_kv_cache.models`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility path.
    from enum import Enum

    class StrEnum(str, Enum):
        pass

from typing import TypeAlias

from document_kv_cache.engine_protocol import KVStorageLayout, kv_storage_layout_from_value
from document_kv_cache.models import (
    DOCUMENT_CHUNK_TYPES,
    LEGACY_RESTAURANT_CHUNK_TYPES,
    CacheChunkType,
    CacheChunkTypeSet,
    CacheGenerationMethod,
    ChunkId,
    ChunkRef,
    ChunkType,
    DocumentChunkMap,
    DocumentChunkRole,
    DocumentChunkType,
    DocumentKVRequest,
    FrozenDocumentChunkMap,
    KVCacheKey,
    MaterializationPlan,
    PlanSegment,
    RestaurantKVRequest,
    SHA256_HEX_LENGTH,
    chunk_type_role,
    chunk_type_sort_order,
    chunk_types_for_request,
)
