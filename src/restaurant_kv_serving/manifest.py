"""Compatibility wrapper for :mod:`document_kv_cache.manifest`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from document_kv_cache.manifest import InMemoryManifestStore, ManifestStore
from restaurant_kv_serving.models import CacheChunkType, ChunkRef, KVCacheKey, chunk_type_sort_order
