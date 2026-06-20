"""Compatibility wrapper for :mod:`document_kv_cache.planner`."""

from __future__ import annotations

from document_kv_cache.planner import CachePlanner, CacheRequest
from restaurant_kv_serving.manifest import ManifestStore
from restaurant_kv_serving.models import (
    DocumentKVRequest,
    KVCacheKey,
    MaterializationPlan,
    PlanSegment,
    RestaurantKVRequest,
    chunk_types_for_request,
)
