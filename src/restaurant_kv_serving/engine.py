"""Compatibility wrapper for :mod:`document_kv_cache.engine`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from document_kv_cache.cache import CacheTier
from document_kv_cache.engine import (
    EngineReadyRequest,
    ServingEngineConnector,
    _cache_method_value,
    _payload,
    _segment_from_plan,
    _total_bytes,
    _validate_layout_matches_materialized,
    build_engine_ready_request,
    build_handle_from_materialized,
)
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment
from document_kv_cache.materializer import MaterializedKV, SegmentedMaterializedKV, normalize_segment_tiers
from document_kv_cache.models import CacheGenerationMethod
