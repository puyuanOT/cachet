"""Compatibility wrapper for :mod:`document_kv_cache.materializer`."""

from __future__ import annotations

import time
from dataclasses import dataclass

from document_kv_cache.materializer import (
    KVMaterializer,
    MaterializedKV,
    SegmentedMaterializedKV,
    normalize_segment_tiers,
)
from restaurant_kv_serving.cache import CacheTier, ChunkCache
from restaurant_kv_serving.models import MaterializationPlan
from restaurant_kv_serving.storage import RangeReader
