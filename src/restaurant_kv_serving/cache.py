"""Compatibility wrapper for :mod:`document_kv_cache.cache`."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from document_kv_cache.cache import ByteLRU, CacheTier, ChunkCache, ChunkCacheResult, ChunkCacheStats
from restaurant_kv_serving.models import ChunkRef
