"""Compatibility wrapper for :mod:`document_kv_cache.engine_protocol`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from document_kv_cache.engine_protocol import (
    DTYPE_BYTE_WIDTHS,
    AttentionMechanism,
    KVCacheHandle,
    KVLayout,
    KVSegment,
    KVStorageLayout,
    dtype_byte_width,
    kv_storage_layout_from_value,
)
