"""Compatibility wrapper for :mod:`document_kv_cache.kvpack`."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from document_kv_cache.kvpack import PackChunk, _write_kvpack
from restaurant_kv_serving.engine_protocol import KVStorageLayout, kv_storage_layout_from_value
from restaurant_kv_serving.models import ChunkRef, KVCacheKey
from restaurant_kv_serving.storage import DiskRangeReader, local_path


LocalRangeReader = DiskRangeReader


def write_kvpack(path: str | Path, chunks: Iterable[PackChunk], *, align_bytes: int = 4096) -> list[ChunkRef]:
    return _write_kvpack(path, chunks, align_bytes=align_bytes, path_resolver=local_path)
