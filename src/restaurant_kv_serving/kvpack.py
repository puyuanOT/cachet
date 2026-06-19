from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from restaurant_kv_serving.engine_protocol import (
    KVStorageLayout,
    kv_storage_layout_from_value,
)
from restaurant_kv_serving.models import ChunkRef, KVCacheKey
from restaurant_kv_serving.storage import DiskRangeReader, local_path


@dataclass(frozen=True, slots=True)
class PackChunk:
    key: KVCacheKey
    payload: bytes
    token_count: int
    dtype: str
    layout_version: str
    storage_layout: KVStorageLayout | str = KVStorageLayout.SEPARATE_KEY_VALUE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "storage_layout",
            kv_storage_layout_from_value(self.storage_layout, field_name="storage_layout"),
        )
        try:
            payload = _coerce_payload(self.payload)
        except TypeError as exc:
            raise ValueError("payload must be bytes-like") from exc
        object.__setattr__(self, "payload", payload)
        if not payload:
            raise ValueError("payload must be non-empty")
        if type(self.token_count) is not int:
            raise ValueError("token_count must be an integer")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("dtype must be non-empty")
        if not isinstance(self.layout_version, str) or not self.layout_version:
            raise ValueError("layout_version must be non-empty")


LocalRangeReader = DiskRangeReader


def write_kvpack(path: str | Path, chunks: Iterable[PackChunk], *, align_bytes: int = 4096) -> list[ChunkRef]:
    if type(align_bytes) is not int:
        raise ValueError("align_bytes must be an integer")
    if align_bytes <= 0:
        raise ValueError("align_bytes must be positive")
    chunk_sequence = tuple(chunks)
    target = local_path(str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    refs: list[ChunkRef] = []
    offset = 0
    with target.open("wb") as handle:
        for chunk in chunk_sequence:
            if align_bytes > 1:
                padding = (-offset) % align_bytes
                if padding:
                    handle.write(b"\0" * padding)
                    offset += padding
            payload = bytes(chunk.payload)
            handle.write(payload)
            checksum = hashlib.sha256(payload).hexdigest()
            refs.append(
                ChunkRef(
                    key=chunk.key,
                    shard_uri=str(path),
                    byte_offset=offset,
                    byte_length=len(payload),
                    token_count=chunk.token_count,
                    dtype=chunk.dtype,
                    layout_version=chunk.layout_version,
                    checksum=checksum,
                    storage_layout=chunk.storage_layout,
                )
            )
            offset += len(payload)
    return refs


def _coerce_payload(payload: object) -> bytes:
    if isinstance(payload, bytes):
        return payload
    return memoryview(payload).tobytes()


_local_path = local_path
