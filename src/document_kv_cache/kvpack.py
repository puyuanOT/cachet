from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from document_kv_cache.engine_protocol import (
    KVStorageLayout,
    kv_storage_layout_from_value,
)
from document_kv_cache.models import ChunkRef, KVCacheKey
from document_kv_cache.storage import DiskRangeReader, local_path


__all__ = ["PackChunk", "LocalRangeReader", "write_kvpack", "write_kvpack_bytes"]


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


def write_kvpack(
    path: str | Path,
    chunks: Iterable[PackChunk],
    *,
    align_bytes: int = 4096,
    path_resolver: Callable[[str], Path] | None = None,
) -> list[ChunkRef]:
    return _write_kvpack(path, chunks, align_bytes=align_bytes, path_resolver=path_resolver or local_path)


def write_kvpack_bytes(
    shard_uri: str,
    chunks: Iterable[PackChunk],
    *,
    align_bytes: int = 4096,
) -> tuple[bytes, list[ChunkRef]]:
    buffer = BytesIO()
    refs = _write_kvpack_payload(shard_uri, chunks, align_bytes=align_bytes, write=buffer.write)
    return buffer.getvalue(), refs


def _write_kvpack(
    path: str | Path,
    chunks: Iterable[PackChunk],
    *,
    align_bytes: int = 4096,
    path_resolver: Callable[[str], Path],
) -> list[ChunkRef]:
    _validate_align_bytes(align_bytes)
    chunk_sequence = tuple(chunks)
    target = path_resolver(str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        return _write_kvpack_payload(str(path), chunk_sequence, align_bytes=align_bytes, write=handle.write)


def _write_kvpack_payload(
    shard_uri: str | Path,
    chunks: Iterable[PackChunk],
    *,
    align_bytes: int = 4096,
    write: Callable[[bytes], object],
) -> list[ChunkRef]:
    _validate_align_bytes(align_bytes)
    chunk_sequence = tuple(chunks)
    refs: list[ChunkRef] = []
    offset = 0
    for chunk in chunk_sequence:
        if align_bytes > 1:
            padding = (-offset) % align_bytes
            if padding:
                write(b"\0" * padding)
                offset += padding
        payload = bytes(chunk.payload)
        write(payload)
        checksum = hashlib.sha256(payload).hexdigest()
        refs.append(
            ChunkRef(
                key=chunk.key,
                shard_uri=str(shard_uri),
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


def _validate_align_bytes(align_bytes: int) -> None:
    if type(align_bytes) is not int:
        raise ValueError("align_bytes must be an integer")
    if align_bytes <= 0:
        raise ValueError("align_bytes must be positive")


def _coerce_payload(payload: object) -> bytes:
    if isinstance(payload, bytes):
        return payload
    return memoryview(payload).tobytes()


_local_path = local_path
