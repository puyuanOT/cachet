from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from restaurant_kv_serving.models import ChunkRef, KVCacheKey


@dataclass(frozen=True, slots=True)
class PackChunk:
    key: KVCacheKey
    payload: bytes
    token_count: int
    dtype: str
    layout_version: str


class LocalRangeReader:
    def read(self, ref: ChunkRef) -> bytes:
        path = _local_path(ref.shard_uri)
        with path.open("rb") as handle:
            handle.seek(ref.byte_offset)
            payload = handle.read(ref.byte_length)
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != ref.checksum:
            raise ValueError(f"Checksum mismatch for {ref.key.storage_key()}: expected {ref.checksum}, got {checksum}")
        return payload


def write_kvpack(path: str | Path, chunks: Iterable[PackChunk], *, align_bytes: int = 4096) -> list[ChunkRef]:
    target = _local_path(str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    refs: list[ChunkRef] = []
    offset = 0
    with target.open("wb") as handle:
        for chunk in chunks:
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
                )
            )
            offset += len(payload)
    return refs


def _local_path(uri: str) -> Path:
    if uri.startswith("file:"):
        return Path(uri[len("file:") :])
    if uri.startswith("dbfs:/"):
        return Path("/dbfs") / uri[len("dbfs:/") :].lstrip("/")
    return Path(uri)

