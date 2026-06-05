from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from restaurant_kv_serving.cache import ChunkCache
from restaurant_kv_serving.models import ChunkRef, MaterializationPlan


class RangeReader(Protocol):
    def read(self, ref: ChunkRef) -> bytes: ...


@dataclass(frozen=True, slots=True)
class MaterializedKV:
    plan: MaterializationPlan
    payload: bytes
    segment_byte_offsets: tuple[int, ...]
    materialization_seconds: float


class KVMaterializer:
    def __init__(self, *, cache: ChunkCache, reader: RangeReader) -> None:
        self.cache = cache
        self.reader = reader

    def materialize(self, plan: MaterializationPlan) -> MaterializedKV:
        started = time.perf_counter()
        parts: list[bytes] = []
        offsets: list[int] = []
        cursor = 0
        for segment in plan.segments:
            offsets.append(cursor)
            payload = self.cache.get_or_load(segment.ref, self.reader.read)
            parts.append(payload)
            cursor += len(payload)
        return MaterializedKV(
            plan=plan,
            payload=b"".join(parts),
            segment_byte_offsets=tuple(offsets),
            materialization_seconds=time.perf_counter() - started,
        )

