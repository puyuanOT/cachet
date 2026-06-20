"""Materialize planned KV cache chunks into contiguous or segmented payloads."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from document_kv_cache.cache import CacheTier, ChunkCache, ChunkCacheResult
from document_kv_cache.models import MaterializationPlan
from document_kv_cache.storage import RangeReader

__all__ = ["MaterializedKV", "SegmentedMaterializedKV", "KVMaterializer"]


@dataclass(frozen=True, slots=True)
class MaterializedKV:
    plan: MaterializationPlan
    payload: bytes
    segment_byte_offsets: tuple[int, ...]
    materialization_seconds: float
    segment_tiers: tuple[CacheTier | str, ...] = ()

    def __post_init__(self) -> None:
        _validate_materialization_seconds(self.materialization_seconds)
        _validate_segment_offsets(self.plan, self.segment_byte_offsets)
        object.__setattr__(
            self,
            "segment_tiers",
            normalize_segment_tiers(self.segment_tiers, len(self.plan.segments), segments_label="plan segments"),
        )
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must be bytes")
        if len(self.payload) != self.plan.total_bytes:
            raise ValueError(f"payload byte length {len(self.payload)} != plan total_bytes {self.plan.total_bytes}")


@dataclass(frozen=True, slots=True)
class SegmentedMaterializedKV:
    plan: MaterializationPlan
    payloads: tuple[bytes, ...]
    segment_byte_offsets: tuple[int, ...]
    total_bytes: int
    materialization_seconds: float
    segment_tiers: tuple[CacheTier | str, ...] = ()

    def __post_init__(self) -> None:
        _validate_materialization_seconds(self.materialization_seconds)
        _validate_segment_offsets(self.plan, self.segment_byte_offsets)
        object.__setattr__(
            self,
            "segment_tiers",
            normalize_segment_tiers(self.segment_tiers, len(self.plan.segments), segments_label="plan segments"),
        )
        if type(self.total_bytes) is not int:
            raise ValueError("total_bytes must be an integer")
        if self.total_bytes != self.plan.total_bytes:
            raise ValueError(f"total_bytes {self.total_bytes} != plan total_bytes {self.plan.total_bytes}")
        if not isinstance(self.payloads, tuple):
            raise TypeError("payloads must be a tuple of bytes")
        if len(self.payloads) != len(self.plan.segments):
            raise ValueError("payload count must match plan segments")
        for index, (payload, segment) in enumerate(zip(self.payloads, self.plan.segments, strict=True)):
            if not isinstance(payload, bytes):
                raise TypeError(f"payload {index} must be bytes")
            if len(payload) != segment.ref.byte_length:
                raise ValueError(
                    f"payload {index} byte length {len(payload)} != segment byte_length {segment.ref.byte_length}"
                )


class KVMaterializer:
    def __init__(self, *, cache: ChunkCache, reader: RangeReader) -> None:
        self.cache = cache
        self.reader = reader

    def materialize(self, plan: MaterializationPlan) -> MaterializedKV:
        started = time.perf_counter()
        results = self._load_plan_segments(plan)
        parts: list[bytes] = []
        offsets: list[int] = []
        cursor = 0
        for result in results:
            offsets.append(cursor)
            parts.append(result.payload)
            cursor += len(result.payload)
        return MaterializedKV(
            plan=plan,
            payload=b"".join(parts),
            segment_byte_offsets=tuple(offsets),
            materialization_seconds=time.perf_counter() - started,
            segment_tiers=tuple(result.tier for result in results),
        )

    def materialize_segmented(self, plan: MaterializationPlan) -> SegmentedMaterializedKV:
        """Load chunk payloads without merging them into one large CPU buffer."""
        started = time.perf_counter()
        results = self._load_plan_segments(plan)
        parts: list[bytes] = []
        offsets: list[int] = []
        cursor = 0
        for result in results:
            offsets.append(cursor)
            parts.append(result.payload)
            cursor += len(result.payload)
        return SegmentedMaterializedKV(
            plan=plan,
            payloads=tuple(parts),
            segment_byte_offsets=tuple(offsets),
            total_bytes=cursor,
            materialization_seconds=time.perf_counter() - started,
            segment_tiers=tuple(result.tier for result in results),
        )

    def _load_plan_segments(self, plan: MaterializationPlan) -> tuple[ChunkCacheResult, ...]:
        refs = tuple(segment.ref for segment in plan.segments)
        batch_loader = getattr(self.reader, "read_many", None)
        return self.cache.get_many_or_load_with_tier(
            refs,
            self.reader.read,
            batch_loader=batch_loader if callable(batch_loader) else None,
        )


def _validate_materialization_seconds(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("materialization_seconds must be numeric")
    if not math.isfinite(value):
        raise ValueError("materialization_seconds must be finite")
    if value < 0:
        raise ValueError("materialization_seconds must be non-negative")


def normalize_segment_tiers(
    segment_tiers: tuple[CacheTier | str, ...],
    expected_count: int,
    *,
    segments_label: str,
) -> tuple[CacheTier, ...]:
    if not isinstance(segment_tiers, tuple):
        raise TypeError("segment_tiers must be a tuple of CacheTier values")
    if not segment_tiers and expected_count:
        return (CacheTier.COLD_STORAGE,) * expected_count
    if len(segment_tiers) != expected_count:
        raise ValueError(f"segment_tiers count must match {segments_label}")
    normalized: list[CacheTier] = []
    for tier in segment_tiers:
        try:
            normalized.append(tier if isinstance(tier, CacheTier) else CacheTier(tier))
        except (TypeError, ValueError):
            raise ValueError("segment_tiers entries must be valid CacheTier values") from None
    return tuple(normalized)


def _validate_segment_offsets(plan: MaterializationPlan, segment_byte_offsets: tuple[int, ...]) -> None:
    if type(plan.total_bytes) is not int:
        raise ValueError("plan total_bytes must be an integer")
    if plan.total_bytes < 0:
        raise ValueError("plan total_bytes must be non-negative")
    if not isinstance(segment_byte_offsets, tuple):
        raise TypeError("segment_byte_offsets must be a tuple of integers")
    if len(segment_byte_offsets) != len(plan.segments):
        raise ValueError("segment_byte_offsets count must match plan segments")
    for offset in segment_byte_offsets:
        if type(offset) is not int:
            raise ValueError("segment_byte_offsets entries must be integers")
        if offset < 0:
            raise ValueError("segment_byte_offsets entries must be non-negative")
    cursor = 0
    expected_offsets: list[int] = []
    for segment in plan.segments:
        if type(segment.output_byte_start) is not int:
            raise ValueError(f"Plan segment {segment.ref.key.chunk_id} output_byte_start must be an integer")
        if segment.output_byte_start < 0:
            raise ValueError(f"Plan segment {segment.ref.key.chunk_id} output_byte_start must be non-negative")
        if segment.output_byte_start != cursor:
            raise ValueError(
                f"Plan segment {segment.ref.key.chunk_id} output_byte_start {segment.output_byte_start} "
                f"!= expected byte cursor {cursor}"
            )
        expected_offsets.append(cursor)
        cursor += segment.ref.byte_length
    if cursor != plan.total_bytes:
        raise ValueError(f"Plan segment byte lengths {cursor} != plan total_bytes {plan.total_bytes}")
    if tuple(segment_byte_offsets) != tuple(expected_offsets):
        raise ValueError("segment_byte_offsets must match plan segment output_byte_start values")
