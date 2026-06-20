"""Build engine-ready KV handles and payload handoff requests."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from document_kv_cache.cache import CacheTier
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment
from document_kv_cache.materializer import MaterializedKV, SegmentedMaterializedKV, normalize_segment_tiers
from document_kv_cache.models import CacheGenerationMethod

__all__ = [
    "EngineReadyRequest",
    "ServingEngineConnector",
    "build_handle_from_materialized",
    "build_engine_ready_request",
]


@dataclass(frozen=True, slots=True)
class EngineReadyRequest:
    handle: KVCacheHandle
    payload: bytes | tuple[bytes, ...]
    estimated_gpu_bytes: int
    segment_tiers: tuple[CacheTier | str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "segment_tiers",
            normalize_segment_tiers(self.segment_tiers, len(self.handle.segments), segments_label="handle segments"),
        )

    @property
    def request_id(self) -> str:
        return self.handle.request_id

    def validate(self) -> None:
        self.handle.validate()
        if self.estimated_gpu_bytes < 0:
            raise ValueError("estimated_gpu_bytes must be non-negative")
        normalize_segment_tiers(self.segment_tiers, len(self.handle.segments), segments_label="handle segments")
        if isinstance(self.payload, tuple):
            if len(self.payload) != len(self.handle.segments):
                raise ValueError("Segmented payload count must match handle segments")
            for index, (payload, segment) in enumerate(zip(self.payload, self.handle.segments, strict=True)):
                if not isinstance(payload, bytes):
                    raise TypeError(f"Segmented payload {index} must be bytes")
                if len(payload) != segment.byte_length:
                    raise ValueError(
                        f"Segmented payload {index} byte length {len(payload)} "
                        f"!= segment byte_length {segment.byte_length}"
                    )
            return
        if not isinstance(self.payload, bytes):
            raise TypeError("Payload must be bytes or a tuple of bytes")
        if len(self.payload) != self.handle.total_bytes:
            raise ValueError(f"Payload byte length {len(self.payload)} != handle total_bytes {self.handle.total_bytes}")


class ServingEngineConnector(Protocol):
    """Minimal adapter surface implemented by vLLM, SGLang, or test doubles."""

    def submit(self, request: EngineReadyRequest) -> None: ...

    def release(self, request_id: str) -> None: ...


def build_handle_from_materialized(
    materialized: MaterializedKV | SegmentedMaterializedKV,
    *,
    layout: KVLayout,
    handle_uri: str | None = None,
    metadata: Mapping[str, str] | None = None,
    cache_method: CacheGenerationMethod | str = CacheGenerationMethod.VANILLA_PREFILL,
    adapter_ids: tuple[str, ...] = (),
) -> KVCacheHandle:
    request_id = materialized.plan.request.request_id
    _validate_layout_matches_materialized(materialized, layout)
    segments = tuple(_segment_from_plan(index, materialized) for index in range(len(materialized.plan.segments)))
    handle = KVCacheHandle(
        request_id=request_id,
        handle_uri=handle_uri or f"document-kv://{request_id}",
        layout=layout,
        segments=segments,
        total_tokens=materialized.plan.total_tokens,
        total_bytes=_total_bytes(materialized),
        metadata=metadata or {},
        cache_method=_cache_method_value(cache_method),
        adapter_ids=adapter_ids,
    )
    handle.validate()
    return handle


def build_engine_ready_request(
    materialized: MaterializedKV | SegmentedMaterializedKV,
    *,
    layout: KVLayout,
    handle_uri: str | None = None,
    metadata: Mapping[str, str] | None = None,
    cache_method: CacheGenerationMethod | str = CacheGenerationMethod.VANILLA_PREFILL,
    adapter_ids: tuple[str, ...] = (),
    kv_gpu_bytes_per_payload_byte: float = 1.0,
) -> EngineReadyRequest:
    gpu_byte_multiplier = _normalize_gpu_byte_multiplier(kv_gpu_bytes_per_payload_byte)
    handle = build_handle_from_materialized(
        materialized,
        layout=layout,
        handle_uri=handle_uri,
        metadata=metadata,
        cache_method=cache_method,
        adapter_ids=adapter_ids,
    )
    ready_request = EngineReadyRequest(
        handle=handle,
        payload=_payload(materialized),
        estimated_gpu_bytes=int(handle.total_bytes * gpu_byte_multiplier),
        segment_tiers=materialized.segment_tiers,
    )
    ready_request.validate()
    return ready_request


def _normalize_gpu_byte_multiplier(kv_gpu_bytes_per_payload_byte: float) -> float:
    if isinstance(kv_gpu_bytes_per_payload_byte, bool) or not isinstance(kv_gpu_bytes_per_payload_byte, int | float):
        raise TypeError("kv_gpu_bytes_per_payload_byte must be numeric")
    multiplier = float(kv_gpu_bytes_per_payload_byte)
    if not math.isfinite(multiplier):
        raise ValueError("kv_gpu_bytes_per_payload_byte must be finite")
    if multiplier < 0:
        raise ValueError("kv_gpu_bytes_per_payload_byte must be non-negative")
    return multiplier


def _segment_from_plan(index: int, materialized: MaterializedKV | SegmentedMaterializedKV) -> KVSegment:
    plan_segment = materialized.plan.segments[index]
    ref = plan_segment.ref
    return KVSegment(
        document_id=ref.key.document_id,
        chunk_type=ref.key.chunk_type.value,
        chunk_id=ref.key.chunk_id,
        token_start=plan_segment.output_token_start,
        token_count=ref.token_count,
        byte_start=materialized.segment_byte_offsets[index],
        byte_length=ref.byte_length,
        content_hash=ref.key.content_hash,
    )


def _validate_layout_matches_materialized(
    materialized: MaterializedKV | SegmentedMaterializedKV,
    layout: KVLayout,
) -> None:
    request = materialized.plan.request
    if layout.model_id != request.model_id:
        raise ValueError(f"Layout model_id {layout.model_id!r} does not match request model_id {request.model_id!r}")
    if layout.lora_id != request.lora_id:
        raise ValueError(f"Layout lora_id {layout.lora_id!r} does not match request lora_id {request.lora_id!r}")
    for plan_segment in materialized.plan.segments:
        ref = plan_segment.ref
        if ref.key.model_id != layout.model_id:
            raise ValueError(f"Chunk {ref.key.chunk_id} model_id does not match layout")
        if ref.key.lora_id != layout.lora_id:
            raise ValueError(f"Chunk {ref.key.chunk_id} lora_id does not match layout")
        if ref.dtype != layout.dtype:
            raise ValueError(f"Chunk {ref.key.chunk_id} dtype does not match layout")
        if ref.layout_version != layout.layout_version:
            raise ValueError(f"Chunk {ref.key.chunk_id} layout_version does not match layout")
        if ref.storage_layout != layout.storage_layout:
            raise ValueError(f"Chunk {ref.key.chunk_id} storage_layout does not match layout")
        expected_bytes = ref.token_count * layout.bytes_per_token
        if ref.byte_length != expected_bytes:
            raise ValueError(
                f"Chunk {ref.key.chunk_id} byte_length {ref.byte_length} "
                f"!= token_count * bytes_per_token {expected_bytes}"
            )


def _total_bytes(materialized: MaterializedKV | SegmentedMaterializedKV) -> int:
    if isinstance(materialized, MaterializedKV):
        return len(materialized.payload)
    return materialized.total_bytes


def _payload(materialized: MaterializedKV | SegmentedMaterializedKV) -> bytes | tuple[bytes, ...]:
    if isinstance(materialized, MaterializedKV):
        return materialized.payload
    return materialized.payloads


def _cache_method_value(cache_method: CacheGenerationMethod | str) -> str:
    if isinstance(cache_method, CacheGenerationMethod):
        return cache_method.value
    return cache_method
