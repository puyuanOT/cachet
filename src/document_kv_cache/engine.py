"""Build engine-ready KV handles and payload handoff requests."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from document_kv_cache.artifact_identity import ArtifactIdentity
from document_kv_cache.cache import CacheTier
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment
from document_kv_cache.materializer import MaterializedKV, SegmentedMaterializedKV, normalize_segment_tiers
from document_kv_cache.methods import (
    MethodRegistry,
    default_method_registry,
    validate_registered_reuse_plan,
)
from document_kv_cache.models import CacheGenerationMethod
from document_kv_cache.reuse_contract import ArtifactEncoding, ReusePlan

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
    reuse_plan: ReusePlan | None = None

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
        _validate_estimated_gpu_bytes(self.estimated_gpu_bytes)
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
        else:
            if not isinstance(self.payload, bytes):
                raise TypeError("Payload must be bytes or a tuple of bytes")
            if len(self.payload) != self.handle.total_bytes:
                raise ValueError(
                    f"Payload byte length {len(self.payload)} != handle total_bytes {self.handle.total_bytes}"
                )
        if self.handle.payload_checksum:
            actual_checksum = _payload_checksum(self.payload)
            if actual_checksum != self.handle.payload_checksum:
                raise ValueError("Payload checksum does not match KV handle")
        if self.reuse_plan is not None:
            _validate_reuse_plan_for_handle(self.reuse_plan, self.handle)


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
    cache_method: CacheGenerationMethod | str | None = None,
    adapter_ids: Iterable[str] = (),
    reuse_plan: ReusePlan | None = None,
) -> KVCacheHandle:
    request_id = materialized.plan.request.request_id
    _validate_layout_matches_materialized(
        materialized,
        layout,
        reuse_plan=reuse_plan,
    )
    segments = tuple(_segment_from_plan(index, materialized) for index in range(len(materialized.plan.segments)))
    artifact_identity = _artifact_identity_from_materialized(materialized)
    resolved_cache_method = _resolved_cache_method(
        cache_method,
        artifact_identity,
        layout=layout,
    )
    payload = _payload(materialized)
    handle = KVCacheHandle(
        request_id=request_id,
        handle_uri=handle_uri or f"document-kv://{request_id}",
        layout=layout,
        segments=segments,
        total_tokens=materialized.plan.total_tokens,
        total_bytes=_total_bytes(materialized),
        metadata={} if metadata is None else metadata,
        cache_method=resolved_cache_method,
        adapter_ids=adapter_ids,
        artifact_identity=artifact_identity,
        payload_checksum=_payload_checksum(payload),
    )
    handle.validate()
    if reuse_plan is not None:
        _validate_reuse_plan_for_handle(reuse_plan, handle)
    return handle


def build_engine_ready_request(
    materialized: MaterializedKV | SegmentedMaterializedKV,
    *,
    layout: KVLayout,
    handle_uri: str | None = None,
    metadata: Mapping[str, str] | None = None,
    cache_method: CacheGenerationMethod | str | None = None,
    adapter_ids: Iterable[str] = (),
    kv_gpu_bytes_per_payload_byte: float = 1.0,
    reuse_plan: ReusePlan | None = None,
    allow_legacy_reuse_plan: bool = False,
    method_registry: MethodRegistry | None = None,
) -> EngineReadyRequest:
    gpu_byte_multiplier = _normalize_gpu_byte_multiplier(kv_gpu_bytes_per_payload_byte)
    artifact_identity = _artifact_identity_from_materialized(materialized)
    resolved_cache_method = _resolved_cache_method(
        cache_method,
        artifact_identity,
        layout=layout,
    )
    resolved_reuse_plan = _resolve_reuse_plan(
        resolved_cache_method,
        reuse_plan,
        artifact_identity=artifact_identity,
        allow_legacy=allow_legacy_reuse_plan,
        method_registry=method_registry,
    )
    handle = build_handle_from_materialized(
        materialized,
        layout=layout,
        handle_uri=handle_uri,
        metadata=metadata,
        cache_method=resolved_cache_method,
        adapter_ids=adapter_ids,
        reuse_plan=resolved_reuse_plan,
    )
    runtime_bytes = handle.total_tokens * handle.layout.bytes_per_token
    ready_request = EngineReadyRequest(
        handle=handle,
        payload=_payload(materialized),
        estimated_gpu_bytes=_estimate_gpu_bytes(runtime_bytes, gpu_byte_multiplier),
        segment_tiers=materialized.segment_tiers,
        reuse_plan=resolved_reuse_plan,
    )
    ready_request.validate()
    return ready_request


def _normalize_gpu_byte_multiplier(kv_gpu_bytes_per_payload_byte: float) -> float:
    if isinstance(kv_gpu_bytes_per_payload_byte, bool) or not isinstance(kv_gpu_bytes_per_payload_byte, int | float):
        raise TypeError("kv_gpu_bytes_per_payload_byte must be numeric")
    try:
        multiplier = float(kv_gpu_bytes_per_payload_byte)
    except OverflowError as exc:
        raise ValueError("kv_gpu_bytes_per_payload_byte must be finite") from exc
    if not math.isfinite(multiplier):
        raise ValueError("kv_gpu_bytes_per_payload_byte must be finite")
    if multiplier < 0:
        raise ValueError("kv_gpu_bytes_per_payload_byte must be non-negative")
    return multiplier


def _estimate_gpu_bytes(total_bytes: int, multiplier: float) -> int:
    estimated_gpu_bytes = total_bytes * multiplier
    if not math.isfinite(estimated_gpu_bytes):
        raise ValueError("estimated_gpu_bytes must be finite")
    estimated_gpu_bytes_int = int(estimated_gpu_bytes)
    _validate_estimated_gpu_bytes(estimated_gpu_bytes_int)
    return estimated_gpu_bytes_int


def _validate_estimated_gpu_bytes(estimated_gpu_bytes: int) -> None:
    if type(estimated_gpu_bytes) is not int:
        raise ValueError("estimated_gpu_bytes must be an integer")
    if estimated_gpu_bytes < 0:
        raise ValueError("estimated_gpu_bytes must be non-negative")


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
        token_contract=ref.key.token_contract,
    )


def _validate_layout_matches_materialized(
    materialized: MaterializedKV | SegmentedMaterializedKV,
    layout: KVLayout,
    *,
    reuse_plan: ReusePlan | None,
) -> None:
    request = materialized.plan.request
    if layout.model_id != request.model_id:
        raise ValueError(f"Layout model_id {layout.model_id!r} does not match request model_id {request.model_id!r}")
    if layout.lora_id != request.lora_id:
        raise ValueError(f"Layout lora_id {layout.lora_id!r} does not match request lora_id {request.lora_id!r}")
    artifact_encoding = (
        ArtifactEncoding.RAW_KV
        if reuse_plan is None
        else reuse_plan.artifact_format.encoding
    )
    if artifact_encoding == ArtifactEncoding.ENGINE_NATIVE:
        raise ValueError(
            "engine-native reuse plans cannot validate persisted Cachet artifacts"
        )
    for plan_segment in materialized.plan.segments:
        ref = plan_segment.ref
        if ref.key.model_id != layout.model_id:
            raise ValueError(f"Chunk {ref.key.chunk_id} model_id does not match layout")
        if ref.key.lora_id != layout.lora_id:
            raise ValueError(f"Chunk {ref.key.chunk_id} lora_id does not match layout")
        if ref.layout_version != layout.layout_version:
            raise ValueError(f"Chunk {ref.key.chunk_id} layout_version does not match layout")
        if ref.storage_layout != layout.storage_layout:
            raise ValueError(f"Chunk {ref.key.chunk_id} storage_layout does not match layout")
        identity = ref.key.artifact_identity
        if identity is not None:
            if identity.kv_dtype != ref.dtype:
                raise ValueError(
                    f"Chunk {ref.key.chunk_id} persisted dtype does not match "
                    "artifact identity"
                )
            if identity.runtime_kv_dtype != layout.dtype:
                raise ValueError(
                    f"Chunk {ref.key.chunk_id} runtime_kv_dtype does not match layout"
                )
        if artifact_encoding == ArtifactEncoding.RAW_KV:
            if ref.dtype != layout.dtype:
                raise ValueError(
                    f"Chunk {ref.key.chunk_id} raw-KV dtype does not match layout"
                )
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


def _artifact_identity_from_materialized(
    materialized: MaterializedKV | SegmentedMaterializedKV,
):
    identities = {
        segment.ref.key.artifact_identity
        for segment in materialized.plan.segments
    }
    if len(identities) > 1:
        raise ValueError("Materialized KV segments must share one artifact_identity")
    identity = next(iter(identities), None)
    request_identity = materialized.plan.request.artifact_identity
    if request_identity is not None and request_identity != identity:
        raise ValueError("Materialized KV artifact_identity does not match request")
    return identity


def _resolved_cache_method(
    cache_method: CacheGenerationMethod | str | None,
    artifact_identity: ArtifactIdentity | None,
    *,
    layout: KVLayout,
) -> str:
    artifact_method = None if artifact_identity is None else artifact_identity.method_id
    if cache_method is None:
        if artifact_method is not None:
            return artifact_method
        return (
            CacheGenerationMethod.VANILLA_PREFILL.value
            if layout.pre_rope
            else CacheGenerationMethod.FULL_PREFIX_PREFILL.value
        )
    explicit = _cache_method_value(cache_method)
    if artifact_method is not None and explicit != artifact_method:
        raise ValueError(
            f"cache_method {explicit!r} does not match artifact method {artifact_method!r}"
        )
    return explicit


def _payload_checksum(payload: bytes | tuple[bytes, ...]) -> str:
    digest = hashlib.sha256()
    if isinstance(payload, bytes):
        digest.update(payload)
    else:
        for segment in payload:
            digest.update(segment)
    return digest.hexdigest()


def _resolve_reuse_plan(
    cache_method: str,
    reuse_plan: ReusePlan | None,
    *,
    artifact_identity,
    allow_legacy: bool,
    method_registry: MethodRegistry | None,
) -> ReusePlan | None:
    registry = default_method_registry() if method_registry is None else method_registry
    if not isinstance(registry, MethodRegistry):
        raise TypeError("method_registry must be a MethodRegistry or None")
    if reuse_plan is not None:
        if not isinstance(reuse_plan, ReusePlan):
            raise TypeError("reuse_plan must be a ReusePlan or None")
        if reuse_plan.method_id != cache_method:
            raise ValueError("reuse_plan.method_id must match cache_method")
        validate_registered_reuse_plan(
            reuse_plan,
            artifact_identity=artifact_identity,
            registry=registry,
        )
        return reuse_plan
    try:
        resolved = registry.get(
            cache_method,
            require_implemented=True,
        ).reuse_plan()
    except (KeyError, NotImplementedError) as exc:
        if allow_legacy:
            return None
        raise ValueError(
            f"cache method {cache_method!r} requires an explicit registered reuse plan"
        ) from exc
    validate_registered_reuse_plan(
        resolved,
        artifact_identity=artifact_identity,
        registry=registry,
    )
    return resolved


def _validate_reuse_plan_for_handle(
    reuse_plan: ReusePlan,
    handle: KVCacheHandle,
) -> None:
    if not isinstance(reuse_plan, ReusePlan):
        raise TypeError("reuse_plan must be a ReusePlan")
    if reuse_plan.method_id != handle.cache_method:
        raise ValueError("reuse_plan.method_id must match handle.cache_method")
    if not reuse_plan.requires_artifact:
        raise ValueError("engine-native reuse plans cannot carry a Cachet artifact payload")
    reuse_plan.validate_runtime_layout(handle.layout)
    identity = handle.artifact_identity
    if identity is not None:
        artifact_format = reuse_plan.artifact_format
        if (
            identity.artifact_format_id != artifact_format.format_id
            or identity.artifact_format_version != artifact_format.version
        ):
            raise ValueError("reuse_plan artifact format does not match artifact identity")
        if (
            artifact_format.encoding == ArtifactEncoding.RAW_KV
            and identity.kv_dtype != identity.runtime_kv_dtype
        ):
            raise ValueError(
                "raw-KV artifact dtype must match runtime_kv_dtype"
            )
