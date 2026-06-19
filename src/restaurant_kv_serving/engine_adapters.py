"""Adapter contracts for external vLLM and SGLang KV injection."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from restaurant_kv_serving.cache import CacheTier
from restaurant_kv_serving.engine import EngineReadyRequest
from restaurant_kv_serving.engine_protocol import (
    KVCacheHandle,
    KVLayout,
    KVSegment,
    kv_storage_layout_from_value,
)
from restaurant_kv_serving.storage import local_path

RESERVED_METADATA_PREFIXES = ("document_kv.", "engine.")
ENGINE_ADAPTER_HANDOFF_RECORD_TYPE = "document_kv.engine_adapter_request.v1"
ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION = 2
ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE = "document_kv.engine_kv_connector_probe.v1"
ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION = 2
_NON_NATIVE_PROBE_KIND_VALUES = frozenset({"debug_in_memory", "in_memory_debug", "non_native_debug"})
_NON_NATIVE_PROBE_VALUES = frozenset({"debug_in_memory", "in_memory_debug", "non_native_debug"})
_PROBE_KIND_METADATA_SUFFIX = ".probe_kind"
_PROBE_METADATA_SUFFIX = ".probe"
_PROBE_NATIVE_RUNTIME_METADATA_SUFFIX = ".native_runtime"
IN_PROCESS_PAYLOAD_SOURCE = "in_process"
EXTERNAL_URI_PAYLOAD_SOURCE = "external_uri"
_EXTERNAL_PAYLOAD_URI_SCHEMES = {
    "abfss",
    "dbfs",
    "disk",
    "file",
    "gs",
    "s3",
    "s3a",
    "s3n",
    "uc-volume",
    "wasbs",
}


class ServingBackend(StrEnum):
    VLLM = "vllm"
    SGLANG = "sglang"


class PayloadMode(StrEnum):
    MERGED = "merged"
    SEGMENTED = "segmented"


@dataclass(frozen=True, slots=True)
class EngineAdapterSpec:
    """Capabilities expected from an external serving-engine adapter."""

    backend: ServingBackend
    connector_package: str
    kv_injection_method: str
    payload_contract: str
    supports_merged_payload: bool = True
    supports_segmented_payload: bool = True
    supports_lora_adapters: bool = True
    supports_dynamic_loading: bool = True
    required_steps: tuple[str, ...] = (
        "reserve_engine_kv_blocks",
        "load_or_map_document_kv_payload",
        "bind_kv_handle_to_request",
        "schedule_decode_with_engine",
        "release_kv_handle",
    )
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _backend_from_value(self.backend, field_name="backend"))
        if not self.connector_package:
            raise ValueError("connector_package must be non-empty")
        _validate_connector_package_matches_backend(self.backend, self.connector_package)
        if not self.kv_injection_method:
            raise ValueError("kv_injection_method must be non-empty")
        if not self.payload_contract:
            raise ValueError("payload_contract must be non-empty")
        if not self.supports_merged_payload and not self.supports_segmented_payload:
            raise ValueError("Engine adapter must support at least one payload mode")
        required_steps = _normalize_required_steps(self.required_steps)
        if not required_steps:
            raise ValueError("required_steps must be non-empty")
        object.__setattr__(self, "required_steps", required_steps)
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def validate_ready_request(self, request: EngineReadyRequest) -> None:
        request.validate()
        _reject_reserved_metadata(request.handle.metadata)
        payload_mode = payload_mode_for(request)
        if payload_mode == PayloadMode.SEGMENTED and not self.supports_segmented_payload:
            raise ValueError(f"{self.backend.value} adapter does not support segmented payloads")
        if payload_mode == PayloadMode.MERGED and not self.supports_merged_payload:
            raise ValueError(f"{self.backend.value} adapter does not support merged payloads")
        if request.handle.adapter_ids and not self.supports_lora_adapters:
            raise ValueError(f"{self.backend.value} adapter does not support LoRA adapter ids")


@dataclass(frozen=True, slots=True)
class EngineAdapterRequest:
    """Engine-specific plan handed to a vLLM or SGLang integration layer."""

    backend: ServingBackend
    ready_request: EngineReadyRequest
    connector_package: str
    kv_injection_method: str
    payload_contract: str
    required_steps: tuple[str, ...]
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _backend_from_value(self.backend, field_name="backend"))
        _validate_connector_package_matches_backend(self.backend, self.connector_package)
        object.__setattr__(self, "required_steps", _normalize_required_steps(self.required_steps))
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def request_id(self) -> str:
        return self.ready_request.request_id

    @property
    def handle_uri(self) -> str:
        return self.ready_request.handle.handle_uri

    @property
    def payload_mode(self) -> PayloadMode:
        return payload_mode_for(self.ready_request)


@dataclass(frozen=True, slots=True)
class EngineKVSegmentBinding:
    """Validated source span and token/block span for one document KV segment."""

    document_id: str
    chunk_type: str
    chunk_id: str
    token_start: int
    token_count: int
    token_end: int
    byte_start: int
    byte_length: int
    byte_end: int
    first_block_index: int
    last_block_index_exclusive: int
    content_hash: str = ""
    cache_tier: CacheTier | str = CacheTier.COLD_STORAGE

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_tier", _cache_tier_from_value(self.cache_tier, field_name="cache_tier"))

    @property
    def block_count(self) -> int:
        return self.last_block_index_exclusive - self.first_block_index


@dataclass(frozen=True, slots=True)
class EngineKVInjectionPlan:
    """Reference reservation/copy plan consumed by thin vLLM or SGLang adapters."""

    backend: ServingBackend
    request_id: str
    handle_uri: str
    connector_package: str
    kv_injection_method: str
    payload_mode: PayloadMode
    payload_source_uri: str | None
    layout: KVLayout
    cache_method: str
    adapter_ids: tuple[str, ...]
    total_tokens: int
    total_bytes: int
    total_blocks: int
    estimated_gpu_bytes: int
    segments: tuple[EngineKVSegmentBinding, ...]
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _backend_from_value(self.backend, field_name="backend"))
        object.__setattr__(self, "payload_mode", _payload_mode_from_value(self.payload_mode, field_name="payload_mode"))
        _validate_connector_package_matches_backend(self.backend, self.connector_package)
        if self.total_blocks != _block_count(self.total_tokens, self.layout.block_size):
            raise ValueError("total_blocks does not match total_tokens and layout.block_size")
        if self.segments:
            if self.segments[0].token_start != 0 or self.segments[0].byte_start != 0:
                raise ValueError("First KV segment binding must start at token and byte offset zero")
            if self.segments[-1].token_end != self.total_tokens:
                raise ValueError("Segment bindings do not cover total_tokens")
            if self.segments[-1].byte_end != self.total_bytes:
                raise ValueError("Segment bindings do not cover total_bytes")
        object.__setattr__(self, "adapter_ids", tuple(self.adapter_ids))
        object.__setattr__(self, "segments", tuple(self.segments))
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class EngineKVReservationAction:
    """Engine-native KV block reservation requested by an adapter."""

    backend: ServingBackend
    request_id: str
    total_blocks: int
    total_tokens: int
    estimated_gpu_bytes: int
    layout: KVLayout
    adapter_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _backend_from_value(self.backend, field_name="backend"))
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        self.layout.validate()
        if self.total_blocks != _block_count(self.total_tokens, self.layout.block_size):
            raise ValueError("total_blocks does not match total_tokens and layout.block_size")
        if self.estimated_gpu_bytes < 0:
            raise ValueError("estimated_gpu_bytes must be non-negative")
        if any(not isinstance(adapter_id, str) or not adapter_id for adapter_id in self.adapter_ids):
            raise ValueError("adapter_ids entries must be non-empty strings")
        object.__setattr__(self, "adapter_ids", tuple(self.adapter_ids))


@dataclass(frozen=True, slots=True)
class EngineKVSegmentCopyAction:
    """Source byte range and destination token/block span for one segment copy."""

    request_id: str
    document_id: str
    chunk_type: str
    chunk_id: str
    payload_index: int | None
    source_byte_start: int
    source_byte_length: int
    global_byte_start: int
    global_byte_end: int
    token_start: int
    token_count: int
    token_end: int
    first_block_index: int
    last_block_index_exclusive: int
    content_hash: str = ""
    cache_tier: CacheTier | str = CacheTier.COLD_STORAGE

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_tier", _cache_tier_from_value(self.cache_tier, field_name="cache_tier"))
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.document_id:
            raise ValueError("document_id must be non-empty")
        if not self.chunk_type:
            raise ValueError("chunk_type must be non-empty")
        if not self.chunk_id:
            raise ValueError("chunk_id must be non-empty")
        if self.payload_index is not None and self.payload_index < 0:
            raise ValueError("payload_index must be non-negative or None")
        if self.source_byte_start < 0:
            raise ValueError("source_byte_start must be non-negative")
        if self.source_byte_length <= 0:
            raise ValueError("source_byte_length must be positive")
        if self.global_byte_start < 0 or self.global_byte_end <= self.global_byte_start:
            raise ValueError("global byte range must be positive")
        if self.token_start < 0 or self.token_count <= 0:
            raise ValueError("token range must be positive")
        if self.token_start + self.token_count != self.token_end:
            raise ValueError("token_end does not match token_start + token_count")
        if self.first_block_index < 0 or self.last_block_index_exclusive <= self.first_block_index:
            raise ValueError("block range must be positive")
        if not isinstance(self.content_hash, str):
            raise TypeError("content_hash must be a string")

    @property
    def source_byte_end(self) -> int:
        return self.source_byte_start + self.source_byte_length

    @property
    def block_count(self) -> int:
        return self.last_block_index_exclusive - self.first_block_index


@dataclass(frozen=True, slots=True)
class EngineKVBindAction:
    """Bind the imported KV handle to the engine request before scheduling decode."""

    request_id: str
    handle_uri: str
    cache_method: str
    adapter_ids: tuple[str, ...]
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.handle_uri:
            raise ValueError("handle_uri must be non-empty")
        if not self.cache_method:
            raise ValueError("cache_method must be non-empty")
        if any(not isinstance(adapter_id, str) or not adapter_id for adapter_id in self.adapter_ids):
            raise ValueError("adapter_ids entries must be non-empty strings")
        object.__setattr__(self, "adapter_ids", tuple(self.adapter_ids))
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class EngineKVReleaseAction:
    """Release adapter-owned KV state when the serving engine finishes the request."""

    request_id: str

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")


@dataclass(frozen=True, slots=True)
class EngineKVConnectorActions:
    """Reserve/copy/bind/release descriptors for a native engine adapter."""

    reservation: EngineKVReservationAction
    copies: tuple[EngineKVSegmentCopyAction, ...]
    bind: EngineKVBindAction
    release: EngineKVReleaseAction

    def __post_init__(self) -> None:
        request_id = self.reservation.request_id
        if self.bind.request_id != request_id or self.release.request_id != request_id:
            raise ValueError("Connector action request ids must match")
        if any(copy.request_id != request_id for copy in self.copies):
            raise ValueError("Connector copy action request ids must match reservation")
        object.__setattr__(self, "copies", tuple(self.copies))


class EngineKVBlockManagerProbe(Protocol):
    """Validation-only facade over a native vLLM/SGLang KV block manager."""

    def reserve_kv_blocks(self, action: EngineKVReservationAction) -> Any:
        """Reserve native KV blocks and return the engine-owned reservation object."""
        ...

    def import_kv_segment(
        self,
        reservation: Any,
        action: EngineKVSegmentCopyAction,
        payload: memoryview,
    ) -> None:
        """Import one validated payload slice into the reserved native KV blocks."""
        ...

    def bind_kv_handle(self, reservation: Any, action: EngineKVBindAction) -> None:
        """Bind the imported KV reservation to the engine request."""
        ...

    def release_kv_blocks(self, reservation: Any, action: EngineKVReleaseAction) -> None:
        """Release the reserved native KV blocks after validation or decode completion."""
        ...


@dataclass(frozen=True, slots=True)
class EngineKVConnectorProbeResult:
    """Summary from validating connector actions against a native block-manager probe."""

    backend: ServingBackend
    request_id: str
    total_blocks: int
    copied_segments: int
    copied_tokens: int
    copied_bytes: int
    bound: bool
    released: bool
    model_id: str
    layout_version: str
    layout: KVLayout
    payload_mode: PayloadMode
    connector_package: str
    engine_version: str
    native_probe: bool = True
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _backend_from_value(self.backend, field_name="backend"))
        object.__setattr__(self, "payload_mode", _payload_mode_from_value(self.payload_mode, field_name="payload_mode"))
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.layout_version:
            raise ValueError("layout_version must be non-empty")
        self.layout.validate()
        if self.layout.model_id != self.model_id:
            raise ValueError("layout.model_id must match model_id")
        if self.layout.layout_version != self.layout_version:
            raise ValueError("layout.layout_version must match layout_version")
        if not self.connector_package:
            raise ValueError("connector_package must be non-empty")
        _validate_connector_package_matches_backend(self.backend, self.connector_package)
        if not self.engine_version:
            raise ValueError("engine_version must be non-empty")
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def engine_kv_connector_probe_result_to_record(result: EngineKVConnectorProbeResult) -> dict[str, Any]:
    return {
        "record_type": ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE,
        "schema_version": ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION,
        "backend": result.backend.value,
        "request_id": result.request_id,
        "total_blocks": result.total_blocks,
        "copied_segments": result.copied_segments,
        "copied_tokens": result.copied_tokens,
        "copied_bytes": result.copied_bytes,
        "bound": result.bound,
        "released": result.released,
        "model_id": result.model_id,
        "layout_version": result.layout_version,
        "layout": _layout_to_record(result.layout),
        "payload_mode": result.payload_mode.value,
        "connector_package": result.connector_package,
        "engine_version": result.engine_version,
        "native_probe": result.native_probe,
        "metadata": dict(result.metadata),
    }


def validate_engine_kv_connector_probe_record(
    record: Mapping[str, Any],
    *,
    expected_backend: str | ServingBackend | None = None,
) -> None:
    if record.get("record_type") != ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE:
        raise ValueError(f"Unsupported engine KV probe record_type {record.get('record_type')!r}")
    if record.get("schema_version") != ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported engine KV probe schema_version {record.get('schema_version')!r}; "
            f"expected {ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION}"
        )
    backend = _backend_from_value(record.get("backend"), field_name="backend")
    if expected_backend is not None:
        expected = _backend_from_value(expected_backend, field_name="expected_backend")
        if backend != expected:
            raise ValueError(f"Engine KV probe backend {backend.value!r} != expected {expected.value!r}")
    if not isinstance(record.get("request_id"), str) or not record["request_id"]:
        raise ValueError("Engine KV probe request_id must be a non-empty string")
    for field_name in ("total_blocks", "copied_segments", "copied_tokens", "copied_bytes"):
        if not _is_positive_int(record.get(field_name)):
            raise ValueError(f"Engine KV probe {field_name} must be a positive integer")
    if record.get("bound") is not True:
        raise ValueError("Engine KV probe did not bind the KV handle")
    if record.get("released") is not True:
        raise ValueError("Engine KV probe did not release the KV blocks")
    for field_name in ("model_id", "layout_version", "connector_package", "engine_version"):
        if not isinstance(record.get(field_name), str) or not record[field_name]:
            raise ValueError(f"Engine KV probe {field_name} must be a non-empty string")
    _validate_connector_package_matches_backend(backend, record["connector_package"])
    layout = _layout_from_record(_required_mapping(record, "layout"))
    if record["model_id"] != layout.model_id:
        raise ValueError("Engine KV probe model_id must match layout.model_id")
    if record["layout_version"] != layout.layout_version:
        raise ValueError("Engine KV probe layout_version must match layout.layout_version")
    copied_tokens = record["copied_tokens"]
    if record["copied_bytes"] != copied_tokens * layout.bytes_per_token:
        raise ValueError("Engine KV probe copied_bytes must match copied_tokens * layout.bytes_per_token")
    if record["total_blocks"] != _block_count(copied_tokens, layout.block_size):
        raise ValueError("Engine KV probe total_blocks must match copied_tokens and layout.block_size")
    if record["copied_segments"] > copied_tokens:
        raise ValueError("Engine KV probe copied_segments cannot exceed copied_tokens")
    _payload_mode_from_value(record.get("payload_mode"), field_name="payload_mode")
    if record.get("native_probe") is not True:
        raise ValueError("Engine KV probe must be marked native_probe=true")
    metadata = _required_mapping(record, "metadata")
    _validate_metadata_strings(metadata)
    _reject_non_native_probe_metadata(metadata)


def vllm_adapter_spec() -> EngineAdapterSpec:
    return EngineAdapterSpec(
        backend=ServingBackend.VLLM,
        connector_package="vllm",
        kv_injection_method="engine-native-kv-block-import",
        payload_contract=(
            "External adapter reserves vLLM KV-cache blocks, imports or maps the "
            "materialized document KV payload into those blocks, then schedules "
            "decode through the vLLM scheduler."
        ),
        metadata={"engine.scheduler": "vllm"},
    )


def sglang_adapter_spec() -> EngineAdapterSpec:
    return EngineAdapterSpec(
        backend=ServingBackend.SGLANG,
        connector_package="sglang",
        kv_injection_method="runtime-prefix-cache-bind",
        payload_contract=(
            "External adapter binds the materialized document KV handle to an "
            "SGLang runtime request, then lets SGLang own scheduling and decode."
        ),
        metadata={"engine.scheduler": "sglang"},
    )


def build_engine_adapter_request(
    ready_request: EngineReadyRequest,
    *,
    spec: EngineAdapterSpec,
) -> EngineAdapterRequest:
    spec.validate_ready_request(ready_request)
    handle = ready_request.handle
    metadata = {
        **handle.metadata,
        **spec.metadata,
        "document_kv.request_id": handle.request_id,
        "document_kv.handle_uri": handle.handle_uri,
        "document_kv.total_tokens": str(handle.total_tokens),
        "document_kv.total_bytes": str(handle.total_bytes),
        "document_kv.cache_method": handle.cache_method,
        "document_kv.payload_mode": payload_mode_for(ready_request).value,
        "engine.backend": spec.backend.value,
        "engine.connector_package": spec.connector_package,
        "engine.kv_injection_method": spec.kv_injection_method,
        "engine.dynamic_loading": str(spec.supports_dynamic_loading).lower(),
    }
    return EngineAdapterRequest(
        backend=spec.backend,
        ready_request=ready_request,
        connector_package=spec.connector_package,
        kv_injection_method=spec.kv_injection_method,
        payload_contract=spec.payload_contract,
        required_steps=spec.required_steps,
        metadata=metadata,
    )


def engine_adapter_request_to_record(
    request: EngineAdapterRequest,
    *,
    payload_uri: str | None = None,
) -> dict[str, Any]:
    """Serialize an engine handoff without embedding raw KV payload bytes."""

    request.ready_request.validate()
    payload_source_uri = _payload_source_uri(request.handle_uri, payload_uri)
    record = {
        "record_type": ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
        "schema_version": ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
        "backend": request.backend.value,
        "request_id": request.request_id,
        "handle_uri": request.handle_uri,
        "connector_package": request.connector_package,
        "kv_injection_method": request.kv_injection_method,
        "payload_contract": request.payload_contract,
        "payload_mode": request.payload_mode.value,
        "required_steps": list(request.required_steps),
        "metadata": dict(request.metadata),
        "estimated_gpu_bytes": request.ready_request.estimated_gpu_bytes,
        "payload_source": _payload_source_to_record(request, payload_source_uri),
        "handle": _handle_to_record(request.ready_request),
    }
    validate_engine_adapter_request_record(record, require_external_payload_uri=False)
    return record


def write_engine_adapter_request_json(
    request: EngineAdapterRequest,
    path: str | Path,
    *,
    payload_uri: str | None = None,
    require_external_payload_uri: bool = True,
) -> Path:
    record = engine_adapter_request_to_record(request, payload_uri=payload_uri)
    if require_external_payload_uri and record["payload_source"]["availability"] != EXTERNAL_URI_PAYLOAD_SOURCE:
        raise ValueError(
            "write_engine_adapter_request_json requires an adapter-readable payload_uri "
            "or external handle_uri; pass require_external_payload_uri=False for debug-only records"
        )
    target_path = Path(path)
    target_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target_path


def read_engine_adapter_request_json(
    path: str | Path,
    *,
    expected_backend: ServingBackend | str | None = None,
    require_external_payload_uri: bool = True,
) -> dict[str, Any]:
    record = json.loads(local_path(str(path)).read_text(encoding="utf-8"))
    validate_engine_adapter_request_record(
        record,
        expected_backend=expected_backend,
        require_external_payload_uri=require_external_payload_uri,
    )
    return record


def validate_engine_adapter_request_record(
    record: Mapping[str, Any],
    *,
    expected_backend: ServingBackend | str | None = None,
    require_external_payload_uri: bool = True,
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("Engine adapter handoff record must be a mapping")
    if record.get("record_type") != ENGINE_ADAPTER_HANDOFF_RECORD_TYPE:
        raise ValueError(f"Unsupported engine adapter handoff record_type {record.get('record_type')!r}")
    if record.get("schema_version") != ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION:
        raise ValueError(f"Unsupported engine adapter handoff schema_version {record.get('schema_version')!r}")

    backend = _backend_from_value(_required_str(record, "backend"), field_name="backend")
    if expected_backend is not None and backend != _backend_from_value(expected_backend, field_name="expected_backend"):
        raise ValueError(f"Engine adapter handoff backend {backend.value!r} does not match expected_backend")
    payload_mode = _payload_mode_from_value(_required_str(record, "payload_mode"), field_name="payload_mode")
    connector_package = _required_str(record, "connector_package")
    if not connector_package:
        raise ValueError("connector_package must be non-empty")
    _validate_connector_package_matches_backend(backend, connector_package)
    if not _required_str(record, "kv_injection_method"):
        raise ValueError("kv_injection_method must be non-empty")
    if not _required_str(record, "payload_contract"):
        raise ValueError("payload_contract must be non-empty")
    required_steps = _normalize_required_steps(_required_str_sequence(record, "required_steps"))
    if not required_steps:
        raise ValueError("required_steps must be non-empty")
    metadata = _required_mapping(record, "metadata")
    _validate_metadata_strings(metadata)
    _required_nonnegative_int(record, "estimated_gpu_bytes")

    handle = _required_mapping(record, "handle")
    payload_source = _required_mapping(record, "payload_source")
    _validate_payload_source_record(
        payload_source,
        payload_mode=payload_mode,
        require_external_payload_uri=require_external_payload_uri,
    )
    _validate_handle_record(handle)
    if _required_str(handle, "request_id") != _required_str(record, "request_id"):
        raise ValueError("Engine adapter handoff request_id does not match handle.request_id")
    if _required_str(handle, "handle_uri") != _required_str(record, "handle_uri"):
        raise ValueError("Engine adapter handoff handle_uri does not match handle.handle_uri")
    if _required_nonnegative_int(payload_source, "total_bytes") != _required_nonnegative_int(handle, "total_bytes"):
        raise ValueError("payload_source.total_bytes does not match handle.total_bytes")
    if _required_nonnegative_int(payload_source, "segment_count") != len(_required_sequence(handle, "segments")):
        raise ValueError("payload_source.segment_count does not match handle.segments")
    _validate_reserved_record_metadata(record, handle, metadata)


def view_engine_adapter_payload(
    record: Mapping[str, Any],
    payload: bytes | memoryview,
) -> memoryview | tuple[memoryview, ...]:
    """Return zero-copy payload views matching a validated handoff record."""

    validate_engine_adapter_request_record(record, require_external_payload_uri=False)
    if not isinstance(payload, bytes | memoryview):
        raise TypeError("Engine adapter payload must be bytes or memoryview")
    handle = _required_mapping(record, "handle")
    total_bytes = _required_nonnegative_int(handle, "total_bytes")
    payload_view = _byte_memoryview(payload)
    if payload_view.nbytes != total_bytes:
        raise ValueError(f"Engine adapter payload length {payload_view.nbytes} != handle.total_bytes {total_bytes}")
    payload_mode = _payload_mode_from_value(_required_str(record, "payload_mode"), field_name="payload_mode")
    if payload_mode == PayloadMode.MERGED:
        return payload_view
    return tuple(
        payload_view[
            _required_nonnegative_int(segment, "byte_start") : _required_nonnegative_int(segment, "byte_end")
        ]
        for segment in _required_mapping_sequence(handle, "segments")
    )


def split_engine_adapter_payload(record: Mapping[str, Any], payload: bytes | memoryview) -> bytes | tuple[bytes, ...]:
    """Return independent payload bytes for callers that cannot consume memoryviews."""

    payload_view = view_engine_adapter_payload(record, payload)
    if isinstance(payload_view, memoryview):
        if isinstance(payload, bytes) and payload_view.nbytes == len(payload):
            return payload
        return payload_view.tobytes()
    return tuple(segment.tobytes() for segment in payload_view)


def build_engine_kv_injection_plan(
    record: Mapping[str, Any],
    *,
    expected_backend: ServingBackend | str | None = None,
    require_external_payload_uri: bool = True,
) -> EngineKVInjectionPlan:
    validate_engine_adapter_request_record(
        record,
        expected_backend=expected_backend,
        require_external_payload_uri=require_external_payload_uri,
    )
    handle = _required_mapping(record, "handle")
    layout = _layout_from_record(_required_mapping(handle, "layout"))
    total_tokens = _required_nonnegative_int(handle, "total_tokens")
    total_bytes = _required_nonnegative_int(handle, "total_bytes")
    payload_source = _required_mapping(record, "payload_source")
    return EngineKVInjectionPlan(
        backend=_backend_from_value(_required_str(record, "backend"), field_name="backend"),
        request_id=_required_str(record, "request_id"),
        handle_uri=_required_str(record, "handle_uri"),
        connector_package=_required_str(record, "connector_package"),
        kv_injection_method=_required_str(record, "kv_injection_method"),
        payload_mode=_payload_mode_from_value(_required_str(record, "payload_mode"), field_name="payload_mode"),
        payload_source_uri=_optional_str(payload_source, "uri"),
        layout=layout,
        cache_method=_required_str(handle, "cache_method"),
        adapter_ids=tuple(_required_str_sequence(handle, "adapter_ids")),
        total_tokens=total_tokens,
        total_bytes=total_bytes,
        total_blocks=_block_count(total_tokens, layout.block_size),
        estimated_gpu_bytes=_required_nonnegative_int(record, "estimated_gpu_bytes"),
        segments=tuple(
            _segment_binding_from_record(segment, block_size=layout.block_size)
            for segment in _required_mapping_sequence(handle, "segments")
        ),
        metadata=_required_mapping(record, "metadata"),
    )


def build_engine_kv_connector_actions(
    plan: EngineKVInjectionPlan,
    payload_or_segments: bytes | memoryview | tuple[bytes | memoryview, ...],
) -> EngineKVConnectorActions:
    """Create native adapter action descriptors without embedding raw KV bytes."""

    payload_mode = _payload_mode_for_connector_payload(payload_or_segments)
    if payload_mode != plan.payload_mode:
        raise ValueError("Connector payload mode does not match injection plan payload_mode")
    _validate_connector_payload_lengths(plan, payload_or_segments)
    return EngineKVConnectorActions(
        reservation=EngineKVReservationAction(
            backend=plan.backend,
            request_id=plan.request_id,
            total_blocks=plan.total_blocks,
            total_tokens=plan.total_tokens,
            estimated_gpu_bytes=plan.estimated_gpu_bytes,
            layout=plan.layout,
            adapter_ids=plan.adapter_ids,
        ),
        copies=tuple(
            _copy_action_from_binding(
                plan.request_id,
                binding,
                payload_index=None if payload_mode == PayloadMode.MERGED else index,
                source_byte_start=binding.byte_start if payload_mode == PayloadMode.MERGED else 0,
            )
            for index, binding in enumerate(plan.segments)
        ),
        bind=EngineKVBindAction(
            request_id=plan.request_id,
            handle_uri=plan.handle_uri,
            cache_method=plan.cache_method,
            adapter_ids=plan.adapter_ids,
            metadata=plan.metadata,
        ),
        release=EngineKVReleaseAction(request_id=plan.request_id),
    )


def validate_engine_kv_connector_actions(actions: EngineKVConnectorActions) -> None:
    """Validate that reserve/copy/bind/release descriptors cover one contiguous KV payload."""

    if not isinstance(actions, EngineKVConnectorActions):
        raise TypeError("actions must be an EngineKVConnectorActions instance")
    actions.reservation.layout.validate()
    if not actions.copies:
        raise ValueError("Connector actions must include at least one copy action")
    if actions.bind.adapter_ids != actions.reservation.adapter_ids:
        raise ValueError("Connector bind adapter_ids do not match reservation adapter_ids")
    metadata_backend = actions.bind.metadata.get("engine.backend", actions.reservation.backend.value)
    if _backend_from_value(metadata_backend, field_name="engine.backend") != actions.reservation.backend:
        raise ValueError("Connector bind engine.backend metadata does not match reservation backend")
    _validate_connector_package_matches_backend(
        actions.reservation.backend,
        actions.bind.metadata.get("engine.connector_package", actions.reservation.backend.value),
    )

    expected_total_bytes = actions.reservation.total_tokens * actions.reservation.layout.bytes_per_token
    token_cursor = 0
    byte_cursor = 0
    for copy_action in actions.copies:
        if copy_action.token_start != token_cursor:
            raise ValueError(f"Non-contiguous token copy action {copy_action.chunk_id!r}")
        if copy_action.global_byte_start != byte_cursor:
            raise ValueError(f"Non-contiguous byte copy action {copy_action.chunk_id!r}")
        expected_first_block = copy_action.token_start // actions.reservation.layout.block_size
        expected_last_block = _block_count(copy_action.token_end, actions.reservation.layout.block_size)
        if copy_action.first_block_index != expected_first_block:
            raise ValueError(
                f"Copy action {copy_action.chunk_id!r} first_block_index "
                f"{copy_action.first_block_index} != token_start // block_size {expected_first_block}"
            )
        if copy_action.last_block_index_exclusive != expected_last_block:
            raise ValueError(
                f"Copy action {copy_action.chunk_id!r} last_block_index_exclusive "
                f"{copy_action.last_block_index_exclusive} != ceil(token_end / block_size) {expected_last_block}"
            )
        if copy_action.source_byte_length != copy_action.global_byte_end - copy_action.global_byte_start:
            raise ValueError(f"Copy action {copy_action.chunk_id!r} source length does not match global byte span")
        expected_copy_bytes = copy_action.token_count * actions.reservation.layout.bytes_per_token
        if copy_action.source_byte_length != expected_copy_bytes:
            raise ValueError(
                f"Copy action {copy_action.chunk_id!r} source length "
                f"{copy_action.source_byte_length} != token_count * bytes_per_token {expected_copy_bytes}"
            )
        if copy_action.last_block_index_exclusive > actions.reservation.total_blocks:
            raise ValueError(f"Copy action {copy_action.chunk_id!r} block range exceeds reservation")
        token_cursor = copy_action.token_end
        byte_cursor = copy_action.global_byte_end

    if token_cursor != actions.reservation.total_tokens:
        raise ValueError(
            f"Connector copy token coverage {token_cursor} != reservation total_tokens "
            f"{actions.reservation.total_tokens}"
        )
    if byte_cursor != expected_total_bytes:
        raise ValueError(
            f"Connector copy byte coverage {byte_cursor} != reservation expected bytes {expected_total_bytes}"
        )


def probe_engine_kv_connector_actions(
    actions: EngineKVConnectorActions,
    payload_or_segments: bytes | memoryview | tuple[bytes | memoryview, ...],
    probe: EngineKVBlockManagerProbe,
    *,
    engine_version: str = "unknown",
    native_probe: bool = True,
    metadata: Mapping[str, str] | None = None,
) -> EngineKVConnectorProbeResult:
    """Run reserve/import/bind/release descriptors against a native block-manager probe.

    This is a validation harness for vLLM/SGLang integrations. It deliberately
    stops before decode scheduling so the serving engine remains the scheduler.
    """

    validate_engine_kv_connector_actions(actions)
    payload_mode = _payload_mode_for_connector_payload(payload_or_segments)
    reservation = probe.reserve_kv_blocks(actions.reservation)
    if reservation is None:
        raise ValueError("Engine KV block manager probe returned no reservation")

    copied_bytes = 0
    copied_tokens = 0
    bound = False
    released = False
    try:
        for copy_action in actions.copies:
            payload_slice = _payload_view_for_copy_action(copy_action, payload_or_segments, payload_mode=payload_mode)
            probe.import_kv_segment(reservation, copy_action, payload_slice)
            copied_bytes += payload_slice.nbytes
            copied_tokens += copy_action.token_count
        probe.bind_kv_handle(reservation, actions.bind)
        bound = True
    finally:
        probe.release_kv_blocks(reservation, actions.release)
        released = True

    return EngineKVConnectorProbeResult(
        backend=actions.reservation.backend,
        request_id=actions.reservation.request_id,
        total_blocks=actions.reservation.total_blocks,
        copied_segments=len(actions.copies),
        copied_tokens=copied_tokens,
        copied_bytes=copied_bytes,
        bound=bound,
        released=released,
        model_id=actions.reservation.layout.model_id,
        layout_version=actions.reservation.layout.layout_version,
        layout=actions.reservation.layout,
        payload_mode=payload_mode,
        connector_package=actions.bind.metadata.get("engine.connector_package", actions.reservation.backend.value),
        engine_version=engine_version,
        native_probe=native_probe,
        metadata=metadata or {},
    )


def payload_mode_for(request: EngineReadyRequest) -> PayloadMode:
    if isinstance(request.payload, tuple):
        return PayloadMode.SEGMENTED
    return PayloadMode.MERGED


def _payload_mode_for_connector_payload(
    payload_or_segments: bytes | memoryview | tuple[bytes | memoryview, ...],
) -> PayloadMode:
    if isinstance(payload_or_segments, tuple):
        if any(not _is_payload_buffer(segment) for segment in payload_or_segments):
            raise TypeError("Segmented connector payload entries must be bytes or byte-addressable memoryview")
        return PayloadMode.SEGMENTED
    if not _is_payload_buffer(payload_or_segments):
        raise TypeError("Connector payload must be bytes, byte-addressable memoryview, or a tuple of those")
    return PayloadMode.MERGED


def _validate_connector_payload_lengths(
    plan: EngineKVInjectionPlan,
    payload_or_segments: bytes | memoryview | tuple[bytes | memoryview, ...],
) -> None:
    if _is_payload_buffer(payload_or_segments):
        if _payload_nbytes(payload_or_segments) != plan.total_bytes:
            raise ValueError(
                f"Connector payload length {_payload_nbytes(payload_or_segments)} "
                f"!= plan.total_bytes {plan.total_bytes}"
            )
        return
    if len(payload_or_segments) != len(plan.segments):
        raise ValueError("Segmented connector payload count must match injection plan segments")
    for index, (payload, segment) in enumerate(zip(payload_or_segments, plan.segments, strict=True)):
        if _payload_nbytes(payload) != segment.byte_length:
            raise ValueError(
                f"Segmented connector payload {index} byte length {_payload_nbytes(payload)} "
                f"!= segment byte_length {segment.byte_length}"
            )


def _is_payload_buffer(value: Any) -> bool:
    if isinstance(value, bytes):
        return True
    return isinstance(value, memoryview) and value.ndim == 1 and value.itemsize == 1


def _payload_nbytes(payload: bytes | memoryview) -> int:
    if isinstance(payload, memoryview):
        return payload.nbytes
    return len(payload)


def _payload_view_for_copy_action(
    copy_action: EngineKVSegmentCopyAction,
    payload_or_segments: bytes | memoryview | tuple[bytes | memoryview, ...],
    *,
    payload_mode: PayloadMode,
) -> memoryview:
    if payload_mode == PayloadMode.MERGED:
        if not _is_payload_buffer(payload_or_segments):
            raise TypeError("Merged connector payload must be bytes or byte-addressable memoryview")
        if copy_action.payload_index is not None:
            raise ValueError("Merged connector copy actions must use payload_index=None")
        source_view = _byte_memoryview(payload_or_segments)
    else:
        if not isinstance(payload_or_segments, tuple):
            raise TypeError("Segmented connector payload must be a tuple")
        if copy_action.payload_index is None:
            raise ValueError("Segmented connector copy actions must include payload_index")
        if copy_action.payload_index >= len(payload_or_segments):
            raise ValueError("Segmented connector copy action payload_index is out of range")
        source_view = _byte_memoryview(payload_or_segments[copy_action.payload_index])
    if copy_action.source_byte_end > source_view.nbytes:
        raise ValueError(f"Copy action {copy_action.chunk_id!r} source range exceeds payload")
    payload_slice = source_view[copy_action.source_byte_start : copy_action.source_byte_end]
    if payload_slice.nbytes != copy_action.source_byte_length:
        raise ValueError(f"Copy action {copy_action.chunk_id!r} payload slice length mismatch")
    return payload_slice


def _byte_memoryview(payload: bytes | memoryview) -> memoryview:
    view = memoryview(payload)
    if view.ndim == 1 and view.itemsize == 1:
        return view
    try:
        return view.cast("B")
    except TypeError as exc:
        raise TypeError("Engine adapter payload memoryview must be contiguous and byte-addressable") from exc


def _copy_action_from_binding(
    request_id: str,
    binding: EngineKVSegmentBinding,
    *,
    payload_index: int | None,
    source_byte_start: int,
) -> EngineKVSegmentCopyAction:
    return EngineKVSegmentCopyAction(
        request_id=request_id,
        document_id=binding.document_id,
        chunk_type=binding.chunk_type,
        chunk_id=binding.chunk_id,
        payload_index=payload_index,
        source_byte_start=source_byte_start,
        source_byte_length=binding.byte_length,
        global_byte_start=binding.byte_start,
        global_byte_end=binding.byte_end,
        token_start=binding.token_start,
        token_count=binding.token_count,
        token_end=binding.token_end,
        first_block_index=binding.first_block_index,
        last_block_index_exclusive=binding.last_block_index_exclusive,
        content_hash=binding.content_hash,
        cache_tier=binding.cache_tier,
    )


def _handle_to_record(request: EngineReadyRequest) -> dict[str, Any]:
    handle = request.handle
    return {
        "request_id": handle.request_id,
        "handle_uri": handle.handle_uri,
        "total_tokens": handle.total_tokens,
        "total_bytes": handle.total_bytes,
        "cache_method": handle.cache_method,
        "adapter_ids": list(handle.adapter_ids),
        "metadata": dict(handle.metadata),
        "layout": _layout_to_record(handle.layout),
        "segments": [
            _segment_to_record(segment, cache_tier)
            for segment, cache_tier in zip(handle.segments, request.segment_tiers, strict=True)
        ],
    }


def _payload_source_to_record(request: EngineAdapterRequest, payload_uri: str | None) -> dict[str, Any]:
    availability = EXTERNAL_URI_PAYLOAD_SOURCE if payload_uri is not None else IN_PROCESS_PAYLOAD_SOURCE
    return {
        "availability": availability,
        "uri": payload_uri,
        "format": "document_kv.materialized_payload.v1",
        "payload_mode": request.payload_mode.value,
        "total_bytes": request.ready_request.handle.total_bytes,
        "segment_count": len(request.ready_request.handle.segments),
    }


def _payload_source_uri(handle_uri: str, payload_uri: str | None) -> str | None:
    if payload_uri is not None:
        if not _is_external_payload_uri(payload_uri):
            raise ValueError("payload_uri must be an absolute path or adapter-readable URI")
        return payload_uri
    if _is_external_payload_uri(handle_uri):
        return handle_uri
    return None


def _validate_payload_source_record(
    payload_source: Mapping[str, Any],
    *,
    payload_mode: PayloadMode,
    require_external_payload_uri: bool,
) -> None:
    availability = _required_str(payload_source, "availability")
    if availability not in {IN_PROCESS_PAYLOAD_SOURCE, EXTERNAL_URI_PAYLOAD_SOURCE}:
        raise ValueError(f"Unsupported payload_source.availability {availability!r}")
    uri = payload_source.get("uri")
    if availability == EXTERNAL_URI_PAYLOAD_SOURCE:
        if not isinstance(uri, str) or not _is_external_payload_uri(uri):
            raise ValueError("payload_source.uri must be an adapter-readable URI when availability is external_uri")
    elif uri is not None:
        raise ValueError("payload_source.uri must be null when availability is in_process")
    if require_external_payload_uri and availability != EXTERNAL_URI_PAYLOAD_SOURCE:
        raise ValueError("Engine adapter handoff record requires an external payload source")
    if _required_str(payload_source, "format") != "document_kv.materialized_payload.v1":
        raise ValueError(f"Unsupported payload_source.format {payload_source.get('format')!r}")
    source_payload_mode = _payload_mode_from_value(
        _required_str(payload_source, "payload_mode"),
        field_name="payload_source.payload_mode",
    )
    if source_payload_mode != payload_mode:
        raise ValueError("payload_source.payload_mode does not match record payload_mode")
    _required_nonnegative_int(payload_source, "total_bytes")
    _required_nonnegative_int(payload_source, "segment_count")


def _validate_handle_record(handle: Mapping[str, Any]) -> None:
    layout = _required_mapping(handle, "layout")
    kv_layout = _layout_from_record(layout)
    kv_layout.validate()
    segments = _required_mapping_sequence(handle, "segments")
    total_tokens = _required_nonnegative_int(handle, "total_tokens")
    total_bytes = _required_nonnegative_int(handle, "total_bytes")
    if not _required_str(handle, "request_id"):
        raise ValueError("handle.request_id must be non-empty")
    if not _required_str(handle, "handle_uri"):
        raise ValueError("handle.handle_uri must be non-empty")
    if not _required_str(handle, "cache_method"):
        raise ValueError("handle.cache_method must be non-empty")
    if any(
        not isinstance(adapter_id, str) or not adapter_id
        for adapter_id in _required_sequence(handle, "adapter_ids")
    ):
        raise ValueError("handle.adapter_ids entries must be non-empty strings")
    _reject_reserved_metadata(_required_mapping(handle, "metadata"))

    token_cursor = 0
    byte_cursor = 0
    for segment in segments:
        if _required_str(segment, "document_id") == "":
            raise ValueError("segment.document_id must be non-empty")
        if _required_str(segment, "chunk_type") == "":
            raise ValueError("segment.chunk_type must be non-empty")
        if _required_str(segment, "chunk_id") == "":
            raise ValueError("segment.chunk_id must be non-empty")
        _cache_tier_from_value(_required_str(segment, "cache_tier"), field_name="segment.cache_tier")
        token_start = _required_nonnegative_int(segment, "token_start")
        token_count = _required_nonnegative_int(segment, "token_count")
        token_end = _required_nonnegative_int(segment, "token_end")
        byte_start = _required_nonnegative_int(segment, "byte_start")
        byte_length = _required_nonnegative_int(segment, "byte_length")
        byte_end = _required_nonnegative_int(segment, "byte_end")
        if token_count == 0:
            raise ValueError(f"Segment {segment.get('chunk_id')!r} token_count must be positive")
        if byte_length == 0:
            raise ValueError(f"Segment {segment.get('chunk_id')!r} byte_length must be positive")
        expected_byte_length = token_count * kv_layout.bytes_per_token
        if byte_length != expected_byte_length:
            raise ValueError(
                f"Segment {segment.get('chunk_id')!r} byte_length {byte_length} "
                f"!= token_count * bytes_per_token {expected_byte_length}"
            )
        if token_start != token_cursor:
            raise ValueError(f"Non-contiguous token segment {segment.get('chunk_id')!r}")
        if byte_start != byte_cursor:
            raise ValueError(f"Non-contiguous byte segment {segment.get('chunk_id')!r}")
        if token_start + token_count != token_end:
            raise ValueError(f"Segment {segment.get('chunk_id')!r} token_end does not match token range")
        if byte_start + byte_length != byte_end:
            raise ValueError(f"Segment {segment.get('chunk_id')!r} byte_end does not match byte range")
        content_hash = segment.get("content_hash", "")
        if not isinstance(content_hash, str):
            raise TypeError("segment.content_hash must be a string")
        token_cursor = token_end
        byte_cursor = byte_end
    if token_cursor != total_tokens:
        raise ValueError(f"Segment tokens {token_cursor} != handle.total_tokens {total_tokens}")
    if byte_cursor != total_bytes:
        raise ValueError(f"Segment bytes {byte_cursor} != handle.total_bytes {total_bytes}")


def _validate_reserved_record_metadata(
    record: Mapping[str, Any],
    handle: Mapping[str, Any],
    metadata: Mapping[str, str],
) -> None:
    backend = _backend_from_value(_required_str(record, "backend"), field_name="backend")
    payload_mode = _payload_mode_from_value(_required_str(record, "payload_mode"), field_name="payload_mode")
    expected_values = {
        "document_kv.request_id": _required_str(record, "request_id"),
        "document_kv.handle_uri": _required_str(record, "handle_uri"),
        "document_kv.total_tokens": str(_required_nonnegative_int(handle, "total_tokens")),
        "document_kv.total_bytes": str(_required_nonnegative_int(handle, "total_bytes")),
        "document_kv.cache_method": _required_str(handle, "cache_method"),
        "document_kv.payload_mode": payload_mode.value,
        "engine.backend": backend.value,
        "engine.connector_package": _required_str(record, "connector_package"),
        "engine.kv_injection_method": _required_str(record, "kv_injection_method"),
    }
    mismatches = sorted(
        key
        for key, expected_value in expected_values.items()
        if key in metadata and metadata[key] != expected_value
    )
    if mismatches:
        raise ValueError(f"Reserved metadata does not match handoff fields: {', '.join(mismatches)}")


def _layout_from_record(layout: Mapping[str, Any]) -> KVLayout:
    kv_layout = KVLayout(
        model_id=_required_str(layout, "model_id"),
        lora_id=_required_str(layout, "lora_id"),
        layout_version=_required_str(layout, "layout_version"),
        dtype=_required_str(layout, "dtype"),
        num_layers=_required_positive_int(layout, "num_layers"),
        block_size=_required_positive_int(layout, "block_size"),
        bytes_per_token=_required_positive_int(layout, "bytes_per_token"),
        num_query_heads=_optional_positive_int(layout, "num_query_heads"),
        num_kv_heads=_optional_positive_int(layout, "num_kv_heads"),
        head_size=_optional_positive_int(layout, "head_size"),
        kv_stride_bytes=_optional_positive_int(layout, "kv_stride_bytes"),
        shares_kv_storage=_required_bool(layout, "shares_kv_storage"),
        storage_layout=kv_storage_layout_from_value(
            _required_str(layout, "storage_layout"),
            field_name="layout.storage_layout",
        ),
    )
    attention_mechanism = layout.get("attention_mechanism")
    expected_attention = kv_layout.attention_mechanism
    expected_attention_value = expected_attention.value if expected_attention is not None else None
    if attention_mechanism != expected_attention_value:
        raise ValueError("layout.attention_mechanism does not match layout geometry")
    if layout.get("query_heads_per_kv_head") != kv_layout.query_heads_per_kv_head:
        raise ValueError("layout.query_heads_per_kv_head does not match layout geometry")
    kv_layout.validate()
    return kv_layout


def _is_external_payload_uri(uri: str) -> bool:
    if Path(uri).is_absolute():
        return True
    if ":" not in uri:
        return False
    scheme, target = uri.split(":", maxsplit=1)
    scheme = scheme.lower()
    if not scheme or "/" in scheme:
        return False
    if not target:
        return False
    if scheme in {"disk", "file"}:
        return Path(target).is_absolute()
    return scheme in _EXTERNAL_PAYLOAD_URI_SCHEMES


def _segment_binding_from_record(segment: Mapping[str, Any], *, block_size: int) -> EngineKVSegmentBinding:
    token_start = _required_nonnegative_int(segment, "token_start")
    token_end = _required_nonnegative_int(segment, "token_end")
    return EngineKVSegmentBinding(
        document_id=_required_str(segment, "document_id"),
        chunk_type=_required_str(segment, "chunk_type"),
        chunk_id=_required_str(segment, "chunk_id"),
        token_start=token_start,
        token_count=_required_nonnegative_int(segment, "token_count"),
        token_end=token_end,
        byte_start=_required_nonnegative_int(segment, "byte_start"),
        byte_length=_required_nonnegative_int(segment, "byte_length"),
        byte_end=_required_nonnegative_int(segment, "byte_end"),
        first_block_index=token_start // block_size,
        last_block_index_exclusive=_block_count(token_end, block_size),
        content_hash=_optional_str(segment, "content_hash") or "",
        cache_tier=_cache_tier_from_value(_required_str(segment, "cache_tier"), field_name="segment.cache_tier"),
    )


def _block_count(token_count: int, block_size: int) -> int:
    if token_count == 0:
        return 0
    return math.ceil(token_count / block_size)


def _backend_from_value(value: ServingBackend | str, *, field_name: str) -> ServingBackend:
    if isinstance(value, str):
        value = value.strip().lower()
    try:
        return value if isinstance(value, ServingBackend) else ServingBackend(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported {field_name} {value!r}") from exc


def _is_positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _payload_mode_from_value(value: PayloadMode | str, *, field_name: str) -> PayloadMode:
    if isinstance(value, str):
        value = value.strip().lower()
    try:
        return value if isinstance(value, PayloadMode) else PayloadMode(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported {field_name} {value!r}") from exc


def _cache_tier_from_value(value: CacheTier | str, *, field_name: str) -> CacheTier:
    if isinstance(value, str):
        value = value.strip().lower()
    try:
        return value if isinstance(value, CacheTier) else CacheTier(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported {field_name} {value!r}") from exc


def _validate_connector_package_matches_backend(backend: ServingBackend, connector_package: str) -> None:
    if connector_package != backend.value:
        raise ValueError("connector_package must match backend")


def _required_mapping(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = record.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _required_sequence(record: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    value = record.get(key)
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise TypeError(f"{key} must be a sequence")
    return tuple(value)


def _required_mapping_sequence(record: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    values = _required_sequence(record, key)
    if any(not isinstance(value, Mapping) for value in values):
        raise TypeError(f"{key} entries must be mappings")
    return values


def _required_str_sequence(record: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = _required_sequence(record, key)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{key} entries must be non-empty strings")
    return values


def _required_str(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _optional_str(record: Mapping[str, Any], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be null or a string")
    return value


def _required_nonnegative_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _required_positive_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _optional_positive_int(record: Mapping[str, Any], key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be null or a positive integer")
    return value


def _required_bool(record: Mapping[str, Any], key: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _layout_to_record(layout: KVLayout) -> dict[str, Any]:
    attention_mechanism = layout.attention_mechanism
    return {
        "model_id": layout.model_id,
        "lora_id": layout.lora_id,
        "layout_version": layout.layout_version,
        "dtype": layout.dtype,
        "num_layers": layout.num_layers,
        "block_size": layout.block_size,
        "bytes_per_token": layout.bytes_per_token,
        "num_query_heads": layout.num_query_heads,
        "num_kv_heads": layout.num_kv_heads,
        "head_size": layout.head_size,
        "kv_stride_bytes": layout.kv_stride_bytes,
        "shares_kv_storage": layout.shares_kv_storage,
        "storage_layout": layout.storage_layout.value,
        "attention_mechanism": attention_mechanism.value if attention_mechanism is not None else None,
        "query_heads_per_kv_head": layout.query_heads_per_kv_head,
    }


def _segment_to_record(segment: KVSegment, cache_tier: CacheTier | str) -> dict[str, Any]:
    return {
        "document_id": segment.document_id,
        "chunk_type": segment.chunk_type,
        "chunk_id": segment.chunk_id,
        "cache_tier": _cache_tier_from_value(cache_tier, field_name="segment.cache_tier").value,
        "token_start": segment.token_start,
        "token_count": segment.token_count,
        "token_end": segment.token_end,
        "byte_start": segment.byte_start,
        "byte_length": segment.byte_length,
        "byte_end": segment.byte_end,
        "content_hash": segment.content_hash,
    }


def _normalize_required_steps(required_steps: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(required_steps, str):
        raise ValueError("required_steps must be a sequence of non-empty strings")
    normalized = tuple(required_steps)
    if any(not isinstance(step, str) or not step for step in normalized):
        raise ValueError("required_steps entries must be non-empty strings")
    return normalized


def _reject_reserved_metadata(metadata: Mapping[str, str]) -> None:
    _validate_metadata_strings(metadata)
    reserved_keys = sorted(
        key for key in metadata if any(key.startswith(prefix) for prefix in RESERVED_METADATA_PREFIXES)
    )
    if reserved_keys:
        raise ValueError(f"Handle metadata uses reserved adapter keys: {', '.join(reserved_keys)}")


def _validate_metadata_strings(metadata: Mapping[str, str]) -> None:
    invalid_entries = [
        key
        for key, value in metadata.items()
        if not isinstance(key, str) or not isinstance(value, str)
    ]
    if invalid_entries:
        raise TypeError("Adapter metadata keys and values must be strings")


def _reject_non_native_probe_metadata(metadata: Mapping[str, str]) -> None:
    non_native_entries = []
    for key, value in metadata.items():
        normalized_value = value.strip().lower()
        if key.endswith(_PROBE_NATIVE_RUNTIME_METADATA_SUFFIX) and normalized_value != "true":
            non_native_entries.append(key)
        elif key.endswith(_PROBE_KIND_METADATA_SUFFIX) and normalized_value in _NON_NATIVE_PROBE_KIND_VALUES:
            non_native_entries.append(key)
        elif key.endswith(_PROBE_METADATA_SUFFIX) and normalized_value in _NON_NATIVE_PROBE_VALUES:
            non_native_entries.append(key)
    if non_native_entries:
        raise ValueError(
            "Engine KV probe metadata identifies a non-native/debug probe: "
            + ", ".join(sorted(non_native_entries))
        )
