"""Adapter contracts for external vLLM and SGLang KV injection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TypeGuard, TypeVar, cast

from document_kv_cache.artifact_identity import ArtifactIdentity, TokenContract
from document_kv_cache.cache import CacheTier
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_protocol import (
    KVLayout,
    KVPayloadAxisOrder,
    KVSegment,
    KVStorageLayout,
    kv_key_position_encoding_from_value,
    kv_payload_axis_order_from_value,
    kv_storage_layout_from_value,
)
from document_kv_cache.methods import (
    CACHET_CONNECTOR_MODE,
    MethodRegistry,
    default_method_registry,
    validate_registered_reuse_plan,
)
from document_kv_cache.reuse_contract import (
    ArtifactEncoding,
    PayloadDecodeStage,
    PositionHandling,
    ReusePlan,
    RuntimeOperationHandlerRegistry,
    RuntimeOperationPhase,
    TokenRecomputePolicy,
)
from document_kv_cache.storage import local_path

_EnumT = TypeVar("_EnumT", bound=StrEnum)

_VLLM_RUNTIME_DTYPES = (
    "bf16",
    "bfloat16",
    "fp16",
    "float16",
    "fp32",
    "float32",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "float8",
    "int8",
    "uint8",
)
_VLLM_REROPE_DTYPES = tuple(
    dtype for dtype in _VLLM_RUNTIME_DTYPES if dtype not in {"float8", "int8", "uint8"}
)
_SGLANG_REROPE_DTYPES = (
    "bf16",
    "bfloat16",
    "fp16",
    "float16",
    "fp32",
    "float32",
)

RESERVED_METADATA_PREFIXES = ("document_kv.", "engine.")
ENGINE_ADAPTER_HANDOFF_RECORD_TYPE = "document_kv.engine_adapter_request.v1"
ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION = 4
_ENGINE_ADAPTER_HANDOFF_LEGACY_SCHEMA_VERSION = 2
_ENGINE_ADAPTER_HANDOFF_SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {_ENGINE_ADAPTER_HANDOFF_LEGACY_SCHEMA_VERSION, ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION}
)
_ENGINE_ADAPTER_HANDOFF_RECORD_KEYS_V2 = frozenset(
    {
        "record_type",
        "schema_version",
        "backend",
        "request_id",
        "handle_uri",
        "connector_package",
        "kv_injection_method",
        "payload_contract",
        "payload_mode",
        "required_steps",
        "metadata",
        "estimated_gpu_bytes",
        "payload_source",
        "handle",
    }
)
_ENGINE_ADAPTER_HANDOFF_RECORD_KEYS_V4 = (
    _ENGINE_ADAPTER_HANDOFF_RECORD_KEYS_V2 | {"reuse_plan"}
)
ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE = "document_kv.engine_kv_connector_actions.v1"
ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION = 2
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
_ENGINE_KV_CONNECTOR_ACTIONS_RECORD_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "backend",
        "request_id",
        "reservation",
        "copies",
        "bind",
        "release",
        "reuse_plan",
    }
)
_ENGINE_KV_RESERVATION_ACTION_KEYS = frozenset(
    {
        "backend",
        "request_id",
        "total_blocks",
        "total_tokens",
        "estimated_gpu_bytes",
        "adapter_ids",
        "layout",
        "artifact_identity",
        "payload_checksum",
    }
)
_ENGINE_KV_COPY_ACTION_KEYS = frozenset(
    {
        "request_id",
        "document_id",
        "chunk_type",
        "chunk_id",
        "payload_index",
        "source_byte_start",
        "source_byte_length",
        "source_byte_end",
        "global_byte_start",
        "global_byte_end",
        "token_start",
        "token_count",
        "token_end",
        "first_block_index",
        "last_block_index_exclusive",
        "content_hash",
        "token_contract",
        "cache_tier",
    }
)
_ENGINE_KV_BIND_ACTION_KEYS = frozenset({"request_id", "handle_uri", "cache_method", "adapter_ids", "metadata"})
_ENGINE_KV_RELEASE_ACTION_KEYS = frozenset({"request_id"})

__all__ = [
    "EngineAdapterRequest",
    "EngineAdapterSpec",
    "RuntimeOperationSupport",
    "ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE",
    "ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION",
    "ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE",
    "ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION",
    "EngineKVBlockManagerProbe",
    "EngineKVBindAction",
    "EngineKVConnectorActions",
    "EngineKVConnectorProbeResult",
    "EngineKVInjectionPlan",
    "EngineKVReleaseAction",
    "EngineKVReservationAction",
    "EngineKVSegmentCopyAction",
    "EngineKVSegmentBinding",
    "PayloadMode",
    "ServingBackend",
    "build_engine_adapter_request",
    "build_engine_kv_connector_actions",
    "build_engine_kv_injection_plan",
    "engine_kv_connector_actions_from_record",
    "engine_kv_connector_actions_to_record",
    "engine_kv_connector_probe_result_to_record",
    "engine_adapter_request_to_record",
    "payload_mode_for",
    "probe_engine_kv_connector_actions",
    "read_engine_adapter_request_json",
    "sglang_adapter_spec",
    "split_engine_adapter_payload",
    "validate_engine_adapter_request_record",
    "validate_engine_kv_connector_actions_record",
    "validate_engine_kv_connector_probe_record",
    "validate_engine_kv_connector_actions",
    "view_engine_adapter_payload",
    "vllm_adapter_spec",
    "write_engine_adapter_request_json",
]


class ServingBackend(StrEnum):
    VLLM = "vllm"
    SGLANG = "sglang"


class PayloadMode(StrEnum):
    MERGED = "merged"
    SEGMENTED = "segmented"


@dataclass(frozen=True, slots=True)
class RuntimeOperationSupport:
    """One method-owned strategy implemented by an engine adapter backend."""

    phase: RuntimeOperationPhase
    strategy_id: str
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", RuntimeOperationPhase(self.phase))
        _validate_nonempty_str_value(self.strategy_id, field_name="strategy_id")
        _validate_nonempty_str_value(self.version, field_name="version")

    @property
    def key(self) -> tuple[RuntimeOperationPhase, str, str]:
        return self.phase, self.strategy_id, self.version


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
    supported_connector_modes: tuple[str, ...] = (CACHET_CONNECTOR_MODE,)
    supported_artifact_encodings: tuple[ArtifactEncoding, ...] = (
        ArtifactEncoding.RAW_KV,
    )
    supported_position_handling: tuple[PositionHandling, ...] = (
        PositionHandling.STORED_POST_ROPE,
        PositionHandling.REROPE_AT_INJECTION,
    )
    supported_payload_decode_stages: tuple[PayloadDecodeStage, ...] = (
        PayloadDecodeStage.NONE,
    )
    supported_token_recompute_policies: tuple[TokenRecomputePolicy, ...] = (
        TokenRecomputePolicy.NONE,
    )
    supported_storage_layouts: tuple[KVStorageLayout, ...] = tuple(
        KVStorageLayout
    )
    supported_payload_axis_orders: tuple[KVPayloadAxisOrder, ...] = (
        KVPayloadAxisOrder.TOKEN_MAJOR,
    )
    supported_runtime_operations: tuple[RuntimeOperationSupport, ...] = ()
    supported_runtime_dtypes: tuple[str, ...] = _VLLM_RUNTIME_DTYPES
    supported_rerope_dtypes: tuple[str, ...] = _VLLM_REROPE_DTYPES
    supported_rerope_storage_layouts: tuple[KVStorageLayout, ...] = (
        KVStorageLayout.SEPARATE_KEY_VALUE,
    )
    requires_complete_rerope_geometry: bool = True
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
        connector_modes = _nonempty_unique_string_tuple(
            self.supported_connector_modes,
            field_name="supported_connector_modes",
        )
        artifact_encodings = _nonempty_unique_enum_tuple(
            self.supported_artifact_encodings,
            ArtifactEncoding,
            field_name="supported_artifact_encodings",
        )
        position_handling = _nonempty_unique_enum_tuple(
            self.supported_position_handling,
            PositionHandling,
            field_name="supported_position_handling",
        )
        payload_decode_stages = _nonempty_unique_enum_tuple(
            self.supported_payload_decode_stages,
            PayloadDecodeStage,
            field_name="supported_payload_decode_stages",
        )
        recompute_policies = _nonempty_unique_enum_tuple(
            self.supported_token_recompute_policies,
            TokenRecomputePolicy,
            field_name="supported_token_recompute_policies",
        )
        object.__setattr__(self, "supported_connector_modes", connector_modes)
        object.__setattr__(self, "supported_artifact_encodings", artifact_encodings)
        object.__setattr__(self, "supported_position_handling", position_handling)
        object.__setattr__(self, "supported_payload_decode_stages", payload_decode_stages)
        object.__setattr__(
            self,
            "supported_token_recompute_policies",
            recompute_policies,
        )
        storage_layouts = _nonempty_unique_enum_tuple(
            self.supported_storage_layouts,
            KVStorageLayout,
            field_name="supported_storage_layouts",
        )
        payload_axis_orders = _nonempty_unique_enum_tuple(
            self.supported_payload_axis_orders,
            KVPayloadAxisOrder,
            field_name="supported_payload_axis_orders",
        )
        runtime_operations = tuple(self.supported_runtime_operations)
        if any(
            not isinstance(operation, RuntimeOperationSupport)
            for operation in runtime_operations
        ):
            raise TypeError(
                "supported_runtime_operations entries must be "
                "RuntimeOperationSupport instances"
            )
        operation_keys = tuple(operation.key for operation in runtime_operations)
        if len(set(operation_keys)) != len(operation_keys):
            raise ValueError(
                "supported_runtime_operations must not contain duplicate "
                "phase/strategy/version entries"
            )
        object.__setattr__(self, "supported_storage_layouts", storage_layouts)
        object.__setattr__(
            self,
            "supported_payload_axis_orders",
            payload_axis_orders,
        )
        object.__setattr__(
            self,
            "supported_runtime_operations",
            runtime_operations,
        )
        runtime_dtypes = _nonempty_unique_string_tuple(
            self.supported_runtime_dtypes,
            field_name="supported_runtime_dtypes",
        )
        rerope_dtypes = _nonempty_unique_string_tuple(
            self.supported_rerope_dtypes,
            field_name="supported_rerope_dtypes",
        )
        unsupported_rerope_dtypes = sorted(set(rerope_dtypes) - set(runtime_dtypes))
        if unsupported_rerope_dtypes:
            raise ValueError(
                "supported_rerope_dtypes must be a subset of "
                "supported_runtime_dtypes"
            )
        rerope_storage_layouts = _nonempty_unique_enum_tuple(
            self.supported_rerope_storage_layouts,
            KVStorageLayout,
            field_name="supported_rerope_storage_layouts",
        )
        if not set(rerope_storage_layouts).issubset(set(storage_layouts)):
            raise ValueError(
                "supported_rerope_storage_layouts must be a subset of "
                "supported_storage_layouts"
            )
        if type(self.requires_complete_rerope_geometry) is not bool:
            raise TypeError("requires_complete_rerope_geometry must be a boolean")
        object.__setattr__(self, "supported_runtime_dtypes", runtime_dtypes)
        object.__setattr__(self, "supported_rerope_dtypes", rerope_dtypes)
        object.__setattr__(
            self,
            "supported_rerope_storage_layouts",
            rerope_storage_layouts,
        )
        required_steps = _normalize_required_steps(self.required_steps)
        if not required_steps:
            raise ValueError("required_steps must be non-empty")
        object.__setattr__(self, "required_steps", required_steps)
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def validate_ready_request(
        self,
        request: EngineReadyRequest,
        *,
        operation_handlers: RuntimeOperationHandlerRegistry | None = None,
        method_registry: MethodRegistry | None = None,
    ) -> ReusePlan:
        request.validate()
        _reject_reserved_metadata(request.handle.metadata)
        payload_mode = payload_mode_for(request)
        if payload_mode == PayloadMode.SEGMENTED and not self.supports_segmented_payload:
            raise ValueError(f"{self.backend.value} adapter does not support segmented payloads")
        if payload_mode == PayloadMode.MERGED and not self.supports_merged_payload:
            raise ValueError(f"{self.backend.value} adapter does not support merged payloads")
        if request.handle.adapter_ids and not self.supports_lora_adapters:
            raise ValueError(f"{self.backend.value} adapter does not support LoRA adapter ids")
        reuse_plan = _reuse_plan_for_ready_request(
            request,
            method_registry=method_registry,
        )
        self.validate_reuse_plan(
            reuse_plan,
            layout=request.handle.layout,
            artifact_identity=request.handle.artifact_identity,
            operation_handlers=operation_handlers,
            method_registry=method_registry,
        )
        return reuse_plan

    def validate_reuse_plan(
        self,
        reuse_plan: ReusePlan,
        *,
        layout: KVLayout,
        artifact_identity: ArtifactIdentity | None = None,
        operation_handlers: RuntimeOperationHandlerRegistry | None = None,
        method_registry: MethodRegistry | None = None,
    ) -> None:
        """Reject method operations this backend does not explicitly execute."""

        if not isinstance(reuse_plan, ReusePlan):
            raise TypeError("reuse_plan must be a ReusePlan")
        reuse_plan.validate_runtime_layout(layout)
        unsupported: list[str] = []
        if layout.storage_layout not in self.supported_storage_layouts:
            unsupported.append(
                f"storage layout {layout.storage_layout.value!r}"
            )
        if layout.payload_axis_order not in self.supported_payload_axis_orders:
            unsupported.append(
                f"payload axis order {layout.payload_axis_order.value!r}"
            )
        normalized_dtype = layout.dtype.lower()
        if normalized_dtype not in self.supported_runtime_dtypes:
            unsupported.append(f"runtime dtype {layout.dtype!r}")
        if reuse_plan.position_handling == PositionHandling.REROPE_AT_INJECTION:
            if normalized_dtype not in self.supported_rerope_dtypes:
                unsupported.append(f"re-rope dtype {layout.dtype!r}")
            if layout.storage_layout not in self.supported_rerope_storage_layouts:
                unsupported.append(
                    "re-rope storage layout "
                    f"{layout.storage_layout.value!r}"
                )
            if self.requires_complete_rerope_geometry:
                missing_geometry = [
                    field_name
                    for field_name in (
                        "num_query_heads",
                        "num_kv_heads",
                        "head_size",
                        "kv_stride_bytes",
                    )
                    if getattr(layout, field_name) is None
                ]
                if missing_geometry:
                    unsupported.append(
                        "re-rope geometry missing " + ", ".join(missing_geometry)
                    )
        if reuse_plan.connector_mode not in self.supported_connector_modes:
            unsupported.append(f"connector mode {reuse_plan.connector_mode!r}")
        if reuse_plan.artifact_format.encoding not in self.supported_artifact_encodings:
            unsupported.append(
                f"artifact encoding {reuse_plan.artifact_format.encoding.value!r}"
            )
        if reuse_plan.position_handling not in self.supported_position_handling:
            unsupported.append(
                f"position handling {reuse_plan.position_handling.value!r}"
            )
        if reuse_plan.payload_decode_stage not in self.supported_payload_decode_stages:
            unsupported.append(
                f"payload decode stage {reuse_plan.payload_decode_stage.value!r}"
            )
        if (
            reuse_plan.token_recompute_policy
            not in self.supported_token_recompute_policies
        ):
            unsupported.append(
                "token recompute policy "
                f"{reuse_plan.token_recompute_policy.value!r}"
            )
        if unsupported:
            raise ValueError(
                f"{self.backend.value} adapter does not support reuse plan "
                f"{reuse_plan.capability_id}: {', '.join(unsupported)}"
            )
        registry = _operation_handler_registry(operation_handlers)
        supported_operation_keys = {
            operation.key for operation in self.supported_runtime_operations
        }
        for phase, descriptor in reuse_plan.runtime_operations:
            operation_key = (
                phase,
                descriptor.strategy_id,
                descriptor.version,
            )
            if operation_key not in supported_operation_keys:
                raise ValueError(
                    f"{self.backend.value} adapter does not advertise runtime "
                    "operation "
                    f"{phase.value}:{descriptor.strategy_version_id}"
                )
            try:
                registry.resolve(phase, descriptor)
            except KeyError as exc:
                raise ValueError(
                    f"{self.backend.value} adapter has no injected handler for "
                    f"{phase.value}:{descriptor.strategy_version_id}"
                ) from exc
        validate_registered_reuse_plan(
            reuse_plan,
            artifact_identity=artifact_identity,
            registry=_method_registry(method_registry),
        )


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
    reuse_plan: ReusePlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _backend_from_value(self.backend, field_name="backend"))
        _validate_connector_package_matches_backend(self.backend, self.connector_package)
        object.__setattr__(self, "required_steps", _normalize_required_steps(self.required_steps))
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.reuse_plan is not None:
            if not isinstance(self.reuse_plan, ReusePlan):
                raise TypeError("reuse_plan must be a ReusePlan or None")
            if self.reuse_plan.method_id != self.ready_request.handle.cache_method:
                raise ValueError("reuse_plan.method_id must match handle.cache_method")

    @property
    def request_id(self) -> str:
        request_id = self.ready_request.request_id
        if not isinstance(request_id, str):
            raise TypeError("ready_request.request_id must be a string")
        return request_id

    @property
    def handle_uri(self) -> str:
        handle_uri = self.ready_request.handle.handle_uri
        if not isinstance(handle_uri, str):
            raise TypeError("ready_request.handle.handle_uri must be a string")
        return handle_uri

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
    token_contract: TokenContract | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_tier", _cache_tier_from_value(self.cache_tier, field_name="cache_tier"))
        _validate_nonempty_str_value(self.document_id, field_name="document_id")
        _validate_nonempty_str_value(self.chunk_type, field_name="chunk_type")
        _validate_nonempty_str_value(self.chunk_id, field_name="chunk_id")
        for field_name in (
            "token_start",
            "token_count",
            "token_end",
            "byte_start",
            "byte_length",
            "byte_end",
            "first_block_index",
            "last_block_index_exclusive",
        ):
            _validate_nonnegative_int_value(getattr(self, field_name), field_name=field_name)
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.byte_length <= 0:
            raise ValueError("byte_length must be positive")
        if self.token_start + self.token_count != self.token_end:
            raise ValueError("token_end does not match token_start + token_count")
        if self.byte_start + self.byte_length != self.byte_end:
            raise ValueError("byte_end does not match byte_start + byte_length")
        if self.last_block_index_exclusive <= self.first_block_index:
            raise ValueError("block range must be positive")
        if not isinstance(self.content_hash, str):
            raise TypeError("content_hash must be a string")
        if self.token_contract is not None:
            if not isinstance(self.token_contract, TokenContract):
                raise TypeError("token_contract must be a TokenContract or None")
            if self.token_contract.token_count != self.token_count:
                raise ValueError("token_contract token_count must match token_count")

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
    reuse_plan: ReusePlan
    artifact_identity: ArtifactIdentity | None = None
    payload_checksum: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _backend_from_value(self.backend, field_name="backend"))
        object.__setattr__(self, "payload_mode", _payload_mode_from_value(self.payload_mode, field_name="payload_mode"))
        _validate_connector_package_matches_backend(self.backend, self.connector_package)
        _validate_nonempty_str_value(self.request_id, field_name="request_id")
        _validate_nonempty_str_value(self.handle_uri, field_name="handle_uri")
        _validate_nonempty_str_value(self.kv_injection_method, field_name="kv_injection_method")
        _validate_nonempty_str_value(self.cache_method, field_name="cache_method")
        if not isinstance(self.reuse_plan, ReusePlan):
            raise TypeError("reuse_plan must be a ReusePlan")
        if self.reuse_plan.method_id != self.cache_method:
            raise ValueError("reuse_plan.method_id must match cache_method")
        self.reuse_plan.validate_runtime_layout(self.layout)
        if self.artifact_identity is not None:
            if not isinstance(self.artifact_identity, ArtifactIdentity):
                raise TypeError("artifact_identity must be an ArtifactIdentity or None")
            if self.artifact_identity.method_id != self.cache_method:
                raise ValueError("artifact_identity.method_id must match cache_method")
            _validate_reuse_plan_artifact_identity(
                self.reuse_plan,
                self.artifact_identity,
            )
        _validate_optional_sha256(self.payload_checksum, field_name="payload_checksum")
        self.layout.validate()
        _validate_nonnegative_int_value(self.total_tokens, field_name="total_tokens")
        _validate_nonnegative_int_value(self.total_bytes, field_name="total_bytes")
        _validate_nonnegative_int_value(self.total_blocks, field_name="total_blocks")
        _validate_nonnegative_int_value(self.estimated_gpu_bytes, field_name="estimated_gpu_bytes")
        _validate_plan_totals(
            total_tokens=self.total_tokens,
            total_bytes=self.total_bytes,
            total_blocks=self.total_blocks,
            layout=self.layout,
            reuse_plan=self.reuse_plan,
        )
        adapter_ids = _normalize_connector_adapter_ids(self.adapter_ids)
        segments = tuple(self.segments)
        if any(not isinstance(segment, EngineKVSegmentBinding) for segment in segments):
            raise TypeError("segments entries must be EngineKVSegmentBinding instances")
        _validate_injection_plan_segments(
            segments,
            total_tokens=self.total_tokens,
            total_bytes=self.total_bytes,
            layout=self.layout,
            reuse_plan=self.reuse_plan,
        )
        object.__setattr__(self, "adapter_ids", adapter_ids)
        object.__setattr__(self, "segments", segments)
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
    artifact_identity: ArtifactIdentity | None = None
    payload_checksum: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _backend_from_value(self.backend, field_name="backend"))
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        self.layout.validate()
        if self.total_blocks != _block_count(self.total_tokens, self.layout.block_size):
            raise ValueError("total_blocks does not match total_tokens and layout.block_size")
        _validate_nonnegative_int_value(self.estimated_gpu_bytes, field_name="estimated_gpu_bytes")
        object.__setattr__(self, "adapter_ids", _normalize_connector_adapter_ids(self.adapter_ids))
        if self.artifact_identity is not None and not isinstance(self.artifact_identity, ArtifactIdentity):
            raise TypeError("artifact_identity must be an ArtifactIdentity or None")
        if self.artifact_identity is not None:
            identity_layout = {
                "model_id": self.layout.model_id,
                "lora_id": self.layout.lora_id,
                "layout_version": self.layout.layout_version,
                "runtime_kv_dtype": self.layout.dtype,
                "block_size": self.layout.block_size,
                "payload_axis_order": self.layout.payload_axis_order.value,
                "key_position_encoding": (
                    self.layout.key_position_encoding.value
                ),
                "rope_theta": self.layout.rope_theta,
                "rope_rotary_dim": self.layout.rope_rotary_dim,
            }
            mismatches = [
                name
                for name, value in identity_layout.items()
                if getattr(self.artifact_identity, name) != value
            ]
            if mismatches:
                raise ValueError(
                    "artifact_identity does not match reservation layout: "
                    + ", ".join(mismatches)
                )
        _validate_optional_sha256(self.payload_checksum, field_name="payload_checksum")


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
    token_contract: TokenContract | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_tier", _cache_tier_from_value(self.cache_tier, field_name="cache_tier"))
        _validate_nonempty_str_value(self.request_id, field_name="request_id")
        _validate_nonempty_str_value(self.document_id, field_name="document_id")
        _validate_nonempty_str_value(self.chunk_type, field_name="chunk_type")
        _validate_nonempty_str_value(self.chunk_id, field_name="chunk_id")
        if self.payload_index is not None:
            _validate_nonnegative_int_value(self.payload_index, field_name="payload_index")
        for field_name in (
            "source_byte_start",
            "source_byte_length",
            "global_byte_start",
            "global_byte_end",
            "token_start",
            "token_count",
            "token_end",
            "first_block_index",
            "last_block_index_exclusive",
        ):
            _validate_nonnegative_int_value(getattr(self, field_name), field_name=field_name)
        if self.source_byte_length <= 0:
            raise ValueError("source_byte_length must be positive")
        if self.global_byte_start + self.source_byte_length != self.global_byte_end:
            raise ValueError("global_byte_end does not match global_byte_start + source_byte_length")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.token_start + self.token_count != self.token_end:
            raise ValueError("token_end does not match token_start + token_count")
        if self.last_block_index_exclusive <= self.first_block_index:
            raise ValueError("block range must be positive")
        if not isinstance(self.content_hash, str):
            raise TypeError("content_hash must be a string")
        if self.token_contract is not None:
            if not isinstance(self.token_contract, TokenContract):
                raise TypeError("token_contract must be a TokenContract or None")
            if self.token_contract.token_count != self.token_count:
                raise ValueError("token_contract token_count must match token_count")

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
        object.__setattr__(self, "adapter_ids", _normalize_connector_adapter_ids(self.adapter_ids))
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
    reuse_plan: ReusePlan | None = None

    def __post_init__(self) -> None:
        request_id = self.reservation.request_id
        if self.bind.request_id != request_id or self.release.request_id != request_id:
            raise ValueError("Connector action request ids must match")
        if any(copy.request_id != request_id for copy in self.copies):
            raise ValueError("Connector copy action request ids must match reservation")
        object.__setattr__(self, "copies", tuple(self.copies))
        reuse_plan = self.reuse_plan
        if reuse_plan is None:
            try:
                reuse_plan = default_method_registry().get(
                    self.bind.cache_method,
                    require_implemented=True,
                ).reuse_plan()
            except (KeyError, NotImplementedError) as exc:
                raise ValueError(
                    "custom connector actions require an explicit reuse_plan"
                ) from exc
        if not isinstance(reuse_plan, ReusePlan):
            raise TypeError("reuse_plan must be a ReusePlan")
        if reuse_plan.method_id != self.bind.cache_method:
            raise ValueError("reuse_plan.method_id must match bind.cache_method")
        reuse_plan.validate_runtime_layout(self.reservation.layout)
        _validate_reuse_plan_artifact_identity(
            reuse_plan,
            self.reservation.artifact_identity,
        )
        object.__setattr__(self, "reuse_plan", reuse_plan)


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
    backend = _backend_from_value(
        _required_str(record, "backend"),
        field_name="backend",
    )
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
    _payload_mode_from_value(
        _required_str(record, "payload_mode"),
        field_name="payload_mode",
    )
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
        supported_storage_layouts=(
            KVStorageLayout.SEPARATE_KEY_VALUE,
            KVStorageLayout.SHARED_KEY_VALUE,
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
        supported_storage_layouts=(
            KVStorageLayout.SEPARATE_KEY_VALUE,
            KVStorageLayout.SHARED_KEY_VALUE,
        ),
        supported_rerope_dtypes=_SGLANG_REROPE_DTYPES,
        metadata={"engine.scheduler": "sglang"},
    )


def build_engine_adapter_request(
    ready_request: EngineReadyRequest,
    *,
    spec: EngineAdapterSpec,
    operation_handlers: RuntimeOperationHandlerRegistry | None = None,
    method_registry: MethodRegistry | None = None,
) -> EngineAdapterRequest:
    reuse_plan = spec.validate_ready_request(
        ready_request,
        operation_handlers=operation_handlers,
        method_registry=method_registry,
    )
    handle = ready_request.handle
    artifact_identity = handle.artifact_identity
    metadata = {
        **handle.metadata,
        **spec.metadata,
        "document_kv.request_id": handle.request_id,
        "document_kv.handle_uri": handle.handle_uri,
        "document_kv.total_tokens": str(handle.total_tokens),
        "document_kv.total_bytes": str(handle.total_bytes),
        "document_kv.cache_method": handle.cache_method,
        "document_kv.artifact_id": (
            "" if artifact_identity is None else artifact_identity.artifact_id
        ),
        "document_kv.method_version": (
            "" if artifact_identity is None else artifact_identity.method_version
        ),
        "document_kv.method_config_digest": (
            ""
            if artifact_identity is None
            else artifact_identity.method_config_digest
        ),
        "document_kv.payload_checksum": handle.payload_checksum,
        "document_kv.payload_mode": payload_mode_for(ready_request).value,
        "document_kv.reuse_capability_id": reuse_plan.capability_id,
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
        reuse_plan=reuse_plan,
    )


def engine_adapter_request_to_record(
    request: EngineAdapterRequest,
    *,
    payload_uri: str | None = None,
    adapter_spec: EngineAdapterSpec | None = None,
    operation_handlers: RuntimeOperationHandlerRegistry | None = None,
    method_registry: MethodRegistry | None = None,
) -> dict[str, Any]:
    """Serialize an engine handoff without embedding raw KV payload bytes."""

    request.ready_request.validate()
    if request.reuse_plan is None:
        raise ValueError("strict engine handoff requires a reuse_plan")
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
        "reuse_plan": request.reuse_plan.to_record(),
        "payload_source": _payload_source_to_record(request, payload_source_uri),
        "handle": _handle_to_record(request.ready_request),
    }
    validate_engine_adapter_request_record(
        record,
        require_external_payload_uri=False,
        adapter_spec=adapter_spec,
        operation_handlers=operation_handlers,
        method_registry=method_registry,
    )
    return record


def write_engine_adapter_request_json(
    request: EngineAdapterRequest,
    path: str | Path,
    *,
    payload_uri: str | None = None,
    require_external_payload_uri: bool = True,
    adapter_spec: EngineAdapterSpec | None = None,
    operation_handlers: RuntimeOperationHandlerRegistry | None = None,
    method_registry: MethodRegistry | None = None,
) -> Path:
    record = engine_adapter_request_to_record(
        request,
        payload_uri=payload_uri,
        adapter_spec=adapter_spec,
        operation_handlers=operation_handlers,
        method_registry=method_registry,
    )
    if require_external_payload_uri and record["payload_source"]["availability"] != EXTERNAL_URI_PAYLOAD_SOURCE:
        raise ValueError(
            "write_engine_adapter_request_json requires an adapter-readable payload_uri "
            "or external handle_uri; pass require_external_payload_uri=False for debug-only records"
        )
    target_path = Path(local_path(str(path)))
    target_path.parent.mkdir(parents=True, exist_ok=True)
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
    allow_legacy_reuse_plan: bool = False,
    adapter_spec: EngineAdapterSpec | None = None,
    operation_handlers: RuntimeOperationHandlerRegistry | None = None,
    method_registry: MethodRegistry | None = None,
) -> dict[str, Any]:
    loaded: object = json.loads(
        local_path(str(path)).read_text(encoding="utf-8")
    )
    if not isinstance(loaded, dict):
        raise TypeError("Engine adapter handoff JSON must contain an object")
    record = cast(dict[str, Any], loaded)
    validate_engine_adapter_request_record(
        record,
        expected_backend=expected_backend,
        require_external_payload_uri=require_external_payload_uri,
        allow_legacy_reuse_plan=allow_legacy_reuse_plan,
        adapter_spec=adapter_spec,
        operation_handlers=operation_handlers,
        method_registry=method_registry,
    )
    return record


def validate_engine_adapter_request_record(
    record: Mapping[str, Any],
    *,
    expected_backend: ServingBackend | str | None = None,
    require_external_payload_uri: bool = True,
    allow_legacy_reuse_plan: bool = False,
    adapter_spec: EngineAdapterSpec | None = None,
    operation_handlers: RuntimeOperationHandlerRegistry | None = None,
    method_registry: MethodRegistry | None = None,
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("Engine adapter handoff record must be a mapping")
    if record.get("record_type") != ENGINE_ADAPTER_HANDOFF_RECORD_TYPE:
        raise ValueError(f"Unsupported engine adapter handoff record_type {record.get('record_type')!r}")
    schema_version = record.get("schema_version")
    if schema_version not in _ENGINE_ADAPTER_HANDOFF_SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            "Unsupported engine adapter handoff schema_version "
            f"{schema_version!r}"
        )
    if (
        schema_version == _ENGINE_ADAPTER_HANDOFF_LEGACY_SCHEMA_VERSION
        and not allow_legacy_reuse_plan
    ):
        raise ValueError(
            "schema_version 2 omits reuse_plan; pass "
            "allow_legacy_reuse_plan=True only for legacy raw-KV handoffs"
        )
    expected_record_keys = (
        _ENGINE_ADAPTER_HANDOFF_RECORD_KEYS_V4
        if schema_version == ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION
        else _ENGINE_ADAPTER_HANDOFF_RECORD_KEYS_V2
    )
    _reject_unsupported_keys(
        record,
        expected_record_keys,
        label="engine adapter handoff record",
    )

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
    reuse_plan = _reuse_plan_from_handoff_record(
        record,
        handle=handle,
        allow_legacy=allow_legacy_reuse_plan,
        method_registry=method_registry,
    )
    payload_source = _required_mapping(record, "payload_source")
    _validate_payload_source_record(
        payload_source,
        payload_mode=payload_mode,
        require_external_payload_uri=require_external_payload_uri,
    )
    _validate_handle_record(handle, reuse_plan=reuse_plan)
    if reuse_plan.method_id != _required_str(handle, "cache_method"):
        raise ValueError("reuse_plan.method_id does not match handle.cache_method")
    layout = _layout_from_record(_required_mapping(handle, "layout"))
    resolved_adapter_spec = (
        _adapter_spec_for_backend(backend)
        if adapter_spec is None
        else adapter_spec
    )
    if not isinstance(resolved_adapter_spec, EngineAdapterSpec):
        raise TypeError("adapter_spec must be an EngineAdapterSpec or None")
    if resolved_adapter_spec.backend != backend:
        raise ValueError("adapter_spec.backend does not match handoff backend")
    _validate_reuse_plan_artifact_identity(
        reuse_plan,
        _optional_artifact_identity(handle, "artifact_identity"),
    )
    resolved_adapter_spec.validate_reuse_plan(
        reuse_plan,
        layout=layout,
        artifact_identity=_optional_artifact_identity(handle, "artifact_identity"),
        operation_handlers=operation_handlers,
        method_registry=method_registry,
    )
    if _required_str(handle, "request_id") != _required_str(record, "request_id"):
        raise ValueError("Engine adapter handoff request_id does not match handle.request_id")
    if _required_str(handle, "handle_uri") != _required_str(record, "handle_uri"):
        raise ValueError("Engine adapter handoff handle_uri does not match handle.handle_uri")
    if _required_nonnegative_int(payload_source, "total_bytes") != _required_nonnegative_int(handle, "total_bytes"):
        raise ValueError("payload_source.total_bytes does not match handle.total_bytes")
    if _required_nonnegative_int(payload_source, "segment_count") != len(_required_sequence(handle, "segments")):
        raise ValueError("payload_source.segment_count does not match handle.segments")
    if (_optional_str(payload_source, "checksum") or "") != (
        _optional_str(handle, "payload_checksum") or ""
    ):
        raise ValueError("payload_source.checksum does not match handle.payload_checksum")
    _validate_reserved_record_metadata(
        record,
        handle,
        metadata,
        method_registry=method_registry,
    )


def view_engine_adapter_payload(
    record: Mapping[str, Any],
    payload: bytes | memoryview,
    *,
    adapter_spec: EngineAdapterSpec | None = None,
    operation_handlers: RuntimeOperationHandlerRegistry | None = None,
    method_registry: MethodRegistry | None = None,
) -> memoryview | tuple[memoryview, ...]:
    """Return zero-copy payload views matching a validated handoff record."""

    validate_engine_adapter_request_record(
        record,
        require_external_payload_uri=False,
        adapter_spec=adapter_spec,
        operation_handlers=operation_handlers,
        method_registry=method_registry,
    )
    if not isinstance(payload, bytes | memoryview):
        raise TypeError("Engine adapter payload must be bytes or memoryview")
    handle = _required_mapping(record, "handle")
    total_bytes = _required_nonnegative_int(handle, "total_bytes")
    payload_view = _byte_memoryview(payload)
    if payload_view.nbytes != total_bytes:
        raise ValueError(f"Engine adapter payload length {payload_view.nbytes} != handle.total_bytes {total_bytes}")
    expected_checksum = _optional_str(handle, "payload_checksum") or ""
    if expected_checksum and hashlib.sha256(payload_view).hexdigest() != expected_checksum:
        raise ValueError("Engine adapter payload checksum does not match handle.payload_checksum")
    payload_mode = _payload_mode_from_value(_required_str(record, "payload_mode"), field_name="payload_mode")
    if payload_mode == PayloadMode.MERGED:
        return payload_view
    return tuple(
        payload_view[
            _required_nonnegative_int(segment, "byte_start") : _required_nonnegative_int(segment, "byte_end")
        ]
        for segment in _required_mapping_sequence(handle, "segments")
    )


def split_engine_adapter_payload(
    record: Mapping[str, Any],
    payload: bytes | memoryview,
    *,
    adapter_spec: EngineAdapterSpec | None = None,
    operation_handlers: RuntimeOperationHandlerRegistry | None = None,
    method_registry: MethodRegistry | None = None,
) -> bytes | tuple[bytes, ...]:
    """Return independent payload bytes for callers that cannot consume memoryviews."""

    payload_view = view_engine_adapter_payload(
        record,
        payload,
        adapter_spec=adapter_spec,
        operation_handlers=operation_handlers,
        method_registry=method_registry,
    )
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
    allow_legacy_reuse_plan: bool = False,
    adapter_spec: EngineAdapterSpec | None = None,
    operation_handlers: RuntimeOperationHandlerRegistry | None = None,
    method_registry: MethodRegistry | None = None,
) -> EngineKVInjectionPlan:
    validate_engine_adapter_request_record(
        record,
        expected_backend=expected_backend,
        require_external_payload_uri=require_external_payload_uri,
        allow_legacy_reuse_plan=allow_legacy_reuse_plan,
        adapter_spec=adapter_spec,
        operation_handlers=operation_handlers,
        method_registry=method_registry,
    )
    handle = _required_mapping(record, "handle")
    layout = _layout_from_record(_required_mapping(handle, "layout"))
    total_tokens = _required_nonnegative_int(handle, "total_tokens")
    total_bytes = _required_nonnegative_int(handle, "total_bytes")
    payload_source = _required_mapping(record, "payload_source")
    reuse_plan = _reuse_plan_from_handoff_record(
        record,
        handle=handle,
        allow_legacy=allow_legacy_reuse_plan,
        method_registry=method_registry,
    )
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
        adapter_ids=_required_adapter_ids(handle, "adapter_ids"),
        total_tokens=total_tokens,
        total_bytes=total_bytes,
        total_blocks=_block_count(total_tokens, layout.block_size),
        estimated_gpu_bytes=_required_nonnegative_int(record, "estimated_gpu_bytes"),
        segments=tuple(
            _segment_binding_from_record(segment, block_size=layout.block_size)
            for segment in _required_mapping_sequence(handle, "segments")
        ),
        metadata=_required_mapping(record, "metadata"),
        reuse_plan=reuse_plan,
        artifact_identity=_optional_artifact_identity(handle, "artifact_identity"),
        payload_checksum=_optional_str(handle, "payload_checksum") or "",
    )


def build_engine_kv_connector_actions(
    plan: EngineKVInjectionPlan,
    payload_or_segments: bytes | memoryview | tuple[bytes | memoryview, ...],
    *,
    method_registry: MethodRegistry | None = None,
) -> EngineKVConnectorActions:
    """Create native adapter action descriptors without embedding raw KV bytes."""

    validate_registered_reuse_plan(
        plan.reuse_plan,
        artifact_identity=plan.artifact_identity,
        registry=_method_registry(method_registry),
    )
    payload_mode = _payload_mode_for_connector_payload(payload_or_segments)
    if payload_mode != plan.payload_mode:
        raise ValueError("Connector payload mode does not match injection plan payload_mode")
    _validate_connector_payload_lengths(plan, payload_or_segments)
    _validate_connector_payload_checksum(plan, payload_or_segments)
    return EngineKVConnectorActions(
        reservation=EngineKVReservationAction(
            backend=plan.backend,
            request_id=plan.request_id,
            total_blocks=plan.total_blocks,
            total_tokens=plan.total_tokens,
            estimated_gpu_bytes=plan.estimated_gpu_bytes,
            layout=plan.layout,
            adapter_ids=plan.adapter_ids,
            artifact_identity=plan.artifact_identity,
            payload_checksum=plan.payload_checksum,
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
        reuse_plan=plan.reuse_plan,
    )


def validate_engine_kv_connector_actions(
    actions: EngineKVConnectorActions,
    *,
    method_registry: MethodRegistry | None = None,
) -> None:
    """Validate that reserve/copy/bind/release descriptors cover one contiguous KV payload."""

    if not isinstance(actions, EngineKVConnectorActions):
        raise TypeError("actions must be an EngineKVConnectorActions instance")
    actions.reservation.layout.validate()
    if not actions.copies:
        raise ValueError("Connector actions must include at least one copy action")
    if actions.bind.adapter_ids != actions.reservation.adapter_ids:
        raise ValueError("Connector bind adapter_ids do not match reservation adapter_ids")
    artifact_identity = actions.reservation.artifact_identity
    if artifact_identity is not None and actions.bind.cache_method != artifact_identity.method_id:
        raise ValueError("Connector bind cache_method does not match artifact identity")
    if actions.reuse_plan is None:  # pragma: no cover - normalized in __post_init__.
        raise ValueError("Connector actions require a reuse_plan")
    if actions.reuse_plan.method_id != actions.bind.cache_method:
        raise ValueError("Connector reuse_plan does not match bind cache_method")
    _validate_reuse_plan_artifact_identity(actions.reuse_plan, artifact_identity)
    if artifact_identity is not None:
        expected_identity_metadata = {
            "document_kv.artifact_id": artifact_identity.artifact_id,
            "document_kv.method_version": artifact_identity.method_version,
            "document_kv.method_config_digest": (
                artifact_identity.method_config_digest
            ),
            "document_kv.reuse_capability_id": (
                actions.reuse_plan.capability_id
            ),
        }
        identity_metadata_mismatches = tuple(
            key
            for key, expected in expected_identity_metadata.items()
            if actions.bind.metadata.get(key) != expected
        )
        if identity_metadata_mismatches:
            raise ValueError(
                "Connector bind metadata does not match artifact identity: "
                + ", ".join(identity_metadata_mismatches)
            )
    metadata_backend = actions.bind.metadata.get("engine.backend", actions.reservation.backend.value)
    if _backend_from_value(metadata_backend, field_name="engine.backend") != actions.reservation.backend:
        raise ValueError("Connector bind engine.backend metadata does not match reservation backend")
    _validate_connector_package_matches_backend(
        actions.reservation.backend,
        actions.bind.metadata.get("engine.connector_package", actions.reservation.backend.value),
    )

    expected_runtime_bytes = (
        actions.reservation.total_tokens
        * actions.reservation.layout.bytes_per_token
    )
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
        expected_copy_bytes = (
            copy_action.token_count
            * actions.reservation.layout.bytes_per_token
        )
        if (
            actions.reuse_plan.artifact_format.encoding
            == ArtifactEncoding.RAW_KV
            and copy_action.source_byte_length != expected_copy_bytes
        ):
            raise ValueError(
                f"Copy action {copy_action.chunk_id!r} source length "
                f"{copy_action.source_byte_length} != token_count * bytes_per_token {expected_copy_bytes}"
            )
        if copy_action.token_contract is not None and artifact_identity is not None:
            if copy_action.token_contract.tokenizer_id != artifact_identity.tokenizer_id:
                raise ValueError("Copy token_contract tokenizer_id does not match artifact identity")
            if copy_action.token_contract.tokenizer_revision != artifact_identity.tokenizer_revision:
                raise ValueError(
                    "Copy token_contract tokenizer_revision does not match artifact identity"
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
    if (
        actions.reuse_plan.artifact_format.encoding
        == ArtifactEncoding.RAW_KV
        and byte_cursor != expected_runtime_bytes
    ):
        raise ValueError(
            f"Connector copy byte coverage {byte_cursor} != reservation expected bytes {expected_runtime_bytes}"
        )
    validate_registered_reuse_plan(
        actions.reuse_plan,
        artifact_identity=artifact_identity,
        registry=_method_registry(method_registry),
    )


def engine_kv_connector_actions_to_record(
    actions: EngineKVConnectorActions,
    *,
    method_registry: MethodRegistry | None = None,
) -> dict[str, Any]:
    """Serialize connector action descriptors for an out-of-process native adapter."""

    validate_engine_kv_connector_actions(
        actions,
        method_registry=method_registry,
    )
    assert actions.reuse_plan is not None
    return {
        "record_type": ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE,
        "schema_version": ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION,
        "backend": actions.reservation.backend.value,
        "request_id": actions.reservation.request_id,
        "reservation": _reservation_action_to_record(actions.reservation),
        "copies": [_copy_action_to_record(copy_action) for copy_action in actions.copies],
        "bind": _bind_action_to_record(actions.bind),
        "release": _release_action_to_record(actions.release),
        "reuse_plan": actions.reuse_plan.to_record(),
    }


def engine_kv_connector_actions_from_record(
    record: Mapping[str, Any],
    *,
    expected_backend: str | ServingBackend | None = None,
    method_registry: MethodRegistry | None = None,
) -> EngineKVConnectorActions:
    """Parse and validate connector action descriptors from a JSON-compatible record."""

    _reject_unsupported_keys(record, _ENGINE_KV_CONNECTOR_ACTIONS_RECORD_KEYS, label="connector actions record")
    if record.get("record_type") != ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE:
        raise ValueError(f"Unsupported connector actions record_type {record.get('record_type')!r}")
    if record.get("schema_version") != ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported connector actions schema_version {record.get('schema_version')!r}; "
            f"expected {ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION}"
        )
    backend = _backend_from_value(_required_str(record, "backend"), field_name="backend")
    if expected_backend is not None:
        expected = _backend_from_value(expected_backend, field_name="expected_backend")
        if backend != expected:
            raise ValueError(f"Connector actions backend {backend.value!r} != expected {expected.value!r}")
    request_id = _required_str(record, "request_id")
    reservation = _reservation_action_from_record(_required_mapping(record, "reservation"))
    if reservation.backend != backend:
        raise ValueError("Connector actions reservation.backend must match record backend")
    if reservation.request_id != request_id:
        raise ValueError("Connector actions reservation.request_id must match record request_id")
    actions = EngineKVConnectorActions(
        reservation=reservation,
        copies=tuple(
            _copy_action_from_record(copy_action, index=index)
            for index, copy_action in enumerate(_required_mapping_sequence(record, "copies"))
        ),
        bind=_bind_action_from_record(_required_mapping(record, "bind")),
        release=_release_action_from_record(_required_mapping(record, "release")),
        reuse_plan=ReusePlan.from_record(
            _required_mapping(record, "reuse_plan")
        ),
    )
    validate_engine_kv_connector_actions(
        actions,
        method_registry=method_registry,
    )
    return actions


def validate_engine_kv_connector_actions_record(
    record: Mapping[str, Any],
    *,
    expected_backend: str | ServingBackend | None = None,
    method_registry: MethodRegistry | None = None,
) -> None:
    """Validate a serialized connector actions record without returning descriptors."""

    engine_kv_connector_actions_from_record(
        record,
        expected_backend=expected_backend,
        method_registry=method_registry,
    )


def probe_engine_kv_connector_actions(
    actions: EngineKVConnectorActions,
    payload_or_segments: bytes | memoryview | tuple[bytes | memoryview, ...],
    probe: EngineKVBlockManagerProbe,
    *,
    engine_version: str = "unknown",
    native_probe: bool = True,
    metadata: Mapping[str, str] | None = None,
    method_registry: MethodRegistry | None = None,
) -> EngineKVConnectorProbeResult:
    """Run reserve/import/bind/release descriptors against a native block-manager probe.

    This is a validation harness for vLLM/SGLang integrations. It deliberately
    stops before decode scheduling so the serving engine remains the scheduler.
    """

    validate_engine_kv_connector_actions(
        actions,
        method_registry=method_registry,
    )
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
    if not isinstance(payload_or_segments, tuple):
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


def _validate_connector_payload_checksum(
    plan: EngineKVInjectionPlan,
    payload_or_segments: bytes | memoryview | tuple[bytes | memoryview, ...],
) -> None:
    if not plan.payload_checksum:
        return
    digest = hashlib.sha256()
    if not isinstance(payload_or_segments, tuple):
        digest.update(_byte_memoryview(payload_or_segments))
    else:
        for payload in payload_or_segments:
            digest.update(_byte_memoryview(payload))
    if digest.hexdigest() != plan.payload_checksum:
        raise ValueError("Connector payload checksum does not match injection plan")


def _is_payload_buffer(value: object) -> TypeGuard[bytes | memoryview]:
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
        token_contract=binding.token_contract,
    )


def _reservation_action_to_record(action: EngineKVReservationAction) -> dict[str, Any]:
    return {
        "backend": action.backend.value,
        "request_id": action.request_id,
        "total_blocks": action.total_blocks,
        "total_tokens": action.total_tokens,
        "estimated_gpu_bytes": action.estimated_gpu_bytes,
        "adapter_ids": list(action.adapter_ids),
        "layout": _layout_to_record(action.layout),
        "artifact_identity": (
            None
            if action.artifact_identity is None
            else action.artifact_identity.to_record()
        ),
        "payload_checksum": action.payload_checksum,
    }


def _reservation_action_from_record(record: Mapping[str, Any]) -> EngineKVReservationAction:
    _reject_unsupported_keys(record, _ENGINE_KV_RESERVATION_ACTION_KEYS, label="connector actions reservation")
    return EngineKVReservationAction(
        backend=_backend_from_value(_required_str(record, "backend"), field_name="reservation.backend"),
        request_id=_required_str(record, "request_id"),
        total_blocks=_required_nonnegative_int(record, "total_blocks"),
        total_tokens=_required_nonnegative_int(record, "total_tokens"),
        estimated_gpu_bytes=_required_nonnegative_int(record, "estimated_gpu_bytes"),
        layout=_layout_from_record(_required_mapping(record, "layout")),
        adapter_ids=_required_str_sequence(record, "adapter_ids"),
        artifact_identity=_optional_artifact_identity(record, "artifact_identity"),
        payload_checksum=_optional_str(record, "payload_checksum") or "",
    )


def _copy_action_to_record(action: EngineKVSegmentCopyAction) -> dict[str, Any]:
    return {
        "request_id": action.request_id,
        "document_id": action.document_id,
        "chunk_type": action.chunk_type,
        "chunk_id": action.chunk_id,
        "payload_index": action.payload_index,
        "source_byte_start": action.source_byte_start,
        "source_byte_length": action.source_byte_length,
        "source_byte_end": action.source_byte_end,
        "global_byte_start": action.global_byte_start,
        "global_byte_end": action.global_byte_end,
        "token_start": action.token_start,
        "token_count": action.token_count,
        "token_end": action.token_end,
        "first_block_index": action.first_block_index,
        "last_block_index_exclusive": action.last_block_index_exclusive,
        "content_hash": action.content_hash,
        "token_contract": (
            None if action.token_contract is None else action.token_contract.to_record()
        ),
        "cache_tier": _cache_tier_from_value(
            action.cache_tier,
            field_name="cache_tier",
        ).value,
    }


def _copy_action_from_record(record: Mapping[str, Any], *, index: int) -> EngineKVSegmentCopyAction:
    _reject_unsupported_keys(record, _ENGINE_KV_COPY_ACTION_KEYS, label=f"connector actions copies[{index}]")
    payload_index = _optional_nonnegative_int(record, "payload_index")
    source_byte_start = _required_nonnegative_int(record, "source_byte_start")
    source_byte_length = _required_positive_int(record, "source_byte_length")
    source_byte_end = _required_nonnegative_int(record, "source_byte_end")
    if source_byte_end != source_byte_start + source_byte_length:
        raise ValueError("copy action source_byte_end must match source_byte_start + source_byte_length")
    return EngineKVSegmentCopyAction(
        request_id=_required_str(record, "request_id"),
        document_id=_required_str(record, "document_id"),
        chunk_type=_required_str(record, "chunk_type"),
        chunk_id=_required_str(record, "chunk_id"),
        payload_index=payload_index,
        source_byte_start=source_byte_start,
        source_byte_length=source_byte_length,
        global_byte_start=_required_nonnegative_int(record, "global_byte_start"),
        global_byte_end=_required_nonnegative_int(record, "global_byte_end"),
        token_start=_required_nonnegative_int(record, "token_start"),
        token_count=_required_positive_int(record, "token_count"),
        token_end=_required_nonnegative_int(record, "token_end"),
        first_block_index=_required_nonnegative_int(record, "first_block_index"),
        last_block_index_exclusive=_required_nonnegative_int(record, "last_block_index_exclusive"),
        content_hash=_optional_str(record, "content_hash") or "",
        cache_tier=_cache_tier_from_value(_required_str(record, "cache_tier"), field_name="copy.cache_tier"),
        token_contract=_optional_token_contract(record, "token_contract"),
    )


def _bind_action_to_record(action: EngineKVBindAction) -> dict[str, Any]:
    return {
        "request_id": action.request_id,
        "handle_uri": action.handle_uri,
        "cache_method": action.cache_method,
        "adapter_ids": list(action.adapter_ids),
        "metadata": dict(action.metadata),
    }


def _bind_action_from_record(record: Mapping[str, Any]) -> EngineKVBindAction:
    _reject_unsupported_keys(record, _ENGINE_KV_BIND_ACTION_KEYS, label="connector actions bind")
    return EngineKVBindAction(
        request_id=_required_str(record, "request_id"),
        handle_uri=_required_str(record, "handle_uri"),
        cache_method=_required_str(record, "cache_method"),
        adapter_ids=_required_str_sequence(record, "adapter_ids"),
        metadata=_required_mapping(record, "metadata"),
    )


def _release_action_to_record(action: EngineKVReleaseAction) -> dict[str, Any]:
    return {"request_id": action.request_id}


def _release_action_from_record(record: Mapping[str, Any]) -> EngineKVReleaseAction:
    _reject_unsupported_keys(record, _ENGINE_KV_RELEASE_ACTION_KEYS, label="connector actions release")
    return EngineKVReleaseAction(request_id=_required_str(record, "request_id"))


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
        "artifact_identity": (
            None
            if handle.artifact_identity is None
            else handle.artifact_identity.to_record()
        ),
        "payload_checksum": handle.payload_checksum,
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
        "checksum": request.ready_request.handle.payload_checksum,
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
    _validate_optional_sha256(
        _optional_str(payload_source, "checksum") or "",
        field_name="payload_source.checksum",
    )


def _validate_handle_record(
    handle: Mapping[str, Any],
    *,
    reuse_plan: ReusePlan,
) -> None:
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
    artifact_identity = _optional_artifact_identity(handle, "artifact_identity")
    payload_checksum = _optional_str(handle, "payload_checksum") or ""
    _validate_optional_sha256(payload_checksum, field_name="handle.payload_checksum")
    if artifact_identity is not None:
        if artifact_identity.method_id != _required_str(handle, "cache_method"):
            raise ValueError("handle.artifact_identity.method_id must match cache_method")
        identity_layout = {
            "model_id": kv_layout.model_id,
            "lora_id": kv_layout.lora_id,
            "layout_version": kv_layout.layout_version,
            "runtime_kv_dtype": kv_layout.dtype,
            "block_size": kv_layout.block_size,
            "payload_axis_order": kv_layout.payload_axis_order.value,
            "key_position_encoding": (
                kv_layout.key_position_encoding.value
            ),
            "rope_theta": kv_layout.rope_theta,
            "rope_rotary_dim": kv_layout.rope_rotary_dim,
        }
        mismatches = [
            name
            for name, value in identity_layout.items()
            if getattr(artifact_identity, name) != value
        ]
        if mismatches:
            raise ValueError(
                "handle.artifact_identity does not match layout: " + ", ".join(mismatches)
            )
    _required_adapter_ids(handle, "adapter_ids", field_name="handle.adapter_ids")
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
        if (
            reuse_plan.artifact_format.encoding == ArtifactEncoding.RAW_KV
            and byte_length != expected_byte_length
        ):
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
        token_contract = _optional_token_contract(segment, "token_contract")
        if token_contract is not None:
            if token_contract.token_count != token_count:
                raise ValueError("segment.token_contract token_count does not match segment")
            if artifact_identity is not None:
                if token_contract.tokenizer_id != artifact_identity.tokenizer_id:
                    raise ValueError("segment.token_contract tokenizer_id does not match artifact")
                if token_contract.tokenizer_revision != artifact_identity.tokenizer_revision:
                    raise ValueError(
                        "segment.token_contract tokenizer_revision does not match artifact"
                    )
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
    *,
    method_registry: MethodRegistry | None,
) -> None:
    backend = _backend_from_value(_required_str(record, "backend"), field_name="backend")
    reuse_plan = _reuse_plan_from_handoff_record(
        record,
        handle=handle,
        allow_legacy=(
            record.get("schema_version")
            == _ENGINE_ADAPTER_HANDOFF_LEGACY_SCHEMA_VERSION
        ),
        method_registry=method_registry,
    )
    payload_mode = _payload_mode_from_value(_required_str(record, "payload_mode"), field_name="payload_mode")
    artifact_identity = _optional_artifact_identity(handle, "artifact_identity")
    expected_values = {
        "document_kv.request_id": _required_str(record, "request_id"),
        "document_kv.handle_uri": _required_str(record, "handle_uri"),
        "document_kv.total_tokens": str(_required_nonnegative_int(handle, "total_tokens")),
        "document_kv.total_bytes": str(_required_nonnegative_int(handle, "total_bytes")),
        "document_kv.cache_method": _required_str(handle, "cache_method"),
        "document_kv.artifact_id": (
            ""
            if artifact_identity is None
            else artifact_identity.artifact_id
        ),
        "document_kv.method_version": (
            "" if artifact_identity is None else artifact_identity.method_version
        ),
        "document_kv.method_config_digest": (
            ""
            if artifact_identity is None
            else artifact_identity.method_config_digest
        ),
        "document_kv.payload_checksum": _optional_str(handle, "payload_checksum") or "",
        "document_kv.payload_mode": payload_mode.value,
        "document_kv.reuse_capability_id": reuse_plan.capability_id,
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
    if record.get("schema_version") == ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION:
        required_identity_keys = (
            "document_kv.artifact_id",
            "document_kv.method_version",
            "document_kv.method_config_digest",
            "document_kv.reuse_capability_id",
        )
        missing_or_mismatched = tuple(
            key
            for key in required_identity_keys
            if metadata.get(key) != expected_values[key]
        )
        if missing_or_mismatched:
            raise ValueError(
                "schema_version 4 metadata must bind the handle artifact and "
                "reuse capability: "
                + ", ".join(missing_or_mismatched)
            )


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
        payload_axis_order=kv_payload_axis_order_from_value(
            layout.get("payload_axis_order", "token_major"),
            field_name="layout.payload_axis_order",
        ),
        pre_rope=bool(layout.get("pre_rope", False)),
        rope_theta=layout.get("rope_theta"),
        rope_rotary_dim=_optional_positive_int(layout, "rope_rotary_dim"),
        key_position_encoding=kv_key_position_encoding_from_value(
            layout.get(
                "key_position_encoding",
                "pre_rope" if layout.get("pre_rope", False) else "stored_post_rope",
            ),
            field_name="layout.key_position_encoding",
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
        token_contract=_optional_token_contract(segment, "token_contract"),
    )


def _block_count(token_count: int, block_size: int) -> int:
    if token_count == 0:
        return 0
    return math.ceil(token_count / block_size)


def _reuse_plan_for_ready_request(
    request: EngineReadyRequest,
    *,
    method_registry: MethodRegistry | None,
) -> ReusePlan:
    registry = _method_registry(method_registry)
    plan = request.reuse_plan
    if plan is None:
        try:
            plan = registry.get(
                request.handle.cache_method,
                require_implemented=True,
            ).reuse_plan()
        except (KeyError, NotImplementedError) as exc:
            raise ValueError(
                f"cache method {request.handle.cache_method!r} requires an explicit "
                "registered reuse plan"
            ) from exc
    if plan.method_id != request.handle.cache_method:
        raise ValueError("reuse_plan.method_id must match handle.cache_method")
    if not plan.requires_artifact:
        raise ValueError("engine-native reuse plans cannot carry a Cachet artifact payload")
    identity = request.handle.artifact_identity
    if identity is not None and (
        identity.artifact_format_id != plan.artifact_format.format_id
        or identity.artifact_format_version != plan.artifact_format.version
    ):
        raise ValueError("reuse_plan artifact format does not match artifact identity")
    if not isinstance(plan, ReusePlan):
        raise TypeError("registered method reuse_plan() must return a ReusePlan")
    return plan


def _reuse_plan_from_handoff_record(
    record: Mapping[str, Any],
    *,
    handle: Mapping[str, Any],
    allow_legacy: bool,
    method_registry: MethodRegistry | None,
) -> ReusePlan:
    registry = _method_registry(method_registry)
    schema_version = record.get("schema_version")
    if schema_version == ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION:
        reuse_plan_record = record.get("reuse_plan")
        if not isinstance(reuse_plan_record, Mapping):
            raise TypeError("reuse_plan must be a mapping")
        plan = ReusePlan.from_record(reuse_plan_record)
        return plan
    if (
        schema_version != _ENGINE_ADAPTER_HANDOFF_LEGACY_SCHEMA_VERSION
        or not allow_legacy
    ):
        raise ValueError("engine handoff does not contain a supported reuse plan")
    method_id = _required_str(handle, "cache_method")
    try:
        plan = registry.get(
            method_id,
            require_implemented=True,
        ).reuse_plan()
    except (KeyError, NotImplementedError) as exc:
        raise ValueError(
            f"legacy handoff method {method_id!r} has no registered default reuse plan"
        ) from exc
    if (
        plan.artifact_format.encoding != ArtifactEncoding.RAW_KV
        or plan.payload_decode_stage != PayloadDecodeStage.NONE
        or plan.token_recompute_policy != TokenRecomputePolicy.NONE
    ):
        raise ValueError(
            "legacy handoffs are supported only for registered raw-KV methods "
            "without decode or recomputation operations"
        )
    validate_registered_reuse_plan(
        plan,
        artifact_identity=_optional_artifact_identity(
            handle,
            "artifact_identity",
        ),
        registry=registry,
    )
    return plan


def _adapter_spec_for_backend(backend: ServingBackend) -> EngineAdapterSpec:
    if backend == ServingBackend.VLLM:
        return vllm_adapter_spec()
    return sglang_adapter_spec()


def _operation_handler_registry(
    registry: RuntimeOperationHandlerRegistry | None,
) -> RuntimeOperationHandlerRegistry:
    if registry is None:
        return RuntimeOperationHandlerRegistry()
    if not isinstance(registry, RuntimeOperationHandlerRegistry):
        raise TypeError(
            "operation_handlers must be a RuntimeOperationHandlerRegistry or None"
        )
    return registry


def _method_registry(registry: MethodRegistry | None) -> MethodRegistry:
    if registry is None:
        return default_method_registry()
    if not isinstance(registry, MethodRegistry):
        raise TypeError("method_registry must be a MethodRegistry or None")
    return registry


def _validate_reuse_plan_artifact_identity(
    reuse_plan: ReusePlan,
    artifact_identity: ArtifactIdentity | None,
) -> None:
    if artifact_identity is None:
        return
    if reuse_plan.method_id != artifact_identity.method_id:
        raise ValueError("reuse_plan.method_id does not match artifact identity")
    artifact_format = reuse_plan.artifact_format
    if (
        artifact_identity.artifact_format_id != artifact_format.format_id
        or artifact_identity.artifact_format_version != artifact_format.version
    ):
        raise ValueError(
            "reuse_plan artifact format/version does not match artifact identity"
        )
    if (
        artifact_format.encoding == ArtifactEncoding.RAW_KV
        and artifact_identity.kv_dtype != artifact_identity.runtime_kv_dtype
    ):
        raise ValueError(
            "raw-KV artifact dtype must match runtime_kv_dtype"
        )


def _nonempty_unique_string_tuple(
    values: Iterable[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, str | bytes | bytearray):
        raise TypeError(f"{field_name} must be an iterable of strings")
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one value")
    for value in normalized:
        _validate_nonempty_str_value(value, field_name=field_name)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


def _nonempty_unique_enum_tuple(
    values: Iterable[_EnumT | str],
    enum_type: type[_EnumT],
    *,
    field_name: str,
) -> tuple[_EnumT, ...]:
    if isinstance(values, str | bytes | bytearray):
        raise TypeError(f"{field_name} must be an iterable of enum values")
    normalized = tuple(enum_type(value) for value in values)
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one value")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


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


def _normalize_connector_adapter_ids(
    adapter_ids: list[str] | tuple[str, ...],
    *,
    field_name: str = "adapter_ids",
) -> tuple[str, ...]:
    if isinstance(adapter_ids, str) or not isinstance(adapter_ids, list | tuple):
        raise TypeError(f"{field_name} must be a sequence of non-empty strings")
    normalized = tuple(adapter_ids)
    if any(not isinstance(adapter_id, str) or not adapter_id for adapter_id in normalized):
        raise ValueError(f"{field_name} entries must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} entries must be unique")
    return normalized


def _required_mapping(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = record.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _optional_artifact_identity(
    record: Mapping[str, Any],
    key: str,
) -> ArtifactIdentity | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be null or a mapping")
    return ArtifactIdentity.from_record(value)


def _optional_token_contract(
    record: Mapping[str, Any],
    key: str,
) -> TokenContract | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be null or a mapping")
    return TokenContract.from_record(value)


def _reject_unsupported_keys(record: Mapping[str, Any], allowed_keys: frozenset[str], *, label: str) -> None:
    unsupported = sorted(str(key) for key in record if key not in allowed_keys)
    if unsupported:
        raise ValueError(f"{label} has unsupported keys: {unsupported}")


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


def _required_adapter_ids(
    record: Mapping[str, Any],
    key: str,
    *,
    field_name: str = "adapter_ids",
) -> tuple[str, ...]:
    return _normalize_connector_adapter_ids(_required_str_sequence(record, key), field_name=field_name)


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


def _validate_optional_sha256(value: str, *, field_name: str) -> None:
    if not value:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _required_nonnegative_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    return _validate_nonnegative_int_value(value, field_name=key)


def _optional_nonnegative_int(record: Mapping[str, Any], key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    return _validate_nonnegative_int_value(value, field_name=key, allow_null=True)


def _validate_nonnegative_int_value(
    value: Any,
    *,
    field_name: str,
    allow_null: bool = False,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        qualifier = "null or a " if allow_null else "a "
        raise ValueError(f"{field_name} must be {qualifier}non-negative integer")
    return value


def _validate_nonempty_str_value(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _validate_plan_totals(
    *,
    total_tokens: int,
    total_bytes: int,
    total_blocks: int,
    layout: KVLayout,
    reuse_plan: ReusePlan,
) -> None:
    expected_total_bytes = total_tokens * layout.bytes_per_token
    if (
        reuse_plan.artifact_format.encoding == ArtifactEncoding.RAW_KV
        and total_bytes != expected_total_bytes
    ):
        raise ValueError("total_bytes does not match total_tokens * layout.bytes_per_token")
    if (
        reuse_plan.artifact_format.encoding != ArtifactEncoding.RAW_KV
        and total_bytes <= 0
    ):
        raise ValueError("encoded artifact total_bytes must be positive")
    if total_blocks != _block_count(total_tokens, layout.block_size):
        raise ValueError("total_blocks does not match total_tokens and layout.block_size")


def _validate_injection_plan_segments(
    segments: tuple[EngineKVSegmentBinding, ...],
    *,
    total_tokens: int,
    total_bytes: int,
    layout: KVLayout,
    reuse_plan: ReusePlan,
) -> None:
    token_cursor = 0
    byte_cursor = 0
    for segment in segments:
        for field_name in (
            "token_start",
            "token_count",
            "token_end",
            "byte_start",
            "byte_length",
            "byte_end",
            "first_block_index",
            "last_block_index_exclusive",
        ):
            _validate_nonnegative_int_value(
                getattr(segment, field_name),
                field_name=f"segment.{field_name}",
            )
        if segment.token_count <= 0:
            raise ValueError(f"Segment binding {segment.chunk_id!r} token_count must be positive")
        if segment.byte_length <= 0:
            raise ValueError(f"Segment binding {segment.chunk_id!r} byte_length must be positive")
        if segment.token_start != token_cursor:
            raise ValueError(f"Non-contiguous token segment binding {segment.chunk_id!r}")
        if segment.byte_start != byte_cursor:
            raise ValueError(f"Non-contiguous byte segment binding {segment.chunk_id!r}")
        if segment.token_start + segment.token_count != segment.token_end:
            raise ValueError(f"Segment binding {segment.chunk_id!r} token_end does not match token range")
        if segment.byte_start + segment.byte_length != segment.byte_end:
            raise ValueError(f"Segment binding {segment.chunk_id!r} byte_end does not match byte range")
        expected_byte_length = segment.token_count * layout.bytes_per_token
        if (
            reuse_plan.artifact_format.encoding == ArtifactEncoding.RAW_KV
            and segment.byte_length != expected_byte_length
        ):
            raise ValueError(
                f"Segment binding {segment.chunk_id!r} byte_length {segment.byte_length} "
                f"!= token_count * bytes_per_token {expected_byte_length}"
            )
        expected_first_block = segment.token_start // layout.block_size
        expected_last_block = _block_count(segment.token_end, layout.block_size)
        if segment.first_block_index != expected_first_block:
            raise ValueError(
                f"Segment binding {segment.chunk_id!r} first_block_index "
                f"{segment.first_block_index} != token_start // block_size {expected_first_block}"
            )
        if segment.last_block_index_exclusive != expected_last_block:
            raise ValueError(
                f"Segment binding {segment.chunk_id!r} last_block_index_exclusive "
                f"{segment.last_block_index_exclusive} != ceil(token_end / block_size) {expected_last_block}"
            )
        token_cursor = segment.token_end
        byte_cursor = segment.byte_end
    if token_cursor != total_tokens:
        raise ValueError(f"Segment bindings cover {token_cursor} tokens != total_tokens {total_tokens}")
    if byte_cursor != total_bytes:
        raise ValueError(f"Segment bindings cover {byte_cursor} bytes != total_bytes {total_bytes}")


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
        "payload_axis_order": layout.payload_axis_order.value,
        "pre_rope": layout.pre_rope,
        "rope_theta": layout.rope_theta,
        "rope_rotary_dim": layout.rope_rotary_dim,
        "key_position_encoding": layout.key_position_encoding.value,
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
        "token_contract": (
            None if segment.token_contract is None else segment.token_contract.to_record()
        ),
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
