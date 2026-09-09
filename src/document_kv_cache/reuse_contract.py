"""Typed separation between stored artifacts and runtime KV reuse behavior."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from document_kv_cache.engine_protocol import (
    KVKeyPositionEncoding,
    KVLayout,
    KVPayloadAxisOrder,
    KVStorageLayout,
)


ARTIFACT_FORMAT_RECORD_TYPE = "document_kv.artifact_format.v1"
RUNTIME_OPERATION_DESCRIPTOR_RECORD_TYPE = (
    "document_kv.runtime_operation_descriptor.v1"
)
REUSE_PLAN_RECORD_TYPE = "document_kv.reuse_plan.v2"

_RUNTIME_OPERATION_DESCRIPTOR_RECORD_KEYS = frozenset(
    {
        "record_type",
        "strategy_id",
        "version",
        "config_digest",
    }
)

_ARTIFACT_FORMAT_RECORD_KEYS = frozenset(
    {
        "record_type",
        "format_id",
        "version",
        "encoding",
        "bits_per_element",
        "storage_layouts",
        "payload_axis_orders",
    }
)
_REUSE_PLAN_RECORD_KEYS = frozenset(
    {
        "record_type",
        "capability_id",
        "method_id",
        "connector_mode",
        "artifact_format",
        "position_handling",
        "payload_decode_stage",
        "token_recompute_policy",
        "payload_decoder",
        "token_selector",
        "token_recomputer",
    }
)

__all__ = [
    "ARTIFACT_FORMAT_RECORD_TYPE",
    "RUNTIME_OPERATION_DESCRIPTOR_RECORD_TYPE",
    "REUSE_PLAN_RECORD_TYPE",
    "ArtifactEncoding",
    "PositionHandling",
    "PayloadDecodeStage",
    "TokenRecomputePolicy",
    "RuntimeOperationPhase",
    "RuntimeOperationDescriptor",
    "RuntimeOperationRequest",
    "RuntimeOperationResult",
    "RuntimeOperationHandler",
    "RuntimeOperationHandlerBinding",
    "RuntimeOperationHandlerRegistry",
    "runtime_operation_config_digest",
    "apply_runtime_operation_handlers",
    "ArtifactFormat",
    "ReusePlan",
    "RAW_KV_ARTIFACT_FORMAT",
    "PACKED_Q4_ARTIFACT_FORMAT",
    "ENGINE_NATIVE_ARTIFACT_FORMAT",
]


def _nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


class ArtifactEncoding(StrEnum):
    RAW_KV = "raw_kv"
    PACKED_Q4 = "packed_q4"
    ENGINE_NATIVE = "engine_native"


class PositionHandling(StrEnum):
    STORED_POST_ROPE = "stored_post_rope"
    REROPE_AT_INJECTION = "rerope_at_injection"
    ENGINE_NATIVE = "engine_native"


class PayloadDecodeStage(StrEnum):
    NONE = "none"
    PROVIDER = "provider"
    ENGINE_NATIVE = "engine_native"


class TokenRecomputePolicy(StrEnum):
    NONE = "none"
    SELECTIVE = "selective"
    ENGINE_NATIVE = "engine_native"


class RuntimeOperationPhase(StrEnum):
    """Provider execution phases addressable by method-owned handlers."""

    PAYLOAD_DECODE = "payload_decode"
    TOKEN_SELECT = "token_select"
    TOKEN_RECOMPUTE = "token_recompute"


@dataclass(frozen=True, slots=True)
class RuntimeOperationDescriptor:
    """Immutable identity for one method-specific runtime operation."""

    strategy_id: str
    version: str
    config_digest: str

    def __post_init__(self) -> None:
        _nonempty_string(self.strategy_id, "strategy_id")
        _nonempty_string(self.version, "version")
        _sha256_digest(self.config_digest, "config_digest")

    @property
    def strategy_version_id(self) -> str:
        return f"{self.strategy_id}:{self.version}"

    def to_record(self) -> dict[str, str]:
        return {
            "record_type": RUNTIME_OPERATION_DESCRIPTOR_RECORD_TYPE,
            "strategy_id": self.strategy_id,
            "version": self.version,
            "config_digest": self.config_digest,
        }

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
    ) -> "RuntimeOperationDescriptor":
        _validate_record_keys(
            record,
            _RUNTIME_OPERATION_DESCRIPTOR_RECORD_KEYS,
            "runtime operation descriptor",
        )
        if record.get("record_type") != RUNTIME_OPERATION_DESCRIPTOR_RECORD_TYPE:
            raise ValueError(
                "record_type must be "
                f"{RUNTIME_OPERATION_DESCRIPTOR_RECORD_TYPE!r}"
            )
        return cls(
            strategy_id=_required_string(record, "strategy_id"),
            version=_required_string(record, "version"),
            config_digest=_required_string(record, "config_digest"),
        )


def runtime_operation_config_digest(config: Mapping[str, Any]) -> str:
    """Return a stable digest for public, serializable handler configuration."""

    if not isinstance(config, Mapping):
        raise TypeError("runtime operation config must be a mapping")
    try:
        canonical = json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "runtime operation config must be JSON-serializable"
        ) from exc
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeOperationRequest:
    """Immutable invocation passed to an application-injected operation handler.

    ``descriptor.config_digest`` identifies the handler configuration selected by
    the method. The registry verifies it against the binding before invocation,
    and the handler may use the same digest to dispatch its immutable local config.
    """

    phase: RuntimeOperationPhase
    descriptor: RuntimeOperationDescriptor
    reuse_plan: "ReusePlan"
    layout: KVLayout
    total_tokens: int
    payload: bytes
    selected_token_indices: tuple[int, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    runtime_context: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", RuntimeOperationPhase(self.phase))
        if not isinstance(self.descriptor, RuntimeOperationDescriptor):
            raise TypeError("descriptor must be a RuntimeOperationDescriptor")
        if not isinstance(self.reuse_plan, ReusePlan):
            raise TypeError("reuse_plan must be a ReusePlan")
        if not isinstance(self.layout, KVLayout):
            raise TypeError("layout must be a KVLayout")
        self.layout.validate()
        if type(self.total_tokens) is not int or self.total_tokens <= 0:
            raise ValueError("total_tokens must be a positive integer")
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must be bytes")
        indices = _validated_token_indices(
            self.selected_token_indices,
            total_tokens=self.total_tokens,
        )
        object.__setattr__(self, "selected_token_indices", indices)
        object.__setattr__(self, "metadata", _immutable_string_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class RuntimeOperationResult:
    """Typed output from a runtime operation handler."""

    payload: bytes | None = None
    selected_token_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.payload is not None and not isinstance(self.payload, bytes):
            raise TypeError("payload must be bytes or None")
        indices = tuple(self.selected_token_indices)
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError(
                "selected_token_indices must contain non-negative integers"
            )
        if len(set(indices)) != len(indices):
            raise ValueError("selected_token_indices must not contain duplicates")
        object.__setattr__(self, "selected_token_indices", indices)


RuntimeOperationHandler = Callable[[RuntimeOperationRequest], RuntimeOperationResult]
_RuntimeOperationHandlerKey = tuple[RuntimeOperationPhase, str, str]


@dataclass(frozen=True, slots=True)
class RuntimeOperationHandlerBinding:
    """Handler plus the exact authenticated configuration it implements."""

    config_digest: str
    handler: RuntimeOperationHandler

    def __post_init__(self) -> None:
        _sha256_digest(self.config_digest, "config_digest")
        if not callable(self.handler):
            raise TypeError("handler must be callable")


@dataclass(frozen=True, slots=True)
class RuntimeOperationHandlerRegistry:
    """Immutable application-owned handlers keyed by phase/strategy/version.

    Each keyed binding also pins one configuration digest, so resolving the same
    strategy/version with different recorded configuration fails before execution.
    """

    handlers: Mapping[
        _RuntimeOperationHandlerKey,
        RuntimeOperationHandlerBinding,
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.handlers, Mapping):
            raise TypeError("handlers must be a mapping")
        normalized: dict[
            _RuntimeOperationHandlerKey,
            RuntimeOperationHandlerBinding,
        ] = {}
        for raw_key, binding in self.handlers.items():
            key = _runtime_operation_handler_key(raw_key)
            if not isinstance(binding, RuntimeOperationHandlerBinding):
                raise TypeError(
                    "runtime operation handler entries must be "
                    "RuntimeOperationHandlerBinding instances"
                )
            normalized[key] = binding
        object.__setattr__(self, "handlers", MappingProxyType(normalized))

    def with_handler(
        self,
        phase: RuntimeOperationPhase | str,
        descriptor: RuntimeOperationDescriptor,
        handler: RuntimeOperationHandler,
        *,
        replace: bool = False,
    ) -> "RuntimeOperationHandlerRegistry":
        if not isinstance(descriptor, RuntimeOperationDescriptor):
            raise TypeError("descriptor must be a RuntimeOperationDescriptor")
        if not callable(handler):
            raise TypeError("handler must be callable")
        key = (
            RuntimeOperationPhase(phase),
            descriptor.strategy_id,
            descriptor.version,
        )
        entries = dict(self.handlers)
        existing = entries.get(key)
        if existing is not None and not replace:
            if (
                existing.handler is handler
                and existing.config_digest == descriptor.config_digest
            ):
                return self
            raise ValueError(
                f"runtime operation handler {key!r} is already registered"
            )
        entries[key] = RuntimeOperationHandlerBinding(
            config_digest=descriptor.config_digest,
            handler=handler,
        )
        return RuntimeOperationHandlerRegistry(entries)

    def resolve(
        self,
        phase: RuntimeOperationPhase | str,
        descriptor: RuntimeOperationDescriptor,
    ) -> RuntimeOperationHandler:
        if not isinstance(descriptor, RuntimeOperationDescriptor):
            raise TypeError("descriptor must be a RuntimeOperationDescriptor")
        key = (
            RuntimeOperationPhase(phase),
            descriptor.strategy_id,
            descriptor.version,
        )
        try:
            binding = self.handlers[key]
        except KeyError as exc:
            raise KeyError(
                "No runtime operation handler registered for "
                f"{key[0].value}:{key[1]}:{key[2]}"
            ) from exc
        if binding.config_digest != descriptor.config_digest:
            raise ValueError(
                "Runtime operation handler configuration digest does not match "
                f"descriptor for {key[0].value}:{key[1]}:{key[2]}"
            )
        return binding.handler


@dataclass(frozen=True, slots=True)
class ArtifactFormat:
    """Encoding capabilities of persisted bytes, independent of runtime KV."""

    format_id: str
    version: str
    encoding: ArtifactEncoding
    bits_per_element: int | None
    storage_layouts: tuple[KVStorageLayout, ...]
    payload_axis_orders: tuple[KVPayloadAxisOrder, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.format_id, "format_id")
        _nonempty_string(self.version, "version")
        encoding = ArtifactEncoding(self.encoding)
        object.__setattr__(self, "encoding", encoding)
        if self.bits_per_element is not None and (
            type(self.bits_per_element) is not int or self.bits_per_element <= 0
        ):
            raise ValueError("bits_per_element must be positive when provided")
        storage_layouts = tuple(KVStorageLayout(value) for value in self.storage_layouts)
        payload_axis_orders = tuple(KVPayloadAxisOrder(value) for value in self.payload_axis_orders)
        if len(set(storage_layouts)) != len(storage_layouts):
            raise ValueError("storage_layouts must not contain duplicates")
        if len(set(payload_axis_orders)) != len(payload_axis_orders):
            raise ValueError("payload_axis_orders must not contain duplicates")
        if encoding == ArtifactEncoding.ENGINE_NATIVE:
            if self.bits_per_element is not None or storage_layouts or payload_axis_orders:
                raise ValueError("engine-native artifact formats must not describe persisted bytes")
        elif not storage_layouts or not payload_axis_orders:
            raise ValueError("persisted artifact formats require layouts and axis orders")
        if encoding == ArtifactEncoding.PACKED_Q4 and self.bits_per_element != 4:
            raise ValueError("packed-Q4 artifact formats require bits_per_element=4")
        object.__setattr__(self, "storage_layouts", storage_layouts)
        object.__setattr__(self, "payload_axis_orders", payload_axis_orders)

    @property
    def format_version_id(self) -> str:
        return f"{self.format_id}:{self.version}"

    @property
    def persisted(self) -> bool:
        return self.encoding != ArtifactEncoding.ENGINE_NATIVE

    def supports_layout(self, layout: KVLayout) -> bool:
        if not isinstance(layout, KVLayout):
            raise TypeError("layout must be a KVLayout")
        return (
            layout.storage_layout in self.storage_layouts
            and layout.payload_axis_order in self.payload_axis_orders
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": ARTIFACT_FORMAT_RECORD_TYPE,
            "format_id": self.format_id,
            "version": self.version,
            "encoding": self.encoding.value,
            "bits_per_element": self.bits_per_element,
            "storage_layouts": [value.value for value in self.storage_layouts],
            "payload_axis_orders": [value.value for value in self.payload_axis_orders],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ArtifactFormat":
        """Parse a closed artifact-format record.

        Handoff records are evidence, not a best-effort configuration surface.  A
        parser therefore rejects unknown fields instead of silently ignoring a
        capability introduced by a newer producer.
        """

        _validate_record_keys(record, _ARTIFACT_FORMAT_RECORD_KEYS, "artifact format")
        if record.get("record_type") != ARTIFACT_FORMAT_RECORD_TYPE:
            raise ValueError(
                f"record_type must be {ARTIFACT_FORMAT_RECORD_TYPE!r}"
            )
        bits_per_element = record.get("bits_per_element")
        storage_layouts = _string_sequence(record, "storage_layouts")
        payload_axis_orders = _string_sequence(record, "payload_axis_orders")
        return cls(
            format_id=_required_string(record, "format_id"),
            version=_required_string(record, "version"),
            encoding=ArtifactEncoding(_required_string(record, "encoding")),
            bits_per_element=bits_per_element,
            storage_layouts=tuple(KVStorageLayout(value) for value in storage_layouts),
            payload_axis_orders=tuple(
                KVPayloadAxisOrder(value) for value in payload_axis_orders
            ),
        )


@dataclass(frozen=True, slots=True)
class ReusePlan:
    """Method-level operations required to turn an artifact into runtime KV."""

    method_id: str
    connector_mode: str
    artifact_format: ArtifactFormat
    position_handling: PositionHandling
    payload_decode_stage: PayloadDecodeStage
    token_recompute_policy: TokenRecomputePolicy
    payload_decoder: RuntimeOperationDescriptor | None = None
    token_selector: RuntimeOperationDescriptor | None = None
    token_recomputer: RuntimeOperationDescriptor | None = None

    def __post_init__(self) -> None:
        _nonempty_string(self.method_id, "method_id")
        _nonempty_string(self.connector_mode, "connector_mode")
        if not isinstance(self.artifact_format, ArtifactFormat):
            raise TypeError("artifact_format must be an ArtifactFormat")
        position = PositionHandling(self.position_handling)
        decode = PayloadDecodeStage(self.payload_decode_stage)
        recompute = TokenRecomputePolicy(self.token_recompute_policy)
        object.__setattr__(self, "position_handling", position)
        object.__setattr__(self, "payload_decode_stage", decode)
        object.__setattr__(self, "token_recompute_policy", recompute)
        for field_name in (
            "payload_decoder",
            "token_selector",
            "token_recomputer",
        ):
            descriptor = getattr(self, field_name)
            if descriptor is not None and not isinstance(
                descriptor,
                RuntimeOperationDescriptor,
            ):
                raise TypeError(
                    f"{field_name} must be a RuntimeOperationDescriptor or None"
                )
        engine_native_values = (
            position == PositionHandling.ENGINE_NATIVE,
            decode == PayloadDecodeStage.ENGINE_NATIVE,
            recompute == TokenRecomputePolicy.ENGINE_NATIVE,
            not self.artifact_format.persisted,
        )
        if any(engine_native_values) and not all(engine_native_values):
            raise ValueError("engine-native reuse plans must be engine-native in every stage")
        if (
            self.artifact_format.encoding == ArtifactEncoding.PACKED_Q4
            and decode == PayloadDecodeStage.NONE
        ):
            raise ValueError("packed-Q4 artifacts require a payload decode stage")
        if (
            self.artifact_format.encoding == ArtifactEncoding.RAW_KV
            and decode != PayloadDecodeStage.NONE
        ):
            raise ValueError("raw KV artifacts must not declare payload decoding")
        if decode == PayloadDecodeStage.PROVIDER:
            if self.payload_decoder is None:
                raise ValueError(
                    "provider payload decoding requires a payload_decoder descriptor"
                )
        elif self.payload_decoder is not None:
            raise ValueError(
                "payload_decoder requires payload_decode_stage='provider'"
            )
        if recompute == TokenRecomputePolicy.SELECTIVE:
            if self.token_selector is None or self.token_recomputer is None:
                raise ValueError(
                    "selective recomputation requires token_selector and "
                    "token_recomputer descriptors"
                )
        elif self.token_selector is not None or self.token_recomputer is not None:
            raise ValueError(
                "token selector/recomputer descriptors require selective recomputation"
            )

    @property
    def requires_artifact(self) -> bool:
        return self.artifact_format.persisted

    @property
    def requires_selective_recompute(self) -> bool:
        return self.token_recompute_policy == TokenRecomputePolicy.SELECTIVE

    @property
    def runtime_operations(
        self,
    ) -> tuple[tuple[RuntimeOperationPhase, RuntimeOperationDescriptor], ...]:
        operations: list[
            tuple[RuntimeOperationPhase, RuntimeOperationDescriptor]
        ] = []
        if self.payload_decoder is not None:
            operations.append(
                (RuntimeOperationPhase.PAYLOAD_DECODE, self.payload_decoder)
            )
        if self.token_selector is not None:
            operations.append(
                (RuntimeOperationPhase.TOKEN_SELECT, self.token_selector)
            )
        if self.token_recomputer is not None:
            operations.append(
                (RuntimeOperationPhase.TOKEN_RECOMPUTE, self.token_recomputer)
            )
        return tuple(operations)

    @property
    def capability_id(self) -> str:
        """Stable identity for the executable operations required by this plan."""

        canonical = json.dumps(
            self._capability_record(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def validate_runtime_layout(self, layout: KVLayout) -> None:
        if not isinstance(layout, KVLayout):
            raise TypeError("layout must be a KVLayout")
        layout.validate()
        if self.requires_artifact and not self.artifact_format.supports_layout(layout):
            raise ValueError(
                f"artifact format {self.artifact_format.format_version_id!r} "
                "does not support the runtime payload layout"
            )
        if self.position_handling == PositionHandling.REROPE_AT_INJECTION and not layout.pre_rope:
            raise ValueError("re-rope reuse plans require a pre-RoPE payload layout")
        if (
            self.position_handling == PositionHandling.STORED_POST_ROPE
            and layout.key_position_encoding
            != KVKeyPositionEncoding.STORED_POST_ROPE
        ):
            raise ValueError(
                "stored post-RoPE reuse plans cannot reposition keys at injection"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": REUSE_PLAN_RECORD_TYPE,
            "capability_id": self.capability_id,
            **self._capability_record(),
        }

    def _capability_record(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "connector_mode": self.connector_mode,
            "artifact_format": self.artifact_format.to_record(),
            "position_handling": self.position_handling.value,
            "payload_decode_stage": self.payload_decode_stage.value,
            "token_recompute_policy": self.token_recompute_policy.value,
            "payload_decoder": _optional_descriptor_record(
                self.payload_decoder
            ),
            "token_selector": _optional_descriptor_record(
                self.token_selector
            ),
            "token_recomputer": _optional_descriptor_record(
                self.token_recomputer
            ),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ReusePlan":
        """Parse and authenticate an immutable reuse-plan handoff record."""

        _validate_record_keys(record, _REUSE_PLAN_RECORD_KEYS, "reuse plan")
        if record.get("record_type") != REUSE_PLAN_RECORD_TYPE:
            raise ValueError(f"record_type must be {REUSE_PLAN_RECORD_TYPE!r}")
        artifact_record = record.get("artifact_format")
        if not isinstance(artifact_record, Mapping):
            raise TypeError("artifact_format must be a mapping")
        plan = cls(
            method_id=_required_string(record, "method_id"),
            connector_mode=_required_string(record, "connector_mode"),
            artifact_format=ArtifactFormat.from_record(artifact_record),
            position_handling=PositionHandling(
                _required_string(record, "position_handling")
            ),
            payload_decode_stage=PayloadDecodeStage(
                _required_string(record, "payload_decode_stage")
            ),
            token_recompute_policy=TokenRecomputePolicy(
                _required_string(record, "token_recompute_policy")
            ),
            payload_decoder=_optional_descriptor(record, "payload_decoder"),
            token_selector=_optional_descriptor(record, "token_selector"),
            token_recomputer=_optional_descriptor(record, "token_recomputer"),
        )
        capability_id = _required_string(record, "capability_id")
        if capability_id != plan.capability_id:
            raise ValueError("reuse plan capability_id does not match its operations")
        return plan


def apply_runtime_operation_handlers(
    reuse_plan: ReusePlan,
    payload: bytes,
    *,
    layout: KVLayout,
    total_tokens: int,
    handler_registry: RuntimeOperationHandlerRegistry,
    metadata: Mapping[str, str] | None = None,
    runtime_context: object | None = None,
) -> RuntimeOperationResult:
    """Apply declared provider operations in decode/select/recompute order.

    This function supplies the lifecycle and validates handler outputs; the
    method-specific algorithms remain application-owned handler implementations.
    """

    if not isinstance(reuse_plan, ReusePlan):
        raise TypeError("reuse_plan must be a ReusePlan")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not isinstance(handler_registry, RuntimeOperationHandlerRegistry):
        raise TypeError(
            "handler_registry must be a RuntimeOperationHandlerRegistry"
        )
    reuse_plan.validate_runtime_layout(layout)
    if type(total_tokens) is not int or total_tokens <= 0:
        raise ValueError("total_tokens must be a positive integer")
    current_payload = payload
    selected_token_indices: tuple[int, ...] = ()
    for phase, descriptor in reuse_plan.runtime_operations:
        handler = handler_registry.resolve(phase, descriptor)
        request = RuntimeOperationRequest(
            phase=phase,
            descriptor=descriptor,
            reuse_plan=reuse_plan,
            layout=layout,
            total_tokens=total_tokens,
            payload=current_payload,
            selected_token_indices=selected_token_indices,
            metadata={} if metadata is None else metadata,
            runtime_context=runtime_context,
        )
        result = handler(request)
        if not isinstance(result, RuntimeOperationResult):
            raise TypeError(
                "runtime operation handlers must return RuntimeOperationResult"
            )
        if phase == RuntimeOperationPhase.TOKEN_SELECT:
            if result.payload is not None:
                raise ValueError("token selector handlers must not return payload")
            selected_token_indices = _validated_token_indices(
                result.selected_token_indices,
                total_tokens=total_tokens,
            )
            continue
        if result.payload is None:
            raise ValueError(
                f"{phase.value} handlers must return transformed payload bytes"
            )
        if result.selected_token_indices:
            raise ValueError(
                f"{phase.value} handlers must not return selected token indices"
            )
        current_payload = result.payload
    expected_runtime_bytes = total_tokens * layout.bytes_per_token
    if len(current_payload) != expected_runtime_bytes:
        raise ValueError(
            "runtime operation output length does not match "
            "total_tokens * layout.bytes_per_token"
        )
    return RuntimeOperationResult(
        payload=current_payload,
        selected_token_indices=selected_token_indices,
    )


def _validate_record_keys(
    record: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} must be a mapping")
    unexpected = sorted(str(key) for key in record if key not in expected)
    missing = sorted(key for key in expected if key not in record)
    if unexpected:
        raise ValueError(f"{label} has unsupported keys: {unexpected}")
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")


def _required_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    _nonempty_string(value, key)
    assert isinstance(value, str)
    return value


def _string_sequence(record: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = record.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{key} must be an array")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{key} entries must be strings")
    return tuple(value)


def _optional_descriptor_record(
    descriptor: RuntimeOperationDescriptor | None,
) -> dict[str, str] | None:
    return None if descriptor is None else descriptor.to_record()


def _optional_descriptor(
    record: Mapping[str, Any],
    key: str,
) -> RuntimeOperationDescriptor | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping or null")
    return RuntimeOperationDescriptor.from_record(value)


def _sha256_digest(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _runtime_operation_handler_key(
    raw_key: object,
) -> _RuntimeOperationHandlerKey:
    if not isinstance(raw_key, tuple) or len(raw_key) != 3:
        raise TypeError(
            "runtime operation handler keys must be "
            "(phase, strategy_id, version) tuples"
        )
    phase, strategy_id, version = raw_key
    normalized_phase = RuntimeOperationPhase(phase)
    _nonempty_string(strategy_id, "strategy_id")
    _nonempty_string(version, "version")
    assert isinstance(strategy_id, str)
    assert isinstance(version, str)
    return normalized_phase, strategy_id, version


def _validated_token_indices(
    indices: tuple[int, ...],
    *,
    total_tokens: int,
) -> tuple[int, ...]:
    normalized = tuple(indices)
    if any(
        type(index) is not int or index < 0 or index >= total_tokens
        for index in normalized
    ):
        raise ValueError(
            "selected_token_indices must be unique integers within the token range"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("selected_token_indices must not contain duplicates")
    return normalized


def _immutable_string_mapping(
    value: Mapping[str, str],
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        _nonempty_string(key, "metadata key")
        if not isinstance(item, str):
            raise TypeError("metadata values must be strings")
        normalized[key] = item
    return MappingProxyType(normalized)


_ALL_STORAGE_LAYOUTS = tuple(KVStorageLayout)
_ALL_PAYLOAD_AXIS_ORDERS = tuple(KVPayloadAxisOrder)

RAW_KV_ARTIFACT_FORMAT = ArtifactFormat(
    format_id="raw_kv",
    version="1",
    encoding=ArtifactEncoding.RAW_KV,
    bits_per_element=None,
    storage_layouts=_ALL_STORAGE_LAYOUTS,
    payload_axis_orders=_ALL_PAYLOAD_AXIS_ORDERS,
)
PACKED_Q4_ARTIFACT_FORMAT = ArtifactFormat(
    format_id="packed_q4",
    version="1",
    encoding=ArtifactEncoding.PACKED_Q4,
    bits_per_element=4,
    storage_layouts=(
        KVStorageLayout.SEPARATE_KEY_VALUE,
        KVStorageLayout.INTERLEAVED_KEY_VALUE,
    ),
    payload_axis_orders=_ALL_PAYLOAD_AXIS_ORDERS,
)
ENGINE_NATIVE_ARTIFACT_FORMAT = ArtifactFormat(
    format_id="engine_native",
    version="1",
    encoding=ArtifactEncoding.ENGINE_NATIVE,
    bits_per_element=None,
    storage_layouts=(),
    payload_axis_orders=(),
)
