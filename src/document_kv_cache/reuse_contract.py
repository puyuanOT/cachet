"""Typed separation between stored artifacts and runtime KV reuse behavior."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from document_kv_cache.engine_protocol import (
    KVKeyPositionEncoding,
    KVLayout,
    KVPayloadAxisOrder,
    KVStorageLayout,
)


ARTIFACT_FORMAT_RECORD_TYPE = "document_kv.artifact_format.v1"
REUSE_PLAN_RECORD_TYPE = "document_kv.reuse_plan.v1"

__all__ = [
    "ARTIFACT_FORMAT_RECORD_TYPE",
    "REUSE_PLAN_RECORD_TYPE",
    "ArtifactEncoding",
    "PositionHandling",
    "PayloadDecodeStage",
    "TokenRecomputePolicy",
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


@dataclass(frozen=True, slots=True)
class ReusePlan:
    """Method-level operations required to turn an artifact into runtime KV."""

    method_id: str
    connector_mode: str
    artifact_format: ArtifactFormat
    position_handling: PositionHandling
    payload_decode_stage: PayloadDecodeStage
    token_recompute_policy: TokenRecomputePolicy

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

    @property
    def requires_artifact(self) -> bool:
        return self.artifact_format.persisted

    @property
    def requires_selective_recompute(self) -> bool:
        return self.token_recompute_policy == TokenRecomputePolicy.SELECTIVE

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
            "method_id": self.method_id,
            "connector_mode": self.connector_mode,
            "artifact_format": self.artifact_format.to_record(),
            "position_handling": self.position_handling.value,
            "payload_decode_stage": self.payload_decode_stage.value,
            "token_recompute_policy": self.token_recompute_policy.value,
        }


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
