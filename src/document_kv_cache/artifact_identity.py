"""Portable identities and compatibility checks for reusable KV artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


ARTIFACT_IDENTITY_RECORD_TYPE = "document_kv.artifact_identity.v2"
TOKEN_CONTRACT_RECORD_TYPE = "document_kv.token_contract.v1"
RUNTIME_IDENTITY_RECORD_TYPE = "document_kv.runtime_identity.v2"
UNRESOLVED_IDENTITY = "unresolved"
SHA256_HEX_LENGTH = 64

__all__ = [
    "ARTIFACT_IDENTITY_RECORD_TYPE",
    "TOKEN_CONTRACT_RECORD_TYPE",
    "RUNTIME_IDENTITY_RECORD_TYPE",
    "UNRESOLVED_IDENTITY",
    "ArtifactIdentity",
    "TokenContract",
    "RuntimeIdentity",
    "CompatibilityIssue",
    "RuntimeCompatibilityHandshake",
    "canonical_json_sha256",
    "method_config_digest",
    "token_ids_digest",
]


@dataclass(frozen=True, slots=True)
class TokenContract:
    """Exact tokenizer contract for one stored KV token segment."""

    tokenizer_id: str
    tokenizer_revision: str
    add_special_tokens: bool
    prompt_template_version: str
    token_count: int
    token_ids_digest: str

    def __post_init__(self) -> None:
        for name in ("tokenizer_id", "tokenizer_revision", "prompt_template_version"):
            _validate_nonempty_string(name, getattr(self, name))
        if type(self.add_special_tokens) is not bool:
            raise ValueError("add_special_tokens must be a boolean")
        _validate_positive_integer("token_count", self.token_count)
        _validate_sha256("token_ids_digest", self.token_ids_digest)

    @classmethod
    def from_token_ids(
        cls,
        token_ids: Iterable[int],
        *,
        tokenizer_id: str,
        tokenizer_revision: str,
        add_special_tokens: bool,
        prompt_template_version: str,
    ) -> "TokenContract":
        normalized = _normalized_token_ids(token_ids)
        return cls(
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            add_special_tokens=add_special_tokens,
            prompt_template_version=prompt_template_version,
            token_count=len(normalized),
            token_ids_digest=token_ids_digest(normalized),
        )

    @property
    def fingerprint(self) -> str:
        return canonical_json_sha256(self.to_record())

    def verifies(self, token_ids: Iterable[int]) -> bool:
        normalized = _normalized_token_ids(token_ids)
        return len(normalized) == self.token_count and token_ids_digest(normalized) == self.token_ids_digest

    def require_match(self, token_ids: Iterable[int], *, label: str = "runtime token ids") -> None:
        if not self.verifies(token_ids):
            raise ValueError(
                f"{label} do not satisfy token contract "
                f"{self.fingerprint} (expected {self.token_count} tokens)"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": TOKEN_CONTRACT_RECORD_TYPE,
            **asdict(self),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TokenContract":
        values = _closed_record(
            record,
            record_type=TOKEN_CONTRACT_RECORD_TYPE,
            fields={
                "tokenizer_id",
                "tokenizer_revision",
                "add_special_tokens",
                "prompt_template_version",
                "token_count",
                "token_ids_digest",
            },
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Immutable identity for method semantics and generated KV bytes."""

    method_id: str
    method_version: str
    method_config_digest: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    lora_id: str
    prompt_template_version: str
    layout_version: str
    kv_dtype: str
    block_size: int
    payload_axis_order: str
    key_position_encoding: str = "stored_post_rope"
    rope_theta: float | None = None
    rope_rotary_dim: int | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    generator_family: str = "transformers"
    generator_version: str = UNRESOLVED_IDENTITY
    artifact_format_id: str = "raw_kv"
    artifact_format_version: str = "1"
    runtime_kv_dtype: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "method_id",
            "method_version",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "lora_id",
            "prompt_template_version",
            "layout_version",
            "kv_dtype",
            "payload_axis_order",
            "key_position_encoding",
            "generator_family",
            "generator_version",
            "artifact_format_id",
            "artifact_format_version",
        ):
            _validate_nonempty_string(name, getattr(self, name))
        runtime_kv_dtype = self.kv_dtype if self.runtime_kv_dtype is None else self.runtime_kv_dtype
        _validate_nonempty_string("runtime_kv_dtype", runtime_kv_dtype)
        object.__setattr__(self, "runtime_kv_dtype", runtime_kv_dtype)
        _validate_sha256("method_config_digest", self.method_config_digest)
        for name in ("block_size", "tensor_parallel_size", "pipeline_parallel_size"):
            _validate_positive_integer(name, getattr(self, name))
        _validate_rope_identity(
            key_position_encoding=self.key_position_encoding,
            rope_theta=self.rope_theta,
            rope_rotary_dim=self.rope_rotary_dim,
        )

    @property
    def artifact_id(self) -> str:
        return canonical_json_sha256(self.to_record())

    @property
    def has_unresolved_fields(self) -> bool:
        return any(
            value == UNRESOLVED_IDENTITY
            for value in (
                self.model_revision,
                self.tokenizer_revision,
                self.generator_version,
                self.artifact_format_id,
                self.artifact_format_version,
                self.runtime_kv_dtype,
            )
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": ARTIFACT_IDENTITY_RECORD_TYPE,
            **asdict(self),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ArtifactIdentity":
        values = _closed_record(
            record,
            record_type=ARTIFACT_IDENTITY_RECORD_TYPE,
            fields={
                "method_id",
                "method_version",
                "method_config_digest",
                "model_id",
                "model_revision",
                "tokenizer_id",
                "tokenizer_revision",
                "lora_id",
                "prompt_template_version",
                "layout_version",
                "kv_dtype",
                "block_size",
                "payload_axis_order",
                "key_position_encoding",
                "rope_theta",
                "rope_rotary_dim",
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "generator_family",
                "generator_version",
                "artifact_format_id",
                "artifact_format_version",
                "runtime_kv_dtype",
            },
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Identity reported by a serving runtime before KV injection."""

    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    lora_id: str
    prompt_template_version: str
    layout_version: str
    kv_dtype: str
    block_size: int
    payload_axis_order: str
    key_position_encoding: str = "stored_post_rope"
    rope_theta: float | None = None
    rope_rotary_dim: int | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "lora_id",
            "prompt_template_version",
            "layout_version",
            "kv_dtype",
            "payload_axis_order",
            "key_position_encoding",
        ):
            _validate_nonempty_string(name, getattr(self, name))
        for name in ("block_size", "tensor_parallel_size", "pipeline_parallel_size"):
            _validate_positive_integer(name, getattr(self, name))
        _validate_rope_identity(
            key_position_encoding=self.key_position_encoding,
            rope_theta=self.rope_theta,
            rope_rotary_dim=self.rope_rotary_dim,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": RUNTIME_IDENTITY_RECORD_TYPE,
            **asdict(self),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "RuntimeIdentity":
        values = _closed_record(
            record,
            record_type=RUNTIME_IDENTITY_RECORD_TYPE,
            fields={
                "model_id",
                "model_revision",
                "tokenizer_id",
                "tokenizer_revision",
                "lora_id",
                "prompt_template_version",
                "layout_version",
                "kv_dtype",
                "block_size",
                "payload_axis_order",
                "key_position_encoding",
                "rope_theta",
                "rope_rotary_dim",
                "tensor_parallel_size",
                "pipeline_parallel_size",
            },
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    field: str
    artifact_value: str | int | float | None
    runtime_value: str | int | float | None
    reason: str


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityHandshake:
    """Result of comparing a KV artifact with the serving runtime."""

    artifact_id: str
    compatible: bool
    issues: tuple[CompatibilityIssue, ...]

    @classmethod
    def compare(
        cls,
        artifact: ArtifactIdentity,
        runtime: RuntimeIdentity,
        *,
        reject_unresolved: bool = True,
    ) -> "RuntimeCompatibilityHandshake":
        if not isinstance(artifact, ArtifactIdentity):
            raise TypeError("artifact must be an ArtifactIdentity")
        if not isinstance(runtime, RuntimeIdentity):
            raise TypeError("runtime must be a RuntimeIdentity")
        issues: list[CompatibilityIssue] = []
        fields = (
            ("model_id", "model_id"),
            ("model_revision", "model_revision"),
            ("tokenizer_id", "tokenizer_id"),
            ("tokenizer_revision", "tokenizer_revision"),
            ("lora_id", "lora_id"),
            ("prompt_template_version", "prompt_template_version"),
            ("layout_version", "layout_version"),
            ("runtime_kv_dtype", "kv_dtype"),
            ("block_size", "block_size"),
            ("payload_axis_order", "payload_axis_order"),
            ("key_position_encoding", "key_position_encoding"),
            ("rope_theta", "rope_theta"),
            ("rope_rotary_dim", "rope_rotary_dim"),
            ("tensor_parallel_size", "tensor_parallel_size"),
            ("pipeline_parallel_size", "pipeline_parallel_size"),
        )
        for artifact_field, runtime_field in fields:
            artifact_value = getattr(artifact, artifact_field)
            runtime_value = getattr(runtime, runtime_field)
            if reject_unresolved and (
                artifact_value == UNRESOLVED_IDENTITY or runtime_value == UNRESOLVED_IDENTITY
            ):
                issues.append(
                    CompatibilityIssue(
                        field=runtime_field,
                        artifact_value=artifact_value,
                        runtime_value=runtime_value,
                        reason="identity is unresolved",
                    )
                )
            elif artifact_value != runtime_value:
                issues.append(
                    CompatibilityIssue(
                        field=runtime_field,
                        artifact_value=artifact_value,
                        runtime_value=runtime_value,
                        reason="values differ",
                    )
                )
        return cls(
            artifact_id=artifact.artifact_id,
            compatible=not issues,
            issues=tuple(issues),
        )

    def require_compatible(self) -> None:
        if self.compatible:
            return
        details = "; ".join(
            f"{issue.field}: artifact={issue.artifact_value!r}, "
            f"runtime={issue.runtime_value!r} ({issue.reason})"
            for issue in self.issues
        )
        raise ValueError(f"KV artifact {self.artifact_id} is incompatible with runtime: {details}")


def canonical_json_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Hash a JSON-compatible value using one canonical representation."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def method_config_digest(config: Mapping[str, Any]) -> str:
    if not isinstance(config, Mapping):
        raise TypeError("method config must be a mapping")
    return canonical_json_sha256(dict(config))


def token_ids_digest(token_ids: Iterable[int]) -> str:
    """Hash token ids without depending on host integer width or endianness."""

    normalized = _normalized_token_ids(token_ids)
    return canonical_json_sha256({"token_ids": list(normalized)})


def _normalized_token_ids(token_ids: Iterable[int]) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes, bytearray, memoryview, Mapping)):
        raise TypeError("token_ids must be an iterable of integers")
    try:
        normalized = tuple(token_ids)
    except TypeError as exc:
        raise TypeError("token_ids must be an iterable of integers") from exc
    if not normalized:
        raise ValueError("token_ids must contain at least one token")
    for token_id in normalized:
        if type(token_id) is not int:
            raise TypeError("token_ids entries must be integers")
        if token_id < 0:
            raise ValueError("token_ids entries must be non-negative")
    return normalized


def _closed_record(
    record: Mapping[str, Any],
    *,
    record_type: str,
    fields: set[str],
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    allowed = fields | {"record_type"}
    unsupported = sorted(str(key) for key in record if key not in allowed)
    if unsupported:
        raise ValueError(f"record has unsupported keys: {unsupported}")
    if record.get("record_type") != record_type:
        raise ValueError(f"record_type must be {record_type!r}")
    missing = sorted(field for field in fields if field not in record)
    if missing:
        raise ValueError(f"record is missing required keys: {missing}")
    return {field: record[field] for field in fields}


def _validate_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _validate_rope_identity(
    *,
    key_position_encoding: str,
    rope_theta: float | None,
    rope_rotary_dim: int | None,
) -> None:
    if key_position_encoding not in {
        "stored_post_rope",
        "pre_rope",
    }:
        raise ValueError(
            "key_position_encoding must be stored_post_rope or pre_rope"
        )
    if (rope_theta is None) != (rope_rotary_dim is None):
        raise ValueError(
            "rope_theta and rope_rotary_dim must be provided together"
        )
    if key_position_encoding == "pre_rope" and (
        rope_theta is None or rope_rotary_dim is None
    ):
        raise ValueError(
            f"{key_position_encoding} identity requires RoPE parameters"
        )
    if rope_theta is not None and (
        not isinstance(rope_theta, (int, float))
        or isinstance(rope_theta, bool)
        or not math.isfinite(float(rope_theta))
        or rope_theta <= 0
    ):
        raise ValueError("rope_theta must be a positive finite number or None")
    if rope_rotary_dim is not None and (
        type(rope_rotary_dim) is not int
        or rope_rotary_dim <= 0
        or rope_rotary_dim % 2
    ):
        raise ValueError(
            "rope_rotary_dim must be a positive even integer or None"
        )


def _validate_positive_integer(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
