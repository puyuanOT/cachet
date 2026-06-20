"""Model-specific KV-cache layout profiles."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from document_kv_cache.engine_protocol import (
    DTYPE_BYTE_WIDTHS,
    AttentionMechanism,
    KVLayout,
    KVStorageLayout,
    dtype_byte_width,
    kv_storage_layout_from_value,
)
from document_kv_cache.storage import local_path

__all__ = [
    "DTYPE_BYTE_WIDTHS",
    "AttentionMechanism",
    "KVStorageLayout",
    "KVLayout",
    "dtype_byte_width",
    "kv_storage_layout_from_value",
    "KVModelProfile",
    "MODEL_PROFILE_RECORD_TYPE",
    "ModelProfileDefinition",
    "ModelProfileRegistry",
    "QWEN3_4B_BASE_HF_MODEL_ID",
    "QWEN3_4B_INSTRUCT_HF_MODEL_ID",
    "QWEN3_4B_INSTRUCT_PROFILE",
    "builtin_model_profiles",
    "default_model_profile_registry",
    "get_model_profile",
    "layout_for_model",
    "model_profile_definition_from_record",
    "model_profile_definition_to_record",
    "read_model_profile_definition_json",
    "write_model_profile_definition_json",
]


MODEL_PROFILE_RECORD_TYPE = "document_kv.model_profile.v1"
_MODEL_PROFILE_RECORD_KEYS = frozenset(
    {
        "record_type",
        "model_id",
        "architecture",
        "num_layers",
        "num_query_heads",
        "num_kv_heads",
        "head_size",
        "max_context_tokens",
        "default_layout_version",
        "default_dtype",
        "default_block_size",
        "default_lora_id",
        "metadata",
        "aliases",
    }
)


@dataclass(frozen=True, slots=True)
class KVModelProfile:
    """Static attention geometry needed to build compatible KV layouts."""

    model_id: str
    architecture: str
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    max_context_tokens: int
    default_layout_version: str
    default_dtype: str = "int8"
    default_block_size: int = 16
    default_lora_id: str = "base"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()
        object.__setattr__(self, "metadata", MappingProxyType(dict(_validated_metadata(self.metadata))))

    @property
    def attention_mechanism(self) -> AttentionMechanism:
        if self.num_kv_heads == self.num_query_heads:
            return AttentionMechanism.MULTI_HEAD
        if self.num_kv_heads == 1:
            return AttentionMechanism.MULTI_QUERY
        return AttentionMechanism.GROUPED_QUERY

    @property
    def query_heads_per_kv_head(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def kv_scalars_per_token(self) -> int:
        return self.num_layers * self.num_kv_heads * self.head_size * 2

    def bytes_per_token(self, dtype: str | None = None) -> int:
        layout_dtype = self.default_dtype if dtype is None else dtype
        return self.kv_scalars_per_token * dtype_byte_width(layout_dtype)

    def to_layout(
        self,
        *,
        model_id: str | None = None,
        dtype: str | None = None,
        lora_id: str | None = None,
        block_size: int | None = None,
        layout_version: str | None = None,
        kv_stride_bytes: int | None = None,
        shares_kv_storage: bool = True,
        storage_layout: KVStorageLayout | str | None = None,
    ) -> KVLayout:
        layout_dtype = self.default_dtype if dtype is None else dtype
        byte_width = dtype_byte_width(layout_dtype)
        if storage_layout is None:
            resolved_storage_layout = (
                KVStorageLayout.SHARED_KEY_VALUE if shares_kv_storage else KVStorageLayout.SEPARATE_KEY_VALUE
            )
        else:
            resolved_storage_layout = storage_layout
        layout = KVLayout(
            model_id=self.model_id if model_id is None else model_id,
            lora_id=self.default_lora_id if lora_id is None else lora_id,
            layout_version=self.default_layout_version if layout_version is None else layout_version,
            dtype=layout_dtype,
            num_layers=self.num_layers,
            block_size=self.default_block_size if block_size is None else block_size,
            bytes_per_token=self.bytes_per_token(layout_dtype),
            num_query_heads=self.num_query_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            kv_stride_bytes=self.head_size * byte_width if kv_stride_bytes is None else kv_stride_bytes,
            shares_kv_storage=shares_kv_storage,
            storage_layout=resolved_storage_layout,
        )
        layout.validate()
        return layout

    def validate(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not isinstance(self.architecture, str) or not self.architecture:
            raise ValueError("architecture must be non-empty")
        if not isinstance(self.default_layout_version, str) or not self.default_layout_version:
            raise ValueError("default_layout_version must be non-empty")
        _validate_positive_int(self.num_layers, "num_layers")
        _validate_positive_int(self.num_query_heads, "num_query_heads")
        _validate_positive_int(self.num_kv_heads, "num_kv_heads")
        _validate_positive_int(self.head_size, "head_size")
        _validate_positive_int(self.max_context_tokens, "max_context_tokens")
        _validate_positive_int(self.default_block_size, "default_block_size")
        if self.num_kv_heads > self.num_query_heads:
            raise ValueError("num_kv_heads cannot exceed num_query_heads")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        if not isinstance(self.default_lora_id, str) or not self.default_lora_id:
            raise ValueError("default_lora_id must be non-empty")
        if not isinstance(self.default_dtype, str) or not self.default_dtype:
            raise ValueError("default_dtype must be non-empty")
        dtype_byte_width(self.default_dtype)


@dataclass(frozen=True, slots=True)
class ModelProfileDefinition:
    """Portable model profile plus aliases for user-provided registries."""

    profile: KVModelProfile
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.profile, KVModelProfile):
            raise TypeError("profile must be a KVModelProfile")
        object.__setattr__(self, "aliases", _validated_aliases(self.aliases))


@dataclass(frozen=True, slots=True)
class ModelProfileRegistry:
    """Immutable registry for built-in or application-provided model profiles."""

    profiles: Mapping[str, KVModelProfile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_profiles = dict(self.profiles)
        for alias, profile in normalized_profiles.items():
            if not isinstance(alias, str):
                raise TypeError("model profile aliases must be strings")
            if not alias:
                raise ValueError("model profile aliases must be non-empty")
            if not isinstance(profile, KVModelProfile):
                raise TypeError(f"Model profile {alias!r} must be a KVModelProfile")
        object.__setattr__(self, "profiles", MappingProxyType(normalized_profiles))

    def __contains__(self, model_id: str) -> bool:
        return model_id in self.profiles

    def __len__(self) -> int:
        return len(self.profiles)

    def get(self, model_id: str) -> KVModelProfile:
        if not isinstance(model_id, str):
            raise TypeError("model_id must be a string")
        if not model_id:
            raise ValueError("model_id must be non-empty")
        try:
            return self.profiles[model_id]
        except KeyError as exc:
            supported = ", ".join(sorted(self.profiles))
            raise KeyError(f"Unknown model profile {model_id!r}; supported profiles: {supported}") from exc

    def with_profile(self, profile: KVModelProfile, *, aliases: Iterable[str] = ()) -> ModelProfileRegistry:
        if not isinstance(profile, KVModelProfile):
            raise TypeError("profile must be a KVModelProfile")
        aliases = _validated_aliases(aliases)
        entries = dict(self.profiles)
        for model_id in (profile.model_id, *aliases):
            existing_profile = entries.get(model_id)
            if existing_profile is not None and existing_profile != profile:
                raise ValueError(f"model profile alias {model_id!r} is already registered")
            entries[model_id] = profile
        return ModelProfileRegistry(entries)

    def with_definition(self, definition: ModelProfileDefinition) -> ModelProfileRegistry:
        if not isinstance(definition, ModelProfileDefinition):
            raise TypeError("definition must be a ModelProfileDefinition")
        return self.with_profile(definition.profile, aliases=definition.aliases)

    def layout_for_model(
        self,
        model_id: str,
        *,
        dtype: str | None = None,
        lora_id: str | None = None,
        block_size: int | None = None,
        layout_version: str | None = None,
        kv_stride_bytes: int | None = None,
        shares_kv_storage: bool = True,
        storage_layout: KVStorageLayout | str | None = None,
    ) -> KVLayout:
        return self.get(model_id).to_layout(
            model_id=model_id,
            dtype=dtype,
            lora_id=lora_id,
            block_size=block_size,
            layout_version=layout_version,
            kv_stride_bytes=kv_stride_bytes,
            shares_kv_storage=shares_kv_storage,
            storage_layout=storage_layout,
        )


def model_profile_definition_to_record(definition: ModelProfileDefinition) -> dict[str, Any]:
    if not isinstance(definition, ModelProfileDefinition):
        raise TypeError("definition must be a ModelProfileDefinition")
    profile = definition.profile
    return {
        "record_type": MODEL_PROFILE_RECORD_TYPE,
        "model_id": profile.model_id,
        "architecture": profile.architecture,
        "num_layers": profile.num_layers,
        "num_query_heads": profile.num_query_heads,
        "num_kv_heads": profile.num_kv_heads,
        "head_size": profile.head_size,
        "max_context_tokens": profile.max_context_tokens,
        "default_layout_version": profile.default_layout_version,
        "default_dtype": profile.default_dtype,
        "default_block_size": profile.default_block_size,
        "default_lora_id": profile.default_lora_id,
        "metadata": dict(sorted(profile.metadata.items())),
        "aliases": list(definition.aliases),
    }


def model_profile_definition_from_record(record: Mapping[str, Any]) -> ModelProfileDefinition:
    if not isinstance(record, Mapping):
        raise ValueError("model profile record must be an object")
    _reject_unsupported_keys(record, _MODEL_PROFILE_RECORD_KEYS, label="model profile record")
    if record.get("record_type") != MODEL_PROFILE_RECORD_TYPE:
        raise ValueError(f"record_type must be {MODEL_PROFILE_RECORD_TYPE!r}")
    metadata = _validated_metadata(record.get("metadata", {}))
    aliases = _validated_aliases(_required_sequence(record, "aliases"))
    profile = KVModelProfile(
        model_id=_required_str(record, "model_id"),
        architecture=_required_str(record, "architecture"),
        num_layers=_required_int(record, "num_layers"),
        num_query_heads=_required_int(record, "num_query_heads"),
        num_kv_heads=_required_int(record, "num_kv_heads"),
        head_size=_required_int(record, "head_size"),
        max_context_tokens=_required_int(record, "max_context_tokens"),
        default_layout_version=_required_str(record, "default_layout_version"),
        default_dtype=_required_str(record, "default_dtype"),
        default_block_size=_required_int(record, "default_block_size"),
        default_lora_id=_required_str(record, "default_lora_id"),
        metadata=metadata,
    )
    return ModelProfileDefinition(profile=profile, aliases=aliases)


def write_model_profile_definition_json(definition: ModelProfileDefinition, path: str | Path) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(model_profile_definition_to_record(definition), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_model_profile_definition_json(path: str | Path) -> ModelProfileDefinition:
    record = json.loads(local_path(str(path)).read_text(encoding="utf-8"))
    if not isinstance(record, Mapping):
        raise ValueError("model profile JSON root must be an object")
    return model_profile_definition_from_record(record)


def _validated_aliases(aliases: Iterable[str]) -> tuple[str, ...]:
    if isinstance(aliases, (str, bytes, bytearray, memoryview)) or not isinstance(aliases, Iterable):
        raise TypeError("aliases must be an iterable of strings")
    normalized = []
    seen = set()
    for alias in tuple(aliases):
        if not isinstance(alias, str):
            raise TypeError("model profile aliases must be strings")
        if not alias:
            raise ValueError("model profile aliases must be non-empty")
        if alias in seen:
            raise ValueError("model profile aliases must be unique")
        seen.add(alias)
        normalized.append(alias)
    return tuple(normalized)


def _validated_metadata(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")
    metadata = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if not isinstance(item, str):
            raise ValueError("metadata values must be strings")
        metadata[key] = item
    return metadata


def _reject_unsupported_keys(record: Mapping[str, Any], allowed_keys: frozenset[str], *, label: str) -> None:
    unsupported = sorted(str(key) for key in record if key not in allowed_keys)
    if unsupported:
        raise ValueError(f"{label} has unsupported keys: {unsupported}")


def _required_sequence(record: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = record.get(key)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (tuple, list)):
        raise ValueError(f"{key} must be a sequence")
    return tuple(value)


def _required_str(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _validate_positive_int(value: Any, name: str) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


QWEN3_4B_BASE_HF_MODEL_ID = "Qwen/Qwen3-4B"
QWEN3_4B_INSTRUCT_HF_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


QWEN3_4B_INSTRUCT_PROFILE = KVModelProfile(
    model_id="qwen3:4b-instruct",
    architecture="Qwen3ForCausalLM",
    num_layers=36,
    num_query_heads=32,
    num_kv_heads=8,
    head_size=128,
    max_context_tokens=262144,
    default_layout_version="qwen3-v1",
    default_dtype="int8",
    metadata={
        "attention": "gqa",
        "hf_model_id": QWEN3_4B_INSTRUCT_HF_MODEL_ID,
    },
)

_BUILTIN_MODEL_PROFILE_REGISTRY = ModelProfileRegistry().with_profile(
    QWEN3_4B_INSTRUCT_PROFILE,
    aliases=(QWEN3_4B_INSTRUCT_HF_MODEL_ID, QWEN3_4B_BASE_HF_MODEL_ID),
)


def builtin_model_profiles() -> Mapping[str, KVModelProfile]:
    return _BUILTIN_MODEL_PROFILE_REGISTRY.profiles


def default_model_profile_registry() -> ModelProfileRegistry:
    return _BUILTIN_MODEL_PROFILE_REGISTRY


def get_model_profile(model_id: str) -> KVModelProfile:
    return _BUILTIN_MODEL_PROFILE_REGISTRY.get(model_id)


def layout_for_model(
    model_id: str,
    *,
    dtype: str | None = None,
    lora_id: str | None = None,
    block_size: int | None = None,
    layout_version: str | None = None,
    kv_stride_bytes: int | None = None,
    shares_kv_storage: bool = True,
    storage_layout: KVStorageLayout | str | None = None,
) -> KVLayout:
    return _BUILTIN_MODEL_PROFILE_REGISTRY.layout_for_model(
        model_id,
        dtype=dtype,
        lora_id=lora_id,
        block_size=block_size,
        layout_version=layout_version,
        kv_stride_bytes=kv_stride_bytes,
        shares_kv_storage=shares_kv_storage,
        storage_layout=storage_layout,
    )
