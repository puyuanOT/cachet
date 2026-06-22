"""Lightweight vLLM layer-name mapping helpers for Cachet probes."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE = "vllm_kv_injection.document_kv_layer_mapping.v1"
DOCUMENT_KV_VLLM_LAYER_MAPPING_SCHEMA_VERSION = 1
_VLLM_LAYER_INDEX_RE = re.compile(r"(?:^|\.)(?:layers?|h|blocks?)\.(\d+)(?:\.|$)")
_DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "runtime",
        "layer_names",
        "layer_indices",
        "unresolved_layer_names",
        "duplicate_layer_indices",
        "ok",
    }
)

__all__ = [
    "DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE",
    "DOCUMENT_KV_VLLM_LAYER_MAPPING_SCHEMA_VERSION",
    "DocumentKVVLLMLayerMappingInspection",
    "document_kv_vllm_layer_index_from_name",
    "document_kv_vllm_layer_mapping_record_issues",
    "document_kv_vllm_layer_mapping_to_record",
    "document_kv_vllm_probe_layer_names",
    "inspect_document_kv_vllm_layer_mapping",
    "validate_document_kv_vllm_layer_mapping_record",
]


@dataclass(frozen=True, slots=True)
class DocumentKVVLLMLayerMappingInspection:
    """Diagnostic for mapping vLLM KV cache layer names to Cachet layer indices."""

    layer_names: tuple[str, ...]
    layer_indices: Mapping[str, int]
    unresolved_layer_names: tuple[str, ...] = ()
    duplicate_layer_indices: Mapping[int, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_names", _string_tuple(self.layer_names, "layer_names"))
        layer_name_set = set(self.layer_names)
        normalized_indices: dict[str, int] = {}
        for layer_name, layer_index in self.layer_indices.items():
            if layer_name not in layer_name_set:
                raise ValueError("layer_indices keys must be present in layer_names")
            if not isinstance(layer_index, int) or isinstance(layer_index, bool) or layer_index < 0:
                raise ValueError("layer_indices values must be non-negative integers")
            normalized_indices[layer_name] = layer_index
        object.__setattr__(self, "layer_indices", MappingProxyType(normalized_indices))
        object.__setattr__(
            self,
            "unresolved_layer_names",
            _string_tuple(self.unresolved_layer_names, "unresolved_layer_names"),
        )
        normalized_duplicates: dict[int, tuple[str, ...]] = {}
        for layer_index, layer_names in self.duplicate_layer_indices.items():
            if not isinstance(layer_index, int) or isinstance(layer_index, bool) or layer_index < 0:
                raise ValueError("duplicate_layer_indices keys must be non-negative integers")
            names = _string_tuple(layer_names, "duplicate_layer_indices values")
            if len(names) < 2:
                raise ValueError("duplicate_layer_indices values must include at least two layer names")
            if any(name not in layer_name_set for name in names):
                raise ValueError("duplicate_layer_indices values must be present in layer_names")
            normalized_duplicates[layer_index] = tuple(sorted(names))
        object.__setattr__(self, "duplicate_layer_indices", MappingProxyType(normalized_duplicates))

    @property
    def ok(self) -> bool:
        return bool(self.layer_names) and not self.unresolved_layer_names and not self.duplicate_layer_indices


def document_kv_vllm_layer_index_from_name(layer_name: object) -> int | None:
    """Return the vLLM decoder-layer index encoded in a KV cache layer name."""

    if not isinstance(layer_name, str):
        return None
    matches = _VLLM_LAYER_INDEX_RE.findall(layer_name)
    if len(matches) != 1:
        return None
    return int(matches[0])


def document_kv_vllm_probe_layer_names(layout: object) -> tuple[str, ...]:
    """Return deterministic vLLM KV cache layer names for native probe fixtures."""

    return tuple(
        f"probe.layer.{layer_index}"
        for layer_index in range(_positive_int(getattr(layout, "num_layers", None), field_name="num_layers"))
    )


def inspect_document_kv_vllm_layer_mapping(
    kv_caches_or_layer_names: Mapping[str, object] | Sequence[str],
) -> DocumentKVVLLMLayerMappingInspection:
    """Inspect whether vLLM KV cache layer names can be mapped safely."""

    layer_names = _layer_names(kv_caches_or_layer_names)
    indices: dict[str, int] = {}
    unresolved_layer_names: list[str] = []
    duplicate_layer_indices: dict[int, list[str]] = {}
    for layer_name in layer_names:
        layer_index = document_kv_vllm_layer_index_from_name(layer_name)
        if layer_index is None:
            unresolved_layer_names.append(layer_name)
            continue
        duplicate_layer_indices.setdefault(layer_index, []).append(layer_name)
        indices[layer_name] = layer_index
    duplicates: dict[int, tuple[str, ...]] = {
        layer_index: tuple(names)
        for layer_index, names in duplicate_layer_indices.items()
        if len(names) > 1
    }
    return DocumentKVVLLMLayerMappingInspection(
        layer_names=layer_names,
        layer_indices=indices,
        unresolved_layer_names=tuple(unresolved_layer_names),
        duplicate_layer_indices=duplicates,
    )


def document_kv_vllm_layer_mapping_to_record(
    inspection: DocumentKVVLLMLayerMappingInspection | Mapping[str, object] | Sequence[str],
) -> dict[str, Any]:
    """Serialize a vLLM layer-name mapping inspection for runtime preflights."""

    if not isinstance(inspection, DocumentKVVLLMLayerMappingInspection):
        inspection = inspect_document_kv_vllm_layer_mapping(inspection)
    return {
        "record_type": DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE,
        "schema_version": DOCUMENT_KV_VLLM_LAYER_MAPPING_SCHEMA_VERSION,
        "runtime": "vllm-kv-connector-v1",
        "layer_names": list(inspection.layer_names),
        "layer_indices": dict(inspection.layer_indices),
        "unresolved_layer_names": list(inspection.unresolved_layer_names),
        "duplicate_layer_indices": {
            str(layer_index): list(names)
            for layer_index, names in inspection.duplicate_layer_indices.items()
        },
        "ok": inspection.ok,
    }


def validate_document_kv_vllm_layer_mapping_record(record: Mapping[str, Any]) -> None:
    """Raise when a serialized vLLM layer-name preflight record is invalid."""

    issues = document_kv_vllm_layer_mapping_record_issues(record)
    if issues:
        raise ValueError("; ".join(issues))


def document_kv_vllm_layer_mapping_record_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return structural issues for a serialized vLLM layer-name preflight."""

    issues: list[str] = []
    unexpected = sorted(str(key) for key in record if key not in _DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_KEYS)
    if unexpected:
        issues.append(f"Document KV vLLM layer mapping record has unsupported keys: {unexpected}")
    if record.get("record_type") != DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE:
        issues.append(f"record_type must be {DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE!r}")
    if record.get("schema_version") != DOCUMENT_KV_VLLM_LAYER_MAPPING_SCHEMA_VERSION:
        issues.append(f"schema_version must be {DOCUMENT_KV_VLLM_LAYER_MAPPING_SCHEMA_VERSION}")
    if record.get("runtime") != "vllm-kv-connector-v1":
        issues.append("runtime must be 'vllm-kv-connector-v1'")
    layer_names = _record_string_list(record.get("layer_names"))
    if layer_names is None:
        issues.append("layer_names must be a string array")
    layer_indices = _record_layer_indices(record.get("layer_indices"))
    if layer_indices is None:
        issues.append("layer_indices must be an object of non-negative integer values")
    unresolved_layer_names = _record_string_list(record.get("unresolved_layer_names"))
    if unresolved_layer_names is None:
        issues.append("unresolved_layer_names must be a string array")
    duplicate_layer_indices = _record_duplicate_layer_indices(record.get("duplicate_layer_indices"))
    if duplicate_layer_indices is None:
        issues.append("duplicate_layer_indices must map layer-index strings to string arrays")
    ok = record.get("ok")
    if type(ok) is not bool:
        issues.append("ok must be boolean")
    if (
        layer_names is not None
        and layer_indices is not None
        and unresolved_layer_names is not None
        and duplicate_layer_indices is not None
        and type(ok) is bool
    ):
        expected = document_kv_vllm_layer_mapping_to_record(layer_names)
        for field_name in ("layer_indices", "unresolved_layer_names", "duplicate_layer_indices", "ok"):
            if record.get(field_name) != expected[field_name]:
                issues.append(f"{field_name} must match layer_names")
        if ok is False:
            issues.append("ok must be true for a safe vLLM layer mapping preflight")
    return tuple(issues)


def _layer_names(kv_caches_or_layer_names: Mapping[str, object] | Sequence[str]) -> tuple[str, ...]:
    if isinstance(kv_caches_or_layer_names, Mapping):
        layer_names = tuple(kv_caches_or_layer_names)
    elif isinstance(kv_caches_or_layer_names, Sequence) and not isinstance(
        kv_caches_or_layer_names,
        (str, bytes, bytearray),
    ):
        layer_names = tuple(kv_caches_or_layer_names)
    else:
        raise TypeError("vLLM layer mapping input must be a mapping or sequence of layer names")
    return _string_tuple(layer_names, "layer_names")


def _record_layer_indices(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, int] = {}
    for layer_name, layer_index in value.items():
        if not isinstance(layer_name, str) or not layer_name:
            return None
        if not isinstance(layer_index, int) or isinstance(layer_index, bool) or layer_index < 0:
            return None
        normalized[layer_name] = layer_index
    return normalized


def _record_duplicate_layer_indices(value: object) -> dict[int, tuple[str, ...]] | None:
    if not isinstance(value, Mapping):
        return None
    normalized: dict[int, tuple[str, ...]] = {}
    for layer_index_text, layer_names in value.items():
        if not isinstance(layer_index_text, str):
            return None
        try:
            layer_index = int(layer_index_text)
        except ValueError:
            return None
        if layer_index < 0:
            return None
        names = _record_string_list(layer_names)
        if names is None or len(names) < 2:
            return None
        normalized[layer_index] = tuple(names)
    return normalized


def _record_string_list(value: object) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    items = list(value)
    if not all(isinstance(item, str) and item for item in items):
        return None
    return items


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of non-empty strings")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings")
        normalized.append(value.strip())
    return tuple(normalized)


def _positive_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value
