"""Shared internal metadata policy for backend-native probe delegates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from document_kv_cache.engine_adapters import ServingBackend

VLLM_NATIVE_PROBE_DELEGATE_FACTORY = "vllm_kv_injection.probe:build_native_connector_probe"
VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY = (
    "vllm_kv_injection.probe:build_document_kv_native_probe_connector"
)
VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA = (
    f"vllm_kv_injection.connector_factory={VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY}"
)
SGLANG_NATIVE_PROBE_DELEGATE_FACTORY = "sglang_kv_injection.probe:build_native_connector_probe"
SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY = (
    "sglang_kv_injection.probe:build_document_kv_hicache_probe_connector"
)
SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA = (
    f"sglang_kv_injection.connector_factory={SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY}"
)
SGLANG_CONNECTOR_FACTORY_METADATA_EXAMPLE = (
    SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA
)

_PLACEHOLDER_CONNECTOR_FACTORY_PATHS = frozenset({"module:factory"})


@dataclass(frozen=True, slots=True)
class NativeDelegateMetadataRequirement:
    backend: ServingBackend
    metadata_key: str
    example_metadata: str


KNOWN_NATIVE_DELEGATE_METADATA_REQUIREMENTS: Mapping[str, NativeDelegateMetadataRequirement] = {
    VLLM_NATIVE_PROBE_DELEGATE_FACTORY: NativeDelegateMetadataRequirement(
        backend=ServingBackend.VLLM,
        metadata_key="vllm_kv_injection.connector_factory",
        example_metadata=VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
    ),
    SGLANG_NATIVE_PROBE_DELEGATE_FACTORY: NativeDelegateMetadataRequirement(
        backend=ServingBackend.SGLANG,
        metadata_key="sglang_kv_injection.connector_factory",
        example_metadata=SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
    ),
}


def validate_known_native_delegate_metadata(
    *,
    backend: ServingBackend,
    native_probe_delegate_factory: str | None,
    metadata: Sequence[str],
    label: str,
    backend_field_label: str,
) -> None:
    if native_probe_delegate_factory is None:
        return
    requirement = KNOWN_NATIVE_DELEGATE_METADATA_REQUIREMENTS.get(native_probe_delegate_factory)
    if requirement is None:
        return
    if backend != requirement.backend:
        raise ValueError(
            f"{label} native_probe_delegate_factory {native_probe_delegate_factory!r} is for "
            f"{requirement.backend.value}, but {backend_field_label} is {backend.value}"
        )
    metadata_value = _metadata_item_map(metadata).get(requirement.metadata_key)
    if not metadata_value:
        raise ValueError(
            f"{label} native_probe_delegate_factory {native_probe_delegate_factory!r} requires "
            f"metadata entry {requirement.example_metadata}"
        )
    _validate_connector_factory_metadata_value(
        metadata_value,
        metadata_key=requirement.metadata_key,
        label=label,
    )


def _metadata_item_map(items: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        key, _separator, value = item.partition("=")
        metadata[key] = value
    return metadata


def _validate_connector_factory_metadata_value(
    value: str,
    *,
    metadata_key: str,
    label: str,
) -> None:
    module_name, separator, attribute_name = value.partition(":")
    if (
        value in _PLACEHOLDER_CONNECTOR_FACTORY_PATHS
        or any(character.isspace() for character in value)
        or not separator
        or not module_name
        or not attribute_name
    ):
        raise ValueError(
            f"{label} metadata entry {metadata_key} must be a real module:attribute connector factory"
        )
