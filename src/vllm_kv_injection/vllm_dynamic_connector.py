"""vLLM V1 dynamic connector entrypoint for document KV providers."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol

DOCUMENT_KV_CONNECTOR_CLASS = "DocumentKVConnector"
DOCUMENT_KV_CONNECTOR_MODULE_PATH = "vllm_kv_injection.vllm_dynamic_connector"
DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY = "document_kv.provider_factory"

__all__ = [
    "DOCUMENT_KV_CONNECTOR_CLASS",
    "DOCUMENT_KV_CONNECTOR_MODULE_PATH",
    "DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY",
    "DocumentKVConnector",
    "DocumentKVProvider",
    "NoOpDocumentKVProvider",
    "VLLMSupportsHMA",
    "load_document_kv_provider_factory",
    "vllm_runtime_import_error",
]

_VLLM_RUNTIME_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - exercised in live vLLM environments.
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore[import-not-found]
        KVConnectorBase_V1 as _KVConnectorBaseV1,
        SupportsHMA as _SupportsHMA,
    )
except Exception as exc:  # pragma: no cover - local lightweight or broken-runtime path.
    _VLLM_RUNTIME_IMPORT_ERROR = exc

    class _SupportsHMA(ABC):
        @abstractmethod
        def request_finished_all_groups(
            self,
            request: object,
            block_ids: tuple[list[int], ...],
        ) -> tuple[bool, Mapping[str, Any] | None]: ...

    class _KVConnectorBaseV1:
        def __init__(
            self,
            vllm_config: object | None,
            role: object | None,
            kv_cache_config: object | None,
        ) -> None:
            self._connector_metadata: object | None = None
            self._vllm_config = vllm_config
            self._kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
            self._kv_cache_config = kv_cache_config
            self._role = role

        @property
        def role(self) -> object | None:
            return self._role

        def bind_connector_metadata(self, connector_metadata: object) -> None:
            self._connector_metadata = connector_metadata

        def clear_connector_metadata(self) -> None:
            self._connector_metadata = None

        def has_connector_metadata(self) -> bool:
            return self._connector_metadata is not None


VLLMSupportsHMA = _SupportsHMA


def vllm_runtime_import_error() -> Exception | None:
    """Return the optional vLLM runtime import error captured at module import.

    The adapter package must stay importable in environments that do not install
    vLLM, or where a target runtime has a transient dependency mismatch. Native
    evidence still validates that a real runtime is usable before a connector is
    accepted.
    """

    return _VLLM_RUNTIME_IMPORT_ERROR


class DocumentKVProvider(Protocol):
    """Provider boundary for version-sensitive vLLM KV-buffer operations."""

    def get_num_new_matched_tokens(
        self,
        request: object,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]: ...

    def update_state_after_alloc(self, request: object, blocks: object, num_external_tokens: int) -> None: ...

    def build_connector_meta(self, scheduler_output: object) -> object: ...

    def register_kv_caches(self, kv_caches: Mapping[str, object]) -> None: ...

    def start_load_kv(self, forward_context: object, **kwargs: object) -> None: ...

    def wait_for_layer_load(self, layer_name: str) -> None: ...

    def save_kv_layer(self, layer_name: str, kv_layer: object, attn_metadata: object, **kwargs: object) -> None: ...

    def wait_for_save(self) -> None: ...

    def request_finished(self, request: object, block_ids: list[int]) -> tuple[bool, Mapping[str, Any] | None]: ...

    def request_finished_all_groups(
        self,
        request: object,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Mapping[str, Any] | None]: ...


class NoOpDocumentKVProvider:
    """Safe provider used when no document KV provider factory is configured."""

    def get_num_new_matched_tokens(
        self,
        request: object,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(self, request: object, blocks: object, num_external_tokens: int) -> None:
        return None

    def build_connector_meta(self, scheduler_output: object) -> dict[str, object]:
        return {}

    def register_kv_caches(self, kv_caches: Mapping[str, object]) -> None:
        return None

    def start_load_kv(self, forward_context: object, **kwargs: object) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, layer_name: str, kv_layer: object, attn_metadata: object, **kwargs: object) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def request_finished(self, request: object, block_ids: list[int]) -> tuple[bool, None]:
        return False, None

    def request_finished_all_groups(self, request: object, block_ids: tuple[list[int], ...]) -> tuple[bool, None]:
        return False, None


class DocumentKVConnector(_KVConnectorBaseV1, _SupportsHMA):
    """Dynamic vLLM V1 connector that delegates KV work to a provider.

    vLLM owns scheduling, block allocation, attention, and request cleanup. This
    connector is intentionally a thin bridge from vLLM's V1 KV connector hooks to
    a provider that knows how to materialize Cachet payloads into vLLM-owned
    paged KV buffers for the deployed engine version.
    """

    def __init__(
        self,
        vllm_config: object | None = None,
        role: object | None = None,
        kv_cache_config: object | None = None,
        *,
        provider: DocumentKVProvider | None = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self.vllm_config = vllm_config
        self.connector_role = role
        self.kv_cache_config = kv_cache_config
        self.provider = provider or _provider_from_vllm_config(vllm_config)

    def get_num_new_matched_tokens(
        self,
        request: object,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return self.provider.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: object, blocks: object, num_external_tokens: int) -> None:
        self.provider.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: object) -> object:
        return self.provider.build_connector_meta(scheduler_output)

    def register_kv_caches(self, kv_caches: Mapping[str, object]) -> None:
        self.provider.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: object) -> None:
        super().bind_connector_metadata(connector_metadata)
        binder = getattr(self.provider, "bind_connector_metadata", None)
        if callable(binder):
            binder(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        clearer = getattr(self.provider, "clear_connector_metadata", None)
        if callable(clearer):
            clearer()

    def start_load_kv(self, forward_context: object, **kwargs: object) -> None:
        self.provider.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self.provider.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: object, attn_metadata: object, **kwargs: object) -> None:
        self.provider.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        self.provider.wait_for_save()

    def request_finished(self, request: object, block_ids: list[int]) -> tuple[bool, Mapping[str, Any] | None]:
        return self.provider.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: object,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Mapping[str, Any] | None]:
        return self.provider.request_finished_all_groups(request, block_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        getter = getattr(self.provider, "get_block_ids_with_load_errors", None)
        if callable(getter):
            return set(getter())
        return set()

    def get_kv_connector_stats(self) -> object | None:
        getter = getattr(self.provider, "get_kv_connector_stats", None)
        if callable(getter):
            return getter()
        return None

    def take_events(self) -> list[object]:
        getter = getattr(self.provider, "take_events", None)
        if callable(getter):
            return list(getter())
        return []


def load_document_kv_provider_factory(factory_path: str) -> object:
    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("document KV provider factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError(f"document KV provider factory {factory_path!r} is not callable")
    return factory


def _provider_from_vllm_config(vllm_config: object | None) -> DocumentKVProvider:
    extra_config = _kv_connector_extra_config(vllm_config)
    factory_path = extra_config.get(DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY)
    if factory_path is None:
        return NoOpDocumentKVProvider()
    if not isinstance(factory_path, str) or not factory_path.strip():
        raise ValueError(f"{DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY} must be a non-empty module:attribute string")
    factory = load_document_kv_provider_factory(factory_path)
    provider = factory(vllm_config=vllm_config, extra_config=extra_config)
    if isinstance(provider, NoOpDocumentKVProvider):
        raise ValueError("configured document KV provider factory cannot return NoOpDocumentKVProvider")
    _validate_provider(provider)
    return provider


def _kv_connector_extra_config(vllm_config: object | None) -> Mapping[str, Any]:
    if vllm_config is None:
        return {}
    for candidate in (
        getattr(vllm_config, "kv_transfer_config", None),
        getattr(getattr(vllm_config, "cache_config", None), "kv_transfer_config", None),
    ):
        extra_config = _extra_config_from_transfer_config(candidate)
        if extra_config is not None:
            return extra_config
    return {}


def _extra_config_from_transfer_config(transfer_config: object | None) -> Mapping[str, Any] | None:
    if transfer_config is None:
        return None
    if isinstance(transfer_config, Mapping):
        value = transfer_config.get("kv_connector_extra_config")
    else:
        value = getattr(transfer_config, "kv_connector_extra_config", None)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("kv_connector_extra_config must be a mapping")
    return value


def _validate_provider(provider: object) -> None:
    missing = [
        method_name
        for method_name in (
            "get_num_new_matched_tokens",
            "update_state_after_alloc",
            "build_connector_meta",
            "register_kv_caches",
            "start_load_kv",
            "wait_for_layer_load",
            "save_kv_layer",
            "wait_for_save",
            "request_finished",
            "request_finished_all_groups",
        )
        if not callable(getattr(provider, method_name, None))
    ]
    if missing:
        raise TypeError("document KV provider must provide callable methods: " + ", ".join(missing))
