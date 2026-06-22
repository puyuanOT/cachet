"""SGLang HiCache dynamic backend entrypoint for document KV providers."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

DOCUMENT_KV_HICACHE_BACKEND_CLASS = "DocumentKVHiCacheBackend"
DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH = "sglang_kv_injection.sglang_dynamic_backend"
DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY = "document_kv.provider_factory"

__all__ = [
    "DOCUMENT_KV_HICACHE_BACKEND_CLASS",
    "DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH",
    "DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY",
    "DocumentKVHiCacheBackend",
    "DocumentKVHiCacheProvider",
    "NoOpDocumentKVHiCacheProvider",
    "load_document_kv_hicache_provider_factory",
]


class DocumentKVHiCacheProvider(Protocol):
    """Provider boundary for version-sensitive SGLang HiCache storage operations."""

    def get(self, key: object) -> object | None: ...

    def exists(self, key: object) -> bool: ...

    def set(self, key: object, value: object) -> object | None: ...


class NoOpDocumentKVHiCacheProvider:
    """Safe miss-only provider used when no provider factory is configured."""

    def get(self, key: object) -> None:
        return None

    def exists(self, key: object) -> bool:
        return False

    def exist(self, key: object) -> bool:
        return self.exists(key)

    def set(self, key: object, value: object) -> None:
        return None


class DocumentKVHiCacheBackend:
    """Dynamic SGLang HiCache backend that delegates storage calls to a provider.

    SGLang owns RadixAttention, HiCache page management, prefetch, write-back,
    and synchronization. This class is only the importable dynamic backend that
    hands documented HiCache storage operations to a Cachet-aware provider.
    """

    def __init__(
        self,
        *args: object,
        provider: DocumentKVHiCacheProvider | None = None,
        **kwargs: object,
    ) -> None:
        self.extra_config = _backend_extra_config(args=args, kwargs=kwargs)
        self.provider = provider or _provider_from_extra_config(self.extra_config)

    def get(self, key: object) -> object | None:
        return self.provider.get(key)

    def exists(self, key: object) -> bool:
        return bool(_provider_exists(self.provider, key))

    def exist(self, key: object) -> bool:
        return self.exists(key)

    def set(self, key: object, value: object) -> object | None:
        return self.provider.set(key, value)

    def batch_get_v1(self, keys: Iterable[object]) -> list[object | None]:
        handler = getattr(self.provider, "batch_get_v1", None)
        if callable(handler):
            return list(handler(keys))
        return [self.get(key) for key in keys]

    def batch_set_v1(self, keys: Iterable[object], values: Iterable[object]) -> object | None:
        handler = getattr(self.provider, "batch_set_v1", None)
        if callable(handler):
            return handler(keys, values)
        for key, value in zip(keys, values, strict=True):
            self.set(key, value)
        return None


def load_document_kv_hicache_provider_factory(factory_path: str) -> object:
    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("document KV HiCache provider factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError(f"document KV HiCache provider factory {factory_path!r} is not callable")
    return factory


def _provider_from_extra_config(extra_config: Mapping[str, Any]) -> DocumentKVHiCacheProvider:
    factory_path = extra_config.get(DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY)
    if factory_path is None:
        return NoOpDocumentKVHiCacheProvider()
    if not isinstance(factory_path, str) or not factory_path.strip():
        raise ValueError(f"{DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY} must be a non-empty module:attribute string")
    factory = load_document_kv_hicache_provider_factory(factory_path)
    provider = factory(extra_config=extra_config)
    if isinstance(provider, NoOpDocumentKVHiCacheProvider):
        raise ValueError("configured document KV HiCache provider factory cannot return NoOpDocumentKVHiCacheProvider")
    _validate_provider(provider)
    return provider


def _backend_extra_config(*, args: tuple[object, ...], kwargs: Mapping[str, object]) -> dict[str, Any]:
    extra_config: dict[str, Any] = {}
    for candidate in args:
        _merge_extra_config(extra_config, candidate)
    for key, value in kwargs.items():
        if key in {"config", "extra_config", "hicache_storage_backend_extra_config"}:
            _merge_extra_config(extra_config, value)
        else:
            extra_config[key] = value
    return extra_config


def _merge_extra_config(target: dict[str, Any], value: object) -> None:
    if value is None:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("SGLang HiCache extra config keys must be strings")
            target[key] = item
        return
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("SGLang HiCache extra config strings must be JSON objects") from exc
        if not isinstance(decoded, Mapping):
            raise TypeError("SGLang HiCache extra config JSON must decode to an object")
        _merge_extra_config(target, decoded)


def _validate_provider(provider: object) -> None:
    missing = [
        method_name
        for method_name in ("get", "set")
        if not callable(getattr(provider, method_name, None))
    ]
    if _provider_exists_handler(provider) is None:
        missing.append("exists")
    if missing:
        raise TypeError(
            "document KV HiCache provider must provide callable methods: "
            + ", ".join(missing)
            + "; exist is accepted as an alias for exists"
        )


def _provider_exists(provider: object, key: object) -> bool:
    handler = _provider_exists_handler(provider)
    if handler is None:  # pragma: no cover - provider validation catches this.
        raise TypeError("document KV HiCache provider must provide exists or exist")
    return bool(handler(key))


def _provider_exists_handler(provider: object) -> object | None:
    handler = getattr(provider, "exists", None)
    if callable(handler):
        return handler
    handler = getattr(provider, "exist", None)
    if callable(handler):
        return handler
    return None
