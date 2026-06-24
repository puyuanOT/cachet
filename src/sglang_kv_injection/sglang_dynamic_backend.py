"""SGLang HiCache dynamic backend entrypoint for document KV providers."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import logging
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from document_kv_cache.engine_adapters import (
    EngineKVInjectionPlan,
    ServingBackend,
    build_engine_kv_injection_plan,
    read_engine_adapter_request_json,
)
from document_kv_cache.engine_probe import read_engine_adapter_payload
from document_kv_cache.benchmarks import DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM
from sglang_kv_injection.sglang_request_metadata_bridge import (
    DOCUMENT_KV_SGLANG_HICACHE_LAST_HASH_EXTRA_INFO_KEY,
    SGLangRequestMetadataBridgeStatus,
    install_sglang_request_metadata_bridge,
)


def _load_sglang_hicache_storage_base() -> type:
    try:
        module = importlib.import_module("sglang.srt.mem_cache.hicache_storage")
        base = getattr(module, "HiCacheStorage")
    except Exception:
        return object
    if isinstance(base, type):
        return base
    return object


DOCUMENT_KV_HICACHE_BACKEND_CLASS = "DocumentKVHiCacheBackend"
DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH = "sglang_kv_injection.sglang_dynamic_backend"
DOCUMENT_KV_HICACHE_PROVIDER_FACTORY = (
    "sglang_kv_injection.sglang_dynamic_backend:build_document_kv_hicache_provider"
)
DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY = "document_kv.provider_factory"
DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY = "cachet.hicache_page_store_uri"
DOCUMENT_KV_HICACHE_REQUEST_CONTEXT_KWARG = "document_kv_request_context"
_SGLANG_CUSTOM_PARAMS_KEY = "custom_params"
_SGLANG_KV_TRANSFER_PARAMS_KEY = "kv_transfer_params"
_DOCUMENT_KV_REQUEST_ID_PARAM = "document_kv.request_id"
_DOCUMENT_KV_HANDOFF_JSON_PARAM = "document_kv.handoff_json"
_DOCUMENT_KV_HANDOFF_RECORD_PARAM = "document_kv.handoff_record"
_DOCUMENT_KV_PAYLOAD_URI_PARAM = "document_kv.payload_uri"
_DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM = "document_kv.prompt_text_mode"
DOCUMENT_KV_HICACHE_RUNTIME_METHODS = (
    "register_mem_pool_host",
    "register_mem_host_pool_v2",
    "batch_exists",
    "batch_exists_v2",
    "batch_get",
    "batch_get_v1",
    "batch_get_v2",
    "batch_set",
    "batch_set_v1",
    "batch_set_v2",
    "get",
    "set",
    "exists",
    "clear",
    "get_stats",
)
_SGLANG_HICACHE_STORAGE_BASE = _load_sglang_hicache_storage_base()
_LOGGER = logging.getLogger(__name__)

__all__ = [
    "DOCUMENT_KV_HICACHE_BACKEND_CLASS",
    "DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH",
    "DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY",
    "DOCUMENT_KV_HICACHE_PROVIDER_FACTORY",
    "DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY",
    "DOCUMENT_KV_HICACHE_REQUEST_CONTEXT_KWARG",
    "DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM",
    "DOCUMENT_KV_SGLANG_HICACHE_LAST_HASH_EXTRA_INFO_KEY",
    "DOCUMENT_KV_HICACHE_RUNTIME_METHODS",
    "DocumentKVHiCacheBackend",
    "DocumentKVHiCachePageProvider",
    "DocumentKVHiCacheProvider",
    "DocumentKVHiCacheRequestContext",
    "NoOpDocumentKVHiCacheProvider",
    "SGLangRequestMetadataBridgeStatus",
    "build_document_kv_hicache_provider",
    "document_kv_request_context_from_extra_info",
    "load_document_kv_hicache_provider_factory",
]


class DocumentKVHiCacheProvider(Protocol):
    """Provider boundary for version-sensitive SGLang HiCache storage operations."""

    def get(
        self,
        key: object,
        target_location: object | None = None,
        target_sizes: object | None = None,
    ) -> object | None: ...

    def exists(self, key: object) -> bool: ...

    def set(
        self,
        key: object,
        value: object | None = None,
        target_location: object | None = None,
        target_sizes: object | None = None,
    ) -> object | None: ...


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


@dataclass(frozen=True, slots=True)
class DocumentKVHiCacheRequestContext:
    """Cachet request metadata forwarded from SGLang HiCache extra_info."""

    kv_transfer_params: Mapping[str, Any]
    request_id: str | None = None
    handoff_json: str | None = None
    payload_uri: str | None = None
    handoff_record: Mapping[str, Any] | None = None
    prompt_text_mode: str | None = None
    sglang_hicache_page_keys: tuple[str, ...] = ()
    sglang_hicache_last_hash: str | None = None
    prefix_keys: tuple[str, ...] = ()
    raw_extra_info: Mapping[str, Any] | None = None


class DocumentKVHiCachePageProvider:
    """Runtime-facing HiCache page provider backed by memory or a filesystem path."""

    document_kv_hicache_provider = True

    def __init__(self, *, store_uri: str | None = None, storage_identity: str | None = None) -> None:
        store_uri = _optional_config_string(
            store_uri,
            field_name=DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY,
        )
        self.store_uri = store_uri
        self.storage_identity = storage_identity
        self.root = None if _is_memory_page_store_uri(store_uri) else _page_store_root(store_uri)
        self._memory_pages: dict[str, bytes] = {}
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.mem_pool_host: object | None = None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def register_mem_pool_host(self, mem_pool_host: object) -> None:
        self.mem_pool_host = mem_pool_host

    def get(
        self,
        key: object,
        target_location: object | None = None,
        target_sizes: object | None = None,
    ) -> object | None:
        raw_page = self._read_page(key)
        if raw_page is None:
            self.misses += 1
            return None
        self.hits += 1
        return _copy_page_to_target(raw_page, target_location)

    def exists(self, key: object) -> bool:
        if self.root is None:
            return _page_key(key, self.storage_identity) in self._memory_pages
        return _page_path(self.root, key, self.storage_identity).is_file()

    def exist(self, key: object) -> bool:
        return self.exists(key)

    def set(
        self,
        key: object,
        value: object | None = None,
        target_location: object | None = None,
        target_sizes: object | None = None,
    ) -> bool:
        raw_page = _page_bytes(value if value is not None else target_location)
        try:
            self._write_page(key, raw_page)
        except OSError:
            return False
        self.sets += 1
        return True

    def batch_exists(
        self,
        keys: Iterable[object],
        *,
        extra_info: object | None = None,
        document_kv_request_context: DocumentKVHiCacheRequestContext | None = None,
    ) -> int:
        del extra_info
        key_list = list(keys)
        hydrated_count = self._hydrate_request_pages(key_list, document_kv_request_context)
        if _requires_strict_handoff_binding(document_kv_request_context):
            _log_sglang_handoff_operation(
                "batch_exists",
                key_count=len(key_list),
                request_context=document_kv_request_context,
                hydrated_count=hydrated_count,
            )
            if hydrated_count <= 0:
                return 0
            for index, key in enumerate(key_list[:hydrated_count]):
                if not self.exists(key):
                    return index
            return hydrated_count
        for index, key in enumerate(key_list):
            if not self.exists(key):
                return index
        return len(key_list)

    def batch_get_v1(
        self,
        keys: Iterable[object],
        *,
        host_indices: object | None = None,
        extra_info: object | None = None,
        document_kv_request_context: DocumentKVHiCacheRequestContext | None = None,
    ) -> list[object | None] | list[bool]:
        del extra_info
        key_list = list(keys)
        hydrated_count = self._hydrate_request_pages(key_list, document_kv_request_context)
        strict_handoff = _requires_strict_handoff_binding(document_kv_request_context)
        if strict_handoff:
            _log_sglang_handoff_operation(
                "batch_get_v1",
                key_count=len(key_list),
                request_context=document_kv_request_context,
                hydrated_count=hydrated_count,
                host_indices_present=host_indices is not None,
            )
        if strict_handoff and hydrated_count <= 0:
            if host_indices is None:
                return [None] * len(key_list)
            return [False] * len(key_list)
        if host_indices is None:
            if strict_handoff and hydrated_count < len(key_list):
                values: list[object | None] = [None] * len(key_list)
                for index, key in enumerate(key_list[:hydrated_count]):
                    values[index] = self.get(key)
                return values
            return [self.get(key) for key in key_list]
        load_count = hydrated_count if strict_handoff else len(key_list)
        mem_pool_host = self.mem_pool_host
        host_indices_cover_pages = (
            _host_indices_cover_at_least_pages
            if strict_handoff and load_count < len(key_list)
            else _host_indices_cover_pages
        )
        if mem_pool_host is None or not host_indices_cover_pages(host_indices, load_count, mem_pool_host):
            return [False] * len(key_list)
        loaded = [False] * len(key_list)
        for index, key in enumerate(key_list[:load_count]):
            loaded[index] = _load_host_page(
                self,
                key,
                mem_pool_host,
                _host_page_offset(host_indices, index, mem_pool_host),
            )
        return loaded

    def batch_set_v1(
        self,
        keys: Iterable[object],
        *,
        host_indices: object | None = None,
        extra_info: object | None = None,
        document_kv_request_context: DocumentKVHiCacheRequestContext | None = None,
    ) -> list[bool] | None:
        del extra_info, document_kv_request_context
        key_list = list(keys)
        if host_indices is None:
            return [False] * len(key_list)
        mem_pool_host = self.mem_pool_host
        if mem_pool_host is not None:
            if not _host_indices_cover_pages(host_indices, len(key_list), mem_pool_host):
                return [False] * len(key_list)
            return [
                _store_provider_host_page(
                    self,
                    key,
                    mem_pool_host,
                    _host_page_offset(host_indices, index, mem_pool_host),
                )
                for index, key in enumerate(key_list)
            ]
        if _looks_like_host_indices(host_indices):
            return [False] * len(key_list)
        for key, value in zip(key_list, host_indices, strict=True):
            self.set(key, value)
        return None

    def clear(self) -> None:
        self._memory_pages.clear()
        if self.root is not None:
            for path in self.root.glob(f"{_page_namespace_prefix(self.storage_identity)}-*{_PAGE_FILE_SUFFIX}"):
                path.unlink(missing_ok=True)

    def delete(self, key: object) -> bool:
        if self.root is None:
            return self._memory_pages.pop(_page_key(key, self.storage_identity), None) is not None
        try:
            _page_path(self.root, key, self.storage_identity).unlink()
        except FileNotFoundError:
            return False
        except OSError:
            return False
        return True

    def get_stats(self) -> dict[str, object]:
        return {
            "provider": self.__class__.__name__,
            "store": "memory" if self.root is None else "filesystem",
            "store_uri": self.store_uri,
            "storage_identity": self.storage_identity,
            "pages": self._page_count(),
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
        }

    def _read_page(self, key: object) -> bytes | None:
        if self.root is None:
            return self._memory_pages.get(_page_key(key, self.storage_identity))
        path = _page_path(self.root, key, self.storage_identity)
        if not path.is_file():
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    def _write_page(self, key: object, raw_page: bytes) -> None:
        if self.root is None:
            self._memory_pages[_page_key(key, self.storage_identity)] = raw_page
            return
        _atomic_write_page(_page_path(self.root, key, self.storage_identity), raw_page)

    def _page_count(self) -> int:
        if self.root is None:
            return len(self._memory_pages)
        return sum(1 for _ in self.root.glob(f"{_page_namespace_prefix(self.storage_identity)}-*{_PAGE_FILE_SUFFIX}"))

    def _hydrate_request_pages(
        self,
        keys: list[object],
        request_context: DocumentKVHiCacheRequestContext | None,
    ) -> int:
        if not keys or request_context is None:
            return 0
        binding = _sglang_hicache_page_binding(keys, request_context)
        if binding is None:
            if _requires_strict_handoff_binding(request_context):
                _log_sglang_handoff_hydration(
                    "binding_miss",
                    keys=keys,
                    request_context=request_context,
                    level=logging.WARNING,
                )
            return 0
        first_page_index, key_count = binding
        handoff_plan = self._handoff_plan_for_context(request_context)
        if handoff_plan is None:
            _log_sglang_handoff_hydration(
                "handoff_missing",
                keys=keys,
                request_context=request_context,
                binding_key_count=key_count,
                level=logging.WARNING,
            )
            return 0
        plan, payload_uri = handoff_plan
        full_page_count = _sglang_hicache_full_page_count(plan)
        if first_page_index + key_count > full_page_count:
            _log_sglang_handoff_hydration(
                "binding_exceeds_handoff_pages",
                keys=keys,
                request_context=request_context,
                binding_key_count=key_count,
                full_page_count=full_page_count,
                level=logging.WARNING,
            )
            return 0
        binding_keys = keys[:key_count]
        if all(self.exists(key) for key in binding_keys):
            _log_sglang_handoff_hydration(
                "pages_already_hydrated",
                keys=keys,
                request_context=request_context,
                binding_key_count=key_count,
                hydrated_count=key_count,
                full_page_count=full_page_count,
            )
            return key_count
        pages = _sglang_hicache_payload_pages(
            plan,
            read_engine_adapter_payload(payload_uri, expected_bytes=plan.total_bytes),
        )
        if not pages:
            _log_sglang_handoff_hydration(
                "payload_has_no_full_pages",
                keys=keys,
                request_context=request_context,
                binding_key_count=key_count,
                full_page_count=full_page_count,
                level=logging.WARNING,
            )
            return 0
        for offset, key in enumerate(binding_keys):
            page_index = first_page_index + offset
            if self.exists(key):
                continue
            self._write_page(key, pages[page_index])
        _log_sglang_handoff_hydration(
            "pages_hydrated",
            keys=keys,
            request_context=request_context,
            binding_key_count=key_count,
            hydrated_count=key_count,
            full_page_count=full_page_count,
        )
        return key_count

    def _handoff_plan_for_context(
        self,
        request_context: DocumentKVHiCacheRequestContext,
    ) -> tuple[EngineKVInjectionPlan, str] | None:
        handoff_record = _handoff_record_from_request_context(request_context)
        if handoff_record is None:
            return None
        plan = build_engine_kv_injection_plan(
            handoff_record,
            expected_backend=ServingBackend.SGLANG,
            require_external_payload_uri=request_context.payload_uri is None,
        )
        _validate_handoff_request_context(request_context, plan)
        payload_uri = request_context.payload_uri or plan.payload_source_uri
        if payload_uri is None:
            raise ValueError("SGLang document KV handoff requires an external payload URI")
        return plan, payload_uri


class DocumentKVHiCacheBackend(_SGLANG_HICACHE_STORAGE_BASE):
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
        self.storage_identity = _storage_identity_from_runtime_args(args=args, kwargs=kwargs)
        self.extra_config = _backend_extra_config(args=args, kwargs=kwargs)
        self.request_metadata_bridge_status = install_sglang_request_metadata_bridge()
        self.provider = provider or _provider_from_extra_config(
            self.extra_config,
            storage_identity=self.storage_identity,
        )

    def register_mem_pool_host(self, mem_pool_host: object) -> object | None:
        self.mem_pool_host = mem_pool_host
        handler = getattr(self.provider, "register_mem_pool_host", None)
        if callable(handler):
            return handler(mem_pool_host)
        return None

    def register_mem_host_pool_v2(self, host_pool: object, host_pool_name: object) -> object | None:
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        self.registered_pools[host_pool_name] = host_pool
        handler = getattr(self.provider, "register_mem_host_pool_v2", None)
        if callable(handler):
            return handler(host_pool, host_pool_name)
        return None

    def get(
        self,
        key: object,
        target_location: object | None = None,
        target_sizes: object | None = None,
    ) -> object | None:
        return _call_provider_method(
            self.provider.get,
            key,
            optional_kwargs={
                "target_location": target_location,
                "target_sizes": target_sizes,
            },
        )

    def exists(self, key: object) -> bool:
        return bool(_provider_exists(self.provider, key))

    def exist(self, key: object) -> bool:
        return self.exists(key)

    def set(
        self,
        key: object,
        value: object | None = None,
        target_location: object | None = None,
        target_sizes: object | None = None,
    ) -> object | None:
        return _call_provider_method(
            self.provider.set,
            key,
            value,
            optional_kwargs={
                "target_location": target_location,
                "target_sizes": target_sizes,
            },
        )

    def batch_exists(self, keys: Iterable[object], extra_info: object | None = None) -> int:
        key_list = list(keys)
        request_context = document_kv_request_context_from_extra_info(extra_info)
        handler = getattr(self.provider, "batch_exists", None)
        if callable(handler):
            return int(
                _call_provider_method(
                    handler,
                    key_list,
                    optional_kwargs={
                        "extra_info": extra_info,
                        DOCUMENT_KV_HICACHE_REQUEST_CONTEXT_KWARG: request_context,
                    },
                )
            )
        for index, key in enumerate(key_list):
            if not self.exists(key):
                return index
        return len(key_list)

    def batch_exists_v2(
        self,
        keys: Iterable[object],
        pool_transfers: Iterable[object] | None = None,
        extra_info: object | None = None,
    ) -> object:
        key_list = list(keys)
        transfer_list = list(pool_transfers or ())
        request_context = document_kv_request_context_from_extra_info(extra_info)
        handler = getattr(self.provider, "batch_exists_v2", None)
        if callable(handler):
            return _call_provider_method(
                handler,
                key_list,
                optional_kwargs={
                    "pool_transfers": transfer_list,
                    "extra_info": extra_info,
                    DOCUMENT_KV_HICACHE_REQUEST_CONTEXT_KWARG: request_context,
                },
            )
        kv_hit_pages = self.batch_exists(key_list, extra_info=extra_info)
        extra_pool_hit_pages = _extra_pool_hit_pages(self, key_list, transfer_list, kv_hit_pages)
        if extra_pool_hit_pages:
            kv_hit_pages = min(kv_hit_pages, *extra_pool_hit_pages.values())
        return _pool_transfer_result(kv_hit_pages, extra_pool_hit_pages)

    def batch_get(
        self,
        keys: Iterable[object],
        target_locations: Iterable[object] | None = None,
        target_sizes: Iterable[object] | object | None = None,
    ) -> list[object | None] | int:
        key_list = list(keys)
        handler = getattr(self.provider, "batch_get", None)
        if callable(handler):
            return _call_provider_method(
                handler,
                key_list,
                optional_kwargs={
                    "target_locations": target_locations,
                    "target_sizes": target_sizes,
                },
            )
        locations = _optional_sequence(target_locations, len(key_list))
        sizes = _optional_sequence(target_sizes, len(key_list))
        return [
            self.get(key, target_location=location, target_sizes=size)
            for key, location, size in zip(key_list, locations, sizes, strict=True)
        ]

    def batch_get_v1(
        self,
        keys: Iterable[object],
        host_indices: object | None = None,
        extra_info: object | None = None,
    ) -> list[object | None] | list[bool]:
        key_list = list(keys)
        request_context = document_kv_request_context_from_extra_info(extra_info)
        handler = getattr(self.provider, "batch_get_v1", None)
        if callable(handler):
            return list(
                _call_provider_method(
                    handler,
                    key_list,
                    optional_kwargs={
                        "host_indices": host_indices,
                        "extra_info": extra_info,
                        DOCUMENT_KV_HICACHE_REQUEST_CONTEXT_KWARG: request_context,
                    },
                )
            )
        if host_indices is None:
            return [self.get(key) for key in key_list]
        mem_pool_host = getattr(self, "mem_pool_host", None)
        if mem_pool_host is None:
            return [False] * len(key_list)
        if not _host_indices_cover_pages(host_indices, len(key_list), mem_pool_host):
            return [False] * len(key_list)
        return [
            _load_host_page(self, key, mem_pool_host, _host_page_offset(host_indices, index, mem_pool_host))
            for index, key in enumerate(key_list)
        ]

    def batch_get_v2(
        self,
        transfers: Iterable[object],
        extra_info: object | None = None,
    ) -> dict[str, list[bool]]:
        transfer_list = list(transfers)
        request_context = document_kv_request_context_from_extra_info(extra_info)
        handler = getattr(self.provider, "batch_get_v2", None)
        if callable(handler):
            return dict(
                _call_provider_method(
                    handler,
                    transfer_list,
                    optional_kwargs={
                        "extra_info": extra_info,
                        DOCUMENT_KV_HICACHE_REQUEST_CONTEXT_KWARG: request_context,
                    },
                )
            )
        return _batch_pool_transfer(self, transfer_list, operation="get")

    def batch_set(
        self,
        keys: Iterable[object],
        values: Iterable[object] | None = None,
        target_locations: Iterable[object] | None = None,
        target_sizes: Iterable[object] | object | None = None,
    ) -> bool:
        key_list = list(keys)
        handler = getattr(self.provider, "batch_set", None)
        if callable(handler):
            return bool(
                _call_provider_method(
                    handler,
                    key_list,
                    optional_kwargs={
                        "values": values,
                        "target_locations": target_locations,
                        "target_sizes": target_sizes,
                    },
                )
            )
        value_list = _optional_sequence(values if values is not None else target_locations, len(key_list))
        sizes = _optional_sequence(target_sizes, len(key_list))
        results = [
            _set_result_ok(self, self.set(key, value, target_sizes=size))
            for key, value, size in zip(key_list, value_list, sizes, strict=True)
        ]
        return all(results)

    def batch_set_v1(
        self,
        keys: Iterable[object],
        host_indices: object | None = None,
        extra_info: object | None = None,
    ) -> object | None:
        key_list = list(keys)
        request_context = document_kv_request_context_from_extra_info(extra_info)
        handler = getattr(self.provider, "batch_set_v1", None)
        if callable(handler):
            return _call_provider_method(
                handler,
                key_list,
                optional_kwargs={
                    "host_indices": host_indices,
                    "extra_info": extra_info,
                    DOCUMENT_KV_HICACHE_REQUEST_CONTEXT_KWARG: request_context,
                },
            )
        if host_indices is None:
            return [False] * len(key_list)
        mem_pool_host = getattr(self, "mem_pool_host", None)
        if mem_pool_host is None and _looks_like_host_indices(host_indices):
            return [False] * len(key_list)
        if mem_pool_host is not None:
            if not _host_indices_cover_pages(host_indices, len(key_list), mem_pool_host):
                return [False] * len(key_list)
            return [
                _store_host_page(self, key, mem_pool_host, _host_page_offset(host_indices, index, mem_pool_host))
                for index, key in enumerate(key_list)
            ]
        for key, value in zip(key_list, host_indices, strict=True):
            self.set(key, value)
        return None

    def batch_set_v2(
        self,
        transfers: Iterable[object],
        extra_info: object | None = None,
    ) -> dict[str, list[bool]]:
        transfer_list = list(transfers)
        request_context = document_kv_request_context_from_extra_info(extra_info)
        handler = getattr(self.provider, "batch_set_v2", None)
        if callable(handler):
            return dict(
                _call_provider_method(
                    handler,
                    transfer_list,
                    optional_kwargs={
                        "extra_info": extra_info,
                        DOCUMENT_KV_HICACHE_REQUEST_CONTEXT_KWARG: request_context,
                    },
                )
            )
        return _batch_pool_transfer(self, transfer_list, operation="set")

    def clear(self) -> object | None:
        handler = getattr(self.provider, "clear", None)
        if callable(handler):
            return handler()
        return None

    def get_stats(self) -> object | None:
        handler = getattr(self.provider, "get_stats", None)
        if callable(handler):
            return handler()
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


def build_document_kv_hicache_provider(
    *,
    extra_config: Mapping[str, Any] | None = None,
    storage_identity: str | None = None,
) -> DocumentKVHiCachePageProvider:
    """Build Cachet's built-in SGLang HiCache page provider."""

    if extra_config is None:
        extra_config = {}
    if not isinstance(extra_config, Mapping):
        raise TypeError("SGLang HiCache provider extra_config must be a mapping")
    store_uri = _optional_config_string(
        extra_config.get(DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY),
        field_name=DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY,
    )
    return DocumentKVHiCachePageProvider(store_uri=store_uri, storage_identity=storage_identity)


def document_kv_request_context_from_extra_info(extra_info: object | None) -> DocumentKVHiCacheRequestContext | None:
    """Extract Cachet request metadata from SGLang HiCache extra_info when present."""

    raw_extra_info = _request_extra_info_mapping(extra_info)
    if raw_extra_info is None:
        return None
    kv_transfer_params = _kv_transfer_params_from_extra_info(raw_extra_info)
    if kv_transfer_params is None:
        return None
    return DocumentKVHiCacheRequestContext(
        kv_transfer_params=kv_transfer_params,
        request_id=_optional_string(kv_transfer_params.get(_DOCUMENT_KV_REQUEST_ID_PARAM)),
        handoff_json=_optional_string(kv_transfer_params.get(_DOCUMENT_KV_HANDOFF_JSON_PARAM)),
        payload_uri=_optional_string(kv_transfer_params.get(_DOCUMENT_KV_PAYLOAD_URI_PARAM)),
        handoff_record=_optional_mapping(kv_transfer_params.get(_DOCUMENT_KV_HANDOFF_RECORD_PARAM)),
        prompt_text_mode=_optional_string(kv_transfer_params.get(_DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM)),
        sglang_hicache_page_keys=_sglang_hicache_page_keys_from_params(kv_transfer_params),
        sglang_hicache_last_hash=_sglang_hicache_last_hash_from_extra_info(raw_extra_info),
        prefix_keys=_prefix_keys_from_extra_info(extra_info),
        raw_extra_info=raw_extra_info,
    )


def _handoff_record_from_request_context(
    request_context: DocumentKVHiCacheRequestContext,
) -> Mapping[str, Any] | None:
    handoff_record = request_context.handoff_record
    handoff_json = request_context.handoff_json
    if handoff_record is None and handoff_json is None:
        return None
    if handoff_record is not None and handoff_json is not None:
        raise ValueError(
            f"Use only one of {_DOCUMENT_KV_HANDOFF_RECORD_PARAM} "
            f"or {_DOCUMENT_KV_HANDOFF_JSON_PARAM}"
        )
    if handoff_record is not None:
        return handoff_record
    assert handoff_json is not None
    return read_engine_adapter_request_json(
        handoff_json,
        expected_backend=ServingBackend.SGLANG,
        require_external_payload_uri=request_context.payload_uri is None,
    )


def _validate_handoff_request_context(
    request_context: DocumentKVHiCacheRequestContext,
    plan: EngineKVInjectionPlan,
) -> None:
    if request_context.request_id is not None and request_context.request_id != plan.request_id:
        raise ValueError(f"{_DOCUMENT_KV_REQUEST_ID_PARAM} must match handoff request_id")


def _requires_strict_handoff_binding(
    request_context: DocumentKVHiCacheRequestContext | None,
) -> bool:
    if request_context is None:
        return False
    return request_context.handoff_json is not None or request_context.handoff_record is not None


def _sglang_hicache_page_binding(
    keys: list[object],
    request_context: DocumentKVHiCacheRequestContext,
) -> tuple[int, int] | None:
    expected_page_keys = request_context.sglang_hicache_page_keys
    if not expected_page_keys:
        return None
    runtime_keys = _runtime_hicache_keys(keys)
    if runtime_keys is None:
        return None
    prefix_keys = request_context.prefix_keys
    if prefix_keys:
        first_page_index = len(prefix_keys)
        if first_page_index > len(expected_page_keys):
            return None
        if prefix_keys != expected_page_keys[:first_page_index]:
            return None
        return _sglang_hicache_page_binding_from_offset(
            runtime_keys,
            expected_page_keys,
            first_page_index,
        )
    last_hash = request_context.sglang_hicache_last_hash
    if last_hash is not None:
        return _sglang_hicache_page_binding_after_last_hash(
            runtime_keys,
            expected_page_keys,
            last_hash,
        )
    return _sglang_hicache_page_binding_by_runtime_keys(runtime_keys, expected_page_keys)


def _sglang_hicache_page_binding_after_last_hash(
    runtime_keys: tuple[str, ...],
    expected_page_keys: tuple[str, ...],
    last_hash: str,
) -> tuple[int, int] | None:
    try:
        first_page_index = expected_page_keys.index(last_hash) + 1
    except ValueError:
        return None
    return _sglang_hicache_page_binding_from_offset(
        runtime_keys,
        expected_page_keys,
        first_page_index,
    )


def _sglang_hicache_page_binding_by_runtime_keys(
    runtime_keys: tuple[str, ...],
    expected_page_keys: tuple[str, ...],
) -> tuple[int, int] | None:
    if not runtime_keys:
        return None
    first_runtime_key = runtime_keys[0]
    for first_page_index, expected_page_key in enumerate(expected_page_keys):
        if first_runtime_key != expected_page_key:
            continue
        binding = _sglang_hicache_page_binding_from_offset(
            runtime_keys,
            expected_page_keys,
            first_page_index,
        )
        if binding is not None:
            return binding
    return None


def _sglang_hicache_page_binding_from_offset(
    runtime_keys: tuple[str, ...],
    expected_page_keys: tuple[str, ...],
    first_page_index: int,
) -> tuple[int, int] | None:
    available_count = len(expected_page_keys) - first_page_index
    key_count = min(len(runtime_keys), available_count)
    if key_count <= 0:
        return None
    if len(runtime_keys) <= available_count:
        expected_runtime_keys = expected_page_keys[first_page_index : first_page_index + len(runtime_keys)]
        if runtime_keys != expected_runtime_keys:
            return None
        return first_page_index, len(runtime_keys)
    expected_runtime_keys = expected_page_keys[first_page_index:]
    if runtime_keys[:key_count] != expected_runtime_keys:
        return None
    return first_page_index, key_count


def _log_sglang_handoff_operation(
    operation: str,
    *,
    key_count: int,
    request_context: DocumentKVHiCacheRequestContext | None,
    hydrated_count: int,
    host_indices_present: bool | None = None,
) -> None:
    if request_context is None:
        return
    _LOGGER.info(
        "Cachet SGLang HiCache handoff operation: operation=%s key_count=%d "
        "expected_page_keys=%d prefix_keys=%d last_hash_present=%s hydrated_count=%d "
        "has_handoff_json=%s has_inline_handoff=%s has_payload_uri=%s prompt_text_mode=%s "
        "host_indices_present=%s",
        operation,
        key_count,
        len(request_context.sglang_hicache_page_keys),
        len(request_context.prefix_keys),
        request_context.sglang_hicache_last_hash is not None,
        hydrated_count,
        request_context.handoff_json is not None,
        request_context.handoff_record is not None,
        request_context.payload_uri is not None,
        request_context.prompt_text_mode,
        host_indices_present,
    )


def _log_sglang_handoff_hydration(
    reason: str,
    *,
    keys: list[object],
    request_context: DocumentKVHiCacheRequestContext,
    binding_key_count: int = 0,
    hydrated_count: int = 0,
    full_page_count: int | None = None,
    level: int = logging.INFO,
) -> None:
    runtime_key_count = len(_runtime_hicache_keys(keys) or ())
    _LOGGER.log(
        level,
        "Cachet SGLang HiCache handoff hydration: reason=%s key_count=%d runtime_key_count=%d "
        "expected_page_keys=%d prefix_keys=%d last_hash_present=%s binding_key_count=%d "
        "hydrated_count=%d full_page_count=%s has_handoff_json=%s has_inline_handoff=%s "
        "has_payload_uri=%s prompt_text_mode=%s",
        reason,
        len(keys),
        runtime_key_count,
        len(request_context.sglang_hicache_page_keys),
        len(request_context.prefix_keys),
        request_context.sglang_hicache_last_hash is not None,
        binding_key_count,
        hydrated_count,
        full_page_count,
        request_context.handoff_json is not None,
        request_context.handoff_record is not None,
        request_context.payload_uri is not None,
        request_context.prompt_text_mode,
    )


def _runtime_hicache_keys(keys: list[object]) -> tuple[str, ...] | None:
    if not keys:
        return ()
    if not all(isinstance(key, str) and key for key in keys):
        return None
    return tuple(keys)


def _sglang_hicache_payload_pages(
    plan: EngineKVInjectionPlan,
    payload: bytes,
) -> tuple[bytes, ...]:
    page_tokens = plan.layout.block_size
    page_bytes = page_tokens * plan.layout.bytes_per_token
    full_page_count = _sglang_hicache_full_page_count(plan)
    return tuple(
        payload[index * page_bytes : (index + 1) * page_bytes]
        for index in range(full_page_count)
    )


def _sglang_hicache_full_page_count(plan: EngineKVInjectionPlan) -> int:
    return plan.total_tokens // plan.layout.block_size


def _provider_from_extra_config(
    extra_config: Mapping[str, Any],
    *,
    storage_identity: str | None = None,
) -> DocumentKVHiCacheProvider:
    factory_path = extra_config.get(DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY)
    if factory_path is None:
        return NoOpDocumentKVHiCacheProvider()
    if not isinstance(factory_path, str) or not factory_path.strip():
        raise ValueError(
            f"{DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY} "
            "must be a non-empty module:attribute string"
        )
    factory = load_document_kv_hicache_provider_factory(factory_path)
    provider = _call_provider_factory(factory, extra_config=extra_config, storage_identity=storage_identity)
    if isinstance(provider, NoOpDocumentKVHiCacheProvider):
        raise ValueError("configured document KV HiCache provider factory cannot return NoOpDocumentKVHiCacheProvider")
    _validate_provider(provider)
    return provider


_PAGE_FILE_SUFFIX = ".cachet-hicache-page"


def _is_memory_page_store_uri(store_uri: str | None) -> bool:
    if store_uri is None:
        return True
    normalized = store_uri.strip()
    return normalized in {"memory", "mem"} or normalized.startswith(("memory:", "mem:"))


def _page_store_root(store_uri: str | None) -> Path | None:
    if store_uri is None:
        return None
    from document_kv_cache.storage import local_path

    return local_path(store_uri)


def _page_namespace_prefix(storage_identity: str | None) -> str:
    raw_identity = storage_identity or "default"
    return hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()[:16]


def _page_key(key: object, storage_identity: str | None) -> str:
    key_hash = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return f"{_page_namespace_prefix(storage_identity)}-{key_hash}"


def _page_path(root: Path, key: object, storage_identity: str | None) -> Path:
    return root / f"{_page_key(key, storage_identity)}{_PAGE_FILE_SUFFIX}"


def _atomic_write_page(path: Path, raw_page: bytes) -> None:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            handle.write(raw_page)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _call_provider_factory(
    factory: object,
    *,
    extra_config: Mapping[str, Any],
    storage_identity: str | None,
) -> object:
    if not callable(factory):  # pragma: no cover - load helper validates this.
        raise TypeError("document KV HiCache provider factory must be callable")
    kwargs: dict[str, object] = {"extra_config": extra_config}
    if storage_identity is not None and _accepts_provider_factory_kwarg(factory, "storage_identity"):
        kwargs["storage_identity"] = storage_identity
    return factory(**kwargs)


def _accepts_provider_factory_kwarg(factory: object, name: str) -> bool:
    try:
        parameters = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return True
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ) and parameter.name == name:
            return True
    return False


def _storage_identity_from_runtime_args(*, args: tuple[object, ...], kwargs: Mapping[str, object]) -> str | None:
    for candidate in (*args, *kwargs.values()):
        identity = _storage_identity(candidate)
        if identity is not None:
            return identity
    return None


def _storage_identity(value: object) -> str | None:
    model_name = _first_attr(value, "model_name", "served_model_name", "model_path")
    tp_rank = _first_attr(value, "tp_rank")
    tp_size = _first_attr(value, "tp_size")
    pp_rank = _first_attr(value, "pp_rank")
    pp_size = _first_attr(value, "pp_size")
    cp_rank = _first_attr(value, "cp_rank", "attn_cp_rank", "context_parallel_rank")
    cp_size = _first_attr(value, "cp_size", "attn_cp_size", "context_parallel_size")
    is_mla_model = getattr(value, "is_mla_model", None)
    if all(item is None for item in (model_name, tp_rank, tp_size, pp_rank, pp_size, cp_rank, cp_size, is_mla_model)):
        return None
    model_segment = _safe_identity_segment(model_name or "model")
    segments = [model_segment]
    if is_mla_model is True:
        segments.append("mla")
    if tp_rank is not None or tp_size is not None or is_mla_model is not True:
        segments.extend(("tp", _identity_int(tp_rank, default=0), "of", _identity_int(tp_size, default=1)))
    if pp_rank is not None or _identity_int(pp_size, default=1) > 1:
        segments.extend(("pp", _identity_int(pp_rank, default=0), "of", _identity_int(pp_size, default=1)))
    if cp_rank is not None or _identity_int(cp_size, default=1) > 1:
        segments.extend(("cp", _identity_int(cp_rank, default=0), "of", _identity_int(cp_size, default=1)))
    return "_".join(str(segment) for segment in segments)


def _first_attr(value: object, *names: str) -> object | None:
    for name in names:
        attribute = getattr(value, name, None)
        if attribute is not None:
            return attribute
    return None


def _safe_identity_segment(value: object) -> str:
    raw_value = str(value)
    return "-".join(part for part in raw_value.replace("\\", "/").split("/") if part) or "model"


def _identity_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_config_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided")
    return value


def _request_extra_info_mapping(extra_info: object | None) -> Mapping[str, Any] | None:
    if extra_info is None:
        return None
    raw_extra_info = getattr(extra_info, "extra_info", None)
    if raw_extra_info is None and isinstance(extra_info, Mapping):
        raw_extra_info = extra_info.get("extra_info")
        if raw_extra_info is None and (
            _SGLANG_CUSTOM_PARAMS_KEY in extra_info or _SGLANG_KV_TRANSFER_PARAMS_KEY in extra_info
        ):
            raw_extra_info = extra_info
    if raw_extra_info is None:
        return None
    if not isinstance(raw_extra_info, Mapping):
        raise ValueError("SGLang HiCache extra_info.extra_info must be a mapping when provided")
    return dict(raw_extra_info)


def _kv_transfer_params_from_extra_info(extra_info: Mapping[str, Any]) -> Mapping[str, Any] | None:
    params = None
    custom_params = extra_info.get(_SGLANG_CUSTOM_PARAMS_KEY)
    if custom_params is not None:
        if not isinstance(custom_params, Mapping):
            raise ValueError("SGLang custom_params must be a mapping when provided")
        params = custom_params.get(_SGLANG_KV_TRANSFER_PARAMS_KEY)
    if params is None:
        params = extra_info.get(_SGLANG_KV_TRANSFER_PARAMS_KEY)
    if params is None:
        return None
    if not isinstance(params, Mapping):
        raise ValueError("SGLang custom_params.kv_transfer_params must be a mapping")
    return dict(params)


def _prefix_keys_from_extra_info(extra_info: object | None) -> tuple[str, ...]:
    if extra_info is None:
        return ()
    prefix_keys = getattr(extra_info, "prefix_keys", None)
    if prefix_keys is None and isinstance(extra_info, Mapping):
        prefix_keys = extra_info.get("prefix_keys")
    if prefix_keys is None:
        return ()
    if isinstance(prefix_keys, (str, bytes, bytearray)):
        return ()
    try:
        items = list(prefix_keys)
    except TypeError:
        return ()
    if not items or not all(isinstance(item, str) and item for item in items):
        return ()
    return tuple(items)


def _sglang_hicache_last_hash_from_extra_info(extra_info: Mapping[str, Any]) -> str | None:
    value = extra_info.get(DOCUMENT_KV_SGLANG_HICACHE_LAST_HASH_EXTRA_INFO_KEY)
    if not isinstance(value, str) or not value:
        return None
    return value


def _sglang_hicache_page_keys_from_params(params: Mapping[str, Any]) -> tuple[str, ...]:
    value = params.get(DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM)
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM} must be a sequence of strings")
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(f"{DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM} must be a sequence of strings") from exc
    if not all(isinstance(item, str) and item for item in items):
        raise ValueError(
            f"{DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM} entries must be non-empty strings"
        )
    return tuple(items)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError("document KV request context string fields must be non-empty strings when provided")


def _optional_mapping(value: object) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("document KV request context handoff_record must be a mapping when provided")


def _page_bytes(value: object | None) -> bytes:
    if value is None:
        raise ValueError("SGLang HiCache page payload must be provided")
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    tensor_bytes = _torch_tensor_bytes(value)
    if tensor_bytes is not None:
        return tensor_bytes
    try:
        return memoryview(value).tobytes()
    except TypeError as exc:
        raise TypeError("SGLang HiCache page payload must be bytes-like or a torch.Tensor") from exc


def _torch_tensor_bytes(value: object) -> bytes | None:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return None
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.detach().contiguous()
    byte_tensor = tensor.view(torch.uint8).reshape(-1)
    if byte_tensor.device.type != "cpu":
        byte_tensor = byte_tensor.cpu()
    return byte_tensor.numpy().tobytes()


def _copy_page_to_target(raw_page: bytes, target_location: object | None) -> object:
    if target_location is None:
        return raw_page
    if _copy_page_to_torch_target(raw_page, target_location):
        return target_location
    try:
        target_view = memoryview(target_location).cast("B")
    except TypeError as exc:
        raise TypeError("SGLang HiCache target_location must be a mutable byte buffer or torch.Tensor") from exc
    if target_view.readonly:
        raise TypeError("SGLang HiCache target_location must be mutable")
    if len(target_view) != len(raw_page):
        raise ValueError(
            "SGLang HiCache target_location byte length must match stored page byte length"
        )
    target_view[:] = raw_page
    return target_location


def _copy_page_to_torch_target(raw_page: bytes, target_location: object) -> bool:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return False
    if not isinstance(target_location, torch.Tensor):
        return False
    if not target_location.is_contiguous():
        raise ValueError("SGLang HiCache target_location tensor must be contiguous")
    target_bytes = target_location.view(torch.uint8).reshape(-1)
    if int(target_bytes.numel()) != len(raw_page):
        raise ValueError(
            "SGLang HiCache target_location byte length must match stored page byte length"
        )
    source = torch.frombuffer(bytearray(raw_page), dtype=torch.uint8)
    if target_bytes.device != source.device:
        source = source.to(device=target_bytes.device)
    target_bytes.copy_(source)
    return True


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
    storage_extra_config = getattr(value, "extra_config", None)
    if storage_extra_config is not None:
        _merge_extra_config(target, storage_extra_config)
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


def _call_provider_method(
    handler: object,
    *args: object,
    optional_kwargs: Mapping[str, object | None],
) -> object:
    if not callable(handler):  # pragma: no cover - callers guard this.
        raise TypeError("provider handler must be callable")
    return handler(*args, **_accepted_provider_kwargs(handler, optional_kwargs))


def _accepted_provider_kwargs(
    handler: object,
    optional_kwargs: Mapping[str, object | None],
) -> dict[str, object]:
    candidates = {key: value for key, value in optional_kwargs.items() if value is not None}
    if not candidates:
        return {}
    try:
        parameters = inspect.signature(handler).parameters.values()
    except (TypeError, ValueError):
        return candidates
    parameter_names = set()
    accepts_kwargs = False
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
            break
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            parameter_names.add(parameter.name)
    if accepts_kwargs:
        return candidates
    return {key: value for key, value in candidates.items() if key in parameter_names}


def _optional_sequence(value: Iterable[object] | object | None, length: int) -> list[object | None]:
    if value is None:
        return [None] * length
    if isinstance(value, (str, bytes, bytearray)):
        return [value] * length
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError:
        return [value] * length
    if len(items) == length:
        return items
    if length == 0:
        return []
    return [value] * length


def _looks_like_host_indices(value: object) -> bool:
    return callable(getattr(value, "numel", None)) or hasattr(value, "shape")


def _host_page_offset(host_indices: object, page_index: int, host_pool: object) -> int | None:
    page_size = int(getattr(host_pool, "page_size", 1) or 1)
    try:
        raw_index = host_indices[page_index * page_size]  # type: ignore[index]
        item = getattr(raw_index, "item", None)
        return int(item() if callable(item) else raw_index)
    except (IndexError, OverflowError, TypeError, ValueError):
        return None


def _host_indices_cover_pages(host_indices: object, page_count: int, host_pool: object) -> bool:
    page_size = int(getattr(host_pool, "page_size", 1) or 1)
    expected = page_count * page_size
    numel = getattr(host_indices, "numel", None)
    if callable(numel):
        return int(numel()) == expected
    try:
        return len(host_indices) == expected  # type: ignore[arg-type]
    except TypeError:
        return False


def _host_indices_cover_at_least_pages(host_indices: object, page_count: int, host_pool: object) -> bool:
    page_size = int(getattr(host_pool, "page_size", 1) or 1)
    expected = page_count * page_size
    numel = getattr(host_indices, "numel", None)
    if callable(numel):
        return int(numel()) >= expected
    try:
        return len(host_indices) >= expected  # type: ignore[arg-type]
    except TypeError:
        return False


def _load_host_page(
    backend: DocumentKVHiCacheBackend,
    key: object,
    host_pool: object,
    page_offset: int | None,
) -> bool:
    if page_offset is None:
        return False
    get_dummy_flat_data_page = getattr(host_pool, "get_dummy_flat_data_page", None)
    set_from_flat_data_page = getattr(host_pool, "set_from_flat_data_page", None)
    if not callable(get_dummy_flat_data_page) or not callable(set_from_flat_data_page):
        return False
    data_page = backend.get(key, target_location=get_dummy_flat_data_page())
    if data_page is None:
        return False
    set_from_flat_data_page(page_offset, data_page)
    return True


def _store_host_page(
    backend: DocumentKVHiCacheBackend,
    key: object,
    host_pool: object,
    page_offset: int | None,
) -> bool:
    if page_offset is None:
        return False
    get_data_page = getattr(host_pool, "get_data_page", None)
    if not callable(get_data_page):
        return False
    return _set_result_ok(backend, backend.set(key, get_data_page(page_offset, flat=True)))


def _store_provider_host_page(
    provider: DocumentKVHiCachePageProvider,
    key: object,
    host_pool: object,
    page_offset: int | None,
) -> bool:
    if page_offset is None:
        return False
    get_data_page = getattr(host_pool, "get_data_page", None)
    if not callable(get_data_page):
        return False
    return provider.set(key, get_data_page(page_offset, flat=True)) is not False


def _batch_pool_transfer(
    backend: DocumentKVHiCacheBackend,
    transfers: Iterable[object],
    *,
    operation: str,
) -> dict[str, list[bool]]:
    results: dict[str, list[bool]] = {}
    registered_pools = getattr(backend, "registered_pools", {})
    for transfer in transfers:
        name = _transfer_name(transfer)
        keys = list(getattr(transfer, "keys", None) or ())
        host_indices = getattr(transfer, "host_indices", None)
        host_pool = registered_pools.get(getattr(transfer, "name", name), registered_pools.get(name))
        if (
            host_pool is None
            or host_indices is None
            or not _host_indices_cover_pages(host_indices, len(keys), host_pool)
        ):
            results[name] = [False] * len(keys)
            continue
        transfer_results = []
        for index, key in enumerate(keys):
            storage_key = _component_key(key, name)
            page_offset = _host_page_offset(host_indices, index, host_pool)
            if operation == "get":
                transfer_results.append(_load_host_page(backend, storage_key, host_pool, page_offset))
            else:
                transfer_results.append(_store_host_page(backend, storage_key, host_pool, page_offset))
        results[name] = transfer_results
    return results


def _extra_pool_hit_pages(
    backend: DocumentKVHiCacheBackend,
    keys: list[object],
    transfers: Iterable[object],
    kv_hit_pages: int,
) -> dict[str, int]:
    results: dict[str, int] = {}
    for transfer in transfers:
        name = _transfer_name(transfer)
        if _transfer_hit_policy(transfer) == "trailing_pages":
            results[name] = _trailing_pool_hit_pages(backend, keys, name, transfer, kv_hit_pages)
        else:
            results[name] = _all_pages_pool_hit_pages(backend, keys, name, kv_hit_pages)
    return results


def _all_pages_pool_hit_pages(
    backend: DocumentKVHiCacheBackend,
    keys: list[object],
    name: str,
    kv_hit_pages: int,
) -> int:
    for index in range(kv_hit_pages):
        if not backend.exists(_component_key(keys[index], name)):
            return index
    return kv_hit_pages


def _trailing_pool_hit_pages(
    backend: DocumentKVHiCacheBackend,
    keys: list[object],
    name: str,
    transfer: object,
    kv_hit_pages: int,
) -> int:
    trailing = max(1, len(getattr(transfer, "keys", None) or ()) or 1)
    for prefix_len in range(kv_hit_pages, 0, -1):
        start = max(0, prefix_len - trailing)
        if all(backend.exists(_component_key(keys[index], name)) for index in range(start, prefix_len)):
            return prefix_len
    return 0


def _pool_transfer_result(kv_hit_pages: int, extra_pool_hit_pages: Mapping[str, int]) -> object:
    try:
        module = importlib.import_module("sglang.srt.mem_cache.hicache_storage")
        result_cls = getattr(module, "PoolTransferResult")
    except Exception:
        return {
            "kv_hit_pages": kv_hit_pages,
            "extra_pool_hit_pages": dict(extra_pool_hit_pages),
        }
    return result_cls(kv_hit_pages=kv_hit_pages, extra_pool_hit_pages=dict(extra_pool_hit_pages))


def _set_result_ok(backend: DocumentKVHiCacheBackend, result: object) -> bool:
    if isinstance(backend.provider, NoOpDocumentKVHiCacheProvider):
        return False
    return result is not False


def _component_key(key: object, name: str) -> object:
    if name in {"", "kv", "__default__", "PoolName.KV"}:
        return key
    return f"{key}.{name}"


def _transfer_name(transfer: object) -> str:
    name = getattr(transfer, "name", "")
    value = getattr(name, "value", name)
    return str(value)


def _transfer_hit_policy(transfer: object) -> str:
    policy = getattr(transfer, "hit_policy", "")
    value = getattr(policy, "value", policy)
    return str(value)
