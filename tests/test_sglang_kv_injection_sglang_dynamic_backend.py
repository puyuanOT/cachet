from __future__ import annotations

from types import ModuleType
import json
import sys

import pytest

import sglang_kv_injection
from sglang_kv_injection.sglang_dynamic_backend import (
    DOCUMENT_KV_HICACHE_BACKEND_CLASS,
    DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY,
    DocumentKVHiCacheBackend,
    NoOpDocumentKVHiCacheProvider,
    load_document_kv_hicache_provider_factory,
)


class RecordingProvider:
    def __init__(self) -> None:
        self.values: dict[object, object] = {}
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def get(self, key: object) -> object | None:
        self.calls.append(("get", (key,)))
        return self.values.get(key)

    def exists(self, key: object) -> bool:
        self.calls.append(("exists", (key,)))
        return key in self.values

    def set(self, key: object, value: object) -> None:
        self.calls.append(("set", (key, value)))
        self.values[key] = value


def test_document_kv_hicache_backend_defaults_to_safe_miss_provider():
    backend = DocumentKVHiCacheBackend()

    assert isinstance(backend.provider, NoOpDocumentKVHiCacheProvider)
    assert backend.exists("missing") is False
    assert backend.get("missing") is None
    assert backend.set("key", b"value") is None
    assert backend.batch_get_v1(["a", "b"]) == [None, None]
    assert backend.batch_set_v1(["a"], [b"value"]) is None
    assert backend.get("a") is None


def test_document_kv_hicache_backend_loads_provider_factory_from_mapping(monkeypatch):
    module = ModuleType("sglang_document_kv_provider")
    provider = RecordingProvider()

    def build_provider(*, extra_config):
        assert extra_config["tenant"] == "qa"
        return provider

    module.build_provider = build_provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = DocumentKVHiCacheBackend(
        {
            "tenant": "qa",
            DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider",
        }
    )

    assert backend.provider is provider
    assert backend.exists("doc-1") is False
    assert backend.set("doc-1", b"payload") is None
    assert backend.exists("doc-1") is True
    assert backend.get("doc-1") == b"payload"
    assert [call[0] for call in provider.calls] == ["exists", "set", "exists", "get"]


def test_document_kv_hicache_backend_loads_provider_factory_from_json(monkeypatch):
    module = ModuleType("sglang_document_kv_json_provider")
    provider = RecordingProvider()
    module.build_provider = lambda *, extra_config: provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = DocumentKVHiCacheBackend(
        extra_config=json.dumps(
            {
                DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider",
            }
        )
    )

    backend.batch_set_v1(["a", "b"], [b"1", b"2"])

    assert backend.batch_get_v1(["a", "b", "c"]) == [b"1", b"2", None]


def test_document_kv_hicache_backend_accepts_exist_provider_alias(monkeypatch):
    class ExistAliasProvider:
        def get(self, key):
            return b"value" if key == "doc-1" else None

        def exist(self, key):
            return key == "doc-1"

        def set(self, key, value):
            return None

    module = ModuleType("sglang_document_kv_exist_alias_provider")
    module.build_provider = lambda *, extra_config: ExistAliasProvider()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = DocumentKVHiCacheBackend(
        {
            DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider",
        }
    )

    assert backend.exists("doc-1") is True
    assert backend.exist("doc-1") is True
    assert backend.exists("missing") is False
    assert backend.get("doc-1") == b"value"


def test_load_document_kv_hicache_provider_factory_requires_module_attribute(monkeypatch):
    module = ModuleType("sglang_document_kv_bad_provider")
    module.not_callable = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(ValueError, match="module:attribute"):
        load_document_kv_hicache_provider_factory("no_colon")
    with pytest.raises(TypeError, match="not callable"):
        load_document_kv_hicache_provider_factory(f"{module.__name__}:not_callable")


def test_document_kv_hicache_backend_rejects_invalid_provider(monkeypatch):
    module = ModuleType("sglang_document_kv_invalid_provider")
    module.build_provider = lambda *, extra_config: object()
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(TypeError, match="get"):
        DocumentKVHiCacheBackend(
            {
                DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider",
            }
        )


def test_document_kv_hicache_backend_rejects_noop_provider_factory(monkeypatch):
    module = ModuleType("sglang_document_kv_noop_provider")
    module.build_provider = lambda *, extra_config: NoOpDocumentKVHiCacheProvider()
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(ValueError, match="NoOpDocumentKVHiCacheProvider"):
        DocumentKVHiCacheBackend(
            {
                DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider",
            }
        )


def test_document_kv_hicache_backend_identity_is_exported_from_package_root():
    assert DOCUMENT_KV_HICACHE_BACKEND_CLASS == "DocumentKVHiCacheBackend"
    assert DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH == "sglang_kv_injection.sglang_dynamic_backend"
    assert sglang_kv_injection.DocumentKVHiCacheBackend is DocumentKVHiCacheBackend
    assert sglang_kv_injection.DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH == DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH
