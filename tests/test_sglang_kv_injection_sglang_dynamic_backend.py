from __future__ import annotations

import importlib.util
from types import ModuleType
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import sglang_kv_injection
import sglang_kv_injection.sglang_dynamic_backend as sglang_dynamic_backend
from sglang_kv_injection.sglang_dynamic_backend import (
    DOCUMENT_KV_HICACHE_BACKEND_CLASS,
    DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY,
    DOCUMENT_KV_HICACHE_RUNTIME_METHODS,
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


class FakeHostPool:
    page_size = 2

    def __init__(self) -> None:
        self.pages: dict[int, object] = {}
        self.loaded: list[tuple[int, object]] = []

    def get_data_page(self, index: int, flat: bool = True) -> object:
        return self.pages[index]

    def get_dummy_flat_data_page(self) -> bytearray:
        return bytearray(b"empty")

    def set_from_flat_data_page(self, index: int, data_page: object) -> None:
        self.loaded.append((index, data_page))


def test_document_kv_hicache_backend_defaults_to_safe_miss_provider():
    backend = DocumentKVHiCacheBackend()

    assert isinstance(backend.provider, NoOpDocumentKVHiCacheProvider)
    assert backend.exists("missing") is False
    assert backend.get("missing") is None
    assert backend.set("key", b"value") is None
    assert backend.batch_get_v1(["a", "b"]) == [None, None]
    assert backend.batch_set_v1(["a"], [b"value"]) is None
    assert backend.get("a") is None
    assert backend.batch_exists(["a", "b"]) == 0
    assert backend.batch_get(["a", "b"]) == [None, None]
    assert backend.batch_set(["a"], [b"value"]) is False
    assert backend.batch_get_v2([]) == {}
    assert backend.batch_set_v2([]) == {}
    assert backend.clear() is None
    assert backend.get_stats() is None


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


def test_document_kv_hicache_backend_loads_provider_factory_from_sglang_storage_config(monkeypatch):
    module = ModuleType("sglang_document_kv_storage_config_provider")
    provider = RecordingProvider()

    def build_provider(*, extra_config):
        assert extra_config["tenant"] == "runtime"
        return provider

    module.build_provider = build_provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    storage_config = SimpleNamespace(
        extra_config={
            "tenant": "runtime",
            DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider",
        }
    )

    backend = DocumentKVHiCacheBackend(storage_config, {})

    assert backend.provider is provider
    assert backend.extra_config["tenant"] == "runtime"


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


def test_document_kv_hicache_backend_exposes_sglang_runtime_storage_methods():
    assert all(
        callable(getattr(DocumentKVHiCacheBackend, method_name, None))
        for method_name in DOCUMENT_KV_HICACHE_RUNTIME_METHODS
    )


def test_document_kv_hicache_backend_subclasses_installed_sglang_hicache_base(monkeypatch):
    sglang_module = ModuleType("sglang")
    srt_module = ModuleType("sglang.srt")
    mem_cache_module = ModuleType("sglang.srt.mem_cache")
    hicache_storage_module = ModuleType("sglang.srt.mem_cache.hicache_storage")

    class HiCacheStorage:
        pass

    hicache_storage_module.HiCacheStorage = HiCacheStorage
    for module in (sglang_module, srt_module, mem_cache_module, hicache_storage_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    module_path = Path(sglang_dynamic_backend.__file__)
    spec = importlib.util.spec_from_file_location("sglang_dynamic_backend_subclass_probe", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    assert issubclass(module.DocumentKVHiCacheBackend, HiCacheStorage)


def test_document_kv_hicache_backend_transfers_v1_pages_through_registered_host_pool():
    provider = RecordingProvider()
    backend = DocumentKVHiCacheBackend(provider=provider)
    host_pool = FakeHostPool()
    host_pool.pages[0] = b"page-a"
    host_pool.pages[2] = b"page-b"
    backend.register_mem_pool_host(host_pool)

    assert backend.batch_set_v1(["a", "b"], [0, 1, 2, 3]) == [True, True]
    assert provider.values == {"a": b"page-a", "b": b"page-b"}
    assert backend.batch_get_v1(["a", "missing"], [10, 11, 12, 13]) == [True, False]
    assert host_pool.loaded == [(10, b"page-a")]


def test_document_kv_hicache_backend_transfers_v2_named_pools():
    provider = RecordingProvider()
    provider.values["doc.aux"] = b"aux-page"
    backend = DocumentKVHiCacheBackend(provider=provider)
    host_pool = FakeHostPool()
    host_pool.pages[4] = b"updated-aux"
    backend.register_mem_host_pool_v2(host_pool, "aux")
    transfer = SimpleNamespace(name="aux", keys=["doc"], host_indices=[4, 5])

    assert backend.batch_get_v2([transfer]) == {"aux": [True]}
    assert host_pool.loaded == [(4, b"aux-page")]
    assert backend.batch_set_v2([transfer]) == {"aux": [True]}
    assert provider.values["doc.aux"] == b"updated-aux"


def test_document_kv_hicache_backend_batch_exists_v2_respects_trailing_pool_policy():
    provider = RecordingProvider()
    provider.values.update(
        {
            "k1": b"kv-1",
            "k2": b"kv-2",
            "k3": b"kv-3",
            "k3.swa": b"tail-sidecar",
        }
    )
    backend = DocumentKVHiCacheBackend(provider=provider)
    trailing_transfer = SimpleNamespace(
        name="swa",
        keys=["tail"],
        hit_policy=SimpleNamespace(value="trailing_pages"),
    )

    result = backend.batch_exists_v2(["k1", "k2", "k3"], [trailing_transfer])

    assert result == {
        "kv_hit_pages": 3,
        "extra_pool_hit_pages": {"swa": 3},
    }


def test_document_kv_hicache_backend_batch_exists_v2_shrinks_all_pages_pool_prefix():
    provider = RecordingProvider()
    provider.values.update(
        {
            "k1": b"kv-1",
            "k2": b"kv-2",
            "k3": b"kv-3",
            "k1.aux": b"sidecar-1",
        }
    )
    backend = DocumentKVHiCacheBackend(provider=provider)
    all_pages_transfer = SimpleNamespace(name="aux")

    result = backend.batch_exists_v2(["k1", "k2", "k3"], [all_pages_transfer])

    assert result == {
        "kv_hit_pages": 1,
        "extra_pool_hit_pages": {"aux": 1},
    }


def test_document_kv_hicache_backend_v2_pool_transfer_rejects_short_host_indices():
    provider = RecordingProvider()
    provider.values["doc.aux"] = b"aux-page"
    backend = DocumentKVHiCacheBackend(provider=provider)
    host_pool = FakeHostPool()
    host_pool.pages[4] = b"updated-aux"
    backend.register_mem_host_pool_v2(host_pool, "aux")
    transfer = SimpleNamespace(name="aux", keys=["doc"], host_indices=[4])

    assert backend.batch_get_v2([transfer]) == {"aux": [False]}
    assert backend.batch_set_v2([transfer]) == {"aux": [False]}
    assert host_pool.loaded == []
    assert provider.values["doc.aux"] == b"aux-page"


@pytest.mark.parametrize(
    "malformed_indices",
    (
        ["bad", "also-bad"],
        [float("inf"), 0],
    ),
)
def test_document_kv_hicache_backend_rejects_malformed_host_indices_without_raising(malformed_indices):
    provider = RecordingProvider()
    provider.values["doc"] = b"kv-page"
    provider.values["doc.aux"] = b"aux-page"
    backend = DocumentKVHiCacheBackend(provider=provider)
    host_pool = FakeHostPool()
    host_pool.pages[0] = b"updated-kv"
    host_pool.pages[4] = b"updated-aux"
    backend.register_mem_pool_host(host_pool)
    backend.register_mem_host_pool_v2(host_pool, "aux")
    transfer = SimpleNamespace(name="aux", keys=["doc"], host_indices=malformed_indices)

    assert backend.batch_get_v1(["doc"], malformed_indices) == [False]
    assert backend.batch_set_v1(["doc"], malformed_indices) == [False]
    assert backend.batch_get_v2([transfer]) == {"aux": [False]}
    assert backend.batch_set_v2([transfer]) == {"aux": [False]}
    assert host_pool.loaded == []
    assert provider.values["doc"] == b"kv-page"
    assert provider.values["doc.aux"] == b"aux-page"


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
