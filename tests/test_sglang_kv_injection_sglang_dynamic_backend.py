from __future__ import annotations

import importlib.util
from types import ModuleType
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from document_kv_cache.benchmarks import (
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
)
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import (
    build_engine_adapter_request,
    sglang_adapter_spec,
    vllm_adapter_spec,
    write_engine_adapter_request_json,
)
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment
import sglang_kv_injection
import sglang_kv_injection.sglang_dynamic_backend as sglang_dynamic_backend
from sglang_kv_injection.sglang_dynamic_backend import (
    DOCUMENT_KV_HICACHE_BACKEND_CLASS,
    DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH,
    DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY,
    DOCUMENT_KV_HICACHE_RUNTIME_METHODS,
    DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM,
    DocumentKVHiCacheBackend,
    DocumentKVHiCachePageProvider,
    DocumentKVHiCacheRequestContext,
    NoOpDocumentKVHiCacheProvider,
    build_document_kv_hicache_provider,
    document_kv_request_context_from_extra_info,
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


def sglang_ready_request(*, request_id: str = "cachet-live-sglang-handoff") -> EngineReadyRequest:
    layout = KVLayout(
        model_id="tiny-sglang-model",
        lora_id="base",
        layout_version="tiny-v1",
        dtype="int8",
        num_layers=1,
        block_size=2,
        bytes_per_token=4,
    )
    return EngineReadyRequest(
        handle=KVCacheHandle(
            request_id=request_id,
            handle_uri=f"document-kv://{request_id}",
            layout=layout,
            segments=(
                KVSegment(
                    "doc-a",
                    "document_static",
                    "static",
                    0,
                    4,
                    0,
                    16,
                ),
            ),
            total_tokens=4,
            total_bytes=16,
        ),
        payload=b"page-onepage-two",
        estimated_gpu_bytes=16,
    )


def write_sglang_handoff(tmp_path, ready: EngineReadyRequest) -> tuple[Path, Path]:
    payload_path = tmp_path / f"{ready.request_id}.kv"
    payload_path.write_bytes(ready.payload if isinstance(ready.payload, bytes) else b"".join(ready.payload))
    handoff_path = write_engine_adapter_request_json(
        build_engine_adapter_request(ready, spec=sglang_adapter_spec()),
        tmp_path / f"{ready.request_id}.handoff.json",
        payload_uri=f"disk:{payload_path}",
    )
    return handoff_path, payload_path


class RequestContextProvider(RecordingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.context_calls: list[tuple[str, DocumentKVHiCacheRequestContext | None, object | None]] = []

    def batch_exists(
        self,
        keys,
        *,
        extra_info=None,
        document_kv_request_context=None,
    ):
        self.context_calls.append(("batch_exists", document_kv_request_context, extra_info))
        return len(keys)

    def batch_get_v1(
        self,
        keys,
        *,
        host_indices=None,
        extra_info=None,
        document_kv_request_context=None,
    ):
        self.context_calls.append(("batch_get_v1", document_kv_request_context, extra_info))
        return [True] * len(keys)

    def batch_set_v1(
        self,
        keys,
        *,
        host_indices=None,
        extra_info=None,
        document_kv_request_context=None,
    ):
        self.context_calls.append(("batch_set_v1", document_kv_request_context, extra_info))
        return [True] * len(keys)


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


def test_document_kv_hicache_page_provider_round_trips_memory_pages():
    provider = DocumentKVHiCachePageProvider()
    target = bytearray(b"\x00" * 5)

    assert provider.set("doc-1", bytearray(b"page1")) is True
    assert provider.exists("doc-1") is True
    assert provider.get("doc-1", target_location=target) is target
    assert target == bytearray(b"page1")
    assert provider.delete("doc-1") is True
    assert provider.exists("doc-1") is False
    assert provider.delete("doc-1") is False
    assert provider.set("doc-1", bytearray(b"page1")) is True
    assert provider.get("missing") is None
    assert provider.get_stats() == {
        "provider": "DocumentKVHiCachePageProvider",
        "store": "memory",
        "store_uri": None,
        "storage_identity": None,
        "pages": 1,
        "hits": 1,
        "misses": 1,
        "sets": 2,
    }


def test_document_kv_hicache_page_provider_round_trips_disk_pages(tmp_path):
    provider = build_document_kv_hicache_provider(
        extra_config={DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY: f"disk:{tmp_path}"}
    )

    assert provider.set("doc-1", b"disk-page") is True
    assert provider.exists("doc-1") is True
    assert provider.get("doc-1") == b"disk-page"
    assert len(list(tmp_path.glob("*.cachet-hicache-page"))) == 1
    assert provider.delete("doc-1") is True
    assert provider.exists("doc-1") is False
    assert list(tmp_path.glob("*.cachet-hicache-page")) == []
    assert provider.set("doc-1", b"disk-page") is True

    provider.clear()

    assert provider.exists("doc-1") is False
    assert list(tmp_path.glob("*.cachet-hicache-page")) == []


def test_document_kv_hicache_page_provider_namespaces_disk_pages_by_runtime_identity(tmp_path):
    extra_config = {
        DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY: DOCUMENT_KV_HICACHE_PROVIDER_FACTORY,
        DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY: f"disk:{tmp_path}",
    }
    rank0_config = SimpleNamespace(
        extra_config=extra_config,
        model_path="Qwen/Qwen3",
        tp_rank=0,
        tp_size=2,
        pp_rank=0,
        pp_size=2,
        attn_cp_rank=0,
        attn_cp_size=2,
        is_mla_model=False,
    )
    rank1_config = SimpleNamespace(
        extra_config=extra_config,
        model_path="Qwen/Qwen3",
        tp_rank=1,
        tp_size=2,
        pp_rank=1,
        pp_size=2,
        attn_cp_rank=1,
        attn_cp_size=2,
        is_mla_model=False,
    )
    backend0 = DocumentKVHiCacheBackend(rank0_config, {})
    backend1 = DocumentKVHiCacheBackend(rank1_config, {})

    assert backend0.storage_identity == "Qwen-Qwen3_tp_0_of_2_pp_0_of_2_cp_0_of_2"
    assert backend1.storage_identity == "Qwen-Qwen3_tp_1_of_2_pp_1_of_2_cp_1_of_2"
    assert backend0.set("shared-page", b"rank0") is True
    assert backend1.set("shared-page", b"rank1") is True
    assert backend0.get("shared-page") == b"rank0"
    assert backend1.get("shared-page") == b"rank1"
    assert len(list(tmp_path.glob("*.cachet-hicache-page"))) == 2


def test_document_kv_hicache_page_provider_returns_false_when_atomic_disk_write_fails(tmp_path, monkeypatch):
    provider = build_document_kv_hicache_provider(
        extra_config={DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY: f"disk:{tmp_path}"},
        storage_identity="rank0",
    )

    def fail_replace(source, destination):
        assert Path(source).is_file()
        raise OSError("replace failed")

    monkeypatch.setattr(sglang_dynamic_backend.os, "replace", fail_replace)

    assert provider.set("doc-1", b"payload") is False
    assert provider.exists("doc-1") is False
    assert provider.get("doc-1") is None
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob("*.cachet-hicache-page")) == []


def test_document_kv_hicache_backend_builtin_provider_transfers_v1_host_pages():
    backend = DocumentKVHiCacheBackend(provider=DocumentKVHiCachePageProvider())
    host_pool = FakeHostPool()
    host_pool.pages[0] = b"page1"
    host_pool.pages[2] = b"page2"
    backend.register_mem_pool_host(host_pool)

    assert backend.batch_set_v1(["a", "b"], [0, 1, 2, 3]) == [True, True]
    assert backend.batch_get_v1(["a", "missing"], [10, 11, 12, 13]) == [True, False]
    assert host_pool.loaded == [(10, bytearray(b"page1"))]


def test_document_kv_hicache_builtin_provider_factory_path_constructs_non_noop_provider():
    factory = load_document_kv_hicache_provider_factory(DOCUMENT_KV_HICACHE_PROVIDER_FACTORY)
    provider = factory(extra_config={})

    assert isinstance(provider, DocumentKVHiCachePageProvider)
    assert not isinstance(provider, NoOpDocumentKVHiCacheProvider)


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
    seen_storage_identities = []

    def build_provider(*, extra_config, storage_identity):
        assert extra_config["tenant"] == "runtime"
        seen_storage_identities.append(storage_identity)
        return provider

    module.build_provider = build_provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    storage_config = SimpleNamespace(
        extra_config={
            "tenant": "runtime",
            DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider",
        },
        model_name="runtime-model",
        tp_rank=1,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
    )

    backend = DocumentKVHiCacheBackend(storage_config, {})

    assert backend.provider is provider
    assert backend.extra_config["tenant"] == "runtime"
    assert seen_storage_identities == ["runtime-model_tp_1_of_2_pp_0_of_1"]


def test_document_kv_hicache_backend_keeps_legacy_provider_factory_signature(monkeypatch):
    module = ModuleType("sglang_document_kv_legacy_provider")
    provider = RecordingProvider()

    def build_provider(*, extra_config):
        assert extra_config["tenant"] == "legacy"
        return provider

    module.build_provider = build_provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    storage_config = SimpleNamespace(
        extra_config={
            "tenant": "legacy",
            DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider",
        },
        model_name="legacy-model",
        tp_rank=0,
        tp_size=1,
    )

    backend = DocumentKVHiCacheBackend(storage_config, {})

    assert backend.provider is provider
    assert backend.storage_identity == "legacy-model_tp_0_of_1"


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


def test_document_kv_hicache_request_context_parses_sglang_custom_params():
    extra_info_payload = {
        "custom_params": {
            "kv_transfer_params": {
                DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-live-sglang-1",
                DOCUMENT_KV_HANDOFF_JSON_PARAM: "/Volumes/cachet/live/sglang.handoff.json",
                DOCUMENT_KV_PAYLOAD_URI_PARAM: "uc-volume:/catalog/schema/volume/live/sglang.kv",
                DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM: "runtime",
                DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: ["page-a", "page-b"],
            },
        },
    }
    sglang_extra_info = SimpleNamespace(
        prefix_keys=["prefix-a", "prefix-b"],
        extra_info=extra_info_payload,
    )

    context = document_kv_request_context_from_extra_info(sglang_extra_info)

    assert context == DocumentKVHiCacheRequestContext(
        kv_transfer_params=extra_info_payload["custom_params"]["kv_transfer_params"],
        request_id="cachet-live-sglang-1",
        handoff_json="/Volumes/cachet/live/sglang.handoff.json",
        payload_uri="uc-volume:/catalog/schema/volume/live/sglang.kv",
        prompt_text_mode="runtime",
        sglang_hicache_page_keys=("page-a", "page-b"),
        prefix_keys=("prefix-a", "prefix-b"),
        raw_extra_info=extra_info_payload,
    )


def test_document_kv_hicache_request_context_accepts_direct_kv_transfer_params():
    context = document_kv_request_context_from_extra_info(
        {
            "prefix_keys": ["prefix-a"],
            "extra_info": {
                "kv_transfer_params": {
                    DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-live-sglang-2",
                },
            },
        }
    )

    assert context is not None
    assert context.request_id == "cachet-live-sglang-2"
    assert context.prefix_keys == ("prefix-a",)


def test_document_kv_hicache_request_context_rejects_malformed_kv_transfer_params():
    with pytest.raises(ValueError, match="kv_transfer_params must be a mapping"):
        document_kv_request_context_from_extra_info(
            {
                "custom_params": {
                    "kv_transfer_params": ["not", "a", "mapping"],
                },
            }
        )


def test_document_kv_hicache_request_context_rejects_malformed_sglang_page_keys():
    with pytest.raises(ValueError, match=r"sglang_hicache_page_keys must be a sequence"):
        document_kv_request_context_from_extra_info(
            {
                "custom_params": {
                    "kv_transfer_params": {
                        DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-live-sglang-keys",
                        DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: "not-a-sequence",
                    },
                },
            }
        )


def test_document_kv_hicache_backend_forwards_request_context_to_batch_provider_methods():
    provider = RequestContextProvider()
    backend = DocumentKVHiCacheBackend(provider=provider)
    sglang_extra_info = SimpleNamespace(
        prefix_keys=["prefix-a"],
        extra_info={
            "custom_params": {
                "kv_transfer_params": {
                    DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-live-sglang-3",
                    DOCUMENT_KV_HANDOFF_JSON_PARAM: "/Volumes/cachet/live/sglang.handoff.json",
                },
            },
        },
    )

    assert backend.batch_exists(["page-a", "page-b"], extra_info=sglang_extra_info) == 2
    assert backend.batch_get_v1(["page-a"], [0], extra_info=sglang_extra_info) == [True]
    assert backend.batch_set_v1(["page-a"], [0], extra_info=sglang_extra_info) == [True]

    assert [name for name, _context, _extra_info in provider.context_calls] == [
        "batch_exists",
        "batch_get_v1",
        "batch_set_v1",
    ]
    for _name, context, observed_extra_info in provider.context_calls:
        assert observed_extra_info is sglang_extra_info
        assert context is not None
        assert context.request_id == "cachet-live-sglang-3"
        assert context.handoff_json == "/Volumes/cachet/live/sglang.handoff.json"
        assert context.prefix_keys == ("prefix-a",)


def test_document_kv_hicache_page_provider_hydrates_sglang_handoff_pages(tmp_path):
    ready = sglang_ready_request()
    handoff_path, payload_path = write_sglang_handoff(tmp_path, ready)
    provider = DocumentKVHiCachePageProvider()
    context = DocumentKVHiCacheRequestContext(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: ready.request_id,
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
            DOCUMENT_KV_PAYLOAD_URI_PARAM: f"disk:{payload_path}",
            DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: [
                "sglang-hash-page-0",
                "sglang-hash-page-1",
            ],
        },
        request_id=ready.request_id,
        handoff_json=str(handoff_path),
        payload_uri=f"disk:{payload_path}",
        sglang_hicache_page_keys=("sglang-hash-page-0", "sglang-hash-page-1"),
    )

    assert (
        provider.batch_exists(
            ["sglang-hash-page-0", "sglang-hash-page-1"],
            document_kv_request_context=context,
        )
        == 2
    )

    assert provider.get("sglang-hash-page-0") == b"page-one"
    assert provider.get("sglang-hash-page-1") == b"page-two"


def test_document_kv_hicache_page_provider_returns_cached_prefix_before_logical_suffix(tmp_path):
    ready = sglang_ready_request()
    handoff_path, payload_path = write_sglang_handoff(tmp_path, ready)
    provider = DocumentKVHiCachePageProvider()
    context = DocumentKVHiCacheRequestContext(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: ready.request_id,
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
            DOCUMENT_KV_PAYLOAD_URI_PARAM: f"disk:{payload_path}",
            DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: [
                "sglang-hash-page-0",
                "sglang-hash-page-1",
            ],
        },
        request_id=ready.request_id,
        handoff_json=str(handoff_path),
        payload_uri=f"disk:{payload_path}",
        sglang_hicache_page_keys=("sglang-hash-page-0", "sglang-hash-page-1"),
    )

    keys = ["sglang-hash-page-0", "sglang-hash-page-1", "runtime-query-page"]

    assert provider.batch_exists(keys, document_kv_request_context=context) == 2
    assert provider.batch_get_v1(keys, document_kv_request_context=context) == [
        b"page-one",
        b"page-two",
        None,
    ]
    assert provider.get("runtime-query-page") is None


def test_document_kv_hicache_page_provider_loads_cached_prefix_with_full_host_indices(tmp_path):
    ready = sglang_ready_request()
    handoff_path, payload_path = write_sglang_handoff(tmp_path, ready)
    provider = DocumentKVHiCachePageProvider()

    class SGLangHandoffHostPool(FakeHostPool):
        def get_dummy_flat_data_page(self) -> bytearray:
            return bytearray(8)

    host_pool = SGLangHandoffHostPool()
    provider.register_mem_pool_host(host_pool)
    context = DocumentKVHiCacheRequestContext(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: ready.request_id,
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
            DOCUMENT_KV_PAYLOAD_URI_PARAM: f"disk:{payload_path}",
            DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: [
                "sglang-hash-page-0",
                "sglang-hash-page-1",
            ],
        },
        request_id=ready.request_id,
        handoff_json=str(handoff_path),
        payload_uri=f"disk:{payload_path}",
        sglang_hicache_page_keys=("sglang-hash-page-0", "sglang-hash-page-1"),
    )

    keys = ["sglang-hash-page-0", "sglang-hash-page-1", "runtime-query-page"]

    assert (
        provider.batch_get_v1(
            keys,
            host_indices=[10, 11, 12, 13, 14, 15],
            document_kv_request_context=context,
        )
        == [True, True, False]
    )
    assert host_pool.loaded == [
        (10, bytearray(b"page-one")),
        (12, bytearray(b"page-two")),
    ]


def test_document_kv_hicache_page_provider_rejects_mismatch_before_cached_prefix_end(tmp_path):
    ready = sglang_ready_request()
    handoff_path, payload_path = write_sglang_handoff(tmp_path, ready)
    provider = DocumentKVHiCachePageProvider()
    context = DocumentKVHiCacheRequestContext(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: ready.request_id,
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
            DOCUMENT_KV_PAYLOAD_URI_PARAM: f"disk:{payload_path}",
            DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: [
                "sglang-hash-page-0",
                "sglang-hash-page-1",
            ],
        },
        request_id=ready.request_id,
        handoff_json=str(handoff_path),
        payload_uri=f"disk:{payload_path}",
        sglang_hicache_page_keys=("sglang-hash-page-0", "sglang-hash-page-1"),
    )

    assert (
        provider.batch_exists(
            ["sglang-hash-page-0", "wrong-live-page-1"],
            document_kv_request_context=context,
        )
        == 0
    )
    assert provider.get("sglang-hash-page-0") is None


def test_document_kv_hicache_page_provider_hydrates_with_prefix_key_offset(tmp_path):
    ready = sglang_ready_request()
    handoff_path, payload_path = write_sglang_handoff(tmp_path, ready)
    provider = DocumentKVHiCachePageProvider()
    context = DocumentKVHiCacheRequestContext(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: ready.request_id,
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
            DOCUMENT_KV_PAYLOAD_URI_PARAM: f"disk:{payload_path}",
            DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: [
                "already-matched-page",
                "sglang-hash-page-1",
            ],
        },
        request_id=ready.request_id,
        handoff_json=str(handoff_path),
        payload_uri=f"disk:{payload_path}",
        sglang_hicache_page_keys=("already-matched-page", "sglang-hash-page-1"),
        prefix_keys=("already-matched-page",),
    )

    assert provider.batch_exists(["sglang-hash-page-1"], document_kv_request_context=context) == 1

    assert provider.get("sglang-hash-page-1") == b"page-two"
    assert provider.get("sglang-hash-page-0") is None


def test_document_kv_hicache_page_provider_accepts_inline_handoff_record(tmp_path):
    ready = sglang_ready_request()
    handoff_path, payload_path = write_sglang_handoff(tmp_path, ready)
    handoff_record = json.loads(handoff_path.read_text(encoding="utf-8"))
    provider = DocumentKVHiCachePageProvider()
    context = DocumentKVHiCacheRequestContext(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: ready.request_id,
            DOCUMENT_KV_HANDOFF_RECORD_PARAM: handoff_record,
            DOCUMENT_KV_PAYLOAD_URI_PARAM: f"disk:{payload_path}",
            DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: ["sglang-hash-page-0", "sglang-hash-page-1"],
        },
        request_id=ready.request_id,
        handoff_record=handoff_record,
        payload_uri=f"disk:{payload_path}",
        sglang_hicache_page_keys=("sglang-hash-page-0", "sglang-hash-page-1"),
    )

    assert provider.batch_get_v1(["sglang-hash-page-0"], document_kv_request_context=context) == [b"page-one"]


def test_document_kv_hicache_page_provider_requires_expected_page_keys_for_handoff_hydration(tmp_path):
    ready = sglang_ready_request()
    handoff_path, payload_path = write_sglang_handoff(tmp_path, ready)
    provider = DocumentKVHiCachePageProvider()
    context = DocumentKVHiCacheRequestContext(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: ready.request_id,
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
            DOCUMENT_KV_PAYLOAD_URI_PARAM: f"disk:{payload_path}",
        },
        request_id=ready.request_id,
        handoff_json=str(handoff_path),
        payload_uri=f"disk:{payload_path}",
    )

    assert provider.batch_exists(["sglang-hash-page-0"], document_kv_request_context=context) == 0
    assert provider.batch_get_v1(["sglang-hash-page-0"], document_kv_request_context=context) == [None]
    assert provider.get("sglang-hash-page-0") is None


def test_document_kv_hicache_page_provider_rejects_mismatched_runtime_page_keys(tmp_path):
    ready = sglang_ready_request()
    handoff_path, payload_path = write_sglang_handoff(tmp_path, ready)
    provider = DocumentKVHiCachePageProvider()
    context = DocumentKVHiCacheRequestContext(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: ready.request_id,
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
            DOCUMENT_KV_PAYLOAD_URI_PARAM: f"disk:{payload_path}",
            DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: [
                "expected-page-0",
                "expected-page-1",
            ],
        },
        request_id=ready.request_id,
        handoff_json=str(handoff_path),
        payload_uri=f"disk:{payload_path}",
        sglang_hicache_page_keys=("expected-page-0", "expected-page-1"),
    )

    assert provider.batch_exists(["wrong-live-page-0"], document_kv_request_context=context) == 0
    assert provider.get("wrong-live-page-0") is None


def test_document_kv_hicache_page_provider_rejects_wrong_backend_handoff(tmp_path):
    ready = sglang_ready_request()
    payload_path = tmp_path / f"{ready.request_id}.kv"
    payload_path.write_bytes(ready.payload if isinstance(ready.payload, bytes) else b"".join(ready.payload))
    handoff_path = write_engine_adapter_request_json(
        build_engine_adapter_request(ready, spec=vllm_adapter_spec()),
        tmp_path / f"{ready.request_id}.handoff.json",
        payload_uri=f"disk:{payload_path}",
    )
    provider = DocumentKVHiCachePageProvider()
    context = DocumentKVHiCacheRequestContext(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: ready.request_id,
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
            DOCUMENT_KV_PAYLOAD_URI_PARAM: f"disk:{payload_path}",
            DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: ["sglang-hash-page-0", "sglang-hash-page-1"],
        },
        request_id=ready.request_id,
        handoff_json=str(handoff_path),
        payload_uri=f"disk:{payload_path}",
        sglang_hicache_page_keys=("sglang-hash-page-0", "sglang-hash-page-1"),
    )

    with pytest.raises(ValueError, match="does not match expected_backend"):
        provider.batch_exists(["sglang-hash-page-0"], document_kv_request_context=context)


def test_document_kv_hicache_page_provider_validates_handoff_before_warm_page_hits(tmp_path):
    ready = sglang_ready_request()
    payload_path = tmp_path / f"{ready.request_id}.kv"
    payload_path.write_bytes(ready.payload if isinstance(ready.payload, bytes) else b"".join(ready.payload))
    handoff_path = write_engine_adapter_request_json(
        build_engine_adapter_request(ready, spec=vllm_adapter_spec()),
        tmp_path / f"{ready.request_id}.handoff.json",
        payload_uri=f"disk:{payload_path}",
    )
    provider = DocumentKVHiCachePageProvider()
    provider.set("sglang-hash-page-0", b"stale-page")
    context = DocumentKVHiCacheRequestContext(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: ready.request_id,
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
            DOCUMENT_KV_PAYLOAD_URI_PARAM: f"disk:{payload_path}",
            DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: ["sglang-hash-page-0", "sglang-hash-page-1"],
        },
        request_id=ready.request_id,
        handoff_json=str(handoff_path),
        payload_uri=f"disk:{payload_path}",
        sglang_hicache_page_keys=("sglang-hash-page-0", "sglang-hash-page-1"),
    )

    with pytest.raises(ValueError, match="does not match expected_backend"):
        provider.batch_exists(["sglang-hash-page-0"], document_kv_request_context=context)


def test_document_kv_hicache_page_provider_rejects_request_id_mismatch(tmp_path):
    ready = sglang_ready_request()
    handoff_path, payload_path = write_sglang_handoff(tmp_path, ready)
    provider = DocumentKVHiCachePageProvider()
    context = DocumentKVHiCacheRequestContext(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: "wrong-request",
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
            DOCUMENT_KV_PAYLOAD_URI_PARAM: f"disk:{payload_path}",
            DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: ["sglang-hash-page-0", "sglang-hash-page-1"],
        },
        request_id="wrong-request",
        handoff_json=str(handoff_path),
        payload_uri=f"disk:{payload_path}",
        sglang_hicache_page_keys=("sglang-hash-page-0", "sglang-hash-page-1"),
    )

    with pytest.raises(ValueError, match="must match handoff request_id"):
        provider.batch_exists(["sglang-hash-page-0"], document_kv_request_context=context)


def test_document_kv_hicache_backend_omits_context_for_legacy_batch_provider_signatures():
    class LegacyBatchProvider(RecordingProvider):
        def batch_exists(self, keys, *, extra_info=None):
            self.calls.append(("batch_exists", (tuple(keys), extra_info)))
            return len(keys)

    provider = LegacyBatchProvider()
    backend = DocumentKVHiCacheBackend(provider=provider)
    sglang_extra_info = {
        "custom_params": {
            "kv_transfer_params": {
                DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-live-sglang-4",
            },
        },
    }

    assert backend.batch_exists(["page-a"], extra_info=sglang_extra_info) == 1
    assert provider.calls == [("batch_exists", (("page-a",), sglang_extra_info))]


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
