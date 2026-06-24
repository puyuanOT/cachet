from __future__ import annotations

import argparse
import importlib
import json
import sys
from types import ModuleType

import pytest

import sglang_kv_injection.sglang_dynamic_backend as sglang_dynamic_backend
import sglang_kv_injection.sglang_runtime_preflight as sglang_runtime_preflight
from sglang_kv_injection.sglang_dynamic_backend import (
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY,
)
from sglang_kv_injection.sglang_hicache_config import sglang_hicache_launch_config
from sglang_kv_injection.sglang_request_metadata_bridge import (
    DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SCHEMA_VERSION,
    sglang_request_metadata_bridge_status_to_record,
)
from sglang_kv_injection.sglang_runtime_preflight import (
    DOCUMENT_KV_SGLANG_INSTALLED_HICACHE_CONTRACT_RECORD_TYPE,
    DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_RECORD_TYPE,
    DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
    SGLANG_DYNAMIC_BACKEND_FACTORY_RECORD_TYPE,
    SGLANG_HICACHE_DYNAMIC_RUNTIME,
    SGLANG_HICACHE_REQUIRED_BACKEND_METHODS,
    SGLANG_HICACHE_REQUIRED_CLI_OPTIONS,
    SGLANG_HICACHE_REQUIRED_SERVER_ARG_FIELDS,
    SGLANG_HICACHE_REQUIRED_STORAGE_BACKEND_FACTORY_METHODS,
    SGLangInstalledHiCacheContract,
    document_kv_sglang_runtime_preflight_record_issues,
    document_kv_sglang_runtime_preflight_to_record,
    installed_sglang_hicache_contract_record_issues,
    installed_sglang_hicache_contract_to_record,
    validate_document_kv_sglang_runtime_preflight_record,
    validate_installed_sglang_hicache_contract_record,
)


def matching_installed_contract() -> dict:
    return installed_sglang_hicache_contract_to_record(
        SGLangInstalledHiCacheContract(
            package_version="0.5.10.post1",
            importable=True,
            server_args_importable=True,
            storage_backend_factory_importable=True,
            hicache_storage_base_importable=True,
            server_arg_fields=SGLANG_HICACHE_REQUIRED_SERVER_ARG_FIELDS,
            cli_options=SGLANG_HICACHE_REQUIRED_CLI_OPTIONS,
            hicache_storage_backend_choices=("file", "dynamic"),
            hicache_storage_extra_info_fields=("prefix_keys", "extra_info"),
            storage_backend_factory_methods=SGLANG_HICACHE_REQUIRED_STORAGE_BACKEND_FACTORY_METHODS,
            document_kv_backend_importable=True,
            document_kv_backend_subclasses_hicache_storage=True,
            document_kv_backend_methods=SGLANG_HICACHE_REQUIRED_BACKEND_METHODS,
            request_custom_params_available=True,
            request_metadata_extra_info_bridge=False,
            request_metadata_bridge_sources=(),
        )
    )


def upstream_bridge_installed_contract() -> dict:
    return installed_sglang_hicache_contract_to_record(
        SGLangInstalledHiCacheContract(
            package_version="0.5.10.post1",
            importable=True,
            server_args_importable=True,
            storage_backend_factory_importable=True,
            hicache_storage_base_importable=True,
            server_arg_fields=SGLANG_HICACHE_REQUIRED_SERVER_ARG_FIELDS,
            cli_options=SGLANG_HICACHE_REQUIRED_CLI_OPTIONS,
            hicache_storage_backend_choices=("file", "dynamic"),
            hicache_storage_extra_info_fields=("prefix_keys", "extra_info"),
            storage_backend_factory_methods=SGLANG_HICACHE_REQUIRED_STORAGE_BACKEND_FACTORY_METHODS,
            document_kv_backend_importable=True,
            document_kv_backend_subclasses_hicache_storage=True,
            document_kv_backend_methods=SGLANG_HICACHE_REQUIRED_BACKEND_METHODS,
            request_custom_params_available=True,
            request_metadata_extra_info_bridge=True,
            request_metadata_bridge_sources=("sglang.srt.managers.cache_controller",),
        )
    )


def drifting_installed_contract() -> dict:
    return installed_sglang_hicache_contract_to_record(
        SGLangInstalledHiCacheContract(
            package_version="0.5.11",
            importable=True,
            server_args_importable=True,
            storage_backend_factory_importable=True,
            hicache_storage_base_importable=True,
            server_arg_fields=tuple(
                field_name
                for field_name in SGLANG_HICACHE_REQUIRED_SERVER_ARG_FIELDS
                if field_name != "hicache_storage_backend_extra_config"
            ),
            cli_options=tuple(
                option
                for option in SGLANG_HICACHE_REQUIRED_CLI_OPTIONS
                if option != "--hicache-storage-backend-extra-config"
            ),
            hicache_storage_backend_choices=("file",),
            hicache_storage_extra_info_fields=("prefix_keys",),
            storage_backend_factory_methods=SGLANG_HICACHE_REQUIRED_STORAGE_BACKEND_FACTORY_METHODS,
            document_kv_backend_importable=True,
            document_kv_backend_subclasses_hicache_storage=True,
            document_kv_backend_methods=tuple(
                method_name
                for method_name in SGLANG_HICACHE_REQUIRED_BACKEND_METHODS
                if method_name != "batch_exists_v2"
            ),
            request_custom_params_available=False,
            request_metadata_extra_info_bridge=False,
            request_metadata_bridge_sources=(),
        )
    )


def install_provider_module(monkeypatch: pytest.MonkeyPatch, module_name: str = "sglang_preflight_provider") -> str:
    module = ModuleType(module_name)

    class Provider:
        def get(self, key):
            return None

        def exists(self, key):
            return False

        def set(self, key, value):
            return None

    def build_provider(*, extra_config=None):
        return Provider()

    module.build_provider = build_provider
    monkeypatch.setitem(sys.modules, module_name, module)
    return f"{module_name}:build_provider"


def install_document_kv_backend_class(monkeypatch: pytest.MonkeyPatch, hicache_storage_cls: type) -> type:
    class DocumentKVHiCacheBackend(hicache_storage_cls):
        def __init__(self, storage_config=None, *args, **kwargs):
            from sglang_kv_injection.sglang_request_metadata_bridge import (
                install_sglang_request_metadata_bridge,
            )

            self.request_metadata_bridge_status = install_sglang_request_metadata_bridge()
            self.provider = None
            extra_config = getattr(storage_config, "extra_config", None)
            if not isinstance(extra_config, dict):
                return
            factory_path = extra_config.get(DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY)
            if not isinstance(factory_path, str):
                return
            module_name, _, attribute_name = factory_path.partition(":")
            provider_factory = getattr(importlib.import_module(module_name), attribute_name)
            self.provider = provider_factory(extra_config=extra_config)

    for method_name in SGLANG_HICACHE_REQUIRED_BACKEND_METHODS:
        setattr(DocumentKVHiCacheBackend, method_name, lambda self, *args, **kwargs: None)

    monkeypatch.setattr(
        sglang_dynamic_backend,
        "DocumentKVHiCacheBackend",
        DocumentKVHiCacheBackend,
    )
    return DocumentKVHiCacheBackend


def install_storage_backend_factory_module(monkeypatch: pytest.MonkeyPatch) -> None:
    sglang_module = ModuleType("sglang")
    sglang_module.__path__ = []
    srt_module = ModuleType("sglang.srt")
    srt_module.__path__ = []
    managers_module = ModuleType("sglang.srt.managers")
    managers_module.__path__ = []
    scheduler_module = ModuleType("sglang.srt.managers.scheduler")
    cache_controller_module = ModuleType("sglang.srt.managers.cache_controller")
    mem_cache_module = ModuleType("sglang.srt.mem_cache")
    mem_cache_module.__path__ = []
    hicache_storage_module = ModuleType("sglang.srt.mem_cache.hicache_storage")
    storage_module = ModuleType("sglang.srt.mem_cache.storage")
    storage_module.__path__ = []
    backend_factory_module = ModuleType("sglang.srt.mem_cache.storage.backend_factory")

    class HiCacheStorage:
        pass

    class Scheduler:
        def _prefetch_kvcache(self, req):
            return None

    class HiCacheStorageExtraInfo:
        def __init__(self, prefix_keys=None, extra_info=None):
            self.prefix_keys = prefix_keys
            self.extra_info = extra_info

    class PrefetchOperation:
        def __init__(
            self,
            request_id,
            host_indices,
            token_ids,
            last_hash=None,
            prefix_keys=None,
        ):
            self.request_id = request_id
            self.host_indices = host_indices
            self.token_ids = token_ids
            self.last_hash = last_hash
            self.prefix_keys = prefix_keys

    class HiCacheController:
        def prefetch(
            self,
            request_id,
            host_indices,
            new_input_tokens,
            last_hash=None,
            prefix_keys=None,
        ):
            operation = PrefetchOperation(
                request_id,
                host_indices,
                new_input_tokens,
                last_hash,
                prefix_keys,
            )
            self.prefetch_queue.put(operation)
            return operation

        def _storage_hit_query(self, operation):
            return [], 0

        def _page_transfer(self, operation):
            return None

        def _page_backup(self, operation):
            return None

    install_document_kv_backend_class(monkeypatch, HiCacheStorage)

    class StorageBackendFactory:
        loader_calls = []

        @staticmethod
        def _load_backend_class(module_path, class_name, backend_name):
            StorageBackendFactory.loader_calls.append((module_path, class_name, backend_name))
            module = importlib.import_module(module_path)
            backend_cls = getattr(module, class_name)
            if not issubclass(backend_cls, HiCacheStorage):
                raise TypeError(f"{module_path}:{class_name} must subclass HiCacheStorage")
            return backend_cls

        @classmethod
        def _create_dynamic_backend(cls, backend_config, storage_config, mem_pool_host, **kwargs):
            backend_cls = cls._load_backend_class(
                backend_config["module_path"],
                backend_config["class_name"],
                backend_config["backend_name"],
            )
            return backend_cls(storage_config, mem_pool_host, **kwargs)

        @classmethod
        def create_backend(cls, backend_name, storage_config, mem_pool_host, **kwargs):
            assert backend_name == "dynamic"
            return cls._create_dynamic_backend(storage_config.extra_config, storage_config, mem_pool_host, **kwargs)

    hicache_storage_module.HiCacheStorage = HiCacheStorage
    scheduler_module.Scheduler = Scheduler
    cache_controller_module.HiCacheStorageExtraInfo = HiCacheStorageExtraInfo
    cache_controller_module.PrefetchOperation = PrefetchOperation
    cache_controller_module.HiCacheController = HiCacheController
    backend_factory_module.StorageBackendFactory = StorageBackendFactory
    for module in (
        sglang_module,
        srt_module,
        managers_module,
        scheduler_module,
        cache_controller_module,
        mem_cache_module,
        hicache_storage_module,
        storage_module,
        backend_factory_module,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)


def ok_must_match_issue() -> str:
    return (
        "ok must match installed contract, launch config, provider factory, "
        "dynamic backend factory safety, and live metadata bridge readiness"
    )


def launch_config_without_provider_factory() -> dict:
    launch_config = sglang_hicache_launch_config()
    extra_config = json.loads(launch_config["hicache_storage_backend_extra_config"])
    extra_config.pop(DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY, None)
    launch_config["hicache_storage_backend_extra_config"] = json.dumps(
        extra_config,
        separators=(",", ":"),
        sort_keys=True,
    )
    return launch_config


def test_installed_sglang_hicache_contract_accepts_dynamic_runtime_surface():
    record = matching_installed_contract()

    assert record["record_type"] == DOCUMENT_KV_SGLANG_INSTALLED_HICACHE_CONTRACT_RECORD_TYPE
    assert record["schema_version"] == DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION
    assert record["runtime"] == SGLANG_HICACHE_DYNAMIC_RUNTIME
    assert record["package_version"] == "0.5.10.post1"
    assert "dynamic" in record["hicache_storage_backend_choices"]
    assert record["hicache_storage_extra_info_fields"] == ["extra_info", "prefix_keys"]
    assert record["document_kv_backend_importable"] is True
    assert record["document_kv_backend_subclasses_hicache_storage"] is True
    assert "batch_exists_v2" in record["document_kv_backend_methods"]
    assert record["request_custom_params_available"] is True
    assert record["request_metadata_extra_info_bridge"] is False
    assert record["request_metadata_bridge_sources"] == []
    assert record["live_request_metadata_bridge_ok"] is False
    assert record["ok"] is True
    validate_installed_sglang_hicache_contract_record(record)


def test_installed_sglang_hicache_contract_rejects_runtime_drift():
    record = drifting_installed_contract()

    assert record["ok"] is False
    issues = installed_sglang_hicache_contract_record_issues(record)

    assert any("hicache_storage_backend_extra_config" in issue for issue in issues)
    assert any("dynamic" in issue for issue in issues)
    assert any("hicache_storage_extra_info_fields" in issue and "extra_info" in issue for issue in issues)
    assert any("request_custom_params_available" in issue for issue in issues)
    assert any("batch_exists_v2" in issue for issue in issues)
    with pytest.raises(ValueError, match="hicache_storage_backend_extra_config"):
        validate_installed_sglang_hicache_contract_record(record)


def test_sglang_request_metadata_bridge_source_detection():
    assert (
        sglang_runtime_preflight._source_contains_request_metadata_bridge(
            "extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)"
        )
        is False
    )
    assert (
        sglang_runtime_preflight._source_contains_request_metadata_bridge(
            "custom_params = request.sampling_params.custom_params\n"
            "extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys, extra_info={})"
        )
        is False
    )
    assert (
        sglang_runtime_preflight._source_contains_request_metadata_bridge(
            "extra_info = hicache_storage.HiCacheStorageExtraInfo("
            "prefix_keys=prefix_keys, extra_info=extra_info)"
        )
        is False
    )
    assert (
        sglang_runtime_preflight._source_contains_request_metadata_bridge(
            "extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys, "
            "extra_info=operation.extra_info)"
        )
        is True
    )
    assert (
        sglang_runtime_preflight._source_contains_request_metadata_bridge(
            "extra_info = HiCacheStorageExtraInfo(extra_info={'custom_params': custom_params})"
        )
        is True
    )


def test_installed_sglang_hicache_contract_can_inspect_fake_installed_modules(monkeypatch):
    sglang_module = ModuleType("sglang")
    srt_module = ModuleType("sglang.srt")
    server_args_module = ModuleType("sglang.srt.server_args")
    mem_cache_module = ModuleType("sglang.srt.mem_cache")
    storage_module = ModuleType("sglang.srt.mem_cache.storage")
    backend_factory_module = ModuleType("sglang.srt.mem_cache.storage.backend_factory")
    hicache_storage_module = ModuleType("sglang.srt.mem_cache.hicache_storage")

    class ServerArgs:
        enable_hierarchical_cache = False
        hicache_io_backend = "kernel"
        hicache_mem_layout = "layer_first"
        hicache_storage_backend = None
        hicache_storage_backend_extra_config = None
        hicache_storage_prefetch_policy = "best_effort"
        hicache_write_policy = "write_through"

        @staticmethod
        def add_cli_args(parser: argparse.ArgumentParser):
            parser.add_argument("--enable-hierarchical-cache", action="store_true")
            parser.add_argument("--hicache-io-backend")
            parser.add_argument("--hicache-mem-layout")
            parser.add_argument("--hicache-storage-backend", choices=["file", "dynamic"])
            parser.add_argument("--hicache-storage-backend-extra-config")
            parser.add_argument("--hicache-storage-prefetch-policy")
            parser.add_argument("--hicache-write-policy")

    class StorageBackendFactory:
        @staticmethod
        def _load_backend_class(module_path, class_name, backend_name):
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        @classmethod
        def _create_dynamic_backend(cls, backend_config, storage_config, mem_pool_host, **kwargs):
            return cls._load_backend_class(
                backend_config["module_path"],
                backend_config["class_name"],
                backend_config["backend_name"],
            )(storage_config)

        @classmethod
        def create_backend(cls, backend_name, storage_config, mem_pool_host, **kwargs):
            return cls._create_dynamic_backend(storage_config.extra_config, storage_config, mem_pool_host, **kwargs)

    class HiCacheStorage:
        pass

    class HiCacheStorageExtraInfo:
        __annotations__ = {"prefix_keys": list, "extra_info": dict}

    install_document_kv_backend_class(monkeypatch, HiCacheStorage)
    server_args_module.ServerArgs = ServerArgs
    backend_factory_module.StorageBackendFactory = StorageBackendFactory
    hicache_storage_module.HiCacheStorage = HiCacheStorage
    hicache_storage_module.HiCacheStorageExtraInfo = HiCacheStorageExtraInfo
    for module in (
        sglang_module,
        srt_module,
        server_args_module,
        mem_cache_module,
        storage_module,
        backend_factory_module,
        hicache_storage_module,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sglang_runtime_preflight.importlib_metadata, "version", lambda name: "0.5.10.post1")
    monkeypatch.setattr(sglang_runtime_preflight, "_request_custom_params_available", lambda: True)
    monkeypatch.setattr(sglang_runtime_preflight, "_request_metadata_bridge_sources", lambda: ())

    record = sglang_runtime_preflight.installed_sglang_hicache_contract_to_record()

    assert record["ok"] is True
    assert record["package_version"] == "0.5.10.post1"
    assert record["hicache_storage_extra_info_fields"] == ["extra_info", "prefix_keys"]
    assert record["document_kv_backend_subclasses_hicache_storage"] is True
    assert record["live_request_metadata_bridge_ok"] is False
    sglang_runtime_preflight.validate_installed_sglang_hicache_contract_record(record)


def test_installed_sglang_hicache_contract_rejects_document_backend_that_is_not_hicache_storage():
    record = matching_installed_contract()
    record["document_kv_backend_subclasses_hicache_storage"] = False
    record["ok"] = False

    issues = installed_sglang_hicache_contract_record_issues(record)

    assert "installed SGLang HiCache contract document_kv_backend_subclasses_hicache_storage must be true" in issues
    assert "installed SGLang HiCache contract ok must be true for a safe runtime preflight" in issues


def test_installed_sglang_hicache_contract_rejects_missing_document_backend_methods():
    record = matching_installed_contract()
    record["document_kv_backend_methods"] = [
        method_name for method_name in SGLANG_HICACHE_REQUIRED_BACKEND_METHODS if method_name != "batch_get_v2"
    ]
    record["ok"] = False

    issues = installed_sglang_hicache_contract_record_issues(record)

    assert any("document_kv_backend_methods" in issue and "batch_get_v2" in issue for issue in issues)
    assert "installed SGLang HiCache contract ok must be true for a safe runtime preflight" in issues


def test_sglang_runtime_preflight_accepts_dynamic_hicache_config_and_provider(monkeypatch):
    provider_factory = install_provider_module(monkeypatch)
    install_storage_backend_factory_module(monkeypatch)
    record = document_kv_sglang_runtime_preflight_to_record(
        sglang_hicache_launch_config(provider_factory=provider_factory),
        installed_contract=matching_installed_contract(),
    )

    assert record["record_type"] == DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_RECORD_TYPE
    assert record["schema_version"] == DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION
    assert record["runtime"] == SGLANG_HICACHE_DYNAMIC_RUNTIME
    assert record["installed_contract"]["ok"] is True
    assert record["launch_config"]["ok"] is True
    assert record["provider_factory"]["path"] == provider_factory
    assert record["provider_factory"]["provider_constructed"] is True
    assert record["provider_factory"]["provider_methods"] == ["get", "set", "exists"]
    assert record["provider_factory"]["ok"] is True
    assert record["dynamic_backend_factory"]["record_type"] == SGLANG_DYNAMIC_BACKEND_FACTORY_RECORD_TYPE
    assert record["dynamic_backend_factory"]["provider_factory"] == provider_factory
    assert record["dynamic_backend_factory"]["backend_constructed"] is True
    assert record["dynamic_backend_factory"]["backend_class"] == "DocumentKVHiCacheBackend"
    assert record["dynamic_backend_factory"]["backend_provider_class"] == "Provider"
    assert record["dynamic_backend_factory"]["ok"] is True
    assert record["dynamic_backend_factory"]["request_metadata_bridge"]["ok"] is True
    assert record["request_metadata_bridge"]["ok"] is True
    assert record["live_request_metadata_bridge_ok"] is True
    factory_cls = sys.modules["sglang.srt.mem_cache.storage.backend_factory"].StorageBackendFactory
    assert factory_cls.loader_calls == [
        (
            "sglang_kv_injection.sglang_dynamic_backend",
            "DocumentKVHiCacheBackend",
            "document_kv",
        )
    ]
    assert record["ok"] is True
    validate_document_kv_sglang_runtime_preflight_record(record)


def test_sglang_runtime_preflight_accepts_upstream_bridge_without_cachet_patch(monkeypatch):
    provider_factory = install_provider_module(monkeypatch)
    install_storage_backend_factory_module(monkeypatch)
    record = document_kv_sglang_runtime_preflight_to_record(
        sglang_hicache_launch_config(provider_factory=provider_factory),
        installed_contract=upstream_bridge_installed_contract(),
    )
    failed_cachet_bridge = sglang_request_metadata_bridge_status_to_record()
    record["dynamic_backend_factory"]["request_metadata_bridge"] = failed_cachet_bridge
    record["request_metadata_bridge"] = failed_cachet_bridge
    record["live_request_metadata_bridge_ok"] = True
    record["ok"] = True

    assert record["installed_contract"]["live_request_metadata_bridge_ok"] is True
    assert record["request_metadata_bridge"]["ok"] is False
    assert document_kv_sglang_runtime_preflight_record_issues(record) == ()
    validate_document_kv_sglang_runtime_preflight_record(record)


def test_sglang_request_metadata_bridge_record_rejects_stale_v1_schema(monkeypatch):
    provider_factory = install_provider_module(monkeypatch)
    install_storage_backend_factory_module(monkeypatch)
    record = document_kv_sglang_runtime_preflight_to_record(
        sglang_hicache_launch_config(provider_factory=provider_factory),
        installed_contract=matching_installed_contract(),
    )
    stale_bridge = dict(record["request_metadata_bridge"])
    assert stale_bridge["schema_version"] == DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SCHEMA_VERSION
    stale_bridge["schema_version"] = DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SCHEMA_VERSION - 1
    stale_bridge.pop("prefetch_operation_patched")
    stale_bridge["ok"] = True
    record["request_metadata_bridge"] = stale_bridge
    record["dynamic_backend_factory"]["request_metadata_bridge"] = stale_bridge
    record["live_request_metadata_bridge_ok"] = True
    record["ok"] = True

    issues = document_kv_sglang_runtime_preflight_record_issues(record)

    assert any("schema_version" in issue for issue in issues)
    assert any("prefetch_operation_patched" in issue for issue in issues)


@pytest.mark.parametrize(
    ("field", "value", "expected_issue"),
    [
        ("module_path", "evil.module", "module_path"),
        ("class_name", "EvilBackend", "class_name"),
        ("document_kv_record_type", "sglang_kv_injection.evil.v1", "document_kv.record_type"),
        ("document_kv_schema_version", 999, "document_kv.schema_version"),
        ("document_kv_connector_package", "evil", "document_kv.connector_package"),
        ("document_kv_kv_injection_method", "evil-method", "document_kv.kv_injection_method"),
        ("document_kv_engine_handoff_record_type", "evil.handoff.v1", "document_kv.engine_handoff_record_type"),
        ("document_kv_engine_handoff_schema_version", 999, "document_kv.engine_handoff_schema_version"),
    ],
)
def test_sglang_runtime_preflight_rejects_mutated_launch_subrecord(
    monkeypatch,
    field,
    value,
    expected_issue,
):
    provider_factory = install_provider_module(monkeypatch)
    install_storage_backend_factory_module(monkeypatch)
    record = document_kv_sglang_runtime_preflight_to_record(
        sglang_hicache_launch_config(provider_factory=provider_factory),
        installed_contract=matching_installed_contract(),
    )
    record["launch_config"][field] = value

    issues = document_kv_sglang_runtime_preflight_record_issues(record)

    assert any(expected_issue in issue for issue in issues)
    assert ok_must_match_issue() in issues


def test_sglang_runtime_preflight_rejects_launch_provider_mismatch(monkeypatch):
    provider_factory = install_provider_module(monkeypatch)
    install_storage_backend_factory_module(monkeypatch)
    record = document_kv_sglang_runtime_preflight_to_record(
        sglang_hicache_launch_config(provider_factory=provider_factory),
        installed_contract=matching_installed_contract(),
    )
    record["launch_config"]["provider_factory"] = (
        "sglang_kv_injection.sglang_dynamic_backend:NoOpDocumentKVHiCacheProvider"
    )

    issues = document_kv_sglang_runtime_preflight_record_issues(record)

    assert "provider_factory.path must match launch_config.provider_factory" in issues
    assert "dynamic_backend_factory.provider_factory must match launch_config.provider_factory" in issues
    assert ok_must_match_issue() in issues


def test_sglang_runtime_preflight_rejects_mutated_dynamic_backend_factory_subrecord(monkeypatch):
    provider_factory = install_provider_module(monkeypatch)
    install_storage_backend_factory_module(monkeypatch)
    record = document_kv_sglang_runtime_preflight_to_record(
        sglang_hicache_launch_config(provider_factory=provider_factory),
        installed_contract=matching_installed_contract(),
    )
    record["dynamic_backend_factory"]["backend_constructed"] = False

    issues = document_kv_sglang_runtime_preflight_record_issues(record)

    assert (
        "dynamic_backend_factory.SGLang dynamic backend factory must construct the backend through create_backend"
        in issues
    )
    assert ok_must_match_issue() in issues


def test_sglang_runtime_preflight_rejects_missing_provider_factory_and_runtime_drift():
    record = document_kv_sglang_runtime_preflight_to_record(
        launch_config_without_provider_factory(),
        installed_contract=drifting_installed_contract(),
    )

    assert record["installed_contract"]["ok"] is False
    assert record["launch_config"]["ok"] is False
    assert record["provider_factory"]["ok"] is False
    assert record["ok"] is False
    issues = document_kv_sglang_runtime_preflight_record_issues(record)

    assert any("installed_contract" in issue and "hicache_storage_backend_extra_config" in issue for issue in issues)
    assert any(DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY in issue for issue in issues)
    assert "ok must be true for a safe SGLang runtime preflight" in issues
    with pytest.raises(ValueError, match="provider_factory"):
        validate_document_kv_sglang_runtime_preflight_record(record)


def test_sglang_runtime_preflight_rejects_noop_provider_factory():
    record = document_kv_sglang_runtime_preflight_to_record(
        sglang_hicache_launch_config(
            provider_factory="sglang_kv_injection.sglang_dynamic_backend:NoOpDocumentKVHiCacheProvider"
        ),
        installed_contract=matching_installed_contract(),
    )

    assert record["provider_factory"]["known_noop"] is True
    assert record["provider_factory"]["ok"] is False
    assert record["ok"] is False
    issues = document_kv_sglang_runtime_preflight_record_issues(record)
    assert any("NoOpDocumentKVHiCacheProvider" in issue for issue in issues)


def test_sglang_runtime_preflight_rejects_provider_factory_returning_noop(monkeypatch):
    module = ModuleType("sglang_preflight_noop_provider")
    from sglang_kv_injection.sglang_dynamic_backend import NoOpDocumentKVHiCacheProvider

    def build_provider(*, extra_config=None):
        return NoOpDocumentKVHiCacheProvider()

    module.build_provider = build_provider
    monkeypatch.setitem(sys.modules, module.__name__, module)

    record = document_kv_sglang_runtime_preflight_to_record(
        sglang_hicache_launch_config(provider_factory=f"{module.__name__}:build_provider"),
        installed_contract=matching_installed_contract(),
    )

    assert record["provider_factory"]["returns_known_noop"] is True
    assert record["provider_factory"]["ok"] is False
    issues = document_kv_sglang_runtime_preflight_record_issues(record)
    assert any("cannot return NoOpDocumentKVHiCacheProvider" in issue for issue in issues)


def test_sglang_runtime_preflight_rejects_provider_factory_returning_incomplete_provider(monkeypatch):
    module = ModuleType("sglang_preflight_incomplete_provider")

    class IncompleteProvider:
        def get(self, key):
            return None

    def build_provider(*, extra_config=None):
        return IncompleteProvider()

    module.build_provider = build_provider
    monkeypatch.setitem(sys.modules, module.__name__, module)

    record = document_kv_sglang_runtime_preflight_to_record(
        sglang_hicache_launch_config(provider_factory=f"{module.__name__}:build_provider"),
        installed_contract=matching_installed_contract(),
    )

    assert record["provider_factory"]["provider_constructed"] is True
    assert record["provider_factory"]["provider_method_issues"]
    assert record["provider_factory"]["ok"] is False
    issues = document_kv_sglang_runtime_preflight_record_issues(record)
    assert any("provider_method_issues" in issue and "set" in issue and "exists" in issue for issue in issues)


def test_sglang_runtime_preflight_rejects_mismatched_ok_flag(monkeypatch):
    provider_factory = install_provider_module(monkeypatch)
    install_storage_backend_factory_module(monkeypatch)
    record = document_kv_sglang_runtime_preflight_to_record(
        sglang_hicache_launch_config(provider_factory=provider_factory),
        installed_contract=matching_installed_contract(),
    )
    record["provider_factory"]["importable"] = False

    issues = document_kv_sglang_runtime_preflight_record_issues(record)

    assert "provider_factory.SGLang provider factory importable must be true" in issues
    assert ok_must_match_issue() in issues


def test_sglang_runtime_preflight_cli_writes_strict_record(tmp_path, monkeypatch):
    provider_factory = install_provider_module(monkeypatch)
    install_storage_backend_factory_module(monkeypatch)
    monkeypatch.setattr(
        sglang_runtime_preflight,
        "installed_sglang_hicache_contract_to_record",
        lambda contract=None: matching_installed_contract(),
    )
    output_path = tmp_path / "sglang-preflight.json"

    exit_code = sglang_runtime_preflight.main(
        [
            "--provider-factory",
            provider_factory,
            "--output-json",
            str(output_path),
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["ok"] is True
    validate_document_kv_sglang_runtime_preflight_record(record)


def test_sglang_runtime_preflight_cli_accepts_launch_config_json_file(tmp_path, monkeypatch):
    provider_factory = install_provider_module(monkeypatch)
    install_storage_backend_factory_module(monkeypatch)
    monkeypatch.setattr(
        sglang_runtime_preflight,
        "installed_sglang_hicache_contract_to_record",
        lambda contract=None: matching_installed_contract(),
    )
    launch_config_path = tmp_path / "sglang-launch.json"
    launch_config_path.write_text(
        json.dumps(sglang_hicache_launch_config(provider_factory=provider_factory)),
        encoding="utf-8",
    )
    output_path = tmp_path / "sglang-preflight.json"

    exit_code = sglang_runtime_preflight.main(
        [
            "--launch-config-json",
            str(launch_config_path),
            "--output-json",
            str(output_path),
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["provider_factory"]["path"] == provider_factory
    validate_document_kv_sglang_runtime_preflight_record(record)


def test_sglang_runtime_preflight_cli_uses_builtin_provider_factory_by_default(tmp_path, monkeypatch):
    install_storage_backend_factory_module(monkeypatch)
    monkeypatch.setattr(
        sglang_runtime_preflight,
        "installed_sglang_hicache_contract_to_record",
        lambda contract=None: matching_installed_contract(),
    )
    output_path = tmp_path / "sglang-preflight.json"

    exit_code = sglang_runtime_preflight.main(["--output-json", str(output_path)])

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["provider_factory"]["path"] == DOCUMENT_KV_HICACHE_PROVIDER_FACTORY
    assert record["provider_factory"]["provider_class"] == "DocumentKVHiCachePageProvider"
    assert record["provider_factory"]["ok"] is True
    assert record["ok"] is True
    validate_document_kv_sglang_runtime_preflight_record(record)


def test_sglang_runtime_preflight_is_exposed_through_document_and_cachet_facades():
    import cachet.sglang_runtime_preflight as cachet_preflight
    import document_kv_cache.sglang_runtime_preflight as document_preflight
    import sglang_kv_injection

    assert cachet_preflight is document_preflight
    assert (
        document_preflight.document_kv_sglang_runtime_preflight_to_record
        is sglang_runtime_preflight.document_kv_sglang_runtime_preflight_to_record
    )
    assert (
        sglang_kv_injection.document_kv_sglang_runtime_preflight_to_record
        is sglang_runtime_preflight.document_kv_sglang_runtime_preflight_to_record
    )
