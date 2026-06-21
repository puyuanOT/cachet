from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import vllm_kv_injection
from vllm_kv_injection.vllm_dynamic_connector import (
    DOCUMENT_KV_CONNECTOR_CLASS,
    DOCUMENT_KV_CONNECTOR_MODULE_PATH,
    DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY,
    DocumentKVConnector,
    NoOpDocumentKVProvider,
    VLLMSupportsHMA,
    load_document_kv_provider_factory,
    vllm_runtime_import_error,
)
from vllm_kv_injection.vllm_runtime_contract import validate_vllm_kv_connector_v1_methods

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


class RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.calls.append(("get_num_new_matched_tokens", (request, num_computed_tokens), {}))
        return 7, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.calls.append(("update_state_after_alloc", (request, blocks, num_external_tokens), {}))

    def build_connector_meta(self, scheduler_output):
        self.calls.append(("build_connector_meta", (scheduler_output,), {}))
        return {"ready": True}

    def register_kv_caches(self, kv_caches):
        self.calls.append(("register_kv_caches", (kv_caches,), {}))

    def start_load_kv(self, forward_context, **kwargs):
        self.calls.append(("start_load_kv", (forward_context,), kwargs))

    def wait_for_layer_load(self, layer_name):
        self.calls.append(("wait_for_layer_load", (layer_name,), {}))

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        self.calls.append(("save_kv_layer", (layer_name, kv_layer, attn_metadata), kwargs))

    def wait_for_save(self):
        self.calls.append(("wait_for_save", (), {}))

    def request_finished(self, request, block_ids):
        self.calls.append(("request_finished", (request, block_ids), {}))
        return True, {"request_id": "req-1"}

    def request_finished_all_groups(self, request, block_ids):
        self.calls.append(("request_finished_all_groups", (request, block_ids), {}))
        return True, {"request_id": "req-1", "groups": len(block_ids)}

    def get_block_ids_with_load_errors(self):
        return {3}

    def get_kv_connector_stats(self):
        return {"loads": 1}

    def take_events(self):
        return ("loaded",)


def test_document_kv_connector_exposes_vllm_v1_lifecycle():
    provider = RecordingProvider()
    connector = DocumentKVConnector(provider=provider)

    validate_vllm_kv_connector_v1_methods(connector)

    assert connector.get_num_new_matched_tokens("request", 5) == (7, True)
    connector.update_state_after_alloc("request", "blocks", 7)
    assert connector.build_connector_meta("scheduler") == {"ready": True}
    connector.register_kv_caches({"layer.0": "cache"})
    connector.start_load_kv("forward", layer="layer.0")
    connector.wait_for_layer_load("layer.0")
    connector.save_kv_layer("layer.0", "kv", "metadata", reason="test")
    connector.wait_for_save()
    assert connector.request_finished("request", [1, 2]) == (True, {"request_id": "req-1"})
    assert connector.request_finished_all_groups("request", ([4, 5], [6])) == (
        True,
        {"request_id": "req-1", "groups": 2},
    )
    assert connector.get_block_ids_with_load_errors() == {3}
    assert connector.get_kv_connector_stats() == {"loads": 1}
    assert connector.take_events() == ["loaded"]
    assert [call[0] for call in provider.calls] == [
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
    ]


def test_document_kv_connector_defaults_to_no_external_tokens_without_provider():
    connector = DocumentKVConnector()

    validate_vllm_kv_connector_v1_methods(connector)

    assert isinstance(connector.provider, NoOpDocumentKVProvider)
    assert connector.get_num_new_matched_tokens("request", 0) == (0, False)
    assert connector.build_connector_meta("scheduler") == {}
    assert connector.request_finished("request", [1]) == (False, None)
    assert connector.request_finished_all_groups("request", ([1],)) == (False, None)
    assert connector.request_finished_all_groups("request", ([1], [2])) == (False, None)
    assert connector.get_block_ids_with_load_errors() == set()
    assert connector.get_kv_connector_stats() is None
    assert connector.take_events() == []


def test_document_kv_connector_loads_provider_factory_from_transfer_config(monkeypatch):
    module = ModuleType("document_kv_test_provider")
    provider = RecordingProvider()

    def build_provider(*, vllm_config, extra_config):
        assert extra_config["tenant"] == "qa"
        return provider

    module.build_provider = build_provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "tenant": "qa",
                DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider",
            }
        )
    )

    connector = DocumentKVConnector(vllm_config=vllm_config)

    assert connector.provider is provider
    assert connector.get_num_new_matched_tokens("request", 0) == (7, True)


def test_document_kv_connector_reads_cache_config_transfer_config(monkeypatch):
    module = ModuleType("document_kv_cache_config_provider")
    provider = RecordingProvider()
    module.build_provider = lambda *, vllm_config, extra_config: provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_transfer_config={
                "kv_connector_extra_config": {
                    DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider"
                }
            }
        )
    )

    assert DocumentKVConnector(vllm_config=vllm_config).provider is provider


def test_load_document_kv_provider_factory_requires_module_attribute(monkeypatch):
    module = ModuleType("document_kv_bad_provider")
    module.not_callable = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(ValueError, match="module:attribute"):
        load_document_kv_provider_factory("no_colon")
    with pytest.raises(TypeError, match="not callable"):
        load_document_kv_provider_factory(f"{module.__name__}:not_callable")


def test_document_kv_connector_rejects_invalid_provider(monkeypatch):
    module = ModuleType("document_kv_invalid_provider")
    module.build_provider = lambda *, vllm_config, extra_config: object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider"
            }
        )
    )

    with pytest.raises(TypeError, match="get_num_new_matched_tokens"):
        DocumentKVConnector(vllm_config=vllm_config)


def test_document_kv_connector_rejects_configured_noop_provider_factory(monkeypatch):
    module = ModuleType("document_kv_noop_provider_factory")
    module.build_provider = lambda *, vllm_config, extra_config: NoOpDocumentKVProvider()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider"
            }
        )
    )

    with pytest.raises(ValueError, match="NoOpDocumentKVProvider"):
        DocumentKVConnector(vllm_config=vllm_config)


def test_document_kv_connector_requires_hma_finish_provider(monkeypatch):
    class MissingHMAFinishProvider(RecordingProvider):
        request_finished_all_groups = None

    module = ModuleType("document_kv_missing_hma_finish_provider")
    module.build_provider = lambda *, vllm_config, extra_config: MissingHMAFinishProvider()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY: f"{module.__name__}:build_provider"
            }
        )
    )

    with pytest.raises(TypeError, match="request_finished_all_groups"):
        DocumentKVConnector(vllm_config=vllm_config)


def test_dynamic_connector_identity_is_exported_from_package_root():
    assert DOCUMENT_KV_CONNECTOR_CLASS == "DocumentKVConnector"
    assert DOCUMENT_KV_CONNECTOR_MODULE_PATH == "vllm_kv_injection.vllm_dynamic_connector"
    assert vllm_kv_injection.DocumentKVConnector is DocumentKVConnector
    assert vllm_kv_injection.VLLMSupportsHMA is VLLMSupportsHMA
    assert vllm_kv_injection.vllm_runtime_import_error is vllm_runtime_import_error
    assert vllm_kv_injection.DOCUMENT_KV_CONNECTOR_MODULE_PATH == DOCUMENT_KV_CONNECTOR_MODULE_PATH


def test_probe_import_survives_broken_optional_vllm_runtime():
    package_src = os.fspath(REPO_ROOT / "src")
    document_src = os.fspath(WORKSPACE_ROOT / "restaurant-kv-serving" / "src")
    code = r'''
import builtins
import json

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "vllm.distributed.kv_transfer.kv_connector.v1.base":
        raise RuntimeError("torch inductor mismatch")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from vllm_kv_injection.probe import build_native_connector_probe
from vllm_kv_injection.vllm_dynamic_connector import vllm_runtime_import_error

error = vllm_runtime_import_error()
print(json.dumps({
    "probe_name": build_native_connector_probe.__name__,
    "error_type": type(error).__name__ if error is not None else None,
    "error": str(error) if error is not None else None,
}))
'''
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([package_src, document_src]),
    }

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    record = json.loads(result.stdout)
    assert record == {
        "probe_name": "build_native_connector_probe",
        "error_type": "RuntimeError",
        "error": "torch inductor mismatch",
    }


def test_document_kv_connector_uses_vllm_base_when_available():
    repo_root = os.fspath(REPO_ROOT)
    code = r'''
from abc import ABC, abstractmethod
from types import ModuleType, SimpleNamespace
import json
import sys

module_names = [
    "vllm",
    "vllm.distributed",
    "vllm.distributed.kv_transfer",
    "vllm.distributed.kv_transfer.kv_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1",
]
for name in module_names:
    sys.modules[name] = ModuleType(name)

base = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")

class SupportsHMA(ABC):
    @abstractmethod
    def request_finished_all_groups(self, request, block_ids):
        raise NotImplementedError

class KVConnectorBase_V1:
    def __init__(self, vllm_config, role, kv_cache_config):
        self.base_initialized = True
        self._vllm_config = vllm_config
        self._role = role
        self._kv_cache_config = kv_cache_config

    @property
    def role(self):
        return self._role

base.SupportsHMA = SupportsHMA
base.KVConnectorBase_V1 = KVConnectorBase_V1
sys.modules[base.__name__] = base

from vllm_kv_injection.vllm_dynamic_connector import DocumentKVConnector

config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}))
connector = DocumentKVConnector(config, "worker", "kv-cache-config")
print(json.dumps({
    "base_initialized": connector.base_initialized,
    "is_hma": isinstance(connector, SupportsHMA),
    "role": connector.role,
    "connector_role": connector.connector_role,
}))
'''
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(REPO_ROOT / "src"), str(WORKSPACE_ROOT / "restaurant-kv-serving" / "src")))}

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(completed.stdout) == {
        "base_initialized": True,
        "is_hma": True,
        "role": "worker",
        "connector_role": "worker",
    }
