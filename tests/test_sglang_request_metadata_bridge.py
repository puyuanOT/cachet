from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from sglang_kv_injection.sglang_request_metadata_bridge import (
    DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_RECORD_TYPE,
    DOCUMENT_KV_SGLANG_HICACHE_LAST_HASH_EXTRA_INFO_KEY,
    install_sglang_request_metadata_bridge,
    sglang_request_metadata_bridge_status_to_record,
)


def install_fake_sglang_bridge_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    packages = [
        "sglang",
        "sglang.srt",
        "sglang.srt.managers",
    ]
    for package_name in packages:
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    cache_controller_module = ModuleType("sglang.srt.managers.cache_controller")
    exec(
        """
class HiCacheStorageExtraInfo:
    def __init__(self, prefix_keys=None, extra_info=None):
        self.prefix_keys = prefix_keys
        self.extra_info = extra_info


class PrefetchOperation:
    def __init__(self, request_id, host_indices, token_ids, last_hash=None, prefix_keys=None):
        self.request_id = request_id
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.prefix_keys = prefix_keys
        self.hash_value = []
        self.completed_tokens = 0

    def increment(self, num_tokens):
        self.completed_tokens += num_tokens
        return True


class Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class StorageBackend:
    def __init__(self):
        self.exists_extra_info = []
        self.get_extra_info = []
        self.set_extra_info = []

    def batch_exists(self, keys, extra_info=None):
        self.exists_extra_info.append(extra_info)
        return len(keys)

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        self.get_extra_info.append(extra_info)
        return [True for _key in keys]

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        self.set_extra_info.append(extra_info)
        return [True for _key in keys]


class HiCacheController:
    def __init__(self):
        self.prefetch_queue = Queue()
        self.storage_backend = StorageBackend()
        self.prefetch_side_effects = []
        self.page_size = 2
        self.storage_batch_size = 1
        self.page_get_func = self._page_get_zero_copy
        self.page_set_func = self._page_set_zero_copy

    def prefetch(self, request_id, host_indices, new_input_tokens, last_hash=None, prefix_keys=None):
        self.prefetch_side_effects.append(("original_prefetch", request_id))
        operation = PrefetchOperation(
            request_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.prefetch_side_effects.append(
            ("constructed_extra_info", getattr(operation, "document_kv_extra_info", None))
        )
        self.prefetch_queue.put(operation)
        return operation

    def get_hash_str(self, tokens, last_hash):
        return "hash-" + "-".join(str(token) for token in tokens)

    def _storage_hit_query(self, operation):
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None
        batch_hashes = [self.get_hash_str(operation.token_ids, operation.last_hash)]
        extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
        self.storage_backend.batch_exists(batch_hashes, extra_info)
        operation.hash_value = batch_hashes
        return batch_hashes, len(batch_hashes) * self.page_size

    def _page_get_zero_copy(self, operation, hash_values, host_indices, extra_info=None):
        self.storage_backend.batch_get_v1(hash_values, host_indices, extra_info)
        operation.increment(len(hash_values) * self.page_size)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None):
        return all(self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info))

    def _page_transfer(self, operation):
        prefix_keys = operation.prefix_keys
        extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys, extra_info=None)
        self.page_get_func(operation, operation.hash_value, operation.host_indices, extra_info)

    def _page_backup(self, operation):
        prefix_keys = operation.prefix_keys
        extra_info = HiCacheStorageExtraInfo(prefix_keys, None)
        self.page_set_func(operation.hash_value, operation.host_indices, extra_info)
""",
        cache_controller_module.__dict__,
    )

    scheduler_module = ModuleType("sglang.srt.managers.scheduler")
    exec(
        """
class Scheduler:
    def __init__(self, controller):
        self.tree_cache = type("TreeCache", (), {})()
        self.tree_cache.cache_controller = controller

    def _prefetch_kvcache(self, req):
        if getattr(req, "skip_prefetch", False):
            return "skipped"
        self.tree_cache.cache_controller.prefetch(
            req.rid,
            [0, 1],
            [101, 102],
            "last-hash",
            ["prefix-hash"],
        )
""",
        scheduler_module.__dict__,
    )

    monkeypatch.setitem(sys.modules, cache_controller_module.__name__, cache_controller_module)
    monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
    return scheduler_module, cache_controller_module


def test_sglang_request_metadata_bridge_forwards_custom_params_to_hicache_extra_info(
    monkeypatch,
    caplog,
):
    scheduler_module, cache_controller_module = install_fake_sglang_bridge_modules(monkeypatch)
    caplog.set_level(logging.INFO, logger="sglang_kv_injection.sglang_request_metadata_bridge")

    status = install_sglang_request_metadata_bridge()
    record = sglang_request_metadata_bridge_status_to_record(status)

    assert status.installed is True
    assert record["record_type"] == DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_RECORD_TYPE
    assert record["prefetch_operation_patched"] is True
    assert record["ok"] is True
    assert "page_backup_patched" not in record

    controller = cache_controller_module.HiCacheController()
    scheduler = scheduler_module.Scheduler(controller)
    req_backref = object()
    req = SimpleNamespace(
        rid="cachet-sglang-request-1",
        sampling_params=SimpleNamespace(
            custom_params={
                "kv_transfer_params": {
                    "document_kv.request_id": "cachet-sglang-request-1",
                    "document_kv.handoff_json": "/tmp/cachet-secret-handoff.json",
                    "document_kv.payload_uri": "disk:/tmp/cachet-secret-payload.kv",
                    "document_kv.sglang_hicache_page_keys": ["page-secret-1", "page-secret-2"],
                },
                "ordinary_sglang_param": "kept",
                "__req__": req_backref,
            },
        ),
    )

    scheduler._prefetch_kvcache(req)

    assert len(controller.prefetch_queue.items) == 1
    operation = controller.prefetch_queue.items[0]
    expected_extra_info = {
        "custom_params": {
            "kv_transfer_params": {
                "document_kv.request_id": "cachet-sglang-request-1",
                "document_kv.handoff_json": "/tmp/cachet-secret-handoff.json",
                "document_kv.payload_uri": "disk:/tmp/cachet-secret-payload.kv",
                "document_kv.sglang_hicache_page_keys": ["page-secret-1", "page-secret-2"],
            },
            "ordinary_sglang_param": "kept",
        },
    }
    assert controller.prefetch_side_effects == [
        ("original_prefetch", "cachet-sglang-request-1"),
        ("constructed_extra_info", expected_extra_info),
    ]
    assert operation.document_kv_extra_info == expected_extra_info
    assert "event=request_registered" in caplog.text
    assert "sglang_hicache_page_key_count=2" in caplog.text
    assert "cachet-sglang-request-1" not in caplog.text
    assert "cachet-secret" not in caplog.text
    assert "page-secret" not in caplog.text

    controller._storage_hit_query(operation)
    controller._page_transfer(operation)

    expected_runtime_extra_info = {
        **expected_extra_info,
        DOCUMENT_KV_SGLANG_HICACHE_LAST_HASH_EXTRA_INFO_KEY: "last-hash",
    }
    assert controller.storage_backend.exists_extra_info[0].extra_info == expected_runtime_extra_info
    assert controller.storage_backend.get_extra_info[0].extra_info == expected_runtime_extra_info
    assert "__req__" not in operation.document_kv_extra_info["custom_params"]


def test_sglang_request_metadata_bridge_is_idempotent(monkeypatch):
    install_fake_sglang_bridge_modules(monkeypatch)

    first_status = install_sglang_request_metadata_bridge()
    second_status = install_sglang_request_metadata_bridge()

    assert first_status.installed is True
    assert second_status.installed is True


def test_sglang_request_metadata_bridge_cleans_registry_when_prefetch_is_skipped(monkeypatch):
    scheduler_module, cache_controller_module = install_fake_sglang_bridge_modules(monkeypatch)
    status = install_sglang_request_metadata_bridge()
    assert status.installed is True
    controller = cache_controller_module.HiCacheController()
    scheduler = scheduler_module.Scheduler(controller)
    req = SimpleNamespace(
        rid="cachet-sglang-request-skipped",
        skip_prefetch=True,
        sampling_params=SimpleNamespace(
            custom_params={
                "kv_transfer_params": {"document_kv.request_id": "cachet-sglang-request-skipped"},
            },
        ),
    )

    assert scheduler._prefetch_kvcache(req) == "skipped"

    assert getattr(controller, "_document_kv_request_metadata_by_rid", {}) == {}
    assert controller.prefetch_queue.items == []


def test_sglang_request_metadata_bridge_fails_closed_when_runtime_shape_is_missing(monkeypatch):
    for package_name in ("sglang", "sglang.srt", "sglang.srt.managers"):
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", ModuleType("sglang.srt.managers.scheduler"))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.cache_controller",
        ModuleType("sglang.srt.managers.cache_controller"),
    )

    status = install_sglang_request_metadata_bridge()
    record = sglang_request_metadata_bridge_status_to_record(status)

    assert status.installed is False
    assert record["ok"] is False
    assert "error" in record
