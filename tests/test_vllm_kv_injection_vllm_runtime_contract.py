from types import SimpleNamespace

import pytest

from document_kv_cache.native_probe_factories import native_probe_adapter_contract_to_record
import vllm_kv_injection.vllm_runtime_contract as vllm_runtime_contract
from vllm_kv_injection.vllm_runtime_contract import (
    VLLMInstalledKVConnectorContract,
    VLLM_KV_CONNECTOR_V1_BASE_MODULE,
    VLLM_KV_CONNECTOR_V1_CONTRACT,
    VLLM_KV_CONNECTOR_V1_CONTRACT_RECORD_TYPE,
    VLLM_KV_CONNECTOR_V1_CONTRACT_SCHEMA_VERSION,
    VLLM_KV_CONNECTOR_V1_DOC_URL,
    VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_RECORD_TYPE,
    VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_SCHEMA_VERSION,
    VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS,
    VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
    VLLM_KV_CONNECTOR_V1_RUNTIME,
    inspect_installed_vllm_kv_connector_v1_contract,
    installed_vllm_kv_connector_v1_contract_record_issues,
    installed_vllm_kv_connector_v1_contract_to_record,
    validate_installed_vllm_kv_connector_v1_contract_record,
    validate_vllm_kv_connector_v1_contract_record,
    validate_vllm_kv_connector_v1_methods,
    vllm_kv_connector_v1_contract_record_issues,
    vllm_kv_connector_v1_contract_to_record,
    vllm_kv_connector_v1_method_issues,
)


class CompleteVLLMKVConnectorV1:
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return None, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return None

    def build_connector_meta(self, scheduler_output):
        return {}

    def register_kv_caches(self, kv_caches):
        return None

    def start_load_kv(self, forward_context, **kwargs):
        return None

    def wait_for_layer_load(self, layer_name):
        return None

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return None

    def wait_for_save(self):
        return None

    def request_finished(self, request, block_ids):
        return False, None

    def request_finished_all_groups(self, request, block_ids):
        return False, None


def _method(*args, **kwargs):
    return None


def _runtime_base_class(*, missing: set[str] | None = None, extra: set[str] | None = None):
    missing = missing or set()
    attrs = {
        method_name: _method
        for method_name in (*VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS, *VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS)
        if method_name not in missing and method_name != "request_finished_all_groups"
    }
    for method_name in extra or set():
        attrs[method_name] = _method
    attrs["prefer_cross_layer_blocks"] = property(lambda self: False)
    attrs["role"] = property(lambda self: None)
    return type("KVConnectorBase_V1", (), attrs)


def _runtime_hma_class(*, missing: set[str] | None = None):
    attrs = {}
    if "request_finished_all_groups" not in (missing or set()):
        attrs["request_finished_all_groups"] = _method
    return type("SupportsHMA", (), attrs)


def test_vllm_kv_connector_v1_contract_record_documents_runtime_lifecycle():
    record = vllm_kv_connector_v1_contract_to_record(
        handoff_contract=native_probe_adapter_contract_to_record(),
    )

    assert record == {
        "record_type": VLLM_KV_CONNECTOR_V1_CONTRACT_RECORD_TYPE,
        "schema_version": VLLM_KV_CONNECTOR_V1_CONTRACT_SCHEMA_VERSION,
        "runtime": VLLM_KV_CONNECTOR_V1_RUNTIME,
        "doc_url": VLLM_KV_CONNECTOR_V1_DOC_URL,
        "required_methods": list(VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS),
        "optional_methods": list(VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS),
        "handoff_contract": native_probe_adapter_contract_to_record(),
    }
    assert "get_num_new_matched_tokens" in record["required_methods"]
    assert "start_load_kv" in record["required_methods"]
    assert "wait_for_layer_load" in record["required_methods"]
    assert "save_kv_layer" in record["required_methods"]
    assert "wait_for_save" in record["required_methods"]
    assert "request_finished" in record["required_methods"]
    assert "request_finished_all_groups" in record["required_methods"]
    assert len(record["optional_methods"]) == len(set(record["optional_methods"]))
    assert "build_connector_worker_meta" in record["optional_methods"]
    assert "build_kv_connector_stats" in record["optional_methods"]
    assert "build_prom_metrics" in record["optional_methods"]
    assert "get_required_kvcache_layout" in record["optional_methods"]
    assert "handle_preemptions" in record["optional_methods"]
    assert "has_pending_push_work" in record["optional_methods"]
    assert "register_cross_layers_kv_cache" in record["optional_methods"]
    assert "update_connector_output" in record["optional_methods"]
    validate_vllm_kv_connector_v1_contract_record(record)


def test_inspect_installed_vllm_kv_connector_v1_contract_reports_matching_runtime(monkeypatch):
    module = SimpleNamespace(
        KVConnectorBase_V1=_runtime_base_class(),
        SupportsHMA=_runtime_hma_class(),
    )
    monkeypatch.setattr(
        vllm_runtime_contract.importlib,
        "import_module",
        lambda name: module if name == VLLM_KV_CONNECTOR_V1_BASE_MODULE else None,
    )
    monkeypatch.setattr(
        vllm_runtime_contract.package_metadata,
        "version",
        lambda package_name: "0.23.0" if package_name == "vllm" else None,
    )

    inspection = inspect_installed_vllm_kv_connector_v1_contract()
    record = installed_vllm_kv_connector_v1_contract_to_record(inspection)

    assert inspection.ok is True
    assert record["record_type"] == VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_RECORD_TYPE
    assert record["schema_version"] == VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_SCHEMA_VERSION
    assert record["runtime"] == VLLM_KV_CONNECTOR_V1_RUNTIME
    assert record["base_module"] == VLLM_KV_CONNECTOR_V1_BASE_MODULE
    assert record["package_version"] == "0.23.0"
    assert record["importable"] is True
    assert record["ok"] is True
    assert record["missing_required_methods"] == []
    assert record["extra_installed_methods"] == []
    assert record["extra_installed_properties"] == []
    assert "has_pending_push_work" in record["installed_methods"]
    assert "prefer_cross_layer_blocks" in record["installed_properties"]
    validate_installed_vllm_kv_connector_v1_contract_record(record)


def test_installed_vllm_kv_connector_v1_contract_reports_runtime_drift():
    inspection = VLLMInstalledKVConnectorContract(
        package_version="0.24.0",
        importable=True,
        installed_methods=tuple(
            sorted(
                (
                    set(VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS)
                    | set(VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS)
                    | {"future_hook"}
                )
                - {"start_load_kv"}
            )
        ),
        installed_properties=("prefer_cross_layer_blocks", "role", "future_property"),
    )

    record = installed_vllm_kv_connector_v1_contract_to_record(inspection)

    assert record["ok"] is False
    assert record["missing_required_methods"] == ["start_load_kv"]
    assert record["extra_installed_methods"] == ["future_hook"]
    assert record["extra_installed_properties"] == ["future_property"]
    validate_installed_vllm_kv_connector_v1_contract_record(record)


def test_inspect_installed_vllm_kv_connector_v1_contract_reports_import_error(monkeypatch):
    def fail_import(name):
        raise RuntimeError(f"cannot import {name}")

    monkeypatch.setattr(vllm_runtime_contract.importlib, "import_module", fail_import)
    monkeypatch.setattr(
        vllm_runtime_contract.package_metadata,
        "version",
        lambda package_name: (_ for _ in ()).throw(
            vllm_runtime_contract.package_metadata.PackageNotFoundError(package_name)
        ),
    )

    record = installed_vllm_kv_connector_v1_contract_to_record()

    assert record["package_version"] is None
    assert record["importable"] is False
    assert record["ok"] is False
    assert record["import_error_type"] == "RuntimeError"
    assert VLLM_KV_CONNECTOR_V1_BASE_MODULE in record["import_error"]
    validate_installed_vllm_kv_connector_v1_contract_record(record)


def test_installed_vllm_kv_connector_v1_contract_record_rejects_shape_drift():
    record = installed_vllm_kv_connector_v1_contract_to_record(
        VLLMInstalledKVConnectorContract(
            package_version="0.23.0",
            importable=True,
            installed_methods=VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
            installed_properties=("prefer_cross_layer_blocks",),
        )
    )
    record["extra"] = True
    record["required_methods"] = ["reserve", "inject", "release"]

    issues = installed_vllm_kv_connector_v1_contract_record_issues(record)

    assert any("unsupported keys" in issue and "extra" in issue for issue in issues)
    assert any("required_methods" in issue for issue in issues)
    with pytest.raises(ValueError, match="required_methods"):
        validate_installed_vllm_kv_connector_v1_contract_record(record)


def test_installed_vllm_kv_connector_v1_contract_record_rejects_inconsistent_derived_fields():
    record = installed_vllm_kv_connector_v1_contract_to_record(
        VLLMInstalledKVConnectorContract(
            package_version="0.24.0",
            importable=True,
            installed_methods=tuple(
                method_name
                for method_name in (
                    *VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
                    *VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS,
                )
                if method_name != "start_load_kv"
            ),
            installed_properties=("prefer_cross_layer_blocks", "role"),
        )
    )
    record["package_version"] = 24
    record["missing_required_methods"] = []
    record["ok"] = True

    issues = installed_vllm_kv_connector_v1_contract_record_issues(record)

    assert "installed vLLM KV connector contract package_version must be a non-empty string or null" in issues
    assert "installed vLLM KV connector contract missing_required_methods must match installed_methods" in issues
    assert "installed vLLM KV connector contract ok must match importable and detected drift" in issues
    with pytest.raises(ValueError, match="package_version"):
        validate_installed_vllm_kv_connector_v1_contract_record(record)


def test_vllm_kv_connector_v1_contract_record_rejects_shape_drift():
    record = vllm_kv_connector_v1_contract_to_record()
    record["required_methods"] = ["reserve", "inject", "release"]
    record["extra"] = True

    issues = vllm_kv_connector_v1_contract_record_issues(record)

    assert "vLLM V1 KV connector contract required_methods must match the package contract" in issues
    assert any("unsupported keys" in issue and "extra" in issue for issue in issues)
    with pytest.raises(ValueError, match="required_methods"):
        validate_vllm_kv_connector_v1_contract_record(record)


def test_validate_vllm_kv_connector_v1_methods_accepts_required_lifecycle():
    validate_vllm_kv_connector_v1_methods(CompleteVLLMKVConnectorV1())
    assert vllm_kv_connector_v1_method_issues(CompleteVLLMKVConnectorV1()) == ()


def test_validate_vllm_kv_connector_v1_methods_reports_missing_hooks():
    class IncompleteConnector:
        def start_load_kv(self, forward_context, **kwargs):
            return None

    issues = vllm_kv_connector_v1_method_issues(IncompleteConnector())

    assert len(issues) == 1
    assert "get_num_new_matched_tokens" in issues[0]
    assert "wait_for_layer_load" in issues[0]
    assert "request_finished" in issues[0]
    assert "request_finished_all_groups" in issues[0]
    with pytest.raises(TypeError, match="get_num_new_matched_tokens"):
        validate_vllm_kv_connector_v1_methods(IncompleteConnector())


def test_runtime_contract_is_exported_from_package_root():
    import vllm_kv_injection

    assert vllm_kv_injection.VLLM_KV_CONNECTOR_V1_RUNTIME == VLLM_KV_CONNECTOR_V1_RUNTIME
    assert vllm_kv_injection.VLLM_KV_CONNECTOR_V1_CONTRACT == VLLM_KV_CONNECTOR_V1_CONTRACT
    assert (
        vllm_kv_injection.installed_vllm_kv_connector_v1_contract_to_record
        is installed_vllm_kv_connector_v1_contract_to_record
    )
    assert (
        vllm_kv_injection.vllm_kv_connector_v1_contract_to_record()
        == vllm_kv_connector_v1_contract_to_record()
    )
