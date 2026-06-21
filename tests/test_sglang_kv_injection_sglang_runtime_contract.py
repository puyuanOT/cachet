import pytest

from document_kv_cache.native_probe_factories import native_probe_adapter_contract_to_record
from sglang_kv_injection.sglang_runtime_contract import (
    SGLANG_RUNTIME_CACHE_CONTRACT,
    SGLANG_RUNTIME_CACHE_CONTRACT_RECORD_TYPE,
    SGLANG_RUNTIME_CACHE_CONTRACT_SCHEMA_VERSION,
    SGLANG_RUNTIME_CACHE_DOC_URL,
    SGLANG_RUNTIME_CACHE_OPTIONAL_METHODS,
    SGLANG_RUNTIME_CACHE_REQUIRED_METHODS,
    SGLANG_RUNTIME_CACHE_RUNTIME,
    sglang_runtime_cache_contract_record_issues,
    sglang_runtime_cache_contract_to_record,
    sglang_runtime_cache_method_issues,
    validate_sglang_runtime_cache_contract_record,
    validate_sglang_runtime_cache_methods,
)


class CompleteSGLangRuntimeCacheConnector:
    def stage(self, record, *, payload=None):
        return None

    def attach(self, *, request_id, record):
        return None

    def release(self, request_id):
        return None


def test_sglang_runtime_cache_contract_record_documents_runtime_lifecycle():
    record = sglang_runtime_cache_contract_to_record(
        handoff_contract=native_probe_adapter_contract_to_record(),
    )

    assert record == {
        "record_type": SGLANG_RUNTIME_CACHE_CONTRACT_RECORD_TYPE,
        "schema_version": SGLANG_RUNTIME_CACHE_CONTRACT_SCHEMA_VERSION,
        "runtime": SGLANG_RUNTIME_CACHE_RUNTIME,
        "doc_url": SGLANG_RUNTIME_CACHE_DOC_URL,
        "required_methods": list(SGLANG_RUNTIME_CACHE_REQUIRED_METHODS),
        "optional_methods": list(SGLANG_RUNTIME_CACHE_OPTIONAL_METHODS),
        "handoff_contract": native_probe_adapter_contract_to_record(),
    }
    assert record["required_methods"] == ["stage", "attach", "release"]
    validate_sglang_runtime_cache_contract_record(record)


def test_sglang_runtime_cache_contract_constant_is_deeply_immutable():
    with pytest.raises(TypeError):
        SGLANG_RUNTIME_CACHE_CONTRACT["required_methods"] = ("stage",)

    with pytest.raises(AttributeError):
        SGLANG_RUNTIME_CACHE_CONTRACT["required_methods"].append("prefetch")


def test_sglang_runtime_cache_contract_record_rejects_shape_drift():
    record = sglang_runtime_cache_contract_to_record()
    record["required_methods"] = ["reserve", "inject", "release"]
    record["extra"] = True

    issues = sglang_runtime_cache_contract_record_issues(record)

    assert "SGLang runtime-cache contract required_methods must match the package contract" in issues
    assert any("unsupported keys" in issue and "extra" in issue for issue in issues)
    with pytest.raises(ValueError, match="required_methods"):
        validate_sglang_runtime_cache_contract_record(record)


def test_sglang_runtime_cache_contract_record_rejects_handoff_drift():
    record = sglang_runtime_cache_contract_to_record(
        handoff_contract={**native_probe_adapter_contract_to_record(), "schema_version": 999},
    )

    issues = sglang_runtime_cache_contract_record_issues(record)

    assert "SGLang runtime-cache contract handoff_contract must match the Document KV native-probe contract" in issues
    with pytest.raises(ValueError, match="handoff_contract"):
        validate_sglang_runtime_cache_contract_record(record)


def test_validate_sglang_runtime_cache_methods_accepts_required_lifecycle():
    validate_sglang_runtime_cache_methods(CompleteSGLangRuntimeCacheConnector())
    assert sglang_runtime_cache_method_issues(CompleteSGLangRuntimeCacheConnector()) == ()


def test_validate_sglang_runtime_cache_methods_reports_missing_hooks():
    class IncompleteConnector:
        def stage(self, record, *, payload=None):
            return None

    issues = sglang_runtime_cache_method_issues(IncompleteConnector())

    assert len(issues) == 1
    assert "attach" in issues[0]
    assert "release" in issues[0]
    with pytest.raises(TypeError, match="attach"):
        validate_sglang_runtime_cache_methods(IncompleteConnector())


def test_runtime_contract_is_exported_from_package_root():
    import sglang_kv_injection

    assert sglang_kv_injection.SGLANG_RUNTIME_CACHE_RUNTIME == SGLANG_RUNTIME_CACHE_RUNTIME
    assert sglang_kv_injection.SGLANG_RUNTIME_CACHE_CONTRACT == SGLANG_RUNTIME_CACHE_CONTRACT
    assert (
        sglang_kv_injection.sglang_runtime_cache_contract_to_record()
        == sglang_runtime_cache_contract_to_record()
    )
