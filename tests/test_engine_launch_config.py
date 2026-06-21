import json

import pytest

from document_kv_cache.engine_adapters import (
    ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
    ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
    ServingBackend,
)
from document_kv_cache.engine_launch_config import (
    ENGINE_LAUNCH_CONFIG_EVIDENCE_RECORD_TYPE,
    ENGINE_LAUNCH_CONFIG_EVIDENCE_SCHEMA_VERSION,
    REQUIRED_ENGINE_LAUNCH_CONFIG_BACKENDS,
    EngineLaunchConfigEvidence,
    engine_launch_config_evidence_to_record,
    engine_launch_config_record_issues,
    evaluate_engine_launch_config_evidence,
    read_engine_launch_config_json,
    validate_engine_launch_config_record,
    write_engine_launch_config_evidence_json,
)


def _document_kv_extra(backend: str, *, requires_native_runtime: bool = True) -> dict[str, object]:
    return {
        "document_kv.record_type": f"{backend}_kv_injection.test_config.v1",
        "document_kv.schema_version": 1,
        "document_kv.backend": backend,
        "document_kv.connector_package": backend,
        "document_kv.kv_injection_method": "native-kv-import",
        "document_kv.engine_handoff_record_type": ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
        "document_kv.engine_handoff_schema_version": ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
        "document_kv.requires_native_runtime": requires_native_runtime,
    }


def _vllm_launch_config(**extra_overrides: object) -> dict[str, object]:
    extra = _document_kv_extra("vllm")
    extra.update(extra_overrides)
    return {
        "kv_connector": "DocumentKVConnector",
        "kv_connector_module_path": "company_vllm_patch.document_kv_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra,
    }


def _sglang_launch_config(**extra_overrides: object) -> dict[str, object]:
    extra = {
        "backend_name": "document_kv",
        "module_path": "company_sglang_patch.document_kv_backend",
        "class_name": "DocumentKVHiCacheBackend",
        **_document_kv_extra("sglang"),
    }
    extra.update(extra_overrides)
    return {
        "enable_hierarchical_cache": True,
        "hicache_storage_backend": "dynamic",
        "hicache_storage_backend_extra_config": json.dumps(extra, sort_keys=True),
    }


def test_validate_engine_launch_config_record_accepts_adapter_launch_shapes():
    validate_engine_launch_config_record(_vllm_launch_config(), expected_backend=ServingBackend.VLLM)
    validate_engine_launch_config_record(_sglang_launch_config(), expected_backend=" SGLANG ")


def test_validate_engine_launch_config_record_rejects_wrong_backend():
    issues = engine_launch_config_record_issues(_vllm_launch_config(), expected_backend=" SGLANG ")

    assert issues == ("engine launch config backend 'vllm' does not match expected_backend",)


@pytest.mark.parametrize(
    ("field", "value", "expected_issue"),
    [
        ("kv_connector", "OtherConnector", "vLLM launch config kv_connector must be 'DocumentKVConnector'"),
        (
            "kv_connector_module_path",
            "other.module",
            "vLLM launch config kv_connector_module_path must end with 'document_kv_connector'",
        ),
        ("kv_role", "other_role", "vLLM launch config kv_role must be one of"),
    ],
)
def test_validate_engine_launch_config_record_rejects_wrong_vllm_adapter_identity(
    field,
    value,
    expected_issue,
):
    record = _vllm_launch_config()
    record[field] = value

    assert any(expected_issue in issue for issue in engine_launch_config_record_issues(record))


@pytest.mark.parametrize(
    ("field", "value", "expected_issue"),
    [
        ("backend_name", "other_backend", "SGLang HiCache extra config backend_name must be 'document_kv'"),
        (
            "module_path",
            "other.module",
            "SGLang HiCache extra config module_path must end with 'document_kv_backend'",
        ),
        ("class_name", "OtherClass", "SGLang HiCache extra config class_name must be 'DocumentKVHiCacheBackend'"),
    ],
)
def test_validate_engine_launch_config_record_rejects_wrong_sglang_adapter_identity(
    field,
    value,
    expected_issue,
):
    extra = {
        "backend_name": "document_kv",
        "module_path": "company_sglang_patch.document_kv_backend",
        "class_name": "DocumentKVHiCacheBackend",
    }
    extra[field] = value
    record = _sglang_launch_config(**extra)

    assert any(expected_issue in issue for issue in engine_launch_config_record_issues(record))


def test_validate_engine_launch_config_record_rejects_non_native_or_mismatched_contract():
    record = _vllm_launch_config(
        **{
            "document_kv.backend": "sglang",
            "document_kv.requires_native_runtime": False,
        }
    )

    with pytest.raises(ValueError, match="document_kv.backend must be 'vllm'"):
        validate_engine_launch_config_record(record)
    assert "document_kv.requires_native_runtime must be true" in engine_launch_config_record_issues(record)


def test_validate_engine_launch_config_record_rejects_invalid_sglang_extra_config():
    record = _sglang_launch_config()
    record["hicache_storage_backend_extra_config"] = "not-json"

    with pytest.raises(ValueError, match="not valid JSON"):
        validate_engine_launch_config_record(record)


def test_evaluate_engine_launch_config_evidence_requires_vllm_and_sglang():
    evidence = evaluate_engine_launch_config_evidence([_vllm_launch_config(), _sglang_launch_config()])

    assert evidence.ok
    assert evidence.backends == REQUIRED_ENGINE_LAUNCH_CONFIG_BACKENDS
    assert engine_launch_config_evidence_to_record(evidence) == {
        "record_type": ENGINE_LAUNCH_CONFIG_EVIDENCE_RECORD_TYPE,
        "schema_version": ENGINE_LAUNCH_CONFIG_EVIDENCE_SCHEMA_VERSION,
        "ok": True,
        "issues": [],
        "backends": ["vllm", "sglang"],
        "missing_backends": [],
        "duplicate_backends": [],
        "invalid_records": [],
        "required_backends": ["vllm", "sglang"],
    }


def test_evaluate_engine_launch_config_evidence_reports_missing_duplicate_and_invalid_records():
    evidence = evaluate_engine_launch_config_evidence(
        [
            _vllm_launch_config(),
            _vllm_launch_config(),
            {"kv_connector": "missing extra"},
        ]
    )

    assert not evidence.ok
    assert evidence.backends == ("vllm",)
    assert evidence.missing_backends == ("sglang",)
    assert evidence.duplicate_backends == ("vllm",)
    assert evidence.invalid_records == (
        "record[2]: engine launch config must match either the vLLM transfer config shape "
        "or the SGLang HiCache config shape",
    )


def test_read_and_write_engine_launch_config_json(tmp_path):
    launch_config_path = tmp_path / "vllm-launch.json"
    launch_config_path.write_text(json.dumps(_vllm_launch_config()), encoding="utf-8")

    assert read_engine_launch_config_json(launch_config_path, expected_backend="vllm")["kv_connector"] == (
        "DocumentKVConnector"
    )

    evidence = EngineLaunchConfigEvidence(
        backends=("vllm", "sglang"),
        missing_backends=(),
        invalid_records=(),
    )
    evidence_path = write_engine_launch_config_evidence_json(evidence, tmp_path / "evidence.json")

    assert json.loads(evidence_path.read_text(encoding="utf-8"))["ok"] is True
