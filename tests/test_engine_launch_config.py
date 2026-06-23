import json

import pytest

from document_kv_cache.engine_adapters import (
    ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
    ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
    ServingBackend,
)
from document_kv_cache.engine_launch_config import (
    DEFAULT_SGLANG_DOCUMENT_KV_PROVIDER_FACTORY,
    DEFAULT_SGLANG_ENGINE_LAUNCH_CONFIG_RECORD_TYPE,
    DEFAULT_VLLM_DOCUMENT_KV_PROVIDER_FACTORY,
    DEFAULT_VLLM_ENGINE_LAUNCH_CONFIG_RECORD_TYPE,
    ENGINE_LAUNCH_CONFIG_EVIDENCE_RECORD_TYPE,
    ENGINE_LAUNCH_CONFIG_EVIDENCE_SCHEMA_VERSION,
    REQUIRED_ENGINE_LAUNCH_CONFIG_BACKENDS,
    EngineLaunchConfigEvidence,
    build_sglang_launch_config,
    build_vllm_launch_config,
    engine_launch_config_evidence_to_record,
    engine_launch_config_record_issues,
    evaluate_engine_launch_config_evidence,
    main,
    read_engine_launch_config_json,
    validate_engine_launch_config_record,
    write_engine_launch_config_json,
    write_engine_launch_config_evidence_json,
)
from sglang_kv_injection.sglang_dynamic_backend import DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY
from vllm_kv_injection.vllm_native_provider_constants import DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY


SGLANG_TEST_PROVIDER_FACTORY = "company_sglang_patch.providers:build_provider"


def _document_kv_extra(
    backend: str,
    *,
    provider_factory: str | None = None,
    requires_native_runtime: bool = True,
) -> dict[str, object]:
    extra = {
        "document_kv.record_type": f"{backend}_kv_injection.test_config.v1",
        "document_kv.schema_version": 1,
        "document_kv.backend": backend,
        "document_kv.connector_package": backend,
        "document_kv.kv_injection_method": "native-kv-import",
        "document_kv.engine_handoff_record_type": ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
        "document_kv.engine_handoff_schema_version": ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
        "document_kv.requires_native_runtime": requires_native_runtime,
    }
    if backend == "vllm":
        extra["document_kv.provider_factory"] = DEFAULT_VLLM_DOCUMENT_KV_PROVIDER_FACTORY
    elif provider_factory is not None:
        extra[DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY] = provider_factory
    return extra


def _vllm_launch_config(**extra_overrides: object) -> dict[str, object]:
    extra = _document_kv_extra("vllm")
    extra.update(extra_overrides)
    return {
        "kv_connector": "DocumentKVConnector",
        "kv_connector_module_path": "company_vllm_patch.document_kv_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra,
    }


def _vllm_package_launch_config(**extra_overrides: object) -> dict[str, object]:
    record = _vllm_launch_config(**extra_overrides)
    record["kv_connector_module_path"] = "vllm_kv_injection.vllm_dynamic_connector"
    return record


def _sglang_launch_config(
    *,
    provider_factory: str | None = SGLANG_TEST_PROVIDER_FACTORY,
    **extra_overrides: object,
) -> dict[str, object]:
    extra = {
        "backend_name": "document_kv",
        "module_path": "company_sglang_patch.document_kv_backend",
        "class_name": "DocumentKVHiCacheBackend",
        **_document_kv_extra("sglang", provider_factory=provider_factory),
    }
    extra.update(extra_overrides)
    return {
        "enable_hierarchical_cache": True,
        "hicache_storage_backend": "dynamic",
        "hicache_storage_backend_extra_config": json.dumps(extra, sort_keys=True),
    }


def _sglang_package_launch_config(**extra_overrides: object) -> dict[str, object]:
    return _sglang_launch_config(module_path="sglang_kv_injection.sglang_dynamic_backend", **extra_overrides)


def test_validate_engine_launch_config_record_accepts_adapter_launch_shapes():
    validate_engine_launch_config_record(_vllm_launch_config(), expected_backend=ServingBackend.VLLM)
    validate_engine_launch_config_record(_sglang_launch_config(), expected_backend=" SGLANG ")
    validate_engine_launch_config_record(_vllm_package_launch_config(), expected_backend=ServingBackend.VLLM)
    validate_engine_launch_config_record(_sglang_package_launch_config(), expected_backend=" SGLANG ")


def test_build_vllm_launch_config_emits_valid_release_shape():
    record = build_vllm_launch_config(extra_config={"deployment": "qa", "max_model_len": 32768})

    assert record == {
        "kv_connector": "DocumentKVConnector",
        "kv_connector_module_path": "vllm_kv_injection.vllm_dynamic_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "deployment": "qa",
            "max_model_len": 32768,
            **_document_kv_extra("vllm"),
            "document_kv.record_type": DEFAULT_VLLM_ENGINE_LAUNCH_CONFIG_RECORD_TYPE,
            "document_kv.provider_factory": DEFAULT_VLLM_DOCUMENT_KV_PROVIDER_FACTORY,
        },
    }
    validate_engine_launch_config_record(record, expected_backend=ServingBackend.VLLM)


def test_build_vllm_launch_config_accepts_custom_provider_factory():
    record = build_vllm_launch_config(provider_factory="company_vllm_patch.provider:build_provider")

    assert (
        record["kv_connector_extra_config"]["document_kv.provider_factory"]
        == "company_vllm_patch.provider:build_provider"
    )
    validate_engine_launch_config_record(record, expected_backend=ServingBackend.VLLM)


def test_build_vllm_launch_config_accepts_payload_cache_budget():
    record = build_vllm_launch_config(payload_cache_max_bytes=4096)

    assert record["kv_connector_extra_config"][DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY] == 4096
    validate_engine_launch_config_record(record, expected_backend=ServingBackend.VLLM)


@pytest.mark.parametrize("value", [-1, True, "4096"])
def test_build_vllm_launch_config_rejects_invalid_payload_cache_budget(value):
    with pytest.raises(ValueError, match="payload_cache_max_bytes"):
        build_vllm_launch_config(payload_cache_max_bytes=value)


def test_build_sglang_launch_config_emits_valid_release_shape():
    record = build_sglang_launch_config(
        extra_config={"deployment": "qa"},
        provider_factory=SGLANG_TEST_PROVIDER_FACTORY,
    )
    extra = json.loads(record["hicache_storage_backend_extra_config"])

    assert record["enable_hierarchical_cache"] is True
    assert record["hicache_storage_backend"] == "dynamic"
    assert extra == {
        "backend_name": "document_kv",
        "module_path": "sglang_kv_injection.sglang_dynamic_backend",
        "class_name": "DocumentKVHiCacheBackend",
        "deployment": "qa",
        **_document_kv_extra("sglang", provider_factory=SGLANG_TEST_PROVIDER_FACTORY),
        "document_kv.record_type": DEFAULT_SGLANG_ENGINE_LAUNCH_CONFIG_RECORD_TYPE,
    }
    validate_engine_launch_config_record(record, expected_backend=ServingBackend.SGLANG)


def test_build_sglang_launch_config_defaults_to_builtin_provider_factory():
    record = build_sglang_launch_config()
    extra = json.loads(record["hicache_storage_backend_extra_config"])

    assert extra[DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY] == DEFAULT_SGLANG_DOCUMENT_KV_PROVIDER_FACTORY
    validate_engine_launch_config_record(record, expected_backend=ServingBackend.SGLANG)


def test_build_launch_config_rejects_reserved_extra_config_keys():
    with pytest.raises(ValueError, match=r"reserved document_kv\.\*"):
        build_vllm_launch_config(extra_config={"document_kv.backend": "sglang"})

    with pytest.raises(ValueError, match="reserved key 'module_path'"):
        build_sglang_launch_config(extra_config={"module_path": "other.module"})


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
            "vLLM launch config kv_connector_module_path must end with one of",
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
            "SGLang HiCache extra config module_path must end with one of",
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


def test_validate_engine_launch_config_record_rejects_missing_vllm_provider_factory():
    record = _vllm_launch_config()
    record["kv_connector_extra_config"].pop("document_kv.provider_factory")

    issues = engine_launch_config_record_issues(record, expected_backend="vllm")

    assert "document_kv.provider_factory must be a non-empty module:attribute string" in issues


def test_validate_engine_launch_config_record_rejects_malformed_vllm_provider_factory():
    record = _vllm_launch_config(**{"document_kv.provider_factory": "missing_attribute"})

    issues = engine_launch_config_record_issues(record, expected_backend="vllm")

    assert "document_kv.provider_factory must use module:attribute syntax" in issues


def test_validate_engine_launch_config_record_rejects_dotted_vllm_provider_factory_attribute():
    record = _vllm_launch_config(**{"document_kv.provider_factory": "json:loads.decoder"})

    issues = engine_launch_config_record_issues(record, expected_backend="vllm")

    assert "document_kv.provider_factory attribute must be a Python identifier" in issues


def test_validate_engine_launch_config_record_rejects_malformed_sglang_provider_factory():
    record = _sglang_launch_config(provider_factory="missing_attribute")

    issues = engine_launch_config_record_issues(record, expected_backend="sglang")

    assert "document_kv.provider_factory must use module:attribute syntax" in issues


def test_validate_engine_launch_config_record_rejects_invalid_sglang_extra_config():
    record = _sglang_launch_config()
    record["hicache_storage_backend_extra_config"] = "not-json"

    with pytest.raises(ValueError, match="not valid JSON"):
        validate_engine_launch_config_record(record)


def test_evaluate_engine_launch_config_evidence_requires_vllm_and_sglang():
    evidence = evaluate_engine_launch_config_evidence([_vllm_package_launch_config(), _sglang_package_launch_config()])

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

    generated_path = write_engine_launch_config_json(
        build_sglang_launch_config(encode_extra_config_as_json=False),
        tmp_path / "sglang-launch.json",
        expected_backend="sglang",
    )
    assert read_engine_launch_config_json(generated_path, expected_backend="sglang")[
        "hicache_storage_backend"
    ] == "dynamic"

    evidence = EngineLaunchConfigEvidence(
        backends=("vllm", "sglang"),
        missing_backends=(),
        invalid_records=(),
    )
    evidence_path = write_engine_launch_config_evidence_json(evidence, tmp_path / "evidence.json")

    assert json.loads(evidence_path.read_text(encoding="utf-8"))["ok"] is True


def test_main_builds_launch_config_sidecars(tmp_path):
    vllm_path = tmp_path / "vllm.json"
    sglang_path = tmp_path / "sglang.json"

    assert (
        main(
            [
                "build-vllm",
                "--output-json",
                str(vllm_path),
                "--extra-config",
                "max_model_len=32768",
                "--payload-cache-max-bytes",
                "4096",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "build-sglang",
                "--output-json",
                str(sglang_path),
                "--provider-factory",
                SGLANG_TEST_PROVIDER_FACTORY,
                "--extra-config",
                "deployment=\"qa\"",
            ]
        )
        == 0
    )

    assert read_engine_launch_config_json(vllm_path, expected_backend="vllm")[
        "kv_connector_extra_config"
    ]["max_model_len"] == 32768
    assert read_engine_launch_config_json(vllm_path, expected_backend="vllm")[
        "kv_connector_extra_config"
    ]["document_kv.provider_factory"] == DEFAULT_VLLM_DOCUMENT_KV_PROVIDER_FACTORY
    assert read_engine_launch_config_json(vllm_path, expected_backend="vllm")[
        "kv_connector_extra_config"
    ][DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY] == 4096
    sglang_extra = json.loads(
        read_engine_launch_config_json(sglang_path, expected_backend="sglang")[
            "hicache_storage_backend_extra_config"
        ]
    )
    assert sglang_extra["deployment"] == "qa"
    assert sglang_extra[DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY] == SGLANG_TEST_PROVIDER_FACTORY


def test_main_reports_invalid_cli_input_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["build-vllm", "--extra-config", "document_kv.backend=sglang"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "extra_config must not override reserved document_kv.* keys" in captured.err
    assert "Traceback" not in captured.err
