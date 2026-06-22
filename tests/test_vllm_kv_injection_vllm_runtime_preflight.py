from __future__ import annotations

import json

import pytest

import vllm_kv_injection.vllm_runtime_preflight as vllm_runtime_preflight
from vllm_kv_injection.vllm_native_provider import (
    DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
    DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE,
)
from vllm_kv_injection.vllm_runtime_contract import (
    VLLMInstalledKVConnectorContract,
    VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS,
    VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
    installed_vllm_kv_connector_v1_contract_to_record,
)
from vllm_kv_injection.vllm_runtime_preflight import (
    DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_RECORD_TYPE,
    DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
    document_kv_vllm_runtime_preflight_record_issues,
    document_kv_vllm_runtime_preflight_to_record,
    validate_document_kv_vllm_runtime_preflight_record,
)


def matching_installed_contract() -> dict:
    return installed_vllm_kv_connector_v1_contract_to_record(
        VLLMInstalledKVConnectorContract(
            package_version="0.23.0",
            importable=True,
            installed_methods=tuple(
                sorted(
                    (
                        *VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
                        *VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS,
                    )
                )
            ),
            installed_properties=("prefer_cross_layer_blocks", "role"),
        )
    )


def drifting_installed_contract() -> dict:
    return installed_vllm_kv_connector_v1_contract_to_record(
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


def test_vllm_runtime_preflight_accepts_matching_contract_and_layer_mapping():
    record = document_kv_vllm_runtime_preflight_to_record(
        [
            "model.layers.0.self_attn.attn",
            "model.layers.1.self_attn.attn",
        ],
        installed_contract=matching_installed_contract(),
    )

    assert record["record_type"] == DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_RECORD_TYPE
    assert record["schema_version"] == DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_SCHEMA_VERSION
    assert record["provider_factory"] == DOCUMENT_KV_NATIVE_PROVIDER_FACTORY
    assert record["installed_contract"]["ok"] is True
    assert record["layer_mapping"]["record_type"] == DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE
    assert record["layer_mapping"]["ok"] is True
    assert record["ok"] is True
    validate_document_kv_vllm_runtime_preflight_record(record)


def test_vllm_runtime_preflight_rejects_runtime_drift_and_bad_layer_mapping():
    record = document_kv_vllm_runtime_preflight_to_record(
        ["attention_without_index"],
        installed_contract=drifting_installed_contract(),
    )

    assert record["installed_contract"]["ok"] is False
    assert record["layer_mapping"]["ok"] is False
    assert record["ok"] is False
    issues = document_kv_vllm_runtime_preflight_record_issues(record)

    assert "installed_contract.ok must be true for a safe vLLM runtime preflight" in issues
    assert "layer_mapping.ok must be true for a safe vLLM layer mapping preflight" in issues
    assert "ok must be true for a safe vLLM runtime preflight" in issues
    with pytest.raises(ValueError, match="installed_contract.ok"):
        validate_document_kv_vllm_runtime_preflight_record(record)


def test_vllm_runtime_preflight_rejects_mismatched_provider_factory_and_ok_flag():
    record = document_kv_vllm_runtime_preflight_to_record(
        ["model.layers.0.self_attn.attn"],
        installed_contract=matching_installed_contract(),
        provider_factory="vllm_kv_injection.noop:build_provider",
    )
    record["ok"] = True

    issues = document_kv_vllm_runtime_preflight_record_issues(record)

    assert f"provider_factory must be {DOCUMENT_KV_NATIVE_PROVIDER_FACTORY!r}" in issues
    assert "ok must match provider factory, installed contract, and layer mapping safety" in issues
    with pytest.raises(ValueError, match="provider_factory"):
        validate_document_kv_vllm_runtime_preflight_record(record)


def test_vllm_runtime_preflight_cli_writes_strict_record(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vllm_runtime_preflight,
        "installed_vllm_kv_connector_v1_contract_to_record",
        matching_installed_contract,
    )
    output_path = tmp_path / "vllm-preflight.json"

    exit_code = vllm_runtime_preflight.main(
        [
            "--layer-name",
            "model.layers.0.self_attn.attn",
            "--output-json",
            str(output_path),
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["ok"] is True
    validate_document_kv_vllm_runtime_preflight_record(record)


def test_vllm_runtime_preflight_cli_fails_without_registered_layer_names(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vllm_runtime_preflight,
        "installed_vllm_kv_connector_v1_contract_to_record",
        matching_installed_contract,
    )
    output_path = tmp_path / "vllm-preflight.json"

    exit_code = vllm_runtime_preflight.main(["--output-json", str(output_path)])

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert record["layer_mapping"]["layer_names"] == []
    assert record["ok"] is False


def test_vllm_runtime_preflight_is_exposed_through_document_and_cachet_facades():
    import cachet.vllm_runtime_preflight as cachet_preflight
    import document_kv_cache.vllm_runtime_preflight as document_preflight

    assert cachet_preflight is document_preflight
    assert (
        document_preflight.document_kv_vllm_runtime_preflight_to_record
        is vllm_runtime_preflight.document_kv_vllm_runtime_preflight_to_record
    )
