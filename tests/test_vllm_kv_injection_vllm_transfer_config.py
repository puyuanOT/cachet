import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import vllm_kv_injection
from vllm_kv_injection.vllm_transfer_config import (
    DOCUMENT_KV_CONNECTOR_CLASS,
    DOCUMENT_KV_CONNECTOR_MODULE_PATH,
    DOCUMENT_KV_DEFAULT_ROLE,
    DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
    DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY,
    DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY,
    DOCUMENT_KV_TRANSFER_CONFIG_RECORD_TYPE,
    DOCUMENT_KV_TRANSFER_CONFIG_SCHEMA_VERSION,
    document_kv_transfer_config,
    document_kv_transfer_config_json,
    main,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def test_document_kv_transfer_config_builds_vllm_payload():
    config = document_kv_transfer_config(
        extra_config={"tenant": "qa", "max_ready_queue": 16},
        provider_factory="company_vllm_patch.document_kv_provider:build_provider",
    )

    assert config["kv_connector"] == DOCUMENT_KV_CONNECTOR_CLASS
    assert config["kv_connector_module_path"] == DOCUMENT_KV_CONNECTOR_MODULE_PATH
    assert config["kv_role"] == DOCUMENT_KV_DEFAULT_ROLE
    extra = config["kv_connector_extra_config"]
    assert extra["tenant"] == "qa"
    assert extra["max_ready_queue"] == 16
    assert extra["document_kv.record_type"] == DOCUMENT_KV_TRANSFER_CONFIG_RECORD_TYPE
    assert extra["document_kv.schema_version"] == DOCUMENT_KV_TRANSFER_CONFIG_SCHEMA_VERSION
    assert extra["document_kv.backend"] == "vllm"
    assert extra["document_kv.kv_injection_method"] == "engine-native-kv-block-import"
    assert extra["document_kv.engine_handoff_record_type"] == "document_kv.engine_adapter_request.v1"
    assert extra["document_kv.engine_handoff_schema_version"] == 2
    assert extra["document_kv.provider_factory"] == "company_vllm_patch.document_kv_provider:build_provider"
    assert extra["document_kv.requires_native_runtime"] is True


def test_document_kv_transfer_config_defaults_to_native_provider_factory():
    config = document_kv_transfer_config()

    extra = config["kv_connector_extra_config"]
    assert extra["document_kv.provider_factory"] == DOCUMENT_KV_NATIVE_PROVIDER_FACTORY


def test_document_kv_transfer_config_accepts_custom_handoff_source_factory():
    config = document_kv_transfer_config(
        handoff_source_factory="company_vllm_patch.document_kv_source:build_source"
    )

    extra = config["kv_connector_extra_config"]
    assert extra["document_kv.provider_factory"] == DOCUMENT_KV_NATIVE_PROVIDER_FACTORY
    assert (
        extra["document_kv.handoff_source_factory"]
        == "company_vllm_patch.document_kv_source:build_source"
    )


def test_document_kv_transfer_config_accepts_payload_cache_budget():
    config = document_kv_transfer_config(payload_cache_max_bytes=4096)

    extra = config["kv_connector_extra_config"]
    assert extra[DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY] == 4096


def test_document_kv_transfer_config_accepts_telemetry_jsonl_path():
    config = document_kv_transfer_config(telemetry_jsonl="/local_disk0/cachet/connector.jsonl")

    extra = config["kv_connector_extra_config"]
    assert extra[DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY] == "/local_disk0/cachet/connector.jsonl"


@pytest.mark.parametrize("value", [-1, True, "4096"])
def test_document_kv_transfer_config_rejects_invalid_payload_cache_budget(value):
    with pytest.raises((TypeError, ValueError), match="payload_cache_max_bytes"):
        document_kv_transfer_config(payload_cache_max_bytes=value)


@pytest.mark.parametrize("value", ["", " "])
def test_document_kv_transfer_config_rejects_invalid_telemetry_jsonl(value):
    with pytest.raises(ValueError, match="telemetry_jsonl"):
        document_kv_transfer_config(telemetry_jsonl=value)


def test_document_kv_transfer_config_json_is_cli_ready():
    payload = document_kv_transfer_config_json(
        kv_role="kv_producer",
    )

    decoded = json.loads(payload)

    assert decoded == document_kv_transfer_config(
        kv_role="kv_producer",
    )
    assert "\n" not in payload


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("kv_connector", {"kv_connector": "", "kv_connector_module_path": "mod"}),
        ("kv_connector_module_path", {"kv_connector": "Connector", "kv_connector_module_path": ""}),
        ("kv_role", {"kv_connector": "Connector", "kv_connector_module_path": "mod", "kv_role": " "}),
    ],
)
def test_document_kv_transfer_config_requires_connector_identity(field, kwargs):
    with pytest.raises(ValueError, match=field):
        document_kv_transfer_config(**kwargs)


def test_document_kv_transfer_config_rejects_invalid_provider_factory():
    with pytest.raises(ValueError, match="provider_factory"):
        document_kv_transfer_config(provider_factory="missing_attribute")


def test_document_kv_transfer_config_rejects_invalid_handoff_source_factory():
    with pytest.raises(ValueError, match="handoff_source_factory"):
        document_kv_transfer_config(handoff_source_factory="missing_attribute")


def test_document_kv_transfer_config_rejects_reserved_extra_config_keys():
    with pytest.raises(ValueError, match="document_kv"):
        document_kv_transfer_config(
            kv_connector="DocumentKVConnector",
            kv_connector_module_path="company_vllm_patch.document_kv_connector",
            extra_config={"document_kv.record_type": "override"},
        )


def test_document_kv_transfer_config_rejects_non_json_extra_config():
    with pytest.raises(TypeError, match="JSON-serializable"):
        document_kv_transfer_config(
            kv_connector="DocumentKVConnector",
            kv_connector_module_path="company_vllm_patch.document_kv_connector",
            extra_config={"bad": object()},
        )


def test_package_root_reexports_transfer_config_helpers():
    assert vllm_kv_injection.document_kv_transfer_config is document_kv_transfer_config
    assert vllm_kv_injection.document_kv_transfer_config_json is document_kv_transfer_config_json
    assert vllm_kv_injection.DOCUMENT_KV_CONNECTOR_CLASS == DOCUMENT_KV_CONNECTOR_CLASS
    assert vllm_kv_injection.DOCUMENT_KV_CONNECTOR_MODULE_PATH == DOCUMENT_KV_CONNECTOR_MODULE_PATH
    assert vllm_kv_injection.DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY == DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY


def test_main_writes_transfer_config_sidecar(tmp_path):
    output_json = tmp_path / "vllm-launch-config.json"

    exit_code = main(
        [
            "--kv-connector-module-path",
            "vllm_kv_injection.vllm_dynamic_connector",
            "--kv-role",
            "kv_consumer",
            "--provider-factory",
            "company_vllm_patch.document_kv_provider:build_provider",
            "--extra-config",
            "tenant=\"qa\"",
            "--extra-config",
            "max_ready_queue=16",
            "--telemetry-jsonl",
            "/local_disk0/cachet/connector.jsonl",
            "--output-json",
            str(output_json),
        ]
    )

    config = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert config == document_kv_transfer_config(
        kv_connector="DocumentKVConnector",
        kv_connector_module_path="vllm_kv_injection.vllm_dynamic_connector",
        kv_role="kv_consumer",
        extra_config={"tenant": "qa", "max_ready_queue": 16},
        provider_factory="company_vllm_patch.document_kv_provider:build_provider",
        telemetry_jsonl="/local_disk0/cachet/connector.jsonl",
    )


def test_main_prints_transfer_config_to_stdout(capsys):
    exit_code = main(
        [
            "--kv-connector",
            "DocumentKVConnector",
        ]
    )

    output = capsys.readouterr().out
    config = json.loads(output)

    assert exit_code == 0
    assert config["kv_connector"] == "DocumentKVConnector"
    assert config["kv_role"] == DOCUMENT_KV_DEFAULT_ROLE


def test_main_reports_invalid_extra_config(capsys):
    exit_code = main(
        [
            "--kv-connector-module-path",
            "company_vllm_patch.document_kv_connector",
            "--extra-config",
            "tenant=not-json",
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert record["error_type"] == "ValueError"
    assert "valid JSON" in record["error"]
