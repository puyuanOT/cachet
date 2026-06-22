import hashlib
import json
import os
import subprocess
import sys

import pytest

import restaurant_kv_serving.probe_fixtures as legacy_probe_fixtures
from document_kv_cache.engine_adapters import (
    PayloadMode,
    ServingBackend,
    build_engine_kv_connector_actions,
    build_engine_kv_injection_plan,
    engine_kv_connector_actions_to_record,
    read_engine_adapter_request_json,
    validate_engine_kv_connector_actions_record,
    view_engine_adapter_payload,
)
from document_kv_cache.engine_probe import read_engine_adapter_payload
from document_kv_cache.model_profiles import QWEN3_4B_INSTRUCT_PROFILE
from document_kv_cache.probe_fixtures import (
    DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES,
    DEFAULT_ENGINE_PROBE_FIXTURE_REQUEST_ID,
    ENGINE_PROBE_FIXTURE_RECORD_TYPE,
    ENGINE_PROBE_FIXTURE_SCHEMA_VERSION,
    EngineProbeFixtureConfig,
    engine_probe_fixture_result_to_record,
    write_qwen3_v1_engine_probe_fixture,
)
from vllm_kv_injection.vllm_native_provider import (
    document_kv_vllm_probe_layer_names,
)
from vllm_kv_injection.vllm_runtime_contract import (
    VLLMInstalledKVConnectorContract,
    VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS,
    VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
    installed_vllm_kv_connector_v1_contract_to_record,
)
from vllm_kv_injection.vllm_runtime_preflight import (
    document_kv_vllm_runtime_preflight_to_record,
    validate_document_kv_vllm_runtime_preflight_record,
)


def test_write_qwen3_v1_engine_probe_fixture_generates_valid_vllm_merged_bundle(tmp_path):
    result = write_qwen3_v1_engine_probe_fixture(
        EngineProbeFixtureConfig(
            output_dir=tmp_path,
            backend=ServingBackend.VLLM,
            payload_mode=PayloadMode.MERGED,
            tokens_per_segment=2,
            metadata={"fixture.owner": "native-adapter-test"},
        )
    )

    record = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    handoff = read_engine_adapter_request_json(result.handoff_json, expected_backend=ServingBackend.VLLM)
    plan = build_engine_kv_injection_plan(handoff, expected_backend=ServingBackend.VLLM)
    payload = read_engine_adapter_payload(result.payload_uri, expected_bytes=plan.total_bytes)
    actions = build_engine_kv_connector_actions(plan, payload)
    actions_record = engine_kv_connector_actions_to_record(actions)
    written_actions_record = json.loads(result.actions_json.read_text(encoding="utf-8"))
    layer_names_record = json.loads(result.vllm_layer_names_json.read_text(encoding="utf-8"))

    assert record == engine_probe_fixture_result_to_record(result)
    assert record["record_type"] == ENGINE_PROBE_FIXTURE_RECORD_TYPE
    assert record["schema_version"] == ENGINE_PROBE_FIXTURE_SCHEMA_VERSION
    assert record["request_id"] == DEFAULT_ENGINE_PROBE_FIXTURE_REQUEST_ID
    assert record["payload_mode"] == "merged"
    assert record["model_id"] == "qwen3:4b-instruct"
    assert record["layout_version"] == "qwen3-v1"
    assert record["storage_layout"] == "shared_key_value"
    assert record["shares_kv_storage"] is True
    assert record["bytes_per_token"] == QWEN3_4B_INSTRUCT_PROFILE.bytes_per_token("int8")
    assert plan.layout.num_query_heads == 32
    assert plan.layout.num_kv_heads == 8
    assert plan.layout.query_heads_per_kv_head == 4
    assert plan.total_tokens == 6
    assert plan.total_bytes == 6 * plan.layout.bytes_per_token
    assert plan.total_blocks == 1
    assert len(plan.segments) == 3
    assert [segment.chunk_type for segment in plan.segments] == [
        "document_static",
        "document_chunk",
        "document_chunk",
    ]
    assert result.actions_json.name == "qwen3-v1-fixture.actions.json"
    assert result.vllm_layer_names_json.name == DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES["vllm_layer_names"]
    assert record["uris"]["actions_json"] == result.actions_uri
    assert record["paths"]["actions_json"] == str(result.actions_json)
    assert record["sha256"]["actions_json"] == result.actions_sha256
    assert record["uris"]["vllm_layer_names_json"] == result.vllm_layer_names_uri
    assert record["paths"]["vllm_layer_names_json"] == str(result.vllm_layer_names_json)
    assert record["sha256"]["vllm_layer_names_json"] == result.vllm_layer_names_sha256
    assert hashlib.sha256(result.actions_json.read_bytes()).hexdigest() == result.actions_sha256
    assert (
        hashlib.sha256(result.vllm_layer_names_json.read_bytes()).hexdigest()
        == result.vllm_layer_names_sha256
    )
    assert hashlib.sha256(payload).hexdigest() == result.payload_sha256
    assert written_actions_record == actions_record
    assert layer_names_record == {"layer_names": list(document_kv_vllm_probe_layer_names(result.layout))}
    preflight_record = document_kv_vllm_runtime_preflight_to_record(
        layer_names_record["layer_names"],
        installed_contract=_matching_installed_vllm_contract_record(),
    )
    validate_document_kv_vllm_runtime_preflight_record(preflight_record)
    validate_engine_kv_connector_actions_record(actions_record)
    assert actions_record["reservation"]["total_tokens"] == 6
    assert [copy["payload_index"] for copy in actions_record["copies"]] == [None, None, None]


def test_write_qwen3_v1_engine_probe_fixture_generates_segmented_sglang_bundle(tmp_path):
    result = write_qwen3_v1_engine_probe_fixture(
        EngineProbeFixtureConfig(
            output_dir=tmp_path,
            backend="sglang",
            payload_mode="segmented",
            tokens_per_segment=1,
            chunk_ids=("body",),
            adapter_ids=("selection-lora",),
        )
    )

    handoff = read_engine_adapter_request_json(result.handoff_json, expected_backend=ServingBackend.SGLANG)
    plan = build_engine_kv_injection_plan(handoff, expected_backend=ServingBackend.SGLANG)
    payload = read_engine_adapter_payload(result.payload_uri, expected_bytes=plan.total_bytes)
    payload_segments = view_engine_adapter_payload(handoff, payload)
    actions = build_engine_kv_connector_actions(plan, payload_segments)
    actions_record = engine_kv_connector_actions_to_record(actions)
    written_actions_record = json.loads(result.actions_json.read_text(encoding="utf-8"))

    assert result.adapter_request.backend == ServingBackend.SGLANG
    assert result.adapter_request.payload_mode == PayloadMode.SEGMENTED
    assert result.vllm_layer_names_uri is None
    assert result.vllm_layer_names_json is None
    assert result.vllm_layer_names_sha256 is None
    assert plan.adapter_ids == ("selection-lora",)
    assert plan.total_tokens == 2
    assert len(payload_segments) == 2
    assert [len(segment) for segment in payload_segments] == [plan.layout.bytes_per_token] * 2
    assert [copy["payload_index"] for copy in actions_record["copies"]] == [0, 1]
    assert actions_record["bind"]["metadata"]["engine.connector_package"] == "sglang"
    assert written_actions_record == actions_record
    validate_engine_kv_connector_actions_record(actions_record, expected_backend=ServingBackend.SGLANG)


def test_qwen3_v1_engine_probe_fixture_payload_is_deterministic(tmp_path):
    first = write_qwen3_v1_engine_probe_fixture(
        EngineProbeFixtureConfig(output_dir=tmp_path / "first", tokens_per_segment=1)
    )
    second = write_qwen3_v1_engine_probe_fixture(
        EngineProbeFixtureConfig(output_dir=tmp_path / "second", tokens_per_segment=1)
    )

    assert first.payload_sha256 == second.payload_sha256
    assert first.pack_sha256 == second.pack_sha256
    assert first.actions_sha256 == second.actions_sha256


def test_qwen3_v1_engine_probe_fixture_cli_writes_manifest(tmp_path):
    output_dir = tmp_path / "fixture"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.probe_fixtures",
            "--output-dir",
            str(output_dir),
            "--backend",
            "vllm",
            "--payload-mode",
            "merged",
            "--tokens-per-segment",
            "1",
            "--chunk-id",
            "section-1",
            "--print-json",
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
    )

    stdout_record = json.loads(completed.stdout)
    manifest_record = json.loads((output_dir / "qwen3-v1-fixture.manifest.json").read_text(encoding="utf-8"))
    assert stdout_record == manifest_record
    assert stdout_record["total_tokens"] == 2
    assert stdout_record["payload_mode"] == "merged"
    assert (output_dir / "qwen3-v1-fixture.actions.json").exists()
    assert (output_dir / "vllm-layer-names.json").exists()
    assert stdout_record["uris"]["actions_json"].endswith("/qwen3-v1-fixture.actions.json")
    assert stdout_record["uris"]["vllm_layer_names_json"].endswith("/vllm-layer-names.json")


def test_qwen3_v1_engine_probe_fixture_accepts_relative_output_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = write_qwen3_v1_engine_probe_fixture(
        EngineProbeFixtureConfig(output_dir="relative-fixture", tokens_per_segment=1)
    )
    handoff = read_engine_adapter_request_json(result.handoff_json, expected_backend=ServingBackend.VLLM)
    payload_uri = handoff["payload_source"]["uri"]

    assert result.pack_path.is_absolute()
    assert result.payload_path.is_absolute()
    assert result.actions_json.is_absolute()
    assert payload_uri == result.payload_uri
    assert payload_uri.startswith(str(tmp_path))
    assert (tmp_path / "relative-fixture" / "qwen3-v1-fixture.kvpack").exists()
    assert (tmp_path / "relative-fixture" / "qwen3-v1-fixture.actions.json").exists()
    assert (tmp_path / "relative-fixture" / "vllm-layer-names.json").exists()


def test_qwen3_v1_engine_probe_fixture_rejects_reserved_metadata(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        EngineProbeFixtureConfig(
            output_dir=tmp_path,
            metadata={"document_kv.bad": "reserved"},
        )


def test_legacy_probe_fixtures_reexports_document_fixture_api():
    import document_kv_cache.probe_fixtures as public_probe_fixtures

    assert legacy_probe_fixtures.__all__ == public_probe_fixtures.__all__
    assert legacy_probe_fixtures.EngineProbeFixtureConfig is public_probe_fixtures.EngineProbeFixtureConfig
    assert (
        legacy_probe_fixtures.write_qwen3_v1_engine_probe_fixture
        is public_probe_fixtures.write_qwen3_v1_engine_probe_fixture
    )


def _matching_installed_vllm_contract_record() -> dict:
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
