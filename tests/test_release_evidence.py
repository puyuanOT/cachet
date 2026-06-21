import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import document_kv_cache.release_evidence as public_release_evidence
import restaurant_kv_serving.release_evidence as legacy_release_evidence
from document_kv_cache.engine_adapters import (
    EngineKVBindAction,
    EngineKVConnectorActions,
    EngineKVConnectorProbeResult,
    EngineKVReleaseAction,
    EngineKVReservationAction,
    EngineKVSegmentCopyAction,
    PayloadMode,
    ServingBackend,
)
from document_kv_cache.engine_adapters import (
    engine_kv_connector_actions_to_record,
    engine_kv_connector_probe_result_to_record,
)
from document_kv_cache.engine_probe import (
    ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE,
    ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION,
)
from document_kv_cache.benchmark_runner import BENCHMARK_RUN_RECORD_TYPE
from document_kv_cache.release_evidence import (
    RELEASE_EVIDENCE_RECORD_TYPE,
    RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE,
    REQUIRED_ENGINE_PROBE_BACKENDS,
    ReleaseEvidence,
    ReleaseEvidenceArtifactSource,
    ReleaseEvidenceInputFileStatus,
    ReleaseEvidenceInputStatus,
    evaluate_release_evidence,
    evaluate_release_evidence_files,
    inspect_release_evidence_input_files,
    release_evidence_input_status_to_record,
    release_evidence_to_record,
)
from document_kv_cache.model_profiles import layout_for_model
from document_kv_cache.serving_env import serving_environment_profile
from document_kv_cache.storage_benchmark import STORAGE_BENCHMARK_RECORD_TYPE


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records():
    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM),
            _actions_record(ServingBackend.SGLANG),
        ),
    )
    record = release_evidence_to_record(evidence)

    assert evidence.ok
    assert evidence.v1_benchmark_ok
    assert evidence.storage_benchmark_ok
    assert evidence.engine_probe_backends == REQUIRED_ENGINE_PROBE_BACKENDS
    assert record == {
        "record_type": RELEASE_EVIDENCE_RECORD_TYPE,
        "ok": True,
        "v1_benchmark_ok": True,
        "storage_benchmark_ok": True,
        "engine_probe_backends": ["vllm", "sglang"],
        "missing_engine_probe_backends": [],
        "duplicate_engine_probe_backends": [],
        "invalid_engine_probe_records": [],
        "engine_action_backends": ["vllm", "sglang"],
        "missing_engine_action_backends": [],
        "duplicate_engine_action_backends": [],
        "invalid_engine_action_records": [],
        "artifact_sources": [],
        "issues": [],
    }


def test_evaluate_release_evidence_reports_missing_and_invalid_artifacts():
    invalid_probe_record = {**_probe_record(ServingBackend.VLLM), "bound": False}

    evidence = evaluate_release_evidence(
        _v1_record(ok=False, hardware_target="aws-g6", model_id="qwen3.5:4b"),
        _storage_record(ok=False, uc_volume_is_real=False),
        engine_probe_records=(invalid_probe_record,),
    )

    assert not evidence.ok
    assert not evidence.v1_benchmark_ok
    assert not evidence.storage_benchmark_ok
    assert evidence.engine_probe_backends == ()
    assert evidence.missing_engine_probe_backends == ("vllm", "sglang")
    assert evidence.duplicate_engine_probe_backends == ()
    assert any("hardware_target" in issue for issue in evidence.issues)
    assert any("model_id" in issue for issue in evidence.issues)
    assert any(issue.startswith("v1 benchmark evidence") for issue in evidence.issues)
    assert any(issue.startswith("storage benchmark evidence") for issue in evidence.issues)
    assert any(issue.startswith("invalid engine probe record: vllm") for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_summary_only_stubs():
    evidence = evaluate_release_evidence(
        {
            "record_type": BENCHMARK_RUN_RECORD_TYPE,
            "suite": {"hardware_target": "aws-g5", "model_id": "qwen3:4b-instruct"},
            "v1_evidence": {"ok": True, "required_datasets": ["biography", "hotpotqa", "musique", "niah"]},
        },
        {
            "record_type": STORAGE_BENCHMARK_RECORD_TYPE,
            "readers": ["memory", "disk", "unity_catalog"],
            "release_storage_evidence": {
                "ok": True,
                "required_readers": ["memory", "disk", "unity_catalog"],
                "uc_volume_is_real": True,
            },
        },
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("measurements must be a sequence" in issue for issue in evidence.issues)
    assert any("report_rows must be a sequence" in issue for issue in evidence.issues)
    assert any("comparisons must be a sequence" in issue for issue in evidence.issues)
    assert any("results must be a sequence" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_wrong_v1_record_type():
    v1_record = _v1_record(ok=True)
    v1_record["record_type"] = "document_kv.benchmark_summary.v1"

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("v1 benchmark record_type" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_malformed_v1_suite_metadata():
    v1_record = _v1_record(ok=True)
    v1_record["suite"] = {
        **v1_record["suite"],
        "suite_id": "",
        "datasets": ["biography", "hotpotqa"],
        "examples": 0,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("suite_id must be non-empty" in issue for issue in evidence.issues)
    assert any("suite datasets must match" in issue for issue in evidence.issues)
    assert any("suite examples must be a positive integer" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_v1_suite_example_count_mismatch():
    v1_record = _v1_record(ok=True)
    v1_record["suite"] = {**v1_record["suite"], "examples": 3}

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("suite examples must match unique measurement examples" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_unpaired_measurement_examples():
    v1_record = _v1_record(ok=True)
    v1_record["suite"] = {**v1_record["suite"], "examples": 5}
    v1_record["measurements"][0] = {
        **v1_record["measurements"][0],
        "example_id": "biography-baseline-only",
    }
    v1_record["measurements"][1] = {
        **v1_record["measurements"][1],
        "example_id": "biography-cache-only",
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any(
        "biography:biography-baseline-only missing required arms: document_kv_cache" in issue
        for issue in evidence.issues
    )
    assert any(
        "biography:biography-cache-only missing required arms: baseline_prefill" in issue
        for issue in evidence.issues
    )


def test_evaluate_release_evidence_rejects_inconsistent_measurement_expected_answers():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"][1] = {
        **v1_record["measurements"][1],
        "expected_answer": "Grace Hopper",
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any(
        "biography:biography-1 expected_answer must be consistent across arms" in issue
        for issue in evidence.issues
    )


def test_evaluate_release_evidence_rejects_unknown_engine_version():
    probe_record = {**_probe_record(ServingBackend.VLLM), "engine_version": "unknown"}

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(probe_record, _probe_record(ServingBackend.SGLANG)),
    )

    assert not evidence.ok
    assert any("engine_version" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_non_profile_engine_version():
    probe_record = {**_probe_record(ServingBackend.VLLM), "engine_version": "0.0.0"}

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(probe_record, _probe_record(ServingBackend.SGLANG)),
    )

    assert not evidence.ok
    assert evidence.engine_probe_backends == ("sglang",)
    assert evidence.missing_engine_probe_backends == ("vllm",)
    assert any("engine_version must match the backend serving profile" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_wrong_serving_profile_metadata():
    vllm_record = _probe_record(ServingBackend.VLLM)
    vllm_record["metadata"] = {
        **vllm_record["metadata"],
        ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE: "vllm-nightly",
    }
    sglang_record = _probe_record(ServingBackend.SGLANG)
    sglang_record["metadata"] = {
        **sglang_record["metadata"],
        ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION: "0.0.0",
    }

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(vllm_record, sglang_record),
    )

    assert not evidence.ok
    assert evidence.engine_probe_backends == ()
    assert evidence.missing_engine_probe_backends == ("vllm", "sglang")
    assert any("serving_engine_package" in issue for issue in evidence.issues)
    assert any("serving_engine_version" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_duplicate_valid_engine_probe_backend():
    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            {**_probe_record(ServingBackend.VLLM), "request_id": "vllm-probe-2"},
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert evidence.engine_probe_backends == REQUIRED_ENGINE_PROBE_BACKENDS
    assert evidence.missing_engine_probe_backends == ()
    assert evidence.duplicate_engine_probe_backends == ("vllm",)
    assert any("duplicate engine probe backends: vllm" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_probe_action_request_mismatch():
    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM, request_id="different-vllm-request"),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert evidence.engine_probe_backends == REQUIRED_ENGINE_PROBE_BACKENDS
    assert evidence.engine_action_backends == REQUIRED_ENGINE_PROBE_BACKENDS
    assert any("request_id mismatch between engine probe and connector actions" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_probe_action_token_and_byte_mismatch():
    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM, total_tokens=2),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("copied_tokens mismatch between engine probe and connector actions" in issue for issue in evidence.issues)
    assert any("copied_bytes mismatch between engine probe and connector actions" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_probe_action_payload_mode_mismatch():
    segmented_actions = _actions_record(ServingBackend.VLLM)
    segmented_actions["copies"][0]["payload_index"] = 0

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            segmented_actions,
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert evidence.missing_engine_action_backends == ("vllm",)
    assert any("Engine KV action payload_mode must match" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_probe_action_payload_mode_outside_adapter_contract():
    segmented_probe = _probe_record(ServingBackend.VLLM)
    segmented_probe["payload_mode"] = "segmented"
    segmented_actions = _actions_record(ServingBackend.VLLM)
    segmented_actions["copies"][0]["payload_index"] = 0

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(
            segmented_probe,
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            segmented_actions,
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert evidence.missing_engine_probe_backends == ("vllm",)
    assert evidence.missing_engine_action_backends == ("vllm",)
    assert any("Engine KV probe payload_mode must match" in issue for issue in evidence.issues)
    assert any("Engine KV action payload_mode must match" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_probe_action_layout_mismatch():
    action_record = _actions_record(ServingBackend.VLLM)
    action_record["reservation"]["layout"] = {
        **action_record["reservation"]["layout"],
        "lora_id": "selection-lora",
    }

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            action_record,
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("layout mismatch between engine probe and connector actions" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_missing_or_non_qwen3_gqa_engine_probe_layout():
    missing_layout = _probe_record(ServingBackend.VLLM)
    missing_layout.pop("layout")
    wrong_layout = _probe_record(ServingBackend.SGLANG)
    wrong_layout["layout"] = {
        **wrong_layout["layout"],
        "head_size": 64,
        "kv_stride_bytes": 64,
        "bytes_per_token": 36 * 8 * 64 * 2,
    }
    wrong_layout["copied_bytes"] = wrong_layout["layout"]["bytes_per_token"]

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(missing_layout, wrong_layout),
    )

    assert not evidence.ok
    assert evidence.engine_probe_backends == ()
    assert evidence.missing_engine_probe_backends == ("vllm", "sglang")
    assert any("layout must be a mapping" in issue for issue in evidence.issues)
    assert any("must use the V1 Qwen3 GQA geometry" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_engine_probe_layout_without_storage_layout():
    stale_layout = _probe_record(ServingBackend.VLLM)
    stale_layout["layout"] = {
        key: value for key, value in stale_layout["layout"].items() if key != "storage_layout"
    }

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(stale_layout, _probe_record(ServingBackend.SGLANG)),
    )

    assert not evidence.ok
    assert evidence.missing_engine_probe_backends == ("vllm",)
    assert any("storage_layout" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_non_shared_v1_storage_layout():
    wrong_storage = _probe_record(ServingBackend.VLLM)
    wrong_storage["layout"] = {
        **wrong_storage["layout"],
        "shares_kv_storage": False,
        "storage_layout": "separate_key_value",
    }

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(wrong_storage, _probe_record(ServingBackend.SGLANG)),
    )

    assert not evidence.ok
    assert evidence.missing_engine_probe_backends == ("vllm",)
    assert any("qwen3-v1 layout requires shared K/V storage" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_accepts_backend_block_size_and_lora_layout_variants():
    backend_layout = layout_for_model("qwen3:4b-instruct", block_size=32, lora_id="selection-lora")

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM, layout=backend_layout),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM, layout=backend_layout),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert evidence.ok


def test_evaluate_release_evidence_accepts_padded_qwen3_kv_stride():
    padded_layout = layout_for_model("qwen3:4b-instruct", kv_stride_bytes=256)

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM, layout=padded_layout),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM, layout=padded_layout),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert padded_layout.bytes_per_token == 36 * 8 * 256 * 2
    assert evidence.ok


def test_evaluate_release_evidence_rejects_wrong_comparison_arms_and_missing_quality_delta():
    v1_record = _v1_record(ok=True)
    v1_record["comparisons"][0] = {
        **v1_record["comparisons"][0],
        "baseline_arm_id": "other_baseline",
        "cache_arm_id": "other_cache",
        "exact_match_delta": None,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("baseline_arm_id" in issue for issue in evidence.issues)
    assert any("cache_arm_id" in issue for issue in evidence.issues)
    assert any("exact_match_delta" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_unexpected_v1_evidence_arms():
    v1_record = _v1_record(ok=True)
    v1_record["v1_evidence"]["unexpected_arms"] = ["experimental_cache"]

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert "v1 benchmark evidence unexpected_arms must be empty" in evidence.issues


def test_evaluate_release_evidence_rejects_duplicate_v1_evidence_identities():
    v1_record = _v1_record(ok=True)
    v1_record["v1_evidence"]["ok"] = False
    v1_record["v1_evidence"]["duplicate_report_rows"] = ["biography:baseline_prefill"]
    v1_record["v1_evidence"]["issues"] = ["duplicate report rows: biography:baseline_prefill"]

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert "v1 benchmark evidence duplicate_report_rows must be empty" in evidence.issues
    assert "v1 benchmark evidence: duplicate report rows: biography:baseline_prefill" in evidence.issues


def test_evaluate_release_evidence_allows_legacy_v1_evidence_without_duplicate_identity_fields():
    v1_record = _v1_record(ok=True)
    for field_name in (
        "duplicate_required_datasets",
        "duplicate_report_rows",
        "duplicate_comparisons",
    ):
        v1_record["v1_evidence"].pop(field_name)

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert evidence.v1_benchmark_ok
    assert not any("duplicate_" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_non_numeric_comparison_metrics():
    v1_record = _v1_record(ok=True)
    v1_record["comparisons"][0] = {
        **v1_record["comparisons"][0],
        "ttft_speedup": "fast",
        "time_to_completion_speedup": 0.0,
        "exact_match_delta": float("nan"),
        "answer_found_delta": True,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("ttft_speedup must be a positive finite number" in issue for issue in evidence.issues)
    assert any(
        "time_to_completion_speedup must be a positive finite number" in issue
        for issue in evidence.issues
    )
    assert any("exact_match_delta must be a finite number" in issue for issue in evidence.issues)
    assert any("answer_found_delta must be a finite number" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_out_of_bounds_comparison_quality_deltas():
    v1_record = _v1_record(ok=True)
    v1_record["comparisons"][0] = {
        **v1_record["comparisons"][0],
        "exact_match_delta": 1.1,
    }
    v1_record["comparisons"][1] = {
        **v1_record["comparisons"][1],
        "answer_found_delta": -1.1,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any(
        "v1 benchmark comparison biography exact_match_delta must be a finite number between -1 and 1" in issue
        for issue in evidence.issues
    )
    assert any(
        "v1 benchmark comparison hotpotqa answer_found_delta must be a finite number between -1 and 1" in issue
        for issue in evidence.issues
    )


def test_evaluate_release_evidence_rejects_duplicate_summary_and_malformed_v1_identities():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"].append(
        {
            **v1_record["measurements"][0],
            "dataset": ["biography"],
            "arm_id": "other_arm",
        }
    )
    v1_record["report_rows"].append({**v1_record["report_rows"][0]})
    v1_record["report_rows"].append(
        {
            **v1_record["report_rows"][0],
            "dataset": "unsupported_dataset",
            "arm_id": "other_arm",
        }
    )
    v1_record["comparisons"].append({**v1_record["comparisons"][0]})
    v1_record["comparisons"].append(
        {
            **v1_record["comparisons"][0],
            "dataset": "unsupported_dataset",
        }
    )

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("measurement 8 has unsupported dataset ['biography']" in issue for issue in evidence.issues)
    assert any("measurement 8 has unsupported arm_id 'other_arm'" in issue for issue in evidence.issues)
    assert any("report_rows has duplicate row for biography:baseline_prefill" in issue for issue in evidence.issues)
    assert any("report_rows[9] has unsupported dataset 'unsupported_dataset'" in issue for issue in evidence.issues)
    assert any("report_rows[9] has unsupported arm_id 'other_arm'" in issue for issue in evidence.issues)
    assert any("comparisons has duplicate comparison for biography" in issue for issue in evidence.issues)
    assert any("comparison 5 has unsupported dataset 'unsupported_dataset'" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_malformed_measurement_quality_flags():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"][0] = {
        **v1_record["measurements"][0],
        "exact_match": "yes",
    }
    v1_record["measurements"][1] = {
        **v1_record["measurements"][1],
        "answer_found": 1,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("exact_match must be boolean" in issue for issue in evidence.issues)
    assert any("answer_found must be boolean" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_incorrect_measurement_quality_labels():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"][0] = {
        **v1_record["measurements"][0],
        "exact_match": False,
    }
    v1_record["measurements"][1] = {
        **v1_record["measurements"][1],
        "answer_found": False,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("biography:baseline_prefill exact_match must match output_text" in issue for issue in evidence.issues)
    assert any("biography:document_kv_cache answer_found must match output_text" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_malformed_measurement_trace_fields():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"][0] = {
        **v1_record["measurements"][0],
        "example_id": "",
    }
    v1_record["measurements"][1] = {
        **v1_record["measurements"][1],
        "output_text": {"answer": "Ada Lovelace"},
    }
    v1_record["measurements"][2] = {
        **v1_record["measurements"][2],
        "expected_answer": "",
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("biography:baseline_prefill example_id must be non-empty" in issue for issue in evidence.issues)
    assert any("biography:document_kv_cache output_text must be a string" in issue for issue in evidence.issues)
    assert any("hotpotqa:baseline_prefill expected_answer must be non-empty" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_malformed_report_quality_rates():
    v1_record = _v1_record(ok=True)
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "exact_match_rate": "perfect",
    }
    v1_record["report_rows"][1] = {
        **v1_record["report_rows"][1],
        "answer_found_rate": 1.2,
    }
    v1_record["report_rows"][2] = {
        **v1_record["report_rows"][2],
        "answer_found_rate": True,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any(
        "biography:baseline_prefill exact_match_rate must be a finite rate between 0 and 1" in issue
        for issue in evidence.issues
    )
    assert any(
        "biography:document_kv_cache answer_found_rate must be a finite rate between 0 and 1" in issue
        for issue in evidence.issues
    )
    assert any(
        "hotpotqa:baseline_prefill answer_found_rate must be a finite rate between 0 and 1" in issue
        for issue in evidence.issues
    )


def test_evaluate_release_evidence_requires_both_report_quality_rates():
    v1_record = _v1_record(ok=True)
    v1_record["report_rows"][0].pop("exact_match_rate")
    v1_record["report_rows"][1].pop("answer_found_rate")

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any(
        "biography:baseline_prefill must include exact_match_rate" in issue
        for issue in evidence.issues
    )
    assert any(
        "biography:document_kv_cache must include answer_found_rate" in issue
        for issue in evidence.issues
    )


def test_evaluate_release_evidence_allows_repeated_raw_measurements():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"].append({**v1_record["measurements"][0]})
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "requests": 2,
        "ttft": {**v1_record["report_rows"][0]["ttft"], "count": 2},
        "time_to_completion": {**v1_record["report_rows"][0]["time_to_completion"], "count": 2},
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert evidence.ok


def test_evaluate_release_evidence_rejects_stub_measurement_rows():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"] = [{}]

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("unsupported dataset" in issue for issue in evidence.issues)
    assert any("example_id" in issue for issue in evidence.issues)
    assert any("output_text" in issue for issue in evidence.issues)
    assert any("expected_answer" in issue for issue in evidence.issues)
    assert any("exact_match" in issue for issue in evidence.issues)
    assert any("answer_found" in issue for issue in evidence.issues)
    assert any("prompt_tokens" in issue for issue in evidence.issues)
    assert any("missing required dataset/arm pairs" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_requires_prompt_token_context_metadata():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"][0] = {
        **v1_record["measurements"][0],
        "metadata": {},
    }
    v1_record["measurements"][1] = {
        **v1_record["measurements"][1],
        "metadata": {
            "prompt_text_mode": "logical",
            "prompt_token_source": "logical",
            "logical_prompt_tokens": "1024",
            "runtime_prompt_tokens": "1024",
        },
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("metadata.prompt_text_mode" in issue for issue in evidence.issues)
    assert any("metadata.prompt_token_source" in issue for issue in evidence.issues)
    assert any("metadata.logical_prompt_tokens" in issue for issue in evidence.issues)
    assert any("metadata.runtime_prompt_tokens" in issue for issue in evidence.issues)
    assert any("cache runtime_prompt_tokens must be smaller than logical_prompt_tokens" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_wrong_prompt_text_mode_for_arm():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"][0] = {
        **v1_record["measurements"][0],
        "metadata": {
            **v1_record["measurements"][0]["metadata"],
            "prompt_text_mode": "runtime",
        },
    }
    v1_record["measurements"][1] = {
        **v1_record["measurements"][1],
        "metadata": {
            **v1_record["measurements"][1]["metadata"],
            "prompt_text_mode": "logical",
        },
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any(
        "biography:baseline_prefill baseline metadata.prompt_text_mode must be 'logical'" in issue
        for issue in evidence.issues
    )
    assert any(
        "biography:document_kv_cache cache metadata.prompt_text_mode must be 'runtime'" in issue
        for issue in evidence.issues
    )


def test_evaluate_release_evidence_rejects_prompt_tokens_that_do_not_match_arm_context():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"][0] = {
        **v1_record["measurements"][0],
        "prompt_tokens": 128,
    }
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "prompt_tokens_mean": 128.0,
    }
    v1_record["measurements"][1] = {
        **v1_record["measurements"][1],
        "prompt_tokens": 1024,
    }
    v1_record["report_rows"][1] = {
        **v1_record["report_rows"][1],
        "prompt_tokens_mean": 1024.0,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any(
        "biography:baseline_prefill baseline prompt_tokens must equal metadata.logical_prompt_tokens"
        in issue
        for issue in evidence.issues
    )
    assert any(
        "biography:document_kv_cache cache prompt_tokens must equal metadata.runtime_prompt_tokens"
        in issue
        for issue in evidence.issues
    )


def test_evaluate_release_evidence_rejects_zero_token_volume_and_all_error_summary_rows():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"][0] = {
        **v1_record["measurements"][0],
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "errors": 1,
        "prompt_tokens_mean": 0.0,
        "completion_tokens_mean": 0.0,
        "output_tokens_per_second": 0.0,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("prompt_tokens must be a positive integer" in issue for issue in evidence.issues)
    assert any("completion_tokens must be a positive integer" in issue for issue in evidence.issues)
    assert any("must include at least one successful request" in issue for issue in evidence.issues)
    assert any("prompt_tokens_mean must be positive" in issue for issue in evidence.issues)
    assert any("completion_tokens_mean must be positive" in issue for issue in evidence.issues)
    assert any("output_tokens_per_second must be positive" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_report_row_count_mismatch():
    v1_record = _v1_record(ok=True)
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "requests": 2,
        "errors": 1,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("requests must match measurements" in issue for issue in evidence.issues)
    assert any("errors must match measurements" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_report_row_aggregate_mismatch():
    v1_record = _v1_record(ok=True)
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "prompt_tokens_mean": 1.0,
        "completion_tokens_mean": 1.0,
        "output_tokens_per_second": 99.0,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("prompt_tokens_mean must match measurements" in issue for issue in evidence.issues)
    assert any("completion_tokens_mean must match measurements" in issue for issue in evidence.issues)
    assert any("output_tokens_per_second must match measurements" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_report_row_quality_rate_mismatch():
    v1_record = _v1_record(ok=True)
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "exact_match_rate": 0.0,
        "answer_found_rate": 0.0,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("exact_match_rate must match measurements" in issue for issue in evidence.issues)
    assert any("answer_found_rate must match measurements" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_comparison_report_row_mismatch():
    v1_record = _v1_record(ok=True)
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "exact_match_rate": 1.0,
    }
    v1_record["report_rows"][1] = {
        **v1_record["report_rows"][1],
        "exact_match_rate": 1.0,
    }
    v1_record["comparisons"][0] = {
        **v1_record["comparisons"][0],
        "ttft_speedup": 2.0,
        "time_to_completion_speedup": 2.0,
        "exact_match_delta": 1.0,
        "answer_found_delta": 1.0,
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("ttft_speedup must match report rows" in issue for issue in evidence.issues)
    assert any("time_to_completion_speedup must match report rows" in issue for issue in evidence.issues)
    assert any("exact_match_delta must match report rows" in issue for issue in evidence.issues)
    assert any("answer_found_delta must match report rows" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_comparison_speedup_for_zero_report_p50():
    v1_record = _v1_record(ok=True)
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "ttft": {"count": 1, "mean": 0.0, "p50": 0.0, "p95": 0.0},
    }
    v1_record["report_rows"][1] = {
        **v1_record["report_rows"][1],
        "ttft": {"count": 1, "mean": 0.0, "p50": 0.0, "p95": 0.0},
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("ttft_speedup cannot be computed from non-positive report row p50" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_throughput_when_measurements_have_zero_total_time():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"][0] = {
        **v1_record["measurements"][0],
        "ttft_seconds": 0.0,
        "time_to_completion_seconds": 0.0,
    }
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "ttft": {"count": 1, "mean": 0.0, "p50": 0.0, "p95": 0.0},
        "time_to_completion": {"count": 1, "mean": 0.0, "p50": 0.0, "p95": 0.0},
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any(
        "output_tokens_per_second must be absent when measurements have zero total" in issue
        for issue in evidence.issues
    )


def test_evaluate_release_evidence_rejects_latency_count_mismatch():
    v1_record = _v1_record(ok=True)
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "ttft": {**v1_record["report_rows"][0]["ttft"], "count": 2},
        "time_to_completion": {**v1_record["report_rows"][0]["time_to_completion"], "count": 0},
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("ttft count must match successful measurements" in issue for issue in evidence.issues)
    assert any("time_to_completion count must be positive" in issue for issue in evidence.issues)
    assert any(
        "time_to_completion count must match successful measurements" in issue
        for issue in evidence.issues
    )


def test_evaluate_release_evidence_rejects_latency_summary_mismatch():
    v1_record = _v1_record(ok=True)
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "ttft": {"count": 1, "mean": 1.5, "p50": 1.5, "p95": 1.5},
        "time_to_completion": {"count": 1, "mean": 2.5, "p50": 2.5, "p95": 2.5},
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("ttft mean must match measurements" in issue for issue in evidence.issues)
    assert any("ttft p50 must match measurements" in issue for issue in evidence.issues)
    assert any(
        "time_to_completion p95 must match measurements" in issue
        for issue in evidence.issues
    )


def test_evaluate_release_evidence_rejects_impossible_latency_measurements_and_summaries():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"][0] = {
        **v1_record["measurements"][0],
        "ttft_seconds": 2.0,
        "time_to_completion_seconds": 1.0,
    }
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "ttft": {"p50": 2.0, "p95": 1.0},
        "time_to_completion": {"p50": 1.0, "p95": False},
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("time_to_completion_seconds must be greater than or equal to ttft_seconds" in issue for issue in evidence.issues)
    assert any("ttft p95 must be greater than or equal to p50" in issue for issue in evidence.issues)
    assert any("time_to_completion p95 must be a non-negative finite number" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_non_finite_latency_values():
    v1_record = _v1_record(ok=True)
    v1_record["measurements"][0] = {
        **v1_record["measurements"][0],
        "ttft_seconds": float("nan"),
        "time_to_completion_seconds": float("inf"),
    }
    v1_record["report_rows"][0] = {
        **v1_record["report_rows"][0],
        "ttft": {"p50": float("inf"), "p95": float("inf")},
    }
    v1_record["report_rows"][1] = {
        **v1_record["report_rows"][1],
        "time_to_completion": {"p50": -0.01, "p95": "slow"},
    }

    evidence = evaluate_release_evidence(
        v1_record,
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("ttft_seconds must be a non-negative finite number" in issue for issue in evidence.issues)
    assert any("time_to_completion_seconds must be a non-negative finite number" in issue for issue in evidence.issues)
    assert any("ttft p50 must be a non-negative finite number" in issue for issue in evidence.issues)
    assert any("ttft p95 must be a non-negative finite number" in issue for issue in evidence.issues)
    assert any("time_to_completion p50 must be a non-negative finite number" in issue for issue in evidence.issues)
    assert any("time_to_completion p95 must be a non-negative finite number" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_minimal_storage_rows_and_missing_uc_root():
    storage_record = _storage_record(ok=True)
    storage_record.pop("uc_volume_root")
    storage_record["release_storage_evidence"].pop("uc_volume_root")
    storage_record["release_storage_evidence"]["require_real_uc_volume"] = False
    storage_record["results"] = [{"reader_id": "memory", "errors": 0}]

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        storage_record,
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("uc_volume_root" in issue for issue in evidence.issues)
    assert any("require a real UC Volume" in issue for issue in evidence.issues)
    assert any("total_reads" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_rejects_impossible_storage_latency_rows():
    storage_record = _storage_record(ok=True)
    storage_record["results"][0] = {
        **storage_record["results"][0],
        "wall_seconds": float("nan"),
        "latency_p50_seconds": float("inf"),
        "latency_p95_seconds": float("inf"),
        "throughput_bytes_per_second": float("inf"),
    }
    storage_record["results"][1] = {
        **storage_record["results"][1],
        "latency_p50_seconds": 0.02,
        "latency_p95_seconds": 0.01,
    }
    storage_record["results"][2] = {
        **storage_record["results"][2],
        "latency_p50_seconds": -0.01,
        "latency_p95_seconds": "slow",
    }

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        storage_record,
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("wall_seconds must be a positive finite number" in issue for issue in evidence.issues)
    assert any("latency p50 must be a non-negative finite number" in issue for issue in evidence.issues)
    assert any("latency p95 must be a non-negative finite number" in issue for issue in evidence.issues)
    assert any("latency p95 must be greater than or equal to p50" in issue for issue in evidence.issues)
    assert any("throughput must be a positive finite number" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_accepts_storage_readers_in_any_order():
    storage_record = _storage_record(ok=True)
    storage_record["readers"] = ["disk", "memory", "unity_catalog"]
    storage_record["release_storage_evidence"]["required_readers"] = ["unity_catalog", "disk", "memory"]

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        storage_record,
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert evidence.ok


@pytest.mark.parametrize(
    "readers",
    (
        ["memory", "disk", "disk"],
        ["memory", "disk"],
        ["memory", "disk", "unity_catalog", "memory"],
    ),
)
def test_evaluate_release_evidence_rejects_duplicate_or_missing_storage_readers(readers):
    storage_record = _storage_record(ok=True)
    storage_record["readers"] = readers
    storage_record["release_storage_evidence"]["required_readers"] = readers

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        storage_record,
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("release readers" in issue or "Memory, Disk, and Unity Catalog" in issue for issue in evidence.issues)


@pytest.mark.parametrize(
    ("mutate_results", "expected_issue"),
    [
        (
            lambda results: results.append({**results[0]}),
            "storage benchmark results duplicate readers: memory",
        ),
        (
            lambda results: results.append({**results[0], "reader_id": "object_store"}),
            "storage benchmark reader object_store is not a supported release reader",
        ),
        (
            lambda results: results[0].__setitem__("reader_id", ""),
            "storage benchmark results[0].reader_id must be a supported release reader",
        ),
    ],
)
def test_evaluate_release_evidence_rejects_invalid_storage_result_readers(mutate_results, expected_issue):
    storage_record = _storage_record(ok=True)
    mutate_results(storage_record["results"])

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        storage_record,
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        engine_action_records=(
            _actions_record(ServingBackend.VLLM),
            _actions_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert expected_issue in evidence.issues


@pytest.mark.parametrize(
    "uc_volume_root",
    (
        "/Volumes/catalog/schema/volume/../secret",
        "/Volumes/../../etc/passwd",
    ),
)
def test_evaluate_release_evidence_rejects_traversing_uc_volume_roots(uc_volume_root):
    storage_record = _storage_record(ok=True)
    storage_record["uc_volume_root"] = uc_volume_root
    storage_record["release_storage_evidence"]["uc_volume_root"] = uc_volume_root

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        storage_record,
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
    )

    assert not evidence.ok
    assert any("UC Volume" in issue or "uc_volume_root" in issue for issue in evidence.issues)


def test_evaluate_release_evidence_validates_required_engine_backends():
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_release_evidence(
            _v1_record(ok=True),
            _storage_record(ok=True),
            required_engine_probe_backends=(),
        )

    with pytest.raises(ValueError, match="Unsupported"):
        evaluate_release_evidence(
            _v1_record(ok=True),
            _storage_record(ok=True),
            required_engine_probe_backends=("triton",),
        )


def test_evaluate_release_evidence_files_reads_json_artifacts(tmp_path):
    v1_path = tmp_path / "v1.json"
    storage_path = tmp_path / "storage.json"
    vllm_path = tmp_path / "vllm-probe.json"
    sglang_path = tmp_path / "sglang-probe.json"
    vllm_actions_path = tmp_path / "vllm-actions.json"
    sglang_actions_path = tmp_path / "sglang-actions.json"
    _write_json(v1_path, _v1_record(ok=True))
    _write_json(storage_path, _storage_record(ok=True))
    _write_json(vllm_path, _probe_record(ServingBackend.VLLM))
    _write_json(sglang_path, _probe_record(ServingBackend.SGLANG))
    _write_json(vllm_actions_path, _actions_record(ServingBackend.VLLM))
    _write_json(sglang_actions_path, _actions_record(ServingBackend.SGLANG))

    evidence = evaluate_release_evidence_files(
        v1_benchmark_json=v1_path,
        storage_benchmark_json=storage_path,
        engine_probe_jsons=(vllm_path, sglang_path),
        engine_actions_jsons=(vllm_actions_path, sglang_actions_path),
    )

    assert evidence.ok
    assert release_evidence_to_record(evidence)["artifact_sources"] == [
        _artifact_source_record(
            v1_path,
            role="v1_benchmark",
            record_type=BENCHMARK_RUN_RECORD_TYPE,
        ),
        _artifact_source_record(
            storage_path,
            role="storage_benchmark",
            record_type=STORAGE_BENCHMARK_RECORD_TYPE,
        ),
        _artifact_source_record(
            vllm_path,
            role="engine_probe",
            record_type="document_kv.engine_kv_connector_probe.v1",
            backend="vllm",
        ),
        _artifact_source_record(
            sglang_path,
            role="engine_probe",
            record_type="document_kv.engine_kv_connector_probe.v1",
            backend="sglang",
        ),
        _artifact_source_record(
            vllm_actions_path,
            role="engine_connector_actions",
            record_type="document_kv.engine_kv_connector_actions.v1",
            backend="vllm",
        ),
        _artifact_source_record(
            sglang_actions_path,
            role="engine_connector_actions",
            record_type="document_kv.engine_kv_connector_actions.v1",
            backend="sglang",
        ),
    ]


def test_inspect_release_evidence_input_files_reports_record_types_and_missing_backends(tmp_path):
    v1_path = tmp_path / "v1.json"
    storage_path = tmp_path / "storage.json"
    vllm_path = tmp_path / "vllm-probe.json"
    vllm_actions_path = tmp_path / "vllm-actions.json"
    _write_json(v1_path, _v1_record(ok=True))
    _write_json(storage_path, _storage_record(ok=True))
    _write_json(vllm_path, _probe_record(ServingBackend.VLLM))
    _write_json(vllm_actions_path, _actions_record(ServingBackend.VLLM))

    status = inspect_release_evidence_input_files(
        v1_benchmark_json=v1_path,
        storage_benchmark_json=storage_path,
        engine_probe_jsons=(vllm_path,),
        engine_actions_jsons=(vllm_actions_path,),
    )
    record = release_evidence_input_status_to_record(status)

    assert not status.ok
    assert status.missing_paths == ()
    assert status.unreadable_paths == ()
    assert status.missing_engine_probe_backends == ("sglang",)
    assert status.missing_engine_action_backends == ("sglang",)
    assert record["record_type"] == RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE
    assert record["issues"] == [
        "missing engine probe backends: sglang",
        "missing engine action backends: sglang",
    ]
    assert record["input_files"] == [
        {
            "role": "v1_benchmark",
            "path": str(v1_path),
            "exists": True,
            "readable_json": True,
            "record_type": BENCHMARK_RUN_RECORD_TYPE,
        },
        {
            "role": "storage_benchmark",
            "path": str(storage_path),
            "exists": True,
            "readable_json": True,
            "record_type": STORAGE_BENCHMARK_RECORD_TYPE,
        },
        {
            "role": "engine_probe",
            "path": str(vllm_path),
            "exists": True,
            "readable_json": True,
            "record_type": "document_kv.engine_kv_connector_probe.v1",
            "backend": "vllm",
        },
        {
            "role": "engine_connector_actions",
            "path": str(vllm_actions_path),
            "exists": True,
            "readable_json": True,
            "record_type": "document_kv.engine_kv_connector_actions.v1",
            "backend": "vllm",
        },
    ]


def test_inspect_release_evidence_input_files_accepts_complete_backend_set(tmp_path):
    v1_path = tmp_path / "v1.json"
    storage_path = tmp_path / "storage.json"
    vllm_path = tmp_path / "vllm-probe.json"
    sglang_path = tmp_path / "sglang-probe.json"
    vllm_actions_path = tmp_path / "vllm-actions.json"
    sglang_actions_path = tmp_path / "sglang-actions.json"
    _write_json(v1_path, _v1_record(ok=True))
    _write_json(storage_path, _storage_record(ok=True))
    _write_json(vllm_path, _probe_record(ServingBackend.VLLM))
    _write_json(sglang_path, _probe_record(ServingBackend.SGLANG))
    _write_json(vllm_actions_path, _actions_record(ServingBackend.VLLM))
    _write_json(sglang_actions_path, _actions_record(ServingBackend.SGLANG))

    status = inspect_release_evidence_input_files(
        v1_benchmark_json=v1_path,
        storage_benchmark_json=storage_path,
        engine_probe_jsons=(vllm_path, sglang_path),
        engine_actions_jsons=(vllm_actions_path, sglang_actions_path),
    )
    record = release_evidence_input_status_to_record(status)

    assert status.ok
    assert status.missing_paths == ()
    assert status.unreadable_paths == ()
    assert status.missing_engine_probe_backends == ()
    assert status.missing_engine_action_backends == ()
    assert record["ok"] is True
    assert record["issues"] == []
    assert record["missing_engine_probe_backends"] == []
    assert record["missing_engine_action_backends"] == []


def test_inspect_release_evidence_input_files_rejects_wrong_record_types(tmp_path):
    v1_path = tmp_path / "v1.json"
    storage_path = tmp_path / "storage.json"
    vllm_path = tmp_path / "vllm-probe.json"
    sglang_path = tmp_path / "sglang-probe.json"
    vllm_actions_path = tmp_path / "vllm-actions.json"
    sglang_actions_path = tmp_path / "sglang-actions.json"
    _write_json(v1_path, _storage_record(ok=True))
    _write_json(storage_path, _v1_record(ok=True))
    _write_json(vllm_path, _actions_record(ServingBackend.VLLM))
    _write_json(sglang_path, _probe_record(ServingBackend.SGLANG))
    _write_json(vllm_actions_path, _probe_record(ServingBackend.VLLM))
    _write_json(sglang_actions_path, _actions_record(ServingBackend.SGLANG))

    status = inspect_release_evidence_input_files(
        v1_benchmark_json=v1_path,
        storage_benchmark_json=storage_path,
        engine_probe_jsons=(vllm_path, sglang_path),
        engine_actions_jsons=(vllm_actions_path, sglang_actions_path),
    )
    record = release_evidence_input_status_to_record(status)

    assert not status.ok
    assert status.missing_paths == ()
    assert status.unreadable_paths == ()
    assert status.missing_engine_probe_backends == ("vllm",)
    assert status.missing_engine_action_backends == ("vllm",)
    assert record["invalid_record_type_paths"] == [
        str(v1_path),
        str(storage_path),
        str(vllm_path),
        str(vllm_actions_path),
    ]
    assert any("invalid input record types" in issue for issue in status.issues)
    assert any("missing engine probe backends: vllm" in issue for issue in status.issues)
    assert any("missing engine action backends: vllm" in issue for issue in status.issues)
    assert "expected document_kv.benchmark_run.v1" in record["input_files"][0]["error"]
    assert "expected document_kv.storage_benchmark.v1" in record["input_files"][1]["error"]
    assert "expected document_kv.engine_kv_connector_probe.v1" in record["input_files"][2]["error"]
    assert "expected document_kv.engine_kv_connector_actions.v1" in record["input_files"][4]["error"]


def test_inspect_release_evidence_input_files_reports_missing_and_unreadable_paths(tmp_path):
    v1_path = tmp_path / "missing-v1.json"
    storage_path = tmp_path / "storage.json"
    sglang_path = tmp_path / "sglang-probe.json"
    storage_path.write_text("not json", encoding="utf-8")
    _write_json(sglang_path, _probe_record(ServingBackend.SGLANG))

    status = inspect_release_evidence_input_files(
        v1_benchmark_json=v1_path,
        storage_benchmark_json=storage_path,
        engine_probe_jsons=(sglang_path,),
    )

    assert not status.ok
    assert status.missing_paths == (str(v1_path),)
    assert status.unreadable_paths == (str(storage_path),)
    assert status.missing_engine_probe_backends == ("vllm",)
    assert any("missing input paths" in issue for issue in status.issues)
    assert any("unreadable input paths" in issue for issue in status.issues)


def test_release_evidence_input_status_validates_json_safe_schema():
    good = ReleaseEvidenceInputFileStatus(
        role="engine_probe",
        path="vllm.json",
        exists=True,
        readable_json=True,
        backend="vllm",
    )

    assert ReleaseEvidenceInputStatus(
        input_files=(good,),
        missing_paths=(),
        unreadable_paths=(),
        missing_engine_probe_backends=("sglang",),
        missing_engine_action_backends=("sglang",),
    ).issues == (
        "missing engine probe backends: sglang",
        "missing engine action backends: sglang",
    )
    normalized = ReleaseEvidenceInputStatus(
        input_files=[good],
        missing_paths=["missing.json"],
        unreadable_paths=["bad.json"],
        missing_engine_probe_backends=["vllm", "vllm"],
        missing_engine_action_backends=["sglang", "sglang"],
        required_engine_probe_backends=["sglang", "vllm", "vllm"],
        required_engine_action_backends=["vllm", "sglang", "sglang"],
    )
    assert normalized.input_files == (good,)
    assert normalized.missing_paths == ("missing.json",)
    assert normalized.unreadable_paths == ("bad.json",)
    assert normalized.missing_engine_probe_backends == ("vllm",)
    assert normalized.missing_engine_action_backends == ("sglang",)
    assert normalized.required_engine_probe_backends == ("sglang", "vllm")
    assert normalized.required_engine_action_backends == ("vllm", "sglang")

    with pytest.raises(ValueError, match="Unsupported artifact role"):
        ReleaseEvidenceInputFileStatus(role="dataset", path="x.json", exists=True, readable_json=True)
    with pytest.raises(ValueError, match="backend can only"):
        ReleaseEvidenceInputFileStatus(
            role="v1_benchmark",
            path="x.json",
            exists=True,
            readable_json=True,
            backend="vllm",
        )
    with pytest.raises(ValueError, match="Unsupported artifact backend"):
        ReleaseEvidenceInputFileStatus(
            role="engine_probe",
            path="x.json",
            exists=True,
            readable_json=True,
            backend="triton",
        )
    with pytest.raises(TypeError, match="input_files"):
        ReleaseEvidenceInputStatus(
            input_files=(object(),),
            missing_paths=(),
            unreadable_paths=(),
            missing_engine_probe_backends=(),
            missing_engine_action_backends=(),
        )
    with pytest.raises(TypeError, match="missing_paths"):
        ReleaseEvidenceInputStatus(
            input_files=(),
            missing_paths="missing.json",
            unreadable_paths=(),
            missing_engine_probe_backends=(),
            missing_engine_action_backends=(),
        )
    with pytest.raises(ValueError, match="unreadable_paths"):
        ReleaseEvidenceInputStatus(
            input_files=(),
            missing_paths=(),
            unreadable_paths=("",),
            missing_engine_probe_backends=(),
            missing_engine_action_backends=(),
        )


def test_evaluate_release_evidence_accepts_explicit_artifact_sources():
    source = ReleaseEvidenceArtifactSource(
        role="v1_benchmark",
        path="dbfs:/release/v1.json",
        record_type="document_kv.benchmark_run.v1",
    )

    evidence = evaluate_release_evidence(
        _v1_record(ok=True),
        _storage_record(ok=True),
        engine_probe_records=(
            _probe_record(ServingBackend.VLLM),
            _probe_record(ServingBackend.SGLANG),
        ),
        artifact_sources=(source,),
    )

    assert release_evidence_to_record(evidence)["artifact_sources"] == [
        {
            "role": "v1_benchmark",
            "path": "dbfs:/release/v1.json",
            "record_type": "document_kv.benchmark_run.v1",
        }
    ]


def test_release_evidence_preserves_positional_artifact_sources_slot():
    source = ReleaseEvidenceArtifactSource(role="engine_probe", path="vllm.json", backend="vllm")

    evidence = ReleaseEvidence(
        True,
        True,
        ("vllm", "sglang"),
        (),
        (),
        ("vllm", "sglang"),
        (),
        (),
        (),
        (source,),
    )
    record = release_evidence_to_record(evidence)

    assert evidence.duplicate_engine_probe_backends == ()
    assert evidence.duplicate_engine_action_backends == ()
    assert record["duplicate_engine_probe_backends"] == []
    assert record["duplicate_engine_action_backends"] == []
    assert record["artifact_sources"] == [
        {
            "role": "engine_probe",
            "path": "vllm.json",
            "backend": "vllm",
        }
    ]


def test_release_evidence_artifact_source_validates_json_safe_schema(tmp_path):
    path_source = ReleaseEvidenceArtifactSource(role="v1_benchmark", path=tmp_path / "v1.json")

    assert release_evidence_to_record(
        evaluate_release_evidence(
            _v1_record(ok=True),
            _storage_record(ok=True),
            engine_probe_records=(
                _probe_record(ServingBackend.VLLM),
                _probe_record(ServingBackend.SGLANG),
            ),
            artifact_sources=(path_source,),
        )
    )["artifact_sources"] == [{"role": "v1_benchmark", "path": str(tmp_path / "v1.json")}]

    with pytest.raises(ValueError, match="Unsupported artifact role"):
        ReleaseEvidenceArtifactSource(role="dataset", path="v1.json")
    with pytest.raises(ValueError, match="path must be non-empty"):
        ReleaseEvidenceArtifactSource(role="v1_benchmark", path="")
    with pytest.raises(ValueError, match="path must be non-empty"):
        ReleaseEvidenceArtifactSource(role="v1_benchmark", path=object())
    with pytest.raises(ValueError, match="record_type must be non-empty"):
        ReleaseEvidenceArtifactSource(role="v1_benchmark", path="v1.json", record_type=object())
    with pytest.raises(ValueError, match="backend can only be set"):
        ReleaseEvidenceArtifactSource(role="v1_benchmark", path="v1.json", backend="vllm")
    with pytest.raises(ValueError, match="Unsupported artifact backend"):
        ReleaseEvidenceArtifactSource(role="engine_probe", path="probe.json", backend="triton")
    with pytest.raises(ValueError, match="size_bytes"):
        ReleaseEvidenceArtifactSource(role="v1_benchmark", path="v1.json", size_bytes=-1)
    with pytest.raises(ValueError, match="sha256"):
        ReleaseEvidenceArtifactSource(role="v1_benchmark", path="v1.json", sha256="abc")


def test_public_release_evidence_cli_writes_json_and_returns_readiness_status(tmp_path):
    v1_path = tmp_path / "v1.json"
    storage_path = tmp_path / "storage.json"
    output_path = tmp_path / "release-evidence.json"
    _write_json(v1_path, _v1_record(ok=True))
    _write_json(storage_path, _storage_record(ok=False))

    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.release_evidence",
            "--v1-benchmark-json",
            str(v1_path),
            "--storage-benchmark-json",
            str(storage_path),
            "--output-json",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    record = json.loads(output_path.read_text(encoding="utf-8"))

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert record["record_type"] == RELEASE_EVIDENCE_RECORD_TYPE
    assert record["ok"] is False
    assert record["artifact_sources"] == [
        _artifact_source_record(
            v1_path,
            role="v1_benchmark",
            record_type=BENCHMARK_RUN_RECORD_TYPE,
        ),
        _artifact_source_record(
            storage_path,
            role="storage_benchmark",
            record_type=STORAGE_BENCHMARK_RECORD_TYPE,
        ),
    ]
    assert "sglang" in record["missing_engine_probe_backends"]


def test_public_release_evidence_cli_can_preflight_without_strict_validation(tmp_path):
    v1_path = tmp_path / "missing-v1.json"
    storage_path = tmp_path / "storage.json"
    preflight_path = tmp_path / "preflight.json"
    _write_json(storage_path, _storage_record(ok=True))

    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.release_evidence",
            "--v1-benchmark-json",
            str(v1_path),
            "--storage-benchmark-json",
            str(storage_path),
            "--preflight-output-json",
            str(preflight_path),
            "--preflight-only",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    record = json.loads(preflight_path.read_text(encoding="utf-8"))

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert record["record_type"] == RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE
    assert record["ok"] is False
    assert str(v1_path) in record["missing_paths"]
    assert record["missing_engine_probe_backends"] == ["vllm", "sglang"]


def test_public_release_evidence_cli_main_respects_public_stdout_serializers(monkeypatch, capsys, tmp_path):
    v1_path = tmp_path / "v1.json"
    storage_path = tmp_path / "storage.json"
    _write_json(v1_path, _v1_record(ok=True))
    _write_json(storage_path, _storage_record(ok=True))
    original_status_serializer = legacy_release_evidence.release_evidence_input_status_to_record
    original_evidence_serializer = legacy_release_evidence.release_evidence_to_record

    def fake_status_serializer(status):
        assert status.missing_engine_probe_backends == ("vllm", "sglang")
        return {"ok": "public-status-serializer"}

    monkeypatch.setattr(public_release_evidence, "release_evidence_input_status_to_record", fake_status_serializer)
    assert public_release_evidence.main(
        [
            "--v1-benchmark-json",
            str(v1_path),
            "--storage-benchmark-json",
            str(storage_path),
            "--preflight-only",
        ]
    ) == 2
    assert json.loads(capsys.readouterr().out) == {"ok": "public-status-serializer"}

    def fake_evidence_serializer(evidence):
        assert evidence.missing_engine_probe_backends == ("vllm", "sglang")
        return {"ok": "public-evidence-serializer"}

    monkeypatch.setattr(public_release_evidence, "release_evidence_to_record", fake_evidence_serializer)
    assert public_release_evidence.main(
        [
            "--v1-benchmark-json",
            str(v1_path),
            "--storage-benchmark-json",
            str(storage_path),
        ]
    ) == 2
    assert json.loads(capsys.readouterr().out) == {"ok": "public-evidence-serializer"}
    assert legacy_release_evidence.release_evidence_input_status_to_record is original_status_serializer
    assert legacy_release_evidence.release_evidence_to_record is original_evidence_serializer


def test_legacy_release_evidence_cli_main_respects_legacy_stdout_serializers(monkeypatch, capsys, tmp_path):
    v1_path = tmp_path / "v1.json"
    storage_path = tmp_path / "storage.json"
    _write_json(v1_path, _v1_record(ok=True))
    _write_json(storage_path, _storage_record(ok=True))
    original_status_serializer = public_release_evidence.release_evidence_input_status_to_record
    original_evidence_serializer = public_release_evidence.release_evidence_to_record

    def fake_status_serializer(status):
        assert status.missing_engine_probe_backends == ("vllm", "sglang")
        return {"ok": "legacy-status-serializer"}

    monkeypatch.setattr(legacy_release_evidence, "release_evidence_input_status_to_record", fake_status_serializer)
    assert legacy_release_evidence.main(
        [
            "--v1-benchmark-json",
            str(v1_path),
            "--storage-benchmark-json",
            str(storage_path),
            "--preflight-only",
        ]
    ) == 2
    assert json.loads(capsys.readouterr().out) == {"ok": "legacy-status-serializer"}

    def fake_evidence_serializer(evidence):
        assert evidence.missing_engine_probe_backends == ("vllm", "sglang")
        return {"ok": "legacy-evidence-serializer"}

    monkeypatch.setattr(legacy_release_evidence, "release_evidence_to_record", fake_evidence_serializer)
    assert legacy_release_evidence.main(
        [
            "--v1-benchmark-json",
            str(v1_path),
            "--storage-benchmark-json",
            str(storage_path),
        ]
    ) == 2
    assert json.loads(capsys.readouterr().out) == {"ok": "legacy-evidence-serializer"}
    assert public_release_evidence.release_evidence_input_status_to_record is original_status_serializer
    assert public_release_evidence.release_evidence_to_record is original_evidence_serializer


def test_public_release_evidence_cli_rejects_wrong_v1_record_type(tmp_path):
    v1_path = tmp_path / "v1.json"
    storage_path = tmp_path / "storage.json"
    output_path = tmp_path / "release-evidence.json"
    v1_record = _v1_record(ok=True)
    v1_record["record_type"] = "document_kv.benchmark_summary.v1"
    _write_json(v1_path, v1_record)
    _write_json(storage_path, _storage_record(ok=True))

    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.release_evidence",
            "--v1-benchmark-json",
            str(v1_path),
            "--storage-benchmark-json",
            str(storage_path),
            "--output-json",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    record = json.loads(output_path.read_text(encoding="utf-8"))

    assert completed.returncode == 2
    assert record["ok"] is False
    assert any("v1 benchmark record_type" in issue for issue in record["issues"])


def _v1_record(*, ok: bool, hardware_target: str = "aws-g5", model_id: str = "qwen3:4b-instruct"):
    datasets = ("biography", "hotpotqa", "musique", "niah")
    arms = ("baseline_prefill", "document_kv_cache")
    return {
        "record_type": BENCHMARK_RUN_RECORD_TYPE,
        "suite": {
            "suite_id": "v1-suite",
            "hardware_target": hardware_target,
            "model_id": model_id,
            "datasets": list(datasets),
            "examples": len(datasets),
        },
        "measurements": [_v1_measurement_record(dataset, arm) for dataset in datasets for arm in arms],
        "report_rows": [_v1_report_row_record(dataset, arm) for dataset in datasets for arm in arms],
        "comparisons": [
            {
                "dataset": dataset,
                "baseline_arm_id": "baseline_prefill",
                "cache_arm_id": "document_kv_cache",
                "ttft_speedup": 1.0,
                "time_to_completion_speedup": 1.0,
                "exact_match_delta": 0.0,
                "answer_found_delta": 0.0,
            }
            for dataset in datasets
        ],
        "v1_evidence": {
            "ok": ok,
            "required_datasets": list(datasets),
            "duplicate_required_datasets": [],
            "duplicate_report_rows": [],
            "duplicate_comparisons": [],
            "missing_report_rows": [],
            "missing_comparisons": [],
            "comparisons_without_metrics": [],
            "rows_without_successful_requests": [],
            "rows_without_latency": [],
            "rows_without_quality": [],
            "unexpected_arms": [],
            "unexpected_datasets": [],
            "issues": [] if ok else ["missing report rows: hotpotqa:baseline_prefill"],
        },
    }


def _v1_measurement_record(dataset: str, arm: str):
    prompt_tokens = 1024 if arm == "baseline_prefill" else 128
    return {
        "example_id": f"{dataset}-1",
        "dataset": dataset,
        "arm_id": arm,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 16,
        "ttft_seconds": 1.0,
        "time_to_completion_seconds": 2.0,
        "output_text": "Ada Lovelace",
        "expected_answer": "Ada Lovelace",
        "exact_match": True,
        "answer_found": True,
        "error": None,
        "metadata": _v1_measurement_metadata(arm),
    }


def _v1_report_row_record(dataset: str, arm: str):
    prompt_tokens = 1024.0 if arm == "baseline_prefill" else 128.0
    return {
        "dataset": dataset,
        "arm_id": arm,
        "requests": 1,
        "errors": 0,
        "prompt_tokens_mean": prompt_tokens,
        "completion_tokens_mean": 16.0,
        "ttft": {"count": 1, "mean": 1.0, "p50": 1.0, "p95": 1.0},
        "time_to_completion": {"count": 1, "mean": 2.0, "p50": 2.0, "p95": 2.0},
        "exact_match_rate": 1.0,
        "answer_found_rate": 1.0,
        "output_tokens_per_second": 8.0,
    }


def _v1_measurement_metadata(arm: str):
    if arm == "baseline_prefill":
        return {
            "prompt_text_mode": "logical",
            "prompt_token_source": "logical",
            "logical_prompt_tokens": "1024",
            "runtime_prompt_tokens": "1024",
        }
    return {
        "prompt_text_mode": "runtime",
        "prompt_token_source": "server_usage",
        "logical_prompt_tokens": "1024",
        "runtime_prompt_tokens": "128",
    }


def _storage_record(*, ok: bool, uc_volume_is_real: bool = True):
    readers = ("memory", "disk", "unity_catalog")
    return {
        "record_type": STORAGE_BENCHMARK_RECORD_TYPE,
        "readers": list(readers),
        "uc_volume_root": "/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        "results": [
            {
                "reader_id": reader,
                "total_reads": 4,
                "total_bytes": 4096,
                "parallelism": 2,
                "wall_seconds": 0.01,
                "errors": 0,
                "latency_p50_seconds": 0.001,
                "latency_p95_seconds": 0.002,
                "throughput_bytes_per_second": 1024.0,
            }
            for reader in readers
        ],
        "uc_volume_is_real": uc_volume_is_real,
        "release_storage_evidence": {
            "ok": ok,
            "required_readers": list(readers),
            "missing_readers": [],
            "readers_with_errors": [],
            "readers_without_latency": [],
            "readers_without_throughput": [],
            "require_real_uc_volume": True,
            "uc_volume_root": "/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            "uc_volume_is_real": uc_volume_is_real,
            "issues": [] if ok else ["missing storage readers: unity_catalog"],
        },
    }


def _probe_record(backend: ServingBackend, *, layout=None):
    layout = layout or layout_for_model("qwen3:4b-instruct")
    profile = serving_environment_profile(backend)
    return engine_kv_connector_probe_result_to_record(
        EngineKVConnectorProbeResult(
            backend=backend,
            request_id=f"{backend.value}-probe",
            total_blocks=1,
            copied_segments=1,
            copied_tokens=1,
            copied_bytes=layout.bytes_per_token,
            bound=True,
            released=True,
            model_id="qwen3:4b-instruct",
            layout_version="qwen3-v1",
            layout=layout,
            payload_mode=PayloadMode.MERGED,
            connector_package=backend.value,
            engine_version=profile.engine_version,
            metadata={
                ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE: profile.engine_package,
                ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION: profile.engine_version,
            },
        )
    )


def _actions_record(backend: ServingBackend, *, layout=None, request_id=None, total_tokens: int = 1):
    layout = layout or layout_for_model("qwen3:4b-instruct")
    request_id = request_id or f"{backend.value}-probe"
    total_blocks = max(1, (total_tokens + layout.block_size - 1) // layout.block_size)
    total_bytes = total_tokens * layout.bytes_per_token
    return engine_kv_connector_actions_to_record(
        EngineKVConnectorActions(
            reservation=EngineKVReservationAction(
                backend=backend,
                request_id=request_id,
                total_blocks=total_blocks,
                total_tokens=total_tokens,
                estimated_gpu_bytes=total_bytes,
                layout=layout,
                adapter_ids=("base",),
            ),
            copies=(
                EngineKVSegmentCopyAction(
                    request_id=request_id,
                    document_id="doc-a",
                    chunk_type="document_chunk",
                    chunk_id="chunk-1",
                    payload_index=None,
                    source_byte_start=0,
                    source_byte_length=total_bytes,
                    global_byte_start=0,
                    global_byte_end=total_bytes,
                    token_start=0,
                    token_count=total_tokens,
                    token_end=total_tokens,
                    first_block_index=0,
                    last_block_index_exclusive=total_blocks,
                ),
            ),
            bind=EngineKVBindAction(
                request_id=request_id,
                handle_uri=f"engine://{backend.value}/{request_id}",
                cache_method="vanilla",
                adapter_ids=("base",),
                metadata={
                    "engine.backend": backend.value,
                    "engine.connector_package": backend.value,
                },
            ),
            release=EngineKVReleaseAction(request_id=request_id),
        )
    )


def _write_json(path: Path, record) -> None:
    path.write_text(json.dumps(record), encoding="utf-8")


def _artifact_source_record(path: Path, *, role: str, record_type: str, backend: str | None = None) -> dict[str, object]:
    payload = path.read_bytes()
    record: dict[str, object] = {
        "role": role,
        "path": str(path),
        "record_type": record_type,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if backend is not None:
        record["backend"] = backend
    return record
