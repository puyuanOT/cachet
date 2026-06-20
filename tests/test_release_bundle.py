import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import document_kv_cache.release_bundle as public_release_bundle
import restaurant_kv_serving.release_bundle as legacy_release_bundle
from document_kv_cache.benchmark_runner import BENCHMARK_RUN_RECORD_TYPE
from document_kv_cache.benchmark_plan_executor import (
    BENCHMARK_PLAN_EXECUTION_RECORD_TYPE,
    BENCHMARK_PLAN_SOURCE_RECORD_TYPE,
)
from document_kv_cache.databricks_runs import (
    DATABRICKS_RUN_STATUS_RECORD_TYPE,
    DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE,
)
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
from document_kv_cache.github_governance import GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE
from document_kv_cache.model_profiles import layout_for_model
from document_kv_cache.native_probe_factories import (
    NATIVE_PROBE_FACTORIES_RECORD_TYPE,
    SGLANG_NATIVE_PROBE_FACTORY,
    VLLM_NATIVE_PROBE_FACTORY,
)
from document_kv_cache.release_bundle import (
    RELEASE_BUNDLE_MANIFEST_FILENAME,
    RELEASE_BUNDLE_RECORD_TYPE,
    STRICT_V1_RELEASE_REQUIRED_DATABRICKS_PURPOSES,
    ReleaseBundle,
    ReleaseBundleArtifact,
    build_release_bundle,
    release_bundle_to_record,
)
from document_kv_cache.repository_hygiene import (
    FORBIDDEN_TRACKED_ARTIFACT_PATTERNS,
    REPOSITORY_HYGIENE_RECORD_TYPE,
    REQUIRED_GITIGNORE_PATTERNS,
)
from document_kv_cache.release_evidence import (
    evaluate_release_evidence_files,
    inspect_release_evidence_input_files,
    write_release_evidence_input_status_json,
    write_release_evidence_json,
)
from document_kv_cache.serving_env import serving_environment_profile, serving_environment_profile_to_record
from document_kv_cache.storage_benchmark import STORAGE_BENCHMARK_RECORD_TYPE


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_release_bundle_copies_artifacts_and_writes_checksummed_manifest(tmp_path):
    source_dir = tmp_path / "sources"
    bundle_dir = tmp_path / "bundle"
    artifacts = _write_release_ready_artifacts(source_dir)

    bundle = build_release_bundle(
        v1_benchmark_json=artifacts["v1"],
        storage_benchmark_json=artifacts["storage"],
        engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
        release_evidence_json=artifacts["evidence"],
        preflight_json=artifacts["preflight"],
        output_dir=bundle_dir,
    )
    record = json.loads((bundle_dir / RELEASE_BUNDLE_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert record == release_bundle_to_record(bundle)
    assert record["record_type"] == RELEASE_BUNDLE_RECORD_TYPE
    assert record["ok"] is True
    assert record["artifact_count"] == 8
    assert [artifact["role"] for artifact in record["artifacts"]] == [
        "v1_benchmark",
        "storage_benchmark",
        "engine_probe",
        "engine_probe",
        "engine_connector_actions",
        "engine_connector_actions",
        "release_evidence",
        "preflight",
    ]
    assert [artifact.get("backend") for artifact in record["artifacts"][2:6]] == [
        "vllm",
        "sglang",
        "vllm",
        "sglang",
    ]

    for artifact in record["artifacts"]:
        source_payload = Path(artifact["source_path"]).read_bytes()
        bundled_payload = (bundle_dir / artifact["bundled_path"]).read_bytes()
        assert bundled_payload == source_payload
        assert artifact["size_bytes"] == len(source_payload)
        assert artifact["sha256"] == hashlib.sha256(source_payload).hexdigest()


def test_build_release_bundle_plan_execution_stays_out_of_release_sidecar_matching(tmp_path):
    source_dir = tmp_path / "sources"
    bundle_dir = tmp_path / "bundle"
    artifacts = _write_release_ready_artifacts(source_dir)
    plan_execution = _write_json(source_dir / "plan-execution.json", _plan_execution_record(ok=True))

    bundle = build_release_bundle(
        v1_benchmark_json=artifacts["v1"],
        storage_benchmark_json=artifacts["storage"],
        engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
        release_evidence_json=artifacts["evidence"],
        preflight_json=artifacts["preflight"],
        plan_execution_jsons=(plan_execution,),
        output_dir=bundle_dir,
    )
    record = release_bundle_to_record(bundle)

    assert [artifact["role"] for artifact in record["artifacts"]] == [
        "v1_benchmark",
        "storage_benchmark",
        "engine_probe",
        "engine_probe",
        "engine_connector_actions",
        "engine_connector_actions",
        "release_evidence",
        "preflight",
        "plan_execution",
    ]
    assert record["artifacts"][-1]["record_type"] == BENCHMARK_PLAN_EXECUTION_RECORD_TYPE


def test_build_release_bundle_databricks_status_stays_out_of_release_sidecar_matching(tmp_path):
    source_dir = tmp_path / "sources"
    bundle_dir = tmp_path / "bundle"
    artifacts = _write_release_ready_artifacts(source_dir)
    run_status = _write_json(
        source_dir / "databricks-run-status.json",
        _databricks_run_status_cli_record(succeeded=True),
    )

    bundle = build_release_bundle(
        v1_benchmark_json=artifacts["v1"],
        storage_benchmark_json=artifacts["storage"],
        engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
        release_evidence_json=artifacts["evidence"],
        preflight_json=artifacts["preflight"],
        databricks_run_status_jsons=(run_status,),
        output_dir=bundle_dir,
    )
    record = release_bundle_to_record(bundle)

    assert [artifact["role"] for artifact in record["artifacts"]] == [
        "v1_benchmark",
        "storage_benchmark",
        "engine_probe",
        "engine_probe",
        "engine_connector_actions",
        "engine_connector_actions",
        "release_evidence",
        "preflight",
        "databricks_run_status",
    ]
    status_artifact = record["artifacts"][-1]
    assert status_artifact["record_type"] == DATABRICKS_RUN_STATUS_RECORD_TYPE
    assert status_artifact["bundled_path"] == "databricks_run_status_09.json"
    bundled_record = json.loads((bundle_dir / status_artifact["bundled_path"]).read_text(encoding="utf-8"))
    assert bundled_record["summary"]["succeeded"] is True


def test_build_release_bundle_can_include_package_wheel_pr_evidence_and_github_governance(tmp_path):
    source_dir = tmp_path / "sources"
    bundle_dir = tmp_path / "bundle"
    artifacts = _write_release_ready_artifacts(source_dir)
    package_wheel = _write_wheel(source_dir / "document_kv_cache-0.2.0-py3-none-any.whl")
    plan_execution = _write_json(source_dir / "plan-execution.json", _plan_execution_record(ok=True))
    pr_evidence = _write_json(source_dir / "pr-evidence.json", _pr_evidence_record(ok=True))
    github_governance = _write_json(source_dir / "github-governance.json", _github_governance_cli_record(ok=True))
    repository_hygiene = _write_json(source_dir / "repository-hygiene.json", _repository_hygiene_record(ok=True))

    bundle = build_release_bundle(
        v1_benchmark_json=artifacts["v1"],
        storage_benchmark_json=artifacts["storage"],
        engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
        plan_execution_jsons=(plan_execution,),
        package_wheel=package_wheel,
        pr_evidence_jsons=(pr_evidence,),
        github_governance_json=github_governance,
        repository_hygiene_json=repository_hygiene,
        output_dir=bundle_dir,
    )
    record = release_bundle_to_record(bundle)

    assert [artifact["role"] for artifact in record["artifacts"]] == [
        "v1_benchmark",
        "storage_benchmark",
        "engine_probe",
        "engine_probe",
        "engine_connector_actions",
        "engine_connector_actions",
        "plan_execution",
        "package_wheel",
        "pr_evidence",
        "github_governance",
        "repository_hygiene",
    ]
    execution_artifact = record["artifacts"][6]
    wheel_artifact = record["artifacts"][7]
    pr_artifact = record["artifacts"][8]
    github_artifact = record["artifacts"][9]
    hygiene_artifact = record["artifacts"][10]
    assert execution_artifact["record_type"] == BENCHMARK_PLAN_EXECUTION_RECORD_TYPE
    assert json.loads((bundle_dir / execution_artifact["bundled_path"]).read_text(encoding="utf-8"))["ok"] is True
    assert wheel_artifact["bundled_path"] == package_wheel.name
    assert "record_type" not in wheel_artifact
    assert (bundle_dir / wheel_artifact["bundled_path"]).read_bytes() == package_wheel.read_bytes()
    assert pr_artifact["record_type"] == "document_kv.pr_evidence.v1"
    assert json.loads((bundle_dir / pr_artifact["bundled_path"]).read_text(encoding="utf-8"))["ok"] is True
    assert github_artifact["record_type"] == GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE
    assert github_artifact["bundled_path"] == "github_governance.json"
    assert json.loads((bundle_dir / github_artifact["bundled_path"]).read_text(encoding="utf-8"))["ok"] is True
    assert hygiene_artifact["record_type"] == REPOSITORY_HYGIENE_RECORD_TYPE
    assert hygiene_artifact["bundled_path"] == "repository_hygiene.json"
    assert json.loads((bundle_dir / hygiene_artifact["bundled_path"]).read_text(encoding="utf-8"))["ok"] is True


def test_build_release_bundle_strict_v1_rejects_incomplete_release_artifact_set(tmp_path):
    source_dir = tmp_path / "sources"
    artifacts = _write_release_ready_artifacts(source_dir)

    with pytest.raises(ValueError, match="require_complete_v1 must be boolean"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            output_dir=tmp_path / "strict-invalid-flag-bundle",
            require_complete_v1=1,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Strict V1 release bundle requires .*preflight sidecar"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
            engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            release_evidence_json=artifacts["evidence"],
            output_dir=tmp_path / "strict-incomplete-bundle",
            require_complete_v1=True,
        )


def test_build_release_bundle_strict_v1_accepts_complete_release_artifact_set(tmp_path):
    source_dir = tmp_path / "sources"
    bundle_dir = tmp_path / "bundle"
    artifacts = _write_release_ready_artifacts(source_dir)
    package_wheel = _write_wheel(source_dir / "document_kv_cache-0.2.0-py3-none-any.whl")
    plan_execution = _write_json(source_dir / "plan-execution.json", _plan_execution_record(ok=True))
    v1_run_status = _write_json(
        source_dir / "databricks-run-status-v1.json",
        _databricks_run_status_cli_record(
            succeeded=True,
            purpose="document-kv-v1-benchmark",
            run_name="document-kv-v1-benchmark",
            task_key="document_kv_v1_benchmark",
        ),
    )
    storage_run_status = _write_json(
        source_dir / "databricks-run-status-storage.json",
        _databricks_run_status_cli_record(
            succeeded=True,
            purpose="document-kv-storage-benchmark",
            run_name="document-kv-storage-benchmark",
            task_key="document_kv_storage_benchmark",
        ),
    )
    engine_probe_run_status = _write_json(
        source_dir / "databricks-run-status-engine-probe.json",
        _databricks_run_status_cli_record(
            succeeded=True,
            purpose="document-kv-engine-probe",
            run_name="document-kv-engine-probe",
            task_key="document_kv_engine_probe",
        ),
    )
    pr_evidence = _write_json(source_dir / "pr-evidence.json", _pr_evidence_record(ok=True))
    github_governance = _write_json(source_dir / "github-governance.json", _github_governance_cli_record(ok=True))
    repository_hygiene = _write_json(source_dir / "repository-hygiene.json", _repository_hygiene_record(ok=True))
    native_probe_factories = _write_json(source_dir / "native-probe-factories.json", _native_probe_factories_record())

    with pytest.raises(ValueError, match="storage-reader benchmark Databricks run-status evidence"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
            engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            release_evidence_json=artifacts["evidence"],
            preflight_json=artifacts["preflight"],
            plan_execution_jsons=(plan_execution,),
            databricks_run_status_jsons=(v1_run_status,),
            package_wheel=package_wheel,
            pr_evidence_jsons=(pr_evidence,),
            github_governance_json=github_governance,
            repository_hygiene_json=repository_hygiene,
            native_probe_factories_jsons=(native_probe_factories,),
            output_dir=tmp_path / "strict-missing-databricks-purpose-bundle",
            require_complete_v1=True,
        )

    bundle = build_release_bundle(
        v1_benchmark_json=artifacts["v1"],
        storage_benchmark_json=artifacts["storage"],
        engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
        release_evidence_json=artifacts["evidence"],
        preflight_json=artifacts["preflight"],
        plan_execution_jsons=(plan_execution,),
        databricks_run_status_jsons=(v1_run_status, storage_run_status, engine_probe_run_status),
        package_wheel=package_wheel,
        pr_evidence_jsons=(pr_evidence,),
        github_governance_json=github_governance,
        repository_hygiene_json=repository_hygiene,
        native_probe_factories_jsons=(native_probe_factories,),
        output_dir=bundle_dir,
        require_complete_v1=True,
    )
    record = release_bundle_to_record(bundle)

    assert record["ok"] is True
    assert [artifact["role"] for artifact in record["artifacts"]] == [
        "v1_benchmark",
        "storage_benchmark",
        "engine_probe",
        "engine_probe",
        "engine_connector_actions",
        "engine_connector_actions",
        "release_evidence",
        "preflight",
        "plan_execution",
        "databricks_run_status",
        "databricks_run_status",
        "databricks_run_status",
        "package_wheel",
        "pr_evidence",
        "github_governance",
        "repository_hygiene",
        "native_probe_factories",
    ]
    status_records = [
        json.loads((bundle_dir / artifact["bundled_path"]).read_text(encoding="utf-8"))["summary"]
        for artifact in record["artifacts"]
        if artifact["role"] == "databricks_run_status"
    ]
    purposes = {
        task["purpose"]
        for status in status_records
        for task in status["submit_payload"]["tasks"]
    }
    assert purposes == {purpose for purpose, _label in STRICT_V1_RELEASE_REQUIRED_DATABRICKS_PURPOSES}


def test_build_release_bundle_can_include_native_probe_factories(tmp_path):
    source_dir = tmp_path / "sources"
    bundle_dir = tmp_path / "bundle"
    artifacts = _write_release_ready_artifacts(source_dir)
    native_probe_factories = _write_json(
        source_dir / "native-probe-factories.json",
        _native_probe_factories_record(),
    )

    bundle = build_release_bundle(
        v1_benchmark_json=artifacts["v1"],
        storage_benchmark_json=artifacts["storage"],
        engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
        native_probe_factories_jsons=(native_probe_factories,),
        output_dir=bundle_dir,
    )
    record = release_bundle_to_record(bundle)

    assert record["artifacts"][-1]["role"] == "native_probe_factories"
    assert record["artifacts"][-1]["record_type"] == NATIVE_PROBE_FACTORIES_RECORD_TYPE
    assert record["artifacts"][-1]["bundled_path"] == "native_probe_factories_07.json"
    bundled_record = json.loads((bundle_dir / "native_probe_factories_07.json").read_text(encoding="utf-8"))
    assert bundled_record["record_type"] == NATIVE_PROBE_FACTORIES_RECORD_TYPE
    assert {factory["backend"] for factory in bundled_record["factories"]} == {"vllm", "sglang"}


def test_build_release_bundle_rejects_invalid_native_probe_factories(tmp_path):
    artifacts = _write_release_ready_artifacts(tmp_path / "sources")
    invalid_record = _native_probe_factories_record()
    invalid_record["factories"] = invalid_record["factories"][:1]
    invalid_native_probe_factories = _write_json(tmp_path / "invalid-native-probe-factories.json", invalid_record)

    with pytest.raises(ValueError, match="native probe factories sidecar backends must match required backends"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
            engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            native_probe_factories_jsons=(invalid_native_probe_factories,),
            output_dir=tmp_path / "invalid-native-probe-factories-bundle",
        )

    wrong_path_record = _native_probe_factories_record()
    wrong_path_record["factories"][0]["factory_path"] = "downstream:factory"
    wrong_path_native_probe_factories = _write_json(
        tmp_path / "wrong-path-native-probe-factories.json",
        wrong_path_record,
    )
    with pytest.raises(ValueError, match="factory_path must match the built-in vllm factory path"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
            engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            native_probe_factories_jsons=(wrong_path_native_probe_factories,),
            output_dir=tmp_path / "wrong-path-native-probe-factories-bundle",
        )

    wrong_profile_record = _native_probe_factories_record()
    del wrong_profile_record["factories"][0]["serving_environment_profile"]["dependency_constraints"]
    wrong_profile_native_probe_factories = _write_json(
        tmp_path / "wrong-profile-native-probe-factories.json",
        wrong_profile_record,
    )
    with pytest.raises(ValueError, match="serving_environment_profile must match the built-in vllm profile"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
            engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            native_probe_factories_jsons=(wrong_profile_native_probe_factories,),
            output_dir=tmp_path / "wrong-profile-native-probe-factories-bundle",
        )


def test_build_release_bundle_rejects_invalid_package_wheel_pr_evidence_or_github_governance(tmp_path):
    artifacts = _write_release_ready_artifacts(tmp_path / "sources")
    bad_wheel = tmp_path / "not-a-wheel.txt"
    bad_wheel.write_bytes(b"not a wheel")
    failed_pr_evidence = _write_json(tmp_path / "failed-pr-evidence.json", _pr_evidence_record(ok=False))
    failed_github_governance = _write_json(
        tmp_path / "failed-github-governance.json",
        _github_governance_cli_record(ok=False),
    )
    internal_github_governance_record = _github_governance_cli_record(ok=True)
    internal_github_governance_record["summary"]["visibility"] = "internal"
    internal_github_governance = _write_json(
        tmp_path / "internal-github-governance.json",
        internal_github_governance_record,
    )
    raw_response_github_governance = _write_json(
        tmp_path / "raw-response-github-governance.json",
        {
            **_github_governance_cli_record(ok=True),
            "response": {"headers": {"authorization": "do-not-bundle-me"}},
        },
    )
    raw_summary_github_governance_record = _github_governance_cli_record(ok=True)
    raw_summary_github_governance_record["summary"]["response"] = {
        "headers": {"authorization": "do-not-bundle-me"}
    }
    raw_summary_github_governance = _write_json(
        tmp_path / "raw-summary-github-governance.json",
        raw_summary_github_governance_record,
    )
    raw_allowed_summary_field_github_governance_record = _github_governance_cli_record(ok=True)
    raw_allowed_summary_field_github_governance_record["summary"]["description"] = {
        "raw": {"authorization": "do-not-bundle-me"}
    }
    raw_allowed_summary_field_github_governance = _write_json(
        tmp_path / "raw-allowed-summary-field-github-governance.json",
        raw_allowed_summary_field_github_governance_record,
    )
    raw_branch_protection_github_governance_record = _github_governance_cli_record(ok=True)
    raw_branch_protection_github_governance_record["summary"]["branch_protection"]["raw"] = {
        "headers": {"authorization": "do-not-bundle-me"}
    }
    raw_branch_protection_github_governance = _write_json(
        tmp_path / "raw-branch-protection-github-governance.json",
        raw_branch_protection_github_governance_record,
    )
    raw_status_checks_github_governance_record = _github_governance_cli_record(ok=True)
    raw_status_checks_github_governance_record["summary"]["branch_protection"]["required_status_checks"][
        "checks"
    ] = [{"context": "Test and build", "app": {"slug": "do-not-bundle-me"}}]
    raw_status_checks_github_governance = _write_json(
        tmp_path / "raw-status-checks-github-governance.json",
        raw_status_checks_github_governance_record,
    )
    raw_pr_reviews_github_governance_record = _github_governance_cli_record(ok=True)
    raw_pr_reviews_github_governance_record["summary"]["branch_protection"]["required_pull_request_reviews"][
        "bypass_pull_request_allowances"
    ] = {"users": [{"login": "do-not-bundle-me"}]}
    raw_pr_reviews_github_governance = _write_json(
        tmp_path / "raw-pr-reviews-github-governance.json",
        raw_pr_reviews_github_governance_record,
    )
    raw_contexts_github_governance_record = _github_governance_cli_record(ok=True)
    raw_contexts_github_governance_record["summary"]["branch_protection"]["required_status_checks"]["contexts"] = [
        "Test and build",
        {"context": "do-not-bundle-me"},
    ]
    raw_contexts_github_governance = _write_json(
        tmp_path / "raw-contexts-github-governance.json",
        raw_contexts_github_governance_record,
    )
    unexpected_open_pr_github_governance_record = _github_governance_cli_record(ok=True)
    unexpected_open_pr_github_governance_record["summary"]["ok"] = False
    unexpected_open_pr_github_governance_record["summary"]["open_pull_requests"]["total_count"] = 1
    unexpected_open_pr_github_governance_record["summary"]["open_pull_requests"]["unexpected_count"] = 1
    unexpected_open_pr_github_governance_record["summary"]["open_pull_requests"]["unexpected"] = [
        {
            "number": 72,
            "title": "Stale experiment branch",
            "draft": True,
            "html_url": "https://github.com/owner/document-kv-cache/pull/72",
            "head_ref": "experiment",
            "base_ref": "main",
        }
    ]
    unexpected_open_pr_github_governance_record["summary"]["issues"] = [
        "repository must not have unexpected open pull requests: #72",
    ]
    unexpected_open_pr_github_governance = _write_json(
        tmp_path / "unexpected-open-pr-github-governance.json",
        unexpected_open_pr_github_governance_record,
    )
    truncated_open_pr_github_governance_record = _github_governance_cli_record(ok=True)
    truncated_open_pr_github_governance_record["summary"]["open_pull_requests"]["truncated"] = True
    truncated_open_pr_github_governance = _write_json(
        tmp_path / "truncated-open-pr-github-governance.json",
        truncated_open_pr_github_governance_record,
    )
    raw_open_pr_github_governance_record = _github_governance_cli_record(ok=True)
    raw_open_pr_github_governance_record["summary"]["open_pull_requests"]["response"] = {
        "headers": {"authorization": "do-not-bundle-me"}
    }
    raw_open_pr_github_governance = _write_json(
        tmp_path / "raw-open-pr-github-governance.json",
        raw_open_pr_github_governance_record,
    )
    unresolved_review_pr_evidence_record = _pr_evidence_record(ok=True)
    unresolved_review_pr_evidence_record["gpt55_review_outcome"] = "findings_resolved"
    unresolved_review_pr_evidence_record["gpt55_review_findings_resolved"] = False
    unresolved_review_pr_evidence = _write_json(
        tmp_path / "unresolved-review-pr-evidence.json",
        unresolved_review_pr_evidence_record,
    )
    failed_plan_execution = _write_json(tmp_path / "failed-plan-execution.json", _plan_execution_record(ok=False))
    failed_run_status = _write_json(
        tmp_path / "failed-databricks-run-status.json",
        _databricks_run_status_record(succeeded=False),
    )
    raw_response_run_status = _write_json(
        tmp_path / "raw-response-databricks-run-status.json",
        {
            **_databricks_run_status_cli_record(succeeded=True),
            "response": {"tasks": [{"notebook_task": {"base_parameters": {"token": "do-not-bundle-me"}}}]},
        },
    )
    missing_submit_payload_status_record = _databricks_run_status_record(succeeded=True)
    missing_submit_payload_status_record.pop("submit_payload")
    missing_submit_payload_status = _write_json(
        tmp_path / "missing-submit-payload-databricks-run-status.json",
        missing_submit_payload_status_record,
    )
    non_g5_submit_payload_status_record = _databricks_run_status_record(succeeded=True)
    non_g5_submit_payload_status_record["submit_payload"]["aws_g5_node_type"] = False
    non_g5_submit_payload_status_record["submit_payload"]["tasks"][0]["node_type_id"] = "g6.4xlarge"
    non_g5_submit_payload_status = _write_json(
        tmp_path / "non-g5-submit-payload-databricks-run-status.json",
        non_g5_submit_payload_status_record,
    )
    empty_tasks_status_record = _databricks_run_status_record(succeeded=True)
    empty_tasks_status_record["task_count"] = 0
    empty_tasks_status_record["tasks"] = []
    empty_tasks_status = _write_json(
        tmp_path / "empty-tasks-databricks-run-status.json",
        empty_tasks_status_record,
    )
    nested_raw_status_record = _databricks_run_status_cli_record(succeeded=True)
    nested_raw_status_record["summary"]["tasks"][0]["notebook_task"] = {
        "base_parameters": {"token": "do-not-bundle-me"}
    }
    nested_raw_status_record["summary"]["submit_payload"]["tasks"][0]["new_cluster"] = {
        "spark_conf": {"spark.secret": "do-not-bundle-me"}
    }
    nested_raw_status = _write_json(
        tmp_path / "nested-raw-databricks-run-status.json",
        nested_raw_status_record,
    )
    raw_object_in_allowed_field_record = _databricks_run_status_cli_record(succeeded=True)
    raw_object_in_allowed_field_record["summary"]["state_message"] = {
        "response": {"token": "do-not-bundle-me"}
    }
    raw_object_in_allowed_field_record["summary"]["tasks"][0]["state_message"] = {
        "notebook_task": {"base_parameters": {"token": "do-not-bundle-me"}}
    }
    raw_object_in_allowed_field_record["summary"]["submit_payload"]["run_name"] = {
        "raw_payload": {"token": "do-not-bundle-me"}
    }
    raw_object_in_allowed_field = _write_json(
        tmp_path / "raw-object-in-allowed-field-databricks-run-status.json",
        raw_object_in_allowed_field_record,
    )
    raw_object_in_wrapper_action_record = _databricks_run_status_cli_record(succeeded=True)
    raw_object_in_wrapper_action_record["action"] = {"response": {"token": "do-not-bundle-me"}}
    raw_object_in_wrapper_action = _write_json(
        tmp_path / "raw-object-in-wrapper-action-databricks-run-status.json",
        raw_object_in_wrapper_action_record,
    )
    failed_repository_hygiene = _write_json(
        tmp_path / "failed-repository-hygiene.json",
        _repository_hygiene_record(ok=False),
    )
    raw_repository_hygiene_record = _repository_hygiene_record(ok=True)
    raw_repository_hygiene_record["raw_status"] = {"tracked": ["do-not-bundle-me"]}
    raw_repository_hygiene = _write_json(
        tmp_path / "raw-repository-hygiene.json",
        raw_repository_hygiene_record,
    )
    stale_policy_repository_hygiene_record = _repository_hygiene_record(ok=True)
    stale_policy_repository_hygiene_record["required_gitignore_patterns"] = [".venv/"]
    stale_policy_repository_hygiene_record["forbidden_tracked_artifact_patterns"] = ["*.tmp"]
    stale_policy_repository_hygiene = _write_json(
        tmp_path / "stale-policy-repository-hygiene.json",
        stale_policy_repository_hygiene_record,
    )

    with pytest.raises(ValueError, match="package wheel artifact source_path"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            package_wheel=bad_wheel,
            output_dir=tmp_path / "bad-wheel-bundle",
        )

    invalid_named_wheel = _write_wheel(tmp_path / "package_wheel.whl")
    with pytest.raises(ValueError, match="valid wheel filename"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            package_wheel=invalid_named_wheel,
            output_dir=tmp_path / "bad-wheel-name-bundle",
        )

    invalid_build_tag_wheel = _write_wheel(tmp_path / "document_kv_cache-0.2.0-build-py3-none-any.whl")
    with pytest.raises(ValueError, match="valid wheel filename"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            package_wheel=invalid_build_tag_wheel,
            output_dir=tmp_path / "bad-wheel-build-tag-bundle",
        )

    valid_named_bad_payload = tmp_path / "document_kv_cache-0.2.0-py3-none-any.whl"
    valid_named_bad_payload.write_bytes(b"PK\x03\x04not a valid zip")
    with pytest.raises(ValueError, match="valid wheel zip payload"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            package_wheel=valid_named_bad_payload,
            output_dir=tmp_path / "bad-wheel-payload-bundle",
        )

    nested_dist_info_wheel = _write_wheel(
        tmp_path / "nested-dist-info-wheel" / "document_kv_cache-0.2.0-py3-none-any.whl",
        dist_info_prefix="nested/document_kv_cache-0.2.0.dist-info",
    )
    with pytest.raises(ValueError, match="root-level .dist-info"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            package_wheel=nested_dist_info_wheel,
            output_dir=tmp_path / "bad-wheel-nested-dist-info-bundle",
        )

    mismatched_dist_info_wheel = _write_wheel(
        tmp_path / "mismatched-dist-info-wheel" / "document_kv_cache-0.2.0-py3-none-any.whl",
        dist_info_prefix="document_kv_cache-0.2.1.dist-info",
    )
    with pytest.raises(ValueError, match="match wheel filename"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            package_wheel=mismatched_dist_info_wheel,
            output_dir=tmp_path / "bad-wheel-dist-info-version-bundle",
        )

    wrong_name_wheel = _write_wheel(
        tmp_path / "wrong-name-wheel" / "document_kv_cache-0.2.0-py3-none-any.whl",
        metadata_name="other-package",
    )
    with pytest.raises(ValueError, match="METADATA Name"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            package_wheel=wrong_name_wheel,
            output_dir=tmp_path / "bad-wheel-metadata-name-bundle",
        )

    missing_version_wheel = _write_wheel(
        tmp_path / "missing-version-wheel" / "document_kv_cache-0.2.1-py3-none-any.whl",
        metadata_version=None,
        dist_info_prefix="document_kv_cache-0.2.1.dist-info",
    )
    with pytest.raises(ValueError, match="METADATA Version"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            package_wheel=missing_version_wheel,
            output_dir=tmp_path / "bad-wheel-metadata-version-bundle",
        )

    with pytest.raises(ValueError, match="PR evidence sidecar ok"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            pr_evidence_jsons=(failed_pr_evidence,),
            output_dir=tmp_path / "bad-pr-evidence-bundle",
        )

    with pytest.raises(ValueError, match="GPT-5.5 findings must be resolved"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            pr_evidence_jsons=(unresolved_review_pr_evidence,),
            output_dir=tmp_path / "unresolved-review-pr-evidence-bundle",
        )

    with pytest.raises(ValueError, match="GitHub governance sidecar ok must be true"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            github_governance_json=failed_github_governance,
            output_dir=tmp_path / "failed-github-governance-bundle",
        )

    with pytest.raises(ValueError, match="open_pull_requests.unexpected_count must be 0"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            github_governance_json=unexpected_open_pr_github_governance,
            output_dir=tmp_path / "unexpected-open-pr-github-governance-bundle",
        )

    with pytest.raises(ValueError, match="open_pull_requests.truncated must be false"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            github_governance_json=truncated_open_pr_github_governance,
            output_dir=tmp_path / "truncated-open-pr-github-governance-bundle",
        )

    with pytest.raises(ValueError, match="visibility must be 'public'"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            github_governance_json=internal_github_governance,
            output_dir=tmp_path / "internal-github-governance-bundle",
        )

    with pytest.raises(ValueError, match="repository hygiene sidecar ok must be true"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            repository_hygiene_json=failed_repository_hygiene,
            output_dir=tmp_path / "failed-repository-hygiene-bundle",
        )

    with pytest.raises(ValueError, match="repository hygiene sidecar has unsupported keys"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            repository_hygiene_json=raw_repository_hygiene,
            output_dir=tmp_path / "raw-repository-hygiene-bundle",
        )

    with pytest.raises(ValueError, match="must match the current repository hygiene policy"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            repository_hygiene_json=stale_policy_repository_hygiene,
            output_dir=tmp_path / "stale-policy-repository-hygiene-bundle",
        )

    with pytest.raises(ValueError, match="unsupported keys"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            github_governance_json=raw_response_github_governance,
            output_dir=tmp_path / "raw-response-github-governance-bundle",
        )

    for raw_payload_path, bundle_name in (
        (raw_summary_github_governance, "raw-summary-github-governance-bundle"),
        (raw_branch_protection_github_governance, "raw-branch-protection-github-governance-bundle"),
        (raw_status_checks_github_governance, "raw-status-checks-github-governance-bundle"),
        (raw_pr_reviews_github_governance, "raw-pr-reviews-github-governance-bundle"),
        (raw_open_pr_github_governance, "raw-open-pr-github-governance-bundle"),
    ):
        with pytest.raises(ValueError, match="unsupported keys"):
            build_release_bundle(
                v1_benchmark_json=artifacts["v1"],
                storage_benchmark_json=artifacts["storage"],
                engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
                github_governance_json=raw_payload_path,
                output_dir=tmp_path / bundle_name,
            )

    with pytest.raises(ValueError, match=r"summary\.description must be a string or null"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            github_governance_json=raw_allowed_summary_field_github_governance,
            output_dir=tmp_path / "raw-allowed-summary-field-github-governance-bundle",
        )

    with pytest.raises(ValueError, match="contexts must be an array of strings"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            github_governance_json=raw_contexts_github_governance,
            output_dir=tmp_path / "raw-contexts-github-governance-bundle",
        )

    with pytest.raises(ValueError, match="benchmark plan execution sidecar ok"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            plan_execution_jsons=(failed_plan_execution,),
            output_dir=tmp_path / "bad-plan-execution-bundle",
        )

    bad_returncode_record = _plan_execution_record(ok=True)
    bad_returncode_record["commands"][0]["returncode"] = False
    bad_returncode_path = _write_json(tmp_path / "bad-plan-returncode.json", bad_returncode_record)
    with pytest.raises(ValueError, match=r"commands\[0\]\.returncode"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            plan_execution_jsons=(bad_returncode_path,),
            output_dir=tmp_path / "bad-plan-returncode-bundle",
        )

    with pytest.raises(ValueError, match="Databricks run status sidecar succeeded"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            databricks_run_status_jsons=(failed_run_status,),
            output_dir=tmp_path / "bad-databricks-status-bundle",
        )

    with pytest.raises(ValueError, match="raw Jobs API response"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            databricks_run_status_jsons=(raw_response_run_status,),
            output_dir=tmp_path / "raw-response-databricks-status-bundle",
        )

    with pytest.raises(ValueError, match="submit_payload must be an object"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            databricks_run_status_jsons=(missing_submit_payload_status,),
            output_dir=tmp_path / "missing-submit-payload-databricks-status-bundle",
        )

    with pytest.raises(ValueError, match="aws_g5_node_type"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            databricks_run_status_jsons=(non_g5_submit_payload_status,),
            output_dir=tmp_path / "non-g5-submit-payload-databricks-status-bundle",
        )

    with pytest.raises(ValueError, match="tasks must be a non-empty array"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            databricks_run_status_jsons=(empty_tasks_status,),
            output_dir=tmp_path / "empty-tasks-databricks-status-bundle",
        )

    with pytest.raises(ValueError, match="unsupported keys"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            databricks_run_status_jsons=(nested_raw_status,),
            output_dir=tmp_path / "nested-raw-databricks-status-bundle",
        )

    with pytest.raises(ValueError, match="state_message must be a string or null"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            databricks_run_status_jsons=(raw_object_in_allowed_field,),
            output_dir=tmp_path / "raw-object-in-allowed-field-databricks-status-bundle",
        )

    with pytest.raises(ValueError, match="wrapper.action must be 'get'"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            databricks_run_status_jsons=(raw_object_in_wrapper_action,),
            output_dir=tmp_path / "raw-object-in-wrapper-action-databricks-status-bundle",
        )


def test_build_release_bundle_rejects_existing_outputs_without_overwrite(tmp_path):
    artifacts = _write_release_ready_artifacts(tmp_path / "sources")
    bundle_dir = tmp_path / "bundle"

    build_release_bundle(
        v1_benchmark_json=artifacts["v1"],
        storage_benchmark_json=artifacts["storage"],
        engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
        output_dir=bundle_dir,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
            engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            output_dir=bundle_dir,
        )

    bundle = build_release_bundle(
        v1_benchmark_json=artifacts["v1"],
        storage_benchmark_json=artifacts["storage"],
        engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
        output_dir=bundle_dir,
        overwrite=True,
    )

    assert len(bundle.artifacts) == 6


def test_build_release_bundle_rejects_inputs_that_fail_release_evidence_validation(tmp_path):
    artifacts = _write_release_ready_artifacts(tmp_path / "sources")
    bad_v1_record = _v1_record(ok=True)
    bad_v1_record["record_type"] = "document_kv.benchmark_summary.v1"
    _write_json(Path(artifacts["v1"]), bad_v1_record)

    with pytest.raises(ValueError, match="v1 benchmark record_type"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            output_dir=tmp_path / "bundle",
        )

    assert not (tmp_path / "bundle" / RELEASE_BUNDLE_MANIFEST_FILENAME).exists()


def test_build_release_bundle_rejects_failed_release_evidence_or_preflight_sidecars(tmp_path):
    artifacts = _write_release_ready_artifacts(tmp_path / "sources")
    failed_evidence_path = _write_json(
        tmp_path / "failed-evidence.json",
        {
            "record_type": "document_kv.release_evidence.v1",
            "ok": False,
            "v1_benchmark_ok": True,
            "storage_benchmark_ok": True,
            "engine_probe_backends": ["vllm", "sglang"],
            "missing_engine_probe_backends": [],
            "duplicate_engine_probe_backends": [],
            "invalid_engine_probe_records": [],
            "issues": ["not ready"],
        },
    )
    failed_preflight_path = _write_json(
        tmp_path / "failed-preflight.json",
        {
            "record_type": "document_kv.release_evidence_inputs.v1",
            "ok": False,
            "required_engine_probe_backends": ["vllm", "sglang"],
            "missing_paths": ["missing.json"],
            "unreadable_paths": [],
            "missing_engine_probe_backends": [],
            "issues": ["missing input paths: missing.json"],
        },
    )

    with pytest.raises(ValueError, match="release evidence sidecar ok must be true"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            release_evidence_json=failed_evidence_path,
            output_dir=tmp_path / "evidence-bundle",
        )

    with pytest.raises(ValueError, match="preflight sidecar ok must be true"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            preflight_json=failed_preflight_path,
            output_dir=tmp_path / "preflight-bundle",
        )


def test_build_release_bundle_rejects_stale_release_evidence_or_preflight_sidecars(tmp_path):
    current_artifacts = _write_release_ready_artifacts(tmp_path / "current")
    stale_artifacts = _write_release_ready_artifacts(tmp_path / "stale")

    with pytest.raises(ValueError, match="artifact_sources must match"):
        build_release_bundle(
            v1_benchmark_json=current_artifacts["v1"],
            storage_benchmark_json=current_artifacts["storage"],
            engine_probe_jsons=(current_artifacts["vllm"], current_artifacts["sglang"]),
            engine_actions_jsons=(current_artifacts["vllm_actions"], current_artifacts["sglang_actions"]),
            release_evidence_json=stale_artifacts["evidence"],
            output_dir=tmp_path / "stale-evidence-bundle",
        )

    with pytest.raises(ValueError, match="input_files must match"):
        build_release_bundle(
            v1_benchmark_json=current_artifacts["v1"],
            storage_benchmark_json=current_artifacts["storage"],
            engine_probe_jsons=(current_artifacts["vllm"], current_artifacts["sglang"]),
            engine_actions_jsons=(current_artifacts["vllm_actions"], current_artifacts["sglang_actions"]),
            preflight_json=stale_artifacts["preflight"],
            output_dir=tmp_path / "stale-preflight-bundle",
        )


def test_build_release_bundle_preflights_output_collisions_before_copying(tmp_path):
    artifacts = _write_release_ready_artifacts(tmp_path / "sources")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "storage_benchmark.json").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="storage_benchmark.json"):
        build_release_bundle(
            v1_benchmark_json=artifacts["v1"],
            storage_benchmark_json=artifacts["storage"],
            engine_probe_jsons=(artifacts["vllm"], artifacts["sglang"]),
        engine_actions_jsons=(artifacts["vllm_actions"], artifacts["sglang_actions"]),
            output_dir=bundle_dir,
        )

    assert not (bundle_dir / "v1_benchmark.json").exists()
    assert not (bundle_dir / RELEASE_BUNDLE_MANIFEST_FILENAME).exists()


def test_release_bundle_rejects_non_object_json_artifacts(tmp_path):
    v1_path = tmp_path / "v1.json"
    storage_path = _write_record(tmp_path / "storage.json", "document_kv.storage_benchmark.v1")
    v1_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON root must be an object"):
        build_release_bundle(
            v1_benchmark_json=v1_path,
            storage_benchmark_json=storage_path,
            output_dir=tmp_path / "bundle",
        )


def test_release_bundle_dataclasses_validate_json_safe_schema():
    artifact = ReleaseBundleArtifact(
        role="engine_probe",
        source_path="vllm.json",
        bundled_path="engine_probe_01_vllm.json",
        size_bytes=128,
        sha256="a" * 64,
        backend="vllm",
    )
    bundle = ReleaseBundle(
        output_dir="/tmp/bundle",
        manifest_path="/tmp/bundle/manifest.json",
        artifacts=[artifact],
    )

    assert bundle.artifacts == (artifact,)

    with pytest.raises(ValueError, match="Unsupported release bundle artifact role"):
        ReleaseBundleArtifact(role="dataset", source_path="x", bundled_path="x", size_bytes=1, sha256="a")
    with pytest.raises(ValueError, match="source_path"):
        ReleaseBundleArtifact(role="v1_benchmark", source_path="", bundled_path="x", size_bytes=1, sha256="a")
    with pytest.raises(ValueError, match="sha256"):
        ReleaseBundleArtifact(role="v1_benchmark", source_path="x", bundled_path="x", size_bytes=1, sha256="a")
    with pytest.raises(ValueError, match="size_bytes"):
        ReleaseBundleArtifact(role="v1_benchmark", source_path="x", bundled_path="x", size_bytes=-1, sha256="a" * 64)
    with pytest.raises(ValueError, match="backend can only"):
        ReleaseBundleArtifact(
            role="v1_benchmark",
            source_path="x",
            bundled_path="x",
            size_bytes=1,
            sha256="a" * 64,
            backend="vllm",
        )
    with pytest.raises(TypeError, match="artifacts"):
        ReleaseBundle(output_dir="/tmp/bundle", manifest_path="/tmp/bundle/manifest.json", artifacts=(object(),))


def test_public_release_bundle_cli_writes_manifest_and_output_json(tmp_path):
    artifacts = _write_release_ready_artifacts(tmp_path / "sources")
    github_governance = _write_json(tmp_path / "github-governance.json", _github_governance_cli_record(ok=True))
    native_probe_factories = _write_json(
        tmp_path / "native-probe-factories.json",
        _native_probe_factories_record(),
    )
    output_json = tmp_path / "bundle-record.json"
    bundle_dir = tmp_path / "bundle"

    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.release_bundle",
            "--v1-benchmark-json",
            str(artifacts["v1"]),
            "--storage-benchmark-json",
            str(artifacts["storage"]),
            "--engine-probe-json",
            str(artifacts["vllm"]),
            "--engine-probe-json",
            str(artifacts["sglang"]),
            "--engine-actions-json",
            str(artifacts["vllm_actions"]),
            "--engine-actions-json",
            str(artifacts["sglang_actions"]),
            "--github-governance-json",
            str(github_governance),
            "--native-probe-factories-json",
            str(native_probe_factories),
            "--output-dir",
            str(bundle_dir),
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert json.loads(output_json.read_text(encoding="utf-8")) == json.loads(
        (bundle_dir / RELEASE_BUNDLE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    record = json.loads(output_json.read_text(encoding="utf-8"))
    assert record["artifacts"][-2]["role"] == "github_governance"
    assert record["artifacts"][-2]["record_type"] == GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE
    assert record["artifacts"][-1]["role"] == "native_probe_factories"
    assert record["artifacts"][-1]["record_type"] == NATIVE_PROBE_FACTORIES_RECORD_TYPE


def test_public_release_bundle_cli_rejects_unresolved_pr_evidence(tmp_path):
    artifacts = _write_release_ready_artifacts(tmp_path / "sources")
    unresolved_review_pr_evidence_record = _pr_evidence_record(ok=True)
    unresolved_review_pr_evidence_record["gpt55_review_outcome"] = "findings_resolved"
    unresolved_review_pr_evidence_record["gpt55_review_findings_resolved"] = False
    unresolved_review_pr_evidence = _write_json(
        tmp_path / "unresolved-review-pr-evidence.json",
        unresolved_review_pr_evidence_record,
    )

    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.release_bundle",
            "--v1-benchmark-json",
            str(artifacts["v1"]),
            "--storage-benchmark-json",
            str(artifacts["storage"]),
            "--engine-probe-json",
            str(artifacts["vllm"]),
            "--engine-probe-json",
            str(artifacts["sglang"]),
            "--pr-evidence-json",
            str(unresolved_review_pr_evidence),
            "--output-dir",
            str(tmp_path / "bundle"),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "GPT-5.5 findings must be resolved" in completed.stderr
    assert not (tmp_path / "bundle" / RELEASE_BUNDLE_MANIFEST_FILENAME).exists()


def test_public_release_bundle_cli_main_respects_public_hooks(monkeypatch, capsys, tmp_path):
    original_builder = legacy_release_bundle.build_release_bundle
    original_serializer = legacy_release_bundle.release_bundle_to_record
    original_writer = legacy_release_bundle.write_release_bundle_manifest_json
    fake_bundle = ReleaseBundle(
        output_dir=str(tmp_path / "bundle"),
        manifest_path=str(tmp_path / "bundle" / "manifest.json"),
        artifacts=(),
    )

    def fake_builder(**kwargs):
        assert kwargs["output_dir"] == str(tmp_path / "bundle")
        assert kwargs["require_complete_v1"] is True
        return fake_bundle

    def fake_serializer(bundle):
        assert bundle is fake_bundle
        return {"record_type": "fake-bundle"}

    monkeypatch.setattr(public_release_bundle, "build_release_bundle", fake_builder)
    monkeypatch.setattr(public_release_bundle, "release_bundle_to_record", fake_serializer)

    assert public_release_bundle.main(
        [
            "--v1-benchmark-json",
            "v1.json",
            "--storage-benchmark-json",
            "storage.json",
            "--require-complete-v1",
            "--output-dir",
            str(tmp_path / "bundle"),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out) == {"record_type": "fake-bundle"}
    assert legacy_release_bundle.build_release_bundle is original_builder
    assert legacy_release_bundle.release_bundle_to_record is original_serializer
    assert legacy_release_bundle.write_release_bundle_manifest_json is original_writer


def test_legacy_release_bundle_cli_main_respects_legacy_hooks(monkeypatch, capsys, tmp_path):
    output_json = tmp_path / "bundle-record.json"
    original_builder = public_release_bundle.build_release_bundle
    original_serializer = public_release_bundle.release_bundle_to_record
    original_writer = public_release_bundle.write_release_bundle_manifest_json
    fake_bundle = ReleaseBundle(
        output_dir=str(tmp_path / "bundle"),
        manifest_path=str(tmp_path / "bundle" / "manifest.json"),
        artifacts=(),
    )

    def fake_builder(**kwargs):
        assert kwargs["output_dir"] == str(tmp_path / "bundle")
        return fake_bundle

    def fake_serializer(bundle):
        assert bundle is fake_bundle
        return {"record_type": "legacy-fake-bundle"}

    monkeypatch.setattr(legacy_release_bundle, "build_release_bundle", fake_builder)
    monkeypatch.setattr(legacy_release_bundle, "release_bundle_to_record", fake_serializer)

    assert legacy_release_bundle.main(
        [
            "--v1-benchmark-json",
            "v1.json",
            "--storage-benchmark-json",
            "storage.json",
            "--output-dir",
            str(tmp_path / "bundle"),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out) == {"record_type": "legacy-fake-bundle"}
    assert legacy_release_bundle.main(
        [
            "--v1-benchmark-json",
            "v1.json",
            "--storage-benchmark-json",
            "storage.json",
            "--output-dir",
            str(tmp_path / "bundle"),
            "--output-json",
            str(output_json),
        ]
    ) == 0
    assert json.loads(output_json.read_text(encoding="utf-8")) == {"record_type": "legacy-fake-bundle"}
    assert public_release_bundle.build_release_bundle is original_builder
    assert public_release_bundle.release_bundle_to_record is original_serializer
    assert public_release_bundle.write_release_bundle_manifest_json is original_writer


def _write_record(path: Path, record_type: str, *, backend: str | None = None) -> Path:
    record = {
        "record_type": record_type,
        "payload": path.stem,
    }
    if backend is not None:
        record["backend"] = backend
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_release_ready_artifacts(source_dir: Path) -> dict[str, Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "v1": _write_json(source_dir / "v1.json", _v1_record(ok=True)),
        "storage": _write_json(source_dir / "storage.json", _storage_record(ok=True)),
        "vllm": _write_json(source_dir / "vllm-probe.json", _probe_record(ServingBackend.VLLM)),
        "sglang": _write_json(source_dir / "sglang-probe.json", _probe_record(ServingBackend.SGLANG)),
        "vllm_actions": _write_json(source_dir / "vllm-actions.json", _actions_record(ServingBackend.VLLM)),
        "sglang_actions": _write_json(source_dir / "sglang-actions.json", _actions_record(ServingBackend.SGLANG)),
    }
    evidence = evaluate_release_evidence_files(
        v1_benchmark_json=paths["v1"],
        storage_benchmark_json=paths["storage"],
        engine_probe_jsons=(paths["vllm"], paths["sglang"]),
        engine_actions_jsons=(paths["vllm_actions"], paths["sglang_actions"]),
    )
    status = inspect_release_evidence_input_files(
        v1_benchmark_json=paths["v1"],
        storage_benchmark_json=paths["storage"],
        engine_probe_jsons=(paths["vllm"], paths["sglang"]),
        engine_actions_jsons=(paths["vllm_actions"], paths["sglang_actions"]),
    )
    paths["evidence"] = source_dir / "release-evidence.json"
    paths["preflight"] = source_dir / "release-inputs.json"
    write_release_evidence_json(evidence, paths["evidence"])
    write_release_evidence_input_status_json(status, paths["preflight"])
    return paths


def _v1_record(*, ok: bool):
    datasets = ("biography", "hotpotqa", "musique", "niah")
    arms = ("baseline_prefill", "document_kv_cache")
    return {
        "record_type": BENCHMARK_RUN_RECORD_TYPE,
        "suite": {
            "hardware_target": "aws-g5",
            "model_id": "qwen3:4b-instruct",
        },
        "measurements": [
            {
                "example_id": f"{dataset}-1",
                "dataset": dataset,
                "arm_id": arm,
                "prompt_tokens": 1024,
                "completion_tokens": 16,
                "ttft_seconds": 1.0,
                "time_to_completion_seconds": 2.0,
                "answer_found": True,
                "error": None,
            }
            for dataset in datasets
            for arm in arms
        ],
        "report_rows": [
            {
                "dataset": dataset,
                "arm_id": arm,
                "requests": 1,
                "errors": 0,
                "prompt_tokens_mean": 1024.0,
                "completion_tokens_mean": 16.0,
                "ttft": {"p50": 1.0, "p95": 1.0},
                "time_to_completion": {"p50": 2.0, "p95": 2.0},
                "answer_found_rate": 1.0,
                "output_tokens_per_second": 8.0,
            }
            for dataset in datasets
            for arm in arms
        ],
        "comparisons": [
            {
                "dataset": dataset,
                "baseline_arm_id": "baseline_prefill",
                "cache_arm_id": "document_kv_cache",
                "ttft_speedup": 2.0,
                "time_to_completion_speedup": 2.0,
                "exact_match_delta": 0.0,
                "answer_found_delta": 0.0,
            }
            for dataset in datasets
        ],
        "v1_evidence": {
            "ok": ok,
            "required_datasets": list(datasets),
            "missing_report_rows": [],
            "missing_comparisons": [],
            "comparisons_without_metrics": [],
            "rows_without_successful_requests": [],
            "rows_without_latency": [],
            "rows_without_quality": [],
            "unexpected_datasets": [],
            "issues": [] if ok else ["missing report rows: hotpotqa:baseline_prefill"],
        },
    }


def _storage_record(*, ok: bool):
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
        "uc_volume_is_real": True,
        "release_storage_evidence": {
            "ok": ok,
            "required_readers": list(readers),
            "missing_readers": [],
            "readers_with_errors": [],
            "readers_without_latency": [],
            "readers_without_throughput": [],
            "require_real_uc_volume": True,
            "uc_volume_root": "/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            "uc_volume_is_real": True,
            "issues": [] if ok else ["missing storage readers: unity_catalog"],
        },
    }


def _probe_record(backend: ServingBackend):
    layout = layout_for_model("qwen3:4b-instruct")
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


def _actions_record(backend: ServingBackend, *, layout=None):
    layout = layout or layout_for_model("qwen3:4b-instruct")
    request_id = f"{backend.value}-probe"
    return engine_kv_connector_actions_to_record(
        EngineKVConnectorActions(
            reservation=EngineKVReservationAction(
                backend=backend,
                request_id=request_id,
                total_blocks=1,
                total_tokens=1,
                estimated_gpu_bytes=layout.bytes_per_token,
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
                    source_byte_length=layout.bytes_per_token,
                    global_byte_start=0,
                    global_byte_end=layout.bytes_per_token,
                    token_start=0,
                    token_count=1,
                    token_end=1,
                    first_block_index=0,
                    last_block_index_exclusive=1,
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


def _write_json(path: Path, record) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_wheel(
    path: Path,
    *,
    metadata_name: str = "document-kv-cache",
    metadata_version: str | None = "0.2.0",
    dist_info_prefix: str = "document_kv_cache-0.2.0.dist-info",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_lines = [f"Name: {metadata_name}"]
    if metadata_version is not None:
        metadata_lines.append(f"Version: {metadata_version}")
    metadata_lines.append("")
    with zipfile.ZipFile(path, "w") as wheel_zip:
        wheel_zip.writestr("document_kv_cache/__init__.py", "")
        wheel_zip.writestr(
            f"{dist_info_prefix}/WHEEL",
            "\n".join(
                [
                    "Wheel-Version: 1.0",
                    "Generator: document-kv-cache test fixture",
                    "Root-Is-Purelib: true",
                    "Tag: py3-none-any",
                    "",
                ]
            ),
        )
        wheel_zip.writestr(f"{dist_info_prefix}/METADATA", "\n".join(metadata_lines))
        wheel_zip.writestr(f"{dist_info_prefix}/RECORD", "")
    return path


def _pr_evidence_record(*, ok: bool):
    return {
        "record_type": "document_kv.pr_evidence.v1",
        "ok": ok,
        "what_changed": ["release provenance"],
        "why": "release bundles should be auditable",
        "scope": ["release_bundle.py"],
        "verification": ["pytest"],
        "refactor_skill_applied": True,
        "gpt55_review_completed": True,
        "gpt55_review_findings_resolved": True,
        "gpt55_review_outcome": "clean",
        "gpt55_review_summary": "clean" if ok else "",
        "issues": [] if ok else ["missing review summary"],
    }


def _github_governance_cli_record(*, ok: bool):
    summary = {
        "record_type": GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE,
        "ok": ok,
        "repository": "owner/document-kv-cache",
        "default_branch": "main",
        "branch": "main",
        "private": False,
        "visibility": "public",
        "archived": False,
        "disabled": False,
        "description": "Cachet document KV cache.",
        "homepage": "https://github.com/owner/document-kv-cache",
        "topics": ["cachet", "kv-cache"],
        "branch_protection": {
            "enabled": ok,
            "required_status_checks": {
                "strict": ok,
                "contexts": ["Test and build"] if ok else [],
            },
            "required_pull_request_reviews": {
                "dismiss_stale_reviews": ok,
                "require_last_push_approval": ok,
                "required_approving_review_count": 1 if ok else 0,
            },
            "required_linear_history": ok,
            "required_conversation_resolution": ok,
            "enforce_admins": ok,
            "allow_force_pushes": False,
            "allow_deletions": False,
        },
        "open_pull_requests": {
            "checked": ok,
            "total_count": 0,
            "allowed_numbers": [],
            "unexpected_count": 0,
            "unexpected": [],
            "truncated": False,
        },
        "issues": [] if ok else ["main branch protection must be enabled"],
    }
    return {"ok": ok, "summary": summary}


def _repository_hygiene_record(*, ok: bool):
    return {
        "record_type": REPOSITORY_HYGIENE_RECORD_TYPE,
        "ok": ok,
        "repository_root": "/workspace/document-kv-cache",
        "tracked_path_count": 128,
        "required_gitignore_patterns": list(REQUIRED_GITIGNORE_PATTERNS),
        "missing_gitignore_patterns": [] if ok else [".env"],
        "forbidden_tracked_artifact_patterns": list(FORBIDDEN_TRACKED_ARTIFACT_PATTERNS),
        "forbidden_tracked_paths": [] if ok else ["dist/document_kv_cache-0.2.0-py3-none-any.whl"],
        "issues": [] if ok else ["forbidden generated or secret-like tracked artifacts"],
    }


def _native_probe_factories_record():
    return {
        "record_type": NATIVE_PROBE_FACTORIES_RECORD_TYPE,
        "factories": [
            _native_probe_factory_record("vllm", VLLM_NATIVE_PROBE_FACTORY),
            _native_probe_factory_record("sglang", SGLANG_NATIVE_PROBE_FACTORY),
        ],
    }


def _native_probe_factory_record(backend: str, factory_path: str):
    return {
        "backend": backend,
        "factory_path": factory_path,
        "package_name": backend,
        "package_importable": False,
        "package_version": None,
        "serving_environment_profile": serving_environment_profile_to_record(
            serving_environment_profile(backend)
        ),
        "supported": False,
        "reason": f"{backend} adapter is not installed",
    }


def _plan_execution_record(*, ok: bool):
    return {
        "record_type": BENCHMARK_PLAN_EXECUTION_RECORD_TYPE,
        "ok": ok,
        "plan_source": {
            "record_type": BENCHMARK_PLAN_SOURCE_RECORD_TYPE,
            "path": "dbfs:/benchmarks/v1-plan.json",
            "driver_path": "/dbfs/benchmarks/v1-plan.json",
            "size_bytes": 512,
            "sha256": "a" * 64,
            "suite_id": "v1-suite",
            "model_id": "qwen3:4b-instruct",
            "hardware_target": "aws-g5",
            "command_count": 1,
        },
        "commands": [
            {
                "name": "run-benchmark",
                "argv": ["python", "-m", "document_kv_cache.benchmark_runner"],
                "returncode": 0 if ok else 2,
                "skipped": False,
                "error": None if ok else "failed",
            }
        ],
    }


def _databricks_run_status_cli_record(
    *,
    succeeded: bool,
    purpose: str = "document-kv-v1-benchmark",
    run_name: str = "document-kv-v1",
    task_key: str = "run-benchmark",
):
    return {
        "ok": True,
        "action": "get",
        "summary": _databricks_run_status_record(
            succeeded=succeeded,
            purpose=purpose,
            run_name=run_name,
            task_key=task_key,
        ),
    }


def _databricks_run_status_record(
    *,
    succeeded: bool,
    purpose: str = "document-kv-v1-benchmark",
    run_name: str = "document-kv-v1",
    task_key: str = "run-benchmark",
):
    life_cycle_state = "TERMINATED" if succeeded else "RUNNING"
    result_state = "SUCCESS" if succeeded else None
    return {
        "record_type": DATABRICKS_RUN_STATUS_RECORD_TYPE,
        "run_id": 123,
        "run_name": run_name,
        "run_page_url": "https://dbc.example/#job/123",
        "life_cycle_state": life_cycle_state,
        "result_state": result_state,
        "state_message": None,
        "start_time": 1000,
        "end_time": 2000 if succeeded else None,
        "terminal": succeeded,
        "succeeded": succeeded,
        "active_task_key": None if succeeded else task_key,
        "task_count": 1,
        "tasks": [
            {
                "task_key": task_key,
                "run_id": 124,
                "life_cycle_state": life_cycle_state,
                "result_state": result_state,
                "state_message": None,
                "cluster_id": "cluster-123",
                "start_time": 1001,
                "end_time": 2000 if succeeded else None,
            }
        ],
        "cluster_id": "cluster-123",
        "submit_payload": _databricks_run_submit_payload_record(
            purpose=purpose,
            run_name=run_name,
            task_key=task_key,
        ),
    }


def _databricks_run_submit_payload_record(
    *,
    purpose: str = "document-kv-v1-benchmark",
    run_name: str = "document-kv-v1",
    task_key: str = "run-benchmark",
):
    return {
        "record_type": DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE,
        "source_path": "/Volumes/catalog/schema/volume/databricks-run-submit.json",
        "sha256": "a" * 64,
        "run_name": run_name,
        "task_count": 1,
        "task_keys": [task_key],
        "tasks": [
            {
                "task_key": task_key,
                "node_type_id": "g5.4xlarge",
                "driver_node_type_id": "g5.4xlarge",
                "spark_version": "15.4.x-gpu-ml-scala2.12",
                "data_security_mode": "SINGLE_USER",
                "num_workers": 0,
                "single_node": True,
                "purpose": purpose,
            }
        ],
        "node_type_ids": ["g5.4xlarge"],
        "driver_node_type_ids": ["g5.4xlarge"],
        "spark_versions": ["15.4.x-gpu-ml-scala2.12"],
        "data_security_modes": ["SINGLE_USER"],
        "single_node": True,
        "aws_g5_node_type": True,
    }
