from __future__ import annotations

import copy
import inspect
import json
import os
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import document_kv_cache.gpu_qualification_databricks as qualification_job
import document_kv_cache.publication_handoff_closure_coordinator as coordinator
from document_kv_cache.databricks_resource_ledger import (
    DatabricksClusterHourLedger,
    databricks_cluster_hour_ledger_to_record,
    databricks_ledger_prefix,
)
from document_kv_cache.databricks_runs import DatabricksWorkspaceConfig
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PATCHED_WHEEL_SHA256,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    GPUQualificationSelection,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_EVIDENCE_RECORD_TYPE,
    GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE,
    GPUQualificationArtifactPinsV2,
)
from document_kv_cache.gpu_qualification_databricks import (
    GPUQualificationLaunchAuthorization,
)
from document_kv_cache.publication_latency_handoff_generation import (
    PublicationLatencyGeneratorHardwareQualificationV2,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
)


VOLUME_ROOT = "dbfs:/Volumes/catalog/schema/volume"
TEST_WORKSPACE_HOST = "https://workspace.example"
TEST_WORKSPACE_USER = "publication@example.com"


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _authenticated_test_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    def authenticate(workspace: DatabricksWorkspaceConfig, *, expected_user_name: str):
        assert expected_user_name == TEST_WORKSPACE_USER
        return {
            "workspace_host_sha256": sha256(
                workspace.normalized_host.encode("utf-8")
            ).hexdigest(),
            "user_name_sha256": sha256(expected_user_name.encode("utf-8")).hexdigest(),
        }

    monkeypatch.setattr(
        coordinator, "require_databricks_current_user_name", authenticate
    )


def _config(
    *,
    request_root_uri: str | None = None,
    output_root_uri: str | None = None,
    stage: str = "q8",
):
    output_root = output_root_uri or f"{VOLUME_ROOT}/handoffs/{stage}"
    return coordinator.PublicationHandoffClosureCoordinatorConfig(
        runner_python_file=f"{VOLUME_ROOT}/inputs/coordinator-runner.py",
        package_wheel_uri=f"{VOLUME_ROOT}/inputs/cachet.whl",
        package_wheel_sha256=_digest("package-wheel"),
        runtime_lock_uri=f"{VOLUME_ROOT}/inputs/runtime.lock",
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_uri=f"{VOLUME_ROOT}/inputs/vllm.whl",
        patched_vllm_wheel_sha256=GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        patched_flashinfer_wheel_uri=f"{VOLUME_ROOT}/inputs/flashinfer.whl",
        patched_flashinfer_wheel_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
        runtime_closure_manifest_uri=(f"{VOLUME_ROOT}/inputs/runtime-closure.json"),
        runtime_closure_manifest_sha256=RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        source_closure_uri=f"{VOLUME_ROOT}/inputs/cachet-source-closure.json",
        cachet_source_tree_sha256=_digest("source-closure-file"),
        request_root_uri=(
            coordinator._handoff_closure_request_root_uri(output_root, stage=stage)
            if request_root_uri is None
            else request_root_uri
        ),
        source_revision="a" * 40,
        single_user_name="publication@example.com",
    )


def _qualified_pins(
    config: coordinator.PublicationHandoffClosureCoordinatorConfig,
    *,
    input_bundle_sha256: str = GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    runner_sha256: str | None = None,
) -> GPUQualificationArtifactPinsV2:
    return GPUQualificationArtifactPinsV2(
        runtime_lock_sha256=config.runtime_lock_sha256,
        patched_vllm_wheel_sha256=config.patched_vllm_wheel_sha256,
        patched_flashinfer_wheel_sha256=config.patched_flashinfer_wheel_sha256,
        runtime_closure_manifest_sha256=(config.runtime_closure_manifest_sha256),
        package_wheel_sha256=config.package_wheel_sha256,
        cachet_source_tree_sha256=config.cachet_source_tree_sha256,
        runner_sha256=runner_sha256 or _digest("qualified-producer-runner"),
        input_bundle_sha256=input_bundle_sha256,
    )


def _request(*, stage: str = "q8", large: bool = False) -> dict[str, Any]:
    ledger = DatabricksClusterHourLedger(ledger_id="campaign-ledger")
    ledger_record = databricks_cluster_hour_ledger_to_record(ledger)
    prefix = databricks_ledger_prefix(ledger).to_record()
    output_root_uri = f"{VOLUME_ROOT}/handoffs/{stage}"
    config = _config(output_root_uri=output_root_uri, stage=stage)
    execution_contract: dict[str, Any] = {"contract": "exact-test-contract"}
    if large:
        execution_contract["production_artifact_closure"] = [
            _digest(f"artifact-{index}") for index in range(256)
        ]
    evidence = []
    for worker_index in range(16):
        item = {
            "attempt_id": f"{stage}-worker-{worker_index:02d}",
            "attestation_closed_record_sha256": _digest(
                f"{stage}-attestation-closed-{worker_index}"
            ),
            "attestation_file_sha256": _digest(
                f"{stage}-attestation-file-{worker_index}"
            ),
            "worker_index": worker_index,
        }
        if stage == coordinator.PUBLICATION_HANDOFF_CLOSURE_BF16_STAGE:
            item["control_plane_status_sha256"] = _digest(
                f"{stage}-control-{worker_index}"
            )
        evidence.append(item)
    controller_lease_root = _test_controller_lease_root(stage)
    phase_evidence = {
        "batch_marker_closed_record_sha256": _digest("batch-marker-closed"),
        "batch_marker_file_sha256": _digest("batch-marker-file"),
        "phase_lease_closed_record_sha256": _digest("phase-lease-closed"),
        "phase_lease_file_sha256": _digest("phase-lease-file"),
        "phase_lease_root_sha256": _digest("phase-lease-root"),
    }
    batch = {
        "attempt_ids": [item["attempt_id"] for item in evidence],
        "batch_prefix": copy.deepcopy(prefix),
        "ledger_path_sha256": _digest("controller-ledger-path"),
        "predecessor_prefix": copy.deepcopy(prefix),
        "submit_payload_sha256s": [
            _digest(f"typed-submit-{index}") for index in range(16)
        ],
    }
    singleton: dict[str, Any] = {
        "batch_identity_sha256": coordinator._closure_batch_identity_sha256(batch),
        "controller_lease_root_sha256": coordinator._controller_path_sha256(
            controller_lease_root,
            domain="cachet.publication.handoff_closure.controller_lease.v2",
        ),
        "durable_output_root_uri": output_root_uri,
        "phase_evidence": phase_evidence,
        "stage": stage,
    }
    singleton["identity_sha256"] = coordinator._canonical_sha256(
        {
            "domain": "cachet.publication.handoff_closure.singleton.v2",
            **singleton,
        }
    )
    attempt_id = coordinator._handoff_closure_attempt_id(singleton)
    request: dict[str, Any] = {
        "attempt_id": attempt_id,
        "closed_record_sha256": "",
        "coordinator": config.to_record(),
        "controller_singleton": singleton,
        "execution_contract": execution_contract,
        "execution_contract_sha256": coordinator._canonical_sha256(execution_contract),
        "expected_qualification_closed_record_sha256": _digest("qualification"),
        "input_bundle_sha256": GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
        "ledger_lineage": {
            "ledger_id": ledger.ledger_id,
            "ledger_path_sha256": _digest("controller-ledger-path"),
            "predecessor_prefix": prefix,
            "producer_batch_prefix": prefix,
            "terminal_prefix": prefix,
        },
        "ledger_snapshot": {
            "record": ledger_record,
            "record_sha256": coordinator._canonical_sha256(ledger_record),
        },
        "output_root_uri": output_root_uri,
        "plan": {
            "closed_record_sha256": _digest(f"{stage}-plan-closed"),
            "file_sha256": _digest(f"{stage}-plan-file"),
            "uri": f"{VOLUME_ROOT}/plans/{stage}-plan.json",
        },
        "prepared_input_root_uri": f"{VOLUME_ROOT}/prepared/main-latency",
        "qualified_artifact_pins": _qualified_pins(config).to_record(),
        "record_type": coordinator.PUBLICATION_HANDOFF_CLOSURE_REQUEST_RECORD_TYPE,
        "request_uri": (f"{config.request_root_uri}/request.json"),
        "result_uri": coordinator._handoff_closure_result_uri(
            output_root_uri, stage=stage
        ),
        "schema_version": coordinator.PUBLICATION_HANDOFF_CLOSURE_SCHEMA_VERSION,
        "stage": stage,
        "worker_evidence": evidence,
    }
    request["closed_record_sha256"] = coordinator._closed_record_sha256(request)
    coordinator._validate_closure_request(request)
    return request


def _test_controller_lease_root(stage: str) -> Path:
    return Path.cwd() / ".test-handoff-controller" / stage


def _rebind_request_controller_lease(request: dict[str, Any], root: str | Path) -> None:
    singleton = request["controller_singleton"]
    singleton["controller_lease_root_sha256"] = coordinator._controller_path_sha256(
        root,
        domain="cachet.publication.handoff_closure.controller_lease.v2",
    )
    identity = {
        key: value for key, value in singleton.items() if key != "identity_sha256"
    }
    singleton["identity_sha256"] = coordinator._canonical_sha256(
        {
            "domain": "cachet.publication.handoff_closure.singleton.v2",
            **identity,
        }
    )
    request["attempt_id"] = coordinator._handoff_closure_attempt_id(singleton)
    request["closed_record_sha256"] = coordinator._closed_record_sha256(request)


def _authorize_request(
    request: dict[str, Any],
    *,
    evidence_label: str = "typed-submit",
    batch_label: str = "typed-submit",
    controller_lease_root: str | Path | None = None,
) -> coordinator.PublicationHandoffClosureRequestAuthorization:
    lease_root = (
        _test_controller_lease_root(request["stage"])
        if controller_lease_root is None
        else Path(controller_lease_root)
    )
    _rebind_request_controller_lease(request, lease_root)
    evidence = copy.deepcopy(request["worker_evidence"])
    lineage = request["ledger_lineage"]
    qualified_pins = coordinator._q8.gpu_qualification_artifact_pins_v2_from_record(
        request["qualified_artifact_pins"]
    )
    qualification_plan_sha256 = _digest("qualification-plan-closed")
    return coordinator.PublicationHandoffClosureRequestAuthorization(
        request=request,
        batch_evidence={
            "batch_authorization": {
                "attempt_ids": [item["attempt_id"] for item in evidence],
                "batch_prefix": copy.deepcopy(lineage["producer_batch_prefix"]),
                "ledger_path_sha256": lineage["ledger_path_sha256"],
                "predecessor_prefix": copy.deepcopy(lineage["predecessor_prefix"]),
                "submit_payload_sha256s": [
                    _digest(f"{batch_label}-{index}") for index in range(16)
                ],
            },
            "phase_evidence": copy.deepcopy(
                request["controller_singleton"]["phase_evidence"]
            ),
            "worker_evidence": evidence,
        },
        qualified_artifact_pins=qualified_pins,
        qualification_authorization_binding={
            "artifact_pins_sha256": coordinator._canonical_sha256(
                qualified_pins.to_record()
            ),
            "authorization_causal_closure_sha256": _digest(
                f"qualification-causal-{evidence_label}"
            ),
            "authorization_ledger_id": lineage["ledger_id"],
            "authorization_ledger_path_sha256": lineage["ledger_path_sha256"],
            "authorization_ledger_prefix": copy.deepcopy(lineage["predecessor_prefix"]),
            "evidence_closed_record_sha256": request[
                "expected_qualification_closed_record_sha256"
            ],
            "evidence_file_sha256": _digest("qualification-evidence-file"),
            "evidence_uri": "dbfs:/qualification/evidence.json",
            "plan_closed_record_sha256": qualification_plan_sha256,
            "plan_file_sha256": _digest("qualification-plan-file"),
            "plan_uri": "dbfs:/qualification/plan.json",
            "selection": {
                "attention_backend": "TRITON_ATTN",
                "generation_artifacts_sha256": _digest("generation-artifacts"),
                "generation_databricks_node_type_id": "g6e.4xlarge",
                "generation_hardware_id": "aws-g6e-l40s",
                "generation_prefix_tokens_per_second": 40.0,
                "gpu_memory_utilization": 0.75,
                "plan_sha256": qualification_plan_sha256,
            },
        },
        controller_lease_root=lease_root,
        workspace_identity={
            "workspace_host_sha256": _digest(TEST_WORKSPACE_HOST),
            "user_name_sha256": _digest(TEST_WORKSPACE_USER),
        },
        _issuer=coordinator._REQUEST_AUTHORIZATION_ISSUER,
    )


def _hardware_qualification(
    monkeypatch: pytest.MonkeyPatch,
    config: coordinator.PublicationHandoffClosureCoordinatorConfig,
    *,
    input_bundle_sha256: str,
) -> PublicationLatencyGeneratorHardwareQualificationV2:
    selection = GPUQualificationSelection(
        attention_backend="TRITON_ATTN",
        gpu_memory_utilization=0.75,
        generation_hardware_id="aws-g6e-l40s",
        generation_databricks_node_type_id="g6e.4xlarge",
        generation_artifacts_sha256=_digest("generation-artifacts"),
        generation_prefix_tokens_per_second=40.0,
        plan_sha256=_digest("qualification-plan-closed"),
    )
    monkeypatch.setattr(
        coordinator._q8,
        "validate_gpu_qualification_evidence_v2_record",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        coordinator._q8,
        "validate_gpu_qualification_plan_v2_record",
        lambda *_args, **_kwargs: None,
    )
    return PublicationLatencyGeneratorHardwareQualificationV2(
        evidence_record={
            "closed_record_sha256": _digest("qualification"),
            "record_type": GPU_QUALIFICATION_V2_EVIDENCE_RECORD_TYPE,
        },
        plan_record={
            "closed_record_sha256": selection.plan_sha256,
            "record_type": GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE,
        },
        expected_campaign_id="vllm-0271-publication-v1",
        expected_artifact_pins=_qualified_pins(
            config,
            input_bundle_sha256=input_bundle_sha256,
        ),
        evidence_uri="dbfs:/qualification/evidence.json",
        evidence_file_sha256=_digest("qualification-evidence-file"),
        plan_uri="dbfs:/qualification/plan.json",
        plan_file_sha256=_digest("qualification-plan-file"),
    )


def _qualification_launch_authorization(
    hardware_qualification: PublicationLatencyGeneratorHardwareQualificationV2,
) -> GPUQualificationLaunchAuthorization:
    ledger = DatabricksClusterHourLedger(ledger_id="qualification-ledger")
    prefix = databricks_ledger_prefix(ledger)
    return GPUQualificationLaunchAuthorization(
        selection=hardware_qualification.selection,
        plan_sha256=hardware_qualification.selection.plan_sha256,
        evidence_closed_record_sha256=hardware_qualification.evidence_record[
            "closed_record_sha256"
        ],
        evidence_file_sha256=hardware_qualification.evidence_file_sha256,
        ledger_id=ledger.ledger_id,
        ledger_path_sha256=_digest("qualification-ledger-path"),
        predecessor_prefix=prefix,
        producer_batch_prefix=prefix,
        ledger_prefix=prefix,
        causal_closure_sha256=_digest("qualification-causal-closure"),
        _issuer=qualification_job._LAUNCH_AUTHORIZATION_ISSUER,
    )


def _allow_synthetic_manifests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        coordinator._handoff_artifacts,
        "_validated_bundle_record",
        lambda record: dict(record),
    )
    monkeypatch.setattr(
        coordinator._q8,
        "_validate_publication_manifest_contract",
        lambda _record: None,
    )
    monkeypatch.setattr(
        coordinator._bf16,
        "_validate_bf16_manifest_contract",
        lambda _record: None,
    )


def _result(request: dict[str, Any], *, run_id: str = "12345") -> dict[str, Any]:
    stage = request["stage"]
    contexts = [8192, 16384, 32768] if stage == "q8" else [16384]
    manifests = []
    bundles = []
    for context_tokens in contexts:
        portable = _digest(f"{stage}-portable-{context_tokens}")
        manifest: dict[str, Any] = {
            "closed_record_sha256": "",
            "context_tokens": context_tokens,
            "input_bundle_sha256": request["input_bundle_sha256"],
            "portable_bundle_sha256": portable,
        }
        manifest["closed_record_sha256"] = coordinator._closed_record_sha256(manifest)
        manifest_relative = (
            f"manifests/{context_tokens}-{portable}.json"
            if stage == "q8"
            else (
                "manifests/"
                f"{coordinator._bf16.PUBLICATION_BF16_HANDOFF_MANIFEST_FILENAME}"
            )
        )
        source_relative = f"bundles/{context_tokens}-{portable}"
        manifest_file_sha256 = sha256(
            coordinator._canonical_json_bytes(manifest, pretty=True)
        ).hexdigest()
        bundle = {
            "closed_record_sha256": manifest["closed_record_sha256"],
            "context_tokens": context_tokens,
            "manifest_relative_path": manifest_relative,
            "portable_bundle_sha256": portable,
            "source_root_relative_path": source_relative,
        }
        if stage == "bf16":
            bundle["manifest_file_sha256"] = manifest_file_sha256
        bundles.append(bundle)
        manifests.append(
            {
                "closed_record_sha256": manifest["closed_record_sha256"],
                "context_tokens": context_tokens,
                "file_sha256": manifest_file_sha256,
                "portable_bundle_sha256": portable,
                "record": manifest,
                "source_root_uri": (f"{request['output_root_uri']}/{source_relative}"),
                "uri": f"{request['output_root_uri']}/{manifest_relative}",
            }
        )
    reconciliation: dict[str, Any] = {"ledger_id": "campaign-ledger"}
    if stage == "q8":
        attempts = [
            {
                "attempt_id": request["worker_evidence"][index]["attempt_id"],
                "attestation_closed_record_sha256": request["worker_evidence"][index][
                    "attestation_closed_record_sha256"
                ],
                "attestation_file_sha256": request["worker_evidence"][index][
                    "attestation_file_sha256"
                ],
                "verification_source": "direct_databricks_runs_get",
                "worker_index": index,
            }
            for index in range(16)
        ]
        reconciliation.update(
            {
                "attempts": attempts,
                "attempts_sha256": coordinator._canonical_sha256(attempts),
            }
        )
    else:
        attempts = [
            {
                "attempt_id": request["worker_evidence"][index]["attempt_id"],
                "attestation_closed_record_sha256": request["worker_evidence"][index][
                    "attestation_closed_record_sha256"
                ],
                "worker_index": index,
            }
            for index in range(16)
        ]
        reconciliation.update(
            {
                "attempt_count": 16,
                "attempts": attempts,
                "attempts_sha256": coordinator._canonical_sha256(attempts),
                "verification_source": "direct_databricks_runs_get",
            }
        )
    execution_record: dict[str, Any] = {
        "accounting": {
            "coordinator_gpu_hours": 0.0,
            "payload_copy_count_during_closure": 0,
            "worker_count": 16,
        },
        "closed_record_sha256": "",
        "execution_contract": copy.deepcopy(request["execution_contract"]),
        "execution_mode": (
            coordinator._q8.PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED
            if stage == "q8"
            else coordinator._bf16.PUBLICATION_BF16_HANDOFF_EXECUTION_MODE
        ),
        "generator_hardware": {
            "qualification_closed_record_sha256": request[
                "expected_qualification_closed_record_sha256"
            ]
        },
        "input_bundle_sha256": request["input_bundle_sha256"],
        "ledger_reconciliation": reconciliation,
        "plan_closed_record_sha256": request["plan"]["closed_record_sha256"],
        "record_type": (
            coordinator._q8.PUBLICATION_LATENCY_HANDOFF_EXECUTION_RECORD_TYPE
            if stage == "q8"
            else coordinator._bf16.PUBLICATION_BF16_HANDOFF_EXECUTION_RECORD_TYPE
        ),
        "schema_version": (
            coordinator._q8.PUBLICATION_LATENCY_HANDOFF_EXECUTION_SCHEMA_VERSION
            if stage == "q8"
            else coordinator._bf16.PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION
        ),
        "serving_reuse": {},
        "workers": [{"worker_index": index} for index in range(16)],
    }
    if stage == "q8":
        execution_record["bundles"] = bundles
        execution_record["bundles_sha256"] = coordinator._canonical_sha256(bundles)
        execution_record["serving_reuse"] = {"context_bundles": copy.deepcopy(bundles)}
        execution_filename = (
            coordinator._q8.PUBLICATION_LATENCY_HANDOFF_EXECUTION_FILENAME
        )
    else:
        execution_record["bundle"] = bundles[0]
        execution_filename = (
            coordinator._bf16.PUBLICATION_BF16_HANDOFF_EXECUTION_FILENAME
        )
    execution_record["closed_record_sha256"] = (
        coordinator._q8._closed_record_sha256(execution_record)
        if stage == "q8"
        else coordinator._bf16._closed_record_sha256(execution_record)
    )
    result: dict[str, Any] = {
        "closed_record_sha256": "",
        "coordinator": {
            "close_function": (
                "close_publication_latency_handoff_generation_from_workers"
                if stage == "q8"
                else "close_publication_bf16_handoff_generation_from_workers"
            ),
            "node_type_id": coordinator.PUBLICATION_HANDOFF_CLOSURE_NODE_TYPE_ID,
            "run_id": run_id,
            "runner_sha256": coordinator.PUBLICATION_HANDOFF_CLOSURE_RUNNER_SHA256,
            "tree_validation": "full_mounted_byte_replay",
        },
        "execution": {
            "closed_record_sha256": execution_record["closed_record_sha256"],
            "file_sha256": sha256(
                coordinator._canonical_json_bytes(execution_record, pretty=True)
            ).hexdigest(),
            "record": execution_record,
            "uri": f"{request['output_root_uri']}/{execution_filename}",
        },
        "ledger_lineage": copy.deepcopy(request["ledger_lineage"]),
        "manifests": manifests,
        "output_root_uri": request["output_root_uri"],
        "pins": {
            "coordinator": copy.deepcopy(request["coordinator"]),
            "execution_contract_sha256": request["execution_contract_sha256"],
            "input_bundle_sha256": request["input_bundle_sha256"],
            "plan": copy.deepcopy(request["plan"]),
            "qualified_artifact_pins": copy.deepcopy(
                request["qualified_artifact_pins"]
            ),
            "qualification_closed_record_sha256": request[
                "expected_qualification_closed_record_sha256"
            ],
        },
        "record_type": coordinator.PUBLICATION_HANDOFF_CLOSURE_RESULT_RECORD_TYPE,
        "request_closed_record_sha256": request["closed_record_sha256"],
        "schema_version": coordinator.PUBLICATION_HANDOFF_CLOSURE_SCHEMA_VERSION,
        "stage": stage,
    }
    result["closed_record_sha256"] = coordinator._closed_record_sha256(result)
    return result


def test_volume_uris_use_the_uc_volume_mount_not_dbfs() -> None:
    assert coordinator._cluster_path(f"{VOLUME_ROOT}/inputs/plan.json") == Path(
        "/Volumes/catalog/schema/volume/inputs/plan.json"
    )
    assert 'if uri.startswith("dbfs:/Volumes/"):' in (
        coordinator.PUBLICATION_HANDOFF_CLOSURE_RUNNER_SCRIPT
    )
    assert 'return "/Volumes/" + uri.removeprefix("dbfs:/Volumes/")' in (
        coordinator.PUBLICATION_HANDOFF_CLOSURE_RUNNER_SCRIPT
    )


def test_source_closure_requires_native_v2_runtime_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    runtime = coordinator._native_v2_source_closure_runtime_identity()
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "files": [
            {
                "role": "cachet_package_wheel",
                "sha256": config.package_wheel_sha256,
            }
        ],
        "git": {"commit": config.source_revision},
        "record_type": "cachet.publication_source_closure.v2",
        "runtime": runtime,
        "schema_version": 2,
    }
    record["closed_record_sha256"] = coordinator._closed_record_sha256(record)
    source_path = tmp_path / "source-closure.json"
    source_bytes = coordinator._canonical_json_bytes(record, pretty=True)
    source_path.write_bytes(source_bytes)
    bound = replace(
        config,
        cachet_source_tree_sha256=sha256(source_bytes).hexdigest(),
    )
    monkeypatch.setattr(coordinator, "_cluster_path", lambda _uri: source_path)
    coordinator._verify_source_closure(bound.to_record())

    tampered_runtime = copy.deepcopy(record)
    tampered_runtime["runtime"]["base_lock"]["byte_count"] -= 1
    tampered_runtime["closed_record_sha256"] = coordinator._closed_record_sha256(
        tampered_runtime
    )
    tampered_bytes = coordinator._canonical_json_bytes(tampered_runtime, pretty=True)
    source_path.write_bytes(tampered_bytes)
    tampered_bound = replace(
        config,
        cachet_source_tree_sha256=sha256(tampered_bytes).hexdigest(),
    )
    with pytest.raises(ValueError, match="runtime identity drift"):
        coordinator._verify_source_closure(tampered_bound.to_record())

    legacy = copy.deepcopy(record)
    legacy["record_type"] = "cachet.publication_source_closure.v1"
    legacy["schema_version"] = 1
    legacy["closed_record_sha256"] = coordinator._closed_record_sha256(legacy)
    legacy_bytes = coordinator._canonical_json_bytes(legacy, pretty=True)
    source_path.write_bytes(legacy_bytes)
    legacy_bound = replace(
        config,
        cachet_source_tree_sha256=sha256(legacy_bytes).hexdigest(),
    )
    with pytest.raises(ValueError, match="source closure identity drift"):
        coordinator._verify_source_closure(legacy_bound.to_record())


def _terminal(payload: dict[str, Any], *, succeeded: bool = True) -> dict[str, Any]:
    result_state = "SUCCESS" if succeeded else "FAILED"
    return {
        "cluster_instance": {"cluster_id": "cluster-coordinator"},
        "end_time": 1_002_000,
        "original_attempt_run_id": 12345,
        "repair_history": [],
        "run_id": 12345,
        "run_name": payload["run_name"],
        "start_time": 1_000_000,
        "state": {"life_cycle_state": "TERMINATED", "result_state": result_state},
        "tasks": [
            {
                "attempt_number": 0,
                "cluster_instance": {"cluster_id": "cluster-coordinator"},
                "end_time": 1_001_900,
                "new_cluster": copy.deepcopy(payload["tasks"][0]["new_cluster"]),
                "run_id": 22345,
                "spark_python_task": copy.deepcopy(
                    payload["tasks"][0]["spark_python_task"]
                ),
                "start_time": 1_000_100,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": result_state,
                },
                "task_key": payload["tasks"][0]["task_key"],
            }
        ],
    }


def _write_collection_reservation(
    root: Path,
    request_authorization: coordinator.PublicationHandoffClosureRequestAuthorization,
) -> dict[str, Any]:
    request = dict(request_authorization.request_record)
    payload = coordinator.render_publication_handoff_closure_submit_payload(
        request_authorization
    )
    coordinator._reserve_closure_attempt(
        request_authorization,
        payload,
        reservation_root=root,
    )
    request_bytes = coordinator._canonical_json_bytes(request, pretty=True)
    coordinator._write_or_require_exact(
        root / "request-upload.json",
        coordinator._canonical_json_bytes(
            {
                "dbfs_uri": request["request_uri"],
                "exclusive_bytes_proven": True,
                "file_sha256": sha256(request_bytes).hexdigest(),
                "size_bytes": len(request_bytes),
            },
            pretty=True,
        ),
    )
    coordinator._write_or_require_exact(
        root / "submit-response.json",
        coordinator._canonical_json_bytes({"run_id": 12345}, pretty=True),
    )
    return payload


def test_renderer_uses_one_cpu_task_and_a_bounded_request_pointer() -> None:
    request = _request(large=True)
    request_bytes = coordinator._canonical_json_bytes(request, pretty=True)
    assert len(request_bytes) > 10_000

    request_authorization = _authorize_request(request)
    payload = coordinator.render_publication_handoff_closure_submit_payload(
        request_authorization
    )
    task = payload["tasks"][0]
    cluster = task["new_cluster"]
    parameters = task["spark_python_task"]["parameters"]
    assert task["max_retries"] == 0
    assert task["timeout_seconds"] == 12 * 60 * 60
    assert cluster["node_type_id"] == "c5d.4xlarge"
    assert cluster["driver_node_type_id"] == "c5d.4xlarge"
    assert cluster["num_workers"] == 0
    assert "spark.databricks.cluster.profile" in cluster["spark_conf"]
    assert "spark.master" in cluster["spark_conf"]
    for flag, field_name in (
        ("--runtime-lock-sha256", "runtime_lock_sha256"),
        ("--patched-vllm-wheel-sha256", "patched_vllm_wheel_sha256"),
        (
            "--patched-flashinfer-wheel-sha256",
            "patched_flashinfer_wheel_sha256",
        ),
        (
            "--runtime-closure-manifest-sha256",
            "runtime_closure_manifest_sha256",
        ),
    ):
        assert (
            parameters[parameters.index(flag) + 1] == request["coordinator"][field_name]
        )
    assert "--request-json-b64" not in parameters
    assert parameters[parameters.index("--request-uri") + 1] == request["request_uri"]
    assert (
        parameters[parameters.index("--request-file-sha256") + 1]
        == sha256(request_bytes).hexdigest()
    )
    assert (
        len(coordinator._canonical_json_bytes(parameters, pretty=False))
        <= coordinator.PUBLICATION_HANDOFF_CLOSURE_PARAMETER_BYTES_MAX
    )
    assert (
        "opener"
        not in inspect.signature(
            coordinator.collect_publication_handoff_closure
        ).parameters
    )


def test_closure_runner_inherits_native_v2_four_step_cpu_runtime() -> None:
    script = coordinator.PUBLICATION_HANDOFF_CLOSURE_RUNNER_SCRIPT
    assert "CACHET_HANDOFF_CLOSURE_LOCKED_RUNTIME" in script
    assert 'variable_name.upper().startswith(("PIP_", "_PIP_"))' in script
    assert '[sys.executable, "-m", "venv", "--copies", venv_dir]' in script
    assert "verify_gpu_qualification_v2_runtime_installation" in script
    install_markers = (
        '"--require-hashes", "--only-binary", ":all:"',
        '"vllm", patched_vllm_wheel',
        '"flashinfer-python"',
        '"cachet-kv", package_wheel',
    )
    positions = tuple(script.index(marker) for marker in install_markers)
    assert positions == tuple(sorted(positions))


def test_closure_request_rejects_v1_exact8_tamper_and_runner_conflation() -> None:
    request = _request()
    assert request["record_type"].endswith(".v2")
    assert request["schema_version"] == 2
    assert request["coordinator"]["record_type"].endswith(".v2")
    assert request["coordinator"]["schema_version"] == 2

    legacy = copy.deepcopy(request)
    legacy["record_type"] = "cachet.publication_handoff_closure_request.v1"
    legacy["schema_version"] = 1
    legacy["closed_record_sha256"] = coordinator._closed_record_sha256(legacy)
    with pytest.raises(ValueError, match="request envelope"):
        coordinator._validate_closure_request(legacy)

    for mutation in ("missing", "extra", "flashinfer"):
        tampered = copy.deepcopy(request)
        pins = tampered["qualified_artifact_pins"]
        if mutation == "missing":
            pins.pop("runtime_closure_manifest_sha256")
        elif mutation == "extra":
            pins["downstream_runner_sha256"] = _digest("extra")
        else:
            pins["patched_flashinfer_wheel_sha256"] = _digest("tampered-fi")
        tampered["closed_record_sha256"] = coordinator._closed_record_sha256(tampered)
        with pytest.raises(ValueError):
            coordinator._validate_closure_request(tampered)

    conflated = copy.deepcopy(request)
    conflated["qualified_artifact_pins"]["runner_sha256"] = conflated["coordinator"][
        "runner_sha256"
    ]
    conflated["closed_record_sha256"] = coordinator._closed_record_sha256(conflated)
    with pytest.raises(ValueError, match="runner must remain distinct"):
        coordinator._validate_closure_request(conflated)


def test_controller_cli_rejects_raw_request_as_launch_authority(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request = _request(large=True)
    request_path = tmp_path / "request.json"
    request_path.write_bytes(coordinator._canonical_json_bytes(request, pretty=True))
    assert coordinator.main(["render-submit", "--request", str(request_path)]) == 1
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["ok"] is False
    assert "raw request files are deliberately nonauthorizing" in rendered["error"]


def test_raw_or_resealed_package_request_cannot_authorize_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    authorization = _authorize_request(request)
    malicious = copy.deepcopy(request)
    malicious["coordinator"]["package_wheel_uri"] = (
        f"{VOLUME_ROOT}/inputs/malicious-cachet.whl"
    )
    malicious["coordinator"]["package_wheel_sha256"] = _digest("malicious-wheel")
    malicious["qualified_artifact_pins"]["package_wheel_sha256"] = _digest(
        "malicious-wheel"
    )
    malicious["closed_record_sha256"] = coordinator._closed_record_sha256(malicious)
    coordinator._validate_closure_request(malicious)

    with pytest.raises(
        TypeError, match="PublicationHandoffClosureRequestAuthorization"
    ):
        coordinator.render_publication_handoff_closure_submit_payload(malicious)
    with pytest.raises(
        TypeError, match="PublicationHandoffClosureRequestAuthorization"
    ):
        coordinator.reserve_and_submit_publication_handoff_closure(
            DatabricksWorkspaceConfig("https://workspace.example", "token"),
            malicious,
            reservation_root=tmp_path / "must-not-reserve",
        )
    with pytest.raises(TypeError, match="typed batch issuer"):
        coordinator.PublicationHandoffClosureRequestAuthorization(
            request=malicious,
            batch_evidence={},
            qualified_artifact_pins=_qualified_pins(_config()),
            qualification_authorization_binding={},
            controller_lease_root=_test_controller_lease_root("q8"),
            _issuer=object(),
        )

    exposed = dict(authorization.request_record)
    exposed["coordinator"]["package_wheel_sha256"] = _digest("mutated-view")
    payload = coordinator.render_publication_handoff_closure_submit_payload(
        authorization
    )
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert (
        parameters[parameters.index("--package-wheel-sha256") + 1]
        == request["coordinator"]["package_wheel_sha256"]
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "package_wheel_sha256",
        "cachet_source_tree_sha256",
    ],
)
def test_coordinator_config_must_match_qualified_producer_artifacts_before_render(
    field_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    input_bundle_sha256 = GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256
    qualification = _hardware_qualification(
        monkeypatch,
        config,
        input_bundle_sha256=input_bundle_sha256,
    )
    qualification_authorization = _qualification_launch_authorization(qualification)
    coordinator._require_matching_qualified_producer(
        config,
        qualification,
        qualification_authorization,
        expected_input_bundle_sha256=input_bundle_sha256,
        expected_qualification_closed_record_sha256=_digest("qualification"),
    )
    drifted = replace(config, **{field_name: _digest(f"drift-{field_name}")})
    with pytest.raises(ValueError, match="package/source pins differ"):
        coordinator._require_matching_qualified_producer(
            drifted,
            qualification,
            qualification_authorization,
            expected_input_bundle_sha256=input_bundle_sha256,
            expected_qualification_closed_record_sha256=_digest("qualification"),
        )


def test_request_authority_rejects_qualified_pin_substitution() -> None:
    request = _request()
    pins = replace(
        coordinator._q8.gpu_qualification_artifact_pins_v2_from_record(
            request["qualified_artifact_pins"]
        ),
        package_wheel_sha256=_digest("substituted-qualified-package"),
    )
    with pytest.raises(ValueError, match="qualification artifact pins drift"):
        coordinator.PublicationHandoffClosureRequestAuthorization(
            request=request,
            batch_evidence={
                "batch_authorization": {},
                "worker_evidence": request["worker_evidence"],
            },
            qualified_artifact_pins=pins,
            qualification_authorization_binding={},
            controller_lease_root=_test_controller_lease_root("q8"),
            _issuer=coordinator._REQUEST_AUTHORIZATION_ISSUER,
        )


def test_caller_resealed_qualification_and_arbitrary_pins_cannot_issue_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    input_bundle_sha256 = GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256
    qualification = _hardware_qualification(
        monkeypatch,
        config,
        input_bundle_sha256=input_bundle_sha256,
    )
    authorization = _qualification_launch_authorization(qualification)
    monkeypatch.setattr(
        coordinator._q8,
        "_validate_closed_plan_envelope",
        lambda _record: None,
    )
    monkeypatch.setattr(
        coordinator._bf16,
        "_validate_plan_envelope",
        lambda _record: None,
    )
    common = {
        "attempt_id": "closure-attempt",
        "coordinator_config": config,
        "plan_uri": f"{VOLUME_ROOT}/inputs/plan.json",
        "plan_file_sha256": _digest("plan-file"),
        "plan_record": {
            "closed_record_sha256": _digest("producer-plan"),
            "input_bundle_sha256": input_bundle_sha256,
        },
        "prepared_input_root_uri": f"{VOLUME_ROOT}/prepared",
        "durable_output_root_uri": f"{VOLUME_ROOT}/outputs",
        "execution_contract": {},
        "ledger_path": "/must-not-read-ledger.json",
        "attempt_ids_by_worker": {},
        "submission_authorization": object(),
        "expected_qualification_closed_record_sha256": _digest("qualification"),
    }
    builders = (
        (
            coordinator.build_q8_handoff_closure_request,
            {"worker_authorizations": {}},
        ),
        (
            coordinator.build_bf16_handoff_closure_request,
            {"worker_authorizations": {}},
        ),
    )
    arbitrary = PublicationLatencyGeneratorHardwareQualificationV2(
        evidence_record=dict(qualification.evidence_record),
        plan_record=dict(qualification.plan_record),
        expected_campaign_id=qualification.expected_campaign_id,
        expected_artifact_pins=replace(
            qualification.expected_artifact_pins,
            package_wheel_sha256=_digest("caller-arbitrary-package"),
        ),
        evidence_uri=qualification.evidence_uri,
        evidence_file_sha256=qualification.evidence_file_sha256,
        plan_uri=qualification.plan_uri,
        plan_file_sha256=qualification.plan_file_sha256,
    )
    for builder, stage_args in builders:
        with pytest.raises(ValueError, match="package/source pins differ"):
            builder(
                **common,
                **stage_args,
                hardware_qualification=arbitrary,
                qualification_launch_authorization=authorization,
            )
    resealed_evidence = replace(
        qualification,
        evidence_file_sha256=_digest("caller-resealed-evidence-file"),
    )
    for builder, stage_args in builders:
        with pytest.raises(ValueError, match="authorization evidence binding differs"):
            builder(
                **common,
                **stage_args,
                hardware_qualification=resealed_evidence,
                qualification_launch_authorization=authorization,
            )
        with pytest.raises(TypeError, match="GPUQualificationLaunchAuthorization"):
            builder(
                **common,
                **stage_args,
                hardware_qualification=qualification,
                qualification_launch_authorization=object(),
            )


def test_recovery_requires_the_exact_durable_batch_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    root = tmp_path / "reservation"
    original = _authorize_request(request, controller_lease_root=root)
    with pytest.raises(ValueError, match="typed batch evidence differs"):
        _authorize_request(request, batch_label="substituted-batch")
    substituted = _authorize_request(
        request,
        evidence_label="substituted-request-authority",
        controller_lease_root=root,
    )
    _write_collection_reservation(root, original)
    monkeypatch.setattr(
        coordinator,
        "get_databricks_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mismatched authority reached cloud collection")
        ),
    )
    with pytest.raises(ValueError, match="reservation source binding drift"):
        coordinator.collect_publication_handoff_closure(
            DatabricksWorkspaceConfig("https://workspace.example", "token"),
            reservation_root=root,
            request_authorization=substituted,
        )


def test_renderer_rejects_alternate_control_root_before_databricks() -> None:
    request = _request()
    huge_root = f"{VOLUME_ROOT}/control/{'x' * 9_500}"
    request["coordinator"]["request_root_uri"] = huge_root
    request["request_uri"] = f"{huge_root}/request.json"
    request["closed_record_sha256"] = coordinator._closed_record_sha256(request)
    with pytest.raises(ValueError, match="request root is not singleton-derived"):
        coordinator._validate_closure_request(request)


@pytest.mark.parametrize(
    "uri",
    [
        f"{VOLUME_ROOT}/control/%2e%2e/request.json",
        f"{VOLUME_ROOT}/control/../request.json",
    ],
)
def test_volume_pointer_rejects_encoded_or_plain_traversal(uri: str) -> None:
    with pytest.raises(ValueError, match="canonical|URL syntax"):
        coordinator._canonical_volume_file_uri(uri, "request_uri")


def test_reserve_stages_exact_request_before_post_and_replays_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    root = tmp_path / "reservation"
    request_authorization = _authorize_request(request, controller_lease_root=root)
    calls = {"put": 0, "post": 0}

    def upload(_workspace, uri, content, *, max_bytes):
        calls["put"] += 1
        assert (root / "reservation.json").is_file()
        assert uri == request["request_uri"]
        assert content == coordinator._canonical_json_bytes(request, pretty=True)
        assert len(content) <= max_bytes
        return {
            "created": calls["put"] == 1,
            "dbfs_uri": uri,
            "file_sha256": sha256(content).hexdigest(),
            "size_bytes": len(content),
        }

    def submit(_workspace, payload):
        calls["post"] += 1
        assert (root / "request-upload.json").is_file()
        assert payload == coordinator.render_publication_handoff_closure_submit_payload(
            request_authorization
        )
        return {"run_id": 12345}

    monkeypatch.setattr(
        coordinator, "upload_databricks_volume_file_bytes_exclusive", upload
    )
    monkeypatch.setattr(coordinator, "submit_databricks_run", submit)
    workspace = DatabricksWorkspaceConfig("https://workspace.example", "token")

    assert coordinator.reserve_and_submit_publication_handoff_closure(
        workspace, request_authorization, reservation_root=root
    ) == {"run_id": 12345}
    with pytest.raises(ValueError, match="reservation_root differs"):
        coordinator.reserve_and_submit_publication_handoff_closure(
            workspace,
            request_authorization,
            reservation_root=tmp_path / "alternate-reservation",
        )
    assert calls == {"put": 1, "post": 1}
    assert coordinator.reserve_and_submit_publication_handoff_closure(
        workspace, request_authorization, reservation_root=root
    ) == {"run_id": 12345}
    assert calls == {"put": 2, "post": 1}


@pytest.mark.parametrize("stage", ["q8", "bf16"])
def test_handoff_singleton_rejects_alternate_attempt_output_and_attestation_root(
    stage: str,
) -> None:
    request = _request(stage=stage)
    alternate_attempt = copy.deepcopy(request)
    alternate_attempt["attempt_id"] = f"{stage}-handoff-closure-alternate"
    alternate_attempt["closed_record_sha256"] = coordinator._closed_record_sha256(
        alternate_attempt
    )
    with pytest.raises(ValueError, match="attempt identity drift"):
        coordinator._validate_closure_request(alternate_attempt)

    with pytest.raises(ValueError, match="producer phase authority"):
        coordinator._require_submission_output_root(
            request["output_root_uri"],
            f"{request['output_root_uri']}-alternate",
            stage=stage,
        )

    directory = (
        coordinator._q8.PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY
        if stage == "q8"
        else coordinator._bf16.PUBLICATION_BF16_HANDOFF_ATTESTATION_DIRECTORY
    )
    canonical_path = (
        Path(coordinator.local_path(request["output_root_uri"]))
        / directory
        / "worker-00.json"
    )
    with pytest.raises(ValueError, match="attestation path differs"):
        coordinator._require_handoff_attestation_path(
            canonical_path.parent.parent / "alternate" / "worker-00.json",
            request["output_root_uri"],
            directory=directory,
            worker_index=0,
            stage=stage,
        )


def test_mac_collector_never_resolves_dbfs_and_issues_live_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_synthetic_manifests(monkeypatch)
    request = _request()
    root = tmp_path / "reservation"
    request_authorization = _authorize_request(request, controller_lease_root=root)
    result = _result(request)
    coordinator._validate_closure_result(result, request=request)
    payload = _write_collection_reservation(root, request_authorization)
    request_bytes = coordinator._canonical_json_bytes(request, pretty=True)
    result_bytes = coordinator._canonical_json_bytes(result, pretty=True)

    monkeypatch.setattr(
        coordinator,
        "_cluster_path",
        lambda _uri: (_ for _ in ()).throw(AssertionError("Mac touched /dbfs")),
    )
    monkeypatch.setattr(
        coordinator,
        "get_databricks_run",
        lambda _workspace, _run_id: _terminal(payload),
    )

    def download(_workspace, uri, *, max_bytes=16 * 1024 * 1024):
        content = request_bytes if uri == request["request_uri"] else result_bytes
        assert len(content) <= max_bytes
        return content

    monkeypatch.setattr(coordinator, "download_databricks_volume_file_bytes", download)
    workspace = DatabricksWorkspaceConfig("https://workspace.example", "token")
    authority = coordinator.collect_publication_handoff_closure(
        workspace,
        reservation_root=root,
        request_authorization=request_authorization,
    )

    assert authority.coordinator_run_id == "12345"
    assert authority.execution_record == result["execution"]["record"]
    assert len(authority.manifest_records) == 3
    assert (root / "runs-get.json").is_file()
    assert (root / "coordinator-result.json").read_bytes() == result_bytes
    assert (
        coordinator.require_q8_handoff_remote_closure_authorization(
            authority,
            expected_output_root_uri=request["output_root_uri"],
            expected_execution_file_sha256=result["execution"]["file_sha256"],
            expected_input_bundle_sha256=request["input_bundle_sha256"],
            expected_qualification_closed_record_sha256=request[
                "expected_qualification_closed_record_sha256"
            ],
        )
        is authority
    )


def test_collector_rejects_raw_python_task_substitution_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    root = tmp_path / "reservation"
    request_authorization = _authorize_request(request, controller_lease_root=root)
    payload = _write_collection_reservation(root, request_authorization)
    monkeypatch.setattr(
        coordinator,
        "download_databricks_volume_file_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("substituted Python task fetched an artifact")
        ),
    )
    workspace = DatabricksWorkspaceConfig("https://workspace.example", "token")
    for mutation in ("python_file", "ordered_parameters"):
        terminal = _terminal(payload)
        observed_python_task = terminal["tasks"][0]["spark_python_task"]
        if mutation == "python_file":
            observed_python_task["python_file"] = "dbfs:/attacker.py"
        else:
            parameters = observed_python_task["parameters"]
            parameters[0], parameters[1] = parameters[1], parameters[0]
        monkeypatch.setattr(
            coordinator,
            "get_databricks_run",
            lambda _workspace, _run_id, terminal=terminal: terminal,
        )
        with pytest.raises(ValueError, match="spark_python_task differs"):
            coordinator.collect_publication_handoff_closure(
                workspace,
                reservation_root=root,
                request_authorization=request_authorization,
            )


def test_collector_rejects_resealed_tampered_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_synthetic_manifests(monkeypatch)
    request = _request()
    root = tmp_path / "reservation"
    request_authorization = _authorize_request(request, controller_lease_root=root)
    result = _result(request)
    result["coordinator"]["close_function"] = "forged_close"
    result["closed_record_sha256"] = coordinator._closed_record_sha256(result)
    payload = _write_collection_reservation(root, request_authorization)
    request_bytes = coordinator._canonical_json_bytes(request, pretty=True)
    result_bytes = coordinator._canonical_json_bytes(result, pretty=True)
    monkeypatch.setattr(
        coordinator,
        "get_databricks_run",
        lambda _workspace, _run_id: _terminal(payload),
    )
    monkeypatch.setattr(
        coordinator,
        "download_databricks_volume_file_bytes",
        lambda _workspace, uri, **_kwargs: (
            request_bytes if uri == request["request_uri"] else result_bytes
        ),
    )
    workspace = DatabricksWorkspaceConfig("https://workspace.example", "token")
    with pytest.raises(ValueError, match="full mounted closure"):
        coordinator.collect_publication_handoff_closure(
            workspace,
            reservation_root=root,
            request_authorization=request_authorization,
        )


@pytest.mark.parametrize("stage", ["q8", "bf16"])
def test_compact_result_rejects_resealed_worker_batch_substitution(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_synthetic_manifests(monkeypatch)
    request = _request(stage=stage)
    result = _result(request)
    execution = result["execution"]["record"]
    reconciliation = execution["ledger_reconciliation"]
    reconciliation["attempts"][0]["attempt_id"] = "substituted-attempt"
    reconciliation["attempts_sha256"] = coordinator._canonical_sha256(
        reconciliation["attempts"]
    )
    execution["closed_record_sha256"] = (
        coordinator._q8._closed_record_sha256(execution)
        if stage == "q8"
        else coordinator._bf16._closed_record_sha256(execution)
    )
    result["execution"]["closed_record_sha256"] = execution["closed_record_sha256"]
    result["execution"]["file_sha256"] = sha256(
        coordinator._canonical_json_bytes(execution, pretty=True)
    ).hexdigest()
    result["closed_record_sha256"] = coordinator._closed_record_sha256(result)
    with pytest.raises(ValueError, match="ledger closure is incomplete"):
        coordinator._validate_closure_result(result, request=request)


def test_failed_run_cannot_fetch_or_authorize_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    root = tmp_path / "reservation"
    request_authorization = _authorize_request(request, controller_lease_root=root)
    payload = _write_collection_reservation(root, request_authorization)
    monkeypatch.setattr(
        coordinator,
        "get_databricks_run",
        lambda _workspace, _run_id: _terminal(payload, succeeded=False),
    )
    monkeypatch.setattr(
        coordinator,
        "download_databricks_volume_file_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("failed coordinator fetched an artifact")
        ),
    )
    workspace = DatabricksWorkspaceConfig("https://workspace.example", "token")
    with pytest.raises(ValueError, match="not one successful"):
        coordinator.collect_publication_handoff_closure(
            workspace,
            reservation_root=root,
            request_authorization=request_authorization,
        )


@pytest.mark.parametrize("attempt_number", [False, 1])
def test_nonzero_or_boolean_attempt_number_cannot_authorize_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempt_number: object,
) -> None:
    request = _request()
    root = tmp_path / "reservation"
    request_authorization = _authorize_request(request, controller_lease_root=root)
    payload = _write_collection_reservation(root, request_authorization)
    terminal = _terminal(payload)
    terminal["tasks"][0]["attempt_number"] = attempt_number
    monkeypatch.setattr(
        coordinator,
        "get_databricks_run",
        lambda _workspace, _run_id: terminal,
    )
    monkeypatch.setattr(
        coordinator,
        "download_databricks_volume_file_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("repaired coordinator fetched an artifact")
        ),
    )
    workspace = DatabricksWorkspaceConfig("https://workspace.example", "token")
    with pytest.raises(ValueError, match="attempt zero"):
        coordinator.collect_publication_handoff_closure(
            workspace,
            reservation_root=root,
            request_authorization=request_authorization,
        )


@pytest.mark.parametrize("task_run_id", [None, False, 0, "022345", 12345])
def test_noncanonical_or_parent_equal_task_run_id_cannot_authorize_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_run_id: object,
) -> None:
    request = _request()
    root = tmp_path / "reservation"
    request_authorization = _authorize_request(request, controller_lease_root=root)
    payload = _write_collection_reservation(root, request_authorization)
    terminal = _terminal(payload)
    terminal["tasks"][0]["run_id"] = task_run_id
    monkeypatch.setattr(
        coordinator,
        "get_databricks_run",
        lambda _workspace, _run_id: terminal,
    )
    monkeypatch.setattr(
        coordinator,
        "download_databricks_volume_file_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid child run ID fetched an artifact")
        ),
    )
    workspace = DatabricksWorkspaceConfig("https://workspace.example", "token")
    with pytest.raises(ValueError, match="run ID|distinct child"):
        coordinator.collect_publication_handoff_closure(
            workspace,
            reservation_root=root,
            request_authorization=request_authorization,
        )


def test_remote_authority_constructor_is_issuer_only() -> None:
    with pytest.raises(TypeError, match="collector issuer"):
        coordinator.PublicationHandoffRemoteClosureAuthorization(
            request={},
            result={},
            result_file_sha256="0" * 64,
            coordinator_run_id="12345",
            control_plane_status_sha256="1" * 64,
            workspace_host_sha256="2" * 64,
            user_name_sha256="3" * 64,
            _issuer=object(),
        )


@pytest.mark.parametrize("stage", ["q8", "bf16"])
def test_remote_ledger_snapshot_override_is_coordinator_issuer_only(
    stage: str, tmp_path: Path
) -> None:
    ledger = DatabricksClusterHourLedger(ledger_id="campaign-ledger")
    common = {
        "prepared_input_dir": tmp_path,
        "durable_output_root": tmp_path,
        "tokenizer": object(),
        "config": object(),
        "ledger_path": tmp_path / "missing-ledger.json",
        "attempt_ids_by_worker": {},
        "_ledger_snapshot": ledger,
        "_ledger_path_sha256": _digest("ledger-path"),
        "_remote_ledger_issuer": object(),
    }
    with pytest.raises(TypeError, match="coordinator issuer"):
        if stage == "q8":
            coordinator._q8.close_publication_latency_handoff_generation_from_workers(
                {}, attestations_by_worker={}, **common
            )
        else:
            coordinator._bf16.close_publication_bf16_handoff_generation_from_workers(
                {}, worker_authorizations={}, **common
            )


@pytest.mark.parametrize("stage", ["q8", "bf16"])
def test_mounted_coordinator_calls_existing_closer_with_remote_ledger_snapshot(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(stage=stage)
    root = tmp_path / stage
    root.mkdir()
    prepared = tmp_path / f"{stage}-prepared"
    prepared.mkdir()
    plan = {"closed_record_sha256": request["plan"]["closed_record_sha256"]}
    plan_bytes = coordinator._canonical_json_bytes(plan, pretty=True)
    plan_path = tmp_path / f"{stage}-plan.json"
    plan_path.write_bytes(plan_bytes)
    request["plan"]["file_sha256"] = sha256(plan_bytes).hexdigest()
    request["closed_record_sha256"] = coordinator._closed_record_sha256(request)
    coordinator._validate_closure_request(request)
    result_path = tmp_path / f"{stage}-result.json"
    paths = {
        request["output_root_uri"]: root,
        request["prepared_input_root_uri"]: prepared,
        request["plan"]["uri"]: plan_path,
        request["result_uri"]: result_path,
    }
    monkeypatch.setattr(coordinator, "_cluster_path", lambda uri: paths[uri])
    monkeypatch.setattr(coordinator, "_verify_source_closure", lambda _record: None)
    tokenizer = object()
    monkeypatch.setattr(coordinator, "load_main_latency_tokenizer", lambda: tokenizer)
    captured: dict[str, Any] = {}
    closed = SimpleNamespace(stage=stage)

    def close(_plan, **kwargs):
        captured.update(kwargs)
        return closed

    if stage == "q8":
        monkeypatch.setattr(
            coordinator._q8,
            "_execution_config_from_record",
            lambda _record: "q8-config",
        )
        monkeypatch.setattr(
            coordinator._q8,
            "close_publication_latency_handoff_generation_from_workers",
            close,
        )
    else:
        monkeypatch.setattr(
            coordinator._bf16,
            "_execution_config_from_record",
            lambda _record: "bf16-config",
        )
        monkeypatch.setattr(
            coordinator._bf16,
            "close_publication_bf16_handoff_generation_from_workers",
            close,
        )
    compact = {
        "closed_record_sha256": _digest(f"{stage}-compact"),
        "stage": stage,
    }

    def build_compact(_request, *, closed: Any, coordinator_run_id: str):
        assert closed is not None
        assert coordinator_run_id == "12345"
        return compact

    monkeypatch.setattr(coordinator, "_build_compact_result", build_compact)
    assert (
        coordinator.run_publication_handoff_closure_coordinator(
            request, coordinator_run_id="12345"
        )
        == compact
    )
    assert captured["prepared_input_dir"] == prepared
    assert captured["durable_output_root"] == root
    assert captured["tokenizer"] is tokenizer
    assert captured["ledger_path"] == Path(
        "/local_disk0/cachet-remote-ledger-snapshot.json"
    )
    assert captured["_ledger_snapshot"].ledger_id == "campaign-ledger"
    assert (
        captured["_ledger_path_sha256"]
        == request["ledger_lineage"]["ledger_path_sha256"]
    )
    if stage == "q8":
        assert (
            captured["_expected_producer_batch_prefix"].to_record()
            == request["ledger_lineage"]["producer_batch_prefix"]
        )
    assert set(captured["attempt_ids_by_worker"]) == set(range(16))
    evidence_name = (
        "attestations_by_worker" if stage == "q8" else "worker_authorizations"
    )
    assert set(captured[evidence_name]) == set(range(16))


@pytest.mark.parametrize("stage", ["q8", "bf16"])
@pytest.mark.parametrize("recovery", ["execution", "compact"])
def test_mounted_recovery_runs_full_post_close_replay(
    stage: str,
    recovery: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(stage=stage)
    root = tmp_path / stage
    root.mkdir()
    prepared = tmp_path / f"{stage}-prepared"
    prepared.mkdir()
    plan = {"closed_record_sha256": request["plan"]["closed_record_sha256"]}
    plan_bytes = coordinator._canonical_json_bytes(plan, pretty=True)
    plan_path = tmp_path / f"{stage}-plan.json"
    plan_path.write_bytes(plan_bytes)
    request["plan"]["file_sha256"] = sha256(plan_bytes).hexdigest()
    request["closed_record_sha256"] = coordinator._closed_record_sha256(request)
    result_path = tmp_path / f"{stage}-result.json"
    paths = {
        request["output_root_uri"]: root,
        request["prepared_input_root_uri"]: prepared,
        request["plan"]["uri"]: plan_path,
        request["result_uri"]: result_path,
    }
    monkeypatch.setattr(coordinator, "_cluster_path", lambda uri: paths[uri])
    monkeypatch.setattr(coordinator, "_verify_source_closure", lambda _record: None)
    monkeypatch.setattr(coordinator, "load_main_latency_tokenizer", lambda: object())
    compact = {
        "closed_record_sha256": "",
        "coordinator": {"run_id": "12345"},
        "stage": stage,
    }
    compact["closed_record_sha256"] = coordinator._closed_record_sha256(compact)
    closed = SimpleNamespace(stage=stage)
    captured: dict[str, Any] = {}

    def replay(_plan, **kwargs):
        captured.update(kwargs)
        return closed

    if stage == "q8":
        execution_path = (
            root / coordinator._q8.PUBLICATION_LATENCY_HANDOFF_EXECUTION_FILENAME
        )
        monkeypatch.setattr(
            coordinator._q8, "_execution_config_from_record", lambda _record: "config"
        )
        monkeypatch.setattr(
            coordinator._q8,
            "_replay_closed_publication_latency_handoff_generation",
            replay,
        )
        monkeypatch.setattr(
            coordinator._q8,
            "close_publication_latency_handoff_generation_from_workers",
            lambda *_args, **_kwargs: pytest.fail("recovery must not close again"),
        )
    else:
        execution_path = (
            root / coordinator._bf16.PUBLICATION_BF16_HANDOFF_EXECUTION_FILENAME
        )
        monkeypatch.setattr(
            coordinator._bf16,
            "_execution_config_from_record",
            lambda _record: "config",
        )
        monkeypatch.setattr(
            coordinator._bf16,
            "_replay_closed_publication_bf16_handoff_generation",
            replay,
        )
        monkeypatch.setattr(
            coordinator._bf16,
            "close_publication_bf16_handoff_generation_from_workers",
            lambda *_args, **_kwargs: pytest.fail("recovery must not close again"),
        )
    if recovery == "execution":
        execution_path.write_bytes(b"closed execution marker")
    else:
        result_path.write_bytes(coordinator._canonical_json_bytes(compact, pretty=True))
        monkeypatch.setattr(
            coordinator, "_validate_closure_result", lambda *_args, **_kwargs: None
        )
    monkeypatch.setattr(
        coordinator,
        "_build_compact_result",
        lambda _request, *, closed, coordinator_run_id: compact,
    )

    assert (
        coordinator.run_publication_handoff_closure_coordinator(
            request, coordinator_run_id="12345"
        )
        == compact
    )
    assert captured["ledger_snapshot"].ledger_id == "campaign-ledger"
    assert (
        captured["ledger_path_sha256"]
        == request["ledger_lineage"]["ledger_path_sha256"]
    )
    assert (
        captured["expected_producer_batch_prefix"].to_record()
        == request["ledger_lineage"]["producer_batch_prefix"]
    )
    issuer = (
        coordinator._q8._POST_CLOSE_REPLAY_ISSUER
        if stage == "q8"
        else coordinator._bf16._POST_CLOSE_REPLAY_ISSUER
    )
    assert captured["_issuer"] is issuer


@pytest.mark.parametrize("stage", ["q8", "bf16"])
def test_existing_resealed_compact_result_must_equal_fresh_replay(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(stage=stage)
    root = tmp_path / stage
    root.mkdir()
    prepared = tmp_path / f"{stage}-prepared"
    prepared.mkdir()
    plan = {"closed_record_sha256": request["plan"]["closed_record_sha256"]}
    plan_bytes = coordinator._canonical_json_bytes(plan, pretty=True)
    plan_path = tmp_path / f"{stage}-plan.json"
    plan_path.write_bytes(plan_bytes)
    request["plan"]["file_sha256"] = sha256(plan_bytes).hexdigest()
    request["closed_record_sha256"] = coordinator._closed_record_sha256(request)
    result_path = tmp_path / f"{stage}-result.json"
    paths = {
        request["output_root_uri"]: root,
        request["prepared_input_root_uri"]: prepared,
        request["plan"]["uri"]: plan_path,
        request["result_uri"]: result_path,
    }
    monkeypatch.setattr(coordinator, "_cluster_path", lambda uri: paths[uri])
    monkeypatch.setattr(coordinator, "_verify_source_closure", lambda _record: None)
    monkeypatch.setattr(coordinator, "load_main_latency_tokenizer", lambda: object())
    expected = {
        "closed_record_sha256": "",
        "coordinator": {"run_id": "12345"},
        "stage": stage,
    }
    expected["closed_record_sha256"] = coordinator._closed_record_sha256(expected)
    resealed = copy.deepcopy(expected)
    resealed["unreviewed_resealed_field"] = True
    resealed["closed_record_sha256"] = coordinator._closed_record_sha256(resealed)
    result_path.write_bytes(coordinator._canonical_json_bytes(resealed, pretty=True))
    monkeypatch.setattr(
        coordinator, "_validate_closure_result", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        coordinator,
        "_build_compact_result",
        lambda _request, *, closed, coordinator_run_id: expected,
    )
    if stage == "q8":
        monkeypatch.setattr(
            coordinator._q8, "_execution_config_from_record", lambda _record: object()
        )
        monkeypatch.setattr(
            coordinator._q8,
            "_replay_closed_publication_latency_handoff_generation",
            lambda *_args, **_kwargs: SimpleNamespace(stage=stage),
        )
    else:
        monkeypatch.setattr(
            coordinator._bf16, "_execution_config_from_record", lambda _record: object()
        )
        monkeypatch.setattr(
            coordinator._bf16,
            "_replay_closed_publication_bf16_handoff_generation",
            lambda *_args, **_kwargs: SimpleNamespace(stage=stage),
        )
    with pytest.raises(ValueError, match="differs from full post-close replay"):
        coordinator.run_publication_handoff_closure_coordinator(
            request, coordinator_run_id="12345"
        )


@pytest.mark.parametrize("stage", ["q8", "bf16"])
def test_post_close_replay_helpers_are_coordinator_issuer_only(
    stage: str, tmp_path: Path
) -> None:
    common = {
        "prepared_input_dir": tmp_path,
        "durable_output_root": tmp_path,
        "tokenizer": object(),
        "config": object(),
        "ledger_snapshot": DatabricksClusterHourLedger(ledger_id="ledger"),
        "ledger_path_sha256": _digest("ledger-path"),
        "expected_producer_batch_prefix": databricks_ledger_prefix(
            DatabricksClusterHourLedger(ledger_id="ledger")
        ),
        "attempt_ids_by_worker": {},
        "_issuer": object(),
    }
    with pytest.raises(TypeError, match="coordinator issuer"):
        if stage == "q8":
            coordinator._q8._replay_closed_publication_latency_handoff_generation(
                {}, attestations_by_worker={}, **common
            )
        else:
            coordinator._bf16._replay_closed_publication_bf16_handoff_generation(
                {}, worker_authorizations={}, **common
            )


def test_mirror_closure_path_requires_workspace_and_remote_byte_equality(
    tmp_path, monkeypatch
):
    mirror = tmp_path / "mirror"
    mirror.mkdir(mode=0o700)
    directory = (
        coordinator._q8.PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY
    )
    record = {"attempt": {"worker_index": 0}, "closed_record_sha256": "a" * 64}
    content = coordinator._canonical_json_bytes(record, pretty=True)
    local = mirror / directory / "worker-00.json"
    coordinator._q8._write_q8_exact_mirror_bytes(
        mirror, f"{directory}/worker-00.json", content
    )
    canonical = (
        Path("/dbfs/Volumes/catalog/schema/volume/output")
        / directory
        / "worker-00.json"
    )
    monkeypatch.setattr(
        coordinator._q8,
        "_validate_publication_latency_handoff_databricks_attestation_record",
        lambda _record: None,
    )
    with pytest.raises(TypeError, match="workspace"):
        coordinator._require_handoff_attestation_path(
            canonical,
            "dbfs:/Volumes/catalog/schema/volume/output",
            directory=directory,
            worker_index=0,
            stage="q8",
            local_mirror_root=mirror,
            expected_file_sha256=sha256(content).hexdigest(),
            expected_closed_record_sha256="a" * 64,
        )
    workspace = coordinator.DatabricksWorkspaceConfig("https://example.com", "token")
    monkeypatch.setattr(
        coordinator, "download_databricks_volume_file_bytes", lambda *_a, **_k: b"bad"
    )
    with pytest.raises(ValueError, match="remote attestation bytes"):
        coordinator._require_handoff_attestation_path(
            canonical,
            "dbfs:/Volumes/catalog/schema/volume/output",
            directory=directory,
            worker_index=0,
            stage="q8",
            local_mirror_root=mirror,
            expected_file_sha256=sha256(content).hexdigest(),
            expected_closed_record_sha256="a" * 64,
            collection_workspace=workspace,
        )
    assert local.is_file()


@pytest.mark.parametrize(
    ("module", "root_helper", "writer", "reader"),
    [
        (
            coordinator._q8,
            "_require_q8_local_evidence_mirror_root",
            "_write_q8_exact_mirror_bytes",
            "_read_q8_stable_regular_file",
        ),
        (
            coordinator._bf16,
            "_require_bf16_local_evidence_mirror_root",
            "_write_bf16_exact_mirror_bytes",
            "_read_bf16_stable_regular_file",
        ),
    ],
)
def test_mirror_io_rejects_fifo_hardlink_symlink_and_permissive_descendant(
    tmp_path: Path,
    module: Any,
    root_helper: str,
    writer: str,
    reader: str,
) -> None:
    root = getattr(module, root_helper)(
        tmp_path / module.__name__.rsplit(".", 1)[-1], create=True
    )
    fifo = root / "fifo"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(ValueError, match="single-link regular file"):
        getattr(module, reader)(fifo, mirror_root=root)

    regular = root / "regular"
    regular.write_bytes(b"same")
    regular.chmod(0o600)
    os.link(regular, root / "second-link")
    with pytest.raises(ValueError, match="single-link regular file"):
        getattr(module, reader)(regular, mirror_root=root)

    target = root / "target"
    target.mkdir(mode=0o700)
    (root / "linked-dir").symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        getattr(module, writer)(root, "linked-dir/value.json", b"value")

    permissive = root / "permissive"
    permissive.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="current-UID mode 0700"):
        getattr(module, writer)(root, "permissive/value.json", b"value")

    root.chmod(0o755)
    with pytest.raises(ValueError, match="mirror root must be current-UID mode 0700"):
        getattr(module, writer)(root, "value.json", b"value")


@pytest.mark.parametrize(
    ("module", "root_helper", "writer"),
    [
        (
            coordinator._q8,
            "_require_q8_local_evidence_mirror_root",
            "_write_q8_exact_mirror_bytes",
        ),
        (
            coordinator._bf16,
            "_require_bf16_local_evidence_mirror_root",
            "_write_bf16_exact_mirror_bytes",
        ),
    ],
)
def test_mirror_writer_rejects_staging_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    root_helper: str,
    writer: str,
) -> None:
    root = getattr(module, root_helper)(
        tmp_path / module.__name__.rsplit(".", 1)[-1], create=True
    )
    original_link = os.link

    def substitute(source: str, destination: str, **kwargs: Any) -> None:
        source_dir_fd = kwargs["src_dir_fd"]
        os.unlink(source, dir_fd=source_dir_fd)
        replacement = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=source_dir_fd,
        )
        try:
            os.write(replacement, b"evil!")
            os.fsync(replacement)
        finally:
            os.close(replacement)
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(module.os, "link", substitute)
    with pytest.raises(RuntimeError, match="staging identity changed"):
        getattr(module, writer)(root, "nested/value.json", b"right")
    assert not (root / "nested" / "value.json").exists()


@pytest.mark.parametrize(
    ("module", "root_helper", "writer"),
    [
        (
            coordinator._q8,
            "_require_q8_local_evidence_mirror_root",
            "_write_q8_exact_mirror_bytes",
        ),
        (
            coordinator._bf16,
            "_require_bf16_local_evidence_mirror_root",
            "_write_bf16_exact_mirror_bytes",
        ),
    ],
)
def test_mirror_writer_recovers_only_one_exact_crash_link(
    tmp_path: Path,
    module: Any,
    root_helper: str,
    writer: str,
) -> None:
    root = getattr(module, root_helper)(
        tmp_path / f"crash-{module.__name__.rsplit('.', 1)[-1]}", create=True
    )
    nested = root / "nested"
    nested.mkdir(mode=0o700)
    final = nested / "value.json"
    final.write_bytes(b"exact")
    final.chmod(0o600)
    stale = nested / ".value.json.tmp-0123456789abcdef0123456789abcdef"
    os.link(final, stale)
    assert final.stat().st_nlink == 2
    assert getattr(module, writer)(root, "nested/value.json", b"exact") == final
    assert final.stat().st_nlink == 1
    assert not stale.exists()

    ambiguous = nested / ".value.json.tmp-fedcba9876543210fedcba9876543210"
    second = nested / ".value.json.tmp-11111111111111111111111111111111"
    os.link(final, ambiguous)
    os.link(final, second)
    with pytest.raises(ValueError, match="recovery is ambiguous"):
        getattr(module, writer)(root, "nested/value.json", b"exact")


@pytest.mark.parametrize(
    ("module", "root_helper", "reader_at"),
    [
        (
            coordinator._q8,
            "_require_q8_local_evidence_mirror_root",
            "_read_q8_stable_regular_file_at",
        ),
        (
            coordinator._bf16,
            "_require_bf16_local_evidence_mirror_root",
            "_read_bf16_stable_regular_file_at",
        ),
    ],
)
def test_stable_reader_rejects_same_size_metadata_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    root_helper: str,
    reader_at: str,
) -> None:
    root = getattr(module, root_helper)(
        tmp_path / f"race-{module.__name__.rsplit('.', 1)[-1]}", create=True
    )
    leaf = root / "value"
    leaf.write_bytes(b"same")
    leaf.chmod(0o600)
    parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    original_fstat = os.fstat
    calls = 0

    def raced_fstat(descriptor: int) -> Any:
        nonlocal calls
        observed = original_fstat(descriptor)
        calls += 1
        if calls == 2:
            values = {
                name: getattr(observed, name)
                for name in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_uid",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            }
            values["st_ctime_ns"] += 1
            return SimpleNamespace(**values)
        return observed

    monkeypatch.setattr(module.os, "fstat", raced_fstat)
    try:
        with pytest.raises(ValueError, match="changed while being read"):
            getattr(module, reader_at)(parent_fd, "value", require_mode_0600=True)
    finally:
        os.close(parent_fd)


def test_verified_publication_input_reader_rejects_special_links_and_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"publication input"
    digest = sha256(content).hexdigest()
    regular = tmp_path / "regular"
    regular.write_bytes(content)
    assert coordinator._read_verified_file_bytes(regular, digest, "input") == content

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="single-link"):
        coordinator._read_verified_file_bytes(fifo, digest, "input")

    linked = tmp_path / "linked"
    os.link(regular, linked)
    with pytest.raises(ValueError, match="single-link"):
        coordinator._read_verified_file_bytes(regular, digest, "input")
    linked.unlink()

    directory = tmp_path / "directory"
    directory.mkdir()
    nested = directory / "nested"
    nested.write_bytes(content)
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(OSError):
        coordinator._read_verified_file_bytes(alias / "nested", digest, "input")

    original_fstat = os.fstat
    calls = 0

    def raced_fstat(descriptor: int) -> Any:
        nonlocal calls
        observed = original_fstat(descriptor)
        calls += 1
        if calls == 2:
            values = {
                name: getattr(observed, name)
                for name in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_uid",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            }
            values["st_mtime_ns"] += 1
            return SimpleNamespace(**values)
        return observed

    monkeypatch.setattr(coordinator.os, "fstat", raced_fstat)
    with pytest.raises(ValueError, match="changed while being read"):
        coordinator._read_verified_file_bytes(regular, digest, "input")


def test_workspace_identity_is_ephemeral_and_subclass_authority_is_rejected() -> None:
    request = _request()

    def assert_no_workspace_identity(value: Any) -> None:
        if isinstance(value, dict):
            assert "workspace_host_sha256" not in value
            assert "user_name_sha256" not in value
            for nested in value.values():
                assert_no_workspace_identity(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_no_workspace_identity(nested)

    assert_no_workspace_identity(request)

    class ForgedAuthorization(
        coordinator.PublicationHandoffClosureRequestAuthorization
    ):
        pass

    forged = object.__new__(ForgedAuthorization)
    with pytest.raises(
        TypeError, match="PublicationHandoffClosureRequestAuthorization"
    ):
        coordinator.require_publication_handoff_closure_request_authorization(forged)

    class ForgedQ8Submission(
        coordinator._q8.PublicationLatencyHandoffSubmissionAuthorization
    ):
        pass

    with pytest.raises(
        TypeError, match="PublicationLatencyHandoffSubmissionAuthorization"
    ):
        coordinator._q8.require_publication_latency_handoff_submission_authorization(
            object.__new__(ForgedQ8Submission)
        )

    class ForgedBF16Submission(
        coordinator._bf16.PublicationBF16HandoffSubmissionAuthorization
    ):
        pass

    with pytest.raises(
        TypeError, match="PublicationBF16HandoffSubmissionAuthorization"
    ):
        coordinator._bf16.require_publication_bf16_handoff_submission_authorization(
            object.__new__(ForgedBF16Submission)
        )


def test_closure_reserve_and_collect_reject_workspace_and_principal_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    root = tmp_path / "reservation"
    authorization = _authorize_request(request, controller_lease_root=root)
    other_workspace = DatabricksWorkspaceConfig("https://other.example", "token")
    with pytest.raises(ValueError, match="workspace/principal authority drift"):
        coordinator.reserve_and_submit_publication_handoff_closure(
            other_workspace,
            authorization,
            reservation_root=root,
        )
    assert not root.exists()

    _write_collection_reservation(root, authorization)
    with pytest.raises(ValueError, match="workspace/principal authority drift"):
        coordinator.collect_publication_handoff_closure(
            other_workspace,
            reservation_root=root,
            request_authorization=authorization,
        )

    monkeypatch.setattr(
        coordinator,
        "require_databricks_current_user_name",
        lambda _workspace, *, expected_user_name: {
            "workspace_host_sha256": _digest(TEST_WORKSPACE_HOST),
            "user_name_sha256": _digest("other-principal@example.com"),
        },
    )
    with pytest.raises(ValueError, match="workspace/principal authority drift"):
        coordinator.collect_publication_handoff_closure(
            DatabricksWorkspaceConfig(TEST_WORKSPACE_HOST, "refreshed-token"),
            reservation_root=root,
            request_authorization=authorization,
        )
