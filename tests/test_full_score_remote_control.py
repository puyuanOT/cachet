import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import document_kv_cache.full_score_remote_control as remote
from document_kv_cache.databricks_resource_ledger import DatabricksLedgerPrefix
from document_kv_cache.databricks_runs import DatabricksWorkspaceConfig
from document_kv_cache.full_score_execution import FullScorePhaseAuthorization


def _digest(label):
    return remote.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _bind_remote_current_user(monkeypatch):
    def bind_current_user(workspace, *, expected_user_name, opener=None):
        return {
            "authenticated": True,
            "user_name_sha256": _digest(expected_user_name),
            "workspace_host_sha256": _digest(workspace.normalized_host),
        }

    monkeypatch.setattr(
        remote,
        "require_databricks_current_user_name",
        bind_current_user,
    )


def _close(record):
    record["closed_record_sha256"] = remote._closed_record_sha256(record)
    return record


def _remote_control_uri(action, filename, *, wave_index=0):
    return (
        "dbfs:/Volumes/catalog/schema/volume/durable/control/"
        f"full-score-remote/{action}/wave-{wave_index:03d}/{filename}"
    )


def _remote_runner_uri():
    return (
        "dbfs:/Volumes/catalog/schema/volume/durable/control/"
        "full-score-remote/runtime/runner-"
        f"{remote.FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256}.py"
    )


def _request_uri(case):
    return _remote_control_uri(
        case.request["action"],
        "request.json",
        wave_index=case.request["wave_index"],
    )


def _closed_record(label, **extra):
    return _close(
        {
            "closed_record_sha256": "",
            "label": label,
            **extra,
        }
    )


def _phase_authorization(
    execution_sha,
    *,
    wave_index=0,
    phase="producer",
    terminal_record_sha256=None,
    phase_lease_root=None,
    workspace_host="https://dbc.example",
    user_name="researcher@example.com",
):
    predecessor = DatabricksLedgerPrefix(
        ledger_id="publication-full-score",
        cap_cluster_hours=1024.0,
        reservation_count=7,
        submission_receipt_count=7,
        terminal_actual_count=7,
        prefix_sha256=_digest("predecessor"),
    )
    terminal = DatabricksLedgerPrefix(
        ledger_id="publication-full-score",
        cap_cluster_hours=1024.0,
        reservation_count=8,
        submission_receipt_count=8,
        terminal_actual_count=8,
        prefix_sha256=_digest("terminal"),
    )
    return FullScorePhaseAuthorization(
        execution_plan_sha256=execution_sha,
        wave_index=wave_index,
        phase=phase,
        ledger_path_sha256=_digest("ledger-path"),
        predecessor_prefix=predecessor,
        ledger_prefix=terminal,
        phase_lease_root=(
            Path("/private/tmp/cachet-full-score-phase-leases")
            if phase_lease_root is None
            else phase_lease_root
        ),
        terminal_record_sha256=(
            _digest("phase-terminal")
            if terminal_record_sha256 is None
            else terminal_record_sha256
        ),
        causal_closure_sha256=_digest("phase-causal-closure"),
        workspace_host_sha256=_digest(workspace_host),
        user_name_sha256=_digest(user_name),
        _issuer=remote.full_score._FULL_SCORE_PHASE_AUTHORIZATION_ISSUER,
    )


def _request_case(*, action="producer_ready", wave_index=0, phase_lease_root=None):
    shard_ids = [f"shard-{index:03d}" for index in range(16)]
    execution = _closed_record(
        "execution",
        waves=[
            {
                "shard_ids": shard_ids,
            }
        ],
    )
    inventory = _closed_record("inventory")
    shard_plan = _closed_record("shard-plan")
    role = "producer" if action == "producer_ready" else "consumer"
    package_wheel_sha256 = _digest("package-wheel")
    patched_wheel_sha256 = remote.full_score.VLLM_PATCHED_WHEEL_SHA256
    patched_flashinfer_sha256 = remote.full_score.FLASHINFER_PATCHED_WHEEL_SHA256
    runtime_closure_sha256 = remote.full_score.RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
    runner_sha256 = remote.full_score.FULL_SCORE_RUNNER_SHA256
    qualification_runner_sha256 = _digest("qualification-runner")
    runtime_lock_sha256 = remote.full_score.VLLM_RUNTIME_BASE_LOCK_SHA256
    locked_runtime_sha256 = remote.full_score._locked_runtime_identity_sha256(
        runner_sha256=runner_sha256,
        package_wheel_sha256=package_wheel_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
        patched_vllm_wheel_sha256=patched_wheel_sha256,
        patched_flashinfer_wheel_sha256=patched_flashinfer_sha256,
        runtime_closure_manifest_sha256=runtime_closure_sha256,
    )
    payloads = [
        (
            f"dbfs:/Volumes/catalog/schema/volume/workers/{role}-{index:02d}.json",
            _closed_record(
                f"worker-{index}",
                bootstrap_artifacts={
                    "locked_runtime_identity_sha256": locked_runtime_sha256,
                    "package_wheel_sha256": package_wheel_sha256,
                    "package_wheel_uri": (
                        "dbfs:/Volumes/catalog/schema/volume/runtime/cachet.whl"
                    ),
                    "patched_vllm_wheel_sha256": patched_wheel_sha256,
                    "patched_vllm_wheel_uri": (
                        "dbfs:/Volumes/catalog/schema/volume/runtime/vllm.whl"
                    ),
                    "patched_flashinfer_wheel_sha256": patched_flashinfer_sha256,
                    "patched_flashinfer_wheel_uri": (
                        "dbfs:/Volumes/catalog/schema/volume/runtime/flashinfer.whl"
                    ),
                    "runner_sha256": runner_sha256,
                    "runtime_closure_manifest_sha256": runtime_closure_sha256,
                    "runtime_closure_manifest_uri": (
                        "dbfs:/Volumes/catalog/schema/volume/runtime/closure.json"
                    ),
                    "runtime_lock_sha256": runtime_lock_sha256,
                },
                durable_output_root="dbfs:/Volumes/catalog/schema/volume/durable",
                execution_plan={
                    "uri": "dbfs:/Volumes/catalog/schema/volume/inputs/execution.json"
                },
                gpu_qualification={
                    "artifact_pins": {
                        "package_wheel_sha256": package_wheel_sha256,
                        "patched_vllm_wheel_sha256": patched_wheel_sha256,
                        "patched_flashinfer_wheel_sha256": (patched_flashinfer_sha256),
                        "runtime_closure_manifest_sha256": runtime_closure_sha256,
                        "runner_sha256": qualification_runner_sha256,
                        "runtime_lock_sha256": runtime_lock_sha256,
                    }
                },
                inventory={
                    "uri": "dbfs:/Volumes/catalog/schema/volume/inputs/inventory.json"
                },
                role=role,
                runtime={
                    "patched_vllm_wheel_sha256": patched_wheel_sha256,
                    "patched_flashinfer_wheel_sha256": patched_flashinfer_sha256,
                    "runtime_closure_manifest_sha256": runtime_closure_sha256,
                    "runtime_lock_sha256": runtime_lock_sha256,
                },
                shards=[{"shard_id": shard_id}],
                shard_plan={
                    "uri": "dbfs:/Volumes/catalog/schema/volume/inputs/shards.json"
                },
                wave_index=wave_index,
                worker_index=index,
            ),
        )
        for index, shard_id in enumerate(shard_ids)
    ]
    phase_terminal = _closed_record(
        "phase-terminal",
        authorization_scope=remote.full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE,
        execution_plan_sha256=execution["closed_record_sha256"],
        phase=role,
        record_type=remote.full_score.FULL_SCORE_PHASE_TERMINAL_RECORD_TYPE,
        schema_version=remote.full_score.FULL_SCORE_PHASE_TERMINAL_SCHEMA_VERSION,
        task_billing=[
            {
                "worker_payload_file_sha256": remote.sha256(
                    remote._pretty_json_bytes(record)
                ).hexdigest(),
                "worker_payload_record_sha256": record["closed_record_sha256"],
                "worker_payload_uri": uri,
            }
            for uri, record in payloads
        ],
        wave_index=wave_index,
    )
    phase = _phase_authorization(
        execution["closed_record_sha256"],
        wave_index=wave_index,
        phase=role,
        terminal_record_sha256=phase_terminal["closed_record_sha256"],
        phase_lease_root=phase_lease_root,
    )
    coordinator_config = _coordinator_config()
    request = remote.build_full_score_remote_coordinator_request(
        action=action,
        wave_index=wave_index,
        inventory_uri="dbfs:/Volumes/catalog/schema/volume/inputs/inventory.json",
        inventory_record=inventory,
        shard_plan_uri="dbfs:/Volumes/catalog/schema/volume/inputs/shards.json",
        shard_plan=shard_plan,
        execution_plan_uri="dbfs:/Volumes/catalog/schema/volume/inputs/execution.json",
        execution_plan=execution,
        worker_payloads=payloads,
        durable_output_root="dbfs:/Volumes/catalog/schema/volume/durable",
        result_uri=(
            _remote_control_uri(
                action,
                "result.json",
                wave_index=wave_index,
            )
        ),
        attestation_uri=(
            _remote_control_uri(
                action,
                "attestation.json",
                wave_index=wave_index,
            )
        ),
        coordinator_config=coordinator_config,
        phase_authorization=phase,
        phase_terminal_record=phase_terminal,
    )
    return SimpleNamespace(
        execution=execution,
        coordinator_config=coordinator_config,
        inventory=inventory,
        payloads=payloads,
        phase=phase,
        phase_terminal=phase_terminal,
        request=request,
        shard_ids=shard_ids,
        shard_plan=shard_plan,
    )


def _coordinator_config():
    return remote.FullScoreRemoteCoordinatorJobConfig(
        runner_python_file=_remote_runner_uri(),
        package_wheel_uri="dbfs:/Volumes/catalog/schema/volume/runtime/cachet.whl",
        package_wheel_sha256=_digest("package-wheel"),
        single_user_name="researcher@example.com",
    )


def _rebuild_request(
    case,
    *,
    worker_payloads=None,
    durable_output_root=None,
    result_uri=None,
    attestation_uri=None,
    runner_python_file=None,
    package_wheel_uri="dbfs:/Volumes/catalog/schema/volume/runtime/cachet.whl",
    package_wheel_sha256=None,
    phase_authorization=None,
):
    coordinator_config = case.coordinator_config
    if runner_python_file is not None:
        coordinator_config = replace(
            coordinator_config,
            runner_python_file=runner_python_file,
        )
    if package_wheel_uri != coordinator_config.package_wheel_uri:
        coordinator_config = replace(
            coordinator_config,
            package_wheel_uri=package_wheel_uri,
        )
    if package_wheel_sha256 is not None:
        coordinator_config = replace(
            coordinator_config,
            package_wheel_sha256=package_wheel_sha256,
        )
    return remote.build_full_score_remote_coordinator_request(
        action=case.request["action"],
        wave_index=case.request["wave_index"],
        inventory_uri=case.request["sources"]["inventory"]["uri"],
        inventory_record=case.inventory,
        shard_plan_uri=case.request["sources"]["shard_plan"]["uri"],
        shard_plan=case.shard_plan,
        execution_plan_uri=case.request["sources"]["execution_plan"]["uri"],
        execution_plan=case.execution,
        worker_payloads=case.payloads if worker_payloads is None else worker_payloads,
        durable_output_root=(
            case.request["durable_output_root"]
            if durable_output_root is None
            else durable_output_root
        ),
        result_uri=case.request["result_uri"] if result_uri is None else result_uri,
        attestation_uri=(
            case.request["attestation_uri"]
            if attestation_uri is None
            else attestation_uri
        ),
        coordinator_config=coordinator_config,
        phase_authorization=(
            case.phase if phase_authorization is None else phase_authorization
        ),
        phase_terminal_record=case.phase_terminal,
    )


def _result_and_attestation(case):
    action = case.request["action"]
    if action == "producer_ready":
        result = {
            "authorization_scope": "publication",
            "closed_record_sha256": "",
            "consumer_phase_authorized": True,
            "execution_plan_sha256": case.request["execution_plan_sha256"],
            "ready_shards": [],
            "record_type": remote.full_score.FULL_SCORE_PRODUCER_PHASE_COMPLETION_RECORD_TYPE,
            "schema_version": 1,
            "shard_ids": case.shard_ids,
            "total_ready_bytes": 1,
            "wave_index": 0,
        }
        tree_entries = case.shard_ids
        closures = [
            {
                "file_sha256": _digest(f"file-{shard_id}"),
                "files_sha256": _digest(f"files-{shard_id}"),
                "record_sha256": _digest(f"record-{shard_id}"),
                "shard_id": shard_id,
            }
            for shard_id in case.shard_ids
        ]
    else:
        consumer_artifacts = {}
        closures = []
        deletion_record_sha256 = []
        for shard_id in case.shard_ids:
            evidence = _close(
                {
                    "authorization_scope": "publication",
                    "closed_record_sha256": "",
                    "durable_evidence_committed": True,
                    "execution_plan_sha256": case.request["execution_plan_sha256"],
                    "paired_examples": [],
                    "ready_shard_sha256": _digest(f"ready-{shard_id}"),
                    "record_type": remote.full_score.FULL_SCORE_SHARD_EVIDENCE_RECORD_TYPE,
                    "schema_version": remote.full_score.FULL_SCORE_SHARD_EVIDENCE_SCHEMA_VERSION,
                    "shard_id": shard_id,
                    "wave_index": case.request["wave_index"],
                }
            )
            deletion = _close(
                {
                    "authorization_scope": "publication",
                    "closed_record_sha256": "",
                    "evidence_closed_record_sha256": evidence["closed_record_sha256"],
                    "execution_plan_sha256": case.request["execution_plan_sha256"],
                    "lifecycle": [
                        "verify_ready_shard",
                        "baseline_inference",
                        "vanilla_inference",
                        "validate_paired_outputs",
                        "commit_durable_evidence",
                        "delete_ephemeral_q8_kv",
                    ],
                    "ready_shard_sha256": evidence["ready_shard_sha256"],
                    "record_type": remote.full_score.FULL_SCORE_DELETION_ATTESTATION_RECORD_TYPE,
                    "schema_version": remote.full_score.FULL_SCORE_DELETION_ATTESTATION_SCHEMA_VERSION,
                    "shard_id": shard_id,
                    "wave_index": case.request["wave_index"],
                }
            )
            evidence_bytes = remote._pretty_json_bytes(evidence)
            deletion_bytes = remote._pretty_json_bytes(deletion)
            evidence_uri = remote._consumer_evidence_artifact_uri(
                case.request["durable_output_root"],
                wave_index=case.request["wave_index"],
                shard_id=shard_id,
                filename="evidence.json",
            )
            deletion_uri = remote._consumer_evidence_artifact_uri(
                case.request["durable_output_root"],
                wave_index=case.request["wave_index"],
                shard_id=shard_id,
                filename="deletion-attestation.json",
            )
            consumer_artifacts[evidence_uri] = evidence_bytes
            consumer_artifacts[deletion_uri] = deletion_bytes
            deletion_record_sha256.append(deletion["closed_record_sha256"])
            closures.append(
                {
                    "deletion_file_sha256": remote.sha256(deletion_bytes).hexdigest(),
                    "deletion_record_sha256": deletion["closed_record_sha256"],
                    "evidence_file_sha256": remote.sha256(evidence_bytes).hexdigest(),
                    "evidence_record_sha256": evidence["closed_record_sha256"],
                    "shard_id": shard_id,
                }
            )
        case.consumer_artifacts = consumer_artifacts
        result = {
            "authorization_scope": "publication",
            "closed_record_sha256": "",
            "deletion_attestation_sha256": deletion_record_sha256,
            "execution_plan_sha256": case.request["execution_plan_sha256"],
            "next_wave_authorized": True,
            "record_type": remote.full_score.FULL_SCORE_WAVE_COMPLETION_RECORD_TYPE,
            "schema_version": 1,
            "shard_ids": case.shard_ids,
            "wave_index": 0,
        }
        tree_entries = []
    _close(result)
    result_bytes = remote._pretty_json_bytes(result)
    attestation = {
        "action": action,
        "attempt_id": case.request["attempt_id"],
        "closed_record_sha256": "",
        "durable_output_root": case.request["durable_output_root"],
        "execution_plan_sha256": case.request["execution_plan_sha256"],
        "package": case.request["package"],
        "phase_terminal": case.request["phase_terminal"],
        "record_type": remote.FULL_SCORE_REMOTE_COORDINATOR_ATTESTATION_RECORD_TYPE,
        "request_sha256": case.request["closed_record_sha256"],
        "result": {
            "file_sha256": remote.sha256(result_bytes).hexdigest(),
            "record_sha256": result["closed_record_sha256"],
            "record_type": result["record_type"],
            "uri": case.request["result_uri"],
        },
        "runner": case.request["runner"],
        "schema_version": remote.FULL_SCORE_REMOTE_COORDINATOR_ATTESTATION_SCHEMA_VERSION,
        "shard_count": 16,
        "shard_ids": case.shard_ids,
        "tree_closures": closures,
        "tree_entries": tree_entries,
        "wave_index": 0,
    }
    _close(attestation)
    return result, attestation


def _successful_run(payload, run_id=701):
    task = payload["tasks"][0]
    cluster = task["new_cluster"]
    return {
        "original_attempt_run_id": run_id,
        "run_id": run_id,
        "run_name": payload["run_name"],
        "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
        "tasks": [
            {
                "attempt_number": 0,
                "cluster_instance": {"cluster_id": "0824-remote-cpu"},
                "cluster_spec": {"new_cluster": copy.deepcopy(cluster)},
                "run_id": run_id + 1,
                "spark_python_task": copy.deepcopy(task["spark_python_task"]),
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "SUCCESS",
                },
                "task_key": task["task_key"],
            }
        ],
    }


def _persist_submission(
    tmp_path,
    monkeypatch,
    *,
    case,
    submit_payload,
    run_id=701,
):
    request_uri = submit_payload["tasks"][0]["spark_python_task"]["parameters"][7]

    def upload(_workspace, uri, content, *, max_bytes):
        expected = {
            request_uri: remote._pretty_json_bytes(case.request),
            case.request["runner"]["uri"]: (
                remote.FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SCRIPT.encode("utf-8")
            ),
        }
        assert content == expected[uri]
        assert len(content) <= max_bytes
        return {
            "created": True,
            "dbfs_uri": uri,
            "file_sha256": remote.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }

    monkeypatch.setattr(remote, "upload_databricks_volume_file_bytes_exclusive", upload)
    monkeypatch.setattr(
        remote,
        "submit_databricks_run",
        lambda _workspace, payload: {"run_id": run_id},
    )
    root = case.request.controller_lease_root
    response = remote.submit_full_score_remote_coordinator(
        DatabricksWorkspaceConfig("https://dbc.example", "secret"),
        submit_payload,
        request_uri=request_uri,
        request=case.request,
        controller_root=root,
    )
    assert response == {"run_id": run_id}
    return root


def test_remote_request_and_cpu_payload_are_closed_and_c5d_only():
    case = _request_case()
    payload = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        _request_uri(case),
        case.request,
    )
    assert case.request["closed_record_sha256"] == remote._closed_record_sha256(
        case.request
    )
    assert case.request["shard_ids"] == case.shard_ids
    assert payload["tasks"][0]["new_cluster"]["node_type_id"] == "c5d.4xlarge"
    assert (
        payload["tasks"][0]["new_cluster"]["spark_version"] == "15.4.x-cpu-ml-scala2.12"
    )
    assert payload["tasks"][0]["new_cluster"]["num_workers"] == 0
    assert (
        payload["tasks"][0]["new_cluster"]["custom_tags"]["campaign_closure"]
        == remote.PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256
    )
    assert payload["tasks"][0]["max_retries"] == 0
    assert payload["tasks"][0]["timeout_seconds"] == 7200
    assert remote.FULL_SCORE_REMOTE_COORDINATOR_JOB_COUNT == 20
    assert "idempotency_token" in payload
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert parameters[-2:] == [
        "--expected-request-record-sha256",
        case.request["closed_record_sha256"],
    ]
    assert (
        len(
            remote.json.dumps(
                parameters,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        <= remote.FULL_SCORE_REMOTE_COORDINATOR_PARAMETERS_MAX_BYTES
    )


def test_remote_request_authority_is_immutable_before_io(monkeypatch):
    case = _request_case()
    external_calls = []
    monkeypatch.setattr(
        remote,
        "upload_databricks_volume_file_bytes_exclusive",
        lambda *args, **kwargs: external_calls.append((args, kwargs)),
    )

    with pytest.raises(AttributeError, match="immutable"):
        case.request.workspace_host_sha256 = _digest("other-host")
    with pytest.raises(AttributeError, match="immutable"):
        case.request._request_bytes = b"{}\n"
    assert external_calls == []


def test_remote_workspace_identity_is_auth_only_and_preserves_canonical_bytes():
    first = _request_case()
    second_phase = _phase_authorization(
        first.execution["closed_record_sha256"],
        terminal_record_sha256=first.phase.terminal_record_sha256,
        phase_lease_root=first.phase.phase_lease_root,
        workspace_host="https://dbc-second.example",
        user_name="second-researcher@example.com",
    )
    second = _rebuild_request(first, phase_authorization=second_phase)

    assert second.to_record() == first.request.to_record()
    assert second.authorization_record() == first.request.authorization_record()
    assert second.workspace_host_sha256 != first.request.workspace_host_sha256
    assert second.user_name_sha256 != first.request.user_name_sha256
    assert remote.render_full_score_remote_coordinator_submit_payload(
        first.coordinator_config,
        _request_uri(first),
        second,
    ) == remote.render_full_score_remote_coordinator_submit_payload(
        first.coordinator_config,
        _request_uri(first),
        first.request,
    )
    first_result, first_attestation = _result_and_attestation(first)
    second_case_values = vars(first).copy()
    second_case_values["request"] = second
    second_case = SimpleNamespace(**second_case_values)
    second_result, second_attestation = _result_and_attestation(second_case)
    assert remote._pretty_json_bytes(second_result) == remote._pretty_json_bytes(
        first_result
    )
    assert remote._pretty_json_bytes(second_attestation) == remote._pretty_json_bytes(
        first_attestation
    )


def test_remote_request_binds_one_runner_and_attempt_identity():
    case = _request_case()
    request_uri = _request_uri(case)
    first = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        request_uri,
        case.request,
    )
    second = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        request_uri,
        case.request,
    )

    assert case.request["attempt_id"] == (
        "vllm-0271-publication-v1:full-score-remote:producer_ready:wave:000"
    )
    assert case.request["runner"] == {
        "file_sha256": remote.FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256,
        "uri": _remote_runner_uri(),
    }
    assert first == second
    assert first["idempotency_token"] == second["idempotency_token"]


def test_remote_request_rejects_alternate_control_uris_and_root():
    case = _request_case()
    alternate = "dbfs:/Volumes/catalog/schema/volume/durable/control/alternate.json"

    with pytest.raises(ValueError, match="result URI is not canonical"):
        _rebuild_request(case, result_uri=alternate)
    with pytest.raises(ValueError, match="attestation URI is not canonical"):
        _rebuild_request(case, attestation_uri=alternate)
    with pytest.raises(ValueError, match="runner URI is not canonical"):
        _rebuild_request(case, runner_python_file=alternate)
    with pytest.raises(ValueError, match="scope differs from worker payloads"):
        _rebuild_request(
            case,
            durable_output_root=(
                "dbfs:/Volumes/catalog/schema/volume/alternate-durable"
            ),
        )
    with pytest.raises(ValueError, match="request URI is not canonical"):
        remote.render_full_score_remote_coordinator_submit_payload(
            _coordinator_config(),
            alternate,
            case.request,
        )


def test_remote_render_rejects_runner_config_substitution():
    case = _request_case()
    substituted = remote.FullScoreRemoteCoordinatorJobConfig(
        runner_python_file=(
            "dbfs:/Volumes/catalog/schema/volume/runtime/unreviewed-runner.py"
        ),
        package_wheel_uri=("dbfs:/Volumes/catalog/schema/volume/runtime/cachet.whl"),
        package_wheel_sha256=_digest("package-wheel"),
        single_user_name="researcher@example.com",
    )

    with pytest.raises(ValueError, match="config/request runner binding drift"):
        remote.render_full_score_remote_coordinator_submit_payload(
            substituted,
            _request_uri(case),
            case.request,
        )


@pytest.mark.parametrize(
    "invalid_principal",
    ["researcher\x00@example.com", "researcher\x7f@example.com"],
)
def test_remote_coordinator_config_rejects_control_character_principal(
    invalid_principal,
):
    with pytest.raises(ValueError, match="SINGLE_USER principal"):
        replace(
            _coordinator_config(),
            single_user_name=invalid_principal,
        )


def test_remote_submit_rejects_runner_or_attempt_substitution_before_io(
    tmp_path,
    monkeypatch,
):
    case = _request_case(phase_lease_root=tmp_path / "phase-leases")
    request_uri = _request_uri(case)
    valid = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        request_uri,
        case.request,
    )
    substituted_runner = copy.deepcopy(valid)
    substituted_runner["tasks"][0]["spark_python_task"]["python_file"] = (
        "dbfs:/Volumes/catalog/schema/volume/runtime/unreviewed-runner.py"
    )
    substituted_attempt = remote.bind_databricks_run_idempotency_token(
        {key: value for key, value in valid.items() if key != "idempotency_token"},
        attempt_id="unreviewed-second-attempt",
    )
    calls = []
    monkeypatch.setattr(
        remote,
        "upload_databricks_volume_file_bytes_exclusive",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        remote,
        "submit_databricks_run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    workspace = DatabricksWorkspaceConfig("https://dbc.example", "secret")

    with pytest.raises(ValueError, match="runner URI binding drift"):
        remote.submit_full_score_remote_coordinator(
            workspace,
            substituted_runner,
            request_uri=request_uri,
            request=case.request,
            controller_root=case.request.controller_lease_root,
        )
    with pytest.raises(ValueError, match="idempotency token drift"):
        remote.submit_full_score_remote_coordinator(
            workspace,
            substituted_attempt,
            request_uri=request_uri,
            request=case.request,
            controller_root=case.request.controller_lease_root,
        )
    assert calls == []
    assert not case.request.controller_lease_root.exists()


def test_remote_wrong_current_user_fails_before_controller_state_or_io(
    tmp_path,
    monkeypatch,
):
    case = _request_case(phase_lease_root=tmp_path / "phase-leases")
    request_uri = _request_uri(case)
    payload = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        request_uri,
        case.request,
    )
    observed = []
    external_calls = []

    def reject_current_user(_workspace, *, expected_user_name, opener=None):
        observed.append((expected_user_name, opener))
        raise ValueError("Databricks current-user identity differs")

    monkeypatch.setattr(
        remote,
        "require_databricks_current_user_name",
        reject_current_user,
    )
    monkeypatch.setattr(
        remote,
        "upload_databricks_volume_file_bytes_exclusive",
        lambda *args, **kwargs: external_calls.append(("upload", args, kwargs)),
    )
    monkeypatch.setattr(
        remote,
        "submit_databricks_run",
        lambda *args, **kwargs: external_calls.append(("post", args, kwargs)),
    )
    workspace = DatabricksWorkspaceConfig("https://dbc.example", "secret")
    controller_root = case.request.controller_lease_root

    with pytest.raises(ValueError, match="current-user identity differs"):
        remote.submit_full_score_remote_coordinator(
            workspace,
            payload,
            request_uri=request_uri,
            request=case.request,
            controller_root=controller_root,
        )
    assert external_calls == []
    assert not controller_root.exists()

    with pytest.raises(ValueError, match="current-user identity differs"):
        remote.recover_full_score_remote_coordinator_submission(
            workspace,
            controller_root=controller_root,
            request_authorization=case.request,
        )
    assert observed == [
        ("researcher@example.com", None),
        ("researcher@example.com", None),
    ]
    assert external_calls == []
    assert not controller_root.exists()


def test_remote_payload_rejects_noncanonical_uri_and_parameters_above_9500_bytes():
    case = _request_case()
    oversized_uri = (
        "dbfs:/Volumes/catalog/schema/volume/durable/control/" + "x" * 9_500 + ".json"
    )

    with pytest.raises(ValueError, match="request URI is not canonical"):
        remote.render_full_score_remote_coordinator_submit_payload(
            _coordinator_config(),
            oversized_uri,
            case.request,
        )
    with pytest.raises(ValueError, match="9500-byte"):
        remote._require_bounded_remote_coordinator_parameters(["x" * 9_501])


def test_remote_request_rejects_arbitrary_self_hashed_coordinator_wheel():
    case = _request_case()
    with pytest.raises(ValueError, match="differs from worker payloads"):
        _rebuild_request(
            case,
            package_wheel_uri=(
                "dbfs:/Volumes/catalog/schema/volume/runtime/unreviewed.whl"
            ),
            package_wheel_sha256=_digest("unreviewed-self-hashed-wheel"),
        )

    unreviewed_config = remote.FullScoreRemoteCoordinatorJobConfig(
        runner_python_file=(_remote_runner_uri()),
        package_wheel_uri=(
            "dbfs:/Volumes/catalog/schema/volume/runtime/unreviewed.whl"
        ),
        package_wheel_sha256=_digest("unreviewed-self-hashed-wheel"),
        single_user_name="researcher@example.com",
    )
    with pytest.raises(ValueError, match="config/request package binding drift"):
        remote.render_full_score_remote_coordinator_submit_payload(
            unreviewed_config,
            _request_uri(case),
            case.request,
        )


def test_resealed_raw_request_cannot_render_or_submit_before_io(tmp_path, monkeypatch):
    case = _request_case()
    forged = case.request.to_record()
    forged["package"] = {
        "runner_sha256": remote.FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256,
        "wheel_sha256": _digest("unreviewed-self-hashed-wheel"),
        "wheel_uri": "dbfs:/Volumes/catalog/schema/volume/runtime/unreviewed.whl",
    }
    forged["coordinator"]["package_wheel_uri"] = forged["package"]["wheel_uri"]
    forged["coordinator"]["package_wheel_sha256"] = forged["package"]["wheel_sha256"]
    _close(forged)
    remote.validate_full_score_remote_coordinator_request(forged)
    unreviewed_config = remote.FullScoreRemoteCoordinatorJobConfig(
        runner_python_file=(_remote_runner_uri()),
        package_wheel_uri=forged["package"]["wheel_uri"],
        package_wheel_sha256=forged["package"]["wheel_sha256"],
        single_user_name="researcher@example.com",
    )
    request_uri = _request_uri(case)

    with pytest.raises(TypeError, match="builder-issued request authorization"):
        remote.render_full_score_remote_coordinator_submit_payload(
            unreviewed_config,
            request_uri,
            forged,
        )

    valid_payload = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        request_uri,
        case.request,
    )
    calls = []
    monkeypatch.setattr(
        remote,
        "upload_databricks_volume_file_bytes_exclusive",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        remote,
        "submit_databricks_run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with pytest.raises(TypeError, match="builder-issued request authorization"):
        remote.submit_full_score_remote_coordinator(
            DatabricksWorkspaceConfig("https://dbc.example", "secret"),
            valid_payload,
            request_uri=request_uri,
            request=forged,
            controller_root=tmp_path / "must-not-exist",
        )
    assert calls == []
    assert not (tmp_path / "must-not-exist").exists()


def test_runner_checks_request_package_authority_before_import():
    script = remote.FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SCRIPT
    package_check = script.index("package authority drift")
    wheel_check = script.index("package wheel drift")
    import_package = script.index(
        "from document_kv_cache.full_score_remote_control import coordinator_main"
    )
    assert package_check < wheel_check < import_package


def test_remote_request_rejects_cross_worker_package_drift():
    case = _request_case()
    drifted = [(uri, copy.deepcopy(record)) for uri, record in case.payloads]
    drift_sha = _digest("cross-worker-package-drift")
    drift_record = drifted[-1][1]
    bootstrap = drift_record["bootstrap_artifacts"]
    bootstrap["package_wheel_sha256"] = drift_sha
    drift_record["gpu_qualification"]["artifact_pins"]["package_wheel_sha256"] = (
        drift_sha
    )
    bootstrap["locked_runtime_identity_sha256"] = (
        remote.full_score._locked_runtime_identity_sha256(
            runner_sha256=bootstrap["runner_sha256"],
            package_wheel_sha256=drift_sha,
            runtime_lock_sha256=bootstrap["runtime_lock_sha256"],
            patched_vllm_wheel_sha256=bootstrap["patched_vllm_wheel_sha256"],
            patched_flashinfer_wheel_sha256=(
                bootstrap["patched_flashinfer_wheel_sha256"]
            ),
            runtime_closure_manifest_sha256=(
                bootstrap["runtime_closure_manifest_sha256"]
            ),
        )
    )
    _close(drift_record)

    with pytest.raises(ValueError, match="differ from phase terminal"):
        _rebuild_request(case, worker_payloads=drifted)


def test_remote_worker_runtime_closure_is_v2_and_runner_pins_are_separate():
    case = _request_case()
    records = [record for _uri, record in case.payloads]
    remote._require_worker_package_binding(
        records,
        package_wheel_uri=("dbfs:/Volumes/catalog/schema/volume/runtime/cachet.whl"),
        package_wheel_sha256=_digest("package-wheel"),
    )
    first = records[0]
    assert (
        first["gpu_qualification"]["artifact_pins"]["runner_sha256"]
        != (first["bootstrap_artifacts"]["runner_sha256"])
    )

    conflated = copy.deepcopy(records)
    conflated[0]["gpu_qualification"]["artifact_pins"]["runner_sha256"] = conflated[0][
        "bootstrap_artifacts"
    ]["runner_sha256"]
    with pytest.raises(ValueError, match="qualification/runtime package binding drift"):
        remote._require_worker_package_binding(
            conflated,
            package_wheel_uri=(
                "dbfs:/Volumes/catalog/schema/volume/runtime/cachet.whl"
            ),
            package_wheel_sha256=_digest("package-wheel"),
        )

    tampered = copy.deepcopy(records)
    tampered[0]["bootstrap_artifacts"]["runtime_closure_manifest_sha256"] = _digest(
        "unreviewed-runtime-closure"
    )
    with pytest.raises(ValueError, match="qualification/runtime package binding drift"):
        remote._require_worker_package_binding(
            tampered,
            package_wheel_uri=(
                "dbfs:/Volumes/catalog/schema/volume/runtime/cachet.whl"
            ),
            package_wheel_sha256=_digest("package-wheel"),
        )


def test_remote_request_rejects_resealed_workers_not_bound_by_phase_terminal():
    case = _request_case()
    unreviewed_uri = "dbfs:/Volumes/catalog/schema/volume/runtime/unreviewed.whl"
    unreviewed_sha = _digest("unreviewed-resealed-worker-wheel")
    drifted = [(uri, copy.deepcopy(record)) for uri, record in case.payloads]
    for _uri, record in drifted:
        bootstrap = record["bootstrap_artifacts"]
        bootstrap["package_wheel_uri"] = unreviewed_uri
        bootstrap["package_wheel_sha256"] = unreviewed_sha
        record["gpu_qualification"]["artifact_pins"]["package_wheel_sha256"] = (
            unreviewed_sha
        )
        bootstrap["locked_runtime_identity_sha256"] = (
            remote.full_score._locked_runtime_identity_sha256(
                runner_sha256=bootstrap["runner_sha256"],
                package_wheel_sha256=unreviewed_sha,
                runtime_lock_sha256=bootstrap["runtime_lock_sha256"],
                patched_vllm_wheel_sha256=bootstrap["patched_vllm_wheel_sha256"],
                patched_flashinfer_wheel_sha256=(
                    bootstrap["patched_flashinfer_wheel_sha256"]
                ),
                runtime_closure_manifest_sha256=(
                    bootstrap["runtime_closure_manifest_sha256"]
                ),
            )
        )
        _close(record)

    with pytest.raises(ValueError, match="differ from phase terminal"):
        _rebuild_request(
            case,
            worker_payloads=drifted,
            package_wheel_uri=unreviewed_uri,
            package_wheel_sha256=unreviewed_sha,
        )


def test_controller_concurrent_submission_uploads_and_posts_once(tmp_path, monkeypatch):
    case = _request_case(phase_lease_root=tmp_path / "phase-leases")
    request_uri = _request_uri(case)
    payload = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        request_uri,
        case.request,
    )
    upload_calls = []
    post_calls = []

    def upload(_workspace, uri, content, *, max_bytes):
        upload_calls.append((uri, content, max_bytes))
        return {
            "created": True,
            "dbfs_uri": uri,
            "file_sha256": remote.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }

    def post(_workspace, observed_payload):
        post_calls.append(copy.deepcopy(observed_payload))
        return {"run_id": 701}

    monkeypatch.setattr(remote, "upload_databricks_volume_file_bytes_exclusive", upload)
    monkeypatch.setattr(remote, "submit_databricks_run", post)
    workspace = DatabricksWorkspaceConfig("https://dbc.example", "secret")
    controller_root = case.request.controller_lease_root

    def submit_once(_index):
        return remote.submit_full_score_remote_coordinator(
            workspace,
            payload,
            request_uri=request_uri,
            request=case.request,
            controller_root=controller_root,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(submit_once, range(2)))

    assert responses == [{"run_id": 701}, {"run_id": 701}]
    assert len(upload_calls) == 2
    assert len(post_calls) == 1
    assert post_calls[0]["idempotency_token"] == payload["idempotency_token"]
    assert {
        "request.json",
        "submit-payload.json",
        "request-upload-receipt.json",
        "runner-upload-receipt.json",
        "post-intent.json",
        "submit-response.json",
        "workspace-authority.json",
    }.issubset(path.name for path in controller_root.iterdir())


def test_controller_rejects_second_config_and_root_before_second_post(
    tmp_path,
    monkeypatch,
):
    case = _request_case(phase_lease_root=tmp_path / "phase-leases")
    request_uri = _request_uri(case)
    payload = remote.render_full_score_remote_coordinator_submit_payload(
        case.coordinator_config,
        request_uri,
        case.request,
    )
    alternate_config = replace(
        case.coordinator_config,
        single_user_name="other-researcher@example.com",
        custom_tags={"unreviewed": "alternate"},
    )
    external_calls = []

    def upload(_workspace, uri, content, *, max_bytes):
        external_calls.append(("upload", uri))
        return {
            "created": True,
            "dbfs_uri": uri,
            "file_sha256": remote.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }

    def post(_workspace, observed_payload):
        external_calls.append(("post", observed_payload["idempotency_token"]))
        return {"run_id": 701}

    monkeypatch.setattr(remote, "upload_databricks_volume_file_bytes_exclusive", upload)
    monkeypatch.setattr(remote, "submit_databricks_run", post)
    workspace = DatabricksWorkspaceConfig("https://dbc.example", "secret")
    assert remote.submit_full_score_remote_coordinator(
        workspace,
        payload,
        request_uri=request_uri,
        request=case.request,
        controller_root=case.request.controller_lease_root,
    ) == {"run_id": 701}
    accepted_call_count = len(external_calls)

    with pytest.raises(ValueError, match="config/request job binding drift"):
        remote.render_full_score_remote_coordinator_submit_payload(
            alternate_config,
            request_uri,
            case.request,
        )
    assert len(external_calls) == accepted_call_count
    alternate_payload = copy.deepcopy(payload)
    cluster = alternate_payload["tasks"][0]["new_cluster"]
    cluster["single_user_name"] = alternate_config.single_user_name
    cluster["custom_tags"]["unreviewed"] = "alternate"
    alternate_payload = remote.bind_databricks_run_idempotency_token(
        {
            key: value
            for key, value in alternate_payload.items()
            if key != "idempotency_token"
        },
        attempt_id=case.request["attempt_id"],
    )
    with pytest.raises(ValueError, match="CPU job config binding drift"):
        remote.submit_full_score_remote_coordinator(
            workspace,
            alternate_payload,
            request_uri=request_uri,
            request=case.request,
            controller_root=case.request.controller_lease_root,
        )
    assert len(external_calls) == accepted_call_count

    alternate_root = tmp_path / "alternate-controller"
    with pytest.raises(ValueError, match="differs from singleton authority"):
        remote.submit_full_score_remote_coordinator(
            workspace,
            payload,
            request_uri=request_uri,
            request=case.request,
            controller_root=alternate_root,
        )

    assert len(external_calls) == accepted_call_count
    assert [kind for kind, _value in external_calls].count("post") == 1
    assert not alternate_root.exists()


def test_compact_cas_serializes_concurrent_total_cap_admission(tmp_path):
    root = tmp_path / "cas"
    first = remote.FullScoreCompactArtifactCAS(
        root,
        max_file_bytes=8,
        max_total_bytes=6,
    )
    second = remote.FullScoreCompactArtifactCAS(
        root,
        max_file_bytes=8,
        max_total_bytes=6,
    )

    def publish(cas, suffix):
        return cas.bind_bytes(
            f"dbfs:/Volumes/catalog/schema/volume/{suffix}.json",
            suffix.encode("utf-8") * 4,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publish, first, "a"),
            executor.submit(publish, second, "b"),
        ]
    successes = [future for future in futures if future.exception() is None]
    failures = [future.exception() for future in futures if future.exception()]

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert (
        sum(path.stat().st_size for path in (root / "blobs" / "sha256").iterdir()) <= 6
    )


def test_compact_cas_rejects_symlink_ancestors_and_nonregular_entries(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        remote.FullScoreCompactArtifactCAS(linked_parent / "cas")

    parent = tmp_path / "stable-parent"
    cas = remote.FullScoreCompactArtifactCAS(parent / "cas")
    uri = "dbfs:/Volumes/catalog/schema/volume/result.json"
    cas.bind_bytes(uri, b"record")
    (cas.root / "blobs" / "sha256" / "unexpected").mkdir()
    with pytest.raises(ValueError, match="regular file"):
        cas.resolve(uri)


def test_compact_cas_resolve_rechecks_symlink_ancestors(tmp_path):
    parent = tmp_path / "cas-parent"
    cas = remote.FullScoreCompactArtifactCAS(parent / "cas")
    uri = "dbfs:/Volumes/catalog/schema/volume/result.json"
    cas.bind_bytes(uri, b"record")
    relocated = tmp_path / "relocated-parent"
    parent.rename(relocated)
    parent.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        cas.resolve(uri)


def test_publication_cas_admission_bound_covers_all_paired_scores_and_metadata():
    assert remote.FULL_SCORE_REMOTE_EVIDENCE_PAIR_OUTPUT_MAX_BYTES == 1_024
    assert remote.FULL_SCORE_REMOTE_EVIDENCE_PAIR_MAX_BYTES == 13_312
    assert remote.FULL_SCORE_REMOTE_CAS_MAX_BYTES == 1_543_504_896
    assert remote.FULL_SCORE_REMOTE_CAS_MIRROR_MAX_BYTES == 1_545_143_296
    assert 2 * 160 + 4 * 10 == 360
    assert remote.FULL_SCORE_REMOTE_CAS_MAX_BINDINGS == 400


def test_controller_recovers_lost_submit_response_with_same_token(
    tmp_path, monkeypatch
):
    case = _request_case(phase_lease_root=tmp_path / "phase-leases")
    request_uri = _request_uri(case)
    payload = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        request_uri,
        case.request,
    )
    upload_calls = []
    post_tokens = []

    def upload(_workspace, uri, content, *, max_bytes):
        upload_calls.append(uri)
        return {
            "created": True,
            "dbfs_uri": uri,
            "file_sha256": remote.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }

    def post(_workspace, observed_payload):
        post_tokens.append(observed_payload["idempotency_token"])
        if len(post_tokens) == 1:
            raise TimeoutError("accepted response lost")
        return {"run_id": 701}

    monkeypatch.setattr(remote, "upload_databricks_volume_file_bytes_exclusive", upload)
    monkeypatch.setattr(remote, "submit_databricks_run", post)
    workspace = DatabricksWorkspaceConfig("https://dbc.example", "secret")
    controller_root = case.request.controller_lease_root

    with pytest.raises(TimeoutError, match="accepted response lost"):
        remote.submit_full_score_remote_coordinator(
            workspace,
            payload,
            request_uri=request_uri,
            request=case.request,
            controller_root=controller_root,
        )

    assert (controller_root / "post-intent.json").is_file()
    assert not (controller_root / "submit-response.json").exists()
    controller_snapshot = {
        path.relative_to(controller_root): path.read_bytes()
        for path in controller_root.rglob("*")
        if path.is_file()
    }
    observed_principals = []

    def reject_current_user(_workspace, *, expected_user_name, opener=None):
        observed_principals.append((expected_user_name, opener))
        raise ValueError("Databricks current-user identity differs")

    with monkeypatch.context() as identity_patch:
        identity_patch.setattr(
            remote,
            "require_databricks_current_user_name",
            reject_current_user,
        )
        with pytest.raises(ValueError, match="current-user identity differs"):
            remote.recover_full_score_remote_coordinator_submission(
                workspace,
                controller_root=controller_root,
                request_authorization=case.request,
            )
    assert observed_principals == [("researcher@example.com", None)]
    assert post_tokens == [payload["idempotency_token"]]
    assert not (controller_root / "submit-response.json").exists()
    assert {
        path.relative_to(controller_root): path.read_bytes()
        for path in controller_root.rglob("*")
        if path.is_file()
    } == controller_snapshot

    assert remote.recover_full_score_remote_coordinator_submission(
        workspace,
        controller_root=controller_root,
        request_authorization=case.request,
    ) == {"run_id": 701}
    assert post_tokens == [payload["idempotency_token"]] * 2
    assert upload_calls == [case.request["runner"]["uri"], request_uri]
    assert (controller_root / "submit-response.json").is_file()


def test_controller_recovery_rejects_tampered_local_request(tmp_path, monkeypatch):
    case = _request_case(phase_lease_root=tmp_path / "phase-leases")
    submit = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        _request_uri(case),
        case.request,
    )
    root = _persist_submission(
        tmp_path,
        monkeypatch,
        case=case,
        submit_payload=submit,
    )
    request_path = root / "request.json"
    tampered = remote.json.loads(request_path.read_text(encoding="utf-8"))
    tampered["wave_index"] = 9
    request_path.write_bytes(remote._pretty_json_bytes(tampered))

    with pytest.raises(ValueError, match="closure drift"):
        remote.recover_full_score_remote_coordinator_submission(
            DatabricksWorkspaceConfig("https://dbc.example", "secret"),
            controller_root=root,
            request_authorization=case.request,
        )


@pytest.mark.parametrize("mutation", ["tamper", "missing", "fifo"])
def test_controller_recovery_requires_exact_workspace_authority(
    tmp_path, monkeypatch, mutation
):
    case = _request_case(phase_lease_root=tmp_path / "phase-leases")
    submit = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        _request_uri(case),
        case.request,
    )
    root = _persist_submission(
        tmp_path,
        monkeypatch,
        case=case,
        submit_payload=submit,
    )
    sidecar = root / "workspace-authority.json"
    if mutation == "tamper":
        record = remote.json.loads(sidecar.read_text(encoding="utf-8"))
        record["workspace_host_sha256"] = _digest("other-workspace")
        record["closed_record_sha256"] = remote._closed_record_sha256(record)
        sidecar.write_bytes(remote._pretty_json_bytes(record))
    else:
        sidecar.unlink()
        if mutation == "fifo":
            remote.os.mkfifo(sidecar, 0o600)
    post_calls = []
    monkeypatch.setattr(
        remote,
        "submit_databricks_run",
        lambda *args, **kwargs: post_calls.append((args, kwargs)),
    )

    with pytest.raises((ValueError, FileNotFoundError)):
        remote.recover_full_score_remote_coordinator_submission(
            DatabricksWorkspaceConfig("https://dbc.example", "secret"),
            controller_root=root,
            request_authorization=case.request,
        )
    assert post_calls == []


def test_controller_recovery_requires_durable_request_authorization(
    tmp_path, monkeypatch
):
    case = _request_case(phase_lease_root=tmp_path / "phase-leases")
    request_uri = _request_uri(case)
    submit = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        request_uri,
        case.request,
    )
    root = _persist_submission(
        tmp_path,
        monkeypatch,
        case=case,
        submit_payload=submit,
    )
    (root / "request-authorization.json").unlink()
    post_calls = []
    monkeypatch.setattr(
        remote,
        "submit_databricks_run",
        lambda *args, **kwargs: post_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="request authorization.*missing"):
        remote.recover_full_score_remote_coordinator_submission(
            DatabricksWorkspaceConfig("https://dbc.example", "secret"),
            controller_root=root,
            request_authorization=case.request,
        )
    assert post_calls == []


def test_controller_recovery_requires_bound_runner_upload_receipt(
    tmp_path, monkeypatch
):
    case = _request_case(phase_lease_root=tmp_path / "phase-leases")
    submit = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        _request_uri(case),
        case.request,
    )
    root = _persist_submission(
        tmp_path,
        monkeypatch,
        case=case,
        submit_payload=submit,
    )
    (root / "submit-response.json").unlink()
    (root / "runner-upload-receipt.json").unlink()
    post_calls = []
    monkeypatch.setattr(
        remote,
        "submit_databricks_run",
        lambda *args, **kwargs: post_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="runner upload receipt.*missing"):
        remote.recover_full_score_remote_coordinator_submission(
            DatabricksWorkspaceConfig("https://dbc.example", "secret"),
            controller_root=root,
            request_authorization=case.request,
        )
    assert post_calls == []


def test_controller_recovery_rejects_coordinated_self_resealed_request_files(
    tmp_path, monkeypatch
):
    case = _request_case(phase_lease_root=tmp_path / "phase-leases")
    request_uri = _request_uri(case)
    submit = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        request_uri,
        case.request,
    )
    root = _persist_submission(
        tmp_path,
        monkeypatch,
        case=case,
        submit_payload=submit,
    )
    (root / "submit-response.json").unlink()
    forged = case.request.to_record()
    forged["package"] = {
        "runner_sha256": remote.FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256,
        "wheel_sha256": _digest("coordinated-unreviewed-wheel"),
        "wheel_uri": "dbfs:/Volumes/catalog/schema/volume/runtime/unreviewed.whl",
    }
    forged["coordinator"]["package_wheel_uri"] = forged["package"]["wheel_uri"]
    forged["coordinator"]["package_wheel_sha256"] = forged["package"]["wheel_sha256"]
    _close(forged)
    forged_authorization = remote._remote_coordinator_request_authorization_record(
        forged
    )
    (root / "request.json").write_bytes(remote._pretty_json_bytes(forged))
    (root / "request-authorization.json").write_bytes(
        remote._pretty_json_bytes(forged_authorization)
    )
    post_calls = []
    monkeypatch.setattr(
        remote,
        "submit_databricks_run",
        lambda *args, **kwargs: post_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="differs from live authority"):
        remote.recover_full_score_remote_coordinator_submission(
            DatabricksWorkspaceConfig("https://dbc.example", "secret"),
            controller_root=root,
            request_authorization=case.request,
        )
    assert post_calls == []


def test_exact_tree_inventory_rejects_missing_extra_file_and_symlink(tmp_path):
    ready = tmp_path / "ready"
    ready.mkdir()
    (ready / "shard-a").mkdir()
    (ready / "shard-b").mkdir()
    assert remote._require_exact_child_directories(
        ready, {"shard-a", "shard-b"}, label="ready"
    ) == ["shard-a", "shard-b"]

    (ready / "shard-b").rmdir()
    with pytest.raises(ValueError, match="missing or extra"):
        remote._require_exact_child_directories(
            ready, {"shard-a", "shard-b"}, label="ready"
        )
    (ready / "extra").mkdir()
    with pytest.raises(ValueError, match="missing or extra"):
        remote._require_exact_child_directories(ready, {"shard-a"}, label="ready")
    (ready / "extra").rmdir()
    (ready / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="non-directory"):
        remote._require_exact_child_directories(ready, {"shard-a"}, label="ready")
    (ready / "unexpected.json").unlink()
    (ready / "link").symlink_to(ready / "shard-a", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        remote._require_exact_child_directories(
            ready, {"shard-a", "link"}, label="ready"
        )


def test_remote_attestation_rejects_tampered_tree_closure():
    case = _request_case()
    result, attestation = _result_and_attestation(case)
    tampered = copy.deepcopy(attestation)
    tampered["tree_closures"][0]["files_sha256"] = _digest("tampered")

    with pytest.raises(ValueError, match="identity/closure drift"):
        remote.validate_full_score_remote_coordinator_attestation(
            tampered,
            request=case.request,
            result=result,
            result_file_sha256=remote.sha256(
                remote._pretty_json_bytes(result)
            ).hexdigest(),
        )


def test_remote_worker_invokes_existing_exact_producer_validator(tmp_path, monkeypatch):
    case = _request_case()
    records = {
        binding["uri"]: record
        for binding, (_uri, record) in zip(
            case.request["worker_payloads"], case.payloads, strict=True
        )
    }
    records.update(
        {
            case.request["sources"]["inventory"]["uri"]: case.inventory,
            case.request["sources"]["shard_plan"]["uri"]: case.shard_plan,
            case.request["sources"]["execution_plan"]["uri"]: case.execution,
        }
    )
    monkeypatch.setattr(
        remote,
        "_read_bound_volume_record",
        lambda binding: copy.deepcopy(records[binding["uri"]]),
    )
    monkeypatch.setattr(
        remote.full_score,
        "full_score_inventory_from_record",
        lambda _record: object(),
    )
    ready_wave = tmp_path / "ready" / "wave-000"
    for shard_id in case.shard_ids:
        shard = ready_wave / shard_id
        shard.mkdir(parents=True)
        ready_record = _closed_record(
            shard_id,
            files_sha256=_digest(f"files-{shard_id}"),
        )
        (shard / "ready-record.json").write_bytes(
            remote._pretty_json_bytes(ready_record)
        )
    monkeypatch.setattr(
        remote,
        "_volume_mount_path",
        lambda uri, _field: (
            tmp_path / "result.json"
            if uri == case.request["result_uri"]
            else tmp_path / "attestation.json"
            if uri == case.request["attestation_uri"]
            else tmp_path
        ),
    )
    monkeypatch.setattr(
        remote,
        "_remote_cluster_path",
        lambda value: tmp_path / str(value).split("/")[-1],
    )
    called = []
    result, attestation = _result_and_attestation(case)

    def exact_validator(execution, **kwargs):
        called.append((execution, kwargs))
        return result

    monkeypatch.setattr(
        remote.full_score,
        "build_governed_full_score_producer_phase_completion_record",
        exact_validator,
    )
    monkeypatch.setattr(
        remote,
        "validate_full_score_remote_coordinator_attestation",
        lambda *args, **kwargs: None,
    )
    request_path = tmp_path / "request.json"
    request_path.write_bytes(remote._pretty_json_bytes(case.request))

    observed_result, observed_attestation = remote.run_full_score_remote_coordinator(
        request_path,
        expected_request_file_sha256=remote.sha256(
            request_path.read_bytes()
        ).hexdigest(),
        expected_request_record_sha256=case.request["closed_record_sha256"],
    )

    assert observed_result == result
    assert observed_attestation["action"] == attestation["action"]
    assert len(called) == 1
    assert called[0][1]["producer_payloads"] == tuple(
        record for _uri, record in case.payloads
    )


def test_mac_collection_uses_runs_get_files_api_cas_and_no_dbfs_mount(
    tmp_path, monkeypatch
):
    case = _request_case(phase_lease_root=tmp_path / "phase-leases")
    submit = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        _request_uri(case),
        case.request,
    )
    controller_root = _persist_submission(
        tmp_path,
        monkeypatch,
        case=case,
        submit_payload=submit,
    )
    result, attestation = _result_and_attestation(case)
    downloads = {
        case.request["result_uri"]: remote._pretty_json_bytes(result),
        case.request["attestation_uri"]: remote._pretty_json_bytes(attestation),
    }
    observed_uris = []
    monkeypatch.setattr(
        remote,
        "get_databricks_run",
        lambda _workspace, run_id: _successful_run(submit, int(run_id)),
    )

    def download(_workspace, uri, *, max_bytes):
        observed_uris.append(("download", uri, max_bytes))
        return downloads[uri]

    monkeypatch.setattr(remote, "download_databricks_volume_file_bytes", download)

    def listing(_workspace, uri, *, max_entries):
        observed_uris.append(("list", uri, max_entries))
        return tuple(
            {
                "is_directory": True,
                "name": shard_id,
                "path": uri.removeprefix("dbfs:") + "/" + shard_id + "/",
            }
            for shard_id in case.shard_ids
        )

    monkeypatch.setattr(remote, "list_databricks_volume_directory", listing)
    cas = remote.FullScoreCompactArtifactCAS(tmp_path / "cas")
    authority = remote.collect_full_score_remote_coordinator(
        DatabricksWorkspaceConfig("https://dbc.example", "secret"),
        controller_root=controller_root,
        cas=cas,
        request_authorization=case.request,
    )

    assert authority.result_record == result
    assert authority.coordinator_run_id == "701"
    assert authority.controller_authorization_record_sha256
    assert (controller_root / "runs-get-receipt.json").is_file()
    assert (controller_root / "authorization.json").is_file()
    assert cas.resolve(case.request["result_uri"]).is_file()
    assert all(item[1].startswith("dbfs:/Volumes/") for item in observed_uris)
    assert all("/dbfs" not in item[1] for item in observed_uris)
    replayed = remote.replay_full_score_remote_coordinator_authorization(
        DatabricksWorkspaceConfig("https://dbc.example", "secret"),
        controller_root=controller_root,
        cas=cas,
        request_authorization=case.request,
    )
    assert replayed.result_record == authority.result_record
    assert (
        replayed.controller_authorization_record_sha256
        == authority.controller_authorization_record_sha256
    )

    monkeypatch.setattr(
        remote,
        "get_databricks_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("authenticated remote run unavailable")
        ),
    )
    with pytest.raises(ValueError, match="authenticated remote run unavailable"):
        remote.replay_full_score_remote_coordinator_authorization(
            DatabricksWorkspaceConfig("https://dbc.example", "secret"),
            controller_root=controller_root,
            cas=cas,
            request_authorization=case.request,
        )


def test_failed_coordinator_cannot_issue_remote_authority(tmp_path, monkeypatch):
    case = _request_case(phase_lease_root=tmp_path / "phase-leases")
    submit = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        _request_uri(case),
        case.request,
    )
    controller_root = _persist_submission(
        tmp_path,
        monkeypatch,
        case=case,
        submit_payload=submit,
    )
    failed = _successful_run(submit)
    failed["state"]["result_state"] = "FAILED"
    monkeypatch.setattr(remote, "get_databricks_run", lambda *_args: failed)
    monkeypatch.setattr(
        remote,
        "download_databricks_volume_file_bytes",
        lambda *_args, **_kwargs: pytest.fail("failed run must not download outputs"),
    )

    with pytest.raises(ValueError, match="not successful"):
        remote.collect_full_score_remote_coordinator(
            DatabricksWorkspaceConfig("https://dbc.example", "secret"),
            controller_root=controller_root,
            cas=remote.FullScoreCompactArtifactCAS(tmp_path / "cas"),
            request_authorization=case.request,
        )


def test_remote_run_requires_unrepaired_attempt_zero_and_bound_task_cluster_ids():
    case = _request_case()
    submit = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        _request_uri(case),
        case.request,
    )
    run = _successful_run(submit)

    for attempt_number in (1, False):
        mutated = copy.deepcopy(run)
        mutated["tasks"][0]["attempt_number"] = attempt_number
        with pytest.raises(ValueError, match="attempt zero"):
            remote._validate_successful_remote_coordinator_run(
                mutated, submit_payload=submit
            )

    mutated = copy.deepcopy(run)
    mutated["original_attempt_run_id"] += 10
    with pytest.raises(ValueError, match="original attempt"):
        remote._validate_successful_remote_coordinator_run(
            mutated, submit_payload=submit
        )

    mutated = copy.deepcopy(run)
    mutated["repair_history"] = [{"type": "REPAIR_ALL"}]
    with pytest.raises(ValueError, match="repaired runs"):
        remote._validate_successful_remote_coordinator_run(
            mutated, submit_payload=submit
        )

    mutated = copy.deepcopy(run)
    mutated["tasks"][0]["run_id"] = mutated["run_id"]
    with pytest.raises(ValueError, match="must differ"):
        remote._validate_successful_remote_coordinator_run(
            mutated, submit_payload=submit
        )

    mutated = copy.deepcopy(run)
    mutated["tasks"][0]["cluster_instance"]["cluster_id"] = ""
    with pytest.raises(ValueError, match="cluster_id"):
        remote._validate_successful_remote_coordinator_run(
            mutated, submit_payload=submit
        )

    governed_cluster = submit["tasks"][0]["new_cluster"]
    for cluster_field, substituted_value in (
        ("single_user_name", "substituted-researcher@example.com"),
        (
            "custom_tags",
            {**governed_cluster["custom_tags"], "purpose": "substituted-verifier"},
        ),
        (
            "spark_conf",
            {**governed_cluster["spark_conf"], "spark.master": "local[1]"},
        ),
        (
            "aws_attributes",
            {**governed_cluster["aws_attributes"], "availability": "SPOT"},
        ),
    ):
        mutated = copy.deepcopy(run)
        mutated["tasks"][0]["cluster_spec"]["new_cluster"][cluster_field] = (
            substituted_value
        )
        with pytest.raises(ValueError, match="status/topology drift"):
            remote._validate_successful_remote_coordinator_run(
                mutated, submit_payload=submit
            )

    mutated = copy.deepcopy(run)
    del mutated["tasks"][0]["cluster_spec"]["new_cluster"]["single_user_name"]
    with pytest.raises(ValueError, match="status/topology drift"):
        remote._validate_successful_remote_coordinator_run(
            mutated, submit_payload=submit
        )

    mutated = copy.deepcopy(run)
    mutated["tasks"][0].pop("cluster_spec")
    mutated["tasks"][0]["node_type_id"] = "c5d.4xlarge"
    with pytest.raises(ValueError, match="topology is missing"):
        remote._validate_successful_remote_coordinator_run(
            mutated, submit_payload=submit
        )

    mutated = copy.deepcopy(run)
    mutated["tasks"][0]["spark_python_task"]["python_file"] = (
        "dbfs:/Volumes/catalog/schema/volume/runtime/substituted-runner.py"
    )
    with pytest.raises(ValueError, match="spark_python_task binding drift"):
        remote._validate_successful_remote_coordinator_run(
            mutated, submit_payload=submit
        )

    mutated = copy.deepcopy(run)
    parameters = mutated["tasks"][0]["spark_python_task"]["parameters"]
    parameters[:4] = parameters[2:4] + parameters[:2]
    with pytest.raises(ValueError, match="spark_python_task binding drift"):
        remote._validate_successful_remote_coordinator_run(
            mutated, submit_payload=submit
        )


def test_consumer_collection_requires_authenticated_ready_tree_absence(
    tmp_path, monkeypatch
):
    case = _request_case(
        action="consumer_evidence",
        phase_lease_root=tmp_path / "phase-leases",
    )
    submit = remote.render_full_score_remote_coordinator_submit_payload(
        _coordinator_config(),
        _request_uri(case),
        case.request,
    )
    controller_root = _persist_submission(
        tmp_path,
        monkeypatch,
        case=case,
        submit_payload=submit,
    )
    result, attestation = _result_and_attestation(case)
    controller_snapshot = {
        path.relative_to(controller_root): path.read_bytes()
        for path in controller_root.rglob("*")
        if path.is_file()
    }
    external_calls = []
    rejected_cas = remote.FullScoreCompactArtifactCAS(tmp_path / "rejected-cas")
    rejected_cas_snapshot = {
        path.relative_to(rejected_cas.root): path.read_bytes()
        for path in rejected_cas.root.rglob("*")
        if path.is_file()
    }

    def reject_current_user(_workspace, *, expected_user_name, opener=None):
        assert expected_user_name == "researcher@example.com"
        assert opener is None
        raise ValueError("Databricks current-user identity differs")

    with monkeypatch.context() as identity_patch:
        identity_patch.setattr(
            remote,
            "require_databricks_current_user_name",
            reject_current_user,
        )
        identity_patch.setattr(
            remote,
            "get_databricks_run",
            lambda *_args: external_calls.append("runs-get"),
        )
        identity_patch.setattr(
            remote,
            "download_databricks_volume_file_bytes",
            lambda *_args, **_kwargs: external_calls.append("download"),
        )
        identity_patch.setattr(
            remote,
            "list_databricks_volume_directory",
            lambda *_args, **_kwargs: external_calls.append("list"),
        )
        with pytest.raises(ValueError, match="current-user identity differs"):
            remote.collect_full_score_remote_coordinator(
                DatabricksWorkspaceConfig("https://dbc.example", "secret"),
                controller_root=controller_root,
                cas=rejected_cas,
                request_authorization=case.request,
            )
    assert external_calls == []
    assert {
        path.relative_to(controller_root): path.read_bytes()
        for path in controller_root.rglob("*")
        if path.is_file()
    } == controller_snapshot
    assert {
        path.relative_to(rejected_cas.root): path.read_bytes()
        for path in rejected_cas.root.rglob("*")
        if path.is_file()
    } == rejected_cas_snapshot
    monkeypatch.setattr(
        remote, "get_databricks_run", lambda *_args: _successful_run(submit)
    )
    monkeypatch.setattr(
        remote,
        "download_databricks_volume_file_bytes",
        lambda _workspace, uri, **_kwargs: {
            case.request["result_uri"]: remote._pretty_json_bytes(result),
            case.request["attestation_uri"]: remote._pretty_json_bytes(attestation),
            **case.consumer_artifacts,
        }[uri],
    )
    monkeypatch.setattr(
        remote,
        "list_databricks_volume_directory",
        lambda _workspace, uri, **_kwargs: (
            {
                "is_directory": True,
                "name": "leftover-q8",
                "path": uri.removeprefix("dbfs:") + "/leftover-q8/",
            },
        ),
    )

    with pytest.raises(ValueError, match="metadata corroboration"):
        remote.collect_full_score_remote_coordinator(
            DatabricksWorkspaceConfig("https://dbc.example", "secret"),
            controller_root=controller_root,
            cas=remote.FullScoreCompactArtifactCAS(tmp_path / "cas"),
            request_authorization=case.request,
        )
    monkeypatch.setattr(
        remote,
        "list_databricks_volume_directory",
        lambda _workspace, uri, **_kwargs: (),
    )
    authority = remote.collect_full_score_remote_coordinator(
        DatabricksWorkspaceConfig("https://dbc.example", "secret"),
        controller_root=controller_root,
        cas=remote.FullScoreCompactArtifactCAS(tmp_path / "cas"),
        request_authorization=case.request,
    )
    assert authority.action == "consumer_evidence"
    assert len(authority.evidence_bindings) == 16
    assert all(
        "/dbfs" not in binding["evidence_uri"]
        for binding in authority.evidence_bindings
    )
    with monkeypatch.context() as identity_patch:
        identity_patch.setattr(
            remote,
            "require_databricks_current_user_name",
            reject_current_user,
        )
        with pytest.raises(ValueError, match="current-user identity differs"):
            remote.replay_full_score_remote_coordinator_authorization(
                DatabricksWorkspaceConfig("https://dbc.example", "secret"),
                controller_root=controller_root,
                cas=remote.FullScoreCompactArtifactCAS(tmp_path / "cas"),
                request_authorization=case.request,
            )


def test_final_coverage_requires_all_ten_waves_and_160_unique_shards(monkeypatch):
    waves = []
    attestations = []
    for wave_index in range(10):
        shard_ids = [f"shard-{wave_index * 16 + offset:03d}" for offset in range(16)]
        waves.append({"shard_ids": shard_ids})
    execution = _closed_record("publication-execution", waves=waves)
    for wave_index, wave in enumerate(waves):
        attestations.append(
            _close(
                {
                    "action": "consumer_evidence",
                    "closed_record_sha256": "",
                    "execution_plan_sha256": execution["closed_record_sha256"],
                    "record_type": remote.FULL_SCORE_REMOTE_COORDINATOR_ATTESTATION_RECORD_TYPE,
                    "shard_ids": wave["shard_ids"],
                    "wave_index": wave_index,
                }
            )
        )

    with pytest.raises(
        TypeError,
        match="publication workflow requires remote consumer authority",
    ):
        remote.build_full_score_remote_final_coverage_record(
            execution,
            attestations,
        )


def test_consumer_authority_requires_exact_ten_wave_160_shard_coverage():
    waves = []
    for wave_index in range(10):
        shard_ids = [f"shard-{wave_index * 16 + offset:03d}" for offset in range(16)]
        waves.append({"shard_ids": shard_ids})
    execution = _closed_record("publication-execution", waves=waves)
    durable_root = "dbfs:/Volumes/catalog/schema/volume/durable"
    authorizations = []
    for wave_index, wave in enumerate(waves):
        result = _close(
            {
                "authorization_scope": "publication",
                "closed_record_sha256": "",
                "deletion_attestation_sha256": [
                    _digest(f"deletion-{shard_id}") for shard_id in wave["shard_ids"]
                ],
                "execution_plan_sha256": execution["closed_record_sha256"],
                "next_wave_authorized": True,
                "record_type": remote.full_score.FULL_SCORE_WAVE_COMPLETION_RECORD_TYPE,
                "schema_version": 1,
                "shard_ids": wave["shard_ids"],
                "wave_index": wave_index,
            }
        )
        bindings = []
        for shard_id in wave["shard_ids"]:
            bindings.append(
                {
                    "deletion_file_sha256": _digest(f"deletion-file-{shard_id}"),
                    "deletion_record_sha256": _digest(f"deletion-{shard_id}"),
                    "deletion_uri": remote._consumer_evidence_artifact_uri(
                        durable_root,
                        wave_index=wave_index,
                        shard_id=shard_id,
                        filename="deletion-attestation.json",
                    ),
                    "evidence_file_sha256": _digest(f"evidence-file-{shard_id}"),
                    "evidence_record_sha256": _digest(f"evidence-{shard_id}"),
                    "evidence_uri": remote._consumer_evidence_artifact_uri(
                        durable_root,
                        wave_index=wave_index,
                        shard_id=shard_id,
                        filename="evidence.json",
                    ),
                    "shard_id": shard_id,
                }
            )
        authorizations.append(
            remote.FullScoreRemoteTreeAuthorization(
                action="consumer_evidence",
                execution_plan_sha256=execution["closed_record_sha256"],
                wave_index=wave_index,
                durable_output_root=durable_root,
                request_sha256=_digest(f"request-{wave_index}"),
                result_uri=f"{durable_root}/control/result-{wave_index}.json",
                result_file_sha256=_digest(f"result-file-{wave_index}"),
                result_record_sha256=result["closed_record_sha256"],
                result_record=result,
                attestation_uri=(
                    f"{durable_root}/control/attestation-{wave_index}.json"
                ),
                attestation_file_sha256=_digest(f"attestation-file-{wave_index}"),
                attestation_record_sha256=_digest(f"attestation-record-{wave_index}"),
                coordinator_run_id=str(700 + wave_index),
                coordinator_run_record_sha256=_digest(f"run-{wave_index}"),
                controller_authorization_record_sha256=_digest(
                    f"authorization-{wave_index}"
                ),
                runs_get_receipt_record_sha256=_digest(f"receipt-{wave_index}"),
                phase_terminal_record_sha256=_digest(f"terminal-{wave_index}"),
                    evidence_bindings=bindings,
                    workspace_host_sha256=_digest("https://dbc.example"),
                    user_name_sha256=_digest("researcher@example.com"),
                    _issuer=remote._REMOTE_AUTHORIZATION_ISSUER,
            )
        )

    required = remote.require_full_score_remote_consumer_evidence_authorizations(
        authorizations,
        execution_plan=execution,
    )
    assert len(required) == 10
    assert sum(len(item.evidence_bindings) for item in required) == 160
    coverage = remote.build_full_score_remote_final_coverage_record(
        execution,
        authorizations,
    )
    assert coverage["wave_count"] == 10
    assert coverage["shard_count"] == 160
    assert coverage["attestation_sha256"] == [
        item.attestation_record_sha256 for item in authorizations
    ]
    tampered_execution = copy.deepcopy(execution)
    tampered_execution["waves"][0]["shard_ids"][0] = "tampered"
    with pytest.raises(ValueError, match="execution-plan closure drift"):
        remote.build_full_score_remote_final_coverage_record(
            tampered_execution,
            authorizations,
        )
    with pytest.raises(ValueError, match="omits an execution wave"):
        remote.build_full_score_remote_final_coverage_record(
            execution,
            authorizations[:-1],
        )
    with pytest.raises(ValueError, match="phase/wave binding drift"):
        remote.build_full_score_remote_final_coverage_record(
            execution,
            [*authorizations, authorizations[-1]],
        )
    with pytest.raises(ValueError, match="omits an execution wave"):
        remote.require_full_score_remote_consumer_evidence_authorizations(
            authorizations[:-1],
            execution_plan=execution,
        )
    with pytest.raises(ValueError, match="phase/wave binding drift"):
        remote.require_full_score_remote_consumer_evidence_authorizations(
            [*authorizations, authorizations[-1]],
            execution_plan=execution,
        )
    escaped_result = authorizations[0].result_record
    escaped_result["shard_ids"][0] = "extra-shard"
    escaped_bindings = authorizations[0].evidence_bindings
    escaped_bindings[0]["shard_id"] = "extra-shard"
    assert authorizations[0].result_record is not escaped_result
    assert authorizations[0].result_record["shard_ids"][0] == "shard-000"
    assert authorizations[0].evidence_bindings is not escaped_bindings
    assert authorizations[0].evidence_bindings[0]["shard_id"] == "shard-000"
    assert remote.require_full_score_remote_consumer_evidence_authorizations(
        authorizations,
        execution_plan=execution,
    ) == required
    mixed = authorizations[-1]
    object.__setattr__(mixed, "workspace_host_sha256", _digest("other-workspace"))
    object.__setattr__(
        mixed,
        "workspace_authority_closure_sha256",
        remote._canonical_sha256(
            {
                "controller_authorization_record_sha256": (
                    mixed.controller_authorization_record_sha256
                ),
                "user_name_sha256": mixed.user_name_sha256,
                "workspace_host_sha256": mixed.workspace_host_sha256,
            }
        ),
    )
    with pytest.raises(ValueError, match="different workspaces"):
        remote.require_full_score_remote_consumer_evidence_authorizations(
            authorizations,
            execution_plan=execution,
        )
