"""Synthetic, non-authorizing inputs for legacy UC failure reconciliation.

No retained cloud response or operator ledger is copied. The historical wire
shape, source-pinned legacy runtime/runner, public accounting producers and
all reconciliation validators remain in use.
"""

import hashlib
from pathlib import Path

import document_kv_cache.databricks_resource_ledger as ledger_api
import document_kv_cache.gpu_qualification as qualification_api
import document_kv_cache.gpu_qualification_databricks as api
import document_kv_cache.publication_campaign as campaign_api
from qualification_ledger_fixture import (
    bind_qualification_opening,
    write_opening_ledger,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path: Path, record, *, pretty: bool = False) -> None:
    content = api._canonical_stdlib_json_bytes(record, pretty=pretty)
    path.write_bytes(content if pretty else content + b"\n")


def _legacy_payloads(plan, pins):
    """Use production wire producers; validate the exact legacy shape later.

    The current public renderer intentionally refuses this retired security
    mode, so it is not monkeypatched to issue a legacy live submission.
    """
    plan_sha = plan["closed_record_sha256"]
    encoded_plan = api._encode_qualification_plan_parameter(
        qualification_api.canonical_gpu_qualification_json(plan)
    )
    uris = {
        key: f"dbfs:/unit-fixtures/legacy-uc/{key}.artifact"
        for key in api.GPU_QUALIFICATION_ARTIFACT_KEYS
    }
    payloads = []
    for job in plan["cloud_qualification"]["jobs"]:
        job_id = job["job_id"]
        attempt_id = api.gpu_qualification_reservation_attempt_id(plan_sha, job_id)
        parameters = api._runner_parameters(
            encoded_plan=encoded_plan,
            plan_digest=plan_sha,
            job_id=job_id,
            output_json=(
                f"dbfs:/unit-fixtures/legacy-uc/results/{plan_sha}/{job_id}/"
                + api.GPU_QUALIFICATION_OUTPUT_FILENAME
            ),
            work_dir=str(api._expected_local_work_dir(plan_sha, job_id)),
            runner_uri=uris["runner_sha256"],
            package_wheel_uri=uris["package_wheel_sha256"],
            patched_vllm_wheel_uri=uris["patched_vllm_wheel_sha256"],
            artifact_uris=uris,
            artifact_pins=pins,
            reservation_attempt_id=attempt_id,
        )
        payloads.append(
            api.bind_databricks_run_idempotency_token(
                {
                    "run_name": api._run_name(plan["campaign_id"], job_id),
                    "timeout_seconds": api.GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS,
                    "tasks": [
                        {
                            "task_key": api._task_key(job_id),
                            "timeout_seconds": api.GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS,
                            "max_retries": 0,
                            "new_cluster": api._legacy_uc_broken_qualification_cluster(
                                hardware_id=job["hardware_id"],
                                custom_tags={
                                    "campaign": api._safe_tag_value(
                                        plan["campaign_id"]
                                    ),
                                    "job_id": job_id,
                                    "plan_sha256": plan_sha[:32],
                                },
                            ),
                            "spark_python_task": {
                                "python_file": uris["runner_sha256"],
                                "parameters": parameters,
                            },
                        }
                    ],
                },
                attempt_id=attempt_id,
            )
        )
    return payloads


def build_legacy_uc_failure_fixture(tmp_path, monkeypatch):
    root = tmp_path / "synthetic-legacy-uc"
    root.mkdir()
    ledger_path = root / "cluster-hours.json"
    opening = write_opening_ledger(
        ledger_path,
        ledger_id=campaign_api.PUBLICATION_CAMPAIGN_LEDGER_ID,
        prior_attempt_count=138,
    )
    prefix, hours = bind_qualification_opening(monkeypatch, opening)
    pins = qualification_api.GPUQualificationArtifactPins(
        runtime_lock_sha256=api.GPU_QUALIFICATION_LEGACY_UC_RUNTIME_LOCK_SHA256,
        patched_vllm_wheel_sha256=api.GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        package_wheel_sha256=_digest("synthetic-legacy-package"),
        cachet_source_tree_sha256=_digest("synthetic-legacy-source"),
        runner_sha256=api.GPU_QUALIFICATION_LEGACY_UC_BROKEN_RUNNER_SHA256,
        input_bundle_sha256=api.GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    )
    plan = qualification_api.build_gpu_qualification_plan(
        campaign_id=campaign_api.PUBLICATION_CAMPAIGN_ID,
        campaign_record_sha256=campaign_api.PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
        campaign_ledger_id=opening.ledger_id,
        campaign_ledger_path_sha256=campaign_api.PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=prefix,
        campaign_opening_terminal_gpu_hours=hours,
        artifact_pins=pins,
    )
    # Bind only synthetic whole-record identities at the historical fixture
    # boundary. Runtime, runner, schema and exact batch-count guards stay fixed.
    monkeypatch.setattr(
        api,
        "GPU_QUALIFICATION_LEGACY_UC_FAILURE_PLAN_SHA256",
        plan["closed_record_sha256"],
    )
    api._validated_legacy_uc_failure_plan_and_pins(plan)
    payloads = _legacy_payloads(plan, pins)
    contracts = api._validated_qualification_payloads(
        plan, payloads, require_legacy_uc_broken_security_shape=True
    )
    _write(root / "gpu-qualification-plan.json", plan)
    _write(root / "submit-payloads.json", payloads)
    preflight_dir = root / "local-preflight-valid"
    preflight_dir.mkdir()
    preflight_path = preflight_dir / "local-preflight-evidence.json"
    preflight = qualification_api.build_local_preflight_evidence(
        plan_sha256=plan["closed_record_sha256"],
        completed_at_utc="2026-08-23T00:00:00Z",
        check_evidence_sha256={
            name: _digest(f"synthetic-local-preflight:{name}")
            for name in plan["local_preflight"]["check_ids"]
        },
    )
    _write(preflight_path, preflight)
    binding = api._non_authorizing_local_preflight_binding(preflight_path, plan=plan)
    submit_root = root / "submit-receipts"
    submit_root.mkdir()
    lease = api._qualification_phase_lease_record(
        plan=plan,
        ledger_path_sha256=plan["campaign_ledger_path_sha256"],
        predecessor_prefix=prefix,
        contracts=contracts,
        local_preflight_binding=binding,
    )
    _write(submit_root / api._QUALIFICATION_PHASE_LEASE_FILENAME, lease)
    ledger, authorization = (
        ledger_api.reserve_databricks_run_attempt_batch_authorized_json(
            ledger_path,
            api._qualification_batch_requests(plan, contracts),
            expected_predecessor_prefix=prefix,
        )
    )
    marker = api._qualification_batch_marker_record(
        lease_record=lease, batch_authorization=authorization
    )
    _write(submit_root / api._QUALIFICATION_BATCH_MARKER_FILENAME, marker)
    evidence_root = root / "failed-attempt-uc-volume-access"
    evidence_root.mkdir()
    runs = {}
    entries = []
    for index, contract in enumerate(contracts):
        parent_id = 90_000 + index
        ledger = ledger_api.record_databricks_run_submission_receipt_json(
            ledger_path,
            attempt_id=contract["reservation_attempt_id"],
            submit_response={"run_id": parent_id},
        )
        receipt = api._qualification_submit_receipt_record(
            plan=plan,
            contract=contract,
            ledger=ledger,
            phase_batch_record_sha256=marker["closed_record_sha256"],
            submitted_at_utc="2026-08-24T00:00:00.000000Z",
        )
        _write(submit_root / f"{contract['job_id']}.json", receipt)
        start = 1_787_533_140_000 + index * 10_000
        state = {"life_cycle_state": "INTERNAL_ERROR", "result_state": "FAILED"}
        run = {
            "run_id": parent_id,
            "run_name": contract["payload"]["run_name"],
            "run_type": "SUBMIT_RUN",
            "state": state,
            "tasks": [
                {
                    "run_id": 100_000 + index,
                    "task_key": contract["task_key"],
                    "attempt_number": 0,
                    "state": state,
                    "start_time": start,
                    "end_time": start + 60_000 + index * 1_000,
                }
            ],
        }
        path = evidence_root / f"{contract['job_id']}.runs-get.json"
        _write(path, run)
        entries.append(
            api._failed_attempt_reconciliation_entry(
                run,
                contract=contract,
                submit_receipt=receipt,
                evidence_file_sha256=api._file_sha256(path),
            )
        )
        runs[contract["job_id"]] = run
    manifest = {
        "closed_record_sha256": "",
        "entries": sorted(entries, key=lambda entry: entry["job_id"]),
        "plan_sha256": plan["closed_record_sha256"],
        "reason": api.GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_REASON,
        "record_type": api.GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_RECORD_TYPE,
        "schema_version": api.GPU_QUALIFICATION_SCHEMA_VERSION,
    }
    manifest["closed_record_sha256"] = api._canonical_json_sha256(manifest)
    api._validate_failed_attempt_reconciliation_manifest(
        manifest, plan=plan, expected_entries=entries
    )
    _write(evidence_root / "reconciliation-manifest.json", manifest, pretty=True)
    monkeypatch.setattr(
        api,
        "GPU_QUALIFICATION_LEGACY_UC_FAILURE_MANIFEST_CLOSED_RECORD_SHA256",
        manifest["closed_record_sha256"],
    )
    for contract in sorted(contracts, key=lambda item: item["job_id"]):
        ledger = ledger_api.record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id=contract["reservation_attempt_id"],
            run_record=runs[contract["job_id"]],
        )
    terminal_prefix = api._require_qualification_phase_ledger_closure(
        ledger, batch_authorization=authorization, contracts=contracts
    )
    monkeypatch.setattr(
        api,
        "GPU_QUALIFICATION_LEGACY_UC_FAILURE_TERMINAL_PREFIX_SHA256",
        terminal_prefix.prefix_sha256,
    )
    assert len(tuple(evidence_root.iterdir())) == 15
    assert (
        terminal_prefix.reservation_count,
        terminal_prefix.submission_receipt_count,
        terminal_prefix.terminal_actual_count,
    ) == (152, 14, 152)
    return root, ledger_path, terminal_prefix
