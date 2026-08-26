import inspect
import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

import document_kv_cache.publication_latency_execution as execution
from document_kv_cache.benchmarks import SUPPORTED_V1_DATASETS
from document_kv_cache.databricks_resource_ledger import (
    DatabricksClusterHourLedger,
    create_databricks_cluster_hour_ledger_json,
    databricks_ledger_prefix,
    reserve_databricks_run_attempt_json,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_ID,
    PUBLICATION_CAMPAIGN_LEDGER_ID,
    PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS,
    PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS,
    build_publication_campaign_plan,
    publication_campaign_plan_to_record,
)
from document_kv_cache.publication_bf16_handoff_generation import (
    PublicationBF16HandoffGenerationResult,
)
from document_kv_cache.publication_latency_handoff_generation import (
    PublicationLatencyHandoffGenerationResult,
)
from document_kv_cache.publication_inputs import (
    PublicationLatencyExample,
    build_publication_storage_block_schedule,
)


def _descriptors():
    campaign = publication_campaign_plan_to_record(
        build_publication_campaign_plan(
            PUBLICATION_CAMPAIGN_ID,
            campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
            campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
            campaign_ledger_prefix=PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
            campaign_opening_terminal_gpu_hours=(
                PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
            ),
        )
    )
    jobs = [
        *(execution._core_job_descriptor(item) for item in campaign["latency_cells"]),
        *(
            execution._auxiliary_job_descriptor(item)
            for item in campaign["auxiliary_latency_cells"]
        ),
    ]
    return execution._assign_execution_zones(jobs, seed_sha256="a" * 64)


def test_frozen_design_closes_115_jobs_matched_zones_and_randomized_waves():
    jobs = _descriptors()
    waves = execution._launch_waves(jobs, seed_sha256="a" * 64)

    assert len(jobs) == PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS == 115
    assert sum(wave["job_count"] for wave in waves) == 115
    assert max(wave["job_count"] for wave in waves) <= 16
    assert {job["zone_id"] for job in jobs} == set(
        execution.PUBLICATION_LATENCY_DATABRICKS_ZONES
    )
    by_id = {job["job_id"]: job for job in jobs}
    wave_by_job = {
        job_id: wave["wave_index"] for wave in waves for job_id in wave["job_ids"]
    }
    for job in jobs:
        if job["job_kind"] == "core":
            pair = [
                item
                for item in jobs
                if item.get("matched_pair_id") == job["matched_pair_id"]
            ]
            assert len({item["zone_id"] for item in pair}) == 1
            assert len({wave_by_job[item["job_id"]] for item in pair}) == 1
        elif job.get("setting_id") not in {
            "storage-disk",
            "storage-ram",
            "storage-uc",
        }:
            assert job["zone_id"] == by_id[job["reference_core_cell_id"]]["zone_id"]
    for block in range(1, 6):
        storage_ids = [
            job["job_id"]
            for job in jobs
            if job["deployment_block"] == block
            and job.get("setting_id") in {"storage-disk", "storage-ram", "storage-uc"}
        ]
        assert len(storage_ids) == 3
        assert len({wave_by_job[job_id] for job_id in storage_ids}) == 1
        assert {by_id[job_id]["zone_id"] for job_id in storage_ids} == {
            execution._matched_unit_zone(
                f"block-{block:02d}-storage-trio",
                seed_sha256="a" * 64,
            )
        }

        anchor = next(
            job
            for job in jobs
            if job["deployment_block"] == block
            and job.get("job_kind") == "core"
            and job.get("input_tokens") == 16_384
            and job.get("request_parallelism") == 4
            and job.get("method_id") == "vanilla_prefill"
        )
        matched_unit_ids = [
            job["job_id"]
            for job in jobs
            if job.get("matched_pair_id") == anchor["matched_pair_id"]
            or (
                job["deployment_block"] == block
                and job.get("setting_id") in {"precision-bf16", "hardware-a10g"}
            )
        ]
        assert len(matched_unit_ids) == 4
        assert len({wave_by_job[job_id] for job_id in matched_unit_ids}) == 1


def test_condition_timeouts_and_runtime_zone_are_closed():
    jobs = _descriptors()
    core_32k_c1 = next(
        item
        for item in jobs
        if item["job_kind"] == "core"
        and item["input_tokens"] == 32_768
        and item["request_parallelism"] == 1
    )
    core_8k_c1 = next(
        item
        for item in jobs
        if item["job_kind"] == "core"
        and item["input_tokens"] == 8_192
        and item["request_parallelism"] == 1
    )
    auxiliary = next(item for item in jobs if item["job_kind"] == "auxiliary")

    runtime = execution._job_runtime_policy(core_32k_c1, selected_32k_gmu=0.75)
    assert runtime["run_timeout_seconds"] == 12 * 60 * 60
    assert runtime["zone_id"] == core_32k_c1["zone_id"]
    assert runtime["availability"] == "ON_DEMAND"
    assert runtime["databricks_spark_version"] == "15.4.x-gpu-ml-scala2.12"
    assert execution._job_timeout_seconds(core_8k_c1) == 6 * 60 * 60
    assert execution._job_timeout_seconds(auxiliary) == 4 * 60 * 60


def test_submit_payload_rejects_timeout_and_zone_tampering():
    descriptor = next(
        item
        for item in _descriptors()
        if item["job_kind"] == "core"
        and item["input_tokens"] == 32_768
        and item["request_parallelism"] == 1
    )
    runtime = execution._job_runtime_policy(descriptor, selected_32k_gmu=0.75)
    job = {
        "reservation_attempt_id": "publication-latency/test",
        "runtime": runtime,
        "task_key": "latency_task",
    }
    cluster = {
        "aws_attributes": {
            "availability": runtime["availability"],
            "zone_id": runtime["zone_id"],
        },
        "data_security_mode": runtime["data_security_mode"],
        "driver_node_type_id": runtime["node_type_id"],
        "node_type_id": runtime["node_type_id"],
        "num_workers": 0,
        "spark_version": runtime["databricks_spark_version"],
    }
    parameters = [
        "--cloud-run-id",
        execution._DATABRICKS_JOB_RUN_ID_TEMPLATE,
        "--task-run-id",
        execution._DATABRICKS_TASK_RUN_ID_TEMPLATE,
        "--job-record-json",
        execution._canonical_json(job),
    ]
    payload = execution.bind_databricks_run_idempotency_token(
        {
            "run_name": "latency",
            "tasks": [
                {
                    "max_retries": 0,
                    "new_cluster": cluster,
                    "spark_python_task": {"parameters": parameters},
                    "task_key": "latency_task",
                    "timeout_seconds": runtime["run_timeout_seconds"],
                }
            ],
            "timeout_seconds": runtime["run_timeout_seconds"],
        },
        attempt_id=job["reservation_attempt_id"],
    )
    execution._validate_submit_payload(payload, job_record=job)

    payload["timeout_seconds"] = 4 * 60 * 60
    payload = execution.bind_databricks_run_idempotency_token(
        {key: value for key, value in payload.items() if key != "idempotency_token"},
        attempt_id=job["reservation_attempt_id"],
    )
    with pytest.raises(ValueError, match="run timeout"):
        execution._validate_submit_payload(payload, job_record=job)
    payload["timeout_seconds"] = runtime["run_timeout_seconds"]
    payload["tasks"][0]["new_cluster"]["aws_attributes"]["zone_id"] = "us-west-2z"
    payload = execution.bind_databricks_run_idempotency_token(
        {key: value for key, value in payload.items() if key != "idempotency_token"},
        attempt_id=job["reservation_attempt_id"],
    )
    with pytest.raises(ValueError, match="availability/zone"):
        execution._validate_submit_payload(payload, job_record=job)


def test_failed_control_plane_run_is_not_a_successful_latency_identity():
    cluster = {"node_type_id": "g6.8xlarge", "num_workers": 0}
    payload = {
        "run_name": "latency-job-a",
        "tasks": [{"new_cluster": cluster, "task_key": "latency_task_a"}],
    }
    run = {
        "end_time": 3000,
        "run_id": 101,
        "run_name": "latency-job-a",
        "run_type": "SUBMIT_RUN",
        "start_time": 1000,
        "state": {"life_cycle_state": "TERMINATED", "result_state": "FAILED"},
        "tasks": [
            {
                "attempt_number": 0,
                "cluster_instance": {"cluster_id": "cluster-a"},
                "end_time": 2500,
                "new_cluster": cluster,
                "run_id": 1001,
                "start_time": 1100,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "FAILED",
                },
                "task_key": "latency_task_a",
            }
        ],
    }

    identity = execution._validate_latency_control_plane_run(
        run,
        job_record={"task_key": "latency_task_a"},
        submit_payload=payload,
        receipt_run_id="101",
    )

    assert identity["terminal_state"] == "failed"


def test_sequential_reservations_preserve_unreserved_headroom(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="latency-ledger", cap_cluster_hours=130.0
    )
    payload = {
        "run_name": "latency",
        "tasks": [
            {
                "max_retries": 0,
                "new_cluster": {"node_type_id": "g6.8xlarge"},
                "task_key": "latency",
                "timeout_seconds": 4 * 60 * 60,
            }
        ],
        "timeout_seconds": 4 * 60 * 60,
    }
    digest = execution._submit_payload_sha256(payload)
    first_validator = execution._latency_reservation_validator(
        attempt_id="latency/first",
        payload_sha256=digest,
        timeout_seconds=4 * 60 * 60,
        ledger_path=ledger_path,
        expected_ledger_id="latency-ledger",
    )
    reserve_databricks_run_attempt_json(
        ledger_path,
        payload,
        attempt_id="latency/first",
        workload_id="publication-latency",
        reservation_validator=first_validator,
    )

    second_validator = execution._latency_reservation_validator(
        attempt_id="latency/second",
        payload_sha256=digest,
        timeout_seconds=4 * 60 * 60,
        ledger_path=ledger_path,
        expected_ledger_id="latency-ledger",
    )
    with pytest.raises(ValueError, match="124-hour"):
        reserve_databricks_run_attempt_json(
            ledger_path,
            payload,
            attempt_id="latency/second",
            workload_id="publication-latency",
            reservation_validator=second_validator,
        )


def test_latency_reservation_rejects_seventeenth_active_task(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="latency-ledger", cap_cluster_hours=1024.0
    )
    payload = {
        "run_name": "latency",
        "tasks": [
            {
                "max_retries": 0,
                "new_cluster": {"node_type_id": "g6.8xlarge"},
                "task_key": "latency",
                "timeout_seconds": 4 * 60 * 60,
            }
        ],
        "timeout_seconds": 4 * 60 * 60,
    }
    for index in range(16):
        reserve_databricks_run_attempt_json(
            ledger_path,
            payload,
            attempt_id=f"other/{index:02d}",
            workload_id="other-publication-work",
        )
    digest = execution._submit_payload_sha256(payload)
    validator = execution._latency_reservation_validator(
        attempt_id="latency/seventeenth",
        payload_sha256=digest,
        timeout_seconds=4 * 60 * 60,
        ledger_path=ledger_path,
        expected_ledger_id="latency-ledger",
    )
    with pytest.raises(ValueError, match="global 16-job"):
        reserve_databricks_run_attempt_json(
            ledger_path,
            payload,
            attempt_id="latency/seventeenth",
            workload_id="publication-latency",
            reservation_validator=validator,
        )


def test_storage_cache_policies_preserve_disk_ram_uc_semantics():
    jobs = _descriptors()
    by_setting = {
        item.get("setting_id"): execution._cache_policy(item)
        for item in jobs
        if item.get("setting_id") in {"storage-disk", "storage-ram", "storage-uc"}
    }

    assert by_setting["storage-disk"]["host_cache_state"] == "cold_eviction_required"
    assert by_setting["storage-ram"] == {
        "connector_loads": "required_exact_request_coverage",
        "host_cache_state": "prewarmed_payload_cache",
        "payload_cache": "prewarmed_16_gib_exact_hits",
        "storage_source": "ram_payload_cache",
    }
    assert by_setting["storage-uc"]["host_cache_state"] == (
        "mounted_path_evicted_backend_cache_unproven"
    )


def test_hierarchical_bootstrap_resamples_examples_within_each_dataset():
    by_block = {
        block: {
            (dataset, f"{dataset}-{index:02d}"): (math.log(2.0), math.log(2.0))
            for dataset in SUPPORTED_V1_DATASETS
            for index in range(32)
        }
        for block in range(1, 6)
    }
    strata = execution._dataset_stratified_example_identities(by_block[1])
    sample = execution._draw_dataset_stratified_example_sample(random.Random(7), strata)

    assert {
        dataset: sum(item[0] == dataset for item in sample)
        for dataset in SUPPORTED_V1_DATASETS
    } == {dataset: 32 for dataset in SUPPORTED_V1_DATASETS}
    point, lower, upper = execution._paired_hierarchical_bootstrap(
        by_block, draws=100, seed=7
    )
    assert (point, lower, upper) == pytest.approx((2.0, 2.0, 2.0))


def test_storage_bootstrap_keeps_two_identities_per_dataset_and_all_repeats():
    by_block = {
        block: {
            (dataset, f"{dataset}-{index:02d}"): (math.log(1.5),) * 32
            for dataset in SUPPORTED_V1_DATASETS
            for index in range(2)
        }
        for block in range(1, 6)
    }

    point, lower, upper = execution._paired_hierarchical_bootstrap(
        by_block, draws=100, seed=11
    )
    assert (point, lower, upper) == pytest.approx((1.5, 1.5, 1.5))


def test_production_boundaries_require_nonrecord_qualification_capability():
    for function in (
        execution.build_publication_latency_execution_plan,
        execution.render_publication_latency_job_record,
        execution.build_databricks_publication_latency_run_submit_payload,
        execution.publication_latency_submit_payloads,
        execution.submit_publication_latency_launch_wave,
        execution.resume_publication_latency_launch_wave,
        execution.collect_publication_latency_launch_wave,
        execution.collect_publication_latency_campaign,
        execution.aggregate_publication_latency_campaign,
    ):
        assert (
            "qualification_launch_authorization"
            in inspect.signature(function).parameters
        )
        assert "handoff_serving_authorization" in inspect.signature(function).parameters
        assert (
            "bf16_handoff_serving_authorization"
            in inspect.signature(function).parameters
        )
        assert "source_closure_authorization" in inspect.signature(function).parameters
        hints = get_type_hints(function)
        assert hints["handoff_serving_authorization"] is (
            execution.PublicationHandoffRemoteClosureAuthorization
        )
        assert hints["bf16_handoff_serving_authorization"] is (
            execution.PublicationHandoffRemoteClosureAuthorization
        )
        assert hints["source_closure_authorization"] is (
            execution.PublicationLatencySourceClosureAuthorization
        )

    source_signature = inspect.signature(
        execution.validate_publication_latency_execution_sources
    )
    assert "handoff_serving_authorization" in source_signature.parameters
    assert "bf16_handoff_serving_authorization" in source_signature.parameters
    assert "source_closure_authorization" in source_signature.parameters
    for function in (
        execution.collect_publication_latency_launch_wave,
        execution.collect_publication_latency_campaign,
    ):
        assert "controller_cas_root" in inspect.signature(function).parameters

    with pytest.raises(TypeError, match="PublicationLatencyCollectionAuthorization"):
        execution.aggregate_publication_latency_campaign(  # type: ignore[arg-type]
            {},
            execution_plan_record={},
            qualification_launch_authorization=None,  # type: ignore[arg-type]
            handoff_serving_authorization=None,  # type: ignore[arg-type]
            bf16_handoff_serving_authorization=None,  # type: ignore[arg-type]
            source_closure_authorization=None,  # type: ignore[arg-type]
        )


def test_wave_zero_lost_response_resume_collects_then_authorizes_wave_one(
    tmp_path, monkeypatch
):
    ledger_path = tmp_path / "ledger.json"
    opening_ledger = create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="campaign"
    )
    opening_prefix = databricks_ledger_prefix(opening_ledger)
    plan_sha256 = "a" * 64
    result_paths = {
        "job-a": tmp_path / "job-a-result.json",
        "job-b": tmp_path / "job-b-result.json",
    }
    jobs = {
        job_id: {
            "output": {"result_uri": str(result_paths[job_id])},
            "reservation_attempt_id": f"latency/{job_id}",
            "runtime": {"run_timeout_seconds": 3600},
            "task_key": f"task_{job_id[-1]}",
        }
        for job_id in result_paths
    }

    def payload(job_id):
        job = jobs[job_id]
        return execution.bind_databricks_run_idempotency_token(
            {
                "run_name": job_id,
                "tasks": [
                    {
                        "max_retries": 0,
                        "new_cluster": {"node_type_id": "g6.8xlarge"},
                        "task_key": job["task_key"],
                        "timeout_seconds": 3600,
                    }
                ],
                "timeout_seconds": 3600,
            },
            attempt_id=job["reservation_attempt_id"],
        )

    plan = {
        "closed_record_sha256": plan_sha256,
        "launch_waves": [
            {"job_ids": ["job-a"]},
            {"job_ids": ["job-b"]},
        ],
        "sources": {
            "campaign_ledger_id": "campaign",
            "campaign_ledger_path_sha256": execution.databricks_ledger_path_sha256(
                ledger_path
            ),
        },
    }
    monkeypatch.setattr(
        execution, "_require_latency_launch_authorization", lambda *a: None
    )
    monkeypatch.setattr(
        execution,
        "validate_publication_latency_execution_sources",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        execution,
        "_render_publication_latency_job_record",
        lambda _plan, job_id: jobs[job_id],
    )
    monkeypatch.setattr(
        execution,
        "_build_databricks_publication_latency_run_submit_payload",
        lambda _plan, job_id: payload(job_id),
    )
    monkeypatch.setattr(execution, "_validate_submit_payload", lambda *a, **k: None)
    monkeypatch.setattr(
        execution, "_require_prior_waves_succeeded", lambda *a, **k: None
    )
    bf16_authorization = SimpleNamespace(ledger_prefix=opening_prefix)
    source_authorization = SimpleNamespace(ledger_prefix=opening_prefix)
    qualification_authorization = object()
    q8_authorization = object()

    class Response:
        status = 200

        def __init__(self, run_id):
            self._body = json.dumps({"run_id": run_id}).encode()
            self._offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, amt=-1):
            if amt < 0:
                amt = len(self._body) - self._offset
            end = min(self._offset + amt, len(self._body))
            chunk = self._body[self._offset : end]
            self._offset = end
            return chunk

    with pytest.raises(TimeoutError, match="lost response"):
        execution.submit_publication_latency_launch_wave(
            execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
            execution_plan_record=plan,
            qualification_launch_authorization=qualification_authorization,
            handoff_serving_authorization=q8_authorization,
            bf16_handoff_serving_authorization=bf16_authorization,
            source_closure_authorization=source_authorization,
            ledger_path=ledger_path,
            wave_index=0,
            phase_lease_root=tmp_path / "wave-0",
            opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                TimeoutError("lost response")
            ),
        )
    (tmp_path / "wave-0" / "batch-reserved.json").unlink()
    submission, submission_authorization = (
        execution.resume_publication_latency_launch_wave(
            execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
            execution_plan_record=plan,
            qualification_launch_authorization=qualification_authorization,
            handoff_serving_authorization=q8_authorization,
            bf16_handoff_serving_authorization=bf16_authorization,
            source_closure_authorization=source_authorization,
            ledger_path=ledger_path,
            wave_index=0,
            phase_lease_root=tmp_path / "wave-0",
            opener=lambda *_args, **_kwargs: Response(101),
        )
    )
    assert submission["jobs"][0]["run_id"] == "101"
    assert (tmp_path / "wave-0" / "batch-reserved.json").is_file()

    result_paths["job-a"].write_text(
        json.dumps(
            {
                "closed_record_sha256": "b" * 64,
                "task_identity": {"cloud_run_id": "101", "task_run_id": "1001"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    run = {
        "end_time": 3000,
        "run_id": 101,
        "start_time": 1000,
        "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
        "tasks": [
            {
                "end_time": 2100,
                "run_id": 1001,
                "start_time": 1100,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "SUCCESS",
                },
                "task_key": "task_a",
            }
        ],
    }
    monkeypatch.setattr(execution, "get_databricks_run", lambda *_args: run)
    monkeypatch.setattr(
        execution,
        "_validate_latency_control_plane_run",
        lambda *a, **k: {
            "cluster_id": "cluster-a",
            "task_run_id": "1001",
            "terminal_state": "succeeded",
        },
    )
    monkeypatch.setattr(
        execution,
        "validate_publication_latency_job_result_record",
        lambda *a, **k: None,
    )
    result_record = json.loads(result_paths["job-a"].read_text(encoding="utf-8"))
    monkeypatch.setattr(
        execution,
        "_collect_remote_publication_latency_result",
        lambda *_args, **_kwargs: execution._PublicationLatencyRemoteResult(
            record=result_record,
            result_tree={
                "directory_uri": "dbfs:/Volumes/catalog/schema/volume/job-a",
                "file_count": 1,
                "files": [],
                "total_bytes": 1,
            },
            result_file_sha256="c" * 64,
            result_file_byte_count=1,
            result_tree_sha256="d" * 64,
            result_tree_file_count=1,
            result_tree_total_bytes=1,
        ),
    )
    _terminal_record, wave_zero_authorization = (
        execution.collect_publication_latency_launch_wave(
            execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
            execution_plan_record=plan,
            qualification_launch_authorization=qualification_authorization,
            handoff_serving_authorization=q8_authorization,
            bf16_handoff_serving_authorization=bf16_authorization,
            source_closure_authorization=source_authorization,
            ledger_path=ledger_path,
            wave_index=0,
            submission_authorization=submission_authorization,
            controller_cas_root=tmp_path / "cas",
        )
    )
    wave_one, _wave_one_submission_authorization = (
        execution.submit_publication_latency_launch_wave(
            execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
            execution_plan_record=plan,
            qualification_launch_authorization=qualification_authorization,
            handoff_serving_authorization=q8_authorization,
            bf16_handoff_serving_authorization=bf16_authorization,
            source_closure_authorization=source_authorization,
            ledger_path=ledger_path,
            wave_index=1,
            phase_lease_root=tmp_path / "wave-1",
            prior_wave_authorization=wave_zero_authorization,
            opener=lambda *_args, **_kwargs: Response(102),
        )
    )
    assert wave_one["jobs"][0]["run_id"] == "102"


def test_symlink_ancestor_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink ancestor"):
        execution._reject_existing_symlink_ancestors(linked / "result.json", "result")


def _remote_result_case(monkeypatch, *, listing_mode="exact", tamper_role=None):
    directory_uri = "dbfs:/Volumes/catalog/schema/volume/latency/job-a"
    result_uri = f"{directory_uri}/{execution.PUBLICATION_LATENCY_RESULT_FILENAME}"
    artifact_bytes = {
        "benchmark": b'{"benchmark":true}\n',
        "metadata": b'{"metadata":true}\n',
    }
    files = [
        {
            "role": role,
            "sha256": execution.sha256(raw).hexdigest(),
            "uri": f"{directory_uri}/{role}.json",
        }
        for role, raw in artifact_bytes.items()
    ]
    result = {"files": files, "job_id": "job-a"}
    result_bytes = (execution._canonical_json(result) + "\n").encode()
    remote = {result_uri: result_bytes}
    remote.update(
        {
            item["uri"]: (
                b"tampered\n" if item["role"] == tamper_role else artifact_bytes[item["role"]]
            )
            for item in files
        }
    )
    listing = [
        {
            "file_size": len(raw),
            "is_directory": False,
            "path": uri.removeprefix("dbfs:"),
        }
        for uri, raw in remote.items()
    ]
    if listing_mode == "missing":
        listing.pop()
    elif listing_mode == "extra":
        listing.append(
            {
                "file_size": 1,
                "is_directory": False,
                "path": directory_uri.removeprefix("dbfs:") + "/extra.json",
            }
        )
    elif listing_mode == "known_auxiliary":
        listing.append(
            {
                "file_size": 123,
                "is_directory": False,
                "path": directory_uri.removeprefix("dbfs:") + "/vllm-server.log",
            }
        )
    monkeypatch.setattr(
        execution,
        "download_databricks_volume_file_bytes",
        lambda _config, uri, **_kwargs: remote[uri],
    )
    monkeypatch.setattr(
        execution,
        "list_databricks_volume_directory",
        lambda *_args, **_kwargs: tuple(listing),
    )
    monkeypatch.setattr(
        execution,
        "validate_publication_latency_job_result_record",
        lambda *_args, **_kwargs: None,
    )
    return (
        {"job_id": "job-a", "output": {"directory_uri": directory_uri, "result_uri": result_uri}},
        result_bytes,
        artifact_bytes,
    )


def _source_closure_records():
    digest = "a" * 64
    files = [
        execution.PublicationLatencyArtifactFile(
            role=role,
            sha256={
                "runner": execution.PUBLICATION_LATENCY_RUNNER_SHA256,
                "patched_vllm_wheel": (
                    execution.GPU_QUALIFICATION_PATCHED_WHEEL_SHA256
                ),
                "runtime_lock": execution.VLLM_RUNTIME_LOCK_SHA256,
            }.get(role, digest),
            uri=f"dbfs:/Volumes/catalog/schema/volume/artifacts/{role}",
        )
        for role in execution._final_artifact_roles()
    ]
    final_artifacts = execution.PublicationLatencyFinalArtifactPins(
        source_revision="deadbeef",
        files=tuple(files),
        output_root_uri="dbfs:/Volumes/catalog/schema/volume/results",
        handoff_generation_root_uri="dbfs:/Volumes/catalog/schema/volume/q8",
        bf16_handoff_generation_root_uri="dbfs:/Volumes/catalog/schema/volume/bf16",
        bf16_handoff_source_root_uri=(
            "dbfs:/Volumes/catalog/schema/volume/bf16/bundle"
        ),
        uc_handoff_stage_root_uri="/Volumes/catalog/schema/volume/stage",
    )
    config = execution.PublicationLatencySourceClosureCoordinatorConfig(
        runner_python_file=(
            "dbfs:/Volumes/catalog/schema/volume/control/source-runner.py"
        ),
        package_wheel_uri=final_artifacts.file("package_wheel").uri,
        package_wheel_sha256=digest,
        runtime_lock_uri=final_artifacts.file("runtime_lock").uri,
        runtime_lock_sha256=execution.VLLM_RUNTIME_LOCK_SHA256,
        patched_vllm_wheel_uri=final_artifacts.file("patched_vllm_wheel").uri,
        patched_vllm_wheel_sha256=(
            execution.GPU_QUALIFICATION_PATCHED_WHEEL_SHA256
        ),
        request_root_uri="dbfs:/Volumes/catalog/schema/volume/control/requests",
        result_root_uri="dbfs:/Volumes/catalog/schema/volume/control/results",
        single_user_name="publication@example.com",
    )
    opening = databricks_ledger_prefix(
        DatabricksClusterHourLedger(ledger_id="campaign")
    )
    selection = {
        "attention_backend": "FLASH_ATTN",
        "generation_artifacts_sha256": "b" * 64,
        "generation_databricks_node_type_id": "g6e.4xlarge",
        "generation_hardware_id": "aws-g6e-l40s",
        "generation_prefix_tokens_per_second": 35.0,
        "gpu_memory_utilization": 0.75,
        "plan_sha256": "c" * 64,
    }
    schedules = [
        {
            "closed_record_sha256": f"{block:x}" * 64,
            "deployment_block": block,
            "requests_sha256": "d" * 64,
            "seed_sha256": "e" * 64,
        }
        for block in range(1, 6)
    ]
    storage_schedules = [
        {**item, "selection_sha256": "f" * 64} for item in schedules
    ]
    semantic = {
        "campaign_closed_record_sha256": "1" * 64,
        "qualification_evidence_closed_record_sha256": "2" * 64,
        "qualification_plan_closed_record_sha256": "3" * 64,
        "qualification_selection": selection,
        "schedules": schedules,
        "storage_inputs": {
            "closed_record_sha256": "4" * 64,
            "selection_sha256": "f" * 64,
        },
        "storage_schedules": storage_schedules,
    }

    def binding(role, closed, **extra):
        artifact = final_artifacts.file(role)
        return {
            "closed_record_sha256": closed,
            "file_sha256": artifact.sha256,
            "role": role,
            "uri": artifact.uri,
            **extra,
        }

    remote_binding = {
        "control_plane_status_sha256": "5" * 64,
        "coordinator_run_id": "100",
        "execution_closed_record_sha256": "6" * 64,
        "execution_file_sha256": digest,
        "execution_uri": final_artifacts.file("handoff_execution").uri,
        "output_root_uri": final_artifacts.handoff_generation_root_uri,
        "request_closed_record_sha256": "7" * 64,
        "result_closed_record_sha256": "8" * 64,
        "result_file_sha256": "9" * 64,
        "result_uri": "dbfs:/Volumes/catalog/schema/volume/q8/result.json",
        "stage": "q8",
    }
    request = {
        "attempt_id": "latency-source-closure",
        "closed_record_sha256": "",
        "coordinator": config.to_record(),
        "expected_semantic": semantic,
        "final_artifacts": final_artifacts.to_record(),
        "handoff_closures": {
            "bf16": {
                **remote_binding,
                "execution_uri": final_artifacts.file("bf16_handoff_execution").uri,
                "output_root_uri": final_artifacts.bf16_handoff_generation_root_uri,
                "result_uri": "dbfs:/Volumes/catalog/schema/volume/bf16/result.json",
                "stage": "bf16",
            },
            "q8": remote_binding,
        },
        "input_bundle_sha256": "0" * 64,
        "ledger_lineage": {
            "ledger_id": "campaign",
            "ledger_path_sha256": "a" * 64,
            "predecessor_prefix": opening.to_record(),
        },
        "qualification_artifact_pins": {
            "cachet_source_tree_sha256": "b" * 64,
            "input_bundle_sha256": "0" * 64,
            "package_wheel_sha256": digest,
            "patched_vllm_wheel_sha256": (
                execution.GPU_QUALIFICATION_PATCHED_WHEEL_SHA256
            ),
            "runner_sha256": "d" * 64,
            "runtime_lock_sha256": execution.VLLM_RUNTIME_LOCK_SHA256,
        },
        "record_bindings": {
            "campaign": binding("campaign_plan", "1" * 64),
            "qualification_evidence": binding(
                "qualification_evidence", "2" * 64
            ),
            "qualification_plan": binding("qualification_plan", "3" * 64),
            "schedules": [
                binding(
                    f"schedule_block_{block:02d}",
                    schedules[block - 1]["closed_record_sha256"],
                    deployment_block=block,
                    requests_sha256="d" * 64,
                    seed_sha256="e" * 64,
                )
                for block in range(1, 6)
            ],
            "storage_inputs": binding("storage_inputs", "4" * 64),
            "storage_schedules": [
                binding(
                    f"storage_schedule_block_{block:02d}",
                    storage_schedules[block - 1]["closed_record_sha256"],
                    deployment_block=block,
                    requests_sha256="d" * 64,
                    seed_sha256="e" * 64,
                    selection_sha256="f" * 64,
                )
                for block in range(1, 6)
            ],
        },
        "record_type": execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_REQUEST_RECORD_TYPE,
        "request_uri": (
            "dbfs:/Volumes/catalog/schema/volume/control/requests/id/request.json"
        ),
        "result_uri": (
            "dbfs:/Volumes/catalog/schema/volume/control/results/id/result.json"
        ),
        "schema_version": execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_SCHEMA_VERSION,
    }
    singleton_identity_sha256 = (
        execution._source_closure_singleton_identity_from_request(request)
    )
    request["singleton_identity_sha256"] = singleton_identity_sha256
    request["attempt_id"] = (
        execution._publication_latency_source_closure_attempt_id(
            singleton_identity_sha256
        )
    )
    request_root, result_root = (
        execution.publication_latency_source_closure_control_roots(
            final_artifacts.handoff_generation_root_uri,
            final_artifacts.bf16_handoff_generation_root_uri,
        )
    )
    request["coordinator"]["request_root_uri"] = request_root
    request["coordinator"]["result_root_uri"] = result_root
    request["request_uri"] = execution._join_durable_uri(
        request_root, "request.json"
    )
    request["result_uri"] = execution._join_durable_uri(
        result_root, "result.json"
    )
    request["closed_record_sha256"] = execution._closed_record_sha256(request)
    result = {
        "artifacts": [
            {**item.to_record(), "byte_count": 1}
            for item in final_artifacts.files
            if item.role
            not in execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_EXCLUDED_ROLES
        ],
        "artifacts_sha256": "",
        "closed_record_sha256": "",
        "coordinator": {"run_id": "101"},
        "record_type": execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_RESULT_RECORD_TYPE,
        "request_closed_record_sha256": request["closed_record_sha256"],
        "schema_version": execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_SCHEMA_VERSION,
        "semantic": semantic,
    }
    result["artifacts_sha256"] = execution._canonical_sha256(result["artifacts"])
    result["closed_record_sha256"] = execution._closed_record_sha256(result)
    return request, result, final_artifacts


def _source_closure_request_authorization(request, *, phase_lease_root=None):
    lease_root = (
        Path.cwd()
        / ".test-latency-source-closure"
        / request["singleton_identity_sha256"][:24]
        if phase_lease_root is None
        else Path(phase_lease_root)
    )
    return execution.PublicationLatencySourceClosureRequestAuthorization(
        request=request,
        phase_lease_root=lease_root,
        _issuer=execution._SOURCE_CLOSURE_REQUEST_AUTHORIZATION_ISSUER,
    )


def _reseal_source_closure_singleton(request):
    identity = execution._source_closure_singleton_identity_from_request(request)
    request["singleton_identity_sha256"] = identity
    request["attempt_id"] = (
        execution._publication_latency_source_closure_attempt_id(identity)
    )
    closures = request["handoff_closures"]
    request_root, result_root = (
        execution.publication_latency_source_closure_control_roots(
            closures["q8"]["output_root_uri"],
            closures["bf16"]["output_root_uri"],
        )
    )
    request["coordinator"]["request_root_uri"] = request_root
    request["coordinator"]["result_root_uri"] = result_root
    request["request_uri"] = execution._join_durable_uri(
        request_root, "request.json"
    )
    request["result_uri"] = execution._join_durable_uri(result_root, "result.json")
    request["closed_record_sha256"] = execution._closed_record_sha256(request)


def test_source_closure_request_result_and_cpu_payload_are_closed(monkeypatch):
    request, result, _final_artifacts = _source_closure_records()

    monkeypatch.setenv("PIP_CONFIG_FILE", "/attacker/pip.conf")
    monkeypatch.setenv("PIP_INDEX_URL", "https://attacker.invalid/simple")
    monkeypatch.setenv("pip_no_index", "1")
    monkeypatch.setenv("PIP_REQUIREMENT", "/attacker/requirements.txt")
    monkeypatch.setenv("PYTHONHOME", "/attacker/python-home")
    monkeypatch.setenv("PYTHONPATH", "/attacker/python-path")
    monkeypatch.setenv("VIRTUAL_ENV", "/attacker/venv")
    for filename, script in (
        ("publication_latency_runner.py", execution.PUBLICATION_LATENCY_RUNNER_SCRIPT),
        (
            "publication_latency_source_closure_runner.py",
            execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT,
        ),
    ):
        namespace = {"__name__": "latency_runner_test"}
        exec(compile(script, filename, "exec"), namespace)
        environment = namespace["_pip_subprocess_environment"]()
        assert {
            key for key in environment if key.upper().startswith("PIP_")
        } == {
            "PIP_CONFIG_FILE",
            "PIP_DISABLE_PIP_VERSION_CHECK",
            "PIP_NO_INPUT",
        }
        assert environment["PIP_CONFIG_FILE"] == os.devnull
        assert environment["PYTHONNOUSERSITE"] == "1"
        assert all(
            variable not in environment
            for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")
        )
    assert '"--extra-index-url"' not in (
        execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT
    )
    execution.validate_publication_latency_source_closure_request(request)
    execution.validate_publication_latency_source_closure_result(
        result, request=request
    )
    payload = execution.render_publication_latency_source_closure_submit_payload(
        _source_closure_request_authorization(request)
    )
    assert len(payload["tasks"]) == 1
    task = payload["tasks"][0]
    assert task["max_retries"] == 0
    assert task["new_cluster"]["node_type_id"] == "c5d.4xlarge"
    assert task["new_cluster"]["driver_node_type_id"] == "c5d.4xlarge"
    assert task["new_cluster"]["num_workers"] == 0
    assert task["new_cluster"]["spark_version"] == "15.4.x-cpu-ml-scala2.12"
    assert task["new_cluster"]["custom_tags"]["ResourceClass"] == "SingleNode"
    assert task["timeout_seconds"] == 2 * 60 * 60
    assert payload["timeout_seconds"] == 2 * 60 * 60
    parameters = task["spark_python_task"]["parameters"]
    assert parameters.count("--runner-sha256") == 1
    assert parameters.count("--runtime-lock-sha256") == 1
    assert parameters.count("--patched-vllm-wheel-sha256") == 1
    assert len(result["artifacts"]) == (
        len(execution._final_artifact_roles())
        - len(execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_EXCLUDED_ROLES)
    )


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_source_closure_result_rejects_missing_or_tampered_artifact(mutation):
    request, result, _final_artifacts = _source_closure_records()
    result = deepcopy(result)
    if mutation == "missing":
        result["artifacts"].pop()
    else:
        result["artifacts"][0]["sha256"] = "0" * 64
    result["artifacts_sha256"] = execution._canonical_sha256(result["artifacts"])
    result["closed_record_sha256"] = execution._closed_record_sha256(result)

    with pytest.raises(ValueError, match="artifact"):
        execution.validate_publication_latency_source_closure_result(
            result, request=request
        )


def test_source_closure_authorization_is_issuer_only():
    request, result, _final_artifacts = _source_closure_records()
    prefix = databricks_ledger_prefix(
        DatabricksClusterHourLedger(ledger_id="campaign")
    )

    assert get_type_hints(
        execution.build_publication_latency_source_closure_request
    )["return"] is execution.PublicationLatencySourceClosureRequestAuthorization
    with pytest.raises(TypeError, match="request authority is issuer-only"):
        execution.PublicationLatencySourceClosureRequestAuthorization(
            request=request,
            phase_lease_root=Path.cwd() / ".never-authorized-source-closure",
            _issuer=object(),
        )
    with pytest.raises(TypeError, match="collector issuer"):
        execution.PublicationLatencySourceClosureAuthorization(
            request=request,
            result=result,
            result_file_sha256=execution.sha256(
                execution._canonical_json_bytes(result)
            ).hexdigest(),
            coordinator_run_id="101",
            control_plane_status_sha256="a" * 64,
            ledger_prefix=prefix,
            _issuer=object(),
        )


def test_source_closure_raw_and_resealed_swapped_package_fail_before_side_effects(
    tmp_path, monkeypatch
):
    request, _result, _final_artifacts = _source_closure_records()
    swapped = deepcopy(request)
    swapped_sha256 = "b" * 64
    swapped_uri = (
        "dbfs:/Volumes/catalog/schema/volume/artifacts/swapped-package.whl"
    )
    swapped["coordinator"]["package_wheel_sha256"] = swapped_sha256
    swapped["coordinator"]["package_wheel_uri"] = swapped_uri
    swapped["qualification_artifact_pins"]["package_wheel_sha256"] = (
        swapped_sha256
    )
    package_file = next(
        item
        for item in swapped["final_artifacts"]["files"]
        if item["role"] == "package_wheel"
    )
    package_file["sha256"] = swapped_sha256
    package_file["uri"] = swapped_uri
    _reseal_source_closure_singleton(swapped)
    execution.validate_publication_latency_source_closure_request(swapped)

    side_effects = []

    def forbidden(*_args, **_kwargs):
        side_effects.append(True)
        raise AssertionError("unauthorized request reached a side effect")

    monkeypatch.setattr(
        execution, "read_databricks_cluster_hour_ledger_json", forbidden
    )
    monkeypatch.setattr(
        execution, "upload_databricks_volume_file_bytes_exclusive", forbidden
    )
    monkeypatch.setattr(execution, "submit_databricks_run", forbidden)
    workspace = execution.DatabricksWorkspaceConfig(
        "https://dbc.example", "token"
    )
    for raw_request in (request, swapped):
        with pytest.raises(
            TypeError,
            match="PublicationLatencySourceClosureRequestAuthorization",
        ):
            execution.render_publication_latency_source_closure_submit_payload(
                raw_request  # type: ignore[arg-type]
            )
        with pytest.raises(
            TypeError,
            match="PublicationLatencySourceClosureRequestAuthorization",
        ):
            execution.submit_publication_latency_source_closure(
                workspace,
                request_authorization=raw_request,  # type: ignore[arg-type]
                ledger_path=tmp_path / "never-read-ledger.json",
                phase_lease_root=tmp_path / "never-created-lease",
                opener=forbidden,
            )

    assert side_effects == []
    assert not (tmp_path / "never-created-lease").exists()


def test_source_closure_singleton_rejects_alternate_attempt_and_control_roots():
    request, _result, _final_artifacts = _source_closure_records()
    alternate_attempt = deepcopy(request)
    alternate_attempt["attempt_id"] = "latency-source-closure-alternate"
    alternate_attempt["closed_record_sha256"] = execution._closed_record_sha256(
        alternate_attempt
    )
    with pytest.raises(ValueError, match="attempt identity drift"):
        execution.validate_publication_latency_source_closure_request(
            alternate_attempt
        )

    alternate_root = deepcopy(request)
    alternate_root["coordinator"]["request_root_uri"] += "-alternate"
    alternate_root["request_uri"] = execution._join_durable_uri(
        alternate_root["coordinator"]["request_root_uri"], "request.json"
    )
    alternate_root["closed_record_sha256"] = execution._closed_record_sha256(
        alternate_root
    )
    with pytest.raises(ValueError, match="control URI/root singleton drift"):
        execution.validate_publication_latency_source_closure_request(alternate_root)


def _bind_source_closure_to_ledger(request, result, ledger_path):
    request = deepcopy(request)
    result = deepcopy(result)
    request["ledger_lineage"]["ledger_path_sha256"] = (
        execution.databricks_ledger_path_sha256(ledger_path)
    )
    _reseal_source_closure_singleton(request)
    result["request_closed_record_sha256"] = request["closed_record_sha256"]
    result["closed_record_sha256"] = execution._closed_record_sha256(result)
    return request, result


def test_source_closure_lost_response_resume_is_idempotent_and_gpu_ledger_read_only(
    tmp_path, monkeypatch
):
    ledger_path = tmp_path / "gpu-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="campaign", cap_cluster_hours=1024.0
    )
    request, result, _final_artifacts = _source_closure_records()
    request, _result = _bind_source_closure_to_ledger(
        request, result, ledger_path
    )
    lease = tmp_path / "source-lease"
    request_authorization = _source_closure_request_authorization(
        request, phase_lease_root=lease
    )
    monkeypatch.setattr(
        execution,
        "_cluster_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("source-closure controller must not resolve DBFS mounts")
        ),
    )
    ledger_before = ledger_path.read_bytes()
    request_bytes = execution._pretty_json_bytes(request)
    upload_calls = []

    def upload_request(_workspace, uri, content, **_kwargs):
        upload_calls.append((uri, content))
        return {
            "created": len(upload_calls) == 1,
            "dbfs_uri": uri,
            "file_sha256": execution.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }

    monkeypatch.setattr(
        execution,
        "upload_databricks_volume_file_bytes_exclusive",
        upload_request,
    )
    workspace = execution.DatabricksWorkspaceConfig(
        "https://dbc.example", "token"
    )
    with pytest.raises(TimeoutError, match="lost response"):
        execution.submit_publication_latency_source_closure(
            workspace,
            request_authorization=request_authorization,
            ledger_path=ledger_path,
            phase_lease_root=lease,
            opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                TimeoutError("lost response")
            ),
        )
    persisted_authorization = execution._read_latency_controller_record(
        lease / "request-authorization.json",
        "persisted source request authorization",
    )
    assert persisted_authorization == dict(
        request_authorization.authorization_record
    )
    assert persisted_authorization["closed_record_sha256"] == (
        request_authorization.authorization_record_sha256
    )

    class Response:
        status = 200

        def __init__(self):
            self._body = b'{"run_id":101}'
            self._offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, amt=-1):
            if amt < 0:
                amt = len(self._body) - self._offset
            end = min(self._offset + amt, len(self._body))
            chunk = self._body[self._offset : end]
            self._offset = end
            return chunk

    submission, authorization = execution.resume_publication_latency_source_closure(
        workspace,
        request_authorization=request_authorization,
        ledger_path=ledger_path,
        phase_lease_root=lease,
        opener=lambda *_args, **_kwargs: Response(),
    )
    replayed, replayed_authorization = (
        execution.resume_publication_latency_source_closure(
            workspace,
            request_authorization=request_authorization,
            ledger_path=ledger_path,
            phase_lease_root=lease,
            opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("durable response replay must not POST")
            ),
        )
    )

    assert ledger_path.read_bytes() == ledger_before
    assert submission == replayed
    assert authorization.run_id == replayed_authorization.run_id == "101"
    assert authorization.request_authorization_record_sha256 == (
        request_authorization.authorization_record_sha256
    )
    assert submission["request_file_byte_count"] == len(request_bytes)
    assert submission["submit_payload_byte_count"] > 0
    assert len(upload_calls) == 3

    with pytest.raises(ValueError, match="phase_lease_root differs"):
        execution.resume_publication_latency_source_closure(
            workspace,
            request_authorization=request_authorization,
            ledger_path=ledger_path,
            phase_lease_root=tmp_path / "alternate-source-lease",
            opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("alternate source lease must not POST")
            ),
        )
    assert len(upload_calls) == 3

    tampered_authorization = deepcopy(persisted_authorization)
    tampered_authorization["package_wheel_sha256"] = "f" * 64
    tampered_authorization["closed_record_sha256"] = (
        execution._closed_record_sha256(tampered_authorization)
    )
    (lease / "request-authorization.json").write_bytes(
        execution._canonical_json_bytes(tampered_authorization)
    )
    monkeypatch.setattr(
        execution,
        "read_databricks_cluster_hour_ledger_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tampered authority must fail before ledger access")
        ),
    )
    with pytest.raises(ValueError, match="differs from the replayed authority"):
        execution.resume_publication_latency_source_closure(
            workspace,
            request_authorization=request_authorization,
            ledger_path=ledger_path,
            phase_lease_root=lease,
            opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("tampered authority must fail before HTTP")
            ),
        )
    assert len(upload_calls) == 3
    assert ledger_path.read_bytes() == ledger_before

    swapped_request = deepcopy(request)
    swapped_sha256 = "b" * 64
    swapped_uri = "dbfs:/Volumes/catalog/schema/volume/swapped-package.whl"
    swapped_request["coordinator"]["package_wheel_sha256"] = swapped_sha256
    swapped_request["coordinator"]["package_wheel_uri"] = swapped_uri
    swapped_request["qualification_artifact_pins"]["package_wheel_sha256"] = (
        swapped_sha256
    )
    swapped_package = next(
        item
        for item in swapped_request["final_artifacts"]["files"]
        if item["role"] == "package_wheel"
    )
    swapped_package["sha256"] = swapped_sha256
    swapped_package["uri"] = swapped_uri
    _reseal_source_closure_singleton(swapped_request)
    swapped_authorization = execution._source_closure_request_authorization_record(
        swapped_request
    )
    (lease / "request.json").write_bytes(
        execution._canonical_json_bytes(swapped_request)
    )
    (lease / "request-authorization.json").write_bytes(
        execution._canonical_json_bytes(swapped_authorization)
    )
    with pytest.raises(ValueError, match="differs from the replayed authority"):
        execution.resume_publication_latency_source_closure(
            workspace,
            request_authorization=request_authorization,
            ledger_path=ledger_path,
            phase_lease_root=lease,
            opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("coordinated tamper must fail before HTTP")
            ),
        )
    assert len(upload_calls) == 3
    assert ledger_path.read_bytes() == ledger_before


def test_source_closure_collector_uses_direct_get_files_and_cas_without_gpu_ledger_write(
    tmp_path, monkeypatch
):
    ledger_path = tmp_path / "gpu-ledger.json"
    ledger = create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="campaign", cap_cluster_hours=1024.0
    )
    request, result, _final_artifacts = _source_closure_records()
    request, result = _bind_source_closure_to_ledger(request, result, ledger_path)
    request_authorization = _source_closure_request_authorization(request)
    payload = execution.render_publication_latency_source_closure_submit_payload(
        request_authorization
    )
    monkeypatch.setattr(
        execution,
        "_cluster_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("source-closure collector must not resolve DBFS mounts")
        ),
    )
    cluster = deepcopy(payload["tasks"][0]["new_cluster"])
    run = {
        "end_time": 3000,
        "original_attempt_run_id": 101,
        "run_id": 101,
        "run_name": payload["run_name"],
        "run_type": "SUBMIT_RUN",
        "start_time": 1000,
        "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
        "tasks": [
            {
                "attempt_number": 0,
                "cluster_instance": {"cluster_id": "cluster-source"},
                "end_time": 2500,
                "new_cluster": cluster,
                "run_id": 1001,
                "start_time": 1100,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "SUCCESS",
                },
                "task_key": "publication_latency_source_closure",
            }
        ],
    }
    get_calls = []
    monkeypatch.setattr(
        execution,
        "get_databricks_run",
        lambda _workspace, run_id: get_calls.append(run_id) or run,
    )
    remote = {
        request["request_uri"]: execution._pretty_json_bytes(request),
        request["result_uri"]: execution._canonical_json_bytes(result),
    }
    monkeypatch.setattr(
        execution,
        "download_databricks_volume_file_bytes",
        lambda _workspace, uri, **_kwargs: remote[uri],
    )
    _snapshot, canonical_payload = (
        execution.canonical_databricks_submit_payload_snapshot(payload)
    )
    submission_authorization = (
        execution.PublicationLatencySourceClosureSubmissionAuthorization(
            request_closed_record_sha256=request["closed_record_sha256"],
            request_authorization_record_sha256=(
                request_authorization.authorization_record_sha256
            ),
            ledger_path_sha256=execution.databricks_ledger_path_sha256(ledger_path),
            predecessor_prefix=execution.databricks_ledger_prefix(ledger),
            run_id="101",
            submit_payload_sha256=execution.sha256(canonical_payload).hexdigest(),
            submit_response_sha256="a" * 64,
            _issuer=execution._SOURCE_CLOSURE_SUBMISSION_AUTHORIZATION_ISSUER,
        )
    )
    ledger_before = ledger_path.read_bytes()

    authorization = execution.collect_publication_latency_source_closure(
        execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
        request=request,
        submission_authorization=submission_authorization,
        ledger_path=ledger_path,
        controller_cas_root=tmp_path / "cas",
    )

    assert get_calls == ["101"]
    assert ledger_path.read_bytes() == ledger_before
    assert authorization.ledger_prefix == execution.databricks_ledger_prefix(ledger)
    assert len(list((tmp_path / "cas" / "sha256").glob("*/*"))) == 3


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda run: run.pop("original_attempt_run_id"), "original_attempt_run_id"),
        (
            lambda run: run.__setitem__("original_attempt_run_id", 0),
            "original_attempt_run_id",
        ),
        (
            lambda run: run["tasks"][0].__setitem__("attempt_number", False),
            "task attempt zero",
        ),
        (
            lambda run: run["tasks"][0].__setitem__("run_id", 101),
            "must differ from its parent run",
        ),
        (
            lambda run: run.__setitem__("repair_history", [{"type": "REPAIR"}]),
            "repair history",
        ),
    ],
)
def test_source_closure_control_plane_requires_exact_unrepaired_attempt_zero(
    mutation, message
):
    cluster = {
        "driver_node_type_id": execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_NODE_TYPE_ID,
        "node_type_id": execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_NODE_TYPE_ID,
        "num_workers": 0,
    }
    payload = {
        "run_name": "latency-source-closure",
        "tasks": [
            {
                "new_cluster": cluster,
                "task_key": "publication_latency_source_closure",
            }
        ],
    }
    run = {
        "end_time": 3000,
        "original_attempt_run_id": 101,
        "run_id": 101,
        "run_name": payload["run_name"],
        "run_type": "SUBMIT_RUN",
        "start_time": 1000,
        "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
        "tasks": [
            {
                "attempt_number": 0,
                "cluster_instance": {"cluster_id": "cluster-source"},
                "end_time": 2500,
                "new_cluster": cluster,
                "run_id": 1001,
                "start_time": 1100,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "SUCCESS",
                },
                "task_key": "publication_latency_source_closure",
            }
        ],
    }
    mutation(run)

    with pytest.raises(ValueError, match=message):
        execution._validate_source_closure_control_plane_run(
            run,
            submit_payload=payload,
            receipt_run_id="101",
        )


def test_remote_result_collection_uses_files_api_and_idempotent_cas(
    tmp_path, monkeypatch
):
    job, result_bytes, artifact_bytes = _remote_result_case(monkeypatch)
    monkeypatch.setattr(
        execution,
        "_cluster_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("controller must not use /dbfs")
        ),
    )
    config = execution.DatabricksWorkspaceConfig("https://dbc.example", "token")
    first = execution._collect_remote_publication_latency_result(
        config,
        job_record=job,
        controller_cas_root=tmp_path / "cas",
    )
    second = execution._collect_remote_publication_latency_result(
        config,
        job_record=job,
        controller_cas_root=tmp_path / "cas",
    )

    assert first == second
    assert first.result_file_sha256 == execution.sha256(result_bytes).hexdigest()
    assert first.result_tree_file_count == 3
    assert first.result_tree_total_bytes == len(result_bytes) + sum(
        len(raw) for raw in artifact_bytes.values()
    )
    assert len(list((tmp_path / "cas" / "sha256").glob("*/*"))) == 3


def test_remote_result_collection_rejects_tampered_existing_cas_object(
    tmp_path, monkeypatch
):
    job, result_bytes, _artifact_bytes = _remote_result_case(monkeypatch)
    config = execution.DatabricksWorkspaceConfig("https://dbc.example", "token")
    execution._collect_remote_publication_latency_result(
        config,
        job_record=job,
        controller_cas_root=tmp_path / "cas",
    )
    digest = execution.sha256(result_bytes).hexdigest()
    result_object = tmp_path / "cas" / "sha256" / digest[:2] / digest
    result_object.chmod(0o640)
    result_object.write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="CAS object SHA-256 drift"):
        execution._collect_remote_publication_latency_result(
            config,
            job_record=job,
            controller_cas_root=tmp_path / "cas",
        )


def test_remote_result_collection_records_known_unsealed_worker_log(
    tmp_path, monkeypatch
):
    job, _result_bytes, _artifact_bytes = _remote_result_case(
        monkeypatch,
        listing_mode="known_auxiliary",
    )
    collected = execution._collect_remote_publication_latency_result(
        execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
        job_record=job,
        controller_cas_root=tmp_path / "cas",
    )

    assert collected.result_tree["auxiliary_files"] == [
        {
            "byte_count": 123,
            "uri": "dbfs:/Volumes/catalog/schema/volume/latency/job-a/vllm-server.log",
        }
    ]


def test_controller_plan_and_source_validation_never_translate_dbfs_mounts():
    build_names = set(
        execution.build_publication_latency_execution_plan.__code__.co_names
    )
    validation_names = set(
        execution.validate_publication_latency_execution_sources.__code__.co_names
    )

    assert "_cluster_path" not in build_names
    assert "_cluster_path" not in validation_names
    assert "require_q8_handoff_remote_closure_authorization" in validation_names
    assert "require_bf16_handoff_remote_closure_authorization" in validation_names
    assert (
        "require_publication_latency_source_closure_authorization"
        in validation_names
    )


def test_mac_boundaries_never_resolve_dbfs_and_volume_workers_use_volume_mount():
    controller_functions = (
        execution.build_publication_latency_execution_plan,
        execution.validate_publication_latency_execution_sources,
        execution.render_publication_latency_job_record,
        execution.build_databricks_publication_latency_run_submit_payload,
        execution.submit_publication_latency_launch_wave,
        execution.resume_publication_latency_launch_wave,
        execution.collect_publication_latency_launch_wave,
        execution.collect_publication_latency_campaign,
        execution.submit_publication_latency_source_closure,
        execution.resume_publication_latency_source_closure,
        execution.collect_publication_latency_source_closure,
        execution._collect_remote_publication_latency_result,
    )
    for function in controller_functions:
        assert "_cluster_path" not in function.__code__.co_names

    assert execution._cluster_path(
        "dbfs:/Volumes/catalog/schema/volume/artifact.json"
    ) == execution.Path("/Volumes/catalog/schema/volume/artifact.json")


def test_cpu_source_closure_does_not_mutate_gpu_campaign_ledger():
    controller_names = {
        name
        for function in (
            execution.submit_publication_latency_source_closure,
            execution.resume_publication_latency_source_closure,
            execution.collect_publication_latency_source_closure,
        )
        for name in function.__code__.co_names
    }

    assert not {
        "reserve_databricks_run_attempt_batch_authorized_json",
        "record_databricks_verified_run_terminal_actual_json",
        "resume_pre_reserved_databricks_run",
        "submit_pre_reserved_databricks_run",
    }.intersection(controller_names)


def test_remote_handoff_phase_order_rejects_cross_phase_prefixes():
    opening = databricks_ledger_prefix(
        DatabricksClusterHourLedger(ledger_id="campaign")
    )
    later = execution.DatabricksLedgerPrefix(
        ledger_id="campaign",
        cap_cluster_hours=1024.0,
        reservation_count=1,
        submission_receipt_count=0,
        terminal_actual_count=0,
        prefix_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="Q8 authority"):
        execution._require_remote_handoff_phase_order(
            qualification_prefix=opening,
            q8_predecessor_prefix=later,
            q8_terminal_prefix=later,
            bf16_predecessor_prefix=later,
        )
    with pytest.raises(ValueError, match="BF16 authority"):
        execution._require_remote_handoff_phase_order(
            qualification_prefix=opening,
            q8_predecessor_prefix=opening,
            q8_terminal_prefix=later,
            bf16_predecessor_prefix=opening,
        )


def test_remote_storage_input_validation_uses_closed_records_not_dbfs(
    monkeypatch,
):
    bundle_sha = "a" * 64
    examples = tuple(
        PublicationLatencyExample(dataset, f"example-{index:02d}")
        for dataset in SUPPORTED_V1_DATASETS
        for index in range(32)
    )
    schedules = {
        block: build_publication_storage_block_schedule(
            campaign_id="campaign",
            deployment_block=block,
            input_bundle_sha256=bundle_sha,
            examples=examples,
        )
        for block in range(1, 6)
    }
    selected = execution.select_publication_storage_examples(
        examples,
        input_bundle_sha256=bundle_sha,
    )
    files = []
    artifact_records = []
    for role in execution._final_artifact_roles():
        digest = "e" * 64
        if role.startswith("input_16384_"):
            dataset = role.removeprefix("input_16384_")
            digest = execution.sha256(f"source-{dataset}".encode()).hexdigest()
        elif role.startswith("storage_input_16384_"):
            dataset = role.removeprefix("storage_input_16384_")
            digest = execution.sha256(f"storage-{dataset}".encode()).hexdigest()
        artifact_records.append(
            execution.PublicationLatencyArtifactFile(
                role=role,
                sha256=digest,
                uri=f"dbfs:/Volumes/catalog/schema/volume/artifacts/{role}",
            )
        )
    final_artifacts = execution.PublicationLatencyFinalArtifactPins(
        source_revision="deadbeef",
        files=tuple(artifact_records),
        output_root_uri="dbfs:/Volumes/catalog/schema/volume/latency",
        handoff_generation_root_uri="dbfs:/Volumes/catalog/schema/volume/q8",
        bf16_handoff_generation_root_uri="dbfs:/Volumes/catalog/schema/volume/bf16",
        bf16_handoff_source_root_uri=(
            "dbfs:/Volumes/catalog/schema/volume/bf16/bundle"
        ),
        uc_handoff_stage_root_uri="/Volumes/catalog/schema/volume/stage",
    )
    for dataset in SUPPORTED_V1_DATASETS:
        output_artifact = final_artifacts.file(f"storage_input_16384_{dataset}")
        source_artifact = final_artifacts.file(f"input_16384_{dataset}")
        identities = sorted(
            item.example_id for item in selected if item.dataset == dataset
        )
        files.append(
            {
                "byte_count": 128,
                "dataset": dataset,
                "identities": identities,
                "record_count": 2,
                "rows_sha256": execution.sha256(dataset.encode()).hexdigest(),
                "sha256": output_artifact.sha256,
                "source_sha256": source_artifact.sha256,
                "uri": output_artifact.uri,
            }
        )
    selection = schedules[1]["protocol"]["selection"]
    record = {
        "closed_record_sha256": "",
        "files": files,
        "input_bundle_sha256": bundle_sha,
        "output_root": "dbfs:/Volumes/catalog/schema/volume/artifacts",
        "record_type": execution.PUBLICATION_STORAGE_INPUTS_RECORD_TYPE,
        "schedule_bindings": [
            {
                "closed_record_sha256": schedules[block]["closed_record_sha256"],
                "deployment_block": block,
                "requests_sha256": schedules[block]["requests_sha256"],
                "selection_sha256": schedules[block]["protocol"]["selection"][
                    "selection_sha256"
                ],
            }
            for block in range(1, 6)
        ],
        "schema_version": execution.PUBLICATION_STORAGE_INPUTS_SCHEMA_VERSION,
        "selection_protocol": {
            **selection,
            "examples_per_dataset": 2,
            "repeats_per_example": 32,
            "request_count": 256,
            "source_row_bytes_preserved": True,
        },
    }
    record["closed_record_sha256"] = execution._closed_record_sha256(record)
    monkeypatch.setattr(
        execution,
        "_cluster_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("storage controller must not use /dbfs")
        ),
    )

    execution._validate_remote_publication_storage_inputs_record(
        record,
        source_examples=examples,
        schedule_records=schedules,
        expected_input_bundle_sha256=bundle_sha,
        final_artifacts=final_artifacts,
    )


@pytest.mark.parametrize("listing_mode", ["missing", "extra"])
def test_remote_result_collection_rejects_missing_or_extra_outputs(
    tmp_path, monkeypatch, listing_mode
):
    job, _result_bytes, _artifact_bytes = _remote_result_case(
        monkeypatch,
        listing_mode=listing_mode,
    )
    with pytest.raises(ValueError, match="directory closure drift"):
        execution._collect_remote_publication_latency_result(
            execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
            job_record=job,
            controller_cas_root=tmp_path / "cas",
        )


def test_remote_result_collection_rejects_tampered_referenced_file(
    tmp_path, monkeypatch
):
    job, _result_bytes, _artifact_bytes = _remote_result_case(
        monkeypatch,
        tamper_role="metadata",
    )
    with pytest.raises(ValueError, match="metadata SHA-256 mismatch"):
        execution._collect_remote_publication_latency_result(
            execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
            job_record=job,
            controller_cas_root=tmp_path / "cas",
        )


def test_remote_result_collection_rejects_noncanonical_result_json(
    tmp_path, monkeypatch
):
    job, _result_bytes, _artifact_bytes = _remote_result_case(monkeypatch)
    monkeypatch.setattr(
        execution,
        "download_databricks_volume_file_bytes",
        lambda *_args, **_kwargs: b'{ "files": [], "job_id": "job-a" }\n',
    )
    with pytest.raises(ValueError, match="canonical newline JSON"):
        execution._collect_remote_publication_latency_result(
            execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
            job_record=job,
            controller_cas_root=tmp_path / "cas",
        )


def test_final_artifact_roles_include_storage_inputs_and_schedules():
    roles = execution._final_artifact_roles()

    assert "bf16_handoff_execution" in roles
    assert (
        len([role for role in roles if role.startswith("storage_schedule_block_")]) == 5
    )
    assert {
        f"storage_input_16384_{dataset}" for dataset in SUPPORTED_V1_DATASETS
    }.issubset(roles)


def test_execution_plan_rejects_structural_bf16_result_authority(tmp_path):
    hints = get_type_hints(execution.build_publication_latency_execution_plan)
    assert hints["bf16_handoff_serving_authorization"] is (
        execution.PublicationHandoffRemoteClosureAuthorization
    )
    assert hints["handoff_serving_authorization"] is (
        execution.PublicationHandoffRemoteClosureAuthorization
    )
    structural_result = PublicationBF16HandoffGenerationResult(
        root=tmp_path,
        source_root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        execution_record_path=tmp_path / "execution.json",
        manifest={},
        record={},
    )
    with pytest.raises(TypeError, match="coordinator-issued"):
        execution._validated_bf16_generation_binding(  # type: ignore[arg-type]
            structural_result,
            expected_input_bundle_sha256="a" * 64,
            expected_qualification_closed_record_sha256="b" * 64,
            final_artifacts=None,  # type: ignore[arg-type]
        )

    structural_q8_result = PublicationLatencyHandoffGenerationResult(
        root=tmp_path,
        execution_record_path=tmp_path / "execution.json",
        record={},
    )
    with pytest.raises(TypeError, match="remote handoff closure"):
        execution.require_q8_handoff_remote_closure_authorization(
            structural_q8_result,
            expected_output_root_uri=(
                "dbfs:/Volumes/catalog/schema/volume/publication/q8"
            ),
            expected_execution_file_sha256="a" * 64,
            expected_input_bundle_sha256="b" * 64,
            expected_qualification_closed_record_sha256="c" * 64,
        )


def test_reclosed_plan_cannot_rebind_handoff_authority(monkeypatch):
    file_sha = "e" * 64
    input_sha = "4" * 64
    qualification_plan_sha = "a" * 64
    qualification_evidence_sha = "b" * 64
    qualification_causal_sha = "c" * 64
    q8_causal_sha = "d" * 64
    bf16_causal_sha = "f" * 64
    q8_execution_sha = "1" * 64
    bf16_execution_sha = "2" * 64
    bf16_manifest_sha = "3" * 64
    ledger_path_sha = "9" * 64
    ledger_prefix = databricks_ledger_prefix(
        DatabricksClusterHourLedger(ledger_id="ledger")
    )
    ledger_prefix_record = ledger_prefix.to_record()
    selection = SimpleNamespace(
        attention_backend="FLASH_ATTN",
        generation_artifacts_sha256="5" * 64,
        generation_databricks_node_type_id="g6e.4xlarge",
        generation_hardware_id="aws-g6e-l40s",
        generation_prefix_tokens_per_second=35.0,
        gpu_memory_utilization=0.75,
        plan_sha256=qualification_plan_sha,
    )
    final_files = [
        {
            "role": role,
            "sha256": file_sha,
            "uri": f"dbfs:/publication/final/{role}",
        }
        for role in execution._final_artifact_roles()
    ]
    for item in final_files:
        if item["role"] == "handoff_execution":
            item["uri"] = "dbfs:/publication/q8/execution.json"
        elif item["role"] == "bf16_handoff_execution":
            item["uri"] = "dbfs:/publication/bf16/execution.json"
        elif item["role"] == "bf16_handoff_manifest":
            item["uri"] = "dbfs:/publication/bf16/manifest.json"
    plan = {
        "closed_record_sha256": "6" * 64,
        "sources": {
            "campaign_ledger_id": "ledger",
            "campaign_ledger_path_sha256": ledger_path_sha,
            "campaign_ledger_prefix": ledger_prefix_record,
            "campaign_opening_terminal_gpu_hours": 0.0,
            "bf16_handoff": {
                "accounting": {"closed_sha256": "7" * 64},
                "authorization": {
                    "causal_closure_sha256": bf16_causal_sha,
                    "ledger_id": "ledger",
                    "ledger_path_sha256": ledger_path_sha,
                    "ledger_prefix": ledger_prefix_record,
                    "predecessor_prefix": ledger_prefix_record,
                    "producer_batch_prefix": ledger_prefix_record,
                },
                "execution": {
                    "closed_record_sha256": bf16_execution_sha,
                    "sha256": file_sha,
                    "uri": "dbfs:/publication/bf16/execution.json",
                },
                "ledger_reconciliation_sha256": bf16_causal_sha,
                "manifest": {
                    "closed_record_sha256": bf16_manifest_sha,
                    "sha256": file_sha,
                    "uri": "dbfs:/publication/bf16/manifest.json",
                },
                "output_root_uri": "dbfs:/publication/bf16",
                "source_root_uri": "dbfs:/publication/bf16/bundle",
            },
            "final_artifacts": {
                "bf16_handoff_generation_root_uri": "dbfs:/publication/bf16",
                "bf16_handoff_source_root_uri": "dbfs:/publication/bf16/bundle",
                "files": final_files,
                "handoff_generation_root_uri": "dbfs:/publication/q8",
                "output_root_uri": "dbfs:/publication/results",
                "source_revision": "deadbeef",
                "uc_handoff_stage_root_uri": "/Volumes/catalog/schema/volume/stage",
            },
            "handoff_generation": {
                "accounting_sha256": "8" * 64,
                "authorization": {
                    "causal_closure_sha256": q8_causal_sha,
                    "ledger_id": "ledger",
                    "ledger_path_sha256": ledger_path_sha,
                    "ledger_prefix": ledger_prefix_record,
                    "predecessor_prefix": ledger_prefix_record,
                    "producer_batch_prefix": ledger_prefix_record,
                },
                "execution": {
                    "closed_record_sha256": q8_execution_sha,
                    "sha256": file_sha,
                    "uri": "dbfs:/publication/q8/execution.json",
                },
                "output_root_uri": "dbfs:/publication/q8",
            },
            "qualification": {
                "artifact_pins": {"input_bundle_sha256": input_sha},
                "authorization": {
                    "causal_closure_sha256": qualification_causal_sha,
                    "ledger_id": "ledger",
                    "ledger_path_sha256": ledger_path_sha,
                    "ledger_prefix": ledger_prefix_record,
                },
                "evidence": {
                    "closed_record_sha256": qualification_evidence_sha,
                    "sha256": file_sha,
                },
                "plan": {"closed_record_sha256": qualification_plan_sha},
                "selection": execution._selection_record(selection),
            },
        },
    }
    q8_result = SimpleNamespace(
        execution_record_path=execution.Path("/dbfs/publication/q8/execution.json"),
        record={"closed_record_sha256": q8_execution_sha},
        root=execution.Path("/dbfs/publication/q8"),
    )
    bf16_result = SimpleNamespace(
        execution_record_path=execution.Path("/dbfs/publication/bf16/execution.json"),
        manifest_path=execution.Path("/dbfs/publication/bf16/manifest.json"),
        record={"closed_record_sha256": bf16_execution_sha},
        root=execution.Path("/dbfs/publication/bf16"),
        source_root=execution.Path("/dbfs/publication/bf16/bundle"),
    )
    qualification_authorization = SimpleNamespace(
        causal_closure_sha256=qualification_causal_sha,
        ledger_id="ledger",
        ledger_path_sha256=ledger_path_sha,
        ledger_prefix=ledger_prefix,
    )
    q8_authorization = SimpleNamespace(
        causal_closure_sha256=q8_causal_sha,
        ledger_id="ledger",
        ledger_path_sha256=ledger_path_sha,
        ledger_prefix=ledger_prefix,
        predecessor_prefix=ledger_prefix,
        producer_batch_prefix=ledger_prefix,
    )
    bf16_authorization = SimpleNamespace(
        causal_closure_sha256=bf16_causal_sha,
        ledger_id="ledger",
        ledger_path_sha256=ledger_path_sha,
        ledger_prefix=ledger_prefix,
        predecessor_prefix=ledger_prefix,
        producer_batch_prefix=ledger_prefix,
    )
    source_authorization = SimpleNamespace(
        ledger_id="ledger",
        ledger_path_sha256=ledger_path_sha,
    )
    monkeypatch.setattr(
        execution,
        "validate_publication_latency_execution_plan_record",
        lambda _record: None,
    )
    monkeypatch.setattr(
        execution,
        "require_gpu_qualification_launch_authorization",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        execution,
        "require_q8_handoff_remote_closure_authorization",
        lambda *_args, **_kwargs: q8_result,
    )
    monkeypatch.setattr(
        execution,
        "require_bf16_handoff_remote_closure_authorization",
        lambda *_args, **_kwargs: bf16_result,
    )
    monkeypatch.setattr(execution, "_file_sha256", lambda _path: file_sha)

    forged = deepcopy(plan)
    forged["sources"]["handoff_generation"]["authorization"][
        "causal_closure_sha256"
    ] = "0" * 64
    forged["sources_sha256"] = execution._canonical_sha256(forged["sources"])
    forged["closed_record_sha256"] = execution._closed_record_sha256(forged)
    with pytest.raises(ValueError, match="Q8 handoff serving authorization binding"):
        execution._require_latency_launch_authorization(
            forged,
            qualification_authorization,  # type: ignore[arg-type]
            q8_authorization,  # type: ignore[arg-type]
            bf16_authorization,  # type: ignore[arg-type]
            source_authorization,  # type: ignore[arg-type]
        )

    cross_ledger_bf16_authorization = SimpleNamespace(
        causal_closure_sha256=bf16_causal_sha,
        ledger_id="other-ledger",
    )
    with pytest.raises(ValueError, match="share the execution plan campaign ledger"):
        execution._require_latency_launch_authorization(
            plan,
            qualification_authorization,  # type: ignore[arg-type]
            q8_authorization,  # type: ignore[arg-type]
            cross_ledger_bf16_authorization,  # type: ignore[arg-type]
            source_authorization,  # type: ignore[arg-type]
        )


def test_descriptive_cell_record_is_closed_and_retains_block_values():
    metrics = {
        "observation_count": 256,
        "configured_closed_loop_concurrency": 4,
        "p50_decode_tokens_per_second": 75.0,
        "p50_time_to_completion_seconds": 4.0,
        "p50_ttft_seconds": 1.0,
        "p95_time_to_completion_seconds": 6.0,
        "p95_ttft_seconds": 2.0,
        "peak_gpu_process_memory_bytes": 10,
        "peak_host_memory_used_bytes": 20,
        "peak_process_tree_rss_bytes": 30,
    }
    record = {
        **metrics,
        "cell_id": "auxiliary-storage-ram",
        "cell_kind": "auxiliary_pooled_five_blocks",
        "cell_sha256": "",
        "comparison_family": "storage",
        "input_tokens": 16_384,
        "method_id": "vanilla_prefill",
        "observation_count": 1280,
        "physical_blocks": [
            {
                **metrics,
                "deployment_block": block,
                "job_id": f"block-{block:02d}-storage-ram",
            }
            for block in range(1, 6)
        ],
        "quantile_method": "empirical_nearest_rank",
        "request_parallelism": 4,
        "setting_id": "storage-ram",
    }
    record["cell_sha256"] = execution._descriptive_cell_sha256(record)

    execution._validate_descriptive_cell_record(record)
    record["p95_ttft_seconds"] = 3.0
    with pytest.raises(ValueError, match="digest"):
        execution._validate_descriptive_cell_record(record)
