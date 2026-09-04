import inspect
import json
import math
import os
import random
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

import document_kv_cache.gpu_qualification_v2 as qualification_v2
import document_kv_cache.publication_latency_execution as execution
import document_kv_cache.vllm_smoke as vllm_smoke
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_A10G_GPU_MEMORY_UTILIZATION,
    build_gpu_qualification_system_cuda_parent_attestation,
)
from document_kv_cache.benchmarks import SUPPORTED_V1_DATASETS
from document_kv_cache.databricks_resource_ledger import (
    DatabricksClusterHourLedger,
    create_databricks_cluster_hour_ledger_json,
    databricks_ledger_prefix,
    record_databricks_run_submission_receipt_json,
    record_databricks_verified_run_terminal_actual_json,
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
from document_kv_cache.serving_env import (
    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
    GPU_RUNTIME_PYTHONWARNINGS,
)


@pytest.fixture(autouse=True)
def _bind_latency_current_user(monkeypatch):
    def bind_current_user(workspace, *, expected_user_name, opener=None):
        return {
            "authenticated": True,
            "user_name_sha256": execution.sha256(
                expected_user_name.encode("utf-8")
            ).hexdigest(),
            "workspace_host_sha256": execution.sha256(
                workspace.normalized_host.encode("utf-8")
            ).hexdigest(),
        }

    def bind_remote_closure_identity(
        workspace,
        _q8_authorization,
        _bf16_authorization,
        *,
        expected_user_name,
        opener=None,
    ):
        return execution.require_databricks_current_user_name(
            workspace,
            expected_user_name=expected_user_name,
            opener=opener,
        )

    monkeypatch.setattr(
        execution,
        "require_databricks_current_user_name",
        bind_current_user,
    )
    monkeypatch.setattr(
        execution,
        "require_publication_handoff_remote_closure_workspace_identity",
        bind_remote_closure_identity,
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


def test_reviewed_v2_constants_and_successor_verifier_use_ordered_streams(
    tmp_path,
    monkeypatch,
):
    assert (
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.reservation_count,
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.submission_receipt_count,
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.terminal_actual_count,
    ) == (236, 98, 236)
    assert PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.prefix_sha256 == (
        "07b9663e42c2dd8040f689d08fabdd6d7eefaf25f8f1decedc23af683e0011c7"
    )
    reviewed_v2_prefix = qualification_v2.GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    assert (
        reviewed_v2_prefix.reservation_count,
        reviewed_v2_prefix.submission_receipt_count,
        reviewed_v2_prefix.terminal_actual_count,
    ) == (265, 127, 265)
    assert reviewed_v2_prefix.prefix_sha256 == (
        "e3aaca37d5e01cbb5060800ef2e3e115e048fc35c7e1ae74539d0085c7b5c8e1"
    )
    assert PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS == 71.39012833333337
    assert (
        qualification_v2.GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS
        == 77.50443361111115
    )

    ledger_path = tmp_path / "ordered-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="ordered-successor", cap_cluster_hours=1024.0
    )
    payload = {
        "run_name": "ordered-successor",
        "tasks": [
            {
                "max_retries": 0,
                "new_cluster": {"node_type_id": "g6.8xlarge"},
                "task_key": "ordered_successor",
                "timeout_seconds": 3600,
            }
        ],
        "timeout_seconds": 3600,
    }

    def close_attempt(attempt_id, workload_id, *, run_id, duration_seconds):
        reserve_databricks_run_attempt_json(
            ledger_path,
            payload,
            attempt_id=attempt_id,
            workload_id=workload_id,
        )
        record_databricks_run_submission_receipt_json(
            ledger_path,
            attempt_id=attempt_id,
            submit_response={"run_id": run_id},
        )
        return record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id=attempt_id,
            run_record={
                "end_time": 1_000 + int(duration_seconds * 1_000),
                "run_id": run_id,
                "start_time": 1_000,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "SUCCESS",
                },
                "tasks": [
                    {
                        "end_time": 1_000 + int(duration_seconds * 1_000),
                        "run_id": run_id * 100,
                        "start_time": 1_000,
                        "state": {
                            "life_cycle_state": "TERMINATED",
                            "result_state": "SUCCESS",
                        },
                        "task_key": "ordered_successor",
                    }
                ],
            },
        )

    campaign_ledger = close_attempt(
        "campaign-opening",
        "campaign",
        run_id=101,
        duration_seconds=1800.0,
    )
    campaign_prefix = databricks_ledger_prefix(campaign_ledger)
    assert (
        campaign_prefix.reservation_count,
        campaign_prefix.submission_receipt_count,
        campaign_prefix.terminal_actual_count,
    ) == (1, 1, 1)
    qualification_ledger = close_attempt(
        "qualification-opening",
        "qualification",
        run_id=102,
        duration_seconds=900.0,
    )
    qualification_prefix = databricks_ledger_prefix(qualification_ledger)
    assert (
        qualification_prefix.reservation_count,
        qualification_prefix.submission_receipt_count,
        qualification_prefix.terminal_actual_count,
    ) == (2, 2, 2)
    ledger_path_sha256 = execution.databricks_ledger_path_sha256(ledger_path)
    campaign_hours = campaign_ledger.terminal_actual_cluster_hours
    qualification_hours = qualification_ledger.terminal_actual_cluster_hours
    monkeypatch.setattr(
        execution, "PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX", campaign_prefix
    )
    monkeypatch.setattr(
        execution,
        "PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS",
        campaign_hours,
    )
    monkeypatch.setattr(
        execution,
        "GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX",
        qualification_prefix,
    )
    monkeypatch.setattr(
        execution,
        "GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS",
        qualification_hours,
    )
    campaign = {
        "campaign_id": "ordered-successor",
        "campaign_ledger_id": "ordered-successor",
        "campaign_ledger_path_sha256": ledger_path_sha256,
        "campaign_ledger_prefix": campaign_prefix.to_record(),
        "campaign_opening_terminal_gpu_hours": campaign_hours,
        "closed_record_sha256": "a" * 64,
    }
    qualification = {
        "campaign_id": campaign["campaign_id"],
        "campaign_ledger_id": campaign["campaign_ledger_id"],
        "campaign_ledger_path_sha256": campaign["campaign_ledger_path_sha256"],
        "campaign_ledger_prefix": qualification_prefix.to_record(),
        "campaign_opening_terminal_gpu_hours": qualification_hours,
        "campaign_record_sha256": campaign["closed_record_sha256"],
    }

    assert (
        execution._require_reviewed_qualification_plan_campaign_successor(
            qualification_ledger,
            ledger_path=ledger_path,
            campaign_plan_record=campaign,
            qualification_plan_record=qualification,
        )
        == qualification_prefix
    )

    with pytest.raises(ValueError, match="shorter than its authorized prefix"):
        execution._require_reviewed_qualification_plan_campaign_successor(
            campaign_ledger,
            ledger_path=ledger_path,
            campaign_plan_record=campaign,
            qualification_plan_record=qualification,
        )

    rebound = deepcopy(qualification)
    rebound["campaign_ledger_prefix"] = campaign_prefix.to_record()
    with pytest.raises(ValueError, match="reviewed campaign successor authority"):
        execution._require_reviewed_qualification_plan_campaign_binding(
            campaign, rebound
        )


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

    runtime = execution._job_runtime_policy(
        core_32k_c1,
        selected_32k_gmu=0.75,
        single_user_name="publication@example.com",
    )
    assert runtime["run_timeout_seconds"] == 12 * 60 * 60
    assert runtime["zone_id"] == core_32k_c1["zone_id"]
    assert runtime["availability"] == "ON_DEMAND"
    assert runtime["data_security_mode"] == "SINGLE_USER"
    assert runtime["single_user_name"] == "publication@example.com"
    assert runtime["databricks_spark_version"] == "15.4.x-gpu-ml-scala2.12"
    assert execution._job_timeout_seconds(core_8k_c1) == 6 * 60 * 60
    assert execution._job_timeout_seconds(auxiliary) == 4 * 60 * 60

    a10g_jobs = [job for job in jobs if job.get("setting_id") == "hardware-a10g"]
    assert len(a10g_jobs) == 5
    assert {
        execution._job_runtime_policy(
            job,
            selected_32k_gmu=0.75,
            single_user_name="publication@example.com",
        )["gpu_memory_utilization"]
        for job in a10g_jobs
    } == {GPU_QUALIFICATION_A10G_GPU_MEMORY_UTILIZATION}
    l4_16k = next(
        job
        for job in jobs
        if job["job_kind"] == "core"
        and job["input_tokens"] == 16_384
        and job["request_parallelism"] == 4
    )
    assert (
        execution._job_runtime_policy(
            l4_16k,
            selected_32k_gmu=0.75,
            single_user_name="publication@example.com",
        )["gpu_memory_utilization"]
        == 0.90
    )
    assert runtime["gpu_memory_utilization"] == 0.75


def test_submit_payload_rejects_timeout_and_zone_tampering():
    descriptor = next(
        item
        for item in _descriptors()
        if item["job_kind"] == "core"
        and item["input_tokens"] == 32_768
        and item["request_parallelism"] == 1
    )
    runtime = execution._job_runtime_policy(
        descriptor,
        selected_32k_gmu=0.75,
        single_user_name="publication@example.com",
    )
    runner_uri = "dbfs:/Volumes/catalog/schema/volume/runtime/latency-runner.py"
    package_uri = "dbfs:/Volumes/catalog/schema/volume/runtime/cachet.whl"
    patched_vllm_uri = "dbfs:/Volumes/catalog/schema/volume/runtime/patched-vllm.whl"
    job = {
        "artifact_files": [
            {
                "role": "runner",
                "sha256": execution.PUBLICATION_LATENCY_RUNNER_SHA256,
                "uri": runner_uri,
            },
            {
                "role": "package_wheel",
                "sha256": "a" * 64,
                "uri": package_uri,
            },
            {
                "role": "patched_vllm_wheel",
                "sha256": execution.GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
                "uri": patched_vllm_uri,
            },
        ],
        "cell": descriptor,
        "closed_record_sha256": "f" * 64,
        "job_id": "test",
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
        "single_user_name": runtime["single_user_name"],
        "spark_env_vars": {
            execution.VLLM_PATCHED_WHEEL_SHA256_ENV: (
                execution.GPU_QUALIFICATION_PATCHED_WHEEL_SHA256
            ),
            execution.VLLM_PATCHED_WHEEL_URI_ENV: patched_vllm_uri,
            "DOCUMENT_KV_EVICT_PAGE_CACHE": "1",
        },
        "spark_version": runtime["databricks_spark_version"],
    }
    parameters = [
        "--job-record-json",
        execution._canonical_json(job),
        "--expected-job-sha256",
        job["closed_record_sha256"],
        "--runner-uri",
        runner_uri,
        "--runner-sha256",
        execution.PUBLICATION_LATENCY_RUNNER_SHA256,
        "--package-wheel-uri",
        package_uri,
        "--package-wheel-sha256",
        "a" * 64,
        "--cloud-run-id",
        execution._DATABRICKS_JOB_RUN_ID_TEMPLATE,
        "--task-run-id",
        execution._DATABRICKS_TASK_RUN_ID_TEMPLATE,
    ]
    payload = execution.bind_databricks_run_idempotency_token(
        {
            "run_name": "cachet-publication-latency-test",
            "tasks": [
                {
                    "max_retries": 0,
                    "new_cluster": cluster,
                    "spark_python_task": {
                        "parameters": parameters,
                        "python_file": runner_uri,
                    },
                    "task_key": "latency_task",
                    "timeout_seconds": runtime["run_timeout_seconds"],
                }
            ],
            "timeout_seconds": runtime["run_timeout_seconds"],
        },
        attempt_id=job["reservation_attempt_id"],
    )
    execution._validate_submit_payload(payload, job_record=job)
    assert payload["tasks"][0]["new_cluster"]["data_security_mode"] == ("SINGLE_USER")
    assert payload["tasks"][0]["new_cluster"]["single_user_name"] == (
        "publication@example.com"
    )
    spark_environment = payload["tasks"][0]["new_cluster"]["spark_env_vars"]
    assert "FLASHINFER_LOGGING_LEVEL" not in spark_environment
    assert "PYTHONWARNINGS" not in spark_environment

    mutated_warning_policy = deepcopy(payload)
    mutated_warning_policy["tasks"][0]["new_cluster"]["spark_env_vars"][
        "PYTHONWARNINGS"
    ] = GPU_RUNTIME_PYTHONWARNINGS
    mutated_warning_policy = execution.bind_databricks_run_idempotency_token(
        {
            key: value
            for key, value in mutated_warning_policy.items()
            if key != "idempotency_token"
        },
        attempt_id=job["reservation_attempt_id"],
    )
    with pytest.raises(ValueError, match="Spark environment drift"):
        execution._validate_submit_payload(
            mutated_warning_policy,
            job_record=job,
        )

    mutated_principal = deepcopy(payload)
    mutated_principal["tasks"][0]["new_cluster"]["single_user_name"] = (
        "attacker@example.com"
    )
    mutated_principal = execution.bind_databricks_run_idempotency_token(
        {
            key: value
            for key, value in mutated_principal.items()
            if key != "idempotency_token"
        },
        attempt_id=job["reservation_attempt_id"],
    )
    with pytest.raises(ValueError, match="cluster hardware drift"):
        execution._validate_submit_payload(mutated_principal, job_record=job)

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

    def rebound(value):
        return execution.bind_databricks_run_idempotency_token(
            {key: item for key, item in value.items() if key != "idempotency_token"},
            attempt_id=job["reservation_attempt_id"],
        )

    for flag, replacement in (
        ("--expected-job-sha256", "0" * 64),
        ("--runner-uri", "dbfs:/Volumes/attacker/runner.py"),
        ("--runner-sha256", "0" * 64),
        ("--package-wheel-uri", "dbfs:/Volumes/attacker/cachet.whl"),
        ("--package-wheel-sha256", "0" * 64),
    ):
        mutated = deepcopy(payload)
        mutated["tasks"][0]["new_cluster"]["aws_attributes"]["zone_id"] = runtime[
            "zone_id"
        ]
        mutated_parameters = mutated["tasks"][0]["spark_python_task"]["parameters"]
        mutated_parameters[mutated_parameters.index(flag) + 1] = replacement
        with pytest.raises(ValueError, match="runner parameter binding"):
            execution._validate_submit_payload(rebound(mutated), job_record=job)

    mutated = deepcopy(payload)
    mutated["tasks"][0]["new_cluster"]["aws_attributes"]["zone_id"] = runtime["zone_id"]
    mutated["tasks"][0]["spark_python_task"]["python_file"] = (
        "dbfs:/Volumes/attacker/substituted-runner.py"
    )
    with pytest.raises(ValueError, match="runner python_file drift"):
        execution._validate_submit_payload(rebound(mutated), job_record=job)

    mutated = deepcopy(payload)
    mutated["tasks"][0]["new_cluster"]["aws_attributes"]["zone_id"] = runtime["zone_id"]
    mutated["tasks"][0]["spark_python_task"]["parameters"].extend(
        ["--attacker-flag", "value"]
    )
    with pytest.raises(ValueError, match="runner parameter binding"):
        execution._validate_submit_payload(rebound(mutated), job_record=job)


def test_failed_control_plane_run_is_not_a_successful_latency_identity():
    cluster = {"node_type_id": "g6.8xlarge", "num_workers": 0}
    python_task = {
        "parameters": ["--runner-sha256", execution.PUBLICATION_LATENCY_RUNNER_SHA256],
        "python_file": "dbfs:/runner.py",
    }
    payload = {
        "run_name": "latency-job-a",
        "tasks": [
            {
                "new_cluster": cluster,
                "spark_python_task": python_task,
                "task_key": "latency_task_a",
            }
        ],
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
                "spark_python_task": python_task,
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

    mutated = deepcopy(run)
    mutated["tasks"][0]["spark_python_task"]["python_file"] = (
        "dbfs:/attacker/substituted-runner.py"
    )
    with pytest.raises(ValueError, match="Python task drift"):
        execution._validate_latency_control_plane_run(
            mutated,
            job_record={"task_key": "latency_task_a"},
            submit_payload=payload,
            receipt_run_id="101",
        )


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
            "source_closure": {
                "single_user_name": "publication@example.com",
            },
        },
        "runtime_policy": {"single_user_name": "publication@example.com"},
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
    principal_checks = []

    def record_current_user(config, *, expected_user_name, opener=None):
        principal_checks.append((expected_user_name, opener is not None))
        return {
            "authenticated": True,
            "user_name_sha256": execution.sha256(
                expected_user_name.encode("utf-8")
            ).hexdigest(),
            "workspace_host_sha256": execution.sha256(
                config.normalized_host.encode("utf-8")
            ).hexdigest(),
        }

    monkeypatch.setattr(
        execution,
        "require_databricks_current_user_name",
        record_current_user,
    )
    workspace_host_sha256 = execution.sha256(b"https://dbc.example").hexdigest()
    user_name_sha256 = execution.sha256(b"publication@example.com").hexdigest()
    bf16_authorization = SimpleNamespace(
        ledger_prefix=opening_prefix,
        workspace_host_sha256=workspace_host_sha256,
        user_name_sha256=user_name_sha256,
    )
    source_authorization = SimpleNamespace(
        ledger_prefix=opening_prefix,
        workspace_host_sha256=workspace_host_sha256,
        user_name_sha256=user_name_sha256,
    )
    qualification_authorization = object()
    q8_authorization = object()
    wrong_principals = []

    def reject_current_user(_config, *, expected_user_name, opener=None):
        wrong_principals.append((expected_user_name, opener))
        raise ValueError("Databricks current-user identity differs")

    def rejected_opener(*_args, **_kwargs):
        raise AssertionError("wrong-principal wave launch must not perform HTTP")

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

    ledger_before_wrong_principal = ledger_path.read_bytes()
    with monkeypatch.context() as identity_patch:
        identity_patch.setattr(
            execution,
            "require_databricks_current_user_name",
            reject_current_user,
        )
        with pytest.raises(ValueError, match="current-user identity differs"):
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
                opener=rejected_opener,
            )
    assert ledger_path.read_bytes() == ledger_before_wrong_principal
    assert not (tmp_path / "wave-0").exists()

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
    resume_ledger_snapshot = ledger_path.read_bytes()
    resume_lease_snapshot = {
        path.relative_to(tmp_path / "wave-0"): path.read_bytes()
        for path in (tmp_path / "wave-0").rglob("*")
        if path.is_file()
    }
    with monkeypatch.context() as identity_patch:
        identity_patch.setattr(
            execution,
            "require_databricks_current_user_name",
            reject_current_user,
        )
        with pytest.raises(ValueError, match="current-user identity differs"):
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
                opener=rejected_opener,
            )
    assert ledger_path.read_bytes() == resume_ledger_snapshot
    assert {
        path.relative_to(tmp_path / "wave-0"): path.read_bytes()
        for path in (tmp_path / "wave-0").rglob("*")
        if path.is_file()
    } == resume_lease_snapshot
    assert not (tmp_path / "wave-0" / "batch-reserved.json").exists()

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
    assert wave_zero_authorization.workspace_host_sha256 == workspace_host_sha256
    assert wave_zero_authorization.user_name_sha256 == user_name_sha256
    assert wave_zero_authorization.workspace_authority_closure_sha256 == (
        execution._canonical_sha256(
            {
                "causal_closure_sha256": (
                    wave_zero_authorization.causal_closure_sha256
                ),
                "user_name_sha256": user_name_sha256,
                "workspace_host_sha256": workspace_host_sha256,
            }
        )
    )

    for suffix, live_host, live_user in (
        (
            "other-host",
            "https://dbc.other",
            "publication@example.com",
        ),
        (
            "other-principal",
            "https://dbc.example",
            "other-publication@example.com",
        ),
    ):
        live_host_sha256 = execution.sha256(live_host.encode("utf-8")).hexdigest()
        live_user_sha256 = execution.sha256(live_user.encode("utf-8")).hexdigest()
        aligned_source_authorization = SimpleNamespace(
            ledger_prefix=opening_prefix,
            workspace_host_sha256=live_host_sha256,
            user_name_sha256=live_user_sha256,
        )
        aligned_bf16_authorization = SimpleNamespace(
            ledger_prefix=opening_prefix,
            workspace_host_sha256=live_host_sha256,
            user_name_sha256=live_user_sha256,
        )
        before = ledger_path.read_bytes()
        rejected_lease = tmp_path / f"wave-1-{suffix}"

        with monkeypatch.context() as identity_patch:
            identity_patch.setattr(
                execution,
                "require_databricks_current_user_name",
                lambda config, *, expected_user_name, opener=None, _user=live_user: {
                    "authenticated": True,
                    "user_name_sha256": execution.sha256(
                        _user.encode("utf-8")
                    ).hexdigest(),
                    "workspace_host_sha256": execution.sha256(
                        config.normalized_host.encode("utf-8")
                    ).hexdigest(),
                },
            )
            with pytest.raises(ValueError, match="prior-wave authority binding drift"):
                execution.submit_publication_latency_launch_wave(
                    execution.DatabricksWorkspaceConfig(live_host, "token"),
                    execution_plan_record=plan,
                    qualification_launch_authorization=qualification_authorization,
                    handoff_serving_authorization=q8_authorization,
                    bf16_handoff_serving_authorization=(aligned_bf16_authorization),
                    source_closure_authorization=(aligned_source_authorization),
                    ledger_path=ledger_path,
                    wave_index=1,
                    phase_lease_root=rejected_lease,
                    prior_wave_authorization=wave_zero_authorization,
                    opener=lambda *_args, **_kwargs: pytest.fail(
                        "cross-identity wave chain must not perform HTTP"
                    ),
                )
        assert ledger_path.read_bytes() == before
        assert not rejected_lease.exists()

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
    assert len(principal_checks) >= 3
    assert {item[0] for item in principal_checks} == {"publication@example.com"}
    assert {item[1] for item in principal_checks} == {False, True}
    assert wrong_principals == [
        ("publication@example.com", rejected_opener),
        ("publication@example.com", rejected_opener),
    ]


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
                b"tampered\n"
                if item["role"] == tamper_role
                else artifact_bytes[item["role"]]
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
        {
            "job_id": "job-a",
            "output": {"directory_uri": directory_uri, "result_uri": result_uri},
        },
        result_bytes,
        artifact_bytes,
    )


def _native_v2_runtime_attestation(*, vllm_uri, flashinfer_uri):
    closure = qualification_v2.gpu_qualification_v2_runtime_closure()
    return {
        "base_lock_distribution_count": (
            qualification_v2.VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT
        ),
        "base_lock_hash_count": qualification_v2.VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
        "base_lock_sha256": qualification_v2.VLLM_RUNTIME_BASE_LOCK_SHA256,
        "cachet_package_version": (
            qualification_v2.GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION
        ),
        "flashinfer_annotation": (
            qualification_v2.GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION
        ),
        "flashinfer_direct_url": flashinfer_uri,
        "flashinfer_import_ok": True,
        "closure_bound_flashinfer_manifest_closed_record_sha256": (
            qualification_v2.FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
        ),
        "closure_bound_flashinfer_manifest_file_sha256": (
            qualification_v2.FLASHINFER_PATCHED_MANIFEST_FILE_SHA256
        ),
        "closure_bound_vllm_manifest_file_sha256": (
            qualification_v2.VLLM_PATCHED_MANIFEST_SHA256
        ),
        "flashinfer_member_sha256": (qualification_v2.FLASHINFER_TARGET_PATCHED_SHA256),
        "flashinfer_package_version": qualification_v2.FLASHINFER_PACKAGE_VERSION,
        "flashinfer_wheel_sha256": qualification_v2.FLASHINFER_PATCHED_WHEEL_SHA256,
        "installed_distribution_count": (
            qualification_v2.GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT
        ),
        "ok": True,
        "packaged_base_lock_sha256": qualification_v2.VLLM_RUNTIME_BASE_LOCK_SHA256,
        "pip_check_ok": True,
        "runtime_closure_closed_record_sha256": (
            qualification_v2.RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
        ),
        "runtime_closure_file_sha256": (
            qualification_v2.RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
        ),
        "system_cuda_parent_attestation": (
            build_gpu_qualification_system_cuda_parent_attestation(
                distribution_root="/databricks/python/lib/python3.11/site-packages",
                libcudart_path=(
                    "/databricks/python/lib/python3.11/site-packages/"
                    "nvidia/cuda_runtime/lib/libcudart.so.12"
                ),
            )
        ),
        "unexpected_distributions": [],
        "vllm_direct_url": vllm_uri,
        "vllm_member_sha256": closure["vllm"]["member_sha256"],
        "vllm_package_version": qualification_v2.GPU_QUALIFICATION_VLLM_VERSION,
        "vllm_wheel_sha256": qualification_v2.GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        "with_flashinfer_distribution_count": (
            qualification_v2.GPU_QUALIFICATION_V2_WITH_FLASHINFER_DISTRIBUTION_COUNT
        ),
        "with_vllm_distribution_count": (
            qualification_v2.GPU_QUALIFICATION_V2_WITH_VLLM_DISTRIBUTION_COUNT
        ),
    }


def _native_v2_job_record(*, method_id="baseline_prefill"):
    cell = next(
        item
        for item in _descriptors()
        if item["method_id"] == method_id
        and item["input_tokens"] == 8_192
        and item["request_parallelism"] == 1
    )
    runtime = execution._job_runtime_policy(
        cell,
        selected_32k_gmu=0.75,
        single_user_name="publication@example.com",
    )
    artifact_sha256 = {
        "runner": execution.PUBLICATION_LATENCY_RUNNER_SHA256,
        "package_wheel": "a" * 64,
        "patched_vllm_wheel": execution.GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        "patched_flashinfer_wheel": execution.FLASHINFER_PATCHED_WHEEL_SHA256,
        "runtime_closure_manifest": (execution.RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256),
        "runtime_lock": execution.VLLM_RUNTIME_BASE_LOCK_SHA256,
    }
    return {
        "artifact_files": [
            {
                "role": role,
                "sha256": digest,
                "uri": f"dbfs:/Volumes/catalog/schema/volume/runtime/{role}",
            }
            for role, digest in artifact_sha256.items()
        ],
        "cache_telemetry_policy": {
            "host_cache_state": "not_applicable",
            "storage_source": "not_applicable",
        },
        "campaign_id": PUBLICATION_CAMPAIGN_ID,
        "cell": cell,
        "execution_plan_sha256": "c" * 64,
        "handoff": (
            None
            if method_id == "baseline_prefill"
            else {
                "execution": {
                    "closed_record_sha256": "d" * 64,
                    "sha256": "e" * 64,
                },
                "output_root_uri": (
                    "dbfs:/Volumes/catalog/schema/volume/handoff-generation"
                ),
                "source_kind": "distributed_q8_generation",
                "stage_kind": "local_nvme",
                "stage_uri": (
                    "/local_disk0/cachet-publication-latency/"
                    f"{'c' * 16}/{cell['job_id']}/handoff"
                ),
            }
        ),
        "input_files": [
            {
                "dataset": dataset,
                "uri": (f"dbfs:/Volumes/catalog/schema/volume/inputs/{dataset}.jsonl"),
            }
            for dataset in SUPPORTED_V1_DATASETS
        ],
        "job_id": cell["job_id"],
        "output": {"directory_uri": "dbfs:/Volumes/catalog/schema/volume/results/job"},
        "request_order": {
            "input_bundle_sha256": (
                qualification_v2.GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256
            ),
            "schedule_uri": (
                "dbfs:/Volumes/catalog/schema/volume/inputs/schedule.json"
            ),
        },
        "runtime": runtime,
        "source_revision": "deadbeef",
        "source_tree_sha256": "b" * 64,
    }


def test_publication_vllm_config_binds_complete_native_v2_runtime(monkeypatch):
    job = _native_v2_job_record()
    monkeypatch.setattr(
        execution,
        "validate_publication_latency_job_record",
        lambda _record: None,
    )

    config = execution.publication_latency_vllm_config(job)

    artifact_files = {item["role"]: item for item in job["artifact_files"]}
    assert config.native_runtime_v2 is not None
    assert config.native_runtime_v2.to_record() == {
        "package_wheel_sha256": artifact_files["package_wheel"]["sha256"],
        "package_wheel_uri": artifact_files["package_wheel"]["uri"],
        "patched_flashinfer_wheel_sha256": (
            artifact_files["patched_flashinfer_wheel"]["sha256"]
        ),
        "patched_flashinfer_wheel_uri": (
            artifact_files["patched_flashinfer_wheel"]["uri"]
        ),
        "patched_vllm_wheel_sha256": (artifact_files["patched_vllm_wheel"]["sha256"]),
        "patched_vllm_wheel_uri": artifact_files["patched_vllm_wheel"]["uri"],
        "runtime_closure_manifest_sha256": (
            artifact_files["runtime_closure_manifest"]["sha256"]
        ),
        "runtime_closure_manifest_uri": (
            artifact_files["runtime_closure_manifest"]["uri"]
        ),
        "runtime_lock_sha256": artifact_files["runtime_lock"]["sha256"],
        "runtime_lock_uri": artifact_files["runtime_lock"]["uri"],
    }
    assert config.package_install_spec == config.native_runtime_v2.package_wheel_uri
    assert (
        config.benchmark_evidence_policy
        == execution.PUBLICATION_LATENCY_COMPONENT_EVIDENCE_POLICY
        == "smoke"
    )
    with pytest.raises(ValueError, match="package_install_spec must match"):
        replace(
            config,
            package_install_spec=(
                "dbfs:/Volumes/catalog/schema/volume/runtime/other-cachet.whl"
            ),
        )


@pytest.mark.parametrize("method_id", ["baseline_prefill", "vanilla_prefill"])
def test_one_arm_latency_configs_emit_component_evidence(monkeypatch, method_id):
    job = _native_v2_job_record(method_id=method_id)
    monkeypatch.setattr(
        execution,
        "validate_publication_latency_job_record",
        lambda _record: None,
    )

    config = execution.publication_latency_vllm_config(job)

    assert config.benchmark_evidence_policy == "smoke"
    runner_args = vllm_smoke.build_benchmark_runner_args(
        config,
        vllm_smoke.parse_dataset_specs(config.dataset_specs),
    )
    assert "--cache-runtime-prompt" not in runner_args
    for flag in ("--baseline-extra-body-json", "--cache-extra-body-json"):
        assert (
            json.loads(runner_args[runner_args.index(flag) + 1])["add_special_tokens"]
            is False
        )
    assert execution.PUBLICATION_LATENCY_REQUEST_CUSTOMIZATION_DIGEST == (
        execution.sha256(b'{"add_special_tokens":false}').hexdigest()
    )


def test_component_evidence_gate_recomputes_full_canonical_record(monkeypatch):
    result = object()
    artifact_identities = {"artifact": object()}
    cache_state_attestations = (object(),)
    payload_digest = "a" * 64
    gate_result = object()
    canonical_gate = {
        "benchmark_payload_digest": payload_digest,
        "checked_cache_arms": [],
        "checked_cache_requests": 0,
        "checked_distinct_examples": 1,
        "cold_attested_requests": 0,
        "gate_version": 1,
        "issues": [],
        "measurement_scopes": ["latency", "resource"],
        "ok": True,
        "policy": "smoke",
        "record_type": "document_kv.benchmark_evidence_gate.v1",
    }
    record = {
        "evidence_gate": deepcopy(canonical_gate),
        "gate_inputs": {"sentinel": True},
    }

    monkeypatch.setattr(
        execution,
        "benchmark_gate_inputs_from_record",
        lambda observed: (
            (
                artifact_identities,
                cache_state_attestations,
            )
            if observed is record
            else (_ for _ in ()).throw(AssertionError("unexpected benchmark record"))
        ),
    )
    monkeypatch.setattr(
        execution,
        "benchmark_record_payload_digest",
        lambda observed: (
            payload_digest
            if observed is record
            else (_ for _ in ()).throw(AssertionError("unexpected benchmark record"))
        ),
    )

    def evaluate(observed_result, **kwargs):
        assert observed_result is result
        assert kwargs == {
            "artifact_identities": artifact_identities,
            "benchmark_payload_digest": payload_digest,
            "cache_state_attestations": cache_state_attestations,
            "policy": "smoke",
        }
        return gate_result

    monkeypatch.setattr(execution, "evaluate_benchmark_evidence_gate", evaluate)
    monkeypatch.setattr(
        execution,
        "benchmark_evidence_gate_to_record",
        lambda observed: (
            deepcopy(canonical_gate)
            if observed is gate_result
            else (_ for _ in ()).throw(AssertionError("unexpected gate result"))
        ),
    )

    execution._require_publication_latency_component_evidence_gate(
        record,
        result=result,
    )

    for field_name, tampered_value in (
        ("benchmark_payload_digest", "b" * 64),
        ("checked_distinct_examples", 2),
        ("measurement_scopes", ["latency"]),
    ):
        record["evidence_gate"] = {
            **canonical_gate,
            field_name: tampered_value,
        }
        with pytest.raises(ValueError, match="does not match recomputed evidence"):
            execution._require_publication_latency_component_evidence_gate(
                record,
                result=result,
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
                "patched_flashinfer_wheel": (execution.FLASHINFER_PATCHED_WHEEL_SHA256),
                "runtime_closure_manifest": (
                    execution.RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
                ),
                "runtime_lock": execution.VLLM_RUNTIME_BASE_LOCK_SHA256,
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
        runtime_lock_sha256=execution.VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_uri=final_artifacts.file("patched_vllm_wheel").uri,
        patched_vllm_wheel_sha256=(execution.GPU_QUALIFICATION_PATCHED_WHEEL_SHA256),
        patched_flashinfer_wheel_uri=final_artifacts.file(
            "patched_flashinfer_wheel"
        ).uri,
        patched_flashinfer_wheel_sha256=(execution.FLASHINFER_PATCHED_WHEEL_SHA256),
        runtime_closure_manifest_uri=final_artifacts.file(
            "runtime_closure_manifest"
        ).uri,
        runtime_closure_manifest_sha256=(
            execution.RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
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
    storage_schedules = [{**item, "selection_sha256": "f" * 64} for item in schedules]
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
        "input_bundle_sha256": (
            qualification_v2.GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256
        ),
        "ledger_lineage": {
            "ledger_id": "campaign",
            "ledger_path_sha256": "a" * 64,
            "predecessor_prefix": opening.to_record(),
        },
        "qualification_artifact_pins": {
            "cachet_source_tree_sha256": "b" * 64,
            "input_bundle_sha256": (
                qualification_v2.GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256
            ),
            "package_wheel_sha256": digest,
            "patched_flashinfer_wheel_sha256": (
                execution.FLASHINFER_PATCHED_WHEEL_SHA256
            ),
            "patched_vllm_wheel_sha256": (
                execution.GPU_QUALIFICATION_PATCHED_WHEEL_SHA256
            ),
            "runner_sha256": "d" * 64,
            "runtime_closure_manifest_sha256": (
                execution.RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
            ),
            "runtime_lock_sha256": execution.VLLM_RUNTIME_BASE_LOCK_SHA256,
        },
        "record_bindings": {
            "campaign": binding("campaign_plan", "1" * 64),
            "qualification_evidence": binding("qualification_evidence", "2" * 64),
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
    request["attempt_id"] = execution._publication_latency_source_closure_attempt_id(
        singleton_identity_sha256
    )
    request_root, result_root = (
        execution.publication_latency_source_closure_control_roots(
            final_artifacts.handoff_generation_root_uri,
            final_artifacts.bf16_handoff_generation_root_uri,
        )
    )
    request["coordinator"]["request_root_uri"] = request_root
    request["coordinator"]["result_root_uri"] = result_root
    request["request_uri"] = execution._join_durable_uri(request_root, "request.json")
    request["result_uri"] = execution._join_durable_uri(result_root, "result.json")
    request["closed_record_sha256"] = execution._closed_record_sha256(request)
    native_runtime = execution._source_closure_native_runtime_v2(config)
    runtime_verification = _native_v2_runtime_attestation(
        vllm_uri=native_runtime.local_path("patched_vllm_wheel").resolve().as_uri(),
        flashinfer_uri=(
            native_runtime.local_path("patched_flashinfer_wheel").resolve().as_uri()
        ),
    )
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
        "runtime_attestation": execution._source_closure_runtime_attestation_record(
            config, runtime_verification
        ),
        "schema_version": execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_SCHEMA_VERSION,
        "semantic": semantic,
    }
    result["artifacts_sha256"] = execution._canonical_sha256(result["artifacts"])
    result["closed_record_sha256"] = execution._closed_record_sha256(result)
    return request, result, final_artifacts


def _source_closure_request_authorization(
    request,
    *,
    phase_lease_root=None,
    workspace_host="https://dbc.example",
    user_name="publication@example.com",
):
    lease_root = (
        Path.cwd()
        / ".test-latency-source-closure"
        / request["singleton_identity_sha256"][:24]
        if phase_lease_root is None
        else Path(phase_lease_root)
    )
    q8 = object.__new__(execution.PublicationHandoffRemoteClosureAuthorization)
    bf16 = object.__new__(execution.PublicationHandoffRemoteClosureAuthorization)
    return execution.PublicationLatencySourceClosureRequestAuthorization(
        request=request,
        phase_lease_root=lease_root,
        workspace_identity={
            "workspace_host_sha256": execution.sha256(
                workspace_host.encode("utf-8")
            ).hexdigest(),
            "user_name_sha256": execution.sha256(user_name.encode("utf-8")).hexdigest(),
        },
        q8_authorization=q8,
        bf16_authorization=bf16,
        _issuer=execution._SOURCE_CLOSURE_REQUEST_AUTHORIZATION_ISSUER,
    )


def _reseal_source_closure_singleton(request):
    identity = execution._source_closure_singleton_identity_from_request(request)
    request["singleton_identity_sha256"] = identity
    request["attempt_id"] = execution._publication_latency_source_closure_attempt_id(
        identity
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
    request["request_uri"] = execution._join_durable_uri(request_root, "request.json")
    request["result_uri"] = execution._join_durable_uri(result_root, "result.json")
    request["closed_record_sha256"] = execution._closed_record_sha256(request)


def test_source_closure_request_result_and_cpu_payload_are_closed(
    monkeypatch,
    tmp_path,
):
    request, result, _final_artifacts = _source_closure_records()

    monkeypatch.setenv("PIP_CONFIG_FILE", "/attacker/pip.conf")
    monkeypatch.setenv("PIP_INDEX_URL", "https://attacker.invalid/simple")
    monkeypatch.setenv("pip_no_index", "1")
    monkeypatch.setenv("PIP_REQUIREMENT", "/attacker/requirements.txt")
    monkeypatch.setenv("_PIP_STANDALONE_CERT", "/attacker/cert.pem")
    monkeypatch.setenv("PYTHONHOME", "/attacker/python-home")
    monkeypatch.setenv("PYTHONPATH", "/attacker/python-path")
    monkeypatch.setenv("VIRTUAL_ENV", "/attacker/venv")
    monkeypatch.setenv("FLASHINFER_LOGGING_LEVEL", "DEBUG")
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
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
        assert {key for key in environment if key.upper().startswith("PIP_")} == {
            "PIP_CONFIG_FILE",
            "PIP_DISABLE_PIP_VERSION_CHECK",
            "PIP_NO_INPUT",
        }
        assert environment["PIP_CONFIG_FILE"] == os.devnull
        assert (
            environment["FLASHINFER_LOGGING_LEVEL"]
            == GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL
        )
        assert not any(key.upper().startswith("_PIP_") for key in environment)
        assert environment["PYTHONNOUSERSITE"] == "1"
        assert environment["PYTHONWARNINGS"] == GPU_RUNTIME_PYTHONWARNINGS
        assert all(
            variable not in environment
            for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")
        )
    assert '"--extra-index-url"' not in (
        execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT
    )
    assert "_verified(__file__, args.runner_sha256" in (
        execution.PUBLICATION_LATENCY_RUNNER_SCRIPT
    )
    bound_runner = tmp_path / "bound-publication-latency-runner.py"
    bound_runner.write_text(
        execution.PUBLICATION_LATENCY_RUNNER_SCRIPT,
        encoding="utf-8",
    )
    substituted_runner = tmp_path / "substituted-publication-latency-runner.py"
    substituted_runner.write_text("# unreviewed runner\n", encoding="utf-8")
    runner_namespace = {
        "__file__": str(substituted_runner),
        "__name__": "latency_runner_provenance_test",
    }
    exec(
        compile(
            execution.PUBLICATION_LATENCY_RUNNER_SCRIPT,
            str(substituted_runner),
            "exec",
        ),
        runner_namespace,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(substituted_runner),
            "--job-record-json",
            "{}",
            "--expected-job-sha256",
            "0" * 64,
            "--runner-uri",
            str(bound_runner),
            "--runner-sha256",
            execution.sha256(bound_runner.read_bytes()).hexdigest(),
            "--package-wheel-uri",
            str(tmp_path / "unreached.whl"),
            "--package-wheel-sha256",
            "1" * 64,
            "--cloud-run-id",
            "1",
            "--task-run-id",
            "2",
        ],
    )
    with pytest.raises(RuntimeError, match="executing publication latency runner"):
        runner_namespace["main"]()

    wheel = tmp_path / "cachet.whl"
    wheel.write_bytes(b"publication-latency-wheel")
    valid_runner_namespace = {
        "__file__": str(bound_runner),
        "__name__": "latency_runner_child_test",
    }
    exec(
        compile(
            execution.PUBLICATION_LATENCY_RUNNER_SCRIPT,
            str(bound_runner),
            "exec",
        ),
        valid_runner_namespace,
    )
    calls = []

    def capture_check_call(command, *, env):
        calls.append((command, env))

    monkeypatch.setattr(
        valid_runner_namespace["subprocess"],
        "check_call",
        capture_check_call,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(bound_runner),
            "--job-record-json",
            "{}",
            "--expected-job-sha256",
            "0" * 64,
            "--runner-uri",
            str(bound_runner),
            "--runner-sha256",
            execution.sha256(bound_runner.read_bytes()).hexdigest(),
            "--package-wheel-uri",
            str(wheel),
            "--package-wheel-sha256",
            execution.sha256(wheel.read_bytes()).hexdigest(),
            "--cloud-run-id",
            "1",
            "--task-run-id",
            "2",
        ],
    )
    valid_runner_namespace["main"]()
    assert len(calls) == 2
    child_command, child_environment = calls[1]
    assert child_command[:3] == [sys.executable, "-P", "-c"]
    assert child_command[4:] == [
        "run-job",
        "--job-record-json",
        "{}",
        "--expected-job-sha256",
        "0" * 64,
        "--cloud-run-id",
        "1",
        "--task-run-id",
        "2",
    ]
    child_stub = child_command[3]
    assert child_stub.index("sys.warnoptions") < child_stub.index("runpy.run_module")
    assert (
        child_environment["FLASHINFER_LOGGING_LEVEL"]
        == GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL
    )
    assert child_environment["PYTHONWARNINGS"] == GPU_RUNTIME_PYTHONWARNINGS
    child_cases = []
    missing_policy_environment = dict(child_environment)
    missing_policy_environment.pop("PYTHONWARNINGS")
    child_cases.append(([], missing_policy_environment, "pinned CUDA warning"))
    hostile_policy_environment = {
        **child_environment,
        "FLASHINFER_LOGGING_LEVEL": "DEBUG",
        "PYTHONWARNINGS": "ignore",
    }
    child_cases.append(([], hostile_policy_environment, "pinned CUDA warning"))
    child_cases.append((["-W", "ignore"], child_environment, "pinned CUDA warning"))
    for extra_python_args, environment, expected_error in child_cases:
        completed = subprocess.run(
            [
                sys.executable,
                "-P",
                *extra_python_args,
                "-c",
                child_stub,
            ],
            capture_output=True,
            env=environment,
            text=True,
        )
        assert completed.returncode != 0
        assert expected_error in completed.stderr
        assert "ModuleNotFoundError" not in completed.stderr
    source_install = execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT.split(
        'pip = [venv_python, "-m", "pip"]', maxsplit=1
    )[1]
    install_positions = [
        source_install.index('"--require-hashes"'),
        source_install.index('"vllm"'),
        source_install.index('"flashinfer-python"'),
        source_install.index('"cachet-kv"'),
        source_install.index("_verify_locked_runtime("),
    ]
    assert install_positions == sorted(install_positions)
    assert '"--copies"' in execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT
    assert "validate_gpu_qualification_v2_runtime_attestation" in (
        execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT
    )
    assert "timeout=_FINAL_RUNTIME_VERIFIER_TIMEOUT_SECONDS" in (
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
    assert task["new_cluster"]["data_security_mode"] == "SINGLE_USER"
    assert task["new_cluster"]["single_user_name"] == ("publication@example.com")
    assert task["new_cluster"]["custom_tags"]["ResourceClass"] == "SingleNode"
    assert task["timeout_seconds"] == 2 * 60 * 60
    assert payload["timeout_seconds"] == 2 * 60 * 60
    parameters = task["spark_python_task"]["parameters"]
    assert parameters.count("--runner-sha256") == 1
    assert parameters.count("--runtime-lock-sha256") == 1
    assert parameters.count("--patched-vllm-wheel-sha256") == 1
    assert parameters.count("--patched-flashinfer-wheel-sha256") == 1
    assert parameters.count("--runtime-closure-manifest-sha256") == 1
    assert result["runtime_attestation"]["verification"]["ok"] is True
    assert len(result["artifacts"]) == (
        len(execution._final_artifact_roles())
        - len(execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_EXCLUDED_ROLES)
    )


@pytest.mark.parametrize(
    "single_user_name",
    [
        " publication@example.com",
        "publication@example.com ",
        "pub\x00lication@example.com",
        "pub\x7flication@example.com",
    ],
)
def test_source_closure_rejects_noncanonical_single_user_name(single_user_name):
    request, _result, _final_artifacts = _source_closure_records()
    coordinator = deepcopy(request["coordinator"])
    coordinator["single_user_name"] = single_user_name

    with pytest.raises(ValueError, match="normalized non-empty string"):
        execution._source_closure_config_from_record(coordinator)


def test_source_closure_rejects_qualification_runner_conflation():
    request, _result, _final_artifacts = _source_closure_records()
    for downstream_runner_sha256 in (
        execution.PUBLICATION_LATENCY_RUNNER_SHA256,
        execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SHA256,
    ):
        conflated = deepcopy(request)
        conflated["qualification_artifact_pins"]["runner_sha256"] = (
            downstream_runner_sha256
        )
        _reseal_source_closure_singleton(conflated)
        with pytest.raises(ValueError, match="runner identities must be distinct"):
            execution.validate_publication_latency_source_closure_request(conflated)


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


def test_source_closure_result_rejects_rebound_native_v2_runtime_origin():
    request, result, _final_artifacts = _source_closure_records()
    result = deepcopy(result)
    result["runtime_attestation"]["verification"]["flashinfer_direct_url"] = (
        "file:///tmp/attacker-flashinfer.whl"
    )
    result["closed_record_sha256"] = execution._closed_record_sha256(result)

    with pytest.raises(ValueError, match="artifact origin drift"):
        execution.validate_publication_latency_source_closure_result(
            result, request=request
        )


def test_source_closure_authorization_is_issuer_only():
    request, result, _final_artifacts = _source_closure_records()
    prefix = databricks_ledger_prefix(DatabricksClusterHourLedger(ledger_id="campaign"))

    assert (
        get_type_hints(execution.build_publication_latency_source_closure_request)[
            "return"
        ]
        is execution.PublicationLatencySourceClosureRequestAuthorization
    )
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
            workspace_host_sha256="b" * 64,
            user_name_sha256="c" * 64,
            _issuer=object(),
        )


@pytest.mark.parametrize(
    ("live_host", "live_user"),
    (
        ("https://dbc.other", "publication@example.com"),
        ("https://dbc.example", "other-publication@example.com"),
    ),
    ids=("different-host", "different-principal"),
)
def test_source_closure_workspace_identity_is_authority_only_and_fail_closed(
    tmp_path,
    monkeypatch,
    live_host,
    live_user,
):
    request, _result, _final_artifacts = _source_closure_records()
    request_bytes = execution._pretty_json_bytes(request)
    authorization = _source_closure_request_authorization(
        request,
        phase_lease_root=tmp_path / "source-closure-lease",
    )
    repeated = _source_closure_request_authorization(
        request,
        phase_lease_root=tmp_path / "source-closure-repeated-lease",
    )
    expected_host = execution.sha256(b"https://dbc.example").hexdigest()
    expected_user = execution.sha256(b"publication@example.com").hexdigest()

    assert execution._pretty_json_bytes(authorization.request_record) == request_bytes
    assert execution.sha256(request_bytes).hexdigest() == (
        authorization.request_file_sha256
    )
    assert b"workspace_host_sha256" not in request_bytes
    assert b"user_name_sha256" not in request_bytes
    assert authorization.authorization_record == repeated.authorization_record
    assert authorization.authorization_record["workspace_host_sha256"] == (
        expected_host
    )
    assert authorization.authorization_record["user_name_sha256"] == expected_user
    assert (
        execution._require_source_closure_request_authorization(authorization)
        is authorization
    )
    with pytest.raises(
        TypeError,
        match="PublicationLatencySourceClosureRequestAuthorization",
    ):
        execution._require_source_closure_request_authorization(SimpleNamespace())

    workspace = execution.DatabricksWorkspaceConfig("https://dbc.example", "token")
    execution._require_source_closure_workspace_identity(workspace, authorization)

    monkeypatch.setattr(
        execution,
        "require_publication_handoff_remote_closure_workspace_identity",
        lambda *_args, **_kwargs: {
            "authenticated": True,
            "workspace_host_sha256": execution.sha256(
                live_host.encode("utf-8")
            ).hexdigest(),
            "user_name_sha256": execution.sha256(live_user.encode("utf-8")).hexdigest(),
        },
    )
    with pytest.raises(ValueError, match="workspace/principal authority drift"):
        execution._require_source_closure_workspace_identity(workspace, authorization)
    assert not authorization.phase_lease_root.exists()
    assert execution._pretty_json_bytes(authorization.request_record) == request_bytes


def test_source_closure_raw_and_resealed_swapped_package_fail_before_side_effects(
    tmp_path, monkeypatch
):
    request, _result, _final_artifacts = _source_closure_records()
    swapped = deepcopy(request)
    swapped_sha256 = "b" * 64
    swapped_uri = "dbfs:/Volumes/catalog/schema/volume/artifacts/swapped-package.whl"
    swapped["coordinator"]["package_wheel_sha256"] = swapped_sha256
    swapped["coordinator"]["package_wheel_uri"] = swapped_uri
    swapped["qualification_artifact_pins"]["package_wheel_sha256"] = swapped_sha256
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
    workspace = execution.DatabricksWorkspaceConfig("https://dbc.example", "token")
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
        execution.validate_publication_latency_source_closure_request(alternate_attempt)

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


def test_source_closure_wrong_current_user_fails_before_lease_or_io(
    tmp_path,
    monkeypatch,
):
    ledger_path = tmp_path / "gpu-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="campaign", cap_cluster_hours=1024.0
    )
    request, result, _final_artifacts = _source_closure_records()
    request, _result = _bind_source_closure_to_ledger(request, result, ledger_path)
    lease = tmp_path / "never-created-source-lease"
    request_authorization = _source_closure_request_authorization(
        request, phase_lease_root=lease
    )
    ledger_before = ledger_path.read_bytes()
    observed = []
    external_calls = []

    def reject_current_user(_workspace, *, expected_user_name, opener=None):
        observed.append((expected_user_name, opener))
        raise ValueError("Databricks current-user identity differs")

    monkeypatch.setattr(
        execution,
        "require_databricks_current_user_name",
        reject_current_user,
    )
    monkeypatch.setattr(
        execution,
        "upload_databricks_volume_file_bytes_exclusive",
        lambda *args, **kwargs: external_calls.append(("upload", args, kwargs)),
    )
    monkeypatch.setattr(
        execution,
        "submit_databricks_run",
        lambda *args, **kwargs: external_calls.append(("post", args, kwargs)),
    )

    with pytest.raises(ValueError, match="current-user identity differs"):
        execution.submit_publication_latency_source_closure(
            execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
            request_authorization=request_authorization,
            ledger_path=ledger_path,
            phase_lease_root=lease,
        )

    assert observed == [("publication@example.com", None)]
    assert external_calls == []
    assert ledger_path.read_bytes() == ledger_before
    assert not lease.exists()


def test_source_closure_lost_response_resume_is_idempotent_and_gpu_ledger_read_only(
    tmp_path, monkeypatch
):
    ledger_path = tmp_path / "gpu-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="campaign", cap_cluster_hours=1024.0
    )
    request, result, _final_artifacts = _source_closure_records()
    request, _result = _bind_source_closure_to_ledger(request, result, ledger_path)
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
    workspace = execution.DatabricksWorkspaceConfig("https://dbc.example", "token")
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
    assert persisted_authorization == dict(request_authorization.authorization_record)
    assert persisted_authorization["closed_record_sha256"] == (
        request_authorization.authorization_record_sha256
    )
    assert persisted_authorization["workspace_host_sha256"] == (
        request_authorization.workspace_host_sha256
    )
    assert persisted_authorization["user_name_sha256"] == (
        request_authorization.user_name_sha256
    )
    assert "workspace_host_sha256" not in request
    assert "user_name_sha256" not in request
    lease_snapshot = {
        path.relative_to(lease): path.read_bytes()
        for path in lease.rglob("*")
        if path.is_file()
    }
    observed_principals = []

    def reject_current_user(_workspace, *, expected_user_name, opener=None):
        observed_principals.append((expected_user_name, opener))
        raise ValueError("Databricks current-user identity differs")

    def rejected_opener(*_args, **_kwargs):
        raise AssertionError("wrong-principal recovery must not perform HTTP")

    with monkeypatch.context() as identity_patch:
        identity_patch.setattr(
            execution,
            "require_databricks_current_user_name",
            reject_current_user,
        )
        with pytest.raises(ValueError, match="current-user identity differs"):
            execution.resume_publication_latency_source_closure(
                workspace,
                request_authorization=request_authorization,
                ledger_path=ledger_path,
                phase_lease_root=lease,
                opener=rejected_opener,
            )
    assert observed_principals == [("publication@example.com", rejected_opener)]
    assert len(upload_calls) == 1
    assert {
        path.relative_to(lease): path.read_bytes()
        for path in lease.rglob("*")
        if path.is_file()
    } == lease_snapshot

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
    assert (
        authorization.workspace_host_sha256
        == replayed_authorization.workspace_host_sha256
        == request_authorization.workspace_host_sha256
    )
    assert (
        authorization.user_name_sha256
        == replayed_authorization.user_name_sha256
        == request_authorization.user_name_sha256
    )
    assert authorization.workspace_authority_closure_sha256 == (
        replayed_authorization.workspace_authority_closure_sha256
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
    tampered_authorization["closed_record_sha256"] = execution._closed_record_sha256(
        tampered_authorization
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
        swapped_request,
        workspace_host_sha256=request_authorization.workspace_host_sha256,
        user_name_sha256=request_authorization.user_name_sha256,
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
                "spark_python_task": deepcopy(payload["tasks"][0]["spark_python_task"]),
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
            workspace_host_sha256=request_authorization.workspace_host_sha256,
            user_name_sha256=request_authorization.user_name_sha256,
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
    assert (
        execution._source_closure_authorization_binding(authorization)[
            "single_user_name"
        ]
        == "publication@example.com"
    )
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
        "require_publication_latency_source_closure_authorization" in validation_names
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
    assert "patched_flashinfer_wheel" in roles
    assert "runtime_closure_manifest" in roles
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
    with pytest.raises(TypeError, match="GPUQualificationLaunchAuthorization"):
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
    with pytest.raises(TypeError, match="GPUQualificationLaunchAuthorization"):
        execution._require_latency_launch_authorization(
            plan,
            qualification_authorization,  # type: ignore[arg-type]
            q8_authorization,  # type: ignore[arg-type]
            cross_ledger_bf16_authorization,  # type: ignore[arg-type]
            source_authorization,  # type: ignore[arg-type]
        )


def _ram_cache_telemetry(*, requests: int) -> dict[str, int]:
    return {
        "backend_bytes_read": 0,
        "cold_read_attested_count": 0,
        "eviction_requested_count": 0,
        "eviction_succeeded_count": 0,
        "expected_backend_bytes_read": 0,
        "load_count": requests,
        "mounted_path_load_count": 0,
        "payload_cache_hit_count": requests,
        "payload_cache_miss_count": 0,
        "storage_materialization_count": 0,
    }


def _descriptive_ram_record() -> dict:
    latency_metrics = {
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
    sample_counts = (10, 20, 30, 40, 50)
    mean_utilizations = (10.0, 20.0, 30.0, 40.0, 50.0)
    peak_utilizations = (15.0, 25.0, 35.0, 45.0, 55.0)
    total_samples = sum(sample_counts)
    record = {
        **latency_metrics,
        "cache_telemetry": _ram_cache_telemetry(requests=1280),
        "cell_id": "auxiliary-storage-ram",
        "cell_kind": "auxiliary_pooled_five_blocks",
        "cell_sha256": "",
        "comparison_family": "storage",
        "gpu_utilization_sample_count": total_samples,
        "input_tokens": 16_384,
        "mean_gpu_utilization_percent": sum(
            mean * count
            for mean, count in zip(mean_utilizations, sample_counts, strict=True)
        )
        / total_samples,
        "method_id": "vanilla_prefill",
        "observation_count": 1280,
        "peak_gpu_utilization_percent": max(peak_utilizations),
        "physical_blocks": [
            {
                **latency_metrics,
                "cache_telemetry": _ram_cache_telemetry(requests=256),
                "deployment_block": block,
                "gpu_utilization_sample_count": sample_counts[block - 1],
                "job_id": f"block-{block:02d}-storage-ram",
                "mean_gpu_utilization_percent": mean_utilizations[block - 1],
                "observation_count": 256,
                "peak_gpu_utilization_percent": peak_utilizations[block - 1],
            }
            for block in range(1, 6)
        ],
        "quantile_method": "empirical_nearest_rank",
        "request_parallelism": 4,
        "setting_id": "storage-ram",
    }
    record["cell_sha256"] = execution._descriptive_cell_sha256(record)
    return record


def _reclose_descriptive_cell(record: dict) -> dict:
    record["cell_sha256"] = execution._descriptive_cell_sha256(record)
    return record


def test_descriptive_cell_record_is_closed_and_retains_sanitized_evidence():
    record = _descriptive_ram_record()

    execution._validate_descriptive_cell_record(record)
    expected_cache_keys = {
        "backend_bytes_read",
        "cold_read_attested_count",
        "eviction_requested_count",
        "eviction_succeeded_count",
        "expected_backend_bytes_read",
        "load_count",
        "mounted_path_load_count",
        "payload_cache_hit_count",
        "payload_cache_miss_count",
        "storage_materialization_count",
    }
    forbidden_cache_keys = {
        "benchmark_request_ids_sha256",
        "principal",
        "telemetry_file_sha256",
        "uri",
    }
    for container in (record, *record["physical_blocks"]):
        assert set(container["cache_telemetry"]) == expected_cache_keys
        assert not forbidden_cache_keys.intersection(container["cache_telemetry"])
        assert {
            "gpu_utilization_sample_count",
            "mean_gpu_utilization_percent",
            "peak_gpu_utilization_percent",
        } <= set(container)

    record["p95_ttft_seconds"] = 3.0
    with pytest.raises(ValueError, match="digest"):
        execution._validate_descriptive_cell_record(record)


def test_descriptive_cache_projection_drops_provenance_bearing_telemetry():
    source_cache = {
        **_ram_cache_telemetry(requests=256),
        "benchmark_request_ids_sha256": "a" * 64,
        "telemetry_file_sha256": "b" * 64,
    }

    projection = execution._descriptive_cache_telemetry_projection(
        {"cache_telemetry": source_cache},
        method_id="vanilla_prefill",
    )

    assert projection == _ram_cache_telemetry(requests=256)
    baseline_source = {
        key: 0
        for key in _ram_cache_telemetry(requests=0)
        if key != "expected_backend_bytes_read"
    }
    assert (
        execution._descriptive_cache_telemetry_projection(
            {"cache_telemetry": baseline_source},
            method_id="baseline_prefill",
        )["expected_backend_bytes_read"]
        == 0
    )


def test_descriptive_cache_claims_distinguish_uc_from_strict_cold_storage():
    uc = _ram_cache_telemetry(requests=256)
    uc.update(
        {
            "backend_bytes_read": 4096,
            "eviction_requested_count": 256,
            "eviction_succeeded_count": 256,
            "expected_backend_bytes_read": 4096,
            "mounted_path_load_count": 256,
            "payload_cache_hit_count": 0,
        }
    )
    execution._validate_descriptive_cache_claim(
        uc,
        method_id="vanilla_prefill",
        observation_count=256,
        setting_id="storage-uc",
    )
    forged_uc = {**uc, "expected_backend_bytes_read": 4097}
    with pytest.raises(ValueError, match="UC descriptive cache telemetry"):
        execution._validate_descriptive_cache_claim(
            forged_uc,
            method_id="vanilla_prefill",
            observation_count=256,
            setting_id="storage-uc",
        )
    uc_with_bounded_cold_attestations = {
        **uc,
        "cold_read_attested_count": 255,
    }
    execution._validate_descriptive_cache_claim(
        uc_with_bounded_cold_attestations,
        method_id="vanilla_prefill",
        observation_count=256,
        setting_id="storage-uc",
    )
    counter_fields = set(uc).difference(
        {"backend_bytes_read", "expected_backend_bytes_read"}
    )
    for field_name in counter_fields:
        forged_counter = {**uc, field_name: 257}
        with pytest.raises(ValueError, match="exceeds descriptive observation_count"):
            execution._validate_descriptive_cache_claim(
                forged_counter,
                method_id="vanilla_prefill",
                observation_count=256,
                setting_id="storage-uc",
            )
    forged_uc_miss = {
        **uc,
        "payload_cache_miss_count": 1,
        "storage_materialization_count": 1,
    }
    with pytest.raises(ValueError, match="UC descriptive cache telemetry"):
        execution._validate_descriptive_cache_claim(
            forged_uc_miss,
            method_id="vanilla_prefill",
            observation_count=256,
            setting_id="storage-uc",
        )
    forged_uc_zero_bytes = {
        **uc,
        "backend_bytes_read": 0,
        "expected_backend_bytes_read": 0,
    }
    with pytest.raises(ValueError, match="UC descriptive cache telemetry"):
        execution._validate_descriptive_cache_claim(
            forged_uc_zero_bytes,
            method_id="vanilla_prefill",
            observation_count=256,
            setting_id="storage-uc",
        )

    strict_cold = {
        **uc,
        "cold_read_attested_count": 256,
        "mounted_path_load_count": 0,
    }
    execution._validate_descriptive_cache_claim(
        strict_cold,
        method_id="vanilla_prefill",
        observation_count=256,
        setting_id="storage-disk",
    )
    forged_cold = {**strict_cold, "cold_read_attested_count": 255}
    with pytest.raises(ValueError, match="cold descriptive cache telemetry"):
        execution._validate_descriptive_cache_claim(
            forged_cold,
            method_id="vanilla_prefill",
            observation_count=256,
            setting_id="storage-disk",
        )
    forged_cold_miss = {
        **strict_cold,
        "payload_cache_miss_count": 1,
        "storage_materialization_count": 1,
    }
    with pytest.raises(ValueError, match="cold descriptive cache telemetry"):
        execution._validate_descriptive_cache_claim(
            forged_cold_miss,
            method_id="vanilla_prefill",
            observation_count=256,
            setting_id="storage-disk",
        )
    forged_cold_zero_bytes = {
        **strict_cold,
        "backend_bytes_read": 0,
        "expected_backend_bytes_read": 0,
    }
    with pytest.raises(ValueError, match="cold descriptive cache telemetry"):
        execution._validate_descriptive_cache_claim(
            forged_cold_zero_bytes,
            method_id="vanilla_prefill",
            observation_count=256,
            setting_id="storage-disk",
        )


def _realistic_local_path_connector_load(
    request_id: str,
    *,
    cold_read_attested: bool,
    payload_cache_hit: bool,
) -> dict:
    expected_stored_bytes = 4096
    expected_runtime_bytes = 8192
    expected_tokens = 128
    return {
        "benchmark_request_id": request_id,
        "cache_state_attestation": {
            "bytes_read": 0 if payload_cache_hit else expected_stored_bytes,
            "cold_read_attested": cold_read_attested,
            "eviction_requested": not payload_cache_hit,
            "eviction_succeeded": not payload_cache_hit,
            "expected_runtime_bytes": expected_runtime_bytes,
            "expected_stored_bytes": expected_stored_bytes,
            "expected_tokens": expected_tokens,
            "loaded_tokens": expected_tokens,
            "payload_cache_hit": payload_cache_hit,
            "source": "local_path",
            "successful_loads": 1,
        },
        "counts": {
            "decoded_runtime_payload_bytes": expected_runtime_bytes,
            "expected_runtime_payload_bytes": expected_runtime_bytes,
            "expected_stored_payload_bytes": expected_stored_bytes,
            "payload_cache_hits": int(payload_cache_hit),
            "payload_cache_misses": 0,
            "token_count": expected_tokens,
        },
        "event": "load_request",
        "layout": {"dtype": "bfloat16"},
        "payload": {
            "payload_cache_enabled": payload_cache_hit,
            "source": "uri",
            "uri_scheme": "local_path",
        },
        "record_type": "document_kv.vllm_native_provider_load.v1",
        "success": True,
    }


def test_mounted_path_load_count_requires_verified_uc_backend_reads(monkeypatch):
    request_ids = [
        f"request-{index:03d}"
        for index in range(execution.PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL)
    ]
    benchmark = SimpleNamespace(
        measurements=tuple(
            SimpleNamespace(
                metadata={
                    execution.PUBLICATION_LATENCY_REQUEST_ID_METADATA_KEY: request_id
                }
            )
            for request_id in request_ids
        )
    )
    monkeypatch.setattr(
        execution,
        "benchmark_run_result_from_record",
        lambda *_args, **_kwargs: benchmark,
    )
    monkeypatch.setattr(execution, "_file_sha256", lambda _path: "a" * 64)

    def summarize(
        host_cache_state: str,
        *,
        cold_read_attested: bool,
        payload_cache_hit: bool,
    ) -> dict:
        records = [
            _realistic_local_path_connector_load(
                request_id,
                cold_read_attested=cold_read_attested,
                payload_cache_hit=payload_cache_hit,
            )
            for request_id in request_ids
        ]
        monkeypatch.setattr(
            execution,
            "_read_jsonl_file",
            lambda *_args, **_kwargs: records,
        )
        return execution._publication_latency_cache_telemetry_summary(
            Path("connector-telemetry.jsonl"),
            job_record={
                "cache_telemetry_policy": {"host_cache_state": host_cache_state},
                "cell": {"method_id": "vanilla_prefill"},
                "runtime": {"runtime_kv_dtype": "bfloat16"},
            },
            benchmark_record={},
        )

    ram = summarize(
        "prewarmed_payload_cache",
        cold_read_attested=False,
        payload_cache_hit=True,
    )
    uc = summarize(
        "mounted_path_evicted_backend_cache_unproven",
        cold_read_attested=False,
        payload_cache_hit=False,
    )
    disk = summarize(
        "cold_eviction_required",
        cold_read_attested=True,
        payload_cache_hit=False,
    )

    assert ram["mounted_path_load_count"] == 0
    assert ram["payload_cache_hit_count"] == len(request_ids)
    assert ram["backend_bytes_read"] == 0
    assert uc["mounted_path_load_count"] == len(request_ids)
    assert uc["backend_bytes_read"] == uc["expected_backend_bytes_read"]
    assert disk["mounted_path_load_count"] == 0
    assert disk["cold_read_attested_count"] == len(request_ids)


def test_descriptive_cell_validator_rejects_resource_and_cache_tampering():
    mutations = (
        (
            lambda item: item.__setitem__(
                "gpu_utilization_sample_count",
                item["gpu_utilization_sample_count"] + 1,
            ),
            "sample-count",
        ),
        (
            lambda item: item.__setitem__(
                "mean_gpu_utilization_percent",
                item["mean_gpu_utilization_percent"] + 1.0,
            ),
            "weighted GPU mean",
        ),
        (
            lambda item: item.__setitem__("peak_gpu_utilization_percent", 54.0),
            "utilization peak",
        ),
        (
            lambda item: item["cache_telemetry"].__setitem__("load_count", 1281),
            "pooled cache telemetry sum",
        ),
        (
            lambda item: item["physical_blocks"][0].__setitem__(
                "mean_gpu_utilization_percent", 10
            ),
            "finite float",
        ),
        (
            lambda item: item["physical_blocks"][0].__setitem__(
                "peak_gpu_utilization_percent", 101.0
            ),
            "finite float",
        ),
        (
            lambda item: item["physical_blocks"][0].__setitem__(
                "gpu_utilization_sample_count", True
            ),
            "positive integer",
        ),
        (
            lambda item: item["physical_blocks"][0]["cache_telemetry"].__setitem__(
                "load_count", True
            ),
            "non-negative integer",
        ),
        (
            lambda item: item["physical_blocks"][0]["cache_telemetry"].__setitem__(
                "telemetry_file_sha256", "a" * 64
            ),
            "schema is not closed",
        ),
    )
    for mutate, message in mutations:
        forged = deepcopy(_descriptive_ram_record())
        mutate(forged)
        _reclose_descriptive_cell(forged)
        with pytest.raises(ValueError, match=message):
            execution._validate_descriptive_cell_record(forged)

    for non_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite float"):
            execution._gpu_utilization_percentage(
                non_finite,
                "mean_gpu_utilization_percent",
            )

    forged_claim = deepcopy(_descriptive_ram_record())
    forged_claim["physical_blocks"][0]["cache_telemetry"]["backend_bytes_read"] = 1
    forged_claim["cache_telemetry"]["backend_bytes_read"] = 1
    _reclose_descriptive_cell(forged_claim)
    with pytest.raises(ValueError, match="RAM descriptive cache telemetry"):
        execution._validate_descriptive_cell_record(forged_claim)


def test_descriptive_cell_rejects_reclosed_job_identity_and_small_count_tampering():
    core_coordinates = execution._publication_latency_descriptive_cell_coordinates()[
        "core-baseline_prefill-8192-c1"
    ]
    assert (
        execution._publication_latency_descriptive_physical_job_id(
            core_coordinates,
            deployment_block=3,
        )
        == "block-03-8k-c1-baseline"
    )
    ram_coordinates = execution._publication_latency_descriptive_cell_coordinates()[
        "auxiliary-storage-ram"
    ]
    assert (
        execution._publication_latency_descriptive_physical_job_id(
            ram_coordinates,
            deployment_block=5,
        )
        == "block-05-storage-ram"
    )

    forged_identity = deepcopy(_descriptive_ram_record())
    forged_identity["physical_blocks"][0]["job_id"] = "arbitrary-job"
    _reclose_descriptive_cell(forged_identity)
    with pytest.raises(ValueError, match="frozen physical job identity"):
        execution._validate_descriptive_cell_record(forged_identity)

    forged_counts = deepcopy(_descriptive_ram_record())
    forged_counts["observation_count"] = 5
    forged_counts["cache_telemetry"]["load_count"] = 5
    forged_counts["cache_telemetry"]["payload_cache_hit_count"] = 5
    for block in forged_counts["physical_blocks"]:
        block["observation_count"] = 1
        block["cache_telemetry"]["load_count"] = 1
        block["cache_telemetry"]["payload_cache_hit_count"] = 1
    _reclose_descriptive_cell(forged_counts)
    with pytest.raises(ValueError, match="frozen observation-count"):
        execution._validate_descriptive_cell_record(forged_counts)


def _valid_latency_estimand_records() -> list[dict]:
    return [
        {
            **item,
            "metrics": {
                "ttft": {
                    "confidence_interval_95": [1.1, 1.3],
                    "geometric_mean_speedup": 1.2,
                },
                "time_to_completion": {
                    "confidence_interval_95": [1.0, 1.2],
                    "geometric_mean_speedup": 1.1,
                },
            },
        }
        for item in execution._publication_latency_estimand_projection_design()
    ]


def test_latency_estimand_projection_names_exact_frozen_control_and_treatment_cells():
    estimates = _valid_latency_estimand_records()

    execution._validate_publication_latency_estimate_records(estimates)
    assert estimates[0]["control_cell_id"] == "core-baseline_prefill-8192-c1"
    assert estimates[0]["treatment_cell_id"] == "core-vanilla_prefill-8192-c1"
    assert estimates[9]["control_cell_id"] == "core-vanilla_prefill-16384-c4"
    assert estimates[10]["control_cell_id"] == "auxiliary-storage-disk"
    assert estimates[11]["control_cell_id"] == "auxiliary-storage-disk"
    assert estimates[12]["control_cell_id"] == "core-vanilla_prefill-16384-c4"


def test_latency_estimand_validator_rejects_schema_lineage_and_order_tampering():
    forged = _valid_latency_estimand_records()
    forged[0]["unexpected"] = True
    with pytest.raises(ValueError, match="schema is not closed"):
        execution._validate_publication_latency_estimate_records(forged)

    for field_name, value in (
        ("comparison_family", "storage"),
        ("control_cell_id", "core-vanilla_prefill-8192-c1"),
        ("input_tokens", 16_384),
        ("request_parallelism", 4),
        ("setting_id", "storage-ram"),
        ("treatment_cell_id", "auxiliary-storage-ram"),
    ):
        forged = _valid_latency_estimand_records()
        forged[0][field_name] = value
        with pytest.raises(ValueError, match="frozen-design"):
            execution._validate_publication_latency_estimate_records(forged)

    forged = _valid_latency_estimand_records()
    forged[0], forged[1] = forged[1], forged[0]
    with pytest.raises(ValueError, match="estimand order"):
        execution._validate_publication_latency_estimate_records(forged)

    forged = _valid_latency_estimand_records()
    forged[0]["metrics"]["ttft"]["unexpected"] = 1
    with pytest.raises(ValueError, match="schema is not closed"):
        execution._validate_publication_latency_estimate_records(forged)

    forged = _valid_latency_estimand_records()
    forged[0]["metrics"]["throughput"] = {
        "confidence_interval_95": [1.0, 1.0],
        "geometric_mean_speedup": 1.0,
    }
    with pytest.raises(ValueError, match="schema is not closed"):
        execution._validate_publication_latency_estimate_records(forged)
