import inspect
import json
import math
import random
from copy import deepcopy
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

    with pytest.raises(TypeError, match="PublicationLatencyCollectionAuthorization"):
        execution.aggregate_publication_latency_campaign(  # type: ignore[arg-type]
            {},
            execution_plan_record={},
            qualification_launch_authorization=None,  # type: ignore[arg-type]
            handoff_serving_authorization=None,  # type: ignore[arg-type]
            bf16_handoff_serving_authorization=None,  # type: ignore[arg-type]
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
        execution, "validate_publication_latency_execution_sources", lambda *a: None
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
    qualification_authorization = object()
    q8_authorization = object()

    class Response:
        status = 200

        def __init__(self, run_id):
            self.run_id = run_id

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"run_id": self.run_id}).encode()

    with pytest.raises(TimeoutError, match="lost response"):
        execution.submit_publication_latency_launch_wave(
            execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
            execution_plan_record=plan,
            qualification_launch_authorization=qualification_authorization,
            handoff_serving_authorization=q8_authorization,
            bf16_handoff_serving_authorization=bf16_authorization,
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
    _terminal_record, wave_zero_authorization = (
        execution.collect_publication_latency_launch_wave(
            execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
            execution_plan_record=plan,
            qualification_launch_authorization=qualification_authorization,
            handoff_serving_authorization=q8_authorization,
            bf16_handoff_serving_authorization=bf16_authorization,
            ledger_path=ledger_path,
            wave_index=0,
            submission_authorization=submission_authorization,
        )
    )
    wave_one, _wave_one_submission_authorization = (
        execution.submit_publication_latency_launch_wave(
            execution.DatabricksWorkspaceConfig("https://dbc.example", "token"),
            execution_plan_record=plan,
            qualification_launch_authorization=qualification_authorization,
            handoff_serving_authorization=q8_authorization,
            bf16_handoff_serving_authorization=bf16_authorization,
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
        execution.PublicationBF16HandoffServingAuthorization
    )
    assert hints["handoff_serving_authorization"] is (
        execution.PublicationLatencyHandoffServingAuthorization
    )
    structural_result = PublicationBF16HandoffGenerationResult(
        root=tmp_path,
        source_root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        execution_record_path=tmp_path / "execution.json",
        manifest={},
        record={},
    )
    with pytest.raises(TypeError, match="collector-issued"):
        execution._validated_bf16_generation_binding(  # type: ignore[arg-type]
            structural_result,
            expected_input_bundle_sha256="a" * 64,
            final_artifacts=None,  # type: ignore[arg-type]
        )

    structural_q8_result = PublicationLatencyHandoffGenerationResult(
        root=tmp_path,
        execution_record_path=tmp_path / "execution.json",
        record={},
    )
    with pytest.raises(TypeError, match="ServingAuthorization"):
        execution.require_publication_latency_handoff_serving_authorization(
            structural_q8_result,
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
        "require_publication_latency_handoff_serving_authorization",
        lambda *_args, **_kwargs: q8_result,
    )
    monkeypatch.setattr(
        execution,
        "require_publication_bf16_handoff_serving_authorization",
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
