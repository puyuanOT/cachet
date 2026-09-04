import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import document_kv_cache.gpu_qualification_databricks as qualification_job
from document_kv_cache.databricks_resource_ledger import (
    DatabricksLedgerPrefix,
    create_databricks_cluster_hour_ledger_json,
)
import document_kv_cache.publication_campaign as publication_campaign
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_BF16_HANDOFF_MAX_GPU_HOURS_AT_GATE,
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_FULL_SCORE_GENERATION_MAX_GPU_HOURS_AT_GATE,
    PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS,
    PUBLICATION_CAMPAIGN_ENGINE_VERSION,
    PUBLICATION_CAMPAIGN_ID,
    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_INPUT_TOKEN_SLOTS,
    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_GPU_HOURS_AT_GATE,
    PUBLICATION_CAMPAIGN_LEDGER_ID,
    PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS,
    PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_PRE_BOOTSTRAP_FAILURE_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_PRE_CLUSTER_IDENTITY_FAILURE_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_PRE_FAILED_QUALIFICATION_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_PRE_REJECTED_QUALIFICATION_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_PRE_RUNTIME_LOCK_INDEX_FAILURE_LEDGER_FILE_SHA256,
    PUBLICATION_CAMPAIGN_PRE_RUNTIME_LOCK_INDEX_FAILURE_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_CAMPAIGN_FILE_SHA256,
    PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_LEDGER_FILE_SHA256,
    PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_PRE_SITE_PACKAGES_PATH_FAILURE_LEDGER_FILE_SHA256,
    PUBLICATION_CAMPAIGN_PRE_SITE_PACKAGES_PATH_FAILURE_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL,
    PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_SHA256_BY_JOB,
    PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_UTF8_BYTES_BY_JOB,
    PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_JOB_IDS,
    PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OPAQUE_WORKER_JOB_IDS,
    PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_REASON,
    PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS,
    PUBLICATION_CAMPAIGN_TOTAL_GENERATION_MAX_GPU_HOURS_AT_GATE,
    build_publication_campaign_plan,
    main,
    publication_campaign_latency_timeout_policy,
    publication_campaign_plan_to_record,
    publication_campaign_full_launch_budget_projection,
    validate_publication_campaign_plan_record,
)


CAMPAIGN_LEDGER_ID = PUBLICATION_CAMPAIGN_LEDGER_ID
CAMPAIGN_LEDGER_PATH_SHA256 = PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256
CAMPAIGN_LEDGER_PREFIX = PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX


def test_publication_campaign_is_the_frozen_115_job_design():
    plan = build_publication_campaign_plan(
        "vllm-0271-publication-v1",
        campaign_ledger_id=CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=CAMPAIGN_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
        ),
    )
    record = publication_campaign_plan_to_record(plan)

    assert PUBLICATION_CAMPAIGN_ENGINE_VERSION == "0.27.1"
    assert PUBLICATION_CAMPAIGN_ID == "vllm-0271-publication-v1"
    assert PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS == 5
    assert PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL == 256
    assert record["campaign_ledger_id"] == CAMPAIGN_LEDGER_ID
    assert record["campaign_ledger_path_sha256"] == CAMPAIGN_LEDGER_PATH_SHA256
    assert record["campaign_ledger_prefix"] == CAMPAIGN_LEDGER_PREFIX.to_record()
    assert record["campaign_ledger_prefix"]["reservation_count"] == 236
    assert record["campaign_ledger_prefix"]["submission_receipt_count"] == 98
    assert record["campaign_ledger_prefix"]["terminal_actual_count"] == 236
    assert record["closed_record_sha256"] == PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256
    assert record["campaign_opening_terminal_gpu_hours"] == (
        PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
    )
    assert len(plan.latency_cells) == 90
    assert len({cell.cell_id for cell in plan.latency_cells}) == 90
    assert len({cell.matched_pair_id for cell in plan.latency_cells}) == 45
    assert len(plan.auxiliary_latency_cells) == 25
    assert len({cell.cell_id for cell in plan.auxiliary_latency_cells}) == 25
    assert PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS == 115
    assert {cell.setting_id for cell in plan.auxiliary_latency_cells} == {
        "precision-bf16",
        "storage-disk",
        "storage-ram",
        "storage-uc",
        "hardware-a10g",
    }
    assert all(
        cell.reference_core_cell_id
        == (
            f"block-{cell.deployment_block:02d}-storage-disk"
            if cell.setting_id in {"storage-ram", "storage-uc"}
            else f"block-{cell.deployment_block:02d}-16k-c4-vanilla"
        )
        for cell in plan.auxiliary_latency_cells
    )
    assert record["budget"] == {
        "active_reservation_hour_cap": 900.0,
        "aggregate_gpu_hour_cap": 1024.0,
        "auxiliary_latency_jobs": 25,
        "bf16_handoff_generation": {
            "absolute_slot_envelope_bytes": 288 * 1024**3,
            "cache_prefix_generation_tokens": 2_091_797,
            "included_in_aggregate_gpu_hour_cap": True,
            "max_gpu_hours_at_min_throughput": (
                PUBLICATION_CAMPAIGN_BF16_HANDOFF_MAX_GPU_HOURS_AT_GATE
            ),
            "payload_bytes": 308_448_018_432,
            "payload_gib": pytest.approx(287.26460349559784),
            "producer_gpu_tasks": 16,
            "producer_task_timeout_seconds": 18_000,
            "worst_case_reserved_gpu_hours": 80.0,
        },
        "core_latency_jobs": 90,
        "cpu_control_plane": {
            "data_security_mode": "SINGLE_USER",
            "databricks_node_type_id": "c5d.4xlarge",
            "full_score_tree_closure": {
                "actions_per_wave": ["producer_ready", "consumer_evidence"],
                "job_count": 20,
                "task_timeout_seconds": 7_200,
                "wave_count": 10,
            },
            "gpu_tasks": 0,
            "handoff_tree_closure": {
                "job_count": 2,
                "stages": ["q8", "bf16"],
                "task_timeout_seconds": 43_200,
            },
            "included_in_gpu_hour_ledger": False,
            "latency_source_closure": {
                "job_count": 1,
                "task_timeout_seconds": 7_200,
            },
            "max_retries": 0,
            "num_workers": 0,
            "single_node": True,
            "spark_version": "15.4.x-cpu-ml-scala2.12",
            "timeout_upper_bound_cpu_node_hours": 66.0,
            "total_job_count": 23,
        },
        "full_score_execution": {
            "cache_prefix_generation_tokens": 63_455_746,
            "consumer_timeout_upper_bound_gpu_hours": 960.0,
            "example_count": 83_653,
            "execution_plan_sha256": (
                "88315f8bcae2317659fea61c05b7e5c56e7e56dd1330368fdccfe816c453ad84"
            ),
            "generation_max_gpu_hours_at_min_throughput": (
                PUBLICATION_CAMPAIGN_FULL_SCORE_GENERATION_MAX_GPU_HOURS_AT_GATE
            ),
            "inventory_sha256": (
                "e19fefa656d8975946b13bb9987f801ec486c4bfde5e9d5ed82a877e80676b11"
            ),
            "live_p90_admission_required_after_each_matched_wave": True,
            "natural_prompt_inference_tokens": 66_448_937,
            "phase_count": 20,
            "producer_and_consumer_phases_per_wave": 2,
            "producer_timeout_upper_bound_gpu_hours": 960.0,
            "shard_count": 160,
            "shard_plan_sha256": (
                "605c15ef5317bb0b6d6f6a4057dbacbd97ae31af94a3d497585a88c138c9ba84"
            ),
            "task_timeout_seconds": 21_600,
            "tasks_per_phase": 16,
            "wave_count": 10,
            "worst_case_reserved_gpu_hours_per_phase": 96.0,
        },
        "full_launch_min_generation_tokens_per_second": 35.0,
        "generation_workload_total": {
            "cache_prefix_generation_tokens": 72_871_510,
            "max_gpu_hours_at_min_throughput": (
                PUBLICATION_CAMPAIGN_TOTAL_GENERATION_MAX_GPU_HOURS_AT_GATE
            ),
            "non_generation_gpu_hours_available_after_opening_balance_and_headroom": (
                pytest.approx(250.2645542063492)
            ),
            "scope": [
                "latency_q8_handoffs",
                "latency_bf16_handoffs",
                "full_score_q8_handoffs",
            ],
        },
        "gpu_qualification": {
            "all_jobs_required": True,
            "job_count": 14,
            "max_retries": 0,
            "task_timeout_seconds": 14_400,
            "worst_case_reserved_gpu_hours": 56.0,
        },
        "latency_handoff_generation": {
            "accounting_input_token_slots": 7_340_032,
            "cache_prefix_generation_tokens": 7_323_967,
            "coordinator_gpu_hours": 0.0,
            "included_in_aggregate_gpu_hour_cap": True,
            "max_gpu_hours_at_min_throughput": (
                PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_GPU_HOURS_AT_GATE
            ),
            "max_persistent_gpu_workers": 16,
            "producer_gpu_tasks": 16,
            "producer_submission_shape": ("independent_single_gpu_single_task_runs"),
            "producer_task_timeout_seconds": 18000,
            "reservation_reconciliation": "per_producer_attempt",
            "throughput_scope": (
                "generation_plus_worker_result_durable_write_per_gpu_second"
            ),
            "worst_case_reserved_gpu_hours": 80.0,
        },
        "max_parallel_jobs": 16,
        "max_latency_wave_reserved_gpu_hours": 192.0,
        "latency_timeout_upper_bound": {
            "completion_guaranteed_by_timeout_bounds": False,
            "gpu_hours": 660,
            "job_counts_by_timeout_hours": {
                "4": 65,
                "6": 20,
                "8": 20,
                "12": 10,
            },
            "launch_policy": "terminal_actual_and_hard_headroom_gated",
        },
        "opening_terminal_gpu_hours": pytest.approx(71.39012833333337),
        "total_latency_jobs": 115,
        "unreserved_headroom_hours": 124.0,
    }
    assert record["storage_request_protocol"] == {
        "datasets": ["biography", "hotpotqa", "musique", "niah"],
        "deployment_blocks": 5,
        "examples_per_dataset": 2,
        "matched_settings": ["storage-disk", "storage-ram", "storage-uc"],
        "payload_cache_max_bytes": 16 * 1024**3,
        "selection": {
            "caller_selectable": False,
            "domain": "cachet.publication.storage_subset.selection.v1",
            "rule": "lowest_domain_separated_sha256_per_dataset",
            "source_examples_per_dataset": 32,
        },
        "repeats_per_example": 32,
        "request_count_per_cell": 256,
        "request_parallelism": 4,
    }
    assert record["latency_timeout_policy"] == {
        "auxiliary_c4_hours": 4,
        "core_hours_by_context_and_concurrency": {
            "8k": {"c1": 6, "c2": 4, "c4": 4},
            "16k": {"c1": 8, "c2": 6, "c4": 4},
            "32k": {"c1": 12, "c2": 8, "c4": 4},
        },
        "max_retries": 0,
    }
    assert record["latency_timeout_policy"] == (
        publication_campaign_latency_timeout_policy()
    )
    assert record["analysis"]["experimental_units"] == {
        "core_method": "matched_fresh_cluster_baseline_vanilla_pair",
        "precision_hardware": "matched_16k_c4_core_pair_plus_bf16_and_a10g_wave",
        "storage": "matched_fresh_cluster_disk_ram_uc_trio",
    }
    assert record["analysis"]["opening_ledger_provenance"] == {
        "prefix_before_rejected_gpu_qualification": (
            PUBLICATION_CAMPAIGN_PRE_REJECTED_QUALIFICATION_LEDGER_PREFIX.to_record()
        ),
        "rejected_gpu_qualification": {
            "actual_gpu_hours": 0.0,
            "evidence_closed_record_sha256": (
                "6102fe08f1ea3385ce862201c3b6b3351396315aba35bda56309bc240554f083"
            ),
            "evidence_file_sha256": (
                "1f6a76369658f6dd39021dc72807345194cd43e2ecd746e831846391dc4e5a2b"
            ),
            "failed_before_run_creation": True,
            "http_status": 400,
            "observed_parameters_json_bytes": 18_292,
            "plan_sha256": (
                "b0bf7fdc182a099fae8f7d2fef1441974f69e49e840601f20397032709baf9f4"
            ),
            "reservation_count_delta": 14,
            "remote_active_runs_observed": 0,
            "server_parameters_json_limit_bytes": 10_000,
            "submission_receipt_count_delta": 0,
            "terminal_actual_count_delta": 14,
            "terminal_state": "failed",
            "verification_source": "legacy_manual",
        },
        "prefix_before_failed_live_gpu_qualification": (
            PUBLICATION_CAMPAIGN_PRE_FAILED_QUALIFICATION_LEDGER_PREFIX.to_record()
        ),
        "failed_live_gpu_qualification": {
            "actual_gpu_hours": 1.5991302777777774,
            "data_security_mode": "NONE",
            "failed_before_run_creation": False,
            "failure_class": "unity_catalog_volume_access",
            "failure_reason": (
                "qualification payload used NONE access mode and could not resolve "
                "Unity Catalog Volume bootstrap; remaining pending jobs canceled "
                "after first failures"
            ),
            "plan_sha256": (
                "ebfeaf53cfa9c74400be59546b391b77ebde4e85defa1f1b11bc4b4255c80341"
            ),
            "reconciliation_manifest_closed_record_sha256": (
                "644048afcd8f478aa6ba2776be97f4e6fce4396ddf853001c3d200cfbbd259eb"
            ),
            "reservation_count_delta": 14,
            "run_creation_count": 14,
            "submission_receipt_count_delta": 14,
            "terminal_actual_count_delta": 14,
            "terminal_result_state_counts": {"CANCELED": 7, "FAILED": 7},
            "verification_source": "direct_runs_get",
        },
        "prefix_before_bootstrap_failure_gpu_qualification": (
            PUBLICATION_CAMPAIGN_PRE_BOOTSTRAP_FAILURE_LEDGER_PREFIX.to_record()
        ),
        "bootstrap_failure_gpu_qualification": {
            "actual_cluster_duration_seconds": 4_585.718,
            "actual_gpu_hours": 1.2738105555555554,
            "data_security_mode": "SINGLE_USER",
            "failed_before_run_creation": False,
            "failure_class": "spark_python_task_missing_dunder_file",
            "failure_reason": (
                "all fourteen tasks failed before package installation because "
                "the reviewed bootstrap referenced undefined __file__ under "
                "Databricks spark_python_task execution"
            ),
            "plan_sha256": (
                "2cf4ef1092a435c1e713f2a94115021ea7069ab6295d18ce5fcb5d4a479ce997"
            ),
            "reconciliation_manifest_closed_record_sha256": (
                "8c7623aa2618066ea0ccedcba1d35a340308da04aaa040f89364bc4ea3d1b71c"
            ),
            "reconciliation_manifest_file_sha256": (
                "1d0246ece1d6f844420d22a26b729d3f0d971ca0b30c0bf1ef0b5a84dcf6f360"
            ),
            "reservation_count_delta": 14,
            "reviewed_runner_sha256": (
                "f5ee833621428d630df1a59952a485d4ac55cabf987186d98a40274a2cf8a958"
            ),
            "run_creation_count": 14,
            "submission_receipt_count_delta": 14,
            "terminal_actual_count_delta": 14,
            "terminal_life_cycle_state_counts": {"INTERNAL_ERROR": 14},
            "terminal_result_state_counts": {"FAILED": 14},
            "verification_source": "direct_runs_get_and_runs_get_output",
        },
        "prefix_before_cluster_identity_failure_gpu_qualification": (
            PUBLICATION_CAMPAIGN_PRE_CLUSTER_IDENTITY_FAILURE_LEDGER_PREFIX.to_record()
        ),
        "cluster_identity_failure_gpu_qualification": {
            "actual_cluster_duration_seconds": 4_564.259,
            "actual_gpu_hours": 1.2678497222222225,
            "data_security_mode": "SINGLE_USER",
            "expected_error": (
                "RuntimeError: Databricks cluster identity is unavailable; expected "
                "DATABRICKS_CLUSTER_ID or DB_CLUSTER_ID"
            ),
            "failed_before_run_creation": False,
            "failure_class": "databricks_cluster_identity_unavailable",
            "failure_reason": (
                "qualification bootstrap could not resolve Databricks cluster identity"
            ),
            "plan_sha256": (
                "d6f7619f6a70311fac571b31bedc7974e756a1679218cf63b76a7e7ceb91ebec"
            ),
            "reconciled_ledger_file_sha256": (
                PUBLICATION_CAMPAIGN_PRE_RUNTIME_LOCK_INDEX_FAILURE_LEDGER_FILE_SHA256
            ),
            "reconciliation_manifest_closed_record_sha256": (
                "fbb1fd4250b3fc62b58778047b12fe3775e6cffbc8641b38a00c721a9d4c768d"
            ),
            "reconciliation_manifest_file_sha256": (
                "06c527102283bb379ecb26a345e76467d7e1614771d9a3c8313e9ebe6d941cf9"
            ),
            "reservation_count_delta": 14,
            "reviewed_runner_sha256": (
                "04cfe3a16200f011710317d829b7c52c0e4ca12f95fd8d277c949e7d6856d5b0"
            ),
            "run_creation_count": 14,
            "runs_get_output_keys": [
                "error",
                "error_trace",
                "logs",
                "logs_truncated",
                "metadata",
            ],
            "single_user_name": "pliu@opentable.com",
            "submission_receipt_count_delta": 14,
            "task_life_cycle_state_counts": {"TERMINATED": 14},
            "task_result_state_counts": {"FAILED": 14},
            "terminal_actual_count_delta": 14,
            "terminal_life_cycle_state_counts": {"INTERNAL_ERROR": 14},
            "terminal_prefix_sha256": (
                "376114c27f35725bab5418969d28a77d4a3600dba44d049b597512142856d86f"
            ),
            "terminal_result_state_counts": {"FAILED": 14},
            "verification_source": "direct_runs_get_and_runs_get_output",
        },
        "prefix_before_runtime_lock_index_failure_gpu_qualification": (
            PUBLICATION_CAMPAIGN_PRE_RUNTIME_LOCK_INDEX_FAILURE_LEDGER_PREFIX.to_record()
        ),
        "runtime_lock_index_failure_gpu_qualification": {
            "actual_cluster_duration_seconds": 7_754.755,
            "actual_gpu_hours": 2.1540986111111113,
            "data_security_mode": "SINGLE_USER",
            "evidence_tree_byte_count": 1_564_133,
            "evidence_tree_file_count": 29,
            "evidence_tree_sha256": (
                "5016ed50001b77b77f329e858c01b1a65c5e927f1c55eec7fbc01208d8f25886"
            ),
            "failed_before_run_creation": False,
            "failure_class": "pip_requirements_file_index_precedence",
            "failure_reason": (
                "pip requirements-file index precedence omitted the PyTorch CU129 "
                "index and prevented hash-locked torch resolution"
            ),
            "normalized_error_sha256": (
                "7544cab6366fc1813af8d04da00a8a1f76f1098e3b06c738d8ff8ddd392ae235"
            ),
            "plan_sha256": (
                "f991036176d59df70f0e339be4eb4a67a7c03a51536f62bf440df1ac72fd0e33"
            ),
            "predicted_terminal_prefix_sha256": (
                "381ed88dfca75a17cf11b09b7e3dedb435328e518e8f1f0f0d9591be27796f26"
            ),
            "reconciled_ledger_file_sha256": (
                PUBLICATION_CAMPAIGN_PRE_SITE_PACKAGES_PATH_FAILURE_LEDGER_FILE_SHA256
            ),
            "reconciliation_manifest_closed_record_sha256": (
                "2ee650e0e05ea059bd9f552d6975149c05cbda6dc8d3a715a73594913f078b29"
            ),
            "reconciliation_manifest_file_sha256": (
                "e0f56f1250c4ce213d1a8ba0384ccdad1a1b38fb964c1b6bfcf5729006150455"
            ),
            "reservation_count_delta": 14,
            "reviewed_runner_sha256": (
                "04cfe3a16200f011710317d829b7c52c0e4ca12f95fd8d277c949e7d6856d5b0"
            ),
            "run_creation_count": 14,
            "runs_get_output_keys": [
                "error",
                "error_trace",
                "logs",
                "logs_truncated",
                "metadata",
            ],
            "single_user_name": "pliu@opentable.com",
            "submission_receipt_count_delta": 14,
            "task_life_cycle_state_counts": {"TERMINATED": 14},
            "task_result_state_counts": {"FAILED": 14},
            "terminal_actual_count_delta": 14,
            "terminal_life_cycle_state_counts": {"INTERNAL_ERROR": 14},
            "terminal_prefix_sha256": (
                "381ed88dfca75a17cf11b09b7e3dedb435328e518e8f1f0f0d9591be27796f26"
            ),
            "terminal_result_state_counts": {"FAILED": 14},
            "torch_resolution_log_marker": (
                "No matching distribution found for torch==2.13.0+cu129"
            ),
            "verification_source": "direct_runs_get_and_runs_get_output",
        },
        "prefix_before_site_packages_path_failure_gpu_qualification": (
            PUBLICATION_CAMPAIGN_PRE_SITE_PACKAGES_PATH_FAILURE_LEDGER_PREFIX.to_record()
        ),
        "site_packages_path_failure_gpu_qualification": {
            "actual_cluster_duration_seconds": 11_498.35,
            "actual_gpu_hours": 3.193986111111111,
            "data_security_mode": "SINGLE_USER",
            "evidence_tree_byte_count": 1_945_499,
            "evidence_tree_file_count": 29,
            "evidence_tree_sha256": (
                "2c555ea534fc3d41d3bc998fcaff8f07aedf42e1872200e39f9ed46796081607"
            ),
            "failed_before_run_creation": False,
            "failed_before_sentinel_worker_launch": True,
            "failure_class": "nonexistent_debian_site_packages_scheme_path",
            "failure_reason": (
                "all fourteen hash-locked qualification runtimes installed and "
                "verified, then failed before sentinel worker launch because the "
                "site-packages read-only freezer rejected a nonexistent Debian "
                "local dist-packages scheme path reported by site.getsitepackages()"
            ),
            "normalized_error_sha256": (
                "8937fb907ae789c647754b2bbe9dbc4d9e167b67b8e437613260373b658c0da3"
            ),
            "plan_file_sha256": (
                "c63521b29233addc1c5ab4435dfa0d639135765bce7a54298c0b0b1200741651"
            ),
            "plan_sha256": (
                "be4cb0e80e17c99d9c4bd8abb89b24efb6e1202072fb734c739d322812218c9c"
            ),
            "predicted_terminal_prefix_sha256": (
                "a71cee32c1ae056d7db7c72c70fa72bcf5622d8a3ae6d72590c4435bb9db4af9"
            ),
            "reconciled_ledger_file_sha256": (
                PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_LEDGER_FILE_SHA256
            ),
            "reconciliation_manifest_closed_record_sha256": (
                "a685849f6446063bdd5b220cd3ac5218c6e49a1e2d8487acac36316537b35eb7"
            ),
            "reconciliation_manifest_file_sha256": (
                "2996e67b6c6305544c11231266500dcb9c53aa2bbc701fa6d6e626299c2ab06e"
            ),
            "reservation_count_delta": 14,
            "reviewed_runner_sha256": (
                "ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc"
            ),
            "run_creation_count": 14,
            "runs_get_output_keys": [
                "error",
                "error_trace",
                "logs",
                "logs_truncated",
                "metadata",
            ],
            "single_user_name": "pliu@opentable.com",
            "submission_receipt_count_delta": 14,
            "task_life_cycle_state_counts": {"TERMINATED": 14},
            "task_result_state_counts": {"FAILED": 14},
            "terminal_actual_count_delta": 14,
            "terminal_life_cycle_state_counts": {"INTERNAL_ERROR": 14},
            "terminal_prefix_sha256": (
                "a71cee32c1ae056d7db7c72c70fa72bcf5622d8a3ae6d72590c4435bb9db4af9"
            ),
            "terminal_result_state_counts": {"FAILED": 14},
            "verification_source": "direct_runs_get_and_runs_get_output",
        },
        "prefix_before_runtime_observation_and_worker_subprocess_failure_gpu_qualification": (
            PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_LEDGER_PREFIX.to_record()
        ),
        "runtime_observation_and_worker_subprocess_failure_gpu_qualification": {
            "actual_cluster_duration_seconds": 12_410.279,
            "actual_gpu_hours": 3.447299722222222,
            "data_security_mode": "SINGLE_USER",
            "evidence_tree_byte_count": 1_828_218,
            "evidence_tree_file_count": 29,
            "evidence_tree_sha256": (
                "bb6636f3b9bdf5afae0b7d1beb97f5f3192017ba5b04abb651f2a389889aa57f"
            ),
            "failed_before_run_creation": False,
            "failure_class": (
                "post_success_runtime_observation_and_opaque_worker_subprocess"
            ),
            "failure_reason": (
                PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_REASON
            ),
            "normalized_observer_error_sha256": (
                "3662915979987aef1fe4bcf9e0e62f06c67992ee73da679e44f6b6a261e634f5"
            ),
            "normalized_worker_error_sha256": (
                "3f1ddd73298cd46347cf57b84d6cf22f7d6e98802b50ded9457d7a999563786b"
            ),
            "observer_job_count": 2,
            "observer_job_ids": list(
                PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_JOB_IDS
            ),
            "observer_jobs_returned_measurements": True,
            "opaque_worker_job_count": 12,
            "opaque_worker_job_ids": list(
                PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OPAQUE_WORKER_JOB_IDS
            ),
            "opaque_worker_underlying_causes_known": False,
            "plan_file_sha256": (
                "fe59e32c44ab50f91bae5114a587268d44ebb9acfba74500aedb66158e2541b7"
            ),
            "plan_sha256": (
                "c0bede45ea211798c9a5eb31010a91074ded70e370f8ea4fcbeb59b3b9f95598"
            ),
            "predecessor_campaign_closed_record_sha256": (
                PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_CAMPAIGN_CLOSED_RECORD_SHA256
            ),
            "predecessor_campaign_file_sha256": (
                PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_CAMPAIGN_FILE_SHA256
            ),
            "predicted_terminal_prefix_sha256": (
                "22ac65492fa0871f528552cfcae0bd6332b1429cd9fc2e92c373c5e534202d4a"
            ),
            "raw_error_sha256_by_job": dict(
                PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_SHA256_BY_JOB
            ),
            "raw_error_utf8_bytes_by_job": dict(
                PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_UTF8_BYTES_BY_JOB
            ),
            "reconciled_accounted_gpu_hours": 67.93033611111115,
            "reconciled_active_reserved_gpu_hours": 0.0,
            "reconciled_ledger_file_sha256": (
                publication_campaign.PUBLICATION_CAMPAIGN_PRE_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_CAMPAIGN_OPENING_LEDGER_FILE_SHA256
            ),
            "reconciled_remaining_gpu_hours": 956.0696638888888,
            "reconciled_terminal_actual_gpu_hours": 67.93033611111115,
            "reconciliation_manifest_closed_record_sha256": (
                "6c4cca0ec4fbcf4ccb434573f965eeb8022909ce5bdd6afdf31d61085807fa9b"
            ),
            "reconciliation_manifest_file_sha256": (
                "53fd4b076a642101790d21ebbc03b1eb7e609428c2ccd7eafb8cbad5a9a3a112"
            ),
            "reservation_count_delta": 14,
            "reviewed_runner_sha256": (
                "ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc"
            ),
            "run_creation_count": 14,
            "runs_get_output_keys": [
                "error",
                "error_trace",
                "logs",
                "logs_truncated",
                "metadata",
            ],
            "single_user_name": "pliu@opentable.com",
            "submission_receipt_count_delta": 14,
            "task_life_cycle_state_counts": {"TERMINATED": 14},
            "task_result_state_counts": {"FAILED": 14},
            "terminal_actual_count_delta": 14,
            "terminal_life_cycle_state_counts": {"INTERNAL_ERROR": 14},
            "terminal_prefix_sha256": (
                "22ac65492fa0871f528552cfcae0bd6332b1429cd9fc2e92c373c5e534202d4a"
            ),
            "terminal_result_state_counts": {"FAILED": 14},
            "verification_source": "direct_runs_get_and_runs_get_output",
        },
        "prefix_before_mixed_sentinel_and_result_validation_failure_gpu_qualification": (
            publication_campaign.PUBLICATION_CAMPAIGN_PRE_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_CAMPAIGN_OPENING_LEDGER_PREFIX.to_record()
        ),
        "mixed_sentinel_and_result_validation_failure_gpu_qualification": {
            "actual_cluster_duration_seconds": 12_455.252,
            "actual_gpu_hours": 3.459792222222222,
            "batch_marker_closed_record_sha256": (
                "249aeba5bed1f0283db74a612998cb6941713e0ed8844bfa800e56886bb29389"
            ),
            "batch_marker_file_sha256": (
                "b4f2cb8ea2c1f637c31b3745217670622657845f3a893ea2e7a5f78025c125d8"
            ),
            "data_security_mode": "SINGLE_USER",
            "evidence_tree_byte_count": 2_094_892,
            "evidence_tree_file_count": 29,
            "evidence_tree_sha256": (
                "7455fa1e30356bb79ccb75a8dbe24df32f33a365141505e0270eb13c7f39b71d"
            ),
            "failed_before_run_creation": False,
            "failure_categories": {
                "auto_backend_flashinfer_array_array_type_error_engine_initialization": {
                    "job_count": 2,
                    "job_ids": list(
                        publication_campaign.PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FLASHINFER_ARRAY_ARRAY_TYPE_ERROR_JOB_IDS
                    ),
                },
                "forced_runtime_handoff_unresolved_native_object": {
                    "job_count": 2,
                    "job_ids": list(
                        publication_campaign.PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_UNRESOLVED_NATIVE_OBJECT_JOB_IDS
                    ),
                },
                "post_measurement_vllm_version_contract_mismatch": {
                    "job_count": 2,
                    "job_ids": list(
                        publication_campaign.PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_VERSION_CONTRACT_MISMATCH_JOB_IDS
                    ),
                },
                "sentinel_layout_conflict": {
                    "job_count": 8,
                    "job_ids": list(
                        publication_campaign.PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_SENTINEL_LAYOUT_CONFLICT_JOB_IDS
                    ),
                },
            },
            "failure_class": "mixed_sentinel_and_result_validation",
            "failure_reason": (
                publication_campaign.PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_REASON
            ),
            "ledger_lineage": {
                "batch_reserved_prefix": (
                    publication_campaign.PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_BATCH_RESERVED_LEDGER_PREFIX.to_record()
                ),
                "campaign_predecessor_prefix": (
                    publication_campaign.PUBLICATION_CAMPAIGN_PRE_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_CAMPAIGN_OPENING_LEDGER_PREFIX.to_record()
                ),
                "receipt_predecessor_prefix": (
                    publication_campaign.PUBLICATION_CAMPAIGN_PRE_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_RECONCILIATION_PREDECESSOR_LEDGER_PREFIX.to_record()
                ),
                "terminal_prefix": CAMPAIGN_LEDGER_PREFIX.to_record(),
            },
            "logs_truncated_false_job_count": 14,
            "phase_lease_closed_record_sha256": (
                "92add29ae13ccafb848a5075a24fb99410309f244fc842bd7883907d24a217f2"
            ),
            "phase_lease_file_sha256": (
                "ea9f9ec3c415001ef0cf65e9d3673950f7a49d7a2733eb76bf2668bdf7d80344"
            ),
            "plan_file_sha256": (
                "e19e9b173ad8e2705d11cfbd637aa3702a98e37d827e7d1489460c1462c5a649"
            ),
            "plan_sha256": (
                "694441bffc253141156f9c808666112d39bb5829d22825d1d88c93ab47a5e830"
            ),
            "predecessor_campaign_closed_record_sha256": (
                "5f90b531b30ac6f4b29e0151d688a005b0377b205ca39645376d7d43aef5e305"
            ),
            "predecessor_campaign_file_byte_count": 66_635,
            "predecessor_campaign_file_sha256": (
                "eb306f9a8be50730bfef81121c2a83ebec7e50e89386addb7f77ce6001bcd85f"
            ),
            "predecessor_campaign_opening_ledger_file_sha256": (
                "38677fff866e0a7268398c4b616b4be968df3a8191381db74ebd8fcb71af50ef"
            ),
            "predecessor_campaign_opening_ledger_prefix": (
                publication_campaign.PUBLICATION_CAMPAIGN_PRE_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_CAMPAIGN_OPENING_LEDGER_PREFIX.to_record()
            ),
            "predecessor_campaign_opening_terminal_gpu_hours": (67.93033611111115),
            "predicted_terminal_prefix_sha256": (
                "07b9663e42c2dd8040f689d08fabdd6d7eefaf25f8f1decedc23af683e0011c7"
            ),
            "raw_error_sha256_by_job": dict(
                publication_campaign.PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ERROR_SHA256_BY_JOB
            ),
            "raw_error_utf8_bytes_by_job": dict(
                publication_campaign.PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ERROR_UTF8_BYTES_BY_JOB
            ),
            "receipt_predecessor_accounted_gpu_hours": 123.93033611111115,
            "receipt_predecessor_active_reserved_gpu_hours": 56.0,
            "receipt_predecessor_ledger_file_sha256": (
                "1ea82ff83621d97448cf79138061d933a211ca2d7b619d13cc27c6357d1831be"
            ),
            "receipt_predecessor_ledger_byte_count": 213_024,
            "receipt_predecessor_remaining_gpu_hours": 900.0696638888888,
            "receipt_predecessor_terminal_actual_gpu_hours": 67.93033611111115,
            "reconciled_accounted_gpu_hours": 71.39012833333337,
            "reconciled_active_reserved_gpu_hours": 0.0,
            "reconciled_ledger_file_sha256": (
                "784a43eafec2f6d6086b4258959b308043e183f361218463be14dea3702bd62d"
            ),
            "reconciled_ledger_byte_count": 220_426,
            "reconciled_remaining_gpu_hours": 952.6098716666667,
            "reconciled_terminal_actual_gpu_hours": 71.39012833333337,
            "reconciliation_manifest_closed_record_sha256": (
                "13ad4eabd10bde1b5c7e0aa7b9721dd3bd8fbe57f6c20204093749df8d84954f"
            ),
            "reconciliation_manifest_file_sha256": (
                "a6e0c985d64b0072776dd1247094600d81b885cfe4a3fb0f6418e8b811134304"
            ),
            "reservation_count_delta": 14,
            "reviewed_runner_sha256": (
                "ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc"
            ),
            "run_creation_count": 14,
            "runs_get_output_keys": [
                "error",
                "error_trace",
                "logs",
                "logs_truncated",
                "metadata",
            ],
            "single_user_name": "pliu@opentable.com",
            "submission_receipt_count_delta": 14,
            "submit_receipt_tree_byte_count": 22_468,
            "submit_receipt_tree_file_count": 16,
            "submit_receipt_tree_sha256": (
                "81817e833e6878ff5bfd45fff2a94ffafb341d7acecc6e4f7212d268646f8f72"
            ),
            "task_attempt_number_counts": {"0": 14},
            "task_intervals_by_job": {
                job_id: {
                    "duration_seconds": (task_end - task_start) / 1_000.0,
                    "task_end_time_epoch_milliseconds": task_end,
                    "task_start_time_epoch_milliseconds": task_start,
                }
                for job_id, task_start, task_end in (
                    publication_campaign.PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_TASK_INTERVAL_EPOCH_MILLISECONDS_BY_JOB
                )
            },
            "task_life_cycle_state_counts": {"TERMINATED": 14},
            "task_repair_history_entry_count": 0,
            "task_result_state_counts": {"FAILED": 14},
            "terminal_actual_count_delta": 14,
            "terminal_life_cycle_state_counts": {"INTERNAL_ERROR": 14},
            "terminal_prefix_sha256": (
                "07b9663e42c2dd8040f689d08fabdd6d7eefaf25f8f1decedc23af683e0011c7"
            ),
            "terminal_result_state_counts": {"FAILED": 14},
            "verification_source": "direct_runs_get_and_runs_get_output",
        },
        "retained_opening_prefix": CAMPAIGN_LEDGER_PREFIX.to_record(),
    }
    assert (
        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_SHA256_BY_JOB
        == qualification_job.GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_SHA256_BY_JOB
    )
    assert (
        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_UTF8_BYTES_BY_JOB
        == qualification_job.GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_UTF8_BYTES_BY_JOB
    )
    assert record["full_score_program"] == {
        "cache_prefix_generation_tokens": 63_455_746,
        "complete_population_required": True,
        "datasets": ["biography", "hotpotqa", "musique", "niah"],
        "example_count": 83_653,
        "max_natural_prompt_tokens": 32768,
        "max_parallel_workers": 16,
        "methods": ["baseline_prefill", "vanilla_prefill"],
        "natural_prompt_inference_tokens": 66_448_937,
        "paired_example_bootstrap_draws": 20000,
        "passes_per_method": 1,
        "padding": False,
        "quality_preservation_gate": False,
        "shard_count": 160,
        "streaming_lifecycle": [
            "generate_q8_kv",
            "baseline_inference",
            "vanilla_inference",
            "validate_paired_outputs",
            "delete_ephemeral_q8_kv",
        ],
        "tokenizer_truncation": False,
        "unsupported_datasets_remain_na": ["longbench_v2", "ruler"],
        "wave_count": 10,
    }
    validate_publication_campaign_plan_record(record)


def test_publication_campaign_budget_includes_latency_handoff_generation():
    assert PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_INPUT_TOKEN_SLOTS == 7_340_032
    assert PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_GPU_HOURS_AT_GATE == pytest.approx(
        58.12672222222223
    )
    projection = publication_campaign_full_launch_budget_projection(
        latency_handoff_generation_tokens_per_second=35.0,
        latency_handoff_generation_gpu_hours=(
            PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_GPU_HOURS_AT_GATE
        ),
        other_terminal_gpu_hours=1.0,
        current_active_reserved_gpu_hours=10.0,
        proposed_full_launch_reserved_gpu_hours=700.0,
    )
    assert projection["projected_accounted_gpu_hours"] == pytest.approx(
        769.1267222222222
    )
    assert projection["projected_active_reserved_gpu_hours"] == 710.0
    assert projection["projected_unreserved_gpu_hours"] > 124.0

    with pytest.raises(ValueError, match="35-token/s"):
        publication_campaign_full_launch_budget_projection(
            latency_handoff_generation_tokens_per_second=34.999,
            latency_handoff_generation_gpu_hours=1.0,
            other_terminal_gpu_hours=0.0,
            current_active_reserved_gpu_hours=0.0,
            proposed_full_launch_reserved_gpu_hours=0.0,
        )
    with pytest.raises(ValueError, match="900-hour"):
        publication_campaign_full_launch_budget_projection(
            latency_handoff_generation_tokens_per_second=40.0,
            latency_handoff_generation_gpu_hours=1.0,
            other_terminal_gpu_hours=0.0,
            current_active_reserved_gpu_hours=899.0,
            proposed_full_launch_reserved_gpu_hours=2.0,
        )
    with pytest.raises(ValueError, match="preserve 124"):
        publication_campaign_full_launch_budget_projection(
            latency_handoff_generation_tokens_per_second=40.0,
            latency_handoff_generation_gpu_hours=1.0,
            other_terminal_gpu_hours=900.0,
            current_active_reserved_gpu_hours=0.0,
            proposed_full_launch_reserved_gpu_hours=0.0,
        )


def test_publication_campaign_rejects_protocol_and_digest_tampering():
    record = publication_campaign_plan_to_record(
        build_publication_campaign_plan(
            "vllm-0271-publication-v1",
            campaign_ledger_id=CAMPAIGN_LEDGER_ID,
            campaign_ledger_path_sha256=CAMPAIGN_LEDGER_PATH_SHA256,
            campaign_ledger_prefix=CAMPAIGN_LEDGER_PREFIX,
            campaign_opening_terminal_gpu_hours=(
                PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
            ),
        )
    )

    tampered = deepcopy(record)
    tampered["latency_cells"][0]["request_count"] = 64
    with pytest.raises(ValueError, match="closed_record_sha256"):
        validate_publication_campaign_plan_record(tampered)

    tampered = deepcopy(record)
    tampered["unexpected"] = True
    with pytest.raises(ValueError, match="closed schema"):
        validate_publication_campaign_plan_record(tampered)

    resealed_tampers = []
    tampered = deepcopy(record)
    mixed = tampered["analysis"]["opening_ledger_provenance"][
        "mixed_sentinel_and_result_validation_failure_gpu_qualification"
    ]
    mixed["failure_reason"] += "."
    resealed_tampers.append(tampered)
    tampered = deepcopy(record)
    mixed = tampered["analysis"]["opening_ledger_provenance"][
        "mixed_sentinel_and_result_validation_failure_gpu_qualification"
    ]
    mixed["failure_categories"]["sentinel_layout_conflict"]["job_count"] = 7
    resealed_tampers.append(tampered)
    tampered = deepcopy(record)
    mixed = tampered["analysis"]["opening_ledger_provenance"][
        "mixed_sentinel_and_result_validation_failure_gpu_qualification"
    ]
    mixed["task_intervals_by_job"]["aws-g6-l4-packed-page-roundtrip"][
        "duration_seconds"
    ] += 0.001
    resealed_tampers.append(tampered)
    tampered = deepcopy(record)
    mixed = tampered["analysis"]["opening_ledger_provenance"][
        "mixed_sentinel_and_result_validation_failure_gpu_qualification"
    ]
    mixed["raw_error_sha256_by_job"]["aws-g6-l4-packed-page-roundtrip"] = "0" * 64
    resealed_tampers.append(tampered)
    tampered = deepcopy(record)
    mixed = tampered["analysis"]["opening_ledger_provenance"][
        "mixed_sentinel_and_result_validation_failure_gpu_qualification"
    ]
    mixed["ledger_lineage"]["receipt_predecessor_prefix"]["prefix_sha256"] = "0" * 64
    resealed_tampers.append(tampered)
    tampered = deepcopy(record)
    mixed = tampered["analysis"]["opening_ledger_provenance"][
        "mixed_sentinel_and_result_validation_failure_gpu_qualification"
    ]
    mixed["reconciled_ledger_byte_count"] -= 1
    resealed_tampers.append(tampered)
    for tampered in resealed_tampers:
        tampered["closed_record_sha256"] = publication_campaign._closed_record_sha256(
            tampered
        )
        with pytest.raises(ValueError, match="does not match the frozen plan"):
            validate_publication_campaign_plan_record(tampered)

    empty_same_id_prefix = DatabricksLedgerPrefix(
        ledger_id=CAMPAIGN_LEDGER_ID,
        cap_cluster_hours=1024.0,
        reservation_count=0,
        submission_receipt_count=0,
        terminal_actual_count=0,
        prefix_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="retained opening"):
        build_publication_campaign_plan(
            "vllm-0271-publication-v1",
            campaign_ledger_id=CAMPAIGN_LEDGER_ID,
            campaign_ledger_path_sha256=CAMPAIGN_LEDGER_PATH_SHA256,
            campaign_ledger_prefix=empty_same_id_prefix,
            campaign_opening_terminal_gpu_hours=(
                PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
            ),
        )
    with pytest.raises(ValueError, match="frozen publication campaign"):
        build_publication_campaign_plan(
            "alternate-publication-campaign",
            campaign_ledger_id=CAMPAIGN_LEDGER_ID,
            campaign_ledger_path_sha256=CAMPAIGN_LEDGER_PATH_SHA256,
            campaign_ledger_prefix=CAMPAIGN_LEDGER_PREFIX,
            campaign_opening_terminal_gpu_hours=(
                PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
            ),
        )


def test_publication_campaign_preserves_predecessor_and_aligns_v2_authority(tmp_path):
    plan = build_publication_campaign_plan(
        PUBLICATION_CAMPAIGN_ID,
        campaign_ledger_id=CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=CAMPAIGN_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
        ),
    )
    record = publication_campaign_plan_to_record(plan)
    canonical = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert len(canonical) == 80_378
    assert hashlib.sha256(canonical).hexdigest() == (
        "5c4e90b63f6c44ef7a924c6ec5cea6ab3f31d70ca592b2395b55c56825c5170d"
    )
    assert (
        publication_campaign.PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_TASK_INTERVAL_EPOCH_MILLISECONDS_BY_JOB
        == (
            ("aws-g5-a10g-16k-c4-capacity", 1_787_700_619_005, 1_787_701_516_877),
            (
                "aws-g5-a10g-auto-backend-diagnostic",
                1_787_700_622_410,
                1_787_701_609_067,
            ),
            (
                "aws-g5-a10g-forced-triton-runtime-handoff",
                1_787_700_613_630,
                1_787_701_451_067,
            ),
            (
                "aws-g5-a10g-matched-token-logit",
                1_787_700_615_401,
                1_787_701_432_589,
            ),
            (
                "aws-g5-a10g-packed-page-roundtrip",
                1_787_700_614_516,
                1_787_701_414_420,
            ),
            ("aws-g6-l4-32k-c4-gmu-70", 1_787_700_616_341, 1_787_701_527_877),
            ("aws-g6-l4-32k-c4-gmu-75", 1_787_700_617_186, 1_787_701_410_333),
            ("aws-g6-l4-32k-c4-gmu-80", 1_787_700_618_058, 1_787_701_558_032),
            (
                "aws-g6-l4-auto-backend-diagnostic",
                1_787_700_621_546,
                1_787_701_688_304,
            ),
            (
                "aws-g6-l4-forced-triton-runtime-handoff",
                1_787_700_610_584,
                1_787_701_416_680,
            ),
            (
                "aws-g6-l4-generation-throughput",
                1_787_700_619_833,
                1_787_701_528_065,
            ),
            (
                "aws-g6-l4-matched-token-logit",
                1_787_700_612_663,
                1_787_701_411_469,
            ),
            (
                "aws-g6-l4-packed-page-roundtrip",
                1_787_700_611_665,
                1_787_701_390_579,
            ),
            (
                "aws-g6e-l40s-generation-throughput",
                1_787_700_620_668,
                1_787_701_733_399,
            ),
        )
    )

    first_output = tmp_path / "a" / "publication-campaign.json"
    second_output = tmp_path / "b" / "publication-campaign.json"
    publication_campaign.write_publication_campaign_plan_json(plan, first_output)
    publication_campaign.write_publication_campaign_plan_json(plan, second_output)
    assert first_output.read_bytes() == second_output.read_bytes() == canonical
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "absent.json")
    with pytest.raises(FileExistsError, match="already exists"):
        publication_campaign.write_publication_campaign_plan_json(plan, dangling)
    assert dangling.is_symlink()
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="contains a symlink"):
        publication_campaign.write_publication_campaign_plan_json(
            plan, linked_parent / "campaign.json"
        )

    repository_root = Path(__file__).resolve().parents[1]
    predecessor_path = repository_root / (
        "databricks-runs/vllm-0271-publication-prep/"
        "publication-campaign-plan-sha256-"
        "eb306f9a8be50730bfef81121c2a83ebec7e50e89386addb7f77ce6001bcd85f.json"
    )
    predecessor_stat = predecessor_path.stat()
    predecessor_bytes = predecessor_path.read_bytes()
    assert len(predecessor_bytes) == 66_635
    assert hashlib.sha256(predecessor_bytes).hexdigest() == (
        "eb306f9a8be50730bfef81121c2a83ebec7e50e89386addb7f77ce6001bcd85f"
    )
    predecessor = json.loads(predecessor_bytes)
    assert predecessor["closed_record_sha256"] == (
        "5f90b531b30ac6f4b29e0151d688a005b0377b205ca39645376d7d43aef5e305"
    )
    predecessor_provenance = predecessor["analysis"]["opening_ledger_provenance"]
    current_provenance = record["analysis"]["opening_ledger_provenance"]
    predecessor_history = {
        key: value
        for key, value in predecessor_provenance.items()
        if key != "retained_opening_prefix"
    }
    current_history = {
        key: value
        for key, value in current_provenance.items()
        if key
        not in {
            "mixed_sentinel_and_result_validation_failure_gpu_qualification",
            "prefix_before_mixed_sentinel_and_result_validation_failure_gpu_qualification",
            "retained_opening_prefix",
        }
    }
    assert current_history == predecessor_history
    assert predecessor_path.read_bytes() == predecessor_bytes
    assert predecessor_path.stat() == predecessor_stat

    from document_kv_cache.gpu_qualification_v2 import (
        GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
        GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS,
    )

    assert (
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX
        != GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    )
    assert (
        GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX.reservation_count,
        GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX.submission_receipt_count,
        GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX.terminal_actual_count,
        GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX.prefix_sha256,
    ) == (
        430,
        292,
        430,
        "116251d3ca5fce37ce5749565e1059fdf65b30ce17fd12ebc50b877835f9772b",
    )
    assert GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS == 116.12134277777776
    for public_name in (
        "PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ERROR_SHA256_BY_JOB",
        "PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_SUBMIT_RECEIPT_TREE_SHA256",
        "PUBLICATION_CAMPAIGN_PRE_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_CAMPAIGN_FILE_SHA256",
    ):
        assert public_name in publication_campaign.__all__


def test_publication_campaign_mixed_failure_source_authority_fails_closed(
    monkeypatch,
):
    source_tampers = (
        (
            "PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_PHASE_LEASE_FILE_SHA256",
            "g" * 64,
        ),
        (
            "PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_SUBMIT_RECEIPT_TREE_FILE_COUNT",
            15,
        ),
        (
            "PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_VERSION_CONTRACT_MISMATCH_JOB_IDS",
            ("aws-g5-a10g-packed-page-roundtrip",),
        ),
        (
            "PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ACTUAL_CLUSTER_DURATION_SECONDS",
            12_455.253,
        ),
        (
            "PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_TASK_INTERVAL_EPOCH_MILLISECONDS_BY_JOB",
            publication_campaign.PUBLICATION_CAMPAIGN_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_TASK_INTERVAL_EPOCH_MILLISECONDS_BY_JOB[
                :-1
            ],
        ),
        ("PUBLICATION_CAMPAIGN_OPENING_LEDGER_BYTE_COUNT", 220_425),
    )
    for name, value in source_tampers:
        with monkeypatch.context() as context:
            context.setattr(publication_campaign, name, value)
            with pytest.raises(
                RuntimeError,
                match="mixed sentinel and result validation failure source closure drift",
            ):
                publication_campaign_plan_to_record(
                    build_publication_campaign_plan(
                        PUBLICATION_CAMPAIGN_ID,
                        campaign_ledger_id=CAMPAIGN_LEDGER_ID,
                        campaign_ledger_path_sha256=CAMPAIGN_LEDGER_PATH_SHA256,
                        campaign_ledger_prefix=CAMPAIGN_LEDGER_PREFIX,
                        campaign_opening_terminal_gpu_hours=(
                            PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
                        ),
                    )
                )


def test_publication_campaign_cli_writes_once(tmp_path, monkeypatch):
    output = tmp_path / "publication-campaign.json"
    ledger_path = tmp_path / "campaign-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id=CAMPAIGN_LEDGER_ID,
    )
    monkeypatch.setattr(
        publication_campaign,
        "read_databricks_cluster_hour_ledger_json",
        lambda _path: type(
            "OpeningLedger",
            (),
            {
                "ledger_id": CAMPAIGN_LEDGER_ID,
                "cap_cluster_hours": 1024.0,
                "active_reserved_cluster_hours": 0.0,
                "terminal_actual_cluster_hours": (
                    PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
                ),
            },
        )(),
    )
    monkeypatch.setattr(
        publication_campaign,
        "databricks_ledger_prefix",
        lambda _ledger: CAMPAIGN_LEDGER_PREFIX,
    )
    monkeypatch.setattr(
        publication_campaign,
        "databricks_ledger_path_sha256",
        lambda _path: CAMPAIGN_LEDGER_PATH_SHA256,
    )

    assert (
        main(
            [
                "--campaign-id",
                "vllm-0271-publication-v1",
                "--campaign-ledger-id",
                CAMPAIGN_LEDGER_ID,
                "--campaign-ledger-json",
                str(ledger_path),
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    validate_publication_campaign_plan_record(record)

    with pytest.raises(FileExistsError):
        main(
            [
                "--campaign-id",
                "vllm-0271-publication-v1",
                "--campaign-ledger-id",
                CAMPAIGN_LEDGER_ID,
                "--campaign-ledger-json",
                str(ledger_path),
                "--output-json",
                str(output),
            ]
        )


def test_publication_campaign_cli_rejects_unmigrated_or_symlinked_ledger(tmp_path):
    low_cap = tmp_path / "low-cap-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        low_cap,
        ledger_id=CAMPAIGN_LEDGER_ID,
        cap_cluster_hours=120.0,
    )
    with pytest.raises(ValueError, match="migrated to the 1,024-hour cap"):
        main(
            [
                "--campaign-id",
                "vllm-0271-publication-v1",
                "--campaign-ledger-id",
                CAMPAIGN_LEDGER_ID,
                "--campaign-ledger-json",
                str(low_cap),
                "--output-json",
                str(tmp_path / "low-cap-campaign.json"),
            ]
        )

    migrated = tmp_path / "migrated-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        migrated,
        ledger_id=CAMPAIGN_LEDGER_ID,
    )
    symlink = tmp_path / "ledger-link.json"
    symlink.symlink_to(migrated)
    with pytest.raises(ValueError, match="symlink"):
        main(
            [
                "--campaign-id",
                "vllm-0271-publication-v1",
                "--campaign-ledger-id",
                CAMPAIGN_LEDGER_ID,
                "--campaign-ledger-json",
                str(symlink),
                "--output-json",
                str(tmp_path / "symlink-campaign.json"),
            ]
        )
