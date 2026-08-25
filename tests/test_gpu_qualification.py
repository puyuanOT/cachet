import hashlib
import json
import os
import site
import stat
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from document_kv_cache import _gpu_qualification_sentinel_worker as sentinel_worker
from document_kv_cache import gpu_qualification_sentinels as qualification_sentinels
from document_kv_cache.databricks_resource_ledger import DatabricksLedgerPrefix
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_A10G_INPUT_CONTEXT_TOKENS,
    GPU_QUALIFICATION_A10G_MAX_MODEL_LEN,
    GPU_QUALIFICATION_A10G_REQUIRED_KV_CAPACITY_TOKENS,
    GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256,
    GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS,
    GPU_QUALIFICATION_CONNECTOR_SOURCE_SHA256,
    GPU_QUALIFICATION_MAX_CLOUD_JOBS,
    GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES,
    GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND,
    GPU_QUALIFICATION_INPUT_CONTEXT_TOKENS,
    GPU_QUALIFICATION_MAX_MODEL_LEN,
    GPU_QUALIFICATION_OFFICIAL_WHEEL_SHA256,
    GPU_QUALIFICATION_PUBLICATION_BACKEND,
    GPU_QUALIFICATION_REQUIRED_KV_CAPACITY_TOKENS,
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPU_QUALIFICATION_TERMINAL_RECEIPT_RECORD_TYPE,
    GPU_QUALIFICATION_VLLM_VERSION,
    GPUQualificationArtifactPins,
    _build_governed_cloud_gpu_evidence,
    _build_governed_gpu_qualification_evidence,
    build_cloud_gpu_evidence,
    build_gpu_job_result,
    build_gpu_qualification_evidence,
    build_gpu_qualification_plan,
    build_local_preflight_evidence,
    canonical_gpu_qualification_json,
    validate_gpu_qualification_evidence_record,
    validate_gpu_qualification_plan_record,
    write_canonical_gpu_qualification_json,
    write_gpu_qualification_evidence_json,
    write_gpu_qualification_plan_json,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_ID,
    PUBLICATION_CAMPAIGN_LEDGER_ID,
    PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS,
)
from document_kv_cache.serving_env import (
    VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_LOCK_SHA256,
)


CAMPAIGN_ID = PUBLICATION_CAMPAIGN_ID
CAMPAIGN_LEDGER_ID = PUBLICATION_CAMPAIGN_LEDGER_ID
CAMPAIGN_RECORD_SHA256 = PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256
PINS = GPUQualificationArtifactPins(
    runtime_lock_sha256=VLLM_RUNTIME_LOCK_SHA256,
    patched_vllm_wheel_sha256=GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    package_wheel_sha256="6" * 64,
    cachet_source_tree_sha256="3" * 64,
    runner_sha256="4" * 64,
    input_bundle_sha256="5" * 64,
)
CORE_VERSIONS = {
    "bitsandbytes": "0.49.2",
    "flashinfer-cubin": "0.6.16.post3",
    "flashinfer-jit-cache": "0.6.16.post3+cu129",
    "flashinfer-python": "0.6.16.post3",
    "torch": "2.13.0+cu129",
    "torchaudio": "2.11.0+cu129",
    "torchcodec": "0.16.0+cu129",
    "torchvision": "0.28.0+cu129",
    "triton": "3.7.1",
}
PATCH_MEMBERS = {
    "vllm/model_executor/layers/attention/attention.py": (
        "5735acfb390cf344caeec950c2f286344bcd84721ce287e0a56701f2a18bc839"
    ),
    "vllm/v1/attention/backends/triton_attn.py": (
        "4dae0ff6c4ee8f11c1f195151a11673d595d457c413032e7bae7550913f94390"
    ),
    "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py": (
        "0682ca7bc56edf7cea5419188a81c78510b54192471472b160aa447ac0ceeb08"
    ),
}


def _debian_site_package_candidates(runtime_root: Path) -> list[str]:
    return [
        str(runtime_root / "lib" / "python3.11" / "site-packages"),
        str(runtime_root / "local" / "lib" / "python3.11" / "dist-packages"),
        str(runtime_root / "lib" / "python3" / "dist-packages"),
        str(runtime_root / "lib" / "python3.11" / "dist-packages"),
    ]


def _weight_quantizer_attestation() -> dict[str, Any]:
    return {
        "bitsandbytes_loader_sha256": GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256,
        "bitsandbytes_version": "0.49.2",
        "dynamic_quant_call": {
            "compress_statistics": True,
            "input_dtype": "bfloat16",
            "nested_state": True,
            "packed_dtype": "uint8",
            "quant_type": "nf4",
        },
        "hf_generator_config": {
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_storage": "uint8",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "load_in_4bit": True,
        },
    }


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _seal(record: dict[str, Any]) -> None:
    payload = deepcopy(record)
    payload.pop("closed_record_sha256", None)
    record["closed_record_sha256"] = hashlib.sha256(
        canonical_gpu_qualification_json(payload).encode()
    ).hexdigest()


def _runtime_measurements(*, software_path: bool) -> dict[str, Any]:
    return {
        "attention_backend_observed": "TRITON_ATTN",
        "attention_backend_requested": "TRITON_ATTN",
        "compute_dtype": "bfloat16",
        "connector_source_sha256": GPU_QUALIFICATION_CONNECTOR_SOURCE_SHA256,
        "direct_url_matches_patched_wheel": True,
        "driver_cuda_compatibility_ok": True,
        "e5m2_software_path_exercised": software_path,
        "finite_logits": True,
        "handoff_injected": True,
        "handoff_injected_layer_count": 36,
        "handoff_kv_bits": 8,
        "handoff_kv_dtype": "fp8_e5m2",
        "handoff_loaded": True,
        "handoff_loaded_layer_count": 36,
        "handoff_written": True,
        "handoff_written_layer_count": 36,
        "installed_core_distribution_versions": CORE_VERSIONS,
        "installed_connector_base_py_sha256": (
            GPU_QUALIFICATION_CONNECTOR_SOURCE_SHA256
        ),
        "installed_patch_member_sha256": PATCH_MEMBERS,
        "libcudart_major_versions": [12],
        "libcudart_so_12_present": True,
        "libcudart_so_13_present": False,
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "native_shared_object_count": 23,
        "pip_check_ok": True,
        "python_version": "3.11.11",
        "query_dtype": "bfloat16",
        "runtime_kv_dtype": "fp8_e5m2",
        "runtime_kv_bits": 8,
        "runtime_lock_attestation": {
            "locked_distribution_count": VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
            "ok": True,
            "runtime_lock_sha256": VLLM_RUNTIME_LOCK_SHA256,
            "unexpected_distributions": [],
            "vllm_direct_url": "file:///locked/vllm-0.27.1.whl",
            "vllm_package_version": GPU_QUALIFICATION_VLLM_VERSION,
            "vllm_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        },
        "runtime_lock_verifier_ok": True,
        "site_packages_read_only": True,
        "strict_direct_url_verifier_ok": True,
        "system_cuda_version": "12.1",
        "glibc_version": "2.35",
        "triton_cache_miss_compile": True,
        "triton_compile_count": 3,
        "triton_compiled_kernel_names": [
            "triton_reshape_and_cache_flash",
            "triton_unified_attention",
        ],
        "triton_kernel_launch_count": 19,
        "unresolved_native_shared_object_count": 0,
        "weight_bits": 4,
        "weight_quantization": "bitsandbytes",
        "trust_remote_code": False,
        "weight_quantizer_attestation": _weight_quantizer_attestation(),
    }


def _packed_measurements() -> dict[str, Any]:
    cases = []
    for layout in ("NHD", "HND"):
        raw_sha256 = _digest(f"raw-{layout}")
        cases.append(
            {
                "bf16_reference_max_abs_error": 0.0625,
                "bf16_reference_scope": "attention_output",
                "cache_page_layout": "B_H_N_2D",
                "cache_page_shape": ["B", "H", "N", "2D"],
                "input_value_max": 1.0,
                "input_value_min": -1.0,
                "negative_slot_guard_passed": True,
                "noncontiguous_stride_passed": True,
                "partial_slot_guard_passed": True,
                "payload_layout": layout,
                "query_dtype": "bfloat16",
                "raw_byte_mismatch_count": 0,
                "raw_bytes_written": 4096,
                "read_raw_sha256": raw_sha256,
                "untouched_guard_mismatch_count": 0,
                "written_raw_sha256": raw_sha256,
            }
        )
    return {
        "attention_backend_observed": "TRITON_ATTN",
        "attention_backend_requested": "TRITON_ATTN",
        "cases": cases,
        "triton_compile_count": 2,
        "triton_kernel_launch_count": 10,
    }


def _token_measurements() -> dict[str, Any]:
    examples = []
    for index in range(4):
        token_sha256 = _digest(f"prompt-{index}")
        arms = []
        for arm_id in ("baseline_prefill", "vanilla_prefill"):
            output_sha256 = _digest(f"{arm_id}-output-{index}")
            arms.append(
                {
                    "arm_id": arm_id,
                    "finite_logits": True,
                    "logit_probe_position_ids_sha256": _digest(
                        f"logit-positions-{index}"
                    ),
                    "max_abs_logit_drift": 0.00001,
                    "output_token_count": 16,
                    "output_token_ids_repeat_sha256": [
                        output_sha256,
                        output_sha256,
                    ],
                    "repeat_count": 2,
                }
            )
        examples.append(
            {
                "arms": arms,
                "baseline_full_prompt_token_ids_sha256": token_sha256,
                "example_id": f"example-{index}",
                "full_prompt_token_count": 8192,
                "input_bundle_sha256": PINS.input_bundle_sha256,
                "vanilla_prefix_token_count": 8000,
                "vanilla_reconstructed_full_prompt_token_ids_sha256": (token_sha256),
                "vanilla_suffix_token_count": 192,
            }
        )
    return {
        "attention_backend_observed": "TRITON_ATTN",
        "attention_backend_requested": "TRITON_ATTN",
        "baseline_handoff_absent": True,
        "cache_phases": ["cold", "warm"],
        "execution_mode": "real_end_to_end_requests",
        "examples": examples,
        "request_parallelism": 1,
        "triton_compile_count": 1,
        "triton_kernel_launch_count": 12,
        "vanilla_handoff_injected": True,
        "trust_remote_code": False,
    }


def _gmu_measurements(gmu: float) -> dict[str, Any]:
    total = 24 * 1024**3
    headroom = (3 * 1024**3) if gmu < 0.80 else (1 * 1024**3)
    return {
        "active_request_memory_observation_count": 3,
        "attention_backend_observed": "TRITON_ATTN",
        "attention_backend_requested": "TRITON_ATTN",
        "candidate_qualified": gmu < 0.80,
        "cold_disk_evicted_file_count": 5,
        "connector_loaded_layer_counts": [36, 36, 36, 36],
        "connector_successful_load_count": 4,
        "fatal_error_count": 0,
        "forced_decode_tokens": GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS,
        "gpu_memory_utilization": gmu,
        "input_tokens_per_request": GPU_QUALIFICATION_INPUT_CONTEXT_TOKENS,
        "kv_cache_capacity_tokens": GPU_QUALIFICATION_REQUIRED_KV_CAPACITY_TOKENS,
        "max_model_len": GPU_QUALIFICATION_MAX_MODEL_LEN,
        "observed_peak_headroom_bytes": headroom,
        "observed_peak_used_memory_bytes": total - headroom,
        "observed_total_memory_bytes": total,
        "oom_count": 0,
        "request_parallelism": 4,
        "request_success_count": 4,
        "q8_pre_rope_handoffs": True,
        "selected_examples": [
            {"dataset": dataset, "example_id": f"{dataset}-32768"}
            for dataset in ("biography", "hotpotqa", "musique", "niah")
        ],
        "triton_compile_count": 1,
        "triton_kernel_launch_count": 32,
        "trust_remote_code": False,
        "vanilla_handoff_injected": True,
        "weight_quantizer_attestation": _weight_quantizer_attestation(),
    }


def _throughput_measurements(rate: float = 40.0) -> dict[str, Any]:
    buckets = []
    samples = []
    total_tokens = 0
    total_seconds = 0.0
    for length in (8192, 16384, 32768):
        exact_tokens_per_sample = length - 1
        tokens = exact_tokens_per_sample * 4
        wall_seconds = tokens / rate
        buckets.append(
            {
                "durable_write_completed_count": 4,
                "length_bucket_tokens": length,
                "prefix_tokens": tokens,
                "sample_count": 4,
                "tokens_per_second": rate,
                "wall_seconds": wall_seconds,
            }
        )
        total_tokens += tokens
        total_seconds += wall_seconds
        for dataset in ("biography", "hotpotqa", "musique", "niah"):
            segments = [
                {
                    "index": index,
                    "token_count": 2047 if index == length // 2048 - 1 else 2048,
                    "token_ids_sha256": _digest(f"{length}-{dataset}-segment-{index}"),
                }
                for index in range(length // 2048)
            ]
            samples.append(
                {
                    "cache_prefix_token_count": exact_tokens_per_sample,
                    "cache_prefix_token_ids_sha256": _digest(
                        f"{length}-{dataset}-prefix"
                    ),
                    "dataset": dataset,
                    "example_id": f"{dataset}-{length}",
                    "input_tokens_target": length,
                    "raw_artifact_bytes": length * 36 * 2 * 8 * 128,
                    "raw_artifact_sha256": _digest(f"{length}-{dataset}-artifact"),
                    "segment_count": len(segments),
                    "segments": segments,
                }
            )
    return {
        "aggregate_prefix_tokens": total_tokens,
        "aggregate_tokens_per_second": total_tokens / total_seconds,
        "aggregate_wall_seconds": total_seconds,
        "attention_backend_observed": "TRITON_ATTN",
        "attention_backend_requested": "TRITON_ATTN",
        "buckets": buckets,
        "clock_scope": "prefix_generation_through_durable_kv_write",
        "failed_write_count": 0,
        "generator_device_map": "auto",
        "samples": samples,
        "trust_remote_code": False,
        "weight_quantizer_attestation": _weight_quantizer_attestation(),
        "triton_compile_count": 1,
        "triton_kernel_launch_count": 100,
        "writes_included": True,
    }


def _a10g_capacity_measurements() -> dict[str, Any]:
    total = 24 * 1024**3
    headroom = 3 * 1024**3
    return {
        "active_request_memory_observation_count": 3,
        "attention_backend_observed": "TRITON_ATTN",
        "attention_backend_requested": "TRITON_ATTN",
        "capacity_qualified": True,
        "cold_disk_evicted_file_count": 5,
        "connector_loaded_layer_counts": [36, 36, 36, 36],
        "connector_successful_load_count": 4,
        "fatal_error_count": 0,
        "forced_decode_tokens": GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS,
        "gpu_memory_utilization": 0.90,
        "input_tokens_per_request": GPU_QUALIFICATION_A10G_INPUT_CONTEXT_TOKENS,
        "kv_cache_capacity_tokens": (
            GPU_QUALIFICATION_A10G_REQUIRED_KV_CAPACITY_TOKENS
        ),
        "max_model_len": GPU_QUALIFICATION_A10G_MAX_MODEL_LEN,
        "observed_peak_headroom_bytes": headroom,
        "observed_peak_used_memory_bytes": total - headroom,
        "observed_total_memory_bytes": total,
        "oom_count": 0,
        "request_parallelism": 4,
        "request_success_count": 4,
        "q8_pre_rope_handoffs": True,
        "selected_examples": [
            {"dataset": dataset, "example_id": f"{dataset}-16384"}
            for dataset in ("biography", "hotpotqa", "musique", "niah")
        ],
        "triton_compile_count": 1,
        "triton_kernel_launch_count": 16,
        "trust_remote_code": False,
        "vanilla_handoff_injected": True,
        "weight_quantizer_attestation": _weight_quantizer_attestation(),
    }


def _measurements(job: dict[str, Any]) -> dict[str, Any]:
    sentinel = job["sentinel"]
    if sentinel == "forced_triton_runtime_handoff":
        return _runtime_measurements(software_path=job["hardware_id"] == "aws-g5-a10g")
    if sentinel == "packed_page_raw_byte_roundtrip":
        return _packed_measurements()
    if sentinel == "matched_token_contract_and_determinism":
        return _token_measurements()
    if sentinel == "l4_32k_c4_gmu_sweep":
        return _gmu_measurements(float(job["requirements"]["gpu_memory_utilization"]))
    if sentinel == "a10g_16k_c4_capacity":
        return _a10g_capacity_measurements()
    if sentinel == "generation_throughput_with_writes":
        return _throughput_measurements()
    if sentinel == "auto_backend_diagnostic":
        return {
            "backend_selection_mode": "auto",
            "observed_backend": "FLASHINFER",
            "publication_backend_changed": False,
            "trust_remote_code": False,
        }
    raise AssertionError(sentinel)


def _valid_plan() -> dict[str, Any]:
    return build_gpu_qualification_plan(
        campaign_id=CAMPAIGN_ID,
        campaign_record_sha256=CAMPAIGN_RECORD_SHA256,
        campaign_ledger_id=CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
        ),
        artifact_pins=PINS,
    )


def _valid_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _valid_plan()
    plan_sha256 = plan["closed_record_sha256"]
    local = build_local_preflight_evidence(
        plan_sha256=plan_sha256,
        completed_at_utc="2026-08-24T00:00:00Z",
        check_evidence_sha256={
            check_id: _digest(check_id)
            for check_id in plan["local_preflight"]["check_ids"]
        },
    )
    jobs = []
    terminal_receipts = []
    for index, job in enumerate(plan["cloud_qualification"]["jobs"]):
        reservation_attempt_id = f"gpuq-{plan_sha256[:16]}-{job['job_id']}"
        task_key = "gpu_qualification_" + job["job_id"].replace("-", "_").replace(
            ".", "_"
        )
        result = build_gpu_job_result(
            plan_record=plan,
            job_id=job["job_id"],
            reservation_attempt_id=reservation_attempt_id,
            task_key=task_key,
            output_json=(
                f"dbfs:/qualification/{plan_sha256}/{job['job_id']}/gpu-job-result.json"
            ),
            cloud_run_id=str(10_000 + index),
            cloud_cluster_id=f"cluster-{index}",
            started_at_utc="2026-08-24T01:00:00Z",
            finished_at_utc="2026-08-24T02:00:00Z",
            nvidia_driver_version="570.172.08",
            observed_gpu=job["gpu"],
            observed_gpu_compute_capability=job["compute_capability"],
            observed_vllm_version=GPU_QUALIFICATION_VLLM_VERSION,
            observed_torch_cuda_version="12.9",
            observed_artifact_sha256=PINS.to_record(),
            measurements=_measurements(job),
        )
        jobs.append(result)
        terminal_receipts.append(_terminal_receipt(result, index=index))
    cloud = _build_governed_cloud_gpu_evidence(
        plan_sha256=plan_sha256,
        jobs=jobs,
        terminal_receipts=terminal_receipts,
        selected_gpu_memory_utilization=0.75,
    )
    evidence = _build_governed_gpu_qualification_evidence(
        campaign_id=CAMPAIGN_ID,
        plan_sha256=plan_sha256,
        local_preflight_evidence=local,
        cloud_gpu_evidence=cloud,
    )
    return plan, evidence


def _terminal_receipt(result: dict[str, Any], *, index: int) -> dict[str, Any]:
    result_file_sha256 = hashlib.sha256(
        (canonical_gpu_qualification_json(result) + "\n").encode()
    ).hexdigest()
    duration_seconds = 3660.0
    ledger_terminal_actual = {
        "actual_cluster_duration_seconds": duration_seconds,
        "actual_cluster_hours": duration_seconds / 3600.0,
        "attempt_id": result["reservation_attempt_id"],
        "control_plane_status_sha256": _digest(f"control-plane-{index}"),
        "run_id": result["cloud_run_id"],
        "submit_payload_sha256": _digest(f"submit-payload-{index}"),
        "terminal_state": "succeeded",
        "verification_source": "direct_databricks_runs_get",
    }
    receipt = {
        "authorization_source": "direct_databricks_runs_get",
        "closed_record_sha256": "",
        "cloud_cluster_id": result["cloud_cluster_id"],
        "cloud_run_id": result["cloud_run_id"],
        "collected_at_utc": "2026-08-24T02:02:00Z",
        "control_plane_status_sha256": _digest(f"control-plane-{index}"),
        "driver_node_type_id": _node_type(result["hardware_id"]),
        "end_time_ms": 1787536860000,
        "job_id": result["job_id"],
        "ledger_actual_cluster_duration_seconds": duration_seconds,
        "ledger_id": "gpu-qualification-test-ledger",
        "ledger_terminal_actual_sha256": hashlib.sha256(
            canonical_gpu_qualification_json(ledger_terminal_actual).encode()
        ).hexdigest(),
        "life_cycle_state": "TERMINATED",
        "node_type_id": _node_type(result["hardware_id"]),
        "output_json": result["output_json"],
        "phase_batch_record_sha256": _digest("test-phase-batch"),
        "phase_terminal_prefix": DatabricksLedgerPrefix(
            ledger_id="gpu-qualification-test-ledger",
            cap_cluster_hours=1024.0,
            reservation_count=14,
            submission_receipt_count=14,
            terminal_actual_count=14,
            prefix_sha256=_digest("test-phase-terminal-prefix"),
        ).to_record(),
        "plan_sha256": result["plan_sha256"],
        "record_type": GPU_QUALIFICATION_TERMINAL_RECEIPT_RECORD_TYPE,
        "reservation_attempt_id": result["reservation_attempt_id"],
        "result_file_sha256": result_file_sha256,
        "result_record_sha256": result["closed_record_sha256"],
        "result_state": "SUCCESS",
        "run_name": (f"cachet-gpu-qualification-{CAMPAIGN_ID}-{result['job_id']}"),
        "schema_version": 1,
        "start_time_ms": 1787533140000,
        "submit_payload_sha256": _digest(f"submit-payload-{index}"),
        "task_attempt_number": 0,
        "task_end_time_ms": 1787536830000,
        "task_key": result["task_key"],
        "task_life_cycle_state": "TERMINATED",
        "task_max_retries": 0,
        "task_result_state": "SUCCESS",
        "task_run_id": str(20_000 + index),
        "task_start_time_ms": 1787533170000,
    }
    _seal(receipt)
    return receipt


def _node_type(hardware_id: str) -> str:
    return {
        "aws-g6-l4": "g6.8xlarge",
        "aws-g5-a10g": "g5.8xlarge",
        "aws-g6e-l40s": "g6e.4xlarge",
    }[hardware_id]


def _reseal_evidence(evidence: dict[str, Any]) -> None:
    cloud = evidence["cloud_gpu_evidence"]
    for job in cloud["jobs"]:
        _seal(job)
    jobs_by_id = {job["job_id"]: job for job in cloud["jobs"]}
    for receipt in cloud["terminal_receipts"]:
        job = jobs_by_id.get(receipt["job_id"])
        if job is not None:
            receipt["result_record_sha256"] = job["closed_record_sha256"]
            receipt["result_file_sha256"] = hashlib.sha256(
                (canonical_gpu_qualification_json(job) + "\n").encode()
            ).hexdigest()
            _seal(receipt)
    _seal(cloud)
    _seal(evidence)


def test_debian_site_package_candidates_lock_down_and_attest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_root = tmp_path / "runtime"
    (runtime_root / "bin").mkdir(parents=True)
    runtime_root = runtime_root.resolve(strict=True)
    actual_root = runtime_root / "lib" / "python3.11" / "site-packages"
    nested = actual_root / "document_kv_cache"
    nested.mkdir(parents=True)
    payload = nested / "worker.py"
    payload.write_text("reviewed = True\n", encoding="utf-8")
    actual_root.chmod(0o770)
    nested.chmod(0o770)
    payload.chmod(0o660)
    candidates = _debian_site_package_candidates(runtime_root)

    def discover(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(candidates),
            stderr="",
        )

    monkeypatch.setattr(qualification_sentinels.subprocess, "run", discover)
    monkeypatch.setattr(sentinel_worker.sys, "prefix", str(runtime_root))
    monkeypatch.setattr(site, "getsitepackages", lambda: candidates)
    real_chmod = os.chmod

    def reject_path_no_follow_chmod(*args: Any, **kwargs: Any) -> None:
        if kwargs.get("follow_symlinks") is False:
            raise NotImplementedError("Linux does not support no-follow chmod")
        real_chmod(*args, **kwargs)

    monkeypatch.setattr(os, "chmod", reject_path_no_follow_chmod)

    assert sentinel_worker._site_packages_read_only() is False
    qualification_sentinels._make_site_packages_read_only(
        runtime_root / "bin" / "python"
    )
    locked_modes = {
        path: stat.S_IMODE(path.stat().st_mode)
        for path in (actual_root, nested, payload)
    }
    assert locked_modes == {
        actual_root: 0o550,
        nested: 0o550,
        payload: 0o440,
    }
    assert sentinel_worker._site_packages_read_only() is True

    qualification_sentinels._make_site_packages_read_only(
        runtime_root / "bin" / "python"
    )
    assert {
        path: stat.S_IMODE(path.stat().st_mode)
        for path in (actual_root, nested, payload)
    } == locked_modes


def test_site_package_candidates_reject_before_any_permission_change(
    tmp_path: Path,
):
    runtime_root = tmp_path / "runtime"
    actual_root = runtime_root / "lib" / "python3.11" / "site-packages"
    actual_root.mkdir(parents=True)
    payload = actual_root / "worker.py"
    payload.write_text("reviewed = True\n", encoding="utf-8")
    actual_root.chmod(0o770)
    payload.chmod(0o660)
    runtime_root = runtime_root.resolve(strict=True)
    valid = str(actual_root)
    outside = tmp_path / "outside" / "lib" / "python3.11" / "site-packages"
    invalid_candidates: list[object] = [
        7,
        "",
        " relative/site-packages",
        str(runtime_root / "lib" / "python3.10" / "site-packages"),
        str(runtime_root / "lib" / "python3.11" / "unexpected"),
        f"{runtime_root}/lib/../lib/python3.11/site-packages",
        str(outside),
        valid,
    ]
    original_modes = (
        stat.S_IMODE(actual_root.stat().st_mode),
        stat.S_IMODE(payload.stat().st_mode),
    )
    for invalid in invalid_candidates:
        with pytest.raises(RuntimeError):
            qualification_sentinels._validated_site_packages_tree(
                runtime_root,
                [valid, invalid],
            )
        assert (
            stat.S_IMODE(actual_root.stat().st_mode),
            stat.S_IMODE(payload.stat().st_mode),
        ) == original_modes

    candidate_file = runtime_root / "local" / "lib" / "python3.11" / "dist-packages"
    candidate_file.parent.mkdir(parents=True)
    candidate_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(RuntimeError):
        qualification_sentinels._validated_site_packages_tree(
            runtime_root,
            [valid, str(candidate_file)],
        )
    assert (
        stat.S_IMODE(actual_root.stat().st_mode),
        stat.S_IMODE(payload.stat().st_mode),
    ) == original_modes


def test_site_package_candidates_reject_missing_and_symlink_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    empty_root = tmp_path / "empty-runtime"
    empty_root.mkdir()
    empty_root = empty_root.resolve(strict=True)
    missing = _debian_site_package_candidates(empty_root)
    with pytest.raises(RuntimeError, match="no existing site-packages"):
        qualification_sentinels._validated_site_packages_tree(empty_root, missing)
    monkeypatch.setattr(sentinel_worker.sys, "prefix", str(empty_root))
    monkeypatch.setattr(site, "getsitepackages", lambda: missing)
    assert sentinel_worker._site_packages_read_only() is False

    runtime_root = tmp_path / "runtime"
    actual_root = runtime_root / "lib" / "python3.11" / "site-packages"
    actual_root.mkdir(parents=True)
    runtime_root = runtime_root.resolve(strict=True)
    external = tmp_path / "external"
    external.mkdir()
    external.chmod(0o770)
    (actual_root / "escape").symlink_to(external, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic link"):
        qualification_sentinels._validated_site_packages_tree(
            runtime_root,
            [str(actual_root)],
        )
    assert stat.S_IMODE(external.stat().st_mode) == 0o770

    (actual_root / "escape").unlink()
    (runtime_root / "local").symlink_to(external, target_is_directory=True)
    with pytest.raises(RuntimeError, match="site-packages"):
        qualification_sentinels._validated_site_packages_tree(
            runtime_root,
            [
                str(actual_root),
                str(
                    runtime_root
                    / "local"
                    / "lib"
                    / "python3.11"
                    / "dist-packages"
                ),
            ],
        )

    (runtime_root / "local").unlink()
    dangling = runtime_root / "local" / "lib" / "python3.11" / "dist-packages"
    dangling.parent.mkdir(parents=True)
    dangling.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(RuntimeError, match="site-packages"):
        qualification_sentinels._validated_site_packages_tree(
            runtime_root,
            [str(actual_root), str(dangling)],
        )

    root_link = tmp_path / "runtime-link"
    root_link.symlink_to(runtime_root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="runtime root is invalid"):
        qualification_sentinels._validated_site_packages_tree(
            root_link,
            [str(root_link / "lib" / "python3.11" / "site-packages")],
        )


def test_every_allowed_site_package_shape_is_accepted(tmp_path: Path):
    allowed = (
        ("lib", "python3", "dist-packages"),
        ("lib", "python3.11", "dist-packages"),
        ("lib", "python3.11", "site-packages"),
        ("local", "lib", "python3.11", "dist-packages"),
    )
    for index, parts in enumerate(allowed):
        runtime_root = tmp_path / f"runtime-{index}"
        candidate = runtime_root.joinpath(*parts)
        candidate.mkdir(parents=True)
        runtime_root = runtime_root.resolve(strict=True)
        assert qualification_sentinels._validated_site_packages_tree(
            runtime_root,
            [str(candidate)],
        ) == (candidate,)


def test_site_package_lockdown_rejects_hardlinked_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_root = tmp_path / "runtime"
    (runtime_root / "bin").mkdir(parents=True)
    actual_root = runtime_root / "lib" / "python3.11" / "site-packages"
    actual_root.mkdir(parents=True)
    reviewed_payload = actual_root / "reviewed.py"
    reviewed_payload.write_text("reviewed = True\n", encoding="utf-8")
    external_payload = tmp_path / "external.py"
    external_payload.write_text("external = True\n", encoding="utf-8")
    linked_payload = actual_root / "linked.py"
    os.link(external_payload, linked_payload)
    actual_root.chmod(0o770)
    reviewed_payload.chmod(0o660)
    external_payload.chmod(0o660)
    runtime_root = runtime_root.resolve(strict=True)
    candidates = _debian_site_package_candidates(runtime_root)

    def discover(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(candidates),
            stderr="",
        )

    monkeypatch.setattr(qualification_sentinels.subprocess, "run", discover)
    monkeypatch.setattr(sentinel_worker.sys, "prefix", str(runtime_root))
    monkeypatch.setattr(site, "getsitepackages", lambda: candidates)

    assert sentinel_worker._site_packages_read_only() is False
    with pytest.raises(RuntimeError, match="unsafe link count"):
        qualification_sentinels._make_site_packages_read_only(
            runtime_root / "bin" / "python"
        )
    assert stat.S_IMODE(actual_root.stat().st_mode) == 0o770
    assert stat.S_IMODE(reviewed_payload.stat().st_mode) == 0o660
    assert stat.S_IMODE(external_payload.stat().st_mode) == 0o660

    linked_payload.unlink()
    real_open_tree = qualification_sentinels._open_validated_site_packages_tree
    raced_alias = tmp_path / "raced-alias.py"

    def add_link_after_snapshot(runtime: Path, raw_paths: object):
        descriptor, snapshot = real_open_tree(runtime, raw_paths)
        os.link(reviewed_payload, raced_alias)
        return descriptor, snapshot

    monkeypatch.setattr(
        qualification_sentinels,
        "_open_validated_site_packages_tree",
        add_link_after_snapshot,
    )
    with pytest.raises(RuntimeError, match="changed after validation"):
        qualification_sentinels._make_site_packages_read_only(
            runtime_root / "bin" / "python"
        )
    assert stat.S_IMODE(actual_root.stat().st_mode) == 0o770
    assert stat.S_IMODE(reviewed_payload.stat().st_mode) == 0o660
    assert stat.S_IMODE(raced_alias.stat().st_mode) == 0o660

    raced_alias.unlink()
    monkeypatch.setattr(
        qualification_sentinels,
        "_open_validated_site_packages_tree",
        real_open_tree,
    )
    real_fchmod = os.fchmod
    chmod_alias = tmp_path / "chmod-alias.py"
    linked_during_fchmod = False

    def link_during_fchmod(descriptor: int, mode: int) -> None:
        nonlocal linked_during_fchmod
        if (
            not linked_during_fchmod
            and stat.S_ISREG(os.fstat(descriptor).st_mode)
        ):
            os.link(reviewed_payload, chmod_alias)
            linked_during_fchmod = True
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(qualification_sentinels.os, "fchmod", link_during_fchmod)
    with pytest.raises(RuntimeError, match="link count changed during lockdown"):
        qualification_sentinels._make_site_packages_read_only(
            runtime_root / "bin" / "python"
        )
    assert stat.S_IMODE(actual_root.stat().st_mode) == 0o770
    assert stat.S_IMODE(reviewed_payload.stat().st_mode) == 0o660
    assert stat.S_IMODE(chmod_alias.stat().st_mode) == 0o660


def test_site_package_lockdown_rejects_runtime_root_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_root = tmp_path / "runtime"
    (runtime_root / "bin").mkdir(parents=True)
    reviewed_root = runtime_root / "lib" / "python3.11" / "site-packages"
    reviewed_root.mkdir(parents=True)
    reviewed_payload = reviewed_root / "reviewed.py"
    reviewed_payload.write_text("reviewed = True\n", encoding="utf-8")
    reviewed_root.chmod(0o770)
    reviewed_payload.chmod(0o660)
    runtime_root = runtime_root.resolve(strict=True)

    replacement = tmp_path / "replacement"
    (replacement / "bin").mkdir(parents=True)
    replacement_root = replacement / "lib" / "python3.11" / "site-packages"
    replacement_root.mkdir(parents=True)
    replacement_payload = replacement_root / "replacement.py"
    replacement_payload.write_text("replacement = True\n", encoding="utf-8")
    replacement_root.chmod(0o770)
    replacement_payload.chmod(0o660)
    candidates = [str(reviewed_root)]

    def discover(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(candidates),
            stderr="",
        )

    real_open = os.open
    reviewed_backup = tmp_path / "runtime-reviewed"
    swapped = False

    def swap_before_root_open(
        path: str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "/" and dir_fd is None and not swapped:
            runtime_root.rename(reviewed_backup)
            replacement.rename(runtime_root)
            swapped = True
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(qualification_sentinels.subprocess, "run", discover)
    monkeypatch.setattr(qualification_sentinels.os, "open", swap_before_root_open)
    try:
        with pytest.raises(RuntimeError, match="changed during validation"):
            qualification_sentinels._make_site_packages_read_only(
                runtime_root / "bin" / "python"
            )
        assert stat.S_IMODE(
            (reviewed_backup / "lib/python3.11/site-packages").stat().st_mode
        ) == 0o770
        assert stat.S_IMODE(
            (
                reviewed_backup
                / "lib/python3.11/site-packages/reviewed.py"
            ).stat().st_mode
        ) == 0o660
        assert stat.S_IMODE(
            (runtime_root / "lib/python3.11/site-packages").stat().st_mode
        ) == 0o770
        assert stat.S_IMODE(
            (
                runtime_root
                / "lib/python3.11/site-packages/replacement.py"
            ).stat().st_mode
        ) == 0o660
    finally:
        if swapped:
            runtime_root.rename(replacement)
            reviewed_backup.rename(runtime_root)


def test_site_package_lockdown_rejects_post_validation_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_root = tmp_path / "runtime"
    (runtime_root / "bin").mkdir(parents=True)
    reviewed_root = runtime_root / "lib" / "python3.11" / "site-packages"
    reviewed_root.mkdir(parents=True)
    reviewed_payload = reviewed_root / "reviewed.py"
    reviewed_payload.write_text("reviewed = True\n", encoding="utf-8")
    reviewed_root.chmod(0o770)
    reviewed_payload.chmod(0o660)
    runtime_root = runtime_root.resolve(strict=True)

    external_lib = tmp_path / "external" / "lib"
    external_root = external_lib / "python3.11" / "site-packages"
    external_root.mkdir(parents=True)
    external_payload = external_root / "external.py"
    external_payload.write_text("external = True\n", encoding="utf-8")
    external_root.chmod(0o770)
    external_payload.chmod(0o660)
    candidates = [str(reviewed_root)]

    def discover(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(candidates),
            stderr="",
        )

    real_open_tree = qualification_sentinels._open_validated_site_packages_tree
    reviewed_lib = runtime_root / "lib-reviewed"

    def swap_after_validation(runtime: Path, raw_paths: object):
        descriptor, snapshot = real_open_tree(runtime, raw_paths)
        (runtime_root / "lib").rename(reviewed_lib)
        (runtime_root / "lib").symlink_to(external_lib, target_is_directory=True)
        return descriptor, snapshot

    monkeypatch.setattr(qualification_sentinels.subprocess, "run", discover)
    monkeypatch.setattr(
        qualification_sentinels,
        "_open_validated_site_packages_tree",
        swap_after_validation,
    )
    try:
        with pytest.raises(RuntimeError, match="safely reopen"):
            qualification_sentinels._make_site_packages_read_only(
                runtime_root / "bin" / "python"
            )
        assert stat.S_IMODE(external_root.stat().st_mode) == 0o770
        assert stat.S_IMODE(external_payload.stat().st_mode) == 0o660
        assert stat.S_IMODE((reviewed_lib / "python3.11/site-packages").stat().st_mode) == 0o770
        assert stat.S_IMODE(
            (reviewed_lib / "python3.11/site-packages/reviewed.py").stat().st_mode
        ) == 0o660
    finally:
        (runtime_root / "lib").unlink()
        reviewed_lib.rename(runtime_root / "lib")


def test_site_package_scan_error_fails_freezer_and_worker_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_root = tmp_path / "runtime"
    (runtime_root / "bin").mkdir(parents=True)
    actual_root = runtime_root / "lib" / "python3.11" / "site-packages"
    actual_root.mkdir(parents=True)
    payload = actual_root / "worker.py"
    payload.write_text("reviewed = True\n", encoding="utf-8")
    actual_root.chmod(0o770)
    payload.chmod(0o660)
    runtime_root = runtime_root.resolve(strict=True)
    candidates = _debian_site_package_candidates(runtime_root)

    def discover(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(candidates),
            stderr="",
        )

    def denied_scan(_descriptor: int):
        raise PermissionError("injected site-packages scan denial")

    monkeypatch.setattr(qualification_sentinels.subprocess, "run", discover)
    monkeypatch.setattr(qualification_sentinels.os, "scandir", denied_scan)
    monkeypatch.setattr(sentinel_worker.sys, "prefix", str(runtime_root))
    monkeypatch.setattr(site, "getsitepackages", lambda: candidates)

    assert sentinel_worker._site_packages_read_only() is False
    with pytest.raises(RuntimeError, match="could not scan"):
        qualification_sentinels._make_site_packages_read_only(
            runtime_root / "bin" / "python"
        )
    assert stat.S_IMODE(actual_root.stat().st_mode) == 0o770
    assert stat.S_IMODE(payload.stat().st_mode) == 0o660


def test_plan_freezes_the_complete_bounded_gpu_qualification_matrix():
    plan = _valid_plan()
    validate_gpu_qualification_plan_record(
        plan,
        expected_campaign_id=CAMPAIGN_ID,
        expected_artifact_pins=PINS,
    )

    runtime = plan["runtime_contract"]
    assert runtime["vllm_version"] == GPU_QUALIFICATION_VLLM_VERSION
    assert runtime["official_vllm_wheel_sha256"] == (
        GPU_QUALIFICATION_OFFICIAL_WHEEL_SHA256
    )
    assert runtime["artifact_sha256"] == PINS.to_record()
    assert runtime["publication_attention_backend"] == (
        GPU_QUALIFICATION_PUBLICATION_BACKEND
    )
    assert runtime["publication_attention_backend_argument"] == [
        "--attention-backend",
        "TRITON_ATTN",
    ]
    assert runtime["attention_backend_auto_selection"] == "diagnostic_only"
    assert runtime["weight_quantization"] == "bitsandbytes"
    assert runtime["weight_bits"] == 4
    assert runtime["compute_dtype"] == "bfloat16"
    assert runtime["runtime_kv_dtype"] == "fp8_e5m2"
    assert runtime["runtime_kv_bits"] == 8
    assert runtime["handoff_kv_dtype"] == "fp8_e5m2"
    assert runtime["handoff_kv_bits"] == 8
    assert runtime["locked_core_distribution_versions"] == CORE_VERSIONS
    assert runtime["installed_patch_member_sha256"] == PATCH_MEMBERS
    assert runtime["platform"] == {
        "glibc_version": "2.35",
        "python_version": "3.11.11",
        "system_cuda_version": "12.1",
        "torch_cuda_version": "12.9",
    }

    cloud = plan["cloud_qualification"]
    assert cloud["job_count"] == 14
    assert cloud["job_count"] <= GPU_QUALIFICATION_MAX_CLOUD_JOBS
    assert cloud["success_attempt_number"] == 0
    assert cloud["max_retries"] == 0
    assert all(job["attempt_number"] == 0 for job in cloud["jobs"])
    assert all(job["max_retries"] == 0 for job in cloud["jobs"])
    assert [job["sentinel"] for job in cloud["jobs"]].count("l4_32k_c4_gmu_sweep") == 3
    assert [job["sentinel"] for job in cloud["jobs"]].count("a10g_16k_c4_capacity") == 1
    assert {
        job["hardware_id"]
        for job in cloud["jobs"]
        if job["sentinel"] == "forced_triton_runtime_handoff"
    } == {"aws-g6-l4", "aws-g5-a10g"}
    assert all(
        job["backend_mode"] == "forced_triton"
        for job in cloud["jobs"]
        if job["evidence_class"] == "publication_gate"
    )
    assert all(
        job["backend_mode"] == "auto" and job["evidence_class"] == "diagnostic_only"
        for job in cloud["jobs"]
        if job["sentinel"] == "auto_backend_diagnostic"
    )
    assert plan["local_preflight"] == {
        "check_ids": [
            "canonical_plan_schema",
            "runtime_lock_require_hashes",
            "patched_wheel_record_and_manifest",
            "source_runner_input_closure",
            "unit_tests",
            "ruff",
            "mypy",
        ],
        "cloud_success_credit": False,
        "must_complete_before_cloud": True,
        "scope": "local_preflight_only",
    }
    assert plan["unsupported_methods"] == [
        {
            "method_id": "lmcache",
            "publication_status": "N/A",
            "reason": "no_vllm_0271_combined_hash_lock_or_gpu_qualification",
        },
        {
            "method_id": "multi",
            "publication_status": "N/A",
            "reason": "no_vllm_0271_combined_hash_lock_or_gpu_qualification",
        },
    ]


def test_valid_first_attempt_evidence_selects_highest_safe_gmu():
    plan, evidence = _valid_evidence()

    selection = validate_gpu_qualification_evidence_record(
        evidence,
        plan_record=plan,
        expected_campaign_id=CAMPAIGN_ID,
        expected_artifact_pins=PINS,
    )

    assert selection.attention_backend == "TRITON_ATTN"
    assert selection.gpu_memory_utilization == 0.75
    assert selection.plan_sha256 == plan["closed_record_sha256"]
    cloud = evidence["cloud_gpu_evidence"]
    assert cloud["max_parallel_jobs_observed"] == 14
    assert cloud["job_count"] == 14


def test_caller_supplied_cloud_builder_is_explicitly_nonauthorizing():
    plan, governed = _valid_evidence()
    synthetic_cloud = build_cloud_gpu_evidence(
        plan_sha256=plan["closed_record_sha256"],
        jobs=governed["cloud_gpu_evidence"]["jobs"],
        selected_gpu_memory_utilization=0.75,
    )
    assert synthetic_cloud["authorization_source"] == ("caller_supplied_nonauthorizing")
    assert synthetic_cloud["terminal_receipts"] == []
    synthetic = build_gpu_qualification_evidence(
        campaign_id=CAMPAIGN_ID,
        plan_sha256=plan["closed_record_sha256"],
        local_preflight_evidence=governed["local_preflight_evidence"],
        cloud_gpu_evidence=synthetic_cloud,
    )
    assert synthetic["qualification_status"] == "unverified"
    with pytest.raises(ValueError, match="must declare passed"):
        validate_gpu_qualification_evidence_record(
            synthetic,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def test_record_only_builder_cannot_promote_resealed_governed_labels():
    plan, governed = _valid_evidence()
    synthetic_cloud = build_cloud_gpu_evidence(
        plan_sha256=plan["closed_record_sha256"],
        jobs=deepcopy(governed["cloud_gpu_evidence"]["jobs"]),
        selected_gpu_memory_utilization=0.75,
    )
    synthetic_cloud["terminal_receipts"] = deepcopy(
        governed["cloud_gpu_evidence"]["terminal_receipts"]
    )
    synthetic_cloud["authorization_source"] = "direct_databricks_runs_get"
    synthetic_cloud["scope"] = "governed_cloud_gpu_terminal_evidence"
    _seal(synthetic_cloud)

    synthetic = build_gpu_qualification_evidence(
        campaign_id=CAMPAIGN_ID,
        plan_sha256=plan["closed_record_sha256"],
        local_preflight_evidence=governed["local_preflight_evidence"],
        cloud_gpu_evidence=synthetic_cloud,
    )

    assert synthetic["qualification_status"] == "unverified"
    with pytest.raises(ValueError, match="must declare passed"):
        validate_gpu_qualification_evidence_record(
            synthetic,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


@pytest.mark.parametrize(
    "identity", ["cloud_run_id", "cloud_cluster_id", "task_run_id"]
)
def test_persisted_evidence_rejects_reused_independent_job_identity(identity: str):
    plan, evidence = _valid_evidence()
    cloud = evidence["cloud_gpu_evidence"]
    if identity in {"cloud_run_id", "cloud_cluster_id"}:
        replacement = "10000" if identity == "cloud_run_id" else "shared-cluster"
        for job in cloud["jobs"]:
            job[identity] = replacement
        for receipt in cloud["terminal_receipts"]:
            receipt[identity] = replacement
    else:
        for receipt in cloud["terminal_receipts"]:
            receipt[identity] = "20000"
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match=f"{identity} values must be unique"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("cloud_run_id", "run-1", "canonical decimal run ID"),
        ("reservation_attempt_id", "reservation-1", "does not match the plan"),
        ("task_key", "qualification_task_1", "does not match the plan"),
        ("output_json", "dbfs:/wrong/gpu-job-result.json", "frozen plan/job path"),
    ],
)
def test_persisted_evidence_rejects_nondeterministic_result_bindings(
    field_name: str, replacement: str, message: str
):
    plan, evidence = _valid_evidence()
    job = evidence["cloud_gpu_evidence"]["jobs"][0]
    receipt = evidence["cloud_gpu_evidence"]["terminal_receipts"][0]
    job[field_name] = replacement
    if field_name in receipt:
        receipt[field_name] = replacement
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match=message):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def test_terminal_receipt_rejects_resealed_ledger_reconciliation_tamper():
    plan, evidence = _valid_evidence()
    receipt = evidence["cloud_gpu_evidence"]["terminal_receipts"][0]
    receipt["ledger_actual_cluster_duration_seconds"] = 1.0
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match="ledger duration"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def test_plan_rejects_tampering_even_with_a_recomputed_digest():
    plan = _valid_plan()
    plan["runtime_contract"]["publication_attention_backend"] = "AUTO"
    _seal(plan)

    with pytest.raises(ValueError, match="frozen plan"):
        validate_gpu_qualification_plan_record(
            plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def test_plan_rejects_a_different_exact_artifact_closure():
    plan = _valid_plan()
    different_pins = GPUQualificationArtifactPins(
        runtime_lock_sha256="a" * 64,
        patched_vllm_wheel_sha256="b" * 64,
        package_wheel_sha256="f" * 64,
        cachet_source_tree_sha256="c" * 64,
        runner_sha256="d" * 64,
        input_bundle_sha256="e" * 64,
    )

    with pytest.raises(ValueError, match="frozen plan"):
        validate_gpu_qualification_plan_record(
            plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=different_pins,
        )


def test_patched_wheel_pin_cannot_be_the_pristine_wheel():
    with pytest.raises(ValueError, match="repacked wheel"):
        GPUQualificationArtifactPins(
            runtime_lock_sha256="1" * 64,
            patched_vllm_wheel_sha256=GPU_QUALIFICATION_OFFICIAL_WHEEL_SHA256,
            package_wheel_sha256="6" * 64,
            cachet_source_tree_sha256="3" * 64,
            runner_sha256="4" * 64,
            input_bundle_sha256="5" * 64,
        )


def test_evidence_rejects_missing_cloud_job():
    plan, evidence = _valid_evidence()
    evidence["cloud_gpu_evidence"]["jobs"].pop()
    evidence["cloud_gpu_evidence"]["job_count"] -= 1
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match="every planned job"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def test_evidence_rejects_retry_success_even_when_resealed():
    plan, evidence = _valid_evidence()
    result = evidence["cloud_gpu_evidence"]["jobs"][0]
    result["attempt_number"] = 1
    result["retry_count"] = 1
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match="attempt 0 without retries"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


@pytest.mark.parametrize(
    ("job_suffix", "mutation", "message"),
    [
        (
            "forced-triton-runtime-handoff",
            lambda m: m.update(attention_backend_observed="FLASHINFER"),
            "attention_backend_observed",
        ),
        (
            "packed-page-roundtrip",
            lambda m: m["cases"][0].update(read_raw_sha256="f" * 64),
            "raw bytes did not round-trip",
        ),
        (
            "matched-token-logit",
            lambda m: m["examples"][0].update(
                vanilla_reconstructed_full_prompt_token_ids_sha256="f" * 64
            ),
            "token contracts differ",
        ),
        (
            "auto-backend-diagnostic",
            lambda m: m.update(publication_backend_changed=True),
            "cannot change publication backend",
        ),
    ],
)
def test_semantic_sentinels_reject_resealed_failures(job_suffix, mutation, message):
    plan, evidence = _valid_evidence()
    job = next(
        item
        for item in evidence["cloud_gpu_evidence"]["jobs"]
        if item["job_id"].startswith("aws-g6-l4")
        and item["job_id"].endswith(job_suffix)
    )
    mutation(job["measurements"])
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match=message):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def test_a10g_must_exercise_software_e5m2_and_real_triton_compile():
    plan, evidence = _valid_evidence()
    result = next(
        item
        for item in evidence["cloud_gpu_evidence"]["jobs"]
        if item["job_id"] == "aws-g5-a10g-forced-triton-runtime-handoff"
    )
    result["measurements"]["e5m2_software_path_exercised"] = False
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match="A10G must exercise"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )

    _, evidence = _valid_evidence()
    result = evidence["cloud_gpu_evidence"]["jobs"][0]
    result["measurements"]["triton_compile_count"] = 0
    _reseal_evidence(evidence)
    with pytest.raises(ValueError, match="triton_compile_count"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def test_runtime_attestation_rejects_cuda13_linkage_and_synthetic_token_probe():
    plan, evidence = _valid_evidence()
    result = next(
        item
        for item in evidence["cloud_gpu_evidence"]["jobs"]
        if item["job_id"] == "aws-g6-l4-forced-triton-runtime-handoff"
    )
    result["measurements"]["libcudart_so_13_present"] = True
    _reseal_evidence(evidence)
    with pytest.raises(ValueError, match="libcudart_so_13_present"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )

    _, evidence = _valid_evidence()
    result = next(
        item
        for item in evidence["cloud_gpu_evidence"]["jobs"]
        if item["job_id"] == "aws-g6-l4-matched-token-logit"
    )
    result["measurements"]["execution_mode"] = "synthetic"
    _reseal_evidence(evidence)
    with pytest.raises(ValueError, match="execution_mode"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def test_a10g_campaign_shaped_16k_c4_capacity_gate_is_required():
    plan, evidence = _valid_evidence()
    result = next(
        item
        for item in evidence["cloud_gpu_evidence"]["jobs"]
        if item["job_id"] == "aws-g5-a10g-16k-c4-capacity"
    )
    measurements = result["measurements"]
    total = measurements["observed_total_memory_bytes"]
    measurements["observed_peak_headroom_bytes"] = 1024**3
    measurements["observed_peak_used_memory_bytes"] = total - 1024**3
    measurements["capacity_qualified"] = False
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match="capacity/headroom"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def test_gmu_sweep_rejects_oom_and_requires_a_qualified_candidate():
    plan, evidence = _valid_evidence()
    gmu_jobs = [
        job
        for job in evidence["cloud_gpu_evidence"]["jobs"]
        if "32k-c4-gmu" in job["job_id"]
    ]
    gmu_jobs[0]["measurements"]["oom_count"] = 1
    _reseal_evidence(evidence)
    with pytest.raises(ValueError, match="zero OOM"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )

    _, evidence = _valid_evidence()
    for job in evidence["cloud_gpu_evidence"]["jobs"]:
        if "32k-c4-gmu" not in job["job_id"]:
            continue
        measurements = job["measurements"]
        measurements["kv_cache_capacity_tokens"] = (
            GPU_QUALIFICATION_REQUIRED_KV_CAPACITY_TOKENS - 1
        )
        measurements["candidate_qualified"] = False
    _reseal_evidence(evidence)
    with pytest.raises(ValueError, match="no 32k c4 GMU candidate"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def test_gmu_selected_value_must_be_highest_candidate_meeting_both_gates():
    plan, evidence = _valid_evidence()
    evidence["cloud_gpu_evidence"]["selected_gpu_memory_utilization"] = 0.70
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match="highest qualifying"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )
    assert GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES == 2 * 1024**3
    assert GPU_QUALIFICATION_REQUIRED_KV_CAPACITY_TOKENS == 4 * (32768 + 512)


def test_throughput_gate_recomputes_end_to_end_rate_including_writes():
    plan, evidence = _valid_evidence()
    result = next(
        item
        for item in evidence["cloud_gpu_evidence"]["jobs"]
        if item["job_id"] == "aws-g6e-l40s-generation-throughput"
    )
    bucket = result["measurements"]["buckets"][1]
    bucket["wall_seconds"] = bucket["prefix_tokens"] / 34.0
    bucket["tokens_per_second"] = 34.0
    result["measurements"]["aggregate_wall_seconds"] = sum(
        item["wall_seconds"] for item in result["measurements"]["buckets"]
    )
    result["measurements"]["aggregate_tokens_per_second"] = (
        result["measurements"]["aggregate_prefix_tokens"]
        / result["measurements"]["aggregate_wall_seconds"]
    )
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match="below the launch threshold"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )
    assert GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND == 35.0


def test_throughput_gate_binds_production_generator_device_map():
    plan, evidence = _valid_evidence()
    result = next(
        item
        for item in evidence["cloud_gpu_evidence"]["jobs"]
        if item["job_id"] == "aws-g6e-l40s-generation-throughput"
    )
    result["measurements"]["generator_device_map"] = "cuda"
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match="device_map=auto"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def test_local_preflight_is_distinct_and_must_finish_before_cloud():
    plan, evidence = _valid_evidence()
    local = evidence["local_preflight_evidence"]
    assert local["scope"] == "local_preflight_only_no_cloud_success_credit"
    assert evidence["cloud_gpu_evidence"]["scope"] == (
        "governed_cloud_gpu_terminal_evidence"
    )
    local["completed_at_utc"] = "2026-08-24T01:30:00Z"
    _seal(local)
    _seal(evidence)

    with pytest.raises(ValueError, match="complete before cloud"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def test_canonical_json_writer_is_deterministic_and_never_overwrites(tmp_path):
    plan = _valid_plan()
    output = tmp_path / "qualification-plan.json"

    write_gpu_qualification_plan_json(
        plan,
        output,
        expected_campaign_id=CAMPAIGN_ID,
        expected_artifact_pins=PINS,
    )

    payload = output.read_text(encoding="utf-8")
    assert payload == canonical_gpu_qualification_json(plan) + "\n"
    assert json.loads(payload) == plan
    with pytest.raises(FileExistsError):
        write_canonical_gpu_qualification_json(plan, output)


def test_evidence_writer_validates_before_emitting(tmp_path):
    plan, evidence = _valid_evidence()
    output = tmp_path / "qualification-evidence.json"

    selection = write_gpu_qualification_evidence_json(
        evidence,
        output,
        plan_record=plan,
        expected_campaign_id=CAMPAIGN_ID,
        expected_artifact_pins=PINS,
    )

    assert selection.gpu_memory_utilization == 0.75
    assert output.read_text(encoding="utf-8") == (
        canonical_gpu_qualification_json(evidence) + "\n"
    )
