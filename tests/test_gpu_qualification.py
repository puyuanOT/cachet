import hashlib
import io
import json
import os
import site
import signal
import stat
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
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
        "loaded_native_library_member": "bitsandbytes/libbitsandbytes_cuda129.so",
        "loaded_native_library_path": (
            "/runtime/lib/python3.11/site-packages/"
            "bitsandbytes/libbitsandbytes_cuda129.so"
        ),
    }


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _seal(record: dict[str, Any]) -> None:
    payload = deepcopy(record)
    payload.pop("closed_record_sha256", None)
    record["closed_record_sha256"] = hashlib.sha256(
        canonical_gpu_qualification_json(payload).encode()
    ).hexdigest()


def _set_native_record_soname_bindings(
    record: dict[str, Any], bindings: dict[str, str | None]
) -> None:
    ordered_bindings = sorted(bindings.items())
    lines = [
        (
            f"{soname} => not found"
            if resolved_path is None
            else f"{soname} => {resolved_path} (0x00000001)"
        )
        for soname, resolved_path in ordered_bindings
    ]
    stdout = "".join(f"{line}\n" for line in lines)
    stdout_bytes = stdout.encode("utf-8")
    record.update(
        {
            "ldd_stdout": stdout,
            "ldd_stdout_lines": lines,
            "ldd_stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
            "ldd_stdout_utf8_bytes": len(stdout_bytes),
            "soname_bindings": [
                {"resolved_path": resolved_path, "soname": soname}
                for soname, resolved_path in ordered_bindings
            ],
        }
    )


def _native_shared_object_evidence() -> list[dict[str, Any]]:
    root = "/runtime/lib/python3.11/site-packages"
    resolved_members = (
        (
            "bitsandbytes",
            CORE_VERSIONS["bitsandbytes"],
            "bitsandbytes/libbitsandbytes_cuda129.so",
        ),
        ("torch", CORE_VERSIONS["torch"], "torch/lib/libtorch.so.2"),
        ("triton", CORE_VERSIONS["triton"], "triton/_C/libtriton.so"),
        ("vllm", GPU_QUALIFICATION_VLLM_VERSION, "vllm/_C.abi3.so"),
    )
    dormant_members = {
        "bitsandbytes/libbitsandbytes_cuda118.so": frozenset(
            {
                "libcublas.so.11",
                "libcublasLt.so.11",
                "libcudart.so.11.0",
                "libcusparse.so.11",
            }
        ),
        "bitsandbytes/libbitsandbytes_cuda130.so": frozenset(
            {
                "libcublas.so.13",
                "libcublasLt.so.13",
                "libcudart.so.13",
                "libnvJitLink.so.13",
            }
        ),
        **{
            f"bitsandbytes/libbitsandbytes_rocm{version}.so": frozenset(
                {
                    "libhipblas.so.2",
                    "libhipblaslt.so.0",
                    "libhipsparse.so.1",
                }
            )
            for version in ("62", "63", "64")
        },
        **{
            f"bitsandbytes/libbitsandbytes_rocm{version}.so": frozenset(
                {
                    "libhipblas.so.3",
                    "libhipblaslt.so.1",
                    "libhipsparse.so.4",
                }
            )
            for version in ("70", "71", "72")
        },
        "bitsandbytes/libbitsandbytes_xpu.so": frozenset(
            {
                "libimf.so",
                "libintlc.so.5",
                "libirng.so",
                "libsvml.so",
                "libsycl.so.8",
            }
        ),
        "triton/plugins/libMLIRDialectPlugin.so": frozenset({"libtriton.so"}),
        "triton/plugins/libMLIRDialectPlugin.so.23.0git": frozenset(
            {"libtriton.so"}
        ),
        "triton/plugins/libTritonPluginsTestLib.so": frozenset({"libtriton.so"}),
    }
    evidence: list[dict[str, Any]] = []
    for distribution, version, member in resolved_members:
        resolved_binding = f"/lib/{distribution}/libc.so.6"
        record = _native_shared_object_record(
            distribution=distribution,
            version=version,
            member=member,
            root=root,
            resolution_scope="runtime_reachable",
        )
        _set_native_record_soname_bindings(
            record, {"libc.so.6": resolved_binding}
        )
        evidence.append(record)
    for member, missing_sonames in dormant_members.items():
        distribution = member.split("/", 1)[0]
        version = CORE_VERSIONS[distribution]
        resolved_member = (
            "triton/plugins/libMLIRDialectPlugin.so.23.0git"
            if member == "triton/plugins/libMLIRDialectPlugin.so"
            else member
        )
        record = _native_shared_object_record(
            distribution=distribution,
            version=version,
            member=member,
            root=root,
            resolution_scope="platform_inapplicable",
            resolved_member=resolved_member,
        )
        _set_native_record_soname_bindings(
            record, {soname: None for soname in missing_sonames}
        )
        evidence.append(record)
    return sorted(
        evidence,
        key=lambda record: (
            record["distribution"],
            record["member"],
            record["path"],
        ),
    )


def _native_shared_object_record(
    *,
    distribution: str,
    version: str,
    member: str,
    root: str,
    resolution_scope: str,
    resolved_member: str | None = None,
) -> dict[str, Any]:
    resolved_member = member if resolved_member is None else resolved_member
    return {
        "distribution": distribution,
        "distribution_version": version,
        "is_symlink": resolved_member != member,
        "ldd_returncode": 0,
        "ldd_stderr": "",
        "ldd_stderr_lines": [],
        "ldd_stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "ldd_stderr_utf8_bytes": 0,
        "ldd_stdout": "",
        "ldd_stdout_lines": [],
        "ldd_stdout_sha256": hashlib.sha256(b"").hexdigest(),
        "ldd_stdout_utf8_bytes": 0,
        "member": member,
        "path": f"{root}/{member}",
        "resolved_path": f"{root}/{resolved_member}",
        "resolution_scope": resolution_scope,
        "soname_bindings": [],
    }


def _runtime_measurements(*, software_path: bool) -> dict[str, Any]:
    native_shared_objects = _native_shared_object_evidence()
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
        "native_shared_object_count": len(native_shared_objects),
        "native_shared_object_evidence": native_shared_objects,
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
        "unresolved_native_shared_object_count": 12,
        "unresolved_runtime_reachable_native_shared_object_count": 0,
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


@pytest.mark.parametrize(
    ("extra_bytes", "truncated"),
    [(0, False), (1, True)],
)
def test_worker_binary_capture_enforces_exact_tail_boundaries(
    tmp_path: Path,
    extra_bytes: int,
    truncated: bool,
):
    stdout_tail_max_bytes = qualification_sentinels._WORKER_STDOUT_TAIL_MAX_BYTES
    stderr_tail_max_bytes = qualification_sentinels._WORKER_STDERR_TAIL_MAX_BYTES
    assert stdout_tail_max_bytes == 2_000
    assert stderr_tail_max_bytes == 16_384
    stdout = b"A" * (stdout_tail_max_bytes + extra_bytes)
    stderr = b"B" * (stderr_tail_max_bytes + extra_bytes)
    code = (
        "import os; "
        f"os.write(1,{stdout!r}); "
        f"os.write(2,{stderr!r})"
    )

    result = qualification_sentinels._run_bounded_worker_process(
        [sys.executable, "-c", code],
        job_id="tail-boundary",
        timeout_seconds=5,
        environment=os.environ,
        cwd=tmp_path,
        drain_timeout_seconds=0.5,
        termination_grace_seconds=0.5,
    )

    assert result.returncode == 0
    assert result.stdout.byte_count == len(stdout)
    assert result.stdout.sha256 == hashlib.sha256(stdout).hexdigest()
    assert result.stdout.truncated is truncated
    assert len(result.stdout.tail) == stdout_tail_max_bytes
    assert result.stdout.tail == stdout[-stdout_tail_max_bytes:]
    assert result.stderr.byte_count == len(stderr)
    assert result.stderr.sha256 == hashlib.sha256(stderr).hexdigest()
    assert result.stderr.truncated is truncated
    assert len(result.stderr.tail) == stderr_tail_max_bytes
    assert result.stderr.tail == stderr[-stderr_tail_max_bytes:]


def test_worker_nonzero_diagnostic_is_exact_and_utf8_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stdout = b"\xffA"
    stderr = b"X\xe2\x82\xac"
    stderr_tail = stderr[-2:]
    code = (
        "import os,sys; "
        f"os.write(1,{stdout!r}); "
        f"os.write(2,{stderr!r}); "
        "sys.exit(7)"
    )
    expected = (
        "GPU sentinel 'invalid-utf8' worker exited with status 7; "
        f"stdout(bytes=2,sha256={hashlib.sha256(stdout).hexdigest()},"
        "truncated=false,tail='�A'); "
        f"stderr(bytes=4,sha256={hashlib.sha256(stderr).hexdigest()},"
        f"truncated=true,tail={stderr_tail.decode('utf-8', errors='replace')!r})"
    )

    with pytest.raises(RuntimeError) as captured:
        qualification_sentinels._run_bounded_worker_process(
            [sys.executable, "-c", code],
            job_id="invalid-utf8",
            timeout_seconds=5,
            environment=os.environ,
            cwd=tmp_path,
            stdout_tail_max_bytes=8,
            stderr_tail_max_bytes=2,
            drain_timeout_seconds=0.5,
            termination_grace_seconds=0.5,
        )

    assert str(captured.value) == expected

    serialized_request: dict[str, Any] = {}

    class _CompletionResponse:
        def __enter__(self) -> "_CompletionResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "logprobs": {"token_logprobs": [-0.25]},
                            "token_ids": [7],
                        }
                    ]
                }
            ).encode("utf-8")

    def _urlopen(request: Any, *, timeout: int) -> _CompletionResponse:
        serialized_request.update(json.loads(request.data.decode("utf-8")))
        assert timeout == 600
        return _CompletionResponse()

    monkeypatch.setattr(sentinel_worker.urllib.request, "urlopen", _urlopen)
    completion = sentinel_worker._completion_request(
        endpoint="https://example.invalid/v1/completions",
        model="qualification-model",
        prompt="logical full prompt",
        kv_transfer_params={"document_kv": {"prompt_text_mode": "logical"}},
        cache_salt="qualification-capacity:test",
        max_tokens=1,
        request_id="gpuq-capacity-e00",
        sentinel_phase="capacity",
        arm_id="vanilla_prefill",
        example_ordinal=0,
        repeat_ordinal=0,
    )
    assert completion == {"token_ids": [7], "token_logprobs": [-0.25]}
    assert serialized_request["add_special_tokens"] is False
    assert serialized_request["prompt"] == "logical full prompt"
    assert serialized_request["request_id"] == "gpuq-capacity-e00"

    class _OversizedCompletionResponse(_CompletionResponse):
        def read(self, size: int = -1) -> bytes:
            return b"R" * size

    monkeypatch.setattr(
        sentinel_worker.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _OversizedCompletionResponse(),
    )
    with pytest.raises(
        sentinel_worker._QualificationCompletionFailure
    ) as oversized_failure:
        sentinel_worker._completion_request(
            endpoint="https://example.invalid/v1/completions",
            model="qualification-model",
            prompt="logical full prompt",
            kv_transfer_params=None,
            cache_salt="qualification-capacity:test",
            max_tokens=1,
            request_id="gpuq-capacity-e00",
            sentinel_phase="capacity",
            arm_id="vanilla_prefill",
            example_ordinal=0,
            repeat_ordinal=0,
        )
    oversized_diagnostic = oversized_failure.value.request_diagnostic
    assert oversized_diagnostic["captured_byte_count"] == 256 * 1024
    assert oversized_diagnostic["captured_truncated"] is True
    assert oversized_diagnostic["kind"] == "response_too_large"

    secret = b"private-prompt-fragment"
    error_body = (
        b"ValueError: request exposes 8158 token IDs, fewer than the cached "
        b"prefix length 8192; "
        + secret
        + b"X" * 20_000
    )
    http_error = sentinel_worker.urllib.error.HTTPError(
        "https://example.invalid/v1/completions",
        500,
        "Internal Server Error",
        {},
        io.BytesIO(error_body),
    )
    diagnostic = sentinel_worker._bounded_http_error_diagnostic(http_error)
    captured_body = error_body[: 16 * 1024]
    assert diagnostic == {
        "captured_byte_count": len(captured_body),
        "captured_sha256": hashlib.sha256(captured_body).hexdigest(),
        "captured_truncated": True,
        "category": "token_contract_shortfall",
        "counts": {
            "cached_prefix_token_count": 8192,
            "visible_token_count": 8158,
        },
        "exception_type": "ValueError",
        "http_status": 500,
        "kind": "http_error",
    }
    nested_body = json.dumps(
        {
            "error": {
                "message": secret.decode("ascii"),
                "type": "InternalServerError",
            }
        }
    ).encode("utf-8")
    nested_error = sentinel_worker.urllib.error.HTTPError(
        "https://example.invalid/v1/completions",
        500,
        "Internal Server Error",
        {},
        io.BytesIO(nested_body),
    )
    nested_diagnostic = sentinel_worker._bounded_http_error_diagnostic(nested_error)
    assert nested_diagnostic["api_error_type"] == "InternalServerError"
    assert secret.decode("ascii") not in json.dumps(nested_diagnostic)

    request_id = "gpuq-matched-e00-vanilla-r0"
    split = 13
    log_bytes = (
        b"L" * (1024 * 1024 - split)
        + request_id[:split].encode("utf-8")
        + request_id[split:].encode("utf-8")
        + b" EngineDeadError enginecore failed "
        + secret
    )
    log_path = tmp_path / "qualification-server.log"
    log_path.write_bytes(log_bytes)
    server_diagnostic = sentinel_worker._server_log_failure_diagnostic(
        log_path,
        known_request_ids=(request_id, "gpuq-matched-e00-baseline-r0"),
    )
    assert server_diagnostic == {
        "available": True,
        "byte_count": len(log_bytes),
        "category": "engine_dead",
        "exception_type": "EngineDeadError",
        "request_ids_seen": [request_id],
        "sha256": hashlib.sha256(log_bytes).hexdigest(),
        "tail_inspected_byte_count": 256 * 1024,
        "tail_truncated": True,
    }
    assert sentinel_worker._server_log_failure_diagnostic(
        tmp_path / "missing-server.log",
        known_request_ids=(request_id,),
    ) == {
        "available": False,
        "category": "unknown",
        "io_error_type": "FileNotFoundError",
        "unavailable_reason": "io_error",
    }
    symlink_path = tmp_path / "symlink-server.log"
    symlink_path.symlink_to(log_path)
    assert sentinel_worker._server_log_failure_diagnostic(
        symlink_path,
        known_request_ids=(request_id,),
    ) == {
        "available": False,
        "category": "unknown",
        "io_error_type": "OSError",
        "unavailable_reason": "io_error",
    }

    failure = sentinel_worker._QualificationCompletionFailure(
        request_id=request_id,
        context=sentinel_worker._qualification_completion_context(
            request_id=request_id,
            sentinel_phase="matched_token",
            arm_id="vanilla_prefill",
            example_ordinal=0,
            repeat_ordinal=0,
        ),
        request_diagnostic=diagnostic,
    )
    with pytest.raises(RuntimeError) as bounded_failure:
        sentinel_worker._raise_qualification_completion_failure(
            (failure,),
            server_log_path=log_path,
            known_request_ids=(request_id, "gpuq-matched-e00-baseline-r0"),
        )
    failure_text = str(bounded_failure.value)
    assert secret.decode("ascii") not in failure_text
    assert "\n" not in failure_text
    assert len(failure_text.encode("utf-8")) < 4 * 1024
    failure_record = json.loads(
        failure_text.removeprefix("qualification completion failure: ")
    )
    assert failure_record["requests"][0]["request_id"] == request_id

    capacity_ids = tuple(
        f"gpuq-capacity-e{ordinal:02d}" for ordinal in range(4)
    )
    real_completion_request = sentinel_worker._completion_request
    monkeypatch.setattr(
        sentinel_worker,
        "_completion_request",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("private failure")),
    )
    capacity_requests = [
        {
            "cache_salt": f"capacity:{ordinal}",
            "example_ordinal": ordinal,
            "kv_transfer_params": {"document_kv.request_id": request_id},
            "prompt": "logical full prompt",
            "request_id": request_id,
        }
        for ordinal, request_id in enumerate(capacity_ids)
    ]
    with pytest.raises(
        sentinel_worker._QualificationCompletionBatchFailure
    ) as capacity_batch:
        sentinel_worker._run_capacity_requests_concurrently(
            endpoint="https://example.invalid/v1/completions",
            model="qualification-model",
            requests=capacity_requests,
        )
    assert [failure.request_id for failure in capacity_batch.value.failures] == list(
        capacity_ids
    )
    assert all(
        failure.request_diagnostic
        == {
            "category": "unknown",
            "exception_type": "RuntimeError",
            "kind": "client_request_error",
        }
        for failure in capacity_batch.value.failures
    )
    capacity_loads = [
        {
            "benchmark_request_id": client_id,
            "counts": {"layers_loaded": 36},
            "request_id": f"cmpl-{client_id}-0",
        }
        for client_id in capacity_ids
    ]
    assert sentinel_worker._require_exact_connector_loads(
        capacity_loads,
        expected_client_request_ids=capacity_ids,
        label="capacity",
    ) == [36] * 4
    matched_ids = tuple(
        f"gpuq-matched-e{example:02d}-vanilla-r{repeat}"
        for example in range(4)
        for repeat in range(2)
    )
    matched_loads = [
        {
            "benchmark_request_id": client_id,
            "counts": {"layers_loaded": 36},
            "request_id": f"cmpl-{client_id}-0",
        }
        for client_id in matched_ids
    ]
    assert sentinel_worker._require_exact_connector_loads(
        matched_loads,
        expected_client_request_ids=matched_ids,
        label="matched-token",
    ) == [36] * 8
    matched_loads[-1]["request_id"] = "cmpl-tampered-0"
    with pytest.raises(RuntimeError, match="identity closure differs"):
        sentinel_worker._require_exact_connector_loads(
            matched_loads,
            expected_client_request_ids=matched_ids,
            label="matched-token",
        )

    from document_kv_cache import vllm_smoke
    from document_kv_cache._benchmark_datasets import _example_from_record
    from document_kv_cache.benchmarks import (
        DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM,
        DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
        DOCUMENT_KV_REQUEST_ID_PARAM,
        DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM,
        build_cache_prefix_text,
        build_cache_suffix_text,
        build_prefill_prompt,
    )

    protocol_records = [
        {
            "dataset": "biography",
            "documents": [f"Qualification source document {ordinal}."],
            "example_id": f"bio-{ordinal}",
            "expected_answer": f"Person {ordinal}",
            "query": f"Who is person {ordinal}?",
        }
        for ordinal in range(4)
    ]
    full_prompts: list[str] = []
    prefix_prompts: list[str] = []
    suffix_prompts: list[str] = []
    token_ids_by_text: dict[str, list[int]] = {}
    for ordinal, record in enumerate(protocol_records):
        example = _example_from_record(
            record,
            default_dataset="biography",
            record_index=ordinal + 1,
            require_dataset=True,
        )
        full_prompt = build_prefill_prompt(example)
        prefix_prompt = build_cache_prefix_text(example)
        suffix_prompt = build_cache_suffix_text(example)
        full_prompts.append(full_prompt)
        prefix_prompts.append(prefix_prompt)
        suffix_prompts.append(suffix_prompt)
        base = ordinal * 10_000
        prefix_ids = list(range(base, base + 8_000))
        suffix_ids = list(range(base + 8_000, base + 8_192))
        token_ids_by_text[prefix_prompt] = prefix_ids
        token_ids_by_text[suffix_prompt] = suffix_ids
        token_ids_by_text[full_prompt] = prefix_ids + suffix_ids

    tokenizer_loads: list[tuple[str, str, bool]] = []
    tokenizer_calls: list[tuple[str, bool]] = []

    class _ProtocolTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            tokenizer_calls.append((text, add_special_tokens))
            return list(token_ids_by_text[text])

    class _ProtocolAutoTokenizer:
        @staticmethod
        def from_pretrained(
            model_id: str,
            *,
            revision: str,
            trust_remote_code: bool,
        ) -> _ProtocolTokenizer:
            tokenizer_loads.append((model_id, revision, trust_remote_code))
            return _ProtocolTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_ProtocolAutoTokenizer),
    )
    runtime_prefixes = [
        f"preserved runtime-prefix metadata {ordinal}" for ordinal in range(4)
    ]
    params_by_key = {
        ("biography", f"bio-{ordinal}"): {
            DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM: f"stale-benchmark-{ordinal}",
            DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM: "runtime",
            DOCUMENT_KV_REQUEST_ID_PARAM: f"handoff-source-{ordinal}",
            DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM: runtime_prefixes[ordinal],
        }
        for ordinal in range(4)
    }
    built_capacity_requests = sentinel_worker._capacity_vanilla_requests(
        protocol_records,
        params_by_key=params_by_key,
        input_context_tokens=8_192,
    )
    assert [request["request_id"] for request in built_capacity_requests] == list(
        capacity_ids
    )
    for ordinal, request in enumerate(built_capacity_requests):
        assert request["prompt"] == full_prompts[ordinal]
        assert request["prompt"] != runtime_prefixes[ordinal]
        runtime_params = request["kv_transfer_params"]
        assert runtime_params[DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM] == capacity_ids[
            ordinal
        ]
        assert runtime_params[DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM] == "logical"
        assert (
            runtime_params[DOCUMENT_KV_REQUEST_ID_PARAM]
            == f"handoff-source-{ordinal}"
        )
        assert (
            runtime_params[DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM]
            == runtime_prefixes[ordinal]
        )

    monkeypatch.setattr(
        sentinel_worker,
        "_completion_request",
        real_completion_request,
    )
    matched_payloads: list[tuple[str, dict[str, Any]]] = []

    class _MatchedCompletionResponse(_CompletionResponse):
        def read(self, _size: int = -1) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "logprobs": {
                                "token_logprobs": [-0.25] * 16,
                            },
                            "token_ids": list(range(16)),
                        }
                    ]
                }
            ).encode("utf-8")

    def _matched_urlopen(request: Any, *, timeout: int) -> _MatchedCompletionResponse:
        assert timeout == 600
        matched_payloads.append(
            (request.full_url, json.loads(request.data.decode("utf-8")))
        )
        return _MatchedCompletionResponse()

    monkeypatch.setattr(sentinel_worker.urllib.request, "urlopen", _matched_urlopen)
    matched_results = sentinel_worker._run_matched_http_requests(
        config=SimpleNamespace(server_base_url="https://example.invalid"),
        selected_records=protocol_records,
        params_by_key=params_by_key,
        input_bundle_sha256="a" * 64,
        served_model_name="qualification-model",
    )
    expected_matched_requests = [
        (example_ordinal, arm_label, repeat_ordinal)
        for example_ordinal in range(4)
        for arm_label in ("baseline", "vanilla")
        for repeat_ordinal in range(2)
    ]
    assert len(matched_results) == 4
    assert [
        payload["request_id"] for _endpoint, payload in matched_payloads
    ] == [
        (
            f"gpuq-matched-e{example_ordinal:02d}-{arm_label}"
            f"-r{repeat_ordinal}"
        )
        for example_ordinal, arm_label, repeat_ordinal in expected_matched_requests
    ]
    for (endpoint, payload), (
        example_ordinal,
        arm_label,
        _repeat_ordinal,
    ) in zip(matched_payloads, expected_matched_requests, strict=True):
        assert endpoint == "https://example.invalid/v1/completions"
        assert payload["add_special_tokens"] is False
        assert payload["prompt"] == full_prompts[example_ordinal]
        assert payload["prompt"] != runtime_prefixes[example_ordinal]
        if arm_label == "baseline":
            assert "kv_transfer_params" not in payload
        else:
            runtime_params = payload["kv_transfer_params"]
            assert (
                runtime_params[DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM]
                == payload["request_id"]
            )
            assert runtime_params[DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM] == "logical"
            assert (
                runtime_params[DOCUMENT_KV_REQUEST_ID_PARAM]
                == f"handoff-source-{example_ordinal}"
            )
            assert (
                runtime_params[DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM]
                == runtime_prefixes[example_ordinal]
            )
    assert len(tokenizer_calls) == 24
    assert {text for text, _flag in tokenizer_calls} == {
        *full_prompts,
        *prefix_prompts,
        *suffix_prompts,
    }
    assert all(add_special_tokens is False for _text, add_special_tokens in tokenizer_calls)
    assert tokenizer_loads == [
        (
            sentinel_worker.GPU_QUALIFICATION_MODEL_ID,
            sentinel_worker.GPU_QUALIFICATION_MODEL_REVISION,
            False,
        )
    ] * 2

    server_log_path = tmp_path / "server-start" / "qualification.log"
    server_config = SimpleNamespace(server_log_path=server_log_path)
    popen_calls: list[tuple[list[str], dict[str, Any]]] = []
    process = object()

    def _build_server_args(config: Any, executable: Path) -> list[str]:
        assert config is server_config
        assert executable == Path(sys.executable)
        return [
            str(executable),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--no-enable-log-requests",
        ]

    def _server_env(config: Any) -> dict[str, str]:
        assert config is server_config
        return {"QUALIFICATION_PROTOCOL": "v2"}

    def _popen(argv: list[str], **kwargs: Any) -> object:
        popen_calls.append((list(argv), kwargs))
        return process

    monkeypatch.setattr(vllm_smoke, "build_vllm_server_args", _build_server_args)
    monkeypatch.setattr(vllm_smoke, "server_env", _server_env)
    monkeypatch.setattr(sentinel_worker.subprocess, "Popen", _popen)
    assert sentinel_worker._start_qualification_vllm_server(server_config) is process
    assert len(popen_calls) == 1
    server_argv, server_kwargs = popen_calls[0]
    assert server_argv.count("--no-enable-log-requests") == 1
    assert server_argv.count("--log-error-stack") == 1
    assert "--trust-remote-code" not in server_argv
    assert server_kwargs["stderr"] is subprocess.STDOUT
    assert server_kwargs["text"] is True
    assert server_kwargs["env"] == {"QUALIFICATION_PROTOCOL": "v2"}
    assert server_kwargs["stdout"].closed is True
    assert Path(server_kwargs["stdout"].name) == server_log_path


def test_worker_simultaneous_noisy_streams_are_drained_without_deadlock(
    tmp_path: Path,
):
    byte_count = 2 * 1024 * 1024
    stdout = b"O" * byte_count
    stderr = b"E" * byte_count
    code = f"""
import os
import sys
import threading

def emit(descriptor, value):
    chunk = value * 65536
    for _ in range({byte_count} // len(chunk)):
        os.write(descriptor, chunk)

threads = [
    threading.Thread(target=emit, args=(1, b'O')),
    threading.Thread(target=emit, args=(2, b'E')),
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
sys.exit(9)
"""

    with pytest.raises(RuntimeError) as captured:
        qualification_sentinels._run_bounded_worker_process(
            [sys.executable, "-c", code],
            job_id="noisy",
            timeout_seconds=10,
            environment=os.environ,
            cwd=tmp_path,
            stdout_tail_max_bytes=64,
            stderr_tail_max_bytes=96,
            drain_timeout_seconds=1,
            termination_grace_seconds=0.5,
        )

    message = str(captured.value)
    assert f"stdout(bytes={byte_count},sha256={hashlib.sha256(stdout).hexdigest()}" in message
    assert f"stderr(bytes={byte_count},sha256={hashlib.sha256(stderr).hexdigest()}" in message
    assert "truncated=true" in message


def test_worker_timeout_preserves_partial_stream_diagnostics(tmp_path: Path):
    stdout = b"partial-out"
    stderr = b"partial-err"
    code = (
        "import os,time; "
        f"os.write(1,{stdout!r}); "
        f"os.write(2,{stderr!r}); "
        "time.sleep(60)"
    )

    with pytest.raises(RuntimeError) as captured:
        qualification_sentinels._run_bounded_worker_process(
            [sys.executable, "-c", code],
            job_id="timeout",
            timeout_seconds=0.1,
            environment=os.environ,
            cwd=tmp_path,
            stdout_tail_max_bytes=32,
            stderr_tail_max_bytes=32,
            drain_timeout_seconds=0.2,
            termination_grace_seconds=0.2,
        )

    message = str(captured.value)
    assert message.startswith("GPU sentinel 'timeout' worker timed out after 0.1 seconds")
    assert f"bytes={len(stdout)},sha256={hashlib.sha256(stdout).hexdigest()}" in message
    assert f"bytes={len(stderr)},sha256={hashlib.sha256(stderr).hexdigest()}" in message


def test_worker_negative_returncode_reports_signal(tmp_path: Path):
    with pytest.raises(RuntimeError) as captured:
        qualification_sentinels._run_bounded_worker_process(
            [
                sys.executable,
                "-c",
                "import os,signal; os.kill(os.getpid(), signal.SIGTERM)",
            ],
            job_id="signal",
            timeout_seconds=5,
            environment=os.environ,
            cwd=tmp_path,
            drain_timeout_seconds=0.5,
            termination_grace_seconds=0.5,
        )

    assert str(captured.value).startswith(
        "GPU sentinel 'signal' worker terminated by signal "
        f"{signal.SIGTERM} (SIGTERM);"
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX process groups")
def test_worker_normal_exit_cleans_descendant_held_pipes(tmp_path: Path):
    child_pid_path = tmp_path / "held-pipe-child.pid"
    code = f"""
import os
from pathlib import Path
import time

if os.fork() == 0:
    Path({str(child_pid_path)!r}).write_text(str(os.getpid()))
    time.sleep(60)
    os._exit(0)
while not Path({str(child_pid_path)!r}).exists():
    time.sleep(0.001)
os.write(1, b'parent-finished')
"""
    started = time.monotonic()
    try:
        result = qualification_sentinels._run_bounded_worker_process(
            [sys.executable, "-c", code],
            job_id="descendant-pipes",
            timeout_seconds=5,
            environment=os.environ,
            cwd=tmp_path,
            drain_timeout_seconds=0.1,
            termination_grace_seconds=0.1,
        )

        child_pid = int(child_pid_path.read_text())
        assert result.returncode == 0
        assert result.stdout.tail == b"parent-finished"
        assert time.monotonic() - started < 2
        assert _wait_for_process_exit(child_pid, timeout_seconds=1)
    finally:
        if child_pid_path.exists():
            _kill_test_process(int(child_pid_path.read_text()))


def _process_is_alive(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_exit(process_id: int, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _process_is_alive(process_id):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def _kill_test_process(process_id: int) -> None:
    if _process_is_alive(process_id):
        os.kill(process_id, signal.SIGKILL)
        _wait_for_process_exit(process_id, timeout_seconds=1)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX process groups")
def test_worker_normal_exit_settles_owned_child_after_it_closes_pipes(
    tmp_path: Path,
):
    child_pid_path = tmp_path / "closed-pipe-child.pid"
    code = f"""
import os
from pathlib import Path
import time

if os.fork() == 0:
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 1)
    os.dup2(null, 2)
    if null > 2:
        os.close(null)
    Path({str(child_pid_path)!r}).write_text(str(os.getpid()))
    time.sleep(60)
    os._exit(0)
while not Path({str(child_pid_path)!r}).exists():
    time.sleep(0.001)
"""
    started = time.monotonic()
    try:
        result = qualification_sentinels._run_bounded_worker_process(
            [sys.executable, "-c", code],
            job_id="closed-pipe-descendant",
            timeout_seconds=5,
            environment=os.environ,
            cwd=tmp_path,
            drain_timeout_seconds=0.1,
            termination_grace_seconds=0.1,
        )

        child_pid = int(child_pid_path.read_text())
        assert result.returncode == 0
        assert time.monotonic() - started < 2
        assert _wait_for_process_exit(child_pid, timeout_seconds=1)
    finally:
        if child_pid_path.exists():
            _kill_test_process(int(child_pid_path.read_text()))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX process groups")
def test_worker_escaped_child_retaining_pipes_cannot_hold_controller(
    tmp_path: Path,
):
    child_pid_path = tmp_path / "escaped-pipe-child.pid"
    code = f"""
import os
from pathlib import Path
import time

if os.fork() == 0:
    os.setsid()
    Path({str(child_pid_path)!r}).write_text(str(os.getpid()))
    time.sleep(60)
    os._exit(0)
while not Path({str(child_pid_path)!r}).exists():
    time.sleep(0.001)
os.write(1, b'leader-finished')
"""
    started = time.monotonic()
    try:
        result = qualification_sentinels._run_bounded_worker_process(
            [sys.executable, "-c", code],
            job_id="escaped-pipe-descendant",
            timeout_seconds=5,
            environment=os.environ,
            cwd=tmp_path,
            drain_timeout_seconds=0.1,
            termination_grace_seconds=0.1,
        )

        child_pid = int(child_pid_path.read_text())
        assert result.returncode == 0
        assert result.stdout.tail == b"leader-finished"
        assert time.monotonic() - started < 2
        assert _process_is_alive(child_pid)
    finally:
        if child_pid_path.exists():
            _kill_test_process(int(child_pid_path.read_text()))


def test_worker_drainer_failure_is_not_silenced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def reject_stream(stream, accumulator, errors, error_lock, stop_event):
        with error_lock:
            errors.append(OSError("reviewed drain failure"))
        stream.close()

    monkeypatch.setattr(
        qualification_sentinels,
        "_drain_worker_stream",
        reject_stream,
    )
    with pytest.raises(RuntimeError, match="pipe drain failed") as captured:
        qualification_sentinels._run_bounded_worker_process(
            [sys.executable, "-c", "pass"],
            job_id="drain-failure",
            timeout_seconds=5,
            environment=os.environ,
            cwd=tmp_path,
            drain_timeout_seconds=0.2,
            termination_grace_seconds=0.2,
        )
    assert isinstance(captured.value.__cause__, OSError)


def test_worker_base_exception_cleans_owned_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    real_popen = subprocess.Popen
    launched = []

    def interrupting_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        launched.append(process)
        real_wait = process.wait
        wait_calls = 0

        def interrupt_once(timeout=None):
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                raise KeyboardInterrupt
            return real_wait(timeout=timeout)

        process.wait = interrupt_once
        return process

    monkeypatch.setattr(
        qualification_sentinels.subprocess,
        "Popen",
        interrupting_popen,
    )
    with pytest.raises(KeyboardInterrupt):
        qualification_sentinels._run_bounded_worker_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            job_id="interrupt",
            timeout_seconds=5,
            environment=os.environ,
            cwd=tmp_path,
            drain_timeout_seconds=0.2,
            termination_grace_seconds=0.2,
        )

    assert len(launched) == 1
    assert launched[0].poll() is not None


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("member", "path does not end with its owned member"),
        ("resolved", "non-symlink path and resolved_path differ"),
        ("stdout-digest", "ldd_stdout digest differs"),
        ("stdout-bytes", "ldd_stdout byte count differs"),
        ("bindings", "soname_bindings differ from ldd_stdout"),
        ("double-slash", "canonical absolute path"),
        ("unowned-symlink", "symlink target is not owned"),
        ("second-root", "distribution has multiple owned roots"),
    ),
)
def test_runtime_native_evidence_rejects_resealed_internal_contradictions(
    mutation: str,
    message: str,
) -> None:
    plan, evidence = _valid_evidence()
    runtime_job = next(
        job
        for job in evidence["cloud_gpu_evidence"]["jobs"]
        if job["job_id"].endswith("forced-triton-runtime-handoff")
    )
    record = _native_record(
        runtime_job["measurements"],
        "bitsandbytes/libbitsandbytes_cuda129.so",
    )
    if mutation == "member":
        record["member"] = "bitsandbytes/libdifferent.so"
    elif mutation == "resolved":
        record["resolved_path"] = f"{record['path']}.different"
    elif mutation == "stdout-digest":
        record["ldd_stdout_sha256"] = "f" * 64
    elif mutation == "stdout-bytes":
        record["ldd_stdout_utf8_bytes"] += 1
    elif mutation == "bindings":
        for escaping_reported_path in (
            "/../libc.so.6",
            "/lib/../../libc.so.6",
        ):
            escaped_evidence = deepcopy(evidence)
            escaped_measurements = _forced_runtime_measurements(escaped_evidence)
            escaped_record = escaped_measurements[
                "native_shared_object_evidence"
            ][0]
            _set_native_record_soname_bindings(
                escaped_record, {"libc.so.6": escaping_reported_path}
            )
            _reseal_evidence(escaped_evidence)
            with pytest.raises(ValueError, match="ldd-reported absolute path"):
                validate_gpu_qualification_evidence_record(
                    escaped_evidence,
                    plan_record=plan,
                    expected_campaign_id=CAMPAIGN_ID,
                    expected_artifact_pins=PINS,
                )
        reported_path = (
            "/runtime/lib/python3.11/site-packages/"
            "torch/lib/../../nvidia/cuda_runtime/lib/libcudart.so.12"
        )
        _set_native_record_soname_bindings(
            record, {"libcudart.so.12": reported_path}
        )
        record["soname_bindings"][0]["resolved_path"] = (
            "/runtime/lib/python3.11/site-packages/"
            "nvidia/cuda_runtime/lib/libcudart.so.12"
        )
    elif mutation == "double-slash":
        record["path"] = f"/{record['path']}"
    elif mutation == "unowned-symlink":
        record = next(
            item
            for item in runtime_job["measurements"][
                "native_shared_object_evidence"
            ]
            if item["distribution"] == "torch"
        )
        record["is_symlink"] = True
        record["resolved_path"] = f"{record['path']}.unowned"
    else:
        second_record = deepcopy(record)
        second_record["member"] = "bitsandbytes/libzz.so"
        second_record["path"] = "/other/site-packages/bitsandbytes/libzz.so"
        second_record["resolved_path"] = second_record["path"]
        runtime_job["measurements"]["native_shared_object_evidence"].insert(
            1, second_record
        )
        runtime_job["measurements"]["native_shared_object_count"] += 1
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match=message):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


def _forced_runtime_measurements(evidence: dict[str, Any]) -> dict[str, Any]:
    return next(
        job["measurements"]
        for job in evidence["cloud_gpu_evidence"]["jobs"]
        if job["job_id"].endswith("forced-triton-runtime-handoff")
    )


def _native_record(measurements: dict[str, Any], member: str) -> dict[str, Any]:
    return next(
        record
        for record in measurements["native_shared_object_evidence"]
        if record["member"] == member
    )


def test_runtime_native_evidence_accepts_exact_platform_inapplicable_closure() -> None:
    plan, evidence = _valid_evidence()
    measurements = _forced_runtime_measurements(evidence)
    records = measurements["native_shared_object_evidence"]

    assert len(records) == 16
    assert sum(
        any(binding["resolved_path"] is None for binding in record["soname_bindings"])
        for record in records
    ) == 12
    assert sum(
        binding["resolved_path"] is None
        for record in records
        for binding in record["soname_bindings"]
    ) == 34
    assert measurements["unresolved_native_shared_object_count"] == 12
    assert (
        measurements["unresolved_runtime_reachable_native_shared_object_count"] == 0
    )

    reported_path = (
        "/runtime/lib/python3.11/site-packages/"
        "torch/lib/../../nvidia/cuda_runtime/lib/libcudart.so.12"
    )
    torch_record = _native_record(measurements, "torch/lib/libtorch.so.2")
    _set_native_record_soname_bindings(
        torch_record, {"libcudart.so.12": reported_path}
    )
    _reseal_evidence(evidence)
    assert torch_record["soname_bindings"] == [
        {"resolved_path": reported_path, "soname": "libcudart.so.12"}
    ]

    validate_gpu_qualification_evidence_record(
        evidence,
        plan_record=plan,
        expected_campaign_id=CAMPAIGN_ID,
        expected_artifact_pins=PINS,
    )


def test_runtime_native_evidence_requires_complete_platform_member_closure() -> None:
    plan, evidence = _valid_evidence()
    measurements = _forced_runtime_measurements(evidence)
    records = measurements["native_shared_object_evidence"]
    removed = _native_record(
        measurements, "triton/plugins/libTritonPluginsTestLib.so"
    )
    records.remove(removed)
    measurements["native_shared_object_count"] -= 1
    measurements["unresolved_native_shared_object_count"] -= 1
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match="platform-inapplicable member closure"):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


@pytest.mark.parametrize(
    ("resolved_sonames", "expected_unresolved_count"),
    (
        (("libhipblas.so.3",), 12),
        (
            (
                "libhipblas.so.3",
                "libhipblaslt.so.1",
                "libhipsparse.so.4",
            ),
            11,
        ),
    ),
)
def test_runtime_native_evidence_accepts_missing_subset_or_empty(
    resolved_sonames: tuple[str, ...], expected_unresolved_count: int
) -> None:
    plan, evidence = _valid_evidence()
    measurements = _forced_runtime_measurements(evidence)
    record = _native_record(
        measurements, "bitsandbytes/libbitsandbytes_rocm70.so"
    )
    bindings = {
        binding["soname"]: binding["resolved_path"]
        for binding in record["soname_bindings"]
    }
    for soname in resolved_sonames:
        bindings[soname] = f"/opt/rocm/lib/{soname}"
    _set_native_record_soname_bindings(record, bindings)
    measurements["unresolved_native_shared_object_count"] = (
        expected_unresolved_count
    )
    _reseal_evidence(evidence)

    validate_gpu_qualification_evidence_record(
        evidence,
        plan_record=plan,
        expected_campaign_id=CAMPAIGN_ID,
        expected_artifact_pins=PINS,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("wrong-member", "non-permitted unresolved binding"),
        ("extra-missing", "non-permitted unresolved binding"),
        ("required-missing", "non-permitted unresolved binding"),
        ("platform-scope", "resolution_scope differs from member policy"),
        ("runtime-scope", "resolution_scope differs from member policy"),
        ("total-count", "unresolved_native_shared_object_count differs"),
        (
            "reachable-count",
            "unresolved_runtime_reachable_native_shared_object_count",
        ),
    ),
)
def test_runtime_native_evidence_rejects_member_policy_contradictions(
    mutation: str, message: str
) -> None:
    plan, evidence = _valid_evidence()
    measurements = _forced_runtime_measurements(evidence)
    dormant_record = _native_record(
        measurements, "bitsandbytes/libbitsandbytes_rocm70.so"
    )
    selected_record = _native_record(
        measurements, "bitsandbytes/libbitsandbytes_cuda129.so"
    )
    if mutation == "wrong-member":
        dormant_record["member"] = "bitsandbytes/libbitsandbytes_rocm69.so"
        dormant_record["path"] = dormant_record["path"].replace("rocm70", "rocm69")
        dormant_record["resolved_path"] = dormant_record["path"]
        dormant_record["resolution_scope"] = "runtime_reachable"
    elif mutation == "extra-missing":
        bindings = {
            binding["soname"]: binding["resolved_path"]
            for binding in dormant_record["soname_bindings"]
        }
        bindings["libunexpected.so"] = None
        _set_native_record_soname_bindings(dormant_record, bindings)
    elif mutation == "required-missing":
        _set_native_record_soname_bindings(selected_record, {"libc.so.6": None})
    elif mutation == "platform-scope":
        dormant_record["resolution_scope"] = "runtime_reachable"
    elif mutation == "runtime-scope":
        selected_record["resolution_scope"] = "platform_inapplicable"
    elif mutation == "total-count":
        measurements["unresolved_native_shared_object_count"] = 11
    else:
        measurements["unresolved_runtime_reachable_native_shared_object_count"] = 1
    measurements["native_shared_object_evidence"].sort(
        key=lambda record: (
            record["distribution"],
            record["member"],
            record["path"],
        )
    )
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match=message):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("attestation", "loaded bitsandbytes native library member mismatch"),
        ("evidence", "lacks the selected bitsandbytes member"),
    ),
)
def test_runtime_native_evidence_requires_selected_bitsandbytes_member(
    mutation: str, message: str
) -> None:
    plan, evidence = _valid_evidence()
    measurements = _forced_runtime_measurements(evidence)
    if mutation == "attestation":
        measurements["weight_quantizer_attestation"][
            "loaded_native_library_member"
        ] = "bitsandbytes/libbitsandbytes_cuda128.so"
    else:
        selected_record = _native_record(
            measurements, "bitsandbytes/libbitsandbytes_cuda129.so"
        )
        selected_record["member"] = "bitsandbytes/libbitsandbytes_cuda128.so"
        selected_record["path"] = selected_record["path"].replace(
            "cuda129", "cuda128"
        )
        selected_record["resolved_path"] = selected_record["path"]
        measurements["native_shared_object_evidence"].sort(
            key=lambda record: (
                record["distribution"],
                record["member"],
                record["path"],
            )
        )
    _reseal_evidence(evidence)

    with pytest.raises(ValueError, match=message):
        validate_gpu_qualification_evidence_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=PINS,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("symlink", "selected bitsandbytes member must not be a symlink"),
        ("attested-path", "path differs from native evidence"),
    ),
)
def test_runtime_native_evidence_binds_selected_bitsandbytes_regular_path(
    mutation: str, message: str
) -> None:
    plan, evidence = _valid_evidence()
    measurements = _forced_runtime_measurements(evidence)
    selected_record = _native_record(
        measurements, "bitsandbytes/libbitsandbytes_cuda129.so"
    )
    if mutation == "symlink":
        selected_record["is_symlink"] = True
        selected_record["resolved_path"] = _native_record(
            measurements, "bitsandbytes/libbitsandbytes_rocm70.so"
        )["path"]
    else:
        measurements["weight_quantizer_attestation"][
            "loaded_native_library_path"
        ] = (
            "/alternate/runtime/lib/python3.11/site-packages/"
            "bitsandbytes/libbitsandbytes_cuda129.so"
        )
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
