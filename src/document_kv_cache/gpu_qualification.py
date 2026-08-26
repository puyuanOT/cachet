"""Closed GPU qualification protocol for the vLLM 0.27.1 publication reset.

This module is deliberately independent from launch and serving code.  It can
describe the GPU work that must run and validate the resulting evidence, but it
cannot submit a cloud job or mutate a runtime.  Publication evidence therefore
has two visibly separate parts: local preflight evidence and first-attempt cloud
GPU evidence.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast
from urllib.parse import unquote, urlsplit

from document_kv_cache.databricks_resource_ledger import (
    DatabricksLedgerPrefix,
    databricks_ledger_prefix_from_record,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_ID,
    PUBLICATION_CAMPAIGN_LEDGER_ID,
    PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS,
    build_publication_campaign_plan,
    publication_campaign_plan_to_record,
)
from document_kv_cache.serving_env import (
    VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_LOCK_SHA256,
)


GPU_QUALIFICATION_PLAN_RECORD_TYPE: Final = "cachet.vllm_0271_gpu_qualification_plan.v1"
GPU_QUALIFICATION_EVIDENCE_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_qualification_evidence.v1"
)
GPU_QUALIFICATION_LOCAL_PREFLIGHT_RECORD_TYPE: Final = (
    "cachet.vllm_0271_local_preflight_evidence.v1"
)
GPU_QUALIFICATION_CLOUD_EVIDENCE_RECORD_TYPE: Final = (
    "cachet.vllm_0271_cloud_gpu_evidence.v1"
)
GPU_QUALIFICATION_JOB_RESULT_RECORD_TYPE: Final = "cachet.vllm_0271_gpu_job_result.v1"
GPU_QUALIFICATION_TERMINAL_RECEIPT_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_terminal_receipt.v1"
)
GPU_QUALIFICATION_SCHEMA_VERSION: Final = 1

GPU_QUALIFICATION_VLLM_VERSION: Final = "0.27.1+cu129"
GPU_QUALIFICATION_OFFICIAL_WHEEL_SHA256: Final = (
    "bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b"
)
GPU_QUALIFICATION_PATCHED_WHEEL_SHA256: Final = (
    "65120c48a9352b9eb65bab7a67090558d27af985ad366e469d3b87751073cff4"
)
GPU_QUALIFICATION_CONNECTOR_SOURCE_SHA256: Final = (
    "ec3e461a802fdbc0d324d8cb200a161042da0614cfc0cb342accdd9d58dca31d"
)
GPU_QUALIFICATION_MODEL_ID: Final = "Qwen/Qwen3-4B-Instruct-2507"
GPU_QUALIFICATION_MODEL_REVISION: Final = "cdbee75f17c01a7cc42f958dc650907174af0554"
GPU_QUALIFICATION_PUBLICATION_BACKEND: Final = "TRITON_ATTN"
GPU_QUALIFICATION_MAX_CLOUD_JOBS: Final = 16
GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND: Final = 35.0
GPU_QUALIFICATION_INPUT_CONTEXT_TOKENS: Final = 32_768
GPU_QUALIFICATION_DECODE_HEADROOM_TOKENS: Final = 512
GPU_QUALIFICATION_MAX_MODEL_LEN: Final = (
    GPU_QUALIFICATION_INPUT_CONTEXT_TOKENS + GPU_QUALIFICATION_DECODE_HEADROOM_TOKENS
)
GPU_QUALIFICATION_REQUEST_PARALLELISM: Final = 4
GPU_QUALIFICATION_REQUIRED_KV_CAPACITY_TOKENS: Final = (
    GPU_QUALIFICATION_REQUEST_PARALLELISM * GPU_QUALIFICATION_MAX_MODEL_LEN
)
GPU_QUALIFICATION_A10G_INPUT_CONTEXT_TOKENS: Final = 16_384
GPU_QUALIFICATION_A10G_MAX_MODEL_LEN: Final = (
    GPU_QUALIFICATION_A10G_INPUT_CONTEXT_TOKENS
    + GPU_QUALIFICATION_DECODE_HEADROOM_TOKENS
)
GPU_QUALIFICATION_A10G_REQUIRED_KV_CAPACITY_TOKENS: Final = (
    GPU_QUALIFICATION_REQUEST_PARALLELISM * GPU_QUALIFICATION_A10G_MAX_MODEL_LEN
)
GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS: Final = 256
GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES: Final = 2 * 1024**3
GPU_QUALIFICATION_GMU_SWEEP: Final = (0.70, 0.75, 0.80)
GPU_QUALIFICATION_THROUGHPUT_BUCKETS: Final = (8_192, 16_384, 32_768)
GPU_QUALIFICATION_THROUGHPUT_SAMPLES_PER_BUCKET: Final = 4
GPU_QUALIFICATION_MATCHED_EXAMPLES: Final = 4
GPU_QUALIFICATION_DETERMINISM_REPEATS: Final = 2
GPU_QUALIFICATION_MAX_LOGIT_DRIFT: Final = 1e-4
GPU_QUALIFICATION_MAX_E5M2_BF16_ERROR: Final = 0.125
GPU_QUALIFICATION_MODEL_LAYER_COUNT: Final = 36
GPU_QUALIFICATION_GENERATION_HARDWARE_ID: Final = "aws-g6e-l40s"
GPU_QUALIFICATION_GENERATION_GPU: Final = "NVIDIA L40S"
GPU_QUALIFICATION_GENERATION_COMPUTE_CAPABILITY: Final = "8.9"
GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE: Final = "g6e.4xlarge"
GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256: Final = (
    "cf2c10fcaf5b6e997a8d5f80712af1f251a7292b5d76813e28b39bc03fa7c629"
)
GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256: Final = (
    "7ff6cf6a1553c0e844853d21de9780c75211f1be8304754da72e9cbebbd164ec"
)
GPU_QUALIFICATION_INPUT_DATASETS: Final = (
    "biography",
    "hotpotqa",
    "musique",
    "niah",
)
GPU_QUALIFICATION_INPUT_ROWS_PER_DATASET_BUCKET: Final = 32

_RUNTIME_CORE_VERSIONS: Final = {
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
_PATCH_MEMBER_SHA256: Final = {
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
_FORCED_TRITON_KERNEL_NAMES: Final = (
    "triton_reshape_and_cache_flash",
    "triton_unified_attention",
)
_NATIVE_SHARED_OBJECT_DISTRIBUTIONS: Final = frozenset(
    {"bitsandbytes", "torch", "triton", "vllm"}
)
_NATIVE_SHARED_OBJECT_NAME_RE: Final = re.compile(r".+\.so(?:\..+)?\Z")
_NATIVE_LDD_MAX_STREAM_BYTES: Final = 256 * 1024

_SHA256_HEX_LENGTH = 64
_HARDWARE = (
    ("aws-g6-l4", "NVIDIA L4", "8.9"),
    ("aws-g5-a10g", "NVIDIA A10G", "8.6"),
)
_QUALIFICATION_HARDWARE = (
    *_HARDWARE,
    (
        GPU_QUALIFICATION_GENERATION_HARDWARE_ID,
        GPU_QUALIFICATION_GENERATION_GPU,
        GPU_QUALIFICATION_GENERATION_COMPUTE_CAPABILITY,
    ),
)
_LOCAL_CHECK_IDS = (
    "canonical_plan_schema",
    "runtime_lock_require_hashes",
    "patched_wheel_record_and_manifest",
    "source_runner_input_closure",
    "unit_tests",
    "ruff",
    "mypy",
)
_UNSUPPORTED_METHODS = (
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
)

_PLAN_KEYS = frozenset(
    {
        "campaign_id",
        "campaign_ledger_id",
        "campaign_ledger_path_sha256",
        "campaign_ledger_prefix",
        "campaign_opening_terminal_gpu_hours",
        "campaign_record_sha256",
        "closed_record_sha256",
        "cloud_qualification",
        "local_preflight",
        "record_type",
        "runtime_contract",
        "schema_version",
        "unsupported_methods",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "campaign_id",
        "closed_record_sha256",
        "cloud_gpu_evidence",
        "local_preflight_evidence",
        "plan_sha256",
        "qualification_status",
        "record_type",
        "schema_version",
    }
)
_LOCAL_EVIDENCE_KEYS = frozenset(
    {
        "checks",
        "closed_record_sha256",
        "completed_at_utc",
        "plan_sha256",
        "record_type",
        "schema_version",
        "scope",
    }
)
_LOCAL_CHECK_KEYS = frozenset({"check_id", "evidence_sha256", "status"})
_CLOUD_EVIDENCE_KEYS = frozenset(
    {
        "all_planned_jobs_succeeded",
        "authorization_source",
        "auto_backend_diagnostics_only",
        "closed_record_sha256",
        "job_count",
        "jobs",
        "max_parallel_jobs_observed",
        "plan_sha256",
        "publication_attention_backend",
        "record_type",
        "schema_version",
        "scope",
        "selected_gpu_memory_utilization",
        "terminal_receipts",
    }
)
_JOB_RESULT_KEYS = frozenset(
    {
        "authorization_scope",
        "artifact_sha256",
        "attempt_number",
        "closed_record_sha256",
        "cloud_cluster_id",
        "cloud_run_id",
        "finished_at_utc",
        "gpu",
        "gpu_compute_capability",
        "hardware_id",
        "job_id",
        "max_retries",
        "measurements",
        "nvidia_driver_version",
        "record_type",
        "output_json",
        "plan_sha256",
        "reservation_attempt_id",
        "retry_count",
        "schema_version",
        "started_at_utc",
        "status",
        "task_key",
        "torch_cuda_version",
        "vllm_version",
    }
)
_TERMINAL_RECEIPT_KEYS = frozenset(
    {
        "authorization_source",
        "closed_record_sha256",
        "cloud_cluster_id",
        "cloud_run_id",
        "collected_at_utc",
        "control_plane_status_sha256",
        "driver_node_type_id",
        "end_time_ms",
        "job_id",
        "ledger_actual_cluster_duration_seconds",
        "ledger_id",
        "ledger_terminal_actual_sha256",
        "life_cycle_state",
        "node_type_id",
        "output_json",
        "phase_batch_record_sha256",
        "phase_terminal_prefix",
        "plan_sha256",
        "record_type",
        "reservation_attempt_id",
        "result_file_sha256",
        "result_record_sha256",
        "result_state",
        "run_name",
        "schema_version",
        "start_time_ms",
        "submit_payload_sha256",
        "task_attempt_number",
        "task_end_time_ms",
        "task_key",
        "task_life_cycle_state",
        "task_max_retries",
        "task_result_state",
        "task_run_id",
        "task_start_time_ms",
    }
)


@dataclass(frozen=True, slots=True)
class GPUQualificationArtifactPins:
    """Exact immutable inputs shared by every qualification job."""

    runtime_lock_sha256: str
    patched_vllm_wheel_sha256: str
    package_wheel_sha256: str
    cachet_source_tree_sha256: str
    runner_sha256: str
    input_bundle_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_lock_sha256",
            "patched_vllm_wheel_sha256",
            "package_wheel_sha256",
            "cachet_source_tree_sha256",
            "runner_sha256",
            "input_bundle_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name=field_name)
        if self.patched_vllm_wheel_sha256 == (GPU_QUALIFICATION_OFFICIAL_WHEEL_SHA256):
            raise ValueError(
                "patched_vllm_wheel_sha256 must identify the repacked wheel"
            )

    def to_record(self) -> dict[str, str]:
        """Return the closed artifact-pin mapping."""

        return {
            "cachet_source_tree_sha256": self.cachet_source_tree_sha256,
            "input_bundle_sha256": self.input_bundle_sha256,
            "package_wheel_sha256": self.package_wheel_sha256,
            "patched_vllm_wheel_sha256": self.patched_vllm_wheel_sha256,
            "runner_sha256": self.runner_sha256,
            "runtime_lock_sha256": self.runtime_lock_sha256,
        }


@dataclass(frozen=True, slots=True)
class GPUQualificationSelection:
    """Publication runtime selected by validated qualification evidence."""

    attention_backend: str
    gpu_memory_utilization: float
    generation_hardware_id: str
    generation_databricks_node_type_id: str
    generation_artifacts_sha256: str
    generation_prefix_tokens_per_second: float
    plan_sha256: str


def build_gpu_qualification_plan(
    *,
    campaign_id: str,
    campaign_record_sha256: str,
    campaign_ledger_id: str,
    campaign_ledger_path_sha256: str,
    campaign_ledger_prefix: DatabricksLedgerPrefix,
    campaign_opening_terminal_gpu_hours: float,
    artifact_pins: GPUQualificationArtifactPins,
) -> dict[str, Any]:
    """Build the canonical, closed GPU qualification plan."""

    if campaign_id != PUBLICATION_CAMPAIGN_ID:
        raise ValueError("campaign_id differs from the frozen publication campaign")
    _require_sha256(
        campaign_record_sha256,
        field_name="campaign_record_sha256",
    )
    frozen_campaign = publication_campaign_plan_to_record(
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
    if (
        frozen_campaign.get("closed_record_sha256")
        != PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256
        or campaign_record_sha256 != PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256
    ):
        raise ValueError("campaign_record_sha256 differs from the frozen campaign")
    if campaign_ledger_id != PUBLICATION_CAMPAIGN_LEDGER_ID:
        raise ValueError("campaign_ledger_id differs from the frozen campaign")
    _require_sha256(
        campaign_ledger_path_sha256,
        field_name="campaign_ledger_path_sha256",
    )
    if not isinstance(campaign_ledger_prefix, DatabricksLedgerPrefix):
        raise TypeError("campaign_ledger_prefix must be a DatabricksLedgerPrefix")
    if (
        campaign_ledger_path_sha256 != PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256
        or campaign_ledger_prefix != PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX
    ):
        raise ValueError("campaign ledger path/prefix differs from the frozen campaign")
    if (
        isinstance(campaign_opening_terminal_gpu_hours, bool)
        or not isinstance(campaign_opening_terminal_gpu_hours, (int, float))
        or not math.isfinite(float(campaign_opening_terminal_gpu_hours))
        or float(campaign_opening_terminal_gpu_hours) < 0
    ):
        raise ValueError(
            "campaign_opening_terminal_gpu_hours must be finite/nonnegative"
        )
    if float(campaign_opening_terminal_gpu_hours) != (
        PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
    ):
        raise ValueError("campaign opening balance differs from the frozen campaign")
    if not isinstance(artifact_pins, GPUQualificationArtifactPins):
        raise TypeError("artifact_pins must be GPUQualificationArtifactPins")

    jobs = _qualification_jobs()
    if len(jobs) > GPU_QUALIFICATION_MAX_CLOUD_JOBS:
        raise RuntimeError("frozen qualification job matrix exceeds its hard cap")
    record: dict[str, Any] = {
        "campaign_id": campaign_id,
        "campaign_ledger_id": campaign_ledger_id,
        "campaign_ledger_path_sha256": campaign_ledger_path_sha256,
        "campaign_ledger_prefix": campaign_ledger_prefix.to_record(),
        "campaign_opening_terminal_gpu_hours": float(
            campaign_opening_terminal_gpu_hours
        ),
        "campaign_record_sha256": campaign_record_sha256,
        "closed_record_sha256": "",
        "cloud_qualification": {
            "all_jobs_required_to_finish_on_attempt_zero": True,
            "job_count": len(jobs),
            "jobs": jobs,
            "max_cloud_jobs": GPU_QUALIFICATION_MAX_CLOUD_JOBS,
            "max_retries": 0,
            "publication_launch_requires_passed_evidence": True,
            "success_attempt_number": 0,
        },
        "local_preflight": {
            "check_ids": list(_LOCAL_CHECK_IDS),
            "cloud_success_credit": False,
            "must_complete_before_cloud": True,
            "scope": "local_preflight_only",
        },
        "record_type": GPU_QUALIFICATION_PLAN_RECORD_TYPE,
        "runtime_contract": {
            "artifact_sha256": artifact_pins.to_record(),
            "attention_backend_auto_selection": "diagnostic_only",
            "compute_dtype": "bfloat16",
            "connector_source_sha256": (GPU_QUALIFICATION_CONNECTOR_SOURCE_SHA256),
            "installed_patch_member_sha256": dict(_PATCH_MEMBER_SHA256),
            "qualification_input_contract": {
                "bound_input_bundle_sha256": (artifact_pins.input_bundle_sha256),
                "publication_input_bundle_sha256": (
                    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256
                ),
                "datasets": list(GPU_QUALIFICATION_INPUT_DATASETS),
                "input_token_buckets": list(GPU_QUALIFICATION_THROUGHPUT_BUCKETS),
                "rows_per_dataset_bucket": (
                    GPU_QUALIFICATION_INPUT_ROWS_PER_DATASET_BUCKET
                ),
                "selected_rows_per_dataset_bucket": 1,
                "selection": "minimum_example_id_utf8_lexicographic",
            },
            "locked_core_distribution_versions": dict(_RUNTIME_CORE_VERSIONS),
            "handoff_kv_dtype": "fp8_e5m2",
            "handoff_kv_bits": 8,
            "model_id": GPU_QUALIFICATION_MODEL_ID,
            "model_revision": GPU_QUALIFICATION_MODEL_REVISION,
            "official_vllm_wheel_sha256": (GPU_QUALIFICATION_OFFICIAL_WHEEL_SHA256),
            "platform": {
                "glibc_version": "2.35",
                "python_version": "3.11.11",
                "system_cuda_version": "12.1",
                "torch_cuda_version": "12.9",
            },
            "publication_attention_backend": (GPU_QUALIFICATION_PUBLICATION_BACKEND),
            "publication_attention_backend_argument": [
                "--attention-backend",
                GPU_QUALIFICATION_PUBLICATION_BACKEND,
            ],
            "runtime_kv_dtype": "fp8_e5m2",
            "runtime_kv_bits": 8,
            "vllm_version": GPU_QUALIFICATION_VLLM_VERSION,
            "weight_bits": 4,
            "weight_quantization": "bitsandbytes",
            "trust_remote_code": False,
            "bitsandbytes_loader_sha256": (
                GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
            ),
            "weight_quantizer_contract": {
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_storage": "uint8",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "compress_statistics": True,
                "load_in_4bit": True,
            },
        },
        "schema_version": GPU_QUALIFICATION_SCHEMA_VERSION,
        "unsupported_methods": [dict(item) for item in _UNSUPPORTED_METHODS],
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def validate_gpu_qualification_plan_record(
    record: Mapping[str, Any],
    *,
    expected_campaign_id: str,
    expected_artifact_pins: GPUQualificationArtifactPins,
) -> None:
    """Fail closed unless *record* exactly matches the frozen expected plan."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    _require_exact_keys(record, _PLAN_KEYS, "GPU qualification plan")
    _require_closed_record_digest(record, "GPU qualification plan")
    campaign_record_sha256 = record.get("campaign_record_sha256")
    _require_sha256(
        campaign_record_sha256,
        field_name="campaign_record_sha256",
    )
    assert isinstance(campaign_record_sha256, str)
    campaign_ledger_id = record.get("campaign_ledger_id")
    if not isinstance(campaign_ledger_id, str) or not campaign_ledger_id:
        raise ValueError("campaign_ledger_id must be a non-empty string")
    campaign_ledger_path_sha256 = record.get("campaign_ledger_path_sha256")
    _require_sha256(
        campaign_ledger_path_sha256,
        field_name="campaign_ledger_path_sha256",
    )
    assert isinstance(campaign_ledger_path_sha256, str)
    campaign_ledger_prefix = databricks_ledger_prefix_from_record(
        _mapping(record.get("campaign_ledger_prefix"), "campaign_ledger_prefix")
    )
    campaign_opening_terminal_gpu_hours = record.get(
        "campaign_opening_terminal_gpu_hours"
    )
    expected = build_gpu_qualification_plan(
        campaign_id=expected_campaign_id,
        campaign_record_sha256=campaign_record_sha256,
        campaign_ledger_id=campaign_ledger_id,
        campaign_ledger_path_sha256=campaign_ledger_path_sha256,
        campaign_ledger_prefix=campaign_ledger_prefix,
        campaign_opening_terminal_gpu_hours=cast(
            float, campaign_opening_terminal_gpu_hours
        ),
        artifact_pins=expected_artifact_pins,
    )
    if canonical_gpu_qualification_json(record) != canonical_gpu_qualification_json(
        expected
    ):
        raise ValueError("GPU qualification plan does not match the frozen plan")


def build_local_preflight_evidence(
    *,
    plan_sha256: str,
    completed_at_utc: str,
    check_evidence_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build local-only evidence without granting cloud-success credit."""

    _require_sha256(plan_sha256, field_name="plan_sha256")
    _parse_utc_timestamp(completed_at_utc, field_name="completed_at_utc")
    if frozenset(check_evidence_sha256) != frozenset(_LOCAL_CHECK_IDS):
        raise ValueError(
            "local preflight check evidence must cover the frozen check IDs"
        )
    checks: list[dict[str, str]] = []
    for check_id in _LOCAL_CHECK_IDS:
        evidence_sha256 = check_evidence_sha256[check_id]
        _require_sha256(evidence_sha256, field_name=f"{check_id}.evidence_sha256")
        checks.append(
            {
                "check_id": check_id,
                "evidence_sha256": evidence_sha256,
                "status": "passed",
            }
        )
    record: dict[str, Any] = {
        "checks": checks,
        "closed_record_sha256": "",
        "completed_at_utc": completed_at_utc,
        "plan_sha256": plan_sha256,
        "record_type": GPU_QUALIFICATION_LOCAL_PREFLIGHT_RECORD_TYPE,
        "schema_version": GPU_QUALIFICATION_SCHEMA_VERSION,
        "scope": "local_preflight_only_no_cloud_success_credit",
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def validate_local_preflight_evidence_record(
    record: Mapping[str, Any],
    *,
    plan_sha256: str,
) -> datetime:
    """Validate the closed local preflight and return its completion time."""

    if not isinstance(record, Mapping):
        raise TypeError("local preflight evidence must be a mapping")
    _require_sha256(plan_sha256, field_name="plan_sha256")
    return _validate_local_preflight_evidence(record, plan_sha256)


def build_gpu_job_result(
    *,
    plan_record: Mapping[str, Any],
    job_id: str,
    reservation_attempt_id: str,
    task_key: str,
    output_json: str,
    cloud_run_id: str,
    cloud_cluster_id: str,
    started_at_utc: str,
    finished_at_utc: str,
    nvidia_driver_version: str,
    observed_gpu: str,
    observed_gpu_compute_capability: str,
    observed_vllm_version: str,
    observed_torch_cuda_version: str,
    observed_artifact_sha256: Mapping[str, str],
    measurements: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal measurement output that still requires a direct terminal receipt.

    This builder never grants campaign authority by itself.  The governed
    collector must bind the record to the reserved submit payload and a direct
    terminal ``jobs/runs/get`` response.
    """

    job = _plan_job(plan_record, job_id)
    started = _parse_utc_timestamp(started_at_utc, field_name="started_at_utc")
    finished = _parse_utc_timestamp(finished_at_utc, field_name="finished_at_utc")
    if finished <= started:
        raise ValueError("finished_at_utc must be after started_at_utc")
    for field_name, value in (
        ("reservation_attempt_id", reservation_attempt_id),
        ("task_key", task_key),
        ("output_json", output_json),
        ("cloud_run_id", cloud_run_id),
        ("cloud_cluster_id", cloud_cluster_id),
        ("nvidia_driver_version", nvidia_driver_version),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} must be a non-empty string")
    for field_name, value in (
        ("observed_gpu", observed_gpu),
        ("observed_gpu_compute_capability", observed_gpu_compute_capability),
        ("observed_vllm_version", observed_vllm_version),
        ("observed_torch_cuda_version", observed_torch_cuda_version),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} must be a non-empty string")
    artifact_sha256 = _mapping(observed_artifact_sha256, "observed_artifact_sha256")
    record: dict[str, Any] = {
        "authorization_scope": ("measurement_only_requires_direct_terminal_receipt"),
        "artifact_sha256": dict(artifact_sha256),
        "attempt_number": 0,
        "closed_record_sha256": "",
        "cloud_cluster_id": cloud_cluster_id,
        "cloud_run_id": cloud_run_id,
        "finished_at_utc": finished_at_utc,
        "gpu": observed_gpu,
        "gpu_compute_capability": observed_gpu_compute_capability,
        "hardware_id": job["hardware_id"],
        "job_id": job_id,
        "max_retries": 0,
        "measurements": dict(measurements),
        "nvidia_driver_version": nvidia_driver_version,
        "output_json": output_json,
        "plan_sha256": plan_record["closed_record_sha256"],
        "record_type": GPU_QUALIFICATION_JOB_RESULT_RECORD_TYPE,
        "reservation_attempt_id": reservation_attempt_id,
        "retry_count": 0,
        "schema_version": GPU_QUALIFICATION_SCHEMA_VERSION,
        "started_at_utc": started_at_utc,
        "status": "SUCCESS",
        "task_key": task_key,
        "torch_cuda_version": observed_torch_cuda_version,
        "vllm_version": observed_vllm_version,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def build_cloud_gpu_evidence(
    *,
    plan_sha256: str,
    jobs: Sequence[Mapping[str, Any]],
    selected_gpu_memory_utilization: float,
) -> dict[str, Any]:
    """Build a caller-supplied fixture that cannot authorize publication."""

    return _build_cloud_gpu_evidence_record(
        plan_sha256=plan_sha256,
        jobs=jobs,
        selected_gpu_memory_utilization=selected_gpu_memory_utilization,
        terminal_receipts=(),
        authorization_source="caller_supplied_nonauthorizing",
        scope="synthetic_fixture_no_campaign_authority",
    )


def _build_governed_cloud_gpu_evidence(
    *,
    plan_sha256: str,
    jobs: Sequence[Mapping[str, Any]],
    terminal_receipts: Sequence[Mapping[str, Any]],
    selected_gpu_memory_utilization: float,
) -> dict[str, Any]:
    """Build cloud evidence only after direct control-plane collection."""

    return _build_cloud_gpu_evidence_record(
        plan_sha256=plan_sha256,
        jobs=jobs,
        selected_gpu_memory_utilization=selected_gpu_memory_utilization,
        terminal_receipts=terminal_receipts,
        authorization_source="direct_databricks_runs_get",
        scope="governed_cloud_gpu_terminal_evidence",
    )


def _build_cloud_gpu_evidence_record(
    *,
    plan_sha256: str,
    jobs: Sequence[Mapping[str, Any]],
    selected_gpu_memory_utilization: float,
    terminal_receipts: Sequence[Mapping[str, Any]],
    authorization_source: str,
    scope: str,
) -> dict[str, Any]:
    """Seal cloud evidence with an explicit authorization provenance."""

    _require_sha256(plan_sha256, field_name="plan_sha256")
    job_records = [dict(job) for job in jobs]
    receipt_records = [dict(receipt) for receipt in terminal_receipts]
    record: dict[str, Any] = {
        "all_planned_jobs_succeeded": True,
        "authorization_source": authorization_source,
        "auto_backend_diagnostics_only": True,
        "closed_record_sha256": "",
        "job_count": len(job_records),
        "jobs": job_records,
        "max_parallel_jobs_observed": _max_parallel_jobs(job_records),
        "plan_sha256": plan_sha256,
        "publication_attention_backend": GPU_QUALIFICATION_PUBLICATION_BACKEND,
        "record_type": GPU_QUALIFICATION_CLOUD_EVIDENCE_RECORD_TYPE,
        "schema_version": GPU_QUALIFICATION_SCHEMA_VERSION,
        "scope": scope,
        "selected_gpu_memory_utilization": selected_gpu_memory_utilization,
        "terminal_receipts": receipt_records,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def validate_gpu_job_result_record(
    record: Mapping[str, Any],
    *,
    plan_record: Mapping[str, Any],
    expected_artifact_pins: GPUQualificationArtifactPins,
) -> None:
    """Validate one planned GPU result, including its sentinel measurements.

    This is the per-task counterpart to the aggregate cloud-evidence validator.
    A Databricks task uses it before publishing its result so a structurally
    valid but semantically incomplete measurement object cannot be mistaken for
    first-attempt qualification evidence.
    """

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    job_id = record.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("GPU job result job_id must be a non-empty string")
    planned_job = _plan_job(plan_record, job_id)
    _validate_job_result_common(
        record,
        planned_job=planned_job,
        expected_artifact_pins=expected_artifact_pins,
        expected_plan_sha256=str(plan_record.get("closed_record_sha256")),
    )
    measurements = _mapping(record.get("measurements"), f"measurements {job_id}")
    sentinel = planned_job["sentinel"]
    if sentinel == "forced_triton_runtime_handoff":
        _validate_runtime_handoff_measurements(
            measurements, hardware_id=planned_job["hardware_id"]
        )
    elif sentinel == "packed_page_raw_byte_roundtrip":
        _validate_packed_roundtrip_measurements(measurements)
    elif sentinel == "matched_token_contract_and_determinism":
        _validate_token_determinism_measurements(
            measurements,
            expected_input_bundle_sha256=(expected_artifact_pins.input_bundle_sha256),
        )
    elif sentinel == "l4_32k_c4_gmu_sweep":
        _validate_gmu_measurements(
            measurements,
            expected_gmu=planned_job["requirements"]["gpu_memory_utilization"],
        )
    elif sentinel == "a10g_16k_c4_capacity":
        _validate_a10g_capacity_measurements(measurements)
    elif sentinel == "generation_throughput_with_writes":
        _validate_throughput_measurements(
            measurements,
            hardware_id=planned_job["hardware_id"],
        )
    elif sentinel == "auto_backend_diagnostic":
        _validate_auto_backend_measurements(measurements)
    else:
        raise ValueError(f"unsupported frozen sentinel: {sentinel!r}")


def build_gpu_qualification_evidence(
    *,
    campaign_id: str,
    plan_sha256: str,
    local_preflight_evidence: Mapping[str, Any],
    cloud_gpu_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a record-only envelope that can never authorize a launch.

    Authority is deliberately not inferred from fields inside a caller-owned
    mapping.  The Databricks collector has a separate private constructor and
    returns a non-record launch capability after it has joined live control-
    plane responses to the append-only ledger closure.
    """

    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("campaign_id must be a non-empty string")
    _require_sha256(plan_sha256, field_name="plan_sha256")
    record: dict[str, Any] = {
        "campaign_id": campaign_id,
        "closed_record_sha256": "",
        "cloud_gpu_evidence": dict(cloud_gpu_evidence),
        "local_preflight_evidence": dict(local_preflight_evidence),
        "plan_sha256": plan_sha256,
        "qualification_status": "unverified",
        "record_type": GPU_QUALIFICATION_EVIDENCE_RECORD_TYPE,
        "schema_version": GPU_QUALIFICATION_SCHEMA_VERSION,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _build_governed_gpu_qualification_evidence(
    *,
    campaign_id: str,
    plan_sha256: str,
    local_preflight_evidence: Mapping[str, Any],
    cloud_gpu_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the collector-only persisted envelope.

    This constructor is not a launch authority.  The live collector or the
    durable replay boundary must additionally issue a
    ``GPUQualificationLaunchAuthorization`` capability.
    """

    record = build_gpu_qualification_evidence(
        campaign_id=campaign_id,
        plan_sha256=plan_sha256,
        local_preflight_evidence=local_preflight_evidence,
        cloud_gpu_evidence=cloud_gpu_evidence,
    )
    if (
        cloud_gpu_evidence.get("authorization_source") != "direct_databricks_runs_get"
        or cloud_gpu_evidence.get("scope") != "governed_cloud_gpu_terminal_evidence"
    ):
        raise ValueError("collector-governed cloud evidence is required")
    record["qualification_status"] = "passed"
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def validate_gpu_qualification_evidence_record(
    record: Mapping[str, Any],
    *,
    plan_record: Mapping[str, Any],
    expected_campaign_id: str,
    expected_artifact_pins: GPUQualificationArtifactPins,
) -> GPUQualificationSelection:
    """Validate complete local and cloud evidence and return the safe selection."""

    validate_gpu_qualification_plan_record(
        plan_record,
        expected_campaign_id=expected_campaign_id,
        expected_artifact_pins=expected_artifact_pins,
    )
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    _require_exact_keys(record, _EVIDENCE_KEYS, "GPU qualification evidence")
    _require_closed_record_digest(record, "GPU qualification evidence")
    plan_sha256 = plan_record["closed_record_sha256"]
    if record.get("record_type") != GPU_QUALIFICATION_EVIDENCE_RECORD_TYPE:
        raise ValueError("unexpected GPU qualification evidence record_type")
    if (
        type(record.get("schema_version")) is not int
        or record.get("schema_version") != GPU_QUALIFICATION_SCHEMA_VERSION
    ):
        raise ValueError("unexpected GPU qualification evidence schema_version")
    if record.get("campaign_id") != expected_campaign_id:
        raise ValueError("GPU qualification evidence campaign_id mismatch")
    if record.get("plan_sha256") != plan_sha256:
        raise ValueError("GPU qualification evidence plan_sha256 mismatch")
    if record.get("qualification_status") != "passed":
        raise ValueError("GPU qualification evidence must declare passed")

    local = _mapping(record.get("local_preflight_evidence"), "local_preflight_evidence")
    local_completed = _validate_local_preflight_evidence(local, plan_sha256)
    cloud = _mapping(record.get("cloud_gpu_evidence"), "cloud_gpu_evidence")
    selection, first_cloud_start = _validate_cloud_gpu_evidence(
        cloud,
        plan_record=plan_record,
        plan_sha256=plan_sha256,
        expected_artifact_pins=expected_artifact_pins,
    )
    if local_completed >= first_cloud_start:
        raise ValueError("local preflight must complete before cloud GPU execution")
    return selection


def canonical_gpu_qualification_json(record: Mapping[str, Any]) -> str:
    """Return RFC-8259-compatible canonical JSON used by closure digests."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    return json.dumps(
        dict(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def write_canonical_gpu_qualification_json(
    record: Mapping[str, Any], path: str | Path
) -> None:
    """Write canonical JSON once; existing evidence is never overwritten."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"GPU qualification record already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        canonical_gpu_qualification_json(record) + "\n",
        encoding="utf-8",
    )


def write_gpu_qualification_plan_json(
    record: Mapping[str, Any],
    path: str | Path,
    *,
    expected_campaign_id: str,
    expected_artifact_pins: GPUQualificationArtifactPins,
) -> None:
    """Validate and write one canonical frozen plan without overwriting it."""

    validate_gpu_qualification_plan_record(
        record,
        expected_campaign_id=expected_campaign_id,
        expected_artifact_pins=expected_artifact_pins,
    )
    write_canonical_gpu_qualification_json(record, path)


def write_gpu_qualification_evidence_json(
    record: Mapping[str, Any],
    path: str | Path,
    *,
    plan_record: Mapping[str, Any],
    expected_campaign_id: str,
    expected_artifact_pins: GPUQualificationArtifactPins,
) -> GPUQualificationSelection:
    """Validate, canonically write, and return a qualified runtime selection."""

    selection = validate_gpu_qualification_evidence_record(
        record,
        plan_record=plan_record,
        expected_campaign_id=expected_campaign_id,
        expected_artifact_pins=expected_artifact_pins,
    )
    write_canonical_gpu_qualification_json(record, path)
    return selection


def _qualification_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for hardware_id, gpu, compute_capability in _HARDWARE:
        jobs.append(
            _job(
                job_id=f"{hardware_id}-forced-triton-runtime-handoff",
                hardware_id=hardware_id,
                gpu=gpu,
                sentinel="forced_triton_runtime_handoff",
                evidence_class="publication_gate",
                backend_mode="forced_triton",
                requirements={
                    "compute_capability": compute_capability,
                    "q4_bitsandbytes_weights": True,
                    "bf16_compute": True,
                    "q8_e5m2_runtime_kv": True,
                    "q8_e5m2_handoff": True,
                    "query_remains_bf16": True,
                    "real_triton_compile_and_launch": True,
                    "triton_cache_miss_kernel_names": list(_FORCED_TRITON_KERNEL_NAMES),
                    "all_layer_handoff_count": (GPU_QUALIFICATION_MODEL_LAYER_COUNT),
                    "gpu_runtime_attestation": {
                        "direct_url_matches_patched_wheel": True,
                        "installed_lock_matches": True,
                        "native_shared_objects_resolved": True,
                        "pip_check": True,
                        "site_packages_read_only": True,
                    },
                    "a10g_software_e5m2_path_required": hardware_id == "aws-g5-a10g",
                },
            )
        )
        jobs.append(
            _job(
                job_id=f"{hardware_id}-packed-page-roundtrip",
                hardware_id=hardware_id,
                gpu=gpu,
                sentinel="packed_page_raw_byte_roundtrip",
                evidence_class="publication_gate",
                backend_mode="forced_triton",
                requirements={
                    "bf16_reference_max_abs_error": (
                        GPU_QUALIFICATION_MAX_E5M2_BF16_ERROR
                    ),
                    "bf16_reference_scope": "attention_output",
                    "cache_page_layout": "B_H_N_2D",
                    "cache_page_shape": ["B", "H", "N", "2D"],
                    "input_value_range": [-1.0, 1.0],
                    "payload_layouts": ["NHD", "HND"],
                    "negative_slots": True,
                    "partial_slots": True,
                    "noncontiguous_strides": True,
                    "query_dtype": "bfloat16",
                    "raw_byte_identity": True,
                },
            )
        )
        jobs.append(
            _job(
                job_id=f"{hardware_id}-matched-token-logit",
                hardware_id=hardware_id,
                gpu=gpu,
                sentinel="matched_token_contract_and_determinism",
                evidence_class="publication_gate",
                backend_mode="forced_triton",
                requirements={
                    "arms": ["baseline_prefill", "vanilla_prefill"],
                    "determinism_repeats": (GPU_QUALIFICATION_DETERMINISM_REPEATS),
                    "matched_examples": GPU_QUALIFICATION_MATCHED_EXAMPLES,
                    "max_abs_logit_drift": GPU_QUALIFICATION_MAX_LOGIT_DRIFT,
                    "quality_equivalence_required": False,
                    "execution_mode": "real_end_to_end_requests",
                    "request_parallelism": 1,
                    "cache_phases": ["cold", "warm"],
                    "sampling": {
                        "max_tokens": 16,
                        "seed": 17,
                        "temperature": 0.0,
                    },
                },
            )
        )

    for gpu_memory_utilization in GPU_QUALIFICATION_GMU_SWEEP:
        gmu_label = str(round(gpu_memory_utilization * 100))
        jobs.append(
            _job(
                job_id=f"aws-g6-l4-32k-c4-gmu-{gmu_label}",
                hardware_id="aws-g6-l4",
                gpu="NVIDIA L4",
                sentinel="l4_32k_c4_gmu_sweep",
                evidence_class="publication_gate",
                backend_mode="forced_triton",
                requirements={
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "max_model_len": GPU_QUALIFICATION_MAX_MODEL_LEN,
                    "input_tokens_per_request": (
                        GPU_QUALIFICATION_INPUT_CONTEXT_TOKENS
                    ),
                    "decode_headroom_tokens": (
                        GPU_QUALIFICATION_DECODE_HEADROOM_TOKENS
                    ),
                    "forced_decode_tokens": (GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS),
                    "minimum_kv_capacity_tokens": (
                        GPU_QUALIFICATION_REQUIRED_KV_CAPACITY_TOKENS
                    ),
                    "minimum_observed_peak_headroom_bytes": (
                        GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES
                    ),
                    "request_parallelism": (GPU_QUALIFICATION_REQUEST_PARALLELISM),
                    "selection": "highest_qualifying_candidate",
                    "cold_disk_q8_pre_rope_vanilla_handoffs": True,
                    "successful_connector_loads": 4,
                    "connector_layers_per_load": (GPU_QUALIFICATION_MODEL_LAYER_COUNT),
                    "zero_fatal_errors": True,
                    "zero_ooms": True,
                },
            )
        )

    jobs.append(
        _job(
            job_id="aws-g5-a10g-16k-c4-capacity",
            hardware_id="aws-g5-a10g",
            gpu="NVIDIA A10G",
            sentinel="a10g_16k_c4_capacity",
            evidence_class="publication_gate",
            backend_mode="forced_triton",
            requirements={
                "gpu_memory_utilization": 0.90,
                "max_model_len": GPU_QUALIFICATION_A10G_MAX_MODEL_LEN,
                "input_tokens_per_request": (
                    GPU_QUALIFICATION_A10G_INPUT_CONTEXT_TOKENS
                ),
                "decode_headroom_tokens": (GPU_QUALIFICATION_DECODE_HEADROOM_TOKENS),
                "forced_decode_tokens": GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS,
                "minimum_kv_capacity_tokens": (
                    GPU_QUALIFICATION_A10G_REQUIRED_KV_CAPACITY_TOKENS
                ),
                "minimum_observed_peak_headroom_bytes": (
                    GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES
                ),
                "request_parallelism": 4,
                "cold_disk_q8_pre_rope_vanilla_handoffs": True,
                "successful_connector_loads": 4,
                "connector_layers_per_load": GPU_QUALIFICATION_MODEL_LAYER_COUNT,
                "zero_fatal_errors": True,
                "zero_ooms": True,
            },
        )
    )

    jobs.append(
        _job(
            job_id="aws-g6-l4-generation-throughput",
            hardware_id="aws-g6-l4",
            gpu="NVIDIA L4",
            sentinel="generation_throughput_with_writes",
            evidence_class="publication_gate",
            backend_mode="forced_triton",
            requirements={
                "clock_scope": "prefix_generation_through_durable_kv_write",
                "length_buckets": list(GPU_QUALIFICATION_THROUGHPUT_BUCKETS),
                "minimum_prefix_tokens_per_second": (
                    GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND
                ),
                "samples_per_bucket": (GPU_QUALIFICATION_THROUGHPUT_SAMPLES_PER_BUCKET),
                "threshold_applies_to": "l40s_every_bucket_and_aggregate",
                "segmented_pre_rope_document_tokens": 2_048,
                "artifact_identity_peer_hardware_id": (
                    GPU_QUALIFICATION_GENERATION_HARDWARE_ID
                ),
                "generator_device_map": "auto",
                "writes_included": True,
            },
        )
    )
    jobs.append(
        _job(
            job_id="aws-g6e-l40s-generation-throughput",
            hardware_id=GPU_QUALIFICATION_GENERATION_HARDWARE_ID,
            gpu=GPU_QUALIFICATION_GENERATION_GPU,
            sentinel="generation_throughput_with_writes",
            evidence_class="publication_gate",
            backend_mode="forced_triton",
            requirements={
                "clock_scope": "prefix_generation_through_durable_kv_write",
                "length_buckets": list(GPU_QUALIFICATION_THROUGHPUT_BUCKETS),
                "minimum_prefix_tokens_per_second": (
                    GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND
                ),
                "samples_per_bucket": (GPU_QUALIFICATION_THROUGHPUT_SAMPLES_PER_BUCKET),
                "threshold_applies_to": "l40s_every_bucket_and_aggregate",
                "segmented_pre_rope_document_tokens": 2_048,
                "artifact_identity_peer_hardware_id": "aws-g6-l4",
                "generator_device_map": "auto",
                "writes_included": True,
            },
        )
    )

    for hardware_id, gpu, _compute_capability in _HARDWARE:
        jobs.append(
            _job(
                job_id=f"{hardware_id}-auto-backend-diagnostic",
                hardware_id=hardware_id,
                gpu=gpu,
                sentinel="auto_backend_diagnostic",
                evidence_class="diagnostic_only",
                backend_mode="auto",
                requirements={
                    "can_select_publication_backend": False,
                    "observed_backend_required": True,
                    "publication_backend_remains": (
                        GPU_QUALIFICATION_PUBLICATION_BACKEND
                    ),
                },
            )
        )
    return jobs


def _job(
    *,
    job_id: str,
    hardware_id: str,
    gpu: str,
    sentinel: str,
    evidence_class: str,
    backend_mode: str,
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "attempt_number": 0,
        "backend_mode": backend_mode,
        "compute_capability": _hardware_compute_capability(hardware_id),
        "evidence_class": evidence_class,
        "gpu": gpu,
        "hardware_id": hardware_id,
        "job_id": job_id,
        "max_retries": 0,
        "requirements": dict(requirements),
        "sentinel": sentinel,
    }


def _validate_local_preflight_evidence(
    record: Mapping[str, Any], plan_sha256: str
) -> datetime:
    _require_exact_keys(record, _LOCAL_EVIDENCE_KEYS, "local preflight evidence")
    _require_closed_record_digest(record, "local preflight evidence")
    if record.get("record_type") != GPU_QUALIFICATION_LOCAL_PREFLIGHT_RECORD_TYPE:
        raise ValueError("unexpected local preflight record_type")
    if (
        type(record.get("schema_version")) is not int
        or record.get("schema_version") != GPU_QUALIFICATION_SCHEMA_VERSION
    ):
        raise ValueError("unexpected local preflight schema_version")
    if record.get("scope") != "local_preflight_only_no_cloud_success_credit":
        raise ValueError("local preflight scope cannot grant cloud success credit")
    if record.get("plan_sha256") != plan_sha256:
        raise ValueError("local preflight plan_sha256 mismatch")
    checks = _sequence(record.get("checks"), "local preflight checks")
    if len(checks) != len(_LOCAL_CHECK_IDS):
        raise ValueError("local preflight is missing required checks")
    for expected_id, raw_check in zip(_LOCAL_CHECK_IDS, checks, strict=True):
        check = _mapping(raw_check, f"local check {expected_id}")
        _require_exact_keys(check, _LOCAL_CHECK_KEYS, f"local check {expected_id}")
        if check.get("check_id") != expected_id or check.get("status") != "passed":
            raise ValueError(
                f"local check {expected_id} did not pass in canonical order"
            )
        _require_sha256(
            check.get("evidence_sha256"),
            field_name=f"local check {expected_id}.evidence_sha256",
        )
    return _parse_utc_timestamp(
        record.get("completed_at_utc"), field_name="completed_at_utc"
    )


def _validate_cloud_gpu_evidence(
    record: Mapping[str, Any],
    *,
    plan_record: Mapping[str, Any],
    plan_sha256: str,
    expected_artifact_pins: GPUQualificationArtifactPins,
) -> tuple[GPUQualificationSelection, datetime]:
    _require_exact_keys(record, _CLOUD_EVIDENCE_KEYS, "cloud GPU evidence")
    _require_closed_record_digest(record, "cloud GPU evidence")
    if record.get("record_type") != GPU_QUALIFICATION_CLOUD_EVIDENCE_RECORD_TYPE:
        raise ValueError("unexpected cloud GPU evidence record_type")
    if (
        type(record.get("schema_version")) is not int
        or record.get("schema_version") != GPU_QUALIFICATION_SCHEMA_VERSION
    ):
        raise ValueError("unexpected cloud GPU evidence schema_version")
    if record.get("scope") != "governed_cloud_gpu_terminal_evidence":
        raise ValueError("cloud GPU evidence is not governed terminal evidence")
    if record.get("authorization_source") != "direct_databricks_runs_get":
        raise ValueError("cloud GPU evidence lacks direct control-plane authorization")
    if record.get("plan_sha256") != plan_sha256:
        raise ValueError("cloud GPU evidence plan_sha256 mismatch")
    if record.get("publication_attention_backend") != (
        GPU_QUALIFICATION_PUBLICATION_BACKEND
    ):
        raise ValueError("publication attention backend must remain explicit Triton")
    if record.get("auto_backend_diagnostics_only") is not True:
        raise ValueError("auto backend results must remain diagnostic only")
    if record.get("all_planned_jobs_succeeded") is not True:
        raise ValueError("all planned GPU jobs must succeed")

    plan_jobs = _plan_jobs(plan_record)
    jobs = _sequence(record.get("jobs"), "cloud GPU jobs")
    terminal_receipts = _sequence(
        record.get("terminal_receipts"), "cloud GPU terminal receipts"
    )
    if (
        type(record.get("job_count")) is not int
        or record.get("job_count") != len(plan_jobs)
        or len(jobs) != len(plan_jobs)
        or len(terminal_receipts) != len(plan_jobs)
    ):
        raise ValueError(
            "cloud GPU evidence must contain every planned job exactly once"
        )
    if len(jobs) > GPU_QUALIFICATION_MAX_CLOUD_JOBS:
        raise ValueError("cloud GPU job count exceeds the frozen cap")

    job_records: list[Mapping[str, Any]] = []
    run_ids: set[str] = set()
    cluster_ids: set[str] = set()
    task_run_ids: set[str] = set()
    reservation_attempt_ids: set[str] = set()
    task_keys: set[str] = set()
    output_paths: set[str] = set()
    ledger_ids: set[str] = set()
    submit_payload_sha256_values: set[str] = set()
    gmu_candidates: list[tuple[float, bool]] = []
    generation_results: dict[str, tuple[str, float, tuple[Mapping[str, Any], ...]]] = {}
    receipt_run_ids: set[str] = set()
    for planned_job, raw_result, raw_receipt in zip(
        plan_jobs, jobs, terminal_receipts, strict=True
    ):
        result = _mapping(raw_result, f"job result {planned_job['job_id']}")
        _validate_job_result_common(
            result,
            planned_job=planned_job,
            expected_artifact_pins=expected_artifact_pins,
            expected_plan_sha256=plan_sha256,
        )
        run_id = result["cloud_run_id"]
        if run_id in run_ids:
            raise ValueError("cloud_run_id values must be unique")
        run_ids.add(run_id)
        receipt = _mapping(raw_receipt, f"terminal receipt {planned_job['job_id']}")
        _validate_terminal_receipt(
            receipt,
            result=result,
            planned_job=planned_job,
            plan_record=plan_record,
        )
        receipt_run_id = str(receipt["cloud_run_id"])
        if receipt_run_id in receipt_run_ids:
            raise ValueError("terminal receipt cloud_run_id values must be unique")
        receipt_run_ids.add(receipt_run_id)
        _require_unique_identity(
            cluster_ids,
            str(receipt["cloud_cluster_id"]),
            "cloud_cluster_id",
        )
        _require_unique_identity(
            task_run_ids,
            str(receipt["task_run_id"]),
            "task_run_id",
        )
        _require_unique_identity(
            reservation_attempt_ids,
            str(receipt["reservation_attempt_id"]),
            "reservation_attempt_id",
        )
        _require_unique_identity(
            task_keys,
            str(receipt["task_key"]),
            "task_key",
        )
        _require_unique_identity(
            output_paths,
            str(receipt["output_json"]),
            "output_json",
        )
        ledger_ids.add(str(receipt["ledger_id"]))
        _require_unique_identity(
            submit_payload_sha256_values,
            str(receipt["submit_payload_sha256"]),
            "submit_payload_sha256",
        )
        measurements = _mapping(
            result.get("measurements"), f"measurements {planned_job['job_id']}"
        )
        sentinel = planned_job["sentinel"]
        if sentinel == "forced_triton_runtime_handoff":
            _validate_runtime_handoff_measurements(
                measurements, hardware_id=planned_job["hardware_id"]
            )
        elif sentinel == "packed_page_raw_byte_roundtrip":
            _validate_packed_roundtrip_measurements(measurements)
        elif sentinel == "matched_token_contract_and_determinism":
            _validate_token_determinism_measurements(
                measurements,
                expected_input_bundle_sha256=(
                    expected_artifact_pins.input_bundle_sha256
                ),
            )
        elif sentinel == "l4_32k_c4_gmu_sweep":
            gmu_candidates.append(
                _validate_gmu_measurements(
                    measurements,
                    expected_gmu=planned_job["requirements"]["gpu_memory_utilization"],
                )
            )
        elif sentinel == "a10g_16k_c4_capacity":
            _validate_a10g_capacity_measurements(measurements)
        elif sentinel == "generation_throughput_with_writes":
            generation_results[planned_job["hardware_id"]] = (
                _validate_throughput_measurements(
                    measurements,
                    hardware_id=planned_job["hardware_id"],
                )
            )
        elif sentinel == "auto_backend_diagnostic":
            _validate_auto_backend_measurements(measurements)
        else:
            raise ValueError(f"unsupported frozen sentinel: {sentinel!r}")
        job_records.append(result)

    if len(ledger_ids) != 1:
        raise ValueError("all terminal receipts must bind one ledger_id")

    expected_parallel = _max_parallel_jobs(job_records)
    if (
        type(record.get("max_parallel_jobs_observed")) is not int
        or record.get("max_parallel_jobs_observed") != expected_parallel
    ):
        raise ValueError("max_parallel_jobs_observed does not match job timestamps")
    if expected_parallel > GPU_QUALIFICATION_MAX_CLOUD_JOBS:
        raise ValueError("observed cloud parallelism exceeds the hard cap")
    qualifying_gmu = [gmu for gmu, qualified in gmu_candidates if qualified]
    if not qualifying_gmu:
        raise ValueError("no 32k c4 GMU candidate met capacity and headroom gates")
    selected_gmu = max(qualifying_gmu)
    observed_selection = _finite_float(
        record.get("selected_gpu_memory_utilization"),
        "selected_gpu_memory_utilization",
    )
    if not math.isclose(observed_selection, selected_gmu, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("selected GMU must be the highest qualifying sweep candidate")
    expected_generation_hardware = {
        "aws-g6-l4",
        GPU_QUALIFICATION_GENERATION_HARDWARE_ID,
    }
    if set(generation_results) != expected_generation_hardware:
        raise ValueError("generation qualification must cover L4 and L40S")
    l4_digest, _l4_rate, l4_samples = generation_results["aws-g6-l4"]
    l40s_digest, l40s_rate, l40s_samples = generation_results[
        GPU_QUALIFICATION_GENERATION_HARDWARE_ID
    ]
    if l4_samples != l40s_samples or l4_digest != l40s_digest:
        raise ValueError("L4 and L40S generation artifacts are not byte-identical")
    if l40s_rate < GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND:
        raise ValueError("L40S generation throughput is below the launch threshold")
    first_start = min(
        _parse_utc_timestamp(job["started_at_utc"], field_name="started_at_utc")
        for job in job_records
    )
    return (
        GPUQualificationSelection(
            attention_backend=GPU_QUALIFICATION_PUBLICATION_BACKEND,
            gpu_memory_utilization=selected_gmu,
            generation_hardware_id=(GPU_QUALIFICATION_GENERATION_HARDWARE_ID),
            generation_databricks_node_type_id=(
                GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE
            ),
            generation_artifacts_sha256=l40s_digest,
            generation_prefix_tokens_per_second=l40s_rate,
            plan_sha256=plan_sha256,
        ),
        first_start,
    )


def _validate_job_result_common(
    result: Mapping[str, Any],
    *,
    planned_job: Mapping[str, Any],
    expected_artifact_pins: GPUQualificationArtifactPins,
    expected_plan_sha256: str,
) -> None:
    label = f"job result {planned_job['job_id']}"
    _require_exact_keys(result, _JOB_RESULT_KEYS, label)
    _require_closed_record_digest(result, label)
    if result.get("record_type") != GPU_QUALIFICATION_JOB_RESULT_RECORD_TYPE:
        raise ValueError(f"{label} has unexpected record_type")
    if result.get("authorization_scope") != (
        "measurement_only_requires_direct_terminal_receipt"
    ):
        raise ValueError(f"{label} cannot claim standalone authorization")
    if (
        type(result.get("schema_version")) is not int
        or result.get("schema_version") != GPU_QUALIFICATION_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} has unexpected schema_version")
    for field_name in ("job_id", "hardware_id", "gpu"):
        if result.get(field_name) != planned_job[field_name]:
            raise ValueError(f"{label} {field_name} does not match the plan")
    if result.get("gpu_compute_capability") != planned_job["compute_capability"]:
        raise ValueError(f"{label} gpu_compute_capability does not match the plan")
    if result.get("status") != "SUCCESS":
        raise ValueError(f"{label} did not succeed")
    if any(
        type(result.get(field_name)) is not int or result.get(field_name) != 0
        for field_name in ("attempt_number", "retry_count", "max_retries")
    ):
        raise ValueError(f"{label} must succeed on attempt 0 without retries")
    if result.get("vllm_version") != GPU_QUALIFICATION_VLLM_VERSION:
        raise ValueError(f"{label} vLLM version mismatch")
    if result.get("torch_cuda_version") != "12.9":
        raise ValueError(f"{label} must execute the cu129 torch stack")
    if result.get("artifact_sha256") != expected_artifact_pins.to_record():
        raise ValueError(f"{label} artifact hashes do not match the frozen inputs")
    if result.get("plan_sha256") != expected_plan_sha256:
        raise ValueError(f"{label} plan_sha256 does not match the frozen plan")
    for field_name in ("cloud_cluster_id", "nvidia_driver_version"):
        value = result.get(field_name)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{label} {field_name} must be non-empty")
    _canonical_databricks_run_id(result.get("cloud_run_id"), f"{label}.cloud_run_id")
    expected_attempt_id = _expected_reservation_attempt_id(
        expected_plan_sha256, str(planned_job["job_id"])
    )
    if result.get("reservation_attempt_id") != expected_attempt_id:
        raise ValueError(f"{label} reservation_attempt_id does not match the plan")
    expected_task_key = _expected_task_key(str(planned_job["job_id"]))
    if result.get("task_key") != expected_task_key:
        raise ValueError(f"{label} task_key does not match the plan")
    _validate_output_json_binding(
        result.get("output_json"),
        plan_sha256=expected_plan_sha256,
        job_id=str(planned_job["job_id"]),
        label=f"{label}.output_json",
    )
    started = _parse_utc_timestamp(
        result.get("started_at_utc"), field_name=f"{label}.started_at_utc"
    )
    finished = _parse_utc_timestamp(
        result.get("finished_at_utc"), field_name=f"{label}.finished_at_utc"
    )
    if finished <= started:
        raise ValueError(f"{label} finished_at_utc must follow started_at_utc")


def _validate_terminal_receipt(
    receipt: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    plan_record: Mapping[str, Any],
) -> None:
    label = f"terminal receipt {planned_job['job_id']}"
    plan_sha256 = str(plan_record["closed_record_sha256"])
    _require_exact_keys(receipt, _TERMINAL_RECEIPT_KEYS, label)
    _require_closed_record_digest(receipt, label)
    exact_values: dict[str, Any] = {
        "authorization_source": "direct_databricks_runs_get",
        "cloud_cluster_id": result["cloud_cluster_id"],
        "cloud_run_id": result["cloud_run_id"],
        "job_id": planned_job["job_id"],
        "life_cycle_state": "TERMINATED",
        "output_json": result["output_json"],
        "plan_sha256": plan_sha256,
        "record_type": GPU_QUALIFICATION_TERMINAL_RECEIPT_RECORD_TYPE,
        "reservation_attempt_id": result["reservation_attempt_id"],
        "result_record_sha256": result["closed_record_sha256"],
        "result_state": "SUCCESS",
        "schema_version": GPU_QUALIFICATION_SCHEMA_VERSION,
        "task_attempt_number": 0,
        "task_key": result["task_key"],
        "task_life_cycle_state": "TERMINATED",
        "task_max_retries": 0,
        "task_result_state": "SUCCESS",
    }
    for field_name, expected in exact_values.items():
        if not _is_exact_scalar(receipt.get(field_name), expected):
            raise ValueError(f"{label} {field_name} does not match")
    expected_run_name = _expected_run_name(
        str(plan_record["campaign_id"]), str(planned_job["job_id"])
    )
    if receipt.get("run_name") != expected_run_name:
        raise ValueError(f"{label} run_name does not match the plan")
    _canonical_databricks_run_id(receipt.get("task_run_id"), f"{label}.task_run_id")
    expected_node_type = _expected_node_type(str(planned_job["hardware_id"]))
    for field_name in ("node_type_id", "driver_node_type_id"):
        if receipt.get(field_name) != expected_node_type:
            raise ValueError(f"{label} {field_name} does not match the plan")
    for field_name in ("ledger_id",):
        value = receipt.get(field_name)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{label} {field_name} must be non-empty")
    for field_name in (
        "control_plane_status_sha256",
        "ledger_terminal_actual_sha256",
        "phase_batch_record_sha256",
        "result_file_sha256",
        "submit_payload_sha256",
    ):
        _require_sha256(receipt.get(field_name), field_name=f"{label}.{field_name}")
    terminal_prefix = databricks_ledger_prefix_from_record(
        _mapping(receipt.get("phase_terminal_prefix"), f"{label}.phase_terminal_prefix")
    )
    if terminal_prefix.ledger_id != receipt.get("ledger_id"):
        raise ValueError(f"{label} phase terminal prefix ledger identity differs")
    expected_file_sha256 = sha256(
        (canonical_gpu_qualification_json(result) + "\n").encode("utf-8")
    ).hexdigest()
    if receipt.get("result_file_sha256") != expected_file_sha256:
        raise ValueError(f"{label} does not bind the canonical result file")

    times: dict[str, int] = {}
    for field_name in (
        "start_time_ms",
        "end_time_ms",
        "task_start_time_ms",
        "task_end_time_ms",
    ):
        value = receipt.get(field_name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} {field_name} must be a non-negative integer")
        times[field_name] = value
    if not (
        times["start_time_ms"]
        <= times["task_start_time_ms"]
        < times["task_end_time_ms"]
        <= times["end_time_ms"]
    ):
        raise ValueError(f"{label} run/task times are not nested increasing intervals")
    duration_seconds = (
        times["task_end_time_ms"] - times["task_start_time_ms"]
    ) / 1000.0
    if not _is_exact_scalar(
        receipt.get("ledger_actual_cluster_duration_seconds"),
        duration_seconds,
    ):
        raise ValueError(f"{label} ledger duration does not match the task interval")
    ledger_terminal_actual = {
        "actual_cluster_duration_seconds": duration_seconds,
        "actual_cluster_hours": duration_seconds / 3600.0,
        "attempt_id": result["reservation_attempt_id"],
        "control_plane_status_sha256": receipt["control_plane_status_sha256"],
        "run_id": result["cloud_run_id"],
        "submit_payload_sha256": receipt["submit_payload_sha256"],
        "terminal_state": "succeeded",
        "verification_source": "direct_databricks_runs_get",
    }
    if (
        receipt.get("ledger_terminal_actual_sha256")
        != sha256(
            canonical_gpu_qualification_json(ledger_terminal_actual).encode("utf-8")
        ).hexdigest()
    ):
        raise ValueError(f"{label} does not bind the reconciled ledger terminal event")
    result_started_ms = int(
        _parse_utc_timestamp(
            result.get("started_at_utc"), field_name=f"{label}.result_started"
        ).timestamp()
        * 1000
    )
    result_finished_ms = int(
        _parse_utc_timestamp(
            result.get("finished_at_utc"), field_name=f"{label}.result_finished"
        ).timestamp()
        * 1000
    )
    if not (
        times["task_start_time_ms"]
        <= result_started_ms
        < result_finished_ms
        <= times["task_end_time_ms"]
    ):
        raise ValueError(f"{label} does not enclose the GPU measurement interval")
    collected = _parse_utc_timestamp(
        receipt.get("collected_at_utc"), field_name=f"{label}.collected_at_utc"
    )
    terminal_end = datetime.fromtimestamp(times["end_time_ms"] / 1000.0, tz=UTC)
    if collected < terminal_end:
        raise ValueError(f"{label} was allegedly collected before terminal end")


def _validate_runtime_handoff_measurements(
    value: Mapping[str, Any], *, hardware_id: str
) -> None:
    expected_keys = frozenset(
        {
            "attention_backend_observed",
            "attention_backend_requested",
            "compute_dtype",
            "connector_source_sha256",
            "direct_url_matches_patched_wheel",
            "driver_cuda_compatibility_ok",
            "e5m2_software_path_exercised",
            "finite_logits",
            "handoff_injected",
            "handoff_injected_layer_count",
            "handoff_kv_bits",
            "handoff_kv_dtype",
            "handoff_loaded",
            "handoff_loaded_layer_count",
            "handoff_written",
            "handoff_written_layer_count",
            "installed_core_distribution_versions",
            "installed_connector_base_py_sha256",
            "installed_patch_member_sha256",
            "libcudart_major_versions",
            "libcudart_so_12_present",
            "libcudart_so_13_present",
            "model_id",
            "model_revision",
            "native_shared_object_count",
            "native_shared_object_evidence",
            "pip_check_ok",
            "python_version",
            "query_dtype",
            "runtime_kv_dtype",
            "runtime_kv_bits",
            "runtime_lock_attestation",
            "runtime_lock_verifier_ok",
            "site_packages_read_only",
            "strict_direct_url_verifier_ok",
            "system_cuda_version",
            "glibc_version",
            "triton_cache_miss_compile",
            "triton_compile_count",
            "triton_compiled_kernel_names",
            "triton_kernel_launch_count",
            "unresolved_native_shared_object_count",
            "weight_bits",
            "weight_quantization",
            "trust_remote_code",
            "weight_quantizer_attestation",
        }
    )
    _require_exact_keys(value, expected_keys, "runtime/handoff measurements")
    expected_values: dict[str, Any] = {
        "attention_backend_observed": GPU_QUALIFICATION_PUBLICATION_BACKEND,
        "attention_backend_requested": GPU_QUALIFICATION_PUBLICATION_BACKEND,
        "compute_dtype": "bfloat16",
        "connector_source_sha256": GPU_QUALIFICATION_CONNECTOR_SOURCE_SHA256,
        "direct_url_matches_patched_wheel": True,
        "driver_cuda_compatibility_ok": True,
        "finite_logits": True,
        "handoff_injected": True,
        "handoff_injected_layer_count": GPU_QUALIFICATION_MODEL_LAYER_COUNT,
        "handoff_kv_bits": 8,
        "handoff_kv_dtype": "fp8_e5m2",
        "handoff_loaded": True,
        "handoff_loaded_layer_count": GPU_QUALIFICATION_MODEL_LAYER_COUNT,
        "handoff_written": True,
        "handoff_written_layer_count": GPU_QUALIFICATION_MODEL_LAYER_COUNT,
        "installed_core_distribution_versions": dict(_RUNTIME_CORE_VERSIONS),
        "installed_connector_base_py_sha256": (
            GPU_QUALIFICATION_CONNECTOR_SOURCE_SHA256
        ),
        "installed_patch_member_sha256": dict(_PATCH_MEMBER_SHA256),
        "libcudart_major_versions": [12],
        "libcudart_so_12_present": True,
        "libcudart_so_13_present": False,
        "model_id": GPU_QUALIFICATION_MODEL_ID,
        "model_revision": GPU_QUALIFICATION_MODEL_REVISION,
        "pip_check_ok": True,
        "python_version": "3.11.11",
        "query_dtype": "bfloat16",
        "runtime_kv_dtype": "fp8_e5m2",
        "runtime_kv_bits": 8,
        "runtime_lock_verifier_ok": True,
        "site_packages_read_only": True,
        "strict_direct_url_verifier_ok": True,
        "system_cuda_version": "12.1",
        "glibc_version": "2.35",
        "triton_cache_miss_compile": True,
        "triton_compiled_kernel_names": list(_FORCED_TRITON_KERNEL_NAMES),
        "unresolved_native_shared_object_count": 0,
        "weight_bits": 4,
        "weight_quantization": "bitsandbytes",
        "trust_remote_code": False,
    }
    for field_name, expected in expected_values.items():
        if not _is_exact_scalar(value.get(field_name), expected):
            raise ValueError(f"runtime/handoff {field_name} is not qualified")
    _require_positive_int(value.get("triton_compile_count"), "triton_compile_count")
    _require_positive_int(
        value.get("triton_kernel_launch_count"), "triton_kernel_launch_count"
    )
    native_shared_object_count = _positive_int(
        value.get("native_shared_object_count"), "native_shared_object_count"
    )
    _validate_native_shared_object_evidence(
        _sequence(
            value.get("native_shared_object_evidence"),
            "native_shared_object_evidence",
        ),
        expected_count=native_shared_object_count,
    )
    software_path = value.get("e5m2_software_path_exercised")
    if not isinstance(software_path, bool):
        raise ValueError("e5m2_software_path_exercised must be boolean")
    if hardware_id == "aws-g5-a10g" and software_path is not True:
        raise ValueError("A10G must exercise the E5M2 software path")
    _validate_weight_quantizer_attestation(
        _mapping(
            value.get("weight_quantizer_attestation"),
            "weight_quantizer_attestation",
        )
    )
    _validate_runtime_lock_attestation(
        _mapping(
            value.get("runtime_lock_attestation"),
            "runtime_lock_attestation",
        )
    )


def _validate_native_shared_object_evidence(
    value: Sequence[Any],
    *,
    expected_count: int,
) -> None:
    if len(value) != expected_count:
        raise ValueError(
            "native shared-object evidence count differs from "
            "native_shared_object_count"
        )
    expected_record_keys = frozenset(
        {
            "distribution",
            "distribution_version",
            "is_symlink",
            "ldd_returncode",
            "ldd_stderr",
            "ldd_stderr_lines",
            "ldd_stderr_sha256",
            "ldd_stderr_utf8_bytes",
            "ldd_stdout",
            "ldd_stdout_lines",
            "ldd_stdout_sha256",
            "ldd_stdout_utf8_bytes",
            "member",
            "path",
            "resolved_path",
            "soname_bindings",
        }
    )
    expected_empty_sha256 = sha256(b"").hexdigest()
    owners: set[str] = set()
    ordering: list[tuple[str, str, str]] = []
    object_paths: list[tuple[str, str, str, bool]] = []
    path_owners: dict[str, str] = {}
    distribution_roots: dict[str, str] = {}
    for index, raw_record in enumerate(value):
        label = f"native shared-object evidence {index}"
        record = _mapping(raw_record, label)
        _require_exact_keys(record, expected_record_keys, label)
        distribution = record.get("distribution")
        if distribution not in _NATIVE_SHARED_OBJECT_DISTRIBUTIONS:
            raise ValueError(f"{label} distribution is not audited")
        assert isinstance(distribution, str)
        owners.add(distribution)
        expected_version = (
            GPU_QUALIFICATION_VLLM_VERSION
            if distribution == "vllm"
            else _RUNTIME_CORE_VERSIONS[distribution]
        )
        if record.get("distribution_version") != expected_version:
            raise ValueError(f"{label} distribution_version differs")
        member = _canonical_native_member(record.get("member"), f"{label}.member")
        member_path = PurePosixPath(member)
        path = _canonical_native_path(record.get("path"), f"{label}.path")
        path_object = PurePosixPath(path)
        if (
            len(path_object.parts) <= len(member_path.parts)
            or path_object.parts[-len(member_path.parts) :] != member_path.parts
        ):
            raise ValueError(f"{label} path does not end with its owned member")
        distribution_root = PurePosixPath(
            *path_object.parts[: -len(member_path.parts)]
        )
        root_text = distribution_root.as_posix()
        previous_root = distribution_roots.setdefault(distribution, root_text)
        if previous_root != root_text:
            raise ValueError(f"{label} distribution has multiple owned roots")
        resolved_path = _canonical_native_path(
            record.get("resolved_path"), f"{label}.resolved_path"
        )
        resolved_object = PurePosixPath(resolved_path)
        if not resolved_object.is_relative_to(distribution_root):
            raise ValueError(f"{label} resolved_path escapes its distribution root")
        is_symlink = record.get("is_symlink")
        if type(is_symlink) is not bool:
            raise ValueError(f"{label} is_symlink must be boolean")
        assert isinstance(is_symlink, bool)
        if not is_symlink and resolved_path != path:
            raise ValueError(f"{label} non-symlink path and resolved_path differ")
        if type(record.get("ldd_returncode")) is not int or record.get(
            "ldd_returncode"
        ) != 0:
            raise ValueError(f"{label} ldd_returncode must be integer zero")
        for stream_name in ("stdout", "stderr"):
            stream_field = f"ldd_{stream_name}"
            digest_field = f"ldd_{stream_name}_sha256"
            bytes_field = f"ldd_{stream_name}_utf8_bytes"
            lines_field = f"ldd_{stream_name}_lines"
            stream = record.get(stream_field)
            if not isinstance(stream, str):
                raise ValueError(f"{label} {stream_field} must be a string")
            encoded_stream = stream.encode("utf-8")
            if len(encoded_stream) > _NATIVE_LDD_MAX_STREAM_BYTES:
                raise ValueError(f"{label} {stream_field} exceeds the evidence bound")
            if record.get(digest_field) != sha256(encoded_stream).hexdigest():
                raise ValueError(f"{label} {stream_field} digest differs")
            byte_count = record.get(bytes_field)
            if type(byte_count) is not int or byte_count != len(encoded_stream):
                raise ValueError(f"{label} {stream_field} byte count differs")
            lines = _sequence(record.get(lines_field), f"{label}.{lines_field}")
            normalized_lines: list[str] = []
            for line in lines:
                if (
                    not isinstance(line, str)
                    or not line
                    or line.strip() != line
                    or any(ord(character) < 32 for character in line)
                ):
                    raise ValueError(f"{label} {lines_field} contains an invalid line")
                normalized_lines.append(line)
            expected_lines = sorted(
                line.strip() for line in stream.splitlines() if line.strip()
            )
            if normalized_lines != expected_lines:
                raise ValueError(f"{label} {lines_field} differs from {stream_field}")
        if (
            record.get("ldd_stderr") != ""
            or record.get("ldd_stderr_utf8_bytes") != 0
            or record.get("ldd_stderr_sha256") != expected_empty_sha256
            or record.get("ldd_stderr_lines") != []
        ):
            raise ValueError(f"{label} ldd stderr is not empty")
        parsed_bindings = _native_ldd_soname_bindings(
            cast(str, record.get("ldd_stdout")),
            label=label,
        )
        raw_bindings = _sequence(
            record.get("soname_bindings"), f"{label}.soname_bindings"
        )
        bindings: list[tuple[str, str]] = []
        for binding_index, raw_binding in enumerate(raw_bindings):
            binding_label = f"{label}.soname_bindings {binding_index}"
            binding = _mapping(raw_binding, binding_label)
            _require_exact_keys(
                binding,
                frozenset({"resolved_path", "soname"}),
                binding_label,
            )
            soname = binding.get("soname")
            if (
                not isinstance(soname, str)
                or not soname
                or any(character.isspace() for character in soname)
            ):
                raise ValueError(f"{binding_label} SONAME is invalid")
            resolved_binding = _canonical_native_path(
                binding.get("resolved_path"), f"{binding_label}.resolved_path"
            )
            bindings.append((soname, resolved_binding))
        if bindings != sorted(bindings) or len(set(bindings)) != len(bindings):
            raise ValueError(f"{label} soname_bindings are not canonical and unique")
        if bindings != parsed_bindings:
            raise ValueError(f"{label} soname_bindings differ from ldd_stdout")
        previous_owner = path_owners.setdefault(path, distribution)
        if previous_owner != distribution:
            raise ValueError(f"{label} path is claimed by multiple distributions")
        ordering.append((distribution, member, path))
        object_paths.append((distribution, path, resolved_path, is_symlink))
    if owners != _NATIVE_SHARED_OBJECT_DISTRIBUTIONS:
        raise ValueError("native shared-object evidence owner closure differs")
    if ordering != sorted(ordering) or len(set(ordering)) != len(ordering):
        raise ValueError("native shared-object evidence is not canonical and unique")
    regular_paths_by_owner: dict[str, set[str]] = {}
    for distribution, path, _resolved_path, is_symlink in object_paths:
        if not is_symlink:
            regular_paths_by_owner.setdefault(distribution, set()).add(path)
    for distribution, _path, resolved_path, is_symlink in object_paths:
        if is_symlink and resolved_path not in regular_paths_by_owner.get(
            distribution, set()
        ):
            raise ValueError(
                "native shared-object symlink target is not owned by its distribution"
            )


def _canonical_native_member(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    member = PurePosixPath(value)
    if (
        not value
        or value != member.as_posix()
        or member.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in member.parts)
        or _NATIVE_SHARED_OBJECT_NAME_RE.fullmatch(member.name) is None
    ):
        raise ValueError(f"{label} must be a canonical shared-object member")
    return value


def _canonical_native_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or not path.is_absolute()
        or value.startswith("//")
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{label} must be a canonical absolute path")
    return value


def _native_ldd_soname_bindings(stdout: str, *, label: str) -> list[tuple[str, str]]:
    bindings: list[tuple[str, str]] = []
    observed_sonames: set[str] = set()
    for raw_line in stdout.splitlines():
        stripped = raw_line.strip()
        if "=>" not in stripped:
            continue
        parts = stripped.split("=>")
        if len(parts) != 2:
            raise ValueError(f"{label} ldd_stdout repeats a binding separator")
        soname = parts[0].strip()
        resolution = parts[1].strip()
        if (
            not soname
            or any(character.isspace() for character in soname)
            or soname in observed_sonames
        ):
            raise ValueError(f"{label} ldd_stdout SONAME is invalid or repeated")
        observed_sonames.add(soname)
        if resolution == "not found":
            raise ValueError(f"{label} contains an unresolved ldd binding")
        match = re.fullmatch(
            r"(?P<path>/.*?)(?:\s+\(0x[0-9a-fA-F]+\))?",
            resolution,
        )
        if match is None:
            raise ValueError(f"{label} ldd_stdout binding is not an absolute path")
        resolved_path = _canonical_native_path(
            match.group("path"), f"{label} ldd_stdout binding path"
        )
        bindings.append((soname, resolved_path))
    return sorted(bindings)


def _validate_runtime_lock_attestation(value: Mapping[str, Any]) -> None:
    expected_keys = frozenset(
        {
            "locked_distribution_count",
            "ok",
            "runtime_lock_sha256",
            "unexpected_distributions",
            "vllm_direct_url",
            "vllm_package_version",
            "vllm_wheel_sha256",
        }
    )
    _require_exact_keys(value, expected_keys, "runtime lock attestation")
    expected = {
        "locked_distribution_count": VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
        "ok": True,
        "runtime_lock_sha256": VLLM_RUNTIME_LOCK_SHA256,
        "unexpected_distributions": [],
        "vllm_package_version": GPU_QUALIFICATION_VLLM_VERSION,
        "vllm_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    }
    for field_name, expected_value in expected.items():
        if not _is_exact_scalar(value.get(field_name), expected_value):
            raise ValueError(f"runtime lock attestation {field_name} differs")
    direct_url = value.get("vllm_direct_url")
    if not isinstance(direct_url, str) or not direct_url.startswith("file:"):
        raise ValueError("runtime lock attestation vLLM origin is not local")


def _validate_packed_roundtrip_measurements(value: Mapping[str, Any]) -> None:
    expected_keys = frozenset(
        {
            "attention_backend_observed",
            "attention_backend_requested",
            "cases",
            "triton_compile_count",
            "triton_kernel_launch_count",
        }
    )
    _require_exact_keys(value, expected_keys, "packed roundtrip measurements")
    _validate_forced_triton(value, "packed roundtrip")
    cases = _sequence(value.get("cases"), "packed roundtrip cases")
    if len(cases) != 2:
        raise ValueError("packed roundtrip must cover NHD and HND exactly once")
    case_keys = frozenset(
        {
            "bf16_reference_max_abs_error",
            "bf16_reference_scope",
            "cache_page_layout",
            "cache_page_shape",
            "input_value_max",
            "input_value_min",
            "negative_slot_guard_passed",
            "noncontiguous_stride_passed",
            "partial_slot_guard_passed",
            "payload_layout",
            "query_dtype",
            "raw_byte_mismatch_count",
            "raw_bytes_written",
            "read_raw_sha256",
            "untouched_guard_mismatch_count",
            "written_raw_sha256",
        }
    )
    for expected_layout, raw_case in zip(("NHD", "HND"), cases, strict=True):
        case = _mapping(raw_case, f"packed {expected_layout} case")
        _require_exact_keys(case, case_keys, f"packed {expected_layout} case")
        exact: dict[str, Any] = {
            "bf16_reference_scope": "attention_output",
            "cache_page_layout": "B_H_N_2D",
            "cache_page_shape": ["B", "H", "N", "2D"],
            "negative_slot_guard_passed": True,
            "noncontiguous_stride_passed": True,
            "partial_slot_guard_passed": True,
            "payload_layout": expected_layout,
            "query_dtype": "bfloat16",
        }
        for field_name, expected in exact.items():
            if not _is_exact_scalar(case.get(field_name), expected):
                raise ValueError(f"packed {expected_layout} {field_name} failed")
        if _finite_float(case.get("input_value_max"), "input_value_max") != 1.0:
            raise ValueError(f"packed {expected_layout} input_value_max failed")
        if _finite_float(case.get("input_value_min"), "input_value_min") != -1.0:
            raise ValueError(f"packed {expected_layout} input_value_min failed")
        _require_zero_int(
            case.get("raw_byte_mismatch_count"), "raw_byte_mismatch_count"
        )
        _require_zero_int(
            case.get("untouched_guard_mismatch_count"),
            "untouched_guard_mismatch_count",
        )
        _require_positive_int(case.get("raw_bytes_written"), "raw_bytes_written")
        written = case.get("written_raw_sha256")
        read = case.get("read_raw_sha256")
        _require_sha256(written, field_name="written_raw_sha256")
        _require_sha256(read, field_name="read_raw_sha256")
        if written != read:
            raise ValueError(f"packed {expected_layout} raw bytes did not round-trip")
        error = _finite_float(
            case.get("bf16_reference_max_abs_error"),
            "bf16_reference_max_abs_error",
        )
        if error < 0.0 or error > GPU_QUALIFICATION_MAX_E5M2_BF16_ERROR:
            raise ValueError(
                f"packed {expected_layout} exceeds BF16 reference tolerance"
            )


def _validate_token_determinism_measurements(
    value: Mapping[str, Any], *, expected_input_bundle_sha256: str
) -> None:
    expected_keys = frozenset(
        {
            "attention_backend_observed",
            "attention_backend_requested",
            "baseline_handoff_absent",
            "cache_phases",
            "execution_mode",
            "examples",
            "request_parallelism",
            "triton_compile_count",
            "triton_kernel_launch_count",
            "vanilla_handoff_injected",
            "trust_remote_code",
        }
    )
    _require_exact_keys(value, expected_keys, "token determinism measurements")
    _validate_forced_triton(value, "token determinism")
    token_execution_contract: dict[str, Any] = {
        "baseline_handoff_absent": True,
        "cache_phases": ["cold", "warm"],
        "execution_mode": "real_end_to_end_requests",
        "request_parallelism": 1,
        "vanilla_handoff_injected": True,
        "trust_remote_code": False,
    }
    for field_name, expected in token_execution_contract.items():
        if not _is_exact_scalar(value.get(field_name), expected):
            raise ValueError(f"token determinism {field_name} contract mismatch")
    examples = _sequence(value.get("examples"), "matched token examples")
    if len(examples) != GPU_QUALIFICATION_MATCHED_EXAMPLES:
        raise ValueError(
            "token determinism evidence has the wrong matched example count"
        )
    example_keys = frozenset(
        {
            "arms",
            "baseline_full_prompt_token_ids_sha256",
            "example_id",
            "full_prompt_token_count",
            "input_bundle_sha256",
            "vanilla_prefix_token_count",
            "vanilla_reconstructed_full_prompt_token_ids_sha256",
            "vanilla_suffix_token_count",
        }
    )
    previous_example_id = ""
    for raw_example in examples:
        example = _mapping(raw_example, "matched token example")
        _require_exact_keys(example, example_keys, "matched token example")
        example_id = example.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("matched token example_id must be non-empty")
        if example_id <= previous_example_id:
            raise ValueError(
                "matched token examples must be uniquely sorted by example_id"
            )
        previous_example_id = example_id
        input_bundle_sha256 = example.get("input_bundle_sha256")
        _require_sha256(input_bundle_sha256, field_name="input_bundle_sha256")
        if input_bundle_sha256 != expected_input_bundle_sha256:
            raise ValueError("matched token example input bundle hash mismatch")
        baseline_hash = example.get("baseline_full_prompt_token_ids_sha256")
        vanilla_hash = example.get("vanilla_reconstructed_full_prompt_token_ids_sha256")
        _require_sha256(
            baseline_hash, field_name="baseline_full_prompt_token_ids_sha256"
        )
        _require_sha256(
            vanilla_hash,
            field_name="vanilla_reconstructed_full_prompt_token_ids_sha256",
        )
        if baseline_hash != vanilla_hash:
            raise ValueError(
                "Baseline and Vanilla reconstructed token contracts differ"
            )
        full_count = _positive_int(
            example.get("full_prompt_token_count"), "full_prompt_token_count"
        )
        prefix_count = _positive_int(
            example.get("vanilla_prefix_token_count"), "vanilla_prefix_token_count"
        )
        suffix_count = _positive_int(
            example.get("vanilla_suffix_token_count"), "vanilla_suffix_token_count"
        )
        if prefix_count + suffix_count != full_count:
            raise ValueError(
                "Vanilla prefix and suffix do not reconstruct the full prompt"
            )
        arms = _sequence(example.get("arms"), "determinism arms")
        if len(arms) != 2:
            raise ValueError("determinism evidence must contain Baseline and Vanilla")
        logit_position_hashes: list[str] = []
        for expected_arm, raw_arm in zip(
            ("baseline_prefill", "vanilla_prefill"), arms, strict=True
        ):
            logit_position_hashes.append(
                _validate_determinism_arm(raw_arm, expected_arm=expected_arm)
            )
        if len(set(logit_position_hashes)) != 1:
            raise ValueError(
                "Baseline and Vanilla must probe logits at matched positions"
            )


def _validate_determinism_arm(value: Any, *, expected_arm: str) -> str:
    arm = _mapping(value, f"determinism arm {expected_arm}")
    expected_keys = frozenset(
        {
            "arm_id",
            "finite_logits",
            "logit_probe_position_ids_sha256",
            "max_abs_logit_drift",
            "output_token_count",
            "output_token_ids_repeat_sha256",
            "repeat_count",
        }
    )
    _require_exact_keys(arm, expected_keys, f"determinism arm {expected_arm}")
    if arm.get("arm_id") != expected_arm:
        raise ValueError("determinism arm order or identity mismatch")
    if type(arm.get("repeat_count")) is not int or arm.get("repeat_count") != (
        GPU_QUALIFICATION_DETERMINISM_REPEATS
    ):
        raise ValueError("determinism arm repeat_count mismatch")
    if arm.get("finite_logits") is not True:
        raise ValueError("determinism arm contains non-finite logits")
    _require_positive_int(arm.get("output_token_count"), "output_token_count")
    logit_positions_sha256 = arm.get("logit_probe_position_ids_sha256")
    _require_sha256(
        logit_positions_sha256,
        field_name="logit_probe_position_ids_sha256",
    )
    output_hashes = _sequence(
        arm.get("output_token_ids_repeat_sha256"), "output token repeat hashes"
    )
    if len(output_hashes) != GPU_QUALIFICATION_DETERMINISM_REPEATS:
        raise ValueError("determinism output hash repeat count mismatch")
    for output_hash in output_hashes:
        _require_sha256(output_hash, field_name="output_token_ids_repeat_sha256")
    if len(set(output_hashes)) != 1:
        raise ValueError("deterministic repeats produced different output token IDs")
    drift = _finite_float(arm.get("max_abs_logit_drift"), "max_abs_logit_drift")
    if drift < 0.0 or drift > GPU_QUALIFICATION_MAX_LOGIT_DRIFT:
        raise ValueError("deterministic repeat logits exceed the frozen drift bound")
    assert isinstance(logit_positions_sha256, str)
    return logit_positions_sha256


def _validate_gmu_measurements(
    value: Mapping[str, Any], *, expected_gmu: Any
) -> tuple[float, bool]:
    expected_keys = frozenset(
        {
            "active_request_memory_observation_count",
            "attention_backend_observed",
            "attention_backend_requested",
            "candidate_qualified",
            "cold_disk_evicted_file_count",
            "connector_loaded_layer_counts",
            "connector_successful_load_count",
            "fatal_error_count",
            "forced_decode_tokens",
            "gpu_memory_utilization",
            "input_tokens_per_request",
            "kv_cache_capacity_tokens",
            "max_model_len",
            "observed_peak_headroom_bytes",
            "observed_peak_used_memory_bytes",
            "observed_total_memory_bytes",
            "oom_count",
            "request_parallelism",
            "request_success_count",
            "q8_pre_rope_handoffs",
            "selected_examples",
            "triton_compile_count",
            "triton_kernel_launch_count",
            "trust_remote_code",
            "vanilla_handoff_injected",
            "weight_quantizer_attestation",
        }
    )
    _require_exact_keys(value, expected_keys, "GMU sweep measurements")
    _validate_forced_triton(value, "GMU sweep")
    if value.get("trust_remote_code") is not False:
        raise ValueError("GMU sweep must keep trust_remote_code disabled")
    _validate_connector_capacity_contract(
        value,
        input_tokens=GPU_QUALIFICATION_INPUT_CONTEXT_TOKENS,
        label="GMU sweep",
    )
    gmu = _finite_float(value.get("gpu_memory_utilization"), "gpu_memory_utilization")
    planned_gmu = _finite_float(expected_gmu, "planned gpu_memory_utilization")
    if not math.isclose(gmu, planned_gmu, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("GMU sweep result does not match its planned candidate")
    if (
        type(value.get("max_model_len")) is not int
        or value.get("max_model_len") != GPU_QUALIFICATION_MAX_MODEL_LEN
    ):
        raise ValueError("GMU sweep max_model_len mismatch")
    if (
        type(value.get("request_parallelism")) is not int
        or value.get("request_parallelism") != GPU_QUALIFICATION_REQUEST_PARALLELISM
    ):
        raise ValueError("GMU sweep request_parallelism mismatch")
    try:
        _require_zero_int(value.get("oom_count"), "oom_count")
        _require_zero_int(value.get("fatal_error_count"), "fatal_error_count")
    except ValueError as exc:
        raise ValueError("GMU sweep must have zero OOM and fatal errors") from exc
    capacity = _positive_int(
        value.get("kv_cache_capacity_tokens"), "kv_cache_capacity_tokens"
    )
    total = _positive_int(
        value.get("observed_total_memory_bytes"), "observed_total_memory_bytes"
    )
    used = _positive_int(
        value.get("observed_peak_used_memory_bytes"),
        "observed_peak_used_memory_bytes",
    )
    headroom = _positive_int(
        value.get("observed_peak_headroom_bytes"),
        "observed_peak_headroom_bytes",
    )
    if used >= total or total - used != headroom:
        raise ValueError(
            "GMU observed peak headroom does not match total minus peak used"
        )
    qualified = (
        capacity >= GPU_QUALIFICATION_REQUIRED_KV_CAPACITY_TOKENS
        and headroom >= GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES
    )
    if value.get("candidate_qualified") is not qualified:
        raise ValueError("GMU candidate_qualified disagrees with measured gates")
    return gmu, qualified


def _validate_a10g_capacity_measurements(value: Mapping[str, Any]) -> None:
    expected_keys = frozenset(
        {
            "active_request_memory_observation_count",
            "attention_backend_observed",
            "attention_backend_requested",
            "capacity_qualified",
            "cold_disk_evicted_file_count",
            "connector_loaded_layer_counts",
            "connector_successful_load_count",
            "fatal_error_count",
            "forced_decode_tokens",
            "gpu_memory_utilization",
            "input_tokens_per_request",
            "kv_cache_capacity_tokens",
            "max_model_len",
            "observed_peak_headroom_bytes",
            "observed_peak_used_memory_bytes",
            "observed_total_memory_bytes",
            "oom_count",
            "request_parallelism",
            "request_success_count",
            "q8_pre_rope_handoffs",
            "selected_examples",
            "triton_compile_count",
            "triton_kernel_launch_count",
            "trust_remote_code",
            "vanilla_handoff_injected",
            "weight_quantizer_attestation",
        }
    )
    _require_exact_keys(value, expected_keys, "A10G 16k c4 capacity measurements")
    _validate_forced_triton(value, "A10G 16k c4 capacity")
    if value.get("trust_remote_code") is not False:
        raise ValueError("A10G capacity must keep trust_remote_code disabled")
    _validate_connector_capacity_contract(
        value,
        input_tokens=GPU_QUALIFICATION_A10G_INPUT_CONTEXT_TOKENS,
        label="A10G capacity",
    )
    gmu = _finite_float(value.get("gpu_memory_utilization"), "gpu_memory_utilization")
    if not math.isclose(gmu, 0.90, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("A10G 16k c4 gpu_memory_utilization must equal 0.90")
    if (
        type(value.get("max_model_len")) is not int
        or value.get("max_model_len") != GPU_QUALIFICATION_A10G_MAX_MODEL_LEN
    ):
        raise ValueError("A10G capacity max_model_len mismatch")
    if (
        type(value.get("request_parallelism")) is not int
        or value.get("request_parallelism") != 4
    ):
        raise ValueError("A10G capacity request_parallelism must equal 4")
    try:
        _require_zero_int(value.get("oom_count"), "oom_count")
        _require_zero_int(value.get("fatal_error_count"), "fatal_error_count")
    except ValueError as exc:
        raise ValueError(
            "A10G capacity run must have zero OOM and fatal errors"
        ) from exc
    capacity = _positive_int(
        value.get("kv_cache_capacity_tokens"), "kv_cache_capacity_tokens"
    )
    total = _positive_int(
        value.get("observed_total_memory_bytes"), "observed_total_memory_bytes"
    )
    used = _positive_int(
        value.get("observed_peak_used_memory_bytes"),
        "observed_peak_used_memory_bytes",
    )
    headroom = _positive_int(
        value.get("observed_peak_headroom_bytes"),
        "observed_peak_headroom_bytes",
    )
    if used >= total or total - used != headroom:
        raise ValueError(
            "A10G observed peak headroom does not match total minus peak used"
        )
    qualified = capacity >= GPU_QUALIFICATION_A10G_REQUIRED_KV_CAPACITY_TOKENS and (
        headroom >= GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES
    )
    if value.get("capacity_qualified") is not qualified or not qualified:
        raise ValueError("A10G 16k c4 capacity/headroom gates did not pass")


def _validate_connector_capacity_contract(
    value: Mapping[str, Any], *, input_tokens: int, label: str
) -> None:
    if value.get("q8_pre_rope_handoffs") is not True:
        raise ValueError(f"{label} must use Q8 pre-RoPE handoffs")
    if value.get("vanilla_handoff_injected") is not True:
        raise ValueError(f"{label} must execute Vanilla connector injection")
    if value.get("input_tokens_per_request") != input_tokens:
        raise ValueError(f"{label} input token contract mismatch")
    if value.get("forced_decode_tokens") != (GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS):
        raise ValueError(f"{label} forced decode contract mismatch")
    if value.get("request_success_count") != GPU_QUALIFICATION_REQUEST_PARALLELISM:
        raise ValueError(f"{label} must complete four concurrent requests")
    if (
        _positive_int(
            value.get("active_request_memory_observation_count"),
            "active_request_memory_observation_count",
        )
        < 2
    ):
        raise ValueError(f"{label} memory sampler did not span active requests")
    if value.get("connector_successful_load_count") != (
        GPU_QUALIFICATION_REQUEST_PARALLELISM
    ):
        raise ValueError(f"{label} must observe four successful connector loads")
    layer_counts = _sequence(
        value.get("connector_loaded_layer_counts"),
        f"{label} connector layer counts",
    )
    if (
        list(layer_counts)
        != [GPU_QUALIFICATION_MODEL_LAYER_COUNT] * GPU_QUALIFICATION_REQUEST_PARALLELISM
    ):
        raise ValueError(f"{label} must inject all model layers for every request")
    _positive_int(
        value.get("cold_disk_evicted_file_count"),
        "cold_disk_evicted_file_count",
    )
    selected_examples = _sequence(
        value.get("selected_examples"), f"{label} selected_examples"
    )
    selected_identities: list[tuple[str, str]] = []
    for raw_example in selected_examples:
        example = _mapping(raw_example, f"{label} selected example")
        _require_exact_keys(
            example,
            frozenset({"dataset", "example_id"}),
            f"{label} selected example",
        )
        dataset = example.get("dataset")
        example_id = example.get("example_id")
        if not isinstance(dataset, str) or not isinstance(example_id, str):
            raise ValueError(f"{label} selected example identity is invalid")
        selected_identities.append((dataset, example_id))
    if (
        len(selected_identities) != len(GPU_QUALIFICATION_INPUT_DATASETS)
        or selected_identities != sorted(selected_identities)
        or {dataset for dataset, _example_id in selected_identities}
        != set(GPU_QUALIFICATION_INPUT_DATASETS)
    ):
        raise ValueError(f"{label} must bind one selected row per frozen dataset")
    if len({identity for identity in selected_identities}) != len(selected_identities):
        raise ValueError(f"{label} selected example identities must be unique")
    _validate_weight_quantizer_attestation(
        _mapping(
            value.get("weight_quantizer_attestation"),
            f"{label} weight_quantizer_attestation",
        )
    )


def _validate_throughput_measurements(
    value: Mapping[str, Any], *, hardware_id: str
) -> tuple[str, float, tuple[Mapping[str, Any], ...]]:
    expected_keys = frozenset(
        {
            "aggregate_prefix_tokens",
            "aggregate_tokens_per_second",
            "aggregate_wall_seconds",
            "attention_backend_observed",
            "attention_backend_requested",
            "buckets",
            "clock_scope",
            "failed_write_count",
            "generator_device_map",
            "samples",
            "triton_compile_count",
            "triton_kernel_launch_count",
            "writes_included",
            "trust_remote_code",
            "weight_quantizer_attestation",
        }
    )
    _require_exact_keys(value, expected_keys, "generation throughput measurements")
    _validate_forced_triton(value, "generation throughput")
    if value.get("trust_remote_code") is not False:
        raise ValueError("generation throughput must keep trust_remote_code disabled")
    if value.get("generator_device_map") != "auto":
        raise ValueError("generation throughput must use generator device_map=auto")
    _validate_weight_quantizer_attestation(
        _mapping(
            value.get("weight_quantizer_attestation"),
            "weight_quantizer_attestation",
        )
    )
    if value.get("clock_scope") != "prefix_generation_through_durable_kv_write":
        raise ValueError("throughput clock must include generation and durable writes")
    if value.get("writes_included") is not True:
        raise ValueError("throughput pilot requires successful writes inside the clock")
    try:
        _require_zero_int(value.get("failed_write_count"), "failed_write_count")
    except ValueError as exc:
        raise ValueError(
            "throughput pilot requires successful writes inside the clock"
        ) from exc
    buckets = _sequence(value.get("buckets"), "throughput buckets")
    if len(buckets) != len(GPU_QUALIFICATION_THROUGHPUT_BUCKETS):
        raise ValueError("throughput pilot is missing a frozen length bucket")
    total_tokens = 0
    total_seconds = 0.0
    bucket_prefix_tokens: dict[int, int] = {}
    bucket_keys = frozenset(
        {
            "durable_write_completed_count",
            "length_bucket_tokens",
            "prefix_tokens",
            "sample_count",
            "tokens_per_second",
            "wall_seconds",
        }
    )
    for expected_bucket, raw_bucket in zip(
        GPU_QUALIFICATION_THROUGHPUT_BUCKETS, buckets, strict=True
    ):
        bucket = _mapping(raw_bucket, f"throughput bucket {expected_bucket}")
        _require_exact_keys(bucket, bucket_keys, f"throughput bucket {expected_bucket}")
        if (
            type(bucket.get("length_bucket_tokens")) is not int
            or bucket.get("length_bucket_tokens") != expected_bucket
        ):
            raise ValueError("throughput length buckets must be in canonical order")
        if (
            type(bucket.get("sample_count")) is not int
            or bucket.get("sample_count")
            != GPU_QUALIFICATION_THROUGHPUT_SAMPLES_PER_BUCKET
        ):
            raise ValueError("throughput sample_count does not match the frozen pilot")
        if (
            type(bucket.get("durable_write_completed_count")) is not int
            or bucket.get("durable_write_completed_count")
            != GPU_QUALIFICATION_THROUGHPUT_SAMPLES_PER_BUCKET
        ):
            raise ValueError("throughput bucket did not durably write every sample")
        prefix_tokens = _positive_int(bucket.get("prefix_tokens"), "prefix_tokens")
        bucket_prefix_tokens[expected_bucket] = prefix_tokens
        wall_seconds = _positive_float(bucket.get("wall_seconds"), "wall_seconds")
        measured_rate = _positive_float(
            bucket.get("tokens_per_second"), "tokens_per_second"
        )
        computed_rate = prefix_tokens / wall_seconds
        if not math.isclose(measured_rate, computed_rate, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("throughput bucket rate does not match tokens / wall time")
        if (
            hardware_id == GPU_QUALIFICATION_GENERATION_HARDWARE_ID
            and measured_rate < GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND
        ):
            raise ValueError("throughput bucket is below the launch threshold")
        total_tokens += prefix_tokens
        total_seconds += wall_seconds
    if value.get("aggregate_prefix_tokens") != total_tokens:
        raise ValueError("aggregate throughput token count mismatch")
    aggregate_seconds = _positive_float(
        value.get("aggregate_wall_seconds"), "aggregate_wall_seconds"
    )
    if not math.isclose(aggregate_seconds, total_seconds, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("aggregate throughput wall time mismatch")
    aggregate_rate = _positive_float(
        value.get("aggregate_tokens_per_second"), "aggregate_tokens_per_second"
    )
    computed_aggregate = total_tokens / total_seconds
    if not math.isclose(aggregate_rate, computed_aggregate, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("aggregate throughput rate mismatch")
    if (
        hardware_id == GPU_QUALIFICATION_GENERATION_HARDWARE_ID
        and aggregate_rate < GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND
    ):
        raise ValueError("aggregate throughput is below the launch threshold")
    samples = _sequence(value.get("samples"), "throughput samples")
    expected_sample_count = (
        len(GPU_QUALIFICATION_THROUGHPUT_BUCKETS)
        * GPU_QUALIFICATION_THROUGHPUT_SAMPLES_PER_BUCKET
    )
    if len(samples) != expected_sample_count:
        raise ValueError("throughput evidence has the wrong sample count")
    sample_keys = frozenset(
        {
            "cache_prefix_token_count",
            "cache_prefix_token_ids_sha256",
            "dataset",
            "example_id",
            "input_tokens_target",
            "raw_artifact_bytes",
            "raw_artifact_sha256",
            "segment_count",
            "segments",
        }
    )
    segment_keys = frozenset({"index", "token_count", "token_ids_sha256"})
    normalized_samples: list[Mapping[str, Any]] = []
    previous_identity: tuple[int, str, str] | None = None
    counts_by_bucket = {bucket: 0 for bucket in GPU_QUALIFICATION_THROUGHPUT_BUCKETS}
    datasets_by_bucket: dict[int, set[str]] = {
        bucket: set() for bucket in GPU_QUALIFICATION_THROUGHPUT_BUCKETS
    }
    exact_tokens_by_bucket = {
        bucket: 0 for bucket in GPU_QUALIFICATION_THROUGHPUT_BUCKETS
    }
    for raw_sample in samples:
        sample = _mapping(raw_sample, "throughput sample")
        _require_exact_keys(sample, sample_keys, "throughput sample")
        target = _positive_int(sample.get("input_tokens_target"), "input_tokens_target")
        if target not in counts_by_bucket:
            raise ValueError("throughput sample has an unsupported length bucket")
        dataset = sample.get("dataset")
        example_id = sample.get("example_id")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("throughput sample dataset must be non-empty")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("throughput sample example_id must be non-empty")
        identity = (target, dataset, example_id)
        if previous_identity is not None and identity <= previous_identity:
            raise ValueError("throughput samples must be uniquely sorted")
        previous_identity = identity
        counts_by_bucket[target] += 1
        datasets_by_bucket[target].add(dataset)
        _require_sha256(
            sample.get("cache_prefix_token_ids_sha256"),
            field_name="cache_prefix_token_ids_sha256",
        )
        _require_sha256(
            sample.get("raw_artifact_sha256"),
            field_name="raw_artifact_sha256",
        )
        _positive_int(sample.get("raw_artifact_bytes"), "raw_artifact_bytes")
        prefix_count = _positive_int(
            sample.get("cache_prefix_token_count"),
            "cache_prefix_token_count",
        )
        exact_tokens_by_bucket[target] += prefix_count
        segment_count = _positive_int(sample.get("segment_count"), "segment_count")
        if segment_count != target // 2_048:
            raise ValueError(
                "throughput sample is not segmented at the 2k document grid"
            )
        segments = _sequence(sample.get("segments"), "throughput segments")
        if len(segments) != segment_count:
            raise ValueError("throughput segment count mismatch")
        observed_prefix_count = 0
        for expected_index, raw_segment in enumerate(segments):
            segment = _mapping(raw_segment, "throughput segment")
            _require_exact_keys(segment, segment_keys, "throughput segment")
            if segment.get("index") != expected_index:
                raise ValueError("throughput segments must be in canonical order")
            observed_prefix_count += _positive_int(
                segment.get("token_count"), "segment token_count"
            )
            _require_sha256(
                segment.get("token_ids_sha256"),
                field_name="segment token_ids_sha256",
            )
        if observed_prefix_count != prefix_count:
            raise ValueError("throughput segment tokens do not compose the prefix")
        normalized_samples.append(sample)
    if any(
        count != GPU_QUALIFICATION_THROUGHPUT_SAMPLES_PER_BUCKET
        for count in counts_by_bucket.values()
    ):
        raise ValueError("throughput evidence does not cover four samples per bucket")
    if any(
        datasets != set(GPU_QUALIFICATION_INPUT_DATASETS)
        for datasets in datasets_by_bucket.values()
    ):
        raise ValueError("throughput evidence must select every frozen dataset")
    if bucket_prefix_tokens != exact_tokens_by_bucket:
        raise ValueError(
            "throughput bucket prefix token counts do not match exact sample tokens"
        )
    artifacts_sha256 = sha256(
        canonical_gpu_qualification_json(
            {"samples": [dict(sample) for sample in normalized_samples]}
        ).encode("utf-8")
    ).hexdigest()
    return artifacts_sha256, aggregate_rate, tuple(normalized_samples)


def _validate_auto_backend_measurements(value: Mapping[str, Any]) -> None:
    expected_keys = frozenset(
        {
            "backend_selection_mode",
            "observed_backend",
            "publication_backend_changed",
            "trust_remote_code",
        }
    )
    _require_exact_keys(value, expected_keys, "auto backend diagnostic")
    if value.get("backend_selection_mode") != "auto":
        raise ValueError("auto backend diagnostic did not use auto selection")
    observed = value.get("observed_backend")
    if not isinstance(observed, str) or not observed:
        raise ValueError("auto backend diagnostic must record the observed backend")
    if value.get("publication_backend_changed") is not False:
        raise ValueError("auto backend diagnostic cannot change publication backend")
    if value.get("trust_remote_code") is not False:
        raise ValueError("auto backend diagnostic must keep trust_remote_code disabled")


def _validate_weight_quantizer_attestation(value: Mapping[str, Any]) -> None:
    expected_keys = frozenset(
        {
            "bitsandbytes_loader_sha256",
            "bitsandbytes_version",
            "dynamic_quant_call",
            "hf_generator_config",
        }
    )
    _require_exact_keys(value, expected_keys, "weight quantizer attestation")
    if value.get("bitsandbytes_loader_sha256") != (
        GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
    ):
        raise ValueError("bitsandbytes loader source hash mismatch")
    if value.get("bitsandbytes_version") != _RUNTIME_CORE_VERSIONS["bitsandbytes"]:
        raise ValueError("bitsandbytes version mismatch")
    dynamic = _mapping(value.get("dynamic_quant_call"), "dynamic quant call")
    _require_exact_keys(
        dynamic,
        frozenset(
            {
                "compress_statistics",
                "input_dtype",
                "nested_state",
                "packed_dtype",
                "quant_type",
            }
        ),
        "dynamic quant call",
    )
    expected_dynamic = {
        "compress_statistics": True,
        "input_dtype": "bfloat16",
        "nested_state": True,
        "packed_dtype": "uint8",
        "quant_type": "nf4",
    }
    if dict(dynamic) != expected_dynamic:
        raise ValueError("dynamic NF4/double-quant call contract mismatch")
    config = _mapping(value.get("hf_generator_config"), "HF generator config")
    expected_config = {
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_storage": "uint8",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "load_in_4bit": True,
    }
    _require_exact_keys(config, frozenset(expected_config), "HF generator config")
    if dict(config) != expected_config:
        raise ValueError("HF generator NF4 configuration mismatch")


def _validate_forced_triton(value: Mapping[str, Any], label: str) -> None:
    if value.get("attention_backend_requested") != (
        GPU_QUALIFICATION_PUBLICATION_BACKEND
    ) or value.get("attention_backend_observed") != (
        GPU_QUALIFICATION_PUBLICATION_BACKEND
    ):
        raise ValueError(f"{label} must request and observe forced TRITON_ATTN")
    _require_positive_int(value.get("triton_compile_count"), "triton_compile_count")
    _require_positive_int(
        value.get("triton_kernel_launch_count"), "triton_kernel_launch_count"
    )


def _plan_jobs(plan_record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cloud = _mapping(plan_record.get("cloud_qualification"), "cloud_qualification")
    return [
        _mapping(job, "planned job")
        for job in _sequence(cloud.get("jobs"), "planned jobs")
    ]


def _plan_job(plan_record: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job_id must be non-empty")
    matches = [job for job in _plan_jobs(plan_record) if job.get("job_id") == job_id]
    if len(matches) != 1:
        raise ValueError(f"job_id is not unique in the qualification plan: {job_id!r}")
    return matches[0]


def _hardware_compute_capability(hardware_id: str) -> str:
    matches = [
        compute_capability
        for expected_hardware_id, _gpu, compute_capability in _QUALIFICATION_HARDWARE
        if expected_hardware_id == hardware_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown qualification hardware_id: {hardware_id!r}")
    return matches[0]


def _max_parallel_jobs(jobs: Sequence[Mapping[str, Any]]) -> int:
    if not jobs:
        return 0
    events: list[tuple[datetime, int]] = []
    for index, job in enumerate(jobs):
        started = _parse_utc_timestamp(
            job.get("started_at_utc"), field_name=f"jobs[{index}].started_at_utc"
        )
        finished = _parse_utc_timestamp(
            job.get("finished_at_utc"), field_name=f"jobs[{index}].finished_at_utc"
        )
        if finished <= started:
            raise ValueError("cloud job finish time must be after its start time")
        events.append((started, 1))
        events.append((finished, -1))
    active = 0
    peak = 0
    for _timestamp, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        if active < 0:
            raise ValueError("cloud job timestamps produce an invalid overlap sequence")
        peak = max(peak, active)
    if active != 0:
        raise ValueError("cloud job timestamps do not form closed intervals")
    return peak


def _require_unique_identity(observed: set[str], value: str, field_name: str) -> None:
    if value in observed:
        raise ValueError(f"{field_name} values must be unique")
    observed.add(value)


def _canonical_databricks_run_id(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 128
        or re.fullmatch(r"[1-9][0-9]*", value) is None
    ):
        raise ValueError(
            f"{field_name} must be a strictly positive canonical decimal run ID"
        )
    return value


def _expected_reservation_attempt_id(plan_sha256: str, job_id: str) -> str:
    _require_sha256(plan_sha256, field_name="plan_sha256")
    if not job_id:
        raise ValueError("job_id must be non-empty")
    return f"gpuq-{plan_sha256[:16]}-{job_id}"


def _expected_task_key(job_id: str) -> str:
    value = "gpu_qualification_" + re.sub(r"[^a-zA-Z0-9_]", "_", job_id)
    if not value[0].isalpha() or len(value) > 100:
        raise ValueError(f"job_id cannot form a Databricks task key: {job_id!r}")
    return value


def _expected_run_name(campaign_id: str, job_id: str) -> str:
    if not campaign_id or not job_id:
        raise ValueError("campaign_id and job_id must be non-empty")
    return f"cachet-gpu-qualification-{campaign_id}-{job_id}"[:4096]


def _expected_node_type(hardware_id: str) -> str:
    expected = {
        "aws-g6-l4": "g6.8xlarge",
        "aws-g5-a10g": "g5.8xlarge",
        GPU_QUALIFICATION_GENERATION_HARDWARE_ID: (
            GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE
        ),
    }.get(hardware_id)
    if expected is None:
        raise ValueError(f"unsupported qualification hardware_id: {hardware_id!r}")
    return expected


def _validate_output_json_binding(
    value: Any,
    *,
    plan_sha256: str,
    job_id: str,
    label: str,
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty durable path")
    parsed = urlsplit(value)
    if parsed.query or parsed.fragment:
        raise ValueError(f"{label} cannot contain a query or fragment")
    if value.startswith("dbfs:/"):
        path = PurePosixPath("/dbfs") / value.removeprefix("dbfs:/").lstrip("/")
    elif parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"{label} file URI authority is unsupported")
        path = PurePosixPath(unquote(parsed.path))
    elif parsed.scheme:
        raise ValueError(f"{label} uses an unsupported URI scheme")
    else:
        path = PurePosixPath(value)
    if not path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts[1:]
    ):
        raise ValueError(f"{label} must be a normalized absolute durable path")
    if path.parts[:2] not in {("/", "dbfs"), ("/", "Volumes")}:
        raise ValueError(f"{label} must use DBFS or a UC Volume")
    expected_suffix = (plan_sha256, job_id, "gpu-job-result.json")
    if tuple(path.parts[-3:]) != expected_suffix:
        raise ValueError(f"{label} does not match the frozen plan/job path")
    return value


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("closed_record_sha256", None)
    return sha256(canonical_gpu_qualification_json(payload).encode("utf-8")).hexdigest()


def _require_closed_record_digest(record: Mapping[str, Any], label: str) -> None:
    digest = record.get("closed_record_sha256")
    _require_sha256(digest, field_name=f"{label}.closed_record_sha256")
    if digest != _closed_record_sha256(record):
        raise ValueError(f"{label} closed_record_sha256 is invalid")


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{label} must use a closed schema; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: Any, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence")
    return value


def _parse_utc_timestamp(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field_name} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid ISO-8601 timestamp") from exc
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{field_name} must be UTC")
    return parsed


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_int(value: Any, field_name: str) -> int:
    _require_positive_int(value, field_name)
    return int(value)


def _require_positive_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_zero_int(value: Any, field_name: str) -> None:
    if type(value) is not int or value != 0:
        raise ValueError(f"{field_name} must be integer zero")


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def _is_exact_scalar(value: Any, expected: Any) -> bool:
    return type(value) is type(expected) and value == expected


__all__ = [
    "GPU_QUALIFICATION_A10G_INPUT_CONTEXT_TOKENS",
    "GPU_QUALIFICATION_A10G_MAX_MODEL_LEN",
    "GPU_QUALIFICATION_A10G_REQUIRED_KV_CAPACITY_TOKENS",
    "GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256",
    "GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS",
    "GPU_QUALIFICATION_CLOUD_EVIDENCE_RECORD_TYPE",
    "GPU_QUALIFICATION_CONNECTOR_SOURCE_SHA256",
    "GPU_QUALIFICATION_DETERMINISM_REPEATS",
    "GPU_QUALIFICATION_EVIDENCE_RECORD_TYPE",
    "GPU_QUALIFICATION_GMU_SWEEP",
    "GPU_QUALIFICATION_GENERATION_COMPUTE_CAPABILITY",
    "GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE",
    "GPU_QUALIFICATION_GENERATION_GPU",
    "GPU_QUALIFICATION_GENERATION_HARDWARE_ID",
    "GPU_QUALIFICATION_INPUT_CONTEXT_TOKENS",
    "GPU_QUALIFICATION_INPUT_DATASETS",
    "GPU_QUALIFICATION_INPUT_ROWS_PER_DATASET_BUCKET",
    "GPU_QUALIFICATION_JOB_RESULT_RECORD_TYPE",
    "GPU_QUALIFICATION_DECODE_HEADROOM_TOKENS",
    "GPU_QUALIFICATION_LOCAL_PREFLIGHT_RECORD_TYPE",
    "GPU_QUALIFICATION_MATCHED_EXAMPLES",
    "GPU_QUALIFICATION_MAX_CLOUD_JOBS",
    "GPU_QUALIFICATION_MAX_E5M2_BF16_ERROR",
    "GPU_QUALIFICATION_MAX_LOGIT_DRIFT",
    "GPU_QUALIFICATION_MAX_MODEL_LEN",
    "GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES",
    "GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND",
    "GPU_QUALIFICATION_MODEL_ID",
    "GPU_QUALIFICATION_MODEL_REVISION",
    "GPU_QUALIFICATION_OFFICIAL_WHEEL_SHA256",
    "GPU_QUALIFICATION_PATCHED_WHEEL_SHA256",
    "GPU_QUALIFICATION_PLAN_RECORD_TYPE",
    "GPU_QUALIFICATION_PUBLICATION_BACKEND",
    "GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256",
    "GPU_QUALIFICATION_REQUEST_PARALLELISM",
    "GPU_QUALIFICATION_REQUIRED_KV_CAPACITY_TOKENS",
    "GPU_QUALIFICATION_SCHEMA_VERSION",
    "GPU_QUALIFICATION_THROUGHPUT_BUCKETS",
    "GPU_QUALIFICATION_THROUGHPUT_SAMPLES_PER_BUCKET",
    "GPU_QUALIFICATION_TERMINAL_RECEIPT_RECORD_TYPE",
    "GPU_QUALIFICATION_VLLM_VERSION",
    "GPUQualificationArtifactPins",
    "GPUQualificationSelection",
    "build_cloud_gpu_evidence",
    "build_gpu_job_result",
    "build_gpu_qualification_evidence",
    "build_gpu_qualification_plan",
    "build_local_preflight_evidence",
    "canonical_gpu_qualification_json",
    "validate_gpu_qualification_evidence_record",
    "validate_gpu_job_result_record",
    "validate_gpu_qualification_plan_record",
    "validate_local_preflight_evidence_record",
    "write_canonical_gpu_qualification_json",
    "write_gpu_qualification_evidence_json",
    "write_gpu_qualification_plan_json",
]
