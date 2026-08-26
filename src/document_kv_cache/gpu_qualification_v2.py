"""Additive v2 records for the patched-FlashInfer GPU qualification runtime.

The v1 protocol remains the authority for every retained attempt and evidence
record.  This module defines a disjoint v2 record family whose runtime closure
adds a directly installed, source-pinned FlashInfer wheel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import math
import re
from types import MappingProxyType
from typing import Any, Final, cast
from pathlib import Path
from urllib.parse import unquote, urlsplit

from document_kv_cache.databricks_resource_ledger import (
    DatabricksLedgerPrefix,
    databricks_ledger_prefix_from_record,
)
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PACKAGE_VERSION,
    FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256,
    FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
    FLASHINFER_PATCHED_MANIFEST_SIZE,
    FLASHINFER_PATCHED_WHEEL_SHA256,
    FLASHINFER_PATCHED_WHEEL_SIZE,
    FLASHINFER_TARGET_PATCHED_SHA256,
)
import document_kv_cache.gpu_qualification as qualification_v1
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    GPU_QUALIFICATION_VLLM_VERSION,
    GPUQualificationArtifactPins,
    build_gpu_qualification_plan,
    canonical_gpu_qualification_json,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE,
    RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE,
    RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION,
    VLLM_PATCHED_MANIFEST_SHA256,
    VLLM_PATCHED_MANIFEST_SIZE,
    VLLM_PATCHED_WHEEL_SHA256,
    VLLM_PATCHED_WHEEL_SIZE,
    VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SIZE,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS,
)


GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_qualification_plan.v2"
)
GPU_QUALIFICATION_V2_JOB_RESULT_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_job_result.v2"
)
GPU_QUALIFICATION_V2_RUNTIME_VERIFICATION_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_runtime_verification.v2"
)
GPU_QUALIFICATION_V2_LOCAL_PREFLIGHT_RECORD_TYPE: Final = (
    "cachet.vllm_0271_local_preflight_evidence.v2"
)
GPU_QUALIFICATION_V2_SCHEMA_VERSION: Final = 2
GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION: Final = "0.2.0"
GPU_QUALIFICATION_V2_ARTIFACT_KEYS: Final = (
    "cachet_source_tree_sha256",
    "input_bundle_sha256",
    "package_wheel_sha256",
    "patched_flashinfer_wheel_sha256",
    "patched_vllm_wheel_sha256",
    "runner_sha256",
    "runtime_closure_manifest_sha256",
    "runtime_lock_sha256",
)
GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS: Final = (
    "canonical_plan_schema",
    "runtime_lock_require_hashes",
    "patched_wheel_record_and_manifest",
    "runtime_artifact_closure",
    "source_runner_input_closure",
    "unit_tests",
    "ruff",
    "mypy",
)
GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT: Final = 198
GPU_QUALIFICATION_V2_WITH_FLASHINFER_DISTRIBUTION_COUNT: Final = 196
GPU_QUALIFICATION_V2_WITH_VLLM_DISTRIBUTION_COUNT: Final = 197
GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION: Final = (
    "tuple[tuple[int, int, array.array[int]]]"
)
GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX: Final = DatabricksLedgerPrefix(
    ledger_id=PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.ledger_id,
    cap_cluster_hours=PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.cap_cluster_hours,
    reservation_count=264,
    submission_receipt_count=126,
    terminal_actual_count=264,
    prefix_sha256=("6314f9ddd5204f5072eac3084269cc854e950b7a0e0adb1572b4a0773071b563"),
)
GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS: Final = 77.30916277777781

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PLAN_KEYS: Final = frozenset(
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
_JOB_RESULT_KEYS: Final = frozenset(
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
        "output_json",
        "plan_sha256",
        "record_type",
        "reservation_attempt_id",
        "retry_count",
        "runtime_verification",
        "schema_version",
        "started_at_utc",
        "status",
        "task_key",
        "torch_cuda_version",
        "vllm_version",
    }
)
_VLLM_PATCH_MEMBER_SHA256: Final = MappingProxyType(
    {
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
)
_GPU_QUALIFICATION_V2_RUNTIME_CLOSURE_TEMPLATE = {
    "base_lock": {
        "bytes": VLLM_RUNTIME_BASE_LOCK_SIZE,
        "distribution_count": VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
        "hash_count": VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
        "sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
    },
    "distribution_counts": {
        "base_lock": VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
        "separately_allowed_cachet": 1,
        "with_flashinfer": GPU_QUALIFICATION_V2_WITH_FLASHINFER_DISTRIBUTION_COUNT,
        "with_vllm": GPU_QUALIFICATION_V2_WITH_VLLM_DISTRIBUTION_COUNT,
    },
    "flashinfer": {
        "manifest_closed_record_sha256": (
            FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
        ),
        "manifest_file_bytes": FLASHINFER_PATCHED_MANIFEST_SIZE,
        "manifest_file_sha256": FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
        "patched_member_sha256": FLASHINFER_TARGET_PATCHED_SHA256,
        "version": FLASHINFER_PACKAGE_VERSION,
        "wheel_bytes": FLASHINFER_PATCHED_WHEEL_SIZE,
        "wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
    },
    "install_order": [
        "runtime_lock_sha256",
        "patched_vllm_wheel_sha256",
        "patched_flashinfer_wheel_sha256",
        "package_wheel_sha256",
    ],
    "manifest": {
        "closed_record_sha256": RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
        "file_bytes": RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE,
        "file_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "record_type": RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE,
        "schema_version": RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION,
    },
    "pep610_direct_url_required": ["flashinfer-python", "vllm"],
    "pip_check_required": True,
    "vllm": {
        "manifest_file_bytes": VLLM_PATCHED_MANIFEST_SIZE,
        "manifest_file_sha256": VLLM_PATCHED_MANIFEST_SHA256,
        "member_sha256": dict(_VLLM_PATCH_MEMBER_SHA256),
        "version": GPU_QUALIFICATION_VLLM_VERSION,
        "wheel_bytes": VLLM_PATCHED_WHEEL_SIZE,
        "wheel_sha256": VLLM_PATCHED_WHEEL_SHA256,
    },
}
_GPU_QUALIFICATION_V2_RUNTIME_CLOSURE_JSON: Final = canonical_gpu_qualification_json(
    _GPU_QUALIFICATION_V2_RUNTIME_CLOSURE_TEMPLATE
)
del _GPU_QUALIFICATION_V2_RUNTIME_CLOSURE_TEMPLATE

_RUNTIME_ATTESTATION_KEYS: Final = frozenset(
    {
        "base_lock_distribution_count",
        "base_lock_hash_count",
        "base_lock_sha256",
        "cachet_package_version",
        "flashinfer_annotation",
        "flashinfer_direct_url",
        "flashinfer_import_ok",
        "closure_bound_flashinfer_manifest_closed_record_sha256",
        "closure_bound_flashinfer_manifest_file_sha256",
        "closure_bound_vllm_manifest_file_sha256",
        "flashinfer_member_sha256",
        "flashinfer_package_version",
        "flashinfer_wheel_sha256",
        "installed_distribution_count",
        "ok",
        "packaged_base_lock_sha256",
        "pip_check_ok",
        "runtime_closure_closed_record_sha256",
        "runtime_closure_file_sha256",
        "unexpected_distributions",
        "vllm_direct_url",
        "vllm_member_sha256",
        "vllm_package_version",
        "vllm_wheel_sha256",
        "with_flashinfer_distribution_count",
        "with_vllm_distribution_count",
    }
)
_RUNTIME_VERIFICATION_KEYS: Final = frozenset(
    {
        "artifact_sha256",
        "attestation",
        "closed_record_sha256",
        "job_id",
        "plan_sha256",
        "record_type",
        "schema_version",
    }
)
_LOCAL_PREFLIGHT_KEYS: Final = frozenset(
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
_LOCAL_PREFLIGHT_CHECK_KEYS: Final = frozenset(
    {"check_id", "evidence_sha256", "status"}
)


@dataclass(frozen=True, slots=True)
class GPUQualificationArtifactPinsV2:
    """Eight immutable artifact identities required by every v2 job."""

    runtime_lock_sha256: str
    patched_vllm_wheel_sha256: str
    patched_flashinfer_wheel_sha256: str
    runtime_closure_manifest_sha256: str
    package_wheel_sha256: str
    cachet_source_tree_sha256: str
    runner_sha256: str
    input_bundle_sha256: str

    def __post_init__(self) -> None:
        for field_name in GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
            _require_sha256(getattr(self, field_name), field_name)
        fixed = {
            "input_bundle_sha256": GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
            "patched_flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
            "patched_vllm_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
            "runtime_closure_manifest_sha256": (RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256),
            "runtime_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        }
        for field_name, expected in fixed.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} differs from the reviewed v2 authority")

    def to_record(self) -> dict[str, str]:
        """Return the exact canonical eight-key pin mapping."""

        return {
            key: cast(str, getattr(self, key))
            for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS
        }

    def v1_projection(self) -> GPUQualificationArtifactPins:
        """Project only fields shared with v1 for behavior-only reuse."""

        return GPUQualificationArtifactPins(
            runtime_lock_sha256=self.runtime_lock_sha256,
            patched_vllm_wheel_sha256=self.patched_vllm_wheel_sha256,
            package_wheel_sha256=self.package_wheel_sha256,
            cachet_source_tree_sha256=self.cachet_source_tree_sha256,
            runner_sha256=self.runner_sha256,
            input_bundle_sha256=self.input_bundle_sha256,
        )


def gpu_qualification_v2_runtime_closure() -> dict[str, Any]:
    """Return a fresh copy of the immutable reviewed runtime closure."""

    value = json.loads(_GPU_QUALIFICATION_V2_RUNTIME_CLOSURE_JSON)
    if not isinstance(value, dict):
        raise RuntimeError("internal GPU qualification v2 runtime closure is invalid")
    return value


def build_gpu_qualification_plan_v2(
    *,
    campaign_id: str,
    campaign_record_sha256: str,
    campaign_ledger_id: str,
    campaign_ledger_path_sha256: str,
    campaign_ledger_prefix: DatabricksLedgerPrefix,
    campaign_opening_terminal_gpu_hours: float,
    artifact_pins: GPUQualificationArtifactPinsV2,
) -> dict[str, Any]:
    """Build the additive v2 plan without changing the v1 builder."""

    if not isinstance(artifact_pins, GPUQualificationArtifactPinsV2):
        raise TypeError("artifact_pins must be GPUQualificationArtifactPinsV2")
    if not isinstance(campaign_ledger_prefix, DatabricksLedgerPrefix):
        raise TypeError("campaign_ledger_prefix must be a DatabricksLedgerPrefix")
    if (
        isinstance(campaign_opening_terminal_gpu_hours, bool)
        or not isinstance(campaign_opening_terminal_gpu_hours, (int, float))
        or not math.isfinite(float(campaign_opening_terminal_gpu_hours))
        or float(campaign_opening_terminal_gpu_hours) < 0
    ):
        raise ValueError(
            "campaign_opening_terminal_gpu_hours must be finite/nonnegative"
        )
    if campaign_ledger_prefix != GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX:
        raise ValueError("v2 campaign ledger prefix differs from reviewed authority")
    if float(campaign_opening_terminal_gpu_hours) != (
        GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS
    ):
        raise ValueError("v2 campaign opening balance differs from reviewed authority")
    record: dict[str, Any] = build_gpu_qualification_plan(
        campaign_id=campaign_id,
        campaign_record_sha256=campaign_record_sha256,
        campaign_ledger_id=campaign_ledger_id,
        campaign_ledger_path_sha256=campaign_ledger_path_sha256,
        campaign_ledger_prefix=PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
        ),
        artifact_pins=artifact_pins.v1_projection(),
    )
    record["campaign_ledger_prefix"] = campaign_ledger_prefix.to_record()
    record["campaign_opening_terminal_gpu_hours"] = float(
        campaign_opening_terminal_gpu_hours
    )
    record["record_type"] = GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE
    record["schema_version"] = GPU_QUALIFICATION_V2_SCHEMA_VERSION
    local_preflight = _mapping_copy(record["local_preflight"], "local_preflight")
    local_preflight["check_ids"] = list(GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS)
    record["local_preflight"] = local_preflight
    runtime_contract = _mapping_copy(record["runtime_contract"], "runtime_contract")
    runtime_contract["artifact_sha256"] = artifact_pins.to_record()
    runtime_contract["runtime_artifact_closure"] = (
        gpu_qualification_v2_runtime_closure()
    )
    record["runtime_contract"] = runtime_contract
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def build_local_preflight_evidence_v2(
    *,
    plan_sha256: str,
    completed_at_utc: str,
    check_evidence_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Seal the exact eight-check local-only v2 preflight evidence."""

    plan_digest = _required_sha256(plan_sha256, "plan_sha256")
    qualification_v1._parse_utc_timestamp(  # noqa: SLF001
        completed_at_utc,
        field_name="completed_at_utc",
    )
    if tuple(check_evidence_sha256) != GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS:
        raise ValueError(
            "v2 local preflight evidence lacks canonical eight-check coverage"
        )
    checks: list[dict[str, str]] = []
    for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS:
        checks.append(
            {
                "check_id": check_id,
                "evidence_sha256": _required_sha256(
                    check_evidence_sha256[check_id],
                    f"{check_id}.evidence_sha256",
                ),
                "status": "passed",
            }
        )
    record: dict[str, Any] = {
        "checks": checks,
        "closed_record_sha256": "",
        "completed_at_utc": completed_at_utc,
        "plan_sha256": plan_digest,
        "record_type": GPU_QUALIFICATION_V2_LOCAL_PREFLIGHT_RECORD_TYPE,
        "schema_version": GPU_QUALIFICATION_V2_SCHEMA_VERSION,
        "scope": "local_preflight_only_no_cloud_success_credit_v2",
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def validate_local_preflight_evidence_v2_record(
    record: Mapping[str, Any],
    *,
    plan_sha256: str,
) -> datetime:
    """Validate the sealed v2 preflight and return its completion time."""

    normalized = _mapping_copy(record, "v2 local preflight evidence")
    _require_exact_keys(
        normalized,
        _LOCAL_PREFLIGHT_KEYS,
        "v2 local preflight evidence",
    )
    _require_closed_record_digest(normalized, "v2 local preflight evidence")
    if normalized.get("record_type") != (
        GPU_QUALIFICATION_V2_LOCAL_PREFLIGHT_RECORD_TYPE
    ):
        raise ValueError("unexpected v2 local preflight record_type")
    if (
        type(normalized.get("schema_version")) is not int
        or normalized.get("schema_version") != GPU_QUALIFICATION_V2_SCHEMA_VERSION
    ):
        raise ValueError("unexpected v2 local preflight schema_version")
    if normalized.get("scope") != ("local_preflight_only_no_cloud_success_credit_v2"):
        raise ValueError("v2 local preflight scope cannot grant cloud credit")
    if normalized.get("plan_sha256") != _required_sha256(plan_sha256, "plan_sha256"):
        raise ValueError("v2 local preflight plan SHA-256 differs")
    checks = normalized.get("checks")
    if not isinstance(checks, list) or len(checks) != len(
        GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS
    ):
        raise ValueError("v2 local preflight lacks exact eight-check coverage")
    for expected_id, raw_check in zip(
        GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS, checks, strict=True
    ):
        check = _mapping_copy(raw_check, f"v2 local check {expected_id}")
        _require_exact_keys(
            check,
            _LOCAL_PREFLIGHT_CHECK_KEYS,
            f"v2 local check {expected_id}",
        )
        if check.get("check_id") != expected_id or check.get("status") != "passed":
            raise ValueError(f"v2 local check {expected_id} is not canonical")
        _required_sha256(
            check.get("evidence_sha256"),
            f"{expected_id}.evidence_sha256",
        )
    return qualification_v1._parse_utc_timestamp(  # noqa: SLF001
        normalized.get("completed_at_utc"),
        field_name="completed_at_utc",
    )


def validate_gpu_qualification_plan_v2_record(
    record: Mapping[str, Any],
    *,
    expected_campaign_id: str,
    expected_artifact_pins: GPUQualificationArtifactPinsV2,
) -> None:
    """Require the exact frozen v2 plan and reject every v1 record."""

    normalized = _mapping_copy(record, "GPU qualification v2 plan")
    _require_exact_keys(normalized, _PLAN_KEYS, "GPU qualification v2 plan")
    _require_closed_record_digest(normalized, "GPU qualification v2 plan")
    if normalized.get("record_type") != GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE:
        raise ValueError("unexpected GPU qualification v2 plan record_type")
    if (
        type(normalized.get("schema_version")) is not int
        or normalized.get("schema_version") != GPU_QUALIFICATION_V2_SCHEMA_VERSION
    ):
        raise ValueError("unexpected GPU qualification v2 plan schema_version")
    if pins_from_gpu_qualification_plan_v2(normalized) != expected_artifact_pins:
        raise ValueError("GPU qualification v2 plan artifact pins differ")
    expected = build_gpu_qualification_plan_v2(
        campaign_id=expected_campaign_id,
        campaign_record_sha256=_required_sha256(
            normalized.get("campaign_record_sha256"), "campaign_record_sha256"
        ),
        campaign_ledger_id=_required_string(
            normalized.get("campaign_ledger_id"), "campaign_ledger_id"
        ),
        campaign_ledger_path_sha256=_required_sha256(
            normalized.get("campaign_ledger_path_sha256"),
            "campaign_ledger_path_sha256",
        ),
        campaign_ledger_prefix=databricks_ledger_prefix_from_record(
            _mapping_copy(
                normalized.get("campaign_ledger_prefix"),
                "campaign_ledger_prefix",
            )
        ),
        campaign_opening_terminal_gpu_hours=cast(
            float, normalized.get("campaign_opening_terminal_gpu_hours")
        ),
        artifact_pins=expected_artifact_pins,
    )
    if canonical_gpu_qualification_json(normalized) != (
        canonical_gpu_qualification_json(expected)
    ):
        raise ValueError("GPU qualification v2 plan differs from the frozen plan")


def build_gpu_runtime_verification_v2(
    *,
    plan_sha256: str,
    job_id: str,
    artifact_sha256: Mapping[str, str],
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the final runtime verifier output for one exact v2 job."""

    plan_digest = _required_sha256(plan_sha256, "plan_sha256")
    normalized_job_id = _required_string(job_id, "job_id")
    if tuple(artifact_sha256) != GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        raise ValueError("runtime verification lacks exact eight-key artifacts")
    pins = {
        key: _required_sha256(artifact_sha256[key], f"artifact_sha256.{key}")
        for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS
    }
    validate_gpu_qualification_v2_runtime_attestation(attestation)
    record: dict[str, Any] = {
        "artifact_sha256": pins,
        "attestation": dict(attestation),
        "closed_record_sha256": "",
        "job_id": normalized_job_id,
        "plan_sha256": plan_digest,
        "record_type": GPU_QUALIFICATION_V2_RUNTIME_VERIFICATION_RECORD_TYPE,
        "schema_version": GPU_QUALIFICATION_V2_SCHEMA_VERSION,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def validate_gpu_runtime_verification_v2_record(
    record: Mapping[str, Any],
    *,
    plan_record: Mapping[str, Any],
    expected_job_id: str,
    expected_artifact_pins: GPUQualificationArtifactPinsV2,
) -> None:
    """Validate a sealed verifier record bound to plan, job, and all pins."""

    _validate_bound_plan_v2(plan_record, expected_artifact_pins)
    normalized = _mapping_copy(record, "GPU runtime verification v2")
    _require_exact_keys(
        normalized, _RUNTIME_VERIFICATION_KEYS, "GPU runtime verification v2"
    )
    _require_closed_record_digest(normalized, "GPU runtime verification v2")
    if normalized.get("record_type") != (
        GPU_QUALIFICATION_V2_RUNTIME_VERIFICATION_RECORD_TYPE
    ):
        raise ValueError("unexpected GPU runtime verification v2 record_type")
    if (
        type(normalized.get("schema_version")) is not int
        or normalized.get("schema_version") != GPU_QUALIFICATION_V2_SCHEMA_VERSION
    ):
        raise ValueError("unexpected GPU runtime verification v2 schema_version")
    if normalized.get("plan_sha256") != plan_record.get("closed_record_sha256"):
        raise ValueError("GPU runtime verification v2 plan SHA-256 differs")
    if normalized.get("job_id") != _required_string(expected_job_id, "job_id"):
        raise ValueError("GPU runtime verification v2 job ID differs")
    artifact_sha256 = _mapping_copy(
        normalized.get("artifact_sha256"), "runtime artifact_sha256"
    )
    if (
        tuple(artifact_sha256) != GPU_QUALIFICATION_V2_ARTIFACT_KEYS
        or artifact_sha256 != expected_artifact_pins.to_record()
    ):
        raise ValueError("GPU runtime verification v2 artifact pins differ")
    validate_gpu_qualification_v2_runtime_attestation(
        _mapping_copy(normalized.get("attestation"), "runtime attestation")
    )


def build_gpu_job_result_v2(
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
    runtime_verification: Mapping[str, Any],
    measurements: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one v2 measurement-only job result."""

    pins = pins_from_gpu_qualification_plan_v2(plan_record)
    _validate_bound_plan_v2(plan_record, pins)
    if dict(observed_artifact_sha256) != pins.to_record():
        raise ValueError("observed v2 artifact SHA-256 mapping differs from the plan")
    validate_gpu_runtime_verification_v2_record(
        runtime_verification,
        plan_record=plan_record,
        expected_job_id=job_id,
        expected_artifact_pins=pins,
    )
    record: dict[str, Any] = qualification_v1.build_gpu_job_result(
        plan_record=plan_record,
        job_id=job_id,
        reservation_attempt_id=reservation_attempt_id,
        task_key=task_key,
        output_json=output_json,
        cloud_run_id=cloud_run_id,
        cloud_cluster_id=cloud_cluster_id,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        nvidia_driver_version=nvidia_driver_version,
        observed_gpu=observed_gpu,
        observed_gpu_compute_capability=observed_gpu_compute_capability,
        observed_vllm_version=observed_vllm_version,
        observed_torch_cuda_version=observed_torch_cuda_version,
        observed_artifact_sha256=pins.v1_projection().to_record(),
        measurements=measurements,
    )
    record["artifact_sha256"] = pins.to_record()
    record["record_type"] = GPU_QUALIFICATION_V2_JOB_RESULT_RECORD_TYPE
    record["runtime_verification"] = dict(runtime_verification)
    record["schema_version"] = GPU_QUALIFICATION_V2_SCHEMA_VERSION
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def validate_gpu_job_result_v2_record(
    record: Mapping[str, Any],
    *,
    plan_record: Mapping[str, Any],
    expected_artifact_pins: GPUQualificationArtifactPinsV2,
) -> None:
    """Validate a v2 result while reusing only v1 sentinel behavior checks."""

    _validate_bound_plan_v2(plan_record, expected_artifact_pins)
    normalized = _mapping_copy(record, "GPU job result v2")
    _require_exact_keys(normalized, _JOB_RESULT_KEYS, "GPU job result v2")
    _require_closed_record_digest(normalized, "GPU job result v2")
    if normalized.get("record_type") != GPU_QUALIFICATION_V2_JOB_RESULT_RECORD_TYPE:
        raise ValueError("unexpected GPU job result v2 record_type")
    if (
        type(normalized.get("schema_version")) is not int
        or normalized.get("schema_version") != GPU_QUALIFICATION_V2_SCHEMA_VERSION
    ):
        raise ValueError("unexpected GPU job result v2 schema_version")
    artifact_sha256 = _mapping_copy(
        normalized.get("artifact_sha256"), "job artifact_sha256"
    )
    if (
        tuple(artifact_sha256) != GPU_QUALIFICATION_V2_ARTIFACT_KEYS
        or artifact_sha256 != expected_artifact_pins.to_record()
    ):
        raise ValueError("GPU job result v2 artifact hashes differ")
    if pins_from_gpu_qualification_plan_v2(plan_record) != expected_artifact_pins:
        raise ValueError("GPU job result v2 expected pins differ from the plan")
    job_id = _required_string(normalized.get("job_id"), "job_id")
    runtime_verification = _mapping_copy(
        normalized.get("runtime_verification"), "runtime_verification"
    )
    validate_gpu_runtime_verification_v2_record(
        runtime_verification,
        plan_record=plan_record,
        expected_job_id=job_id,
        expected_artifact_pins=expected_artifact_pins,
    )
    projected = dict(normalized)
    projected.pop("runtime_verification")
    projected["artifact_sha256"] = expected_artifact_pins.v1_projection().to_record()
    projected["record_type"] = qualification_v1.GPU_QUALIFICATION_JOB_RESULT_RECORD_TYPE
    projected["schema_version"] = qualification_v1.GPU_QUALIFICATION_SCHEMA_VERSION
    measurements = _mapping_copy(projected.get("measurements"), "measurements")
    job = qualification_v1._plan_job(  # noqa: SLF001
        plan_record, job_id
    )
    projected["closed_record_sha256"] = qualification_v1._closed_record_sha256(  # noqa: SLF001
        projected
    )
    qualification_v1._validate_job_result_common(  # noqa: SLF001
        projected,
        planned_job=job,
        expected_artifact_pins=expected_artifact_pins.v1_projection(),
        expected_plan_sha256=str(plan_record.get("closed_record_sha256")),
    )
    sentinel = job.get("sentinel")
    if sentinel == "forced_triton_runtime_handoff":
        qualification_v1._validate_runtime_handoff_measurements_with_attestation(  # noqa: SLF001
            measurements,
            hardware_id=str(job["hardware_id"]),
            attestation_validator=(validate_gpu_qualification_v2_runtime_attestation),
        )
        measured_attestation = _mapping_copy(
            measurements.get("runtime_lock_attestation"),
            "measurements.runtime_lock_attestation",
        )
        verified_attestation = _mapping_copy(
            runtime_verification.get("attestation"),
            "runtime_verification.attestation",
        )
        if canonical_gpu_qualification_json(measured_attestation) != (
            canonical_gpu_qualification_json(verified_attestation)
        ):
            raise ValueError("v2 runtime attestations differ within the job result")
    elif sentinel == "packed_page_raw_byte_roundtrip":
        qualification_v1._validate_packed_roundtrip_measurements(  # noqa: SLF001
            measurements
        )
    elif sentinel == "matched_token_contract_and_determinism":
        qualification_v1._validate_token_determinism_measurements(  # noqa: SLF001
            measurements,
            expected_input_bundle_sha256=expected_artifact_pins.input_bundle_sha256,
        )
    elif sentinel == "l4_32k_c4_gmu_sweep":
        requirements = cast(Mapping[str, Any], job["requirements"])
        qualification_v1._validate_gmu_measurements(  # noqa: SLF001
            measurements,
            expected_gmu=cast(float, requirements["gpu_memory_utilization"]),
        )
    elif sentinel == "a10g_16k_c4_capacity":
        qualification_v1._validate_a10g_capacity_measurements(  # noqa: SLF001
            measurements
        )
    elif sentinel == "generation_throughput_with_writes":
        qualification_v1._validate_throughput_measurements(  # noqa: SLF001
            measurements, hardware_id=str(job["hardware_id"])
        )
    elif sentinel == "auto_backend_diagnostic":
        qualification_v1._validate_auto_backend_measurements(  # noqa: SLF001
            measurements
        )
    else:
        raise ValueError(f"unsupported v2 sentinel: {sentinel!r}")


def pins_from_gpu_qualification_plan_v2(
    plan_record: Mapping[str, Any],
) -> GPUQualificationArtifactPinsV2:
    """Extract the exact v2 pin type from a v2 plan."""

    runtime_contract = _mapping_copy(
        plan_record.get("runtime_contract"), "runtime_contract"
    )
    raw = _mapping_copy(runtime_contract.get("artifact_sha256"), "artifact_sha256")
    if tuple(raw) != GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        raise ValueError(
            "v2 plan artifact pins lack exact canonical eight-key coverage"
        )
    return GPUQualificationArtifactPinsV2(
        runtime_lock_sha256=_required_sha256(
            raw.get("runtime_lock_sha256"), "runtime_lock_sha256"
        ),
        patched_vllm_wheel_sha256=_required_sha256(
            raw.get("patched_vllm_wheel_sha256"), "patched_vllm_wheel_sha256"
        ),
        patched_flashinfer_wheel_sha256=_required_sha256(
            raw.get("patched_flashinfer_wheel_sha256"),
            "patched_flashinfer_wheel_sha256",
        ),
        runtime_closure_manifest_sha256=_required_sha256(
            raw.get("runtime_closure_manifest_sha256"),
            "runtime_closure_manifest_sha256",
        ),
        package_wheel_sha256=_required_sha256(
            raw.get("package_wheel_sha256"), "package_wheel_sha256"
        ),
        cachet_source_tree_sha256=_required_sha256(
            raw.get("cachet_source_tree_sha256"), "cachet_source_tree_sha256"
        ),
        runner_sha256=_required_sha256(raw.get("runner_sha256"), "runner_sha256"),
        input_bundle_sha256=_required_sha256(
            raw.get("input_bundle_sha256"), "input_bundle_sha256"
        ),
    )


def validate_gpu_qualification_v2_runtime_attestation(
    value: Mapping[str, Any],
) -> None:
    """Validate the final isolated-runtime verifier's exact result."""

    normalized = _mapping_copy(value, "GPU qualification v2 runtime attestation")
    _require_exact_keys(
        normalized, _RUNTIME_ATTESTATION_KEYS, "GPU qualification v2 attestation"
    )
    expected: dict[str, Any] = {
        "base_lock_distribution_count": VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
        "base_lock_hash_count": VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
        "base_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "cachet_package_version": GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION,
        "flashinfer_annotation": GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION,
        "flashinfer_import_ok": True,
        "closure_bound_flashinfer_manifest_closed_record_sha256": (
            FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
        ),
        "closure_bound_flashinfer_manifest_file_sha256": (
            FLASHINFER_PATCHED_MANIFEST_FILE_SHA256
        ),
        "closure_bound_vllm_manifest_file_sha256": VLLM_PATCHED_MANIFEST_SHA256,
        "flashinfer_member_sha256": FLASHINFER_TARGET_PATCHED_SHA256,
        "flashinfer_package_version": FLASHINFER_PACKAGE_VERSION,
        "flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
        "installed_distribution_count": (
            GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT
        ),
        "ok": True,
        "packaged_base_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "pip_check_ok": True,
        "runtime_closure_closed_record_sha256": (
            RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
        ),
        "runtime_closure_file_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "unexpected_distributions": [],
        "vllm_member_sha256": dict(_VLLM_PATCH_MEMBER_SHA256),
        "vllm_package_version": GPU_QUALIFICATION_VLLM_VERSION,
        "vllm_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        "with_flashinfer_distribution_count": (
            GPU_QUALIFICATION_V2_WITH_FLASHINFER_DISTRIBUTION_COUNT
        ),
        "with_vllm_distribution_count": (
            GPU_QUALIFICATION_V2_WITH_VLLM_DISTRIBUTION_COUNT
        ),
    }
    for field_name, expected_value in expected.items():
        if type(normalized.get(field_name)) is not type(expected_value) or (
            normalized.get(field_name) != expected_value
        ):
            raise ValueError(f"GPU qualification v2 attestation {field_name} differs")
    for field_name in (
        "flashinfer_direct_url",
        "vllm_direct_url",
    ):
        value_text = normalized.get(field_name)
        if not isinstance(value_text, str) or not value_text:
            raise ValueError(f"GPU qualification v2 attestation {field_name} is empty")
    for field_name in ("flashinfer_direct_url", "vllm_direct_url"):
        if not _is_canonical_local_file_uri(cast(str, normalized[field_name])):
            raise ValueError(
                f"GPU qualification v2 attestation {field_name} is not canonical"
            )


def _validate_bound_plan_v2(
    plan_record: Mapping[str, Any],
    expected_artifact_pins: GPUQualificationArtifactPinsV2,
) -> None:
    validate_gpu_qualification_plan_v2_record(
        plan_record,
        expected_campaign_id=_required_string(
            plan_record.get("campaign_id"), "campaign_id"
        ),
        expected_artifact_pins=expected_artifact_pins,
    )


def _is_canonical_local_file_uri(value: str) -> bool:
    parsed = urlsplit(value)
    decoded_path = unquote(parsed.path)
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
        or decoded_path.startswith("//")
        or any(
            ord(character) < 32 or ord(character) == 127 for character in decoded_path
        )
        or any(part in {".", ".."} for part in decoded_path.split("/"))
    ):
        return False
    try:
        return Path(decoded_path).as_uri() == value
    except ValueError:
        return False


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload["closed_record_sha256"] = ""
    return sha256(canonical_gpu_qualification_json(payload).encode("utf-8")).hexdigest()


def _require_closed_record_digest(record: Mapping[str, Any], label: str) -> None:
    observed = _required_sha256(
        record.get("closed_record_sha256"), "closed_record_sha256"
    )
    if observed != _closed_record_sha256(record):
        raise ValueError(f"{label} closed_record_sha256 differs")


def _require_sha256(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be lowercase SHA-256")


def _required_sha256(value: Any, field_name: str) -> str:
    _require_sha256(value, field_name)
    return cast(str, value)


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


def _mapping_copy(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{label} does not use the closed schema")


__all__ = [
    "GPU_QUALIFICATION_V2_ARTIFACT_KEYS",
    "GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION",
    "GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION",
    "GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT",
    "GPU_QUALIFICATION_V2_JOB_RESULT_RECORD_TYPE",
    "GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS",
    "GPU_QUALIFICATION_V2_LOCAL_PREFLIGHT_RECORD_TYPE",
    "GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX",
    "GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS",
    "GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE",
    "GPU_QUALIFICATION_V2_RUNTIME_VERIFICATION_RECORD_TYPE",
    "GPU_QUALIFICATION_V2_SCHEMA_VERSION",
    "GPU_QUALIFICATION_V2_WITH_FLASHINFER_DISTRIBUTION_COUNT",
    "GPU_QUALIFICATION_V2_WITH_VLLM_DISTRIBUTION_COUNT",
    "GPUQualificationArtifactPinsV2",
    "build_gpu_job_result_v2",
    "build_gpu_qualification_plan_v2",
    "build_gpu_runtime_verification_v2",
    "build_local_preflight_evidence_v2",
    "gpu_qualification_v2_runtime_closure",
    "pins_from_gpu_qualification_plan_v2",
    "validate_gpu_job_result_v2_record",
    "validate_gpu_qualification_plan_v2_record",
    "validate_gpu_qualification_v2_runtime_attestation",
    "validate_gpu_runtime_verification_v2_record",
    "validate_local_preflight_evidence_v2_record",
]
