"""Databricks execution boundary for the closed vLLM GPU qualification plan.

The plan module deliberately has no cloud side effects.  This module turns one
validated plan into one independent ``runs/submit`` payload per sentinel and
provides the GPU-side result sealing boundary.  It still does not upload or
submit anything.

Every task is attempt-zero-only, binds all immutable artifacts by URI and
SHA-256, and writes to a plan/job-specific path with exclusive creation.  A
sentinel implementation returns measurements only through an in-process
callable; the executor validates the complete sentinel-specific schema before
publishing a canonical job-result record.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import shutil
import stat
import subprocess
import urllib.request
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol, cast
from urllib.parse import unquote, urlsplit

from document_kv_cache._hardware_targets import (
    databricks_node_type_for_hardware_target,
)
from document_kv_cache.databricks_job import (
    DEFAULT_DATABRICKS_RUN_TIMEOUT_SECONDS,
    DEFAULT_DATABRICKS_SPARK_VERSION,
    DatabricksSingleNodeGPUClusterConfig,
    build_single_node_gpu_cluster,
)
from document_kv_cache.databricks_resource_ledger import (
    MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS,
    MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    DatabricksBatchReservationAuthorization,
    DatabricksClusterHourLedger,
    DatabricksClusterHourReservation,
    DatabricksClusterHourTerminalActual,
    DatabricksLedgerPrefix,
    DatabricksRunAttemptReservationRequest,
    canonical_databricks_submit_payload_snapshot,
    databricks_ledger_prefix_at_counts,
    databricks_ledger_prefix_from_record,
    databricks_ledger_path_sha256,
    read_databricks_cluster_hour_ledger_json,
    replay_databricks_run_attempt_batch_authorization_json,
    record_databricks_run_submission_receipt_json,
    record_databricks_verified_run_terminal_actual_json,
    require_databricks_ledger_prefix,
    require_databricks_publication_batch_admission,
    reserve_databricks_run_attempt_batch_authorized_json,
)
from document_kv_cache.databricks_runs import (
    DatabricksURLOpener,
    DatabricksWorkspaceConfig,
    bind_databricks_run_idempotency_token,
    get_databricks_run,
    require_databricks_run_idempotency_token,
    resume_pre_reserved_databricks_run,
    submit_pre_reserved_databricks_run,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_SCHEMA_VERSION,
    GPU_QUALIFICATION_TERMINAL_RECEIPT_RECORD_TYPE,
    GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE,
    GPU_QUALIFICATION_GENERATION_HARDWARE_ID,
    GPU_QUALIFICATION_MAX_CLOUD_JOBS,
    GPU_QUALIFICATION_PLAN_RECORD_TYPE,
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPUQualificationArtifactPins,
    GPUQualificationSelection,
    _build_governed_cloud_gpu_evidence,
    _build_governed_gpu_qualification_evidence,
    build_gpu_job_result,
    canonical_gpu_qualification_json,
    validate_gpu_job_result_record,
    validate_gpu_qualification_evidence_record,
    validate_gpu_qualification_plan_record,
    validate_local_preflight_evidence_record,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS,
    PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS,
)
from document_kv_cache.serving_env import VLLM_RUNTIME_LOCK_SHA256


GPU_QUALIFICATION_DATABRICKS_PURPOSE: Final = "cachet-vllm-0271-gpu-qualification"
GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS: Final = (
    DEFAULT_DATABRICKS_RUN_TIMEOUT_SECONDS
)
GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES: Final = 9_500
GPU_QUALIFICATION_DATABRICKS_DATA_SECURITY_MODE: Final = "NONE"
GPU_QUALIFICATION_ARTIFACT_KEYS: Final = (
    "cachet_source_tree_sha256",
    "input_bundle_sha256",
    "package_wheel_sha256",
    "patched_vllm_wheel_sha256",
    "runner_sha256",
    "runtime_lock_sha256",
)
GPU_QUALIFICATION_OUTPUT_FILENAME: Final = "gpu-job-result.json"
GPU_QUALIFICATION_SUBMIT_RECEIPT_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_submit_receipt.v1"
)
GPU_QUALIFICATION_SUBMISSION_REJECTION_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_qualification_submission_rejection.v1"
)
GPU_QUALIFICATION_LOCAL_WORK_ROOT: Final = "/local_disk0/cachet-vllm-0271-qualification"
_QUALIFICATION_PHASE_LEASE_FILENAME: Final = "phase-lease.json"
_QUALIFICATION_BATCH_MARKER_FILENAME: Final = "batch-reserved.json"
_QUALIFICATION_PHASE_LEASE_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_qualification_phase_lease.v1"
)
_QUALIFICATION_BATCH_MARKER_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_qualification_batch_reserved.v1"
)
_QUALIFICATION_PREFLIGHT_PATH_DOMAIN: Final = (
    "cachet.vllm_0271_gpu_qualification_preflight_path.v1"
)
_QUALIFICATION_PREFLIGHT_BINDING_KEYS: Final = frozenset(
    {
        "completed_at_utc",
        "file_sha256",
        "path_sha256",
        "record_sha256",
    }
)
_QUALIFICATION_PLAN_PARAMETER_OPTION: Final = "--plan-record-zlib-base64"
_QUALIFICATION_PLAN_ZLIB_LEVEL: Final = 9
_QUALIFICATION_PLAN_MAX_CANONICAL_BYTES: Final = 64 * 1024
_QUALIFICATION_PLAN_MAX_ENCODED_CHARS: Final = (
    GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES
)
_QUALIFICATION_SUBMISSION_REJECTION_KEYS: Final = frozenset(
    {
        "attempt_ids",
        "batch_marker_file_sha256",
        "closed_record_sha256",
        "failed_before_run_creation",
        "first_post_intent_file_sha256",
        "http_status",
        "observed_parameters_json_bytes",
        "plan_sha256",
        "reconciled_actual_gpu_seconds_per_attempt",
        "record_type",
        "rejected_at_utc",
        "remote_active_runs_observed",
        "schema_version",
        "server_parameters_json_limit_bytes",
        "server_reason",
        "submit_payloads_file_sha256",
    }
)

_SUBMIT_RECEIPT_KEYS: Final = frozenset(
    {
        "authorization_scope",
        "closed_record_sha256",
        "cloud_run_id",
        "job_id",
        "ledger_id",
        "output_json",
        "plan_sha256",
        "phase_batch_record_sha256",
        "record_type",
        "reservation_attempt_id",
        "schema_version",
        "submit_payload_sha256",
        "submit_response_sha256",
        "submitted_at_utc",
        "task_key",
    }
)

_INPUT_PROVENANCE_FILENAME: Final = "main-latency-inputs.provenance.json"
_INPUT_PROVENANCE_FIELDS: Final = frozenset(
    {
        "bundle_sha256",
        "closed_record_sha256",
        "outputs",
        "outputs_sha256",
        "protocol",
        "record_type",
        "schema_version",
        "sources",
        "sources_sha256",
    }
)
_INPUT_OUTPUT_FIELDS: Final = frozenset(
    {
        "byte_count",
        "dataset",
        "input_tokens_target",
        "jsonl_sha256",
        "record_count",
        "records",
        "records_sha256",
        "relative_path",
        "segment_count",
    }
)
_INPUT_DATASETS: Final = ("biography", "hotpotqa", "musique", "niah")
_INPUT_TARGET_SEGMENT_COUNTS: Final = ((8192, 4), (16384, 8), (32768, 16))
_INPUT_EXAMPLES_PER_DATASET: Final = 32
_INPUT_PROTOCOL: Final = {
    "datasets": list(_INPUT_DATASETS),
    "prompt_contract": {
        "prompt_template_version": "v2-final-answer",
        "system_prompt_position": "start",
    },
    "selection": {
        "domain": "cachet.main_latency.content_hash_selection.v1",
        "identity_reused_across_targets": True,
        "ordering": "sha256_domain_dataset_identity_and_source_record",
        "selected_examples_per_dataset": _INPUT_EXAMPLES_PER_DATASET,
    },
    "targets": [
        {"input_tokens_target": target, "segment_count": segment_count}
        for target, segment_count in _INPUT_TARGET_SEGMENT_COUNTS
    ],
    "tokenizer": {
        "add_special_tokens": False,
        "tokenizer_id": "Qwen/Qwen3-4B-Instruct-2507",
        "tokenizer_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    },
    "transformation": {
        "document_context_tiling": "lossless_contiguous_unicode_codepoints",
        "id": "cachet.main_latency.lossless_context_tiling.v1",
        "padding": "balanced_exact_token_count_irrelevant_units",
        "vanilla_composition": (
            "concatenated_independent_segment_token_ids_equal_logical_cache_prefix"
        ),
    },
}

GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT: Final = """from __future__ import annotations

import hashlib
import os
import subprocess
import sys


_KEYS = {
    "cachet_source_tree_sha256",
    "input_bundle_sha256",
    "package_wheel_sha256",
    "patched_vllm_wheel_sha256",
    "runner_sha256",
    "runtime_lock_sha256",
}


def _cluster_path(value: str) -> str:
    if value.startswith("dbfs:/"):
        return "/dbfs/" + value.removeprefix("dbfs:/").lstrip("/")
    if value.startswith("file://"):
        from urllib.parse import unquote, urlsplit

        parsed = urlsplit(value)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("unsupported file URI authority")
        return unquote(parsed.path)
    return value


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap(argv: list[str]) -> list[str]:
    package_uri = None
    pins = {}
    index = 0
    while index < len(argv):
        option = argv[index]
        if option in {"--package-wheel-uri", "--artifact-sha256"}:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} requires a value")
            value = argv[index + 1]
            if option == "--package-wheel-uri":
                if package_uri is not None:
                    raise ValueError("duplicate --package-wheel-uri")
                package_uri = value
            else:
                key, separator, digest = value.partition("=")
                if not separator or key in pins:
                    raise ValueError("invalid or duplicate --artifact-sha256")
                pins[key] = digest
            index += 2
            continue
        index += 1
    if package_uri is None or set(pins) != _KEYS:
        raise ValueError("bootstrap requires the closed artifact pin set")
    for key, digest in pins.items():
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid SHA-256 for {key}")
    runner_path = os.path.realpath(__file__)
    if _sha256(runner_path) != pins["runner_sha256"]:
        raise ValueError("GPU qualification bootstrap runner SHA-256 mismatch")
    package_path = _cluster_path(package_uri)
    if _sha256(package_path) != pins["package_wheel_sha256"]:
        raise ValueError("Cachet package wheel SHA-256 mismatch")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-deps",
            "--force-reinstall",
            package_path,
        ],
        check=True,
    )
    return argv


if __name__ == "__main__":
    remaining = _bootstrap(sys.argv[1:])
    from document_kv_cache.gpu_qualification_databricks import main

    raise SystemExit(main(remaining))
"""
GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256: Final = sha256(
    GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT.encode("utf-8")
).hexdigest()

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_DATABRICKS_RUN_ID_TEMPLATE = "{{job.run_id}}"
_LAUNCH_AUTHORIZATION_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class GPUQualificationLaunchAuthorization:
    """Non-record capability issued only by live collection or durable replay."""

    selection: GPUQualificationSelection
    plan_sha256: str
    evidence_closed_record_sha256: str
    evidence_file_sha256: str
    ledger_id: str
    ledger_path_sha256: str
    predecessor_prefix: DatabricksLedgerPrefix
    producer_batch_prefix: DatabricksLedgerPrefix
    ledger_prefix: DatabricksLedgerPrefix
    causal_closure_sha256: str

    def __init__(
        self,
        *,
        selection: GPUQualificationSelection,
        plan_sha256: str,
        evidence_closed_record_sha256: str,
        evidence_file_sha256: str,
        ledger_id: str,
        ledger_path_sha256: str,
        predecessor_prefix: DatabricksLedgerPrefix,
        producer_batch_prefix: DatabricksLedgerPrefix,
        ledger_prefix: DatabricksLedgerPrefix,
        causal_closure_sha256: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _LAUNCH_AUTHORIZATION_ISSUER:
            raise TypeError(
                "GPU qualification launch authority must come from live collection "
                "or its durable replay boundary"
            )
        if not isinstance(selection, GPUQualificationSelection):
            raise TypeError("selection must be GPUQualificationSelection")
        object.__setattr__(self, "selection", selection)
        object.__setattr__(
            self, "plan_sha256", _required_sha256(plan_sha256, "plan_sha256")
        )
        object.__setattr__(
            self,
            "evidence_closed_record_sha256",
            _required_sha256(
                evidence_closed_record_sha256,
                "evidence_closed_record_sha256",
            ),
        )
        object.__setattr__(
            self,
            "evidence_file_sha256",
            _required_sha256(evidence_file_sha256, "evidence_file_sha256"),
        )
        object.__setattr__(self, "ledger_id", _non_empty_string(ledger_id, "ledger_id"))
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _required_sha256(ledger_path_sha256, "ledger_path_sha256"),
        )
        if any(
            not isinstance(prefix, DatabricksLedgerPrefix)
            for prefix in (predecessor_prefix, producer_batch_prefix, ledger_prefix)
        ):
            raise TypeError("authorization ledger prefixes have the wrong type")
        if any(
            prefix.ledger_id != ledger_id
            for prefix in (predecessor_prefix, producer_batch_prefix, ledger_prefix)
        ):
            raise ValueError("authorization ledger prefix identity drift")
        object.__setattr__(self, "predecessor_prefix", predecessor_prefix)
        object.__setattr__(self, "producer_batch_prefix", producer_batch_prefix)
        object.__setattr__(self, "ledger_prefix", ledger_prefix)
        object.__setattr__(
            self,
            "causal_closure_sha256",
            _required_sha256(causal_closure_sha256, "causal_closure_sha256"),
        )


class GPUQualificationSentinelRunner(Protocol):
    """Internal callable that performs one frozen GPU sentinel."""

    def __call__(
        self,
        *,
        plan_record: Mapping[str, Any],
        planned_job: Mapping[str, Any],
        artifact_paths: Mapping[str, Path],
        work_dir: Path,
    ) -> Mapping[str, Any]: ...


def render_gpu_qualification_submit_payloads(
    plan_record: Mapping[str, Any],
    *,
    runner_uri: str,
    package_wheel_uri: str,
    patched_vllm_wheel_uri: str,
    artifact_uris: Mapping[str, str],
    output_root: str,
) -> tuple[dict[str, Any], ...]:
    """Render one exact, no-retry Databricks payload for every planned job.

    ``artifact_uris`` is a closed mapping keyed by the six names in
    :data:`GPU_QUALIFICATION_ARTIFACT_KEYS`.  The runner, Cachet package wheel,
    and patched vLLM wheel arguments must repeat their corresponding mapping
    values; this makes accidental URI substitution visible before submission.
    """

    plan, pins = _validated_plan_and_pins(plan_record)
    uris = _validated_artifact_uris(
        artifact_uris,
        runner_uri=runner_uri,
        package_wheel_uri=package_wheel_uri,
        patched_vllm_wheel_uri=patched_vllm_wheel_uri,
    )
    normalized_output_root = _validated_output_root(output_root)
    plan_digest = _required_sha256(
        plan.get("closed_record_sha256"), "plan.closed_record_sha256"
    )
    jobs = _planned_jobs(plan)
    if not jobs or len(jobs) > GPU_QUALIFICATION_MAX_CLOUD_JOBS:
        raise ValueError("GPU qualification plan has an invalid cloud job count")

    encoded_plan = _encode_qualification_plan_parameter(
        canonical_gpu_qualification_json(plan)
    )
    payloads: list[dict[str, Any]] = []
    output_paths: set[str] = set()
    for planned_job in jobs:
        job_id = _safe_id(planned_job.get("job_id"), "planned job_id")
        hardware_id = _safe_id(planned_job.get("hardware_id"), "planned hardware_id")
        if (
            planned_job.get("attempt_number") != 0
            or planned_job.get("max_retries") != 0
        ):
            raise ValueError(f"planned job {job_id!r} is not attempt-zero-only")
        output_dir = _join_cluster_uri(normalized_output_root, plan_digest, job_id)
        output_json = _join_cluster_uri(output_dir, GPU_QUALIFICATION_OUTPUT_FILENAME)
        work_dir = str(_expected_local_work_dir(plan_digest, job_id))
        if output_json in output_paths:
            raise ValueError("GPU qualification output paths must be unique")
        output_paths.add(output_json)

        parameters = _runner_parameters(
            encoded_plan=encoded_plan,
            plan_digest=plan_digest,
            job_id=job_id,
            output_json=output_json,
            work_dir=work_dir,
            runner_uri=runner_uri,
            package_wheel_uri=package_wheel_uri,
            patched_vllm_wheel_uri=patched_vllm_wheel_uri,
            artifact_uris=uris,
            artifact_pins=pins,
            reservation_attempt_id=gpu_qualification_reservation_attempt_id(
                plan_digest, job_id
            ),
        )
        cluster = _qualification_cluster(
            hardware_id=hardware_id,
            custom_tags={
                "campaign": _safe_tag_value(plan.get("campaign_id")),
                "job_id": job_id,
                "plan_sha256": plan_digest[:32],
            },
        )
        task = {
            "task_key": _task_key(job_id),
            "timeout_seconds": GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS,
            "max_retries": 0,
            "new_cluster": cluster,
            "spark_python_task": {
                "python_file": runner_uri,
                "parameters": parameters,
            },
        }
        attempt_id = gpu_qualification_reservation_attempt_id(plan_digest, job_id)
        payloads.append(
            bind_databricks_run_idempotency_token(
                {
                    "run_name": _run_name(plan.get("campaign_id"), job_id),
                    "timeout_seconds": GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS,
                    "tasks": [task],
                },
                attempt_id=attempt_id,
            )
        )
    return tuple(payloads)


def _qualification_batch_requests(
    plan: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
) -> tuple[DatabricksRunAttemptReservationRequest, ...]:
    return tuple(
        DatabricksRunAttemptReservationRequest(
            attempt_id=str(contract["reservation_attempt_id"]),
            workload_id=(
                f"gpuq/{plan['closed_record_sha256'][:16]}/{contract['job_id']}"
            ),
            submit_payload=_required_mapping(contract.get("payload"), "payload"),
        )
        for contract in contracts
    )


def _validated_local_preflight_binding(
    path: str | Path,
    *,
    plan: Mapping[str, Any],
) -> tuple[dict[str, str], datetime, dict[str, Any]]:
    preflight_path = _validated_existing_regular_file(
        path, "local_preflight_evidence_path"
    )
    record = _read_canonical_json_object_file(
        preflight_path, "local preflight evidence"
    )
    plan_sha256 = _required_sha256(
        plan.get("closed_record_sha256"), "plan.closed_record_sha256"
    )
    completed_at = validate_local_preflight_evidence_record(
        record,
        plan_sha256=plan_sha256,
    )
    binding = {
        "completed_at_utc": _non_empty_string(
            record.get("completed_at_utc"), "local preflight completed_at_utc"
        ),
        "file_sha256": _file_sha256(preflight_path),
        "path_sha256": _canonical_json_sha256(
            {
                "domain": _QUALIFICATION_PREFLIGHT_PATH_DOMAIN,
                "path": str(preflight_path),
            }
        ),
        "record_sha256": _required_sha256(
            record.get("closed_record_sha256"),
            "local preflight closed_record_sha256",
        ),
    }
    return binding, completed_at, record


def _require_local_preflight_before_submission(
    completed_at: datetime,
    *,
    submission_boundary: datetime,
) -> None:
    boundary = _parse_utc_timestamp(
        _utc_timestamp(submission_boundary),
        "submission boundary",
    )
    if completed_at >= boundary:
        raise ValueError("local preflight must complete before qualification submission")


def _qualification_phase_lease_record(
    *,
    plan: Mapping[str, Any],
    ledger_path_sha256: str,
    predecessor_prefix: DatabricksLedgerPrefix,
    contracts: Sequence[Mapping[str, Any]],
    local_preflight_binding: Mapping[str, str],
) -> dict[str, Any]:
    if frozenset(local_preflight_binding) != _QUALIFICATION_PREFLIGHT_BINDING_KEYS:
        raise ValueError("local preflight binding has an open schema")
    record: dict[str, Any] = {
        "attempt_ids": [str(item["reservation_attempt_id"]) for item in contracts],
        "closed_record_sha256": "",
        "ledger_path_sha256": ledger_path_sha256,
        "local_preflight": dict(local_preflight_binding),
        "plan_sha256": plan["closed_record_sha256"],
        "predecessor_prefix": predecessor_prefix.to_record(),
        "record_type": _QUALIFICATION_PHASE_LEASE_RECORD_TYPE,
        "submit_payload_sha256": [
            str(item["submit_payload_sha256"]) for item in contracts
        ],
    }
    _seal_record(record)
    return record


def _qualification_batch_marker_record(
    *,
    lease_record: Mapping[str, Any],
    batch_authorization: DatabricksBatchReservationAuthorization,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt_ids": list(batch_authorization.attempt_ids),
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "ledger_path_sha256": batch_authorization.ledger_path_sha256,
        "phase_lease_record_sha256": lease_record["closed_record_sha256"],
        "predecessor_prefix": batch_authorization.predecessor_prefix.to_record(),
        "record_type": _QUALIFICATION_BATCH_MARKER_RECORD_TYPE,
        "submit_payload_sha256": list(batch_authorization.submit_payload_sha256s),
    }
    _seal_record(record)
    return record


def _replay_qualification_batch_marker(
    *,
    plan: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_binding: Mapping[str, str],
) -> tuple[DatabricksBatchReservationAuthorization, dict[str, Any]]:
    root = _validated_existing_controller_evidence_root(
        submit_receipt_root, "submit_receipt_root"
    )
    lease = _read_canonical_json_object_file(
        root / _QUALIFICATION_PHASE_LEASE_FILENAME,
        "qualification phase lease",
    )
    expected_predecessor = databricks_ledger_prefix_from_record(
        _required_mapping(plan.get("campaign_ledger_prefix"), "campaign_ledger_prefix")
    )
    expected_lease = _qualification_phase_lease_record(
        plan=plan,
        ledger_path_sha256=_required_sha256(
            plan.get("campaign_ledger_path_sha256"),
            "campaign_ledger_path_sha256",
        ),
        predecessor_prefix=expected_predecessor,
        contracts=contracts,
        local_preflight_binding=local_preflight_binding,
    )
    if lease != expected_lease:
        raise ValueError("qualification phase lease differs from the frozen batch")
    live = read_databricks_cluster_hour_ledger_json(ledger_path)
    predecessor_ledger = DatabricksClusterHourLedger(
        ledger_id=live.ledger_id,
        cap_cluster_hours=live.cap_cluster_hours,
        reservations=live.reservations[: expected_predecessor.reservation_count],
        submission_receipts=live.submission_receipts[
            : expected_predecessor.submission_receipt_count
        ],
        terminal_actuals=live.terminal_actuals[
            : expected_predecessor.terminal_actual_count
        ],
    )
    if (
        databricks_ledger_prefix_at_counts(
            predecessor_ledger,
            reservation_count=len(predecessor_ledger.reservations),
            submission_receipt_count=len(predecessor_ledger.submission_receipts),
            terminal_actual_count=len(predecessor_ledger.terminal_actuals),
        )
        != expected_predecessor
    ):
        raise ValueError("qualification phase lease predecessor history drift")
    _require_qualification_ledger_admission(
        predecessor_ledger,
        proposed_task_count=len(contracts),
        proposed_reserved_cluster_hours=(
            len(contracts) * GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS / 3600.0
        ),
        label="qualification durable batch replay",
    )
    authorization = replay_databricks_run_attempt_batch_authorization_json(
        ledger_path,
        _qualification_batch_requests(plan, contracts),
        expected_predecessor_prefix=expected_predecessor,
    )
    require_databricks_publication_batch_admission(live, authorization)
    expected_marker = _qualification_batch_marker_record(
        lease_record=lease,
        batch_authorization=authorization,
    )
    marker_path = root / _QUALIFICATION_BATCH_MARKER_FILENAME
    if marker_path.exists() or marker_path.is_symlink():
        marker = _read_canonical_json_object_file(
            marker_path,
            "qualification batch marker",
        )
        if marker != expected_marker:
            raise ValueError("qualification batch marker differs from the ledger batch")
    else:
        _write_canonical_exclusive(expected_marker, marker_path)
        marker = expected_marker
    return authorization, marker


def _require_qualification_phase_ledger_closure(
    ledger: DatabricksClusterHourLedger,
    *,
    batch_authorization: DatabricksBatchReservationAuthorization,
    contracts: Sequence[Mapping[str, Any]],
) -> DatabricksLedgerPrefix:
    attempts = tuple(str(item["reservation_attempt_id"]) for item in contracts)
    digests = tuple(str(item["submit_payload_sha256"]) for item in contracts)
    if (
        batch_authorization.attempt_ids != attempts
        or batch_authorization.submit_payload_sha256s != digests
    ):
        raise ValueError("qualification batch authority member closure drift")
    predecessor = batch_authorization.predecessor_prefix
    batch_prefix = batch_authorization.batch_prefix
    require_databricks_ledger_prefix(ledger, batch_prefix)
    receipt_start = predecessor.submission_receipt_count
    receipt_stop = receipt_start + len(attempts)
    terminal_start = predecessor.terminal_actual_count
    terminal_stop = terminal_start + len(attempts)
    receipts = ledger.submission_receipts[receipt_start:receipt_stop]
    terminals = ledger.terminal_actuals[terminal_start:terminal_stop]
    if tuple(item.attempt_id for item in receipts) != attempts:
        raise ValueError("qualification ledger receipt slice is not the exact batch")
    if tuple(item.attempt_id for item in terminals) != attempts:
        raise ValueError("qualification ledger terminal slice is not the exact batch")
    if tuple(item.submit_payload_sha256 for item in receipts) != digests:
        raise ValueError("qualification ledger receipt payload slice drift")
    if tuple(item.submit_payload_sha256 for item in terminals) != digests:
        raise ValueError("qualification ledger terminal payload slice drift")
    if any(attempt_id not in ledger.closed_attempt_ids for attempt_id in attempts):
        raise ValueError("qualification ledger still has an active batch member")
    receipt_prefix = databricks_ledger_prefix_at_counts(
        ledger,
        reservation_count=batch_prefix.reservation_count,
        submission_receipt_count=receipt_stop,
        terminal_actual_count=terminal_start,
    )
    terminal_prefix = databricks_ledger_prefix_at_counts(
        ledger,
        reservation_count=batch_prefix.reservation_count,
        submission_receipt_count=receipt_stop,
        terminal_actual_count=terminal_stop,
    )
    if (
        receipt_prefix.reservation_count != batch_prefix.reservation_count
        or terminal_prefix.reservation_count != batch_prefix.reservation_count
    ):
        raise RuntimeError("qualification historical prefix reconstruction drift")
    return terminal_prefix


def _qualification_submit_receipt_record(
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    ledger: DatabricksClusterHourLedger,
    phase_batch_record_sha256: str,
    submitted_at_utc: str,
) -> dict[str, Any]:
    attempt_id = str(contract["reservation_attempt_id"])
    ledger_receipt = next(
        item for item in ledger.submission_receipts if item.attempt_id == attempt_id
    )
    receipt: dict[str, Any] = {
        "authorization_scope": (
            "submission_identity_only_requires_direct_terminal_collection"
        ),
        "closed_record_sha256": "",
        "cloud_run_id": ledger_receipt.run_id,
        "job_id": contract["job_id"],
        "ledger_id": ledger.ledger_id,
        "output_json": contract["output_json"],
        "phase_batch_record_sha256": phase_batch_record_sha256,
        "plan_sha256": plan["closed_record_sha256"],
        "record_type": GPU_QUALIFICATION_SUBMIT_RECEIPT_RECORD_TYPE,
        "reservation_attempt_id": attempt_id,
        "schema_version": GPU_QUALIFICATION_SCHEMA_VERSION,
        "submit_payload_sha256": contract["submit_payload_sha256"],
        "submit_response_sha256": ledger_receipt.submit_response_sha256,
        "submitted_at_utc": submitted_at_utc,
        "task_key": contract["task_key"],
    }
    _seal_record(receipt)
    return receipt


def _qualification_post_intent_record(
    *,
    contract: Mapping[str, Any],
    batch_authorization: DatabricksBatchReservationAuthorization,
    phase_batch_record_sha256: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt_id": contract["reservation_attempt_id"],
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "job_id": contract["job_id"],
        "phase_batch_record_sha256": phase_batch_record_sha256,
        "state": "post_may_be_ambiguous_if_no_ledger_receipt",
        "submit_payload_sha256": contract["submit_payload_sha256"],
    }
    _seal_record(record)
    return record


def submit_gpu_qualification_jobs(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    opener: DatabricksURLOpener | None = None,
    now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Reserve, submit, and durably receipt-bind the exact fourteen jobs."""

    plan, _pins = _validated_plan_and_pins(plan_record)
    contracts = _validated_qualification_payloads(plan, submit_payloads)
    clock = now or _utc_now
    local_preflight_binding, preflight_completed_at, _preflight_record = (
        _validated_local_preflight_binding(
            local_preflight_evidence_path,
            plan=plan,
        )
    )
    _require_local_preflight_before_submission(
        preflight_completed_at,
        submission_boundary=clock(),
    )
    initial_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if databricks_ledger_path_sha256(ledger_path) != plan.get(
        "campaign_ledger_path_sha256"
    ):
        raise ValueError("qualification ledger path differs from the campaign plan")
    if initial_ledger.ledger_id != plan["campaign_ledger_id"]:
        raise ValueError("qualification ledger differs from the campaign plan")
    campaign_ledger_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(plan.get("campaign_ledger_prefix"), "campaign_ledger_prefix")
    )
    _require_qualification_ledger_admission(
        initial_ledger,
        proposed_task_count=len(contracts),
        proposed_reserved_cluster_hours=(
            len(contracts) * GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS / 3600.0
        ),
        label="qualification launch",
    )
    require_databricks_ledger_prefix(initial_ledger, campaign_ledger_prefix)
    if initial_ledger.terminal_actual_cluster_hours != plan.get(
        "campaign_opening_terminal_gpu_hours"
    ):
        raise ValueError("qualification ledger opening terminal balance drift")
    requests = _qualification_batch_requests(plan, contracts)

    def validate_batch(
        live: DatabricksClusterHourLedger,
        reservations: tuple[DatabricksClusterHourReservation, ...],
        snapshots: tuple[Mapping[str, Any], ...],
    ) -> None:
        if databricks_ledger_path_sha256(ledger_path) != plan.get(
            "campaign_ledger_path_sha256"
        ):
            raise ValueError("qualification ledger path differs from the campaign plan")
        if live.ledger_id != plan["campaign_ledger_id"]:
            raise ValueError("qualification ledger differs from the campaign plan")
        if live.terminal_actual_cluster_hours != plan.get(
            "campaign_opening_terminal_gpu_hours"
        ):
            raise ValueError("qualification ledger opening terminal balance drift")
        require_databricks_ledger_prefix(live, campaign_ledger_prefix)
        if len(reservations) != len(contracts) or len(snapshots) != len(contracts):
            raise ValueError("qualification batch is not the exact fourteen jobs")
        for contract, reservation, snapshot in zip(
            contracts, reservations, snapshots, strict=True
        ):
            if (
                reservation.attempt_id != contract["reservation_attempt_id"]
                or reservation.submit_payload_sha256
                != contract["submit_payload_sha256"]
                or canonical_gpu_qualification_json(snapshot)
                != canonical_gpu_qualification_json(
                    _required_mapping(contract.get("payload"), "payload")
                )
            ):
                raise ValueError("qualification batch reservation changed a payload")
        _require_qualification_ledger_admission(
            live,
            proposed_task_count=sum(
                len(item.task_timeout_seconds) for item in reservations
            ),
            proposed_reserved_cluster_hours=sum(
                item.reserved_cluster_hours for item in reservations
            ),
            label="qualification batch reservation",
        )

    receipt_root = _create_fresh_controller_evidence_root(submit_receipt_root)
    lease_record = _qualification_phase_lease_record(
        plan=plan,
        ledger_path_sha256=_required_sha256(
            plan.get("campaign_ledger_path_sha256"),
            "campaign_ledger_path_sha256",
        ),
        predecessor_prefix=campaign_ledger_prefix,
        contracts=contracts,
        local_preflight_binding=local_preflight_binding,
    )
    lease_path = receipt_root / _QUALIFICATION_PHASE_LEASE_FILENAME
    _write_canonical_exclusive(lease_record, lease_path)
    try:
        _batch_ledger, batch_authorization = (
            reserve_databricks_run_attempt_batch_authorized_json(
                ledger_path,
                requests,
                expected_predecessor_prefix=campaign_ledger_prefix,
                batch_validator=validate_batch,
            )
        )
    except BaseException:
        if lease_path.is_file() and not lease_path.is_symlink():
            lease_path.unlink()
            _fsync_directory(receipt_root)
        if receipt_root.is_dir() and not any(receipt_root.iterdir()):
            receipt_root.rmdir()
            _fsync_directory(receipt_root.parent)
        raise
    batch_marker = _qualification_batch_marker_record(
        lease_record=lease_record,
        batch_authorization=batch_authorization,
    )
    batch_marker_path = receipt_root / _QUALIFICATION_BATCH_MARKER_FILENAME
    _write_canonical_exclusive(batch_marker, batch_marker_path)
    resolved_opener = (
        cast(DatabricksURLOpener, urllib.request.urlopen) if opener is None else opener
    )
    receipts: list[dict[str, Any]] = []
    for contract in contracts:
        attempt_id = str(contract["reservation_attempt_id"])
        payload = _required_mapping(contract.get("payload"), "payload")
        intent_path = receipt_root / f"{contract['job_id']}.post-intent"
        intent = _qualification_post_intent_record(
            contract=contract,
            batch_authorization=batch_authorization,
            phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
        )
        _write_canonical_exclusive(intent, intent_path)
        response = submit_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id=attempt_id,
            batch_authorization=batch_authorization,
            opener=resolved_opener,
        )
        ledger = record_databricks_run_submission_receipt_json(
            ledger_path,
            attempt_id=attempt_id,
            submit_response=response,
        )
        submitted_at = _utc_timestamp(clock())
        receipt = _qualification_submit_receipt_record(
            plan=plan,
            contract=contract,
            ledger=ledger,
            phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
            submitted_at_utc=submitted_at,
        )
        _write_canonical_exclusive(receipt, receipt_root / f"{contract['job_id']}.json")
        intent_path.unlink()
        _fsync_directory(receipt_root)
        receipts.append(receipt)
    return tuple(receipts)


def resume_gpu_qualification_job_submissions(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    opener: DatabricksURLOpener | None = None,
    now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Resume the exact durable fourteen-job phase after a controller restart."""

    plan, _pins = _validated_plan_and_pins(plan_record)
    contracts = _validated_qualification_payloads(plan, submit_payloads)
    clock = now or _utc_now
    local_preflight_binding, preflight_completed_at, _preflight_record = (
        _validated_local_preflight_binding(
            local_preflight_evidence_path,
            plan=plan,
        )
    )
    _require_local_preflight_before_submission(
        preflight_completed_at,
        submission_boundary=clock(),
    )
    batch_authorization, batch_marker = _replay_qualification_batch_marker(
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_binding=local_preflight_binding,
    )
    root = _validated_existing_controller_evidence_root(
        submit_receipt_root, "submit_receipt_root"
    )
    receipts: list[dict[str, Any]] = []
    batch_marker_sha256 = str(batch_marker["closed_record_sha256"])
    for contract in contracts:
        job_id = str(contract["job_id"])
        attempt_id = str(contract["reservation_attempt_id"])
        payload = _required_mapping(contract.get("payload"), "payload")
        receipt_path = root / f"{job_id}.json"
        intent_path = root / f"{job_id}.post-intent"
        if receipt_path.exists() or receipt_path.is_symlink():
            ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
            receipt = _read_canonical_json_object_file(
                receipt_path, f"submit receipt {job_id}"
            )
            _validate_submit_receipt(
                receipt,
                contract=contract,
                plan=plan,
                ledger=ledger,
                phase_batch_record_sha256=batch_marker_sha256,
            )
            if intent_path.is_file() and not intent_path.is_symlink():
                intent_path.unlink()
                _fsync_directory(root)
            receipts.append(receipt)
            continue
        expected_intent = _qualification_post_intent_record(
            contract=contract,
            batch_authorization=batch_authorization,
            phase_batch_record_sha256=batch_marker_sha256,
        )
        if intent_path.exists() or intent_path.is_symlink():
            observed_intent = _read_canonical_json_object_file(
                intent_path, f"post intent {job_id}"
            )
            if observed_intent != expected_intent:
                raise ValueError(f"qualification post intent {job_id!r} drift")
        else:
            _write_canonical_exclusive(expected_intent, intent_path)
        resume_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id=attempt_id,
            batch_authorization=batch_authorization,
            opener=opener,
        )
        ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
        receipt = _qualification_submit_receipt_record(
            plan=plan,
            contract=contract,
            ledger=ledger,
            phase_batch_record_sha256=batch_marker_sha256,
            submitted_at_utc=_utc_timestamp(clock()),
        )
        try:
            _write_canonical_exclusive(receipt, receipt_path)
        except FileExistsError:
            observed_receipt = _read_canonical_json_object_file(
                receipt_path, f"submit receipt {job_id}"
            )
            _validate_submit_receipt(
                observed_receipt,
                contract=contract,
                plan=plan,
                ledger=ledger,
                phase_batch_record_sha256=batch_marker_sha256,
            )
            receipt = observed_receipt
        if intent_path.is_file() and not intent_path.is_symlink():
            intent_path.unlink()
            _fsync_directory(root)
        receipts.append(receipt)
    expected_names = {
        _QUALIFICATION_PHASE_LEASE_FILENAME,
        _QUALIFICATION_BATCH_MARKER_FILENAME,
        *(f"{item['job_id']}.json" for item in contracts),
    }
    if {item.name for item in root.iterdir()} != expected_names:
        raise ValueError("resumed qualification receipt directory is not closed")
    return tuple(receipts)


def _require_qualification_ledger_admission(
    ledger: DatabricksClusterHourLedger,
    *,
    proposed_task_count: int,
    proposed_reserved_cluster_hours: float,
    label: str,
) -> None:
    if ledger.cap_cluster_hours != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS:
        raise ValueError(f"{label} requires the migrated 1024-hour campaign ledger")
    if (
        ledger.active_reserved_task_count + proposed_task_count
        > PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
    ):
        raise ValueError(f"{label} exceeds the global 16-job concurrency cap")
    if (
        ledger.active_reserved_cluster_hours + proposed_reserved_cluster_hours
        > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
    ):
        raise ValueError(f"{label} exceeds the 900-hour active reservation cap")
    if (
        ledger.accounted_cluster_hours
        + proposed_reserved_cluster_hours
        + PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
        > MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
    ):
        raise ValueError(f"{label} would consume the 124-hour campaign headroom")


def _validated_qualification_payloads(
    plan: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(submit_payloads, (str, bytes, bytearray)) or not isinstance(
        submit_payloads, Sequence
    ):
        raise TypeError("submit_payloads must be a sequence")
    jobs = _planned_jobs(plan)
    if len(submit_payloads) != len(jobs):
        raise ValueError(
            "qualification submission requires the exact planned job closure"
        )
    pins = pins_from_plan_record(plan)
    plan_digest = _required_sha256(
        plan.get("closed_record_sha256"), "plan.closed_record_sha256"
    )
    encoded_plan = _encode_qualification_plan_parameter(
        canonical_gpu_qualification_json(plan)
    )
    contracts: list[dict[str, Any]] = []
    for planned_job, raw_payload in zip(jobs, submit_payloads, strict=True):
        payload = _json_object(raw_payload, "qualification submit payload")
        if set(payload) != {
            "idempotency_token",
            "run_name",
            "tasks",
            "timeout_seconds",
        }:
            raise ValueError("qualification submit payload has an open schema")
        job_id = _safe_id(planned_job.get("job_id"), "planned job_id")
        if payload.get("run_name") != _run_name(plan.get("campaign_id"), job_id):
            raise ValueError("qualification run_name does not match the plan")
        if (
            payload.get("timeout_seconds")
            != GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS
        ):
            raise ValueError("qualification run timeout differs")
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or len(raw_tasks) != 1:
            raise ValueError("qualification payload must contain exactly one task")
        task = _required_mapping(raw_tasks[0], "qualification task")
        if set(task) != {
            "max_retries",
            "new_cluster",
            "spark_python_task",
            "task_key",
            "timeout_seconds",
        }:
            raise ValueError("qualification task has an open schema")
        task_key = _task_key(job_id)
        if (
            task.get("task_key") != task_key
            or task.get("max_retries") != 0
            or task.get("timeout_seconds")
            != GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS
        ):
            raise ValueError("qualification task retry/timeout identity differs")
        expected_cluster = _qualification_cluster(
            hardware_id=_safe_id(planned_job.get("hardware_id"), "hardware_id"),
            custom_tags={
                "campaign": _safe_tag_value(plan.get("campaign_id")),
                "job_id": job_id,
                "plan_sha256": plan_digest[:32],
            },
        )
        if task.get("new_cluster") != expected_cluster:
            raise ValueError("qualification cluster specification differs")
        python_task = _required_mapping(
            task.get("spark_python_task"), "spark_python_task"
        )
        if set(python_task) != {"parameters", "python_file"}:
            raise ValueError("qualification spark_python_task has an open schema")
        parameters = python_task.get("parameters")
        if not isinstance(parameters, list) or any(
            not isinstance(item, str) for item in parameters
        ):
            raise ValueError("qualification parameters must be strings")
        _require_qualification_parameters_size(parameters)
        runner_uri = _one_parameter(parameters, "--runner-uri")
        package_wheel_uri = _one_parameter(parameters, "--package-wheel-uri")
        patched_wheel_uri = _one_parameter(parameters, "--patched-vllm-wheel-uri")
        artifact_uris = _parse_key_value_args(
            _all_parameters(parameters, "--artifact-uri"),
            option_name="--artifact-uri",
        )
        validated_uris = _validated_artifact_uris(
            artifact_uris,
            runner_uri=runner_uri,
            package_wheel_uri=package_wheel_uri,
            patched_vllm_wheel_uri=patched_wheel_uri,
        )
        output_json = _validated_result_output_json(
            _one_parameter(parameters, "--output-json"),
            plan_digest=plan_digest,
            job_id=job_id,
        )
        work_dir = str(_expected_local_work_dir(plan_digest, job_id))
        attempt_id = gpu_qualification_reservation_attempt_id(plan_digest, job_id)
        require_databricks_run_idempotency_token(payload, attempt_id=attempt_id)
        expected_parameters = _runner_parameters(
            encoded_plan=encoded_plan,
            plan_digest=plan_digest,
            job_id=job_id,
            output_json=output_json,
            work_dir=work_dir,
            runner_uri=runner_uri,
            package_wheel_uri=package_wheel_uri,
            patched_vllm_wheel_uri=patched_wheel_uri,
            artifact_uris=validated_uris,
            artifact_pins=pins,
            reservation_attempt_id=attempt_id,
        )
        if (
            parameters != expected_parameters
            or python_task.get("python_file") != runner_uri
        ):
            raise ValueError("qualification task parameters differ from the renderer")
        snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(
            payload
        )
        contracts.append(
            {
                "job_id": job_id,
                "output_json": output_json,
                "payload": snapshot,
                "reservation_attempt_id": attempt_id,
                "submit_payload_sha256": sha256(canonical_payload).hexdigest(),
                "task_key": task_key,
            }
        )
    return tuple(contracts)


def collect_gpu_qualification_evidence(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    terminal_receipt_root: str | Path,
    evidence_output_json: str | Path,
    now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], GPUQualificationLaunchAuthorization]:
    """Authorize qualification only through the real terminal Jobs API transport.

    Unlike submission, this authority-bearing boundary intentionally has no
    injectable opener.  Tests may monkeypatch the package-owned ``runs/get``
    function, but callers cannot pass fabricated status mappings into an
    authorizing production invocation.
    """

    plan, pins = _validated_plan_and_pins(plan_record)
    contracts = _validated_qualification_payloads(plan, submit_payloads)
    local_preflight_binding, _preflight_completed_at, local_preflight = (
        _validated_local_preflight_binding(
            local_preflight_evidence_path,
            plan=plan,
        )
    )
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if databricks_ledger_path_sha256(ledger_path) != plan.get(
        "campaign_ledger_path_sha256"
    ):
        raise ValueError("collection ledger path differs from the campaign plan")
    batch_authorization, batch_marker = _replay_qualification_batch_marker(
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_binding=local_preflight_binding,
    )
    submit_receipts = _load_submit_receipts(
        submit_receipt_root,
        contracts=contracts,
        plan=plan,
        ledger=ledger,
        phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
    )
    terminal_root = _validated_fresh_controller_evidence_root(terminal_receipt_root)
    _require_fresh_output_path(Path(evidence_output_json))
    clock = now or _utc_now
    job_results: list[dict[str, Any]] = []
    terminal_receipts: list[dict[str, Any]] = []
    for planned_job, contract, submit_receipt in zip(
        _planned_jobs(plan), contracts, submit_receipts, strict=True
    ):
        run = get_databricks_run(
            config,
            str(submit_receipt["cloud_run_id"]),
        )
        # Reconcile every direct terminal outcome before applying the stricter
        # success-only qualification launch/result contract.  A rejected job
        # with no allocated task timestamps must release its reservation while
        # still failing the campaign.
        reconciled = record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id=str(contract["reservation_attempt_id"]),
            run_record=run,
        )
        actual = next(
            item
            for item in reconciled.terminal_actuals
            if item.attempt_id == contract["reservation_attempt_id"]
        )
        run_identity = _validate_control_plane_run(
            run,
            planned_job=planned_job,
            contract=contract,
            submit_receipt=submit_receipt,
        )
        control_plane_status_sha256 = _canonical_json_sha256(run)
        if (
            actual.verification_source != "direct_databricks_runs_get"
            or actual.run_id != run_identity["cloud_run_id"]
            or actual.submit_payload_sha256 != contract["submit_payload_sha256"]
            or actual.control_plane_status_sha256 != control_plane_status_sha256
        ):
            raise RuntimeError(
                "qualification ledger terminal is not bound to this control-plane response"
            )
        if (
            run_identity["succeeded"] is not True
            or actual.terminal_state != "succeeded"
        ):
            raise RuntimeError(
                f"GPU qualification job {contract['job_id']!r} did not succeed"
            )
        result_path = _cluster_file_path(str(contract["output_json"]))
        result = _read_canonical_json_object_file(
            result_path, f"GPU result {contract['job_id']}"
        )
        validate_gpu_job_result_record(
            result,
            plan_record=plan,
            expected_artifact_pins=pins,
        )
        _validate_result_submission_binding(
            result,
            contract=contract,
            submit_receipt=submit_receipt,
            run_identity=run_identity,
        )
        terminal_receipt = _terminal_receipt_record(
            plan=plan,
            contract=contract,
            submit_receipt=submit_receipt,
            ledger_id=reconciled.ledger_id,
            ledger_terminal_actual=actual,
            run=run,
            run_identity=run_identity,
            result=result,
            collected_at_utc=_utc_timestamp(clock()),
            phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
        )
        job_results.append(result)
        terminal_receipts.append(terminal_receipt)

    final_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    terminal_prefix = _require_qualification_phase_ledger_closure(
        final_ledger,
        batch_authorization=batch_authorization,
        contracts=contracts,
    )
    for receipt in terminal_receipts:
        receipt["phase_terminal_prefix"] = terminal_prefix.to_record()
        receipt["closed_record_sha256"] = ""
        _seal_record(receipt)
    _validate_collected_identity_closure(terminal_receipts, contracts=contracts)
    selected_gmus = [
        float(result["measurements"]["gpu_memory_utilization"])
        for result in job_results
        if result["job_id"].startswith("aws-g6-l4-32k-c4-gmu-")
        and result["measurements"].get("candidate_qualified") is True
    ]
    if not selected_gmus:
        raise RuntimeError("no governed GMU result qualified")
    cloud = _build_governed_cloud_gpu_evidence(
        plan_sha256=str(plan["closed_record_sha256"]),
        jobs=job_results,
        terminal_receipts=terminal_receipts,
        selected_gpu_memory_utilization=max(selected_gmus),
    )
    evidence = _build_governed_gpu_qualification_evidence(
        campaign_id=str(plan["campaign_id"]),
        plan_sha256=str(plan["closed_record_sha256"]),
        local_preflight_evidence=local_preflight,
        cloud_gpu_evidence=cloud,
    )
    validate_gpu_qualification_evidence_record(
        evidence,
        plan_record=plan,
        expected_campaign_id=str(plan["campaign_id"]),
        expected_artifact_pins=pins,
    )
    _publish_terminal_receipts_atomic(terminal_root, terminal_receipts)
    evidence_path = Path(evidence_output_json)
    _write_canonical_exclusive(evidence, evidence_path)
    authorization = replay_gpu_qualification_launch_authorization(
        plan_record=plan,
        submit_payloads=submit_payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_evidence_path=local_preflight_evidence_path,
        terminal_receipt_root=terminal_root,
        evidence_path=evidence_path,
        expected_campaign_id=str(plan["campaign_id"]),
        expected_artifact_pins=pins,
    )
    return evidence, authorization


def replay_gpu_qualification_launch_authorization(
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    terminal_receipt_root: str | Path,
    evidence_path: str | Path,
    expected_campaign_id: str,
    expected_artifact_pins: GPUQualificationArtifactPins,
) -> GPUQualificationLaunchAuthorization:
    """Reissue launch authority from the complete durable causal closure.

    A qualification JSON record by itself is intentionally insufficient.  A
    replay must rejoin the exact rendered submit payloads, append-only ledger,
    submit receipts, terminal receipts, and canonical evidence file.
    """

    plan, pins = _validated_plan_and_pins(plan_record)
    if str(plan["campaign_id"]) != expected_campaign_id:
        raise ValueError("replay campaign_id differs from the frozen expectation")
    if pins != expected_artifact_pins:
        raise ValueError("replay artifact pins differ from the frozen expectation")
    contracts = _validated_qualification_payloads(plan, submit_payloads)
    local_preflight_binding, _preflight_completed_at, local_preflight = (
        _validated_local_preflight_binding(
            local_preflight_evidence_path,
            plan=plan,
        )
    )
    ledger_file = _validated_existing_regular_file(ledger_path, "ledger_path")
    ledger = read_databricks_cluster_hour_ledger_json(ledger_file)
    ledger_path_sha256 = databricks_ledger_path_sha256(ledger_file)
    if ledger_path_sha256 != plan.get("campaign_ledger_path_sha256"):
        raise ValueError("replay ledger path differs from the campaign plan")
    require_databricks_ledger_prefix(
        ledger,
        databricks_ledger_prefix_from_record(
            _required_mapping(
                plan.get("campaign_ledger_prefix"), "campaign_ledger_prefix"
            )
        ),
    )
    batch_authorization, batch_marker = _replay_qualification_batch_marker(
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_file,
        submit_receipt_root=submit_receipt_root,
        local_preflight_binding=local_preflight_binding,
    )
    submit_receipts = _load_submit_receipts(
        submit_receipt_root,
        contracts=contracts,
        plan=plan,
        ledger=ledger,
        phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
    )
    terminal_root = _validated_existing_controller_evidence_root(
        terminal_receipt_root, "terminal_receipt_root"
    )
    expected_names = {f"{contract['job_id']}.json" for contract in contracts}
    observed_names = {path.name for path in terminal_root.iterdir()}
    if observed_names != expected_names:
        raise ValueError("terminal receipt directory is not the exact planned closure")
    terminal_receipts = tuple(
        _read_canonical_json_object_file(
            terminal_root / f"{contract['job_id']}.json",
            f"terminal receipt {contract['job_id']}",
        )
        for contract in contracts
    )
    evidence_file = _validated_existing_regular_file(evidence_path, "evidence_path")
    evidence = _read_canonical_json_object_file(
        evidence_file, "GPU qualification evidence"
    )
    selection = validate_gpu_qualification_evidence_record(
        evidence,
        plan_record=plan,
        expected_campaign_id=expected_campaign_id,
        expected_artifact_pins=expected_artifact_pins,
    )
    if dict(
        _required_mapping(
            evidence.get("local_preflight_evidence"),
            "local_preflight_evidence",
        )
    ) != local_preflight:
        raise ValueError(
            "persisted local preflight differs from qualification evidence"
        )
    cloud = _required_mapping(evidence.get("cloud_gpu_evidence"), "cloud_gpu_evidence")
    embedded_receipts = cloud.get("terminal_receipts")
    if not isinstance(embedded_receipts, list) or embedded_receipts != list(
        terminal_receipts
    ):
        raise ValueError(
            "persisted terminal receipt closure differs from qualification evidence"
        )

    terminal_actual_hashes: list[str] = []
    terminal_prefix = _require_qualification_phase_ledger_closure(
        ledger,
        batch_authorization=batch_authorization,
        contracts=contracts,
    )
    for contract, submit_receipt, terminal_receipt in zip(
        contracts, submit_receipts, terminal_receipts, strict=True
    ):
        attempt_id = str(contract["reservation_attempt_id"])
        actual = next(
            (item for item in ledger.terminal_actuals if item.attempt_id == attempt_id),
            None,
        )
        if actual is None:
            raise ValueError(f"replay ledger has no terminal actual for {attempt_id!r}")
        actual_record = _ledger_terminal_actual_record(actual)
        actual_sha256 = _canonical_json_sha256(actual_record)
        if (
            actual.verification_source != "direct_databricks_runs_get"
            or actual.terminal_state != "succeeded"
            or actual.run_id != submit_receipt["cloud_run_id"]
            or actual.submit_payload_sha256 != contract["submit_payload_sha256"]
            or actual.control_plane_status_sha256
            != terminal_receipt["control_plane_status_sha256"]
            or actual_sha256 != terminal_receipt["ledger_terminal_actual_sha256"]
            or terminal_receipt.get("phase_batch_record_sha256")
            != batch_marker["closed_record_sha256"]
            or terminal_receipt.get("phase_terminal_prefix")
            != terminal_prefix.to_record()
        ):
            raise ValueError(
                f"replay ledger terminal does not match {contract['job_id']!r}"
            )
        terminal_actual_hashes.append(actual_sha256)

    evidence_file_sha256 = _file_sha256(evidence_file)
    ledger_prefix = terminal_prefix
    causal_closure = {
        "evidence_closed_record_sha256": evidence["closed_record_sha256"],
        "evidence_file_sha256": evidence_file_sha256,
        "ledger_id": ledger.ledger_id,
        "ledger_path_sha256": ledger_path_sha256,
        "ledger_prefix": ledger_prefix.to_record(),
        "predecessor_prefix": batch_authorization.predecessor_prefix.to_record(),
        "producer_batch_prefix": batch_authorization.batch_prefix.to_record(),
        "plan_sha256": plan["closed_record_sha256"],
        "submit_payload_sha256": [
            contract["submit_payload_sha256"] for contract in contracts
        ],
        "submit_receipt_sha256": [
            receipt["closed_record_sha256"] for receipt in submit_receipts
        ],
        "terminal_actual_sha256": terminal_actual_hashes,
        "terminal_receipt_sha256": [
            receipt["closed_record_sha256"] for receipt in terminal_receipts
        ],
    }
    return GPUQualificationLaunchAuthorization(
        selection=selection,
        plan_sha256=str(plan["closed_record_sha256"]),
        evidence_closed_record_sha256=str(evidence["closed_record_sha256"]),
        evidence_file_sha256=evidence_file_sha256,
        ledger_id=ledger.ledger_id,
        ledger_path_sha256=ledger_path_sha256,
        predecessor_prefix=batch_authorization.predecessor_prefix,
        producer_batch_prefix=batch_authorization.batch_prefix,
        ledger_prefix=ledger_prefix,
        causal_closure_sha256=_canonical_json_sha256(causal_closure),
        _issuer=_LAUNCH_AUTHORIZATION_ISSUER,
    )


def require_gpu_qualification_launch_authorization(
    value: object,
    *,
    expected_plan_sha256: str,
    expected_evidence_file_sha256: str,
) -> GPUQualificationSelection:
    """Return the selection only from a collector/replay-issued capability."""

    if not isinstance(value, GPUQualificationLaunchAuthorization):
        raise TypeError(
            "publication launch requires GPUQualificationLaunchAuthorization"
        )
    if value.plan_sha256 != _required_sha256(
        expected_plan_sha256, "expected_plan_sha256"
    ):
        raise ValueError("GPU qualification authorization plan binding differs")
    if value.evidence_file_sha256 != _required_sha256(
        expected_evidence_file_sha256, "expected_evidence_file_sha256"
    ):
        raise ValueError("GPU qualification authorization evidence binding differs")
    return value.selection


def _ledger_terminal_actual_record(
    actual: DatabricksClusterHourTerminalActual,
) -> dict[str, Any]:
    return {
        "actual_cluster_duration_seconds": actual.actual_cluster_duration_seconds,
        "actual_cluster_hours": actual.actual_cluster_hours,
        "attempt_id": actual.attempt_id,
        "control_plane_status_sha256": actual.control_plane_status_sha256,
        "run_id": actual.run_id,
        "submit_payload_sha256": actual.submit_payload_sha256,
        "terminal_state": actual.terminal_state,
        "verification_source": actual.verification_source,
    }


def _validate_collected_identity_closure(
    terminal_receipts: Sequence[Mapping[str, Any]],
    *,
    contracts: Sequence[Mapping[str, Any]],
) -> None:
    if len(terminal_receipts) != len(contracts):
        raise ValueError("collected identity closure is incomplete")
    run_ids: set[str] = set()
    task_run_ids: set[str] = set()
    cluster_ids: set[str] = set()
    attempt_ids: set[str] = set()
    task_keys: set[str] = set()
    output_paths: set[str] = set()
    for receipt, contract in zip(terminal_receipts, contracts, strict=True):
        exact = {
            "job_id": contract["job_id"],
            "output_json": contract["output_json"],
            "reservation_attempt_id": contract["reservation_attempt_id"],
            "submit_payload_sha256": contract["submit_payload_sha256"],
            "task_key": contract["task_key"],
        }
        for field_name, expected in exact.items():
            if receipt.get(field_name) != expected:
                raise ValueError(
                    f"collected terminal receipt {field_name} differs from submission"
                )
        identities = (
            (
                run_ids,
                _databricks_run_id(receipt.get("cloud_run_id"), "cloud_run_id"),
                "cloud_run_id",
            ),
            (
                task_run_ids,
                _databricks_run_id(receipt.get("task_run_id"), "task_run_id"),
                "task_run_id",
            ),
            (
                cluster_ids,
                _non_empty_string(receipt.get("cloud_cluster_id"), "cloud_cluster_id"),
                "cloud_cluster_id",
            ),
            (
                attempt_ids,
                str(receipt["reservation_attempt_id"]),
                "reservation_attempt_id",
            ),
            (task_keys, str(receipt["task_key"]), "task_key"),
            (output_paths, str(receipt["output_json"]), "output_json"),
        )
        for observed, value, field_name in identities:
            if value in observed:
                raise ValueError(f"collected {field_name} values must be unique")
            observed.add(value)


def _load_submit_receipts(
    root: str | Path,
    *,
    contracts: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    ledger: DatabricksClusterHourLedger,
    phase_batch_record_sha256: str,
) -> tuple[dict[str, Any], ...]:
    directory = _validated_existing_controller_evidence_root(
        root, "submit_receipt_root"
    )
    expected_names = {
        _QUALIFICATION_PHASE_LEASE_FILENAME,
        _QUALIFICATION_BATCH_MARKER_FILENAME,
        *(f"{contract['job_id']}.json" for contract in contracts),
    }
    observed_names = {path.name for path in directory.iterdir()}
    if observed_names != expected_names:
        raise ValueError("submit receipt directory is not the exact planned closure")
    receipts: list[dict[str, Any]] = []
    for contract in contracts:
        receipt = _read_canonical_json_object_file(
            directory / f"{contract['job_id']}.json",
            f"submit receipt {contract['job_id']}",
        )
        _validate_submit_receipt(
            receipt,
            contract=contract,
            plan=plan,
            ledger=ledger,
            phase_batch_record_sha256=phase_batch_record_sha256,
        )
        receipts.append(receipt)
    return tuple(receipts)


def _validate_submit_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    ledger: DatabricksClusterHourLedger,
    phase_batch_record_sha256: str,
) -> None:
    if set(receipt) != _SUBMIT_RECEIPT_KEYS:
        raise ValueError("submit receipt has an open schema")
    _require_closed_record_digest(receipt, "submit receipt")
    expected = {
        "authorization_scope": (
            "submission_identity_only_requires_direct_terminal_collection"
        ),
        "job_id": contract["job_id"],
        "ledger_id": ledger.ledger_id,
        "output_json": contract["output_json"],
        "plan_sha256": plan["closed_record_sha256"],
        "phase_batch_record_sha256": phase_batch_record_sha256,
        "record_type": GPU_QUALIFICATION_SUBMIT_RECEIPT_RECORD_TYPE,
        "reservation_attempt_id": contract["reservation_attempt_id"],
        "schema_version": GPU_QUALIFICATION_SCHEMA_VERSION,
        "submit_payload_sha256": contract["submit_payload_sha256"],
        "task_key": contract["task_key"],
    }
    for field_name, expected_value in expected.items():
        if receipt.get(field_name) != expected_value:
            raise ValueError(f"submit receipt {field_name} differs")
    _required_sha256(receipt.get("submit_response_sha256"), "submit_response_sha256")
    _parse_utc_timestamp(receipt.get("submitted_at_utc"), "submitted_at_utc")
    cloud_run_id = _non_empty_string(receipt.get("cloud_run_id"), "cloud_run_id")
    ledger_receipt = next(
        (
            item
            for item in ledger.submission_receipts
            if item.attempt_id == contract["reservation_attempt_id"]
        ),
        None,
    )
    if ledger_receipt is None or (
        ledger_receipt.run_id != cloud_run_id
        or ledger_receipt.submit_payload_sha256 != contract["submit_payload_sha256"]
        or ledger_receipt.submit_response_sha256 != receipt["submit_response_sha256"]
    ):
        raise ValueError("submit receipt does not match the append-only ledger")


def _validate_control_plane_run(
    run: Mapping[str, Any],
    *,
    planned_job: Mapping[str, Any],
    contract: Mapping[str, Any],
    submit_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    control_plane_run_id = _databricks_run_id(run.get("run_id"), "runs/get run_id")
    if control_plane_run_id != submit_receipt["cloud_run_id"]:
        raise ValueError("runs/get run_id differs from the submit receipt")
    payload = _required_mapping(contract.get("payload"), "payload")
    if run.get("run_name") != payload.get("run_name"):
        raise ValueError("runs/get run_name differs from the immutable submit payload")
    if run.get("run_type") not in (None, "SUBMIT_RUN"):
        raise ValueError("qualification run is not a one-time submit run")
    if run.get("repair_history") not in (None, []):
        raise ValueError("qualification run has repair history")
    if run.get("original_attempt_run_id") not in (None, 0, "0"):
        raise ValueError("qualification run is not attempt zero")
    state = _required_mapping(run.get("state"), "runs/get state")
    life_cycle_state = state.get("life_cycle_state")
    result_state = state.get("result_state")
    if life_cycle_state not in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR", "BLOCKED"}:
        raise ValueError("qualification runs/get response is not terminal")
    start_time = _nonnegative_int(run.get("start_time"), "run.start_time")
    end_time = _positive_int(run.get("end_time"), "run.end_time")
    if end_time <= start_time:
        raise ValueError("qualification run terminal times do not increase")
    tasks = run.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 1
        or not isinstance(tasks[0], Mapping)
    ):
        raise ValueError("qualification runs/get must contain exactly one task")
    task = tasks[0]
    if task.get("task_key") != contract["task_key"]:
        raise ValueError("runs/get task_key differs")
    if task.get("attempt_number") not in (None, 0):
        raise ValueError("runs/get task was retried")
    task_state = _required_mapping(task.get("state"), "runs/get task state")
    task_life_cycle_state = task_state.get("life_cycle_state")
    task_result_state = task_state.get("result_state")
    if task_life_cycle_state not in {
        "TERMINATED",
        "SKIPPED",
        "INTERNAL_ERROR",
        "BLOCKED",
    }:
        raise ValueError("qualification task is not terminal")
    task_start = _nonnegative_int(task.get("start_time"), "task.start_time")
    task_end = _positive_int(task.get("end_time"), "task.end_time")
    if not start_time <= task_start < task_end <= end_time:
        raise ValueError("qualification task interval is not nested in the run")
    task_run_id_value = _databricks_run_id(
        task.get("run_id"), "qualification task run_id"
    )
    cluster_instance = _required_mapping(
        task.get("cluster_instance"), "task.cluster_instance"
    )
    cluster_id = _non_empty_string(
        cluster_instance.get("cluster_id"), "task cluster_id"
    )
    submitted_task = _required_mapping(payload["tasks"][0], "submitted task")
    submitted_cluster = _required_mapping(
        submitted_task.get("new_cluster"), "submitted cluster"
    )
    observed_cluster = _control_plane_launch_cluster(task)
    for field_name in (
        "node_type_id",
        "driver_node_type_id",
        "spark_version",
        "data_security_mode",
        "num_workers",
    ):
        if observed_cluster.get(field_name) != submitted_cluster.get(field_name):
            raise ValueError(f"runs/get cluster {field_name} differs")
    expected_node_type = submitted_cluster["node_type_id"]
    if planned_job.get("hardware_id") == GPU_QUALIFICATION_GENERATION_HARDWARE_ID:
        if expected_node_type != GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE:
            raise ValueError("L40S control-plane node type differs")
    succeeded = (
        life_cycle_state == "TERMINATED"
        and result_state == "SUCCESS"
        and task_life_cycle_state == "TERMINATED"
        and task_result_state == "SUCCESS"
    )
    return {
        "cloud_cluster_id": cluster_id,
        "cloud_run_id": control_plane_run_id,
        "driver_node_type_id": str(observed_cluster["driver_node_type_id"]),
        "end_time_ms": end_time,
        "life_cycle_state": life_cycle_state,
        "node_type_id": str(observed_cluster["node_type_id"]),
        "result_state": result_state,
        "run_name": str(run["run_name"]),
        "start_time_ms": start_time,
        "succeeded": succeeded,
        "task_end_time_ms": task_end,
        "task_life_cycle_state": task_life_cycle_state,
        "task_result_state": task_result_state,
        "task_run_id": task_run_id_value,
        "task_start_time_ms": task_start,
    }


def _validate_result_submission_binding(
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    submit_receipt: Mapping[str, Any],
    run_identity: Mapping[str, Any],
) -> None:
    expected = {
        "cloud_cluster_id": run_identity["cloud_cluster_id"],
        "cloud_run_id": submit_receipt["cloud_run_id"],
        "job_id": contract["job_id"],
        "output_json": contract["output_json"],
        "reservation_attempt_id": contract["reservation_attempt_id"],
        "task_key": contract["task_key"],
    }
    for field_name, expected_value in expected.items():
        if result.get(field_name) != expected_value:
            raise ValueError(f"GPU result {field_name} differs from submission")


def _terminal_receipt_record(
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    submit_receipt: Mapping[str, Any],
    ledger_id: str,
    ledger_terminal_actual: DatabricksClusterHourTerminalActual,
    run: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    result: Mapping[str, Any],
    collected_at_utc: str,
    phase_batch_record_sha256: str,
) -> dict[str, Any]:
    result_bytes = (canonical_gpu_qualification_json(result) + "\n").encode("utf-8")
    control_plane_bytes = json.dumps(
        run,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    ledger_terminal_record = {
        "actual_cluster_duration_seconds": (
            ledger_terminal_actual.actual_cluster_duration_seconds
        ),
        "actual_cluster_hours": ledger_terminal_actual.actual_cluster_hours,
        "attempt_id": ledger_terminal_actual.attempt_id,
        "control_plane_status_sha256": (
            ledger_terminal_actual.control_plane_status_sha256
        ),
        "run_id": ledger_terminal_actual.run_id,
        "submit_payload_sha256": (ledger_terminal_actual.submit_payload_sha256),
        "terminal_state": ledger_terminal_actual.terminal_state,
        "verification_source": ledger_terminal_actual.verification_source,
    }
    receipt: dict[str, Any] = {
        "authorization_source": "direct_databricks_runs_get",
        "closed_record_sha256": "",
        "cloud_cluster_id": run_identity["cloud_cluster_id"],
        "cloud_run_id": run_identity["cloud_run_id"],
        "collected_at_utc": collected_at_utc,
        "control_plane_status_sha256": sha256(control_plane_bytes).hexdigest(),
        "driver_node_type_id": run_identity["driver_node_type_id"],
        "end_time_ms": run_identity["end_time_ms"],
        "job_id": contract["job_id"],
        "ledger_actual_cluster_duration_seconds": (
            ledger_terminal_actual.actual_cluster_duration_seconds
        ),
        "ledger_id": _non_empty_string(ledger_id, "ledger_id"),
        "ledger_terminal_actual_sha256": _canonical_json_sha256(ledger_terminal_record),
        "life_cycle_state": run_identity["life_cycle_state"],
        "node_type_id": run_identity["node_type_id"],
        "output_json": contract["output_json"],
        "phase_batch_record_sha256": _required_sha256(
            phase_batch_record_sha256, "phase_batch_record_sha256"
        ),
        "plan_sha256": plan["closed_record_sha256"],
        "record_type": GPU_QUALIFICATION_TERMINAL_RECEIPT_RECORD_TYPE,
        "reservation_attempt_id": contract["reservation_attempt_id"],
        "result_file_sha256": sha256(result_bytes).hexdigest(),
        "result_record_sha256": result["closed_record_sha256"],
        "result_state": run_identity["result_state"],
        "run_name": run_identity["run_name"],
        "schema_version": GPU_QUALIFICATION_SCHEMA_VERSION,
        "start_time_ms": run_identity["start_time_ms"],
        "submit_payload_sha256": contract["submit_payload_sha256"],
        "task_attempt_number": 0,
        "task_end_time_ms": run_identity["task_end_time_ms"],
        "task_key": contract["task_key"],
        "task_life_cycle_state": run_identity["task_life_cycle_state"],
        "task_max_retries": 0,
        "task_result_state": run_identity["task_result_state"],
        "task_run_id": run_identity["task_run_id"],
        "task_start_time_ms": run_identity["task_start_time_ms"],
    }
    _seal_record(receipt)
    return receipt


def _control_plane_launch_cluster(task: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = task.get("new_cluster")
    if isinstance(direct, Mapping):
        return direct
    cluster_spec = task.get("cluster_spec")
    if isinstance(cluster_spec, Mapping):
        nested = cluster_spec.get("new_cluster")
        if isinstance(nested, Mapping):
            return nested
    raise ValueError("runs/get task does not expose its launch cluster")


def _qualification_cluster(
    *, hardware_id: str, custom_tags: Mapping[str, str]
) -> dict[str, Any]:
    """Build a closed single-node cluster without widening V1 serving targets."""

    if hardware_id != GPU_QUALIFICATION_GENERATION_HARDWARE_ID:
        return build_single_node_gpu_cluster(
            DatabricksSingleNodeGPUClusterConfig(
                purpose=GPU_QUALIFICATION_DATABRICKS_PURPOSE,
                node_type_id=databricks_node_type_for_hardware_target(hardware_id),
                spark_version=DEFAULT_DATABRICKS_SPARK_VERSION,
                data_security_mode=GPU_QUALIFICATION_DATABRICKS_DATA_SECURITY_MODE,
                custom_tags=custom_tags,
            )
        )
    # g6e/L40S is qualification/producer-only.  Construct this one reviewed
    # shape locally rather than registering it as a benchmark serving target.
    tags = {
        "ResourceClass": "SingleNode",
        "purpose": GPU_QUALIFICATION_DATABRICKS_PURPOSE,
        **dict(custom_tags),
    }
    return {
        "spark_version": DEFAULT_DATABRICKS_SPARK_VERSION,
        "node_type_id": GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE,
        "driver_node_type_id": GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE,
        "data_security_mode": GPU_QUALIFICATION_DATABRICKS_DATA_SECURITY_MODE,
        "num_workers": 0,
        "spark_conf": {
            "spark.master": "local[*]",
            "spark.databricks.cluster.profile": "singleNode",
        },
        "custom_tags": tags,
        "aws_attributes": {"availability": "ON_DEMAND", "zone_id": "auto"},
    }


def execute_gpu_qualification_job(
    *,
    plan_record: Mapping[str, Any],
    expected_plan_sha256: str,
    job_id: str,
    reservation_attempt_id: str,
    runner_uri: str,
    package_wheel_uri: str,
    patched_vllm_wheel_uri: str,
    artifact_uris: Mapping[str, str],
    artifact_sha256: Mapping[str, str],
    output_json: str | Path,
    work_dir: str | Path,
    cloud_run_id: str,
    cloud_cluster_id: str,
    sentinel_runner: GPUQualificationSentinelRunner,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Execute, validate, and exclusively publish one planned GPU result.

    The callable boundary is intentionally in-process and is not configurable
    through the command line.  Production runners pass Cachet's reviewed
    sentinel dispatcher; tests can pass a deterministic implementation without
    pretending that CPU execution is GPU qualification.
    """

    if not callable(sentinel_runner):
        raise TypeError("sentinel_runner must be callable")
    plan, pins = _validated_plan_and_pins(plan_record)
    plan_digest = _required_sha256(
        plan.get("closed_record_sha256"), "plan.closed_record_sha256"
    )
    if _required_sha256(expected_plan_sha256, "expected_plan_sha256") != plan_digest:
        raise ValueError("expected plan SHA-256 does not match the closed plan")
    normalized_job_id = _safe_id(job_id, "job_id")
    planned_job = _planned_job(plan, normalized_job_id)
    expected_attempt_id = gpu_qualification_reservation_attempt_id(
        plan_digest, normalized_job_id
    )
    if reservation_attempt_id != expected_attempt_id:
        raise ValueError("reservation_attempt_id does not match the frozen plan/job")
    uris = _validated_artifact_uris(
        artifact_uris,
        runner_uri=runner_uri,
        package_wheel_uri=package_wheel_uri,
        patched_vllm_wheel_uri=patched_vllm_wheel_uri,
    )
    observed_pin_mapping = _validated_artifact_sha256(artifact_sha256)
    if observed_pin_mapping != pins.to_record():
        raise ValueError("artifact SHA-256 arguments do not match the plan")

    normalized_output_json = _validated_result_output_json(
        output_json,
        plan_digest=plan_digest,
        job_id=normalized_job_id,
    )
    output_path = _cluster_file_path(normalized_output_json)
    local_work_dir = _validated_local_work_dir(
        work_dir,
        plan_digest=plan_digest,
        job_id=normalized_job_id,
    )
    _require_fresh_output_path(output_path)
    _create_fresh_work_dir(local_work_dir)
    try:
        source_artifact_paths = {
            key: _cluster_file_path(uri) for key, uri in uris.items()
        }
        artifact_paths = _snapshot_artifacts_to_local_work(
            source_artifact_paths,
            expected=pins.to_record(),
            snapshot_root=local_work_dir / "artifact-snapshot",
        )

        started_clock = now or _utc_now
        started_at = _utc_timestamp(started_clock())
        measurements = sentinel_runner(
            plan_record=plan,
            planned_job=planned_job,
            artifact_paths=artifact_paths,
            work_dir=local_work_dir,
        )
        if not isinstance(measurements, Mapping):
            raise TypeError("sentinel runner must return a measurement mapping")
        normalized_measurements = _json_object(measurements, "measurements")
        finished_at = _utc_timestamp(started_clock())
        if finished_at <= started_at:
            # A canonical job record requires a strict interval.  Wall-clock
            # resolution can be coarse on managed images, so obtain a later sample
            # instead of manufacturing a timestamp.
            finished_at = _utc_timestamp(started_clock())
        if finished_at <= started_at:
            raise RuntimeError("GPU sentinel timestamps did not advance")

        runtime = _observe_gpu_runtime(local_work_dir)
        record = build_gpu_job_result(
            plan_record=plan,
            job_id=normalized_job_id,
            reservation_attempt_id=expected_attempt_id,
            task_key=_task_key(normalized_job_id),
            output_json=normalized_output_json,
            cloud_run_id=_non_empty_string(cloud_run_id, "cloud_run_id"),
            cloud_cluster_id=_non_empty_string(cloud_cluster_id, "cloud_cluster_id"),
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            nvidia_driver_version=runtime["nvidia_driver_version"],
            observed_gpu=runtime["gpu"],
            observed_gpu_compute_capability=runtime["gpu_compute_capability"],
            observed_vllm_version=runtime["vllm_version"],
            observed_torch_cuda_version=runtime["torch_cuda_version"],
            observed_artifact_sha256=observed_pin_mapping,
            measurements=normalized_measurements,
        )
        validate_gpu_job_result_record(
            record,
            plan_record=plan,
            expected_artifact_pins=pins,
        )
        # A terminal task must never leave a valid SUCCESS result behind when
        # cleanup itself fails.  Make read-only runtime directories removable,
        # complete cleanup, and only then seal the durable result.
        _remove_success_work_dir(local_work_dir)
        _write_canonical_exclusive(record, output_path)
        return record
    except BaseException:
        # Failed attempts retain node-local diagnostics and never publish a
        # canonical SUCCESS record.
        raise


def write_gpu_qualification_bootstrap_runner(path: str | Path) -> str:
    """Write the exact stdlib-only pre-install runner and return its digest."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"bootstrap runner already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o750)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256


def pins_from_plan_record(
    plan_record: Mapping[str, Any],
) -> GPUQualificationArtifactPins:
    """Extract the exact artifact pins from a closed qualification plan."""

    runtime_contract = _required_mapping(
        plan_record.get("runtime_contract"), "plan.runtime_contract"
    )
    raw_pins = _required_mapping(
        runtime_contract.get("artifact_sha256"),
        "plan.runtime_contract.artifact_sha256",
    )
    if frozenset(raw_pins) != frozenset(GPU_QUALIFICATION_ARTIFACT_KEYS):
        raise ValueError("plan artifact_sha256 mapping must use the closed key set")
    return GPUQualificationArtifactPins(
        runtime_lock_sha256=_required_sha256(
            raw_pins.get("runtime_lock_sha256"), "runtime_lock_sha256"
        ),
        patched_vllm_wheel_sha256=_required_sha256(
            raw_pins.get("patched_vllm_wheel_sha256"),
            "patched_vllm_wheel_sha256",
        ),
        package_wheel_sha256=_required_sha256(
            raw_pins.get("package_wheel_sha256"), "package_wheel_sha256"
        ),
        cachet_source_tree_sha256=_required_sha256(
            raw_pins.get("cachet_source_tree_sha256"),
            "cachet_source_tree_sha256",
        ),
        runner_sha256=_required_sha256(raw_pins.get("runner_sha256"), "runner_sha256"),
        input_bundle_sha256=_required_sha256(
            raw_pins.get("input_bundle_sha256"), "input_bundle_sha256"
        ),
    )


def _validated_plan_and_pins(
    plan_record: Mapping[str, Any],
) -> tuple[dict[str, Any], GPUQualificationArtifactPins]:
    if not isinstance(plan_record, Mapping):
        raise TypeError("plan_record must be a mapping")
    plan = _json_object(plan_record, "plan_record")
    if plan.get("record_type") != GPU_QUALIFICATION_PLAN_RECORD_TYPE:
        raise ValueError("unexpected GPU qualification plan record_type")
    campaign_id = _non_empty_string(plan.get("campaign_id"), "campaign_id")
    pins = pins_from_plan_record(plan)
    if pins.runner_sha256 != GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256:
        raise ValueError(
            "plan runner_sha256 does not identify the reviewed bootstrap runner"
        )
    if pins.input_bundle_sha256 != (GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256):
        raise ValueError(
            "qualification jobs require the frozen 7ff6 publication input bundle"
        )
    if pins.patched_vllm_wheel_sha256 != GPU_QUALIFICATION_PATCHED_WHEEL_SHA256:
        raise ValueError(
            "qualification jobs require the reviewed 65120c48 patched vLLM wheel"
        )
    if pins.runtime_lock_sha256 != VLLM_RUNTIME_LOCK_SHA256:
        raise ValueError(
            "qualification jobs require the reviewed packaged runtime lock"
        )
    validate_gpu_qualification_plan_record(
        plan,
        expected_campaign_id=campaign_id,
        expected_artifact_pins=pins,
    )
    return plan, pins


def _validated_artifact_uris(
    artifact_uris: Mapping[str, str],
    *,
    runner_uri: str,
    package_wheel_uri: str,
    patched_vllm_wheel_uri: str,
) -> dict[str, str]:
    if not isinstance(artifact_uris, Mapping):
        raise TypeError("artifact_uris must be a mapping")
    if frozenset(artifact_uris) != frozenset(GPU_QUALIFICATION_ARTIFACT_KEYS):
        raise ValueError("artifact_uris must use the closed artifact key set")
    normalized = {
        key: _validated_cluster_artifact_uri(artifact_uris[key], key)
        for key in GPU_QUALIFICATION_ARTIFACT_KEYS
    }
    repeated = {
        "runner_sha256": _validated_cluster_artifact_uri(runner_uri, "runner_uri"),
        "package_wheel_sha256": _validated_cluster_artifact_uri(
            package_wheel_uri, "package_wheel_uri"
        ),
        "patched_vllm_wheel_sha256": _validated_cluster_artifact_uri(
            patched_vllm_wheel_uri, "patched_vllm_wheel_uri"
        ),
    }
    for key, value in repeated.items():
        if normalized[key] != value:
            raise ValueError(f"{key} URI does not match its dedicated argument")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("artifact URI roles must be distinct and cannot be conflated")
    return normalized


def _validated_artifact_sha256(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("artifact_sha256 must be a mapping")
    if frozenset(value) != frozenset(GPU_QUALIFICATION_ARTIFACT_KEYS):
        raise ValueError("artifact_sha256 must use the closed artifact key set")
    return {
        key: _required_sha256(value[key], f"artifact_sha256.{key}")
        for key in GPU_QUALIFICATION_ARTIFACT_KEYS
    }


def validate_gpu_qualification_submission_rejection_record(
    record: Mapping[str, Any],
    *,
    plan_record: Mapping[str, Any],
) -> None:
    """Validate one immutable, pre-run Databricks submission rejection record."""

    normalized = _json_object(record, "GPU qualification submission rejection")
    if frozenset(normalized) != _QUALIFICATION_SUBMISSION_REJECTION_KEYS:
        raise ValueError(
            "GPU qualification submission rejection does not use the closed schema"
        )
    if normalized["record_type"] != GPU_QUALIFICATION_SUBMISSION_REJECTION_RECORD_TYPE:
        raise ValueError("GPU qualification submission rejection identity differs")
    if type(normalized["schema_version"]) is not int or normalized["schema_version"] != 1:
        raise ValueError("GPU qualification submission rejection schema differs")
    _require_closed_record_digest(
        normalized,
        "GPU qualification submission rejection",
    )
    validated_plan = _json_object(plan_record, "historical GPU qualification plan")
    if (
        validated_plan.get("record_type") != GPU_QUALIFICATION_PLAN_RECORD_TYPE
        or type(validated_plan.get("schema_version")) is not int
        or validated_plan.get("schema_version") != GPU_QUALIFICATION_SCHEMA_VERSION
    ):
        raise ValueError("historical GPU qualification plan identity differs")
    _require_closed_record_digest(
        validated_plan,
        "historical GPU qualification plan",
    )
    plan_sha256 = _required_sha256(
        validated_plan.get("closed_record_sha256"),
        "plan_record.closed_record_sha256",
    )
    if normalized["plan_sha256"] != plan_sha256:
        raise ValueError("submission rejection plan SHA-256 differs")
    attempts = normalized["attempt_ids"]
    expected_attempts = [
        gpu_qualification_reservation_attempt_id(
            plan_sha256,
            _safe_id(job.get("job_id"), "cloud job ID"),
        )
        for job in _planned_jobs(validated_plan)
    ]
    if attempts != expected_attempts:
        raise ValueError("submission rejection attempt IDs differ from the plan")
    for field_name in (
        "batch_marker_file_sha256",
        "first_post_intent_file_sha256",
        "submit_payloads_file_sha256",
    ):
        _required_sha256(normalized[field_name], field_name)
    if normalized["failed_before_run_creation"] is not True:
        raise ValueError("submission rejection must precede run creation")
    http_status = _positive_int(normalized["http_status"], "http_status")
    if not 400 <= http_status < 500:
        raise ValueError("submission rejection must carry a client-error HTTP status")
    observed_bytes = _positive_int(
        normalized["observed_parameters_json_bytes"],
        "observed_parameters_json_bytes",
    )
    server_limit = _positive_int(
        normalized["server_parameters_json_limit_bytes"],
        "server_parameters_json_limit_bytes",
    )
    if observed_bytes <= server_limit:
        raise ValueError("submission rejection must exceed the server parameter limit")
    if (
        _nonnegative_int(
            normalized["remote_active_runs_observed"],
            "remote_active_runs_observed",
        )
        != 0
    ):
        raise ValueError("submission rejection must observe zero remote active runs")
    if (
        _nonnegative_int(
            normalized["reconciled_actual_gpu_seconds_per_attempt"],
            "reconciled_actual_gpu_seconds_per_attempt",
        )
        != 0
    ):
        raise ValueError("submission rejection must reconcile zero GPU seconds")
    _non_empty_string(normalized["server_reason"], "server_reason")
    _parse_utc_timestamp(normalized["rejected_at_utc"], "rejected_at_utc")


def _encode_qualification_plan_parameter(canonical_plan: str) -> str:
    if not isinstance(canonical_plan, str):
        raise TypeError("canonical_plan must be a string")
    canonical_bytes = canonical_plan.encode("utf-8")
    if len(canonical_bytes) > _QUALIFICATION_PLAN_MAX_CANONICAL_BYTES:
        raise ValueError("canonical qualification plan exceeds the decoded size cap")
    encoded = base64.urlsafe_b64encode(
        zlib.compress(canonical_bytes, level=_QUALIFICATION_PLAN_ZLIB_LEVEL)
    ).decode("ascii")
    if len(encoded) > _QUALIFICATION_PLAN_MAX_ENCODED_CHARS:
        raise ValueError("encoded qualification plan exceeds the transport size cap")
    return encoded


def _decode_qualification_plan_parameter(
    encoded_plan: str,
    *,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    expected_digest = _required_sha256(
        expected_plan_sha256,
        "expected_plan_sha256",
    )
    if not isinstance(encoded_plan, str) or not encoded_plan:
        raise ValueError("encoded qualification plan must be a non-empty string")
    if len(encoded_plan) > _QUALIFICATION_PLAN_MAX_ENCODED_CHARS:
        raise ValueError("encoded qualification plan exceeds the transport size cap")
    try:
        encoded_bytes = encoded_plan.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("encoded qualification plan must be ASCII") from exc
    try:
        compressed = base64.b64decode(
            encoded_bytes,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise ValueError("encoded qualification plan is not strict base64url") from exc
    if base64.urlsafe_b64encode(compressed) != encoded_bytes:
        raise ValueError("encoded qualification plan is not canonical base64url")
    decompressor = zlib.decompressobj()
    try:
        canonical_bytes = decompressor.decompress(
            compressed,
            _QUALIFICATION_PLAN_MAX_CANONICAL_BYTES + 1,
        )
        if (
            len(canonical_bytes) > _QUALIFICATION_PLAN_MAX_CANONICAL_BYTES
            or decompressor.unconsumed_tail
        ):
            raise ValueError(
                "decoded qualification plan exceeds the canonical size cap"
            )
        canonical_bytes += decompressor.flush()
    except zlib.error as exc:
        raise ValueError("encoded qualification plan is not a valid zlib stream") from exc
    if (
        len(canonical_bytes) > _QUALIFICATION_PLAN_MAX_CANONICAL_BYTES
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ValueError("encoded qualification plan has an invalid zlib closure")
    try:
        canonical_plan = canonical_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("decoded qualification plan is not UTF-8") from exc
    try:
        decoded = json.loads(canonical_plan)
    except json.JSONDecodeError as exc:
        raise ValueError("decoded qualification plan is not JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("decoded qualification plan must contain an object")
    plan = dict(decoded)
    if canonical_gpu_qualification_json(plan) != canonical_plan:
        raise ValueError("decoded qualification plan is not canonical JSON")
    if plan.get("closed_record_sha256") != expected_digest:
        raise ValueError("decoded qualification plan SHA-256 differs from expectation")
    validated_plan, _pins = _validated_plan_and_pins(plan)
    return validated_plan


def _qualification_parameters_json_bytes(parameters: Sequence[str]) -> int:
    return len(
        json.dumps(
            list(parameters),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _require_qualification_parameters_size(parameters: Sequence[str]) -> None:
    observed_bytes = _qualification_parameters_json_bytes(parameters)
    if observed_bytes > GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES:
        raise ValueError(
            "qualification parameters JSON exceeds the 9500-byte safety cap: "
            f"{observed_bytes} bytes"
        )


def _runner_parameters(
    *,
    encoded_plan: str,
    plan_digest: str,
    job_id: str,
    output_json: str,
    work_dir: str,
    runner_uri: str,
    package_wheel_uri: str,
    patched_vllm_wheel_uri: str,
    artifact_uris: Mapping[str, str],
    artifact_pins: GPUQualificationArtifactPins,
    reservation_attempt_id: str,
) -> list[str]:
    parameters = [
        _QUALIFICATION_PLAN_PARAMETER_OPTION,
        encoded_plan,
        "--expected-plan-sha256",
        plan_digest,
        "--job-id",
        job_id,
        "--reservation-attempt-id",
        reservation_attempt_id,
        "--cloud-run-id",
        _DATABRICKS_RUN_ID_TEMPLATE,
        "--attempt-number",
        "0",
        "--retry-count",
        "0",
        "--runner-uri",
        runner_uri,
        "--package-wheel-uri",
        package_wheel_uri,
        "--patched-vllm-wheel-uri",
        patched_vllm_wheel_uri,
        "--output-json",
        output_json,
        "--work-dir",
        work_dir,
    ]
    pin_mapping = artifact_pins.to_record()
    for key in GPU_QUALIFICATION_ARTIFACT_KEYS:
        parameters.extend(("--artifact-uri", f"{key}={artifact_uris[key]}"))
        parameters.extend(("--artifact-sha256", f"{key}={pin_mapping[key]}"))
    _require_qualification_parameters_size(parameters)
    return parameters


def _planned_jobs(plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    cloud = _required_mapping(plan.get("cloud_qualification"), "cloud_qualification")
    jobs = cloud.get("jobs")
    if not isinstance(jobs, Sequence) or isinstance(jobs, (str, bytes, bytearray)):
        raise ValueError("cloud_qualification.jobs must be an array")
    normalized: list[Mapping[str, Any]] = []
    for index, job in enumerate(jobs):
        normalized.append(_required_mapping(job, f"cloud job {index}"))
    return tuple(normalized)


def _planned_job(plan: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    matches = [job for job in _planned_jobs(plan) if job.get("job_id") == job_id]
    if len(matches) != 1:
        raise ValueError(f"job_id is not unique in the plan: {job_id!r}")
    return matches[0]


def _observe_gpu_runtime(work_dir: Path) -> dict[str, str]:
    """Read runtime identity from the sentinel's isolated Python executable."""

    runtime_python = work_dir / "runtime" / "bin" / "python"
    if not runtime_python.is_file() or runtime_python.is_symlink():
        raise RuntimeError(
            "sentinel did not materialize the required isolated runtime Python at "
            f"{runtime_python}"
        )
    probe = (
        "import json, torch, vllm; "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'gpu':p.name,'gpu_compute_capability':"
        "f'{p.major}.{p.minor}','torch_cuda_version':torch.version.cuda,"
        "'vllm_version':vllm.__version__},sort_keys=True))"
    )
    completed = subprocess.run(
        [str(runtime_python), "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    observed = json.loads(completed.stdout)
    if not isinstance(observed, dict):
        raise RuntimeError("GPU runtime identity probe did not return an object")
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    driver_versions = {item.strip() for item in driver if item.strip()}
    if len(driver_versions) != 1:
        raise RuntimeError("GPU job must observe one NVIDIA driver version")
    result = {
        "gpu": _non_empty_string(observed.get("gpu"), "observed GPU"),
        "gpu_compute_capability": _non_empty_string(
            observed.get("gpu_compute_capability"),
            "observed GPU compute capability",
        ),
        "torch_cuda_version": _non_empty_string(
            observed.get("torch_cuda_version"), "observed torch CUDA version"
        ),
        "vllm_version": _non_empty_string(
            observed.get("vllm_version"), "observed vLLM version"
        ),
        "nvidia_driver_version": next(iter(driver_versions)),
    }
    return result


def _verify_artifact_files(
    artifact_paths: Mapping[str, Path], *, expected: Mapping[str, str]
) -> None:
    for key in GPU_QUALIFICATION_ARTIFACT_KEYS:
        path = artifact_paths[key]
        if key == "input_bundle_sha256":
            _verify_input_bundle_byte_closure(path, expected_sha256=expected[key])
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"artifact {key} is not one regular file: {path}")
        observed = _file_sha256(path)
        if observed != expected[key]:
            raise ValueError(
                f"artifact {key} SHA-256 mismatch: expected {expected[key]}, "
                f"found {observed}"
            )


def _snapshot_artifacts_to_local_work(
    source_paths: Mapping[str, Path],
    *,
    expected: Mapping[str, str],
    snapshot_root: Path,
) -> dict[str, Path]:
    """Materialize the durable closure once, then execute only from local bytes."""

    if snapshot_root.exists() or snapshot_root.is_symlink():
        raise FileExistsError(f"artifact snapshot already exists: {snapshot_root}")
    snapshot_root.mkdir(parents=False, exist_ok=False)
    snapshots: dict[str, Path] = {}
    for key in GPU_QUALIFICATION_ARTIFACT_KEYS:
        source = source_paths[key]
        _require_no_symlink_ancestors(
            source,
            label=f"artifact {key} source path",
            include_leaf=True,
        )
        role_root = snapshot_root / key
        if key == "input_bundle_sha256":
            if not source.is_dir() or source.is_symlink():
                raise ValueError("input bundle source must be one regular directory")
            shutil.copytree(source, role_root, symlinks=True)
            destination = role_root
        else:
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"artifact {key} source must be one regular file")
            if not source.name or source.name in {".", ".."}:
                raise ValueError(f"artifact {key} source filename is unsafe")
            role_root.mkdir()
            destination = role_root / source.name
            shutil.copyfile(source, destination, follow_symlinks=False)
        snapshots[key] = destination
    _verify_artifact_files(snapshots, expected=expected)
    _make_tree_read_only(snapshot_root)
    return snapshots


def _make_tree_read_only(root: Path) -> None:
    for current_root, directories, files in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in files:
            child = current / name
            if not child.is_symlink():
                child.chmod(
                    child.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                )
        for name in directories:
            child = current / name
            if not child.is_symlink():
                child.chmod(
                    child.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                )
    root.chmod(root.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _verify_input_bundle_byte_closure(
    path: Path,
    *,
    expected_sha256: str,
) -> str:
    """Verify the frozen directory/provenance/raw-byte closure without ML deps.

    The tokenizer-aware invariant check intentionally runs later inside the
    hash-locked isolated runtime.  This early check is stdlib-only so the
    Databricks bootstrap interpreter cannot silently supply ambient
    Transformers behavior.
    """

    expected_digest = _required_sha256(expected_sha256, "input bundle closure SHA-256")
    if not path.is_dir() or path.is_symlink():
        raise ValueError(
            "input_bundle_sha256 must identify a verified input-bundle directory"
        )
    provenance_path = path / _INPUT_PROVENANCE_FILENAME
    if not provenance_path.is_file() or provenance_path.is_symlink():
        raise ValueError("input bundle is missing its regular provenance file")
    provenance = _canonical_json_object_from_bytes(
        provenance_path.read_bytes(),
        pretty=True,
        label="input bundle provenance",
    )
    if frozenset(provenance) != _INPUT_PROVENANCE_FIELDS:
        raise ValueError("input bundle provenance does not use the closed schema")
    if provenance.get("record_type") != "cachet.main_latency_inputs":
        raise ValueError("input bundle provenance record_type is unsupported")
    if provenance.get("schema_version") != 3:
        raise ValueError("input bundle provenance schema_version is unsupported")
    if provenance.get("protocol") != _INPUT_PROTOCOL:
        raise ValueError("input bundle protocol pins do not match qualification")
    closed_digest = _required_sha256(
        provenance.get("closed_record_sha256"),
        "input bundle closed_record_sha256",
    )
    unsigned = dict(provenance)
    unsigned.pop("closed_record_sha256")
    if _canonical_json_sha256(unsigned) != closed_digest:
        raise ValueError("input bundle provenance closure digest mismatch")

    raw_outputs = provenance.get("outputs")
    if not isinstance(raw_outputs, list):
        raise ValueError("input bundle provenance outputs must be an array")
    expected_order = [
        (dataset, target, segment_count)
        for target, segment_count in _INPUT_TARGET_SEGMENT_COUNTS
        for dataset in _INPUT_DATASETS
    ]
    if len(raw_outputs) != len(expected_order):
        raise ValueError("input bundle must describe exactly twelve output shards")
    manifest: list[dict[str, str]] = []
    expected_files = {_INPUT_PROVENANCE_FILENAME}
    for index, (raw_output, expected_output) in enumerate(
        zip(raw_outputs, expected_order, strict=True)
    ):
        if not isinstance(raw_output, dict):
            raise ValueError(f"input bundle output {index} must be an object")
        if frozenset(raw_output) != _INPUT_OUTPUT_FIELDS:
            raise ValueError(
                f"input bundle output {index} does not use the closed schema"
            )
        dataset, target, segment_count = expected_output
        relative_path = f"{target}/{dataset}.jsonl"
        if (
            raw_output.get("dataset") != dataset
            or raw_output.get("input_tokens_target") != target
            or raw_output.get("segment_count") != segment_count
            or raw_output.get("relative_path") != relative_path
            or raw_output.get("record_count") != _INPUT_EXAMPLES_PER_DATASET
        ):
            raise ValueError(
                f"input bundle output {index} does not match the frozen shard layout"
            )
        byte_count = raw_output.get("byte_count")
        if type(byte_count) is not int or byte_count <= 0:
            raise ValueError(f"input bundle output {index} has invalid byte_count")
        jsonl_digest = _required_sha256(
            raw_output.get("jsonl_sha256"),
            f"input bundle output {index} jsonl_sha256",
        )
        raw_records = raw_output.get("records")
        if not isinstance(raw_records, list) or len(raw_records) != (
            _INPUT_EXAMPLES_PER_DATASET
        ):
            raise ValueError(
                f"input bundle output {index} must describe exactly 32 records"
            )
        records_digest = _required_sha256(
            raw_output.get("records_sha256"),
            f"input bundle output {index} records_sha256",
        )
        if _canonical_json_sha256(raw_records) != records_digest:
            raise ValueError(
                f"input bundle output {index} records_sha256 does not close records"
            )

        shard_path = path / PurePosixPath(relative_path)
        if not shard_path.is_file() or shard_path.is_symlink():
            raise ValueError(
                f"input bundle shard is not one regular file: {relative_path}"
            )
        shard_bytes = shard_path.read_bytes()
        if len(shard_bytes) != byte_count:
            raise ValueError(f"input bundle shard byte count mismatch: {relative_path}")
        if sha256(shard_bytes).hexdigest() != jsonl_digest:
            raise ValueError(f"input bundle shard SHA-256 mismatch: {relative_path}")
        _verify_canonical_input_jsonl(
            shard_bytes,
            dataset=dataset,
            relative_path=relative_path,
        )
        expected_files.add(relative_path)
        manifest.append({"jsonl_sha256": jsonl_digest, "relative_path": relative_path})

    outputs_digest = _required_sha256(
        provenance.get("outputs_sha256"), "input bundle outputs_sha256"
    )
    if _canonical_json_sha256(raw_outputs) != outputs_digest:
        raise ValueError("input bundle outputs_sha256 does not close outputs")
    observed_bundle_digest = _canonical_json_sha256(manifest)
    if (
        _required_sha256(provenance.get("bundle_sha256"), "input bundle bundle_sha256")
        != observed_bundle_digest
    ):
        raise ValueError("input bundle manifest digest does not match provenance")
    if observed_bundle_digest != expected_digest:
        raise ValueError(
            "input bundle closure digest mismatch: expected "
            f"{expected_digest}, found {observed_bundle_digest}"
        )
    _verify_closed_input_directory(path, expected_files=expected_files)
    return observed_bundle_digest


def _verify_canonical_input_jsonl(
    content: bytes,
    *,
    dataset: str,
    relative_path: str,
) -> None:
    if not content or not content.endswith(b"\n"):
        raise ValueError(
            f"input bundle shard is not newline-terminated: {relative_path}"
        )
    lines = content[:-1].split(b"\n")
    if len(lines) != _INPUT_EXAMPLES_PER_DATASET or any(not line for line in lines):
        raise ValueError(
            f"input bundle shard must contain exactly 32 rows: {relative_path}"
        )
    example_ids: set[str] = set()
    for row_index, line in enumerate(lines, start=1):
        record = _canonical_json_object_from_bytes(
            line,
            pretty=False,
            label=f"{relative_path} row {row_index}",
        )
        if record.get("dataset") != dataset:
            raise ValueError(f"input bundle row dataset mismatch: {relative_path}")
        example_id = record.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"input bundle row example_id is invalid: {relative_path}")
        if example_id in example_ids:
            raise ValueError(
                f"input bundle row example_id is duplicated: {relative_path}"
            )
        example_ids.add(example_id)


def _verify_closed_input_directory(path: Path, *, expected_files: set[str]) -> None:
    expected_directories = {str(target) for target, _ in _INPUT_TARGET_SEGMENT_COUNTS}
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for root, directory_names, file_names in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in directory_names:
            child = root_path / name
            if child.is_symlink():
                raise ValueError(f"input bundle contains a symlink directory: {child}")
            observed_directories.add(child.relative_to(path).as_posix())
        for name in file_names:
            child = root_path / name
            if not child.is_file() or child.is_symlink():
                raise ValueError(f"input bundle contains a non-regular file: {child}")
            observed_files.add(child.relative_to(path).as_posix())
    if observed_directories != expected_directories or observed_files != expected_files:
        raise ValueError("input bundle directory is not the closed twelve-shard layout")


def _canonical_json_object_from_bytes(
    content: bytes,
    *,
    pretty: bool,
    label: str,
) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate object key {key!r}")
            result[key] = value
        return result

    try:
        decoded = content.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    if content != _canonical_stdlib_json_bytes(value, pretty=pretty):
        raise ValueError(f"{label} is not canonically encoded")
    return value


def _canonical_stdlib_json_bytes(value: Any, *, pretty: bool) -> bytes:
    kwargs: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    suffix = "\n" if pretty else ""
    return (json.dumps(value, allow_nan=False, **kwargs) + suffix).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return sha256(_canonical_stdlib_json_bytes(value, pretty=False)).hexdigest()


def _seal_record(record: dict[str, Any]) -> None:
    if "closed_record_sha256" not in record:
        raise ValueError("sealed record is missing closed_record_sha256")
    unsigned = dict(record)
    unsigned.pop("closed_record_sha256")
    record["closed_record_sha256"] = _canonical_json_sha256(unsigned)


def _require_closed_record_digest(record: Mapping[str, Any], field_name: str) -> str:
    observed = _required_sha256(
        record.get("closed_record_sha256"),
        f"{field_name}.closed_record_sha256",
    )
    unsigned = dict(record)
    unsigned.pop("closed_record_sha256")
    if _canonical_json_sha256(unsigned) != observed:
        raise ValueError(f"{field_name} closed_record_sha256 mismatch")
    return observed


def _read_canonical_json_object_file(path: str | Path, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{label} must be one regular file")
    content = candidate.read_bytes()
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError(f"{label} must contain one newline-terminated JSON object")
    record = _canonical_json_object_from_bytes(
        content[:-1],
        pretty=False,
        label=label,
    )
    expected = (canonical_gpu_qualification_json(record) + "\n").encode("utf-8")
    if content != expected:
        raise ValueError(f"{label} is not canonically encoded")
    return record


def _create_fresh_controller_evidence_root(value: str | Path) -> Path:
    directory = _validated_fresh_controller_evidence_root(value)
    directory.mkdir(parents=True, exist_ok=False)
    if directory.is_symlink():  # pragma: no cover - concurrent filesystem attack.
        raise ValueError("controller evidence root cannot be a symlink")
    _fsync_directory(directory)
    _fsync_directory(directory.parent)
    return directory


def _validated_fresh_controller_evidence_root(value: str | Path) -> Path:
    raw = str(value)
    directory = Path(raw)
    if (
        not directory.is_absolute()
        or directory != Path(os.path.normpath(raw))
        or directory == Path("/")
    ):
        raise ValueError("controller evidence root must be a normalized absolute path")
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"controller evidence root already exists: {directory}")
    _require_no_symlink_ancestors(
        directory, label="controller evidence root", include_leaf=True
    )
    return directory


def _validated_existing_controller_evidence_root(value: str | Path, label: str) -> Path:
    raw = str(value)
    directory = Path(raw)
    if (
        not directory.is_absolute()
        or directory != Path(os.path.normpath(raw))
        or directory == Path("/")
    ):
        raise ValueError(f"{label} must be a normalized absolute path")
    _require_no_symlink_ancestors(directory, label=label, include_leaf=True)
    if not directory.is_dir():
        raise ValueError(f"{label} must be one regular directory")
    return directory


def _validated_existing_regular_file(value: str | Path, label: str) -> Path:
    raw = str(value)
    path = Path(raw)
    if not path.is_absolute() or path != Path(os.path.normpath(raw)):
        raise ValueError(f"{label} must be a normalized absolute path")
    _require_no_symlink_ancestors(path, label=label, include_leaf=True)
    if not path.is_file():
        raise ValueError(f"{label} must be one regular file")
    return path


def _require_no_symlink_ancestors(
    path: Path, *, label: str, include_leaf: bool
) -> None:
    candidates = ((path,) if include_leaf else ()) + tuple(path.parents)
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"{label} cannot traverse a symlink: {candidate}")


def _publish_terminal_receipts_atomic(
    root: Path, receipts: Sequence[Mapping[str, Any]]
) -> None:
    """Publish only the complete terminal closure, never a partial job prefix."""

    _validated_fresh_controller_evidence_root(root)
    closure_digest = _canonical_json_sha256({"receipts": list(receipts)})
    staging = root.with_name(f".{root.name}.staging-{closure_digest[:16]}")
    staging_root = _create_fresh_controller_evidence_root(staging)
    try:
        for receipt in receipts:
            job_id = _safe_id(receipt.get("job_id"), "terminal receipt job_id")
            _write_canonical_exclusive(receipt, staging_root / f"{job_id}.json")
        _fsync_directory(staging_root)
        os.rename(staging_root, root)
        _fsync_directory(root.parent)
    except BaseException:
        if staging_root.exists() and not staging_root.is_symlink():
            shutil.rmtree(staging_root)
            _fsync_directory(staging_root.parent)
        raise


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_canonical_exclusive(record: Mapping[str, Any], path: Path) -> None:
    _require_no_symlink_ancestors(
        path, label="canonical output path", include_leaf=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(
        path, label="canonical output path", include_leaf=True
    )
    content = (canonical_gpu_qualification_json(record) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        path.unlink(missing_ok=True)
        if path.parent.is_dir() and not path.parent.is_symlink():
            _fsync_directory(path.parent)
        raise


def _fsync_directory(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"directory durability target is invalid: {path}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_fresh_output_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"GPU qualification output already exists: {path}")


def _create_fresh_work_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(
            f"GPU qualification work directory already exists: {path}"
        )
    _require_no_symlink_ancestors(
        path,
        label="GPU qualification local-work path",
        include_leaf=True,
    )
    path.mkdir(parents=True, exist_ok=False)
    _require_no_symlink_ancestors(
        path,
        label="GPU qualification local-work path",
        include_leaf=True,
    )


def _remove_success_work_dir(path: Path) -> None:
    """Remove read-only success workspaces without weakening failure diagnostics."""

    _require_no_symlink_ancestors(
        path,
        label="GPU qualification local-work path",
        include_leaf=True,
    )
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError("successful qualification work directory is unavailable")
    for current_root, directories, files in os.walk(path, followlinks=False):
        current = Path(current_root)
        current.chmod(
            current.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
        )
        for name in directories:
            child = current / name
            if not child.is_symlink():
                child.chmod(
                    child.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
                )
        for name in files:
            child = current / name
            if not child.is_symlink():
                child.chmod(child.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR)
    shutil.rmtree(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError("successful qualification work directory was not removed")


def _expected_local_work_dir(plan_digest: str, job_id: str) -> Path:
    root = Path(GPU_QUALIFICATION_LOCAL_WORK_ROOT)
    if not root.is_absolute() or str(root) in {"/", "/local_disk0"}:
        raise ValueError("GPU qualification local-work root is unsafe")
    return (
        root / _required_sha256(plan_digest, "plan digest") / _safe_id(job_id, "job_id")
    )


def _validated_local_work_dir(
    value: str | Path,
    *,
    plan_digest: str,
    job_id: str,
) -> Path:
    raw = str(value)
    parsed = urlsplit(raw)
    if raw.startswith("dbfs:/") or parsed.scheme:
        raise ValueError(
            "GPU qualification work_dir must be a node-local absolute path, not a URI"
        )
    path = Path(raw)
    if not path.is_absolute() or path != Path(os.path.normpath(raw)):
        raise ValueError(
            "GPU qualification work_dir must be a normalized absolute path"
        )
    expected = _expected_local_work_dir(plan_digest, job_id)
    if path != expected:
        raise ValueError(
            "GPU qualification work_dir must equal the frozen node-local plan/job path"
        )
    return path


def _validated_cluster_artifact_uri(value: Any, field_name: str) -> str:
    raw = _non_empty_string(value, field_name)
    if raw.startswith("dbfs:/"):
        path = PurePosixPath("/", raw.removeprefix("dbfs:/").lstrip("/"))
        _reject_unsafe_parts(path, field_name)
        return "dbfs:/" + path.as_posix().lstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise ValueError(f"{field_name} has an unsupported file URI")
        path = PurePosixPath(unquote(parsed.path))
        _reject_unsafe_parts(path, field_name)
        if not _is_durable_cluster_path(path):
            raise ValueError(f"{field_name} file URI must use DBFS or a UC Volume")
        return raw
    if parsed.scheme:
        raise ValueError(
            f"{field_name} must be a dbfs:/ URI, file URI, or absolute path"
        )
    path = PurePosixPath(raw)
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be cluster-visible and absolute")
    _reject_unsafe_parts(path, field_name)
    if not _is_durable_cluster_path(path):
        raise ValueError(f"{field_name} must use DBFS or a UC Volume")
    return path.as_posix()


def _validated_output_root(value: Any) -> str:
    root = _validated_cluster_artifact_uri(value, "output_root").rstrip("/")
    if not root:
        raise ValueError("output_root must not be the filesystem root")
    if root in {"dbfs:", "file:", "/"}:
        raise ValueError("output_root must not be a broad filesystem root")
    return root


def _validated_result_output_json(
    value: str | Path,
    *,
    plan_digest: str,
    job_id: str,
) -> str:
    normalized = _validated_cluster_artifact_uri(value, "output_json")
    cluster_path = _cluster_file_path(normalized)
    expected_suffix = (
        _required_sha256(plan_digest, "plan_digest"),
        _safe_id(job_id, "job_id"),
        GPU_QUALIFICATION_OUTPUT_FILENAME,
    )
    if tuple(cluster_path.parts[-3:]) != expected_suffix:
        raise ValueError("output_json must use the frozen plan/job result path")
    return normalized


def _is_durable_cluster_path(path: PurePosixPath) -> bool:
    parts = path.parts
    return len(parts) >= 3 and (
        parts[:2] == ("/", "dbfs") or parts[:2] == ("/", "Volumes")
    )


def _cluster_file_path(value: str | Path) -> Path:
    raw = str(value)
    if raw.startswith("dbfs:/"):
        return Path("/dbfs", raw.removeprefix("dbfs:/").lstrip("/"))
    parsed = urlsplit(raw)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("file URI authority must be empty or localhost")
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"unsupported cluster artifact URI scheme: {parsed.scheme}")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("cluster path must be absolute")
    return path


def _join_cluster_uri(root: str, *parts: str) -> str:
    suffix = "/".join(_safe_id(part, "output path component") for part in parts)
    return f"{root.rstrip('/')}/{suffix}"


def _reject_unsafe_parts(path: PurePosixPath, field_name: str) -> None:
    if not path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts[1:]
    ):
        raise ValueError(f"{field_name} contains an unsafe path")


def _run_name(campaign_id: Any, job_id: str) -> str:
    campaign = _safe_id(campaign_id, "campaign_id")
    return f"cachet-gpu-qualification-{campaign}-{job_id}"[:4096]


def _task_key(job_id: str) -> str:
    value = "gpu_qualification_" + re.sub(r"[^a-zA-Z0-9_]", "_", job_id)
    if not value[0].isalpha() or len(value) > 100:
        raise ValueError(f"job_id cannot form a Databricks task key: {job_id!r}")
    return value


def gpu_qualification_reservation_attempt_id(plan_sha256: str, job_id: str) -> str:
    """Return the deterministic ledger attempt ID embedded in one submit payload."""

    digest = _required_sha256(plan_sha256, "plan_sha256")
    normalized_job_id = _safe_id(job_id, "job_id")
    return f"gpuq-{digest[:16]}-{normalized_job_id}"


def _safe_tag_value(value: Any) -> str:
    normalized = _safe_id(value, "tag value")
    if len(normalized) > 255:
        raise ValueError("Databricks custom tag value is too long")
    return normalized


def _safe_id(value: Any, field_name: str) -> str:
    normalized = _non_empty_string(value, field_name)
    if _SAFE_ID_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} contains unsupported characters")
    return normalized


def _required_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty, trimmed string")
    return value


def _databricks_run_id(value: Any, field_name: str) -> str:
    if type(value) is int and value > 0:
        return str(value)
    if (
        isinstance(value, str)
        and len(value) <= 128
        and re.fullmatch(r"[1-9][0-9]*", value) is not None
    ):
        return value
    raise ValueError(
        f"{field_name} must be a strictly positive canonical decimal run ID"
    )


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be an object with string keys")
    return value


def _json_object(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    try:
        normalized = json.loads(
            json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite JSON object") from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return normalized


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("timestamp provider must return timezone-aware datetimes")
    normalized = value.astimezone(UTC)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field_name} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a canonical UTC timestamp") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != datetime.min.replace(tzinfo=UTC).utcoffset()
    ):
        raise ValueError(f"{field_name} must be UTC")
    if _utc_timestamp(parsed) != value:
        raise ValueError(f"{field_name} is not canonically encoded")
    return parsed


def _nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _all_parameters(parameters: Sequence[str], option: str) -> list[str]:
    if len(parameters) % 2 != 0:
        raise ValueError("qualification parameters must contain option/value pairs")
    values: list[str] = []
    for index in range(0, len(parameters), 2):
        observed_option = parameters[index]
        observed_value = parameters[index + 1]
        if not observed_option.startswith("--") or not observed_value:
            raise ValueError(
                "qualification parameters contain an invalid option/value pair"
            )
        if observed_option == option:
            values.append(observed_value)
    return values


def _one_parameter(parameters: Sequence[str], option: str) -> str:
    values = _all_parameters(parameters, option)
    if len(values) != 1:
        raise ValueError(f"qualification parameters require exactly one {option}")
    return values[0]


def _parse_key_value_args(values: Sequence[str], *, option_name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"{option_name} entries must use KEY=VALUE")
        if key in result:
            raise ValueError(f"{option_name} contains duplicate key {key!r}")
        result[key] = value
    return result


def _cloud_cluster_id() -> str:
    for name in ("DATABRICKS_CLUSTER_ID", "DB_CLUSTER_ID"):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(
        "Databricks cluster identity is unavailable; expected "
        "DATABRICKS_CLUSTER_ID or DB_CLUSTER_ID"
    )


def _builtin_sentinel_runner(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
    work_dir: Path,
) -> Mapping[str, Any]:
    """Run the reviewed GPU sentinel dispatcher packaged with Cachet.

    The dispatcher is imported lazily so local payload rendering does not
    import torch/vLLM.  It is a package-owned callable, never a CLI-provided
    factory or an externally supplied measurement JSON file.
    """

    try:
        from document_kv_cache.gpu_qualification_sentinels import (
            run_gpu_qualification_sentinel,
        )
    except ImportError as exc:  # pragma: no cover - packaging failure on GPU.
        raise RuntimeError(
            "the packaged GPU qualification sentinel dispatcher is unavailable"
        ) from exc
    return run_gpu_qualification_sentinel(
        plan_record=plan_record,
        planned_job=planned_job,
        artifact_paths=artifact_paths,
        work_dir=work_dir,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one exact vLLM 0.27.1 GPU qualification sentinel and "
            "write its canonical first-attempt result."
        )
    )
    parser.add_argument(_QUALIFICATION_PLAN_PARAMETER_OPTION, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--reservation-attempt-id", required=True)
    parser.add_argument("--cloud-run-id", required=True)
    parser.add_argument("--attempt-number", type=int, required=True)
    parser.add_argument("--retry-count", type=int, required=True)
    parser.add_argument("--runner-uri", required=True)
    parser.add_argument("--package-wheel-uri", required=True)
    parser.add_argument("--patched-vllm-wheel-uri", required=True)
    parser.add_argument("--artifact-uri", action="append", default=[])
    parser.add_argument("--artifact-sha256", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--work-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.attempt_number != 0 or args.retry_count != 0:
        raise ValueError("GPU qualification jobs must execute on attempt zero")
    plan = _decode_qualification_plan_parameter(
        args.plan_record_zlib_base64,
        expected_plan_sha256=args.expected_plan_sha256,
    )
    artifact_uris = _parse_key_value_args(
        args.artifact_uri, option_name="--artifact-uri"
    )
    artifact_sha256 = _parse_key_value_args(
        args.artifact_sha256, option_name="--artifact-sha256"
    )
    execute_gpu_qualification_job(
        plan_record=plan,
        expected_plan_sha256=args.expected_plan_sha256,
        job_id=args.job_id,
        reservation_attempt_id=args.reservation_attempt_id,
        runner_uri=args.runner_uri,
        package_wheel_uri=args.package_wheel_uri,
        patched_vllm_wheel_uri=args.patched_vllm_wheel_uri,
        artifact_uris=artifact_uris,
        artifact_sha256=artifact_sha256,
        output_json=args.output_json,
        work_dir=args.work_dir,
        cloud_run_id=args.cloud_run_id,
        cloud_cluster_id=_cloud_cluster_id(),
        sentinel_runner=_builtin_sentinel_runner,
    )
    return 0


__all__ = [
    "GPU_QUALIFICATION_ARTIFACT_KEYS",
    "GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT",
    "GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256",
    "GPU_QUALIFICATION_DATABRICKS_PURPOSE",
    "GPU_QUALIFICATION_DATABRICKS_DATA_SECURITY_MODE",
    "GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES",
    "GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS",
    "GPU_QUALIFICATION_LOCAL_WORK_ROOT",
    "GPU_QUALIFICATION_OUTPUT_FILENAME",
    "GPU_QUALIFICATION_SUBMIT_RECEIPT_RECORD_TYPE",
    "GPU_QUALIFICATION_SUBMISSION_REJECTION_RECORD_TYPE",
    "GPUQualificationLaunchAuthorization",
    "GPUQualificationSentinelRunner",
    "collect_gpu_qualification_evidence",
    "execute_gpu_qualification_job",
    "gpu_qualification_reservation_attempt_id",
    "main",
    "pins_from_plan_record",
    "replay_gpu_qualification_launch_authorization",
    "render_gpu_qualification_submit_payloads",
    "resume_gpu_qualification_job_submissions",
    "require_gpu_qualification_launch_authorization",
    "submit_gpu_qualification_jobs",
    "validate_gpu_qualification_submission_rejection_record",
    "write_gpu_qualification_bootstrap_runner",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
