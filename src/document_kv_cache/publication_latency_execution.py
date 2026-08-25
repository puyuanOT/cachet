"""Governed execution and analysis for the frozen 115-job latency campaign.

The module deliberately separates four authorities:

* a closed controller plan proves the frozen factorial and immutable inputs;
* one Databricks run per cell proves physical deployment isolation;
* direct ``jobs/runs/get`` plus the resource ledger proves terminal billing;
* a sealed collection, never caller-supplied scalar measurements, authorizes the
  hierarchical paired publication summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, cast

from document_kv_cache._hardware_targets import (
    databricks_node_type_for_hardware_target,
)
from document_kv_cache.artifact_identity import RuntimeIdentity
from document_kv_cache.benchmark_runner import (
    PUBLICATION_LATENCY_DEPLOYMENT_BLOCK_METADATA_KEY,
    PUBLICATION_LATENCY_INPUT_BUNDLE_SHA256_METADATA_KEY,
    PUBLICATION_LATENCY_LANE_METADATA_KEY,
    PUBLICATION_LATENCY_LANE_POSITION_METADATA_KEY,
    PUBLICATION_LATENCY_REQUEST_ID_METADATA_KEY,
    PUBLICATION_LATENCY_REQUEST_INDEX_METADATA_KEY,
    PUBLICATION_LATENCY_REQUESTS_SHA256_METADATA_KEY,
    PUBLICATION_LATENCY_SCHEDULE_SHA256_METADATA_KEY,
    PUBLICATION_LATENCY_SEED_SHA256_METADATA_KEY,
    benchmark_record_aggregate_issues,
    benchmark_run_result_from_record,
)
from document_kv_cache.benchmark_metrics import request_decode_tokens_per_second
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    SUPPORTED_V1_DATASETS,
    method_benchmark_arm,
)
from document_kv_cache.databricks_job import (
    DEFAULT_DATABRICKS_SPARK_VERSION,
    DatabricksSingleNodeGPUClusterConfig,
    build_single_node_gpu_cluster,
)
from document_kv_cache.databricks_resource_ledger import (
    MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS,
    MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    DatabricksBatchReservationAuthorization,
    DatabricksClusterHourReservation,
    DatabricksLedgerPrefix,
    DatabricksRunAttemptReservationRequest,
    canonical_databricks_submit_payload_snapshot,
    databricks_ledger_prefix,
    databricks_ledger_prefix_from_record,
    databricks_ledger_path_sha256,
    read_databricks_cluster_hour_ledger_json,
    replay_databricks_run_attempt_batch_authorization_json,
    record_databricks_verified_run_terminal_actual_json,
    require_databricks_ledger_prefix,
    require_databricks_batch_reservation_authorization,
    require_databricks_batch_terminal_closure,
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
    GPU_QUALIFICATION_DECODE_HEADROOM_TOKENS,
    GPU_QUALIFICATION_MODEL_ID,
    GPU_QUALIFICATION_MODEL_REVISION,
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPU_QUALIFICATION_PUBLICATION_BACKEND,
    GPUQualificationArtifactPins,
    GPUQualificationSelection,
    validate_gpu_qualification_evidence_record,
)
from document_kv_cache.gpu_qualification_databricks import (
    GPUQualificationLaunchAuthorization,
    require_gpu_qualification_launch_authorization,
)
from document_kv_cache.main_latency_inputs import (
    MAIN_LATENCY_TOKENIZER_ID,
    MAIN_LATENCY_TOKENIZER_REVISION,
)
from document_kv_cache.model_profiles import (
    QWEN3_4B_ROPE_ROTARY_DIM,
    QWEN3_4B_ROPE_THETA,
    layout_for_model,
)
from document_kv_cache.models import CacheGenerationMethod
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_AUXILIARY_SETTINGS,
    PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS,
    PUBLICATION_CAMPAIGN_CONTEXT_TOKENS,
    PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS,
    PUBLICATION_CAMPAIGN_ENGINE_VERSION,
    PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS,
    PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE,
    PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL,
    PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET,
    PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE,
    PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL,
    PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS,
    PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS,
    build_publication_campaign_plan,
    publication_campaign_latency_timeout_policy,
    publication_campaign_plan_to_record,
    validate_publication_campaign_plan_record,
)
from document_kv_cache.publication_handoff_artifacts import (
    read_publication_latency_handoff_bundle,
    validate_publication_latency_handoff_bundle,
)
from document_kv_cache.publication_inputs import (
    PublicationLatencyExample,
    load_publication_storage_selection_examples,
    project_publication_latency_request_order,
    validate_publication_latency_block_schedule,
    validate_publication_storage_block_schedule,
    validate_publication_storage_inputs_record,
)
from document_kv_cache.publication_bf16_handoff_generation import (
    PUBLICATION_BF16_HANDOFF_EXECUTION_MODE,
    PUBLICATION_BF16_HANDOFF_WORKER_COUNT,
    PublicationBF16HandoffServingAuthorization,
    read_publication_bf16_handoff_generation_result,
    require_publication_bf16_handoff_serving_authorization,
)
from document_kv_cache.publication_latency_handoff_generation import (
    PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED,
    PublicationLatencyHandoffServingAuthorization,
    read_publication_latency_handoff_generation_result,
    require_publication_latency_handoff_serving_authorization,
)
from document_kv_cache.serving_env import (
    VLLM_PATCHED_WHEEL_SHA256_ENV,
    VLLM_PATCHED_WHEEL_URI_ENV,
    VLLM_RUNTIME_LOCK_SHA256,
)
from document_kv_cache.vllm_smoke import (
    VLLMSmokeBenchmarkConfig,
    run_vllm_smoke_benchmark,
)


PUBLICATION_LATENCY_EXECUTION_PLAN_RECORD_TYPE: Final = (
    "cachet.publication_latency_execution_plan.v1"
)
PUBLICATION_LATENCY_JOB_RECORD_TYPE: Final = "cachet.publication_latency_job.v1"
PUBLICATION_LATENCY_JOB_RESULT_RECORD_TYPE: Final = (
    "cachet.publication_latency_job_result.v1"
)
PUBLICATION_LATENCY_SUBMISSION_RECORD_TYPE: Final = (
    "cachet.publication_latency_wave_submission.v1"
)
PUBLICATION_LATENCY_TERMINAL_RECEIPT_RECORD_TYPE: Final = (
    "cachet.publication_latency_terminal_receipt.v1"
)
PUBLICATION_LATENCY_COLLECTION_RECORD_TYPE: Final = (
    "cachet.publication_latency_collection.v1"
)
PUBLICATION_LATENCY_SUMMARY_RECORD_TYPE: Final = (
    "cachet.publication_latency_estimation_summary.v1"
)
PUBLICATION_LATENCY_SCHEMA_VERSION: Final = 1
PUBLICATION_LATENCY_RUN_TIMEOUT_SECONDS: Final = 12 * 60 * 60
PUBLICATION_LATENCY_TASK_MAX_RETRIES: Final = 0
PUBLICATION_LATENCY_MAX_OUTPUT_TOKENS: Final = 256
PUBLICATION_LATENCY_TEMPERATURE: Final = 0.0
PUBLICATION_LATENCY_GENERATION_SEED: Final = 17
PUBLICATION_LATENCY_DATABRICKS_AVAILABILITY: Final = "ON_DEMAND"
PUBLICATION_LATENCY_DATABRICKS_DATA_SECURITY_MODE: Final = "NONE"
PUBLICATION_LATENCY_DATABRICKS_ZONES: Final = (
    "us-west-2a",
    "us-west-2b",
    "us-west-2c",
    "us-west-2d",
)
PUBLICATION_LATENCY_RAM_PAYLOAD_CACHE_BYTES: Final = 16 * 1024**3
PUBLICATION_LATENCY_RAM_PRIME_TARGETS: Final = 8
PUBLICATION_LATENCY_Q8_DTYPE: Final = "fp8_e5m2"
PUBLICATION_LATENCY_BF16_DTYPE: Final = "bfloat16"
PUBLICATION_LATENCY_MODEL_QUANTIZATION: Final = "bitsandbytes"
PUBLICATION_LATENCY_VANILLA_ARM_ID: Final = "vanilla"
PUBLICATION_LATENCY_RESULT_FILENAME: Final = "publication-latency-job-result.json"
PUBLICATION_LATENCY_BENCHMARK_FILENAME: Final = "v1-benchmark.json"
PUBLICATION_LATENCY_METADATA_FILENAME: Final = "metadata.json"
PUBLICATION_LATENCY_CONNECTOR_TELEMETRY_FILENAME: Final = (
    "document-kv-connector-telemetry.jsonl"
)
PUBLICATION_LATENCY_RUNTIME_TELEMETRY_FILENAME: Final = "runtime-telemetry.json"
PUBLICATION_LATENCY_PROMPT_BUDGET_FILENAME: Final = "prompt-token-budget.json"
PUBLICATION_LATENCY_IMPORT_PROBE_FILENAME: Final = "vllm-import-probe.json"
PUBLICATION_LATENCY_RUNNER_FILENAME: Final = "run_publication_latency.py"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CLOUD_ID_RE = re.compile(r"[1-9][0-9]*\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,126}\Z")
_DATABRICKS_JOB_RUN_ID_TEMPLATE = "{{job.run_id}}"
_DATABRICKS_TASK_RUN_ID_TEMPLATE = "{{task.run_id}}"
_TERMINAL_LIFE_CYCLE_STATES = frozenset({"TERMINATED", "SKIPPED", "INTERNAL_ERROR"})
_DESCRIPTIVE_AUXILIARY_SETTING_IDS = (
    "precision-bf16",
    "storage-disk",
    "storage-ram",
    "storage-uc",
    "hardware-a10g",
)


PUBLICATION_LATENCY_RUNNER_SCRIPT = r"""from __future__ import annotations

import argparse
import hashlib
import os
import runpy
import subprocess
import sys


def _cluster_path(uri: str) -> str:
    if uri.startswith("dbfs:/"):
        return "/dbfs/" + uri.removeprefix("dbfs:/").lstrip("/")
    if uri.startswith("file:"):
        return uri.removeprefix("file:")
    return uri


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(uri: str, expected: str, label: str) -> str:
    path = _cluster_path(uri)
    if not os.path.isfile(path):
        raise RuntimeError(label + " is missing: " + path)
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(label + " SHA-256 mismatch: " + observed)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-record-json", required=True)
    parser.add_argument("--expected-job-sha256", required=True)
    parser.add_argument("--runner-uri", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--package-wheel-uri", required=True)
    parser.add_argument("--package-wheel-sha256", required=True)
    parser.add_argument("--cloud-run-id", required=True)
    parser.add_argument("--task-run-id", required=True)
    args = parser.parse_args()
    _verified(args.runner_uri, args.runner_sha256, "publication latency runner")
    wheel = _verified(
        args.package_wheel_uri,
        args.package_wheel_sha256,
        "Cachet package wheel",
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel]
    )
    sys.argv = [
        "document_kv_cache.publication_latency_execution",
        "run-job",
        "--job-record-json",
        args.job_record_json,
        "--expected-job-sha256",
        args.expected_job_sha256,
        "--cloud-run-id",
        args.cloud_run_id,
        "--task-run-id",
        args.task_run_id,
    ]
    runpy.run_module("document_kv_cache.publication_latency_execution", run_name="__main__")


if __name__ == "__main__":
    main()
"""
PUBLICATION_LATENCY_RUNNER_SHA256: Final = sha256(
    PUBLICATION_LATENCY_RUNNER_SCRIPT.encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicationLatencyArtifactFile:
    """One immutable durable campaign file."""

    role: str
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        _safe_id(self.role, "artifact role")
        _durable_uri(self.uri, f"artifact {self.role} URI")
        _require_sha256(self.sha256, f"artifact {self.role} sha256")

    def to_record(self) -> dict[str, str]:
        return {"role": self.role, "sha256": self.sha256, "uri": self.uri}


@dataclass(frozen=True, slots=True)
class PublicationLatencyFinalArtifactPins:
    """Closed files and durable roots required by every latency job."""

    source_revision: str
    files: tuple[PublicationLatencyArtifactFile, ...]
    output_root_uri: str
    handoff_generation_root_uri: str
    bf16_handoff_generation_root_uri: str
    bf16_handoff_source_root_uri: str
    uc_handoff_stage_root_uri: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_revision, str)
            or not self.source_revision
            or self.source_revision in {"unknown", "unresolved"}
            or any(character.isspace() for character in self.source_revision)
        ):
            raise ValueError("source_revision must be resolved and whitespace-free")
        files = tuple(self.files)
        if any(not isinstance(item, PublicationLatencyArtifactFile) for item in files):
            raise TypeError("files entries must be PublicationLatencyArtifactFile")
        roles = tuple(item.role for item in files)
        expected_roles = _final_artifact_roles()
        if roles != expected_roles:
            raise ValueError("final artifact files do not match the closed role order")
        if len({item.uri for item in files}) != len(files):
            raise ValueError("final artifact file roles must use distinct URIs")
        object.__setattr__(self, "files", files)
        for field_name in (
            "output_root_uri",
            "handoff_generation_root_uri",
            "bf16_handoff_generation_root_uri",
            "bf16_handoff_source_root_uri",
        ):
            _durable_uri(getattr(self, field_name), field_name)
        _uc_volume_uri(self.uc_handoff_stage_root_uri, "uc_handoff_stage_root_uri")

    def file(self, role: str) -> PublicationLatencyArtifactFile:
        try:
            return next(item for item in self.files if item.role == role)
        except StopIteration as exc:  # pragma: no cover - closed in __post_init__.
            raise KeyError(role) from exc

    def to_record(self) -> dict[str, Any]:
        return {
            "bf16_handoff_generation_root_uri": (self.bf16_handoff_generation_root_uri),
            "bf16_handoff_source_root_uri": self.bf16_handoff_source_root_uri,
            "files": [item.to_record() for item in self.files],
            "handoff_generation_root_uri": self.handoff_generation_root_uri,
            "output_root_uri": self.output_root_uri,
            "source_revision": self.source_revision,
            "uc_handoff_stage_root_uri": self.uc_handoff_stage_root_uri,
        }


def _final_artifact_roles() -> tuple[str, ...]:
    roles = [
        "runner",
        "package_wheel",
        "patched_vllm_wheel",
        "runtime_lock",
        "campaign_plan",
        "qualification_plan",
        "qualification_evidence",
        "handoff_execution",
        "bf16_handoff_execution",
        "bf16_handoff_manifest",
        "storage_inputs",
    ]
    roles.extend(
        f"schedule_block_{block:02d}"
        for block in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)
    )
    roles.extend(
        f"storage_schedule_block_{block:02d}"
        for block in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)
    )
    roles.extend(
        f"input_{context_tokens}_{dataset}"
        for context_tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS
        for dataset in SUPPORTED_V1_DATASETS
    )
    roles.extend(f"storage_input_16384_{dataset}" for dataset in SUPPORTED_V1_DATASETS)
    return tuple(roles)


def write_publication_latency_runner_script(path: str | Path) -> Path:
    """Write the reviewed bootstrap runner once."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"publication latency runner already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(PUBLICATION_LATENCY_RUNNER_SCRIPT, encoding="utf-8")
    return destination


def build_publication_latency_execution_plan(
    *,
    campaign_plan_record: Mapping[str, Any],
    schedule_records: Mapping[int, Mapping[str, Any]],
    storage_schedule_records: Mapping[int, Mapping[str, Any]],
    qualification_plan_record: Mapping[str, Any],
    qualification_evidence_record: Mapping[str, Any],
    qualification_artifact_pins: GPUQualificationArtifactPins,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationLatencyHandoffServingAuthorization,
    bf16_handoff_serving_authorization: PublicationBF16HandoffServingAuthorization,
    storage_inputs_record: Mapping[str, Any],
    final_artifacts: PublicationLatencyFinalArtifactPins,
) -> dict[str, Any]:
    """Close all source evidence and the exact 115 isolated serving jobs."""

    validate_publication_campaign_plan_record(campaign_plan_record)
    campaign_id = _required_string(campaign_plan_record, "campaign_id")
    campaign_ledger_id = _required_string(campaign_plan_record, "campaign_ledger_id")
    campaign_ledger_path_sha256 = _required_sha256(
        campaign_plan_record, "campaign_ledger_path_sha256"
    )
    campaign_opening_terminal_gpu_hours = _finite_nonnegative_number(
        campaign_plan_record.get("campaign_opening_terminal_gpu_hours"),
        "campaign_opening_terminal_gpu_hours",
    )
    campaign_record_sha256 = _required_sha256(
        campaign_plan_record, "closed_record_sha256"
    )
    campaign_ledger_prefix = databricks_ledger_prefix_from_record(
        _mapping(campaign_plan_record, "campaign_ledger_prefix")
    )
    if not isinstance(qualification_artifact_pins, GPUQualificationArtifactPins):
        raise TypeError("qualification_artifact_pins has the wrong type")
    if not isinstance(final_artifacts, PublicationLatencyFinalArtifactPins):
        raise TypeError("final_artifacts has the wrong type")
    if final_artifacts.file("runner").sha256 != PUBLICATION_LATENCY_RUNNER_SHA256:
        raise ValueError("final runner does not match the reviewed latency runner")
    if final_artifacts.file("package_wheel").sha256 != (
        qualification_artifact_pins.package_wheel_sha256
    ):
        raise ValueError("package wheel differs from GPU qualification")
    if final_artifacts.file("patched_vllm_wheel").sha256 != (
        qualification_artifact_pins.patched_vllm_wheel_sha256
    ) or final_artifacts.file("patched_vllm_wheel").sha256 != (
        GPU_QUALIFICATION_PATCHED_WHEEL_SHA256
    ):
        raise ValueError("patched vLLM wheel differs from qualification")
    if (
        final_artifacts.file("runtime_lock").sha256
        != (qualification_artifact_pins.runtime_lock_sha256)
        or final_artifacts.file("runtime_lock").sha256 != VLLM_RUNTIME_LOCK_SHA256
    ):
        raise ValueError("runtime lock differs from qualification")

    evidence_selection = validate_gpu_qualification_evidence_record(
        qualification_evidence_record,
        plan_record=qualification_plan_record,
        expected_campaign_id=campaign_id,
        expected_artifact_pins=qualification_artifact_pins,
    )
    selection = require_gpu_qualification_launch_authorization(
        qualification_launch_authorization,
        expected_plan_sha256=_required_sha256(
            qualification_plan_record, "closed_record_sha256"
        ),
        expected_evidence_file_sha256=final_artifacts.file(
            "qualification_evidence"
        ).sha256,
    )
    if selection != evidence_selection:
        raise ValueError(
            "qualification launch authority selection differs from evidence"
        )
    if selection.attention_backend != GPU_QUALIFICATION_PUBLICATION_BACKEND:
        raise ValueError("qualification did not select the publication backend")
    if (
        qualification_plan_record.get("campaign_record_sha256")
        != campaign_record_sha256
        or qualification_plan_record.get("campaign_ledger_id") != campaign_ledger_id
        or qualification_plan_record.get("campaign_ledger_prefix")
        != campaign_ledger_prefix.to_record()
    ):
        raise ValueError("GPU qualification plan differs from the campaign binding")

    handoff_execution_file = final_artifacts.file("handoff_execution")
    authenticated_handoff = require_publication_latency_handoff_serving_authorization(
        handoff_serving_authorization,
        expected_execution_file_sha256=handoff_execution_file.sha256,
        expected_input_bundle_sha256=(qualification_artifact_pins.input_bundle_sha256),
        expected_qualification_closed_record_sha256=_required_sha256(
            qualification_evidence_record,
            "closed_record_sha256",
        ),
    )
    if (
        authenticated_handoff.record.get("execution_mode")
        != PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED
    ):
        raise ValueError("latency serving requires the distributed handoff result")
    handoff_accounting = _mapping(authenticated_handoff.record, "accounting")
    if handoff_accounting.get("full_launch_throughput_gate_passed") is not True:
        raise ValueError("distributed handoff result did not pass its launch gate")
    if (
        _file_sha256(authenticated_handoff.execution_record_path)
        != handoff_execution_file.sha256
        or _cluster_path(handoff_execution_file.uri).absolute()
        != authenticated_handoff.execution_record_path.absolute()
        or _cluster_path(final_artifacts.handoff_generation_root_uri).absolute()
        != authenticated_handoff.root.absolute()
    ):
        raise ValueError("handoff execution file pin is stale")
    if authenticated_handoff.record.get("input_bundle_sha256") != (
        qualification_artifact_pins.input_bundle_sha256
    ):
        raise ValueError("handoff input bundle differs from qualification")
    generator_hardware = _mapping(authenticated_handoff.record, "generator_hardware")
    if generator_hardware.get("qualification_closed_record_sha256") != (
        qualification_evidence_record.get("closed_record_sha256")
    ):
        raise ValueError("handoff generation used a different GPU qualification")

    storage_source_paths = {
        dataset: _cluster_path(final_artifacts.file(f"input_16384_{dataset}").uri)
        for dataset in SUPPORTED_V1_DATASETS
    }
    storage_source_examples = load_publication_storage_selection_examples(
        storage_source_paths
    )
    schedule_bindings = _validated_schedule_bindings(
        campaign_id=campaign_id,
        schedule_records=schedule_records,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        final_artifacts=final_artifacts,
        role_prefix="schedule_block",
        expected_request_count=PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL,
        expected_workload_id="main",
    )
    storage_schedule_bindings = _validated_schedule_bindings(
        campaign_id=campaign_id,
        schedule_records=storage_schedule_records,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        final_artifacts=final_artifacts,
        role_prefix="storage_schedule_block",
        expected_request_count=PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL,
        expected_workload_id="storage",
        storage_source_examples=storage_source_examples,
    )
    validate_publication_storage_inputs_record(
        storage_inputs_record,
        source_paths=storage_source_paths,
        schedule_records=storage_schedule_records,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
    )
    storage_inputs_artifact = final_artifacts.file("storage_inputs")
    if _file_sha256(_cluster_path(storage_inputs_artifact.uri)) != (
        storage_inputs_artifact.sha256
    ):
        raise ValueError("storage input record final artifact hash drift")
    storage_input_files = {
        item.get("dataset"): item
        for item in _mapping_sequence(storage_inputs_record, "files")
    }
    for dataset in SUPPORTED_V1_DATASETS:
        artifact = final_artifacts.file(f"storage_input_16384_{dataset}")
        file_record = _mapping_value(
            storage_input_files.get(dataset), f"storage input {dataset}"
        )
        if (
            artifact.sha256 != file_record.get("sha256")
            or _cluster_path(artifact.uri).absolute()
            != Path(_required_string(file_record, "uri")).absolute()
        ):
            raise ValueError("storage input final artifact binding drift")
    bf16_binding = _validated_bf16_generation_binding(
        bf16_handoff_serving_authorization,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        final_artifacts=final_artifacts,
    )
    authorization_ledger_ids = {
        qualification_launch_authorization.ledger_id,
        handoff_serving_authorization.ledger_id,
        bf16_handoff_serving_authorization.ledger_id,
    }
    if authorization_ledger_ids != {campaign_ledger_id}:
        raise ValueError(
            "GPU qualification, Q8 handoff, and BF16 handoff must share one "
            "publication campaign ledger"
        )
    authorization_ledger_paths = {
        qualification_launch_authorization.ledger_path_sha256,
        handoff_serving_authorization.ledger_path_sha256,
        bf16_handoff_serving_authorization.ledger_path_sha256,
    }
    if authorization_ledger_paths != {campaign_ledger_path_sha256}:
        raise ValueError("publication authorities use a different campaign ledger path")
    if (
        qualification_plan_record.get("campaign_record_sha256")
        != (campaign_record_sha256)
        or qualification_plan_record.get("campaign_ledger_id") != campaign_ledger_id
    ):
        raise ValueError("qualification plan differs from the campaign record")
    if qualification_plan_record.get("campaign_ledger_path_sha256") != (
        campaign_ledger_path_sha256
    ):
        raise ValueError("qualification plan uses a different campaign ledger path")
    if qualification_plan_record.get("campaign_ledger_prefix") != (
        campaign_ledger_prefix.to_record()
    ):
        raise ValueError("qualification plan uses a different campaign ledger prefix")
    if qualification_plan_record.get("campaign_opening_terminal_gpu_hours") != (
        campaign_opening_terminal_gpu_hours
    ):
        raise ValueError("qualification plan opening terminal balance drift")
    if qualification_launch_authorization.ledger_prefix.reservation_count < (
        campaign_ledger_prefix.reservation_count
    ):
        raise ValueError("qualification authority does not extend campaign genesis")
    if handoff_serving_authorization.predecessor_prefix != (
        qualification_launch_authorization.ledger_prefix
    ):
        raise ValueError("Q8 authority does not extend qualification in phase order")
    if bf16_handoff_serving_authorization.predecessor_prefix != (
        handoff_serving_authorization.ledger_prefix
    ):
        raise ValueError("BF16 authority does not extend Q8 in phase order")
    campaign_binding = {
        **final_artifacts.file("campaign_plan").to_record(),
        "closed_record_sha256": _required_sha256(
            campaign_plan_record,
            "closed_record_sha256",
        ),
    }
    qualification_binding = {
        "authorization": {
            "causal_closure_sha256": (
                qualification_launch_authorization.causal_closure_sha256
            ),
            "ledger_id": qualification_launch_authorization.ledger_id,
            "ledger_path_sha256": (
                qualification_launch_authorization.ledger_path_sha256
            ),
            "ledger_prefix": qualification_launch_authorization.ledger_prefix.to_record(),
        },
        "artifact_pins": qualification_artifact_pins.to_record(),
        "evidence": {
            **final_artifacts.file("qualification_evidence").to_record(),
            "closed_record_sha256": _required_sha256(
                qualification_evidence_record,
                "closed_record_sha256",
            ),
        },
        "plan": {
            **final_artifacts.file("qualification_plan").to_record(),
            "closed_record_sha256": _required_sha256(
                qualification_plan_record,
                "closed_record_sha256",
            ),
        },
        "selection": _selection_record(selection),
    }
    handoff_binding = {
        "accounting_sha256": _canonical_sha256(handoff_accounting),
        "authorization": {
            "causal_closure_sha256": (
                handoff_serving_authorization.causal_closure_sha256
            ),
            "ledger_id": handoff_serving_authorization.ledger_id,
            "ledger_path_sha256": handoff_serving_authorization.ledger_path_sha256,
            "ledger_prefix": handoff_serving_authorization.ledger_prefix.to_record(),
            "predecessor_prefix": (
                handoff_serving_authorization.predecessor_prefix.to_record()
            ),
            "producer_batch_prefix": (
                handoff_serving_authorization.producer_batch_prefix.to_record()
            ),
        },
        "execution": {
            **handoff_execution_file.to_record(),
            "closed_record_sha256": _required_sha256(
                authenticated_handoff.record,
                "closed_record_sha256",
            ),
        },
        "output_root_uri": final_artifacts.handoff_generation_root_uri,
    }
    sources: dict[str, Any] = {
        "bf16_handoff": bf16_binding,
        "campaign": campaign_binding,
        "campaign_ledger_id": campaign_ledger_id,
        "campaign_ledger_path_sha256": campaign_ledger_path_sha256,
        "campaign_ledger_prefix": campaign_ledger_prefix.to_record(),
        "campaign_opening_terminal_gpu_hours": campaign_opening_terminal_gpu_hours,
        "final_artifacts": final_artifacts.to_record(),
        "handoff_generation": handoff_binding,
        "qualification": qualification_binding,
        "schedules": schedule_bindings,
        "storage_schedules": storage_schedule_bindings,
        "storage_inputs": {
            **final_artifacts.file("storage_inputs").to_record(),
            "closed_record_sha256": _required_sha256(
                storage_inputs_record, "closed_record_sha256"
            ),
            "selection_sha256": _required_sha256(
                _mapping(storage_inputs_record, "selection_protocol"),
                "selection_sha256",
            ),
        },
    }
    sources_sha256 = _canonical_sha256(sources)

    core_jobs = [
        _core_job_descriptor(cell)
        for cell in _mapping_sequence(campaign_plan_record, "latency_cells")
    ]
    auxiliary_jobs = [
        _auxiliary_job_descriptor(cell)
        for cell in _mapping_sequence(
            campaign_plan_record,
            "auxiliary_latency_cells",
        )
    ]
    jobs = _assign_execution_zones(
        [*core_jobs, *auxiliary_jobs],
        seed_sha256=sources_sha256,
    )
    waves = _launch_waves(jobs, seed_sha256=sources_sha256)
    record: dict[str, Any] = {
        "analysis": {
            "bootstrap": "paired_hierarchical_deployment_block_and_example",
            "bootstrap_draws": PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS,
            "estimands": [
                "geometric_mean_ttft_speedup",
                "geometric_mean_time_to_completion_speedup",
            ],
            "multiplicity_policy": {
                "decision_mode": "estimation_only",
                "family_count": 13,
                "intervals": "pointwise_95_percent",
                "null_hypothesis_rejections": False,
                "post_hoc_significance_claims": False,
            },
        },
        "campaign_id": campaign_id,
        "closed_record_sha256": "",
        "jobs": jobs,
        "launch_waves": waves,
        "record_type": PUBLICATION_LATENCY_EXECUTION_PLAN_RECORD_TYPE,
        "runtime_policy": {
            "attention_backend": selection.attention_backend,
            "availability": PUBLICATION_LATENCY_DATABRICKS_AVAILABILITY,
            "data_security_mode": PUBLICATION_LATENCY_DATABRICKS_DATA_SECURITY_MODE,
            "databricks_spark_version": DEFAULT_DATABRICKS_SPARK_VERSION,
            "decode_headroom_tokens": GPU_QUALIFICATION_DECODE_HEADROOM_TOKENS,
            "max_output_tokens": PUBLICATION_LATENCY_MAX_OUTPUT_TOKENS,
            "max_parallel_jobs": PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS,
            "max_retries": PUBLICATION_LATENCY_TASK_MAX_RETRIES,
            "model_id": GPU_QUALIFICATION_MODEL_ID,
            "model_quantization": PUBLICATION_LATENCY_MODEL_QUANTIZATION,
            "model_revision": GPU_QUALIFICATION_MODEL_REVISION,
            "condition_timeout_policy": (publication_campaign_latency_timeout_policy()),
            "max_run_timeout_seconds": PUBLICATION_LATENCY_RUN_TIMEOUT_SECONDS,
            "selected_32k_l4_gpu_memory_utilization": (
                selection.gpu_memory_utilization
            ),
            "serving_engine": "vllm",
            "serving_engine_version": PUBLICATION_CAMPAIGN_ENGINE_VERSION,
            "zones": list(PUBLICATION_LATENCY_DATABRICKS_ZONES),
        },
        "schema_version": PUBLICATION_LATENCY_SCHEMA_VERSION,
        "sources": sources,
        "sources_sha256": sources_sha256,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    validate_publication_latency_execution_plan_record(record)
    return record


def validate_publication_latency_execution_plan_record(
    record: Mapping[str, Any],
) -> None:
    """Rebuild the frozen factorial, runtime policy, and deterministic waves."""

    _require_exact_keys(
        record,
        {
            "analysis",
            "campaign_id",
            "closed_record_sha256",
            "jobs",
            "launch_waves",
            "record_type",
            "runtime_policy",
            "schema_version",
            "sources",
            "sources_sha256",
        },
        "publication latency execution plan",
    )
    if record.get("record_type") != PUBLICATION_LATENCY_EXECUTION_PLAN_RECORD_TYPE:
        raise ValueError("publication latency execution plan record_type is invalid")
    if record.get("schema_version") != PUBLICATION_LATENCY_SCHEMA_VERSION:
        raise ValueError("publication latency execution plan schema_version is invalid")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("publication latency execution plan closure is invalid")
    campaign_id = _required_string(record, "campaign_id")
    sources = _mapping(record, "sources")
    _require_exact_keys(
        sources,
        {
            "bf16_handoff",
            "campaign",
            "campaign_ledger_id",
            "campaign_ledger_path_sha256",
            "campaign_ledger_prefix",
            "campaign_opening_terminal_gpu_hours",
            "final_artifacts",
            "handoff_generation",
            "qualification",
            "schedules",
            "storage_schedules",
            "storage_inputs",
        },
        "publication latency sources",
    )
    if record.get("sources_sha256") != _canonical_sha256(sources):
        raise ValueError("publication latency source closure is invalid")
    campaign_ledger_id = _required_string(sources, "campaign_ledger_id")
    campaign_ledger_path_sha256 = _required_sha256(
        sources, "campaign_ledger_path_sha256"
    )
    campaign_ledger_path_sha256 = _required_sha256(
        sources, "campaign_ledger_path_sha256"
    )
    campaign_opening_terminal_gpu_hours = _finite_nonnegative_number(
        sources.get("campaign_opening_terminal_gpu_hours"),
        "campaign_opening_terminal_gpu_hours",
    )
    campaign_ledger_prefix = databricks_ledger_prefix_from_record(
        _mapping(sources, "campaign_ledger_prefix")
    )
    if campaign_ledger_prefix.ledger_id != campaign_ledger_id:
        raise ValueError("campaign ledger prefix identity drift")
    canonical_campaign = publication_campaign_plan_to_record(
        build_publication_campaign_plan(
            campaign_id,
            campaign_ledger_id=campaign_ledger_id,
            campaign_ledger_path_sha256=campaign_ledger_path_sha256,
            campaign_ledger_prefix=campaign_ledger_prefix,
            campaign_opening_terminal_gpu_hours=campaign_opening_terminal_gpu_hours,
        )
    )
    campaign_binding = _mapping(sources, "campaign")
    if campaign_binding.get("closed_record_sha256") != canonical_campaign.get(
        "closed_record_sha256"
    ):
        raise ValueError("publication latency campaign binding drift")
    final_artifacts = _final_artifacts_from_record(_mapping(sources, "final_artifacts"))
    qualification = _mapping(sources, "qualification")
    authorization_binding = _mapping(qualification, "authorization")
    _require_exact_keys(
        authorization_binding,
        {
            "causal_closure_sha256",
            "ledger_id",
            "ledger_path_sha256",
            "ledger_prefix",
        },
        "GPU qualification authorization binding",
    )
    _required_sha256(authorization_binding, "causal_closure_sha256")
    _required_string(authorization_binding, "ledger_id")
    if _required_sha256(authorization_binding, "ledger_path_sha256") != (
        campaign_ledger_path_sha256
    ):
        raise ValueError("GPU qualification authorization ledger path drift")
    qualification_prefix = databricks_ledger_prefix_from_record(
        _mapping(authorization_binding, "ledger_prefix")
    )
    if qualification_prefix.ledger_id != campaign_ledger_id:
        raise ValueError("GPU qualification prefix uses a different campaign ledger")
    pins = GPUQualificationArtifactPins(
        **dict(_mapping(qualification, "artifact_pins"))
    )
    if final_artifacts.file("package_wheel").sha256 != pins.package_wheel_sha256:
        raise ValueError("execution package wheel pin differs from qualification")
    if final_artifacts.file("patched_vllm_wheel").sha256 != (
        pins.patched_vllm_wheel_sha256
    ):
        raise ValueError("execution patched wheel pin differs from qualification")
    if final_artifacts.file("runtime_lock").sha256 != pins.runtime_lock_sha256:
        raise ValueError("execution runtime lock pin differs from qualification")
    selection = _mapping(qualification, "selection")
    selected_gmu = _finite_positive_number(
        selection.get("gpu_memory_utilization"),
        "qualification selected GMU",
    )
    if selected_gmu not in (0.70, 0.75, 0.80):
        raise ValueError("qualification selected GMU is outside the frozen sweep")
    if selection.get("attention_backend") != GPU_QUALIFICATION_PUBLICATION_BACKEND:
        raise ValueError("qualification selection backend drift")
    handoff_binding = _mapping(sources, "handoff_generation")
    _require_exact_keys(
        handoff_binding,
        {"accounting_sha256", "authorization", "execution", "output_root_uri"},
        "Q8 handoff source binding",
    )
    bf16_binding = _mapping(sources, "bf16_handoff")
    _require_exact_keys(
        bf16_binding,
        {
            "accounting",
            "authorization",
            "execution",
            "ledger_reconciliation_sha256",
            "manifest",
            "output_root_uri",
            "source_root_uri",
        },
        "BF16 handoff source binding",
    )
    for label, source_binding in (
        ("Q8 handoff", handoff_binding),
        ("BF16 handoff", bf16_binding),
    ):
        source_authorization = _mapping(source_binding, "authorization")
        _require_exact_keys(
            source_authorization,
            {
                "causal_closure_sha256",
                "ledger_id",
                "ledger_path_sha256",
                "ledger_prefix",
                "predecessor_prefix",
                "producer_batch_prefix",
            },
            f"{label} authorization binding",
        )
        _required_sha256(source_authorization, "causal_closure_sha256")
        if _required_string(source_authorization, "ledger_id") != campaign_ledger_id:
            raise ValueError(f"{label} uses a different campaign ledger")
        if _required_sha256(source_authorization, "ledger_path_sha256") != (
            campaign_ledger_path_sha256
        ):
            raise ValueError(f"{label} uses a different campaign ledger path")
        for prefix_name in (
            "ledger_prefix",
            "predecessor_prefix",
            "producer_batch_prefix",
        ):
            prefix = databricks_ledger_prefix_from_record(
                _mapping(source_authorization, prefix_name)
            )
            if prefix.ledger_id != campaign_ledger_id:
                raise ValueError(f"{label} {prefix_name} identity drift")
    q8_authorization_binding = _mapping(handoff_binding, "authorization")
    bf16_authorization_binding = _mapping(bf16_binding, "authorization")
    if _mapping(q8_authorization_binding, "predecessor_prefix") != (
        _mapping(authorization_binding, "ledger_prefix")
    ):
        raise ValueError("Q8 predecessor is not the qualification prefix")
    if _mapping(bf16_authorization_binding, "predecessor_prefix") != (
        _mapping(q8_authorization_binding, "ledger_prefix")
    ):
        raise ValueError("BF16 predecessor is not the Q8 terminal prefix")
    if _required_string(authorization_binding, "ledger_id") != campaign_ledger_id:
        raise ValueError("GPU qualification uses a different campaign ledger")
    runtime = _mapping(record, "runtime_policy")
    expected_runtime = {
        "attention_backend": GPU_QUALIFICATION_PUBLICATION_BACKEND,
        "availability": PUBLICATION_LATENCY_DATABRICKS_AVAILABILITY,
        "data_security_mode": PUBLICATION_LATENCY_DATABRICKS_DATA_SECURITY_MODE,
        "databricks_spark_version": DEFAULT_DATABRICKS_SPARK_VERSION,
        "decode_headroom_tokens": GPU_QUALIFICATION_DECODE_HEADROOM_TOKENS,
        "max_output_tokens": PUBLICATION_LATENCY_MAX_OUTPUT_TOKENS,
        "max_parallel_jobs": PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS,
        "max_retries": 0,
        "model_id": GPU_QUALIFICATION_MODEL_ID,
        "model_quantization": PUBLICATION_LATENCY_MODEL_QUANTIZATION,
        "model_revision": GPU_QUALIFICATION_MODEL_REVISION,
        "condition_timeout_policy": publication_campaign_latency_timeout_policy(),
        "max_run_timeout_seconds": PUBLICATION_LATENCY_RUN_TIMEOUT_SECONDS,
        "selected_32k_l4_gpu_memory_utilization": selected_gmu,
        "serving_engine": "vllm",
        "serving_engine_version": PUBLICATION_CAMPAIGN_ENGINE_VERSION,
        "zones": list(PUBLICATION_LATENCY_DATABRICKS_ZONES),
    }
    if dict(runtime) != expected_runtime:
        raise ValueError("publication latency runtime policy drift")
    schedule_bindings = _mapping_sequence(sources, "schedules")
    if tuple(item.get("deployment_block") for item in schedule_bindings) != tuple(
        range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)
    ):
        raise ValueError("publication latency schedule bindings are incomplete")
    storage_schedule_bindings = _mapping_sequence(sources, "storage_schedules")
    if tuple(
        item.get("deployment_block") for item in storage_schedule_bindings
    ) != tuple(range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)):
        raise ValueError("publication storage schedule bindings are incomplete")
    storage_selection_digests = {
        _required_sha256(item, "selection_sha256") for item in storage_schedule_bindings
    }
    storage_inputs_binding = _mapping(sources, "storage_inputs")
    storage_input_selection_sha256 = _required_sha256(
        storage_inputs_binding, "selection_sha256"
    )
    if storage_selection_digests != {storage_input_selection_sha256}:
        raise ValueError("publication storage selection digests are inconsistent")

    jobs = _mapping_sequence(record, "jobs")
    expected_jobs = _assign_execution_zones(
        [
            *(
                _core_job_descriptor(item)
                for item in _mapping_sequence(canonical_campaign, "latency_cells")
            ),
            *(
                _auxiliary_job_descriptor(item)
                for item in _mapping_sequence(
                    canonical_campaign,
                    "auxiliary_latency_cells",
                )
            ),
        ],
        seed_sha256=str(record["sources_sha256"]),
    )
    if jobs != expected_jobs or len(jobs) != PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS:
        raise ValueError(
            "publication latency jobs differ from the frozen 115-job design"
        )
    if len({item.get("job_id") for item in jobs}) != len(jobs):
        raise ValueError("publication latency job IDs are not unique")
    expected_waves = _launch_waves(jobs, seed_sha256=str(record["sources_sha256"]))
    if _mapping_sequence(record, "launch_waves") != expected_waves:
        raise ValueError("publication latency launch waves are not canonical")
    analysis = _mapping(record, "analysis")
    if analysis.get("bootstrap_draws") != PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS:
        raise ValueError("publication latency bootstrap draw count drift")
    multiplicity = _mapping(analysis, "multiplicity_policy")
    if (
        multiplicity.get("decision_mode") != "estimation_only"
        or multiplicity.get("null_hypothesis_rejections") is not False
        or multiplicity.get("post_hoc_significance_claims") is not False
        or multiplicity.get("family_count") != 13
    ):
        raise ValueError("publication latency multiplicity policy drift")


def _validated_schedule_bindings(
    *,
    campaign_id: str,
    schedule_records: Mapping[int, Mapping[str, Any]],
    expected_input_bundle_sha256: str,
    final_artifacts: PublicationLatencyFinalArtifactPins,
    role_prefix: str,
    expected_request_count: int,
    expected_workload_id: str,
    storage_source_examples: Sequence[PublicationLatencyExample] | None = None,
) -> list[dict[str, Any]]:
    if set(schedule_records) != set(
        range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)
    ):
        raise ValueError("schedule_records must contain exactly five deployment blocks")
    bindings: list[dict[str, Any]] = []
    for block in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1):
        schedule = schedule_records[block]
        examples = _schedule_examples(schedule)
        if expected_workload_id == "storage":
            if storage_source_examples is None:
                raise ValueError(
                    "storage schedule authorization requires all source examples"
                )
            validate_publication_storage_block_schedule(
                schedule,
                source_examples=storage_source_examples,
                expected_input_bundle_sha256=expected_input_bundle_sha256,
            )
        else:
            validate_publication_latency_block_schedule(
                schedule,
                examples=examples,
                expected_input_bundle_sha256=expected_input_bundle_sha256,
            )
        if (
            schedule.get("campaign_id") != campaign_id
            or schedule.get("deployment_block") != block
        ):
            raise ValueError("latency schedule campaign/block binding drift")
        projection = project_publication_latency_request_order(
            schedule,
            examples=examples,
            expected_input_bundle_sha256=expected_input_bundle_sha256,
        )
        protocol = _mapping(schedule, "protocol")
        if (
            len(projection) != expected_request_count
            or protocol.get("workload_id") != expected_workload_id
        ):
            raise ValueError("latency schedule request closure is incomplete")
        file = final_artifacts.file(f"{role_prefix}_{block:02d}")
        binding = {
            **file.to_record(),
            "closed_record_sha256": _required_sha256(
                schedule,
                "closed_record_sha256",
            ),
            "deployment_block": block,
            "requests_sha256": _required_sha256(schedule, "requests_sha256"),
            "seed_sha256": _required_sha256(schedule, "seed_sha256"),
        }
        if expected_workload_id == "storage":
            binding["selection_sha256"] = _required_sha256(
                _mapping(_mapping(schedule, "protocol"), "selection"),
                "selection_sha256",
            )
        bindings.append(binding)
    return bindings


def _validated_bf16_generation_binding(
    authorization: PublicationBF16HandoffServingAuthorization,
    *,
    expected_input_bundle_sha256: str,
    final_artifacts: PublicationLatencyFinalArtifactPins,
) -> dict[str, Any]:
    if not isinstance(authorization, PublicationBF16HandoffServingAuthorization):
        raise TypeError(
            "bf16_handoff_serving_authorization must be a collector-issued "
            "PublicationBF16HandoffServingAuthorization"
        )
    manifest_artifact = final_artifacts.file("bf16_handoff_manifest")
    authenticated = require_publication_bf16_handoff_serving_authorization(
        authorization,
        expected_manifest_file_sha256=manifest_artifact.sha256,
        expected_manifest_closed_record_sha256=(
            authorization.manifest_closed_record_sha256
        ),
        expected_input_bundle_sha256=expected_input_bundle_sha256,
    )
    manifest = authenticated.manifest
    execution = authenticated.record
    execution_artifact = final_artifacts.file("bf16_handoff_execution")
    bound_paths = (
        (
            manifest_artifact,
            authenticated.manifest_path,
            "BF16 manifest",
        ),
        (
            execution_artifact,
            authenticated.execution_record_path,
            "BF16 execution record",
        ),
    )
    for artifact, authenticated_path, label in bound_paths:
        if (
            _cluster_path(artifact.uri).absolute() != authenticated_path.absolute()
            or _file_sha256(authenticated_path) != artifact.sha256
        ):
            raise ValueError(f"{label} final artifact pin drift")
    if (
        _cluster_path(final_artifacts.bf16_handoff_generation_root_uri).absolute()
        != authenticated.root.absolute()
        or _cluster_path(final_artifacts.bf16_handoff_source_root_uri).absolute()
        != authenticated.source_root.absolute()
    ):
        raise ValueError("BF16 generation/source root final artifact pin drift")
    if manifest.get("input_bundle_sha256") != expected_input_bundle_sha256:
        raise ValueError("BF16 handoff input bundle differs from the campaign")
    if manifest.get("context_tokens") != 16_384:
        raise ValueError("BF16 auxiliary handoff must cover the 16k anchor")
    identity = _mapping(manifest, "identity")
    layout = _mapping(identity, "layout_identity")
    if (
        layout.get("dtype") not in {"bf16", "bfloat16"}
        or layout.get("pre_rope") is not True
        or layout.get("key_position_encoding") != "pre_rope"
        or layout.get("rope_theta") != QWEN3_4B_ROPE_THETA
        or layout.get("rope_rotary_dim") != QWEN3_4B_ROPE_ROTARY_DIM
    ):
        raise ValueError("BF16 auxiliary handoff layout is not pre-RoPE BF16")
    datasets = _mapping_sequence(manifest, "datasets")
    if tuple(item.get("dataset") for item in datasets) != tuple(SUPPORTED_V1_DATASETS):
        raise ValueError("BF16 auxiliary handoff dataset coverage is incomplete")
    for dataset in datasets:
        entries = _mapping_sequence(dataset, "entries")
        if len(entries) != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET or any(
            entry.get("cache_method") != CacheGenerationMethod.VANILLA_PREFILL.value
            for entry in entries
        ):
            raise ValueError("BF16 auxiliary handoff is not complete Vanilla evidence")
    accounting = _mapping(execution, "accounting")
    reconciliation = _mapping(execution, "ledger_reconciliation")
    if (
        execution.get("execution_mode") != PUBLICATION_BF16_HANDOFF_EXECUTION_MODE
        or accounting.get("worker_count") != PUBLICATION_BF16_HANDOFF_WORKER_COUNT
        or accounting.get("full_launch_throughput_gate_passed") is not True
        or reconciliation.get("attempt_count") != PUBLICATION_BF16_HANDOFF_WORKER_COUNT
        or reconciliation.get("verification_source") != "direct_databricks_runs_get"
    ):
        raise ValueError("BF16 distributed generation authority is incomplete")
    return {
        "accounting": {
            "closed_sha256": _canonical_sha256(accounting),
            "full_launch_throughput_gate_passed": True,
            "worker_count": PUBLICATION_BF16_HANDOFF_WORKER_COUNT,
        },
        "authorization": {
            "causal_closure_sha256": authorization.causal_closure_sha256,
            "ledger_id": authorization.ledger_id,
            "ledger_path_sha256": authorization.ledger_path_sha256,
            "ledger_prefix": authorization.ledger_prefix.to_record(),
            "predecessor_prefix": authorization.predecessor_prefix.to_record(),
            "producer_batch_prefix": authorization.producer_batch_prefix.to_record(),
        },
        "execution": {
            **execution_artifact.to_record(),
            "closed_record_sha256": _required_sha256(
                execution,
                "closed_record_sha256",
            ),
            "execution_mode": PUBLICATION_BF16_HANDOFF_EXECUTION_MODE,
        },
        "ledger_reconciliation_sha256": _canonical_sha256(reconciliation),
        "manifest": {
            **manifest_artifact.to_record(),
            "closed_record_sha256": _required_sha256(
                manifest,
                "closed_record_sha256",
            ),
            "portable_bundle_sha256": _required_sha256(
                manifest,
                "portable_bundle_sha256",
            ),
        },
        "output_root_uri": final_artifacts.bf16_handoff_generation_root_uri,
        "source_root_uri": final_artifacts.bf16_handoff_source_root_uri,
    }


def _schedule_examples(
    record: Mapping[str, Any],
) -> tuple[PublicationLatencyExample, ...]:
    seen: set[tuple[str, str]] = set()
    examples: list[PublicationLatencyExample] = []
    for request in _mapping_sequence(record, "requests"):
        key = (
            _required_string(request, "dataset"),
            _required_string(request, "example_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        examples.append(PublicationLatencyExample(dataset=key[0], example_id=key[1]))
    examples.sort(
        key=lambda item: (SUPPORTED_V1_DATASETS.index(item.dataset), item.example_id)
    )
    protocol = _mapping(record, "protocol")
    examples_per_dataset = _required_int(protocol, "examples_per_dataset")
    expected = len(SUPPORTED_V1_DATASETS) * examples_per_dataset
    if len(examples) != expected:
        raise ValueError("latency schedule has incomplete example coverage")
    return tuple(examples)


def _selection_record(selection: GPUQualificationSelection) -> dict[str, Any]:
    return {
        "attention_backend": selection.attention_backend,
        "generation_artifacts_sha256": selection.generation_artifacts_sha256,
        "generation_databricks_node_type_id": (
            selection.generation_databricks_node_type_id
        ),
        "generation_hardware_id": selection.generation_hardware_id,
        "generation_prefix_tokens_per_second": (
            selection.generation_prefix_tokens_per_second
        ),
        "gpu_memory_utilization": selection.gpu_memory_utilization,
        "plan_sha256": selection.plan_sha256,
    }


def _core_job_descriptor(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": _required_string(cell, "cell_id"),
        "deployment_block": _required_int(cell, "deployment_block"),
        "examples_per_dataset": _required_int(cell, "examples_per_dataset"),
        "input_tokens": _required_int(cell, "input_tokens"),
        "job_id": _required_string(cell, "cell_id"),
        "job_kind": "core",
        "matched_pair_id": _required_string(cell, "matched_pair_id"),
        "method_id": _required_string(cell, "method_id"),
        "request_parallelism": _required_int(cell, "request_parallelism"),
        "repeats_per_example": _required_int(cell, "repeats_per_example"),
        "request_count": _required_int(cell, "request_count"),
    }


def _auxiliary_job_descriptor(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": _required_string(cell, "cell_id"),
        "comparison_family": _required_string(cell, "comparison_family"),
        "deployment_block": _required_int(cell, "deployment_block"),
        "examples_per_dataset": _required_int(cell, "examples_per_dataset"),
        "input_tokens": _required_int(cell, "input_tokens"),
        "job_id": _required_string(cell, "cell_id"),
        "job_kind": "auxiliary",
        "method_id": _required_string(cell, "method_id"),
        "reference_core_cell_id": _required_string(
            cell,
            "reference_core_cell_id",
        ),
        "request_parallelism": _required_int(cell, "request_parallelism"),
        "repeats_per_example": _required_int(cell, "repeats_per_example"),
        "request_count": _required_int(cell, "request_count"),
        "setting_id": _required_string(cell, "setting_id"),
    }


def _assign_execution_zones(
    jobs: Sequence[Mapping[str, Any]],
    *,
    seed_sha256: str,
) -> list[dict[str, Any]]:
    """Bind every comparison unit to one explicit QA zone deterministically."""

    _require_sha256_value(seed_sha256, "zone assignment seed")
    zone_by_job: dict[str, str] = {}
    output: list[dict[str, Any]] = []
    for job in jobs:
        if job.get("job_kind") != "core":
            continue
        pair_id = _required_string(job, "matched_pair_id")
        zone_by_job[_required_string(job, "job_id")] = _matched_unit_zone(
            pair_id,
            seed_sha256=seed_sha256,
        )
    storage_settings = {"storage-disk", "storage-ram", "storage-uc"}
    for block in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1):
        storage_jobs = [
            job
            for job in jobs
            if job.get("setting_id") in storage_settings
            and job.get("deployment_block") == block
        ]
        if len(storage_jobs) != len(storage_settings):
            raise ValueError("storage matched-unit zone coverage is incomplete")
        zone = _matched_unit_zone(
            f"block-{block:02d}-storage-trio",
            seed_sha256=seed_sha256,
        )
        for job in storage_jobs:
            zone_by_job[_required_string(job, "job_id")] = zone
    unresolved = [
        job
        for job in jobs
        if job.get("job_kind") != "core"
        and job.get("setting_id") not in storage_settings
    ]
    while unresolved:
        next_unresolved: list[Mapping[str, Any]] = []
        progressed = False
        for job in unresolved:
            reference_id = _required_string(job, "reference_core_cell_id")
            referenced_zone = zone_by_job.get(reference_id)
            if referenced_zone is None:
                next_unresolved.append(job)
                continue
            zone_by_job[_required_string(job, "job_id")] = referenced_zone
            progressed = True
        if not progressed:
            raise ValueError("auxiliary comparison zone reference is unresolved")
        unresolved = next_unresolved
    for job in jobs:
        job_id = _required_string(job, "job_id")
        output.append({**dict(job), "zone_id": zone_by_job[job_id]})
    return output


def _matched_unit_zone(unit_id: str, *, seed_sha256: str) -> str:
    _safe_id(unit_id, "matched unit ID")
    _require_sha256_value(seed_sha256, "zone assignment seed")
    return PUBLICATION_LATENCY_DATABRICKS_ZONES[
        int(
            _canonical_sha256(
                {
                    "domain": "cachet.publication_latency.zone.v1",
                    "matched_unit_id": unit_id,
                    "seed_sha256": seed_sha256,
                }
            )[:16],
            16,
        )
        % len(PUBLICATION_LATENCY_DATABRICKS_ZONES)
    ]


def _launch_waves(
    jobs: Sequence[Mapping[str, Any]],
    *,
    seed_sha256: str,
) -> list[dict[str, Any]]:
    _require_sha256_value(seed_sha256, "launch-wave seed")
    by_block: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for job in jobs:
        by_block[_required_int(job, "deployment_block")].append(job)
    waves: list[dict[str, Any]] = []
    global_index = 0
    for block in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1):
        block_jobs = by_block[block]
        block_job_by_id = {_required_string(job, "job_id"): job for job in block_jobs}
        if len(block_job_by_id) != len(block_jobs):
            raise ValueError("latency job IDs must be unique within each block")
        pairs: dict[str, list[str]] = defaultdict(list)
        auxiliary_by_setting: dict[str, str] = {}
        for job in block_jobs:
            job_id = _required_string(job, "job_id")
            if job.get("job_kind") == "core":
                pairs[_required_string(job, "matched_pair_id")].append(job_id)
            else:
                setting_id = _required_string(job, "setting_id")
                if setting_id in auxiliary_by_setting:
                    raise ValueError("auxiliary setting is duplicated within a block")
                auxiliary_by_setting[setting_id] = job_id
        expected_auxiliary_settings = {
            "precision-bf16",
            "storage-disk",
            "storage-ram",
            "storage-uc",
            "hardware-a10g",
        }
        if set(auxiliary_by_setting) != expected_auxiliary_settings:
            raise ValueError("auxiliary matched-unit coverage is incomplete")
        unit_members: dict[str, list[str]] = {}
        for pair_id, pair_job_ids in pairs.items():
            if len(pair_job_ids) != 2:
                raise ValueError("each core latency pair must contain two jobs")
            unit_members[pair_id] = list(pair_job_ids)

        for setting_id in ("precision-bf16", "hardware-a10g"):
            auxiliary_job_id = auxiliary_by_setting[setting_id]
            reference_id = _required_string(
                block_job_by_id[auxiliary_job_id], "reference_core_cell_id"
            )
            reference = block_job_by_id.get(reference_id)
            if reference is None or reference.get("job_kind") != "core":
                raise ValueError("precision/hardware reference core is missing")
            reference_pair_id = _required_string(reference, "matched_pair_id")
            unit_members[reference_pair_id].append(auxiliary_job_id)

        storage_job_ids = [
            auxiliary_by_setting[setting_id]
            for setting_id in ("storage-disk", "storage-ram", "storage-uc")
        ]
        storage_unit_id = f"block-{block:02d}-storage-trio"
        unit_members[storage_unit_id] = storage_job_ids

        units: list[tuple[str, tuple[str, ...]]] = []
        for unit_id, member_ids in unit_members.items():
            ordered_members = tuple(
                sorted(
                    member_ids,
                    key=lambda job_id: _canonical_sha256(
                        {
                            "domain": "cachet.publication_latency.unit_order.v1",
                            "job_id": job_id,
                            "seed_sha256": seed_sha256,
                            "unit_id": unit_id,
                        }
                    ),
                )
            )
            units.append((unit_id, ordered_members))
        units.sort(
            key=lambda unit: _canonical_sha256(
                {
                    "deployment_block": block,
                    "domain": "cachet.publication_latency.wave_order.v1",
                    "seed_sha256": seed_sha256,
                    "unit_id": unit[0],
                }
            )
        )
        current: list[str] = []
        for _unit_id, unit_jobs in units:
            if current and len(current) + len(unit_jobs) > (
                PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
            ):
                waves.append(
                    {
                        "deployment_block": block,
                        "job_count": len(current),
                        "job_ids": current,
                        "wave_index": global_index,
                    }
                )
                global_index += 1
                current = []
            current.extend(unit_jobs)
        if current:
            waves.append(
                {
                    "deployment_block": block,
                    "job_count": len(current),
                    "job_ids": current,
                    "wave_index": global_index,
                }
            )
            global_index += 1
    if any(
        not 0
        < _required_int(wave, "job_count")
        <= PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
        for wave in waves
    ):
        raise RuntimeError("publication latency launch wave exceeds max parallelism")
    flattened = [
        job_id for wave in waves for job_id in cast(list[str], wave["job_ids"])
    ]
    expected_ids = [_required_string(job, "job_id") for job in jobs]
    if len(flattened) != len(expected_ids) or set(flattened) != set(expected_ids):
        raise RuntimeError("publication latency launch waves do not close all jobs")
    wave_by_job = {
        job_id: _required_int(wave, "wave_index")
        for wave in waves
        for job_id in cast(list[str], wave["job_ids"])
    }
    for job in jobs:
        if job.get("job_kind") != "core":
            continue
        pair_id = _required_string(job, "matched_pair_id")
        pair_jobs = [
            _required_string(candidate, "job_id")
            for candidate in jobs
            if candidate.get("matched_pair_id") == pair_id
        ]
        if len({wave_by_job[job_id] for job_id in pair_jobs}) != 1:
            raise RuntimeError("matched Baseline/Vanilla pair was split across waves")
    by_block_and_setting = {
        (
            _required_int(job, "deployment_block"),
            _required_string(job, "setting_id"),
        ): _required_string(job, "job_id")
        for job in jobs
        if job.get("job_kind") == "auxiliary"
    }
    for block in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1):
        storage_ids = [
            by_block_and_setting[(block, setting_id)]
            for setting_id in ("storage-disk", "storage-ram", "storage-uc")
        ]
        if len({wave_by_job[job_id] for job_id in storage_ids}) != 1:
            raise RuntimeError("matched Disk/RAM/UC trio was split across waves")
        for setting_id in ("precision-bf16", "hardware-a10g"):
            auxiliary_id = by_block_and_setting[(block, setting_id)]
            reference_id = _required_string(
                next(
                    job
                    for job in jobs
                    if _required_string(job, "job_id") == auxiliary_id
                ),
                "reference_core_cell_id",
            )
            if wave_by_job[auxiliary_id] != wave_by_job[reference_id]:
                raise RuntimeError(
                    "matched core/precision/hardware unit was split across waves"
                )
    return waves


def publication_latency_reservation_attempt_id(
    execution_plan_sha256: str,
    job_id: str,
) -> str:
    """Return the stable ledger identity for one physical job attempt."""

    _require_sha256_value(execution_plan_sha256, "execution_plan_sha256")
    _safe_id(job_id, "job_id")
    return f"publication-latency/{execution_plan_sha256[:20]}/{job_id}"


def render_publication_latency_job_record(
    execution_plan_record: Mapping[str, Any],
    job_id: str,
    *,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationLatencyHandoffServingAuthorization,
    bf16_handoff_serving_authorization: PublicationBF16HandoffServingAuthorization,
) -> dict[str, Any]:
    """Render a worker only after all replay-backed launch authorities."""

    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
    )
    validate_publication_latency_execution_sources(execution_plan_record)
    return _render_publication_latency_job_record(execution_plan_record, job_id)


def _render_publication_latency_job_record(
    execution_plan_record: Mapping[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Render one closed worker contract from the authenticated campaign plan."""

    validate_publication_latency_execution_plan_record(execution_plan_record)
    _safe_id(job_id, "job_id")
    jobs = _mapping_sequence(execution_plan_record, "jobs")
    try:
        descriptor = next(item for item in jobs if item.get("job_id") == job_id)
    except StopIteration as exc:
        raise KeyError(job_id) from exc
    plan_sha256 = _required_sha256(
        execution_plan_record,
        "closed_record_sha256",
    )
    sources = _mapping(execution_plan_record, "sources")
    final_artifacts = _final_artifacts_from_record(_mapping(sources, "final_artifacts"))
    qualification = _mapping(sources, "qualification")
    selection = _mapping(qualification, "selection")
    deployment_block = _required_int(descriptor, "deployment_block")
    storage_workload = descriptor.get("setting_id") in {
        "storage-disk",
        "storage-ram",
        "storage-uc",
    }
    schedule = next(
        item
        for item in _mapping_sequence(
            sources,
            "storage_schedules" if storage_workload else "schedules",
        )
        if item.get("deployment_block") == deployment_block
    )
    input_tokens = _required_int(descriptor, "input_tokens")
    request_parallelism = _required_int(descriptor, "request_parallelism")
    runtime = _job_runtime_policy(
        descriptor,
        selected_32k_gmu=_finite_positive_number(
            selection.get("gpu_memory_utilization"),
            "selected 32k GMU",
        ),
    )
    input_files = [
        final_artifacts.file(
            (
                f"storage_input_16384_{dataset}"
                if storage_workload
                else f"input_{input_tokens}_{dataset}"
            )
        ).to_record()
        | {"dataset": dataset}
        for dataset in SUPPORTED_V1_DATASETS
    ]
    cache_policy = _cache_policy(descriptor)
    handoff: dict[str, Any] | None
    if descriptor.get("method_id") == "baseline_prefill":
        handoff = None
    elif descriptor.get("setting_id") == "precision-bf16":
        bf16 = _mapping(sources, "bf16_handoff")
        handoff = {
            "manifest": dict(_mapping(bf16, "manifest")),
            "source_kind": "closed_bf16_bundle",
            "source_root_uri": _required_string(bf16, "source_root_uri"),
            "stage_kind": "local_nvme",
            "stage_uri": (
                f"/local_disk0/cachet-publication-latency/{plan_sha256[:16]}/"
                f"{job_id}/handoff"
            ),
        }
    else:
        generated = _mapping(sources, "handoff_generation")
        stage_kind = (
            "uc_mounted"
            if descriptor.get("setting_id") == "storage-uc"
            else "local_nvme"
        )
        stage_uri = (
            _join_durable_uri(
                final_artifacts.uc_handoff_stage_root_uri,
                plan_sha256,
                job_id,
            )
            if stage_kind == "uc_mounted"
            else (
                f"/local_disk0/cachet-publication-latency/{plan_sha256[:16]}/"
                f"{job_id}/handoff"
            )
        )
        handoff = {
            "execution": dict(_mapping(generated, "execution")),
            "output_root_uri": _required_string(generated, "output_root_uri"),
            "source_kind": "distributed_q8_generation",
            "stage_kind": stage_kind,
            "stage_uri": stage_uri,
        }
    output_dir_uri = _join_durable_uri(
        final_artifacts.output_root_uri,
        plan_sha256,
        job_id,
    )
    job_record: dict[str, Any] = {
        "artifact_files": [
            final_artifacts.file(role).to_record()
            for role in (
                "runner",
                "package_wheel",
                "patched_vllm_wheel",
                "runtime_lock",
            )
        ],
        "cache_telemetry_policy": cache_policy,
        "campaign_id": _required_string(execution_plan_record, "campaign_id"),
        "cell": dict(descriptor),
        "closed_record_sha256": "",
        "execution_plan_sha256": plan_sha256,
        "handoff": handoff,
        "input_files": input_files,
        "job_id": job_id,
        "output": {
            "directory_uri": output_dir_uri,
            "result_uri": _join_durable_uri(
                output_dir_uri,
                PUBLICATION_LATENCY_RESULT_FILENAME,
            ),
        },
        "record_type": PUBLICATION_LATENCY_JOB_RECORD_TYPE,
        "request_order": {
            "closed_record_sha256": _required_sha256(
                schedule,
                "closed_record_sha256",
            ),
            "deployment_block": deployment_block,
            "file_sha256": _required_sha256(schedule, "sha256"),
            "input_bundle_sha256": _required_sha256(
                _mapping(qualification, "artifact_pins"),
                "input_bundle_sha256",
            ),
            "requests_sha256": _required_sha256(schedule, "requests_sha256"),
            "schedule_uri": _required_string(schedule, "uri"),
            "seed_sha256": _required_sha256(schedule, "seed_sha256"),
            "workload_id": "storage" if storage_workload else "main",
        },
        "reservation_attempt_id": publication_latency_reservation_attempt_id(
            plan_sha256,
            job_id,
        ),
        "runtime": runtime,
        "schema_version": PUBLICATION_LATENCY_SCHEMA_VERSION,
        "source_revision": final_artifacts.source_revision,
        "source_tree_sha256": _required_sha256(
            _mapping(qualification, "artifact_pins"),
            "cachet_source_tree_sha256",
        ),
        "sources_sha256": _required_sha256(execution_plan_record, "sources_sha256"),
        "task_key": _task_key(job_id),
    }
    job_record["closed_record_sha256"] = _closed_record_sha256(job_record)
    validate_publication_latency_job_record(job_record)
    if input_tokens not in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
        raise RuntimeError("rendered job context escaped the campaign")
    if request_parallelism not in (1, 2, 4):
        raise RuntimeError("rendered job parallelism escaped the campaign")
    return job_record


def validate_publication_latency_job_record(record: Mapping[str, Any]) -> None:
    """Validate a standalone worker contract without trusting result values."""

    _require_exact_keys(
        record,
        {
            "artifact_files",
            "cache_telemetry_policy",
            "campaign_id",
            "cell",
            "closed_record_sha256",
            "execution_plan_sha256",
            "handoff",
            "input_files",
            "job_id",
            "output",
            "record_type",
            "request_order",
            "reservation_attempt_id",
            "runtime",
            "schema_version",
            "source_revision",
            "source_tree_sha256",
            "sources_sha256",
            "task_key",
        },
        "publication latency job",
    )
    if record.get("record_type") != PUBLICATION_LATENCY_JOB_RECORD_TYPE:
        raise ValueError("publication latency job record_type is invalid")
    if record.get("schema_version") != PUBLICATION_LATENCY_SCHEMA_VERSION:
        raise ValueError("publication latency job schema_version is invalid")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("publication latency job closure is invalid")
    job_id = _safe_id(record.get("job_id"), "job_id")
    cell = _mapping(record, "cell")
    if cell.get("job_id") != job_id or cell.get("cell_id") != job_id:
        raise ValueError("publication latency job/cell identity drift")
    plan_sha256 = _required_sha256(record, "execution_plan_sha256")
    if record.get("reservation_attempt_id") != (
        publication_latency_reservation_attempt_id(plan_sha256, job_id)
    ):
        raise ValueError("publication latency reservation identity drift")
    if record.get("task_key") != _task_key(job_id):
        raise ValueError("publication latency task key drift")
    artifacts = _mapping_sequence(record, "artifact_files")
    if tuple(item.get("role") for item in artifacts) != (
        "runner",
        "package_wheel",
        "patched_vllm_wheel",
        "runtime_lock",
    ):
        raise ValueError("publication latency runtime artifact closure is incomplete")
    if artifacts[0].get("sha256") != PUBLICATION_LATENCY_RUNNER_SHA256:
        raise ValueError("publication latency runner hash drift")
    if artifacts[2].get("sha256") != GPU_QUALIFICATION_PATCHED_WHEEL_SHA256:
        raise ValueError("publication latency patched wheel hash drift")
    if artifacts[3].get("sha256") != VLLM_RUNTIME_LOCK_SHA256:
        raise ValueError("publication latency runtime lock hash drift")
    _required_sha256(record, "source_tree_sha256")
    inputs = _mapping_sequence(record, "input_files")
    if tuple(item.get("dataset") for item in inputs) != tuple(SUPPORTED_V1_DATASETS):
        raise ValueError("publication latency input files are incomplete")
    for item in (*artifacts, *inputs):
        _durable_uri(_required_string(item, "uri"), "job artifact URI")
        _required_sha256(item, "sha256")
    request_order = _mapping(record, "request_order")
    if request_order.get("deployment_block") != cell.get("deployment_block"):
        raise ValueError("publication latency schedule block drift")
    for field_name in (
        "closed_record_sha256",
        "file_sha256",
        "input_bundle_sha256",
        "requests_sha256",
        "seed_sha256",
    ):
        _required_sha256(request_order, field_name)
    _durable_uri(
        _required_string(request_order, "schedule_uri"),
        "publication schedule URI",
    )
    expected_workload = (
        "storage"
        if cell.get("setting_id") in {"storage-disk", "storage-ram", "storage-uc"}
        else "main"
    )
    if request_order.get("workload_id") != expected_workload:
        raise ValueError("publication latency schedule workload binding drift")
    runtime = _mapping(record, "runtime")
    expected_runtime = _job_runtime_policy(
        cell,
        selected_32k_gmu=_finite_positive_number(
            runtime.get("selected_32k_l4_gpu_memory_utilization"),
            "selected_32k_l4_gpu_memory_utilization",
        ),
    )
    if dict(runtime) != expected_runtime:
        raise ValueError("publication latency job runtime policy drift")
    cache_policy = _cache_policy(cell)
    if record.get("cache_telemetry_policy") != cache_policy:
        raise ValueError("publication latency cache telemetry policy drift")
    handoff = record.get("handoff")
    if cell.get("method_id") == "baseline_prefill":
        if handoff is not None:
            raise ValueError("Baseline publication job cannot carry a handoff")
    else:
        handoff_record = _mapping_value(handoff, "handoff")
        if handoff_record.get("stage_kind") not in {"local_nvme", "uc_mounted"}:
            raise ValueError("publication latency handoff stage kind is invalid")
        if handoff_record.get("stage_kind") == "uc_mounted":
            _uc_volume_uri(
                _required_string(handoff_record, "stage_uri"),
                "UC handoff stage URI",
            )
        elif not _required_string(handoff_record, "stage_uri").startswith(
            "/local_disk0/"
        ):
            raise ValueError("local handoff stage must use /local_disk0")
    output = _mapping(record, "output")
    directory_uri = _durable_uri(
        _required_string(output, "directory_uri"),
        "job output directory",
    )
    if output.get("result_uri") != _join_durable_uri(
        directory_uri,
        PUBLICATION_LATENCY_RESULT_FILENAME,
    ):
        raise ValueError("publication latency result path drift")


def _job_runtime_policy(
    descriptor: Mapping[str, Any],
    *,
    selected_32k_gmu: float,
) -> dict[str, Any]:
    input_tokens = _required_int(descriptor, "input_tokens")
    request_parallelism = _required_int(descriptor, "request_parallelism")
    setting_id = descriptor.get("setting_id")
    hardware_target = "aws-g5-a10g" if setting_id == "hardware-a10g" else "aws-g6-l4"
    node_type_id = databricks_node_type_for_hardware_target(hardware_target)
    runtime_kv_dtype = (
        PUBLICATION_LATENCY_BF16_DTYPE
        if setting_id == "precision-bf16"
        else PUBLICATION_LATENCY_Q8_DTYPE
    )
    gpu_memory_utilization = (
        selected_32k_gmu
        if hardware_target == "aws-g6-l4" and input_tokens == 32_768
        else 0.90
    )
    timeout_seconds = _job_timeout_seconds(descriptor)
    return {
        "attention_backend": GPU_QUALIFICATION_PUBLICATION_BACKEND,
        "availability": PUBLICATION_LATENCY_DATABRICKS_AVAILABILITY,
        "data_security_mode": PUBLICATION_LATENCY_DATABRICKS_DATA_SECURITY_MODE,
        "databricks_spark_version": DEFAULT_DATABRICKS_SPARK_VERSION,
        "force_max_tokens": True,
        "generation_seed": PUBLICATION_LATENCY_GENERATION_SEED,
        "gpu_memory_utilization": gpu_memory_utilization,
        "hardware_target": hardware_target,
        "max_model_len": input_tokens + GPU_QUALIFICATION_DECODE_HEADROOM_TOKENS,
        "max_num_seqs": request_parallelism,
        "max_output_tokens": PUBLICATION_LATENCY_MAX_OUTPUT_TOKENS,
        "model_dtype": PUBLICATION_LATENCY_BF16_DTYPE,
        "model_id": GPU_QUALIFICATION_MODEL_ID,
        "model_quantization": PUBLICATION_LATENCY_MODEL_QUANTIZATION,
        "model_revision": GPU_QUALIFICATION_MODEL_REVISION,
        "node_type_id": node_type_id,
        "request_parallelism": request_parallelism,
        "run_timeout_seconds": timeout_seconds,
        "runtime_kv_dtype": runtime_kv_dtype,
        "selected_32k_l4_gpu_memory_utilization": selected_32k_gmu,
        "temperature": PUBLICATION_LATENCY_TEMPERATURE,
        "zone_id": _required_string(descriptor, "zone_id"),
    }


def _job_timeout_seconds(descriptor: Mapping[str, Any]) -> int:
    timeout_policy = publication_campaign_latency_timeout_policy()
    if descriptor.get("job_kind") == "auxiliary":
        return _required_int(timeout_policy, "auxiliary_c4_hours") * 60 * 60
    input_tokens = _required_int(descriptor, "input_tokens")
    request_parallelism = _required_int(descriptor, "request_parallelism")
    hours_by_cell = _mapping(
        timeout_policy,
        "core_hours_by_context_and_concurrency",
    )
    try:
        context_policy = _mapping_value(
            hours_by_cell[f"{input_tokens // 1024}k"],
            "latency timeout context",
        )
        timeout_hours = _required_int(context_policy, f"c{request_parallelism}")
        return timeout_hours * 60 * 60
    except (KeyError, ValueError) as exc:
        raise ValueError("latency timeout cell is outside the frozen design") from exc


def _cache_policy(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    if descriptor.get("method_id") == "baseline_prefill":
        return {
            "connector_loads": "forbidden",
            "host_cache_state": "not_applicable",
            "payload_cache": "disabled",
            "storage_source": "none",
        }
    setting_id = descriptor.get("setting_id")
    if setting_id == "storage-ram":
        return {
            "connector_loads": "required_exact_request_coverage",
            "host_cache_state": "prewarmed_payload_cache",
            "payload_cache": "prewarmed_16_gib_exact_hits",
            "storage_source": "ram_payload_cache",
        }
    if setting_id == "storage-uc":
        return {
            "connector_loads": "required_exact_request_coverage",
            "host_cache_state": "mounted_path_evicted_backend_cache_unproven",
            "payload_cache": "disabled",
            "storage_source": "uc_mounted_path",
        }
    return {
        "connector_loads": "required_exact_request_coverage",
        "host_cache_state": "cold_eviction_required",
        "payload_cache": "disabled",
        "storage_source": "local_nvme",
    }


def build_databricks_publication_latency_run_submit_payload(
    execution_plan_record: Mapping[str, Any],
    job_id: str,
    *,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationLatencyHandoffServingAuthorization,
    bf16_handoff_serving_authorization: PublicationBF16HandoffServingAuthorization,
) -> dict[str, Any]:
    """Render one task only after all replay-backed launch authorities."""

    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
    )
    validate_publication_latency_execution_sources(execution_plan_record)
    return _build_databricks_publication_latency_run_submit_payload(
        execution_plan_record, job_id
    )


def _build_databricks_publication_latency_run_submit_payload(
    execution_plan_record: Mapping[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Internal exact no-retry, condition-timeout, single-GPU renderer."""

    job = _render_publication_latency_job_record(execution_plan_record, job_id)
    runtime = _mapping(job, "runtime")
    artifact_files = {
        _required_string(item, "role"): item
        for item in _mapping_sequence(job, "artifact_files")
    }
    cluster = build_single_node_gpu_cluster(
        DatabricksSingleNodeGPUClusterConfig(
            purpose="cachet-publication-latency",
            node_type_id=_required_string(runtime, "node_type_id"),
            spark_version=_required_string(runtime, "databricks_spark_version"),
            data_security_mode=_required_string(runtime, "data_security_mode"),
            availability=_required_string(runtime, "availability"),
            zone_id=_required_string(runtime, "zone_id"),
            custom_tags={
                "campaign": sha256(
                    _required_string(job, "campaign_id").encode("utf-8")
                ).hexdigest()[:24],
                "job_id": sha256(job_id.encode("utf-8")).hexdigest()[:24],
                "plan": _required_sha256(job, "execution_plan_sha256")[:24],
            },
        )
    )
    cluster["spark_env_vars"] = {
        VLLM_PATCHED_WHEEL_SHA256_ENV: _required_string(
            artifact_files["patched_vllm_wheel"],
            "sha256",
        ),
        VLLM_PATCHED_WHEEL_URI_ENV: _required_string(
            artifact_files["patched_vllm_wheel"],
            "uri",
        ),
        "DOCUMENT_KV_EVICT_PAGE_CACHE": (
            "0" if job["cell"].get("setting_id") == "storage-ram" else "1"
        ),
    }
    job_json = _canonical_json(job)
    parameters = [
        "--job-record-json",
        job_json,
        "--expected-job-sha256",
        _required_sha256(job, "closed_record_sha256"),
        "--runner-uri",
        _required_string(artifact_files["runner"], "uri"),
        "--runner-sha256",
        _required_string(artifact_files["runner"], "sha256"),
        "--package-wheel-uri",
        _required_string(artifact_files["package_wheel"], "uri"),
        "--package-wheel-sha256",
        _required_string(artifact_files["package_wheel"], "sha256"),
        "--cloud-run-id",
        _DATABRICKS_JOB_RUN_ID_TEMPLATE,
        "--task-run-id",
        _DATABRICKS_TASK_RUN_ID_TEMPLATE,
    ]
    payload = bind_databricks_run_idempotency_token(
        {
            "run_name": f"cachet-publication-latency-{job_id}",
            "tasks": [
                {
                    "max_retries": PUBLICATION_LATENCY_TASK_MAX_RETRIES,
                    "new_cluster": cluster,
                    "spark_python_task": {
                        "parameters": parameters,
                        "python_file": _required_string(
                            artifact_files["runner"], "uri"
                        ),
                    },
                    "task_key": _required_string(job, "task_key"),
                    "timeout_seconds": _required_int(runtime, "run_timeout_seconds"),
                }
            ],
            "timeout_seconds": _required_int(runtime, "run_timeout_seconds"),
        },
        attempt_id=_required_string(job, "reservation_attempt_id"),
    )
    _validate_submit_payload(payload, job_record=job)
    return payload


def publication_latency_submit_payloads(
    execution_plan_record: Mapping[str, Any],
    *,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationLatencyHandoffServingAuthorization,
    bf16_handoff_serving_authorization: PublicationBF16HandoffServingAuthorization,
) -> tuple[dict[str, Any], ...]:
    """Return all 115 payloads in canonical campaign order."""

    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
    )
    validate_publication_latency_execution_sources(execution_plan_record)
    payloads = tuple(
        _build_databricks_publication_latency_run_submit_payload(
            execution_plan_record,
            _required_string(job, "job_id"),
        )
        for job in _mapping_sequence(execution_plan_record, "jobs")
    )
    if len(payloads) != PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS:
        raise RuntimeError("publication latency payload closure is incomplete")
    return payloads


_WAVE_SUBMISSION_AUTHORIZATION_ISSUER = object()
_WAVE_COLLECTION_AUTHORIZATION_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class PublicationLatencyWaveSubmissionAuthorization:
    """Non-record authority over one atomically reserved latency wave."""

    execution_plan_sha256: str
    wave_index: int
    ledger_path_sha256: str
    predecessor_prefix: DatabricksLedgerPrefix
    batch_authorization: DatabricksBatchReservationAuthorization

    def __init__(
        self,
        *,
        execution_plan_sha256: str,
        wave_index: int,
        ledger_path_sha256: str,
        predecessor_prefix: DatabricksLedgerPrefix,
        batch_authorization: DatabricksBatchReservationAuthorization,
        _issuer: object,
    ) -> None:
        if _issuer is not _WAVE_SUBMISSION_AUTHORIZATION_ISSUER:
            raise TypeError(
                "latency wave submission authority requires atomic admission"
            )
        object.__setattr__(
            self,
            "execution_plan_sha256",
            _require_sha256_value(execution_plan_sha256, "execution_plan_sha256"),
        )
        if type(wave_index) is not int or wave_index < 0:
            raise ValueError("wave_index must be non-negative")
        object.__setattr__(self, "wave_index", wave_index)
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _require_sha256_value(ledger_path_sha256, "ledger_path_sha256"),
        )
        if not isinstance(predecessor_prefix, DatabricksLedgerPrefix):
            raise TypeError("predecessor_prefix has the wrong type")
        if not isinstance(batch_authorization, DatabricksBatchReservationAuthorization):
            raise TypeError("batch_authorization has the wrong type")
        if (
            batch_authorization.predecessor_prefix != predecessor_prefix
            or batch_authorization.ledger_path_sha256 != ledger_path_sha256
        ):
            raise ValueError("latency wave batch authority binding drift")
        object.__setattr__(self, "predecessor_prefix", predecessor_prefix)
        object.__setattr__(self, "batch_authorization", batch_authorization)


@dataclass(frozen=True, slots=True, init=False)
class PublicationLatencyWaveAuthorization:
    """Non-record authority issued after one exact wave is terminal."""

    execution_plan_sha256: str
    wave_index: int
    ledger_path_sha256: str
    ledger_prefix: DatabricksLedgerPrefix
    causal_closure_sha256: str

    def __init__(
        self,
        *,
        execution_plan_sha256: str,
        wave_index: int,
        ledger_path_sha256: str,
        ledger_prefix: DatabricksLedgerPrefix,
        causal_closure_sha256: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _WAVE_COLLECTION_AUTHORIZATION_ISSUER:
            raise TypeError("latency wave authority requires terminal collection")
        object.__setattr__(
            self,
            "execution_plan_sha256",
            _require_sha256_value(execution_plan_sha256, "execution_plan_sha256"),
        )
        if type(wave_index) is not int or wave_index < 0:
            raise ValueError("wave_index must be non-negative")
        object.__setattr__(self, "wave_index", wave_index)
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _require_sha256_value(ledger_path_sha256, "ledger_path_sha256"),
        )
        if not isinstance(ledger_prefix, DatabricksLedgerPrefix):
            raise TypeError("ledger_prefix has the wrong type")
        object.__setattr__(self, "ledger_prefix", ledger_prefix)
        object.__setattr__(
            self,
            "causal_closure_sha256",
            _require_sha256_value(causal_closure_sha256, "causal_closure_sha256"),
        )


def submit_publication_latency_launch_wave(
    config: DatabricksWorkspaceConfig,
    *,
    execution_plan_record: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationLatencyHandoffServingAuthorization,
    bf16_handoff_serving_authorization: PublicationBF16HandoffServingAuthorization,
    ledger_path: str | Path,
    wave_index: int,
    phase_lease_root: str | Path,
    prior_wave_authorization: PublicationLatencyWaveAuthorization | None = None,
    opener: DatabricksURLOpener | None = None,
) -> tuple[dict[str, Any], PublicationLatencyWaveSubmissionAuthorization]:
    """Reserve, submit, and receipt-bind one deterministic wave (at most 16)."""

    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
    )
    validate_publication_latency_execution_sources(execution_plan_record)
    if type(wave_index) is not int or wave_index < 0:
        raise ValueError("wave_index must be a non-negative integer")
    waves = _mapping_sequence(execution_plan_record, "launch_waves")
    if wave_index >= len(waves):
        raise ValueError("wave_index is outside the execution plan")
    wave = waves[wave_index]
    raw_job_ids = wave.get("job_ids")
    if (
        not isinstance(raw_job_ids, list)
        or not raw_job_ids
        or len(raw_job_ids) > PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
        or any(not isinstance(item, str) for item in raw_job_ids)
    ):
        raise ValueError("launch wave has invalid job IDs")
    ledger_file = Path(ledger_path).expanduser().absolute()
    ledger = read_databricks_cluster_hour_ledger_json(ledger_file)
    campaign_ledger_id = _required_string(
        _mapping(execution_plan_record, "sources"), "campaign_ledger_id"
    )
    if ledger.ledger_id != campaign_ledger_id:
        raise ValueError("latency ledger differs from the campaign ledger")
    ledger_path_sha256 = databricks_ledger_path_sha256(ledger_file)
    expected_path_sha256 = _required_sha256(
        _mapping(execution_plan_record, "sources"),
        "campaign_ledger_path_sha256",
    )
    if ledger_path_sha256 != expected_path_sha256:
        raise ValueError("latency ledger path differs from the campaign plan")
    plan_sha256 = _required_sha256(execution_plan_record, "closed_record_sha256")
    if wave_index == 0:
        if prior_wave_authorization is not None:
            raise ValueError("latency wave zero must not accept prior-wave authority")
        predecessor = bf16_handoff_serving_authorization.ledger_prefix
    else:
        if not isinstance(
            prior_wave_authorization, PublicationLatencyWaveAuthorization
        ):
            raise TypeError("later latency waves require prior-wave authority")
        if (
            prior_wave_authorization.execution_plan_sha256 != plan_sha256
            or prior_wave_authorization.wave_index != wave_index - 1
            or prior_wave_authorization.ledger_path_sha256 != ledger_path_sha256
        ):
            raise ValueError("latency prior-wave authority binding drift")
        predecessor = prior_wave_authorization.ledger_prefix
    require_databricks_ledger_prefix(ledger, predecessor)
    if databricks_ledger_prefix(ledger) != predecessor:
        raise ValueError("latency wave predecessor is not the complete live ledger")
    _require_prior_waves_succeeded(
        execution_plan_record,
        ledger=ledger,
        wave_index=wave_index,
    )

    jobs: list[tuple[str, Mapping[str, Any], dict[str, Any], str, str, int]] = []
    requests: list[DatabricksRunAttemptReservationRequest] = []
    for job_id in cast(list[str], raw_job_ids):
        payload = _build_databricks_publication_latency_run_submit_payload(
            execution_plan_record, job_id
        )
        _snapshot, canonical = canonical_databricks_submit_payload_snapshot(payload)
        payload_sha256 = sha256(canonical).hexdigest()
        job = _render_publication_latency_job_record(execution_plan_record, job_id)
        attempt_id = _required_string(job, "reservation_attempt_id")
        timeout_seconds = _required_int(_mapping(job, "runtime"), "run_timeout_seconds")
        jobs.append((job_id, job, payload, attempt_id, payload_sha256, timeout_seconds))
        requests.append(
            DatabricksRunAttemptReservationRequest(
                attempt_id=attempt_id,
                workload_id=f"publication-latency/{plan_sha256[:20]}",
                submit_payload=payload,
            )
        )

    lease_root = _create_latency_phase_lease_root(phase_lease_root)
    lease = {
        "attempt_ids": [item[3] for item in jobs],
        "closed_record_sha256": "",
        "execution_plan_sha256": plan_sha256,
        "ledger_path_sha256": ledger_path_sha256,
        "predecessor_prefix": predecessor.to_record(),
        "record_type": "cachet.publication_latency_wave_phase_lease.v1",
        "submit_payload_sha256": [item[4] for item in jobs],
        "wave_index": wave_index,
    }
    lease["closed_record_sha256"] = _closed_record_sha256(lease)
    _write_canonical_json_exclusive(lease_root / "phase-lease.json", lease)

    def validate_batch(
        batch_live: Any,
        reservations: tuple[DatabricksClusterHourReservation, ...],
        snapshots: tuple[Mapping[str, Any], ...],
    ) -> None:
        if databricks_ledger_path_sha256(ledger_file) != ledger_path_sha256:
            raise ValueError("latency batch ledger path binding drift")
        require_databricks_ledger_prefix(batch_live, predecessor)
        if len(reservations) != len(jobs) or len(snapshots) != len(jobs):
            raise ValueError("latency batch differs from the exact launch wave")
        for job_entry, reservation, snapshot in zip(
            jobs, reservations, snapshots, strict=True
        ):
            _job_id, job, payload, attempt_id, payload_sha256, timeout_seconds = (
                job_entry
            )
            _validate_submit_payload(snapshot, job_record=job)
            if _canonical_json(snapshot) != _canonical_json(payload):
                raise ValueError("latency batch snapshot changed after rendering")
            if (
                reservation.attempt_id != attempt_id
                or reservation.submit_payload_sha256 != payload_sha256
                or reservation.run_timeout_seconds != timeout_seconds
                or reservation.task_timeout_seconds != (timeout_seconds,)
            ):
                raise ValueError("latency batch reservation member drift")
        proposed_tasks = sum(len(item.task_timeout_seconds) for item in reservations)
        proposed_hours = sum(item.reserved_cluster_hours for item in reservations)
        if batch_live.cap_cluster_hours != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS:
            raise ValueError("latency publication requires the 1024-hour ledger")
        if (
            batch_live.active_reserved_task_count + proposed_tasks
            > PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
        ):
            raise ValueError("latency wave exceeds the global 16-job concurrency cap")
        if (
            batch_live.active_reserved_cluster_hours + proposed_hours
            > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
        ):
            raise ValueError("latency wave exceeds the 900-hour active cap")
        if (
            batch_live.accounted_cluster_hours + proposed_hours
            > MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
            - PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
        ):
            raise ValueError("latency wave consumes the 124-hour headroom")

    try:
        _batch_ledger, batch_authorization = (
            reserve_databricks_run_attempt_batch_authorized_json(
                ledger_file,
                tuple(requests),
                expected_predecessor_prefix=predecessor,
                batch_validator=validate_batch,
            )
        )
    except BaseException:
        _remove_empty_latency_phase_lease_root(lease_root)
        raise
    attempt_ids = tuple(item[3] for item in jobs)
    payload_digests = tuple(item[4] for item in jobs)
    require_databricks_batch_reservation_authorization(
        batch_authorization,
        expected_predecessor_prefix=predecessor,
        expected_attempt_ids=attempt_ids,
        expected_submit_payload_sha256s=payload_digests,
    )
    batch_record = {
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "record_type": "cachet.publication_latency_wave_batch_reserved.v1",
        "wave_index": wave_index,
    }
    batch_record["closed_record_sha256"] = _closed_record_sha256(batch_record)
    _write_canonical_json_exclusive(lease_root / "batch-reserved.json", batch_record)

    submitted: list[dict[str, Any]] = []
    for job_id, _job, payload, attempt_id, payload_sha256, _timeout in jobs:
        intent = {
            "attempt_id": attempt_id,
            "batch_prefix": batch_authorization.batch_prefix.to_record(),
            "closed_record_sha256": "",
            "job_id": job_id,
            "record_type": "cachet.publication_latency_wave_post_intent.v1",
            "submit_payload_sha256": payload_sha256,
        }
        intent["closed_record_sha256"] = _closed_record_sha256(intent)
        intent_path = lease_root / f"{job_id}.post-intent.json"
        _write_canonical_json_exclusive(intent_path, intent)
        response = submit_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_file,
            attempt_id=attempt_id,
            batch_authorization=batch_authorization,
            opener=opener,
        )
        response_run_id = _databricks_id(response.get("run_id"), "submit run_id")
        current = read_databricks_cluster_hour_ledger_json(ledger_file)
        receipt = next(
            item
            for item in current.submission_receipts
            if item.attempt_id == attempt_id
        )
        if receipt.run_id != response_run_id:
            raise RuntimeError("persisted latency receipt run ID drift")
        receipt_record = {
            "attempt_id": attempt_id,
            "closed_record_sha256": "",
            "job_id": job_id,
            "record_type": "cachet.publication_latency_wave_submit_receipt.v1",
            "run_id": receipt.run_id,
            "submit_payload_sha256": receipt.submit_payload_sha256,
            "submit_response_sha256": receipt.submit_response_sha256,
        }
        receipt_record["closed_record_sha256"] = _closed_record_sha256(receipt_record)
        _write_canonical_json_exclusive(
            lease_root / f"{job_id}.receipt.json", receipt_record
        )
        intent_path.unlink()
        _fsync_directory(lease_root)
        submitted.append(
            {
                "attempt_id": attempt_id,
                "job_id": job_id,
                "run_id": receipt.run_id,
                "submit_payload_sha256": receipt.submit_payload_sha256,
                "submit_response_sha256": receipt.submit_response_sha256,
            }
        )
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "execution_plan_sha256": plan_sha256,
        "jobs": submitted,
        "record_type": PUBLICATION_LATENCY_SUBMISSION_RECORD_TYPE,
        "schema_version": PUBLICATION_LATENCY_SCHEMA_VERSION,
        "wave_index": wave_index,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    authorization = PublicationLatencyWaveSubmissionAuthorization(
        execution_plan_sha256=plan_sha256,
        wave_index=wave_index,
        ledger_path_sha256=ledger_path_sha256,
        predecessor_prefix=predecessor,
        batch_authorization=batch_authorization,
        _issuer=_WAVE_SUBMISSION_AUTHORIZATION_ISSUER,
    )
    return record, authorization


def resume_publication_latency_launch_wave(
    config: DatabricksWorkspaceConfig,
    *,
    execution_plan_record: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationLatencyHandoffServingAuthorization,
    bf16_handoff_serving_authorization: PublicationBF16HandoffServingAuthorization,
    ledger_path: str | Path,
    wave_index: int,
    phase_lease_root: str | Path,
    prior_wave_authorization: PublicationLatencyWaveAuthorization | None = None,
    opener: DatabricksURLOpener | None = None,
) -> tuple[dict[str, Any], PublicationLatencyWaveSubmissionAuthorization]:
    """Resume one exact latency wave from its durable phase lease."""

    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
    )
    validate_publication_latency_execution_sources(execution_plan_record)
    waves = _mapping_sequence(execution_plan_record, "launch_waves")
    if type(wave_index) is not int or not 0 <= wave_index < len(waves):
        raise ValueError("wave_index is outside the execution plan")
    job_ids = waves[wave_index].get("job_ids")
    if (
        not isinstance(job_ids, list)
        or not job_ids
        or len(job_ids) > PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
        or any(not isinstance(item, str) for item in job_ids)
    ):
        raise ValueError("launch wave has invalid job IDs")
    plan_sha256 = _required_sha256(execution_plan_record, "closed_record_sha256")
    ledger_file = Path(ledger_path).expanduser().absolute()
    ledger_path_sha256 = databricks_ledger_path_sha256(ledger_file)
    sources = _mapping(execution_plan_record, "sources")
    if ledger_path_sha256 != _required_sha256(sources, "campaign_ledger_path_sha256"):
        raise ValueError("latency resume ledger path differs from campaign")
    if wave_index == 0:
        if prior_wave_authorization is not None:
            raise ValueError("latency wave zero must not accept prior-wave authority")
        predecessor = bf16_handoff_serving_authorization.ledger_prefix
    else:
        if not isinstance(
            prior_wave_authorization, PublicationLatencyWaveAuthorization
        ):
            raise TypeError("later latency waves require prior-wave authority")
        if (
            prior_wave_authorization.execution_plan_sha256 != plan_sha256
            or prior_wave_authorization.wave_index != wave_index - 1
            or prior_wave_authorization.ledger_path_sha256 != ledger_path_sha256
        ):
            raise ValueError("latency prior-wave authority binding drift")
        predecessor = prior_wave_authorization.ledger_prefix
    live = read_databricks_cluster_hour_ledger_json(ledger_file)
    require_databricks_ledger_prefix(live, predecessor)
    jobs: list[tuple[str, dict[str, Any], dict[str, Any], str, str]] = []
    requests: list[DatabricksRunAttemptReservationRequest] = []
    for job_id in cast(list[str], job_ids):
        job = _render_publication_latency_job_record(execution_plan_record, job_id)
        payload = _build_databricks_publication_latency_run_submit_payload(
            execution_plan_record, job_id
        )
        _validate_submit_payload(payload, job_record=job)
        attempt_id = _required_string(job, "reservation_attempt_id")
        payload_sha256 = _submit_payload_sha256(payload)
        jobs.append((job_id, job, payload, attempt_id, payload_sha256))
        requests.append(
            DatabricksRunAttemptReservationRequest(
                attempt_id=attempt_id,
                workload_id=f"publication-latency/{plan_sha256[:20]}",
                submit_payload=payload,
            )
        )
    lease_root = Path(phase_lease_root).expanduser().absolute()
    _reject_existing_symlink_ancestors(lease_root, "latency phase lease")
    if not lease_root.is_dir() or lease_root.is_symlink():
        raise ValueError("latency resume requires the existing real phase lease")
    expected_lease: dict[str, Any] = {
        "attempt_ids": [item[3] for item in jobs],
        "closed_record_sha256": "",
        "execution_plan_sha256": plan_sha256,
        "ledger_path_sha256": ledger_path_sha256,
        "predecessor_prefix": predecessor.to_record(),
        "record_type": "cachet.publication_latency_wave_phase_lease.v1",
        "submit_payload_sha256": [item[4] for item in jobs],
        "wave_index": wave_index,
    }
    expected_lease["closed_record_sha256"] = _closed_record_sha256(expected_lease)
    if _read_latency_controller_record(
        lease_root / "phase-lease.json", "latency phase lease"
    ) != expected_lease:
        raise ValueError("latency phase lease differs from the frozen wave")
    batch_authorization = replay_databricks_run_attempt_batch_authorization_json(
        ledger_file,
        tuple(requests),
        expected_predecessor_prefix=predecessor,
    )
    require_databricks_publication_batch_admission(live, batch_authorization)
    expected_batch: dict[str, Any] = {
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "record_type": "cachet.publication_latency_wave_batch_reserved.v1",
        "wave_index": wave_index,
    }
    expected_batch["closed_record_sha256"] = _closed_record_sha256(expected_batch)
    batch_path = lease_root / "batch-reserved.json"
    if batch_path.exists() or batch_path.is_symlink():
        if _read_latency_controller_record(
            batch_path, "latency batch marker"
        ) != expected_batch:
            raise ValueError("latency batch marker differs from the ledger batch")
    else:
        _write_canonical_json_exclusive(batch_path, expected_batch)
    submitted: list[dict[str, Any]] = []
    for job_id, _job, payload, attempt_id, payload_sha256 in jobs:
        intent_path = lease_root / f"{job_id}.post-intent.json"
        receipt_path = lease_root / f"{job_id}.receipt.json"
        expected_intent: dict[str, Any] = {
            "attempt_id": attempt_id,
            "batch_prefix": batch_authorization.batch_prefix.to_record(),
            "closed_record_sha256": "",
            "job_id": job_id,
            "record_type": "cachet.publication_latency_wave_post_intent.v1",
            "submit_payload_sha256": payload_sha256,
        }
        expected_intent["closed_record_sha256"] = _closed_record_sha256(expected_intent)
        if intent_path.exists() or intent_path.is_symlink():
            if (
                _read_latency_controller_record(
                    intent_path, f"latency job {job_id} post intent"
                )
                != expected_intent
            ):
                raise ValueError("latency post intent drift")
        elif not receipt_path.exists():
            _write_canonical_json_exclusive(intent_path, expected_intent)
        response = resume_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_file,
            attempt_id=attempt_id,
            batch_authorization=batch_authorization,
            opener=opener,
        )
        current = read_databricks_cluster_hour_ledger_json(ledger_file)
        receipt = next(
            item
            for item in current.submission_receipts
            if item.attempt_id == attempt_id
        )
        if receipt.run_id != _databricks_id(response.get("run_id"), "submit run_id"):
            raise RuntimeError("persisted latency receipt run ID drift")
        expected_receipt: dict[str, Any] = {
            "attempt_id": attempt_id,
            "closed_record_sha256": "",
            "job_id": job_id,
            "record_type": "cachet.publication_latency_wave_submit_receipt.v1",
            "run_id": receipt.run_id,
            "submit_payload_sha256": receipt.submit_payload_sha256,
            "submit_response_sha256": receipt.submit_response_sha256,
        }
        expected_receipt["closed_record_sha256"] = _closed_record_sha256(
            expected_receipt
        )
        if receipt_path.exists() or receipt_path.is_symlink():
            if (
                _read_latency_controller_record(
                    receipt_path, f"latency job {job_id} receipt"
                )
                != expected_receipt
            ):
                raise ValueError("latency durable submit receipt drift")
        else:
            _write_canonical_json_exclusive(receipt_path, expected_receipt)
        if intent_path.is_file() and not intent_path.is_symlink():
            intent_path.unlink()
        _fsync_directory(lease_root)
        submitted.append(
            {
                "attempt_id": attempt_id,
                "job_id": job_id,
                "run_id": receipt.run_id,
                "submit_payload_sha256": receipt.submit_payload_sha256,
                "submit_response_sha256": receipt.submit_response_sha256,
            }
        )
    expected_names = {"phase-lease.json", "batch-reserved.json"} | {
        f"{item[0]}.receipt.json" for item in jobs
    }
    if {item.name for item in lease_root.iterdir()} != expected_names:
        raise ValueError("resumed latency phase lease directory is not closed")
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "execution_plan_sha256": plan_sha256,
        "jobs": submitted,
        "record_type": PUBLICATION_LATENCY_SUBMISSION_RECORD_TYPE,
        "schema_version": PUBLICATION_LATENCY_SCHEMA_VERSION,
        "wave_index": wave_index,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record, PublicationLatencyWaveSubmissionAuthorization(
        execution_plan_sha256=plan_sha256,
        wave_index=wave_index,
        ledger_path_sha256=ledger_path_sha256,
        predecessor_prefix=predecessor,
        batch_authorization=batch_authorization,
        _issuer=_WAVE_SUBMISSION_AUTHORIZATION_ISSUER,
    )


def _latency_reservation_validator(
    *,
    attempt_id: str,
    payload_sha256: str,
    timeout_seconds: int,
    ledger_path: Path,
    expected_ledger_id: str,
) -> Any:
    def validate(
        reservation: DatabricksClusterHourReservation,
        snapshot: Mapping[str, Any],
    ) -> None:
        if (
            reservation.attempt_id != attempt_id
            or reservation.submit_payload_sha256 != payload_sha256
            or reservation.run_timeout_seconds != timeout_seconds
            or reservation.task_timeout_seconds != (timeout_seconds,)
        ):
            raise ValueError("latency reservation differs from the condition timeout")
        tasks = snapshot.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise ValueError("latency reservation snapshot is not one task")
        live = read_databricks_cluster_hour_ledger_json(ledger_path)
        if live.ledger_id != expected_ledger_id:
            raise ValueError("latency reservation uses a different campaign ledger")
        if (
            live.active_reserved_task_count + len(reservation.task_timeout_seconds)
            > PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
        ):
            raise ValueError(
                "latency reservation exceeds the global 16-job concurrency cap"
            )
        projected_active = (
            live.active_reserved_cluster_hours + reservation.reserved_cluster_hours
        )
        if projected_active > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS:
            raise ValueError("latency reservation exceeds the 900-hour active cap")
        protected_projection = (
            live.terminal_actual_cluster_hours
            + projected_active
            + PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
        )
        if protected_projection > live.cap_cluster_hours:
            raise ValueError(
                "latency reservation would consume the 124-hour unreserved headroom"
            )

    return validate


def collect_publication_latency_launch_wave(
    config: DatabricksWorkspaceConfig,
    *,
    execution_plan_record: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationLatencyHandoffServingAuthorization,
    bf16_handoff_serving_authorization: PublicationBF16HandoffServingAuthorization,
    ledger_path: str | Path,
    wave_index: int,
    submission_authorization: PublicationLatencyWaveSubmissionAuthorization,
) -> tuple[dict[str, Any], PublicationLatencyWaveAuthorization]:
    """Reconcile one completed wave so the next deterministic wave may launch."""

    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
    )
    validate_publication_latency_execution_sources(execution_plan_record)
    waves = _mapping_sequence(execution_plan_record, "launch_waves")
    if type(wave_index) is not int or not 0 <= wave_index < len(waves):
        raise ValueError("wave_index is outside the execution plan")
    wave = waves[wave_index]
    job_ids = wave.get("job_ids")
    if not isinstance(job_ids, list) or any(
        not isinstance(item, str) for item in job_ids
    ):
        raise ValueError("launch wave job IDs are invalid")
    plan_sha256 = _required_sha256(execution_plan_record, "closed_record_sha256")
    if not isinstance(
        submission_authorization, PublicationLatencyWaveSubmissionAuthorization
    ):
        raise TypeError("wave collection requires atomic submission authority")
    if (
        submission_authorization.execution_plan_sha256 != plan_sha256
        or submission_authorization.wave_index != wave_index
    ):
        raise ValueError("wave submission authority binding drift")
    ledger_file = Path(ledger_path).expanduser().absolute()
    ledger = read_databricks_cluster_hour_ledger_json(ledger_file)
    if ledger.ledger_id != _required_string(
        _mapping(execution_plan_record, "sources"), "campaign_ledger_id"
    ):
        raise ValueError("latency ledger differs from the campaign ledger")
    ledger_path_sha256 = databricks_ledger_path_sha256(ledger_file)
    if ledger_path_sha256 != submission_authorization.ledger_path_sha256:
        raise ValueError("wave collection ledger path binding drift")
    expected_attempt_ids: list[str] = []
    expected_payload_digests: list[str] = []
    for job_id in cast(list[str], job_ids):
        expected_job = _render_publication_latency_job_record(
            execution_plan_record, job_id
        )
        expected_attempt_ids.append(
            _required_string(expected_job, "reservation_attempt_id")
        )
        expected_payload_digests.append(
            _submit_payload_sha256(
                _build_databricks_publication_latency_run_submit_payload(
                    execution_plan_record, job_id
                )
            )
        )
    batch_prefix = require_databricks_batch_reservation_authorization(
        submission_authorization.batch_authorization,
        expected_predecessor_prefix=submission_authorization.predecessor_prefix,
        expected_attempt_ids=expected_attempt_ids,
        expected_submit_payload_sha256s=expected_payload_digests,
    )
    require_databricks_ledger_prefix(ledger, batch_prefix)
    receipts_by_attempt = {item.attempt_id: item for item in ledger.submission_receipts}
    terminal_rows: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    seen_tasks: set[str] = set()
    seen_clusters: set[str] = set()
    for job_id in cast(list[str], job_ids):
        job = _render_publication_latency_job_record(execution_plan_record, job_id)
        attempt_id = _required_string(job, "reservation_attempt_id")
        receipt = receipts_by_attempt.get(attempt_id)
        if receipt is None:
            raise ValueError(f"wave job {job_id!r} has no submit receipt")
        payload = _build_databricks_publication_latency_run_submit_payload(
            execution_plan_record, job_id
        )
        run = get_databricks_run(config, receipt.run_id)
        reconciled = record_databricks_verified_run_terminal_actual_json(
            ledger_file,
            attempt_id=attempt_id,
            run_record=run,
        )
        actual = next(
            item
            for item in reconciled.terminal_actuals
            if item.attempt_id == attempt_id
        )
        identity = _validate_latency_control_plane_run(
            run,
            job_record=job,
            submit_payload=payload,
            receipt_run_id=receipt.run_id,
        )
        if (
            actual.terminal_state != "succeeded"
            or identity["terminal_state"] != "succeeded"
        ):
            raise RuntimeError(f"publication latency wave job {job_id!r} failed")
        result_path = _cluster_path(
            _required_string(_mapping(job, "output"), "result_uri")
        )
        result = _read_json_file(result_path, f"latency result {job_id}")
        validate_publication_latency_job_result_record(
            result,
            expected_job_record=job,
            verify_files=True,
        )
        result_identity = _mapping(result, "task_identity")
        if (
            result_identity.get("cloud_run_id") != receipt.run_id
            or result_identity.get("task_run_id") != identity["task_run_id"]
        ):
            raise ValueError("wave result/control-plane identity drift")
        status_sha256 = _control_plane_status_sha256(run)
        if actual.control_plane_status_sha256 != status_sha256:
            raise RuntimeError("wave ledger/control-plane status digest drift")
        for observed, value in (
            (seen_runs, receipt.run_id),
            (seen_tasks, str(identity["task_run_id"])),
            (seen_clusters, str(identity["cluster_id"])),
        ):
            if value in observed:
                raise ValueError("wave physical execution identities must be unique")
            observed.add(value)
        terminal_rows.append(
            {
                "attempt_id": attempt_id,
                "cluster_id": identity["cluster_id"],
                "control_plane_status_sha256": status_sha256,
                "job_id": job_id,
                "ledger_terminal_actual_sha256": _canonical_sha256(
                    _terminal_actual_record(actual)
                ),
                "result_closed_record_sha256": _required_sha256(
                    result, "closed_record_sha256"
                ),
                "result_file_sha256": _file_sha256(result_path),
                "run_id": receipt.run_id,
                "task_run_id": identity["task_run_id"],
            }
        )
    if seen_runs.intersection(seen_tasks):
        raise ValueError("wave parent and task run IDs must be disjoint")
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "execution_plan_sha256": plan_sha256,
        "ledger_path_sha256": ledger_path_sha256,
        "record_type": PUBLICATION_LATENCY_TERMINAL_RECEIPT_RECORD_TYPE,
        "schema_version": PUBLICATION_LATENCY_SCHEMA_VERSION,
        "terminals": terminal_rows,
        "wave_index": wave_index,
    }
    final_ledger = read_databricks_cluster_hour_ledger_json(ledger_file)
    if databricks_ledger_path_sha256(ledger_file) != ledger_path_sha256:
        raise RuntimeError("latency ledger path changed during collection")
    if databricks_ledger_path_sha256(ledger_file) != ledger_path_sha256:
        raise RuntimeError("wave ledger path changed during collection")
    require_databricks_ledger_prefix(final_ledger, batch_prefix)
    for attempt_id in expected_attempt_ids:
        if attempt_id not in final_ledger.closed_attempt_ids:
            raise RuntimeError("wave terminal closure is incomplete")
    ledger_prefix = require_databricks_batch_terminal_closure(
        final_ledger,
        submission_authorization.batch_authorization,
        require_complete_current_prefix=True,
    )
    record["ledger_prefix"] = ledger_prefix.to_record()
    record["closed_record_sha256"] = _closed_record_sha256(record)
    authorization = PublicationLatencyWaveAuthorization(
        execution_plan_sha256=plan_sha256,
        wave_index=wave_index,
        ledger_path_sha256=ledger_path_sha256,
        ledger_prefix=ledger_prefix,
        causal_closure_sha256=_canonical_sha256(
            {
                "batch_prefix": batch_prefix.to_record(),
                "terminal_record_sha256": record["closed_record_sha256"],
            }
        ),
        _issuer=_WAVE_COLLECTION_AUTHORIZATION_ISSUER,
    )
    return record, authorization


@dataclass(frozen=True, slots=True, init=False)
class PublicationLatencyCollectionAuthorization:
    """Non-record authority over one causally collected 115-job closure."""

    collection: Mapping[str, Any]
    collection_sha256: str
    ledger_path_sha256: str
    ledger_prefix: DatabricksLedgerPrefix

    def __init__(
        self,
        *,
        collection: Mapping[str, Any],
        ledger_path_sha256: str,
        ledger_prefix: DatabricksLedgerPrefix,
        _issuer: object,
    ) -> None:
        if _issuer is not _COLLECTION_AUTHORIZATION_ISSUER:
            raise TypeError("latency collection authority must come from the collector")
        normalized = json.loads(_canonical_json(collection))
        if not isinstance(normalized, dict):  # pragma: no cover - mapping above.
            raise TypeError("collection must be an object")
        object.__setattr__(self, "collection", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "collection_sha256",
            _required_sha256(normalized, "closed_record_sha256"),
        )
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _require_sha256_value(ledger_path_sha256, "ledger_path_sha256"),
        )
        if not isinstance(ledger_prefix, DatabricksLedgerPrefix):
            raise TypeError("ledger_prefix has the wrong type")
        ledger_binding = _mapping(normalized, "ledger")
        if ledger_prefix.to_record() != ledger_binding.get("ledger_prefix"):
            raise ValueError("latency collection ledger prefix binding drift")
        object.__setattr__(self, "ledger_prefix", ledger_prefix)


_COLLECTION_AUTHORIZATION_ISSUER = object()


def require_publication_latency_collection_authorization(
    authorization: object,
    *,
    execution_plan_record: Mapping[str, Any],
    ledger_path: str | Path,
) -> DatabricksLedgerPrefix:
    """Replay the post-latency capability against the canonical live ledger."""

    if not isinstance(authorization, PublicationLatencyCollectionAuthorization):
        raise TypeError(
            "publication launch requires PublicationLatencyCollectionAuthorization"
        )
    validate_publication_latency_collection_record(
        authorization.collection,
        execution_plan_record=execution_plan_record,
    )
    if authorization.collection.get("closed_record_sha256") != (
        authorization.collection_sha256
    ):
        raise ValueError("latency collection capability was mutated")
    path_sha256 = databricks_ledger_path_sha256(ledger_path)
    if path_sha256 != authorization.ledger_path_sha256:
        raise ValueError("latency collection ledger path binding drift")
    live = read_databricks_cluster_hour_ledger_json(ledger_path)
    require_databricks_ledger_prefix(live, authorization.ledger_prefix)
    if databricks_ledger_prefix(live) != authorization.ledger_prefix:
        raise ValueError("latency collection is not the latest live ledger prefix")
    return authorization.ledger_prefix


def collect_publication_latency_campaign(
    config: DatabricksWorkspaceConfig,
    *,
    execution_plan_record: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationLatencyHandoffServingAuthorization,
    bf16_handoff_serving_authorization: PublicationBF16HandoffServingAuthorization,
    final_wave_authorization: PublicationLatencyWaveAuthorization,
    ledger_path: str | Path,
) -> tuple[dict[str, Any], PublicationLatencyCollectionAuthorization]:
    """Join runs/get, receipt ledger, and all 115 sealed result artifacts."""

    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
    )
    validate_publication_latency_execution_sources(execution_plan_record)
    ledger_file = Path(ledger_path)
    initial_ledger = read_databricks_cluster_hour_ledger_json(ledger_file)
    campaign_ledger_id = _required_string(
        _mapping(execution_plan_record, "sources"), "campaign_ledger_id"
    )
    if initial_ledger.ledger_id != campaign_ledger_id:
        raise ValueError("latency ledger differs from the campaign ledger")
    if not isinstance(final_wave_authorization, PublicationLatencyWaveAuthorization):
        raise TypeError("campaign collection requires final-wave authority")
    plan_sha256 = _required_sha256(execution_plan_record, "closed_record_sha256")
    waves = _mapping_sequence(execution_plan_record, "launch_waves")
    ledger_path_sha256 = databricks_ledger_path_sha256(ledger_file)
    if (
        final_wave_authorization.execution_plan_sha256 != plan_sha256
        or final_wave_authorization.wave_index != len(waves) - 1
        or final_wave_authorization.ledger_path_sha256 != ledger_path_sha256
    ):
        raise ValueError("final latency wave authority binding drift")
    require_databricks_ledger_prefix(
        initial_ledger, final_wave_authorization.ledger_prefix
    )
    if databricks_ledger_prefix(initial_ledger) != (
        final_wave_authorization.ledger_prefix
    ):
        raise ValueError("final wave prefix is not the complete live ledger")
    jobs = _mapping_sequence(execution_plan_record, "jobs")
    expected_attempts = {
        publication_latency_reservation_attempt_id(
            _required_sha256(execution_plan_record, "closed_record_sha256"),
            _required_string(job, "job_id"),
        )
        for job in jobs
    }
    attempt_prefix = (
        "publication-latency/"
        + _required_sha256(execution_plan_record, "closed_record_sha256")[:20]
        + "/"
    )
    if {
        item.attempt_id
        for item in initial_ledger.reservations
        if item.attempt_id.startswith(attempt_prefix)
    } != expected_attempts or {
        item.attempt_id
        for item in initial_ledger.submission_receipts
        if item.attempt_id.startswith(attempt_prefix)
    } != expected_attempts:
        raise ValueError(
            "latency ledger scoped attempt closure is not exactly 115 jobs"
        )
    campaign_reservations = {
        item.attempt_id: item
        for item in initial_ledger.reservations
        if item.attempt_id in expected_attempts
    }
    campaign_receipts = {
        item.attempt_id: item
        for item in initial_ledger.submission_receipts
        if item.attempt_id in expected_attempts
    }
    if (
        set(campaign_reservations) != expected_attempts
        or set(campaign_receipts) != expected_attempts
    ):
        raise ValueError("latency ledger does not contain the exact 115 receipts")

    terminal_receipts: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    seen_task_run_ids: set[str] = set()
    seen_clusters: set[str] = set()
    for descriptor in jobs:
        job_id = _required_string(descriptor, "job_id")
        job = _render_publication_latency_job_record(execution_plan_record, job_id)
        attempt_id = _required_string(job, "reservation_attempt_id")
        receipt = campaign_receipts[attempt_id]
        payload = _build_databricks_publication_latency_run_submit_payload(
            execution_plan_record, job_id
        )
        run = get_databricks_run(config, receipt.run_id)
        reconciled = record_databricks_verified_run_terminal_actual_json(
            ledger_file,
            attempt_id=attempt_id,
            run_record=run,
        )
        actual = next(
            item
            for item in reconciled.terminal_actuals
            if item.attempt_id == attempt_id
        )
        identity = _validate_latency_control_plane_run(
            run,
            job_record=job,
            submit_payload=payload,
            receipt_run_id=receipt.run_id,
        )
        if (
            identity["terminal_state"] != "succeeded"
            or actual.terminal_state != "succeeded"
        ):
            raise RuntimeError(f"publication latency job {job_id!r} did not succeed")
        status_sha256 = _control_plane_status_sha256(run)
        if (
            actual.verification_source != "direct_databricks_runs_get"
            or actual.run_id != receipt.run_id
            or actual.submit_payload_sha256 != receipt.submit_payload_sha256
            or actual.control_plane_status_sha256 != status_sha256
        ):
            raise RuntimeError("latency ledger terminal causal binding drift")
        result_path = _cluster_path(
            _required_string(_mapping(job, "output"), "result_uri")
        )
        result = _read_json_file(result_path, f"latency result {job_id}")
        validate_publication_latency_job_result_record(
            result,
            expected_job_record=job,
            verify_files=True,
        )
        result_identity = _mapping(result, "task_identity")
        if (
            result_identity.get("cloud_run_id") != receipt.run_id
            or result_identity.get("task_run_id") != identity["task_run_id"]
            or result_identity.get("task_key") != job.get("task_key")
        ):
            raise ValueError("latency result does not match the control-plane task")
        for observed, value, label in (
            (seen_run_ids, receipt.run_id, "run ID"),
            (seen_task_run_ids, str(identity["task_run_id"]), "task run ID"),
            (seen_clusters, str(identity["cluster_id"]), "cluster ID"),
        ):
            if value in observed:
                raise ValueError(f"latency {label} values must be unique")
            observed.add(value)
        actual_record = _terminal_actual_record(actual)
        terminal_receipt: dict[str, Any] = {
            "attempt_id": attempt_id,
            "cluster_id": identity["cluster_id"],
            "control_plane_status_sha256": status_sha256,
            "job_id": job_id,
            "ledger_terminal_actual": actual_record,
            "ledger_terminal_actual_sha256": _canonical_sha256(actual_record),
            "result_closed_record_sha256": _required_sha256(
                result, "closed_record_sha256"
            ),
            "result_file_sha256": _file_sha256(result_path),
            "run_id": receipt.run_id,
            "submit_payload_sha256": receipt.submit_payload_sha256,
            "task_run_id": identity["task_run_id"],
        }
        terminal_receipts.append(terminal_receipt)
        results.append(result)

    final_ledger = read_databricks_cluster_hour_ledger_json(ledger_file)
    if final_ledger.ledger_id != campaign_ledger_id:
        raise RuntimeError("latency campaign ledger identity changed during collection")
    if databricks_ledger_path_sha256(ledger_file) != ledger_path_sha256:
        raise RuntimeError("latency ledger path changed during campaign collection")
    require_databricks_ledger_prefix(
        final_ledger, final_wave_authorization.ledger_prefix
    )
    campaign_actuals = {
        item.attempt_id: item
        for item in final_ledger.terminal_actuals
        if item.attempt_id in expected_attempts
    }
    if set(campaign_actuals) != expected_attempts:
        raise RuntimeError("latency terminal ledger closure is incomplete")
    if {
        item.attempt_id
        for item in final_ledger.terminal_actuals
        if item.attempt_id.startswith(attempt_prefix)
    } != expected_attempts:
        raise RuntimeError(
            "latency terminal ledger contains an unplanned scoped attempt"
        )
    if seen_run_ids.intersection(seen_task_run_ids):
        raise ValueError("latency parent and task run ID domains must be disjoint")
    final_prefix = databricks_ledger_prefix(final_ledger)
    if final_prefix != final_wave_authorization.ledger_prefix:
        raise RuntimeError("latency ledger changed after final-wave authorization")
    record: dict[str, Any] = {
        "campaign_id": _required_string(execution_plan_record, "campaign_id"),
        "closed_record_sha256": "",
        "execution_plan_sha256": _required_sha256(
            execution_plan_record, "closed_record_sha256"
        ),
        "job_count": len(results),
        "ledger": {
            "ledger_id": final_ledger.ledger_id,
            "ledger_path_sha256": ledger_path_sha256,
            "ledger_prefix": final_prefix.to_record(),
            "terminal_actuals_sha256": _canonical_sha256(
                [
                    _terminal_actual_record(campaign_actuals[attempt_id])
                    for attempt_id in sorted(expected_attempts)
                ]
            ),
        },
        "record_type": PUBLICATION_LATENCY_COLLECTION_RECORD_TYPE,
        "results": results,
        "schema_version": PUBLICATION_LATENCY_SCHEMA_VERSION,
        "terminal_receipts": terminal_receipts,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    validate_publication_latency_collection_record(
        record,
        execution_plan_record=execution_plan_record,
    )
    authorization = PublicationLatencyCollectionAuthorization(
        collection=record,
        ledger_path_sha256=ledger_path_sha256,
        ledger_prefix=final_prefix,
        _issuer=_COLLECTION_AUTHORIZATION_ISSUER,
    )
    return record, authorization


def validate_publication_latency_collection_record(
    record: Mapping[str, Any],
    *,
    execution_plan_record: Mapping[str, Any],
) -> None:
    validate_publication_latency_execution_plan_record(execution_plan_record)
    _require_exact_keys(
        record,
        {
            "campaign_id",
            "closed_record_sha256",
            "execution_plan_sha256",
            "job_count",
            "ledger",
            "record_type",
            "results",
            "schema_version",
            "terminal_receipts",
        },
        "publication latency collection",
    )
    if (
        record.get("record_type") != PUBLICATION_LATENCY_COLLECTION_RECORD_TYPE
        or record.get("schema_version") != PUBLICATION_LATENCY_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("publication latency collection envelope is invalid")
    if (
        record.get("campaign_id") != execution_plan_record.get("campaign_id")
        or record.get("execution_plan_sha256")
        != execution_plan_record.get("closed_record_sha256")
        or record.get("job_count") != PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS
    ):
        raise ValueError("publication latency collection plan binding drift")
    ledger_binding = _mapping(record, "ledger")
    _require_exact_keys(
        ledger_binding,
        {
            "ledger_id",
            "ledger_path_sha256",
            "ledger_prefix",
            "terminal_actuals_sha256",
        },
        "publication latency collection ledger",
    )
    if ledger_binding.get("ledger_id") != _required_string(
        _mapping(execution_plan_record, "sources"), "campaign_ledger_id"
    ):
        raise ValueError("publication latency collection uses a different ledger")
    if ledger_binding.get("ledger_path_sha256") != _required_sha256(
        _mapping(execution_plan_record, "sources"),
        "campaign_ledger_path_sha256",
    ):
        raise ValueError("publication latency collection uses a different ledger path")
    collection_prefix = databricks_ledger_prefix_from_record(
        _mapping(ledger_binding, "ledger_prefix")
    )
    if collection_prefix.ledger_id != ledger_binding.get("ledger_id"):
        raise ValueError("publication latency collection prefix identity drift")
    results = _mapping_sequence(record, "results")
    receipts = _mapping_sequence(record, "terminal_receipts")
    jobs = _mapping_sequence(execution_plan_record, "jobs")
    if len(results) != len(jobs) or len(receipts) != len(jobs):
        raise ValueError("publication latency collection is incomplete")
    for descriptor, result, receipt in zip(jobs, results, receipts, strict=True):
        job_id = _required_string(descriptor, "job_id")
        job = _render_publication_latency_job_record(execution_plan_record, job_id)
        validate_publication_latency_job_result_record(
            result,
            expected_job_record=job,
            verify_files=False,
        )
        if (
            receipt.get("job_id") != job_id
            or receipt.get("result_closed_record_sha256")
            != result.get("closed_record_sha256")
            or receipt.get("submit_payload_sha256")
            != _submit_payload_sha256(
                _build_databricks_publication_latency_run_submit_payload(
                    execution_plan_record, job_id
                )
            )
        ):
            raise ValueError("publication latency terminal receipt binding drift")
        for field_name in (
            "control_plane_status_sha256",
            "ledger_terminal_actual_sha256",
            "result_closed_record_sha256",
            "result_file_sha256",
            "submit_payload_sha256",
        ):
            _required_sha256(receipt, field_name)
        if receipt.get("ledger_terminal_actual_sha256") != _canonical_sha256(
            _mapping(receipt, "ledger_terminal_actual")
        ):
            raise ValueError("publication latency terminal actual digest drift")
    if (
        len({item.get("run_id") for item in receipts}) != len(receipts)
        or len({item.get("task_run_id") for item in receipts}) != len(receipts)
        or len({item.get("cluster_id") for item in receipts}) != len(receipts)
    ):
        raise ValueError(
            "publication latency collected physical identities are not unique"
        )


def aggregate_publication_latency_campaign(
    authorization: PublicationLatencyCollectionAuthorization,
    *,
    execution_plan_record: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationLatencyHandoffServingAuthorization,
    bf16_handoff_serving_authorization: PublicationBF16HandoffServingAuthorization,
) -> dict[str, Any]:
    """Produce estimation-only paired hierarchical bootstrap summaries."""

    if not isinstance(authorization, PublicationLatencyCollectionAuthorization):
        raise TypeError(
            "latency aggregation requires PublicationLatencyCollectionAuthorization"
        )
    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
    )
    validate_publication_latency_execution_sources(execution_plan_record)
    collection = json.loads(_canonical_json(authorization.collection))
    if not isinstance(collection, dict):  # pragma: no cover - normalized capability.
        raise TypeError("authorized latency collection is invalid")
    if collection.get("closed_record_sha256") != authorization.collection_sha256:
        raise ValueError("latency collection capability was mutated")
    validate_publication_latency_collection_record(
        collection,
        execution_plan_record=execution_plan_record,
    )
    descriptors = {
        _required_string(item, "job_id"): item
        for item in _mapping_sequence(execution_plan_record, "jobs")
    }
    result_by_job = {
        _required_string(item, "job_id"): item
        for item in _mapping_sequence(collection, "results")
    }
    estimand_specs: list[dict[str, Any]] = []
    for input_tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
        for concurrency in (1, 2, 4):
            control_by_block: dict[int, str] = {}
            treatment_by_block: dict[int, str] = {}
            for job_id, descriptor in descriptors.items():
                if (
                    descriptor.get("job_kind") == "core"
                    and descriptor.get("input_tokens") == input_tokens
                    and descriptor.get("request_parallelism") == concurrency
                ):
                    target = (
                        control_by_block
                        if descriptor.get("method_id") == "baseline_prefill"
                        else treatment_by_block
                    )
                    target[_required_int(descriptor, "deployment_block")] = job_id
            estimand_specs.append(
                {
                    "comparison_family": "method",
                    "control_jobs": control_by_block,
                    "estimand_id": f"method-{input_tokens}-c{concurrency}",
                    "input_tokens": input_tokens,
                    "request_parallelism": concurrency,
                    "treatment_jobs": treatment_by_block,
                }
            )
    for setting_id, family, _description in PUBLICATION_CAMPAIGN_AUXILIARY_SETTINGS:
        control_by_block = {}
        treatment_by_block = {}
        for job_id, descriptor in descriptors.items():
            if descriptor.get("setting_id") != setting_id:
                continue
            block = _required_int(descriptor, "deployment_block")
            treatment_by_block[block] = job_id
            control_by_block[block] = _required_string(
                descriptor, "reference_core_cell_id"
            )
        estimand_specs.append(
            {
                "comparison_family": family,
                "control_jobs": control_by_block,
                "estimand_id": f"auxiliary-{setting_id}",
                "input_tokens": 16_384,
                "request_parallelism": 4,
                "setting_id": setting_id,
                "treatment_jobs": treatment_by_block,
            }
        )
    if len(estimand_specs) != 13:
        raise RuntimeError("latency estimand design does not contain 13 families")

    descriptive_cells = _publication_latency_descriptive_cells(
        descriptors=descriptors,
        result_by_job=result_by_job,
    )

    estimates: list[dict[str, Any]] = []
    for spec in estimand_specs:
        paired_logs = _paired_log_ratios_by_block(
            spec,
            result_by_job=result_by_job,
        )
        metric_records: dict[str, Any] = {}
        for metric_name in ("ttft_seconds", "time_to_completion_seconds"):
            seed = int(
                _canonical_sha256(
                    {
                        "collection_sha256": authorization.collection_sha256,
                        "domain": "cachet.publication_latency.bootstrap.v1",
                        "estimand_id": spec["estimand_id"],
                        "metric": metric_name,
                    }
                )[:16],
                16,
            )
            point, lower, upper = _paired_hierarchical_bootstrap(
                paired_logs[metric_name],
                draws=PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS,
                seed=seed,
            )
            metric_records[metric_name.removesuffix("_seconds")] = {
                "confidence_interval_95": [lower, upper],
                "geometric_mean_speedup": point,
            }
        estimates.append(
            {
                **{
                    key: value
                    for key, value in spec.items()
                    if key not in {"control_jobs", "treatment_jobs"}
                },
                "deployment_block_count": PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS,
                "example_count_per_block": (
                    len(SUPPORTED_V1_DATASETS)
                    * (
                        PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET
                        if spec.get("setting_id") in {"storage-ram", "storage-uc"}
                        else PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
                    )
                ),
                "paired_request_count": (
                    PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS
                    * PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL
                ),
                "speedup_direction": "control_latency_divided_by_treatment_latency",
                "metrics": metric_records,
            }
        )
    summary: dict[str, Any] = {
        "analysis": {
            "bootstrap": "paired_hierarchical_deployment_block_and_example",
            "bootstrap_draws": PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS,
            "confidence_intervals": "pointwise_95_percent",
            "decision_mode": "estimation_only",
            "null_hypothesis_rejections": False,
            "post_hoc_significance_claims": False,
            "descriptive_quantiles": "empirical_nearest_rank",
            "storage_workload": "2_examples_per_dataset_x_32_repeats",
        },
        "campaign_id": _required_string(collection, "campaign_id"),
        "closed_record_sha256": "",
        "collection_sha256": authorization.collection_sha256,
        "descriptive_cell_count": len(descriptive_cells),
        "descriptive_cells": descriptive_cells,
        "estimand_count": len(estimates),
        "estimates": estimates,
        "execution_plan_sha256": _required_sha256(
            execution_plan_record, "closed_record_sha256"
        ),
        "record_type": PUBLICATION_LATENCY_SUMMARY_RECORD_TYPE,
        "schema_version": PUBLICATION_LATENCY_SCHEMA_VERSION,
    }
    summary["closed_record_sha256"] = _closed_record_sha256(summary)
    validate_publication_latency_summary_record(
        summary,
        expected_collection_sha256=authorization.collection_sha256,
        expected_execution_plan_sha256=_required_sha256(
            execution_plan_record, "closed_record_sha256"
        ),
    )
    return summary


def validate_publication_latency_summary_record(
    record: Mapping[str, Any],
    *,
    expected_collection_sha256: str,
    expected_execution_plan_sha256: str,
) -> None:
    _require_exact_keys(
        record,
        {
            "analysis",
            "campaign_id",
            "closed_record_sha256",
            "collection_sha256",
            "descriptive_cell_count",
            "descriptive_cells",
            "estimand_count",
            "estimates",
            "execution_plan_sha256",
            "record_type",
            "schema_version",
        },
        "publication latency summary",
    )
    if (
        record.get("record_type") != PUBLICATION_LATENCY_SUMMARY_RECORD_TYPE
        or record.get("schema_version") != PUBLICATION_LATENCY_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("publication latency summary envelope is invalid")
    if record.get("collection_sha256") != _require_sha256_value(
        expected_collection_sha256, "expected_collection_sha256"
    ) or record.get("execution_plan_sha256") != _require_sha256_value(
        expected_execution_plan_sha256, "expected_execution_plan_sha256"
    ):
        raise ValueError("publication latency summary source binding drift")
    estimates = _mapping_sequence(record, "estimates")
    if record.get("estimand_count") != 13 or len(estimates) != 13:
        raise ValueError("publication latency summary must contain 13 estimands")
    descriptive_cells = _mapping_sequence(record, "descriptive_cells")
    expected_descriptive_count = len(PUBLICATION_CAMPAIGN_CONTEXT_TOKENS) * 3 * 2 + len(
        _DESCRIPTIVE_AUXILIARY_SETTING_IDS
    )
    if (
        record.get("descriptive_cell_count") != expected_descriptive_count
        or len(descriptive_cells) != expected_descriptive_count
    ):
        raise ValueError("publication latency descriptive-cell closure is incomplete")
    for cell in descriptive_cells:
        _validate_descriptive_cell_record(cell)
    expected_descriptive_ids = [
        f"core-{method_id}-{tokens}-c{concurrency}"
        for tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS
        for concurrency in (1, 2, 4)
        for method_id in ("baseline_prefill", "vanilla_prefill")
    ] + [f"auxiliary-{setting_id}" for setting_id in _DESCRIPTIVE_AUXILIARY_SETTING_IDS]
    if [cell.get("cell_id") for cell in descriptive_cells] != (
        expected_descriptive_ids
    ):
        raise ValueError("publication latency descriptive-cell order drift")
    analysis = _mapping(record, "analysis")
    if analysis != {
        "bootstrap": "paired_hierarchical_deployment_block_and_example",
        "bootstrap_draws": PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS,
        "confidence_intervals": "pointwise_95_percent",
        "decision_mode": "estimation_only",
        "null_hypothesis_rejections": False,
        "post_hoc_significance_claims": False,
        "descriptive_quantiles": "empirical_nearest_rank",
        "storage_workload": "2_examples_per_dataset_x_32_repeats",
    }:
        raise ValueError("publication latency summary inference policy drift")
    ids = [item.get("estimand_id") for item in estimates]
    expected_ids = [
        f"method-{tokens}-c{concurrency}"
        for tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS
        for concurrency in (1, 2, 4)
    ] + [
        f"auxiliary-{setting_id}"
        for setting_id, _family, _description in PUBLICATION_CAMPAIGN_AUXILIARY_SETTINGS
    ]
    if ids != expected_ids:
        raise ValueError("publication latency summary estimand order drift")
    for estimate in estimates:
        metrics = _mapping(estimate, "metrics")
        if set(metrics) != {"ttft", "time_to_completion"}:
            raise ValueError("publication latency summary metric closure drift")
        for metric in metrics.values():
            metric_record = _mapping_value(metric, "summary metric")
            point = _finite_positive_number(
                metric_record.get("geometric_mean_speedup"), "speedup"
            )
            interval = metric_record.get("confidence_interval_95")
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or any(
                    not math.isfinite(float(value)) or float(value) <= 0
                    for value in interval
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                )
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in interval
                )
                or float(interval[0]) > float(interval[1])
            ):
                raise ValueError("publication latency confidence interval is invalid")
            if not math.isfinite(point):  # pragma: no cover - helper already checks.
                raise ValueError("publication latency speedup is invalid")


def _publication_latency_descriptive_cells(
    *,
    descriptors: Mapping[str, Mapping[str, Any]],
    result_by_job: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
        for concurrency in (1, 2, 4):
            for method_id in ("baseline_prefill", "vanilla_prefill"):
                job_ids = [
                    job_id
                    for job_id, descriptor in descriptors.items()
                    if descriptor.get("job_kind") == "core"
                    and descriptor.get("input_tokens") == input_tokens
                    and descriptor.get("request_parallelism") == concurrency
                    and descriptor.get("method_id") == method_id
                ]
                rows.append(
                    _descriptive_cell_record(
                        cell_id=f"core-{method_id}-{input_tokens}-c{concurrency}",
                        cell_kind="core_pooled_five_blocks",
                        descriptors=descriptors,
                        result_by_job=result_by_job,
                        job_ids=job_ids,
                    )
                )
    for setting_id in _DESCRIPTIVE_AUXILIARY_SETTING_IDS:
        job_ids = [
            job_id
            for job_id, descriptor in descriptors.items()
            if descriptor.get("job_kind") == "auxiliary"
            and descriptor.get("setting_id") == setting_id
        ]
        rows.append(
            _descriptive_cell_record(
                cell_id=f"auxiliary-{setting_id}",
                cell_kind="auxiliary_pooled_five_blocks",
                descriptors=descriptors,
                result_by_job=result_by_job,
                job_ids=job_ids,
            )
        )
    return rows


def _descriptive_cell_record(
    *,
    cell_id: str,
    cell_kind: str,
    descriptors: Mapping[str, Mapping[str, Any]],
    result_by_job: Mapping[str, Mapping[str, Any]],
    job_ids: Sequence[str],
) -> dict[str, Any]:
    expected_blocks = (
        5
        if cell_kind in {"core_pooled_five_blocks", "auxiliary_pooled_five_blocks"}
        else 0
    )
    if expected_blocks == 0:
        raise ValueError("descriptive cell kind is outside the frozen design")
    if len(job_ids) != expected_blocks:
        raise ValueError("descriptive cell physical-block coverage drift")
    ordered_ids = sorted(
        job_ids,
        key=lambda job_id: _required_int(descriptors[job_id], "deployment_block"),
    )
    block_values: list[dict[str, Any]] = []
    all_ttft: list[float] = []
    all_ttc: list[float] = []
    all_decode: list[float] = []
    resources: list[Any] = []
    concurrency: int | None = None
    for job_id in ordered_ids:
        descriptor = descriptors[job_id]
        result_record = result_by_job.get(job_id)
        if result_record is None:
            raise ValueError("descriptive cell references a missing job result")
        benchmark = benchmark_run_result_from_record(
            _mapping(result_record, "benchmark_record"), evidence_policy="publication"
        )
        configured_concurrency = benchmark.request_parallelism
        if concurrency is None:
            concurrency = configured_concurrency
        elif concurrency != configured_concurrency:
            raise ValueError("descriptive cell serving concurrency drift")
        ttft = [float(item.ttft_seconds) for item in benchmark.measurements]
        ttc = [
            float(item.time_to_completion_seconds) for item in benchmark.measurements
        ]
        decode = [_required_decode_rate(item) for item in benchmark.measurements]
        if len(benchmark.resource_evidence) != 1:
            raise ValueError("descriptive cell requires one resource record per block")
        resource = benchmark.resource_evidence[0]
        resources.append(resource)
        all_ttft.extend(ttft)
        all_ttc.extend(ttc)
        all_decode.extend(decode)
        block_values.append(
            {
                "deployment_block": _required_int(descriptor, "deployment_block"),
                "job_id": job_id,
                "observation_count": len(ttft),
                "configured_closed_loop_concurrency": configured_concurrency,
                "p50_decode_tokens_per_second": _empirical_nearest_rank(decode, 0.50),
                "p50_time_to_completion_seconds": _empirical_nearest_rank(ttc, 0.50),
                "p50_ttft_seconds": _empirical_nearest_rank(ttft, 0.50),
                "p95_time_to_completion_seconds": _empirical_nearest_rank(ttc, 0.95),
                "p95_ttft_seconds": _empirical_nearest_rank(ttft, 0.95),
                "peak_gpu_process_memory_bytes": resource.peak_gpu_process_memory_bytes,
                "peak_host_memory_used_bytes": resource.peak_host_memory_used_bytes,
                "peak_process_tree_rss_bytes": resource.peak_process_tree_rss_bytes,
            }
        )
    assert concurrency is not None
    first = descriptors[ordered_ids[0]]
    record: dict[str, Any] = {
        "cell_id": cell_id,
        "cell_kind": cell_kind,
        "cell_sha256": "",
        "comparison_family": first.get("comparison_family"),
        "input_tokens": _required_int(first, "input_tokens"),
        "method_id": _required_string(first, "method_id"),
        "observation_count": len(all_ttft),
        "configured_closed_loop_concurrency": concurrency,
        "p50_decode_tokens_per_second": _empirical_nearest_rank(all_decode, 0.50),
        "p50_time_to_completion_seconds": _empirical_nearest_rank(all_ttc, 0.50),
        "p50_ttft_seconds": _empirical_nearest_rank(all_ttft, 0.50),
        "p95_time_to_completion_seconds": _empirical_nearest_rank(all_ttc, 0.95),
        "p95_ttft_seconds": _empirical_nearest_rank(all_ttft, 0.95),
        "peak_gpu_process_memory_bytes": max(
            item.peak_gpu_process_memory_bytes for item in resources
        ),
        "peak_host_memory_used_bytes": max(
            item.peak_host_memory_used_bytes for item in resources
        ),
        "peak_process_tree_rss_bytes": max(
            item.peak_process_tree_rss_bytes for item in resources
        ),
        "physical_blocks": block_values,
        "quantile_method": "empirical_nearest_rank",
        "request_parallelism": _required_int(first, "request_parallelism"),
        "setting_id": first.get("setting_id"),
    }
    record["cell_sha256"] = _descriptive_cell_sha256(record)
    return record


def _validate_descriptive_cell_record(record: Mapping[str, Any]) -> None:
    metric_fields = {
        "configured_closed_loop_concurrency",
        "observation_count",
        "p50_decode_tokens_per_second",
        "p50_time_to_completion_seconds",
        "p50_ttft_seconds",
        "p95_time_to_completion_seconds",
        "p95_ttft_seconds",
        "peak_gpu_process_memory_bytes",
        "peak_host_memory_used_bytes",
        "peak_process_tree_rss_bytes",
    }
    _require_exact_keys(
        record,
        metric_fields
        | {
            "cell_id",
            "cell_kind",
            "cell_sha256",
            "comparison_family",
            "input_tokens",
            "method_id",
            "physical_blocks",
            "quantile_method",
            "request_parallelism",
            "setting_id",
        },
        "publication latency descriptive cell",
    )
    recorded_sha = _required_sha256(record, "cell_sha256")
    if recorded_sha != _descriptive_cell_sha256(record):
        raise ValueError("descriptive cell digest drift")
    if record.get("quantile_method") != "empirical_nearest_rank":
        raise ValueError("descriptive cell quantile method drift")
    blocks = _mapping_sequence(record, "physical_blocks")
    if record.get("cell_kind") not in {
        "core_pooled_five_blocks",
        "auxiliary_pooled_five_blocks",
    }:
        raise ValueError("descriptive cell kind is outside the frozen design")
    expected_blocks = 5
    if len(blocks) != expected_blocks:
        raise ValueError("descriptive cell physical-block count drift")
    for block in blocks:
        _require_exact_keys(
            block,
            metric_fields | {"deployment_block", "job_id"},
            "publication latency descriptive physical block",
        )
    if [block.get("deployment_block") for block in blocks] != list(
        range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)
    ) or len({block.get("job_id") for block in blocks}) != len(blocks):
        raise ValueError("descriptive cell block identity/order drift")
    for container in (record, *blocks):
        _finite_positive_number(
            container.get("p50_decode_tokens_per_second"),
            "p50_decode_tokens_per_second",
        )
        p50_ttc = _finite_positive_number(
            container.get("p50_time_to_completion_seconds"),
            "p50_time_to_completion_seconds",
        )
        p50_ttft = _finite_positive_number(
            container.get("p50_ttft_seconds"),
            "p50_ttft_seconds",
        )
        p95_ttc = _finite_positive_number(
            container.get("p95_time_to_completion_seconds"),
            "p95_time_to_completion_seconds",
        )
        p95_ttft = _finite_positive_number(
            container.get("p95_ttft_seconds"),
            "p95_ttft_seconds",
        )
        for field_name in (
            "observation_count",
            "configured_closed_loop_concurrency",
            "peak_gpu_process_memory_bytes",
            "peak_host_memory_used_bytes",
            "peak_process_tree_rss_bytes",
        ):
            _nonnegative_int(container.get(field_name), field_name)
        if (
            p50_ttft > p95_ttft
            or p50_ttc > p95_ttc
            or p50_ttft > p50_ttc
            or (p95_ttft > p95_ttc)
        ):
            raise ValueError("descriptive cell quantile ordering is invalid")
    if (
        record.get("observation_count")
        != sum(_required_int(block, "observation_count") for block in blocks)
        or record.get("configured_closed_loop_concurrency")
        != record.get("request_parallelism")
        or any(
            block.get("configured_closed_loop_concurrency")
            != record.get("request_parallelism")
            for block in blocks
        )
    ):
        raise ValueError("descriptive cell configured concurrency/count drift")
    for peak_field in (
        "peak_gpu_process_memory_bytes",
        "peak_host_memory_used_bytes",
        "peak_process_tree_rss_bytes",
    ):
        if record.get(peak_field) != max(
            _required_int(block, peak_field) for block in blocks
        ):
            raise ValueError("descriptive cell pooled resource peak drift")


def _required_decode_rate(measurement: Any) -> float:
    value = request_decode_tokens_per_second(
        measurement.completion_tokens,
        measurement.ttft_seconds,
        measurement.time_to_completion_seconds,
    )
    if value is None:
        raise ValueError("publication latency decode duration must be positive")
    return _finite_positive_number(value, "decode tokens per second")


def _descriptive_cell_sha256(record: Mapping[str, Any]) -> str:
    return _canonical_sha256({**dict(record), "cell_sha256": ""})


def _empirical_nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(
        _finite_positive_number(value, "empirical observation") for value in values
    )
    if not ordered:
        raise ValueError("empirical quantile requires observations")
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _paired_log_ratios_by_block(
    spec: Mapping[str, Any],
    *,
    result_by_job: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[int, dict[tuple[str, str], tuple[float, ...]]]]:
    control_jobs = {
        int(key): str(value) for key, value in _mapping(spec, "control_jobs").items()
    }
    treatment_jobs = {
        int(key): str(value) for key, value in _mapping(spec, "treatment_jobs").items()
    }
    output: dict[str, dict[int, dict[tuple[str, str], tuple[float, ...]]]] = {
        "ttft_seconds": {},
        "time_to_completion_seconds": {},
    }
    expected_blocks = set(range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1))
    if set(control_jobs) != expected_blocks or set(treatment_jobs) != expected_blocks:
        raise ValueError("latency estimand does not cover all deployment blocks")
    for block in sorted(expected_blocks):
        control_id = control_jobs[block]
        treatment_id = treatment_jobs[block]
        if control_id not in result_by_job or treatment_id not in result_by_job:
            raise ValueError("latency estimand references a missing job result")
        control = benchmark_run_result_from_record(
            _mapping(result_by_job[control_id], "benchmark_record"),
            evidence_policy="publication",
        )
        treatment = benchmark_run_result_from_record(
            _mapping(result_by_job[treatment_id], "benchmark_record"),
            evidence_policy="publication",
        )
        control_by_key = {
            (item.dataset, item.example_id, item.repeat_index): item
            for item in control.measurements
        }
        treatment_by_key = {
            (item.dataset, item.example_id, item.repeat_index): item
            for item in treatment.measurements
        }
        if (
            set(control_by_key) != set(treatment_by_key)
            or len(control_by_key) != (len(control.measurements))
            or len(control_by_key) != (len(treatment.measurements))
        ):
            raise ValueError("latency paired request membership drift")
        for metric_name in output:
            by_example: dict[tuple[str, str], list[float]] = defaultdict(list)
            for key in sorted(control_by_key):
                control_value = _finite_positive_number(
                    getattr(control_by_key[key], metric_name), metric_name
                )
                treatment_value = _finite_positive_number(
                    getattr(treatment_by_key[key], metric_name), metric_name
                )
                by_example[key[:2]].append(math.log(control_value / treatment_value))
            storage_comparison = spec.get("setting_id") in {
                "storage-ram",
                "storage-uc",
            }
            expected_examples_per_dataset = (
                PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET
                if storage_comparison
                else PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
            )
            expected_repeats = (
                PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE
                if storage_comparison
                else PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE
            )
            counts = {
                dataset: sum(
                    1
                    for dataset_key, _example_id in by_example
                    if dataset_key == dataset
                )
                for dataset in SUPPORTED_V1_DATASETS
            }
            if any(
                counts[dataset] != expected_examples_per_dataset
                for dataset in SUPPORTED_V1_DATASETS
            ) or any(
                len(repeats) != expected_repeats for repeats in by_example.values()
            ):
                raise ValueError("latency paired example/repeat closure drift")
            output[metric_name][block] = {
                key: tuple(values) for key, values in by_example.items()
            }
    return output


def _paired_hierarchical_bootstrap(
    by_block: Mapping[int, Mapping[tuple[str, str], Sequence[float]]],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    if set(by_block) != set(range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)):
        raise ValueError("hierarchical bootstrap requires exactly five blocks")
    if type(draws) is not int or draws <= 0:
        raise ValueError("bootstrap draws must be a positive integer")
    block_ids = sorted(by_block)
    example_ids = {
        block: _dataset_stratified_example_identities(by_block[block])
        for block in block_ids
    }
    all_logs = [
        value
        for block in block_ids
        for dataset in SUPPORTED_V1_DATASETS
        for example in example_ids[block][dataset]
        for value in by_block[block][example]
    ]
    point = math.exp(sum(all_logs) / len(all_logs))
    rng = random.Random(seed)
    sampled: list[float] = []
    for _draw in range(draws):
        total = 0.0
        count = 0
        for _ in block_ids:
            block = block_ids[rng.randrange(len(block_ids))]
            identities = _draw_dataset_stratified_example_sample(
                rng, example_ids[block]
            )
            for example in identities:
                values = by_block[block][example]
                total += sum(values)
                count += len(values)
        sampled.append(math.exp(total / count))
    sampled.sort()
    return point, _type7_quantile(sampled, 0.025), _type7_quantile(sampled, 0.975)


def _dataset_stratified_example_identities(
    by_example: Mapping[tuple[str, str], Sequence[float]],
) -> dict[str, tuple[tuple[str, str], ...]]:
    output = {
        dataset: tuple(sorted(key for key in by_example if key[0] == dataset))
        for dataset in SUPPORTED_V1_DATASETS
    }
    counts = {len(values) for values in output.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
        raise ValueError("hierarchical bootstrap requires balanced dataset strata")
    repeat_counts = {
        len(by_example[key]) for values in output.values() for key in values
    }
    if len(repeat_counts) != 1 or not repeat_counts or next(iter(repeat_counts)) <= 0:
        raise ValueError("hierarchical bootstrap requires complete paired repeats")
    if any(
        not math.isfinite(float(value))
        for values in by_example.values()
        for value in values
    ):
        raise ValueError("hierarchical bootstrap observations must be finite")
    return output


def _draw_dataset_stratified_example_sample(
    rng: random.Random,
    identities_by_dataset: Mapping[str, Sequence[tuple[str, str]]],
) -> tuple[tuple[str, str], ...]:
    sampled: list[tuple[str, str]] = []
    for dataset in SUPPORTED_V1_DATASETS:
        identities = identities_by_dataset[dataset]
        sampled.extend(identities[rng.randrange(len(identities))] for _ in identities)
    return tuple(sampled)


def _type7_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires observations")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def _validate_submit_payload(
    payload: Mapping[str, Any],
    *,
    job_record: Mapping[str, Any],
) -> None:
    if set(payload) != {
        "idempotency_token",
        "run_name",
        "tasks",
        "timeout_seconds",
    }:
        raise ValueError("publication latency submit payload schema is open")
    require_databricks_run_idempotency_token(
        payload,
        attempt_id=_required_string(job_record, "reservation_attempt_id"),
    )
    runtime = _mapping(job_record, "runtime")
    timeout_seconds = _required_int(runtime, "run_timeout_seconds")
    if payload.get("timeout_seconds") != timeout_seconds:
        raise ValueError("publication latency run timeout drift")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("publication latency requires exactly one isolated task")
    task = _mapping_value(tasks[0], "publication latency task")
    if (
        task.get("max_retries") != 0
        or task.get("timeout_seconds") != timeout_seconds
        or task.get("task_key") != job_record.get("task_key")
    ):
        raise ValueError("publication latency task retry/timeout identity drift")
    cluster = _mapping(task, "new_cluster")
    if (
        cluster.get("node_type_id") != runtime.get("node_type_id")
        or cluster.get("driver_node_type_id") != runtime.get("node_type_id")
        or cluster.get("num_workers") != 0
        or cluster.get("spark_version") != runtime.get("databricks_spark_version")
        or cluster.get("data_security_mode") != runtime.get("data_security_mode")
    ):
        raise ValueError("publication latency cluster hardware drift")
    aws_attributes = _mapping(cluster, "aws_attributes")
    if aws_attributes != {
        "availability": runtime.get("availability"),
        "zone_id": runtime.get("zone_id"),
    }:
        raise ValueError("publication latency cluster availability/zone drift")
    python_task = _mapping(task, "spark_python_task")
    parameters = python_task.get("parameters")
    if not isinstance(parameters, list) or any(
        not isinstance(item, str) for item in parameters
    ):
        raise ValueError("publication latency runner parameters are invalid")
    if _one_parameter(parameters, "--cloud-run-id") != (
        _DATABRICKS_JOB_RUN_ID_TEMPLATE
    ) or _one_parameter(parameters, "--task-run-id") != (
        _DATABRICKS_TASK_RUN_ID_TEMPLATE
    ):
        raise ValueError("publication latency cloud identity templates drift")
    if _one_parameter(parameters, "--job-record-json") != _canonical_json(job_record):
        raise ValueError("publication latency job record parameter drift")


def validate_publication_latency_execution_sources(
    execution_plan_record: Mapping[str, Any],
) -> None:
    """Re-read every controller source file and reject stale plan bindings."""

    validate_publication_latency_execution_plan_record(execution_plan_record)
    sources = _mapping(execution_plan_record, "sources")
    campaign_binding = _mapping(sources, "campaign")
    campaign = _read_bound_json_uri(campaign_binding, "campaign plan")
    validate_publication_campaign_plan_record(campaign)
    if (
        campaign.get("closed_record_sha256")
        != campaign_binding.get("closed_record_sha256")
        or campaign.get("campaign_ledger_id") != sources.get("campaign_ledger_id")
        or campaign.get("campaign_ledger_prefix")
        != sources.get("campaign_ledger_prefix")
    ):
        raise ValueError("campaign plan record binding drift")

    qualification = _mapping(sources, "qualification")
    qualification_plan = _read_bound_json_uri(
        _mapping(qualification, "plan"), "qualification plan"
    )
    qualification_evidence = _read_bound_json_uri(
        _mapping(qualification, "evidence"), "qualification evidence"
    )
    pins = GPUQualificationArtifactPins(
        **dict(_mapping(qualification, "artifact_pins"))
    )
    observed_selection = validate_gpu_qualification_evidence_record(
        qualification_evidence,
        plan_record=qualification_plan,
        expected_campaign_id=_required_string(execution_plan_record, "campaign_id"),
        expected_artifact_pins=pins,
    )
    if _selection_record(observed_selection) != dict(
        _mapping(qualification, "selection")
    ):
        raise ValueError("qualification selection binding drift")
    if (
        qualification_plan.get("campaign_record_sha256")
        != campaign.get("closed_record_sha256")
        or qualification_plan.get("campaign_ledger_id")
        != sources.get("campaign_ledger_id")
        or qualification_plan.get("campaign_ledger_prefix")
        != sources.get("campaign_ledger_prefix")
    ):
        raise ValueError("qualification/campaign ledger binding drift")

    main_schedule_bindings = _mapping_sequence(sources, "schedules")
    storage_schedule_bindings = _mapping_sequence(sources, "storage_schedules")
    schedule_bindings = [*main_schedule_bindings, *storage_schedule_bindings]
    final_artifacts = _final_artifacts_from_record(_mapping(sources, "final_artifacts"))
    storage_source_paths = {
        dataset: _cluster_path(final_artifacts.file(f"input_16384_{dataset}").uri)
        for dataset in SUPPORTED_V1_DATASETS
    }
    storage_source_examples = load_publication_storage_selection_examples(
        storage_source_paths
    )
    observed_storage_schedules: dict[int, Mapping[str, Any]] = {}
    for binding in schedule_bindings:
        schedule = _read_bound_json_uri(binding, "publication schedule")
        is_storage = binding in storage_schedule_bindings
        if is_storage:
            validate_publication_storage_block_schedule(
                schedule,
                source_examples=storage_source_examples,
                expected_input_bundle_sha256=pins.input_bundle_sha256,
            )
        else:
            validate_publication_latency_block_schedule(
                schedule,
                examples=_schedule_examples(schedule),
                expected_input_bundle_sha256=pins.input_bundle_sha256,
            )
        for field_name in (
            "closed_record_sha256",
            "deployment_block",
            "requests_sha256",
            "seed_sha256",
        ):
            if schedule.get(field_name) != binding.get(field_name):
                raise ValueError(f"publication schedule {field_name} binding drift")
        if is_storage:
            if _mapping(_mapping(schedule, "protocol"), "selection").get(
                "selection_sha256"
            ) != binding.get("selection_sha256"):
                raise ValueError("publication storage selection binding drift")
            observed_storage_schedules[_required_int(schedule, "deployment_block")] = (
                schedule
            )

    storage_inputs_binding = _mapping(sources, "storage_inputs")
    storage_inputs = _read_bound_json_uri(
        storage_inputs_binding, "publication storage inputs"
    )
    if storage_inputs.get("closed_record_sha256") != storage_inputs_binding.get(
        "closed_record_sha256"
    ) or _mapping(storage_inputs, "selection_protocol").get(
        "selection_sha256"
    ) != storage_inputs_binding.get("selection_sha256"):
        raise ValueError("publication storage input record binding drift")
    validate_publication_storage_inputs_record(
        storage_inputs,
        source_paths=storage_source_paths,
        schedule_records=observed_storage_schedules,
        expected_input_bundle_sha256=pins.input_bundle_sha256,
    )

    for artifact in final_artifacts.files:
        path = _cluster_path(artifact.uri)
        _verify_regular_file_sha256(path, artifact.sha256, f"artifact {artifact.role}")

    handoff = _mapping(sources, "handoff_generation")
    authenticated = read_publication_latency_handoff_generation_result(
        _cluster_path(_required_string(handoff, "output_root_uri"))
    )
    execution = _mapping(handoff, "execution")
    authenticated_handoff_reconciliation = _mapping(
        authenticated.record, "ledger_reconciliation"
    )
    handoff_authorization_binding = _mapping(handoff, "authorization")
    q8_causal_closure = _canonical_sha256(
        {
            "direct_control_plane_status_sha256": [
                _required_sha256(item, "control_plane_status_sha256")
                for item in _mapping_sequence(
                    authenticated_handoff_reconciliation,
                    "attempts",
                )
            ],
            "ledger_reconciliation": dict(authenticated_handoff_reconciliation),
        }
    )
    if (
        authenticated.record.get("closed_record_sha256")
        != execution.get("closed_record_sha256")
        or _file_sha256(authenticated.execution_record_path) != execution.get("sha256")
        or authenticated.execution_record_path.absolute()
        != _cluster_path(_required_string(execution, "uri")).absolute()
        or authenticated.record.get("execution_mode")
        != PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED
        or handoff_authorization_binding.get("causal_closure_sha256")
        != q8_causal_closure
        or handoff_authorization_binding.get("ledger_id")
        != authenticated_handoff_reconciliation.get("ledger_id")
    ):
        raise ValueError("distributed handoff source binding drift")
    if _mapping(authenticated.record, "generator_hardware").get(
        "qualification_closed_record_sha256"
    ) != qualification_evidence.get("closed_record_sha256"):
        raise ValueError("distributed handoff qualification binding drift")

    bf16 = _mapping(sources, "bf16_handoff")
    authenticated_bf16 = read_publication_bf16_handoff_generation_result(
        _cluster_path(_required_string(bf16, "output_root_uri"))
    )
    bf16_execution_binding = _mapping(bf16, "execution")
    bf16_accounting_binding = _mapping(bf16, "accounting")
    bf16_authorization_binding = _mapping(bf16, "authorization")
    authenticated_bf16_reconciliation = _mapping(
        authenticated_bf16.record, "ledger_reconciliation"
    )
    if (
        authenticated_bf16.record.get("closed_record_sha256")
        != bf16_execution_binding.get("closed_record_sha256")
        or authenticated_bf16.record.get("execution_mode")
        != bf16_execution_binding.get("execution_mode")
        or _file_sha256(authenticated_bf16.execution_record_path)
        != bf16_execution_binding.get("sha256")
        or authenticated_bf16.execution_record_path.absolute()
        != _cluster_path(_required_string(bf16_execution_binding, "uri")).absolute()
        or _canonical_sha256(authenticated_bf16_reconciliation)
        != bf16.get("ledger_reconciliation_sha256")
        or bf16_authorization_binding.get("causal_closure_sha256")
        != bf16.get("ledger_reconciliation_sha256")
        or bf16_authorization_binding.get("ledger_id")
        != authenticated_bf16_reconciliation.get("ledger_id")
        or _canonical_sha256(_mapping(authenticated_bf16.record, "accounting"))
        != bf16_accounting_binding.get("closed_sha256")
    ):
        raise ValueError("distributed BF16 generation source binding drift")
    bf16_manifest_binding = _mapping(bf16, "manifest")
    bf16_manifest = authenticated_bf16.manifest
    validate_publication_latency_handoff_bundle(
        bf16_manifest,
        bundle_root=_cluster_path(_required_string(bf16, "source_root_uri")),
    )
    if (
        bf16_manifest.get("closed_record_sha256")
        != bf16_manifest_binding.get("closed_record_sha256")
        or _file_sha256(authenticated_bf16.manifest_path)
        != bf16_manifest_binding.get("sha256")
        or authenticated_bf16.manifest_path.absolute()
        != _cluster_path(_required_string(bf16_manifest_binding, "uri")).absolute()
        or authenticated_bf16.source_root.absolute()
        != _cluster_path(_required_string(bf16, "source_root_uri")).absolute()
    ):
        raise ValueError("BF16 handoff source binding drift")


def publication_latency_vllm_config(
    job_record: Mapping[str, Any],
) -> VLLMSmokeBenchmarkConfig:
    """Materialize the exact one-arm vLLM configuration for a worker."""

    validate_publication_latency_job_record(job_record)
    cell = _mapping(job_record, "cell")
    runtime = _mapping(job_record, "runtime")
    request_order = _mapping(job_record, "request_order")
    runtime_kv_dtype = _required_string(runtime, "runtime_kv_dtype")
    layout = layout_for_model(
        GPU_QUALIFICATION_MODEL_ID,
        dtype=runtime_kv_dtype,
        key_position_encoding="stored_post_rope",
    )
    payload_axis_order = getattr(layout.payload_axis_order, "value", None)
    key_position_encoding = getattr(layout.key_position_encoding, "value", None)
    if not isinstance(payload_axis_order, str) or not isinstance(
        key_position_encoding, str
    ):
        raise RuntimeError("resolved Qwen layout has noncanonical enum values")
    identity = RuntimeIdentity(
        model_id=GPU_QUALIFICATION_MODEL_ID,
        model_revision=GPU_QUALIFICATION_MODEL_REVISION,
        tokenizer_id=MAIN_LATENCY_TOKENIZER_ID,
        tokenizer_revision=MAIN_LATENCY_TOKENIZER_REVISION,
        lora_id=layout.lora_id,
        prompt_template_version=DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
        layout_version=layout.layout_version,
        kv_dtype=runtime_kv_dtype,
        block_size=layout.block_size,
        payload_axis_order=payload_axis_order,
        key_position_encoding=key_position_encoding,
        rope_theta=layout.rope_theta,
        rope_rotary_dim=layout.rope_rotary_dim,
    )
    job_id = _required_string(job_record, "job_id")
    input_tokens = _required_int(cell, "input_tokens")
    method_id = _required_string(cell, "method_id")
    artifact_files = {
        _required_string(item, "role"): item
        for item in _mapping_sequence(job_record, "artifact_files")
    }
    provenance = {
        "cache_state": _required_string(
            _mapping(job_record, "cache_telemetry_policy"), "host_cache_state"
        ),
        "canonical_model_id": GPU_QUALIFICATION_MODEL_ID,
        "engine_id": "vllm",
        "engine_version": PUBLICATION_CAMPAIGN_ENGINE_VERSION,
        "hardware_fingerprint": _required_string(runtime, "node_type_id"),
        "input_tokens_target": input_tokens,
        "block_size": layout.block_size,
        "key_position_encoding": key_position_encoding,
        "layout_version": layout.layout_version,
        "lora_id": layout.lora_id,
        "measurement_scopes": ["latency", "resource"],
        "model_dtype": _required_string(runtime, "model_dtype"),
        "model_quantization": _required_string(runtime, "model_quantization"),
        "model_revision": GPU_QUALIFICATION_MODEL_REVISION,
        "package_revisions": {
            "cachet-kv": (
                "wheel-sha256:"
                + _required_sha256(artifact_files["package_wheel"], "sha256")
            ),
            "cachet-runner": (
                "sha256:" + _required_sha256(artifact_files["runner"], "sha256")
            ),
            "cachet-source": "git:" + _required_string(job_record, "source_revision"),
            "cachet-source-tree": (
                "sha256:" + _required_sha256(job_record, "source_tree_sha256")
            ),
        },
        "prompt_template_version": DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
        "payload_axis_order": payload_axis_order,
        "runtime_id": _canonical_sha256(
            {
                "execution_plan_sha256": _required_sha256(
                    job_record, "execution_plan_sha256"
                ),
                "job_id": job_id,
                "runtime": runtime,
            }
        ),
        "runtime_kv_dtype": runtime_kv_dtype,
        "runtime_version": VLLM_RUNTIME_LOCK_SHA256,
        "pipeline_parallel_size": 1,
        "serving_platform": "databricks",
        "storage_identity": _required_string(
            _mapping(job_record, "cache_telemetry_policy"), "storage_source"
        ),
        "tokenizer_id": MAIN_LATENCY_TOKENIZER_ID,
        "tokenizer_revision": MAIN_LATENCY_TOKENIZER_REVISION,
        "tensor_parallel_size": 1,
    }
    handoff = job_record.get("handoff")
    handoff_kwargs: dict[str, Any] = {}
    if method_id == "vanilla_prefill":
        handoff_record = _mapping_value(handoff, "handoff")
        stage_path = _cluster_path(_required_string(handoff_record, "stage_uri"))
        handoff_kwargs = {
            "publication_handoff_local_nvme_dir": stage_path,
            "publication_handoff_stage_kind": _required_string(
                handoff_record, "stage_kind"
            ),
        }
        if handoff_record.get("source_kind") == "distributed_q8_generation":
            execution = _mapping(handoff_record, "execution")
            handoff_kwargs.update(
                {
                    "publication_handoff_generation_output_root": _cluster_path(
                        _required_string(handoff_record, "output_root_uri")
                    ),
                    "publication_handoff_generation_execution_file_sha256": (
                        _required_sha256(execution, "sha256")
                    ),
                    "publication_handoff_generation_execution_closed_record_sha256": (
                        _required_sha256(execution, "closed_record_sha256")
                    ),
                }
            )
        elif handoff_record.get("source_kind") == "closed_bf16_bundle":
            manifest = _mapping(handoff_record, "manifest")
            handoff_kwargs.update(
                {
                    "publication_handoff_bundle_manifest_path": _cluster_path(
                        _required_string(manifest, "uri")
                    ),
                    "publication_handoff_bundle_source_root": _cluster_path(
                        _required_string(handoff_record, "source_root_uri")
                    ),
                    "publication_handoff_bundle_manifest_file_sha256": (
                        _required_sha256(manifest, "sha256")
                    ),
                    "publication_handoff_bundle_manifest_closed_record_sha256": (
                        _required_sha256(manifest, "closed_record_sha256")
                    ),
                }
            )
        else:
            raise ValueError("publication Vanilla handoff source_kind is invalid")

    input_specs = tuple(
        f"{_required_string(item, 'dataset')}={_cluster_path(_required_string(item, 'uri'))}"
        for item in _mapping_sequence(job_record, "input_files")
    )
    output_dir = _cluster_path(
        _required_string(_mapping(job_record, "output"), "directory_uri")
    )
    local_root = (
        Path("/local_disk0/cachet-publication-latency")
        / (_required_sha256(job_record, "execution_plan_sha256")[:16])
        / job_id
    )
    if method_id == "baseline_prefill":
        arm_kwargs: dict[str, Any] = {"benchmark_arms": (BASELINE_PREFILL_ARM,)}
    else:
        arm_kwargs = {"benchmark_arm_specs": (_vanilla_arm_spec(),)}
    return VLLMSmokeBenchmarkConfig(
        benchmark_id=job_id,
        benchmark_suite_id=(
            f"{_required_string(job_record, 'campaign_id')}-latency-{input_tokens}"
        ),
        output_dir=output_dir,
        local_root=local_root,
        model_id=GPU_QUALIFICATION_MODEL_ID,
        model_revision=GPU_QUALIFICATION_MODEL_REVISION,
        tokenizer_revision=MAIN_LATENCY_TOKENIZER_REVISION,
        model_dtype=_required_string(runtime, "model_dtype"),
        model_quantization=_required_string(runtime, "model_quantization"),
        kv_cache_dtype=runtime_kv_dtype,
        attention_backend=_required_string(runtime, "attention_backend"),
        max_tokens=_required_int(runtime, "max_output_tokens"),
        force_max_tokens=runtime.get("force_max_tokens") is True,
        temperature=_finite_nonnegative_number(
            runtime.get("temperature"), "temperature"
        ),
        generation_seed=_required_int(runtime, "generation_seed"),
        timeout_seconds=float(_required_int(runtime, "run_timeout_seconds")),
        server_start_timeout_seconds=1200.0,
        max_model_len=_required_int(runtime, "max_model_len"),
        max_num_seqs=_required_int(runtime, "max_num_seqs"),
        gpu_memory_utilization=_finite_positive_number(
            runtime.get("gpu_memory_utilization"), "gpu_memory_utilization"
        ),
        benchmark_repeats=_required_int(cell, "repeats_per_example"),
        request_parallelism=_required_int(runtime, "request_parallelism"),
        benchmark_interleave_examples=False,
        benchmark_evidence_policy="publication",
        benchmark_manifest_provenance=provenance,
        prefix_cache_salt_mode="per_request",
        prewarm_cache_prefix=False,
        cache_runtime_prompt=False,
        payload_cache_max_bytes=(
            PUBLICATION_LATENCY_RAM_PAYLOAD_CACHE_BYTES
            if cell.get("setting_id") == "storage-ram"
            else 0
        ),
        payload_cache_prime_target_count=(
            PUBLICATION_LATENCY_RAM_PRIME_TARGETS
            if cell.get("setting_id") == "storage-ram"
            else None
        ),
        prewarm_payload_cache=cell.get("setting_id") == "storage-ram",
        hardware_target=_required_string(runtime, "hardware_target"),
        dataset_specs=input_specs,
        allow_dataset_subset=False,
        runtime_identity=identity,
        publication_latency_schedule_path=_cluster_path(
            _required_string(request_order, "schedule_uri")
        ),
        publication_latency_expected_input_bundle_sha256=(
            _required_sha256(request_order, "input_bundle_sha256")
        ),
        **arm_kwargs,
        **handoff_kwargs,
    )


def execute_publication_latency_job_record(
    job_record: Mapping[str, Any],
    *,
    expected_job_sha256: str,
    cloud_run_id: str,
    task_run_id: str,
) -> dict[str, Any]:
    """Execute one immutable worker contract and seal its result last."""

    validate_publication_latency_job_record(job_record)
    if _required_sha256(job_record, "closed_record_sha256") != (
        _require_sha256_value(expected_job_sha256, "expected_job_sha256")
    ):
        raise ValueError("publication latency job argument digest drift")
    cloud_run_id = _databricks_id(cloud_run_id, "cloud_run_id")
    task_run_id = _databricks_id(task_run_id, "task_run_id")
    if cloud_run_id == task_run_id:
        raise ValueError("cloud and task run IDs must be distinct")
    _validate_job_bound_source_files(job_record)
    config = publication_latency_vllm_config(job_record)
    _reject_existing_symlink_ancestors(config.output_dir, "worker durable output")
    _reject_existing_symlink_ancestors(config.local_root, "worker local root")
    handoff_value = job_record.get("handoff")
    if isinstance(handoff_value, Mapping):
        _reject_existing_symlink_ancestors(
            _cluster_path(_required_string(handoff_value, "stage_uri")),
            "worker handoff stage",
        )
    if config.output_dir.exists() or config.output_dir.is_symlink():
        raise FileExistsError(
            f"publication latency output already exists: {config.output_dir}"
        )
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlink_ancestors(config.output_dir, "worker durable output")
    snapshot_schedule, snapshot_specs = _snapshot_publication_latency_inputs(
        job_record,
        snapshot_root=config.local_root / "source-snapshot",
    )
    config = replace(
        config,
        dataset_specs=snapshot_specs,
        publication_latency_schedule_path=snapshot_schedule,
    )
    artifacts = {
        _required_string(item, "role"): item
        for item in _mapping_sequence(job_record, "artifact_files")
    }
    os.environ[VLLM_PATCHED_WHEEL_URI_ENV] = _required_string(
        artifacts["patched_vllm_wheel"], "uri"
    )
    os.environ[VLLM_PATCHED_WHEEL_SHA256_ENV] = _required_sha256(
        artifacts["patched_vllm_wheel"], "sha256"
    )
    os.environ["DOCUMENT_KV_EVICT_PAGE_CACHE"] = (
        "0" if _mapping(job_record, "cell").get("setting_id") == "storage-ram" else "1"
    )
    run_vllm_smoke_benchmark(config)
    schedule_record = _read_json_file(
        snapshot_schedule, "snapshotted publication schedule"
    )
    _cleanup_publication_latency_worker_state(job_record, config=config)
    result = seal_publication_latency_job_result(
        job_record,
        cloud_run_id=cloud_run_id,
        task_run_id=task_run_id,
        schedule_record=schedule_record,
    )
    result_path = _cluster_path(
        _required_string(_mapping(job_record, "output"), "result_uri")
    )
    _write_canonical_json_exclusive(result_path, result)
    return result


def _vanilla_arm_spec() -> dict[str, Any]:
    arm = method_benchmark_arm(
        "vanilla_prefill",
        arm_id=PUBLICATION_LATENCY_VANILLA_ARM_ID,
        physical_transform_id="cachet.vanilla.per_document_segments",
    )
    return {
        "arm_id": arm.arm_id,
        "cache_method": arm.cache_method,
        "connector_mode": arm.connector_mode,
        "description": arm.description,
        "implementation_kind": arm.implementation_kind,
        "method_config_digest": arm.method_config_digest,
        "method_version": arm.method_version,
        "physical_transform_id": arm.physical_transform_id,
        "physical_transform_version": arm.physical_transform_version,
        "requires_cachet_handoff": arm.requires_cachet_handoff,
        "uses_cache": arm.uses_cache,
        "variant_id": arm.variant_id,
    }


def seal_publication_latency_job_result(
    job_record: Mapping[str, Any],
    *,
    cloud_run_id: str,
    task_run_id: str,
    schedule_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate all worker artifacts and return the last-write result seal."""

    validate_publication_latency_job_record(job_record)
    cloud_run_id = _databricks_id(cloud_run_id, "cloud_run_id")
    task_run_id = _databricks_id(task_run_id, "task_run_id")
    config = publication_latency_vllm_config(job_record)
    schedule = (
        _read_json_file(
            _cluster_path(
                _required_string(_mapping(job_record, "request_order"), "schedule_uri")
            ),
            "publication latency schedule",
        )
        if schedule_record is None
        else json.loads(_canonical_json(schedule_record))
    )
    if not isinstance(schedule, dict):
        raise TypeError("schedule_record must be a JSON object")
    _validate_job_schedule_record(schedule, job_record=job_record)
    benchmark = _read_json_file(config.benchmark_output_path, "benchmark result")
    _validate_publication_latency_benchmark(
        benchmark,
        job_record=job_record,
        schedule=schedule,
    )
    metadata = _read_json_file(config.metadata_path, "vLLM metadata")
    _validate_publication_latency_runtime_metadata(metadata, job_record=job_record)
    cache_summary = _publication_latency_cache_telemetry_summary(
        config.connector_telemetry_copy_path,
        job_record=job_record,
        benchmark_record=benchmark,
    )
    if _mapping(job_record, "cell").get("setting_id") == "storage-ram":
        cache_summary.update(
            _ram_payload_cache_artifact_summary(
                config.prewarm_payload_cache_path,
                config.payload_cache_attestation_path,
                expected_benchmark_id=_required_string(job_record, "job_id"),
            )
        )
    artifact_paths: list[tuple[str, Path]] = [
        ("benchmark", config.benchmark_output_path),
        ("metadata", config.metadata_path),
        ("runtime_telemetry", config.runtime_telemetry_copy_path),
        ("prompt_token_budget", config.prompt_token_budget_path),
        ("import_probe", config.import_probe_path),
    ]
    if _mapping(job_record, "cell").get("method_id") == "vanilla_prefill":
        artifact_paths.extend(
            (
                ("connector_telemetry", config.connector_telemetry_copy_path),
                (
                    "handoff_staging_attestation",
                    config.publication_handoff_staging_attestation_copy_path,
                ),
            )
        )
    if _mapping(job_record, "cell").get("setting_id") == "storage-ram":
        artifact_paths.extend(
            (
                ("payload_cache_prime", config.prewarm_payload_cache_path),
                ("payload_cache_attestation", config.payload_cache_attestation_path),
            )
        )
    files = [
        {
            "role": role,
            "sha256": _verified_output_file_sha256(path, role),
            "uri": _job_output_uri(job_record, path.name),
        }
        for role, path in artifact_paths
    ]
    result: dict[str, Any] = {
        "benchmark_record": benchmark,
        "cache_telemetry": cache_summary,
        "campaign_id": _required_string(job_record, "campaign_id"),
        "closed_record_sha256": "",
        "completed_at": datetime.now(UTC).isoformat(),
        "execution_plan_sha256": _required_sha256(job_record, "execution_plan_sha256"),
        "files": files,
        "job_id": _required_string(job_record, "job_id"),
        "job_record_sha256": _required_sha256(job_record, "closed_record_sha256"),
        "record_type": PUBLICATION_LATENCY_JOB_RESULT_RECORD_TYPE,
        "runtime_attestation": {
            "metadata_sha256": next(
                item["sha256"] for item in files if item["role"] == "metadata"
            ),
            "patched_vllm_wheel_sha256": _required_sha256(
                next(
                    item
                    for item in _mapping_sequence(job_record, "artifact_files")
                    if item.get("role") == "patched_vllm_wheel"
                ),
                "sha256",
            ),
            "runtime_lock_sha256": _required_sha256(
                next(
                    item
                    for item in _mapping_sequence(job_record, "artifact_files")
                    if item.get("role") == "runtime_lock"
                ),
                "sha256",
            ),
            "strict_runtime_closure": True,
            "vllm_version": PUBLICATION_CAMPAIGN_ENGINE_VERSION,
        },
        "schema_version": PUBLICATION_LATENCY_SCHEMA_VERSION,
        "schedule_record": schedule,
        "source_inputs_sha256": _canonical_sha256(
            {
                "input_files": job_record.get("input_files"),
                "request_order": job_record.get("request_order"),
                "handoff": job_record.get("handoff"),
            }
        ),
        "task_identity": {
            "cloud_run_id": cloud_run_id,
            "task_key": _required_string(job_record, "task_key"),
            "task_run_id": task_run_id,
        },
    }
    result["closed_record_sha256"] = _closed_record_sha256(result)
    validate_publication_latency_job_result_record(
        result,
        expected_job_record=job_record,
        verify_files=True,
    )
    return result


def validate_publication_latency_job_result_record(
    record: Mapping[str, Any],
    *,
    expected_job_record: Mapping[str, Any],
    verify_files: bool = False,
) -> None:
    """Validate one sealed result; optional file checks are mandatory at collection."""

    validate_publication_latency_job_record(expected_job_record)
    _require_exact_keys(
        record,
        {
            "benchmark_record",
            "cache_telemetry",
            "campaign_id",
            "closed_record_sha256",
            "completed_at",
            "execution_plan_sha256",
            "files",
            "job_id",
            "job_record_sha256",
            "record_type",
            "runtime_attestation",
            "schedule_record",
            "schema_version",
            "source_inputs_sha256",
            "task_identity",
        },
        "publication latency job result",
    )
    if record.get("record_type") != PUBLICATION_LATENCY_JOB_RESULT_RECORD_TYPE:
        raise ValueError("publication latency job result record_type is invalid")
    if record.get("schema_version") != PUBLICATION_LATENCY_SCHEMA_VERSION:
        raise ValueError("publication latency job result schema_version is invalid")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("publication latency job result closure is invalid")
    for field_name in ("campaign_id", "execution_plan_sha256", "job_id"):
        if record.get(field_name) != expected_job_record.get(field_name):
            raise ValueError(f"publication latency result {field_name} drift")
    if record.get("job_record_sha256") != expected_job_record.get(
        "closed_record_sha256"
    ):
        raise ValueError("publication latency result job binding drift")
    if record.get("source_inputs_sha256") != _canonical_sha256(
        {
            "input_files": expected_job_record.get("input_files"),
            "request_order": expected_job_record.get("request_order"),
            "handoff": expected_job_record.get("handoff"),
        }
    ):
        raise ValueError("publication latency result source binding drift")
    task_identity = _mapping(record, "task_identity")
    _databricks_id(task_identity.get("cloud_run_id"), "cloud_run_id")
    _databricks_id(task_identity.get("task_run_id"), "task_run_id")
    if task_identity.get("task_key") != expected_job_record.get("task_key"):
        raise ValueError("publication latency result task identity drift")
    benchmark = _mapping(record, "benchmark_record")
    schedule = _mapping(record, "schedule_record")
    _validate_job_schedule_record(schedule, job_record=expected_job_record)
    _validate_publication_latency_benchmark(
        benchmark,
        job_record=expected_job_record,
        schedule=schedule,
    )
    files = _mapping_sequence(record, "files")
    expected_roles = [
        "benchmark",
        "metadata",
        "runtime_telemetry",
        "prompt_token_budget",
        "import_probe",
    ]
    if _mapping(expected_job_record, "cell").get("method_id") == "vanilla_prefill":
        expected_roles.extend(("connector_telemetry", "handoff_staging_attestation"))
    if _mapping(expected_job_record, "cell").get("setting_id") == "storage-ram":
        expected_roles.extend(("payload_cache_prime", "payload_cache_attestation"))
    if [item.get("role") for item in files] != expected_roles:
        raise ValueError("publication latency result file closure is incomplete")
    if len({item.get("uri") for item in files}) != len(files):
        raise ValueError("publication latency result file URIs are not unique")
    for item in files:
        uri = _durable_uri(_required_string(item, "uri"), "result file URI")
        digest = _required_sha256(item, "sha256")
        if verify_files:
            _verify_regular_file_sha256(
                _cluster_path(uri), digest, f"result file {item.get('role')}"
            )
    runtime = _mapping(record, "runtime_attestation")
    if runtime != {
        "metadata_sha256": next(
            item["sha256"] for item in files if item["role"] == "metadata"
        ),
        "patched_vllm_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        "runtime_lock_sha256": VLLM_RUNTIME_LOCK_SHA256,
        "strict_runtime_closure": True,
        "vllm_version": PUBLICATION_CAMPAIGN_ENGINE_VERSION,
    }:
        raise ValueError("publication latency result runtime attestation drift")
    cache = _mapping(record, "cache_telemetry")
    _validate_cache_telemetry_summary(cache, job_record=expected_job_record)
    if _mapping(expected_job_record, "cell").get("setting_id") == "storage-ram":
        for field_name in (
            "payload_cache_attestation_sha256",
            "payload_cache_prime_sha256",
        ):
            _required_sha256(cache, field_name)
        role_by_name = {item.get("role"): item for item in files}
        if cache.get("payload_cache_prime_sha256") != role_by_name[
            "payload_cache_prime"
        ].get("sha256") or cache.get(
            "payload_cache_attestation_sha256"
        ) != role_by_name["payload_cache_attestation"].get("sha256"):
            raise ValueError("RAM payload-cache artifact digest binding drift")


def _validate_publication_latency_benchmark(
    record: Mapping[str, Any],
    *,
    job_record: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> None:
    issues = benchmark_record_aggregate_issues(record)
    if issues:
        raise ValueError(
            "benchmark aggregate authentication failed: " + "; ".join(issues)
        )
    result = benchmark_run_result_from_record(record, evidence_policy="publication")
    cell = _mapping(job_record, "cell")
    method_id = _required_string(cell, "method_id")
    expected_arm = (
        BASELINE_PREFILL_ARM
        if method_id == "baseline_prefill"
        else PUBLICATION_LATENCY_VANILLA_ARM_ID
    )
    if (
        len(result.arms) != 1
        or result.arms[0].arm_id != expected_arm
        or result.request_parallelism != cell.get("request_parallelism")
        or result.repeats != cell.get("repeats_per_example")
        or result.shuffle
        or result.interleave_examples
        or result.prefix_cache_salt_mode != "per_request"
    ):
        raise ValueError("benchmark one-arm execution design drift")
    if len(result.measurements) != cell.get("request_count"):
        raise ValueError("benchmark measurement closure is incomplete")
    manifest = result.experiment_manifest
    if (
        manifest is None
        or manifest.temperature != PUBLICATION_LATENCY_TEMPERATURE
        or manifest.generation_seed != PUBLICATION_LATENCY_GENERATION_SEED
        or manifest.output_tokens_target != PUBLICATION_LATENCY_MAX_OUTPUT_TOKENS
        or manifest.decode_settings.get("ignore_eos") is not True
    ):
        raise ValueError("benchmark fixed decoding contract drift")
    if any(
        not measurement.ok
        or measurement.arm_id != expected_arm
        or measurement.completion_tokens != PUBLICATION_LATENCY_MAX_OUTPUT_TOKENS
        or measurement.ttft_seconds <= 0
        or measurement.time_to_completion_seconds < measurement.ttft_seconds
        for measurement in result.measurements
    ):
        raise ValueError(
            "benchmark contains failed or invalid fixed-decode measurements"
        )
    gate = _mapping(record, "evidence_gate")
    if gate.get("ok") is not True or gate.get("policy") != "publication":
        raise ValueError("benchmark publication evidence gate did not pass")
    if len(result.resource_evidence) != 1:
        raise ValueError("benchmark must contain one physical-arm resource record")
    resource = result.resource_evidence[0]
    if (
        resource.arm_id != expected_arm
        or not resource.complete
        or resource.error_count != 0
        or resource.source_revision != job_record.get("source_revision")
        or resource.source_tree_sha256 != job_record.get("source_tree_sha256")
    ):
        raise ValueError("benchmark resource evidence closure failed")
    artifact_files = {
        item.get("role"): item
        for item in _mapping_sequence(job_record, "artifact_files")
    }
    if resource.wheel_sha256 != artifact_files["package_wheel"].get(
        "sha256"
    ) or resource.runner_sha256 != artifact_files["runner"].get("sha256"):
        raise ValueError("benchmark resource software hashes drift")

    request_order = _mapping(job_record, "request_order")
    projection = project_publication_latency_request_order(
        schedule,
        examples=_schedule_examples(schedule),
        expected_input_bundle_sha256=_required_sha256(
            request_order, "input_bundle_sha256"
        ),
    )
    raw_requests = _mapping_sequence(schedule, "requests")
    lanes = schedule.get("lanes")
    if not isinstance(lanes, Mapping):
        raise ValueError("publication schedule lanes are missing")
    raw_lanes = lanes.get(str(_required_int(cell, "request_parallelism")))
    if not isinstance(raw_lanes, Sequence):
        raise ValueError("publication schedule concurrency lane is missing")
    lane_positions: dict[int, tuple[int, int]] = {}
    for lane, raw_lane in enumerate(raw_lanes):
        if not isinstance(raw_lane, Sequence):
            raise ValueError("publication schedule lane is invalid")
        for position, request_index in enumerate(raw_lane):
            if type(request_index) is not int or request_index in lane_positions:
                raise ValueError("publication schedule lane membership is invalid")
            lane_positions[request_index] = (lane, position)
    by_index: dict[int, Any] = {}
    for measurement in result.measurements:
        raw_index = measurement.metadata.get(
            PUBLICATION_LATENCY_REQUEST_INDEX_METADATA_KEY
        )
        try:
            request_index = int(raw_index or "")
        except ValueError as exc:
            raise ValueError("benchmark measurement request index is invalid") from exc
        if request_index in by_index:
            raise ValueError("benchmark request indices are not unique")
        by_index[request_index] = measurement
    if set(by_index) != set(range(len(projection))):
        raise ValueError("benchmark request index coverage is incomplete")
    for request_index, logical_key in enumerate(projection):
        measurement = by_index[request_index]
        raw_request = raw_requests[request_index]
        lane, lane_position = lane_positions[request_index]
        expected_metadata = {
            PUBLICATION_LATENCY_DEPLOYMENT_BLOCK_METADATA_KEY: str(
                _required_int(cell, "deployment_block")
            ),
            PUBLICATION_LATENCY_INPUT_BUNDLE_SHA256_METADATA_KEY: _required_sha256(
                request_order, "input_bundle_sha256"
            ),
            PUBLICATION_LATENCY_LANE_METADATA_KEY: str(lane),
            PUBLICATION_LATENCY_LANE_POSITION_METADATA_KEY: str(lane_position),
            PUBLICATION_LATENCY_REQUEST_ID_METADATA_KEY: _required_string(
                raw_request, "request_id"
            ),
            PUBLICATION_LATENCY_REQUEST_INDEX_METADATA_KEY: str(request_index),
            PUBLICATION_LATENCY_REQUESTS_SHA256_METADATA_KEY: _required_sha256(
                request_order, "requests_sha256"
            ),
            PUBLICATION_LATENCY_SCHEDULE_SHA256_METADATA_KEY: _required_sha256(
                request_order, "closed_record_sha256"
            ),
            PUBLICATION_LATENCY_SEED_SHA256_METADATA_KEY: _required_sha256(
                request_order, "seed_sha256"
            ),
        }
        if any(
            measurement.metadata.get(key) != value
            for key, value in expected_metadata.items()
        ):
            raise ValueError("benchmark scheduled request metadata drift")
        if (
            measurement.dataset,
            measurement.example_id,
            measurement.repeat_index,
        ) != logical_key:
            raise ValueError("benchmark logical request identity drift")


def _validate_publication_latency_runtime_metadata(
    metadata: Mapping[str, Any],
    *,
    job_record: Mapping[str, Any],
) -> None:
    runtime = _mapping(job_record, "runtime")
    expected = {
        "attention_backend": runtime.get("attention_backend"),
        "gpu_memory_utilization": runtime.get("gpu_memory_utilization"),
        "kv_cache_dtype": runtime.get("runtime_kv_dtype"),
        "max_model_len": runtime.get("max_model_len"),
        "max_num_seqs": runtime.get("max_num_seqs"),
        "model_dtype": runtime.get("model_dtype"),
        "model_quantization": runtime.get("model_quantization"),
        "model_revision": runtime.get("model_revision"),
        "payload_cache_max_bytes": (
            PUBLICATION_LATENCY_RAM_PAYLOAD_CACHE_BYTES
            if _mapping(job_record, "cell").get("setting_id") == "storage-ram"
            else 0
        ),
        "payload_cache_prime_target_count": (
            PUBLICATION_LATENCY_RAM_PRIME_TARGETS
            if _mapping(job_record, "cell").get("setting_id") == "storage-ram"
            else None
        ),
        "prewarm_payload_cache": (
            _mapping(job_record, "cell").get("setting_id") == "storage-ram"
        ),
        "temperature": runtime.get("temperature"),
        "generation_seed": runtime.get("generation_seed"),
        "vllm_patched_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        "vllm_version_requested": PUBLICATION_CAMPAIGN_ENGINE_VERSION,
    }
    for field_name, value in expected.items():
        if metadata.get(field_name) != value:
            raise ValueError(f"vLLM metadata {field_name} drift")
    if metadata.get("strict_runtime_closure") is not True:
        raise ValueError("vLLM metadata does not attest strict runtime closure")
    lock = metadata.get("vllm_runtime_lock_verification")
    if not isinstance(lock, Mapping) or lock.get("ok") is not True:
        raise ValueError("vLLM runtime lock verification did not pass")
    patches = metadata.get("vllm_runtime_patch_closure")
    if (
        not isinstance(patches, list)
        or not patches
        or any(
            not isinstance(item, Mapping)
            or item.get("verified") is not True
            or item.get("wheel_sha256") != GPU_QUALIFICATION_PATCHED_WHEEL_SHA256
            for item in patches
        )
    ):
        raise ValueError("vLLM installed patch closure did not pass")


def _publication_latency_cache_telemetry_summary(
    path: Path,
    *,
    job_record: Mapping[str, Any],
    benchmark_record: Mapping[str, Any],
) -> dict[str, Any]:
    method_id = _required_string(_mapping(job_record, "cell"), "method_id")
    if method_id == "baseline_prefill":
        records = _read_jsonl_file(path, required=False)
        loads = [item for item in records if item.get("event") == "load_request"]
        if loads:
            raise ValueError("Baseline benchmark unexpectedly emitted connector loads")
        return {
            "backend_bytes_read": 0,
            "benchmark_request_ids_sha256": _canonical_sha256([]),
            "cold_read_attested_count": 0,
            "eviction_requested_count": 0,
            "eviction_succeeded_count": 0,
            "load_count": 0,
            "payload_cache_hit_count": 0,
            "payload_cache_miss_count": 0,
            "storage_materialization_count": 0,
            "mounted_path_load_count": 0,
            "telemetry_file_sha256": None,
        }
    records = _read_jsonl_file(path, required=True)
    loads = [
        item
        for item in records
        if item.get("record_type") == "document_kv.vllm_native_provider_load.v1"
        and item.get("event") == "load_request"
        and item.get("success") is True
    ]
    benchmark = benchmark_run_result_from_record(
        benchmark_record, evidence_policy="publication"
    )
    expected_ids = {
        measurement.metadata[PUBLICATION_LATENCY_REQUEST_ID_METADATA_KEY]
        for measurement in benchmark.measurements
    }
    observed_ids = [item.get("benchmark_request_id") for item in loads]
    if (
        len(loads) != PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL
        or len(set(observed_ids)) != len(observed_ids)
        or set(observed_ids) != expected_ids
    ):
        raise ValueError("Vanilla connector telemetry request coverage is not exact")
    cold_count = 0
    eviction_requested = 0
    eviction_succeeded = 0
    payload_hits = 0
    payload_misses = 0
    backend_bytes_read = 0
    expected_backend_bytes = 0
    mounted_path_loads = 0
    runtime_dtype = _required_string(
        _mapping(job_record, "runtime"), "runtime_kv_dtype"
    )
    policy = _mapping(job_record, "cache_telemetry_policy")
    ram_payload_cache = policy.get("host_cache_state") == "prewarmed_payload_cache"
    for load in loads:
        counts = _mapping(load, "counts")
        attestation = _mapping(load, "cache_state_attestation")
        payload = _mapping(load, "payload")
        layout = _mapping(load, "layout")
        if (
            layout.get("dtype") != runtime_dtype
            or counts.get("decoded_runtime_payload_bytes")
            != counts.get("expected_runtime_payload_bytes")
            or counts.get("expected_stored_payload_bytes")
            != attestation.get("expected_stored_bytes")
            or counts.get("expected_runtime_payload_bytes")
            != attestation.get("expected_runtime_bytes")
            or counts.get("token_count") != attestation.get("expected_tokens")
            or attestation.get("loaded_tokens") != attestation.get("expected_tokens")
            or attestation.get("successful_loads") != 1
        ):
            raise ValueError("Vanilla connector byte/token/layout telemetry drift")
        expected_stored_bytes = _nonnegative_int(
            attestation.get("expected_stored_bytes"), "expected_stored_bytes"
        )
        bytes_read = _nonnegative_int(attestation.get("bytes_read"), "bytes_read")
        hits = _nonnegative_int(counts.get("payload_cache_hits"), "payload_cache_hits")
        misses = _nonnegative_int(
            counts.get("payload_cache_misses"), "payload_cache_misses"
        )
        if ram_payload_cache:
            if (
                payload.get("payload_cache_enabled") is not True
                or attestation.get("payload_cache_hit") is not True
                or bytes_read != 0
                or hits != 1
                or misses != 0
            ):
                raise ValueError("RAM payload-cache load is not one exact hit")
        elif (
            payload.get("payload_cache_enabled") is not False
            or attestation.get("payload_cache_hit") is not False
            or bytes_read != expected_stored_bytes
            or hits != 0
        ):
            raise ValueError("storage-backed Vanilla load byte proof drift")
        expected_backend_bytes += 0 if ram_payload_cache else expected_stored_bytes
        backend_bytes_read += bytes_read
        cold_count += int(attestation.get("cold_read_attested") is True)
        eviction_requested += int(attestation.get("eviction_requested") is True)
        eviction_succeeded += int(attestation.get("eviction_succeeded") is True)
        payload_hits += int(attestation.get("payload_cache_hit") is True)
        payload_misses += misses
        mounted_path_loads += int(
            attestation.get("source") == "local_path"
            and payload.get("source") == "uri"
            and payload.get("uri_scheme") == "local_path"
        )
    summary = {
        "benchmark_request_ids_sha256": _canonical_sha256(sorted(expected_ids)),
        "backend_bytes_read": backend_bytes_read,
        "expected_backend_bytes_read": expected_backend_bytes,
        "cold_read_attested_count": cold_count,
        "eviction_requested_count": eviction_requested,
        "eviction_succeeded_count": eviction_succeeded,
        "load_count": len(loads),
        "payload_cache_hit_count": payload_hits,
        "payload_cache_miss_count": payload_misses,
        "storage_materialization_count": payload_misses,
        "mounted_path_load_count": mounted_path_loads,
        "telemetry_file_sha256": _file_sha256(path),
    }
    _validate_cache_telemetry_summary(summary, job_record=job_record)
    return summary


def _ram_payload_cache_artifact_summary(
    prime_path: Path,
    attestation_path: Path,
    *,
    expected_benchmark_id: str,
) -> dict[str, Any]:
    prime = _read_json_file(prime_path, "RAM payload-cache prime record")
    attestation = _read_json_file(
        attestation_path, "RAM payload-cache measurement attestation"
    )
    isolation = _mapping(prime, "prefix_cache_isolation")
    if (
        prime.get("record_type") != "document_kv.vllm_payload_cache_prime.v1"
        or prime.get("ok") is not True
        or prime.get("benchmark_id") != expected_benchmark_id
        or prime.get("payload_cache_max_bytes")
        != PUBLICATION_LATENCY_RAM_PAYLOAD_CACHE_BYTES
        or prime.get("target_count") != PUBLICATION_LATENCY_RAM_PRIME_TARGETS
        or prime.get("request_count") != 2 * PUBLICATION_LATENCY_RAM_PRIME_TARGETS
        or prime.get("verification_all_hits") is not True
        or isolation.get("measurement_prefix_cache_salt_mode") != "per_request"
        or isolation.get("measurement_prefix_prewarmed") is not False
        or isolation.get("priming_requests_load_isolated_gpu_blocks") is not True
    ):
        raise ValueError("RAM payload-cache prime proof drift")
    payload_cache = _mapping(attestation, "payload_cache")
    gpu_prefix = _mapping(attestation, "gpu_prefix_cache")
    if (
        attestation.get("record_type")
        != "document_kv.vllm_ram_payload_cache_attestation.v1"
        or attestation.get("ok") is not True
        or attestation.get("benchmark_id") != expected_benchmark_id
        or attestation.get("measurement_protocol") != "ram_payload_cache_to_gpu_hydrate"
        or payload_cache.get("enabled") is not True
        or payload_cache.get("max_bytes") != PUBLICATION_LATENCY_RAM_PAYLOAD_CACHE_BYTES
        or payload_cache.get("priming_target_count")
        != PUBLICATION_LATENCY_RAM_PRIME_TARGETS
        or payload_cache.get("priming_verification_all_hits") is not True
        or payload_cache.get("measurement_request_count")
        != PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL
        or payload_cache.get("measurement_load_count")
        != PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL
        or payload_cache.get("measurement_all_hits") is not True
        or payload_cache.get("measurement_cache_hits")
        != PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL
        or payload_cache.get("measurement_cache_misses") != 0
        or payload_cache.get("measurement_storage_bytes_read") != 0
        or payload_cache.get("measurement_storage_materializations") != 0
        or gpu_prefix.get("prewarm_cache_prefix_enabled") is not False
        or gpu_prefix.get("measurement_cache_salt_mode") != "per_request"
        or gpu_prefix.get("measurement_prefix_prewarmed") is not False
        or gpu_prefix.get("priming_and_measurement_request_ids_disjoint") is not True
        or gpu_prefix.get("priming_and_measurement_cache_salts_disjoint") is not True
        or gpu_prefix.get("reuse_prevented_by_salt_namespace") is not True
    ):
        raise ValueError("RAM payload-cache measurement proof drift")
    return {
        "payload_cache_attestation_sha256": _file_sha256(attestation_path),
        "payload_cache_prime_sha256": _file_sha256(prime_path),
    }


def _validate_cache_telemetry_summary(
    summary: Mapping[str, Any],
    *,
    job_record: Mapping[str, Any],
) -> None:
    method_id = _required_string(_mapping(job_record, "cell"), "method_id")
    if method_id == "baseline_prefill":
        if (
            summary.get("load_count") != 0
            or summary.get("telemetry_file_sha256") is not None
        ):
            raise ValueError("Baseline cache telemetry summary drift")
        return
    for field_name in (
        "benchmark_request_ids_sha256",
        "telemetry_file_sha256",
    ):
        _required_sha256(summary, field_name)
    if summary.get("load_count") != PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL:
        raise ValueError("Vanilla cache telemetry load count drift")
    policy = _mapping(job_record, "cache_telemetry_policy")
    host_state = policy.get("host_cache_state")
    if host_state == "prewarmed_payload_cache":
        if (
            summary.get("payload_cache_hit_count")
            != PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL
            or summary.get("payload_cache_miss_count") != 0
            or summary.get("backend_bytes_read") != 0
            or summary.get("storage_materialization_count") != 0
            or summary.get("eviction_requested_count") != 0
        ):
            raise ValueError("RAM setting lacks exact prewarmed payload-cache hits")
    elif host_state == "cold_eviction_required":
        if (
            summary.get("eviction_requested_count")
            != PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL
            or summary.get("eviction_succeeded_count")
            != PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL
            or summary.get("cold_read_attested_count")
            != PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL
            or summary.get("payload_cache_hit_count") != 0
            or summary.get("backend_bytes_read")
            != summary.get("expected_backend_bytes_read")
        ):
            raise ValueError("cold Vanilla setting lacks per-request cold-read proof")
    elif host_state == "mounted_path_evicted_backend_cache_unproven":
        if (
            summary.get("eviction_requested_count")
            != PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL
            or summary.get("eviction_succeeded_count")
            != PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL
            or summary.get("mounted_path_load_count")
            != PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL
            or summary.get("payload_cache_hit_count") != 0
            or summary.get("backend_bytes_read")
            != summary.get("expected_backend_bytes_read")
        ):
            raise ValueError("UC setting lacks exact mounted-path byte/eviction proof")
    else:
        raise ValueError("Vanilla cache telemetry policy is unknown")


def _require_latency_launch_authorization(
    execution_plan_record: Mapping[str, Any],
    qualification_authorization: GPUQualificationLaunchAuthorization,
    handoff_authorization: PublicationLatencyHandoffServingAuthorization,
    bf16_handoff_authorization: PublicationBF16HandoffServingAuthorization,
) -> GPUQualificationSelection:
    validate_publication_latency_execution_plan_record(execution_plan_record)
    sources = _mapping(execution_plan_record, "sources")
    campaign_ledger_id = _required_string(sources, "campaign_ledger_id")
    campaign_ledger_path_sha256 = _required_sha256(
        sources, "campaign_ledger_path_sha256"
    )
    if (
        qualification_authorization.ledger_id != campaign_ledger_id
        or handoff_authorization.ledger_id != campaign_ledger_id
        or bf16_handoff_authorization.ledger_id != campaign_ledger_id
    ):
        raise ValueError(
            "GPU qualification, Q8 handoff, and BF16 handoff authority must "
            "share the execution plan campaign ledger"
        )
    if {
        qualification_authorization.ledger_path_sha256,
        handoff_authorization.ledger_path_sha256,
        bf16_handoff_authorization.ledger_path_sha256,
    } != {campaign_ledger_path_sha256}:
        raise ValueError("publication launch authority ledger path binding drift")
    qualification = _mapping(sources, "qualification")
    selection = require_gpu_qualification_launch_authorization(
        qualification_authorization,
        expected_plan_sha256=_required_sha256(
            _mapping(qualification, "plan"), "closed_record_sha256"
        ),
        expected_evidence_file_sha256=_required_sha256(
            _mapping(qualification, "evidence"), "sha256"
        ),
    )
    binding = _mapping(qualification, "authorization")
    if (
        qualification_authorization.causal_closure_sha256
        != binding.get("causal_closure_sha256")
        or qualification_authorization.ledger_id != binding.get("ledger_id")
        or qualification_authorization.ledger_path_sha256
        != binding.get("ledger_path_sha256")
        or qualification_authorization.ledger_prefix.to_record()
        != binding.get("ledger_prefix")
        or _selection_record(selection) != dict(_mapping(qualification, "selection"))
    ):
        raise ValueError("GPU qualification launch authorization binding drift")

    final_artifacts = _final_artifacts_from_record(_mapping(sources, "final_artifacts"))
    artifact_pins = _mapping(qualification, "artifact_pins")
    qualification_evidence = _mapping(qualification, "evidence")
    handoff_binding = _mapping(sources, "handoff_generation")
    authenticated_handoff = require_publication_latency_handoff_serving_authorization(
        handoff_authorization,
        expected_execution_file_sha256=final_artifacts.file("handoff_execution").sha256,
        expected_input_bundle_sha256=_required_sha256(
            artifact_pins, "input_bundle_sha256"
        ),
        expected_qualification_closed_record_sha256=_required_sha256(
            qualification_evidence, "closed_record_sha256"
        ),
    )
    handoff_authorization_binding = _mapping(handoff_binding, "authorization")
    handoff_execution_binding = _mapping(handoff_binding, "execution")
    if (
        handoff_authorization.causal_closure_sha256
        != handoff_authorization_binding.get("causal_closure_sha256")
        or handoff_authorization.ledger_id
        != handoff_authorization_binding.get("ledger_id")
        or handoff_authorization.ledger_path_sha256
        != handoff_authorization_binding.get("ledger_path_sha256")
        or handoff_authorization.ledger_prefix.to_record()
        != handoff_authorization_binding.get("ledger_prefix")
        or handoff_authorization.predecessor_prefix.to_record()
        != handoff_authorization_binding.get("predecessor_prefix")
        or handoff_authorization.producer_batch_prefix.to_record()
        != handoff_authorization_binding.get("producer_batch_prefix")
        or authenticated_handoff.record.get("closed_record_sha256")
        != handoff_execution_binding.get("closed_record_sha256")
        or _file_sha256(authenticated_handoff.execution_record_path)
        != handoff_execution_binding.get("sha256")
        or authenticated_handoff.execution_record_path.absolute()
        != _cluster_path(_required_string(handoff_execution_binding, "uri")).absolute()
        or authenticated_handoff.root.absolute()
        != _cluster_path(
            _required_string(handoff_binding, "output_root_uri")
        ).absolute()
    ):
        raise ValueError("Q8 handoff serving authorization binding drift")

    bf16_binding = _mapping(sources, "bf16_handoff")
    bf16_manifest_binding = _mapping(bf16_binding, "manifest")
    authenticated_bf16 = require_publication_bf16_handoff_serving_authorization(
        bf16_handoff_authorization,
        expected_manifest_file_sha256=_required_sha256(bf16_manifest_binding, "sha256"),
        expected_manifest_closed_record_sha256=_required_sha256(
            bf16_manifest_binding, "closed_record_sha256"
        ),
        expected_input_bundle_sha256=_required_sha256(
            artifact_pins, "input_bundle_sha256"
        ),
    )
    bf16_authorization_binding = _mapping(bf16_binding, "authorization")
    bf16_execution_binding = _mapping(bf16_binding, "execution")
    if (
        bf16_handoff_authorization.causal_closure_sha256
        != bf16_authorization_binding.get("causal_closure_sha256")
        or bf16_handoff_authorization.ledger_id
        != bf16_authorization_binding.get("ledger_id")
        or bf16_handoff_authorization.ledger_path_sha256
        != bf16_authorization_binding.get("ledger_path_sha256")
        or bf16_handoff_authorization.ledger_prefix.to_record()
        != bf16_authorization_binding.get("ledger_prefix")
        or bf16_handoff_authorization.predecessor_prefix.to_record()
        != bf16_authorization_binding.get("predecessor_prefix")
        or bf16_handoff_authorization.producer_batch_prefix.to_record()
        != bf16_authorization_binding.get("producer_batch_prefix")
        or authenticated_bf16.record.get("closed_record_sha256")
        != bf16_execution_binding.get("closed_record_sha256")
        or _file_sha256(authenticated_bf16.execution_record_path)
        != bf16_execution_binding.get("sha256")
        or authenticated_bf16.execution_record_path.absolute()
        != _cluster_path(_required_string(bf16_execution_binding, "uri")).absolute()
        or authenticated_bf16.manifest_path.absolute()
        != _cluster_path(_required_string(bf16_manifest_binding, "uri")).absolute()
        or authenticated_bf16.root.absolute()
        != _cluster_path(_required_string(bf16_binding, "output_root_uri")).absolute()
        or authenticated_bf16.source_root.absolute()
        != _cluster_path(_required_string(bf16_binding, "source_root_uri")).absolute()
    ):
        raise ValueError("BF16 handoff serving authorization binding drift")
    return selection


def _require_prior_waves_succeeded(
    execution_plan_record: Mapping[str, Any],
    *,
    ledger: Any,
    wave_index: int,
) -> None:
    prior_job_ids = [
        job_id
        for wave in _mapping_sequence(execution_plan_record, "launch_waves")
        if _required_int(wave, "wave_index") < wave_index
        for job_id in cast(list[str], wave.get("job_ids"))
    ]
    plan_sha256 = _required_sha256(execution_plan_record, "closed_record_sha256")
    terminal_by_attempt = {item.attempt_id: item for item in ledger.terminal_actuals}
    for job_id in prior_job_ids:
        attempt_id = publication_latency_reservation_attempt_id(plan_sha256, job_id)
        actual = terminal_by_attempt.get(attempt_id)
        if (
            actual is None
            or actual.terminal_state != "succeeded"
            or actual.verification_source != "direct_databricks_runs_get"
        ):
            raise RuntimeError(
                "all earlier latency waves must have verified success before launch"
            )


def _validate_latency_control_plane_run(
    run: Mapping[str, Any],
    *,
    job_record: Mapping[str, Any],
    submit_payload: Mapping[str, Any],
    receipt_run_id: str,
) -> dict[str, Any]:
    run_id = _databricks_id(run.get("run_id"), "runs/get run_id")
    if run_id != receipt_run_id:
        raise ValueError("runs/get run ID differs from latency submit receipt")
    if run.get("run_name") != submit_payload.get("run_name"):
        raise ValueError("runs/get run name differs from latency submit payload")
    if run.get("run_type") not in (None, "SUBMIT_RUN"):
        raise ValueError("latency run is not a one-time submit run")
    if run.get("repair_history") not in (None, []):
        raise ValueError("latency run has repair history")
    if run.get("original_attempt_run_id") not in (None, 0, "0"):
        raise ValueError("latency run is not attempt zero")
    state = _mapping(run, "state")
    life_cycle = state.get("life_cycle_state")
    result_state = state.get("result_state")
    if life_cycle not in _TERMINAL_LIFE_CYCLE_STATES:
        raise ValueError("latency runs/get response is not terminal")
    run_start = _nonnegative_int(run.get("start_time"), "run.start_time")
    run_end = _positive_int(run.get("end_time"), "run.end_time")
    if run_end <= run_start:
        raise ValueError("latency run terminal interval is invalid")
    tasks = run.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 1
        or not isinstance(tasks[0], Mapping)
    ):
        raise ValueError("latency runs/get must contain exactly one task")
    task = tasks[0]
    if task.get("task_key") != job_record.get("task_key"):
        raise ValueError("latency runs/get task key drift")
    if task.get("attempt_number") not in (None, 0):
        raise ValueError("latency task was retried")
    task_state = _mapping(task, "state")
    task_life_cycle = task_state.get("life_cycle_state")
    task_result_state = task_state.get("result_state")
    if task_life_cycle not in _TERMINAL_LIFE_CYCLE_STATES:
        raise ValueError("latency runs/get task is not terminal")
    task_start = _nonnegative_int(task.get("start_time"), "task.start_time")
    task_end = _positive_int(task.get("end_time"), "task.end_time")
    if not run_start <= task_start < task_end <= run_end:
        raise ValueError("latency task interval is not nested in the run")
    task_run_id = _databricks_id(task.get("run_id"), "task run_id")
    cluster_instance = _mapping(task, "cluster_instance")
    cluster_id = _required_string(cluster_instance, "cluster_id")
    submitted_task = _mapping_value(submit_payload["tasks"][0], "submitted task")
    submitted_cluster = _mapping(submitted_task, "new_cluster")
    observed_cluster_value = task.get("new_cluster")
    if not isinstance(observed_cluster_value, Mapping):
        cluster_spec = run.get("cluster_spec")
        if isinstance(cluster_spec, Mapping):
            observed_cluster_value = cluster_spec.get("new_cluster")
    observed_cluster = _mapping_value(observed_cluster_value, "observed cluster")
    for field_name, expected in submitted_cluster.items():
        if observed_cluster.get(field_name) != expected:
            raise ValueError(f"latency runs/get cluster {field_name} drift")
    succeeded = (
        life_cycle == "TERMINATED"
        and result_state == "SUCCESS"
        and task_life_cycle == "TERMINATED"
        and task_result_state == "SUCCESS"
    )
    return {
        "cluster_id": cluster_id,
        "run_id": run_id,
        "task_run_id": task_run_id,
        "terminal_state": "succeeded" if succeeded else "failed",
    }


def _terminal_actual_record(actual: Any) -> dict[str, Any]:
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


def _submit_payload_sha256(payload: Mapping[str, Any]) -> str:
    _snapshot, canonical = canonical_databricks_submit_payload_snapshot(payload)
    return sha256(canonical).hexdigest()


def _control_plane_status_sha256(run: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(
            run,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _validate_job_bound_source_files(job_record: Mapping[str, Any]) -> None:
    for item in _mapping_sequence(job_record, "artifact_files"):
        _verify_regular_file_sha256(
            _cluster_path(_required_string(item, "uri")),
            _required_sha256(item, "sha256"),
            f"worker artifact {item.get('role')}",
        )
    for item in _mapping_sequence(job_record, "input_files"):
        _verify_regular_file_sha256(
            _cluster_path(_required_string(item, "uri")),
            _required_sha256(item, "sha256"),
            f"input dataset {item.get('dataset')}",
        )
    request_order = _mapping(job_record, "request_order")
    schedule_path = _cluster_path(_required_string(request_order, "schedule_uri"))
    _verify_regular_file_sha256(
        schedule_path,
        _required_sha256(request_order, "file_sha256"),
        "publication schedule",
    )
    schedule = _read_json_file(schedule_path, "publication schedule")
    _validate_job_schedule_record(schedule, job_record=job_record)
    handoff_value = job_record.get("handoff")
    if handoff_value is None:
        return
    handoff = _mapping_value(handoff_value, "handoff")
    if handoff.get("source_kind") == "distributed_q8_generation":
        execution = _mapping(handoff, "execution")
        root = _cluster_path(_required_string(handoff, "output_root_uri"))
        result = read_publication_latency_handoff_generation_result(root)
        if (
            _file_sha256(result.execution_record_path) != execution.get("sha256")
            or result.record.get("closed_record_sha256")
            != execution.get("closed_record_sha256")
            or result.record.get("execution_mode")
            != PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED
        ):
            raise ValueError("worker distributed handoff binding drift")
    elif handoff.get("source_kind") == "closed_bf16_bundle":
        manifest_binding = _mapping(handoff, "manifest")
        manifest_path = _cluster_path(_required_string(manifest_binding, "uri"))
        _verify_regular_file_sha256(
            manifest_path,
            _required_sha256(manifest_binding, "sha256"),
            "BF16 handoff manifest",
        )
        manifest = read_publication_latency_handoff_bundle(manifest_path)
        validate_publication_latency_handoff_bundle(
            manifest,
            bundle_root=_cluster_path(_required_string(handoff, "source_root_uri")),
        )
        if manifest.get("closed_record_sha256") != manifest_binding.get(
            "closed_record_sha256"
        ):
            raise ValueError("worker BF16 handoff binding drift")
    else:
        raise ValueError("worker handoff source kind is invalid")


def _validate_job_schedule_record(
    schedule: Mapping[str, Any],
    *,
    job_record: Mapping[str, Any],
) -> None:
    request_order = _mapping(job_record, "request_order")
    validate_publication_latency_block_schedule(
        schedule,
        examples=_schedule_examples(schedule),
        expected_input_bundle_sha256=_required_sha256(
            request_order, "input_bundle_sha256"
        ),
    )
    for field_name in (
        "closed_record_sha256",
        "deployment_block",
        "requests_sha256",
        "seed_sha256",
    ):
        if schedule.get(field_name) != request_order.get(field_name):
            raise ValueError(f"worker schedule {field_name} binding drift")


def _snapshot_publication_latency_inputs(
    job_record: Mapping[str, Any],
    *,
    snapshot_root: Path,
) -> tuple[Path, tuple[str, ...]]:
    _reject_existing_symlink_ancestors(snapshot_root, "worker source snapshot")
    if snapshot_root.exists() or snapshot_root.is_symlink():
        raise FileExistsError(f"worker source snapshot already exists: {snapshot_root}")
    snapshot_root.mkdir(parents=True, exist_ok=False)
    request_order = _mapping(job_record, "request_order")
    schedule_source = _cluster_path(_required_string(request_order, "schedule_uri"))
    schedule_target = snapshot_root / "publication-latency-schedule.json"
    _copy_verified_file_exclusive(
        schedule_source,
        schedule_target,
        _required_sha256(request_order, "file_sha256"),
        "publication schedule snapshot",
    )
    specs: list[str] = []
    for item in _mapping_sequence(job_record, "input_files"):
        dataset = _required_string(item, "dataset")
        target = snapshot_root / f"{dataset}.jsonl"
        _copy_verified_file_exclusive(
            _cluster_path(_required_string(item, "uri")),
            target,
            _required_sha256(item, "sha256"),
            f"{dataset} input snapshot",
        )
        specs.append(f"{dataset}={target}")
    directory = os.open(snapshot_root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return schedule_target, tuple(specs)


def _copy_verified_file_exclusive(
    source: Path,
    destination: Path,
    expected_sha256: str,
    field_name: str,
) -> None:
    _verify_regular_file_sha256(source, expected_sha256, field_name)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    digest = sha256()
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as target:
            for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                digest.update(block)
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != expected_sha256 or _file_sha256(destination) != (
        expected_sha256
    ):
        destination.unlink(missing_ok=True)
        raise ValueError(f"{field_name} changed during snapshot")


def _cleanup_publication_latency_worker_state(
    job_record: Mapping[str, Any],
    *,
    config: VLLMSmokeBenchmarkConfig,
) -> None:
    """Remove node-local/UC staging before a SUCCESS result can be sealed."""

    roots = [config.local_root]
    handoff_value = job_record.get("handoff")
    if isinstance(handoff_value, Mapping) and handoff_value.get("stage_kind") == (
        "uc_mounted"
    ):
        stage = _cluster_path(_required_string(handoff_value, "stage_uri"))
        if not str(stage).startswith("/Volumes/"):
            raise ValueError("UC cleanup target escaped /Volumes")
        roots.append(stage)
    for root in roots:
        _reject_existing_symlink_ancestors(root, "worker cleanup target")
        if root.is_symlink():
            raise ValueError(f"worker cleanup target is a symlink: {root}")
        if root.exists():
            if not root.is_dir():
                raise ValueError(f"worker cleanup target is not a directory: {root}")
            shutil.rmtree(root)
        if root.exists() or root.is_symlink():
            raise RuntimeError(f"worker cleanup did not remove {root}")


def _cluster_path(uri: str) -> Path:
    if uri.startswith("dbfs:/"):
        return Path("/dbfs") / uri.removeprefix("dbfs:/").lstrip("/")
    if uri.startswith("file:"):
        return Path(uri.removeprefix("file:"))
    return Path(uri)


def _reject_existing_symlink_ancestors(path: Path, field_name: str) -> None:
    """Reject any symlink in the existing prefix without resolving through it."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"{field_name} has a symlink ancestor: {current}")


def _verify_regular_file_sha256(
    path: Path,
    expected: str,
    field_name: str,
) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{field_name} is not a regular nonsymlink file: {path}")
    if _file_sha256(path) != _require_sha256_value(expected, field_name):
        raise ValueError(f"{field_name} SHA-256 drift")


def _verified_output_file_sha256(path: Path, field_name: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required {field_name} output is missing")
    return _file_sha256(path)


def _read_json_file(path: Path, field_name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{field_name} is not a regular nonsymlink file")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} is not canonical JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must contain a JSON object")
    return value


def _read_latency_controller_record(path: Path, field_name: str) -> dict[str, Any]:
    _reject_existing_symlink_ancestors(path, field_name)
    value = _read_json_file(path, field_name)
    if path.read_bytes() != (_canonical_json(value) + "\n").encode("utf-8"):
        raise ValueError(f"{field_name} is not canonical JSON")
    observed = value.get("closed_record_sha256")
    if not isinstance(observed, str) or observed != _closed_record_sha256(value):
        raise ValueError(f"{field_name} closed digest mismatch")
    return value


def _read_bound_json_uri(
    binding: Mapping[str, Any],
    field_name: str,
) -> dict[str, Any]:
    path = _cluster_path(_required_string(binding, "uri"))
    _verify_regular_file_sha256(path, _required_sha256(binding, "sha256"), field_name)
    return _read_json_file(path, field_name)


def _read_jsonl_file(path: Path, *, required: bool) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise ValueError(f"required connector telemetry is missing: {path}")
        return []
    if not path.is_file() or path.is_symlink():
        raise ValueError("connector telemetry is not a regular nonsymlink file")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"connector telemetry line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError("connector telemetry rows must be objects")
        records.append(value)
    return records


def _job_output_uri(job_record: Mapping[str, Any], filename: str) -> str:
    return _join_durable_uri(
        _required_string(_mapping(job_record, "output"), "directory_uri"),
        filename,
    )


def _write_canonical_json_exclusive(path: Path, record: Mapping[str, Any]) -> None:
    _reject_existing_symlink_ancestors(path, "result seal")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlink_ancestors(path, "result seal")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"result seal already exists: {path}")
    raw = (_canonical_json(record) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _create_latency_phase_lease_root(path: str | Path) -> Path:
    root = Path(path).expanduser().absolute()
    _reject_existing_symlink_ancestors(root, "latency phase lease")
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"latency phase lease already exists: {root}")
    if not root.parent.is_dir():
        raise ValueError("latency phase lease parent must already be a real directory")
    root.mkdir()
    _fsync_directory(root.parent)
    return root


def _remove_empty_latency_phase_lease_root(root: Path) -> None:
    for path in tuple(root.iterdir()):
        if path.name == "phase-lease.json" and path.is_file() and not path.is_symlink():
            path.unlink()
    if root.is_dir() and not root.is_symlink() and not any(root.iterdir()):
        root.rmdir()
        _fsync_directory(root.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _databricks_id(value: Any, field_name: str) -> str:
    text = str(value) if type(value) is int else value
    if not isinstance(text, str) or _CLOUD_ID_RE.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a positive canonical decimal ID")
    return text


def _nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _task_key(job_id: str) -> str:
    return f"latency_{sha256(job_id.encode('utf-8')).hexdigest()[:24]}"


def _mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    item = value.get(field_name)
    if not isinstance(item, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return item


def _mapping_value(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _mapping_sequence(
    value: Mapping[str, Any],
    field_name: str,
) -> list[Mapping[str, Any]]:
    items = value.get(field_name)
    if (
        isinstance(items, (str, bytes, bytearray))
        or not isinstance(items, Sequence)
        or any(not isinstance(item, Mapping) for item in items)
    ):
        raise ValueError(f"{field_name} must be an array of objects")
    return [cast(Mapping[str, Any], item) for item in items]


def _required_string(value: Mapping[str, Any], field_name: str) -> str:
    item = value.get(field_name)
    if not isinstance(item, str) or not item or item != item.strip():
        raise ValueError(f"{field_name} must be a nonempty canonical string")
    return item


def _required_int(value: Mapping[str, Any], field_name: str) -> int:
    item = value.get(field_name)
    if type(item) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return item


def _finite_positive_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return result


def _finite_nonnegative_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return result


def _required_sha256(value: Mapping[str, Any], field_name: str) -> str:
    return _require_sha256_value(value.get(field_name), field_name)


def _require_sha256(value: Any, field_name: str) -> str:
    return _require_sha256_value(value, field_name)


def _require_sha256_value(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _safe_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} is not a safe identifier")
    return value


def _durable_uri(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical durable URI")
    if not (value.startswith("dbfs:/") or value.startswith("/Volumes/")):
        raise ValueError(f"{field_name} must use dbfs:/ or /Volumes/")
    if "//" in value.removeprefix("dbfs:") or any(
        part in {"", ".", ".."}
        for part in PurePosixPath(value.removeprefix("dbfs:")).parts[1:]
    ):
        raise ValueError(f"{field_name} is not a canonical durable URI")
    return value.rstrip("/")


def _uc_volume_uri(value: Any, field_name: str) -> str:
    uri = _durable_uri(value, field_name)
    if not uri.startswith("/Volumes/"):
        raise ValueError(f"{field_name} must be a Unity Catalog volume URI")
    return uri


def _join_durable_uri(root: str, *parts: str) -> str:
    prefix = _durable_uri(root, "durable root")
    safe_parts: list[str] = []
    for index, part in enumerate(parts):
        if not isinstance(part, str) or not part or "/" in part or part in {".", ".."}:
            raise ValueError(f"durable path component {index} is invalid")
        safe_parts.append(part)
    return prefix.rstrip("/") + "/" + "/".join(safe_parts)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    body = dict(record)
    body["closed_record_sha256"] = ""
    return _canonical_sha256(body)


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    field_name: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise ValueError(
            f"{field_name} schema is not closed (missing={missing}, extra={extra})"
        )


def _final_artifacts_from_record(
    record: Mapping[str, Any],
) -> PublicationLatencyFinalArtifactPins:
    _require_exact_keys(
        record,
        {
            "bf16_handoff_generation_root_uri",
            "bf16_handoff_source_root_uri",
            "files",
            "handoff_generation_root_uri",
            "output_root_uri",
            "source_revision",
            "uc_handoff_stage_root_uri",
        },
        "final artifacts",
    )
    files = tuple(
        PublicationLatencyArtifactFile(
            role=_required_string(item, "role"),
            uri=_required_string(item, "uri"),
            sha256=_required_sha256(item, "sha256"),
        )
        for item in _mapping_sequence(record, "files")
    )
    return PublicationLatencyFinalArtifactPins(
        source_revision=_required_string(record, "source_revision"),
        files=files,
        output_root_uri=_required_string(record, "output_root_uri"),
        handoff_generation_root_uri=_required_string(
            record, "handoff_generation_root_uri"
        ),
        bf16_handoff_generation_root_uri=_required_string(
            record, "bf16_handoff_generation_root_uri"
        ),
        bf16_handoff_source_root_uri=_required_string(
            record, "bf16_handoff_source_root_uri"
        ),
        uc_handoff_stage_root_uri=_required_string(record, "uc_handoff_stage_root_uri"),
    )


def _one_parameter(parameters: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(parameters) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(parameters):
        raise ValueError(f"runner flag {flag!r} must appear exactly once with a value")
    return parameters[positions[0] + 1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute one governed publication latency worker"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_job = subparsers.add_parser("run-job")
    run_job.add_argument("--job-record-json", required=True)
    run_job.add_argument("--expected-job-sha256", required=True)
    run_job.add_argument("--cloud-run-id", required=True)
    run_job.add_argument("--task-run-id", required=True)
    write_runner = subparsers.add_parser("write-runner")
    write_runner.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "write-runner":
        path = write_publication_latency_runner_script(args.output)
        print(path)
        return 0
    try:
        value = json.loads(args.job_record_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--job-record-json is invalid JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != args.job_record_json:
        raise ValueError("--job-record-json must be one canonical JSON object")
    result = execute_publication_latency_job_record(
        value,
        expected_job_sha256=args.expected_job_sha256,
        cloud_run_id=args.cloud_run_id,
        task_run_id=args.task_run_id,
    )
    print(_required_sha256(result, "closed_record_sha256"))
    return 0


__all__ = [
    "PUBLICATION_LATENCY_COLLECTION_RECORD_TYPE",
    "PUBLICATION_LATENCY_DATABRICKS_AVAILABILITY",
    "PUBLICATION_LATENCY_DATABRICKS_DATA_SECURITY_MODE",
    "PUBLICATION_LATENCY_DATABRICKS_ZONES",
    "PUBLICATION_LATENCY_EXECUTION_PLAN_RECORD_TYPE",
    "PUBLICATION_LATENCY_JOB_RECORD_TYPE",
    "PUBLICATION_LATENCY_JOB_RESULT_RECORD_TYPE",
    "PUBLICATION_LATENCY_MAX_OUTPUT_TOKENS",
    "PUBLICATION_LATENCY_Q8_DTYPE",
    "PUBLICATION_LATENCY_RAM_PAYLOAD_CACHE_BYTES",
    "PUBLICATION_LATENCY_RAM_PRIME_TARGETS",
    "PUBLICATION_LATENCY_RESULT_FILENAME",
    "PUBLICATION_LATENCY_RUNNER_SCRIPT",
    "PUBLICATION_LATENCY_RUNNER_SHA256",
    "PUBLICATION_LATENCY_RUN_TIMEOUT_SECONDS",
    "PUBLICATION_LATENCY_SUBMISSION_RECORD_TYPE",
    "PUBLICATION_LATENCY_SUMMARY_RECORD_TYPE",
    "PublicationLatencyArtifactFile",
    "PublicationLatencyCollectionAuthorization",
    "PublicationLatencyFinalArtifactPins",
    "PublicationLatencyWaveAuthorization",
    "PublicationLatencyWaveSubmissionAuthorization",
    "aggregate_publication_latency_campaign",
    "build_databricks_publication_latency_run_submit_payload",
    "build_publication_latency_execution_plan",
    "collect_publication_latency_campaign",
    "collect_publication_latency_launch_wave",
    "execute_publication_latency_job_record",
    "main",
    "publication_latency_reservation_attempt_id",
    "publication_latency_submit_payloads",
    "publication_latency_vllm_config",
    "render_publication_latency_job_record",
    "require_publication_latency_collection_authorization",
    "seal_publication_latency_job_result",
    "submit_publication_latency_launch_wave",
    "resume_publication_latency_launch_wave",
    "validate_publication_latency_collection_record",
    "validate_publication_latency_execution_plan_record",
    "validate_publication_latency_execution_sources",
    "validate_publication_latency_job_record",
    "validate_publication_latency_job_result_record",
    "validate_publication_latency_summary_record",
    "write_publication_latency_runner_script",
]


if __name__ == "__main__":  # pragma: no cover - CLI boundary.
    raise SystemExit(main())
