"""Governed execution and analysis for the frozen 115-job latency campaign.

The module deliberately separates five authorities:

* a closed controller plan proves the frozen factorial and immutable inputs;
* issuer-only CPU closures authenticate mounted sources without a Mac DBFS mount;
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
from dataclasses import dataclass, field, replace
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
    BenchmarkRunResult,
    benchmark_gate_inputs_from_record,
    benchmark_record_aggregate_issues,
    benchmark_record_payload_digest,
    benchmark_run_result_from_record,
)
from document_kv_cache.benchmark_gates import (
    benchmark_evidence_gate_to_record,
    evaluate_benchmark_evidence_gate,
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
    DatabricksClusterHourLedger,
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
    DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES,
    DatabricksURLOpener,
    DatabricksWorkspaceConfig,
    bind_databricks_run_idempotency_token,
    download_databricks_volume_file_bytes,
    get_databricks_run,
    list_databricks_volume_directory,
    require_databricks_current_user_name,
    require_databricks_run_idempotency_token,
    resume_pre_reserved_databricks_run,
    submit_databricks_run,
    submit_pre_reserved_databricks_run,
    upload_databricks_volume_file_bytes_exclusive,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_A10G_GPU_MEMORY_UTILIZATION,
    GPU_QUALIFICATION_DECODE_HEADROOM_TOKENS,
    GPU_QUALIFICATION_MODEL_ID,
    GPU_QUALIFICATION_MODEL_REVISION,
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPU_QUALIFICATION_PUBLICATION_BACKEND,
    GPUQualificationSelection,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_ARTIFACT_KEYS,
    GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
    GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS,
    GPUQualificationArtifactPinsV2,
    validate_gpu_qualification_evidence_v2_record,
    validate_gpu_qualification_plan_v2_record,
    validate_gpu_qualification_v2_runtime_attestation,
)
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PATCHED_WHEEL_SHA256,
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
    PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS,
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
    PUBLICATION_STORAGE_INPUTS_RECORD_TYPE,
    PUBLICATION_STORAGE_INPUTS_SCHEMA_VERSION,
    PublicationLatencyExample,
    project_publication_latency_request_order,
    select_publication_storage_examples,
    validate_publication_latency_block_schedule,
    validate_publication_storage_inputs_record,
    validate_publication_storage_block_schedule,
)
from document_kv_cache.publication_bf16_handoff_generation import (
    PUBLICATION_BF16_HANDOFF_EXECUTION_MODE,
    PUBLICATION_BF16_HANDOFF_WORKER_COUNT,
)
from document_kv_cache.publication_handoff_closure_coordinator import (
    PublicationHandoffRemoteClosureAuthorization,
    require_bf16_handoff_remote_closure_authorization,
    require_q8_handoff_remote_closure_authorization,
)
from document_kv_cache.publication_latency_handoff_generation import (
    PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED,
    read_publication_latency_handoff_generation_result,
)
from document_kv_cache.serving_env import (
    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
    GPU_RUNTIME_PYTHONWARNINGS,
    VLLM_PATCHED_WHEEL_SHA256_ENV,
    VLLM_PATCHED_WHEEL_URI_ENV,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
)
from document_kv_cache.vllm_smoke import (
    VLLMNativeRuntimeBundleV2,
    VLLMSmokeBenchmarkConfig,
    run_vllm_smoke_benchmark,
)


PUBLICATION_LATENCY_EXECUTION_PLAN_RECORD_TYPE: Final = (
    "cachet.publication_latency_execution_plan.v2"
)
PUBLICATION_LATENCY_JOB_RECORD_TYPE: Final = "cachet.publication_latency_job.v2"
PUBLICATION_LATENCY_JOB_RESULT_RECORD_TYPE: Final = (
    "cachet.publication_latency_job_result.v2"
)
PUBLICATION_LATENCY_COMPONENT_EVIDENCE_POLICY: Final = "smoke"
PUBLICATION_LATENCY_REQUEST_CUSTOMIZATION_DIGEST: Final = (
    "440181b5f7930106194b542de751661bbd5662a071e7d10b64cf8172ac29774f"
)
PUBLICATION_LATENCY_SUBMISSION_RECORD_TYPE: Final = (
    "cachet.publication_latency_wave_submission.v2"
)
PUBLICATION_LATENCY_TERMINAL_RECEIPT_RECORD_TYPE: Final = (
    "cachet.publication_latency_terminal_receipt.v2"
)
PUBLICATION_LATENCY_COLLECTION_RECORD_TYPE: Final = (
    "cachet.publication_latency_collection.v2"
)
PUBLICATION_LATENCY_SUMMARY_RECORD_TYPE: Final = (
    "cachet.publication_latency_estimation_summary.v2"
)
PUBLICATION_LATENCY_SCHEMA_VERSION: Final = 2
PUBLICATION_LATENCY_SOURCE_CLOSURE_REQUEST_RECORD_TYPE: Final = (
    "cachet.publication_latency_source_closure_request.v2"
)
PUBLICATION_LATENCY_SOURCE_CLOSURE_REQUEST_AUTHORIZATION_RECORD_TYPE: Final = (
    "cachet.publication_latency_source_closure_request_authorization.v2"
)
PUBLICATION_LATENCY_SOURCE_CLOSURE_RESULT_RECORD_TYPE: Final = (
    "cachet.publication_latency_source_closure_result.v2"
)
PUBLICATION_LATENCY_SOURCE_CLOSURE_SCHEMA_VERSION: Final = 2
PUBLICATION_LATENCY_SOURCE_CLOSURE_NODE_TYPE_ID: Final = "c5d.4xlarge"
PUBLICATION_LATENCY_SOURCE_CLOSURE_SPARK_VERSION: Final = (
    "15.4.x-cpu-ml-scala2.12"
)
PUBLICATION_LATENCY_SOURCE_CLOSURE_TIMEOUT_SECONDS: Final = 2 * 60 * 60
PUBLICATION_LATENCY_SOURCE_CLOSURE_MAX_BYTES: Final = (
    DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES
)
PUBLICATION_LATENCY_RUN_TIMEOUT_SECONDS: Final = 12 * 60 * 60
PUBLICATION_LATENCY_TASK_MAX_RETRIES: Final = 0
PUBLICATION_LATENCY_MAX_OUTPUT_TOKENS: Final = 256
PUBLICATION_LATENCY_TEMPERATURE: Final = 0.0
PUBLICATION_LATENCY_GENERATION_SEED: Final = 17
PUBLICATION_LATENCY_DATABRICKS_AVAILABILITY: Final = "ON_DEMAND"
PUBLICATION_LATENCY_DATABRICKS_DATA_SECURITY_MODE: Final = "SINGLE_USER"
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
PUBLICATION_LATENCY_REMOTE_RESULT_MAX_BYTES: Final = (
    DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES
)
PUBLICATION_LATENCY_REMOTE_ARTIFACT_MAX_BYTES: Final = (
    DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES
)
PUBLICATION_LATENCY_CONTROL_PLANE_MAX_BYTES: Final = 16 * 1024 * 1024
PUBLICATION_LATENCY_REMOTE_TREE_MAX_BYTES: Final = 64 * 1024 * 1024
PUBLICATION_LATENCY_REMOTE_DIRECTORY_MAX_BYTES: Final = 128 * 1024 * 1024
PUBLICATION_LATENCY_REMOTE_TREE_MAX_FILES: Final = 10
PUBLICATION_LATENCY_REMOTE_AUXILIARY_FILENAMES: Final = frozenset(
    {
        "prepared-handoff-coverage.json",
        "prepared-handoff-generation.json",
        "prewarm-cache-prefix.json",
        "vllm-server.log",
    }
)
PUBLICATION_LATENCY_REMOTE_DIRECTORY_MAX_FILES: Final = (
    PUBLICATION_LATENCY_REMOTE_TREE_MAX_FILES
    + len(PUBLICATION_LATENCY_REMOTE_AUXILIARY_FILENAMES)
)

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
_DESCRIPTIVE_CACHE_TELEMETRY_FIELDS = (
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
)


PUBLICATION_LATENCY_RUNNER_SCRIPT = r"""from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path


_FLASHINFER_LOGGING_LEVEL = "__GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL__"
_PYTHONWARNINGS = "__GPU_RUNTIME_PYTHONWARNINGS__"


def _cluster_path(uri: str) -> str:
    if uri.startswith("dbfs:/Volumes/"):
        return "/Volumes/" + uri.removeprefix("dbfs:/Volumes/")
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


def _pip_subprocess_environment() -> dict[str, str]:
    env = dict(os.environ)
    for variable_name in tuple(env):
        if variable_name.upper().startswith(("PIP_", "_PIP_")):
            env.pop(variable_name)
    for variable_name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(variable_name, None)
    env.update(
        {
            "FLASHINFER_LOGGING_LEVEL": _FLASHINFER_LOGGING_LEVEL,
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONWARNINGS": _PYTHONWARNINGS,
        }
    )
    return env


_CHILD_STUB = (
    "import os\n"
    "import runpy\n"
    "import sys\n"
    "if (\n"
    "    os.environ.get('FLASHINFER_LOGGING_LEVEL')\n"
    f"    != {_FLASHINFER_LOGGING_LEVEL!r}\n"
    "    or os.environ.get('PYTHONWARNINGS')\n"
    f"    != {_PYTHONWARNINGS!r}\n"
    "    or tuple(sys.warnoptions)\n"
    f"    != tuple({_PYTHONWARNINGS!r}.split(','))\n"
    "):\n"
    "    raise RuntimeError(\n"
    "        'publication latency child lacks the pinned CUDA warning startup policy'\n"
    "    )\n"
    "sys.argv = ['document_kv_cache.publication_latency_execution', *sys.argv[1:]]\n"
    "runpy.run_module(\n"
    "    'document_kv_cache.publication_latency_execution', run_name='__main__'\n"
    ")\n"
)


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
    _verified(__file__, args.runner_sha256, "executing publication latency runner")
    wheel = _verified(
        args.package_wheel_uri,
        args.package_wheel_sha256,
        "Cachet package wheel",
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            "cachet-kv @ " + Path(wheel).resolve().as_uri()
            + "#sha256=" + args.package_wheel_sha256,
        ],
        env=_pip_subprocess_environment(),
    )
    subprocess.check_call(
        [
            sys.executable,
            "-P",
            "-c",
            _CHILD_STUB,
            "run-job",
            "--job-record-json",
            args.job_record_json,
            "--expected-job-sha256",
            args.expected_job_sha256,
            "--cloud-run-id",
            args.cloud_run_id,
            "--task-run-id",
            args.task_run_id,
        ],
        env=_pip_subprocess_environment(),
    )


if __name__ == "__main__":
    main()
""".replace(
    "__GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL__",
    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
).replace("__GPU_RUNTIME_PYTHONWARNINGS__", GPU_RUNTIME_PYTHONWARNINGS)
PUBLICATION_LATENCY_RUNNER_SHA256: Final = sha256(
    PUBLICATION_LATENCY_RUNNER_SCRIPT.encode("utf-8")
).hexdigest()

PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT = r"""from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


_FINAL_RUNTIME_VERIFIER_TIMEOUT_SECONDS = 300.0


def _volume_path(uri: str) -> str:
    prefix = "dbfs:/Volumes/"
    if not uri.startswith(prefix):
        raise ValueError("source-closure artifacts must use dbfs:/Volumes URIs")
    return "/Volumes/" + uri.removeprefix(prefix)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(uri: str, expected: str, label: str) -> str:
    path = _volume_path(uri)
    if _sha256(path) != expected:
        raise ValueError(label + " SHA-256 drift")
    return path


def _pip_subprocess_environment() -> dict[str, str]:
    env = dict(os.environ)
    for variable_name in tuple(env):
        if variable_name.upper().startswith(("PIP_", "_PIP_")):
            env.pop(variable_name)
    for variable_name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(variable_name, None)
    env.update(
        {
            "FLASHINFER_LOGGING_LEVEL": "__GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL__",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONWARNINGS": "__GPU_RUNTIME_PYTHONWARNINGS__",
        }
    )
    return env


def _direct_reference(distribution: str, path: str, expected: str) -> str:
    return distribution + " @ " + Path(path).resolve().as_uri() + "#sha256=" + expected


def _runtime_marker(args: argparse.Namespace) -> str:
    record = {
        "domain": "cachet.publication.latency_source_closure.runtime.v2",
        "package_wheel_sha256": args.package_wheel_sha256,
        "patched_flashinfer_wheel_sha256": args.patched_flashinfer_wheel_sha256,
        "patched_vllm_wheel_sha256": args.patched_vllm_wheel_sha256,
        "runtime_closure_manifest_sha256": args.runtime_closure_manifest_sha256,
        "runtime_lock_sha256": args.runtime_lock_sha256,
    }
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _verify_locked_runtime(
    *,
    venv_python: str,
    runtime_lock: str,
    patched_vllm_wheel: str,
    patched_flashinfer_wheel: str,
    runtime_closure_manifest: str,
    package_wheel: str,
    package_wheel_sha256: str,
    environment: dict[str, str],
) -> dict[str, object]:
    verifier = (
        "import json,sys; from document_kv_cache._gpu_qualification_sentinels_v2 "
        "import verify_gpu_qualification_v2_runtime_installation as verify; "
        "print(json.dumps(verify(runtime_lock=sys.argv[1],vllm_uri=sys.argv[2],"
        "flashinfer_uri=sys.argv[3],runtime_closure_manifest=sys.argv[4],"
        "package_uri=sys.argv[5],package_sha256=sys.argv[6]),sort_keys=True,"
        "separators=(',',':')))"
    )
    completed = subprocess.run(
        [
            venv_python,
            "-c",
            verifier,
            runtime_lock,
            Path(patched_vllm_wheel).resolve().as_uri(),
            Path(patched_flashinfer_wheel).resolve().as_uri(),
            runtime_closure_manifest,
            Path(package_wheel).resolve().as_uri(),
            package_wheel_sha256,
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=_FINAL_RUNTIME_VERIFIER_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError("source-closure native-v2 verifier process failed")
    if completed.stderr != "":
        raise RuntimeError("source-closure native-v2 verifier wrote stderr")
    try:
        verified = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("source-closure native-v2 verifier output is invalid") from exc
    if not isinstance(verified, dict):
        raise RuntimeError("source-closure native-v2 verifier did not emit an object")
    canonical = (
        json.dumps(verified, sort_keys=True, separators=(",", ":")) + "\n"
    )
    if completed.stdout != canonical:
        raise RuntimeError("source-closure native-v2 verifier output is not canonical")
    from document_kv_cache.gpu_qualification_v2 import (
        validate_gpu_qualification_v2_runtime_attestation,
    )

    validate_gpu_qualification_v2_runtime_attestation(verified)
    expected_direct_urls = {
        "flashinfer_direct_url": Path(patched_flashinfer_wheel).resolve().as_uri(),
        "vllm_direct_url": Path(patched_vllm_wheel).resolve().as_uri(),
    }
    if any(
        verified.get(field_name) != expected
        for field_name, expected in expected_direct_urls.items()
    ):
        raise RuntimeError("source-closure native-v2 verifier artifact origin drift")
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--package-wheel-uri", required=True)
    parser.add_argument("--package-wheel-sha256", required=True)
    parser.add_argument("--runtime-lock-uri", required=True)
    parser.add_argument("--runtime-lock-sha256", required=True)
    parser.add_argument("--patched-vllm-wheel-uri", required=True)
    parser.add_argument("--patched-vllm-wheel-sha256", required=True)
    parser.add_argument("--patched-flashinfer-wheel-uri", required=True)
    parser.add_argument("--patched-flashinfer-wheel-sha256", required=True)
    parser.add_argument("--runtime-closure-manifest-uri", required=True)
    parser.add_argument("--runtime-closure-manifest-sha256", required=True)
    parser.add_argument("--runtime-venv-dir", required=True)
    parser.add_argument("--request-uri", required=True)
    parser.add_argument("--request-file-sha256", required=True)
    parser.add_argument("--request-closed-record-sha256", required=True)
    parser.add_argument("--coordinator-run-id", required=True)
    args = parser.parse_args()
    if _sha256(__file__) != args.runner_sha256:
        raise ValueError("source-closure runner SHA-256 drift")
    package_wheel = _verified(
        args.package_wheel_uri,
        args.package_wheel_sha256,
        "source-closure package wheel",
    )
    runtime_lock = _verified(
        args.runtime_lock_uri,
        args.runtime_lock_sha256,
        "source-closure runtime lock",
    )
    patched_vllm_wheel = _verified(
        args.patched_vllm_wheel_uri,
        args.patched_vllm_wheel_sha256,
        "source-closure patched vLLM wheel",
    )
    patched_flashinfer_wheel = _verified(
        args.patched_flashinfer_wheel_uri,
        args.patched_flashinfer_wheel_sha256,
        "source-closure patched FlashInfer wheel",
    )
    runtime_closure_manifest = _verified(
        args.runtime_closure_manifest_uri,
        args.runtime_closure_manifest_sha256,
        "source-closure runtime closure manifest",
    )
    request = _verified(
        args.request_uri,
        args.request_file_sha256,
        "source-closure request file",
    )
    venv_dir = os.path.abspath(args.runtime_venv_dir)
    if not venv_dir.startswith("/local_disk0/"):
        raise ValueError("source-closure runtime venv escaped /local_disk0")
    if os.path.exists(venv_dir):
        raise FileExistsError("refusing to reuse an unverified source-closure runtime")
    pip_environment = _pip_subprocess_environment()
    subprocess.check_call(
        [sys.executable, "-m", "venv", "--copies", venv_dir],
        env=pip_environment,
    )
    venv_python = os.path.join(venv_dir, "bin", "python")
    pip_environment["VIRTUAL_ENV"] = venv_dir
    pip_environment["PATH"] = (
        os.path.dirname(venv_python)
        + os.pathsep
        + pip_environment.get("PATH", "")
    )
    pip = [venv_python, "-m", "pip"]
    subprocess.check_call(
        [*pip, "install", "--require-hashes", "--only-binary", ":all:", "-r", runtime_lock],
        env=pip_environment,
    )
    subprocess.check_call(
        [
            *pip,
            "install",
            "--no-deps",
            _direct_reference(
                "vllm", patched_vllm_wheel, args.patched_vllm_wheel_sha256
            ),
        ],
        env=pip_environment,
    )
    subprocess.check_call(
        [
            *pip,
            "install",
            "--no-deps",
            _direct_reference(
                "flashinfer-python",
                patched_flashinfer_wheel,
                args.patched_flashinfer_wheel_sha256,
            ),
        ],
        env=pip_environment,
    )
    subprocess.check_call(
        [
            *pip,
            "install",
            "--no-deps",
            _direct_reference(
                "cachet-kv", package_wheel, args.package_wheel_sha256
            ),
        ],
        env=pip_environment,
    )
    pip_check = subprocess.run(
        [*pip, "check"],
        check=True,
        capture_output=True,
        text=True,
        env=pip_environment,
    )
    if pip_check.stdout != "No broken requirements found.\n" or pip_check.stderr != "":
        raise RuntimeError("source-closure native-v2 pip check output differs")
    runtime_attestation = _verify_locked_runtime(
        venv_python=venv_python,
        runtime_lock=runtime_lock,
        patched_vllm_wheel=patched_vllm_wheel,
        patched_flashinfer_wheel=patched_flashinfer_wheel,
        runtime_closure_manifest=runtime_closure_manifest,
        package_wheel=package_wheel,
        package_wheel_sha256=args.package_wheel_sha256,
        environment=pip_environment,
    )
    env = dict(pip_environment)
    env["CACHET_LATENCY_SOURCE_CLOSURE_LOCKED_RUNTIME"] = (
        _runtime_marker(args)
    )
    env["CACHET_LATENCY_SOURCE_CLOSURE_RUNTIME_ATTESTATION"] = json.dumps(
        runtime_attestation, sort_keys=True, separators=(",", ":")
    )
    os.execve(
        venv_python,
        [
            venv_python,
            "-m",
            "document_kv_cache.publication_latency_execution",
            "run-source-closure",
            "--request-path",
            request,
            "--expected-request-file-sha256",
            args.request_file_sha256,
            "--expected-request-closed-record-sha256",
            args.request_closed_record_sha256,
            "--coordinator-run-id",
            args.coordinator_run_id,
        ],
        env,
    )


if __name__ == "__main__":
    main()
""".replace(
    "__GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL__",
    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
).replace("__GPU_RUNTIME_PYTHONWARNINGS__", GPU_RUNTIME_PYTHONWARNINGS)
PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SHA256: Final = sha256(
    PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT.encode("utf-8")
).hexdigest()
PUBLICATION_LATENCY_SOURCE_CLOSURE_EXCLUDED_ROLES: Final = frozenset(
    {"handoff_execution", "bf16_handoff_execution", "bf16_handoff_manifest"}
)
_SOURCE_CLOSURE_AUTHORIZATION_ISSUER = object()
_SOURCE_CLOSURE_REQUEST_AUTHORIZATION_ISSUER = object()
_SOURCE_CLOSURE_SUBMISSION_AUTHORIZATION_ISSUER = object()


@dataclass(frozen=True, slots=True)
class PublicationLatencySourceClosureCoordinatorConfig:
    """Immutable single-node CPU topology for the mounted source verifier."""

    runner_python_file: str
    package_wheel_uri: str
    package_wheel_sha256: str
    runtime_lock_uri: str
    runtime_lock_sha256: str
    patched_vllm_wheel_uri: str
    patched_vllm_wheel_sha256: str
    patched_flashinfer_wheel_uri: str
    patched_flashinfer_wheel_sha256: str
    runtime_closure_manifest_uri: str
    runtime_closure_manifest_sha256: str
    request_root_uri: str
    result_root_uri: str
    single_user_name: str
    runner_sha256: str = PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SHA256
    node_type_id: str = PUBLICATION_LATENCY_SOURCE_CLOSURE_NODE_TYPE_ID
    spark_version: str = PUBLICATION_LATENCY_SOURCE_CLOSURE_SPARK_VERSION
    data_security_mode: str = "SINGLE_USER"
    timeout_seconds: int = PUBLICATION_LATENCY_SOURCE_CLOSURE_TIMEOUT_SECONDS
    runtime_venv_dir: str = "/local_disk0/cachet-latency-source-closure-runtime"
    custom_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name, label in (
            ("runner_python_file", "source-closure runner URI"),
            ("package_wheel_uri", "source-closure package wheel URI"),
            ("runtime_lock_uri", "source-closure runtime lock URI"),
            ("patched_vllm_wheel_uri", "source-closure patched vLLM wheel URI"),
            (
                "patched_flashinfer_wheel_uri",
                "source-closure patched FlashInfer wheel URI",
            ),
            (
                "runtime_closure_manifest_uri",
                "source-closure runtime closure manifest URI",
            ),
            ("request_root_uri", "source-closure request root URI"),
            ("result_root_uri", "source-closure result root URI"),
        ):
            object.__setattr__(
                self,
                field_name,
                _databricks_volume_uri(getattr(self, field_name), label),
            )
        for field_name in (
            "package_wheel_sha256",
            "runtime_lock_sha256",
            "patched_vllm_wheel_sha256",
            "patched_flashinfer_wheel_sha256",
            "runtime_closure_manifest_sha256",
        ):
            _require_sha256_value(getattr(self, field_name), field_name)
        fixed_runtime = {
            "runtime_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
            "patched_vllm_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
            "patched_flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
            "runtime_closure_manifest_sha256": (
                RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
            ),
        }
        for field_name, expected in fixed_runtime.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"source-closure {field_name} authority drift")
        if self.runner_sha256 != PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SHA256:
            raise ValueError("source-closure runner SHA-256 drift")
        if self.node_type_id != PUBLICATION_LATENCY_SOURCE_CLOSURE_NODE_TYPE_ID:
            raise ValueError("source closure must use c5d.4xlarge")
        if self.spark_version != PUBLICATION_LATENCY_SOURCE_CLOSURE_SPARK_VERSION:
            raise ValueError("source-closure Databricks Runtime drift")
        if self.data_security_mode != "SINGLE_USER":
            raise ValueError("source closure requires SINGLE_USER mode")
        object.__setattr__(
            self,
            "single_user_name",
            _validated_single_user_name(self.single_user_name),
        )
        if self.timeout_seconds != PUBLICATION_LATENCY_SOURCE_CLOSURE_TIMEOUT_SECONDS:
            raise ValueError("source-closure timeout is frozen to two hours")
        runtime_root = PurePosixPath(self.runtime_venv_dir)
        if (
            not self.runtime_venv_dir.startswith("/local_disk0/")
            or runtime_root.as_posix() != self.runtime_venv_dir
            or any(part in {"", ".", ".."} for part in runtime_root.parts[1:])
        ):
            raise ValueError(
                "source-closure runtime venv must be canonical under /local_disk0"
            )
        if not isinstance(self.custom_tags, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in self.custom_tags.items()
        ):
            raise ValueError("source-closure custom tags must be non-empty strings")
        object.__setattr__(self, "custom_tags", MappingProxyType(dict(self.custom_tags)))

    def to_record(self) -> dict[str, Any]:
        return {
            "custom_tags": dict(self.custom_tags),
            "data_security_mode": self.data_security_mode,
            "node_type_id": self.node_type_id,
            "package_wheel_sha256": self.package_wheel_sha256,
            "package_wheel_uri": self.package_wheel_uri,
            "patched_vllm_wheel_sha256": self.patched_vllm_wheel_sha256,
            "patched_vllm_wheel_uri": self.patched_vllm_wheel_uri,
            "patched_flashinfer_wheel_sha256": (
                self.patched_flashinfer_wheel_sha256
            ),
            "patched_flashinfer_wheel_uri": self.patched_flashinfer_wheel_uri,
            "request_root_uri": self.request_root_uri,
            "result_root_uri": self.result_root_uri,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "runtime_lock_uri": self.runtime_lock_uri,
            "runtime_closure_manifest_sha256": (
                self.runtime_closure_manifest_sha256
            ),
            "runtime_closure_manifest_uri": self.runtime_closure_manifest_uri,
            "runtime_venv_dir": self.runtime_venv_dir,
            "runner_python_file": self.runner_python_file,
            "runner_sha256": self.runner_sha256,
            "single_user_name": self.single_user_name,
            "spark_version": self.spark_version,
            "timeout_seconds": self.timeout_seconds,
        }


def _source_closure_native_runtime_v2(
    config: PublicationLatencySourceClosureCoordinatorConfig,
) -> VLLMNativeRuntimeBundleV2:
    return VLLMNativeRuntimeBundleV2(
        runtime_lock_uri=config.runtime_lock_uri,
        runtime_lock_sha256=config.runtime_lock_sha256,
        patched_vllm_wheel_uri=config.patched_vllm_wheel_uri,
        patched_vllm_wheel_sha256=config.patched_vllm_wheel_sha256,
        patched_flashinfer_wheel_uri=config.patched_flashinfer_wheel_uri,
        patched_flashinfer_wheel_sha256=config.patched_flashinfer_wheel_sha256,
        runtime_closure_manifest_uri=config.runtime_closure_manifest_uri,
        runtime_closure_manifest_sha256=config.runtime_closure_manifest_sha256,
        package_wheel_uri=config.package_wheel_uri,
        package_wheel_sha256=config.package_wheel_sha256,
    )


def _source_closure_runtime_attestation_record(
    config: PublicationLatencySourceClosureCoordinatorConfig,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = _source_closure_native_runtime_v2(config)
    _validate_native_runtime_v2_attestation_binding(verification, bundle=bundle)
    artifacts = bundle.to_record()
    record = {
        "artifacts": artifacts,
        "artifacts_sha256": _canonical_sha256(artifacts),
        "runner_sha256": config.runner_sha256,
        "verification": dict(verification),
    }
    record["runtime_identity_sha256"] = _canonical_sha256(record)
    return record


@dataclass(frozen=True, slots=True, init=False)
class PublicationLatencySourceClosureRequestAuthorization:
    """Issuer-only authority over one builder-validated CPU closure request."""

    request_closed_record_sha256: str
    request_file_sha256: str
    authorization_record_sha256: str
    phase_lease_root: Path
    _request_json: str = field(repr=False)
    _authorization_json: str = field(repr=False)

    def __init__(
        self,
        *,
        request: Mapping[str, Any],
        phase_lease_root: str | Path,
        _issuer: object,
    ) -> None:
        if _issuer is not _SOURCE_CLOSURE_REQUEST_AUTHORIZATION_ISSUER:
            raise TypeError("source-closure request authority is issuer-only")
        validate_publication_latency_source_closure_request(request)
        normalized_request = cast(dict[str, Any], json.loads(_canonical_json(request)))
        authorization_record = _source_closure_request_authorization_record(
            normalized_request
        )
        request_closed_record_sha256 = _required_sha256(
            normalized_request, "closed_record_sha256"
        )
        request_file_sha256 = sha256(
            _pretty_json_bytes(normalized_request)
        ).hexdigest()
        if (
            authorization_record.get("request_closed_record_sha256")
            != request_closed_record_sha256
            or authorization_record.get("request_file_sha256")
            != request_file_sha256
        ):
            raise ValueError("source-closure request authority binding drift")
        object.__setattr__(
            self, "request_closed_record_sha256", request_closed_record_sha256
        )
        object.__setattr__(self, "request_file_sha256", request_file_sha256)
        object.__setattr__(
            self,
            "authorization_record_sha256",
            _required_sha256(authorization_record, "closed_record_sha256"),
        )
        object.__setattr__(self, "_request_json", _canonical_json(normalized_request))
        object.__setattr__(
            self, "_authorization_json", _canonical_json(authorization_record)
        )
        lease_root = Path(phase_lease_root).expanduser().absolute()
        _reject_existing_symlink_ancestors(
            lease_root, "source-closure phase lease"
        )
        object.__setattr__(self, "phase_lease_root", lease_root)

    @property
    def request_record(self) -> Mapping[str, Any]:
        """Return a fresh detached copy of the authorized request bytes."""

        return self.to_record()

    def to_record(self) -> dict[str, Any]:
        """Return the detached structural request for persistence or collection."""

        return cast(dict[str, Any], json.loads(self._request_json))

    @property
    def authorization_record(self) -> Mapping[str, Any]:
        """Return a fresh copy of the exact restart-safe authorization receipt."""

        return cast(dict[str, Any], json.loads(self._authorization_json))


@dataclass(frozen=True, slots=True, init=False)
class PublicationLatencySourceClosureSubmissionAuthorization:
    """Issuer-only authority over one idempotently submitted CPU verifier."""

    request_closed_record_sha256: str
    request_authorization_record_sha256: str
    ledger_path_sha256: str
    predecessor_prefix: DatabricksLedgerPrefix
    run_id: str
    submit_payload_sha256: str
    submit_response_sha256: str

    def __init__(
        self,
        *,
        request_closed_record_sha256: str,
        request_authorization_record_sha256: str,
        ledger_path_sha256: str,
        predecessor_prefix: DatabricksLedgerPrefix,
        run_id: str,
        submit_payload_sha256: str,
        submit_response_sha256: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _SOURCE_CLOSURE_SUBMISSION_AUTHORIZATION_ISSUER:
            raise TypeError("source-closure submission authority is issuer-only")
        object.__setattr__(
            self,
            "request_closed_record_sha256",
            _require_sha256_value(
                request_closed_record_sha256, "request_closed_record_sha256"
            ),
        )
        object.__setattr__(
            self,
            "request_authorization_record_sha256",
            _require_sha256_value(
                request_authorization_record_sha256,
                "request_authorization_record_sha256",
            ),
        )
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _require_sha256_value(ledger_path_sha256, "ledger_path_sha256"),
        )
        if not isinstance(predecessor_prefix, DatabricksLedgerPrefix):
            raise TypeError("source-closure predecessor prefix has the wrong type")
        object.__setattr__(self, "predecessor_prefix", predecessor_prefix)
        object.__setattr__(self, "run_id", _databricks_id(run_id, "source run ID"))
        object.__setattr__(
            self,
            "submit_payload_sha256",
            _require_sha256_value(submit_payload_sha256, "submit_payload_sha256"),
        )
        object.__setattr__(
            self,
            "submit_response_sha256",
            _require_sha256_value(submit_response_sha256, "submit_response_sha256"),
        )


@dataclass(frozen=True, slots=True, init=False)
class PublicationLatencySourceClosureAuthorization:
    """Issuer-only authority over remotely rehashed non-handoff inputs."""

    request_closed_record_sha256: str
    request_file_sha256: str
    result_uri: str
    result_file_sha256: str
    result_closed_record_sha256: str
    artifacts_sha256: str
    coordinator_run_id: str
    control_plane_status_sha256: str
    ledger_id: str
    ledger_path_sha256: str
    predecessor_prefix: DatabricksLedgerPrefix
    ledger_prefix: DatabricksLedgerPrefix
    causal_closure_sha256: str
    request_record: Mapping[str, Any]
    result_record: Mapping[str, Any]

    def __init__(
        self,
        *,
        request: Mapping[str, Any],
        result: Mapping[str, Any],
        result_file_sha256: str,
        coordinator_run_id: str,
        control_plane_status_sha256: str,
        ledger_prefix: DatabricksLedgerPrefix,
        _issuer: object,
    ) -> None:
        if _issuer is not _SOURCE_CLOSURE_AUTHORIZATION_ISSUER:
            raise TypeError("source-closure authority requires the collector issuer")
        validate_publication_latency_source_closure_request(request)
        validate_publication_latency_source_closure_result(result, request=request)
        run_id = _databricks_id(coordinator_run_id, "source coordinator run ID")
        control_sha = _require_sha256_value(
            control_plane_status_sha256, "control_plane_status_sha256"
        )
        result_sha = _require_sha256_value(result_file_sha256, "result_file_sha256")
        if result_sha != sha256(
            (_canonical_json(result) + "\n").encode("utf-8")
        ).hexdigest():
            raise ValueError("source-closure result file SHA-256 drift")
        if _mapping(result, "coordinator").get("run_id") != run_id:
            raise ValueError("source-closure result belongs to another run")
        lineage = _mapping(request, "ledger_lineage")
        predecessor = databricks_ledger_prefix_from_record(
            _mapping(lineage, "predecessor_prefix")
        )
        if not isinstance(ledger_prefix, DatabricksLedgerPrefix):
            raise TypeError("source-closure ledger prefixes have the wrong type")
        if ledger_prefix.ledger_id != predecessor.ledger_id:
            raise ValueError("source-closure ledger prefix identity drift")
        if ledger_prefix != predecessor:
            raise ValueError("source closure must not mutate the GPU campaign ledger")
        request_file_sha = sha256(_pretty_json_bytes(request)).hexdigest()
        causal = _canonical_sha256(
            {
                "control_plane_status_sha256": control_sha,
                "coordinator_run_id": run_id,
                "ledger_prefix": ledger_prefix.to_record(),
                "request_closed_record_sha256": request["closed_record_sha256"],
                "request_file_sha256": request_file_sha,
                "result_closed_record_sha256": result["closed_record_sha256"],
                "result_file_sha256": result_sha,
            }
        )
        normalized_request = json.loads(_canonical_json(request))
        normalized_result = json.loads(_canonical_json(result))
        object.__setattr__(
            self, "request_closed_record_sha256", request["closed_record_sha256"]
        )
        object.__setattr__(self, "request_file_sha256", request_file_sha)
        object.__setattr__(self, "result_uri", _required_string(request, "result_uri"))
        object.__setattr__(self, "result_file_sha256", result_sha)
        object.__setattr__(
            self, "result_closed_record_sha256", result["closed_record_sha256"]
        )
        object.__setattr__(
            self, "artifacts_sha256", _required_sha256(result, "artifacts_sha256")
        )
        object.__setattr__(self, "coordinator_run_id", run_id)
        object.__setattr__(self, "control_plane_status_sha256", control_sha)
        object.__setattr__(self, "ledger_id", _required_string(lineage, "ledger_id"))
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _required_sha256(lineage, "ledger_path_sha256"),
        )
        object.__setattr__(self, "predecessor_prefix", predecessor)
        object.__setattr__(self, "ledger_prefix", ledger_prefix)
        object.__setattr__(self, "causal_closure_sha256", causal)
        object.__setattr__(
            self, "request_record", MappingProxyType(normalized_request)
        )
        object.__setattr__(self, "result_record", MappingProxyType(normalized_result))


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
        "patched_flashinfer_wheel",
        "runtime_closure_manifest",
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


def _gpu_qualification_artifact_pins_v2_from_record(
    value: Mapping[str, Any],
) -> GPUQualificationArtifactPinsV2:
    _require_exact_keys(
        value,
        set(GPU_QUALIFICATION_V2_ARTIFACT_KEYS),
        "native-v2 GPU qualification artifact pins",
    )
    return GPUQualificationArtifactPinsV2(**dict(value))


def _require_native_v2_final_runtime_artifacts(
    final_artifacts: PublicationLatencyFinalArtifactPins,
    pins: GPUQualificationArtifactPinsV2,
) -> None:
    if not isinstance(final_artifacts, PublicationLatencyFinalArtifactPins):
        raise TypeError("final_artifacts has the wrong type")
    if not isinstance(pins, GPUQualificationArtifactPinsV2):
        raise TypeError("qualification artifact pins must be native v2")
    _require_distinct_qualification_runner(pins)
    expected = {
        "package_wheel": pins.package_wheel_sha256,
        "patched_vllm_wheel": pins.patched_vllm_wheel_sha256,
        "patched_flashinfer_wheel": pins.patched_flashinfer_wheel_sha256,
        "runtime_closure_manifest": pins.runtime_closure_manifest_sha256,
        "runtime_lock": pins.runtime_lock_sha256,
    }
    for role, expected_sha256 in expected.items():
        if final_artifacts.file(role).sha256 != expected_sha256:
            raise ValueError(f"final {role} differs from native-v2 qualification")


def _require_distinct_qualification_runner(
    pins: GPUQualificationArtifactPinsV2,
) -> None:
    if pins.runner_sha256 in {
        PUBLICATION_LATENCY_RUNNER_SHA256,
        PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SHA256,
    }:
        raise ValueError(
            "GPU qualification and publication latency runner identities must be distinct"
        )


def build_publication_latency_source_closure_request(
    *,
    attempt_id: str | None = None,
    coordinator_config: PublicationLatencySourceClosureCoordinatorConfig,
    campaign_plan_record: Mapping[str, Any],
    schedule_records: Mapping[int, Mapping[str, Any]],
    storage_schedule_records: Mapping[int, Mapping[str, Any]],
    qualification_plan_record: Mapping[str, Any],
    qualification_evidence_record: Mapping[str, Any],
    qualification_artifact_pins: GPUQualificationArtifactPinsV2,
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    storage_inputs_record: Mapping[str, Any],
    final_artifacts: PublicationLatencyFinalArtifactPins,
    ledger_path: str | Path,
) -> PublicationLatencySourceClosureRequestAuthorization:
    """Close one governed CPU request for every non-handoff source byte."""

    if not isinstance(
        coordinator_config, PublicationLatencySourceClosureCoordinatorConfig
    ):
        raise TypeError("coordinator_config has the wrong type")
    if not isinstance(qualification_artifact_pins, GPUQualificationArtifactPinsV2):
        raise TypeError("qualification_artifact_pins must be native v2")
    if not isinstance(final_artifacts, PublicationLatencyFinalArtifactPins):
        raise TypeError("final_artifacts has the wrong type")
    if final_artifacts.file("runner").sha256 != PUBLICATION_LATENCY_RUNNER_SHA256:
        raise ValueError("source closure requires the reviewed latency runner")
    _require_native_v2_final_runtime_artifacts(
        final_artifacts, qualification_artifact_pins
    )
    validate_publication_campaign_plan_record(campaign_plan_record)
    selection = validate_gpu_qualification_evidence_v2_record(
        qualification_evidence_record,
        plan_record=qualification_plan_record,
        expected_campaign_id=_required_string(campaign_plan_record, "campaign_id"),
        expected_artifact_pins=qualification_artifact_pins,
    )
    _require_reviewed_qualification_plan_campaign_binding(
        campaign_plan_record, qualification_plan_record
    )
    if coordinator_config.package_wheel_sha256 != final_artifacts.file(
        "package_wheel"
    ).sha256 or not _same_durable_file_location(
        coordinator_config.package_wheel_uri,
        final_artifacts.file("package_wheel").uri,
    ):
        raise ValueError("source coordinator package wheel differs from final artifacts")
    if (
        coordinator_config.runtime_lock_sha256
        != final_artifacts.file("runtime_lock").sha256
        or not _same_durable_file_location(
            coordinator_config.runtime_lock_uri,
            final_artifacts.file("runtime_lock").uri,
        )
        or coordinator_config.patched_vllm_wheel_sha256
        != final_artifacts.file("patched_vllm_wheel").sha256
        or not _same_durable_file_location(
            coordinator_config.patched_vllm_wheel_uri,
            final_artifacts.file("patched_vllm_wheel").uri,
        )
        or coordinator_config.patched_flashinfer_wheel_sha256
        != final_artifacts.file("patched_flashinfer_wheel").sha256
        or not _same_durable_file_location(
            coordinator_config.patched_flashinfer_wheel_uri,
            final_artifacts.file("patched_flashinfer_wheel").uri,
        )
        or coordinator_config.runtime_closure_manifest_sha256
        != final_artifacts.file("runtime_closure_manifest").sha256
        or not _same_durable_file_location(
            coordinator_config.runtime_closure_manifest_uri,
            final_artifacts.file("runtime_closure_manifest").uri,
        )
    ):
        raise ValueError("source coordinator locked runtime differs from final artifacts")
    qualification_closed = _required_sha256(
        qualification_evidence_record, "closed_record_sha256"
    )
    q8 = require_q8_handoff_remote_closure_authorization(
        handoff_serving_authorization,
        expected_output_root_uri=final_artifacts.handoff_generation_root_uri,
        expected_execution_file_sha256=final_artifacts.file(
            "handoff_execution"
        ).sha256,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        expected_qualification_closed_record_sha256=qualification_closed,
    )
    bf16 = require_bf16_handoff_remote_closure_authorization(
        bf16_handoff_serving_authorization,
        expected_output_root_uri=final_artifacts.bf16_handoff_generation_root_uri,
        expected_execution_file_sha256=final_artifacts.file(
            "bf16_handoff_execution"
        ).sha256,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        expected_qualification_closed_record_sha256=qualification_closed,
    )
    _require_handoff_closure_runtime_pins(
        q8,
        expected_final_artifacts=final_artifacts,
        expected_qualification_artifact_pins=qualification_artifact_pins,
    )
    _require_handoff_closure_runtime_pins(
        bf16,
        expected_final_artifacts=final_artifacts,
        expected_qualification_artifact_pins=qualification_artifact_pins,
    )
    if bf16.predecessor_prefix != q8.ledger_prefix:
        raise ValueError("source closure requires BF16 to extend Q8")
    ledger_file = Path(ledger_path).expanduser().absolute()
    ledger_path_sha256 = databricks_ledger_path_sha256(ledger_file)
    if ledger_path_sha256 != bf16.ledger_path_sha256:
        raise ValueError("source closure ledger path differs from BF16")
    ledger = read_databricks_cluster_hour_ledger_json(ledger_file)
    require_databricks_ledger_prefix(ledger, bf16.ledger_prefix)
    if databricks_ledger_prefix(ledger) != bf16.ledger_prefix:
        raise ValueError("BF16 is not the complete live source-closure predecessor")
    _require_reviewed_qualification_plan_campaign_successor(
        ledger,
        ledger_path=ledger_file,
        campaign_plan_record=campaign_plan_record,
        qualification_plan_record=qualification_plan_record,
    )

    schedule_bindings = _source_closure_schedule_bindings(
        schedule_records,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        final_artifacts=final_artifacts,
        storage=False,
    )
    source_examples = _schedule_examples(schedule_records[1])
    if any(
        _schedule_examples(schedule_records[block]) != source_examples
        for block in range(2, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)
    ):
        raise ValueError("source-closure main schedules have different identities")
    storage_schedule_bindings = _source_closure_schedule_bindings(
        storage_schedule_records,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        final_artifacts=final_artifacts,
        storage=True,
        source_examples=source_examples,
    )
    _validate_remote_publication_storage_inputs_record(
        storage_inputs_record,
        source_examples=source_examples,
        schedule_records=storage_schedule_records,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        final_artifacts=final_artifacts,
    )
    record_bindings = {
        "campaign": _source_closure_record_binding(
            final_artifacts.file("campaign_plan"), campaign_plan_record
        ),
        "qualification_evidence": _source_closure_record_binding(
            final_artifacts.file("qualification_evidence"),
            qualification_evidence_record,
        ),
        "qualification_plan": _source_closure_record_binding(
            final_artifacts.file("qualification_plan"), qualification_plan_record
        ),
        "schedules": schedule_bindings,
        "storage_inputs": _source_closure_record_binding(
            final_artifacts.file("storage_inputs"), storage_inputs_record
        ),
        "storage_schedules": storage_schedule_bindings,
    }
    expected_semantic = _source_closure_expected_semantic(
        campaign_plan_record=campaign_plan_record,
        qualification_plan_record=qualification_plan_record,
        qualification_evidence_record=qualification_evidence_record,
        selection=selection,
        schedule_bindings=schedule_bindings,
        storage_schedule_bindings=storage_schedule_bindings,
        storage_inputs_record=storage_inputs_record,
    )
    handoff_closures = {
        "bf16": _remote_handoff_authorization_binding(bf16),
        "q8": _remote_handoff_authorization_binding(q8),
    }
    singleton_identity_sha256 = _canonical_sha256(
        {
            "artifacts": final_artifacts.to_record(),
            "domain": "cachet.publication.latency_source_closure.singleton.v2",
            "expected_semantic": expected_semantic,
            "handoff_closures": handoff_closures,
            "ledger_lineage": {
                "ledger_id": bf16.ledger_id,
                "ledger_path_sha256": ledger_path_sha256,
                "predecessor_prefix": bf16.ledger_prefix.to_record(),
            },
            "qualification_artifact_pins": qualification_artifact_pins.to_record(),
            "record_bindings": record_bindings,
        }
    )
    normalized_attempt_id = _publication_latency_source_closure_attempt_id(
        singleton_identity_sha256
    )
    if attempt_id is not None and _safe_id(
        attempt_id, "source-closure attempt ID"
    ) != normalized_attempt_id:
        raise ValueError("source-closure attempt_id differs from singleton identity")
    expected_request_root, expected_result_root = (
        publication_latency_source_closure_control_roots(
            q8.output_root_uri,
            bf16.output_root_uri,
        )
    )
    if (
        coordinator_config.request_root_uri != expected_request_root
        or coordinator_config.result_root_uri != expected_result_root
    ):
        raise ValueError(
            "source-closure control roots differ from authenticated handoff outputs"
        )
    request_uri = _join_durable_uri(expected_request_root, "request.json")
    result_uri = _join_durable_uri(expected_result_root, "result.json")
    record: dict[str, Any] = {
        "attempt_id": normalized_attempt_id,
        "closed_record_sha256": "",
        "coordinator": coordinator_config.to_record(),
        "expected_semantic": expected_semantic,
        "final_artifacts": final_artifacts.to_record(),
        "handoff_closures": handoff_closures,
        "input_bundle_sha256": qualification_artifact_pins.input_bundle_sha256,
        "ledger_lineage": {
            "ledger_id": bf16.ledger_id,
            "ledger_path_sha256": ledger_path_sha256,
            "predecessor_prefix": bf16.ledger_prefix.to_record(),
        },
        "qualification_artifact_pins": qualification_artifact_pins.to_record(),
        "record_bindings": record_bindings,
        "record_type": PUBLICATION_LATENCY_SOURCE_CLOSURE_REQUEST_RECORD_TYPE,
        "request_uri": request_uri,
        "result_uri": result_uri,
        "schema_version": PUBLICATION_LATENCY_SOURCE_CLOSURE_SCHEMA_VERSION,
        "singleton_identity_sha256": singleton_identity_sha256,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    validate_publication_latency_source_closure_request(record)
    return PublicationLatencySourceClosureRequestAuthorization(
        request=record,
        phase_lease_root=_source_closure_phase_lease_root(
            ledger_file, singleton_identity_sha256=singleton_identity_sha256
        ),
        _issuer=_SOURCE_CLOSURE_REQUEST_AUTHORIZATION_ISSUER,
    )


def validate_publication_latency_source_closure_request(
    record: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        record,
        {
            "attempt_id",
            "closed_record_sha256",
            "coordinator",
            "expected_semantic",
            "final_artifacts",
            "handoff_closures",
            "input_bundle_sha256",
            "ledger_lineage",
            "qualification_artifact_pins",
            "record_bindings",
            "record_type",
            "request_uri",
            "result_uri",
            "schema_version",
            "singleton_identity_sha256",
        },
        "publication latency source-closure request",
    )
    if (
        record.get("record_type")
        != PUBLICATION_LATENCY_SOURCE_CLOSURE_REQUEST_RECORD_TYPE
        or record.get("schema_version")
        != PUBLICATION_LATENCY_SOURCE_CLOSURE_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("publication latency source-closure request envelope drift")
    if len(_pretty_json_bytes(record)) > PUBLICATION_LATENCY_SOURCE_CLOSURE_MAX_BYTES:
        raise ValueError("publication latency source-closure request exceeds byte cap")
    singleton_identity_sha256 = _required_sha256(
        record, "singleton_identity_sha256"
    )
    if singleton_identity_sha256 != _source_closure_singleton_identity_from_request(
        record
    ):
        raise ValueError("source-closure singleton identity digest drift")
    if _safe_id(record.get("attempt_id"), "source-closure attempt ID") != (
        _publication_latency_source_closure_attempt_id(singleton_identity_sha256)
    ):
        raise ValueError("source-closure attempt identity drift")
    request_uri = _databricks_volume_uri(
        _required_string(record, "request_uri"), "source-closure request URI"
    )
    result_uri = _databricks_volume_uri(
        _required_string(record, "result_uri"), "source-closure result URI"
    )
    if request_uri == result_uri:
        raise ValueError("source-closure request/result URI collision")
    coordinator = _mapping(record, "coordinator")
    config = _source_closure_config_from_record(coordinator)
    closures = _mapping(record, "handoff_closures")
    q8_output = _required_string(_mapping(closures, "q8"), "output_root_uri")
    bf16_output = _required_string(_mapping(closures, "bf16"), "output_root_uri")
    expected_request_root, expected_result_root = (
        publication_latency_source_closure_control_roots(
            q8_output,
            bf16_output,
        )
    )
    if (
        config.request_root_uri != expected_request_root
        or config.result_root_uri != expected_result_root
        or request_uri != _join_durable_uri(expected_request_root, "request.json")
        or result_uri != _join_durable_uri(expected_result_root, "result.json")
    ):
        raise ValueError("source-closure control URI/root singleton drift")
    pins = _gpu_qualification_artifact_pins_v2_from_record(
        _mapping(record, "qualification_artifact_pins")
    )
    _require_distinct_qualification_runner(pins)
    if record.get("input_bundle_sha256") != pins.input_bundle_sha256:
        raise ValueError("source-closure input bundle pin drift")
    final_artifacts = _final_artifacts_from_record(
        _mapping(record, "final_artifacts")
    )
    if final_artifacts.file("runner").sha256 != PUBLICATION_LATENCY_RUNNER_SHA256:
        raise ValueError("source-closure reviewed runner artifact pin drift")
    _require_native_v2_final_runtime_artifacts(final_artifacts, pins)
    if (
        config.package_wheel_sha256 != final_artifacts.file("package_wheel").sha256
        or not _same_durable_file_location(
            config.package_wheel_uri,
            final_artifacts.file("package_wheel").uri,
        )
    ):
        raise ValueError("source-closure coordinator package binding drift")
    if (
        config.runtime_lock_sha256 != final_artifacts.file("runtime_lock").sha256
        or not _same_durable_file_location(
            config.runtime_lock_uri,
            final_artifacts.file("runtime_lock").uri,
        )
        or config.patched_vllm_wheel_sha256
        != final_artifacts.file("patched_vllm_wheel").sha256
        or not _same_durable_file_location(
            config.patched_vllm_wheel_uri,
            final_artifacts.file("patched_vllm_wheel").uri,
        )
        or config.patched_flashinfer_wheel_sha256
        != final_artifacts.file("patched_flashinfer_wheel").sha256
        or not _same_durable_file_location(
            config.patched_flashinfer_wheel_uri,
            final_artifacts.file("patched_flashinfer_wheel").uri,
        )
        or config.runtime_closure_manifest_sha256
        != final_artifacts.file("runtime_closure_manifest").sha256
        or not _same_durable_file_location(
            config.runtime_closure_manifest_uri,
            final_artifacts.file("runtime_closure_manifest").uri,
        )
    ):
        raise ValueError("source-closure coordinator locked-runtime binding drift")
    lineage = _mapping(record, "ledger_lineage")
    _require_exact_keys(
        lineage,
        {"ledger_id", "ledger_path_sha256", "predecessor_prefix"},
        "source-closure ledger lineage",
    )
    ledger_id = _required_string(lineage, "ledger_id")
    _required_sha256(lineage, "ledger_path_sha256")
    predecessor = databricks_ledger_prefix_from_record(
        _mapping(lineage, "predecessor_prefix")
    )
    if predecessor.ledger_id != ledger_id:
        raise ValueError("source-closure predecessor ledger identity drift")
    _require_exact_keys(closures, {"bf16", "q8"}, "source handoff closures")
    for stage in ("q8", "bf16"):
        binding = _mapping(closures, stage)
        _require_exact_keys(
            binding,
            {
                "control_plane_status_sha256",
                "coordinator_run_id",
                "execution_closed_record_sha256",
                "execution_file_sha256",
                "execution_uri",
                "output_root_uri",
                "request_closed_record_sha256",
                "result_closed_record_sha256",
                "result_file_sha256",
                "result_uri",
                "stage",
            },
            f"source {stage} handoff closure",
        )
        if binding.get("stage") != stage:
            raise ValueError("source handoff closure stage drift")
        for name in (
            "control_plane_status_sha256",
            "execution_closed_record_sha256",
            "execution_file_sha256",
            "request_closed_record_sha256",
            "result_closed_record_sha256",
            "result_file_sha256",
        ):
            _required_sha256(binding, name)
        _databricks_id(binding.get("coordinator_run_id"), "handoff coordinator run")
        _databricks_volume_uri(binding.get("execution_uri"), "handoff execution URI")
        _databricks_volume_uri(binding.get("output_root_uri"), "handoff output root URI")
        _databricks_volume_uri(binding.get("result_uri"), "handoff result URI")
        execution_artifact = final_artifacts.file(
            "handoff_execution" if stage == "q8" else "bf16_handoff_execution"
        )
        if (
            binding.get("execution_file_sha256") != execution_artifact.sha256
            or binding.get("execution_uri")
            != _databricks_volume_uri(
                execution_artifact.uri, f"source {stage} execution artifact URI"
            )
        ):
            raise ValueError(f"source {stage} handoff execution artifact drift")
    bindings = _mapping(record, "record_bindings")
    _require_exact_keys(
        bindings,
        {
            "campaign",
            "qualification_evidence",
            "qualification_plan",
            "schedules",
            "storage_inputs",
            "storage_schedules",
        },
        "source-closure record bindings",
    )
    for name in (
        "campaign",
        "qualification_evidence",
        "qualification_plan",
        "storage_inputs",
    ):
        _validate_source_closure_record_binding(
            _mapping(bindings, name), name, expected_kind="record"
        )
    for name, expected_kind in (
        ("schedules", "schedule"),
        ("storage_schedules", "storage_schedule"),
    ):
        values = _mapping_sequence(bindings, name)
        if len(values) != PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS:
            raise ValueError(f"source-closure {name} coverage drift")
        for block, binding in enumerate(values, 1):
            _validate_source_closure_record_binding(
                binding, name, expected_kind=expected_kind
            )
            if binding.get("deployment_block") != block:
                raise ValueError(f"source-closure {name} block order drift")
    expected_semantic = _mapping(record, "expected_semantic")
    _validate_source_closure_semantic(expected_semantic)
    _validate_source_closure_record_binding_closure(
        bindings,
        final_artifacts=final_artifacts,
        expected_semantic=expected_semantic,
    )


def _source_closure_request_authorization_record(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the builder-reviewed request into one durable restart receipt."""

    validate_publication_latency_source_closure_request(request)
    coordinator = _mapping(request, "coordinator")
    config = _source_closure_config_from_record(coordinator)
    final_artifacts_record = _mapping(request, "final_artifacts")
    final_artifacts = _final_artifacts_from_record(final_artifacts_record)
    qualification_pins = _mapping(request, "qualification_artifact_pins")
    package_artifact = final_artifacts.file("package_wheel")
    if (
        config.package_wheel_sha256 != package_artifact.sha256
        or not _same_durable_file_location(
            config.package_wheel_uri, package_artifact.uri
        )
        or qualification_pins.get("package_wheel_sha256")
        != package_artifact.sha256
    ):
        raise ValueError("source-closure authorized package binding drift")
    record: dict[str, Any] = {
        "attempt_id": _required_string(request, "attempt_id"),
        "closed_record_sha256": "",
        "coordinator_sha256": _canonical_sha256(coordinator),
        "final_artifacts_sha256": _canonical_sha256(final_artifacts_record),
        "ledger_lineage_sha256": _canonical_sha256(
            _mapping(request, "ledger_lineage")
        ),
        "package_wheel_sha256": package_artifact.sha256,
        "package_wheel_uri": config.package_wheel_uri,
        "qualification_artifact_pins_sha256": _canonical_sha256(
            qualification_pins
        ),
        "record_type": (
            PUBLICATION_LATENCY_SOURCE_CLOSURE_REQUEST_AUTHORIZATION_RECORD_TYPE
        ),
        "request_closed_record_sha256": _required_sha256(
            request, "closed_record_sha256"
        ),
        "request_file_sha256": sha256(_pretty_json_bytes(request)).hexdigest(),
        "request_uri": _required_string(request, "request_uri"),
        "schema_version": PUBLICATION_LATENCY_SOURCE_CLOSURE_SCHEMA_VERSION,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    _validate_source_closure_request_authorization_record(record)
    return record


def _validate_source_closure_request_authorization_record(
    record: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        record,
        {
            "attempt_id",
            "closed_record_sha256",
            "coordinator_sha256",
            "final_artifacts_sha256",
            "ledger_lineage_sha256",
            "package_wheel_sha256",
            "package_wheel_uri",
            "qualification_artifact_pins_sha256",
            "record_type",
            "request_closed_record_sha256",
            "request_file_sha256",
            "request_uri",
            "schema_version",
        },
        "source-closure request authorization",
    )
    if (
        record.get("record_type")
        != PUBLICATION_LATENCY_SOURCE_CLOSURE_REQUEST_AUTHORIZATION_RECORD_TYPE
        or record.get("schema_version")
        != PUBLICATION_LATENCY_SOURCE_CLOSURE_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("source-closure request authorization envelope drift")
    _safe_id(record.get("attempt_id"), "source-closure authorized attempt ID")
    for field_name in (
        "coordinator_sha256",
        "final_artifacts_sha256",
        "ledger_lineage_sha256",
        "package_wheel_sha256",
        "qualification_artifact_pins_sha256",
        "request_closed_record_sha256",
        "request_file_sha256",
    ):
        _required_sha256(record, field_name)
    _databricks_volume_uri(
        record.get("package_wheel_uri"), "authorized package wheel URI"
    )
    _databricks_volume_uri(record.get("request_uri"), "authorized request URI")


def _require_source_closure_request_authorization(
    authorization: object,
) -> PublicationLatencySourceClosureRequestAuthorization:
    if not isinstance(
        authorization, PublicationLatencySourceClosureRequestAuthorization
    ):
        raise TypeError(
            "source closure requires "
            "PublicationLatencySourceClosureRequestAuthorization"
        )
    request = authorization.request_record
    expected = _source_closure_request_authorization_record(request)
    if (
        dict(authorization.authorization_record) != expected
        or authorization.authorization_record_sha256
        != expected["closed_record_sha256"]
        or authorization.request_closed_record_sha256
        != expected["request_closed_record_sha256"]
        or authorization.request_file_sha256 != expected["request_file_sha256"]
    ):
        raise ValueError("source-closure request authorization drift")
    return authorization


def validate_publication_latency_source_closure_result(
    record: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> None:
    validate_publication_latency_source_closure_request(request)
    _require_exact_keys(
        record,
        {
            "artifacts",
            "artifacts_sha256",
            "closed_record_sha256",
            "coordinator",
            "record_type",
            "request_closed_record_sha256",
            "runtime_attestation",
            "schema_version",
            "semantic",
        },
        "publication latency source-closure result",
    )
    if (
        record.get("record_type")
        != PUBLICATION_LATENCY_SOURCE_CLOSURE_RESULT_RECORD_TYPE
        or record.get("schema_version")
        != PUBLICATION_LATENCY_SOURCE_CLOSURE_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
        or record.get("request_closed_record_sha256")
        != request.get("closed_record_sha256")
    ):
        raise ValueError("publication latency source-closure result envelope drift")
    if len(_canonical_json_bytes(record)) > (
        PUBLICATION_LATENCY_SOURCE_CLOSURE_MAX_BYTES
    ):
        raise ValueError("publication latency source-closure result exceeds byte cap")
    coordinator = _mapping(record, "coordinator")
    _require_exact_keys(coordinator, {"run_id"}, "source-closure result coordinator")
    _databricks_id(coordinator.get("run_id"), "source-closure coordinator run ID")
    runtime_attestation = _mapping(record, "runtime_attestation")
    expected_runtime_attestation = _source_closure_runtime_attestation_record(
        _source_closure_config_from_record(_mapping(request, "coordinator")),
        _mapping(runtime_attestation, "verification"),
    )
    if dict(runtime_attestation) != expected_runtime_attestation:
        raise ValueError("source-closure native-v2 runtime attestation drift")
    expected_files = [
        item
        for item in _mapping_sequence(_mapping(request, "final_artifacts"), "files")
        if item.get("role") not in PUBLICATION_LATENCY_SOURCE_CLOSURE_EXCLUDED_ROLES
    ]
    artifacts = _mapping_sequence(record, "artifacts")
    if len(artifacts) != len(expected_files):
        raise ValueError("source-closure artifact coverage is incomplete")
    for artifact, expected in zip(artifacts, expected_files, strict=True):
        _require_exact_keys(
            artifact,
            {"byte_count", "role", "sha256", "uri"},
            "source-closure artifact",
        )
        if (
            artifact.get("role") != expected.get("role")
            or artifact.get("sha256") != expected.get("sha256")
            or artifact.get("uri") != expected.get("uri")
            or _positive_int(
                artifact.get("byte_count"), "source-closure artifact byte count"
            )
            <= 0
        ):
            raise ValueError("source-closure artifact binding drift")
    if record.get("artifacts_sha256") != _canonical_sha256(artifacts):
        raise ValueError("source-closure artifact inventory digest drift")
    semantic = _mapping(record, "semantic")
    _validate_source_closure_semantic(semantic)
    if dict(semantic) != dict(_mapping(request, "expected_semantic")):
        raise ValueError("source-closure semantic validation result drift")


def _source_closure_schedule_bindings(
    schedule_records: Mapping[int, Mapping[str, Any]],
    *,
    expected_input_bundle_sha256: str,
    final_artifacts: PublicationLatencyFinalArtifactPins,
    storage: bool,
    source_examples: Sequence[PublicationLatencyExample] | None = None,
) -> list[dict[str, Any]]:
    if set(schedule_records) != set(
        range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)
    ):
        raise ValueError("source closure requires exactly five schedule blocks")
    bindings: list[dict[str, Any]] = []
    for block in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1):
        schedule = schedule_records[block]
        if storage:
            if source_examples is None:
                raise ValueError("storage closure requires main source identities")
            validate_publication_storage_block_schedule(
                schedule,
                source_examples=source_examples,
                expected_input_bundle_sha256=expected_input_bundle_sha256,
            )
            role = f"storage_schedule_block_{block:02d}"
        else:
            validate_publication_latency_block_schedule(
                schedule,
                examples=_schedule_examples(schedule),
                expected_input_bundle_sha256=expected_input_bundle_sha256,
            )
            role = f"schedule_block_{block:02d}"
        if schedule.get("deployment_block") != block:
            raise ValueError("source-closure schedule deployment block drift")
        binding = _source_closure_record_binding(
            final_artifacts.file(role), schedule
        )
        binding.update(
            {
                "deployment_block": block,
                "requests_sha256": _required_sha256(schedule, "requests_sha256"),
                "seed_sha256": _required_sha256(schedule, "seed_sha256"),
            }
        )
        if storage:
            binding["selection_sha256"] = _required_sha256(
                _mapping(_mapping(schedule, "protocol"), "selection"),
                "selection_sha256",
            )
        bindings.append(binding)
    return bindings


def _source_closure_record_binding(
    artifact: PublicationLatencyArtifactFile,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if not _canonical_json_file_sha256_matches(record, artifact.sha256):
        raise ValueError(f"source-closure {artifact.role} file SHA-256 drift")
    return {
        "closed_record_sha256": _required_sha256(record, "closed_record_sha256"),
        "file_sha256": artifact.sha256,
        "role": artifact.role,
        "uri": _databricks_volume_uri(
            artifact.uri, f"source-closure {artifact.role} URI"
        ),
    }


def _validate_source_closure_record_binding(
    binding: Mapping[str, Any],
    field_name: str,
    *,
    expected_kind: str | None = None,
) -> None:
    base = {"closed_record_sha256", "file_sha256", "role", "uri"}
    schedule = {"deployment_block", "requests_sha256", "seed_sha256"}
    schemas = {
        "record": base,
        "schedule": base | schedule,
        "storage_schedule": base | schedule | {"selection_sha256"},
    }
    if expected_kind is not None and expected_kind not in schemas:
        raise ValueError("source-closure binding validator kind drift")
    allowed = (schemas[expected_kind],) if expected_kind else tuple(schemas.values())
    if set(binding) not in allowed:
        raise ValueError(f"source-closure {field_name} binding schema drift")
    _safe_id(binding.get("role"), f"source-closure {field_name} role")
    _databricks_volume_uri(
        binding.get("uri"), f"source-closure {field_name} URI"
    )
    _required_sha256(binding, "closed_record_sha256")
    _required_sha256(binding, "file_sha256")
    if "deployment_block" in binding:
        block = _required_int(binding, "deployment_block")
        if not 1 <= block <= PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS:
            raise ValueError("source-closure schedule block is outside the design")
        _required_sha256(binding, "requests_sha256")
        _required_sha256(binding, "seed_sha256")
    if "selection_sha256" in binding:
        _required_sha256(binding, "selection_sha256")


def _validate_source_closure_record_binding_closure(
    bindings: Mapping[str, Any],
    *,
    final_artifacts: PublicationLatencyFinalArtifactPins,
    expected_semantic: Mapping[str, Any],
) -> None:
    named_roles = {
        "campaign": "campaign_plan",
        "qualification_evidence": "qualification_evidence",
        "qualification_plan": "qualification_plan",
        "storage_inputs": "storage_inputs",
    }
    semantic_digests = {
        "campaign": "campaign_closed_record_sha256",
        "qualification_evidence": (
            "qualification_evidence_closed_record_sha256"
        ),
        "qualification_plan": "qualification_plan_closed_record_sha256",
    }
    for name, role in named_roles.items():
        binding = _mapping(bindings, name)
        _require_source_closure_binding_artifact(
            binding, final_artifacts=final_artifacts, expected_role=role
        )
        if name == "storage_inputs":
            expected_closed = _required_sha256(
                _mapping(expected_semantic, "storage_inputs"),
                "closed_record_sha256",
            )
        else:
            expected_closed = _required_sha256(
                expected_semantic, semantic_digests[name]
            )
        if binding.get("closed_record_sha256") != expected_closed:
            raise ValueError(f"source-closure {name} semantic binding drift")

    for name, role_prefix in (
        ("schedules", "schedule_block"),
        ("storage_schedules", "storage_schedule_block"),
    ):
        rows = _mapping_sequence(bindings, name)
        semantic_rows = _mapping_sequence(expected_semantic, name)
        for block, (binding, semantic) in enumerate(
            zip(rows, semantic_rows, strict=True), 1
        ):
            _require_source_closure_binding_artifact(
                binding,
                final_artifacts=final_artifacts,
                expected_role=f"{role_prefix}_{block:02d}",
            )
            for digest_name in (
                "closed_record_sha256",
                "requests_sha256",
                "seed_sha256",
            ):
                if binding.get(digest_name) != semantic.get(digest_name):
                    raise ValueError(
                        f"source-closure {name} semantic binding drift"
                    )
            if name == "storage_schedules" and binding.get(
                "selection_sha256"
            ) != semantic.get("selection_sha256"):
                raise ValueError("source-closure storage selection binding drift")


def _require_source_closure_binding_artifact(
    binding: Mapping[str, Any],
    *,
    final_artifacts: PublicationLatencyFinalArtifactPins,
    expected_role: str,
) -> None:
    artifact = final_artifacts.file(expected_role)
    if (
        binding.get("role") != expected_role
        or binding.get("file_sha256") != artifact.sha256
        or binding.get("uri")
        != _databricks_volume_uri(
            artifact.uri, f"source-closure {expected_role} artifact URI"
        )
    ):
        raise ValueError(f"source-closure {expected_role} artifact binding drift")


def _source_closure_expected_semantic(
    *,
    campaign_plan_record: Mapping[str, Any],
    qualification_plan_record: Mapping[str, Any],
    qualification_evidence_record: Mapping[str, Any],
    selection: GPUQualificationSelection,
    schedule_bindings: Sequence[Mapping[str, Any]],
    storage_schedule_bindings: Sequence[Mapping[str, Any]],
    storage_inputs_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "campaign_closed_record_sha256": _required_sha256(
            campaign_plan_record, "closed_record_sha256"
        ),
        "qualification_evidence_closed_record_sha256": _required_sha256(
            qualification_evidence_record, "closed_record_sha256"
        ),
        "qualification_plan_closed_record_sha256": _required_sha256(
            qualification_plan_record, "closed_record_sha256"
        ),
        "qualification_selection": _selection_record(selection),
        "schedules": [
            {
                "closed_record_sha256": item["closed_record_sha256"],
                "deployment_block": item["deployment_block"],
                "requests_sha256": item["requests_sha256"],
                "seed_sha256": item["seed_sha256"],
            }
            for item in schedule_bindings
        ],
        "storage_inputs": {
            "closed_record_sha256": _required_sha256(
                storage_inputs_record, "closed_record_sha256"
            ),
            "selection_sha256": _required_sha256(
                _mapping(storage_inputs_record, "selection_protocol"),
                "selection_sha256",
            ),
        },
        "storage_schedules": [
            {
                "closed_record_sha256": item["closed_record_sha256"],
                "deployment_block": item["deployment_block"],
                "requests_sha256": item["requests_sha256"],
                "seed_sha256": item["seed_sha256"],
                "selection_sha256": item["selection_sha256"],
            }
            for item in storage_schedule_bindings
        ],
    }


def _validate_source_closure_semantic(value: Mapping[str, Any]) -> None:
    _require_exact_keys(
        value,
        {
            "campaign_closed_record_sha256",
            "qualification_evidence_closed_record_sha256",
            "qualification_plan_closed_record_sha256",
            "qualification_selection",
            "schedules",
            "storage_inputs",
            "storage_schedules",
        },
        "source-closure semantic record",
    )
    for name in (
        "campaign_closed_record_sha256",
        "qualification_evidence_closed_record_sha256",
        "qualification_plan_closed_record_sha256",
    ):
        _required_sha256(value, name)
    selection = _mapping(value, "qualification_selection")
    _require_exact_keys(
        selection,
        {
            "attention_backend",
            "generation_artifacts_sha256",
            "generation_databricks_node_type_id",
            "generation_hardware_id",
            "generation_prefix_tokens_per_second",
            "gpu_memory_utilization",
            "plan_sha256",
        },
        "source-closure qualification selection",
    )
    _required_sha256(selection, "generation_artifacts_sha256")
    _required_sha256(selection, "plan_sha256")
    for name, storage in (("schedules", False), ("storage_schedules", True)):
        rows = _mapping_sequence(value, name)
        if len(rows) != PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS:
            raise ValueError(f"source-closure {name} semantic coverage drift")
        for block, row in enumerate(rows, 1):
            expected_keys = {
                "closed_record_sha256",
                "deployment_block",
                "requests_sha256",
                "seed_sha256",
            }
            if storage:
                expected_keys.add("selection_sha256")
            _require_exact_keys(row, expected_keys, f"source-closure {name} row")
            if row.get("deployment_block") != block:
                raise ValueError(f"source-closure {name} block order drift")
            for digest in expected_keys.difference({"deployment_block"}):
                _required_sha256(row, digest)
    storage_inputs = _mapping(value, "storage_inputs")
    _require_exact_keys(
        storage_inputs,
        {"closed_record_sha256", "selection_sha256"},
        "source-closure storage inputs semantic",
    )
    _required_sha256(storage_inputs, "closed_record_sha256")
    _required_sha256(storage_inputs, "selection_sha256")


def _source_closure_semantic_from_plan_sources(
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    qualification = _mapping(sources, "qualification")
    return {
        "campaign_closed_record_sha256": _required_sha256(
            _mapping(sources, "campaign"), "closed_record_sha256"
        ),
        "qualification_evidence_closed_record_sha256": _required_sha256(
            _mapping(qualification, "evidence"), "closed_record_sha256"
        ),
        "qualification_plan_closed_record_sha256": _required_sha256(
            _mapping(qualification, "plan"), "closed_record_sha256"
        ),
        "qualification_selection": dict(_mapping(qualification, "selection")),
        "schedules": [
            {
                "closed_record_sha256": _required_sha256(
                    item, "closed_record_sha256"
                ),
                "deployment_block": _required_int(item, "deployment_block"),
                "requests_sha256": _required_sha256(item, "requests_sha256"),
                "seed_sha256": _required_sha256(item, "seed_sha256"),
            }
            for item in _mapping_sequence(sources, "schedules")
        ],
        "storage_inputs": {
            "closed_record_sha256": _required_sha256(
                _mapping(sources, "storage_inputs"), "closed_record_sha256"
            ),
            "selection_sha256": _required_sha256(
                _mapping(sources, "storage_inputs"), "selection_sha256"
            ),
        },
        "storage_schedules": [
            {
                "closed_record_sha256": _required_sha256(
                    item, "closed_record_sha256"
                ),
                "deployment_block": _required_int(item, "deployment_block"),
                "requests_sha256": _required_sha256(item, "requests_sha256"),
                "seed_sha256": _required_sha256(item, "seed_sha256"),
                "selection_sha256": _required_sha256(item, "selection_sha256"),
            }
            for item in _mapping_sequence(sources, "storage_schedules")
        ],
    }


def _source_closure_config_from_record(
    record: Mapping[str, Any],
) -> PublicationLatencySourceClosureCoordinatorConfig:
    _require_exact_keys(
        record,
        {
            "custom_tags",
            "data_security_mode",
            "node_type_id",
            "package_wheel_sha256",
            "package_wheel_uri",
            "patched_flashinfer_wheel_sha256",
            "patched_flashinfer_wheel_uri",
            "patched_vllm_wheel_sha256",
            "patched_vllm_wheel_uri",
            "request_root_uri",
            "result_root_uri",
            "runner_python_file",
            "runner_sha256",
            "runtime_lock_sha256",
            "runtime_lock_uri",
            "runtime_closure_manifest_sha256",
            "runtime_closure_manifest_uri",
            "runtime_venv_dir",
            "single_user_name",
            "spark_version",
            "timeout_seconds",
        },
        "source-closure coordinator config",
    )
    tags = _mapping(record, "custom_tags")
    config = PublicationLatencySourceClosureCoordinatorConfig(
        runner_python_file=_required_string(record, "runner_python_file"),
        package_wheel_uri=_required_string(record, "package_wheel_uri"),
        package_wheel_sha256=_required_sha256(record, "package_wheel_sha256"),
        runtime_lock_uri=_required_string(record, "runtime_lock_uri"),
        runtime_lock_sha256=_required_sha256(record, "runtime_lock_sha256"),
        patched_vllm_wheel_uri=_required_string(
            record, "patched_vllm_wheel_uri"
        ),
        patched_vllm_wheel_sha256=_required_sha256(
            record, "patched_vllm_wheel_sha256"
        ),
        patched_flashinfer_wheel_uri=_required_string(
            record, "patched_flashinfer_wheel_uri"
        ),
        patched_flashinfer_wheel_sha256=_required_sha256(
            record, "patched_flashinfer_wheel_sha256"
        ),
        runtime_closure_manifest_uri=_required_string(
            record, "runtime_closure_manifest_uri"
        ),
        runtime_closure_manifest_sha256=_required_sha256(
            record, "runtime_closure_manifest_sha256"
        ),
        request_root_uri=_required_string(record, "request_root_uri"),
        result_root_uri=_required_string(record, "result_root_uri"),
        single_user_name=_validated_single_user_name(
            record.get("single_user_name")
        ),
        runner_sha256=_required_sha256(record, "runner_sha256"),
        node_type_id=_required_string(record, "node_type_id"),
        spark_version=_required_string(record, "spark_version"),
        data_security_mode=_required_string(record, "data_security_mode"),
        timeout_seconds=_required_int(record, "timeout_seconds"),
        runtime_venv_dir=_required_string(record, "runtime_venv_dir"),
        custom_tags=cast(Mapping[str, str], tags),
    )
    if config.to_record() != dict(record):
        raise ValueError("source-closure coordinator config normalization drift")
    return config


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


def write_publication_latency_source_closure_runner_script(
    path: str | Path,
) -> Path:
    """Write the reviewed source-closure bootstrap once."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"publication latency source-closure runner exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT, encoding="utf-8"
    )
    return destination


def render_publication_latency_source_closure_submit_payload(
    request_authorization: PublicationLatencySourceClosureRequestAuthorization,
) -> dict[str, Any]:
    """Render one idempotent, no-retry c5d CPU source verifier."""

    authorization = _require_source_closure_request_authorization(
        request_authorization
    )
    return _render_publication_latency_source_closure_submit_payload_from_request(
        authorization.request_record
    )


def _render_publication_latency_source_closure_submit_payload_from_request(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Render from bytes already authenticated at a typed controller boundary."""

    validate_publication_latency_source_closure_request(request)
    config = _source_closure_config_from_record(_mapping(request, "coordinator"))
    request_bytes = _pretty_json_bytes(request)
    parameters = [
        "--runner-sha256",
        config.runner_sha256,
        "--package-wheel-uri",
        config.package_wheel_uri,
        "--package-wheel-sha256",
        config.package_wheel_sha256,
        "--runtime-lock-uri",
        config.runtime_lock_uri,
        "--runtime-lock-sha256",
        config.runtime_lock_sha256,
        "--patched-vllm-wheel-uri",
        config.patched_vllm_wheel_uri,
        "--patched-vllm-wheel-sha256",
        config.patched_vllm_wheel_sha256,
        "--patched-flashinfer-wheel-uri",
        config.patched_flashinfer_wheel_uri,
        "--patched-flashinfer-wheel-sha256",
        config.patched_flashinfer_wheel_sha256,
        "--runtime-closure-manifest-uri",
        config.runtime_closure_manifest_uri,
        "--runtime-closure-manifest-sha256",
        config.runtime_closure_manifest_sha256,
        "--runtime-venv-dir",
        config.runtime_venv_dir,
        "--request-uri",
        _required_string(request, "request_uri"),
        "--request-file-sha256",
        sha256(request_bytes).hexdigest(),
        "--request-closed-record-sha256",
        _required_sha256(request, "closed_record_sha256"),
        "--coordinator-run-id",
        _DATABRICKS_JOB_RUN_ID_TEMPLATE,
    ]
    if len(_canonical_json(parameters).encode("utf-8")) > 9_500:
        raise ValueError("source-closure runner parameters exceed the compact cap")
    payload = {
        "run_name": "cachet-vllm-0271-publication-latency-source-closure",
        "tasks": [
            {
                "max_retries": 0,
                "new_cluster": {
                    "aws_attributes": {"availability": "ON_DEMAND", "zone_id": "auto"},
                    "custom_tags": {
                        **dict(config.custom_tags),
                        "ResourceClass": "SingleNode",
                        "campaign": "vllm-0271-publication-v1",
                        "purpose": "cachet-vllm-0271-latency-source-closure",
                        "request_sha256": request["closed_record_sha256"][:32],
                    },
                    "data_security_mode": config.data_security_mode,
                    "driver_node_type_id": config.node_type_id,
                    "node_type_id": config.node_type_id,
                    "num_workers": 0,
                    "single_user_name": config.single_user_name,
                    "spark_conf": {
                        "spark.databricks.cluster.profile": "singleNode",
                        "spark.master": "local[*]",
                    },
                    "spark_version": config.spark_version,
                },
                "spark_python_task": {
                    "parameters": parameters,
                    "python_file": config.runner_python_file,
                },
                "task_key": "publication_latency_source_closure",
                "timeout_seconds": config.timeout_seconds,
            }
        ],
        "timeout_seconds": config.timeout_seconds,
    }
    return bind_databricks_run_idempotency_token(
        payload, attempt_id=_required_string(request, "attempt_id")
    )


def run_publication_latency_source_closure_coordinator(
    request_path: str | Path,
    *,
    expected_request_file_sha256: str,
    expected_request_closed_record_sha256: str,
    coordinator_run_id: str,
) -> dict[str, Any]:
    """Rehash and semantically validate every non-handoff source on Volume."""

    path = Path(request_path)
    _verify_regular_file_sha256(
        path,
        _require_sha256_value(
            expected_request_file_sha256, "expected_request_file_sha256"
        ),
        "source-closure request",
    )
    request_bytes = path.read_bytes()
    request = _canonical_pretty_remote_json_record(
        request_bytes, field_name="source-closure request"
    )
    validate_publication_latency_source_closure_request(request)
    if request.get("closed_record_sha256") != _require_sha256_value(
        expected_request_closed_record_sha256,
        "expected_request_closed_record_sha256",
    ):
        raise ValueError("source-closure request closed digest drift")
    run_id = _databricks_id(coordinator_run_id, "source coordinator run ID")
    raw_runtime_attestation = os.environ.pop(
        "CACHET_LATENCY_SOURCE_CLOSURE_RUNTIME_ATTESTATION", None
    )
    if not isinstance(raw_runtime_attestation, str) or not raw_runtime_attestation:
        raise RuntimeError("source-closure native-v2 runtime attestation is missing")
    try:
        runtime_verification = json.loads(raw_runtime_attestation)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "source-closure native-v2 runtime attestation is invalid"
        ) from exc
    if (
        not isinstance(runtime_verification, dict)
        or raw_runtime_attestation != _canonical_json(runtime_verification)
    ):
        raise RuntimeError(
            "source-closure native-v2 runtime attestation is not canonical"
        )
    runtime_attestation = _source_closure_runtime_attestation_record(
        _source_closure_config_from_record(_mapping(request, "coordinator")),
        runtime_verification,
    )
    final_artifacts = _final_artifacts_from_record(
        _mapping(request, "final_artifacts")
    )
    artifact_records: list[dict[str, Any]] = []
    for artifact in final_artifacts.files:
        if artifact.role in PUBLICATION_LATENCY_SOURCE_CLOSURE_EXCLUDED_ROLES:
            continue
        artifact_path = _cluster_path(artifact.uri)
        _verify_regular_file_sha256(
            artifact_path, artifact.sha256, f"source artifact {artifact.role}"
        )
        byte_count = artifact_path.stat().st_size
        if byte_count <= 0:
            raise ValueError(f"source artifact {artifact.role} is empty")
        artifact_records.append(
            {
                "byte_count": byte_count,
                "role": artifact.role,
                "sha256": artifact.sha256,
                "uri": artifact.uri,
            }
        )

    bindings = _mapping(request, "record_bindings")
    campaign = _read_source_closure_bound_record(
        _mapping(bindings, "campaign"), "campaign"
    )
    qualification_plan = _read_source_closure_bound_record(
        _mapping(bindings, "qualification_plan"), "qualification plan"
    )
    qualification_evidence = _read_source_closure_bound_record(
        _mapping(bindings, "qualification_evidence"), "qualification evidence"
    )
    storage_inputs = _read_source_closure_bound_record(
        _mapping(bindings, "storage_inputs"), "storage inputs"
    )
    schedule_records = {
        _required_int(binding, "deployment_block"): _read_source_closure_bound_record(
            binding, "publication schedule"
        )
        for binding in _mapping_sequence(bindings, "schedules")
    }
    storage_schedule_records = {
        _required_int(binding, "deployment_block"): _read_source_closure_bound_record(
            binding, "publication storage schedule"
        )
        for binding in _mapping_sequence(bindings, "storage_schedules")
    }
    validate_publication_campaign_plan_record(campaign)
    pins = _gpu_qualification_artifact_pins_v2_from_record(
        _mapping(request, "qualification_artifact_pins")
    )
    selection = validate_gpu_qualification_evidence_v2_record(
        qualification_evidence,
        plan_record=qualification_plan,
        expected_campaign_id=_required_string(campaign, "campaign_id"),
        expected_artifact_pins=pins,
    )
    schedule_bindings = _source_closure_schedule_bindings(
        schedule_records,
        expected_input_bundle_sha256=pins.input_bundle_sha256,
        final_artifacts=final_artifacts,
        storage=False,
    )
    source_examples = _schedule_examples(schedule_records[1])
    if any(
        _schedule_examples(schedule_records[block]) != source_examples
        for block in range(2, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)
    ):
        raise ValueError("remotely validated schedules use different identities")
    storage_schedule_bindings = _source_closure_schedule_bindings(
        storage_schedule_records,
        expected_input_bundle_sha256=pins.input_bundle_sha256,
        final_artifacts=final_artifacts,
        storage=True,
        source_examples=source_examples,
    )
    storage_source_paths = {
        dataset: _cluster_path(final_artifacts.file(f"input_16384_{dataset}").uri)
        for dataset in SUPPORTED_V1_DATASETS
    }
    validate_publication_storage_inputs_record(
        storage_inputs,
        source_paths=storage_source_paths,
        schedule_records=storage_schedule_records,
        expected_input_bundle_sha256=pins.input_bundle_sha256,
    )
    observed_semantic = _source_closure_expected_semantic(
        campaign_plan_record=campaign,
        qualification_plan_record=qualification_plan,
        qualification_evidence_record=qualification_evidence,
        selection=selection,
        schedule_bindings=schedule_bindings,
        storage_schedule_bindings=storage_schedule_bindings,
        storage_inputs_record=storage_inputs,
    )
    if observed_semantic != dict(_mapping(request, "expected_semantic")):
        raise ValueError("source-closure mounted semantic result differs from request")
    result: dict[str, Any] = {
        "artifacts": artifact_records,
        "artifacts_sha256": _canonical_sha256(artifact_records),
        "closed_record_sha256": "",
        "coordinator": {"run_id": run_id},
        "record_type": PUBLICATION_LATENCY_SOURCE_CLOSURE_RESULT_RECORD_TYPE,
        "request_closed_record_sha256": request["closed_record_sha256"],
        "runtime_attestation": runtime_attestation,
        "schema_version": PUBLICATION_LATENCY_SOURCE_CLOSURE_SCHEMA_VERSION,
        "semantic": observed_semantic,
    }
    result["closed_record_sha256"] = _closed_record_sha256(result)
    validate_publication_latency_source_closure_result(result, request=request)
    result_path = _cluster_path(_required_string(request, "result_uri"))
    _write_or_require_exact_bytes(
        result_path, (_canonical_json(result) + "\n").encode("utf-8")
    )
    return result


def _read_source_closure_bound_record(
    binding: Mapping[str, Any], field_name: str
) -> dict[str, Any]:
    _validate_source_closure_record_binding(binding, field_name)
    path = _cluster_path(_required_string(binding, "uri"))
    _verify_regular_file_sha256(
        path, _required_sha256(binding, "file_sha256"), field_name
    )
    record = _read_json_file(path, field_name)
    if record.get("closed_record_sha256") != binding.get("closed_record_sha256"):
        raise ValueError(f"source-closure {field_name} closed digest drift")
    return record


def submit_publication_latency_source_closure(
    workspace: DatabricksWorkspaceConfig,
    *,
    request_authorization: PublicationLatencySourceClosureRequestAuthorization,
    ledger_path: str | Path,
    phase_lease_root: str | Path,
    opener: DatabricksURLOpener | None = None,
) -> tuple[dict[str, Any], PublicationLatencySourceClosureSubmissionAuthorization]:
    """Durably stage and idempotently submit one unmetered CPU verifier."""

    authorization = _require_source_closure_request_authorization(
        request_authorization
    )
    _require_source_closure_phase_lease_root(authorization, phase_lease_root)
    request = authorization.request_record
    if not isinstance(workspace, DatabricksWorkspaceConfig):
        raise TypeError("workspace has the wrong type")
    payload = render_publication_latency_source_closure_submit_payload(authorization)
    ledger_file = Path(ledger_path).expanduser().absolute()
    predecessor = databricks_ledger_prefix_from_record(
        _mapping(_mapping(request, "ledger_lineage"), "predecessor_prefix")
    )
    ledger_path_sha256 = databricks_ledger_path_sha256(ledger_file)
    if ledger_path_sha256 != _required_sha256(
        _mapping(request, "ledger_lineage"), "ledger_path_sha256"
    ):
        raise ValueError("source-closure ledger path binding drift")
    live = read_databricks_cluster_hour_ledger_json(ledger_file)
    require_databricks_ledger_prefix(live, predecessor)
    if databricks_ledger_prefix(live) != predecessor:
        raise ValueError("source-closure predecessor is not the complete live ledger")
    coordinator = _source_closure_config_from_record(_mapping(request, "coordinator"))
    require_databricks_current_user_name(
        workspace,
        expected_user_name=coordinator.single_user_name,
        opener=opener,
    )
    lease_root = _create_latency_phase_lease_root(phase_lease_root)
    _write_canonical_json_exclusive(lease_root / "request.json", request)
    _write_canonical_json_exclusive(
        lease_root / "request-authorization.json",
        authorization.authorization_record,
    )
    _write_canonical_json_exclusive(lease_root / "submit-payload.json", payload)
    return _submit_or_resume_publication_latency_source_closure(
        workspace,
        request_authorization=authorization,
        payload=payload,
        ledger_file=ledger_file,
        lease_root=lease_root,
        predecessor=predecessor,
        opener=opener,
    )


def resume_publication_latency_source_closure(
    workspace: DatabricksWorkspaceConfig,
    *,
    request_authorization: PublicationLatencySourceClosureRequestAuthorization,
    ledger_path: str | Path,
    phase_lease_root: str | Path,
    opener: DatabricksURLOpener | None = None,
) -> tuple[dict[str, Any], PublicationLatencySourceClosureSubmissionAuthorization]:
    """Recover any source-closure crash point with the identical wire body."""

    expected_authorization = _require_source_closure_request_authorization(
        request_authorization
    )
    _require_source_closure_phase_lease_root(
        expected_authorization, phase_lease_root
    )
    if not isinstance(workspace, DatabricksWorkspaceConfig):
        raise TypeError("workspace has the wrong type")
    lease_root = Path(phase_lease_root).expanduser().absolute()
    _reject_existing_symlink_ancestors(lease_root, "source-closure phase lease")
    if not lease_root.is_dir() or lease_root.is_symlink():
        raise ValueError("source-closure phase lease must be an existing directory")
    authorization_record = _read_latency_controller_record(
        lease_root / "request-authorization.json",
        "source-closure request authorization",
    )
    request = _read_latency_controller_record(
        lease_root / "request.json", "source-closure request"
    )
    if (
        request != expected_authorization.request_record
        or authorization_record
        != dict(expected_authorization.authorization_record)
    ):
        raise ValueError(
            "source-closure durable request differs from the replayed authority"
        )
    authorization = expected_authorization
    payload_path = lease_root / "submit-payload.json"
    payload = _read_json_file(payload_path, "source-closure submit payload")
    if payload_path.read_bytes() != _canonical_json_bytes(payload):
        raise ValueError("source-closure submit payload is not canonical JSON")
    if payload != render_publication_latency_source_closure_submit_payload(
        authorization
    ):
        raise ValueError("source-closure reserved submit payload drift")
    _require_source_closure_durable_authorization_bindings(
        lease_root, authorization
    )
    ledger_file = Path(ledger_path).expanduser().absolute()
    lineage = _mapping(request, "ledger_lineage")
    if databricks_ledger_path_sha256(ledger_file) != _required_sha256(
        lineage, "ledger_path_sha256"
    ):
        raise ValueError("source-closure resume ledger path drift")
    predecessor = databricks_ledger_prefix_from_record(
        _mapping(lineage, "predecessor_prefix")
    )
    _require_unchanged_source_closure_gpu_ledger(ledger_file, predecessor)
    coordinator = _source_closure_config_from_record(_mapping(request, "coordinator"))
    require_databricks_current_user_name(
        workspace,
        expected_user_name=coordinator.single_user_name,
        opener=opener,
    )
    return _submit_or_resume_publication_latency_source_closure(
        workspace,
        request_authorization=authorization,
        payload=payload,
        ledger_file=ledger_file,
        lease_root=lease_root,
        predecessor=predecessor,
        opener=opener,
    )


def _require_source_closure_durable_authorization_bindings(
    lease_root: Path,
    authorization: PublicationLatencySourceClosureRequestAuthorization,
) -> None:
    """Pin every already-durable crash receipt to the issuer's request authority."""

    expected = authorization.authorization_record_sha256
    for filename, label in (
        ("request-upload.json", "source-closure request upload"),
        ("post-intent.json", "source-closure post intent"),
        ("submit-response.json", "source-closure submit response"),
    ):
        path = lease_root / filename
        if not path.exists() and not path.is_symlink():
            continue
        record = _read_latency_controller_record(path, label)
        if record.get("request_authorization_record_sha256") != expected:
            raise ValueError(f"{label} request authorization binding drift")


def _require_source_closure_phase_lease_root(
    authorization: PublicationLatencySourceClosureRequestAuthorization,
    value: str | Path,
) -> Path:
    root = Path(value).expanduser().absolute()
    _reject_existing_symlink_ancestors(root, "source-closure phase lease")
    if root != authorization.phase_lease_root:
        raise ValueError(
            "source-closure phase_lease_root differs from singleton authority"
        )
    return root


def _submit_or_resume_publication_latency_source_closure(
    workspace: DatabricksWorkspaceConfig,
    *,
    request_authorization: PublicationLatencySourceClosureRequestAuthorization,
    payload: Mapping[str, Any],
    ledger_file: Path,
    lease_root: Path,
    predecessor: DatabricksLedgerPrefix,
    opener: DatabricksURLOpener | None,
) -> tuple[dict[str, Any], PublicationLatencySourceClosureSubmissionAuthorization]:
    authorization = _require_source_closure_request_authorization(
        request_authorization
    )
    request = authorization.request_record
    if payload != render_publication_latency_source_closure_submit_payload(
        authorization
    ):
        raise ValueError("source-closure submit payload/request authority drift")
    _require_unchanged_source_closure_gpu_ledger(ledger_file, predecessor)
    request_bytes = _pretty_json_bytes(request)
    upload = upload_databricks_volume_file_bytes_exclusive(
        workspace,
        _required_string(request, "request_uri"),
        request_bytes,
        max_bytes=PUBLICATION_LATENCY_SOURCE_CLOSURE_MAX_BYTES,
    )
    upload_record = {
        "closed_record_sha256": "",
        "dbfs_uri": upload.get("dbfs_uri"),
        "file_sha256": upload.get("file_sha256"),
        "record_type": "cachet.publication_latency_source_upload.v1",
        "request_authorization_record_sha256": (
            authorization.authorization_record_sha256
        ),
        "size_bytes": upload.get("size_bytes"),
    }
    if (
        upload.get("created") not in {True, False}
        or upload_record["dbfs_uri"] != request.get("request_uri")
        or upload_record["file_sha256"] != sha256(request_bytes).hexdigest()
        or upload_record["size_bytes"] != len(request_bytes)
    ):
        raise ValueError("source-closure request upload receipt drift")
    upload_record["closed_record_sha256"] = _closed_record_sha256(upload_record)
    _write_or_require_exact_bytes(
        lease_root / "request-upload.json", _canonical_json_bytes(upload_record)
    )
    _require_unchanged_source_closure_gpu_ledger(ledger_file, predecessor)
    snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(payload)
    payload_sha = sha256(canonical_payload).hexdigest()
    idempotency_token = require_databricks_run_idempotency_token(
        snapshot, attempt_id=_required_string(request, "attempt_id")
    )
    post_intent = {
        "closed_record_sha256": "",
        "gpu_ledger_path_sha256": databricks_ledger_path_sha256(ledger_file),
        "gpu_ledger_prefix": predecessor.to_record(),
        "idempotency_token": idempotency_token,
        "record_type": "cachet.publication_latency_source_post_intent.v1",
        "request_authorization_record_sha256": (
            authorization.authorization_record_sha256
        ),
        "request_file_byte_count": len(request_bytes),
        "request_file_sha256": sha256(request_bytes).hexdigest(),
        "submit_payload_byte_count": len(canonical_payload),
        "submit_payload_sha256": payload_sha,
    }
    post_intent["closed_record_sha256"] = _closed_record_sha256(post_intent)
    _write_or_require_exact_bytes(
        lease_root / "post-intent.json", _canonical_json_bytes(post_intent)
    )
    response_path = lease_root / "submit-response.json"
    if response_path.exists() or response_path.is_symlink():
        response_record = _read_latency_controller_record(
            response_path, "source-closure submit response"
        )
        if response_record.get("request_authorization_record_sha256") != (
            authorization.authorization_record_sha256
        ):
            raise ValueError("source-closure response request authority drift")
        run_id = _databricks_id(
            response_record.get("run_id"), "source-closure submit run ID"
        )
        response_sha = _required_sha256(
            response_record, "submit_response_sha256"
        )
    else:
        response = (
            submit_databricks_run(workspace, dict(payload))
            if opener is None
            else submit_databricks_run(workspace, dict(payload), opener=opener)
        )
        run_id = _databricks_id(
            response.get("run_id"), "source-closure submit run ID"
        )
        response_sha = _canonical_sha256(response)
        response_record = {
            "closed_record_sha256": "",
            "record_type": "cachet.publication_latency_source_submit_response.v1",
            "request_authorization_record_sha256": (
                authorization.authorization_record_sha256
            ),
            "run_id": run_id,
            "submit_response_sha256": response_sha,
        }
        response_record["closed_record_sha256"] = _closed_record_sha256(
            response_record
        )
        _write_or_require_exact_bytes(
            response_path, _canonical_json_bytes(response_record)
        )
    _require_unchanged_source_closure_gpu_ledger(ledger_file, predecessor)
    record = {
        "closed_record_sha256": "",
        "request_authorization_record_sha256": (
            authorization.authorization_record_sha256
        ),
        "request_closed_record_sha256": request["closed_record_sha256"],
        "request_file_byte_count": len(request_bytes),
        "record_type": "cachet.publication_latency_source_submission.v1",
        "run_id": run_id,
        "submit_payload_byte_count": len(canonical_payload),
        "submit_payload_sha256": payload_sha,
        "submit_response_sha256": response_sha,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    submission_authorization = PublicationLatencySourceClosureSubmissionAuthorization(
        request_closed_record_sha256=_required_sha256(
            request, "closed_record_sha256"
        ),
        request_authorization_record_sha256=(
            authorization.authorization_record_sha256
        ),
        ledger_path_sha256=databricks_ledger_path_sha256(ledger_file),
        predecessor_prefix=predecessor,
        run_id=run_id,
        submit_payload_sha256=payload_sha,
        submit_response_sha256=response_sha,
        _issuer=_SOURCE_CLOSURE_SUBMISSION_AUTHORIZATION_ISSUER,
    )
    return record, submission_authorization


def _require_unchanged_source_closure_gpu_ledger(
    ledger_file: Path,
    predecessor: DatabricksLedgerPrefix,
) -> None:
    ledger = read_databricks_cluster_hour_ledger_json(ledger_file)
    require_databricks_ledger_prefix(ledger, predecessor)
    if databricks_ledger_prefix(ledger) != predecessor:
        raise ValueError("source closure must not mutate or race the GPU ledger")


def collect_publication_latency_source_closure(
    workspace: DatabricksWorkspaceConfig,
    *,
    request: Mapping[str, Any],
    submission_authorization: PublicationLatencySourceClosureSubmissionAuthorization,
    ledger_path: str | Path,
    controller_cas_root: str | Path,
) -> PublicationLatencySourceClosureAuthorization:
    """Collect direct status and compact closure without mounting DBFS on Mac."""

    if not isinstance(workspace, DatabricksWorkspaceConfig):
        raise TypeError("workspace has the wrong type")
    validate_publication_latency_source_closure_request(request)
    if not isinstance(
        submission_authorization, PublicationLatencySourceClosureSubmissionAuthorization
    ):
        raise TypeError("source closure collection requires submission authority")
    request_sha = _required_sha256(request, "closed_record_sha256")
    request_authorization_record = _source_closure_request_authorization_record(
        request
    )
    ledger_file = Path(ledger_path).expanduser().absolute()
    if (
        submission_authorization.request_closed_record_sha256 != request_sha
        or submission_authorization.request_authorization_record_sha256
        != request_authorization_record["closed_record_sha256"]
        or submission_authorization.ledger_path_sha256
        != databricks_ledger_path_sha256(ledger_file)
    ):
        raise ValueError("source-closure submission authority binding drift")
    payload = _render_publication_latency_source_closure_submit_payload_from_request(
        request
    )
    _snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(payload)
    if submission_authorization.submit_payload_sha256 != sha256(
        canonical_payload
    ).hexdigest():
        raise ValueError("source-closure submitted payload binding drift")
    _require_unchanged_source_closure_gpu_ledger(
        ledger_file, submission_authorization.predecessor_prefix
    )
    run_id = submission_authorization.run_id
    run = get_databricks_run(workspace, run_id)
    identity = _validate_source_closure_control_plane_run(
        run,
        submit_payload=payload,
        receipt_run_id=run_id,
    )
    if identity["terminal_state"] != "succeeded":
        raise RuntimeError("publication latency source-closure coordinator failed")
    control_sha = _control_plane_status_sha256(run)
    request_bytes = download_databricks_volume_file_bytes(
        workspace,
        _required_string(request, "request_uri"),
        max_bytes=PUBLICATION_LATENCY_SOURCE_CLOSURE_MAX_BYTES,
    )
    if request_bytes != _pretty_json_bytes(request):
        raise ValueError("remote source-closure request bytes drift")
    result_bytes = download_databricks_volume_file_bytes(
        workspace,
        _required_string(request, "result_uri"),
        max_bytes=PUBLICATION_LATENCY_SOURCE_CLOSURE_MAX_BYTES,
    )
    result = _canonical_remote_json_record(
        result_bytes, field_name="source-closure coordinator result"
    )
    validate_publication_latency_source_closure_result(result, request=request)
    if _mapping(result, "coordinator").get("run_id") != run_id:
        raise ValueError("source-closure result/control-plane run identity drift")
    for raw in (request_bytes, result_bytes, _control_plane_status_bytes(run)):
        _store_publication_latency_controller_cas_bytes(controller_cas_root, raw)
    _require_unchanged_source_closure_gpu_ledger(
        ledger_file, submission_authorization.predecessor_prefix
    )
    return PublicationLatencySourceClosureAuthorization(
        request=request,
        result=result,
        result_file_sha256=sha256(result_bytes).hexdigest(),
        coordinator_run_id=run_id,
        control_plane_status_sha256=control_sha,
        ledger_prefix=submission_authorization.predecessor_prefix,
        _issuer=_SOURCE_CLOSURE_AUTHORIZATION_ISSUER,
    )


def require_publication_latency_source_closure_authorization(
    authorization: object,
    *,
    expected_final_artifacts: PublicationLatencyFinalArtifactPins,
    expected_qualification_artifact_pins: GPUQualificationArtifactPinsV2,
    expected_semantic: Mapping[str, Any],
    expected_predecessor_prefix: DatabricksLedgerPrefix,
    expected_q8_handoff_authorization: PublicationHandoffRemoteClosureAuthorization,
    expected_bf16_handoff_authorization: PublicationHandoffRemoteClosureAuthorization,
) -> PublicationLatencySourceClosureAuthorization:
    """Replay one exact issuer capability against immutable source identities."""

    if not isinstance(authorization, PublicationLatencySourceClosureAuthorization):
        raise TypeError(
            "latency launch requires PublicationLatencySourceClosureAuthorization"
        )
    if not isinstance(expected_final_artifacts, PublicationLatencyFinalArtifactPins):
        raise TypeError("expected_final_artifacts has the wrong type")
    if not isinstance(
        expected_qualification_artifact_pins, GPUQualificationArtifactPinsV2
    ):
        raise TypeError("expected qualification artifact pins must be native v2")
    validate_publication_latency_source_closure_result(
        authorization.result_record, request=authorization.request_record
    )
    request = authorization.request_record
    request_pins = _gpu_qualification_artifact_pins_v2_from_record(
        _mapping(request, "qualification_artifact_pins")
    )
    if request_pins != expected_qualification_artifact_pins:
        raise ValueError("publication latency source-closure qualification pin drift")
    for handoff_authorization in (
        expected_q8_handoff_authorization,
        expected_bf16_handoff_authorization,
    ):
        _require_handoff_closure_runtime_pins(
            handoff_authorization,
            expected_final_artifacts=expected_final_artifacts,
            expected_qualification_artifact_pins=request_pins,
        )
    expected_handoff_closures = {
        "bf16": _remote_handoff_authorization_binding(
            expected_bf16_handoff_authorization
        ),
        "q8": _remote_handoff_authorization_binding(
            expected_q8_handoff_authorization
        ),
    }
    if (
        dict(_mapping(request, "final_artifacts"))
        != expected_final_artifacts.to_record()
        or request.get("input_bundle_sha256")
        != expected_qualification_artifact_pins.input_bundle_sha256
        or dict(_mapping(request, "expected_semantic")) != dict(expected_semantic)
        or dict(_mapping(request, "handoff_closures"))
        != expected_handoff_closures
        or authorization.predecessor_prefix != expected_predecessor_prefix
        or authorization.request_closed_record_sha256
        != request.get("closed_record_sha256")
        or authorization.result_closed_record_sha256
        != authorization.result_record.get("closed_record_sha256")
        or authorization.artifacts_sha256
        != authorization.result_record.get("artifacts_sha256")
    ):
        raise ValueError("publication latency source-closure authority binding drift")
    expected_causal = _canonical_sha256(
        {
            "control_plane_status_sha256": authorization.control_plane_status_sha256,
            "coordinator_run_id": authorization.coordinator_run_id,
            "ledger_prefix": authorization.ledger_prefix.to_record(),
            "request_closed_record_sha256": authorization.request_closed_record_sha256,
            "request_file_sha256": authorization.request_file_sha256,
            "result_closed_record_sha256": authorization.result_closed_record_sha256,
            "result_file_sha256": authorization.result_file_sha256,
        }
    )
    if expected_causal != authorization.causal_closure_sha256:
        raise ValueError("publication latency source-closure causal binding drift")
    return authorization


def build_publication_latency_execution_plan(
    *,
    campaign_plan_record: Mapping[str, Any],
    schedule_records: Mapping[int, Mapping[str, Any]],
    storage_schedule_records: Mapping[int, Mapping[str, Any]],
    qualification_plan_record: Mapping[str, Any],
    qualification_evidence_record: Mapping[str, Any],
    qualification_artifact_pins: GPUQualificationArtifactPinsV2,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
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
    campaign_ledger_prefix = databricks_ledger_prefix_from_record(
        _mapping(campaign_plan_record, "campaign_ledger_prefix")
    )
    if not isinstance(qualification_artifact_pins, GPUQualificationArtifactPinsV2):
        raise TypeError("qualification_artifact_pins must be native v2")
    if not isinstance(final_artifacts, PublicationLatencyFinalArtifactPins):
        raise TypeError("final_artifacts has the wrong type")
    if final_artifacts.file("runner").sha256 != PUBLICATION_LATENCY_RUNNER_SHA256:
        raise ValueError("final runner does not match the reviewed latency runner")
    _require_native_v2_final_runtime_artifacts(
        final_artifacts, qualification_artifact_pins
    )

    evidence_selection = validate_gpu_qualification_evidence_v2_record(
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
    _require_reviewed_qualification_plan_campaign_binding(
        campaign_plan_record, qualification_plan_record
    )

    handoff_execution_file = final_artifacts.file("handoff_execution")
    authenticated_handoff = require_q8_handoff_remote_closure_authorization(
        handoff_serving_authorization,
        expected_output_root_uri=final_artifacts.handoff_generation_root_uri,
        expected_execution_file_sha256=handoff_execution_file.sha256,
        expected_input_bundle_sha256=(qualification_artifact_pins.input_bundle_sha256),
        expected_qualification_closed_record_sha256=_required_sha256(
            qualification_evidence_record,
            "closed_record_sha256",
        ),
    )
    handoff_execution = authenticated_handoff.execution_record
    if (
        handoff_execution.get("execution_mode")
        != PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED
    ):
        raise ValueError("latency serving requires the distributed handoff result")
    handoff_accounting = _mapping(handoff_execution, "accounting")
    if handoff_accounting.get("full_launch_throughput_gate_passed") is not True:
        raise ValueError("distributed handoff result did not pass its launch gate")
    if (
        authenticated_handoff.execution_file_sha256 != handoff_execution_file.sha256
        or authenticated_handoff.execution_uri != handoff_execution_file.uri
        or authenticated_handoff.output_root_uri
        != final_artifacts.handoff_generation_root_uri
    ):
        raise ValueError("handoff execution file pin is stale")
    if handoff_execution.get("input_bundle_sha256") != (
        qualification_artifact_pins.input_bundle_sha256
    ):
        raise ValueError("handoff input bundle differs from qualification")
    generator_hardware = _mapping(handoff_execution, "generator_hardware")
    if generator_hardware.get("qualification_closed_record_sha256") != (
        qualification_evidence_record.get("closed_record_sha256")
    ):
        raise ValueError("handoff generation used a different GPU qualification")

    schedule_bindings = _validated_schedule_bindings(
        campaign_id=campaign_id,
        schedule_records=schedule_records,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        final_artifacts=final_artifacts,
        role_prefix="schedule_block",
        expected_request_count=PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL,
        expected_workload_id="main",
    )
    main_examples_by_block = {
        block: _schedule_examples(schedule_records[block])
        for block in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)
    }
    storage_source_examples = main_examples_by_block[1]
    if any(
        examples != storage_source_examples
        for examples in main_examples_by_block.values()
    ):
        raise ValueError("main latency schedules do not share one source identity set")
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
    _validate_remote_publication_storage_inputs_record(
        storage_inputs_record,
        source_examples=storage_source_examples,
        schedule_records=storage_schedule_records,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        final_artifacts=final_artifacts,
    )
    storage_inputs_artifact = final_artifacts.file("storage_inputs")
    if not _canonical_json_file_sha256_matches(
        storage_inputs_record,
        storage_inputs_artifact.sha256,
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
            or not _same_durable_file_location(
                artifact.uri,
                _required_string(file_record, "uri"),
            )
        ):
            raise ValueError("storage input final artifact binding drift")
    bf16_binding = _validated_bf16_generation_binding(
        bf16_handoff_serving_authorization,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        expected_qualification_closed_record_sha256=_required_sha256(
            qualification_evidence_record,
            "closed_record_sha256",
        ),
        final_artifacts=final_artifacts,
    )
    source_closure_schedule_bindings = _source_closure_schedule_bindings(
        schedule_records,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        final_artifacts=final_artifacts,
        storage=False,
    )
    source_closure_storage_schedule_bindings = _source_closure_schedule_bindings(
        storage_schedule_records,
        expected_input_bundle_sha256=qualification_artifact_pins.input_bundle_sha256,
        final_artifacts=final_artifacts,
        storage=True,
        source_examples=storage_source_examples,
    )
    expected_source_semantic = _source_closure_expected_semantic(
        campaign_plan_record=campaign_plan_record,
        qualification_plan_record=qualification_plan_record,
        qualification_evidence_record=qualification_evidence_record,
        selection=selection,
        schedule_bindings=source_closure_schedule_bindings,
        storage_schedule_bindings=source_closure_storage_schedule_bindings,
        storage_inputs_record=storage_inputs_record,
    )
    authenticated_source_closure = (
        require_publication_latency_source_closure_authorization(
            source_closure_authorization,
            expected_final_artifacts=final_artifacts,
            expected_qualification_artifact_pins=qualification_artifact_pins,
            expected_semantic=expected_source_semantic,
            expected_predecessor_prefix=(
                bf16_handoff_serving_authorization.ledger_prefix
            ),
            expected_q8_handoff_authorization=handoff_serving_authorization,
            expected_bf16_handoff_authorization=(
                bf16_handoff_serving_authorization
            ),
        )
    )
    authorization_ledger_ids = {
        qualification_launch_authorization.ledger_id,
        handoff_serving_authorization.ledger_id,
        bf16_handoff_serving_authorization.ledger_id,
        authenticated_source_closure.ledger_id,
    }
    if authorization_ledger_ids != {campaign_ledger_id}:
        raise ValueError(
            "GPU qualification, Q8, BF16, and source closure must share one "
            "publication campaign ledger"
        )
    authorization_ledger_paths = {
        qualification_launch_authorization.ledger_path_sha256,
        handoff_serving_authorization.ledger_path_sha256,
        bf16_handoff_serving_authorization.ledger_path_sha256,
        authenticated_source_closure.ledger_path_sha256,
    }
    if authorization_ledger_paths != {campaign_ledger_path_sha256}:
        raise ValueError("publication authorities use a different campaign ledger path")
    if qualification_launch_authorization.ledger_prefix.reservation_count < (
        campaign_ledger_prefix.reservation_count
    ):
        raise ValueError("qualification authority does not extend campaign genesis")
    _require_remote_handoff_phase_order(
        qualification_prefix=qualification_launch_authorization.ledger_prefix,
        q8_predecessor_prefix=handoff_serving_authorization.predecessor_prefix,
        q8_terminal_prefix=handoff_serving_authorization.ledger_prefix,
        bf16_predecessor_prefix=bf16_handoff_serving_authorization.predecessor_prefix,
    )
    if authenticated_source_closure.predecessor_prefix != (
        bf16_handoff_serving_authorization.ledger_prefix
    ):
        raise ValueError("source closure does not extend BF16 in phase order")
    for source_record, source_file, label in (
        (
            campaign_plan_record,
            final_artifacts.file("campaign_plan"),
            "campaign plan",
        ),
        (
            qualification_plan_record,
            final_artifacts.file("qualification_plan"),
            "qualification plan",
        ),
        (
            qualification_evidence_record,
            final_artifacts.file("qualification_evidence"),
            "qualification evidence",
        ),
    ):
        if not _canonical_json_file_sha256_matches(
            source_record,
            source_file.sha256,
        ):
            raise ValueError(f"{label} final artifact hash drift")
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
        "plan_record": dict(qualification_plan_record),
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
            "remote_closure": _remote_handoff_authorization_binding(
                handoff_serving_authorization
            ),
        },
        "execution": {
            **handoff_execution_file.to_record(),
            "closed_record_sha256": _required_sha256(
                handoff_execution,
                "closed_record_sha256",
            ),
        },
        "output_root_uri": final_artifacts.handoff_generation_root_uri,
    }
    source_closure_binding = _source_closure_authorization_binding(
        authenticated_source_closure
    )
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
        "source_closure": source_closure_binding,
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
            "single_user_name": _required_string(
                source_closure_binding,
                "single_user_name",
            ),
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
            "source_closure",
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
    _require_exact_keys(
        qualification,
        {"artifact_pins", "authorization", "evidence", "plan", "plan_record", "selection"},
        "native-v2 GPU qualification source binding",
    )
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
    pins = _gpu_qualification_artifact_pins_v2_from_record(
        _mapping(qualification, "artifact_pins")
    )
    embedded_qualification_plan = _mapping(qualification, "plan_record")
    validate_gpu_qualification_plan_v2_record(
        embedded_qualification_plan,
        expected_campaign_id=campaign_id,
        expected_artifact_pins=pins,
    )
    _require_reviewed_qualification_plan_campaign_binding(
        canonical_campaign, embedded_qualification_plan
    )
    if embedded_qualification_plan.get("closed_record_sha256") != _mapping(
        qualification, "plan"
    ).get("closed_record_sha256"):
        raise ValueError("embedded native-v2 qualification plan binding drift")
    _require_native_v2_final_runtime_artifacts(final_artifacts, pins)
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
                "remote_closure",
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
        remote_closure = _mapping(source_authorization, "remote_closure")
        _require_exact_keys(
            remote_closure,
            {
                "control_plane_status_sha256",
                "coordinator_run_id",
                "execution_closed_record_sha256",
                "execution_file_sha256",
                "execution_uri",
                "request_closed_record_sha256",
                "result_closed_record_sha256",
                "result_file_sha256",
                "result_uri",
                "stage",
            },
            f"{label} remote closure binding",
        )
        expected_stage = "q8" if label == "Q8 handoff" else "bf16"
        if remote_closure.get("stage") != expected_stage:
            raise ValueError(f"{label} remote closure stage drift")
        _databricks_id(remote_closure.get("coordinator_run_id"), "coordinator run ID")
        _durable_uri(
            _required_string(remote_closure, "execution_uri"),
            f"{label} remote execution URI",
        )
        _durable_uri(
            _required_string(remote_closure, "result_uri"),
            f"{label} remote result URI",
        )
        for digest_name in (
            "control_plane_status_sha256",
            "execution_closed_record_sha256",
            "execution_file_sha256",
            "request_closed_record_sha256",
            "result_closed_record_sha256",
            "result_file_sha256",
        ):
            _required_sha256(remote_closure, digest_name)
        source_execution = _mapping(source_binding, "execution")
        if (
            source_execution.get("uri") != remote_closure.get("execution_uri")
            or source_execution.get("sha256")
            != remote_closure.get("execution_file_sha256")
            or source_execution.get("closed_record_sha256")
            != remote_closure.get("execution_closed_record_sha256")
        ):
            raise ValueError(f"{label} remote execution binding drift")
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
    source_closure_binding = _mapping(sources, "source_closure")
    _require_exact_keys(
        source_closure_binding,
        {
            "artifacts_sha256",
            "causal_closure_sha256",
            "control_plane_status_sha256",
            "coordinator_run_id",
            "ledger_id",
            "ledger_path_sha256",
            "ledger_prefix",
            "predecessor_prefix",
            "request_closed_record_sha256",
            "request_file_sha256",
            "result_closed_record_sha256",
            "result_file_sha256",
            "result_uri",
            "single_user_name",
        },
        "publication latency source-closure binding",
    )
    for digest_name in (
        "artifacts_sha256",
        "causal_closure_sha256",
        "control_plane_status_sha256",
        "request_closed_record_sha256",
        "request_file_sha256",
        "result_closed_record_sha256",
        "result_file_sha256",
    ):
        _required_sha256(source_closure_binding, digest_name)
    _databricks_id(
        source_closure_binding.get("coordinator_run_id"),
        "source-closure coordinator run ID",
    )
    _databricks_volume_uri(
        source_closure_binding.get("result_uri"), "source-closure result URI"
    )
    source_single_user_name = _validated_single_user_name(
        source_closure_binding.get("single_user_name")
    )
    if (
        source_closure_binding.get("ledger_id") != campaign_ledger_id
        or source_closure_binding.get("ledger_path_sha256")
        != campaign_ledger_path_sha256
        or _mapping(source_closure_binding, "predecessor_prefix")
        != _mapping(bf16_authorization_binding, "ledger_prefix")
    ):
        raise ValueError("source closure is not the BF16 campaign successor")
    for prefix_name in (
        "predecessor_prefix",
        "ledger_prefix",
    ):
        prefix = databricks_ledger_prefix_from_record(
            _mapping(source_closure_binding, prefix_name)
        )
        if prefix.ledger_id != campaign_ledger_id:
            raise ValueError("source-closure ledger prefix identity drift")
    if _mapping(source_closure_binding, "ledger_prefix") != _mapping(
        source_closure_binding, "predecessor_prefix"
    ):
        raise ValueError("source closure must not append to the GPU ledger")
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
        "single_user_name": source_single_user_name,
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
        if not _canonical_json_file_sha256_matches(schedule, file.sha256):
            raise ValueError("publication schedule final artifact hash drift")
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


def _validate_remote_publication_storage_inputs_record(
    record: Mapping[str, Any],
    *,
    source_examples: Sequence[PublicationLatencyExample],
    schedule_records: Mapping[int, Mapping[str, Any]],
    expected_input_bundle_sha256: str,
    final_artifacts: PublicationLatencyFinalArtifactPins,
) -> None:
    """Validate compact storage inputs without reopening mounted source rows."""

    _require_exact_keys(
        record,
        {
            "closed_record_sha256",
            "files",
            "input_bundle_sha256",
            "output_root",
            "record_type",
            "schedule_bindings",
            "schema_version",
            "selection_protocol",
        },
        "publication storage inputs",
    )
    if (
        record.get("record_type") != PUBLICATION_STORAGE_INPUTS_RECORD_TYPE
        or record.get("schema_version") != PUBLICATION_STORAGE_INPUTS_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
        or record.get("input_bundle_sha256") != expected_input_bundle_sha256
    ):
        raise ValueError("publication storage input envelope is invalid")
    selected = select_publication_storage_examples(
        source_examples,
        input_bundle_sha256=expected_input_bundle_sha256,
    )
    selected_ids = {
        dataset: sorted(
            item.example_id for item in selected if item.dataset == dataset
        )
        for dataset in SUPPORTED_V1_DATASETS
    }
    expected_schedule_bindings: list[dict[str, Any]] = []
    selection_records: list[Mapping[str, Any]] = []
    for block in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1):
        schedule = schedule_records[block]
        validate_publication_storage_block_schedule(
            schedule,
            source_examples=source_examples,
            expected_input_bundle_sha256=expected_input_bundle_sha256,
        )
        selection = _mapping(_mapping(schedule, "protocol"), "selection")
        selection_records.append(selection)
        expected_schedule_bindings.append(
            {
                "closed_record_sha256": _required_sha256(
                    schedule,
                    "closed_record_sha256",
                ),
                "deployment_block": block,
                "requests_sha256": _required_sha256(schedule, "requests_sha256"),
                "selection_sha256": _required_sha256(
                    selection,
                    "selection_sha256",
                ),
            }
        )
    if any(selection != selection_records[0] for selection in selection_records):
        raise ValueError("publication storage schedule selections are inconsistent")
    selection = selection_records[0]
    expected_selection_protocol = {
        **dict(selection),
        "examples_per_dataset": PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET,
        "repeats_per_example": PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE,
        "request_count": PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL,
        "source_row_bytes_preserved": True,
    }
    if (
        _mapping_sequence(record, "schedule_bindings")
        != expected_schedule_bindings
        or dict(_mapping(record, "selection_protocol"))
        != expected_selection_protocol
    ):
        raise ValueError("publication storage schedule/selection binding drift")
    files = _mapping_sequence(record, "files")
    if tuple(item.get("dataset") for item in files) != tuple(SUPPORTED_V1_DATASETS):
        raise ValueError("publication storage input dataset coverage is incomplete")
    output_root = _normalized_volume_path(
        _required_string(record, "output_root"),
        "publication storage output root",
    )
    for item in files:
        _require_exact_keys(
            item,
            {
                "byte_count",
                "dataset",
                "identities",
                "record_count",
                "rows_sha256",
                "sha256",
                "source_sha256",
                "uri",
            },
            "publication storage input file",
        )
        dataset = _required_string(item, "dataset")
        artifact = final_artifacts.file(f"storage_input_16384_{dataset}")
        source_artifact = final_artifacts.file(f"input_16384_{dataset}")
        identities = item.get("identities")
        if (
            not isinstance(identities, list)
            or identities != selected_ids[dataset]
            or _required_int(item, "record_count")
            != PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET
            or _positive_int(item.get("byte_count"), "storage input byte count") <= 0
            or _required_sha256(item, "sha256") != artifact.sha256
            or _required_sha256(item, "source_sha256") != source_artifact.sha256
            or _normalized_volume_path(
                _required_string(item, "uri"),
                "publication storage input URI",
            ).parent
            != output_root
            or not _same_durable_file_location(
                artifact.uri,
                _required_string(item, "uri"),
            )
        ):
            raise ValueError("publication storage input file binding drift")
        _required_sha256(item, "rows_sha256")


def _validated_bf16_generation_binding(
    authorization: PublicationHandoffRemoteClosureAuthorization,
    *,
    expected_input_bundle_sha256: str,
    expected_qualification_closed_record_sha256: str,
    final_artifacts: PublicationLatencyFinalArtifactPins,
) -> dict[str, Any]:
    if not isinstance(authorization, PublicationHandoffRemoteClosureAuthorization):
        raise TypeError(
            "bf16_handoff_serving_authorization must be a coordinator-issued "
            "PublicationHandoffRemoteClosureAuthorization"
    )
    manifest_artifact = final_artifacts.file("bf16_handoff_manifest")
    execution_artifact = final_artifacts.file("bf16_handoff_execution")
    authenticated = require_bf16_handoff_remote_closure_authorization(
        authorization,
        expected_output_root_uri=final_artifacts.bf16_handoff_generation_root_uri,
        expected_execution_file_sha256=execution_artifact.sha256,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
        expected_qualification_closed_record_sha256=(
            expected_qualification_closed_record_sha256
        ),
    )
    manifest_bindings = _mapping_sequence(authenticated.result_record, "manifests")
    if len(manifest_bindings) != 1 or len(authenticated.manifest_records) != 1:
        raise ValueError("BF16 remote closure must contain exactly one manifest")
    manifest_binding = manifest_bindings[0]
    manifest = authenticated.manifest_records[0]
    execution = authenticated.execution_record
    if (
        authenticated.execution_uri != execution_artifact.uri
        or authenticated.execution_file_sha256 != execution_artifact.sha256
        or manifest_binding.get("uri") != manifest_artifact.uri
        or manifest_binding.get("file_sha256") != manifest_artifact.sha256
        or manifest_binding.get("source_root_uri")
        != final_artifacts.bf16_handoff_source_root_uri
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
            "remote_closure": _remote_handoff_authorization_binding(authorization),
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


def _remote_handoff_authorization_binding(
    authorization: PublicationHandoffRemoteClosureAuthorization,
) -> dict[str, Any]:
    if not isinstance(authorization, PublicationHandoffRemoteClosureAuthorization):
        raise TypeError("remote handoff closure authorization has the wrong type")
    return {
        "control_plane_status_sha256": authorization.control_plane_status_sha256,
        "coordinator_run_id": authorization.coordinator_run_id,
        "execution_closed_record_sha256": (
            authorization.execution_closed_record_sha256
        ),
        "execution_file_sha256": authorization.execution_file_sha256,
        "execution_uri": authorization.execution_uri,
        "output_root_uri": authorization.output_root_uri,
        "request_closed_record_sha256": (
            authorization.request_closed_record_sha256
        ),
        "result_closed_record_sha256": authorization.result_closed_record_sha256,
        "result_file_sha256": authorization.result_file_sha256,
        "result_uri": authorization.result_uri,
        "stage": authorization.stage,
    }


def _source_closure_singleton_identity_from_request(
    request: Mapping[str, Any],
) -> str:
    return _canonical_sha256(
        {
            "artifacts": dict(_mapping(request, "final_artifacts")),
            "domain": "cachet.publication.latency_source_closure.singleton.v2",
            "expected_semantic": dict(_mapping(request, "expected_semantic")),
            "handoff_closures": dict(_mapping(request, "handoff_closures")),
            "ledger_lineage": dict(_mapping(request, "ledger_lineage")),
            "qualification_artifact_pins": dict(
                _mapping(request, "qualification_artifact_pins")
            ),
            "record_bindings": dict(_mapping(request, "record_bindings")),
        }
    )


def _publication_latency_source_closure_attempt_id(
    singleton_identity_sha256: str,
) -> str:
    identity = _require_sha256_value(
        singleton_identity_sha256, "source-closure singleton identity"
    )
    return f"latency-source-closure-{identity[:24]}"


def publication_latency_source_closure_control_roots(
    q8_output_root_uri: str,
    bf16_output_root_uri: str,
) -> tuple[str, str]:
    q8 = _normalized_volume_path(q8_output_root_uri, "Q8 output root")
    bf16 = _normalized_volume_path(bf16_output_root_uri, "BF16 output root")
    common_parts: list[str] = []
    for q8_part, bf16_part in zip(q8.parts, bf16.parts, strict=False):
        if q8_part != bf16_part:
            break
        common_parts.append(q8_part)
    if len(common_parts) < 5:
        raise ValueError("source-closure handoff outputs must share one UC volume")
    common = PurePosixPath(*common_parts).as_posix()
    output_pair_identity = _canonical_sha256(
        {
            "bf16_output_root_uri": _databricks_volume_uri(
                bf16_output_root_uri, "BF16 output root"
            ),
            "domain": "cachet.publication.latency_source_closure.control_root.v2",
            "q8_output_root_uri": _databricks_volume_uri(
                q8_output_root_uri, "Q8 output root"
            ),
        }
    )
    control = _join_durable_uri(
        "dbfs:" + common,
        "publication-latency-source-closure-control",
        output_pair_identity[:32],
    )
    return (
        _join_durable_uri(control, "requests"),
        _join_durable_uri(control, "results"),
    )


def _source_closure_phase_lease_root(
    ledger_path: str | Path,
    *,
    singleton_identity_sha256: str,
) -> Path:
    ledger = Path(ledger_path).expanduser().absolute()
    identity = _require_sha256_value(
        singleton_identity_sha256, "source-closure singleton identity"
    )
    root = ledger.parent / (
        f".{ledger.name}.latency-source-closure-{identity[:24]}.lease"
    )
    _reject_existing_symlink_ancestors(root, "source-closure phase lease")
    return root


def _require_handoff_closure_runtime_pins(
    authorization: PublicationHandoffRemoteClosureAuthorization,
    *,
    expected_final_artifacts: PublicationLatencyFinalArtifactPins,
    expected_qualification_artifact_pins: GPUQualificationArtifactPinsV2,
) -> None:
    if not isinstance(authorization, PublicationHandoffRemoteClosureAuthorization):
        raise TypeError("handoff runtime closure authorization has the wrong type")
    coordinator = _mapping(authorization.request_record, "coordinator")
    expected_output_root = (
        expected_final_artifacts.handoff_generation_root_uri
        if authorization.stage == "q8"
        else expected_final_artifacts.bf16_handoff_generation_root_uri
    )
    if (
        not _same_durable_file_location(
            authorization.output_root_uri, expected_output_root
        )
        or authorization.request_record.get("input_bundle_sha256")
        != expected_qualification_artifact_pins.input_bundle_sha256
    ):
        raise ValueError(f"{authorization.stage} handoff source-root binding drift")
    if authorization.stage == "bf16":
        manifests = _mapping_sequence(authorization.result_record, "manifests")
        if len(manifests) != 1 or not _same_durable_file_location(
            _required_string(manifests[0], "source_root_uri"),
            expected_final_artifacts.bf16_handoff_source_root_uri,
        ):
            raise ValueError("bf16 handoff manifest source-root binding drift")
    expected_files = {
        "package_wheel": ("package_wheel_uri", "package_wheel_sha256"),
        "patched_vllm_wheel": (
            "patched_vllm_wheel_uri",
            "patched_vllm_wheel_sha256",
        ),
        "patched_flashinfer_wheel": (
            "patched_flashinfer_wheel_uri",
            "patched_flashinfer_wheel_sha256",
        ),
        "runtime_closure_manifest": (
            "runtime_closure_manifest_uri",
            "runtime_closure_manifest_sha256",
        ),
        "runtime_lock": ("runtime_lock_uri", "runtime_lock_sha256"),
    }
    for role, (uri_name, digest_name) in expected_files.items():
        artifact = expected_final_artifacts.file(role)
        if (
            coordinator.get(digest_name) != artifact.sha256
            or not _same_durable_file_location(
                _required_string(coordinator, uri_name), artifact.uri
            )
        ):
            raise ValueError(
                f"{authorization.stage} handoff {role} runtime binding drift"
            )
    if (
        coordinator.get("source_revision")
        != expected_final_artifacts.source_revision
        or coordinator.get("cachet_source_tree_sha256")
        != expected_qualification_artifact_pins.cachet_source_tree_sha256
        or coordinator.get("package_wheel_sha256")
        != expected_qualification_artifact_pins.package_wheel_sha256
        or coordinator.get("patched_vllm_wheel_sha256")
        != expected_qualification_artifact_pins.patched_vllm_wheel_sha256
        or coordinator.get("patched_flashinfer_wheel_sha256")
        != expected_qualification_artifact_pins.patched_flashinfer_wheel_sha256
        or coordinator.get("runtime_closure_manifest_sha256")
        != expected_qualification_artifact_pins.runtime_closure_manifest_sha256
        or coordinator.get("runtime_lock_sha256")
        != expected_qualification_artifact_pins.runtime_lock_sha256
    ):
        raise ValueError(f"{authorization.stage} handoff package/source pin drift")


def _source_closure_authorization_binding(
    authorization: PublicationLatencySourceClosureAuthorization,
) -> dict[str, Any]:
    if not isinstance(authorization, PublicationLatencySourceClosureAuthorization):
        raise TypeError("source-closure authorization has the wrong type")
    coordinator = _source_closure_config_from_record(
        _mapping(authorization.request_record, "coordinator")
    )
    return {
        "artifacts_sha256": authorization.artifacts_sha256,
        "causal_closure_sha256": authorization.causal_closure_sha256,
        "control_plane_status_sha256": authorization.control_plane_status_sha256,
        "coordinator_run_id": authorization.coordinator_run_id,
        "ledger_id": authorization.ledger_id,
        "ledger_path_sha256": authorization.ledger_path_sha256,
        "ledger_prefix": authorization.ledger_prefix.to_record(),
        "predecessor_prefix": authorization.predecessor_prefix.to_record(),
        "request_closed_record_sha256": (
            authorization.request_closed_record_sha256
        ),
        "request_file_sha256": authorization.request_file_sha256,
        "result_closed_record_sha256": authorization.result_closed_record_sha256,
        "result_file_sha256": authorization.result_file_sha256,
        "result_uri": authorization.result_uri,
        "single_user_name": coordinator.single_user_name,
    }


def _require_reviewed_qualification_plan_campaign_binding(
    campaign_plan_record: Mapping[str, Any],
    qualification_plan_record: Mapping[str, Any],
) -> DatabricksLedgerPrefix:
    """Bind the reviewed campaign and v2 openings to the same frozen lineage."""

    campaign_prefix = databricks_ledger_prefix_from_record(
        _mapping(campaign_plan_record, "campaign_ledger_prefix")
    )
    qualification_prefix = databricks_ledger_prefix_from_record(
        _mapping(qualification_plan_record, "campaign_ledger_prefix")
    )
    campaign_hours = _finite_nonnegative_number(
        campaign_plan_record.get("campaign_opening_terminal_gpu_hours"),
        "campaign opening terminal GPU hours",
    )
    qualification_hours = _finite_nonnegative_number(
        qualification_plan_record.get("campaign_opening_terminal_gpu_hours"),
        "qualification opening terminal GPU hours",
    )
    if (
        campaign_prefix != PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX
        or campaign_hours != PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
        or qualification_prefix != GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
        or qualification_hours != GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS
        or qualification_plan_record.get("campaign_id")
        != campaign_plan_record.get("campaign_id")
        or qualification_plan_record.get("campaign_record_sha256")
        != campaign_plan_record.get("closed_record_sha256")
        or qualification_plan_record.get("campaign_ledger_id")
        != campaign_plan_record.get("campaign_ledger_id")
        or qualification_plan_record.get("campaign_ledger_path_sha256")
        != campaign_plan_record.get("campaign_ledger_path_sha256")
        or qualification_prefix.ledger_id != campaign_prefix.ledger_id
        or qualification_prefix.cap_cluster_hours != campaign_prefix.cap_cluster_hours
    ):
        raise ValueError(
            "GPU qualification plan does not carry the reviewed campaign successor "
            "authority"
        )
    return qualification_prefix


def _require_reviewed_qualification_plan_campaign_successor(
    ledger: DatabricksClusterHourLedger,
    *,
    ledger_path: str | Path,
    campaign_plan_record: Mapping[str, Any],
    qualification_plan_record: Mapping[str, Any],
) -> DatabricksLedgerPrefix:
    """Prove the reviewed campaign opening is an ordered slice of v2."""

    qualification_prefix = _require_reviewed_qualification_plan_campaign_binding(
        campaign_plan_record, qualification_plan_record
    )
    campaign_prefix = databricks_ledger_prefix_from_record(
        _mapping(campaign_plan_record, "campaign_ledger_prefix")
    )
    if databricks_ledger_path_sha256(ledger_path) != _required_sha256(
        qualification_plan_record, "campaign_ledger_path_sha256"
    ):
        raise ValueError("reviewed campaign successor uses a different ledger path")
    require_databricks_ledger_prefix(ledger, qualification_prefix)
    qualification_ledger = DatabricksClusterHourLedger(
        ledger_id=ledger.ledger_id,
        cap_cluster_hours=ledger.cap_cluster_hours,
        reservations=ledger.reservations[: qualification_prefix.reservation_count],
        submission_receipts=ledger.submission_receipts[
            : qualification_prefix.submission_receipt_count
        ],
        terminal_actuals=ledger.terminal_actuals[
            : qualification_prefix.terminal_actual_count
        ],
    )
    require_databricks_ledger_prefix(qualification_ledger, qualification_prefix)
    require_databricks_ledger_prefix(qualification_ledger, campaign_prefix)
    campaign_ledger = DatabricksClusterHourLedger(
        ledger_id=ledger.ledger_id,
        cap_cluster_hours=ledger.cap_cluster_hours,
        reservations=ledger.reservations[: campaign_prefix.reservation_count],
        submission_receipts=ledger.submission_receipts[
            : campaign_prefix.submission_receipt_count
        ],
        terminal_actuals=ledger.terminal_actuals[
            : campaign_prefix.terminal_actual_count
        ],
    )
    require_databricks_ledger_prefix(campaign_ledger, campaign_prefix)
    if campaign_ledger.terminal_actual_cluster_hours != (
        PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
    ):
        raise ValueError("reviewed campaign opening terminal balance drift")
    if qualification_ledger.terminal_actual_cluster_hours != (
        GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS
    ):
        raise ValueError("reviewed qualification opening terminal balance drift")
    return qualification_prefix


def _require_remote_handoff_phase_order(
    *,
    qualification_prefix: DatabricksLedgerPrefix,
    q8_predecessor_prefix: DatabricksLedgerPrefix,
    q8_terminal_prefix: DatabricksLedgerPrefix,
    bf16_predecessor_prefix: DatabricksLedgerPrefix,
) -> None:
    prefixes = (
        qualification_prefix,
        q8_predecessor_prefix,
        q8_terminal_prefix,
        bf16_predecessor_prefix,
    )
    if any(not isinstance(item, DatabricksLedgerPrefix) for item in prefixes):
        raise TypeError("remote handoff phase order requires ledger-prefix authorities")
    if q8_predecessor_prefix != qualification_prefix:
        raise ValueError("Q8 authority does not extend qualification in phase order")
    if bf16_predecessor_prefix != q8_terminal_prefix:
        raise ValueError("BF16 authority does not extend Q8 in phase order")


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
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
) -> dict[str, Any]:
    """Render a worker only after all replay-backed launch authorities."""

    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
        source_closure_authorization,
    )
    validate_publication_latency_execution_sources(
        execution_plan_record,
        handoff_serving_authorization=handoff_serving_authorization,
        bf16_handoff_serving_authorization=bf16_handoff_serving_authorization,
        source_closure_authorization=source_closure_authorization,
    )
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
        single_user_name=_required_string(
            _mapping(sources, "source_closure"),
            "single_user_name",
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
                "patched_flashinfer_wheel",
                "runtime_closure_manifest",
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
        "qualification_artifact_pins": dict(
            _mapping(qualification, "artifact_pins")
        ),
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
            "qualification_artifact_pins",
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
        "patched_flashinfer_wheel",
        "runtime_closure_manifest",
        "runtime_lock",
    ):
        raise ValueError("publication latency runtime artifact closure is incomplete")
    if artifacts[0].get("sha256") != PUBLICATION_LATENCY_RUNNER_SHA256:
        raise ValueError("publication latency runner hash drift")
    expected_runtime_hashes = {
        "patched_vllm_wheel": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        "patched_flashinfer_wheel": FLASHINFER_PATCHED_WHEEL_SHA256,
        "runtime_closure_manifest": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "runtime_lock": VLLM_RUNTIME_BASE_LOCK_SHA256,
    }
    artifacts_by_role = {item.get("role"): item for item in artifacts}
    for role, expected_sha256 in expected_runtime_hashes.items():
        if artifacts_by_role[role].get("sha256") != expected_sha256:
            raise ValueError(f"publication latency {role} hash drift")
    pins = _gpu_qualification_artifact_pins_v2_from_record(
        _mapping(record, "qualification_artifact_pins")
    )
    expected_artifact_pins = {
        "package_wheel": pins.package_wheel_sha256,
        "patched_vllm_wheel": pins.patched_vllm_wheel_sha256,
        "patched_flashinfer_wheel": pins.patched_flashinfer_wheel_sha256,
        "runtime_closure_manifest": pins.runtime_closure_manifest_sha256,
        "runtime_lock": pins.runtime_lock_sha256,
    }
    if any(
        artifacts_by_role[role].get("sha256") != expected_sha256
        for role, expected_sha256 in expected_artifact_pins.items()
    ):
        raise ValueError("publication latency runtime differs from qualification")
    if _required_sha256(record, "source_tree_sha256") != (
        pins.cachet_source_tree_sha256
    ):
        raise ValueError("publication latency source tree differs from qualification")
    inputs = _mapping_sequence(record, "input_files")
    if tuple(item.get("dataset") for item in inputs) != tuple(SUPPORTED_V1_DATASETS):
        raise ValueError("publication latency input files are incomplete")
    for item in (*artifacts, *inputs):
        _durable_uri(_required_string(item, "uri"), "job artifact URI")
        _required_sha256(item, "sha256")
    request_order = _mapping(record, "request_order")
    if request_order.get("input_bundle_sha256") != pins.input_bundle_sha256:
        raise ValueError("publication latency input differs from qualification")
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
        single_user_name=_required_string(runtime, "single_user_name"),
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
    single_user_name: str,
) -> dict[str, Any]:
    principal = _validated_single_user_name(single_user_name)
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
    if hardware_target == "aws-g5-a10g":
        gpu_memory_utilization = GPU_QUALIFICATION_A10G_GPU_MEMORY_UTILIZATION
    elif input_tokens == 32_768:
        gpu_memory_utilization = selected_32k_gmu
    else:
        gpu_memory_utilization = 0.90
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
        "single_user_name": principal,
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
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
) -> dict[str, Any]:
    """Render one task only after all replay-backed launch authorities."""

    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
        source_closure_authorization,
    )
    validate_publication_latency_execution_sources(
        execution_plan_record,
        handoff_serving_authorization=handoff_serving_authorization,
        bf16_handoff_serving_authorization=bf16_handoff_serving_authorization,
        source_closure_authorization=source_closure_authorization,
    )
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
            single_user_name=_required_string(runtime, "single_user_name"),
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
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
) -> tuple[dict[str, Any], ...]:
    """Return all 115 payloads in canonical campaign order."""

    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
        source_closure_authorization,
    )
    validate_publication_latency_execution_sources(
        execution_plan_record,
        handoff_serving_authorization=handoff_serving_authorization,
        bf16_handoff_serving_authorization=bf16_handoff_serving_authorization,
        source_closure_authorization=source_closure_authorization,
    )
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
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
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
        source_closure_authorization,
    )
    validate_publication_latency_execution_sources(
        execution_plan_record,
        handoff_serving_authorization=handoff_serving_authorization,
        bf16_handoff_serving_authorization=bf16_handoff_serving_authorization,
        source_closure_authorization=source_closure_authorization,
    )
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

    require_databricks_current_user_name(
        config,
        expected_user_name=_publication_latency_single_user_name(
            execution_plan_record
        ),
        opener=opener,
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
    for (
        job_id,
        job_record,
        submit_payload,
        attempt_id,
        payload_sha256,
        _timeout,
    ) in jobs:
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
        _validate_submit_payload(submit_payload, job_record=job_record)
        response = submit_pre_reserved_databricks_run(
            config,
            submit_payload,
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
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
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
        source_closure_authorization,
    )
    validate_publication_latency_execution_sources(
        execution_plan_record,
        handoff_serving_authorization=handoff_serving_authorization,
        bf16_handoff_serving_authorization=bf16_handoff_serving_authorization,
        source_closure_authorization=source_closure_authorization,
    )
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
    batch_exists = batch_path.exists() or batch_path.is_symlink()
    if batch_exists:
        if _read_latency_controller_record(
            batch_path, "latency batch marker"
        ) != expected_batch:
            raise ValueError("latency batch marker differs from the ledger batch")
    require_databricks_current_user_name(
        config,
        expected_user_name=_publication_latency_single_user_name(
            execution_plan_record
        ),
        opener=opener,
    )
    if not batch_exists:
        _write_canonical_json_exclusive(batch_path, expected_batch)
    submitted: list[dict[str, Any]] = []
    for job_id, job, payload, attempt_id, payload_sha256 in jobs:
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
        _validate_submit_payload(payload, job_record=job)
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


@dataclass(frozen=True, slots=True)
class _PublicationLatencyRemoteResult:
    record: Mapping[str, Any]
    result_tree: Mapping[str, Any]
    result_file_sha256: str
    result_file_byte_count: int
    result_tree_sha256: str
    result_tree_file_count: int
    result_tree_total_bytes: int


def _collect_remote_publication_latency_result(
    config: DatabricksWorkspaceConfig,
    *,
    job_record: Mapping[str, Any],
    controller_cas_root: str | Path,
) -> _PublicationLatencyRemoteResult:
    """Fetch and authenticate one exact result tree without a DBFS mount."""

    output = _mapping(job_record, "output")
    directory_uri = _databricks_volume_uri(
        _required_string(output, "directory_uri"),
        "latency result directory URI",
    )
    result_uri = _databricks_volume_uri(
        _required_string(output, "result_uri"),
        "latency result URI",
    )
    if result_uri != _join_durable_uri(
        directory_uri,
        PUBLICATION_LATENCY_RESULT_FILENAME,
    ):
        raise ValueError("latency result URI is not the exact result directory child")

    result_bytes = download_databricks_volume_file_bytes(
        config,
        result_uri,
        max_bytes=PUBLICATION_LATENCY_REMOTE_RESULT_MAX_BYTES,
    )
    result = _canonical_remote_json_record(
        result_bytes,
        field_name=f"latency result {job_record.get('job_id')}",
    )
    result_file_sha256 = sha256(result_bytes).hexdigest()
    _store_publication_latency_controller_cas_bytes(
        controller_cas_root,
        result_bytes,
    )
    validate_publication_latency_job_result_record(
        result,
        expected_job_record=job_record,
        verify_files=False,
    )

    result_files = _mapping_sequence(result, "files")
    if len(result_files) + 1 > PUBLICATION_LATENCY_REMOTE_TREE_MAX_FILES:
        raise ValueError("latency result tree exceeds the closed file-count cap")
    expected_by_path: dict[str, tuple[str, str, str | None]] = {
        result_uri.removeprefix("dbfs:"): (
            "result_seal",
            result_uri,
            result_file_sha256,
        )
    }
    for item in result_files:
        role = _safe_id(item.get("role"), "latency result file role")
        uri = _databricks_volume_uri(
            _required_string(item, "uri"),
            f"latency result file {role} URI",
        )
        path = PurePosixPath(uri.removeprefix("dbfs:"))
        if path.parent.as_posix() != directory_uri.removeprefix("dbfs:"):
            raise ValueError("latency result file escaped its exact result directory")
        raw_path = path.as_posix()
        if raw_path in expected_by_path:
            raise ValueError("latency result tree contains a duplicate file path")
        expected_by_path[raw_path] = (
            role,
            uri,
            _required_sha256(item, "sha256"),
        )

    listing = list_databricks_volume_directory(
        config,
        directory_uri,
        max_entries=PUBLICATION_LATENCY_REMOTE_DIRECTORY_MAX_FILES,
    )
    listed_by_path = {
        _required_string(item, "path"): item
        for item in listing
    }
    missing = sorted(set(expected_by_path).difference(listed_by_path))
    extra = sorted(set(listed_by_path).difference(expected_by_path))
    unexpected_extra = [
        path
        for path in extra
        if PurePosixPath(path).name
        not in PUBLICATION_LATENCY_REMOTE_AUXILIARY_FILENAMES
    ]
    if missing or unexpected_extra:
        raise ValueError(
            "latency remote result directory closure drift: "
            f"missing={missing}, extra={unexpected_extra}"
        )
    if any(item.get("is_directory") is not False for item in listing):
        raise ValueError("latency remote result tree must contain files only")
    directory_total_bytes = sum(
        _nonnegative_int(item.get("file_size"), "remote result file_size")
        for item in listing
    )
    if directory_total_bytes > PUBLICATION_LATENCY_REMOTE_DIRECTORY_MAX_BYTES:
        raise ValueError("latency remote result directory exceeds the byte cap")
    listed_total_bytes = sum(
        _required_int(listed_by_path[path], "file_size")
        for path in expected_by_path
    )
    if listed_total_bytes > PUBLICATION_LATENCY_REMOTE_TREE_MAX_BYTES:
        raise ValueError("latency remote result tree exceeds the controller byte cap")
    result_listing = listed_by_path[result_uri.removeprefix("dbfs:")]
    if result_listing.get("file_size") != len(result_bytes):
        raise ValueError("latency result seal size differs from the directory listing")

    tree_files: list[dict[str, Any]] = [
        {
            "byte_count": len(result_bytes),
            "role": "result_seal",
            "sha256": result_file_sha256,
            "uri": result_uri,
        }
    ]
    downloaded_total_bytes = len(result_bytes)
    for item in result_files:
        role = _required_string(item, "role")
        uri = _databricks_volume_uri(
            _required_string(item, "uri"),
            f"latency result file {role} URI",
        )
        expected_sha256 = _required_sha256(item, "sha256")
        raw = download_databricks_volume_file_bytes(
            config,
            uri,
            max_bytes=PUBLICATION_LATENCY_REMOTE_ARTIFACT_MAX_BYTES,
        )
        observed_sha256 = sha256(raw).hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError(f"latency result file {role} SHA-256 mismatch")
        listing_entry = listed_by_path[uri.removeprefix("dbfs:")]
        if listing_entry.get("file_size") != len(raw):
            raise ValueError(
                f"latency result file {role} size differs from the directory listing"
            )
        downloaded_total_bytes += len(raw)
        if downloaded_total_bytes > PUBLICATION_LATENCY_REMOTE_TREE_MAX_BYTES:
            raise ValueError(
                "latency downloaded result tree exceeds the controller byte cap"
            )
        _store_publication_latency_controller_cas_bytes(controller_cas_root, raw)
        tree_files.append(
            {
                "byte_count": len(raw),
                "role": role,
                "sha256": observed_sha256,
                "uri": uri,
            }
        )
    if downloaded_total_bytes != listed_total_bytes:
        raise ValueError("latency result tree byte closure differs from its listing")
    final_listing = list_databricks_volume_directory(
        config,
        directory_uri,
        max_entries=PUBLICATION_LATENCY_REMOTE_DIRECTORY_MAX_FILES,
    )
    if final_listing != listing:
        raise ValueError("latency remote result directory changed during collection")
    tree = {
        "auxiliary_files": [
            {
                "byte_count": _required_int(listed_by_path[path], "file_size"),
                "uri": "dbfs:" + path,
            }
            for path in extra
        ],
        "directory_uri": directory_uri,
        "file_count": len(tree_files),
        "files": tree_files,
        "total_bytes": downloaded_total_bytes,
    }
    return _PublicationLatencyRemoteResult(
        record=MappingProxyType(result),
        result_tree=MappingProxyType(tree),
        result_file_sha256=result_file_sha256,
        result_file_byte_count=len(result_bytes),
        result_tree_sha256=_canonical_sha256(tree),
        result_tree_file_count=len(tree_files),
        result_tree_total_bytes=downloaded_total_bytes,
    )


def collect_publication_latency_launch_wave(
    config: DatabricksWorkspaceConfig,
    *,
    execution_plan_record: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
    ledger_path: str | Path,
    wave_index: int,
    submission_authorization: PublicationLatencyWaveSubmissionAuthorization,
    controller_cas_root: str | Path,
) -> tuple[dict[str, Any], PublicationLatencyWaveAuthorization]:
    """Reconcile one completed wave so the next deterministic wave may launch."""

    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
        source_closure_authorization,
    )
    validate_publication_latency_execution_sources(
        execution_plan_record,
        handoff_serving_authorization=handoff_serving_authorization,
        bf16_handoff_serving_authorization=bf16_handoff_serving_authorization,
        source_closure_authorization=source_closure_authorization,
    )
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
        remote_result = _collect_remote_publication_latency_result(
            config,
            job_record=job,
            controller_cas_root=controller_cas_root,
        )
        result = remote_result.record
        result_identity = _mapping(result, "task_identity")
        if (
            result_identity.get("cloud_run_id") != receipt.run_id
            or result_identity.get("task_run_id") != identity["task_run_id"]
        ):
            raise ValueError("wave result/control-plane identity drift")
        status_sha256 = _control_plane_status_sha256(run)
        _store_publication_latency_controller_cas_bytes(
            controller_cas_root,
            _control_plane_status_bytes(run),
        )
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
                "result_file_byte_count": remote_result.result_file_byte_count,
                "result_file_sha256": remote_result.result_file_sha256,
                "result_tree": dict(remote_result.result_tree),
                "result_tree_file_count": remote_result.result_tree_file_count,
                "result_tree_sha256": remote_result.result_tree_sha256,
                "result_tree_total_bytes": remote_result.result_tree_total_bytes,
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
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
    final_wave_authorization: PublicationLatencyWaveAuthorization,
    ledger_path: str | Path,
    controller_cas_root: str | Path,
) -> tuple[dict[str, Any], PublicationLatencyCollectionAuthorization]:
    """Join runs/get, receipt ledger, and all 115 sealed result artifacts."""

    _require_latency_launch_authorization(
        execution_plan_record,
        qualification_launch_authorization,
        handoff_serving_authorization,
        bf16_handoff_serving_authorization,
        source_closure_authorization,
    )
    validate_publication_latency_execution_sources(
        execution_plan_record,
        handoff_serving_authorization=handoff_serving_authorization,
        bf16_handoff_serving_authorization=bf16_handoff_serving_authorization,
        source_closure_authorization=source_closure_authorization,
    )
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
        remote_result = _collect_remote_publication_latency_result(
            config,
            job_record=job,
            controller_cas_root=controller_cas_root,
        )
        result = remote_result.record
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
        _store_publication_latency_controller_cas_bytes(
            controller_cas_root,
            _control_plane_status_bytes(run),
        )
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
            "result_file_byte_count": remote_result.result_file_byte_count,
            "result_file_sha256": remote_result.result_file_sha256,
            "result_tree": dict(remote_result.result_tree),
            "result_tree_file_count": remote_result.result_tree_file_count,
            "result_tree_sha256": remote_result.result_tree_sha256,
            "result_tree_total_bytes": remote_result.result_tree_total_bytes,
            "run_id": receipt.run_id,
            "submit_payload_sha256": receipt.submit_payload_sha256,
            "task_run_id": identity["task_run_id"],
        }
        terminal_receipts.append(terminal_receipt)
        results.append(dict(result))

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
            "result_tree_sha256",
            "submit_payload_sha256",
        ):
            _required_sha256(receipt, field_name)
        if receipt.get("ledger_terminal_actual_sha256") != _canonical_sha256(
            _mapping(receipt, "ledger_terminal_actual")
        ):
            raise ValueError("publication latency terminal actual digest drift")
        result_tree = _mapping(receipt, "result_tree")
        _require_exact_keys(
            result_tree,
            {
                "auxiliary_files",
                "directory_uri",
                "file_count",
                "files",
                "total_bytes",
            },
            "publication latency remote result tree",
        )
        result_file_byte_count = _positive_int(
            receipt.get("result_file_byte_count"),
            "latency result file byte count",
        )
        result_tree_file_count = _positive_int(
            receipt.get("result_tree_file_count"),
            "latency result tree file count",
        )
        result_tree_total_bytes = _positive_int(
            receipt.get("result_tree_total_bytes"),
            "latency result tree total bytes",
        )
        if (
            receipt.get("result_tree_sha256") != _canonical_sha256(result_tree)
            or result_tree_file_count != len(_mapping_sequence(result_tree, "files"))
            or result_tree_file_count
            != len(_mapping_sequence(result, "files")) + 1
            or result_tree_total_bytes
            != _positive_int(
                result_tree.get("total_bytes"),
                "latency result tree total bytes",
            )
            or result_tree.get("file_count") != result_tree_file_count
            or result_tree.get("total_bytes") != result_tree_total_bytes
            or _required_string(result_tree, "directory_uri")
            != _databricks_volume_uri(
                _required_string(_mapping(job, "output"), "directory_uri"),
                "latency collection result directory URI",
            )
        ):
            raise ValueError("publication latency result tree receipt binding drift")
        tree_files = _mapping_sequence(result_tree, "files")
        auxiliary_files = _mapping_sequence(result_tree, "auxiliary_files")
        result_directory_path = _normalized_volume_path(
            _required_string(result_tree, "directory_uri"),
            "latency result tree directory URI",
        )
        auxiliary_uris: set[str] = set()
        auxiliary_total_bytes = 0
        for auxiliary in auxiliary_files:
            _require_exact_keys(
                auxiliary,
                {"byte_count", "uri"},
                "publication latency auxiliary result file",
            )
            auxiliary_uri = _databricks_volume_uri(
                _required_string(auxiliary, "uri"),
                "publication latency auxiliary result URI",
            )
            auxiliary_path = _normalized_volume_path(
                auxiliary_uri,
                "publication latency auxiliary result URI",
            )
            auxiliary_total_bytes += _nonnegative_int(
                auxiliary.get("byte_count"),
                "publication latency auxiliary result byte count",
            )
            if (
                auxiliary_path.parent != result_directory_path
                or auxiliary_path.name
                not in PUBLICATION_LATENCY_REMOTE_AUXILIARY_FILENAMES
                or auxiliary_uri in auxiliary_uris
            ):
                raise ValueError("publication latency auxiliary result tree drift")
            auxiliary_uris.add(auxiliary_uri)
        referenced_tree_uris = {
            _databricks_volume_uri(
                _required_string(item, "uri"),
                "publication latency referenced result tree URI",
            )
            for item in tree_files
        }
        if auxiliary_uris.intersection(referenced_tree_uris):
            raise ValueError("publication latency auxiliary/referenced tree overlap")
        if (
            auxiliary_total_bytes + result_tree_total_bytes
            > PUBLICATION_LATENCY_REMOTE_DIRECTORY_MAX_BYTES
        ):
            raise ValueError("publication latency result directory exceeds byte cap")
        result_seal = tree_files[0]
        for tree_file in tree_files:
            _require_exact_keys(
                tree_file,
                {"byte_count", "role", "sha256", "uri"},
                "publication latency referenced result tree file",
            )
            _safe_id(tree_file.get("role"), "latency result tree file role")
            _required_sha256(tree_file, "sha256")
            _databricks_volume_uri(
                tree_file.get("uri"), "latency result tree file URI"
            )
        if sum(
            _positive_int(item.get("byte_count"), "latency tree file byte count")
            for item in tree_files
        ) != result_tree_total_bytes:
            raise ValueError("publication latency result tree byte sum drift")
        if (
            result_seal.get("role") != "result_seal"
            or result_seal.get("sha256") != receipt.get("result_file_sha256")
            or result_seal.get("byte_count") != result_file_byte_count
            or result_seal.get("uri")
            != _databricks_volume_uri(
                _required_string(_mapping(job, "output"), "result_uri"),
                "latency collection result URI",
            )
            or result_file_byte_count > PUBLICATION_LATENCY_REMOTE_RESULT_MAX_BYTES
            or result_tree_file_count > PUBLICATION_LATENCY_REMOTE_TREE_MAX_FILES
            or result_tree_total_bytes > PUBLICATION_LATENCY_REMOTE_TREE_MAX_BYTES
        ):
            raise ValueError("publication latency result seal tree binding drift")
        for result_file, tree_file in zip(
            _mapping_sequence(result, "files"),
            tree_files[1:],
            strict=True,
        ):
            if (
                result_file.get("role") != tree_file.get("role")
                or result_file.get("sha256") != tree_file.get("sha256")
                or _databricks_volume_uri(
                    _required_string(result_file, "uri"),
                    "latency collection referenced result URI",
                )
                != tree_file.get("uri")
                or _positive_int(
                    tree_file.get("byte_count"),
                    "latency result tree file byte count",
                )
                > PUBLICATION_LATENCY_REMOTE_ARTIFACT_MAX_BYTES
            ):
                raise ValueError("publication latency referenced result tree drift")
    if (
        len({item.get("run_id") for item in receipts}) != len(receipts)
        or len({item.get("task_run_id") for item in receipts}) != len(receipts)
        or len({item.get("cluster_id") for item in receipts}) != len(receipts)
    ):
        raise ValueError(
            "publication latency collected physical identities are not unique"
        )


def _publication_latency_estimand_projection_design() -> list[dict[str, Any]]:
    design: list[dict[str, Any]] = []
    for input_tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
        for concurrency in (1, 2, 4):
            design.append(
                {
                    "comparison_family": "method",
                    "control_cell_id": (
                        f"core-baseline_prefill-{input_tokens}-c{concurrency}"
                    ),
                    "deployment_block_count": (
                        PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS
                    ),
                    "estimand_id": f"method-{input_tokens}-c{concurrency}",
                    "example_count_per_block": (
                        len(SUPPORTED_V1_DATASETS)
                        * PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
                    ),
                    "input_tokens": input_tokens,
                    "paired_request_count": (
                        PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS
                        * PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL
                    ),
                    "request_parallelism": concurrency,
                    "setting_id": None,
                    "speedup_direction": (
                        "control_latency_divided_by_treatment_latency"
                    ),
                    "treatment_cell_id": (
                        f"core-vanilla_prefill-{input_tokens}-c{concurrency}"
                    ),
                }
            )
    for setting_id, comparison_family, _description in (
        PUBLICATION_CAMPAIGN_AUXILIARY_SETTINGS
    ):
        storage_comparison = setting_id in {"storage-ram", "storage-uc"}
        design.append(
            {
                "comparison_family": comparison_family,
                "control_cell_id": (
                    "auxiliary-storage-disk"
                    if storage_comparison
                    else "core-vanilla_prefill-16384-c4"
                ),
                "deployment_block_count": PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS,
                "estimand_id": f"auxiliary-{setting_id}",
                "example_count_per_block": (
                    len(SUPPORTED_V1_DATASETS)
                    * (
                        PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET
                        if storage_comparison
                        else PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
                    )
                ),
                "input_tokens": 16_384,
                "paired_request_count": (
                    PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS
                    * (
                        PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL
                        if storage_comparison
                        else PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL
                    )
                ),
                "request_parallelism": 4,
                "setting_id": setting_id,
                "speedup_direction": (
                    "control_latency_divided_by_treatment_latency"
                ),
                "treatment_cell_id": f"auxiliary-{setting_id}",
            }
        )
    if len(design) != 13:  # pragma: no cover - frozen constants are static.
        raise RuntimeError("latency estimand projection does not contain 13 families")
    return design


def aggregate_publication_latency_campaign(
    authorization: PublicationLatencyCollectionAuthorization,
    *,
    execution_plan_record: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
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
        source_closure_authorization,
    )
    validate_publication_latency_execution_sources(
        execution_plan_record,
        handoff_serving_authorization=handoff_serving_authorization,
        bf16_handoff_serving_authorization=bf16_handoff_serving_authorization,
        source_closure_authorization=source_closure_authorization,
    )
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
    estimand_projection_by_id = {
        _required_string(item, "estimand_id"): item
        for item in _publication_latency_estimand_projection_design()
    }

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
                **estimand_projection_by_id[
                    _required_string(spec, "estimand_id")
                ],
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
    _validate_publication_latency_estimate_records(estimates)


def _validate_publication_latency_estimate_records(
    estimates: Sequence[Mapping[str, Any]],
) -> None:
    expected_design = _publication_latency_estimand_projection_design()
    if len(estimates) != len(expected_design):
        raise ValueError("publication latency summary must contain 13 estimands")
    if [item.get("estimand_id") for item in estimates] != [
        item["estimand_id"] for item in expected_design
    ]:
        raise ValueError("publication latency summary estimand order drift")
    for estimate, expected in zip(estimates, expected_design, strict=True):
        _require_exact_keys(
            estimate,
            set(expected) | {"metrics"},
            "publication latency estimand",
        )
        if any(
            type(estimate.get(field_name)) is not type(expected_value)
            or estimate.get(field_name) != expected_value
            for field_name, expected_value in expected.items()
        ):
            raise ValueError("publication latency estimand frozen-design drift")
        metrics = _mapping(estimate, "metrics")
        _require_exact_keys(
            metrics,
            {"ttft", "time_to_completion"},
            "publication latency summary metrics",
        )
        for metric_name in ("ttft", "time_to_completion"):
            metric_record = _mapping_value(metrics[metric_name], "summary metric")
            _require_exact_keys(
                metric_record,
                {"confidence_interval_95", "geometric_mean_speedup"},
                "publication latency summary metric",
            )
            raw_point = metric_record.get("geometric_mean_speedup")
            if type(raw_point) is not float:
                raise ValueError("publication latency speedup must be a float")
            point = _finite_positive_number(raw_point, "speedup")
            interval = metric_record.get("confidence_interval_95")
            if (
                type(interval) is not list
                or len(interval) != 2
                or any(type(value) is not float for value in interval)
                or any(not math.isfinite(value) or value <= 0 for value in interval)
                or interval[0] > interval[1]
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


def _descriptive_cache_telemetry_projection(
    result_record: Mapping[str, Any],
    *,
    method_id: str,
) -> dict[str, int]:
    telemetry = _mapping(result_record, "cache_telemetry")
    projection: dict[str, int] = {}
    for field_name in _DESCRIPTIVE_CACHE_TELEMETRY_FIELDS:
        value = telemetry.get(field_name)
        if (
            field_name == "expected_backend_bytes_read"
            and method_id == "baseline_prefill"
            and field_name not in telemetry
        ):
            value = 0
        projection[field_name] = _nonnegative_int(value, field_name)
    return projection


def _pooled_descriptive_cache_telemetry(
    blocks: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        field_name: sum(
            _required_int(_mapping(block, "cache_telemetry"), field_name)
            for block in blocks
        )
        for field_name in _DESCRIPTIVE_CACHE_TELEMETRY_FIELDS
    }


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
            _mapping(result_record, "benchmark_record"),
            evidence_policy=PUBLICATION_LATENCY_COMPONENT_EVIDENCE_POLICY,
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
        cache_telemetry = _descriptive_cache_telemetry_projection(
            result_record,
            method_id=_required_string(descriptor, "method_id"),
        )
        all_ttft.extend(ttft)
        all_ttc.extend(ttc)
        all_decode.extend(decode)
        block_values.append(
            {
                "cache_telemetry": cache_telemetry,
                "deployment_block": _required_int(descriptor, "deployment_block"),
                "job_id": job_id,
                "observation_count": len(ttft),
                "configured_closed_loop_concurrency": configured_concurrency,
                "gpu_utilization_sample_count": resource.sample_count,
                "mean_gpu_utilization_percent": float(
                    resource.mean_gpu_utilization_percent
                ),
                "p50_decode_tokens_per_second": _empirical_nearest_rank(decode, 0.50),
                "p50_time_to_completion_seconds": _empirical_nearest_rank(ttc, 0.50),
                "p50_ttft_seconds": _empirical_nearest_rank(ttft, 0.50),
                "p95_time_to_completion_seconds": _empirical_nearest_rank(ttc, 0.95),
                "p95_ttft_seconds": _empirical_nearest_rank(ttft, 0.95),
                "peak_gpu_process_memory_bytes": resource.peak_gpu_process_memory_bytes,
                "peak_gpu_utilization_percent": float(
                    resource.peak_gpu_utilization_percent
                ),
                "peak_host_memory_used_bytes": resource.peak_host_memory_used_bytes,
                "peak_process_tree_rss_bytes": resource.peak_process_tree_rss_bytes,
            }
        )
    assert concurrency is not None
    first = descriptors[ordered_ids[0]]
    gpu_utilization_sample_count = sum(item.sample_count for item in resources)
    record: dict[str, Any] = {
        "cache_telemetry": _pooled_descriptive_cache_telemetry(block_values),
        "cell_id": cell_id,
        "cell_kind": cell_kind,
        "cell_sha256": "",
        "comparison_family": first.get("comparison_family"),
        "input_tokens": _required_int(first, "input_tokens"),
        "method_id": _required_string(first, "method_id"),
        "observation_count": len(all_ttft),
        "configured_closed_loop_concurrency": concurrency,
        "gpu_utilization_sample_count": gpu_utilization_sample_count,
        "mean_gpu_utilization_percent": sum(
            float(item.mean_gpu_utilization_percent) * item.sample_count
            for item in resources
        )
        / gpu_utilization_sample_count,
        "p50_decode_tokens_per_second": _empirical_nearest_rank(all_decode, 0.50),
        "p50_time_to_completion_seconds": _empirical_nearest_rank(all_ttc, 0.50),
        "p50_ttft_seconds": _empirical_nearest_rank(all_ttft, 0.50),
        "p95_time_to_completion_seconds": _empirical_nearest_rank(all_ttc, 0.95),
        "p95_ttft_seconds": _empirical_nearest_rank(all_ttft, 0.95),
        "peak_gpu_process_memory_bytes": max(
            item.peak_gpu_process_memory_bytes for item in resources
        ),
        "peak_gpu_utilization_percent": max(
            float(item.peak_gpu_utilization_percent) for item in resources
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
        "cache_telemetry",
        "configured_closed_loop_concurrency",
        "gpu_utilization_sample_count",
        "mean_gpu_utilization_percent",
        "observation_count",
        "p50_decode_tokens_per_second",
        "p50_time_to_completion_seconds",
        "p50_ttft_seconds",
        "p95_time_to_completion_seconds",
        "p95_ttft_seconds",
        "peak_gpu_process_memory_bytes",
        "peak_gpu_utilization_percent",
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
    cell_id = record.get("cell_id")
    expected_coordinates = (
        _publication_latency_descriptive_cell_coordinates().get(cell_id)
        if isinstance(cell_id, str)
        else None
    )
    if expected_coordinates is None or any(
        type(record.get(field_name)) is not type(expected_value)
        or record.get(field_name) != expected_value
        for field_name, expected_value in expected_coordinates.items()
    ):
        raise ValueError("descriptive cell frozen-design coordinate drift")
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
    expected_deployment_blocks = list(
        range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)
    )
    if [block.get("deployment_block") for block in blocks] != (
        expected_deployment_blocks
    ):
        raise ValueError("descriptive cell block identity/order drift")
    for block, deployment_block in zip(
        blocks,
        expected_deployment_blocks,
        strict=True,
    ):
        _positive_int(block.get("deployment_block"), "deployment_block")
        job_id = _safe_id(block.get("job_id"), "job_id")
        if job_id != _publication_latency_descriptive_physical_job_id(
            expected_coordinates,
            deployment_block=deployment_block,
        ):
            raise ValueError("descriptive cell frozen physical job identity drift")
    if record.get("observation_count") != 1_280 or any(
        block.get("observation_count") != 256 for block in blocks
    ):
        raise ValueError("descriptive cell frozen observation-count drift")
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
            "gpu_utilization_sample_count",
        ):
            _positive_int(container.get(field_name), field_name)
        for field_name in (
            "peak_gpu_process_memory_bytes",
            "peak_host_memory_used_bytes",
            "peak_process_tree_rss_bytes",
        ):
            _nonnegative_int(container.get(field_name), field_name)
        mean_gpu_utilization = _gpu_utilization_percentage(
            container.get("mean_gpu_utilization_percent"),
            "mean_gpu_utilization_percent",
        )
        peak_gpu_utilization = _gpu_utilization_percentage(
            container.get("peak_gpu_utilization_percent"),
            "peak_gpu_utilization_percent",
        )
        if mean_gpu_utilization > peak_gpu_utilization:
            raise ValueError("mean GPU utilization exceeds its physical peak")
        _validate_descriptive_cache_telemetry(container)
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
    pooled_gpu_sample_count = sum(
        _required_int(block, "gpu_utilization_sample_count") for block in blocks
    )
    if record.get("gpu_utilization_sample_count") != pooled_gpu_sample_count:
        raise ValueError("descriptive cell pooled GPU sample-count drift")
    weighted_mean_gpu_utilization = sum(
        cast(float, block["mean_gpu_utilization_percent"])
        * _required_int(block, "gpu_utilization_sample_count")
        for block in blocks
    ) / pooled_gpu_sample_count
    if record.get("mean_gpu_utilization_percent") != weighted_mean_gpu_utilization:
        raise ValueError("descriptive cell pooled weighted GPU mean drift")
    if record.get("peak_gpu_utilization_percent") != max(
        cast(float, block["peak_gpu_utilization_percent"]) for block in blocks
    ):
        raise ValueError("descriptive cell pooled GPU utilization peak drift")
    if dict(_mapping(record, "cache_telemetry")) != (
        _pooled_descriptive_cache_telemetry(blocks)
    ):
        raise ValueError("descriptive cell pooled cache telemetry sum drift")
    for container in (record, *blocks):
        _validate_descriptive_cache_claim(
            _mapping(container, "cache_telemetry"),
            method_id=cast(str, record["method_id"]),
            observation_count=_required_int(container, "observation_count"),
            setting_id=cast(str | None, record.get("setting_id")),
        )
    for peak_field in (
        "peak_gpu_process_memory_bytes",
        "peak_host_memory_used_bytes",
        "peak_process_tree_rss_bytes",
    ):
        if record.get(peak_field) != max(
            _required_int(block, peak_field) for block in blocks
        ):
            raise ValueError("descriptive cell pooled resource peak drift")


def _publication_latency_descriptive_cell_coordinates() -> dict[str, dict[str, Any]]:
    coordinates: dict[str, dict[str, Any]] = {}
    for input_tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
        for concurrency in (1, 2, 4):
            for method_id in ("baseline_prefill", "vanilla_prefill"):
                cell_id = f"core-{method_id}-{input_tokens}-c{concurrency}"
                coordinates[cell_id] = {
                    "cell_kind": "core_pooled_five_blocks",
                    "comparison_family": None,
                    "input_tokens": input_tokens,
                    "method_id": method_id,
                    "request_parallelism": concurrency,
                    "setting_id": None,
                }
    auxiliary_families = {
        "storage-disk": "storage",
        **{
            setting_id: family
            for setting_id, family, _description in (
                PUBLICATION_CAMPAIGN_AUXILIARY_SETTINGS
            )
        },
    }
    for setting_id in _DESCRIPTIVE_AUXILIARY_SETTING_IDS:
        cell_id = f"auxiliary-{setting_id}"
        coordinates[cell_id] = {
            "cell_kind": "auxiliary_pooled_five_blocks",
            "comparison_family": auxiliary_families[setting_id],
            "input_tokens": 16_384,
            "method_id": "vanilla_prefill",
            "request_parallelism": 4,
            "setting_id": setting_id,
        }
    return coordinates


def _publication_latency_descriptive_physical_job_id(
    coordinates: Mapping[str, Any],
    *,
    deployment_block: int,
) -> str:
    block = _positive_int(deployment_block, "deployment_block")
    setting_id = coordinates.get("setting_id")
    if setting_id is not None:
        return f"block-{block:02d}-{_safe_id(setting_id, 'setting_id')}"
    input_tokens = _positive_int(coordinates.get("input_tokens"), "input_tokens")
    request_parallelism = _positive_int(
        coordinates.get("request_parallelism"),
        "request_parallelism",
    )
    method_id = coordinates.get("method_id")
    if method_id == "baseline_prefill":
        method_label = "baseline"
    elif method_id == "vanilla_prefill":
        method_label = "vanilla"
    else:  # pragma: no cover - coordinates are generated from the frozen design.
        raise ValueError("descriptive cell method is outside the frozen design")
    return (
        f"block-{block:02d}-{input_tokens // 1024}k-"
        f"c{request_parallelism}-{method_label}"
    )


def _gpu_utilization_percentage(value: Any, field_name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError(f"{field_name} must be a finite float in [0, 100]")
    return value


def _validate_descriptive_cache_telemetry(
    container: Mapping[str, Any],
) -> Mapping[str, Any]:
    telemetry = _mapping(container, "cache_telemetry")
    _require_exact_keys(
        telemetry,
        set(_DESCRIPTIVE_CACHE_TELEMETRY_FIELDS),
        "publication latency descriptive cache telemetry",
    )
    for field_name in _DESCRIPTIVE_CACHE_TELEMETRY_FIELDS:
        _nonnegative_int(telemetry.get(field_name), field_name)
    return telemetry


def _validate_descriptive_cache_claim(
    telemetry: Mapping[str, Any],
    *,
    method_id: str,
    observation_count: int,
    setting_id: str | None,
) -> None:
    observations = _positive_int(observation_count, "observation_count")
    byte_fields = {"backend_bytes_read", "expected_backend_bytes_read"}
    validated_telemetry: dict[str, int] = {}
    for field_name in _DESCRIPTIVE_CACHE_TELEMETRY_FIELDS:
        value = _nonnegative_int(telemetry.get(field_name), field_name)
        validated_telemetry[field_name] = value
        if field_name not in byte_fields and value > observations:
            raise ValueError(
                f"{field_name} exceeds descriptive observation_count"
            )
    backend_bytes_read = validated_telemetry["backend_bytes_read"]
    expected_backend_bytes_read = validated_telemetry[
        "expected_backend_bytes_read"
    ]
    if method_id == "baseline_prefill":
        if any(telemetry[field_name] != 0 for field_name in telemetry):
            raise ValueError("Baseline descriptive cache telemetry must be zero")
        return
    if telemetry.get("load_count") != observations:
        raise ValueError("Vanilla descriptive cache load-count drift")
    if telemetry.get("storage_materialization_count") != telemetry.get(
        "payload_cache_miss_count"
    ):
        raise ValueError("descriptive cache materialization/miss drift")
    if setting_id == "storage-ram":
        if (
            telemetry.get("payload_cache_hit_count") != observations
            or telemetry.get("payload_cache_miss_count") != 0
            or telemetry.get("backend_bytes_read") != 0
            or telemetry.get("expected_backend_bytes_read") != 0
            or telemetry.get("cold_read_attested_count") != 0
            or telemetry.get("eviction_requested_count") != 0
            or telemetry.get("eviction_succeeded_count") != 0
            or telemetry.get("mounted_path_load_count") != 0
        ):
            raise ValueError("RAM descriptive cache telemetry claim drift")
        return
    if setting_id == "storage-uc":
        if (
            telemetry.get("eviction_requested_count") != observations
            or telemetry.get("eviction_succeeded_count") != observations
            or telemetry.get("mounted_path_load_count") != observations
            or telemetry.get("payload_cache_hit_count") != 0
            or telemetry.get("payload_cache_miss_count") != 0
            or telemetry.get("storage_materialization_count") != 0
            or backend_bytes_read <= 0
            or backend_bytes_read != expected_backend_bytes_read
        ):
            raise ValueError("UC descriptive cache telemetry claim drift")
        return
    if (
        telemetry.get("cold_read_attested_count") != observations
        or telemetry.get("eviction_requested_count") != observations
        or telemetry.get("eviction_succeeded_count") != observations
        or telemetry.get("mounted_path_load_count") != 0
        or telemetry.get("payload_cache_hit_count") != 0
        or telemetry.get("payload_cache_miss_count") != 0
        or telemetry.get("storage_materialization_count") != 0
        or backend_bytes_read <= 0
        or backend_bytes_read != expected_backend_bytes_read
    ):
        raise ValueError("cold descriptive cache telemetry claim drift")


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
            evidence_policy=PUBLICATION_LATENCY_COMPONENT_EVIDENCE_POLICY,
        )
        treatment = benchmark_run_result_from_record(
            _mapping(result_by_job[treatment_id], "benchmark_record"),
            evidence_policy=PUBLICATION_LATENCY_COMPONENT_EVIDENCE_POLICY,
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
    if payload.get("run_name") != (
        f"cachet-publication-latency-{_required_string(job_record, 'job_id')}"
    ):
        raise ValueError("publication latency run name drift")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("publication latency requires exactly one isolated task")
    task = _mapping_value(tasks[0], "publication latency task")
    if set(task) != {
        "max_retries",
        "new_cluster",
        "spark_python_task",
        "task_key",
        "timeout_seconds",
    }:
        raise ValueError("publication latency task schema is open")
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
        or cluster.get("single_user_name")
        != _validated_single_user_name(runtime.get("single_user_name"))
    ):
        raise ValueError("publication latency cluster hardware drift")
    aws_attributes = _mapping(cluster, "aws_attributes")
    if aws_attributes != {
        "availability": runtime.get("availability"),
        "zone_id": runtime.get("zone_id"),
    }:
        raise ValueError("publication latency cluster availability/zone drift")
    python_task = _mapping(task, "spark_python_task")
    if set(python_task) != {"parameters", "python_file"}:
        raise ValueError("publication latency Python task schema is open")
    artifacts_by_role = {
        _required_string(item, "role"): item
        for item in _mapping_sequence(job_record, "artifact_files")
    }
    runner = artifacts_by_role["runner"]
    package_wheel = artifacts_by_role["package_wheel"]
    runner_uri = _required_string(runner, "uri")
    if python_task.get("python_file") != runner_uri:
        raise ValueError("publication latency runner python_file drift")
    expected_spark_environment = {
        VLLM_PATCHED_WHEEL_SHA256_ENV: _required_sha256(
            artifacts_by_role["patched_vllm_wheel"],
            "sha256",
        ),
        VLLM_PATCHED_WHEEL_URI_ENV: _required_string(
            artifacts_by_role["patched_vllm_wheel"],
            "uri",
        ),
        "DOCUMENT_KV_EVICT_PAGE_CACHE": (
            "0"
            if _mapping(job_record, "cell").get("setting_id") == "storage-ram"
            else "1"
        ),
    }
    if _mapping(cluster, "spark_env_vars") != expected_spark_environment:
        raise ValueError("publication latency Spark environment drift")
    parameters = python_task.get("parameters")
    if not isinstance(parameters, list) or any(
        not isinstance(item, str) for item in parameters
    ):
        raise ValueError("publication latency runner parameters are invalid")
    expected_parameters = [
        "--job-record-json",
        _canonical_json(job_record),
        "--expected-job-sha256",
        _required_sha256(job_record, "closed_record_sha256"),
        "--runner-uri",
        runner_uri,
        "--runner-sha256",
        _required_sha256(runner, "sha256"),
        "--package-wheel-uri",
        _required_string(package_wheel, "uri"),
        "--package-wheel-sha256",
        _required_sha256(package_wheel, "sha256"),
        "--cloud-run-id",
        _DATABRICKS_JOB_RUN_ID_TEMPLATE,
        "--task-run-id",
        _DATABRICKS_TASK_RUN_ID_TEMPLATE,
    ]
    if parameters != expected_parameters:
        raise ValueError("publication latency runner parameter binding drift")


def validate_publication_latency_execution_sources(
    execution_plan_record: Mapping[str, Any],
    *,
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
) -> None:
    """Replay the closed plan and compact authorities without mounted paths."""

    validate_publication_latency_execution_plan_record(execution_plan_record)
    sources = _mapping(execution_plan_record, "sources")
    qualification = _mapping(sources, "qualification")
    qualification_evidence = _mapping(qualification, "evidence")
    pins = _gpu_qualification_artifact_pins_v2_from_record(
        _mapping(qualification, "artifact_pins")
    )
    handoff = _mapping(sources, "handoff_generation")
    execution = _mapping(handoff, "execution")
    authenticated = require_q8_handoff_remote_closure_authorization(
        handoff_serving_authorization,
        expected_output_root_uri=_required_string(handoff, "output_root_uri"),
        expected_execution_file_sha256=_required_sha256(execution, "sha256"),
        expected_input_bundle_sha256=pins.input_bundle_sha256,
        expected_qualification_closed_record_sha256=_required_sha256(
            qualification_evidence,
            "closed_record_sha256",
        ),
    )
    authenticated_handoff_reconciliation = _mapping(
        authenticated.execution_record, "ledger_reconciliation"
    )
    handoff_authorization_binding = _mapping(handoff, "authorization")
    if (
        authenticated.execution_record.get("closed_record_sha256")
        != execution.get("closed_record_sha256")
        or authenticated.execution_file_sha256 != execution.get("sha256")
        or authenticated.execution_uri != execution.get("uri")
        or authenticated.execution_record.get("execution_mode")
        != PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED
        or handoff_authorization_binding.get("causal_closure_sha256")
        != authenticated.causal_closure_sha256
        or handoff_authorization_binding.get("remote_closure")
        != _remote_handoff_authorization_binding(authenticated)
        or handoff_authorization_binding.get("ledger_id")
        != authenticated_handoff_reconciliation.get("ledger_id")
    ):
        raise ValueError("distributed handoff source binding drift")
    if _mapping(authenticated.execution_record, "generator_hardware").get(
        "qualification_closed_record_sha256"
    ) != qualification_evidence.get("closed_record_sha256"):
        raise ValueError("distributed handoff qualification binding drift")

    bf16 = _mapping(sources, "bf16_handoff")
    bf16_execution_binding = _mapping(bf16, "execution")
    authenticated_bf16 = require_bf16_handoff_remote_closure_authorization(
        bf16_handoff_serving_authorization,
        expected_output_root_uri=_required_string(bf16, "output_root_uri"),
        expected_execution_file_sha256=_required_sha256(
            bf16_execution_binding,
            "sha256",
        ),
        expected_input_bundle_sha256=pins.input_bundle_sha256,
        expected_qualification_closed_record_sha256=_required_sha256(
            qualification_evidence,
            "closed_record_sha256",
        ),
    )
    bf16_accounting_binding = _mapping(bf16, "accounting")
    bf16_authorization_binding = _mapping(bf16, "authorization")
    authenticated_bf16_reconciliation = _mapping(
        authenticated_bf16.execution_record, "ledger_reconciliation"
    )
    if (
        authenticated_bf16.execution_record.get("closed_record_sha256")
        != bf16_execution_binding.get("closed_record_sha256")
        or authenticated_bf16.execution_record.get("execution_mode")
        != bf16_execution_binding.get("execution_mode")
        or authenticated_bf16.execution_file_sha256
        != bf16_execution_binding.get("sha256")
        or authenticated_bf16.execution_uri != bf16_execution_binding.get("uri")
        or _canonical_sha256(authenticated_bf16_reconciliation)
        != bf16.get("ledger_reconciliation_sha256")
        or bf16_authorization_binding.get("causal_closure_sha256")
        != authenticated_bf16.causal_closure_sha256
        or bf16_authorization_binding.get("remote_closure")
        != _remote_handoff_authorization_binding(authenticated_bf16)
        or bf16_authorization_binding.get("ledger_id")
        != authenticated_bf16_reconciliation.get("ledger_id")
        or _canonical_sha256(
            _mapping(authenticated_bf16.execution_record, "accounting")
        )
        != bf16_accounting_binding.get("closed_sha256")
    ):
        raise ValueError("distributed BF16 generation source binding drift")
    bf16_manifest_binding = _mapping(bf16, "manifest")
    remote_bf16_manifests = _mapping_sequence(
        authenticated_bf16.result_record,
        "manifests",
    )
    if (
        len(remote_bf16_manifests) != 1
        or authenticated_bf16.manifest_records[0].get("closed_record_sha256")
        != bf16_manifest_binding.get("closed_record_sha256")
        or remote_bf16_manifests[0].get("file_sha256")
        != bf16_manifest_binding.get("sha256")
        or remote_bf16_manifests[0].get("uri")
        != bf16_manifest_binding.get("uri")
        or remote_bf16_manifests[0].get("source_root_uri")
        != bf16.get("source_root_uri")
    ):
        raise ValueError("BF16 handoff source binding drift")
    expected_source_semantic = _source_closure_semantic_from_plan_sources(sources)
    authenticated_source = require_publication_latency_source_closure_authorization(
        source_closure_authorization,
        expected_final_artifacts=_final_artifacts_from_record(
            _mapping(sources, "final_artifacts")
        ),
        expected_qualification_artifact_pins=pins,
        expected_semantic=expected_source_semantic,
        expected_predecessor_prefix=bf16_handoff_serving_authorization.ledger_prefix,
        expected_q8_handoff_authorization=handoff_serving_authorization,
        expected_bf16_handoff_authorization=bf16_handoff_serving_authorization,
    )
    if _source_closure_authorization_binding(authenticated_source) != dict(
        _mapping(sources, "source_closure")
    ):
        raise ValueError("publication latency source-closure plan binding drift")
    qualification_authorization_binding = _mapping(
        qualification,
        "authorization",
    )
    for label, authorization, binding in (
        (
            "Q8",
            handoff_serving_authorization,
            handoff_authorization_binding,
        ),
        (
            "BF16",
            bf16_handoff_serving_authorization,
            bf16_authorization_binding,
        ),
    ):
        if (
            authorization.ledger_id != binding.get("ledger_id")
            or authorization.ledger_path_sha256
            != binding.get("ledger_path_sha256")
            or authorization.ledger_prefix.to_record()
            != binding.get("ledger_prefix")
            or authorization.predecessor_prefix.to_record()
            != binding.get("predecessor_prefix")
            or authorization.producer_batch_prefix.to_record()
            != binding.get("producer_batch_prefix")
        ):
            raise ValueError(f"{label} remote source ledger binding drift")
    _require_remote_handoff_phase_order(
        qualification_prefix=databricks_ledger_prefix_from_record(
            _mapping(qualification_authorization_binding, "ledger_prefix")
        ),
        q8_predecessor_prefix=handoff_serving_authorization.predecessor_prefix,
        q8_terminal_prefix=handoff_serving_authorization.ledger_prefix,
        bf16_predecessor_prefix=(
            bf16_handoff_serving_authorization.predecessor_prefix
        ),
    )
    if source_closure_authorization.predecessor_prefix != (
        bf16_handoff_serving_authorization.ledger_prefix
    ):
        raise ValueError("source closure does not extend BF16 in phase order")


def _native_runtime_v2_from_job_record(
    job_record: Mapping[str, Any],
) -> VLLMNativeRuntimeBundleV2:
    artifact_files = {
        _required_string(item, "role"): item
        for item in _mapping_sequence(job_record, "artifact_files")
    }
    return VLLMNativeRuntimeBundleV2(
        runtime_lock_uri=_required_string(artifact_files["runtime_lock"], "uri"),
        runtime_lock_sha256=_required_sha256(
            artifact_files["runtime_lock"], "sha256"
        ),
        patched_vllm_wheel_uri=_required_string(
            artifact_files["patched_vllm_wheel"], "uri"
        ),
        patched_vllm_wheel_sha256=_required_sha256(
            artifact_files["patched_vllm_wheel"], "sha256"
        ),
        patched_flashinfer_wheel_uri=_required_string(
            artifact_files["patched_flashinfer_wheel"], "uri"
        ),
        patched_flashinfer_wheel_sha256=_required_sha256(
            artifact_files["patched_flashinfer_wheel"], "sha256"
        ),
        runtime_closure_manifest_uri=_required_string(
            artifact_files["runtime_closure_manifest"], "uri"
        ),
        runtime_closure_manifest_sha256=_required_sha256(
            artifact_files["runtime_closure_manifest"], "sha256"
        ),
        package_wheel_uri=_required_string(artifact_files["package_wheel"], "uri"),
        package_wheel_sha256=_required_sha256(
            artifact_files["package_wheel"], "sha256"
        ),
    )


def _validate_native_runtime_v2_attestation_binding(
    value: Mapping[str, Any],
    *,
    bundle: VLLMNativeRuntimeBundleV2,
) -> None:
    validate_gpu_qualification_v2_runtime_attestation(value)
    expected_urls = {
        "vllm_direct_url": bundle.local_path("patched_vllm_wheel").resolve().as_uri(),
        "flashinfer_direct_url": (
            bundle.local_path("patched_flashinfer_wheel").resolve().as_uri()
        ),
    }
    if any(value.get(name) != expected for name, expected in expected_urls.items()):
        raise ValueError("native-v2 runtime attestation artifact origin drift")


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
    native_runtime_v2 = _native_runtime_v2_from_job_record(job_record)
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
        "runtime_version": VLLM_RUNTIME_BASE_LOCK_SHA256,
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
        benchmark_evidence_policy=PUBLICATION_LATENCY_COMPONENT_EVIDENCE_POLICY,
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
        package_install_spec=native_runtime_v2.package_wheel_uri,
        native_runtime_v2=native_runtime_v2,
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
    native_runtime_v2 = config.native_runtime_v2
    if native_runtime_v2 is None:  # pragma: no cover - production config enforces it.
        raise RuntimeError("publication latency requires native-v2 runtime artifacts")
    native_runtime_attestation = _mapping(
        metadata, "native_runtime_v2_attestation"
    )
    _validate_native_runtime_v2_attestation_binding(
        native_runtime_attestation,
        bundle=native_runtime_v2,
    )
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
            "native_runtime_v2": native_runtime_v2.to_record(),
            "native_runtime_v2_attestation": dict(native_runtime_attestation),
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
    _require_exact_keys(
        runtime,
        {
            "metadata_sha256",
            "native_runtime_v2",
            "native_runtime_v2_attestation",
            "strict_runtime_closure",
            "vllm_version",
        },
        "publication latency native-v2 runtime attestation",
    )
    native_runtime_v2 = _native_runtime_v2_from_job_record(expected_job_record)
    if (
        runtime.get("metadata_sha256")
        != next(item["sha256"] for item in files if item["role"] == "metadata")
        or dict(_mapping(runtime, "native_runtime_v2"))
        != native_runtime_v2.to_record()
        or runtime.get("strict_runtime_closure") is not True
        or runtime.get("vllm_version") != PUBLICATION_CAMPAIGN_ENGINE_VERSION
    ):
        raise ValueError("publication latency result runtime attestation drift")
    _validate_native_runtime_v2_attestation_binding(
        _mapping(runtime, "native_runtime_v2_attestation"),
        bundle=native_runtime_v2,
    )
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


def _require_publication_latency_component_evidence_gate(
    record: Mapping[str, Any],
    *,
    result: BenchmarkRunResult,
) -> None:
    artifact_identities, cache_state_attestations = (
        benchmark_gate_inputs_from_record(record)
    )
    expected_gate = benchmark_evidence_gate_to_record(
        evaluate_benchmark_evidence_gate(
            result,
            policy=PUBLICATION_LATENCY_COMPONENT_EVIDENCE_POLICY,
            cache_state_attestations=cache_state_attestations,
            artifact_identities=artifact_identities,
            benchmark_payload_digest=benchmark_record_payload_digest(record),
        )
    )
    if dict(_mapping(record, "evidence_gate")) != expected_gate:
        raise ValueError(
            "benchmark component evidence gate does not match recomputed evidence"
        )
    if (
        expected_gate.get("ok") is not True
        or expected_gate.get("policy")
        != PUBLICATION_LATENCY_COMPONENT_EVIDENCE_POLICY
    ):
        raise ValueError("benchmark component evidence gate did not pass")


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
    result = benchmark_run_result_from_record(
        record,
        evidence_policy=PUBLICATION_LATENCY_COMPONENT_EVIDENCE_POLICY,
    )
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
        or len(manifest.arms) != 1
        or manifest.arms[0].arm_id != expected_arm
        or manifest.arms[0].request_customization_digest
        != PUBLICATION_LATENCY_REQUEST_CUSTOMIZATION_DIGEST
    ):
        raise ValueError("benchmark fixed decoding contract drift")
    if any(
        not measurement.ok
        or measurement.arm_id != expected_arm
        or measurement.metadata.get("prompt_text_mode") != "logical"
        or measurement.metadata.get("request_payload_add_special_tokens") != "false"
        or measurement.completion_tokens != PUBLICATION_LATENCY_MAX_OUTPUT_TOKENS
        or measurement.ttft_seconds <= 0
        or measurement.time_to_completion_seconds < measurement.ttft_seconds
        for measurement in result.measurements
    ):
        raise ValueError(
            "benchmark contains failed or invalid fixed-decode measurements"
        )
    _require_publication_latency_component_evidence_gate(record, result=result)
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
    native_runtime_v2 = _native_runtime_v2_from_job_record(job_record)
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
    if (
        metadata.get("strict_runtime_closure") is not True
        or metadata.get("vllm_runtime_lock_verification_scope") != "final-runtime"
    ):
        raise ValueError("vLLM metadata does not attest strict runtime closure")
    if dict(_mapping(metadata, "native_runtime_v2")) != (
        native_runtime_v2.to_record()
    ):
        raise ValueError("vLLM metadata native-v2 runtime bundle drift")
    native_attestation = _mapping(metadata, "native_runtime_v2_attestation")
    _validate_native_runtime_v2_attestation_binding(
        native_attestation,
        bundle=native_runtime_v2,
    )
    if dict(_mapping(metadata, "vllm_runtime_lock_verification")) != dict(
        native_attestation
    ):
        raise ValueError("vLLM native-v2 runtime verification binding drift")
    runtime_lock = _mapping(metadata, "vllm_runtime_lock")
    if runtime_lock != {
        "platform": "CPython 3.11 / Linux x86_64 / glibc 2.35",
        "runtime_contract": "native-v2",
        "sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "uri": native_runtime_v2.runtime_lock_uri,
    }:
        raise ValueError("vLLM metadata native-v2 runtime lock drift")
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
        benchmark_record,
        evidence_policy=PUBLICATION_LATENCY_COMPONENT_EVIDENCE_POLICY,
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
    uc_mounted_backend = (
        policy.get("host_cache_state")
        == "mounted_path_evicted_backend_cache_unproven"
    )
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
            uc_mounted_backend
            and attestation.get("payload_cache_hit") is False
            and payload.get("payload_cache_enabled") is False
            and hits == 0
            and bytes_read > 0
            and bytes_read == expected_stored_bytes
            and attestation.get("source") == "local_path"
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
    handoff_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
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
        or source_closure_authorization.ledger_id != campaign_ledger_id
    ):
        raise ValueError(
            "GPU qualification, Q8, BF16, and source-closure authority must "
            "share the execution plan campaign ledger"
        )
    if {
        qualification_authorization.ledger_path_sha256,
        handoff_authorization.ledger_path_sha256,
        bf16_handoff_authorization.ledger_path_sha256,
        source_closure_authorization.ledger_path_sha256,
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
    authenticated_handoff = require_q8_handoff_remote_closure_authorization(
        handoff_authorization,
        expected_output_root_uri=_required_string(
            handoff_binding,
            "output_root_uri",
        ),
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
        or _remote_handoff_authorization_binding(authenticated_handoff)
        != handoff_authorization_binding.get("remote_closure")
        or authenticated_handoff.execution_record.get("closed_record_sha256")
        != handoff_execution_binding.get("closed_record_sha256")
        or authenticated_handoff.execution_file_sha256
        != handoff_execution_binding.get("sha256")
        or authenticated_handoff.execution_uri
        != _required_string(handoff_execution_binding, "uri")
        or authenticated_handoff.output_root_uri
        != _required_string(handoff_binding, "output_root_uri")
    ):
        raise ValueError("Q8 handoff serving authorization binding drift")

    bf16_binding = _mapping(sources, "bf16_handoff")
    bf16_manifest_binding = _mapping(bf16_binding, "manifest")
    authenticated_bf16 = require_bf16_handoff_remote_closure_authorization(
        bf16_handoff_authorization,
        expected_output_root_uri=_required_string(
            bf16_binding,
            "output_root_uri",
        ),
        expected_execution_file_sha256=_required_sha256(
            _mapping(bf16_binding, "execution"),
            "sha256",
        ),
        expected_input_bundle_sha256=_required_sha256(
            artifact_pins, "input_bundle_sha256"
        ),
        expected_qualification_closed_record_sha256=_required_sha256(
            qualification_evidence,
            "closed_record_sha256",
        ),
    )
    bf16_authorization_binding = _mapping(bf16_binding, "authorization")
    bf16_execution_binding = _mapping(bf16_binding, "execution")
    authenticated_bf16_manifests = _mapping_sequence(
        authenticated_bf16.result_record,
        "manifests",
    )
    if (
        len(authenticated_bf16_manifests) != 1
        or bf16_handoff_authorization.causal_closure_sha256
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
        or _remote_handoff_authorization_binding(authenticated_bf16)
        != bf16_authorization_binding.get("remote_closure")
        or authenticated_bf16.execution_record.get("closed_record_sha256")
        != bf16_execution_binding.get("closed_record_sha256")
        or authenticated_bf16.execution_file_sha256
        != bf16_execution_binding.get("sha256")
        or authenticated_bf16.execution_uri
        != _required_string(bf16_execution_binding, "uri")
        or authenticated_bf16_manifests[0].get("file_sha256")
        != bf16_manifest_binding.get("sha256")
        or authenticated_bf16_manifests[0].get("closed_record_sha256")
        != bf16_manifest_binding.get("closed_record_sha256")
        or authenticated_bf16_manifests[0].get("portable_bundle_sha256")
        != bf16_manifest_binding.get("portable_bundle_sha256")
        or authenticated_bf16_manifests[0].get("uri")
        != bf16_manifest_binding.get("uri")
        or authenticated_bf16_manifests[0].get("source_root_uri")
        != _required_string(bf16_binding, "source_root_uri")
    ):
        raise ValueError("BF16 handoff serving authorization binding drift")
    typed_artifact_pins = _gpu_qualification_artifact_pins_v2_from_record(
        artifact_pins
    )
    authenticated_source = require_publication_latency_source_closure_authorization(
        source_closure_authorization,
        expected_final_artifacts=final_artifacts,
        expected_qualification_artifact_pins=typed_artifact_pins,
        expected_semantic=_source_closure_semantic_from_plan_sources(sources),
        expected_predecessor_prefix=bf16_handoff_authorization.ledger_prefix,
        expected_q8_handoff_authorization=handoff_authorization,
        expected_bf16_handoff_authorization=bf16_handoff_authorization,
    )
    if _source_closure_authorization_binding(authenticated_source) != dict(
        _mapping(sources, "source_closure")
    ):
        raise ValueError("source-closure launch authorization binding drift")
    _require_remote_handoff_phase_order(
        qualification_prefix=qualification_authorization.ledger_prefix,
        q8_predecessor_prefix=handoff_authorization.predecessor_prefix,
        q8_terminal_prefix=handoff_authorization.ledger_prefix,
        bf16_predecessor_prefix=bf16_handoff_authorization.predecessor_prefix,
    )
    if source_closure_authorization.predecessor_prefix != (
        bf16_handoff_authorization.ledger_prefix
    ):
        raise ValueError("source closure does not extend BF16")
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
    submitted_python_task = _mapping(submitted_task, "spark_python_task")
    observed_python_task = task.get("spark_python_task")
    if not isinstance(observed_python_task, Mapping) or dict(
        observed_python_task
    ) != dict(submitted_python_task):
        raise ValueError("latency runs/get Python task drift")
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


def _validate_source_closure_control_plane_run(
    run: Mapping[str, Any],
    *,
    submit_payload: Mapping[str, Any],
    receipt_run_id: str,
) -> dict[str, Any]:
    """Require one successful, unrepaired, exact attempt-zero c5d task."""

    parent_run_id = _databricks_id(run.get("run_id"), "runs/get run_id")
    if parent_run_id != receipt_run_id:
        raise ValueError("runs/get run ID differs from source-closure submit receipt")
    original_attempt_run_id = _databricks_id(
        run.get("original_attempt_run_id"),
        "source-closure original_attempt_run_id",
    )
    if original_attempt_run_id != parent_run_id:
        raise ValueError(
            "source-closure coordinator original attempt does not equal its parent run"
        )
    if run.get("repair_history") not in (None, []):
        raise ValueError("source-closure coordinator has repair history")
    tasks = run.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 1
        or not isinstance(tasks[0], Mapping)
    ):
        raise ValueError("source-closure coordinator must contain exactly one task")
    task = tasks[0]
    attempt_number = task.get("attempt_number")
    if type(attempt_number) is not int or attempt_number != 0:
        raise ValueError("source-closure coordinator must finish on task attempt zero")
    task_run_id = _databricks_id(task.get("run_id"), "source-closure task run_id")
    if task_run_id == parent_run_id:
        raise ValueError(
            "source-closure coordinator task run ID must differ from its parent run"
        )

    # The generic latency validator models Databricks' attempt-zero marker as zero.
    # Prove the stronger source-coordinator parent identity above before adapting it.
    normalized = json.loads(_canonical_json(run))
    normalized["original_attempt_run_id"] = 0
    identity = _validate_latency_control_plane_run(
        normalized,
        job_record={"task_key": "publication_latency_source_closure"},
        submit_payload=submit_payload,
        receipt_run_id=receipt_run_id,
    )
    submitted_cluster = _mapping(
        _mapping_value(
            _mapping_sequence(submit_payload, "tasks")[0], "source submitted task"
        ),
        "new_cluster",
    )
    if (
        submitted_cluster.get("node_type_id")
        != PUBLICATION_LATENCY_SOURCE_CLOSURE_NODE_TYPE_ID
        or submitted_cluster.get("driver_node_type_id")
        != PUBLICATION_LATENCY_SOURCE_CLOSURE_NODE_TYPE_ID
        or submitted_cluster.get("num_workers") != 0
    ):
        raise ValueError("source-closure coordinator is not one c5d.4xlarge task")
    return identity


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
    return sha256(_control_plane_status_bytes(run)).hexdigest()


def _control_plane_status_bytes(run: Mapping[str, Any]) -> bytes:
    raw = json.dumps(
        run,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(raw) > PUBLICATION_LATENCY_CONTROL_PLANE_MAX_BYTES:
        raise ValueError("latency control-plane status exceeds the compact byte cap")
    return raw


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


def _databricks_volume_uri(value: Any, field_name: str) -> str:
    uri = _durable_uri(value, field_name)
    normalized = uri if uri.startswith("dbfs:") else "dbfs:" + uri
    raw_path = normalized.removeprefix("dbfs:")
    path = PurePosixPath(raw_path)
    if (
        not raw_path.startswith("/Volumes/")
        or path.as_posix() != raw_path
        or len(path.parts) < 5
    ):
        raise ValueError(f"{field_name} must be a canonical Unity Catalog volume URI")
    return normalized


def _canonical_remote_json_record(
    raw: bytes,
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw:
        raise ValueError(f"{field_name} must be non-empty bytes")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    if raw != (_canonical_json(value) + "\n").encode("utf-8"):
        raise ValueError(f"{field_name} is not canonical newline JSON")
    return value


def _canonical_pretty_remote_json_record(
    raw: bytes,
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw:
        raise ValueError(f"{field_name} must be non-empty bytes")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or raw != _pretty_json_bytes(value):
        raise ValueError(f"{field_name} is not canonical pretty JSON")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_or_require_exact_bytes(path: Path, raw: bytes) -> None:
    _reject_existing_symlink_ancestors(path, "exclusive controller artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlink_ancestors(path, "exclusive controller artifact")
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != raw:
            raise FileExistsError(f"existing artifact differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _store_publication_latency_controller_cas_bytes(
    controller_cas_root: str | Path,
    raw: bytes,
) -> Path:
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("latency controller CAS objects must be non-empty bytes")
    root = Path(controller_cas_root).expanduser().absolute()
    _reject_existing_symlink_ancestors(root, "latency controller CAS root")
    root.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlink_ancestors(root, "latency controller CAS root")
    digest = sha256(raw).hexdigest()
    bucket = root / "sha256" / digest[:2]
    _reject_existing_symlink_ancestors(bucket, "latency controller CAS bucket")
    bucket.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlink_ancestors(bucket, "latency controller CAS bucket")
    destination = bucket / digest
    if destination.exists() or destination.is_symlink():
        _verify_regular_file_sha256(
            destination,
            digest,
            "latency controller CAS object",
        )
        if destination.stat().st_size != len(raw):
            raise ValueError("latency controller CAS object byte count drift")
        return destination
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(bucket)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _cluster_path(uri: str) -> Path:
    if uri.startswith("dbfs:/Volumes/"):
        return Path("/Volumes") / uri.removeprefix("dbfs:/Volumes/")
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


def _validated_single_user_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(
            ord(character) < 32 or 127 <= ord(character) <= 159
            for character in value
        )
    ):
        raise ValueError("single_user_name must be a normalized non-empty string")
    return value


def _publication_latency_single_user_name(
    execution_plan_record: Mapping[str, Any],
) -> str:
    sources = _mapping(execution_plan_record, "sources")
    principal = _validated_single_user_name(
        _mapping(sources, "source_closure").get("single_user_name")
    )
    runtime_principal = _validated_single_user_name(
        _mapping(execution_plan_record, "runtime_policy").get("single_user_name")
    )
    if runtime_principal != principal:
        raise ValueError("publication latency SINGLE_USER principal drift")
    return principal


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


def _normalized_volume_path(value: Any, field_name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical volume path")
    raw = value.removeprefix("dbfs:")
    if raw.startswith("/dbfs/Volumes/"):
        raw = raw.removeprefix("/dbfs")
    path = PurePosixPath(raw)
    if (
        not raw.startswith("/Volumes/")
        or path.as_posix() != raw
        or len(path.parts) < 5
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError(f"{field_name} must remain inside one Unity Catalog volume")
    return path


def _same_durable_file_location(first: str, second: str) -> bool:
    return _normalized_volume_path(first, "first durable file") == (
        _normalized_volume_path(second, "second durable file")
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_json_file_sha256_matches(
    record: Mapping[str, Any],
    expected_sha256: str,
) -> bool:
    expected = _require_sha256_value(expected_sha256, "JSON file sha256")
    compact = _canonical_json(record).encode("utf-8")
    pretty = (
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    candidates = (compact, compact + b"\n", pretty)
    return any(sha256(raw).hexdigest() == expected for raw in candidates)


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
    write_source_runner = subparsers.add_parser("write-source-closure-runner")
    write_source_runner.add_argument("--output", required=True)
    run_source_closure = subparsers.add_parser("run-source-closure")
    run_source_closure.add_argument("--request-path", required=True)
    run_source_closure.add_argument("--expected-request-file-sha256", required=True)
    run_source_closure.add_argument(
        "--expected-request-closed-record-sha256", required=True
    )
    run_source_closure.add_argument("--coordinator-run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "write-runner":
        path = write_publication_latency_runner_script(args.output)
        print(path)
        return 0
    if args.command == "write-source-closure-runner":
        path = write_publication_latency_source_closure_runner_script(args.output)
        print(path)
        return 0
    if args.command == "run-source-closure":
        result = run_publication_latency_source_closure_coordinator(
            args.request_path,
            expected_request_file_sha256=args.expected_request_file_sha256,
            expected_request_closed_record_sha256=(
                args.expected_request_closed_record_sha256
            ),
            coordinator_run_id=args.coordinator_run_id,
        )
        print(_required_sha256(result, "closed_record_sha256"))
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
    "PUBLICATION_LATENCY_COMPONENT_EVIDENCE_POLICY",
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
    "PUBLICATION_LATENCY_SOURCE_CLOSURE_MAX_BYTES",
    "PUBLICATION_LATENCY_SOURCE_CLOSURE_NODE_TYPE_ID",
    "PUBLICATION_LATENCY_SOURCE_CLOSURE_REQUEST_AUTHORIZATION_RECORD_TYPE",
    "PUBLICATION_LATENCY_SOURCE_CLOSURE_REQUEST_RECORD_TYPE",
    "PUBLICATION_LATENCY_SOURCE_CLOSURE_RESULT_RECORD_TYPE",
    "PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT",
    "PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SHA256",
    "PUBLICATION_LATENCY_SOURCE_CLOSURE_SCHEMA_VERSION",
    "PUBLICATION_LATENCY_SOURCE_CLOSURE_SPARK_VERSION",
    "PUBLICATION_LATENCY_SOURCE_CLOSURE_TIMEOUT_SECONDS",
    "PUBLICATION_LATENCY_SUBMISSION_RECORD_TYPE",
    "PUBLICATION_LATENCY_SUMMARY_RECORD_TYPE",
    "PublicationLatencyArtifactFile",
    "PublicationLatencyCollectionAuthorization",
    "PublicationLatencyFinalArtifactPins",
    "PublicationLatencySourceClosureAuthorization",
    "PublicationLatencySourceClosureCoordinatorConfig",
    "PublicationLatencySourceClosureRequestAuthorization",
    "PublicationLatencySourceClosureSubmissionAuthorization",
    "PublicationLatencyWaveAuthorization",
    "PublicationLatencyWaveSubmissionAuthorization",
    "aggregate_publication_latency_campaign",
    "build_databricks_publication_latency_run_submit_payload",
    "build_publication_latency_execution_plan",
    "build_publication_latency_source_closure_request",
    "collect_publication_latency_campaign",
    "collect_publication_latency_launch_wave",
    "collect_publication_latency_source_closure",
    "execute_publication_latency_job_record",
    "main",
    "publication_latency_reservation_attempt_id",
    "publication_latency_source_closure_control_roots",
    "publication_latency_submit_payloads",
    "publication_latency_vllm_config",
    "render_publication_latency_job_record",
    "render_publication_latency_source_closure_submit_payload",
    "require_publication_latency_source_closure_authorization",
    "resume_publication_latency_source_closure",
    "run_publication_latency_source_closure_coordinator",
    "require_publication_latency_collection_authorization",
    "seal_publication_latency_job_result",
    "submit_publication_latency_launch_wave",
    "submit_publication_latency_source_closure",
    "resume_publication_latency_launch_wave",
    "validate_publication_latency_collection_record",
    "validate_publication_latency_execution_plan_record",
    "validate_publication_latency_execution_sources",
    "validate_publication_latency_source_closure_request",
    "validate_publication_latency_source_closure_result",
    "validate_publication_latency_job_record",
    "validate_publication_latency_job_result_record",
    "validate_publication_latency_summary_record",
    "write_publication_latency_runner_script",
    "write_publication_latency_source_closure_runner_script",
]


if __name__ == "__main__":  # pragma: no cover - CLI boundary.
    raise SystemExit(main())
