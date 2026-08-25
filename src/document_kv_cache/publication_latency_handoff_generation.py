"""Generate and close the reusable Vanilla KV inputs for latency evidence.

The latency campaign consumes the same 128 identities at 8k, 16k, and 32k.
This module turns that closed input suite into 384 exactly-once Q8 pre-RoPE
generation items, balances them by exact cache-prefix token counts, and runs one
persistent generator per worker.  Generated files are never exposed in place:
all context bundles are closed, content-addressed, durably synced, and atomically
published before a serving binding can be resolved.
"""

from __future__ import annotations

import argparse
import gc
import hmac
import importlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol, cast


from document_kv_cache.benchmark_handoffs import (
    enrich_benchmark_records_with_handoffs,
    generate_benchmark_handoff_bundles,
)
from document_kv_cache.benchmarks import (
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    SUPPORTED_V1_DATASETS,
    benchmark_cache_prefix_segments,
)
from document_kv_cache.artifact_identity import TokenContract, token_ids_digest
from document_kv_cache.engine_protocol import (
    KVKeyPositionEncoding,
    KVLayout,
    KVPayloadAxisOrder,
    KVStorageLayout,
)
from document_kv_cache.main_latency_inputs import (
    MAIN_LATENCY_ADD_SPECIAL_TOKENS,
    MAIN_LATENCY_EXAMPLES_PER_DATASET,
    MAIN_LATENCY_PROVENANCE_FILENAME,
    MAIN_LATENCY_TOKENIZER_ID,
    MAIN_LATENCY_TOKENIZER_REVISION,
    MainLatencyTokenizer,
    PreparedMainLatencyInputs,
    load_main_latency_tokenizer,
    verify_main_latency_inputs,
)
from document_kv_cache.databricks_job import (
    DEFAULT_DATABRICKS_DATA_SECURITY_MODE,
    DEFAULT_DATABRICKS_SPARK_VERSION,
    DEFAULT_DATABRICKS_TASK_MAX_RETRIES,
    _validated_databricks_run_timeout_seconds,
    _validated_databricks_task_max_retries,
)
from document_kv_cache.databricks_resource_ledger import (
    MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS,
    MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    DatabricksClusterHourLedger,
    DatabricksClusterHourReservation,
    DatabricksBatchReservationAuthorization,
    DatabricksLedgerPrefix,
    DatabricksRunAttemptReservationRequest,
    canonical_databricks_submit_payload_snapshot,
    databricks_ledger_prefix,
    databricks_ledger_prefix_from_record,
    databricks_ledger_path_sha256,
    read_databricks_cluster_hour_ledger_json,
    replay_databricks_run_attempt_batch_authorization_json,
    record_databricks_run_submission_receipt_json,
    record_databricks_verified_run_terminal_actual_json,
    reserve_databricks_run_attempt_json,
    reserve_databricks_run_attempt_batch_authorized_json,
    require_databricks_batch_reservation_authorization,
    require_databricks_batch_terminal_closure,
    require_databricks_ledger_prefix,
    require_databricks_publication_batch_admission,
)
from document_kv_cache.databricks_runs import (
    DatabricksURLOpener,
    DatabricksWorkspaceConfig,
    bind_databricks_run_idempotency_token,
    databricks_run_status_record,
    get_databricks_run,
    reserve_and_submit_databricks_run,
    require_databricks_run_idempotency_token,
    resume_pre_reserved_databricks_run,
    submit_pre_reserved_databricks_run,
    summarize_databricks_run,
)
from document_kv_cache.model_profiles import (
    QWEN3_4B_INSTRUCT_HF_MODEL_ID,
    QWEN3_4B_ROPE_ROTARY_DIM,
    QWEN3_4B_ROPE_THETA,
    layout_for_model,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256,
    GPUQualificationArtifactPins,
    GPUQualificationSelection,
    canonical_gpu_qualification_json,
    validate_gpu_qualification_evidence_record,
)
from document_kv_cache.gpu_qualification_databricks import (
    GPUQualificationLaunchAuthorization,
    require_gpu_qualification_launch_authorization,
)
from document_kv_cache.models import CacheGenerationMethod
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_CONTEXT_TOKENS,
    PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_INPUT_TOKEN_SLOTS,
    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_RESERVED_GPU_HOURS,
    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS,
    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS,
    PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS,
    PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND,
    PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS,
    publication_campaign_full_launch_budget_projection,
)
from document_kv_cache.serving_env import (
    TRANSFORMERS_CONSTRAINT,
    VLLM_RUNTIME_LOCK_SHA256,
)
from document_kv_cache.publication_handoff_artifacts import (
    close_publication_latency_handoff_bundle,
    read_publication_latency_handoff_bundle,
    validate_publication_latency_handoff_bundle,
    write_publication_latency_handoff_bundle,
)
from document_kv_cache.storage import local_path
from document_kv_cache.transformers_generator import (
    CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV,
    CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV,
    CACHET_TRANSFORMERS_DEVICE_ENV,
    CACHET_TRANSFORMERS_DEVICE_MAP_ENV,
    CACHET_TRANSFORMERS_MODEL_ID_ENV,
    CACHET_TRANSFORMERS_MODEL_REVISION_ENV,
    CACHET_TRANSFORMERS_PRE_ROPE_ENV,
    CACHET_TRANSFORMERS_QUANTIZATION_ENV,
    CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_ID_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV,
    CACHET_TRANSFORMERS_TORCH_DTYPE_ENV,
    CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV,
    build_pre_rope_transformers_kv_chunk_generator,
)
from document_kv_cache.workflow import KVChunkGenerator


PUBLICATION_LATENCY_HANDOFF_PLAN_RECORD_TYPE = (
    "cachet.publication_latency_handoff_generation_plan.v1"
)
PUBLICATION_LATENCY_HANDOFF_PLAN_SCHEMA_VERSION = 1
PUBLICATION_LATENCY_HANDOFF_EXECUTION_RECORD_TYPE = (
    "cachet.publication_latency_handoff_generation_execution.v1"
)
PUBLICATION_LATENCY_HANDOFF_EXECUTION_SCHEMA_VERSION = 1
PUBLICATION_LATENCY_HANDOFF_EXECUTION_FILENAME = (
    "publication-latency-handoff-generation.execution.json"
)
PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED = (
    "distributed_16x_l40s_qualified"
)
PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_LOCAL_TEST = "injected_cpu_test_helper"
PUBLICATION_LATENCY_HANDOFF_DTYPE = "fp8_e5m2"
PUBLICATION_LATENCY_HANDOFF_ARM_ID = "vanilla"
PUBLICATION_LATENCY_HANDOFF_TASK_COUNT = (
    len(SUPPORTED_V1_DATASETS)
    * PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
    * len(PUBLICATION_CAMPAIGN_CONTEXT_TOKENS)
)
PUBLICATION_LATENCY_HANDOFF_WORKER_PAYLOAD_RECORD_TYPE = (
    "cachet.publication_latency_handoff_worker_payload.v1"
)
PUBLICATION_LATENCY_HANDOFF_WORKER_PAYLOAD_SCHEMA_VERSION = 1
PUBLICATION_LATENCY_HANDOFF_WORKER_RESULT_RECORD_TYPE = (
    "cachet.publication_latency_handoff_worker_result.v1"
)
PUBLICATION_LATENCY_HANDOFF_WORKER_RESULT_SCHEMA_VERSION = 1
PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_RECORD_TYPE = (
    "cachet.publication_latency_handoff_databricks_execution.v1"
)
PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_SCHEMA_VERSION = 1
PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY = "databricks-attestations"
PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET = "aws-g6e-l40s"
PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID = "g6e.4xlarge"
PUBLICATION_LATENCY_HANDOFF_GENERATOR_GPU_MODEL = "NVIDIA L40S"
PUBLICATION_LATENCY_HANDOFF_GENERATOR_QUANTIZATION = "bitsandbytes-4bit"
PUBLICATION_LATENCY_HANDOFF_GENERATOR_MODEL_DTYPE = "bfloat16"
PUBLICATION_LATENCY_HANDOFF_TRANSFORMERS_VERSION = TRANSFORMERS_CONSTRAINT.removeprefix(
    "transformers=="
)
if (
    not TRANSFORMERS_CONSTRAINT.startswith("transformers==")
    or not PUBLICATION_LATENCY_HANDOFF_TRANSFORMERS_VERSION
):
    raise RuntimeError("publication Transformers version constraint is invalid")
PUBLICATION_LATENCY_HANDOFF_RUNNER_FILENAME = (
    "publication_latency_handoff_generation_runner.py"
)

PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT = """from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import subprocess
import sys


def _cluster_path(uri: str) -> str:
    if uri.startswith("dbfs:/Volumes/"):
        return "/Volumes/" + uri.removeprefix("dbfs:/Volumes/")
    if uri.startswith("dbfs:/"):
        return "/dbfs/" + uri.removeprefix("dbfs:/").lstrip("/")
    return uri


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_path(uri: str, expected: str, label: str) -> str:
    path = _cluster_path(uri)
    if not hmac.compare_digest(_sha256(path), expected):
        raise ValueError(f"{label} SHA-256 does not match")
    return path


def _pip_subprocess_environment() -> dict[str, str]:
    env = dict(os.environ)
    for variable_name in tuple(env):
        if variable_name.upper().startswith("PIP_"):
            env.pop(variable_name)
    for variable_name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(variable_name, None)
    env.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def _bootstrap(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--package-wheel-uri", required=True)
    parser.add_argument("--package-wheel-sha256", required=True)
    parser.add_argument("--runtime-lock-uri", required=True)
    parser.add_argument("--runtime-lock-sha256", required=True)
    parser.add_argument("--patched-vllm-wheel-uri", required=True)
    parser.add_argument("--patched-vllm-wheel-sha256", required=True)
    parser.add_argument("--runtime-venv-dir", required=True)
    args, remaining = parser.parse_known_args(argv)
    if not hmac.compare_digest(_sha256(__file__), args.runner_sha256):
        raise ValueError("latency handoff runner SHA-256 does not match")
    package_wheel = _verified_path(
        args.package_wheel_uri, args.package_wheel_sha256, "Cachet wheel"
    )
    runtime_lock = _verified_path(
        args.runtime_lock_uri, args.runtime_lock_sha256, "runtime lock"
    )
    patched_wheel = _verified_path(
        args.patched_vllm_wheel_uri,
        args.patched_vllm_wheel_sha256,
        "patched vLLM wheel",
    )
    venv_dir = os.path.abspath(args.runtime_venv_dir)
    if not venv_dir.startswith("/local_disk0/"):
        raise ValueError("runtime venv must be rooted under /local_disk0")
    marker = os.environ.get("CACHET_LATENCY_HANDOFF_LOCKED_RUNTIME")
    venv_python = os.path.join(venv_dir, "bin", "python")
    if marker == args.runtime_lock_sha256:
        if os.path.realpath(sys.executable) != os.path.realpath(venv_python):
            raise RuntimeError("locked-runtime marker is set outside the bound venv")
        from document_kv_cache.publication_latency_handoff_generation import main

        raise SystemExit(main(remaining))
    if os.path.exists(venv_dir):
        raise FileExistsError("refusing to reuse an unverified handoff runtime")
    pip_environment = _pip_subprocess_environment()
    subprocess.check_call(
        [sys.executable, "-m", "venv", venv_dir],
        env=pip_environment,
    )
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
        [*pip, "install", "--no-deps", patched_wheel],
        env=pip_environment,
    )
    subprocess.check_call(
        [*pip, "install", "--no-deps", package_wheel],
        env=pip_environment,
    )
    subprocess.check_call([*pip, "check"], env=pip_environment)
    expected_spec = (
        "vllm @ file://" + os.path.abspath(patched_wheel)
        + "#sha256=" + args.patched_vllm_wheel_sha256
    )
    verifier = (
        "import json,sys; from document_kv_cache.serving_env import "
        "verify_installed_vllm_runtime_lock as verify; "
        "print(json.dumps(verify(sys.argv[1]), sort_keys=True))"
    )
    verified = subprocess.check_output(
        [venv_python, "-c", verifier, expected_spec],
        text=True,
        env=pip_environment,
    )
    if json.loads(verified).get("ok") is not True:
        raise RuntimeError("locked runtime verifier did not attest success")
    env = dict(pip_environment)
    env["CACHET_LATENCY_HANDOFF_LOCKED_RUNTIME"] = args.runtime_lock_sha256
    os.execve(
        venv_python,
        [
            venv_python,
            "-m",
            "document_kv_cache.publication_latency_handoff_generation",
            *remaining,
        ],
        env,
    )


if __name__ == "__main__":
    _bootstrap(sys.argv[1:])
"""
PUBLICATION_LATENCY_HANDOFF_RUNNER_SHA256 = sha256(
    PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT.encode("utf-8")
).hexdigest()

_PLAN_ORDER_DOMAIN = "cachet.publication.latency_handoff.lpt.v1"
_SHA256_LENGTH = 64
_SUBMISSION_AUTHORIZATION_ISSUER = object()
_SERVING_AUTHORIZATION_ISSUER = object()
_REMOTE_CLOSURE_LEDGER_ISSUER = object()
_POST_CLOSE_REPLAY_ISSUER = object()


class PublicationLatencyGeneratorFactory(Protocol):
    """Create one GPU-resident generator for a stable worker index."""

    def __call__(self, worker_index: int) -> KVChunkGenerator: ...


@dataclass(frozen=True, slots=True)
class PublicationLatencyHandoffExecutionConfig:
    """Pinned artifact identity shared by all persistent generation workers."""

    layout: KVLayout
    model_revision: str
    generator_version: str
    vllm_bitsandbytes_loader_source_sha256: str
    tokenizer_id: str = MAIN_LATENCY_TOKENIZER_ID
    tokenizer_revision: str = MAIN_LATENCY_TOKENIZER_REVISION
    generator_family: str = "transformers"
    generator_device_map: str = "auto"
    generator_quantization: str = PUBLICATION_LATENCY_HANDOFF_GENERATOR_QUANTIZATION
    generator_model_dtype: str = PUBLICATION_LATENCY_HANDOFF_GENERATOR_MODEL_DTYPE
    generator_cache_axis_order: str = "head_major"
    generator_trust_remote_code: bool = False
    generator_add_special_tokens: bool = False
    generator_quantization_config: Mapping[str, Any] = field(
        default_factory=lambda: {
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_storage": "uint8",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "load_in_4bit": True,
        }
    )
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    align_bytes: int = 4096

    def __post_init__(self) -> None:
        if not isinstance(self.layout, KVLayout):
            raise TypeError("layout must be a KVLayout")
        self.layout.validate()
        if self.layout.dtype != PUBLICATION_LATENCY_HANDOFF_DTYPE:
            raise ValueError("publication latency handoffs require fp8_e5m2 Q8 KV")
        if (
            not self.layout.pre_rope
            or self.layout.key_position_encoding != KVKeyPositionEncoding.PRE_ROPE
        ):
            raise ValueError("publication latency handoffs require pre-RoPE keys")
        if (
            self.layout.shares_kv_storage
            or self.layout.storage_layout != KVStorageLayout.SEPARATE_KEY_VALUE
        ):
            raise ValueError("pre-RoPE publication handoffs require separate K/V")
        for field_name in (
            "model_revision",
            "generator_version",
            "tokenizer_id",
            "tokenizer_revision",
            "generator_family",
            "generator_device_map",
            "generator_quantization",
            "generator_model_dtype",
            "generator_cache_axis_order",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value == "unresolved":
                raise ValueError(f"{field_name} must be pinned")
        _require_sha256(
            self.vllm_bitsandbytes_loader_source_sha256,
            field_name="vllm_bitsandbytes_loader_source_sha256",
        )
        if self.vllm_bitsandbytes_loader_source_sha256 != (
            GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
        ):
            raise ValueError("vLLM bitsandbytes loader source hash is not qualified")
        expected_quantization = {
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_storage": "uint8",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "load_in_4bit": True,
        }
        if self.generator_device_map != "auto":
            raise ValueError("latency generator device_map must equal auto")
        if self.generator_quantization != (
            PUBLICATION_LATENCY_HANDOFF_GENERATOR_QUANTIZATION
        ):
            raise ValueError("latency generator must use bitsandbytes-4bit")
        if self.generator_model_dtype != (
            PUBLICATION_LATENCY_HANDOFF_GENERATOR_MODEL_DTYPE
        ):
            raise ValueError("latency generator compute dtype must be bfloat16")
        if self.generator_cache_axis_order != "head_major":
            raise ValueError("latency generator cache axis order must be head_major")
        if self.generator_trust_remote_code is not False:
            raise ValueError("latency generator trust_remote_code must be false")
        if self.generator_add_special_tokens is not False:
            raise ValueError("latency generator add_special_tokens must be false")
        if dict(self.generator_quantization_config) != expected_quantization:
            raise ValueError("latency generator BnB NF4 configuration drift")
        object.__setattr__(
            self,
            "generator_quantization_config",
            MappingProxyType(dict(self.generator_quantization_config)),
        )
        if self.tokenizer_id != MAIN_LATENCY_TOKENIZER_ID:
            raise ValueError("tokenizer_id must match the main-latency tokenizer")
        if self.tokenizer_revision != MAIN_LATENCY_TOKENIZER_REVISION:
            raise ValueError("tokenizer_revision must match the main-latency tokenizer")
        for field_name in ("tensor_parallel_size", "pipeline_parallel_size"):
            value = getattr(self, field_name)
            if type(value) is not int or value != 1:
                raise ValueError(f"{field_name} must equal 1")
        if type(self.align_bytes) is not int or self.align_bytes <= 0:
            raise ValueError("align_bytes must be a positive integer")


@dataclass(frozen=True, slots=True)
class PublicationLatencyGeneratorHardwareQualification:
    """Canonical sealed GPU-qualifier result plus immutable file bindings."""

    evidence_record: Mapping[str, Any]
    plan_record: Mapping[str, Any]
    expected_campaign_id: str
    expected_artifact_pins: GPUQualificationArtifactPins
    evidence_uri: str
    evidence_file_sha256: str
    plan_uri: str
    plan_file_sha256: str
    selection: GPUQualificationSelection = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.expected_campaign_id, str) or not (
            self.expected_campaign_id
        ):
            raise ValueError("expected_campaign_id must be non-empty")
        if not isinstance(self.expected_artifact_pins, GPUQualificationArtifactPins):
            raise TypeError("expected_artifact_pins has the wrong type")
        for field_name in ("evidence_uri", "plan_uri"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in ("evidence_file_sha256", "plan_file_sha256"):
            _require_sha256(getattr(self, field_name), field_name=field_name)
        selection = validate_gpu_qualification_evidence_record(
            self.evidence_record,
            plan_record=self.plan_record,
            expected_campaign_id=self.expected_campaign_id,
            expected_artifact_pins=self.expected_artifact_pins,
        )
        if selection.generation_hardware_id != (
            PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET
        ):
            raise ValueError("canonical GPU qualifier did not select L40S")
        if selection.generation_databricks_node_type_id != (
            PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
        ):
            raise ValueError("canonical GPU qualifier did not select g6e.4xlarge")
        if selection.generation_prefix_tokens_per_second < (
            PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
        ):
            raise ValueError("canonical L40S qualification is below 35 tokens/s")
        object.__setattr__(
            self, "evidence_record", MappingProxyType(dict(self.evidence_record))
        )
        object.__setattr__(
            self, "plan_record", MappingProxyType(dict(self.plan_record))
        )
        object.__setattr__(self, "selection", selection)


@dataclass(frozen=True, slots=True)
class DatabricksPublicationLatencyHandoffJobConfig:
    """Bootstrap and one-task job settings for sixteen independent producers."""

    runner_python_file: str
    worker_payload_uri_template: str
    package_wheel_uri: str
    package_wheel_sha256: str
    runtime_lock_uri: str
    runtime_lock_sha256: str
    patched_vllm_wheel_uri: str
    patched_vllm_wheel_sha256: str
    runner_sha256: str = PUBLICATION_LATENCY_HANDOFF_RUNNER_SHA256
    runtime_venv_dir_template: str = (
        "/local_disk0/cachet-latency-handoff-runtime-{worker_index}"
    )
    run_name: str = "cachet-vllm-0271-latency-handoff-generation"
    task_key_prefix: str = "latency_handoff_worker"
    run_timeout_seconds: int = PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS
    task_max_retries: int = DEFAULT_DATABRICKS_TASK_MAX_RETRIES
    node_type_id: str = PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
    spark_version: str = DEFAULT_DATABRICKS_SPARK_VERSION
    data_security_mode: str = DEFAULT_DATABRICKS_DATA_SECURITY_MODE
    single_user_name: str | None = None
    availability: str = "ON_DEMAND"
    zone_id: str = "auto"
    custom_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "runner_python_file",
            "worker_payload_uri_template",
            "package_wheel_uri",
            "runtime_lock_uri",
            "patched_vllm_wheel_uri",
            "runtime_venv_dir_template",
            "run_name",
            "spark_version",
            "data_security_mode",
            "availability",
            "zone_id",
            "task_key_prefix",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be non-empty")
        if "{worker_index}" not in self.worker_payload_uri_template:
            raise ValueError("worker_payload_uri_template requires {worker_index}")
        if "{worker_index}" not in self.runtime_venv_dir_template:
            raise ValueError("runtime_venv_dir_template requires {worker_index}")
        for field_name in (
            "runner_sha256",
            "package_wheel_sha256",
            "runtime_lock_sha256",
            "patched_vllm_wheel_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name=field_name)
        if self.runner_sha256 != PUBLICATION_LATENCY_HANDOFF_RUNNER_SHA256:
            raise ValueError("latency handoff runner hash drift")
        if self.runtime_lock_sha256 != VLLM_RUNTIME_LOCK_SHA256:
            raise ValueError("latency handoff runtime lock hash drift")
        if self.node_type_id != PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID:
            raise ValueError("publication handoff producers require g6e.4xlarge")
        if self.task_key_prefix != "latency_handoff_worker":
            raise ValueError("latency handoff task_key_prefix is frozen")
        if self.run_timeout_seconds != (
            PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS
        ):
            raise ValueError("latency handoff producer timeout must equal five hours")
        _validated_databricks_run_timeout_seconds(self.run_timeout_seconds)
        _validated_databricks_task_max_retries(self.task_max_retries)
        if self.data_security_mode == "SINGLE_USER" and not self.single_user_name:
            raise ValueError("single_user_name is required for SINGLE_USER")
        object.__setattr__(self, "custom_tags", dict(self.custom_tags))


@dataclass(frozen=True, slots=True)
class PublicationLatencyServingHandoffBundle:
    """One validated durable bundle that timed serving must reuse."""

    context_tokens: int
    manifest_path: Path
    source_root: Path
    portable_bundle_sha256: str
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.context_tokens not in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
            raise ValueError("context_tokens is outside the publication campaign")
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "source_root", Path(self.source_root))
        _require_sha256(
            self.portable_bundle_sha256,
            field_name="portable_bundle_sha256",
        )
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


@dataclass(frozen=True, slots=True)
class PublicationLatencyHandoffGenerationResult:
    """Atomically published execution evidence and serving bindings."""

    root: Path
    execution_record_path: Path
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(
            self,
            "execution_record_path",
            Path(self.execution_record_path),
        )
        object.__setattr__(self, "record", MappingProxyType(dict(self.record)))

    def serving_bundle(
        self,
        context_tokens: int,
        *,
        authorization: PublicationLatencyHandoffServingAuthorization,
    ) -> PublicationLatencyServingHandoffBundle:
        """Resolve and re-authenticate the exact bundle for one serving context."""

        return resolve_publication_latency_serving_handoff_bundle(
            authorization,
            context_tokens=context_tokens,
        )


@dataclass(frozen=True, slots=True, init=False)
class PublicationLatencyHandoffSubmissionAuthorization:
    """Ephemeral authority proving the durable exact-16 Q8 phase admission."""

    batch_authorization: DatabricksBatchReservationAuthorization
    phase_lease_root: Path
    phase_lease_root_sha256: str
    phase_lease_file_sha256: str
    phase_lease_closed_record_sha256: str
    batch_marker_file_sha256: str
    batch_marker_closed_record_sha256: str
    durable_output_root: str

    def __init__(
        self,
        *,
        batch_authorization: DatabricksBatchReservationAuthorization,
        phase_lease_root: str | Path,
        durable_output_root: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _SUBMISSION_AUTHORIZATION_ISSUER:
            raise TypeError(
                "Q8 submission authority requires the durable phase issuer"
            )
        if not isinstance(
            batch_authorization, DatabricksBatchReservationAuthorization
        ):
            raise TypeError("batch_authorization has the wrong type")
        root = Path(phase_lease_root).expanduser().absolute()
        output_root = _normalized_q8_durable_output_root(durable_output_root)
        lease, marker = _validate_q8_submission_phase_files(
            root,
            batch_authorization,
            durable_output_root=output_root,
        )
        root = root.resolve(strict=True)
        object.__setattr__(self, "batch_authorization", batch_authorization)
        object.__setattr__(self, "phase_lease_root", root)
        object.__setattr__(
            self,
            "phase_lease_root_sha256",
            _q8_phase_lease_root_sha256(root),
        )
        object.__setattr__(
            self,
            "phase_lease_file_sha256",
            _file_sha256(root / "phase-lease.json"),
        )
        object.__setattr__(
            self,
            "phase_lease_closed_record_sha256",
            _required_string(lease, "closed_record_sha256"),
        )
        object.__setattr__(
            self,
            "batch_marker_file_sha256",
            _file_sha256(root / "batch-reserved.json"),
        )
        object.__setattr__(
            self,
            "batch_marker_closed_record_sha256",
            _required_string(marker, "closed_record_sha256"),
        )
        object.__setattr__(self, "durable_output_root", output_root)


@dataclass(frozen=True, slots=True, init=False)
class PublicationLatencyHandoffServingAuthorization:
    """Ephemeral authority issued only after live replay of all 16 cloud runs."""

    result_root: Path
    execution_file_sha256: str
    execution_closed_record_sha256: str
    ledger_id: str
    ledger_path_sha256: str
    predecessor_prefix: DatabricksLedgerPrefix
    producer_batch_prefix: DatabricksLedgerPrefix
    ledger_prefix: DatabricksLedgerPrefix
    causal_closure_sha256: str

    def __init__(
        self,
        *,
        result: PublicationLatencyHandoffGenerationResult,
        ledger_id: str,
        ledger_path_sha256: str,
        predecessor_prefix: DatabricksLedgerPrefix,
        producer_batch_prefix: DatabricksLedgerPrefix,
        ledger_prefix: DatabricksLedgerPrefix,
        causal_closure_sha256: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _SERVING_AUTHORIZATION_ISSUER:
            raise TypeError("Q8 serving authority requires live runs/get causal replay")
        if not isinstance(result, PublicationLatencyHandoffGenerationResult):
            raise TypeError("result has the wrong type")
        object.__setattr__(self, "result_root", result.root.resolve())
        object.__setattr__(
            self,
            "execution_file_sha256",
            _file_sha256(result.execution_record_path),
        )
        object.__setattr__(
            self,
            "execution_closed_record_sha256",
            _required_string(result.record, "closed_record_sha256"),
        )
        if not isinstance(ledger_id, str) or not ledger_id:
            raise ValueError("ledger_id must be non-empty")
        object.__setattr__(self, "ledger_id", ledger_id)
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _require_sha256(
                ledger_path_sha256,
                field_name="ledger_path_sha256",
            ),
        )
        if not isinstance(ledger_prefix, DatabricksLedgerPrefix):
            raise TypeError("ledger_prefix must be a DatabricksLedgerPrefix")
        if not isinstance(predecessor_prefix, DatabricksLedgerPrefix):
            raise TypeError("predecessor_prefix must be a DatabricksLedgerPrefix")
        if not isinstance(producer_batch_prefix, DatabricksLedgerPrefix):
            raise TypeError("producer_batch_prefix must be a DatabricksLedgerPrefix")
        if ledger_prefix.ledger_id != ledger_id:
            raise ValueError("Q8 ledger prefix identity drift")
        if (
            predecessor_prefix.ledger_id != ledger_id
            or producer_batch_prefix.ledger_id != ledger_id
        ):
            raise ValueError("Q8 producer batch prefix identity drift")
        object.__setattr__(self, "predecessor_prefix", predecessor_prefix)
        object.__setattr__(self, "producer_batch_prefix", producer_batch_prefix)
        object.__setattr__(self, "ledger_prefix", ledger_prefix)
        object.__setattr__(
            self,
            "causal_closure_sha256",
            _require_sha256(
                causal_closure_sha256,
                field_name="causal_closure_sha256",
            ),
        )


@dataclass(frozen=True, slots=True)
class PublicationLatencyHandoffDatabricksAttestationBinding:
    """Immutable file and record binding for one sanitized cloud execution."""

    worker_index: int
    path: Path
    file_sha256: str
    closed_record_sha256: str

    def __post_init__(self) -> None:
        _validate_producer_worker_index(self.worker_index)
        object.__setattr__(self, "path", Path(self.path))
        _require_sha256(self.file_sha256, field_name="attestation file_sha256")
        _require_sha256(
            self.closed_record_sha256,
            field_name="attestation closed_record_sha256",
        )


@dataclass(frozen=True, slots=True)
class _WorkerBatchResult:
    context_tokens: int
    worker_index: int
    records: tuple[dict[str, Any], ...]
    task_ids: tuple[str, ...]
    cache_prefix_tokens: int
    input_token_slots: int
    generation_seconds: float
    durable_sync_seconds: float
    durable_byte_count: int


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    worker_index: int
    initialization_seconds: float
    lifecycle_seconds: float
    batches: tuple[_WorkerBatchResult, ...]


def build_publication_latency_handoff_execution_config(
    *,
    vllm_bitsandbytes_loader_source_sha256: str,
    generator_version: str = PUBLICATION_LATENCY_HANDOFF_TRANSFORMERS_VERSION,
) -> PublicationLatencyHandoffExecutionConfig:
    """Build the frozen Qwen3 Q8 pre-RoPE/NF4 generation contract."""

    layout = layout_for_model(
        QWEN3_4B_INSTRUCT_HF_MODEL_ID,
        dtype=PUBLICATION_LATENCY_HANDOFF_DTYPE,
        pre_rope=True,
        rope_theta=QWEN3_4B_ROPE_THETA,
        rope_rotary_dim=QWEN3_4B_ROPE_ROTARY_DIM,
        shares_kv_storage=False,
        storage_layout=KVStorageLayout.SEPARATE_KEY_VALUE,
        payload_axis_order=KVPayloadAxisOrder.TOKEN_MAJOR,
    )
    return PublicationLatencyHandoffExecutionConfig(
        layout=layout,
        model_revision=MAIN_LATENCY_TOKENIZER_REVISION,
        generator_version=generator_version,
        vllm_bitsandbytes_loader_source_sha256=(vllm_bitsandbytes_loader_source_sha256),
    )


def build_publication_latency_handoff_generation_plan(
    prepared_input_dir: str | Path,
    *,
    plan_id: str,
    tokenizer: MainLatencyTokenizer,
    worker_count: int = PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS,
    source_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build the exact-token, deterministic worker plan for all 384 artifacts."""

    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError("plan_id must be a non-empty string")
    workers = _validated_worker_count(worker_count)
    prepared = verify_main_latency_inputs(
        prepared_input_dir,
        tokenizer=tokenizer,
        source_paths=source_paths,
        examples_per_dataset=PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    )
    items, identities_sha256 = _generation_items(prepared, tokenizer=tokenizer)
    assignments = _lpt_assignments(items, worker_count=workers)
    cache_prefix_tokens = sum(cast(int, item["cache_prefix_tokens"]) for item in items)
    input_token_slots = sum(cast(int, item["input_token_slots"]) for item in items)
    if input_token_slots != PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_INPUT_TOKEN_SLOTS:
        raise ValueError(
            "latency handoff input-token accounting diverged from campaign"
        )

    worker_records: list[dict[str, Any]] = []
    for worker_index, worker_items in enumerate(assignments):
        ordered = sorted(
            worker_items,
            key=lambda item: (
                cast(int, item["context_tokens"]),
                cast(str, item["dataset"]),
                cast(int, item["row_index"]),
            ),
        )
        worker_records.append(
            {
                "cache_prefix_tokens": sum(
                    cast(int, item["cache_prefix_tokens"]) for item in ordered
                ),
                "input_token_slots": sum(
                    cast(int, item["input_token_slots"]) for item in ordered
                ),
                "item_count": len(ordered),
                "items": ordered,
                "items_sha256": _canonical_sha256(ordered),
                "persistent_generator_instances": 1,
                "worker_id": f"latency-handoff-worker-{worker_index:02d}",
                "worker_index": worker_index,
            }
        )

    worker_loads = [
        cast(int, worker["cache_prefix_tokens"]) for worker in worker_records
    ]
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "coverage": {
            "cache_prefix_generation_tokens": cache_prefix_tokens,
            "context_tokens": list(PUBLICATION_CAMPAIGN_CONTEXT_TOKENS),
            "dataset_count": len(SUPPORTED_V1_DATASETS),
            "datasets": list(SUPPORTED_V1_DATASETS),
            "identities_per_context": (
                len(SUPPORTED_V1_DATASETS) * PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
            ),
            "identities_sha256": identities_sha256,
            "input_token_slots": input_token_slots,
            "no_duplicate_context_identities": True,
            "task_count": len(items),
        },
        "generation_contract": {
            "add_special_tokens": MAIN_LATENCY_ADD_SPECIAL_TOKENS,
            "cache_method": CacheGenerationMethod.VANILLA_PREFILL.value,
            "durability_boundary": (
                "payload_handoff_dataset_manifest_fsync_then_atomic_publish"
            ),
            "kv_dtype": PUBLICATION_LATENCY_HANDOFF_DTYPE,
            "position_encoding": KVKeyPositionEncoding.PRE_ROPE.value,
            "regenerate_inside_timed_serving_jobs": False,
            "segment_per_document": True,
            "storage_layout": KVStorageLayout.SEPARATE_KEY_VALUE.value,
        },
        "input_bundle_sha256": prepared.bundle_sha256,
        "plan_id": plan_id,
        "record_type": PUBLICATION_LATENCY_HANDOFF_PLAN_RECORD_TYPE,
        "schema_version": PUBLICATION_LATENCY_HANDOFF_PLAN_SCHEMA_VERSION,
        "sharding": {
            "algorithm": "deterministic_lpt_exact_cache_prefix_tokens_v1",
            "exact_token_counts": True,
            "max_parallel_workers": PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS,
            "max_worker_cache_prefix_tokens": max(worker_loads),
            "min_worker_cache_prefix_tokens": min(worker_loads),
            "persistent_worker": True,
            "worker_count": workers,
            "worker_imbalance_tokens": max(worker_loads) - min(worker_loads),
        },
        "workers": worker_records,
        "workers_sha256": _canonical_sha256(worker_records),
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def validate_publication_latency_handoff_generation_plan(
    record: Mapping[str, Any],
    *,
    prepared_input_dir: str | Path,
    tokenizer: MainLatencyTokenizer,
    source_paths: Mapping[str, str | Path] | None = None,
) -> None:
    """Rebuild a plan from verified inputs and reject every changed assignment."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("latency handoff plan closed_record_sha256 is invalid")
    plan_id = record.get("plan_id")
    sharding = record.get("sharding")
    if not isinstance(plan_id, str) or not plan_id or not isinstance(sharding, Mapping):
        raise ValueError("latency handoff plan identity is invalid")
    worker_count = sharding.get("worker_count")
    if type(worker_count) is not int:
        raise ValueError("latency handoff plan worker_count is invalid")
    expected = build_publication_latency_handoff_generation_plan(
        prepared_input_dir,
        plan_id=plan_id,
        tokenizer=tokenizer,
        worker_count=worker_count,
        source_paths=source_paths,
    )
    if dict(record) != expected:
        raise ValueError("latency handoff plan does not match verified inputs")


def write_publication_latency_handoff_generation_plan(
    record: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Write one canonical closed plan without permitting replacement."""

    _validate_closed_plan_envelope(record)
    destination = Path(path).expanduser().absolute()
    _write_canonical_json_exclusive(record, destination)
    return destination


def read_publication_latency_handoff_generation_plan(
    path: str | Path,
) -> dict[str, Any]:
    """Read a canonical plan; input-bound validation remains a separate gate."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError("latency handoff plan must be a real file")
    content = source.read_bytes()
    record = _json_object(content, field_name="latency handoff generation plan")
    if content != _canonical_json_bytes(record, pretty=True):
        raise ValueError("latency handoff generation plan is not canonical JSON")
    _validate_closed_plan_envelope(record)
    return record


def build_publication_latency_handoff_worker_payloads(
    record: Mapping[str, Any],
    *,
    plan_uri: str,
    plan_file_sha256: str,
    prepared_input_uri: str,
    prepared_provenance_file_sha256: str,
    prepared_provenance_closed_record_sha256: str,
    durable_output_root: str,
    local_work_root_template: str,
    config: PublicationLatencyHandoffExecutionConfig,
    hardware_qualification: PublicationLatencyGeneratorHardwareQualification,
) -> tuple[dict[str, Any], ...]:
    """Bind every exact-token shard to one independent one-GPU producer."""

    _validate_closed_plan_envelope(record)
    workers = _mapping_sequence(record.get("workers"), field_name="workers")
    if len(workers) != PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS:
        raise ValueError("production latency generation requires exactly 16 workers")
    for field_name, value in (
        ("plan_uri", plan_uri),
        ("prepared_input_uri", prepared_input_uri),
        ("durable_output_root", durable_output_root),
        ("local_work_root_template", local_work_root_template),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} must be non-empty")
    if "{worker_index}" not in local_work_root_template:
        raise ValueError("local_work_root_template requires {worker_index}")
    for field_name, value in (
        ("plan_file_sha256", plan_file_sha256),
        ("prepared_provenance_file_sha256", prepared_provenance_file_sha256),
        (
            "prepared_provenance_closed_record_sha256",
            prepared_provenance_closed_record_sha256,
        ),
    ):
        _require_sha256(value, field_name=field_name)
    if not isinstance(config, PublicationLatencyHandoffExecutionConfig):
        raise TypeError("config must be a PublicationLatencyHandoffExecutionConfig")
    if not isinstance(
        hardware_qualification,
        PublicationLatencyGeneratorHardwareQualification,
    ):
        raise TypeError("hardware_qualification has the wrong type")
    plan_closed = _required_string(record, "closed_record_sha256")
    input_bundle = _required_string(record, "input_bundle_sha256")
    qualification = _hardware_qualification_record(hardware_qualification)
    payloads: list[dict[str, Any]] = []
    for worker in workers:
        worker_index = _required_int(worker, "worker_index")
        worker_id = _required_string(worker, "worker_id")
        if worker_index != len(payloads):
            raise ValueError("plan worker indices must be contiguous")
        contexts = sorted(
            {
                _required_int(item, "context_tokens")
                for item in _mapping_sequence(
                    worker.get("items"),
                    field_name="worker.items",
                )
            }
        )
        output_binding = {
            "context_worker_relative_roots": {
                str(context): f"pending/{context}/worker-{worker_index:02d}"
                for context in contexts
            },
            "partial_records_relative_root": (
                f"worker-records/worker-{worker_index:02d}"
            ),
            "result_relative_path": (f"worker-results/worker-{worker_index:02d}.json"),
        }
        payload: dict[str, Any] = {
            "assignment": {
                "cache_prefix_tokens": _required_int(
                    worker,
                    "cache_prefix_tokens",
                ),
                "input_token_slots": _required_int(worker, "input_token_slots"),
                "item_count": _required_int(worker, "item_count"),
                "items_sha256": _required_string(worker, "items_sha256"),
                "task_ids_sha256": _canonical_sha256(
                    sorted(
                        _required_string(item, "task_id")
                        for item in _mapping_sequence(
                            worker.get("items"),
                            field_name="worker.items",
                        )
                    )
                ),
            },
            "closed_record_sha256": "",
            "durable_output_root": durable_output_root.rstrip("/"),
            "execution_contract": _execution_config_record(config),
            "generator_hardware_qualification": qualification,
            "input_bundle_sha256": input_bundle,
            "local_work_root": local_work_root_template.format(
                worker_index=f"{worker_index:02d}"
            ),
            "output_binding": output_binding,
            "plan": {
                "closed_record_sha256": plan_closed,
                "file_sha256": plan_file_sha256,
                "uri": plan_uri,
            },
            "prepared_inputs": {
                "bundle_sha256": input_bundle,
                "provenance_closed_record_sha256": (
                    prepared_provenance_closed_record_sha256
                ),
                "provenance_file_sha256": prepared_provenance_file_sha256,
                "uri": prepared_input_uri,
            },
            "record_type": PUBLICATION_LATENCY_HANDOFF_WORKER_PAYLOAD_RECORD_TYPE,
            "schema_version": PUBLICATION_LATENCY_HANDOFF_WORKER_PAYLOAD_SCHEMA_VERSION,
            "worker_id": worker_id,
            "worker_index": worker_index,
        }
        payload["closed_record_sha256"] = _closed_record_sha256(payload)
        payloads.append(payload)
    if len({item["closed_record_sha256"] for item in payloads}) != len(payloads):
        raise ValueError("worker payload closures collide")
    return tuple(payloads)


def validate_publication_latency_handoff_worker_payload(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> None:
    """Reject a payload unless it is the exact binding for its plan worker."""

    if payload.get("record_type") != (
        PUBLICATION_LATENCY_HANDOFF_WORKER_PAYLOAD_RECORD_TYPE
    ):
        raise ValueError("latency handoff worker payload record_type is invalid")
    if payload.get("schema_version") != (
        PUBLICATION_LATENCY_HANDOFF_WORKER_PAYLOAD_SCHEMA_VERSION
    ):
        raise ValueError("latency handoff worker payload schema_version is invalid")
    if payload.get("closed_record_sha256") != _closed_record_sha256(payload):
        raise ValueError("latency handoff worker payload closure is invalid")
    _validate_closed_plan_envelope(plan)
    plan_binding = _required_mapping(payload, "plan")
    if plan_binding.get("closed_record_sha256") != plan.get("closed_record_sha256"):
        raise ValueError("worker payload plan closure drift")
    if payload.get("input_bundle_sha256") != plan.get("input_bundle_sha256"):
        raise ValueError("worker payload input bundle drift")
    worker_index = _required_int(payload, "worker_index")
    workers = _mapping_sequence(plan.get("workers"), field_name="workers")
    if not 0 <= worker_index < len(workers):
        raise ValueError("worker payload index is outside the plan")
    worker = workers[worker_index]
    if worker.get("worker_index") != worker_index or payload.get(
        "worker_id"
    ) != worker.get("worker_id"):
        raise ValueError("worker payload identity drift")
    assignment = _required_mapping(payload, "assignment")
    expected_assignment = {
        "cache_prefix_tokens": worker.get("cache_prefix_tokens"),
        "input_token_slots": worker.get("input_token_slots"),
        "item_count": worker.get("item_count"),
        "items_sha256": worker.get("items_sha256"),
        "task_ids_sha256": _canonical_sha256(
            sorted(
                _required_string(item, "task_id")
                for item in _mapping_sequence(
                    worker.get("items"),
                    field_name="worker.items",
                )
            )
        ),
    }
    if dict(assignment) != expected_assignment:
        raise ValueError("worker payload assignment drift")
    _execution_config_from_record(_required_mapping(payload, "execution_contract"))
    _validate_hardware_qualification_record(
        _required_mapping(payload, "generator_hardware_qualification")
    )
    _validate_worker_output_binding(payload, worker=worker)


def write_publication_latency_handoff_worker_payloads(
    payloads: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Write the sixteen closed worker payloads without replacement."""

    values = tuple(payloads)
    if len(values) != PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS:
        raise ValueError("exactly sixteen production worker payloads are required")
    destination = Path(output_dir).expanduser().absolute()
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for expected_index, payload in enumerate(values):
        if _required_int(payload, "worker_index") != expected_index:
            raise ValueError("worker payload order must be contiguous")
        if payload.get("closed_record_sha256") != _closed_record_sha256(payload):
            raise ValueError("worker payload closure is invalid")
        path = destination / f"latency-handoff-worker-{expected_index:02d}.json"
        _write_canonical_json_exclusive(payload, path)
        paths.append(path)
    return tuple(paths)


def build_databricks_publication_latency_handoff_worker_submit_payloads(
    config: DatabricksPublicationLatencyHandoffJobConfig,
    worker_payloads: Sequence[Mapping[str, Any]],
    *,
    ledger_path: str | Path,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
) -> tuple[dict[str, Any], ...]:
    """Render sixteen authorized, separate one-task runs/submit payloads.

    A task owns one GPU and loads one generator exactly once.  Combining these
    tasks into a single one-GPU run is intentionally unsupported.
    """

    if not isinstance(config, DatabricksPublicationLatencyHandoffJobConfig):
        raise TypeError("config has the wrong type")
    _require_matching_qualification_ledger(
        ledger_path,
        qualification_launch_authorization,
    )
    payloads = tuple(worker_payloads)
    if len(payloads) != PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS:
        raise ValueError("production generation requires exactly sixteen payloads")
    cluster = _build_l40s_single_node_cluster(config)
    submit_payloads: list[dict[str, Any]] = []
    durable_roots: set[str] = set()
    qualification_closures: set[str] = set()
    for expected_index, payload in enumerate(payloads):
        worker_index = _required_int(payload, "worker_index")
        if worker_index != expected_index:
            raise ValueError("worker payloads must be ordered 0..15")
        if payload.get("closed_record_sha256") != _closed_record_sha256(payload):
            raise ValueError("worker payload closure is invalid")
        durable_root = _required_string(payload, "durable_output_root")
        plan_binding = _required_mapping(payload, "plan")
        if not _is_databricks_durable_uri(
            durable_root,
            plan_closed_record_sha256=_required_string(
                plan_binding,
                "closed_record_sha256",
            ),
        ):
            raise ValueError(
                "production durable root must be a campaign/plan-bound UC/DBFS child"
            )
        durable_roots.add(durable_root)
        qualification = _required_mapping(
            payload,
            "generator_hardware_qualification",
        )
        _require_authorized_worker_payload(
            payload,
            qualification_launch_authorization,
            worker_index=worker_index,
        )
        qualification_pins = _required_mapping(
            qualification,
            "expected_artifact_pins",
        )
        expected_qualified_artifacts = {
            "package_wheel_sha256": config.package_wheel_sha256,
            "patched_vllm_wheel_sha256": config.patched_vllm_wheel_sha256,
            "runtime_lock_sha256": config.runtime_lock_sha256,
            "input_bundle_sha256": payload.get("input_bundle_sha256"),
        }
        if any(
            qualification_pins.get(key) != value
            for key, value in expected_qualified_artifacts.items()
        ):
            raise ValueError(
                "producer artifacts differ from canonical GPU qualification"
            )
        qualification_closures.add(
            _required_string(qualification, "evidence_closed_record_sha256")
        )
        worker_label = f"{worker_index:02d}"
        worker_uri = config.worker_payload_uri_template.format(
            worker_index=worker_label
        )
        runtime_venv = config.runtime_venv_dir_template.format(
            worker_index=worker_label
        )
        if not runtime_venv.startswith("/local_disk0/"):
            raise ValueError("worker runtime venv must be on local NVMe")
        expected_payload_file_sha256 = sha256(
            _canonical_json_bytes(payload, pretty=True)
        ).hexdigest()
        task_key = f"{config.task_key_prefix}_{worker_index:02d}"
        task = {
            "max_retries": config.task_max_retries,
            "new_cluster": json.loads(json.dumps(cluster)),
            "spark_python_task": {
                "parameters": [
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
                    "--runtime-venv-dir",
                    runtime_venv,
                    "run-worker",
                    "--worker-payload-json",
                    worker_uri,
                    "--expected-worker-payload-sha256",
                    expected_payload_file_sha256,
                ],
                "python_file": config.runner_python_file,
            },
            "task_key": task_key,
            "timeout_seconds": config.run_timeout_seconds,
        }
        attempt_id = publication_latency_handoff_worker_attempt_id(
            payload,
            worker_index=worker_index,
        )
        submit_payloads.append(
            bind_databricks_run_idempotency_token(
                {
                    "run_name": f"{config.run_name}-worker-{worker_index:02d}",
                    "timeout_seconds": config.run_timeout_seconds,
                    "tasks": [task],
                },
                attempt_id=attempt_id,
            )
        )
    if len(durable_roots) != 1:
        raise ValueError("all producers must share exactly one durable root")
    if len(qualification_closures) != 1:
        raise ValueError("all producers must bind one hardware qualification")
    worst_case_hours = sum(
        cast(int, item["timeout_seconds"]) / 3600.0 for item in submit_payloads
    )
    if worst_case_hours != PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_RESERVED_GPU_HOURS:
        raise ValueError("producer reservation does not equal the frozen 80 GPU-hours")
    return tuple(submit_payloads)


def publication_latency_handoff_worker_attempt_id(
    worker_payload: Mapping[str, Any],
    *,
    worker_index: int,
) -> str:
    """Return the sole publication attempt identity for one Q8 worker."""

    _validate_producer_worker_index(worker_index)
    if _required_int(worker_payload, "worker_index") != worker_index:
        raise ValueError("Q8 attempt worker index differs from its worker payload")
    plan_sha256 = _require_sha256(
        _required_mapping(worker_payload, "plan").get("closed_record_sha256"),
        field_name="worker plan closed_record_sha256",
    )
    return f"publication-q8/{plan_sha256[:20]}/worker-{worker_index:02d}"


def write_publication_latency_handoff_runner_script(path: str | Path) -> Path:
    """Write the hash-pinned standalone Databricks bootstrap runner once."""

    destination = Path(path).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"latency handoff runner already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT.encode("utf-8"))
    if _file_sha256(destination) != PUBLICATION_LATENCY_HANDOFF_RUNNER_SHA256:
        raise RuntimeError("written latency handoff runner hash drift")
    return destination


def _require_matching_qualification_ledger(
    ledger_path: str | Path,
    authorization: GPUQualificationLaunchAuthorization,
) -> DatabricksClusterHourLedger:
    if not isinstance(authorization, GPUQualificationLaunchAuthorization):
        raise TypeError(
            "qualification_launch_authorization must be a "
            "GPUQualificationLaunchAuthorization"
        )
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if databricks_ledger_path_sha256(ledger_path) != authorization.ledger_path_sha256:
        raise ValueError("Q8 ledger path differs from GPU qualification authority")
    if ledger.ledger_id != authorization.ledger_id:
        raise ValueError("publication ledger differs from GPU qualification authority")
    require_databricks_ledger_prefix(ledger, authorization.ledger_prefix)
    return ledger


def reserve_publication_latency_handoff_worker_attempt_json(
    ledger_path: str | Path,
    submit_payload: Mapping[str, Any],
    *,
    worker_payload: Mapping[str, Any],
    worker_index: int,
    attempt_id: str,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
) -> DatabricksClusterHourLedger:
    """Authorize and reserve one five-hour task under the campaign caps."""

    _validate_producer_worker_index(worker_index)
    _validate_single_producer_submit_payload(
        submit_payload,
        worker_index=worker_index,
    )
    worker_payload_file_sha256 = _worker_payload_file_sha256(worker_payload)
    _require_authorized_worker_payload(
        worker_payload,
        qualification_launch_authorization,
        worker_index=worker_index,
        expected_worker_payload_file_sha256=worker_payload_file_sha256,
    )
    _validate_worker_payload_submit_binding(
        submit_payload,
        worker_payload=worker_payload,
        worker_index=worker_index,
        expected_worker_payload_file_sha256=worker_payload_file_sha256,
    )
    path = Path(ledger_path)
    _require_matching_qualification_ledger(
        path,
        qualification_launch_authorization,
    )

    raise RuntimeError(
        "single-worker Q8 reservation is nonpublication; use the exact 16-worker wave"
    )

    return reserve_databricks_run_attempt_json(
        path,
        submit_payload,
        attempt_id=attempt_id,
        workload_id=f"publication-latency-handoff-worker-{worker_index:02d}",
        reservation_validator=_publication_latency_handoff_reservation_validator(
            path,
            worker_payload=worker_payload,
            worker_index=worker_index,
            qualification_launch_authorization=qualification_launch_authorization,
            expected_worker_payload_file_sha256=worker_payload_file_sha256,
        ),
    )


def reserve_and_submit_publication_latency_handoff_worker(
    workspace: DatabricksWorkspaceConfig,
    submit_payload: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    worker_payload: Mapping[str, Any],
    worker_index: int,
    attempt_id: str,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    opener: DatabricksURLOpener | None = None,
) -> dict[str, Any]:
    """Authorize, reserve, then submit the exact same canonical payload bytes."""

    _validate_producer_worker_index(worker_index)
    _validate_single_producer_submit_payload(
        submit_payload,
        worker_index=worker_index,
    )
    worker_payload_file_sha256 = _worker_payload_file_sha256(worker_payload)
    _require_authorized_worker_payload(
        worker_payload,
        qualification_launch_authorization,
        worker_index=worker_index,
        expected_worker_payload_file_sha256=worker_payload_file_sha256,
    )
    _validate_worker_payload_submit_binding(
        submit_payload,
        worker_payload=worker_payload,
        worker_index=worker_index,
        expected_worker_payload_file_sha256=worker_payload_file_sha256,
    )
    path = Path(ledger_path)
    _require_matching_qualification_ledger(
        path,
        qualification_launch_authorization,
    )
    raise RuntimeError(
        "single-worker Q8 submission is nonpublication; use the exact 16-worker wave"
    )
    response = reserve_and_submit_databricks_run(
        workspace,
        submit_payload,
        ledger_path=path,
        attempt_id=attempt_id,
        workload_id=f"publication-latency-handoff-worker-{worker_index:02d}",
        reservation_validator=_publication_latency_handoff_reservation_validator(
            path,
            worker_payload=worker_payload,
            worker_index=worker_index,
            qualification_launch_authorization=qualification_launch_authorization,
            expected_worker_payload_file_sha256=worker_payload_file_sha256,
        ),
        opener=opener,
    )
    _databricks_cloud_id(response.get("run_id"), field_name="submit response run_id")
    record_databricks_run_submission_receipt_json(
        path,
        attempt_id=attempt_id,
        submit_response=response,
    )
    return response


def reserve_and_submit_publication_latency_handoff_worker_wave(
    workspace: DatabricksWorkspaceConfig,
    submit_payloads: Sequence[Mapping[str, Any]],
    *,
    ledger_path: str | Path,
    worker_payloads: Sequence[Mapping[str, Any]],
    attempt_ids_by_worker: Mapping[int, str],
    phase_lease_root: str | Path,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    opener: DatabricksURLOpener | None = None,
) -> tuple[
    tuple[dict[str, Any], ...],
    PublicationLatencyHandoffSubmissionAuthorization,
]:
    """Atomically admit all sixteen Q8 producers before the first POST."""

    submissions = tuple(submit_payloads)
    workers = tuple(worker_payloads)
    worker_indexes = tuple(range(PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS))
    if len(submissions) != len(worker_indexes) or len(workers) != len(worker_indexes):
        raise ValueError("Q8 production wave requires exactly sixteen members")
    if set(attempt_ids_by_worker) != set(worker_indexes):
        raise ValueError("Q8 attempt IDs must cover workers 0..15 exactly")
    durable_output_root = _common_q8_durable_output_root(workers)
    predecessor = qualification_launch_authorization.ledger_prefix
    live = _require_matching_qualification_ledger(
        ledger_path,
        qualification_launch_authorization,
    )
    if databricks_ledger_prefix(live) != predecessor:
        raise ValueError("Q8 wave predecessor is not the complete live ledger")

    attempts = tuple(attempt_ids_by_worker[index] for index in worker_indexes)
    expected_attempts = tuple(
        publication_latency_handoff_worker_attempt_id(
            workers[index], worker_index=index
        )
        for index in worker_indexes
    )
    if attempts != expected_attempts:
        raise ValueError("Q8 attempt IDs differ from the frozen worker identities")
    requests: list[DatabricksRunAttemptReservationRequest] = []
    payload_digests: list[str] = []
    for index, submit_payload, worker_payload in zip(
        worker_indexes,
        submissions,
        workers,
        strict=True,
    ):
        expected_worker_sha256 = _worker_payload_file_sha256(worker_payload)
        _validate_single_producer_submit_payload(submit_payload, worker_index=index)
        _require_authorized_worker_payload(
            worker_payload,
            qualification_launch_authorization,
            worker_index=index,
            expected_worker_payload_file_sha256=expected_worker_sha256,
        )
        _validate_worker_payload_submit_binding(
            submit_payload,
            worker_payload=worker_payload,
            worker_index=index,
            expected_worker_payload_file_sha256=expected_worker_sha256,
        )
        require_databricks_run_idempotency_token(
            submit_payload,
            attempt_id=attempts[index],
        )
        _snapshot, canonical = canonical_databricks_submit_payload_snapshot(
            submit_payload
        )
        payload_digests.append(sha256(canonical).hexdigest())
        requests.append(
            DatabricksRunAttemptReservationRequest(
                attempt_id=attempts[index],
                workload_id=f"publication-latency-handoff-worker-{index:02d}",
                submit_payload=submit_payload,
            )
        )

    lease_root = _create_q8_phase_lease_root(phase_lease_root)
    lease_record: dict[str, Any] = {
        "attempt_ids": list(attempts),
        "closed_record_sha256": "",
        "durable_output_root": durable_output_root,
        "ledger_path_sha256": qualification_launch_authorization.ledger_path_sha256,
        "predecessor_prefix": predecessor.to_record(),
        "record_type": "cachet.publication_q8_handoff_phase_lease.v1",
        "submit_payload_sha256": payload_digests,
    }
    lease_record["closed_record_sha256"] = _closed_record_sha256(lease_record)
    _write_canonical_json_exclusive(lease_record, lease_root / "phase-lease.json")
    _sync_file(lease_root / "phase-lease.json")
    _sync_directory(lease_root)

    def validate_batch(
        batch_live: DatabricksClusterHourLedger,
        reservations: tuple[DatabricksClusterHourReservation, ...],
        snapshots: tuple[Mapping[str, Any], ...],
    ) -> None:
        if databricks_ledger_path_sha256(ledger_path) != (
            qualification_launch_authorization.ledger_path_sha256
        ):
            raise ValueError("Q8 batch ledger path binding drift")
        require_databricks_ledger_prefix(batch_live, predecessor)
        if len(reservations) != len(worker_indexes) or len(snapshots) != len(
            worker_indexes
        ):
            raise ValueError("Q8 batch must contain exactly sixteen producers")
        for index, reservation, snapshot, worker_payload in zip(
            worker_indexes,
            reservations,
            snapshots,
            workers,
            strict=True,
        ):
            expected_worker_sha256 = _worker_payload_file_sha256(worker_payload)
            _validate_single_producer_submit_payload(snapshot, worker_index=index)
            _validate_worker_payload_submit_binding(
                snapshot,
                worker_payload=worker_payload,
                worker_index=index,
                expected_worker_payload_file_sha256=expected_worker_sha256,
            )
            require_databricks_run_idempotency_token(
                snapshot,
                attempt_id=attempts[index],
            )
            if (
                reservation.attempt_id != attempts[index]
                or reservation.submit_payload_sha256 != payload_digests[index]
                or reservation.reserved_cluster_hours != 5.0
            ):
                raise ValueError("Q8 batch reservation member drift")
        proposed_hours = sum(item.reserved_cluster_hours for item in reservations)
        proposed_tasks = sum(len(item.task_timeout_seconds) for item in reservations)
        if batch_live.cap_cluster_hours != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS:
            raise ValueError("Q8 generation requires the 1024-hour ledger")
        if (
            batch_live.active_reserved_task_count + proposed_tasks
            > PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
        ):
            raise ValueError("Q8 wave exceeds the global 16-job concurrency cap")
        if (
            batch_live.active_reserved_cluster_hours + proposed_hours
            > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
        ):
            raise ValueError("Q8 wave exceeds the active 900-hour cap")
        if (
            batch_live.accounted_cluster_hours + proposed_hours
            > MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
            - PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
        ):
            raise ValueError("Q8 wave consumes the 124-hour headroom")

    try:
        _batch_ledger, batch_authorization = (
            reserve_databricks_run_attempt_batch_authorized_json(
                ledger_path,
                tuple(requests),
                expected_predecessor_prefix=predecessor,
                batch_validator=validate_batch,
            )
        )
    except BaseException:
        _remove_empty_q8_phase_lease_root(lease_root)
        raise
    require_databricks_batch_reservation_authorization(
        batch_authorization,
        expected_predecessor_prefix=predecessor,
        expected_attempt_ids=attempts,
        expected_submit_payload_sha256s=payload_digests,
    )
    batch_record = {
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "record_type": "cachet.publication_q8_handoff_batch_reserved.v1",
    }
    batch_record["closed_record_sha256"] = _closed_record_sha256(batch_record)
    _write_canonical_json_exclusive(batch_record, lease_root / "batch-reserved.json")
    _sync_file(lease_root / "batch-reserved.json")
    _sync_directory(lease_root)
    submission_authorization = _issue_q8_submission_authorization(
        batch_authorization,
        lease_root,
        durable_output_root=durable_output_root,
    )

    responses: list[dict[str, Any]] = []
    for index, submit_payload in zip(worker_indexes, submissions, strict=True):
        intent_path = lease_root / f"worker-{index:02d}.post-intent.json"
        intent = {
            "attempt_id": attempts[index],
            "batch_prefix": batch_authorization.batch_prefix.to_record(),
            "closed_record_sha256": "",
            "record_type": "cachet.publication_q8_handoff_post_intent.v1",
            "submit_payload_sha256": payload_digests[index],
            "worker_index": index,
        }
        intent["closed_record_sha256"] = _closed_record_sha256(intent)
        _write_canonical_json_exclusive(intent, intent_path)
        _sync_file(intent_path)
        _sync_directory(lease_root)
        response = submit_pre_reserved_databricks_run(
            workspace,
            submit_payload,
            ledger_path=ledger_path,
            attempt_id=attempts[index],
            batch_authorization=batch_authorization,
            opener=opener,
        )
        receipt_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
        receipt = next(
            item
            for item in receipt_ledger.submission_receipts
            if item.attempt_id == attempts[index]
        )
        receipt_record = {
            "attempt_id": attempts[index],
            "closed_record_sha256": "",
            "record_type": "cachet.publication_q8_handoff_submit_receipt.v1",
            "run_id": receipt.run_id,
            "submit_payload_sha256": receipt.submit_payload_sha256,
            "submit_response_sha256": receipt.submit_response_sha256,
            "worker_index": index,
        }
        receipt_record["closed_record_sha256"] = _closed_record_sha256(receipt_record)
        _write_canonical_json_exclusive(
            receipt_record,
            lease_root / f"worker-{index:02d}.receipt.json",
        )
        _sync_file(lease_root / f"worker-{index:02d}.receipt.json")
        intent_path.unlink()
        _sync_directory(lease_root)
        responses.append(response)
    return tuple(responses), submission_authorization


def resume_publication_latency_handoff_worker_wave(
    workspace: DatabricksWorkspaceConfig,
    submit_payloads: Sequence[Mapping[str, Any]],
    *,
    ledger_path: str | Path,
    worker_payloads: Sequence[Mapping[str, Any]],
    attempt_ids_by_worker: Mapping[int, str],
    phase_lease_root: str | Path,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    opener: DatabricksURLOpener | None = None,
) -> tuple[
    tuple[dict[str, Any], ...],
    PublicationLatencyHandoffSubmissionAuthorization,
]:
    """Resume the exact Q8 producer wave from its durable phase lease."""

    submissions = tuple(submit_payloads)
    workers = tuple(worker_payloads)
    indexes = tuple(range(PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS))
    if len(submissions) != len(indexes) or len(workers) != len(indexes):
        raise ValueError("Q8 production wave requires exactly sixteen members")
    if set(attempt_ids_by_worker) != set(indexes):
        raise ValueError("Q8 attempt IDs must cover workers 0..15 exactly")
    durable_output_root = _common_q8_durable_output_root(workers)
    predecessor = qualification_launch_authorization.ledger_prefix
    live = _require_matching_qualification_ledger(
        ledger_path, qualification_launch_authorization
    )
    require_databricks_ledger_prefix(live, predecessor)
    attempts = tuple(attempt_ids_by_worker[index] for index in indexes)
    expected_attempts = tuple(
        publication_latency_handoff_worker_attempt_id(
            workers[index], worker_index=index
        )
        for index in indexes
    )
    if attempts != expected_attempts:
        raise ValueError("Q8 attempt IDs differ from the frozen worker identities")
    requests: list[DatabricksRunAttemptReservationRequest] = []
    payload_digests: list[str] = []
    for index, submit_payload, worker_payload in zip(
        indexes, submissions, workers, strict=True
    ):
        worker_sha256 = _worker_payload_file_sha256(worker_payload)
        _validate_single_producer_submit_payload(submit_payload, worker_index=index)
        _require_authorized_worker_payload(
            worker_payload,
            qualification_launch_authorization,
            worker_index=index,
            expected_worker_payload_file_sha256=worker_sha256,
        )
        _validate_worker_payload_submit_binding(
            submit_payload,
            worker_payload=worker_payload,
            worker_index=index,
            expected_worker_payload_file_sha256=worker_sha256,
        )
        require_databricks_run_idempotency_token(
            submit_payload, attempt_id=attempts[index]
        )
        _snapshot, canonical = canonical_databricks_submit_payload_snapshot(
            submit_payload
        )
        payload_digests.append(sha256(canonical).hexdigest())
        requests.append(
            DatabricksRunAttemptReservationRequest(
                attempt_id=attempts[index],
                workload_id=f"publication-latency-handoff-worker-{index:02d}",
                submit_payload=submit_payload,
            )
        )
    lease_root = Path(phase_lease_root).expanduser().absolute()
    for candidate in (lease_root, *lease_root.parents):
        if candidate.is_symlink():
            raise ValueError(f"Q8 phase lease path traverses a symlink: {candidate}")
    if not lease_root.is_dir() or lease_root.is_symlink():
        raise ValueError("Q8 resume requires the existing real phase lease")
    expected_lease: dict[str, Any] = {
        "attempt_ids": list(attempts),
        "closed_record_sha256": "",
        "durable_output_root": durable_output_root,
        "ledger_path_sha256": qualification_launch_authorization.ledger_path_sha256,
        "predecessor_prefix": predecessor.to_record(),
        "record_type": "cachet.publication_q8_handoff_phase_lease.v1",
        "submit_payload_sha256": payload_digests,
    }
    expected_lease["closed_record_sha256"] = _closed_record_sha256(expected_lease)
    if (
        _read_q8_controller_record(lease_root / "phase-lease.json", "Q8 phase lease")
        != expected_lease
    ):
        raise ValueError("Q8 phase lease differs from the frozen wave")
    batch_authorization = replay_databricks_run_attempt_batch_authorization_json(
        ledger_path,
        tuple(requests),
        expected_predecessor_prefix=predecessor,
    )
    require_databricks_publication_batch_admission(live, batch_authorization)
    expected_batch: dict[str, Any] = {
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "record_type": "cachet.publication_q8_handoff_batch_reserved.v1",
    }
    expected_batch["closed_record_sha256"] = _closed_record_sha256(expected_batch)
    batch_path = lease_root / "batch-reserved.json"
    if batch_path.exists() or batch_path.is_symlink():
        if _read_q8_controller_record(batch_path, "Q8 batch marker") != expected_batch:
            raise ValueError("Q8 batch marker differs from the ledger batch")
    else:
        _write_canonical_json_exclusive(expected_batch, batch_path)
        _sync_file(batch_path)
        _sync_directory(lease_root)
    submission_authorization = _issue_q8_submission_authorization(
        batch_authorization,
        lease_root,
        durable_output_root=durable_output_root,
    )
    responses: list[dict[str, Any]] = []
    for index, submit_payload in zip(indexes, submissions, strict=True):
        intent_path = lease_root / f"worker-{index:02d}.post-intent.json"
        receipt_path = lease_root / f"worker-{index:02d}.receipt.json"
        expected_intent: dict[str, Any] = {
            "attempt_id": attempts[index],
            "batch_prefix": batch_authorization.batch_prefix.to_record(),
            "closed_record_sha256": "",
            "record_type": "cachet.publication_q8_handoff_post_intent.v1",
            "submit_payload_sha256": payload_digests[index],
            "worker_index": index,
        }
        expected_intent["closed_record_sha256"] = _closed_record_sha256(expected_intent)
        if intent_path.exists() or intent_path.is_symlink():
            if (
                _read_q8_controller_record(
                    intent_path, f"Q8 worker {index} post intent"
                )
                != expected_intent
            ):
                raise ValueError("Q8 post intent drift")
        elif not receipt_path.exists():
            _write_canonical_json_exclusive(expected_intent, intent_path)
            _sync_file(intent_path)
            _sync_directory(lease_root)
        response = resume_pre_reserved_databricks_run(
            workspace,
            submit_payload,
            ledger_path=ledger_path,
            attempt_id=attempts[index],
            batch_authorization=batch_authorization,
            opener=opener,
        )
        ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
        receipt = next(
            item
            for item in ledger.submission_receipts
            if item.attempt_id == attempts[index]
        )
        expected_receipt: dict[str, Any] = {
            "attempt_id": attempts[index],
            "closed_record_sha256": "",
            "record_type": "cachet.publication_q8_handoff_submit_receipt.v1",
            "run_id": receipt.run_id,
            "submit_payload_sha256": receipt.submit_payload_sha256,
            "submit_response_sha256": receipt.submit_response_sha256,
            "worker_index": index,
        }
        expected_receipt["closed_record_sha256"] = _closed_record_sha256(
            expected_receipt
        )
        if receipt_path.exists() or receipt_path.is_symlink():
            if (
                _read_q8_controller_record(receipt_path, f"Q8 worker {index} receipt")
                != expected_receipt
            ):
                raise ValueError("Q8 durable submit receipt drift")
        else:
            _write_canonical_json_exclusive(expected_receipt, receipt_path)
            _sync_file(receipt_path)
        if intent_path.is_file() and not intent_path.is_symlink():
            intent_path.unlink()
        _sync_directory(lease_root)
        responses.append(response)
    expected_names = {"phase-lease.json", "batch-reserved.json"} | {
        f"worker-{index:02d}.receipt.json" for index in indexes
    }
    if {item.name for item in lease_root.iterdir()} != expected_names:
        raise ValueError("resumed Q8 phase lease directory is not closed")
    return tuple(responses), submission_authorization


def require_publication_latency_handoff_submission_authorization(
    authorization: object,
) -> DatabricksBatchReservationAuthorization:
    """Replay the Q8 phase lease and marker behind submission authority."""

    if not isinstance(
        authorization, PublicationLatencyHandoffSubmissionAuthorization
    ):
        raise TypeError(
            "Q8 publication collection requires "
            "PublicationLatencyHandoffSubmissionAuthorization"
        )
    lease, marker = _validate_q8_submission_phase_files(
        authorization.phase_lease_root,
        authorization.batch_authorization,
        durable_output_root=authorization.durable_output_root,
    )
    observed = (
        _q8_phase_lease_root_sha256(authorization.phase_lease_root),
        _file_sha256(authorization.phase_lease_root / "phase-lease.json"),
        _required_string(lease, "closed_record_sha256"),
        _file_sha256(authorization.phase_lease_root / "batch-reserved.json"),
        _required_string(marker, "closed_record_sha256"),
    )
    expected = (
        authorization.phase_lease_root_sha256,
        authorization.phase_lease_file_sha256,
        authorization.phase_lease_closed_record_sha256,
        authorization.batch_marker_file_sha256,
        authorization.batch_marker_closed_record_sha256,
    )
    if observed != expected:
        raise ValueError("Q8 durable phase authorization evidence drift")
    return authorization.batch_authorization


def _issue_q8_submission_authorization(
    batch_authorization: DatabricksBatchReservationAuthorization,
    phase_lease_root: Path,
    *,
    durable_output_root: str,
) -> PublicationLatencyHandoffSubmissionAuthorization:
    return PublicationLatencyHandoffSubmissionAuthorization(
        batch_authorization=batch_authorization,
        phase_lease_root=phase_lease_root,
        durable_output_root=durable_output_root,
        _issuer=_SUBMISSION_AUTHORIZATION_ISSUER,
    )


def _validate_q8_submission_phase_files(
    phase_lease_root: Path,
    batch_authorization: DatabricksBatchReservationAuthorization,
    *,
    durable_output_root: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(batch_authorization, DatabricksBatchReservationAuthorization):
        raise TypeError("Q8 phase batch authorization has the wrong type")
    root = Path(phase_lease_root).expanduser().absolute()
    _reject_q8_symlink_ancestors(root, "Q8 phase lease")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Q8 submission authority requires a real phase lease")
    lease = _read_q8_controller_record(
        root / "phase-lease.json", "Q8 phase lease"
    )
    expected_lease: dict[str, Any] = {
        "attempt_ids": list(batch_authorization.attempt_ids),
        "closed_record_sha256": "",
        "durable_output_root": _normalized_q8_durable_output_root(
            durable_output_root
        ),
        "ledger_path_sha256": batch_authorization.ledger_path_sha256,
        "predecessor_prefix": batch_authorization.predecessor_prefix.to_record(),
        "record_type": "cachet.publication_q8_handoff_phase_lease.v1",
        "submit_payload_sha256": list(
            batch_authorization.submit_payload_sha256s
        ),
    }
    expected_lease["closed_record_sha256"] = _closed_record_sha256(expected_lease)
    if lease != expected_lease:
        raise ValueError("Q8 phase lease differs from the authorized atomic batch")
    marker = _read_q8_controller_record(
        root / "batch-reserved.json", "Q8 batch marker"
    )
    expected_marker: dict[str, Any] = {
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "record_type": "cachet.publication_q8_handoff_batch_reserved.v1",
    }
    expected_marker["closed_record_sha256"] = _closed_record_sha256(
        expected_marker
    )
    if marker != expected_marker:
        raise ValueError("Q8 batch marker differs from the authorized atomic batch")
    return lease, marker


def _normalized_q8_durable_output_root(value: object) -> str:
    if not isinstance(value, str) or not value.rstrip("/"):
        raise ValueError("Q8 durable_output_root must be a non-empty string")
    return value.rstrip("/")


def _common_q8_durable_output_root(
    worker_payloads: Sequence[Mapping[str, Any]],
) -> str:
    roots = {
        _normalized_q8_durable_output_root(
            _required_string(payload, "durable_output_root")
        )
        for payload in worker_payloads
    }
    if len(roots) != 1:
        raise ValueError("Q8 workers must share one durable_output_root")
    return next(iter(roots))


def _q8_phase_lease_root_sha256(path: Path) -> str:
    root = Path(path).expanduser().absolute()
    _reject_q8_symlink_ancestors(root, "Q8 phase lease")
    canonical = root.resolve(strict=True)
    return _canonical_sha256(
        {
            "domain": "cachet.publication.q8_handoff.phase_lease_path.v1",
            "resolved_absolute_path": str(canonical),
        }
    )


def _publication_latency_handoff_reservation_validator(
    ledger_path: Path,
    *,
    worker_payload: Mapping[str, Any],
    worker_index: int,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    expected_worker_payload_file_sha256: str,
) -> Callable[[DatabricksClusterHourReservation, Mapping[str, Any]], None]:
    def validate_reservation(
        reservation: DatabricksClusterHourReservation,
        snapshot: Mapping[str, Any],
    ) -> None:
        _validate_single_producer_submit_payload(
            snapshot,
            worker_index=worker_index,
        )
        _require_authorized_worker_payload(
            worker_payload,
            qualification_launch_authorization,
            worker_index=worker_index,
            expected_worker_payload_file_sha256=expected_worker_payload_file_sha256,
        )
        _validate_worker_payload_submit_binding(
            snapshot,
            worker_payload=worker_payload,
            worker_index=worker_index,
            expected_worker_payload_file_sha256=expected_worker_payload_file_sha256,
        )
        if reservation.reserved_cluster_hours != 5.0:
            raise ValueError("one latency handoff producer must reserve five GPU-hours")
        ledger = _require_matching_qualification_ledger(
            ledger_path,
            qualification_launch_authorization,
        )
        if ledger.cap_cluster_hours != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS:
            raise ValueError("publication ledger must use the 1024-hour cap")
        if (
            ledger.active_reserved_task_count + len(reservation.task_timeout_seconds)
            > PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
        ):
            raise ValueError(
                "producer reservation exceeds the global 16-job concurrency cap"
            )
        projected_active = (
            ledger.active_reserved_cluster_hours + reservation.reserved_cluster_hours
        )
        if projected_active > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS:
            raise ValueError("producer reservation exceeds the 900-hour active cap")
        projected_accounted = (
            ledger.accounted_cluster_hours + reservation.reserved_cluster_hours
        )
        if projected_accounted > (
            MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
            - PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
        ):
            raise ValueError("producer reservation would consume 124-hour headroom")

    return validate_reservation


def build_publication_latency_handoff_databricks_attestation(
    submit_payload: Mapping[str, Any],
    submit_response: Mapping[str, Any],
    terminal_run: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    durable_output_root: str | Path,
    worker_index: int,
    attempt_id: str,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    submission_authorization: PublicationLatencyHandoffSubmissionAuthorization,
) -> dict[str, Any]:
    """Sanitize and close one direct attempt-0 ``runs/get`` response."""

    _validate_producer_worker_index(worker_index)
    _validate_single_producer_submit_payload(
        submit_payload,
        worker_index=worker_index,
    )
    snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(
        submit_payload
    )
    response_snapshot, canonical_response = (
        canonical_databricks_submit_payload_snapshot(submit_response)
    )
    terminal_snapshot, canonical_terminal = (
        canonical_databricks_submit_payload_snapshot(terminal_run)
    )
    _validate_single_producer_submit_payload(snapshot, worker_index=worker_index)
    submit_payload_sha256 = sha256(canonical_payload).hexdigest()
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    batch_reservation_authorization = (
        require_publication_latency_handoff_submission_authorization(
            submission_authorization
        )
    )
    if (
        batch_reservation_authorization.predecessor_prefix
        != qualification_launch_authorization.ledger_prefix
        or batch_reservation_authorization.ledger_path_sha256
        != qualification_launch_authorization.ledger_path_sha256
        or databricks_ledger_path_sha256(ledger_path)
        != qualification_launch_authorization.ledger_path_sha256
    ):
        raise ValueError("Q8 attestation batch/ledger authority drift")
    if attempt_id not in batch_reservation_authorization.attempt_ids:
        raise ValueError("Q8 attestation attempt is outside the atomic batch")
    batch_index = batch_reservation_authorization.attempt_ids.index(attempt_id)
    if (
        batch_reservation_authorization.submit_payload_sha256s[batch_index]
        != submit_payload_sha256
    ):
        raise ValueError("Q8 attestation payload differs from atomic batch")
    require_databricks_ledger_prefix(
        ledger, batch_reservation_authorization.batch_prefix
    )
    reservation = next(
        (item for item in ledger.reservations if item.attempt_id == attempt_id),
        None,
    )
    receipt = next(
        (item for item in ledger.submission_receipts if item.attempt_id == attempt_id),
        None,
    )
    expected_workload_id = f"publication-latency-handoff-worker-{worker_index:02d}"
    if (
        reservation is None
        or reservation.workload_id != expected_workload_id
        or reservation.submit_payload_sha256 != submit_payload_sha256
        or reservation.reserved_cluster_hours != 5.0
        or receipt is None
        or receipt.submit_payload_sha256 != submit_payload_sha256
        or receipt.submit_response_sha256 != sha256(canonical_response).hexdigest()
    ):
        raise ValueError(
            "Databricks attestation has no exact reservation/submission receipt"
        )
    if attempt_id in ledger.closed_attempt_ids:
        raise ValueError("Databricks attestation must precede ledger reconciliation")

    parent_run_id = _databricks_cloud_id(
        response_snapshot.get("run_id"),
        field_name="submit response run_id",
    )
    if receipt.run_id != parent_run_id:
        raise ValueError("submission receipt belongs to another Databricks run")
    original_attempt_run_id = _databricks_cloud_id(
        terminal_snapshot.get("original_attempt_run_id"),
        field_name="terminal original_attempt_run_id",
    )
    if original_attempt_run_id != parent_run_id:
        raise ValueError("terminal run is not the original submitted attempt")
    raw_tasks = _mapping_sequence(
        terminal_snapshot.get("tasks"),
        field_name="terminal run tasks",
    )
    if len(raw_tasks) != 1:
        raise ValueError("publication producer run must contain exactly one task")
    attempt_number = _required_int(raw_tasks[0], "attempt_number")
    if attempt_number != 0:
        raise ValueError("publication producer must use Databricks attempt_number=0")
    raw_repair_history = terminal_snapshot.get("repair_history")
    repair_history = (
        ()
        if raw_repair_history is None
        else _mapping_sequence(
            raw_repair_history,
            field_name="terminal run repair_history",
        )
    )
    repair_count = len(repair_history)
    if repair_count != 0:
        raise ValueError("publication producer must not use a repaired run")
    terminal_run_status = summarize_databricks_run(
        terminal_snapshot,
        submit_payload=snapshot,
    )
    status = databricks_run_status_record(terminal_run_status)
    if status is None:
        raise ValueError("Databricks terminal sidecar has no sanitized status record")
    _validate_publication_latency_handoff_terminal_status(
        status,
        worker_index=worker_index,
        expected_run_name=_required_string(snapshot, "run_name"),
        submit_payload_sha256=submit_payload_sha256,
    )
    if (
        _databricks_cloud_id(
            status.get("run_id"),
            field_name="terminal parent run_id",
        )
        != parent_run_id
    ):
        raise ValueError("terminal status belongs to a different submitted parent run")
    status_submit = _required_mapping(status, "submit_payload")
    if status_submit.get("sha256") != submit_payload_sha256:
        raise ValueError("terminal status submit payload differs from reserved payload")
    status_tasks = _mapping_sequence(status.get("tasks"), field_name="status.tasks")
    if len(status_tasks) != 1:
        raise ValueError("publication producer status must contain exactly one task")
    task = status_tasks[0]
    expected_task_key = f"latency_handoff_worker_{worker_index:02d}"
    if task.get("task_key") != expected_task_key:
        raise ValueError("terminal task key does not match producer worker")
    task_run_id = _databricks_cloud_id(
        task.get("run_id"),
        field_name="terminal task run_id",
    )
    if task_run_id == parent_run_id:
        raise ValueError("parent and task Databricks run IDs must be distinct")
    cluster_id = _databricks_cloud_id(
        task.get("cluster_id"),
        field_name="terminal task cluster_id",
    )
    parent_cluster_id = status.get("cluster_id")
    if (
        parent_cluster_id is not None
        and _databricks_cloud_id(
            parent_cluster_id,
            field_name="terminal parent cluster_id",
        )
        != cluster_id
    ):
        raise ValueError("parent and task terminal status use different clusters")
    parent_start = _positive_epoch_millis(status, "start_time")
    parent_end = _positive_epoch_millis(status, "end_time")
    task_start = _positive_epoch_millis(task, "start_time")
    task_end = _positive_epoch_millis(task, "end_time")
    if not parent_start <= task_start < task_end <= parent_end:
        raise ValueError("Databricks parent/task timestamps are not causally nested")
    actual_gpu_duration_seconds = (task_end - task_start) / 1000.0
    if actual_gpu_duration_seconds > (
        PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS
    ):
        raise ValueError("Databricks task duration exceeds its five-hour bound")

    root = Path(local_path(str(durable_output_root))).expanduser().resolve()
    result_relative_path = f"worker-results/worker-{worker_index:02d}.json"
    result_path = _confined_relative_path(
        root,
        result_relative_path,
        field_name="attested worker result",
    )
    worker_result = _read_worker_result(result_path)
    if worker_result.get("worker_index") != worker_index:
        raise ValueError("attested worker result belongs to another worker")
    record: dict[str, Any] = {
        "attempt": {
            "attempt_id": attempt_id,
            "ledger_id": ledger.ledger_id,
            "ledger_path_sha256": (
                qualification_launch_authorization.ledger_path_sha256
            ),
            "producer_batch_prefix": (
                batch_reservation_authorization.batch_prefix.to_record()
            ),
            "reserved_gpu_hours": reservation.reserved_cluster_hours,
            "submit_response_sha256": receipt.submit_response_sha256,
            "submit_payload_sha256": submit_payload_sha256,
            "worker_index": worker_index,
            "workload_id": expected_workload_id,
        },
        "closed_record_sha256": "",
        "cloud_execution": {
            "actual_gpu_duration_seconds": actual_gpu_duration_seconds,
            "attempt_number": attempt_number,
            "cluster_id": cluster_id,
            "control_plane_status_sha256": sha256(canonical_terminal).hexdigest(),
            "life_cycle_state": _required_string(status, "life_cycle_state"),
            "original_attempt_run_id": original_attempt_run_id,
            "parent_end_time_epoch_ms": parent_end,
            "parent_run_id": parent_run_id,
            "parent_start_time_epoch_ms": parent_start,
            "repair_count": repair_count,
            "result_state": _required_string(status, "result_state"),
            "task_end_time_epoch_ms": task_end,
            "task_key": expected_task_key,
            "task_run_id": task_run_id,
            "task_start_time_epoch_ms": task_start,
            "terminal_state": "succeeded",
        },
        "record_type": (PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_RECORD_TYPE),
        "schema_version": (
            PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_SCHEMA_VERSION
        ),
        "worker_result": {
            "closed_record_sha256": worker_result["closed_record_sha256"],
            "file_sha256": _file_sha256(result_path),
            "relative_path": result_relative_path,
        },
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    _validate_publication_latency_handoff_databricks_attestation_record(record)
    return record


def write_publication_latency_handoff_databricks_attestation(
    record: Mapping[str, Any],
    path: str | Path,
) -> PublicationLatencyHandoffDatabricksAttestationBinding:
    """Persist one canonical cloud attestation and return its immutable binding."""

    _validate_publication_latency_handoff_databricks_attestation_record(record)
    worker_index = _required_int(_required_mapping(record, "attempt"), "worker_index")
    destination = Path(path).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        observed = _read_q8_controller_record(destination, "Q8 Databricks attestation")
        if observed != dict(record):
            raise ValueError("existing Q8 Databricks attestation differs")
    else:
        _write_canonical_json_exclusive(record, destination)
    _sync_file(destination)
    _sync_directory(destination.parent)
    _sync_directory(destination.parent.parent)
    return PublicationLatencyHandoffDatabricksAttestationBinding(
        worker_index=worker_index,
        path=destination,
        file_sha256=_file_sha256(destination),
        closed_record_sha256=_required_string(record, "closed_record_sha256"),
    )


def read_publication_latency_handoff_databricks_attestation(
    binding: PublicationLatencyHandoffDatabricksAttestationBinding,
    *,
    durable_output_root: str | Path,
) -> dict[str, Any]:
    """Re-authenticate an attestation and its exact worker-result file join."""

    if not isinstance(
        binding,
        PublicationLatencyHandoffDatabricksAttestationBinding,
    ):
        raise TypeError("binding has the wrong type")
    root = Path(local_path(str(durable_output_root))).expanduser().resolve()
    expected_path = (
        root
        / PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY
        / (f"worker-{binding.worker_index:02d}.json")
    )
    if binding.path.expanduser().resolve() != expected_path.resolve():
        raise ValueError("Databricks attestation path is not the canonical worker path")
    if not expected_path.is_file() or expected_path.is_symlink():
        raise ValueError("Databricks attestation must be a real file")
    content = expected_path.read_bytes()
    if not hmac.compare_digest(sha256(content).hexdigest(), binding.file_sha256):
        raise ValueError("Databricks attestation file SHA-256 drift")
    record = _json_object(content, field_name="Databricks execution attestation")
    if content != _canonical_json_bytes(record, pretty=True):
        raise ValueError("Databricks execution attestation is not canonical JSON")
    _validate_publication_latency_handoff_databricks_attestation_record(record)
    if record.get("closed_record_sha256") != binding.closed_record_sha256:
        raise ValueError("Databricks attestation closed-record binding drift")
    attempt = _required_mapping(record, "attempt")
    if _required_int(attempt, "worker_index") != binding.worker_index:
        raise ValueError("Databricks attestation worker binding drift")
    worker_binding = _required_mapping(record, "worker_result")
    expected_result_relative = f"worker-results/worker-{binding.worker_index:02d}.json"
    if worker_binding.get("relative_path") != expected_result_relative:
        raise ValueError("Databricks attestation worker-result path drift")
    result_path = _confined_relative_path(
        root,
        expected_result_relative,
        field_name="attested worker result",
    )
    result = _read_worker_result(result_path)
    if result.get("worker_index") != binding.worker_index:
        raise ValueError("Databricks attestation worker-result identity drift")
    if result.get("closed_record_sha256") != worker_binding.get(
        "closed_record_sha256"
    ) or not hmac.compare_digest(
        _file_sha256(result_path),
        _required_string(worker_binding, "file_sha256"),
    ):
        raise ValueError("Databricks attestation worker-result file binding drift")
    return record


def reconcile_publication_latency_handoff_worker_attempt_json(
    ledger_path: str | Path,
    *,
    worker_index: int,
    attempt_id: str,
    durable_output_root: str | Path,
    attestation: PublicationLatencyHandoffDatabricksAttestationBinding,
    terminal_run: Mapping[str, Any],
) -> DatabricksClusterHourLedger:
    """Derive terminal state and duration from one closed cloud attestation."""

    _validate_producer_worker_index(worker_index)
    if attestation.worker_index != worker_index:
        raise ValueError("Databricks attestation belongs to another worker")
    record = read_publication_latency_handoff_databricks_attestation(
        attestation,
        durable_output_root=durable_output_root,
    )
    attempt = _required_mapping(record, "attempt")
    if attempt.get("attempt_id") != attempt_id:
        raise ValueError("Databricks attestation belongs to another attempt")
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    reservation = next(
        (item for item in ledger.reservations if item.attempt_id == attempt_id),
        None,
    )
    if (
        reservation is None
        or reservation.workload_id != attempt.get("workload_id")
        or reservation.submit_payload_sha256 != attempt.get("submit_payload_sha256")
        or ledger.ledger_id != attempt.get("ledger_id")
    ):
        raise ValueError("attestation does not match the reserved producer attempt")
    cloud = _required_mapping(record, "cloud_execution")
    terminal_snapshot, canonical_terminal = (
        canonical_databricks_submit_payload_snapshot(terminal_run)
    )
    if (
        cloud.get("control_plane_status_sha256")
        != sha256(canonical_terminal).hexdigest()
    ):
        raise ValueError("terminal runs/get response differs from the attestation")
    if _databricks_cloud_id(
        terminal_snapshot.get("run_id"),
        field_name="terminal run_id",
    ) != cloud.get("parent_run_id"):
        raise ValueError("terminal runs/get response belongs to another run")
    updated = record_databricks_verified_run_terminal_actual_json(
        ledger_path,
        attempt_id=attempt_id,
        run_record=terminal_snapshot,
    )
    actual = next(
        item for item in updated.terminal_actuals if item.attempt_id == attempt_id
    )
    if (
        actual.terminal_state != cloud.get("terminal_state")
        or actual.actual_cluster_duration_seconds
        != cloud.get("actual_gpu_duration_seconds")
        or actual.run_id != cloud.get("parent_run_id")
        or actual.submit_payload_sha256 != attempt.get("submit_payload_sha256")
        or actual.control_plane_status_sha256
        != cloud.get("control_plane_status_sha256")
    ):
        raise ValueError("verified terminal ledger event differs from the attestation")
    return updated


def publication_latency_handoff_terminal_actual_gpu_seconds_from_ledger(
    ledger_path: str | Path,
    *,
    attempt_ids_by_worker: Mapping[int, str],
    durable_output_root: str | Path,
    attestations_by_worker: Mapping[
        int,
        PublicationLatencyHandoffDatabricksAttestationBinding,
    ],
) -> dict[int, float]:
    """Extract all sixteen successful one-GPU terminal actuals for closure."""

    reconciliation = _publication_latency_handoff_ledger_reconciliation(
        ledger_path,
        attempt_ids_by_worker=attempt_ids_by_worker,
        durable_output_root=durable_output_root,
        attestations_by_worker=attestations_by_worker,
    )
    durations = {
        _required_int(item, "worker_index"): _required_positive_number(
            item,
            "actual_gpu_duration_seconds",
        )
        for item in _mapping_sequence(
            reconciliation.get("attempts"),
            field_name="ledger reconciliation attempts",
        )
    }
    return _validated_terminal_actual_gpu_seconds(
        durations,
        worker_count=PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS,
    )


def _publication_latency_handoff_ledger_reconciliation(
    ledger_path: str | Path,
    *,
    attempt_ids_by_worker: Mapping[int, str],
    durable_output_root: str | Path,
    attestations_by_worker: Mapping[
        int,
        PublicationLatencyHandoffDatabricksAttestationBinding,
    ],
    _ledger: DatabricksClusterHourLedger | None = None,
    _ledger_path_sha256: str | None = None,
    _expected_producer_batch_prefix: DatabricksLedgerPrefix | None = None,
) -> dict[str, Any]:
    expected_workers = set(range(PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS))
    if set(attempt_ids_by_worker) != expected_workers:
        raise ValueError("attempt IDs must cover producer workers 0..15 exactly")
    if set(attestations_by_worker) != expected_workers:
        raise ValueError("cloud attestations must cover producer workers 0..15 exactly")
    ledger = (
        read_databricks_cluster_hour_ledger_json(ledger_path)
        if _ledger is None
        else _ledger
    )
    ledger_path_binding = (
        databricks_ledger_path_sha256(ledger_path)
        if _ledger_path_sha256 is None
        else _require_sha256(
            _ledger_path_sha256,
            field_name="ledger_path_sha256",
        )
    )
    reservations = {item.attempt_id: item for item in ledger.reservations}
    receipts = {item.attempt_id: item for item in ledger.submission_receipts}
    actuals = {item.attempt_id: item for item in ledger.terminal_actuals}
    attempts: list[dict[str, Any]] = []
    parent_run_ids: set[str] = set()
    task_run_ids: set[str] = set()
    cluster_ids: set[str] = set()
    producer_batch_prefixes: set[str] = set()
    root = Path(local_path(str(durable_output_root))).expanduser().resolve()
    for worker_index in range(PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS):
        attempt_id = attempt_ids_by_worker[worker_index]
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("attempt IDs must be non-empty strings")
        reservation = reservations.get(attempt_id)
        receipt = receipts.get(attempt_id)
        actual = actuals.get(attempt_id)
        if reservation is None or receipt is None or actual is None:
            raise ValueError("producer ledger reconciliation is incomplete")
        if (
            reservation.workload_id
            != (f"publication-latency-handoff-worker-{worker_index:02d}")
            or actual.terminal_state != "succeeded"
        ):
            raise ValueError("producer ledger reconciliation identity is invalid")
        binding = attestations_by_worker[worker_index]
        if binding.worker_index != worker_index:
            raise ValueError("cloud attestation mapping has the wrong worker index")
        attestation = read_publication_latency_handoff_databricks_attestation(
            binding,
            durable_output_root=root,
        )
        attested_attempt = _required_mapping(attestation, "attempt")
        attested_batch_prefix = databricks_ledger_prefix_from_record(
            _required_mapping(attested_attempt, "producer_batch_prefix")
        )
        require_databricks_ledger_prefix(ledger, attested_batch_prefix)
        producer_batch_prefixes.add(attested_batch_prefix.prefix_sha256)
        cloud = _required_mapping(attestation, "cloud_execution")
        worker_result = _required_mapping(attestation, "worker_result")
        if (
            attested_attempt.get("attempt_id") != attempt_id
            or attested_attempt.get("ledger_id") != ledger.ledger_id
            or attested_attempt.get("ledger_path_sha256") != ledger_path_binding
            or attested_attempt.get("workload_id") != reservation.workload_id
            or attested_attempt.get("submit_payload_sha256")
            != reservation.submit_payload_sha256
            or attested_attempt.get("submit_response_sha256")
            != receipt.submit_response_sha256
            or receipt.run_id != cloud.get("parent_run_id")
            or actual.actual_cluster_duration_seconds
            != cloud.get("actual_gpu_duration_seconds")
            or actual.verification_source != "direct_databricks_runs_get"
            or actual.run_id != receipt.run_id
            or actual.submit_payload_sha256 != reservation.submit_payload_sha256
            or actual.control_plane_status_sha256
            != cloud.get("control_plane_status_sha256")
        ):
            raise ValueError("cloud attestation differs from immutable ledger events")
        parent_run_id = _required_string(cloud, "parent_run_id")
        task_run_id = _required_string(cloud, "task_run_id")
        cluster_id = _required_string(cloud, "cluster_id")
        parent_run_ids.add(parent_run_id)
        task_run_ids.add(task_run_id)
        cluster_ids.add(cluster_id)
        attempts.append(
            {
                "actual_gpu_duration_seconds": (actual.actual_cluster_duration_seconds),
                "attestation_closed_record_sha256": (binding.closed_record_sha256),
                "attestation_file_sha256": binding.file_sha256,
                "attestation_relative_path": binding.path.resolve()
                .relative_to(root)
                .as_posix(),
                "attempt_id": attempt_id,
                "attempt_number": cloud["attempt_number"],
                "cluster_id": cluster_id,
                "control_plane_status_sha256": actual.control_plane_status_sha256,
                "parent_run_id": parent_run_id,
                "repair_count": cloud["repair_count"],
                "reserved_gpu_hours": reservation.reserved_cluster_hours,
                "submit_response_sha256": receipt.submit_response_sha256,
                "submit_payload_sha256": reservation.submit_payload_sha256,
                "task_run_id": task_run_id,
                "terminal_state": actual.terminal_state,
                "verification_source": actual.verification_source,
                "worker_index": worker_index,
                "worker_result_closed_record_sha256": worker_result[
                    "closed_record_sha256"
                ],
                "worker_result_file_sha256": worker_result["file_sha256"],
                "workload_id": reservation.workload_id,
            }
        )
    expected_count = PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS
    if not (
        len(parent_run_ids) == expected_count
        and len(task_run_ids) == expected_count
        and len(cluster_ids) == expected_count
    ) or parent_run_ids.intersection(task_run_ids):
        raise ValueError(
            "production workers require unique parent-run, task-run, and cluster IDs"
        )
    if len(producer_batch_prefixes) != 1:
        raise ValueError("producer attestations bind different atomic batches")
    if _expected_producer_batch_prefix is not None:
        if not isinstance(_expected_producer_batch_prefix, DatabricksLedgerPrefix):
            raise TypeError("expected producer batch prefix has the wrong type")
        require_databricks_ledger_prefix(ledger, _expected_producer_batch_prefix)
        if producer_batch_prefixes != {_expected_producer_batch_prefix.prefix_sha256}:
            raise ValueError("producer attestations bind the wrong atomic batch")
    record = {
        "attempts": attempts,
        "attempts_sha256": _canonical_sha256(attempts),
        "cap_gpu_hours": ledger.cap_cluster_hours,
        "cloud_identity_closure_sha256": _canonical_sha256(
            {
                "cluster_ids": sorted(cluster_ids),
                "parent_run_ids": sorted(parent_run_ids),
                "task_run_ids": sorted(task_run_ids),
            }
        ),
        "ledger_id": ledger.ledger_id,
    }
    return record


def run_publication_latency_handoff_worker(
    worker_payload_json: str | Path,
    *,
    expected_worker_payload_sha256: str,
    tokenizer: MainLatencyTokenizer | None = None,
    worker_factory: PublicationLatencyGeneratorFactory | None = None,
    clock: Callable[[], float] = time.perf_counter,
    local_work_root_override: str | Path | None = None,
    hardware_probe: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one plan shard with one persistent generator on one GPU."""

    _require_sha256(
        expected_worker_payload_sha256,
        field_name="expected_worker_payload_sha256",
    )
    payload_path = Path(local_path(str(worker_payload_json))).expanduser().resolve()
    payload_bytes = payload_path.read_bytes()
    if not hmac.compare_digest(
        sha256(payload_bytes).hexdigest(),
        expected_worker_payload_sha256,
    ):
        raise ValueError("worker payload file SHA-256 does not match")
    payload = _json_object(payload_bytes, field_name="worker payload")
    if payload_bytes != _canonical_json_bytes(payload, pretty=True):
        raise ValueError("worker payload is not canonical JSON")
    plan_binding = _required_mapping(payload, "plan")
    plan = _read_bound_closed_json(
        _required_string(plan_binding, "uri"),
        file_sha256=_required_string(plan_binding, "file_sha256"),
        closed_record_sha256=_required_string(
            plan_binding,
            "closed_record_sha256",
        ),
        field_name="generation plan",
    )
    validate_publication_latency_handoff_worker_payload(payload, plan=plan)
    prepared_binding = _required_mapping(payload, "prepared_inputs")
    prepared_root = (
        Path(local_path(_required_string(prepared_binding, "uri")))
        .expanduser()
        .resolve()
    )
    provenance_path = prepared_root / MAIN_LATENCY_PROVENANCE_FILENAME
    _verify_file_sha256(
        provenance_path,
        _required_string(prepared_binding, "provenance_file_sha256"),
        field_name="prepared input provenance",
    )
    provenance = _json_object(
        provenance_path.read_bytes(),
        field_name="prepared input provenance",
    )
    if provenance.get("closed_record_sha256") != _required_string(
        prepared_binding,
        "provenance_closed_record_sha256",
    ):
        raise ValueError("prepared input provenance closure drift")
    qualification = _required_mapping(
        payload,
        "generator_hardware_qualification",
    )
    _verify_bound_hardware_qualification_file(qualification)
    resolved_tokenizer = tokenizer or load_main_latency_tokenizer()
    validate_publication_latency_handoff_generation_plan(
        plan,
        prepared_input_dir=prepared_root,
        tokenizer=resolved_tokenizer,
    )
    prepared = verify_main_latency_inputs(
        prepared_root,
        tokenizer=resolved_tokenizer,
        examples_per_dataset=PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    )
    if prepared.bundle_sha256 != payload.get("input_bundle_sha256"):
        raise ValueError("worker prepared input bundle drift")
    worker_index = _required_int(payload, "worker_index")
    workers = _mapping_sequence(plan.get("workers"), field_name="workers")
    worker = workers[worker_index]
    output_root = (
        Path(local_path(_required_string(payload, "durable_output_root")))
        .expanduser()
        .absolute()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _require_worker_output_is_fresh(payload, output_root=output_root)
    work_root = (
        Path(local_work_root_override).expanduser().absolute()
        if local_work_root_override is not None
        else Path(_required_string(payload, "local_work_root")).expanduser().absolute()
    )
    _require_fresh_output_path(work_root)
    work_root.mkdir(parents=True)
    config = _execution_config_from_record(
        _required_mapping(payload, "execution_contract")
    )
    _apply_production_generator_environment(config)
    if worker_factory is None:
        _verify_installed_vllm_bitsandbytes_loader_source(config)
    observed_hardware = dict((hardware_probe or _probe_single_l40s_hardware)())
    _validate_observed_l40s_hardware(observed_hardware)
    factory = worker_factory or _production_generator_factory
    rows_by_task = _prepared_rows_by_task(prepared)
    producer_start = clock()
    try:
        worker_result = _execute_worker(
            worker,
            rows_by_task=rows_by_task,
            work_root=work_root,
            pending_root=output_root / "pending",
            worker_factory=factory,
            config=config,
            clock=clock,
            enforce_production_generator_identity=(worker_factory is None),
        )
        partial_root_relative = _required_string(
            _required_mapping(payload, "output_binding"),
            "partial_records_relative_root",
        )
        partial_root = _confined_relative_path(
            output_root,
            partial_root_relative,
            field_name="partial_records_relative_root",
        )
        partial_records: list[dict[str, Any]] = []
        for batch in worker_result.batches:
            path = partial_root / f"{batch.context_tokens}.jsonl"
            _write_canonical_jsonl_exclusive(batch.records, path)
            _sync_file(path)
            partial_records.append(
                {
                    "byte_count": path.stat().st_size,
                    "context_tokens": batch.context_tokens,
                    "record_count": len(batch.records),
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "sha256": _file_sha256(path),
                }
            )
        _sync_directory(partial_root)
        _sync_worker_data_ancestor_directories(output_root)
        bundle_files = _worker_bundle_file_records(
            output_root,
            worker_result=worker_result,
        )
        producer_end = clock()
        producer_seconds = _positive_duration(producer_end - producer_start)
        task_ids = sorted(
            task_id for batch in worker_result.batches for task_id in batch.task_ids
        )
        result: dict[str, Any] = {
            "accounting": {
                "cache_prefix_tokens": sum(
                    batch.cache_prefix_tokens for batch in worker_result.batches
                ),
                "durable_byte_count": sum(
                    cast(int, item["byte_count"]) for item in bundle_files
                )
                + sum(cast(int, item["byte_count"]) for item in partial_records),
                "generation_seconds": sum(
                    batch.generation_seconds for batch in worker_result.batches
                ),
                "includes_generation_payload_hash_and_durable_sync": True,
                "input_token_slots": sum(
                    batch.input_token_slots for batch in worker_result.batches
                ),
                "producer_metered_seconds": producer_seconds,
            },
            "bundle_files": bundle_files,
            "bundle_files_sha256": _canonical_sha256(bundle_files),
            "closed_record_sha256": "",
            "execution_contract": _execution_config_record(config),
            "generator_hardware": {
                "observed": observed_hardware,
                "qualification": dict(qualification),
            },
            "input_bundle_sha256": prepared.bundle_sha256,
            "partial_record_files": partial_records,
            "plan_closed_record_sha256": plan["closed_record_sha256"],
            "record_type": PUBLICATION_LATENCY_HANDOFF_WORKER_RESULT_RECORD_TYPE,
            "schema_version": PUBLICATION_LATENCY_HANDOFF_WORKER_RESULT_SCHEMA_VERSION,
            "task_ids": task_ids,
            "task_ids_sha256": _canonical_sha256(task_ids),
            "worker_id": payload["worker_id"],
            "worker_index": worker_index,
        }
        result["closed_record_sha256"] = _closed_record_sha256(result)
        result_relative = _required_string(
            _required_mapping(payload, "output_binding"),
            "result_relative_path",
        )
        result_path = _confined_relative_path(
            output_root,
            result_relative,
            field_name="result_relative_path",
        )
        _write_canonical_json_exclusive(result, result_path)
        _sync_file(result_path)
        _sync_directory(result_path.parent)
        _sync_directory(output_root)
        return result
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        gc.collect()
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            pass
        else:
            torch.cuda.empty_cache()


def close_publication_latency_handoff_generation_from_workers(
    record: Mapping[str, Any],
    *,
    prepared_input_dir: str | Path,
    durable_output_root: str | Path,
    tokenizer: MainLatencyTokenizer,
    config: PublicationLatencyHandoffExecutionConfig,
    ledger_path: str | Path,
    attempt_ids_by_worker: Mapping[int, str],
    attestations_by_worker: Mapping[
        int,
        PublicationLatencyHandoffDatabricksAttestationBinding,
    ],
    source_paths: Mapping[str, str | Path] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    _ledger_snapshot: DatabricksClusterHourLedger | None = None,
    _ledger_path_sha256: str | None = None,
    _expected_producer_batch_prefix: DatabricksLedgerPrefix | None = None,
    _remote_ledger_issuer: object | None = None,
) -> PublicationLatencyHandoffGenerationResult:
    """Verify sixteen worker closures and close three bundles without copying KV."""

    if (
        _ledger_snapshot is not None
        or _ledger_path_sha256 is not None
        or _expected_producer_batch_prefix is not None
        or _remote_ledger_issuer is not None
    ):
        if _remote_ledger_issuer is not _REMOTE_CLOSURE_LEDGER_ISSUER:
            raise TypeError("remote ledger snapshot requires the coordinator issuer")
        if (
            not isinstance(_ledger_snapshot, DatabricksClusterHourLedger)
            or (_ledger_path_sha256 is None)
            or not isinstance(_expected_producer_batch_prefix, DatabricksLedgerPrefix)
        ):
            raise ValueError(
                "remote ledger snapshot, path digest, and batch prefix are required"
            )

    validate_publication_latency_handoff_generation_plan(
        record,
        prepared_input_dir=prepared_input_dir,
        tokenizer=tokenizer,
        source_paths=source_paths,
    )
    workers = _mapping_sequence(record.get("workers"), field_name="workers")
    if len(workers) != PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS:
        raise ValueError("coordinator requires all sixteen production workers")
    root = Path(local_path(str(durable_output_root))).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("durable_output_root must be a real shared directory")
    for reserved in (
        "bundles",
        "manifests",
        PUBLICATION_LATENCY_HANDOFF_EXECUTION_FILENAME,
    ):
        target = root / reserved
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"coordinator output is not fresh: {target}")
    actuals: dict[int, float]
    if _ledger_snapshot is None:
        actuals = publication_latency_handoff_terminal_actual_gpu_seconds_from_ledger(
            ledger_path,
            attempt_ids_by_worker=attempt_ids_by_worker,
            durable_output_root=root,
            attestations_by_worker=attestations_by_worker,
        )
    ledger_reconciliation = _publication_latency_handoff_ledger_reconciliation(
        ledger_path,
        attempt_ids_by_worker=attempt_ids_by_worker,
        durable_output_root=root,
        attestations_by_worker=attestations_by_worker,
        _ledger=_ledger_snapshot,
        _ledger_path_sha256=_ledger_path_sha256,
        _expected_producer_batch_prefix=_expected_producer_batch_prefix,
    )
    if _ledger_snapshot is not None:
        actuals = {
            _required_int(item, "worker_index"): _required_positive_number(
                item,
                "actual_gpu_duration_seconds",
            )
            for item in _mapping_sequence(
                ledger_reconciliation.get("attempts"),
                field_name="ledger reconciliation attempts",
            )
        }
        actuals = _validated_terminal_actual_gpu_seconds(
            actuals,
            worker_count=PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS,
        )
    coordinator_start = clock()
    worker_results: list[dict[str, Any]] = []
    all_batches: list[_WorkerBatchResult] = []
    all_task_ids: list[str] = []
    qualification_closures: set[str] = set()
    observed_hardware_sha256: set[str] = set()
    for worker_index, worker in enumerate(workers):
        result_path = root / "worker-results" / f"worker-{worker_index:02d}.json"
        result = _read_worker_result(result_path)
        batches = _validate_worker_result_and_load_batches(
            result,
            result_path=result_path,
            output_root=root,
            worker=worker,
            plan=record,
            config=config,
        )
        metered = _required_positive_number(
            _required_mapping(result, "accounting"),
            "producer_metered_seconds",
        )
        if actuals[worker_index] + 1e-12 < metered:
            raise ValueError("terminal GPU actual is shorter than worker metering")
        hardware = _required_mapping(result, "generator_hardware")
        qualification = _required_mapping(hardware, "qualification")
        qualification_closures.add(
            _required_string(qualification, "evidence_closed_record_sha256")
        )
        observed_hardware_sha256.add(
            _canonical_sha256(_required_mapping(hardware, "observed"))
        )
        worker_results.append(result)
        all_batches.extend(batches)
        all_task_ids.extend(cast(list[str], result["task_ids"]))
    _validate_shared_worker_output_tree(
        root,
        plan=record,
        worker_results=worker_results,
    )
    if len(qualification_closures) != 1:
        raise ValueError("workers used different hardware qualification records")
    expected_task_ids = _plan_task_ids(record)
    if Counter(all_task_ids) != Counter(expected_task_ids):
        raise ValueError("worker result task coverage is incomplete")
    if len(set(all_task_ids)) != len(all_task_ids):
        raise ValueError("worker result task coverage contains duplicates")

    bundles_root = root / "bundles"
    manifests_root = root / "manifests"
    bundles_root.mkdir()
    manifests_root.mkdir()
    bundle_records: list[dict[str, Any]] = []
    for context_tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
        pending_context = root / "pending" / str(context_tokens)
        dataset_paths = _write_context_dataset_files(
            pending_context,
            context_tokens=context_tokens,
            batches=all_batches,
        )
        _sync_tree(pending_context)
        manifest = close_publication_latency_handoff_bundle(
            pending_context,
            dataset_paths,
            context_tokens=context_tokens,
            input_bundle_sha256=_required_string(record, "input_bundle_sha256"),
        )
        _validate_publication_manifest_contract(manifest)
        portable_digest = _required_string(manifest, "portable_bundle_sha256")
        content_root = bundles_root / f"{context_tokens}-{portable_digest}"
        if content_root.exists() or content_root.is_symlink():
            raise ValueError("content-addressed context bundle collision")
        os.rename(pending_context, content_root)
        validate_publication_latency_handoff_bundle(
            manifest,
            bundle_root=content_root,
        )
        manifest_relative = (
            PurePosixPath("manifests") / f"{context_tokens}-{portable_digest}.json"
        )
        manifest_path = root / manifest_relative
        write_publication_latency_handoff_bundle(manifest, manifest_path)
        _sync_file(manifest_path)
        bundle_records.append(
            {
                "closed_record_sha256": manifest["closed_record_sha256"],
                "context_tokens": context_tokens,
                "manifest_relative_path": str(manifest_relative),
                "portable_bundle_sha256": portable_digest,
                "source_root_relative_path": str(content_root.relative_to(root)),
            }
        )
    pending = root / "pending"
    if any(pending.iterdir()):
        raise ValueError("pending generation tree contains unclosed output")
    pending.rmdir()
    _sync_tree(bundles_root)
    _sync_tree(manifests_root)
    coordinator_seconds = _positive_duration(clock() - coordinator_start)
    charged_gpu_seconds = sum(actuals.values())
    cache_prefix_tokens = _coverage_int(record, "cache_prefix_generation_tokens")
    input_token_slots = _coverage_int(record, "input_token_slots")
    tokens_per_gpu_second = cache_prefix_tokens / charged_gpu_seconds
    input_slots_per_gpu_second = input_token_slots / charged_gpu_seconds
    report_workers = [
        {
            "charged_gpu_seconds": actuals[index],
            "producer_metered_seconds": _required_mapping(result, "accounting")[
                "producer_metered_seconds"
            ],
            "result_closed_record_sha256": result["closed_record_sha256"],
            "result_relative_path": (f"worker-results/worker-{index:02d}.json"),
            "worker_index": index,
        }
        for index, result in enumerate(worker_results)
    ]
    total_durable_bytes = sum(
        _tree_byte_count(root / name)
        for name in (
            "bundles",
            "manifests",
            "worker-records",
            "worker-results",
            PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY,
        )
    )
    report: dict[str, Any] = {
        "accounting": {
            "charged_gpu_hours": charged_gpu_seconds / 3600.0,
            "charged_gpu_seconds": charged_gpu_seconds,
            "coordinator_gpu_hours": 0.0,
            "coordinator_wall_seconds": coordinator_seconds,
            "cost_model": "sum_independent_one_gpu_worker_terminal_lifecycles",
            "durable_byte_count": total_durable_bytes,
            "end_to_end_cache_prefix_tokens_per_gpu_second": tokens_per_gpu_second,
            "end_to_end_input_token_slots_per_gpu_second": input_slots_per_gpu_second,
            "end_to_end_wall_seconds": max(actuals.values()),
            "full_launch_min_tokens_per_gpu_second": (
                PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
            ),
            "full_launch_throughput_gate_passed": (
                tokens_per_gpu_second
                >= PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
            ),
            "includes_bootstrap_generation_hash_and_durable_write": True,
            "payload_copy_count_during_closure": 0,
            "worker_count": len(workers),
        },
        "bundles": bundle_records,
        "bundles_sha256": _canonical_sha256(bundle_records),
        "closed_record_sha256": "",
        "coverage": {
            "cache_prefix_generation_tokens": cache_prefix_tokens,
            "context_bundle_count": len(bundle_records),
            "generated_task_count": len(all_task_ids),
            "input_token_slots": input_token_slots,
            "no_duplicate_tasks": True,
            "task_ids_sha256": _canonical_sha256(sorted(all_task_ids)),
        },
        "execution_contract": _execution_config_record(config),
        "execution_mode": PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED,
        "generator_hardware": {
            "hardware_target": PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET,
            "node_type_id": PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID,
            "observed_hardware_identity_sha256": sorted(observed_hardware_sha256),
            "qualification_closed_record_sha256": next(iter(qualification_closures)),
        },
        "input_bundle_sha256": record["input_bundle_sha256"],
        "ledger_reconciliation": ledger_reconciliation,
        "plan_closed_record_sha256": record["closed_record_sha256"],
        "record_type": PUBLICATION_LATENCY_HANDOFF_EXECUTION_RECORD_TYPE,
        "schema_version": PUBLICATION_LATENCY_HANDOFF_EXECUTION_SCHEMA_VERSION,
        "serving_reuse": {
            "context_bundles": bundle_records,
            "regenerate_inside_timed_serving_jobs": False,
            "required_action": "validate_manifest_then_stage_content_addressed_bundle",
            "stage_target": "node_local_nvme",
        },
        "workers": report_workers,
    }
    report["closed_record_sha256"] = _closed_record_sha256(report)
    execution_path = root / PUBLICATION_LATENCY_HANDOFF_EXECUTION_FILENAME
    _write_canonical_json_exclusive(report, execution_path)
    _sync_file(execution_path)
    _sync_directory(root)
    return read_publication_latency_handoff_generation_result(root)


def _replay_closed_publication_latency_handoff_generation(
    plan: Mapping[str, Any],
    *,
    prepared_input_dir: str | Path,
    durable_output_root: str | Path,
    tokenizer: MainLatencyTokenizer,
    config: PublicationLatencyHandoffExecutionConfig,
    ledger_snapshot: DatabricksClusterHourLedger,
    ledger_path_sha256: str,
    expected_producer_batch_prefix: DatabricksLedgerPrefix,
    attempt_ids_by_worker: Mapping[int, str],
    attestations_by_worker: Mapping[
        int, PublicationLatencyHandoffDatabricksAttestationBinding
    ],
    source_paths: Mapping[str, str | Path] | None = None,
    _issuer: object | None = None,
) -> PublicationLatencyHandoffGenerationResult:
    """Reopen every raw Q8 producer byte after the pending tree was renamed."""

    if _issuer is not _POST_CLOSE_REPLAY_ISSUER:
        raise TypeError("post-close Q8 replay requires the coordinator issuer")
    if not isinstance(config, PublicationLatencyHandoffExecutionConfig):
        raise TypeError("config must be a PublicationLatencyHandoffExecutionConfig")
    if not isinstance(ledger_snapshot, DatabricksClusterHourLedger):
        raise TypeError("ledger_snapshot must be a DatabricksClusterHourLedger")
    if not isinstance(expected_producer_batch_prefix, DatabricksLedgerPrefix):
        raise TypeError("expected_producer_batch_prefix has the wrong type")
    path_digest = _require_sha256(
        ledger_path_sha256,
        field_name="ledger_path_sha256",
    )
    validate_publication_latency_handoff_generation_plan(
        plan,
        prepared_input_dir=prepared_input_dir,
        tokenizer=tokenizer,
        source_paths=source_paths,
    )
    root = Path(local_path(str(durable_output_root))).expanduser().resolve()
    result = read_publication_latency_handoff_generation_result(root)
    execution = result.record
    _require_exact_mapping_keys(
        execution,
        {
            "accounting",
            "bundles",
            "bundles_sha256",
            "closed_record_sha256",
            "coverage",
            "execution_contract",
            "execution_mode",
            "generator_hardware",
            "input_bundle_sha256",
            "ledger_reconciliation",
            "plan_closed_record_sha256",
            "record_type",
            "schema_version",
            "serving_reuse",
            "workers",
        },
        label="post-close Q8 execution",
    )
    if (
        execution.get("execution_contract") != _execution_config_record(config)
        or execution.get("plan_closed_record_sha256")
        != plan.get("closed_record_sha256")
        or execution.get("input_bundle_sha256") != plan.get("input_bundle_sha256")
    ):
        raise ValueError("post-close Q8 plan/input/config binding drift")

    reconciliation = _publication_latency_handoff_ledger_reconciliation(
        Path("/local_disk0/cachet-post-close-ledger-snapshot.json"),
        attempt_ids_by_worker=attempt_ids_by_worker,
        durable_output_root=root,
        attestations_by_worker=attestations_by_worker,
        _ledger=ledger_snapshot,
        _ledger_path_sha256=path_digest,
        _expected_producer_batch_prefix=expected_producer_batch_prefix,
    )
    if dict(_required_mapping(execution, "ledger_reconciliation")) != reconciliation:
        raise ValueError("post-close Q8 ledger reconciliation drift")
    attempts = _mapping_sequence(
        reconciliation.get("attempts"),
        field_name="post-close Q8 reconciliation attempts",
    )
    attempt_by_worker = {
        _required_int(item, "worker_index"): item for item in attempts
    }
    if set(attempt_by_worker) != set(
        range(PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS)
    ):
        raise ValueError("post-close Q8 ledger worker coverage drift")

    bundle_records = _mapping_sequence(execution.get("bundles"), field_name="bundles")
    bundle_by_context = {
        _required_int(item, "context_tokens"): item for item in bundle_records
    }
    if set(bundle_by_context) != set(PUBLICATION_CAMPAIGN_CONTEXT_TOKENS):
        raise ValueError("post-close Q8 bundle context coverage drift")
    manifest_paths: set[Path] = set()
    expected_bundle_paths: set[Path] = set()
    source_roots: dict[int, Path] = {}
    manifest_files: dict[int, dict[str, Mapping[str, Any]]] = {}
    manifest_entries: dict[int, dict[tuple[str, str], Mapping[str, Any]]] = {}
    final_rows: dict[tuple[int, str], tuple[dict[str, Any], ...]] = {}
    for context_tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
        bundle = bundle_by_context[context_tokens]
        manifest_path = _confined_relative_path(
            root,
            _required_string(bundle, "manifest_relative_path"),
            field_name="post-close Q8 manifest",
        )
        source_root = _confined_relative_path(
            root,
            _required_string(bundle, "source_root_relative_path"),
            field_name="post-close Q8 source root",
        )
        manifest = read_publication_latency_handoff_bundle(manifest_path)
        manifest_paths.add(manifest_path)
        source_roots[context_tokens] = source_root
        files = {
            _required_string(item, "relative_name"): item
            for item in _mapping_sequence(manifest.get("files"), field_name="files")
        }
        if len(files) != len(
            _mapping_sequence(manifest.get("files"), field_name="files")
        ):
            raise ValueError("post-close Q8 manifest contains duplicate file names")
        manifest_files[context_tokens] = files
        expected_bundle_paths.update(
            source_root / PurePosixPath(relative_name) for relative_name in files
        )
        entries: dict[tuple[str, str], Mapping[str, Any]] = {}
        for dataset_record in _mapping_sequence(
            manifest.get("datasets"), field_name="datasets"
        ):
            dataset = _required_string(dataset_record, "dataset")
            dataset_path = source_root / PurePosixPath(
                _required_string(dataset_record, "relative_name")
            )
            final_rows[(context_tokens, dataset)] = _post_close_jsonl_objects(
                dataset_path
            )
            for entry in _mapping_sequence(
                dataset_record.get("entries"), field_name="dataset.entries"
            ):
                identity = (dataset, _required_string(entry, "example_id"))
                if identity in entries:
                    raise ValueError("post-close Q8 manifest identity duplication")
                entries[identity] = entry
        manifest_entries[context_tokens] = entries

    workers = _mapping_sequence(plan.get("workers"), field_name="workers")
    if len(workers) != PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS:
        raise ValueError("post-close Q8 plan requires sixteen workers")
    expected_result_paths: set[Path] = set()
    expected_record_paths: set[Path] = set()
    expected_attestation_paths = {
        binding.path.resolve() for binding in attestations_by_worker.values()
    }
    observed_bundle_records: dict[int, dict[str, Mapping[str, Any]]] = {
        context: {} for context in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS
    }
    rows_by_context: dict[int, list[dict[str, Any]]] = {
        context: [] for context in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS
    }
    worker_results: list[dict[str, Any]] = []
    all_task_ids: list[str] = []
    qualification_records: list[Mapping[str, Any]] = []
    qualification_closures: set[str] = set()
    observed_hardware_sha256: set[str] = set()
    for worker_index, worker in enumerate(workers):
        if _required_int(worker, "worker_index") != worker_index:
            raise ValueError("post-close Q8 plan worker order drift")
        result_path = root / "worker-results" / f"worker-{worker_index:02d}.json"
        expected_result_paths.add(result_path)
        worker_result = _read_worker_result(result_path)
        _require_exact_mapping_keys(
            worker_result,
            {
                "accounting",
                "bundle_files",
                "bundle_files_sha256",
                "closed_record_sha256",
                "execution_contract",
                "generator_hardware",
                "input_bundle_sha256",
                "partial_record_files",
                "plan_closed_record_sha256",
                "record_type",
                "schema_version",
                "task_ids",
                "task_ids_sha256",
                "worker_id",
                "worker_index",
            },
            label="post-close Q8 worker result",
        )
        if (
            worker_result.get("worker_index") != worker_index
            or worker_result.get("worker_id") != worker.get("worker_id")
            or worker_result.get("plan_closed_record_sha256")
            != plan.get("closed_record_sha256")
            or worker_result.get("input_bundle_sha256")
            != plan.get("input_bundle_sha256")
            or worker_result.get("execution_contract")
            != _execution_config_record(config)
        ):
            raise ValueError("post-close Q8 worker source/config binding drift")
        planned_items = _mapping_sequence(worker.get("items"), field_name="worker.items")
        expected_task_ids = sorted(
            _required_string(item, "task_id") for item in planned_items
        )
        if worker_result.get("task_ids") != expected_task_ids or worker_result.get(
            "task_ids_sha256"
        ) != _canonical_sha256(expected_task_ids):
            raise ValueError("post-close Q8 worker task closure drift")
        all_task_ids.extend(expected_task_ids)

        bundle_files = _mapping_sequence(
            worker_result.get("bundle_files"), field_name="bundle_files"
        )
        if worker_result.get("bundle_files_sha256") != _canonical_sha256(bundle_files):
            raise ValueError("post-close Q8 worker bundle closure drift")
        worker_bundle_bytes = 0
        for file_record in bundle_files:
            _require_exact_mapping_keys(
                file_record,
                {"byte_count", "context_tokens", "relative_name", "sha256"},
                label="post-close Q8 worker bundle file",
            )
            context_tokens = _required_int(file_record, "context_tokens")
            relative_name = _required_string(file_record, "relative_name")
            expected_prefix = f"worker-{worker_index:02d}/"
            if (
                context_tokens not in source_roots
                or not relative_name.startswith(expected_prefix)
                or relative_name in observed_bundle_records[context_tokens]
            ):
                raise ValueError("post-close Q8 worker bundle path drift")
            manifest_file = manifest_files[context_tokens].get(relative_name)
            expected_file_record = (
                {
                    "byte_count": manifest_file.get("byte_count"),
                    "context_tokens": context_tokens,
                    "relative_name": relative_name,
                    "sha256": manifest_file.get("sha256"),
                }
                if manifest_file is not None
                and manifest_file.get("role") in {"handoff_json", "payload"}
                else None
            )
            if expected_file_record != dict(file_record):
                raise ValueError(
                    "post-close Q8 worker file differs from finalized manifest"
                )
            observed_bundle_records[context_tokens][relative_name] = file_record
            worker_bundle_bytes += _required_int(file_record, "byte_count")

        partial_files = _mapping_sequence(
            worker_result.get("partial_record_files"),
            field_name="partial_record_files",
        )
        expected_contexts = {
            _required_int(item, "context_tokens") for item in planned_items
        }
        seen_contexts: set[int] = set()
        worker_record_bytes = 0
        for file_record in partial_files:
            _require_exact_mapping_keys(
                file_record,
                {
                    "byte_count",
                    "context_tokens",
                    "record_count",
                    "relative_path",
                    "sha256",
                },
                label="post-close Q8 worker record file",
            )
            context_tokens = _required_int(file_record, "context_tokens")
            expected_relative = (
                f"worker-records/worker-{worker_index:02d}/{context_tokens}.jsonl"
            )
            if (
                context_tokens not in expected_contexts
                or context_tokens in seen_contexts
                or file_record.get("relative_path") != expected_relative
            ):
                raise ValueError("post-close Q8 worker-record path/coverage drift")
            record_path = _confined_relative_path(
                root,
                expected_relative,
                field_name="post-close Q8 worker record",
            )
            expected_record_paths.add(record_path)
            _verify_file_record(record_path, file_record)
            rows = _canonical_jsonl_records(record_path)
            items = tuple(
                item
                for item in planned_items
                if item.get("context_tokens") == context_tokens
            )
            expected_identities = {
                (_required_string(item, "dataset"), _required_string(item, "example_id"))
                for item in items
            }
            observed_identities = {
                (_required_string(row, "dataset"), _required_string(row, "example_id"))
                for row in rows
            }
            if (
                len(rows) != len(items)
                or file_record.get("record_count") != len(items)
                or observed_identities != expected_identities
            ):
                raise ValueError("post-close Q8 worker-record identity drift")
            plan_by_identity = {
                (_required_string(item, "dataset"), _required_string(item, "example_id")): item
                for item in items
            }
            for row in rows:
                identity = (
                    _required_string(row, "dataset"),
                    _required_string(row, "example_id"),
                )
                manifest_entry = manifest_entries[context_tokens].get(identity)
                if manifest_entry is None:
                    raise ValueError("post-close Q8 row is absent from the manifest")
                rebased = _post_close_rebased_generated_row(
                    row,
                    source_root=source_roots[context_tokens],
                    manifest_entry=manifest_entry,
                    arm_id=PUBLICATION_LATENCY_HANDOFF_ARM_ID,
                )
                planned = plan_by_identity[identity]
                expected_contracts = [
                    dict(item)
                    for item in _mapping_sequence(
                        planned.get("segment_token_contracts"),
                        field_name="segment_token_contracts",
                    )
                ]
                if (
                    _handoff_total_tokens(rebased)
                    != _required_int(planned, "cache_prefix_tokens")
                    or _handoff_segment_token_contracts(rebased)
                    != expected_contracts
                ):
                    raise ValueError("post-close Q8 handoff differs from the exact plan")
                rows_by_context[context_tokens].append(dict(row))
            seen_contexts.add(context_tokens)
            worker_record_bytes += _required_int(file_record, "byte_count")
        if seen_contexts != expected_contexts:
            raise ValueError("post-close Q8 worker-record context coverage drift")

        worker_accounting = _required_mapping(worker_result, "accounting")
        _require_exact_mapping_keys(
            worker_accounting,
            {
                "cache_prefix_tokens",
                "durable_byte_count",
                "generation_seconds",
                "includes_generation_payload_hash_and_durable_sync",
                "input_token_slots",
                "producer_metered_seconds",
            },
            label="post-close Q8 worker accounting",
        )
        _required_positive_number(worker_accounting, "generation_seconds")
        metered = _required_positive_number(
            worker_accounting, "producer_metered_seconds"
        )
        if (
            worker_accounting.get("cache_prefix_tokens")
            != worker.get("cache_prefix_tokens")
            or worker_accounting.get("input_token_slots")
            != worker.get("input_token_slots")
            or worker_accounting.get("durable_byte_count")
            != worker_bundle_bytes + worker_record_bytes
            or worker_accounting.get(
                "includes_generation_payload_hash_and_durable_sync"
            )
            is not True
        ):
            raise ValueError("post-close Q8 worker accounting drift")
        attempt = attempt_by_worker[worker_index]
        if (
            _file_sha256(result_path) != attempt.get("worker_result_file_sha256")
            or worker_result.get("closed_record_sha256")
            != attempt.get("worker_result_closed_record_sha256")
            or _required_positive_number(attempt, "actual_gpu_duration_seconds")
            + 1e-12
            < metered
        ):
            raise ValueError("post-close Q8 worker/ledger evidence drift")
        hardware = _required_mapping(worker_result, "generator_hardware")
        _require_exact_mapping_keys(
            hardware,
            {"observed", "qualification"},
            label="post-close Q8 worker hardware",
        )
        qualification = _required_mapping(hardware, "qualification")
        _validate_hardware_qualification_record(qualification)
        observed = _required_mapping(hardware, "observed")
        _validate_observed_l40s_hardware(observed)
        qualification_records.append(qualification)
        qualification_closures.add(
            _required_string(qualification, "evidence_closed_record_sha256")
        )
        observed_hardware_sha256.add(_canonical_sha256(observed))
        worker_results.append(worker_result)

    for context_tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
        expected_worker_files = {
            relative_name: item
            for relative_name, item in manifest_files[context_tokens].items()
            if item.get("role") in {"handoff_json", "payload"}
        }
        observed = observed_bundle_records[context_tokens]
        if set(observed) != set(expected_worker_files):
            raise ValueError("post-close Q8 worker bundle inventory drift")
        for dataset in SUPPORTED_V1_DATASETS:
            expected_rows = tuple(
                sorted(
                    (
                        row
                        for row in rows_by_context[context_tokens]
                        if row.get("dataset") == dataset
                    ),
                    key=lambda row: _required_string(row, "example_id"),
                )
            )
            if final_rows.get((context_tokens, dataset)) != expected_rows:
                raise ValueError(
                    "post-close Q8 worker records differ from finalized datasets"
                )

    result_paths = _post_close_regular_file_inventory(
        root / "worker-results", label="post-close Q8 worker-results"
    )
    record_paths = _post_close_regular_file_inventory(
        root / "worker-records", label="post-close Q8 worker-records"
    )
    attestation_paths = _post_close_regular_file_inventory(
        root / PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY,
        label="post-close Q8 attestations",
    )
    actual_manifest_paths = _post_close_regular_file_inventory(
        root / "manifests", label="post-close Q8 manifests"
    )
    actual_bundle_paths = _post_close_regular_file_inventory(
        root / "bundles", label="post-close Q8 bundles"
    )
    if (
        result_paths != expected_result_paths
        or record_paths != expected_record_paths
        or attestation_paths != expected_attestation_paths
        or actual_manifest_paths != manifest_paths
        or actual_bundle_paths != expected_bundle_paths
    ):
        raise ValueError("post-close Q8 durable file inventory drift")
    if len(qualification_closures) != 1 or len(
        {_canonical_sha256(item) for item in qualification_records}
    ) != 1:
        raise ValueError("post-close Q8 hardware qualification drift")
    _verify_bound_hardware_qualification_file(qualification_records[0])

    planned_task_ids = _plan_task_ids(plan)
    if Counter(all_task_ids) != Counter(planned_task_ids) or len(
        set(all_task_ids)
    ) != len(all_task_ids):
        raise ValueError("post-close Q8 task coverage drift")
    actuals = {
        index: _required_positive_number(
            attempt_by_worker[index], "actual_gpu_duration_seconds"
        )
        for index in range(PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS)
    }
    expected_workers = [
        {
            "charged_gpu_seconds": actuals[index],
            "producer_metered_seconds": _required_mapping(
                worker_results[index], "accounting"
            )["producer_metered_seconds"],
            "result_closed_record_sha256": worker_results[index][
                "closed_record_sha256"
            ],
            "result_relative_path": f"worker-results/worker-{index:02d}.json",
            "worker_index": index,
        }
        for index in range(PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS)
    ]
    if list(_mapping_sequence(execution.get("workers"), field_name="workers")) != (
        expected_workers
    ):
        raise ValueError("post-close Q8 execution worker closure drift")
    cache_prefix_tokens = _coverage_int(plan, "cache_prefix_generation_tokens")
    input_token_slots = _coverage_int(plan, "input_token_slots")
    expected_coverage: dict[str, Any] = {
        "cache_prefix_generation_tokens": cache_prefix_tokens,
        "context_bundle_count": len(PUBLICATION_CAMPAIGN_CONTEXT_TOKENS),
        "generated_task_count": len(all_task_ids),
        "input_token_slots": input_token_slots,
        "no_duplicate_tasks": True,
        "task_ids_sha256": _canonical_sha256(sorted(all_task_ids)),
    }
    if dict(_required_mapping(execution, "coverage")) != expected_coverage:
        raise ValueError("post-close Q8 execution task closure drift")
    expected_hardware = {
        "hardware_target": PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET,
        "node_type_id": PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID,
        "observed_hardware_identity_sha256": sorted(observed_hardware_sha256),
        "qualification_closed_record_sha256": next(iter(qualification_closures)),
    }
    if dict(_required_mapping(execution, "generator_hardware")) != expected_hardware:
        raise ValueError("post-close Q8 execution hardware closure drift")
    charged_seconds = sum(actuals.values())
    accounting = _required_mapping(execution, "accounting")
    coordinator_seconds = _required_positive_number(
        accounting, "coordinator_wall_seconds"
    )
    durable_paths = (
        actual_bundle_paths
        | actual_manifest_paths
        | record_paths
        | result_paths
        | attestation_paths
    )
    expected_accounting = {
        "charged_gpu_hours": charged_seconds / 3600.0,
        "charged_gpu_seconds": charged_seconds,
        "coordinator_gpu_hours": 0.0,
        "coordinator_wall_seconds": coordinator_seconds,
        "cost_model": "sum_independent_one_gpu_worker_terminal_lifecycles",
        "durable_byte_count": sum(path.stat().st_size for path in durable_paths),
        "end_to_end_cache_prefix_tokens_per_gpu_second": (
            cache_prefix_tokens / charged_seconds
        ),
        "end_to_end_input_token_slots_per_gpu_second": (
            input_token_slots / charged_seconds
        ),
        "end_to_end_wall_seconds": max(actuals.values()),
        "full_launch_min_tokens_per_gpu_second": (
            PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
        ),
        "full_launch_throughput_gate_passed": (
            cache_prefix_tokens / charged_seconds
            >= PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
        ),
        "includes_bootstrap_generation_hash_and_durable_write": True,
        "payload_copy_count_during_closure": 0,
        "worker_count": PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS,
    }
    if dict(accounting) != expected_accounting:
        raise ValueError("post-close Q8 execution accounting closure drift")
    return result


def _post_close_rebased_generated_row(
    row: Mapping[str, Any],
    *,
    source_root: Path,
    manifest_entry: Mapping[str, Any],
    arm_id: str,
) -> dict[str, Any]:
    """Point a pre-close enriched row at the same files beneath its final root."""

    rebased = dict(row)
    arms = dict(_required_mapping(row, "arm_kv_transfer_params"))
    params = dict(_required_mapping(arms, arm_id))
    bindings = {
        DOCUMENT_KV_HANDOFF_JSON_PARAM: _required_string(
            manifest_entry, "handoff_relative_name"
        ),
        DOCUMENT_KV_PAYLOAD_URI_PARAM: _required_string(
            manifest_entry, "payload_relative_name"
        ),
    }
    for field_name, relative_name in bindings.items():
        original = _required_string(params, field_name).replace("\\", "/")
        if not original.endswith(f"/{relative_name}"):
            raise ValueError("pre-close worker path does not rebase to the manifest")
        params[field_name] = str(source_root / PurePosixPath(relative_name))
    arms[arm_id] = params
    rebased["arm_kv_transfer_params"] = arms
    return rebased


def _post_close_regular_file_inventory(root: Path, *, label: str) -> set[Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} must be one real directory")
    files: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} contains a symlink")
        if path.is_file():
            files.add(path)
        elif not path.is_dir():
            raise ValueError(f"{label} contains a non-regular entry")
    return files


def _post_close_jsonl_objects(path: Path) -> tuple[dict[str, Any], ...]:
    content = path.read_bytes()
    if not content or not content.endswith(b"\n"):
        raise ValueError("post-close dataset JSONL must be non-empty and terminated")
    rows: list[dict[str, Any]] = []
    for line in content.splitlines():
        value = _json_object(line, field_name="post-close dataset row")
        rows.append(value)
    return tuple(rows)


def _hardware_qualification_record(
    value: PublicationLatencyGeneratorHardwareQualification,
) -> dict[str, Any]:
    selection = value.selection
    record = {
        "evidence_closed_record_sha256": _required_string(
            value.evidence_record,
            "closed_record_sha256",
        ),
        "evidence_file_sha256": value.evidence_file_sha256,
        "evidence_uri": value.evidence_uri,
        "expected_artifact_pins": value.expected_artifact_pins.to_record(),
        "expected_campaign_id": value.expected_campaign_id,
        "generation_artifacts_sha256": selection.generation_artifacts_sha256,
        "generation_databricks_node_type_id": (
            selection.generation_databricks_node_type_id
        ),
        "generation_hardware_id": selection.generation_hardware_id,
        "generation_prefix_tokens_per_second": (
            selection.generation_prefix_tokens_per_second
        ),
        "plan_closed_record_sha256": selection.plan_sha256,
        "plan_file_sha256": value.plan_file_sha256,
        "plan_uri": value.plan_uri,
    }
    _validate_hardware_qualification_record(record)
    return record


def _validate_hardware_qualification_record(record: Mapping[str, Any]) -> None:
    for field_name in (
        "evidence_closed_record_sha256",
        "evidence_file_sha256",
        "generation_artifacts_sha256",
        "plan_closed_record_sha256",
        "plan_file_sha256",
    ):
        _require_sha256(_required_string(record, field_name), field_name=field_name)
    for field_name in (
        "evidence_uri",
        "expected_campaign_id",
        "plan_uri",
    ):
        _required_string(record, field_name)
    pins = _required_mapping(record, "expected_artifact_pins")
    _gpu_qualification_artifact_pins_from_record(pins)
    if record.get("generation_hardware_id") != (
        PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET
    ):
        raise ValueError("GPU qualification binding must select L40S")
    if record.get("generation_databricks_node_type_id") != (
        PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
    ):
        raise ValueError("GPU qualification binding must select g6e.4xlarge")
    throughput = _required_positive_number(
        record,
        "generation_prefix_tokens_per_second",
    )
    if throughput < PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND:
        raise ValueError("GPU qualification binding is below 35 tokens/s")


def _worker_payload_file_sha256(payload: Mapping[str, Any]) -> str:
    if not isinstance(payload, Mapping):
        raise TypeError("worker_payload must be a mapping")
    return sha256(_canonical_json_bytes(payload, pretty=True)).hexdigest()


def _require_authorized_worker_payload(
    payload: Mapping[str, Any],
    authorization: object,
    *,
    worker_index: int,
    expected_worker_payload_file_sha256: str | None = None,
) -> GPUQualificationSelection:
    """Join one worker record to a replay-issued qualification capability."""

    _validate_producer_worker_index(worker_index)
    if payload.get("closed_record_sha256") != _closed_record_sha256(payload):
        raise ValueError("worker payload closure is invalid")
    if _required_int(payload, "worker_index") != worker_index:
        raise ValueError("worker payload identity differs from the producer")
    observed_payload_sha256 = _worker_payload_file_sha256(payload)
    if expected_worker_payload_file_sha256 is not None:
        expected_payload_sha256 = _require_sha256(
            expected_worker_payload_file_sha256,
            field_name="expected_worker_payload_file_sha256",
        )
        if not hmac.compare_digest(
            observed_payload_sha256,
            expected_payload_sha256,
        ):
            raise ValueError(
                "worker payload changed after launch-authorization preflight"
            )

    qualification = _required_mapping(
        payload,
        "generator_hardware_qualification",
    )
    _validate_hardware_qualification_record(qualification)
    selection = require_gpu_qualification_launch_authorization(
        authorization,
        expected_plan_sha256=_required_string(
            qualification,
            "plan_closed_record_sha256",
        ),
        expected_evidence_file_sha256=_required_string(
            qualification,
            "evidence_file_sha256",
        ),
    )
    if not isinstance(authorization, GPUQualificationLaunchAuthorization):
        raise TypeError(
            "publication launch requires GPUQualificationLaunchAuthorization"
        )
    if authorization.evidence_closed_record_sha256 != _required_string(
        qualification,
        "evidence_closed_record_sha256",
    ):
        raise ValueError(
            "GPU qualification authorization evidence closure differs from worker payload"
        )
    exact_selection = {
        "generation_artifacts_sha256": selection.generation_artifacts_sha256,
        "generation_databricks_node_type_id": (
            selection.generation_databricks_node_type_id
        ),
        "generation_hardware_id": selection.generation_hardware_id,
        "generation_prefix_tokens_per_second": (
            selection.generation_prefix_tokens_per_second
        ),
        "plan_closed_record_sha256": selection.plan_sha256,
    }
    if any(
        qualification.get(field_name) != expected
        for field_name, expected in exact_selection.items()
    ):
        raise ValueError(
            "GPU qualification launch authority selection differs from worker payload"
        )
    return selection


def _single_spark_python_parameter(
    submit_payload: Mapping[str, Any],
    flag: str,
) -> str:
    tasks = submit_payload.get("tasks")
    if (
        not isinstance(tasks, Sequence)
        or isinstance(tasks, (str, bytes, bytearray))
        or len(tasks) != 1
        or not isinstance(tasks[0], Mapping)
    ):
        raise ValueError("producer submit payload requires exactly one task")
    task = cast(Mapping[str, Any], tasks[0])
    spark_python_task = _required_mapping(task, "spark_python_task")
    parameters = spark_python_task.get("parameters")
    if (
        not isinstance(parameters, Sequence)
        or isinstance(parameters, (str, bytes, bytearray))
        or any(not isinstance(value, str) for value in parameters)
    ):
        raise ValueError("producer spark_python_task parameters are invalid")
    positions = [index for index, value in enumerate(parameters) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(parameters):
        raise ValueError(f"producer parameters must contain one {flag}")
    value = parameters[positions[0] + 1]
    if not isinstance(value, str) or not value:
        raise ValueError(f"producer parameter {flag} must have a value")
    return value


def _validate_worker_payload_submit_binding(
    submit_payload: Mapping[str, Any],
    *,
    worker_payload: Mapping[str, Any],
    worker_index: int,
    expected_worker_payload_file_sha256: str,
) -> None:
    """Bind the reserved cloud payload to the exact authorized worker bytes."""

    _validate_single_producer_submit_payload(
        submit_payload,
        worker_index=worker_index,
    )
    observed_worker_payload_sha256 = _worker_payload_file_sha256(worker_payload)
    expected_payload_sha256 = _require_sha256(
        expected_worker_payload_file_sha256,
        field_name="expected_worker_payload_file_sha256",
    )
    if not hmac.compare_digest(
        observed_worker_payload_sha256,
        expected_payload_sha256,
    ):
        raise ValueError("worker payload changed after launch-authorization preflight")
    qualification = _required_mapping(
        worker_payload,
        "generator_hardware_qualification",
    )
    pins = _required_mapping(qualification, "expected_artifact_pins")
    expected_parameters = {
        "--expected-worker-payload-sha256": expected_payload_sha256,
        "--package-wheel-sha256": _required_string(
            pins,
            "package_wheel_sha256",
        ),
        "--patched-vllm-wheel-sha256": _required_string(
            pins,
            "patched_vllm_wheel_sha256",
        ),
        "--runtime-lock-sha256": _required_string(
            pins,
            "runtime_lock_sha256",
        ),
    }
    for flag, expected in expected_parameters.items():
        _require_sha256(expected, field_name=flag)
        if not hmac.compare_digest(
            _single_spark_python_parameter(submit_payload, flag),
            expected,
        ):
            raise ValueError(
                f"producer parameter {flag} differs from the authorized worker payload"
            )
    if worker_payload.get("input_bundle_sha256") != pins.get("input_bundle_sha256"):
        raise ValueError(
            "worker input bundle differs from the qualified artifact binding"
        )


def _execution_config_from_record(
    record: Mapping[str, Any],
) -> PublicationLatencyHandoffExecutionConfig:
    layout = _required_mapping(record, "layout")
    return PublicationLatencyHandoffExecutionConfig(
        layout=KVLayout(
            model_id=_required_string(layout, "model_id"),
            lora_id=_required_string(layout, "lora_id"),
            layout_version=_required_string(layout, "layout_version"),
            dtype=_required_string(layout, "dtype"),
            num_layers=_required_int(layout, "num_layers"),
            block_size=_required_int(layout, "block_size"),
            bytes_per_token=_required_int(layout, "bytes_per_token"),
            num_query_heads=_optional_int(layout, "num_query_heads"),
            num_kv_heads=_optional_int(layout, "num_kv_heads"),
            head_size=_optional_int(layout, "head_size"),
            kv_stride_bytes=_optional_int(layout, "kv_stride_bytes"),
            shares_kv_storage=layout.get("shares_kv_storage") is True,
            storage_layout=_required_string(layout, "storage_layout"),
            payload_axis_order=_required_string(layout, "payload_axis_order"),
            pre_rope=layout.get("pre_rope") is True,
            rope_theta=_optional_number(layout, "rope_theta"),
            rope_rotary_dim=_optional_int(layout, "rope_rotary_dim"),
            key_position_encoding=_required_string(
                layout,
                "key_position_encoding",
            ),
        ),
        model_revision=_required_string(record, "model_revision"),
        generator_version=_required_string(record, "generator_version"),
        vllm_bitsandbytes_loader_source_sha256=_required_string(
            record,
            "vllm_bitsandbytes_loader_source_sha256",
        ),
        tokenizer_id=_required_string(record, "tokenizer_id"),
        tokenizer_revision=_required_string(record, "tokenizer_revision"),
        generator_family=_required_string(record, "generator_family"),
        generator_device_map=_required_string(record, "generator_device_map"),
        generator_model_dtype=_required_string(record, "generator_model_dtype"),
        generator_cache_axis_order=_required_string(
            record,
            "generator_cache_axis_order",
        ),
        generator_trust_remote_code=_required_bool(
            record,
            "generator_trust_remote_code",
        ),
        generator_add_special_tokens=_required_bool(
            record,
            "generator_add_special_tokens",
        ),
        generator_quantization=_required_string(
            record,
            "generator_quantization",
        ),
        generator_quantization_config=_required_mapping(
            record,
            "generator_quantization_config",
        ),
        tensor_parallel_size=_required_int(record, "tensor_parallel_size"),
        pipeline_parallel_size=_required_int(record, "pipeline_parallel_size"),
        align_bytes=_required_int(record, "align_bytes"),
    )


def _validate_worker_output_binding(
    payload: Mapping[str, Any],
    *,
    worker: Mapping[str, Any],
) -> None:
    worker_index = _required_int(payload, "worker_index")
    binding = _required_mapping(payload, "output_binding")
    contexts = sorted(
        {
            _required_int(item, "context_tokens")
            for item in _mapping_sequence(
                worker.get("items"),
                field_name="worker.items",
            )
        }
    )
    expected = {
        "context_worker_relative_roots": {
            str(context): f"pending/{context}/worker-{worker_index:02d}"
            for context in contexts
        },
        "partial_records_relative_root": (f"worker-records/worker-{worker_index:02d}"),
        "result_relative_path": f"worker-results/worker-{worker_index:02d}.json",
    }
    if dict(binding) != expected:
        raise ValueError("worker output binding drift")


def _is_databricks_durable_uri(
    value: str,
    *,
    plan_closed_record_sha256: str,
) -> bool:
    _require_sha256(
        plan_closed_record_sha256,
        field_name="plan_closed_record_sha256",
    )
    if value.startswith("dbfs:/"):
        normalized = value.removeprefix("dbfs:/")
    elif value.startswith("uc-volume:"):
        normalized = value.removeprefix("uc-volume:").lstrip("/")
    elif value.startswith("/Volumes/"):
        normalized = value.lstrip("/")
    else:
        return False
    parts = normalized.split("/")
    if len(parts) < 3 or any(not part or part in {".", ".."} for part in parts):
        return False
    if parts[0] == "Volumes" and len(parts) < 6:
        return False
    return parts[-2:] == [
        "publication-latency-handoffs",
        plan_closed_record_sha256,
    ]


def _build_l40s_single_node_cluster(
    config: DatabricksPublicationLatencyHandoffJobConfig,
) -> dict[str, Any]:
    reserved = {"ResourceClass", "purpose", "hardware_target", "gpu_model"}
    overlap = reserved.intersection(config.custom_tags)
    if overlap:
        raise ValueError(f"custom_tags override reserved L40S tags: {sorted(overlap)}")
    cluster: dict[str, Any] = {
        "aws_attributes": {
            "availability": config.availability,
            "zone_id": config.zone_id,
        },
        "custom_tags": {
            "ResourceClass": "SingleNode",
            "gpu_model": PUBLICATION_LATENCY_HANDOFF_GENERATOR_GPU_MODEL,
            "hardware_target": PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET,
            "purpose": "cachet-vllm-0271-latency-handoff-generation",
            **dict(config.custom_tags),
        },
        "data_security_mode": config.data_security_mode,
        "driver_node_type_id": config.node_type_id,
        "node_type_id": config.node_type_id,
        "num_workers": 0,
        "spark_conf": {
            "spark.databricks.cluster.profile": "singleNode",
            "spark.master": "local[*]",
        },
        "spark_version": config.spark_version,
    }
    if config.data_security_mode == "SINGLE_USER":
        cluster["single_user_name"] = config.single_user_name
    return cluster


def _validate_producer_worker_index(worker_index: int) -> None:
    if type(worker_index) is not int or not (
        0 <= worker_index < PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS
    ):
        raise ValueError("worker_index must be between 0 and 15")


def _validate_single_producer_submit_payload(
    payload: Mapping[str, Any],
    *,
    worker_index: int,
) -> None:
    if payload.get("timeout_seconds") != (
        PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS
    ):
        raise ValueError("producer run timeout must equal five hours")
    tasks = payload.get("tasks")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        raise ValueError("producer submit payload tasks must be an array")
    if len(tasks) != 1 or not isinstance(tasks[0], Mapping):
        raise ValueError("each producer run must contain exactly one GPU task")
    task = cast(Mapping[str, Any], tasks[0])
    if task.get("task_key") != f"latency_handoff_worker_{worker_index:02d}":
        raise ValueError("producer task key does not match worker index")
    if task.get("timeout_seconds") != (
        PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS
    ):
        raise ValueError("producer task timeout must equal five hours")
    if task.get("max_retries") != 0:
        raise ValueError("producer task retries must be disabled")
    cluster = task.get("new_cluster")
    if not isinstance(cluster, Mapping) or cluster.get("node_type_id") != (
        PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
    ):
        raise ValueError("producer task must use one g6e.4xlarge L40S node")
    if cluster.get("num_workers") != 0:
        raise ValueError("producer task must use a single-node cluster")


def _validate_publication_latency_handoff_terminal_status(
    status: Mapping[str, Any],
    *,
    worker_index: int,
    expected_run_name: str,
    submit_payload_sha256: str,
) -> None:
    """Validate the sanitizer output against the L40S producer contract."""

    if (
        status.get("terminal") is not True
        or status.get("succeeded") is not True
        or status.get("life_cycle_state") != "TERMINATED"
        or status.get("result_state") != "SUCCESS"
        or status.get("active_task_key") is not None
        or status.get("task_count") != 1
        or status.get("run_name") != expected_run_name
    ):
        raise ValueError("Databricks terminal status is not one successful task")
    expected_task_key = f"latency_handoff_worker_{worker_index:02d}"
    status_tasks = _mapping_sequence(
        status.get("tasks"),
        field_name="terminal status tasks",
    )
    if len(status_tasks) != 1:
        raise ValueError("Databricks terminal status must contain exactly one task")
    task = status_tasks[0]
    if (
        task.get("task_key") != expected_task_key
        or task.get("life_cycle_state") != "TERMINATED"
        or task.get("result_state") != "SUCCESS"
        or task.get("node_type_id")
        != PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
        or task.get("driver_node_type_id")
        != PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
    ):
        raise ValueError("Databricks terminal task is not the exact L40S producer")
    submit = _required_mapping(status, "submit_payload")
    if (
        submit.get("sha256") != submit_payload_sha256
        or submit.get("run_name") != expected_run_name
        or submit.get("task_count") != 1
        or submit.get("task_keys") != [expected_task_key]
        or submit.get("node_type_ids")
        != [PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID]
        or submit.get("driver_node_type_ids")
        != [PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID]
        or submit.get("single_node") is not True
    ):
        raise ValueError(
            "Databricks terminal submit summary is not the reserved payload"
        )
    submit_tasks = _mapping_sequence(
        submit.get("tasks"),
        field_name="terminal submit tasks",
    )
    if len(submit_tasks) != 1:
        raise ValueError("Databricks terminal submit summary must contain one task")
    submit_task = submit_tasks[0]
    if (
        submit_task.get("task_key") != expected_task_key
        or submit_task.get("node_type_id")
        != PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
        or submit_task.get("driver_node_type_id")
        != PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
        or submit_task.get("single_node") is not True
        or submit_task.get("data_security_mode")
        != DEFAULT_DATABRICKS_DATA_SECURITY_MODE
        or task.get("spark_env_keys") != submit_task.get("spark_env_keys")
    ):
        raise ValueError("Databricks terminal submit task summary is invalid")


def _validate_publication_latency_handoff_databricks_attestation_record(
    record: Mapping[str, Any],
) -> None:
    _require_exact_mapping_keys(
        record,
        {
            "attempt",
            "closed_record_sha256",
            "cloud_execution",
            "record_type",
            "schema_version",
            "worker_result",
        },
        label="Databricks execution attestation",
    )
    if record.get("record_type") != (
        PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_RECORD_TYPE
    ):
        raise ValueError("Databricks execution attestation record_type is invalid")
    if record.get("schema_version") != (
        PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_SCHEMA_VERSION
    ):
        raise ValueError("Databricks execution attestation schema_version is invalid")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("Databricks execution attestation closure is invalid")

    attempt = _required_mapping(record, "attempt")
    _require_exact_mapping_keys(
        attempt,
        {
            "attempt_id",
            "ledger_id",
            "ledger_path_sha256",
            "producer_batch_prefix",
            "reserved_gpu_hours",
            "submit_response_sha256",
            "submit_payload_sha256",
            "worker_index",
            "workload_id",
        },
        label="Databricks execution attestation attempt",
    )
    worker_index = _required_int(attempt, "worker_index")
    _validate_producer_worker_index(worker_index)
    _required_string(attempt, "attempt_id")
    _required_string(attempt, "ledger_id")
    _require_sha256(
        attempt.get("ledger_path_sha256"),
        field_name="attested ledger_path_sha256",
    )
    producer_batch_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(attempt, "producer_batch_prefix")
    )
    if producer_batch_prefix.ledger_id != attempt.get("ledger_id"):
        raise ValueError("attested producer batch prefix identity drift")
    expected_workload_id = f"publication-latency-handoff-worker-{worker_index:02d}"
    if attempt.get("workload_id") != expected_workload_id:
        raise ValueError("Databricks attestation workload identity is invalid")
    if attempt.get("reserved_gpu_hours") != 5.0:
        raise ValueError("Databricks attestation must bind a five-hour reservation")
    _require_sha256(
        attempt.get("submit_payload_sha256"),
        field_name="attested submit_payload_sha256",
    )
    _require_sha256(
        attempt.get("submit_response_sha256"),
        field_name="attested submit_response_sha256",
    )

    cloud = _required_mapping(record, "cloud_execution")
    _require_exact_mapping_keys(
        cloud,
        {
            "actual_gpu_duration_seconds",
            "attempt_number",
            "cluster_id",
            "control_plane_status_sha256",
            "life_cycle_state",
            "original_attempt_run_id",
            "parent_end_time_epoch_ms",
            "parent_run_id",
            "parent_start_time_epoch_ms",
            "repair_count",
            "result_state",
            "task_end_time_epoch_ms",
            "task_key",
            "task_run_id",
            "task_start_time_epoch_ms",
            "terminal_state",
        },
        label="Databricks execution attestation cloud_execution",
    )
    if cloud.get("attempt_number") != 0 or cloud.get("repair_count") != 0:
        raise ValueError(
            "Databricks execution attestation must be attempt 0 without repair"
        )
    if (
        cloud.get("life_cycle_state") != "TERMINATED"
        or cloud.get("result_state") != "SUCCESS"
        or cloud.get("terminal_state") != "succeeded"
    ):
        raise ValueError(
            "Databricks execution attestation is not a successful terminal run"
        )
    if cloud.get("task_key") != f"latency_handoff_worker_{worker_index:02d}":
        raise ValueError("Databricks execution attestation task key is invalid")
    for field_name in ("parent_run_id", "task_run_id", "cluster_id"):
        _databricks_cloud_id(cloud.get(field_name), field_name=field_name)
    if cloud.get("original_attempt_run_id") != cloud.get("parent_run_id"):
        raise ValueError("Databricks attestation is not the original run attempt")
    if cloud.get("task_run_id") == cloud.get("parent_run_id"):
        raise ValueError("Databricks attestation parent/task run IDs are not distinct")
    _require_sha256(
        cloud.get("control_plane_status_sha256"),
        field_name="control_plane_status_sha256",
    )
    parent_start = _positive_epoch_millis(cloud, "parent_start_time_epoch_ms")
    parent_end = _positive_epoch_millis(cloud, "parent_end_time_epoch_ms")
    task_start = _positive_epoch_millis(cloud, "task_start_time_epoch_ms")
    task_end = _positive_epoch_millis(cloud, "task_end_time_epoch_ms")
    if not parent_start <= task_start < task_end <= parent_end:
        raise ValueError("Databricks attestation timestamps are not causally nested")
    duration = _required_positive_number(cloud, "actual_gpu_duration_seconds")
    if not math.isclose(
        duration,
        (task_end - task_start) / 1000.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Databricks attestation duration is not derived from timestamps"
        )
    if duration > PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS:
        raise ValueError("Databricks attestation duration exceeds five hours")

    worker_result = _required_mapping(record, "worker_result")
    _require_exact_mapping_keys(
        worker_result,
        {"closed_record_sha256", "file_sha256", "relative_path"},
        label="Databricks execution attestation worker_result",
    )
    if worker_result.get("relative_path") != (
        f"worker-results/worker-{worker_index:02d}.json"
    ):
        raise ValueError("Databricks attestation worker-result path is invalid")
    for field_name in ("closed_record_sha256", "file_sha256"):
        _require_sha256(
            worker_result.get(field_name),
            field_name=f"attested worker result {field_name}",
        )


def _read_bound_closed_json(
    uri: str,
    *,
    file_sha256: str,
    closed_record_sha256: str,
    field_name: str,
) -> dict[str, Any]:
    path = Path(local_path(uri)).expanduser().resolve()
    content = path.read_bytes()
    if not hmac.compare_digest(sha256(content).hexdigest(), file_sha256):
        raise ValueError(f"{field_name} file SHA-256 drift")
    record = _json_object(content, field_name=field_name)
    if content != _canonical_json_bytes(record, pretty=True):
        raise ValueError(f"{field_name} is not canonical JSON")
    if record.get("closed_record_sha256") != closed_record_sha256:
        raise ValueError(f"{field_name} bound closure drift")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError(f"{field_name} closed-record SHA-256 drift")
    return record


def _read_bound_gpu_qualification_json(
    uri: str,
    *,
    file_sha256: str,
    closed_record_sha256: str,
    field_name: str,
) -> dict[str, Any]:
    path = Path(local_path(uri)).expanduser().resolve()
    content = path.read_bytes()
    if not hmac.compare_digest(sha256(content).hexdigest(), file_sha256):
        raise ValueError(f"{field_name} file SHA-256 drift")
    record = _json_object(content, field_name=field_name)
    expected = (canonical_gpu_qualification_json(record) + "\n").encode("utf-8")
    if content != expected:
        raise ValueError(f"{field_name} is not canonical JSON")
    if record.get("closed_record_sha256") != closed_record_sha256:
        raise ValueError(f"{field_name} bound closure drift")
    return record


def _gpu_qualification_artifact_pins_from_record(
    record: Mapping[str, Any],
) -> GPUQualificationArtifactPins:
    return GPUQualificationArtifactPins(
        runtime_lock_sha256=_required_string(record, "runtime_lock_sha256"),
        patched_vllm_wheel_sha256=_required_string(
            record,
            "patched_vllm_wheel_sha256",
        ),
        package_wheel_sha256=_required_string(record, "package_wheel_sha256"),
        cachet_source_tree_sha256=_required_string(
            record,
            "cachet_source_tree_sha256",
        ),
        runner_sha256=_required_string(record, "runner_sha256"),
        input_bundle_sha256=_required_string(record, "input_bundle_sha256"),
    )


def _verify_bound_hardware_qualification_file(
    binding: Mapping[str, Any],
) -> None:
    _validate_hardware_qualification_record(binding)
    evidence = _read_bound_gpu_qualification_json(
        _required_string(binding, "evidence_uri"),
        file_sha256=_required_string(binding, "evidence_file_sha256"),
        closed_record_sha256=_required_string(
            binding,
            "evidence_closed_record_sha256",
        ),
        field_name="GPU qualification evidence",
    )
    plan = _read_bound_gpu_qualification_json(
        _required_string(binding, "plan_uri"),
        file_sha256=_required_string(binding, "plan_file_sha256"),
        closed_record_sha256=_required_string(
            binding,
            "plan_closed_record_sha256",
        ),
        field_name="GPU qualification plan",
    )
    selection = validate_gpu_qualification_evidence_record(
        evidence,
        plan_record=plan,
        expected_campaign_id=_required_string(binding, "expected_campaign_id"),
        expected_artifact_pins=_gpu_qualification_artifact_pins_from_record(
            _required_mapping(binding, "expected_artifact_pins")
        ),
    )
    expected = {
        "generation_artifacts_sha256": selection.generation_artifacts_sha256,
        "generation_databricks_node_type_id": (
            selection.generation_databricks_node_type_id
        ),
        "generation_hardware_id": selection.generation_hardware_id,
        "generation_prefix_tokens_per_second": (
            selection.generation_prefix_tokens_per_second
        ),
        "plan_closed_record_sha256": selection.plan_sha256,
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("worker GPU qualification selection binding drift")


def _verify_file_sha256(path: Path, expected: str, *, field_name: str) -> None:
    _require_sha256(expected, field_name=f"{field_name} SHA-256")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{field_name} must be a real file")
    if not hmac.compare_digest(_file_sha256(path), expected):
        raise ValueError(f"{field_name} SHA-256 drift")


def _require_worker_output_is_fresh(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
) -> None:
    binding = _required_mapping(payload, "output_binding")
    context_roots = _required_mapping(
        binding,
        "context_worker_relative_roots",
    )
    paths = [
        _confined_relative_path(output_root, str(value), field_name="context root")
        for value in context_roots.values()
    ]
    paths.extend(
        [
            _confined_relative_path(
                output_root,
                _required_string(binding, "partial_records_relative_root"),
                field_name="partial records root",
            ),
            _confined_relative_path(
                output_root,
                _required_string(binding, "result_relative_path"),
                field_name="worker result path",
            ),
        ]
    )
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"worker output is not fresh: {path}")


def _apply_production_generator_environment(
    config: PublicationLatencyHandoffExecutionConfig,
) -> None:
    fixed = {
        CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV: (
            "1" if config.generator_add_special_tokens else "0"
        ),
        CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV: config.generator_cache_axis_order,
        CACHET_TRANSFORMERS_DEVICE_ENV: "cuda",
        CACHET_TRANSFORMERS_DEVICE_MAP_ENV: config.generator_device_map,
        CACHET_TRANSFORMERS_MODEL_ID_ENV: config.layout.model_id,
        CACHET_TRANSFORMERS_MODEL_REVISION_ENV: config.model_revision,
        CACHET_TRANSFORMERS_PRE_ROPE_ENV: "1",
        CACHET_TRANSFORMERS_QUANTIZATION_ENV: config.generator_quantization,
        CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV: json.dumps(
            dict(config.generator_quantization_config),
            sort_keys=True,
            separators=(",", ":"),
        ),
        CACHET_TRANSFORMERS_TOKENIZER_ID_ENV: config.tokenizer_id,
        CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV: config.tokenizer_revision,
        CACHET_TRANSFORMERS_TORCH_DTYPE_ENV: (config.generator_model_dtype),
        CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV: (
            "1" if config.generator_trust_remote_code else "0"
        ),
    }
    for key, value in fixed.items():
        observed = os.environ.get(key)
        if observed is not None and observed != value:
            raise ValueError(f"runtime environment {key} conflicts with protocol")
        os.environ[key] = value


def _verify_installed_vllm_bitsandbytes_loader_source(
    config: PublicationLatencyHandoffExecutionConfig,
) -> None:
    module = importlib.import_module(
        "vllm.model_executor.model_loader.bitsandbytes_loader"
    )
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise RuntimeError("vLLM bitsandbytes loader has no source file")
    path = Path(source).resolve()
    if path.suffix == ".pyc" and path.with_suffix(".py").is_file():
        path = path.with_suffix(".py")
    _verify_file_sha256(
        path,
        config.vllm_bitsandbytes_loader_source_sha256,
        field_name="vLLM bitsandbytes loader source",
    )


def _production_generator_factory(worker_index: int) -> KVChunkGenerator:
    del worker_index
    return build_pre_rope_transformers_kv_chunk_generator()


def _probe_single_l40s_hardware() -> Mapping[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - production dependency.
        raise RuntimeError("torch is required to attest producer hardware") from exc
    count = int(torch.cuda.device_count())
    if count != 1:
        raise ValueError("latency producer requires exactly one visible GPU")
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_device_count": count,
        "cuda_device_name": str(torch.cuda.get_device_name(0)),
        "cuda_device_total_memory_bytes": int(properties.total_memory),
        "cuda_major": int(properties.major),
        "cuda_minor": int(properties.minor),
        "gpu_model": PUBLICATION_LATENCY_HANDOFF_GENERATOR_GPU_MODEL,
        "hardware_target": PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET,
        "node_type_id": PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID,
    }


def _validate_observed_l40s_hardware(record: Mapping[str, Any]) -> None:
    if record.get("cuda_device_count") != 1:
        raise ValueError("producer hardware must expose exactly one GPU")
    if "L40S" not in _required_string(record, "cuda_device_name").upper():
        raise ValueError("producer hardware must be an NVIDIA L40S")
    expected = {
        "gpu_model": PUBLICATION_LATENCY_HANDOFF_GENERATOR_GPU_MODEL,
        "hardware_target": PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET,
        "node_type_id": PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("producer hardware identity drift")


def _worker_bundle_file_records(
    output_root: Path,
    *,
    worker_result: _WorkerResult,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch in worker_result.batches:
        context_root = output_root / "pending" / str(batch.context_tokens)
        worker_root = context_root / f"worker-{worker_result.worker_index:02d}"
        if not worker_root.is_dir() or worker_root.is_symlink():
            raise ValueError("worker context output is missing")
        for path in sorted(worker_root.rglob("*")):
            if path.is_symlink():
                raise ValueError("worker output contains a symlink")
            if not path.is_file():
                continue
            relative_name = path.relative_to(context_root).as_posix()
            records.append(
                {
                    "byte_count": path.stat().st_size,
                    "context_tokens": batch.context_tokens,
                    "relative_name": relative_name,
                    "sha256": _file_sha256(path),
                }
            )
    if not records:
        raise ValueError("worker did not produce durable bundle files")
    return records


def _read_worker_result(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"worker result is missing: {path}")
    content = path.read_bytes()
    record = _json_object(content, field_name="worker result")
    if content != _canonical_json_bytes(record, pretty=True):
        raise ValueError("worker result is not canonical JSON")
    if (
        record.get("record_type")
        != PUBLICATION_LATENCY_HANDOFF_WORKER_RESULT_RECORD_TYPE
    ):
        raise ValueError("worker result record_type is invalid")
    if (
        record.get("schema_version")
        != PUBLICATION_LATENCY_HANDOFF_WORKER_RESULT_SCHEMA_VERSION
    ):
        raise ValueError("worker result schema_version is invalid")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("worker result closure is invalid")
    return record


def _validate_worker_result_and_load_batches(
    result: Mapping[str, Any],
    *,
    result_path: Path,
    output_root: Path,
    worker: Mapping[str, Any],
    plan: Mapping[str, Any],
    config: PublicationLatencyHandoffExecutionConfig,
) -> tuple[_WorkerBatchResult, ...]:
    worker_index = _required_int(worker, "worker_index")
    if result.get("worker_index") != worker_index or result.get(
        "worker_id"
    ) != worker.get("worker_id"):
        raise ValueError("worker result identity drift")
    if result.get("plan_closed_record_sha256") != plan.get("closed_record_sha256"):
        raise ValueError("worker result plan drift")
    if result.get("input_bundle_sha256") != plan.get("input_bundle_sha256"):
        raise ValueError("worker result input bundle drift")
    if result.get("execution_contract") != _execution_config_record(config):
        raise ValueError("worker result execution contract drift")
    task_ids = result.get("task_ids")
    if not isinstance(task_ids, list) or any(
        not isinstance(item, str) or not item for item in task_ids
    ):
        raise ValueError("worker result task_ids are invalid")
    expected_task_ids = sorted(
        _required_string(item, "task_id")
        for item in _mapping_sequence(worker.get("items"), field_name="worker.items")
    )
    if task_ids != expected_task_ids or result.get("task_ids_sha256") != (
        _canonical_sha256(expected_task_ids)
    ):
        raise ValueError("worker result task coverage drift")
    bundle_files = _mapping_sequence(
        result.get("bundle_files"),
        field_name="bundle_files",
    )
    if result.get("bundle_files_sha256") != _canonical_sha256(bundle_files):
        raise ValueError("worker bundle file closure drift")
    bundle_paths: set[Path] = set()
    for file_record in bundle_files:
        context = _required_int(file_record, "context_tokens")
        if context not in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
            raise ValueError("worker bundle file context is invalid")
        context_root = output_root / "pending" / str(context)
        path = _confined_relative_path(
            context_root,
            _required_string(file_record, "relative_name"),
            field_name="worker bundle file",
        )
        expected_worker_root = context_root / f"worker-{worker_index:02d}"
        _require_confined_path(
            expected_worker_root.resolve(),
            path.resolve(),
            field_name="worker bundle file",
        )
        _verify_file_record(path, file_record)
        if path in bundle_paths:
            raise ValueError("worker result contains duplicate bundle paths")
        bundle_paths.add(path)
    actual_bundle_paths = {
        path
        for context in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS
        for path in (
            output_root / "pending" / str(context) / f"worker-{worker_index:02d}"
        ).rglob("*")
        if path.is_file()
    }
    if actual_bundle_paths != bundle_paths:
        raise ValueError("worker bundle tree contains unrecorded files")
    partial_files = _mapping_sequence(
        result.get("partial_record_files"),
        field_name="partial_record_files",
    )
    partial_by_context: dict[int, tuple[dict[str, Any], ...]] = {}
    for file_record in partial_files:
        context = _required_int(file_record, "context_tokens")
        path = _confined_relative_path(
            output_root,
            _required_string(file_record, "relative_path"),
            field_name="partial record file",
        )
        expected_parent = output_root / "worker-records" / f"worker-{worker_index:02d}"
        if path.parent.resolve() != expected_parent.resolve():
            raise ValueError("partial record path belongs to another worker")
        _verify_file_record(path, file_record)
        rows = _canonical_jsonl_records(path)
        if len(rows) != _required_int(file_record, "record_count"):
            raise ValueError("partial record count drift")
        if context in partial_by_context:
            raise ValueError("worker has duplicate partial context records")
        partial_by_context[context] = rows
    planned_items = _mapping_sequence(worker.get("items"), field_name="worker.items")
    expected_contexts = {
        _required_int(item, "context_tokens") for item in planned_items
    }
    if set(partial_by_context) != expected_contexts:
        raise ValueError("worker partial context coverage is incomplete")
    batches: list[_WorkerBatchResult] = []
    for context in sorted(expected_contexts):
        items = tuple(
            item for item in planned_items if item.get("context_tokens") == context
        )
        rows = partial_by_context[context]
        expected_identities = {
            (_required_string(item, "dataset"), _required_string(item, "example_id"))
            for item in items
        }
        observed_identities = {
            (_required_string(row, "dataset"), _required_string(row, "example_id"))
            for row in rows
        }
        if observed_identities != expected_identities or len(rows) != len(items):
            raise ValueError("worker partial records do not match planned identities")
        context_task_ids = tuple(_required_string(item, "task_id") for item in items)
        batches.append(
            _WorkerBatchResult(
                context_tokens=context,
                worker_index=worker_index,
                records=rows,
                task_ids=context_task_ids,
                cache_prefix_tokens=sum(
                    _required_int(item, "cache_prefix_tokens") for item in items
                ),
                input_token_slots=sum(
                    _required_int(item, "input_token_slots") for item in items
                ),
                generation_seconds=1e-12,
                durable_sync_seconds=0.0,
                durable_byte_count=sum(
                    _required_int(item, "byte_count")
                    for item in bundle_files
                    if item.get("context_tokens") == context
                ),
            )
        )
    accounting = _required_mapping(result, "accounting")
    if accounting.get("includes_generation_payload_hash_and_durable_sync") is not True:
        raise ValueError("worker result accounting scope is incomplete")
    if _required_int(accounting, "cache_prefix_tokens") != worker.get(
        "cache_prefix_tokens"
    ):
        raise ValueError("worker cache-prefix accounting drift")
    if _required_int(accounting, "input_token_slots") != worker.get(
        "input_token_slots"
    ):
        raise ValueError("worker token-slot accounting drift")
    durable_bytes = sum(_required_int(item, "byte_count") for item in bundle_files)
    durable_bytes += sum(_required_int(item, "byte_count") for item in partial_files)
    if _required_int(accounting, "durable_byte_count") != durable_bytes:
        raise ValueError("worker durable-byte accounting drift")
    hardware = _required_mapping(result, "generator_hardware")
    _validate_hardware_qualification_record(
        _required_mapping(hardware, "qualification")
    )
    _validate_observed_l40s_hardware(_required_mapping(hardware, "observed"))
    expected_result_path = (
        output_root / "worker-results" / (f"worker-{worker_index:02d}.json")
    )
    if result_path != expected_result_path:
        raise ValueError("worker result path drift")
    return tuple(batches)


def _validate_shared_worker_output_tree(
    root: Path,
    *,
    plan: Mapping[str, Any],
    worker_results: Sequence[Mapping[str, Any]],
) -> None:
    expected_result_paths = {
        root / "worker-results" / f"worker-{index:02d}.json"
        for index in range(PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS)
    }
    actual_result_paths = {
        path for path in (root / "worker-results").rglob("*") if path.is_file()
    }
    if actual_result_paths != expected_result_paths:
        raise ValueError("shared worker-results tree has extra or missing files")
    expected_record_paths = {
        _confined_relative_path(
            root,
            _required_string(item, "relative_path"),
            field_name="partial record file",
        )
        for result in worker_results
        for item in _mapping_sequence(
            result.get("partial_record_files"),
            field_name="partial_record_files",
        )
    }
    actual_record_paths = {
        path for path in (root / "worker-records").rglob("*") if path.is_file()
    }
    if actual_record_paths != expected_record_paths:
        raise ValueError("shared worker-records tree has extra or missing files")
    expected_worker_dirs = {
        (context, _required_int(worker, "worker_index"))
        for worker in _mapping_sequence(plan.get("workers"), field_name="workers")
        for context in {
            _required_int(item, "context_tokens")
            for item in _mapping_sequence(
                worker.get("items"),
                field_name="worker.items",
            )
        }
    }
    actual_worker_dirs = {
        (int(path.parent.name), int(path.name.removeprefix("worker-")))
        for context in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS
        for path in (root / "pending" / str(context)).iterdir()
        if path.is_dir() and path.name.startswith("worker-")
    }
    if actual_worker_dirs != expected_worker_dirs:
        raise ValueError("shared pending tree has extra or missing worker roots")


def _validated_terminal_actual_gpu_seconds(
    values: Mapping[int, float],
    *,
    worker_count: int,
) -> dict[int, float]:
    if not isinstance(values, Mapping) or set(values) != set(range(worker_count)):
        raise ValueError("terminal GPU actuals must cover every worker exactly once")
    normalized = {
        index: _positive_float(value, field_name=f"worker {index} terminal seconds")
        for index, value in values.items()
    }
    if any(
        value > PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS
        for value in normalized.values()
    ):
        raise ValueError("worker terminal actual exceeds the five-hour bound")
    return normalized


def _verify_file_record(path: Path, record: Mapping[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"recorded durable file is missing: {path}")
    if path.stat().st_size != _required_int(record, "byte_count"):
        raise ValueError("recorded durable file byte count drift")
    expected = _required_string(record, "sha256")
    _require_sha256(expected, field_name="file sha256")
    if not hmac.compare_digest(_file_sha256(path), expected):
        raise ValueError("recorded durable file SHA-256 drift")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute_publication_latency_handoff_generation_plan_local_test_helper(
    record: Mapping[str, Any],
    *,
    prepared_input_dir: str | Path,
    output_dir: str | Path,
    tokenizer: MainLatencyTokenizer,
    worker_factory: PublicationLatencyGeneratorFactory,
    config: PublicationLatencyHandoffExecutionConfig,
    source_paths: Mapping[str, str | Path] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> PublicationLatencyHandoffGenerationResult:
    """Exercise the generation contract with injected CPU-safe test doubles.

    This helper deliberately uses threads so unit tests can cover the complete
    closure path without Databricks.  It is not a production GPU executor: a
    production producer is one independent one-GPU process and must use the
    distributed worker/coordinator APIs below.
    """

    if not callable(worker_factory):
        raise TypeError("worker_factory must be callable")
    if not callable(clock):
        raise TypeError("clock must be callable")
    if not isinstance(config, PublicationLatencyHandoffExecutionConfig):
        raise TypeError("config must be a PublicationLatencyHandoffExecutionConfig")
    validate_publication_latency_handoff_generation_plan(
        record,
        prepared_input_dir=prepared_input_dir,
        tokenizer=tokenizer,
        source_paths=source_paths,
    )
    prepared = verify_main_latency_inputs(
        prepared_input_dir,
        tokenizer=tokenizer,
        source_paths=source_paths,
        examples_per_dataset=PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    )
    rows_by_task = _prepared_rows_by_task(prepared)
    workers = _mapping_sequence(record.get("workers"), field_name="workers")
    destination = Path(output_dir).expanduser().absolute()
    _require_fresh_output_path(destination)
    prepared_root = prepared.output_dir.expanduser().resolve()
    if _paths_overlap(destination.resolve(strict=False), prepared_root):
        raise ValueError("output_dir must not overlap the prepared input bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.generation-",
            dir=destination.parent,
        )
    )
    published = False
    execution_start = clock()
    try:
        work_root = temporary / "work"
        pending_root = temporary / "pending"
        work_root.mkdir()
        pending_root.mkdir()
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = [
                executor.submit(
                    _execute_worker,
                    worker,
                    rows_by_task=rows_by_task,
                    work_root=work_root,
                    pending_root=pending_root,
                    worker_factory=worker_factory,
                    config=config,
                    clock=clock,
                    enforce_production_generator_identity=False,
                )
                for worker in workers
            ]
            worker_results = tuple(future.result() for future in futures)
        shutil.rmtree(work_root)

        batches = tuple(
            batch for worker_result in worker_results for batch in worker_result.batches
        )
        expected_task_ids = _plan_task_ids(record)
        observed_task_ids = tuple(
            task_id for batch in batches for task_id in batch.task_ids
        )
        if Counter(observed_task_ids) != Counter(expected_task_ids):
            raise ValueError("generated latency handoff coverage is incomplete")
        if len(set(observed_task_ids)) != len(observed_task_ids):
            raise ValueError("generated latency handoff coverage contains duplicates")

        bundles_root = temporary / "bundles"
        manifests_root = temporary / "manifests"
        bundles_root.mkdir()
        manifests_root.mkdir()
        bundle_records: list[dict[str, Any]] = []
        for context_tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
            pending_context = pending_root / str(context_tokens)
            dataset_paths = _write_context_dataset_files(
                pending_context,
                context_tokens=context_tokens,
                batches=batches,
            )
            _sync_tree(pending_context)
            manifest = close_publication_latency_handoff_bundle(
                pending_context,
                dataset_paths,
                context_tokens=context_tokens,
                input_bundle_sha256=prepared.bundle_sha256,
            )
            _validate_publication_manifest_contract(manifest)
            portable_digest = cast(str, manifest["portable_bundle_sha256"])
            content_root = bundles_root / f"{context_tokens}-{portable_digest}"
            if content_root.exists() or content_root.is_symlink():
                raise ValueError("content-addressed context bundle collision")
            os.rename(pending_context, content_root)
            validate_publication_latency_handoff_bundle(
                manifest,
                bundle_root=content_root,
            )
            manifest_relative = (
                PurePosixPath("manifests") / f"{context_tokens}-{portable_digest}.json"
            )
            manifest_path = temporary / manifest_relative
            write_publication_latency_handoff_bundle(manifest, manifest_path)
            _sync_file(manifest_path)
            bundle_records.append(
                {
                    "closed_record_sha256": manifest["closed_record_sha256"],
                    "context_tokens": context_tokens,
                    "manifest_relative_path": str(manifest_relative),
                    "portable_bundle_sha256": portable_digest,
                    "source_root_relative_path": str(
                        content_root.relative_to(temporary)
                    ),
                }
            )
        if any(pending_root.iterdir()):
            raise ValueError("pending generation tree contains unclosed output")
        pending_root.rmdir()
        _sync_tree(bundles_root)
        _sync_tree(manifests_root)
        total_durable_bytes = _tree_byte_count(bundles_root) + _tree_byte_count(
            manifests_root
        )

        execution_end = clock()
        wall_seconds = _positive_duration(execution_end - execution_start)
        worker_count = len(workers)
        charged_gpu_seconds = wall_seconds * worker_count
        cache_prefix_tokens = _coverage_int(record, "cache_prefix_generation_tokens")
        input_token_slots = _coverage_int(record, "input_token_slots")
        tokens_per_gpu_second = cache_prefix_tokens / charged_gpu_seconds
        input_slots_per_gpu_second = input_token_slots / charged_gpu_seconds
        charged_gpu_hours = charged_gpu_seconds / 3600.0
        report: dict[str, Any] = {
            "accounting": {
                "charged_gpu_hours": charged_gpu_hours,
                "charged_gpu_seconds": charged_gpu_seconds,
                "cost_model": "persistent_worker_count_times_end_to_end_wall_time",
                "durable_byte_count": total_durable_bytes,
                "end_to_end_cache_prefix_tokens_per_gpu_second": (
                    tokens_per_gpu_second
                ),
                "end_to_end_input_token_slots_per_gpu_second": (
                    input_slots_per_gpu_second
                ),
                "end_to_end_wall_seconds": wall_seconds,
                "full_launch_min_tokens_per_gpu_second": (
                    PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
                ),
                "full_launch_throughput_gate_passed": (
                    tokens_per_gpu_second
                    >= PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
                ),
                "includes_bundle_closure_and_durable_sync": True,
                "worker_count": worker_count,
            },
            "bundles": bundle_records,
            "bundles_sha256": _canonical_sha256(bundle_records),
            "closed_record_sha256": "",
            "coverage": {
                "cache_prefix_generation_tokens": cache_prefix_tokens,
                "context_bundle_count": len(bundle_records),
                "generated_task_count": len(observed_task_ids),
                "input_token_slots": input_token_slots,
                "no_duplicate_tasks": True,
                "task_ids_sha256": _canonical_sha256(sorted(observed_task_ids)),
            },
            "execution_contract": _execution_config_record(config),
            "execution_mode": PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_LOCAL_TEST,
            "input_bundle_sha256": prepared.bundle_sha256,
            "plan_closed_record_sha256": record["closed_record_sha256"],
            "record_type": PUBLICATION_LATENCY_HANDOFF_EXECUTION_RECORD_TYPE,
            "schema_version": PUBLICATION_LATENCY_HANDOFF_EXECUTION_SCHEMA_VERSION,
            "serving_reuse": {
                "context_bundles": bundle_records,
                "regenerate_inside_timed_serving_jobs": False,
                "required_action": (
                    "validate_manifest_then_stage_content_addressed_bundle"
                ),
            },
            "workers": [_worker_result_record(result) for result in worker_results],
        }
        report["closed_record_sha256"] = _closed_record_sha256(report)
        execution_path = temporary / PUBLICATION_LATENCY_HANDOFF_EXECUTION_FILENAME
        _write_canonical_json_exclusive(report, execution_path)
        _sync_file(execution_path)
        _sync_directory(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite generation output: {destination}"
            )
        os.rename(temporary, destination)
        published = True
        return read_publication_latency_handoff_generation_result(destination)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def read_publication_latency_handoff_generation_result(
    output_dir: str | Path,
) -> PublicationLatencyHandoffGenerationResult:
    """Read and re-authenticate a published generation result and every bundle."""

    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("generation output must be a real directory")
    path = root / PUBLICATION_LATENCY_HANDOFF_EXECUTION_FILENAME
    content = path.read_bytes()
    record = _json_object(content, field_name="generation execution record")
    if content != _canonical_json_bytes(record, pretty=True):
        raise ValueError("generation execution record is not canonical JSON")
    validate_publication_latency_handoff_generation_execution_record(
        record,
        output_dir=root,
    )
    return PublicationLatencyHandoffGenerationResult(
        root=root,
        execution_record_path=path,
        record=record,
    )


def validate_publication_latency_handoff_generation_execution_record(
    record: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> None:
    """Validate coverage, accounting, and all durable serving bundle closures."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if record.get("record_type") != PUBLICATION_LATENCY_HANDOFF_EXECUTION_RECORD_TYPE:
        raise ValueError("generation execution record_type is invalid")
    if (
        record.get("schema_version")
        != PUBLICATION_LATENCY_HANDOFF_EXECUTION_SCHEMA_VERSION
    ):
        raise ValueError("generation execution schema_version is invalid")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("generation execution closed_record_sha256 is invalid")
    root = Path(output_dir).expanduser().resolve()
    coverage = _required_mapping(record, "coverage")
    if _required_int(coverage, "generated_task_count") != (
        PUBLICATION_LATENCY_HANDOFF_TASK_COUNT
    ):
        raise ValueError("generation execution does not cover all 384 tasks")
    if _required_int(coverage, "input_token_slots") != (
        PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_INPUT_TOKEN_SLOTS
    ):
        raise ValueError("generation execution token-slot accounting is invalid")
    if coverage.get("no_duplicate_tasks") is not True:
        raise ValueError("generation execution must prove no duplicate tasks")
    accounting = _required_mapping(record, "accounting")
    execution_mode = _required_string(record, "execution_mode")
    worker_count = _required_int(accounting, "worker_count")
    _validated_worker_count(worker_count)
    wall_seconds = _required_positive_number(accounting, "end_to_end_wall_seconds")
    charged_seconds = _required_positive_number(accounting, "charged_gpu_seconds")
    cost_model = accounting.get("cost_model")
    if cost_model == "persistent_worker_count_times_end_to_end_wall_time":
        if execution_mode != PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_LOCAL_TEST:
            raise ValueError("local test accounting requires local-test execution mode")
        if not math.isclose(
            charged_seconds,
            wall_seconds * worker_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("generation execution charged GPU seconds are invalid")
    elif cost_model == "sum_independent_one_gpu_worker_terminal_lifecycles":
        if execution_mode != PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED:
            raise ValueError(
                "distributed producer accounting requires distributed execution mode"
            )
        workers = _mapping_sequence(record.get("workers"), field_name="workers")
        if worker_count != PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS:
            raise ValueError("distributed execution requires exactly sixteen workers")
        if len(workers) != worker_count:
            raise ValueError("distributed worker accounting coverage is invalid")
        if tuple(_required_int(item, "worker_index") for item in workers) != tuple(
            range(PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS)
        ):
            raise ValueError("distributed worker indices must cover 0..15 exactly")
        ledger_reconciliation = _required_mapping(record, "ledger_reconciliation")
        _require_exact_mapping_keys(
            ledger_reconciliation,
            {
                "attempts",
                "attempts_sha256",
                "cap_gpu_hours",
                "cloud_identity_closure_sha256",
                "ledger_id",
            },
            label="distributed ledger reconciliation",
        )
        if ledger_reconciliation.get("cap_gpu_hours") != (
            MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
        ):
            raise ValueError("distributed execution ledger cap is invalid")
        ledger_id = _required_string(ledger_reconciliation, "ledger_id")
        ledger_attempts = _mapping_sequence(
            ledger_reconciliation.get("attempts"),
            field_name="ledger_reconciliation.attempts",
        )
        if len(ledger_attempts) != PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS:
            raise ValueError("distributed execution ledger coverage is incomplete")
        if ledger_reconciliation.get("attempts_sha256") != _canonical_sha256(
            ledger_attempts
        ):
            raise ValueError("distributed execution ledger closure is invalid")
        hardware_summary = _required_mapping(record, "generator_hardware")
        if hardware_summary.get("hardware_target") != (
            PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET
        ) or hardware_summary.get("node_type_id") != (
            PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
        ):
            raise ValueError("distributed execution hardware target is invalid")
        qualification_closure = _required_string(
            hardware_summary,
            "qualification_closed_record_sha256",
        )
        _require_sha256(
            qualification_closure,
            field_name="qualification_closed_record_sha256",
        )
        observed_hardware_hashes: set[str] = set()
        qualification_records: list[Mapping[str, Any]] = []
        expected_result_paths: set[Path] = set()
        expected_attestation_paths: set[Path] = set()
        parent_run_ids: set[str] = set()
        task_run_ids: set[str] = set()
        cluster_ids: set[str] = set()
        for index, worker_summary in enumerate(workers):
            ledger_attempt = ledger_attempts[index]
            _require_exact_mapping_keys(
                ledger_attempt,
                {
                    "actual_gpu_duration_seconds",
                    "attestation_closed_record_sha256",
                    "attestation_file_sha256",
                    "attestation_relative_path",
                    "attempt_id",
                    "attempt_number",
                    "cluster_id",
                    "control_plane_status_sha256",
                    "parent_run_id",
                    "repair_count",
                    "reserved_gpu_hours",
                    "submit_payload_sha256",
                    "submit_response_sha256",
                    "task_run_id",
                    "terminal_state",
                    "verification_source",
                    "worker_index",
                    "worker_result_closed_record_sha256",
                    "worker_result_file_sha256",
                    "workload_id",
                },
                label="distributed ledger attempt",
            )
            expected_workload_id = f"publication-latency-handoff-worker-{index:02d}"
            if (
                ledger_attempt.get("worker_index") != index
                or ledger_attempt.get("workload_id") != expected_workload_id
                or ledger_attempt.get("terminal_state") != "succeeded"
                or ledger_attempt.get("reserved_gpu_hours") != 5.0
                or ledger_attempt.get("attempt_number") != 0
                or ledger_attempt.get("repair_count") != 0
                or ledger_attempt.get("verification_source")
                != "direct_databricks_runs_get"
            ):
                raise ValueError("distributed execution ledger attempt is invalid")
            _required_string(ledger_attempt, "attempt_id")
            _require_sha256(
                ledger_attempt.get("submit_payload_sha256"),
                field_name="ledger submit_payload_sha256",
            )
            for field_name in (
                "submit_response_sha256",
                "control_plane_status_sha256",
                "attestation_file_sha256",
                "attestation_closed_record_sha256",
                "worker_result_file_sha256",
                "worker_result_closed_record_sha256",
            ):
                _require_sha256(
                    ledger_attempt.get(field_name),
                    field_name=f"ledger {field_name}",
                )
            parent_run_id = _databricks_cloud_id(
                ledger_attempt.get("parent_run_id"),
                field_name="ledger parent_run_id",
            )
            task_run_id = _databricks_cloud_id(
                ledger_attempt.get("task_run_id"),
                field_name="ledger task_run_id",
            )
            cluster_id = _databricks_cloud_id(
                ledger_attempt.get("cluster_id"),
                field_name="ledger cluster_id",
            )
            if task_run_id == parent_run_id:
                raise ValueError("distributed parent/task run IDs are not distinct")
            parent_run_ids.add(parent_run_id)
            task_run_ids.add(task_run_id)
            cluster_ids.add(cluster_id)
            expected_attestation_relative = (
                f"{PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY}/"
                f"worker-{index:02d}.json"
            )
            if ledger_attempt.get("attestation_relative_path") != (
                expected_attestation_relative
            ):
                raise ValueError("distributed cloud attestation path is invalid")
            attestation_path = _confined_relative_path(
                root,
                expected_attestation_relative,
                field_name="distributed cloud attestation",
            )
            expected_attestation_paths.add(attestation_path)
            attestation = read_publication_latency_handoff_databricks_attestation(
                PublicationLatencyHandoffDatabricksAttestationBinding(
                    worker_index=index,
                    path=attestation_path,
                    file_sha256=_required_string(
                        ledger_attempt,
                        "attestation_file_sha256",
                    ),
                    closed_record_sha256=_required_string(
                        ledger_attempt,
                        "attestation_closed_record_sha256",
                    ),
                ),
                durable_output_root=root,
            )
            attested_attempt = _required_mapping(attestation, "attempt")
            attested_cloud = _required_mapping(attestation, "cloud_execution")
            attested_worker_result = _required_mapping(attestation, "worker_result")
            if (
                attested_attempt.get("attempt_id") != ledger_attempt.get("attempt_id")
                or attested_attempt.get("ledger_id") != ledger_id
                or attested_attempt.get("worker_index") != index
                or attested_attempt.get("workload_id") != expected_workload_id
                or attested_attempt.get("reserved_gpu_hours") != 5.0
                or attested_attempt.get("submit_payload_sha256")
                != ledger_attempt.get("submit_payload_sha256")
                or attested_attempt.get("submit_response_sha256")
                != ledger_attempt.get("submit_response_sha256")
                or attested_cloud.get("parent_run_id") != parent_run_id
                or attested_cloud.get("task_run_id") != task_run_id
                or attested_cloud.get("cluster_id") != cluster_id
                or attested_cloud.get("attempt_number") != 0
                or attested_cloud.get("repair_count") != 0
                or attested_cloud.get("terminal_state") != "succeeded"
                or attested_cloud.get("actual_gpu_duration_seconds")
                != ledger_attempt.get("actual_gpu_duration_seconds")
                or attested_cloud.get("control_plane_status_sha256")
                != ledger_attempt.get("control_plane_status_sha256")
                or attested_worker_result.get("file_sha256")
                != ledger_attempt.get("worker_result_file_sha256")
                or attested_worker_result.get("closed_record_sha256")
                != ledger_attempt.get("worker_result_closed_record_sha256")
            ):
                raise ValueError(
                    "distributed cloud attestation does not join ledger closure"
                )
            expected_relative = f"worker-results/worker-{index:02d}.json"
            if worker_summary.get("result_relative_path") != expected_relative:
                raise ValueError("distributed worker result path is invalid")
            result_path = _confined_relative_path(
                root,
                expected_relative,
                field_name="distributed worker result",
            )
            expected_result_paths.add(result_path)
            worker_result = _read_worker_result(result_path)
            if worker_result.get("worker_index") != index:
                raise ValueError("distributed worker result identity is invalid")
            if worker_result.get("closed_record_sha256") != worker_summary.get(
                "result_closed_record_sha256"
            ):
                raise ValueError("distributed worker result closure is invalid")
            if worker_result.get("closed_record_sha256") != ledger_attempt.get(
                "worker_result_closed_record_sha256"
            ) or _file_sha256(result_path) != ledger_attempt.get(
                "worker_result_file_sha256"
            ):
                raise ValueError(
                    "distributed worker result does not join cloud attestation"
                )
            if worker_result.get("plan_closed_record_sha256") != record.get(
                "plan_closed_record_sha256"
            ) or worker_result.get("input_bundle_sha256") != record.get(
                "input_bundle_sha256"
            ):
                raise ValueError("distributed worker source binding is invalid")
            if worker_result.get("execution_contract") != record.get(
                "execution_contract"
            ):
                raise ValueError("distributed worker execution contract is invalid")
            worker_accounting = _required_mapping(worker_result, "accounting")
            if worker_summary.get("producer_metered_seconds") != (
                worker_accounting.get("producer_metered_seconds")
            ):
                raise ValueError("distributed worker metering binding is invalid")
            producer_metered_seconds = _required_positive_number(
                worker_accounting,
                "producer_metered_seconds",
            )
            billed_seconds = _required_positive_number(
                worker_summary,
                "charged_gpu_seconds",
            )
            if ledger_attempt.get("actual_gpu_duration_seconds") != billed_seconds:
                raise ValueError("distributed worker billing differs from ledger")
            if billed_seconds < producer_metered_seconds or billed_seconds > (
                PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS
            ):
                raise ValueError("distributed worker billed duration is invalid")
            worker_hardware = _required_mapping(worker_result, "generator_hardware")
            qualification = _required_mapping(worker_hardware, "qualification")
            _validate_hardware_qualification_record(qualification)
            qualification_records.append(qualification)
            if qualification.get("evidence_closed_record_sha256") != (
                qualification_closure
            ):
                raise ValueError("distributed workers used different qualifications")
            observed_hardware = _required_mapping(worker_hardware, "observed")
            _validate_observed_l40s_hardware(observed_hardware)
            observed_hardware_hashes.add(_canonical_sha256(observed_hardware))
        expected_identity_count = PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS
        if not (
            len(parent_run_ids) == expected_identity_count
            and len(task_run_ids) == expected_identity_count
            and len(cluster_ids) == expected_identity_count
        ) or parent_run_ids.intersection(task_run_ids):
            raise ValueError(
                "distributed workers require unique parent-run, task-run, and "
                "cluster IDs"
            )
        expected_identity_closure = _canonical_sha256(
            {
                "cluster_ids": sorted(cluster_ids),
                "parent_run_ids": sorted(parent_run_ids),
                "task_run_ids": sorted(task_run_ids),
            }
        )
        if ledger_reconciliation.get("cloud_identity_closure_sha256") != (
            expected_identity_closure
        ):
            raise ValueError("distributed cloud identity closure is invalid")
        actual_attestation_paths = {
            path
            for path in (
                root / PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY
            ).rglob("*")
            if path.is_file()
        }
        if actual_attestation_paths != expected_attestation_paths:
            raise ValueError("distributed cloud-attestation tree is incomplete")
        actual_result_paths = {
            path for path in (root / "worker-results").rglob("*") if path.is_file()
        }
        if actual_result_paths != expected_result_paths:
            raise ValueError("distributed worker-results tree is incomplete")
        if len({_canonical_sha256(item) for item in qualification_records}) != 1:
            raise ValueError("distributed workers used different qualification records")
        _verify_bound_hardware_qualification_file(qualification_records[0])
        recorded_hardware_hashes = hardware_summary.get(
            "observed_hardware_identity_sha256"
        )
        if not isinstance(recorded_hardware_hashes, list) or (
            recorded_hardware_hashes != sorted(observed_hardware_hashes)
        ):
            raise ValueError("distributed observed-hardware closure is invalid")
        worker_seconds = [
            _required_positive_number(worker, "charged_gpu_seconds")
            for worker in workers
        ]
        if not math.isclose(
            charged_seconds,
            sum(worker_seconds),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("distributed GPU seconds do not sum worker actuals")
        if not math.isclose(wall_seconds, max(worker_seconds), rel_tol=1e-12):
            raise ValueError("distributed wall time must equal the slowest worker")
        if accounting.get("coordinator_gpu_hours") != 0.0:
            raise ValueError("CPU coordinator must not be charged as GPU time")
        if accounting.get("payload_copy_count_during_closure") != 0:
            raise ValueError("coordinator must close worker payloads without copying")
        if (
            accounting.get("includes_bootstrap_generation_hash_and_durable_write")
            is not True
        ):
            raise ValueError("distributed accounting scope is incomplete")
    else:
        raise ValueError("generation execution cost model is invalid")
    tokens = _required_int(coverage, "cache_prefix_generation_tokens")
    observed = _required_positive_number(
        accounting,
        "end_to_end_cache_prefix_tokens_per_gpu_second",
    )
    if not math.isclose(observed, tokens / charged_seconds, rel_tol=1e-12):
        raise ValueError("generation execution throughput accounting is invalid")
    charged_hours = _required_positive_number(accounting, "charged_gpu_hours")
    if not math.isclose(charged_hours, charged_seconds / 3600.0, rel_tol=1e-12):
        raise ValueError("generation execution GPU-hour accounting is invalid")
    input_slots = _required_int(coverage, "input_token_slots")
    input_slot_throughput = _required_positive_number(
        accounting,
        "end_to_end_input_token_slots_per_gpu_second",
    )
    if not math.isclose(
        input_slot_throughput,
        input_slots / charged_seconds,
        rel_tol=1e-12,
    ):
        raise ValueError("generation execution input-slot throughput is invalid")
    if cost_model == "persistent_worker_count_times_end_to_end_wall_time" and (
        accounting.get("includes_bundle_closure_and_durable_sync") is not True
    ):
        raise ValueError("generation accounting must include closure and durable sync")
    expected_gate = observed >= PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
    if accounting.get("full_launch_throughput_gate_passed") is not expected_gate:
        raise ValueError("generation execution throughput gate is inconsistent")

    bundles = _mapping_sequence(record.get("bundles"), field_name="bundles")
    if len(bundles) != len(PUBLICATION_CAMPAIGN_CONTEXT_TOKENS):
        raise ValueError("generation execution must contain three context bundles")
    contexts: list[int] = []
    identities_by_context: dict[int, set[tuple[str, str]]] = {}
    reconstructed_task_ids: list[str] = []
    for item in bundles:
        context_tokens = _required_int(item, "context_tokens")
        contexts.append(context_tokens)
        manifest_path = _confined_relative_path(
            root,
            _required_string(item, "manifest_relative_path"),
            field_name="manifest_relative_path",
        )
        source_root = _confined_relative_path(
            root,
            _required_string(item, "source_root_relative_path"),
            field_name="source_root_relative_path",
        )
        manifest = read_publication_latency_handoff_bundle(manifest_path)
        if manifest.get("context_tokens") != context_tokens:
            raise ValueError("serving bundle context does not match execution record")
        if manifest.get("portable_bundle_sha256") != item.get("portable_bundle_sha256"):
            raise ValueError("serving bundle portable digest does not match")
        if manifest.get("closed_record_sha256") != item.get("closed_record_sha256"):
            raise ValueError("serving bundle closed digest does not match")
        validate_publication_latency_handoff_bundle(
            manifest,
            bundle_root=source_root,
        )
        _validate_publication_manifest_contract(manifest)
        expected_source_relative = (
            f"bundles/{context_tokens}-{item['portable_bundle_sha256']}"
        )
        expected_manifest_relative = (
            f"manifests/{context_tokens}-{item['portable_bundle_sha256']}.json"
        )
        if item.get("source_root_relative_path") != expected_source_relative:
            raise ValueError("serving source root is not content-addressed")
        if item.get("manifest_relative_path") != expected_manifest_relative:
            raise ValueError("serving manifest path is not content-addressed")
        identities = _manifest_identity_set(manifest)
        identities_by_context[context_tokens] = identities
        for dataset, example_id in identities:
            task_identity = {
                "context_tokens": context_tokens,
                "dataset": dataset,
                "example_id": example_id,
            }
            reconstructed_task_ids.append(
                f"{context_tokens}-{dataset}-{_canonical_sha256(task_identity)[:20]}"
            )
    if tuple(contexts) != tuple(PUBLICATION_CAMPAIGN_CONTEXT_TOKENS):
        raise ValueError("serving bundle contexts are incomplete or out of order")
    if record.get("bundles_sha256") != _canonical_sha256(bundles):
        raise ValueError("generation execution bundles_sha256 is invalid")
    identity_sets = tuple(
        identities_by_context[context]
        for context in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS
    )
    expected_identity_count = (
        len(SUPPORTED_V1_DATASETS) * PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
    )
    if any(len(identities) != expected_identity_count for identities in identity_sets):
        raise ValueError("generation execution context identity coverage is incomplete")
    if any(identities != identity_sets[0] for identities in identity_sets[1:]):
        raise ValueError("generation execution identities differ across contexts")
    if len(set(reconstructed_task_ids)) != PUBLICATION_LATENCY_HANDOFF_TASK_COUNT:
        raise ValueError("generation execution task identities collide")
    if coverage.get("task_ids_sha256") != _canonical_sha256(
        sorted(reconstructed_task_ids)
    ):
        raise ValueError("generation execution task identity closure is invalid")
    durable_bytes = _required_int(accounting, "durable_byte_count")
    expected_durable_bytes = _tree_byte_count(root / "bundles") + _tree_byte_count(
        root / "manifests"
    )
    if cost_model == "sum_independent_one_gpu_worker_terminal_lifecycles":
        expected_durable_bytes += (
            _tree_byte_count(root / "worker-records")
            + _tree_byte_count(root / "worker-results")
            + _tree_byte_count(
                root / PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY
            )
        )
    if durable_bytes != expected_durable_bytes:
        raise ValueError("generation execution durable-byte accounting is invalid")
    serving = _required_mapping(record, "serving_reuse")
    if serving.get("regenerate_inside_timed_serving_jobs") is not False:
        raise ValueError("timed serving jobs must reuse generated handoffs")
    serving_bundles = _mapping_sequence(
        serving.get("context_bundles"),
        field_name="serving_reuse.context_bundles",
    )
    if serving_bundles != bundles:
        raise ValueError("serving reuse bindings diverge from closed bundles")


def authorize_publication_latency_handoff_serving(
    workspace: DatabricksWorkspaceConfig,
    result: PublicationLatencyHandoffGenerationResult,
    *,
    ledger_path: str | Path,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    submission_authorization: PublicationLatencyHandoffSubmissionAuthorization,
    attempt_ids_by_worker: Mapping[int, str],
    attestations_by_worker: Mapping[
        int,
        PublicationLatencyHandoffDatabricksAttestationBinding,
    ],
) -> PublicationLatencyHandoffServingAuthorization:
    """Replay all 16 runs directly and issue ephemeral serving authority."""

    if not isinstance(workspace, DatabricksWorkspaceConfig):
        raise TypeError("workspace has the wrong type")
    if not isinstance(result, PublicationLatencyHandoffGenerationResult):
        raise TypeError("result must be a PublicationLatencyHandoffGenerationResult")
    authenticated = read_publication_latency_handoff_generation_result(result.root)
    if (
        authenticated.execution_record_path.resolve()
        != result.execution_record_path.resolve()
        or dict(authenticated.record) != dict(result.record)
        or authenticated.record.get("execution_mode")
        != PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED
    ):
        raise ValueError("Q8 serving result is not the authenticated distributed run")
    qualification_ledger = _require_matching_qualification_ledger(
        ledger_path,
        qualification_launch_authorization,
    )
    hardware = _required_mapping(authenticated.record, "generator_hardware")
    if hardware.get("qualification_closed_record_sha256") != (
        qualification_launch_authorization.evidence_closed_record_sha256
    ):
        raise ValueError("Q8 result differs from GPU qualification authority")
    reconciliation = _publication_latency_handoff_ledger_reconciliation(
        ledger_path,
        attempt_ids_by_worker=attempt_ids_by_worker,
        durable_output_root=authenticated.root,
        attestations_by_worker=attestations_by_worker,
    )
    if dict(_required_mapping(authenticated.record, "ledger_reconciliation")) != (
        reconciliation
    ):
        raise ValueError("Q8 execution record differs from live ledger replay")
    reconciliation_attempts = _mapping_sequence(
        reconciliation.get("attempts"),
        field_name="ledger reconciliation attempts",
    )
    expected_attempt_ids = tuple(
        _required_string(item, "attempt_id") for item in reconciliation_attempts
    )
    expected_payload_digests = tuple(
        _required_string(item, "submit_payload_sha256")
        for item in reconciliation_attempts
    )
    batch_reservation_authorization = (
        require_publication_latency_handoff_submission_authorization(
            submission_authorization
        )
    )
    batch_prefix = require_databricks_batch_reservation_authorization(
        batch_reservation_authorization,
        expected_predecessor_prefix=qualification_launch_authorization.ledger_prefix,
        expected_attempt_ids=expected_attempt_ids,
        expected_submit_payload_sha256s=expected_payload_digests,
    )
    require_databricks_ledger_prefix(qualification_ledger, batch_prefix)
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if ledger.ledger_id != qualification_ledger.ledger_id:
        raise ValueError("Q8 ledger identity changed during serving authorization")
    receipts = {item.attempt_id: item for item in ledger.submission_receipts}
    direct_status_hashes: list[str] = []
    for worker_index in range(PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS):
        attempt_id = attempt_ids_by_worker[worker_index]
        receipt = receipts.get(attempt_id)
        if receipt is None:
            raise ValueError("Q8 live replay lacks a submission receipt")
        terminal = get_databricks_run(workspace, receipt.run_id)
        terminal_snapshot, canonical_terminal = (
            canonical_databricks_submit_payload_snapshot(terminal)
        )
        attestation = read_publication_latency_handoff_databricks_attestation(
            attestations_by_worker[worker_index],
            durable_output_root=authenticated.root,
        )
        cloud = _required_mapping(attestation, "cloud_execution")
        status_sha256 = sha256(canonical_terminal).hexdigest()
        if _databricks_cloud_id(
            terminal_snapshot.get("run_id"),
            field_name="live replay parent run_id",
        ) != receipt.run_id or status_sha256 != cloud.get(
            "control_plane_status_sha256"
        ):
            raise ValueError("Q8 live runs/get response differs from durable evidence")
        direct_status_hashes.append(status_sha256)
    final_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if databricks_ledger_path_sha256(ledger_path) != (
        qualification_launch_authorization.ledger_path_sha256
    ):
        raise ValueError("Q8 ledger path changed during serving authorization")
    require_databricks_ledger_prefix(
        final_ledger, qualification_launch_authorization.ledger_prefix
    )
    require_databricks_ledger_prefix(final_ledger, batch_prefix)
    final_reconciliation = _publication_latency_handoff_ledger_reconciliation(
        ledger_path,
        attempt_ids_by_worker=attempt_ids_by_worker,
        durable_output_root=authenticated.root,
        attestations_by_worker=attestations_by_worker,
        _ledger=final_ledger,
    )
    if final_reconciliation != reconciliation:
        raise ValueError("Q8 ledger changed during causal serving replay")
    ledger_prefix = require_databricks_batch_terminal_closure(
        final_ledger,
        batch_reservation_authorization,
        require_complete_current_prefix=True,
    )
    causal_closure = {
        "direct_control_plane_status_sha256": direct_status_hashes,
        "ledger_reconciliation": reconciliation,
        "predecessor_prefix": (
            qualification_launch_authorization.ledger_prefix.to_record()
        ),
        "producer_batch_prefix": batch_prefix.to_record(),
        "ledger_prefix": ledger_prefix.to_record(),
    }
    return PublicationLatencyHandoffServingAuthorization(
        result=authenticated,
        ledger_id=ledger.ledger_id,
        ledger_path_sha256=qualification_launch_authorization.ledger_path_sha256,
        predecessor_prefix=qualification_launch_authorization.ledger_prefix,
        producer_batch_prefix=batch_prefix,
        ledger_prefix=ledger_prefix,
        causal_closure_sha256=_canonical_sha256(causal_closure),
        _issuer=_SERVING_AUTHORIZATION_ISSUER,
    )


def _resolve_authorized_publication_latency_handoff_result(
    authorization: object,
) -> PublicationLatencyHandoffGenerationResult:
    if not isinstance(authorization, PublicationLatencyHandoffServingAuthorization):
        raise TypeError(
            "Q8 serving requires PublicationLatencyHandoffServingAuthorization"
        )
    result = read_publication_latency_handoff_generation_result(
        authorization.result_root
    )
    reconciliation = _required_mapping(result.record, "ledger_reconciliation")
    direct_hashes = [
        _required_string(item, "control_plane_status_sha256")
        for item in _mapping_sequence(
            reconciliation.get("attempts"),
            field_name="ledger reconciliation attempts",
        )
    ]
    expected_causal_closure = _canonical_sha256(
        {
            "direct_control_plane_status_sha256": direct_hashes,
            "ledger_reconciliation": dict(reconciliation),
            "predecessor_prefix": authorization.predecessor_prefix.to_record(),
            "producer_batch_prefix": authorization.producer_batch_prefix.to_record(),
            "ledger_prefix": authorization.ledger_prefix.to_record(),
        }
    )
    if (
        _file_sha256(result.execution_record_path)
        != authorization.execution_file_sha256
        or result.record.get("closed_record_sha256")
        != authorization.execution_closed_record_sha256
        or reconciliation.get("ledger_id") != authorization.ledger_id
        or authorization.ledger_prefix.ledger_id != authorization.ledger_id
        or authorization.producer_batch_prefix.ledger_id != authorization.ledger_id
        or authorization.predecessor_prefix.ledger_id != authorization.ledger_id
        or expected_causal_closure != authorization.causal_closure_sha256
    ):
        raise ValueError("Q8 serving authorization binding drift")
    return result


def require_publication_latency_handoff_serving_authorization(
    authorization: object,
    *,
    expected_execution_file_sha256: str,
    expected_input_bundle_sha256: str,
    expected_qualification_closed_record_sha256: str,
) -> PublicationLatencyHandoffGenerationResult:
    """Authenticate one Q8 serving capability against final campaign pins."""

    result = _resolve_authorized_publication_latency_handoff_result(authorization)
    hardware = _required_mapping(result.record, "generator_hardware")
    if (
        not hmac.compare_digest(
            _file_sha256(result.execution_record_path),
            _require_sha256(
                expected_execution_file_sha256,
                field_name="expected_execution_file_sha256",
            ),
        )
        or result.record.get("input_bundle_sha256")
        != _require_sha256(
            expected_input_bundle_sha256,
            field_name="expected_input_bundle_sha256",
        )
        or hardware.get("qualification_closed_record_sha256")
        != _require_sha256(
            expected_qualification_closed_record_sha256,
            field_name="expected_qualification_closed_record_sha256",
        )
    ):
        raise ValueError("Q8 serving authority differs from final artifact pins")
    return result


def resolve_publication_latency_serving_handoff_bundle(
    authorization: object,
    *,
    context_tokens: int,
) -> PublicationLatencyServingHandoffBundle:
    """Return staging bindings only from an ephemeral serving capability."""

    result = _resolve_authorized_publication_latency_handoff_result(authorization)
    return resolve_publication_latency_worker_handoff_bundle(
        result,
        context_tokens=context_tokens,
    )


def resolve_publication_latency_worker_handoff_bundle(
    authenticated_result: PublicationLatencyHandoffGenerationResult,
    *,
    context_tokens: int,
) -> PublicationLatencyServingHandoffBundle:
    """Resolve a bundle inside an already authorized, launched worker.

    This helper is deliberately nonauthorizing.  It exists because the
    in-memory serving capability cannot be serialized into a Databricks worker
    payload.  Production launch construction, reservation, and submission must
    continue to require ``PublicationLatencyHandoffServingAuthorization``; the
    worker may call this helper only after checking the exact execution-file
    and closed-record hashes embedded by that authorized launch plan.
    """

    if not isinstance(
        authenticated_result,
        PublicationLatencyHandoffGenerationResult,
    ):
        raise TypeError(
            "worker handoff resolution requires an authenticated "
            "PublicationLatencyHandoffGenerationResult"
        )
    if context_tokens not in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
        raise ValueError("context_tokens is outside the publication campaign")
    result = read_publication_latency_handoff_generation_result(
        authenticated_result.root
    )
    if (
        result.execution_record_path.resolve()
        != authenticated_result.execution_record_path.resolve()
        or dict(result.record) != dict(authenticated_result.record)
        or result.record.get("execution_mode")
        != PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED
    ):
        raise ValueError(
            "worker handoff result is not the authenticated distributed execution"
        )
    bundles = _mapping_sequence(result.record.get("bundles"), field_name="bundles")
    binding = next(
        item for item in bundles if item.get("context_tokens") == context_tokens
    )
    manifest_path = _confined_relative_path(
        result.root,
        _required_string(binding, "manifest_relative_path"),
        field_name="manifest_relative_path",
    )
    source_root = _confined_relative_path(
        result.root,
        _required_string(binding, "source_root_relative_path"),
        field_name="source_root_relative_path",
    )
    manifest = read_publication_latency_handoff_bundle(manifest_path)
    return PublicationLatencyServingHandoffBundle(
        context_tokens=context_tokens,
        manifest_path=manifest_path,
        source_root=source_root,
        portable_bundle_sha256=_required_string(
            binding,
            "portable_bundle_sha256",
        ),
        manifest=manifest,
    )


def require_publication_latency_full_launch_ready(
    authorization: PublicationLatencyHandoffServingAuthorization,
    *,
    other_terminal_gpu_hours: float,
    current_active_reserved_gpu_hours: float,
    proposed_full_launch_reserved_gpu_hours: float,
) -> dict[str, Any]:
    """Enforce throughput and campaign headroom before any full launch.

    ``other_terminal_gpu_hours`` deliberately excludes this generation run; its
    measured charged hours are added here exactly once.
    """

    authenticated = _resolve_authorized_publication_latency_handoff_result(
        authorization
    )
    accounting = _required_mapping(authenticated.record, "accounting")
    if (
        authenticated.record.get("execution_mode")
        != PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED
    ):
        raise ValueError(
            "full launch requires the qualified distributed 16x L40S generation result"
        )
    observed = _required_positive_number(
        accounting,
        "end_to_end_cache_prefix_tokens_per_gpu_second",
    )
    generation_hours = _required_positive_number(accounting, "charged_gpu_hours")
    return publication_campaign_full_launch_budget_projection(
        latency_handoff_generation_tokens_per_second=observed,
        latency_handoff_generation_gpu_hours=generation_hours,
        other_terminal_gpu_hours=other_terminal_gpu_hours,
        current_active_reserved_gpu_hours=current_active_reserved_gpu_hours,
        proposed_full_launch_reserved_gpu_hours=(
            proposed_full_launch_reserved_gpu_hours
        ),
    )


def _generation_items(
    prepared: PreparedMainLatencyInputs,
    *,
    tokenizer: MainLatencyTokenizer,
) -> tuple[list[dict[str, Any]], str]:
    if MAIN_LATENCY_EXAMPLES_PER_DATASET != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET:
        raise ValueError("main-latency and campaign example counts diverged")
    expected_files = {
        (dataset, context_tokens)
        for context_tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS
        for dataset in SUPPORTED_V1_DATASETS
    }
    actual_files = {(item.dataset, item.input_tokens_target) for item in prepared.files}
    if actual_files != expected_files or len(prepared.files) != len(expected_files):
        raise ValueError("prepared bundle does not contain all 12 latency files")
    identities_by_context: dict[int, set[tuple[str, str]]] = {}
    items: list[dict[str, Any]] = []
    for artifact in sorted(
        prepared.files,
        key=lambda item: (item.input_tokens_target, item.dataset),
    ):
        records = _canonical_jsonl_records(artifact.jsonl_path)
        if len(records) != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET:
            raise ValueError("each prepared latency file must contain exactly 32 rows")
        relative_path = _relative_input_path(prepared.output_dir, artifact.jsonl_path)
        identities = identities_by_context.setdefault(
            artifact.input_tokens_target,
            set(),
        )
        for row_index, row in enumerate(records):
            dataset = _required_string(row, "dataset")
            example_id = _required_string(row, "example_id")
            if dataset != artifact.dataset:
                raise ValueError("prepared row dataset does not match its file")
            identity = (dataset, example_id)
            if identity in identities:
                raise ValueError("prepared context contains a duplicate identity")
            identities.add(identity)
            segment_token_contracts = _cache_prefix_segment_token_contracts(
                row,
                tokenizer=tokenizer,
            )
            cache_prefix_tokens = sum(
                cast(int, item["token_count"]) for item in segment_token_contracts
            )
            task_identity = {
                "context_tokens": artifact.input_tokens_target,
                "dataset": dataset,
                "example_id": example_id,
            }
            task_id = (
                f"{artifact.input_tokens_target}-{dataset}-"
                f"{_canonical_sha256(task_identity)[:20]}"
            )
            item: dict[str, Any] = {
                "cache_prefix_tokens": cache_prefix_tokens,
                "context_tokens": artifact.input_tokens_target,
                "dataset": dataset,
                "example_id": example_id,
                "input_token_slots": artifact.input_tokens_target,
                "prepared_jsonl_relative_path": relative_path,
                "prepared_record_sha256": _canonical_sha256(row),
                "row_index": row_index,
                "segment_token_contracts": segment_token_contracts,
                "segment_token_contracts_sha256": _canonical_sha256(
                    segment_token_contracts
                ),
                "task_id": task_id,
            }
            item["assignment_sha256"] = _canonical_sha256(
                {
                    "domain": _PLAN_ORDER_DOMAIN,
                    "input_bundle_sha256": prepared.bundle_sha256,
                    "item": item,
                }
            )
            items.append(item)
    expected_identity_count = (
        len(SUPPORTED_V1_DATASETS) * PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
    )
    context_sets = tuple(
        identities_by_context[context]
        for context in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS
    )
    if any(len(identities) != expected_identity_count for identities in context_sets):
        raise ValueError("prepared latency identity coverage is incomplete")
    if any(identities != context_sets[0] for identities in context_sets[1:]):
        raise ValueError("prepared latency identities are not reused across contexts")
    if len(items) != PUBLICATION_LATENCY_HANDOFF_TASK_COUNT:
        raise ValueError("publication latency generation must contain 384 tasks")
    task_ids = [cast(str, item["task_id"]) for item in items]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("publication latency generation task IDs collide")
    identity_records = [
        {"dataset": dataset, "example_id": example_id}
        for dataset, example_id in sorted(context_sets[0])
    ]
    return items, _canonical_sha256(identity_records)


def _lpt_assignments(
    items: Sequence[dict[str, Any]],
    *,
    worker_count: int,
) -> tuple[list[dict[str, Any]], ...]:
    assignments: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for item in sorted(
        items,
        key=lambda value: (
            -cast(int, value["cache_prefix_tokens"]),
            cast(str, value["assignment_sha256"]),
        ),
    ):
        worker_index = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[worker_index].append(item)
        loads[worker_index] += cast(int, item["cache_prefix_tokens"])
    if any(not assignment for assignment in assignments):
        raise ValueError("token-balanced plan produced an empty worker")
    return tuple(assignments)


def _prepared_rows_by_task(
    prepared: PreparedMainLatencyInputs,
) -> dict[tuple[int, str, int], dict[str, Any]]:
    rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    for artifact in prepared.files:
        for row_index, row in enumerate(_canonical_jsonl_records(artifact.jsonl_path)):
            key = (artifact.input_tokens_target, artifact.dataset, row_index)
            if key in rows:
                raise ValueError("prepared latency row key collision")
            rows[key] = row
    return rows


def _execute_worker(
    worker: Mapping[str, Any],
    *,
    rows_by_task: Mapping[tuple[int, str, int], dict[str, Any]],
    work_root: Path,
    pending_root: Path,
    worker_factory: PublicationLatencyGeneratorFactory,
    config: PublicationLatencyHandoffExecutionConfig,
    clock: Callable[[], float],
    enforce_production_generator_identity: bool,
) -> _WorkerResult:
    worker_index = _required_int(worker, "worker_index")
    lifecycle_start = clock()
    initialization_start = lifecycle_start
    generator = worker_factory(worker_index)
    initialization_end = clock()
    _validate_publication_generator(
        generator,
        config=config,
        enforce_production_identity=enforce_production_generator_identity,
    )
    batch_results: list[_WorkerBatchResult] = []
    try:
        items = _mapping_sequence(worker.get("items"), field_name="worker.items")
        for context_tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
            context_items = tuple(
                item for item in items if item.get("context_tokens") == context_tokens
            )
            if not context_items:
                continue
            batch_results.append(
                _execute_worker_context_batch(
                    worker_index=worker_index,
                    context_tokens=context_tokens,
                    items=context_items,
                    rows_by_task=rows_by_task,
                    work_root=work_root,
                    pending_root=pending_root,
                    generator=generator,
                    config=config,
                    clock=clock,
                )
            )
    finally:
        close = getattr(generator, "close", None)
        if callable(close):
            close()
    lifecycle_end = clock()
    return _WorkerResult(
        worker_index=worker_index,
        initialization_seconds=_nonnegative_duration(
            initialization_end - initialization_start
        ),
        lifecycle_seconds=_positive_duration(lifecycle_end - lifecycle_start),
        batches=tuple(batch_results),
    )


def _execute_worker_context_batch(
    *,
    worker_index: int,
    context_tokens: int,
    items: Sequence[Mapping[str, Any]],
    rows_by_task: Mapping[tuple[int, str, int], dict[str, Any]],
    work_root: Path,
    pending_root: Path,
    generator: KVChunkGenerator,
    config: PublicationLatencyHandoffExecutionConfig,
    clock: Callable[[], float],
) -> _WorkerBatchResult:
    worker_label = f"worker-{worker_index:02d}"
    batch_work = work_root / worker_label / str(context_tokens)
    batch_output = pending_root / str(context_tokens) / worker_label
    if batch_work.exists() or batch_output.exists():
        raise ValueError("worker batch output collision")
    batch_work.mkdir(parents=True)
    batch_output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    task_ids: list[str] = []
    for item in items:
        dataset = _required_string(item, "dataset")
        row_index = _required_int(item, "row_index")
        key = (context_tokens, dataset, row_index)
        try:
            row = rows_by_task[key]
        except KeyError as exc:
            raise ValueError(f"planned prepared row is missing: {key}") from exc
        if _canonical_sha256(row) != _required_string(
            item,
            "prepared_record_sha256",
        ):
            raise ValueError("planned prepared row digest changed before generation")
        if _required_string(row, "example_id") != _required_string(
            item,
            "example_id",
        ):
            raise ValueError("planned prepared row identity changed before generation")
        records.append(row)
        task_ids.append(_required_string(item, "task_id"))
    input_jsonl = batch_work / "input.jsonl"
    _write_canonical_jsonl_exclusive(records, input_jsonl)
    generation_start = clock()
    result = generate_benchmark_handoff_bundles(
        input_jsonl,
        output_dir=batch_output,
        generator=generator,
        layout=config.layout,
        backend="vllm",
        model_id=config.layout.model_id,
        lora_id=config.layout.lora_id,
        cache_method=CacheGenerationMethod.VANILLA_PREFILL,
        model_revision=config.model_revision,
        tokenizer_id=config.tokenizer_id,
        tokenizer_revision=config.tokenizer_revision,
        generator_family=config.generator_family,
        generator_version=config.generator_version,
        tensor_parallel_size=config.tensor_parallel_size,
        pipeline_parallel_size=config.pipeline_parallel_size,
        storage_layout=config.layout.storage_layout,
        segment_per_document=True,
        align_bytes=config.align_bytes,
    )
    shard_path = Path(local_path(result.shard_uri)).expanduser().resolve()
    _require_confined_path(batch_output.resolve(), shard_path, field_name="shard_uri")
    if not shard_path.is_file() or shard_path.is_symlink():
        raise ValueError("worker intermediate shard is missing or unsafe")
    shard_path.unlink()
    enriched = enrich_benchmark_records_with_handoffs(
        records,
        result.manifest,
        arm_id=PUBLICATION_LATENCY_HANDOFF_ARM_ID,
    )
    expected_keys = {
        (_required_string(item, "dataset"), _required_string(item, "example_id"))
        for item in items
    }
    observed_keys = {
        (_required_string(row, "dataset"), _required_string(row, "example_id"))
        for row in enriched
    }
    if observed_keys != expected_keys or len(enriched) != len(expected_keys):
        raise ValueError("worker generated incomplete or duplicate handoff coverage")
    expected_tokens = {
        (_required_string(item, "dataset"), _required_string(item, "example_id")): (
            _required_int(item, "cache_prefix_tokens")
        )
        for item in items
    }
    expected_segment_contracts = {
        (_required_string(item, "dataset"), _required_string(item, "example_id")): [
            dict(value)
            for value in _mapping_sequence(
                item.get("segment_token_contracts"),
                field_name="segment_token_contracts",
            )
        ]
        for item in items
    }
    for row in enriched:
        identity_key = (
            _required_string(row, "dataset"),
            _required_string(row, "example_id"),
        )
        if _handoff_total_tokens(row) != expected_tokens[identity_key]:
            raise ValueError("generated handoff token count differs from exact plan")
        if (
            _handoff_segment_token_contracts(row)
            != expected_segment_contracts[identity_key]
        ):
            raise ValueError(
                "generated handoff token identities differ from exact plan"
            )
    generation_end = clock()
    durable_start = generation_end
    _sync_tree(batch_output)
    _sync_directory(batch_output.parent)
    _sync_directory(pending_root)
    durable_end = clock()
    return _WorkerBatchResult(
        context_tokens=context_tokens,
        worker_index=worker_index,
        records=tuple(enriched),
        task_ids=tuple(task_ids),
        cache_prefix_tokens=sum(expected_tokens.values()),
        input_token_slots=sum(
            _required_int(item, "input_token_slots") for item in items
        ),
        generation_seconds=_positive_duration(generation_end - generation_start),
        durable_sync_seconds=_nonnegative_duration(durable_end - durable_start),
        durable_byte_count=_tree_byte_count(batch_output),
    )


def _write_context_dataset_files(
    context_root: Path,
    *,
    context_tokens: int,
    batches: Sequence[_WorkerBatchResult],
) -> dict[str, Path]:
    records = [
        row
        for batch in batches
        if batch.context_tokens == context_tokens
        for row in batch.records
    ]
    expected_count = (
        len(SUPPORTED_V1_DATASETS) * PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
    )
    keys = [
        (_required_string(row, "dataset"), _required_string(row, "example_id"))
        for row in records
    ]
    if len(records) != expected_count or len(set(keys)) != expected_count:
        raise ValueError("context handoff coverage is incomplete or duplicated")
    paths: dict[str, Path] = {}
    for dataset in SUPPORTED_V1_DATASETS:
        selected = sorted(
            (row for row in records if row.get("dataset") == dataset),
            key=lambda row: _required_string(row, "example_id"),
        )
        if len(selected) != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET:
            raise ValueError(f"context handoff coverage is incomplete for {dataset}")
        path = context_root / "datasets" / f"{dataset}.jsonl"
        _write_handoff_dataset_jsonl_exclusive(selected, path)
        paths[dataset] = path
    return paths


def _validate_publication_manifest_contract(manifest: Mapping[str, Any]) -> None:
    identity = _required_mapping(manifest, "identity")
    layout = _required_mapping(identity, "layout_identity")
    if layout.get("dtype") != PUBLICATION_LATENCY_HANDOFF_DTYPE:
        raise ValueError("closed publication handoff is not Q8 E5M2")
    if (
        layout.get("pre_rope") is not True
        or layout.get("key_position_encoding") != KVKeyPositionEncoding.PRE_ROPE.value
    ):
        raise ValueError("closed publication handoff is not pre-RoPE")
    if layout.get("storage_layout") != KVStorageLayout.SEPARATE_KEY_VALUE.value:
        raise ValueError("closed publication handoff does not use separate K/V")
    datasets = _mapping_sequence(manifest.get("datasets"), field_name="datasets")
    entries = [
        entry
        for dataset in datasets
        for entry in _mapping_sequence(dataset.get("entries"), field_name="entries")
    ]
    if not entries or any(
        entry.get("cache_method") != CacheGenerationMethod.VANILLA_PREFILL.value
        for entry in entries
    ):
        raise ValueError("closed publication handoff is not entirely Vanilla")


def _manifest_identity_set(
    manifest: Mapping[str, Any],
) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    datasets = _mapping_sequence(manifest.get("datasets"), field_name="datasets")
    if tuple(item.get("dataset") for item in datasets) != tuple(SUPPORTED_V1_DATASETS):
        raise ValueError("closed publication handoff dataset order is invalid")
    for dataset_record in datasets:
        dataset = _required_string(dataset_record, "dataset")
        if _required_int(dataset_record, "row_count") != (
            PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
        ):
            raise ValueError(
                "closed publication handoff dataset coverage is incomplete"
            )
        entries = _mapping_sequence(
            dataset_record.get("entries"),
            field_name="dataset.entries",
        )
        if len(entries) != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET:
            raise ValueError(
                "closed publication handoff transfer coverage is incomplete"
            )
        for entry in entries:
            if entry.get("transfer_scope") != (
                f"arm_kv_transfer_params/{PUBLICATION_LATENCY_HANDOFF_ARM_ID}"
            ):
                raise ValueError("closed publication handoff transfer scope is invalid")
            identity = (dataset, _required_string(entry, "example_id"))
            if identity in identities:
                raise ValueError(
                    "closed publication handoff contains duplicate identities"
                )
            identities.add(identity)
    return identities


def _handoff_total_tokens(row: Mapping[str, Any]) -> int:
    handoff = _read_generated_handoff(row)
    handle = _required_mapping(handoff, "handle")
    return _required_int(handle, "total_tokens")


def _handoff_segment_token_contracts(
    row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    handoff = _read_generated_handoff(row)
    handle = _required_mapping(handoff, "handle")
    segments = _mapping_sequence(handle.get("segments"), field_name="handle.segments")
    contracts: list[dict[str, Any]] = []
    for segment in segments:
        contract = TokenContract.from_record(
            _required_mapping(segment, "token_contract")
        )
        if contract.token_count != _required_int(segment, "token_count"):
            raise ValueError("generated segment token count differs from contract")
        contracts.append(
            {
                "chunk_id": _required_string(segment, "chunk_id"),
                "token_count": contract.token_count,
                "token_ids_digest": contract.token_ids_digest,
            }
        )
    if not contracts:
        raise ValueError("generated handoff contains no token segments")
    return contracts


def _read_generated_handoff(row: Mapping[str, Any]) -> dict[str, Any]:
    arms = _required_mapping(row, "arm_kv_transfer_params")
    params = _required_mapping(arms, PUBLICATION_LATENCY_HANDOFF_ARM_ID)
    handoff_path = Path(
        local_path(_required_string(params, DOCUMENT_KV_HANDOFF_JSON_PARAM))
    )
    payload_path = Path(
        local_path(_required_string(params, DOCUMENT_KV_PAYLOAD_URI_PARAM))
    )
    if not handoff_path.is_file() or handoff_path.is_symlink():
        raise ValueError("generated handoff JSON is missing or unsafe")
    if not payload_path.is_file() or payload_path.is_symlink():
        raise ValueError("generated handoff payload is missing or unsafe")
    return _json_object(handoff_path.read_bytes(), field_name="generated handoff")


def _cache_prefix_segment_token_contracts(
    row: Mapping[str, Any],
    *,
    tokenizer: MainLatencyTokenizer,
) -> list[dict[str, Any]]:
    from document_kv_cache._benchmark_datasets import _example_from_record

    example = _example_from_record(
        row,
        default_dataset=_required_string(row, "dataset"),
        record_index=1,
        require_dataset=True,
    )
    segments = benchmark_cache_prefix_segments(example)
    contracts: list[dict[str, Any]] = []
    for chunk_id, text in segments:
        values = tokenizer.encode(
            text,
            add_special_tokens=MAIN_LATENCY_ADD_SPECIAL_TOKENS,
        )
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(
            values,
            Sequence,
        ):
            raise TypeError("tokenizer.encode() must return a token-id sequence")
        token_ids = tuple(values)
        if not token_ids or any(
            type(token_id) is not int or token_id < 0 for token_id in token_ids
        ):
            raise ValueError("prepared latency segment has invalid token ids")
        contracts.append(
            {
                "chunk_id": chunk_id,
                "token_count": len(token_ids),
                "token_ids_digest": token_ids_digest(token_ids),
            }
        )
    if not contracts:
        raise ValueError("prepared latency cache prefix must contain segments")
    return contracts


def _worker_result_record(result: _WorkerResult) -> dict[str, Any]:
    batches = [
        {
            "cache_prefix_tokens": batch.cache_prefix_tokens,
            "context_tokens": batch.context_tokens,
            "durable_byte_count": batch.durable_byte_count,
            "durable_sync_seconds": batch.durable_sync_seconds,
            "generation_seconds": batch.generation_seconds,
            "input_token_slots": batch.input_token_slots,
            "task_count": len(batch.task_ids),
            "task_ids_sha256": _canonical_sha256(sorted(batch.task_ids)),
        }
        for batch in result.batches
    ]
    return {
        "batches": batches,
        "cache_prefix_tokens": sum(
            batch.cache_prefix_tokens for batch in result.batches
        ),
        "initialization_seconds": result.initialization_seconds,
        "lifecycle_seconds": result.lifecycle_seconds,
        "persistent_generator_instances": 1,
        "worker_index": result.worker_index,
    }


def _execution_config_record(
    config: PublicationLatencyHandoffExecutionConfig,
) -> dict[str, Any]:
    return {
        "align_bytes": config.align_bytes,
        "generator_family": config.generator_family,
        "generator_device_map": config.generator_device_map,
        "generator_add_special_tokens": config.generator_add_special_tokens,
        "generator_cache_axis_order": config.generator_cache_axis_order,
        "generator_model_dtype": config.generator_model_dtype,
        "generator_quantization": config.generator_quantization,
        "generator_quantization_config": dict(config.generator_quantization_config),
        "generator_version": config.generator_version,
        "generator_trust_remote_code": config.generator_trust_remote_code,
        "layout": {
            "block_size": config.layout.block_size,
            "bytes_per_token": config.layout.bytes_per_token,
            "dtype": config.layout.dtype,
            "head_size": config.layout.head_size,
            "key_position_encoding": cast(
                KVKeyPositionEncoding,
                config.layout.key_position_encoding,
            ).value,
            "kv_stride_bytes": config.layout.kv_stride_bytes,
            "layout_version": config.layout.layout_version,
            "lora_id": config.layout.lora_id,
            "model_id": config.layout.model_id,
            "num_kv_heads": config.layout.num_kv_heads,
            "num_layers": config.layout.num_layers,
            "num_query_heads": config.layout.num_query_heads,
            "payload_axis_order": cast(
                KVPayloadAxisOrder,
                config.layout.payload_axis_order,
            ).value,
            "pre_rope": config.layout.pre_rope,
            "rope_rotary_dim": config.layout.rope_rotary_dim,
            "rope_theta": config.layout.rope_theta,
            "shares_kv_storage": config.layout.shares_kv_storage,
            "storage_layout": cast(
                KVStorageLayout,
                config.layout.storage_layout,
            ).value,
        },
        "model_revision": config.model_revision,
        "pipeline_parallel_size": config.pipeline_parallel_size,
        "tensor_parallel_size": config.tensor_parallel_size,
        "tokenizer_id": config.tokenizer_id,
        "tokenizer_revision": config.tokenizer_revision,
        "vllm_bitsandbytes_loader_source_sha256": (
            config.vllm_bitsandbytes_loader_source_sha256
        ),
    }


def _validate_publication_generator(
    generator: object,
    *,
    config: PublicationLatencyHandoffExecutionConfig,
    enforce_production_identity: bool,
) -> None:
    if not callable(getattr(generator, "generate", None)):
        raise TypeError("worker_factory must return a KV chunk generator")
    if getattr(generator, "pre_rope", None) is not True:
        raise ValueError("publication latency generator must capture pre-RoPE KV")
    if getattr(generator, "add_special_tokens", None) is not (
        MAIN_LATENCY_ADD_SPECIAL_TOKENS
    ):
        raise ValueError("publication generator tokenizer contract is invalid")
    if not enforce_production_identity:
        return
    expected_identity = {
        "cache_axis_order": config.generator_cache_axis_order,
        "generator_family": config.generator_family,
        "generator_version": config.generator_version,
        "model_id": config.layout.model_id,
        "model_revision": config.model_revision,
        "tokenizer_id": config.tokenizer_id,
        "tokenizer_revision": config.tokenizer_revision,
    }
    if any(
        getattr(generator, key, None) != value
        for key, value in expected_identity.items()
    ):
        raise ValueError(
            "production generator identity differs from execution contract"
        )
    resolved = getattr(generator, "config", None)
    expected_resolved = {
        "add_special_tokens": config.generator_add_special_tokens,
        "cache_axis_order": config.generator_cache_axis_order,
        "device": "cuda",
        "device_map": config.generator_device_map,
        "model_id": config.layout.model_id,
        "model_revision": config.model_revision,
        "quantization": config.generator_quantization,
        "tokenizer_revision": config.tokenizer_revision,
        "torch_dtype": config.generator_model_dtype,
        "trust_remote_code": config.generator_trust_remote_code,
    }
    if resolved is None or any(
        getattr(resolved, key, None) != value
        for key, value in expected_resolved.items()
    ):
        raise ValueError("production generator configuration differs from contract")
    resolved_tokenizer_id = getattr(resolved, "resolved_tokenizer_id", None)
    if resolved_tokenizer_id != config.tokenizer_id or dict(
        getattr(resolved, "quantization_config", {})
    ) != dict(config.generator_quantization_config):
        raise ValueError("production generator tokenizer/quantization contract drift")
    if dict(getattr(resolved, "model_kwargs", {})) or dict(
        getattr(resolved, "tokenizer_kwargs", {})
    ):
        raise ValueError("production generator contains unbound loader kwargs")


def _plan_task_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    workers = _mapping_sequence(record.get("workers"), field_name="workers")
    task_ids = tuple(
        _required_string(item, "task_id")
        for worker in workers
        for item in _mapping_sequence(worker.get("items"), field_name="worker.items")
    )
    if len(task_ids) != PUBLICATION_LATENCY_HANDOFF_TASK_COUNT:
        raise ValueError("latency handoff plan task coverage is incomplete")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("latency handoff plan task IDs are duplicated")
    return task_ids


def _validate_closed_plan_envelope(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if record.get("record_type") != PUBLICATION_LATENCY_HANDOFF_PLAN_RECORD_TYPE:
        raise ValueError("latency handoff plan record_type is invalid")
    if record.get("schema_version") != PUBLICATION_LATENCY_HANDOFF_PLAN_SCHEMA_VERSION:
        raise ValueError("latency handoff plan schema_version is invalid")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("latency handoff plan closed_record_sha256 is invalid")
    _plan_task_ids(record)


def _coverage_int(record: Mapping[str, Any], field_name: str) -> int:
    return _required_int(_required_mapping(record, "coverage"), field_name)


def _validated_worker_count(value: Any) -> int:
    if (
        type(value) is not int
        or not 1 <= value <= PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
    ):
        raise ValueError(
            "worker_count must be between 1 and "
            f"{PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS}"
        )
    return value


def _relative_input_path(root: Path, path: Path) -> str:
    resolved_root = root.expanduser().resolve()
    resolved_path = path.expanduser().resolve()
    _require_confined_path(resolved_root, resolved_path, field_name="prepared JSONL")
    return resolved_path.relative_to(resolved_root).as_posix()


def _confined_relative_path(root: Path, value: str, *, field_name: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{field_name} must be a confined relative path")
    target = (root / relative).resolve()
    _require_confined_path(root.resolve(), target, field_name=field_name)
    return target


def _require_confined_path(root: Path, path: Path, *, field_name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field_name} must remain inside its output root") from exc


def _require_fresh_output_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite generation output: {path}")
    cursor = path.parent
    while not cursor.exists():
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise ValueError("generation output parent must be a real directory")


def _create_q8_phase_lease_root(path: str | Path) -> Path:
    root = Path(path).expanduser().absolute()
    for candidate in (*reversed(root.parents), root):
        if candidate.is_symlink():
            raise ValueError(f"Q8 phase lease path traverses a symlink: {candidate}")
    if root.exists():
        raise FileExistsError(f"Q8 phase lease already exists: {root}")
    if not root.parent.is_dir():
        raise ValueError("Q8 phase lease parent must already be a real directory")
    root.mkdir()
    _sync_directory(root.parent)
    return root


def _remove_empty_q8_phase_lease_root(root: Path) -> None:
    for path in tuple(root.iterdir()):
        if path.name == "phase-lease.json" and path.is_file() and not path.is_symlink():
            path.unlink()
    if root.is_dir() and not root.is_symlink() and not any(root.iterdir()):
        root.rmdir()
        _sync_directory(root.parent)


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _canonical_jsonl_records(path: Path) -> tuple[dict[str, Any], ...]:
    content = path.read_bytes()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("prepared latency JSONL must be UTF-8") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError("prepared latency JSONL must not contain blank lines")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"prepared latency JSONL line {line_number} is invalid"
            ) from exc
        if not isinstance(value, Mapping):
            raise ValueError("prepared latency JSONL rows must be objects")
        record = cast(dict[str, Any], json.loads(json.dumps(dict(value))))
        if line.encode("utf-8") != _canonical_json_bytes(record, pretty=False):
            raise ValueError("prepared latency JSONL is not canonical")
        records.append(record)
    if not content.endswith(b"\n"):
        raise ValueError("prepared latency JSONL must end with a newline")
    return tuple(records)


def _write_canonical_jsonl_exclusive(
    records: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite JSONL: {path}")
    content = b"".join(
        _canonical_json_bytes(record, pretty=False) + b"\n" for record in records
    )
    with path.open("xb") as handle:
        handle.write(content)


def _write_handoff_dataset_jsonl_exclusive(
    records: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """Mirror the handoff-bundle canonical JSONL contract byte-for-byte."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite JSONL: {path}")
    content = "".join(
        json.dumps(dict(record), sort_keys=True) + "\n" for record in records
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(content)


def _write_canonical_json_exclusive(record: Mapping[str, Any], path: Path) -> None:
    _reject_q8_symlink_ancestors(path, "Q8 canonical JSON output")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_q8_symlink_ancestors(path, "Q8 canonical JSON output")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite JSON: {path}")
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(record, pretty=True))


def _read_q8_controller_record(path: Path, field_name: str) -> dict[str, Any]:
    _reject_q8_symlink_ancestors(path, field_name)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{field_name} must be one regular file")
    content = path.read_bytes()
    value = json.loads(content)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must contain one JSON object")
    if content != _canonical_json_bytes(value, pretty=True):
        raise ValueError(f"{field_name} is not canonical JSON")
    expected_digest = value.get("closed_record_sha256")
    if not isinstance(expected_digest, str) or not hmac.compare_digest(
        expected_digest, _closed_record_sha256(value)
    ):
        raise ValueError(f"{field_name} closed digest mismatch")
    return value


def _reject_q8_symlink_ancestors(path: Path, field_name: str) -> None:
    absolute = path.expanduser().absolute()
    for candidate in (*reversed(absolute.parents), absolute):
        if candidate.is_symlink():
            raise ValueError(f"{field_name} path traverses a symlink: {candidate}")


def _sync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"durable output contains a symlink: {path}")
        if path.is_file():
            _sync_file(path)
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_dir()),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        _sync_directory(path)
    _sync_directory(root)


def _sync_worker_data_ancestor_directories(output_root: Path) -> None:
    """Persist every parent entry before the worker-result commit marker."""

    for relative in ("pending", "worker-records"):
        parent = output_root / relative
        if not parent.is_dir() or parent.is_symlink():
            raise ValueError(f"worker durable parent is missing or unsafe: {parent}")
        _sync_directory(parent)
    _sync_directory(output_root)


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tree_byte_count(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _positive_duration(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("clock duration must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError("clock duration must be non-negative and finite")
    if normalized == 0:
        return 1e-12
    return normalized


def _nonnegative_duration(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("clock duration must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError("clock duration must be non-negative and finite")
    return normalized


def _required_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    item = value.get(field_name)
    if not isinstance(item, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return item


def _require_exact_mapping_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} must use a closed schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _mapping_sequence(value: Any, *, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be an array")
    items = tuple(value)
    if any(not isinstance(item, Mapping) for item in items):
        raise ValueError(f"{field_name} entries must be objects")
    return cast(tuple[Mapping[str, Any], ...], items)


def _required_string(value: Mapping[str, Any], field_name: str) -> str:
    item = value.get(field_name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{field_name} must be a non-empty string")
    return item


def _required_int(value: Mapping[str, Any], field_name: str) -> int:
    item = value.get(field_name)
    if type(item) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return item


def _positive_epoch_millis(value: Mapping[str, Any], field_name: str) -> int:
    result = _required_int(value, field_name)
    if result <= 0:
        raise ValueError(f"{field_name} must be a positive epoch-millisecond integer")
    return result


def _databricks_cloud_id(value: Any, *, field_name: str) -> str:
    if type(value) is int and value >= 0:
        return str(value)
    if (
        isinstance(value, str)
        and value
        and value == value.strip()
        and len(value) <= 256
        and not any(character.isspace() for character in value)
    ):
        return value
    raise ValueError(f"{field_name} must be a sanitized non-empty Databricks ID")


def _required_bool(value: Mapping[str, Any], field_name: str) -> bool:
    item = value.get(field_name)
    if type(item) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return item


def _optional_int(value: Mapping[str, Any], field_name: str) -> int | None:
    item = value.get(field_name)
    if item is None:
        return None
    if type(item) is not int:
        raise ValueError(f"{field_name} must be an integer or null")
    return item


def _optional_number(value: Mapping[str, Any], field_name: str) -> float | None:
    item = value.get(field_name)
    if item is None:
        return None
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise ValueError(f"{field_name} must be numeric or null")
    normalized = float(item)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _required_positive_number(value: Mapping[str, Any], field_name: str) -> float:
    item = value.get(field_name)
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise ValueError(f"{field_name} must be numeric")
    normalized = float(item)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field_name} must be positive and finite")
    return normalized


def _positive_float(value: Any, *, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field_name} must be positive and finite")
    return normalized


def _require_sha256(value: Any, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _json_object(content: bytes, *, field_name: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must be valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return cast(dict[str, Any], json.loads(json.dumps(dict(value))))


def _canonical_json_bytes(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        return (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value, pretty=False)).hexdigest()


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("closed_record_sha256", None)
    return _canonical_sha256(payload)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one hash-bound production producer payload."""

    parser = argparse.ArgumentParser(
        description="Execute one publication latency handoff producer."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("run-worker")
    worker.add_argument("--worker-payload-json", required=True)
    worker.add_argument("--expected-worker-payload-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_publication_latency_handoff_worker(
            args.worker_payload_json,
            expected_worker_payload_sha256=(args.expected_worker_payload_sha256),
        )
        print(
            json.dumps(
                {
                    "closed_record_sha256": result["closed_record_sha256"],
                    "ok": True,
                    "worker_index": result["worker_index"],
                },
                sort_keys=True,
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "ok": False,
                },
                sort_keys=True,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PUBLICATION_LATENCY_HANDOFF_ARM_ID",
    "PUBLICATION_LATENCY_HANDOFF_DTYPE",
    "PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY",
    "PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_RECORD_TYPE",
    "PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_SCHEMA_VERSION",
    "PUBLICATION_LATENCY_HANDOFF_EXECUTION_FILENAME",
    "PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED",
    "PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_LOCAL_TEST",
    "PUBLICATION_LATENCY_HANDOFF_EXECUTION_RECORD_TYPE",
    "PUBLICATION_LATENCY_HANDOFF_EXECUTION_SCHEMA_VERSION",
    "PUBLICATION_LATENCY_HANDOFF_PLAN_RECORD_TYPE",
    "PUBLICATION_LATENCY_HANDOFF_PLAN_SCHEMA_VERSION",
    "PUBLICATION_LATENCY_HANDOFF_RUNNER_FILENAME",
    "PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT",
    "PUBLICATION_LATENCY_HANDOFF_RUNNER_SHA256",
    "PUBLICATION_LATENCY_HANDOFF_TASK_COUNT",
    "PUBLICATION_LATENCY_HANDOFF_WORKER_PAYLOAD_RECORD_TYPE",
    "PUBLICATION_LATENCY_HANDOFF_WORKER_PAYLOAD_SCHEMA_VERSION",
    "PUBLICATION_LATENCY_HANDOFF_WORKER_RESULT_RECORD_TYPE",
    "PUBLICATION_LATENCY_HANDOFF_WORKER_RESULT_SCHEMA_VERSION",
    "DatabricksPublicationLatencyHandoffJobConfig",
    "PublicationLatencyGeneratorFactory",
    "PublicationLatencyGeneratorHardwareQualification",
    "PublicationLatencyHandoffExecutionConfig",
    "PublicationLatencyHandoffDatabricksAttestationBinding",
    "PublicationLatencyHandoffGenerationResult",
    "PublicationLatencyHandoffSubmissionAuthorization",
    "PublicationLatencyHandoffServingAuthorization",
    "PublicationLatencyServingHandoffBundle",
    "authorize_publication_latency_handoff_serving",
    "build_publication_latency_handoff_generation_plan",
    "build_publication_latency_handoff_databricks_attestation",
    "build_publication_latency_handoff_execution_config",
    "build_publication_latency_handoff_worker_payloads",
    "build_databricks_publication_latency_handoff_worker_submit_payloads",
    "close_publication_latency_handoff_generation_from_workers",
    "execute_publication_latency_handoff_generation_plan_local_test_helper",
    "publication_latency_handoff_worker_attempt_id",
    "publication_latency_handoff_terminal_actual_gpu_seconds_from_ledger",
    "read_publication_latency_handoff_generation_plan",
    "read_publication_latency_handoff_databricks_attestation",
    "read_publication_latency_handoff_generation_result",
    "reconcile_publication_latency_handoff_worker_attempt_json",
    "reserve_and_submit_publication_latency_handoff_worker_wave",
    "resume_publication_latency_handoff_worker_wave",
    "require_publication_latency_full_launch_ready",
    "require_publication_latency_handoff_submission_authorization",
    "require_publication_latency_handoff_serving_authorization",
    "resolve_publication_latency_serving_handoff_bundle",
    "run_publication_latency_handoff_worker",
    "validate_publication_latency_handoff_generation_execution_record",
    "validate_publication_latency_handoff_generation_plan",
    "validate_publication_latency_handoff_worker_payload",
    "write_publication_latency_handoff_generation_plan",
    "write_publication_latency_handoff_databricks_attestation",
    "write_publication_latency_handoff_runner_script",
    "write_publication_latency_handoff_worker_payloads",
]
