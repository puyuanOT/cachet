"""Fail-closed streaming execution for the complete publication score pass.

The inventory and token-balanced shard plan are built by
``publication_inputs``.  This module binds those immutable inputs to exact
worker manifests, renders at most sixteen persistent Databricks tasks, and
executes each worker's shards sequentially.  A shard's generated Q8 KV is
deleted only after both raw method outputs have been re-scored, copied to the
durable evidence directory, and closed by a reread-verified commit record.
Generic planners remain useful for dry runs; Databricks rendering and worker
execution accept only the frozen 83,653-example publication inventory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast

from document_kv_cache._benchmark_datasets import _example_from_record
from document_kv_cache._benchmark_manifest import (
    _build_experiment_manifest,
    benchmark_experiment_manifest_to_record,
)
from document_kv_cache._benchmark_models import BenchmarkManifestContext
from document_kv_cache._hardware_targets import (
    DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE,
    validate_aws_single_node_gpu_type,
)
from document_kv_cache.benchmark_runner import (
    DEFAULT_OPENAI_COMPLETIONS_ENDPOINT,
    benchmark_run_result_from_record,
)
from document_kv_cache.benchmark_handoffs import (
    enrich_benchmark_jsonl_with_handoffs,
    generate_benchmark_handoff_bundles,
    load_benchmark_kv_chunk_generator,
    read_benchmark_handoff_manifest_json,
)
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    DOCUMENT_KV_ARTIFACT_ID_PARAM,
    DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM,
    DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM,
    FINAL_ANSWER_EXTRACTED_METADATA_KEY,
    FINAL_ANSWER_NO_EXTRACTION_VALUE,
    FINAL_ANSWER_PARSER_DIGEST_METADATA_KEY,
    FINAL_ANSWER_PARSER_ID_METADATA_KEY,
    FINAL_ANSWER_PARSER_PLUGIN_METADATA_KEY,
    FINAL_ANSWER_PARSER_STATUSES,
    FINAL_ANSWER_PARSER_STATUS_METADATA_KEY,
    FINAL_ANSWER_PARSER_VALID_METADATA_KEY,
    FINAL_ANSWER_PARSER_VERSION_METADATA_KEY,
    NIAH_CELL_IDS,
    BenchmarkSuite,
    DatasetScoreContext,
    baseline_prefill_arm,
    benchmark_cache_prefix_segments,
    build_prompt_parts,
    default_dataset_scorer_registry,
    method_benchmark_arm,
    SUPPORTED_V1_DATASETS,
)
from document_kv_cache.databricks_job import (
    DEFAULT_DATABRICKS_DATA_SECURITY_MODE,
    DEFAULT_DATABRICKS_SPARK_VERSION,
    DEFAULT_DATABRICKS_TASK_MAX_RETRIES,
    DatabricksSingleNodeGPUClusterConfig,
    _validated_databricks_run_timeout_seconds,
    _validated_databricks_task_max_retries,
    build_single_node_gpu_cluster,
)
from document_kv_cache.databricks_resource_ledger import (
    DatabricksBatchReservationAuthorization,
    DatabricksClusterHourLedger,
    DatabricksClusterHourReservation,
    DatabricksLedgerPrefix,
    DatabricksRunAttemptReservationRequest,
    MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS,
    MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    canonical_databricks_submit_payload_snapshot,
    databricks_cluster_hour_ledger_from_record,
    databricks_cluster_hour_ledger_to_record,
    databricks_ledger_path_sha256,
    databricks_ledger_prefix,
    databricks_ledger_prefix_at_counts,
    databricks_ledger_prefix_from_record,
    databricks_submit_payload_reservation,
    read_databricks_cluster_hour_ledger_json,
    record_databricks_verified_run_terminal_actual_json,
    require_databricks_batch_reservation_authorization,
    require_databricks_ledger_prefix,
    replay_databricks_run_attempt_batch_authorization_json,
    reserve_databricks_run_attempt_batch_authorized_json,
)
from document_kv_cache.databricks_runs import (
    DatabricksURLOpener,
    DatabricksWorkspaceConfig,
    bind_databricks_run_idempotency_token,
    databricks_run_status_record,
    get_databricks_run,
    recover_pre_reserved_databricks_run,
    require_databricks_current_user_name,
    require_databricks_run_idempotency_token,
    resume_pre_reserved_databricks_run,
    summarize_databricks_run,
    submit_pre_reserved_databricks_run,
    validate_databricks_run_status_sidecar,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE,
    GPU_QUALIFICATION_GENERATION_GPU,
    GPU_QUALIFICATION_GENERATION_HARDWARE_ID,
    GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND,
    GPUQualificationSelection,
    canonical_gpu_qualification_json,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPUQualificationArtifactPinsV2,
    validate_gpu_qualification_evidence_v2_record,
    validate_gpu_qualification_v2_runtime_attestation,
)
from document_kv_cache._gpu_qualification_sentinels_v2 import (
    _BoundedSubprocessStartFailure,
    _BoundedSubprocessTransportFailure,
    _bounded_stream_result_is_exact,
    _run_bounded_binary_subprocess,
)
from document_kv_cache.serving_env import (
    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
    GPU_RUNTIME_PYTHONWARNINGS,
    gpu_runtime_warning_environment_overrides,
)
from document_kv_cache.gpu_qualification_databricks import (
    GPUQualificationLaunchAuthorization,
    require_gpu_qualification_launch_authorization,
)
from document_kv_cache.main_latency_inputs import (
    MAIN_LATENCY_ADD_SPECIAL_TOKENS,
    MAIN_LATENCY_TOKENIZER_ID,
    MAIN_LATENCY_TOKENIZER_REVISION,
    MainLatencyTokenizer,
    load_main_latency_tokenizer,
)
from document_kv_cache.model_profiles import (
    QWEN3_4B_ROPE_ROTARY_DIM,
    QWEN3_4B_ROPE_THETA,
    layout_for_model,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS,
    PUBLICATION_CAMPAIGN_ENGINE_VERSION,
    PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS,
)
from document_kv_cache.publication_inputs import (
    FULL_SCORE_MAX_NATURAL_PROMPT_TOKENS,
    FULL_SCORE_MAX_WORKERS,
    FullScoreDatasetSource,
    FullScoreInventory,
    FullScoreInventoryItem,
    full_score_inventory_to_record,
    validate_full_score_inventory_record,
    validate_full_score_shard_plan,
)
from document_kv_cache.publication_latency_execution import (
    PublicationLatencyCollectionAuthorization,
    require_publication_latency_collection_authorization,
    validate_publication_latency_collection_record,
)
from document_kv_cache.runtime_telemetry import RuntimeTelemetrySampler
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PATCHED_WHEEL_SHA256,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    VLLM_PATCHED_WHEEL_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
)
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
)
from vllm_kv_injection.vllm_native_provider_constants import (
    DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
)


FULL_SCORE_WORKER_PAYLOAD_RECORD_TYPE = "cachet.full_score_worker_payload.v2"
FULL_SCORE_WORKER_PAYLOAD_SCHEMA_VERSION = 2
FULL_SCORE_SHARD_EVIDENCE_RECORD_TYPE = "cachet.full_score_shard_evidence.v2"
FULL_SCORE_SHARD_EVIDENCE_SCHEMA_VERSION = 2
FULL_SCORE_DELETION_ATTESTATION_RECORD_TYPE = (
    "cachet.full_score_deletion_attestation.v1"
)
FULL_SCORE_DELETION_ATTESTATION_SCHEMA_VERSION = 1
FULL_SCORE_AGGREGATE_RECORD_TYPE = "cachet.full_score_aggregate.v2"
FULL_SCORE_AGGREGATE_SCHEMA_VERSION = 2
FULL_SCORE_EXECUTION_PLAN_RECORD_TYPE = "cachet.full_score_execution_plan.v2"
FULL_SCORE_EXECUTION_PLAN_SCHEMA_VERSION = 2
FULL_SCORE_READY_SHARD_RECORD_TYPE = "cachet.full_score_ready_shard.v2"
FULL_SCORE_READY_SHARD_SCHEMA_VERSION = 2
FULL_SCORE_WAVE_COMPLETION_RECORD_TYPE = "cachet.full_score_wave_completion.v1"
FULL_SCORE_WAVE_COMPLETION_SCHEMA_VERSION = 1
FULL_SCORE_PRODUCER_PHASE_COMPLETION_RECORD_TYPE = (
    "cachet.full_score_producer_phase_completion.v1"
)
FULL_SCORE_PRODUCER_PHASE_COMPLETION_SCHEMA_VERSION = 1
FULL_SCORE_MATCHED_BLOCK_RECORD_TYPE = "cachet.full_score_matched_billing_block.v1"
FULL_SCORE_MATCHED_BLOCK_SCHEMA_VERSION = 1
FULL_SCORE_LIVE_P90_RECORD_TYPE = "cachet.full_score_live_p90_budget_gate.v1"
FULL_SCORE_LIVE_P90_SCHEMA_VERSION = 1
FULL_SCORE_PHASE_TERMINAL_RECORD_TYPE = "cachet.full_score_phase_terminal.v1"
FULL_SCORE_PHASE_TERMINAL_SCHEMA_VERSION = 1
FULL_SCORE_CONNECTOR_PROOF_RECORD_TYPE = "cachet.full_score_connector_proof.v1"
FULL_SCORE_CONNECTOR_PROOF_SCHEMA_VERSION = 1
FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE = "publication"
FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE = "local_fixture_only"
FULL_SCORE_LIVE_P90_SEED = 20_260_824
FULL_SCORE_LIVE_P90_DRAWS = 10_000
FULL_SCORE_PROTOCOL_ID = "cachet-vllm-0.27.1-complete-score-v2"
FULL_SCORE_REQUEST_CUSTOMIZATION_DIGEST = (
    "440181b5f7930106194b542de751661bbd5662a071e7d10b64cf8172ac29774f"
)
FULL_SCORE_METHODS = ("baseline_prefill", "vanilla_prefill")
FULL_SCORE_MAX_TOKENS = 64
FULL_SCORE_REQUEST_PARALLELISM = 4
FULL_SCORE_DATABRICKS_TASK_TIMEOUT_SECONDS = 21_600
FULL_SCORE_RUNTIME_VERIFIER_TIMEOUT_SECONDS = 300.0
FULL_SCORE_RUNTIME_VERIFIER_OUTPUT_LIMIT_BYTES = 1_048_576
FULL_SCORE_TEMPERATURE = 0.0
FULL_SCORE_PASSES_PER_METHOD = 1
FULL_SCORE_MODEL_ID = MAIN_LATENCY_TOKENIZER_ID
FULL_SCORE_SERVED_MODEL_NAME = "qwen3:4b-instruct"
FULL_SCORE_MODEL_DTYPE = "bfloat16"
FULL_SCORE_MODEL_QUANTIZATION = "bitsandbytes"
FULL_SCORE_GENERATOR_QUANTIZATION = "bitsandbytes-4bit"
FULL_SCORE_GENERATOR_QUANTIZATION_CONFIG = {
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_storage": "uint8",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "load_in_4bit": True,
}
FULL_SCORE_GENERATOR_FACTORY = (
    "document_kv_cache.transformers_generator:"
    "build_pre_rope_transformers_kv_chunk_generator"
)
FULL_SCORE_GENERATOR_VERSION = "vllm-0.27.1-publication-q8-e5m2"
FULL_SCORE_VLLM_BNB_LOADER_MEMBER = (
    "vllm/model_executor/model_loader/bitsandbytes_loader.py"
)
FULL_SCORE_VLLM_BNB_LOADER_SHA256 = (
    "cf2c10fcaf5b6e997a8d5f80712af1f251a7292b5d76813e28b39bc03fa7c629"
)
FULL_SCORE_PRODUCER_HARDWARE_TARGET = GPU_QUALIFICATION_GENERATION_HARDWARE_ID
FULL_SCORE_PRODUCER_GPU_NAME = GPU_QUALIFICATION_GENERATION_GPU
FULL_SCORE_PRODUCER_NODE_TYPE_ID = GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE
FULL_SCORE_PRODUCER_ZONE_ID = "us-west-2a"
FULL_SCORE_CONSUMER_NODE_TYPE_ID = "g6.8xlarge"
FULL_SCORE_KV_DTYPE = "fp8_e5m2"
FULL_SCORE_ATTENTION_BACKEND = "TRITON_ATTN"
FULL_SCORE_GPU_MEMORY_UTILIZATION_CANDIDATES = (0.70, 0.75, 0.80)
FULL_SCORE_Q8_BYTES_PER_CACHE_PREFIX_TOKEN = 73_728
FULL_SCORE_MODEL_NUM_LAYERS = 36
FULL_SCORE_DEFAULT_MAX_BACKLOG_BYTES = 500_000_000_000
FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE = 16
FULL_SCORE_READY_SHARD_METADATA_ALLOWANCE_BYTES = 64 * 1024 * 1024
FULL_SCORE_PUBLICATION_INVENTORY_SHA256 = (
    "e19fefa656d8975946b13bb9987f801ec486c4bfde5e9d5ed82a877e80676b11"
)
FULL_SCORE_PUBLICATION_SHARD_PLAN_SHA256 = (
    "605c15ef5317bb0b6d6f6a4057dbacbd97ae31af94a3d497585a88c138c9ba84"
)
FULL_SCORE_PUBLICATION_EXECUTION_PLAN_SHA256 = (
    "88315f8bcae2317659fea61c05b7e5c56e7e56dd1330368fdccfe816c453ad84"
)
FULL_SCORE_PUBLICATION_ITEM_COUNT = 83_653
FULL_SCORE_PUBLICATION_SHARD_COUNT = 160
FULL_SCORE_PUBLICATION_CACHE_PREFIX_TOKENS = 63_455_746
FULL_SCORE_PUBLICATION_NATURAL_PROMPT_TOKENS = 66_448_937
_FULL_SCORE_PUBLICATION_SOURCE_RECORDS = (
    (
        "biography",
        77_790_712,
        72_831,
        "8996bd76ed7b86c0f82a6eccdedaef1e37c5f2bcdbe359c787feb1dea9f03168",
        "94285da4fa086ade7d5e3c941ee3a195b72333daab5e3fbdcd702bd18a3f2233",
        "491cf2a9515ade17164bf15cbc103dcce0b53b93cc0852c26364cea1d00e206a",
    ),
    (
        "hotpotqa",
        51_118_561,
        7_405,
        "c871d8536c48e8db5e39e2215ae5e9eb37260dbc4eb871816e2049f1863ae2dc",
        "1a06929de4ffb1b8507169932189916034727058ab23af2f1e36695b80d46545",
        "77ecb949dc897c4e6d9ce0347d552bd13b07e403ebfbcdcfa4e6ea3536942ae1",
    ),
    (
        "musique",
        30_669_730,
        2_417,
        "f1404c3ccef4b1df8954525efc95f09ce26e4cbf39747372c06f452fd9d13c6a",
        "4a8e033e92f0d15ae0286afca1fe7d9b1a68797d382a2af1652baa63b2b428d6",
        "a381e6174eef132f51a59279fe9850395e8b322617cdaf2ed2b2878243e53f7f",
    ),
    (
        "niah",
        74_968_975,
        1_000,
        "fbf00dd68b88a73c0290cd372b712c6e25a4141192867b6642e2141e5bbb95d8",
        "0afc5432a08f9182c7574d37e5ef165ab5c1bb2be79de3ce2fb0630e27b9d3c2",
        "2bfc80b49efac8a2312cff2b12543516099008fe0babbd7219e6b7eae66dd9f7",
    ),
)
FULL_SCORE_MIN_MAX_MODEL_LEN = (
    FULL_SCORE_MAX_NATURAL_PROMPT_TOKENS + FULL_SCORE_MAX_TOKENS
)
FULL_SCORE_VANILLA_ARM_ID = "document_kv_cache:vanilla_prefill"
FULL_SCORE_RUNNER_SCRIPT = """from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import pathlib
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


def _bootstrap(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
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
    args, remaining = parser.parse_known_args(argv)
    _verified_path(__file__, args.runner_sha256, "full-score runner")
    package_wheel = _verified_path(
        args.package_wheel_uri, args.package_wheel_sha256, "Cachet wheel"
    )
    runtime_lock = _verified_path(
        args.runtime_lock_uri, args.runtime_lock_sha256, "runtime lock"
    )
    patched_vllm_wheel = _verified_path(
        args.patched_vllm_wheel_uri,
        args.patched_vllm_wheel_sha256,
        "patched vLLM wheel",
    )
    patched_flashinfer_wheel = _verified_path(
        args.patched_flashinfer_wheel_uri,
        args.patched_flashinfer_wheel_sha256,
        "patched FlashInfer wheel",
    )
    runtime_closure_manifest = _verified_path(
        args.runtime_closure_manifest_uri,
        args.runtime_closure_manifest_sha256,
        "runtime closure manifest",
    )
    venv_dir = os.path.abspath(args.runtime_venv_dir)
    if not venv_dir.startswith("/local_disk0/"):
        raise ValueError("runtime venv must be rooted under /local_disk0")
    identity = hashlib.sha256(
        (
            "cachet.full_score.locked_runtime.v2\\0"
            + args.runner_sha256
            + args.package_wheel_sha256
            + args.runtime_lock_sha256
            + args.patched_vllm_wheel_sha256
            + args.patched_flashinfer_wheel_sha256
            + args.runtime_closure_manifest_sha256
        ).encode("ascii")
    ).hexdigest()
    marker = os.environ.get("CACHET_FULL_SCORE_LOCKED_RUNTIME")
    venv_python = os.path.join(venv_dir, "bin", "python")
    if marker == identity:
        if os.path.realpath(sys.executable) != os.path.realpath(venv_python):
            raise RuntimeError("locked-runtime marker is set outside the bound venv")
        from document_kv_cache.full_score_execution import main

        raise SystemExit(main(remaining))
    if os.path.exists(venv_dir):
        raise FileExistsError("refusing to reuse an unverified full-score runtime")
    pip_environment = _pip_subprocess_environment()
    subprocess.check_call(
        [sys.executable, "-m", "venv", "--copies", venv_dir],
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
    vllm_spec = (
        "vllm @ " + pathlib.Path(patched_vllm_wheel).resolve().as_uri()
        + "#sha256=" + args.patched_vllm_wheel_sha256
    )
    flashinfer_spec = (
        "flashinfer-python @ "
        + pathlib.Path(patched_flashinfer_wheel).resolve().as_uri()
        + "#sha256=" + args.patched_flashinfer_wheel_sha256
    )
    package_spec = (
        "cachet-kv @ " + pathlib.Path(package_wheel).resolve().as_uri()
        + "#sha256=" + args.package_wheel_sha256
    )
    subprocess.check_call(
        [*pip, "install", "--no-deps", vllm_spec],
        env=pip_environment,
    )
    subprocess.check_call(
        [*pip, "install", "--no-deps", flashinfer_spec],
        env=pip_environment,
    )
    subprocess.check_call(
        [*pip, "install", "--no-deps", package_spec],
        env=pip_environment,
    )
    env = dict(pip_environment)
    env["CACHET_FULL_SCORE_LOCKED_RUNTIME"] = identity
    os.execve(
        venv_python,
        [venv_python, "-m", "document_kv_cache.full_score_execution", *remaining],
        env,
    )


if __name__ == "__main__":
    _bootstrap(sys.argv[1:])
""".replace(
    "__GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL__",
    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
).replace("__GPU_RUNTIME_PYTHONWARNINGS__", GPU_RUNTIME_PYTHONWARNINGS)
FULL_SCORE_RUNNER_SHA256 = sha256(FULL_SCORE_RUNNER_SCRIPT.encode("utf-8")).hexdigest()

_SHA256_LENGTH = 64
_TRANSFER_FIELDS = frozenset({"kv_transfer_params", "arm_kv_transfer_params"})
_SECRET_KEY_PARTS = frozenset(
    {"credential", "key", "pass", "password", "pat", "secret", "token"}
)


class FullScoreCommandRunner(Protocol):
    """Subprocess surface used by the production worker and deterministic tests."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path | None = None,
    ) -> None: ...


class FullScoreCompactArtifactResolver(Protocol):
    """Resolve one compact DBFS/Volume URI to a bounded local CAS blob."""

    def __call__(self, uri: str) -> Path: ...


class FullScoreCompactArtifactPublisher(Protocol):
    """Publish one compact record and return its verified local CAS blob."""

    def __call__(self, uri: str, content: bytes) -> Path: ...


@dataclass(frozen=True, slots=True)
class FullScoreRuntimeConfig:
    """Exact runtime identity and vLLM launch settings for every worker."""

    python_executable: str
    runtime_lock_uri: str
    runtime_lock_sha256: str
    patched_vllm_wheel_uri: str
    patched_vllm_wheel_sha256: str
    patched_flashinfer_wheel_uri: str
    patched_flashinfer_wheel_sha256: str
    runtime_closure_manifest_uri: str
    runtime_closure_manifest_sha256: str
    vllm_wheel_install_spec: str
    flashinfer_wheel_install_spec: str
    kv_transfer_config: Mapping[str, Any]
    model_id: str = FULL_SCORE_MODEL_ID
    served_model_name: str = FULL_SCORE_SERVED_MODEL_NAME
    model_revision: str = MAIN_LATENCY_TOKENIZER_REVISION
    tokenizer_id: str = MAIN_LATENCY_TOKENIZER_ID
    tokenizer_revision: str = MAIN_LATENCY_TOKENIZER_REVISION
    hardware_target: str = "aws-g6-l4"
    engine_version: str = PUBLICATION_CAMPAIGN_ENGINE_VERSION
    gpu_memory_utilization: float = 0.80
    max_model_len: int = FULL_SCORE_MIN_MAX_MODEL_LEN
    max_num_seqs: int = FULL_SCORE_REQUEST_PARALLELISM
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    server_start_timeout_seconds: float = 900.0
    request_timeout_seconds: float = 900.0
    generator_timeout_seconds: float = float(FULL_SCORE_DATABRICKS_TASK_TIMEOUT_SECONDS)
    telemetry_interval_seconds: float = 1.0
    generator_factory: str = FULL_SCORE_GENERATOR_FACTORY
    generator_version: str = FULL_SCORE_GENERATOR_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "python_executable",
            "runtime_lock_uri",
            "patched_vllm_wheel_uri",
            "patched_flashinfer_wheel_uri",
            "runtime_closure_manifest_uri",
            "vllm_wheel_install_spec",
            "flashinfer_wheel_install_spec",
            "generator_factory",
            "generator_version",
        ):
            _require_nonempty(getattr(self, field_name), field_name)
        fixed_hashes = {
            "runtime_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
            "patched_vllm_wheel_sha256": VLLM_PATCHED_WHEEL_SHA256,
            "patched_flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
            "runtime_closure_manifest_sha256": (RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256),
        }
        for field_name, expected in fixed_hashes.items():
            observed = _require_sha256(
                getattr(self, field_name),
                field_name=field_name,
            )
            if observed != expected:
                raise ValueError(f"{field_name} differs from native-v2 authority")
        if self.model_id != FULL_SCORE_MODEL_ID:
            raise ValueError("full-score model_id is frozen")
        if self.served_model_name != FULL_SCORE_SERVED_MODEL_NAME:
            raise ValueError("full-score served_model_name is frozen")
        if self.model_revision != MAIN_LATENCY_TOKENIZER_REVISION:
            raise ValueError("full-score model_revision is frozen")
        if (
            self.tokenizer_id != MAIN_LATENCY_TOKENIZER_ID
            or self.tokenizer_revision != MAIN_LATENCY_TOKENIZER_REVISION
        ):
            raise ValueError("full-score tokenizer identity is frozen")
        if self.engine_version != PUBLICATION_CAMPAIGN_ENGINE_VERSION:
            raise ValueError("full-score engine_version must be 0.27.1")
        if self.hardware_target != "aws-g6-l4":
            raise ValueError("full-score consumer benchmark hardware is frozen to L4")
        if self.gpu_memory_utilization not in (
            FULL_SCORE_GPU_MEMORY_UTILIZATION_CANDIDATES
        ):
            raise ValueError(
                "gpu_memory_utilization must be a qualified 0.70/0.75/0.80 value"
            )
        if self.max_model_len != FULL_SCORE_MIN_MAX_MODEL_LEN:
            raise ValueError(
                "full-score max_model_len is frozen to 32k prompt plus decode"
            )
        if self.max_num_seqs != FULL_SCORE_REQUEST_PARALLELISM:
            raise ValueError("full-score max_num_seqs is frozen to concurrency=4")
        if self.server_host != "127.0.0.1":
            raise ValueError("full-score server_host is frozen to loopback")
        if not 0 < self.server_port < 65_536:
            raise ValueError("server_port must be between 1 and 65535")
        for field_name in (
            "server_start_timeout_seconds",
            "request_timeout_seconds",
            "generator_timeout_seconds",
            "telemetry_interval_seconds",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be positive")
        if float(self.generator_timeout_seconds) != float(
            FULL_SCORE_DATABRICKS_TASK_TIMEOUT_SECONDS
        ):
            raise ValueError("full-score generator timeout is frozen to six hours")
        transfer = _json_mapping(self.kv_transfer_config, "kv_transfer_config")
        if set(transfer) != {
            "kv_connector",
            "kv_connector_extra_config",
            "kv_role",
        }:
            raise ValueError("full-score KV transfer config schema drift")
        if transfer.get("kv_connector") != "DocumentKVConnector":
            raise ValueError("full-score runtime requires DocumentKVConnector")
        if transfer.get("kv_role") != "kv_consumer":
            raise ValueError("full-score runtime requires kv_role=kv_consumer")
        if transfer.get("kv_connector") == "MultiConnector":
            raise ValueError("MultiConnector is unsupported in the full-score program")
        extra_config = transfer.get("kv_connector_extra_config")
        if not isinstance(extra_config, Mapping):
            raise ValueError("full-score KV transfer extra config must be an object")
        if set(extra_config) != {
            "document_kv.payload_cache_max_bytes",
            "document_kv.require_runtime_handshake",
        }:
            raise ValueError("full-score KV transfer extra config schema drift")
        if extra_config.get("document_kv.payload_cache_max_bytes") != 0:
            raise ValueError("full-score execution requires payload cache disabled")
        if extra_config.get("document_kv.require_runtime_handshake") is not True:
            raise ValueError("full-score execution requires the runtime handshake")
        object.__setattr__(self, "kv_transfer_config", transfer)
        _validate_hashed_install_spec(
            self.vllm_wheel_install_spec,
            project="vllm",
            expected_sha256=self.patched_vllm_wheel_sha256,
        )
        _validate_hashed_install_spec(
            self.flashinfer_wheel_install_spec,
            project="flashinfer-python",
            expected_sha256=self.patched_flashinfer_wheel_sha256,
        )
        install_origins = (
            (
                "vLLM",
                self.vllm_wheel_install_spec,
                self.patched_vllm_wheel_uri,
            ),
            (
                "FlashInfer",
                self.flashinfer_wheel_install_spec,
                self.patched_flashinfer_wheel_uri,
            ),
        )
        for label, install_spec, artifact_uri in install_origins:
            if _install_spec_uri(install_spec) != (
                _cluster_artifact_file_uri(artifact_uri)
            ):
                raise ValueError(
                    f"{label} install spec URI differs from staged artifact"
                )


@dataclass(frozen=True, slots=True)
class FullScoreGPUQualificationConfig:
    """Sealed qualification evidence required before any full-score GPU task."""

    campaign_id: str
    plan_uri: str
    evidence_uri: str
    evidence_file_sha256: str
    plan_record: Mapping[str, Any]
    evidence_record: Mapping[str, Any]
    artifact_pins: GPUQualificationArtifactPinsV2

    def __post_init__(self) -> None:
        for field_name in ("campaign_id", "plan_uri", "evidence_uri"):
            _require_nonempty(getattr(self, field_name), field_name)
        for field_name in ("plan_uri", "evidence_uri"):
            _require_shared_dbfs_path(getattr(self, field_name), field_name)
        _require_sha256(
            self.evidence_file_sha256,
            field_name="evidence_file_sha256",
        )
        if not isinstance(self.artifact_pins, GPUQualificationArtifactPinsV2):
            raise TypeError("artifact_pins must be GPUQualificationArtifactPinsV2")
        plan = _json_mapping(self.plan_record, "GPU qualification plan")
        evidence = _json_mapping(self.evidence_record, "GPU qualification evidence")
        canonical_evidence = (canonical_gpu_qualification_json(evidence) + "\n").encode(
            "utf-8"
        )
        if sha256(canonical_evidence).hexdigest() != self.evidence_file_sha256:
            raise ValueError("GPU qualification evidence raw file SHA-256 drift")
        selection = validate_gpu_qualification_evidence_v2_record(
            evidence,
            plan_record=plan,
            expected_campaign_id=self.campaign_id,
            expected_artifact_pins=self.artifact_pins,
        )
        _validate_full_score_gpu_selection(selection)
        object.__setattr__(self, "plan_record", plan)
        object.__setattr__(self, "evidence_record", evidence)

    @property
    def selection(self) -> GPUQualificationSelection:
        return validate_gpu_qualification_evidence_v2_record(
            self.evidence_record,
            plan_record=self.plan_record,
            expected_campaign_id=self.campaign_id,
            expected_artifact_pins=self.artifact_pins,
        )


@dataclass(frozen=True, slots=True)
class FullScoreWorkerBundleConfig:
    """Paths bound into each content-addressed persistent-worker manifest."""

    inventory_uri: str
    shard_plan_uri: str
    execution_plan_uri: str
    source_jsonl_uris: Mapping[str, str]
    durable_output_root: str
    ephemeral_root: str
    runtime: FullScoreRuntimeConfig
    runner_python_file: str
    runner_sha256: str
    package_wheel_uri: str
    package_wheel_sha256: str
    gpu_qualification: FullScoreGPUQualificationConfig
    authorization_scope: str = FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE

    def __post_init__(self) -> None:
        for field_name in (
            "inventory_uri",
            "shard_plan_uri",
            "execution_plan_uri",
            "durable_output_root",
            "ephemeral_root",
            "runner_python_file",
            "package_wheel_uri",
        ):
            _require_nonempty(getattr(self, field_name), field_name)
        if self.authorization_scope not in {
            FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE,
            FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
        }:
            raise ValueError("unsupported full-score worker authorization_scope")
        sources = dict(self.source_jsonl_uris)
        if set(sources) != set(SUPPORTED_V1_DATASETS):
            raise ValueError(
                "source_jsonl_uris must contain exactly all score datasets"
            )
        for dataset, uri in sources.items():
            _require_nonempty(uri, f"source_jsonl_uris.{dataset}")
        if not isinstance(self.runtime, FullScoreRuntimeConfig):
            raise TypeError("runtime must be FullScoreRuntimeConfig")
        if self.authorization_scope == FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE:
            for field_name in (
                "inventory_uri",
                "shard_plan_uri",
                "execution_plan_uri",
                "durable_output_root",
                "runner_python_file",
                "package_wheel_uri",
            ):
                _require_shared_dbfs_path(getattr(self, field_name), field_name)
            for dataset, uri in sources.items():
                _require_shared_dbfs_path(uri, f"source_jsonl_uris.{dataset}")
            _require_local_disk_path(self.ephemeral_root, "ephemeral_root")
            for field_name in (
                "runtime_lock_uri",
                "patched_vllm_wheel_uri",
                "patched_flashinfer_wheel_uri",
                "runtime_closure_manifest_uri",
            ):
                _require_shared_dbfs_path(
                    getattr(self.runtime, field_name),
                    f"runtime.{field_name}",
                )
            if (
                self.runtime.generator_factory != FULL_SCORE_GENERATOR_FACTORY
                or self.runtime.generator_version != FULL_SCORE_GENERATOR_VERSION
            ):
                raise ValueError(
                    "publication generator implementation/version is frozen"
                )
        if not isinstance(
            self.gpu_qualification,
            FullScoreGPUQualificationConfig,
        ):
            raise TypeError("gpu_qualification must be FullScoreGPUQualificationConfig")
        if self.runtime.gpu_memory_utilization != (
            self.gpu_qualification.selection.gpu_memory_utilization
        ):
            raise ValueError("runtime GMU differs from GPU qualification selection")
        pins = self.gpu_qualification.artifact_pins
        if (
            pins.package_wheel_sha256 != self.package_wheel_sha256
            or pins.patched_vllm_wheel_sha256 != self.runtime.patched_vllm_wheel_sha256
            or pins.patched_flashinfer_wheel_sha256
            != self.runtime.patched_flashinfer_wheel_sha256
            or pins.runtime_closure_manifest_sha256
            != self.runtime.runtime_closure_manifest_sha256
            or pins.runtime_lock_sha256 != self.runtime.runtime_lock_sha256
        ):
            raise ValueError("worker artifacts differ from GPU qualification pins")
        if pins.runner_sha256 == self.runner_sha256:
            raise ValueError(
                "GPU qualification and full-score runner identities must be distinct"
            )
        _require_sha256(self.runner_sha256, field_name="runner_sha256")
        if self.runner_sha256 != FULL_SCORE_RUNNER_SHA256:
            raise ValueError("runner_sha256 does not identify the frozen runner")
        _require_sha256(
            self.package_wheel_sha256,
            field_name="package_wheel_sha256",
        )
        object.__setattr__(self, "source_jsonl_uris", sources)


@dataclass(frozen=True, slots=True)
class DatabricksFullScoreJobConfig:
    """Databricks runs/submit settings for one task per persistent worker."""

    runner_python_file: str
    runner_sha256: str
    worker_payload_uri_template: str
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
    gpu_qualification: FullScoreGPUQualificationConfig
    runtime_venv_dir: str = "/local_disk0/cachet-full-score-runtime"
    run_name: str = "cachet-vllm-0271-full-score"
    task_key_prefix: str = "full_score_worker"
    run_timeout_seconds: int = FULL_SCORE_DATABRICKS_TASK_TIMEOUT_SECONDS
    task_max_retries: int = DEFAULT_DATABRICKS_TASK_MAX_RETRIES
    producer_node_type_id: str = FULL_SCORE_PRODUCER_NODE_TYPE_ID
    producer_zone_id: str = FULL_SCORE_PRODUCER_ZONE_ID
    consumer_node_type_id: str = FULL_SCORE_CONSUMER_NODE_TYPE_ID
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
            "patched_flashinfer_wheel_uri",
            "runtime_closure_manifest_uri",
            "runtime_venv_dir",
            "run_name",
            "task_key_prefix",
        ):
            _require_nonempty(getattr(self, field_name), field_name)
        if "{worker_index}" not in self.worker_payload_uri_template:
            raise ValueError("worker_payload_uri_template requires {worker_index}")
        for field_name in (
            "runner_python_file",
            "worker_payload_uri_template",
            "package_wheel_uri",
            "runtime_lock_uri",
            "patched_vllm_wheel_uri",
            "patched_flashinfer_wheel_uri",
            "runtime_closure_manifest_uri",
        ):
            _require_shared_dbfs_path(getattr(self, field_name), field_name)
        _require_local_disk_path(self.runtime_venv_dir, "runtime_venv_dir")
        _require_sha256(self.package_wheel_sha256, field_name="package_wheel_sha256")
        _require_sha256(self.runner_sha256, field_name="runner_sha256")
        if self.runner_sha256 != FULL_SCORE_RUNNER_SHA256:
            raise ValueError("runner_sha256 does not identify the frozen runner")
        _require_sha256(self.runtime_lock_sha256, field_name="runtime_lock_sha256")
        _require_sha256(
            self.patched_vllm_wheel_sha256,
            field_name="patched_vllm_wheel_sha256",
        )
        _require_sha256(
            self.patched_flashinfer_wheel_sha256,
            field_name="patched_flashinfer_wheel_sha256",
        )
        _require_sha256(
            self.runtime_closure_manifest_sha256,
            field_name="runtime_closure_manifest_sha256",
        )
        if self.runtime_lock_sha256 != VLLM_RUNTIME_BASE_LOCK_SHA256:
            raise ValueError("Databricks job runtime lock hash drift")
        if self.patched_vllm_wheel_sha256 != VLLM_PATCHED_WHEEL_SHA256:
            raise ValueError("Databricks job vLLM wheel hash drift")
        if self.patched_flashinfer_wheel_sha256 != FLASHINFER_PATCHED_WHEEL_SHA256:
            raise ValueError("Databricks job FlashInfer wheel hash drift")
        if self.runtime_closure_manifest_sha256 != RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256:
            raise ValueError("Databricks job runtime closure hash drift")
        if not isinstance(
            self.gpu_qualification,
            FullScoreGPUQualificationConfig,
        ):
            raise TypeError("gpu_qualification must be FullScoreGPUQualificationConfig")
        pins = self.gpu_qualification.artifact_pins
        if (
            pins.package_wheel_sha256 != self.package_wheel_sha256
            or pins.patched_vllm_wheel_sha256 != self.patched_vllm_wheel_sha256
            or pins.patched_flashinfer_wheel_sha256
            != self.patched_flashinfer_wheel_sha256
            or pins.runtime_closure_manifest_sha256
            != self.runtime_closure_manifest_sha256
            or pins.runtime_lock_sha256 != self.runtime_lock_sha256
        ):
            raise ValueError("Databricks artifacts differ from GPU qualification pins")
        if pins.runner_sha256 == self.runner_sha256:
            raise ValueError(
                "GPU qualification and full-score runner identities must be distinct"
            )
        _validated_databricks_run_timeout_seconds(self.run_timeout_seconds)
        if self.run_timeout_seconds != FULL_SCORE_DATABRICKS_TASK_TIMEOUT_SECONDS:
            raise ValueError("full-score Databricks timeout is frozen to six hours")
        _validated_databricks_task_max_retries(self.task_max_retries)
        if self.task_max_retries != 0:
            raise ValueError("full-score publication tasks are no-retry/idempotent")
        if self.producer_node_type_id != FULL_SCORE_PRODUCER_NODE_TYPE_ID:
            raise ValueError("full-score producers require g6e.4xlarge/L40S")
        if self.producer_zone_id != FULL_SCORE_PRODUCER_ZONE_ID:
            raise ValueError("full-score producers require the reviewed L40S zone")
        validate_aws_single_node_gpu_type(self.consumer_node_type_id)
        if self.consumer_node_type_id != DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE:
            raise ValueError("full-score consumers require g6.8xlarge/L4")
        if self.spark_version != DEFAULT_DATABRICKS_SPARK_VERSION:
            raise ValueError("full-score Databricks Runtime is frozen")
        if self.data_security_mode != "SINGLE_USER":
            raise ValueError("full-score tasks require a SINGLE_USER principal")
        object.__setattr__(
            self,
            "single_user_name",
            _validated_full_score_single_user_name(self.single_user_name),
        )
        if self.availability != "ON_DEMAND" or self.zone_id != "auto":
            raise ValueError("full-score tasks require the reviewed on-demand topology")
        object.__setattr__(self, "custom_tags", dict(self.custom_tags))


_FULL_SCORE_PHASE_SUBMISSION_AUTHORIZATION_ISSUER = object()
_FULL_SCORE_PHASE_AUTHORIZATION_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class FullScorePhaseSubmissionAuthorization:
    """Non-record authority over one atomically reserved full-score phase."""

    execution_plan_sha256: str
    wave_index: int
    phase: str
    ledger_path_sha256: str
    predecessor_prefix: DatabricksLedgerPrefix
    batch_authorization: DatabricksBatchReservationAuthorization
    attempt_id: str
    submit_payload_sha256: str
    intent_record_sha256: str
    workspace_host_sha256: str
    user_name_sha256: str

    def __init__(
        self,
        *,
        execution_plan_sha256: str,
        wave_index: int,
        phase: str,
        ledger_path_sha256: str,
        predecessor_prefix: DatabricksLedgerPrefix,
        batch_authorization: DatabricksBatchReservationAuthorization,
        attempt_id: str,
        submit_payload_sha256: str,
        intent_record_sha256: str,
        workspace_host_sha256: str,
        user_name_sha256: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _FULL_SCORE_PHASE_SUBMISSION_AUTHORIZATION_ISSUER:
            raise TypeError(
                "full-score submission authority requires atomic ledger admission"
            )
        object.__setattr__(
            self,
            "execution_plan_sha256",
            _require_sha256(
                execution_plan_sha256,
                field_name="execution_plan_sha256",
            ),
        )
        _validate_full_score_phase_position(wave_index, phase)
        object.__setattr__(self, "wave_index", wave_index)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _require_sha256(
                ledger_path_sha256,
                field_name="ledger_path_sha256",
            ),
        )
        if not isinstance(predecessor_prefix, DatabricksLedgerPrefix):
            raise TypeError("predecessor_prefix has the wrong type")
        if type(batch_authorization) is not DatabricksBatchReservationAuthorization:
            raise TypeError("batch_authorization has the wrong type")
        if (
            batch_authorization.predecessor_prefix != predecessor_prefix
            or batch_authorization.ledger_path_sha256 != ledger_path_sha256
            or batch_authorization.attempt_ids != (attempt_id,)
            or batch_authorization.submit_payload_sha256s != (submit_payload_sha256,)
        ):
            raise ValueError("full-score atomic batch authority binding drift")
        _require_nonempty(attempt_id, "attempt_id")
        _require_sha256(
            submit_payload_sha256,
            field_name="submit_payload_sha256",
        )
        _require_sha256(
            intent_record_sha256,
            field_name="intent_record_sha256",
        )
        object.__setattr__(self, "predecessor_prefix", predecessor_prefix)
        object.__setattr__(self, "batch_authorization", batch_authorization)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "submit_payload_sha256", submit_payload_sha256)
        object.__setattr__(self, "intent_record_sha256", intent_record_sha256)
        object.__setattr__(
            self,
            "workspace_host_sha256",
            _require_sha256(workspace_host_sha256, field_name="workspace_host_sha256"),
        )
        object.__setattr__(
            self,
            "user_name_sha256",
            _require_sha256(user_name_sha256, field_name="user_name_sha256"),
        )


@dataclass(frozen=True, slots=True, init=False)
class FullScorePhaseAuthorization:
    """Non-record authority issued after one exact phase is terminal."""

    execution_plan_sha256: str
    wave_index: int
    phase: str
    ledger_path_sha256: str
    predecessor_prefix: DatabricksLedgerPrefix
    ledger_prefix: DatabricksLedgerPrefix
    phase_lease_root: Path
    terminal_record_sha256: str
    causal_closure_sha256: str
    workspace_authority_closure_sha256: str
    workspace_host_sha256: str
    user_name_sha256: str

    def __init__(
        self,
        *,
        execution_plan_sha256: str,
        wave_index: int,
        phase: str,
        ledger_path_sha256: str,
        predecessor_prefix: DatabricksLedgerPrefix,
        ledger_prefix: DatabricksLedgerPrefix,
        phase_lease_root: str | Path,
        terminal_record_sha256: str,
        causal_closure_sha256: str,
        workspace_host_sha256: str,
        user_name_sha256: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _FULL_SCORE_PHASE_AUTHORIZATION_ISSUER:
            raise TypeError(
                "full-score phase authority requires direct terminal collection"
            )
        object.__setattr__(
            self,
            "execution_plan_sha256",
            _require_sha256(
                execution_plan_sha256,
                field_name="execution_plan_sha256",
            ),
        )
        _validate_full_score_phase_position(wave_index, phase)
        object.__setattr__(self, "wave_index", wave_index)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _require_sha256(
                ledger_path_sha256,
                field_name="ledger_path_sha256",
            ),
        )
        if not isinstance(predecessor_prefix, DatabricksLedgerPrefix) or not isinstance(
            ledger_prefix, DatabricksLedgerPrefix
        ):
            raise TypeError("full-score phase ledger prefixes have the wrong type")
        if (
            predecessor_prefix.ledger_id != ledger_prefix.ledger_id
            or predecessor_prefix.cap_cluster_hours != ledger_prefix.cap_cluster_hours
            or ledger_prefix.reservation_count
            != predecessor_prefix.reservation_count + 1
            or ledger_prefix.submission_receipt_count
            != predecessor_prefix.submission_receipt_count + 1
            or ledger_prefix.terminal_actual_count
            != predecessor_prefix.terminal_actual_count + 1
        ):
            raise ValueError("full-score phase prefix transition is invalid")
        object.__setattr__(self, "predecessor_prefix", predecessor_prefix)
        object.__setattr__(self, "ledger_prefix", ledger_prefix)
        normalized_lease_root = Path(phase_lease_root).expanduser().absolute()
        _require_no_symlink_ancestors(
            normalized_lease_root,
            label="full-score phase lease root",
            include_leaf=True,
        )
        if normalized_lease_root.exists() and (
            normalized_lease_root.is_symlink() or not normalized_lease_root.is_dir()
        ):
            raise ValueError("full-score phase lease root must be a real directory")
        object.__setattr__(self, "phase_lease_root", normalized_lease_root)
        object.__setattr__(
            self,
            "terminal_record_sha256",
            _require_sha256(
                terminal_record_sha256,
                field_name="terminal_record_sha256",
            ),
        )
        object.__setattr__(
            self,
            "causal_closure_sha256",
            _require_sha256(
                causal_closure_sha256,
                field_name="causal_closure_sha256",
            ),
        )
        object.__setattr__(
            self,
            "workspace_host_sha256",
            _require_sha256(workspace_host_sha256, field_name="workspace_host_sha256"),
        )
        object.__setattr__(
            self,
            "user_name_sha256",
            _require_sha256(user_name_sha256, field_name="user_name_sha256"),
        )
        object.__setattr__(
            self,
            "workspace_authority_closure_sha256",
            _canonical_sha256(
                {
                    "causal_closure_sha256": self.causal_closure_sha256,
                    "phase_lease_root_sha256": _canonical_sha256(
                        {
                            "domain": "cachet.full_score_phase_lease_root_authority.v1",
                            "phase_lease_root": str(self.phase_lease_root),
                        }
                    ),
                    "user_name_sha256": self.user_name_sha256,
                    "workspace_host_sha256": self.workspace_host_sha256,
                }
            ),
        )


class FullScoreShardLifecycle:
    """Role-aware state machine for phased or split producer/consumer waves."""

    _TRANSITIONS = {
        ("producer_pending", "generate_q8_kv"): "q8_kv_generated",
        ("q8_kv_generated", "commit_ready_shard"): "ready_shard_committed",
        ("consumer_pending", "verify_ready_shard"): "ready_shard_verified",
        ("ready_shard_verified", "baseline_inference"): "baseline_complete",
        ("baseline_complete", "vanilla_inference"): "vanilla_complete",
        ("vanilla_complete", "validate_paired_outputs"): "paired_validated",
        ("paired_validated", "commit_durable_evidence"): "evidence_committed",
        ("evidence_committed", "delete_ephemeral_q8_kv"): "ephemeral_deleted",
    }

    def __init__(self, role: str = "consumer") -> None:
        if role not in {"producer", "consumer"}:
            raise ValueError("full-score lifecycle role must be producer or consumer")
        self._role = role
        self._state = f"{role}_pending"
        self._events: list[str] = []

    @property
    def state(self) -> str:
        return self._state

    @property
    def events(self) -> tuple[str, ...]:
        return tuple(self._events)

    def advance(self, event: str) -> None:
        next_state = self._TRANSITIONS.get((self._state, event))
        if next_state is None:
            raise RuntimeError(
                f"invalid full-score lifecycle transition {self._state!r} -> {event!r}"
            )
        self._events.append(event)
        self._state = next_state


def build_full_score_execution_plan(
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    *,
    producer_workers: int = FULL_SCORE_MAX_WORKERS,
    consumer_workers: int = FULL_SCORE_MAX_WORKERS,
    max_shards_per_wave: int = FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE,
    max_backlog_bytes: int = FULL_SCORE_DEFAULT_MAX_BACKLOG_BYTES,
    scheduling_mode: str = "phased",
) -> dict[str, Any]:
    """Group shards into bounded waves and token-balance both phase queues.

    ``phased`` is the publication default: up to sixteen producers run, stop,
    then up to sixteen consumers run. ``split`` reserves both pools at once and
    therefore requires their sum to remain at most sixteen.
    """

    if not isinstance(inventory, FullScoreInventory):
        raise TypeError("inventory must be FullScoreInventory")
    validate_full_score_shard_plan(shard_plan, inventory=inventory)
    for name, value in (
        ("producer_workers", producer_workers),
        ("consumer_workers", consumer_workers),
        ("max_shards_per_wave", max_shards_per_wave),
        ("max_backlog_bytes", max_backlog_bytes),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        producer_workers > FULL_SCORE_MAX_WORKERS
        or consumer_workers > FULL_SCORE_MAX_WORKERS
    ):
        raise ValueError("producer and consumer pools are each capped at sixteen")
    if max_shards_per_wave > FULL_SCORE_MAX_WORKERS:
        raise ValueError("a publication wave may contain at most sixteen shards")
    if scheduling_mode not in {"phased", "split"}:
        raise ValueError("scheduling_mode must be phased or split")
    if (
        scheduling_mode == "split"
        and producer_workers + consumer_workers > FULL_SCORE_MAX_WORKERS
    ):
        raise ValueError("split producer+consumer concurrency cannot exceed sixteen")
    raw_shards = shard_plan.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("shard plan shards must be a non-empty array")
    shards = [_json_mapping(shard, "shard") for shard in raw_shards]
    item_by_key = {(item.dataset, item.example_id): item for item in inventory.items}
    bounded: list[dict[str, Any]] = []
    for shard in shards:
        segment_count = 0
        for raw_item in cast(list[Mapping[str, Any]], shard["items"]):
            key = (cast(str, raw_item["dataset"]), cast(str, raw_item["example_id"]))
            segment_count += item_by_key[key].segment_count
        upper_bound = (
            cast(int, shard["cache_prefix_tokens"])
            * FULL_SCORE_Q8_BYTES_PER_CACHE_PREFIX_TOKEN
            + segment_count * 4096
            + cast(int, shard["item_count"]) * 1024 * 1024
            + FULL_SCORE_READY_SHARD_METADATA_ALLOWANCE_BYTES
        )
        if upper_bound > max_backlog_bytes:
            raise ValueError("one shard exceeds the hard durable-backlog byte cap")
        bounded.append({**shard, "ready_bytes_upper_bound": upper_bound})
    waves: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(bounded):
        selected: list[dict[str, Any]] = []
        selected_bytes = 0
        while cursor < len(bounded) and len(selected) < max_shards_per_wave:
            candidate = bounded[cursor]
            candidate_bytes = cast(int, candidate["ready_bytes_upper_bound"])
            if selected and selected_bytes + candidate_bytes > max_backlog_bytes:
                break
            selected.append(candidate)
            selected_bytes += candidate_bytes
            cursor += 1
        if not selected:
            raise ValueError("could not place a shard within the backlog cap")
        producer_count = min(producer_workers, len(selected))
        consumer_count = min(consumer_workers, len(selected))
        producer_assignment = _balanced_shard_assignment(
            selected,
            worker_count=producer_count,
            weight_field="cache_prefix_tokens",
        )
        consumer_assignment = _balanced_shard_assignment(
            selected,
            worker_count=consumer_count,
            weight_field="natural_prompt_tokens",
        )
        wave_index = len(waves)
        waves.append(
            {
                "consumer_assignments": consumer_assignment,
                "consumer_workers": consumer_count,
                "max_backlog_bytes": max_backlog_bytes,
                "producer_assignments": producer_assignment,
                "producer_workers": producer_count,
                "ready_bytes_upper_bound": selected_bytes,
                "scheduling_mode": scheduling_mode,
                "shard_ids": [shard["shard_id"] for shard in selected],
                "shards": selected,
                "wave_index": wave_index,
            }
        )
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "inventory_sha256": inventory.inventory_sha256,
        "max_backlog_bytes": max_backlog_bytes,
        "max_live_gpu_tasks": FULL_SCORE_MAX_WORKERS,
        "next_wave_condition": "all_prior_wave_ready_shards_inferred_validated_and_deleted",
        "protocol": _full_score_protocol_record(),
        "record_type": FULL_SCORE_EXECUTION_PLAN_RECORD_TYPE,
        "scheduling_mode": scheduling_mode,
        "schema_version": FULL_SCORE_EXECUTION_PLAN_SCHEMA_VERSION,
        "shard_plan_sha256": shard_plan.get("closed_record_sha256"),
        "waves": waves,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def build_full_score_worker_payloads(
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    *,
    config: FullScoreWorkerBundleConfig,
) -> tuple[dict[str, Any], ...]:
    """Bind every wave's persistent producer and consumer task manifests."""

    if not isinstance(config, FullScoreWorkerBundleConfig):
        raise TypeError("config must be FullScoreWorkerBundleConfig")
    _validate_execution_plan(execution_plan, inventory=inventory, shard_plan=shard_plan)
    inventory_record = full_score_inventory_to_record(inventory)
    source_bindings = []
    sources_by_dataset = {source.dataset: source for source in inventory.sources}
    for dataset in SUPPORTED_V1_DATASETS:
        source = sources_by_dataset[dataset]
        source_bindings.append(
            {
                "byte_count": source.byte_count,
                "dataset": dataset,
                "identities_sha256": source.identities_sha256,
                "record_count": source.record_count,
                "source_jsonl_sha256": source.source_jsonl_sha256,
                "source_records_sha256": source.source_records_sha256,
                "uri": config.source_jsonl_uris[dataset],
            }
        )
    shard_by_id = {
        cast(str, shard["shard_id"]): shard
        for shard in cast(list[Mapping[str, Any]], shard_plan["shards"])
    }
    payloads: list[dict[str, Any]] = []
    for wave in cast(list[Mapping[str, Any]], execution_plan["waves"]):
        wave_index = cast(int, wave["wave_index"])
        for role, assignment_key in (
            ("producer", "producer_assignments"),
            ("consumer", "consumer_assignments"),
        ):
            for assignment in cast(list[Mapping[str, Any]], wave[assignment_key]):
                worker_index = cast(int, assignment["worker_index"])
                worker_shards = [
                    _json_mapping(shard_by_id[shard_id], "shard")
                    for shard_id in cast(list[str], assignment["shard_ids"])
                ]
                bootstrap_artifacts = {
                    "locked_runtime_identity_sha256": _locked_runtime_identity_sha256(
                        runner_sha256=config.runner_sha256,
                        package_wheel_sha256=config.package_wheel_sha256,
                        runtime_lock_sha256=config.runtime.runtime_lock_sha256,
                        patched_vllm_wheel_sha256=(
                            config.runtime.patched_vllm_wheel_sha256
                        ),
                        patched_flashinfer_wheel_sha256=(
                            config.runtime.patched_flashinfer_wheel_sha256
                        ),
                        runtime_closure_manifest_sha256=(
                            config.runtime.runtime_closure_manifest_sha256
                        ),
                    ),
                    "package_wheel_sha256": config.package_wheel_sha256,
                    "package_wheel_uri": config.package_wheel_uri,
                    "patched_vllm_wheel_sha256": (
                        config.runtime.patched_vllm_wheel_sha256
                    ),
                    "patched_vllm_wheel_uri": config.runtime.patched_vllm_wheel_uri,
                    "patched_flashinfer_wheel_sha256": (
                        config.runtime.patched_flashinfer_wheel_sha256
                    ),
                    "patched_flashinfer_wheel_uri": (
                        config.runtime.patched_flashinfer_wheel_uri
                    ),
                    "runner_python_file": config.runner_python_file,
                    "runner_sha256": config.runner_sha256,
                    "runtime_closure_manifest_sha256": (
                        config.runtime.runtime_closure_manifest_sha256
                    ),
                    "runtime_closure_manifest_uri": (
                        config.runtime.runtime_closure_manifest_uri
                    ),
                    "runtime_lock_sha256": config.runtime.runtime_lock_sha256,
                    "runtime_lock_uri": config.runtime.runtime_lock_uri,
                }
                record: dict[str, Any] = {
                    "authorization_scope": config.authorization_scope,
                    "bootstrap_artifacts": bootstrap_artifacts,
                    "closed_record_sha256": "",
                    "durable_output_root": config.durable_output_root,
                    "ephemeral_root": config.ephemeral_root,
                    "execution_plan": {
                        "closed_record_sha256": execution_plan["closed_record_sha256"],
                        "record_type": execution_plan["record_type"],
                        "schema_version": execution_plan["schema_version"],
                        "uri": config.execution_plan_uri,
                    },
                    "generator_artifact_contract": (
                        _generator_artifact_contract_record(config.runtime)
                    ),
                    "gpu_qualification": _gpu_qualification_binding_record(
                        config.gpu_qualification
                    ),
                    "inventory": {
                        "closed_record_sha256": inventory.inventory_sha256,
                        "record_type": inventory_record["record_type"],
                        "schema_version": inventory_record["schema_version"],
                        "uri": config.inventory_uri,
                    },
                    "protocol": _full_score_protocol_record(),
                    "record_type": FULL_SCORE_WORKER_PAYLOAD_RECORD_TYPE,
                    "role": role,
                    "runtime": _runtime_record(config.runtime),
                    "schema_version": FULL_SCORE_WORKER_PAYLOAD_SCHEMA_VERSION,
                    "scorers": _scorer_contract_record(),
                    "shard_plan": {
                        "closed_record_sha256": shard_plan["closed_record_sha256"],
                        "plan_id": shard_plan.get("plan_id"),
                        "record_type": shard_plan.get("record_type"),
                        "schema_version": shard_plan.get("schema_version"),
                        "uri": config.shard_plan_uri,
                    },
                    "shards": worker_shards,
                    "source_jsonls": source_bindings,
                    "wave": dict(wave),
                    "wave_index": wave_index,
                    "worker_index": worker_index,
                }
                record["closed_record_sha256"] = _closed_record_sha256(record)
                payloads.append(record)
    return tuple(payloads)


def validate_full_score_worker_payload(
    record: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
) -> None:
    """Reject worker, plan, inventory, scorer, or runtime binding drift."""

    if not isinstance(record, Mapping):
        raise TypeError("worker payload must be an object")
    if record.get("record_type") != FULL_SCORE_WORKER_PAYLOAD_RECORD_TYPE:
        raise ValueError("unsupported full-score worker payload record_type")
    if record.get("schema_version") != FULL_SCORE_WORKER_PAYLOAD_SCHEMA_VERSION:
        raise ValueError("unsupported full-score worker payload schema_version")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("full-score worker payload closed_record_sha256 is invalid")
    if record.get("authorization_scope") not in {
        FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE,
        FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
    }:
        raise ValueError("worker authorization_scope is invalid")
    validate_full_score_shard_plan(shard_plan, inventory=inventory)
    _validate_execution_plan(execution_plan, inventory=inventory, shard_plan=shard_plan)
    inventory_binding = _required_mapping(record, "inventory")
    plan_binding = _required_mapping(record, "shard_plan")
    execution_binding = _required_mapping(record, "execution_plan")
    if inventory_binding.get("closed_record_sha256") != inventory.inventory_sha256:
        raise ValueError("worker inventory hash drift")
    if plan_binding.get("closed_record_sha256") != shard_plan.get(
        "closed_record_sha256"
    ):
        raise ValueError("worker shard-plan hash drift")
    if execution_binding.get("closed_record_sha256") != execution_plan.get(
        "closed_record_sha256"
    ):
        raise ValueError("worker execution-plan hash drift")
    if record.get("protocol") != _full_score_protocol_record():
        raise ValueError("worker full-score protocol drift")
    if not _json_type_exact_equal(
        record.get("scorers"),
        _scorer_contract_record(),
    ):
        raise ValueError("worker scorer/parser identity drift")
    runtime_binding = _required_mapping(record, "runtime")
    runtime = _runtime_from_record(runtime_binding)
    if record.get("generator_artifact_contract") != (
        _generator_artifact_contract_record(runtime)
    ):
        raise ValueError("worker generator artifact-contract drift")
    qualification_binding = _required_mapping(record, "gpu_qualification")
    _validate_gpu_qualification_binding_shape(
        qualification_binding,
        runtime=runtime,
    )
    bootstrap_binding = _required_mapping(record, "bootstrap_artifacts")
    _validate_bootstrap_artifact_binding(
        bootstrap_binding,
        runtime=runtime,
    )
    qualification_pins = GPUQualificationArtifactPinsV2(
        **_json_mapping(
            _required_mapping(qualification_binding, "artifact_pins"),
            "GPU artifact pins",
        )
    )
    if qualification_pins.package_wheel_sha256 != _required_string(
        bootstrap_binding,
        "package_wheel_sha256",
    ):
        raise ValueError("worker package wheel differs from GPU qualification pins")
    if qualification_pins.runner_sha256 == _required_string(
        bootstrap_binding,
        "runner_sha256",
    ):
        raise ValueError(
            "GPU qualification and full-score runner identities must be distinct"
        )
    worker_index = record.get("worker_index")
    if type(worker_index) is not int or not 0 <= worker_index < FULL_SCORE_MAX_WORKERS:
        raise ValueError("worker_index is invalid")
    role = record.get("role")
    wave_index = record.get("wave_index")
    if role not in {"producer", "consumer"} or type(wave_index) is not int:
        raise ValueError("worker role/wave identity is invalid")
    waves = cast(list[Mapping[str, Any]], execution_plan["waves"])
    if not 0 <= wave_index < len(waves):
        raise ValueError("worker wave_index is invalid")
    wave = waves[wave_index]
    assignment_key = f"{role}_assignments"
    assignments = cast(list[Mapping[str, Any]], wave[assignment_key])
    assignment = next(
        (item for item in assignments if item.get("worker_index") == worker_index),
        None,
    )
    if assignment is None:
        raise ValueError("worker is not assigned in its wave")
    shard_by_id = {
        cast(str, shard["shard_id"]): dict(shard)
        for shard in cast(list[Mapping[str, Any]], shard_plan["shards"])
    }
    expected_shards = [
        shard_by_id[shard_id] for shard_id in cast(list[str], assignment["shard_ids"])
    ]
    observed_shards = record.get("shards")
    if observed_shards != expected_shards:
        raise ValueError("worker shard assignment drift")
    _validate_source_bindings(record, inventory)
    if record.get("authorization_scope") == FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE:
        if (
            runtime.generator_factory != FULL_SCORE_GENERATOR_FACTORY
            or runtime.generator_version != FULL_SCORE_GENERATOR_VERSION
        ):
            raise ValueError("publication generator implementation/version is frozen")
        _require_shared_dbfs_path(
            _required_string(inventory_binding, "uri"),
            "worker inventory URI",
        )
        _require_shared_dbfs_path(
            _required_string(plan_binding, "uri"),
            "worker shard-plan URI",
        )
        _require_shared_dbfs_path(
            _required_string(execution_binding, "uri"),
            "worker execution-plan URI",
        )
        _require_shared_dbfs_path(
            _required_string(record, "durable_output_root"),
            "worker durable output root",
        )
        _require_local_disk_path(
            _required_string(record, "ephemeral_root"),
            "worker ephemeral root",
        )
        for raw_source in cast(list[Mapping[str, Any]], record["source_jsonls"]):
            source = _json_mapping(raw_source, "worker source binding")
            _require_shared_dbfs_path(
                _required_string(source, "uri"),
                "worker source URI",
            )
        for field_name in (
            "runtime_lock_uri",
            "patched_vllm_wheel_uri",
            "patched_flashinfer_wheel_uri",
            "runtime_closure_manifest_uri",
        ):
            _require_shared_dbfs_path(
                _required_string(runtime_binding, field_name),
                f"worker runtime {field_name}",
            )
        for field_name in (
            "runner_python_file",
            "package_wheel_uri",
            "runtime_lock_uri",
            "patched_vllm_wheel_uri",
            "patched_flashinfer_wheel_uri",
            "runtime_closure_manifest_uri",
        ):
            _require_shared_dbfs_path(
                _required_string(bootstrap_binding, field_name),
                f"worker bootstrap {field_name}",
            )
        for field_name in ("plan_uri", "evidence_uri"):
            _require_shared_dbfs_path(
                _required_string(qualification_binding, field_name),
                f"worker qualification {field_name}",
            )


def full_score_inventory_from_record(record: Mapping[str, Any]) -> FullScoreInventory:
    """Reconstruct and independently validate a serialized complete inventory."""

    if not isinstance(record, Mapping):
        raise TypeError("full-score inventory must be an object")
    raw_sources = record.get("sources")
    raw_items = record.get("items")
    policy = record.get("input_length_policy")
    if not isinstance(raw_sources, list) or not isinstance(raw_items, list):
        raise ValueError("full-score inventory sources/items must be arrays")
    if not isinstance(policy, Mapping):
        raise ValueError("full-score inventory input_length_policy must be an object")
    sources = tuple(
        FullScoreDatasetSource(**_json_mapping(source, "inventory source"))
        for source in raw_sources
    )
    items: list[FullScoreInventoryItem] = []
    for raw_item in raw_items:
        item_record = _json_mapping(raw_item, "inventory item")
        identity_sha256 = item_record.pop("identity_sha256", None)
        item = FullScoreInventoryItem(**item_record)
        if identity_sha256 != item.identity_sha256:
            raise ValueError("full-score inventory item identity hash drift")
        items.append(item)
    inventory = FullScoreInventory(
        sources=sources,
        items=tuple(items),
        max_natural_prompt_tokens=_required_int(
            policy,
            "max_natural_prompt_tokens",
        ),
    )
    validate_full_score_inventory_record(record, inventory=inventory)
    return inventory


def render_full_score_worker_command_plan(
    worker_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Render exact subprocess argv without touching files or launching GPUs."""

    runtime = _runtime_from_record(_required_mapping(worker_payload, "runtime"))
    bootstrap = _required_mapping(worker_payload, "bootstrap_artifacts")
    worker_index = _required_int(worker_payload, "worker_index")
    wave_index = _required_int(worker_payload, "wave_index")
    role = _required_string(worker_payload, "role")
    ephemeral_root = Path(_required_string(worker_payload, "ephemeral_root"))
    durable_root = Path(_required_string(worker_payload, "durable_output_root"))
    shards = worker_payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("worker payload shards must be a non-empty array")
    rendered_shards = []
    for raw_shard in shards:
        shard = _json_mapping(raw_shard, "worker shard")
        shard_id = _required_string(shard, "shard_id")
        shard_ephemeral = ephemeral_root / f"worker-{worker_index:02d}" / shard_id
        shard_durable = durable_root / f"worker-{worker_index:02d}" / shard_id
        datasets = sorted(
            {cast(str, item["dataset"]) for item in cast(list[Any], shard["items"])}
        )
        if role == "producer":
            rendered_shards.append(
                {
                    "datasets": datasets,
                    "ephemeral_dir": str(shard_ephemeral),
                    "operation": "persistent_generator_api_generate_and_close_ready_shard",
                    "ready_dir": str(_ready_shard_dir(worker_payload, shard_id)),
                    "shard_id": shard_id,
                }
            )
        elif role == "consumer":
            ready_dir = _ready_shard_dir(worker_payload, shard_id)
            ready_inputs = {
                dataset: ready_dir / "inputs" / f"{dataset}.jsonl"
                for dataset in datasets
            }
            ready_enriched = {
                dataset: ready_dir / "enriched" / f"{dataset}.jsonl"
                for dataset in datasets
            }
            rendered_shards.append(
                {
                    "baseline": _benchmark_command(
                        runtime,
                        method="baseline_prefill",
                        shard_id=shard_id,
                        dataset_paths=ready_inputs,
                        output_json=shard_ephemeral / "baseline.json",
                    ),
                    "durable_dir": str(shard_durable),
                    "ephemeral_dir": str(shard_ephemeral),
                    "ready_dir": str(ready_dir),
                    "shard_id": shard_id,
                    "vanilla": _benchmark_command(
                        runtime,
                        method="vanilla_prefill",
                        shard_id=shard_id,
                        dataset_paths=ready_enriched,
                        output_json=shard_ephemeral / "vanilla.json",
                    ),
                }
            )
        else:
            raise ValueError("worker role must be producer or consumer")
    return {
        "protocol": _full_score_protocol_record(),
        "role": role,
        "runtime_verifier": _runtime_verifier_command(
            runtime,
            package_wheel_uri=_required_string(bootstrap, "package_wheel_uri"),
            package_wheel_sha256=_required_string(
                bootstrap,
                "package_wheel_sha256",
            ),
        ),
        "server": _vllm_server_command(runtime) if role == "consumer" else None,
        "shards": rendered_shards,
        "wave_index": wave_index,
        "worker_index": worker_index,
    }


def _build_databricks_full_score_run_submit_payload(
    config: DatabricksFullScoreJobConfig,
    worker_payloads: Sequence[Mapping[str, Any]],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    prior_wave_completion: Mapping[str, Any] | None = None,
    producer_phase_completion: Mapping[str, Any] | None = None,
    producer_phase_completion_uri: str | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorization: object | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
    budget_admission: Mapping[str, Any] | None = None,
    qualification_launch_authorization: object | None = None,
    idempotency_attempt_id: str | None = None,
    publication_authorizing: bool,
    allow_unadmitted_publication_candidate: bool = False,
) -> dict[str, Any]:
    """Render one exact, independently bounded publication phase submission."""

    if not isinstance(config, DatabricksFullScoreJobConfig):
        raise TypeError("config must be DatabricksFullScoreJobConfig")
    if allow_unadmitted_publication_candidate and not publication_authorizing:
        raise ValueError(
            "only a publication-authorizing render may form an unadmitted candidate"
        )
    payloads = tuple(worker_payloads)
    if not payloads:
        raise ValueError("Databricks full-score job requires worker payloads")
    wave_indices = {_required_int(payload, "wave_index") for payload in payloads}
    if len(wave_indices) != 1:
        raise ValueError("one runs/submit payload may contain exactly one wave")
    wave_index = next(iter(wave_indices))
    roles = {_required_string(payload, "role") for payload in payloads}
    if len(roles) != 1:
        raise ValueError("one runs/submit payload may contain exactly one phase")
    phase = next(iter(roles))
    _validate_publication_full_score_inputs(
        inventory,
        shard_plan,
        execution_plan,
    )
    if publication_authorizing:
        authorized_selection = require_gpu_qualification_launch_authorization(
            qualification_launch_authorization,
            expected_plan_sha256=config.gpu_qualification.selection.plan_sha256,
            expected_evidence_file_sha256=(
                config.gpu_qualification.evidence_file_sha256
            ),
        )
        _validate_full_score_gpu_selection(authorized_selection)
        if authorized_selection != config.gpu_qualification.selection:
            raise ValueError(
                "GPU qualification launch authority selection differs from job config"
            )
    elif qualification_launch_authorization is not None:
        raise ValueError("local fixture preview cannot consume launch authority")
    for payload in payloads:
        validate_full_score_worker_payload(
            payload,
            inventory=inventory,
            shard_plan=shard_plan,
            execution_plan=execution_plan,
        )
        if payload.get("authorization_scope") != (
            FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
        ):
            raise ValueError(
                "Databricks publication rendering rejects local-fixture payloads"
            )
    execution_plan_sha256 = _required_string(
        _required_mapping(payloads[0], "execution_plan"),
        "closed_record_sha256",
    )
    _validate_budget_execution_plan(execution_plan)
    if execution_plan.get("closed_record_sha256") != execution_plan_sha256:
        raise ValueError("Databricks wave execution-plan binding drift")
    _validate_complete_phase_payload_set(payloads, phase=phase)
    if wave_index == 0:
        if prior_wave_completion is not None:
            raise ValueError("wave zero must not declare prior-wave completion")
        if budget_admission is not None:
            raise ValueError("wave zero uses only the sealed >=35 token/s pilot gate")
        wave = _required_mapping(payloads[0], "wave")
        if len(cast(list[Any], wave.get("shard_ids"))) != (
            FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE
        ):
            raise ValueError("wave zero must be the frozen sixteen-shard pilot")
    else:
        _validate_prior_wave_completion(
            prior_wave_completion,
            inventory=inventory,
            shard_plan=shard_plan,
            execution_plan=execution_plan,
            replay_raw_evidence=False,
            expected_wave_index=wave_index - 1,
            expected_execution_plan_sha256=execution_plan_sha256,
            remote_consumer_authorization=remote_consumer_authorization,
            compact_artifact_resolver=compact_artifact_resolver,
        )
        expected_reservation = full_score_wave_worst_case_gpu_hours(
            payloads,
            task_timeout_seconds=config.run_timeout_seconds,
        )
        if allow_unadmitted_publication_candidate:
            if budget_admission is not None:
                raise ValueError(
                    "an unadmitted candidate must not consume a P90 admission"
                )
        else:
            _validate_live_p90_budget_admission(
                budget_admission,
                execution_plan=execution_plan,
                inventory=inventory,
                shard_plan=shard_plan,
                expected_execution_plan_sha256=execution_plan_sha256,
                expected_next_wave_index=wave_index,
                expected_next_phase=phase,
                expected_next_wave_reserved_gpu_hours=expected_reservation,
                require_publication=publication_authorizing,
                compact_artifact_resolver=compact_artifact_resolver,
            )
        if publication_authorizing:
            _validate_prior_wave_completion(
                prior_wave_completion,
                inventory=inventory,
                shard_plan=shard_plan,
                execution_plan=execution_plan,
                replay_raw_evidence=True,
                expected_wave_index=wave_index - 1,
                expected_execution_plan_sha256=execution_plan_sha256,
                remote_consumer_authorization=remote_consumer_authorization,
                compact_artifact_resolver=compact_artifact_resolver,
            )
    if phase == "producer":
        if (
            producer_phase_completion is not None
            or producer_phase_completion_uri is not None
            or remote_ready_authorization is not None
        ):
            raise ValueError("producer phase cannot consume its own completion record")
    else:
        if producer_phase_completion_uri is None:
            raise ValueError("consumer phase requires a producer-phase completion URI")
        completion_uri = _require_nonempty(
            producer_phase_completion_uri,
            "producer_phase_completion_uri",
        )
        _validate_producer_phase_completion(
            producer_phase_completion,
            execution_plan=execution_plan,
            expected_wave_index=wave_index,
        )
        if remote_ready_authorization is None:
            completion_file = _governed_compact_file(
                completion_uri,
                "producer_phase_completion_uri",
                compact_artifact_resolver,
            )
            if (
                _json_object(
                    completion_file.read_bytes(),
                    "producer-phase completion file",
                )
                != producer_phase_completion
            ):
                raise ValueError("producer-phase completion URI/content binding drift")
            if publication_authorizing:
                _validate_governed_producer_ready_phase(
                    producer_phase_completion,
                    payloads=payloads,
                    inventory=inventory,
                    shard_plan=shard_plan,
                    execution_plan=execution_plan,
                )
        else:
            if not publication_authorizing:
                raise ValueError(
                    "local fixture rendering cannot consume remote tree authority"
                )
            from document_kv_cache.full_score_remote_control import (
                require_full_score_remote_ready_authorization,
            )

            durable_roots = {
                _required_string(payload, "durable_output_root") for payload in payloads
            }
            if len(durable_roots) != 1:
                raise ValueError("consumer payloads do not share one durable root")
            require_full_score_remote_ready_authorization(
                remote_ready_authorization,
                execution_plan_sha256=execution_plan_sha256,
                wave_index=wave_index,
                durable_output_root=next(iter(durable_roots)),
                completion_uri=completion_uri,
                completion_record=cast(Mapping[str, Any], producer_phase_completion),
            )
    tasks = []
    ordered_payloads = sorted(
        payloads,
        key=lambda payload: (
            _required_int(payload, "wave_index"),
            0 if payload.get("role") == "producer" else 1,
            _required_int(payload, "worker_index"),
        ),
    )
    for payload in ordered_payloads:
        worker_index = _required_int(payload, "worker_index")
        wave_index = _required_int(payload, "wave_index")
        role = _required_string(payload, "role")
        closure_sha256 = _require_sha256(
            payload.get("closed_record_sha256"), field_name="worker payload closure"
        )
        if closure_sha256 != _closed_record_sha256(payload):
            raise ValueError("worker payload is not closed")
        runtime_binding = _required_mapping(payload, "runtime")
        bootstrap_binding = _required_mapping(payload, "bootstrap_artifacts")
        qualification_binding = _required_mapping(payload, "gpu_qualification")
        expected_runtime_artifacts = {
            "runtime_lock_sha256": config.runtime_lock_sha256,
            "runtime_lock_uri": config.runtime_lock_uri,
            "patched_vllm_wheel_sha256": config.patched_vllm_wheel_sha256,
            "patched_vllm_wheel_uri": config.patched_vllm_wheel_uri,
            "patched_flashinfer_wheel_sha256": (config.patched_flashinfer_wheel_sha256),
            "patched_flashinfer_wheel_uri": config.patched_flashinfer_wheel_uri,
            "runtime_closure_manifest_sha256": (config.runtime_closure_manifest_sha256),
            "runtime_closure_manifest_uri": config.runtime_closure_manifest_uri,
            "python_executable": f"{config.runtime_venv_dir}/bin/python",
            "vllm_wheel_install_spec": (
                "vllm @ file://"
                f"{_databricks_worker_mount_path(config.patched_vllm_wheel_uri)}"
                f"#sha256={config.patched_vllm_wheel_sha256}"
            ),
            "flashinfer_wheel_install_spec": (
                "flashinfer-python @ file://"
                f"{_databricks_worker_mount_path(config.patched_flashinfer_wheel_uri)}"
                f"#sha256={config.patched_flashinfer_wheel_sha256}"
            ),
        }
        if any(
            runtime_binding.get(key) != value
            for key, value in expected_runtime_artifacts.items()
        ):
            raise ValueError("Databricks bootstrap artifacts drift from worker payload")
        expected_bootstrap_artifacts = {
            "locked_runtime_identity_sha256": _locked_runtime_identity_sha256(
                runner_sha256=config.runner_sha256,
                package_wheel_sha256=config.package_wheel_sha256,
                runtime_lock_sha256=config.runtime_lock_sha256,
                patched_vllm_wheel_sha256=config.patched_vllm_wheel_sha256,
                patched_flashinfer_wheel_sha256=(
                    config.patched_flashinfer_wheel_sha256
                ),
                runtime_closure_manifest_sha256=(
                    config.runtime_closure_manifest_sha256
                ),
            ),
            "package_wheel_sha256": config.package_wheel_sha256,
            "package_wheel_uri": config.package_wheel_uri,
            "patched_vllm_wheel_sha256": config.patched_vllm_wheel_sha256,
            "patched_vllm_wheel_uri": config.patched_vllm_wheel_uri,
            "patched_flashinfer_wheel_sha256": (config.patched_flashinfer_wheel_sha256),
            "patched_flashinfer_wheel_uri": config.patched_flashinfer_wheel_uri,
            "runner_python_file": config.runner_python_file,
            "runner_sha256": config.runner_sha256,
            "runtime_closure_manifest_sha256": (config.runtime_closure_manifest_sha256),
            "runtime_closure_manifest_uri": config.runtime_closure_manifest_uri,
            "runtime_lock_sha256": config.runtime_lock_sha256,
            "runtime_lock_uri": config.runtime_lock_uri,
        }
        if dict(bootstrap_binding) != expected_bootstrap_artifacts:
            raise ValueError("runner/package bootstrap binding drift")
        if dict(qualification_binding) != _gpu_qualification_binding_record(
            config.gpu_qualification
        ):
            raise ValueError("GPU qualification binding drift")
        selection = config.gpu_qualification.selection
        if runtime_binding.get("gpu_memory_utilization") != (
            selection.gpu_memory_utilization
        ):
            raise ValueError("runtime GMU differs from sealed qualification")
        expected_sha256 = sha256(_canonical_pretty_json_bytes(payload)).hexdigest()
        worker_uri = _full_score_worker_payload_uri(config, payload)
        task_key = (
            f"{config.task_key_prefix}_wave_{wave_index:03d}_{role}_{worker_index:02d}"
        )
        task: dict[str, Any] = {
            "max_retries": config.task_max_retries,
            "new_cluster": _build_full_score_role_cluster(config, role=role),
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
                    "run-worker",
                    "--worker-payload-json",
                    worker_uri,
                    "--expected-worker-payload-sha256",
                    expected_sha256,
                ],
                "python_file": config.runner_python_file,
            },
            "task_key": task_key,
            "timeout_seconds": config.run_timeout_seconds,
        }
        if role == "consumer":
            parameters = cast(list[str], task["spark_python_task"]["parameters"])
            parameters.extend(
                [
                    "--producer-phase-completion-json",
                    completion_uri,
                    "--expected-producer-phase-completion-sha256",
                    _required_string(
                        cast(Mapping[str, Any], producer_phase_completion),
                        "closed_record_sha256",
                    ),
                ]
            )
        tasks.append(task)
    if len(tasks) > FULL_SCORE_MAX_WORKERS:
        raise ValueError("a full-score phase exceeds sixteen live GPU tasks")
    result = {
        "run_name": f"{config.run_name}-wave-{wave_index:03d}-{phase}",
        "tasks": tasks,
        "timeout_seconds": config.run_timeout_seconds,
    }
    if publication_authorizing:
        if idempotency_attempt_id is None:
            raise ValueError(
                "publication rendering requires a deterministic attempt_id"
            )
        result = bind_databricks_run_idempotency_token(
            result,
            attempt_id=idempotency_attempt_id,
        )
    elif idempotency_attempt_id is not None:
        raise ValueError("local fixture preview cannot bind a publication token")
    if (
        publication_authorizing
        and wave_index > 0
        and not allow_unadmitted_publication_candidate
    ):
        admission = cast(Mapping[str, Any], budget_admission)
        _snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(
            result
        )
        if (
            admission.get("next_submit_payload_sha256")
            != sha256(canonical_payload).hexdigest()
        ):
            raise ValueError("live P90 admission is not bound to the rendered payload")
    return result


def _full_score_worker_payload_uri(
    config: DatabricksFullScoreJobConfig,
    worker_payload: Mapping[str, Any],
) -> str:
    worker_index = _required_int(worker_payload, "worker_index")
    wave_index = _required_int(worker_payload, "wave_index")
    role = _required_string(worker_payload, "role")
    return config.worker_payload_uri_template.format(
        worker_index=f"wave-{wave_index:03d}-{role}-{worker_index:02d}"
    )


def _build_governed_full_score_live_p90_submit_candidate(
    config: DatabricksFullScoreJobConfig,
    worker_payloads: Sequence[Mapping[str, Any]],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    attempt_id: str,
    prior_wave_completion: Mapping[str, Any],
    ledger_path: str | Path,
    predecessor_authorization: object,
    producer_phase_completion: Mapping[str, Any] | None = None,
    producer_phase_completion_uri: str | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render exact nonzero-wave bytes for a governed P90 gate to bind.

    The returned mapping is deliberately not a launch authorization.  Governed
    reserve, submit, and replay APIs still require an immutable P90 admission
    built from these exact bytes and a subsequent publication render.
    """

    payloads = tuple(worker_payloads)
    wave_indices = {_required_int(payload, "wave_index") for payload in payloads}
    if wave_indices == {0}:
        raise ValueError("wave zero does not use a live P90 submit candidate")
    if len(wave_indices) != 1:
        raise ValueError("one P90 candidate may contain exactly one wave")
    wave_index = next(iter(wave_indices))
    phase = _required_string(payloads[0], "role")
    remote_by_wave = _remote_consumer_authorizations_by_wave(
        ()
        if remote_consumer_authorizations is None
        else remote_consumer_authorizations,
        execution_plan=execution_plan,
    )
    if remote_by_wave and set(remote_by_wave) != set(range(wave_index)):
        raise ValueError(
            "nonzero candidate requires every completed-wave remote authority"
        )
    opening_predecessor = _require_full_score_phase_predecessor_authorization(
        predecessor_authorization,
        execution_plan=execution_plan,
        ledger_path=ledger_path,
        wave_index=wave_index,
        phase=phase,
        latency_execution_plan_record=latency_execution_plan_record,
    )
    result = _build_databricks_full_score_run_submit_payload(
        config,
        payloads,
        inventory=inventory,
        shard_plan=shard_plan,
        execution_plan=execution_plan,
        qualification_launch_authorization=qualification_launch_authorization,
        prior_wave_completion=prior_wave_completion,
        producer_phase_completion=producer_phase_completion,
        producer_phase_completion_uri=producer_phase_completion_uri,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorization=remote_by_wave.get(wave_index - 1),
        compact_artifact_resolver=compact_artifact_resolver,
        budget_admission=None,
        idempotency_attempt_id=attempt_id,
        publication_authorizing=True,
        allow_unadmitted_publication_candidate=True,
    )
    closing_predecessor = _require_full_score_phase_predecessor_authorization(
        predecessor_authorization,
        execution_plan=execution_plan,
        ledger_path=ledger_path,
        wave_index=wave_index,
        phase=phase,
        latency_execution_plan_record=latency_execution_plan_record,
    )
    if closing_predecessor != opening_predecessor:
        raise ValueError("P90 candidate predecessor changed during rendering")
    return result


def prepare_governed_full_score_live_p90_phase_submission(
    budget_admission_path: str | Path,
    config: DatabricksFullScoreJobConfig,
    worker_payloads: Sequence[Mapping[str, Any]],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    attempt_id: str,
    completed_block_paths: Sequence[str | Path],
    prior_wave_completion: Mapping[str, Any],
    ledger_path: str | Path,
    predecessor_authorization: object,
    producer_phase_completion: Mapping[str, Any] | None = None,
    producer_phase_completion_uri: str | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
    compact_artifact_publisher: FullScoreCompactArtifactPublisher | None = None,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Crash-safely prepare a nonzero phase's P90 gate and final submit bytes.

    A deterministic launch-shaped candidate exists only inside this function.
    The function publishes and rereads the gate, renders again through the
    ordinary publication path, and returns ``(submit_payload, admission)`` only
    when the admitted and final canonical bytes are identical.  Repeating the
    call after a crash is safe because the gate writer accepts only identical
    existing bytes.
    """

    payloads = tuple(worker_payloads)
    roles = {_required_string(payload, "role") for payload in payloads}
    wave_indices = {_required_int(payload, "wave_index") for payload in payloads}
    if len(roles) != 1 or len(wave_indices) != 1:
        raise ValueError("one P90 preparation requires exactly one phase")
    if wave_indices == {0}:
        raise ValueError("wave zero does not use a live P90 admission")
    if (compact_artifact_resolver is None) != (compact_artifact_publisher is None):
        raise TypeError("P90 compact publication requires publisher and resolver")
    candidate = _build_governed_full_score_live_p90_submit_candidate(
        config,
        payloads,
        inventory=inventory,
        shard_plan=shard_plan,
        execution_plan=execution_plan,
        qualification_launch_authorization=qualification_launch_authorization,
        attempt_id=attempt_id,
        prior_wave_completion=prior_wave_completion,
        ledger_path=ledger_path,
        predecessor_authorization=predecessor_authorization,
        producer_phase_completion=producer_phase_completion,
        producer_phase_completion_uri=producer_phase_completion_uri,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_artifact_resolver,
        latency_execution_plan_record=latency_execution_plan_record,
    )
    admission = write_governed_full_score_live_p90_budget_admission(
        budget_admission_path,
        execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        completed_block_paths=completed_block_paths,
        next_wave_index=next(iter(wave_indices)),
        next_phase=next(iter(roles)),
        attempt_id=attempt_id,
        next_submit_payload=candidate,
        ledger_path=ledger_path,
        predecessor_authorization=predecessor_authorization,
        latency_execution_plan_record=latency_execution_plan_record,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_artifact_resolver,
        compact_artifact_publisher=compact_artifact_publisher,
    )
    final_payload = build_databricks_full_score_run_submit_payload(
        config,
        payloads,
        inventory=inventory,
        shard_plan=shard_plan,
        execution_plan=execution_plan,
        qualification_launch_authorization=qualification_launch_authorization,
        attempt_id=attempt_id,
        prior_wave_completion=prior_wave_completion,
        producer_phase_completion=producer_phase_completion,
        producer_phase_completion_uri=producer_phase_completion_uri,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_artifact_resolver,
        budget_admission_path=budget_admission_path,
        ledger_path=ledger_path,
        predecessor_authorization=predecessor_authorization,
        latency_execution_plan_record=latency_execution_plan_record,
    )
    if final_payload != candidate:
        raise ValueError("P90 preparation changed the exact submit payload")
    return final_payload, admission


def build_databricks_full_score_run_submit_payload(
    config: DatabricksFullScoreJobConfig,
    worker_payloads: Sequence[Mapping[str, Any]],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    attempt_id: str,
    prior_wave_completion: Mapping[str, Any] | None = None,
    producer_phase_completion: Mapping[str, Any] | None = None,
    producer_phase_completion_uri: str | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
    budget_admission_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    predecessor_authorization: object | None = None,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render from a governed P90 file; caller mappings cannot authorize it."""

    payloads = tuple(worker_payloads)
    wave_indices = {_required_int(payload, "wave_index") for payload in payloads}
    if len(wave_indices) != 1:
        raise ValueError("one runs/submit payload may contain exactly one wave")
    wave_index = next(iter(wave_indices))
    remote_by_wave = _remote_consumer_authorizations_by_wave(
        ()
        if remote_consumer_authorizations is None
        else remote_consumer_authorizations,
        execution_plan=execution_plan,
    )
    immediate_remote_consumer_authorization = remote_by_wave.get(wave_index - 1)
    admission: Mapping[str, Any] | None = None
    if wave_index == 0:
        if budget_admission_path is not None:
            raise ValueError("wave zero must not consume a live P90 admission")
        if remote_by_wave:
            raise ValueError("wave zero cannot consume remote consumer authority")
    else:
        if remote_by_wave and set(remote_by_wave) != set(range(wave_index)):
            raise ValueError(
                "nonzero rendering requires every completed-wave remote authority"
            )
        if ledger_path is None:
            raise ValueError(
                "nonzero publication rendering requires the canonical live ledger"
            )
        _require_full_score_phase_predecessor_authorization(
            predecessor_authorization,
            execution_plan=execution_plan,
            ledger_path=ledger_path,
            wave_index=wave_index,
            phase=_required_string(payloads[0], "role"),
            latency_execution_plan_record=latency_execution_plan_record,
        )
        if budget_admission_path is None:
            raise ValueError(
                "nonzero publication rendering requires a governed live P90 "
                "admission path"
            )
        admission_file = _governed_compact_file(
            budget_admission_path,
            "live P90 admission",
            compact_artifact_resolver,
        )
        admission = _json_object(
            admission_file.read_bytes(),
            "live P90 admission",
        )
    result = _build_databricks_full_score_run_submit_payload(
        config,
        payloads,
        inventory=inventory,
        shard_plan=shard_plan,
        execution_plan=execution_plan,
        qualification_launch_authorization=qualification_launch_authorization,
        prior_wave_completion=prior_wave_completion,
        producer_phase_completion=producer_phase_completion,
        producer_phase_completion_uri=producer_phase_completion_uri,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorization=immediate_remote_consumer_authorization,
        compact_artifact_resolver=compact_artifact_resolver,
        budget_admission=admission,
        idempotency_attempt_id=attempt_id,
        publication_authorizing=True,
    )
    if admission is not None:
        replayed = load_governed_full_score_live_p90_budget_admission(
            cast(str | Path, budget_admission_path),
            execution_plan=execution_plan,
            inventory=inventory,
            shard_plan=shard_plan,
            next_submit_payload=result,
            ledger_path=cast(str | Path, ledger_path),
            predecessor_authorization=predecessor_authorization,
            latency_execution_plan_record=latency_execution_plan_record,
            remote_ready_authorization=remote_ready_authorization,
            remote_consumer_authorizations=remote_consumer_authorizations,
            compact_artifact_resolver=compact_artifact_resolver,
        )
        if replayed != admission:
            raise ValueError("live P90 admission file changed during rendering")
    return result


def preview_local_fixture_databricks_full_score_run_submit_payload(
    config: DatabricksFullScoreJobConfig,
    worker_payloads: Sequence[Mapping[str, Any]],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    prior_wave_completion: Mapping[str, Any] | None = None,
    producer_phase_completion: Mapping[str, Any] | None = None,
    producer_phase_completion_uri: str | None = None,
    budget_admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a deterministic candidate only; this API performs no admission."""

    return _build_databricks_full_score_run_submit_payload(
        config,
        worker_payloads,
        inventory=inventory,
        shard_plan=shard_plan,
        execution_plan=execution_plan,
        prior_wave_completion=prior_wave_completion,
        producer_phase_completion=producer_phase_completion,
        producer_phase_completion_uri=producer_phase_completion_uri,
        budget_admission=budget_admission,
        qualification_launch_authorization=None,
        idempotency_attempt_id=None,
        publication_authorizing=False,
    )


def full_score_wave_worst_case_gpu_hours(
    worker_payloads: Sequence[Mapping[str, Any]],
    *,
    task_timeout_seconds: int = FULL_SCORE_DATABRICKS_TASK_TIMEOUT_SECONDS,
) -> float:
    """Return one phase's reservation if every task consumes its full cap."""

    _validated_databricks_run_timeout_seconds(task_timeout_seconds)
    if task_timeout_seconds != FULL_SCORE_DATABRICKS_TASK_TIMEOUT_SECONDS:
        raise ValueError("full-score task timeout is frozen to six hours")
    payloads = tuple(worker_payloads)
    if not payloads:
        raise ValueError("worker_payloads must not be empty")
    if len({_required_int(payload, "wave_index") for payload in payloads}) != 1:
        raise ValueError("GPU-hour reservation is computed one wave at a time")
    if len({_required_string(payload, "role") for payload in payloads}) != 1:
        raise ValueError("GPU-hour reservation is computed one phase at a time")
    if len(payloads) > FULL_SCORE_MAX_WORKERS:
        raise ValueError("GPU-hour reservation exceeds sixteen live tasks")
    return len(payloads) * task_timeout_seconds / 3600.0


def _validate_full_score_phase_position(wave_index: int, phase: str) -> None:
    if type(wave_index) is not int or wave_index < 0:
        raise ValueError("full-score wave_index must be non-negative")
    if phase not in {"producer", "consumer"}:
        raise ValueError("full-score phase must be producer or consumer")


def _full_score_phase_workload_id(
    execution_plan: Mapping[str, Any],
    *,
    wave_index: int,
    phase: str,
) -> str:
    """Return the ledger identity shared by every retry of one exact phase."""

    execution_sha256 = _require_sha256(
        execution_plan.get("closed_record_sha256"),
        field_name="full-score execution-plan SHA-256",
    )
    _validate_full_score_phase_position(wave_index, phase)
    return f"full-score:{execution_sha256}:wave-{wave_index:03d}:{phase}"


def _build_full_score_role_cluster(
    config: DatabricksFullScoreJobConfig,
    *,
    role: str,
) -> dict[str, Any]:
    if role not in {"producer", "consumer"}:
        raise ValueError("full-score Databricks role must be producer or consumer")
    purpose = f"cachet-vllm-0271-full-score-{role}"
    # The shared serving target registry deliberately excludes qualification-only
    # L40S. Build the reviewed single-node shape through the L4 validator, then
    # substitute only the two node identity fields for a sealed producer task.
    cluster: dict[str, Any] = build_single_node_gpu_cluster(
        DatabricksSingleNodeGPUClusterConfig(
            purpose=purpose,
            node_type_id=config.consumer_node_type_id,
            spark_version=config.spark_version,
            data_security_mode=config.data_security_mode,
            single_user_name=config.single_user_name,
            availability=config.availability,
            zone_id=(config.producer_zone_id if role == "producer" else config.zone_id),
            custom_tags=config.custom_tags,
        )
    )
    if role == "producer":
        selection = config.gpu_qualification.selection
        if selection.generation_databricks_node_type_id != (
            config.producer_node_type_id
        ):
            raise ValueError("producer cluster differs from sealed GPU qualification")
        cluster["node_type_id"] = config.producer_node_type_id
        cluster["driver_node_type_id"] = config.producer_node_type_id
        tags = cast(dict[str, str], cluster["custom_tags"])
        tags["hardware_target"] = FULL_SCORE_PRODUCER_HARDWARE_TARGET
        tags["generation_artifacts"] = selection.generation_artifacts_sha256[:32]
    return cluster


def build_full_score_producer_phase_completion_record(
    execution_plan: Mapping[str, Any],
    *,
    wave_index: int,
    ready_shard_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate every ready shard and authorize only that wave's consumer phase."""

    _validate_budget_execution_plan(execution_plan)
    waves = cast(list[Mapping[str, Any]], execution_plan["waves"])
    if type(wave_index) is not int or not 0 <= wave_index < len(waves):
        raise ValueError("producer-phase wave_index is invalid")
    wave = waves[wave_index]
    expected_ids = set(cast(list[str], wave["shard_ids"]))
    planned_by_id = {
        cast(str, shard["shard_id"]): shard
        for shard in cast(list[Mapping[str, Any]], wave["shards"])
    }
    observed: dict[str, dict[str, Any]] = {}
    for raw_ready in ready_shard_records:
        ready = _json_mapping(raw_ready, "ready-shard record")
        if ready.get("record_type") != FULL_SCORE_READY_SHARD_RECORD_TYPE:
            raise ValueError("producer completion requires ready-shard records")
        if ready.get("schema_version") != FULL_SCORE_READY_SHARD_SCHEMA_VERSION:
            raise ValueError("producer completion ready-shard schema drift")
        if ready.get("closed_record_sha256") != _closed_record_sha256(ready):
            raise ValueError("producer completion ready-shard closure drift")
        shard_id = _required_string(ready, "shard_id")
        if shard_id in observed or shard_id not in planned_by_id:
            raise ValueError("producer completion has duplicate or unknown shard")
        planned = planned_by_id[shard_id]
        expected = {
            "execution_plan_sha256": execution_plan.get("closed_record_sha256"),
            "shard_items_sha256": planned.get("items_sha256"),
            "wave_index": wave_index,
        }
        if any(ready.get(key) != value for key, value in expected.items()):
            raise ValueError("producer completion ready-shard identity drift")
        lifecycle = ready.get("lifecycle")
        if not isinstance(lifecycle, list) or lifecycle[-1:] != ["commit_ready_shard"]:
            raise ValueError("producer completion shard was not durably committed")
        ready_bytes = ready.get("ready_bytes")
        upper_bound = ready.get("ready_bytes_upper_bound")
        if (
            type(ready_bytes) is not int
            or type(upper_bound) is not int
            or ready_bytes <= 0
            or ready_bytes > upper_bound
            or upper_bound != planned.get("ready_bytes_upper_bound")
        ):
            raise ValueError("producer completion ready-shard byte bound drift")
        hardware = _required_mapping(ready, "producer_hardware")
        if (
            hardware.get("hardware_target") != FULL_SCORE_PRODUCER_HARDWARE_TARGET
            or hardware.get("gpu_name") != FULL_SCORE_PRODUCER_GPU_NAME
            or hardware.get("node_type_id") != FULL_SCORE_PRODUCER_NODE_TYPE_ID
        ):
            raise ValueError("producer completion hardware evidence drift")
        generator_contract = _required_mapping(
            ready,
            "generator_artifact_contract",
        )
        if generator_contract.get("vllm_bitsandbytes_loader_sha256") != (
            FULL_SCORE_VLLM_BNB_LOADER_SHA256
        ):
            raise ValueError("producer completion generator contract drift")
        observed[shard_id] = {
            "generator_artifact_contract_sha256": _canonical_sha256(generator_contract),
            "ready_bytes": ready_bytes,
            "ready_record_sha256": ready.get("closed_record_sha256"),
            "shard_id": shard_id,
            "shard_items_sha256": ready.get("shard_items_sha256"),
        }
    if set(observed) != expected_ids:
        raise ValueError("producer phase must close every wave shard exactly once")
    total_ready_bytes = sum(
        cast(int, item["ready_bytes"]) for item in observed.values()
    )
    if total_ready_bytes > cast(int, wave["max_backlog_bytes"]):
        raise ValueError("producer completion exceeds the hard durable-backlog cap")
    record: dict[str, Any] = {
        "authorization_scope": FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
        "closed_record_sha256": "",
        "consumer_phase_authorized": True,
        "execution_plan_sha256": execution_plan.get("closed_record_sha256"),
        "ready_shards": [observed[shard_id] for shard_id in sorted(observed)],
        "record_type": FULL_SCORE_PRODUCER_PHASE_COMPLETION_RECORD_TYPE,
        "schema_version": FULL_SCORE_PRODUCER_PHASE_COMPLETION_SCHEMA_VERSION,
        "shard_ids": sorted(observed),
        "total_ready_bytes": total_ready_bytes,
        "wave_index": wave_index,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def build_governed_full_score_producer_phase_completion_record(
    execution_plan: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    wave_index: int,
    producer_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authorize consumers from verified durable ready-shard file trees."""

    _validate_publication_full_score_inputs(inventory, shard_plan, execution_plan)
    payloads = tuple(producer_payloads)
    _validate_complete_phase_payload_set(payloads, phase="producer")
    records: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for payload in payloads:
        if (
            payload.get("authorization_scope")
            != FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
            or payload.get("wave_index") != wave_index
        ):
            raise ValueError("governed producer payload scope/wave drift")
        validate_full_score_worker_payload(
            payload,
            inventory=inventory,
            shard_plan=shard_plan,
            execution_plan=execution_plan,
        )
        for shard in cast(list[Mapping[str, Any]], payload["shards"]):
            shard_id = _required_string(shard, "shard_id")
            ready_dir = _ready_shard_dir(payload, shard_id)
            ready = _validate_ready_shard(
                ready_dir,
                shard=shard,
                payload=payload,
                inventory=inventory,
                shard_plan=shard_plan,
                execution_plan=execution_plan,
            )
            records.append(ready)
            ready_path = ready_dir / "ready-record.json"
            files.append(
                {
                    "file_sha256": sha256(ready_path.read_bytes()).hexdigest(),
                    "path": str(ready_path),
                    "ready_record_sha256": ready["closed_record_sha256"],
                    "shard_id": shard_id,
                }
            )
    record = build_full_score_producer_phase_completion_record(
        execution_plan,
        wave_index=wave_index,
        ready_shard_records=records,
    )
    record["authorization_scope"] = FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
    record["ready_record_files"] = sorted(files, key=lambda item: item["shard_id"])
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def build_full_score_wave_completion_record(
    execution_plan: Mapping[str, Any],
    *,
    wave_index: int,
    deletion_attestations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Close the gate that alone authorizes rendering the following wave."""

    if type(wave_index) is not int or wave_index < 0:
        raise ValueError("wave_index must be a non-negative integer")
    if execution_plan.get("closed_record_sha256") != _closed_record_sha256(
        execution_plan
    ):
        raise ValueError("execution-plan closure drift")
    waves = execution_plan.get("waves")
    if not isinstance(waves, list) or wave_index >= len(waves):
        raise ValueError("wave_index is outside the execution plan")
    wave = _json_mapping(waves[wave_index], "execution wave")
    expected_ids = set(cast(list[str], wave["shard_ids"]))
    observed: dict[str, Mapping[str, Any]] = {}
    for raw in deletion_attestations:
        attestation = _json_mapping(raw, "deletion attestation")
        if (
            attestation.get("record_type")
            != FULL_SCORE_DELETION_ATTESTATION_RECORD_TYPE
        ):
            raise ValueError("wave completion requires deletion attestations")
        if (
            attestation.get("schema_version")
            != FULL_SCORE_DELETION_ATTESTATION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported deletion-attestation schema")
        if attestation.get("closed_record_sha256") != _closed_record_sha256(
            attestation
        ):
            raise ValueError("deletion-attestation closure drift")
        if attestation.get("wave_index") != wave_index:
            raise ValueError("deletion attestation belongs to a different wave")
        if attestation.get("execution_plan_sha256") != execution_plan.get(
            "closed_record_sha256"
        ):
            raise ValueError("deletion attestation execution-plan drift")
        shard_id = _required_string(attestation, "shard_id")
        if shard_id in observed:
            raise ValueError("duplicate deletion attestation")
        lifecycle = attestation.get("lifecycle")
        if not isinstance(lifecycle, list) or lifecycle[-1:] != [
            "delete_ephemeral_q8_kv"
        ]:
            raise ValueError("deletion attestation lacks the terminal delete state")
        observed[shard_id] = attestation
    if set(observed) != expected_ids:
        raise ValueError("wave completion requires every shard deletion exactly once")
    record: dict[str, Any] = {
        "authorization_scope": FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
        "closed_record_sha256": "",
        "deletion_attestation_sha256": sorted(
            cast(str, item["closed_record_sha256"]) for item in observed.values()
        ),
        "execution_plan_sha256": execution_plan.get("closed_record_sha256"),
        "next_wave_authorized": True,
        "record_type": FULL_SCORE_WAVE_COMPLETION_RECORD_TYPE,
        "schema_version": FULL_SCORE_WAVE_COMPLETION_SCHEMA_VERSION,
        "shard_ids": sorted(observed),
        "wave_index": wave_index,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def build_governed_full_score_wave_completion_record(
    execution_plan: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    wave_index: int,
    evidence_dirs: Sequence[str | Path],
) -> dict[str, Any]:
    """Authorize the next wave only from replayed, deleted shard directories."""

    _validate_publication_full_score_inputs(inventory, shard_plan, execution_plan)
    deletions: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for raw_directory in evidence_dirs:
        directory = _cluster_path(
            _require_shared_dbfs_path(
                raw_directory,
                "governed wave evidence directory",
            )
        )
        evidence, deletion = load_governed_full_score_shard_evidence(
            directory,
            inventory=inventory,
            shard_plan=shard_plan,
            execution_plan=execution_plan,
            require_deletion=True,
        )
        if deletion is None or evidence.get("wave_index") != wave_index:
            raise ValueError("governed wave evidence belongs to another wave")
        deletions.append(deletion)
        evidence_path = directory / "evidence.json"
        deletion_path = directory / "deletion-attestation.json"
        bindings.append(
            {
                "deletion_file_sha256": sha256(deletion_path.read_bytes()).hexdigest(),
                "deletion_path": str(deletion_path),
                "evidence_file_sha256": sha256(evidence_path.read_bytes()).hexdigest(),
                "evidence_path": str(evidence_path),
                "shard_id": evidence["shard_id"],
            }
        )
    record = build_full_score_wave_completion_record(
        execution_plan,
        wave_index=wave_index,
        deletion_attestations=deletions,
    )
    record["authorization_scope"] = FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
    record["governed_evidence_files"] = sorted(
        bindings,
        key=lambda item: item["shard_id"],
    )
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _full_score_ledger_prefix_state(
    ledger: DatabricksClusterHourLedger,
    prefix: DatabricksLedgerPrefix,
) -> DatabricksClusterHourLedger:
    """Reconstruct and verify the exact immutable state named by *prefix*."""

    require_databricks_ledger_prefix(ledger, prefix)
    state = DatabricksClusterHourLedger(
        ledger_id=ledger.ledger_id,
        cap_cluster_hours=ledger.cap_cluster_hours,
        reservations=ledger.reservations[: prefix.reservation_count],
        submission_receipts=ledger.submission_receipts[
            : prefix.submission_receipt_count
        ],
        terminal_actuals=ledger.terminal_actuals[: prefix.terminal_actual_count],
    )
    if databricks_ledger_prefix(state) != prefix:
        raise ValueError("ledger prefix does not close its exact ordered slices")
    return state


def _validate_full_score_raw_task_configuration(
    run_record: Mapping[str, Any],
    submit_payload: Mapping[str, Any],
) -> None:
    """Bind each raw runs/get Python task to its submitted configuration."""

    raw_submitted_tasks = submit_payload.get("tasks")
    raw_observed_tasks = run_record.get("tasks")
    if not isinstance(raw_submitted_tasks, list) or not isinstance(
        raw_observed_tasks, list
    ):
        raise ValueError("Databricks raw terminal tasks must be arrays")
    if len(raw_observed_tasks) != len(raw_submitted_tasks):
        raise ValueError("Databricks raw terminal task coverage drift")

    submitted_by_key: dict[str, dict[str, Any]] = {}
    for raw_task in raw_submitted_tasks:
        task = _json_mapping(raw_task, "submitted Databricks task")
        task_key = _required_string(task, "task_key")
        if task_key in submitted_by_key:
            raise ValueError("submitted Databricks task coverage drift")
        submitted_by_key[task_key] = _json_mapping(
            _required_mapping(task, "spark_python_task"),
            "submitted Databricks spark_python_task",
        )

    observed_keys: set[str] = set()
    for raw_task in raw_observed_tasks:
        task = _json_mapping(raw_task, "raw Databricks terminal task")
        task_key = _required_string(task, "task_key")
        expected = submitted_by_key.get(task_key)
        if expected is None or task_key in observed_keys:
            raise ValueError("Databricks raw terminal task coverage drift")
        observed_keys.add(task_key)
        observed = _json_mapping(
            _required_mapping(task, "spark_python_task"),
            "raw Databricks terminal spark_python_task",
        )
        if observed != expected:
            raise ValueError("Databricks raw terminal spark_python_task binding drift")
    if observed_keys != set(submitted_by_key):
        raise ValueError("Databricks raw terminal task coverage drift")


def build_governed_full_score_phase_terminal_record(
    execution_plan: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    wave_index: int,
    phase: str,
    attempt_id: str,
    submit_payload_path: str | Path,
    control_plane_run_path: str | Path,
    ledger_path: str | Path,
    submission_authorization: FullScorePhaseSubmissionAuthorization | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
    _replay_ledger_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive phase billing from immutable submit, runs/get, and ledger snapshots.

    ``ledger_path`` is the canonical mutable local campaign ledger.  The record
    stores only its privacy-safe path digest and append-only prefixes; immutable
    submit/control-plane evidence may live on DBFS/UC.
    """

    _validate_publication_full_score_inputs(inventory, shard_plan, execution_plan)
    submit_file = _governed_compact_file(
        submit_payload_path,
        "full-score submit payload",
        compact_artifact_resolver,
    )
    run_file = _governed_compact_file(
        control_plane_run_path,
        "Databricks control-plane run",
        compact_artifact_resolver,
    )
    ledger_file = Path(ledger_path).expanduser().absolute()
    _require_regular_file_no_follow(ledger_file, "cluster-hour ledger")
    submit_payload = _json_object(
        submit_file.read_bytes(),
        "full-score submit payload",
    )
    run_record = _json_object(
        run_file.read_bytes(),
        "Databricks control-plane run",
    )
    task_bindings = _validated_full_score_phase_submit_payload(
        execution_plan,
        submit_payload,
        inventory=inventory,
        shard_plan=shard_plan,
        wave_index=wave_index,
        phase=phase,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    _validate_full_score_raw_task_configuration(run_record, submit_payload)
    expected_node_type_id = (
        FULL_SCORE_PRODUCER_NODE_TYPE_ID
        if phase == "producer"
        else FULL_SCORE_CONSUMER_NODE_TYPE_ID
    )
    status = summarize_databricks_run(
        dict(run_record),
        submit_payload=submit_payload,
        submit_payload_path=str(submit_payload_path),
    )
    status_record = databricks_run_status_record(status)
    if status_record is None:
        raise ValueError("Databricks run did not produce a status record")
    if phase == "consumer":
        validate_databricks_run_status_sidecar(
            status,
            expected_node_type_id=expected_node_type_id,
        )
    else:
        _validate_full_score_l40s_terminal_status(status_record)
    submit_snapshot, submit_bytes = canonical_databricks_submit_payload_snapshot(
        submit_payload
    )
    submit_sha256 = sha256(submit_bytes).hexdigest()
    ledger_path_sha256 = databricks_ledger_path_sha256(ledger_path)
    if submission_authorization is not None:
        if type(submission_authorization) is not FullScorePhaseSubmissionAuthorization:
            raise TypeError("phase terminal submission authority has the wrong type")
        if (
            submission_authorization.execution_plan_sha256
            != execution_plan.get("closed_record_sha256")
            or submission_authorization.wave_index != wave_index
            or submission_authorization.phase != phase
            or submission_authorization.attempt_id != attempt_id
            or submission_authorization.submit_payload_sha256 != submit_sha256
            or submission_authorization.ledger_path_sha256 != ledger_path_sha256
        ):
            raise ValueError("phase terminal submission authority binding drift")
        predecessor_prefix = submission_authorization.predecessor_prefix
        batch_prefix = require_databricks_batch_reservation_authorization(
            submission_authorization.batch_authorization,
            expected_predecessor_prefix=predecessor_prefix,
            expected_attempt_ids=(attempt_id,),
            expected_submit_payload_sha256s=(submit_sha256,),
        )
        _require_full_score_phase_lease(
            ledger_path,
            submission_authorization,
        )
        intent_record_sha256 = submission_authorization.intent_record_sha256
        phase_lease_record_sha256 = _required_string(
            _full_score_phase_lease_record(submission_authorization),
            "closed_record_sha256",
        )
        workspace_host_sha256 = submission_authorization.workspace_host_sha256
        user_name_sha256 = submission_authorization.user_name_sha256
        expected_terminal_prefix: DatabricksLedgerPrefix | None = None
    else:
        if not isinstance(_replay_ledger_lineage, Mapping):
            raise TypeError("phase terminal build requires atomic submission authority")
        if _replay_ledger_lineage.get("ledger_path_sha256") != ledger_path_sha256:
            raise ValueError("phase terminal replay ledger path binding drift")
        predecessor_prefix = databricks_ledger_prefix_from_record(
            _required_mapping(_replay_ledger_lineage, "predecessor_prefix")
        )
        batch_prefix = databricks_ledger_prefix_from_record(
            _required_mapping(_replay_ledger_lineage, "batch_prefix")
        )
        intent_record_sha256 = _require_sha256(
            _replay_ledger_lineage.get("intent_record_sha256"),
            field_name="phase terminal intent_record_sha256",
        )
        phase_lease_record_sha256 = _require_sha256(
            _replay_ledger_lineage.get("phase_lease_record_sha256"),
            field_name="phase terminal phase_lease_record_sha256",
        )
        replay_lease_path = _full_score_phase_lease_candidate_path(
            ledger_path,
            execution_plan_sha256=_required_string(
                execution_plan, "closed_record_sha256"
            ),
            wave_index=wave_index,
            phase=phase,
        )
        replay_sidecar = _read_full_score_workspace_authority(
            _full_score_workspace_authority_path(replay_lease_path),
            label="full-score replay workspace authority",
        )
        if (
            replay_sidecar.get("record_type")
            != "cachet.full_score_phase_workspace_authority.v1"
            or replay_sidecar.get("schema_version") != 1
            or replay_sidecar.get("closed_record_sha256")
            != _closed_record_sha256(replay_sidecar)
            or replay_sidecar.get("intent_record_sha256") != intent_record_sha256
            or replay_sidecar.get("phase_lease_record_sha256")
            != phase_lease_record_sha256
        ):
            raise ValueError("phase terminal workspace authority replay drift")
        workspace_host_sha256 = _require_sha256(
            replay_sidecar.get("workspace_host_sha256"),
            field_name="workspace_host_sha256",
        )
        user_name_sha256 = _require_sha256(
            replay_sidecar.get("user_name_sha256"), field_name="user_name_sha256"
        )
        if (
            batch_prefix.reservation_count != predecessor_prefix.reservation_count + 1
            or batch_prefix.submission_receipt_count
            != predecessor_prefix.submission_receipt_count
            or batch_prefix.terminal_actual_count
            != predecessor_prefix.terminal_actual_count
        ):
            raise ValueError("phase terminal replay batch-prefix transition drift")
        expected_terminal_prefix = databricks_ledger_prefix_from_record(
            _required_mapping(_replay_ledger_lineage, "terminal_prefix")
        )
        if (
            expected_terminal_prefix.reservation_count != batch_prefix.reservation_count
            or expected_terminal_prefix.submission_receipt_count
            != batch_prefix.submission_receipt_count + 1
            or expected_terminal_prefix.terminal_actual_count
            != batch_prefix.terminal_actual_count + 1
        ):
            raise ValueError("phase terminal replay terminal-prefix transition drift")
        request = DatabricksRunAttemptReservationRequest(
            attempt_id=attempt_id,
            workload_id=_full_score_phase_workload_id(
                execution_plan,
                wave_index=wave_index,
                phase=phase,
            ),
            submit_payload=submit_payload,
        )
        replayed_batch_authorization = (
            replay_databricks_run_attempt_batch_authorization_json(
                ledger_path,
                (request,),
                expected_predecessor_prefix=predecessor_prefix,
            )
        )
        if replayed_batch_authorization.batch_prefix != batch_prefix:
            raise ValueError("phase terminal replay batch authority drift")
        replayed_submission_authorization = FullScorePhaseSubmissionAuthorization(
            execution_plan_sha256=_required_string(
                execution_plan,
                "closed_record_sha256",
            ),
            wave_index=wave_index,
            phase=phase,
            ledger_path_sha256=ledger_path_sha256,
            predecessor_prefix=predecessor_prefix,
            batch_authorization=replayed_batch_authorization,
            attempt_id=attempt_id,
            submit_payload_sha256=submit_sha256,
            intent_record_sha256=intent_record_sha256,
            workspace_host_sha256=workspace_host_sha256,
            user_name_sha256=user_name_sha256,
            _issuer=_FULL_SCORE_PHASE_SUBMISSION_AUTHORIZATION_ISSUER,
        )
        _require_full_score_phase_lease(
            ledger_path,
            replayed_submission_authorization,
        )
        if (
            _required_string(
                _full_score_phase_lease_record(replayed_submission_authorization),
                "closed_record_sha256",
            )
            != phase_lease_record_sha256
        ):
            raise ValueError("phase terminal replay lease record drift")
    submit_summary = _required_mapping(status_record, "submit_payload")
    if submit_summary.get("sha256") != submit_sha256:
        raise ValueError("Databricks status submit-payload binding drift")
    status_tasks = status_record.get("tasks")
    if not isinstance(status_tasks, list):
        raise ValueError("Databricks status tasks must be an array")
    run_id = _required_run_id(status_record.get("run_id"), "run status run_id")
    task_billing: list[dict[str, Any]] = []
    observed_task_keys: set[str] = set()
    observed_task_run_ids: set[str] = set()
    observed_cluster_ids: set[str] = set()
    expected_by_key = {binding["task_key"]: binding for binding in task_bindings}
    for raw_task in status_tasks:
        task = _json_mapping(raw_task, "Databricks terminal task")
        task_key = _required_string(task, "task_key")
        binding = expected_by_key.get(task_key)
        if binding is None or task_key in observed_task_keys:
            raise ValueError("Databricks terminal task coverage drift")
        observed_task_keys.add(task_key)
        task_run_id = _required_run_id(
            task.get("run_id"),
            "Databricks terminal task run_id",
        )
        cluster_id = _require_nonempty(
            task.get("cluster_id"),
            "Databricks terminal task cluster_id",
        )
        if task_run_id == run_id or task_run_id in observed_task_run_ids:
            raise ValueError("Databricks terminal task run_id coverage drift")
        if cluster_id in observed_cluster_ids:
            raise ValueError(
                "Databricks terminal tasks must have distinct billed clusters"
            )
        observed_task_run_ids.add(task_run_id)
        observed_cluster_ids.add(cluster_id)
        start_time = task.get("start_time")
        end_time = task.get("end_time")
        if (
            type(start_time) is not int
            or type(end_time) is not int
            or start_time < 0
            or end_time <= start_time
        ):
            raise ValueError("Databricks terminal task times are invalid")
        task_billing.append(
            {
                "billed_gpu_seconds": (end_time - start_time) / 1000.0,
                "cluster_id": cluster_id,
                "durable_output_root": binding["durable_output_root"],
                "end_time_ms": end_time,
                "shard_id": binding["shard_id"],
                "start_time_ms": start_time,
                "task_key": task_key,
                "task_run_id": task_run_id,
                "worker_index": binding["worker_index"],
                "worker_payload_file_sha256": binding["worker_payload_file_sha256"],
                "worker_payload_record_sha256": binding["worker_payload_record_sha256"],
                "worker_payload_uri": binding["worker_payload_uri"],
            }
        )
    if observed_task_keys != set(expected_by_key):
        raise ValueError("Databricks terminal status omits a full-score task")
    task_billing.sort(key=lambda item: cast(int, item["worker_index"]))
    billed_gpu_seconds = sum(
        cast(float, item["billed_gpu_seconds"]) for item in task_billing
    )

    ledger_record = _json_object(
        ledger_file.read_bytes(),
        "cluster-hour ledger",
    )
    ledger = databricks_cluster_hour_ledger_from_record(ledger_record)
    _require_full_score_ledger_caps(ledger)
    require_databricks_ledger_prefix(ledger, predecessor_prefix)
    require_databricks_ledger_prefix(ledger, batch_prefix)
    predecessor_state = _full_score_ledger_prefix_state(
        ledger,
        predecessor_prefix,
    )
    batch_state = _full_score_ledger_prefix_state(ledger, batch_prefix)
    if batch_state.reservations[:-1] != predecessor_state.reservations:
        raise ValueError("phase terminal batch inserted a non-tail reservation")
    reservation = next(
        (item for item in ledger.reservations if item.attempt_id == attempt_id),
        None,
    )
    receipt = next(
        (item for item in ledger.submission_receipts if item.attempt_id == attempt_id),
        None,
    )
    terminal = next(
        (item for item in ledger.terminal_actuals if item.attempt_id == attempt_id),
        None,
    )
    if reservation is None or receipt is None or terminal is None:
        raise ValueError(
            "phase terminal requires reservation, submission receipt, and verified actual"
        )
    expected_reservation = databricks_submit_payload_reservation(
        submit_snapshot,
        attempt_id=attempt_id,
        workload_id=_full_score_phase_workload_id(
            execution_plan,
            wave_index=wave_index,
            phase=phase,
        ),
    )
    if reservation != expected_reservation:
        raise ValueError("phase terminal reservation differs from the submit payload")
    if batch_state.reservations[-1] != expected_reservation:
        raise ValueError("phase terminal batch tail is not the expected reservation")
    control_plane_sha256 = _ledger_mapping_sha256(run_record)
    if (
        receipt.run_id != run_id
        or receipt.submit_payload_sha256 != submit_sha256
        or terminal.run_id != run_id
        or terminal.submit_payload_sha256 != submit_sha256
        or terminal.control_plane_status_sha256 != control_plane_sha256
        or terminal.verification_source != "direct_databricks_runs_get"
        or terminal.terminal_state != "succeeded"
        or abs(terminal.actual_cluster_duration_seconds - billed_gpu_seconds) > 1e-12
    ):
        raise ValueError("phase terminal ledger/control-plane reconciliation drift")
    terminal_prefix = databricks_ledger_prefix(ledger)
    if submission_authorization is not None:
        if (
            terminal_prefix.reservation_count != batch_prefix.reservation_count
            or terminal_prefix.submission_receipt_count
            != batch_prefix.submission_receipt_count + 1
            or terminal_prefix.terminal_actual_count
            != batch_prefix.terminal_actual_count + 1
            or ledger.active_reserved_cluster_hours != 0
            or ledger.active_reserved_task_count != 0
        ):
            raise ValueError(
                "phase terminal ledger is not one exact post-batch transition"
            )
        terminal_state = _full_score_ledger_prefix_state(
            ledger,
            terminal_prefix,
        )
        if (
            terminal_state.submission_receipts[-1] != receipt
            or terminal_state.terminal_actuals[-1] != terminal
        ):
            raise ValueError(
                "phase terminal receipt/terminal are not the exact new tail events"
            )
    else:
        assert expected_terminal_prefix is not None
        require_databricks_ledger_prefix(ledger, expected_terminal_prefix)
        terminal_state = _full_score_ledger_prefix_state(
            ledger,
            expected_terminal_prefix,
        )
        if (
            terminal_state.reservations != batch_state.reservations
            or terminal_state.submission_receipts[:-1]
            != batch_state.submission_receipts
            or terminal_state.terminal_actuals[:-1] != batch_state.terminal_actuals
            or terminal_state.submission_receipts[-1] != receipt
            or terminal_state.terminal_actuals[-1] != terminal
            or terminal_state.active_reserved_cluster_hours != 0
            or terminal_state.active_reserved_task_count != 0
        ):
            raise ValueError(
                "phase terminal replay contains intervening or active ledger events"
            )
        terminal_prefix = expected_terminal_prefix
    record: dict[str, Any] = {
        "attempt_id": attempt_id,
        "authorization_scope": FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE,
        "billed_gpu_seconds": billed_gpu_seconds,
        "closed_record_sha256": "",
        "control_plane": {
            "canonical_record_sha256": control_plane_sha256,
            "file_sha256": sha256(run_file.read_bytes()).hexdigest(),
            "path": str(control_plane_run_path),
            "run_id": run_id,
        },
        "execution_plan_sha256": execution_plan.get("closed_record_sha256"),
        "ledger": {
            "batch_prefix": batch_prefix.to_record(),
            "intent_record_sha256": intent_record_sha256,
            "ledger_id": ledger.ledger_id,
            "ledger_path_sha256": ledger_path_sha256,
            "phase_lease_record_sha256": phase_lease_record_sha256,
            "predecessor_prefix": predecessor_prefix.to_record(),
            "terminal_prefix": terminal_prefix.to_record(),
        },
        "phase": phase,
        "record_type": FULL_SCORE_PHASE_TERMINAL_RECORD_TYPE,
        "schema_version": FULL_SCORE_PHASE_TERMINAL_SCHEMA_VERSION,
        "submit_payload": {
            "file_sha256": sha256(submit_file.read_bytes()).hexdigest(),
            "path": str(submit_payload_path),
            "sha256": submit_sha256,
        },
        "task_billing": task_billing,
        "wave_index": wave_index,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def write_governed_full_score_phase_terminal_record(
    path: str | Path,
    execution_plan: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Write and reread a phase terminal record without replacement."""

    compact_artifact_publisher = kwargs.pop(
        "compact_artifact_publisher",
        None,
    )
    compact_artifact_resolver = kwargs.get("compact_artifact_resolver")
    if (compact_artifact_publisher is None) != (compact_artifact_resolver is None):
        raise TypeError("phase terminal compact publication requires publisher and CAS")
    record = build_governed_full_score_phase_terminal_record(
        execution_plan,
        **kwargs,
    )
    _publish_governed_compact_file(
        path,
        "phase terminal output path",
        _canonical_pretty_json_bytes(record),
        compact_artifact_publisher,
    )
    return load_governed_full_score_phase_terminal_record(
        path,
        execution_plan=execution_plan,
        inventory=kwargs["inventory"],
        shard_plan=kwargs["shard_plan"],
        ledger_path=kwargs["ledger_path"],
        compact_artifact_resolver=compact_artifact_resolver,
    )


def load_governed_full_score_phase_terminal_record(
    path: str | Path,
    *,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    ledger_path: str | Path,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> dict[str, Any]:
    """Reread and fully replay one publication phase-terminal record."""

    terminal_file = _governed_compact_file(
        path,
        "phase terminal record",
        compact_artifact_resolver,
    )
    record = _json_object(terminal_file.read_bytes(), "phase terminal record")
    if record.get("record_type") != FULL_SCORE_PHASE_TERMINAL_RECORD_TYPE:
        raise ValueError("phase terminal record_type drift")
    if record.get("schema_version") != FULL_SCORE_PHASE_TERMINAL_SCHEMA_VERSION:
        raise ValueError("phase terminal schema drift")
    if record.get("authorization_scope") != FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE:
        raise ValueError("phase terminal is not publication-authorizing")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("phase terminal closure drift")
    rebuilt = build_governed_full_score_phase_terminal_record(
        execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        wave_index=_required_int(record, "wave_index"),
        phase=_required_string(record, "phase"),
        attempt_id=_required_string(record, "attempt_id"),
        submit_payload_path=_required_string(
            _required_mapping(record, "submit_payload"),
            "path",
        ),
        control_plane_run_path=_required_string(
            _required_mapping(record, "control_plane"),
            "path",
        ),
        ledger_path=ledger_path,
        compact_artifact_resolver=compact_artifact_resolver,
        _replay_ledger_lineage=_required_mapping(record, "ledger"),
    )
    if rebuilt != record:
        raise ValueError("phase terminal does not replay from its bound files")
    return record


def replay_governed_full_score_phase_authorization(
    path: str | Path,
    *,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    ledger_path: str | Path,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> FullScorePhaseAuthorization:
    """Reissue a terminal phase authority from its exact historical slices.

    This is the durable controller-restart path.  It deliberately reconstructs
    the historical terminal prefix instead of absorbing any later campaign
    events that may already have been appended to the canonical ledger.
    """

    record = load_governed_full_score_phase_terminal_record(
        path,
        execution_plan=execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        ledger_path=ledger_path,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    live_path = Path(ledger_path).expanduser().absolute()
    ledger_binding = _required_mapping(record, "ledger")
    path_sha256 = databricks_ledger_path_sha256(live_path)
    if ledger_binding.get("ledger_path_sha256") != path_sha256:
        raise ValueError("phase authorization replay ledger path binding drift")
    predecessor_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(ledger_binding, "predecessor_prefix")
    )
    batch_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(ledger_binding, "batch_prefix")
    )
    terminal_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(ledger_binding, "terminal_prefix")
    )
    ledger = read_databricks_cluster_hour_ledger_json(live_path)
    _require_full_score_ledger_caps(ledger)
    for label, expected in (
        ("predecessor", predecessor_prefix),
        ("batch", batch_prefix),
        ("terminal", terminal_prefix),
    ):
        observed = databricks_ledger_prefix_at_counts(
            ledger,
            reservation_count=expected.reservation_count,
            submission_receipt_count=expected.submission_receipt_count,
            terminal_actual_count=expected.terminal_actual_count,
        )
        if observed != expected:
            raise ValueError(f"phase authorization replay {label} prefix binding drift")
    terminal_state = _full_score_ledger_prefix_state(ledger, terminal_prefix)
    if (
        terminal_state.active_reserved_cluster_hours != 0
        or terminal_state.active_reserved_task_count != 0
    ):
        raise ValueError("phase authorization replay terminal prefix is active")
    terminal_record_sha256 = _required_string(record, "closed_record_sha256")
    phase_lease_root = _full_score_phase_lease_path(
        live_path,
        execution_plan_sha256=_required_string(
            record,
            "execution_plan_sha256",
        ),
        wave_index=_required_int(record, "wave_index"),
        phase=_required_string(record, "phase"),
    ).parent
    replay_lease_path = _full_score_phase_lease_candidate_path(
        live_path,
        execution_plan_sha256=_required_string(record, "execution_plan_sha256"),
        wave_index=_required_int(record, "wave_index"),
        phase=_required_string(record, "phase"),
    )
    workspace_sidecar = _read_full_score_workspace_authority(
        _full_score_workspace_authority_path(replay_lease_path),
        label="full-score replay workspace authority",
    )
    if (
        workspace_sidecar.get("record_type")
        != "cachet.full_score_phase_workspace_authority.v1"
        or workspace_sidecar.get("schema_version") != 1
        or workspace_sidecar.get("closed_record_sha256")
        != _closed_record_sha256(workspace_sidecar)
        or workspace_sidecar.get("phase_lease_record_sha256")
        != ledger_binding.get("phase_lease_record_sha256")
        or workspace_sidecar.get("intent_record_sha256")
        != ledger_binding.get("intent_record_sha256")
    ):
        raise ValueError("phase authorization workspace sidecar drift")
    workspace_host_sha256 = _require_sha256(
        workspace_sidecar.get("workspace_host_sha256"),
        field_name="workspace_host_sha256",
    )
    user_name_sha256 = _require_sha256(
        workspace_sidecar.get("user_name_sha256"), field_name="user_name_sha256"
    )
    causal_closure_sha256 = _canonical_sha256(
        {
            "batch_prefix": batch_prefix.to_record(),
            "ledger_path_sha256": path_sha256,
            "terminal_prefix": terminal_prefix.to_record(),
            "terminal_record_sha256": terminal_record_sha256,
        }
    )
    return FullScorePhaseAuthorization(
        execution_plan_sha256=_require_sha256(
            record.get("execution_plan_sha256"),
            field_name="execution_plan_sha256",
        ),
        wave_index=_required_int(record, "wave_index"),
        phase=_required_string(record, "phase"),
        ledger_path_sha256=path_sha256,
        predecessor_prefix=predecessor_prefix,
        ledger_prefix=terminal_prefix,
        phase_lease_root=phase_lease_root,
        terminal_record_sha256=terminal_record_sha256,
        causal_closure_sha256=causal_closure_sha256,
        workspace_host_sha256=workspace_host_sha256,
        user_name_sha256=user_name_sha256,
        _issuer=_FULL_SCORE_PHASE_AUTHORIZATION_ISSUER,
    )


def collect_governed_full_score_phase_attempt(
    workspace: DatabricksWorkspaceConfig,
    *,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    ledger_path: str | Path,
    submission_authorization: FullScorePhaseSubmissionAuthorization,
    submit_payload_path: str | Path,
    control_plane_run_path: str | Path,
    terminal_record_path: str | Path | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
    compact_artifact_publisher: FullScoreCompactArtifactPublisher | None = None,
    opener: DatabricksURLOpener | None = None,
) -> tuple[dict[str, Any], FullScorePhaseAuthorization]:
    """Collect one phase directly from runs/get and issue its successor token."""

    if type(submission_authorization) is not FullScorePhaseSubmissionAuthorization:
        raise TypeError(
            "phase collection requires FullScorePhaseSubmissionAuthorization"
        )
    if terminal_record_path is None:
        raise ValueError("phase collection requires a durable terminal record path")
    if (compact_artifact_publisher is None) != (compact_artifact_resolver is None):
        raise TypeError("phase collection compact I/O requires publisher and CAS")
    submit_file = _governed_compact_file(
        submit_payload_path,
        "full-score submit payload",
        compact_artifact_resolver,
    )
    submit_payload = _json_object(
        submit_file.read_bytes(),
        "full-score submit payload",
    )
    _submit_snapshot, canonical_submit = canonical_databricks_submit_payload_snapshot(
        submit_payload
    )
    if sha256(canonical_submit).hexdigest() != (
        submission_authorization.submit_payload_sha256
    ):
        raise ValueError("phase collection submit-payload authority drift")
    _validate_publication_full_score_inputs(inventory, shard_plan, execution_plan)
    plan_sha256 = _require_sha256(
        execution_plan.get("closed_record_sha256"),
        field_name="full-score execution-plan SHA-256",
    )
    if submission_authorization.execution_plan_sha256 != plan_sha256:
        raise ValueError("phase collection execution-plan binding drift")
    live_path = Path(ledger_path).expanduser().absolute()
    path_sha256 = databricks_ledger_path_sha256(live_path)
    if submission_authorization.ledger_path_sha256 != path_sha256:
        raise ValueError("phase collection ledger path binding drift")
    _require_full_score_phase_lease(live_path, submission_authorization)
    batch_prefix = require_databricks_batch_reservation_authorization(
        submission_authorization.batch_authorization,
        expected_predecessor_prefix=(submission_authorization.predecessor_prefix),
        expected_attempt_ids=(submission_authorization.attempt_id,),
        expected_submit_payload_sha256s=(
            submission_authorization.submit_payload_sha256,
        ),
    )
    _require_full_score_bound_workspace_identity(
        workspace,
        authorization=submission_authorization,
        submit_payload=submit_payload,
        opener=opener,
    )
    before = read_databricks_cluster_hour_ledger_json(live_path)
    require_databricks_ledger_prefix(before, batch_prefix)
    before_prefix = databricks_ledger_prefix(before)
    if (
        before_prefix.reservation_count != batch_prefix.reservation_count
        or before_prefix.submission_receipt_count
        != batch_prefix.submission_receipt_count + 1
        or before_prefix.terminal_actual_count
        not in {
            batch_prefix.terminal_actual_count,
            batch_prefix.terminal_actual_count + 1,
        }
    ):
        raise ValueError("phase collection observed an unexpected ledger event")
    receipts = [
        item
        for item in before.submission_receipts
        if item.attempt_id == submission_authorization.attempt_id
    ]
    if len(receipts) != 1:
        raise ValueError("phase collection requires one durable submit receipt")
    receipt = receipts[0]
    if receipt.submit_payload_sha256 != (
        submission_authorization.submit_payload_sha256
    ):
        raise ValueError("phase collection submit receipt payload drift")
    resolved_opener = (
        cast(DatabricksURLOpener, urllib.request.urlopen) if opener is None else opener
    )
    run_record = get_databricks_run(
        workspace,
        receipt.run_id,
        opener=resolved_opener,
    )
    _publish_governed_compact_file(
        control_plane_run_path,
        "phase control-plane evidence path",
        _canonical_pretty_json_bytes(run_record),
        compact_artifact_publisher,
    )
    if before_prefix.terminal_actual_count == batch_prefix.terminal_actual_count:
        try:
            record_databricks_verified_run_terminal_actual_json(
                live_path,
                attempt_id=submission_authorization.attempt_id,
                run_record=run_record,
            )
        except ValueError:
            raced = read_databricks_cluster_hour_ledger_json(live_path)
            raced_prefix = databricks_ledger_prefix(raced)
            if (
                raced_prefix.reservation_count != batch_prefix.reservation_count
                or raced_prefix.submission_receipt_count
                != batch_prefix.submission_receipt_count + 1
                or raced_prefix.terminal_actual_count
                != batch_prefix.terminal_actual_count + 1
            ):
                raise
    record = build_governed_full_score_phase_terminal_record(
        execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        wave_index=submission_authorization.wave_index,
        phase=submission_authorization.phase,
        attempt_id=submission_authorization.attempt_id,
        submit_payload_path=submit_payload_path,
        control_plane_run_path=control_plane_run_path,
        ledger_path=live_path,
        submission_authorization=submission_authorization,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    final_ledger = read_databricks_cluster_hour_ledger_json(live_path)
    if databricks_ledger_path_sha256(live_path) != path_sha256:
        raise RuntimeError("phase ledger path changed during terminal collection")
    terminal_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(_required_mapping(record, "ledger"), "terminal_prefix")
    )
    require_databricks_ledger_prefix(final_ledger, terminal_prefix)
    if databricks_ledger_prefix(final_ledger) != terminal_prefix:
        raise RuntimeError("phase terminal collection is not the latest ledger prefix")
    reservation = next(
        item
        for item in final_ledger.reservations
        if item.attempt_id == submission_authorization.attempt_id
    )
    actual = next(
        item
        for item in final_ledger.terminal_actuals
        if item.attempt_id == submission_authorization.attempt_id
    )
    if (
        reservation.submit_payload_sha256
        != submission_authorization.submit_payload_sha256
        or actual.run_id != receipt.run_id
        or actual.submit_payload_sha256 != receipt.submit_payload_sha256
        or actual.verification_source != "direct_databricks_runs_get"
        or actual.terminal_state != "succeeded"
        or final_ledger.active_reserved_task_count != 0
    ):
        raise RuntimeError("phase final ledger reconciliation drift")
    _publish_governed_compact_file(
        terminal_record_path,
        "phase terminal evidence path",
        _canonical_pretty_json_bytes(record),
        compact_artifact_publisher,
    )
    reread = load_governed_full_score_phase_terminal_record(
        terminal_record_path,
        execution_plan=execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        ledger_path=live_path,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    if reread != record:
        raise RuntimeError("phase terminal evidence changed during reread")
    authorization = replay_governed_full_score_phase_authorization(
        terminal_record_path,
        execution_plan=execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        ledger_path=live_path,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    return record, authorization


def build_full_score_matched_billing_block(
    execution_plan: Mapping[str, Any],
    *,
    wave_index: int,
    shard_id: str,
    evidence_record: Mapping[str, Any],
    deletion_attestation: Mapping[str, Any],
    producer_billed_gpu_seconds: float,
    consumer_task_billed_gpu_seconds: float,
    billing_source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Seal producer plus indivisible matched-consumer task billing."""

    _validate_budget_execution_plan(execution_plan)
    waves = cast(list[Mapping[str, Any]], execution_plan["waves"])
    if type(wave_index) is not int or not 0 <= wave_index < len(waves):
        raise ValueError("matched billing block wave_index is invalid")
    _require_nonempty(shard_id, "shard_id")
    wave = waves[wave_index]
    planned = next(
        (
            shard
            for shard in cast(list[Mapping[str, Any]], wave["shards"])
            if shard.get("shard_id") == shard_id
        ),
        None,
    )
    if planned is None:
        raise ValueError("matched billing block shard is not planned in its wave")
    evidence = _json_mapping(evidence_record, "shard evidence")
    deletion = _json_mapping(deletion_attestation, "deletion attestation")
    if evidence.get("record_type") != FULL_SCORE_SHARD_EVIDENCE_RECORD_TYPE:
        raise ValueError("matched block requires full-score shard evidence")
    if evidence.get("closed_record_sha256") != _closed_record_sha256(evidence):
        raise ValueError("matched block shard-evidence closure drift")
    if evidence.get("durable_evidence_committed") is not True:
        raise ValueError("matched block evidence is not durably committed")
    if not _json_type_exact_equal(
        evidence.get("scorers"),
        _scorer_contract_record(),
    ):
        raise ValueError("matched block scorer/parser identity drift")
    if not _json_type_exact_equal(
        evidence.get("protocol"),
        _full_score_protocol_record(),
    ):
        raise ValueError("matched block full-score protocol drift")
    if evidence.get("execution_plan_sha256") != execution_plan.get(
        "closed_record_sha256"
    ):
        raise ValueError("matched block evidence execution-plan drift")
    if evidence.get("shard_id") != shard_id or evidence.get("wave_index") != wave_index:
        raise ValueError("matched block evidence shard identity drift")
    if evidence.get("shard_items_sha256") != planned.get("items_sha256"):
        raise ValueError("matched block evidence item closure drift")
    if deletion.get("record_type") != FULL_SCORE_DELETION_ATTESTATION_RECORD_TYPE:
        raise ValueError("matched block requires deletion attestation")
    if deletion.get("closed_record_sha256") != _closed_record_sha256(deletion):
        raise ValueError("matched block deletion closure drift")
    deletion_expected = {
        "evidence_closed_record_sha256": evidence.get("closed_record_sha256"),
        "execution_plan_sha256": execution_plan.get("closed_record_sha256"),
        "shard_id": shard_id,
        "wave_index": wave_index,
    }
    if any(deletion.get(key) != value for key, value in deletion_expected.items()):
        raise ValueError("matched block deletion/evidence identity drift")
    lifecycle = deletion.get("lifecycle")
    if not isinstance(lifecycle, list) or lifecycle[-1:] != ["delete_ephemeral_q8_kv"]:
        raise ValueError("matched block deletion is not terminal")
    billed_seconds = {
        "producer": _positive_finite_float(
            producer_billed_gpu_seconds,
            "producer_billed_gpu_seconds",
        ),
        "consumer_task": _positive_finite_float(
            consumer_task_billed_gpu_seconds,
            "consumer_task_billed_gpu_seconds",
        ),
    }
    if set(billing_source_sha256) != set(billed_seconds):
        raise ValueError("matched block requires one billing source per role")
    billing_sources = {
        role: _require_sha256(digest, field_name=f"billing_source_sha256.{role}")
        for role, digest in sorted(billing_source_sha256.items())
    }
    if evidence.get("method_wall_clock") != "time.monotonic_ns":
        raise ValueError("matched block lacks the frozen monotonic wall clock")
    raw_method_wall = _required_mapping(evidence, "method_wall_seconds")
    if set(raw_method_wall) != set(FULL_SCORE_METHODS):
        raise ValueError("matched block method wall evidence is incomplete")
    method_wall_seconds = {
        method: _positive_finite_float(
            raw_method_wall.get(method),
            f"method_wall_seconds.{method}",
        )
        for method in FULL_SCORE_METHODS
    }
    shared_or_unattributed_seconds = max(
        0.0,
        billed_seconds["consumer_task"] - sum(method_wall_seconds.values()),
    )
    output_tokens = {method: 0 for method in FULL_SCORE_METHODS}
    pairs = evidence.get("paired_examples")
    if not isinstance(pairs, list) or len(pairs) != planned.get("item_count"):
        raise ValueError("matched block paired evidence coverage is incomplete")
    observed_ids: set[tuple[str, str]] = set()
    for raw_pair in pairs:
        pair = _json_mapping(raw_pair, "paired example")
        key = (_required_string(pair, "dataset"), _required_string(pair, "example_id"))
        if key in observed_ids:
            raise ValueError("matched block paired evidence duplicates an ID")
        observed_ids.add(key)
        methods = _required_mapping(pair, "methods")
        if set(methods) != set(FULL_SCORE_METHODS):
            raise ValueError("matched block paired evidence lacks a method")
        for method in FULL_SCORE_METHODS:
            method_record = _required_mapping(methods, method)
            completion_tokens = method_record.get("completion_tokens")
            if type(completion_tokens) is not int or not 0 <= completion_tokens <= (
                FULL_SCORE_MAX_TOKENS
            ):
                raise ValueError("matched block completion-token evidence is invalid")
            output_tokens[method] += completion_tokens
    planned_ids = {
        (cast(str, item["dataset"]), cast(str, item["example_id"]))
        for item in cast(list[Mapping[str, Any]], planned["items"])
    }
    if observed_ids != planned_ids:
        raise ValueError(
            "matched block paired evidence has partial planned-ID coverage"
        )
    record: dict[str, Any] = {
        "authorization_scope": FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
        "billed_gpu_seconds": billed_seconds,
        "billing_source_sha256": billing_sources,
        "cache_prefix_tokens": planned.get("cache_prefix_tokens"),
        "closed_record_sha256": "",
        "consumer_task_diagnostics": {
            "attribution": "indivisible_no_per_arm_billed_seconds",
            "method_wall_clock": "time.monotonic_ns",
            "method_wall_seconds": method_wall_seconds,
            "shared_or_unattributed_seconds": shared_or_unattributed_seconds,
        },
        "deletion_attestation_sha256": deletion.get("closed_record_sha256"),
        "evidence_sha256": evidence.get("closed_record_sha256"),
        "execution_plan_sha256": execution_plan.get("closed_record_sha256"),
        "matched_status": "success_error_free",
        "natural_prompt_tokens": planned.get("natural_prompt_tokens"),
        "observed_completion_tokens": output_tokens,
        "protocol_sha256": _canonical_sha256(_full_score_protocol_record()),
        "record_type": FULL_SCORE_MATCHED_BLOCK_RECORD_TYPE,
        "schema_version": FULL_SCORE_MATCHED_BLOCK_SCHEMA_VERSION,
        "shard_id": shard_id,
        "shard_items_sha256": planned.get("items_sha256"),
        "wave_index": wave_index,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def build_governed_full_score_matched_billing_block(
    execution_plan: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    evidence_dir: str | Path,
    producer_terminal_path: str | Path,
    consumer_terminal_path: str | Path,
    ledger_path: str | Path,
    remote_consumer_authorization: object | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> dict[str, Any]:
    """Build matched billing only from replayed evidence and terminal files."""

    _validate_publication_full_score_inputs(inventory, shard_plan, execution_plan)
    evidence_directory_uri = _require_shared_dbfs_path(
        evidence_dir,
        "governed evidence directory",
    )
    remote_binding: Mapping[str, Any] | None = None
    if remote_consumer_authorization is None:
        if compact_artifact_resolver is not None:
            raise TypeError(
                "matched billing compact CAS requires remote consumer authority"
            )
        evidence, deletion = load_governed_full_score_shard_evidence(
            evidence_dir,
            inventory=inventory,
            shard_plan=shard_plan,
            execution_plan=execution_plan,
            require_deletion=True,
        )
    else:
        from document_kv_cache.full_score_remote_control import (
            require_full_score_remote_consumer_evidence_authorization,
        )

        remote_authorization = (
            require_full_score_remote_consumer_evidence_authorization(
                remote_consumer_authorization,
                execution_plan=execution_plan,
            )
        )
        remote_records = _remote_consumer_evidence_records(
            remote_authorization,
            completion_record=remote_authorization.result_record,
            inventory=inventory,
            shard_plan=shard_plan,
            execution_plan=execution_plan,
            expected_wave_index=remote_authorization.wave_index,
            compact_artifact_resolver=compact_artifact_resolver,
        )
        matching_bindings = [
            binding
            for binding in remote_authorization.evidence_bindings
            if _required_string(binding, "evidence_uri").rsplit("/", 1)[0]
            == evidence_directory_uri.rstrip("/")
        ]
        if len(matching_bindings) != 1:
            raise ValueError("matched billing remote evidence directory drift")
        remote_binding = matching_bindings[0]
        evidence, deletion = remote_records[
            _required_string(remote_binding, "shard_id")
        ]
    if deletion is None:
        raise ValueError("governed matched billing requires deletion evidence")
    producer = load_governed_full_score_phase_terminal_record(
        producer_terminal_path,
        execution_plan=execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        ledger_path=ledger_path,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    consumer = load_governed_full_score_phase_terminal_record(
        consumer_terminal_path,
        execution_plan=execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        ledger_path=ledger_path,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    wave_index = _required_int(evidence, "wave_index")
    shard_id = _required_string(evidence, "shard_id")
    billed: dict[str, float] = {}
    durable_roots: set[str] = set()
    for role, terminal in (("producer", producer), ("consumer_task", consumer)):
        expected_phase = "producer" if role == "producer" else "consumer"
        if (
            terminal.get("phase") != expected_phase
            or terminal.get("wave_index") != wave_index
        ):
            raise ValueError("matched billing terminal belongs to another phase/wave")
        task_billing = terminal.get("task_billing")
        if not isinstance(task_billing, list):
            raise ValueError("matched billing terminal lacks task billing")
        matching = [
            _json_mapping(item, "phase task billing")
            for item in task_billing
            if isinstance(item, Mapping) and item.get("shard_id") == shard_id
        ]
        if len(matching) != 1:
            raise ValueError(
                "matched billing requires exactly one task per shard/phase"
            )
        billed[role] = _positive_finite_float(
            matching[0].get("billed_gpu_seconds"),
            f"{role} billed GPU seconds",
        )
        durable_roots.add(
            _require_shared_dbfs_path(
                _required_string(matching[0], "durable_output_root"),
                f"{role} durable_output_root",
            ).rstrip("/")
        )
    expected_evidence_directories = {
        f"{root}/evidence/wave-{wave_index:03d}/{shard_id}" for root in durable_roots
    }
    if len(durable_roots) != 1 or expected_evidence_directories != {
        evidence_directory_uri.rstrip("/")
    }:
        raise ValueError("matched billing evidence path differs from worker manifests")
    record = build_full_score_matched_billing_block(
        execution_plan,
        wave_index=wave_index,
        shard_id=shard_id,
        evidence_record=evidence,
        deletion_attestation=deletion,
        producer_billed_gpu_seconds=billed["producer"],
        consumer_task_billed_gpu_seconds=billed["consumer_task"],
        billing_source_sha256={
            "consumer_task": _required_string(consumer, "closed_record_sha256"),
            "producer": _required_string(producer, "closed_record_sha256"),
        },
    )
    if remote_binding is None:
        evidence_path = _governed_existing_file(
            f"{evidence_directory_uri.rstrip('/')}/evidence.json",
            "matched billing evidence file",
        )
        deletion_path = _governed_existing_file(
            f"{evidence_directory_uri.rstrip('/')}/deletion-attestation.json",
            "matched billing deletion file",
        )
        evidence_file_sha256 = sha256(evidence_path.read_bytes()).hexdigest()
        deletion_file_sha256 = sha256(deletion_path.read_bytes()).hexdigest()
    else:
        evidence_file_sha256 = _required_string(
            remote_binding,
            "evidence_file_sha256",
        )
        deletion_file_sha256 = _required_string(
            remote_binding,
            "deletion_file_sha256",
        )
    producer_file = _governed_compact_file(
        producer_terminal_path,
        "producer terminal record",
        compact_artifact_resolver,
    )
    consumer_file = _governed_compact_file(
        consumer_terminal_path,
        "consumer terminal record",
        compact_artifact_resolver,
    )
    record["authorization_scope"] = FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
    producer_lineage = _required_mapping(producer, "ledger")
    consumer_lineage = _required_mapping(consumer, "ledger")
    if producer_lineage.get("ledger_path_sha256") != consumer_lineage.get(
        "ledger_path_sha256"
    ) or consumer_lineage.get("predecessor_prefix") != producer_lineage.get(
        "terminal_prefix"
    ):
        raise ValueError("matched billing phase ledger lineage is not contiguous")
    record["ledger_lineage"] = {
        "consumer": dict(consumer_lineage),
        "producer": dict(producer_lineage),
    }
    record["governed_sources"] = {
        "consumer_terminal": {
            "file_sha256": sha256(consumer_file.read_bytes()).hexdigest(),
            "path": str(consumer_terminal_path),
            "record_sha256": consumer["closed_record_sha256"],
        },
        "evidence": {
            "deletion_file_sha256": deletion_file_sha256,
            "deletion_record_sha256": deletion["closed_record_sha256"],
            "directory": str(evidence_dir),
            "evidence_file_sha256": evidence_file_sha256,
            "evidence_record_sha256": evidence["closed_record_sha256"],
        },
        "producer_terminal": {
            "file_sha256": sha256(producer_file.read_bytes()).hexdigest(),
            "path": str(producer_terminal_path),
            "record_sha256": producer["closed_record_sha256"],
        },
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def write_governed_full_score_matched_billing_block(
    path: str | Path,
    execution_plan: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Write and reread one publication-authorizing matched billing block."""

    compact_artifact_publisher = kwargs.pop(
        "compact_artifact_publisher",
        None,
    )
    compact_artifact_resolver = kwargs.get("compact_artifact_resolver")
    if (compact_artifact_publisher is None) != (compact_artifact_resolver is None):
        raise TypeError(
            "matched billing compact publication requires publisher and CAS"
        )
    record = build_governed_full_score_matched_billing_block(
        execution_plan,
        **kwargs,
    )
    _publish_governed_compact_file(
        path,
        "matched block output path",
        _canonical_pretty_json_bytes(record),
        compact_artifact_publisher,
    )
    return load_governed_full_score_matched_billing_block(
        path,
        execution_plan=execution_plan,
        inventory=kwargs["inventory"],
        shard_plan=kwargs["shard_plan"],
        ledger_path=kwargs["ledger_path"],
        remote_consumer_authorization=kwargs.get("remote_consumer_authorization"),
        compact_artifact_resolver=kwargs.get("compact_artifact_resolver"),
    )


def load_governed_full_score_matched_billing_block(
    path: str | Path,
    *,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    ledger_path: str | Path,
    remote_consumer_authorization: object | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> dict[str, Any]:
    """Reread and fully replay one publication matched billing block."""

    block_file = _governed_compact_file(
        path,
        "matched billing block",
        compact_artifact_resolver,
    )
    block = _json_object(block_file.read_bytes(), "matched billing block")
    if block.get("authorization_scope") != FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE:
        raise ValueError("matched billing block is not publication-authorizing")
    if block.get("closed_record_sha256") != _closed_record_sha256(block):
        raise ValueError("matched billing block closure drift")
    sources = _required_mapping(block, "governed_sources")
    evidence_source = _required_mapping(sources, "evidence")
    rebuilt = build_governed_full_score_matched_billing_block(
        execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        evidence_dir=_required_string(evidence_source, "directory"),
        producer_terminal_path=_required_string(
            _required_mapping(sources, "producer_terminal"),
            "path",
        ),
        consumer_terminal_path=_required_string(
            _required_mapping(sources, "consumer_terminal"),
            "path",
        ),
        ledger_path=ledger_path,
        remote_consumer_authorization=remote_consumer_authorization,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    if rebuilt != block:
        raise ValueError("matched billing block does not replay from governed sources")
    return block


def build_full_score_live_p90_budget_admission(
    execution_plan: Mapping[str, Any],
    completed_blocks: Sequence[Mapping[str, Any]],
    *,
    next_wave_index: int,
    ledger_terminal_actual_gpu_hours: float,
    ledger_active_reserved_gpu_hours: float,
    next_wave_reserved_gpu_hours: float,
    next_phase: str = "producer",
) -> dict[str, Any]:
    """Bootstrap a matched-block P90 and close one wave-boundary budget gate."""

    _require_publication_rng_runtime()
    _validate_budget_execution_plan(execution_plan)
    waves = cast(list[Mapping[str, Any]], execution_plan["waves"])
    if type(next_wave_index) is not int or not 1 <= next_wave_index < len(waves):
        raise ValueError("next_wave_index must identify an unsubmitted nonzero wave")
    if next_phase not in {"producer", "consumer"}:
        raise ValueError("next_phase must be producer or consumer")
    terminal_actual = _nonnegative_finite_float(
        ledger_terminal_actual_gpu_hours,
        "ledger_terminal_actual_gpu_hours",
    )
    active_reserved = _nonnegative_finite_float(
        ledger_active_reserved_gpu_hours,
        "ledger_active_reserved_gpu_hours",
    )
    next_reserved = _nonnegative_finite_float(
        next_wave_reserved_gpu_hours,
        "next_wave_reserved_gpu_hours",
    )
    if active_reserved != 0:
        raise ValueError("live P90 gates may run only at active-reservation zero")
    first_wave_ids = cast(list[str], waves[0]["shard_ids"])
    if len(first_wave_ids) != FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE:
        raise ValueError("wave zero must be the frozen sixteen-shard pilot")
    planned_by_id: dict[str, tuple[int, Mapping[str, Any]]] = {}
    expected_completed_ids: set[str] = set()
    remaining_producer_shards: list[Mapping[str, Any]] = []
    remaining_consumer_shards: list[Mapping[str, Any]] = []
    for wave_index, wave in enumerate(waves):
        for shard in cast(list[Mapping[str, Any]], wave["shards"]):
            shard_id = cast(str, shard["shard_id"])
            if shard_id in planned_by_id:
                raise ValueError("execution plan duplicates a shard across waves")
            planned_by_id[shard_id] = (wave_index, shard)
            if wave_index < next_wave_index:
                expected_completed_ids.add(shard_id)
            if wave_index >= next_wave_index:
                remaining_consumer_shards.append(shard)
            if wave_index > next_wave_index or (
                wave_index == next_wave_index and next_phase == "producer"
            ):
                remaining_producer_shards.append(shard)
    normalized_blocks: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for raw_block in completed_blocks:
        block = _json_mapping(raw_block, "matched billing block")
        _validate_matched_billing_block(
            block,
            execution_plan=execution_plan,
            planned_by_id=planned_by_id,
        )
        shard_id = _required_string(block, "shard_id")
        if shard_id in observed_ids:
            raise ValueError("live P90 input duplicates a matched shard block")
        observed_ids.add(shard_id)
        normalized_blocks.append(block)
    if observed_ids != expected_completed_ids:
        raise ValueError(
            "live P90 requires every prior-wave matched block exactly once"
        )
    wave_zero_count = sum(block["wave_index"] == 0 for block in normalized_blocks)
    if wave_zero_count != FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE:
        raise ValueError("wave zero lacks sixteen complete matched blocks")
    normalized_blocks.sort(key=lambda block: (block["wave_index"], block["shard_id"]))
    completed_billed_hours = (
        sum(
            sum(cast(Mapping[str, float], block["billed_gpu_seconds"]).values())
            for block in normalized_blocks
        )
        / 3600.0
    )
    if terminal_actual + 1e-12 < completed_billed_hours:
        raise ValueError("terminal ledger omits matched full-score billed GPU-hours")
    remaining_cache_tokens = sum(
        cast(int, shard["cache_prefix_tokens"]) for shard in remaining_producer_shards
    )
    remaining_natural_tokens = sum(
        cast(int, shard["natural_prompt_tokens"]) for shard in remaining_consumer_shards
    )
    if remaining_cache_tokens < 0 or remaining_natural_tokens <= 0:
        raise ValueError(
            "live P90 requires nonnegative producer and positive consumer work"
        )
    strata: dict[int, list[int]] = defaultdict(list)
    for index, block in enumerate(normalized_blocks):
        strata[cast(int, block["wave_index"])].append(index)
    rng = random.Random(FULL_SCORE_LIVE_P90_SEED)
    sample_indices: list[list[int]] = []
    projected_hours: list[float] = []
    for _draw_index in range(FULL_SCORE_LIVE_P90_DRAWS):
        draw_indices: list[int] = []
        for wave_index in sorted(strata):
            indices = strata[wave_index]
            draw_indices.extend(rng.choices(indices, k=len(indices)))
        sample_indices.append(draw_indices)
        sampled = [normalized_blocks[index] for index in draw_indices]
        sampled_cache_tokens = sum(
            cast(int, block["cache_prefix_tokens"]) for block in sampled
        )
        sampled_natural_tokens = sum(
            cast(int, block["natural_prompt_tokens"]) for block in sampled
        )
        producer_seconds = sum(
            cast(Mapping[str, float], block["billed_gpu_seconds"])["producer"]
            for block in sampled
        )
        consumer_seconds = sum(
            cast(Mapping[str, float], block["billed_gpu_seconds"])["consumer_task"]
            for block in sampled
        )
        projected_seconds = (
            producer_seconds / sampled_cache_tokens * remaining_cache_tokens
            if remaining_cache_tokens
            else 0.0
        ) + consumer_seconds / sampled_natural_tokens * remaining_natural_tokens
        projected_hours.append(projected_seconds / 3600.0)
    p90_remaining = _type7_quantile(sorted(projected_hours), 0.90)
    projected_with_headroom = (
        terminal_actual + p90_remaining + PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
    )
    projected_live_reserved = active_reserved + next_reserved
    aggregate_gate_passed = (
        projected_with_headroom <= MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
    )
    live_reservation_gate_passed = (
        projected_live_reserved <= MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
    )
    record: dict[str, Any] = {
        "admitted": aggregate_gate_passed and live_reservation_gate_passed,
        "authorization_scope": FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
        "bootstrap": {
            "draws": FULL_SCORE_LIVE_P90_DRAWS,
            "matched_block_projection_sha256": _canonical_sha256(normalized_blocks),
            "percentile": 0.90,
            "percentile_method": "linear_interpolated_empirical_quantile_type7_v1",
            "projected_remaining_gpu_hours_samples": projected_hours,
            "projected_remaining_gpu_hours_samples_sha256": _canonical_sha256(
                projected_hours
            ),
            "resample_indices": sample_indices,
            "resample_indices_sha256": _canonical_sha256(sample_indices),
            "resampling_unit": "matched_shard_block_producer_plus_consumer_task",
            "rng_algorithm": "cpython-3.11-random.Random-mt19937-choices-v1",
            "seed": FULL_SCORE_LIVE_P90_SEED,
            "stratification": "completed_wave",
        },
        "closed_record_sha256": "",
        "completed_block_count": len(normalized_blocks),
        "completed_blocks": normalized_blocks,
        "execution_plan_sha256": execution_plan.get("closed_record_sha256"),
        "formula": {
            "admission": "terminal_actual + p90_remaining + 124 <= 1024",
            "consumer_task_scale": (
                "sum(sampled_consumer_task_billed_gpu_seconds) / "
                "sum(sampled_natural_prompt_tokens) * remaining_natural_prompt_tokens; "
                "Baseline+Vanilla remain indivisible"
            ),
            "matched_resampling": True,
            "producer_scale": (
                "sum(sampled_producer_billed_gpu_seconds) / "
                "sum(sampled_cache_prefix_tokens) * remaining_cache_prefix_tokens"
            ),
        },
        "ledger_after_projection": {
            "aggregate_cap_gpu_hours": MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
            "aggregate_gate_passed": aggregate_gate_passed,
            "live_reservation_cap_gpu_hours": (
                MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
            ),
            "live_reservation_gate_passed": live_reservation_gate_passed,
            "p90_remaining_unsubmitted_gpu_hours": p90_remaining,
            "projected_live_reserved_gpu_hours": projected_live_reserved,
            "projected_terminal_plus_p90_gpu_hours": (terminal_actual + p90_remaining),
            "projected_with_required_headroom_gpu_hours": projected_with_headroom,
            "required_unreserved_headroom_gpu_hours": (
                PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
            ),
        },
        "ledger_before": {
            "active_reserved_gpu_hours": active_reserved,
            "matched_completed_billed_gpu_hours": completed_billed_hours,
            "next_wave_reserved_gpu_hours": next_reserved,
            "terminal_actual_gpu_hours": terminal_actual,
        },
        "next_wave_index": next_wave_index,
        "next_phase": next_phase,
        "record_type": FULL_SCORE_LIVE_P90_RECORD_TYPE,
        "remaining_work": {
            "cache_prefix_tokens": remaining_cache_tokens,
            "natural_prompt_tokens": remaining_natural_tokens,
            "consumer_planned_sha256": _canonical_sha256(remaining_consumer_shards),
            "consumer_shard_count": len(remaining_consumer_shards),
            "producer_planned_sha256": _canonical_sha256(remaining_producer_shards),
            "producer_shard_count": len(remaining_producer_shards),
        },
        "schema_version": FULL_SCORE_LIVE_P90_SCHEMA_VERSION,
        "wave_boundary_active_zero": True,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def build_governed_full_score_live_p90_budget_admission(
    execution_plan: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    completed_block_paths: Sequence[str | Path],
    next_wave_index: int,
    next_phase: str,
    attempt_id: str,
    next_submit_payload: Mapping[str, Any],
    ledger_path: str | Path,
    predecessor_authorization: object,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
    _require_latest_predecessor: bool = True,
) -> dict[str, Any]:
    """Build a one-attempt gate against the exact canonical predecessor."""

    _validate_publication_full_score_inputs(inventory, shard_plan, execution_plan)
    _require_nonempty(attempt_id, "attempt_id")
    task_bindings = _validated_full_score_phase_submit_payload(
        execution_plan,
        next_submit_payload,
        inventory=inventory,
        shard_plan=shard_plan,
        wave_index=next_wave_index,
        phase=next_phase,
        require_governed_consumer_ready_phase=True,
        remote_ready_authorization=remote_ready_authorization,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    live_path = Path(ledger_path).expanduser().absolute()
    predecessor_prefix, predecessor_lineage = (
        _require_full_score_phase_predecessor_authorization(
            predecessor_authorization,
            execution_plan=execution_plan,
            ledger_path=live_path,
            wave_index=next_wave_index,
            phase=next_phase,
            latency_execution_plan_record=latency_execution_plan_record,
            require_latest=_require_latest_predecessor,
        )
    )
    _require_full_score_remote_workspace_lineage(
        predecessor_authorization,
        execution_plan=execution_plan,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
    )
    ledger_path_sha256 = databricks_ledger_path_sha256(live_path)
    ledger = read_databricks_cluster_hour_ledger_json(live_path)
    if not _require_latest_predecessor:
        ledger = _full_score_ledger_prefix_state(ledger, predecessor_prefix)
    elif databricks_ledger_prefix(ledger) != predecessor_prefix:
        raise ValueError("live ledger changed while building the P90 admission")
    if ledger.active_reserved_cluster_hours != 0:
        raise ValueError("live P90 admission requires a zero-active live ledger")
    if any(item.attempt_id == attempt_id for item in ledger.reservations):
        raise ValueError("live P90 admission attempt_id is already consumed")
    blocks: list[dict[str, Any]] = []
    block_bindings: list[dict[str, Any]] = []
    remote_by_wave = _remote_consumer_authorizations_by_wave(
        ()
        if remote_consumer_authorizations is None
        else remote_consumer_authorizations,
        execution_plan=execution_plan,
    )
    if compact_artifact_resolver is not None and not remote_by_wave:
        raise TypeError("live P90 compact CAS requires remote consumer authority")
    if compact_artifact_resolver is None and remote_by_wave:
        raise TypeError("live P90 remote consumer authority requires compact CAS")
    if remote_by_wave and set(remote_by_wave) != set(range(next_wave_index)):
        raise ValueError(
            "live P90 requires every completed-wave remote authority exactly once"
        )
    for raw_path in completed_block_paths:
        block_file = _governed_compact_file(
            raw_path,
            "matched billing block",
            compact_artifact_resolver,
        )
        candidate_block = _json_object(
            block_file.read_bytes(),
            "matched billing block",
        )
        block_wave_index = _required_int(candidate_block, "wave_index")
        block_remote_authorization = remote_by_wave.get(block_wave_index)
        if remote_by_wave and block_remote_authorization is None:
            raise ValueError("live P90 omits a completed-wave remote authority")
        block = load_governed_full_score_matched_billing_block(
            raw_path,
            execution_plan=execution_plan,
            inventory=inventory,
            shard_plan=shard_plan,
            ledger_path=live_path,
            remote_consumer_authorization=block_remote_authorization,
            compact_artifact_resolver=compact_artifact_resolver,
        )
        block_lineage = _required_mapping(block, "ledger_lineage")
        producer_lineage = _required_mapping(block_lineage, "producer")
        consumer_lineage = _required_mapping(block_lineage, "consumer")
        if (
            producer_lineage.get("ledger_path_sha256") != ledger_path_sha256
            or consumer_lineage.get("ledger_path_sha256") != ledger_path_sha256
        ):
            raise ValueError("matched block belongs to a copied campaign ledger")
        if (
            next_phase == "producer"
            and block.get("wave_index") == next_wave_index - 1
            and consumer_lineage.get("terminal_prefix")
            != predecessor_prefix.to_record()
        ):
            raise ValueError("P90 predecessor omits the prior consumer terminal")
        blocks.append(block)
        block_bindings.append(
            {
                "file_sha256": sha256(block_file.read_bytes()).hexdigest(),
                "path": str(raw_path),
                "record_sha256": block["closed_record_sha256"],
                "shard_id": block["shard_id"],
            }
        )
    reservation = databricks_submit_payload_reservation(
        next_submit_payload,
        attempt_id=attempt_id,
        workload_id=_full_score_phase_workload_id(
            execution_plan,
            wave_index=next_wave_index,
            phase=next_phase,
        ),
    )
    if len(reservation.task_timeout_seconds) != len(task_bindings):
        raise ValueError("next phase reservation task coverage drift")
    gate = build_full_score_live_p90_budget_admission(
        execution_plan,
        blocks,
        next_wave_index=next_wave_index,
        ledger_terminal_actual_gpu_hours=ledger.terminal_actual_cluster_hours,
        ledger_active_reserved_gpu_hours=ledger.active_reserved_cluster_hours,
        next_wave_reserved_gpu_hours=reservation.reserved_cluster_hours,
        next_phase=next_phase,
    )
    _snapshot, canonical_submit = canonical_databricks_submit_payload_snapshot(
        next_submit_payload
    )
    gate["attempt_id"] = attempt_id
    gate["authorization_scope"] = FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
    gate["ledger"] = {
        "ledger_id": ledger.ledger_id,
        "ledger_path_sha256": ledger_path_sha256,
        "predecessor_prefix": predecessor_prefix.to_record(),
        "record_sha256": _canonical_sha256(
            databricks_cluster_hour_ledger_to_record(ledger)
        ),
    }
    gate["predecessor_lineage"] = predecessor_lineage
    gate["matched_block_files"] = sorted(
        block_bindings,
        key=lambda item: (item["shard_id"], item["path"]),
    )
    if remote_by_wave:
        gate["remote_consumer_authorizations"] = [
            {
                "authorization_record_sha256": (
                    authorization.controller_authorization_record_sha256
                ),
                "result_record_sha256": authorization.result_record_sha256,
                "wave_index": wave_index,
            }
            for wave_index, authorization in sorted(remote_by_wave.items())
        ]
    gate["next_phase"] = next_phase
    gate["next_submit_payload_sha256"] = sha256(canonical_submit).hexdigest()
    gate["closed_record_sha256"] = _closed_record_sha256(gate)
    return gate


def write_governed_full_score_live_p90_budget_admission(
    path: str | Path,
    execution_plan: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Write and reread a one-attempt publication P90 admission."""

    compact_artifact_publisher = kwargs.pop(
        "compact_artifact_publisher",
        None,
    )
    compact_artifact_resolver = kwargs.get("compact_artifact_resolver")
    if (compact_artifact_publisher is None) != (compact_artifact_resolver is None):
        raise TypeError("live P90 compact publication requires publisher and CAS")
    record = build_governed_full_score_live_p90_budget_admission(
        execution_plan,
        **kwargs,
    )
    _publish_governed_compact_file(
        path,
        "live P90 admission output path",
        _canonical_pretty_json_bytes(record),
        compact_artifact_publisher,
    )
    return load_governed_full_score_live_p90_budget_admission(
        path,
        execution_plan=execution_plan,
        inventory=kwargs["inventory"],
        shard_plan=kwargs["shard_plan"],
        next_submit_payload=kwargs["next_submit_payload"],
        ledger_path=kwargs["ledger_path"],
        predecessor_authorization=kwargs["predecessor_authorization"],
        latency_execution_plan_record=kwargs.get("latency_execution_plan_record"),
        remote_ready_authorization=kwargs.get("remote_ready_authorization"),
        remote_consumer_authorizations=kwargs.get("remote_consumer_authorizations"),
        compact_artifact_resolver=kwargs.get("compact_artifact_resolver"),
    )


def load_governed_full_score_live_p90_budget_admission(
    path: str | Path,
    *,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    next_submit_payload: Mapping[str, Any],
    ledger_path: str | Path,
    predecessor_authorization: object,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
    _require_latest_predecessor: bool = True,
) -> dict[str, Any]:
    """Reread and replay a publication P90 gate from immutable source files."""

    _require_full_score_remote_workspace_lineage(
        predecessor_authorization,
        execution_plan=execution_plan,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
    )
    admission_file = _governed_compact_file(
        path,
        "live P90 admission",
        compact_artifact_resolver,
    )
    record = _json_object(admission_file.read_bytes(), "live P90 admission")
    if record.get("authorization_scope") != FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE:
        raise ValueError("live P90 admission is not publication-authorizing")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("live P90 admission closure drift")
    block_files = record.get("matched_block_files")
    if not isinstance(block_files, list):
        raise ValueError("live P90 admission lacks matched-block file bindings")
    rebuilt = build_governed_full_score_live_p90_budget_admission(
        execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        completed_block_paths=[
            _required_string(
                _json_mapping(item, "matched block file binding"),
                "path",
            )
            for item in block_files
        ],
        next_wave_index=_required_int(record, "next_wave_index"),
        next_phase=_required_string(record, "next_phase"),
        attempt_id=_required_string(record, "attempt_id"),
        next_submit_payload=next_submit_payload,
        ledger_path=ledger_path,
        predecessor_authorization=predecessor_authorization,
        latency_execution_plan_record=latency_execution_plan_record,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_artifact_resolver,
        _require_latest_predecessor=_require_latest_predecessor,
    )
    if rebuilt != record:
        raise ValueError("live P90 admission does not replay from governed sources")
    return record


def _require_full_score_phase_launch_authorization(
    task_bindings: Sequence[Mapping[str, Any]],
    authorization: object,
) -> GPUQualificationSelection:
    if not task_bindings:
        raise ValueError("full-score launch authorization has no task bindings")
    plan_sha256s = {
        _required_string(binding, "qualification_plan_sha256")
        for binding in task_bindings
    }
    evidence_sha256s = {
        _required_string(binding, "qualification_evidence_file_sha256")
        for binding in task_bindings
    }
    selection_records = [
        _required_mapping(binding, "qualification_selection")
        for binding in task_bindings
    ]
    if (
        len(plan_sha256s) != 1
        or len(evidence_sha256s) != 1
        or len({_canonical_sha256(item) for item in selection_records}) != 1
    ):
        raise ValueError("full-score tasks do not share one GPU qualification closure")
    selection = require_gpu_qualification_launch_authorization(
        authorization,
        expected_plan_sha256=next(iter(plan_sha256s)),
        expected_evidence_file_sha256=next(iter(evidence_sha256s)),
    )
    _validate_full_score_gpu_selection(selection)
    if _gpu_selection_record(selection) != dict(selection_records[0]):
        raise ValueError(
            "GPU qualification launch authority selection differs from worker payloads"
        )
    return selection


def _require_full_score_phase_predecessor_authorization(
    authorization: object,
    *,
    execution_plan: Mapping[str, Any],
    ledger_path: str | Path,
    wave_index: int,
    phase: str,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
    require_latest: bool = True,
) -> tuple[DatabricksLedgerPrefix, dict[str, Any]]:
    """Require the immediately preceding causal phase on one local ledger."""

    _validate_full_score_phase_position(wave_index, phase)
    waves = execution_plan.get("waves")
    if not isinstance(waves, list) or not 0 <= wave_index < len(waves):
        raise ValueError("full-score phase is outside the frozen execution plan")
    plan_sha256 = _require_sha256(
        execution_plan.get("closed_record_sha256"),
        field_name="full-score execution-plan SHA-256",
    )
    if (
        not (wave_index == 0 and phase == "producer")
        and type(authorization) is not FullScorePhaseAuthorization
    ):
        raise TypeError("full-score phase launch requires FullScorePhaseAuthorization")
    _full_score_predecessor_workspace_identity(authorization)
    path_sha256 = databricks_ledger_path_sha256(ledger_path)
    live = read_databricks_cluster_hour_ledger_json(ledger_path)
    _require_full_score_ledger_caps(live)
    lineage: dict[str, Any]
    if wave_index == 0 and phase == "producer":
        if not isinstance(latency_execution_plan_record, Mapping):
            raise TypeError(
                "wave-zero producer requires the latency execution-plan record"
            )
        if require_latest:
            prefix = require_publication_latency_collection_authorization(
                authorization,
                execution_plan_record=latency_execution_plan_record,
                ledger_path=ledger_path,
            )
        else:
            latency_authorization = cast(
                PublicationLatencyCollectionAuthorization, authorization
            )
            validate_publication_latency_collection_record(
                latency_authorization.collection,
                execution_plan_record=latency_execution_plan_record,
            )
            if latency_authorization.ledger_path_sha256 != path_sha256:
                raise ValueError("latency collection ledger path binding drift")
            prefix = latency_authorization.ledger_prefix
            require_databricks_ledger_prefix(live, prefix)
        collection_sha256 = _require_sha256(
            getattr(authorization, "collection_sha256", None),
            field_name="latency collection SHA-256",
        )
        lineage = {
            "authorization_sha256": collection_sha256,
            "kind": "publication_latency_collection",
            "ledger_path_sha256": path_sha256,
            "ledger_prefix": prefix.to_record(),
        }
    else:
        phase_authorization = cast(FullScorePhaseAuthorization, authorization)
        expected_wave_index = wave_index if phase == "consumer" else wave_index - 1
        expected_phase = "producer" if phase == "consumer" else "consumer"
        if (
            phase_authorization.execution_plan_sha256 != plan_sha256
            or phase_authorization.wave_index != expected_wave_index
            or phase_authorization.phase != expected_phase
            or phase_authorization.ledger_path_sha256 != path_sha256
        ):
            raise ValueError("full-score phase predecessor ordering/path binding drift")
        expected_causal = _canonical_sha256(
            {
                "batch_prefix": databricks_ledger_prefix_at_counts(
                    live,
                    reservation_count=phase_authorization.predecessor_prefix.reservation_count
                    + 1,
                    submission_receipt_count=phase_authorization.predecessor_prefix.submission_receipt_count,
                    terminal_actual_count=phase_authorization.predecessor_prefix.terminal_actual_count,
                ).to_record(),
                "ledger_path_sha256": phase_authorization.ledger_path_sha256,
                "terminal_prefix": phase_authorization.ledger_prefix.to_record(),
                "terminal_record_sha256": phase_authorization.terminal_record_sha256,
            }
        )
        if phase_authorization.causal_closure_sha256 != expected_causal:
            raise ValueError("full-score predecessor causal workspace binding drift")
        expected_workspace_causal = _canonical_sha256(
            {
                "causal_closure_sha256": phase_authorization.causal_closure_sha256,
                "phase_lease_root_sha256": _canonical_sha256(
                    {
                        "domain": "cachet.full_score_phase_lease_root_authority.v1",
                        "phase_lease_root": str(phase_authorization.phase_lease_root),
                    }
                ),
                "user_name_sha256": phase_authorization.user_name_sha256,
                "workspace_host_sha256": phase_authorization.workspace_host_sha256,
            }
        )
        if (
            phase_authorization.workspace_authority_closure_sha256
            != expected_workspace_causal
        ):
            raise ValueError("full-score predecessor workspace authority drift")
        prefix = phase_authorization.ledger_prefix
        require_databricks_ledger_prefix(live, prefix)
        lineage = {
            "authorization_sha256": phase_authorization.causal_closure_sha256,
            "kind": "full_score_phase_terminal",
            "ledger_path_sha256": path_sha256,
            "ledger_prefix": prefix.to_record(),
            "phase": phase_authorization.phase,
            "terminal_record_sha256": phase_authorization.terminal_record_sha256,
            "wave_index": phase_authorization.wave_index,
        }
    if require_latest and databricks_ledger_prefix(live) != prefix:
        raise ValueError(
            "full-score predecessor is not the complete current ledger prefix"
        )
    return prefix, lineage


def _full_score_predecessor_workspace_identity(
    authorization: object,
) -> tuple[str, str]:
    if type(authorization) is PublicationLatencyCollectionAuthorization:
        latency_authorization = authorization
        return (
            latency_authorization.workspace_host_sha256,
            latency_authorization.user_name_sha256,
        )
    if type(authorization) is FullScorePhaseAuthorization:
        return (
            authorization.workspace_host_sha256,
            authorization.user_name_sha256,
        )
    raise TypeError("full-score predecessor lacks workspace authority")


def _require_full_score_remote_workspace_lineage(
    predecessor_authorization: object,
    *,
    execution_plan: Mapping[str, Any],
    remote_ready_authorization: object | None,
    remote_consumer_authorizations: Sequence[object] | None,
) -> None:
    expected = _full_score_predecessor_workspace_identity(predecessor_authorization)
    if remote_ready_authorization is not None:
        from document_kv_cache.full_score_remote_control import (
            _require_remote_tree_workspace_pair,
        )

        if (
            _require_remote_tree_workspace_pair(
                remote_ready_authorization,
                expected_action="producer_ready",
            )
            != expected
        ):
            raise ValueError("full-score remote ready workspace lineage drift")
    remote_by_wave = _remote_consumer_authorizations_by_wave(
        ()
        if remote_consumer_authorizations is None
        else remote_consumer_authorizations,
        execution_plan=execution_plan,
    )
    if any(
        (authorization.workspace_host_sha256, authorization.user_name_sha256)
        != expected
        for authorization in remote_by_wave.values()
    ):
        raise ValueError("full-score remote consumer workspace lineage drift")


def _require_full_score_bound_workspace_identity(
    workspace: DatabricksWorkspaceConfig,
    *,
    authorization: FullScorePhaseSubmissionAuthorization,
    submit_payload: Mapping[str, Any],
    opener: DatabricksURLOpener | None,
) -> None:
    identity = require_databricks_current_user_name(
        workspace,
        expected_user_name=_full_score_phase_single_user_name(submit_payload),
        opener=opener,
    )
    if (
        identity.get("workspace_host_sha256") != authorization.workspace_host_sha256
        or identity.get("user_name_sha256") != authorization.user_name_sha256
    ):
        raise ValueError("full-score phase workspace/principal authority drift")


def _governed_full_score_phase_reservation_validator(
    ledger_path: str | Path,
    submit_payload: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    wave_index: int,
    phase: str,
    attempt_id: str,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    predecessor_authorization: object,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
    budget_admission_path: str | Path | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> tuple[
    Path,
    DatabricksLedgerPrefix,
    Callable[
        [
            DatabricksClusterHourLedger,
            tuple[DatabricksClusterHourReservation, ...],
            tuple[Mapping[str, Any], ...],
        ],
        None,
    ],
    dict[str, Any] | None,
]:
    """Close inputs and return the one-lock atomic phase admission policy."""

    _validate_publication_full_score_inputs(inventory, shard_plan, execution_plan)
    require_databricks_run_idempotency_token(
        submit_payload,
        attempt_id=attempt_id,
    )
    task_bindings = _validated_full_score_phase_submit_payload(
        execution_plan,
        submit_payload,
        inventory=inventory,
        shard_plan=shard_plan,
        wave_index=wave_index,
        phase=phase,
        require_governed_consumer_ready_phase=True,
        remote_ready_authorization=remote_ready_authorization,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    _require_full_score_phase_launch_authorization(
        task_bindings,
        qualification_launch_authorization,
    )
    live_path = Path(ledger_path).expanduser().absolute()
    predecessor_prefix, _lineage = _require_full_score_phase_predecessor_authorization(
        predecessor_authorization,
        execution_plan=execution_plan,
        ledger_path=live_path,
        wave_index=wave_index,
        phase=phase,
        latency_execution_plan_record=latency_execution_plan_record,
    )
    _require_full_score_remote_workspace_lineage(
        predecessor_authorization,
        execution_plan=execution_plan,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
    )
    path_sha256 = databricks_ledger_path_sha256(live_path)
    if getattr(qualification_launch_authorization, "ledger_id", None) != (
        predecessor_prefix.ledger_id
    ):
        raise ValueError(
            "full-score ledger differs from the campaign qualification ledger"
        )
    qualification_path_sha256 = getattr(
        qualification_launch_authorization,
        "ledger_path_sha256",
        path_sha256,
    )
    if qualification_path_sha256 != path_sha256:
        raise ValueError("full-score qualification ledger path binding drift")
    phase_workload_id = _full_score_phase_workload_id(
        execution_plan,
        wave_index=wave_index,
        phase=phase,
    )
    admission: dict[str, Any] | None
    if wave_index == 0:
        if budget_admission_path is not None:
            raise ValueError("wave zero must not consume a live P90 admission")
        admission = None
    else:
        if budget_admission_path is None:
            raise ValueError("nonzero waves require a governed live P90 admission")
        admission = load_governed_full_score_live_p90_budget_admission(
            budget_admission_path,
            execution_plan=execution_plan,
            inventory=inventory,
            shard_plan=shard_plan,
            next_submit_payload=submit_payload,
            ledger_path=live_path,
            predecessor_authorization=predecessor_authorization,
            latency_execution_plan_record=latency_execution_plan_record,
            remote_ready_authorization=remote_ready_authorization,
            remote_consumer_authorizations=remote_consumer_authorizations,
            compact_artifact_resolver=compact_artifact_resolver,
        )
        if (
            admission.get("admitted") is not True
            or admission.get("attempt_id") != attempt_id
            or admission.get("next_wave_index") != wave_index
            or admission.get("next_phase") != phase
        ):
            raise ValueError("live P90 admission does not authorize this exact attempt")

    def validate_batch(
        current_ledger: DatabricksClusterHourLedger,
        reservations: tuple[DatabricksClusterHourReservation, ...],
        snapshots: tuple[Mapping[str, Any], ...],
    ) -> None:
        if databricks_ledger_path_sha256(live_path) != path_sha256:
            raise ValueError("full-score ledger path binding drift under lock")
        require_databricks_ledger_prefix(current_ledger, predecessor_prefix)
        if databricks_ledger_prefix(current_ledger) != predecessor_prefix:
            raise ValueError("full-score batch predecessor prefix is stale or forked")
        if len(reservations) != 1 or len(snapshots) != 1:
            raise ValueError("full-score phase requires one atomic reservation member")
        reservation = reservations[0]
        snapshot = snapshots[0]
        if (
            reservation.attempt_id != attempt_id
            or reservation.workload_id != phase_workload_id
        ):
            raise ValueError("full-score reservation attempt/workload binding drift")
        snapshot_bindings = _validated_full_score_phase_submit_payload(
            execution_plan,
            snapshot,
            inventory=inventory,
            shard_plan=shard_plan,
            wave_index=wave_index,
            phase=phase,
            require_governed_consumer_ready_phase=True,
            remote_ready_authorization=remote_ready_authorization,
            compact_artifact_resolver=compact_artifact_resolver,
        )
        _require_full_score_phase_launch_authorization(
            snapshot_bindings,
            qualification_launch_authorization,
        )
        require_databricks_run_idempotency_token(
            snapshot,
            attempt_id=attempt_id,
        )
        if current_ledger.cap_cluster_hours != (MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS):
            raise ValueError("full-score publication requires the 1024-hour ledger")
        if current_ledger.ledger_id != predecessor_prefix.ledger_id:
            raise ValueError("full-score campaign ledger identity drift")
        if (
            current_ledger.active_reserved_cluster_hours != 0
            or current_ledger.active_reserved_task_count != 0
        ):
            raise ValueError("full-score phases require an active-zero ledger")
        proposed_tasks = len(reservation.task_timeout_seconds)
        if proposed_tasks > FULL_SCORE_MAX_WORKERS:
            raise ValueError("full-score phase exceeds the global 16-task cap")
        if reservation.reserved_cluster_hours > (
            MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
        ):
            raise ValueError("full-score phase exceeds the 900-hour active cap")
        if (
            current_ledger.accounted_cluster_hours
            + reservation.reserved_cluster_hours
            + PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
            > MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
        ):
            raise ValueError(
                "full-score reservation would consume the 124 GPU-hour "
                "campaign headroom"
            )
        if admission is not None:
            ledger_binding = _required_mapping(admission, "ledger")
            if (
                ledger_binding.get("ledger_path_sha256") != path_sha256
                or ledger_binding.get("predecessor_prefix")
                != predecessor_prefix.to_record()
            ):
                raise ValueError("live P90 admission predecessor binding drift")

    return live_path, predecessor_prefix, validate_batch, admission


def _full_score_phase_lease_path(
    ledger_path: str | Path,
    *,
    execution_plan_sha256: str,
    wave_index: int,
    phase: str,
) -> Path:
    path = _full_score_phase_lease_candidate_path(
        ledger_path,
        execution_plan_sha256=execution_plan_sha256,
        wave_index=wave_index,
        phase=phase,
    )
    root = path.parent
    try:
        root.mkdir(mode=0o700)
        _fsync_directory(root.parent)
    except FileExistsError:
        pass
    if not root.is_dir() or root.is_symlink():
        raise ValueError("full-score phase lease root must be a real directory")
    return path


def _full_score_phase_lease_candidate_path(
    ledger_path: str | Path,
    *,
    execution_plan_sha256: str,
    wave_index: int,
    phase: str,
) -> Path:
    """Return the deterministic lease path without creating its directory."""

    ledger = Path(ledger_path).expanduser().absolute()
    root = ledger.with_name(f"{ledger.name}.full-score-phase-leases")
    if root.is_symlink():
        raise ValueError("full-score phase lease root must not be a symlink")
    identity = _canonical_sha256(
        {
            "domain": "cachet.full_score_phase_lease_path.v1",
            "execution_plan_sha256": execution_plan_sha256,
            "phase": phase,
            "wave_index": wave_index,
        }
    )
    return root / f"{identity}.json"


def _full_score_phase_lease_record(
    authorization: FullScorePhaseSubmissionAuthorization,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt_id": authorization.attempt_id,
        "batch_prefix": authorization.batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "execution_plan_sha256": authorization.execution_plan_sha256,
        "intent_record_sha256": authorization.intent_record_sha256,
        "ledger_path_sha256": authorization.ledger_path_sha256,
        "phase": authorization.phase,
        "predecessor_prefix": authorization.predecessor_prefix.to_record(),
        "record_type": "cachet.full_score_phase_lease.v1",
        "submit_payload_sha256": authorization.submit_payload_sha256,
        "wave_index": authorization.wave_index,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _full_score_workspace_authority_record(
    authorization: FullScorePhaseSubmissionAuthorization,
) -> dict[str, Any]:
    lease = _full_score_phase_lease_record(authorization)
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "intent_record_sha256": authorization.intent_record_sha256,
        "phase_lease_record_sha256": lease["closed_record_sha256"],
        "record_type": "cachet.full_score_phase_workspace_authority.v1",
        "schema_version": 1,
        "user_name_sha256": authorization.user_name_sha256,
        "workspace_host_sha256": authorization.workspace_host_sha256,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _full_score_workspace_authority_path(lease_path: Path) -> Path:
    return lease_path.with_name(lease_path.name + ".workspace-authority.json")


def _full_score_phase_intent_path(
    ledger_path: str | Path,
    *,
    execution_plan_sha256: str,
    wave_index: int,
    phase: str,
) -> Path:
    path = _full_score_phase_intent_candidate_path(
        ledger_path,
        execution_plan_sha256=execution_plan_sha256,
        wave_index=wave_index,
        phase=phase,
    )
    root = path.parent
    try:
        root.mkdir(mode=0o700)
        _fsync_directory(root.parent)
    except FileExistsError:
        pass
    if not root.is_dir() or root.is_symlink():
        raise ValueError("full-score phase intent root must be a real directory")
    return path


def _full_score_phase_intent_candidate_path(
    ledger_path: str | Path,
    *,
    execution_plan_sha256: str,
    wave_index: int,
    phase: str,
) -> Path:
    """Return the deterministic intent path without creating its directory."""

    ledger = Path(ledger_path).expanduser().absolute()
    root = ledger.with_name(f"{ledger.name}.full-score-phase-intents")
    if root.is_symlink():
        raise ValueError("full-score phase intent root must not be a symlink")
    identity = _canonical_sha256(
        {
            "domain": "cachet.full_score_phase_intent_path.v1",
            "execution_plan_sha256": execution_plan_sha256,
            "phase": phase,
            "wave_index": wave_index,
        }
    )
    return root / f"{identity}.json"


def _full_score_phase_intent_record(
    *,
    execution_plan_sha256: str,
    wave_index: int,
    phase: str,
    ledger_path: str | Path,
    predecessor_prefix: DatabricksLedgerPrefix,
    attempt_id: str,
    workload_id: str,
    submit_payload_sha256: str,
    budget_admission_path: str | Path | None,
    budget_admission: Mapping[str, Any] | None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> dict[str, Any]:
    if (budget_admission_path is None) != (budget_admission is None):
        raise ValueError("full-score phase intent P90 binding is incomplete")
    admission_binding: dict[str, Any] | None = None
    if budget_admission_path is not None:
        admission_file = _governed_compact_file(
            budget_admission_path,
            "live P90 admission",
            compact_artifact_resolver,
        )
        admission_bytes = admission_file.read_bytes()
        if _json_object(admission_bytes, "live P90 admission") != budget_admission:
            raise ValueError("live P90 admission changed before phase intent")
        assert budget_admission is not None
        admission_binding = {
            "file_sha256": sha256(admission_bytes).hexdigest(),
            "path_sha256": _canonical_sha256(
                {
                    "domain": "cachet.full_score_phase_p90_path.v1",
                    "path": str(budget_admission_path),
                }
            ),
            "record_sha256": _required_string(
                budget_admission,
                "closed_record_sha256",
            ),
        }
    record: dict[str, Any] = {
        "attempt_id": attempt_id,
        "budget_admission": admission_binding,
        "closed_record_sha256": "",
        "execution_plan_sha256": execution_plan_sha256,
        "ledger_path_sha256": databricks_ledger_path_sha256(ledger_path),
        "phase": phase,
        "predecessor_prefix": predecessor_prefix.to_record(),
        "record_type": "cachet.full_score_phase_intent.v1",
        "submit_payload_sha256": submit_payload_sha256,
        "wave_index": wave_index,
        "workload_id": workload_id,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _write_or_require_full_score_phase_intent(
    ledger_path: str | Path,
    record: Mapping[str, Any],
) -> Path:
    path = _full_score_phase_intent_path(
        ledger_path,
        execution_plan_sha256=_required_string(
            record,
            "execution_plan_sha256",
        ),
        wave_index=_required_int(record, "wave_index"),
        phase=_required_string(record, "phase"),
    )
    content = _canonical_pretty_json_bytes(record)
    try:
        _exclusive_write_bytes(path, content)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != content:
            raise ValueError("full-score phase intent binding drift") from None
    return path


def _require_full_score_phase_intent(
    ledger_path: str | Path,
    expected: Mapping[str, Any],
) -> Path:
    path = _full_score_phase_intent_candidate_path(
        ledger_path,
        execution_plan_sha256=_required_string(
            expected,
            "execution_plan_sha256",
        ),
        wave_index=_required_int(expected, "wave_index"),
        phase=_required_string(expected, "phase"),
    )
    content = _canonical_pretty_json_bytes(expected)
    if path.is_symlink() or not path.is_file():
        raise ValueError("full-score phase replay requires pre-reservation intent")
    if path.read_bytes() != content:
        raise ValueError("full-score phase intent binding drift")
    return path


def _write_or_require_full_score_phase_lease(
    ledger_path: str | Path,
    authorization: FullScorePhaseSubmissionAuthorization,
) -> Path:
    record = _full_score_phase_lease_record(authorization)
    path = _full_score_phase_lease_path(
        ledger_path,
        execution_plan_sha256=authorization.execution_plan_sha256,
        wave_index=authorization.wave_index,
        phase=authorization.phase,
    )
    content = _canonical_pretty_json_bytes(record)
    try:
        _exclusive_write_bytes(path, content)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != content:
            raise ValueError("full-score phase lease binding drift") from None
    sidecar = _full_score_workspace_authority_record(authorization)
    sidecar_path = _full_score_workspace_authority_path(path)
    sidecar_content = _canonical_pretty_json_bytes(sidecar)
    _write_or_require_local_authority_bytes(
        sidecar_path,
        sidecar_content,
        label="full-score workspace authority",
    )
    return path


def _require_full_score_phase_lease(
    ledger_path: str | Path,
    authorization: FullScorePhaseSubmissionAuthorization,
) -> Path:
    intent_path = _full_score_phase_intent_candidate_path(
        ledger_path,
        execution_plan_sha256=authorization.execution_plan_sha256,
        wave_index=authorization.wave_index,
        phase=authorization.phase,
    )
    if intent_path.is_symlink() or not intent_path.is_file():
        raise ValueError("full-score phase authority requires its durable intent")
    intent_bytes = intent_path.read_bytes()
    intent = _json_object(intent_bytes, "full-score phase intent")
    if (
        intent.get("record_type") != "cachet.full_score_phase_intent.v1"
        or intent.get("closed_record_sha256") != _closed_record_sha256(intent)
        or intent.get("closed_record_sha256") != authorization.intent_record_sha256
        or intent.get("execution_plan_sha256") != authorization.execution_plan_sha256
        or intent.get("wave_index") != authorization.wave_index
        or intent.get("phase") != authorization.phase
        or intent.get("ledger_path_sha256") != authorization.ledger_path_sha256
        or intent.get("predecessor_prefix")
        != authorization.predecessor_prefix.to_record()
        or intent.get("attempt_id") != authorization.attempt_id
        or intent.get("submit_payload_sha256") != authorization.submit_payload_sha256
        or intent_bytes != _canonical_pretty_json_bytes(intent)
    ):
        raise ValueError("full-score phase intent authority binding drift")
    record = _full_score_phase_lease_record(authorization)
    path = _full_score_phase_lease_candidate_path(
        ledger_path,
        execution_plan_sha256=authorization.execution_plan_sha256,
        wave_index=authorization.wave_index,
        phase=authorization.phase,
    )
    if path.is_symlink() or not path.is_file():
        raise ValueError("full-score phase replay requires the post-batch lease")
    if path.read_bytes() != _canonical_pretty_json_bytes(record):
        raise ValueError("full-score phase lease binding drift")
    sidecar = _full_score_workspace_authority_record(authorization)
    sidecar_path = _full_score_workspace_authority_path(path)
    if (
        _read_full_score_workspace_authority(
            sidecar_path,
            label="full-score workspace authority",
        )
        != sidecar
    ):
        raise ValueError("full-score workspace authority sidecar drift")
    return path


def _require_full_score_historical_reservation_policy(
    ledger: DatabricksClusterHourLedger,
    *,
    ledger_path: str | Path,
    predecessor_prefix: DatabricksLedgerPrefix,
    reservation: DatabricksClusterHourReservation,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    admission: Mapping[str, Any] | None,
) -> DatabricksClusterHourLedger:
    """Replay the under-lock launch policy at its historical predecessor."""

    path_sha256 = databricks_ledger_path_sha256(ledger_path)
    if getattr(qualification_launch_authorization, "ledger_id", None) != (
        predecessor_prefix.ledger_id
    ):
        raise ValueError(
            "full-score ledger differs from the campaign qualification ledger"
        )
    if (
        getattr(
            qualification_launch_authorization,
            "ledger_path_sha256",
            path_sha256,
        )
        != path_sha256
    ):
        raise ValueError("full-score qualification ledger path binding drift")
    predecessor_state = _full_score_ledger_prefix_state(
        ledger,
        predecessor_prefix,
    )
    if predecessor_state.cap_cluster_hours != (MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS):
        raise ValueError("full-score publication requires the 1024-hour ledger")
    if predecessor_state.ledger_id != predecessor_prefix.ledger_id:
        raise ValueError("full-score campaign ledger identity drift")
    if (
        predecessor_state.active_reserved_cluster_hours != 0
        or predecessor_state.active_reserved_task_count != 0
    ):
        raise ValueError("full-score phases require an active-zero predecessor")
    if len(reservation.task_timeout_seconds) > FULL_SCORE_MAX_WORKERS:
        raise ValueError("full-score phase exceeds the global 16-task cap")
    if reservation.reserved_cluster_hours > (
        MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
    ):
        raise ValueError("full-score phase exceeds the 900-hour active cap")
    if (
        predecessor_state.accounted_cluster_hours
        + reservation.reserved_cluster_hours
        + PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
        > MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
    ):
        raise ValueError(
            "full-score reservation would consume the 124 GPU-hour campaign headroom"
        )
    if admission is not None:
        ledger_binding = _required_mapping(admission, "ledger")
        if (
            ledger_binding.get("ledger_path_sha256") != path_sha256
            or ledger_binding.get("predecessor_prefix")
            != predecessor_prefix.to_record()
        ):
            raise ValueError("live P90 admission predecessor binding drift")
    return predecessor_state


def reserve_governed_full_score_phase_attempt(
    ledger_path: str | Path,
    submit_payload: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    wave_index: int,
    phase: str,
    attempt_id: str,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    predecessor_authorization: object,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
    budget_admission_path: str | Path | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> tuple[DatabricksClusterHourLedger, FullScorePhaseSubmissionAuthorization]:
    """Durably reserve an exact phase without performing a cloud submission.

    Production launchers should use
    :func:`reserve_and_submit_governed_full_score_phase_attempt` so the same
    immutable payload snapshot is reserved, posted, and receipt-bound as one
    coupled action.  This lower-level entrypoint exists for offline admission
    verification and recovery tooling.
    """

    live_path, predecessor_prefix, validate_batch, admission = (
        _governed_full_score_phase_reservation_validator(
            ledger_path,
            submit_payload,
            execution_plan=execution_plan,
            inventory=inventory,
            shard_plan=shard_plan,
            wave_index=wave_index,
            phase=phase,
            attempt_id=attempt_id,
            qualification_launch_authorization=(qualification_launch_authorization),
            predecessor_authorization=predecessor_authorization,
            latency_execution_plan_record=latency_execution_plan_record,
            budget_admission_path=budget_admission_path,
            remote_ready_authorization=remote_ready_authorization,
            remote_consumer_authorizations=remote_consumer_authorizations,
            compact_artifact_resolver=compact_artifact_resolver,
        )
    )
    request = DatabricksRunAttemptReservationRequest(
        attempt_id=attempt_id,
        workload_id=_full_score_phase_workload_id(
            execution_plan, wave_index=wave_index, phase=phase
        ),
        submit_payload=submit_payload,
    )
    _snapshot, canonical_submit = canonical_databricks_submit_payload_snapshot(
        submit_payload
    )
    submit_payload_sha256 = sha256(canonical_submit).hexdigest()
    intent = _full_score_phase_intent_record(
        execution_plan_sha256=_required_string(
            execution_plan,
            "closed_record_sha256",
        ),
        wave_index=wave_index,
        phase=phase,
        ledger_path=live_path,
        predecessor_prefix=predecessor_prefix,
        attempt_id=attempt_id,
        workload_id=request.workload_id,
        submit_payload_sha256=submit_payload_sha256,
        budget_admission_path=budget_admission_path,
        budget_admission=admission,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    _write_or_require_full_score_phase_intent(live_path, intent)
    updated, batch_authorization = reserve_databricks_run_attempt_batch_authorized_json(
        live_path,
        (request,),
        expected_predecessor_prefix=predecessor_prefix,
        batch_validator=validate_batch,
    )
    _require_full_score_ledger_caps(updated)
    if not any(item.attempt_id == attempt_id for item in updated.reservations):
        raise RuntimeError("full-score reservation was not durably recorded")
    authorization = FullScorePhaseSubmissionAuthorization(
        execution_plan_sha256=_required_string(execution_plan, "closed_record_sha256"),
        wave_index=wave_index,
        phase=phase,
        ledger_path_sha256=databricks_ledger_path_sha256(live_path),
        predecessor_prefix=predecessor_prefix,
        batch_authorization=batch_authorization,
        attempt_id=attempt_id,
        submit_payload_sha256=submit_payload_sha256,
        intent_record_sha256=_required_string(
            intent,
            "closed_record_sha256",
        ),
        workspace_host_sha256=_full_score_predecessor_workspace_identity(
            predecessor_authorization
        )[0],
        user_name_sha256=_full_score_predecessor_workspace_identity(
            predecessor_authorization
        )[1],
        _issuer=_FULL_SCORE_PHASE_SUBMISSION_AUTHORIZATION_ISSUER,
    )
    _write_or_require_full_score_phase_lease(live_path, authorization)
    return updated, authorization


def submit_governed_full_score_phase_attempt(
    workspace: DatabricksWorkspaceConfig,
    submit_payload: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    submission_authorization: FullScorePhaseSubmissionAuthorization,
    opener: DatabricksURLOpener | None = None,
) -> dict[str, Any]:
    """POST one exact atomically reserved phase through its O_EXCL claim."""

    if type(submission_authorization) is not FullScorePhaseSubmissionAuthorization:
        raise TypeError(
            "full-score submit requires FullScorePhaseSubmissionAuthorization"
        )
    path_sha256 = databricks_ledger_path_sha256(ledger_path)
    if path_sha256 != submission_authorization.ledger_path_sha256:
        raise ValueError("full-score submission ledger path binding drift")
    _require_full_score_phase_lease(
        ledger_path,
        submission_authorization,
    )
    _snapshot, canonical_submit = canonical_databricks_submit_payload_snapshot(
        submit_payload
    )
    payload_sha256 = sha256(canonical_submit).hexdigest()
    require_databricks_batch_reservation_authorization(
        submission_authorization.batch_authorization,
        expected_predecessor_prefix=(submission_authorization.predecessor_prefix),
        expected_attempt_ids=(submission_authorization.attempt_id,),
        expected_submit_payload_sha256s=(payload_sha256,),
    )
    if payload_sha256 != submission_authorization.submit_payload_sha256:
        raise ValueError("full-score submission payload binding drift")
    _require_full_score_bound_workspace_identity(
        workspace,
        authorization=submission_authorization,
        submit_payload=submit_payload,
        opener=opener,
    )
    _require_full_score_phase_lease(
        ledger_path,
        submission_authorization,
    )
    response: dict[str, Any] = submit_pre_reserved_databricks_run(
        workspace,
        submit_payload,
        ledger_path=ledger_path,
        attempt_id=submission_authorization.attempt_id,
        batch_authorization=submission_authorization.batch_authorization,
        opener=opener,
    )
    return response


def _replay_governed_full_score_phase_submission_authorization(
    ledger_path: str | Path,
    submit_payload: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    wave_index: int,
    phase: str,
    attempt_id: str,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    predecessor_authorization: object,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
    budget_admission_path: str | Path | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
    _finalize_missing_lease: bool | None,
) -> FullScorePhaseSubmissionAuthorization:
    """Reissue one historical atomic phase authority after controller restart.

    Lease mode ``None`` is a read-only preflight used before live identity
    authentication; ``True`` finalizes a missing lease and ``False`` requires
    the exact lease to exist.
    """

    _validate_publication_full_score_inputs(inventory, shard_plan, execution_plan)
    require_databricks_run_idempotency_token(
        submit_payload,
        attempt_id=attempt_id,
    )
    bindings = _validated_full_score_phase_submit_payload(
        execution_plan,
        submit_payload,
        inventory=inventory,
        shard_plan=shard_plan,
        wave_index=wave_index,
        phase=phase,
        require_governed_consumer_ready_phase=True,
        remote_ready_authorization=remote_ready_authorization,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    _require_full_score_phase_launch_authorization(
        bindings,
        qualification_launch_authorization,
    )
    live_path = Path(ledger_path).expanduser().absolute()
    predecessor_prefix, _lineage = _require_full_score_phase_predecessor_authorization(
        predecessor_authorization,
        execution_plan=execution_plan,
        ledger_path=live_path,
        wave_index=wave_index,
        phase=phase,
        latency_execution_plan_record=latency_execution_plan_record,
        require_latest=False,
    )
    _require_full_score_remote_workspace_lineage(
        predecessor_authorization,
        execution_plan=execution_plan,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
    )
    admission: dict[str, Any] | None
    if wave_index == 0:
        if budget_admission_path is not None:
            raise ValueError("wave zero replay must not consume a live P90 admission")
        admission = None
    else:
        if budget_admission_path is None:
            raise ValueError("nonzero replay requires its governed live P90 admission")
        admission = load_governed_full_score_live_p90_budget_admission(
            budget_admission_path,
            execution_plan=execution_plan,
            inventory=inventory,
            shard_plan=shard_plan,
            next_submit_payload=submit_payload,
            ledger_path=live_path,
            predecessor_authorization=predecessor_authorization,
            latency_execution_plan_record=latency_execution_plan_record,
            remote_ready_authorization=remote_ready_authorization,
            remote_consumer_authorizations=remote_consumer_authorizations,
            compact_artifact_resolver=compact_artifact_resolver,
            _require_latest_predecessor=False,
        )
        if (
            admission.get("admitted") is not True
            or admission.get("attempt_id") != attempt_id
            or admission.get("next_wave_index") != wave_index
            or admission.get("next_phase") != phase
        ):
            raise ValueError("live P90 admission does not authorize this replay")
    request = DatabricksRunAttemptReservationRequest(
        attempt_id=attempt_id,
        workload_id=_full_score_phase_workload_id(
            execution_plan,
            wave_index=wave_index,
            phase=phase,
        ),
        submit_payload=submit_payload,
    )
    reservation = databricks_submit_payload_reservation(
        submit_payload,
        attempt_id=attempt_id,
        workload_id=request.workload_id,
    )
    _snapshot, canonical_submit = canonical_databricks_submit_payload_snapshot(
        submit_payload
    )
    submit_payload_sha256 = sha256(canonical_submit).hexdigest()
    intent = _full_score_phase_intent_record(
        execution_plan_sha256=_required_string(
            execution_plan,
            "closed_record_sha256",
        ),
        wave_index=wave_index,
        phase=phase,
        ledger_path=live_path,
        predecessor_prefix=predecessor_prefix,
        attempt_id=attempt_id,
        workload_id=request.workload_id,
        submit_payload_sha256=submit_payload_sha256,
        budget_admission_path=budget_admission_path,
        budget_admission=admission,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    _require_full_score_phase_intent(live_path, intent)
    batch_authorization = replay_databricks_run_attempt_batch_authorization_json(
        live_path,
        (request,),
        expected_predecessor_prefix=predecessor_prefix,
    )
    live = read_databricks_cluster_hour_ledger_json(live_path)
    _require_full_score_historical_reservation_policy(
        live,
        ledger_path=live_path,
        predecessor_prefix=predecessor_prefix,
        reservation=reservation,
        qualification_launch_authorization=qualification_launch_authorization,
        admission=admission,
    )
    batch_prefix = batch_authorization.batch_prefix
    if len(live.reservations) != batch_prefix.reservation_count:
        raise ValueError("full-score replay has reservations after its phase batch")
    receipt_delta = len(live.submission_receipts) - (
        batch_prefix.submission_receipt_count
    )
    terminal_delta = len(live.terminal_actuals) - batch_prefix.terminal_actual_count
    if receipt_delta not in {0, 1} or terminal_delta not in {0, 1}:
        raise ValueError("full-score replay contains intervening phase events")
    if terminal_delta > receipt_delta:
        raise ValueError("full-score replay terminal precedes its submission receipt")
    if receipt_delta == 1:
        receipt = live.submission_receipts[-1]
        if (
            receipt.attempt_id != attempt_id
            or receipt.submit_payload_sha256 != submit_payload_sha256
        ):
            raise ValueError("full-score replay receipt tail binding drift")
    if terminal_delta == 1:
        terminal = live.terminal_actuals[-1]
        if (
            terminal.attempt_id != attempt_id
            or terminal.submit_payload_sha256 != submit_payload_sha256
        ):
            raise ValueError("full-score replay terminal tail binding drift")
    authorization = FullScorePhaseSubmissionAuthorization(
        execution_plan_sha256=_required_string(execution_plan, "closed_record_sha256"),
        wave_index=wave_index,
        phase=phase,
        ledger_path_sha256=databricks_ledger_path_sha256(live_path),
        predecessor_prefix=predecessor_prefix,
        batch_authorization=batch_authorization,
        attempt_id=attempt_id,
        submit_payload_sha256=submit_payload_sha256,
        intent_record_sha256=_required_string(
            intent,
            "closed_record_sha256",
        ),
        workspace_host_sha256=_full_score_predecessor_workspace_identity(
            predecessor_authorization
        )[0],
        user_name_sha256=_full_score_predecessor_workspace_identity(
            predecessor_authorization
        )[1],
        _issuer=_FULL_SCORE_PHASE_SUBMISSION_AUTHORIZATION_ISSUER,
    )
    if _finalize_missing_lease is True:
        _write_or_require_full_score_phase_lease(live_path, authorization)
    elif _finalize_missing_lease is False:
        _require_full_score_phase_lease(live_path, authorization)
    elif _finalize_missing_lease is None:
        lease_path = _full_score_phase_lease_candidate_path(
            live_path,
            execution_plan_sha256=authorization.execution_plan_sha256,
            wave_index=authorization.wave_index,
            phase=authorization.phase,
        )
        lease_root = lease_path.parent
        if lease_root.exists() and (not lease_root.is_dir() or lease_root.is_symlink()):
            raise ValueError("full-score phase lease root must be a real directory")
        if lease_path.exists() or lease_path.is_symlink():
            expected_lease = _canonical_pretty_json_bytes(
                _full_score_phase_lease_record(authorization)
            )
            if (
                lease_path.is_symlink()
                or not lease_path.is_file()
                or lease_path.read_bytes() != expected_lease
            ):
                raise ValueError("full-score phase lease binding drift")
    else:
        raise TypeError("full-score replay lease mode is invalid")
    return authorization


def replay_governed_full_score_phase_submission_authorization(
    ledger_path: str | Path,
    submit_payload: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    wave_index: int,
    phase: str,
    attempt_id: str,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    predecessor_authorization: object,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
    budget_admission_path: str | Path | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> FullScorePhaseSubmissionAuthorization:
    """Strictly reissue an authority only when both durable markers exist."""

    return _replay_governed_full_score_phase_submission_authorization(
        ledger_path,
        submit_payload,
        execution_plan=execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        wave_index=wave_index,
        phase=phase,
        attempt_id=attempt_id,
        qualification_launch_authorization=qualification_launch_authorization,
        predecessor_authorization=predecessor_authorization,
        latency_execution_plan_record=latency_execution_plan_record,
        budget_admission_path=budget_admission_path,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_artifact_resolver,
        _finalize_missing_lease=False,
    )


def recover_governed_full_score_phase_reservation(
    ledger_path: str | Path,
    submit_payload: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    wave_index: int,
    phase: str,
    attempt_id: str,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    predecessor_authorization: object,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
    budget_admission_path: str | Path | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> FullScorePhaseSubmissionAuthorization:
    """Close a post-reservation crash only from its preexisting exact intent."""

    return _replay_governed_full_score_phase_submission_authorization(
        ledger_path,
        submit_payload,
        execution_plan=execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        wave_index=wave_index,
        phase=phase,
        attempt_id=attempt_id,
        qualification_launch_authorization=qualification_launch_authorization,
        predecessor_authorization=predecessor_authorization,
        latency_execution_plan_record=latency_execution_plan_record,
        budget_admission_path=budget_admission_path,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_artifact_resolver,
        _finalize_missing_lease=True,
    )


def resume_governed_full_score_phase_attempt(
    workspace: DatabricksWorkspaceConfig,
    submit_payload: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    wave_index: int,
    phase: str,
    attempt_id: str,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    predecessor_authorization: object,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
    budget_admission_path: str | Path | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
    opener: DatabricksURLOpener | None = None,
) -> tuple[dict[str, Any], FullScorePhaseSubmissionAuthorization]:
    """Resume a reserved phase without risking a second physical run."""

    authorization = _replay_governed_full_score_phase_submission_authorization(
        ledger_path,
        submit_payload,
        execution_plan=execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        wave_index=wave_index,
        phase=phase,
        attempt_id=attempt_id,
        qualification_launch_authorization=qualification_launch_authorization,
        predecessor_authorization=predecessor_authorization,
        latency_execution_plan_record=latency_execution_plan_record,
        budget_admission_path=budget_admission_path,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_artifact_resolver,
        _finalize_missing_lease=None,
    )
    _require_full_score_bound_workspace_identity(
        workspace,
        authorization=authorization,
        submit_payload=submit_payload,
        opener=opener,
    )
    recovered_authorization = recover_governed_full_score_phase_reservation(
        ledger_path,
        submit_payload,
        execution_plan=execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        wave_index=wave_index,
        phase=phase,
        attempt_id=attempt_id,
        qualification_launch_authorization=qualification_launch_authorization,
        predecessor_authorization=predecessor_authorization,
        latency_execution_plan_record=latency_execution_plan_record,
        budget_admission_path=budget_admission_path,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    if recovered_authorization != authorization:
        raise RuntimeError("full-score resume authority changed across live auth")
    authorization = recovered_authorization
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    receipt = next(
        (item for item in ledger.submission_receipts if item.attempt_id == attempt_id),
        None,
    )
    if receipt is not None:
        return {"run_id": receipt.run_id}, authorization
    _require_full_score_phase_lease(ledger_path, authorization)
    response = resume_pre_reserved_databricks_run(
        workspace,
        submit_payload,
        ledger_path=ledger_path,
        attempt_id=attempt_id,
        batch_authorization=authorization.batch_authorization,
        opener=opener,
    )
    del response
    reconciled = read_databricks_cluster_hour_ledger_json(ledger_path)
    receipt = next(
        (
            item
            for item in reconciled.submission_receipts
            if item.attempt_id == attempt_id
        ),
        None,
    )
    if receipt is None:
        raise RuntimeError("resumed phase POST has no durable submission receipt")
    return {"run_id": receipt.run_id}, authorization


def recover_governed_full_score_phase_attempt(
    workspace: DatabricksWorkspaceConfig,
    submit_payload: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    submission_authorization: FullScorePhaseSubmissionAuthorization,
    opener: DatabricksURLOpener | None = None,
) -> dict[str, Any]:
    """Recover an accepted/lost phase POST with its exact token and claim."""

    if type(submission_authorization) is not FullScorePhaseSubmissionAuthorization:
        raise TypeError(
            "full-score recovery requires FullScorePhaseSubmissionAuthorization"
        )
    if databricks_ledger_path_sha256(ledger_path) != (
        submission_authorization.ledger_path_sha256
    ):
        raise ValueError("full-score recovery ledger path binding drift")
    _require_full_score_phase_lease(
        ledger_path,
        submission_authorization,
    )
    _snapshot, canonical_submit = canonical_databricks_submit_payload_snapshot(
        submit_payload
    )
    payload_sha256 = sha256(canonical_submit).hexdigest()
    if payload_sha256 != submission_authorization.submit_payload_sha256:
        raise ValueError("full-score recovery payload binding drift")
    require_databricks_run_idempotency_token(
        submit_payload,
        attempt_id=submission_authorization.attempt_id,
    )
    _require_full_score_bound_workspace_identity(
        workspace,
        authorization=submission_authorization,
        submit_payload=submit_payload,
        opener=opener,
    )
    _require_full_score_phase_lease(
        ledger_path,
        submission_authorization,
    )
    response: dict[str, Any] = recover_pre_reserved_databricks_run(
        workspace,
        submit_payload,
        ledger_path=ledger_path,
        attempt_id=submission_authorization.attempt_id,
        batch_authorization=submission_authorization.batch_authorization,
        opener=opener,
    )
    return response


def reserve_and_submit_governed_full_score_phase_attempt(
    workspace: DatabricksWorkspaceConfig,
    submit_payload: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    wave_index: int,
    phase: str,
    attempt_id: str,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    predecessor_authorization: object,
    latency_execution_plan_record: Mapping[str, Any] | None = None,
    budget_admission_path: str | Path | None = None,
    remote_ready_authorization: object | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
    opener: DatabricksURLOpener | None = None,
) -> tuple[dict[str, Any], FullScorePhaseSubmissionAuthorization]:
    """Reserve, submit exact bytes, and persist the run receipt in one action."""

    _governed_full_score_phase_reservation_validator(
        ledger_path,
        submit_payload,
        execution_plan=execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        wave_index=wave_index,
        phase=phase,
        attempt_id=attempt_id,
        qualification_launch_authorization=qualification_launch_authorization,
        predecessor_authorization=predecessor_authorization,
        latency_execution_plan_record=latency_execution_plan_record,
        budget_admission_path=budget_admission_path,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    predecessor_workspace_host_sha256, predecessor_user_name_sha256 = (
        _full_score_predecessor_workspace_identity(predecessor_authorization)
    )
    identity = require_databricks_current_user_name(
        workspace,
        expected_user_name=_full_score_phase_single_user_name(submit_payload),
        opener=opener,
    )
    if (
        identity.get("workspace_host_sha256") != predecessor_workspace_host_sha256
        or identity.get("user_name_sha256") != predecessor_user_name_sha256
    ):
        raise ValueError("full-score predecessor workspace/principal drift")
    _updated, submission_authorization = reserve_governed_full_score_phase_attempt(
        ledger_path,
        submit_payload,
        execution_plan=execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        wave_index=wave_index,
        phase=phase,
        attempt_id=attempt_id,
        qualification_launch_authorization=(qualification_launch_authorization),
        predecessor_authorization=predecessor_authorization,
        latency_execution_plan_record=latency_execution_plan_record,
        budget_admission_path=budget_admission_path,
        remote_ready_authorization=remote_ready_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    response = submit_governed_full_score_phase_attempt(
        workspace,
        submit_payload,
        ledger_path=ledger_path,
        submission_authorization=submission_authorization,
        opener=opener,
    )
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    reservations = [
        item for item in ledger.reservations if item.attempt_id == attempt_id
    ]
    receipts = [
        item for item in ledger.submission_receipts if item.attempt_id == attempt_id
    ]
    if len(reservations) != 1 or len(receipts) != 1:
        raise RuntimeError(
            "full-score submit did not durably bind one reservation and receipt"
        )
    if receipts[0].submit_payload_sha256 != reservations[0].submit_payload_sha256:
        raise RuntimeError("full-score submit receipt payload binding drift")
    return response, submission_authorization


def validate_full_score_live_p90_budget_admission(
    execution_plan: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    """Replay every deterministic draw and reject altered budget evidence."""

    if not isinstance(record, Mapping):
        raise TypeError("live P90 budget admission must be an object")
    if record.get("record_type") != FULL_SCORE_LIVE_P90_RECORD_TYPE:
        raise ValueError("live P90 budget admission record_type drift")
    if record.get("schema_version") != FULL_SCORE_LIVE_P90_SCHEMA_VERSION:
        raise ValueError("live P90 budget admission schema drift")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("live P90 budget admission closure drift")
    scope = record.get("authorization_scope")
    if scope not in {
        FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
        FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE,
    }:
        raise ValueError("live P90 budget admission authorization scope drift")
    replay_record = dict(record)
    if scope == FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE:
        for field_name in (
            "attempt_id",
            "ledger",
            "matched_block_files",
            "next_submit_payload_sha256",
            "predecessor_lineage",
            "remote_consumer_authorizations",
        ):
            replay_record.pop(field_name, None)
        replay_record["authorization_scope"] = (
            FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE
        )
        replay_record["closed_record_sha256"] = _closed_record_sha256(replay_record)
    before = _required_mapping(replay_record, "ledger_before")
    rebuilt = build_full_score_live_p90_budget_admission(
        execution_plan,
        cast(Sequence[Mapping[str, Any]], replay_record.get("completed_blocks")),
        next_wave_index=_required_int(replay_record, "next_wave_index"),
        ledger_terminal_actual_gpu_hours=_required_number(
            before,
            "terminal_actual_gpu_hours",
        ),
        ledger_active_reserved_gpu_hours=_required_number(
            before,
            "active_reserved_gpu_hours",
        ),
        next_wave_reserved_gpu_hours=_required_number(
            before,
            "next_wave_reserved_gpu_hours",
        ),
        next_phase=_required_string(replay_record, "next_phase"),
    )
    if replay_record != rebuilt:
        raise ValueError("live P90 budget admission does not replay exactly")


def validate_paired_full_score_outputs(
    baseline_record: Mapping[str, Any],
    vanilla_record: Mapping[str, Any],
    *,
    shard: Mapping[str, Any],
    examples: Mapping[tuple[str, str], Any],
    vanilla_examples: Mapping[tuple[str, str], Any],
) -> tuple[dict[str, Any], ...]:
    """Re-score raw outputs and require exact one-to-one paired coverage."""

    baseline = benchmark_run_result_from_record(baseline_record)
    vanilla = benchmark_run_result_from_record(vanilla_record)
    expected_items = _shard_items_by_key(shard)
    shard_id = _required_string(shard, "shard_id")
    baseline_by_key = _validated_method_measurements(
        baseline.measurements,
        method="baseline_prefill",
        shard_id=shard_id,
        expected_items=expected_items,
        examples=examples,
    )
    vanilla_by_key = _validated_method_measurements(
        vanilla.measurements,
        method="vanilla_prefill",
        shard_id=shard_id,
        expected_items=expected_items,
        examples=vanilla_examples,
    )
    _validate_full_score_benchmark_protocol(
        baseline,
        method="baseline_prefill",
        shard=shard,
        expected_items=expected_items,
        protocol_examples=examples,
    )
    _validate_full_score_benchmark_protocol(
        vanilla,
        method="vanilla_prefill",
        shard=shard,
        expected_items=expected_items,
        protocol_examples=vanilla_examples,
    )
    if set(baseline_by_key) != set(vanilla_by_key):
        raise ValueError("baseline and Vanilla output identities are not paired")
    paired = []
    for key in sorted(expected_items):
        baseline_measurement = baseline_by_key[key]
        vanilla_measurement = vanilla_by_key[key]
        item = expected_items[key]
        niah_cell_id = baseline_measurement.metadata.get("niah_cell_id", "")
        if niah_cell_id != vanilla_measurement.metadata.get("niah_cell_id", ""):
            raise ValueError("paired NIAH cell identity drift")
        paired.append(
            {
                "dataset": key[0],
                "example_id": key[1],
                "identity_sha256": item["identity_sha256"],
                "methods": {
                    "baseline_prefill": _measurement_score_record(baseline_measurement),
                    "vanilla_prefill": _measurement_score_record(vanilla_measurement),
                },
                "niah_cell_id": niah_cell_id or None,
                "natural_prompt_sha256": item["natural_prompt_sha256"],
            }
        )
    return tuple(paired)


def _build_expected_full_score_benchmark_manifest(
    *,
    method: str,
    shard_id: str,
    protocol_examples: Mapping[tuple[str, str], Any],
    measurements: Sequence[Any],
) -> Any:
    """Rebuild every raw manifest field from the governed command inputs."""

    if method not in FULL_SCORE_METHODS:
        raise ValueError("unsupported full-score benchmark method")
    expected_arm = (
        baseline_prefill_arm()
        if method == "baseline_prefill"
        else method_benchmark_arm(
            "vanilla_prefill",
            physical_transform_id="cachet.vanilla.per_document_segments",
        )
    )
    expected_suite_id = f"{FULL_SCORE_PROTOCOL_ID}:{shard_id}:{method}"
    expected_datasets = tuple(
        sorted({dataset for dataset, _example_id in protocol_examples})
    )
    suite = BenchmarkSuite(
        suite_id=expected_suite_id,
        examples=tuple(protocol_examples.values()),
        model_id=FULL_SCORE_SERVED_MODEL_NAME,
        hardware_target="aws-g6-l4",
        datasets=expected_datasets,
    )
    context = BenchmarkManifestContext(
        model_revision=MAIN_LATENCY_TOKENIZER_REVISION,
        canonical_model_id=FULL_SCORE_MODEL_ID,
        tokenizer_id=MAIN_LATENCY_TOKENIZER_ID,
        tokenizer_revision=MAIN_LATENCY_TOKENIZER_REVISION,
        engine_id="vllm",
        engine_version=PUBLICATION_CAMPAIGN_ENGINE_VERSION,
        serving_platform="databricks-aws-single-gpu",
        model_dtype=FULL_SCORE_MODEL_DTYPE,
        model_quantization=FULL_SCORE_MODEL_QUANTIZATION,
        runtime_kv_dtype=FULL_SCORE_KV_DTYPE,
        max_output_tokens=FULL_SCORE_MAX_TOKENS,
        temperature=FULL_SCORE_TEMPERATURE,
        stream=True,
        runtime_id=f"{FULL_SCORE_PROTOCOL_ID}:{shard_id}",
        measurement_scopes=("quality",),
    )
    return _build_experiment_manifest(
        suite,
        arms=(expected_arm,),
        measurements=measurements,
        scorer_registry=default_dataset_scorer_registry(),
        context=context,
        request_parallelism=FULL_SCORE_REQUEST_PARALLELISM,
        repeats=FULL_SCORE_PASSES_PER_METHOD,
        warmups=0,
        isolate_arms=True,
        shuffle=False,
        seed=None,
        interleave_examples=False,
        baseline_arm_id=expected_arm.arm_id,
        request_customization_digests={
            expected_arm.arm_id: FULL_SCORE_REQUEST_CUSTOMIZATION_DIGEST
        },
    )


def _validate_full_score_benchmark_protocol(
    result: Any,
    *,
    method: str,
    shard: Mapping[str, Any],
    expected_items: Mapping[tuple[str, str], Mapping[str, Any]],
    protocol_examples: Mapping[tuple[str, str], Any],
) -> None:
    """Bind one reconstructed raw run to the exact frozen score protocol."""

    if method not in FULL_SCORE_METHODS:
        raise ValueError("unsupported full-score benchmark method")
    shard_id = _required_string(shard, "shard_id")
    expected_arm_id = (
        BASELINE_PREFILL_ARM
        if method == "baseline_prefill"
        else FULL_SCORE_VANILLA_ARM_ID
    )
    expected_datasets = tuple(sorted({dataset for dataset, _ in expected_items}))
    expected_suite_id = f"{FULL_SCORE_PROTOCOL_ID}:{shard_id}:{method}"
    if set(protocol_examples) != set(expected_items):
        raise ValueError(f"{method} raw benchmark example coverage drift")
    suite = result.suite
    manifest = result.experiment_manifest
    if manifest is None:
        raise ValueError("full-score raw output lacks an experiment manifest")
    result_contract = {
        "baseline_arm_id": result.baseline_arm_id,
        "evidence_policy": result.evidence_policy,
        "execution_isolation_mode": result.execution_isolation_mode,
        "interleave_examples": result.interleave_examples,
        "isolate_arms": result.isolate_arms,
        "prefix_cache_salt_mode": result.prefix_cache_salt_mode,
        "repeats": result.repeats,
        "request_parallelism": result.request_parallelism,
        "seed": result.seed,
        "shuffle": result.shuffle,
        "warmups": result.warmups,
    }
    expected_result_contract = {
        "baseline_arm_id": expected_arm_id,
        "evidence_policy": "smoke",
        "execution_isolation_mode": "shared_process_sequential",
        "interleave_examples": False,
        "isolate_arms": True,
        "prefix_cache_salt_mode": "per_request",
        "repeats": FULL_SCORE_PASSES_PER_METHOD,
        "request_parallelism": FULL_SCORE_REQUEST_PARALLELISM,
        "seed": None,
        "shuffle": False,
        "warmups": 0,
    }
    if not _json_type_exact_equal(result_contract, expected_result_contract):
        raise ValueError(f"{method} raw benchmark execution protocol drift")
    if (
        suite.suite_id != expected_suite_id
        or suite.model_id != FULL_SCORE_SERVED_MODEL_NAME
        or suite.hardware_target != "aws-g6-l4"
        or tuple(suite.datasets) != expected_datasets
        or len(suite.examples) != len(expected_items)
    ):
        raise ValueError(f"{method} raw benchmark suite identity drift")
    expected_manifest = _build_expected_full_score_benchmark_manifest(
        method=method,
        shard_id=shard_id,
        protocol_examples=protocol_examples,
        measurements=result.measurements,
    )
    if not _json_type_exact_equal(
        benchmark_experiment_manifest_to_record(manifest),
        benchmark_experiment_manifest_to_record(expected_manifest),
    ):
        raise ValueError(f"{method} raw benchmark complete manifest protocol drift")
    manifest_contract = {
        "baseline_arm_id": manifest.baseline_arm_id,
        "benchmark_seed": manifest.benchmark_seed,
        "comparison_mode": manifest.comparison_mode,
        "complete_dataset_split": manifest.complete_dataset_split,
        "datasets": list(manifest.datasets),
        "decode_settings": dict(manifest.decode_settings),
        "example_count": manifest.example_count,
        "execution_isolation_mode": manifest.execution_isolation_mode,
        "experiment_id": manifest.experiment_id,
        "generation_seed": manifest.generation_seed,
        "input_tokens_target": manifest.input_tokens_target,
        "isolate_arms": manifest.isolate_arms,
        "measurement_scopes": list(manifest.measurement_scopes),
        "model_id": manifest.model_id,
        "order_mode": manifest.order_mode,
        "output_tokens_target": manifest.output_tokens_target,
        "repeats": manifest.repeats,
        "request_parallelism": manifest.request_parallelism,
        "runtime_id": manifest.runtime_id,
        "shuffle": manifest.shuffle,
        "stream": manifest.stream,
        "temperature": manifest.temperature,
        "varied_setting": manifest.varied_setting,
        "warmups": manifest.warmups,
    }
    expected_manifest_contract: dict[str, Any] = {
        "baseline_arm_id": expected_arm_id,
        "benchmark_seed": None,
        "comparison_mode": "methods_same_setting",
        "complete_dataset_split": False,
        "datasets": list(expected_datasets),
        "decode_settings": {},
        "example_count": len(expected_items),
        "execution_isolation_mode": "shared_process_sequential",
        "experiment_id": expected_suite_id,
        "generation_seed": None,
        "input_tokens_target": None,
        "isolate_arms": True,
        "measurement_scopes": ["quality"],
        "model_id": FULL_SCORE_SERVED_MODEL_NAME,
        "order_mode": "arm_isolated",
        "output_tokens_target": FULL_SCORE_MAX_TOKENS,
        "repeats": FULL_SCORE_PASSES_PER_METHOD,
        "request_parallelism": FULL_SCORE_REQUEST_PARALLELISM,
        "runtime_id": f"{FULL_SCORE_PROTOCOL_ID}:{shard_id}",
        "shuffle": False,
        "stream": True,
        "temperature": FULL_SCORE_TEMPERATURE,
        "varied_setting": "",
        "warmups": 0,
    }
    if not _json_type_exact_equal(manifest_contract, expected_manifest_contract):
        raise ValueError(f"{method} raw benchmark manifest protocol drift")
    if len(manifest.arms) != 1:
        raise ValueError(f"{method} raw benchmark must contain one isolated arm")
    arm = manifest.arms[0]
    runtime_environment = arm.runtime_environment
    arm_contract = {
        "arm_id": arm.arm_id,
        "implementation_kind": arm.implementation_kind,
        "method_id": arm.method_id,
        "requires_cachet_handoff": arm.requires_cachet_handoff,
        "uses_cache": arm.uses_cache,
    }
    expected_arm_contract = {
        "arm_id": expected_arm_id,
        "implementation_kind": (
            "baseline" if method == "baseline_prefill" else "cachet"
        ),
        "method_id": "" if method == "baseline_prefill" else "vanilla_prefill",
        "requires_cachet_handoff": method == "vanilla_prefill",
        "uses_cache": method == "vanilla_prefill",
    }
    if not _json_type_exact_equal(arm_contract, expected_arm_contract):
        raise ValueError(f"{method} raw benchmark arm identity drift")
    runtime_contract = {
        "canonical_model_id": runtime_environment.canonical_model_id,
        "engine_id": runtime_environment.engine_id,
        "engine_version": runtime_environment.engine_version,
        "hardware_target": runtime_environment.hardware_target,
        "model_dtype": runtime_environment.model_dtype,
        "model_quantization": runtime_environment.model_quantization,
        "model_revision": runtime_environment.model_revision,
        "runtime_kv_dtype": runtime_environment.runtime_kv_dtype,
        "served_model_id": runtime_environment.served_model_id,
        "serving_platform": runtime_environment.serving_platform,
        "tokenizer_id": runtime_environment.tokenizer_id,
        "tokenizer_revision": runtime_environment.tokenizer_revision,
    }
    expected_runtime_contract = {
        "canonical_model_id": FULL_SCORE_MODEL_ID,
        "engine_id": "vllm",
        "engine_version": PUBLICATION_CAMPAIGN_ENGINE_VERSION,
        "hardware_target": "aws-g6-l4",
        "model_dtype": FULL_SCORE_MODEL_DTYPE,
        "model_quantization": FULL_SCORE_MODEL_QUANTIZATION,
        "model_revision": MAIN_LATENCY_TOKENIZER_REVISION,
        "runtime_kv_dtype": FULL_SCORE_KV_DTYPE,
        "served_model_id": FULL_SCORE_SERVED_MODEL_NAME,
        "serving_platform": "databricks-aws-single-gpu",
        "tokenizer_id": MAIN_LATENCY_TOKENIZER_ID,
        "tokenizer_revision": MAIN_LATENCY_TOKENIZER_REVISION,
    }
    if not _json_type_exact_equal(runtime_contract, expected_runtime_contract):
        raise ValueError(f"{method} raw benchmark runtime identity drift")


def build_full_score_connector_proof(
    connector_telemetry_path: str | Path,
    *,
    paired_examples: Sequence[Mapping[str, Any]],
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that every Vanilla example loaded its exact Q8 prefix once.

    This parser intentionally accepts only package-owned provider load records.
    File existence is not evidence: every planned Vanilla request must have one
    successful, full-layer, full-token load; any non-Vanilla load is rejected.
    """

    path = Path(connector_telemetry_path)
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("connector telemetry must be non-empty newline JSONL")
    planned_items = _shard_items_by_key(shard)
    expected: dict[str, dict[str, Any]] = {}
    observed_pair_keys: set[tuple[str, str]] = set()
    for raw_pair in paired_examples:
        pair = _json_mapping(raw_pair, "paired connector example")
        key = (_required_string(pair, "dataset"), _required_string(pair, "example_id"))
        if key in observed_pair_keys or key not in planned_items:
            raise ValueError(
                "connector proof has duplicate or unplanned paired example"
            )
        observed_pair_keys.add(key)
        methods = _required_mapping(pair, "methods")
        baseline = _required_mapping(methods, "baseline_prefill")
        vanilla = _required_mapping(methods, "vanilla_prefill")
        baseline_request_id = baseline.get("request_id")
        if baseline_request_id != "":
            raise ValueError(
                "Baseline request ID must reflect the uncached runner's empty ID"
            )
        vanilla_request_id = _required_string(vanilla, "request_id")
        if vanilla_request_id in expected:
            raise ValueError("connector proof request IDs must be unique")
        item = planned_items[key]
        expected[vanilla_request_id] = {
            "artifact_id": _required_string(vanilla, "artifact_id"),
            "cache_prefix_tokens": _required_int(item, "cache_prefix_tokens"),
            "dataset": key[0],
            "example_id": key[1],
        }
    if observed_pair_keys != set(planned_items):
        raise ValueError("connector proof paired coverage is incomplete")

    observed: dict[str, dict[str, Any]] = {}
    load_record_count = 0
    for line_index, raw_line in enumerate(raw[:-1].split(b"\n"), start=1):
        if not raw_line:
            raise ValueError("connector telemetry contains an empty row")
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"connector telemetry row {line_index} is invalid JSON"
            ) from exc
        record = _json_mapping(value, "connector telemetry row")
        if record.get("event") != "load_request":
            continue
        load_record_count += 1
        if record.get("record_type") != "document_kv.vllm_native_provider_load.v1":
            raise ValueError("connector telemetry uses an unsupported load record")
        if record.get("provider_factory") != DOCUMENT_KV_NATIVE_PROVIDER_FACTORY:
            raise ValueError("connector telemetry provider-factory identity drift")
        request_id = _required_string(record, "benchmark_request_id")
        expected_request = expected.get(request_id)
        if expected_request is None:
            raise ValueError("connector telemetry references an unknown request")
        if request_id in observed:
            raise ValueError("Vanilla request performed more than one connector load")
        if record.get("success") is not True or "error" in record:
            raise ValueError("Vanilla connector load was not successful")
        counts = _required_mapping(record, "counts")
        layout = _required_mapping(record, "layout")
        attestation = _required_mapping(record, "cache_state_attestation")
        tokens = cast(int, expected_request["cache_prefix_tokens"])
        expected_runtime_bytes = tokens * FULL_SCORE_Q8_BYTES_PER_CACHE_PREFIX_TOKEN
        required_counts = {
            "decoded_runtime_payload_bytes": expected_runtime_bytes,
            "expected_runtime_payload_bytes": expected_runtime_bytes,
            "handoff_total_tokens": tokens,
            "layers_loaded": FULL_SCORE_MODEL_NUM_LAYERS,
            "token_count": tokens,
        }
        if any(
            counts.get(key) != expected_value
            for key, expected_value in required_counts.items()
        ):
            raise ValueError("connector telemetry count/layout coverage drift")
        required_layout = {
            "bytes_per_token": FULL_SCORE_Q8_BYTES_PER_CACHE_PREFIX_TOKEN,
            "dtype": FULL_SCORE_KV_DTYPE,
            "model_id": FULL_SCORE_MODEL_ID,
            "num_layers": FULL_SCORE_MODEL_NUM_LAYERS,
        }
        if any(
            layout.get(key) != expected_value
            for key, expected_value in required_layout.items()
        ):
            raise ValueError("connector telemetry model layout drift")
        required_attestation = {
            "artifact_id": expected_request["artifact_id"],
            "cache_method": "vanilla_prefill",
            "decoded_runtime_bytes": expected_runtime_bytes,
            "expected_runtime_bytes": expected_runtime_bytes,
            "expected_tokens": tokens,
            "loaded_tokens": tokens,
            "payload_cache_hit": False,
            "successful_loads": 1,
        }
        if any(
            attestation.get(key) != expected_value
            for key, expected_value in required_attestation.items()
        ):
            raise ValueError("connector cache-state attestation drift")
        observed[request_id] = {
            "artifact_id": expected_request["artifact_id"],
            "cache_prefix_tokens": tokens,
            "dataset": expected_request["dataset"],
            "example_id": expected_request["example_id"],
            "request_id": request_id,
        }
    if set(observed) != set(expected) or load_record_count != len(expected):
        raise ValueError("connector telemetry lacks exact Vanilla request coverage")
    proof: dict[str, Any] = {
        "closed_record_sha256": "",
        "connector_telemetry_file_sha256": sha256(raw).hexdigest(),
        "load_count": len(observed),
        "loads": [observed[request_id] for request_id in sorted(observed)],
        "record_type": FULL_SCORE_CONNECTOR_PROOF_RECORD_TYPE,
        "schema_version": FULL_SCORE_CONNECTOR_PROOF_SCHEMA_VERSION,
        "validation": (
            "exactly_one_successful_full_layer_full_token_q8_load_per_vanilla_request;"
            "zero_baseline_loads"
        ),
    }
    proof["closed_record_sha256"] = _closed_record_sha256(proof)
    return proof


def _require_full_score_final_consumer_aggregation_authorization(
    execution_plan: Mapping[str, Any],
    authorization: object,
    *,
    ledger_path: str | Path,
) -> dict[str, Any]:
    """Require the terminal consumer of the final frozen publication wave."""

    if type(authorization) is not FullScorePhaseAuthorization:
        raise TypeError(
            "publication aggregation requires final-consumer "
            "FullScorePhaseAuthorization"
        )
    waves = execution_plan.get("waves")
    if not isinstance(waves, list) or not waves:
        raise ValueError("publication aggregation execution plan has no waves")
    final_wave_index = len(waves) - 1
    plan_sha256 = _require_sha256(
        execution_plan.get("closed_record_sha256"),
        field_name="full-score execution-plan SHA-256",
    )
    live_path = Path(ledger_path).expanduser().absolute()
    path_sha256 = databricks_ledger_path_sha256(live_path)
    if (
        authorization.execution_plan_sha256 != plan_sha256
        or authorization.wave_index != final_wave_index
        or authorization.phase != "consumer"
    ):
        raise ValueError(
            "publication aggregation requires the final-wave consumer authority"
        )
    if authorization.ledger_path_sha256 != path_sha256:
        raise ValueError("publication aggregate ledger path binding drift")
    ledger = read_databricks_cluster_hour_ledger_json(live_path)
    _require_full_score_ledger_caps(ledger)
    require_databricks_ledger_prefix(ledger, authorization.ledger_prefix)
    current_prefix = databricks_ledger_prefix(ledger)
    if current_prefix != authorization.ledger_prefix:
        raise ValueError(
            "publication aggregate final consumer is not the complete current "
            "ledger prefix"
        )
    if (
        ledger.active_reserved_cluster_hours != 0
        or ledger.active_reserved_task_count != 0
    ):
        raise ValueError("publication aggregate final ledger is not active-zero")
    predecessor_prefix = authorization.predecessor_prefix
    require_databricks_ledger_prefix(ledger, predecessor_prefix)
    observed_predecessor = databricks_ledger_prefix_at_counts(
        ledger,
        reservation_count=predecessor_prefix.reservation_count,
        submission_receipt_count=predecessor_prefix.submission_receipt_count,
        terminal_actual_count=predecessor_prefix.terminal_actual_count,
    )
    if observed_predecessor != predecessor_prefix:
        raise ValueError("publication aggregate predecessor prefix drift")
    batch_prefix = databricks_ledger_prefix_at_counts(
        ledger,
        reservation_count=predecessor_prefix.reservation_count + 1,
        submission_receipt_count=predecessor_prefix.submission_receipt_count,
        terminal_actual_count=predecessor_prefix.terminal_actual_count,
    )
    if (
        batch_prefix.ledger_id != predecessor_prefix.ledger_id
        or batch_prefix.cap_cluster_hours != predecessor_prefix.cap_cluster_hours
    ):
        raise ValueError("publication aggregate final batch prefix drift")
    expected_causal_closure = _canonical_sha256(
        {
            "batch_prefix": batch_prefix.to_record(),
            "ledger_path_sha256": path_sha256,
            "terminal_prefix": current_prefix.to_record(),
            "terminal_record_sha256": authorization.terminal_record_sha256,
        }
    )
    if authorization.causal_closure_sha256 != expected_causal_closure:
        raise ValueError("publication aggregate final authority closure drift")
    expected_workspace_closure = _canonical_sha256(
        {
            "causal_closure_sha256": authorization.causal_closure_sha256,
            "phase_lease_root_sha256": _canonical_sha256(
                {
                    "domain": "cachet.full_score_phase_lease_root_authority.v1",
                    "phase_lease_root": str(authorization.phase_lease_root),
                }
            ),
            "user_name_sha256": authorization.user_name_sha256,
            "workspace_host_sha256": authorization.workspace_host_sha256,
        }
    )
    if authorization.workspace_authority_closure_sha256 != expected_workspace_closure:
        raise ValueError("publication aggregate final workspace authority drift")
    return {
        "authorization_sha256": authorization.causal_closure_sha256,
        "batch_prefix": batch_prefix.to_record(),
        "ledger_id": ledger.ledger_id,
        "ledger_path_sha256": path_sha256,
        "predecessor_prefix": predecessor_prefix.to_record(),
        "terminal_prefix": current_prefix.to_record(),
        "terminal_record_sha256": authorization.terminal_record_sha256,
        "wave_index": final_wave_index,
    }


def _full_score_aggregate_bootstrap_contract() -> dict[str, Any]:
    return {
        "confidence_level": 0.95,
        "draws": PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS,
        "resampling_unit": "paired_example_within_dataset_or_niah_cell",
        "rng_algorithm": "cpython-3.11-random.Random-mt19937-choices-v1",
        "seed_domain": "cachet.full_score.paired_bootstrap.seed.v1",
        "stratification": "dataset; niah additionally by frozen 9-cell grid",
        "tail_algorithm": "linear_interpolated_empirical_quantile_type7_v1",
    }


def aggregate_full_score_shard_evidence(
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    evidence_records: Sequence[Mapping[str, Any] | str | Path],
    *,
    authorization_scope: str,
    execution_plan: Mapping[str, Any] | None = None,
    final_consumer_authorization: FullScorePhaseAuthorization | None = None,
    ledger_path: str | Path | None = None,
    remote_consumer_authorizations: Sequence[object] | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> dict[str, Any]:
    """Aggregate every example once; shard-level means are never consumed."""

    validate_full_score_shard_plan(shard_plan, inventory=inventory)
    if authorization_scope not in {
        FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE,
        FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
    }:
        raise ValueError("aggregate authorization_scope is invalid")
    normalized_evidence: list[Mapping[str, Any]] = []
    publication_lineage: dict[str, Any] | None = None
    governed_evidence_bindings: list[dict[str, Any]] = []
    remote_authorization_bindings: list[dict[str, Any]] = []
    if authorization_scope == FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE:
        if execution_plan is None:
            raise ValueError("publication aggregation requires the execution plan")
        _validate_publication_full_score_inputs(inventory, shard_plan, execution_plan)
        if type(final_consumer_authorization) is not FullScorePhaseAuthorization:
            raise TypeError(
                "publication aggregation requires final-consumer "
                "FullScorePhaseAuthorization"
            )
        if ledger_path is None:
            raise ValueError(
                "publication aggregation requires the canonical local ledger path"
            )
        publication_lineage = (
            _require_full_score_final_consumer_aggregation_authorization(
                execution_plan,
                final_consumer_authorization,
                ledger_path=ledger_path,
            )
        )
        if evidence_records:
            raise ValueError(
                "publication aggregation consumes remote CAS authority, not directories"
            )
        if compact_artifact_resolver is None:
            raise TypeError("publication aggregation requires a compact CAS resolver")
        from document_kv_cache.full_score_remote_control import (
            require_full_score_remote_consumer_evidence_authorizations,
        )

        consumer_authorizations = (
            require_full_score_remote_consumer_evidence_authorizations(
                ()
                if remote_consumer_authorizations is None
                else remote_consumer_authorizations,
                execution_plan=execution_plan,
            )
        )
        if any(
            (
                authorization.workspace_host_sha256,
                authorization.user_name_sha256,
            )
            != (
                final_consumer_authorization.workspace_host_sha256,
                final_consumer_authorization.user_name_sha256,
            )
            for authorization in consumer_authorizations
        ):
            raise ValueError(
                "publication aggregation workspace authority lineage drift"
            )
        if (
            not consumer_authorizations
            or consumer_authorizations[-1].wave_index
            != final_consumer_authorization.wave_index
            or consumer_authorizations[-1].phase_terminal_record_sha256
            != final_consumer_authorization.terminal_record_sha256
        ):
            raise ValueError(
                "publication aggregation final ledger/tree authority drift"
            )
        for authorization in consumer_authorizations:
            remote_authorization_bindings.append(
                {
                    "authorization_record_sha256": (
                        authorization.controller_authorization_record_sha256
                    ),
                    "coordinator_run_id": authorization.coordinator_run_id,
                    "execution_plan_sha256": authorization.execution_plan_sha256,
                    "phase_terminal_record_sha256": (
                        authorization.phase_terminal_record_sha256
                    ),
                    "runs_get_receipt_record_sha256": (
                        authorization.runs_get_receipt_record_sha256
                    ),
                    "wave_index": authorization.wave_index,
                }
            )
            for binding in authorization.evidence_bindings:
                evidence_file = _governed_compact_file(
                    cast(str, binding["evidence_uri"]),
                    "aggregate shard evidence",
                    compact_artifact_resolver,
                )
                deletion_file = _governed_compact_file(
                    cast(str, binding["deletion_uri"]),
                    "aggregate deletion attestation",
                    compact_artifact_resolver,
                )
                evidence_bytes = evidence_file.read_bytes()
                deletion_bytes = deletion_file.read_bytes()
                evidence = _json_object(evidence_bytes, "aggregate shard evidence")
                deletion = _json_object(
                    deletion_bytes,
                    "aggregate deletion attestation",
                )
                if (
                    evidence_bytes != _canonical_pretty_json_bytes(evidence)
                    or deletion_bytes != _canonical_pretty_json_bytes(deletion)
                    or sha256(evidence_bytes).hexdigest()
                    != binding["evidence_file_sha256"]
                    or evidence.get("closed_record_sha256")
                    != binding["evidence_record_sha256"]
                    or sha256(deletion_bytes).hexdigest()
                    != binding["deletion_file_sha256"]
                    or deletion.get("closed_record_sha256")
                    != binding["deletion_record_sha256"]
                ):
                    raise ValueError("publication aggregate CAS evidence binding drift")
                _validate_shard_evidence_record(
                    evidence,
                    inventory_sha256=inventory.inventory_sha256,
                    shard_plan_sha256=_required_string(
                        shard_plan,
                        "closed_record_sha256",
                    ),
                )
                shard_id = _required_string(evidence, "shard_id")
                if shard_id != binding["shard_id"]:
                    raise ValueError(
                        "publication aggregate evidence shard binding drift"
                    )
                _validate_full_score_deletion_attestation(
                    deletion,
                    evidence_record=evidence,
                    execution_plan_sha256=_required_string(
                        execution_plan,
                        "closed_record_sha256",
                    ),
                    shard_id=shard_id,
                    wave_index=authorization.wave_index,
                )
                governed_evidence_bindings.append(
                    {
                        **dict(binding),
                        "authorization_record_sha256": (
                            authorization.controller_authorization_record_sha256
                        ),
                        "wave_index": authorization.wave_index,
                    }
                )
                normalized_evidence.append(evidence)
    else:
        if (
            final_consumer_authorization is not None
            or ledger_path is not None
            or remote_consumer_authorizations is not None
            or compact_artifact_resolver is not None
        ):
            raise ValueError(
                "local-fixture aggregation cannot consume publication authority"
            )
        for raw_record in evidence_records:
            if not isinstance(raw_record, Mapping):
                raise ValueError("local-fixture aggregation requires in-memory records")
            fixture = dict(raw_record)
            fixture.setdefault(
                "authorization_scope",
                FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
            )
            fixture["closed_record_sha256"] = _closed_record_sha256(fixture)
            normalized_evidence.append(fixture)
    expected_plan_sha = _required_string(shard_plan, "closed_record_sha256")
    expected_shards = {
        cast(str, shard["shard_id"]): shard
        for shard in cast(list[Mapping[str, Any]], shard_plan["shards"])
    }
    inventory_items = {
        (item.dataset, item.example_id): item for item in inventory.items
    }
    scorer_registry = default_dataset_scorer_registry()
    observed_shards: dict[str, Mapping[str, Any]] = {}
    per_example: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_record in normalized_evidence:
        evidence_record = _json_mapping(raw_record, "shard evidence")
        _validate_shard_evidence_record(
            evidence_record,
            inventory_sha256=inventory.inventory_sha256,
            shard_plan_sha256=expected_plan_sha,
        )
        shard_id = _required_string(evidence_record, "shard_id")
        if shard_id not in expected_shards:
            raise ValueError("shard evidence references an unknown shard")
        if shard_id in observed_shards:
            raise ValueError("duplicate shard evidence")
        if evidence_record.get("shard_items_sha256") != expected_shards[shard_id].get(
            "items_sha256"
        ):
            raise ValueError("shard evidence item closure drift")
        observed_shards[shard_id] = evidence_record
        pairs = evidence_record.get("paired_examples")
        if not isinstance(pairs, list):
            raise ValueError("shard evidence paired_examples must be an array")
        for pair in pairs:
            pair_record = _json_mapping(pair, "paired example")
            key = (
                _required_string(pair_record, "dataset"),
                _required_string(pair_record, "example_id"),
            )
            if key in per_example:
                raise ValueError("full-score evidence scores an example more than once")
            inventory_item = inventory_items.get(key)
            if inventory_item is None:
                raise ValueError("full-score evidence scores an unknown example")
            if pair_record.get("identity_sha256") != inventory_item.identity_sha256:
                raise ValueError("paired-example identity closure drift")
            if pair_record.get("natural_prompt_sha256") != (
                inventory_item.natural_prompt_sha256
            ):
                raise ValueError("paired-example natural-prompt closure drift")
            niah_cell_id = pair_record.get("niah_cell_id")
            if key[0] == "niah":
                if niah_cell_id not in NIAH_CELL_IDS:
                    raise ValueError("paired NIAH example has an invalid cell identity")
            elif niah_cell_id is not None:
                raise ValueError("non-NIAH paired example declares a NIAH cell")
            pair_methods = pair_record.get("methods")
            if not isinstance(pair_methods, Mapping) or tuple(
                sorted(pair_methods)
            ) != tuple(sorted(FULL_SCORE_METHODS)):
                raise ValueError(
                    "paired example must contain both methods exactly once"
                )
            scorer = scorer_registry.get(key[0])
            for method in FULL_SCORE_METHODS:
                method_record = _required_mapping(pair_methods, method)
                if set(method_record) != {
                    "artifact_id",
                    "completion_tokens",
                    "output_sha256",
                    "parser_status",
                    "parser_valid",
                    "quality_scores",
                    "request_id",
                    "scorer_id",
                    "scorer_version",
                }:
                    raise ValueError("paired-example method evidence schema drift")
                if (
                    method_record.get("scorer_id") != scorer.scorer_id
                    or method_record.get("scorer_version") != scorer.version
                ):
                    raise ValueError("paired-example scorer identity drift")
                _require_sha256(
                    method_record.get("output_sha256"),
                    field_name="paired-example output_sha256",
                )
                request_id = method_record.get("request_id")
                artifact_id = method_record.get("artifact_id")
                if method == "baseline_prefill":
                    if request_id != "":
                        raise ValueError(
                            "Baseline paired evidence request ID must be empty"
                        )
                    if artifact_id is not None:
                        raise ValueError(
                            "Baseline paired evidence declares an artifact"
                        )
                else:
                    _require_nonempty(
                        request_id,
                        "paired-example Vanilla request_id",
                    )
                    _require_nonempty(
                        artifact_id,
                        "paired-example Vanilla artifact_id",
                    )
                parser_status = _require_nonempty(
                    method_record.get("parser_status"),
                    "paired-example parser_status",
                )
                if parser_status not in FINAL_ANSWER_PARSER_STATUSES:
                    raise ValueError(
                        "paired-example parser_status is outside the frozen states"
                    )
                parser_valid = method_record.get("parser_valid")
                if type(parser_valid) is not bool:
                    raise ValueError("paired-example parser validity is invalid")
                if parser_valid != (parser_status == "ok"):
                    raise ValueError(
                        "paired-example parser validity/status are inconsistent"
                    )
                completion_tokens = method_record.get("completion_tokens")
                if (
                    type(completion_tokens) is not int
                    or not 0 <= completion_tokens <= FULL_SCORE_MAX_TOKENS
                ):
                    raise ValueError("paired-example completion-token count is invalid")
                scores = _required_mapping(method_record, "quality_scores")
                if set(scores) != set(scorer.metric_names):
                    raise ValueError("paired-example metric identity drift")
                for metric, value in scores.items():
                    score = _nonnegative_finite_float(
                        value,
                        f"paired-example quality_scores.{metric}",
                    )
                    if score > 1:
                        raise ValueError("paired-example quality score exceeds one")
                    if not parser_valid and score != 0:
                        raise ValueError(
                            "invalid parsed answers must receive zero quality scores"
                        )
            per_example[key] = pair_record
    if set(observed_shards) != set(expected_shards):
        raise ValueError("full-score evidence does not cover every planned shard")
    expected_keys = {(item.dataset, item.example_id) for item in inventory.items}
    if set(per_example) != expected_keys:
        raise ValueError("full-score evidence does not cover every inventory ID")

    accumulators: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    invalid_parser_sums: dict[tuple[str, str, str], float] = defaultdict(float)
    parser_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    niah_cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    niah_invalid_parser_sums: dict[tuple[str, str, str], float] = defaultdict(float)
    for (dataset, _example_id), pair in per_example.items():
        pair_methods = cast(Mapping[str, Mapping[str, Any]], pair["methods"])
        for method in FULL_SCORE_METHODS:
            method_record = pair_methods[method]
            quality_scores = method_record.get("quality_scores")
            if not isinstance(quality_scores, Mapping):
                raise ValueError("per-example quality_scores must be an object")
            for metric, raw_value in quality_scores.items():
                if not isinstance(metric, str) or not isinstance(
                    raw_value, (int, float)
                ):
                    raise ValueError("per-example score is invalid")
                accumulators[(dataset, method, metric)].append(float(raw_value))
                if method_record.get("parser_valid") is not True:
                    invalid_parser_sums[(dataset, method, metric)] += float(raw_value)
                if dataset == "niah" and pair.get("niah_cell_id"):
                    niah_cells[
                        (cast(str, pair["niah_cell_id"]), method, metric)
                    ].append(float(raw_value))
                    if method_record.get("parser_valid") is not True:
                        niah_invalid_parser_sums[
                            (cast(str, pair["niah_cell_id"]), method, metric)
                        ] += float(raw_value)
            parser_counts[(dataset, method)][
                cast(str, method_record["parser_status"])
            ] += 1

    niah_identity_count = sum(1 for key in per_example if key[0] == "niah")
    if niah_identity_count != 1000:
        raise ValueError(
            "the corrected full NIAH score requires exactly 1,000 examples"
        )
    observed_niah_cells = {
        cast(str, pair.get("niah_cell_id"))
        for (dataset, _example_id), pair in per_example.items()
        if dataset == "niah" and pair.get("niah_cell_id")
    }
    if observed_niah_cells != set(NIAH_CELL_IDS):
        raise ValueError("the corrected full NIAH score requires all nine cells")

    bootstrap_identity = _full_score_aggregate_bootstrap_contract()
    datasets: dict[str, Any] = {}
    for dataset in SUPPORTED_V1_DATASETS:
        count = sum(1 for key in expected_keys if key[0] == dataset)
        dataset_methods: dict[str, Any] = {}
        for method in FULL_SCORE_METHODS:
            metrics = {
                metric: {
                    "example_count": len(values),
                    "invalid_parser_score_sum": invalid_parser_sums[
                        (candidate_dataset, candidate_method, metric)
                    ],
                    "mean": sum(values) / len(values),
                    "sum": sum(values),
                }
                for (candidate_dataset, candidate_method, metric), values in sorted(
                    accumulators.items()
                )
                if candidate_dataset == dataset and candidate_method == method
            }
            if not metrics or any(
                value["example_count"] != count for value in metrics.values()
            ):
                raise ValueError("dataset metric coverage is incomplete")
            status_counts = {
                status: parser_counts[(dataset, method)][status]
                for status in FINAL_ANSWER_PARSER_STATUSES
            }
            if sum(status_counts.values()) != count:
                raise ValueError("dataset parser-status coverage is incomplete")
            dataset_methods[method] = {
                "example_count": count,
                "metrics": metrics,
                "parser_status_counts": status_counts,
            }
        paired_deltas = _paired_delta_summaries(
            {key: pair for key, pair in per_example.items() if key[0] == dataset},
            dataset=dataset,
            inventory_sha256=inventory.inventory_sha256,
            shard_plan_sha256=expected_plan_sha,
        )
        datasets[dataset] = {
            "example_count": count,
            "methods": dataset_methods,
            "paired_vanilla_minus_baseline": paired_deltas,
        }
    niah_grid: dict[str, Any] = {}
    for cell_id in NIAH_CELL_IDS:
        cell_methods = {
            method: {
                metric: {
                    "example_count": len(values),
                    "invalid_parser_score_sum": niah_invalid_parser_sums[
                        (candidate_cell, candidate_method, metric)
                    ],
                    "mean": sum(values) / len(values),
                    "sum": sum(values),
                }
                for (candidate_cell, candidate_method, metric), values in sorted(
                    niah_cells.items()
                )
                if candidate_cell == cell_id and candidate_method == method
            }
            for method in FULL_SCORE_METHODS
        }
        cell_pairs = {
            key: pair
            for key, pair in per_example.items()
            if key[0] == "niah" and pair.get("niah_cell_id") == cell_id
        }
        if not cell_pairs:
            raise ValueError(f"NIAH cell {cell_id} is empty")
        niah_grid[cell_id] = {
            "example_count": len(cell_pairs),
            "methods": cell_methods,
            "paired_vanilla_minus_baseline": _paired_delta_summaries(
                cell_pairs,
                dataset=f"niah/{cell_id}",
                inventory_sha256=inventory.inventory_sha256,
                shard_plan_sha256=expected_plan_sha,
            ),
        }
    aggregate_record: dict[str, Any] = {
        "aggregation_unit": "per_example_once_never_shard_means",
        "authorization_scope": authorization_scope,
        "closed_record_sha256": "",
        "bootstrap": bootstrap_identity,
        "datasets": datasets,
        "identity_count": len(per_example),
        "inventory_sha256": inventory.inventory_sha256,
        "methods": list(FULL_SCORE_METHODS),
        "niah_grid": niah_grid,
        "passes_per_method": FULL_SCORE_PASSES_PER_METHOD,
        "protocol": _full_score_protocol_record(),
        "record_type": FULL_SCORE_AGGREGATE_RECORD_TYPE,
        "scorers": _scorer_contract_record(),
        "schema_version": FULL_SCORE_AGGREGATE_SCHEMA_VERSION,
        "shard_count": len(observed_shards),
        "shard_plan_sha256": expected_plan_sha,
    }
    if publication_lineage is not None:
        assert execution_plan is not None
        assert ledger_path is not None
        rechecked_lineage = (
            _require_full_score_final_consumer_aggregation_authorization(
                execution_plan,
                final_consumer_authorization,
                ledger_path=ledger_path,
            )
        )
        if rechecked_lineage != publication_lineage:
            raise ValueError(
                "publication aggregate final ledger lineage changed during aggregation"
            )
        publication_lineage = rechecked_lineage
        publication_lineage["evidence"] = sorted(
            governed_evidence_bindings,
            key=lambda item: cast(str, item["shard_id"]),
        )
        publication_lineage["remote_consumer_authorizations"] = sorted(
            remote_authorization_bindings,
            key=lambda item: cast(int, item["wave_index"]),
        )
        aggregate_record["execution_plan_sha256"] = _required_string(
            execution_plan,
            "closed_record_sha256",
        )
        aggregate_record["publication_lineage"] = publication_lineage
    aggregate_record["closed_record_sha256"] = _closed_record_sha256(aggregate_record)
    validate_full_score_aggregate_record(
        aggregate_record,
        inventory=inventory,
        shard_plan=shard_plan,
        execution_plan=execution_plan,
        require_publication=(
            authorization_scope == FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
        ),
    )
    return aggregate_record


def validate_full_score_aggregate_record(
    record: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any] | None = None,
    require_publication: bool = False,
) -> None:
    """Strictly validate one closed full-score publication result.

    The aggregate intentionally omits raw model outputs.  This validator
    therefore proves the complete closed summary algebra and, for publication
    records, the exact remote shard/wave and terminal-ledger lineage.  Raw score
    replay remains the responsibility of ``aggregate_full_score_shard_evidence``.
    """

    if not isinstance(record, Mapping):
        raise TypeError("full-score aggregate must be an object")
    if type(require_publication) is not bool:
        raise TypeError("require_publication must be a boolean")
    if record.get("record_type") != FULL_SCORE_AGGREGATE_RECORD_TYPE:
        raise ValueError("unsupported full-score aggregate record_type")
    if (
        type(record.get("schema_version")) is not int
        or record.get("schema_version") != FULL_SCORE_AGGREGATE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported full-score aggregate schema_version")
    _require_sha256(
        record.get("closed_record_sha256"),
        field_name="full-score aggregate closed_record_sha256",
    )
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("full-score aggregate closure drift")

    scope = record.get("authorization_scope")
    if scope not in {
        FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE,
        FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
    }:
        raise ValueError("full-score aggregate authorization_scope drift")
    publication_scoped = scope == FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
    if require_publication and not publication_scoped:
        raise ValueError(
            "publication full-score aggregate rejects local_fixture_only scope"
        )
    expected_keys = {
        "aggregation_unit",
        "authorization_scope",
        "bootstrap",
        "closed_record_sha256",
        "datasets",
        "identity_count",
        "inventory_sha256",
        "methods",
        "niah_grid",
        "passes_per_method",
        "protocol",
        "record_type",
        "schema_version",
        "scorers",
        "shard_count",
        "shard_plan_sha256",
    }
    if publication_scoped:
        expected_keys.update({"execution_plan_sha256", "publication_lineage"})
    if set(record) != expected_keys:
        raise ValueError("full-score aggregate top-level schema drift")

    if not isinstance(inventory, FullScoreInventory):
        raise TypeError("full-score aggregate inventory must be FullScoreInventory")
    validate_full_score_shard_plan(shard_plan, inventory=inventory)
    shard_plan_sha256 = _require_sha256(
        shard_plan.get("closed_record_sha256"),
        field_name="full-score aggregate shard-plan SHA-256",
    )
    if record.get("inventory_sha256") != inventory.inventory_sha256:
        raise ValueError("full-score aggregate inventory binding drift")
    if record.get("shard_plan_sha256") != shard_plan_sha256:
        raise ValueError("full-score aggregate shard-plan binding drift")

    if publication_scoped:
        if execution_plan is None:
            raise ValueError(
                "publication full-score aggregate requires the remote execution plan"
            )
        _validate_publication_full_score_inputs(
            inventory,
            shard_plan,
            execution_plan,
        )
        execution_plan_sha256 = _require_sha256(
            execution_plan.get("closed_record_sha256"),
            field_name="full-score aggregate execution-plan SHA-256",
        )
        if record.get("execution_plan_sha256") != execution_plan_sha256:
            raise ValueError("full-score aggregate execution-plan binding drift")
    elif execution_plan is not None:
        _validate_execution_plan(
            execution_plan,
            inventory=inventory,
            shard_plan=shard_plan,
        )

    raw_shards = shard_plan.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("full-score aggregate shard plan has no shard array")
    if _required_int(record, "identity_count") != len(inventory.items) or _required_int(
        record, "shard_count"
    ) != len(raw_shards):
        raise ValueError("full-score aggregate inventory/shard count drift")
    if record.get("aggregation_unit") != ("per_example_once_never_shard_means"):
        raise ValueError("full-score aggregate aggregation-unit drift")
    if record.get("methods") != list(FULL_SCORE_METHODS):
        raise ValueError("full-score aggregate method contract drift")
    if _required_int(record, "passes_per_method") != FULL_SCORE_PASSES_PER_METHOD:
        raise ValueError("full-score aggregate pass contract drift")
    if not _json_type_exact_equal(
        record.get("protocol"),
        _full_score_protocol_record(),
    ):
        raise ValueError("full-score aggregate protocol contract drift")
    if record.get("bootstrap") != _full_score_aggregate_bootstrap_contract():
        raise ValueError("full-score aggregate bootstrap contract drift")
    scorer_contract = _scorer_contract_record()
    if not _json_type_exact_equal(record.get("scorers"), scorer_contract):
        raise ValueError("full-score aggregate scorer/parser contract drift")

    expected_counts = Counter(item.dataset for item in inventory.items)
    metric_names_by_dataset = {
        scorer_record["dataset"]: tuple(scorer_record["metric_names"])
        for scorer_record in scorer_contract
    }
    raw_datasets = record.get("datasets")
    if not isinstance(raw_datasets, Mapping) or set(raw_datasets) != set(
        SUPPORTED_V1_DATASETS
    ):
        raise ValueError("full-score aggregate dataset coverage drift")
    dataset_method_summaries: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    dataset_paired_means: dict[str, dict[str, float]] = {}
    bootstrap_draws = _required_int(
        _required_mapping(record, "bootstrap"),
        "draws",
    )
    for dataset in SUPPORTED_V1_DATASETS:
        dataset_record = _required_mapping(raw_datasets, dataset)
        if set(dataset_record) != {
            "example_count",
            "methods",
            "paired_vanilla_minus_baseline",
        }:
            raise ValueError("full-score aggregate dataset schema drift")
        expected_count = expected_counts[dataset]
        if _required_int(dataset_record, "example_count") != expected_count:
            raise ValueError("full-score aggregate dataset count drift")
        metric_names = metric_names_by_dataset.get(dataset)
        if not metric_names:
            raise ValueError("full-score aggregate scorer dataset coverage drift")
        methods = _required_mapping(dataset_record, "methods")
        if set(methods) != set(FULL_SCORE_METHODS):
            raise ValueError("full-score aggregate dataset method coverage drift")
        method_summaries: dict[str, dict[str, tuple[float, float]]] = {}
        for method in FULL_SCORE_METHODS:
            method_record = _required_mapping(methods, method)
            if set(method_record) != {
                "example_count",
                "metrics",
                "parser_status_counts",
            }:
                raise ValueError("full-score aggregate method schema drift")
            if _required_int(method_record, "example_count") != expected_count:
                raise ValueError("full-score aggregate method count drift")
            raw_metrics = _required_mapping(method_record, "metrics")
            if set(raw_metrics) != set(metric_names):
                raise ValueError("full-score aggregate metric coverage drift")
            method_summaries[method] = {
                metric: _validate_full_score_aggregate_metric_summary(
                    _required_mapping(raw_metrics, metric),
                    expected_count=expected_count,
                    label=f"datasets.{dataset}.{method}.{metric}",
                )
                for metric in metric_names
            }
            parser_counts = _required_mapping(
                method_record,
                "parser_status_counts",
            )
            if set(parser_counts) != set(FINAL_ANSWER_PARSER_STATUSES):
                raise ValueError("full-score aggregate parser-status schema drift")
            observed_parser_count = 0
            for status in FINAL_ANSWER_PARSER_STATUSES:
                status_count = parser_counts.get(status)
                if type(status_count) is not int or status_count < 0:
                    raise ValueError("full-score aggregate parser-status count drift")
                observed_parser_count += status_count
            if observed_parser_count != expected_count:
                raise ValueError("full-score aggregate parser-status coverage drift")
            valid_parse_count = parser_counts["ok"]
            if any(
                total > valid_parse_count
                and not _full_score_aggregate_numbers_match(
                    total,
                    float(valid_parse_count),
                )
                for _mean, total in method_summaries[method].values()
            ):
                raise ValueError("full-score aggregate credits invalid parsed answers")
        dataset_method_summaries[dataset] = method_summaries
        dataset_paired_means[dataset] = _validate_full_score_aggregate_paired_deltas(
            _required_mapping(
                dataset_record,
                "paired_vanilla_minus_baseline",
            ),
            dataset_stratum=dataset,
            expected_count=expected_count,
            metric_names=metric_names,
            method_summaries=method_summaries,
            inventory_sha256=inventory.inventory_sha256,
            shard_plan_sha256=shard_plan_sha256,
            bootstrap_draws=bootstrap_draws,
            label=f"datasets.{dataset}.paired",
        )

    _validate_full_score_aggregate_niah_grid(
        record.get("niah_grid"),
        expected_count=expected_counts["niah"],
        metric_names=metric_names_by_dataset["niah"],
        dataset_method_summaries=dataset_method_summaries["niah"],
        dataset_paired_means=dataset_paired_means["niah"],
        inventory_sha256=inventory.inventory_sha256,
        shard_plan_sha256=shard_plan_sha256,
        bootstrap_draws=bootstrap_draws,
    )
    if publication_scoped:
        assert execution_plan is not None
        _validate_full_score_aggregate_publication_lineage(
            _required_mapping(record, "publication_lineage"),
            execution_plan=execution_plan,
            shard_plan=shard_plan,
        )


def _validate_full_score_aggregate_metric_summary(
    record: Mapping[str, Any],
    *,
    expected_count: int,
    label: str,
) -> tuple[float, float]:
    if set(record) != {
        "example_count",
        "invalid_parser_score_sum",
        "mean",
        "sum",
    }:
        raise ValueError(f"{label} metric summary schema drift")
    if _required_int(record, "example_count") != expected_count:
        raise ValueError(f"{label} metric example count drift")
    mean = _full_score_aggregate_finite_number(record.get("mean"), f"{label}.mean")
    total = _full_score_aggregate_finite_number(record.get("sum"), f"{label}.sum")
    invalid_parser_score_sum = _full_score_aggregate_finite_number(
        record.get("invalid_parser_score_sum"),
        f"{label}.invalid_parser_score_sum",
    )
    if invalid_parser_score_sum != 0.0:
        raise ValueError(f"{label} credits an invalid parsed answer")
    if (
        not 0 <= mean <= 1
        or not 0 <= total <= expected_count
        or not _full_score_aggregate_numbers_match(
            mean,
            total / expected_count,
        )
    ):
        raise ValueError(f"{label} metric mean/sum identity drift")
    return mean, total


def _validate_full_score_aggregate_paired_deltas(
    record: Mapping[str, Any],
    *,
    dataset_stratum: str,
    expected_count: int,
    metric_names: Sequence[str],
    method_summaries: Mapping[str, Mapping[str, tuple[float, float]]],
    inventory_sha256: str,
    shard_plan_sha256: str,
    bootstrap_draws: int,
    label: str,
) -> dict[str, float]:
    if set(record) != set(metric_names):
        raise ValueError(f"{label} paired metric coverage drift")
    means: dict[str, float] = {}
    for metric in metric_names:
        summary = _required_mapping(record, metric)
        if set(summary) != {
            "bootstrap_ci95",
            "example_count",
            "mean",
            "seed_sha256",
        }:
            raise ValueError(f"{label}.{metric} paired schema drift")
        if _required_int(summary, "example_count") != expected_count:
            raise ValueError(f"{label}.{metric} paired count drift")
        mean = _full_score_aggregate_finite_number(
            summary.get("mean"),
            f"{label}.{metric}.mean",
        )
        expected_mean = (
            method_summaries["vanilla_prefill"][metric][0]
            - method_summaries["baseline_prefill"][metric][0]
        )
        if not -1 <= mean <= 1 or not _full_score_aggregate_numbers_match(
            mean,
            expected_mean,
        ):
            raise ValueError(f"{label}.{metric} paired mean identity drift")
        expected_seed = _full_score_paired_bootstrap_seed_sha256(
            dataset=dataset_stratum,
            inventory_sha256=inventory_sha256,
            metric=metric,
            shard_plan_sha256=shard_plan_sha256,
        )
        if summary.get("seed_sha256") != expected_seed:
            raise ValueError(f"{label}.{metric} deterministic CI identity drift")
        ci = _required_mapping(summary, "bootstrap_ci95")
        if set(ci) != {"draws", "lower", "upper"}:
            raise ValueError(f"{label}.{metric} bootstrap CI schema drift")
        if _required_int(ci, "draws") != bootstrap_draws:
            raise ValueError(f"{label}.{metric} bootstrap draw identity drift")
        lower = _full_score_aggregate_finite_number(
            ci.get("lower"),
            f"{label}.{metric}.bootstrap_ci95.lower",
        )
        upper = _full_score_aggregate_finite_number(
            ci.get("upper"),
            f"{label}.{metric}.bootstrap_ci95.upper",
        )
        if not -1 <= lower <= upper <= 1:
            raise ValueError(f"{label}.{metric} bootstrap CI bounds drift")
        if expected_count == 1 and (
            not _full_score_aggregate_numbers_match(lower, mean)
            or not _full_score_aggregate_numbers_match(upper, mean)
        ):
            raise ValueError(f"{label}.{metric} singleton bootstrap CI drift")
        means[metric] = mean
    return means


def _validate_full_score_aggregate_niah_grid(
    value: Any,
    *,
    expected_count: int,
    metric_names: Sequence[str],
    dataset_method_summaries: Mapping[str, Mapping[str, tuple[float, float]]],
    dataset_paired_means: Mapping[str, float],
    inventory_sha256: str,
    shard_plan_sha256: str,
    bootstrap_draws: int,
) -> None:
    if expected_count != 1000:
        raise ValueError("full-score aggregate requires exactly 1,000 NIAH examples")
    if not isinstance(value, Mapping) or set(value) != set(NIAH_CELL_IDS):
        raise ValueError("full-score aggregate requires the exact nine-cell NIAH grid")
    observed_count = 0
    cell_metric_counts: dict[tuple[str, str], int] = defaultdict(int)
    cell_metric_sums: dict[tuple[str, str], float] = defaultdict(float)
    cell_paired_weighted_means: dict[str, float] = defaultdict(float)
    for cell_index, cell_id in enumerate(NIAH_CELL_IDS):
        cell = _required_mapping(value, cell_id)
        if set(cell) != {
            "example_count",
            "methods",
            "paired_vanilla_minus_baseline",
        }:
            raise ValueError("full-score aggregate NIAH cell schema drift")
        cell_count = _required_int(cell, "example_count")
        expected_cell_count = 112 if cell_index == 0 else 111
        if cell_count != expected_cell_count:
            raise ValueError("full-score aggregate NIAH cell count distribution drift")
        observed_count += cell_count
        methods = _required_mapping(cell, "methods")
        if set(methods) != set(FULL_SCORE_METHODS):
            raise ValueError("full-score aggregate NIAH method coverage drift")
        method_summaries: dict[str, dict[str, tuple[float, float]]] = {}
        for method in FULL_SCORE_METHODS:
            raw_metrics = _required_mapping(methods, method)
            if set(raw_metrics) != set(metric_names):
                raise ValueError("full-score aggregate NIAH metric coverage drift")
            method_summaries[method] = {}
            for metric in metric_names:
                mean, total = _validate_full_score_aggregate_metric_summary(
                    _required_mapping(raw_metrics, metric),
                    expected_count=cell_count,
                    label=f"niah_grid.{cell_id}.{method}.{metric}",
                )
                method_summaries[method][metric] = (mean, total)
                cell_metric_counts[(method, metric)] += cell_count
                cell_metric_sums[(method, metric)] += total
        paired_means = _validate_full_score_aggregate_paired_deltas(
            _required_mapping(cell, "paired_vanilla_minus_baseline"),
            dataset_stratum=f"niah/{cell_id}",
            expected_count=cell_count,
            metric_names=metric_names,
            method_summaries=method_summaries,
            inventory_sha256=inventory_sha256,
            shard_plan_sha256=shard_plan_sha256,
            bootstrap_draws=bootstrap_draws,
            label=f"niah_grid.{cell_id}.paired",
        )
        for metric, mean in paired_means.items():
            cell_paired_weighted_means[metric] += mean * cell_count
    if observed_count != expected_count:
        raise ValueError("full-score aggregate NIAH cell count coverage drift")
    for method in FULL_SCORE_METHODS:
        for metric in metric_names:
            if cell_metric_counts[(method, metric)] != expected_count or not (
                _full_score_aggregate_numbers_match(
                    cell_metric_sums[(method, metric)],
                    dataset_method_summaries[method][metric][1],
                )
            ):
                raise ValueError(
                    "full-score aggregate NIAH cell/dataset metric identity drift"
                )
    for metric in metric_names:
        if not _full_score_aggregate_numbers_match(
            cell_paired_weighted_means[metric] / expected_count,
            dataset_paired_means[metric],
        ):
            raise ValueError(
                "full-score aggregate NIAH paired cell/dataset identity drift"
            )


def _validate_full_score_aggregate_publication_lineage(
    lineage: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any],
    shard_plan: Mapping[str, Any],
) -> None:
    if set(lineage) != {
        "authorization_sha256",
        "batch_prefix",
        "evidence",
        "ledger_id",
        "ledger_path_sha256",
        "predecessor_prefix",
        "remote_consumer_authorizations",
        "terminal_prefix",
        "terminal_record_sha256",
        "wave_index",
    }:
        raise ValueError("full-score aggregate publication-lineage schema drift")
    authorization_sha256 = _require_sha256(
        lineage.get("authorization_sha256"),
        field_name="aggregate publication authorization_sha256",
    )
    ledger_path_sha256 = _require_sha256(
        lineage.get("ledger_path_sha256"),
        field_name="aggregate publication ledger_path_sha256",
    )
    terminal_record_sha256 = _require_sha256(
        lineage.get("terminal_record_sha256"),
        field_name="aggregate publication terminal_record_sha256",
    )
    ledger_id = _required_string(lineage, "ledger_id")
    predecessor_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(lineage, "predecessor_prefix")
    )
    batch_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(lineage, "batch_prefix")
    )
    terminal_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(lineage, "terminal_prefix")
    )
    if (
        {
            prefix.ledger_id
            for prefix in (
                predecessor_prefix,
                batch_prefix,
                terminal_prefix,
            )
        }
        != {ledger_id}
        or len(
            {
                prefix.cap_cluster_hours
                for prefix in (
                    predecessor_prefix,
                    batch_prefix,
                    terminal_prefix,
                )
            }
        )
        != 1
        or batch_prefix.reservation_count != predecessor_prefix.reservation_count + 1
        or batch_prefix.submission_receipt_count
        != predecessor_prefix.submission_receipt_count
        or batch_prefix.terminal_actual_count
        != predecessor_prefix.terminal_actual_count
        or terminal_prefix.reservation_count != batch_prefix.reservation_count
        or terminal_prefix.submission_receipt_count
        != batch_prefix.submission_receipt_count + 1
        or terminal_prefix.terminal_actual_count
        != batch_prefix.terminal_actual_count + 1
    ):
        raise ValueError("full-score aggregate terminal ledger lineage drift")
    expected_authorization_sha256 = _canonical_sha256(
        {
            "batch_prefix": batch_prefix.to_record(),
            "ledger_path_sha256": ledger_path_sha256,
            "terminal_prefix": terminal_prefix.to_record(),
            "terminal_record_sha256": terminal_record_sha256,
        }
    )
    if authorization_sha256 != expected_authorization_sha256:
        raise ValueError("full-score aggregate terminal authority closure drift")

    execution_plan_sha256 = _require_sha256(
        execution_plan.get("closed_record_sha256"),
        field_name="aggregate remote execution-plan SHA-256",
    )
    waves = execution_plan.get("waves")
    if not isinstance(waves, list) or not waves:
        raise ValueError("aggregate remote execution plan has no waves")
    expected_wave_by_shard: dict[str, int] = {}
    for wave_index, raw_wave in enumerate(waves):
        wave = _json_mapping(raw_wave, "aggregate remote execution wave")
        shard_ids = wave.get("shard_ids")
        raw_wave_shards = wave.get("shards")
        if not isinstance(shard_ids, list) or not isinstance(raw_wave_shards, list):
            raise ValueError("aggregate remote execution wave schema drift")
        normalized_shard_ids = [
            _require_nonempty(
                shard_id,
                "aggregate remote execution shard_id",
            )
            for shard_id in shard_ids
        ]
        wave_shard_ids = [
            _required_string(
                _json_mapping(raw_shard, "aggregate remote execution shard"),
                "shard_id",
            )
            for raw_shard in raw_wave_shards
        ]
        if (
            wave.get("wave_index") != wave_index
            or normalized_shard_ids != wave_shard_ids
            or not normalized_shard_ids
            or len(set(normalized_shard_ids)) != len(normalized_shard_ids)
            or any(
                shard_id in expected_wave_by_shard for shard_id in normalized_shard_ids
            )
        ):
            raise ValueError("aggregate remote execution wave/shard lineage drift")
        expected_wave_by_shard.update(
            {shard_id: wave_index for shard_id in normalized_shard_ids}
        )
    plan_shards = shard_plan.get("shards")
    if not isinstance(plan_shards, list):
        raise ValueError("aggregate remote shard plan has no shards")
    expected_shard_ids = {
        _required_string(
            _json_mapping(raw_shard, "aggregate remote planned shard"),
            "shard_id",
        )
        for raw_shard in plan_shards
    }
    if set(expected_wave_by_shard) != expected_shard_ids:
        raise ValueError("aggregate remote execution-plan shard lineage drift")
    final_wave_index = len(waves) - 1
    if _required_int(lineage, "wave_index") != final_wave_index:
        raise ValueError("full-score aggregate final-wave lineage drift")

    raw_authorizations = lineage.get("remote_consumer_authorizations")
    if not isinstance(raw_authorizations, list) or len(raw_authorizations) != len(
        waves
    ):
        raise ValueError("aggregate remote execution-plan authority coverage drift")
    authorization_by_wave: dict[int, Mapping[str, Any]] = {}
    for wave_index, raw_authorization in enumerate(raw_authorizations):
        remote_authorization = _json_mapping(
            raw_authorization,
            "aggregate remote consumer authorization",
        )
        if set(remote_authorization) != {
            "authorization_record_sha256",
            "coordinator_run_id",
            "execution_plan_sha256",
            "phase_terminal_record_sha256",
            "runs_get_receipt_record_sha256",
            "wave_index",
        }:
            raise ValueError("aggregate remote authorization schema drift")
        if (
            _required_int(remote_authorization, "wave_index") != wave_index
            or remote_authorization.get("execution_plan_sha256")
            != execution_plan_sha256
            or _required_run_id(
                remote_authorization.get("coordinator_run_id"),
                "aggregate remote coordinator_run_id",
            )
            != remote_authorization.get("coordinator_run_id")
        ):
            raise ValueError("aggregate remote execution-plan authority drift")
        for field_name in (
            "authorization_record_sha256",
            "phase_terminal_record_sha256",
            "runs_get_receipt_record_sha256",
        ):
            _require_sha256(
                remote_authorization.get(field_name),
                field_name=f"aggregate remote {field_name}",
            )
        authorization_by_wave[wave_index] = remote_authorization
    for field_name in (
        "authorization_record_sha256",
        "coordinator_run_id",
        "phase_terminal_record_sha256",
        "runs_get_receipt_record_sha256",
    ):
        if len(
            {
                authorization.get(field_name)
                for authorization in authorization_by_wave.values()
            }
        ) != len(waves):
            raise ValueError("aggregate remote authorization identity reuse")
    if (
        authorization_by_wave[final_wave_index].get("phase_terminal_record_sha256")
        != terminal_record_sha256
    ):
        raise ValueError("aggregate remote/final terminal lineage drift")

    raw_evidence = lineage.get("evidence")
    if not isinstance(raw_evidence, list) or len(raw_evidence) != len(
        expected_shard_ids
    ):
        raise ValueError("aggregate remote evidence shard coverage drift")
    observed_shards: list[str] = []
    durable_roots: set[str] = set()
    evidence_identities: dict[str, set[str]] = {
        field_name: set()
        for field_name in (
            "deletion_file_sha256",
            "deletion_record_sha256",
            "deletion_uri",
            "evidence_file_sha256",
            "evidence_record_sha256",
            "evidence_uri",
        )
    }
    from document_kv_cache.full_score_remote_control import (
        _consumer_evidence_artifact_uri,
    )

    for raw_binding in raw_evidence:
        binding = _json_mapping(raw_binding, "aggregate remote evidence binding")
        if set(binding) != {
            "authorization_record_sha256",
            "deletion_file_sha256",
            "deletion_record_sha256",
            "deletion_uri",
            "evidence_file_sha256",
            "evidence_record_sha256",
            "evidence_uri",
            "shard_id",
            "wave_index",
        }:
            raise ValueError("aggregate remote evidence binding schema drift")
        shard_id = _required_string(binding, "shard_id")
        binding_wave_index = binding.get("wave_index")
        if (
            type(binding_wave_index) is not int
            or expected_wave_by_shard.get(shard_id) != binding_wave_index
            or shard_id in observed_shards
            or binding.get("authorization_record_sha256")
            != authorization_by_wave[binding_wave_index].get(
                "authorization_record_sha256"
            )
        ):
            raise ValueError("aggregate remote execution-plan evidence drift")
        for field_name in (
            "authorization_record_sha256",
            "deletion_file_sha256",
            "deletion_record_sha256",
            "evidence_file_sha256",
            "evidence_record_sha256",
        ):
            _require_sha256(
                binding.get(field_name),
                field_name=f"aggregate evidence {field_name}",
            )
        for field_name, identities in evidence_identities.items():
            identity = _required_string(binding, field_name)
            if identity in identities:
                raise ValueError("aggregate remote evidence identity reuse")
            identities.add(identity)
        evidence_uri = _required_string(binding, "evidence_uri")
        evidence_suffix = (
            f"/evidence/wave-{binding_wave_index:03d}/{shard_id}/evidence.json"
        )
        if not evidence_uri.endswith(evidence_suffix):
            raise ValueError("aggregate remote evidence URI lineage drift")
        durable_root = evidence_uri[: -len(evidence_suffix)]
        if evidence_uri != _consumer_evidence_artifact_uri(
            durable_root,
            wave_index=binding_wave_index,
            shard_id=shard_id,
            filename="evidence.json",
        ) or binding.get("deletion_uri") != _consumer_evidence_artifact_uri(
            durable_root,
            wave_index=binding_wave_index,
            shard_id=shard_id,
            filename="deletion-attestation.json",
        ):
            raise ValueError("aggregate remote evidence URI lineage drift")
        durable_roots.add(durable_root)
        observed_shards.append(shard_id)
    if observed_shards != sorted(expected_shard_ids) or len(durable_roots) != 1:
        raise ValueError("aggregate remote evidence ordered coverage drift")


def _full_score_aggregate_finite_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not float("-inf") < float(value) < float("inf")
    ):
        raise ValueError(f"{field_name} must be finite numeric data")
    return float(value)


def _full_score_aggregate_numbers_match(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12 * max(1.0, abs(left), abs(right))


def _paired_delta_summaries(
    per_example: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    dataset: str,
    inventory_sha256: str,
    shard_plan_sha256: str,
) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = defaultdict(list)
    for key in sorted(per_example):
        pair = per_example[key]
        methods = cast(Mapping[str, Mapping[str, Any]], pair["methods"])
        baseline = cast(
            Mapping[str, Any], methods["baseline_prefill"]["quality_scores"]
        )
        vanilla = cast(Mapping[str, Any], methods["vanilla_prefill"]["quality_scores"])
        if set(baseline) != set(vanilla):
            raise ValueError("paired arms expose different metric sets")
        for metric in sorted(baseline):
            metric_values[metric].append(
                float(vanilla[metric]) - float(baseline[metric])
            )
    summaries = {}
    for metric, values in sorted(metric_values.items()):
        seed_sha256 = _full_score_paired_bootstrap_seed_sha256(
            dataset=dataset,
            inventory_sha256=inventory_sha256,
            metric=metric,
            shard_plan_sha256=shard_plan_sha256,
        )
        summaries[metric] = {
            "bootstrap_ci95": _paired_bootstrap_ci(
                values,
                seed_sha256=seed_sha256,
                draws=PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS,
            ),
            "example_count": len(values),
            "mean": sum(values) / len(values),
            "seed_sha256": seed_sha256,
        }
    return summaries


def _full_score_paired_bootstrap_seed_sha256(
    *,
    dataset: str,
    inventory_sha256: str,
    metric: str,
    shard_plan_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "dataset_stratum": dataset,
            "domain": "cachet.full_score.paired_bootstrap.seed.v1",
            "inventory_sha256": inventory_sha256,
            "metric": metric,
            "shard_plan_sha256": shard_plan_sha256,
        }
    )


def _paired_bootstrap_ci(
    values: Sequence[float],
    *,
    seed_sha256: str,
    draws: int,
) -> dict[str, float | int]:
    """Ordinary paired-example bootstrap using a recorded deterministic MT seed."""

    _require_publication_rng_runtime()
    _require_sha256(seed_sha256, field_name="bootstrap seed_sha256")
    if type(draws) is not int or draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    observed = tuple(float(value) for value in values)
    if not observed:
        raise ValueError("paired bootstrap requires at least one example")
    rng = random.Random(int(seed_sha256, 16))
    sample_size = len(observed)
    means = [
        sum(rng.choices(observed, k=sample_size)) / sample_size
        for _draw in range(draws)
    ]
    means.sort()
    return {
        "draws": draws,
        "lower": _type7_quantile(means, 0.025),
        "upper": _type7_quantile(means, 0.975),
    }


def _type7_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values or not 0 <= probability <= 1:
        raise ValueError("quantile inputs are invalid")
    position = (len(sorted_values) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return (
        float(sorted_values[lower_index]) * (1 - fraction)
        + float(sorted_values[upper_index]) * fraction
    )


def _require_publication_rng_runtime() -> None:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("publication bootstrap RNG requires CPython 3.11")


def run_full_score_worker(
    worker_payload_path: str | Path,
    *,
    expected_worker_payload_sha256: str,
    producer_phase_completion_path: str | Path | None = None,
    expected_producer_phase_completion_sha256: str | None = None,
    tokenizer: MainLatencyTokenizer | None = None,
    command_runner: FullScoreCommandRunner | None = None,
) -> tuple[dict[str, Any], ...]:
    """Execute all assigned shards sequentially in one persistent task."""

    payload_path = _cluster_path(worker_payload_path)
    _require_regular_file_no_follow(payload_path, "full-score worker payload")
    raw_payload = payload_path.read_bytes()
    payload = _json_object(raw_payload, "worker payload")
    expected_payload_sha = _require_sha256(
        expected_worker_payload_sha256,
        field_name="expected_worker_payload_sha256",
    )
    if sha256(raw_payload).hexdigest() != expected_payload_sha:
        raise ValueError("worker payload file SHA-256 drift")
    if payload.get("closed_record_sha256") != _closed_record_sha256(payload):
        raise ValueError("worker payload closure drift")
    inventory_binding = _required_mapping(payload, "inventory")
    plan_binding = _required_mapping(payload, "shard_plan")
    execution_binding = _required_mapping(payload, "execution_plan")
    inventory_record = _read_bound_json(
        _required_string(inventory_binding, "uri"),
        _required_string(inventory_binding, "closed_record_sha256"),
        closure_digest=True,
    )
    inventory = full_score_inventory_from_record(inventory_record)
    plan_record = _read_bound_json(
        _required_string(plan_binding, "uri"),
        _required_string(plan_binding, "closed_record_sha256"),
        closure_digest=True,
    )
    execution_record = _read_bound_json(
        _required_string(execution_binding, "uri"),
        _required_string(execution_binding, "closed_record_sha256"),
        closure_digest=True,
    )
    validate_full_score_worker_payload(
        payload,
        inventory=inventory,
        shard_plan=plan_record,
        execution_plan=execution_record,
    )
    _validate_publication_full_score_inputs(
        inventory,
        plan_record,
        execution_record,
    )
    runtime = _runtime_from_record(_required_mapping(payload, "runtime"))
    bootstrap_artifacts = _required_mapping(payload, "bootstrap_artifacts")
    _validate_bootstrap_artifact_binding(bootstrap_artifacts, runtime=runtime)
    _verify_bound_gpu_qualification(
        _required_mapping(payload, "gpu_qualification"),
        runtime=runtime,
    )
    expected_runtime_identity = _required_string(
        bootstrap_artifacts,
        "locked_runtime_identity_sha256",
    )
    if os.environ.get("CACHET_FULL_SCORE_LOCKED_RUNTIME") != expected_runtime_identity:
        raise RuntimeError(
            "worker is not executing in its payload-bound locked runtime"
        )
    if os.path.realpath(sys.executable) != os.path.realpath(runtime.python_executable):
        raise RuntimeError("worker Python does not match the payload-bound runtime")
    _verify_runtime_contract(runtime)
    runner = command_runner or _subprocess_command_runner
    runtime_verification_path = (
        _cluster_path(_required_string(payload, "durable_output_root"))
        / "runtime-verification"
        / f"wave-{_required_int(payload, 'wave_index'):03d}"
        / f"{_required_string(payload, 'role')}-{_required_int(payload, 'worker_index'):02d}"
        / "runtime-lock-verification.json"
    )
    runtime_verification = _run_runtime_verifier(
        runtime,
        bootstrap_artifacts,
        runtime_verification_path,
        runner=runner,
    )
    role = _required_string(payload, "role")
    if role == "producer":
        if (
            producer_phase_completion_path is not None
            or expected_producer_phase_completion_sha256 is not None
        ):
            raise ValueError("producer worker cannot consume phase-completion evidence")
        resolved_tokenizer = tokenizer or load_main_latency_tokenizer()
        source_records = _load_and_validate_worker_source_records(
            payload,
            inventory=inventory,
            tokenizer=resolved_tokenizer,
        )
        return _run_producer_worker(
            payload,
            inventory=inventory,
            shard_plan=plan_record,
            execution_plan=execution_record,
            source_records=source_records,
            runtime=runtime,
            runtime_verification=runtime_verification,
        )
    if role == "consumer":
        completion_path = _require_nonempty(
            None
            if producer_phase_completion_path is None
            else str(producer_phase_completion_path),
            "producer_phase_completion_path",
        )
        completion_sha256 = _require_sha256(
            expected_producer_phase_completion_sha256,
            field_name="expected_producer_phase_completion_sha256",
        )
        producer_completion = _read_bound_json(
            completion_path,
            completion_sha256,
            closure_digest=True,
        )
        _validate_producer_phase_completion(
            producer_completion,
            execution_plan=execution_record,
            expected_wave_index=_required_int(payload, "wave_index"),
        )
        return _run_consumer_worker(
            payload,
            inventory=inventory,
            shard_plan=plan_record,
            execution_plan=execution_record,
            producer_completion=producer_completion,
            runtime=runtime,
            runtime_verification=runtime_verification,
            runner=runner,
        )
    raise ValueError("worker role must be producer or consumer")


def write_full_score_worker_payloads(
    payloads: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Write canonical worker manifests without overwriting reviewed inputs."""

    destination = Path(output_dir)
    _require_no_symlink_ancestors(
        destination,
        label="worker payload output directory",
        include_leaf=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for payload in payloads:
        worker_index = _required_int(payload, "worker_index")
        wave_index = _required_int(payload, "wave_index")
        role = _required_string(payload, "role")
        path = destination / (
            f"full-score-wave-{wave_index:03d}-{role}-{worker_index:02d}.json"
        )
        if path.exists():
            raise FileExistsError(f"worker payload already exists: {path}")
        _exclusive_write_bytes(path, _canonical_pretty_json_bytes(payload))
        paths.append(path)
    return tuple(paths)


def write_full_score_runner_script(path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"full-score runner already exists: {destination}")
    _exclusive_write_bytes(destination, FULL_SCORE_RUNNER_SCRIPT.encode("utf-8"))


def write_full_score_producer_phase_completion_record(
    record: Mapping[str, Any],
    path: str | Path,
) -> None:
    if record.get("record_type") != FULL_SCORE_PRODUCER_PHASE_COMPLETION_RECORD_TYPE:
        raise ValueError("producer-phase completion record_type drift")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("producer-phase completion closure drift")
    _exclusive_write_bytes(Path(path), _canonical_pretty_json_bytes(record))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate, dry-run, or execute a closed full-score worker."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry_run = subparsers.add_parser("dry-run-worker")
    dry_run.add_argument("--worker-payload-json", required=True)
    dry_run.add_argument("--output-json")
    worker = subparsers.add_parser("run-worker")
    worker.add_argument("--worker-payload-json", required=True)
    worker.add_argument("--expected-worker-payload-sha256", required=True)
    worker.add_argument("--producer-phase-completion-json")
    worker.add_argument("--expected-producer-phase-completion-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "dry-run-worker":
            payload = _json_object(
                _cluster_path(args.worker_payload_json).read_bytes(),
                "worker payload",
            )
            record = render_full_score_worker_command_plan(payload)
            rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
            if args.output_json:
                _exclusive_write_bytes(Path(args.output_json), rendered.encode("utf-8"))
            else:
                print(rendered, end="")
        else:
            records = run_full_score_worker(
                args.worker_payload_json,
                expected_worker_payload_sha256=(args.expected_worker_payload_sha256),
                producer_phase_completion_path=(args.producer_phase_completion_json),
                expected_producer_phase_completion_sha256=(
                    args.expected_producer_phase_completion_sha256
                ),
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "record_sha256": [
                            record["closed_record_sha256"] for record in records
                        ],
                        "record_types": sorted(
                            {cast(str, record["record_type"]) for record in records}
                        ),
                        "shards": len(records),
                    },
                    sort_keys=True,
                )
            )
    except Exception as exc:
        print(
            json.dumps(
                {"error": str(exc), "error_type": type(exc).__name__, "ok": False},
                sort_keys=True,
            )
        )
        return 1
    return 0


def _run_producer_worker(
    payload: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    source_records: Mapping[tuple[str, str], Mapping[str, Any]],
    runtime: FullScoreRuntimeConfig,
    runtime_verification: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Load the generator once and emit all assigned closed ready shards."""

    env = _worker_environment(runtime)
    os.environ.update(
        {key: value for key, value in env.items() if key.startswith("CACHET_")}
    )
    producer_hardware = _observe_producer_hardware()
    generator = load_benchmark_kv_chunk_generator(runtime.generator_factory)
    if getattr(generator, "pre_rope", None) is not True:
        raise ValueError("full-score generator must emit pre-RoPE KV")
    layout = layout_for_model(
        runtime.model_id,
        dtype=FULL_SCORE_KV_DTYPE,
        pre_rope=True,
        rope_theta=QWEN3_4B_ROPE_THETA,
        rope_rotary_dim=QWEN3_4B_ROPE_ROTARY_DIM,
        shares_kv_storage=False,
        storage_layout="separate_key_value",
    )
    ready_records: list[dict[str, Any]] = []
    try:
        for shard in cast(list[Mapping[str, Any]], payload["shards"]):
            ready_records.append(
                _produce_ready_shard(
                    payload,
                    shard,
                    inventory=inventory,
                    shard_plan=shard_plan,
                    execution_plan=execution_plan,
                    source_records=source_records,
                    runtime=runtime,
                    runtime_verification=runtime_verification,
                    generator=generator,
                    layout=layout,
                    producer_hardware=producer_hardware,
                )
            )
    finally:
        del generator
        gc.collect()
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.cuda.empty_cache()
    return tuple(ready_records)


def _produce_ready_shard(
    payload: Mapping[str, Any],
    shard: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    source_records: Mapping[tuple[str, str], Mapping[str, Any]],
    runtime: FullScoreRuntimeConfig,
    runtime_verification: Mapping[str, Any],
    generator: Any,
    layout: Any,
    producer_hardware: Mapping[str, Any],
) -> dict[str, Any]:
    worker_index = _required_int(payload, "worker_index")
    wave_index = _required_int(payload, "wave_index")
    shard_id = _required_string(shard, "shard_id")
    local_dir = (
        _cluster_path(_required_string(payload, "ephemeral_root"))
        / f"producer-wave-{wave_index:03d}-{worker_index:02d}"
        / shard_id
    )
    ready_dir = _ready_shard_dir(payload, shard_id)
    _require_no_symlink_ancestors(
        local_dir,
        label="producer local shard path",
        include_leaf=True,
    )
    _require_no_symlink_ancestors(
        ready_dir,
        label="producer ready-shard path",
        include_leaf=True,
    )
    if ready_dir.exists():
        existing_ready = _validate_ready_shard(
            ready_dir,
            shard=shard,
            payload=payload,
            inventory=inventory,
            shard_plan=shard_plan,
            execution_plan=execution_plan,
        )
        if existing_ready.get(
            "runtime_verification"
        ) != _validate_runtime_verification_binding(runtime_verification):
            raise ValueError("existing ready-shard runtime verification drift")
        return existing_ready
    if local_dir.exists():
        _delete_directory_tree_no_follow(
            local_dir,
            label="producer local shard cleanup",
        )
    local_dir.mkdir(parents=True)
    lifecycle = FullScoreShardLifecycle("producer")
    inputs, _examples = _write_shard_inputs(
        shard, source_records=source_records, output_dir=local_dir / "inputs"
    )
    for dataset, input_path in sorted(inputs.items()):
        dataset_output = local_dir / "q8-kv" / dataset
        manifest_path = local_dir / "manifests" / f"{dataset}.json"
        result = generate_benchmark_handoff_bundles(
            input_path,
            output_dir=dataset_output,
            generator=generator,
            layout=layout,
            dataset=dataset,
            backend="vllm",
            manifest_json=manifest_path,
            segmented=True,
            segment_per_document=True,
            cache_method="vanilla_prefill",
            model_id=runtime.model_id,
            model_revision=runtime.model_revision,
            tokenizer_id=runtime.tokenizer_id,
            tokenizer_revision=runtime.tokenizer_revision,
            generator_family="transformers",
            generator_version=runtime.generator_version,
            align_bytes=4096,
            require_artifact_contract=True,
        )
        enriched_path = local_dir / "enriched" / f"{dataset}.jsonl"
        enrich_benchmark_jsonl_with_handoffs(
            input_path,
            manifest_path,
            enriched_path,
            dataset=dataset,
            arm_id=FULL_SCORE_VANILLA_ARM_ID,
            overwrite=True,
        )
        if not result.manifest.entries:
            raise RuntimeError("Q8 generation produced an empty manifest")
    lifecycle.advance("generate_q8_kv")
    actual_local_bytes = sum(
        path.stat().st_size for path in local_dir.rglob("*") if path.is_file()
    )
    upper_bound = _wave_shard_upper_bound(payload, shard_id)
    if actual_local_bytes > upper_bound:
        raise RuntimeError("ready shard exceeds its conservative backlog byte bound")
    ready_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = ready_dir.with_name(
        f".{ready_dir.name}.pending-{payload['closed_record_sha256']}"
    )
    _require_no_symlink_ancestors(
        staging_dir,
        label="producer ready-shard staging path",
        include_leaf=True,
    )
    if staging_dir.exists():
        _delete_directory_tree_no_follow(
            staging_dir,
            label="producer ready-shard staging cleanup",
        )
    shutil.copytree(local_dir, staging_dir, symlinks=False)
    _rewrite_json_tree_paths(staging_dir, old_root=local_dir, new_root=ready_dir)
    _fsync_file_tree(staging_dir)
    files = _closed_file_tree(staging_dir)
    ready_bytes = sum(cast(int, item["byte_count"]) for item in files)
    if ready_bytes > upper_bound:
        raise RuntimeError("rewritten ready shard exceeds its backlog byte bound")
    ready: dict[str, Any] = {
        "closed_record_sha256": "",
        "execution_plan_sha256": execution_plan.get("closed_record_sha256"),
        "files": files,
        "files_sha256": _canonical_sha256(files),
        "generator_artifact_contract": _generator_artifact_contract_record(runtime),
        "inventory_sha256": inventory.inventory_sha256,
        "lifecycle": [*lifecycle.events, "commit_ready_shard"],
        "producer_hardware": dict(producer_hardware),
        "ready_bytes": ready_bytes,
        "ready_bytes_upper_bound": upper_bound,
        "record_type": FULL_SCORE_READY_SHARD_RECORD_TYPE,
        "runtime_verification": _validate_runtime_verification_binding(
            runtime_verification
        ),
        "schema_version": FULL_SCORE_READY_SHARD_SCHEMA_VERSION,
        "shard_id": shard_id,
        "shard_items_sha256": shard.get("items_sha256"),
        "shard_plan_sha256": shard_plan.get("closed_record_sha256"),
        "wave_index": wave_index,
        "worker_index": worker_index,
    }
    ready["closed_record_sha256"] = _closed_record_sha256(ready)
    _exclusive_write_bytes(
        staging_dir / "ready-record.json",
        _canonical_pretty_json_bytes(ready),
    )
    _rename_directory_no_follow(staging_dir, ready_dir)
    reread = _json_object((ready_dir / "ready-record.json").read_bytes(), "ready shard")
    if reread != ready:
        raise RuntimeError("ready-shard durable reread mismatch")
    lifecycle.advance("commit_ready_shard")
    _delete_directory_tree_no_follow(
        local_dir,
        label="producer local shard cleanup",
    )
    return ready


def _run_consumer_worker(
    payload: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    producer_completion: Mapping[str, Any],
    runtime: FullScoreRuntimeConfig,
    runtime_verification: Mapping[str, Any],
    runner: FullScoreCommandRunner,
) -> tuple[dict[str, Any], ...]:
    """Boot vLLM once and consume all assigned ready shards sequentially."""

    recovered: dict[str, dict[str, Any]] = {}
    pending_shards: list[Mapping[str, Any]] = []
    for shard in cast(list[Mapping[str, Any]], payload["shards"]):
        existing = _recover_committed_consumer_shard(
            payload,
            shard,
            inventory=inventory,
            shard_plan=shard_plan,
            execution_plan=execution_plan,
            runtime_verification=runtime_verification,
        )
        if existing is None:
            pending_shards.append(shard)
        else:
            recovered[_required_string(shard, "shard_id")] = existing
    if not pending_shards:
        return tuple(
            recovered[_required_string(shard, "shard_id")]
            for shard in cast(list[Mapping[str, Any]], payload["shards"])
        )
    _wait_for_complete_ready_wave(payload, shards=pending_shards)
    worker_index = _required_int(payload, "worker_index")
    wave_index = _required_int(payload, "wave_index")
    local_root = (
        _cluster_path(_required_string(payload, "ephemeral_root"))
        / f"consumer-wave-{wave_index:03d}-{worker_index:02d}"
    )
    _require_no_symlink_ancestors(
        local_root,
        label="consumer local worker path",
        include_leaf=True,
    )
    if local_root.exists():
        _delete_directory_tree_no_follow(
            local_root,
            label="consumer local worker cleanup",
        )
    local_root.mkdir(parents=True)
    server_log = local_root / "vllm-server.log"
    connector_telemetry = local_root / "connector-telemetry.jsonl"
    transfer_config = json.loads(json.dumps(runtime.kv_transfer_config, sort_keys=True))
    extra_config = transfer_config.get("kv_connector_extra_config")
    if not isinstance(extra_config, dict):
        raise ValueError(
            "kv_transfer_config.kv_connector_extra_config must be an object"
        )
    extra_config["document_kv.telemetry_jsonl"] = str(connector_telemetry)
    log_handle = server_log.open("w", encoding="utf-8")
    server = subprocess.Popen(
        _vllm_server_command(runtime, transfer_config=transfer_config),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=_worker_environment(runtime),
    )
    evidence: list[dict[str, Any]] = []
    try:
        _wait_for_server(runtime, server, server_log)
        for shard in pending_shards:
            evidence.append(
                _consume_ready_shard(
                    payload,
                    shard,
                    inventory=inventory,
                    shard_plan=shard_plan,
                    execution_plan=execution_plan,
                    producer_completion=producer_completion,
                    runtime=runtime,
                    runtime_verification=runtime_verification,
                    runner=runner,
                    server=server,
                    server_log=server_log,
                    connector_telemetry=connector_telemetry,
                    log_handle=log_handle,
                    local_root=local_root,
                )
            )
    finally:
        _terminate_process(server)
        log_handle.close()
    by_shard = {
        _required_string(record, "shard_id"): record
        for record in [*recovered.values(), *evidence]
    }
    return tuple(
        by_shard[_required_string(shard, "shard_id")]
        for shard in cast(list[Mapping[str, Any]], payload["shards"])
    )


def _consume_ready_shard(
    payload: Mapping[str, Any],
    shard: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    producer_completion: Mapping[str, Any],
    runtime: FullScoreRuntimeConfig,
    runtime_verification: Mapping[str, Any],
    runner: FullScoreCommandRunner,
    server: subprocess.Popen[Any],
    server_log: Path,
    connector_telemetry: Path,
    log_handle: Any,
    local_root: Path,
) -> dict[str, Any]:
    worker_index = _required_int(payload, "worker_index")
    wave_index = _required_int(payload, "wave_index")
    shard_id = _required_string(shard, "shard_id")
    ready_dir = _ready_shard_dir(payload, shard_id)
    durable_dir = (
        _cluster_path(_required_string(payload, "durable_output_root"))
        / "evidence"
        / f"wave-{wave_index:03d}"
        / shard_id
    )
    _require_no_symlink_ancestors(
        durable_dir,
        label="consumer durable evidence path",
        include_leaf=True,
    )
    if durable_dir.exists():
        raise FileExistsError(f"durable shard evidence already exists: {durable_dir}")
    ready = _validate_ready_shard(
        ready_dir,
        shard=shard,
        payload=payload,
        inventory=inventory,
        shard_plan=shard_plan,
        execution_plan=execution_plan,
    )
    if ready.get("runtime_verification") != _validate_runtime_verification_binding(
        runtime_verification
    ):
        raise ValueError("consumer runtime verification differs from ready shard")
    completion_ready = next(
        (
            item
            for item in cast(
                list[Mapping[str, Any]], producer_completion["ready_shards"]
            )
            if item.get("shard_id") == shard_id
        ),
        None,
    )
    if completion_ready is None or completion_ready.get(
        "ready_record_sha256"
    ) != ready.get("closed_record_sha256"):
        raise ValueError("consumer ready shard differs from producer completion")
    lifecycle = FullScoreShardLifecycle("consumer")
    lifecycle.advance("verify_ready_shard")
    shard_local = local_root / shard_id
    _require_no_symlink_ancestors(
        shard_local,
        label="consumer local shard path",
        include_leaf=True,
    )
    shard_local.mkdir()
    datasets = sorted(
        {
            cast(str, item["dataset"])
            for item in cast(list[Mapping[str, Any]], shard["items"])
        }
    )
    inputs = {
        dataset: ready_dir / "inputs" / f"{dataset}.jsonl" for dataset in datasets
    }
    enriched = {
        dataset: ready_dir / "enriched" / f"{dataset}.jsonl" for dataset in datasets
    }
    examples = _load_ready_examples(inputs)
    vanilla_examples = _load_ready_examples(enriched)
    baseline_path = shard_local / "baseline.json"
    vanilla_path = shard_local / "vanilla.json"
    telemetry_path = shard_local / "runtime-telemetry.json"
    connector_start_offset = (
        connector_telemetry.stat().st_size if connector_telemetry.exists() else 0
    )
    telemetry = RuntimeTelemetrySampler(
        telemetry_path,
        process_pid=server.pid,
        interval_seconds=runtime.telemetry_interval_seconds,
    ).start()
    method_wall_seconds: dict[str, float] = {}
    try:
        baseline_started_ns = time.monotonic_ns()
        runner(
            _benchmark_command(
                runtime,
                method="baseline_prefill",
                shard_id=shard_id,
                dataset_paths=inputs,
                output_json=baseline_path,
            ),
            env=_worker_environment(runtime),
        )
        method_wall_seconds["baseline_prefill"] = (
            time.monotonic_ns() - baseline_started_ns
        ) / 1_000_000_000
        lifecycle.advance("baseline_inference")
        vanilla_started_ns = time.monotonic_ns()
        runner(
            _benchmark_command(
                runtime,
                method="vanilla_prefill",
                shard_id=shard_id,
                dataset_paths=enriched,
                output_json=vanilla_path,
            ),
            env=_worker_environment(runtime),
        )
        method_wall_seconds["vanilla_prefill"] = (
            time.monotonic_ns() - vanilla_started_ns
        ) / 1_000_000_000
        lifecycle.advance("vanilla_inference")
    finally:
        telemetry.stop()
    paired = validate_paired_full_score_outputs(
        _json_object(baseline_path.read_bytes(), "baseline output"),
        _json_object(vanilla_path.read_bytes(), "Vanilla output"),
        shard=shard,
        examples=examples,
        vanilla_examples=vanilla_examples,
    )
    lifecycle.advance("validate_paired_outputs")
    log_handle.flush()
    os.fsync(log_handle.fileno())
    if not connector_telemetry.exists():
        raise ValueError("connector telemetry is required before ready-shard deletion")
    with connector_telemetry.open("rb") as handle:
        handle.seek(connector_start_offset)
        shard_connector_bytes = handle.read()
    if not shard_connector_bytes or not shard_connector_bytes.endswith(b"\n"):
        raise ValueError("shard connector telemetry is empty or unterminated")
    shard_connector_telemetry = shard_local / "connector-telemetry.jsonl"
    _exclusive_write_bytes(shard_connector_telemetry, shard_connector_bytes)
    connector_proof = build_full_score_connector_proof(
        shard_connector_telemetry,
        paired_examples=paired,
        shard=shard,
    )
    staging_dir = durable_dir.with_name(
        f".{durable_dir.name}.pending-{payload['closed_record_sha256']}"
    )
    _require_no_symlink_ancestors(
        staging_dir,
        label="consumer evidence staging path",
        include_leaf=True,
    )
    if staging_dir.exists():
        _delete_directory_tree_no_follow(
            staging_dir,
            label="consumer evidence staging cleanup",
        )
    staging_dir.mkdir(parents=True, exist_ok=False)
    preserved = {}
    preserved_sources = [
        ("baseline_raw_output", baseline_path, "baseline-raw-output.json"),
        ("vanilla_raw_output", vanilla_path, "vanilla-raw-output.json"),
        ("runtime_telemetry", telemetry_path, "runtime-telemetry.json"),
        ("server_log", server_log, "vllm-server.log"),
        (
            "connector_telemetry",
            shard_connector_telemetry,
            "connector-telemetry.jsonl",
        ),
        (
            "ready_shard_manifest",
            ready_dir / "ready-record.json",
            "ready-shard-manifest.json",
        ),
    ]
    preserved_sources.extend(
        (f"input_{dataset}", path, f"input-{dataset}.jsonl")
        for dataset, path in sorted(inputs.items())
    )
    preserved_sources.extend(
        (f"enriched_{dataset}", path, f"enriched-{dataset}.jsonl")
        for dataset, path in sorted(enriched.items())
    )
    preserved_sources.extend(
        (
            f"handoff_manifest_{dataset}",
            ready_dir / "manifests" / f"{dataset}.json",
            f"handoff-manifest-{dataset}.json",
        )
        for dataset in datasets
    )
    for name, source, filename in preserved_sources:
        destination = staging_dir / filename
        _durable_copy(source, destination)
        preserved[name] = _file_record(destination, relative_to=staging_dir)
    evidence: dict[str, Any] = {
        "authorization_scope": payload.get("authorization_scope"),
        "closed_record_sha256": "",
        "connector_proof": connector_proof,
        "durable_evidence_committed": True,
        "execution_plan_sha256": execution_plan.get("closed_record_sha256"),
        "inventory_sha256": inventory.inventory_sha256,
        "lifecycle_before_delete": list(lifecycle.events),
        "method_wall_clock": "time.monotonic_ns",
        "method_wall_seconds": method_wall_seconds,
        "paired_examples": list(paired),
        "preserved_files": preserved,
        "protocol": _full_score_protocol_record(),
        "ready_shard_sha256": ready.get("closed_record_sha256"),
        "record_type": FULL_SCORE_SHARD_EVIDENCE_RECORD_TYPE,
        "runtime_verification": _validate_runtime_verification_binding(
            runtime_verification
        ),
        "schema_version": FULL_SCORE_SHARD_EVIDENCE_SCHEMA_VERSION,
        "scorers": _scorer_contract_record(),
        "shard_id": shard_id,
        "shard_items_sha256": shard.get("items_sha256"),
        "shard_plan_sha256": shard_plan.get("closed_record_sha256"),
        "wave_index": wave_index,
        "worker_index": worker_index,
    }
    evidence["closed_record_sha256"] = _closed_record_sha256(evidence)
    evidence_path = staging_dir / "evidence.json"
    _exclusive_write_bytes(evidence_path, _canonical_pretty_json_bytes(evidence))
    if _json_object(evidence_path.read_bytes(), "committed evidence") != evidence:
        raise RuntimeError("durable evidence reread mismatch")
    durable_dir.parent.mkdir(parents=True, exist_ok=True)
    _rename_directory_no_follow(staging_dir, durable_dir)
    lifecycle.advance("commit_durable_evidence")
    if lifecycle.state != "evidence_committed":
        raise RuntimeError("ready-shard deletion requires committed evidence")
    _delete_ready_shard_tree(ready_dir)
    lifecycle.advance("delete_ephemeral_q8_kv")
    _write_or_validate_deletion_attestation(
        durable_dir,
        payload=payload,
        evidence=evidence,
        execution_plan=execution_plan,
        lifecycle=list(lifecycle.events),
    )
    return evidence


def _write_or_validate_deletion_attestation(
    durable_dir: Path,
    *,
    payload: Mapping[str, Any],
    evidence: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    lifecycle: Sequence[str],
) -> dict[str, Any]:
    deletion: dict[str, Any] = {
        "authorization_scope": payload.get("authorization_scope"),
        "closed_record_sha256": "",
        "evidence_closed_record_sha256": evidence.get("closed_record_sha256"),
        "execution_plan_sha256": execution_plan.get("closed_record_sha256"),
        "lifecycle": list(lifecycle),
        "ready_shard_sha256": evidence.get("ready_shard_sha256"),
        "record_type": FULL_SCORE_DELETION_ATTESTATION_RECORD_TYPE,
        "schema_version": FULL_SCORE_DELETION_ATTESTATION_SCHEMA_VERSION,
        "shard_id": evidence.get("shard_id"),
        "wave_index": evidence.get("wave_index"),
        "worker_index": evidence.get("worker_index"),
    }
    deletion["closed_record_sha256"] = _closed_record_sha256(deletion)
    path = durable_dir / "deletion-attestation.json"
    _require_no_symlink_ancestors(
        path,
        label="deletion attestation path",
        include_leaf=True,
    )
    if path.exists():
        observed = _json_object(path.read_bytes(), "deletion attestation")
        if observed != deletion:
            raise ValueError(
                "existing deletion attestation differs from recovery state"
            )
        return deletion
    _exclusive_write_bytes(path, _canonical_pretty_json_bytes(deletion))
    if _json_object(path.read_bytes(), "deletion attestation") != deletion:
        raise RuntimeError("deletion-attestation durable reread mismatch")
    return deletion


def _delete_ready_shard_tree(ready_dir: Path) -> None:
    """Idempotently finish deletion, including a partially removed tree."""

    _delete_directory_tree_no_follow(
        ready_dir,
        label="ready-shard deletion target",
    )


def _delete_directory_tree_no_follow(target: Path, *, label: str) -> None:
    """Delete a directory by parent fd without following any path symlink."""

    _require_no_symlink_ancestors(
        target,
        label=label,
        include_leaf=True,
    )
    try:
        parent_mode = os.lstat(target.parent).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(parent_mode):
        raise ValueError(f"{label} parent must be a real directory")
    parent_descriptor = _open_directory_no_symlinks(target.parent)
    try:
        try:
            target_status = os.stat(
                target.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(target_status.st_mode):
            raise ValueError(f"{label} must be a real directory")
        if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            raise RuntimeError("platform lacks symlink-safe recursive deletion")
        shutil.rmtree(target.name, dir_fd=parent_descriptor)
        try:
            os.stat(
                target.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("ready-shard deletion was incomplete")
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _recover_committed_consumer_shard(
    payload: Mapping[str, Any],
    shard: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    runtime_verification: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Finish or reuse a committed shard after task/process interruption."""

    wave_index = _required_int(payload, "wave_index")
    shard_id = _required_string(shard, "shard_id")
    durable_dir = (
        _cluster_path(_required_string(payload, "durable_output_root"))
        / "evidence"
        / f"wave-{wave_index:03d}"
        / shard_id
    )
    _require_no_symlink_ancestors(
        durable_dir,
        label="consumer recovery evidence path",
        include_leaf=True,
    )
    evidence_path = durable_dir / "evidence.json"
    if not evidence_path.exists():
        if durable_dir.exists():
            raise ValueError(
                "published durable shard directory lacks committed evidence"
            )
        return None
    evidence, deletion = load_governed_full_score_shard_evidence(
        durable_dir,
        inventory=inventory,
        shard_plan=shard_plan,
        execution_plan=execution_plan,
        require_deletion=False,
    )
    if evidence.get("runtime_verification") != _validate_runtime_verification_binding(
        runtime_verification
    ):
        raise ValueError("recovered shard runtime verification drift")
    ready_dir = _ready_shard_dir(payload, shard_id)
    _require_no_symlink_ancestors(
        ready_dir,
        label="consumer recovery ready-shard path",
        include_leaf=True,
    )
    if deletion is None:
        # The committed evidence has already replayed every preserved source,
        # manifest, output, score, and ready-record closure.  The live ready
        # tree may be incomplete specifically because the prior process died
        # during deletion, so recovery must not require that disposable copy
        # to remain structurally complete before finishing its removal.
        _delete_ready_shard_tree(ready_dir)
        _write_or_validate_deletion_attestation(
            durable_dir,
            payload=payload,
            evidence=evidence,
            execution_plan=execution_plan,
            lifecycle=[
                "verify_ready_shard",
                "baseline_inference",
                "vanilla_inference",
                "validate_paired_outputs",
                "commit_durable_evidence",
                "delete_ephemeral_q8_kv",
            ],
        )
    elif ready_dir.exists() or ready_dir.is_symlink():
        raise ValueError(
            "terminal deletion attestation conflicts with live ready shard"
        )
    # Re-run the full governed check after recovery published its terminal record.
    evidence, _deletion = load_governed_full_score_shard_evidence(
        durable_dir,
        inventory=inventory,
        shard_plan=shard_plan,
        execution_plan=execution_plan,
        require_deletion=True,
    )
    return evidence


def _full_score_expected_prompt_delivery(
    example: Any,
    *,
    method: str,
) -> tuple[str, str, str | None, str | None, str | None]:
    prompt_parts = build_prompt_parts(example)
    logical_prompt = prompt_parts.prefill_prompt
    if method == "baseline_prefill":
        return logical_prompt, logical_prompt, None, None, None
    arm_params = getattr(example, "arm_kv_transfer_params", None)
    if not isinstance(arm_params, Mapping) or set(arm_params) != {
        FULL_SCORE_VANILLA_ARM_ID
    }:
        raise ValueError("Vanilla input lacks its exact handoff parameter mapping")
    params = _json_mapping(
        arm_params[FULL_SCORE_VANILLA_ARM_ID],
        "Vanilla input handoff parameters",
    )
    if params.get(DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM) != "runtime":
        raise ValueError("Vanilla input handoff does not require runtime prompt text")
    runtime_prefix = params.get(DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM, "")
    if not isinstance(runtime_prefix, str):
        raise ValueError("Vanilla runtime prompt prefix must be a string")
    runtime_prompt = f"{runtime_prefix}{prompt_parts.cache_suffix_text}"
    kv_parameter_keys = ",".join(
        sorted(
            {
                *params,
                DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM,
                DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
            }
        )
    )
    expected_request_id = params.get(DOCUMENT_KV_REQUEST_ID_PARAM)
    expected_artifact_id = params.get(DOCUMENT_KV_ARTIFACT_ID_PARAM)
    if (
        not isinstance(expected_request_id, str)
        or not expected_request_id
        or not isinstance(expected_artifact_id, str)
        or not expected_artifact_id
    ):
        raise ValueError("Vanilla handoff lacks request/artifact identity")
    return (
        logical_prompt,
        runtime_prompt,
        kv_parameter_keys,
        expected_request_id,
        expected_artifact_id,
    )


def _required_positive_prompt_metadata_int(
    metadata: Mapping[str, Any],
    field_name: str,
) -> int:
    value = metadata.get(field_name)
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdigit()
        or value.startswith("0")
    ):
        raise ValueError(f"measurement metadata.{field_name} is not canonical")
    return int(value)


def _validate_full_score_measurement_prompt_protocol(
    measurement: Any,
    *,
    method: str,
    expected_logical_prompt_sha256: str,
    expected_runtime_prompt_sha256: str,
    expected_request_prompt_chars: int,
    expected_prefix_cache_salt: str,
    expected_kv_parameter_keys: str | None,
    expected_logical_prompt_tokens: int,
) -> None:
    metadata = measurement.metadata
    expected_prompt_mode = "logical"
    expected_payload_keys = (
        "add_special_tokens,cache_salt,max_tokens,model,prompt,stream,"
        "stream_options,temperature"
        if method == "baseline_prefill"
        else (
            "add_special_tokens,cache_salt,kv_transfer_params,max_tokens,model,"
            "prompt,request_id,stream,stream_options,temperature"
        )
    )
    expected_metadata = {
        "kv_transfer_params_attached": (
            "false" if method == "baseline_prefill" else "true"
        ),
        "logical_prompt_sha256": expected_logical_prompt_sha256,
        "physical_transform_id": (
            "identity"
            if method == "baseline_prefill"
            else "cachet.vanilla.per_document_segments"
        ),
        "physical_transform_version": "1",
        "prefix_cache_salt": expected_prefix_cache_salt,
        "prefix_cache_salt_attached": "true",
        "prompt_text_mode": expected_prompt_mode,
        "prompt_token_source": "server_usage",
        "request_mode": "completion",
        "request_payload_endpoint": DEFAULT_OPENAI_COMPLETIONS_ENDPOINT,
        "request_payload_add_special_tokens": "false",
        "request_payload_keys": expected_payload_keys,
        "request_payload_max_token_fields": "max_tokens",
        "request_payload_max_tokens": str(FULL_SCORE_MAX_TOKENS),
        "request_payload_prompt_chars": str(expected_request_prompt_chars),
        "request_payload_prompt_sha256": expected_logical_prompt_sha256,
        "runtime_prompt_sha256": expected_runtime_prompt_sha256,
        "server": "openai-compatible",
        "server_usage_prompt_tokens_present": "true",
        "stream": "true",
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(f"{method} measurement prompt-delivery protocol drift")
    if expected_kv_parameter_keys is None:
        if "request_payload_kv_transfer_param_keys" in metadata:
            raise ValueError("baseline measurement unexpectedly sent KV parameters")
        if "request_id" in metadata:
            raise ValueError("baseline measurement unexpectedly sent a request ID")
    elif (
        metadata.get("request_payload_kv_transfer_param_keys")
        != expected_kv_parameter_keys
        or metadata.get("request_id") != measurement.request_id
    ):
        raise ValueError("Vanilla measurement request/handoff binding drift")
    logical_tokens = _required_positive_prompt_metadata_int(
        metadata,
        "logical_prompt_tokens",
    )
    runtime_tokens = _required_positive_prompt_metadata_int(
        metadata,
        "runtime_prompt_tokens",
    )
    server_tokens = _required_positive_prompt_metadata_int(
        metadata,
        "server_usage_prompt_tokens",
    )
    if (
        type(measurement.prompt_tokens) is not int
        or measurement.prompt_tokens <= 0
        or measurement.prompt_tokens != server_tokens
        or runtime_tokens != server_tokens
        or logical_tokens != expected_logical_prompt_tokens
        or logical_tokens != runtime_tokens
    ):
        raise ValueError(f"{method} measurement prompt-token accounting drift")


def _validated_method_measurements(
    measurements: Sequence[Any],
    *,
    method: str,
    shard_id: str,
    expected_items: Mapping[tuple[str, str], Mapping[str, Any]],
    examples: Mapping[tuple[str, str], Any],
) -> dict[tuple[str, str], Any]:
    expected_arm = (
        BASELINE_PREFILL_ARM
        if method == "baseline_prefill"
        else FULL_SCORE_VANILLA_ARM_ID
    )
    observed: dict[tuple[str, str], Any] = {}
    registry = default_dataset_scorer_registry()
    for measurement in measurements:
        key = (measurement.dataset, measurement.example_id)
        if measurement.arm_id != expected_arm:
            raise ValueError(f"{method} output contains a different arm")
        if key not in expected_items or key not in examples:
            raise ValueError(f"{method} output contains an unplanned example")
        if key in observed:
            raise ValueError(f"{method} output duplicates an example")
        if measurement.repeat_index != FULL_SCORE_PASSES_PER_METHOD:
            raise ValueError(f"{method} output is not exactly one pass")
        if measurement.error is not None:
            raise ValueError(f"{method} output contains an inference error")
        if method == "baseline_prefill" and measurement.cache_method:
            raise ValueError("baseline output unexpectedly declares a cache method")
        if method == "baseline_prefill" and measurement.artifact_id != "":
            raise ValueError("baseline output unexpectedly declares an artifact ID")
        if (
            method == "vanilla_prefill"
            and measurement.cache_method != "vanilla_prefill"
        ):
            raise ValueError("Vanilla output does not declare vanilla_prefill")
        if method == "baseline_prefill" and measurement.request_id != "":
            raise ValueError("Baseline output unexpectedly declares a request ID")
        if method == "vanilla_prefill":
            _require_nonempty(measurement.request_id, "Vanilla output request_id")
        item = expected_items[key]
        if measurement.metadata.get("logical_prompt_sha256") != item.get(
            "natural_prompt_sha256"
        ):
            raise ValueError("measurement logical prompt hash drift")
        example = examples[key]
        if measurement.expected_answer != example.expected_answer or tuple(
            measurement.references
        ) != tuple(example.references):
            raise ValueError("measurement reference-answer drift")
        observed_niah_cell = measurement.metadata.get("niah_cell_id")
        expected_niah_cell = example.metadata.get("niah_cell_id")
        if measurement.dataset == "niah":
            if (
                expected_niah_cell not in NIAH_CELL_IDS
                or observed_niah_cell != expected_niah_cell
            ):
                raise ValueError("measurement NIAH cell differs from bound source")
        elif observed_niah_cell is not None:
            raise ValueError("non-NIAH measurement declares a NIAH cell")
        (
            logical_prompt,
            runtime_prompt,
            kv_parameter_keys,
            expected_handoff_request_id,
            expected_artifact_id,
        ) = _full_score_expected_prompt_delivery(example, method=method)
        expected_logical_prompt_sha256 = sha256(
            logical_prompt.encode("utf-8")
        ).hexdigest()
        if expected_logical_prompt_sha256 != item.get("natural_prompt_sha256"):
            raise ValueError("measurement source logical prompt hash drift")
        expected_runtime_prompt_sha256 = sha256(
            runtime_prompt.encode("utf-8")
        ).hexdigest()
        if (
            method == "vanilla_prefill"
            and expected_runtime_prompt_sha256 == expected_logical_prompt_sha256
        ):
            raise ValueError("Vanilla runtime prompt did not remove a cached prefix")
        arm_label = "baseline" if method == "baseline_prefill" else "vanilla"
        expected_suite_id = f"{FULL_SCORE_PROTOCOL_ID}:{shard_id}:{method}"
        if method == "vanilla_prefill":
            if (
                expected_handoff_request_id is None or expected_artifact_id is None
            ):  # pragma: no cover - validated by the prompt-delivery helper.
                raise AssertionError("Vanilla handoff identity was not returned")
            expected_measurement_request_id = (
                f"{expected_suite_id}:{measurement.dataset}:"
                f"{measurement.example_id}:{expected_arm}:"
                f"repeat-{measurement.repeat_index}:"
                f"{expected_handoff_request_id}"
            )
            if (
                measurement.request_id != expected_measurement_request_id
                or measurement.artifact_id != expected_artifact_id
            ):
                raise ValueError("Vanilla measurement handoff identity drift")
        expected_prefix_cache_salt = (
            f"{shard_id}:{arm_label}:{expected_suite_id}:"
            f"{measurement.dataset}:{measurement.example_id}:{expected_arm}:"
            f"repeat-{measurement.repeat_index}"
        )
        _validate_full_score_measurement_prompt_protocol(
            measurement,
            method=method,
            expected_logical_prompt_sha256=expected_logical_prompt_sha256,
            expected_runtime_prompt_sha256=expected_runtime_prompt_sha256,
            expected_request_prompt_chars=len(logical_prompt),
            expected_prefix_cache_salt=expected_prefix_cache_salt,
            expected_kv_parameter_keys=kv_parameter_keys,
            expected_logical_prompt_tokens=_required_int(
                item,
                "natural_prompt_tokens",
            ),
        )
        scorer = registry.get(measurement.dataset)
        if (
            measurement.scorer_id != scorer.scorer_id
            or measurement.scorer_version != scorer.version
        ):
            raise ValueError("measurement scorer identity drift")
        extraction = scorer.parse_answer(measurement.output_text)
        if extraction is None:
            raise ValueError("publication scorer must expose the final-answer parser")
        expected_metadata = {
            FINAL_ANSWER_EXTRACTED_METADATA_KEY: (
                extraction.extracted_answer
                if extraction.valid
                else FINAL_ANSWER_NO_EXTRACTION_VALUE
            ),
            FINAL_ANSWER_PARSER_ID_METADATA_KEY: extraction.parser_id,
            FINAL_ANSWER_PARSER_VERSION_METADATA_KEY: extraction.parser_version,
            FINAL_ANSWER_PARSER_PLUGIN_METADATA_KEY: extraction.parser_plugin_path,
            FINAL_ANSWER_PARSER_DIGEST_METADATA_KEY: extraction.parser_digest,
            FINAL_ANSWER_PARSER_VALID_METADATA_KEY: str(extraction.valid).lower(),
            FINAL_ANSWER_PARSER_STATUS_METADATA_KEY: extraction.status,
        }
        if any(
            measurement.metadata.get(key_name) != value
            for key_name, value in expected_metadata.items()
        ):
            raise ValueError("measurement final-answer parser evidence drift")
        expected_scores = (
            scorer.zero_scores()
            if not extraction.valid
            else scorer.score(
                DatasetScoreContext(
                    dataset=measurement.dataset,
                    example_id=measurement.example_id,
                    output_text=extraction.extracted_answer,
                    references=example.references,
                    metadata=example.metadata,
                )
            )
        )
        if dict(measurement.quality_scores) != dict(expected_scores):
            raise ValueError("measurement quality scores do not replay")
        observed[key] = measurement
    if set(observed) != set(expected_items):
        raise ValueError(f"{method} output has partial shard coverage")
    return observed


def _measurement_score_record(measurement: Any) -> dict[str, Any]:
    metadata = measurement.metadata
    return {
        "artifact_id": measurement.artifact_id or None,
        "completion_tokens": measurement.completion_tokens,
        "output_sha256": sha256(measurement.output_text.encode("utf-8")).hexdigest(),
        "parser_status": metadata[FINAL_ANSWER_PARSER_STATUS_METADATA_KEY],
        "parser_valid": metadata[FINAL_ANSWER_PARSER_VALID_METADATA_KEY] == "true",
        "quality_scores": dict(measurement.quality_scores),
        "request_id": measurement.request_id,
        "scorer_id": measurement.scorer_id,
        "scorer_version": measurement.scorer_version,
    }


def _load_and_validate_worker_source_records(
    payload: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    tokenizer: MainLatencyTokenizer,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    assigned_items = {
        (cast(str, item["dataset"]), cast(str, item["example_id"])): item
        for shard in cast(list[Mapping[str, Any]], payload["shards"])
        for item in cast(list[Mapping[str, Any]], shard["items"])
    }
    bindings = payload.get("source_jsonls")
    if not isinstance(bindings, list):
        raise ValueError("source_jsonls must be an array")
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for binding in bindings:
        source = _json_mapping(binding, "source binding")
        dataset = _required_string(source, "dataset")
        path = _governed_existing_file(
            _required_string(source, "uri"),
            f"{dataset} full-score source",
        )
        raw = path.read_bytes()
        if len(raw) != source.get("byte_count") or sha256(
            raw
        ).hexdigest() != source.get("source_jsonl_sha256"):
            raise ValueError(f"{dataset} full-score source hash drift")
        if not raw.endswith(b"\n"):
            raise ValueError(f"{dataset} source must be newline-terminated JSONL")
        lines = raw[:-1].split(b"\n")
        if len(lines) != source.get("record_count") or any(not line for line in lines):
            raise ValueError(f"{dataset} source record-count drift")
        for record_index, raw_line in enumerate(lines, start=1):
            value = json.loads(raw_line.decode("utf-8"))
            if not isinstance(value, Mapping):
                raise ValueError(f"{dataset} source row must be an object")
            record = cast(Mapping[str, Any], value)
            if _TRANSFER_FIELDS.intersection(record):
                raise ValueError(
                    "full-score source unexpectedly contains transfer fields"
                )
            example = _example_from_record(
                record,
                default_dataset=dataset,
                record_index=record_index,
                require_dataset=True,
            )
            key = (dataset, example.example_id)
            item = assigned_items.get(key)
            if item is None:
                continue
            if key in records:
                raise ValueError("full-score source contains a duplicate assigned ID")
            if _canonical_sha256(record) != item.get("source_record_sha256"):
                raise ValueError("assigned source-record hash drift")
            _validate_natural_prompt_item(example, item, tokenizer=tokenizer)
            records[key] = dict(record)
    if set(records) != set(assigned_items):
        raise ValueError("worker source scan has partial assigned-ID coverage")
    return records


def _validate_natural_prompt_item(
    example: Any,
    item: Mapping[str, Any],
    *,
    tokenizer: MainLatencyTokenizer,
) -> None:
    prompt = build_prompt_parts(example)
    natural_ids = _encoded_ids(tokenizer, prompt.prefill_prompt)
    cache_ids = _encoded_ids(tokenizer, prompt.cache_prefix_text)
    segment_ids = tuple(
        _encoded_ids(tokenizer, segment)
        for _segment_id, segment in benchmark_cache_prefix_segments(example)
    )
    checks = {
        "natural_prompt_sha256": sha256(
            prompt.prefill_prompt.encode("utf-8")
        ).hexdigest(),
        "natural_prompt_token_ids_sha256": _token_ids_sha256(natural_ids),
        "natural_prompt_tokens": len(natural_ids),
        "cache_prefix_sha256": sha256(
            prompt.cache_prefix_text.encode("utf-8")
        ).hexdigest(),
        "cache_prefix_token_ids_sha256": _token_ids_sha256(cache_ids),
        "cache_prefix_tokens": len(cache_ids),
        "segment_count": len(segment_ids),
        "segment_token_ids_sha256": _segment_ids_sha256(segment_ids),
    }
    if any(item.get(name) != value for name, value in checks.items()):
        raise ValueError("natural full-score prompt/token identity drift")
    if tuple(token for ids in segment_ids for token in ids) != cache_ids:
        raise ValueError("Vanilla cache segments do not compose exactly")
    if natural_ids[: len(cache_ids)] != cache_ids:
        raise ValueError("Vanilla cache prefix is not the exact natural-prompt prefix")


def _write_shard_inputs(
    shard: Mapping[str, Any],
    *,
    source_records: Mapping[tuple[str, str], Mapping[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Path], dict[tuple[str, str], Any]]:
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    examples: dict[tuple[str, str], Any] = {}
    items = shard.get("items")
    if not isinstance(items, list):
        raise ValueError("shard items must be an array")
    for record_index, raw_item in enumerate(items, start=1):
        item = _json_mapping(raw_item, "shard item")
        key = (_required_string(item, "dataset"), _required_string(item, "example_id"))
        source = source_records.get(key)
        if source is None:
            raise ValueError("shard source ID is missing")
        by_dataset[key[0]].append(source)
        examples[key] = _example_from_record(
            source,
            default_dataset=key[0],
            record_index=record_index,
            require_dataset=True,
        )
    paths = {}
    for dataset, records in sorted(by_dataset.items()):
        path = output_dir / f"{dataset}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(
            path,
            b"".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
                + b"\n"
                for record in records
            ),
        )
        paths[dataset] = path
    return paths, examples


def _generator_command(
    runtime: FullScoreRuntimeConfig,
    *,
    dataset: str,
    input_jsonl: Path,
    output_dir: Path,
    manifest_json: Path,
) -> list[str]:
    code = (
        "from document_kv_cache.benchmark_handoffs import bundle_main; "
        "raise SystemExit(bundle_main())"
    )
    return [
        runtime.python_executable,
        "-c",
        code,
        "--input-jsonl",
        str(input_jsonl),
        "--output-dir",
        str(output_dir),
        "--output-manifest-json",
        str(manifest_json),
        "--generator-factory",
        runtime.generator_factory,
        "--dataset",
        dataset,
        "--backend",
        "vllm",
        "--model-id",
        runtime.model_id,
        "--model-revision",
        runtime.model_revision,
        "--tokenizer-id",
        runtime.tokenizer_id,
        "--tokenizer-revision",
        runtime.tokenizer_revision,
        "--generator-version",
        runtime.generator_version,
        "--dtype",
        FULL_SCORE_KV_DTYPE,
        "--segmented",
        "--segment-per-document",
        "--align-bytes",
        "4096",
        "--cache-method",
        "vanilla_prefill",
    ]


def _enrichment_command(
    runtime: FullScoreRuntimeConfig,
    *,
    dataset: str,
    input_jsonl: Path,
    manifest_json: Path,
    output_jsonl: Path,
) -> list[str]:
    return [
        runtime.python_executable,
        "-m",
        "document_kv_cache.benchmark_handoffs",
        "--input-jsonl",
        str(input_jsonl),
        "--manifest-json",
        str(manifest_json),
        "--output-jsonl",
        str(output_jsonl),
        "--dataset",
        dataset,
        "--arm-id",
        FULL_SCORE_VANILLA_ARM_ID,
    ]


def _benchmark_command(
    runtime: FullScoreRuntimeConfig,
    *,
    method: str,
    shard_id: str,
    dataset_paths: Mapping[str, Path],
    output_json: Path,
) -> list[str]:
    if method not in FULL_SCORE_METHODS:
        raise ValueError("unsupported full-score method")
    suite_id = f"{FULL_SCORE_PROTOCOL_ID}:{shard_id}:{method}"
    args = [
        runtime.python_executable,
        "-m",
        "document_kv_cache.benchmark_runner",
        "--suite-id",
        suite_id,
        "--base-url",
        f"http://{runtime.server_host}:{runtime.server_port}",
        "--cache-base-url",
        f"http://{runtime.server_host}:{runtime.server_port}",
        "--model-id",
        runtime.served_model_name,
        "--canonical-model-id",
        runtime.model_id,
        "--model-revision",
        runtime.model_revision,
        "--tokenizer-id",
        runtime.tokenizer_id,
        "--tokenizer-revision",
        runtime.tokenizer_revision,
        "--hardware-target",
        runtime.hardware_target,
        "--engine-id",
        "vllm",
        "--engine-version",
        runtime.engine_version,
        "--serving-platform",
        "databricks-aws-single-gpu",
        "--model-dtype",
        FULL_SCORE_MODEL_DTYPE,
        "--model-quantization",
        FULL_SCORE_MODEL_QUANTIZATION,
        "--runtime-kv-dtype",
        FULL_SCORE_KV_DTYPE,
        "--max-tokens",
        str(FULL_SCORE_MAX_TOKENS),
        "--temperature",
        "0",
        "--timeout-seconds",
        str(runtime.request_timeout_seconds),
        "--repeats",
        str(FULL_SCORE_PASSES_PER_METHOD),
        "--request-parallelism",
        str(FULL_SCORE_REQUEST_PARALLELISM),
        "--server-usage",
        "--prefix-cache-salt-mode",
        "per_request",
        "--baseline-extra-body-json",
        json.dumps(
            {
                "add_special_tokens": False,
                "cache_salt": f"{shard_id}:baseline",
            },
            sort_keys=True,
        ),
        "--cache-extra-body-json",
        json.dumps(
            {
                "add_special_tokens": False,
                "cache_salt": f"{shard_id}:vanilla",
            },
            sort_keys=True,
        ),
        "--runtime-id",
        f"{FULL_SCORE_PROTOCOL_ID}:{shard_id}",
        "--measurement-scope",
        "quality",
        "--output-json",
        str(output_json),
    ]
    if method == "baseline_prefill":
        args.extend(["--arm", BASELINE_PREFILL_ARM])
    else:
        args.extend(
            [
                "--arm-spec-json",
                json.dumps(_vanilla_arm_spec(), separators=(",", ":"), sort_keys=True),
            ]
        )
    for dataset, path in sorted(dataset_paths.items()):
        args.extend(["--dataset", f"{dataset}={path}"])
    return args


def _vllm_server_command(
    runtime: FullScoreRuntimeConfig,
    *,
    transfer_config: Mapping[str, Any] | None = None,
) -> list[str]:
    return [
        runtime.python_executable,
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        runtime.model_id,
        "--served-model-name",
        runtime.served_model_name,
        "--host",
        runtime.server_host,
        "--port",
        str(runtime.server_port),
        "--dtype",
        FULL_SCORE_MODEL_DTYPE,
        "--max-model-len",
        str(runtime.max_model_len),
        "--max-num-seqs",
        str(runtime.max_num_seqs),
        "--gpu-memory-utilization",
        str(runtime.gpu_memory_utilization),
        "--kv-transfer-config",
        json.dumps(
            runtime.kv_transfer_config if transfer_config is None else transfer_config,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "--enable-prefix-caching",
        "--no-enable-log-requests",
        "--quantization",
        FULL_SCORE_MODEL_QUANTIZATION,
        "--revision",
        runtime.model_revision,
        "--tokenizer-revision",
        runtime.tokenizer_revision,
        "--kv-cache-dtype",
        FULL_SCORE_KV_DTYPE,
        "--attention-backend",
        FULL_SCORE_ATTENTION_BACKEND,
    ]


def _install_spec_uri(value: str) -> str:
    return value.split(" @ ", maxsplit=1)[1].split("#sha256=", maxsplit=1)[0]


def _cluster_artifact_file_uri(value: str) -> str:
    if value.startswith("dbfs:/Volumes/"):
        path = Path("/Volumes") / value.removeprefix("dbfs:/Volumes/")
    elif value.startswith("dbfs:/"):
        path = Path("/dbfs") / value.removeprefix("dbfs:/").lstrip("/")
    else:
        path = Path(value)
    return path.absolute().as_uri()


def _runtime_verifier_command(
    runtime: FullScoreRuntimeConfig,
    *,
    package_wheel_uri: str,
    package_wheel_sha256: str,
) -> list[str]:
    code = (
        "import json,sys; from "
        "document_kv_cache._gpu_qualification_sentinels_v2 import "
        "verify_gpu_qualification_v2_runtime_installation as verify; "
        "record=verify(runtime_lock=sys.argv[1], vllm_uri=sys.argv[2], "
        "flashinfer_uri=sys.argv[3], runtime_closure_manifest=sys.argv[4], "
        "package_uri=sys.argv[5], package_sha256=sys.argv[6]); "
        "sys.stdout.buffer.write((json.dumps(record, ensure_ascii=False, indent=2, "
        "sort_keys=True)+'\\n').encode('utf-8'))"
    )
    package_path = _cluster_path(package_wheel_uri).absolute().as_uri()
    return [
        runtime.python_executable,
        "-c",
        code,
        str(_cluster_path(runtime.runtime_lock_uri)),
        _install_spec_uri(runtime.vllm_wheel_install_spec),
        _install_spec_uri(runtime.flashinfer_wheel_install_spec),
        str(_cluster_path(runtime.runtime_closure_manifest_uri)),
        package_path,
        package_wheel_sha256,
    ]


def _runtime_artifact_binding(
    runtime: FullScoreRuntimeConfig,
    bootstrap_artifacts: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "locked_runtime_identity_sha256": _required_string(
            bootstrap_artifacts,
            "locked_runtime_identity_sha256",
        ),
        "package_wheel_sha256": _required_string(
            bootstrap_artifacts,
            "package_wheel_sha256",
        ),
        "package_wheel_uri": _required_string(
            bootstrap_artifacts,
            "package_wheel_uri",
        ),
        "patched_flashinfer_wheel_sha256": runtime.patched_flashinfer_wheel_sha256,
        "patched_flashinfer_wheel_uri": runtime.patched_flashinfer_wheel_uri,
        "patched_vllm_wheel_sha256": runtime.patched_vllm_wheel_sha256,
        "patched_vllm_wheel_uri": runtime.patched_vllm_wheel_uri,
        "runner_python_file": _required_string(
            bootstrap_artifacts,
            "runner_python_file",
        ),
        "runner_sha256": _required_string(bootstrap_artifacts, "runner_sha256"),
        "runtime_closure_manifest_sha256": runtime.runtime_closure_manifest_sha256,
        "runtime_closure_manifest_uri": runtime.runtime_closure_manifest_uri,
        "runtime_lock_sha256": runtime.runtime_lock_sha256,
        "runtime_lock_uri": runtime.runtime_lock_uri,
    }


def _runtime_verification_binding(
    record: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    attestation = _json_mapping(record, "runtime v2 attestation")
    validate_gpu_qualification_v2_runtime_attestation(attestation)
    artifact_binding = _json_mapping(artifacts, "runtime artifacts")
    _validate_runtime_attestation_artifact_origins(
        attestation,
        artifacts=artifact_binding,
    )
    raw = _canonical_pretty_json_bytes(attestation)
    return {
        "artifacts": artifact_binding,
        "attestation": attestation,
        "attestation_sha256": _canonical_sha256(attestation),
        "file_sha256": sha256(raw).hexdigest(),
    }


def _validate_runtime_attestation_origins(
    record: Mapping[str, Any],
    *,
    runtime: FullScoreRuntimeConfig,
) -> None:
    expected = {
        "flashinfer_direct_url": _install_spec_uri(
            runtime.flashinfer_wheel_install_spec
        ),
        "vllm_direct_url": _install_spec_uri(runtime.vllm_wheel_install_spec),
    }
    for field_name, expected_uri in expected.items():
        if record.get(field_name) != expected_uri:
            raise ValueError(f"runtime attestation {field_name} differs")


def _validate_runtime_attestation_artifact_origins(
    record: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Any],
) -> None:
    expected = {
        "flashinfer_direct_url": _cluster_artifact_file_uri(
            _required_string(artifacts, "patched_flashinfer_wheel_uri")
        ),
        "vllm_direct_url": _cluster_artifact_file_uri(
            _required_string(artifacts, "patched_vllm_wheel_uri")
        ),
    }
    for field_name, expected_uri in expected.items():
        if record.get(field_name) != expected_uri:
            raise ValueError(f"runtime attestation {field_name} differs")


def _validate_runtime_verification_binding(value: Any) -> dict[str, Any]:
    binding = _json_mapping(value, "runtime verification binding")
    if set(binding) != {
        "artifacts",
        "attestation",
        "attestation_sha256",
        "file_sha256",
    }:
        raise ValueError("runtime verification binding keys drift")
    artifacts = _json_mapping(
        _required_mapping(binding, "artifacts"),
        "runtime artifacts",
    )
    expected_artifact_sha256_keys = {
        "locked_runtime_identity_sha256",
        "package_wheel_sha256",
        "patched_flashinfer_wheel_sha256",
        "patched_vllm_wheel_sha256",
        "runner_sha256",
        "runtime_closure_manifest_sha256",
        "runtime_lock_sha256",
    }
    expected_artifact_uri_keys = {
        "package_wheel_uri",
        "patched_flashinfer_wheel_uri",
        "patched_vllm_wheel_uri",
        "runner_python_file",
        "runtime_closure_manifest_uri",
        "runtime_lock_uri",
    }
    expected_artifact_keys = expected_artifact_sha256_keys | expected_artifact_uri_keys
    if set(artifacts) != expected_artifact_keys:
        raise ValueError("runtime artifact binding keys drift")
    for field_name in expected_artifact_sha256_keys:
        _require_sha256(artifacts.get(field_name), field_name=field_name)
    for field_name in expected_artifact_uri_keys:
        _required_string(artifacts, field_name)
    if (
        artifacts["runner_sha256"] != FULL_SCORE_RUNNER_SHA256
        or artifacts["runtime_lock_sha256"] != VLLM_RUNTIME_BASE_LOCK_SHA256
        or artifacts["patched_vllm_wheel_sha256"] != VLLM_PATCHED_WHEEL_SHA256
        or artifacts["patched_flashinfer_wheel_sha256"]
        != FLASHINFER_PATCHED_WHEEL_SHA256
        or artifacts["runtime_closure_manifest_sha256"]
        != RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
        or artifacts["locked_runtime_identity_sha256"]
        != _locked_runtime_identity_sha256(
            runner_sha256=artifacts["runner_sha256"],
            package_wheel_sha256=artifacts["package_wheel_sha256"],
            runtime_lock_sha256=artifacts["runtime_lock_sha256"],
            patched_vllm_wheel_sha256=artifacts["patched_vllm_wheel_sha256"],
            patched_flashinfer_wheel_sha256=(
                artifacts["patched_flashinfer_wheel_sha256"]
            ),
            runtime_closure_manifest_sha256=(
                artifacts["runtime_closure_manifest_sha256"]
            ),
        )
    ):
        raise ValueError("runtime artifact binding identity drift")
    expected = _runtime_verification_binding(
        _required_mapping(binding, "attestation"),
        artifacts=artifacts,
    )
    if binding != expected:
        raise ValueError("runtime verification binding closure drift")
    return binding


def _run_runtime_verifier(
    runtime: FullScoreRuntimeConfig,
    bootstrap_artifacts: Mapping[str, Any],
    output_path: Path,
    *,
    runner: FullScoreCommandRunner,
) -> dict[str, Any]:
    _require_no_symlink_ancestors(
        output_path,
        label="runtime verifier output path",
        include_leaf=True,
    )
    existing_raw: bytes | None = None
    if output_path.exists():
        _require_regular_file_no_follow(output_path, "runtime verifier output")
        existing_raw = output_path.read_bytes()
    command = _runtime_verifier_command(
        runtime,
        package_wheel_uri=_required_string(
            bootstrap_artifacts,
            "package_wheel_uri",
        ),
        package_wheel_sha256=_required_string(
            bootstrap_artifacts,
            "package_wheel_sha256",
        ),
    )
    # The production path needs stdout as an evidence file, so it uses subprocess
    # directly. Injected test runners still exercise the exact command rendering.
    if runner is not _subprocess_command_runner:
        runner(command, env=_worker_environment(runtime))
        if not output_path.exists():
            raise RuntimeError("injected runtime verifier did not publish attestation")
        injected_raw = output_path.read_bytes()
        injected_record = _json_object(injected_raw, "runtime verifier")
        if injected_raw != _canonical_pretty_json_bytes(injected_record):
            raise RuntimeError("injected runtime verifier evidence is not canonical")
        binding = _runtime_verification_binding(
            injected_record,
            artifacts=_runtime_artifact_binding(runtime, bootstrap_artifacts),
        )
        _validate_runtime_attestation_origins(injected_record, runtime=runtime)
        if existing_raw is not None and injected_raw != existing_raw:
            raise RuntimeError(
                "existing runtime verifier evidence differs from current verification"
            )
        if sha256(injected_raw).hexdigest() != binding["file_sha256"]:
            raise RuntimeError("injected runtime verifier evidence digest drift")
        return binding
    try:
        completed = _run_bounded_binary_subprocess(
            command,
            timeout_seconds=FULL_SCORE_RUNTIME_VERIFIER_TIMEOUT_SECONDS,
            output_limit_bytes=FULL_SCORE_RUNTIME_VERIFIER_OUTPUT_LIMIT_BYTES,
            environment=_worker_environment(runtime),
            cwd=Path(runtime.python_executable).parent.parent,
        )
    except _BoundedSubprocessStartFailure:
        raise RuntimeError("native-v2 runtime verifier could not start") from None
    except _BoundedSubprocessTransportFailure:
        raise RuntimeError("native-v2 runtime verifier transport failed") from None
    if completed.timed_out:
        raise RuntimeError("native-v2 runtime verifier timed out")
    if completed.output_limit_exceeded:
        raise RuntimeError("native-v2 runtime verifier output exceeded its bound")
    if not _bounded_stream_result_is_exact(
        completed.stdout
    ) or not _bounded_stream_result_is_exact(completed.stderr):
        raise RuntimeError("native-v2 runtime verifier transport differed")
    if completed.returncode != 0:
        raise RuntimeError("native-v2 runtime verifier exited nonzero")
    if completed.stderr.retained != b"" or completed.stderr.byte_count != 0:
        raise RuntimeError("native-v2 runtime verifier wrote to stderr")
    stdout_raw = completed.stdout.retained
    record = _json_object(stdout_raw, "runtime verifier")
    if stdout_raw != _canonical_pretty_json_bytes(record):
        raise RuntimeError("native-v2 runtime verifier output is not canonical")
    binding = _runtime_verification_binding(
        record,
        artifacts=_runtime_artifact_binding(runtime, bootstrap_artifacts),
    )
    _validate_runtime_attestation_origins(record, runtime=runtime)
    if existing_raw is None:
        _exclusive_write_bytes(output_path, stdout_raw)
    elif existing_raw != stdout_raw:
        raise RuntimeError(
            "existing runtime verifier evidence differs from current verification"
        )
    closed_raw = output_path.read_bytes()
    if (
        closed_raw != stdout_raw
        or sha256(closed_raw).hexdigest() != binding["file_sha256"]
    ):
        raise RuntimeError("runtime verifier evidence durable write drift")
    return binding


def _worker_environment(runtime: FullScoreRuntimeConfig) -> dict[str, str]:
    env = dict(os.environ)
    env.update(gpu_runtime_warning_environment_overrides())
    fixed = {
        CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV: "0",
        CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV: "head_major",
        CACHET_TRANSFORMERS_DEVICE_ENV: "cuda",
        CACHET_TRANSFORMERS_DEVICE_MAP_ENV: "auto",
        CACHET_TRANSFORMERS_MODEL_ID_ENV: runtime.model_id,
        CACHET_TRANSFORMERS_MODEL_REVISION_ENV: runtime.model_revision,
        CACHET_TRANSFORMERS_PRE_ROPE_ENV: "1",
        CACHET_TRANSFORMERS_QUANTIZATION_ENV: FULL_SCORE_GENERATOR_QUANTIZATION,
        CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV: json.dumps(
            FULL_SCORE_GENERATOR_QUANTIZATION_CONFIG,
            separators=(",", ":"),
            sort_keys=True,
        ),
        CACHET_TRANSFORMERS_TOKENIZER_ID_ENV: runtime.tokenizer_id,
        CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV: runtime.tokenizer_revision,
        CACHET_TRANSFORMERS_TORCH_DTYPE_ENV: FULL_SCORE_MODEL_DTYPE,
        CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV: "0",
        "VLLM_ATTENTION_BACKEND": FULL_SCORE_ATTENTION_BACKEND,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    for key, value in fixed.items():
        observed = env.get(key)
        if observed is not None and observed != value:
            raise ValueError(
                f"runtime environment {key} conflicts with full-score protocol"
            )
        env[key] = value
    return env


def _observe_producer_hardware() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("producer hardware attestation requires torch") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("full-score generation requires exactly one visible GPU")
    properties = torch.cuda.get_device_properties(0)
    gpu_name = str(properties.name)
    if gpu_name != FULL_SCORE_PRODUCER_GPU_NAME:
        raise RuntimeError(
            f"full-score generation requires {FULL_SCORE_PRODUCER_GPU_NAME}, got {gpu_name}"
        )
    capability = torch.cuda.get_device_capability(0)
    if tuple(capability) != (8, 9):
        raise RuntimeError("full-score L40S compute-capability drift")
    return {
        "compute_capability": "8.9",
        "gpu_count": 1,
        "gpu_name": gpu_name,
        "hardware_target": FULL_SCORE_PRODUCER_HARDWARE_TARGET,
        "node_type_id": FULL_SCORE_PRODUCER_NODE_TYPE_ID,
        "total_memory_bytes": int(properties.total_memory),
    }


def _generator_artifact_contract_record(
    runtime: FullScoreRuntimeConfig,
) -> dict[str, Any]:
    quantization_config = dict(FULL_SCORE_GENERATOR_QUANTIZATION_CONFIG)
    return {
        "align_bytes": 4096,
        "cache_axis_order": "head_major",
        "cache_method": "vanilla_prefill",
        "dtype": FULL_SCORE_KV_DTYPE,
        "generator_factory": runtime.generator_factory,
        "generator_version": runtime.generator_version,
        "model_id": runtime.model_id,
        "model_revision": runtime.model_revision,
        "pre_rope": True,
        "quantization": FULL_SCORE_GENERATOR_QUANTIZATION,
        "quantization_config": quantization_config,
        "quantization_config_sha256": _canonical_sha256(quantization_config),
        "segment_per_document": True,
        "segmented": True,
        "storage_layout": "separate_key_value",
        "tokenizer_id": runtime.tokenizer_id,
        "tokenizer_revision": runtime.tokenizer_revision,
        "trust_remote_code": False,
        "vllm_bitsandbytes_loader_member": FULL_SCORE_VLLM_BNB_LOADER_MEMBER,
        "vllm_bitsandbytes_loader_sha256": FULL_SCORE_VLLM_BNB_LOADER_SHA256,
    }


def _verify_runtime_contract(runtime: FullScoreRuntimeConfig) -> None:
    lock_path = _governed_existing_file(runtime.runtime_lock_uri, "runtime lock")
    if sha256(lock_path.read_bytes()).hexdigest() != runtime.runtime_lock_sha256:
        raise ValueError("runtime lock SHA-256 drift")
    wheel_path = _governed_existing_file(
        runtime.patched_vllm_wheel_uri,
        "patched vLLM wheel",
    )
    if sha256(wheel_path.read_bytes()).hexdigest() != runtime.patched_vllm_wheel_sha256:
        raise ValueError("patched vLLM wheel SHA-256 drift")
    flashinfer_path = _governed_existing_file(
        runtime.patched_flashinfer_wheel_uri,
        "patched FlashInfer wheel",
    )
    if sha256(flashinfer_path.read_bytes()).hexdigest() != (
        runtime.patched_flashinfer_wheel_sha256
    ):
        raise ValueError("patched FlashInfer wheel SHA-256 drift")
    closure_path = _governed_existing_file(
        runtime.runtime_closure_manifest_uri,
        "runtime closure manifest",
    )
    if sha256(closure_path.read_bytes()).hexdigest() != (
        runtime.runtime_closure_manifest_sha256
    ):
        raise ValueError("runtime closure manifest SHA-256 drift")
    with zipfile.ZipFile(wheel_path) as archive:
        try:
            loader_source = archive.read(FULL_SCORE_VLLM_BNB_LOADER_MEMBER)
        except KeyError as exc:
            raise ValueError(
                "patched wheel lacks the bound BitsAndBytes loader"
            ) from exc
    if sha256(loader_source).hexdigest() != FULL_SCORE_VLLM_BNB_LOADER_SHA256:
        raise ValueError("vLLM BitsAndBytes loader source hash drift")


def _locked_runtime_identity_sha256(
    *,
    runner_sha256: str,
    package_wheel_sha256: str,
    runtime_lock_sha256: str,
    patched_vllm_wheel_sha256: str,
    patched_flashinfer_wheel_sha256: str,
    runtime_closure_manifest_sha256: str,
) -> str:
    digests = (
        _require_sha256(runner_sha256, field_name="runner_sha256"),
        _require_sha256(package_wheel_sha256, field_name="package_wheel_sha256"),
        _require_sha256(runtime_lock_sha256, field_name="runtime_lock_sha256"),
        _require_sha256(
            patched_vllm_wheel_sha256,
            field_name="patched_vllm_wheel_sha256",
        ),
        _require_sha256(
            patched_flashinfer_wheel_sha256,
            field_name="patched_flashinfer_wheel_sha256",
        ),
        _require_sha256(
            runtime_closure_manifest_sha256,
            field_name="runtime_closure_manifest_sha256",
        ),
    )
    return sha256(
        ("cachet.full_score.locked_runtime.v2\0" + "".join(digests)).encode("ascii")
    ).hexdigest()


def _validate_bootstrap_artifact_binding(
    record: Mapping[str, Any],
    *,
    runtime: FullScoreRuntimeConfig,
) -> None:
    expected = {
        "locked_runtime_identity_sha256": _locked_runtime_identity_sha256(
            runner_sha256=_required_string(record, "runner_sha256"),
            package_wheel_sha256=_required_string(record, "package_wheel_sha256"),
            runtime_lock_sha256=_required_string(record, "runtime_lock_sha256"),
            patched_vllm_wheel_sha256=_required_string(
                record,
                "patched_vllm_wheel_sha256",
            ),
            patched_flashinfer_wheel_sha256=_required_string(
                record,
                "patched_flashinfer_wheel_sha256",
            ),
            runtime_closure_manifest_sha256=_required_string(
                record,
                "runtime_closure_manifest_sha256",
            ),
        ),
        "package_wheel_sha256": _required_string(record, "package_wheel_sha256"),
        "package_wheel_uri": _required_string(record, "package_wheel_uri"),
        "patched_vllm_wheel_sha256": runtime.patched_vllm_wheel_sha256,
        "patched_vllm_wheel_uri": runtime.patched_vllm_wheel_uri,
        "patched_flashinfer_wheel_sha256": runtime.patched_flashinfer_wheel_sha256,
        "patched_flashinfer_wheel_uri": runtime.patched_flashinfer_wheel_uri,
        "runner_python_file": _required_string(record, "runner_python_file"),
        "runner_sha256": FULL_SCORE_RUNNER_SHA256,
        "runtime_closure_manifest_sha256": runtime.runtime_closure_manifest_sha256,
        "runtime_closure_manifest_uri": runtime.runtime_closure_manifest_uri,
        "runtime_lock_sha256": runtime.runtime_lock_sha256,
        "runtime_lock_uri": runtime.runtime_lock_uri,
    }
    if dict(record) != expected:
        raise ValueError("worker bootstrap artifact binding drift")


def _validate_full_score_gpu_selection(
    selection: GPUQualificationSelection,
) -> None:
    if selection.attention_backend != FULL_SCORE_ATTENTION_BACKEND:
        raise ValueError("GPU qualification did not select forced TRITON_ATTN")
    if selection.generation_hardware_id != FULL_SCORE_PRODUCER_HARDWARE_TARGET:
        raise ValueError("GPU qualification did not select the L40S producer")
    if selection.generation_databricks_node_type_id != FULL_SCORE_PRODUCER_NODE_TYPE_ID:
        raise ValueError("GPU qualification producer node type drift")
    _require_sha256(
        selection.generation_artifacts_sha256,
        field_name="generation_artifacts_sha256",
    )
    if (
        selection.generation_prefix_tokens_per_second
        < GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND
    ):
        raise ValueError("GPU qualification generation throughput is below 35 token/s")


def _gpu_selection_record(
    selection: GPUQualificationSelection,
) -> dict[str, Any]:
    _validate_full_score_gpu_selection(selection)
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


def _gpu_qualification_binding_record(
    config: FullScoreGPUQualificationConfig,
) -> dict[str, Any]:
    selection = config.selection
    return {
        "artifact_pins": config.artifact_pins.to_record(),
        "campaign_id": config.campaign_id,
        "evidence_closed_record_sha256": config.evidence_record.get(
            "closed_record_sha256"
        ),
        "evidence_file_sha256": config.evidence_file_sha256,
        "evidence_uri": config.evidence_uri,
        "plan_sha256": selection.plan_sha256,
        "plan_uri": config.plan_uri,
        "selection": _gpu_selection_record(selection),
    }


def _validate_gpu_qualification_binding_shape(
    record: Mapping[str, Any],
    *,
    runtime: FullScoreRuntimeConfig,
) -> None:
    expected_keys = {
        "artifact_pins",
        "campaign_id",
        "evidence_closed_record_sha256",
        "evidence_file_sha256",
        "evidence_uri",
        "plan_sha256",
        "plan_uri",
        "selection",
    }
    if set(record) != expected_keys:
        raise ValueError("GPU qualification binding keys drift")
    for field_name in (
        "campaign_id",
        "evidence_uri",
        "plan_uri",
    ):
        _required_string(record, field_name)
    for field_name in (
        "evidence_closed_record_sha256",
        "evidence_file_sha256",
        "plan_sha256",
    ):
        _require_sha256(record.get(field_name), field_name=field_name)
    pins = GPUQualificationArtifactPinsV2(
        **_json_mapping(
            _required_mapping(record, "artifact_pins"),
            "GPU artifact pins",
        )
    )
    expected_runtime_pins = {
        "runtime_lock_sha256": runtime.runtime_lock_sha256,
        "patched_vllm_wheel_sha256": runtime.patched_vllm_wheel_sha256,
        "patched_flashinfer_wheel_sha256": (runtime.patched_flashinfer_wheel_sha256),
        "runtime_closure_manifest_sha256": (runtime.runtime_closure_manifest_sha256),
    }
    if any(
        getattr(pins, field_name) != expected
        for field_name, expected in expected_runtime_pins.items()
    ):
        raise ValueError("GPU qualification v2 runtime closure pin drift")
    selection = _required_mapping(record, "selection")
    if selection.get("attention_backend") != FULL_SCORE_ATTENTION_BACKEND:
        raise ValueError("GPU qualification attention backend drift")
    if selection.get("generation_hardware_id") != FULL_SCORE_PRODUCER_HARDWARE_TARGET:
        raise ValueError("GPU qualification producer hardware drift")
    if selection.get("generation_databricks_node_type_id") != (
        FULL_SCORE_PRODUCER_NODE_TYPE_ID
    ):
        raise ValueError("GPU qualification producer node drift")
    if selection.get("plan_sha256") != record.get("plan_sha256"):
        raise ValueError("GPU qualification plan binding drift")
    if selection.get("gpu_memory_utilization") != runtime.gpu_memory_utilization:
        raise ValueError("GPU qualification selected GMU drift")
    rate = selection.get("generation_prefix_tokens_per_second")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or float(rate) < GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND
    ):
        raise ValueError("GPU qualification generation rate is invalid")
    _require_sha256(
        selection.get("generation_artifacts_sha256"),
        field_name="generation_artifacts_sha256",
    )


def _verify_bound_gpu_qualification(
    record: Mapping[str, Any],
    *,
    runtime: FullScoreRuntimeConfig,
) -> GPUQualificationSelection:
    _validate_gpu_qualification_binding_shape(record, runtime=runtime)
    plan = _read_bound_json(
        _required_string(record, "plan_uri"),
        _required_string(record, "plan_sha256"),
        closure_digest=True,
    )
    evidence_path = _governed_existing_file(
        _required_string(record, "evidence_uri"),
        "GPU qualification evidence",
    )
    evidence_raw = evidence_path.read_bytes()
    if sha256(evidence_raw).hexdigest() != record.get("evidence_file_sha256"):
        raise ValueError("GPU qualification evidence file SHA-256 drift")
    evidence = _json_object(evidence_raw, "GPU qualification evidence")
    if evidence.get("closed_record_sha256") != record.get(
        "evidence_closed_record_sha256"
    ):
        raise ValueError("GPU qualification evidence closure drift")
    pins = GPUQualificationArtifactPinsV2(
        **_json_mapping(record.get("artifact_pins"), "GPU artifact pins")
    )
    selection = validate_gpu_qualification_evidence_v2_record(
        evidence,
        plan_record=plan,
        expected_campaign_id=_required_string(record, "campaign_id"),
        expected_artifact_pins=pins,
    )
    _validate_full_score_gpu_selection(selection)
    if _gpu_selection_record(selection) != record.get("selection"):
        raise ValueError("GPU qualification selection drift")
    return selection


def _wait_for_server(
    runtime: FullScoreRuntimeConfig,
    process: subprocess.Popen[Any],
    log_path: Path,
) -> None:
    deadline = time.monotonic() + runtime.server_start_timeout_seconds
    base_url = f"http://{runtime.server_host}:{runtime.server_port}"
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited with {process.returncode}; log tail:\n{_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
                if not 200 <= response.status < 300:
                    raise RuntimeError(f"health returned {response.status}")
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as response:
                models = json.loads(response.read().decode("utf-8"))
            served = {
                item.get("id")
                for item in models.get("data", [])
                if isinstance(item, Mapping)
            }
            if runtime.served_model_name in served:
                return
            last_error = f"served models were {sorted(str(item) for item in served)}"
        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            KeyError,
            RuntimeError,
        ) as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(
        f"timed out waiting for vLLM: {last_error}; log tail:\n{_tail(log_path)}"
    )


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _runtime_record(runtime: FullScoreRuntimeConfig) -> dict[str, Any]:
    return {
        "attention_backend": FULL_SCORE_ATTENTION_BACKEND,
        "engine_version": runtime.engine_version,
        "generator_factory": runtime.generator_factory,
        "generator_quantization": FULL_SCORE_GENERATOR_QUANTIZATION,
        "generator_timeout_seconds": runtime.generator_timeout_seconds,
        "generator_version": runtime.generator_version,
        "gpu_memory_utilization": runtime.gpu_memory_utilization,
        "hardware_target": runtime.hardware_target,
        "kv_cache_dtype": FULL_SCORE_KV_DTYPE,
        "kv_transfer_config": dict(runtime.kv_transfer_config),
        "max_model_len": runtime.max_model_len,
        "max_num_seqs": runtime.max_num_seqs,
        "model_dtype": FULL_SCORE_MODEL_DTYPE,
        "model_id": runtime.model_id,
        "model_quantization": FULL_SCORE_MODEL_QUANTIZATION,
        "model_revision": runtime.model_revision,
        "python_executable": runtime.python_executable,
        "patched_vllm_wheel_sha256": runtime.patched_vllm_wheel_sha256,
        "patched_vllm_wheel_uri": runtime.patched_vllm_wheel_uri,
        "patched_flashinfer_wheel_sha256": runtime.patched_flashinfer_wheel_sha256,
        "patched_flashinfer_wheel_uri": runtime.patched_flashinfer_wheel_uri,
        "request_timeout_seconds": runtime.request_timeout_seconds,
        "runtime_closure_manifest_sha256": runtime.runtime_closure_manifest_sha256,
        "runtime_closure_manifest_uri": runtime.runtime_closure_manifest_uri,
        "runtime_lock_sha256": runtime.runtime_lock_sha256,
        "runtime_lock_uri": runtime.runtime_lock_uri,
        "served_model_name": runtime.served_model_name,
        "server_host": runtime.server_host,
        "server_port": runtime.server_port,
        "server_start_timeout_seconds": runtime.server_start_timeout_seconds,
        "telemetry_interval_seconds": runtime.telemetry_interval_seconds,
        "tokenizer_id": runtime.tokenizer_id,
        "tokenizer_revision": runtime.tokenizer_revision,
        "trust_remote_code": False,
        "vllm_wheel_install_spec": runtime.vllm_wheel_install_spec,
        "flashinfer_wheel_install_spec": runtime.flashinfer_wheel_install_spec,
    }


def _runtime_from_record(record: Mapping[str, Any]) -> FullScoreRuntimeConfig:
    expected_fixed = {
        "attention_backend": FULL_SCORE_ATTENTION_BACKEND,
        "generator_quantization": FULL_SCORE_GENERATOR_QUANTIZATION,
        "kv_cache_dtype": FULL_SCORE_KV_DTYPE,
        "model_dtype": FULL_SCORE_MODEL_DTYPE,
        "model_quantization": FULL_SCORE_MODEL_QUANTIZATION,
        "runtime_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "trust_remote_code": False,
    }
    if any(record.get(key) != value for key, value in expected_fixed.items()):
        raise ValueError("full-score runtime fixed settings drift")
    return FullScoreRuntimeConfig(
        python_executable=_required_string(record, "python_executable"),
        runtime_lock_uri=_required_string(record, "runtime_lock_uri"),
        runtime_lock_sha256=_required_string(record, "runtime_lock_sha256"),
        patched_vllm_wheel_uri=_required_string(record, "patched_vllm_wheel_uri"),
        patched_vllm_wheel_sha256=_required_string(record, "patched_vllm_wheel_sha256"),
        patched_flashinfer_wheel_uri=_required_string(
            record, "patched_flashinfer_wheel_uri"
        ),
        patched_flashinfer_wheel_sha256=_required_string(
            record, "patched_flashinfer_wheel_sha256"
        ),
        runtime_closure_manifest_uri=_required_string(
            record, "runtime_closure_manifest_uri"
        ),
        runtime_closure_manifest_sha256=_required_string(
            record, "runtime_closure_manifest_sha256"
        ),
        vllm_wheel_install_spec=_required_string(record, "vllm_wheel_install_spec"),
        flashinfer_wheel_install_spec=_required_string(
            record, "flashinfer_wheel_install_spec"
        ),
        kv_transfer_config=_required_mapping(record, "kv_transfer_config"),
        model_id=_required_string(record, "model_id"),
        served_model_name=_required_string(record, "served_model_name"),
        model_revision=_required_string(record, "model_revision"),
        tokenizer_id=_required_string(record, "tokenizer_id"),
        tokenizer_revision=_required_string(record, "tokenizer_revision"),
        hardware_target=_required_string(record, "hardware_target"),
        engine_version=_required_string(record, "engine_version"),
        gpu_memory_utilization=_required_number(record, "gpu_memory_utilization"),
        max_model_len=_required_int(record, "max_model_len"),
        max_num_seqs=_required_int(record, "max_num_seqs"),
        server_host=_required_string(record, "server_host"),
        server_port=_required_int(record, "server_port"),
        server_start_timeout_seconds=_required_number(
            record, "server_start_timeout_seconds"
        ),
        request_timeout_seconds=_required_number(record, "request_timeout_seconds"),
        generator_timeout_seconds=_required_number(record, "generator_timeout_seconds"),
        telemetry_interval_seconds=_required_number(
            record, "telemetry_interval_seconds"
        ),
        generator_factory=_required_string(record, "generator_factory"),
        generator_version=_required_string(record, "generator_version"),
    )


def _full_score_protocol_record() -> dict[str, Any]:
    return {
        "add_special_tokens": False,
        "complete_inventory_required": True,
        "input_length": {
            "max_natural_prompt_tokens": FULL_SCORE_MAX_NATURAL_PROMPT_TOKENS,
            "padding": False,
            "tokenizer_truncation": False,
        },
        "lifecycle": [
            "generate_q8_kv",
            "baseline_inference",
            "vanilla_inference",
            "validate_paired_outputs",
            "commit_durable_evidence",
            "delete_ephemeral_q8_kv",
        ],
        "max_tokens": FULL_SCORE_MAX_TOKENS,
        "methods": list(FULL_SCORE_METHODS),
        "natural_eos": True,
        "passes_per_method": FULL_SCORE_PASSES_PER_METHOD,
        "prompt_text_mode": "logical",
        "protocol_id": FULL_SCORE_PROTOCOL_ID,
        "request_parallelism": FULL_SCORE_REQUEST_PARALLELISM,
        "temperature": FULL_SCORE_TEMPERATURE,
    }


def _scorer_contract_record() -> list[dict[str, Any]]:
    registry = default_dataset_scorer_registry()
    return [
        {
            "answer_parser_digest": scorer.answer_parser_digest,
            "answer_parser_id": scorer.answer_parser_id,
            "answer_parser_plugin_path": scorer.answer_parser_plugin_path,
            "answer_parser_version": scorer.answer_parser_version,
            "dataset": dataset,
            "metric_names": list(scorer.metric_names),
            "plugin_path": scorer.plugin_path,
            "publication_approved": scorer.publication_approved,
            "scorer_id": scorer.scorer_id,
            "scorer_version": scorer.version,
        }
        for dataset, scorer in registry.entries
    ]


def _vanilla_arm_spec() -> dict[str, Any]:
    arm = method_benchmark_arm(
        "vanilla_prefill",
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


def _validate_source_bindings(
    record: Mapping[str, Any],
    inventory: FullScoreInventory,
) -> None:
    bindings = record.get("source_jsonls")
    if not isinstance(bindings, list):
        raise ValueError("worker source_jsonls must be an array")
    observed = {
        binding.get("dataset"): binding
        for binding in bindings
        if isinstance(binding, Mapping)
    }
    if set(observed) != set(SUPPORTED_V1_DATASETS):
        raise ValueError("worker source bindings have incomplete dataset coverage")
    for source in inventory.sources:
        binding = cast(Mapping[str, Any], observed[source.dataset])
        expected = {
            "byte_count": source.byte_count,
            "dataset": source.dataset,
            "identities_sha256": source.identities_sha256,
            "record_count": source.record_count,
            "source_jsonl_sha256": source.source_jsonl_sha256,
            "source_records_sha256": source.source_records_sha256,
        }
        if any(binding.get(key) != value for key, value in expected.items()):
            raise ValueError("worker source binding drift")
        _required_string(binding, "uri")


def _validate_execution_plan(
    record: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
) -> None:
    if record.get("record_type") != FULL_SCORE_EXECUTION_PLAN_RECORD_TYPE:
        raise ValueError("unsupported full-score execution-plan record_type")
    if record.get("schema_version") != FULL_SCORE_EXECUTION_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported full-score execution-plan schema_version")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("full-score execution-plan closure drift")
    if record.get("inventory_sha256") != inventory.inventory_sha256:
        raise ValueError("full-score execution-plan inventory drift")
    if record.get("shard_plan_sha256") != shard_plan.get("closed_record_sha256"):
        raise ValueError("full-score execution-plan shard-plan drift")
    if not _json_type_exact_equal(
        record.get("protocol"),
        _full_score_protocol_record(),
    ):
        raise ValueError("full-score execution-plan protocol drift")
    waves = record.get("waves")
    if not isinstance(waves, list) or not waves:
        raise ValueError("full-score execution plan must contain waves")
    expected_ids = {
        cast(str, shard["shard_id"])
        for shard in cast(list[Mapping[str, Any]], shard_plan["shards"])
    }
    producer_ids: list[str] = []
    consumer_ids: list[str] = []
    for wave_index, raw_wave in enumerate(waves):
        wave = _json_mapping(raw_wave, "execution wave")
        if wave.get("wave_index") != wave_index:
            raise ValueError("execution wave indices must be contiguous")
        upper_bound = wave.get("ready_bytes_upper_bound")
        cap = wave.get("max_backlog_bytes")
        if type(upper_bound) is not int or type(cap) is not int or upper_bound > cap:
            raise ValueError("execution wave violates the backlog byte cap")
        mode = wave.get("scheduling_mode")
        if mode not in {"phased", "split"}:
            raise ValueError("execution wave scheduling_mode is invalid")
        producers = cast(list[Mapping[str, Any]], wave["producer_assignments"])
        consumers = cast(list[Mapping[str, Any]], wave["consumer_assignments"])
        if (
            len(producers) > FULL_SCORE_MAX_WORKERS
            or len(consumers) > FULL_SCORE_MAX_WORKERS
        ):
            raise ValueError("execution wave phase exceeds sixteen workers")
        if mode == "split" and len(producers) + len(consumers) > FULL_SCORE_MAX_WORKERS:
            raise ValueError("split execution wave exceeds sixteen live workers")
        producer_ids.extend(
            shard_id
            for assignment in producers
            for shard_id in cast(list[str], assignment["shard_ids"])
        )
        consumer_ids.extend(
            shard_id
            for assignment in consumers
            for shard_id in cast(list[str], assignment["shard_ids"])
        )
    if set(producer_ids) != expected_ids or len(producer_ids) != len(expected_ids):
        raise ValueError("producer assignments do not cover every shard once")
    if set(consumer_ids) != expected_ids or len(consumer_ids) != len(expected_ids):
        raise ValueError("consumer assignments do not cover every shard once")


def _publication_source_contract() -> dict[str, dict[str, Any]]:
    return {
        dataset: {
            "byte_count": byte_count,
            "dataset": dataset,
            "identities_sha256": identities_sha256,
            "record_count": record_count,
            "source_jsonl_sha256": source_jsonl_sha256,
            "source_records_sha256": source_records_sha256,
        }
        for (
            dataset,
            byte_count,
            record_count,
            source_jsonl_sha256,
            source_records_sha256,
            identities_sha256,
        ) in _FULL_SCORE_PUBLICATION_SOURCE_RECORDS
    }


def _validate_publication_full_score_inputs(
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
) -> None:
    """Freeze the only inputs authorized for the publication GPU campaign."""

    if not isinstance(inventory, FullScoreInventory):
        raise TypeError("publication full-score inventory must be FullScoreInventory")
    validate_full_score_shard_plan(shard_plan, inventory=inventory)
    _validate_execution_plan(
        execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
    )
    if inventory.inventory_sha256 != FULL_SCORE_PUBLICATION_INVENTORY_SHA256:
        raise ValueError("publication full-score inventory closure drift")
    if shard_plan.get("closed_record_sha256") != (
        FULL_SCORE_PUBLICATION_SHARD_PLAN_SHA256
    ):
        raise ValueError("publication full-score shard-plan closure drift")
    if execution_plan.get("closed_record_sha256") != (
        FULL_SCORE_PUBLICATION_EXECUTION_PLAN_SHA256
    ):
        raise ValueError("publication full-score execution-plan closure drift")
    if shard_plan.get("inventory_sha256") != (FULL_SCORE_PUBLICATION_INVENTORY_SHA256):
        raise ValueError("publication shard plan binds a different inventory")
    if len(inventory.items) != FULL_SCORE_PUBLICATION_ITEM_COUNT:
        raise ValueError("publication full-score inventory count drift")
    inventory_prefix_tokens = sum(item.cache_prefix_tokens for item in inventory.items)
    inventory_natural_tokens = sum(
        item.natural_prompt_tokens for item in inventory.items
    )
    if inventory_prefix_tokens != FULL_SCORE_PUBLICATION_CACHE_PREFIX_TOKENS:
        raise ValueError("publication inventory cache-prefix token total drift")
    if inventory_natural_tokens != FULL_SCORE_PUBLICATION_NATURAL_PROMPT_TOKENS:
        raise ValueError("publication inventory natural-prompt token total drift")
    expected_sources = _publication_source_contract()
    observed_sources = {
        source.dataset: {
            "byte_count": source.byte_count,
            "dataset": source.dataset,
            "identities_sha256": source.identities_sha256,
            "record_count": source.record_count,
            "source_jsonl_sha256": source.source_jsonl_sha256,
            "source_records_sha256": source.source_records_sha256,
        }
        for source in inventory.sources
    }
    if observed_sources != expected_sources:
        raise ValueError("publication full-score frozen source hash drift")

    raw_shards = shard_plan.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("publication shard plan shards must be an array")
    if len(raw_shards) != FULL_SCORE_PUBLICATION_SHARD_COUNT:
        raise ValueError("publication full-score shard count drift")
    plan_shards = [
        _json_mapping(shard, "publication plan shard") for shard in raw_shards
    ]
    if sum(_required_int(shard, "item_count") for shard in plan_shards) != (
        FULL_SCORE_PUBLICATION_ITEM_COUNT
    ):
        raise ValueError("publication shard-plan item count drift")
    if (
        sum(_required_int(shard, "cache_prefix_tokens") for shard in plan_shards)
        != FULL_SCORE_PUBLICATION_CACHE_PREFIX_TOKENS
    ):
        raise ValueError("publication shard-plan cache-prefix token total drift")
    if (
        sum(_required_int(shard, "natural_prompt_tokens") for shard in plan_shards)
        != FULL_SCORE_PUBLICATION_NATURAL_PROMPT_TOKENS
    ):
        raise ValueError("publication shard-plan natural-prompt token total drift")

    if execution_plan.get("scheduling_mode") != "phased":
        raise ValueError("publication Databricks campaign requires phased waves")
    if execution_plan.get("max_live_gpu_tasks") != FULL_SCORE_MAX_WORKERS:
        raise ValueError("publication execution plan live-task cap drift")
    max_backlog = execution_plan.get("max_backlog_bytes")
    if (
        type(max_backlog) is not int
        or max_backlog <= 0
        or max_backlog > FULL_SCORE_DEFAULT_MAX_BACKLOG_BYTES
    ):
        raise ValueError("publication execution plan backlog cap drift")
    waves = cast(list[Mapping[str, Any]], execution_plan["waves"])
    if len(cast(list[Any], waves[0].get("shard_ids"))) != (
        FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE
    ):
        raise ValueError("publication wave zero must contain sixteen shards")
    expected_plan_shards = {
        _required_string(shard, "shard_id"): shard for shard in plan_shards
    }
    observed_execution_shards: dict[str, Mapping[str, Any]] = {}
    for wave in waves:
        wave_shards = wave.get("shards")
        if not isinstance(wave_shards, list) or not wave_shards:
            raise ValueError("publication execution wave has no shards")
        if len(wave_shards) > FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE:
            raise ValueError("publication execution wave exceeds sixteen shards")
        wave_shard_ids = {
            _required_string(
                _json_mapping(shard, "publication execution shard"),
                "shard_id",
            )
            for shard in wave_shards
        }
        for assignment_key in (
            "producer_assignments",
            "consumer_assignments",
        ):
            assignments = wave.get(assignment_key)
            if not isinstance(assignments, list) or len(assignments) != len(
                wave_shards
            ):
                raise ValueError(
                    "publication billing requires one worker task per shard"
                )
            assigned_ids: list[str] = []
            for raw_assignment in assignments:
                assignment = _json_mapping(
                    raw_assignment,
                    "publication worker assignment",
                )
                shard_ids = assignment.get("shard_ids")
                if not isinstance(shard_ids, list) or len(shard_ids) != 1:
                    raise ValueError(
                        "publication billing requires one worker task per shard"
                    )
                assigned_ids.append(
                    _require_nonempty(shard_ids[0], "assignment shard_id")
                )
            if set(assigned_ids) != wave_shard_ids:
                raise ValueError("publication worker assignment crosses wave shards")
        for raw_shard in wave_shards:
            shard = _json_mapping(raw_shard, "publication execution shard")
            shard_id = _required_string(shard, "shard_id")
            if shard_id in observed_execution_shards:
                raise ValueError("publication execution plan duplicates a shard")
            planned = expected_plan_shards.get(shard_id)
            if planned is None or any(
                shard.get(key) != value for key, value in planned.items()
            ):
                raise ValueError("publication execution shard differs from shard plan")
            observed_execution_shards[shard_id] = shard
    if set(observed_execution_shards) != set(expected_plan_shards):
        raise ValueError("publication execution plan has partial shard coverage")


def _validate_budget_execution_plan(record: Mapping[str, Any]) -> None:
    if record.get("record_type") != FULL_SCORE_EXECUTION_PLAN_RECORD_TYPE:
        raise ValueError("live P90 requires a full-score execution plan")
    if record.get("schema_version") != FULL_SCORE_EXECUTION_PLAN_SCHEMA_VERSION:
        raise ValueError("live P90 execution-plan schema drift")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("live P90 execution-plan closure drift")
    if not _json_type_exact_equal(
        record.get("protocol"),
        _full_score_protocol_record(),
    ):
        raise ValueError("live P90 execution-plan protocol drift")
    waves = record.get("waves")
    if not isinstance(waves, list) or len(waves) < 2:
        raise ValueError("live P90 requires a multi-wave execution plan")
    observed_ids: set[str] = set()
    for wave_index, raw_wave in enumerate(waves):
        wave = _json_mapping(raw_wave, "execution wave")
        if wave.get("wave_index") != wave_index:
            raise ValueError("live P90 wave indices are not contiguous")
        shards = wave.get("shards")
        shard_ids = wave.get("shard_ids")
        if not isinstance(shards, list) or not isinstance(shard_ids, list):
            raise ValueError("live P90 execution wave has invalid shard arrays")
        derived_ids = [
            _required_string(_json_mapping(shard, "execution shard"), "shard_id")
            for shard in shards
        ]
        if derived_ids != shard_ids or len(set(derived_ids)) != len(derived_ids):
            raise ValueError("live P90 execution wave shard identity drift")
        if observed_ids.intersection(derived_ids):
            raise ValueError("live P90 execution plan duplicates shards")
        observed_ids.update(derived_ids)


def _validate_matched_billing_block(
    block: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any],
    planned_by_id: Mapping[str, tuple[int, Mapping[str, Any]]],
) -> None:
    if block.get("record_type") != FULL_SCORE_MATCHED_BLOCK_RECORD_TYPE:
        raise ValueError("live P90 input has the wrong matched-block record_type")
    if block.get("schema_version") != FULL_SCORE_MATCHED_BLOCK_SCHEMA_VERSION:
        raise ValueError("live P90 matched-block schema drift")
    if block.get("closed_record_sha256") != _closed_record_sha256(block):
        raise ValueError("live P90 matched-block closure drift")
    if block.get("authorization_scope") not in {
        FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
        FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE,
    }:
        raise ValueError("live P90 matched-block authorization scope drift")
    if block.get("execution_plan_sha256") != execution_plan.get("closed_record_sha256"):
        raise ValueError("live P90 matched-block execution-plan drift")
    if block.get("matched_status") != "success_error_free":
        raise ValueError("live P90 rejects a non-successful matched block")
    if block.get("protocol_sha256") != _canonical_sha256(_full_score_protocol_record()):
        raise ValueError("live P90 matched-block protocol drift")
    shard_id = _required_string(block, "shard_id")
    planned_entry = planned_by_id.get(shard_id)
    if planned_entry is None:
        raise ValueError("live P90 matched block references an unknown shard")
    wave_index, planned = planned_entry
    expected = {
        "cache_prefix_tokens": planned.get("cache_prefix_tokens"),
        "natural_prompt_tokens": planned.get("natural_prompt_tokens"),
        "shard_items_sha256": planned.get("items_sha256"),
        "wave_index": wave_index,
    }
    if any(block.get(key) != value for key, value in expected.items()):
        raise ValueError("live P90 matched-block planned work drift")
    for field_name in (
        "deletion_attestation_sha256",
        "evidence_sha256",
    ):
        _require_sha256(block.get(field_name), field_name=field_name)
    billed = _required_mapping(block, "billed_gpu_seconds")
    sources = _required_mapping(block, "billing_source_sha256")
    expected_roles = {"producer", "consumer_task"}
    if set(billed) != expected_roles or set(sources) != expected_roles:
        raise ValueError("live P90 matched block has incomplete role billing")
    for role in sorted(expected_roles):
        _positive_finite_float(billed.get(role), f"billed_gpu_seconds.{role}")
        _require_sha256(sources.get(role), field_name=f"billing_source_sha256.{role}")
    diagnostics = _required_mapping(block, "consumer_task_diagnostics")
    if set(diagnostics) != {
        "attribution",
        "method_wall_clock",
        "method_wall_seconds",
        "shared_or_unattributed_seconds",
    }:
        raise ValueError("live P90 consumer-task diagnostics schema drift")
    if diagnostics.get("attribution") != "indivisible_no_per_arm_billed_seconds":
        raise ValueError("live P90 forbids per-arm billing attribution")
    if diagnostics.get("method_wall_clock") != "time.monotonic_ns":
        raise ValueError("live P90 method wall-clock identity drift")
    method_walls = _required_mapping(diagnostics, "method_wall_seconds")
    if set(method_walls) != set(FULL_SCORE_METHODS):
        raise ValueError("live P90 method wall evidence is incomplete")
    normalized_walls = [
        _positive_finite_float(
            method_walls.get(method),
            f"consumer_task_diagnostics.method_wall_seconds.{method}",
        )
        for method in FULL_SCORE_METHODS
    ]
    consumer_actual = _positive_finite_float(
        billed.get("consumer_task"),
        "billed_gpu_seconds.consumer_task",
    )
    expected_shared = max(0.0, consumer_actual - sum(normalized_walls))
    observed_shared = _nonnegative_finite_float(
        diagnostics.get("shared_or_unattributed_seconds"),
        "consumer_task_diagnostics.shared_or_unattributed_seconds",
    )
    if abs(observed_shared - expected_shared) > 1e-12:
        raise ValueError("live P90 consumer-task shared-time diagnostic drift")
    output_tokens = _required_mapping(block, "observed_completion_tokens")
    if set(output_tokens) != set(FULL_SCORE_METHODS):
        raise ValueError("live P90 matched block lacks output-token evidence")
    item_count = cast(int, planned["item_count"])
    for method in FULL_SCORE_METHODS:
        value = output_tokens.get(method)
        if (
            type(value) is not int
            or not 0 <= value <= item_count * FULL_SCORE_MAX_TOKENS
        ):
            raise ValueError("live P90 matched block output-token evidence is invalid")


def _remote_consumer_evidence_records(
    authorization: object,
    *,
    completion_record: Mapping[str, Any] | None,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    expected_wave_index: int,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    if compact_artifact_resolver is None:
        raise TypeError("remote consumer authority requires a compact CAS resolver")
    from document_kv_cache.full_score_remote_control import (
        require_full_score_remote_consumer_evidence_authorization,
    )

    remote_authorization = require_full_score_remote_consumer_evidence_authorization(
        authorization,
        execution_plan=execution_plan,
        expected_wave_index=expected_wave_index,
        completion_record=completion_record,
    )
    records: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for raw_binding in remote_authorization.evidence_bindings:
        binding = _json_mapping(raw_binding, "remote consumer evidence binding")
        shard_id = _required_string(binding, "shard_id")
        evidence_uri = _required_string(binding, "evidence_uri")
        deletion_uri = _required_string(binding, "deletion_uri")
        evidence_file = _governed_compact_file(
            evidence_uri,
            "remote consumer shard evidence",
            compact_artifact_resolver,
        )
        deletion_file = _governed_compact_file(
            deletion_uri,
            "remote consumer deletion attestation",
            compact_artifact_resolver,
        )
        evidence_bytes = evidence_file.read_bytes()
        deletion_bytes = deletion_file.read_bytes()
        evidence = _json_object(evidence_bytes, "remote consumer shard evidence")
        deletion = _json_object(
            deletion_bytes,
            "remote consumer deletion attestation",
        )
        if (
            evidence_bytes != _canonical_pretty_json_bytes(evidence)
            or deletion_bytes != _canonical_pretty_json_bytes(deletion)
            or sha256(evidence_bytes).hexdigest() != binding.get("evidence_file_sha256")
            or evidence.get("closed_record_sha256")
            != binding.get("evidence_record_sha256")
            or sha256(deletion_bytes).hexdigest() != binding.get("deletion_file_sha256")
            or deletion.get("closed_record_sha256")
            != binding.get("deletion_record_sha256")
        ):
            raise ValueError("remote consumer CAS evidence binding drift")
        _validate_shard_evidence_record(
            evidence,
            inventory_sha256=inventory.inventory_sha256,
            shard_plan_sha256=_required_string(
                shard_plan,
                "closed_record_sha256",
            ),
        )
        _validate_full_score_deletion_attestation(
            deletion,
            evidence_record=evidence,
            execution_plan_sha256=_required_string(
                execution_plan,
                "closed_record_sha256",
            ),
            shard_id=shard_id,
            wave_index=expected_wave_index,
        )
        if shard_id in records:
            raise ValueError("remote consumer CAS duplicates shard evidence")
        records[shard_id] = (evidence, deletion)
    return records


def _remote_consumer_authorizations_by_wave(
    authorizations: Sequence[object],
    *,
    execution_plan: Mapping[str, Any],
) -> dict[int, Any]:
    from document_kv_cache.full_score_remote_control import (
        require_full_score_remote_consumer_evidence_authorization,
    )

    result: dict[int, Any] = {}
    for authorization in authorizations:
        validated = require_full_score_remote_consumer_evidence_authorization(
            authorization,
            execution_plan=execution_plan,
        )
        if validated.wave_index in result:
            raise ValueError("remote consumer authority duplicates a wave")
        result[validated.wave_index] = validated
    return result


def _validate_prior_wave_completion(
    record: Mapping[str, Any] | None,
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    replay_raw_evidence: bool,
    expected_wave_index: int,
    expected_execution_plan_sha256: str,
    remote_consumer_authorization: object | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("the prior wave must be reconciled before rendering this wave")
    if record.get("record_type") != FULL_SCORE_WAVE_COMPLETION_RECORD_TYPE:
        raise ValueError("prior-wave gate has the wrong record_type")
    if record.get("schema_version") != FULL_SCORE_WAVE_COMPLETION_SCHEMA_VERSION:
        raise ValueError("prior-wave gate has the wrong schema_version")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("prior-wave completion closure drift")
    if record.get("authorization_scope") != FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE:
        raise ValueError("prior-wave completion is not publication-authorizing")
    if record.get("wave_index") != expected_wave_index:
        raise ValueError("prior-wave completion is not the immediate predecessor")
    if record.get("execution_plan_sha256") != expected_execution_plan_sha256:
        raise ValueError("prior-wave completion execution-plan drift")
    if record.get("next_wave_authorized") is not True:
        raise ValueError("prior-wave completion does not authorize the next wave")
    bindings = record.get("governed_evidence_files")
    shard_ids = record.get("shard_ids")
    if not isinstance(bindings, list) or not isinstance(shard_ids, list):
        raise ValueError("prior-wave completion lacks governed evidence bindings")
    remote_records: dict[str, tuple[dict[str, Any], dict[str, Any]]] | None = None
    remote_bindings: dict[str, Mapping[str, Any]] = {}
    if remote_consumer_authorization is not None:
        remote_records = _remote_consumer_evidence_records(
            remote_consumer_authorization,
            completion_record=record,
            inventory=inventory,
            shard_plan=shard_plan,
            execution_plan=execution_plan,
            expected_wave_index=expected_wave_index,
            compact_artifact_resolver=compact_artifact_resolver,
        )
        remote_bindings = {
            _required_string(binding, "shard_id"): binding
            for binding in cast(
                Any,
                remote_consumer_authorization,
            ).evidence_bindings
        }
    elif compact_artifact_resolver is not None:
        raise TypeError("prior-wave compact CAS requires remote consumer authority")
    observed_ids: list[str] = []
    for raw_binding in bindings:
        binding = _json_mapping(raw_binding, "prior-wave governed evidence")
        shard_id = _required_string(binding, "shard_id")
        observed_ids.append(shard_id)
        if remote_records is None:
            evidence_file = _governed_existing_file(
                _required_string(binding, "evidence_path"),
                "prior-wave evidence path",
            )
            deletion_file = _governed_existing_file(
                _required_string(binding, "deletion_path"),
                "prior-wave deletion path",
            )
            if evidence_file.parent != deletion_file.parent:
                raise ValueError("prior-wave evidence/deletion directory binding drift")
            evidence_bytes = evidence_file.read_bytes()
            deletion_bytes = deletion_file.read_bytes()
            if sha256(evidence_bytes).hexdigest() != binding.get(
                "evidence_file_sha256"
            ):
                raise ValueError("prior-wave evidence file checksum drift")
            if sha256(deletion_bytes).hexdigest() != binding.get(
                "deletion_file_sha256"
            ):
                raise ValueError("prior-wave deletion file checksum drift")
            evidence = _json_object(evidence_bytes, "prior-wave evidence")
            deletion = _json_object(deletion_bytes, "prior-wave deletion")
        else:
            try:
                evidence, deletion = remote_records[shard_id]
            except KeyError as exc:
                raise ValueError("prior-wave remote evidence coverage drift") from exc
            remote_binding = remote_bindings[shard_id]
            if (
                binding.get("evidence_path") != remote_binding["evidence_uri"]
                or binding.get("deletion_path") != remote_binding["deletion_uri"]
                or binding.get("evidence_file_sha256")
                != remote_binding["evidence_file_sha256"]
                or binding.get("deletion_file_sha256")
                != remote_binding["deletion_file_sha256"]
            ):
                raise ValueError("prior-wave remote evidence binding drift")
        if (
            evidence.get("authorization_scope")
            != FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
            or evidence.get("closed_record_sha256") != _closed_record_sha256(evidence)
            or evidence.get("shard_id") != shard_id
            or evidence.get("wave_index") != expected_wave_index
            or evidence.get("execution_plan_sha256") != expected_execution_plan_sha256
            or deletion.get("authorization_scope")
            != FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
            or deletion.get("closed_record_sha256") != _closed_record_sha256(deletion)
            or deletion.get("shard_id") != shard_id
            or deletion.get("evidence_closed_record_sha256")
            != evidence.get("closed_record_sha256")
        ):
            raise ValueError("prior-wave governed evidence binding drift")
        if replay_raw_evidence:
            if remote_records is None:
                replayed_evidence, replayed_deletion = (
                    load_governed_full_score_shard_evidence(
                        evidence_file.parent,
                        inventory=inventory,
                        shard_plan=shard_plan,
                        execution_plan=execution_plan,
                        require_deletion=True,
                    )
                )
                if replayed_evidence != evidence or replayed_deletion != deletion:
                    raise ValueError("prior-wave raw/deletion evidence does not replay")
            else:
                _validate_shard_evidence_record(
                    evidence,
                    inventory_sha256=inventory.inventory_sha256,
                    shard_plan_sha256=_required_string(
                        shard_plan,
                        "closed_record_sha256",
                    ),
                )
    if observed_ids != sorted(cast(list[str], shard_ids)):
        raise ValueError("prior-wave governed evidence coverage drift")


def _validate_producer_phase_completion(
    record: Mapping[str, Any] | None,
    *,
    execution_plan: Mapping[str, Any],
    expected_wave_index: int,
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("consumer phase requires producer-phase completion")
    if record.get("record_type") != FULL_SCORE_PRODUCER_PHASE_COMPLETION_RECORD_TYPE:
        raise ValueError("producer-phase completion record_type drift")
    if record.get("schema_version") != (
        FULL_SCORE_PRODUCER_PHASE_COMPLETION_SCHEMA_VERSION
    ):
        raise ValueError("producer-phase completion schema drift")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("producer-phase completion closure drift")
    if record.get("authorization_scope") != FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE:
        raise ValueError("consumer phase rejects local-fixture producer completion")
    if record.get("execution_plan_sha256") != execution_plan.get(
        "closed_record_sha256"
    ):
        raise ValueError("producer-phase completion execution-plan drift")
    if record.get("wave_index") != expected_wave_index:
        raise ValueError("producer-phase completion targets a different wave")
    if record.get("consumer_phase_authorized") is not True:
        raise ValueError("producer phase did not authorize consumers")
    waves = cast(list[Mapping[str, Any]], execution_plan["waves"])
    expected_ids = sorted(cast(list[str], waves[expected_wave_index]["shard_ids"]))
    if record.get("shard_ids") != expected_ids:
        raise ValueError("producer-phase completion shard coverage drift")
    ready_shards = record.get("ready_shards")
    if not isinstance(ready_shards, list) or len(ready_shards) != len(expected_ids):
        raise ValueError("producer-phase completion ready-shard coverage drift")
    observed_ids = []
    observed_bytes = 0
    for raw_ready in ready_shards:
        ready = _json_mapping(raw_ready, "producer completion ready shard")
        observed_ids.append(_required_string(ready, "shard_id"))
        _require_sha256(
            ready.get("ready_record_sha256"),
            field_name="ready_record_sha256",
        )
        _require_sha256(
            ready.get("shard_items_sha256"),
            field_name="shard_items_sha256",
        )
        _require_sha256(
            ready.get("generator_artifact_contract_sha256"),
            field_name="generator_artifact_contract_sha256",
        )
        ready_bytes = ready.get("ready_bytes")
        if type(ready_bytes) is not int or ready_bytes <= 0:
            raise ValueError("producer-phase completion ready bytes are invalid")
        observed_bytes += ready_bytes
    if (
        observed_ids != expected_ids
        or record.get("total_ready_bytes") != observed_bytes
    ):
        raise ValueError("producer-phase completion compact projection drift")
    if observed_bytes > cast(
        int,
        waves[expected_wave_index]["max_backlog_bytes"],
    ):
        raise ValueError("producer-phase completion backlog cap drift")
    ready_files = record.get("ready_record_files")
    if not isinstance(ready_files, list) or len(ready_files) != len(expected_ids):
        raise ValueError("producer-phase completion lacks ready-record file bindings")
    file_ids: list[str] = []
    ready_by_id = {
        _required_string(item, "shard_id"): item
        for item in cast(list[Mapping[str, Any]], ready_shards)
    }
    for raw_file in ready_files:
        file_record = _json_mapping(raw_file, "producer ready-record file")
        shard_id = _required_string(file_record, "shard_id")
        file_ids.append(shard_id)
        _require_shared_dbfs_path(
            _required_string(file_record, "path"),
            "producer ready-record path",
        )
        _require_sha256(file_record.get("file_sha256"), field_name="ready file SHA-256")
        if file_record.get("ready_record_sha256") != ready_by_id.get(shard_id, {}).get(
            "ready_record_sha256"
        ):
            raise ValueError("producer ready-record compact/file binding drift")
        ready_file = _governed_existing_file(
            _required_string(file_record, "path"),
            "producer ready-record path",
        )
        if sha256(ready_file.read_bytes()).hexdigest() != file_record.get(
            "file_sha256"
        ):
            raise ValueError("producer ready-record file checksum drift")
        ready_record = _json_object(
            ready_file.read_bytes(),
            "producer ready-record file",
        )
        if (
            ready_record.get("closed_record_sha256")
            != ready_by_id.get(shard_id, {}).get("ready_record_sha256")
            or ready_record.get("closed_record_sha256")
            != _closed_record_sha256(ready_record)
            or ready_record.get("shard_id") != shard_id
            or ready_record.get("wave_index") != expected_wave_index
            or ready_record.get("execution_plan_sha256")
            != execution_plan.get("closed_record_sha256")
        ):
            raise ValueError("producer ready-record governed file binding drift")
    if sorted(file_ids) != expected_ids or len(set(file_ids)) != len(file_ids):
        raise ValueError("producer ready-record file coverage drift")


def _validate_governed_producer_ready_phase(
    completion: Mapping[str, Any],
    *,
    payloads: Sequence[Mapping[str, Any]],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
) -> None:
    """Recheck every ready tree before a publication consumer task is admitted."""

    compact_by_id = {
        _required_string(item, "shard_id"): item
        for item in cast(list[Mapping[str, Any]], completion["ready_shards"])
    }
    files_by_id = {
        _required_string(item, "shard_id"): item
        for item in cast(list[Mapping[str, Any]], completion["ready_record_files"])
    }
    observed: set[str] = set()
    for payload in payloads:
        for shard in cast(list[Mapping[str, Any]], payload["shards"]):
            shard_id = _required_string(shard, "shard_id")
            if shard_id in observed:
                raise ValueError("consumer phase duplicates a governed ready shard")
            observed.add(shard_id)
            ready_dir = _ready_shard_dir(payload, shard_id)
            ready = _validate_ready_shard(
                ready_dir,
                shard=shard,
                payload=payload,
                inventory=inventory,
                shard_plan=shard_plan,
                execution_plan=execution_plan,
            )
            compact = compact_by_id.get(shard_id)
            file_binding = files_by_id.get(shard_id)
            if compact is None or file_binding is None:
                raise ValueError("consumer phase lacks a governed ready-shard binding")
            ready_record_path = ready_dir / "ready-record.json"
            if (
                compact.get("ready_record_sha256") != ready.get("closed_record_sha256")
                or _cluster_path(_required_string(file_binding, "path"))
                != ready_record_path
                or sha256(ready_record_path.read_bytes()).hexdigest()
                != file_binding.get("file_sha256")
            ):
                raise ValueError("consumer phase governed ready tree binding drift")
    if observed != set(compact_by_id) or observed != set(files_by_id):
        raise ValueError("consumer phase governed ready-tree coverage drift")


def _validate_live_p90_budget_admission(
    record: Mapping[str, Any] | None,
    *,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    expected_execution_plan_sha256: str,
    expected_next_wave_index: int,
    expected_next_phase: str,
    expected_next_wave_reserved_gpu_hours: float,
    require_publication: bool,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("nonzero waves require a closed live P90 budget admission")
    validate_full_score_live_p90_budget_admission(execution_plan, record)
    expected_scope = (
        FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
        if require_publication
        else FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE
    )
    if record.get("authorization_scope") != expected_scope:
        raise ValueError(
            "publication rendering rejects local-fixture live P90 admissions"
            if require_publication
            else "local preview requires a nonauthorizing live P90 diagnostic"
        )
    if record.get("execution_plan_sha256") != expected_execution_plan_sha256:
        raise ValueError("live P90 budget admission execution-plan drift")
    if record.get("next_wave_index") != expected_next_wave_index:
        raise ValueError("live P90 budget admission targets a different wave")
    if record.get("admitted") is not True:
        raise ValueError("live P90 budget gate did not admit the next wave")
    if record.get("wave_boundary_active_zero") is not True:
        raise ValueError("live P90 budget gate was not computed at active zero")
    before = _required_mapping(record, "ledger_before")
    if before.get("active_reserved_gpu_hours") != 0:
        raise ValueError("live P90 budget gate has a nonzero active ledger")
    observed_reservation = _required_number(
        before,
        "next_wave_reserved_gpu_hours",
    )
    if abs(observed_reservation - expected_next_wave_reserved_gpu_hours) > 1e-12:
        raise ValueError("live P90 budget gate reservation drift")
    after = _required_mapping(record, "ledger_after_projection")
    if (
        after.get("aggregate_gate_passed") is not True
        or after.get("live_reservation_gate_passed") is not True
    ):
        raise ValueError("live P90 budget gate lacks both passed cap checks")
    if record.get("next_phase") != expected_next_phase:
        raise ValueError("live P90 budget admission targets a different phase")
    if not require_publication:
        return
    _require_nonempty(record.get("attempt_id"), "live P90 attempt_id")
    _require_sha256(
        record.get("next_submit_payload_sha256"),
        field_name="live P90 next submit payload SHA-256",
    )
    ledger_binding = _required_mapping(record, "ledger")
    _require_sha256(
        ledger_binding.get("ledger_path_sha256"),
        field_name="live P90 ledger path SHA-256",
    )
    prefix = databricks_ledger_prefix_from_record(
        _required_mapping(ledger_binding, "predecessor_prefix")
    )
    if ledger_binding.get("ledger_id") != prefix.ledger_id:
        raise ValueError("live P90 ledger identity/prefix drift")
    _require_sha256(
        ledger_binding.get("record_sha256"),
        field_name="live P90 ledger record SHA-256",
    )
    lineage = _required_mapping(record, "predecessor_lineage")
    if (
        lineage.get("ledger_path_sha256") != ledger_binding.get("ledger_path_sha256")
        or lineage.get("ledger_prefix") != prefix.to_record()
    ):
        raise ValueError("live P90 predecessor lineage drift")
    file_bindings = record.get("matched_block_files")
    blocks = record.get("completed_blocks")
    if not isinstance(file_bindings, list) or not isinstance(blocks, list):
        raise ValueError("live P90 gate lacks governed matched-block files")
    expected_blocks = {
        _required_string(_json_mapping(block, "completed block"), "shard_id"): block
        for block in blocks
    }
    observed_ids: set[str] = set()
    for raw_binding in file_bindings:
        binding = _json_mapping(raw_binding, "matched-block file binding")
        shard_id = _required_string(binding, "shard_id")
        if shard_id in observed_ids or shard_id not in expected_blocks:
            raise ValueError("live P90 matched-block file coverage drift")
        observed_ids.add(shard_id)
        block_file = _governed_compact_file(
            _required_string(binding, "path"),
            "matched billing block",
            compact_artifact_resolver,
        )
        if sha256(block_file.read_bytes()).hexdigest() != binding.get("file_sha256"):
            raise ValueError("live P90 matched-block file checksum drift")
        block = _json_object(block_file.read_bytes(), "matched billing block")
        if (
            block != expected_blocks[shard_id]
            or block.get("authorization_scope")
            != FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
            or block.get("closed_record_sha256") != binding.get("record_sha256")
        ):
            raise ValueError("live P90 matched-block record binding drift")
        phase_lineage = _required_mapping(block, "ledger_lineage")
        producer_lineage = _required_mapping(phase_lineage, "producer")
        consumer_lineage = _required_mapping(phase_lineage, "consumer")
        if (
            producer_lineage.get("ledger_path_sha256")
            != ledger_binding.get("ledger_path_sha256")
            or consumer_lineage.get("ledger_path_sha256")
            != ledger_binding.get("ledger_path_sha256")
            or consumer_lineage.get("predecessor_prefix")
            != producer_lineage.get("terminal_prefix")
        ):
            raise ValueError("live P90 matched-block ledger lineage drift")
    if observed_ids != set(expected_blocks):
        raise ValueError("live P90 matched-block file coverage is incomplete")


def _validate_complete_phase_payload_set(
    payloads: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> None:
    if not payloads:
        raise ValueError("phase payload set is empty")
    if phase not in {"producer", "consumer"}:
        raise ValueError("full-score phase must be producer or consumer")
    reference_wave = _json_mapping(payloads[0].get("wave"), "worker wave")
    if reference_wave.get("scheduling_mode") != "phased":
        raise ValueError("independent Databricks phase runs require phased scheduling")
    expected: dict[tuple[str, int], tuple[str, ...]] = {}
    assignments = reference_wave.get(f"{phase}_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("phase payload set has incomplete worker assignments")
    for raw_assignment in assignments:
        assignment = _json_mapping(raw_assignment, "wave assignment")
        worker_index = _required_int(assignment, "worker_index")
        shard_ids = assignment.get("shard_ids")
        if not isinstance(shard_ids, list) or not shard_ids:
            raise ValueError("wave assignment shard_ids must be non-empty")
        expected[(phase, worker_index)] = tuple(
            _require_nonempty(shard_id, "assignment shard_id") for shard_id in shard_ids
        )
    observed: dict[tuple[str, int], tuple[str, ...]] = {}
    execution_sha = _required_string(
        _required_mapping(payloads[0], "execution_plan"),
        "closed_record_sha256",
    )
    for payload in payloads:
        if _json_mapping(payload.get("wave"), "worker wave") != reference_wave:
            raise ValueError("worker payloads disagree on their execution wave")
        if (
            _required_string(
                _required_mapping(payload, "execution_plan"),
                "closed_record_sha256",
            )
            != execution_sha
        ):
            raise ValueError("worker payloads disagree on execution-plan identity")
        role = _required_string(payload, "role")
        if role != phase:
            raise ValueError("phase payload set mixes producer and consumer roles")
        worker_index = _required_int(payload, "worker_index")
        worker_key = (role, worker_index)
        if worker_key in observed:
            raise ValueError("wave payload set duplicates a role/worker")
        shards = payload.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("worker payload shard assignment is empty")
        observed[worker_key] = tuple(
            _required_string(_json_mapping(shard, "worker shard"), "shard_id")
            for shard in shards
        )
    if observed != expected:
        raise ValueError("phase submission requires every assigned worker exactly once")
    if len(observed) > FULL_SCORE_MAX_WORKERS:
        raise ValueError("phase exceeds sixteen concurrently runnable tasks")


def _balanced_shard_assignment(
    shards: Sequence[Mapping[str, Any]],
    *,
    worker_count: int,
    weight_field: str,
) -> list[dict[str, Any]]:
    loads = [0] * worker_count
    assigned: list[list[str]] = [[] for _index in range(worker_count)]
    for shard in sorted(
        shards,
        key=lambda item: (-cast(int, item[weight_field]), cast(str, item["shard_id"])),
    ):
        worker_index = min(range(worker_count), key=lambda index: (loads[index], index))
        assigned[worker_index].append(cast(str, shard["shard_id"]))
        loads[worker_index] += cast(int, shard[weight_field])
    return [
        {
            "shard_ids": shard_ids,
            "token_load": loads[worker_index],
            "weight_field": weight_field,
            "worker_index": worker_index,
        }
        for worker_index, shard_ids in enumerate(assigned)
        if shard_ids
    ]


def _ready_shard_dir(payload: Mapping[str, Any], shard_id: str) -> Path:
    return (
        _cluster_path(_required_string(payload, "durable_output_root"))
        / "ready"
        / f"wave-{_required_int(payload, 'wave_index'):03d}"
        / shard_id
    )


def _wave_shard_upper_bound(payload: Mapping[str, Any], shard_id: str) -> int:
    wave = _required_mapping(payload, "wave")
    shards = wave.get("shards")
    if not isinstance(shards, list):
        raise ValueError("worker wave shards must be an array")
    for raw_shard in shards:
        if isinstance(raw_shard, Mapping) and raw_shard.get("shard_id") == shard_id:
            value = raw_shard.get("ready_bytes_upper_bound")
            if type(value) is not int or value <= 0:
                raise ValueError("ready-shard byte upper bound is invalid")
            return value
    raise ValueError("worker shard is absent from its execution wave")


def _wait_for_complete_ready_wave(
    payload: Mapping[str, Any],
    *,
    shards: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Apply backpressure and wait only for this consumer's assigned ready set."""

    wave = _required_mapping(payload, "wave")
    if cast(int, wave["ready_bytes_upper_bound"]) > cast(
        int, wave["max_backlog_bytes"]
    ):
        raise RuntimeError("planned wave exceeds the durable backlog cap")
    runtime = _runtime_from_record(_required_mapping(payload, "runtime"))
    deadline = time.monotonic() + runtime.generator_timeout_seconds
    pending = {
        _required_string(shard, "shard_id")
        for shard in (
            cast(list[Mapping[str, Any]], payload["shards"])
            if shards is None
            else shards
        )
    }
    while pending and time.monotonic() < deadline:
        next_pending: set[str] = set()
        for shard_id in pending:
            ready_record_path = (
                _ready_shard_dir(payload, shard_id) / "ready-record.json"
            )
            _require_no_symlink_ancestors(
                ready_record_path,
                label="ready-shard wait path",
                include_leaf=True,
            )
            if not ready_record_path.is_file():
                next_pending.add(shard_id)
        pending = next_pending
        if pending:
            time.sleep(5)
    if pending:
        raise TimeoutError(f"timed out waiting for ready shards: {sorted(pending)}")


def _validate_ready_shard(
    ready_dir: Path,
    *,
    shard: Mapping[str, Any],
    payload: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
) -> dict[str, Any]:
    _require_directory_no_follow(ready_dir, "ready-shard directory")
    ready_record_path = ready_dir / "ready-record.json"
    _require_regular_file_no_follow(ready_record_path, "ready-shard record")
    record = _json_object(ready_record_path.read_bytes(), "ready shard")
    if record.get("record_type") != FULL_SCORE_READY_SHARD_RECORD_TYPE:
        raise ValueError("unsupported ready-shard record_type")
    if record.get("schema_version") != FULL_SCORE_READY_SHARD_SCHEMA_VERSION:
        raise ValueError("unsupported ready-shard schema_version")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("ready-shard closure drift")
    expected = {
        "execution_plan_sha256": execution_plan.get("closed_record_sha256"),
        "inventory_sha256": inventory.inventory_sha256,
        "shard_id": shard.get("shard_id"),
        "shard_items_sha256": shard.get("items_sha256"),
        "shard_plan_sha256": shard_plan.get("closed_record_sha256"),
        "wave_index": payload.get("wave_index"),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("ready-shard identity drift")
    runtime = _runtime_from_record(_required_mapping(payload, "runtime"))
    if record.get("generator_artifact_contract") != (
        _generator_artifact_contract_record(runtime)
    ):
        raise ValueError("ready-shard generator artifact-contract drift")
    runtime_verification = _validate_runtime_verification_binding(
        record.get("runtime_verification")
    )
    expected_artifacts = _runtime_artifact_binding(
        runtime,
        _required_mapping(payload, "bootstrap_artifacts"),
    )
    if runtime_verification.get("artifacts") != expected_artifacts:
        raise ValueError("ready-shard runtime artifact binding drift")
    producer_hardware = _required_mapping(record, "producer_hardware")
    expected_hardware = {
        "compute_capability": "8.9",
        "gpu_count": 1,
        "gpu_name": FULL_SCORE_PRODUCER_GPU_NAME,
        "hardware_target": FULL_SCORE_PRODUCER_HARDWARE_TARGET,
        "node_type_id": FULL_SCORE_PRODUCER_NODE_TYPE_ID,
    }
    if any(
        producer_hardware.get(key) != value for key, value in expected_hardware.items()
    ):
        raise ValueError("ready-shard producer hardware drift")
    total_memory = producer_hardware.get("total_memory_bytes")
    if type(total_memory) is not int or total_memory < 40 * 1024**3:
        raise ValueError("ready-shard L40S memory evidence is invalid")
    files = _closed_file_tree(ready_dir, exclude={"ready-record.json"})
    if files != record.get("files") or _canonical_sha256(files) != record.get(
        "files_sha256"
    ):
        raise ValueError("ready-shard file closure drift")
    actual_bytes = sum(cast(int, item["byte_count"]) for item in files)
    if actual_bytes != record.get("ready_bytes"):
        raise ValueError("ready-shard byte count drift")
    if actual_bytes > _wave_shard_upper_bound(payload, cast(str, shard["shard_id"])):
        raise ValueError("ready-shard bytes exceed the hard planned bound")
    return record


def _closed_file_tree(
    root: Path,
    *,
    exclude: set[str] | None = None,
) -> list[dict[str, Any]]:
    _require_directory_no_follow(root, "ready-shard closure root")
    excluded = exclude or set()
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("ready-shard closure rejects symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        raw = path.read_bytes()
        files.append(
            {
                "byte_count": len(raw),
                "relative_path": relative,
                "sha256": sha256(raw).hexdigest(),
            }
        )
    if not files:
        raise ValueError("ready-shard file closure is empty")
    return files


def _rewrite_json_tree_paths(root: Path, *, old_root: Path, new_root: Path) -> None:
    _require_directory_no_follow(root, "ready-shard rewrite root")
    old = str(old_root)
    new = str(new_root)

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith(old):
                return new + value[len(old) :]
            if value.startswith(f"file://{old}"):
                return f"file://{new}" + value[len(f"file://{old}") :]
            return value
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("ready-shard rewrite rejects symlinks")
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            payload = "".join(
                json.dumps(rewrite(row), ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ).encode("utf-8")
        else:
            payload = _canonical_pretty_json_bytes(
                cast(
                    Mapping[str, Any],
                    rewrite(json.loads(path.read_text(encoding="utf-8"))),
                )
            )
        path.write_bytes(payload)


def _load_ready_examples(paths: Mapping[str, Path]) -> dict[tuple[str, str], Any]:
    examples = {}
    for dataset, path in sorted(paths.items()):
        for index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError("ready input row must be an object")
            example = _example_from_record(
                record,
                default_dataset=dataset,
                record_index=index,
                require_dataset=True,
            )
            key = (dataset, example.example_id)
            if key in examples:
                raise ValueError("ready inputs contain a duplicate identity")
            examples[key] = example
    return examples


def _shard_items_by_key(
    shard: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    raw_items = shard.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("shard items must be an array")
    values: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_item in raw_items:
        item = _json_mapping(raw_item, "shard item")
        key = (_required_string(item, "dataset"), _required_string(item, "example_id"))
        if key in values:
            raise ValueError("shard contains a duplicate identity")
        values[key] = item
    if len(values) != shard.get("item_count"):
        raise ValueError("shard item count drift")
    return values


def _validate_shard_evidence_record(
    record: Mapping[str, Any],
    *,
    inventory_sha256: str,
    shard_plan_sha256: str,
) -> None:
    if record.get("record_type") != FULL_SCORE_SHARD_EVIDENCE_RECORD_TYPE:
        raise ValueError("unsupported shard evidence record_type")
    if record.get("schema_version") != FULL_SCORE_SHARD_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported shard evidence schema_version")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("shard evidence closure drift")
    if record.get("durable_evidence_committed") is not True:
        raise ValueError("shard evidence was not durably committed")
    if record.get("inventory_sha256") != inventory_sha256:
        raise ValueError("shard evidence inventory drift")
    if record.get("shard_plan_sha256") != shard_plan_sha256:
        raise ValueError("shard evidence plan drift")
    if not _json_type_exact_equal(
        record.get("scorers"),
        _scorer_contract_record(),
    ):
        raise ValueError("shard evidence scorer/parser drift")
    if not _json_type_exact_equal(
        record.get("protocol"),
        _full_score_protocol_record(),
    ):
        raise ValueError("shard evidence full-score protocol drift")
    _validate_runtime_verification_binding(record.get("runtime_verification"))
    scope = record.get("authorization_scope")
    if scope not in {
        FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE,
        FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
    }:
        raise ValueError("shard evidence authorization_scope drift")
    if scope == FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE:
        for field_name in (
            "execution_plan_sha256",
            "ready_shard_sha256",
            "shard_items_sha256",
        ):
            _require_sha256(record.get(field_name), field_name=field_name)
        lifecycle = record.get("lifecycle_before_delete")
        if lifecycle != [
            "verify_ready_shard",
            "baseline_inference",
            "vanilla_inference",
            "validate_paired_outputs",
        ]:
            raise ValueError("publication shard evidence lifecycle drift")
        preserved = record.get("preserved_files")
        if not isinstance(preserved, Mapping):
            raise ValueError("publication shard evidence lacks preserved files")
        proof = _required_mapping(record, "connector_proof")
        if (
            proof.get("record_type") != FULL_SCORE_CONNECTOR_PROOF_RECORD_TYPE
            or proof.get("schema_version") != FULL_SCORE_CONNECTOR_PROOF_SCHEMA_VERSION
            or proof.get("closed_record_sha256") != _closed_record_sha256(proof)
        ):
            raise ValueError("publication connector proof closure drift")


def _validate_full_score_deletion_attestation(
    record: Mapping[str, Any],
    *,
    evidence_record: Mapping[str, Any],
    execution_plan_sha256: str,
    shard_id: str,
    wave_index: int,
) -> None:
    if record.get("record_type") != FULL_SCORE_DELETION_ATTESTATION_RECORD_TYPE:
        raise ValueError("unsupported deletion-attestation record_type")
    if record.get("schema_version") != FULL_SCORE_DELETION_ATTESTATION_SCHEMA_VERSION:
        raise ValueError("unsupported deletion-attestation schema_version")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("deletion-attestation closure drift")
    expected = {
        "authorization_scope": FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE,
        "evidence_closed_record_sha256": evidence_record.get("closed_record_sha256"),
        "execution_plan_sha256": execution_plan_sha256,
        "ready_shard_sha256": evidence_record.get("ready_shard_sha256"),
        "shard_id": shard_id,
        "wave_index": wave_index,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("deletion-attestation evidence identity drift")
    if record.get("lifecycle") != [
        "verify_ready_shard",
        "baseline_inference",
        "vanilla_inference",
        "validate_paired_outputs",
        "commit_durable_evidence",
        "delete_ephemeral_q8_kv",
    ]:
        raise ValueError("deletion-attestation lifecycle drift")


def _validate_governed_ready_manifest_replay(
    record: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    shard: Mapping[str, Any],
    resolved_files: Mapping[str, Path],
    datasets: Sequence[str],
) -> None:
    """Bind preserved request files to the exact producer-ready closure."""

    if record.get("record_type") != FULL_SCORE_READY_SHARD_RECORD_TYPE:
        raise ValueError("preserved ready-shard record_type drift")
    if record.get("schema_version") != FULL_SCORE_READY_SHARD_SCHEMA_VERSION:
        raise ValueError("preserved ready-shard schema_version drift")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("preserved ready-shard closure drift")
    expected_identity = {
        "execution_plan_sha256": execution_plan.get("closed_record_sha256"),
        "inventory_sha256": inventory.inventory_sha256,
        "shard_id": shard.get("shard_id"),
        "shard_items_sha256": shard.get("items_sha256"),
        "shard_plan_sha256": shard_plan.get("closed_record_sha256"),
        "wave_index": evidence.get("wave_index"),
    }
    if any(record.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("governed ready-shard identity drift")
    if record.get("closed_record_sha256") != evidence.get("ready_shard_sha256"):
        raise ValueError("governed ready-shard evidence binding drift")
    ready_runtime = _validate_runtime_verification_binding(
        record.get("runtime_verification")
    )
    evidence_runtime = _validate_runtime_verification_binding(
        evidence.get("runtime_verification")
    )
    if ready_runtime != evidence_runtime:
        raise ValueError("governed ready/evidence runtime verification drift")
    if record.get("lifecycle") != ["generate_q8_kv", "commit_ready_shard"]:
        raise ValueError("governed ready-shard lifecycle drift")
    expected_contract = {
        "align_bytes": 4096,
        "cache_axis_order": "head_major",
        "cache_method": "vanilla_prefill",
        "dtype": FULL_SCORE_KV_DTYPE,
        "generator_factory": FULL_SCORE_GENERATOR_FACTORY,
        "generator_version": FULL_SCORE_GENERATOR_VERSION,
        "model_id": FULL_SCORE_MODEL_ID,
        "model_revision": MAIN_LATENCY_TOKENIZER_REVISION,
        "pre_rope": True,
        "quantization": FULL_SCORE_GENERATOR_QUANTIZATION,
        "quantization_config": dict(FULL_SCORE_GENERATOR_QUANTIZATION_CONFIG),
        "quantization_config_sha256": _canonical_sha256(
            FULL_SCORE_GENERATOR_QUANTIZATION_CONFIG
        ),
        "segment_per_document": True,
        "segmented": True,
        "storage_layout": "separate_key_value",
        "tokenizer_id": MAIN_LATENCY_TOKENIZER_ID,
        "tokenizer_revision": MAIN_LATENCY_TOKENIZER_REVISION,
        "trust_remote_code": False,
        "vllm_bitsandbytes_loader_member": FULL_SCORE_VLLM_BNB_LOADER_MEMBER,
        "vllm_bitsandbytes_loader_sha256": FULL_SCORE_VLLM_BNB_LOADER_SHA256,
    }
    if record.get("generator_artifact_contract") != expected_contract:
        raise ValueError("governed ready-shard generator contract drift")
    producer_hardware = _required_mapping(record, "producer_hardware")
    expected_hardware = {
        "compute_capability": "8.9",
        "gpu_count": 1,
        "gpu_name": FULL_SCORE_PRODUCER_GPU_NAME,
        "hardware_target": FULL_SCORE_PRODUCER_HARDWARE_TARGET,
        "node_type_id": FULL_SCORE_PRODUCER_NODE_TYPE_ID,
    }
    if any(
        producer_hardware.get(key) != value for key, value in expected_hardware.items()
    ):
        raise ValueError("governed ready-shard producer hardware drift")
    total_memory = producer_hardware.get("total_memory_bytes")
    if type(total_memory) is not int or total_memory < 40 * 1024**3:
        raise ValueError("governed ready-shard L40S memory evidence is invalid")

    wave_index = _required_int(evidence, "wave_index")
    wave = _json_mapping(
        cast(list[Mapping[str, Any]], execution_plan["waves"])[wave_index],
        "governed ready-shard wave",
    )
    shard_id = _required_string(shard, "shard_id")
    assignment = next(
        (
            _json_mapping(item, "producer assignment")
            for item in cast(list[Mapping[str, Any]], wave["producer_assignments"])
            if shard_id in cast(list[str], item.get("shard_ids", []))
        ),
        None,
    )
    if assignment is None or record.get("worker_index") != assignment.get(
        "worker_index"
    ):
        raise ValueError("governed ready-shard producer-task identity drift")
    upper_bound = shard.get("ready_bytes_upper_bound")
    if (
        type(upper_bound) is not int
        or record.get("ready_bytes_upper_bound") != upper_bound
    ):
        raise ValueError("governed ready-shard backlog bound drift")

    raw_file_records = record.get("files")
    if not isinstance(raw_file_records, list) or not raw_file_records:
        raise ValueError("governed ready-shard file closure is empty")
    if record.get("files_sha256") != _canonical_sha256(raw_file_records):
        raise ValueError("governed ready-shard file-list closure drift")
    files_by_path: dict[str, dict[str, Any]] = {}
    ready_bytes = 0
    for raw_file_record in raw_file_records:
        file_record = _json_mapping(raw_file_record, "ready-shard file")
        if set(file_record) != {"byte_count", "relative_path", "sha256"}:
            raise ValueError("governed ready-shard file schema drift")
        relative = _required_string(file_record, "relative_path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("governed ready-shard file path escapes its root")
        if relative in files_by_path:
            raise ValueError("governed ready-shard file closure duplicates a path")
        byte_count = file_record.get("byte_count")
        if type(byte_count) is not int or byte_count <= 0:
            raise ValueError("governed ready-shard file byte count is invalid")
        _require_sha256(file_record.get("sha256"), field_name="ready file sha256")
        files_by_path[relative] = file_record
        ready_bytes += byte_count
    if ready_bytes != record.get("ready_bytes") or ready_bytes > upper_bound:
        raise ValueError("governed ready-shard byte total drift")

    preserved = _required_mapping(evidence, "preserved_files")
    ready_paths: dict[str, str] = {}
    for dataset in datasets:
        ready_paths.update(
            {
                f"input_{dataset}": f"inputs/{dataset}.jsonl",
                f"enriched_{dataset}": f"enriched/{dataset}.jsonl",
                f"handoff_manifest_{dataset}": f"manifests/{dataset}.json",
            }
        )
        q8_prefix = f"q8-kv/{dataset}/"
        if not any(path.startswith(q8_prefix) for path in files_by_path):
            raise ValueError("governed ready-shard closure lacks generated Q8 files")
    for evidence_name, ready_path in ready_paths.items():
        ready_file = files_by_path.get(ready_path)
        if ready_file is None:
            raise ValueError("governed ready-shard closure lacks a request input")
        preserved_file = _json_mapping(
            preserved.get(evidence_name),
            f"preserved_files.{evidence_name}",
        )
        if any(
            preserved_file.get(field_name) != ready_file.get(field_name)
            for field_name in ("byte_count", "sha256")
        ):
            raise ValueError("preserved request file differs from ready-shard closure")
        if resolved_files.get(evidence_name) is None:
            raise ValueError("preserved request file was not resolved")


def _load_governed_ready_source_records(
    paths: Mapping[str, Path],
    *,
    shard: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], Any]]:
    """Replay exact source rows, including NIAH cell metadata, from inventory hashes."""

    planned = _shard_items_by_key(shard)
    source_records: dict[tuple[str, str], dict[str, Any]] = {}
    examples: dict[tuple[str, str], Any] = {}
    for dataset, path in sorted(paths.items()):
        raw = path.read_bytes()
        if not raw or not raw.endswith(b"\n"):
            raise ValueError("governed ready input must be non-empty newline JSONL")
        lines = raw[:-1].split(b"\n")
        if any(not line for line in lines):
            raise ValueError("governed ready input contains an empty row")
        for index, line in enumerate(lines, start=1):
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("governed ready input contains invalid JSON") from exc
            record = _json_mapping(value, "governed ready input row")
            if _TRANSFER_FIELDS.intersection(record):
                raise ValueError(
                    "governed source input unexpectedly contains transfer fields"
                )
            example = _example_from_record(
                record,
                default_dataset=dataset,
                record_index=index,
                require_dataset=True,
            )
            key = (dataset, example.example_id)
            item = planned.get(key)
            if item is None or example.dataset != dataset:
                raise ValueError("governed ready input contains an unplanned identity")
            if key in source_records:
                raise ValueError("governed ready input duplicates an identity")
            if _canonical_sha256(record) != item.get("source_record_sha256"):
                raise ValueError("governed ready input source-record hash drift")
            source_records[key] = record
            examples[key] = example
    if set(source_records) != set(planned):
        raise ValueError("governed ready inputs have partial shard coverage")
    return source_records, examples


def _read_governed_jsonl_rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{label} must be non-empty newline JSONL")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw[:-1].split(b"\n"), start=1):
        if not line:
            raise ValueError(f"{label} contains an empty row")
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} row {index} is invalid JSON") from exc
        rows.append(_json_mapping(value, f"{label} row {index}"))
    return rows


def _validate_governed_handoff_replay(
    *,
    source_records: Mapping[tuple[str, str], Mapping[str, Any]],
    enriched_paths: Mapping[str, Path],
    manifest_paths: Mapping[str, Path],
    paired_examples: Sequence[Mapping[str, Any]],
) -> None:
    """Replay source -> manifest -> enriched request -> measured artifact identity."""

    pair_artifacts: dict[tuple[str, str], str] = {}
    for raw_pair in paired_examples:
        pair = _json_mapping(raw_pair, "paired handoff example")
        key = (_required_string(pair, "dataset"), _required_string(pair, "example_id"))
        methods = _required_mapping(pair, "methods")
        vanilla = _required_mapping(methods, "vanilla_prefill")
        artifact_id = _required_string(vanilla, "artifact_id")
        if key in pair_artifacts:
            raise ValueError("paired handoff evidence duplicates an identity")
        pair_artifacts[key] = artifact_id
    if set(pair_artifacts) != set(source_records):
        raise ValueError("paired handoff evidence has partial source coverage")

    observed: set[tuple[str, str]] = set()
    for dataset, enriched_path in sorted(enriched_paths.items()):
        manifest_path = manifest_paths.get(dataset)
        if manifest_path is None:
            raise ValueError("governed handoff manifest coverage is incomplete")
        manifest = read_benchmark_handoff_manifest_json(manifest_path)
        entries = {entry.key: entry for entry in manifest.entries}
        expected_dataset_keys = {key for key in source_records if key[0] == dataset}
        if set(entries) != expected_dataset_keys:
            raise ValueError("governed handoff manifest identity coverage drift")
        enriched_rows = _read_governed_jsonl_rows(
            enriched_path,
            label=f"governed {dataset} enriched input",
        )
        enriched_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in enriched_rows:
            key = (
                _required_string(row, "dataset"),
                _required_string(row, "example_id"),
            )
            if key[0] != dataset or key in enriched_by_key:
                raise ValueError("governed enriched input identity drift")
            enriched_by_key[key] = row
        if set(enriched_by_key) != expected_dataset_keys:
            raise ValueError("governed enriched input identity coverage drift")
        for key in sorted(expected_dataset_keys):
            entry = entries[key]
            if not entry.artifact_id or entry.cache_method != "vanilla_prefill":
                raise ValueError("governed handoff manifest lacks an artifact identity")
            expected_params = entry.kv_transfer_params()
            if expected_params.get(DOCUMENT_KV_ARTIFACT_ID_PARAM) != entry.artifact_id:
                raise ValueError("governed handoff manifest artifact identity drift")
            enriched = dict(enriched_by_key[key])
            arm_params = enriched.pop("arm_kv_transfer_params", None)
            if not isinstance(arm_params, Mapping) or set(arm_params) != {
                FULL_SCORE_VANILLA_ARM_ID
            }:
                raise ValueError("governed enriched input arm mapping drift")
            actual_params = _json_mapping(
                arm_params[FULL_SCORE_VANILLA_ARM_ID],
                "governed enriched Vanilla handoff",
            )
            if actual_params != expected_params:
                raise ValueError(
                    "governed enriched input differs from handoff manifest"
                )
            expected_source = dict(source_records[key])
            expected_source.setdefault("dataset", dataset)
            if enriched != expected_source:
                raise ValueError("governed enriched input differs from source row")
            if pair_artifacts[key] != entry.artifact_id:
                raise ValueError(
                    "measured Vanilla artifact differs from handoff manifest"
                )
            observed.add(key)
    if set(manifest_paths) != set(enriched_paths) or observed != set(source_records):
        raise ValueError("governed handoff replay has partial dataset coverage")


def load_governed_full_score_shard_evidence(
    evidence_dir: str | Path,
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    require_deletion: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read, checksum, and replay one publication shard evidence directory."""

    directory = _cluster_path(
        _require_shared_dbfs_path(evidence_dir, "governed evidence directory")
    )
    _require_directory_no_follow(
        directory,
        "governed evidence DBFS directory",
    )
    evidence_path = directory / "evidence.json"
    _require_regular_file_no_follow(
        evidence_path,
        "governed evidence record",
    )
    evidence = _json_object(evidence_path.read_bytes(), "governed shard evidence")
    _validate_shard_evidence_record(
        evidence,
        inventory_sha256=inventory.inventory_sha256,
        shard_plan_sha256=_required_string(shard_plan, "closed_record_sha256"),
    )
    if (
        evidence.get("authorization_scope")
        != FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
    ):
        raise ValueError("publication aggregation rejects local-fixture shard evidence")
    execution_sha = _required_string(execution_plan, "closed_record_sha256")
    if evidence.get("execution_plan_sha256") != execution_sha:
        raise ValueError("governed shard evidence execution-plan drift")
    shard_id = _required_string(evidence, "shard_id")
    wave_index = _required_int(evidence, "wave_index")
    waves = cast(list[Mapping[str, Any]], execution_plan.get("waves"))
    if not 0 <= wave_index < len(waves):
        raise ValueError("governed shard evidence wave index is invalid")
    shard = next(
        (
            candidate
            for candidate in cast(list[Mapping[str, Any]], waves[wave_index]["shards"])
            if candidate.get("shard_id") == shard_id
        ),
        None,
    )
    if shard is None or evidence.get("shard_items_sha256") != shard.get("items_sha256"):
        raise ValueError("governed shard evidence is not planned")
    datasets = sorted(
        {
            cast(str, item["dataset"])
            for item in cast(list[Mapping[str, Any]], shard["items"])
        }
    )
    required_preserved = {
        "baseline_raw_output",
        "connector_telemetry",
        "ready_shard_manifest",
        "runtime_telemetry",
        "server_log",
        "vanilla_raw_output",
        *(f"input_{dataset}" for dataset in datasets),
        *(f"enriched_{dataset}" for dataset in datasets),
        *(f"handoff_manifest_{dataset}" for dataset in datasets),
    }
    preserved = _required_mapping(evidence, "preserved_files")
    if set(preserved) != required_preserved:
        raise ValueError("governed shard preserved-file coverage drift")
    resolved_files: dict[str, Path] = {}
    for name, raw_file_record in preserved.items():
        file_record = _json_mapping(raw_file_record, f"preserved_files.{name}")
        relative = _required_string(file_record, "relative_path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("preserved evidence path escapes its shard directory")
        path = directory / relative_path
        _require_regular_file_no_follow(path, "preserved evidence file")
        if _file_record(path, relative_to=directory) != file_record:
            raise ValueError("preserved evidence file checksum drift")
        resolved_files[str(name)] = path
    ready_manifest = _json_object(
        resolved_files["ready_shard_manifest"].read_bytes(),
        "preserved ready-shard manifest",
    )
    _validate_governed_ready_manifest_replay(
        ready_manifest,
        evidence=evidence,
        inventory=inventory,
        shard_plan=shard_plan,
        execution_plan=execution_plan,
        shard=shard,
        resolved_files=resolved_files,
        datasets=datasets,
    )
    input_paths = {dataset: resolved_files[f"input_{dataset}"] for dataset in datasets}
    enriched_paths = {
        dataset: resolved_files[f"enriched_{dataset}"] for dataset in datasets
    }
    source_records, examples = _load_governed_ready_source_records(
        input_paths,
        shard=shard,
    )
    vanilla_examples = _load_ready_examples(enriched_paths)
    paired = validate_paired_full_score_outputs(
        _json_object(
            resolved_files["baseline_raw_output"].read_bytes(),
            "governed baseline output",
        ),
        _json_object(
            resolved_files["vanilla_raw_output"].read_bytes(),
            "governed Vanilla output",
        ),
        shard=shard,
        examples=examples,
        vanilla_examples=vanilla_examples,
    )
    if list(paired) != evidence.get("paired_examples"):
        raise ValueError("governed paired outputs do not replay")
    _validate_governed_handoff_replay(
        source_records=source_records,
        enriched_paths=enriched_paths,
        manifest_paths={
            dataset: resolved_files[f"handoff_manifest_{dataset}"]
            for dataset in datasets
        },
        paired_examples=paired,
    )
    connector_proof = build_full_score_connector_proof(
        resolved_files["connector_telemetry"],
        paired_examples=paired,
        shard=shard,
    )
    if connector_proof != evidence.get("connector_proof"):
        raise ValueError("governed connector proof does not replay")
    deletion_path = directory / "deletion-attestation.json"
    deletion: dict[str, Any] | None = None
    if deletion_path.exists():
        _require_regular_file_no_follow(
            deletion_path,
            "deletion attestation",
        )
        deletion = _json_object(deletion_path.read_bytes(), "deletion attestation")
        _validate_full_score_deletion_attestation(
            deletion,
            evidence_record=evidence,
            execution_plan_sha256=execution_sha,
            shard_id=shard_id,
            wave_index=wave_index,
        )
    elif require_deletion:
        raise ValueError("governed shard evidence lacks deletion attestation")
    if require_deletion:
        durable_root = directory.parents[2]
        ready_dir = durable_root / "ready" / f"wave-{wave_index:03d}" / shard_id
        _require_no_symlink_ancestors(
            ready_dir,
            label="governed deleted ready-shard path",
            include_leaf=True,
        )
        if ready_dir.exists() or ready_dir.is_symlink():
            raise ValueError("governed deletion attestation leaves ready Q8 artifacts")
    return evidence, deletion


def _read_bound_json(
    uri: str,
    expected_sha256: str,
    *,
    closure_digest: bool,
) -> dict[str, Any]:
    path = _governed_existing_file(uri, "bound JSON")
    raw = path.read_bytes()
    record = _json_object(raw, uri)
    observed = (
        record.get("closed_record_sha256")
        if closure_digest
        else sha256(raw).hexdigest()
    )
    if observed != expected_sha256:
        raise ValueError(f"bound JSON hash drift: {uri}")
    return record


def _subprocess_command_runner(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path | None = None,
) -> None:
    completed = subprocess.run(
        list(argv),
        env=dict(env),
        cwd=None if cwd is None else str(cwd),
        timeout=_command_timeout(argv),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"full-score subprocess exited {completed.returncode}: {list(argv)!r}"
        )


def _command_timeout(argv: Sequence[str]) -> float:
    if not argv:
        raise ValueError("full-score subprocess command must not be empty")
    return float(FULL_SCORE_DATABRICKS_TASK_TIMEOUT_SECONDS)


def _file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"preserved evidence file is empty: {path}")
    return {
        "byte_count": len(raw),
        "relative_path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256(raw).hexdigest(),
    }


def _durable_copy(source: Path, destination: Path) -> None:
    _require_no_symlink_ancestors(
        source,
        label="durable evidence source",
        include_leaf=True,
    )
    _require_regular_file_no_follow(source, "durable evidence source")
    _require_no_symlink_ancestors(
        destination,
        label="durable evidence destination",
        include_leaf=True,
    )
    if destination.exists():
        raise FileExistsError(f"durable evidence file already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(
        destination.parent,
        label="durable evidence destination directory",
        include_leaf=True,
    )
    temporary = destination.with_name(f".{destination.name}.pending-{os.getpid()}")
    with source.open("rb") as reader, temporary.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    _rename_file_no_follow(
        temporary,
        destination,
        replace_existing=False,
    )
    _fsync_directory(destination.parent)
    if (
        sha256(source.read_bytes()).hexdigest()
        != sha256(destination.read_bytes()).hexdigest()
    ):
        raise RuntimeError("durable evidence copy checksum mismatch")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    _require_no_symlink_ancestors(
        path,
        label="atomic output path",
        include_leaf=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(
        path.parent,
        label="atomic output directory",
        include_leaf=True,
    )
    temporary = path.with_name(f".{path.name}.pending-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _rename_file_no_follow(
        temporary,
        path,
        replace_existing=True,
    )
    _fsync_directory(path.parent)


def _exclusive_write_bytes(path: Path, payload: bytes) -> None:
    """Publish complete immutable bytes without ever replacing a prior target."""

    _require_no_symlink_ancestors(
        path,
        label="exclusive output path",
        include_leaf=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(
        path.parent,
        label="exclusive output directory",
        include_leaf=True,
    )
    temporary = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    parent_descriptor = _open_directory_no_symlinks(path.parent)
    try:
        temporary_status = os.stat(
            temporary.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(temporary_status.st_mode):
            raise ValueError("exclusive output staging path must be a regular file")
        try:
            # Hard-link publication is atomic and fails with FileExistsError if a
            # duplicate submission already owns the immutable target name.
            os.link(
                temporary.name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        finally:
            os.unlink(temporary.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _write_or_require_exact_bytes(
    path: Path,
    payload: bytes,
    *,
    field_name: str,
) -> None:
    """Create immutable evidence once or require the already-created bytes."""

    try:
        _exclusive_write_bytes(path, payload)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"{field_name} binding drift") from None


def _fsync_file_tree(root: Path) -> None:
    """Flush a fresh ready-shard tree before publishing its commit record."""

    _require_no_symlink_ancestors(
        root,
        label="durable ready-shard tree",
        include_leaf=True,
    )
    directories = {root}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("durable ready-shard trees reject symlinks")
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            directories.add(path.parent)
        elif path.is_dir():
            directories.add(path)
    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    _require_no_symlink_ancestors(
        path,
        label="directory durability target",
        include_leaf=True,
    )
    descriptor = _open_directory_no_symlinks(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _encoded_ids(tokenizer: MainLatencyTokenizer, text: str) -> tuple[int, ...]:
    values = tokenizer.encode(text, add_special_tokens=MAIN_LATENCY_ADD_SPECIAL_TOKENS)
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("tokenizer.encode() must return a sequence")
    token_ids = tuple(values)
    if not token_ids or any(
        type(value) is not int or not 0 <= value < 2**64 for value in token_ids
    ):
        raise ValueError("tokenizer returned invalid token IDs")
    return token_ids


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = sha256(b"cachet.token_ids.uint64be.v1\0")
    digest.update(len(token_ids).to_bytes(8, "big"))
    for token_id in token_ids:
        digest.update(token_id.to_bytes(8, "big"))
    return digest.hexdigest()


def _segment_ids_sha256(segments: Sequence[Sequence[int]]) -> str:
    digest = sha256(b"cachet.segment_token_ids.uint64be.v1\0")
    digest.update(len(segments).to_bytes(8, "big"))
    for segment in segments:
        digest.update(len(segment).to_bytes(8, "big"))
        for token_id in segment:
            digest.update(token_id.to_bytes(8, "big"))
    return digest.hexdigest()


def _validate_hashed_install_spec(
    value: str,
    *,
    project: str,
    expected_sha256: str,
) -> None:
    if not value.startswith(f"{project} @ file://") or "#sha256=" not in value:
        raise ValueError(
            f"{project} install spec must be a hashed local PEP 508 reference"
        )
    digest = value.rsplit("#sha256=", maxsplit=1)[-1]
    _require_sha256(digest, field_name=f"{project} install spec SHA-256")
    if digest != expected_sha256:
        raise ValueError(f"{project} install spec SHA-256 drift")


def _json_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    normalized = json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must be a JSON object")
    return cast(dict[str, Any], normalized)


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    return _json_mapping(value, label)


def _required_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    result = value.get(field_name)
    if not isinstance(result, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return result


def _required_string(value: Mapping[str, Any], field_name: str) -> str:
    result = value.get(field_name)
    return _require_nonempty(result, field_name)


def _required_int(value: Mapping[str, Any], field_name: str) -> int:
    result = value.get(field_name)
    if type(result) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return result


def _required_number(value: Mapping[str, Any], field_name: str) -> float:
    result = value.get(field_name)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(result)


def _nonnegative_finite_float(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not float("-inf") < float(value) < float("inf")
        or float(value) < 0
    ):
        raise ValueError(f"{field_name} must be non-negative and finite")
    return float(value)


def _positive_finite_float(value: Any, field_name: str) -> float:
    result = _nonnegative_finite_float(value, field_name)
    if result <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _require_nonempty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _validated_full_score_single_user_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(
            ord(character) < 32 or 127 <= ord(character) <= 159 for character in value
        )
    ):
        raise ValueError(
            "full-score single_user_name must be a normalized non-empty string"
        )
    return value


def _full_score_phase_single_user_name(
    submit_payload: Mapping[str, Any],
) -> str:
    if not isinstance(submit_payload, Mapping):
        raise TypeError("full-score submit payload must be a mapping")
    tasks = submit_payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("full-score submit payload must contain tasks")
    principals = {
        _validated_full_score_single_user_name(
            _required_mapping(
                _json_mapping(raw_task, "full-score submit task"),
                "new_cluster",
            ).get("single_user_name")
        )
        for raw_task in tasks
    }
    if len(principals) != 1:
        raise ValueError("full-score phase tasks must share one exact principal")
    return next(iter(principals))


def _require_sha256(value: Any, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _require_shared_dbfs_path(value: Any, field_name: str) -> str:
    """Require a durable Databricks filesystem path with no traversal."""

    raw = _require_nonempty(value, field_name)
    if raw.startswith("dbfs:/"):
        relative = raw.removeprefix("dbfs:/")
    elif raw.startswith("/dbfs/"):
        relative = raw.removeprefix("/dbfs/")
    else:
        raise ValueError(f"{field_name} must be rooted in shared DBFS storage")
    parts = Path(relative).parts
    if (
        not relative
        or relative.startswith("/")
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"{field_name} must be a confined shared DBFS path")
    return raw


def _databricks_worker_mount_path(value: Any) -> str:
    """Render a worker mount string without dereferencing it on the controller."""

    raw = _require_shared_dbfs_path(value, "Databricks worker artifact URI")
    if raw.startswith("dbfs:/Volumes/"):
        return "/Volumes/" + raw.removeprefix("dbfs:/Volumes/")
    if raw.startswith("dbfs:/"):
        return "/dbfs/" + raw.removeprefix("dbfs:/").lstrip("/")
    return raw


def _require_local_disk_path(value: Any, field_name: str) -> str:
    """Require an ephemeral, traversal-free Databricks local-disk path."""

    raw = _require_nonempty(value, field_name)
    path = Path(raw)
    if not path.is_absolute() or path == Path("/local_disk0"):
        raise ValueError(f"{field_name} must be beneath /local_disk0")
    if path.parts[:2] != ("/", "local_disk0") or ".." in path.parts:
        raise ValueError(f"{field_name} must be confined beneath /local_disk0")
    return raw


def _absolute_path_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_no_symlink_ancestors(
    path: Path,
    *,
    label: str,
    include_leaf: bool,
) -> None:
    """Use lstat on every existing component without resolving any symlink."""

    absolute = _absolute_path_without_symlink_resolution(path)
    target = absolute if include_leaf else absolute.parent
    for candidate in reversed((target, *target.parents)):
        try:
            mode = os.lstat(candidate).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} cannot traverse a symlink: {candidate}")


def _require_regular_file_no_follow(path: Path, label: str) -> None:
    _require_no_symlink_ancestors(path, label=label, include_leaf=True)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"{label} must be an existing regular file") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be an existing regular file")


def _require_directory_no_follow(path: Path, label: str) -> None:
    _require_no_symlink_ancestors(path, label=label, include_leaf=True)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"{label} must be an existing directory") from exc
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} must be an existing directory")


def _open_directory_no_symlinks(path: Path) -> int:
    """Open an absolute directory one non-following component at a time."""

    absolute = _absolute_path_without_symlink_resolution(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for part in absolute.parts[1:]:
            next_descriptor = os.open(
                part,
                flags | nofollow,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"path is not a directory: {path}")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_stable_local_authority_bytes(
    path: Path,
    *,
    label: str,
    max_bytes: int = 1024 * 1024,
) -> bytes:
    """Read one local authority file through held no-follow descriptors."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("secure local authority reads require no-follow support")
    parent_fd = _open_directory_no_symlinks(path.parent)
    try:
        parent_status = os.fstat(parent_fd)
        if (
            parent_status.st_uid != os.getuid()
            or stat.S_IMODE(parent_status.st_mode) != 0o700
        ):
            raise ValueError(f"{label} parent must be current-UID mode 0700")
        return _read_stable_local_authority_bytes_at(
            parent_fd, path.name, label=label, max_bytes=max_bytes
        )
    finally:
        os.close(parent_fd)


def _read_stable_local_authority_bytes_at(
    parent_fd: int,
    leaf: str,
    *,
    label: str,
    max_bytes: int = 1024 * 1024,
    allowed_nlinks: tuple[int, ...] = (1,),
) -> bytes:
    fd = os.open(
        leaf,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink not in allowed_nlinks
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size < 0
            or before.st_size > max_bytes
        ):
            raise ValueError(f"{label} must be a bounded single-link regular file")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(fd)

        def identity(item: os.stat_result) -> tuple[int, ...]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_uid,
                item.st_nlink,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        if len(content) > max_bytes or identity(before) != identity(after):
            raise ValueError(f"{label} changed during secure read")
        return content
    finally:
        os.close(fd)


def _read_full_score_workspace_authority(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    content = _read_stable_local_authority_bytes(path, label=label)
    record = _json_object(content, label)
    if set(record) != {
        "closed_record_sha256",
        "intent_record_sha256",
        "phase_lease_record_sha256",
        "record_type",
        "schema_version",
        "user_name_sha256",
        "workspace_host_sha256",
    } or content != _canonical_pretty_json_bytes(record):
        raise ValueError(f"{label} must be exact canonical JSON")
    return record


def _write_or_require_local_authority_bytes(
    path: Path,
    content: bytes,
    *,
    label: str,
) -> None:
    """Publish one mode-0600 authority beneath a held mode-0700 directory."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("secure local authority writes require no-follow support")
    parent_fd = _open_directory_no_symlinks(path.parent)
    temporary = f".{path.name}.pending-{os.getpid()}-{time.time_ns()}"
    descriptor = -1
    try:
        parent_status = os.fstat(parent_fd)
        if (
            parent_status.st_uid != os.getuid()
            or stat.S_IMODE(parent_status.st_mode) != 0o700
        ):
            raise ValueError(f"{label} parent must be current-UID mode 0700")
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(content):
            offset += os.write(descriptor, content[offset:])
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        if (
            not stat.S_ISREG(staged.st_mode)
            or staged.st_uid != os.getuid()
            or staged.st_nlink != 1
            or stat.S_IMODE(staged.st_mode) != 0o600
            or staged.st_size != len(content)
        ):
            raise ValueError(f"{label} staging identity drift")
        try:
            os.link(
                temporary,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            observed = _read_stable_local_authority_bytes_at(
                parent_fd,
                path.name,
                label=label,
                allowed_nlinks=(1, 2),
            )
            if observed != content:
                raise ValueError(f"{label} binding drift") from None
            final_status = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if final_status.st_nlink == 2:
                prefix = f".{path.name}.pending-"
                linked_temps: list[str] = []
                for candidate in os.listdir(parent_fd):
                    if not candidate.startswith(prefix):
                        continue
                    suffix = candidate.removeprefix(prefix).split("-")
                    if len(suffix) != 2 or any(not part.isdigit() for part in suffix):
                        continue
                    candidate_status = os.stat(
                        candidate, dir_fd=parent_fd, follow_symlinks=False
                    )
                    if (
                        candidate_status.st_dev == final_status.st_dev
                        and candidate_status.st_ino == final_status.st_ino
                    ):
                        linked_temps.append(candidate)
                if len(linked_temps) != 1:
                    raise ValueError(f"{label} has ambiguous crash-recovery links")
                os.unlink(linked_temps[0], dir_fd=parent_fd)
                os.fsync(parent_fd)
                recovered = _read_stable_local_authority_bytes_at(
                    parent_fd, path.name, label=label
                )
                if recovered != content:
                    raise ValueError(f"{label} crash recovery binding drift")
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        else:
            os.fsync(parent_fd)
        os.close(parent_fd)


def _rename_directory_no_follow(source: Path, destination: Path) -> None:
    """Atomically publish a sibling directory through a securely opened parent."""

    if source.parent != destination.parent or source.name in {"", ".", ".."}:
        raise ValueError("secure directory rename requires sibling paths")
    if destination.name in {"", ".", ".."}:
        raise ValueError("secure directory rename destination is invalid")
    _require_no_symlink_ancestors(
        source,
        label="directory rename source",
        include_leaf=True,
    )
    _require_no_symlink_ancestors(
        destination,
        label="directory rename destination",
        include_leaf=True,
    )
    parent_descriptor = _open_directory_no_symlinks(source.parent)
    try:
        source_status = os.stat(
            source.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(source_status.st_mode):
            raise ValueError("directory rename source must be a real directory")
        try:
            os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"directory rename destination already exists: {destination}"
            )
        os.rename(
            source.name,
            destination.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _rename_file_no_follow(
    source: Path,
    destination: Path,
    *,
    replace_existing: bool,
) -> None:
    """Rename a sibling regular file without following either leaf path."""

    if source.parent != destination.parent or source.name in {"", ".", ".."}:
        raise ValueError("secure file rename requires sibling paths")
    if destination.name in {"", ".", ".."}:
        raise ValueError("secure file rename destination is invalid")
    _require_no_symlink_ancestors(
        source,
        label="file rename source",
        include_leaf=True,
    )
    _require_no_symlink_ancestors(
        destination,
        label="file rename destination",
        include_leaf=True,
    )
    parent_descriptor = _open_directory_no_symlinks(source.parent)
    try:
        source_status = os.stat(
            source.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(source_status.st_mode):
            raise ValueError("file rename source must be a regular file")
        try:
            destination_status = os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            destination_status = None
        if destination_status is not None:
            if not replace_existing:
                raise FileExistsError(
                    f"file rename destination already exists: {destination}"
                )
            if not stat.S_ISREG(destination_status.st_mode):
                raise ValueError("file rename destination must be a regular file")
        os.rename(
            source.name,
            destination.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _governed_existing_file(
    value: str | Path,
    field_name: str,
    *,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> Path:
    raw = _require_shared_dbfs_path(value, field_name)
    path = (
        _cluster_path(raw)
        if compact_artifact_resolver is None
        else Path(compact_artifact_resolver(raw)).expanduser().absolute()
    )
    _require_regular_file_no_follow(path, f"{field_name} governed path")
    return path


def _governed_compact_file(
    value: str | Path,
    field_name: str,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None,
) -> Path:
    """Preserve the legacy two-argument seam when no Mac CAS is supplied."""

    if compact_artifact_resolver is None:
        return _governed_existing_file(value, field_name)
    raw = _require_shared_dbfs_path(value, field_name)
    path = Path(compact_artifact_resolver(raw)).expanduser().absolute()
    _require_regular_file_no_follow(
        path,
        f"{field_name} compact CAS path",
    )
    return path


def _publish_governed_compact_file(
    value: str | Path,
    field_name: str,
    content: bytes,
    compact_artifact_publisher: FullScoreCompactArtifactPublisher | None,
) -> Path:
    """Publish one immutable compact file locally or through remote CAS I/O."""

    raw = _require_shared_dbfs_path(value, field_name)
    if compact_artifact_publisher is None:
        path = _cluster_path(raw)
        _write_or_require_exact_bytes(path, content, field_name=field_name)
    else:
        path = Path(compact_artifact_publisher(raw, content)).expanduser().absolute()
        _require_regular_file_no_follow(
            path,
            f"{field_name} compact CAS path",
        )
        if path.read_bytes() != content:
            raise ValueError(f"{field_name} compact publisher byte drift")
    return path


def _required_run_id(value: Any, field_name: str) -> str:
    if type(value) is int and value > 0:
        return str(value)
    if (
        isinstance(value, str)
        and value.isascii()
        and value.isdigit()
        and not value.startswith("0")
    ):
        return value
    raise ValueError(
        f"{field_name} must be a strictly positive canonical decimal run ID"
    )


def _validate_full_score_l40s_terminal_status(
    status: Mapping[str, Any],
) -> None:
    """Validate the qualification-only L40S shape absent from serving registries."""

    if (
        status.get("terminal") is not True
        or status.get("succeeded") is not True
        or status.get("life_cycle_state") != "TERMINATED"
        or status.get("result_state") != "SUCCESS"
        or status.get("active_task_key") is not None
    ):
        raise ValueError("L40S producer run is not a successful terminal run")
    _required_run_id(status.get("run_id"), "L40S producer run_id")
    tasks = status.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("L40S producer terminal status has no tasks")
    for raw_task in tasks:
        task = _json_mapping(raw_task, "L40S producer terminal task")
        if (
            task.get("life_cycle_state") != "TERMINATED"
            or task.get("result_state") != "SUCCESS"
            or task.get("node_type_id") != FULL_SCORE_PRODUCER_NODE_TYPE_ID
            or task.get("driver_node_type_id") != FULL_SCORE_PRODUCER_NODE_TYPE_ID
        ):
            raise ValueError("L40S producer terminal task identity/state drift")
        _required_run_id(task.get("run_id"), "L40S producer task run_id")
        _require_nonempty(task.get("task_key"), "L40S producer task_key")
    submit = _required_mapping(status, "submit_payload")
    if (
        submit.get("node_type_ids") != [FULL_SCORE_PRODUCER_NODE_TYPE_ID]
        or submit.get("driver_node_type_ids") != [FULL_SCORE_PRODUCER_NODE_TYPE_ID]
        or submit.get("single_node") is not True
    ):
        raise ValueError("L40S producer submit summary topology drift")


def _ledger_mapping_sha256(value: Mapping[str, Any]) -> str:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Databricks control-plane record is not canonical JSON"
        ) from exc
    return sha256(raw).hexdigest()


def _require_full_score_ledger_caps(ledger: DatabricksClusterHourLedger) -> None:
    if ledger.cap_cluster_hours != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS:
        raise ValueError("full-score ledger cap must be exactly 1024 GPU-hours")
    if MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS != 900.0:
        raise RuntimeError("full-score active reservation guard must be 900 GPU-hours")
    if PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS != 124.0:
        raise RuntimeError("full-score unreserved headroom must be 124 GPU-hours")
    if (
        MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
        - MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
        != PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
    ):
        raise RuntimeError("full-score cap/active/headroom constants are inconsistent")
    if (
        ledger.active_reserved_cluster_hours
        > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
        or ledger.accounted_cluster_hours > MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
    ):
        raise ValueError("full-score ledger exceeds its publication budget guards")


def _validated_full_score_phase_submit_payload(
    execution_plan: Mapping[str, Any],
    submit_payload: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    wave_index: int,
    phase: str,
    require_governed_consumer_ready_phase: bool = False,
    remote_ready_authorization: object | None = None,
    compact_artifact_resolver: FullScoreCompactArtifactResolver | None = None,
) -> list[dict[str, Any]]:
    _validate_publication_full_score_inputs(inventory, shard_plan, execution_plan)
    if phase not in {"producer", "consumer"}:
        raise ValueError("full-score phase must be producer or consumer")
    waves = cast(list[Mapping[str, Any]], execution_plan["waves"])
    if type(wave_index) is not int or not 0 <= wave_index < len(waves):
        raise ValueError("full-score phase wave index is invalid")
    if not isinstance(submit_payload, Mapping):
        raise TypeError("full-score submit payload must be a mapping")
    if frozenset(submit_payload) not in {
        frozenset({"run_name", "tasks", "timeout_seconds"}),
        frozenset({"idempotency_token", "run_name", "tasks", "timeout_seconds"}),
    }:
        raise ValueError("full-score submit payload schema drift")
    run_timeout_seconds = _validated_databricks_run_timeout_seconds(
        submit_payload.get("timeout_seconds")
    )
    if run_timeout_seconds != FULL_SCORE_DATABRICKS_TASK_TIMEOUT_SECONDS:
        raise ValueError("full-score submit timeout is not the frozen six-hour bound")
    wave = waves[wave_index]
    raw_assignments = wave.get(f"{phase}_assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise ValueError("full-score phase lacks worker assignments")
    expected_by_worker: dict[int, str] = {}
    for raw_assignment in raw_assignments:
        assignment = _json_mapping(raw_assignment, "full-score phase assignment")
        worker_index = _required_int(assignment, "worker_index")
        shard_ids = assignment.get("shard_ids")
        if not isinstance(shard_ids, list) or len(shard_ids) != 1:
            raise ValueError("publication billing requires one task per shard")
        if worker_index in expected_by_worker:
            raise ValueError("full-score phase duplicates a worker assignment")
        expected_by_worker[worker_index] = _require_nonempty(
            shard_ids[0],
            "assignment shard_id",
        )
    tasks = submit_payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(expected_by_worker):
        raise ValueError("full-score submit task coverage differs from its wave")
    if len(tasks) > FULL_SCORE_MAX_WORKERS:
        raise ValueError("full-score phase exceeds sixteen live GPU tasks")
    single_user_name = _full_score_phase_single_user_name(submit_payload)
    run_name = _required_string(submit_payload, "run_name")
    if not run_name.endswith(f"-wave-{wave_index:03d}-{phase}"):
        raise ValueError("full-score submit run_name phase/wave drift")
    expected_node = (
        FULL_SCORE_PRODUCER_NODE_TYPE_ID
        if phase == "producer"
        else FULL_SCORE_CONSUMER_NODE_TYPE_ID
    )
    observed_workers: set[int] = set()
    bindings: list[dict[str, Any]] = []
    worker_payload_records: list[dict[str, Any]] = []
    consumer_completion_bindings: list[dict[str, Any]] = []
    for raw_task in tasks:
        task = _json_mapping(raw_task, "full-score submit task")
        if set(task) != {
            "max_retries",
            "new_cluster",
            "spark_python_task",
            "task_key",
            "timeout_seconds",
        }:
            raise ValueError("full-score submit task schema drift")
        task_key = _required_string(task, "task_key")
        matching_workers = [
            worker_index
            for worker_index in expected_by_worker
            if task_key.endswith(f"_wave_{wave_index:03d}_{phase}_{worker_index:02d}")
        ]
        if len(matching_workers) != 1:
            raise ValueError("full-score task_key does not bind one planned worker")
        worker_index = matching_workers[0]
        if worker_index in observed_workers:
            raise ValueError("full-score submit duplicates a worker task")
        observed_workers.add(worker_index)
        if task.get("max_retries") != 0 or "depends_on" in task:
            raise ValueError("full-score phase tasks must be independent and no-retry")
        if (
            _validated_databricks_run_timeout_seconds(task.get("timeout_seconds"))
            != run_timeout_seconds
        ):
            raise ValueError("full-score task/run timeout drift")
        cluster = _required_mapping(task, "new_cluster")
        if set(cluster) != {
            "aws_attributes",
            "custom_tags",
            "data_security_mode",
            "driver_node_type_id",
            "node_type_id",
            "num_workers",
            "single_user_name",
            "spark_conf",
            "spark_version",
        }:
            raise ValueError("full-score phase cluster schema drift")
        if (
            cluster.get("node_type_id") != expected_node
            or cluster.get("driver_node_type_id") != expected_node
            or cluster.get("num_workers") != 0
            or cluster.get("spark_version") != DEFAULT_DATABRICKS_SPARK_VERSION
            or cluster.get("data_security_mode") != "SINGLE_USER"
            or cluster.get("single_user_name") != single_user_name
            or cluster.get("spark_conf")
            != {
                "spark.databricks.cluster.profile": "singleNode",
                "spark.master": "local[*]",
            }
            or cluster.get("aws_attributes")
            != {
                "availability": "ON_DEMAND",
                "zone_id": (
                    FULL_SCORE_PRODUCER_ZONE_ID if phase == "producer" else "auto"
                ),
            }
        ):
            raise ValueError("full-score phase task node topology drift")
        tags = _required_mapping(cluster, "custom_tags")
        if tags.get("purpose") != f"cachet-vllm-0271-full-score-{phase}":
            raise ValueError("full-score phase task purpose drift")
        spark_task = _required_mapping(task, "spark_python_task")
        if set(spark_task) != {"parameters", "python_file"}:
            raise ValueError("full-score spark task schema drift")
        python_file = _require_shared_dbfs_path(
            _required_string(spark_task, "python_file"),
            "full-score runner python_file",
        )
        raw_parameters = spark_task.get("parameters")
        if not isinstance(raw_parameters, list) or any(
            not isinstance(value, str) for value in raw_parameters
        ):
            raise ValueError("full-score task parameters must be string arguments")
        parameter_bindings = _full_score_task_parameter_bindings(
            cast(list[str], raw_parameters),
            phase=phase,
        )
        worker_payload_uri = _require_shared_dbfs_path(
            parameter_bindings["worker_payload_uri"],
            "full-score worker payload URI",
        )
        worker_payload_file = _governed_compact_file(
            worker_payload_uri,
            "full-score worker payload",
            compact_artifact_resolver,
        )
        worker_payload_raw = worker_payload_file.read_bytes()
        worker_payload_file_sha256 = sha256(worker_payload_raw).hexdigest()
        if (
            worker_payload_file_sha256
            != parameter_bindings["expected_worker_payload_sha256"]
        ):
            raise ValueError("full-score worker payload file SHA-256 drift")
        worker_payload = _json_object(
            worker_payload_raw,
            "full-score worker payload",
        )
        validate_full_score_worker_payload(
            worker_payload,
            inventory=inventory,
            shard_plan=shard_plan,
            execution_plan=execution_plan,
        )
        worker_payload_records.append(worker_payload)
        selection = _required_mapping(
            _required_mapping(worker_payload, "gpu_qualification"),
            "selection",
        )
        required_tags = {
            "ResourceClass": "SingleNode",
            "purpose": f"cachet-vllm-0271-full-score-{phase}",
        }
        if phase == "producer":
            required_tags.update(
                {
                    "generation_artifacts": _required_string(
                        selection,
                        "generation_artifacts_sha256",
                    )[:32],
                    "hardware_target": FULL_SCORE_PRODUCER_HARDWARE_TARGET,
                }
            )
        if any(tags.get(key) != value for key, value in required_tags.items()):
            raise ValueError("full-score phase cluster provenance-tag drift")
        worker_shards = worker_payload.get("shards")
        expected_shard_id = expected_by_worker[worker_index]
        if (
            worker_payload.get("authorization_scope")
            != FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
            or worker_payload.get("role") != phase
            or worker_payload.get("wave_index") != wave_index
            or worker_payload.get("worker_index") != worker_index
            or not isinstance(worker_shards, list)
            or len(worker_shards) != 1
            or worker_shards[0].get("shard_id") != expected_shard_id
        ):
            raise ValueError("full-score worker file assignment drift")
        bootstrap = _required_mapping(worker_payload, "bootstrap_artifacts")
        runtime = _required_mapping(worker_payload, "runtime")
        _verify_bound_gpu_qualification(
            _required_mapping(worker_payload, "gpu_qualification"),
            runtime=_runtime_from_record(runtime),
        )
        expected_parameter_bindings = {
            "package_wheel_sha256": _required_string(
                bootstrap,
                "package_wheel_sha256",
            ),
            "package_wheel_uri": _required_string(bootstrap, "package_wheel_uri"),
            "patched_vllm_wheel_sha256": _required_string(
                bootstrap,
                "patched_vllm_wheel_sha256",
            ),
            "patched_vllm_wheel_uri": _required_string(
                bootstrap,
                "patched_vllm_wheel_uri",
            ),
            "patched_flashinfer_wheel_sha256": _required_string(
                bootstrap,
                "patched_flashinfer_wheel_sha256",
            ),
            "patched_flashinfer_wheel_uri": _required_string(
                bootstrap,
                "patched_flashinfer_wheel_uri",
            ),
            "runner_sha256": _required_string(bootstrap, "runner_sha256"),
            "runtime_lock_sha256": _required_string(
                bootstrap,
                "runtime_lock_sha256",
            ),
            "runtime_lock_uri": _required_string(bootstrap, "runtime_lock_uri"),
            "runtime_closure_manifest_sha256": _required_string(
                bootstrap,
                "runtime_closure_manifest_sha256",
            ),
            "runtime_closure_manifest_uri": _required_string(
                bootstrap,
                "runtime_closure_manifest_uri",
            ),
        }
        if any(
            parameter_bindings.get(key) != value
            for key, value in expected_parameter_bindings.items()
        ):
            raise ValueError("full-score task/worker bootstrap binding drift")
        if (
            python_file != bootstrap.get("runner_python_file")
            or parameter_bindings["runner_sha256"] != FULL_SCORE_RUNNER_SHA256
            or parameter_bindings["runtime_lock_sha256"]
            != VLLM_RUNTIME_BASE_LOCK_SHA256
            or runtime.get("python_executable")
            != f"{parameter_bindings['runtime_venv_dir']}/bin/python"
        ):
            raise ValueError("full-score task locked-runtime binding drift")
        completion_uri = parameter_bindings.get("producer_phase_completion_uri")
        completion_sha256 = parameter_bindings.get(
            "expected_producer_phase_completion_sha256"
        )
        if phase == "consumer":
            completion_file = _governed_compact_file(
                cast(str, completion_uri),
                "producer-phase completion task input",
                compact_artifact_resolver,
            )
            completion_record = _json_object(
                completion_file.read_bytes(),
                "producer-phase completion task input",
            )
            if completion_record.get(
                "closed_record_sha256"
            ) != completion_sha256 or completion_sha256 != _closed_record_sha256(
                completion_record
            ):
                raise ValueError("consumer task producer-completion binding drift")
            consumer_completion_bindings.append(
                {
                    "closed_record_sha256": completion_sha256,
                    "path": cast(str, completion_uri),
                    "record": completion_record,
                }
            )
        bindings.append(
            {
                "durable_output_root": _required_string(
                    worker_payload,
                    "durable_output_root",
                ),
                "qualification_evidence_file_sha256": _required_string(
                    _required_mapping(worker_payload, "gpu_qualification"),
                    "evidence_file_sha256",
                ),
                "qualification_plan_sha256": _required_string(
                    _required_mapping(worker_payload, "gpu_qualification"),
                    "plan_sha256",
                ),
                "qualification_selection": dict(selection),
                "shard_id": expected_shard_id,
                "task_key": task_key,
                "worker_index": worker_index,
                "worker_payload_file_sha256": worker_payload_file_sha256,
                "worker_payload_record_sha256": _required_string(
                    worker_payload,
                    "closed_record_sha256",
                ),
                "worker_payload_uri": worker_payload_uri,
            }
        )
    if observed_workers != set(expected_by_worker):
        raise ValueError("full-score submit omits a planned worker")
    if phase == "consumer":
        completion_identities = {
            (
                _required_string(item, "path"),
                _required_string(item, "closed_record_sha256"),
                _canonical_sha256(_required_mapping(item, "record")),
            )
            for item in consumer_completion_bindings
        }
        if len(completion_identities) != 1:
            raise ValueError(
                "consumer tasks must share one exact producer-phase completion"
            )
        if require_governed_consumer_ready_phase:
            completion = _required_mapping(
                consumer_completion_bindings[0],
                "record",
            )
            _validate_producer_phase_completion(
                completion,
                execution_plan=execution_plan,
                expected_wave_index=wave_index,
            )
            if remote_ready_authorization is None:
                _validate_governed_producer_ready_phase(
                    completion,
                    payloads=worker_payload_records,
                    inventory=inventory,
                    shard_plan=shard_plan,
                    execution_plan=execution_plan,
                )
            else:
                from document_kv_cache.full_score_remote_control import (
                    require_full_score_remote_ready_authorization,
                )

                durable_roots = {
                    _required_string(payload, "durable_output_root")
                    for payload in worker_payload_records
                }
                if len(durable_roots) != 1:
                    raise ValueError(
                        "consumer worker payloads do not share one durable root"
                    )
                require_full_score_remote_ready_authorization(
                    remote_ready_authorization,
                    execution_plan_sha256=_required_string(
                        execution_plan,
                        "closed_record_sha256",
                    ),
                    wave_index=wave_index,
                    durable_output_root=next(iter(durable_roots)),
                    completion_uri=_required_string(
                        consumer_completion_bindings[0],
                        "path",
                    ),
                    completion_record=completion,
                )
    elif remote_ready_authorization is not None:
        raise ValueError("producer phase cannot consume remote ready authority")
    bindings.sort(key=lambda item: cast(int, item["worker_index"]))
    return bindings


def _full_score_task_parameter_bindings(
    parameters: Sequence[str],
    *,
    phase: str,
) -> dict[str, str]:
    """Parse only the exact package-owned runner argument sequence."""

    if phase not in {"producer", "consumer"}:
        raise ValueError("full-score task parameter phase is invalid")
    values = tuple(parameters)
    cursor = 0

    def option(name: str, field_name: str) -> str:
        nonlocal cursor
        if cursor + 1 >= len(values) or values[cursor] != name:
            raise ValueError("full-score task parameter order/schema drift")
        value = _require_nonempty(values[cursor + 1], field_name)
        cursor += 2
        return value

    result = {
        "runner_sha256": option("--runner-sha256", "runner_sha256"),
        "package_wheel_uri": option(
            "--package-wheel-uri",
            "package_wheel_uri",
        ),
        "package_wheel_sha256": option(
            "--package-wheel-sha256",
            "package_wheel_sha256",
        ),
        "runtime_lock_uri": option("--runtime-lock-uri", "runtime_lock_uri"),
        "runtime_lock_sha256": option(
            "--runtime-lock-sha256",
            "runtime_lock_sha256",
        ),
        "patched_vllm_wheel_uri": option(
            "--patched-vllm-wheel-uri",
            "patched_vllm_wheel_uri",
        ),
        "patched_vllm_wheel_sha256": option(
            "--patched-vllm-wheel-sha256",
            "patched_vllm_wheel_sha256",
        ),
        "patched_flashinfer_wheel_uri": option(
            "--patched-flashinfer-wheel-uri",
            "patched_flashinfer_wheel_uri",
        ),
        "patched_flashinfer_wheel_sha256": option(
            "--patched-flashinfer-wheel-sha256",
            "patched_flashinfer_wheel_sha256",
        ),
        "runtime_closure_manifest_uri": option(
            "--runtime-closure-manifest-uri",
            "runtime_closure_manifest_uri",
        ),
        "runtime_closure_manifest_sha256": option(
            "--runtime-closure-manifest-sha256",
            "runtime_closure_manifest_sha256",
        ),
        "runtime_venv_dir": option("--runtime-venv-dir", "runtime_venv_dir"),
    }
    if cursor >= len(values) or values[cursor] != "run-worker":
        raise ValueError("full-score task must invoke run-worker")
    cursor += 1
    result["worker_payload_uri"] = option(
        "--worker-payload-json",
        "worker_payload_uri",
    )
    result["expected_worker_payload_sha256"] = option(
        "--expected-worker-payload-sha256",
        "expected_worker_payload_sha256",
    )
    if phase == "consumer":
        result["producer_phase_completion_uri"] = option(
            "--producer-phase-completion-json",
            "producer_phase_completion_uri",
        )
        result["expected_producer_phase_completion_sha256"] = option(
            "--expected-producer-phase-completion-sha256",
            "expected_producer_phase_completion_sha256",
        )
    if cursor != len(values):
        raise ValueError("full-score task has unexpected trailing parameters")
    for field_name in (
        "runner_sha256",
        "package_wheel_sha256",
        "runtime_lock_sha256",
        "patched_vllm_wheel_sha256",
        "expected_worker_payload_sha256",
    ):
        _require_sha256(result[field_name], field_name=field_name)
    for field_name in (
        "package_wheel_uri",
        "runtime_lock_uri",
        "patched_vllm_wheel_uri",
        "worker_payload_uri",
    ):
        _require_shared_dbfs_path(result[field_name], field_name)
    _require_local_disk_path(result["runtime_venv_dir"], "runtime_venv_dir")
    if phase == "consumer":
        _require_shared_dbfs_path(
            result["producer_phase_completion_uri"],
            "producer_phase_completion_uri",
        )
        _require_sha256(
            result["expected_producer_phase_completion_sha256"],
            field_name="expected_producer_phase_completion_sha256",
        )
    return result


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def _json_type_exact_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int/float equality coercions."""

    try:
        return json.dumps(
            left,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ) == json.dumps(
            right,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return False


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("closed_record_sha256", None)
    return _canonical_sha256(payload)


def _canonical_pretty_json_bytes(record: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _cluster_path(value: str | Path) -> Path:
    raw = str(value)
    if raw.startswith("dbfs:/Volumes/"):
        return Path("/Volumes") / raw.removeprefix("dbfs:/Volumes/")
    if raw.startswith("dbfs:/"):
        return Path("/dbfs") / raw.removeprefix("dbfs:/").lstrip("/")
    return Path(raw)


def _tail(path: Path, *, characters: int = 4000) -> str:
    if not path.exists():
        return "<missing>"
    return path.read_text(encoding="utf-8", errors="replace")[-characters:]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FULL_SCORE_AGGREGATE_RECORD_TYPE",
    "FULL_SCORE_AGGREGATE_SCHEMA_VERSION",
    "FULL_SCORE_ATTENTION_BACKEND",
    "FULL_SCORE_CONSUMER_NODE_TYPE_ID",
    "FULL_SCORE_CONNECTOR_PROOF_RECORD_TYPE",
    "FULL_SCORE_CONNECTOR_PROOF_SCHEMA_VERSION",
    "FULL_SCORE_DEFAULT_MAX_BACKLOG_BYTES",
    "FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE",
    "FULL_SCORE_DELETION_ATTESTATION_RECORD_TYPE",
    "FULL_SCORE_DELETION_ATTESTATION_SCHEMA_VERSION",
    "FULL_SCORE_DATABRICKS_TASK_TIMEOUT_SECONDS",
    "FULL_SCORE_EXECUTION_PLAN_RECORD_TYPE",
    "FULL_SCORE_EXECUTION_PLAN_SCHEMA_VERSION",
    "FULL_SCORE_GENERATOR_FACTORY",
    "FULL_SCORE_GENERATOR_QUANTIZATION",
    "FULL_SCORE_GENERATOR_QUANTIZATION_CONFIG",
    "FULL_SCORE_GENERATOR_VERSION",
    "FULL_SCORE_KV_DTYPE",
    "FULL_SCORE_LIVE_P90_DRAWS",
    "FULL_SCORE_LIVE_P90_RECORD_TYPE",
    "FULL_SCORE_LIVE_P90_SCHEMA_VERSION",
    "FULL_SCORE_LIVE_P90_SEED",
    "FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE",
    "FULL_SCORE_MAX_TOKENS",
    "FULL_SCORE_MATCHED_BLOCK_RECORD_TYPE",
    "FULL_SCORE_MATCHED_BLOCK_SCHEMA_VERSION",
    "FULL_SCORE_METHODS",
    "FULL_SCORE_MODEL_DTYPE",
    "FULL_SCORE_MODEL_ID",
    "FULL_SCORE_MODEL_QUANTIZATION",
    "FULL_SCORE_MODEL_NUM_LAYERS",
    "FULL_SCORE_PASSES_PER_METHOD",
    "FULL_SCORE_PHASE_TERMINAL_RECORD_TYPE",
    "FULL_SCORE_PHASE_TERMINAL_SCHEMA_VERSION",
    "FULL_SCORE_PRODUCER_HARDWARE_TARGET",
    "FULL_SCORE_PRODUCER_NODE_TYPE_ID",
    "FULL_SCORE_PRODUCER_ZONE_ID",
    "FULL_SCORE_PRODUCER_PHASE_COMPLETION_RECORD_TYPE",
    "FULL_SCORE_PRODUCER_PHASE_COMPLETION_SCHEMA_VERSION",
    "FULL_SCORE_PROTOCOL_ID",
    "FULL_SCORE_PUBLICATION_CACHE_PREFIX_TOKENS",
    "FULL_SCORE_PUBLICATION_EXECUTION_PLAN_SHA256",
    "FULL_SCORE_PUBLICATION_INVENTORY_SHA256",
    "FULL_SCORE_PUBLICATION_ITEM_COUNT",
    "FULL_SCORE_PUBLICATION_NATURAL_PROMPT_TOKENS",
    "FULL_SCORE_PUBLICATION_SHARD_COUNT",
    "FULL_SCORE_PUBLICATION_SHARD_PLAN_SHA256",
    "FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE",
    "FULL_SCORE_Q8_BYTES_PER_CACHE_PREFIX_TOKEN",
    "FULL_SCORE_READY_SHARD_RECORD_TYPE",
    "FULL_SCORE_READY_SHARD_SCHEMA_VERSION",
    "FULL_SCORE_REQUEST_CUSTOMIZATION_DIGEST",
    "FULL_SCORE_REQUEST_PARALLELISM",
    "FULL_SCORE_RUNNER_SHA256",
    "FULL_SCORE_RUNNER_SCRIPT",
    "FULL_SCORE_SERVED_MODEL_NAME",
    "FULL_SCORE_SHARD_EVIDENCE_RECORD_TYPE",
    "FULL_SCORE_SHARD_EVIDENCE_SCHEMA_VERSION",
    "FULL_SCORE_TEMPERATURE",
    "FULL_SCORE_VLLM_BNB_LOADER_MEMBER",
    "FULL_SCORE_VLLM_BNB_LOADER_SHA256",
    "FULL_SCORE_WAVE_COMPLETION_RECORD_TYPE",
    "FULL_SCORE_WAVE_COMPLETION_SCHEMA_VERSION",
    "FULL_SCORE_WORKER_PAYLOAD_RECORD_TYPE",
    "FULL_SCORE_WORKER_PAYLOAD_SCHEMA_VERSION",
    "DatabricksFullScoreJobConfig",
    "FullScoreCommandRunner",
    "FullScorePhaseAuthorization",
    "FullScorePhaseSubmissionAuthorization",
    "FullScoreGPUQualificationConfig",
    "FullScoreRuntimeConfig",
    "FullScoreShardLifecycle",
    "FullScoreWorkerBundleConfig",
    "aggregate_full_score_shard_evidence",
    "build_databricks_full_score_run_submit_payload",
    "build_full_score_execution_plan",
    "build_full_score_connector_proof",
    "build_full_score_live_p90_budget_admission",
    "build_full_score_matched_billing_block",
    "build_full_score_producer_phase_completion_record",
    "build_full_score_wave_completion_record",
    "build_full_score_worker_payloads",
    "build_governed_full_score_live_p90_budget_admission",
    "build_governed_full_score_matched_billing_block",
    "build_governed_full_score_phase_terminal_record",
    "build_governed_full_score_producer_phase_completion_record",
    "build_governed_full_score_wave_completion_record",
    "full_score_inventory_from_record",
    "full_score_wave_worst_case_gpu_hours",
    "collect_governed_full_score_phase_attempt",
    "main",
    "load_governed_full_score_live_p90_budget_admission",
    "load_governed_full_score_matched_billing_block",
    "load_governed_full_score_phase_terminal_record",
    "load_governed_full_score_shard_evidence",
    "preview_local_fixture_databricks_full_score_run_submit_payload",
    "prepare_governed_full_score_live_p90_phase_submission",
    "render_full_score_worker_command_plan",
    "recover_governed_full_score_phase_attempt",
    "recover_governed_full_score_phase_reservation",
    "replay_governed_full_score_phase_authorization",
    "replay_governed_full_score_phase_submission_authorization",
    "resume_governed_full_score_phase_attempt",
    "reserve_and_submit_governed_full_score_phase_attempt",
    "reserve_governed_full_score_phase_attempt",
    "submit_governed_full_score_phase_attempt",
    "run_full_score_worker",
    "validate_full_score_aggregate_record",
    "validate_full_score_live_p90_budget_admission",
    "validate_full_score_worker_payload",
    "validate_paired_full_score_outputs",
    "write_full_score_runner_script",
    "write_full_score_producer_phase_completion_record",
    "write_full_score_worker_payloads",
    "write_governed_full_score_live_p90_budget_admission",
    "write_governed_full_score_matched_billing_block",
    "write_governed_full_score_phase_terminal_record",
]
