"""Reusable preparation and validation for representative isolated canaries.

The representative matrix deliberately contains only the exact full-context
control and the current per-document vanilla method.  It does not provide a
generic escape hatch for inventing method semantics: every cache arm is tied to
the registered method identity, its own handoff, and its expected segmentation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from document_kv_cache.artifact_identity import (
    UNRESOLVED_IDENTITY,
    ArtifactIdentity,
    method_config_digest,
)
from document_kv_cache.benchmark_handoffs import (
    BenchmarkHandoffEntry,
    BenchmarkHandoffManifest,
    enrich_benchmark_records_with_handoffs,
    read_benchmark_handoff_manifest_json,
)
from document_kv_cache.benchmark_runner import (
    BENCHMARK_RUN_RECORD_TYPE,
    benchmark_record_aggregate_issues,
    load_benchmark_jsonl,
    merge_isolated_benchmark_run_records,
    parse_benchmark_arm_specs,
)
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    DOCUMENT_KV_ARTIFACT_ID_PARAM,
    DOCUMENT_KV_CACHE_METHOD_PARAM,
    build_prompt_parts,
)
from document_kv_cache.databricks_resource_ledger import (
    MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    DatabricksClusterHourLedger,
    DatabricksClusterHourReservation,
    create_databricks_cluster_hour_ledger_json,
    databricks_submit_payload_reservation,
)
from document_kv_cache.databricks_runs import (
    DatabricksURLOpener,
    DatabricksWorkspaceConfig,
    reserve_and_submit_databricks_run,
)
from document_kv_cache.methods import MethodRegistry, default_method_registry
from document_kv_cache.models import CacheGenerationMethod
from document_kv_cache.storage import local_path


FULL_PREFIX_CANARY_ARM = "document_kv_cache:full_prefix_prefill"
VANILLA_CANARY_ARM = "document_kv_cache:vanilla_prefill"
REPRESENTATIVE_CANARY_ARM_IDS = (
    BASELINE_PREFILL_ARM,
    FULL_PREFIX_CANARY_ARM,
    VANILLA_CANARY_ARM,
)
REPRESENTATIVE_CANARY_INPUT_RECORD_TYPE = "document_kv.representative_canary_inputs.v1"
HANDOFF_TOPOLOGY_ATTESTATION_RECORD_TYPE = (
    "document_kv.handoff_topology_attestation.v1"
)
HANDOFF_TOPOLOGY_ATTESTATION_SCHEMA_VERSION = 1
# Kept as a named export for callers that want to assert the aggregate type.  A
# physical-isolated merge is an ordinary benchmark run, not a parallel schema.
ISOLATED_CANARY_AGGREGATE_RECORD_TYPE = BENCHMARK_RUN_RECORD_TYPE
REPRESENTATIVE_CANARY_JOB_COUNT = 10
REPRESENTATIVE_CANARY_AGGREGATE_CLUSTER_HOUR_CAP = (
    MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
)
REPRESENTATIVE_CANARY_FIRST_WAVE_CLUSTER_HOURS = 40.0
REPRESENTATIVE_CANARY_WORKLOAD_MANIFEST_RECORD_TYPE = (
    "document_kv.representative_canary_workload_manifest.v1"
)
REPRESENTATIVE_CANARY_WORKLOAD_MANIFEST_SCHEMA_VERSION = 1
SGLANG_PAIRED_SMOKE_ARM = "baseline_prefill+document_kv_cache"
REPRESENTATIVE_CANARY_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
REPRESENTATIVE_CANARY_MODEL_REVISION = (
    "cdbee75f17c01a7cc42f958dc650907174af0554"
)
REPRESENTATIVE_VLLM_PACKAGE_PINS = (
    "vllm==0.23.0",
    "transformers==5.12.1",
    "huggingface-hub==1.20.1",
    "tokenizers==0.22.2",
    "numpy==2.3.5",
    "fastapi[standard]==0.136.0",
    "prometheus-fastapi-instrumentator==8.0.0",
    "bitsandbytes==0.49.2",
    "accelerate==1.14.0",
)
REPRESENTATIVE_SGLANG_PACKAGE_PINS = ("sglang==0.5.10.post1",)
REPRESENTATIVE_HANDOFF_GENERATOR_FACTORY = (
    "document_kv_cache.transformers_generator:"
    "build_transformers_kv_chunk_generator"
)
REPRESENTATIVE_DATABRICKS_SPARK_VERSION = "15.4.x-gpu-ml-scala2.12"
REPRESENTATIVE_VLLM_RUNNER_BASENAME = "run_vllm_smoke.py"
REPRESENTATIVE_SGLANG_RUNNER_BASENAME = "run_sglang_smoke.py"
REPRESENTATIVE_VLLM_RUNNER_SHA256 = (
    "e5339a43b7b339ec3793881aba7f9a7a7f3b51e540bf82b57dcd32d30f696398"
)
REPRESENTATIVE_SGLANG_RUNNER_SHA256 = (
    "0e289f0ac7379f87917feb772b35ea75f58a419a6d5aa9c1881c75aec15a4500"
)
REPRESENTATIVE_HOTPOT_DATASET_BASENAME = "hotpotqa.jsonl"

_REPRESENTATIVE_COMMON_PARAMETER_FLAGS = frozenset(
    {
        "--benchmark-id",
        "--output-dir",
        "--max-tokens",
        "--timeout-seconds",
        "--import-probe-timeout-seconds",
        "--server-start-timeout-seconds",
        "--local-root",
        "--server-host",
        "--server-port",
        "--client-host",
        "--hardware-target",
        "--model-revision",
        "--tokenizer-revision",
        "--representative-canary",
        "--representative-workload-profile",
        "--package-wheel-uri",
        "--package-wheel-sha256",
    }
)
_REPRESENTATIVE_VLLM_PARAMETER_FLAGS = frozenset(
    {
        "--max-model-len",
        "--max-num-seqs",
        "--gpu-memory-utilization",
        "--benchmark-repeats",
        "--request-parallelism",
        "--runtime-telemetry-interval-seconds",
        "--model-id",
        "--model-dtype",
        "--kv-cache-dtype",
        "--payload-cache-max-bytes",
        "--benchmark-arm-spec-json",
        "--benchmark-evidence-policy",
        "--benchmark-manifest-provenance-json",
        "--benchmark-force-max-tokens",
        "--benchmark-prefix-cache-salt-mode",
        "--dataset",
        "--allow-dataset-subset",
    }
)
_REPRESENTATIVE_VLLM_HANDOFF_PARAMETER_FLAGS = frozenset(
    {
        "--benchmark-handoff-generator-factory",
        "--benchmark-handoff-dtype",
        "--benchmark-handoff-align-bytes",
        "--benchmark-handoff-generation-timeout-seconds",
        "--benchmark-handoff-output-dir",
        "--benchmark-handoff-cache-method",
    }
)
_REPRESENTATIVE_SGLANG_PARAMETER_FLAGS = frozenset(
    {
        "--context-length",
        "--mem-fraction-static",
        "--cache-prompt-text-mode",
        "--live-check-prompt-format",
        "--live-check-request-mode",
        "--live-check-temperature",
        "--representative-package-pin",
        "--flush-cache-timeout-seconds",
        "--live-benchmark-repeats",
        "--sglang-attention-backend",
        "--sglang-sampling-backend",
        "--sglang-enable-deterministic-inference",
        "--sglang-hicache-page-size",
        "--generate-live-handoff",
        "--live-handoff-output-dir",
        "--live-handoff-generator-factory",
        "--live-handoff-dtype",
        "--live-handoff-align-bytes",
        "--live-handoff-generation-timeout-seconds",
        "--hicache-storage-prefetch-policy",
        "--hicache-storage-prefetch-threshold",
    }
)
_REPRESENTATIVE_SUBMIT_PAYLOAD_KEYS = frozenset(
    {"run_name", "timeout_seconds", "tasks"}
)
_REPRESENTATIVE_TASK_KEYS = frozenset(
    {"task_key", "timeout_seconds", "max_retries", "new_cluster", "spark_python_task"}
)
_REPRESENTATIVE_SPARK_PYTHON_TASK_KEYS = frozenset({"python_file", "parameters"})
_REPRESENTATIVE_NEW_CLUSTER_KEYS = frozenset(
    {
        "spark_version",
        "node_type_id",
        "driver_node_type_id",
        "data_security_mode",
        "num_workers",
        "spark_conf",
        "custom_tags",
        "aws_attributes",
        "single_user_name",
        "spark_env_vars",
    }
)

_REPRESENTATIVE_CANARY_WORKLOAD_SPECS = (
    (
        1,
        "g6-vllm-8k-64-baseline",
        "required",
        "vllm-8k-64-v1",
        "vllm",
        "aws-g6-l4",
        "g6.8xlarge",
        BASELINE_PREFILL_ARM,
    ),
    (
        2,
        "g6-vllm-8k-64-full-prefix",
        "required",
        "vllm-8k-64-v1",
        "vllm",
        "aws-g6-l4",
        "g6.8xlarge",
        FULL_PREFIX_CANARY_ARM,
    ),
    (
        3,
        "g6-vllm-8k-64-vanilla",
        "required",
        "vllm-8k-64-v1",
        "vllm",
        "aws-g6-l4",
        "g6.8xlarge",
        VANILLA_CANARY_ARM,
    ),
    (
        4,
        "g6-vllm-16k-256-baseline",
        "required",
        "vllm-16k-256-v1",
        "vllm",
        "aws-g6-l4",
        "g6.8xlarge",
        BASELINE_PREFILL_ARM,
    ),
    (
        5,
        "g6-vllm-16k-256-full-prefix",
        "required",
        "vllm-16k-256-v1",
        "vllm",
        "aws-g6-l4",
        "g6.8xlarge",
        FULL_PREFIX_CANARY_ARM,
    ),
    (
        6,
        "g6-vllm-16k-256-vanilla",
        "required",
        "vllm-16k-256-v1",
        "vllm",
        "aws-g6-l4",
        "g6.8xlarge",
        VANILLA_CANARY_ARM,
    ),
    (
        7,
        "g6-sglang-4k-32-paired-smoke",
        "required",
        "sglang-4k-32-v1",
        "sglang",
        "aws-g6-l4",
        "g6.8xlarge",
        SGLANG_PAIRED_SMOKE_ARM,
    ),
    (
        8,
        "g5-vllm-8k-64-baseline",
        "best_effort",
        "vllm-8k-64-v1",
        "vllm",
        "aws-g5-a10g",
        "g5.8xlarge",
        BASELINE_PREFILL_ARM,
    ),
    (
        9,
        "g5-vllm-8k-64-full-prefix",
        "best_effort",
        "vllm-8k-64-v1",
        "vllm",
        "aws-g5-a10g",
        "g5.8xlarge",
        FULL_PREFIX_CANARY_ARM,
    ),
    (
        10,
        "g5-vllm-8k-64-vanilla",
        "best_effort",
        "vllm-8k-64-v1",
        "vllm",
        "aws-g5-a10g",
        "g5.8xlarge",
        VANILLA_CANARY_ARM,
    ),
)

_PROVENANCE_FIELDS = frozenset(
    {
        "model_revision",
        "canonical_model_id",
        "tokenizer_id",
        "tokenizer_revision",
        "lora_id",
        "engine_id",
        "engine_version",
        "serving_platform",
        "model_dtype",
        "model_quantization",
        "runtime_kv_dtype",
        "layout_version",
        "payload_axis_order",
        "block_size",
        "key_position_encoding",
        "rope_theta",
        "rope_rotary_dim",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "package_revisions",
        "prompt_template_version",
        "input_tokens_target",
        "hardware_fingerprint",
        "runtime_id",
        "runtime_version",
        "storage_identity",
        "cache_state",
        "complete_dataset_split",
        "measurement_scopes",
        "comparison_mode",
        "varied_setting",
    }
)
_PROVENANCE_STRING_FLAGS = (
    ("model_revision", "--model-revision"),
    ("canonical_model_id", "--canonical-model-id"),
    ("tokenizer_id", "--tokenizer-id"),
    ("tokenizer_revision", "--tokenizer-revision"),
    ("lora_id", "--lora-id"),
    ("engine_id", "--engine-id"),
    ("engine_version", "--engine-version"),
    ("serving_platform", "--serving-platform"),
    ("model_dtype", "--model-dtype"),
    ("model_quantization", "--model-quantization"),
    ("runtime_kv_dtype", "--runtime-kv-dtype"),
    ("layout_version", "--layout-version"),
    ("payload_axis_order", "--payload-axis-order"),
    ("key_position_encoding", "--key-position-encoding"),
    ("prompt_template_version", "--prompt-template-version"),
    ("hardware_fingerprint", "--hardware-fingerprint"),
    ("runtime_id", "--runtime-id"),
    ("runtime_version", "--runtime-version"),
    ("storage_identity", "--storage-identity"),
    ("cache_state", "--cache-state"),
    ("comparison_mode", "--comparison-mode"),
    ("varied_setting", "--varied-setting"),
)
_PROVENANCE_INTEGER_FLAGS = (
    ("block_size", "--block-size"),
    ("rope_rotary_dim", "--rope-rotary-dim"),
    ("tensor_parallel_size", "--tensor-parallel-size"),
    ("pipeline_parallel_size", "--pipeline-parallel-size"),
)
_PROVENANCE_FLOAT_FLAGS = (("rope_theta", "--rope-theta"),)

__all__ = [
    "FULL_PREFIX_CANARY_ARM",
    "VANILLA_CANARY_ARM",
    "REPRESENTATIVE_CANARY_ARM_IDS",
    "REPRESENTATIVE_CANARY_INPUT_RECORD_TYPE",
    "HANDOFF_TOPOLOGY_ATTESTATION_RECORD_TYPE",
    "HANDOFF_TOPOLOGY_ATTESTATION_SCHEMA_VERSION",
    "ISOLATED_CANARY_AGGREGATE_RECORD_TYPE",
    "REPRESENTATIVE_CANARY_JOB_COUNT",
    "REPRESENTATIVE_CANARY_AGGREGATE_CLUSTER_HOUR_CAP",
    "REPRESENTATIVE_CANARY_FIRST_WAVE_CLUSTER_HOURS",
    "REPRESENTATIVE_CANARY_WORKLOAD_MANIFEST_RECORD_TYPE",
    "REPRESENTATIVE_CANARY_WORKLOAD_MANIFEST_SCHEMA_VERSION",
    "SGLANG_PAIRED_SMOKE_ARM",
    "REPRESENTATIVE_CANARY_MODEL_ID",
    "REPRESENTATIVE_CANARY_MODEL_REVISION",
    "REPRESENTATIVE_VLLM_PACKAGE_PINS",
    "REPRESENTATIVE_SGLANG_PACKAGE_PINS",
    "REPRESENTATIVE_HANDOFF_GENERATOR_FACTORY",
    "REPRESENTATIVE_DATABRICKS_SPARK_VERSION",
    "REPRESENTATIVE_VLLM_RUNNER_BASENAME",
    "REPRESENTATIVE_SGLANG_RUNNER_BASENAME",
    "REPRESENTATIVE_VLLM_RUNNER_SHA256",
    "REPRESENTATIVE_SGLANG_RUNNER_SHA256",
    "REPRESENTATIVE_HOTPOT_DATASET_BASENAME",
    "RepresentativeCanaryRun",
    "RepresentativeCanaryMatrix",
    "RepresentativeCanaryPreparedInputs",
    "RepresentativeCanaryWorkload",
    "RepresentativeCanaryWorkloadManifest",
    "representative_canary_matrix",
    "representative_canary_workload_manifest",
    "create_representative_canary_cluster_hour_ledger",
    "validate_representative_canary_workload_payload",
    "validate_representative_canary_workload_payloads",
    "validate_representative_canary_reservation",
    "reserve_and_submit_representative_canary_workload",
    "validated_representative_wheel_binding",
    "prepare_representative_canary_inputs",
    "aggregate_isolated_canary_results",
    "validated_benchmark_arm_specs",
    "validated_benchmark_manifest_provenance",
    "benchmark_manifest_provenance_runner_args",
    "benchmark_json_mapping_to_record",
    "resolved_layout_rope_provenance",
    "build_handoff_topology_attestation",
    "build_handoff_topology_attestation_record",
    "generator_token_counter",
    "merge_handoff_topology_attestations",
    "validate_handoff_topology_attestation",
    "require_pinned_revision",
]

_HANDOFF_TOPOLOGY_RECORD_KEYS = frozenset(
    {"record_type", "schema_version", "example_count", "examples_sha256", "examples"}
)
_HANDOFF_TOPOLOGY_EXAMPLE_KEYS = frozenset(
    {
        "example_key_sha256",
        "method_id",
        "method_version",
        "method_config_digest",
        "artifact_id",
        "document_count",
        "segment_count",
        "logical_token_count",
        "logical_prompt_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class RepresentativeCanaryRun:
    """One independently runnable arm in the representative matrix."""

    arm_id: str
    method_id: str
    expected_segments: str
    arm_spec: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.arm_id not in REPRESENTATIVE_CANARY_ARM_IDS:
            raise ValueError(f"unsupported representative canary arm {self.arm_id!r}")
        expected_method = _method_for_arm(self.arm_id)
        if self.method_id != expected_method:
            raise ValueError(
                f"arm {self.arm_id!r} requires method_id {expected_method!r}"
            )
        if self.expected_segments not in {"none", "one", "per_document"}:
            raise ValueError("expected_segments must be none, one, or per_document")
        records = validated_benchmark_arm_specs((self.arm_spec,))
        if records[0]["arm_id"] != self.arm_id:
            raise ValueError("arm_spec.arm_id must match arm_id")
        object.__setattr__(self, "arm_spec", records[0])

    def runner_args(self) -> tuple[str, str]:
        return (
            "--arm-spec-json",
            json.dumps(
                benchmark_json_mapping_to_record(self.arm_spec),
                separators=(",", ":"),
                sort_keys=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class RepresentativeCanaryMatrix:
    """The fixed baseline/full-prefix/vanilla comparison matrix."""

    runs: tuple[RepresentativeCanaryRun, ...]

    def __post_init__(self) -> None:
        runs = tuple(self.runs)
        if tuple(run.arm_id for run in runs) != REPRESENTATIVE_CANARY_ARM_IDS:
            raise ValueError(
                "representative canary runs must be ordered baseline, full-prefix, vanilla"
            )
        object.__setattr__(self, "runs", runs)

    def run_for_arm(self, arm_id: str) -> RepresentativeCanaryRun:
        for run in self.runs:
            if run.arm_id == arm_id:
                return run
        raise KeyError(arm_id)


@dataclass(frozen=True, slots=True)
class RepresentativeCanaryPreparedInputs:
    """Prepared N-way input plus safe one-arm projections for isolated jobs."""

    combined_jsonl: Path
    arm_jsonl: Mapping[str, Path]
    preparation_manifest_json: Path
    logical_sample_digest: str
    example_count: int

    def __post_init__(self) -> None:
        paths = dict(self.arm_jsonl)
        if tuple(paths) != REPRESENTATIVE_CANARY_ARM_IDS:
            raise ValueError("arm_jsonl must contain the representative arm order")
        if len(self.logical_sample_digest) != 64:
            raise ValueError("logical_sample_digest must be a SHA-256 digest")
        if type(self.example_count) is not int or self.example_count <= 0:
            raise ValueError("example_count must be positive")
        object.__setattr__(self, "combined_jsonl", Path(self.combined_jsonl))
        object.__setattr__(
            self,
            "arm_jsonl",
            MappingProxyType({arm_id: Path(path) for arm_id, path in paths.items()}),
        )
        object.__setattr__(
            self,
            "preparation_manifest_json",
            Path(self.preparation_manifest_json),
        )


@dataclass(frozen=True, slots=True)
class RepresentativeCanaryWorkload:
    """One fixed physical job in the representative submission sequence."""

    order: int
    workload_id: str
    requirement: Literal["required", "best_effort"]
    profile_id: str
    serving_platform: Literal["vllm", "sglang"]
    hardware_target: Literal["aws-g6-l4", "aws-g5-a10g"]
    node_type_id: Literal["g6.8xlarge", "g5.8xlarge"]
    arm_id: str

    def __post_init__(self) -> None:
        if type(self.order) is not int or not 1 <= self.order <= REPRESENTATIVE_CANARY_JOB_COUNT:
            raise ValueError(
                f"order must be in 1..{REPRESENTATIVE_CANARY_JOB_COUNT}"
            )
        for field_name in ("workload_id", "profile_id", "arm_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be non-empty")
        if self.requirement not in {"required", "best_effort"}:
            raise ValueError("requirement must be required or best_effort")
        if self.serving_platform not in {"vllm", "sglang"}:
            raise ValueError("serving_platform must be vllm or sglang")
        expected_node = {
            "aws-g6-l4": "g6.8xlarge",
            "aws-g5-a10g": "g5.8xlarge",
        }.get(self.hardware_target)
        if expected_node is None:
            raise ValueError("unsupported representative hardware_target")
        if self.node_type_id != expected_node:
            raise ValueError(
                "representative workload node_type_id must exactly match its "
                "hardware_target"
            )
        if self.serving_platform == "vllm":
            if self.profile_id not in {"vllm-8k-64-v1", "vllm-16k-256-v1"}:
                raise ValueError("unsupported representative vLLM profile_id")
            if self.arm_id not in REPRESENTATIVE_CANARY_ARM_IDS:
                raise ValueError("unsupported representative vLLM arm_id")
        elif (
            self.profile_id != "sglang-4k-32-v1"
            or self.arm_id != SGLANG_PAIRED_SMOKE_ARM
        ):
            raise ValueError("SGLang representative workload must use the paired smoke")

    @property
    def package_pins(self) -> tuple[str, ...]:
        return (
            REPRESENTATIVE_VLLM_PACKAGE_PINS
            if self.serving_platform == "vllm"
            else REPRESENTATIVE_SGLANG_PACKAGE_PINS
        )

    @property
    def runner_basename(self) -> str:
        return (
            REPRESENTATIVE_VLLM_RUNNER_BASENAME
            if self.serving_platform == "vllm"
            else REPRESENTATIVE_SGLANG_RUNNER_BASENAME
        )

    @property
    def runner_sha256(self) -> str:
        return (
            REPRESENTATIVE_VLLM_RUNNER_SHA256
            if self.serving_platform == "vllm"
            else REPRESENTATIVE_SGLANG_RUNNER_SHA256
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "workload_id": self.workload_id,
            "requirement": self.requirement,
            "profile_id": self.profile_id,
            "serving_platform": self.serving_platform,
            "hardware_target": self.hardware_target,
            "node_type_id": self.node_type_id,
            "arm_id": self.arm_id,
            "model_id": REPRESENTATIVE_CANARY_MODEL_ID,
            "model_revision": REPRESENTATIVE_CANARY_MODEL_REVISION,
            "package_pins": list(self.package_pins),
            "spark_version": REPRESENTATIVE_DATABRICKS_SPARK_VERSION,
            "runner_basename": self.runner_basename,
            "runner_sha256": self.runner_sha256,
        }


@dataclass(frozen=True, slots=True)
class RepresentativeCanaryWorkloadManifest:
    """Closed ordered manifest for the ten physical representative jobs."""

    workloads: tuple[RepresentativeCanaryWorkload, ...]

    def __post_init__(self) -> None:
        workloads = tuple(self.workloads)
        expected = tuple(
            _representative_canary_workload_from_spec(spec)
            for spec in _REPRESENTATIVE_CANARY_WORKLOAD_SPECS
        )
        if workloads != expected:
            raise ValueError(
                "representative workload manifest must contain the exact ordered "
                "ten-job sequence"
            )
        object.__setattr__(self, "workloads", workloads)

    def workload_for_id(self, workload_id: str) -> RepresentativeCanaryWorkload:
        for workload in self.workloads:
            if workload.workload_id == workload_id:
                return workload
        raise KeyError(workload_id)

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": REPRESENTATIVE_CANARY_WORKLOAD_MANIFEST_RECORD_TYPE,
            "schema_version": (
                REPRESENTATIVE_CANARY_WORKLOAD_MANIFEST_SCHEMA_VERSION
            ),
            "job_count": len(self.workloads),
            "first_wave_worst_case_cluster_hours": (
                REPRESENTATIVE_CANARY_FIRST_WAVE_CLUSTER_HOURS
            ),
            "workloads": [workload.to_record() for workload in self.workloads],
        }


def representative_canary_workload_manifest() -> RepresentativeCanaryWorkloadManifest:
    """Return the immutable ordered submission manifest."""

    return RepresentativeCanaryWorkloadManifest(
        workloads=tuple(
            _representative_canary_workload_from_spec(spec)
            for spec in _REPRESENTATIVE_CANARY_WORKLOAD_SPECS
        )
    )


def _representative_canary_workload_from_spec(
    spec: tuple[int, str, str, str, str, str, str, str],
) -> RepresentativeCanaryWorkload:
    return RepresentativeCanaryWorkload(
        order=spec[0],
        workload_id=spec[1],
        requirement=cast(Literal["required", "best_effort"], spec[2]),
        profile_id=spec[3],
        serving_platform=cast(Literal["vllm", "sglang"], spec[4]),
        hardware_target=cast(
            Literal["aws-g6-l4", "aws-g5-a10g"],
            spec[5],
        ),
        node_type_id=cast(Literal["g6.8xlarge", "g5.8xlarge"], spec[6]),
        arm_id=spec[7],
    )


def representative_canary_matrix() -> RepresentativeCanaryMatrix:
    """Return the registered three-arm matrix with stable physical transforms."""

    registry = default_method_registry()
    full_prefix = registry.get(
        CacheGenerationMethod.FULL_PREFIX_PREFILL,
        require_implemented=True,
    )
    vanilla = registry.get(
        CacheGenerationMethod.VANILLA_PREFILL,
        require_implemented=True,
    )
    return RepresentativeCanaryMatrix(
        runs=(
            RepresentativeCanaryRun(
                arm_id=BASELINE_PREFILL_ARM,
                method_id="",
                expected_segments="none",
                arm_spec={
                    "arm_id": BASELINE_PREFILL_ARM,
                    "uses_cache": False,
                    "description": "Recompute the complete logical prompt online.",
                    "implementation_kind": "baseline",
                    "physical_transform_id": "identity",
                    "physical_transform_version": "1",
                },
            ),
            RepresentativeCanaryRun(
                arm_id=FULL_PREFIX_CANARY_ARM,
                method_id=full_prefix.method_id,
                expected_segments="one",
                arm_spec=_method_arm_spec(
                    arm_id=FULL_PREFIX_CANARY_ARM,
                    method=full_prefix,
                    physical_transform_id="cachet.full_prefix.single_segment",
                ),
            ),
            RepresentativeCanaryRun(
                arm_id=VANILLA_CANARY_ARM,
                method_id=vanilla.method_id,
                expected_segments="per_document",
                arm_spec=_method_arm_spec(
                    arm_id=VANILLA_CANARY_ARM,
                    method=vanilla,
                    physical_transform_id="cachet.vanilla.per_document_segments",
                ),
            ),
        )
    )


def create_representative_canary_cluster_hour_ledger(
    path: str | Path,
    *,
    ledger_id: str,
) -> DatabricksClusterHourLedger:
    """Create the persistent 120-hour guard for the fixed ten-job canary sequence."""

    return create_databricks_cluster_hour_ledger_json(
        path,
        ledger_id=ledger_id,
        cap_cluster_hours=REPRESENTATIVE_CANARY_AGGREGATE_CLUSTER_HOUR_CAP,
    )


def validated_representative_wheel_binding(
    wheel_uri: object,
    wheel_sha256: object,
) -> tuple[str, str]:
    """Validate a content-addressed staged Cachet wheel for representative jobs."""

    digest = _require_sha256(wheel_sha256, "wheel_sha256")
    if not isinstance(wheel_uri, str) or not wheel_uri:
        raise ValueError("representative wheel_uri must be non-empty")
    if wheel_uri.startswith("dbfs:/"):
        path_text = wheel_uri.removeprefix("dbfs:")
    elif wheel_uri.startswith(("/dbfs/", "/Volumes/")):
        path_text = wheel_uri
    else:
        raise ValueError(
            "representative wheel_uri must use persistent dbfs:/, /dbfs/, or "
            "/Volumes/ storage"
        )
    path = Path(path_text)
    if ".." in path.parts:
        raise ValueError("representative wheel_uri must not contain parent traversal")
    if path.suffix != ".whl":
        raise ValueError("representative wheel_uri must reference a .whl file")
    if digest not in path.parts:
        raise ValueError(
            "representative wheel_uri must contain wheel_sha256 as a path component"
        )
    return wheel_uri, digest


def validate_representative_canary_workload_payload(
    workload: RepresentativeCanaryWorkload,
    submit_payload: Mapping[str, Any],
) -> DatabricksClusterHourReservation:
    """Validate one real job-builder payload against its fixed workload entry."""

    if not isinstance(workload, RepresentativeCanaryWorkload):
        raise TypeError("workload must be a RepresentativeCanaryWorkload")
    manifest_workload = representative_canary_workload_manifest().workload_for_id(
        workload.workload_id
    )
    if workload != manifest_workload:
        raise ValueError("workload does not match the canonical manifest entry")
    reservation = databricks_submit_payload_reservation(
        submit_payload,
        attempt_id=f"manifest-validation-{workload.order}",
        workload_id=workload.workload_id,
    )
    _validate_representative_canary_payload_contract(
        workload,
        submit_payload,
        reservation,
    )
    return reservation


def validate_representative_canary_workload_payloads(
    payloads: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[DatabricksClusterHourReservation, ...]:
    """Validate all ten payloads in exact manifest order and attest 40 hours."""

    ordered_payloads = tuple(payloads)
    manifest = representative_canary_workload_manifest()
    actual_ids = tuple(workload_id for workload_id, _payload in ordered_payloads)
    expected_ids = tuple(workload.workload_id for workload in manifest.workloads)
    if actual_ids != expected_ids:
        raise ValueError(
            "representative payloads must match the exact ordered ten-workload manifest"
        )
    reservations = tuple(
        validate_representative_canary_workload_payload(workload, payload)
        for workload, (_workload_id, payload) in zip(
            manifest.workloads,
            ordered_payloads,
            strict=True,
        )
    )
    first_wave_hours = sum(
        reservation.reserved_cluster_hours for reservation in reservations
    )
    if first_wave_hours != REPRESENTATIVE_CANARY_FIRST_WAVE_CLUSTER_HOURS:
        raise ValueError(
            "representative first-wave worst-case cluster hours must equal "
            f"{REPRESENTATIVE_CANARY_FIRST_WAVE_CLUSTER_HOURS}, got "
            f"{first_wave_hours}"
        )
    return reservations


def validate_representative_canary_reservation(
    reservation: DatabricksClusterHourReservation,
    submit_payload: Mapping[str, Any],
) -> None:
    """Ledger callback that binds a reservation to one manifest workload."""

    if not isinstance(reservation, DatabricksClusterHourReservation):
        raise TypeError("reservation must be a DatabricksClusterHourReservation")
    try:
        workload = representative_canary_workload_manifest().workload_for_id(
            reservation.workload_id
        )
    except KeyError as exc:
        raise ValueError(
            f"unknown representative workload_id {reservation.workload_id!r}"
        ) from exc
    expected = databricks_submit_payload_reservation(
        submit_payload,
        attempt_id=reservation.attempt_id,
        workload_id=reservation.workload_id,
    )
    if reservation != expected:
        raise ValueError(
            "representative reservation must match the exact submit payload snapshot"
        )
    _validate_representative_canary_payload_contract(
        workload,
        submit_payload,
        reservation,
    )


def reserve_and_submit_representative_canary_workload(
    config: DatabricksWorkspaceConfig,
    submit_payload: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    attempt_id: str,
    workload_id: str,
    opener: DatabricksURLOpener | None = None,
) -> dict[str, Any]:
    """Atomically reserve and submit one canonical manifest workload payload."""

    try:
        representative_canary_workload_manifest().workload_for_id(workload_id)
    except KeyError as exc:
        raise ValueError(
            f"unknown representative workload_id {workload_id!r}"
        ) from exc
    return reserve_and_submit_databricks_run(
        config,
        submit_payload,
        ledger_path=ledger_path,
        attempt_id=attempt_id,
        workload_id=workload_id,
        reservation_validator=validate_representative_canary_reservation,
        opener=opener,
    )


def _validate_representative_canary_payload_contract(
    workload: RepresentativeCanaryWorkload,
    submit_payload: Mapping[str, Any],
    reservation: DatabricksClusterHourReservation,
) -> None:
    _require_exact_mapping_keys(
        submit_payload,
        _REPRESENTATIVE_SUBMIT_PAYLOAD_KEYS,
        "representative submit payload",
    )
    if reservation.run_timeout_seconds != 14_400:
        raise ValueError("representative run timeout_seconds must equal 14400")
    if reservation.task_timeout_seconds != (14_400,):
        raise ValueError(
            "representative payload must contain exactly one 14400-second task"
        )
    if reservation.reserved_cluster_hours != 4.0:
        raise ValueError("representative workload must reserve exactly 4 cluster-hours")

    tasks = _payload_mapping_sequence(
        submit_payload.get("tasks"),
        "representative submit_payload.tasks",
    )
    if len(tasks) != 1:
        raise ValueError("representative workload must use exactly one task")
    task = tasks[0]
    _require_exact_mapping_keys(
        task,
        _REPRESENTATIVE_TASK_KEYS,
        "representative task",
    )
    if task.get("max_retries") != 0:
        raise ValueError("representative task max_retries must equal 0")
    cluster = _payload_mapping(
        task.get("new_cluster"),
        "representative task.new_cluster",
    )
    _require_exact_mapping_keys(
        cluster,
        _REPRESENTATIVE_NEW_CLUSTER_KEYS,
        "representative task.new_cluster",
    )
    if cluster.get("spark_version") != REPRESENTATIVE_DATABRICKS_SPARK_VERSION:
        raise ValueError(
            "representative Spark runtime must equal "
            f"{REPRESENTATIVE_DATABRICKS_SPARK_VERSION!r}"
        )
    if cluster.get("data_security_mode") != "SINGLE_USER":
        raise ValueError(
            "representative cluster data_security_mode must equal 'SINGLE_USER'"
        )
    if not isinstance(cluster.get("single_user_name"), str) or not cluster.get(
        "single_user_name"
    ):
        raise ValueError("representative SINGLE_USER cluster requires single_user_name")
    for field_name in ("node_type_id", "driver_node_type_id"):
        if cluster.get(field_name) != workload.node_type_id:
            raise ValueError(
                f"representative {field_name} must equal {workload.node_type_id!r}"
            )
    if cluster.get("num_workers") != 0:
        raise ValueError("representative workload must use a single-node cluster")
    if cluster.get("spark_conf") != {
        "spark.master": "local[*]",
        "spark.databricks.cluster.profile": "singleNode",
    }:
        raise ValueError(
            "representative workload must use the fixed single-node Spark profile"
        )
    if cluster.get("aws_attributes") != {
        "availability": "ON_DEMAND",
        "zone_id": "auto",
    }:
        raise ValueError(
            "representative workload must use fixed on-demand AWS attributes "
            "without disk overrides"
        )
    forbidden_cluster_keys = tuple(
        str(key)
        for key in cluster
        if any(
            marker in str(key).lower()
            for marker in ("autoscale", "autoscaling", "disk", "ebs", "volume")
        )
    )
    if forbidden_cluster_keys:
        raise ValueError(
            "representative workload must not override autoscaling or disk shape: "
            + ", ".join(sorted(forbidden_cluster_keys))
        )
    spark_env_vars = _payload_mapping(
        cluster.get("spark_env_vars"),
        "representative task.new_cluster.spark_env_vars",
    )
    if spark_env_vars != {"DOCUMENT_KV_EVICT_PAGE_CACHE": "1"}:
        raise ValueError(
            "representative spark_env_vars must equal the fixed page-cache "
            "eviction environment"
        )
    spark_python_task = _payload_mapping(
        task.get("spark_python_task"),
        "representative task.spark_python_task",
    )
    _require_exact_mapping_keys(
        spark_python_task,
        _REPRESENTATIVE_SPARK_PYTHON_TASK_KEYS,
        "representative task.spark_python_task",
    )
    _require_persistent_databricks_path(
        spark_python_task.get("python_file"),
        "representative runner python_file",
        expected_basename=workload.runner_basename,
        required_path_component=workload.runner_sha256,
    )
    parameters = _payload_string_sequence(
        spark_python_task.get("parameters"),
        "representative runner parameters",
    )
    _validate_closed_representative_parameter_schema(workload, parameters)
    if _single_parameter_value(parameters, "--benchmark-id") != workload.workload_id:
        raise ValueError(
            "representative runner --benchmark-id must match the manifest workload_id"
        )
    _require_persistent_databricks_path(
        _single_parameter_value(parameters, "--output-dir"),
        "representative result output directory",
        expected_basename=workload.workload_id,
    )
    validated_representative_wheel_binding(
        _single_parameter_value(parameters, "--package-wheel-uri"),
        _single_parameter_value(parameters, "--package-wheel-sha256"),
    )
    _require_parameter_flag(parameters, "--representative-canary")
    if _single_parameter_value(
        parameters,
        "--representative-workload-profile",
    ) != workload.profile_id:
        raise ValueError("representative workload profile does not match manifest")
    if _single_parameter_value(parameters, "--hardware-target") != workload.hardware_target:
        raise ValueError("representative hardware target does not match manifest")
    if _single_parameter_value(parameters, "--local-root") != "/local_disk0":
        raise ValueError("representative local root must equal /local_disk0")
    common_runtime_values = {
        "--timeout-seconds": "240.0",
        "--import-probe-timeout-seconds": "180.0",
        "--server-start-timeout-seconds": "480.0",
        "--server-host": "127.0.0.1",
        "--server-port": "8000",
        "--client-host": "127.0.0.1",
    }
    for flag, expected_value in common_runtime_values.items():
        if _single_parameter_value(parameters, flag) != expected_value:
            raise ValueError(
                f"representative runner {flag} must equal {expected_value!r}"
            )
    for flag in ("--model-revision", "--tokenizer-revision"):
        revision = require_pinned_revision(
            _single_parameter_value(parameters, flag), flag
        )
        if revision != REPRESENTATIVE_CANARY_MODEL_REVISION:
            raise ValueError(
                f"representative {flag} must equal the approved revision"
            )

    if workload.serving_platform == "vllm":
        _validate_representative_vllm_payload_parameters(workload, parameters)
    else:
        _validate_representative_sglang_payload_parameters(workload, parameters)


def _validate_representative_vllm_payload_parameters(
    workload: RepresentativeCanaryWorkload,
    parameters: Sequence[str],
) -> None:
    profile_values = {
        "vllm-8k-64-v1": (8_192, 64, 8_512),
        "vllm-16k-256-v1": (16_384, 256, 16_896),
    }
    input_tokens, max_tokens, profile_max_model_len = profile_values[
        workload.profile_id
    ]
    expected_values = {
        "--max-tokens": max_tokens,
        "--benchmark-repeats": 3,
        "--request-parallelism": 1,
        "--max-num-seqs": 2,
    }
    for flag, expected in expected_values.items():
        if _parameter_integer(parameters, flag) != expected:
            raise ValueError(f"representative vLLM {flag} must equal {expected}")
    if _parameter_integer(parameters, "--max-model-len") != profile_max_model_len:
        raise ValueError("representative vLLM --max-model-len must match the profile")
    if float(_single_parameter_value(parameters, "--gpu-memory-utilization")) != 0.85:
        raise ValueError(
            "representative vLLM --gpu-memory-utilization must equal 0.85"
        )
    if _single_parameter_value(parameters, "--benchmark-evidence-policy") != "canary":
        raise ValueError("representative vLLM evidence policy must be canary")
    expected_runtime_values = {
        "--model-id": REPRESENTATIVE_CANARY_MODEL_ID,
        "--model-dtype": "bfloat16",
        "--kv-cache-dtype": "bfloat16",
        "--payload-cache-max-bytes": "0",
        "--benchmark-prefix-cache-salt-mode": "per_request",
        "--runtime-telemetry-interval-seconds": "1.0",
    }
    for flag, expected_text in expected_runtime_values.items():
        if _single_parameter_value(parameters, flag) != expected_text:
            raise ValueError(
                f"representative vLLM {flag} must equal {expected_text!r}"
            )
    if "--model-quantization" in parameters:
        raise ValueError("representative vLLM model quantization must be disabled")
    _require_parameter_flag(parameters, "--benchmark-force-max-tokens")
    _require_parameter_flag(parameters, "--allow-dataset-subset")
    for forbidden_flag in (
        "--benchmark-prewarm-cache-prefix",
        "--benchmark-cache-runtime-prompt",
    ):
        if forbidden_flag in parameters:
            raise ValueError(
                f"representative vLLM payload must not include {forbidden_flag}"
            )
    dataset_values = _parameter_values(parameters, "--dataset")
    if len(dataset_values) != 1 or not dataset_values[0].startswith("hotpotqa="):
        raise ValueError(
            "representative vLLM payload requires exactly one prepared HotpotQA dataset"
        )
    _require_persistent_databricks_path(
        dataset_values[0].removeprefix("hotpotqa="),
        "representative HotpotQA dataset",
        expected_basename=REPRESENTATIVE_HOTPOT_DATASET_BASENAME,
    )

    arm_values = _parameter_values(parameters, "--benchmark-arm-spec-json")
    if len(arm_values) != 1:
        raise ValueError("representative vLLM payload requires exactly one arm spec")
    try:
        raw_arm = json.loads(arm_values[0])
    except json.JSONDecodeError as exc:
        raise ValueError("representative vLLM arm spec must be valid JSON") from exc
    if not isinstance(raw_arm, Mapping):
        raise ValueError("representative vLLM arm spec must be an object")
    expected_arm = representative_canary_matrix().run_for_arm(workload.arm_id).arm_spec
    if benchmark_json_mapping_to_record(raw_arm) != benchmark_json_mapping_to_record(
        expected_arm
    ):
        raise ValueError("representative vLLM arm spec does not match manifest")
    _validate_representative_vllm_handoff_parameters(workload, parameters)

    provenance_text = _single_parameter_value(
        parameters,
        "--benchmark-manifest-provenance-json",
    )
    try:
        raw_provenance = json.loads(provenance_text)
    except json.JSONDecodeError as exc:
        raise ValueError("representative vLLM provenance must be valid JSON") from exc
    if not isinstance(raw_provenance, Mapping):
        raise ValueError("representative vLLM provenance must be an object")
    provenance = validated_benchmark_manifest_provenance(raw_provenance)
    expected_package_revisions = {
        package: version
        for package, version in (
            pin.split("==", 1) for pin in REPRESENTATIVE_VLLM_PACKAGE_PINS
        )
    }
    expected_provenance: dict[str, object] = {
        "input_tokens_target": input_tokens,
        "canonical_model_id": REPRESENTATIVE_CANARY_MODEL_ID,
        "model_revision": REPRESENTATIVE_CANARY_MODEL_REVISION,
        "tokenizer_id": REPRESENTATIVE_CANARY_MODEL_ID,
        "tokenizer_revision": REPRESENTATIVE_CANARY_MODEL_REVISION,
        "engine_id": "vllm",
        "engine_version": "0.23.0",
        "serving_platform": "vllm",
        "model_dtype": "bfloat16",
        "model_quantization": "none",
        "runtime_kv_dtype": "bfloat16",
        "package_revisions": expected_package_revisions,
    }
    for field_name, expected_value in expected_provenance.items():
        if provenance.get(field_name) != expected_value:
            raise ValueError(
                f"representative vLLM provenance {field_name} must equal "
                f"{expected_value!r}"
            )


def _validate_representative_sglang_payload_parameters(
    workload: RepresentativeCanaryWorkload,
    parameters: Sequence[str],
) -> None:
    if workload.arm_id != SGLANG_PAIRED_SMOKE_ARM:
        raise ValueError("representative SGLang workload must be the paired smoke")
    expected_values = {
        "--max-tokens": "32",
        "--context-length": "4096",
        "--mem-fraction-static": "0.85",
        "--live-benchmark-repeats": "2",
        "--sglang-attention-backend": "triton",
        "--sglang-sampling-backend": "pytorch",
        "--cache-prompt-text-mode": "logical",
        "--live-check-prompt-format": "qwen3_chat",
        "--live-check-request-mode": "chat",
        "--live-check-temperature": "0.0",
        "--flush-cache-timeout-seconds": "30.0",
    }
    for flag, expected in expected_values.items():
        if _single_parameter_value(parameters, flag) != expected:
            raise ValueError(
                f"representative SGLang {flag} must equal {expected!r}"
            )
    _require_parameter_flag(parameters, "--sglang-enable-deterministic-inference")
    for forbidden_flag in (
        "--no-flush-cache-before-cache-arm",
        "--no-flush-cache-before-canary",
    ):
        if forbidden_flag in parameters:
            raise ValueError(
                f"representative SGLang payload must not include {forbidden_flag}"
            )
    if "--baseline-only" in parameters:
        raise ValueError("representative SGLang workload must run the paired smoke")
    if _parameter_values(parameters, "--representative-package-pin") != (
        REPRESENTATIVE_SGLANG_PACKAGE_PINS
    ):
        raise ValueError(
            "representative SGLang payload must bind the approved package pins"
        )
    _require_parameter_flag(parameters, "--generate-live-handoff")
    if _single_parameter_value(
        parameters,
        "--live-handoff-generator-factory",
    ) != REPRESENTATIVE_HANDOFF_GENERATOR_FACTORY:
        raise ValueError(
            "representative SGLang payload must use the approved handoff generator"
        )
    if _single_parameter_value(parameters, "--live-handoff-dtype") != "bfloat16":
        raise ValueError("representative SGLang handoff dtype must be bfloat16")
    expected_handoff_values = {
        "--live-handoff-align-bytes": "4096",
        "--live-handoff-generation-timeout-seconds": "1800.0",
        "--sglang-hicache-page-size": "1",
        "--hicache-storage-prefetch-policy": "wait_complete",
        "--hicache-storage-prefetch-threshold": "1",
    }
    for flag, expected_value in expected_handoff_values.items():
        if _single_parameter_value(parameters, flag) != expected_value:
            raise ValueError(
                f"representative SGLang {flag} must equal {expected_value!r}"
            )
    _require_local_disk0_parameter(
        parameters,
        "--live-handoff-output-dir",
        "representative SGLang handoff output",
    )


def _validate_representative_vllm_handoff_parameters(
    workload: RepresentativeCanaryWorkload,
    parameters: Sequence[str],
) -> None:
    handoff_flags = (
        "--benchmark-handoff-generator-factory",
        "--benchmark-handoff-output-dir",
        "--benchmark-handoff-dtype",
        "--benchmark-handoff-align-bytes",
        "--benchmark-handoff-generation-timeout-seconds",
        "--benchmark-handoff-limit",
        "--benchmark-handoff-chunk-per-document",
        "--benchmark-handoff-cache-method",
        "--benchmark-handoff-allow-legacy-artifact-contract",
    )
    if workload.arm_id == BASELINE_PREFILL_ARM:
        unexpected = tuple(flag for flag in handoff_flags if flag in parameters)
        if unexpected:
            raise ValueError(
                "representative baseline payload must not generate cache handoffs"
            )
        return
    if _single_parameter_value(
        parameters,
        "--benchmark-handoff-generator-factory",
    ) != REPRESENTATIVE_HANDOFF_GENERATOR_FACTORY:
        raise ValueError(
            "representative cache arm must use the approved handoff generator"
        )
    _require_local_disk0_parameter(
        parameters,
        "--benchmark-handoff-output-dir",
        "representative vLLM handoff output",
    )
    if _single_parameter_value(parameters, "--benchmark-handoff-dtype") != "bfloat16":
        raise ValueError("representative vLLM handoff dtype must be bfloat16")
    expected_handoff_values = {
        "--benchmark-handoff-align-bytes": "4096",
        "--benchmark-handoff-generation-timeout-seconds": "1800.0",
    }
    for flag, expected_value in expected_handoff_values.items():
        if _single_parameter_value(parameters, flag) != expected_value:
            raise ValueError(
                f"representative vLLM {flag} must equal {expected_value!r}"
            )
    if "--benchmark-handoff-allow-legacy-artifact-contract" in parameters:
        raise ValueError(
            "representative cache arm requires the strict artifact contract"
        )
    expected_method = {
        FULL_PREFIX_CANARY_ARM: "full_prefix_prefill",
        VANILLA_CANARY_ARM: "vanilla_prefill",
    }[workload.arm_id]
    if _single_parameter_value(
        parameters,
        "--benchmark-handoff-cache-method",
    ) != expected_method:
        raise ValueError(
            "representative cache arm handoff method does not match its arm"
        )
    per_document = "--benchmark-handoff-chunk-per-document" in parameters
    if per_document != (workload.arm_id == VANILLA_CANARY_ARM):
        raise ValueError(
            "representative cache arm handoff topology does not match its arm"
        )


def _require_local_disk0_parameter(
    parameters: Sequence[str],
    flag: str,
    field_name: str,
) -> None:
    value = _single_parameter_value(parameters, flag)
    path = Path(value).resolve(strict=False)
    try:
        path.relative_to(Path("/local_disk0"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be under /local_disk0") from exc


def _validate_closed_representative_parameter_schema(
    workload: RepresentativeCanaryWorkload,
    parameters: Sequence[str],
) -> None:
    flags = tuple(value for value in parameters if value.startswith("--"))
    duplicates = sorted(flag for flag in set(flags) if flags.count(flag) != 1)
    if duplicates:
        raise ValueError(
            "representative runner flags must be unique: " + ", ".join(duplicates)
        )
    allowed = set(_REPRESENTATIVE_COMMON_PARAMETER_FLAGS)
    if workload.serving_platform == "vllm":
        allowed.update(_REPRESENTATIVE_VLLM_PARAMETER_FLAGS)
        if workload.arm_id != BASELINE_PREFILL_ARM:
            allowed.update(_REPRESENTATIVE_VLLM_HANDOFF_PARAMETER_FLAGS)
            if workload.arm_id == VANILLA_CANARY_ARM:
                allowed.add("--benchmark-handoff-chunk-per-document")
    else:
        allowed.update(_REPRESENTATIVE_SGLANG_PARAMETER_FLAGS)
    unsupported = sorted(set(flags) - allowed)
    if unsupported:
        raise ValueError(
            "unsupported representative runner flags: " + ", ".join(unsupported)
        )


def _require_persistent_databricks_path(
    value: object,
    field_name: str,
    *,
    expected_basename: str,
    required_path_component: str | None = None,
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty path")
    if value.startswith("dbfs:/"):
        path_text = value.removeprefix("dbfs:")
    elif value.startswith(("/dbfs/", "/Volumes/")):
        path_text = value
    else:
        raise ValueError(
            f"{field_name} must use persistent dbfs:/, /dbfs/, or /Volumes/ storage"
        )
    path = Path(path_text)
    if ".." in path.parts:
        raise ValueError(f"{field_name} must not contain parent traversal")
    if path.name != expected_basename:
        raise ValueError(
            f"{field_name} final component must equal {expected_basename!r}"
        )
    if required_path_component is not None and required_path_component not in path.parts:
        raise ValueError(
            f"{field_name} must contain the approved SHA-256 as a path component"
        )
    return value


def _payload_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _require_exact_mapping_keys(
    value: Mapping[str, Any],
    expected_keys: frozenset[str],
    field_name: str,
) -> None:
    actual_keys = frozenset(value)
    if actual_keys == expected_keys:
        return
    missing = sorted(expected_keys - actual_keys)
    unsupported = sorted(actual_keys - expected_keys)
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unsupported:
        details.append("unsupported " + ", ".join(unsupported))
    raise ValueError(f"{field_name} must use exact canonical keys: {'; '.join(details)}")


def _payload_mapping_sequence(
    value: object,
    field_name: str,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array")
    records: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{index}] must be an object")
        records.append(item)
    return tuple(records)


def _payload_string_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must contain only strings")
    return tuple(cast(str, item) for item in value)


def _parameter_values(parameters: Sequence[str], flag: str) -> tuple[str, ...]:
    values: list[str] = []
    for index, value in enumerate(parameters):
        if value != flag:
            continue
        if index + 1 >= len(parameters) or parameters[index + 1].startswith("--"):
            raise ValueError(f"representative runner flag {flag} requires a value")
        values.append(parameters[index + 1])
    return tuple(values)


def _single_parameter_value(parameters: Sequence[str], flag: str) -> str:
    values = _parameter_values(parameters, flag)
    if len(values) != 1:
        raise ValueError(f"representative runner requires exactly one {flag}")
    return values[0]


def _parameter_integer(parameters: Sequence[str], flag: str) -> int:
    value = _single_parameter_value(parameters, flag)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"representative runner {flag} must be an integer") from exc


def _require_parameter_flag(parameters: Sequence[str], flag: str) -> None:
    if parameters.count(flag) != 1:
        raise ValueError(f"representative runner requires exactly one {flag}")

def require_pinned_revision(value: object, field_name: str) -> str:
    """Return an immutable revision or reject missing/unresolved identities."""

    raw_revision = _required_string(value, field_name)
    revision = raw_revision.strip()
    if not revision:
        raise ValueError(f"{field_name} must be non-empty")
    if revision != raw_revision:
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    if revision == UNRESOLVED_IDENTITY:
        raise ValueError(f"{field_name} must not be {UNRESOLVED_IDENTITY!r}")
    return revision


def resolved_layout_rope_provenance(layout: object) -> dict[str, Any]:
    """Return a validated nullable-pair RoPE binding from a resolved KV layout."""

    rope_theta = getattr(layout, "rope_theta", None)
    rope_rotary_dim = getattr(layout, "rope_rotary_dim", None)
    if (rope_theta is None) != (rope_rotary_dim is None):
        raise ValueError(
            "resolved layout rope_theta and rope_rotary_dim must be provided together"
        )
    if rope_theta is None:
        return {}
    record = validated_benchmark_manifest_provenance(
        {
            "rope_theta": rope_theta,
            "rope_rotary_dim": rope_rotary_dim,
        }
    )
    return benchmark_json_mapping_to_record(record)


def build_handoff_topology_attestation(
    input_jsonl: str | Path,
    manifest: BenchmarkHandoffManifest,
    *,
    token_counter: Callable[[str], int],
) -> dict[str, Any]:
    """Build a path-free, prompt-free topology record for generated handoffs."""

    if not isinstance(manifest, BenchmarkHandoffManifest):
        raise TypeError("manifest must be a BenchmarkHandoffManifest")
    if not callable(token_counter):
        raise TypeError("token_counter must be callable")
    examples = load_benchmark_jsonl(input_jsonl)
    return _build_handoff_topology_attestation_from_examples(
        examples,
        manifest,
        token_counter=token_counter,
    )


def build_handoff_topology_attestation_record(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the closed topology schema from already-sanitized example rows."""

    return _handoff_topology_record(examples)


def generator_token_counter(generator: object) -> Callable[[str], int]:
    """Return an exact token counter from a handoff generator, when exposed."""

    counter = getattr(generator, "logical_token_count", None)
    if callable(counter):
        return cast(Callable[[str], int], counter)
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is None or not callable(tokenizer):
        raise TypeError(
            "handoff topology attestation requires a generator token counter or tokenizer"
        )

    def count(text: str) -> int:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=bool(getattr(generator, "add_special_tokens", False)),
        )
        input_ids = (
            encoded.get("input_ids")
            if isinstance(encoded, Mapping)
            else getattr(encoded, "input_ids", None)
        )
        if input_ids is None:
            raise ValueError("generator tokenizer output must include input_ids")
        shape = getattr(input_ids, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[-1])
        values = input_ids
        tolist = getattr(values, "tolist", None)
        if callable(tolist):
            values = tolist()
        if isinstance(values, Sequence) and not isinstance(
            values, (str, bytes, bytearray)
        ):
            if len(values) == 1 and isinstance(values[0], Sequence):
                return len(values[0])
            return len(values)
        raise TypeError("generator tokenizer input_ids must be a token sequence")

    return count


def merge_handoff_topology_attestations(
    attestations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge validated topology records with deterministic per-example ordering."""

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for attestation in attestations:
        validated = validate_handoff_topology_attestation(attestation)
        for raw_row in validated["examples"]:
            row = dict(raw_row)
            key = (
                str(row["example_key_sha256"]),
                str(row["method_id"]),
            )
            if key in seen:
                raise ValueError("handoff topology attestations contain a duplicate example")
            seen.add(key)
            rows.append(row)
    return _handoff_topology_record(rows)


def validate_handoff_topology_attestation(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the closed, sanitized handoff-topology evidence schema."""

    if not isinstance(record, Mapping):
        raise TypeError("handoff topology attestation must be an object")
    unexpected = sorted(str(key) for key in record if key not in _HANDOFF_TOPOLOGY_RECORD_KEYS)
    missing = sorted(key for key in _HANDOFF_TOPOLOGY_RECORD_KEYS if key not in record)
    if unexpected or missing:
        raise ValueError(
            "handoff topology attestation keys do not match the closed schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if record.get("record_type") != HANDOFF_TOPOLOGY_ATTESTATION_RECORD_TYPE:
        raise ValueError(
            "handoff topology attestation record_type must be "
            f"{HANDOFF_TOPOLOGY_ATTESTATION_RECORD_TYPE!r}"
        )
    schema_version = record.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != HANDOFF_TOPOLOGY_ATTESTATION_SCHEMA_VERSION
    ):
        raise ValueError(
            "handoff topology attestation schema_version must be "
            f"{HANDOFF_TOPOLOGY_ATTESTATION_SCHEMA_VERSION}"
        )
    raw_rows = _mapping_sequence(record.get("examples"), "handoff topology examples")
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows):
        unexpected_row = sorted(
            str(key) for key in raw_row if key not in _HANDOFF_TOPOLOGY_EXAMPLE_KEYS
        )
        missing_row = sorted(
            key for key in _HANDOFF_TOPOLOGY_EXAMPLE_KEYS if key not in raw_row
        )
        if unexpected_row or missing_row:
            raise ValueError(
                f"handoff topology examples[{index}] keys do not match the closed "
                f"schema; missing={missing_row}, unexpected={unexpected_row}"
            )
        row = dict(raw_row)
        for field_name in (
            "example_key_sha256",
            "method_config_digest",
            "artifact_id",
            "logical_prompt_sha256",
        ):
            _require_sha256(row.get(field_name), f"examples[{index}].{field_name}")
        for field_name in ("method_id", "method_version"):
            _required_string(row.get(field_name), f"examples[{index}].{field_name}")
        for field_name in (
            "document_count",
            "segment_count",
            "logical_token_count",
        ):
            value = row.get(field_name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"examples[{index}].{field_name} must be a non-negative integer"
                )
        if row["document_count"] <= 0:
            raise ValueError(f"examples[{index}].document_count must be positive")
        try:
            method = default_method_registry().get(row["method_id"])
        except KeyError:
            # Custom registries own custom topology validation at generation.
            # The closed attestation remains readable without global registration.
            pass
        else:
            method.validate_handoff_segment_counts(
                document_count=row["document_count"],
                segment_count=row["segment_count"],
            )
        rows.append(row)
    canonical_rows = _sorted_topology_rows(rows)
    if rows != canonical_rows:
        raise ValueError("handoff topology examples must use deterministic ordering")
    example_count = record.get("example_count")
    if type(example_count) is not int or example_count != len(rows):
        raise ValueError("handoff topology example_count does not match examples")
    expected_digest = _canonical_sha256(rows)
    if record.get("examples_sha256") != expected_digest:
        raise ValueError("handoff topology examples_sha256 does not match examples")
    return {
        "record_type": HANDOFF_TOPOLOGY_ATTESTATION_RECORD_TYPE,
        "schema_version": HANDOFF_TOPOLOGY_ATTESTATION_SCHEMA_VERSION,
        "example_count": len(rows),
        "examples_sha256": expected_digest,
        "examples": rows,
    }


def _build_handoff_topology_attestation_from_examples(
    examples: Sequence[Any],
    manifest: BenchmarkHandoffManifest,
    *,
    token_counter: Callable[[str], int],
) -> dict[str, Any]:
    entries = {entry.key: entry for entry in manifest.entries}
    expected_keys = {(example.dataset, example.example_id) for example in examples}
    if set(entries) != expected_keys or len(entries) != len(examples):
        raise ValueError("handoff topology manifest must exactly match input examples")
    rows: list[dict[str, Any]] = []
    for example in examples:
        entry = entries[(example.dataset, example.example_id)]
        record = _handoff_record(entry)
        handle = _required_mapping(record.get("handle"), "handoff.handle")
        segments = handle.get("segments")
        if not isinstance(segments, Sequence) or isinstance(
            segments, (str, bytes, bytearray)
        ):
            raise TypeError("handoff.handle.segments must be an array")
        identity = ArtifactIdentity.from_record(
            _required_mapping(
                handle.get("artifact_identity"),
                "handoff.handle.artifact_identity",
            )
        )
        if entry.cache_method and entry.cache_method != identity.method_id:
            raise ValueError("handoff manifest method does not match ArtifactIdentity")
        if entry.artifact_id and entry.artifact_id != identity.artifact_id:
            raise ValueError("handoff manifest artifact does not match ArtifactIdentity")
        prompt = build_prompt_parts(example).prefill_prompt
        logical_token_count = token_counter(prompt)
        if type(logical_token_count) is not int or logical_token_count < 0:
            raise ValueError("token_counter must return a non-negative integer")
        rows.append(
            {
                "example_key_sha256": _canonical_sha256(
                    {"dataset": example.dataset, "example_id": example.example_id}
                ),
                "method_id": identity.method_id,
                "method_version": identity.method_version,
                "method_config_digest": identity.method_config_digest,
                "artifact_id": identity.artifact_id,
                "document_count": len(example.documents),
                "segment_count": len(segments),
                "logical_token_count": logical_token_count,
                "logical_prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    return _handoff_topology_record(rows)


def _handoff_topology_record(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    canonical_rows = _sorted_topology_rows(rows)
    record = {
        "record_type": HANDOFF_TOPOLOGY_ATTESTATION_RECORD_TYPE,
        "schema_version": HANDOFF_TOPOLOGY_ATTESTATION_SCHEMA_VERSION,
        "example_count": len(canonical_rows),
        "examples_sha256": _canonical_sha256(canonical_rows),
        "examples": canonical_rows,
    }
    return validate_handoff_topology_attestation(record)


def _sorted_topology_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("example_key_sha256", "")),
            str(row.get("method_id", "")),
            str(row.get("artifact_id", "")),
        ),
    )


def _require_sha256(value: Any, field_name: str) -> str:
    digest = _required_string(value, field_name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


def validated_benchmark_arm_specs(
    values: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Validate arbitrary runner arm specs before a smoke job is submitted."""

    records: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise TypeError(f"benchmark_arm_specs[{index}] must be a mapping")
        # A JSON round trip rejects non-serializable values before a remote job starts.
        record = json.loads(json.dumps(dict(value), sort_keys=True))
        records.append(record)
    parse_benchmark_arm_specs(records)
    return tuple(_deep_freeze_json_mapping(record) for record in records)


def validated_benchmark_manifest_provenance(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the runner manifest context carried through the smoke wrapper."""

    if not isinstance(value, Mapping):
        raise TypeError("benchmark_manifest_provenance must be a mapping")
    unknown = set(value).difference(_PROVENANCE_FIELDS)
    if unknown:
        raise ValueError(
            f"benchmark_manifest_provenance has unknown fields: {sorted(unknown)}"
        )
    record = json.loads(json.dumps(_deep_thaw_json_value(value), sort_keys=True))
    for field_name, _flag in _PROVENANCE_STRING_FLAGS:
        if field_name in record:
            string = record[field_name]
            if not isinstance(string, str):
                raise TypeError(f"benchmark_manifest_provenance.{field_name} must be a string")
            if field_name != "varied_setting" and not string:
                raise ValueError(
                    f"benchmark_manifest_provenance.{field_name} must be non-empty"
                )
    package_revisions = record.get("package_revisions", {})
    if not isinstance(package_revisions, Mapping):
        raise TypeError("benchmark_manifest_provenance.package_revisions must be a mapping")
    for package, revision in package_revisions.items():
        _required_string(package, "benchmark_manifest_provenance package name")
        _required_string(revision, f"benchmark_manifest_provenance.package_revisions.{package}")
    if "input_tokens_target" in record:
        target = record["input_tokens_target"]
        if type(target) is not int or target <= 0:
            raise ValueError(
                "benchmark_manifest_provenance.input_tokens_target must be positive"
            )
    for field_name, _flag in _PROVENANCE_INTEGER_FLAGS:
        if field_name in record:
            integer_value = record[field_name]
            if type(integer_value) is not int or integer_value <= 0:
                raise ValueError(
                    f"benchmark_manifest_provenance.{field_name} must be positive"
                )
    for field_name, _flag in _PROVENANCE_FLOAT_FLAGS:
        if field_name in record:
            number = record[field_name]
            if (
                not isinstance(number, (int, float))
                or isinstance(number, bool)
                or not math.isfinite(float(number))
                or number <= 0
            ):
                raise ValueError(
                    f"benchmark_manifest_provenance.{field_name} must be a positive "
                    "finite number"
                )
            record[field_name] = float(number)
    if ("rope_theta" in record) != ("rope_rotary_dim" in record):
        raise ValueError(
            "benchmark_manifest_provenance.rope_theta and rope_rotary_dim must "
            "be provided together"
        )
    if "complete_dataset_split" in record and type(record["complete_dataset_split"]) is not bool:
        raise TypeError(
            "benchmark_manifest_provenance.complete_dataset_split must be a boolean"
        )
    scopes = record.get("measurement_scopes", ())
    if not isinstance(scopes, Sequence) or isinstance(scopes, (str, bytes, bytearray)):
        raise TypeError("benchmark_manifest_provenance.measurement_scopes must be an array")
    normalized_scopes = tuple(scopes)
    if scopes and (
        any(scope not in {"latency", "quality", "resource"} for scope in normalized_scopes)
        or len(set(normalized_scopes)) != len(normalized_scopes)
    ):
        raise ValueError(
            "benchmark_manifest_provenance.measurement_scopes must contain unique "
            "latency, quality, or resource values"
        )
    if "measurement_scopes" in record and not normalized_scopes:
        raise ValueError("benchmark_manifest_provenance.measurement_scopes must not be empty")
    if record.get("comparison_mode") not in {
        None,
        "methods_same_setting",
        "single_method_setting_variation",
    }:
        raise ValueError("benchmark_manifest_provenance.comparison_mode is invalid")
    if (
        record.get("varied_setting")
        and record.get("comparison_mode", "methods_same_setting")
        != "single_method_setting_variation"
    ):
        raise ValueError(
            "benchmark_manifest_provenance.varied_setting requires "
            "single_method_setting_variation"
        )
    return _deep_freeze_json_mapping(record)


def benchmark_json_mapping_to_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached JSON object from an immutable benchmark mapping."""

    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    return {
        str(key): _deep_thaw_json_value(item)
        for key, item in value.items()
    }


def benchmark_manifest_provenance_runner_args(
    value: Mapping[str, Any],
) -> tuple[str, ...]:
    """Translate validated provenance into stable benchmark-runner arguments."""

    record = validated_benchmark_manifest_provenance(value)
    args: list[str] = []
    for field_name, flag in _PROVENANCE_STRING_FLAGS:
        if field_name in record:
            args.extend((flag, record[field_name]))
    for field_name, flag in _PROVENANCE_INTEGER_FLAGS:
        if field_name in record:
            args.extend((flag, str(record[field_name])))
    for field_name, flag in _PROVENANCE_FLOAT_FLAGS:
        if field_name in record:
            args.extend((flag, str(record[field_name])))
    for package, revision in sorted(record.get("package_revisions", {}).items()):
        args.extend(("--package-revision", f"{package}={revision}"))
    if "input_tokens_target" in record:
        args.extend(("--input-tokens-target", str(record["input_tokens_target"])))
    if record.get("complete_dataset_split") is True:
        args.append("--complete-dataset-split")
    for scope in record.get("measurement_scopes", ()):
        args.extend(("--measurement-scope", scope))
    return tuple(args)


def prepare_representative_canary_inputs(
    input_jsonl: str | Path,
    *,
    full_prefix_manifest_json: str | Path,
    vanilla_manifest_json: str | Path,
    output_dir: str | Path,
    dataset: str | None = None,
    input_tokens_target: int,
    token_counter: Callable[[str], int],
    tokenizer_id: str,
    tokenizer_revision: str,
    tokenizer_add_special_tokens: bool,
) -> RepresentativeCanaryPreparedInputs:
    """Prepare a validated N-way file and one-arm projections for isolated jobs."""

    records = _read_jsonl_records(input_jsonl)
    if not records:
        raise ValueError("representative canary input must contain at least one row")
    for index, record in enumerate(records, start=1):
        if record.get("kv_transfer_params"):
            raise ValueError(f"input line {index} already contains legacy kv_transfer_params")
        if record.get("arm_kv_transfer_params"):
            raise ValueError(f"input line {index} already contains arm_kv_transfer_params")
    examples = load_benchmark_jsonl(
        input_jsonl,
        dataset=dataset,
        require_dataset=dataset is not None,
    )
    if len(examples) < 2:
        raise ValueError("representative canary input must contain at least two distinct examples")
    if type(input_tokens_target) is not int or input_tokens_target <= 0:
        raise ValueError("input_tokens_target must be a positive integer")
    if not callable(token_counter):
        raise TypeError("token_counter must be callable")
    tokenizer_id = _required_string(tokenizer_id, "tokenizer_id")
    tokenizer_revision = _required_string(tokenizer_revision, "tokenizer_revision")
    if type(tokenizer_add_special_tokens) is not bool:
        raise TypeError("tokenizer_add_special_tokens must be a boolean")
    for example in examples:
        if len(example.documents) < 2:
            raise ValueError(
                f"representative canary example {example.dataset}:{example.example_id} "
                "must contain at least two documents"
            )
    token_rows = _logical_token_rows(
        examples,
        token_counter=token_counter,
        input_tokens_target=input_tokens_target,
    )
    keys = tuple((example.dataset, example.example_id) for example in examples)
    document_counts = {
        (example.dataset, example.example_id): len(example.documents)
        for example in examples
    }
    full_manifest = read_benchmark_handoff_manifest_json(full_prefix_manifest_json)
    vanilla_manifest = read_benchmark_handoff_manifest_json(vanilla_manifest_json)
    _validate_manifest_keys(full_manifest, keys, label="full-prefix")
    _validate_manifest_keys(vanilla_manifest, keys, label="vanilla")
    artifact_rows = _validate_manifest_pair(
        full_manifest,
        vanilla_manifest,
        document_counts=document_counts,
    )

    # Sequential arm enrichment is intentional: the combined file exercises the
    # N-way request contract, while projections below prevent unknown-arm params
    # from leaking into independently submitted one-arm runs.
    combined = enrich_benchmark_records_with_handoffs(
        records,
        full_manifest,
        dataset=dataset,
        arm_id=FULL_PREFIX_CANARY_ARM,
    )
    combined = enrich_benchmark_records_with_handoffs(
        combined,
        vanilla_manifest,
        dataset=dataset,
        arm_id=VANILLA_CANARY_ARM,
    )
    _validate_combined_params(combined)

    output_base = local_path(str(output_dir))
    output_base.mkdir(parents=True, exist_ok=True)
    combined_path = output_base / "representative-canary.combined.jsonl"
    arm_paths = {
        arm_id: output_base / f"representative-canary.{arm_id.replace(':', '-')}.jsonl"
        for arm_id in REPRESENTATIVE_CANARY_ARM_IDS
    }
    _write_and_reload_canary_jsonl(combined_path, combined)
    _write_and_reload_canary_jsonl(
        arm_paths[BASELINE_PREFILL_ARM],
        _project_records(combined, BASELINE_PREFILL_ARM),
    )
    _write_and_reload_canary_jsonl(
        arm_paths[FULL_PREFIX_CANARY_ARM],
        _project_records(combined, FULL_PREFIX_CANARY_ARM),
    )
    _write_and_reload_canary_jsonl(
        arm_paths[VANILLA_CANARY_ARM],
        _project_records(combined, VANILLA_CANARY_ARM),
    )

    logical_sample_digest = _logical_sample_digest(examples)
    topology_attestations = {
        FULL_PREFIX_CANARY_ARM: _build_handoff_topology_attestation_from_examples(
            examples,
            full_manifest,
            token_counter=token_counter,
        ),
        VANILLA_CANARY_ARM: _build_handoff_topology_attestation_from_examples(
            examples,
            vanilla_manifest,
            token_counter=token_counter,
        ),
    }
    preparation_manifest_path = output_base / "representative-canary-inputs.json"
    preparation_record = {
        "record_type": REPRESENTATIVE_CANARY_INPUT_RECORD_TYPE,
        "schema_version": 1,
        "logical_sample_digest": logical_sample_digest,
        "example_count": len(examples),
        "tokenizer": {
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
            "add_special_tokens": tokenizer_add_special_tokens,
        },
        "input_tokens_target": input_tokens_target,
        "logical_token_counts": token_rows,
        "combined_jsonl": str(combined_path),
        "arm_jsonl": {arm_id: str(path) for arm_id, path in arm_paths.items()},
        "arms": [
            {
                "arm_id": run.arm_id,
                "method_id": run.method_id or None,
                "expected_segments": run.expected_segments,
                "offline_costs": (
                    None
                    if run.method_id == ""
                    else {
                        "artifact_bytes": _manifest_storage_bytes(
                            full_manifest
                            if run.arm_id == FULL_PREFIX_CANARY_ARM
                            else vanilla_manifest
                        )
                    }
                ),
            }
            for run in representative_canary_matrix().runs
        ],
        "artifacts": artifact_rows,
        "handoff_topology_attestations": topology_attestations,
    }
    preparation_manifest_path.write_text(
        json.dumps(preparation_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RepresentativeCanaryPreparedInputs(
        combined_jsonl=combined_path,
        arm_jsonl=arm_paths,
        preparation_manifest_json=preparation_manifest_path,
        logical_sample_digest=logical_sample_digest,
        example_count=len(examples),
    )


def aggregate_isolated_canary_results(
    results: Mapping[str, Mapping[str, Any] | str | Path],
    *,
    evidence_policy: Literal["smoke", "canary", "publication"] = "canary",
    cache_state_attestations: Iterable[Any] = (),
    artifact_identities: Mapping[str, Any] | None = None,
    method_registry: MethodRegistry | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and merge one physical result per arm into a canonical run."""

    if set(results) != set(REPRESENTATIVE_CANARY_ARM_IDS):
        missing = sorted(set(REPRESENTATIVE_CANARY_ARM_IDS).difference(results))
        unexpected = sorted(set(results).difference(REPRESENTATIVE_CANARY_ARM_IDS))
        raise ValueError(
            f"isolated canary results must contain the exact matrix; "
            f"missing={missing}, unexpected={unexpected}"
        )
    records = {
        arm_id: _result_record(results[arm_id], arm_id=arm_id)
        for arm_id in REPRESENTATIVE_CANARY_ARM_IDS
    }
    _validate_shared_result_identity(records)
    measurements: list[dict[str, Any]] = []
    for arm_id in REPRESENTATIVE_CANARY_ARM_IDS:
        record = records[arm_id]
        arm_measurements = _mapping_sequence(record.get("measurements"), "measurements")
        if any(item.get("arm_id") != arm_id for item in arm_measurements):
            raise ValueError(f"result {arm_id!r} contains measurements from another arm")
        _validate_measurement_method_identity(arm_id, arm_measurements)
        measurements.extend(dict(item) for item in arm_measurements)
        _isolated_execution_window(
            record,
            arm_id=arm_id,
            measurements=arm_measurements,
        )
    _validate_measurement_pairing(measurements)
    _validate_distinct_result_artifacts(measurements)
    aggregate = merge_isolated_benchmark_run_records(
        tuple(records[arm_id] for arm_id in REPRESENTATIVE_CANARY_ARM_IDS),
        reference_arm_id=BASELINE_PREFILL_ARM,
        policy=evidence_policy,
        cache_state_attestations=cache_state_attestations,
        artifact_identities=artifact_identities,
        method_registry=method_registry,
    )
    aggregate_issues = benchmark_record_aggregate_issues(aggregate)
    if aggregate_issues:
        raise RuntimeError(
            "canonical isolated merge produced inconsistent aggregates: "
            + "; ".join(aggregate_issues)
        )
    if output_json is not None:
        path = local_path(str(output_json))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return aggregate


def _method_arm_spec(*, arm_id: str, method: Any, physical_transform_id: str) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "uses_cache": True,
        "description": method.description,
        "cache_method": method.method_id,
        "connector_mode": method.connector_mode,
        "variant_id": "default",
        "implementation_kind": "cachet",
        "method_version": method.artifact_version,
        "method_config_digest": method_config_digest({}),
        "physical_transform_id": physical_transform_id,
        "physical_transform_version": "1",
    }


def _method_for_arm(arm_id: str) -> str:
    if arm_id == BASELINE_PREFILL_ARM:
        return ""
    if arm_id == FULL_PREFIX_CANARY_ARM:
        return str(CacheGenerationMethod.FULL_PREFIX_PREFILL.value)
    if arm_id == VANILLA_CANARY_ARM:
        return str(CacheGenerationMethod.VANILLA_PREFILL.value)
    raise ValueError(f"unsupported representative canary arm {arm_id!r}")


def _read_jsonl_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with local_path(str(path)).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"input line {line_number} is invalid JSON: {exc.msg}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"input line {line_number} must be an object")
            records.append(dict(value))
    return tuple(records)


def _validate_manifest_keys(
    manifest: BenchmarkHandoffManifest,
    expected_keys: Sequence[tuple[str, str]],
    *,
    label: str,
) -> None:
    actual = tuple(entry.key for entry in manifest.entries)
    if set(actual) != set(expected_keys) or len(actual) != len(expected_keys):
        raise ValueError(
            f"{label} manifest examples must exactly match the canary input; "
            f"expected={sorted(expected_keys)}, actual={sorted(actual)}"
        )


def _validate_manifest_pair(
    full_manifest: BenchmarkHandoffManifest,
    vanilla_manifest: BenchmarkHandoffManifest,
    *,
    document_counts: Mapping[tuple[str, str], int],
) -> list[dict[str, Any]]:
    full_entries = {entry.key: entry for entry in full_manifest.entries}
    vanilla_entries = {entry.key: entry for entry in vanilla_manifest.entries}
    rows: list[dict[str, Any]] = []
    for key in sorted(document_counts):
        full = full_entries[key]
        vanilla = vanilla_entries[key]
        full_record = _validate_handoff_entry(
            full,
            method_id=CacheGenerationMethod.FULL_PREFIX_PREFILL.value,
            expected_segments=1,
        )
        vanilla_record = _validate_handoff_entry(
            vanilla,
            method_id=CacheGenerationMethod.VANILLA_PREFILL.value,
            expected_segments=document_counts[key],
        )
        if full.artifact_id == vanilla.artifact_id:
            raise ValueError(f"canary artifacts for {key!r} must be method-distinct")
        if full.request_id == vanilla.request_id:
            raise ValueError(f"canary handoff request ids for {key!r} must be distinct")
        if full.handoff_json and full.handoff_json == vanilla.handoff_json:
            raise ValueError(f"canary handoff JSON references for {key!r} must be distinct")
        if full.payload_uri and full.payload_uri == vanilla.payload_uri:
            raise ValueError(f"canary payload references for {key!r} must be distinct")
        rows.append(
            {
                "dataset": key[0],
                "example_id": key[1],
                "document_count": document_counts[key],
                "full_prefix_artifact_id": full.artifact_id,
                "vanilla_artifact_id": vanilla.artifact_id,
                "full_prefix_segments": len(full_record["handle"]["segments"]),
                "vanilla_segments": len(vanilla_record["handle"]["segments"]),
            }
        )
    return rows


def _validate_handoff_entry(
    entry: BenchmarkHandoffEntry,
    *,
    method_id: str,
    expected_segments: int,
) -> Mapping[str, Any]:
    if entry.cache_method != method_id:
        raise ValueError(
            f"handoff {entry.key!r} cache_method must be {method_id!r}; "
            f"got {entry.cache_method!r}"
        )
    if not entry.artifact_id:
        raise ValueError(f"handoff {entry.key!r} must declare artifact_id")
    record = _handoff_record(entry)
    handle = _required_mapping(record.get("handle"), "handoff.handle")
    if handle.get("cache_method") != method_id:
        raise ValueError(f"handoff {entry.key!r} handle.cache_method must be {method_id!r}")
    metadata = _required_mapping(handle.get("metadata"), "handoff.handle.metadata")
    if metadata.get("cachet.benchmark.dataset") != entry.dataset:
        raise ValueError(f"handoff {entry.key!r} dataset metadata does not match")
    if metadata.get("cachet.benchmark.example_id") != entry.example_id:
        raise ValueError(f"handoff {entry.key!r} example metadata does not match")
    reuse_plan = _required_mapping(record.get("reuse_plan"), "handoff.reuse_plan")
    if reuse_plan.get("method_id") != method_id:
        raise ValueError(f"handoff {entry.key!r} reuse_plan.method_id must be {method_id!r}")
    segments = handle.get("segments")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes, bytearray)):
        raise TypeError(f"handoff {entry.key!r} handle.segments must be an array")
    if len(segments) != expected_segments:
        raise ValueError(
            f"handoff {entry.key!r} for {method_id!r} must contain "
            f"{expected_segments} segments; got {len(segments)}"
        )
    identity_record = _required_mapping(
        handle.get("artifact_identity"),
        "handoff.handle.artifact_identity",
    )
    identity = ArtifactIdentity.from_record(identity_record)
    if identity.method_id != method_id:
        raise ValueError(f"handoff {entry.key!r} artifact method must be {method_id!r}")
    if identity.artifact_id != entry.artifact_id:
        raise ValueError(f"handoff {entry.key!r} artifact_id does not match its handle")
    return record


def _handoff_record(entry: BenchmarkHandoffEntry) -> Mapping[str, Any]:
    if entry.handoff_record is not None:
        return entry.handoff_record
    assert entry.handoff_json is not None
    try:
        record = json.loads(local_path(entry.handoff_json).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"handoff JSON for {entry.key!r} is invalid: {exc.msg}"
        ) from exc
    return _required_mapping(record, "handoff record")


def _manifest_storage_bytes(manifest: BenchmarkHandoffManifest) -> int | None:
    paths: set[Path] = set()
    for entry in manifest.entries:
        if entry.payload_uri is None:
            continue
        try:
            paths.add(local_path(entry.payload_uri))
        except (TypeError, ValueError):
            return None
    if not paths:
        return None
    sizes: list[int] = []
    for path in paths:
        try:
            sizes.append(path.stat().st_size)
        except OSError:
            return None
    return sum(sizes)


def _validate_combined_params(records: Sequence[Mapping[str, Any]]) -> None:
    for index, record in enumerate(records, start=1):
        params = _required_mapping(
            record.get("arm_kv_transfer_params"),
            f"line {index}.arm_kv_transfer_params",
        )
        if set(params) != {FULL_PREFIX_CANARY_ARM, VANILLA_CANARY_ARM}:
            raise ValueError(
                f"line {index} must contain exactly both representative cache-arm handoffs"
            )
        for arm_id in (FULL_PREFIX_CANARY_ARM, VANILLA_CANARY_ARM):
            arm_params = _required_mapping(params.get(arm_id), f"line {index}.{arm_id}")
            if arm_params.get(DOCUMENT_KV_CACHE_METHOD_PARAM) != _method_for_arm(arm_id):
                raise ValueError(f"line {index} {arm_id!r} has the wrong cache method")
            if not arm_params.get(DOCUMENT_KV_ARTIFACT_ID_PARAM):
                raise ValueError(f"line {index} {arm_id!r} is missing artifact identity")


def _project_records(
    records: Sequence[Mapping[str, Any]],
    arm_id: str,
) -> tuple[dict[str, Any], ...]:
    projected: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row.pop("kv_transfer_params", None)
        if arm_id == BASELINE_PREFILL_ARM:
            row.pop("arm_kv_transfer_params", None)
        else:
            params = _required_mapping(row["arm_kv_transfer_params"], "arm_kv_transfer_params")
            row["kv_transfer_params"] = dict(params[arm_id])
            row.pop("arm_kv_transfer_params", None)
        projected.append(row)
    return tuple(projected)


def _logical_sample_digest(examples: Sequence[Any]) -> str:
    return _canonical_sha256(
        [
            {
                "dataset": example.dataset,
                "example_id": example.example_id,
                "logical_prompt": build_prompt_parts(example).prefill_prompt,
                "query": example.query,
                "expected_answer": example.expected_answer,
                "metadata": dict(example.metadata),
            }
            for example in examples
        ]
    )


def _logical_token_rows(
    examples: Sequence[Any],
    *,
    token_counter: Callable[[str], int],
    input_tokens_target: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        prompt = build_prompt_parts(example).prefill_prompt
        count = token_counter(prompt)
        if type(count) is not int or count < 0:
            raise ValueError(
                f"token_counter must return a non-negative integer for "
                f"{example.dataset}:{example.example_id}"
            )
        if count != input_tokens_target:
            raise ValueError(
                f"logical prompt {example.dataset}:{example.example_id} has {count} "
                f"tokens; expected exactly {input_tokens_target}"
            )
        rows.append(
            {
                "dataset": example.dataset,
                "example_id": example.example_id,
                "logical_prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                "logical_tokens": count,
            }
        )
    return rows


def _write_and_reload_canary_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = tuple(dict(record) for record in records)
    with path.open("w", encoding="utf-8") as handle:
        for record in expected:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    reloaded = _read_jsonl_records(path)
    if reloaded != expected:
        raise RuntimeError(f"canary JSONL round trip changed request metadata: {path}")
    # Exercise the public loader as a final schema check. This catches projection
    # mistakes before the files are copied to a remote cluster.
    load_benchmark_jsonl(path)


def _result_record(value: Mapping[str, Any] | str | Path, *, arm_id: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        record = dict(value)
    else:
        path = local_path(str(value))
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"result {arm_id!r} is invalid JSON: {exc.msg}") from exc
        if not isinstance(loaded, Mapping):
            raise ValueError(f"result {arm_id!r} must be a JSON object")
        record = dict(loaded)
    if record.get("record_type") != BENCHMARK_RUN_RECORD_TYPE:
        raise ValueError(f"result {arm_id!r} is not a benchmark run record")
    suite = _required_mapping(record.get("suite"), f"result {arm_id}.suite")
    suite_arms = _mapping_sequence(suite.get("arms"), f"result {arm_id}.suite.arms")
    if len(suite_arms) != 1 or suite_arms[0].get("arm_id") != arm_id:
        raise ValueError(f"result {arm_id!r} must contain exactly its isolated arm")
    _only_arm_manifest(record, arm_id=arm_id)
    return record


def _only_arm_manifest(record: Mapping[str, Any], *, arm_id: str) -> Mapping[str, Any]:
    manifest = _required_mapping(record.get("experiment_manifest"), "experiment_manifest")
    arms = _mapping_sequence(manifest.get("arms"), "experiment_manifest.arms")
    if len(arms) != 1 or arms[0].get("arm_id") != arm_id:
        raise ValueError(f"result {arm_id!r} manifest must contain exactly its isolated arm")
    arm = arms[0]
    expected_method = _method_for_arm(arm_id)
    if bool(arm.get("uses_cache")) != bool(expected_method):
        raise ValueError(f"result {arm_id!r} manifest uses_cache is incorrect")
    if (arm.get("method_id") or "") != expected_method:
        raise ValueError(
            f"result {arm_id!r} manifest method_id must be {expected_method!r}"
        )
    return arm


def _validate_shared_result_identity(records: Mapping[str, Mapping[str, Any]]) -> None:
    baseline = records[BASELINE_PREFILL_ARM]
    baseline_suite = dict(_required_mapping(baseline["suite"], "suite"))
    baseline_suite.pop("arms", None)
    baseline_manifest = _required_mapping(baseline["experiment_manifest"], "experiment_manifest")
    for arm_id in REPRESENTATIVE_CANARY_ARM_IDS[1:]:
        candidate_suite = dict(_required_mapping(records[arm_id]["suite"], "suite"))
        candidate_suite.pop("arms", None)
        if candidate_suite != baseline_suite:
            raise ValueError(f"result {arm_id!r} suite identity differs from baseline")
        candidate = _required_mapping(records[arm_id]["experiment_manifest"], "experiment_manifest")
        for section in ("logical_workload", "decoding", "model_runtime", "execution"):
            if candidate.get(section) != baseline_manifest.get(section):
                raise ValueError(
                    f"result {arm_id!r} manifest {section} differs from baseline"
                )
        if candidate.get("experiment_id") != baseline_manifest.get("experiment_id"):
            raise ValueError(f"result {arm_id!r} experiment_id differs from baseline")
        baseline_environment = dict(
            _required_mapping(baseline_manifest.get("environment"), "environment")
        )
        candidate_environment = dict(
            _required_mapping(candidate.get("environment"), "environment")
        )
        # Different ephemeral cluster IDs are expected. Hardware/runtime versions,
        # storage, and declared cache state must still match exactly.
        baseline_environment.pop("runtime_id", None)
        candidate_environment.pop("runtime_id", None)
        if candidate_environment != baseline_environment:
            raise ValueError(f"result {arm_id!r} environment differs from baseline")


def _validate_measurement_pairing(measurements: Sequence[Mapping[str, Any]]) -> None:
    keys_by_arm: dict[str, set[tuple[str, str, int]]] = {}
    logical_digests_by_arm: dict[str, dict[tuple[str, str, int], str]] = {}
    for arm_id in REPRESENTATIVE_CANARY_ARM_IDS:
        keys: set[tuple[str, str, int]] = set()
        logical_digests: dict[tuple[str, str, int], str] = {}
        for record in measurements:
            if record.get("arm_id") != arm_id:
                continue
            repeat_index = record.get("repeat_index")
            if type(repeat_index) is not int or repeat_index <= 0:
                raise ValueError(f"measurement for {arm_id!r} has invalid repeat_index")
            key = (
                _required_string(record.get("dataset"), "measurement.dataset"),
                _required_string(record.get("example_id"), "measurement.example_id"),
                repeat_index,
            )
            if key in keys:
                raise ValueError(f"result {arm_id!r} has duplicate measurement key {key!r}")
            keys.add(key)
            metadata = _required_mapping(
                record.get("metadata", {}),
                "measurement.metadata",
            )
            digest = metadata.get("logical_prompt_sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"result {arm_id!r} measurement {key!r} is missing a logical prompt digest"
                )
            logical_digests[key] = digest
        if not keys:
            raise ValueError(f"result {arm_id!r} contains no measurements")
        keys_by_arm[arm_id] = keys
        logical_digests_by_arm[arm_id] = logical_digests
    baseline_keys = keys_by_arm[BASELINE_PREFILL_ARM]
    for arm_id in REPRESENTATIVE_CANARY_ARM_IDS[1:]:
        if keys_by_arm[arm_id] != baseline_keys:
            raise ValueError(
                f"result {arm_id!r} does not contain the same sample/repeat identities"
            )
        if logical_digests_by_arm[arm_id] != logical_digests_by_arm[BASELINE_PREFILL_ARM]:
            raise ValueError(
                f"result {arm_id!r} does not contain the same logical prompt identities"
            )


def _isolated_execution_window(
    result: Mapping[str, Any],
    *,
    arm_id: str,
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    windows = _mapping_sequence(
        result.get("execution_windows"),
        f"result {arm_id}.execution_windows",
    )
    if len(windows) != 1 or windows[0].get("arm_id") != arm_id:
        raise ValueError(
            f"result {arm_id!r} must contain exactly one arm execution window"
        )
    window = windows[0]
    wall_seconds = window.get("wall_seconds")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(float(wall_seconds))
        or wall_seconds <= 0
    ):
        raise ValueError(f"result {arm_id!r} execution window must be positive")
    successful = [measurement for measurement in measurements if not measurement.get("error")]
    completion_tokens = sum(
        int(measurement.get("completion_tokens", 0)) for measurement in successful
    )
    if window.get("completion_tokens") != completion_tokens:
        raise ValueError(
            f"result {arm_id!r} execution window completion_tokens do not match measurements"
        )
    if window.get("successful_requests") != len(successful):
        raise ValueError(
            f"result {arm_id!r} execution window successful_requests do not match measurements"
        )
    return {
        "arm_id": arm_id,
        "wall_seconds": float(wall_seconds),
        "completion_tokens": completion_tokens,
        "successful_requests": len(successful),
        "aggregate_output_tokens_per_second": completion_tokens / float(wall_seconds),
    }


def _validate_measurement_method_identity(
    arm_id: str,
    measurements: Sequence[Mapping[str, Any]],
) -> None:
    expected = _method_for_arm(arm_id)
    for record in measurements:
        if (record.get("cache_method") or "") != expected:
            raise ValueError(
                f"result {arm_id!r} measurement cache_method must be {expected!r}"
            )
        artifact_id = record.get("artifact_id") or ""
        if bool(artifact_id) != bool(expected):
            raise ValueError(
                f"result {arm_id!r} measurement artifact identity is missing or unexpected"
            )


def _validate_distinct_result_artifacts(measurements: Sequence[Mapping[str, Any]]) -> None:
    artifacts: dict[str, set[str]] = {
        arm_id: {
            str(record["artifact_id"])
            for record in measurements
            if record.get("arm_id") == arm_id and record.get("artifact_id")
        }
        for arm_id in (FULL_PREFIX_CANARY_ARM, VANILLA_CANARY_ARM)
    }
    if not artifacts[FULL_PREFIX_CANARY_ARM] or not artifacts[VANILLA_CANARY_ARM]:
        raise ValueError("both cache canary arms must report artifacts")
    if artifacts[FULL_PREFIX_CANARY_ARM].intersection(artifacts[VANILLA_CANARY_ARM]):
        raise ValueError("full-prefix and vanilla canary results reused an artifact identity")


def _mapping_sequence(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be an array")
    records: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name}[{index}] must be an object")
        records.append(item)
    return tuple(records)


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _deep_freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {str(key): _deep_freeze_json_value(item) for key, item in value.items()}
    )


def _deep_freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _deep_freeze_json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return tuple(_deep_freeze_json_value(item) for item in value)
    return value


def _deep_thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _deep_thaw_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return [_deep_thaw_json_value(item) for item in value]
    return value
