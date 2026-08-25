"""Databricks runs/submit payload helpers for the Qwen3 vLLM smoke benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_kv_cache._hardware_targets import (
    HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES,
    SUPPORTED_V1_HARDWARE_TARGETS,
    databricks_node_type_for_hardware_target,
    validate_aws_single_node_gpu_type,
    validate_aws_single_node_gpu_type_for_hardware_target,
    validate_v1_hardware_target,
    validate_v1_vllm_kv_cache_dtype_for_hardware_target,
)
from document_kv_cache.artifact_identity import RuntimeIdentity
from document_kv_cache.benchmarks import CACHE_REUSE_ARM
from document_kv_cache.databricks_job import (
    DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE,
    DEFAULT_DATABRICKS_DATA_SECURITY_MODE,
    DEFAULT_DATABRICKS_SPARK_VERSION,
    DEFAULT_DATABRICKS_RUN_TIMEOUT_SECONDS,
    DEFAULT_DATABRICKS_TASK_MAX_RETRIES,
    DatabricksSingleNodeGPUClusterConfig,
    _spark_env_vars_from_cli,
    _validated_spark_env_vars,
    _validated_databricks_run_timeout_seconds,
    _validated_databricks_task_max_retries,
    build_single_node_gpu_cluster,
)
from document_kv_cache.vllm_smoke import (
    BENCHMARK_ARM_IDS,
    DEFAULT_LOCAL_ROOT,
    HF_MODEL_ID,
    PREPARED_PREFIX_CACHE_SALT_MODE,
    SERVER_HOST,
    SERVER_PORT,
    VLLM_REPRESENTATIVE_WORKLOAD_PROFILES,
    VLLM_VERSION,
    VLLMRepresentativeWorkloadProfile,
    _arm_spec_requires_cachet_handoff,
    _runtime_identity_from_json,
    parse_dataset_specs,
    vllm_representative_workload_profile,
)
from document_kv_cache.benchmark_runner import PREFIX_CACHE_SALT_MODES
from document_kv_cache.canary_orchestration import (
    REPRESENTATIVE_TASK_RUNTIME_ID_REFERENCE,
    REPRESENTATIVE_VLLM_PACKAGE_PINS,
    benchmark_json_mapping_to_record,
    representative_canary_matrix,
    representative_vllm_comparison_suite_id,
    representative_vllm_environment_provenance,
    require_pinned_revision,
    resolved_layout_rope_provenance,
    validated_representative_wheel_binding,
    validated_benchmark_arm_specs,
    validated_benchmark_manifest_provenance,
)
from document_kv_cache.model_profiles import (
    QWEN3_4B_ROPE_ROTARY_DIM,
    QWEN3_4B_ROPE_THETA,
    layout_for_model,
)


DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME = "document-kv-vllm-smoke"
DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY = "document_kv_vllm_smoke"
DEFAULT_DATABRICKS_VLLM_SMOKE_PURPOSE = "document-kv-vllm-smoke"
VLLM_SMOKE_RUNNER_SCRIPT = """from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import subprocess
import sys


def _cluster_file_path(uri: str) -> str:
    if uri.startswith("dbfs:/"):
        return "/dbfs/" + uri.removeprefix("dbfs:/").lstrip("/")
    return uri


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


def _install_package_wheel(argv: list[str]) -> list[str]:
    os.environ.pop("DOCUMENT_KV_PACKAGE_WHEEL_SHA256", None)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--package-wheel-uri")
    parser.add_argument("--package-wheel-sha256")
    args, remaining = parser.parse_known_args(argv)
    if args.package_wheel_sha256 and not args.package_wheel_uri:
        raise ValueError("--package-wheel-sha256 requires --package-wheel-uri")
    if args.package_wheel_uri:
        package_wheel_path = _cluster_file_path(args.package_wheel_uri)
        verified_digest = None
        if args.package_wheel_sha256:
            digest = hashlib.sha256()
            with open(package_wheel_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            verified_digest = digest.hexdigest()
            if not hmac.compare_digest(verified_digest, args.package_wheel_sha256):
                raise ValueError("Cachet package wheel SHA-256 does not match")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                package_wheel_path,
            ],
            env=_pip_subprocess_environment(),
        )
        os.environ["DOCUMENT_KV_PACKAGE_INSTALL_SPEC"] = package_wheel_path
        if verified_digest is not None:
            os.environ["DOCUMENT_KV_PACKAGE_WHEEL_SHA256"] = verified_digest
    return remaining


if __name__ == "__main__":
    remaining_args = _install_package_wheel(sys.argv[1:])
    from document_kv_cache.vllm_smoke import main

    exit_code = main(remaining_args)
    if exit_code:
        raise SystemExit(exit_code)
"""

__all__ = [
    "DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME",
    "DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY",
    "DEFAULT_DATABRICKS_VLLM_SMOKE_PURPOSE",
    "VLLM_SMOKE_RUNNER_SCRIPT",
    "DatabricksVLLMSmokeJobConfig",
    "build_databricks_vllm_smoke_run_submit_payload",
    "write_databricks_vllm_smoke_run_submit_json",
    "write_databricks_vllm_smoke_runner_script",
    "main",
]


@dataclass(frozen=True, slots=True)
class DatabricksVLLMSmokeJobConfig:
    benchmark_id: str
    output_dir: str
    runner_python_file: str
    run_name: str = DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME
    task_key: str = DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY
    run_timeout_seconds: int = DEFAULT_DATABRICKS_RUN_TIMEOUT_SECONDS
    task_max_retries: int = DEFAULT_DATABRICKS_TASK_MAX_RETRIES
    hardware_target: str | None = None
    node_type_id: str = DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE
    spark_version: str = DEFAULT_DATABRICKS_SPARK_VERSION
    data_security_mode: str = DEFAULT_DATABRICKS_DATA_SECURITY_MODE
    single_user_name: str | None = None
    wheel_uri: str | None = None
    wheel_sha256: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    model_dtype: str = "bfloat16"
    model_quantization: str | None = None
    kv_cache_dtype: str | None = None
    attention_backend: str | None = None
    max_tokens: int = 32
    timeout_seconds: float = 240.0
    import_probe_timeout_seconds: float = 180.0
    server_start_timeout_seconds: float = 480.0
    local_root: str = str(DEFAULT_LOCAL_ROOT)
    server_host: str = SERVER_HOST
    server_port: int = SERVER_PORT
    client_host: str = SERVER_HOST
    max_model_len: int = 4096
    max_num_seqs: int = 2
    gpu_memory_utilization: float = 0.85
    benchmark_repeats: int = 1
    request_parallelism: int = 1
    runtime_telemetry_interval_seconds: float = 1.0
    benchmark_arms: tuple[str, ...] = ()
    benchmark_arm_specs: tuple[Mapping[str, Any], ...] = ()
    benchmark_evidence_policy: str | None = None
    representative_canary: bool = False
    representative_workload_profile: VLLMRepresentativeWorkloadProfile | str | None = None
    benchmark_manifest_provenance: Mapping[str, Any] = field(default_factory=dict)
    benchmark_prewarm_cache_prefix: bool = False
    benchmark_cache_runtime_prompt: bool = False
    benchmark_force_max_tokens: bool = False
    benchmark_prefix_cache_salt_mode: str = PREPARED_PREFIX_CACHE_SALT_MODE
    payload_cache_max_bytes: int = 0
    dataset_specs: tuple[str, ...] = ()
    allow_dataset_subset: bool = False
    benchmark_handoff_generator_factory: str | None = None
    benchmark_handoff_output_dir: str | None = None
    benchmark_handoff_dtype: str = "bfloat16"
    benchmark_handoff_align_bytes: int = 4096
    benchmark_handoff_generation_timeout_seconds: float = 1800.0
    benchmark_handoff_limit: int | None = None
    benchmark_handoff_segment_per_document: bool = False
    benchmark_handoff_cache_method: str | None = None
    benchmark_handoff_require_artifact_contract: bool = True
    runtime_identity: RuntimeIdentity | None = None
    availability: str = "ON_DEMAND"
    zone_id: str = "auto"
    custom_tags: Mapping[str, str] = field(default_factory=dict)
    spark_env_vars: Mapping[str, str] = field(default_factory=dict)
    benchmark_suite_id: str | None = None
    benchmark_runtime_id: str | None = None
    benchmark_prewarm_payload_cache: bool = False

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty")
        if not self.runner_python_file:
            raise ValueError("runner_python_file must be non-empty")
        for field_name in ("benchmark_suite_id", "benchmark_runtime_id"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must be non-empty when provided")
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if not self.task_key:
            raise ValueError("task_key must be non-empty")
        _validated_databricks_run_timeout_seconds(self.run_timeout_seconds)
        if self.run_timeout_seconds > DEFAULT_DATABRICKS_RUN_TIMEOUT_SECONDS:
            raise ValueError(
                "run_timeout_seconds exceeds the four-hour vLLM smoke bound"
            )
        _validated_databricks_task_max_retries(self.task_max_retries)
        resolved_hardware_target = _resolve_hardware_target(
            self.hardware_target,
            self.node_type_id,
        )
        object.__setattr__(self, "hardware_target", resolved_hardware_target)
        if self.wheel_uri is not None and not self.wheel_uri:
            raise ValueError("wheel_uri must be non-empty when provided")
        if self.wheel_sha256 is not None and (
            len(self.wheel_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.wheel_sha256)
        ):
            raise ValueError("wheel_sha256 must be a lowercase SHA-256 digest")
        if self.wheel_sha256 is not None and self.wheel_uri is None:
            raise ValueError("wheel_sha256 requires wheel_uri")
        if self.model_id is not None and not self.model_id.strip():
            raise ValueError("model_id must be non-empty when provided")
        for field_name in ("model_revision", "tokenizer_revision"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty when provided")
        if not self.model_dtype.strip():
            raise ValueError("model_dtype must be non-empty")
        if self.model_quantization is not None and not self.model_quantization.strip():
            raise ValueError("model_quantization must be non-empty when provided")
        if self.kv_cache_dtype is not None and not self.kv_cache_dtype.strip():
            raise ValueError("kv_cache_dtype must be non-empty when provided")
        validate_v1_vllm_kv_cache_dtype_for_hardware_target(
            hardware_target=resolved_hardware_target,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        if self.attention_backend is not None and not self.attention_backend.strip():
            raise ValueError("attention_backend must be non-empty when provided")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.import_probe_timeout_seconds <= 0:
            raise ValueError("import_probe_timeout_seconds must be positive")
        if self.server_start_timeout_seconds <= 0:
            raise ValueError("server_start_timeout_seconds must be positive")
        if not self.local_root:
            raise ValueError("local_root must be non-empty")
        if not self.server_host:
            raise ValueError("server_host must be non-empty")
        if not 0 < self.server_port < 65536:
            raise ValueError("server_port must be between 1 and 65535")
        if not self.client_host:
            raise ValueError("client_host must be non-empty")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if isinstance(self.benchmark_repeats, bool) or not isinstance(self.benchmark_repeats, int):
            raise TypeError("benchmark_repeats must be a positive integer")
        if self.benchmark_repeats <= 0:
            raise ValueError("benchmark_repeats must be a positive integer")
        if isinstance(self.request_parallelism, bool) or not isinstance(self.request_parallelism, int):
            raise TypeError("request_parallelism must be a positive integer")
        if self.request_parallelism <= 0:
            raise ValueError("request_parallelism must be a positive integer")
        if self.runtime_telemetry_interval_seconds <= 0:
            raise ValueError("runtime_telemetry_interval_seconds must be positive")
        object.__setattr__(self, "benchmark_arms", _validated_benchmark_arms(self.benchmark_arms))
        object.__setattr__(
            self,
            "benchmark_arm_specs",
            validated_benchmark_arm_specs(self.benchmark_arm_specs),
        )
        if self.benchmark_arms and self.benchmark_arm_specs:
            raise ValueError("benchmark_arms and benchmark_arm_specs are mutually exclusive")
        if self.benchmark_evidence_policy not in {None, "smoke", "canary", "publication"}:
            raise ValueError(
                "benchmark_evidence_policy must be smoke, canary, publication, or None"
            )
        if type(self.representative_canary) is not bool:
            raise TypeError("representative_canary must be a boolean")
        representative_profile = (
            None
            if self.representative_workload_profile is None
            else vllm_representative_workload_profile(
                self.representative_workload_profile
            )
        )
        if self.representative_canary != (representative_profile is not None):
            raise ValueError(
                "representative_canary and representative_workload_profile must "
                "be provided together"
            )
        object.__setattr__(
            self,
            "representative_workload_profile",
            representative_profile,
        )
        if representative_profile is not None:
            expected_suite_id = representative_vllm_comparison_suite_id(
                hardware_target=resolved_hardware_target,
                profile_id=representative_profile.profile_id,
            )
            if self.benchmark_suite_id is None:
                object.__setattr__(
                    self,
                    "benchmark_suite_id",
                    expected_suite_id,
                )
            elif self.benchmark_suite_id != expected_suite_id:
                raise ValueError(
                    "representative benchmark_suite_id must match the "
                    "hardware/profile comparison group"
                )
            if self.benchmark_runtime_id is None:
                object.__setattr__(
                    self,
                    "benchmark_runtime_id",
                    REPRESENTATIVE_TASK_RUNTIME_ID_REFERENCE,
                )
            elif (
                self.benchmark_runtime_id
                != REPRESENTATIVE_TASK_RUNTIME_ID_REFERENCE
            ):
                raise ValueError(
                    "representative benchmark_runtime_id must use the "
                    "retry-unique Databricks task run reference"
                )
        if (
            self.representative_canary
            and self.run_timeout_seconds != DEFAULT_DATABRICKS_RUN_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "representative_canary requires run and task timeout_seconds "
                f"to be exactly {DEFAULT_DATABRICKS_RUN_TIMEOUT_SECONDS}"
            )
        if self.is_representative_submission and (
            len(self.benchmark_arm_specs) != 1
            or not _is_fixed_representative_canary_arm(self.benchmark_arm_specs[0])
        ):
            raise ValueError(
                "representative_canary requires exactly one fixed matrix arm per task"
            )
        provenance = validated_benchmark_manifest_provenance(
            self.benchmark_manifest_provenance
        )
        if self.requires_pinned_revisions:
            if (
                self.benchmark_handoff_generator_factory is not None
                and not self.benchmark_handoff_require_artifact_contract
            ):
                raise ValueError(
                    "canary and publication handoff generation require the complete "
                    "registered method artifact contract"
                )
            if self.model_revision is None and "model_revision" in provenance:
                object.__setattr__(
                    self,
                    "model_revision",
                    require_pinned_revision(
                        provenance["model_revision"],
                        "benchmark_manifest_provenance.model_revision",
                    ),
                )
            if self.tokenizer_revision is None and "tokenizer_revision" in provenance:
                object.__setattr__(
                    self,
                    "tokenizer_revision",
                    require_pinned_revision(
                        provenance["tokenizer_revision"],
                        "benchmark_manifest_provenance.tokenizer_revision",
                    ),
                )
            require_pinned_revision(self.model_revision, "model_revision")
            require_pinned_revision(self.tokenizer_revision, "tokenizer_revision")
        if self.is_representative_submission:
            provenance = _representative_vllm_provenance(self, provenance)
        if (
            self.benchmark_runtime_id is not None
            and "runtime_id" in provenance
        ):
            raise ValueError(
                "benchmark_runtime_id and "
                "benchmark_manifest_provenance.runtime_id are mutually exclusive"
            )
        if (
            self.model_revision is not None
            and "model_revision" in provenance
            and provenance["model_revision"] != self.model_revision
        ):
            raise ValueError(
                "benchmark_manifest_provenance.model_revision must match model_revision"
            )
        if (
            self.tokenizer_revision is not None
            and "tokenizer_revision" in provenance
            and provenance["tokenizer_revision"] != self.tokenizer_revision
        ):
            raise ValueError(
                "benchmark_manifest_provenance.tokenizer_revision must match tokenizer_revision"
            )
        object.__setattr__(self, "benchmark_manifest_provenance", provenance)
        if (
            "input_tokens_target" in provenance
            and provenance.get("tokenizer_revision", self.tokenizer_revision) is None
        ):
            raise ValueError(
                "benchmark input_tokens_target requires a pinned tokenizer_revision"
            )
        if isinstance(self.payload_cache_max_bytes, bool) or not isinstance(self.payload_cache_max_bytes, int):
            raise TypeError("payload_cache_max_bytes must be a non-negative integer")
        if self.payload_cache_max_bytes < 0:
            raise ValueError("payload_cache_max_bytes must be a non-negative integer")
        if type(self.allow_dataset_subset) is not bool:
            raise TypeError("allow_dataset_subset must be a boolean")
        object.__setattr__(self, "dataset_specs", tuple(self.dataset_specs))
        if self.dataset_specs:
            parse_dataset_specs(self.dataset_specs, allow_subset=self.allow_dataset_subset)
        if type(self.benchmark_prewarm_cache_prefix) is not bool:
            raise TypeError("benchmark_prewarm_cache_prefix must be a boolean")
        if type(self.benchmark_prewarm_payload_cache) is not bool:
            raise TypeError("benchmark_prewarm_payload_cache must be a boolean")
        if type(self.benchmark_cache_runtime_prompt) is not bool:
            raise TypeError("benchmark_cache_runtime_prompt must be a boolean")
        if type(self.benchmark_force_max_tokens) is not bool:
            raise TypeError("benchmark_force_max_tokens must be a boolean")
        if self.benchmark_handoff_generator_factory is not None:
            if not self.benchmark_handoff_generator_factory.strip():
                raise ValueError("benchmark_handoff_generator_factory must be non-empty when provided")
            if not self.dataset_specs:
                raise ValueError("benchmark_handoff_generator_factory requires prepared dataset specs")
        if self.benchmark_prewarm_cache_prefix and not self.dataset_specs:
            raise ValueError("benchmark_prewarm_cache_prefix requires prepared dataset specs")
        if self.benchmark_prewarm_payload_cache and not self.dataset_specs:
            raise ValueError("benchmark_prewarm_payload_cache requires prepared dataset specs")
        if self.benchmark_prewarm_payload_cache and self.payload_cache_max_bytes <= 0:
            raise ValueError(
                "benchmark_prewarm_payload_cache requires a positive payload_cache_max_bytes"
            )
        if self.benchmark_prewarm_payload_cache and self.benchmark_prewarm_cache_prefix:
            raise ValueError(
                "benchmark_prewarm_payload_cache and benchmark_prewarm_cache_prefix "
                "are mutually exclusive"
            )
        if self.benchmark_prewarm_payload_cache:
            runs_cache_arm = (
                any(
                    _arm_spec_requires_cachet_handoff(spec)
                    for spec in self.benchmark_arm_specs
                )
                if self.benchmark_arm_specs
                else not self.benchmark_arms
                or CACHE_REUSE_ARM in self.benchmark_arms
            )
            if not runs_cache_arm:
                raise ValueError(
                    "benchmark_prewarm_payload_cache requires a Cachet handoff benchmark arm"
                )
        if self.benchmark_cache_runtime_prompt and not self.dataset_specs:
            raise ValueError("benchmark_cache_runtime_prompt requires prepared dataset specs")
        if self.benchmark_prefix_cache_salt_mode not in PREFIX_CACHE_SALT_MODES:
            raise ValueError("benchmark_prefix_cache_salt_mode must be 'static' or 'per_request'")
        if self.benchmark_prewarm_cache_prefix and self.benchmark_prefix_cache_salt_mode != "static":
            raise ValueError(
                "benchmark_prewarm_cache_prefix requires benchmark_prefix_cache_salt_mode='static' "
                "so prewarmed prefix-cache blocks can be reused"
            )
        if (
            self.benchmark_prewarm_payload_cache
            and self.benchmark_prefix_cache_salt_mode != "per_request"
        ):
            raise ValueError(
                "benchmark_prewarm_payload_cache requires "
                "benchmark_prefix_cache_salt_mode='per_request'"
            )
        if self.benchmark_handoff_output_dir is not None and not self.benchmark_handoff_output_dir:
            raise ValueError("benchmark_handoff_output_dir must be non-empty when provided")
        if self.benchmark_handoff_output_dir is not None and self.benchmark_handoff_generator_factory is None:
            raise ValueError("benchmark_handoff_output_dir requires benchmark_handoff_generator_factory")
        if not self.benchmark_handoff_dtype:
            raise ValueError("benchmark_handoff_dtype must be non-empty")
        if type(self.benchmark_handoff_align_bytes) is not int or self.benchmark_handoff_align_bytes <= 0:
            raise ValueError("benchmark_handoff_align_bytes must be a positive integer")
        if self.benchmark_handoff_generation_timeout_seconds <= 0:
            raise ValueError("benchmark_handoff_generation_timeout_seconds must be positive")
        if self.benchmark_handoff_limit is not None:
            if (
                isinstance(self.benchmark_handoff_limit, bool)
                or not isinstance(self.benchmark_handoff_limit, int)
                or self.benchmark_handoff_limit < 0
            ):
                raise ValueError("benchmark_handoff_limit must be a non-negative integer")
        if type(self.benchmark_handoff_segment_per_document) is not bool:
            raise TypeError(
                "benchmark_handoff_segment_per_document must be a boolean"
            )
        if self.benchmark_handoff_cache_method is not None:
            if not self.benchmark_handoff_cache_method.strip():
                raise ValueError(
                    "benchmark_handoff_cache_method must be non-empty when provided"
                )
            if self.benchmark_handoff_generator_factory is None:
                raise ValueError(
                    "benchmark_handoff_cache_method requires "
                    "benchmark_handoff_generator_factory"
                )
        if (
            self.benchmark_handoff_cache_method == "vanilla_prefill"
            and not self.benchmark_handoff_segment_per_document
        ):
            raise ValueError(
                "vanilla_prefill handoff generation requires one segment per document"
            )
        if (
            self.benchmark_handoff_cache_method == "full_prefix_prefill"
            and self.benchmark_handoff_segment_per_document
        ):
            raise ValueError(
                "full_prefix_prefill handoff generation requires one full-prefix segment"
            )
        if type(self.benchmark_handoff_require_artifact_contract) is not bool:
            raise TypeError(
                "benchmark_handoff_require_artifact_contract must be a boolean"
            )
        if (
            self.benchmark_handoff_segment_per_document
        ) and self.benchmark_handoff_generator_factory is None:
            raise ValueError(
                "benchmark handoff options require "
                "benchmark_handoff_generator_factory"
            )
        if (
            self.benchmark_handoff_generator_factory is not None
            and not self.benchmark_handoff_require_artifact_contract
            and self.benchmark_evidence_policy in {"canary", "publication"}
        ):
            raise ValueError(
                "canary and publication handoff generation require the complete "
                "registered method artifact contract"
            )
        if self.is_representative_submission:
            validated_representative_wheel_binding(
                self.wheel_uri,
                self.wheel_sha256,
            )
            _validate_representative_vllm_workload(self)
        if self.runtime_identity is not None:
            if not isinstance(self.runtime_identity, RuntimeIdentity):
                raise TypeError("runtime_identity must be a RuntimeIdentity or None")
            if self.model_revision is None or self.tokenizer_revision is None:
                raise ValueError(
                    "runtime_identity requires pinned model_revision and "
                    "tokenizer_revision"
                )
            if self.model_id is None:
                raise ValueError("runtime_identity requires model_id")
            expected_runtime_identity = {
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "tokenizer_id": self.model_id,
                "tokenizer_revision": self.tokenizer_revision,
                "kv_dtype": self.kv_cache_dtype or self.model_dtype,
            }
            mismatches = [
                field_name
                for field_name, expected in expected_runtime_identity.items()
                if getattr(self.runtime_identity, field_name) != expected
            ]
            if mismatches:
                raise ValueError(
                    "runtime_identity does not match pinned vLLM configuration: "
                    + ", ".join(mismatches)
                )
        spark_env_vars = dict(_validated_spark_env_vars(self.spark_env_vars))
        if self.is_representative_submission:
            _validate_representative_node_type_id(
                self.node_type_id,
                self.hardware_target,
            )
            if Path(self.local_root) != DEFAULT_LOCAL_ROOT:
                raise ValueError("representative canary local_root must be /local_disk0")
            if self.benchmark_handoff_output_dir is not None:
                _require_local_disk0_path(
                    self.benchmark_handoff_output_dir,
                    "benchmark_handoff_output_dir",
                )
            existing_evict = spark_env_vars.get("DOCUMENT_KV_EVICT_PAGE_CACHE")
            if existing_evict not in {None, "1"}:
                raise ValueError(
                    "DOCUMENT_KV_EVICT_PAGE_CACHE must be 1 for representative canaries"
                )
            spark_env_vars["DOCUMENT_KV_EVICT_PAGE_CACHE"] = "1"
        object.__setattr__(self, "spark_env_vars", spark_env_vars)
        _DEFAULT_CLUSTER_CONFIG_FROM_VLLM_SMOKE_JOB(self)

    @property
    def is_representative_submission(self) -> bool:
        return self.representative_canary

    @property
    def requires_pinned_revisions(self) -> bool:
        return self.is_representative_submission or self.benchmark_evidence_policy in {
            "canary",
            "publication",
        }


def _is_fixed_representative_canary_arm(value: Mapping[str, Any]) -> bool:
    record = benchmark_json_mapping_to_record(value)
    return any(
        record == benchmark_json_mapping_to_record(run.arm_spec)
        for run in representative_canary_matrix().runs
    )


def _validate_representative_vllm_workload(
    config: DatabricksVLLMSmokeJobConfig,
) -> None:
    profile = config.representative_workload_profile
    if not isinstance(profile, VLLMRepresentativeWorkloadProfile):
        raise ValueError(
            "representative vLLM submission requires a typed workload profile"
        )
    mismatches: list[str] = []
    if (
        config.benchmark_manifest_provenance.get("input_tokens_target")
        != profile.input_tokens_target
    ):
        mismatches.append("input_tokens_target")
    if config.max_tokens != profile.max_output_tokens:
        mismatches.append("max_tokens")
    if config.max_model_len != profile.max_model_len:
        mismatches.append("max_model_len")
    if config.max_num_seqs != profile.max_num_seqs:
        mismatches.append("max_num_seqs")
    if config.gpu_memory_utilization != profile.gpu_memory_utilization:
        mismatches.append("gpu_memory_utilization")
    if config.model_dtype != profile.model_dtype:
        mismatches.append("model_dtype")
    if (config.kv_cache_dtype or config.model_dtype) != profile.runtime_kv_dtype:
        mismatches.append("kv_cache_dtype")
    if config.benchmark_repeats != profile.benchmark_repeats:
        mismatches.append("benchmark_repeats")
    if config.request_parallelism != profile.request_parallelism:
        mismatches.append("request_parallelism")
    if config.benchmark_force_max_tokens != profile.force_max_tokens:
        mismatches.append("benchmark_force_max_tokens")
    if (
        config.benchmark_prefix_cache_salt_mode
        != profile.prefix_cache_salt_mode
    ):
        mismatches.append("benchmark_prefix_cache_salt_mode")
    if config.benchmark_prewarm_cache_prefix != profile.prewarm_cache_prefix:
        mismatches.append("benchmark_prewarm_cache_prefix")
    if config.benchmark_cache_runtime_prompt != profile.cache_runtime_prompt:
        mismatches.append("benchmark_cache_runtime_prompt")
    if config.payload_cache_max_bytes != profile.payload_cache_max_bytes:
        mismatches.append("payload_cache_max_bytes")
    if config.benchmark_evidence_policy != profile.benchmark_evidence_policy:
        mismatches.append("benchmark_evidence_policy")
    if not config.dataset_specs:
        mismatches.append("dataset_specs")
    else:
        dataset_names = {
            spec.split("=", 1)[0]
            for spec in config.dataset_specs
            if isinstance(spec, str) and "=" in spec
        }
        if dataset_names.isdisjoint(profile.multi_document_datasets):
            mismatches.append("multi_document_dataset")
    if mismatches:
        raise ValueError(
            f"representative workload profile {profile.profile_id!r} does not "
            "match config: "
            + ", ".join(mismatches)
        )


def _representative_vllm_provenance(
    config: DatabricksVLLMSmokeJobConfig,
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    resolved_model_id = config.model_id or HF_MODEL_ID
    if resolved_model_id != HF_MODEL_ID:
        raise ValueError(
            f"representative canary model_id must be the canonical {HF_MODEL_ID!r}"
        )
    runtime_kv_dtype = config.kv_cache_dtype or config.model_dtype
    pre_rope = config.benchmark_handoff_cache_method == "vanilla_prefill"
    layout = layout_for_model(
        HF_MODEL_ID,
        dtype=runtime_kv_dtype,
        **(
            {
                "pre_rope": True,
                "rope_theta": QWEN3_4B_ROPE_THETA,
                "rope_rotary_dim": QWEN3_4B_ROPE_ROTARY_DIM,
                "shares_kv_storage": False,
                "storage_layout": "separate_key_value",
            }
            if pre_rope
            else {}
        ),
    )
    _, wheel_sha256 = validated_representative_wheel_binding(
        config.wheel_uri,
        config.wheel_sha256,
    )
    package_revisions = {
        package: version
        for package, version in (
            pin.split("==", 1) for pin in REPRESENTATIVE_VLLM_PACKAGE_PINS
        )
    }
    package_revisions["cachet-kv"] = f"wheel-sha256:{wheel_sha256}"
    expected: dict[str, Any] = {
        "canonical_model_id": HF_MODEL_ID,
        "model_revision": config.model_revision,
        "tokenizer_id": HF_MODEL_ID,
        "tokenizer_revision": config.tokenizer_revision,
        "lora_id": layout.lora_id,
        "engine_id": "vllm",
        "engine_version": VLLM_VERSION,
        "serving_platform": "vllm",
        "model_dtype": config.model_dtype,
        "model_quantization": config.model_quantization or "none",
        "runtime_kv_dtype": runtime_kv_dtype,
        "layout_version": layout.layout_version,
        "payload_axis_order": getattr(
            layout.payload_axis_order,
            "value",
            layout.payload_axis_order,
        ),
        "block_size": layout.block_size,
        "key_position_encoding": getattr(
            layout.key_position_encoding,
            "value",
            layout.key_position_encoding,
        ),
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "package_revisions": package_revisions,
    }
    expected.update(
        representative_vllm_environment_provenance(
            str(config.hardware_target)
        )
    )
    resolved_rope = resolved_layout_rope_provenance(layout)
    expected.update(resolved_rope)
    record = dict(provenance)
    conflicts = {
        field_name
        for field_name, expected_value in expected.items()
        if field_name in record and record[field_name] != expected_value
    }
    if not resolved_rope:
        conflicts.update(
            field_name
            for field_name in ("rope_theta", "rope_rotary_dim")
            if field_name in record
        )
    if conflicts:
        raise ValueError(
            "representative benchmark provenance conflicts with the resolved runtime: "
            + ", ".join(sorted(conflicts))
        )
    record.update(expected)
    return validated_benchmark_manifest_provenance(record)


def _require_local_disk0_path(value: str, field_name: str) -> None:
    path = Path(value).resolve(strict=False)
    try:
        path.relative_to(DEFAULT_LOCAL_ROOT.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"representative canary {field_name} must be under /local_disk0") from exc


def build_databricks_vllm_smoke_run_submit_payload(config: DatabricksVLLMSmokeJobConfig) -> dict[str, Any]:
    cluster = build_single_node_gpu_cluster(_cluster_config_from_vllm_smoke_job(config))
    if config.spark_env_vars:
        cluster["spark_env_vars"] = dict(config.spark_env_vars)
    task: dict[str, Any] = {
        "task_key": config.task_key,
        "timeout_seconds": config.run_timeout_seconds,
        "max_retries": config.task_max_retries,
        "new_cluster": cluster,
        "spark_python_task": {
            "python_file": config.runner_python_file,
            "parameters": _runner_parameters(config),
        },
    }
    if config.wheel_uri is not None:
        task["spark_python_task"]["parameters"].extend(["--package-wheel-uri", config.wheel_uri])
        if config.wheel_sha256 is not None:
            task["spark_python_task"]["parameters"].extend(
                ["--package-wheel-sha256", config.wheel_sha256]
            )
    return {
        "run_name": config.run_name,
        "timeout_seconds": config.run_timeout_seconds,
        "tasks": [task],
    }


def write_databricks_vllm_smoke_run_submit_json(
    config: DatabricksVLLMSmokeJobConfig,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(build_databricks_vllm_smoke_run_submit_payload(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_databricks_vllm_smoke_runner_script(path: str | Path) -> None:
    Path(path).write_text(VLLM_SMOKE_RUNNER_SCRIPT, encoding="utf-8")


def _cluster_config_from_vllm_smoke_job(config: DatabricksVLLMSmokeJobConfig) -> DatabricksSingleNodeGPUClusterConfig:
    return DatabricksSingleNodeGPUClusterConfig(
        purpose=DEFAULT_DATABRICKS_VLLM_SMOKE_PURPOSE,
        node_type_id=config.node_type_id,
        spark_version=config.spark_version,
        data_security_mode=config.data_security_mode,
        single_user_name=config.single_user_name,
        availability=config.availability,
        zone_id=config.zone_id,
        custom_tags=config.custom_tags,
    )


_DEFAULT_CLUSTER_CONFIG_FROM_VLLM_SMOKE_JOB = _cluster_config_from_vllm_smoke_job


def _validate_representative_node_type_id(
    node_type_id: str,
    hardware_target: str | None,
) -> None:
    expected = databricks_node_type_for_hardware_target(hardware_target)
    if node_type_id != expected:
        raise ValueError(
            "representative canary node_type_id must be the exact V1 node type "
            f"{expected!r} for hardware target {hardware_target!r}, got "
            f"{node_type_id!r}"
        )


def _resolve_hardware_target(hardware_target: str | None, node_type_id: str) -> str:
    if hardware_target is not None:
        validate_v1_hardware_target(hardware_target)
        validate_aws_single_node_gpu_type_for_hardware_target(node_type_id, hardware_target)
        return hardware_target
    validate_aws_single_node_gpu_type(node_type_id)
    lowered = node_type_id.lower()
    for target, prefixes in HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES.items():
        if lowered.startswith(prefixes):
            return str(target)
    raise ValueError(f"Unable to derive V1 hardware target from node_type_id {node_type_id!r}")


def _validated_benchmark_arms(value: Sequence[str]) -> tuple[str, ...]:
    if not value:
        return ()
    arms: list[str] = []
    for index, arm_id in enumerate(value):
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError(f"benchmark_arms[{index}] must be a non-empty string")
        arms.append(arm_id)
    if len(set(arms)) != len(arms):
        raise ValueError(f"benchmark_arms must not contain duplicates: {arms}")
    unknown = sorted(set(arms).difference(BENCHMARK_ARM_IDS))
    if unknown:
        raise ValueError(f"Unknown benchmark arms: {unknown}")
    return tuple(arms)


def _runner_parameters(config: DatabricksVLLMSmokeJobConfig) -> list[str]:
    parameters = [
        "--benchmark-id",
        config.benchmark_id,
        "--output-dir",
        config.output_dir,
        "--max-tokens",
        str(config.max_tokens),
        "--timeout-seconds",
        str(config.timeout_seconds),
        "--import-probe-timeout-seconds",
        str(config.import_probe_timeout_seconds),
        "--server-start-timeout-seconds",
        str(config.server_start_timeout_seconds),
        "--local-root",
        config.local_root,
        "--server-host",
        config.server_host,
        "--server-port",
        str(config.server_port),
        "--client-host",
        config.client_host,
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--hardware-target",
        str(config.hardware_target),
        "--benchmark-repeats",
        str(config.benchmark_repeats),
        "--request-parallelism",
        str(config.request_parallelism),
        "--runtime-telemetry-interval-seconds",
        str(config.runtime_telemetry_interval_seconds),
    ]
    if config.benchmark_suite_id is not None:
        parameters.extend(
            ["--benchmark-suite-id", config.benchmark_suite_id]
        )
    if config.benchmark_runtime_id is not None:
        parameters.extend(["--runtime-id", config.benchmark_runtime_id])
    resolved_model_id = (
        config.model_id
        if config.model_id is not None
        else HF_MODEL_ID
        if config.is_representative_submission
        else None
    )
    if resolved_model_id:
        parameters.extend(["--model-id", resolved_model_id])
    if config.model_revision:
        parameters.extend(["--model-revision", config.model_revision])
    if config.tokenizer_revision:
        parameters.extend(
            ["--tokenizer-revision", config.tokenizer_revision]
        )
    if config.model_dtype != "bfloat16" or config.is_representative_submission:
        parameters.extend(["--model-dtype", config.model_dtype])
    if config.model_quantization:
        parameters.extend(["--model-quantization", config.model_quantization])
    resolved_kv_cache_dtype = config.kv_cache_dtype
    if resolved_kv_cache_dtype is None and config.is_representative_submission:
        resolved_kv_cache_dtype = config.model_dtype
    if resolved_kv_cache_dtype:
        parameters.extend(["--kv-cache-dtype", resolved_kv_cache_dtype])
    if config.attention_backend:
        parameters.extend(["--attention-backend", config.attention_backend])
    if config.payload_cache_max_bytes or config.is_representative_submission:
        parameters.extend(["--payload-cache-max-bytes", str(config.payload_cache_max_bytes)])
    if config.runtime_identity is not None:
        parameters.extend(
            [
                "--runtime-identity-json",
                json.dumps(
                    config.runtime_identity.to_record(),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            ]
        )
    for arm_id in config.benchmark_arms:
        parameters.extend(["--benchmark-arm", arm_id])
    for arm_spec in config.benchmark_arm_specs:
        parameters.extend(
            [
                "--benchmark-arm-spec-json",
                json.dumps(
                    benchmark_json_mapping_to_record(arm_spec),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            ]
        )
    if config.benchmark_evidence_policy is not None:
        parameters.extend(
            ["--benchmark-evidence-policy", config.benchmark_evidence_policy]
        )
    if config.is_representative_submission:
        parameters.append("--representative-canary")
        if not isinstance(
            config.representative_workload_profile,
            VLLMRepresentativeWorkloadProfile,
        ):
            raise TypeError(
                "representative submission must have a typed workload profile"
            )
        parameters.extend(
            [
                "--representative-workload-profile",
                config.representative_workload_profile.profile_id,
            ]
        )
    if config.benchmark_manifest_provenance:
        parameters.extend(
            [
                "--benchmark-manifest-provenance-json",
                json.dumps(
                    benchmark_json_mapping_to_record(
                        config.benchmark_manifest_provenance
                    ),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            ]
        )
    if config.benchmark_prewarm_cache_prefix:
        parameters.append("--benchmark-prewarm-cache-prefix")
    if config.benchmark_prewarm_payload_cache:
        parameters.append("--benchmark-prewarm-payload-cache")
    if config.benchmark_cache_runtime_prompt:
        parameters.append("--benchmark-cache-runtime-prompt")
    if config.benchmark_force_max_tokens:
        parameters.append("--benchmark-force-max-tokens")
    if (
        config.is_representative_submission
        or config.benchmark_prewarm_cache_prefix
        or config.benchmark_prewarm_payload_cache
        or config.benchmark_prefix_cache_salt_mode
        != PREPARED_PREFIX_CACHE_SALT_MODE
    ):
        parameters.extend(["--benchmark-prefix-cache-salt-mode", config.benchmark_prefix_cache_salt_mode])
    for dataset_spec in config.dataset_specs:
        parameters.extend(["--dataset", dataset_spec])
    if config.allow_dataset_subset:
        parameters.append("--allow-dataset-subset")
    if config.benchmark_handoff_generator_factory is not None:
        parameters.extend(
            [
                "--benchmark-handoff-generator-factory",
                config.benchmark_handoff_generator_factory,
                "--benchmark-handoff-dtype",
                config.benchmark_handoff_dtype,
                "--benchmark-handoff-align-bytes",
                str(config.benchmark_handoff_align_bytes),
                "--benchmark-handoff-generation-timeout-seconds",
                str(config.benchmark_handoff_generation_timeout_seconds),
            ]
        )
        if config.benchmark_handoff_limit is not None:
            parameters.extend(["--benchmark-handoff-limit", str(config.benchmark_handoff_limit)])
        if config.benchmark_handoff_output_dir is not None:
            parameters.extend(["--benchmark-handoff-output-dir", config.benchmark_handoff_output_dir])
        if config.benchmark_handoff_segment_per_document:
            parameters.append("--benchmark-handoff-chunk-per-document")
        if config.benchmark_handoff_cache_method is not None:
            parameters.extend(
                [
                    "--benchmark-handoff-cache-method",
                    config.benchmark_handoff_cache_method,
                ]
            )
        if not config.benchmark_handoff_require_artifact_contract:
            parameters.append(
                "--benchmark-handoff-allow-legacy-artifact-contract"
            )
    return parameters


def _json_object_from_cli(value: str, option_name: str) -> Mapping[str, Any]:
    try:
        record = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} must be valid JSON: {exc.msg}") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"{option_name} must contain a JSON object")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a Databricks runs/submit payload for a V1 AWS single-node GPU vLLM smoke."
    )
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument(
        "--benchmark-suite-id",
        help=(
            "Shared benchmark suite/experiment ID. Representative isolated arms "
            "derive the canonical hardware/profile group when omitted."
        ),
    )
    parser.add_argument(
        "--runtime-id",
        dest="benchmark_runtime_id",
        help=(
            "Physical benchmark execution ID. Representative jobs derive the "
            "retry-unique Databricks task run reference when omitted."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Cluster-visible output directory for smoke artifacts.")
    parser.add_argument("--runner-python-file", required=True, help="Cluster-visible runner script path or URI.")
    parser.add_argument("--run-name", default=DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME)
    parser.add_argument("--task-key", default=DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY)
    parser.add_argument(
        "--run-timeout-seconds",
        type=int,
        default=DEFAULT_DATABRICKS_RUN_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--task-max-retries",
        type=int,
        default=DEFAULT_DATABRICKS_TASK_MAX_RETRIES,
    )
    parser.add_argument(
        "--hardware-target",
        choices=SUPPORTED_V1_HARDWARE_TARGETS,
        help="V1 hardware target used to derive --node-type-id when it is omitted.",
    )
    parser.add_argument(
        "--node-type-id",
        help="Databricks node type override. Must match --hardware-target when provided.",
    )
    parser.add_argument("--spark-version", default=DEFAULT_DATABRICKS_SPARK_VERSION)
    parser.add_argument("--data-security-mode", default=DEFAULT_DATABRICKS_DATA_SECURITY_MODE)
    parser.add_argument("--single-user-name", help="Required when --data-security-mode SINGLE_USER.")
    parser.add_argument("--wheel-uri", help="Optional cluster-visible wheel URI to install before the task.")
    parser.add_argument("--wheel-sha256")
    parser.add_argument("--model-id", help="HF model path/id passed to vLLM --model.")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--model-dtype", default="bfloat16", help="Model dtype passed to vLLM --dtype.")
    parser.add_argument("--model-quantization", help="Optional vLLM --quantization value.")
    parser.add_argument("--kv-cache-dtype", help="Optional vLLM --kv-cache-dtype value.")
    parser.add_argument("--attention-backend", help="Optional vLLM --attention-backend value.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--import-probe-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--server-start-timeout-seconds", type=float, default=480.0)
    parser.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--server-host", default=SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=SERVER_PORT)
    parser.add_argument("--client-host", default=SERVER_HOST)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=1,
        help=(
            "Number of baseline/cache arm repeats per benchmark example. "
            "Use values greater than 1 for hot-document cache measurements."
        ),
    )
    parser.add_argument(
        "--request-parallelism",
        type=int,
        default=1,
        help="Maximum number of benchmark requests issued concurrently by the client.",
    )
    parser.add_argument(
        "--runtime-telemetry-interval-seconds",
        type=float,
        default=1.0,
        help="Runtime telemetry sampling interval for GPU, host memory, and process RSS artifacts.",
    )
    parser.add_argument(
        "--benchmark-arm",
        action="append",
        choices=BENCHMARK_ARM_IDS,
        default=None,
        help=(
            "Benchmark only this arm. Repeat for multiple arms; omit to run "
            "baseline_prefill and document_kv_cache."
        ),
    )
    parser.add_argument(
        "--benchmark-arm-spec-json",
        action="append",
        default=None,
        help=(
            "Validated arbitrary benchmark arm JSON. Repeat for N-way comparisons; "
            "mutually exclusive with --benchmark-arm."
        ),
    )
    parser.add_argument(
        "--benchmark-evidence-policy",
        choices=("smoke", "canary", "publication"),
    )
    parser.add_argument("--representative-canary", action="store_true")
    parser.add_argument(
        "--representative-workload-profile",
        choices=tuple(
            profile.profile_id
            for profile in VLLM_REPRESENTATIVE_WORKLOAD_PROFILES
        ),
        help=(
            "Registered exact representative workload profile. Must be supplied "
            "together with --representative-canary."
        ),
    )
    parser.add_argument(
        "--benchmark-manifest-provenance-json",
        help="Benchmark manifest provenance JSON forwarded unchanged to the smoke task.",
    )
    parser.add_argument(
        "--benchmark-cache-runtime-prompt",
        action="store_true",
        help="Send only runtime suffix prompts for benchmark cache arms.",
    )
    parser.add_argument(
        "--benchmark-prewarm-cache-prefix",
        action="store_true",
        help=(
            "Before measurement, issue one KV-aware cache-prefix request per prepared "
            "example so vLLM can keep shared document/system prefix blocks resident."
        ),
    )
    parser.add_argument(
        "--benchmark-prewarm-payload-cache",
        action="store_true",
        help=(
            "Prime every prepared Cachet payload into the provider's host-RAM cache "
            "under an isolated GPU prefix-cache namespace and require hit attestations."
        ),
    )
    parser.add_argument(
        "--benchmark-force-max-tokens",
        action="store_true",
        help="Force benchmark requests to emit exactly --max-tokens tokens with ignore_eos=true.",
    )
    parser.add_argument(
        "--benchmark-prefix-cache-salt-mode",
        choices=PREFIX_CACHE_SALT_MODES,
        default=PREPARED_PREFIX_CACHE_SALT_MODE,
        help=(
            "Prefix-cache salt mode for prepared benchmark requests. "
            "'per_request' isolates repeats; 'static' allows repeated documents to share vLLM blocks."
        ),
    )
    parser.add_argument(
        "--payload-cache-max-bytes",
        type=int,
        default=0,
        help=(
            "Optional byte budget for the vLLM provider's in-process payload URI cache. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--runtime-identity-json",
        help=(
            "RuntimeIdentity JSON used by the native provider compatibility "
            "handshake."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Prepared V1 benchmark dataset in DATASET=JSONL_PATH form. Repeat for all four V1 datasets.",
    )
    parser.add_argument(
        "--allow-dataset-subset",
        action="store_true",
        help=(
            "Allow prepared runs to specify only a subset of V1 datasets. "
            "Use for split full-dataset score jobs; omitted smoke runs still require all four datasets."
        ),
    )
    parser.add_argument(
        "--benchmark-handoff-generator-factory",
        help=(
            "Generate Cachet handoff bundles inside the vLLM task before serving. "
            "Value must be a module:callable returning a KVChunkGenerator."
        ),
    )
    parser.add_argument(
        "--benchmark-handoff-output-dir",
        help="Cluster-visible output directory for generated handoff bundles and enriched JSONL.",
    )
    parser.add_argument("--benchmark-handoff-dtype", default="bfloat16")
    parser.add_argument("--benchmark-handoff-align-bytes", type=int, default=4096)
    parser.add_argument(
        "--benchmark-handoff-generation-timeout-seconds",
        type=float,
        default=1800.0,
    )
    parser.add_argument(
        "--benchmark-handoff-limit",
        type=int,
        help=(
            "Optional per-dataset row limit for generated benchmark handoffs. "
            "Use only for canary/debug runs; omit for full benchmark evidence."
        ),
    )
    parser.add_argument(
        "--benchmark-handoff-chunk-per-document",
        action="store_true",
    )
    parser.add_argument("--benchmark-handoff-cache-method")
    parser.add_argument(
        "--benchmark-handoff-allow-legacy-artifact-contract",
        action="store_true",
        help=(
            "Legacy/debug opt-out for incomplete method artifacts; never use for "
            "canary or publication evidence."
        ),
    )
    parser.add_argument(
        "--spark-env-var",
        action="append",
        default=None,
        help=(
            "Non-secret Databricks cluster spark_env_vars entry for runtime configuration, "
            "in KEY=VALUE form. Repeat for values such as CACHET_TRANSFORMERS_DEVICE=cuda."
        ),
    )
    parser.add_argument("--output-json", help="Write the runs/submit payload to this path instead of stdout.")
    parser.add_argument("--runner-script-output", help="Write the tiny vLLM smoke runner script to this path.")
    args = parser.parse_args(argv)

    try:
        config = DatabricksVLLMSmokeJobConfig(
            benchmark_id=args.benchmark_id,
            output_dir=args.output_dir,
            runner_python_file=args.runner_python_file,
            benchmark_suite_id=args.benchmark_suite_id,
            benchmark_runtime_id=args.benchmark_runtime_id,
            run_name=args.run_name,
            task_key=args.task_key,
            run_timeout_seconds=args.run_timeout_seconds,
            task_max_retries=args.task_max_retries,
            node_type_id=databricks_node_type_for_hardware_target(args.hardware_target, args.node_type_id),
            hardware_target=args.hardware_target,
            spark_version=args.spark_version,
            data_security_mode=args.data_security_mode,
            single_user_name=args.single_user_name,
            wheel_uri=args.wheel_uri,
            wheel_sha256=args.wheel_sha256,
            model_id=args.model_id,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            model_dtype=args.model_dtype,
            model_quantization=args.model_quantization,
            kv_cache_dtype=args.kv_cache_dtype,
            attention_backend=args.attention_backend,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            import_probe_timeout_seconds=args.import_probe_timeout_seconds,
            server_start_timeout_seconds=args.server_start_timeout_seconds,
            local_root=args.local_root,
            server_host=args.server_host,
            server_port=args.server_port,
            client_host=args.client_host,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            benchmark_repeats=args.benchmark_repeats,
            request_parallelism=args.request_parallelism,
            runtime_telemetry_interval_seconds=args.runtime_telemetry_interval_seconds,
            benchmark_arms=tuple(args.benchmark_arm or ()),
            benchmark_arm_specs=tuple(
                _json_object_from_cli(value, "--benchmark-arm-spec-json")
                for value in (args.benchmark_arm_spec_json or ())
            ),
            benchmark_evidence_policy=args.benchmark_evidence_policy,
            representative_canary=args.representative_canary,
            representative_workload_profile=args.representative_workload_profile,
            benchmark_manifest_provenance=(
                {}
                if args.benchmark_manifest_provenance_json is None
                else _json_object_from_cli(
                    args.benchmark_manifest_provenance_json,
                    "--benchmark-manifest-provenance-json",
                )
            ),
            benchmark_prewarm_cache_prefix=args.benchmark_prewarm_cache_prefix,
            benchmark_prewarm_payload_cache=args.benchmark_prewarm_payload_cache,
            benchmark_cache_runtime_prompt=args.benchmark_cache_runtime_prompt,
            benchmark_force_max_tokens=args.benchmark_force_max_tokens,
            benchmark_prefix_cache_salt_mode=args.benchmark_prefix_cache_salt_mode,
            payload_cache_max_bytes=args.payload_cache_max_bytes,
            dataset_specs=tuple(args.dataset or ()),
            allow_dataset_subset=args.allow_dataset_subset,
            benchmark_handoff_generator_factory=args.benchmark_handoff_generator_factory,
            benchmark_handoff_output_dir=args.benchmark_handoff_output_dir,
            benchmark_handoff_dtype=args.benchmark_handoff_dtype,
            benchmark_handoff_align_bytes=args.benchmark_handoff_align_bytes,
            benchmark_handoff_generation_timeout_seconds=(
                args.benchmark_handoff_generation_timeout_seconds
            ),
            benchmark_handoff_limit=args.benchmark_handoff_limit,
            benchmark_handoff_segment_per_document=(
                args.benchmark_handoff_chunk_per_document
            ),
            benchmark_handoff_cache_method=(
                args.benchmark_handoff_cache_method
            ),
            benchmark_handoff_require_artifact_contract=(
                not args.benchmark_handoff_allow_legacy_artifact_contract
            ),
            runtime_identity=(
                None
                if args.runtime_identity_json is None
                else _runtime_identity_from_json(args.runtime_identity_json)
            ),
            spark_env_vars=_spark_env_vars_from_cli(args.spark_env_var or ()),
        )
        if args.runner_script_output:
            write_databricks_vllm_smoke_runner_script(args.runner_script_output)
        payload = build_databricks_vllm_smoke_run_submit_payload(config)
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
