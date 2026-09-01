"""Databricks-friendly vLLM smoke benchmark for the V1 Qwen3 path."""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any
import urllib.error
import urllib.request

from document_kv_cache._hardware_targets import (
    validate_v1_vllm_kv_cache_dtype_for_hardware_target,
)
from document_kv_cache._benchmark_manifest import (
    _resource_software_identity_from_package_revisions,
)
from document_kv_cache.artifact_identity import (
    RuntimeIdentity,
    UNRESOLVED_IDENTITY,
)
from document_kv_cache.benchmark_handoffs import (
    enrich_benchmark_jsonl_with_handoffs,
    generate_benchmark_handoff_bundles,
    load_benchmark_kv_chunk_generator,
)
from document_kv_cache.benchmark_runner import (
    PREFIX_CACHE_SALT_MODES,
    load_v1_jsonl_suite,
)
from document_kv_cache.canary_orchestration import (
    REPRESENTATIVE_VLLM_PACKAGE_PINS,
    benchmark_json_mapping_to_record,
    benchmark_manifest_provenance_runner_args,
    build_handoff_topology_attestation,
    generator_token_counter,
    merge_handoff_topology_attestations,
    representative_canary_matrix,
    representative_vllm_comparison_suite_id,
    representative_vllm_environment_provenance,
    require_pinned_revision,
    resolved_layout_rope_provenance,
    validate_handoff_topology_attestation,
    validated_benchmark_arm_specs,
    validated_benchmark_manifest_provenance,
)
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    CACHE_REUSE_ARM,
    CACHET_BENCHMARK_SYSTEM_PROMPT_POSITION_ENV,
    DEFAULT_HARDWARE_TARGET,
    DEFAULT_SYSTEM_PROMPT_POSITION,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM,
    DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    SUPPORTED_V1_DATASETS,
    SUPPORTED_V1_HARDWARE_TARGETS,
    SYSTEM_PROMPT_POSITIONS,
    build_prompt_parts,
    validate_v1_hardware_target,
)
from document_kv_cache.engine_adapters import (
    ServingBackend,
    read_engine_adapter_request_json,
    validate_engine_adapter_request_record,
)
from document_kv_cache.engine_probe import _validate_local_payload_uri
from document_kv_cache.dataset_prep import write_v1_jsonl
from document_kv_cache.model_profiles import (
    QWEN3_4B_INSTRUCT_HF_MODEL_ID,
    QWEN3_4B_ROPE_ROTARY_DIM,
    QWEN3_4B_ROPE_THETA,
    layout_for_model,
)
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PATCHED_WHEEL_SHA256,
)
from document_kv_cache.models import CacheGenerationMethod
from document_kv_cache.publication_handoff_artifacts import (
    PUBLICATION_HANDOFF_STAGING_ATTESTATION_FILENAME,
    read_publication_latency_handoff_bundle,
    stage_publication_latency_handoff_bundle,
    validate_publication_latency_handoff_bundle,
)
from document_kv_cache.publication_latency_handoff_generation import (
    PUBLICATION_LATENCY_HANDOFF_EXECUTION_FILENAME,
    PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED,
    read_publication_latency_handoff_generation_result,
    resolve_publication_latency_worker_handoff_bundle,
)
from document_kv_cache.publication_inputs import (
    PublicationLatencyExample,
    validate_publication_latency_block_schedule,
)
from document_kv_cache.runtime_telemetry import (
    RuntimeTelemetrySampler,
    bind_runtime_resource_evidence_record_file,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    VLLM_PATCHED_WHEEL_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
)
from document_kv_cache.storage import local_path
from document_kv_cache.serving_env import (
    FASTAPI_CONSTRAINT,
    HUGGINGFACE_HUB_CONSTRAINT,
    NUMPY_CONSTRAINT,
    OPENCV_PYTHON_HEADLESS_CONSTRAINT,
    PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT,
    FLASHINFER_PYTHON_CONSTRAINT,
    TOKENIZERS_CONSTRAINT,
    TORCH_CONSTRAINT,
    TORCHAUDIO_CONSTRAINT,
    TORCHCODEC_CONSTRAINT,
    TORCHVISION_CONSTRAINT,
    TRANSFORMERS_CONSTRAINT,
    TRITON_CONSTRAINT,
    VLLM_CUDA_REQUIREMENTS_SHA256,
    VLLM_CUDA_VARIANT,
    VLLM_DOCKERFILE_SHA256,
    VLLM_PACKAGE_VERSION,
    VLLM_PACKAGE_INDEX_URLS,
    VLLM_PATCHED_WHEEL_SHA256_ENV,
    VLLM_PATCHED_WHEEL_URI_ENV,
    VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_LOCK_FILENAME,
    VLLM_RUNTIME_LOCK_SHA256,
    VLLM_SERVING_ENVIRONMENT_PROFILE,
    VLLM_VERSION,
    VLLM_WHEEL_FILENAME,
    VLLM_WHEEL_SHA256,
    VLLM_WHEEL_URL,
    VIRTUALENV_BOOTSTRAP_FILENAME,
    VIRTUALENV_BOOTSTRAP_SHA256,
    VIRTUALENV_BOOTSTRAP_URL,
    VIRTUALENV_BOOTSTRAP_VERSION,
    patched_vllm_wheel_install_spec,
    validate_vllm_runtime_lock_platform,
    vllm_runtime_install_requirements,
    vllm_runtime_lock_path,
)
from document_kv_cache.transformers_generator import (
    CACHET_TRANSFORMERS_DEVICE_MAP_ENV,
    CACHET_TRANSFORMERS_MODEL_ID_ENV,
    CACHET_TRANSFORMERS_MODEL_REVISION_ENV,
    CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV,
    CACHET_TRANSFORMERS_QUANTIZATION_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_ID_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV,
    CACHET_TRANSFORMERS_TORCH_DTYPE_ENV,
    CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV,
)
from vllm_kv_injection.vllm_transfer_config import (
    document_kv_transfer_config,
    multi_connector_transfer_config_json,
)
from vllm_kv_injection.vllm_dynamic_connector import (
    DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY,
    DocumentKVConnector,
    NoOpDocumentKVProvider,
)
from document_kv_cache.vllm_wheel_repack import (
    VLLM_0271_E5M2_PATCH_CLOSURE as _APPROVED_VLLM_0271_E5M2_PATCH_CLOSURE,
    validate_patched_vllm_member_bytes,
)

HF_MODEL_ID = QWEN3_4B_INSTRUCT_HF_MODEL_ID
SERVED_MODEL_NAME = "qwen3:4b-instruct"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
BASELINE_PREFIX_CACHE_SALT = "cachet-baseline-prefill"
CACHE_PREFIX_CACHE_SALT = "cachet-kv-cache"
PAYLOAD_CACHE_PRIME_PREFIX_CACHE_SALT = "cachet-payload-cache-prime"
PREPARED_PREFIX_CACHE_SALT_MODE = "per_request"
SERVER_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
SMOKE_DATASETS = ("biography", "hotpotqa", "musique", "niah")
BENCHMARK_ARM_IDS = (BASELINE_PREFILL_ARM, CACHE_REUSE_ARM)
CACHET_KV_CONNECTOR_MODE = "cachet"
LMCACHE_KV_CONNECTOR_MODE = "lmcache"
# Hybrid handoff: Cachet serves turn-1 document requests, LMCache serves turn-2+
# follow-ups and document-free conversations (via vLLM MultiConnector).
MULTI_KV_CONNECTOR_MODE = "multi"
KV_CONNECTOR_MODES = (
    CACHET_KV_CONNECTOR_MODE,
    LMCACHE_KV_CONNECTOR_MODE,
    MULTI_KV_CONNECTOR_MODE,
)
LMCACHE_CONNECTOR_CLASS = "LMCacheConnectorV1"
DEFAULT_LOCAL_ROOT = Path("/local_disk0")
DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV = "DOCUMENT_KV_PACKAGE_INSTALL_SPEC"
DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV = "DOCUMENT_KV_PACKAGE_WHEEL_SHA256"
VLLM_FIPS_OPENCV_OVERRIDE_CONSTRAINT = OPENCV_PYTHON_HEADLESS_CONSTRAINT
VLLM_USE_FLASHINFER_SAMPLER_ENV = "VLLM_USE_FLASHINFER_SAMPLER"
PROMPT_TOKEN_PROBE_ADD_SPECIAL_TOKENS = False

# Keep the runtime verifier and wheel builder on one authoritative closure.
_VLLM_0271_E5M2_PATCH_CLOSURE = _APPROVED_VLLM_0271_E5M2_PATCH_CLOSURE

__all__ = [
    "VLLM_VERSION",
    "VLLM_PACKAGE_VERSION",
    "TRANSFORMERS_CONSTRAINT",
    "HUGGINGFACE_HUB_CONSTRAINT",
    "TOKENIZERS_CONSTRAINT",
    "NUMPY_CONSTRAINT",
    "FASTAPI_CONSTRAINT",
    "PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT",
    "HF_MODEL_ID",
    "SERVED_MODEL_NAME",
    "SERVER_BASE_URL",
    "SMOKE_DATASETS",
    "DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV",
    "DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV",
    "VLLM_PATCHED_WHEEL_URI_ENV",
    "VLLM_PATCHED_WHEEL_SHA256_ENV",
    "VLLMRepresentativeWorkloadProfile",
    "VLLM_REPRESENTATIVE_WORKLOAD_PROFILES",
    "vllm_representative_workload_profile",
    "VLLMSmokeBenchmarkConfig",
    "VLLMNativeRuntimeBundleV2",
    "VLLMPreparedHandoffGenerationConfig",
    "run_vllm_smoke_benchmark",
    "build_metadata",
    "cache_measurement_protocol",
    "build_vllm_native_provider_probe_record",
    "cuda_wheel_env_paths",
    "dependency_constraints",
    "dependency_index_args",
    "vllm_dependency_install_requirements",
    "patched_vllm_wheel_install_spec",
    "dependency_override_constraints",
    "document_kv_package_install_spec",
    "install_document_kv_package",
    "install_native_v2_runtime",
    "installed_package_freeze",
    "verify_vllm_runtime_lock_installation",
    "materialize_virtualenv_bootstrap",
    "verify_vllm_runtime_patch_closure",
    "build_vllm_server_args",
    "document_kv_transfer_config_for_smoke",
    "build_benchmark_runner_args",
    "build_prompt_token_budget_rows",
    "prepared_benchmark_handoff_coverage_record",
    "validate_prepared_benchmark_handoffs",
    "run_prompt_token_budget_probe",
    "validate_prompt_token_budget",
    "write_prompt_token_budget_jsonl",
    "benchmark_dataset_paths",
    "write_smoke_datasets",
    "prepare_generated_benchmark_handoffs",
    "prepare_publication_latency_inputs",
    "release_handoff_generation_resources",
    "prime_payload_cache",
    "attest_payload_cache_measurements",
    "smoke_dataset_records",
    "parse_dataset_specs",
    "dataset_args",
    "parse_args",
    "site_packages_dirs",
    "main",
    "VLLM_FIPS_OPENCV_OVERRIDE_CONSTRAINT",
]


@dataclass(frozen=True, slots=True)
class VLLMRepresentativeWorkloadProfile:
    """One allowed, reproducible representative vLLM canary workload."""

    profile_id: str
    input_tokens_target: int
    max_output_tokens: int
    max_model_len: int
    max_num_seqs: int = 2
    gpu_memory_utilization: float = 0.85
    model_dtype: str = "bfloat16"
    runtime_kv_dtype: str = "bfloat16"
    benchmark_repeats: int = 3
    request_parallelism: int = 1
    prefix_cache_salt_mode: str = PREPARED_PREFIX_CACHE_SALT_MODE
    force_max_tokens: bool = True
    prewarm_cache_prefix: bool = False
    cache_runtime_prompt: bool = False
    payload_cache_max_bytes: int = 0
    kv_connector_mode: str = CACHET_KV_CONNECTOR_MODE
    benchmark_evidence_policy: str = "canary"
    multi_document_datasets: tuple[str, ...] = ("hotpotqa", "musique")

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        for field_name in (
            "input_tokens_target",
            "max_output_tokens",
            "max_model_len",
            "max_num_seqs",
            "benchmark_repeats",
            "request_parallelism",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.gpu_memory_utilization != 0.85:
            raise ValueError(
                "representative vLLM profiles require gpu_memory_utilization=0.85"
            )
        if self.model_dtype != "bfloat16" or self.runtime_kv_dtype != "bfloat16":
            raise ValueError(
                "representative vLLM profiles require BF16 model and KV dtypes"
            )
        if self.prefix_cache_salt_mode != "per_request":
            raise ValueError(
                "representative vLLM profiles require per_request cache salts"
            )
        if self.force_max_tokens is not True:
            raise ValueError("representative vLLM profiles must force max tokens")
        if self.prewarm_cache_prefix is not False:
            raise ValueError("representative vLLM profiles must disable prewarming")
        if self.cache_runtime_prompt is not False:
            raise ValueError(
                "representative vLLM profiles must send the logical prompt"
            )
        if self.payload_cache_max_bytes != 0:
            raise ValueError(
                "representative vLLM profiles must disable the payload cache"
            )
        if self.kv_connector_mode != CACHET_KV_CONNECTOR_MODE:
            raise ValueError(
                "representative vLLM profiles require the Cachet connector"
            )
        if self.benchmark_evidence_policy != "canary":
            raise ValueError("representative vLLM profiles require canary evidence")
        if not self.multi_document_datasets or any(
            not isinstance(dataset, str) or not dataset
            for dataset in self.multi_document_datasets
        ):
            raise ValueError(
                "representative vLLM profiles require named multi-document datasets"
            )


VLLM_REPRESENTATIVE_WORKLOAD_PROFILES = (
    VLLMRepresentativeWorkloadProfile(
        profile_id="vllm-8k-64-v1",
        input_tokens_target=8_192,
        max_output_tokens=64,
        max_model_len=8_512,
    ),
    VLLMRepresentativeWorkloadProfile(
        profile_id="vllm-16k-256-v1",
        input_tokens_target=16_384,
        max_output_tokens=256,
        max_model_len=16_896,
    ),
)


def vllm_representative_workload_profile(
    value: VLLMRepresentativeWorkloadProfile | str,
) -> VLLMRepresentativeWorkloadProfile:
    if isinstance(value, VLLMRepresentativeWorkloadProfile):
        if value not in VLLM_REPRESENTATIVE_WORKLOAD_PROFILES:
            raise ValueError(
                "representative_workload_profile must be a registered profile"
            )
        return value
    if not isinstance(value, str) or not value:
        raise ValueError("representative_workload_profile must be a profile ID")
    for profile in VLLM_REPRESENTATIVE_WORKLOAD_PROFILES:
        if profile.profile_id == value:
            return profile
    supported = tuple(
        profile.profile_id for profile in VLLM_REPRESENTATIVE_WORKLOAD_PROFILES
    )
    raise ValueError(
        f"unknown representative_workload_profile {value!r}; expected one of {supported}"
    )


@dataclass(frozen=True, slots=True)
class VLLMPreparedHandoffGenerationConfig:
    """Optional generation settings for prepared vLLM benchmark handoffs."""

    generator_factory: str
    output_dir: Path
    dtype: str = "bfloat16"
    align_bytes: int = 4096
    timeout_seconds: float = 1800.0
    limit: int | None = None
    benchmark_handoff_segment_per_document: bool = False
    cache_method: str | None = None
    require_artifact_contract: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.generator_factory, str)
            or not self.generator_factory.strip()
        ):
            raise ValueError("benchmark_handoff_generator_factory must be non-empty")
        if self.output_dir is None:
            raise ValueError("benchmark_handoff_output_dir must be provided")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError("benchmark_handoff_dtype must be non-empty")
        if type(self.align_bytes) is not int or self.align_bytes <= 0:
            raise ValueError("benchmark_handoff_align_bytes must be a positive integer")
        if self.timeout_seconds <= 0:
            raise ValueError("benchmark_handoff_timeout_seconds must be positive")
        if self.limit is not None:
            if (
                isinstance(self.limit, bool)
                or not isinstance(self.limit, int)
                or self.limit < 0
            ):
                raise ValueError(
                    "benchmark_handoff_limit must be a non-negative integer"
                )
        if not isinstance(self.benchmark_handoff_segment_per_document, bool):
            raise ValueError("benchmark_handoff_segment_per_document must be a boolean")
        if self.cache_method is not None:
            object.__setattr__(
                self,
                "cache_method",
                _non_empty_string(
                    self.cache_method,
                    "benchmark_handoff_cache_method",
                ),
            )
        if (
            self.cache_method == CacheGenerationMethod.VANILLA_PREFILL.value
            and not self.benchmark_handoff_segment_per_document
        ):
            raise ValueError(
                "vanilla_prefill handoff generation requires one segment per document"
            )
        if (
            self.cache_method == CacheGenerationMethod.FULL_PREFIX_PREFILL.value
            and self.benchmark_handoff_segment_per_document
        ):
            raise ValueError(
                "full_prefix_prefill handoff generation requires one full-prefix segment"
            )
        if type(self.require_artifact_contract) is not bool:
            raise ValueError(
                "benchmark_handoff_require_artifact_contract must be a boolean"
            )
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def to_metadata(self) -> dict[str, object]:
        return {
            "generator_factory": self.generator_factory,
            "output_dir": str(self.output_dir),
            "dtype": self.dtype,
            "align_bytes": self.align_bytes,
            "timeout_seconds": self.timeout_seconds,
            "limit": self.limit,
            "segment_per_document": self.benchmark_handoff_segment_per_document,
            "cache_method": self.cache_method,
            "require_artifact_contract": self.require_artifact_contract,
        }


_VLLM_NATIVE_RUNTIME_V2_RECORD_KEYS = (
    "package_wheel_sha256",
    "package_wheel_uri",
    "patched_flashinfer_wheel_sha256",
    "patched_flashinfer_wheel_uri",
    "patched_vllm_wheel_sha256",
    "patched_vllm_wheel_uri",
    "runtime_closure_manifest_sha256",
    "runtime_closure_manifest_uri",
    "runtime_lock_sha256",
    "runtime_lock_uri",
)


@dataclass(frozen=True, slots=True)
class VLLMNativeRuntimeBundleV2:
    """Exact mounted artifacts for the native-v2 publication runtime."""

    runtime_lock_uri: str
    runtime_lock_sha256: str
    patched_vllm_wheel_uri: str
    patched_vllm_wheel_sha256: str
    patched_flashinfer_wheel_uri: str
    patched_flashinfer_wheel_sha256: str
    runtime_closure_manifest_uri: str
    runtime_closure_manifest_sha256: str
    package_wheel_uri: str
    package_wheel_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_lock_uri",
            "patched_vllm_wheel_uri",
            "patched_flashinfer_wheel_uri",
            "runtime_closure_manifest_uri",
            "package_wheel_uri",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value
                or value.strip() != value
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError(f"{field_name} must be a non-empty canonical path")
            local_path = Path(_cluster_file_path(value))
            if not local_path.is_absolute():
                raise ValueError(f"{field_name} must resolve to an absolute path")
        local_paths = {
            _cluster_file_path(getattr(self, field_name))
            for field_name in (
                "runtime_lock_uri",
                "patched_vllm_wheel_uri",
                "patched_flashinfer_wheel_uri",
                "runtime_closure_manifest_uri",
                "package_wheel_uri",
            )
        }
        if len(local_paths) != 5:
            raise ValueError("native-v2 runtime artifact paths must be distinct")
        for field_name in (
            "runtime_lock_sha256",
            "patched_vllm_wheel_sha256",
            "patched_flashinfer_wheel_sha256",
            "runtime_closure_manifest_sha256",
            "package_wheel_sha256",
        ):
            _validated_sha256_digest(getattr(self, field_name), field_name)
        fixed = {
            "runtime_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
            "patched_vllm_wheel_sha256": VLLM_PATCHED_WHEEL_SHA256,
            "patched_flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
            "runtime_closure_manifest_sha256": (
                RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
            ),
        }
        for field_name, expected in fixed.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} differs from the native-v2 authority")

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> VLLMNativeRuntimeBundleV2:
        """Parse the exact ten-key native-v2 runtime mapping."""

        if not isinstance(value, Mapping) or set(value) != set(
            _VLLM_NATIVE_RUNTIME_V2_RECORD_KEYS
        ):
            raise ValueError("native-v2 runtime bundle keys differ")
        normalized: dict[str, str] = {}
        for field_name in _VLLM_NATIVE_RUNTIME_V2_RECORD_KEYS:
            field_value = value[field_name]
            if not isinstance(field_value, str):
                raise TypeError(f"{field_name} must be a string")
            normalized[field_name] = field_value
        return cls(**normalized)

    def to_record(self) -> dict[str, str]:
        """Return the canonical ten-key URI and digest mapping."""

        return {
            field_name: getattr(self, field_name)
            for field_name in _VLLM_NATIVE_RUNTIME_V2_RECORD_KEYS
        }

    def local_path(self, artifact: str) -> Path:
        """Resolve one named artifact to its mounted local path."""

        field_name = f"{artifact}_uri"
        if field_name not in {
            "runtime_lock_uri",
            "patched_vllm_wheel_uri",
            "patched_flashinfer_wheel_uri",
            "runtime_closure_manifest_uri",
            "package_wheel_uri",
        }:
            raise ValueError("unknown native-v2 runtime artifact")
        return Path(_cluster_file_path(getattr(self, field_name)))


@dataclass(frozen=True)
class VLLMSmokeBenchmarkConfig:
    """Runtime configuration for a one-node Databricks vLLM smoke run."""

    benchmark_id: str
    output_dir: Path
    model_id: str = HF_MODEL_ID
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    model_dtype: str = "bfloat16"
    model_quantization: str | None = None
    kv_cache_dtype: str | None = None
    attention_backend: str | None = None
    max_tokens: int = 32
    force_max_tokens: bool = False
    timeout_seconds: float = 240.0
    import_probe_timeout_seconds: float = 180.0
    server_start_timeout_seconds: float = 480.0
    local_root: Path = DEFAULT_LOCAL_ROOT
    server_host: str = SERVER_HOST
    server_port: int = SERVER_PORT
    client_host: str = SERVER_HOST
    max_model_len: int = 4096
    max_num_seqs: int = 2
    gpu_memory_utilization: float = 0.85
    data_parallel_size: int = 1
    kv_connector_mode: str = "cachet"
    lmcache_local_dir: str = "/local_disk0/lmcache-store"
    lmcache_max_disk_gb: float = 80.0
    lmcache_chunk_size: int = 256
    lmcache_version: str = ""
    lmcache_local_cpu: bool = False
    lmcache_max_cpu_gb: float = 0.0
    benchmark_repeats: int = 1
    request_parallelism: int = 1
    benchmark_interleave_examples: bool = False
    runtime_telemetry_interval_seconds: float = 1.0
    benchmark_arms: tuple[str, ...] = ()
    benchmark_arm_specs: tuple[Mapping[str, Any], ...] = ()
    benchmark_evidence_policy: str | None = None
    representative_canary: bool = False
    representative_workload_profile: VLLMRepresentativeWorkloadProfile | str | None = (
        None
    )
    benchmark_manifest_provenance: Mapping[str, Any] = field(default_factory=dict)
    prewarm_cache_prefix: bool = False
    cache_runtime_prompt: bool = False
    prefix_cache_salt_mode: str = PREPARED_PREFIX_CACHE_SALT_MODE
    hardware_target: str = DEFAULT_HARDWARE_TARGET
    dataset_specs: tuple[str, ...] = ()
    allow_dataset_subset: bool = False
    package_install_spec: str | None = None
    handoff_generation: VLLMPreparedHandoffGenerationConfig | None = None
    runtime_identity: RuntimeIdentity | None = None
    payload_cache_max_bytes: int = 0
    system_prompt_position: str = DEFAULT_SYSTEM_PROMPT_POSITION
    benchmark_suite_id: str | None = None
    benchmark_runtime_id: str | None = None
    prewarm_payload_cache: bool = False
    publication_latency_schedule_record: Mapping[str, Any] | None = None
    publication_latency_schedule_path: Path | None = None
    publication_latency_expected_input_bundle_sha256: str | None = None
    publication_handoff_generation_output_root: Path | None = None
    publication_handoff_generation_execution_file_sha256: str | None = None
    publication_handoff_generation_execution_closed_record_sha256: str | None = None
    publication_handoff_bundle_manifest_path: Path | None = None
    publication_handoff_bundle_source_root: Path | None = None
    publication_handoff_bundle_manifest_file_sha256: str | None = None
    publication_handoff_bundle_manifest_closed_record_sha256: str | None = None
    publication_handoff_local_nvme_dir: Path | None = None
    publication_handoff_stage_kind: str = "local_nvme"
    temperature: float = 0.0
    generation_seed: int | None = None
    payload_cache_prime_target_count: int | None = None
    native_runtime_v2: VLLMNativeRuntimeBundleV2 | None = None

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if self.output_dir is None:
            raise ValueError("output_dir must be provided")
        for field_name in ("benchmark_suite_id", "benchmark_runtime_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _non_empty_string(value, field_name),
                )
        object.__setattr__(
            self, "model_id", _non_empty_string(self.model_id, "model_id")
        )
        for field_name in ("model_revision", "tokenizer_revision"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _non_empty_string(value, field_name),
                )
        object.__setattr__(
            self, "model_dtype", _non_empty_string(self.model_dtype, "model_dtype")
        )
        if self.model_quantization is not None:
            object.__setattr__(
                self,
                "model_quantization",
                _non_empty_string(self.model_quantization, "model_quantization"),
            )
        if self.kv_cache_dtype is not None:
            object.__setattr__(
                self,
                "kv_cache_dtype",
                _non_empty_string(self.kv_cache_dtype, "kv_cache_dtype"),
            )
        if self.attention_backend is not None:
            object.__setattr__(
                self,
                "attention_backend",
                _non_empty_string(self.attention_backend, "attention_backend"),
            )
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if type(self.force_max_tokens) is not bool:
            raise ValueError("force_max_tokens must be a boolean")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a non-negative finite number")
        object.__setattr__(self, "temperature", float(self.temperature))
        if self.generation_seed is not None and type(self.generation_seed) is not int:
            raise ValueError("generation_seed must be an integer when provided")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.import_probe_timeout_seconds <= 0:
            raise ValueError("import_probe_timeout_seconds must be positive")
        if self.server_start_timeout_seconds <= 0:
            raise ValueError("server_start_timeout_seconds must be positive")
        if self.local_root is None:
            raise ValueError("local_root must be provided")
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
        if self.data_parallel_size <= 0:
            raise ValueError("data_parallel_size must be positive")
        if self.kv_connector_mode not in KV_CONNECTOR_MODES:
            raise ValueError(
                f"kv_connector_mode must be one of {sorted(KV_CONNECTOR_MODES)}"
            )
        if self.system_prompt_position not in SYSTEM_PROMPT_POSITIONS:
            raise ValueError(
                f"system_prompt_position must be one of {sorted(SYSTEM_PROMPT_POSITIONS)}"
            )
        if isinstance(self.benchmark_repeats, bool) or not isinstance(
            self.benchmark_repeats, int
        ):
            raise TypeError("benchmark_repeats must be a positive integer")
        if self.benchmark_repeats <= 0:
            raise ValueError("benchmark_repeats must be a positive integer")
        if isinstance(self.request_parallelism, bool) or not isinstance(
            self.request_parallelism, int
        ):
            raise TypeError("request_parallelism must be a positive integer")
        if self.request_parallelism <= 0:
            raise ValueError("request_parallelism must be a positive integer")
        if self.runtime_telemetry_interval_seconds <= 0:
            raise ValueError("runtime_telemetry_interval_seconds must be positive")
        object.__setattr__(
            self, "benchmark_arms", _validated_benchmark_arms(self.benchmark_arms)
        )
        object.__setattr__(
            self,
            "benchmark_arm_specs",
            validated_benchmark_arm_specs(self.benchmark_arm_specs),
        )
        if self.benchmark_arms and self.benchmark_arm_specs:
            raise ValueError(
                "benchmark_arms and benchmark_arm_specs are mutually exclusive"
            )
        if self.benchmark_evidence_policy not in {
            None,
            "smoke",
            "canary",
            "publication",
        }:
            raise ValueError(
                "benchmark_evidence_policy must be smoke, canary, publication, or None"
            )
        if self.benchmark_evidence_policy in {"canary", "publication"}:
            if self.kv_connector_mode in {
                LMCACHE_KV_CONNECTOR_MODE,
                MULTI_KV_CONNECTOR_MODE,
            }:
                raise ValueError(
                    "vLLM 0.27.1 canary/publication runs do not support "
                    f"kv_connector_mode={self.kv_connector_mode!r}: LMCache and "
                    "Multi are explicitly N/A until they have a separate "
                    "content-addressed, hash-locked runtime closure"
                )
            if self.attention_backend is None:
                object.__setattr__(self, "attention_backend", "TRITON_ATTN")
            elif self.attention_backend != "TRITON_ATTN":
                raise ValueError(
                    "vLLM 0.27.1 canary/publication runs require "
                    "attention_backend='TRITON_ATTN' for the campaign sentinel "
                    "and measurement matrix"
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
                hardware_target=self.hardware_target,
                profile_id=representative_profile.profile_id,
            )
            if self.benchmark_suite_id is None:
                raise ValueError("representative benchmark_suite_id must be resolved")
            if self.benchmark_suite_id != expected_suite_id:
                raise ValueError(
                    "representative benchmark_suite_id must match the "
                    "hardware/profile comparison group"
                )
            if self.benchmark_runtime_id in {None, UNRESOLVED_IDENTITY}:
                raise ValueError("representative benchmark_runtime_id must be resolved")
        provenance = validated_benchmark_manifest_provenance(
            self.benchmark_manifest_provenance
        )
        if self.benchmark_runtime_id is not None and "runtime_id" in provenance:
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
        if "resource" in provenance.get("measurement_scopes", ()):
            _resource_software_identity_from_package_revisions(
                provenance.get("package_revisions", {})
            )
        if self.requires_pinned_revisions:
            if (
                isinstance(
                    self.handoff_generation,
                    VLLMPreparedHandoffGenerationConfig,
                )
                and not self.handoff_generation.require_artifact_contract
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
            model_revision = require_pinned_revision(
                self.model_revision,
                "model_revision",
            )
            tokenizer_revision = require_pinned_revision(
                self.tokenizer_revision,
                "tokenizer_revision",
            )
            if provenance.get("model_revision", model_revision) != model_revision:
                raise ValueError(
                    "benchmark_manifest_provenance.model_revision must match model_revision"
                )
            if (
                provenance.get("tokenizer_revision", tokenizer_revision)
                != tokenizer_revision
            ):
                raise ValueError(
                    "benchmark_manifest_provenance.tokenizer_revision must match "
                    "tokenizer_revision"
                )
        if self.is_representative_submission:
            provenance = _resolved_representative_vllm_provenance(self, provenance)
            object.__setattr__(self, "benchmark_manifest_provenance", provenance)
        if (
            "input_tokens_target" in provenance
            and provenance.get("tokenizer_revision", self.tokenizer_revision) is None
        ):
            raise ValueError(
                "benchmark input_tokens_target requires a pinned tokenizer_revision"
            )
        if type(self.prewarm_cache_prefix) is not bool:
            raise TypeError("prewarm_cache_prefix must be a boolean")
        if type(self.prewarm_payload_cache) is not bool:
            raise TypeError("prewarm_payload_cache must be a boolean")
        if type(self.cache_runtime_prompt) is not bool:
            raise TypeError("cache_runtime_prompt must be a boolean")
        if (
            not isinstance(self.hardware_target, str)
            or not self.hardware_target.strip()
        ):
            raise ValueError("hardware_target must be non-empty")
        validate_v1_hardware_target(self.hardware_target)
        validate_v1_vllm_kv_cache_dtype_for_hardware_target(
            hardware_target=self.hardware_target,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        if type(self.allow_dataset_subset) is not bool:
            raise ValueError("allow_dataset_subset must be a boolean")
        if isinstance(self.payload_cache_max_bytes, bool) or not isinstance(
            self.payload_cache_max_bytes, int
        ):
            raise TypeError("payload_cache_max_bytes must be a non-negative integer")
        if self.payload_cache_max_bytes < 0:
            raise ValueError("payload_cache_max_bytes must be a non-negative integer")
        if self.payload_cache_prime_target_count is not None and (
            type(self.payload_cache_prime_target_count) is not int
            or self.payload_cache_prime_target_count <= 0
        ):
            raise ValueError(
                "payload_cache_prime_target_count must be a positive integer"
            )
        object.__setattr__(self, "dataset_specs", tuple(self.dataset_specs))
        if self.dataset_specs:
            parse_dataset_specs(
                self.dataset_specs, allow_subset=self.allow_dataset_subset
            )
        if (
            self.package_install_spec is not None
            and not self.package_install_spec.strip()
        ):
            raise ValueError("package_install_spec must be non-empty when provided")
        if self.handoff_generation is not None:
            if not isinstance(
                self.handoff_generation, VLLMPreparedHandoffGenerationConfig
            ):
                raise TypeError(
                    "handoff_generation must be a VLLMPreparedHandoffGenerationConfig"
                )
            if not self.dataset_specs:
                raise ValueError(
                    "benchmark_handoff_generator_factory requires prepared dataset specs"
                )
            if (
                not self.handoff_generation.require_artifact_contract
                and self.benchmark_evidence_policy in {"canary", "publication"}
            ):
                raise ValueError(
                    "canary and publication handoff generation require the complete "
                    "registered method artifact contract"
                )
        schedule_record = self.publication_latency_schedule_record
        schedule_path = self.publication_latency_schedule_path
        expected_input_bundle_sha256 = (
            self.publication_latency_expected_input_bundle_sha256
        )
        if schedule_record is not None and schedule_path is not None:
            raise ValueError(
                "publication_latency_schedule_record and "
                "publication_latency_schedule_path are mutually exclusive"
            )
        if schedule_record is not None:
            object.__setattr__(
                self,
                "publication_latency_schedule_record",
                _normalized_json_object(
                    schedule_record,
                    "publication_latency_schedule_record",
                ),
            )
        if schedule_path is not None:
            object.__setattr__(
                self,
                "publication_latency_schedule_path",
                _normalized_cluster_path(
                    schedule_path,
                    "publication_latency_schedule_path",
                ),
            )
        schedule_enabled = schedule_record is not None or schedule_path is not None
        if schedule_enabled != (expected_input_bundle_sha256 is not None):
            raise ValueError(
                "a publication latency schedule and "
                "publication_latency_expected_input_bundle_sha256 must be "
                "provided together"
            )
        if expected_input_bundle_sha256 is not None:
            object.__setattr__(
                self,
                "publication_latency_expected_input_bundle_sha256",
                _validated_sha256_digest(
                    expected_input_bundle_sha256,
                    "publication_latency_expected_input_bundle_sha256",
                ),
            )
        if schedule_enabled:
            if self.benchmark_evidence_policy not in {
                "smoke",
                "canary",
                "publication",
            }:
                raise ValueError(
                    "publication latency schedules require "
                    "benchmark_evidence_policy='smoke', 'canary', or 'publication'"
                )
            if not self.dataset_specs:
                raise ValueError(
                    "publication latency schedules require prepared dataset specs"
                )
            if self.allow_dataset_subset:
                raise ValueError(
                    "publication latency schedules require all governed datasets"
                )
            if self.benchmark_interleave_examples:
                raise ValueError(
                    "publication latency schedules own request order and forbid "
                    "benchmark_interleave_examples"
                )

        generated_handoff_fields = (
            "publication_handoff_generation_output_root",
            "publication_handoff_generation_execution_file_sha256",
            "publication_handoff_generation_execution_closed_record_sha256",
        )
        bundle_handoff_fields = (
            "publication_handoff_bundle_manifest_path",
            "publication_handoff_bundle_source_root",
            "publication_handoff_bundle_manifest_file_sha256",
            "publication_handoff_bundle_manifest_closed_record_sha256",
        )
        generated_values = tuple(
            getattr(self, name) for name in generated_handoff_fields
        )
        bundle_values = tuple(getattr(self, name) for name in bundle_handoff_fields)
        generated_handoff_enabled = all(value is not None for value in generated_values)
        bundle_handoff_enabled = all(value is not None for value in bundle_values)
        if any(value is not None for value in generated_values) and not (
            generated_handoff_enabled
        ):
            raise ValueError(
                "publication handoff generation output root and execution file/record "
                "SHA-256 values must be provided together"
            )
        if (
            any(value is not None for value in bundle_values)
            and not bundle_handoff_enabled
        ):
            raise ValueError(
                "publication handoff bundle manifest/source root and file/record "
                "SHA-256 values must be provided together"
            )
        if generated_handoff_enabled and bundle_handoff_enabled:
            raise ValueError(
                "distributed and directly closed publication handoff sources are "
                "mutually exclusive"
            )
        handoff_enabled = generated_handoff_enabled or bundle_handoff_enabled
        if handoff_enabled:
            if self.publication_handoff_local_nvme_dir is None:
                raise ValueError(
                    "publication handoff staging requires a staging directory"
                )
            path_fields = ["publication_handoff_local_nvme_dir"]
            if generated_handoff_enabled:
                path_fields.append("publication_handoff_generation_output_root")
            else:
                path_fields.extend(
                    (
                        "publication_handoff_bundle_manifest_path",
                        "publication_handoff_bundle_source_root",
                    )
                )
            for field_name in path_fields:
                object.__setattr__(
                    self,
                    field_name,
                    _normalized_cluster_path(getattr(self, field_name), field_name),
                )
            digest_fields = (
                (
                    "publication_handoff_generation_execution_file_sha256",
                    "publication_handoff_generation_execution_closed_record_sha256",
                )
                if generated_handoff_enabled
                else (
                    "publication_handoff_bundle_manifest_file_sha256",
                    "publication_handoff_bundle_manifest_closed_record_sha256",
                )
            )
            for field_name in digest_fields:
                object.__setattr__(
                    self,
                    field_name,
                    _validated_sha256_digest(getattr(self, field_name), field_name),
                )
            if not schedule_enabled:
                raise ValueError(
                    "publication handoff staging requires a publication latency schedule"
                )
            if not self.runs_document_kv_cache_arm:
                raise ValueError(
                    "publication handoff staging requires a Cachet/Vanilla benchmark arm"
                )
            if self.handoff_generation is not None:
                raise ValueError(
                    "publication handoff staging and inline handoff generation are "
                    "mutually exclusive"
                )
            if self.publication_handoff_stage_kind not in {
                "local_nvme",
                "uc_mounted",
            }:
                raise ValueError(
                    "publication_handoff_stage_kind must be local_nvme or uc_mounted"
                )
            assert self.publication_handoff_local_nvme_dir is not None
            stage_path = self.publication_handoff_local_nvme_dir.expanduser().resolve(
                strict=False
            )
            local_root = Path(self.local_root).expanduser().resolve(strict=False)
            if (
                self.publication_handoff_stage_kind == "local_nvme"
                and not stage_path.is_relative_to(local_root)
            ):
                raise ValueError(
                    "publication_handoff_local_nvme_dir must be inside local_root"
                )
            if self.publication_handoff_stage_kind == "uc_mounted" and not str(
                stage_path
            ).startswith("/Volumes/"):
                raise ValueError(
                    "uc_mounted publication handoff staging requires /Volumes"
                )
        elif schedule_enabled and self.runs_document_kv_cache_arm:
            if self.handoff_generation is not None:
                raise ValueError(
                    "publication/canary latency schedules forbid inline handoff "
                    "generation; provide a closed publication handoff bundle"
                )
            raise ValueError(
                "a publication/canary Vanilla latency run requires a closed "
                "distributed handoff-generation record and local NVMe directory"
            )
        if self.runtime_identity is not None and not isinstance(
            self.runtime_identity,
            RuntimeIdentity,
        ):
            raise TypeError("runtime_identity must be a RuntimeIdentity or None")
        if self.runtime_identity is not None:
            if self.model_revision is None or self.tokenizer_revision is None:
                raise ValueError(
                    "runtime_identity requires pinned model_revision and "
                    "tokenizer_revision"
                )
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
        if self.prewarm_cache_prefix and not self.dataset_specs:
            raise ValueError(
                "benchmark_prewarm_cache_prefix requires prepared dataset specs"
            )
        if self.prewarm_payload_cache and not self.dataset_specs:
            raise ValueError(
                "benchmark_prewarm_payload_cache requires prepared dataset specs"
            )
        if self.prewarm_payload_cache and self.payload_cache_max_bytes <= 0:
            raise ValueError(
                "benchmark_prewarm_payload_cache requires a positive payload_cache_max_bytes"
            )
        if (
            self.payload_cache_prime_target_count is not None
            and not self.prewarm_payload_cache
        ):
            raise ValueError(
                "payload_cache_prime_target_count requires prewarm_payload_cache"
            )
        if self.prewarm_payload_cache and self.prewarm_cache_prefix:
            raise ValueError(
                "benchmark_prewarm_payload_cache and benchmark_prewarm_cache_prefix "
                "are mutually exclusive"
            )
        if (
            self.prewarm_payload_cache
            and self.kv_connector_mode != CACHET_KV_CONNECTOR_MODE
        ):
            raise ValueError(
                "benchmark_prewarm_payload_cache requires kv_connector_mode='cachet'"
            )
        if self.prewarm_payload_cache and self.data_parallel_size != 1:
            raise ValueError(
                "benchmark_prewarm_payload_cache requires data_parallel_size=1 because "
                "the provider payload cache is process-local"
            )
        if self.prewarm_payload_cache and not self.runs_document_kv_cache_arm:
            raise ValueError(
                "benchmark_prewarm_payload_cache requires a Cachet handoff benchmark arm"
            )
        if self.cache_runtime_prompt and not self.dataset_specs:
            raise ValueError(
                "benchmark_cache_runtime_prompt requires prepared dataset specs"
            )
        if self.prefix_cache_salt_mode not in PREFIX_CACHE_SALT_MODES:
            raise ValueError("prefix_cache_salt_mode must be 'static' or 'per_request'")
        if self.prewarm_cache_prefix and self.prefix_cache_salt_mode != "static":
            raise ValueError(
                "benchmark_prewarm_cache_prefix requires prefix_cache_salt_mode='static' "
                "so prewarmed prefix-cache blocks can be reused"
            )
        if self.prewarm_payload_cache and self.prefix_cache_salt_mode != "per_request":
            raise ValueError(
                "benchmark_prewarm_payload_cache requires prefix_cache_salt_mode='per_request' "
                "so every measured GPU prefix-cache key is isolated from priming"
            )
        if self.is_representative_submission:
            if (
                VLLM_SERVING_ENVIRONMENT_PROFILE.dependency_constraints
                != REPRESENTATIVE_VLLM_PACKAGE_PINS
            ):
                raise ValueError(
                    "representative vLLM serving package pins do not match the "
                    "approved workload manifest"
                )
            _validate_vllm_representative_workload(self)
        native_runtime = self.native_runtime_v2
        if native_runtime is not None and not isinstance(
            native_runtime, VLLMNativeRuntimeBundleV2
        ):
            raise TypeError("native_runtime_v2 must be a VLLMNativeRuntimeBundleV2")
        if native_runtime is not None and self.package_install_spec is not None:
            configured_package = os.path.normpath(
                _cluster_file_path(self.package_install_spec)
            )
            native_package = os.path.normpath(
                str(native_runtime.local_path("package_wheel"))
            )
            if configured_package != native_package:
                raise ValueError(
                    "package_install_spec must match native_runtime_v2 package_wheel_uri"
                )
        if self.benchmark_evidence_policy == "publication" and native_runtime is None:
            raise ValueError(
                "publication benchmarks require a complete native_runtime_v2 bundle"
            )

    @property
    def local_dir(self) -> Path:
        return self.local_root / f"document-kv-vllm-smoke-{self.benchmark_id}"

    @property
    def hf_cache_dir(self) -> Path:
        return self.local_root / "hf-cache"

    @property
    def server_base_url(self) -> str:
        return f"http://{self.client_host}:{self.server_port}"

    @property
    def venv_dir(self) -> Path:
        return self.local_dir / "vllm-venv"

    @property
    def venv_python(self) -> Path:
        return self.venv_dir / "bin" / "python"

    @property
    def server_log_path(self) -> Path:
        return self.local_dir / "vllm-server.log"

    @property
    def server_log_copy_path(self) -> Path:
        return self.output_dir / "vllm-server.log"

    @property
    def connector_telemetry_path(self) -> Path:
        return self.local_dir / "document-kv-connector-telemetry.jsonl"

    @property
    def connector_telemetry_copy_path(self) -> Path:
        return self.output_dir / "document-kv-connector-telemetry.jsonl"

    @property
    def runtime_telemetry_path(self) -> Path:
        return self.local_dir / "runtime-telemetry.json"

    @property
    def runtime_telemetry_copy_path(self) -> Path:
        return self.output_dir / "runtime-telemetry.json"

    @property
    def benchmark_output_path(self) -> Path:
        return self.output_dir / "v1-benchmark.json"

    @property
    def prompt_token_budget_path(self) -> Path:
        return self.output_dir / "prompt-token-budget.json"

    @property
    def prewarm_cache_prefix_path(self) -> Path:
        return self.output_dir / "prewarm-cache-prefix.json"

    @property
    def prewarm_payload_cache_path(self) -> Path:
        return self.output_dir / "prewarm-payload-cache.json"

    @property
    def payload_cache_attestation_path(self) -> Path:
        return self.output_dir / "payload-cache-attestation.json"

    @property
    def prompt_token_budget_input_path(self) -> Path:
        return self.local_dir / "prompt-token-budget-input.jsonl"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"

    @property
    def import_probe_path(self) -> Path:
        return self.output_dir / "vllm-import-probe.json"

    @property
    def prepared_handoff_coverage_path(self) -> Path:
        return self.output_dir / "prepared-handoff-coverage.json"

    @property
    def is_representative_submission(self) -> bool:
        return self.representative_canary

    @property
    def representative_workload_profile_id(self) -> str | None:
        profile = self.representative_workload_profile
        if profile is None:
            return None
        if not isinstance(profile, VLLMRepresentativeWorkloadProfile):
            raise TypeError("representative_workload_profile was not normalized")
        return profile.profile_id

    @property
    def requires_pinned_revisions(self) -> bool:
        return self.is_representative_submission or self.benchmark_evidence_policy in {
            "canary",
            "publication",
        }

    @property
    def resolved_benchmark_suite_id(self) -> str:
        return self.benchmark_suite_id or self.benchmark_id

    @property
    def prepared_handoff_generation_path(self) -> Path:
        return self.output_dir / "prepared-handoff-generation.json"

    @property
    def publication_latency_schedule_materialized_path(self) -> Path:
        return self.local_dir / "publication-latency-schedule.json"

    @property
    def publication_handoff_staging_attestation_copy_path(self) -> Path:
        return self.output_dir / PUBLICATION_HANDOFF_STAGING_ATTESTATION_FILENAME

    @property
    def uses_publication_latency_schedule(self) -> bool:
        return (
            self.publication_latency_schedule_record is not None
            or self.publication_latency_schedule_path is not None
        )

    @property
    def stages_publication_handoffs(self) -> bool:
        return (
            self.publication_handoff_generation_output_root is not None
            or self.publication_handoff_bundle_manifest_path is not None
        )

    @property
    def uses_prepared_datasets(self) -> bool:
        return bool(self.dataset_specs)

    @property
    def runs_document_kv_cache_arm(self) -> bool:
        if self.benchmark_arm_specs:
            return any(
                _arm_spec_requires_cachet_handoff(spec)
                for spec in self.benchmark_arm_specs
            )
        return not self.benchmark_arms or CACHE_REUSE_ARM in self.benchmark_arms

    @property
    def requires_prepared_handoff_metadata(self) -> bool:
        # Multi (hybrid) mode always runs the multi-turn probe whose turn 1 injects a
        # Cachet document handoff, so prepared multi runs need loadable handoff
        # metadata even when the client benchmark arms are baseline-only.
        return self.uses_prepared_datasets and (
            self.runs_document_kv_cache_arm
            or self.kv_connector_mode == MULTI_KV_CONNECTOR_MODE
        )


def _resolved_representative_vllm_provenance(
    config: VLLMSmokeBenchmarkConfig,
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    runtime_kv_dtype = config.kv_cache_dtype or config.model_dtype
    pre_rope = (
        config.handoff_generation is not None
        and config.handoff_generation.cache_method
        == CacheGenerationMethod.VANILLA_PREFILL.value
    )
    layout = layout_for_model(
        config.model_id,
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
    wheel_sha256 = _verified_document_kv_package_wheel_sha256()
    package_revisions: dict[str, str] = {
        package: version
        for package, version in (
            pin.split("==", 1) for pin in REPRESENTATIVE_VLLM_PACKAGE_PINS
        )
    }
    package_revisions["cachet-kv"] = f"wheel-sha256:{wheel_sha256}"
    record = dict(provenance)
    supplied_package_revisions = dict(record.pop("package_revisions", {}))
    package_conflicts = {
        package
        for package, revision in supplied_package_revisions.items()
        if package in package_revisions and package_revisions[package] != revision
    }
    if package_conflicts:
        raise ValueError(
            "benchmark_manifest_provenance.package_revisions conflicts with "
            "resolved vLLM package settings: " + ", ".join(sorted(package_conflicts))
        )
    package_revisions.update(supplied_package_revisions)
    expected: dict[str, Any] = {
        "canonical_model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_id": config.model_id,
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
    expected.update(representative_vllm_environment_provenance(config.hardware_target))
    resolved_rope = resolved_layout_rope_provenance(layout)
    expected.update(resolved_rope)
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
            "benchmark_manifest_provenance conflicts with resolved vLLM settings: "
            + ", ".join(sorted(conflicts))
        )
    record.update(expected)
    return validated_benchmark_manifest_provenance(record)


def _verified_document_kv_package_wheel_sha256() -> str:
    digest = os.environ.get(DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV)
    if digest is None:
        raise ValueError(
            "representative canary requires the verified Cachet wheel SHA-256 in "
            f"{DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV}"
        )
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(
            f"{DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV} must be a lowercase SHA-256 digest"
        )
    return digest


def _validate_vllm_representative_workload(
    config: VLLMSmokeBenchmarkConfig,
) -> None:
    profile = config.representative_workload_profile
    if not isinstance(profile, VLLMRepresentativeWorkloadProfile):
        raise ValueError(
            "representative vLLM submission requires a typed workload profile"
        )
    input_tokens_target = config.benchmark_manifest_provenance.get(
        "input_tokens_target"
    )
    mismatches: list[str] = []
    if input_tokens_target != profile.input_tokens_target:
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
    if config.force_max_tokens != profile.force_max_tokens:
        mismatches.append("force_max_tokens")
    if config.prefix_cache_salt_mode != profile.prefix_cache_salt_mode:
        mismatches.append("prefix_cache_salt_mode")
    if config.prewarm_cache_prefix != profile.prewarm_cache_prefix:
        mismatches.append("prewarm_cache_prefix")
    if config.cache_runtime_prompt != profile.cache_runtime_prompt:
        mismatches.append("cache_runtime_prompt")
    if config.payload_cache_max_bytes != profile.payload_cache_max_bytes:
        mismatches.append("payload_cache_max_bytes")
    if config.kv_connector_mode != profile.kv_connector_mode:
        mismatches.append("kv_connector_mode")
    if config.benchmark_evidence_policy != profile.benchmark_evidence_policy:
        mismatches.append("benchmark_evidence_policy")
    if len(config.benchmark_arm_specs) != 1 or not _is_fixed_representative_arm_spec(
        config.benchmark_arm_specs[0]
    ):
        mismatches.append("benchmark_arm_specs")
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
            "match config: " + ", ".join(mismatches)
        )


def _is_fixed_representative_arm_spec(value: Mapping[str, Any]) -> bool:
    record = benchmark_json_mapping_to_record(value)
    if "offline_costs" in record:
        offline_costs = record.pop("offline_costs")
        if not isinstance(offline_costs, Mapping) or set(offline_costs) != {
            "artifact_generation_seconds",
            "artifact_bytes",
        }:
            return False
        generation_seconds = offline_costs["artifact_generation_seconds"]
        artifact_bytes = offline_costs["artifact_bytes"]
        if (
            isinstance(generation_seconds, bool)
            or not isinstance(generation_seconds, (int, float))
            or not math.isfinite(generation_seconds)
            or generation_seconds < 0
            or type(artifact_bytes) is not int
            or artifact_bytes < 0
        ):
            return False
    return any(
        record == benchmark_json_mapping_to_record(run.arm_spec)
        for run in representative_canary_matrix().runs
    )


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


def _non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value.strip()


def _normalized_json_object(
    value: Mapping[str, Any],
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = {
        key: _normalized_json_value(item, f"{field_name}.{key}")
        for key, item in value.items()
        if isinstance(key, str) and key
    }
    if len(normalized) != len(value):
        raise ValueError(f"{field_name} keys must be non-empty strings")
    return normalized


def _normalized_json_value(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)) or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be JSON-compatible")
        return value
    if isinstance(value, Mapping):
        return _normalized_json_object(value, field_name)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return [
            _normalized_json_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{field_name} must be JSON-compatible")


def _normalized_cluster_path(value: str | Path, field_name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{field_name} must be a non-empty path")
    return Path(_cluster_file_path(str(value)))


def _validated_sha256_digest(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def run_vllm_smoke_benchmark(config: VLLMSmokeBenchmarkConfig) -> None:
    """Create an isolated vLLM env, start Qwen3, and run the V1 smoke suite."""

    if config.kv_connector_mode == LMCACHE_KV_CONNECTOR_MODE:
        run_lmcache_cold_benchmark(config)
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.local_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(config.hf_cache_dir)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # Propagate system-prompt placement to prompt-building subprocesses (handoff
    # generation, client benchmark runner, budget probe) via inherited os.environ.
    os.environ[CACHET_BENCHMARK_SYSTEM_PROMPT_POSITION_ENV] = (
        config.system_prompt_position
    )

    is_multi = config.kv_connector_mode == MULTI_KV_CONNECTOR_MODE
    metadata = build_metadata(config)
    metadata["kv_connector_mode"] = config.kv_connector_mode
    write_json(config.metadata_path, metadata)

    create_venv(config.venv_dir)
    if config.native_runtime_v2 is None:
        install_vllm(config.venv_python)
        install_document_kv_package(
            config.venv_python, document_kv_package_install_spec(config)
        )
        metadata["vllm_runtime_lock_verification"] = (
            verify_vllm_runtime_lock_installation(config.venv_python)
        )
    else:
        native_runtime_attestation = install_native_v2_runtime(config)
        metadata["native_runtime_v2_attestation"] = native_runtime_attestation
        metadata["vllm_runtime_lock_verification"] = native_runtime_attestation
    metadata["vllm_runtime_lock_verification_scope"] = (
        "base-runtime-before-unlocked-lmcache" if is_multi else "final-runtime"
    )
    metadata["strict_runtime_closure"] = not is_multi
    if is_multi:
        # Hybrid mode runs MultiConnector[Cachet, LMCache]; LMCache must be present
        # in the vLLM venv and configured with its disk tier for the second connector.
        lmcache_version = install_lmcache(config.venv_python, config.lmcache_version)
        lmcache_config_path = write_lmcache_config(config)
        os.environ["LMCACHE_CONFIG_FILE"] = str(lmcache_config_path)
        metadata["lmcache_version_installed"] = lmcache_version
        metadata["lmcache_config_path"] = str(lmcache_config_path)
        metadata["lmcache_config"] = json.loads(
            lmcache_config_path.read_text(encoding="utf-8")
        )
        run(
            [str(config.venv_python), "-m", "pip", "check"],
            env=_pip_subprocess_environment(),
        )
    metadata["vllm_runtime_patch_closure"] = verify_vllm_runtime_patch_closure(config)
    metadata.update(installed_versions(config.venv_python))
    metadata["installed_package_freeze"] = installed_package_freeze(config.venv_python)
    metadata["cuda_wheel_env_paths"] = cuda_wheel_env_paths(config)
    write_json(config.metadata_path, metadata)
    # Multi mode must import both vLLM and LMCache cleanly (ABI check) before boot.
    probe = probe_lmcache_import if is_multi else probe_vllm_import
    probe(
        config.venv_python,
        config.import_probe_path,
        timeout_seconds=config.import_probe_timeout_seconds,
        env=server_env(config),
    )

    dataset_paths = benchmark_dataset_paths(config)
    dataset_paths = prepare_publication_latency_inputs(config, dataset_paths)
    dataset_paths = prepare_generated_benchmark_handoffs(config, dataset_paths)
    config = _config_with_generated_handoff_offline_costs(config)
    metadata["benchmark_arm_specs"] = [
        benchmark_json_mapping_to_record(spec) for spec in config.benchmark_arm_specs
    ]
    validate_prepared_benchmark_handoffs(config, dataset_paths)
    validate_prompt_token_budget(config, dataset_paths)
    metadata["vllm_server_local_log"] = str(config.server_log_path)
    metadata["vllm_server_log"] = str(config.server_log_copy_path)
    metadata["document_kv_connector_telemetry_local_path"] = str(
        config.connector_telemetry_path
    )
    metadata["document_kv_connector_telemetry_path"] = str(
        config.connector_telemetry_copy_path
    )
    metadata["runtime_telemetry_local_path"] = str(config.runtime_telemetry_path)
    metadata["runtime_telemetry_path"] = str(config.runtime_telemetry_copy_path)
    metadata["prompt_token_budget_path"] = str(config.prompt_token_budget_path)
    if config.requires_prepared_handoff_metadata:
        metadata["prepared_handoff_coverage_path"] = str(
            config.prepared_handoff_coverage_path
        )
    if config.handoff_generation is not None:
        metadata["prepared_handoff_generation_path"] = str(
            config.prepared_handoff_generation_path
        )
    if config.stages_publication_handoffs:
        metadata["publication_handoff_staging_attestation_path"] = str(
            config.publication_handoff_staging_attestation_copy_path
        )
    if config.prewarm_cache_prefix:
        metadata["prewarm_cache_prefix_path"] = str(config.prewarm_cache_prefix_path)
    if config.prewarm_payload_cache:
        metadata["prewarm_payload_cache_path"] = str(config.prewarm_payload_cache_path)
        metadata["payload_cache_attestation_path"] = str(
            config.payload_cache_attestation_path
        )
    write_json(config.metadata_path, metadata)

    server = start_vllm_server(config, config.venv_python, config.server_log_path)
    runtime_telemetry = RuntimeTelemetrySampler(
        config.runtime_telemetry_path,
        process_pid=getattr(server, "pid", None),
        interval_seconds=config.runtime_telemetry_interval_seconds,
    ).start()
    try:
        wait_for_server(
            server,
            config.server_log_path,
            config,
            timeout_seconds=config.server_start_timeout_seconds,
        )
        copy_file_if_exists(config.server_log_path, config.server_log_copy_path)
        prewarm_cache_prefixes(config, dataset_paths)
        prime_payload_cache(config, dataset_paths)
        run_benchmark_runner(config, dataset_paths)
        attest_payload_cache_measurements(config)
        if is_multi:
            # Hybrid handoff: measure per-turn TTFT (turn-1 docs->Cachet, follow-ups->LMCache).
            run_multi_turn_hybrid_latency(config, dataset_paths)
    finally:
        terminate_process(server)
        runtime_telemetry.stop()
        if config.benchmark_output_path.exists():
            bind_runtime_resource_evidence_record_file(
                config.benchmark_output_path,
                config.runtime_telemetry_path,
            )
        copy_file_if_exists(config.server_log_path, config.server_log_copy_path)
        copy_file_if_exists(
            config.connector_telemetry_path, config.connector_telemetry_copy_path
        )
        copy_file_if_exists(
            config.runtime_telemetry_path, config.runtime_telemetry_copy_path
        )


def build_metadata(config: VLLMSmokeBenchmarkConfig) -> dict[str, object]:
    return {
        "benchmark_id": config.benchmark_id,
        "benchmark_suite_id": config.resolved_benchmark_suite_id,
        "benchmark_runtime_id": config.benchmark_runtime_id,
        "hf_model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "served_model_name": SERVED_MODEL_NAME,
        "model_dtype": config.model_dtype,
        "model_quantization": config.model_quantization,
        "kv_cache_dtype": config.kv_cache_dtype,
        "attention_backend": config.attention_backend,
        "vllm_version_requested": VLLM_VERSION,
        "vllm_package_version_requested": VLLM_PACKAGE_VERSION,
        "vllm_wheel_filename": VLLM_WHEEL_FILENAME,
        "vllm_wheel_url": VLLM_WHEEL_URL,
        "vllm_wheel_sha256": VLLM_WHEEL_SHA256,
        "vllm_patched_wheel_uri": (
            config.native_runtime_v2.patched_vllm_wheel_uri
            if config.native_runtime_v2 is not None
            else os.environ.get(VLLM_PATCHED_WHEEL_URI_ENV)
        ),
        "vllm_patched_wheel_sha256": (
            config.native_runtime_v2.patched_vllm_wheel_sha256
            if config.native_runtime_v2 is not None
            else os.environ.get(VLLM_PATCHED_WHEEL_SHA256_ENV)
        ),
        "vllm_runtime_lock": (
            {
                "uri": config.native_runtime_v2.runtime_lock_uri,
                "sha256": config.native_runtime_v2.runtime_lock_sha256,
                "platform": "CPython 3.11 / Linux x86_64 / glibc 2.35",
                "runtime_contract": "native-v2",
            }
            if config.native_runtime_v2 is not None
            else {
                "filename": VLLM_RUNTIME_LOCK_FILENAME,
                "sha256": VLLM_RUNTIME_LOCK_SHA256,
                "locked_distribution_count": VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
                "platform": "CPython 3.11 / Linux x86_64 / glibc 2.35",
            }
        ),
        "native_runtime_v2": (
            None
            if config.native_runtime_v2 is None
            else config.native_runtime_v2.to_record()
        ),
        "virtualenv_bootstrap": {
            "version": VIRTUALENV_BOOTSTRAP_VERSION,
            "filename": VIRTUALENV_BOOTSTRAP_FILENAME,
            "url": VIRTUALENV_BOOTSTRAP_URL,
            "sha256": VIRTUALENV_BOOTSTRAP_SHA256,
        },
        "server_bind_host": config.server_host,
        "server_client_host": config.client_host,
        "server_base_url": config.server_base_url,
        "hf_home": str(config.hf_cache_dir),
        "vllm_python": str(config.venv_python),
        "document_kv_connector_telemetry_local_path": str(
            config.connector_telemetry_path
        ),
        "document_kv_connector_telemetry_path": str(
            config.connector_telemetry_copy_path
        ),
        "runtime_telemetry_local_path": str(config.runtime_telemetry_path),
        "runtime_telemetry_path": str(config.runtime_telemetry_copy_path),
        "runtime_telemetry_interval_seconds": config.runtime_telemetry_interval_seconds,
        "dependency_constraints": dependency_constraints(),
        "dependency_index_urls": (
            []
            if config.native_runtime_v2 is not None
            else list(VLLM_PACKAGE_INDEX_URLS)
        ),
        "vllm_cuda_variant": VLLM_CUDA_VARIANT,
        "vllm_cuda_requirements_sha256": VLLM_CUDA_REQUIREMENTS_SHA256,
        "vllm_dockerfile_sha256": VLLM_DOCKERFILE_SHA256,
        "dataset_source": "prepared" if config.dataset_specs else "smoke",
        "dataset_specs": list(config.dataset_specs),
        "allow_dataset_subset": config.allow_dataset_subset,
        "prewarm_cache_prefix": config.prewarm_cache_prefix,
        "prewarm_payload_cache": config.prewarm_payload_cache,
        "payload_cache_max_bytes": config.payload_cache_max_bytes,
        "payload_cache_prime_target_count": (config.payload_cache_prime_target_count),
        "cache_runtime_prompt": config.cache_runtime_prompt,
        "cache_measurement_protocol": cache_measurement_protocol(config),
        "cache_prompt_text_mode": "runtime"
        if config.cache_runtime_prompt
        else "logical",
        "max_tokens": config.max_tokens,
        "force_max_tokens": config.force_max_tokens,
        "temperature": config.temperature,
        "generation_seed": config.generation_seed,
        "latency_decode_protocol": (
            {
                "max_tokens": config.max_tokens,
                "ignore_eos": True,
                "description": "force exactly max_tokens decode tokens for TTC latency measurement",
            }
            if config.force_max_tokens
            else {
                "max_tokens": config.max_tokens,
                "ignore_eos": False,
                "description": "natural stop quality/latency smoke measurement",
            }
        ),
        "prefix_cache_isolation": (
            {
                "baseline_cache_salt": BASELINE_PREFIX_CACHE_SALT,
                "cache_cache_salt": CACHE_PREFIX_CACHE_SALT,
                "cache_salt_mode": config.prefix_cache_salt_mode,
            }
            if config.uses_prepared_datasets
            else None
        ),
        "requires_kv_transfer_params": config.requires_prepared_handoff_metadata,
        "generates_prepared_handoffs": config.handoff_generation is not None,
        "benchmark_handoff_generation": (
            None
            if config.handoff_generation is None
            else config.handoff_generation.to_metadata()
        ),
        "runtime_identity": (
            None
            if config.runtime_identity is None
            else config.runtime_identity.to_record()
        ),
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "data_parallel_size": config.data_parallel_size,
        "benchmark_repeats": config.benchmark_repeats,
        "request_parallelism": config.request_parallelism,
        "benchmark_interleave_examples": config.benchmark_interleave_examples,
        "benchmark_system_prompt_position": config.system_prompt_position,
        "benchmark_arms": list(config.benchmark_arms),
        "benchmark_arm_specs": [
            benchmark_json_mapping_to_record(spec)
            for spec in config.benchmark_arm_specs
        ],
        "benchmark_evidence_policy": config.benchmark_evidence_policy,
        "publication_latency_schedule_source": (
            "inline_record"
            if config.publication_latency_schedule_record is not None
            else str(config.publication_latency_schedule_path)
            if config.publication_latency_schedule_path is not None
            else None
        ),
        "publication_latency_expected_input_bundle_sha256": (
            config.publication_latency_expected_input_bundle_sha256
        ),
        "publication_handoff_generation_output_root": (
            None
            if config.publication_handoff_generation_output_root is None
            else str(config.publication_handoff_generation_output_root)
        ),
        "publication_handoff_generation_execution_file_sha256": (
            config.publication_handoff_generation_execution_file_sha256
        ),
        "publication_handoff_generation_execution_closed_record_sha256": (
            config.publication_handoff_generation_execution_closed_record_sha256
        ),
        "publication_handoff_bundle_manifest_path": (
            None
            if config.publication_handoff_bundle_manifest_path is None
            else str(config.publication_handoff_bundle_manifest_path)
        ),
        "publication_handoff_bundle_source_root": (
            None
            if config.publication_handoff_bundle_source_root is None
            else str(config.publication_handoff_bundle_source_root)
        ),
        "publication_handoff_bundle_manifest_file_sha256": (
            config.publication_handoff_bundle_manifest_file_sha256
        ),
        "publication_handoff_bundle_manifest_closed_record_sha256": (
            config.publication_handoff_bundle_manifest_closed_record_sha256
        ),
        "publication_handoff_local_nvme_dir": (
            None
            if config.publication_handoff_local_nvme_dir is None
            else str(config.publication_handoff_local_nvme_dir)
        ),
        "publication_handoff_stage_kind": config.publication_handoff_stage_kind,
        "publication_handoff_staging_attestation_path": (
            str(config.publication_handoff_staging_attestation_copy_path)
            if config.stages_publication_handoffs
            else None
        ),
        "representative_canary": config.is_representative_submission,
        "representative_workload_profile": config.representative_workload_profile_id,
        "benchmark_manifest_provenance": benchmark_json_mapping_to_record(
            config.benchmark_manifest_provenance
        ),
        "hardware_target": config.hardware_target,
        "document_kv_package_install_spec": document_kv_package_install_spec(config),
        "dependency_override_constraints": dependency_override_constraints(),
        "vllm_server_env_overrides": vllm_server_env_overrides(),
        # Record the transfer config the server is actually launched with so
        # provenance-driven reruns reproduce the same connector. This tracks
        # kv_connector_mode (cachet/lmcache/multi); for cachet it is identical to
        # document_kv_transfer_config_for_smoke(config).
        "vllm_kv_transfer_config": json.loads(kv_transfer_config_json(config)),
    }


def cache_measurement_protocol(config: VLLMSmokeBenchmarkConfig) -> str:
    if not config.uses_prepared_datasets:
        return "stock_smoke"
    if config.prewarm_cache_prefix:
        return "warm_prefix_cache"
    if config.prewarm_payload_cache:
        return "ram_payload_cache_to_gpu_hydrate"
    if config.prefix_cache_salt_mode == "per_request":
        return "cold_disk_to_gpu_hydrate"
    return "static_prefix_cache_mixed_first_cold_then_warm"


def build_vllm_native_provider_probe_record(
    transfer_config: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Instantiate the configured vLLM connector and verify native provider wiring."""

    config = (
        document_kv_transfer_config() if transfer_config is None else transfer_config
    )
    if not isinstance(config, Mapping):
        raise TypeError("vLLM KV transfer config must be a mapping")
    extra_config = config.get("kv_connector_extra_config")
    if not isinstance(extra_config, Mapping):
        raise TypeError(
            "vLLM KV transfer config kv_connector_extra_config must be a mapping"
        )
    provider_factory = extra_config.get(DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY)
    if not isinstance(provider_factory, str) or not provider_factory.strip():
        raise ValueError(
            f"{DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY} must be a non-empty module:attribute string"
        )
    if extra_config.get("document_kv.requires_native_runtime") is not True:
        raise ValueError("document_kv.requires_native_runtime must be true")

    connector = DocumentKVConnector(
        vllm_config=SimpleNamespace(kv_transfer_config=config)
    )
    provider = connector.provider
    if isinstance(provider, NoOpDocumentKVProvider):
        raise ValueError("vLLM smoke cannot run with NoOpDocumentKVProvider")
    if getattr(provider, "document_kv_native_provider", False) is not True:
        raise TypeError("vLLM smoke requires a native document KV provider")

    provider_type = f"{type(provider).__module__}.{type(provider).__qualname__}"
    connector_type = f"{type(connector).__module__}.{type(connector).__qualname__}"
    return {
        "document_kv_native_provider_ok": True,
        "document_kv_provider_factory": provider_factory,
        "document_kv_provider_type": provider_type,
        "document_kv_connector_type": connector_type,
        "document_kv_requires_native_runtime": True,
    }


def document_kv_transfer_config_for_smoke(
    config: VLLMSmokeBenchmarkConfig,
) -> dict[str, Any]:
    return document_kv_transfer_config(
        payload_cache_max_bytes=config.payload_cache_max_bytes or None,
        telemetry_jsonl=str(config.connector_telemetry_path),
        runtime_identity=config.runtime_identity,
        require_runtime_handshake=(
            True if config.runtime_identity is not None else None
        ),
    )


def dependency_constraints() -> list[str]:
    return list(VLLM_SERVING_ENVIRONMENT_PROFILE.dependency_constraints)


def dependency_index_args() -> list[str]:
    """Return the official indexes needed by the pinned CUDA 12.9 closure."""

    return [
        argument
        for index_url in VLLM_PACKAGE_INDEX_URLS
        for argument in ("--extra-index-url", index_url)
    ]


def vllm_dependency_install_requirements() -> list[str]:
    """Substitute the approved patched wheel for the package-name pin."""

    constraints = dependency_constraints()
    expected_package_pin = f"vllm=={VLLM_PACKAGE_VERSION}"
    if not constraints or constraints[0] != expected_package_pin:
        raise RuntimeError(
            "vLLM serving profile must start with the exact cu129 package identity"
        )
    requirements = list(vllm_runtime_install_requirements())
    if requirements[1:] != constraints[1:]:
        raise RuntimeError(
            "vLLM runtime requirements diverged from the serving profile"
        )
    return requirements


def dependency_override_constraints() -> list[str]:
    # OpenCV is part of the hash-locked runtime closure; no post-install
    # replacement is permitted.
    return []


def _cluster_file_path(uri: str) -> str:
    if uri.startswith("dbfs:/Volumes/"):
        return uri.removeprefix("dbfs:")
    if uri.startswith("dbfs:/"):
        return "/dbfs/" + uri.removeprefix("dbfs:/").lstrip("/")
    return uri


def _source_checkout_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (
            parent / "src" / "document_kv_cache"
        ).exists():
            return parent
    return None


def document_kv_package_install_spec(config: VLLMSmokeBenchmarkConfig) -> str:
    """Return the package spec that must be installed into the vLLM venv."""

    if config.native_runtime_v2 is not None:
        return str(config.native_runtime_v2.local_path("package_wheel"))
    if config.package_install_spec is not None:
        return _cluster_file_path(config.package_install_spec)
    env_value = os.environ.get(DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV)
    if env_value is not None:
        if not env_value.strip():
            raise ValueError(
                f"{DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV} must be non-empty when set"
            )
        return _cluster_file_path(env_value)
    source_root = _source_checkout_root()
    if source_root is not None:
        return str(source_root)
    raise RuntimeError(
        "vLLM smoke benchmark requires a Cachet package install spec for the isolated vLLM environment; "
        f"set {DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV} or pass --package-install-spec"
    )


def installed_versions(python_executable: Path) -> dict[str, str]:
    return {
        "vllm_version_installed": installed_package_version(python_executable, "vllm"),
        "document_kv_cache_version_installed": installed_package_version(
            python_executable, "cachet-kv"
        ),
        "transformers_version_installed": installed_package_version(
            python_executable, "transformers"
        ),
        "torch_version_installed": installed_package_version(
            python_executable, "torch"
        ),
        "opencv_python_headless_version_installed": installed_package_version(
            python_executable,
            "opencv-python-headless",
        ),
    }


def installed_package_freeze(python_executable: Path) -> list[str]:
    """Return the complete normalized post-install distribution snapshot."""

    completed = subprocess.run(
        [str(python_executable), "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
        env=_pip_subprocess_environment(),
    )
    return sorted(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()),
        key=str.casefold,
    )


def verify_vllm_runtime_lock_installation(
    python_executable: Path,
) -> dict[str, Any]:
    """Run pip consistency and the packaged lock/direct-URL verifier."""

    pip_environment = _pip_subprocess_environment()
    subprocess.run(
        [str(python_executable), "-m", "pip", "check"],
        check=True,
        env=pip_environment,
    )
    code = (
        "import json,sys; "
        "from document_kv_cache.serving_env import "
        "verify_installed_vllm_runtime_lock; "
        "print(json.dumps(verify_installed_vllm_runtime_lock(sys.argv[1]), "
        "sort_keys=True))"
    )
    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            code,
            patched_vllm_wheel_install_spec(),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=pip_environment,
    )
    record = json.loads(completed.stdout)
    if not isinstance(record, dict) or record.get("ok") is not True:
        raise RuntimeError("vLLM runtime lock verifier did not return an ok record")
    return record


def run(
    argv: list[str],
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, check=True, env=None if env is None else dict(env))


def validate_prompt_token_budget(
    config: VLLMSmokeBenchmarkConfig, dataset_paths: dict[str, Path]
) -> None:
    rows = build_prompt_token_budget_rows(config, dataset_paths)
    write_prompt_token_budget_jsonl(config.prompt_token_budget_input_path, rows)
    expected_prompt_tokens = config.benchmark_manifest_provenance.get(
        "input_tokens_target"
    )
    tokenizer_id = config.benchmark_manifest_provenance.get(
        "tokenizer_id",
        config.model_id,
    )
    tokenizer_revision = config.benchmark_manifest_provenance.get(
        "tokenizer_revision",
        config.tokenizer_revision,
    )
    record = run_prompt_token_budget_probe(
        config.venv_python,
        config.prompt_token_budget_input_path,
        model_id=config.model_id,
        model_revision=config.model_revision,
        tokenizer_id=str(tokenizer_id),
        tokenizer_revision=(
            None if tokenizer_revision is None else str(tokenizer_revision)
        ),
        add_special_tokens=PROMPT_TOKEN_PROBE_ADD_SPECIAL_TOKENS,
        expected_prompt_tokens=(
            None if expected_prompt_tokens is None else int(expected_prompt_tokens)
        ),
        max_model_len=config.max_model_len,
        max_tokens=config.max_tokens,
        timeout_seconds=config.import_probe_timeout_seconds,
        env=server_env(config),
    )
    write_json(config.prompt_token_budget_path, record)
    if record.get("ok") is False:
        raise RuntimeError(
            f"Prompt token budget probe failed: {record.get('error') or record.get('error_type')}. "
            f"See {config.prompt_token_budget_path}."
        )
    token_count_mismatches = record.get("token_count_mismatches")
    if isinstance(token_count_mismatches, list) and token_count_mismatches:
        first = token_count_mismatches[0]
        raise ValueError(
            "Prepared vLLM benchmark prompts do not match the exact logical "
            f"input token target; {len(token_count_mismatches)} prompt(s) differ, "
            f"first={first!r}. See {config.prompt_token_budget_path}."
        )
    over_budget = record.get("over_budget")
    if isinstance(over_budget, list) and over_budget:
        first = over_budget[0]
        raise ValueError(
            "Prepared vLLM benchmark prompts exceed the configured context budget; "
            f"{len(over_budget)} prompt(s) are over budget, first={first!r}. "
            f"See {config.prompt_token_budget_path}."
        )


def build_prompt_token_budget_rows(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: dict[str, Path],
) -> tuple[dict[str, str], ...]:
    suite = load_v1_jsonl_suite(
        suite_id=config.benchmark_id,
        paths=dataset_paths,
        model_id=SERVED_MODEL_NAME,
        hardware_target=config.hardware_target,
    )
    if config.is_representative_submission and not any(
        len(example.documents) >= 2 for example in suite.examples
    ):
        raise ValueError(
            "representative vLLM workload requires at least one prepared "
            "multi-document example"
        )
    rows = []
    for example in suite.examples:
        prompt = build_prompt_parts(example).prefill_prompt
        rows.append(
            {
                "dataset": example.dataset,
                "example_id": example.example_id,
                "prompt": prompt,
            }
        )
    return tuple(rows)


def validate_prepared_benchmark_handoffs(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: dict[str, Path],
) -> dict[str, object] | None:
    """Require prepared benchmark rows to carry loadable Cachet handoff params when Cachet runs."""

    if not config.requires_prepared_handoff_metadata:
        return None
    record = prepared_benchmark_handoff_coverage_record(config, dataset_paths)
    write_json(config.prepared_handoff_coverage_path, record)
    if record.get("ok") is not True:
        missing = record.get("missing_kv_transfer_params")
        invalid = record.get("invalid_handoff_references")
        raise ValueError(
            "Prepared vLLM benchmark datasets must be enriched with Cachet per-arm or legacy kv_transfer_params "
            "that reference readable vLLM handoffs; "
            f"missing rows: {missing!r}; invalid handoff references: {invalid!r}. "
            f"See {config.prepared_handoff_coverage_path}."
        )
    return record


def prepared_benchmark_handoff_coverage_record(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: dict[str, Path],
) -> dict[str, object]:
    suite = load_v1_jsonl_suite(
        suite_id=config.benchmark_id,
        paths=dataset_paths,
        model_id=SERVED_MODEL_NAME,
        hardware_target=config.hardware_target,
    )
    cache_arm_ids = _prepared_cache_arm_ids(config)
    params_by_example = {
        (example.dataset, example.example_id): _prepared_params_by_arm(
            example,
            cache_arm_ids=cache_arm_ids,
        )
        for example in suite.examples
    }
    missing = tuple(
        f"{example.dataset}/{example.example_id}:{arm_id}"
        for example in suite.examples
        for arm_id, params in params_by_example[
            (example.dataset, example.example_id)
        ].items()
        if not params
    )
    invalid = tuple(
        issue
        for example in suite.examples
        for arm_id, params in params_by_example[
            (example.dataset, example.example_id)
        ].items()
        if params
        for issue in (
            _prepared_handoff_reference_issue(
                example,
                params=params,
                arm_id=arm_id,
            ),
        )
        if issue is not None
    )
    incomplete_examples = {(item.split(":", 1)[0]) for item in missing}.union(
        f"{issue['dataset']}/{issue['example_id']}" for issue in invalid
    )
    counts_by_dataset: dict[str, int] = {}
    for example in suite.examples:
        counts_by_dataset[example.dataset] = (
            counts_by_dataset.get(example.dataset, 0) + 1
        )
    issues = []
    if missing:
        issues.append("prepared benchmark rows missing kv_transfer_params")
    if invalid:
        issues.append("prepared benchmark rows reference unloadable Cachet handoffs")
    topology_attestation = _prepared_generation_topology_attestation(config)
    return {
        "ok": not missing and not invalid,
        "required": True,
        "dataset_source": "prepared",
        "datasets": counts_by_dataset,
        "cache_arm_ids": list(cache_arm_ids),
        "examples": len(suite.examples),
        "examples_with_kv_transfer_params": len(suite.examples)
        - len({item.split(":", 1)[0] for item in missing}),
        "examples_with_loadable_handoff_references": (
            len(suite.examples) - len(incomplete_examples)
        ),
        "missing_kv_transfer_params": list(missing),
        "invalid_handoff_references": list(invalid),
        "issues": issues,
        "handoff_topology_attestation": topology_attestation,
    }


def _prepared_generation_topology_attestation(
    config: VLLMSmokeBenchmarkConfig,
) -> dict[str, Any] | None:
    if config.handoff_generation is None:
        return None
    try:
        generation_record = json.loads(
            config.prepared_handoff_generation_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        if config.requires_pinned_revisions:
            raise ValueError(
                "representative handoff coverage requires its generation summary"
            ) from exc
        return None
    topology = (
        generation_record.get("handoff_topology_attestation")
        if isinstance(generation_record, Mapping)
        else None
    )
    if topology is None:
        if config.requires_pinned_revisions:
            raise ValueError(
                "representative handoff generation summary is missing topology attestation"
            )
        return None
    if not isinstance(topology, Mapping):
        raise ValueError("handoff topology attestation must be an object")
    return validate_handoff_topology_attestation(topology)


def _prepared_cache_arm_ids(config: VLLMSmokeBenchmarkConfig) -> tuple[str, ...]:
    if config.benchmark_arm_specs:
        arm_ids = tuple(
            str(spec["arm_id"])
            for spec in config.benchmark_arm_specs
            if _arm_spec_requires_cachet_handoff(spec)
        )
    elif config.benchmark_arms:
        arm_ids = (CACHE_REUSE_ARM,) if CACHE_REUSE_ARM in config.benchmark_arms else ()
    else:
        arm_ids = (CACHE_REUSE_ARM,)
    if not arm_ids and config.kv_connector_mode == MULTI_KV_CONNECTOR_MODE:
        return (CACHE_REUSE_ARM,)
    return arm_ids


def _arm_spec_requires_cachet_handoff(spec: Mapping[str, Any]) -> bool:
    explicit = spec.get("requires_cachet_handoff")
    if explicit is not None:
        return explicit is True
    implementation_kind = spec.get("implementation_kind")
    if not implementation_kind:
        implementation_kind = "cachet" if spec.get("uses_cache") is True else "baseline"
    return spec.get("uses_cache") is True and implementation_kind == "cachet"


def _prepared_params_by_arm(
    example: object,
    *,
    cache_arm_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    per_arm = getattr(example, "arm_kv_transfer_params", {})
    if not isinstance(per_arm, Mapping):
        raise TypeError("arm_kv_transfer_params must be a mapping")
    legacy = getattr(example, "kv_transfer_params", {})
    if not isinstance(legacy, Mapping):
        raise TypeError("kv_transfer_params must be a mapping")
    return {
        arm_id: (
            per_arm[arm_id]
            if arm_id in per_arm
            else legacy
            if len(cache_arm_ids) == 1
            else {}
        )
        for arm_id in cache_arm_ids
    }


def _prepared_handoff_reference_issue(
    example: object,
    *,
    params: Mapping[str, Any],
    arm_id: str,
) -> dict[str, object] | None:
    if not isinstance(params, Mapping):
        return _handoff_reference_issue(example, "kv_transfer_params must be a mapping")
    handoff_json: str | None = None
    payload_override = params.get(DOCUMENT_KV_PAYLOAD_URI_PARAM)
    try:
        handoff_record = params.get(DOCUMENT_KV_HANDOFF_RECORD_PARAM)
        if handoff_record is not None:
            record = handoff_record
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"kv_transfer_params.{DOCUMENT_KV_HANDOFF_RECORD_PARAM} must be an object"
                )
            validate_engine_adapter_request_record(
                record,
                expected_backend=ServingBackend.VLLM,
                require_external_payload_uri=payload_override is None,
            )
        else:
            handoff_json_value = params.get(DOCUMENT_KV_HANDOFF_JSON_PARAM)
            if not isinstance(handoff_json_value, str) or not handoff_json_value:
                raise ValueError(
                    f"kv_transfer_params.{DOCUMENT_KV_HANDOFF_JSON_PARAM} must be a non-empty string"
                )
            handoff_json = handoff_json_value
            record = read_engine_adapter_request_json(
                handoff_json,
                expected_backend=ServingBackend.VLLM,
                require_external_payload_uri=payload_override is None,
            )
        request_id = params.get(DOCUMENT_KV_REQUEST_ID_PARAM)
        if record.get("request_id") != request_id:
            raise ValueError(
                f"handoff request_id {record.get('request_id')!r} does not match "
                f"kv_transfer_params.{DOCUMENT_KV_REQUEST_ID_PARAM} {request_id!r}"
            )
        payload_uri = payload_override
        if payload_uri is None:
            payload_source = record.get("payload_source")
            if not isinstance(payload_source, Mapping):
                raise ValueError("handoff payload_source must be an object")
            payload_uri = payload_source.get("uri")
        if not isinstance(payload_uri, str) or not payload_uri:
            raise ValueError("handoff payload URI must be a non-empty string")
        _validate_local_payload_uri(payload_uri)
    except Exception as exc:
        issue = _handoff_reference_issue(
            example,
            str(exc),
            error_type=type(exc).__name__,
            handoff_json=handoff_json,
        )
        issue["arm_id"] = arm_id
        return issue
    return None


def _handoff_reference_issue(
    example: object,
    error: str,
    *,
    error_type: str = "ValueError",
    handoff_json: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "dataset": str(getattr(example, "dataset", "")),
        "example_id": str(getattr(example, "example_id", "")),
        "error_type": error_type,
        "error": error,
    }
    if handoff_json is not None:
        record["handoff_json"] = handoff_json
    return record


def write_prompt_token_budget_jsonl(
    path: Path, rows: tuple[dict[str, str], ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_generated_benchmark_handoffs(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: dict[str, Path],
) -> dict[str, Path]:
    """Generate and attach Cachet handoffs for prepared vLLM benchmark rows."""

    generation = config.handoff_generation
    if generation is None:
        return dataset_paths
    if config.venv_python.exists():
        generated_paths, record = (
            _generate_prepared_benchmark_handoff_inputs_in_subprocess(
                config,
                dataset_paths,
                generation,
            )
        )
        write_json(config.prepared_handoff_generation_path, record)
        return generated_paths
    generation.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        generated_paths, record = _generate_prepared_benchmark_handoff_inputs(
            config, dataset_paths, generation
        )
    finally:
        release_handoff_generation_resources()
    write_json(config.prepared_handoff_generation_path, record)
    return generated_paths


def _generate_prepared_benchmark_handoff_inputs_in_subprocess(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: dict[str, Path],
    generation: VLLMPreparedHandoffGenerationConfig,
) -> tuple[dict[str, Path], dict[str, object]]:
    input_path = config.local_dir / "prepared-handoff-generation-worker-input.json"
    output_path = config.local_dir / "prepared-handoff-generation-worker-output.json"
    payload: dict[str, object] = {
        "benchmark_id": config.benchmark_id,
        "benchmark_suite_id": config.benchmark_suite_id,
        "benchmark_runtime_id": config.benchmark_runtime_id,
        "output_dir": str(config.output_dir),
        "local_root": str(config.local_root),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "model_dtype": config.model_dtype,
        "model_quantization": config.model_quantization,
        "kv_cache_dtype": config.kv_cache_dtype,
        "attention_backend": config.attention_backend,
        "max_tokens": config.max_tokens,
        "force_max_tokens": config.force_max_tokens,
        "max_model_len": config.max_model_len,
        "data_parallel_size": config.data_parallel_size,
        "kv_connector_mode": config.kv_connector_mode,
        "benchmark_repeats": config.benchmark_repeats,
        "request_parallelism": config.request_parallelism,
        "benchmark_arm_specs": [
            benchmark_json_mapping_to_record(spec)
            for spec in config.benchmark_arm_specs
        ],
        "benchmark_evidence_policy": config.benchmark_evidence_policy,
        "representative_canary": config.is_representative_submission,
        "representative_workload_profile": config.representative_workload_profile_id,
        "benchmark_manifest_provenance": benchmark_json_mapping_to_record(
            config.benchmark_manifest_provenance
        ),
        "prewarm_cache_prefix": config.prewarm_cache_prefix,
        "prewarm_payload_cache": config.prewarm_payload_cache,
        "cache_runtime_prompt": config.cache_runtime_prompt,
        "prefix_cache_salt_mode": config.prefix_cache_salt_mode,
        "payload_cache_max_bytes": config.payload_cache_max_bytes,
        "hardware_target": config.hardware_target,
        "system_prompt_position": config.system_prompt_position,
        "allow_dataset_subset": config.allow_dataset_subset,
        "dataset_paths": {
            dataset: str(path) for dataset, path in dataset_paths.items()
        },
        "handoff_generation": generation.to_metadata(),
    }
    write_json(input_path, payload)
    code = """
import json
import sys
from pathlib import Path

from document_kv_cache.vllm_smoke import (
    VLLMPreparedHandoffGenerationConfig,
    VLLMSmokeBenchmarkConfig,
    _generate_prepared_benchmark_handoff_inputs,
    release_handoff_generation_resources,
    write_json,
)

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
payload = json.loads(input_path.read_text(encoding="utf-8"))
generation_payload = payload["handoff_generation"]
generation = VLLMPreparedHandoffGenerationConfig(
    generator_factory=generation_payload["generator_factory"],
    output_dir=Path(generation_payload["output_dir"]),
    dtype=generation_payload["dtype"],
    align_bytes=int(generation_payload["align_bytes"]),
    timeout_seconds=float(generation_payload["timeout_seconds"]),
    limit=generation_payload.get("limit"),
    benchmark_handoff_segment_per_document=bool(generation_payload.get("segment_per_document", False)),
    cache_method=generation_payload.get("cache_method"),
    require_artifact_contract=bool(generation_payload.get("require_artifact_contract", True)),
)
config = VLLMSmokeBenchmarkConfig(
    benchmark_id=payload["benchmark_id"],
    benchmark_suite_id=payload.get("benchmark_suite_id"),
    benchmark_runtime_id=payload.get("benchmark_runtime_id"),
    output_dir=Path(payload["output_dir"]),
    local_root=Path(payload["local_root"]),
    model_id=payload["model_id"],
    model_revision=payload.get("model_revision"),
    tokenizer_revision=payload.get("tokenizer_revision"),
    model_dtype=payload["model_dtype"],
    model_quantization=payload.get("model_quantization"),
    kv_cache_dtype=payload.get("kv_cache_dtype"),
    attention_backend=payload.get("attention_backend"),
    max_tokens=int(payload["max_tokens"]),
    force_max_tokens=bool(payload.get("force_max_tokens", False)),
    max_model_len=int(payload["max_model_len"]),
    data_parallel_size=int(payload.get("data_parallel_size", 1)),
    kv_connector_mode=payload.get("kv_connector_mode", "cachet"),
    benchmark_repeats=int(payload.get("benchmark_repeats", 1)),
    request_parallelism=int(payload.get("request_parallelism", 1)),
    benchmark_arm_specs=tuple(payload.get("benchmark_arm_specs", ())),
    benchmark_evidence_policy=payload.get("benchmark_evidence_policy"),
    representative_canary=bool(payload.get("representative_canary", False)),
    representative_workload_profile=payload.get("representative_workload_profile"),
    benchmark_manifest_provenance=payload.get("benchmark_manifest_provenance", {}),
    prewarm_cache_prefix=bool(payload.get("prewarm_cache_prefix", False)),
    prewarm_payload_cache=bool(payload.get("prewarm_payload_cache", False)),
    cache_runtime_prompt=bool(payload.get("cache_runtime_prompt", False)),
    prefix_cache_salt_mode=payload.get("prefix_cache_salt_mode", "per_request"),
    payload_cache_max_bytes=int(payload.get("payload_cache_max_bytes", 0)),
    hardware_target=payload.get("hardware_target", "aws-g6-l4"),
    system_prompt_position=payload.get("system_prompt_position", "start"),
    dataset_specs=tuple(
        f"{dataset}={path}" for dataset, path in payload["dataset_paths"].items()
    ),
    allow_dataset_subset=bool(payload.get("allow_dataset_subset", False)),
    handoff_generation=generation,
)
dataset_paths = {
    dataset: Path(path)
    for dataset, path in payload["dataset_paths"].items()
}
try:
    generated_paths, record = _generate_prepared_benchmark_handoff_inputs(config, dataset_paths, generation)
finally:
    release_handoff_generation_resources()
record["generator_python"] = sys.executable
write_json(
    output_path,
    {
        "generated_paths": {dataset: str(path) for dataset, path in generated_paths.items()},
        "record": record,
    },
)
"""
    argv = [
        str(config.venv_python),
        "-c",
        code,
        str(input_path),
        str(output_path),
    ]
    print(
        "+",
        " ".join([argv[0], "-c", "<prepared handoff generation>", *argv[3:]]),
        flush=True,
    )
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=generation.timeout_seconds,
            env=server_env(config),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"prepared handoff generation timed out after {generation.timeout_seconds:.1f}s; "
            f"stdout_tail={tail_text(exc.stdout)!r}; stderr_tail={tail_text(exc.stderr)!r}"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"prepared handoff generation failed with return code {completed.returncode}; "
            f"stdout_tail={tail_text(completed.stdout)!r}; stderr_tail={tail_text(completed.stderr)!r}"
        )
    if not output_path.exists():
        raise RuntimeError(f"prepared handoff generation did not write {output_path}")
    result = json.loads(output_path.read_text(encoding="utf-8"))
    generated_paths_payload = result.get("generated_paths")
    record = result.get("record")
    if not isinstance(generated_paths_payload, dict) or not isinstance(record, dict):
        raise RuntimeError(
            f"prepared handoff generation wrote invalid result {output_path}"
        )
    if record.get("ok") is not True:
        raise RuntimeError(
            f"prepared handoff generation worker result was not ok in {output_path}"
        )
    expected_datasets = set(dataset_paths)
    if set(generated_paths_payload) != expected_datasets:
        raise RuntimeError(
            "prepared handoff generation worker result must include exactly "
            f"{sorted(expected_datasets)!r}; got {sorted(str(dataset) for dataset in generated_paths_payload)!r}"
        )
    generated_paths = {
        str(dataset): Path(str(path))
        for dataset, path in generated_paths_payload.items()
    }
    for dataset, path in generated_paths.items():
        if not str(path):
            raise RuntimeError(
                f"prepared handoff generation worker returned empty path for {dataset}"
            )
        if not path.exists():
            raise RuntimeError(
                f"prepared handoff generation worker output for {dataset} does not exist: {path}"
            )
    return generated_paths, record


def _generate_prepared_benchmark_handoff_inputs(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: dict[str, Path],
    generation: VLLMPreparedHandoffGenerationConfig,
) -> tuple[dict[str, Path], dict[str, object]]:
    generation_started = time.perf_counter()
    generator = load_benchmark_kv_chunk_generator(generation.generator_factory)
    layout = layout_for_model(SERVED_MODEL_NAME, dtype=generation.dtype)
    generator_config = getattr(generator, "config", None)
    adapter_config = getattr(
        getattr(generator, "adapter", None),
        "config",
        None,
    )
    model_id = (
        getattr(generator, "model_id", None)
        or getattr(adapter_config, "model_id", None)
        or getattr(generator_config, "model_id", None)
        or layout.model_id
    )
    tokenizer_id = (
        getattr(generator, "tokenizer_id", None)
        or getattr(adapter_config, "tokenizer_id", None)
        or getattr(generator_config, "tokenizer_id", None)
        or model_id
    )
    model_revision = (
        getattr(generator, "model_revision", None)
        or getattr(adapter_config, "model_revision", None)
        or getattr(generator_config, "model_revision", None)
        or UNRESOLVED_IDENTITY
    )
    tokenizer_revision = (
        getattr(generator, "tokenizer_revision", None)
        or getattr(adapter_config, "tokenizer_revision", None)
        or getattr(generator_config, "tokenizer_revision", None)
        or UNRESOLVED_IDENTITY
    )
    generator_family = getattr(generator, "generator_family", "transformers")
    generator_version = getattr(
        generator,
        "generator_version",
        UNRESOLVED_IDENTITY,
    )
    if config.requires_pinned_revisions:
        model_revision = require_pinned_revision(
            model_revision,
            "handoff generator model_revision",
        )
        tokenizer_revision = require_pinned_revision(
            tokenizer_revision,
            "handoff generator tokenizer_revision",
        )
        require_pinned_revision(
            generator_version,
            "handoff generator generator_version",
        )
        expected_generator_identity = {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "tokenizer_id": config.model_id,
            "tokenizer_revision": config.tokenizer_revision,
        }
        observed_generator_identity = {
            "model_id": model_id,
            "model_revision": model_revision,
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
        }
        mismatches = sorted(
            key
            for key, expected in expected_generator_identity.items()
            if observed_generator_identity[key] != expected
        )
        if mismatches:
            raise ValueError(
                "handoff generator identity differs from the serving identity: "
                + ", ".join(mismatches)
            )
        if generation.cache_method == CacheGenerationMethod.VANILLA_PREFILL.value:
            observed_rope = (
                getattr(generator, "rope_theta", None),
                getattr(generator, "rope_rotary_dim", None),
            )
            expected_rope = (
                QWEN3_4B_ROPE_THETA,
                QWEN3_4B_ROPE_ROTARY_DIM,
            )
            if observed_rope != expected_rope:
                raise ValueError(
                    "representative Vanilla generator RoPE geometry differs from "
                    "the pinned Qwen3 model"
                )
    generator_pre_rope = getattr(generator, "pre_rope", False)
    if type(generator_pre_rope) is not bool:
        raise TypeError("generator.pre_rope must be a boolean when provided")
    layout = replace(
        layout,
        model_id=model_id,
        pre_rope=generator_pre_rope,
        rope_theta=(
            getattr(generator, "rope_theta", None) if generator_pre_rope else None
        ),
        rope_rotary_dim=(
            getattr(generator, "rope_rotary_dim", None) if generator_pre_rope else None
        ),
        key_position_encoding=(
            "pre_rope" if generator_pre_rope else "stored_post_rope"
        ),
        shares_kv_storage=False if generator_pre_rope else layout.shares_kv_storage,
        storage_layout=(
            "separate_key_value" if generator_pre_rope else layout.storage_layout
        ),
    )
    layout.validate()
    generated_paths: dict[str, Path] = {}
    dataset_records: dict[str, dict[str, object]] = {}
    topology_attestations: list[dict[str, Any]] = []
    try:
        topology_token_counter = generator_token_counter(generator)
    except TypeError:
        if config.requires_pinned_revisions:
            raise
        topology_token_counter = None
    for dataset in dataset_paths:
        dataset_generation_started = time.perf_counter()
        input_jsonl = dataset_paths[dataset]
        dataset_output_dir = generation.output_dir / dataset
        generation_input_jsonl = _handoff_generation_input_jsonl(
            input_jsonl,
            output_jsonl=dataset_output_dir / f"{dataset}.limited.jsonl",
            limit=generation.limit,
        )
        manifest_json = generation.output_dir / f"{dataset}-manifest.json"
        output_jsonl = generation.output_dir / f"{dataset}.handoffs.jsonl"
        result = generate_benchmark_handoff_bundles(
            generation_input_jsonl,
            output_dir=dataset_output_dir,
            generator=generator,
            layout=layout,
            dataset=dataset,
            backend="vllm",
            manifest_json=manifest_json,
            align_bytes=generation.align_bytes,
            segmented=generation.benchmark_handoff_segment_per_document,
            segment_per_document=generation.benchmark_handoff_segment_per_document,
            cache_method=generation.cache_method,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            generator_family=generator_family,
            generator_version=generator_version,
            require_artifact_contract=generation.require_artifact_contract,
        )
        enriched_rows = enrich_benchmark_jsonl_with_handoffs(
            generation_input_jsonl,
            manifest_json,
            output_jsonl,
            dataset=dataset,
            overwrite=True,
        )
        topology_attestation = (
            None
            if topology_token_counter is None
            else build_handoff_topology_attestation(
                generation_input_jsonl,
                result.manifest,
                token_counter=topology_token_counter,
            )
        )
        if topology_attestation is not None:
            topology_attestations.append(topology_attestation)
        generated_paths[dataset] = output_jsonl
        artifact_storage_bytes = _artifact_storage_bytes(result.shard_uri)
        dataset_records[dataset] = {
            "input_jsonl": str(input_jsonl),
            "generation_input_jsonl": str(generation_input_jsonl),
            "output_jsonl": str(output_jsonl),
            "manifest_json": str(manifest_json),
            "bundle_output_dir": str(dataset_output_dir),
            "entries": len(result.manifest.entries),
            "enriched_rows": enriched_rows,
            "cache_refs": len(result.cache_refs),
            "shard_uri": result.shard_uri,
            "artifact_generation_seconds": (
                time.perf_counter() - dataset_generation_started
            ),
            "artifact_payload_bytes": result.cache_generation.total_bytes,
            "artifact_storage_bytes": artifact_storage_bytes,
            "handoff_topology_attestation": topology_attestation,
        }

    artifact_payload_bytes = sum(
        int(dataset_record["artifact_payload_bytes"])
        for dataset_record in dataset_records.values()
    )
    artifact_storage_values = [
        int(value)
        for dataset_record in dataset_records.values()
        if (value := dataset_record["artifact_storage_bytes"]) is not None
    ]
    record = {
        "ok": True,
        "dataset_source": "prepared",
        "benchmark_id": config.benchmark_id,
        "generator_factory": generation.generator_factory,
        "output_dir": str(generation.output_dir),
        "dtype": generation.dtype,
        "align_bytes": generation.align_bytes,
        "cache_method": generation.cache_method,
        "require_artifact_contract": generation.require_artifact_contract,
        "artifact_model_id": model_id,
        "artifact_model_revision": model_revision,
        "artifact_tokenizer_id": tokenizer_id,
        "artifact_tokenizer_revision": tokenizer_revision,
        "generator_family": generator_family,
        "generator_version": generator_version,
        "artifact_generation_seconds": time.perf_counter() - generation_started,
        "artifact_payload_bytes": artifact_payload_bytes,
        "artifact_storage_bytes": (
            sum(artifact_storage_values)
            if len(artifact_storage_values) == len(dataset_records)
            else None
        ),
        "handoff_topology_attestation": (
            merge_handoff_topology_attestations(topology_attestations)
            if topology_attestations
            else None
        ),
        "datasets": dataset_records,
    }
    return generated_paths, record


def _artifact_storage_bytes(shard_uri: str) -> int | None:
    try:
        path = local_path(shard_uri)
    except (TypeError, ValueError):
        return None
    try:
        return path.stat().st_size
    except OSError:
        return None


def _config_with_generated_handoff_offline_costs(
    config: VLLMSmokeBenchmarkConfig,
) -> VLLMSmokeBenchmarkConfig:
    generation = config.handoff_generation
    if generation is None or not config.benchmark_arm_specs:
        return config
    record = json.loads(
        config.prepared_handoff_generation_path.read_text(encoding="utf-8")
    )
    duration = record.get("artifact_generation_seconds")
    artifact_bytes = record.get("artifact_storage_bytes")
    if artifact_bytes is None:
        artifact_bytes = record.get("artifact_payload_bytes")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or duration < 0
    ):
        raise ValueError(
            "prepared handoff generation record must include non-negative "
            "artifact_generation_seconds"
        )
    if type(artifact_bytes) is not int or artifact_bytes < 0:
        raise ValueError(
            "prepared handoff generation record must include non-negative artifact bytes"
        )
    expected_method = generation.cache_method or (
        CacheGenerationMethod.VANILLA_PREFILL.value
        if generation.benchmark_handoff_segment_per_document
        else CacheGenerationMethod.FULL_PREFIX_PREFILL.value
    )
    updated_specs: list[Mapping[str, Any]] = []
    matched = 0
    for raw_spec in config.benchmark_arm_specs:
        spec = benchmark_json_mapping_to_record(raw_spec)
        if (
            _arm_spec_requires_cachet_handoff(spec)
            and spec.get("cache_method") == expected_method
        ):
            matched += 1
            costs = dict(spec.get("offline_costs", {}))
            costs["artifact_generation_seconds"] = float(duration)
            costs["artifact_bytes"] = artifact_bytes
            spec["offline_costs"] = costs
        updated_specs.append(spec)
    if matched != 1:
        raise ValueError(
            "generated handoff costs must map to exactly one benchmark arm for "
            f"method {expected_method!r}; matched {matched}"
        )
    return replace(config, benchmark_arm_specs=tuple(updated_specs))


def _handoff_generation_input_jsonl(
    input_jsonl: Path,
    *,
    output_jsonl: Path,
    limit: int | None,
) -> Path:
    if limit is None:
        return input_jsonl
    records: list[Mapping[str, Any]] = []
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    write_v1_jsonl(records, output_jsonl)
    return output_jsonl


def release_handoff_generation_resources() -> None:
    """Release best-effort Transformers/Torch memory before vLLM starts."""

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    cuda = getattr(torch, "cuda", None)
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def run_prompt_token_budget_probe(
    python_executable: Path,
    input_path: Path,
    *,
    model_id: str,
    model_revision: str | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    add_special_tokens: bool = PROMPT_TOKEN_PROBE_ADD_SPECIAL_TOKENS,
    expected_prompt_tokens: int | None = None,
    max_model_len: int,
    max_tokens: int,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    if model_revision is not None and (
        not isinstance(model_revision, str) or not model_revision
    ):
        raise ValueError("model_revision must be non-empty when provided")
    resolved_tokenizer_id = tokenizer_id or model_id
    if not isinstance(resolved_tokenizer_id, str) or not resolved_tokenizer_id:
        raise ValueError("tokenizer_id must be non-empty")
    if tokenizer_revision is not None and (
        not isinstance(tokenizer_revision, str) or not tokenizer_revision
    ):
        raise ValueError("tokenizer_revision must be non-empty when provided")
    if type(add_special_tokens) is not bool:
        raise TypeError("add_special_tokens must be a boolean")
    if expected_prompt_tokens is not None and (
        type(expected_prompt_tokens) is not int or expected_prompt_tokens <= 0
    ):
        raise ValueError("expected_prompt_tokens must be positive when provided")
    tokenizer_record = {
        "tokenizer_id": resolved_tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "add_special_tokens": add_special_tokens,
    }
    model_record = {"model_id": model_id, "model_revision": model_revision}
    code = """
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

tokenizer_id = sys.argv[1]
tokenizer_revision = sys.argv[2] or None
model_id = sys.argv[3]
model_revision = sys.argv[4] or None
add_special_tokens = sys.argv[5] == "true"
expected_prompt_tokens = None if sys.argv[6] == "" else int(sys.argv[6])
input_path, max_model_len, max_tokens = Path(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_id,
    revision=tokenizer_revision,
    trust_remote_code=False,
)
rows = []
over_budget = []
token_count_mismatches = []
with input_path.open("r", encoding="utf-8") as handle:
    for raw_line in handle:
        row = json.loads(raw_line)
        prompt_tokens = len(
            tokenizer(
                row["prompt"],
                add_special_tokens=add_special_tokens,
            )["input_ids"]
        )
        total_tokens = prompt_tokens + max_tokens
        measured = {
            "dataset": row["dataset"],
            "example_id": row["example_id"],
            "logical_prompt_sha256": hashlib.sha256(
                row["prompt"].encode("utf-8")
            ).hexdigest(),
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "total_tokens": total_tokens,
            "max_model_len": max_model_len,
        }
        rows.append(measured)
        if total_tokens > max_model_len:
            over_budget.append(measured)
        if (
            expected_prompt_tokens is not None
            and prompt_tokens != expected_prompt_tokens
        ):
            token_count_mismatches.append(measured)
print(
    json.dumps(
        {
            "model": {"model_id": model_id, "model_revision": model_revision},
            "tokenizer": {
                "tokenizer_id": tokenizer_id,
                "tokenizer_revision": tokenizer_revision,
                "add_special_tokens": add_special_tokens,
            },
            "expected_prompt_tokens": expected_prompt_tokens,
            "rows": rows,
            "over_budget": over_budget,
            "token_count_mismatches": token_count_mismatches,
        },
        sort_keys=True,
    ),
    flush=True,
)
"""
    argv = [
        str(python_executable),
        "-c",
        code,
        resolved_tokenizer_id,
        tokenizer_revision or "",
        model_id,
        model_revision or "",
        "true" if add_special_tokens else "false",
        "" if expected_prompt_tokens is None else str(expected_prompt_tokens),
        str(input_path),
        str(max_model_len),
        str(max_tokens),
    ]
    print(
        "+",
        " ".join([argv[0], "-c", "<prompt token budget probe>", *argv[3:]]),
        flush=True,
    )
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env or os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error_type": "TimeoutExpired",
            "error": f"prompt token budget probe timed out after {timeout_seconds:.1f}s",
            "stdout_tail": tail_text(exc.stdout),
            "stderr_tail": tail_text(exc.stderr),
            "tokenizer": tokenizer_record,
            "model": model_record,
            "expected_prompt_tokens": expected_prompt_tokens,
            "rows": [],
            "over_budget": [],
            "token_count_mismatches": [],
        }
    record = last_json_object(completed.stdout)
    record.update(
        {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "tokenizer": tokenizer_record,
            "model": model_record,
            "expected_prompt_tokens": expected_prompt_tokens,
            "stdout_tail": tail_text(completed.stdout),
            "stderr_tail": tail_text(completed.stderr),
        }
    )
    if completed.returncode != 0:
        record.setdefault(
            "error",
            f"prompt token budget probe failed with return code {completed.returncode}",
        )
        record.setdefault("error_type", "CalledProcessError")
        record.setdefault("rows", [])
        record.setdefault("over_budget", [])
        record.setdefault("token_count_mismatches", [])
    return record


def run_benchmark_runner(
    config: VLLMSmokeBenchmarkConfig, dataset_paths: dict[str, Path]
) -> None:
    try:
        run(build_benchmark_runner_args(config, dataset_paths))
    except subprocess.CalledProcessError as exc:
        summary = benchmark_failure_summary(config.benchmark_output_path)
        raise RuntimeError(
            f"vLLM benchmark runner failed with exit code {exc.returncode}; {summary}"
        ) from exc


def install_lmcache(python_executable: Path, version: str = "") -> str:
    """Install LMCache into the vLLM venv and return the resolved version."""

    spec = f"lmcache=={version}" if version else "lmcache"
    run(
        [str(python_executable), "-m", "pip", "install", spec],
        env=_pip_subprocess_environment(),
    )
    return installed_package_version(python_executable, "lmcache")


def write_lmcache_config(config: VLLMSmokeBenchmarkConfig) -> Path:
    """Write an LMCache config with a NVMe disk tier and optional CPU-RAM tier.

    By default ``local_cpu`` is disabled and ``use_odirect`` is enabled so reads are
    genuine cold NVMe reads. When ``lmcache_local_cpu`` is set, KV is offloaded to a
    bounded CPU-RAM tier (``max_local_cpu_size`` GB) first and spills to the NVMe disk
    tier on overflow (LRU) -- the limited-GPU regime where active-conversation KV lives
    in RAM and only overflows to disk. ``use_odirect`` is left off in that mode so the
    disk-overflow tier can use the OS page cache.
    """

    disk_dir = Path(config.lmcache_local_dir)
    disk_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "chunk_size": config.lmcache_chunk_size,
        "local_cpu": bool(config.lmcache_local_cpu),
        "local_disk": f"file://{disk_dir}/",
        "max_local_disk_size": config.lmcache_max_disk_gb,
    }
    if config.lmcache_local_cpu:
        if config.lmcache_max_cpu_gb > 0:
            payload["max_local_cpu_size"] = config.lmcache_max_cpu_gb
    else:
        payload["extra_config"] = {"use_odirect": True}
    path = config.local_dir / "lmcache-config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def probe_lmcache_import(
    python_executable: Path,
    output_path: Path,
    *,
    timeout_seconds: float,
    env: dict[str, str],
) -> dict[str, object]:
    """Fail fast if lmcache cannot be imported alongside vLLM (torch ABI check)."""

    script = (
        "import json\n"
        "rec = {}\n"
        "try:\n"
        "    import torch; rec['torch'] = torch.__version__\n"
        "    import vllm; rec['vllm'] = vllm.__version__\n"
        "    import lmcache; rec['lmcache'] = getattr(lmcache, '__version__', 'unknown')\n"
        "    rec['ok'] = True\n"
        "except Exception as exc:\n"
        "    rec['ok'] = False; rec['error'] = f'{type(exc).__name__}: {exc}'\n"
        "try:\n"
        "    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl  # noqa: F401\n"
        "    rec['lmcache_v1_adapter_import'] = True\n"
        "except Exception as exc:\n"
        "    rec['lmcache_v1_adapter_import'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(rec))\n"
    )
    completed = subprocess.run(
        [str(python_executable), "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )
    record: dict[str, object]
    try:
        record = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        record = {"ok": False, "error": tail_text(completed.stdout + completed.stderr)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not record.get("ok"):
        raise RuntimeError(
            f"lmcache import probe failed: {record.get('error')!r}; "
            f"stderr tail:\n{tail_text(completed.stderr)}"
        )
    return record


def _build_lmcache_pass_args(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: dict[str, Path],
    output_path: Path,
    suite_suffix: str,
) -> list[str]:
    """Minimal baseline-arm benchmark invocation (LMCache is transparent to the client)."""

    args = [
        sys.executable,
        "-m",
        "document_kv_cache.benchmark_runner",
        "--suite-id",
        f"{config.benchmark_id}-{suite_suffix}",
        "--base-url",
        config.server_base_url,
        "--model-id",
        SERVED_MODEL_NAME,
        "--hardware-target",
        config.hardware_target,
        "--max-tokens",
        str(config.max_tokens),
        "--temperature",
        str(config.temperature),
        "--timeout-seconds",
        str(config.timeout_seconds),
        "--repeats",
        "1",
        "--request-parallelism",
        str(config.request_parallelism),
        "--server-usage",
        "--output-json",
        str(output_path),
        "--arm",
        BASELINE_PREFILL_ARM,
    ]
    if config.generation_seed is not None:
        args.extend(["--generation-seed", str(config.generation_seed)])
    if config.force_max_tokens:
        args.extend(
            [
                "--baseline-extra-body-json",
                json.dumps({"ignore_eos": True}, sort_keys=True),
            ]
        )
    args.extend(dataset_args(dataset_paths))
    return args


def _run_lmcache_two_pass(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: dict[str, Path],
    warm_output: Path,
    measure_output: Path,
) -> None:
    """Warm then measure LMCache cold-disk reload on a *single* server.

    LMCache's local-disk backend keeps its lookup index in memory and does not
    rebuild it from the on-disk chunks when a fresh engine starts, so a
    warm->restart->measure flow makes the restarted engine miss every chunk it
    just wrote. Keeping one server preserves the index; ``local_cpu=false`` plus
    ``use_odirect`` still force the measure pass to read the KV cold from NVMe
    (no CPU tier, page cache bypassed) rather than from RAM.
    """

    log_path = config.local_dir / "vllm-server-lmcache.log"
    server = start_vllm_server(config, config.venv_python, log_path)
    try:
        wait_for_server(
            server,
            log_path,
            config,
            timeout_seconds=config.server_start_timeout_seconds,
        )
        # Phase 1: warm -- prefill each distinct document once so its KV persists to the disk tier.
        run(_build_lmcache_pass_args(config, dataset_paths, warm_output, "warm"))
        # Phase 2: best-effort page-cache drop; O_DIRECT reads bypass it anyway.
        _evict_between_lmcache_phases()
        # Phase 3: measure -- same prompts; LMCache reloads KV cold from disk (index still resident).
        run(_build_lmcache_pass_args(config, dataset_paths, measure_output, "cold"))
    finally:
        terminate_process(server)
        copy_file_if_exists(log_path, config.output_dir / "vllm-server-lmcache.log")


def _evict_between_lmcache_phases() -> None:
    """Best-effort flush so the measured pass reads KV cold from NVMe.

    Both passes share one server, so the GPU KV blocks for the earliest
    documents are already evicted by the time the measure pass revisits them,
    and ``local_cpu=false`` means there is no CPU tier to serve from. Here we
    additionally sync and (best-effort) drop the OS page cache; LMCache's
    ``use_odirect`` reads bypass the page cache regardless, so this is belt and
    suspenders rather than the sole guarantee.
    """

    subprocess.run(["sync"], check=False)
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="utf-8")
    except OSError:
        pass
    time.sleep(2)


def run_lmcache_cold_benchmark(config: VLLMSmokeBenchmarkConfig) -> None:
    """Warm then measure LMCache cold-disk KV reload TTFT on a single server."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.local_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(config.hf_cache_dir)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # Propagate system-prompt placement to prompt-building subprocesses (handoff
    # generation, client benchmark runner, budget probe) via inherited os.environ.
    os.environ[CACHET_BENCHMARK_SYSTEM_PROMPT_POSITION_ENV] = (
        config.system_prompt_position
    )

    metadata = build_metadata(config)
    metadata["kv_connector_mode"] = LMCACHE_KV_CONNECTOR_MODE
    write_json(config.metadata_path, metadata)

    create_venv(config.venv_dir)
    install_vllm(config.venv_python)
    # cachet-kv is not used by the LMCache connector, but installing it keeps the
    # shared metadata/version plumbing (installed_versions) consistent.
    install_document_kv_package(
        config.venv_python, document_kv_package_install_spec(config)
    )
    metadata["vllm_runtime_lock_verification"] = verify_vllm_runtime_lock_installation(
        config.venv_python
    )
    metadata["vllm_runtime_lock_verification_scope"] = (
        "base-runtime-before-unlocked-lmcache"
    )
    metadata["strict_runtime_closure"] = False
    metadata["vllm_runtime_patch_closure"] = verify_vllm_runtime_patch_closure(config)
    lmcache_version = install_lmcache(config.venv_python, config.lmcache_version)
    run(
        [str(config.venv_python), "-m", "pip", "check"],
        env=_pip_subprocess_environment(),
    )
    versions = installed_versions(config.venv_python)
    versions["lmcache_version_installed"] = lmcache_version
    metadata.update(versions)
    metadata["installed_package_freeze"] = installed_package_freeze(config.venv_python)

    lmcache_config_path = write_lmcache_config(config)
    os.environ["LMCACHE_CONFIG_FILE"] = str(lmcache_config_path)
    metadata["lmcache_config_path"] = str(lmcache_config_path)
    metadata["lmcache_config"] = json.loads(
        lmcache_config_path.read_text(encoding="utf-8")
    )
    metadata["lmcache_warm_benchmark_path"] = str(
        config.output_dir / "lmcache-warm-benchmark.json"
    )
    write_json(config.metadata_path, metadata)

    probe_lmcache_import(
        config.venv_python,
        config.import_probe_path,
        timeout_seconds=config.import_probe_timeout_seconds,
        env=server_env(config),
    )

    dataset_paths = benchmark_dataset_paths(config)
    # Preflight the context budget before the expensive warm+measure passes so
    # over-budget prepared prompts fail fast (mirrors the vLLM-native path) and the
    # budget artifact is recorded for LMCache runs too.
    validate_prompt_token_budget(config, dataset_paths)
    metadata["prompt_token_budget_path"] = str(config.prompt_token_budget_path)
    write_json(config.metadata_path, metadata)
    warm_output = config.output_dir / "lmcache-warm-benchmark.json"
    _run_lmcache_two_pass(
        config, dataset_paths, warm_output, config.benchmark_output_path
    )


def prewarm_cache_prefixes(
    config: VLLMSmokeBenchmarkConfig, dataset_paths: dict[str, Path]
) -> None:
    """Load prepared cache prefixes into vLLM's resident prefix cache before measurement."""

    if not config.prewarm_cache_prefix:
        return
    suite = load_v1_jsonl_suite(
        suite_id=f"{config.benchmark_id}-prewarm",
        paths=dataset_paths,
        model_id=SERVED_MODEL_NAME,
        hardware_target=config.hardware_target,
    )
    rows: list[dict[str, object]] = []
    ok = True
    completions_url = f"{config.server_base_url}/v1/completions"
    for example in suite.examples:
        started = time.monotonic()
        kv_transfer_params = dict(example.kv_transfer_params)
        kv_transfer_params[DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM] = "logical"
        body = {
            "model": SERVED_MODEL_NAME,
            "prompt": _prewarm_prompt_text(example),
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
            "cache_salt": CACHE_PREFIX_CACHE_SALT,
            "request_id": _prewarm_request_id(config, example),
            "kv_transfer_params": kv_transfer_params,
        }
        try:
            response = _post_json(
                completions_url, body, timeout_seconds=config.timeout_seconds
            )
            usage = response.get("usage")
            rows.append(
                {
                    "ok": True,
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "elapsed_seconds": time.monotonic() - started,
                    "prompt_tokens": usage.get("prompt_tokens")
                    if isinstance(usage, Mapping)
                    else None,
                    "completion_tokens": usage.get("completion_tokens")
                    if isinstance(usage, Mapping)
                    else None,
                }
            )
        except Exception as exc:
            ok = False
            rows.append(
                {
                    "ok": False,
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "elapsed_seconds": time.monotonic() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc) or type(exc).__name__,
                }
            )
            break
    record = {
        "ok": ok,
        "benchmark_id": config.benchmark_id,
        "server_base_url": config.server_base_url,
        "cache_salt": CACHE_PREFIX_CACHE_SALT,
        "rows": rows,
        "row_count": len(rows),
    }
    write_json(config.prewarm_cache_prefix_path, record)
    if not ok:
        last = rows[-1] if rows else {}
        raise RuntimeError(
            "vLLM cache-prefix prewarm failed; "
            f"last_error={last.get('error')!r}. See {config.prewarm_cache_prefix_path}."
        )


def prime_payload_cache(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: dict[str, Path],
) -> None:
    """Populate and verify the provider's host-RAM payload cache before timing.

    vLLM does not expose a worker-side payload-cache RPC, so priming uses ordinary
    KV-aware completion requests. Every request uses a priming-only ``cache_salt``
    namespace and request identity. The requests therefore allocate isolated GPU
    blocks, but cannot populate or match the salted prefix keys used by measurement.
    A second full pass must hit RAM for every target before the benchmark starts.
    """

    if not config.prewarm_payload_cache:
        return
    targets = _payload_cache_prime_targets(config, dataset_paths)
    completions_url = f"{config.server_base_url}/v1/completions"
    rows: list[dict[str, object]] = []
    expected_request_ids: dict[str, tuple[str, str, str]] = {}
    issues: list[str] = []
    for phase in ("populate", "verify"):
        for target_index, (example, arm_id, params) in enumerate(targets):
            identity = (
                f"{config.benchmark_id}:{phase}:{target_index}:"
                f"{example.dataset}:{example.example_id}:{arm_id}"
            )
            identity_digest = sha256(identity.encode("utf-8")).hexdigest()
            request_id = f"cachet-payload-prime:{phase}:{identity_digest}"
            cache_salt = (
                f"{PAYLOAD_CACHE_PRIME_PREFIX_CACHE_SALT}:{phase}:{identity_digest}"
            )
            expected_request_ids[request_id] = (phase, example.dataset, arm_id)
            kv_transfer_params = dict(params)
            kv_transfer_params[DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM] = "logical"
            kv_transfer_params[DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM] = request_id
            body = {
                "model": SERVED_MODEL_NAME,
                # The handoff's per-segment token contracts were generated from the
                # exact logical benchmark prompt. A shortened warmup suffix can
                # change tokenizer boundary merges inside the cached prefix and make
                # the provider reject an otherwise valid artifact. Preserve the full
                # logical prompt here; the priming-only request ID and cache salt
                # still isolate these GPU blocks from every measurement request.
                "prompt": _payload_cache_prime_prompt_text(example),
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
                "cache_salt": cache_salt,
                "request_id": request_id,
                "kv_transfer_params": kv_transfer_params,
            }
            started = time.monotonic()
            row: dict[str, object] = {
                "phase": phase,
                "dataset": example.dataset,
                "arm_id": arm_id,
                "target_sha256": identity_digest,
                "request_id_sha256": sha256(request_id.encode("utf-8")).hexdigest(),
                "cache_salt_sha256": sha256(cache_salt.encode("utf-8")).hexdigest(),
            }
            try:
                response = _post_json(
                    completions_url,
                    body,
                    timeout_seconds=config.timeout_seconds,
                )
                usage = response.get("usage")
                row.update(
                    {
                        "ok": True,
                        "elapsed_seconds": time.monotonic() - started,
                        "prompt_tokens": (
                            usage.get("prompt_tokens")
                            if isinstance(usage, Mapping)
                            else None
                        ),
                        "completion_tokens": (
                            usage.get("completion_tokens")
                            if isinstance(usage, Mapping)
                            else None
                        ),
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "ok": False,
                        "elapsed_seconds": time.monotonic() - started,
                        "error_type": type(exc).__name__,
                        "error": str(exc) or type(exc).__name__,
                    }
                )
                issues.append(
                    f"{phase} request failed for {example.dataset}:{arm_id}: "
                    f"{type(exc).__name__}"
                )
            rows.append(row)

    try:
        telemetry = _read_jsonl_records(config.connector_telemetry_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        telemetry = []
        issues.append(
            "payload-cache priming telemetry could not be read: "
            f"{type(exc).__name__}: {exc}"
        )
    telemetry_by_request_id: dict[str, list[Mapping[str, Any]]] = {}
    for record in telemetry:
        request_id = record.get("benchmark_request_id")
        if isinstance(request_id, str) and request_id in expected_request_ids:
            telemetry_by_request_id.setdefault(request_id, []).append(record)
    for request_id, (phase, dataset, arm_id) in expected_request_ids.items():
        records = telemetry_by_request_id.get(request_id, [])
        if len(records) != 1:
            issues.append(
                f"{phase} telemetry for {dataset}:{arm_id} expected exactly one load, "
                f"observed {len(records)}"
            )
            continue
        issues.extend(
            _payload_cache_load_issues(
                records[0],
                label=f"{phase} telemetry for {dataset}:{arm_id}",
                require_hit=phase == "verify",
            )
        )

    record = {
        "record_type": "document_kv.vllm_payload_cache_prime.v1",
        "schema_version": 1,
        "ok": not issues,
        "benchmark_id": config.benchmark_id,
        "payload_cache_max_bytes": config.payload_cache_max_bytes,
        "target_count": len(targets),
        "request_count": len(rows),
        "verification_all_hits": not issues,
        "prefix_cache_isolation": {
            "measurement_prefix_cache_salt_mode": config.prefix_cache_salt_mode,
            "measurement_prefix_prewarmed": False,
            "priming_requests_load_isolated_gpu_blocks": True,
            "priming_namespace": PAYLOAD_CACHE_PRIME_PREFIX_CACHE_SALT,
        },
        "request_id_sha256s": sorted(str(row["request_id_sha256"]) for row in rows),
        "cache_salt_sha256s": sorted(str(row["cache_salt_sha256"]) for row in rows),
        "rows": rows,
        "issues": issues,
    }
    write_json(config.prewarm_payload_cache_path, record)
    if issues:
        raise RuntimeError(
            "vLLM payload-cache priming did not prove full RAM residency: "
            f"{'; '.join(issues[:3])}. See {config.prewarm_payload_cache_path}."
        )


def attest_payload_cache_measurements(config: VLLMSmokeBenchmarkConfig) -> None:
    """Fail unless every measured Cachet load was served from the host-RAM cache."""

    if not config.prewarm_payload_cache:
        return
    issues: list[str] = []
    prime_record = _read_json_object(config.prewarm_payload_cache_path)
    if (
        prime_record.get("ok") is not True
        or prime_record.get("verification_all_hits") is not True
    ):
        issues.append("payload-cache priming verification did not pass")
    benchmark_record = _read_json_object(config.benchmark_output_path)
    measurements = benchmark_record.get("measurements")
    if not isinstance(measurements, list):
        raise ValueError("vLLM benchmark output measurements must be a list")
    measured_request_ids: list[str] = []
    measured_cache_salt_sha256s: list[str] = []
    for index, measurement in enumerate(measurements):
        if not isinstance(measurement, Mapping):
            raise ValueError(f"vLLM benchmark measurements[{index}] must be an object")
        request_id = measurement.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        measured_request_ids.append(request_id)
        metadata = measurement.get("metadata")
        if not isinstance(metadata, Mapping):
            issues.append(f"measurement {index} is missing metadata")
            continue
        metadata_request_id = metadata.get("request_id")
        if metadata_request_id != request_id:
            issues.append(
                f"measurement {index} metadata.request_id does not match the canonical "
                "measurement.request_id"
            )
        cache_salt = metadata.get("prefix_cache_salt")
        if not isinstance(cache_salt, str) or not cache_salt:
            issues.append(f"measurement {index} is missing a raw prefix_cache_salt")
            continue
        if metadata.get("prefix_cache_salt_attached") != "true":
            issues.append(
                f"measurement {index} does not attest that its prefix-cache salt was attached"
            )
        if cache_salt.startswith(PAYLOAD_CACHE_PRIME_PREFIX_CACHE_SALT):
            issues.append(
                f"measurement {index} uses the reserved payload-cache priming salt namespace"
            )
        measured_cache_salt_sha256s.append(
            sha256(cache_salt.encode("utf-8")).hexdigest()
        )
    if not measured_request_ids:
        issues.append("benchmark output contains no measured Cachet request identities")
    if len(set(measured_request_ids)) != len(measured_request_ids):
        issues.append("measured Cachet request identities are not unique")

    measured_request_id_set = set(measured_request_ids)
    telemetry_by_request_id: dict[str, list[Mapping[str, Any]]] = {}
    for record in _read_jsonl_records(config.connector_telemetry_path):
        request_id = record.get("benchmark_request_id")
        if isinstance(request_id, str) and request_id in measured_request_id_set:
            telemetry_by_request_id.setdefault(request_id, []).append(record)
    measurement_load_count = 0
    total_bytes_read = 0
    total_cache_hits = 0
    total_cache_misses = 0
    for request_id in measured_request_ids:
        records = telemetry_by_request_id.get(request_id, [])
        request_id_digest = sha256(request_id.encode("utf-8")).hexdigest()
        if not records:
            issues.append(
                f"measurement request {request_id_digest} has no connector load telemetry"
            )
            continue
        measurement_load_count += len(records)
        for record in records:
            issues.extend(
                _payload_cache_load_issues(
                    record,
                    label=f"measurement telemetry for {request_id_digest}",
                    require_hit=True,
                )
            )
            state = record.get("cache_state_attestation")
            counts = record.get("counts")
            if isinstance(state, Mapping) and type(state.get("bytes_read")) is int:
                total_bytes_read += state["bytes_read"]
            if isinstance(counts, Mapping):
                if type(counts.get("payload_cache_hits")) is int:
                    total_cache_hits += counts["payload_cache_hits"]
                if type(counts.get("payload_cache_misses")) is int:
                    total_cache_misses += counts["payload_cache_misses"]

    priming_request_digests = _string_set(
        prime_record.get("request_id_sha256s"),
        field_name="prewarm-payload-cache.request_id_sha256s",
    )
    priming_salt_digests = _string_set(
        prime_record.get("cache_salt_sha256s"),
        field_name="prewarm-payload-cache.cache_salt_sha256s",
    )
    measurement_request_digests = {
        sha256(request_id.encode("utf-8")).hexdigest()
        for request_id in measured_request_ids
    }
    measurement_salt_digests = set(measured_cache_salt_sha256s)
    request_ids_disjoint = priming_request_digests.isdisjoint(
        measurement_request_digests
    )
    cache_salts_disjoint = priming_salt_digests.isdisjoint(measurement_salt_digests)
    if not request_ids_disjoint:
        issues.append("priming and measurement request identities overlap")
    if not cache_salts_disjoint:
        issues.append("priming and measurement prefix-cache salts overlap")

    attestation = {
        "record_type": "document_kv.vllm_ram_payload_cache_attestation.v1",
        "schema_version": 1,
        "ok": not issues,
        "benchmark_id": config.benchmark_id,
        "measurement_protocol": "ram_payload_cache_to_gpu_hydrate",
        "payload_cache": {
            "enabled": True,
            "max_bytes": config.payload_cache_max_bytes,
            "priming_target_count": prime_record.get("target_count"),
            "priming_verification_all_hits": prime_record.get("verification_all_hits"),
            "measurement_request_count": len(measured_request_ids),
            "measurement_load_count": measurement_load_count,
            "measurement_all_hits": (
                measurement_load_count >= len(measured_request_ids)
                and total_cache_hits == measurement_load_count
                and total_cache_misses == 0
                and total_bytes_read == 0
            ),
            "measurement_cache_hits": total_cache_hits,
            "measurement_cache_misses": total_cache_misses,
            "measurement_storage_bytes_read": total_bytes_read,
            "measurement_storage_materializations": total_cache_misses,
        },
        "gpu_prefix_cache": {
            "prewarm_cache_prefix_enabled": config.prewarm_cache_prefix,
            "measurement_cache_salt_mode": config.prefix_cache_salt_mode,
            "measurement_prefix_prewarmed": False,
            "priming_requests_load_isolated_gpu_blocks": True,
            "priming_and_measurement_request_ids_disjoint": request_ids_disjoint,
            "priming_and_measurement_cache_salts_disjoint": cache_salts_disjoint,
            "reuse_prevented_by_salt_namespace": (
                request_ids_disjoint and cache_salts_disjoint
            ),
        },
        "measurement_request_id_sha256s": sorted(measurement_request_digests),
        "measurement_cache_salt_sha256s": sorted(measurement_salt_digests),
        "issues": issues,
    }
    write_json(config.payload_cache_attestation_path, attestation)
    if issues:
        raise RuntimeError(
            "vLLM RAM payload-cache measurement attestation failed: "
            f"{'; '.join(issues[:3])}. See {config.payload_cache_attestation_path}."
        )


def _payload_cache_prime_targets(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: Mapping[str, Path],
) -> list[tuple[Any, str, Mapping[str, Any]]]:
    suite = load_v1_jsonl_suite(
        suite_id=f"{config.benchmark_id}-payload-cache-prime",
        paths=dataset_paths,
        model_id=SERVED_MODEL_NAME,
        hardware_target=config.hardware_target,
    )
    cache_arm_ids = _prepared_cache_arm_ids(config)
    if not cache_arm_ids:
        raise ValueError(
            "payload-cache priming requires at least one Cachet handoff arm"
        )
    targets: list[tuple[Any, str, Mapping[str, Any]]] = []
    for example in suite.examples:
        params_by_arm = _prepared_params_by_arm(
            example,
            cache_arm_ids=cache_arm_ids,
        )
        for arm_id, params in params_by_arm.items():
            if not params:
                raise ValueError(
                    f"prepared example {example.dataset}:{example.example_id} has no "
                    f"Cachet handoff params for {arm_id!r}"
                )
            targets.append((example, arm_id, params))
    if not targets:
        raise ValueError("payload-cache priming resolved no prepared artifacts")
    limit = config.payload_cache_prime_target_count
    if limit is not None:
        if limit > len(targets):
            raise ValueError(
                "payload-cache prime target count exceeds prepared targets"
            )
        by_dataset = {
            dataset: sorted(
                (target for target in targets if target[0].dataset == dataset),
                key=lambda target: (target[0].example_id, target[1]),
            )
            for dataset in SUPPORTED_V1_DATASETS
        }
        if limit % len(SUPPORTED_V1_DATASETS):
            raise ValueError(
                "payload-cache prime target count must balance all datasets"
            )
        per_dataset = limit // len(SUPPORTED_V1_DATASETS)
        if any(len(values) < per_dataset for values in by_dataset.values()):
            raise ValueError("payload-cache prime target coverage is incomplete")
        targets = [
            target
            for dataset in SUPPORTED_V1_DATASETS
            for target in by_dataset[dataset][:per_dataset]
        ]
    return targets


def _payload_cache_load_issues(
    record: Mapping[str, Any],
    *,
    label: str,
    require_hit: bool,
) -> list[str]:
    issues: list[str] = []
    if record.get("success") is not True:
        issues.append(f"{label} was not successful")
    state = record.get("cache_state_attestation")
    counts = record.get("counts")
    payload = record.get("payload")
    if not isinstance(state, Mapping):
        return [*issues, f"{label} is missing cache_state_attestation"]
    if not isinstance(counts, Mapping):
        return [*issues, f"{label} is missing counts"]
    if (
        not isinstance(payload, Mapping)
        or payload.get("payload_cache_enabled") is not True
    ):
        issues.append(f"{label} does not attest an enabled payload cache")
    if require_hit:
        if state.get("payload_cache_hit") is not True:
            issues.append(f"{label} did not attest a payload-cache hit")
        bytes_read = state.get("bytes_read")
        if type(bytes_read) is not int or bytes_read != 0:
            issues.append(f"{label} read payload bytes from storage")
        payload_cache_hits = counts.get("payload_cache_hits")
        if type(payload_cache_hits) is not int or payload_cache_hits != 1:
            issues.append(f"{label} did not report exactly one payload-cache hit")
        payload_cache_misses = counts.get("payload_cache_misses")
        if type(payload_cache_misses) is not int or payload_cache_misses != 0:
            issues.append(f"{label} reported a payload-cache miss")
    return issues


def _read_jsonl_records(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise ValueError(f"required JSONL artifact does not exist: {path}")
    records: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def _read_json_object(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"required JSON artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"required JSON artifact must contain an object: {path}")
    return value


def _string_set(value: object, *, field_name: str) -> set[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    return set(value)


def _prewarm_prompt_text(example: Any) -> str:
    return build_prompt_parts(example).cache_prefix_text + "\n\nCache warmup."


def _payload_cache_prime_prompt_text(example: Any) -> str:
    """Return the exact prompt from which the handoff token contracts were built."""

    return build_prompt_parts(example).prefill_prompt


def _prewarm_request_id(config: VLLMSmokeBenchmarkConfig, example: Any) -> str:
    return (
        f"cachet-prewarm:{config.benchmark_id}:{example.dataset}:{example.example_id}"
    )


# Follow-up turns for the hybrid multi-turn latency measurement. Turn 1 carries the
# document handoff (served by Cachet); every follow-up turn omits the handoff so the
# request falls through to LMCache, which reuses the resident conversation KV.
DEFAULT_MULTI_TURN_FOLLOWUPS = (
    "Based on the document(s) above, what are the three most important points?",
    "Summarize your previous answer in a single sentence.",
)


def build_multi_turn_followup_prompt(
    turn_prompt: str, turn_response: str, followup_question: str
) -> str:
    """Append the model's response and the next user turn to the running conversation.

    Keeping the exact prior text (prompt + generated response) as a prefix is what lets
    the engine's prefix cache / LMCache reuse the already-computed conversation KV, so
    only the new follow-up tokens are prefilled.
    """

    return f"{turn_prompt}{turn_response}\n\n{followup_question}\n"


def _stream_completion_ttft(
    url: str,
    body: Mapping[str, object],
    *,
    timeout_seconds: float,
) -> tuple[float, str, Mapping[str, Any]]:
    """POST a streaming completion and return (ttft_seconds, output_text, usage)."""

    payload = json.dumps(
        {**body, "stream": True, "stream_options": {"include_usage": True}}
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    first_token_at: float | None = None
    parts: list[str] = []
    usage: Mapping[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data_str = line.removeprefix("data:").strip()
            if data_str == "[DONE]":
                break
            data = json.loads(data_str)
            if isinstance(data.get("usage"), Mapping):
                usage = data["usage"]
            choices = data.get("choices") or []
            text = str(choices[0].get("text", "")) if choices else ""
            if text:
                if first_token_at is None:
                    first_token_at = time.monotonic()
                parts.append(text)
    completed = time.monotonic()
    ttft = (first_token_at or completed) - started
    return ttft, "".join(parts), usage


def run_multi_turn_hybrid_latency(
    config: VLLMSmokeBenchmarkConfig, dataset_paths: dict[str, Path]
) -> None:
    """Measure per-turn TTFT for the hybrid handoff on a live MultiConnector server.

    Turn 1 sends the logical document prompt with the Cachet handoff, so the document
    KV is injected by Cachet. Each follow-up turn appends the running conversation and
    omits the handoff, so Cachet advertises no tokens and LMCache (the second connector)
    serves the continuation from the resident conversation KV.
    """

    followups = DEFAULT_MULTI_TURN_FOLLOWUPS
    suite = load_v1_jsonl_suite(
        suite_id=f"{config.benchmark_id}-multiturn",
        paths=dataset_paths,
        model_id=SERVED_MODEL_NAME,
        hardware_target=config.hardware_target,
    )
    url = f"{config.server_base_url}/v1/completions"
    # Round-robin by turn (all turn-1s, then all turn-2s, ...). Processing every
    # conversation's turn N before any turn N+1 means each conversation's KV is evicted
    # from the small GPU cache by the other conversations before its follow-up runs, so
    # follow-ups must reload the conversation KV from LMCache's CPU-RAM / NVMe tier
    # instead of finding it resident in GPU HBM -- the limited-GPU, many-active-conversation
    # regime. Turn 1 carries the Cachet document handoff; follow-ups omit it (-> LMCache).
    convs: list[dict[str, object]] = [
        {
            "example": example,
            "prompt": build_prompt_parts(example).prefill_prompt,
            "output": "",
            "base_id": f"cachet-mt:{config.benchmark_id}:{example.dataset}:{example.example_id}",
            "turns": [],
            "failed": False,
        }
        for example in suite.examples
    ]
    num_turns = 1 + len(followups)
    ok = True
    for turn_index in range(1, num_turns + 1):
        for conv in convs:
            if conv["failed"]:
                continue
            example = conv["example"]
            try:
                if turn_index == 1:
                    kv_transfer_params = dict(example.kv_transfer_params)
                    kv_transfer_params[DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM] = "logical"
                    body: dict[str, object] = {
                        "model": SERVED_MODEL_NAME,
                        "prompt": conv["prompt"],
                        "max_tokens": config.max_tokens,
                        "temperature": 0,
                        "request_id": f"{conv['base_id']}:t1",
                        "kv_transfer_params": kv_transfer_params,
                    }
                    served_by, has_handoff = "cachet", True
                else:
                    conv["prompt"] = build_multi_turn_followup_prompt(
                        conv["prompt"], conv["output"], followups[turn_index - 2]
                    )
                    body = {
                        "model": SERVED_MODEL_NAME,
                        "prompt": conv["prompt"],
                        "max_tokens": config.max_tokens,
                        "temperature": 0,
                        "request_id": f"{conv['base_id']}:t{turn_index}",
                    }
                    served_by, has_handoff = "lmcache", False
                if config.force_max_tokens:
                    body["ignore_eos"] = True
                ttft, output_text, usage = _stream_completion_ttft(
                    url, body, timeout_seconds=config.timeout_seconds
                )
                conv["output"] = output_text
                conv["turns"].append(
                    {
                        "turn": turn_index,
                        "served_by": served_by,
                        "has_document_handoff": has_handoff,
                        "ttft_seconds": ttft,
                        "prompt_tokens": usage.get("prompt_tokens")
                        if isinstance(usage, Mapping)
                        else None,
                    }
                )
            except Exception as exc:
                ok = False
                conv["failed"] = True
                conv["turns"].append(
                    {
                        "turn": turn_index,
                        "error": str(exc) or type(exc).__name__,
                        "error_type": type(exc).__name__,
                    }
                )
    conversations = [
        {
            "dataset": conv["example"].dataset,
            "example_id": conv["example"].example_id,
            "turns": conv["turns"],
        }
        for conv in convs
    ]
    record = {
        "ok": ok,
        "benchmark_id": config.benchmark_id,
        "server_base_url": config.server_base_url,
        "turn_order": "round_robin_by_turn",
        "followups": list(followups),
        "conversations": conversations,
        "conversation_count": len(conversations),
    }
    output_path = config.output_dir / "multi-turn-latency.json"
    write_json(output_path, record)
    # Lenient: preserve partial results (an expensive cloud run) rather than aborting on
    # a single transient turn error; per-turn errors are recorded for the analysis.


def _post_json(
    url: str, body: Mapping[str, object], *, timeout_seconds: float
) -> Mapping[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {tail_text(detail)}") from exc
    if not isinstance(decoded, Mapping):
        raise RuntimeError(f"POST {url} returned non-object JSON")
    return decoded


def benchmark_failure_summary(output_path: Path, *, limit: int = 3) -> str:
    if not output_path.exists():
        return f"benchmark output {output_path} was not written"
    try:
        record = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"benchmark output {output_path} could not be read: {exc}"

    measurements = record.get("measurements")
    if not isinstance(measurements, list):
        return f"benchmark output {output_path} did not include measurements"
    errors = [
        _benchmark_error_summary(measurement)
        for measurement in measurements
        if isinstance(measurement, dict) and measurement.get("error")
    ]
    if not errors:
        return f"benchmark output {output_path} did not include row errors"

    issue_count = len(errors)
    shown = "; ".join(errors[:limit])
    if issue_count > limit:
        shown = f"{shown}; ... {issue_count - limit} more"
    return f"benchmark output had {issue_count}/{len(measurements)} errored measurements: {shown}"


def _benchmark_error_summary(
    measurement: dict[str, object], *, max_chars: int = 400
) -> str:
    dataset = measurement.get("dataset") or "unknown-dataset"
    arm_id = measurement.get("arm_id") or "unknown-arm"
    error = str(measurement.get("error") or "unknown error")
    if len(error) > max_chars:
        error = error[: max_chars - 3] + "..."
    return f"{dataset}/{arm_id}: {error}"


@dataclass(frozen=True, slots=True)
class _CopiedVenvPythonBinding:
    """Stable no-follow identity for one launch-owned copied interpreter."""

    path: str
    device: int
    inode: int
    mode: int
    link_count: int
    size: int
    modified_time_ns: int
    changed_time_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _IsolatedPythonIdentity:
    """Interpreter and file identity observed inside one copied virtualenv."""

    file_binding: _CopiedVenvPythonBinding
    python_implementation: str
    python_version: str
    executable: str
    prefix: str
    base_prefix: str


def _no_follow_directory_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if not isinstance(no_follow, int) or not isinstance(directory, int):
        raise RuntimeError(
            "copied virtualenv validation requires O_NOFOLLOW and O_DIRECTORY"
        )
    return os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)


def _open_absolute_directory_no_follow(path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError("copied virtualenv root must be one canonical absolute path")
    flags = _no_follow_directory_flags()
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _copied_venv_python_binding(
    runtime_root: Path,
    runtime_python: Path | None = None,
) -> _CopiedVenvPythonBinding:
    """Open and hash the exact regular ``bin/python`` without following links."""

    expected_python = runtime_root / "bin" / "python"
    candidate = expected_python if runtime_python is None else runtime_python
    if candidate != expected_python:
        raise RuntimeError("isolated runtime Python is outside its exact runtime root")
    if not runtime_root.is_absolute() or ".." in runtime_root.parts:
        raise RuntimeError("copied virtualenv root must be one canonical absolute path")
    try:
        if runtime_root.resolve(strict=True) != runtime_root:
            raise RuntimeError(
                "copied virtualenv root must not contain symlink ancestors"
            )
        root_descriptor = _open_absolute_directory_no_follow(runtime_root)
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError("copied virtualenv root is not a real directory") from exc

    bin_descriptor = -1
    python_descriptor = -1
    try:
        bin_descriptor = os.open(
            "bin",
            _no_follow_directory_flags(),
            dir_fd=root_descriptor,
        )
        pre_open = os.stat(
            "python",
            dir_fd=bin_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(pre_open.st_mode):
            raise RuntimeError("isolated runtime Python must be one regular file")
        if pre_open.st_nlink != 1:
            raise RuntimeError("isolated runtime Python must have exactly one link")
        if not pre_open.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            raise RuntimeError("isolated runtime Python must be executable")
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW")
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        python_descriptor = os.open(
            "python",
            file_flags,
            dir_fd=bin_descriptor,
        )
        opened_before = os.fstat(python_descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise RuntimeError("isolated runtime Python must be one regular file")
        if opened_before.st_nlink != 1:
            raise RuntimeError("isolated runtime Python must have exactly one link")
        if not opened_before.st_mode & (
            stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        ):
            raise RuntimeError("isolated runtime Python must be executable")
        digest = sha256()
        while True:
            chunk = os.read(python_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(python_descriptor)
        post_read = os.stat(
            "python",
            dir_fd=bin_descriptor,
            follow_symlinks=False,
        )

        def stable_identity(item: os.stat_result) -> tuple[int, ...]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_nlink,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        if len(
            {
                stable_identity(item)
                for item in (pre_open, opened_before, opened_after, post_read)
            }
        ) != 1:
            raise RuntimeError("isolated runtime Python changed while it was hashed")
        return _CopiedVenvPythonBinding(
            path=str(candidate),
            device=opened_after.st_dev,
            inode=opened_after.st_ino,
            mode=stat.S_IMODE(opened_after.st_mode),
            link_count=opened_after.st_nlink,
            size=opened_after.st_size,
            modified_time_ns=opened_after.st_mtime_ns,
            changed_time_ns=opened_after.st_ctime_ns,
            sha256=digest.hexdigest(),
        )
    except OSError as exc:
        raise RuntimeError(
            "isolated runtime Python must be a no-follow regular executable"
        ) from exc
    finally:
        if python_descriptor >= 0:
            os.close(python_descriptor)
        if bin_descriptor >= 0:
            os.close(bin_descriptor)
        os.close(root_descriptor)


def _isolated_python_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = _pip_subprocess_environment(environ)
    for variable_name in tuple(environment):
        if variable_name.startswith("PYTHON"):
            environment.pop(variable_name)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
    )
    return environment


def _attest_isolated_python(
    runtime_root: Path,
    *,
    expected_python_version: str,
    environment: Mapping[str, str] | None = None,
    expected_file_binding: _CopiedVenvPythonBinding | None = None,
) -> _IsolatedPythonIdentity:
    """Attest a copied interpreter under the caller's exclusive ownership.

    The pre/post no-follow snapshots detect replacement or mutation during the
    probe.  Like the site-packages freezer, this boundary assumes no hostile
    process sharing the same UID concurrently modifies the launch-owned tree.
    """

    runtime_python = runtime_root / "bin" / "python"
    before = _copied_venv_python_binding(runtime_root, runtime_python)
    if expected_file_binding is not None and before != expected_file_binding:
        raise RuntimeError("isolated runtime Python differs from its bound identity")
    probe = (
        "import json,sys; "
        "print(json.dumps({'base_prefix':sys.base_prefix,"
        "'executable':sys.executable,'prefix':sys.prefix,"
        "'python_implementation':sys.implementation.name,"
        "'python_version':'.'.join(map(str,sys.version_info[:3]))},"
        "sort_keys=True))"
    )
    completed = subprocess.run(
        [str(runtime_python), "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=_isolated_python_environment(environment),
        cwd=runtime_root,
    )
    after = _copied_venv_python_binding(runtime_root, runtime_python)
    if after != before:
        raise RuntimeError("isolated runtime Python changed during identity probe")
    try:
        observed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("isolated runtime Python identity was not JSON") from exc
    if not isinstance(observed, dict) or set(observed) != {
        "base_prefix",
        "executable",
        "prefix",
        "python_implementation",
        "python_version",
    }:
        raise RuntimeError("isolated runtime Python identity has an open schema")
    if any(type(observed.get(key)) is not str for key in observed):
        raise RuntimeError("isolated runtime Python identity fields must be strings")
    executable = str(observed["executable"])
    prefix = str(observed["prefix"])
    base_prefix = str(observed["base_prefix"])
    python_implementation = str(observed["python_implementation"])
    python_version = str(observed["python_version"])
    if executable != str(runtime_python):
        raise RuntimeError("isolated runtime Python reported the wrong sys.executable")
    if prefix != str(runtime_root):
        raise RuntimeError("isolated runtime Python reported the wrong sys.prefix")
    if base_prefix == prefix:
        raise RuntimeError("isolated runtime Python did not separate sys.base_prefix")
    if python_implementation != "cpython":
        raise RuntimeError("isolated runtime Python is not CPython")
    if python_version != expected_python_version:
        raise RuntimeError(
            "isolated runtime Python version differs from the qualification plan: "
            f"{python_version!r} != {expected_python_version!r}"
        )
    return _IsolatedPythonIdentity(
        file_binding=before,
        python_implementation=python_implementation,
        python_version=python_version,
        executable=executable,
        prefix=prefix,
        base_prefix=base_prefix,
    )


def create_venv(venv_dir: Path, *, copies: bool = False) -> None:
    if type(copies) is not bool:
        raise TypeError("copies must be an exact bool")
    if copies:
        if not venv_dir.is_absolute() or ".." in venv_dir.parts:
            raise RuntimeError("copied virtualenv root must be one canonical absolute path")
        try:
            if venv_dir.parent.resolve(strict=True) != venv_dir.parent:
                raise RuntimeError(
                    "copied virtualenv root must not contain symlink ancestors"
                )
            if venv_dir.exists() or venv_dir.is_symlink():
                root_status = venv_dir.stat(follow_symlinks=False)
                if not stat.S_ISDIR(root_status.st_mode):
                    raise RuntimeError("copied virtualenv root must be a real directory")
                if venv_dir.resolve(strict=True) != venv_dir:
                    raise RuntimeError(
                        "copied virtualenv root must not contain symlink ancestors"
                    )
        except OSError as exc:
            raise RuntimeError("copied virtualenv parent must be a real directory") from exc
    python = venv_python(venv_dir)
    if python.exists() or python.is_symlink():
        if copies:
            _copied_venv_python_binding(venv_dir, python)
        return
    pip_environment = _pip_subprocess_environment()
    copy_argument = ["--copies"] if copies else []
    try:
        run(
            [sys.executable, "-m", "venv", *copy_argument, str(venv_dir)],
            env=pip_environment,
        )
    except subprocess.CalledProcessError:
        bootstrap = materialize_virtualenv_bootstrap(venv_dir.parent)
        run(
            [sys.executable, str(bootstrap), *copy_argument, str(venv_dir)],
            env=pip_environment,
        )
    if copies:
        _copied_venv_python_binding(venv_dir, python)


def materialize_virtualenv_bootstrap(output_dir: Path) -> Path:
    """Download and verify the reviewed self-contained virtualenv zipapp."""

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / (
        f"virtualenv-{VIRTUALENV_BOOTSTRAP_VERSION}-"
        f"{VIRTUALENV_BOOTSTRAP_SHA256[:16]}.pyz"
    )
    if target.exists():
        observed = _file_sha256(target)
        if observed != VIRTUALENV_BOOTSTRAP_SHA256:
            raise RuntimeError(
                f"virtualenv bootstrap SHA-256 {observed} does not match "
                f"{VIRTUALENV_BOOTSTRAP_SHA256}"
            )
        return target

    temporary = target.with_suffix(".tmp")
    request = urllib.request.Request(
        VIRTUALENV_BOOTSTRAP_URL,
        headers={"User-Agent": "cachet-vllm-runtime-bootstrap"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        observed = _file_sha256(temporary)
        if observed != VIRTUALENV_BOOTSTRAP_SHA256:
            raise RuntimeError(
                f"downloaded virtualenv bootstrap SHA-256 {observed} does not match "
                f"{VIRTUALENV_BOOTSTRAP_SHA256}"
            )
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _pip_subprocess_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a pip environment with no inherited option or config authority."""

    environment = dict(os.environ if environ is None else environ)
    for variable_name in tuple(environment):
        if variable_name.upper().startswith(("PIP_", "_PIP_")):
            environment.pop(variable_name)
    for ambient_path_variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(ambient_path_variable, None)
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def install_vllm(python_executable: Path) -> None:
    validate_vllm_runtime_lock_platform()
    runtime_lock = vllm_runtime_lock_path()
    pip_environment = _pip_subprocess_environment()
    run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--only-binary",
            ":all:",
            "--requirement",
            str(runtime_lock),
        ],
        env=pip_environment,
    )
    run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--no-deps",
            patched_vllm_wheel_install_spec(),
        ],
        env=pip_environment,
    )


_NATIVE_RUNTIME_V2_ARTIFACT_NAMES = (
    "runtime_lock",
    "patched_vllm_wheel",
    "patched_flashinfer_wheel",
    "runtime_closure_manifest",
    "package_wheel",
)
_NATIVE_RUNTIME_V2_INSTALL_TIMEOUT_SECONDS = 3_600
_NATIVE_RUNTIME_V2_PIP_CHECK_TIMEOUT_SECONDS = 300
_NATIVE_RUNTIME_V2_FINAL_VERIFIER_TIMEOUT_SECONDS = 300
_NATIVE_RUNTIME_V2_PIP_CHECK_STDOUT = "No broken requirements found.\n"


def _native_runtime_v2_artifact_snapshot(
    bundle: VLLMNativeRuntimeBundleV2,
) -> tuple[dict[str, Path], tuple[tuple[object, ...], ...]]:
    paths: dict[str, Path] = {}
    rows: list[tuple[object, ...]] = []
    for artifact in _NATIVE_RUNTIME_V2_ARTIFACT_NAMES:
        path = bundle.local_path(artifact)
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise RuntimeError(
                f"native-v2 runtime artifact is unavailable: {artifact}"
            ) from exc
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size <= 0:
            raise RuntimeError(
                f"native-v2 runtime artifact must be one non-empty regular file: {artifact}"
            )
        expected_sha256 = getattr(bundle, f"{artifact}_sha256")
        observed_sha256 = _file_sha256(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"native-v2 runtime artifact SHA-256 differs: {artifact}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                f"native-v2 runtime artifact cannot be resolved: {artifact}"
            ) from exc
        paths[artifact] = resolved
        rows.append(
            (
                artifact,
                str(path),
                str(resolved),
                info.st_dev,
                info.st_ino,
                stat.S_IMODE(info.st_mode),
                info.st_uid,
                info.st_gid,
                info.st_nlink,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
                observed_sha256,
            )
        )
    if len({str(path) for path in paths.values()}) != len(paths):
        raise RuntimeError("native-v2 runtime artifacts resolve to duplicate files")
    return paths, tuple(rows)


def _native_v2_direct_reference(
    distribution: str,
    path: Path,
    expected_sha256: str,
) -> str:
    return f"{distribution} @ {path.as_uri()}#sha256={expected_sha256}"


def _run_native_v2_final_runtime_verifier(
    python_executable: Path,
    *,
    paths: Mapping[str, Path],
    bundle: VLLMNativeRuntimeBundleV2,
    environment: Mapping[str, str],
    cwd: Path,
) -> dict[str, Any]:
    code = (
        "import json,sys;"
        "from document_kv_cache._gpu_qualification_sentinels_v2 import "
        "verify_gpu_qualification_v2_runtime_installation as verify;"
        "record=verify(runtime_lock=sys.argv[1],vllm_uri=sys.argv[2],"
        "flashinfer_uri=sys.argv[3],runtime_closure_manifest=sys.argv[4],"
        "package_uri=sys.argv[5],package_sha256=sys.argv[6]);"
        "print(json.dumps(record,allow_nan=False,ensure_ascii=True,"
        "separators=(',',':'),sort_keys=True))"
    )
    vllm_uri = paths["patched_vllm_wheel"].as_uri()
    flashinfer_uri = paths["patched_flashinfer_wheel"].as_uri()
    package_uri = paths["package_wheel"].as_uri()
    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            code,
            str(paths["runtime_lock"]),
            vllm_uri,
            flashinfer_uri,
            str(paths["runtime_closure_manifest"]),
            package_uri,
            bundle.package_wheel_sha256,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=_NATIVE_RUNTIME_V2_FINAL_VERIFIER_TIMEOUT_SECONDS,
        env=dict(environment),
        cwd=cwd,
    )
    if completed.stderr != "":
        raise RuntimeError("native-v2 final runtime verifier wrote stderr")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("native-v2 final runtime verifier output is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError("native-v2 final runtime verifier returned no record")
    canonical_stdout = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    if completed.stdout != canonical_stdout:
        raise RuntimeError("native-v2 final runtime verifier output is not canonical")
    from document_kv_cache.gpu_qualification_v2 import (
        validate_gpu_qualification_v2_runtime_attestation,
    )

    validate_gpu_qualification_v2_runtime_attestation(value)
    if (
        value.get("vllm_direct_url") != vllm_uri
        or value.get("flashinfer_direct_url") != flashinfer_uri
    ):
        raise RuntimeError("native-v2 final runtime verifier origin differs")
    return value


def install_native_v2_runtime(
    config: VLLMSmokeBenchmarkConfig,
) -> dict[str, Any]:
    """Install and attest the complete native-v2 publication runtime."""

    bundle = config.native_runtime_v2
    if bundle is None:
        raise ValueError("native_runtime_v2 is required")
    paths, opening_snapshot = _native_runtime_v2_artifact_snapshot(bundle)
    environment = _pip_subprocess_environment()
    environment["PYTHONSAFEPATH"] = "1"
    commands = [
        [
            str(config.venv_python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--only-binary",
            ":all:",
            "--requirement",
            str(paths["runtime_lock"]),
        ],
        [
            str(config.venv_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            _native_v2_direct_reference(
                "vllm",
                paths["patched_vllm_wheel"],
                bundle.patched_vllm_wheel_sha256,
            ),
        ],
        [
            str(config.venv_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            _native_v2_direct_reference(
                "flashinfer-python",
                paths["patched_flashinfer_wheel"],
                bundle.patched_flashinfer_wheel_sha256,
            ),
        ],
        [
            str(config.venv_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            _native_v2_direct_reference(
                "cachet-kv",
                paths["package_wheel"],
                bundle.package_wheel_sha256,
            ),
        ],
    ]
    for command in commands:
        print("+", " ".join(command), flush=True)
        subprocess.run(
            command,
            check=True,
            timeout=_NATIVE_RUNTIME_V2_INSTALL_TIMEOUT_SECONDS,
            env=environment,
            cwd=config.venv_dir,
        )
    pip_check_command = [str(config.venv_python), "-m", "pip", "check"]
    print("+", " ".join(pip_check_command), flush=True)
    pip_check = subprocess.run(
        pip_check_command,
        check=True,
        capture_output=True,
        text=True,
        timeout=_NATIVE_RUNTIME_V2_PIP_CHECK_TIMEOUT_SECONDS,
        env=environment,
        cwd=config.venv_dir,
    )
    if (
        pip_check.stdout != _NATIVE_RUNTIME_V2_PIP_CHECK_STDOUT
        or pip_check.stderr != ""
    ):
        raise RuntimeError("native-v2 pip check output differs")
    attestation = _run_native_v2_final_runtime_verifier(
        config.venv_python,
        paths=paths,
        bundle=bundle,
        environment=environment,
        cwd=config.venv_dir,
    )
    closing_paths, closing_snapshot = _native_runtime_v2_artifact_snapshot(bundle)
    if closing_paths != paths or closing_snapshot != opening_snapshot:
        raise RuntimeError("native-v2 runtime artifacts changed during installation")
    return attestation


def install_document_kv_package(python_executable: Path, install_spec: str) -> None:
    run(
        [str(python_executable), "-m", "pip", "install", "--no-deps", install_spec],
        env=_pip_subprocess_environment(),
    )


def verify_vllm_runtime_patch_closure(
    config: VLLMSmokeBenchmarkConfig,
) -> list[dict[str, object]]:
    """Read-only preflight for the prepatched vLLM 0.27.1 cu129 runtime."""

    observed_version = installed_package_version(config.venv_python, "vllm")
    if observed_version != VLLM_PACKAGE_VERSION:
        raise RuntimeError(
            f"patched runtime requires vLLM {VLLM_PACKAGE_VERSION}, "
            f"found {observed_version}"
        )

    records: list[dict[str, object]] = []
    for patch in _VLLM_0271_E5M2_PATCH_CLOSURE:
        relative_path = Path(str(patch["relative_path"]))
        candidates = [
            site_packages / relative_path
            for site_packages in site_packages_dirs(config)
        ]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise RuntimeError(
                f"Expected one installed {relative_path}, found {existing!r}"
            )
        path = existing[0]
        raw = path.read_bytes()
        digest = validate_patched_vllm_member_bytes(
            relative_path.as_posix(),
            raw,
            patch_closure=_VLLM_0271_E5M2_PATCH_CLOSURE,
        )
        records.append(
            {
                "id": patch["id"],
                "base_version": VLLM_VERSION,
                "package_version": VLLM_PACKAGE_VERSION,
                "wheel_sha256": (
                    config.native_runtime_v2.patched_vllm_wheel_sha256
                    if config.native_runtime_v2 is not None
                    else os.environ.get(VLLM_PATCHED_WHEEL_SHA256_ENV)
                ),
                "path": str(path),
                "verified": True,
                "installed_sha256": digest,
                "expected_patched_sha256": patch["patched_sha256"],
                "reason": patch["reason"],
            }
        )
    return records


def installed_package_version(python_executable: Path, package_name: str) -> str:
    completed = subprocess.run(
        [str(python_executable), "-m", "pip", "show", package_name],
        check=True,
        capture_output=True,
        text=True,
        env=_pip_subprocess_environment(),
    )
    for line in completed.stdout.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def probe_vllm_import(
    python_executable: Path,
    output_path: Path,
    *,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
) -> None:
    expected_versions = {
        "flashinfer-python": FLASHINFER_PYTHON_CONSTRAINT.split("==", 1)[1],
        "torch": TORCH_CONSTRAINT.split("==", 1)[1],
        "torchaudio": TORCHAUDIO_CONSTRAINT.split("==", 1)[1],
        "torchcodec": TORCHCODEC_CONSTRAINT.split("==", 1)[1],
        "torchvision": TORCHVISION_CONSTRAINT.split("==", 1)[1],
        "triton": TRITON_CONSTRAINT.split("==", 1)[1],
        "vllm": VLLM_PACKAGE_VERSION,
    }
    code = """
import importlib.metadata as md
import json
import subprocess
import torch
import torchaudio
import torchcodec
import torchvision
import triton
import triton.language as tl
import flashinfer
import vllm
import vllm.entrypoints.openai.api_server
import document_kv_cache
import vllm_kv_injection.vllm_dynamic_connector as document_kv_vllm_connector
from document_kv_cache.vllm_smoke import build_vllm_native_provider_probe_record
from vllm_kv_injection.vllm_transfer_config import document_kv_transfer_config

EXPECTED_VERSIONS = __EXPECTED_VERSIONS__
observed_versions = {
    distribution: md.version(distribution)
    for distribution in EXPECTED_VERSIONS
}
version_mismatches = {
    distribution: {
        "expected": expected,
        "observed": observed_versions[distribution],
    }
    for distribution, expected in EXPECTED_VERSIONS.items()
    if observed_versions[distribution] != expected
}
if version_mismatches:
    raise RuntimeError(
        "vLLM cu129 runtime package mismatch: "
        + json.dumps(version_mismatches, sort_keys=True)
    )
if torch.version.cuda != "12.9":
    raise RuntimeError(
        f"vLLM cu129 runtime requires torch.version.cuda == '12.9', "
        f"found {torch.version.cuda!r}"
    )
if not torch.cuda.is_available():
    raise RuntimeError("vLLM cu129 runtime preflight requires an available CUDA GPU")

@triton.jit
def _cachet_triton_probe(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, values + 1.0, mask=mask)

triton_input = torch.arange(256, device="cuda", dtype=torch.float32)
triton_output = torch.empty_like(triton_input)
_cachet_triton_probe[(1,)](
    triton_input,
    triton_output,
    triton_input.numel(),
    BLOCK_SIZE=256,
)
torch.cuda.synchronize()
if not torch.equal(triton_output, triton_input + 1.0):
    raise RuntimeError("Triton JIT probe returned incorrect values")

nvidia_smi = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=driver_version,name",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)
transfer_config = document_kv_transfer_config()
payload = {
    "ok": True,
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "vllm_version": md.version("vllm"),
    "runtime_package_versions": observed_versions,
    "nvidia_smi_driver_and_devices": nvidia_smi.stdout.strip().splitlines(),
    "triton_jit_probe_ok": True,
    "document_kv_cache_version": md.version("cachet-kv"),
    "document_kv_cache_module": document_kv_cache.__name__,
    "document_kv_connector_module": document_kv_vllm_connector.__name__,
    "document_kv_connector": transfer_config["kv_connector"],
}
payload.update(build_vllm_native_provider_probe_record(transfer_config))
if torch.cuda.is_available():
    payload["cuda_device_name"] = torch.cuda.get_device_name(0)
    payload["cuda_device_capability"] = list(torch.cuda.get_device_capability(0))
print(json.dumps(payload, sort_keys=True), flush=True)
""".replace("__EXPECTED_VERSIONS__", repr(expected_versions))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probe_script_path = output_path.with_name(f"{output_path.stem}.py")
    probe_script_path.write_text(code, encoding="utf-8")
    argv = [str(python_executable), str(probe_script_path)]
    print("+", " ".join([argv[0], "<vllm import probe>"]), flush=True)
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env or os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        record = {
            "ok": False,
            "error_type": "TimeoutExpired",
            "error": f"vLLM import probe timed out after {timeout_seconds:.1f}s",
            "stdout_tail": tail_text(exc.stdout),
            "stderr_tail": tail_text(exc.stderr),
        }
        write_json(output_path, record)
        raise RuntimeError(record["error"]) from exc

    record = {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_tail": tail_text(completed.stdout),
        "stderr_tail": tail_text(completed.stderr),
    }
    if completed.returncode == 0:
        record.update(last_json_object(completed.stdout))
    write_json(output_path, record)
    if completed.returncode != 0:
        raise RuntimeError(
            f"vLLM import probe failed with return code {completed.returncode}"
        )


def last_json_object(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def write_smoke_datasets(local_dir: Path) -> dict[str, Path]:
    paths = {}
    for dataset, record in smoke_dataset_records().items():
        path = local_dir / f"{dataset}.jsonl"
        path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        paths[dataset] = path
    return paths


def prepare_publication_latency_inputs(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: dict[str, Path],
) -> dict[str, Path]:
    """Validate the closed schedule and optionally stage reusable Vanilla KV.

    Baseline cells retain ``dataset_paths`` verbatim.  Vanilla cells replace them
    only with the paths returned by the closed-bundle staging API.
    """

    if not config.uses_publication_latency_schedule:
        return dataset_paths
    expected_bundle_sha256 = config.publication_latency_expected_input_bundle_sha256
    assert expected_bundle_sha256 is not None
    runner_dataset_paths = dataset_paths
    if config.stages_publication_handoffs:
        generation_root = config.publication_handoff_generation_output_root
        local_nvme_dir = config.publication_handoff_local_nvme_dir
        assert local_nvme_dir is not None
        input_tokens_target = config.benchmark_manifest_provenance.get(
            "input_tokens_target"
        )
        if type(input_tokens_target) is not int:
            raise ValueError(
                "publication handoff staging requires integer input_tokens_target"
            )
        if generation_root is not None:
            execution_file_sha256 = (
                config.publication_handoff_generation_execution_file_sha256
            )
            execution_closed_record_sha256 = (
                config.publication_handoff_generation_execution_closed_record_sha256
            )
            assert execution_file_sha256 is not None
            assert execution_closed_record_sha256 is not None
            execution_path = generation_root / (
                PUBLICATION_LATENCY_HANDOFF_EXECUTION_FILENAME
            )
            if _file_sha256(execution_path) != execution_file_sha256:
                raise ValueError(
                    "publication handoff generation execution file SHA drift"
                )
            generation_result = read_publication_latency_handoff_generation_result(
                generation_root
            )
            if generation_result.record.get("closed_record_sha256") != (
                execution_closed_record_sha256
            ) or generation_result.record.get("execution_mode") != (
                PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED
            ):
                raise ValueError(
                    "publication handoff generation execution binding drift"
                )
            serving_bundle = resolve_publication_latency_worker_handoff_bundle(
                generation_result,
                context_tokens=input_tokens_target,
            )
            manifest = serving_bundle.manifest
            source_root = serving_bundle.source_root
        else:
            manifest_path = config.publication_handoff_bundle_manifest_path
            bundle_source_root = config.publication_handoff_bundle_source_root
            manifest_file_sha256 = (
                config.publication_handoff_bundle_manifest_file_sha256
            )
            manifest_closed_record_sha256 = (
                config.publication_handoff_bundle_manifest_closed_record_sha256
            )
            assert manifest_path is not None
            assert bundle_source_root is not None
            assert manifest_file_sha256 is not None
            assert manifest_closed_record_sha256 is not None
            source_root = bundle_source_root
            if _file_sha256(manifest_path) != manifest_file_sha256:
                raise ValueError("publication handoff bundle manifest file SHA drift")
            manifest = read_publication_latency_handoff_bundle(manifest_path)
            validate_publication_latency_handoff_bundle(
                manifest,
                bundle_root=source_root,
            )
            if manifest.get("closed_record_sha256") != (manifest_closed_record_sha256):
                raise ValueError("publication handoff bundle manifest record SHA drift")
        if manifest.get("input_bundle_sha256") != expected_bundle_sha256:
            raise ValueError(
                "publication handoff manifest input_bundle_sha256 does not match "
                "the publication latency schedule input bundle"
            )
        if manifest.get("context_tokens") != input_tokens_target:
            raise ValueError(
                "publication handoff manifest context_tokens does not match "
                "benchmark_manifest_provenance.input_tokens_target"
            )
        cache_methods = {
            entry.get("cache_method")
            for dataset in manifest.get("datasets", ())
            for entry in dataset.get("entries", ())
        }
        if cache_methods != {CacheGenerationMethod.VANILLA_PREFILL.value}:
            raise ValueError(
                "publication handoff manifest must contain only vanilla_prefill "
                "artifacts"
            )
        staged = stage_publication_latency_handoff_bundle(
            manifest,
            source_root=source_root,
            local_nvme_dir=local_nvme_dir,
        )
        _persist_publication_handoff_staging_attestation(
            staged.attestation_path,
            config.publication_handoff_staging_attestation_copy_path,
        )
        runner_dataset_paths = staged.dataset_paths

    schedule = _publication_latency_schedule_record(config)
    suite = load_v1_jsonl_suite(
        suite_id=config.resolved_benchmark_suite_id,
        paths=runner_dataset_paths,
        model_id=SERVED_MODEL_NAME,
        hardware_target=config.hardware_target,
    )
    validate_publication_latency_block_schedule(
        schedule,
        examples=tuple(
            PublicationLatencyExample(
                dataset=example.dataset,
                example_id=example.example_id,
            )
            for example in suite.examples
        ),
        expected_input_bundle_sha256=expected_bundle_sha256,
    )
    _publication_latency_schedule_runner_path(config, schedule=schedule)
    return runner_dataset_paths


def _publication_latency_schedule_record(
    config: VLLMSmokeBenchmarkConfig,
) -> dict[str, Any]:
    record = config.publication_latency_schedule_record
    if record is not None:
        return _normalized_json_object(
            record,
            "publication_latency_schedule_record",
        )
    path = config.publication_latency_schedule_path
    if path is None:
        raise ValueError("publication latency schedule is not configured")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not load publication latency schedule from {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("publication latency schedule JSON must contain an object")
    return _normalized_json_object(value, "publication_latency_schedule_path")


def _publication_latency_schedule_runner_path(
    config: VLLMSmokeBenchmarkConfig,
    *,
    schedule: Mapping[str, Any] | None = None,
) -> Path | None:
    if not config.uses_publication_latency_schedule:
        return None
    if config.publication_latency_schedule_path is not None:
        return config.publication_latency_schedule_path
    record = (
        _publication_latency_schedule_record(config)
        if schedule is None
        else _normalized_json_object(schedule, "publication_latency_schedule_record")
    )
    path = config.publication_latency_schedule_materialized_path
    content = (
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(
                "refusing to replace a different materialized publication latency "
                "schedule"
            )
        return path
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(
                "concurrent materialized publication latency schedule differs"
            ) from None
    return path


def _persist_publication_handoff_staging_attestation(
    source_path: Path,
    target_path: Path,
) -> None:
    content = source_path.read_bytes()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        if target_path.read_bytes() != content:
            raise FileExistsError(
                "refusing to overwrite a different publication handoff staging "
                f"attestation: {target_path}"
            )
        return
    try:
        with target_path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if target_path.read_bytes() != content:
            raise FileExistsError(
                "concurrent publication handoff staging attestation differs: "
                f"{target_path}"
            ) from None


def benchmark_dataset_paths(config: VLLMSmokeBenchmarkConfig) -> dict[str, Path]:
    if config.dataset_specs:
        return parse_dataset_specs(
            config.dataset_specs, allow_subset=config.allow_dataset_subset
        )
    return write_smoke_datasets(config.local_dir)


def parse_dataset_specs(
    dataset_specs: tuple[str, ...], *, allow_subset: bool = False
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in dataset_specs:
        dataset, separator, raw_path = spec.partition("=")
        if not separator or not dataset or not raw_path:
            raise ValueError("dataset specs must use DATASET=JSONL_PATH syntax")
        if dataset not in SMOKE_DATASETS:
            raise ValueError(f"Unsupported V1 smoke dataset {dataset!r}")
        if dataset in paths:
            raise ValueError(f"duplicate dataset spec for {dataset!r}")
        paths[dataset] = Path(_cluster_file_path(raw_path))
    missing = set(SMOKE_DATASETS).difference(paths)
    if missing and not allow_subset:
        raise ValueError(
            f"dataset specs missing required V1 datasets: {sorted(missing)}"
        )
    return {dataset: paths[dataset] for dataset in SMOKE_DATASETS if dataset in paths}


def smoke_dataset_records() -> dict[str, dict[str, object]]:
    return {
        "biography": {
            "example_id": "biography-smoke-1",
            "dataset": "biography",
            "query": "Which entity is described in the biography?",
            "expected_answer": "Katherine Johnson",
            "documents": [
                {
                    "document_id": "katherine-johnson",
                    "title": "Katherine Johnson",
                    "text": (
                        "Katherine Johnson was a NASA mathematician whose orbital mechanics calculations "
                        "supported early crewed spaceflight missions."
                    ),
                }
            ],
        },
        "hotpotqa": {
            "example_id": "hotpotqa-smoke-1",
            "dataset": "hotpotqa",
            "query": "The landmark discussed in the first document is located in which city?",
            "expected_answer": "Paris",
            "documents": [
                {
                    "document_id": "eiffel-tower",
                    "title": "Eiffel Tower",
                    "text": "The Eiffel Tower is a wrought-iron landmark on the Champ de Mars.",
                },
                {
                    "document_id": "paris",
                    "title": "Paris",
                    "text": "The Champ de Mars is a large public greenspace in Paris, France.",
                },
            ],
        },
        "musique": {
            "example_id": "musique-smoke-1",
            "dataset": "musique",
            "query": "Who is the mathematician connected to the engine described by Charles Babbage?",
            "expected_answer": "Ada Lovelace",
            "documents": [
                {
                    "document_id": "analytical-engine",
                    "title": "Analytical Engine",
                    "text": "Charles Babbage designed the Analytical Engine as a proposed mechanical computer.",
                },
                {
                    "document_id": "ada-lovelace",
                    "title": "Ada Lovelace",
                    "text": "Ada Lovelace wrote notes about the Analytical Engine and is known for early computing work.",
                },
            ],
        },
        "niah": {
            "example_id": "niah-smoke-1",
            "dataset": "niah",
            "query": "What is the hidden target phrase?",
            "expected_answer": "cerulean lantern",
            "documents": [
                {
                    "document_id": "haystack",
                    "title": "Needle Haystack",
                    "text": (
                        "Most of this context is filler for the retrieval smoke test. "
                        "The hidden target phrase is cerulean lantern. "
                        "Only the exact hidden phrase should be returned."
                    ),
                }
            ],
        },
    }


def dataset_args(dataset_paths: dict[str, Path]) -> list[str]:
    args: list[str] = []
    for dataset in dataset_paths:
        args.extend(["--dataset", f"{dataset}={dataset_paths[dataset]}"])
    return args


def lmcache_transfer_config_json() -> str:
    """KVTransferConfig JSON that routes KV through LMCache's vLLM V1 connector.

    Storage-tier behaviour (CPU/disk sizes, chunk size) is supplied out-of-band
    via ``LMCACHE_*`` environment variables / ``LMCACHE_CONFIG_FILE`` so the same
    connector config works for warm-store and cold-load phases.
    """

    return json.dumps(
        {"kv_connector": LMCACHE_CONNECTOR_CLASS, "kv_role": "kv_both"},
        sort_keys=True,
    )


def cachet_transfer_config_json(config: VLLMSmokeBenchmarkConfig) -> str:
    return json.dumps(
        document_kv_transfer_config_for_smoke(config),
        separators=(",", ":"),
        sort_keys=True,
    )


def multi_transfer_config_json(config: VLLMSmokeBenchmarkConfig) -> str:
    """MultiConnector JSON that runs Cachet first, LMCache second.

    Cachet advertises matched tokens only for requests carrying a Cachet handoff
    (turn-1 document requests); everything else falls through to LMCache. Both
    receive the save-to-all broadcast so LMCache captures the conversation KV.
    """

    cachet_child = json.loads(cachet_transfer_config_json(config))
    lmcache_child = json.loads(lmcache_transfer_config_json())
    return multi_connector_transfer_config_json(
        connectors=[cachet_child, lmcache_child]
    )


def kv_transfer_config_json(config: VLLMSmokeBenchmarkConfig) -> str:
    if config.kv_connector_mode == LMCACHE_KV_CONNECTOR_MODE:
        return lmcache_transfer_config_json()
    if config.kv_connector_mode == MULTI_KV_CONNECTOR_MODE:
        return multi_transfer_config_json(config)
    return cachet_transfer_config_json(config)


def build_vllm_server_args(
    config: VLLMSmokeBenchmarkConfig, python_executable: Path
) -> list[str]:
    args = [
        str(python_executable),
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.model_id,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        config.server_host,
        "--port",
        str(config.server_port),
        "--dtype",
        config.model_dtype,
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        *(
            ["--data-parallel-size", str(config.data_parallel_size)]
            if config.data_parallel_size > 1
            else []
        ),
        "--kv-transfer-config",
        kv_transfer_config_json(config),
        # Prefix caching is enabled for the Cachet and hybrid (multi) paths so
        # turn-2+ continuation can reuse the resident conversation KV. The pure
        # LMCache arm leaves it off so the two arms are not double-cached.
        *(
            ["--enable-prefix-caching"]
            if config.kv_connector_mode != LMCACHE_KV_CONNECTOR_MODE
            else []
        ),
        # MultiConnector's Prometheus metrics path asserts that every child that
        # emits KV stats also registered prom metrics; Cachet's connector emits
        # stats but no prom metrics, so disable server-side stat logging for the
        # hybrid arm. TTFT is measured client-side, so this does not affect results.
        *(
            ["--disable-log-stats"]
            if config.kv_connector_mode == MULTI_KV_CONNECTOR_MODE
            else []
        ),
        "--no-enable-log-requests",
    ]
    if config.model_quantization is not None:
        args.extend(["--quantization", config.model_quantization])
    if config.model_revision is not None:
        args.extend(["--revision", config.model_revision])
    if config.tokenizer_revision is not None:
        args.extend(["--tokenizer-revision", config.tokenizer_revision])
    if config.kv_cache_dtype is not None:
        args.extend(["--kv-cache-dtype", config.kv_cache_dtype])
    if config.attention_backend is not None:
        args.extend(["--attention-backend", config.attention_backend])
    return args


def start_vllm_server(
    config: VLLMSmokeBenchmarkConfig, python_executable: Path, log_path: Path
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = build_vllm_server_args(config, python_executable)
    print("+", " ".join(argv), flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        return subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=server_env(config),
        )


def wait_for_server(
    server: subprocess.Popen,
    log_path: Path,
    config: VLLMSmokeBenchmarkConfig,
    *,
    timeout_seconds: float = 900.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_url = f"{config.server_base_url}/health"
    models_url = f"{config.server_base_url}/v1/models"
    last_model_error = ""
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited with {server.returncode}; log tail:\n{tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    model_ids = fetch_served_model_ids(models_url)
                    if SERVED_MODEL_NAME in model_ids:
                        return
                    last_model_error = (
                        f"health OK but served models were {sorted(model_ids)!r}"
                    )
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as exc:
            last_model_error = str(exc)
            pass
        time.sleep(5)
    raise TimeoutError(
        f"Timed out waiting for vLLM model {SERVED_MODEL_NAME!r} at {config.server_base_url}; "
        f"last readiness error: {last_model_error}; log tail:\n{tail(log_path)}"
    )


def fetch_served_model_ids(models_url: str) -> set[str]:
    with urllib.request.urlopen(models_url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        item["id"]
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def build_benchmark_runner_args(
    config: VLLMSmokeBenchmarkConfig, dataset_paths: dict[str, Path]
) -> list[str]:
    baseline_extra_body = _benchmark_extra_body(
        cache_salt=BASELINE_PREFIX_CACHE_SALT
        if config.uses_prepared_datasets
        else None,
        force_max_tokens=config.force_max_tokens,
    )
    cache_extra_body = _benchmark_extra_body(
        cache_salt=CACHE_PREFIX_CACHE_SALT if config.uses_prepared_datasets else None,
        force_max_tokens=config.force_max_tokens,
    )
    args = [
        sys.executable,
        "-m",
        "document_kv_cache.benchmark_runner",
        "--suite-id",
        config.resolved_benchmark_suite_id,
        "--base-url",
        config.server_base_url,
        "--model-id",
        SERVED_MODEL_NAME,
        "--hardware-target",
        config.hardware_target,
        "--max-tokens",
        str(config.max_tokens),
        "--temperature",
        str(config.temperature),
        "--timeout-seconds",
        str(config.timeout_seconds),
        "--repeats",
        str(config.benchmark_repeats),
        "--request-parallelism",
        str(config.request_parallelism),
        "--server-usage",
        "--output-json",
        str(config.benchmark_output_path),
    ]
    if config.generation_seed is not None:
        args.extend(["--generation-seed", str(config.generation_seed)])
    publication_schedule_path = _publication_latency_schedule_runner_path(config)
    if publication_schedule_path is not None:
        expected_bundle_sha256 = config.publication_latency_expected_input_bundle_sha256
        assert expected_bundle_sha256 is not None
        args.extend(
            [
                "--publication-latency-schedule-json",
                str(publication_schedule_path),
                "--publication-latency-expected-input-bundle-sha256",
                expected_bundle_sha256,
            ]
        )
    if config.benchmark_interleave_examples:
        args.append("--interleave-examples")
    if config.uses_prepared_datasets:
        args.extend(
            [
                "--cache-base-url",
                config.server_base_url,
                "--prefix-cache-salt-mode",
                config.prefix_cache_salt_mode,
            ]
        )
    if baseline_extra_body:
        args.extend(
            [
                "--baseline-extra-body-json",
                json.dumps(baseline_extra_body, sort_keys=True),
            ]
        )
    if cache_extra_body:
        args.extend(
            ["--cache-extra-body-json", json.dumps(cache_extra_body, sort_keys=True)]
        )
    if config.cache_runtime_prompt:
        args.append("--cache-runtime-prompt")
    for arm_id in config.benchmark_arms:
        args.extend(["--arm", arm_id])
    for arm_spec in config.benchmark_arm_specs:
        args.extend(
            [
                "--arm-spec-json",
                json.dumps(
                    benchmark_json_mapping_to_record(arm_spec),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            ]
        )
    if config.benchmark_evidence_policy is not None:
        args.extend(["--evidence-policy", config.benchmark_evidence_policy])
    args.extend(
        benchmark_manifest_provenance_runner_args(config.benchmark_manifest_provenance)
    )
    if config.benchmark_runtime_id is not None:
        args.extend(["--runtime-id", config.benchmark_runtime_id])
    args.extend(dataset_args(dataset_paths))
    return args


def _benchmark_extra_body(
    *,
    cache_salt: str | None,
    force_max_tokens: bool,
) -> dict[str, object]:
    extra_body: dict[str, object] = {}
    if cache_salt:
        extra_body["cache_salt"] = cache_salt
    if force_max_tokens:
        extra_body["ignore_eos"] = True
    return extra_body


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def copy_file_if_exists(source_path: Path, target_path: Path) -> None:
    if source_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)


def server_env(config: VLLMSmokeBenchmarkConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update(vllm_server_env_overrides())
    _populate_handoff_generation_env(env, config)
    env["HF_HOME"] = str(config.hf_cache_dir)
    paths = cuda_wheel_env_paths(config)
    _prepend_env_paths(env, "CPATH", paths["include"])
    _prepend_env_paths(env, "LIBRARY_PATH", paths["library"])
    _prepend_env_paths(env, "LD_LIBRARY_PATH", paths["library"])
    return env


def _populate_handoff_generation_env(
    env: dict[str, str], config: VLLMSmokeBenchmarkConfig
) -> None:
    if config.is_representative_submission:
        _set_or_validate_env(env, "DOCUMENT_KV_EVICT_PAGE_CACHE", "1")
    if config.handoff_generation is None:
        return
    _set_or_validate_env(env, CACHET_TRANSFORMERS_MODEL_ID_ENV, config.model_id)
    _set_or_validate_env(env, CACHET_TRANSFORMERS_TOKENIZER_ID_ENV, config.model_id)
    if config.model_revision is not None:
        _set_or_validate_env(
            env,
            CACHET_TRANSFORMERS_MODEL_REVISION_ENV,
            config.model_revision,
        )
    if config.tokenizer_revision is not None:
        _set_or_validate_env(
            env,
            CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV,
            config.tokenizer_revision,
        )
    env.setdefault(CACHET_TRANSFORMERS_TORCH_DTYPE_ENV, config.model_dtype)
    env.setdefault(CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV, "false")
    if _is_bitsandbytes_4bit_quantization(config.model_quantization):
        env.setdefault(CACHET_TRANSFORMERS_QUANTIZATION_ENV, "bitsandbytes-4bit")
        env.setdefault(CACHET_TRANSFORMERS_DEVICE_MAP_ENV, "auto")
        env.setdefault(
            CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV,
            json.dumps(
                {
                    "bnb_4bit_compute_dtype": config.model_dtype,
                    "bnb_4bit_quant_storage": "uint8",
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "load_in_4bit": True,
                },
                sort_keys=True,
            ),
        )


def _set_or_validate_env(env: dict[str, str], name: str, value: str) -> None:
    existing = env.get(name)
    if existing is not None and existing != value:
        raise ValueError(
            f"{name}={existing!r} conflicts with the pinned value {value!r}"
        )
    env[name] = value


def _is_bitsandbytes_4bit_quantization(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower().replace("_", "-") in {
        "bitsandbytes",
        "bitsandbytes-4bit",
    }


def vllm_server_env_overrides() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        # Databricks' system CUDA 12.1 toolchain can be older than the cu129
        # wheel stack. The native sampler still exercises Cachet KV import while
        # avoiding FlashInfer sampler JIT during the smoke.
        VLLM_USE_FLASHINFER_SAMPLER_ENV: "0",
    }


def cuda_wheel_env_paths(config: VLLMSmokeBenchmarkConfig) -> dict[str, list[str]]:
    site_packages = site_packages_dirs(config)
    include_paths = _existing_paths(
        include_dir
        for site_package_dir in site_packages
        for include_dir in sorted((site_package_dir / "nvidia").glob("*/include"))
    )
    library_paths = _existing_paths(
        library_dir
        for site_package_dir in site_packages
        for library_dir in sorted((site_package_dir / "nvidia").glob("*/lib"))
    )
    return {"include": include_paths, "library": library_paths}


def site_packages_dirs(config: VLLMSmokeBenchmarkConfig) -> list[Path]:
    lib_dir = config.venv_dir / "lib"
    if not lib_dir.exists():
        return []
    return sorted(
        site_packages
        for python_dir in lib_dir.glob("python*")
        for site_packages in (python_dir / "site-packages",)
        if site_packages.is_dir()
    )


def _existing_paths(paths: Iterable[Path]) -> list[str]:
    existing = []
    seen = set()
    for path in paths:
        path = Path(path)
        if not path.is_dir():
            continue
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        existing.append(text)
    return existing


def _prepend_env_paths(env: dict[str, str], name: str, paths: list[str]) -> None:
    if not paths:
        return
    current = env.get(name)
    env[name] = os.pathsep.join([*paths, current] if current else paths)


def tail_text(text: str | bytes | None, *, max_chars: int = 12000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text[-max_chars:]


def tail(path: Path, *, lines: int = 120) -> str:
    if not path.exists():
        return "<missing log>"
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def parse_args(argv: list[str] | None = None) -> VLLMSmokeBenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Run a Qwen3/vLLM V1 benchmark smoke on Databricks g5/g6."
    )
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument(
        "--benchmark-suite-id",
        help=("Shared benchmark suite/experiment ID for independently executed arms."),
    )
    parser.add_argument(
        "--runtime-id",
        dest="benchmark_runtime_id",
        help="Unique physical execution ID recorded in benchmark provenance.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-id",
        default=HF_MODEL_ID,
        help="HF model path/id passed to vLLM --model.",
    )
    parser.add_argument(
        "--model-revision",
        help="Immutable Hugging Face model revision passed to vLLM --revision.",
    )
    parser.add_argument(
        "--tokenizer-revision",
        help=(
            "Immutable Hugging Face tokenizer revision passed to vLLM "
            "--tokenizer-revision."
        ),
    )
    parser.add_argument(
        "--model-dtype", default="bfloat16", help="Model dtype passed to vLLM --dtype."
    )
    parser.add_argument(
        "--model-quantization", help="Optional vLLM --quantization value."
    )
    parser.add_argument(
        "--kv-cache-dtype", help="Optional vLLM --kv-cache-dtype value."
    )
    parser.add_argument(
        "--attention-backend", help="Optional vLLM --attention-backend value."
    )
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
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument(
        "--kv-connector-mode",
        choices=KV_CONNECTOR_MODES,
        default=CACHET_KV_CONNECTOR_MODE,
        help="KV reuse backend: 'cachet' (default) or 'lmcache' (library-mode cold-load comparison).",
    )
    parser.add_argument("--lmcache-local-dir", default="/local_disk0/lmcache-store")
    parser.add_argument("--lmcache-max-disk-gb", type=float, default=80.0)
    parser.add_argument("--lmcache-chunk-size", type=int, default=256)
    parser.add_argument(
        "--lmcache-local-cpu",
        action="store_true",
        help="Offload LMCache KV to a bounded CPU-RAM tier (spills to the NVMe disk tier on overflow).",
    )
    parser.add_argument(
        "--lmcache-max-cpu-gb",
        type=float,
        default=0.0,
        help="CPU-RAM tier budget (GB) when --lmcache-local-cpu is set; 0 uses the LMCache default.",
    )
    parser.add_argument(
        "--lmcache-version",
        default="",
        help="Optional pinned lmcache version (empty installs the latest compatible release).",
    )
    parser.add_argument(
        "--hardware-target",
        choices=SUPPORTED_V1_HARDWARE_TARGETS,
        default=DEFAULT_HARDWARE_TARGET,
        help="V1 hardware target recorded in benchmark metadata.",
    )
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
        "--benchmark-interleave-examples",
        action="store_true",
        help=(
            "Round-robin benchmark requests across examples so a "
            "--request-parallelism N wave draws from N distinct documents "
            "(distinct docs across concurrent requests) instead of repeating one example."
        ),
    )
    schedule_source = parser.add_mutually_exclusive_group()
    schedule_source.add_argument(
        "--publication-latency-schedule-json",
        help=(
            "Path to the canonical closed publication latency schedule passed "
            "unchanged to the benchmark runner."
        ),
    )
    schedule_source.add_argument(
        "--publication-latency-schedule-record-json",
        help=(
            "Canonical closed publication latency schedule as an inline JSON "
            "object; materialized on node-local storage before runner launch."
        ),
    )
    parser.add_argument(
        "--publication-latency-expected-input-bundle-sha256",
        help=(
            "Verified main-latency input bundle SHA-256; required with either "
            "publication latency schedule source."
        ),
    )
    parser.add_argument(
        "--system-prompt-position",
        choices=SYSTEM_PROMPT_POSITIONS,
        default=DEFAULT_SYSTEM_PROMPT_POSITION,
        help=(
            "Where to place the system/task guidance prompt. 'start' (default) bakes it "
            "into the cached document prefix; 'end' places it after the documents so it "
            "is recomputed online with full attention over the injected document KV."
        ),
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
            "Validated arbitrary benchmark-runner arm JSON. Repeat for N-way "
            "method comparisons; mutually exclusive with --benchmark-arm."
        ),
    )
    parser.add_argument(
        "--benchmark-evidence-policy",
        choices=("smoke", "canary", "publication"),
        help="Evidence maturity passed to the benchmark runner.",
    )
    parser.add_argument(
        "--representative-canary",
        action="store_true",
        help=(
            "Require immutable model/tokenizer revisions and local-NVMe handoff "
            "generation for representative evidence."
        ),
    )
    parser.add_argument(
        "--representative-workload-profile",
        choices=tuple(
            profile.profile_id for profile in VLLM_REPRESENTATIVE_WORKLOAD_PROFILES
        ),
        help=(
            "Registered exact representative workload profile. Must be supplied "
            "together with --representative-canary."
        ),
    )
    parser.add_argument(
        "--benchmark-manifest-provenance-json",
        help=(
            "JSON object containing benchmark manifest provenance such as pinned "
            "engine/package/hardware/runtime identities and measurement scopes."
        ),
    )
    parser.add_argument(
        "--benchmark-cache-runtime-prompt",
        action="store_true",
        help="Pass --cache-runtime-prompt to the benchmark runner so cache arms send only the runtime suffix.",
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
            "Before measurement, populate every prepared handoff payload in the "
            "provider's host-RAM cache under an isolated GPU prefix-cache namespace; "
            "fail unless a verification pass and every measured load are RAM hits."
        ),
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
            "handshake. Requires pinned model and tokenizer revisions."
        ),
    )
    parser.add_argument(
        "--package-install-spec",
        help=(
            "Cachet wheel path or source checkout to install into the isolated vLLM environment. "
            f"Defaults to ${DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV} or the local source checkout."
        ),
    )
    parser.add_argument(
        "--native-runtime-v2-json",
        help=(
            "Exact ten-key mounted URI/SHA mapping for the native-v2 publication "
            "runtime. Required by publication evidence runs."
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
            "Generate Cachet handoff bundles for the prepared datasets before starting vLLM. "
            "Value must be a module:callable returning a KVChunkGenerator."
        ),
    )
    parser.add_argument(
        "--publication-handoff-generation-output-root",
        help=(
            "Durable root containing the qualified distributed publication "
            "handoff-generation execution record and closed bundles."
        ),
    )
    parser.add_argument(
        "--publication-handoff-generation-execution-file-sha256",
        help="SHA-256 of the canonical distributed generation execution JSON.",
    )
    parser.add_argument(
        "--publication-handoff-generation-execution-closed-record-sha256",
        help="Closed-record SHA-256 inside the distributed generation execution JSON.",
    )
    parser.add_argument(
        "--publication-handoff-local-nvme-dir",
        help=(
            "Nonexistent node-local destination under --local-root for atomic "
            "publication handoff staging."
        ),
    )
    parser.add_argument(
        "--benchmark-handoff-output-dir",
        help="Output directory for generated handoff bundles and enriched JSONL. Defaults under --output-dir.",
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
        help=(
            "Generate one KV chunk per document for multi-document examples so each "
            "prepared handoff assembles N document segments instead of one prefix chunk."
        ),
    )
    parser.add_argument(
        "--benchmark-handoff-cache-method",
        help=(
            "Method identity stamped on generated handoffs, for example "
            "'vanilla_prefill'."
        ),
    )
    parser.add_argument(
        "--benchmark-handoff-allow-legacy-artifact-contract",
        action="store_true",
        help=(
            "Legacy/debug opt-out: allow handoff generation without the registered "
            "method's complete artifact contract. Never use for canary evidence."
        ),
    )
    parser.add_argument(
        "--benchmark-force-max-tokens",
        action="store_true",
        help=(
            "Add ignore_eos=true to benchmark requests so TTC measures a forced "
            "--max-tokens decode instead of natural early stopping."
        ),
    )
    args = parser.parse_args(argv)
    output_dir = Path(_cluster_file_path(args.output_dir))
    handoff_generation = _handoff_generation_config_from_args(
        args, output_dir=output_dir
    )
    runtime_identity = (
        None
        if args.runtime_identity_json is None
        else _runtime_identity_from_json(args.runtime_identity_json)
    )
    benchmark_arm_specs = tuple(
        _json_object_from_cli(value, "--benchmark-arm-spec-json")
        for value in (args.benchmark_arm_spec_json or ())
    )
    benchmark_manifest_provenance = (
        {}
        if args.benchmark_manifest_provenance_json is None
        else _json_object_from_cli(
            args.benchmark_manifest_provenance_json,
            "--benchmark-manifest-provenance-json",
        )
    )
    publication_latency_schedule_record = (
        None
        if args.publication_latency_schedule_record_json is None
        else _json_object_from_cli(
            args.publication_latency_schedule_record_json,
            "--publication-latency-schedule-record-json",
        )
    )
    native_runtime_v2 = (
        None
        if args.native_runtime_v2_json is None
        else VLLMNativeRuntimeBundleV2.from_record(
            _json_object_from_cli(
                args.native_runtime_v2_json,
                "--native-runtime-v2-json",
            )
        )
    )
    return VLLMSmokeBenchmarkConfig(
        benchmark_id=args.benchmark_id,
        benchmark_suite_id=args.benchmark_suite_id,
        benchmark_runtime_id=args.benchmark_runtime_id,
        output_dir=output_dir,
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        model_dtype=args.model_dtype,
        model_quantization=args.model_quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        attention_backend=args.attention_backend,
        max_tokens=args.max_tokens,
        force_max_tokens=args.benchmark_force_max_tokens,
        timeout_seconds=args.timeout_seconds,
        import_probe_timeout_seconds=args.import_probe_timeout_seconds,
        server_start_timeout_seconds=args.server_start_timeout_seconds,
        local_root=Path(args.local_root),
        server_host=args.server_host,
        server_port=args.server_port,
        client_host=args.client_host,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        data_parallel_size=args.data_parallel_size,
        kv_connector_mode=args.kv_connector_mode,
        lmcache_local_dir=args.lmcache_local_dir,
        lmcache_max_disk_gb=args.lmcache_max_disk_gb,
        lmcache_chunk_size=args.lmcache_chunk_size,
        lmcache_local_cpu=args.lmcache_local_cpu,
        lmcache_max_cpu_gb=args.lmcache_max_cpu_gb,
        lmcache_version=args.lmcache_version,
        benchmark_repeats=args.benchmark_repeats,
        request_parallelism=args.request_parallelism,
        benchmark_interleave_examples=args.benchmark_interleave_examples,
        system_prompt_position=args.system_prompt_position,
        runtime_telemetry_interval_seconds=args.runtime_telemetry_interval_seconds,
        benchmark_arms=tuple(args.benchmark_arm or ()),
        benchmark_arm_specs=benchmark_arm_specs,
        benchmark_evidence_policy=args.benchmark_evidence_policy,
        representative_canary=args.representative_canary,
        representative_workload_profile=args.representative_workload_profile,
        benchmark_manifest_provenance=benchmark_manifest_provenance,
        prewarm_cache_prefix=args.benchmark_prewarm_cache_prefix,
        prewarm_payload_cache=args.benchmark_prewarm_payload_cache,
        cache_runtime_prompt=args.benchmark_cache_runtime_prompt,
        prefix_cache_salt_mode=args.benchmark_prefix_cache_salt_mode,
        hardware_target=args.hardware_target,
        payload_cache_max_bytes=args.payload_cache_max_bytes,
        dataset_specs=tuple(args.dataset or ()),
        allow_dataset_subset=args.allow_dataset_subset,
        package_install_spec=args.package_install_spec,
        handoff_generation=handoff_generation,
        runtime_identity=runtime_identity,
        publication_latency_schedule_record=publication_latency_schedule_record,
        publication_latency_schedule_path=(
            None
            if args.publication_latency_schedule_json is None
            else Path(_cluster_file_path(args.publication_latency_schedule_json))
        ),
        publication_latency_expected_input_bundle_sha256=(
            args.publication_latency_expected_input_bundle_sha256
        ),
        publication_handoff_generation_output_root=(
            None
            if args.publication_handoff_generation_output_root is None
            else Path(
                _cluster_file_path(args.publication_handoff_generation_output_root)
            )
        ),
        publication_handoff_generation_execution_file_sha256=(
            args.publication_handoff_generation_execution_file_sha256
        ),
        publication_handoff_generation_execution_closed_record_sha256=(
            args.publication_handoff_generation_execution_closed_record_sha256
        ),
        publication_handoff_local_nvme_dir=(
            None
            if args.publication_handoff_local_nvme_dir is None
            else Path(_cluster_file_path(args.publication_handoff_local_nvme_dir))
        ),
        native_runtime_v2=native_runtime_v2,
    )


def _handoff_generation_config_from_args(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> VLLMPreparedHandoffGenerationConfig | None:
    if args.benchmark_handoff_generator_factory is None:
        if (
            args.benchmark_handoff_output_dir is not None
            or args.benchmark_handoff_cache_method is not None
            or args.benchmark_handoff_allow_legacy_artifact_contract
            or args.benchmark_handoff_chunk_per_document
        ):
            raise ValueError(
                "benchmark handoff options require "
                "--benchmark-handoff-generator-factory"
            )
        return None
    output = args.benchmark_handoff_output_dir or str(
        Path(args.local_root)
        / f"document-kv-smoke-{args.benchmark_id}"
        / "generated-handoffs"
        if args.representative_canary
        else output_dir / "generated-handoffs"
    )
    return VLLMPreparedHandoffGenerationConfig(
        generator_factory=args.benchmark_handoff_generator_factory,
        output_dir=Path(_cluster_file_path(output)),
        dtype=args.benchmark_handoff_dtype,
        align_bytes=args.benchmark_handoff_align_bytes,
        timeout_seconds=args.benchmark_handoff_generation_timeout_seconds,
        limit=args.benchmark_handoff_limit,
        benchmark_handoff_segment_per_document=args.benchmark_handoff_chunk_per_document,
        cache_method=args.benchmark_handoff_cache_method,
        require_artifact_contract=(
            not args.benchmark_handoff_allow_legacy_artifact_contract
        ),
    )


def _runtime_identity_from_json(value: str) -> RuntimeIdentity:
    try:
        record = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"--runtime-identity-json must be valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(record, Mapping):
        raise ValueError("--runtime-identity-json must contain a JSON object")
    return RuntimeIdentity.from_record(record)


def _json_object_from_cli(value: str, option_name: str) -> Mapping[str, Any]:
    try:
        record = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} must be valid JSON: {exc.msg}") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"{option_name} must contain a JSON object")
    return record


def main(argv: list[str] | None = None) -> int:
    run_vllm_smoke_benchmark(parse_args(argv))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
