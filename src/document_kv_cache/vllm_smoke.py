"""Databricks-friendly vLLM smoke benchmark for the V1 Qwen3 path."""

from __future__ import annotations

import argparse
import gc
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any
import urllib.error
import urllib.request

from document_kv_cache.benchmark_handoffs import (
    enrich_benchmark_jsonl_with_handoffs,
    generate_benchmark_handoff_bundles,
    load_benchmark_kv_chunk_generator,
)
from document_kv_cache.benchmark_runner import load_v1_jsonl_suite
from document_kv_cache.benchmarks import (
    DEFAULT_HARDWARE_TARGET,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    SUPPORTED_V1_HARDWARE_TARGETS,
    build_prompt_parts,
    validate_v1_hardware_target,
)
from document_kv_cache.engine_adapters import (
    ServingBackend,
    read_engine_adapter_request_json,
    validate_engine_adapter_request_record,
)
from document_kv_cache.engine_probe import _validate_local_payload_uri
from document_kv_cache.model_profiles import layout_for_model
from document_kv_cache.model_profiles import QWEN3_4B_INSTRUCT_HF_MODEL_ID
from document_kv_cache.serving_env import (
    FASTAPI_CONSTRAINT,
    HUGGINGFACE_HUB_CONSTRAINT,
    NUMPY_CONSTRAINT,
    PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT,
    TOKENIZERS_CONSTRAINT,
    TRANSFORMERS_CONSTRAINT,
    VLLM_SERVING_ENVIRONMENT_PROFILE,
    VLLM_VERSION,
)
from vllm_kv_injection.vllm_transfer_config import (
    document_kv_transfer_config,
    document_kv_transfer_config_json,
)
from vllm_kv_injection.vllm_dynamic_connector import (
    DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY,
    DocumentKVConnector,
    NoOpDocumentKVProvider,
)

HF_MODEL_ID = QWEN3_4B_INSTRUCT_HF_MODEL_ID
SERVED_MODEL_NAME = "qwen3:4b-instruct"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
BASELINE_PREFIX_CACHE_SALT = "cachet-baseline-prefill"
CACHE_PREFIX_CACHE_SALT = "cachet-kv-cache"
PREPARED_PREFIX_CACHE_SALT_MODE = "per_request"
SERVER_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
SMOKE_DATASETS = ("biography", "hotpotqa", "musique", "niah")
DEFAULT_LOCAL_ROOT = Path("/local_disk0")
DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV = "DOCUMENT_KV_PACKAGE_INSTALL_SPEC"
VLLM_FIPS_OPENCV_OVERRIDE_CONSTRAINT = "opencv-python-headless==4.12.0.88"
VLLM_USE_FLASHINFER_SAMPLER_ENV = "VLLM_USE_FLASHINFER_SAMPLER"

__all__ = [
    "VLLM_VERSION",
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
    "VLLMSmokeBenchmarkConfig",
    "VLLMPreparedHandoffGenerationConfig",
    "run_vllm_smoke_benchmark",
    "build_metadata",
    "build_vllm_native_provider_probe_record",
    "cuda_wheel_env_paths",
    "dependency_constraints",
    "dependency_override_constraints",
    "document_kv_package_install_spec",
    "install_document_kv_package",
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
    "release_handoff_generation_resources",
    "smoke_dataset_records",
    "parse_dataset_specs",
    "dataset_args",
    "parse_args",
    "site_packages_dirs",
    "main",
    "VLLM_FIPS_OPENCV_OVERRIDE_CONSTRAINT",
]


@dataclass(frozen=True, slots=True)
class VLLMPreparedHandoffGenerationConfig:
    """Optional generation settings for prepared vLLM benchmark handoffs."""

    generator_factory: str
    output_dir: Path
    dtype: str = "bfloat16"
    align_bytes: int = 4096
    timeout_seconds: float = 1800.0

    def __post_init__(self) -> None:
        if not isinstance(self.generator_factory, str) or not self.generator_factory.strip():
            raise ValueError("benchmark_handoff_generator_factory must be non-empty")
        if self.output_dir is None:
            raise ValueError("benchmark_handoff_output_dir must be provided")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError("benchmark_handoff_dtype must be non-empty")
        if type(self.align_bytes) is not int or self.align_bytes <= 0:
            raise ValueError("benchmark_handoff_align_bytes must be a positive integer")
        if self.timeout_seconds <= 0:
            raise ValueError("benchmark_handoff_timeout_seconds must be positive")
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def to_metadata(self) -> dict[str, object]:
        return {
            "generator_factory": self.generator_factory,
            "output_dir": str(self.output_dir),
            "dtype": self.dtype,
            "align_bytes": self.align_bytes,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class VLLMSmokeBenchmarkConfig:
    """Runtime configuration for a one-node Databricks vLLM smoke run."""

    benchmark_id: str
    output_dir: Path
    max_tokens: int = 32
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
    benchmark_repeats: int = 1
    hardware_target: str = DEFAULT_HARDWARE_TARGET
    dataset_specs: tuple[str, ...] = ()
    package_install_spec: str | None = None
    handoff_generation: VLLMPreparedHandoffGenerationConfig | None = None
    payload_cache_max_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if self.output_dir is None:
            raise ValueError("output_dir must be provided")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
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
        if isinstance(self.benchmark_repeats, bool) or not isinstance(self.benchmark_repeats, int):
            raise TypeError("benchmark_repeats must be a positive integer")
        if self.benchmark_repeats <= 0:
            raise ValueError("benchmark_repeats must be a positive integer")
        if not isinstance(self.hardware_target, str) or not self.hardware_target.strip():
            raise ValueError("hardware_target must be non-empty")
        validate_v1_hardware_target(self.hardware_target)
        if isinstance(self.payload_cache_max_bytes, bool) or not isinstance(self.payload_cache_max_bytes, int):
            raise TypeError("payload_cache_max_bytes must be a non-negative integer")
        if self.payload_cache_max_bytes < 0:
            raise ValueError("payload_cache_max_bytes must be a non-negative integer")
        object.__setattr__(self, "dataset_specs", tuple(self.dataset_specs))
        if self.dataset_specs:
            parse_dataset_specs(self.dataset_specs)
        if self.package_install_spec is not None and not self.package_install_spec.strip():
            raise ValueError("package_install_spec must be non-empty when provided")
        if self.handoff_generation is not None:
            if not isinstance(self.handoff_generation, VLLMPreparedHandoffGenerationConfig):
                raise TypeError("handoff_generation must be a VLLMPreparedHandoffGenerationConfig")
            if not self.dataset_specs:
                raise ValueError("benchmark_handoff_generator_factory requires prepared dataset specs")

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
    def benchmark_output_path(self) -> Path:
        return self.output_dir / "v1-benchmark.json"

    @property
    def prompt_token_budget_path(self) -> Path:
        return self.output_dir / "prompt-token-budget.json"

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
    def prepared_handoff_generation_path(self) -> Path:
        return self.output_dir / "prepared-handoff-generation.json"

    @property
    def uses_prepared_datasets(self) -> bool:
        return bool(self.dataset_specs)


def run_vllm_smoke_benchmark(config: VLLMSmokeBenchmarkConfig) -> None:
    """Create an isolated vLLM env, start Qwen3, and run the V1 smoke suite."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.local_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(config.hf_cache_dir)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    metadata = build_metadata(config)
    write_json(config.metadata_path, metadata)

    create_venv(config.venv_dir)
    install_vllm(config.venv_python)
    install_document_kv_package(config.venv_python, document_kv_package_install_spec(config))
    metadata.update(installed_versions(config.venv_python))
    metadata["cuda_wheel_env_paths"] = cuda_wheel_env_paths(config)
    write_json(config.metadata_path, metadata)
    probe_vllm_import(
        config.venv_python,
        config.import_probe_path,
        timeout_seconds=config.import_probe_timeout_seconds,
        env=server_env(config),
    )

    dataset_paths = benchmark_dataset_paths(config)
    dataset_paths = prepare_generated_benchmark_handoffs(config, dataset_paths)
    validate_prepared_benchmark_handoffs(config, dataset_paths)
    validate_prompt_token_budget(config, dataset_paths)
    metadata["vllm_server_local_log"] = str(config.server_log_path)
    metadata["vllm_server_log"] = str(config.server_log_copy_path)
    metadata["prompt_token_budget_path"] = str(config.prompt_token_budget_path)
    if config.uses_prepared_datasets:
        metadata["prepared_handoff_coverage_path"] = str(config.prepared_handoff_coverage_path)
    if config.handoff_generation is not None:
        metadata["prepared_handoff_generation_path"] = str(config.prepared_handoff_generation_path)
    write_json(config.metadata_path, metadata)

    server = start_vllm_server(config, config.venv_python, config.server_log_path)
    try:
        wait_for_server(server, config.server_log_path, config, timeout_seconds=config.server_start_timeout_seconds)
        copy_file_if_exists(config.server_log_path, config.server_log_copy_path)
        run_benchmark_runner(config, dataset_paths)
    finally:
        terminate_process(server)
        copy_file_if_exists(config.server_log_path, config.server_log_copy_path)


def build_metadata(config: VLLMSmokeBenchmarkConfig) -> dict[str, object]:
    return {
        "benchmark_id": config.benchmark_id,
        "hf_model_id": HF_MODEL_ID,
        "served_model_name": SERVED_MODEL_NAME,
        "vllm_version_requested": VLLM_VERSION,
        "server_bind_host": config.server_host,
        "server_client_host": config.client_host,
        "server_base_url": config.server_base_url,
        "hf_home": str(config.hf_cache_dir),
        "vllm_python": str(config.venv_python),
        "dependency_constraints": dependency_constraints(),
        "dataset_source": "prepared" if config.dataset_specs else "smoke",
        "dataset_specs": list(config.dataset_specs),
        "cache_runtime_prompt": False,
        "cache_prompt_text_mode": "logical",
        "prefix_cache_isolation": (
            {
                "baseline_cache_salt": BASELINE_PREFIX_CACHE_SALT,
                "cache_cache_salt": CACHE_PREFIX_CACHE_SALT,
                "cache_salt_mode": PREPARED_PREFIX_CACHE_SALT_MODE,
            }
            if config.uses_prepared_datasets
            else None
        ),
        "requires_kv_transfer_params": config.uses_prepared_datasets,
        "generates_prepared_handoffs": config.handoff_generation is not None,
        "benchmark_handoff_generation": (
            None if config.handoff_generation is None else config.handoff_generation.to_metadata()
        ),
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "benchmark_repeats": config.benchmark_repeats,
        "hardware_target": config.hardware_target,
        "document_kv_package_install_spec": document_kv_package_install_spec(config),
        "dependency_override_constraints": dependency_override_constraints(),
        "vllm_server_env_overrides": vllm_server_env_overrides(),
        "vllm_kv_transfer_config": document_kv_transfer_config_for_smoke(config),
    }


def build_vllm_native_provider_probe_record(
    transfer_config: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Instantiate the configured vLLM connector and verify native provider wiring."""

    config = document_kv_transfer_config() if transfer_config is None else transfer_config
    if not isinstance(config, Mapping):
        raise TypeError("vLLM KV transfer config must be a mapping")
    extra_config = config.get("kv_connector_extra_config")
    if not isinstance(extra_config, Mapping):
        raise TypeError("vLLM KV transfer config kv_connector_extra_config must be a mapping")
    provider_factory = extra_config.get(DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY)
    if not isinstance(provider_factory, str) or not provider_factory.strip():
        raise ValueError(
            f"{DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY} must be a non-empty module:attribute string"
        )
    if extra_config.get("document_kv.requires_native_runtime") is not True:
        raise ValueError("document_kv.requires_native_runtime must be true")

    connector = DocumentKVConnector(vllm_config=SimpleNamespace(kv_transfer_config=config))
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


def document_kv_transfer_config_for_smoke(config: VLLMSmokeBenchmarkConfig) -> dict[str, Any]:
    return document_kv_transfer_config(
        payload_cache_max_bytes=config.payload_cache_max_bytes or None,
    )


def dependency_constraints() -> list[str]:
    return list(VLLM_SERVING_ENVIRONMENT_PROFILE.dependency_constraints)


def dependency_override_constraints() -> list[str]:
    return [VLLM_FIPS_OPENCV_OVERRIDE_CONSTRAINT]


def _cluster_file_path(uri: str) -> str:
    if uri.startswith("dbfs:/"):
        return "/dbfs/" + uri.removeprefix("dbfs:/").lstrip("/")
    return uri


def _source_checkout_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "document_kv_cache").exists():
            return parent
    return None


def document_kv_package_install_spec(config: VLLMSmokeBenchmarkConfig) -> str:
    """Return the package spec that must be installed into the vLLM venv."""

    if config.package_install_spec is not None:
        return _cluster_file_path(config.package_install_spec)
    env_value = os.environ.get(DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV)
    if env_value is not None:
        if not env_value.strip():
            raise ValueError(f"{DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV} must be non-empty when set")
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
        "document_kv_cache_version_installed": installed_package_version(python_executable, "cachet-kv"),
        "transformers_version_installed": installed_package_version(python_executable, "transformers"),
        "torch_version_installed": installed_package_version(python_executable, "torch"),
        "opencv_python_headless_version_installed": installed_package_version(
            python_executable,
            "opencv-python-headless",
        ),
    }


def run(argv: list[str]) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def validate_prompt_token_budget(config: VLLMSmokeBenchmarkConfig, dataset_paths: dict[str, Path]) -> None:
    rows = build_prompt_token_budget_rows(config, dataset_paths)
    write_prompt_token_budget_jsonl(config.prompt_token_budget_input_path, rows)
    record = run_prompt_token_budget_probe(
        config.venv_python,
        config.prompt_token_budget_input_path,
        model_id=HF_MODEL_ID,
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
    rows = []
    for example in suite.examples:
        prompt = build_prompt_parts(example).prefill_prompt
        rows.append({"dataset": example.dataset, "example_id": example.example_id, "prompt": prompt})
    return tuple(rows)


def validate_prepared_benchmark_handoffs(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: dict[str, Path],
) -> dict[str, object] | None:
    """Require prepared benchmark rows to carry loadable Cachet handoff params."""

    if not config.uses_prepared_datasets:
        return None
    record = prepared_benchmark_handoff_coverage_record(config, dataset_paths)
    write_json(config.prepared_handoff_coverage_path, record)
    if record.get("ok") is not True:
        missing = record.get("missing_kv_transfer_params")
        invalid = record.get("invalid_handoff_references")
        raise ValueError(
            "Prepared vLLM benchmark datasets must be enriched with Cachet kv_transfer_params "
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
    missing = tuple(
        f"{example.dataset}/{example.example_id}"
        for example in suite.examples
        if not example.kv_transfer_params
    )
    invalid = tuple(
        issue
        for example in suite.examples
        if example.kv_transfer_params
        for issue in (_prepared_handoff_reference_issue(example),)
        if issue is not None
    )
    counts_by_dataset: dict[str, int] = {}
    for example in suite.examples:
        counts_by_dataset[example.dataset] = counts_by_dataset.get(example.dataset, 0) + 1
    issues = []
    if missing:
        issues.append("prepared benchmark rows missing kv_transfer_params")
    if invalid:
        issues.append("prepared benchmark rows reference unloadable Cachet handoffs")
    return {
        "ok": not missing and not invalid,
        "required": True,
        "dataset_source": "prepared",
        "datasets": counts_by_dataset,
        "examples": len(suite.examples),
        "examples_with_kv_transfer_params": len(suite.examples) - len(missing),
        "examples_with_loadable_handoff_references": len(suite.examples) - len(missing) - len(invalid),
        "missing_kv_transfer_params": list(missing),
        "invalid_handoff_references": list(invalid),
        "issues": issues,
    }


def _prepared_handoff_reference_issue(example: object) -> dict[str, object] | None:
    params = getattr(example, "kv_transfer_params", {})
    if not isinstance(params, Mapping):
        return _handoff_reference_issue(example, "kv_transfer_params must be a mapping")
    handoff_json: str | None = None
    payload_override = params.get(DOCUMENT_KV_PAYLOAD_URI_PARAM)
    try:
        handoff_record = params.get(DOCUMENT_KV_HANDOFF_RECORD_PARAM)
        if handoff_record is not None:
            record = handoff_record
            if not isinstance(record, Mapping):
                raise ValueError(f"kv_transfer_params.{DOCUMENT_KV_HANDOFF_RECORD_PARAM} must be an object")
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
        return _handoff_reference_issue(
            example,
            str(exc),
            error_type=type(exc).__name__,
            handoff_json=handoff_json,
        )
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


def write_prompt_token_budget_jsonl(path: Path, rows: tuple[dict[str, str], ...]) -> None:
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
        generated_paths, record = _generate_prepared_benchmark_handoff_inputs_in_subprocess(
            config,
            dataset_paths,
            generation,
        )
        write_json(config.prepared_handoff_generation_path, record)
        return generated_paths
    generation.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        generated_paths, record = _generate_prepared_benchmark_handoff_inputs(config, dataset_paths, generation)
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
        "output_dir": str(config.output_dir),
        "dataset_paths": {dataset: str(path) for dataset, path in dataset_paths.items()},
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
)
config = VLLMSmokeBenchmarkConfig(
    benchmark_id=payload["benchmark_id"],
    output_dir=Path(payload["output_dir"]),
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
    print("+", " ".join([argv[0], "-c", "<prepared handoff generation>", *argv[3:]]), flush=True)
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
        raise RuntimeError(f"prepared handoff generation wrote invalid result {output_path}")
    if record.get("ok") is not True:
        raise RuntimeError(f"prepared handoff generation worker result was not ok in {output_path}")
    if set(generated_paths_payload) != set(SMOKE_DATASETS):
        raise RuntimeError(
            "prepared handoff generation worker result must include exactly "
            f"{sorted(SMOKE_DATASETS)!r}; got {sorted(str(dataset) for dataset in generated_paths_payload)!r}"
        )
    generated_paths = {
        str(dataset): Path(str(path))
        for dataset, path in generated_paths_payload.items()
    }
    for dataset, path in generated_paths.items():
        if not str(path):
            raise RuntimeError(f"prepared handoff generation worker returned empty path for {dataset}")
        if not path.exists():
            raise RuntimeError(f"prepared handoff generation worker output for {dataset} does not exist: {path}")
    return generated_paths, record


def _generate_prepared_benchmark_handoff_inputs(
    config: VLLMSmokeBenchmarkConfig,
    dataset_paths: dict[str, Path],
    generation: VLLMPreparedHandoffGenerationConfig,
) -> tuple[dict[str, Path], dict[str, object]]:
    generator = load_benchmark_kv_chunk_generator(generation.generator_factory)
    layout = layout_for_model(SERVED_MODEL_NAME, dtype=generation.dtype)
    generated_paths: dict[str, Path] = {}
    dataset_records: dict[str, dict[str, object]] = {}
    for dataset in SMOKE_DATASETS:
        input_jsonl = dataset_paths[dataset]
        dataset_output_dir = generation.output_dir / dataset
        manifest_json = generation.output_dir / f"{dataset}-manifest.json"
        output_jsonl = generation.output_dir / f"{dataset}.handoffs.jsonl"
        result = generate_benchmark_handoff_bundles(
            input_jsonl,
            output_dir=dataset_output_dir,
            generator=generator,
            layout=layout,
            dataset=dataset,
            backend="vllm",
            manifest_json=manifest_json,
            align_bytes=generation.align_bytes,
        )
        enriched_rows = enrich_benchmark_jsonl_with_handoffs(
            input_jsonl,
            manifest_json,
            output_jsonl,
            dataset=dataset,
            overwrite=True,
        )
        generated_paths[dataset] = output_jsonl
        dataset_records[dataset] = {
            "input_jsonl": str(input_jsonl),
            "output_jsonl": str(output_jsonl),
            "manifest_json": str(manifest_json),
            "bundle_output_dir": str(dataset_output_dir),
            "entries": len(result.manifest.entries),
            "enriched_rows": enriched_rows,
            "cache_refs": len(result.cache_refs),
            "shard_uri": result.shard_uri,
        }

    record = {
        "ok": True,
        "dataset_source": "prepared",
        "benchmark_id": config.benchmark_id,
        "generator_factory": generation.generator_factory,
        "output_dir": str(generation.output_dir),
        "dtype": generation.dtype,
        "align_bytes": generation.align_bytes,
        "datasets": dataset_records,
    }
    return generated_paths, record


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
    max_model_len: int,
    max_tokens: int,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    code = """
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

model_id, input_path, max_model_len, max_tokens = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
rows = []
over_budget = []
with input_path.open("r", encoding="utf-8") as handle:
    for raw_line in handle:
        row = json.loads(raw_line)
        prompt_tokens = len(tokenizer(row["prompt"])["input_ids"])
        total_tokens = prompt_tokens + max_tokens
        measured = {
            "dataset": row["dataset"],
            "example_id": row["example_id"],
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "total_tokens": total_tokens,
            "max_model_len": max_model_len,
        }
        rows.append(measured)
        if total_tokens > max_model_len:
            over_budget.append(measured)
print(json.dumps({"rows": rows, "over_budget": over_budget}, sort_keys=True), flush=True)
"""
    argv = [
        str(python_executable),
        "-c",
        code,
        model_id,
        str(input_path),
        str(max_model_len),
        str(max_tokens),
    ]
    print("+", " ".join([argv[0], "-c", "<prompt token budget probe>", *argv[3:]]), flush=True)
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
            "rows": [],
            "over_budget": [],
        }
    record = last_json_object(completed.stdout)
    record.update(
        {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
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
    return record


def run_benchmark_runner(config: VLLMSmokeBenchmarkConfig, dataset_paths: dict[str, Path]) -> None:
    try:
        run(build_benchmark_runner_args(config, dataset_paths))
    except subprocess.CalledProcessError as exc:
        summary = benchmark_failure_summary(config.benchmark_output_path)
        raise RuntimeError(
            f"vLLM benchmark runner failed with exit code {exc.returncode}; {summary}"
        ) from exc


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


def _benchmark_error_summary(measurement: dict[str, object], *, max_chars: int = 400) -> str:
    dataset = measurement.get("dataset") or "unknown-dataset"
    arm_id = measurement.get("arm_id") or "unknown-arm"
    error = str(measurement.get("error") or "unknown error")
    if len(error) > max_chars:
        error = error[: max_chars - 3] + "..."
    return f"{dataset}/{arm_id}: {error}"


def create_venv(venv_dir: Path) -> None:
    if venv_python(venv_dir).exists():
        return
    try:
        run([sys.executable, "-m", "venv", str(venv_dir)])
    except subprocess.CalledProcessError:
        run([sys.executable, "-m", "pip", "install", "virtualenv"])
        run([sys.executable, "-m", "virtualenv", str(venv_dir)])


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def install_vllm(python_executable: Path) -> None:
    run([str(python_executable), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            *dependency_constraints(),
        ]
    )
    run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            *dependency_override_constraints(),
        ]
    )


def install_document_kv_package(python_executable: Path, install_spec: str) -> None:
    run([str(python_executable), "-m", "pip", "install", "--no-deps", install_spec])


def installed_package_version(python_executable: Path, package_name: str) -> str:
    completed = subprocess.run(
        [str(python_executable), "-m", "pip", "show", package_name],
        check=True,
        capture_output=True,
        text=True,
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
    code = """
import importlib.metadata as md
import json
import torch
import vllm
import vllm.entrypoints.openai.api_server
import document_kv_cache
import vllm_kv_injection.vllm_dynamic_connector as document_kv_vllm_connector
from document_kv_cache.vllm_smoke import build_vllm_native_provider_probe_record
from vllm_kv_injection.vllm_transfer_config import document_kv_transfer_config

transfer_config = document_kv_transfer_config()
payload = {
    "ok": True,
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "vllm_version": md.version("vllm"),
    "document_kv_cache_version": md.version("cachet-kv"),
    "document_kv_cache_module": document_kv_cache.__name__,
    "document_kv_connector_module": document_kv_vllm_connector.__name__,
    "document_kv_connector": transfer_config["kv_connector"],
}
payload.update(build_vllm_native_provider_probe_record(transfer_config))
if torch.cuda.is_available():
    payload["cuda_device_name"] = torch.cuda.get_device_name(0)
print(json.dumps(payload, sort_keys=True), flush=True)
"""
    argv = [str(python_executable), "-c", code]
    print("+", " ".join([argv[0], "-c", "<vllm import probe>"]), flush=True)
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
        raise RuntimeError(f"vLLM import probe failed with return code {completed.returncode}")


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


def benchmark_dataset_paths(config: VLLMSmokeBenchmarkConfig) -> dict[str, Path]:
    if config.dataset_specs:
        return parse_dataset_specs(config.dataset_specs)
    return write_smoke_datasets(config.local_dir)


def parse_dataset_specs(dataset_specs: tuple[str, ...]) -> dict[str, Path]:
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
    if missing:
        raise ValueError(f"dataset specs missing required V1 datasets: {sorted(missing)}")
    return {dataset: paths[dataset] for dataset in SMOKE_DATASETS}


def smoke_dataset_records() -> dict[str, dict[str, object]]:
    return {
        "biography": {
            "example_id": "biography-smoke-1",
            "dataset": "biography",
            "query": "Which person is described in the biography?",
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
    for dataset in SMOKE_DATASETS:
        args.extend(["--dataset", f"{dataset}={dataset_paths[dataset]}"])
    return args


def build_vllm_server_args(config: VLLMSmokeBenchmarkConfig, python_executable: Path) -> list[str]:
    return [
        str(python_executable),
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        HF_MODEL_ID,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        config.server_host,
        "--port",
        str(config.server_port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--kv-transfer-config",
        document_kv_transfer_config_json(
            payload_cache_max_bytes=config.payload_cache_max_bytes or None,
        ),
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]


def start_vllm_server(
    config: VLLMSmokeBenchmarkConfig, python_executable: Path, log_path: Path
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = build_vllm_server_args(config, python_executable)
    print("+", " ".join(argv), flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        return subprocess.Popen(argv, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=server_env(config))


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
            raise RuntimeError(f"vLLM server exited with {server.returncode}; log tail:\n{tail(log_path)}")
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    model_ids = fetch_served_model_ids(models_url)
                    if SERVED_MODEL_NAME in model_ids:
                        return
                    last_model_error = f"health OK but served models were {sorted(model_ids)!r}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError) as exc:
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
    args = [
        sys.executable,
        "-m",
        "document_kv_cache.benchmark_runner",
        "--suite-id",
        config.benchmark_id,
        "--base-url",
        config.server_base_url,
        "--model-id",
        SERVED_MODEL_NAME,
        "--hardware-target",
        config.hardware_target,
        "--max-tokens",
        str(config.max_tokens),
        "--timeout-seconds",
        str(config.timeout_seconds),
        "--repeats",
        str(config.benchmark_repeats),
        "--server-usage",
        "--output-json",
        str(config.benchmark_output_path),
    ]
    if config.uses_prepared_datasets:
        args.extend(
            [
                "--cache-base-url",
                config.server_base_url,
                "--baseline-extra-body-json",
                json.dumps({"cache_salt": BASELINE_PREFIX_CACHE_SALT}, sort_keys=True),
                "--cache-extra-body-json",
                json.dumps({"cache_salt": CACHE_PREFIX_CACHE_SALT}, sort_keys=True),
                "--prefix-cache-salt-mode",
                PREPARED_PREFIX_CACHE_SALT_MODE,
            ]
        )
    args.extend(dataset_args(dataset_paths))
    return args


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_file_if_exists(source_path: Path, target_path: Path) -> None:
    if source_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)


def server_env(config: VLLMSmokeBenchmarkConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update(vllm_server_env_overrides())
    env["HF_HOME"] = str(config.hf_cache_dir)
    paths = cuda_wheel_env_paths(config)
    _prepend_env_paths(env, "CPATH", paths["include"])
    _prepend_env_paths(env, "LIBRARY_PATH", paths["library"])
    _prepend_env_paths(env, "LD_LIBRARY_PATH", paths["library"])
    return env


def vllm_server_env_overrides() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        # Databricks' system nvcc can be older than the CUDA 13 headers in the
        # vLLM wheel stack. The native sampler still exercises Cachet KV import
        # while avoiding FlashInfer sampler JIT during the smoke.
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
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def parse_args(argv: list[str] | None = None) -> VLLMSmokeBenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run a Qwen3/vLLM V1 benchmark smoke on Databricks g5/g6.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", required=True)
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
        "--payload-cache-max-bytes",
        type=int,
        default=0,
        help=(
            "Optional byte budget for the vLLM provider's in-process payload URI cache. "
            "Use 0 to disable."
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
        "--dataset",
        action="append",
        default=None,
        help="Prepared V1 benchmark dataset in DATASET=JSONL_PATH form. Repeat for all four V1 datasets.",
    )
    parser.add_argument(
        "--benchmark-handoff-generator-factory",
        help=(
            "Generate Cachet handoff bundles for the prepared datasets before starting vLLM. "
            "Value must be a module:callable returning a KVChunkGenerator."
        ),
    )
    parser.add_argument(
        "--benchmark-handoff-output-dir",
        help="Output directory for generated handoff bundles and enriched JSONL. Defaults under --output-dir.",
    )
    parser.add_argument("--benchmark-handoff-dtype", default="bfloat16")
    parser.add_argument("--benchmark-handoff-align-bytes", type=int, default=4096)
    args = parser.parse_args(argv)
    output_dir = Path(_cluster_file_path(args.output_dir))
    handoff_generation = _handoff_generation_config_from_args(args, output_dir=output_dir)
    return VLLMSmokeBenchmarkConfig(
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        max_tokens=args.max_tokens,
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
        benchmark_repeats=args.benchmark_repeats,
        hardware_target=args.hardware_target,
        payload_cache_max_bytes=args.payload_cache_max_bytes,
        dataset_specs=tuple(args.dataset or ()),
        package_install_spec=args.package_install_spec,
        handoff_generation=handoff_generation,
    )


def _handoff_generation_config_from_args(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> VLLMPreparedHandoffGenerationConfig | None:
    if args.benchmark_handoff_generator_factory is None:
        if args.benchmark_handoff_output_dir is not None:
            raise ValueError(
                "--benchmark-handoff-output-dir requires --benchmark-handoff-generator-factory"
            )
        return None
    output = args.benchmark_handoff_output_dir or str(output_dir / "generated-handoffs")
    return VLLMPreparedHandoffGenerationConfig(
        generator_factory=args.benchmark_handoff_generator_factory,
        output_dir=Path(_cluster_file_path(output)),
        dtype=args.benchmark_handoff_dtype,
        align_bytes=args.benchmark_handoff_align_bytes,
    )


def main(argv: list[str] | None = None) -> int:
    run_vllm_smoke_benchmark(parse_args(argv))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
