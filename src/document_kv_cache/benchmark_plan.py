"""Benchmark command-plan generation for Document KV Cache."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_kv_cache.benchmarks import DEFAULT_HARDWARE_TARGET, DEFAULT_V1_MODEL_ID, SUPPORTED_V1_DATASETS
from document_kv_cache.benchmarks import validate_v1_dataset, validate_v1_hardware_target
from document_kv_cache.engine_adapters import PayloadMode, ServingBackend
from document_kv_cache.native_probe_factories import builtin_native_probe_factory_path
from document_kv_cache.probe_fixtures import DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES
from document_kv_cache.storage import local_path
from document_kv_cache.storage_benchmark import (
    RELEASE_STORAGE_BENCHMARK_READERS,
    SUPPORTED_STORAGE_BENCHMARK_READERS,
)


PLAN_VERSION = "v1"
ENGINE_PROBE_TARGETS_RECORD_TYPE = "document_kv.engine_probe_targets.v1"
ENGINE_PROBE_TARGETS_SCHEMA_VERSION = 1
DEFAULT_STORAGE_BENCHMARK_ID = "storage-reader-benchmark"
DEFAULT_STORAGE_BENCHMARK_CHUNK_COUNT = 64
DEFAULT_STORAGE_BENCHMARK_CHUNK_BYTES = 1024 * 1024
DEFAULT_STORAGE_BENCHMARK_REPEATS = 4
DEFAULT_STORAGE_BENCHMARK_PARALLELISM = 4
DEFAULT_STORAGE_BENCHMARK_ALIGN_BYTES = 4096
DEFAULT_STORAGE_BENCHMARK_PLAN_READERS = ("memory", "disk")
DEFAULT_ENGINE_LAUNCH_CONFIG_FILENAMES = {
    ServingBackend.VLLM: "vllm-launch-config.json",
    ServingBackend.SGLANG: "sglang-launch-config.json",
}
STRICT_V1_DATABRICKS_RUN_STATUS_SIDECAR_COUNT = 3
STRICT_V1_DATABRICKS_RUN_STATUS_SIDECAR_LABEL = (
    "exactly three distinct Databricks run-status sidecars "
    "for benchmark, storage, and engine-probe runs"
)

__all__ = [
    "PLAN_VERSION",
    "ENGINE_PROBE_TARGETS_RECORD_TYPE",
    "ENGINE_PROBE_TARGETS_SCHEMA_VERSION",
    "BenchmarkDatasetPath",
    "BenchmarkCommand",
    "StorageBenchmarkPlanConfig",
    "EngineProbePlanConfig",
    "ReleaseEvidencePlanConfig",
    "ReleaseBundlePlanConfig",
    "BenchmarkPlanConfig",
    "BenchmarkJobPlan",
    "build_v1_benchmark_plan",
    "benchmark_job_plan_to_record",
    "engine_probe_targets_to_record",
    "write_benchmark_job_plan_json",
    "write_benchmark_job_plan_shell",
    "write_engine_probe_targets_json",
    "main",
]


@dataclass(frozen=True, slots=True)
class BenchmarkDatasetPath:
    dataset: str
    raw_jsonl: str
    prepared_jsonl: str

    def __post_init__(self) -> None:
        validate_v1_dataset(self.dataset)
        if not self.raw_jsonl:
            raise ValueError("raw_jsonl must be non-empty")
        if not self.prepared_jsonl:
            raise ValueError("prepared_jsonl must be non-empty")


@dataclass(frozen=True, slots=True)
class BenchmarkCommand:
    name: str
    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.argv:
            raise ValueError("argv must be non-empty")
        if any(not item for item in self.argv):
            raise ValueError("argv items must be non-empty")

    @property
    def shell(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True, slots=True)
class StorageBenchmarkPlanConfig:
    workspace_dir: str
    output_json: str
    benchmark_id: str = DEFAULT_STORAGE_BENCHMARK_ID
    chunk_count: int = DEFAULT_STORAGE_BENCHMARK_CHUNK_COUNT
    chunk_bytes: int = DEFAULT_STORAGE_BENCHMARK_CHUNK_BYTES
    repeats: int = DEFAULT_STORAGE_BENCHMARK_REPEATS
    parallelism: int = DEFAULT_STORAGE_BENCHMARK_PARALLELISM
    readers: tuple[str, ...] = DEFAULT_STORAGE_BENCHMARK_PLAN_READERS
    align_bytes: int = DEFAULT_STORAGE_BENCHMARK_ALIGN_BYTES
    uc_volume_root: str | None = None

    def __post_init__(self) -> None:
        if not self.workspace_dir:
            raise ValueError("storage benchmark workspace_dir must be non-empty")
        if not self.output_json:
            raise ValueError("storage benchmark output_json must be non-empty")
        if not self.benchmark_id:
            raise ValueError("storage benchmark benchmark_id must be non-empty")
        if self.chunk_count <= 0:
            raise ValueError("storage benchmark chunk_count must be positive")
        if self.chunk_bytes <= 0:
            raise ValueError("storage benchmark chunk_bytes must be positive")
        if self.repeats <= 0:
            raise ValueError("storage benchmark repeats must be positive")
        if self.parallelism <= 0:
            raise ValueError("storage benchmark parallelism must be positive")
        if type(self.align_bytes) is not int or self.align_bytes <= 0:
            raise ValueError("storage benchmark align_bytes must be a positive integer")
        if not self.readers:
            raise ValueError("storage benchmark readers must be non-empty")
        unsupported = sorted(set(self.readers).difference(SUPPORTED_STORAGE_BENCHMARK_READERS))
        if unsupported:
            raise ValueError(f"Unsupported storage benchmark readers: {unsupported}")
        if self.uc_volume_root is not None and not self.uc_volume_root:
            raise ValueError("storage benchmark uc_volume_root must be non-empty when provided")
        if "unity_catalog" in self.readers and not self.uc_volume_root:
            raise ValueError("storage benchmark unity_catalog reader requires uc_volume_root")
        object.__setattr__(self, "readers", tuple(self.readers))


@dataclass(frozen=True, slots=True)
class ReleaseEvidencePlanConfig:
    output_json: str
    engine_probe_jsons: tuple[str, ...] = ()
    engine_actions_jsons: tuple[str, ...] = ()
    storage_benchmark_json: str | None = None

    def __post_init__(self) -> None:
        if not self.output_json:
            raise ValueError("release evidence output_json must be non-empty")
        if any(not path for path in self.engine_probe_jsons):
            raise ValueError("release evidence engine_probe_jsons entries must be non-empty")
        canonical_probe_paths = tuple(_canonical_artifact_path(path) for path in self.engine_probe_jsons)
        if len(set(canonical_probe_paths)) != len(canonical_probe_paths):
            raise ValueError("release evidence engine_probe_jsons entries must be distinct")
        if any(not path for path in self.engine_actions_jsons):
            raise ValueError("release evidence engine_actions_jsons entries must be non-empty")
        canonical_action_paths = tuple(_canonical_artifact_path(path) for path in self.engine_actions_jsons)
        if len(set(canonical_action_paths)) != len(canonical_action_paths):
            raise ValueError("release evidence engine_actions_jsons entries must be distinct")
        if self.storage_benchmark_json is not None and not self.storage_benchmark_json:
            raise ValueError("release evidence storage_benchmark_json must be non-empty when provided")
        object.__setattr__(self, "engine_probe_jsons", tuple(self.engine_probe_jsons))
        object.__setattr__(self, "engine_actions_jsons", tuple(self.engine_actions_jsons))


@dataclass(frozen=True, slots=True)
class ReleaseBundlePlanConfig:
    output_dir: str
    output_json: str
    preflight_json: str | None = None
    plan_execution_jsons: tuple[str, ...] = ()
    databricks_run_status_jsons: tuple[str, ...] = ()
    package_wheel: str | None = None
    pr_evidence_jsons: tuple[str, ...] = ()
    github_governance_json: str | None = None
    repository_hygiene_json: str | None = None
    native_probe_factories_jsons: tuple[str, ...] = ()
    overwrite: bool = False
    require_complete_v1: bool = False
    requirements_matrix_md: str | None = None
    engine_launch_config_jsons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("release bundle output_dir must be non-empty")
        if not self.output_json:
            raise ValueError("release bundle output_json must be non-empty")
        if self.preflight_json is not None and not self.preflight_json:
            raise ValueError("release bundle preflight_json must be non-empty when provided")
        if any(not path for path in self.plan_execution_jsons):
            raise ValueError("release bundle plan_execution_jsons entries must be non-empty")
        if any(not path for path in self.databricks_run_status_jsons):
            raise ValueError("release bundle databricks_run_status_jsons entries must be non-empty")
        if self.package_wheel is not None and not self.package_wheel:
            raise ValueError("release bundle package_wheel must be non-empty when provided")
        if any(not path for path in self.pr_evidence_jsons):
            raise ValueError("release bundle pr_evidence_jsons entries must be non-empty")
        if self.requirements_matrix_md is not None and not self.requirements_matrix_md:
            raise ValueError("release bundle requirements_matrix_md must be non-empty when provided")
        if self.github_governance_json is not None and not self.github_governance_json:
            raise ValueError("release bundle github_governance_json must be non-empty when provided")
        if self.repository_hygiene_json is not None and not self.repository_hygiene_json:
            raise ValueError("release bundle repository_hygiene_json must be non-empty when provided")
        if any(not path for path in self.native_probe_factories_jsons):
            raise ValueError("release bundle native_probe_factories_jsons entries must be non-empty")
        if any(not path for path in self.engine_launch_config_jsons):
            raise ValueError("release bundle engine_launch_config_jsons entries must be non-empty")
        canonical_launch_config_paths = tuple(
            _canonical_artifact_path(path) for path in self.engine_launch_config_jsons
        )
        if len(set(canonical_launch_config_paths)) != len(canonical_launch_config_paths):
            raise ValueError("release bundle engine_launch_config_jsons entries must be distinct")
        if type(self.overwrite) is not bool:
            raise ValueError("release bundle overwrite must be boolean")
        if type(self.require_complete_v1) is not bool:
            raise ValueError("release bundle require_complete_v1 must be boolean")
        object.__setattr__(self, "plan_execution_jsons", tuple(self.plan_execution_jsons))
        object.__setattr__(self, "databricks_run_status_jsons", tuple(self.databricks_run_status_jsons))
        object.__setattr__(self, "pr_evidence_jsons", tuple(self.pr_evidence_jsons))
        object.__setattr__(self, "native_probe_factories_jsons", tuple(self.native_probe_factories_jsons))
        object.__setattr__(self, "engine_launch_config_jsons", tuple(self.engine_launch_config_jsons))


@dataclass(frozen=True, slots=True)
class EngineProbePlanConfig:
    backend: ServingBackend | str
    handoff_json: str
    probe_factory: str
    output_json: str
    actions_output_json: str | None = None
    payload_uri: str | None = None
    engine_version: str | None = None
    allow_non_native_probe: bool = False
    metadata: tuple[str, ...] = ()
    native_probe_delegate_factory: str | None = None
    fixture_output_dir: str | None = None
    fixture_payload_mode: PayloadMode | str = PayloadMode.SEGMENTED

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", ServingBackend(self.backend))
        object.__setattr__(self, "fixture_payload_mode", PayloadMode(self.fixture_payload_mode))
        if not self.handoff_json:
            raise ValueError("engine probe handoff_json must be non-empty")
        if not self.probe_factory:
            raise ValueError("engine probe probe_factory must be non-empty")
        if not self.output_json:
            raise ValueError("engine probe output_json must be non-empty")
        if self.actions_output_json is not None and not self.actions_output_json:
            raise ValueError("engine probe actions_output_json must be non-empty when provided")
        if self.native_probe_delegate_factory is not None and not self.native_probe_delegate_factory:
            raise ValueError("engine probe native_probe_delegate_factory must be non-empty when provided")
        if self.fixture_output_dir is not None and not self.fixture_output_dir:
            raise ValueError("engine probe fixture_output_dir must be non-empty when provided")
        if self.fixture_output_dir is not None:
            if self.actions_output_json is None:
                object.__setattr__(
                    self,
                    "actions_output_json",
                    _engine_probe_fixture_actions_json(self.fixture_output_dir),
                )
            expected_handoff_json = _engine_probe_fixture_handoff_json(self.fixture_output_dir)
            if not _same_artifact_path(self.handoff_json, expected_handoff_json):
                raise ValueError(
                    "engine probe handoff_json must match the derived fixture handoff path when "
                    f"fixture_output_dir is set: expected {expected_handoff_json!r}, got {self.handoff_json!r}"
                )
        if self.payload_uri is not None and not self.payload_uri:
            raise ValueError("engine probe payload_uri must be non-empty when provided")
        if self.fixture_output_dir is not None and self.payload_uri is not None:
            expected_payload_uri = _engine_probe_fixture_payload_uri(self.fixture_output_dir)
            if not _same_artifact_path(self.payload_uri, expected_payload_uri):
                raise ValueError(
                    "engine probe payload_uri must match the derived fixture payload path when "
                    f"fixture_output_dir is set: expected {expected_payload_uri!r}, got {self.payload_uri!r}"
                )
        if self.engine_version is not None and not self.engine_version:
            raise ValueError("engine probe engine_version must be non-empty when provided")
        if type(self.allow_non_native_probe) is not bool:
            raise ValueError("engine probe allow_non_native_probe must be boolean")
        if any(not _is_metadata_item(item) for item in self.metadata):
            raise ValueError("engine probe metadata entries must be non-empty KEY=VALUE strings")
        object.__setattr__(self, "metadata", tuple(self.metadata))


@dataclass(frozen=True, slots=True)
class BenchmarkPlanConfig:
    suite_id: str
    dataset_paths: tuple[BenchmarkDatasetPath, ...]
    base_url: str
    benchmark_output_json: str
    cache_base_url: str | None = None
    model_id: str = DEFAULT_V1_MODEL_ID
    hardware_target: str = DEFAULT_HARDWARE_TARGET
    python_executable: str = "python"
    require_all_v1_datasets: bool = True
    limit_per_dataset: int | None = None
    max_tokens: int = 128
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    stream: bool = True
    cache_runtime_prompt: bool = False
    server_usage: bool = False
    baseline_extra_body_json: str | None = None
    cache_extra_body_json: str | None = None
    storage_benchmark: StorageBenchmarkPlanConfig | None = None
    engine_probes: tuple[EngineProbePlanConfig, ...] = ()
    release_evidence: ReleaseEvidencePlanConfig | None = None
    release_bundle: ReleaseBundlePlanConfig | None = None
    native_probe_factories_output_json: str | None = None
    repository_hygiene_output_json: str | None = None
    github_governance_output_json: str | None = None
    release_preflight_output_json: str | None = None
    engine_launch_config_output_dir: str | None = None

    def __post_init__(self) -> None:
        if not self.suite_id:
            raise ValueError("suite_id must be non-empty")
        if not self.dataset_paths:
            raise ValueError("dataset_paths must be non-empty")
        if len({path.dataset for path in self.dataset_paths}) != len(self.dataset_paths):
            raise ValueError("dataset_paths must not contain duplicate datasets")
        if self.require_all_v1_datasets:
            missing = set(SUPPORTED_V1_DATASETS).difference(path.dataset for path in self.dataset_paths)
            if missing:
                raise ValueError(f"dataset_paths missing required V1 datasets: {sorted(missing)}")
        if not self.base_url:
            raise ValueError("base_url must be non-empty")
        if self.cache_base_url is not None and not self.cache_base_url:
            raise ValueError("cache_base_url must be non-empty when provided")
        if not self.benchmark_output_json:
            raise ValueError("benchmark_output_json must be non-empty")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.hardware_target:
            raise ValueError("hardware_target must be non-empty")
        validate_v1_hardware_target(self.hardware_target)
        if not self.python_executable:
            raise ValueError("python_executable must be non-empty")
        if self.limit_per_dataset is not None and self.limit_per_dataset <= 0:
            raise ValueError("limit_per_dataset must be positive when provided")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.cache_runtime_prompt and self.cache_base_url is None:
            raise ValueError("cache_runtime_prompt requires cache_base_url")
        if self.github_governance_output_json is not None and not self.github_governance_output_json:
            raise ValueError("github_governance_output_json must be non-empty when provided")
        if self.repository_hygiene_output_json is not None and not self.repository_hygiene_output_json:
            raise ValueError("repository_hygiene_output_json must be non-empty when provided")
        if self.native_probe_factories_output_json is not None and not self.native_probe_factories_output_json:
            raise ValueError("native_probe_factories_output_json must be non-empty when provided")
        if self.release_preflight_output_json is not None and not self.release_preflight_output_json:
            raise ValueError("release_preflight_output_json must be non-empty when provided")
        if self.engine_launch_config_output_dir is not None and not self.engine_launch_config_output_dir:
            raise ValueError("engine_launch_config_output_dir must be non-empty when provided")
        object.__setattr__(self, "engine_probes", tuple(self.engine_probes))
        if len({probe.backend for probe in self.engine_probes}) != len(self.engine_probes):
            raise ValueError("engine_probes must not contain duplicate backends")
        if len({probe.output_json for probe in self.engine_probes}) != len(self.engine_probes):
            raise ValueError("engine_probes must not contain duplicate output_json paths")
        _validate_generated_artifact_output_paths(self)
        _validate_release_bundle_github_governance_path(self)
        _validate_release_bundle_repository_hygiene_path(self)
        _validate_release_bundle_preflight_path(self)
        _validate_release_bundle_engine_launch_config_paths(self)
        if self.release_preflight_output_json is not None and self.release_evidence is None:
            raise ValueError("release_preflight_output_json requires release_evidence")
        if (
            self.release_evidence is not None
            and not self.release_evidence.engine_probe_jsons
            and not self.engine_probes
        ):
            raise ValueError("release_evidence requires engine_probe_jsons or planned engine_probes")
        if (
            self.release_evidence is not None
            and not self.release_evidence.engine_actions_jsons
            and not self.engine_probes
        ):
            raise ValueError("release_evidence requires engine_actions_jsons or planned engine_probes")
        if (
            self.release_evidence is not None
            and not self.release_evidence.engine_probe_jsons
            and self.engine_probes
        ):
            _validate_release_planned_engine_probes(self.engine_probes)
            missing_backends = set(_required_engine_probe_backends()).difference(
                probe.backend.value for probe in self.engine_probes
            )
            if missing_backends:
                raise ValueError(
                    "release_evidence requires planned engine_probes for all release backends: "
                    f"{sorted(missing_backends)}"
                )
        if (
            self.release_evidence is not None
            and not self.release_evidence.engine_actions_jsons
            and self.engine_probes
        ):
            _validate_release_planned_engine_probes(self.engine_probes)
            _validate_release_planned_engine_probe_actions(self.engine_probes)
            missing_backends = set(_required_engine_probe_backends()).difference(
                probe.backend.value for probe in self.engine_probes
            )
            if missing_backends:
                raise ValueError(
                    "release_evidence requires planned engine_probes for all release backends: "
                    f"{sorted(missing_backends)}"
                )
        if (
            self.release_evidence is not None
            and self.release_evidence.engine_probe_jsons
            and len(self.release_evidence.engine_probe_jsons) != len(_required_engine_probe_backends())
        ):
            raise ValueError(
                "release_evidence requires explicit engine_probe_jsons for all release backends: "
                f"{list(_required_engine_probe_backends())}"
            )
        if (
            self.release_evidence is not None
            and self.release_evidence.engine_actions_jsons
            and len(self.release_evidence.engine_actions_jsons) != len(_required_engine_probe_backends())
        ):
            raise ValueError(
                "release_evidence requires explicit engine_actions_jsons for all release backends: "
                f"{list(_required_engine_probe_backends())}"
            )
        if self.release_evidence is not None and self.release_evidence.engine_probe_jsons:
            _validate_explicit_release_probe_paths_do_not_use_planned_debug_outputs(
                self.release_evidence.engine_probe_jsons,
                self.engine_probes,
            )
        if self.release_evidence is not None and self.release_evidence.engine_actions_jsons:
            _validate_explicit_release_action_paths_do_not_use_planned_debug_outputs(
                self.release_evidence.engine_actions_jsons,
                self.engine_probes,
            )
        if (
            self.release_evidence is not None
            and self.storage_benchmark is None
            and self.release_evidence.storage_benchmark_json is None
        ):
            raise ValueError("release_evidence requires storage_benchmark or storage_benchmark_json")
        if (
            self.release_evidence is not None
            and self.release_evidence.storage_benchmark_json is None
            and self.storage_benchmark is not None
            and not _uses_release_storage_benchmark_readers(self.storage_benchmark.readers)
        ):
            raise ValueError("release_evidence requires the planned storage_benchmark to use release readers")
        if self.release_bundle is not None and self.release_evidence is None:
            raise ValueError("release_bundle requires release_evidence")
        _validate_strict_v1_release_bundle_plan(self)


@dataclass(frozen=True, slots=True)
class BenchmarkJobPlan:
    config: BenchmarkPlanConfig
    preparation_commands: tuple[BenchmarkCommand, ...]
    benchmark_command: BenchmarkCommand
    post_benchmark_commands: tuple[BenchmarkCommand, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def commands(self) -> tuple[BenchmarkCommand, ...]:
        return (*self.preparation_commands, self.benchmark_command, *self.post_benchmark_commands)


def build_v1_benchmark_plan(config: BenchmarkPlanConfig) -> BenchmarkJobPlan:
    preparation_commands = tuple(_dataset_prep_command(config, dataset_path) for dataset_path in config.dataset_paths)
    post_benchmark_commands = []
    if config.storage_benchmark is not None:
        post_benchmark_commands.append(_storage_benchmark_command(config))
    for probe_config in config.engine_probes:
        if probe_config.fixture_output_dir is not None:
            post_benchmark_commands.append(_engine_probe_fixture_command(config, probe_config))
        post_benchmark_commands.append(_engine_probe_command(config, probe_config))
    if config.github_governance_output_json is not None:
        post_benchmark_commands.append(_github_governance_command(config))
    if config.repository_hygiene_output_json is not None:
        post_benchmark_commands.append(_repository_hygiene_command(config))
    if config.native_probe_factories_output_json is not None:
        post_benchmark_commands.append(_native_probe_factories_command(config))
    if config.engine_launch_config_output_dir is not None:
        for backend in ServingBackend:
            post_benchmark_commands.append(_engine_launch_config_command(config, backend))
    if config.release_preflight_output_json is not None:
        post_benchmark_commands.append(_release_preflight_command(config))
    if config.release_evidence is not None:
        post_benchmark_commands.append(_release_evidence_command(config))
    if config.release_bundle is not None:
        post_benchmark_commands.append(_release_bundle_command(config))
    return BenchmarkJobPlan(
        config=config,
        preparation_commands=preparation_commands,
        benchmark_command=_benchmark_runner_command(config),
        post_benchmark_commands=tuple(post_benchmark_commands),
        notes=(
            "Run these commands on an AWS g6/L4-compatible environment with the target server already listening.",
            "The benchmark compares baseline full-prefill requests with the document KV-cache arm.",
            "When configured, the storage-reader benchmark runs after inference to capture selected reader load evidence on the same node.",
            "When configured, GitHub governance inspection runs before release validation and can be bundled as release governance evidence.",
            "When configured, repository hygiene inspection runs before release validation and can be bundled as release hygiene evidence.",
            "When configured, native probe factory diagnostics run before release validation and can be bundled as release handoff evidence.",
            "When configured, vLLM/SGLang launch-config sidecars are generated before release bundle assembly.",
            "When configured, deterministic Qwen3 engine-probe fixtures are generated immediately before native engine probes.",
            "When configured, release-evidence preflight runs before release validation and can be bundled as release input evidence.",
            "When configured, release evidence validation runs last and checks V1, storage, and native engine-probe artifacts together.",
            "When configured, release bundle assembly follows release evidence and copies the validated artifacts plus optional sidecars into a checksummed handoff directory.",
        ),
    )


def benchmark_job_plan_to_record(plan: BenchmarkJobPlan) -> dict[str, Any]:
    return {
        "plan_version": PLAN_VERSION,
        "suite_id": plan.config.suite_id,
        "model_id": plan.config.model_id,
        "hardware_target": plan.config.hardware_target,
        "require_all_v1_datasets": plan.config.require_all_v1_datasets,
        "datasets": [
            {
                "dataset": dataset_path.dataset,
                "raw_jsonl": dataset_path.raw_jsonl,
                "prepared_jsonl": dataset_path.prepared_jsonl,
            }
            for dataset_path in plan.config.dataset_paths
        ],
        "commands": [_command_to_record(command) for command in plan.commands],
        "benchmark_output_json": plan.config.benchmark_output_json,
        "storage_benchmark_output_json": (
            plan.config.storage_benchmark.output_json
            if plan.config.storage_benchmark is not None
            else None
        ),
        "release_evidence_output_json": (
            plan.config.release_evidence.output_json
            if plan.config.release_evidence is not None
            else None
        ),
        "release_storage_benchmark_json": (
            _release_storage_benchmark_json(plan.config)
            if plan.config.release_evidence is not None
            else None
        ),
        "release_engine_probe_jsons": (
            list(_release_engine_probe_jsons(plan.config))
            if plan.config.release_evidence is not None
            else []
        ),
        "release_engine_actions_jsons": (
            list(_release_engine_action_jsons(plan.config))
            if plan.config.release_evidence is not None
            else []
        ),
        "release_bundle_output_dir": (
            plan.config.release_bundle.output_dir
            if plan.config.release_bundle is not None
            else None
        ),
        "release_bundle_output_json": (
            plan.config.release_bundle.output_json
            if plan.config.release_bundle is not None
            else None
        ),
        "release_bundle": (
            _release_bundle_plan_to_record(plan.config)
            if plan.config.release_bundle is not None
            else None
        ),
        "github_governance_output_json": plan.config.github_governance_output_json,
        "repository_hygiene_output_json": plan.config.repository_hygiene_output_json,
        "native_probe_factories_output_json": plan.config.native_probe_factories_output_json,
        "release_preflight_output_json": plan.config.release_preflight_output_json,
        "engine_launch_config_output_dir": plan.config.engine_launch_config_output_dir,
        "planned_engine_probes": [_planned_engine_probe_to_record(probe) for probe in plan.config.engine_probes],
        "notes": list(plan.notes),
    }


def _planned_engine_probe_to_record(probe: EngineProbePlanConfig) -> dict[str, Any]:
    record: dict[str, Any] = {
        "backend": probe.backend.value,
        "handoff_json": probe.handoff_json,
        "probe_factory": probe.probe_factory,
        "output_json": probe.output_json,
        "actions_output_json": probe.actions_output_json,
        "payload_uri": probe.payload_uri,
        "engine_version": probe.engine_version,
        "allow_non_native_probe": probe.allow_non_native_probe,
        "metadata": list(probe.metadata),
    }
    if probe.native_probe_delegate_factory is not None:
        record["native_probe_delegate_factory"] = probe.native_probe_delegate_factory
    if probe.fixture_output_dir is not None:
        record["fixture_output_dir"] = probe.fixture_output_dir
        record["fixture_payload_mode"] = probe.fixture_payload_mode.value
    return record


def engine_probe_targets_to_record(
    engine_probes: Sequence[EngineProbePlanConfig],
    *,
    release_safe: bool = False,
) -> dict[str, Any]:
    """Serialize planned native probes as Databricks matrix helper input."""

    probes = tuple(engine_probes)
    _validate_engine_probe_targets(probes, release_safe=release_safe)
    return {
        "record_type": ENGINE_PROBE_TARGETS_RECORD_TYPE,
        "schema_version": ENGINE_PROBE_TARGETS_SCHEMA_VERSION,
        "release_safe": release_safe,
        "probes": [_engine_probe_target_to_record(probe) for probe in probes],
    }


def write_engine_probe_targets_json(
    plan: BenchmarkJobPlan,
    path: str | Path,
    *,
    release_safe: bool = False,
) -> None:
    Path(path).write_text(
        json.dumps(
            engine_probe_targets_to_record(plan.config.engine_probes, release_safe=release_safe),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_benchmark_job_plan_json(plan: BenchmarkJobPlan, path: str | Path) -> None:
    Path(path).write_text(json.dumps(benchmark_job_plan_to_record(plan), indent=2, sort_keys=True) + "\n")


def write_benchmark_job_plan_shell(plan: BenchmarkJobPlan, path: str | Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        *(command.shell for command in plan.commands),
        "",
    ]
    output_path = Path(path)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    output_path.chmod(0o755)


def _dataset_prep_command(config: BenchmarkPlanConfig, dataset_path: BenchmarkDatasetPath) -> BenchmarkCommand:
    argv = (
        config.python_executable,
        "-m",
        "document_kv_cache.dataset_prep",
        "--dataset",
        dataset_path.dataset,
        "--input-jsonl",
        dataset_path.raw_jsonl,
        "--output-jsonl",
        dataset_path.prepared_jsonl,
    )
    if config.limit_per_dataset is not None:
        argv = (*argv, "--limit", str(config.limit_per_dataset))
    return BenchmarkCommand(name=f"prepare-{dataset_path.dataset}", argv=argv)


def _benchmark_runner_command(config: BenchmarkPlanConfig) -> BenchmarkCommand:
    argv = (
        config.python_executable,
        "-m",
        "document_kv_cache.benchmark_runner",
        "--suite-id",
        config.suite_id,
        "--base-url",
        config.base_url,
        "--model-id",
        config.model_id,
        "--hardware-target",
        config.hardware_target,
        "--max-tokens",
        str(config.max_tokens),
        "--temperature",
        str(config.temperature),
        "--timeout-seconds",
        str(config.timeout_seconds),
        "--output-json",
        config.benchmark_output_json,
    )
    for dataset_path in config.dataset_paths:
        argv = (*argv, "--dataset", f"{dataset_path.dataset}={dataset_path.prepared_jsonl}")
    if config.cache_base_url is not None:
        argv = (*argv, "--cache-base-url", config.cache_base_url)
    if config.limit_per_dataset is not None:
        argv = (*argv, "--limit-per-dataset", str(config.limit_per_dataset))
    if not config.stream:
        argv = (*argv, "--no-stream")
    if config.cache_runtime_prompt:
        argv = (*argv, "--cache-runtime-prompt")
    if config.server_usage:
        argv = (*argv, "--server-usage")
    if config.baseline_extra_body_json is not None:
        argv = (*argv, "--baseline-extra-body-json", config.baseline_extra_body_json)
    if config.cache_extra_body_json is not None:
        argv = (*argv, "--cache-extra-body-json", config.cache_extra_body_json)
    return BenchmarkCommand(name="run-benchmark", argv=argv)


def _storage_benchmark_command(config: BenchmarkPlanConfig) -> BenchmarkCommand:
    storage_config = config.storage_benchmark
    if storage_config is None:
        raise ValueError("storage_benchmark must be configured")
    argv = (
        config.python_executable,
        "-m",
        "document_kv_cache.storage_benchmark",
        "--workspace-dir",
        storage_config.workspace_dir,
        "--benchmark-id",
        storage_config.benchmark_id,
        "--chunk-count",
        str(storage_config.chunk_count),
        "--chunk-bytes",
        str(storage_config.chunk_bytes),
        "--repeats",
        str(storage_config.repeats),
        "--parallelism",
        str(storage_config.parallelism),
        "--align-bytes",
        str(storage_config.align_bytes),
        "--output-json",
        storage_config.output_json,
    )
    for reader in storage_config.readers:
        argv = (*argv, "--reader", reader)
    if storage_config.uc_volume_root is not None:
        argv = (*argv, "--uc-volume-root", storage_config.uc_volume_root)
    return BenchmarkCommand(name="run-storage-benchmark", argv=argv)


def _engine_probe_command(config: BenchmarkPlanConfig, probe_config: EngineProbePlanConfig) -> BenchmarkCommand:
    argv = (
        config.python_executable,
        "-m",
        "document_kv_cache.engine_probe",
        "--handoff-json",
        probe_config.handoff_json,
        "--probe-factory",
        probe_config.probe_factory,
        "--output-json",
        probe_config.output_json,
        "--expected-backend",
        probe_config.backend.value,
    )
    if probe_config.payload_uri is not None:
        argv = (*argv, "--payload-uri", probe_config.payload_uri)
    if probe_config.actions_output_json is not None and not _engine_probe_uses_fixture_actions_output(probe_config):
        argv = (*argv, "--actions-output-json", probe_config.actions_output_json)
    if probe_config.engine_version is not None:
        argv = (*argv, "--engine-version", probe_config.engine_version)
    if probe_config.allow_non_native_probe:
        argv = (*argv, "--allow-non-native-probe")
    for metadata in probe_config.metadata:
        argv = (*argv, "--metadata", metadata)
    return BenchmarkCommand(name=f"run-{probe_config.backend.value}-engine-probe", argv=argv)


def _engine_probe_fixture_command(config: BenchmarkPlanConfig, probe_config: EngineProbePlanConfig) -> BenchmarkCommand:
    if probe_config.fixture_output_dir is None:
        raise ValueError("engine probe fixture_output_dir must be configured")
    return BenchmarkCommand(
        name=f"write-{probe_config.backend.value}-engine-probe-fixture",
        argv=(
            config.python_executable,
            "-m",
            "document_kv_cache.probe_fixtures",
            "--output-dir",
            probe_config.fixture_output_dir,
            "--backend",
            probe_config.backend.value,
            "--payload-mode",
            probe_config.fixture_payload_mode.value,
        ),
    )


def _github_governance_command(config: BenchmarkPlanConfig) -> BenchmarkCommand:
    if config.github_governance_output_json is None:
        raise ValueError("github_governance_output_json must be configured")
    return BenchmarkCommand(
        name="inspect-github-governance",
        argv=(
            config.python_executable,
            "-m",
            "document_kv_cache.github_governance",
            "--output-json",
            config.github_governance_output_json,
        ),
    )


def _repository_hygiene_command(config: BenchmarkPlanConfig) -> BenchmarkCommand:
    if config.repository_hygiene_output_json is None:
        raise ValueError("repository_hygiene_output_json must be configured")
    return BenchmarkCommand(
        name="inspect-repository-hygiene",
        argv=(
            config.python_executable,
            "-m",
            "document_kv_cache.repository_hygiene",
            "--repository-root",
            ".",
            "--output-json",
            config.repository_hygiene_output_json,
        ),
    )


def _native_probe_factories_command(config: BenchmarkPlanConfig) -> BenchmarkCommand:
    if config.native_probe_factories_output_json is None:
        raise ValueError("native_probe_factories_output_json must be configured")
    return BenchmarkCommand(
        name="inspect-native-probe-factories",
        argv=(
            config.python_executable,
            "-m",
            "document_kv_cache.native_probe_factories",
            "--output-json",
            config.native_probe_factories_output_json,
        ),
    )


def _engine_launch_config_command(config: BenchmarkPlanConfig, backend: ServingBackend) -> BenchmarkCommand:
    if config.engine_launch_config_output_dir is None:
        raise ValueError("engine_launch_config_output_dir must be configured")
    return BenchmarkCommand(
        name=f"write-{backend.value}-engine-launch-config",
        argv=(
            config.python_executable,
            "-m",
            "document_kv_cache.engine_launch_config",
            f"build-{backend.value}",
            "--output-json",
            _engine_launch_config_json(config.engine_launch_config_output_dir, backend),
        ),
    )


def _release_evidence_command(config: BenchmarkPlanConfig) -> BenchmarkCommand:
    release_config = config.release_evidence
    if release_config is None:
        raise ValueError("release_evidence must be configured")
    argv = (
        *_release_evidence_input_argv(config),
        "--output-json",
        release_config.output_json,
    )
    return BenchmarkCommand(name="validate-release-evidence", argv=argv)


def _release_preflight_command(config: BenchmarkPlanConfig) -> BenchmarkCommand:
    if config.release_preflight_output_json is None:
        raise ValueError("release_preflight_output_json must be configured")
    argv = (
        *_release_evidence_input_argv(config),
        "--preflight-only",
        "--preflight-output-json",
        config.release_preflight_output_json,
    )
    return BenchmarkCommand(name="preflight-release-evidence", argv=argv)


def _release_evidence_input_argv(config: BenchmarkPlanConfig) -> tuple[str, ...]:
    if config.release_evidence is None:
        raise ValueError("release_evidence must be configured")
    argv = (
        config.python_executable,
        "-m",
        "document_kv_cache.release_evidence",
        "--v1-benchmark-json",
        config.benchmark_output_json,
        "--storage-benchmark-json",
        _release_storage_benchmark_json(config),
    )
    for engine_probe_json in _release_engine_probe_jsons(config):
        argv = (*argv, "--engine-probe-json", engine_probe_json)
    for engine_action_json in _release_engine_action_jsons(config):
        argv = (*argv, "--engine-actions-json", engine_action_json)
    return argv


def _release_bundle_command(config: BenchmarkPlanConfig) -> BenchmarkCommand:
    bundle_config = config.release_bundle
    release_config = config.release_evidence
    if bundle_config is None:
        raise ValueError("release_bundle must be configured")
    if release_config is None:
        raise ValueError("release_bundle requires release_evidence")
    argv = (
        config.python_executable,
        "-m",
        "document_kv_cache.release_bundle",
        "--v1-benchmark-json",
        config.benchmark_output_json,
        "--storage-benchmark-json",
        _release_storage_benchmark_json(config),
        "--release-evidence-json",
        release_config.output_json,
        "--output-dir",
        bundle_config.output_dir,
        "--output-json",
        bundle_config.output_json,
    )
    for engine_probe_json in _release_engine_probe_jsons(config):
        argv = (*argv, "--engine-probe-json", engine_probe_json)
    for engine_action_json in _release_engine_action_jsons(config):
        argv = (*argv, "--engine-actions-json", engine_action_json)
    for engine_launch_config_json in _release_bundle_engine_launch_config_jsons(config):
        argv = (*argv, "--engine-launch-config-json", engine_launch_config_json)
    preflight_json = _release_bundle_preflight_json(config)
    if preflight_json is not None:
        argv = (*argv, "--preflight-json", preflight_json)
    for plan_execution_json in bundle_config.plan_execution_jsons:
        argv = (*argv, "--plan-execution-json", plan_execution_json)
    for status_json in bundle_config.databricks_run_status_jsons:
        argv = (*argv, "--databricks-run-status-json", status_json)
    if bundle_config.package_wheel is not None:
        argv = (*argv, "--package-wheel", bundle_config.package_wheel)
    for pr_evidence_json in bundle_config.pr_evidence_jsons:
        argv = (*argv, "--pr-evidence-json", pr_evidence_json)
    if bundle_config.requirements_matrix_md is not None:
        argv = (*argv, "--requirements-matrix-md", bundle_config.requirements_matrix_md)
    github_governance_json = _release_bundle_github_governance_json(config)
    if github_governance_json is not None:
        argv = (*argv, "--github-governance-json", github_governance_json)
    repository_hygiene_json = _release_bundle_repository_hygiene_json(config)
    if repository_hygiene_json is not None:
        argv = (*argv, "--repository-hygiene-json", repository_hygiene_json)
    for native_probe_factories_json in _release_bundle_native_probe_factories_jsons(config):
        argv = (*argv, "--native-probe-factories-json", native_probe_factories_json)
    if bundle_config.require_complete_v1:
        argv = (*argv, "--require-complete-v1")
    if bundle_config.overwrite:
        argv = (*argv, "--overwrite")
    return BenchmarkCommand(name="build-release-bundle", argv=argv)


def _release_bundle_plan_to_record(config: BenchmarkPlanConfig) -> dict[str, Any]:
    bundle_config = config.release_bundle
    release_config = config.release_evidence
    if bundle_config is None:
        raise ValueError("release_bundle must be configured")
    if release_config is None:
        raise ValueError("release_bundle requires release_evidence")
    return {
        "output_dir": bundle_config.output_dir,
        "output_json": bundle_config.output_json,
        "v1_benchmark_json": config.benchmark_output_json,
        "storage_benchmark_json": _release_storage_benchmark_json(config),
        "engine_probe_jsons": list(_release_engine_probe_jsons(config)),
        "engine_actions_jsons": list(_release_engine_action_jsons(config)),
        "engine_launch_config_jsons": list(_release_bundle_engine_launch_config_jsons(config)),
        "release_evidence_json": release_config.output_json,
        "preflight_json": _release_bundle_preflight_json(config),
        "plan_execution_jsons": list(bundle_config.plan_execution_jsons),
        "databricks_run_status_jsons": list(bundle_config.databricks_run_status_jsons),
        "package_wheel": bundle_config.package_wheel,
        "pr_evidence_jsons": list(bundle_config.pr_evidence_jsons),
        "requirements_matrix_md": bundle_config.requirements_matrix_md,
        "github_governance_json": _release_bundle_github_governance_json(config),
        "repository_hygiene_json": _release_bundle_repository_hygiene_json(config),
        "native_probe_factories_jsons": list(_release_bundle_native_probe_factories_jsons(config)),
        "overwrite": bundle_config.overwrite,
        "require_complete_v1": bundle_config.require_complete_v1,
    }


def _release_bundle_github_governance_json(config: BenchmarkPlanConfig) -> str | None:
    bundle_config = config.release_bundle
    if bundle_config is None:
        return None
    if config.github_governance_output_json is not None:
        return config.github_governance_output_json
    return bundle_config.github_governance_json


def _release_bundle_preflight_json(config: BenchmarkPlanConfig) -> str | None:
    bundle_config = config.release_bundle
    if bundle_config is None:
        return None
    if config.release_preflight_output_json is not None:
        return config.release_preflight_output_json
    return bundle_config.preflight_json


def _validate_release_bundle_preflight_path(config: BenchmarkPlanConfig) -> None:
    bundle_config = config.release_bundle
    if bundle_config is None:
        return
    _validate_release_bundle_single_sidecar_path(
        release_bundle_label="release bundle preflight_json",
        generated_label="release_preflight_output_json",
        generated_path=config.release_preflight_output_json,
        explicit_path=bundle_config.preflight_json,
    )


def _validate_release_bundle_engine_launch_config_paths(config: BenchmarkPlanConfig) -> None:
    bundle_config = config.release_bundle
    if (
        bundle_config is None
        or config.engine_launch_config_output_dir is None
        or not bundle_config.engine_launch_config_jsons
    ):
        return
    generated_paths = {
        _canonical_artifact_path(path)
        for path in _generated_engine_launch_config_jsons(config)
    }
    explicit_paths = {
        _canonical_artifact_path(path)
        for path in bundle_config.engine_launch_config_jsons
    }
    if explicit_paths != generated_paths:
        raise ValueError(
            "release bundle engine_launch_config_jsons must match engine_launch_config_output_dir "
            "when both are provided"
        )


def _validate_release_bundle_github_governance_path(config: BenchmarkPlanConfig) -> None:
    bundle_config = config.release_bundle
    if bundle_config is None:
        return
    _validate_release_bundle_single_sidecar_path(
        release_bundle_label="release bundle github_governance_json",
        generated_label="github_governance_output_json",
        generated_path=config.github_governance_output_json,
        explicit_path=bundle_config.github_governance_json,
    )


def _release_bundle_repository_hygiene_json(config: BenchmarkPlanConfig) -> str | None:
    bundle_config = config.release_bundle
    if bundle_config is None:
        return None
    if config.repository_hygiene_output_json is not None:
        return config.repository_hygiene_output_json
    return bundle_config.repository_hygiene_json


def _validate_release_bundle_repository_hygiene_path(config: BenchmarkPlanConfig) -> None:
    bundle_config = config.release_bundle
    if bundle_config is None:
        return
    _validate_release_bundle_single_sidecar_path(
        release_bundle_label="release bundle repository_hygiene_json",
        generated_label="repository_hygiene_output_json",
        generated_path=config.repository_hygiene_output_json,
        explicit_path=bundle_config.repository_hygiene_json,
    )


def _validate_strict_v1_release_bundle_plan(config: BenchmarkPlanConfig) -> None:
    bundle_config = config.release_bundle
    if bundle_config is None or not bundle_config.require_complete_v1:
        return

    missing = []
    if _release_bundle_preflight_json(config) is None:
        missing.append("preflight sidecar")
    if not bundle_config.plan_execution_jsons:
        missing.append("benchmark plan execution sidecar")
    if (
        len(bundle_config.databricks_run_status_jsons) != STRICT_V1_DATABRICKS_RUN_STATUS_SIDECAR_COUNT
        or len(set(bundle_config.databricks_run_status_jsons)) != len(bundle_config.databricks_run_status_jsons)
    ):
        missing.append(STRICT_V1_DATABRICKS_RUN_STATUS_SIDECAR_LABEL)
    if bundle_config.package_wheel is None:
        missing.append("tested package wheel")
    if not bundle_config.pr_evidence_jsons:
        missing.append("PR evidence sidecar")
    if not _has_strict_release_bundle_engine_launch_config_sidecars(config):
        missing.append("vLLM/SGLang engine launch config sidecars")
    if bundle_config.requirements_matrix_md is None:
        missing.append("V1 requirements matrix")
    if _release_bundle_github_governance_json(config) is None:
        missing.append("GitHub governance sidecar")
    if _release_bundle_repository_hygiene_json(config) is None:
        missing.append("repository hygiene sidecar")
    if not _release_bundle_native_probe_factories_jsons(config):
        missing.append("native probe factory diagnostics sidecar")

    if missing:
        raise ValueError("strict V1 release bundle plans require " + ", ".join(missing))


def _validate_release_bundle_single_sidecar_path(
    *,
    release_bundle_label: str,
    generated_label: str,
    generated_path: str | None,
    explicit_path: str | None,
) -> None:
    if generated_path is None or explicit_path is None:
        return
    if _canonical_artifact_path(generated_path) != _canonical_artifact_path(explicit_path):
        raise ValueError(f"{release_bundle_label} must match {generated_label} when both are provided")


def _release_bundle_native_probe_factories_jsons(config: BenchmarkPlanConfig) -> tuple[str, ...]:
    bundle_config = config.release_bundle
    if bundle_config is None:
        return ()
    paths = []
    if config.native_probe_factories_output_json is not None:
        paths.append(config.native_probe_factories_output_json)
    paths.extend(bundle_config.native_probe_factories_jsons)
    return _dedupe_artifact_paths(paths)


def _release_bundle_engine_launch_config_jsons(config: BenchmarkPlanConfig) -> tuple[str, ...]:
    bundle_config = config.release_bundle
    if bundle_config is None:
        return ()
    paths = [*_generated_engine_launch_config_jsons(config)]
    paths.extend(bundle_config.engine_launch_config_jsons)
    return _dedupe_artifact_paths(paths)


def _has_strict_release_bundle_engine_launch_config_sidecars(config: BenchmarkPlanConfig) -> bool:
    return len(_release_bundle_engine_launch_config_jsons(config)) >= len(DEFAULT_ENGINE_LAUNCH_CONFIG_FILENAMES)


def _generated_engine_launch_config_jsons(config: BenchmarkPlanConfig) -> tuple[str, ...]:
    if config.engine_launch_config_output_dir is None:
        return ()
    return tuple(
        _engine_launch_config_json(config.engine_launch_config_output_dir, backend)
        for backend in ServingBackend
    )


def _engine_launch_config_json(output_dir: str, backend: ServingBackend) -> str:
    return _uri_child(output_dir, DEFAULT_ENGINE_LAUNCH_CONFIG_FILENAMES[backend])


def _dedupe_artifact_paths(paths: Sequence[str]) -> tuple[str, ...]:
    deduped = []
    seen_canonical_paths = set()
    for path in paths:
        canonical_path = _canonical_artifact_path(path)
        if canonical_path not in seen_canonical_paths:
            seen_canonical_paths.add(canonical_path)
            deduped.append(path)
    return tuple(deduped)


def _release_engine_probe_jsons(config: BenchmarkPlanConfig) -> tuple[str, ...]:
    release_config = config.release_evidence
    if release_config is None:
        raise ValueError("release_evidence must be configured")
    if release_config.engine_probe_jsons:
        return release_config.engine_probe_jsons
    if not config.engine_probes:
        raise ValueError("release_evidence requires engine_probe_jsons or planned engine_probes")
    return tuple(probe.output_json for probe in config.engine_probes)


def _release_engine_action_jsons(config: BenchmarkPlanConfig) -> tuple[str, ...]:
    release_config = config.release_evidence
    if release_config is None:
        raise ValueError("release_evidence must be configured")
    if release_config.engine_actions_jsons:
        return release_config.engine_actions_jsons
    if not config.engine_probes:
        raise ValueError("release_evidence requires engine_actions_jsons or planned engine_probes")
    if any(probe.actions_output_json is None for probe in config.engine_probes):
        raise ValueError("release_evidence requires planned engine probes to write connector action descriptors")
    return tuple(probe.actions_output_json for probe in config.engine_probes if probe.actions_output_json is not None)


def _release_storage_benchmark_json(config: BenchmarkPlanConfig) -> str:
    release_config = config.release_evidence
    if release_config is None:
        raise ValueError("release_evidence must be configured")
    if release_config.storage_benchmark_json is not None:
        return release_config.storage_benchmark_json
    storage_config = config.storage_benchmark
    if storage_config is None:
        raise ValueError("release_evidence requires storage_benchmark or storage_benchmark_json")
    return storage_config.output_json


def _uses_release_storage_benchmark_readers(readers: Sequence[str]) -> bool:
    return (
        len(readers) == len(RELEASE_STORAGE_BENCHMARK_READERS)
        and set(readers) == set(RELEASE_STORAGE_BENCHMARK_READERS)
    )


def _validate_release_planned_engine_probes(engine_probes: Sequence[EngineProbePlanConfig]) -> None:
    debug_engine_versions = sorted(probe.backend.value for probe in engine_probes if probe.engine_version is not None)
    non_native_probes = sorted(probe.backend.value for probe in engine_probes if probe.allow_non_native_probe)
    issues = []
    if debug_engine_versions:
        issues.append(f"engine_version overrides for {debug_engine_versions}")
    if non_native_probes:
        issues.append(f"non-native debug probes for {non_native_probes}")
    if issues:
        raise ValueError(
            "release_evidence cannot consume planned debug engine probes; remove "
            + " and ".join(issues)
            + " or pass explicit --release-engine-probe-json artifacts instead"
        )


def _validate_release_planned_engine_probe_actions(engine_probes: Sequence[EngineProbePlanConfig]) -> None:
    missing_actions = sorted(probe.backend.value for probe in engine_probes if probe.actions_output_json is None)
    if missing_actions:
        raise ValueError(
            "release_evidence requires planned engine probes to write connector action descriptors for "
            f"{missing_actions}; set actions_output_json or pass explicit --release-engine-actions-json artifacts"
        )


def _validate_explicit_release_probe_paths_do_not_use_planned_debug_outputs(
    release_engine_probe_jsons: Sequence[str],
    engine_probes: Sequence[EngineProbePlanConfig],
) -> None:
    release_paths = {_canonical_artifact_path(path) for path in release_engine_probe_jsons}
    unsafe_aliases = [
        f"{probe.backend.value}={probe.output_json}"
        for probe in engine_probes
        if (probe.engine_version is not None or probe.allow_non_native_probe)
        and _canonical_artifact_path(probe.output_json) in release_paths
    ]
    if unsafe_aliases:
        raise ValueError(
            "release_evidence explicit engine_probe_jsons must not point at planned debug engine probe outputs: "
            f"{sorted(unsafe_aliases)}"
        )


def _validate_explicit_release_action_paths_do_not_use_planned_debug_outputs(
    release_engine_actions_jsons: Sequence[str],
    engine_probes: Sequence[EngineProbePlanConfig],
) -> None:
    release_paths = {_canonical_artifact_path(path) for path in release_engine_actions_jsons}
    unsafe_aliases = [
        f"{probe.backend.value}={probe.actions_output_json}"
        for probe in engine_probes
        if probe.actions_output_json is not None
        and (probe.engine_version is not None or probe.allow_non_native_probe)
        and _canonical_artifact_path(probe.actions_output_json) in release_paths
    ]
    if unsafe_aliases:
        raise ValueError(
            "release_evidence explicit engine_actions_jsons must not point at planned debug engine probe outputs: "
            f"{sorted(unsafe_aliases)}"
        )


def _required_engine_probe_backends() -> tuple[str, ...]:
    return tuple(backend.value for backend in ServingBackend)


def _validate_generated_artifact_output_paths(config: BenchmarkPlanConfig) -> None:
    _validate_distinct_artifact_paths(_generated_artifact_output_paths(config))


def _validate_plan_output_paths(
    config: BenchmarkPlanConfig,
    *,
    plan_output_json: str | None,
    plan_output_sh: str | None,
    engine_probe_targets_output_json: str | None = None,
) -> None:
    output_paths = list(_generated_artifact_output_paths(config))
    if plan_output_json is not None:
        output_paths.append(("plan_output_json", plan_output_json))
    if plan_output_sh is not None:
        output_paths.append(("plan_output_sh", plan_output_sh))
    if engine_probe_targets_output_json is not None:
        output_paths.append(("engine_probe_targets_output_json", engine_probe_targets_output_json))
    _validate_distinct_artifact_paths(output_paths)


def _generated_artifact_output_paths(config: BenchmarkPlanConfig) -> tuple[tuple[str, str], ...]:
    output_paths: list[tuple[str, str]] = [("benchmark_output_json", config.benchmark_output_json)]
    if config.storage_benchmark is not None:
        output_paths.append(("storage_benchmark.output_json", config.storage_benchmark.output_json))
    output_paths.extend(
        (f"engine_probes[{probe.backend.value}].output_json", probe.output_json)
        for probe in config.engine_probes
    )
    output_paths.extend(
        (f"engine_probes[{probe.backend.value}].actions_output_json", probe.actions_output_json)
        for probe in config.engine_probes
        if probe.actions_output_json is not None and not _engine_probe_uses_fixture_actions_output(probe)
    )
    output_paths.extend(
        (f"engine_probes[{probe.backend.value}].fixture_output_dir", probe.fixture_output_dir)
        for probe in config.engine_probes
        if probe.fixture_output_dir is not None
    )
    for probe in config.engine_probes:
        if probe.fixture_output_dir is None:
            continue
        output_paths.extend(
            (
                f"engine_probes[{probe.backend.value}].fixture_{artifact_name}",
                _uri_child(probe.fixture_output_dir, filename),
            )
            for artifact_name, filename in DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES.items()
        )
    if config.release_evidence is not None:
        output_paths.append(("release_evidence.output_json", config.release_evidence.output_json))
    if config.release_bundle is not None:
        output_paths.append(("release_bundle.output_json", config.release_bundle.output_json))
        output_paths.append(("release_bundle.output_dir", config.release_bundle.output_dir))
    if config.github_governance_output_json is not None:
        output_paths.append(("github_governance_output_json", config.github_governance_output_json))
    if config.repository_hygiene_output_json is not None:
        output_paths.append(("repository_hygiene_output_json", config.repository_hygiene_output_json))
    if config.native_probe_factories_output_json is not None:
        output_paths.append(("native_probe_factories_output_json", config.native_probe_factories_output_json))
    if config.release_preflight_output_json is not None:
        output_paths.append(("release_preflight_output_json", config.release_preflight_output_json))
    if config.engine_launch_config_output_dir is not None:
        output_paths.append(("engine_launch_config_output_dir", config.engine_launch_config_output_dir))
        output_paths.extend(
            (
                f"engine_launch_config_output_dir.{backend.value}",
                _engine_launch_config_json(config.engine_launch_config_output_dir, backend),
            )
            for backend in ServingBackend
        )
    return tuple(output_paths)


def _validate_distinct_artifact_paths(output_paths: Sequence[tuple[str, str]]) -> None:
    labels_by_path: dict[str, list[str]] = {}
    for label, path in output_paths:
        labels_by_path.setdefault(_canonical_artifact_path(path), []).append(label)
    collisions = {
        path: labels
        for path, labels in labels_by_path.items()
        if len(labels) > 1
    }
    if collisions:
        raise ValueError(f"Generated artifact output paths must be distinct: {collisions}")


def _canonical_artifact_path(path: str) -> str:
    return str(local_path(path).expanduser().resolve(strict=False))


def _is_metadata_item(item: str) -> bool:
    if not item:
        return False
    key, separator, _value = item.partition("=")
    return bool(separator and key)


def _validate_engine_probe_targets(
    engine_probes: Sequence[EngineProbePlanConfig],
    *,
    release_safe: bool,
) -> None:
    if not engine_probes:
        raise ValueError("engine_probe_targets require planned engine_probes")
    backends = tuple(probe.backend.value for probe in engine_probes)
    duplicate_backends = sorted(backend for backend in set(backends) if backends.count(backend) > 1)
    if duplicate_backends:
        raise ValueError(f"engine_probe_targets must not contain duplicate backends: {duplicate_backends}")
    if not release_safe:
        return
    _validate_release_planned_engine_probes(engine_probes)
    missing_backends = sorted(set(_required_engine_probe_backends()).difference(backends))
    unexpected_backends = sorted(set(backends).difference(_required_engine_probe_backends()))
    if missing_backends or unexpected_backends:
        raise ValueError(
            "release-safe engine_probe_targets require exactly the release backends; "
            f"missing={missing_backends}, unexpected={unexpected_backends}"
        )
    _validate_release_planned_engine_probe_actions(engine_probes)


def _engine_probe_target_to_record(probe: EngineProbePlanConfig) -> dict[str, Any]:
    record: dict[str, Any] = {
        "backend": probe.backend.value,
        "handoff_json": probe.handoff_json,
        "probe_factory": probe.probe_factory,
        "output_json": probe.output_json,
        "allow_non_native_probe": probe.allow_non_native_probe,
        "metadata": list(probe.metadata),
    }
    if probe.payload_uri is not None:
        record["payload_uri"] = probe.payload_uri
    if probe.actions_output_json is not None:
        record["actions_output_json"] = probe.actions_output_json
    if probe.native_probe_delegate_factory is not None:
        record["native_probe_delegate_factory"] = probe.native_probe_delegate_factory
    if probe.engine_version is not None:
        record["engine_version"] = probe.engine_version
    if probe.fixture_output_dir is not None:
        record["fixture_output_dir"] = probe.fixture_output_dir
        record["fixture_payload_mode"] = probe.fixture_payload_mode.value
    return record


def _command_to_record(command: BenchmarkCommand) -> dict[str, Any]:
    return {"name": command.name, "argv": list(command.argv), "shell": command.shell}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit a reproducible V1 benchmark command plan.")
    parser.add_argument(
        "--raw-dataset",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="Raw dataset JSONL path. Repeat for biography, hotpotqa, musique, and niah.",
    )
    parser.add_argument("--prepared-dir", required=True, help="Directory for prepared canonical JSONL files.")
    parser.add_argument("--suite-id", default="v1-openai-compatible")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--cache-base-url")
    parser.add_argument("--model-id", default=DEFAULT_V1_MODEL_ID)
    parser.add_argument("--hardware-target", default=DEFAULT_HARDWARE_TARGET)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--allow-partial", action="store_true", help="Allow a subset of the four V1 datasets.")
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--cache-runtime-prompt", action="store_true")
    parser.add_argument("--server-usage", action="store_true")
    parser.add_argument("--baseline-extra-body-json")
    parser.add_argument("--cache-extra-body-json")
    parser.add_argument("--benchmark-output-json", help="Benchmark result path. Defaults under --prepared-dir.")
    parser.add_argument(
        "--storage-benchmark-workspace-dir",
        help="Enable the storage-reader benchmark and use this workspace directory for synthetic shards.",
    )
    parser.add_argument(
        "--storage-benchmark-output-json",
        help="Storage-reader benchmark output path. Defaults under --prepared-dir when enabled.",
    )
    parser.add_argument("--storage-benchmark-id")
    parser.add_argument("--storage-benchmark-chunk-count", type=int)
    parser.add_argument("--storage-benchmark-chunk-bytes", type=int)
    parser.add_argument("--storage-benchmark-repeats", type=int)
    parser.add_argument("--storage-benchmark-parallelism", type=int)
    parser.add_argument(
        "--storage-benchmark-reader",
        action="append",
        choices=SUPPORTED_STORAGE_BENCHMARK_READERS,
        help="Storage reader to benchmark. Repeat for multiple readers; defaults to all readers when enabled.",
    )
    parser.add_argument("--storage-benchmark-align-bytes", type=int)
    parser.add_argument("--storage-benchmark-uc-volume-root", help="Real UC Volume root, usually /Volumes/catalog/schema/volume.")
    parser.add_argument(
        "--engine-probe-handoff-json",
        action="append",
        metavar="BACKEND=PATH",
        help="Plan a native engine probe for backend vllm or sglang using this handoff JSON.",
    )
    parser.add_argument(
        "--engine-probe-fixture-output-dir",
        action="append",
        metavar="BACKEND=DIR",
        help=(
            "Generate a deterministic Qwen3 V1 engine-probe fixture in DIR before probing BACKEND. "
            "When --engine-probe-handoff-json is omitted for that backend, the handoff path is derived from DIR."
        ),
    )
    parser.add_argument(
        "--engine-probe-fixture-payload-mode",
        action="append",
        metavar="BACKEND=MODE",
        help="Fixture payload mode for a backend configured with --engine-probe-fixture-output-dir.",
    )
    parser.add_argument(
        "--engine-probe-factory",
        action="append",
        metavar="BACKEND=MODULE:CALLABLE",
        help="Native probe factory for a planned engine probe backend.",
    )
    parser.add_argument(
        "--engine-probe-use-builtin-factories",
        action="store_true",
        help=(
            "Fill missing planned engine-probe factories with package-owned "
            "vLLM/SGLang native factory paths. These factories fail closed until "
            "a backend-native block-manager adapter is available."
        ),
    )
    parser.add_argument(
        "--engine-probe-output-json",
        action="append",
        metavar="BACKEND=PATH",
        help="Output JSON path for a planned engine probe backend.",
    )
    parser.add_argument(
        "--engine-probe-actions-output-json",
        action="append",
        metavar="BACKEND=PATH",
        help="Connector actions JSON sidecar path for a planned engine probe backend.",
    )
    parser.add_argument(
        "--engine-probe-native-delegate-factory",
        action="append",
        metavar="BACKEND=MODULE:CALLABLE",
        help=(
            "Backend-native delegate factory for a planned built-in native probe. "
            "This is emitted into Databricks engine-probe target JSON."
        ),
    )
    parser.add_argument(
        "--engine-probe-payload-uri",
        action="append",
        metavar="BACKEND=URI",
        help="Optional payload URI override for a planned engine probe backend.",
    )
    parser.add_argument(
        "--engine-probe-engine-version",
        action="append",
        metavar="BACKEND=VERSION",
        help="Optional fallback engine version for a planned engine probe backend.",
    )
    parser.add_argument(
        "--engine-probe-metadata",
        action="append",
        metavar="BACKEND=KEY=VALUE",
        help="Metadata item to pass to a planned engine probe backend. Repeat as needed.",
    )
    parser.add_argument(
        "--allow-non-native-engine-probe",
        action="append",
        choices=[backend.value for backend in ServingBackend],
        help="Mark a planned backend probe as non-native debug evidence. Release evidence rejects this.",
    )
    parser.add_argument(
        "--release-evidence-output-json",
        help="Append release-evidence validation and write its JSON output here.",
    )
    parser.add_argument(
        "--release-engine-probe-json",
        action="append",
        help="Native vLLM/SGLang engine probe JSON path. Repeat for each backend.",
    )
    parser.add_argument(
        "--release-engine-actions-json",
        action="append",
        help="Native vLLM/SGLang connector actions JSON path. Repeat for each backend.",
    )
    parser.add_argument(
        "--release-storage-benchmark-json",
        help="Existing storage benchmark JSON for release evidence validation. Defaults to the planned storage benchmark output.",
    )
    parser.add_argument(
        "--release-preflight-output-json",
        help=(
            "Append release-evidence input preflight and write "
            "document_kv.release_evidence_inputs.v1 here."
        ),
    )
    parser.add_argument(
        "--release-bundle-output-dir",
        help="Append release-bundle assembly and copy validated artifacts into this directory.",
    )
    parser.add_argument(
        "--release-bundle-output-json",
        help="Release-bundle manifest JSON sidecar. Defaults under --prepared-dir when enabled.",
    )
    parser.add_argument("--release-bundle-preflight-json", help="Optional release-input preflight sidecar to bundle.")
    parser.add_argument(
        "--release-bundle-plan-execution-json",
        action="append",
        help="Benchmark-plan execution sidecar to include in the release bundle. Repeat as needed.",
    )
    parser.add_argument(
        "--release-bundle-databricks-run-status-json",
        action="append",
        help="Compact successful Databricks run-status sidecar to include in the release bundle. Repeat as needed.",
    )
    parser.add_argument("--release-bundle-package-wheel", help="Tested document-kv-cache wheel to include.")
    parser.add_argument(
        "--release-bundle-pr-evidence-json",
        action="append",
        help="PR evidence sidecar to include in the release bundle. Repeat as needed.",
    )
    parser.add_argument("--release-bundle-requirements-matrix-md", help="V1 requirements matrix Markdown to include.")
    parser.add_argument("--release-bundle-github-governance-json", help="GitHub governance sidecar to include.")
    parser.add_argument("--release-bundle-repository-hygiene-json", help="Repository hygiene sidecar to include.")
    parser.add_argument(
        "--release-bundle-native-probe-factories-json",
        action="append",
        help="Native probe factory diagnostics sidecar to include in the release bundle. Repeat as needed.",
    )
    parser.add_argument(
        "--release-bundle-engine-launch-config-json",
        action="append",
        help="vLLM/SGLang engine launch config sidecar to include in the release bundle. Repeat as needed.",
    )
    parser.add_argument(
        "--release-bundle-require-complete-v1",
        action="store_true",
        help="Emit --require-complete-v1 on the release bundle command for strict V1 publishing.",
    )
    parser.add_argument("--release-bundle-overwrite", action="store_true")
    parser.add_argument("--plan-output-json", help="Write the command plan JSON to this path.")
    parser.add_argument("--plan-output-sh", help="Write an executable shell script to this path.")
    parser.add_argument(
        "--engine-probe-targets-output-json",
        help="Write Databricks engine-probe matrix backend config JSON from planned engine probes.",
    )
    parser.add_argument(
        "--engine-probe-targets-release-safe",
        action="store_true",
        help="Require planned engine probes to be native release probes for exactly vLLM and SGLang before writing targets.",
    )
    parser.add_argument(
        "--github-governance-output-json",
        help=(
            "Append GitHub governance inspection and write "
            "document_kv.github_repository_governance.v1 here."
        ),
    )
    parser.add_argument(
        "--repository-hygiene-output-json",
        help="Append repository hygiene inspection and write document_kv.repository_hygiene.v1 here.",
    )
    parser.add_argument(
        "--native-probe-factories-output-json",
        help="Append native-probe factory diagnostics and write document_kv.native_probe_factories.v1 here.",
    )
    parser.add_argument(
        "--engine-launch-config-output-dir",
        help=(
            "Append generated vLLM/SGLang engine launch-config sidecars under this directory "
            "and include them in release bundles."
        ),
    )
    args = parser.parse_args(argv)

    try:
        prepared_dir = Path(args.prepared_dir)
        engine_probes = _engine_probe_configs_from_cli(args)
        config = BenchmarkPlanConfig(
            suite_id=args.suite_id,
            dataset_paths=_dataset_paths_from_cli(args.raw_dataset, prepared_dir=prepared_dir),
            base_url=args.base_url,
            cache_base_url=args.cache_base_url,
            model_id=args.model_id,
            hardware_target=args.hardware_target,
            python_executable=args.python_executable,
            require_all_v1_datasets=not args.allow_partial,
            limit_per_dataset=args.limit_per_dataset,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            stream=not args.no_stream,
            cache_runtime_prompt=args.cache_runtime_prompt,
            server_usage=args.server_usage,
            baseline_extra_body_json=args.baseline_extra_body_json,
            cache_extra_body_json=args.cache_extra_body_json,
            benchmark_output_json=args.benchmark_output_json
            or str(prepared_dir / f"{args.suite_id}-results.json"),
            storage_benchmark=_storage_benchmark_config_from_cli(args, prepared_dir=prepared_dir),
            engine_probes=engine_probes,
            release_evidence=_release_evidence_config_from_cli(
                args,
                has_planned_engine_probes=bool(engine_probes),
            ),
            release_bundle=_release_bundle_config_from_cli(args, prepared_dir=prepared_dir),
            github_governance_output_json=args.github_governance_output_json,
            repository_hygiene_output_json=args.repository_hygiene_output_json,
            native_probe_factories_output_json=args.native_probe_factories_output_json,
            release_preflight_output_json=args.release_preflight_output_json,
            engine_launch_config_output_dir=args.engine_launch_config_output_dir,
        )
        _validate_plan_output_paths(
            config,
            plan_output_json=args.plan_output_json,
            plan_output_sh=args.plan_output_sh,
            engine_probe_targets_output_json=args.engine_probe_targets_output_json,
        )
        plan = build_v1_benchmark_plan(config)
        if args.plan_output_json:
            write_benchmark_job_plan_json(plan, args.plan_output_json)
        if args.plan_output_sh:
            write_benchmark_job_plan_shell(plan, args.plan_output_sh)
        if args.engine_probe_targets_output_json:
            write_engine_probe_targets_json(
                plan,
                args.engine_probe_targets_output_json,
                release_safe=args.engine_probe_targets_release_safe,
            )
        if not args.plan_output_json and not args.plan_output_sh and not args.engine_probe_targets_output_json:
            print(json.dumps(benchmark_job_plan_to_record(plan), indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


def _storage_benchmark_config_from_cli(
    args: argparse.Namespace,
    *,
    prepared_dir: Path,
) -> StorageBenchmarkPlanConfig | None:
    if args.storage_benchmark_workspace_dir is None:
        if _has_storage_benchmark_options(args):
            raise ValueError("storage benchmark options require --storage-benchmark-workspace-dir")
        return None
    return StorageBenchmarkPlanConfig(
        workspace_dir=args.storage_benchmark_workspace_dir,
        output_json=args.storage_benchmark_output_json
        or str(prepared_dir / f"{args.suite_id}-storage-benchmark.json"),
        benchmark_id=(
            args.storage_benchmark_id
            if args.storage_benchmark_id is not None
            else DEFAULT_STORAGE_BENCHMARK_ID
        ),
        chunk_count=(
            args.storage_benchmark_chunk_count
            if args.storage_benchmark_chunk_count is not None
            else DEFAULT_STORAGE_BENCHMARK_CHUNK_COUNT
        ),
        chunk_bytes=(
            args.storage_benchmark_chunk_bytes
            if args.storage_benchmark_chunk_bytes is not None
            else DEFAULT_STORAGE_BENCHMARK_CHUNK_BYTES
        ),
        repeats=(
            args.storage_benchmark_repeats
            if args.storage_benchmark_repeats is not None
            else DEFAULT_STORAGE_BENCHMARK_REPEATS
        ),
        parallelism=(
            args.storage_benchmark_parallelism
            if args.storage_benchmark_parallelism is not None
            else DEFAULT_STORAGE_BENCHMARK_PARALLELISM
        ),
        readers=(
            tuple(args.storage_benchmark_reader)
            if args.storage_benchmark_reader
            else _default_storage_benchmark_readers(args.storage_benchmark_uc_volume_root)
        ),
        align_bytes=(
            args.storage_benchmark_align_bytes
            if args.storage_benchmark_align_bytes is not None
            else DEFAULT_STORAGE_BENCHMARK_ALIGN_BYTES
        ),
        uc_volume_root=args.storage_benchmark_uc_volume_root,
    )


def _has_storage_benchmark_options(args: argparse.Namespace) -> bool:
    return any(
        option is not None
        for option in (
            args.storage_benchmark_output_json,
            args.storage_benchmark_id,
            args.storage_benchmark_chunk_count,
            args.storage_benchmark_chunk_bytes,
            args.storage_benchmark_repeats,
            args.storage_benchmark_parallelism,
            args.storage_benchmark_reader,
            args.storage_benchmark_align_bytes,
            args.storage_benchmark_uc_volume_root,
        )
    )


def _engine_probe_configs_from_cli(args: argparse.Namespace) -> tuple[EngineProbePlanConfig, ...]:
    handoff_jsons = _named_value_map(args.engine_probe_handoff_json or (), "--engine-probe-handoff-json")
    fixture_output_dirs = _named_value_map(
        args.engine_probe_fixture_output_dir or (),
        "--engine-probe-fixture-output-dir",
    )
    fixture_payload_modes = _named_value_map(
        args.engine_probe_fixture_payload_mode or (),
        "--engine-probe-fixture-payload-mode",
    )
    factories = _named_value_map(args.engine_probe_factory or (), "--engine-probe-factory")
    output_jsons = _named_value_map(args.engine_probe_output_json or (), "--engine-probe-output-json")
    actions_output_jsons = _named_value_map(
        args.engine_probe_actions_output_json or (),
        "--engine-probe-actions-output-json",
    )
    native_probe_delegate_factories = _named_value_map(
        args.engine_probe_native_delegate_factory or (),
        "--engine-probe-native-delegate-factory",
    )
    payload_uris = _named_value_map(args.engine_probe_payload_uri or (), "--engine-probe-payload-uri")
    engine_versions = _named_value_map(args.engine_probe_engine_version or (), "--engine-probe-engine-version")
    metadata = _named_value_lists(args.engine_probe_metadata or (), "--engine-probe-metadata")
    non_native_backends = set(args.allow_non_native_engine_probe or ())
    _require_subset_backend_keys(
        set(fixture_output_dirs),
        fixture_payload_modes,
        "--engine-probe-fixture-payload-mode",
    )
    handoff_jsons = _engine_probe_handoff_jsons_with_fixtures(handoff_jsons, fixture_output_dirs)

    if not handoff_jsons:
        if (
            factories
            or output_jsons
            or actions_output_jsons
            or native_probe_delegate_factories
            or payload_uris
            or engine_versions
            or metadata
            or non_native_backends
            or args.engine_probe_use_builtin_factories
        ):
            raise ValueError("engine probe options require --engine-probe-handoff-json or --engine-probe-fixture-output-dir")
        return ()

    backends = set(handoff_jsons)
    if args.engine_probe_use_builtin_factories:
        factories = _fill_missing_builtin_engine_probe_factories(backends, factories)
    _require_matching_backend_keys(backends, factories, "--engine-probe-factory")
    _require_matching_backend_keys(backends, output_jsons, "--engine-probe-output-json")
    _require_subset_backend_keys(backends, actions_output_jsons, "--engine-probe-actions-output-json")
    _require_subset_backend_keys(backends, native_probe_delegate_factories, "--engine-probe-native-delegate-factory")
    _require_subset_backend_keys(backends, payload_uris, "--engine-probe-payload-uri")
    _require_subset_backend_keys(backends, engine_versions, "--engine-probe-engine-version")
    _require_subset_backend_keys(backends, metadata, "--engine-probe-metadata")
    _require_subset_backend_keys(backends, fixture_output_dirs, "--engine-probe-fixture-output-dir")
    unsupported_non_native = sorted(non_native_backends.difference(backends))
    if unsupported_non_native:
        raise ValueError(f"--allow-non-native-engine-probe has no planned backend: {unsupported_non_native}")

    return tuple(
        EngineProbePlanConfig(
            backend=backend,
            handoff_json=handoff_jsons[backend],
            probe_factory=factories[backend],
            output_json=output_jsons[backend],
            actions_output_json=actions_output_jsons.get(backend),
            native_probe_delegate_factory=native_probe_delegate_factories.get(backend),
            payload_uri=payload_uris.get(backend),
            engine_version=engine_versions.get(backend),
            allow_non_native_probe=backend in non_native_backends,
            metadata=tuple(metadata.get(backend, ())),
            fixture_output_dir=fixture_output_dirs.get(backend),
            fixture_payload_mode=fixture_payload_modes.get(backend, PayloadMode.SEGMENTED),
        )
        for backend in sorted(backends)
    )


def _engine_probe_handoff_jsons_with_fixtures(
    handoff_jsons: Mapping[str, str],
    fixture_output_dirs: Mapping[str, str],
) -> dict[str, str]:
    merged = dict(handoff_jsons)
    for backend, output_dir in fixture_output_dirs.items():
        fixture_handoff_json = _engine_probe_fixture_handoff_json(output_dir)
        explicit_handoff_json = merged.get(backend)
        if explicit_handoff_json is not None and not _same_artifact_path(explicit_handoff_json, fixture_handoff_json):
            raise ValueError(
                "--engine-probe-handoff-json must match the derived fixture handoff path when "
                f"--engine-probe-fixture-output-dir is set for {backend!r}: "
                f"expected {fixture_handoff_json!r}, got {explicit_handoff_json!r}"
            )
        merged[backend] = fixture_handoff_json
    return merged


def _engine_probe_fixture_handoff_json(fixture_output_dir: str) -> str:
    return _uri_child(fixture_output_dir, DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES["handoff"])


def _engine_probe_fixture_payload_uri(fixture_output_dir: str) -> str:
    return _uri_child(fixture_output_dir, DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES["payload"])


def _engine_probe_fixture_actions_json(fixture_output_dir: str) -> str:
    return _uri_child(fixture_output_dir, DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES["actions"])


def _engine_probe_uses_fixture_actions_output(probe: EngineProbePlanConfig) -> bool:
    return (
        probe.fixture_output_dir is not None
        and probe.actions_output_json is not None
        and _same_artifact_path(probe.actions_output_json, _engine_probe_fixture_actions_json(probe.fixture_output_dir))
    )


def _uri_child(base_uri: str, filename: str) -> str:
    return f"{base_uri.rstrip('/')}/{filename}"


def _same_artifact_path(left: str, right: str) -> bool:
    return _canonical_artifact_path(left) == _canonical_artifact_path(right)


def _fill_missing_builtin_engine_probe_factories(
    backends: set[str],
    factories: Mapping[str, str],
) -> dict[str, str]:
    filled = dict(factories)
    for backend in backends:
        filled.setdefault(backend, builtin_native_probe_factory_path(backend))
    return filled


def _release_evidence_config_from_cli(
    args: argparse.Namespace,
    *,
    has_planned_engine_probes: bool,
) -> ReleaseEvidencePlanConfig | None:
    if args.release_evidence_output_json is None:
        if _has_release_evidence_options(args):
            raise ValueError("release evidence options require --release-evidence-output-json")
        return None
    if not args.release_engine_probe_json and not has_planned_engine_probes:
        raise ValueError("release evidence requires --release-engine-probe-json or planned engine probes")
    if not args.release_engine_actions_json and not has_planned_engine_probes:
        raise ValueError("release evidence requires --release-engine-actions-json or planned engine probes")
    return ReleaseEvidencePlanConfig(
        output_json=args.release_evidence_output_json,
        engine_probe_jsons=tuple(args.release_engine_probe_json or ()),
        engine_actions_jsons=tuple(args.release_engine_actions_json or ()),
        storage_benchmark_json=args.release_storage_benchmark_json,
    )


def _has_release_evidence_options(args: argparse.Namespace) -> bool:
    return (
        args.release_engine_probe_json is not None
        or args.release_engine_actions_json is not None
        or args.release_storage_benchmark_json is not None
        or args.release_preflight_output_json is not None
    )


def _release_bundle_config_from_cli(
    args: argparse.Namespace,
    *,
    prepared_dir: Path,
) -> ReleaseBundlePlanConfig | None:
    if args.release_bundle_output_dir is None:
        if _has_release_bundle_options(args):
            raise ValueError("release bundle options require --release-bundle-output-dir")
        return None
    return ReleaseBundlePlanConfig(
        output_dir=args.release_bundle_output_dir,
        output_json=args.release_bundle_output_json
        or str(prepared_dir / f"{args.suite_id}-release-bundle-manifest.json"),
        preflight_json=args.release_bundle_preflight_json,
        plan_execution_jsons=tuple(args.release_bundle_plan_execution_json or ()),
        databricks_run_status_jsons=tuple(args.release_bundle_databricks_run_status_json or ()),
        package_wheel=args.release_bundle_package_wheel,
        pr_evidence_jsons=tuple(args.release_bundle_pr_evidence_json or ()),
        requirements_matrix_md=args.release_bundle_requirements_matrix_md,
        github_governance_json=args.release_bundle_github_governance_json,
        repository_hygiene_json=args.release_bundle_repository_hygiene_json,
        native_probe_factories_jsons=tuple(args.release_bundle_native_probe_factories_json or ()),
        engine_launch_config_jsons=tuple(args.release_bundle_engine_launch_config_json or ()),
        overwrite=args.release_bundle_overwrite,
        require_complete_v1=args.release_bundle_require_complete_v1,
    )


def _has_release_bundle_options(args: argparse.Namespace) -> bool:
    return (
        args.release_bundle_output_json is not None
        or args.release_bundle_preflight_json is not None
        or args.release_bundle_plan_execution_json is not None
        or args.release_bundle_databricks_run_status_json is not None
        or args.release_bundle_package_wheel is not None
        or args.release_bundle_pr_evidence_json is not None
        or args.release_bundle_requirements_matrix_md is not None
        or args.release_bundle_github_governance_json is not None
        or args.release_bundle_repository_hygiene_json is not None
        or args.release_bundle_native_probe_factories_json is not None
        or args.release_bundle_engine_launch_config_json is not None
        or args.release_bundle_require_complete_v1
        or args.release_bundle_overwrite
    )


def _named_value_map(values: Sequence[str], option_name: str) -> dict[str, str]:
    result = {}
    for value in values:
        key, item = _split_named_value(value, option_name=option_name)
        if key in result:
            raise ValueError(f"Duplicate backend for {option_name}: {key!r}")
        result[key] = item
    return result


def _named_value_lists(values: Sequence[str], option_name: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        key, item = _split_named_value(value, option_name=option_name)
        result.setdefault(key, []).append(item)
    return result


def _split_named_value(value: str, *, option_name: str) -> tuple[str, str]:
    key, separator, item = value.partition("=")
    if not separator or not key or not item:
        raise ValueError(f"{option_name} must use BACKEND=VALUE syntax")
    try:
        backend = ServingBackend(key).value
    except ValueError as exc:
        raise ValueError(f"{option_name} backend must be one of {[backend.value for backend in ServingBackend]}") from exc
    return backend, item


def _require_matching_backend_keys(expected: set[str], actual: Mapping[str, object], option_name: str) -> None:
    missing = sorted(expected.difference(actual))
    extra = sorted(set(actual).difference(expected))
    if missing or extra:
        raise ValueError(f"{option_name} must match planned engine probe backends; missing={missing}, extra={extra}")


def _require_subset_backend_keys(expected: set[str], actual: Mapping[str, object], option_name: str) -> None:
    extra = sorted(set(actual).difference(expected))
    if extra:
        raise ValueError(f"{option_name} has no planned engine probe backend: {extra}")


def _default_storage_benchmark_readers(uc_volume_root: str | None) -> tuple[str, ...]:
    if uc_volume_root is None:
        return DEFAULT_STORAGE_BENCHMARK_PLAN_READERS
    return SUPPORTED_STORAGE_BENCHMARK_READERS


def _dataset_paths_from_cli(values: Sequence[str], *, prepared_dir: Path) -> tuple[BenchmarkDatasetPath, ...]:
    dataset_paths: dict[str, BenchmarkDatasetPath] = {}
    for value in values:
        dataset, raw_jsonl = _split_dataset_path(value, option_name="--raw-dataset")
        if dataset in dataset_paths:
            raise ValueError(f"Duplicate raw dataset path for {dataset!r}")
        dataset_paths[dataset] = BenchmarkDatasetPath(
            dataset=dataset,
            raw_jsonl=raw_jsonl,
            prepared_jsonl=str(prepared_dir / f"{dataset}.jsonl"),
        )
    return tuple(dataset_paths[dataset] for dataset in SUPPORTED_V1_DATASETS if dataset in dataset_paths)


def _split_dataset_path(value: str, *, option_name: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option_name} must use DATASET=PATH")
    dataset, path = value.split("=", 1)
    validate_v1_dataset(dataset)
    if not path:
        raise ValueError(f"{option_name} {dataset}=PATH must include a path")
    return dataset, path


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
