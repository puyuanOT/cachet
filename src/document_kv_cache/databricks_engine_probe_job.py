"""Databricks runs/submit payload helpers for engine KV-connector probes."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_kv_cache._hardware_targets import (
    SUPPORTED_V1_HARDWARE_TARGETS,
    databricks_node_type_for_hardware_target,
)
from document_kv_cache.databricks_job import (
    DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE,
    DEFAULT_DATABRICKS_DATA_SECURITY_MODE,
    DEFAULT_DATABRICKS_SPARK_VERSION,
    DatabricksSingleNodeGPUClusterConfig,
    build_single_node_gpu_cluster,
)
from document_kv_cache.engine_adapters import PayloadMode, ServingBackend
from document_kv_cache.native_probe_factories import (
    SGLANG_NATIVE_PROBE_DELEGATE_ENV,
    SGLANG_NATIVE_PROBE_FACTORY,
    VLLM_NATIVE_PROBE_DELEGATE_ENV,
    VLLM_NATIVE_PROBE_FACTORY,
)
from document_kv_cache._native_probe_metadata import (
    SGLANG_NATIVE_PROBE_DELEGATE_FACTORY,
    SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY,
    SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
    VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
    VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY,
    VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
    validate_known_native_delegate_metadata,
)
from document_kv_cache.probe_fixtures import DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES
from document_kv_cache.release_evidence import REQUIRED_ENGINE_PROBE_BACKENDS
from document_kv_cache.serving_env import SGLANG_VERSION, VLLM_VERSION
from document_kv_cache.storage import local_path


DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME = "document-kv-engine-probe"
DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY = "document_kv_engine_probe"
DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE = "document-kv-engine-probe"
DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY = "probes"
ENGINE_PROBE_TARGETS_RECORD_TYPE = "document_kv.engine_probe_targets.v1"
ENGINE_PROBE_TARGETS_SCHEMA_VERSION = 1
DEFAULT_VLLM_ENGINE_PROBE_RUNTIME_PACKAGE = f"vllm=={VLLM_VERSION}"
DEFAULT_SGLANG_ENGINE_PROBE_RUNTIME_PACKAGE = f"sglang=={SGLANG_VERSION}"
_PROVIDER_BACKED_NATIVE_PROBE_OPTION_NAMES = {
    ServingBackend.VLLM: "--provider-backed-vllm-native-probe",
    ServingBackend.SGLANG: "--provider-backed-sglang-native-probe",
}
_PROVIDER_BACKED_NATIVE_PROBE_FACTORIES = {
    ServingBackend.VLLM: VLLM_NATIVE_PROBE_FACTORY,
    ServingBackend.SGLANG: SGLANG_NATIVE_PROBE_FACTORY,
}
_PROVIDER_BACKED_NATIVE_PROBE_DELEGATE_FACTORIES = {
    ServingBackend.VLLM: VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
    ServingBackend.SGLANG: SGLANG_NATIVE_PROBE_DELEGATE_FACTORY,
}
_PROVIDER_BACKED_NATIVE_PROBE_METADATA = {
    ServingBackend.VLLM: VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
    ServingBackend.SGLANG: SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
}
_PROVIDER_BACKED_NATIVE_PROBE_RUNTIME_PACKAGES = {
    ServingBackend.VLLM: DEFAULT_VLLM_ENGINE_PROBE_RUNTIME_PACKAGE,
    ServingBackend.SGLANG: DEFAULT_SGLANG_ENGINE_PROBE_RUNTIME_PACKAGE,
}
_ENGINE_PROBE_TARGETS_ENVELOPE_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "release_safe",
        DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY,
    }
)
_ENGINE_PROBE_TARGET_KEYS = frozenset(
    {
        "backend",
        "expected_backend",
        "handoff_json",
        "probe_factory",
        "output_json",
        "probe_output_json",
        "payload_uri",
        "task_key",
        "engine_version",
        "allow_non_native_probe",
        "metadata",
        "native_probe_delegate_factory",
        "actions_output_json",
        "connector_actions_output_json",
        "fixture_output_dir",
        "fixture_payload_mode",
        "pip_packages",
        "vllm_runtime_preflight_output_json",
        "vllm_runtime_preflight_layer_names_json",
        "sglang_runtime_preflight_output_json",
        "sglang_runtime_preflight_launch_config_json",
    }
)
ENGINE_PROBE_RUNNER_SCRIPT = """from __future__ import annotations

import argparse
import subprocess
import sys


def _cluster_file_path(uri: str) -> str:
    if uri.startswith("dbfs:/"):
        return "/dbfs/" + uri.removeprefix("dbfs:/").lstrip("/")
    return uri


def _install_runtime_packages(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pip-package", action="append")
    parser.add_argument("--package-wheel-uri", action="append")
    args, remaining = parser.parse_known_args(argv)
    for pip_package in args.pip_package or ():
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_package])
    for package_wheel_uri in args.package_wheel_uri or ():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", _cluster_file_path(package_wheel_uri)]
        )
    return remaining

if __name__ == "__main__":
    remaining_args = _install_runtime_packages(sys.argv[1:])
    from document_kv_cache.databricks_engine_probe_job import run_engine_probe_task

    exit_code = run_engine_probe_task(remaining_args)
    if exit_code:
        raise SystemExit(exit_code)
"""

__all__ = [
    "DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME",
    "DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY",
    "DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE",
    "DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY",
    "ENGINE_PROBE_RUNNER_SCRIPT",
    "SGLANG_NATIVE_PROBE_DELEGATE_FACTORY",
    "SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY",
    "SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA",
    "VLLM_NATIVE_PROBE_DELEGATE_FACTORY",
    "VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY",
    "VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA",
    "DEFAULT_SGLANG_ENGINE_PROBE_RUNTIME_PACKAGE",
    "DEFAULT_VLLM_ENGINE_PROBE_RUNTIME_PACKAGE",
    "DatabricksEngineProbeJobConfig",
    "DatabricksEngineProbeMatrixJobConfig",
    "DatabricksEngineProbeTargetConfig",
    "DatabricksEngineProbeTargetsFile",
    "build_databricks_engine_probe_run_submit_payload",
    "build_databricks_engine_probe_matrix_run_submit_payload",
    "read_databricks_engine_probe_targets_json",
    "read_databricks_engine_probe_targets_file_json",
    "run_engine_probe_task",
    "write_databricks_engine_probe_run_submit_json",
    "write_databricks_engine_probe_matrix_run_submit_json",
    "write_databricks_engine_probe_runner_script",
    "main",
]


@dataclass(frozen=True, slots=True)
class DatabricksEngineProbeTargetsFile:
    """Parsed backend-target JSON plus any release-safety envelope metadata."""

    probe_targets: tuple["DatabricksEngineProbeTargetConfig", ...]
    release_safe: bool = False

    def __post_init__(self) -> None:
        if type(self.release_safe) is not bool:
            raise ValueError("release_safe must be a boolean")
        if not self.probe_targets:
            raise ValueError("probe_targets must be non-empty")
        object.__setattr__(self, "probe_targets", tuple(self.probe_targets))


@dataclass(frozen=True, slots=True)
class DatabricksEngineProbeTargetConfig:
    """One backend-specific native block-manager probe target."""

    expected_backend: ServingBackend | str
    handoff_json: str
    probe_factory: str
    output_json: str
    payload_uri: str | None = None
    task_key: str | None = None
    engine_version: str | None = None
    allow_non_native_probe: bool = False
    metadata: tuple[str, ...] = ()
    actions_output_json: str | None = None
    native_probe_delegate_factory: str | None = None
    fixture_output_dir: str | None = None
    fixture_payload_mode: PayloadMode | str = PayloadMode.SEGMENTED
    pip_packages: tuple[str, ...] = ()
    vllm_runtime_preflight_output_json: str | None = None
    vllm_runtime_preflight_layer_names_json: str | None = None
    sglang_runtime_preflight_output_json: str | None = None
    sglang_runtime_preflight_launch_config_json: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_backend", _DEFAULT_SERVING_BACKEND(self.expected_backend))
        object.__setattr__(self, "fixture_payload_mode", _DEFAULT_PAYLOAD_MODE(self.fixture_payload_mode))
        if not self.handoff_json:
            raise ValueError("handoff_json must be non-empty")
        if not self.probe_factory:
            raise ValueError("probe_factory must be non-empty")
        if not self.output_json:
            raise ValueError("output_json must be non-empty")
        if self.payload_uri is not None and not self.payload_uri:
            raise ValueError("payload_uri must be non-empty when provided")
        if self.task_key is not None and not self.task_key:
            raise ValueError("task_key must be non-empty when provided")
        if self.engine_version is not None and not self.engine_version:
            raise ValueError("engine_version must be non-empty when provided")
        if self.actions_output_json is not None and not self.actions_output_json:
            raise ValueError("actions_output_json must be non-empty when provided")
        if self.native_probe_delegate_factory is not None and not self.native_probe_delegate_factory:
            raise ValueError("native_probe_delegate_factory must be non-empty when provided")
        if self.fixture_output_dir is not None and not self.fixture_output_dir:
            raise ValueError("fixture_output_dir must be non-empty when provided")
        _DEFAULT_DERIVE_VLLM_FIXTURE_RUNTIME_PREFLIGHT_LAYER_NAMES(self, label="engine probe target")
        _DEFAULT_VALIDATE_VLLM_RUNTIME_PREFLIGHT_CONFIG(self, label="engine probe target")
        _DEFAULT_VALIDATE_SGLANG_RUNTIME_PREFLIGHT_CONFIG(self, label="engine probe target")
        if self.fixture_output_dir is not None:
            _DEFAULT_VALIDATE_FIXTURE_OUTPUT_DIR(self.fixture_output_dir)
            _DEFAULT_VALIDATE_FIXTURE_HANDOFF_JSON(
                handoff_json=self.handoff_json,
                fixture_output_dir=self.fixture_output_dir,
            )
            _DEFAULT_VALIDATE_FIXTURE_PAYLOAD_URI(
                payload_uri=self.payload_uri,
                fixture_output_dir=self.fixture_output_dir,
            )
        if type(self.allow_non_native_probe) is not bool:
            raise ValueError("allow_non_native_probe must be a boolean")
        _DEFAULT_VALIDATE_METADATA_ITEMS(self.metadata)
        object.__setattr__(self, "metadata", tuple(self.metadata))
        _DEFAULT_VALIDATE_KNOWN_NATIVE_DELEGATE_METADATA(
            expected_backend=self.expected_backend,
            native_probe_delegate_factory=self.native_probe_delegate_factory,
            metadata=self.metadata,
            label="engine probe target",
        )
        _DEFAULT_VALIDATE_PIP_PACKAGES(self.pip_packages, field_name="pip_packages")
        object.__setattr__(self, "pip_packages", tuple(self.pip_packages))


@dataclass(frozen=True, slots=True)
class DatabricksEngineProbeMatrixJobConfig:
    """A release-oriented Databricks job that runs native probes for required backends."""

    probe_targets: Sequence[DatabricksEngineProbeTargetConfig]
    runner_python_file: str
    run_name: str = DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME
    node_type_id: str = DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE
    spark_version: str = DEFAULT_DATABRICKS_SPARK_VERSION
    data_security_mode: str = DEFAULT_DATABRICKS_DATA_SECURITY_MODE
    single_user_name: str | None = None
    wheel_uri: str | None = None
    release_safe: bool = False
    availability: str = "ON_DEMAND"
    zone_id: str = "auto"
    custom_tags: Mapping[str, str] = field(default_factory=dict)
    extra_wheel_uris: tuple[str, ...] = ()
    serial_tasks: bool = False
    extra_pip_packages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.probe_targets:
            raise ValueError("probe_targets must be non-empty")
        if not self.runner_python_file:
            raise ValueError("runner_python_file must be non-empty")
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if self.wheel_uri is not None and not self.wheel_uri:
            raise ValueError("wheel_uri must be non-empty when provided")
        _DEFAULT_VALIDATE_WHEEL_URIS(self.extra_wheel_uris, field_name="extra_wheel_uris")
        object.__setattr__(self, "extra_wheel_uris", tuple(self.extra_wheel_uris))
        _DEFAULT_VALIDATE_PIP_PACKAGES(self.extra_pip_packages, field_name="extra_pip_packages")
        object.__setattr__(self, "extra_pip_packages", tuple(self.extra_pip_packages))
        if type(self.release_safe) is not bool:
            raise ValueError("release_safe must be a boolean")
        if type(self.serial_tasks) is not bool:
            raise ValueError("serial_tasks must be a boolean")
        targets = tuple(_DEFAULT_COERCE_PROBE_TARGET(target) for target in self.probe_targets)
        _DEFAULT_VALIDATE_PROBE_TARGET_BACKENDS(targets, release_safe=self.release_safe)
        _DEFAULT_VALIDATE_PROBE_TARGET_TASK_KEYS(targets)
        _DEFAULT_VALIDATE_RELEASE_SAFE_PROBE_TARGETS(targets, release_safe=self.release_safe)
        object.__setattr__(self, "probe_targets", targets)
        _DEFAULT_CLUSTER_CONFIG_FROM_ENGINE_PROBE_MATRIX_JOB(self)


@dataclass(frozen=True, slots=True)
class DatabricksEngineProbeJobConfig:
    handoff_json: str
    probe_factory: str
    output_json: str
    runner_python_file: str
    expected_backend: ServingBackend | str
    payload_uri: str | None = None
    run_name: str = DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME
    task_key: str = DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY
    node_type_id: str = DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE
    spark_version: str = DEFAULT_DATABRICKS_SPARK_VERSION
    data_security_mode: str = DEFAULT_DATABRICKS_DATA_SECURITY_MODE
    single_user_name: str | None = None
    wheel_uri: str | None = None
    engine_version: str | None = None
    allow_non_native_probe: bool = False
    metadata: tuple[str, ...] = ()
    release_safe: bool = False
    availability: str = "ON_DEMAND"
    zone_id: str = "auto"
    custom_tags: Mapping[str, str] = field(default_factory=dict)
    actions_output_json: str | None = None
    native_probe_delegate_factory: str | None = None
    fixture_output_dir: str | None = None
    fixture_payload_mode: PayloadMode | str = PayloadMode.SEGMENTED
    extra_wheel_uris: tuple[str, ...] = ()
    extra_pip_packages: tuple[str, ...] = ()
    vllm_runtime_preflight_output_json: str | None = None
    vllm_runtime_preflight_layer_names_json: str | None = None
    sglang_runtime_preflight_output_json: str | None = None
    sglang_runtime_preflight_launch_config_json: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_backend", _DEFAULT_SERVING_BACKEND(self.expected_backend))
        object.__setattr__(self, "fixture_payload_mode", _DEFAULT_PAYLOAD_MODE(self.fixture_payload_mode))
        if not self.handoff_json:
            raise ValueError("handoff_json must be non-empty")
        if not self.probe_factory:
            raise ValueError("probe_factory must be non-empty")
        if not self.output_json:
            raise ValueError("output_json must be non-empty")
        if not self.runner_python_file:
            raise ValueError("runner_python_file must be non-empty")
        if self.payload_uri is not None and not self.payload_uri:
            raise ValueError("payload_uri must be non-empty when provided")
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if not self.task_key:
            raise ValueError("task_key must be non-empty")
        if self.wheel_uri is not None and not self.wheel_uri:
            raise ValueError("wheel_uri must be non-empty when provided")
        _DEFAULT_VALIDATE_WHEEL_URIS(self.extra_wheel_uris, field_name="extra_wheel_uris")
        object.__setattr__(self, "extra_wheel_uris", tuple(self.extra_wheel_uris))
        _DEFAULT_VALIDATE_PIP_PACKAGES(self.extra_pip_packages, field_name="extra_pip_packages")
        object.__setattr__(self, "extra_pip_packages", tuple(self.extra_pip_packages))
        if self.engine_version is not None and not self.engine_version:
            raise ValueError("engine_version must be non-empty when provided")
        if self.actions_output_json is not None and not self.actions_output_json:
            raise ValueError("actions_output_json must be non-empty when provided")
        if self.native_probe_delegate_factory is not None and not self.native_probe_delegate_factory:
            raise ValueError("native_probe_delegate_factory must be non-empty when provided")
        if self.fixture_output_dir is not None and not self.fixture_output_dir:
            raise ValueError("fixture_output_dir must be non-empty when provided")
        _DEFAULT_DERIVE_VLLM_FIXTURE_RUNTIME_PREFLIGHT_LAYER_NAMES(self, label="engine probe job")
        _DEFAULT_VALIDATE_VLLM_RUNTIME_PREFLIGHT_CONFIG(self, label="engine probe job")
        _DEFAULT_VALIDATE_SGLANG_RUNTIME_PREFLIGHT_CONFIG(self, label="engine probe job")
        if self.fixture_output_dir is not None:
            _DEFAULT_VALIDATE_FIXTURE_OUTPUT_DIR(self.fixture_output_dir)
            _DEFAULT_VALIDATE_FIXTURE_HANDOFF_JSON(
                handoff_json=self.handoff_json,
                fixture_output_dir=self.fixture_output_dir,
            )
            _DEFAULT_VALIDATE_FIXTURE_PAYLOAD_URI(
                payload_uri=self.payload_uri,
                fixture_output_dir=self.fixture_output_dir,
            )
        _DEFAULT_VALIDATE_METADATA_ITEMS(self.metadata)
        object.__setattr__(self, "metadata", tuple(self.metadata))
        _DEFAULT_VALIDATE_KNOWN_NATIVE_DELEGATE_METADATA(
            expected_backend=self.expected_backend,
            native_probe_delegate_factory=self.native_probe_delegate_factory,
            metadata=self.metadata,
            label="engine probe job",
        )
        _DEFAULT_VALIDATE_RELEASE_SAFE_PROBE_JOB(self)
        _DEFAULT_CLUSTER_CONFIG_FROM_ENGINE_PROBE_JOB(self)


def build_databricks_engine_probe_run_submit_payload(config: DatabricksEngineProbeJobConfig) -> dict[str, Any]:
    task: dict[str, Any] = {
        "task_key": config.task_key,
        "new_cluster": _engine_probe_cluster(config),
        "spark_python_task": {
            "python_file": config.runner_python_file,
            "parameters": _runner_parameters(config),
        },
    }
    task["spark_python_task"]["parameters"].extend(_pip_package_parameters(config.extra_pip_packages))
    task["spark_python_task"]["parameters"].extend(_package_wheel_parameters(config))
    return {
        "run_name": config.run_name,
        "tasks": [task],
    }


def build_databricks_engine_probe_matrix_run_submit_payload(
    config: DatabricksEngineProbeMatrixJobConfig,
) -> dict[str, Any]:
    tasks = [
        _engine_probe_task_from_target(config, target)
        for target in config.probe_targets
    ]
    if config.serial_tasks:
        _add_serial_task_dependencies(tasks)
    return {
        "run_name": config.run_name,
        "tasks": tasks,
    }


def _add_serial_task_dependencies(tasks: list[dict[str, Any]]) -> None:
    for previous, current in zip(tasks, tasks[1:], strict=False):
        previous_task_key = previous.get("task_key")
        if not isinstance(previous_task_key, str) or not previous_task_key:
            raise ValueError("serial engine-probe tasks require non-empty task_key values")
        current["depends_on"] = [{"task_key": previous_task_key}]


def write_databricks_engine_probe_run_submit_json(
    config: DatabricksEngineProbeJobConfig,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(build_databricks_engine_probe_run_submit_payload(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_databricks_engine_probe_matrix_run_submit_json(
    config: DatabricksEngineProbeMatrixJobConfig,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(build_databricks_engine_probe_matrix_run_submit_payload(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_databricks_engine_probe_runner_script(path: str | Path) -> None:
    Path(path).write_text(ENGINE_PROBE_RUNNER_SCRIPT, encoding="utf-8")


def run_engine_probe_task(argv: Sequence[str] | None = None) -> int:
    """Run an optional generated fixture step, then the native engine probe."""

    runner_argv = list(sys.argv[1:] if argv is None else argv)
    runner_args, engine_probe_argv = _split_fixture_runner_args(runner_argv)
    if runner_args.fixture_output_dir is not None:
        from document_kv_cache import probe_fixtures

        fixture_exit_code = probe_fixtures.main(
            [
                "--output-dir",
                runner_args.fixture_output_dir,
                "--backend",
                runner_args.fixture_backend,
                "--payload-mode",
                runner_args.fixture_payload_mode,
            ]
        )
        if fixture_exit_code:
            return fixture_exit_code
    if runner_args.vllm_runtime_preflight_output_json is not None:
        preflight_exit_code = _run_vllm_runtime_preflight(runner_args)
        if preflight_exit_code:
            return preflight_exit_code
    if runner_args.sglang_runtime_preflight_output_json is not None:
        preflight_exit_code = _run_sglang_runtime_preflight(runner_args)
        if preflight_exit_code:
            return preflight_exit_code
    from document_kv_cache import engine_probe

    return engine_probe.main(engine_probe_argv)


def _split_fixture_runner_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fixture-output-dir")
    parser.add_argument("--fixture-backend", choices=[backend.value for backend in ServingBackend])
    parser.add_argument(
        "--fixture-payload-mode",
        choices=[mode.value for mode in PayloadMode],
        default=PayloadMode.SEGMENTED.value,
    )
    parser.add_argument("--vllm-runtime-preflight-output-json")
    parser.add_argument("--vllm-runtime-preflight-layer-names-json")
    parser.add_argument("--sglang-runtime-preflight-output-json")
    parser.add_argument("--sglang-runtime-preflight-launch-config-json")
    fixture_args, engine_probe_argv = parser.parse_known_args(argv)
    if fixture_args.fixture_output_dir is not None and fixture_args.fixture_backend is None:
        raise ValueError("--fixture-backend is required when --fixture-output-dir is provided")
    if fixture_args.fixture_output_dir is None and fixture_args.fixture_backend is not None:
        raise ValueError("--fixture-backend requires --fixture-output-dir")
    if (
        fixture_args.vllm_runtime_preflight_output_json is None
        and fixture_args.vllm_runtime_preflight_layer_names_json is not None
    ):
        raise ValueError(
            "--vllm-runtime-preflight-layer-names-json requires --vllm-runtime-preflight-output-json"
        )
    if (
        fixture_args.vllm_runtime_preflight_output_json is not None
        and fixture_args.vllm_runtime_preflight_layer_names_json is None
    ):
        raise ValueError(
            "--vllm-runtime-preflight-output-json requires --vllm-runtime-preflight-layer-names-json"
        )
    if (
        fixture_args.sglang_runtime_preflight_output_json is None
        and fixture_args.sglang_runtime_preflight_launch_config_json is not None
    ):
        raise ValueError(
            "--sglang-runtime-preflight-launch-config-json requires --sglang-runtime-preflight-output-json"
        )
    if (
        fixture_args.sglang_runtime_preflight_output_json is not None
        and fixture_args.sglang_runtime_preflight_launch_config_json is None
    ):
        raise ValueError(
            "--sglang-runtime-preflight-output-json requires --sglang-runtime-preflight-launch-config-json"
        )
    return fixture_args, engine_probe_argv


def _run_vllm_runtime_preflight(runner_args: argparse.Namespace) -> int:
    from vllm_kv_injection import vllm_runtime_preflight

    return vllm_runtime_preflight.main(
        [
            "--layer-names-json",
            _cluster_preflight_file_argument(runner_args.vllm_runtime_preflight_layer_names_json),
            "--output-json",
            _cluster_preflight_file_argument(runner_args.vllm_runtime_preflight_output_json),
        ]
    )


def _run_sglang_runtime_preflight(runner_args: argparse.Namespace) -> int:
    from sglang_kv_injection import sglang_runtime_preflight

    return sglang_runtime_preflight.main(
        [
            "--launch-config-json",
            _cluster_preflight_file_argument(runner_args.sglang_runtime_preflight_launch_config_json),
            "--output-json",
            _cluster_preflight_file_argument(runner_args.sglang_runtime_preflight_output_json),
        ]
    )


def _cluster_preflight_file_argument(value: str) -> str:
    stripped_value = value.lstrip()
    if stripped_value.startswith(("{", "[")):
        return value
    scheme = _uri_scheme(value)
    if scheme in {"dbfs", "disk", "file", "uc-volume"} or value == "/Volumes" or value.startswith("/Volumes/"):
        return str(local_path(value))
    return value


def _cluster_config_from_engine_probe_job(
    config: DatabricksEngineProbeJobConfig,
) -> DatabricksSingleNodeGPUClusterConfig:
    return DatabricksSingleNodeGPUClusterConfig(
        purpose=DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE,
        node_type_id=config.node_type_id,
        spark_version=config.spark_version,
        data_security_mode=config.data_security_mode,
        single_user_name=config.single_user_name,
        availability=config.availability,
        zone_id=config.zone_id,
        custom_tags=config.custom_tags,
    )


def _cluster_config_from_engine_probe_matrix_job(
    config: DatabricksEngineProbeMatrixJobConfig,
) -> DatabricksSingleNodeGPUClusterConfig:
    return DatabricksSingleNodeGPUClusterConfig(
        purpose=DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE,
        node_type_id=config.node_type_id,
        spark_version=config.spark_version,
        data_security_mode=config.data_security_mode,
        single_user_name=config.single_user_name,
        availability=config.availability,
        zone_id=config.zone_id,
        custom_tags=config.custom_tags,
    )


def _engine_probe_task_from_target(
    config: DatabricksEngineProbeMatrixJobConfig,
    target: DatabricksEngineProbeTargetConfig,
) -> dict[str, Any]:
    single_config = DatabricksEngineProbeJobConfig(
        handoff_json=target.handoff_json,
        probe_factory=target.probe_factory,
        output_json=target.output_json,
        runner_python_file=config.runner_python_file,
        expected_backend=target.expected_backend,
        payload_uri=target.payload_uri,
        run_name=config.run_name,
        task_key=target.task_key or _default_task_key_for_backend(target.expected_backend),
        node_type_id=config.node_type_id,
        spark_version=config.spark_version,
        data_security_mode=config.data_security_mode,
        single_user_name=config.single_user_name,
        wheel_uri=config.wheel_uri,
        extra_wheel_uris=config.extra_wheel_uris,
        extra_pip_packages=(*config.extra_pip_packages, *target.pip_packages),
        engine_version=target.engine_version,
        allow_non_native_probe=target.allow_non_native_probe,
        metadata=target.metadata,
        release_safe=config.release_safe,
        availability=config.availability,
        zone_id=config.zone_id,
        custom_tags=config.custom_tags,
        actions_output_json=target.actions_output_json,
        native_probe_delegate_factory=target.native_probe_delegate_factory,
        fixture_output_dir=target.fixture_output_dir,
        fixture_payload_mode=target.fixture_payload_mode,
        vllm_runtime_preflight_output_json=target.vllm_runtime_preflight_output_json,
        vllm_runtime_preflight_layer_names_json=target.vllm_runtime_preflight_layer_names_json,
        sglang_runtime_preflight_output_json=target.sglang_runtime_preflight_output_json,
        sglang_runtime_preflight_launch_config_json=target.sglang_runtime_preflight_launch_config_json,
    )
    return build_databricks_engine_probe_run_submit_payload(single_config)["tasks"][0]


def _engine_probe_cluster(config: DatabricksEngineProbeJobConfig) -> dict[str, Any]:
    cluster = build_single_node_gpu_cluster(_cluster_config_from_engine_probe_job(config))
    if config.native_probe_delegate_factory is None:
        return cluster
    spark_env_vars = dict(cluster.get("spark_env_vars", {}))
    spark_env_vars[_native_probe_delegate_env_name(config.expected_backend)] = config.native_probe_delegate_factory
    cluster["spark_env_vars"] = spark_env_vars
    return cluster


def _runner_parameters(config: DatabricksEngineProbeJobConfig) -> list[str]:
    parameters = [
        "--handoff-json",
        config.handoff_json,
        "--probe-factory",
        config.probe_factory,
        "--output-json",
        config.output_json,
        "--expected-backend",
        config.expected_backend.value,
    ]
    if config.vllm_runtime_preflight_output_json is not None:
        if config.vllm_runtime_preflight_layer_names_json is None:
            raise ValueError(
                "vllm_runtime_preflight_layer_names_json must be set when "
                "vllm_runtime_preflight_output_json is set"
            )
        parameters = [
            "--vllm-runtime-preflight-output-json",
            config.vllm_runtime_preflight_output_json,
            "--vllm-runtime-preflight-layer-names-json",
            config.vllm_runtime_preflight_layer_names_json,
            *parameters,
        ]
    if config.sglang_runtime_preflight_output_json is not None:
        if config.sglang_runtime_preflight_launch_config_json is None:
            raise ValueError(
                "sglang_runtime_preflight_launch_config_json must be set when "
                "sglang_runtime_preflight_output_json is set"
            )
        parameters = [
            "--sglang-runtime-preflight-output-json",
            config.sglang_runtime_preflight_output_json,
            "--sglang-runtime-preflight-launch-config-json",
            config.sglang_runtime_preflight_launch_config_json,
            *parameters,
        ]
    if config.fixture_output_dir is not None:
        parameters = [
            "--fixture-output-dir",
            config.fixture_output_dir,
            "--fixture-backend",
            config.expected_backend.value,
            "--fixture-payload-mode",
            config.fixture_payload_mode.value,
            *parameters,
        ]
    if config.actions_output_json is not None and not _engine_probe_uses_fixture_actions_output(config):
        parameters.extend(["--actions-output-json", config.actions_output_json])
    if config.payload_uri is not None:
        parameters.extend(["--payload-uri", config.payload_uri])
    if config.engine_version is not None:
        parameters.extend(["--engine-version", config.engine_version])
    if config.allow_non_native_probe:
        parameters.append("--allow-non-native-probe")
    for metadata in config.metadata:
        parameters.extend(["--metadata", metadata])
    return parameters


def _package_wheel_parameters(
    config: DatabricksEngineProbeJobConfig | DatabricksEngineProbeMatrixJobConfig,
) -> list[str]:
    parameters: list[str] = []
    for wheel_uri in _package_wheel_uris(config):
        parameters.extend(["--package-wheel-uri", wheel_uri])
    return parameters


def _pip_package_parameters(pip_packages: Sequence[str]) -> list[str]:
    parameters: list[str] = []
    for pip_package in pip_packages:
        parameters.extend(["--pip-package", pip_package])
    return parameters


def _package_wheel_uris(
    config: DatabricksEngineProbeJobConfig | DatabricksEngineProbeMatrixJobConfig,
) -> tuple[str, ...]:
    wheel_uris = []
    if config.wheel_uri is not None:
        wheel_uris.append(config.wheel_uri)
    wheel_uris.extend(config.extra_wheel_uris)
    return tuple(wheel_uris)


def _coerce_probe_target(target: DatabricksEngineProbeTargetConfig) -> DatabricksEngineProbeTargetConfig:
    if isinstance(target, DatabricksEngineProbeTargetConfig):
        return target
    raise TypeError("probe_targets entries must be DatabricksEngineProbeTargetConfig")


def _validate_probe_target_backends(
    targets: Sequence[DatabricksEngineProbeTargetConfig],
    *,
    release_safe: bool,
) -> None:
    backends = tuple(target.expected_backend.value for target in targets)
    duplicate_backends = _duplicates(backends)
    if duplicate_backends:
        raise ValueError(f"probe_targets must not contain duplicate backends: {', '.join(duplicate_backends)}")
    if release_safe and set(backends) != set(REQUIRED_ENGINE_PROBE_BACKENDS):
        required = ", ".join(sorted(REQUIRED_ENGINE_PROBE_BACKENDS))
        observed = ", ".join(sorted(backends)) or "<none>"
        raise ValueError(f"release-safe probe matrix must include exactly required backends {required}; got {observed}")


def _validate_probe_target_task_keys(targets: Sequence[DatabricksEngineProbeTargetConfig]) -> None:
    task_keys = tuple(target.task_key or _default_task_key_for_backend(target.expected_backend) for target in targets)
    duplicate_task_keys = _duplicates(task_keys)
    if duplicate_task_keys:
        raise ValueError(f"probe target task keys must be unique: {', '.join(duplicate_task_keys)}")


def _validate_release_safe_probe_targets(
    targets: Sequence[DatabricksEngineProbeTargetConfig],
    *,
    release_safe: bool,
) -> None:
    if not release_safe:
        return
    for target in targets:
        if target.engine_version is not None:
            raise ValueError("release-safe probe matrix targets must not set engine_version")
        if target.allow_non_native_probe:
            raise ValueError("release-safe probe matrix targets must not allow non-native probes")
        _DEFAULT_VALIDATE_RELEASE_SAFE_PROVIDER_BACKED_VLLM_PREFLIGHT(target, label="engine probe matrix target")
        _DEFAULT_VALIDATE_RELEASE_SAFE_SGLANG_RUNTIME_PREFLIGHT(target, label="engine probe matrix target")


def _validate_release_safe_probe_job(config: DatabricksEngineProbeJobConfig) -> None:
    if type(config.release_safe) is not bool:
        raise ValueError("release_safe must be a boolean")
    if not config.release_safe:
        return
    if config.engine_version is not None:
        raise ValueError("release-safe engine probe jobs must not set engine_version")
    if config.allow_non_native_probe:
        raise ValueError("release-safe engine probe jobs must not allow non-native probes")
    _DEFAULT_VALIDATE_RELEASE_SAFE_PROVIDER_BACKED_VLLM_PREFLIGHT(config, label="engine probe job")
    _DEFAULT_VALIDATE_RELEASE_SAFE_SGLANG_RUNTIME_PREFLIGHT(config, label="engine probe job")


def _validate_vllm_runtime_preflight_config(
    config: DatabricksEngineProbeJobConfig | DatabricksEngineProbeTargetConfig,
    *,
    label: str,
) -> None:
    output_json = config.vllm_runtime_preflight_output_json
    layer_names_json = config.vllm_runtime_preflight_layer_names_json
    if output_json is not None and not output_json:
        raise ValueError(f"{label} vllm_runtime_preflight_output_json must be non-empty when provided")
    if layer_names_json is not None and not layer_names_json:
        raise ValueError(f"{label} vllm_runtime_preflight_layer_names_json must be non-empty when provided")
    if (output_json is None) != (layer_names_json is None):
        raise ValueError(
            f"{label} vLLM runtime preflight requires both "
            "vllm_runtime_preflight_output_json and vllm_runtime_preflight_layer_names_json"
        )
    if output_json is not None and config.expected_backend != ServingBackend.VLLM:
        raise ValueError(f"{label} vLLM runtime preflight is only supported for expected_backend vllm")


def _validate_sglang_runtime_preflight_config(
    config: DatabricksEngineProbeJobConfig | DatabricksEngineProbeTargetConfig,
    *,
    label: str,
) -> None:
    output_json = config.sglang_runtime_preflight_output_json
    launch_config_json = config.sglang_runtime_preflight_launch_config_json
    if output_json is not None and not output_json:
        raise ValueError(f"{label} sglang_runtime_preflight_output_json must be non-empty when provided")
    if launch_config_json is not None and not launch_config_json:
        raise ValueError(f"{label} sglang_runtime_preflight_launch_config_json must be non-empty when provided")
    if (output_json is None) != (launch_config_json is None):
        raise ValueError(
            f"{label} SGLang runtime preflight requires both "
            "sglang_runtime_preflight_output_json and sglang_runtime_preflight_launch_config_json"
        )
    if output_json is not None and config.expected_backend != ServingBackend.SGLANG:
        raise ValueError(f"{label} SGLang runtime preflight is only supported for expected_backend sglang")


def _validate_release_safe_provider_backed_vllm_preflight(
    config: DatabricksEngineProbeJobConfig | DatabricksEngineProbeTargetConfig,
    *,
    label: str,
) -> None:
    if not _is_provider_backed_vllm_probe(config):
        return
    if config.vllm_runtime_preflight_output_json is None:
        raise ValueError(
            f"release-safe provider-backed vLLM {label} must set "
            "vllm_runtime_preflight_output_json and vllm_runtime_preflight_layer_names_json"
        )


def _validate_release_safe_sglang_runtime_preflight(
    config: DatabricksEngineProbeJobConfig | DatabricksEngineProbeTargetConfig,
    *,
    label: str,
) -> None:
    if config.expected_backend != ServingBackend.SGLANG:
        return
    if config.sglang_runtime_preflight_output_json is None:
        raise ValueError(
            f"release-safe SGLang {label} must set "
            "sglang_runtime_preflight_output_json and sglang_runtime_preflight_launch_config_json"
        )


def _is_provider_backed_vllm_probe(
    config: DatabricksEngineProbeJobConfig | DatabricksEngineProbeTargetConfig,
) -> bool:
    if config.expected_backend != ServingBackend.VLLM:
        return False
    if config.native_probe_delegate_factory != VLLM_NATIVE_PROBE_DELEGATE_FACTORY:
        return False
    metadata = _metadata_item_map(config.metadata)
    return metadata.get("vllm_kv_injection.connector_factory") == VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY


def _default_task_key_for_backend(backend: ServingBackend) -> str:
    return f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_{backend.value}"


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _validate_metadata_items(items: Sequence[str]) -> None:
    if any(not _is_metadata_item(item) for item in items):
        raise ValueError("metadata entries must be non-empty KEY=VALUE strings")


def _validate_known_native_delegate_metadata(
    *,
    expected_backend: ServingBackend,
    native_probe_delegate_factory: str | None,
    metadata: Sequence[str],
    label: str,
) -> None:
    validate_known_native_delegate_metadata(
        backend=expected_backend,
        native_probe_delegate_factory=native_probe_delegate_factory,
        metadata=metadata,
        label=label,
        backend_field_label="expected_backend",
    )


def _metadata_item_map(items: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        key, _separator, value = item.partition("=")
        metadata[key] = value
    return metadata


def _validate_wheel_uris(items: Sequence[str], *, field_name: str) -> None:
    if isinstance(items, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence of non-empty strings")
    invalid_entries = [item for item in items if not isinstance(item, str) or not item]
    if invalid_entries:
        raise ValueError(f"{field_name} entries must be non-empty strings")


def _validate_pip_packages(items: Sequence[str], *, field_name: str) -> None:
    if isinstance(items, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence of non-empty package specs")
    invalid_entries = [
        item
        for item in items
        if not isinstance(item, str) or not item or item.startswith("-")
    ]
    if invalid_entries:
        raise ValueError(f"{field_name} entries must be non-empty package specs and not pip options")


def _is_metadata_item(item: str) -> bool:
    if not isinstance(item, str) or not item:
        return False
    key, separator, _value = item.partition("=")
    return bool(separator and key)


def _native_probe_delegate_env_name(backend: ServingBackend) -> str:
    if backend == ServingBackend.VLLM:
        return VLLM_NATIVE_PROBE_DELEGATE_ENV
    if backend == ServingBackend.SGLANG:
        return SGLANG_NATIVE_PROBE_DELEGATE_ENV
    raise ValueError(f"Unsupported serving backend {backend!r}")


def _serving_backend(value: ServingBackend | str) -> ServingBackend:
    try:
        return value if isinstance(value, ServingBackend) else ServingBackend(value)
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in ServingBackend)
        raise ValueError(f"expected_backend must be one of: {supported}") from exc


def _payload_mode(value: PayloadMode | str) -> PayloadMode:
    try:
        return value if isinstance(value, PayloadMode) else PayloadMode(value)
    except ValueError as exc:
        supported = ", ".join(mode.value for mode in PayloadMode)
        raise ValueError(f"fixture_payload_mode must be one of: {supported}") from exc


def _engine_probe_fixture_handoff_json(fixture_output_dir: str) -> str:
    return f"{fixture_output_dir.rstrip('/')}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}"


def _engine_probe_fixture_payload_uri(fixture_output_dir: str) -> str:
    return f"{fixture_output_dir.rstrip('/')}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['payload']}"


def _engine_probe_fixture_actions_json(fixture_output_dir: str) -> str:
    return f"{fixture_output_dir.rstrip('/')}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['actions']}"


def _engine_probe_fixture_vllm_layer_names_json(fixture_output_dir: str) -> str:
    return f"{fixture_output_dir.rstrip('/')}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['vllm_layer_names']}"


def _derive_vllm_fixture_runtime_preflight_layer_names(
    config: DatabricksEngineProbeJobConfig | DatabricksEngineProbeTargetConfig,
    *,
    label: str,
) -> None:
    if (
        config.expected_backend != ServingBackend.VLLM
        or config.fixture_output_dir is None
        or config.vllm_runtime_preflight_output_json is None
    ):
        return
    derived_layer_names_json = _engine_probe_fixture_vllm_layer_names_json(config.fixture_output_dir)
    if config.vllm_runtime_preflight_layer_names_json is None:
        object.__setattr__(config, "vllm_runtime_preflight_layer_names_json", derived_layer_names_json)
        return
    if _same_artifact_path(config.vllm_runtime_preflight_layer_names_json, derived_layer_names_json):
        return
    raise ValueError(
        f"{label} vllm_runtime_preflight_layer_names_json must match the derived fixture layer-name path "
        f"when fixture_output_dir is set: expected {derived_layer_names_json!r}, "
        f"got {config.vllm_runtime_preflight_layer_names_json!r}"
    )


def _engine_probe_uses_fixture_actions_output(
    config: DatabricksEngineProbeJobConfig | DatabricksEngineProbeTargetConfig,
) -> bool:
    return (
        config.fixture_output_dir is not None
        and config.actions_output_json is not None
        and _same_artifact_path(config.actions_output_json, _engine_probe_fixture_actions_json(config.fixture_output_dir))
    )


def _same_artifact_path(left: str, right: str) -> bool:
    return str(local_path(left).expanduser().resolve(strict=False)) == str(
        local_path(right).expanduser().resolve(strict=False)
    )


def _validate_fixture_output_dir(fixture_output_dir: str) -> None:
    scheme = _uri_scheme(fixture_output_dir)
    if scheme is not None and scheme not in {"dbfs", "disk", "file", "uc-volume"}:
        raise ValueError(
            "fixture_output_dir URI scheme must be one of dbfs:, disk:, file:, uc-volume:, "
            f"or an absolute/relative local path; got {fixture_output_dir!r}"
        )


def _validate_fixture_handoff_json(*, handoff_json: str, fixture_output_dir: str) -> None:
    expected_handoff_json = _engine_probe_fixture_handoff_json(fixture_output_dir)
    if handoff_json == expected_handoff_json:
        return
    raise ValueError(
        "handoff_json must match the derived fixture handoff path when fixture_output_dir is set: "
        f"expected {expected_handoff_json!r}, got {handoff_json!r}"
    )


def _validate_fixture_payload_uri(*, payload_uri: str | None, fixture_output_dir: str) -> None:
    if payload_uri is None:
        return
    expected_payload_uri = _engine_probe_fixture_payload_uri(fixture_output_dir)
    if payload_uri == expected_payload_uri:
        return
    raise ValueError(
        "payload_uri must match the derived fixture payload path when fixture_output_dir is set: "
        f"expected {expected_payload_uri!r}, got {payload_uri!r}"
    )


def _uri_scheme(uri: str) -> str | None:
    head = uri.split("/", maxsplit=1)[0]
    if ":" not in head:
        return None
    scheme, _separator, _rest = head.partition(":")
    return scheme or None


_DEFAULT_COERCE_PROBE_TARGET = _coerce_probe_target
_DEFAULT_VALIDATE_PROBE_TARGET_BACKENDS = _validate_probe_target_backends
_DEFAULT_VALIDATE_PROBE_TARGET_TASK_KEYS = _validate_probe_target_task_keys
_DEFAULT_VALIDATE_RELEASE_SAFE_PROBE_TARGETS = _validate_release_safe_probe_targets
_DEFAULT_VALIDATE_RELEASE_SAFE_PROBE_JOB = _validate_release_safe_probe_job
_DEFAULT_VALIDATE_VLLM_RUNTIME_PREFLIGHT_CONFIG = _validate_vllm_runtime_preflight_config
_DEFAULT_VALIDATE_RELEASE_SAFE_PROVIDER_BACKED_VLLM_PREFLIGHT = _validate_release_safe_provider_backed_vllm_preflight
_DEFAULT_VALIDATE_METADATA_ITEMS = _validate_metadata_items
_DEFAULT_VALIDATE_KNOWN_NATIVE_DELEGATE_METADATA = _validate_known_native_delegate_metadata
_DEFAULT_VALIDATE_WHEEL_URIS = _validate_wheel_uris
_DEFAULT_VALIDATE_PIP_PACKAGES = _validate_pip_packages
_DEFAULT_SERVING_BACKEND = _serving_backend
_DEFAULT_DERIVE_VLLM_FIXTURE_RUNTIME_PREFLIGHT_LAYER_NAMES = _derive_vllm_fixture_runtime_preflight_layer_names
_DEFAULT_VALIDATE_SGLANG_RUNTIME_PREFLIGHT_CONFIG = _validate_sglang_runtime_preflight_config
_DEFAULT_VALIDATE_RELEASE_SAFE_SGLANG_RUNTIME_PREFLIGHT = _validate_release_safe_sglang_runtime_preflight
_DEFAULT_CLUSTER_CONFIG_FROM_ENGINE_PROBE_JOB = _cluster_config_from_engine_probe_job
_DEFAULT_CLUSTER_CONFIG_FROM_ENGINE_PROBE_MATRIX_JOB = _cluster_config_from_engine_probe_matrix_job
_DEFAULT_PAYLOAD_MODE = _payload_mode
_DEFAULT_VALIDATE_FIXTURE_OUTPUT_DIR = _validate_fixture_output_dir
_DEFAULT_VALIDATE_FIXTURE_HANDOFF_JSON = _validate_fixture_handoff_json
_DEFAULT_VALIDATE_FIXTURE_PAYLOAD_URI = _validate_fixture_payload_uri


def read_databricks_engine_probe_targets_json(path: str | Path) -> tuple[DatabricksEngineProbeTargetConfig, ...]:
    return read_databricks_engine_probe_targets_file_json(path).probe_targets


def read_databricks_engine_probe_targets_file_json(path: str | Path) -> DatabricksEngineProbeTargetsFile:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    release_safe = False
    if isinstance(record, Mapping):
        _validate_engine_probe_targets_record_envelope(record)
        release_safe = bool(record.get("release_safe", False))
        raw_targets = record.get(DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY)
    else:
        raw_targets = record
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes, bytearray)) or not raw_targets:
        raise ValueError("backend config JSON must be a non-empty array or an object with a non-empty probes array")
    return DatabricksEngineProbeTargetsFile(
        probe_targets=tuple(
            _probe_target_from_record(raw_target, index=index)
            for index, raw_target in enumerate(raw_targets)
        ),
        release_safe=release_safe,
    )


def _validate_engine_probe_targets_record_envelope(record: Mapping[str, Any]) -> None:
    _reject_unsupported_keys(record, _ENGINE_PROBE_TARGETS_ENVELOPE_KEYS, label="engine probe targets record")
    record_type = record.get("record_type")
    if record_type is not None and record_type != ENGINE_PROBE_TARGETS_RECORD_TYPE:
        raise ValueError(f"Unsupported engine probe targets record_type {record_type!r}")
    schema_version = record.get("schema_version")
    if schema_version is not None and schema_version != ENGINE_PROBE_TARGETS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported engine probe targets schema_version {schema_version!r}; "
            f"expected {ENGINE_PROBE_TARGETS_SCHEMA_VERSION}"
        )
    release_safe = record.get("release_safe", False)
    if type(release_safe) is not bool:
        raise ValueError("engine probe targets release_safe must be a boolean")


def _probe_target_from_record(record: Any, *, index: int) -> DatabricksEngineProbeTargetConfig:
    if not isinstance(record, Mapping):
        raise ValueError(f"backend config probe {index} must be an object")
    _reject_unsupported_keys(record, _ENGINE_PROBE_TARGET_KEYS, label=f"backend config probe {index}")
    output_json = record.get("output_json", record.get("probe_output_json"))
    allow_non_native_probe = record.get("allow_non_native_probe", False)
    if type(allow_non_native_probe) is not bool:
        raise ValueError(f"backend config probe {index}.allow_non_native_probe must be a boolean")
    metadata = record.get("metadata", ())
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes, bytearray)):
        raise ValueError(f"backend config probe {index}.metadata must be an array of KEY=VALUE strings")
    pip_packages = record.get("pip_packages", ())
    if not isinstance(pip_packages, Sequence) or isinstance(pip_packages, (str, bytes, bytearray)):
        raise ValueError(f"backend config probe {index}.pip_packages must be an array of package specs")
    return DatabricksEngineProbeTargetConfig(
        expected_backend=record.get("expected_backend", record.get("backend")),
        handoff_json=record.get("handoff_json"),
        probe_factory=record.get("probe_factory"),
        output_json=output_json,
        payload_uri=record.get("payload_uri"),
        task_key=record.get("task_key"),
        engine_version=record.get("engine_version"),
        allow_non_native_probe=allow_non_native_probe,
        metadata=tuple(metadata),
        actions_output_json=record.get("actions_output_json", record.get("connector_actions_output_json")),
        native_probe_delegate_factory=record.get("native_probe_delegate_factory"),
        fixture_output_dir=record.get("fixture_output_dir"),
        fixture_payload_mode=record.get("fixture_payload_mode", PayloadMode.SEGMENTED),
        pip_packages=tuple(pip_packages),
        vllm_runtime_preflight_output_json=record.get("vllm_runtime_preflight_output_json"),
        vllm_runtime_preflight_layer_names_json=record.get("vllm_runtime_preflight_layer_names_json"),
        sglang_runtime_preflight_output_json=record.get("sglang_runtime_preflight_output_json"),
        sglang_runtime_preflight_launch_config_json=record.get("sglang_runtime_preflight_launch_config_json"),
    )


def _reject_unsupported_keys(record: Mapping[str, Any], allowed_keys: frozenset[str], *, label: str) -> None:
    unsupported = sorted(str(key) for key in record if key not in allowed_keys)
    if unsupported:
        raise ValueError(f"{label} has unsupported keys: {unsupported}")


def _required_single_arg(args: argparse.Namespace, name: str) -> str:
    value = getattr(args, name)
    if not value:
        cli_name = name.replace("_", "-")
        raise ValueError(f"--{cli_name} is required unless --backend-config-json is provided")
    return value


def _provider_backed_native_probe_preset(args: argparse.Namespace) -> ServingBackend | None:
    selected: list[ServingBackend] = []
    if args.provider_backed_vllm_native_probe:
        selected.append(ServingBackend.VLLM)
    if args.provider_backed_sglang_native_probe:
        selected.append(ServingBackend.SGLANG)
    if len(selected) > 1:
        option_names = ", ".join(_PROVIDER_BACKED_NATIVE_PROBE_OPTION_NAMES[backend] for backend in selected)
        raise ValueError(f"provider-backed native probe presets are mutually exclusive: {option_names}")
    return selected[0] if selected else None


def _provider_backed_native_probe_option_name(backend: ServingBackend) -> str:
    return _PROVIDER_BACKED_NATIVE_PROBE_OPTION_NAMES[backend]


def _single_target_arg_with_provider_backed_default(
    args: argparse.Namespace,
    name: str,
    defaults: Mapping[ServingBackend, str],
) -> str:
    value = getattr(args, name)
    preset = _provider_backed_native_probe_preset(args)
    if preset is None:
        return _required_single_arg(args, name)
    default = defaults[preset]
    if value is not None and value != default:
        cli_name = name.replace("_", "-")
        option_name = _provider_backed_native_probe_option_name(preset)
        raise ValueError(
            f"{option_name} sets --{cli_name} to {default!r}; "
            f"got conflicting value {value!r}"
        )
    return default


def _optional_single_target_arg_with_provider_backed_default(
    args: argparse.Namespace,
    name: str,
    defaults: Mapping[ServingBackend, str],
) -> str | None:
    value = getattr(args, name)
    preset = _provider_backed_native_probe_preset(args)
    if preset is None:
        return value
    default = defaults[preset]
    if value is not None and value != default:
        cli_name = name.replace("_", "-")
        option_name = _provider_backed_native_probe_option_name(preset)
        raise ValueError(
            f"{option_name} sets --{cli_name} to {default!r}; "
            f"got conflicting value {value!r}"
        )
    return default


def _single_target_metadata_from_cli(args: argparse.Namespace) -> tuple[str, ...]:
    metadata = list(args.metadata or ())
    preset = _provider_backed_native_probe_preset(args)
    if preset is not None:
        _append_required_metadata(
            metadata,
            required_item=_PROVIDER_BACKED_NATIVE_PROBE_METADATA[preset],
            option_name=_provider_backed_native_probe_option_name(preset),
        )
    return tuple(metadata)


def _append_required_metadata(metadata: list[str], *, required_item: str, option_name: str) -> None:
    required_key, _separator, required_value = required_item.partition("=")
    for item in metadata:
        key, separator, value = item.partition("=")
        if separator and key == required_key:
            if value != required_value:
                raise ValueError(f"{option_name} requires metadata entry {required_item}")
            return
    metadata.append(required_item)


def _single_target_extra_pip_packages_from_cli(args: argparse.Namespace) -> tuple[str, ...]:
    packages = list(args.extra_pip_package or ())
    preset = _provider_backed_native_probe_preset(args)
    if preset is not None:
        _append_required_pip_package(
            packages,
            required_package=_PROVIDER_BACKED_NATIVE_PROBE_RUNTIME_PACKAGES[preset],
            option_name=_provider_backed_native_probe_option_name(preset),
        )
    return tuple(packages)


def _single_target_extra_wheel_uris_from_cli(args: argparse.Namespace) -> tuple[str, ...]:
    preset = _provider_backed_native_probe_preset(args)
    if preset is not None and args.extra_wheel_uri:
        option_name = _provider_backed_native_probe_option_name(preset)
        raise ValueError(
            f"{option_name} cannot be combined with --extra-wheel-uri; "
            f"the provider-backed {preset.value} adapter modules must come from the Cachet wheel"
        )
    return tuple(args.extra_wheel_uri or ())


def _single_target_engine_version_from_cli(args: argparse.Namespace) -> str | None:
    preset = _provider_backed_native_probe_preset(args)
    if preset is not None and args.engine_version is not None:
        option_name = _provider_backed_native_probe_option_name(preset)
        raise ValueError(f"{option_name} cannot be combined with --engine-version")
    return args.engine_version


def _single_target_allow_non_native_probe_from_cli(args: argparse.Namespace) -> bool:
    preset = _provider_backed_native_probe_preset(args)
    if preset is not None and args.allow_non_native_probe:
        option_name = _provider_backed_native_probe_option_name(preset)
        raise ValueError(f"{option_name} cannot be combined with --allow-non-native-probe")
    return args.allow_non_native_probe


def _append_required_pip_package(packages: list[str], *, required_package: str, option_name: str) -> None:
    required_name = _pip_package_name(required_package)
    for package in packages:
        if _pip_package_name(package) == required_name:
            if package != required_package:
                raise ValueError(f"{option_name} requires pip package {required_package}")
            return
    packages.append(required_package)


def _pip_package_name(package: str) -> str:
    normalized = package.strip()
    name_chars = []
    for char in normalized:
        if char.isalnum() or char in {"-", "_", "."}:
            name_chars.append(char)
            continue
        break
    return "".join(name_chars).replace("_", "-").lower()


def _single_target_handoff_json_from_cli(args: argparse.Namespace) -> str:
    if args.fixture_output_dir is None:
        if args.fixture_payload_mode is not None:
            raise ValueError("--fixture-payload-mode requires --fixture-output-dir")
        return _required_single_arg(args, "handoff_json")
    derived_handoff_json = _engine_probe_fixture_handoff_json(args.fixture_output_dir)
    if args.handoff_json:
        _validate_fixture_handoff_json(
            handoff_json=args.handoff_json,
            fixture_output_dir=args.fixture_output_dir,
        )
        return args.handoff_json
    return derived_handoff_json


def _reject_single_target_args_for_matrix(args: argparse.Namespace) -> None:
    incompatible_values = {
        "handoff-json": args.handoff_json,
        "probe-factory": args.probe_factory,
        "probe-output-json": args.probe_output_json,
        "actions-output-json": args.actions_output_json,
        "expected-backend": args.expected_backend,
        "payload-uri": args.payload_uri,
        "engine-version": args.engine_version,
        "native-probe-delegate-factory": args.native_probe_delegate_factory,
        "fixture-output-dir": args.fixture_output_dir,
        "fixture-payload-mode": args.fixture_payload_mode,
        "vllm-runtime-preflight-output-json": args.vllm_runtime_preflight_output_json,
        "vllm-runtime-preflight-layer-names-json": args.vllm_runtime_preflight_layer_names_json,
        "sglang-runtime-preflight-output-json": args.sglang_runtime_preflight_output_json,
        "sglang-runtime-preflight-launch-config-json": args.sglang_runtime_preflight_launch_config_json,
        "task-key": args.task_key,
    }
    provided = [f"--{name}" for name, value in incompatible_values.items() if value]
    if args.allow_non_native_probe:
        provided.append("--allow-non-native-probe")
    if args.metadata:
        provided.append("--metadata")
    if args.extra_pip_package:
        provided.append("--extra-pip-package")
    if args.provider_backed_vllm_native_probe:
        provided.append("--provider-backed-vllm-native-probe")
    if args.provider_backed_sglang_native_probe:
        provided.append("--provider-backed-sglang-native-probe")
    if provided:
        raise ValueError(
            "--backend-config-json cannot be combined with single-target probe options; "
            f"move per-backend values into the backend config JSON: {', '.join(provided)}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a Databricks runs/submit payload for a V1 AWS single-node GPU engine probe."
    )
    parser.add_argument("--handoff-json", help="Cluster-visible engine handoff JSON path or URI.")
    parser.add_argument("--probe-factory", help="Dotted native probe factory, e.g. module:factory.")
    parser.add_argument("--probe-output-json", help="Cluster-visible native probe evidence output path.")
    parser.add_argument("--actions-output-json", help="Optional cluster-visible connector actions descriptor output path.")
    parser.add_argument("--runner-python-file", required=True, help="Cluster-visible runner script path or URI.")
    parser.add_argument("--expected-backend", choices=[backend.value for backend in ServingBackend])
    parser.add_argument(
        "--provider-backed-vllm-native-probe",
        action="store_true",
        help=(
            "Single-target shortcut for Cachet's built-in provider-backed vLLM native probe. "
            "Sets the Cachet vLLM probe factory, vLLM delegate factory, required connector "
            f"metadata, expected backend vllm, and {DEFAULT_VLLM_ENGINE_PROBE_RUNTIME_PACKAGE}."
        ),
    )
    parser.add_argument(
        "--provider-backed-sglang-native-probe",
        action="store_true",
        help=(
            "Single-target shortcut for Cachet's built-in provider-backed SGLang HiCache native probe. "
            "Sets the Cachet SGLang probe factory, SGLang delegate factory, required connector "
            f"metadata, expected backend sglang, and {DEFAULT_SGLANG_ENGINE_PROBE_RUNTIME_PACKAGE}. "
            "Release-safe jobs still require the SGLang runtime preflight output and launch config sidecar."
        ),
    )
    parser.add_argument(
        "--backend-config-json",
        help=(
            "JSON array, or object with a probes array, describing backend probe targets. "
            "Each target needs expected_backend/backend, handoff_json, probe_factory, "
            "and output_json/probe_output_json."
        ),
    )
    parser.add_argument("--payload-uri", help="Override payload_source.uri from the handoff record.")
    parser.add_argument(
        "--fixture-output-dir",
        help=(
            "Generate a deterministic Qwen3 V1 fixture in this cluster-visible directory before "
            "running a single-target probe. If --handoff-json is omitted, the handoff path is derived from it."
        ),
    )
    parser.add_argument(
        "--fixture-payload-mode",
        choices=[mode.value for mode in PayloadMode],
        help="Fixture payload mode when --fixture-output-dir is provided.",
    )
    parser.add_argument(
        "--vllm-runtime-preflight-output-json",
        help=(
            "Cluster-visible JSON output path for Cachet's strict vLLM runtime preflight. "
            "Required for release-safe provider-backed vLLM native probes."
        ),
    )
    parser.add_argument(
        "--vllm-runtime-preflight-layer-names-json",
        help=(
            "Cluster-visible path, or inline JSON, containing registered vLLM KV cache layer names "
            "for --vllm-runtime-preflight-output-json. When --fixture-output-dir is set for a vLLM "
            "probe, this defaults to the fixture-generated vllm-layer-names.json sidecar."
        ),
    )
    parser.add_argument(
        "--sglang-runtime-preflight-output-json",
        help=(
            "Cluster-visible JSON output path for Cachet's strict SGLang runtime preflight. "
            "Required for release-safe SGLang native probes."
        ),
    )
    parser.add_argument(
        "--sglang-runtime-preflight-launch-config-json",
        help=(
            "Cluster-visible path, or inline JSON, containing the provider-backed SGLang HiCache "
            "launch config for --sglang-runtime-preflight-output-json."
        ),
    )
    parser.add_argument("--run-name", default=DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME)
    parser.add_argument("--task-key")
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
    parser.add_argument(
        "--extra-wheel-uri",
        action="append",
        help=(
            "Additional cluster-visible custom extension wheel URI to install before the task. "
            "Cachet's built-in vLLM/SGLang adapter modules ship in --wheel-uri. "
            "May be repeated; wheels install after --wheel-uri in argument order."
        ),
    )
    parser.add_argument(
        "--extra-pip-package",
        action="append",
        help=(
            "Additional PyPI package spec to install before local wheels. "
            "In matrix mode, use per-target pip_packages for backend-specific engine stacks."
        ),
    )
    parser.add_argument("--engine-version", help="Fallback engine version for legacy or non-native debug probes.")
    parser.add_argument(
        "--native-probe-delegate-factory",
        help=(
            "Backend-native delegate factory to expose through the built-in native probe "
            "factory environment variable on the Databricks cluster."
        ),
    )
    parser.add_argument(
        "--metadata",
        action="append",
        metavar="KEY=VALUE",
        help="Additional string metadata to attach to a single-target probe evidence record.",
    )
    parser.add_argument(
        "--allow-non-native-probe",
        action="store_true",
        help="Pass through to engine-probe debugging mode; release evidence rejects these records.",
    )
    parser.add_argument(
        "--release-safe",
        action="store_true",
        help="Reject debug-only probe options so the generated job stays eligible for release evidence.",
    )
    parser.add_argument(
        "--serial-tasks",
        action="store_true",
        help=(
            "In matrix mode, add task dependencies so backend probes run one after another "
            "instead of requesting multiple GPU clusters at once."
        ),
    )
    parser.add_argument("--output-json", help="Write the runs/submit payload to this path instead of stdout.")
    parser.add_argument("--runner-script-output", help="Write the tiny engine-probe runner script to this path.")
    args = parser.parse_args(argv)

    try:
        if args.backend_config_json:
            _reject_single_target_args_for_matrix(args)
            targets_file = read_databricks_engine_probe_targets_file_json(args.backend_config_json)
            config = DatabricksEngineProbeMatrixJobConfig(
                probe_targets=targets_file.probe_targets,
                runner_python_file=args.runner_python_file,
                run_name=args.run_name,
                node_type_id=databricks_node_type_for_hardware_target(args.hardware_target, args.node_type_id),
                spark_version=args.spark_version,
                data_security_mode=args.data_security_mode,
                single_user_name=args.single_user_name,
                wheel_uri=args.wheel_uri,
                extra_wheel_uris=tuple(args.extra_wheel_uri or ()),
                extra_pip_packages=tuple(args.extra_pip_package or ()),
                release_safe=args.release_safe or targets_file.release_safe,
                serial_tasks=args.serial_tasks,
            )
            payload = build_databricks_engine_probe_matrix_run_submit_payload(config)
        else:
            if args.serial_tasks:
                raise ValueError("--serial-tasks requires --backend-config-json")
            _provider_backed_native_probe_preset(args)
            config = DatabricksEngineProbeJobConfig(
                handoff_json=_single_target_handoff_json_from_cli(args),
                probe_factory=_single_target_arg_with_provider_backed_default(
                    args,
                    "probe_factory",
                    _PROVIDER_BACKED_NATIVE_PROBE_FACTORIES,
                ),
                output_json=_required_single_arg(args, "probe_output_json"),
                runner_python_file=args.runner_python_file,
                expected_backend=_single_target_arg_with_provider_backed_default(
                    args,
                    "expected_backend",
                    {
                        ServingBackend.VLLM: ServingBackend.VLLM.value,
                        ServingBackend.SGLANG: ServingBackend.SGLANG.value,
                    },
                ),
                payload_uri=args.payload_uri,
                run_name=args.run_name,
                task_key=args.task_key or DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY,
                node_type_id=databricks_node_type_for_hardware_target(args.hardware_target, args.node_type_id),
                spark_version=args.spark_version,
                data_security_mode=args.data_security_mode,
                single_user_name=args.single_user_name,
                wheel_uri=args.wheel_uri,
                extra_wheel_uris=_single_target_extra_wheel_uris_from_cli(args),
                extra_pip_packages=_single_target_extra_pip_packages_from_cli(args),
                engine_version=_single_target_engine_version_from_cli(args),
                allow_non_native_probe=_single_target_allow_non_native_probe_from_cli(args),
                metadata=_single_target_metadata_from_cli(args),
                release_safe=args.release_safe,
                actions_output_json=args.actions_output_json,
                native_probe_delegate_factory=_optional_single_target_arg_with_provider_backed_default(
                    args,
                    "native_probe_delegate_factory",
                    _PROVIDER_BACKED_NATIVE_PROBE_DELEGATE_FACTORIES,
                ),
                fixture_output_dir=args.fixture_output_dir,
                fixture_payload_mode=args.fixture_payload_mode or PayloadMode.SEGMENTED,
                vllm_runtime_preflight_output_json=args.vllm_runtime_preflight_output_json,
                vllm_runtime_preflight_layer_names_json=args.vllm_runtime_preflight_layer_names_json,
                sglang_runtime_preflight_output_json=args.sglang_runtime_preflight_output_json,
                sglang_runtime_preflight_launch_config_json=args.sglang_runtime_preflight_launch_config_json,
            )
            payload = build_databricks_engine_probe_run_submit_payload(config)
        if args.runner_script_output:
            write_databricks_engine_probe_runner_script(args.runner_script_output)
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
