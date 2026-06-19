"""Databricks runs/submit payload helpers for engine KV-connector probes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_kv_cache.databricks_job import (
    DEFAULT_AWS_G5_NODE_TYPE,
    DEFAULT_DATABRICKS_DATA_SECURITY_MODE,
    DEFAULT_DATABRICKS_SPARK_VERSION,
    DatabricksSingleNodeG5ClusterConfig,
    build_single_node_g5_cluster,
)
from document_kv_cache.engine_adapters import ServingBackend
from document_kv_cache.release_evidence import REQUIRED_ENGINE_PROBE_BACKENDS


DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME = "document-kv-engine-probe"
DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY = "document_kv_engine_probe"
DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE = "document-kv-engine-probe"
DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY = "probes"
ENGINE_PROBE_TARGETS_RECORD_TYPE = "document_kv.engine_probe_targets.v1"
ENGINE_PROBE_TARGETS_SCHEMA_VERSION = 1
ENGINE_PROBE_RUNNER_SCRIPT = """from document_kv_cache.engine_probe import main

if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
"""

__all__ = [
    "DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME",
    "DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY",
    "DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE",
    "DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY",
    "ENGINE_PROBE_RUNNER_SCRIPT",
    "DatabricksEngineProbeJobConfig",
    "DatabricksEngineProbeMatrixJobConfig",
    "DatabricksEngineProbeTargetConfig",
    "DatabricksEngineProbeTargetsFile",
    "build_databricks_engine_probe_run_submit_payload",
    "build_databricks_engine_probe_matrix_run_submit_payload",
    "read_databricks_engine_probe_targets_json",
    "read_databricks_engine_probe_targets_file_json",
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_backend", _DEFAULT_SERVING_BACKEND(self.expected_backend))
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
        if type(self.allow_non_native_probe) is not bool:
            raise ValueError("allow_non_native_probe must be a boolean")
        _DEFAULT_VALIDATE_METADATA_ITEMS(self.metadata)
        object.__setattr__(self, "metadata", tuple(self.metadata))


@dataclass(frozen=True, slots=True)
class DatabricksEngineProbeMatrixJobConfig:
    """A release-oriented Databricks job that runs native probes for required backends."""

    probe_targets: Sequence[DatabricksEngineProbeTargetConfig]
    runner_python_file: str
    run_name: str = DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME
    node_type_id: str = DEFAULT_AWS_G5_NODE_TYPE
    spark_version: str = DEFAULT_DATABRICKS_SPARK_VERSION
    data_security_mode: str = DEFAULT_DATABRICKS_DATA_SECURITY_MODE
    single_user_name: str | None = None
    wheel_uri: str | None = None
    release_safe: bool = False
    availability: str = "ON_DEMAND"
    zone_id: str = "auto"
    custom_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.probe_targets:
            raise ValueError("probe_targets must be non-empty")
        if not self.runner_python_file:
            raise ValueError("runner_python_file must be non-empty")
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if self.wheel_uri is not None and not self.wheel_uri:
            raise ValueError("wheel_uri must be non-empty when provided")
        if type(self.release_safe) is not bool:
            raise ValueError("release_safe must be a boolean")
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
    node_type_id: str = DEFAULT_AWS_G5_NODE_TYPE
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

    def __post_init__(self) -> None:
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
        if self.engine_version is not None and not self.engine_version:
            raise ValueError("engine_version must be non-empty when provided")
        _DEFAULT_VALIDATE_METADATA_ITEMS(self.metadata)
        object.__setattr__(self, "metadata", tuple(self.metadata))
        _DEFAULT_VALIDATE_RELEASE_SAFE_PROBE_JOB(self)
        object.__setattr__(self, "expected_backend", _DEFAULT_SERVING_BACKEND(self.expected_backend))
        _DEFAULT_CLUSTER_CONFIG_FROM_ENGINE_PROBE_JOB(self)


def build_databricks_engine_probe_run_submit_payload(config: DatabricksEngineProbeJobConfig) -> dict[str, Any]:
    task: dict[str, Any] = {
        "task_key": config.task_key,
        "new_cluster": build_single_node_g5_cluster(_cluster_config_from_engine_probe_job(config)),
        "spark_python_task": {
            "python_file": config.runner_python_file,
            "parameters": _runner_parameters(config),
        },
    }
    if config.wheel_uri is not None:
        task["libraries"] = [{"whl": config.wheel_uri}]
    return {
        "run_name": config.run_name,
        "tasks": [task],
    }


def build_databricks_engine_probe_matrix_run_submit_payload(
    config: DatabricksEngineProbeMatrixJobConfig,
) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "tasks": [
            _engine_probe_task_from_target(config, target)
            for target in config.probe_targets
        ],
    }


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


def _cluster_config_from_engine_probe_job(
    config: DatabricksEngineProbeJobConfig,
) -> DatabricksSingleNodeG5ClusterConfig:
    return DatabricksSingleNodeG5ClusterConfig(
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
) -> DatabricksSingleNodeG5ClusterConfig:
    return DatabricksSingleNodeG5ClusterConfig(
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
        engine_version=target.engine_version,
        allow_non_native_probe=target.allow_non_native_probe,
        metadata=target.metadata,
        release_safe=config.release_safe,
        availability=config.availability,
        zone_id=config.zone_id,
        custom_tags=config.custom_tags,
    )
    return build_databricks_engine_probe_run_submit_payload(single_config)["tasks"][0]


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
    if config.payload_uri is not None:
        parameters.extend(["--payload-uri", config.payload_uri])
    if config.engine_version is not None:
        parameters.extend(["--engine-version", config.engine_version])
    if config.allow_non_native_probe:
        parameters.append("--allow-non-native-probe")
    for metadata in config.metadata:
        parameters.extend(["--metadata", metadata])
    return parameters


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


def _validate_release_safe_probe_job(config: DatabricksEngineProbeJobConfig) -> None:
    if type(config.release_safe) is not bool:
        raise ValueError("release_safe must be a boolean")
    if not config.release_safe:
        return
    if config.engine_version is not None:
        raise ValueError("release-safe engine probe jobs must not set engine_version")
    if config.allow_non_native_probe:
        raise ValueError("release-safe engine probe jobs must not allow non-native probes")


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


def _is_metadata_item(item: str) -> bool:
    if not isinstance(item, str) or not item:
        return False
    key, separator, _value = item.partition("=")
    return bool(separator and key)


def _serving_backend(value: ServingBackend | str) -> ServingBackend:
    try:
        return value if isinstance(value, ServingBackend) else ServingBackend(value)
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in ServingBackend)
        raise ValueError(f"expected_backend must be one of: {supported}") from exc


_DEFAULT_COERCE_PROBE_TARGET = _coerce_probe_target
_DEFAULT_VALIDATE_PROBE_TARGET_BACKENDS = _validate_probe_target_backends
_DEFAULT_VALIDATE_PROBE_TARGET_TASK_KEYS = _validate_probe_target_task_keys
_DEFAULT_VALIDATE_RELEASE_SAFE_PROBE_TARGETS = _validate_release_safe_probe_targets
_DEFAULT_VALIDATE_RELEASE_SAFE_PROBE_JOB = _validate_release_safe_probe_job
_DEFAULT_VALIDATE_METADATA_ITEMS = _validate_metadata_items
_DEFAULT_SERVING_BACKEND = _serving_backend
_DEFAULT_CLUSTER_CONFIG_FROM_ENGINE_PROBE_JOB = _cluster_config_from_engine_probe_job
_DEFAULT_CLUSTER_CONFIG_FROM_ENGINE_PROBE_MATRIX_JOB = _cluster_config_from_engine_probe_matrix_job


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
    output_json = record.get("output_json", record.get("probe_output_json"))
    allow_non_native_probe = record.get("allow_non_native_probe", False)
    if type(allow_non_native_probe) is not bool:
        raise ValueError(f"backend config probe {index}.allow_non_native_probe must be a boolean")
    metadata = record.get("metadata", ())
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes, bytearray)):
        raise ValueError(f"backend config probe {index}.metadata must be an array of KEY=VALUE strings")
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
    )


def _required_single_arg(args: argparse.Namespace, name: str) -> str:
    value = getattr(args, name)
    if not value:
        cli_name = name.replace("_", "-")
        raise ValueError(f"--{cli_name} is required unless --backend-config-json is provided")
    return value


def _reject_single_target_args_for_matrix(args: argparse.Namespace) -> None:
    incompatible_values = {
        "handoff-json": args.handoff_json,
        "probe-factory": args.probe_factory,
        "probe-output-json": args.probe_output_json,
        "expected-backend": args.expected_backend,
        "payload-uri": args.payload_uri,
        "engine-version": args.engine_version,
        "task-key": args.task_key,
    }
    provided = [f"--{name}" for name, value in incompatible_values.items() if value]
    if args.allow_non_native_probe:
        provided.append("--allow-non-native-probe")
    if args.metadata:
        provided.append("--metadata")
    if provided:
        raise ValueError(
            "--backend-config-json cannot be combined with single-target probe options; "
            f"move per-backend values into the backend config JSON: {', '.join(provided)}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit a Databricks runs/submit payload for an AWS g5 engine probe.")
    parser.add_argument("--handoff-json", help="Cluster-visible engine handoff JSON path or URI.")
    parser.add_argument("--probe-factory", help="Dotted native probe factory, e.g. module:factory.")
    parser.add_argument("--probe-output-json", help="Cluster-visible native probe evidence output path.")
    parser.add_argument("--runner-python-file", required=True, help="Cluster-visible runner script path or URI.")
    parser.add_argument("--expected-backend", choices=[backend.value for backend in ServingBackend])
    parser.add_argument(
        "--backend-config-json",
        help=(
            "JSON array, or object with a probes array, describing backend probe targets. "
            "Each target needs expected_backend/backend, handoff_json, probe_factory, "
            "and output_json/probe_output_json."
        ),
    )
    parser.add_argument("--payload-uri", help="Override payload_source.uri from the handoff record.")
    parser.add_argument("--run-name", default=DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME)
    parser.add_argument("--task-key")
    parser.add_argument("--node-type-id", default=DEFAULT_AWS_G5_NODE_TYPE)
    parser.add_argument("--spark-version", default=DEFAULT_DATABRICKS_SPARK_VERSION)
    parser.add_argument("--data-security-mode", default=DEFAULT_DATABRICKS_DATA_SECURITY_MODE)
    parser.add_argument("--single-user-name", help="Required when --data-security-mode SINGLE_USER.")
    parser.add_argument("--wheel-uri", help="Optional cluster-visible wheel URI to install before the task.")
    parser.add_argument("--engine-version", help="Fallback engine version for legacy or non-native debug probes.")
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
                node_type_id=args.node_type_id,
                spark_version=args.spark_version,
                data_security_mode=args.data_security_mode,
                single_user_name=args.single_user_name,
                wheel_uri=args.wheel_uri,
                release_safe=args.release_safe or targets_file.release_safe,
            )
            payload = build_databricks_engine_probe_matrix_run_submit_payload(config)
        else:
            config = DatabricksEngineProbeJobConfig(
                handoff_json=_required_single_arg(args, "handoff_json"),
                probe_factory=_required_single_arg(args, "probe_factory"),
                output_json=_required_single_arg(args, "probe_output_json"),
                runner_python_file=args.runner_python_file,
                expected_backend=_required_single_arg(args, "expected_backend"),
                payload_uri=args.payload_uri,
                run_name=args.run_name,
                task_key=args.task_key or DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY,
                node_type_id=args.node_type_id,
                spark_version=args.spark_version,
                data_security_mode=args.data_security_mode,
                single_user_name=args.single_user_name,
                wheel_uri=args.wheel_uri,
                engine_version=args.engine_version,
                allow_non_native_probe=args.allow_non_native_probe,
                metadata=tuple(args.metadata or ()),
                release_safe=args.release_safe,
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
