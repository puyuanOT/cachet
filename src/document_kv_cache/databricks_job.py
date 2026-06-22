"""Databricks runs/submit payload helpers for V1 benchmark plans."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_kv_cache._hardware_targets import (
    DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE,
    SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES,
    SUPPORTED_V1_HARDWARE_TARGETS,
    databricks_node_type_for_hardware_target,
    validate_aws_single_node_gpu_type as _validate_aws_single_node_gpu_type,
)
from document_kv_cache.native_probe_factories import (
    SGLANG_NATIVE_PROBE_DELEGATE_ENV,
    VLLM_NATIVE_PROBE_DELEGATE_ENV,
)
from document_kv_cache.transformers_generator import (
    CACHET_TRANSFORMERS_MODEL_KWARGS_JSON_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV,
)


DEFAULT_AWS_G5_NODE_TYPE = DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE
DEFAULT_DATABRICKS_SPARK_VERSION = "15.4.x-gpu-ml-scala2.12"
DEFAULT_DATABRICKS_RUN_NAME = "document-kv-v1-benchmark"
DEFAULT_DATABRICKS_TASK_KEY = "document_kv_v1_benchmark"
DEFAULT_DATABRICKS_PURPOSE = "document-kv-v1-benchmark"
DEFAULT_DATABRICKS_DATA_SECURITY_MODE = "SINGLE_USER"
DEDICATED_DATABRICKS_DATA_SECURITY_MODE = "DATA_SECURITY_MODE_DEDICATED"
SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES = frozenset(
    {DEFAULT_DATABRICKS_DATA_SECURITY_MODE, DEDICATED_DATABRICKS_DATA_SECURITY_MODE}
)
RESERVED_SINGLE_NODE_GPU_TAG_KEYS = frozenset({"ResourceClass", "purpose"})
RESERVED_SINGLE_NODE_G5_TAG_KEYS = RESERVED_SINGLE_NODE_GPU_TAG_KEYS
RESERVED_SPARK_ENV_VAR_KEYS = frozenset(
    {
        VLLM_NATIVE_PROBE_DELEGATE_ENV,
        SGLANG_NATIVE_PROBE_DELEGATE_ENV,
    }
)
SECRET_LIKE_SPARK_ENV_KEY_PARTS = frozenset(
    {
        "CREDENTIAL",
        "CREDENTIALS",
        "KEY",
        "PASS",
        "PASSWORD",
        "PAT",
        "SECRET",
        "TOKEN",
    }
)
_DATABRICKS_TOKEN_PATTERN = re.compile(r"dapi[0-9a-fA-F]{32}")
_ENV_KEY_PART_RE = re.compile(r"[A-Za-z0-9]+")
_SPARK_ENV_VAR_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_JSON_SPARK_ENV_VAR_KEYS = frozenset(
    {
        CACHET_TRANSFORMERS_MODEL_KWARGS_JSON_ENV,
        CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV,
    }
)
RUNNER_SCRIPT = """from __future__ import annotations

import argparse
import subprocess
import sys


def _cluster_file_path(uri: str) -> str:
    if uri.startswith("dbfs:/"):
        return "/dbfs/" + uri.removeprefix("dbfs:/").lstrip("/")
    return uri


def _install_package_wheel(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--package-wheel-uri")
    args, remaining = parser.parse_known_args(argv)
    if args.package_wheel_uri:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", _cluster_file_path(args.package_wheel_uri)]
        )
    return remaining

if __name__ == "__main__":
    remaining_args = _install_package_wheel(sys.argv[1:])
    from document_kv_cache.benchmark_plan_executor import main

    exit_code = main(remaining_args)
    if exit_code:
        raise SystemExit(exit_code)
"""

__all__ = [
    "DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE",
    "DEFAULT_AWS_G5_NODE_TYPE",
    "DEFAULT_DATABRICKS_SPARK_VERSION",
    "DEFAULT_DATABRICKS_RUN_NAME",
    "DEFAULT_DATABRICKS_TASK_KEY",
    "DEFAULT_DATABRICKS_PURPOSE",
    "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
    "DEDICATED_DATABRICKS_DATA_SECURITY_MODE",
    "SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES",
    "RESERVED_SINGLE_NODE_GPU_TAG_KEYS",
    "DatabricksSingleNodeG5ClusterConfig",
    "DatabricksSingleNodeGPUClusterConfig",
    "DatabricksBenchmarkJobConfig",
    "validate_aws_g5_node_type",
    "validate_aws_single_node_gpu_type",
    "build_single_node_g5_cluster",
    "build_single_node_gpu_cluster",
    "build_databricks_run_submit_payload",
    "write_databricks_run_submit_json",
    "write_databricks_runner_script",
    "main",
]


@dataclass(frozen=True, slots=True)
class DatabricksSingleNodeGPUClusterConfig:
    purpose: str
    node_type_id: str = DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE
    spark_version: str = DEFAULT_DATABRICKS_SPARK_VERSION
    data_security_mode: str = DEFAULT_DATABRICKS_DATA_SECURITY_MODE
    single_user_name: str | None = None
    availability: str = "ON_DEMAND"
    zone_id: str = "auto"
    custom_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.purpose:
            raise ValueError("purpose must be non-empty")
        _DEFAULT_VALIDATE_AWS_SINGLE_NODE_GPU_TYPE(self.node_type_id)
        if not self.spark_version:
            raise ValueError("spark_version must be non-empty")
        if not self.data_security_mode:
            raise ValueError("data_security_mode must be non-empty")
        if self.single_user_name is not None and not self.single_user_name:
            raise ValueError("single_user_name must be non-empty when provided")
        if _DEFAULT_IS_SINGLE_USER_MODE(self.data_security_mode) and self.single_user_name is None:
            raise ValueError("single_user_name is required when data_security_mode is SINGLE_USER")
        if not self.availability:
            raise ValueError("availability must be non-empty")
        if not self.zone_id:
            raise ValueError("zone_id must be non-empty")
        reserved_tags = RESERVED_SINGLE_NODE_GPU_TAG_KEYS.intersection(self.custom_tags)
        if reserved_tags:
            raise ValueError(f"custom_tags cannot override reserved tags: {sorted(reserved_tags)!r}")


DatabricksSingleNodeG5ClusterConfig = DatabricksSingleNodeGPUClusterConfig


@dataclass(frozen=True, slots=True)
class DatabricksBenchmarkJobConfig:
    plan_json_uri: str
    runner_python_file: str
    run_name: str = DEFAULT_DATABRICKS_RUN_NAME
    task_key: str = DEFAULT_DATABRICKS_TASK_KEY
    node_type_id: str = DEFAULT_AWS_G5_NODE_TYPE
    spark_version: str = DEFAULT_DATABRICKS_SPARK_VERSION
    data_security_mode: str = DEFAULT_DATABRICKS_DATA_SECURITY_MODE
    single_user_name: str | None = None
    wheel_uri: str | None = None
    execution_result_json_uri: str | None = None
    availability: str = "ON_DEMAND"
    zone_id: str = "auto"
    custom_tags: Mapping[str, str] = field(default_factory=dict)
    vllm_native_probe_delegate_factory: str | None = None
    sglang_native_probe_delegate_factory: str | None = None
    spark_env_vars: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plan_json_uri:
            raise ValueError("plan_json_uri must be non-empty")
        if not self.runner_python_file:
            raise ValueError("runner_python_file must be non-empty")
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if not self.task_key:
            raise ValueError("task_key must be non-empty")
        _DEFAULT_CLUSTER_CONFIG_FROM_BENCHMARK_JOB(self)
        if self.wheel_uri is not None and not self.wheel_uri:
            raise ValueError("wheel_uri must be non-empty when provided")
        if self.execution_result_json_uri is not None and not self.execution_result_json_uri:
            raise ValueError("execution_result_json_uri must be non-empty when provided")
        if self.vllm_native_probe_delegate_factory is not None and not self.vllm_native_probe_delegate_factory:
            raise ValueError("vllm_native_probe_delegate_factory must be non-empty when provided")
        if self.sglang_native_probe_delegate_factory is not None and not self.sglang_native_probe_delegate_factory:
            raise ValueError("sglang_native_probe_delegate_factory must be non-empty when provided")
        object.__setattr__(self, "spark_env_vars", _validated_spark_env_vars(self.spark_env_vars))


def validate_aws_single_node_gpu_type(node_type_id: str) -> None:
    _validate_aws_single_node_gpu_type(node_type_id)


validate_aws_g5_node_type = validate_aws_single_node_gpu_type


def build_databricks_run_submit_payload(config: DatabricksBenchmarkJobConfig) -> dict[str, Any]:
    task: dict[str, Any] = {
        "task_key": config.task_key,
        "new_cluster": _single_node_g5_cluster(config),
        "spark_python_task": {
            "python_file": config.runner_python_file,
            "parameters": _runner_parameters(config),
        },
    }
    if config.wheel_uri is not None:
        task["spark_python_task"]["parameters"].extend(["--package-wheel-uri", config.wheel_uri])
    return {
        "run_name": config.run_name,
        "tasks": [task],
    }


def write_databricks_run_submit_json(config: DatabricksBenchmarkJobConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(build_databricks_run_submit_payload(config), indent=2, sort_keys=True) + "\n")


def write_databricks_runner_script(path: str | Path) -> None:
    Path(path).write_text(RUNNER_SCRIPT, encoding="utf-8")


def build_single_node_gpu_cluster(config: DatabricksSingleNodeGPUClusterConfig) -> dict[str, Any]:
    tags = {
        "ResourceClass": "SingleNode",
        "purpose": config.purpose,
        **dict(config.custom_tags),
    }
    cluster = {
        "spark_version": config.spark_version,
        "node_type_id": config.node_type_id,
        "driver_node_type_id": config.node_type_id,
        "data_security_mode": config.data_security_mode,
        "num_workers": 0,
        "spark_conf": {
            "spark.master": "local[*]",
            "spark.databricks.cluster.profile": "singleNode",
        },
        "custom_tags": tags,
        "aws_attributes": {
            "availability": config.availability,
            "zone_id": config.zone_id,
        },
    }
    if _is_single_user_mode(config.data_security_mode):
        cluster["single_user_name"] = config.single_user_name
    return cluster


build_single_node_g5_cluster = build_single_node_gpu_cluster


def _single_node_gpu_cluster(config: DatabricksBenchmarkJobConfig) -> dict[str, Any]:
    cluster = build_single_node_gpu_cluster(_cluster_config_from_benchmark_job(config))
    spark_env_vars = _native_probe_delegate_env_vars(config)
    if spark_env_vars:
        cluster["spark_env_vars"] = spark_env_vars
    return cluster


_single_node_g5_cluster = _single_node_gpu_cluster


def _cluster_config_from_benchmark_job(config: DatabricksBenchmarkJobConfig) -> DatabricksSingleNodeGPUClusterConfig:
    return DatabricksSingleNodeGPUClusterConfig(
        purpose=DEFAULT_DATABRICKS_PURPOSE,
        node_type_id=config.node_type_id,
        spark_version=config.spark_version,
        data_security_mode=config.data_security_mode,
        single_user_name=config.single_user_name,
        availability=config.availability,
        zone_id=config.zone_id,
        custom_tags=config.custom_tags,
    )


def _is_single_user_mode(data_security_mode: str) -> bool:
    return data_security_mode.upper() in SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES


def _native_probe_delegate_env_vars(config: DatabricksBenchmarkJobConfig) -> dict[str, str]:
    spark_env_vars = dict(config.spark_env_vars)
    if config.vllm_native_probe_delegate_factory is not None:
        spark_env_vars[VLLM_NATIVE_PROBE_DELEGATE_ENV] = config.vllm_native_probe_delegate_factory
    if config.sglang_native_probe_delegate_factory is not None:
        spark_env_vars[SGLANG_NATIVE_PROBE_DELEGATE_ENV] = config.sglang_native_probe_delegate_factory
    return spark_env_vars


def _validated_spark_env_vars(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("spark_env_vars must be a mapping")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("spark_env_vars keys must be non-empty strings")
        if _SPARK_ENV_VAR_KEY_RE.fullmatch(key) is None:
            raise ValueError(
                f"spark_env_vars key {key!r} must be a valid environment variable name"
            )
        if key in RESERVED_SPARK_ENV_VAR_KEYS:
            raise ValueError(
                f"spark_env_vars cannot set reserved native-probe key {key!r}; "
                "use the dedicated native-probe delegate factory options"
            )
        if _looks_secret_like_env_key(key):
            raise ValueError(f"spark_env_vars key {key!r} looks secret-bearing")
        if not isinstance(item, str) or not item:
            raise ValueError(f"spark_env_vars.{key} must be a non-empty string")
        if _DATABRICKS_TOKEN_PATTERN.search(item):
            raise ValueError(f"spark_env_vars.{key} must not contain a Databricks token pattern")
        if key in _JSON_SPARK_ENV_VAR_KEYS:
            _validate_non_secret_json_env_value(key, item)
        normalized[key] = item
    return normalized


def _validate_non_secret_json_env_value(env_key: str, value: str) -> None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"spark_env_vars.{env_key} must contain a JSON object") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"spark_env_vars.{env_key} must contain a JSON object")
    _validate_json_value_without_secret_paths(parsed, f"spark_env_vars.{env_key}")


def _validate_json_value_without_secret_paths(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            if _looks_secret_like_env_key(key):
                raise ValueError(f"{path}.{key} looks secret-bearing")
            _validate_json_value_without_secret_paths(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value_without_secret_paths(item, f"{path}[{index}]")
    elif isinstance(value, str) and _DATABRICKS_TOKEN_PATTERN.search(value):
        raise ValueError(f"{path} must not contain a Databricks token pattern")


def _looks_secret_like_env_key(key: str) -> bool:
    parts = {part.upper() for part in _ENV_KEY_PART_RE.findall(key)}
    return bool(parts.intersection(SECRET_LIKE_SPARK_ENV_KEY_PARTS))


_DEFAULT_VALIDATE_AWS_SINGLE_NODE_GPU_TYPE = validate_aws_single_node_gpu_type
_DEFAULT_VALIDATE_AWS_G5_NODE_TYPE = validate_aws_g5_node_type
_DEFAULT_IS_SINGLE_USER_MODE = _is_single_user_mode
_DEFAULT_CLUSTER_CONFIG_FROM_BENCHMARK_JOB = _cluster_config_from_benchmark_job


def _runner_parameters(config: DatabricksBenchmarkJobConfig) -> list[str]:
    parameters = ["--plan-json", config.plan_json_uri]
    if config.execution_result_json_uri is not None:
        parameters.extend(["--result-json", config.execution_result_json_uri])
    return parameters


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a Databricks runs/submit payload for a V1 AWS single-node GPU benchmark."
    )
    parser.add_argument("--plan-json-uri", required=True, help="Cluster-visible plan JSON path or URI.")
    parser.add_argument("--runner-python-file", required=True, help="Cluster-visible runner script path or URI.")
    parser.add_argument("--run-name", default=DEFAULT_DATABRICKS_RUN_NAME)
    parser.add_argument("--task-key", default=DEFAULT_DATABRICKS_TASK_KEY)
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
    parser.add_argument("--execution-result-json-uri", help="Optional cluster-visible execution summary output path.")
    parser.add_argument(
        "--vllm-native-probe-delegate-factory",
        help="Optional backend-native delegate factory for benchmark plans that run Cachet's built-in vLLM probe.",
    )
    parser.add_argument(
        "--sglang-native-probe-delegate-factory",
        help="Optional backend-native delegate factory for benchmark plans that run Cachet's built-in SGLang probe.",
    )
    parser.add_argument(
        "--spark-env-var",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Non-secret Databricks cluster spark_env_vars entry for runtime configuration, "
            "for example CACHET_TRANSFORMERS_DEVICE=cuda. Repeat as needed."
        ),
    )
    parser.add_argument("--output-json", help="Write the runs/submit payload to this path instead of stdout.")
    parser.add_argument("--runner-script-output", help="Write the tiny benchmark plan runner script to this path.")
    args = parser.parse_args(argv)

    try:
        config = DatabricksBenchmarkJobConfig(
            plan_json_uri=args.plan_json_uri,
            runner_python_file=args.runner_python_file,
            run_name=args.run_name,
            task_key=args.task_key,
            node_type_id=databricks_node_type_for_hardware_target(args.hardware_target, args.node_type_id),
            spark_version=args.spark_version,
            data_security_mode=args.data_security_mode,
            single_user_name=args.single_user_name,
            wheel_uri=args.wheel_uri,
            execution_result_json_uri=args.execution_result_json_uri,
            vllm_native_probe_delegate_factory=args.vllm_native_probe_delegate_factory,
            sglang_native_probe_delegate_factory=args.sglang_native_probe_delegate_factory,
            spark_env_vars=_spark_env_vars_from_cli(args.spark_env_var or ()),
        )
        if args.runner_script_output:
            write_databricks_runner_script(args.runner_script_output)
        payload = build_databricks_run_submit_payload(config)
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


def _spark_env_vars_from_cli(values: Sequence[str]) -> dict[str, str]:
    spark_env_vars: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("spark env var entries must use KEY=VALUE syntax")
        key, item = value.split("=", 1)
        if key in spark_env_vars:
            raise ValueError(f"duplicate spark env var key {key!r}")
        spark_env_vars[key] = item
    return _validated_spark_env_vars(spark_env_vars)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
