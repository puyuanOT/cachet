"""Databricks runs/submit payload helpers for V1 benchmark plans."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_kv_cache.native_probe_factories import (
    SGLANG_NATIVE_PROBE_DELEGATE_ENV,
    VLLM_NATIVE_PROBE_DELEGATE_ENV,
)


DEFAULT_AWS_G5_NODE_TYPE = "g5.4xlarge"
DEFAULT_DATABRICKS_SPARK_VERSION = "15.4.x-gpu-ml-scala2.12"
DEFAULT_DATABRICKS_RUN_NAME = "document-kv-v1-benchmark"
DEFAULT_DATABRICKS_TASK_KEY = "document_kv_v1_benchmark"
DEFAULT_DATABRICKS_DATA_SECURITY_MODE = "SINGLE_USER"
DEDICATED_DATABRICKS_DATA_SECURITY_MODE = "DATA_SECURITY_MODE_DEDICATED"
SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES = frozenset(
    {DEFAULT_DATABRICKS_DATA_SECURITY_MODE, DEDICATED_DATABRICKS_DATA_SECURITY_MODE}
)
RESERVED_SINGLE_NODE_G5_TAG_KEYS = frozenset({"ResourceClass", "purpose"})
RUNNER_SCRIPT = """from document_kv_cache.benchmark_plan_executor import main

if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
"""

__all__ = [
    "DEFAULT_AWS_G5_NODE_TYPE",
    "DEFAULT_DATABRICKS_SPARK_VERSION",
    "DEFAULT_DATABRICKS_RUN_NAME",
    "DEFAULT_DATABRICKS_TASK_KEY",
    "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
    "DEDICATED_DATABRICKS_DATA_SECURITY_MODE",
    "SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES",
    "DatabricksSingleNodeG5ClusterConfig",
    "DatabricksBenchmarkJobConfig",
    "validate_aws_g5_node_type",
    "build_single_node_g5_cluster",
    "build_databricks_run_submit_payload",
    "write_databricks_run_submit_json",
    "write_databricks_runner_script",
    "main",
]


@dataclass(frozen=True, slots=True)
class DatabricksSingleNodeG5ClusterConfig:
    purpose: str
    node_type_id: str = DEFAULT_AWS_G5_NODE_TYPE
    spark_version: str = DEFAULT_DATABRICKS_SPARK_VERSION
    data_security_mode: str = DEFAULT_DATABRICKS_DATA_SECURITY_MODE
    single_user_name: str | None = None
    availability: str = "ON_DEMAND"
    zone_id: str = "auto"
    custom_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.purpose:
            raise ValueError("purpose must be non-empty")
        _DEFAULT_VALIDATE_AWS_G5_NODE_TYPE(self.node_type_id)
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
        reserved_tags = RESERVED_SINGLE_NODE_G5_TAG_KEYS.intersection(self.custom_tags)
        if reserved_tags:
            raise ValueError(f"custom_tags cannot override reserved tags: {sorted(reserved_tags)!r}")


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


def validate_aws_g5_node_type(node_type_id: str) -> None:
    if not node_type_id:
        raise ValueError("node_type_id must be non-empty")
    if not node_type_id.lower().startswith("g5."):
        raise ValueError(f"node_type_id must be an AWS g5 Databricks node type, got {node_type_id!r}")


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
        task["libraries"] = [{"whl": config.wheel_uri}]
    return {
        "run_name": config.run_name,
        "tasks": [task],
    }


def write_databricks_run_submit_json(config: DatabricksBenchmarkJobConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(build_databricks_run_submit_payload(config), indent=2, sort_keys=True) + "\n")


def write_databricks_runner_script(path: str | Path) -> None:
    Path(path).write_text(RUNNER_SCRIPT, encoding="utf-8")


def build_single_node_g5_cluster(config: DatabricksSingleNodeG5ClusterConfig) -> dict[str, Any]:
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


def _single_node_g5_cluster(config: DatabricksBenchmarkJobConfig) -> dict[str, Any]:
    cluster = build_single_node_g5_cluster(_cluster_config_from_benchmark_job(config))
    spark_env_vars = _native_probe_delegate_env_vars(config)
    if spark_env_vars:
        cluster["spark_env_vars"] = spark_env_vars
    return cluster


def _cluster_config_from_benchmark_job(config: DatabricksBenchmarkJobConfig) -> DatabricksSingleNodeG5ClusterConfig:
    return DatabricksSingleNodeG5ClusterConfig(
        purpose="document-kv-v1-benchmark",
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
    spark_env_vars: dict[str, str] = {}
    if config.vllm_native_probe_delegate_factory is not None:
        spark_env_vars[VLLM_NATIVE_PROBE_DELEGATE_ENV] = config.vllm_native_probe_delegate_factory
    if config.sglang_native_probe_delegate_factory is not None:
        spark_env_vars[SGLANG_NATIVE_PROBE_DELEGATE_ENV] = config.sglang_native_probe_delegate_factory
    return spark_env_vars


_DEFAULT_VALIDATE_AWS_G5_NODE_TYPE = validate_aws_g5_node_type
_DEFAULT_IS_SINGLE_USER_MODE = _is_single_user_mode
_DEFAULT_CLUSTER_CONFIG_FROM_BENCHMARK_JOB = _cluster_config_from_benchmark_job


def _runner_parameters(config: DatabricksBenchmarkJobConfig) -> list[str]:
    parameters = ["--plan-json", config.plan_json_uri]
    if config.execution_result_json_uri is not None:
        parameters.extend(["--result-json", config.execution_result_json_uri])
    return parameters


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit a Databricks runs/submit payload for a V1 AWS g5 benchmark.")
    parser.add_argument("--plan-json-uri", required=True, help="Cluster-visible plan JSON path or URI.")
    parser.add_argument("--runner-python-file", required=True, help="Cluster-visible runner script path or URI.")
    parser.add_argument("--run-name", default=DEFAULT_DATABRICKS_RUN_NAME)
    parser.add_argument("--task-key", default=DEFAULT_DATABRICKS_TASK_KEY)
    parser.add_argument("--node-type-id", default=DEFAULT_AWS_G5_NODE_TYPE)
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
    parser.add_argument("--output-json", help="Write the runs/submit payload to this path instead of stdout.")
    parser.add_argument("--runner-script-output", help="Write the tiny benchmark plan runner script to this path.")
    args = parser.parse_args(argv)

    try:
        config = DatabricksBenchmarkJobConfig(
            plan_json_uri=args.plan_json_uri,
            runner_python_file=args.runner_python_file,
            run_name=args.run_name,
            task_key=args.task_key,
            node_type_id=args.node_type_id,
            spark_version=args.spark_version,
            data_security_mode=args.data_security_mode,
            single_user_name=args.single_user_name,
            wheel_uri=args.wheel_uri,
            execution_result_json_uri=args.execution_result_json_uri,
            vllm_native_probe_delegate_factory=args.vllm_native_probe_delegate_factory,
            sglang_native_probe_delegate_factory=args.sglang_native_probe_delegate_factory,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
