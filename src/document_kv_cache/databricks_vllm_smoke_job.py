"""Databricks runs/submit payload helpers for the Qwen3 vLLM smoke benchmark."""

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
from document_kv_cache.vllm_smoke import (
    DEFAULT_LOCAL_ROOT,
    SERVER_HOST,
    SERVER_PORT,
)


DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME = "document-kv-vllm-smoke"
DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY = "document_kv_vllm_smoke"
DEFAULT_DATABRICKS_VLLM_SMOKE_PURPOSE = "document-kv-vllm-smoke"
VLLM_SMOKE_RUNNER_SCRIPT = """from document_kv_cache.vllm_smoke import main

if __name__ == "__main__":
    exit_code = main()
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
    node_type_id: str = DEFAULT_AWS_G5_NODE_TYPE
    spark_version: str = DEFAULT_DATABRICKS_SPARK_VERSION
    data_security_mode: str = DEFAULT_DATABRICKS_DATA_SECURITY_MODE
    single_user_name: str | None = None
    wheel_uri: str | None = None
    max_tokens: int = 32
    timeout_seconds: float = 240.0
    import_probe_timeout_seconds: float = 180.0
    server_start_timeout_seconds: float = 480.0
    local_root: str = str(DEFAULT_LOCAL_ROOT)
    server_host: str = SERVER_HOST
    server_port: int = SERVER_PORT
    client_host: str = SERVER_HOST
    availability: str = "ON_DEMAND"
    zone_id: str = "auto"
    custom_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty")
        if not self.runner_python_file:
            raise ValueError("runner_python_file must be non-empty")
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if not self.task_key:
            raise ValueError("task_key must be non-empty")
        if self.wheel_uri is not None and not self.wheel_uri:
            raise ValueError("wheel_uri must be non-empty when provided")
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
        _DEFAULT_CLUSTER_CONFIG_FROM_VLLM_SMOKE_JOB(self)


def build_databricks_vllm_smoke_run_submit_payload(config: DatabricksVLLMSmokeJobConfig) -> dict[str, Any]:
    task: dict[str, Any] = {
        "task_key": config.task_key,
        "new_cluster": build_single_node_g5_cluster(_cluster_config_from_vllm_smoke_job(config)),
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


def _cluster_config_from_vllm_smoke_job(config: DatabricksVLLMSmokeJobConfig) -> DatabricksSingleNodeG5ClusterConfig:
    return DatabricksSingleNodeG5ClusterConfig(
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


def _runner_parameters(config: DatabricksVLLMSmokeJobConfig) -> list[str]:
    return [
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
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit a Databricks runs/submit payload for the AWS g5 vLLM smoke.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", required=True, help="Cluster-visible output directory for smoke artifacts.")
    parser.add_argument("--runner-python-file", required=True, help="Cluster-visible runner script path or URI.")
    parser.add_argument("--run-name", default=DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME)
    parser.add_argument("--task-key", default=DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY)
    parser.add_argument("--node-type-id", default=DEFAULT_AWS_G5_NODE_TYPE)
    parser.add_argument("--spark-version", default=DEFAULT_DATABRICKS_SPARK_VERSION)
    parser.add_argument("--data-security-mode", default=DEFAULT_DATABRICKS_DATA_SECURITY_MODE)
    parser.add_argument("--single-user-name", help="Required when --data-security-mode SINGLE_USER.")
    parser.add_argument("--wheel-uri", help="Optional cluster-visible wheel URI to install before the task.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--import-probe-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--server-start-timeout-seconds", type=float, default=480.0)
    parser.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--server-host", default=SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=SERVER_PORT)
    parser.add_argument("--client-host", default=SERVER_HOST)
    parser.add_argument("--output-json", help="Write the runs/submit payload to this path instead of stdout.")
    parser.add_argument("--runner-script-output", help="Write the tiny vLLM smoke runner script to this path.")
    args = parser.parse_args(argv)

    try:
        config = DatabricksVLLMSmokeJobConfig(
            benchmark_id=args.benchmark_id,
            output_dir=args.output_dir,
            runner_python_file=args.runner_python_file,
            run_name=args.run_name,
            task_key=args.task_key,
            node_type_id=args.node_type_id,
            spark_version=args.spark_version,
            data_security_mode=args.data_security_mode,
            single_user_name=args.single_user_name,
            wheel_uri=args.wheel_uri,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            import_probe_timeout_seconds=args.import_probe_timeout_seconds,
            server_start_timeout_seconds=args.server_start_timeout_seconds,
            local_root=args.local_root,
            server_host=args.server_host,
            server_port=args.server_port,
            client_host=args.client_host,
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
