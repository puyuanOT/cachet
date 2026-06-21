"""Databricks runs/submit payload helpers for storage-reader benchmarks."""

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
from document_kv_cache.storage_benchmark import (
    SUPPORTED_STORAGE_BENCHMARK_READERS,
    StorageBenchmarkConfig,
)
from document_kv_cache.storage import is_real_uc_volume_root


DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME = "document-kv-storage-benchmark"
DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY = "document_kv_storage_benchmark"
DEFAULT_DATABRICKS_STORAGE_BENCHMARK_PURPOSE = "document-kv-storage-benchmark"
STORAGE_BENCHMARK_RUNNER_SCRIPT = """from __future__ import annotations

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
    from document_kv_cache.storage_benchmark import main

    exit_code = main(remaining_args)
    if exit_code:
        raise SystemExit(exit_code)
"""

__all__ = [
    "DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME",
    "DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY",
    "DEFAULT_DATABRICKS_STORAGE_BENCHMARK_PURPOSE",
    "STORAGE_BENCHMARK_RUNNER_SCRIPT",
    "DatabricksStorageBenchmarkJobConfig",
    "build_databricks_storage_benchmark_run_submit_payload",
    "write_databricks_storage_benchmark_run_submit_json",
    "write_databricks_storage_benchmark_runner_script",
    "main",
]


@dataclass(frozen=True, slots=True)
class DatabricksStorageBenchmarkJobConfig:
    workspace_dir: str
    output_json: str
    runner_python_file: str
    uc_volume_root: str
    benchmark_id: str = DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME
    chunk_count: int = 64
    chunk_bytes: int = 1024 * 1024
    repeats: int = 4
    parallelism: int = 4
    readers: tuple[str, ...] = SUPPORTED_STORAGE_BENCHMARK_READERS
    align_bytes: int = 4096
    run_name: str = DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME
    task_key: str = DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY
    node_type_id: str = DEFAULT_AWS_G5_NODE_TYPE
    spark_version: str = DEFAULT_DATABRICKS_SPARK_VERSION
    data_security_mode: str = DEFAULT_DATABRICKS_DATA_SECURITY_MODE
    single_user_name: str | None = None
    wheel_uri: str | None = None
    availability: str = "ON_DEMAND"
    zone_id: str = "auto"
    custom_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.workspace_dir:
            raise ValueError("workspace_dir must be non-empty")
        if not self.output_json:
            raise ValueError("output_json must be non-empty")
        if not self.runner_python_file:
            raise ValueError("runner_python_file must be non-empty")
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if not self.uc_volume_root:
            raise ValueError("uc_volume_root must be non-empty")
        if _DEFAULT_IS_REAL_UC_VOLUME_ROOT(self.uc_volume_root) is not True:
            raise ValueError("uc_volume_root must be a real /Volumes/<catalog>/<schema>/<volume> path")
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if not self.task_key:
            raise ValueError("task_key must be non-empty")
        if self.wheel_uri is not None and not self.wheel_uri:
            raise ValueError("wheel_uri must be non-empty when provided")
        object.__setattr__(self, "readers", tuple(self.readers))
        _DEFAULT_STORAGE_BENCHMARK_CONFIG(
            workspace_dir=self.workspace_dir,
            benchmark_id=self.benchmark_id,
            chunk_count=self.chunk_count,
            chunk_bytes=self.chunk_bytes,
            repeats=self.repeats,
            parallelism=self.parallelism,
            readers=self.readers,
            align_bytes=self.align_bytes,
            uc_volume_root=self.uc_volume_root,
        )
        _DEFAULT_CLUSTER_CONFIG_FROM_STORAGE_BENCHMARK_JOB(self)


def build_databricks_storage_benchmark_run_submit_payload(
    config: DatabricksStorageBenchmarkJobConfig,
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "task_key": config.task_key,
        "new_cluster": build_single_node_g5_cluster(_cluster_config_from_storage_benchmark_job(config)),
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


def write_databricks_storage_benchmark_run_submit_json(
    config: DatabricksStorageBenchmarkJobConfig,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(build_databricks_storage_benchmark_run_submit_payload(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_databricks_storage_benchmark_runner_script(path: str | Path) -> None:
    Path(path).write_text(STORAGE_BENCHMARK_RUNNER_SCRIPT, encoding="utf-8")


def _cluster_config_from_storage_benchmark_job(
    config: DatabricksStorageBenchmarkJobConfig,
) -> DatabricksSingleNodeG5ClusterConfig:
    return DatabricksSingleNodeG5ClusterConfig(
        purpose=DEFAULT_DATABRICKS_STORAGE_BENCHMARK_PURPOSE,
        node_type_id=config.node_type_id,
        spark_version=config.spark_version,
        data_security_mode=config.data_security_mode,
        single_user_name=config.single_user_name,
        availability=config.availability,
        zone_id=config.zone_id,
        custom_tags=config.custom_tags,
    )


_DEFAULT_IS_REAL_UC_VOLUME_ROOT = is_real_uc_volume_root
_DEFAULT_STORAGE_BENCHMARK_CONFIG = StorageBenchmarkConfig
_DEFAULT_CLUSTER_CONFIG_FROM_STORAGE_BENCHMARK_JOB = _cluster_config_from_storage_benchmark_job


def _runner_parameters(config: DatabricksStorageBenchmarkJobConfig) -> list[str]:
    parameters = [
        "--workspace-dir",
        config.workspace_dir,
        "--benchmark-id",
        config.benchmark_id,
        "--chunk-count",
        str(config.chunk_count),
        "--chunk-bytes",
        str(config.chunk_bytes),
        "--repeats",
        str(config.repeats),
        "--parallelism",
        str(config.parallelism),
        "--align-bytes",
        str(config.align_bytes),
        "--output-json",
        config.output_json,
    ]
    for reader in config.readers:
        parameters.extend(["--reader", reader])
    parameters.extend(["--uc-volume-root", config.uc_volume_root])
    return parameters


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a Databricks runs/submit payload for an AWS g5/g6 storage-reader benchmark."
    )
    parser.add_argument("--workspace-dir", required=True, help="Cluster-local directory for synthetic shard artifacts.")
    parser.add_argument("--benchmark-output-json", required=True, help="Cluster-visible storage benchmark JSON output.")
    parser.add_argument("--runner-python-file", required=True, help="Cluster-visible runner script path or URI.")
    parser.add_argument("--benchmark-id", default=DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME)
    parser.add_argument("--chunk-count", type=int, default=64)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument(
        "--reader",
        action="append",
        choices=SUPPORTED_STORAGE_BENCHMARK_READERS,
        help="Reader to benchmark. Repeat for multiple readers; defaults to all readers.",
    )
    parser.add_argument("--align-bytes", type=int, default=4096)
    parser.add_argument("--uc-volume-root", required=True, help="Real UC Volume root, usually /Volumes/catalog/schema/volume.")
    parser.add_argument("--run-name", default=DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME)
    parser.add_argument("--task-key", default=DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY)
    parser.add_argument("--node-type-id", default=DEFAULT_AWS_G5_NODE_TYPE)
    parser.add_argument("--spark-version", default=DEFAULT_DATABRICKS_SPARK_VERSION)
    parser.add_argument("--data-security-mode", default=DEFAULT_DATABRICKS_DATA_SECURITY_MODE)
    parser.add_argument("--single-user-name", help="Required when --data-security-mode SINGLE_USER.")
    parser.add_argument("--wheel-uri", help="Optional cluster-visible wheel URI to install before the task.")
    parser.add_argument("--output-json", help="Write the runs/submit payload to this path instead of stdout.")
    parser.add_argument("--runner-script-output", help="Write the tiny storage-benchmark runner script to this path.")
    args = parser.parse_args(argv)

    try:
        config = DatabricksStorageBenchmarkJobConfig(
            workspace_dir=args.workspace_dir,
            output_json=args.benchmark_output_json,
            runner_python_file=args.runner_python_file,
            benchmark_id=args.benchmark_id,
            chunk_count=args.chunk_count,
            chunk_bytes=args.chunk_bytes,
            repeats=args.repeats,
            parallelism=args.parallelism,
            readers=tuple(args.reader) if args.reader else SUPPORTED_STORAGE_BENCHMARK_READERS,
            align_bytes=args.align_bytes,
            uc_volume_root=args.uc_volume_root,
            run_name=args.run_name,
            task_key=args.task_key,
            node_type_id=args.node_type_id,
            spark_version=args.spark_version,
            data_security_mode=args.data_security_mode,
            single_user_name=args.single_user_name,
            wheel_uri=args.wheel_uri,
        )
        if args.runner_script_output:
            write_databricks_storage_benchmark_runner_script(args.runner_script_output)
        payload = build_databricks_storage_benchmark_run_submit_payload(config)
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
