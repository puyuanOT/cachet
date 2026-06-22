"""Databricks runs/submit payload helpers for the Qwen3 vLLM smoke benchmark."""

from __future__ import annotations

import argparse
import json
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
from document_kv_cache.vllm_smoke import (
    DEFAULT_LOCAL_ROOT,
    SERVER_HOST,
    SERVER_PORT,
    parse_dataset_specs,
)


DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME = "document-kv-vllm-smoke"
DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY = "document_kv_vllm_smoke"
DEFAULT_DATABRICKS_VLLM_SMOKE_PURPOSE = "document-kv-vllm-smoke"
VLLM_SMOKE_RUNNER_SCRIPT = """from __future__ import annotations

import argparse
import os
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
        package_wheel_path = _cluster_file_path(args.package_wheel_uri)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_wheel_path]
        )
        os.environ["DOCUMENT_KV_PACKAGE_INSTALL_SPEC"] = package_wheel_path
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
    node_type_id: str = DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE
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
    max_model_len: int = 4096
    max_num_seqs: int = 2
    gpu_memory_utilization: float = 0.85
    dataset_specs: tuple[str, ...] = ()
    benchmark_handoff_generator_factory: str | None = None
    benchmark_handoff_output_dir: str | None = None
    benchmark_handoff_dtype: str = "bfloat16"
    benchmark_handoff_align_bytes: int = 4096
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
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        object.__setattr__(self, "dataset_specs", tuple(self.dataset_specs))
        if self.dataset_specs:
            parse_dataset_specs(self.dataset_specs)
        if self.benchmark_handoff_generator_factory is not None:
            if not self.benchmark_handoff_generator_factory.strip():
                raise ValueError("benchmark_handoff_generator_factory must be non-empty when provided")
            if not self.dataset_specs:
                raise ValueError("benchmark_handoff_generator_factory requires prepared dataset specs")
        if self.benchmark_handoff_output_dir is not None and not self.benchmark_handoff_output_dir:
            raise ValueError("benchmark_handoff_output_dir must be non-empty when provided")
        if self.benchmark_handoff_output_dir is not None and self.benchmark_handoff_generator_factory is None:
            raise ValueError("benchmark_handoff_output_dir requires benchmark_handoff_generator_factory")
        if not self.benchmark_handoff_dtype:
            raise ValueError("benchmark_handoff_dtype must be non-empty")
        if type(self.benchmark_handoff_align_bytes) is not int or self.benchmark_handoff_align_bytes <= 0:
            raise ValueError("benchmark_handoff_align_bytes must be a positive integer")
        _DEFAULT_CLUSTER_CONFIG_FROM_VLLM_SMOKE_JOB(self)


def build_databricks_vllm_smoke_run_submit_payload(config: DatabricksVLLMSmokeJobConfig) -> dict[str, Any]:
    task: dict[str, Any] = {
        "task_key": config.task_key,
        "new_cluster": build_single_node_gpu_cluster(_cluster_config_from_vllm_smoke_job(config)),
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
    ]
    for dataset_spec in config.dataset_specs:
        parameters.extend(["--dataset", dataset_spec])
    if config.benchmark_handoff_generator_factory is not None:
        parameters.extend(
            [
                "--benchmark-handoff-generator-factory",
                config.benchmark_handoff_generator_factory,
                "--benchmark-handoff-dtype",
                config.benchmark_handoff_dtype,
                "--benchmark-handoff-align-bytes",
                str(config.benchmark_handoff_align_bytes),
            ]
        )
        if config.benchmark_handoff_output_dir is not None:
            parameters.extend(["--benchmark-handoff-output-dir", config.benchmark_handoff_output_dir])
    return parameters


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a Databricks runs/submit payload for a V1 AWS single-node GPU vLLM smoke."
    )
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", required=True, help="Cluster-visible output directory for smoke artifacts.")
    parser.add_argument("--runner-python-file", required=True, help="Cluster-visible runner script path or URI.")
    parser.add_argument("--run-name", default=DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME)
    parser.add_argument("--task-key", default=DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY)
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
        "--dataset",
        action="append",
        default=None,
        help="Prepared V1 benchmark dataset in DATASET=JSONL_PATH form. Repeat for all four V1 datasets.",
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
            node_type_id=databricks_node_type_for_hardware_target(args.hardware_target, args.node_type_id),
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
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dataset_specs=tuple(args.dataset or ()),
            benchmark_handoff_generator_factory=args.benchmark_handoff_generator_factory,
            benchmark_handoff_output_dir=args.benchmark_handoff_output_dir,
            benchmark_handoff_dtype=args.benchmark_handoff_dtype,
            benchmark_handoff_align_bytes=args.benchmark_handoff_align_bytes,
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
