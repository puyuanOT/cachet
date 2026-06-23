"""Databricks runs/submit payload helpers for the Qwen3 SGLang live smoke."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_kv_cache._hardware_targets import (
    HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES,
    SUPPORTED_V1_HARDWARE_TARGETS,
    databricks_node_type_for_hardware_target,
    validate_aws_single_node_gpu_type,
    validate_aws_single_node_gpu_type_for_hardware_target,
    validate_v1_hardware_target,
)
from document_kv_cache.databricks_job import (
    DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE,
    DEFAULT_DATABRICKS_DATA_SECURITY_MODE,
    DEFAULT_DATABRICKS_SPARK_VERSION,
    DatabricksSingleNodeGPUClusterConfig,
    _spark_env_vars_from_cli,
    _validated_spark_env_vars,
    build_single_node_gpu_cluster,
)
from document_kv_cache.sglang_smoke import DEFAULT_LOCAL_ROOT, SERVER_HOST, SERVER_PORT


DEFAULT_DATABRICKS_SGLANG_SMOKE_RUN_NAME = "document-kv-sglang-smoke"
DEFAULT_DATABRICKS_SGLANG_SMOKE_TASK_KEY = "document_kv_sglang_smoke"
DEFAULT_DATABRICKS_SGLANG_SMOKE_PURPOSE = "document-kv-sglang-smoke"
SGLANG_SMOKE_RUNNER_SCRIPT = """from __future__ import annotations

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
    from document_kv_cache.sglang_smoke import main

    exit_code = main(remaining_args)
    if exit_code:
        raise SystemExit(exit_code)
"""

__all__ = [
    "DEFAULT_DATABRICKS_SGLANG_SMOKE_RUN_NAME",
    "DEFAULT_DATABRICKS_SGLANG_SMOKE_TASK_KEY",
    "DEFAULT_DATABRICKS_SGLANG_SMOKE_PURPOSE",
    "SGLANG_SMOKE_RUNNER_SCRIPT",
    "DatabricksSGLangSmokeJobConfig",
    "build_databricks_sglang_smoke_run_submit_payload",
    "write_databricks_sglang_smoke_run_submit_json",
    "write_databricks_sglang_smoke_runner_script",
    "main",
]


@dataclass(frozen=True, slots=True)
class DatabricksSGLangSmokeJobConfig:
    benchmark_id: str
    output_dir: str
    runner_python_file: str
    run_name: str = DEFAULT_DATABRICKS_SGLANG_SMOKE_RUN_NAME
    task_key: str = DEFAULT_DATABRICKS_SGLANG_SMOKE_TASK_KEY
    hardware_target: str | None = None
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
    context_length: int = 4096
    mem_fraction_static: float = 0.85
    stream: bool = True
    baseline_only: bool = False
    cache_prompt_text_mode: str = "runtime"
    handoff_json: str | None = None
    handoff_record_json: str | None = None
    payload_uri: str | None = None
    request_id: str | None = None
    hicache_page_store_uri: str | None = None
    hicache_ratio: float | None = None
    hicache_size_gb: int | None = None
    hicache_io_backend: str | None = None
    hicache_mem_layout: str | None = None
    hicache_storage_prefetch_policy: str | None = None
    hicache_write_policy: str | None = None
    availability: str = "ON_DEMAND"
    zone_id: str = "auto"
    custom_tags: Mapping[str, str] = field(default_factory=dict)
    spark_env_vars: Mapping[str, str] = field(default_factory=dict)

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
        object.__setattr__(
            self,
            "hardware_target",
            _resolve_hardware_target(self.hardware_target, self.node_type_id),
        )
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
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if not 0 < self.mem_fraction_static <= 1:
            raise ValueError("mem_fraction_static must be in (0, 1]")
        if type(self.stream) is not bool:
            raise ValueError("stream must be a boolean")
        if type(self.baseline_only) is not bool:
            raise ValueError("baseline_only must be a boolean")
        if self.cache_prompt_text_mode not in {"logical", "runtime"}:
            raise ValueError("cache_prompt_text_mode must be 'logical' or 'runtime'")
        if self.handoff_json and self.handoff_record_json:
            raise ValueError("SGLang smoke handoff params must use only one of handoff_json or handoff_record_json")
        if not self.baseline_only and self.handoff_json is None and self.handoff_record_json is None:
            raise ValueError("SGLang smoke cache arm requires handoff_json or handoff_record_json unless baseline_only=True")
        if self.handoff_record_json is not None:
            _json_object_from_text(self.handoff_record_json, "handoff_record_json")
        if self.hicache_size_gb is not None and self.hicache_size_gb < 0:
            raise ValueError("hicache_size_gb must be non-negative")
        object.__setattr__(self, "spark_env_vars", _validated_spark_env_vars(self.spark_env_vars))
        _DEFAULT_CLUSTER_CONFIG_FROM_SGLANG_SMOKE_JOB(self)


def build_databricks_sglang_smoke_run_submit_payload(config: DatabricksSGLangSmokeJobConfig) -> dict[str, Any]:
    cluster = build_single_node_gpu_cluster(_cluster_config_from_sglang_smoke_job(config))
    if config.spark_env_vars:
        cluster["spark_env_vars"] = dict(config.spark_env_vars)
    task: dict[str, Any] = {
        "task_key": config.task_key,
        "new_cluster": cluster,
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


def write_databricks_sglang_smoke_run_submit_json(
    config: DatabricksSGLangSmokeJobConfig,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(build_databricks_sglang_smoke_run_submit_payload(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_databricks_sglang_smoke_runner_script(path: str | Path) -> None:
    Path(path).write_text(SGLANG_SMOKE_RUNNER_SCRIPT, encoding="utf-8")


def _cluster_config_from_sglang_smoke_job(
    config: DatabricksSGLangSmokeJobConfig,
) -> DatabricksSingleNodeGPUClusterConfig:
    return DatabricksSingleNodeGPUClusterConfig(
        purpose=DEFAULT_DATABRICKS_SGLANG_SMOKE_PURPOSE,
        node_type_id=config.node_type_id,
        spark_version=config.spark_version,
        data_security_mode=config.data_security_mode,
        single_user_name=config.single_user_name,
        availability=config.availability,
        zone_id=config.zone_id,
        custom_tags=config.custom_tags,
    )


_DEFAULT_CLUSTER_CONFIG_FROM_SGLANG_SMOKE_JOB = _cluster_config_from_sglang_smoke_job


def _resolve_hardware_target(hardware_target: str | None, node_type_id: str) -> str:
    if hardware_target is not None:
        validate_v1_hardware_target(hardware_target)
        validate_aws_single_node_gpu_type_for_hardware_target(node_type_id, hardware_target)
        return hardware_target
    validate_aws_single_node_gpu_type(node_type_id)
    lowered = node_type_id.lower()
    for target, prefixes in HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES.items():
        if lowered.startswith(prefixes):
            return target
    raise ValueError(f"Unable to derive V1 hardware target from node_type_id {node_type_id!r}")


def _runner_parameters(config: DatabricksSGLangSmokeJobConfig) -> list[str]:
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
        "--context-length",
        str(config.context_length),
        "--mem-fraction-static",
        str(config.mem_fraction_static),
        "--hardware-target",
        str(config.hardware_target),
        "--cache-prompt-text-mode",
        config.cache_prompt_text_mode,
    ]
    if not config.stream:
        parameters.append("--no-stream")
    if config.baseline_only:
        parameters.append("--baseline-only")
    if config.handoff_json is not None:
        parameters.extend(["--handoff-json", config.handoff_json])
    if config.handoff_record_json is not None:
        parameters.extend(["--handoff-record-json", config.handoff_record_json])
    if config.payload_uri is not None:
        parameters.extend(["--payload-uri", config.payload_uri])
    if config.request_id is not None:
        parameters.extend(["--request-id", config.request_id])
    if config.hicache_page_store_uri is not None:
        parameters.extend(["--hicache-page-store-uri", config.hicache_page_store_uri])
    if config.hicache_ratio is not None:
        parameters.extend(["--hicache-ratio", str(config.hicache_ratio)])
    if config.hicache_size_gb is not None:
        parameters.extend(["--hicache-size-gb", str(config.hicache_size_gb)])
    if config.hicache_io_backend is not None:
        parameters.extend(["--hicache-io-backend", config.hicache_io_backend])
    if config.hicache_mem_layout is not None:
        parameters.extend(["--hicache-mem-layout", config.hicache_mem_layout])
    if config.hicache_storage_prefetch_policy is not None:
        parameters.extend(["--hicache-storage-prefetch-policy", config.hicache_storage_prefetch_policy])
    if config.hicache_write_policy is not None:
        parameters.extend(["--hicache-write-policy", config.hicache_write_policy])
    return parameters


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a Databricks runs/submit payload for a Qwen3/SGLang live Cachet smoke."
    )
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", required=True, help="Cluster-visible output directory for smoke artifacts.")
    parser.add_argument("--runner-python-file", required=True, help="Cluster-visible runner script path or URI.")
    parser.add_argument("--run-name", default=DEFAULT_DATABRICKS_SGLANG_SMOKE_RUN_NAME)
    parser.add_argument("--task-key", default=DEFAULT_DATABRICKS_SGLANG_SMOKE_TASK_KEY)
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
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--cache-prompt-text-mode", choices=("logical", "runtime"), default="runtime")
    parser.add_argument("--handoff-json")
    parser.add_argument("--handoff-record-json")
    parser.add_argument("--payload-uri")
    parser.add_argument("--request-id")
    parser.add_argument("--hicache-page-store-uri")
    parser.add_argument("--hicache-ratio", type=float)
    parser.add_argument("--hicache-size-gb", type=int)
    parser.add_argument("--hicache-io-backend")
    parser.add_argument("--hicache-mem-layout")
    parser.add_argument("--hicache-storage-prefetch-policy")
    parser.add_argument("--hicache-write-policy")
    parser.add_argument(
        "--spark-env-var",
        action="append",
        default=None,
        help="Non-secret Databricks cluster spark_env_vars entry for runtime configuration, in KEY=VALUE form.",
    )
    parser.add_argument("--output-json", help="Write the runs/submit payload to this path instead of stdout.")
    parser.add_argument("--runner-script-output", help="Write the tiny SGLang smoke runner script to this path.")
    args = parser.parse_args(argv)

    try:
        config = DatabricksSGLangSmokeJobConfig(
            benchmark_id=args.benchmark_id,
            output_dir=args.output_dir,
            runner_python_file=args.runner_python_file,
            run_name=args.run_name,
            task_key=args.task_key,
            node_type_id=databricks_node_type_for_hardware_target(args.hardware_target, args.node_type_id),
            hardware_target=args.hardware_target,
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
            context_length=args.context_length,
            mem_fraction_static=args.mem_fraction_static,
            stream=not args.no_stream,
            baseline_only=args.baseline_only,
            cache_prompt_text_mode=args.cache_prompt_text_mode,
            handoff_json=args.handoff_json,
            handoff_record_json=args.handoff_record_json,
            payload_uri=args.payload_uri,
            request_id=args.request_id,
            hicache_page_store_uri=args.hicache_page_store_uri,
            hicache_ratio=args.hicache_ratio,
            hicache_size_gb=args.hicache_size_gb,
            hicache_io_backend=args.hicache_io_backend,
            hicache_mem_layout=args.hicache_mem_layout,
            hicache_storage_prefetch_policy=args.hicache_storage_prefetch_policy,
            hicache_write_policy=args.hicache_write_policy,
            spark_env_vars=_spark_env_vars_from_cli(args.spark_env_var or ()),
        )
        if args.runner_script_output:
            write_databricks_sglang_smoke_runner_script(args.runner_script_output)
        payload = build_databricks_sglang_smoke_run_submit_payload(config)
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


def _json_object_from_text(value: str, field_name: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must decode to a JSON object") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{field_name} must decode to a JSON object")
    return decoded


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
