"""Compatibility wrapper for :mod:`document_kv_cache.vllm_smoke`."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Iterator

from document_kv_cache._reexport import reexport_public
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

import document_kv_cache.vllm_smoke as _document_module

__all__ = reexport_public(
    "document_kv_cache.vllm_smoke",
    (
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
        "VLLMSmokeBenchmarkConfig",
        "build_metadata",
        "dependency_constraints",
        "build_vllm_server_args",
        "build_benchmark_runner_args",
        "write_smoke_datasets",
        "smoke_dataset_records",
        "dataset_args",
        "parse_args",
    ),
    globals(),
)

__all__ += [
    "run_vllm_smoke_benchmark",
    "main",
    "argparse",
    "dataclass",
    "json",
    "os",
    "Path",
    "shutil",
    "signal",
    "subprocess",
    "sys",
    "time",
    "urllib",
    "VLLM_SERVING_ENVIRONMENT_PROFILE",
    "create_venv",
    "venv_python",
    "install_vllm",
    "installed_versions",
    "installed_package_version",
    "probe_vllm_import",
    "last_json_object",
    "start_vllm_server",
    "wait_for_server",
    "fetch_served_model_ids",
    "terminate_process",
    "write_json",
    "copy_file_if_exists",
    "server_env",
    "tail_text",
    "tail",
    "run",
    "DEFAULT_LOCAL_ROOT",
    "SERVER_HOST",
    "SERVER_PORT",
]

DEFAULT_LOCAL_ROOT = _document_module.DEFAULT_LOCAL_ROOT
SERVER_HOST = _document_module.SERVER_HOST
SERVER_PORT = _document_module.SERVER_PORT


def build_metadata(config: VLLMSmokeBenchmarkConfig) -> dict[str, object]:
    return _call_document_function("build_metadata", config)


def dependency_constraints() -> list[str]:
    return _call_document_function("dependency_constraints")


def build_vllm_server_args(config: VLLMSmokeBenchmarkConfig, python_executable: Path) -> list[str]:
    return _call_document_function("build_vllm_server_args", config, python_executable)


def build_benchmark_runner_args(config: VLLMSmokeBenchmarkConfig, dataset_paths: dict[str, Path]) -> list[str]:
    return _call_document_function("build_benchmark_runner_args", config, dataset_paths)


def write_smoke_datasets(local_dir: Path) -> dict[str, Path]:
    return _call_document_function("write_smoke_datasets", local_dir)


def smoke_dataset_records() -> dict[str, dict[str, object]]:
    return _call_document_function("smoke_dataset_records")


def dataset_args(dataset_paths: dict[str, Path]) -> list[str]:
    return _call_document_function("dataset_args", dataset_paths)


def parse_args(argv: list[str] | None = None) -> VLLMSmokeBenchmarkConfig:
    return _call_document_function("parse_args", argv)


def create_venv(venv_dir: Path) -> None:
    return _call_document_function("create_venv", venv_dir)


def venv_python(venv_dir: Path) -> Path:
    return _call_document_function("venv_python", venv_dir)


def install_vllm(python_executable: Path) -> None:
    return _call_document_function("install_vllm", python_executable)


def installed_versions(python_executable: Path) -> dict[str, str]:
    return _call_document_function("installed_versions", python_executable)


def installed_package_version(python_executable: Path, package_name: str) -> str:
    return _call_document_function("installed_package_version", python_executable, package_name)


def probe_vllm_import(
    python_executable: Path,
    output_path: Path,
    *,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
) -> None:
    return _call_document_function(
        "probe_vllm_import",
        python_executable,
        output_path,
        timeout_seconds=timeout_seconds,
        env=env,
    )


def last_json_object(text: str) -> dict[str, object]:
    return _call_document_function("last_json_object", text)


def start_vllm_server(
    config: VLLMSmokeBenchmarkConfig, python_executable: Path, log_path: Path
) -> subprocess.Popen:
    return _call_document_function("start_vllm_server", config, python_executable, log_path)


def wait_for_server(
    server: subprocess.Popen,
    log_path: Path,
    config: VLLMSmokeBenchmarkConfig,
    *,
    timeout_seconds: float = 900.0,
) -> None:
    return _call_document_function(
        "wait_for_server",
        server,
        log_path,
        config,
        timeout_seconds=timeout_seconds,
    )


def fetch_served_model_ids(models_url: str) -> set[str]:
    return _call_document_function("fetch_served_model_ids", models_url)


def terminate_process(process: subprocess.Popen) -> None:
    return _call_document_function("terminate_process", process)


def write_json(path: Path, payload: dict[str, object]) -> None:
    return _call_document_function("write_json", path, payload)


def copy_file_if_exists(source_path: Path, target_path: Path) -> None:
    return _call_document_function("copy_file_if_exists", source_path, target_path)


def server_env(config: VLLMSmokeBenchmarkConfig) -> dict[str, str]:
    return _call_document_function("server_env", config)


def tail_text(text: str | bytes | None, *, max_chars: int = 12000) -> str:
    return _call_document_function("tail_text", text, max_chars=max_chars)


def tail(path: Path, *, lines: int = 120) -> str:
    return _call_document_function("tail", path, lines=lines)


def run(argv: list[str]) -> None:
    return _call_document_function("run", argv)


def run_vllm_smoke_benchmark(config: VLLMSmokeBenchmarkConfig) -> None:
    with _patched_document_globals(include_runner=False):
        _document_module.run_vllm_smoke_benchmark(config)


_LEGACY_RUN_VLLM_SMOKE_BENCHMARK = run_vllm_smoke_benchmark


def main(argv: list[str] | None = None) -> int:
    runner_was_patched = globals()["run_vllm_smoke_benchmark"] is not _LEGACY_RUN_VLLM_SMOKE_BENCHMARK
    with _patched_document_globals(include_runner=runner_was_patched):
        return _document_module.main(argv)


_DEFAULT_COMPAT_FUNCTIONS = {
    "build_metadata": build_metadata,
    "dependency_constraints": dependency_constraints,
    "build_vllm_server_args": build_vllm_server_args,
    "build_benchmark_runner_args": build_benchmark_runner_args,
    "write_smoke_datasets": write_smoke_datasets,
    "smoke_dataset_records": smoke_dataset_records,
    "dataset_args": dataset_args,
    "parse_args": parse_args,
    "create_venv": create_venv,
    "venv_python": venv_python,
    "install_vllm": install_vllm,
    "installed_versions": installed_versions,
    "installed_package_version": installed_package_version,
    "probe_vllm_import": probe_vllm_import,
    "last_json_object": last_json_object,
    "start_vllm_server": start_vllm_server,
    "wait_for_server": wait_for_server,
    "fetch_served_model_ids": fetch_served_model_ids,
    "terminate_process": terminate_process,
    "write_json": write_json,
    "copy_file_if_exists": copy_file_if_exists,
    "server_env": server_env,
    "tail_text": tail_text,
    "tail": tail,
    "run": run,
    "run_vllm_smoke_benchmark": run_vllm_smoke_benchmark,
}


def _call_document_function(name: str, *args, **kwargs):
    with _patched_document_globals(include_runner=False, excluded_names={name}):
        return getattr(_document_module, name)(*args, **kwargs)


@contextmanager
def _patched_document_globals(*, include_runner: bool, excluded_names: set[str] | None = None) -> Iterator[None]:
    excluded_names = {"main", *(excluded_names or set())}
    if not include_runner:
        excluded_names.add("run_vllm_smoke_benchmark")
    patch_names = tuple(name for name in __all__ if name not in excluded_names)
    active_patch_names = tuple(
        name
        for name in patch_names
        if _DEFAULT_COMPAT_FUNCTIONS.get(name) is not globals().get(name)
    )
    previous = {name: getattr(_document_module, name) for name in active_patch_names if hasattr(_document_module, name)}
    missing = [name for name in active_patch_names if not hasattr(_document_module, name)]
    for name in active_patch_names:
        if hasattr(_document_module, name):
            setattr(_document_module, name, globals()[name])
    try:
        yield
    finally:
        for name in active_patch_names:
            if name in previous:
                setattr(_document_module, name, previous[name])
        for name in missing:
            if hasattr(_document_module, name):
                delattr(_document_module, name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


del reexport_public
