"""Databricks-friendly vLLM smoke benchmark for the V1 Qwen3 path."""

from __future__ import annotations

import argparse
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

from document_kv_cache.benchmarks import DEFAULT_HARDWARE_TARGET
from document_kv_cache.model_profiles import QWEN3_4B_INSTRUCT_HF_MODEL_ID
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

HF_MODEL_ID = QWEN3_4B_INSTRUCT_HF_MODEL_ID
SERVED_MODEL_NAME = "qwen3:4b-instruct"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
SMOKE_DATASETS = ("biography", "hotpotqa", "musique", "niah")
DEFAULT_LOCAL_ROOT = Path("/local_disk0")

__all__ = [
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
    "run_vllm_smoke_benchmark",
    "build_metadata",
    "dependency_constraints",
    "build_vllm_server_args",
    "build_benchmark_runner_args",
    "benchmark_dataset_paths",
    "write_smoke_datasets",
    "smoke_dataset_records",
    "parse_dataset_specs",
    "dataset_args",
    "parse_args",
    "main",
]


@dataclass(frozen=True)
class VLLMSmokeBenchmarkConfig:
    """Runtime configuration for a one-node Databricks vLLM smoke run."""

    benchmark_id: str
    output_dir: Path
    max_tokens: int = 32
    timeout_seconds: float = 240.0
    import_probe_timeout_seconds: float = 180.0
    server_start_timeout_seconds: float = 480.0
    local_root: Path = DEFAULT_LOCAL_ROOT
    server_host: str = SERVER_HOST
    server_port: int = SERVER_PORT
    client_host: str = SERVER_HOST
    max_model_len: int = 4096
    max_num_seqs: int = 2
    gpu_memory_utilization: float = 0.85
    dataset_specs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if self.output_dir is None:
            raise ValueError("output_dir must be provided")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.import_probe_timeout_seconds <= 0:
            raise ValueError("import_probe_timeout_seconds must be positive")
        if self.server_start_timeout_seconds <= 0:
            raise ValueError("server_start_timeout_seconds must be positive")
        if self.local_root is None:
            raise ValueError("local_root must be provided")
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

    @property
    def local_dir(self) -> Path:
        return self.local_root / f"document-kv-vllm-smoke-{self.benchmark_id}"

    @property
    def hf_cache_dir(self) -> Path:
        return self.local_root / "hf-cache"

    @property
    def server_base_url(self) -> str:
        return f"http://{self.client_host}:{self.server_port}"

    @property
    def venv_dir(self) -> Path:
        return self.local_dir / "vllm-venv"

    @property
    def venv_python(self) -> Path:
        return self.venv_dir / "bin" / "python"

    @property
    def server_log_path(self) -> Path:
        return self.local_dir / "vllm-server.log"

    @property
    def server_log_copy_path(self) -> Path:
        return self.output_dir / "vllm-server.log"

    @property
    def benchmark_output_path(self) -> Path:
        return self.output_dir / "v1-benchmark.json"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"

    @property
    def import_probe_path(self) -> Path:
        return self.output_dir / "vllm-import-probe.json"


def run_vllm_smoke_benchmark(config: VLLMSmokeBenchmarkConfig) -> None:
    """Create an isolated vLLM env, start Qwen3, and run the V1 smoke suite."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.local_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(config.hf_cache_dir)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    metadata = build_metadata(config)
    write_json(config.metadata_path, metadata)

    create_venv(config.venv_dir)
    install_vllm(config.venv_python)
    metadata.update(installed_versions(config.venv_python))
    write_json(config.metadata_path, metadata)
    probe_vllm_import(
        config.venv_python,
        config.import_probe_path,
        timeout_seconds=config.import_probe_timeout_seconds,
        env=server_env(config),
    )

    dataset_paths = benchmark_dataset_paths(config)
    metadata["vllm_server_local_log"] = str(config.server_log_path)
    metadata["vllm_server_log"] = str(config.server_log_copy_path)
    write_json(config.metadata_path, metadata)

    server = start_vllm_server(config, config.venv_python, config.server_log_path)
    try:
        wait_for_server(server, config.server_log_path, config, timeout_seconds=config.server_start_timeout_seconds)
        copy_file_if_exists(config.server_log_path, config.server_log_copy_path)
        run_benchmark_runner(config, dataset_paths)
    finally:
        terminate_process(server)
        copy_file_if_exists(config.server_log_path, config.server_log_copy_path)


def build_metadata(config: VLLMSmokeBenchmarkConfig) -> dict[str, object]:
    return {
        "benchmark_id": config.benchmark_id,
        "hf_model_id": HF_MODEL_ID,
        "served_model_name": SERVED_MODEL_NAME,
        "vllm_version_requested": VLLM_VERSION,
        "server_bind_host": config.server_host,
        "server_client_host": config.client_host,
        "server_base_url": config.server_base_url,
        "hf_home": str(config.hf_cache_dir),
        "vllm_python": str(config.venv_python),
        "dependency_constraints": dependency_constraints(),
        "dataset_source": "prepared" if config.dataset_specs else "smoke",
        "dataset_specs": list(config.dataset_specs),
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "gpu_memory_utilization": config.gpu_memory_utilization,
    }


def dependency_constraints() -> list[str]:
    return list(VLLM_SERVING_ENVIRONMENT_PROFILE.dependency_constraints)


def installed_versions(python_executable: Path) -> dict[str, str]:
    return {
        "vllm_version_installed": installed_package_version(python_executable, "vllm"),
        "transformers_version_installed": installed_package_version(python_executable, "transformers"),
        "torch_version_installed": installed_package_version(python_executable, "torch"),
    }


def run(argv: list[str]) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def run_benchmark_runner(config: VLLMSmokeBenchmarkConfig, dataset_paths: dict[str, Path]) -> None:
    try:
        run(build_benchmark_runner_args(config, dataset_paths))
    except subprocess.CalledProcessError as exc:
        summary = benchmark_failure_summary(config.benchmark_output_path)
        raise RuntimeError(
            f"vLLM benchmark runner failed with exit code {exc.returncode}; {summary}"
        ) from exc


def benchmark_failure_summary(output_path: Path, *, limit: int = 3) -> str:
    if not output_path.exists():
        return f"benchmark output {output_path} was not written"
    try:
        record = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"benchmark output {output_path} could not be read: {exc}"

    measurements = record.get("measurements")
    if not isinstance(measurements, list):
        return f"benchmark output {output_path} did not include measurements"
    errors = [
        _benchmark_error_summary(measurement)
        for measurement in measurements
        if isinstance(measurement, dict) and measurement.get("error")
    ]
    if not errors:
        return f"benchmark output {output_path} did not include row errors"

    issue_count = len(errors)
    shown = "; ".join(errors[:limit])
    if issue_count > limit:
        shown = f"{shown}; ... {issue_count - limit} more"
    return f"benchmark output had {issue_count}/{len(measurements)} errored measurements: {shown}"


def _benchmark_error_summary(measurement: dict[str, object], *, max_chars: int = 400) -> str:
    dataset = measurement.get("dataset") or "unknown-dataset"
    arm_id = measurement.get("arm_id") or "unknown-arm"
    error = str(measurement.get("error") or "unknown error")
    if len(error) > max_chars:
        error = error[: max_chars - 3] + "..."
    return f"{dataset}/{arm_id}: {error}"


def create_venv(venv_dir: Path) -> None:
    if venv_python(venv_dir).exists():
        return
    try:
        run([sys.executable, "-m", "venv", str(venv_dir)])
    except subprocess.CalledProcessError:
        run([sys.executable, "-m", "pip", "install", "virtualenv"])
        run([sys.executable, "-m", "virtualenv", str(venv_dir)])


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def install_vllm(python_executable: Path) -> None:
    run([str(python_executable), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            *dependency_constraints(),
        ]
    )


def installed_package_version(python_executable: Path, package_name: str) -> str:
    completed = subprocess.run(
        [str(python_executable), "-m", "pip", "show", package_name],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def probe_vllm_import(
    python_executable: Path,
    output_path: Path,
    *,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
) -> None:
    code = """
import importlib.metadata as md
import json
import torch
import vllm
import vllm.entrypoints.openai.api_server

payload = {
    "ok": True,
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "vllm_version": md.version("vllm"),
}
if torch.cuda.is_available():
    payload["cuda_device_name"] = torch.cuda.get_device_name(0)
print(json.dumps(payload, sort_keys=True), flush=True)
"""
    argv = [str(python_executable), "-c", code]
    print("+", " ".join([argv[0], "-c", "<vllm import probe>"]), flush=True)
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env or os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        record = {
            "ok": False,
            "error_type": "TimeoutExpired",
            "error": f"vLLM import probe timed out after {timeout_seconds:.1f}s",
            "stdout_tail": tail_text(exc.stdout),
            "stderr_tail": tail_text(exc.stderr),
        }
        write_json(output_path, record)
        raise RuntimeError(record["error"]) from exc

    record = {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_tail": tail_text(completed.stdout),
        "stderr_tail": tail_text(completed.stderr),
    }
    if completed.returncode == 0:
        record.update(last_json_object(completed.stdout))
    write_json(output_path, record)
    if completed.returncode != 0:
        raise RuntimeError(f"vLLM import probe failed with return code {completed.returncode}")


def last_json_object(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def write_smoke_datasets(local_dir: Path) -> dict[str, Path]:
    paths = {}
    for dataset, record in smoke_dataset_records().items():
        path = local_dir / f"{dataset}.jsonl"
        path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        paths[dataset] = path
    return paths


def benchmark_dataset_paths(config: VLLMSmokeBenchmarkConfig) -> dict[str, Path]:
    if config.dataset_specs:
        return parse_dataset_specs(config.dataset_specs)
    return write_smoke_datasets(config.local_dir)


def parse_dataset_specs(dataset_specs: tuple[str, ...]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in dataset_specs:
        dataset, separator, raw_path = spec.partition("=")
        if not separator or not dataset or not raw_path:
            raise ValueError("dataset specs must use DATASET=JSONL_PATH syntax")
        if dataset not in SMOKE_DATASETS:
            raise ValueError(f"Unsupported V1 smoke dataset {dataset!r}")
        if dataset in paths:
            raise ValueError(f"duplicate dataset spec for {dataset!r}")
        paths[dataset] = Path(raw_path)
    missing = set(SMOKE_DATASETS).difference(paths)
    if missing:
        raise ValueError(f"dataset specs missing required V1 datasets: {sorted(missing)}")
    return {dataset: paths[dataset] for dataset in SMOKE_DATASETS}


def smoke_dataset_records() -> dict[str, dict[str, object]]:
    return {
        "biography": {
            "example_id": "biography-smoke-1",
            "dataset": "biography",
            "query": "Which person is described in the biography?",
            "expected_answer": "Katherine Johnson",
            "documents": [
                {
                    "document_id": "katherine-johnson",
                    "title": "Katherine Johnson",
                    "text": (
                        "Katherine Johnson was a NASA mathematician whose orbital mechanics calculations "
                        "supported early crewed spaceflight missions."
                    ),
                }
            ],
        },
        "hotpotqa": {
            "example_id": "hotpotqa-smoke-1",
            "dataset": "hotpotqa",
            "query": "The landmark discussed in the first document is located in which city?",
            "expected_answer": "Paris",
            "documents": [
                {
                    "document_id": "eiffel-tower",
                    "title": "Eiffel Tower",
                    "text": "The Eiffel Tower is a wrought-iron landmark on the Champ de Mars.",
                },
                {
                    "document_id": "paris",
                    "title": "Paris",
                    "text": "The Champ de Mars is a large public greenspace in Paris, France.",
                },
            ],
        },
        "musique": {
            "example_id": "musique-smoke-1",
            "dataset": "musique",
            "query": "Who is the mathematician connected to the engine described by Charles Babbage?",
            "expected_answer": "Ada Lovelace",
            "documents": [
                {
                    "document_id": "analytical-engine",
                    "title": "Analytical Engine",
                    "text": "Charles Babbage designed the Analytical Engine as a proposed mechanical computer.",
                },
                {
                    "document_id": "ada-lovelace",
                    "title": "Ada Lovelace",
                    "text": "Ada Lovelace wrote notes about the Analytical Engine and is known for early computing work.",
                },
            ],
        },
        "niah": {
            "example_id": "niah-smoke-1",
            "dataset": "niah",
            "query": "What is the hidden target phrase?",
            "expected_answer": "cerulean lantern",
            "documents": [
                {
                    "document_id": "haystack",
                    "title": "Needle Haystack",
                    "text": (
                        "Most of this context is filler for the retrieval smoke test. "
                        "The hidden target phrase is cerulean lantern. "
                        "Only the exact hidden phrase should be returned."
                    ),
                }
            ],
        },
    }


def dataset_args(dataset_paths: dict[str, Path]) -> list[str]:
    args: list[str] = []
    for dataset in SMOKE_DATASETS:
        args.extend(["--dataset", f"{dataset}={dataset_paths[dataset]}"])
    return args


def build_vllm_server_args(config: VLLMSmokeBenchmarkConfig, python_executable: Path) -> list[str]:
    return [
        str(python_executable),
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        HF_MODEL_ID,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        config.server_host,
        "--port",
        str(config.server_port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]


def start_vllm_server(
    config: VLLMSmokeBenchmarkConfig, python_executable: Path, log_path: Path
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = build_vllm_server_args(config, python_executable)
    print("+", " ".join(argv), flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        return subprocess.Popen(argv, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=server_env(config))


def wait_for_server(
    server: subprocess.Popen,
    log_path: Path,
    config: VLLMSmokeBenchmarkConfig,
    *,
    timeout_seconds: float = 900.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_url = f"{config.server_base_url}/health"
    models_url = f"{config.server_base_url}/v1/models"
    last_model_error = ""
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM server exited with {server.returncode}; log tail:\n{tail(log_path)}")
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    model_ids = fetch_served_model_ids(models_url)
                    if SERVED_MODEL_NAME in model_ids:
                        return
                    last_model_error = f"health OK but served models were {sorted(model_ids)!r}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError) as exc:
            last_model_error = str(exc)
            pass
        time.sleep(5)
    raise TimeoutError(
        f"Timed out waiting for vLLM model {SERVED_MODEL_NAME!r} at {config.server_base_url}; "
        f"last readiness error: {last_model_error}; log tail:\n{tail(log_path)}"
    )


def fetch_served_model_ids(models_url: str) -> set[str]:
    with urllib.request.urlopen(models_url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        item["id"]
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def build_benchmark_runner_args(
    config: VLLMSmokeBenchmarkConfig, dataset_paths: dict[str, Path]
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "document_kv_cache.benchmark_runner",
        "--suite-id",
        config.benchmark_id,
        "--base-url",
        config.server_base_url,
        "--model-id",
        SERVED_MODEL_NAME,
        "--hardware-target",
        DEFAULT_HARDWARE_TARGET,
        "--max-tokens",
        str(config.max_tokens),
        "--timeout-seconds",
        str(config.timeout_seconds),
        "--server-usage",
        "--output-json",
        str(config.benchmark_output_path),
        *dataset_args(dataset_paths),
    ]


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_file_if_exists(source_path: Path, target_path: Path) -> None:
    if source_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)


def server_env(config: VLLMSmokeBenchmarkConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["HF_HOME"] = str(config.hf_cache_dir)
    return env


def tail_text(text: str | bytes | None, *, max_chars: int = 12000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text[-max_chars:]


def tail(path: Path, *, lines: int = 120) -> str:
    if not path.exists():
        return "<missing log>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def parse_args(argv: list[str] | None = None) -> VLLMSmokeBenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run a Qwen3/vLLM V1 benchmark smoke on Databricks g6/L4.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", required=True)
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
    args = parser.parse_args(argv)
    return VLLMSmokeBenchmarkConfig(
        benchmark_id=args.benchmark_id,
        output_dir=Path(args.output_dir),
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        import_probe_timeout_seconds=args.import_probe_timeout_seconds,
        server_start_timeout_seconds=args.server_start_timeout_seconds,
        local_root=Path(args.local_root),
        server_host=args.server_host,
        server_port=args.server_port,
        client_host=args.client_host,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dataset_specs=tuple(args.dataset or ()),
    )


def main(argv: list[str] | None = None) -> int:
    run_vllm_smoke_benchmark(parse_args(argv))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
