"""Reviewed runtime dispatcher for vLLM 0.27.1 GPU qualification sentinels.

Imports of torch, Triton, and vLLM live in the isolated worker module so local
payload rendering and CPU unit tests do not require the GPU runtime.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from document_kv_cache.gpu_qualification import canonical_gpu_qualification_json
from document_kv_cache.serving_env import (
    VLLM_PATCHED_WHEEL_SHA256_ENV,
    VLLM_PATCHED_WHEEL_URI_ENV,
    vllm_runtime_lock_path,
)
from document_kv_cache.vllm_smoke import (
    create_venv,
    install_document_kv_package,
    install_vllm,
    verify_vllm_runtime_lock_installation,
)


_RUNTIME_LOCK_ATTESTATION_ENV = "CACHET_GPU_QUALIFICATION_RUNTIME_LOCK_ATTESTATION"


def run_gpu_qualification_sentinel(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
    work_dir: Path,
) -> Mapping[str, Any]:
    """Install the exact runtime and execute one package-owned GPU worker."""

    job_id = _required_string(planned_job, "job_id")
    runtime_dir = work_dir / "runtime"
    runtime_python = runtime_dir / "bin" / "python"
    patched_wheel = artifact_paths["patched_vllm_wheel_sha256"]
    package_wheel = artifact_paths["package_wheel_sha256"]
    expected_patched_sha256 = _artifact_pin(
        plan_record, "patched_vllm_wheel_sha256"
    )
    expected_runtime_lock_sha256 = _artifact_pin(
        plan_record, "runtime_lock_sha256"
    )
    supplied_runtime_lock = artifact_paths["runtime_lock_sha256"]
    packaged_runtime_lock = vllm_runtime_lock_path()
    for label, path in (
        ("supplied", supplied_runtime_lock),
        ("packaged", packaged_runtime_lock),
    ):
        if _file_sha256(path) != expected_runtime_lock_sha256:
            raise RuntimeError(
                f"{label} runtime lock does not match the qualification plan"
            )
    os.environ[VLLM_PATCHED_WHEEL_URI_ENV] = str(patched_wheel)
    os.environ[VLLM_PATCHED_WHEEL_SHA256_ENV] = expected_patched_sha256

    create_venv(runtime_dir)
    install_vllm(runtime_python)
    install_document_kv_package(runtime_python, str(package_wheel))
    subprocess.run(
        [str(runtime_python), "-m", "pip", "check"],
        check=True,
        timeout=300,
    )

    environment = dict(os.environ)
    for ambient_path_variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(ambient_path_variable, None)
    environment.update(
        {
            "HF_HOME": "/local_disk0/cachet-vllm-0271-hf",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONSAFEPATH": "1",
            "TOKENIZERS_PARALLELISM": "false",
            VLLM_PATCHED_WHEEL_URI_ENV: str(patched_wheel),
            VLLM_PATCHED_WHEEL_SHA256_ENV: expected_patched_sha256,
        }
    )
    _verify_input_bundle_in_isolated_runtime(
        runtime_python,
        artifact_paths["input_bundle_sha256"],
        expected_sha256=_artifact_pin(plan_record, "input_bundle_sha256"),
        environment=environment,
    )
    runtime_lock_attestation = verify_vllm_runtime_lock_installation(runtime_python)
    environment[_RUNTIME_LOCK_ATTESTATION_ENV] = json.dumps(
        runtime_lock_attestation,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    worker_dir = work_dir / "worker"
    worker_dir.mkdir()
    job_path = worker_dir / "planned-job.json"
    job_path.write_text(
        canonical_gpu_qualification_json(planned_job) + "\n",
        encoding="utf-8",
    )
    plan_path = worker_dir / "plan.json"
    plan_path.write_text(
        canonical_gpu_qualification_json(plan_record) + "\n",
        encoding="utf-8",
    )
    _make_site_packages_read_only(runtime_python)

    worker_output = worker_dir / "measurements.json"
    completed = subprocess.run(
        [
            str(runtime_python),
            "-m",
            "document_kv_cache._gpu_qualification_sentinel_worker",
            "--plan-json",
            str(plan_path),
            "--job-json",
            str(job_path),
            "--input-bundle",
            str(artifact_paths["input_bundle_sha256"]),
            "--work-dir",
            str(worker_dir / "runtime-work"),
            "--output-json",
            str(worker_output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=14_000,
        env=environment,
        cwd=worker_dir,
    )
    if not worker_output.is_file() or worker_output.is_symlink():
        raise RuntimeError(
            f"GPU sentinel {job_id!r} did not write its measurement record; "
            f"stdout={completed.stdout[-2000:]!r}, stderr={completed.stderr[-4000:]!r}"
        )
    measurements = json.loads(worker_output.read_text(encoding="utf-8"))
    if not isinstance(measurements, dict):
        raise RuntimeError("GPU sentinel measurement output must be an object")
    return measurements


def _verify_input_bundle_in_isolated_runtime(
    runtime_python: Path,
    input_bundle: Path,
    *,
    expected_sha256: str,
    environment: Mapping[str, str],
) -> str:
    """Run the tokenizer-aware closure verifier only in the locked runtime."""

    verifier = """import sys
from document_kv_cache.main_latency_inputs import verify_main_latency_inputs

verified = verify_main_latency_inputs(sys.argv[1], examples_per_dataset=32)
expected = sys.argv[2]
if verified.bundle_sha256 != expected:
    raise RuntimeError(
        f"input bundle closure mismatch: {verified.bundle_sha256} != {expected}"
    )
print(verified.bundle_sha256)
"""
    completed = subprocess.run(
        [str(runtime_python), "-c", verifier, str(input_bundle), expected_sha256],
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
        env=dict(environment),
        cwd=runtime_python.parent.parent,
    )
    observed = completed.stdout.strip()
    if observed != expected_sha256:
        raise RuntimeError(
            "isolated tokenizer-aware input verifier did not return the exact "
            f"bundle SHA-256; stdout={completed.stdout[-2000:]!r}, "
            f"stderr={completed.stderr[-4000:]!r}"
        )
    return observed


def _make_site_packages_read_only(runtime_python: Path) -> None:
    completed = subprocess.run(
        [
            str(runtime_python),
            "-c",
            (
                "import json,site; "
                "print(json.dumps([p for p in site.getsitepackages() if p]))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    paths = json.loads(completed.stdout)
    if not isinstance(paths, list) or not paths:
        raise RuntimeError("isolated runtime did not report site-packages")
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"invalid isolated site-packages path: {path}")
        for root, directories, files in os.walk(path):
            root_path = Path(root)
            for name in directories:
                child = root_path / name
                if not child.is_symlink():
                    child.chmod(
                        child.stat().st_mode
                        & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                    )
            for name in files:
                child = root_path / name
                if not child.is_symlink():
                    child.chmod(
                        child.stat().st_mode
                        & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                    )
        path.chmod(
            path.stat().st_mode
            & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        )


def _artifact_pin(plan_record: Mapping[str, Any], key: str) -> str:
    runtime = plan_record.get("runtime_contract")
    if not isinstance(runtime, Mapping):
        raise ValueError("plan.runtime_contract must be an object")
    pins = runtime.get("artifact_sha256")
    if not isinstance(pins, Mapping):
        raise ValueError("plan artifact pins must be an object")
    value = pins.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"plan artifact pin {key!r} is invalid")
    return value


def _required_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _file_sha256(path: Path) -> str:
    from hashlib import sha256

    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"qualification artifact is not one regular file: {path}")
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["run_gpu_qualification_sentinel"]
