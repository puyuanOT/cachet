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
    _pip_subprocess_environment,
    create_venv,
    install_document_kv_package,
    install_vllm,
    verify_vllm_runtime_lock_installation,
)


_RUNTIME_LOCK_ATTESTATION_ENV = "CACHET_GPU_QUALIFICATION_RUNTIME_LOCK_ATTESTATION"
_SITE_PACKAGES_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_ALLOWED_SITE_PACKAGES_RELATIVE_PARTS = frozenset(
    {
        ("lib", "python3", "dist-packages"),
        ("lib", "python3.11", "dist-packages"),
        ("lib", "python3.11", "site-packages"),
        ("local", "lib", "python3.11", "dist-packages"),
    }
)


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
        env=_pip_subprocess_environment(),
    )

    environment = _pip_subprocess_environment()
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


def _directory_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise RuntimeError(
            "isolated site-packages authority requires O_NOFOLLOW and O_DIRECTORY"
        )
    return os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)


def _open_runtime_root_no_follow(runtime_root: Path) -> int:
    if not runtime_root.is_absolute() or ".." in runtime_root.parts:
        raise RuntimeError("isolated runtime root is not canonical")
    try:
        pre_open_status = runtime_root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(pre_open_status.st_mode):
            raise RuntimeError("isolated runtime root is invalid")
        if runtime_root.resolve(strict=True) != runtime_root:
            raise RuntimeError("isolated runtime root is invalid")
    except OSError as error:
        raise RuntimeError("isolated runtime root is invalid") from error
    flags = _directory_open_flags()
    descriptor = os.open("/", flags)
    try:
        for component in runtime_root.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        path_status = runtime_root.stat(follow_symlinks=False)
        opened_status = os.fstat(descriptor)
        if (
            not _same_file_identity(pre_open_status, opened_status)
            or not _same_file_identity(path_status, opened_status)
            or not stat.S_ISDIR(opened_status.st_mode)
        ):
            raise RuntimeError("isolated runtime root changed during validation")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_no_follow(
    runtime_root_descriptor: int,
    relative_parts: tuple[str, ...],
    *,
    directory: bool,
) -> int:
    if not relative_parts or any(
        not part or part in {".", ".."} or "/" in part for part in relative_parts
    ):
        raise RuntimeError("isolated site-packages relative path is invalid")
    directory_flags = _directory_open_flags()
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW")
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.dup(runtime_root_descriptor)
    try:
        for index, component in enumerate(relative_parts):
            is_leaf = index == len(relative_parts) - 1
            flags = directory_flags if not is_leaf or directory else file_flags
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _snapshot_site_packages_directory(
    directory_descriptor: int,
    relative_parts: tuple[str, ...],
) -> list[tuple[tuple[str, ...], os.stat_result]]:
    directory_status = os.fstat(directory_descriptor)
    if not stat.S_ISDIR(directory_status.st_mode):
        raise RuntimeError("isolated site-packages directory changed during scan")
    snapshot = [(relative_parts, directory_status)]
    try:
        with os.scandir(directory_descriptor) as iterator:
            children = sorted(iterator, key=lambda entry: entry.name)
    except OSError as error:
        raise RuntimeError("could not scan isolated site-packages directory") from error
    for child in children:
        try:
            child_status = child.stat(follow_symlinks=False)
        except OSError as error:
            raise RuntimeError("could not inspect isolated site-packages entry") from error
        child_parts = (*relative_parts, child.name)
        if stat.S_ISLNK(child_status.st_mode):
            raise RuntimeError(
                "isolated site-packages tree contains a symbolic link"
            )
        if stat.S_ISDIR(child_status.st_mode):
            try:
                child_descriptor = os.open(
                    child.name,
                    _directory_open_flags(),
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                raise RuntimeError(
                    "could not safely open isolated site-packages directory"
                ) from error
            try:
                if not _same_file_identity(
                    child_status,
                    os.fstat(child_descriptor),
                ):
                    raise RuntimeError(
                        "isolated site-packages directory changed during scan"
                    )
                snapshot.extend(
                    _snapshot_site_packages_directory(
                        child_descriptor,
                        child_parts,
                    )
                )
            finally:
                os.close(child_descriptor)
        elif stat.S_ISREG(child_status.st_mode):
            if child_status.st_nlink != 1:
                raise RuntimeError(
                    "isolated site-packages file has an unsafe link count"
                )
            snapshot.append((child_parts, child_status))
        else:
            raise RuntimeError(
                "isolated site-packages tree contains a non-regular entry"
            )
    return snapshot


def _open_validated_site_packages_tree(
    runtime_root: Path,
    raw_paths: object,
) -> tuple[int, tuple[tuple[tuple[str, ...], os.stat_result], ...]]:
    """Open and snapshot the exact Debian package roots without following links."""

    if not isinstance(raw_paths, list) or not raw_paths:
        raise RuntimeError("isolated runtime did not report site-packages")
    relative_roots: list[tuple[str, ...]] = []
    seen_roots: set[tuple[str, ...]] = set()
    for raw_path in raw_paths:
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or raw_path.strip() != raw_path
        ):
            raise RuntimeError("isolated runtime reported an invalid site-packages path")
        path = Path(raw_path)
        if not path.is_absolute() or ".." in path.parts or str(path) != raw_path:
            raise RuntimeError(f"invalid isolated site-packages path: {path}")
        try:
            relative = path.relative_to(runtime_root)
        except ValueError as error:
            raise RuntimeError(
                f"invalid isolated site-packages path: {path}"
            ) from error
        if relative.parts not in _ALLOWED_SITE_PACKAGES_RELATIVE_PARTS:
            raise RuntimeError(f"invalid isolated site-packages path: {path}")
        if relative.parts in seen_roots:
            raise RuntimeError(f"duplicate isolated site-packages path: {path}")
        seen_roots.add(relative.parts)
        relative_roots.append(relative.parts)

    runtime_root_descriptor = _open_runtime_root_no_follow(runtime_root)
    snapshot: list[tuple[tuple[str, ...], os.stat_result]] = []
    try:
        for relative_parts in relative_roots:
            try:
                package_descriptor = _open_relative_no_follow(
                    runtime_root_descriptor,
                    relative_parts,
                    directory=True,
                )
            except FileNotFoundError:
                continue
            except OSError as error:
                raise RuntimeError(
                    "could not safely open isolated site-packages candidate"
                ) from error
            try:
                snapshot.extend(
                    _snapshot_site_packages_directory(
                        package_descriptor,
                        relative_parts,
                    )
                )
            finally:
                os.close(package_descriptor)
        if not snapshot:
            raise RuntimeError(
                "isolated runtime has no existing site-packages directory"
            )
        if len({parts for parts, _status in snapshot}) != len(snapshot):
            raise RuntimeError("isolated site-packages snapshot contains duplicates")
        return runtime_root_descriptor, tuple(snapshot)
    except BaseException:
        os.close(runtime_root_descriptor)
        raise


def _validated_site_packages_tree(
    runtime_root: Path,
    raw_paths: object,
) -> tuple[Path, ...]:
    runtime_root_descriptor, snapshot = _open_validated_site_packages_tree(
        runtime_root,
        raw_paths,
    )
    os.close(runtime_root_descriptor)
    return tuple(runtime_root.joinpath(*parts) for parts, _status in snapshot)


def _open_snapshot_entry(
    runtime_root_descriptor: int,
    relative_parts: tuple[str, ...],
    expected_status: os.stat_result,
) -> int:
    try:
        descriptor = _open_relative_no_follow(
            runtime_root_descriptor,
            relative_parts,
            directory=stat.S_ISDIR(expected_status.st_mode),
        )
    except OSError as error:
        raise RuntimeError(
            "could not safely reopen isolated site-packages entry"
        ) from error
    try:
        reopened_status = os.fstat(descriptor)
        if (
            not _same_file_identity(expected_status, reopened_status)
            or (
                stat.S_ISREG(reopened_status.st_mode)
                and reopened_status.st_nlink != 1
            )
        ):
            raise RuntimeError(
                "isolated site-packages entry changed after validation"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _lock_site_packages_entry_read_only(
    runtime_root_descriptor: int,
    relative_parts: tuple[str, ...],
    expected_status: os.stat_result,
) -> None:
    descriptor = _open_snapshot_entry(
        runtime_root_descriptor,
        relative_parts,
        expected_status,
    )
    try:
        before_lock_status = os.fstat(descriptor)
        before_lock_mode = stat.S_IMODE(before_lock_status.st_mode)
        os.fchmod(
            descriptor,
            before_lock_mode & ~_SITE_PACKAGES_WRITE_BITS,
        )
        locked_status = os.fstat(descriptor)
        if stat.S_ISREG(locked_status.st_mode) and locked_status.st_nlink != 1:
            os.fchmod(descriptor, before_lock_mode)
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != before_lock_mode:
                raise RuntimeError(
                    "could not restore isolated site-packages mode after "
                    "a link-count change"
                )
            raise RuntimeError(
                "isolated site-packages file link count changed during lockdown"
            )
        if locked_status.st_mode & _SITE_PACKAGES_WRITE_BITS:
            raise RuntimeError(
                "isolated site-packages entry remained writable after lockdown"
            )
    finally:
        os.close(descriptor)


def _site_packages_tree_read_only(runtime_root: Path, raw_paths: object) -> bool:
    """Attest the launch-owned runtime while no concurrent modifier is allowed."""

    runtime_root_descriptor, snapshot = _open_validated_site_packages_tree(
        runtime_root,
        raw_paths,
    )
    try:
        for relative_parts, expected_status in snapshot:
            descriptor = _open_snapshot_entry(
                runtime_root_descriptor,
                relative_parts,
                expected_status,
            )
            try:
                if os.fstat(descriptor).st_mode & _SITE_PACKAGES_WRITE_BITS:
                    return False
            finally:
                os.close(descriptor)
        return True
    finally:
        os.close(runtime_root_descriptor)


def _make_site_packages_read_only(runtime_python: Path) -> None:
    """Freeze the launch-owned runtime before the sentinel worker is started.

    The qualification launch contract gives this dispatcher exclusive mutation
    ownership of the new runtime until this function returns.  POSIX mode and
    link-count checks do not provide isolation from a hostile process sharing
    the same UID, so such a concurrent modifier is outside this contract.
    """

    completed = subprocess.run(
        [
            str(runtime_python),
            "-c",
            (
                "import json,site; "
                "print(json.dumps(site.getsitepackages()))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=_pip_subprocess_environment(),
    )
    paths = json.loads(completed.stdout)
    runtime_root = runtime_python.parent.parent
    runtime_root_descriptor, snapshot = _open_validated_site_packages_tree(
        runtime_root,
        paths,
    )
    try:
        for relative_parts, expected_status in sorted(
            snapshot,
            key=lambda entry: len(entry[0]),
            reverse=True,
        ):
            _lock_site_packages_entry_read_only(
                runtime_root_descriptor,
                relative_parts,
                expected_status,
            )
    finally:
        os.close(runtime_root_descriptor)
    if not _site_packages_tree_read_only(runtime_root, paths):
        raise RuntimeError("isolated site-packages tree remained writable")


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
