"""Reviewed runtime dispatcher for vLLM 0.27.1 GPU qualification sentinels.

Imports of torch, Triton, and vLLM live in the isolated worker module so local
payload rendering and CPU unit tests do not require the GPU runtime.
"""

from __future__ import annotations

import errno
import importlib.metadata
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_DISTRIBUTION_NAME,
    GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_DISTRIBUTION_VERSION,
    GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_LIBCUDART_MEMBER,
    GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_LIBCUDART_SHA256,
    GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_LIBCUDART_SIZE_BYTES,
    build_gpu_qualification_system_cuda_parent_attestation,
    canonical_gpu_qualification_json,
    validate_gpu_qualification_system_cuda_parent_attestation,
)
from document_kv_cache.serving_env import (
    VLLM_PATCHED_WHEEL_SHA256_ENV,
    VLLM_PATCHED_WHEEL_URI_ENV,
    gpu_runtime_warning_environment_overrides,
    vllm_runtime_lock_path,
)
from document_kv_cache.vllm_smoke import (
    _attest_isolated_python,
    _pip_subprocess_environment,
    create_venv,
    install_document_kv_package,
    install_vllm,
    verify_vllm_runtime_lock_installation,
)


_RUNTIME_LOCK_ATTESTATION_ENV = "CACHET_GPU_QUALIFICATION_RUNTIME_LOCK_ATTESTATION"
_SYSTEM_CUDA_PARENT_ATTESTATION_ENV = (
    "CACHET_GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_ATTESTATION"
)
_SYSTEM_CUDA_PARENT_FILE_READ_BYTES = 64 * 1024
_WORKER_STDOUT_TAIL_MAX_BYTES = 2_000
_WORKER_STDERR_TAIL_MAX_BYTES = 16_384
_WORKER_STREAM_READ_BYTES = 64 * 1024
_WORKER_CONTROLLER_POLL_SECONDS = 0.05
_WORKER_DRAIN_TIMEOUT_SECONDS = 2.0
_WORKER_TERMINATION_GRACE_SECONDS = 2.0
_SITE_PACKAGES_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_ALLOWED_SITE_PACKAGES_RELATIVE_PARTS = frozenset(
    {
        ("lib", "python3", "dist-packages"),
        ("lib", "python3.11", "dist-packages"),
        ("lib", "python3.11", "site-packages"),
        ("local", "lib", "python3.11", "dist-packages"),
    }
)


def _gpu_runtime_subprocess_environment() -> dict[str, str]:
    environment = _pip_subprocess_environment()
    environment.pop(_SYSTEM_CUDA_PARENT_ATTESTATION_ENV, None)
    environment.update(gpu_runtime_warning_environment_overrides())
    return environment


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _read_bounded_no_follow_regular_file(
    path: Path,
    *,
    max_bytes: int,
    allow_absent: bool,
    label: str,
) -> bytes | None:
    """Read one stable regular leaf through a no-follow descriptor."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("no-follow file read limit must be a positive integer")
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            )
        except FileNotFoundError as exc:
            if allow_absent and exc.errno == errno.ENOENT:
                return None
            raise RuntimeError(f"{label} is unavailable") from exc
        except OSError as exc:
            raise RuntimeError(f"{label} could not be opened without following") from exc
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= max_bytes:
            raise RuntimeError(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(
                descriptor,
                min(_SYSTEM_CUDA_PARENT_FILE_READ_BYTES, max_bytes + 1 - byte_count),
            )
            if not chunk:
                break
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > max_bytes:
                raise RuntimeError(f"{label} exceeds its byte limit")
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after) or byte_count != after.st_size:
            raise RuntimeError(f"{label} changed while it was read")
        try:
            path_status = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(f"{label} path changed after it was read") from exc
        if _file_identity(path_status) != _file_identity(after):
            raise RuntimeError(f"{label} path changed after it was read")
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _capture_system_cuda_parent_attestation() -> dict[str, Any]:
    matches: list[importlib.metadata.Distribution] = []
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata["Name"]
        if (
            isinstance(raw_name, str)
            and _canonical_distribution_name(raw_name)
            == GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_DISTRIBUTION_NAME
        ):
            matches.append(distribution)
    if len(matches) != 1 or matches[0].version != (
        GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_DISTRIBUTION_VERSION
    ):
        raise RuntimeError("Databricks parent CUDA runtime distribution differs")
    distribution = matches[0]
    files = distribution.files
    members = (
        []
        if files is None
        else [
            member
            for member in files
            if str(member) == GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_LIBCUDART_MEMBER
        ]
    )
    if len(members) != 1:
        raise RuntimeError("Databricks parent CUDA runtime member inventory differs")
    distribution_root = Path(str(distribution.locate_file("")))
    libcudart_path = Path(str(distribution.locate_file(members[0])))
    expected_path = distribution_root / GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_LIBCUDART_MEMBER
    if libcudart_path != expected_path:
        raise RuntimeError("Databricks parent CUDA runtime member path differs")
    content = _read_bounded_no_follow_regular_file(
        libcudart_path,
        max_bytes=GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_LIBCUDART_SIZE_BYTES,
        allow_absent=False,
        label="Databricks parent CUDA runtime member",
    )
    if (
        content is None
        or len(content) != GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_LIBCUDART_SIZE_BYTES
        or sha256(content).hexdigest()
        != GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_LIBCUDART_SHA256
    ):
        raise RuntimeError("Databricks parent CUDA runtime member bytes differ")
    return build_gpu_qualification_system_cuda_parent_attestation(
        distribution_root=str(distribution_root),
        libcudart_path=str(libcudart_path),
    )


def _system_cuda_parent_attestation_from_environment() -> dict[str, Any]:
    raw = os.environ.get(_SYSTEM_CUDA_PARENT_ATTESTATION_ENV)
    if raw is None:
        raise RuntimeError("Databricks parent CUDA runtime attestation is unavailable")
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Databricks parent CUDA runtime attestation is invalid JSON"
        ) from exc
    if (
        not isinstance(record, dict)
        or canonical_gpu_qualification_json(record) != raw
    ):
        raise RuntimeError("Databricks parent CUDA runtime attestation is not canonical")
    try:
        validate_gpu_qualification_system_cuda_parent_attestation(record)
    except ValueError as exc:
        raise RuntimeError("Databricks parent CUDA runtime attestation differs") from exc
    member = _read_bounded_no_follow_regular_file(
        Path(record["libcudart_path"]),
        max_bytes=GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_LIBCUDART_SIZE_BYTES,
        allow_absent=False,
        label="attested Databricks parent CUDA runtime member",
    )
    if (
        member is None
        or len(member) != GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_LIBCUDART_SIZE_BYTES
        or sha256(member).hexdigest()
        != GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_LIBCUDART_SHA256
    ):
        raise RuntimeError("attested Databricks parent CUDA runtime member bytes differ")
    return dict(record)


@dataclass(frozen=True, slots=True)
class _BoundedWorkerStream:
    byte_count: int
    sha256: str
    truncated: bool
    tail: bytes

    @property
    def tail_text(self) -> str:
        return self.tail.decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class _BoundedWorkerResult:
    returncode: int
    stdout: _BoundedWorkerStream
    stderr: _BoundedWorkerStream


class _WorkerStreamAccumulator:
    def __init__(self, tail_limit: int) -> None:
        if type(tail_limit) is not int or tail_limit <= 0:
            raise ValueError("worker stream tail limit must be a positive integer")
        self._tail_limit = tail_limit
        self._tail = bytearray()
        self._byte_count = 0
        self._digest = sha256()

    def update(self, chunk: bytes) -> None:
        self._byte_count += len(chunk)
        self._digest.update(chunk)
        if len(chunk) >= self._tail_limit:
            self._tail[:] = chunk[-self._tail_limit :]
            return
        self._tail.extend(chunk)
        overflow = len(self._tail) - self._tail_limit
        if overflow > 0:
            del self._tail[:overflow]

    def finish(self) -> _BoundedWorkerStream:
        return _BoundedWorkerStream(
            byte_count=self._byte_count,
            sha256=self._digest.hexdigest(),
            truncated=self._byte_count > self._tail_limit,
            tail=bytes(self._tail),
        )


def _drain_worker_stream(
    stream: Any,
    accumulator: _WorkerStreamAccumulator,
    errors: list[BaseException],
    error_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    """Drain one nonblocking raw pipe and close it in its owning thread."""

    selector = selectors.DefaultSelector()
    try:
        descriptor = stream.fileno()
        selector.register(descriptor, selectors.EVENT_READ)
        while not stop_event.is_set():
            if not selector.select(timeout=_WORKER_CONTROLLER_POLL_SECONDS):
                continue
            try:
                chunk = os.read(descriptor, _WORKER_STREAM_READ_BYTES)
            except BlockingIOError:
                continue
            if chunk == b"":
                break
            if type(chunk) is not bytes:
                raise TypeError("worker stream must remain binary")
            accumulator.update(chunk)
    except BaseException as exc:
        with error_lock:
            errors.append(exc)
    finally:
        selector.close()
        try:
            stream.close()
        except BaseException as exc:
            with error_lock:
                errors.append(exc)


def _signal_worker_process_group(
    process: subprocess.Popen[bytes],
    signal_number: int,
) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except (PermissionError, ProcessLookupError):
        # Darwin can report EPERM while the last killed, reparented member is
        # disappearing. A live member of this same-UID owned group is
        # signalable; either error therefore means there is no addressable
        # process left inside the controller's authority boundary.
        return


def _worker_process_group_exists(process: subprocess.Popen[bytes]) -> bool:
    try:
        os.killpg(process.pid, 0)
    except (PermissionError, ProcessLookupError):
        return False
    return True


def _wait_worker_process_group_exit(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _worker_process_group_exists(process):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_WORKER_CONTROLLER_POLL_SECONDS, remaining))
    return True


def _settle_worker_process_group(
    process: subprocess.Popen[bytes],
    *,
    termination_grace_seconds: float,
) -> None:
    """Terminate every process that remains in the worker's owned session."""

    _signal_worker_process_group(process, signal.SIGTERM)
    if _wait_worker_process_group_exit(
        process,
        timeout_seconds=termination_grace_seconds,
    ):
        return
    _signal_worker_process_group(process, signal.SIGKILL)
    if not _wait_worker_process_group_exit(
        process,
        timeout_seconds=termination_grace_seconds,
    ):
        raise RuntimeError("GPU sentinel worker process group did not terminate")


def _join_worker_drainers(
    threads: tuple[threading.Thread, threading.Thread],
    *,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    for thread in threads:
        if thread.ident is not None:
            thread.join(max(0.0, deadline - time.monotonic()))
    return all(not thread.is_alive() for thread in threads)


def _stop_worker_drainers(
    threads: tuple[threading.Thread, threading.Thread],
    streams: tuple[Any, Any],
    *,
    stop_event: threading.Event,
    timeout_seconds: float,
) -> bool:
    stop_event.set()
    for thread, stream in zip(threads, streams, strict=True):
        if thread.ident is None and not stream.closed:
            # No other thread ever owned this raw descriptor.
            stream.close()
    return _join_worker_drainers(threads, timeout_seconds=timeout_seconds)


def _finish_worker_process_group(
    process: subprocess.Popen[bytes],
    threads: tuple[threading.Thread, threading.Thread],
    streams: tuple[Any, Any],
    *,
    stop_event: threading.Event,
    request_termination: bool,
    drain_timeout_seconds: float,
    termination_grace_seconds: float,
) -> None:
    if request_termination:
        _signal_worker_process_group(process, signal.SIGTERM)
        if process.poll() is None:
            try:
                process.wait(timeout=termination_grace_seconds)
            except subprocess.TimeoutExpired:
                _signal_worker_process_group(process, signal.SIGKILL)
                process.wait(timeout=termination_grace_seconds)
    if process.poll() is None:
        raise RuntimeError("GPU sentinel worker leader did not terminate")

    # Always settle the original PGID after the leader exits. A child may have
    # closed both inherited pipes yet still be alive in the owned session.
    _settle_worker_process_group(
        process,
        termination_grace_seconds=termination_grace_seconds,
    )
    if _join_worker_drainers(threads, timeout_seconds=drain_timeout_seconds):
        return

    # A hostile descendant can call setsid(2) and leave the owned session. It
    # is outside this controller's portable signalling authority, but an
    # inherited pipe must not let it hold the controller. Nonblocking owner
    # threads observe this stop request and close their own raw descriptors.
    if not _stop_worker_drainers(
        threads,
        streams,
        stop_event=stop_event,
        timeout_seconds=drain_timeout_seconds,
    ):
        if _worker_process_group_exists(process):
            _signal_worker_process_group(process, signal.SIGKILL)
        raise RuntimeError("GPU sentinel worker pipe drain did not terminate")


def _worker_stream_diagnostic(
    label: str,
    captured: _BoundedWorkerStream,
) -> str:
    truncated = "true" if captured.truncated else "false"
    return (
        f"{label}(bytes={captured.byte_count},sha256={captured.sha256},"
        f"truncated={truncated},tail={captured.tail_text!r})"
    )


def _worker_process_failure(
    job_id: str,
    *,
    result: _BoundedWorkerResult,
    timed_out: bool,
    timeout_seconds: float,
) -> RuntimeError:
    streams = (
        f"{_worker_stream_diagnostic('stdout', result.stdout)}; "
        f"{_worker_stream_diagnostic('stderr', result.stderr)}"
    )
    if timed_out:
        return RuntimeError(
            f"GPU sentinel {job_id!r} worker timed out after "
            f"{timeout_seconds:g} seconds; {streams}"
        )
    if result.returncode < 0:
        signal_number = -result.returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = "UNKNOWN"
        return RuntimeError(
            f"GPU sentinel {job_id!r} worker terminated by signal "
            f"{signal_number} ({signal_name}); {streams}"
        )
    return RuntimeError(
        f"GPU sentinel {job_id!r} worker exited with status "
        f"{result.returncode}; {streams}"
    )


def _first_worker_drain_error(
    errors: list[BaseException],
    error_lock: threading.Lock,
) -> BaseException | None:
    with error_lock:
        return errors[0] if errors else None


def _abort_worker_process(
    process: subprocess.Popen[bytes],
    threads: tuple[threading.Thread, threading.Thread],
    streams: tuple[Any, Any],
    *,
    stop_event: threading.Event,
    drain_timeout_seconds: float,
    termination_grace_seconds: float,
) -> None:
    _signal_worker_process_group(process, signal.SIGKILL)
    if process.poll() is None:
        try:
            process.wait(timeout=termination_grace_seconds)
        except subprocess.TimeoutExpired:
            pass
    _signal_worker_process_group(process, signal.SIGKILL)
    _wait_worker_process_group_exit(
        process,
        timeout_seconds=termination_grace_seconds,
    )
    _stop_worker_drainers(
        threads,
        streams,
        stop_event=stop_event,
        timeout_seconds=drain_timeout_seconds,
    )


def _run_bounded_worker_process(
    argv: list[str],
    *,
    job_id: str,
    timeout_seconds: float,
    environment: Mapping[str, str],
    cwd: Path,
    stdout_tail_max_bytes: int = _WORKER_STDOUT_TAIL_MAX_BYTES,
    stderr_tail_max_bytes: int = _WORKER_STDERR_TAIL_MAX_BYTES,
    drain_timeout_seconds: float = _WORKER_DRAIN_TIMEOUT_SECONDS,
    termination_grace_seconds: float = _WORKER_TERMINATION_GRACE_SECONDS,
) -> _BoundedWorkerResult:
    """Run one package-owned process group with bounded binary pipe retention.

    Cleanup authority covers the new session created for the reviewed worker.
    A hostile descendant can deliberately escape it with ``setsid(2)``; raw
    nonblocking owner-thread drains ensure even that process cannot hold this
    controller by retaining an inherited pipe.
    """

    for value, label in (
        (timeout_seconds, "timeout_seconds"),
        (drain_timeout_seconds, "drain_timeout_seconds"),
        (termination_grace_seconds, "termination_grace_seconds"),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{label} must be positive")
    stdout_accumulator = _WorkerStreamAccumulator(stdout_tail_max_bytes)
    stderr_accumulator = _WorkerStreamAccumulator(stderr_tail_max_bytes)
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(environment),
        cwd=cwd,
        start_new_session=True,
        bufsize=0,
    )
    if process.stdout is None or process.stderr is None:
        _signal_worker_process_group(process, signal.SIGKILL)
        process.wait(timeout=termination_grace_seconds)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        raise RuntimeError("GPU sentinel worker pipes were not created")
    streams = (process.stdout, process.stderr)
    drain_errors: list[BaseException] = []
    drain_error_lock = threading.Lock()
    stop_event = threading.Event()
    threads = (
        threading.Thread(
            target=_drain_worker_stream,
            args=(
                process.stdout,
                stdout_accumulator,
                drain_errors,
                drain_error_lock,
                stop_event,
            ),
            name="gpu-sentinel-stdout-drain",
            daemon=True,
        ),
        threading.Thread(
            target=_drain_worker_stream,
            args=(
                process.stderr,
                stderr_accumulator,
                drain_errors,
                drain_error_lock,
                stop_event,
            ),
            name="gpu-sentinel-stderr-drain",
            daemon=True,
        ),
    )
    timed_out = False
    try:
        for stream in streams:
            os.set_blocking(stream.fileno(), False)
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None:
            drain_error = _first_worker_drain_error(
                drain_errors,
                drain_error_lock,
            )
            if drain_error is not None:
                raise RuntimeError("GPU sentinel worker pipe drain failed") from (
                    drain_error
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                process.wait(
                    timeout=min(_WORKER_CONTROLLER_POLL_SECONDS, remaining)
                )
            except subprocess.TimeoutExpired:
                continue
        _finish_worker_process_group(
            process,
            threads,
            streams,
            stop_event=stop_event,
            request_termination=timed_out,
            drain_timeout_seconds=drain_timeout_seconds,
            termination_grace_seconds=termination_grace_seconds,
        )
    except BaseException:
        _abort_worker_process(
            process,
            threads,
            streams,
            stop_event=stop_event,
            drain_timeout_seconds=drain_timeout_seconds,
            termination_grace_seconds=termination_grace_seconds,
        )
        raise
    drain_error = _first_worker_drain_error(drain_errors, drain_error_lock)
    if drain_error is not None:
        raise RuntimeError("GPU sentinel worker pipe drain failed") from drain_error
    returncode = process.returncode
    if type(returncode) is not int:
        raise RuntimeError("GPU sentinel worker did not reach a terminal status")
    result = _BoundedWorkerResult(
        returncode=returncode,
        stdout=stdout_accumulator.finish(),
        stderr=stderr_accumulator.finish(),
    )
    if timed_out or returncode != 0:
        raise _worker_process_failure(
            job_id,
            result=result,
            timed_out=timed_out,
            timeout_seconds=float(timeout_seconds),
        )
    return result


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
    expected_python_version = _runtime_python_version(plan_record)
    system_cuda_parent_attestation = _capture_system_cuda_parent_attestation()
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

    create_venv(runtime_dir, copies=True)
    created_python_identity = _attest_isolated_python(
        runtime_dir,
        expected_python_version=expected_python_version,
    )
    install_vllm(runtime_python)
    install_document_kv_package(runtime_python, str(package_wheel))
    subprocess.run(
        [str(runtime_python), "-m", "pip", "check"],
        check=True,
        timeout=300,
        env=_pip_subprocess_environment(),
    )
    installed_python_identity = _attest_isolated_python(
        runtime_dir,
        expected_python_version=expected_python_version,
        expected_file_binding=created_python_identity.file_binding,
    )
    if installed_python_identity != created_python_identity:
        raise RuntimeError("isolated runtime Python identity changed during installation")

    environment = _gpu_runtime_subprocess_environment()
    environment.update(
        {
            "HF_HOME": "/local_disk0/cachet-vllm-0271-hf",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONSAFEPATH": "1",
            "TOKENIZERS_PARALLELISM": "false",
            _SYSTEM_CUDA_PARENT_ATTESTATION_ENV: canonical_gpu_qualification_json(
                system_cuda_parent_attestation
            ),
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
    completed = _run_bounded_worker_process(
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
        job_id=job_id,
        timeout_seconds=14_000,
        environment=environment,
        cwd=worker_dir,
    )
    if not worker_output.is_file() or worker_output.is_symlink():
        raise RuntimeError(
            f"GPU sentinel {job_id!r} did not write its measurement record; "
            f"{_worker_stream_diagnostic('stdout', completed.stdout)}; "
            f"{_worker_stream_diagnostic('stderr', completed.stderr)}"
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
    if not isinstance(no_follow, int) or not isinstance(directory, int):
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


def _runtime_python_version(plan_record: Mapping[str, Any]) -> str:
    runtime = plan_record.get("runtime_contract")
    if not isinstance(runtime, Mapping):
        raise ValueError("plan.runtime_contract must be an object")
    platform = runtime.get("platform")
    if not isinstance(platform, Mapping):
        raise ValueError("plan runtime platform must be an object")
    value = platform.get("python_version")
    if not isinstance(value, str) or not value:
        raise ValueError("plan runtime Python version is invalid")
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
