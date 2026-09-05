"""Isolated runtime installer and dispatcher for GPU qualification v2."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.resources
import json
import os
import platform
import re
import selectors
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import monotonic, sleep
from types import MappingProxyType
from typing import Any, Final, cast
from urllib.parse import unquote, urlsplit

from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PACKAGE_VERSION,
    FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256,
    FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
    FLASHINFER_PATCHED_WHEEL_SHA256,
    FLASHINFER_TARGET_MEMBER,
    FLASHINFER_TARGET_PATCHED_SHA256,
)
from document_kv_cache.gpu_qualification import canonical_gpu_qualification_json
import document_kv_cache.gpu_qualification as qualification_v1
from document_kv_cache.gpu_qualification_sentinels import (
    _SYSTEM_CUDA_PARENT_ATTESTATION_ENV,
    _capture_system_cuda_parent_attestation,
    _make_site_packages_read_only,
    _open_runtime_root_no_follow,
    _run_bounded_worker_process,
    _system_cuda_parent_attestation_from_environment,
    _verify_input_bundle_in_isolated_runtime,
    _worker_stream_diagnostic,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_ARTIFACT_KEYS,
    GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION,
    GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION,
    GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT,
    GPU_QUALIFICATION_V2_WITH_FLASHINFER_DISTRIBUTION_COUNT,
    GPU_QUALIFICATION_V2_WITH_VLLM_DISTRIBUTION_COUNT,
    build_gpu_runtime_verification_v2,
    gpu_qualification_v2_runtime_closure,
    pins_from_gpu_qualification_plan_v2,
    validate_gpu_qualification_plan_v2_record,
    validate_gpu_qualification_v2_runtime_attestation,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE,
    RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE,
    RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION,
    VLLM_PACKAGE_VERSION,
    VLLM_PATCHED_MANIFEST_SHA256,
    VLLM_PATCHED_WHEEL_SHA256,
    VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_BASE_LOCK_FILENAME,
    VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
)
from document_kv_cache.serving_env import (
    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
    GPU_RUNTIME_PYTHONWARNINGS,
    VLLM_PATCHED_WHEEL_SHA256_ENV,
    VLLM_PATCHED_WHEEL_URI_ENV,
    gpu_runtime_warning_environment_overrides,
)
from document_kv_cache.vllm_smoke import (
    _attest_isolated_python,
    _pip_subprocess_environment,
    create_venv,
)


_RUNTIME_LOCK_ATTESTATION_ENV: Final = (
    "CACHET_GPU_QUALIFICATION_RUNTIME_LOCK_ATTESTATION"
)
_RUNTIME_TORCH_LIBRARY_RELATIVE_PATH: Final = Path(
    "lib/python3.11/site-packages/torch/lib"
)
_BASE_LOCK_REQUIREMENT_RE: Final = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ]+?)(?: \\)?$")
_BASE_LOCK_HASH_RE: Final = re.compile(r"^\s+--hash=sha256:[0-9a-f]{64}(?: \\)?$")
_CANONICAL_NAME_RE: Final = re.compile(r"[-_.]+")
_FINAL_VERIFIER_PIP_CHECK_STDOUT: Final = b"No broken requirements found.\n"
_FINAL_VERIFIER_CHILD_RECORD_TYPE: Final = (
    "cachet.gpu_qualification_final_runtime_verifier_child.v1"
)
_FINAL_VERIFIER_CHILD_SCHEMA_VERSION: Final = 1
_FINAL_VERIFIER_PROCESS_OUTPUT_LIMIT_BYTES: Final = 1_048_576
_FINAL_VERIFIER_CAPTURE_COUNT_LIMIT: Final = (1 << 63) - 1
_FINAL_VERIFIER_EMPTY_STREAM_SHA256: Final = sha256(b"").hexdigest()
_FINAL_VERIFIER_INNER_PIP_TIMEOUT_SECONDS: Final = 180.0
_FINAL_VERIFIER_OUTER_TIMEOUT_SECONDS: Final = 300.0
_FINAL_VERIFIER_INNER_CLEANUP_BUDGET_SECONDS: Final = 10.0
_FINAL_VERIFIER_POST_PIP_BUDGET_SECONDS: Final = 60.0
_FINAL_VERIFIER_REQUIRED_HIERARCHY_MARGIN_SECONDS: Final = 90.0
_BOUNDED_SUBPROCESS_READ_BYTES: Final = 64 * 1024
_BOUNDED_SUBPROCESS_POLL_SECONDS: Final = 0.05
_BOUNDED_SUBPROCESS_TERMINATE_GRACE_SECONDS: Final = 0.5
_BOUNDED_STREAM_JOIN_SECONDS: Final = 1.0
_FINAL_VERIFIER_STAGES: Final = (
    "arguments",
    "platform",
    "pip_check",
    "base_lock",
    "runtime_closure",
    "distribution_inventory",
    "package_origin",
    "vllm_origin",
    "flashinfer_origin",
    "vllm_members",
    "flashinfer_members",
    "flashinfer_import",
    "packaged_lock",
    "attestation",
    "complete",
)
_FINAL_VERIFIER_CATEGORIES: Final = (
    "none",
    "invalid_arguments",
    "subprocess_nonzero",
    "subprocess_start_failure",
    "subprocess_timeout",
    "verification_rejected",
    "unexpected_exception",
)
_FINAL_VERIFIER_FAILURE_RELATIONS: Final = MappingProxyType(
    {
        "arguments": frozenset({"invalid_arguments", "unexpected_exception"}),
        "platform": frozenset({"verification_rejected", "unexpected_exception"}),
        "pip_check": frozenset(
            {
                "subprocess_nonzero",
                "subprocess_start_failure",
                "subprocess_timeout",
                "verification_rejected",
                "unexpected_exception",
            }
        ),
        "base_lock": frozenset({"verification_rejected", "unexpected_exception"}),
        "runtime_closure": frozenset({"verification_rejected", "unexpected_exception"}),
        "distribution_inventory": frozenset(
            {"verification_rejected", "unexpected_exception"}
        ),
        "package_origin": frozenset({"verification_rejected", "unexpected_exception"}),
        "vllm_origin": frozenset({"verification_rejected", "unexpected_exception"}),
        "flashinfer_origin": frozenset(
            {"verification_rejected", "unexpected_exception"}
        ),
        "vllm_members": frozenset({"verification_rejected", "unexpected_exception"}),
        "flashinfer_members": frozenset(
            {"verification_rejected", "unexpected_exception"}
        ),
        "flashinfer_import": frozenset(
            {"verification_rejected", "unexpected_exception"}
        ),
        "packaged_lock": frozenset({"verification_rejected", "unexpected_exception"}),
        "attestation": frozenset({"verification_rejected", "unexpected_exception"}),
    }
)
_FINAL_VERIFIER_CHILD_KEYS: Final = frozenset(
    {
        "attestation",
        "category",
        "ok",
        "record_type",
        "schema_version",
        "stage",
        "stderr_bytes",
        "stderr_sha256",
        "stdout_bytes",
        "stdout_sha256",
    }
)
_RUNTIME_CLOSURE = gpu_qualification_v2_runtime_closure()
_VLLM_MEMBER_SHA256: Final = MappingProxyType(
    dict(
        cast(
            Mapping[str, str],
            cast(Mapping[str, Any], _RUNTIME_CLOSURE["vllm"])["member_sha256"],
        )
    )
)
del _RUNTIME_CLOSURE


class _FinalRuntimeVerifierFailure(RuntimeError):
    """Finite internal failure whose category is safe for the child envelope."""

    def __init__(self, category: str, message: str) -> None:
        if category not in _FINAL_VERIFIER_CATEGORIES or category == "none":
            raise ValueError("invalid final runtime verifier failure category")
        super().__init__(message)
        self.category = category


class _BoundedSubprocessStartFailure(RuntimeError):
    """A fixed, no-detail binary subprocess start failure."""


class _BoundedSubprocessTransportFailure(RuntimeError):
    """A fixed, no-detail binary subprocess transport failure."""


@dataclass(frozen=True)
class _BoundedBinaryStreamResult:
    retained: bytes
    byte_count: int
    sha256: str
    limit_exceeded: bool


@dataclass(frozen=True)
class _BoundedBinarySubprocessResult:
    returncode: int
    stdout: _BoundedBinaryStreamResult
    stderr: _BoundedBinaryStreamResult
    timed_out: bool

    @property
    def output_limit_exceeded(self) -> bool:
        return self.stdout.limit_exceeded or self.stderr.limit_exceeded


class _BoundedBinaryAccumulator:
    def __init__(self, limit_bytes: int) -> None:
        self._limit_bytes = limit_bytes
        self._retained = bytearray()
        self._byte_count = 0
        self._digest = sha256()
        self._lock = threading.Lock()

    def add(self, chunk: bytes) -> bool:
        with self._lock:
            self._byte_count += len(chunk)
            self._digest.update(chunk)
            remaining = self._limit_bytes - len(self._retained)
            if remaining > 0:
                self._retained.extend(chunk[:remaining])
            return self._byte_count > self._limit_bytes

    def result(self) -> _BoundedBinaryStreamResult:
        with self._lock:
            return _BoundedBinaryStreamResult(
                retained=bytes(self._retained),
                byte_count=self._byte_count,
                sha256=self._digest.hexdigest(),
                limit_exceeded=self._byte_count > self._limit_bytes,
            )


def _bounded_stream_result_is_exact(result: _BoundedBinaryStreamResult) -> bool:
    return (
        not result.limit_exceeded
        and result.byte_count == len(result.retained)
        and sha256(result.retained).hexdigest() == result.sha256
    )


def _require_final_verifier_timeout_hierarchy() -> None:
    values = (
        _FINAL_VERIFIER_INNER_PIP_TIMEOUT_SECONDS,
        _FINAL_VERIFIER_OUTER_TIMEOUT_SECONDS,
        _FINAL_VERIFIER_INNER_CLEANUP_BUDGET_SECONDS,
        _FINAL_VERIFIER_POST_PIP_BUDGET_SECONDS,
        _FINAL_VERIFIER_REQUIRED_HIERARCHY_MARGIN_SECONDS,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        for value in values
    ) or not (
        _FINAL_VERIFIER_INNER_CLEANUP_BUDGET_SECONDS
        + _FINAL_VERIFIER_POST_PIP_BUDGET_SECONDS
        <= _FINAL_VERIFIER_REQUIRED_HIERARCHY_MARGIN_SECONDS
        and _FINAL_VERIFIER_INNER_PIP_TIMEOUT_SECONDS
        + _FINAL_VERIFIER_REQUIRED_HIERARCHY_MARGIN_SECONDS
        < _FINAL_VERIFIER_OUTER_TIMEOUT_SECONDS
    ):
        raise RuntimeError("v2 final runtime verifier timeout hierarchy differs")


def _drain_bounded_binary_stream(
    stream: Any,
    accumulator: _BoundedBinaryAccumulator,
    *,
    limit_event: threading.Event,
    failure_event: threading.Event,
    stop_event: threading.Event,
) -> None:
    selector = selectors.DefaultSelector()
    try:
        descriptor = stream.fileno()
        selector.register(descriptor, selectors.EVENT_READ)
        while not stop_event.is_set():
            if not selector.select(timeout=_BOUNDED_SUBPROCESS_POLL_SECONDS):
                continue
            try:
                chunk = os.read(descriptor, _BOUNDED_SUBPROCESS_READ_BYTES)
            except BlockingIOError:
                continue
            if chunk == b"":
                break
            if type(chunk) is not bytes:
                failure_event.set()
                break
            if accumulator.add(chunk):
                limit_event.set()
    except BaseException:  # noqa: BLE001 - stream details must never escape
        failure_event.set()
    finally:
        selector.close()
        try:
            stream.close()
        except BaseException:  # noqa: BLE001 - stream details must never escape
            failure_event.set()


def _bounded_stream_thread(
    stream: Any,
    accumulator: _BoundedBinaryAccumulator,
    *,
    limit_event: threading.Event,
    failure_event: threading.Event,
    stop_event: threading.Event,
    name: str,
) -> threading.Thread:
    return threading.Thread(
        target=_drain_bounded_binary_stream,
        args=(stream, accumulator),
        kwargs={
            "limit_event": limit_event,
            "failure_event": failure_event,
            "stop_event": stop_event,
        },
        name=name,
        daemon=True,
    )


def _signal_bounded_subprocess_group(
    process: subprocess.Popen[bytes], signal_number: int
) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except (PermissionError, ProcessLookupError):
        return
    except OSError:
        raise _BoundedSubprocessTransportFailure(
            "bounded binary subprocess group signaling failed"
        ) from None


def _bounded_subprocess_group_exists(process: subprocess.Popen[bytes]) -> bool:
    try:
        os.killpg(process.pid, 0)
    except (PermissionError, ProcessLookupError):
        return False
    except OSError:
        raise _BoundedSubprocessTransportFailure(
            "bounded binary subprocess group query failed"
        ) from None
    return True


def _wait_bounded_subprocess_group_exit(
    process: subprocess.Popen[bytes], *, timeout_seconds: float
) -> bool:
    deadline = monotonic() + timeout_seconds
    while _bounded_subprocess_group_exists(process):
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False
        sleep(min(_BOUNDED_SUBPROCESS_POLL_SECONDS, remaining))
    return True


def _settle_bounded_subprocess_group(process: subprocess.Popen[bytes]) -> None:
    _signal_bounded_subprocess_group(process, signal.SIGTERM)
    if _wait_bounded_subprocess_group_exit(
        process, timeout_seconds=_BOUNDED_SUBPROCESS_TERMINATE_GRACE_SECONDS
    ):
        return
    _signal_bounded_subprocess_group(process, signal.SIGKILL)
    if not _wait_bounded_subprocess_group_exit(
        process, timeout_seconds=_BOUNDED_SUBPROCESS_TERMINATE_GRACE_SECONDS
    ):
        raise _BoundedSubprocessTransportFailure(
            "bounded binary subprocess group did not terminate"
        )


def _join_bounded_stream_threads(
    threads: tuple[threading.Thread, threading.Thread], *, timeout_seconds: float
) -> bool:
    deadline = monotonic() + timeout_seconds
    for thread in threads:
        if thread.ident is not None:
            thread.join(max(0.0, deadline - monotonic()))
    return all(not thread.is_alive() for thread in threads)


def _stop_bounded_stream_threads(
    threads: tuple[threading.Thread, threading.Thread],
    streams: tuple[Any, Any],
    *,
    stop_event: threading.Event,
) -> bool:
    stop_event.set()
    for thread, stream in zip(threads, streams, strict=True):
        if thread.ident is None and not stream.closed:
            stream.close()
    return _join_bounded_stream_threads(
        threads, timeout_seconds=_BOUNDED_STREAM_JOIN_SECONDS
    )


def _abort_bounded_subprocess(
    process: subprocess.Popen[bytes],
    threads: tuple[threading.Thread, threading.Thread],
    streams: tuple[Any, Any],
    *,
    stop_event: threading.Event,
) -> None:
    _signal_bounded_subprocess_group(process, signal.SIGKILL)
    if process.poll() is None:
        try:
            process.wait(timeout=_BOUNDED_SUBPROCESS_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    _signal_bounded_subprocess_group(process, signal.SIGKILL)
    _wait_bounded_subprocess_group_exit(
        process, timeout_seconds=_BOUNDED_SUBPROCESS_TERMINATE_GRACE_SECONDS
    )
    _stop_bounded_stream_threads(threads, streams, stop_event=stop_event)


def _finish_bounded_subprocess(
    process: subprocess.Popen[bytes],
    threads: tuple[threading.Thread, threading.Thread],
    streams: tuple[Any, Any],
    *,
    stop_event: threading.Event,
    request_termination: bool,
) -> None:
    if request_termination:
        _signal_bounded_subprocess_group(process, signal.SIGTERM)
        if process.poll() is None:
            try:
                process.wait(timeout=_BOUNDED_SUBPROCESS_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _signal_bounded_subprocess_group(process, signal.SIGKILL)
                process.wait(timeout=_BOUNDED_SUBPROCESS_TERMINATE_GRACE_SECONDS)
    if process.poll() is None:
        raise _BoundedSubprocessTransportFailure(
            "bounded binary subprocess leader did not terminate"
        )
    _settle_bounded_subprocess_group(process)
    if _join_bounded_stream_threads(
        threads, timeout_seconds=_BOUNDED_STREAM_JOIN_SECONDS
    ):
        return
    if not _stop_bounded_stream_threads(threads, streams, stop_event=stop_event):
        if _bounded_subprocess_group_exists(process):
            _signal_bounded_subprocess_group(process, signal.SIGKILL)
        raise _BoundedSubprocessTransportFailure(
            "bounded binary subprocess pipe drain did not terminate"
        )


def _run_bounded_binary_subprocess(
    arguments: Sequence[str],
    *,
    timeout_seconds: float,
    output_limit_bytes: int,
    environment: Mapping[str, str],
    cwd: Path,
) -> _BoundedBinarySubprocessResult:
    if (
        isinstance(arguments, (str, bytes))
        or not arguments
        or any(not isinstance(argument, str) or not argument for argument in arguments)
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
        or type(output_limit_bytes) is not int
        or output_limit_bytes <= 0
    ):
        raise ValueError("bounded binary subprocess arguments differ")
    try:
        process = subprocess.Popen(
            list(arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            text=False,
            start_new_session=True,
            env=dict(environment),
            cwd=cwd,
        )
    except OSError:
        raise _BoundedSubprocessStartFailure(
            "bounded binary subprocess could not start"
        ) from None
    except (TypeError, ValueError):
        raise _BoundedSubprocessTransportFailure(
            "bounded binary subprocess launch contract failed"
        ) from None
    if process.stdout is None or process.stderr is None:
        _signal_bounded_subprocess_group(process, signal.SIGKILL)
        process.wait(timeout=_BOUNDED_SUBPROCESS_TERMINATE_GRACE_SECONDS)
        raise _BoundedSubprocessTransportFailure(
            "bounded binary subprocess pipes are unavailable"
        )

    streams = (process.stdout, process.stderr)
    limit_event = threading.Event()
    failure_event = threading.Event()
    stop_event = threading.Event()
    stdout_accumulator = _BoundedBinaryAccumulator(output_limit_bytes)
    stderr_accumulator = _BoundedBinaryAccumulator(output_limit_bytes)
    threads = (
        _bounded_stream_thread(
            process.stdout,
            stdout_accumulator,
            limit_event=limit_event,
            failure_event=failure_event,
            stop_event=stop_event,
            name="cachet-bounded-stdout",
        ),
        _bounded_stream_thread(
            process.stderr,
            stderr_accumulator,
            limit_event=limit_event,
            failure_event=failure_event,
            stop_event=stop_event,
            name="cachet-bounded-stderr",
        ),
    )
    timed_out = False
    try:
        for stream in streams:
            os.set_blocking(stream.fileno(), False)
        for thread in threads:
            thread.start()
        deadline = monotonic() + timeout_seconds
        while process.poll() is None:
            if failure_event.is_set():
                raise _BoundedSubprocessTransportFailure(
                    "bounded binary subprocess stream transport failed"
                )
            if limit_event.is_set():
                break
            remaining = deadline - monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                process.wait(timeout=min(_BOUNDED_SUBPROCESS_POLL_SECONDS, remaining))
            except subprocess.TimeoutExpired:
                continue
        _finish_bounded_subprocess(
            process,
            threads,
            streams,
            stop_event=stop_event,
            request_termination=timed_out or limit_event.is_set(),
        )
    except _BoundedSubprocessTransportFailure:
        _abort_bounded_subprocess(process, threads, streams, stop_event=stop_event)
        raise
    except BaseException:  # noqa: BLE001 - fixed transport failure below
        _abort_bounded_subprocess(process, threads, streams, stop_event=stop_event)
        raise _BoundedSubprocessTransportFailure(
            "bounded binary subprocess stream transport failed"
        ) from None
    if failure_event.is_set() or type(process.returncode) is not int:
        raise _BoundedSubprocessTransportFailure(
            "bounded binary subprocess did not close cleanly"
        )
    return _BoundedBinarySubprocessResult(
        returncode=process.returncode,
        stdout=stdout_accumulator.result(),
        stderr=stderr_accumulator.result(),
        timed_out=timed_out,
    )


def run_gpu_qualification_sentinel_v2(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
    work_dir: Path,
) -> Mapping[str, Any]:
    """Install the exact four-step runtime and run the bounded GPU worker."""

    if tuple(artifact_paths) != GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        raise ValueError("v2 sentinel artifacts lack canonical eight-role coverage")
    pins = pins_from_gpu_qualification_plan_v2(plan_record)
    validate_gpu_qualification_plan_v2_record(
        plan_record,
        expected_campaign_id=_required_string(
            plan_record.get("campaign_id"), "campaign_id"
        ),
        expected_artifact_pins=pins,
    )
    job_id = _required_string(planned_job.get("job_id"), "job_id")
    expected_job = qualification_v1._plan_job(plan_record, job_id)  # noqa: SLF001
    if canonical_gpu_qualification_json(planned_job) != (
        canonical_gpu_qualification_json(expected_job)
    ):
        raise ValueError("v2 planned job differs from the sealed plan")
    pin_record = pins.to_record()
    for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        path = artifact_paths[key]
        if key == "input_bundle_sha256":
            if not path.is_dir() or path.is_symlink():
                raise ValueError("v2 input bundle snapshot is not a regular directory")
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"v2 artifact {key} is not one regular file")
        if _file_sha256(path) != pin_record[key]:
            raise ValueError(f"v2 artifact {key} differs from the plan")
    closure_path = artifact_paths["runtime_closure_manifest_sha256"]
    _read_exact_runtime_closure(closure_path)
    system_cuda_parent_attestation = _capture_system_cuda_parent_attestation()

    runtime_dir = work_dir / "runtime"
    runtime_python = runtime_dir / "bin" / "python"
    expected_python_version = _runtime_python_version(plan_record)
    create_venv(runtime_dir, copies=True)
    created_python_identity = _attest_isolated_python(
        runtime_dir,
        expected_python_version=expected_python_version,
    )
    environment = _pip_subprocess_environment()
    environment.pop(_SYSTEM_CUDA_PARENT_ATTESTATION_ENV, None)
    environment.update(gpu_runtime_warning_environment_overrides())
    environment["PYTHONSAFEPATH"] = "1"
    environment[_SYSTEM_CUDA_PARENT_ATTESTATION_ENV] = (
        canonical_gpu_qualification_json(system_cuda_parent_attestation)
    )
    runtime_lock = artifact_paths["runtime_lock_sha256"]
    patched_vllm = artifact_paths["patched_vllm_wheel_sha256"]
    patched_flashinfer = artifact_paths["patched_flashinfer_wheel_sha256"]
    package_wheel = artifact_paths["package_wheel_sha256"]
    vllm_uri = patched_vllm.resolve(strict=True).as_uri()
    flashinfer_uri = patched_flashinfer.resolve(strict=True).as_uri()
    package_uri = package_wheel.resolve(strict=True).as_uri()
    commands = (
        (
            "base_lock",
            (
                "--require-hashes",
                "--only-binary",
                ":all:",
                "-r",
                str(runtime_lock),
            ),
        ),
        (
            "vllm",
            (
                "--no-deps",
                f"vllm @ {vllm_uri}#sha256={pins.patched_vllm_wheel_sha256}",
            ),
        ),
        (
            "flashinfer",
            (
                "--no-deps",
                "flashinfer-python @ "
                f"{flashinfer_uri}#sha256={pins.patched_flashinfer_wheel_sha256}",
            ),
        ),
        (
            "cachet",
            (
                "--no-deps",
                f"cachet-kv @ {package_uri}#sha256={pins.package_wheel_sha256}",
            ),
        ),
    )
    if tuple(label for label, _arguments in commands) != (
        "base_lock",
        "vllm",
        "flashinfer",
        "cachet",
    ):
        raise RuntimeError("v2 runtime install order differs")
    for _label, arguments in commands:
        subprocess.run(
            [
                str(runtime_python),
                "-m",
                "pip",
                "install",
                *arguments,
            ],
            check=True,
            timeout=3_600,
            env=environment,
            cwd=runtime_dir,
        )
    subprocess.run(
        [str(runtime_python), "-m", "pip", "check"],
        check=True,
        timeout=300,
        env=environment,
        cwd=runtime_dir,
    )
    installed_python_identity = _attest_isolated_python(
        runtime_dir,
        expected_python_version=expected_python_version,
        expected_file_binding=created_python_identity.file_binding,
    )
    if installed_python_identity != created_python_identity:
        raise RuntimeError("v2 isolated Python identity changed during installation")

    launch_environment = _runtime_launch_environment(
        runtime_dir=runtime_dir,
        install_environment=environment,
    )
    attestation = _run_final_runtime_verifier(
        runtime_python,
        runtime_lock=runtime_lock,
        vllm_uri=vllm_uri,
        flashinfer_uri=flashinfer_uri,
        closure_path=closure_path,
        package_uri=package_uri,
        package_sha256=pins.package_wheel_sha256,
        environment=launch_environment,
    )
    validate_gpu_qualification_v2_runtime_attestation(attestation)
    os.environ[VLLM_PATCHED_WHEEL_URI_ENV] = str(patched_vllm)
    os.environ[VLLM_PATCHED_WHEEL_SHA256_ENV] = pins.patched_vllm_wheel_sha256

    worker_environment = dict(launch_environment)
    worker_environment.update(
        {
            "HF_HOME": "/local_disk0/cachet-vllm-0271-hf",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONSAFEPATH": "1",
            "TOKENIZERS_PARALLELISM": "false",
            _SYSTEM_CUDA_PARENT_ATTESTATION_ENV: (
                canonical_gpu_qualification_json(system_cuda_parent_attestation)
            ),
            VLLM_PATCHED_WHEEL_URI_ENV: str(patched_vllm),
            VLLM_PATCHED_WHEEL_SHA256_ENV: pins.patched_vllm_wheel_sha256,
            _RUNTIME_LOCK_ATTESTATION_ENV: json.dumps(
                attestation,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
    )
    _verify_input_bundle_in_isolated_runtime(
        runtime_python,
        artifact_paths["input_bundle_sha256"],
        expected_sha256=pins.input_bundle_sha256,
        environment=worker_environment,
    )

    worker_dir = work_dir / "worker"
    worker_dir.mkdir()
    job_path = worker_dir / "planned-job.json"
    job_path.write_text(
        canonical_gpu_qualification_json(planned_job) + "\n", encoding="utf-8"
    )
    plan_path = worker_dir / "plan.json"
    plan_path.write_text(
        canonical_gpu_qualification_json(plan_record) + "\n", encoding="utf-8"
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
        environment=worker_environment,
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
        raise RuntimeError("GPU sentinel v2 measurement output must be an object")
    runtime_verification = build_gpu_runtime_verification_v2(
        plan_sha256=_required_string(
            plan_record.get("closed_record_sha256"), "plan_sha256"
        ),
        job_id=job_id,
        artifact_sha256=pin_record,
        attestation=attestation,
    )
    return {
        "measurements": measurements,
        "runtime_verification": runtime_verification,
    }


def _runtime_launch_environment(
    *,
    runtime_dir: Path,
    install_environment: Mapping[str, str],
) -> dict[str, str]:
    """Bind post-install children to the exact private torch library directory."""

    error = "v2 isolated torch library directory differs"
    if not runtime_dir.is_absolute() or ".." in runtime_dir.parts:
        raise RuntimeError(error)
    torch_library_dir = runtime_dir / _RUNTIME_TORCH_LIBRARY_RELATIVE_PATH
    try:
        resolved_runtime_dir = runtime_dir.resolve(strict=True)
        resolved_torch_library_dir = torch_library_dir.resolve(strict=True)
        if (
            resolved_runtime_dir != runtime_dir
            or not runtime_dir.is_dir()
            or resolved_torch_library_dir != torch_library_dir
            or not torch_library_dir.is_dir()
            or torch_library_dir.is_symlink()
            or not resolved_torch_library_dir.is_relative_to(resolved_runtime_dir)
            or os.pathsep in str(torch_library_dir)
        ):
            raise RuntimeError(error)
        directory_descriptor = _open_runtime_root_no_follow(torch_library_dir)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(error) from exc
    try:
        os.close(directory_descriptor)
    except OSError as exc:
        raise RuntimeError(error) from exc

    launch_environment = dict(install_environment)
    existing_library_path = launch_environment.get("LD_LIBRARY_PATH")
    launch_environment["LD_LIBRARY_PATH"] = str(torch_library_dir)
    if existing_library_path:
        launch_environment["LD_LIBRARY_PATH"] += os.pathsep + existing_library_path
    return launch_environment


def verify_gpu_qualification_v2_runtime_installation(
    *,
    runtime_lock: str | Path,
    vllm_uri: str,
    flashinfer_uri: str,
    runtime_closure_manifest: str | Path,
    package_uri: str,
    package_sha256: str,
) -> dict[str, Any]:
    """Run the final Linux/CPython/install/provenance/import verifier."""

    return _verify_gpu_qualification_v2_runtime_installation(
        runtime_lock=runtime_lock,
        vllm_uri=vllm_uri,
        flashinfer_uri=flashinfer_uri,
        runtime_closure_manifest=runtime_closure_manifest,
        package_uri=package_uri,
        package_sha256=package_sha256,
        stage_callback=None,
    )


def _verify_gpu_qualification_v2_runtime_installation(
    *,
    runtime_lock: str | Path,
    vllm_uri: str,
    flashinfer_uri: str,
    runtime_closure_manifest: str | Path,
    package_uri: str,
    package_sha256: str,
    stage_callback: Callable[[str], None] | None,
) -> dict[str, Any]:
    def enter_stage(stage: str) -> None:
        if stage not in _FINAL_VERIFIER_STAGES:
            raise AssertionError("invalid final runtime verifier stage")
        if stage_callback is not None:
            stage_callback(stage)

    enter_stage("platform")
    _require_runtime_platform()
    enter_stage("pip_check")
    verifier_environment = _pip_subprocess_environment()
    verifier_environment.update(gpu_runtime_warning_environment_overrides())
    verifier_environment["PYTHONSAFEPATH"] = "1"
    try:
        pip_check = _run_bounded_binary_subprocess(
            [sys.executable, "-m", "pip", "check"],
            timeout_seconds=_FINAL_VERIFIER_INNER_PIP_TIMEOUT_SECONDS,
            output_limit_bytes=_FINAL_VERIFIER_PROCESS_OUTPUT_LIMIT_BYTES,
            environment=verifier_environment,
            cwd=Path(sys.prefix),
        )
    except _BoundedSubprocessStartFailure:
        raise _FinalRuntimeVerifierFailure(
            "subprocess_start_failure",
            "v2 installed runtime pip check could not start",
        ) from None
    except _BoundedSubprocessTransportFailure:
        raise _FinalRuntimeVerifierFailure(
            "unexpected_exception", "v2 installed runtime pip check transport failed"
        ) from None
    if pip_check.timed_out:
        raise _FinalRuntimeVerifierFailure(
            "subprocess_timeout", "v2 installed runtime pip check timed out"
        )
    if pip_check.output_limit_exceeded:
        raise _FinalRuntimeVerifierFailure(
            "verification_rejected", "v2 installed runtime pip check output differs"
        )
    if not _bounded_stream_result_is_exact(
        pip_check.stdout
    ) or not _bounded_stream_result_is_exact(pip_check.stderr):
        raise _FinalRuntimeVerifierFailure(
            "unexpected_exception", "v2 installed runtime pip check transport failed"
        )
    if pip_check.returncode != 0:
        raise _FinalRuntimeVerifierFailure(
            "subprocess_nonzero", "v2 installed runtime pip check failed"
        )
    if (
        pip_check.stdout.retained != _FINAL_VERIFIER_PIP_CHECK_STDOUT
        or pip_check.stdout.byte_count != len(_FINAL_VERIFIER_PIP_CHECK_STDOUT)
        or pip_check.stderr.retained != b""
        or pip_check.stderr.byte_count != 0
    ):
        raise _FinalRuntimeVerifierFailure(
            "verification_rejected", "v2 installed runtime pip check output differs"
        )
    pip_check_ok = True
    enter_stage("base_lock")
    lock_path = Path(runtime_lock)
    if _file_sha256(lock_path) != VLLM_RUNTIME_BASE_LOCK_SHA256:
        raise RuntimeError("v2 installed runtime base lock SHA-256 differs")
    versions, hash_count = _base_lock_projection(lock_path)
    if (
        len(versions) != VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT
        or hash_count != VLLM_RUNTIME_BASE_LOCK_HASH_COUNT
        or "flashinfer-python" in versions
        or "vllm" in versions
        or "cachet-kv" in versions
    ):
        raise RuntimeError("v2 installed runtime base lock closure differs")
    enter_stage("runtime_closure")
    closure = _read_exact_runtime_closure(Path(runtime_closure_manifest))

    enter_stage("distribution_inventory")
    site_packages = _isolated_runtime_site_packages()
    installed: dict[str, list[str]] = {}
    distributions: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        _require_distribution_root(distribution, site_packages=site_packages)
        try:
            raw_name = distribution.metadata["Name"]
        except KeyError as exc:
            raise RuntimeError("installed distribution has no package name") from exc
        if not isinstance(raw_name, str) or not raw_name:
            raise RuntimeError("installed distribution has an invalid package name")
        name = _canonical_name(raw_name)
        installed.setdefault(name, []).append(distribution.version)
        if name in distributions:
            raise RuntimeError(f"duplicate installed distribution {name!r}")
        distributions[name] = distribution
    expected_names = {
        *versions,
        "cachet-kv",
        "flashinfer-python",
        "vllm",
    }
    unexpected = sorted(set(installed) - expected_names)
    if unexpected or set(installed) != expected_names:
        raise RuntimeError("v2 installed distribution name closure differs")
    for name, expected_version in versions.items():
        if installed.get(name) != [expected_version]:
            raise RuntimeError(f"v2 installed base distribution {name!r} differs")
    if len(installed) != GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT:
        raise RuntimeError("v2 installed distribution count differs")
    if installed.get("vllm") != [VLLM_PACKAGE_VERSION]:
        raise RuntimeError("v2 installed vLLM version differs")
    if installed.get("flashinfer-python") != [FLASHINFER_PACKAGE_VERSION]:
        raise RuntimeError("v2 installed FlashInfer version differs")
    cachet_versions = installed.get("cachet-kv")
    if cachet_versions != [GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION]:
        raise RuntimeError("v2 installed Cachet package identity differs")

    enter_stage("package_origin")
    _require_sha256(package_sha256, "package_sha256")
    if _uri_file_sha256(package_uri) != package_sha256:
        raise RuntimeError("v2 Cachet package URI bytes differ")
    enter_stage("vllm_origin")
    vllm_direct_url = _validate_direct_url(
        distributions["vllm"],
        expected_uri=vllm_uri,
        expected_sha256=VLLM_PATCHED_WHEEL_SHA256,
    )
    enter_stage("flashinfer_origin")
    flashinfer_direct_url = _validate_direct_url(
        distributions["flashinfer-python"],
        expected_uri=flashinfer_uri,
        expected_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
    )
    enter_stage("package_origin")
    _validate_direct_url(
        distributions["cachet-kv"],
        expected_uri=package_uri,
        expected_sha256=package_sha256,
    )
    enter_stage("vllm_members")
    vllm_members = _installed_member_hashes(distributions["vllm"], _VLLM_MEMBER_SHA256)
    enter_stage("flashinfer_members")
    flashinfer_members = _installed_member_hashes(
        distributions["flashinfer-python"],
        {FLASHINFER_TARGET_MEMBER: FLASHINFER_TARGET_PATCHED_SHA256},
    )
    if flashinfer_members[FLASHINFER_TARGET_MEMBER] != (
        FLASHINFER_TARGET_PATCHED_SHA256
    ):
        raise RuntimeError("v2 installed FlashInfer patched member differs")
    enter_stage("flashinfer_import")
    module = importlib.import_module("flashinfer.comm.fd_exchange")
    module_file = getattr(module, "__file__", None)
    expected_module_file = (
        Path(str(distributions["flashinfer-python"].locate_file("")))
        / FLASHINFER_TARGET_MEMBER
    )
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError("v2 imported FlashInfer module has no source origin")
    try:
        observed_module_file = Path(module_file).resolve(strict=True)
        reviewed_module_file = expected_module_file.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "v2 imported FlashInfer module origin is unavailable"
        ) from exc
    if (
        observed_module_file != reviewed_module_file
        or _file_sha256(observed_module_file) != FLASHINFER_TARGET_PATCHED_SHA256
    ):
        raise RuntimeError("v2 imported FlashInfer module origin differs")
    function = getattr(module, "_fd_ancillary", None)
    annotations = getattr(function, "__annotations__", None)
    if not isinstance(annotations, dict) or annotations.get("return") != (
        GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION
    ):
        raise RuntimeError("v2 FlashInfer postponed return annotation differs")
    enter_stage("packaged_lock")
    packaged_lock = (
        importlib.resources.files("document_kv_cache")
        .joinpath("runtime_locks")
        .joinpath(VLLM_RUNTIME_BASE_LOCK_FILENAME)
        .read_bytes()
    )
    packaged_lock_sha256 = sha256(packaged_lock).hexdigest()
    if packaged_lock_sha256 != VLLM_RUNTIME_BASE_LOCK_SHA256:
        raise RuntimeError("v2 installed Cachet package base lock differs")
    enter_stage("attestation")
    record = {
        "base_lock_distribution_count": len(versions),
        "base_lock_hash_count": hash_count,
        "base_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "cachet_package_version": cachet_versions[0],
        "flashinfer_annotation": GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION,
        "flashinfer_direct_url": flashinfer_direct_url,
        "flashinfer_import_ok": True,
        "closure_bound_flashinfer_manifest_closed_record_sha256": (
            FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
        ),
        "closure_bound_flashinfer_manifest_file_sha256": (
            FLASHINFER_PATCHED_MANIFEST_FILE_SHA256
        ),
        "closure_bound_vllm_manifest_file_sha256": VLLM_PATCHED_MANIFEST_SHA256,
        "flashinfer_member_sha256": flashinfer_members[FLASHINFER_TARGET_MEMBER],
        "flashinfer_package_version": FLASHINFER_PACKAGE_VERSION,
        "flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
        "installed_distribution_count": len(installed),
        "ok": True,
        "packaged_base_lock_sha256": packaged_lock_sha256,
        "pip_check_ok": pip_check_ok,
        "runtime_closure_closed_record_sha256": closure["closed_record_sha256"],
        "runtime_closure_file_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "system_cuda_parent_attestation": (
            _system_cuda_parent_attestation_from_environment()
        ),
        "unexpected_distributions": unexpected,
        "vllm_direct_url": vllm_direct_url,
        "vllm_member_sha256": vllm_members,
        "vllm_package_version": VLLM_PACKAGE_VERSION,
        "vllm_wheel_sha256": VLLM_PATCHED_WHEEL_SHA256,
        "with_flashinfer_distribution_count": (
            GPU_QUALIFICATION_V2_WITH_FLASHINFER_DISTRIBUTION_COUNT
        ),
        "with_vllm_distribution_count": (
            GPU_QUALIFICATION_V2_WITH_VLLM_DISTRIBUTION_COUNT
        ),
    }
    validate_gpu_qualification_v2_runtime_attestation(record)
    enter_stage("complete")
    return record


def _run_final_runtime_verifier(
    runtime_python: Path,
    *,
    runtime_lock: Path,
    vllm_uri: str,
    flashinfer_uri: str,
    closure_path: Path,
    package_uri: str,
    package_sha256: str,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    _require_final_verifier_timeout_hierarchy()
    code = (
        "import sys;"
        "from document_kv_cache._gpu_qualification_sentinels_v2 import "
        "_final_runtime_verifier_child_main as main;"
        "raise SystemExit(main(sys.argv[1:]))"
    )
    try:
        completed = _run_bounded_binary_subprocess(
            [
                str(runtime_python),
                "-c",
                code,
                str(runtime_lock),
                vllm_uri,
                flashinfer_uri,
                str(closure_path),
                package_uri,
                package_sha256,
            ],
            timeout_seconds=_FINAL_VERIFIER_OUTER_TIMEOUT_SECONDS,
            output_limit_bytes=_FINAL_VERIFIER_PROCESS_OUTPUT_LIMIT_BYTES,
            environment=environment,
            cwd=runtime_python.parent.parent,
        )
    except _BoundedSubprocessStartFailure:
        raise RuntimeError(
            "v2 final runtime verifier process could not start"
        ) from None
    except _BoundedSubprocessTransportFailure:
        raise RuntimeError("v2 final runtime verifier transport failed") from None
    if completed.timed_out:
        raise RuntimeError("v2 final runtime verifier process timed out")
    if completed.output_limit_exceeded:
        raise RuntimeError("v2 final runtime verifier output exceeds its limit")
    if not _bounded_stream_result_is_exact(
        completed.stdout
    ) or not _bounded_stream_result_is_exact(completed.stderr):
        raise RuntimeError("v2 final runtime verifier transport failed")
    if completed.returncode != 0:
        raise RuntimeError("v2 final runtime verifier process failed")
    if completed.stderr.byte_count != 0 or completed.stderr.retained != b"":
        raise RuntimeError("v2 final runtime verifier protocol failed")
    envelope = _parse_final_runtime_verifier_child_envelope(completed.stdout.retained)
    if not envelope["ok"]:
        raise RuntimeError(
            "v2 final runtime verifier rejected the installation "
            f"({envelope['stage']}/{envelope['category']}; "
            f"stdout_bytes={envelope['stdout_bytes']}; "
            f"stdout_sha256={envelope['stdout_sha256']}; "
            f"stderr_bytes={envelope['stderr_bytes']}; "
            f"stderr_sha256={envelope['stderr_sha256']})"
        )
    return cast(dict[str, Any], envelope["attestation"])


def _final_runtime_verifier_child_main(arguments: Sequence[str]) -> int:
    """Emit one bounded canonical envelope without exception or stream text."""

    try:
        envelope = _final_runtime_verifier_child_envelope(arguments)
        encoded = _canonical_final_runtime_verifier_child_envelope(envelope)
    except BaseException:  # noqa: BLE001 - the child must never serialize exceptions
        envelope = _final_runtime_verifier_failure_envelope(
            stage="arguments",
            category="unexpected_exception",
            stdout_bytes=0,
            stdout_sha256=_FINAL_VERIFIER_EMPTY_STREAM_SHA256,
            stderr_bytes=0,
            stderr_sha256=_FINAL_VERIFIER_EMPTY_STREAM_SHA256,
        )
        encoded = _canonical_final_runtime_verifier_child_envelope(envelope)
    if len(encoded) > _FINAL_VERIFIER_PROCESS_OUTPUT_LIMIT_BYTES:
        encoded = _canonical_final_runtime_verifier_child_envelope(
            _final_runtime_verifier_failure_envelope(
                stage="attestation",
                category="verification_rejected",
                stdout_bytes=0,
                stdout_sha256=_FINAL_VERIFIER_EMPTY_STREAM_SHA256,
                stderr_bytes=0,
                stderr_sha256=_FINAL_VERIFIER_EMPTY_STREAM_SHA256,
            )
        )
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(1, encoded[offset:])
            if written <= 0:
                return 1
            offset += written
    except OSError:
        return 1
    return 0


def _final_runtime_verifier_child_envelope(
    arguments: Sequence[str],
) -> dict[str, Any]:
    stage = "arguments"
    category = "none"
    attestation: dict[str, Any] | None = None
    succeeded = False
    stdout_accumulator = _BoundedBinaryAccumulator(
        _FINAL_VERIFIER_PROCESS_OUTPUT_LIMIT_BYTES
    )
    stderr_accumulator = _BoundedBinaryAccumulator(
        _FINAL_VERIFIER_PROCESS_OUTPUT_LIMIT_BYTES
    )
    limit_event = threading.Event()
    failure_event = threading.Event()
    stop_event = threading.Event()
    stdout_read, stdout_write = os.pipe()
    stderr_read, stderr_write = os.pipe()
    stdout_stream = os.fdopen(stdout_read, "rb", buffering=0)
    stderr_stream = os.fdopen(stderr_read, "rb", buffering=0)
    streams = (stdout_stream, stderr_stream)
    threads = (
        _bounded_stream_thread(
            stdout_stream,
            stdout_accumulator,
            limit_event=limit_event,
            failure_event=failure_event,
            stop_event=stop_event,
            name="cachet-final-verifier-stdout",
        ),
        _bounded_stream_thread(
            stderr_stream,
            stderr_accumulator,
            limit_event=limit_event,
            failure_event=failure_event,
            stop_event=stop_event,
            name="cachet-final-verifier-stderr",
        ),
    )
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        for stream in streams:
            os.set_blocking(stream.fileno(), False)
        for thread in threads:
            thread.start()
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_write, 1, inheritable=False)
        os.dup2(stderr_write, 2, inheritable=False)
        os.close(stdout_write)
        stdout_write = -1
        os.close(stderr_write)
        stderr_write = -1
        try:
            values = _final_runtime_verifier_arguments(arguments)

            def observe_stage(value: str) -> None:
                nonlocal stage
                stage = value

            attestation = _verify_gpu_qualification_v2_runtime_installation(
                runtime_lock=values[0],
                vllm_uri=values[1],
                flashinfer_uri=values[2],
                runtime_closure_manifest=values[3],
                package_uri=values[4],
                package_sha256=values[5],
                stage_callback=observe_stage,
            )
            if not isinstance(attestation, dict):
                raise _FinalRuntimeVerifierFailure(
                    "verification_rejected",
                    "v2 final runtime verifier attestation is not an object",
                )
            succeeded = True
        except _FinalRuntimeVerifierFailure as exc:
            category = exc.category
        except subprocess.CalledProcessError:
            category = (
                "subprocess_nonzero"
                if stage == "pip_check"
                else "verification_rejected"
            )
        except subprocess.TimeoutExpired:
            category = (
                "subprocess_timeout"
                if stage == "pip_check"
                else "verification_rejected"
            )
        except (ImportError, OSError, RuntimeError, ValueError):
            category = "verification_rejected"
        except BaseException:  # noqa: BLE001 - finite child failure envelope
            category = "unexpected_exception"
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except BaseException:  # noqa: BLE001 - fixed stream classification below
            failure_event.set()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        for descriptor in (stdout_write, stderr_write):
            if descriptor >= 0:
                os.close(descriptor)
    if not _join_bounded_stream_threads(
        threads, timeout_seconds=_BOUNDED_STREAM_JOIN_SECONDS
    ):
        _stop_bounded_stream_threads(threads, streams, stop_event=stop_event)
        failure_event.set()
    stdout_result = stdout_accumulator.result()
    stderr_result = stderr_accumulator.result()
    stdout_bytes = stdout_result.byte_count
    stdout_sha256 = stdout_result.sha256
    stderr_bytes = stderr_result.byte_count
    stderr_sha256 = stderr_result.sha256
    if failure_event.is_set():
        succeeded = False
        stage = "attestation"
        category = "unexpected_exception"
        attestation = None
    if succeeded and (stdout_bytes != 0 or stderr_bytes != 0):
        succeeded = False
        stage = "attestation"
        category = "verification_rejected"
        attestation = None
    if not succeeded and stage == "complete":
        stage = "attestation"
        if category in {
            "subprocess_nonzero",
            "subprocess_start_failure",
            "subprocess_timeout",
        }:
            category = "verification_rejected"
    if succeeded:
        return {
            "attestation": attestation,
            "category": "none",
            "ok": True,
            "record_type": _FINAL_VERIFIER_CHILD_RECORD_TYPE,
            "schema_version": _FINAL_VERIFIER_CHILD_SCHEMA_VERSION,
            "stage": "complete",
            "stderr_bytes": stderr_bytes,
            "stderr_sha256": stderr_sha256,
            "stdout_bytes": stdout_bytes,
            "stdout_sha256": stdout_sha256,
        }
    return _final_runtime_verifier_failure_envelope(
        stage=stage,
        category=category,
        stdout_bytes=stdout_bytes,
        stdout_sha256=stdout_sha256,
        stderr_bytes=stderr_bytes,
        stderr_sha256=stderr_sha256,
    )


def _final_runtime_verifier_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    if isinstance(arguments, (str, bytes)):
        raise _FinalRuntimeVerifierFailure(
            "invalid_arguments", "v2 final runtime verifier arguments differ"
        )
    values = tuple(arguments)
    if len(values) != 6 or any(
        not isinstance(value, str) or not value or value.strip() != value
        for value in values
    ):
        raise _FinalRuntimeVerifierFailure(
            "invalid_arguments", "v2 final runtime verifier arguments differ"
        )
    try:
        _require_sha256(values[5], "package_sha256")
    except ValueError:
        raise _FinalRuntimeVerifierFailure(
            "invalid_arguments", "v2 final runtime verifier arguments differ"
        ) from None
    return values


def _final_runtime_verifier_failure_envelope(
    *,
    stage: str,
    category: str,
    stdout_bytes: int,
    stdout_sha256: str,
    stderr_bytes: int,
    stderr_sha256: str,
) -> dict[str, Any]:
    return {
        "attestation": None,
        "category": category,
        "ok": False,
        "record_type": _FINAL_VERIFIER_CHILD_RECORD_TYPE,
        "schema_version": _FINAL_VERIFIER_CHILD_SCHEMA_VERSION,
        "stage": stage,
        "stderr_bytes": stderr_bytes,
        "stderr_sha256": stderr_sha256,
        "stdout_bytes": stdout_bytes,
        "stdout_sha256": stdout_sha256,
    }


def _canonical_final_runtime_verifier_child_envelope(
    record: Mapping[str, Any],
) -> bytes:
    return (
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _parse_final_runtime_verifier_child_envelope(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > _FINAL_VERIFIER_PROCESS_OUTPUT_LIMIT_BYTES:
        raise RuntimeError("v2 final runtime verifier protocol failed")
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RuntimeError("v2 final runtime verifier protocol failed") from None
    if not isinstance(value, dict):
        raise RuntimeError("v2 final runtime verifier protocol failed")
    try:
        canonical = _canonical_final_runtime_verifier_child_envelope(value)
    except (TypeError, ValueError):
        raise RuntimeError("v2 final runtime verifier protocol failed") from None
    if raw != canonical or frozenset(value) != _FINAL_VERIFIER_CHILD_KEYS:
        raise RuntimeError("v2 final runtime verifier protocol failed")
    if (
        value.get("record_type") != _FINAL_VERIFIER_CHILD_RECORD_TYPE
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != _FINAL_VERIFIER_CHILD_SCHEMA_VERSION
        or type(value.get("ok")) is not bool
        or value.get("stage") not in _FINAL_VERIFIER_STAGES
        or value.get("category") not in _FINAL_VERIFIER_CATEGORIES
    ):
        raise RuntimeError("v2 final runtime verifier protocol failed")
    for stream_name in ("stdout", "stderr"):
        byte_count = value.get(f"{stream_name}_bytes")
        digest = value.get(f"{stream_name}_sha256")
        if (
            type(byte_count) is not int
            or not 0 <= byte_count <= _FINAL_VERIFIER_CAPTURE_COUNT_LIMIT
            or not _is_sha256(digest)
            or (byte_count == 0) != (digest == _FINAL_VERIFIER_EMPTY_STREAM_SHA256)
        ):
            raise RuntimeError("v2 final runtime verifier protocol failed")
    if value["ok"]:
        if (
            value["category"] != "none"
            or value["stage"] != "complete"
            or not isinstance(value["attestation"], dict)
            or value["stdout_bytes"] != 0
            or value["stderr_bytes"] != 0
        ):
            raise RuntimeError("v2 final runtime verifier protocol failed")
    else:
        allowed_categories = _FINAL_VERIFIER_FAILURE_RELATIONS.get(value["stage"])
        if (
            value["attestation"] is not None
            or allowed_categories is None
            or value["category"] not in allowed_categories
        ):
            raise RuntimeError("v2 final runtime verifier protocol failed")
    return cast(dict[str, Any], value)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(_value: str) -> Any:
    raise ValueError("non-finite JSON number")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError("non-finite JSON number")
    return parsed


def _read_exact_runtime_closure(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("v2 runtime closure is not one regular file")
    raw = path.read_bytes()
    if (
        len(raw) != RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE
        or sha256(raw).hexdigest() != RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
    ):
        raise ValueError("v2 runtime closure file identity differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("v2 runtime closure is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("v2 runtime closure must contain an object")
    expected_bytes = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if raw != expected_bytes:
        raise ValueError("v2 runtime closure is not canonical JSON")
    if (
        value.get("record_type") != RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE
        or value.get("schema_version") != RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION
        or value.get("closed_record_sha256")
        != RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
    ):
        raise ValueError("v2 runtime closure record identity differs")
    payload = dict(value)
    payload.pop("closed_record_sha256")
    if (
        sha256(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        != RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
    ):
        raise ValueError("v2 runtime closure seal differs")
    return value


def _base_lock_projection(path: Path) -> tuple[dict[str, str], int]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("v2 base lock is not UTF-8") from exc
    versions: dict[str, str] = {}
    hash_count = 0
    for line in text.splitlines():
        requirement = _BASE_LOCK_REQUIREMENT_RE.fullmatch(line)
        if requirement is not None:
            name = _canonical_name(requirement.group(1))
            if name in versions:
                raise ValueError("v2 base lock repeats a distribution")
            versions[name] = requirement.group(2)
        if _BASE_LOCK_HASH_RE.fullmatch(line) is not None:
            hash_count += 1
    return versions, hash_count


def _isolated_runtime_site_packages() -> Path:
    """Bind metadata discovery and this verifier to the copied private runtime."""

    if not all(
        isinstance(value, str) and value
        for value in (sys.prefix, sys.base_prefix, sys.executable, __file__)
    ):
        raise RuntimeError("v2 installed runtime path identity differs")
    runtime_root = Path(sys.prefix)
    runtime_python = Path(sys.executable)
    if (
        not runtime_root.is_absolute()
        or ".." in runtime_root.parts
        or str(runtime_root) != sys.prefix
        or runtime_python != runtime_root / "bin" / "python"
        or sys.prefix == sys.base_prefix
    ):
        raise RuntimeError("v2 installed runtime path identity differs")
    try:
        if (
            runtime_root.resolve(strict=True) != runtime_root
            or not runtime_root.is_dir()
            or runtime_python.resolve(strict=True) != runtime_python
            or not runtime_python.is_file()
            or runtime_python.is_symlink()
        ):
            raise RuntimeError("v2 installed runtime path identity differs")
    except OSError as exc:
        raise RuntimeError("v2 installed runtime path identity differs") from exc

    site_packages = runtime_root / "lib/python3.11/site-packages"
    verifier_source = site_packages / (
        "document_kv_cache/_gpu_qualification_sentinels_v2.py"
    )
    try:
        if (
            site_packages.resolve(strict=True) != site_packages
            or not site_packages.is_dir()
            or site_packages.is_symlink()
            or Path(__file__) != verifier_source
            or verifier_source.resolve(strict=True) != verifier_source
            or not verifier_source.is_file()
            or verifier_source.is_symlink()
        ):
            raise RuntimeError("v2 installed runtime path identity differs")
    except OSError as exc:
        raise RuntimeError("v2 installed runtime path identity differs") from exc
    return site_packages


def _require_distribution_root(
    distribution: importlib.metadata.Distribution,
    *,
    site_packages: Path,
) -> None:
    """Reject metadata records that do not belong to the bound private root."""

    root = Path(str(distribution.locate_file("")))
    try:
        if root != site_packages or root.resolve(strict=True) != site_packages:
            raise RuntimeError(
                "v2 installed distribution is outside private site-packages"
            )
    except OSError as exc:
        raise RuntimeError(
            "v2 installed distribution is outside private site-packages"
        ) from exc


def _validate_direct_url(
    distribution: importlib.metadata.Distribution,
    *,
    expected_uri: str,
    expected_sha256: str,
) -> str:
    if _uri_file_sha256(expected_uri) != expected_sha256:
        raise RuntimeError("v2 direct-install source bytes differ")
    text = distribution.read_text("direct_url.json")
    if text is None:
        raise RuntimeError("v2 direct install lacks PEP 610 direct_url.json")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("v2 PEP 610 direct_url.json is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError("v2 PEP 610 direct_url.json must contain an object")
    observed_uri = value.get("url")
    expected_canonical_uri = _normalized_uri(expected_uri)
    if not isinstance(observed_uri, str) or _normalized_uri(observed_uri) != (
        expected_canonical_uri
    ):
        raise RuntimeError("v2 PEP 610 direct URL differs")
    archive = value.get("archive_info")
    hashes = archive.get("hashes") if isinstance(archive, dict) else None
    if not isinstance(hashes, dict) or hashes.get("sha256") != expected_sha256:
        raise RuntimeError("v2 PEP 610 archive SHA-256 differs")
    return expected_canonical_uri


def _installed_member_hashes(
    distribution: importlib.metadata.Distribution,
    expected: Mapping[str, str],
) -> dict[str, str]:
    root = Path(str(distribution.locate_file("")))
    observed: dict[str, str] = {}
    for relative_path, expected_sha256 in expected.items():
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"v2 installed member is missing: {relative_path}")
        digest = _file_sha256(path)
        if digest != expected_sha256:
            raise RuntimeError(f"v2 installed member differs: {relative_path}")
        observed[relative_path] = digest
    return observed


def _require_runtime_platform() -> None:
    if (
        platform.python_implementation() != "CPython"
        or sys.version_info[:2] != (3, 11)
        or platform.python_version() != "3.11.11"
        or sys.platform != "linux"
        or platform.machine().lower() not in {"x86_64", "amd64"}
        or platform.libc_ver() != ("glibc", "2.35")
    ):
        raise RuntimeError(
            "v2 final runtime verifier requires Linux CPython3.11 x86_64 glibc2.35"
        )
    if (
        os.environ.get("PYTHONWARNINGS") != GPU_RUNTIME_PYTHONWARNINGS
        or tuple(sys.warnoptions) != tuple(GPU_RUNTIME_PYTHONWARNINGS.split(","))
        or os.environ.get("FLASHINFER_LOGGING_LEVEL")
        != GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL
    ):
        raise RuntimeError(
            "v2 final runtime verifier requires the pinned CUDA warning startup policy"
        )


def _runtime_python_version(plan_record: Mapping[str, Any]) -> str:
    runtime = plan_record.get("runtime_contract")
    if not isinstance(runtime, Mapping):
        raise ValueError("v2 plan runtime_contract must be an object")
    platform_record = runtime.get("platform")
    if not isinstance(platform_record, Mapping):
        raise ValueError("v2 plan runtime platform must be an object")
    return _required_string(platform_record.get("python_version"), "python_version")


def _normalized_uri(value: str) -> str:
    parsed = urlsplit(value)
    decoded_path = unquote(parsed.path)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
        or decoded_path.startswith("//")
        or any(
            ord(character) < 32 or ord(character) == 127 for character in decoded_path
        )
        or any(part in {".", ".."} for part in decoded_path.split("/"))
    ):
        raise RuntimeError("v2 direct runtime origin must be a local file URI")
    path = Path(decoded_path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError("v2 direct runtime origin must be one regular file")
    return path.resolve(strict=True).as_uri()


def _uri_file_sha256(value: str) -> str:
    normalized = _normalized_uri(value)
    parsed = urlsplit(normalized)
    path = Path(unquote(parsed.path))
    return _file_sha256(path)


def _canonical_name(value: str) -> str:
    return _CANONICAL_NAME_RE.sub("-", value).lower()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


def _require_sha256(value: Any, field_name: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{field_name} must be lowercase SHA-256")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "run_gpu_qualification_sentinel_v2",
    "verify_gpu_qualification_v2_runtime_installation",
]
