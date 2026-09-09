"""Render internal fail-closed commands for canonical runtime-verifier protocols.

This module is deliberately import-safe: it imports only the Python standard
library and does not execute package installation metadata or ``site`` hooks.
The rendered interpreter starts with ``-I -S -B``, restores exactly one CPython
3.11 virtual-environment ``site-packages`` directory without processing
``.pth`` files, and captures package output before releasing a canonical
protocol payload.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final


ISOLATED_RUNTIME_PIP_CHECK_EXECUTION_TIMEOUT_SECONDS: Final = 170.0
ISOLATED_RUNTIME_FINAL_CHILD_EXECUTION_TIMEOUT_SECONDS: Final = 280.0
ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS: Final = 350.0
ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS: Final = 100.0
ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS: Final = 100.0

_MAX_EXECUTION_TIMEOUT_SECONDS: Final = 3_600.0

_RUNTIME_VERIFIER_MODULE: Final = (
    "document_kv_cache._gpu_qualification_sentinels_v2"
)
_RUNTIME_VERIFIER_MODULE_FILE: Final = (
    "document_kv_cache/_gpu_qualification_sentinels_v2.py"
)
_RUNTIME_VALIDATOR_MODULE: Final = "document_kv_cache.gpu_qualification_v2"
_RUNTIME_VALIDATOR_MODULE_FILE: Final = "document_kv_cache/gpu_qualification_v2.py"
_PIP_CHECK_TARGET: Final = (
    "pip._internal.cli.main",
    "pip/_internal/cli/main.py",
    "main",
)
_NATIVE_LOADER_TARGET: Final = (
    "document_kv_cache._runtime_bootstrap_native_loader",
    "document_kv_cache/_runtime_bootstrap_native_loader.py",
    "main",
)
_NATIVE_LOADER_GPU_NAMES: Final = frozenset({"NVIDIA L40S", "NVIDIA L4"})

_VERIFIER_TARGETS: Final = {
    "locked_runtime": "verify_locked_runtime_v2_package_installation",
    "gpu_qualification": "verify_gpu_qualification_v2_runtime_installation",
}
_VALIDATOR_TARGETS: Final = {
    "locked_runtime": "validate_locked_runtime_v2_package_installation_attestation",
    "gpu_qualification": "validate_gpu_qualification_v2_runtime_attestation",
}
_ARGV_MAIN_TARGETS: Final = frozenset(
    {
        (
            _RUNTIME_VERIFIER_MODULE,
            _RUNTIME_VERIFIER_MODULE_FILE,
            "_gpu_final_runtime_verifier_child_main",
        ),
        (
            _RUNTIME_VERIFIER_MODULE,
            _RUNTIME_VERIFIER_MODULE_FILE,
            "_locked_runtime_package_final_verifier_child_main",
        ),
    }
)


# This source is executed by CPython before any installed package is importable.
# It intentionally avoids ``site``, ``sysconfig``, and ``addsitedir`` because
# each can process installation-controlled ``.pth`` files.
ISOLATED_RUNTIME_BOOTSTRAP_SOURCE: Final = r'''import ctypes
import fcntl
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import resource
import signal
import stat
import sys
import threading
from pathlib import Path, PurePosixPath
from time import monotonic, sleep


_cachet_capture_limit_bytes = 1_048_576
_cachet_capture_join_seconds = 2.0
_cachet_close_join_seconds = 0.25
_cachet_reap_seconds = 2.0
_cachet_wait_poll_seconds = 0.05
_cachet_original_supervisor_pid = os.getpid()


def _cachet_emergency_parent_cleanup():
    if os.getpid() != _cachet_original_supervisor_pid:
        return
    _cachet_emergency_worker_pid = globals().get("_cachet_worker_pid", 0)
    if (
        type(_cachet_emergency_worker_pid) is not int
        or _cachet_emergency_worker_pid <= 0
        or globals().get("_cachet_worker_reaped") is True
    ):
        return
    for _cachet_emergency_signal_group in (True, False, True):
        try:
            if _cachet_emergency_signal_group:
                os.killpg(_cachet_emergency_worker_pid, signal.SIGKILL)
            else:
                os.kill(_cachet_emergency_worker_pid, signal.SIGKILL)
        except OSError:
            pass
    _cachet_emergency_reap_deadline = monotonic() + _cachet_reap_seconds
    while True:
        try:
            _cachet_emergency_wait_pid = os.waitpid(
                _cachet_emergency_worker_pid,
                os.WNOHANG,
            )[0]
        except InterruptedError:
            continue
        except ChildProcessError:
            break
        if _cachet_emergency_wait_pid == _cachet_emergency_worker_pid:
            break
        if monotonic() >= _cachet_emergency_reap_deadline:
            break
        sleep(_cachet_wait_poll_seconds)
    try:
        os.killpg(_cachet_emergency_worker_pid, signal.SIGKILL)
    except OSError:
        pass
    globals()["_cachet_worker_reaped"] = True
    globals()["_cachet_worker_pid"] = 0


def _cachet_supervisor_fail(status=70):
    try:
        _cachet_emergency_parent_cleanup()
    except BaseException:
        pass
    os._exit(status)


class _CachetWorkerFailure(BaseException):
    def __init__(self, status):
        self.status = status


class _CachetProtocolComplete(BaseException):
    pass


try:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11):
        _cachet_supervisor_fail()
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
        or sys.flags.safe_path is not True
        or sys.flags.dont_write_bytecode != 1
        or not hasattr(os, "fork")
    ):
        _cachet_supervisor_fail()
    if len(sys.argv) < 7:
        _cachet_supervisor_fail()
    _cachet_runtime_root = Path(sys.argv[1])
    _cachet_module_name = sys.argv[2]
    _cachet_module_relative = PurePosixPath(sys.argv[3])
    _cachet_attribute_name = sys.argv[4]
    _cachet_body_source = sys.argv[5]
    _cachet_execution_timeout_text = sys.argv[6]
    _cachet_arguments = sys.argv[7:]
    try:
        _cachet_execution_timeout_seconds = float(
            _cachet_execution_timeout_text
        )
    except (TypeError, ValueError):
        _cachet_supervisor_fail()
    if (
        not 0.0 < _cachet_execution_timeout_seconds <= 3_600.0
        or format(_cachet_execution_timeout_seconds, ".17g")
        != _cachet_execution_timeout_text
    ):
        _cachet_supervisor_fail()
    _cachet_execution_deadline = (
        monotonic() + _cachet_execution_timeout_seconds
    )
    _cachet_target_spec = (
        _cachet_module_name,
        str(_cachet_module_relative),
        _cachet_attribute_name,
    )
    _cachet_target_protocols = {
        (
            "document_kv_cache._gpu_qualification_sentinels_v2",
            "document_kv_cache/_gpu_qualification_sentinels_v2.py",
            "verify_locked_runtime_v2_package_installation",
        ): ("strict", 350.0, 350.0),
        (
            "document_kv_cache._gpu_qualification_sentinels_v2",
            "document_kv_cache/_gpu_qualification_sentinels_v2.py",
            "verify_gpu_qualification_v2_runtime_installation",
        ): ("strict", 350.0, 350.0),
        (
            "document_kv_cache._gpu_qualification_sentinels_v2",
            "document_kv_cache/_gpu_qualification_sentinels_v2.py",
            "_gpu_final_runtime_verifier_child_main",
        ): ("captured", 280.0, 280.0),
        (
            "document_kv_cache._gpu_qualification_sentinels_v2",
            "document_kv_cache/_gpu_qualification_sentinels_v2.py",
            "_locked_runtime_package_final_verifier_child_main",
        ): ("captured", 280.0, 280.0),
        (
            "document_kv_cache.gpu_qualification_v2",
            "document_kv_cache/gpu_qualification_v2.py",
            "validate_locked_runtime_v2_package_installation_attestation",
        ): ("strict", 0.0, 100.0),
        (
            "document_kv_cache.gpu_qualification_v2",
            "document_kv_cache/gpu_qualification_v2.py",
            "validate_gpu_qualification_v2_runtime_attestation",
        ): ("strict", 0.0, 100.0),
        (
            "document_kv_cache._runtime_bootstrap_native_loader",
            "document_kv_cache/_runtime_bootstrap_native_loader.py",
            "main",
        ): ("native", 0.0, 100.0),
        ("pip._internal.cli.main", "pip/_internal/cli/main.py", "main"): (
            "captured",
            0.0,
            170.0,
        ),
    }
    (
        _cachet_expected_protocol,
        _cachet_minimum_execution_timeout_seconds,
        _cachet_maximum_execution_timeout_seconds,
    ) = _cachet_target_protocols[_cachet_target_spec]
    if (
        not _cachet_minimum_execution_timeout_seconds
        <= _cachet_execution_timeout_seconds
        <= _cachet_maximum_execution_timeout_seconds
    ):
        _cachet_supervisor_fail()
    if (
        not _cachet_runtime_root.is_absolute()
        or _cachet_runtime_root.resolve(strict=True) != _cachet_runtime_root
        or _cachet_module_relative.is_absolute()
        or ".." in _cachet_module_relative.parts
        or not _cachet_body_source
    ):
        _cachet_supervisor_fail()
    _cachet_python = _cachet_runtime_root / "bin" / "python"
    _cachet_site_packages = (
        _cachet_runtime_root / "lib" / "python3.11" / "site-packages"
    )
    _cachet_package_name = (
        "pip" if _cachet_module_name == "pip._internal.cli.main" else "document_kv_cache"
    )
    _cachet_package_file = (
        _cachet_site_packages / _cachet_package_name / "__init__.py"
    )
    _cachet_module_file = _cachet_site_packages.joinpath(
        *_cachet_module_relative.parts
    )
    for _cachet_path, _cachet_kind in (
        (_cachet_runtime_root, "directory"),
        (_cachet_site_packages, "directory"),
        (_cachet_python, "file"),
        (_cachet_package_file, "file"),
        (_cachet_module_file, "file"),
    ):
        _cachet_info = _cachet_path.stat(follow_symlinks=False)
        if _cachet_path.is_symlink():
            _cachet_supervisor_fail()
        if _cachet_kind == "directory":
            if not stat.S_ISDIR(_cachet_info.st_mode):
                _cachet_supervisor_fail()
        elif not stat.S_ISREG(_cachet_info.st_mode):
            _cachet_supervisor_fail()
        if _cachet_path.resolve(strict=True) != _cachet_path:
            _cachet_supervisor_fail()
    if Path(sys.executable).resolve(strict=True) != _cachet_python:
        _cachet_supervisor_fail()
    if str(_cachet_site_packages) in sys.path:
        _cachet_supervisor_fail()

    _cachet_libc_fflush = ctypes.CDLL(None).fflush
    _cachet_libc_fflush.argtypes = [ctypes.c_void_p]
    _cachet_libc_fflush.restype = ctypes.c_int
    _cachet_source_file_loader = importlib.machinery.SourceFileLoader
    _cachet_module_from_spec = importlib.util.module_from_spec
    _cachet_spec_from_file_location = importlib.util.spec_from_file_location

    def _cachet_pipe():
        _cachet_read_fd, _cachet_write_fd = os.pipe()
        os.set_inheritable(_cachet_read_fd, False)
        os.set_inheritable(_cachet_write_fd, False)
        return _cachet_read_fd, _cachet_write_fd

    _cachet_import_stdout_pipe = _cachet_pipe()
    _cachet_import_stderr_pipe = _cachet_pipe()
    _cachet_target_stdout_pipe = _cachet_pipe()
    _cachet_target_stderr_pipe = _cachet_pipe()
    _cachet_protocol_pipe = _cachet_pipe()
    _cachet_worker_pid = 0
    _cachet_worker_reaped = False
    _cachet_termination_requested = [False]

    def _cachet_request_termination(_signal_number, _frame):
        _cachet_termination_requested[0] = True
        if _cachet_worker_reaped or _cachet_worker_pid <= 0:
            return
        try:
            os.killpg(_cachet_worker_pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            os.kill(_cachet_worker_pid, signal.SIGKILL)
        except OSError:
            pass

    _cachet_supervisor_signals = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    for _cachet_signal in _cachet_supervisor_signals:
        signal.signal(_cachet_signal, _cachet_request_termination)
    if _cachet_termination_requested[0] or monotonic() >= _cachet_execution_deadline:
        _cachet_supervisor_fail(75)
    _cachet_worker_pid = os.fork()
    if _cachet_worker_pid == 0:
        for _cachet_signal in _cachet_supervisor_signals:
            signal.signal(_cachet_signal, signal.SIG_DFL)
        if _cachet_termination_requested[0]:
            os._exit(70)
        for _cachet_read_fd in (
            _cachet_import_stdout_pipe[0],
            _cachet_import_stderr_pipe[0],
            _cachet_target_stdout_pipe[0],
            _cachet_target_stderr_pipe[0],
            _cachet_protocol_pipe[0],
        ):
            os.close(_cachet_read_fd)
        _cachet_import_stdout_fd = fcntl.fcntl(
            _cachet_import_stdout_pipe[1], fcntl.F_DUPFD_CLOEXEC, 64
        )
        _cachet_import_stderr_fd = fcntl.fcntl(
            _cachet_import_stderr_pipe[1], fcntl.F_DUPFD_CLOEXEC, 64
        )
        _cachet_target_stdout_fd = fcntl.fcntl(
            _cachet_target_stdout_pipe[1], fcntl.F_DUPFD_CLOEXEC, 64
        )
        _cachet_target_stderr_fd = fcntl.fcntl(
            _cachet_target_stderr_pipe[1], fcntl.F_DUPFD_CLOEXEC, 64
        )
        _cachet_protocol_fd = fcntl.fcntl(
            _cachet_protocol_pipe[1], fcntl.F_DUPFD_CLOEXEC, 64
        )
        _cachet_null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(_cachet_null_fd, 0, inheritable=False)
        os.dup2(_cachet_import_stdout_fd, 1, inheritable=False)
        os.dup2(_cachet_import_stderr_fd, 2, inheritable=False)
        os.dup2(_cachet_target_stdout_fd, 3, inheritable=False)
        os.dup2(_cachet_target_stderr_fd, 4, inheritable=False)
        os.dup2(_cachet_protocol_fd, 5, inheritable=False)
        _cachet_fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if _cachet_fd_limit == resource.RLIM_INFINITY:
            _cachet_fd_limit = os.sysconf("SC_OPEN_MAX")
        if type(_cachet_fd_limit) is not int or _cachet_fd_limit <= 6:
            _cachet_supervisor_fail()
        os.closerange(6, _cachet_fd_limit)
        for _cachet_internal_fd in (0, 1, 2, 3, 4, 5):
            os.set_inheritable(_cachet_internal_fd, False)
        os.setsid()

        def _cachet_close_worker_streams_in_fork_child():
            for _cachet_descriptor in (0, 1, 2, 3, 4, 5):
                try:
                    os.close(_cachet_descriptor)
                except OSError:
                    pass

        os.register_at_fork(
            after_in_child=_cachet_close_worker_streams_in_fork_child
        )
        _cachet_protocol_completed = [False]

        def _cachet_fail(status=70):
            if type(status) is not int or not 1 <= status <= 255:
                status = 70
            raise _CachetWorkerFailure(status)

        def _cachet_write_protocol(tag, payload=b""):
            if (
                _cachet_protocol_completed[0]
                or type(tag) is not bytes
                or len(tag) != 1
                or type(payload) is not bytes
                or len(payload) > _cachet_capture_limit_bytes - 1
            ):
                _cachet_fail()
            _cachet_message = tag + len(payload).to_bytes(8, "big") + payload
            if len(_cachet_message) > _cachet_capture_limit_bytes:
                _cachet_fail()
            try:
                _cachet_offset = 0
                while _cachet_offset < len(_cachet_message):
                    _cachet_written = os.write(
                        5,
                        _cachet_message[_cachet_offset:],
                    )
                    if _cachet_written <= 0:
                        _cachet_fail()
                    _cachet_offset += _cachet_written
                os.close(5)
            except _CachetWorkerFailure:
                raise
            except BaseException:
                _cachet_fail()
            _cachet_protocol_completed[0] = True
            raise _CachetProtocolComplete()

        def _cachet_protocol_exit(payload):
            if type(payload) is not bytes or not payload:
                _cachet_fail()
            _cachet_write_protocol(b"S", payload)

        def _cachet_captured_stdout_exit(status):
            if type(status) is not int or status != 0:
                _cachet_fail()
            _cachet_write_protocol(b"C")

        def _cachet_native_loader_protocol_exit(record):
            _cachet_native_loader_record_fields = {
                "benchmark_executed",
                "cuda_available",
                "cuda_device_count",
                "cuda_synchronized",
                "cuda_tensor_result",
                "gpu_compute_capability",
                "gpu_name",
                "ld_library_path_sha256",
                "model_loaded",
                "network_accessed",
                "record_type",
                "schema_version",
                "torch_cuda_version",
                "torch_distribution_version",
                "torch_lib_is_first",
                "torch_origin",
                "torch_source_version",
                "vllm_distribution_version",
                "vllm_native_module",
                "vllm_native_origin",
            }
            if type(record) is not dict or set(record) != _cachet_native_loader_record_fields:
                _cachet_fail(72)
            try:
                _cachet_payload = json.dumps(
                    record,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            except BaseException:
                _cachet_fail()
            _cachet_write_protocol(b"N", _cachet_payload)

        _cachet_worker_status = 70
        try:
            sys.prefix = str(_cachet_runtime_root)
            sys.exec_prefix = str(_cachet_runtime_root)
            sys.path.append(str(_cachet_site_packages))
            if _cachet_module_name in sys.modules:
                _cachet_fail()
            _cachet_package = importlib.import_module(_cachet_package_name)
            _cachet_package_spec = getattr(_cachet_package, "__spec__", None)
            _cachet_package_loader = getattr(_cachet_package_spec, "loader", None)
            if (
                type(getattr(_cachet_package, "__file__", None)) is not str
                or Path(_cachet_package.__file__) != _cachet_package_file
                or Path(_cachet_package.__file__).resolve(strict=True)
                != _cachet_package_file
                or getattr(_cachet_package_spec, "name", None)
                != _cachet_package_name
                or getattr(_cachet_package_spec, "origin", None)
                != str(_cachet_package_file)
                or type(_cachet_package_loader)
                is not _cachet_source_file_loader
                or getattr(_cachet_package_loader, "name", None)
                != _cachet_package_name
                or getattr(_cachet_package_loader, "path", None)
                != str(_cachet_package_file)
                or _cachet_module_name in sys.modules
            ):
                _cachet_fail()
            _cachet_module_spec = _cachet_spec_from_file_location(
                _cachet_module_name,
                _cachet_module_file,
            )
            _cachet_module_loader = getattr(_cachet_module_spec, "loader", None)
            if (
                getattr(_cachet_module_spec, "name", None) != _cachet_module_name
                or getattr(_cachet_module_spec, "origin", None)
                != str(_cachet_module_file)
                or type(_cachet_module_loader)
                is not _cachet_source_file_loader
                or getattr(_cachet_module_loader, "name", None)
                != _cachet_module_name
                or getattr(_cachet_module_loader, "path", None)
                != str(_cachet_module_file)
            ):
                _cachet_fail()
            _cachet_module = _cachet_module_from_spec(_cachet_module_spec)
            sys.modules[_cachet_module_name] = _cachet_module
            _cachet_module_loader.exec_module(_cachet_module)
            if (
                type(getattr(_cachet_module, "__file__", None)) is not str
                or Path(_cachet_module.__file__) != _cachet_module_file
                or Path(_cachet_module.__file__).resolve(strict=True)
                != _cachet_module_file
                or getattr(_cachet_module, "__spec__", None)
                is not _cachet_module_spec
            ):
                _cachet_fail()
            _cachet_target = getattr(_cachet_module, _cachet_attribute_name)
            if not callable(_cachet_target):
                _cachet_fail()
            sys.stdout.flush()
            sys.stderr.flush()
            if _cachet_libc_fflush(None) != 0:
                _cachet_fail()
            os.dup2(3, 1, inheritable=False)
            os.dup2(4, 2, inheritable=False)
            os.close(3)
            os.close(4)
            exec(
                compile(
                    _cachet_body_source,
                    "<cachet-isolated-runtime-body>",
                    "exec",
                ),
                {
                    "__builtins__": __builtins__,
                    "_cachet_arguments": _cachet_arguments,
                    "_cachet_captured_stdout_exit": _cachet_captured_stdout_exit,
                    "_cachet_fail": _cachet_fail,
                    "_cachet_native_loader_protocol_exit": (
                        _cachet_native_loader_protocol_exit
                    ),
                    "_cachet_protocol_exit": _cachet_protocol_exit,
                    "_cachet_target": _cachet_target,
                },
            )
            _cachet_fail(71)
        except _CachetProtocolComplete:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                if _cachet_libc_fflush(None) != 0:
                    raise RuntimeError("native stream flush failed")
                _cachet_worker_status = 0
            except BaseException:
                _cachet_worker_status = 70
        except _CachetWorkerFailure as _cachet_failure:
            _cachet_worker_status = _cachet_failure.status
        except BaseException:
            _cachet_worker_status = 70
        _cachet_worker_exit_requested = True
        raise SystemExit(_cachet_worker_status)

    for _cachet_write_fd in (
        _cachet_import_stdout_pipe[1],
        _cachet_import_stderr_pipe[1],
        _cachet_target_stdout_pipe[1],
        _cachet_target_stderr_pipe[1],
        _cachet_protocol_pipe[1],
    ):
        os.close(_cachet_write_fd)

    def _cachet_kill_worker_group(worker_pid, include_worker=True):
        if type(worker_pid) is not int or worker_pid <= 0:
            return
        try:
            os.killpg(worker_pid, signal.SIGKILL)
        except OSError:
            pass
        if include_worker:
            try:
                os.kill(worker_pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                os.killpg(worker_pid, signal.SIGKILL)
            except OSError:
                pass
    _cachet_stream_descriptors = {
        "import_stdout": _cachet_import_stdout_pipe[0],
        "import_stderr": _cachet_import_stderr_pipe[0],
        "target_stdout": _cachet_target_stdout_pipe[0],
        "target_stderr": _cachet_target_stderr_pipe[0],
        "protocol": _cachet_protocol_pipe[0],
    }
    _cachet_captures = {
        _cachet_name: {
            "byte_count": 0,
            "sha256": hashlib.sha256(),
            "retained": bytearray(),
            "overflow": False,
            "failed": False,
        }
        for _cachet_name in _cachet_stream_descriptors
    }

    def _cachet_drain(stream_name, descriptor):
        _cachet_stream = _cachet_captures[stream_name]
        try:
            while True:
                _cachet_block = os.read(descriptor, 64 * 1024)
                if not _cachet_block:
                    return
                _cachet_stream["byte_count"] += len(_cachet_block)
                _cachet_stream["sha256"].update(_cachet_block)
                _cachet_remaining = (
                    _cachet_capture_limit_bytes
                    - len(_cachet_stream["retained"])
                )
                if _cachet_remaining > 0:
                    _cachet_stream["retained"].extend(
                        _cachet_block[:_cachet_remaining]
                    )
                if _cachet_stream["byte_count"] > _cachet_capture_limit_bytes:
                    _cachet_stream["overflow"] = True
        except BaseException:
            _cachet_stream["failed"] = True
        finally:
            try:
                os.close(descriptor)
            except OSError:
                _cachet_stream["failed"] = True

    _cachet_threads = []
    for _cachet_name, _cachet_descriptor in _cachet_stream_descriptors.items():
        _cachet_thread = threading.Thread(
            target=_cachet_drain,
            args=(_cachet_name, _cachet_descriptor),
            daemon=True,
        )
        _cachet_threads.append(_cachet_thread)
        _cachet_thread.start()
    _cachet_wait_status = None
    _cachet_reaped_worker_pid = 0
    _cachet_execution_timed_out = False
    _cachet_interrupted = False
    while _cachet_wait_status is None:
        _cachet_now = monotonic()
        if _cachet_termination_requested[0]:
            _cachet_interrupted = True
            break
        _cachet_remaining = _cachet_execution_deadline - _cachet_now
        if _cachet_remaining <= 0.0:
            _cachet_execution_timed_out = True
            break
        try:
            _cachet_wait_pid, _cachet_polled_status = os.waitpid(
                _cachet_worker_pid,
                os.WNOHANG,
            )
        except InterruptedError:
            continue
        if _cachet_wait_pid == _cachet_worker_pid:
            _cachet_wait_status = _cachet_polled_status
            if monotonic() >= _cachet_execution_deadline:
                _cachet_execution_timed_out = True
            _cachet_worker_reaped = True
            _cachet_reaped_worker_pid = _cachet_worker_pid
            _cachet_worker_pid = 0
            break
        if _cachet_wait_pid != 0:
            _cachet_interrupted = True
            break
        _cachet_remaining = _cachet_execution_deadline - monotonic()
        if _cachet_remaining > 0.0:
            sleep(min(_cachet_wait_poll_seconds, _cachet_remaining))

    if _cachet_wait_status is None:
        _cachet_kill_worker_group(_cachet_worker_pid)
        _cachet_reap_deadline = monotonic() + _cachet_reap_seconds
        while _cachet_wait_status is None:
            try:
                _cachet_wait_pid, _cachet_polled_status = os.waitpid(
                    _cachet_worker_pid,
                    os.WNOHANG,
                )
            except InterruptedError:
                continue
            except ChildProcessError:
                _cachet_supervisor_fail(75)
            if _cachet_wait_pid == _cachet_worker_pid:
                _cachet_wait_status = _cachet_polled_status
                _cachet_worker_reaped = True
                _cachet_reaped_worker_pid = _cachet_worker_pid
                _cachet_worker_pid = 0
                break
            if _cachet_wait_pid != 0 or monotonic() >= _cachet_reap_deadline:
                _cachet_kill_worker_group(_cachet_worker_pid)
                _cachet_supervisor_fail(75)
            _cachet_kill_worker_group(_cachet_worker_pid)
            sleep(_cachet_wait_poll_seconds)
    _cachet_kill_worker_group(_cachet_reaped_worker_pid, include_worker=False)

    _cachet_drain_deadline = monotonic() + _cachet_capture_join_seconds
    for _cachet_thread in _cachet_threads:
        _cachet_thread.join(max(0.0, _cachet_drain_deadline - monotonic()))
    if any(_cachet_thread.is_alive() for _cachet_thread in _cachet_threads):
        for _cachet_descriptor in _cachet_stream_descriptors.values():
            try:
                os.close(_cachet_descriptor)
            except OSError:
                pass
        _cachet_close_deadline = monotonic() + _cachet_close_join_seconds
        for _cachet_thread in _cachet_threads:
            _cachet_thread.join(
                max(0.0, _cachet_close_deadline - monotonic())
            )
        _cachet_supervisor_fail(73)
    if _cachet_execution_timed_out:
        _cachet_supervisor_fail(75)
    if _cachet_interrupted or _cachet_termination_requested[0]:
        _cachet_supervisor_fail()
    if any(
        _cachet_captures[_cachet_name][_cachet_field]
        for _cachet_name in _cachet_captures
        for _cachet_field in ("overflow", "failed")
    ):
        _cachet_supervisor_fail(73)
    if not os.WIFEXITED(_cachet_wait_status):
        _cachet_supervisor_fail()
    _cachet_worker_status = os.WEXITSTATUS(_cachet_wait_status)
    if _cachet_worker_status != 0:
        _cachet_supervisor_fail(_cachet_worker_status)
    if (
        _cachet_captures["import_stdout"]["byte_count"] != 0
        or _cachet_captures["import_stderr"]["byte_count"] != 0
    ):
        _cachet_supervisor_fail(72)
    _cachet_protocol = bytes(_cachet_captures["protocol"]["retained"])
    if len(_cachet_protocol) < 9:
        _cachet_supervisor_fail(72)
    _cachet_protocol_tag = _cachet_protocol[:1]
    _cachet_protocol_size = int.from_bytes(_cachet_protocol[1:9], "big")
    if _cachet_protocol_size != len(_cachet_protocol) - 9:
        _cachet_supervisor_fail(72)
    _cachet_protocol_payload = _cachet_protocol[9:]

    if _cachet_expected_protocol == "strict":
        if (
            _cachet_protocol_tag != b"S"
            or not _cachet_protocol_payload
            or _cachet_captures["target_stdout"]["byte_count"] != 0
            or _cachet_captures["target_stderr"]["byte_count"] != 0
        ):
            _cachet_supervisor_fail(72)
        _cachet_approved_payload = _cachet_protocol_payload
    elif _cachet_expected_protocol == "captured":
        if (
            _cachet_protocol_tag != b"C"
            or _cachet_protocol_payload
            or _cachet_captures["target_stdout"]["byte_count"] == 0
            or _cachet_captures["target_stderr"]["byte_count"] != 0
        ):
            _cachet_supervisor_fail(72)
        _cachet_approved_payload = bytes(
            _cachet_captures["target_stdout"]["retained"]
        )
    elif _cachet_expected_protocol == "native":
        if (
            _cachet_protocol_tag != b"N"
            or not _cachet_protocol_payload
            or _cachet_captures["target_stderr"]["byte_count"] != 0
        ):
            _cachet_supervisor_fail(72)

        def _cachet_reject_duplicate_pairs(pairs):
            _cachet_record = {}
            for _cachet_key, _cachet_value in pairs:
                if _cachet_key in _cachet_record:
                    raise ValueError("duplicate key")
                _cachet_record[_cachet_key] = _cachet_value
            return _cachet_record

        try:
            _cachet_record = json.loads(
                _cachet_protocol_payload.decode("utf-8"),
                object_pairs_hook=_cachet_reject_duplicate_pairs,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ValueError("non-finite number")
                ),
            )
            _cachet_expected_record = {
                "benchmark_executed": False,
                "cuda_available": True,
                "cuda_device_count": 1,
                "cuda_synchronized": True,
                "cuda_tensor_result": 2.0,
                "gpu_compute_capability": "8.9",
                "gpu_name": _cachet_arguments[0],
                "ld_library_path_sha256": hashlib.sha256(
                    _cachet_arguments[1].encode("utf-8")
                ).hexdigest(),
                "model_loaded": False,
                "network_accessed": False,
                "record_type": "cachet.runtime_bootstrap_canary_native_loader.v1",
                "schema_version": 1,
                "torch_cuda_version": "12.9",
                "torch_distribution_version": "2.13.0+cu129",
                "torch_lib_is_first": True,
                "torch_origin": "lib/python3.11/site-packages/torch/__init__.py",
                "torch_source_version": "2.13.0+cu129",
                "vllm_distribution_version": "0.27.1+cu129",
                "vllm_native_module": "vllm._C_stable_libtorch",
                "vllm_native_origin": (
                    "lib/python3.11/site-packages/"
                    "vllm/_C_stable_libtorch.abi3.so"
                ),
            }
            if (
                len(_cachet_arguments) != 2
                or type(_cachet_record) is not dict
                or _cachet_record != _cachet_expected_record
                or type(_cachet_record["schema_version"]) is not int
                or type(_cachet_record["cuda_device_count"]) is not int
                or type(_cachet_record["cuda_tensor_result"]) is not float
                or any(
                    type(_cachet_record[_cachet_boolean_field]) is not bool
                    for _cachet_boolean_field in (
                        "benchmark_executed",
                        "cuda_available",
                        "cuda_synchronized",
                        "model_loaded",
                        "network_accessed",
                        "torch_lib_is_first",
                    )
                )
                or json.dumps(
                    _cachet_record,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                != _cachet_protocol_payload
            ):
                _cachet_supervisor_fail(72)
        except BaseException:
            _cachet_supervisor_fail(72)
        _cachet_record.update(
            {
                "loader_stdout_byte_count": _cachet_captures["target_stdout"][
                    "byte_count"
                ],
                "loader_stdout_sha256": _cachet_captures["target_stdout"][
                    "sha256"
                ].hexdigest(),
                "loader_stderr_byte_count": _cachet_captures["target_stderr"][
                    "byte_count"
                ],
                "loader_stderr_sha256": _cachet_captures["target_stderr"][
                    "sha256"
                ].hexdigest(),
            }
        )
        _cachet_approved_payload = (
            json.dumps(
                _cachet_record,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    else:
        _cachet_supervisor_fail()
    if (
        type(_cachet_approved_payload) is not bytes
        or not _cachet_approved_payload
        or len(_cachet_approved_payload) > _cachet_capture_limit_bytes
    ):
        _cachet_supervisor_fail()
    _cachet_offset = 0
    while _cachet_offset < len(_cachet_approved_payload):
        _cachet_written = os.write(1, _cachet_approved_payload[_cachet_offset:])
        if _cachet_written <= 0:
            _cachet_supervisor_fail(74)
        _cachet_offset += _cachet_written
except BaseException:
    if globals().get("_cachet_worker_exit_requested") is True:
        raise
    _cachet_supervisor_fail()
os._exit(0)
'''


_COMPACT_VERIFIER_BODY: Final = r'''import json
if len(_cachet_arguments) != 6:
    _cachet_fail(64)
_cachet_record = _cachet_target(
    runtime_lock=_cachet_arguments[0],
    vllm_uri=_cachet_arguments[1],
    flashinfer_uri=_cachet_arguments[2],
    runtime_closure_manifest=_cachet_arguments[3],
    package_uri=_cachet_arguments[4],
    package_sha256=_cachet_arguments[5],
)
_cachet_payload = (
    json.dumps(
        _cachet_record,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    + "\n"
).encode("utf-8")
_cachet_protocol_exit(_cachet_payload)
'''


_PRETTY_VERIFIER_BODY: Final = r'''import json
if len(_cachet_arguments) != 6:
    _cachet_fail(64)
_cachet_record = _cachet_target(
    runtime_lock=_cachet_arguments[0],
    vllm_uri=_cachet_arguments[1],
    flashinfer_uri=_cachet_arguments[2],
    runtime_closure_manifest=_cachet_arguments[3],
    package_uri=_cachet_arguments[4],
    package_sha256=_cachet_arguments[5],
)
_cachet_payload = (
    json.dumps(
        _cachet_record,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n"
).encode("utf-8")
_cachet_protocol_exit(_cachet_payload)
'''


_VALIDATOR_BODY: Final = r'''import json
if len(_cachet_arguments) != 1:
    _cachet_fail(64)
_cachet_target(json.loads(_cachet_arguments[0]))
_cachet_protocol_exit(b"validated\n")
'''


_ARGV_MAIN_BODY: Final = r'''
try:
    _cachet_status = _cachet_target(_cachet_arguments)
except BaseException:
    _cachet_fail()
_cachet_captured_stdout_exit(_cachet_status)
'''


_PIP_CHECK_BODY: Final = r'''
try:
    _cachet_status = _cachet_target(["check"])
except BaseException:
    _cachet_fail()
_cachet_captured_stdout_exit(_cachet_status)
'''


_NATIVE_LOADER_BODY: Final = r'''
if len(_cachet_arguments) != 2:
    _cachet_fail(64)
_cachet_target(
    _cachet_arguments,
    protocol_exit=_cachet_native_loader_protocol_exit,
)
_cachet_fail(71)
'''


def _warning_filter_arguments(warning_policy: str) -> list[str]:
    filters = warning_policy.split(",")
    if not filters or any(not item for item in filters):
        raise ValueError("runtime warning policy must contain non-empty filters")
    arguments: list[str] = []
    for warning_filter in filters:
        arguments.extend(("-W", warning_filter))
    return arguments


def _execution_timeout_argument(
    execution_timeout_seconds: float,
    *,
    minimum_execution_timeout_seconds: float,
    maximum_execution_timeout_seconds: float,
) -> str:
    if (
        type(execution_timeout_seconds) not in (int, float)
        or type(minimum_execution_timeout_seconds) not in (int, float)
        or type(maximum_execution_timeout_seconds) not in (int, float)
        or not 0.0
        <= minimum_execution_timeout_seconds
        <= execution_timeout_seconds
        <= maximum_execution_timeout_seconds
        <= _MAX_EXECUTION_TIMEOUT_SECONDS
        or execution_timeout_seconds == 0.0
    ):
        raise ValueError(
            "isolated runtime execution timeout must be finite and within target bounds"
        )
    return format(float(execution_timeout_seconds), ".17g")


def _isolated_runtime_python_command(
    python_executable: str | os.PathLike[str],
    *,
    module_name: str,
    module_relative_path: str,
    attribute_name: str,
    body_source: str,
    arguments: Sequence[str],
    warning_policy: str,
    execution_timeout_seconds: float,
    minimum_execution_timeout_seconds: float,
    maximum_execution_timeout_seconds: float,
) -> list[str]:
    executable = Path(python_executable)
    if (
        not executable.is_absolute()
        or executable.name != "python"
        or executable.parent.name != "bin"
    ):
        raise ValueError("runtime Python must be an absolute <venv>/bin/python path")
    if not body_source:
        raise ValueError("isolated runtime body must not be empty")
    runtime_arguments = list(arguments)
    if any(not isinstance(argument, str) for argument in runtime_arguments):
        raise TypeError("isolated runtime arguments must be strings")
    return [
        str(executable),
        "-I",
        "-S",
        "-B",
        *_warning_filter_arguments(warning_policy),
        "-c",
        ISOLATED_RUNTIME_BOOTSTRAP_SOURCE,
        str(executable.parent.parent),
        module_name,
        module_relative_path,
        attribute_name,
        body_source,
        _execution_timeout_argument(
            execution_timeout_seconds,
            minimum_execution_timeout_seconds=minimum_execution_timeout_seconds,
            maximum_execution_timeout_seconds=maximum_execution_timeout_seconds,
        ),
        *runtime_arguments,
    ]


def isolated_runtime_verifier_command(
    python_executable: str | os.PathLike[str],
    *,
    verifier_name: str,
    arguments: Sequence[str],
    warning_policy: str,
    pretty: bool = False,
    execution_timeout_seconds: float = (
        ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS
    ),
) -> list[str]:
    """Render a captured six-argument runtime verifier command."""

    try:
        attribute_name = _VERIFIER_TARGETS[verifier_name]
    except KeyError:
        raise ValueError("unsupported isolated runtime verifier") from None
    return _isolated_runtime_python_command(
        python_executable,
        module_name=_RUNTIME_VERIFIER_MODULE,
        module_relative_path=_RUNTIME_VERIFIER_MODULE_FILE,
        attribute_name=attribute_name,
        body_source=_PRETTY_VERIFIER_BODY if pretty else _COMPACT_VERIFIER_BODY,
        arguments=arguments,
        warning_policy=warning_policy,
        execution_timeout_seconds=execution_timeout_seconds,
        minimum_execution_timeout_seconds=(
            ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS
        ),
        maximum_execution_timeout_seconds=(
            ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS
        ),
    )


def isolated_runtime_validator_command(
    python_executable: str | os.PathLike[str],
    *,
    validator_name: str,
    canonical_attestation: str,
    warning_policy: str,
    execution_timeout_seconds: float = (
        ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS
    ),
) -> list[str]:
    """Render a validator command whose sole success payload is ``validated``."""

    try:
        attribute_name = _VALIDATOR_TARGETS[validator_name]
    except KeyError:
        raise ValueError("unsupported isolated runtime validator") from None
    return _isolated_runtime_python_command(
        python_executable,
        module_name=_RUNTIME_VALIDATOR_MODULE,
        module_relative_path=_RUNTIME_VALIDATOR_MODULE_FILE,
        attribute_name=attribute_name,
        body_source=_VALIDATOR_BODY,
        arguments=(canonical_attestation,),
        warning_policy=warning_policy,
        execution_timeout_seconds=execution_timeout_seconds,
        minimum_execution_timeout_seconds=0.0,
        maximum_execution_timeout_seconds=(
            ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS
        ),
    )


def isolated_runtime_argv_main_command(
    python_executable: str | os.PathLike[str],
    *,
    module_name: str,
    module_relative_path: str,
    attribute_name: str,
    arguments: Sequence[str],
    warning_policy: str,
    execution_timeout_seconds: float = (
        ISOLATED_RUNTIME_FINAL_CHILD_EXECUTION_TIMEOUT_SECONDS
    ),
) -> list[str]:
    """Render an isolated import followed by a canonical argv-main entrypoint."""

    target = (module_name, module_relative_path, attribute_name)
    if target not in _ARGV_MAIN_TARGETS:
        raise ValueError("unsupported isolated runtime argv-main target")
    return _isolated_runtime_python_command(
        python_executable,
        module_name=module_name,
        module_relative_path=module_relative_path,
        attribute_name=attribute_name,
        body_source=_ARGV_MAIN_BODY,
        arguments=arguments,
        warning_policy=warning_policy,
        execution_timeout_seconds=execution_timeout_seconds,
        minimum_execution_timeout_seconds=(
            ISOLATED_RUNTIME_FINAL_CHILD_EXECUTION_TIMEOUT_SECONDS
        ),
        maximum_execution_timeout_seconds=(
            ISOLATED_RUNTIME_FINAL_CHILD_EXECUTION_TIMEOUT_SECONDS
        ),
    )


def isolated_runtime_pip_check_command(
    python_executable: str | os.PathLike[str],
    *,
    warning_policy: str,
    execution_timeout_seconds: float = (
        ISOLATED_RUNTIME_PIP_CHECK_EXECUTION_TIMEOUT_SECONDS
    ),
) -> list[str]:
    """Render an isolated exact ``pip check`` invocation.

    The pip module import must be silent. Target output remains captured and is
    released only after an exact zero status and empty stderr.
    """

    module_name, module_relative_path, attribute_name = _PIP_CHECK_TARGET
    return _isolated_runtime_python_command(
        python_executable,
        module_name=module_name,
        module_relative_path=module_relative_path,
        attribute_name=attribute_name,
        body_source=_PIP_CHECK_BODY,
        arguments=(),
        warning_policy=warning_policy,
        execution_timeout_seconds=execution_timeout_seconds,
        minimum_execution_timeout_seconds=0.0,
        maximum_execution_timeout_seconds=(
            ISOLATED_RUNTIME_PIP_CHECK_EXECUTION_TIMEOUT_SECONDS
        ),
    )


def isolated_runtime_native_loader_command(
    python_executable: str | os.PathLike[str],
    *,
    expected_gpu_name: str,
    expected_ld_library_path: str,
    warning_policy: str,
    execution_timeout_seconds: float = (
        ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS
    ),
) -> list[str]:
    """Render the one fixed native torch/vLLM loader attestation command."""

    if expected_gpu_name not in _NATIVE_LOADER_GPU_NAMES:
        raise ValueError("unsupported native-loader GPU")
    executable = Path(python_executable)
    torch_lib = (
        executable.parent.parent
        / "lib"
        / "python3.11"
        / "site-packages"
        / "torch"
        / "lib"
    )
    ld_components = expected_ld_library_path.split(os.pathsep)
    if (
        not expected_ld_library_path
        or "\x00" in expected_ld_library_path
        or any(not component for component in ld_components)
        or ld_components[0] != str(torch_lib)
    ):
        raise ValueError(
            "native-loader LD_LIBRARY_PATH must start with runtime torch/lib"
        )
    module_name, module_relative_path, attribute_name = _NATIVE_LOADER_TARGET
    return _isolated_runtime_python_command(
        executable,
        module_name=module_name,
        module_relative_path=module_relative_path,
        attribute_name=attribute_name,
        body_source=_NATIVE_LOADER_BODY,
        arguments=(expected_gpu_name, expected_ld_library_path),
        warning_policy=warning_policy,
        execution_timeout_seconds=execution_timeout_seconds,
        minimum_execution_timeout_seconds=0.0,
        maximum_execution_timeout_seconds=(
            ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS
        ),
    )


_GENERATED_RUNNER_TEMPLATE: Final = r'''_ISOLATED_RUNTIME_BOOTSTRAP_SOURCE = __ISOLATED_RUNTIME_BOOTSTRAP_SOURCE__
_ISOLATED_RUNTIME_COMPACT_VERIFIER_BODY = __ISOLATED_RUNTIME_COMPACT_VERIFIER_BODY__
_ISOLATED_RUNTIME_VALIDATOR_BODY = __ISOLATED_RUNTIME_VALIDATOR_BODY__
_ISOLATED_RUNTIME_NATIVE_LOADER_BODY = __ISOLATED_RUNTIME_NATIVE_LOADER_BODY__
_ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS = __ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS__
_ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS = __ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS__
_ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS = __ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS__
_ISOLATED_RUNTIME_MAX_EXECUTION_TIMEOUT_SECONDS = __ISOLATED_RUNTIME_MAX_EXECUTION_TIMEOUT_SECONDS__


def _isolated_runtime_warning_filter_arguments(warning_policy: str) -> list[str]:
    filters = warning_policy.split(",")
    if not filters or any(not item for item in filters):
        raise ValueError("runtime warning policy must contain non-empty filters")
    arguments: list[str] = []
    for warning_filter in filters:
        arguments.extend(("-W", warning_filter))
    return arguments


def _isolated_runtime_execution_timeout_argument(
    execution_timeout_seconds: float,
    *,
    minimum_execution_timeout_seconds: float,
    maximum_execution_timeout_seconds: float,
) -> str:
    if (
        type(execution_timeout_seconds) not in (int, float)
        or type(minimum_execution_timeout_seconds) not in (int, float)
        or type(maximum_execution_timeout_seconds) not in (int, float)
        or not 0.0
        <= minimum_execution_timeout_seconds
        <= execution_timeout_seconds
        <= maximum_execution_timeout_seconds
        <= _ISOLATED_RUNTIME_MAX_EXECUTION_TIMEOUT_SECONDS
        or execution_timeout_seconds == 0.0
    ):
        raise ValueError(
            "isolated runtime execution timeout must be finite and within target bounds"
        )
    return format(float(execution_timeout_seconds), ".17g")


def _isolated_runtime_python_command(
    python_executable: str,
    *,
    module_name: str,
    module_relative_path: str,
    attribute_name: str,
    body_source: str,
    arguments: list[str],
    warning_policy: str,
    execution_timeout_seconds: float,
    minimum_execution_timeout_seconds: float,
    maximum_execution_timeout_seconds: float,
) -> list[str]:
    executable = Path(python_executable)
    if (
        not executable.is_absolute()
        or executable.name != "python"
        or executable.parent.name != "bin"
    ):
        raise ValueError("runtime Python must be an absolute <venv>/bin/python path")
    if not body_source:
        raise ValueError("isolated runtime body must not be empty")
    if any(not isinstance(argument, str) for argument in arguments):
        raise TypeError("isolated runtime arguments must be strings")
    return [
        str(executable),
        "-I",
        "-S",
        "-B",
        *_isolated_runtime_warning_filter_arguments(warning_policy),
        "-c",
        _ISOLATED_RUNTIME_BOOTSTRAP_SOURCE,
        str(executable.parent.parent),
        module_name,
        module_relative_path,
        attribute_name,
        body_source,
        _isolated_runtime_execution_timeout_argument(
            execution_timeout_seconds,
            minimum_execution_timeout_seconds=minimum_execution_timeout_seconds,
            maximum_execution_timeout_seconds=maximum_execution_timeout_seconds,
        ),
        *arguments,
    ]


def _isolated_runtime_verifier_command(
    python_executable: str,
    *,
    verifier_name: str,
    arguments: list[str],
    warning_policy: str,
    execution_timeout_seconds: float = (
        _ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS
    ),
) -> list[str]:
    targets = {
        "locked_runtime": "verify_locked_runtime_v2_package_installation",
        "gpu_qualification": "verify_gpu_qualification_v2_runtime_installation",
    }
    try:
        attribute_name = targets[verifier_name]
    except KeyError:
        raise ValueError("unsupported isolated runtime verifier") from None
    return _isolated_runtime_python_command(
        python_executable,
        module_name="document_kv_cache._gpu_qualification_sentinels_v2",
        module_relative_path=(
            "document_kv_cache/_gpu_qualification_sentinels_v2.py"
        ),
        attribute_name=attribute_name,
        body_source=_ISOLATED_RUNTIME_COMPACT_VERIFIER_BODY,
        arguments=arguments,
        warning_policy=warning_policy,
        execution_timeout_seconds=execution_timeout_seconds,
        minimum_execution_timeout_seconds=(
            _ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS
        ),
        maximum_execution_timeout_seconds=(
            _ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS
        ),
    )


def _isolated_runtime_validator_command(
    python_executable: str,
    *,
    validator_name: str,
    canonical_attestation: str,
    warning_policy: str,
    execution_timeout_seconds: float = (
        _ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS
    ),
) -> list[str]:
    targets = {
        "locked_runtime": (
            "validate_locked_runtime_v2_package_installation_attestation"
        ),
        "gpu_qualification": "validate_gpu_qualification_v2_runtime_attestation",
    }
    try:
        attribute_name = targets[validator_name]
    except KeyError:
        raise ValueError("unsupported isolated runtime validator") from None
    return _isolated_runtime_python_command(
        python_executable,
        module_name="document_kv_cache.gpu_qualification_v2",
        module_relative_path="document_kv_cache/gpu_qualification_v2.py",
        attribute_name=attribute_name,
        body_source=_ISOLATED_RUNTIME_VALIDATOR_BODY,
        arguments=[canonical_attestation],
        warning_policy=warning_policy,
        execution_timeout_seconds=execution_timeout_seconds,
        minimum_execution_timeout_seconds=0.0,
        maximum_execution_timeout_seconds=(
            _ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS
        ),
    )


def _isolated_runtime_native_loader_command(
    python_executable: str,
    *,
    expected_gpu_name: str,
    expected_ld_library_path: str,
    warning_policy: str,
    execution_timeout_seconds: float = (
        _ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS
    ),
) -> list[str]:
    if expected_gpu_name not in {"NVIDIA L40S", "NVIDIA L4"}:
        raise ValueError("unsupported native-loader GPU")
    executable = Path(python_executable)
    torch_lib = (
        executable.parent.parent
        / "lib"
        / "python3.11"
        / "site-packages"
        / "torch"
        / "lib"
    )
    ld_components = expected_ld_library_path.split(":")
    if (
        not expected_ld_library_path
        or "\x00" in expected_ld_library_path
        or any(not component for component in ld_components)
        or ld_components[0] != str(torch_lib)
    ):
        raise ValueError(
            "native-loader LD_LIBRARY_PATH must start with runtime torch/lib"
        )
    return _isolated_runtime_python_command(
        python_executable,
        module_name="document_kv_cache._runtime_bootstrap_native_loader",
        module_relative_path=(
            "document_kv_cache/_runtime_bootstrap_native_loader.py"
        ),
        attribute_name="main",
        body_source=_ISOLATED_RUNTIME_NATIVE_LOADER_BODY,
        arguments=[expected_gpu_name, expected_ld_library_path],
        warning_policy=warning_policy,
        execution_timeout_seconds=execution_timeout_seconds,
        minimum_execution_timeout_seconds=0.0,
        maximum_execution_timeout_seconds=(
            _ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS
        ),
    )
'''


def isolated_runtime_runner_fragment() -> str:
    """Return the deterministic stdlib-only renderer for generated runners."""

    return (
        _GENERATED_RUNNER_TEMPLATE.replace(
            "__ISOLATED_RUNTIME_BOOTSTRAP_SOURCE__",
            repr(ISOLATED_RUNTIME_BOOTSTRAP_SOURCE),
        )
        .replace(
            "__ISOLATED_RUNTIME_COMPACT_VERIFIER_BODY__",
            repr(_COMPACT_VERIFIER_BODY),
        )
        .replace(
            "__ISOLATED_RUNTIME_VALIDATOR_BODY__",
            repr(_VALIDATOR_BODY),
        )
        .replace(
            "__ISOLATED_RUNTIME_NATIVE_LOADER_BODY__",
            repr(_NATIVE_LOADER_BODY),
        )
        .replace(
            "__ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS__",
            repr(ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS),
        )
        .replace(
            "__ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS__",
            repr(ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS),
        )
        .replace(
            "__ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS__",
            repr(ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS),
        )
        .replace(
            "__ISOLATED_RUNTIME_MAX_EXECUTION_TIMEOUT_SECONDS__",
            repr(_MAX_EXECUTION_TIMEOUT_SECONDS),
        )
        .rstrip()
    )


__all__ = [
    "ISOLATED_RUNTIME_BOOTSTRAP_SOURCE",
    "ISOLATED_RUNTIME_FINAL_CHILD_EXECUTION_TIMEOUT_SECONDS",
    "ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS",
    "ISOLATED_RUNTIME_PIP_CHECK_EXECUTION_TIMEOUT_SECONDS",
    "ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS",
    "ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS",
    "isolated_runtime_argv_main_command",
    "isolated_runtime_native_loader_command",
    "isolated_runtime_pip_check_command",
    "isolated_runtime_runner_fragment",
    "isolated_runtime_validator_command",
    "isolated_runtime_verifier_command",
]
