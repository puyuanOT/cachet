import fcntl
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import pytest

from document_kv_cache._isolated_runtime import (
    ISOLATED_RUNTIME_FINAL_CHILD_EXECUTION_TIMEOUT_SECONDS,
    ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS,
    ISOLATED_RUNTIME_PIP_CHECK_EXECUTION_TIMEOUT_SECONDS,
    ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS,
    ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS,
    isolated_runtime_argv_main_command,
    isolated_runtime_native_loader_command,
    isolated_runtime_pip_check_command,
    isolated_runtime_runner_fragment,
    isolated_runtime_validator_command,
    isolated_runtime_verifier_command,
)
from document_kv_cache import _runtime_bootstrap_native_loader as native_loader


_WARNING_POLICY = "error,ignore:reviewed:FutureWarning:reviewed.module:7"
_REQUIRES_CPYTHON_311_RUNTIME = pytest.mark.skipif(
    sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11),
    reason="the isolated runtime protocol is pinned to CPython 3.11",
)


def _fake_runtime(tmp_path: Path, *, module_source: str) -> Path:
    runtime_root = tmp_path / "runtime"
    runtime_python = runtime_root / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    shutil.copy2(sys.executable, runtime_python)
    package = (
        runtime_root
        / "lib"
        / "python3.11"
        / "site-packages"
        / "document_kv_cache"
    )
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "_gpu_qualification_sentinels_v2.py").write_text(
        module_source,
        encoding="utf-8",
    )
    return runtime_python


def _fake_pip_runtime(tmp_path: Path, *, module_source: str) -> Path:
    runtime_root = tmp_path / "pip-runtime"
    runtime_python = runtime_root / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    shutil.copy2(sys.executable, runtime_python)
    pip_package = runtime_root / "lib/python3.11/site-packages/pip"
    cli_package = pip_package / "_internal" / "cli"
    cli_package.mkdir(parents=True)
    for package in (pip_package, pip_package / "_internal", cli_package):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (cli_package / "main.py").write_text(module_source, encoding="utf-8")
    return runtime_python


def _verifier_command(
    runtime_python: Path,
    *,
    execution_timeout_seconds: float = (
        ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS
    ),
    arguments: tuple[str, ...] = (
        "lock",
        "vllm",
        "flashinfer",
        "closure",
        "package",
        "a" * 64,
    ),
) -> list[str]:
    return isolated_runtime_verifier_command(
        runtime_python,
        verifier_name="gpu_qualification",
        arguments=arguments,
        warning_policy=_WARNING_POLICY,
        execution_timeout_seconds=execution_timeout_seconds,
    )


def _pip_check_command(
    runtime_python: Path,
    *,
    execution_timeout_seconds: float,
) -> list[str]:
    return isolated_runtime_pip_check_command(
        runtime_python,
        warning_policy=_WARNING_POLICY,
        execution_timeout_seconds=execution_timeout_seconds,
    )


def _process_or_group_exists(identifier: int, *, group: bool = False) -> bool:
    try:
        if group:
            os.killpg(identifier, 0)
        else:
            os.kill(identifier, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_or_group_exit(
    identifier: int,
    *,
    group: bool = False,
    timeout_seconds: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while _process_or_group_exists(identifier, group=group):
        if time.monotonic() >= deadline:
            pytest.fail(
                f"isolated runtime left {'group' if group else 'process'} "
                f"{identifier} alive"
            )
        time.sleep(0.02)


def _kill_recorded_processes(record: dict[str, int]) -> None:
    worker_pgid = record.get("worker_pgid", 0)
    if worker_pgid > 0 and worker_pgid != os.getpgrp():
        try:
            os.killpg(worker_pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for field_name in ("worker_pid", "descendant_pid"):
        process_id = record.get(field_name, 0)
        if process_id <= 0 or process_id == os.getpid():
            continue
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait_for_exclusive_lock(path: Path, *, timeout_seconds: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    with path.open("a", encoding="utf-8") as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    pytest.fail("isolated runtime descendant retained its process lock")
                time.sleep(0.02)
                continue
            fcntl.flock(stream, fcntl.LOCK_UN)
            return


def _wait_for_process_record(
    path: Path,
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float = 3.0,
) -> dict[str, int]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            value = None
        if isinstance(value, dict) and value and all(
            isinstance(field_value, int) and field_value > 0
            for field_value in value.values()
        ):
            return value
        if process.poll() is not None:
            pytest.fail("isolated runtime exited before its worker became ready")
        if time.monotonic() >= deadline:
            pytest.fail("isolated runtime worker did not become ready")
        time.sleep(0.02)


def _fake_native_runtime(tmp_path: Path, *, torch_output: bytes = b"") -> Path:
    runtime_python = _fake_runtime(
        tmp_path,
        module_source="def verify_gpu_qualification_v2_runtime_installation(**kwargs):\n    return kwargs\n",
    )
    site_packages = runtime_python.parent.parent / "lib/python3.11/site-packages"
    package = site_packages / "document_kv_cache"
    shutil.copy2(
        Path(native_loader.__file__),
        package / "_runtime_bootstrap_native_loader.py",
    )
    torch_package = site_packages / "torch"
    (torch_package / "lib").mkdir(parents=True)
    (torch_package / "__init__.py").write_text(
        f'''import os
os.write(1, {torch_output!r})
__version__ = "2.13.0+cu129"

class _Version:
    cuda = "12.9"

class _Cuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_name(index):
        assert index == 0
        return "NVIDIA L4"

    @staticmethod
    def get_device_capability(index):
        assert index == 0
        return (8, 9)

    @staticmethod
    def synchronize():
        return None

class _Tensor:
    def __init__(self, value):
        self.value = value

    def __add__(self, value):
        return _Tensor(self.value + value)

    def item(self):
        return float(self.value)

def tensor(value, *, device):
    assert device == "cuda"
    return _Tensor(value)

version = _Version()
cuda = _Cuda()
''',
        encoding="utf-8",
    )
    vllm_package = site_packages / "vllm"
    vllm_package.mkdir()
    native_origin = vllm_package / "_C_stable_libtorch.abi3.so"
    (vllm_package / "__init__.py").write_text(
        'raise RuntimeError("vLLM package initialization is forbidden")\n',
        encoding="utf-8",
    )
    native_source = tmp_path / "native_loader.c"
    native_source.write_text(
        '''#include <Python.h>
static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_C_stable_libtorch",
    NULL,
    -1,
    NULL,
};
PyMODINIT_FUNC PyInit__C_stable_libtorch(void) {
    return PyModule_Create(&module_definition);
}
''',
        encoding="utf-8",
    )
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the native-loader fixture")
    if sys.platform == "darwin":
        compile_arguments = [
            compiler,
            "-bundle",
            "-undefined",
            "dynamic_lookup",
        ]
    else:
        compile_arguments = [compiler, "-shared", "-fPIC"]
    subprocess.run(
        [
            *compile_arguments,
            "-I",
            sysconfig.get_path("include"),
            str(native_source),
            "-o",
            str(native_origin),
        ],
        capture_output=True,
        check=True,
    )
    for distribution_name, version in (
        ("torch", "2.13.0+cu129"),
        ("vllm", "0.27.1+cu129"),
    ):
        metadata = site_packages / f"{distribution_name}-{version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            f"Name: {distribution_name}\n"
            f"Version: {version}\n",
            encoding="utf-8",
        )
    return runtime_python


def _native_loader_environment(runtime_python: Path) -> tuple[dict[str, str], str]:
    torch_lib = (
        runtime_python.parent.parent
        / "lib/python3.11/site-packages/torch/lib"
    )
    ld_library_path = f"{torch_lib}:/reviewed/cuda/lib64"
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "LD_LIBRARY_PATH": ld_library_path,
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
        }
    )
    return environment, ld_library_path


def test_isolated_runtime_command_repeats_warning_filters_and_has_exact_target():
    command = _verifier_command(Path("/reviewed/runtime/bin/python"))
    assert command[:4] == [
        "/reviewed/runtime/bin/python",
        "-I",
        "-S",
        "-B",
    ]
    code_index = command.index("-c")
    assert command[4:code_index] == [
        "-W",
        "error",
        "-W",
        "ignore:reviewed:FutureWarning:reviewed.module:7",
    ]
    assert command[code_index + 2] == "/reviewed/runtime"
    assert command[code_index + 3 : code_index + 6] == [
        "document_kv_cache._gpu_qualification_sentinels_v2",
        "document_kv_cache/_gpu_qualification_sentinels_v2.py",
        "verify_gpu_qualification_v2_runtime_installation",
    ]
    assert command[code_index + 7] == "350"
    assert command[code_index + 8 :] == [
        "lock",
        "vllm",
        "flashinfer",
        "closure",
        "package",
        "a" * 64,
    ]
    bootstrap = command[code_index + 1]
    assert "import site" not in bootstrap
    assert "addsitedir" not in bootstrap
    assert "sys.path.append(str(_cachet_site_packages))" in bootstrap


def test_native_loader_command_rejects_noncanonical_gpu_and_ld_path():
    runtime_python = Path("/reviewed/runtime/bin/python")
    with pytest.raises(ValueError, match="GPU"):
        isolated_runtime_native_loader_command(
            runtime_python,
            expected_gpu_name="NVIDIA A10G",
            expected_ld_library_path=(
                "/reviewed/runtime/lib/python3.11/site-packages/torch/lib"
            ),
            warning_policy=_WARNING_POLICY,
        )
    with pytest.raises(ValueError, match="torch/lib"):
        isolated_runtime_native_loader_command(
            runtime_python,
            expected_gpu_name="NVIDIA L4",
            expected_ld_library_path="/unreviewed/lib",
            warning_policy=_WARNING_POLICY,
        )


def test_native_loader_denies_network_send_and_process_audit_events():
    assert {
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.system",
        "socket.bind",
        "socket.connect",
        "socket.sendmsg",
        "socket.sendto",
        "subprocess.Popen",
    } <= native_loader._BLOCKED_NETWORK_AUDIT_EVENTS  # noqa: SLF001


def test_generated_renderer_matches_shared_compact_command():
    namespace = {"Path": Path}
    exec(compile(isolated_runtime_runner_fragment(), "<fragment>", "exec"), namespace)
    direct = _verifier_command(Path("/reviewed/runtime/bin/python"))
    generated = namespace["_isolated_runtime_verifier_command"](
        "/reviewed/runtime/bin/python",
        verifier_name="gpu_qualification",
        arguments=["lock", "vllm", "flashinfer", "closure", "package", "a" * 64],
        warning_policy=_WARNING_POLICY,
    )
    assert generated == direct


def test_generated_native_loader_renderer_matches_shared_command():
    namespace = {"Path": Path}
    exec(compile(isolated_runtime_runner_fragment(), "<fragment>", "exec"), namespace)
    runtime_python = "/reviewed/runtime/bin/python"
    ld_library_path = (
        "/reviewed/runtime/lib/python3.11/site-packages/torch/lib:/cuda/lib64"
    )
    direct = isolated_runtime_native_loader_command(
        runtime_python,
        expected_gpu_name="NVIDIA L40S",
        expected_ld_library_path=ld_library_path,
        warning_policy=_WARNING_POLICY,
    )
    generated = namespace["_isolated_runtime_native_loader_command"](
        runtime_python,
        expected_gpu_name="NVIDIA L40S",
        expected_ld_library_path=ld_library_path,
        warning_policy=_WARNING_POLICY,
    )
    assert generated == direct


def test_generated_validator_renderer_matches_shared_command():
    namespace = {"Path": Path}
    exec(compile(isolated_runtime_runner_fragment(), "<fragment>", "exec"), namespace)
    direct = isolated_runtime_validator_command(
        "/reviewed/runtime/bin/python",
        validator_name="gpu_qualification",
        canonical_attestation='{"reviewed":true}\n',
        warning_policy=_WARNING_POLICY,
    )
    generated = namespace["_isolated_runtime_validator_command"](
        "/reviewed/runtime/bin/python",
        validator_name="gpu_qualification",
        canonical_attestation='{"reviewed":true}\n',
        warning_policy=_WARNING_POLICY,
    )
    assert generated == direct


@pytest.mark.parametrize(
    "invalid_timeout",
    [True, 0.0, -1.0, float("nan"), float("inf"), 349.0, 350.000_001],
)
def test_isolated_runtime_verifier_rejects_invalid_or_above_target_timeout(
    invalid_timeout,
):
    with pytest.raises(ValueError, match="execution timeout"):
        isolated_runtime_verifier_command(
            "/reviewed/runtime/bin/python",
            verifier_name="gpu_qualification",
            arguments=["lock", "vllm", "flashinfer", "closure", "package", "hash"],
            warning_policy=_WARNING_POLICY,
            execution_timeout_seconds=invalid_timeout,
        )


@pytest.mark.parametrize(
    ("renderer", "maximum"),
    [
        (
            lambda timeout: isolated_runtime_pip_check_command(
                "/reviewed/runtime/bin/python",
                warning_policy=_WARNING_POLICY,
                execution_timeout_seconds=timeout,
            ),
            ISOLATED_RUNTIME_PIP_CHECK_EXECUTION_TIMEOUT_SECONDS,
        ),
        (
            lambda timeout: isolated_runtime_argv_main_command(
                "/reviewed/runtime/bin/python",
                module_name="document_kv_cache._gpu_qualification_sentinels_v2",
                module_relative_path=(
                    "document_kv_cache/_gpu_qualification_sentinels_v2.py"
                ),
                attribute_name="_gpu_final_runtime_verifier_child_main",
                arguments=[],
                warning_policy=_WARNING_POLICY,
                execution_timeout_seconds=timeout,
            ),
            ISOLATED_RUNTIME_FINAL_CHILD_EXECUTION_TIMEOUT_SECONDS,
        ),
        (
            lambda timeout: isolated_runtime_validator_command(
                "/reviewed/runtime/bin/python",
                validator_name="gpu_qualification",
                canonical_attestation="{}\n",
                warning_policy=_WARNING_POLICY,
                execution_timeout_seconds=timeout,
            ),
            ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS,
        ),
        (
            lambda timeout: isolated_runtime_native_loader_command(
                "/reviewed/runtime/bin/python",
                expected_gpu_name="NVIDIA L4",
                expected_ld_library_path=(
                    "/reviewed/runtime/lib/python3.11/site-packages/torch/lib"
                ),
                warning_policy=_WARNING_POLICY,
                execution_timeout_seconds=timeout,
            ),
            ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS,
        ),
    ],
)
def test_isolated_runtime_each_renderer_rejects_above_target_timeout(
    renderer,
    maximum,
):
    with pytest.raises(ValueError, match="execution timeout"):
        renderer(maximum + 0.001)


@pytest.mark.parametrize(
    "renderer",
    [
        lambda: isolated_runtime_verifier_command(
            "/reviewed/runtime/bin/python",
            verifier_name="gpu_qualification",
            arguments=["lock", "vllm", "flashinfer", "closure", "package", "hash"],
            warning_policy=_WARNING_POLICY,
            execution_timeout_seconds=(
                ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS - 1.0
            ),
        ),
        lambda: isolated_runtime_argv_main_command(
            "/reviewed/runtime/bin/python",
            module_name="document_kv_cache._gpu_qualification_sentinels_v2",
            module_relative_path=(
                "document_kv_cache/_gpu_qualification_sentinels_v2.py"
            ),
            attribute_name="_gpu_final_runtime_verifier_child_main",
            arguments=[],
            warning_policy=_WARNING_POLICY,
            execution_timeout_seconds=(
                ISOLATED_RUNTIME_FINAL_CHILD_EXECUTION_TIMEOUT_SECONDS - 1.0
            ),
        ),
    ],
)
def test_nested_isolated_runtime_renderers_reject_lower_deadlines(renderer):
    with pytest.raises(ValueError, match="execution timeout"):
        renderer()


def test_isolated_runtime_deadline_families_leave_cleanup_margin():
    cleanup_margin_seconds = 10.0
    assert (
        ISOLATED_RUNTIME_PIP_CHECK_EXECUTION_TIMEOUT_SECONDS
        + cleanup_margin_seconds
        <= 180.0
    )
    assert (
        ISOLATED_RUNTIME_FINAL_CHILD_EXECUTION_TIMEOUT_SECONDS
        + cleanup_margin_seconds
        < 300.0
    )
    assert 300.0 < ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS
    assert (
        ISOLATED_RUNTIME_PUBLIC_EXECUTION_TIMEOUT_SECONDS
        + cleanup_margin_seconds
        <= 360.0
    )
    assert (
        ISOLATED_RUNTIME_VALIDATOR_EXECUTION_TIMEOUT_SECONDS
        + cleanup_margin_seconds
        < 120.0
    )
    assert (
        ISOLATED_RUNTIME_NATIVE_LOADER_EXECUTION_TIMEOUT_SECONDS
        + cleanup_margin_seconds
        < 120.0
    )


@_REQUIRES_CPYTHON_311_RUNTIME
@pytest.mark.parametrize("tampered_timeout", ["349", "351"])
def test_isolated_runtime_bootstrap_rejects_tampered_target_timeout(
    tmp_path,
    tampered_timeout,
):
    runtime_python = _fake_runtime(
        tmp_path,
        module_source='''def verify_gpu_qualification_v2_runtime_installation(**kwargs):
    return {"ok": True}
''',
    )
    command = _verifier_command(runtime_python)
    code_index = command.index("-c")
    command[code_index + 7] = tampered_timeout

    completed = subprocess.run(command, capture_output=True, timeout=3.0)

    assert completed.returncode == 70
    assert completed.stdout == b""
    assert completed.stderr == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_deadline_kills_reaps_worker_and_same_group_descendant(
    tmp_path,
):
    process_record = tmp_path / "processes.json"
    descendant_lock = tmp_path / "descendant.lock"
    runtime_python = _fake_pip_runtime(
        tmp_path,
        module_source=f'''import fcntl
import json
import os
import signal
import time

def main(arguments):
    if arguments != ["check"]:
        return 2
    raw_ready_read, raw_ready_write = os.pipe()
    ready_read = fcntl.fcntl(raw_ready_read, fcntl.F_DUPFD_CLOEXEC, 64)
    ready_write = fcntl.fcntl(raw_ready_write, fcntl.F_DUPFD_CLOEXEC, 64)
    os.close(raw_ready_read)
    os.close(raw_ready_write)
    descendant = os.fork()
    if descendant == 0:
        os.close(ready_read)
        for signal_number in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(signal_number, signal.SIG_IGN)
        with open({str(descendant_lock)!r}, "w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            os.write(ready_write, b"1")
            os.close(ready_write)
            while True:
                time.sleep(60.0)
    os.close(ready_write)
    if os.read(ready_read, 1) != b"1":
        raise RuntimeError("descendant did not acquire its process lock")
    os.close(ready_read)
    record = {{
        "worker_pid": os.getpid(),
        "worker_pgid": os.getpgrp(),
        "descendant_pid": descendant,
        "descendant_pgid": os.getpgid(descendant),
    }}
    with open({str(process_record)!r}, "w", encoding="utf-8") as stream:
        json.dump(record, stream)
    while True:
        time.sleep(60.0)
''',
    )
    command = _pip_check_command(
        runtime_python,
        execution_timeout_seconds=1.0,
    )
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    record: dict[str, int] = {}
    try:
        stdout, stderr = process.communicate(timeout=5.0)
        elapsed = time.monotonic() - started
        record = json.loads(process_record.read_text(encoding="utf-8"))

        assert 0.8 <= elapsed < 4.0
        assert process.returncode == 75
        assert stdout == b""
        assert stderr == b""
        assert record["worker_pgid"] == record["worker_pid"]
        assert record["descendant_pgid"] == record["worker_pgid"]
        _wait_for_process_or_group_exit(record["worker_pid"])
        _wait_for_exclusive_lock(descendant_lock)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)
        if not record and process_record.exists():
            record = json.loads(process_record.read_text(encoding="utf-8"))
        _kill_recorded_processes(record)


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_rejects_completion_observed_after_deadline(tmp_path):
    process_record = tmp_path / "processes.json"
    runtime_python = _fake_pip_runtime(
        tmp_path,
        module_source=f'''import json
import os
import time

def main(arguments):
    if arguments != ["check"]:
        return 2
    with open({str(process_record)!r}, "w", encoding="utf-8") as stream:
        json.dump({{"worker_pid": os.getpid(), "worker_pgid": os.getpgrp()}}, stream)
    time.sleep(0.25)
    os.write(1, b"completed\\n")
    return 0
''',
    )
    process = subprocess.Popen(
        _pip_check_command(
            runtime_python,
            execution_timeout_seconds=1.0,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    record: dict[str, int] = {}
    stopped = False
    try:
        record = _wait_for_process_record(process_record, process)
        process.send_signal(signal.SIGSTOP)
        stopped = True
        time.sleep(1.2)
        process.send_signal(signal.SIGCONT)
        stopped = False
        stdout, stderr = process.communicate(timeout=3.0)

        assert process.returncode == 75
        assert stdout == b""
        assert stderr == b""
        _wait_for_process_or_group_exit(record["worker_pid"])
    finally:
        if stopped and process.poll() is None:
            process.send_signal(signal.SIGCONT)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)
        if not record and process_record.exists():
            record = json.loads(process_record.read_text(encoding="utf-8"))
        _kill_recorded_processes(record)


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_term_signal_kills_and_reaps_worker_group(tmp_path):
    process_record = tmp_path / "processes.json"
    runtime_python = _fake_pip_runtime(
        tmp_path,
        module_source=f'''import json
import os
import time

def main(arguments):
    if arguments != ["check"]:
        return 2
    with open({str(process_record)!r}, "w", encoding="utf-8") as stream:
        json.dump({{"worker_pid": os.getpid(), "worker_pgid": os.getpgrp()}}, stream)
    while True:
        time.sleep(60.0)
''',
    )
    process = subprocess.Popen(
        _pip_check_command(
            runtime_python,
            execution_timeout_seconds=10.0,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    record: dict[str, int] = {}
    try:
        record = _wait_for_process_record(process_record, process)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=4.0)

        assert process.returncode == 70
        assert stdout == b""
        assert stderr == b""
        _wait_for_process_or_group_exit(record["worker_pid"])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)
        if not record and process_record.exists():
            record = json.loads(process_record.read_text(encoding="utf-8"))
        _kill_recorded_processes(record)


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_parent_setup_failure_emergency_cleans_worker_group(tmp_path):
    process_record = tmp_path / "processes.json"
    descendant_lock = tmp_path / "descendant.lock"
    runtime_python = _fake_pip_runtime(
        tmp_path,
        module_source=f'''import fcntl
import json
import os
import time

def main(arguments):
    if arguments != ["check"]:
        return 2
    raw_ready_read, raw_ready_write = os.pipe()
    ready_read = fcntl.fcntl(raw_ready_read, fcntl.F_DUPFD_CLOEXEC, 64)
    ready_write = fcntl.fcntl(raw_ready_write, fcntl.F_DUPFD_CLOEXEC, 64)
    os.close(raw_ready_read)
    os.close(raw_ready_write)
    descendant = os.fork()
    if descendant == 0:
        os.close(ready_read)
        with open({str(descendant_lock)!r}, "w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            os.write(ready_write, b"1")
            os.close(ready_write)
            while True:
                time.sleep(60.0)
    os.close(ready_write)
    if os.read(ready_read, 1) != b"1":
        raise RuntimeError("descendant did not acquire its process lock")
    os.close(ready_read)
    record = {{
        "worker_pid": os.getpid(),
        "worker_pgid": os.getpgrp(),
        "descendant_pid": descendant,
        "descendant_pgid": os.getpgid(descendant),
    }}
    temporary_record = {str(process_record) + ".tmp"!r}
    with open(temporary_record, "w", encoding="utf-8") as stream:
        json.dump(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_record, {str(process_record)!r})
    while True:
        time.sleep(60.0)
''',
    )
    command = _pip_check_command(
        runtime_python,
        execution_timeout_seconds=10.0,
    )
    code_index = command.index("-c")
    setup_marker = "    _cachet_stream_descriptors = {"
    injected_failure = f'''    _cachet_injected_deadline = monotonic() + 3.0
    while not Path({str(process_record)!r}).exists():
        if monotonic() >= _cachet_injected_deadline:
            break
        sleep(0.01)
    raise RuntimeError("injected post-fork parent setup failure")
'''
    bootstrap = command[code_index + 1]
    assert bootstrap.count(setup_marker) == 1
    command[code_index + 1] = bootstrap.replace(
        setup_marker,
        injected_failure + setup_marker,
    )
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    record: dict[str, int] = {}
    try:
        stdout, stderr = process.communicate(timeout=6.0)
        record = json.loads(process_record.read_text(encoding="utf-8"))

        assert process.returncode == 70
        assert stdout == b""
        assert stderr == b""
        assert record["worker_pgid"] == record["worker_pid"]
        assert record["descendant_pgid"] == record["worker_pgid"]
        _wait_for_process_or_group_exit(record["worker_pid"])
        _wait_for_exclusive_lock(descendant_lock)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)
        if not record and process_record.exists():
            record = json.loads(process_record.read_text(encoding="utf-8"))
        _kill_recorded_processes(record)


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_import_output_fails_before_canonical_release(tmp_path):
    runtime_python = _fake_runtime(
        tmp_path,
        module_source='''import os
os.write(1, b"UNTRUSTED-IMPORT-OUTPUT")
os.write(2, b"UNTRUSTED-IMPORT-ERROR")

def verify_gpu_qualification_v2_runtime_installation(**kwargs):
    return {"ok": len(kwargs) == 6}
''',
    )
    completed = subprocess.run(_verifier_command(runtime_python), capture_output=True)
    assert completed.returncode == 72
    assert completed.stdout == b""
    assert completed.stderr == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_rejects_package_preseeded_symlink_target(tmp_path):
    runtime_python = _fake_runtime(
        tmp_path,
        module_source='''raise RuntimeError("the exact target executed")
''',
    )
    package = (
        runtime_python.parent.parent
        / "lib/python3.11/site-packages/document_kv_cache"
    )
    spoof_path = package / "spoof_target.py"
    spoof_path.symlink_to(package / "_gpu_qualification_sentinels_v2.py")
    (package / "__init__.py").write_text(
        f'''import sys
import types

module = types.ModuleType("document_kv_cache._gpu_qualification_sentinels_v2")
module.__file__ = {str(spoof_path)!r}
module.verify_gpu_qualification_v2_runtime_installation = lambda **kwargs: {{"spoof": True}}
sys.modules[module.__name__] = module
''',
        encoding="utf-8",
    )
    completed = subprocess.run(_verifier_command(runtime_python), capture_output=True)
    assert completed.returncode == 70
    assert completed.stdout == b""
    assert completed.stderr == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_skips_site_hooks(tmp_path):
    runtime_python = _fake_runtime(
        tmp_path,
        module_source='''def verify_gpu_qualification_v2_runtime_installation(**kwargs):
    return {"ok": len(kwargs) == 6}
''',
    )
    site_packages = runtime_python.parent.parent / "lib/python3.11/site-packages"
    (site_packages / "sitecustomize.py").write_text(
        'raise RuntimeError("sitecustomize executed")\n', encoding="utf-8"
    )
    (site_packages / "untrusted.pth").write_text(
        'import os; os.write(1, b"UNTRUSTED-PTH-OUTPUT")\n', encoding="utf-8"
    )
    completed = subprocess.run(_verifier_command(runtime_python), capture_output=True)
    assert completed.returncode == 0
    assert json.loads(completed.stdout) == {"ok": True}
    assert completed.stdout == b'{"ok":true}\n'
    assert completed.stderr == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_captures_atexit_output_before_release(tmp_path):
    runtime_python = _fake_runtime(
        tmp_path,
        module_source='''import atexit
import os

atexit.register(lambda: os.write(1, b"UNTRUSTED-ATEXIT-OUTPUT"))

def verify_gpu_qualification_v2_runtime_installation(**kwargs):
    return {"ok": len(kwargs) == 6}
''',
    )
    completed = subprocess.run(_verifier_command(runtime_python), capture_output=True)
    assert completed.returncode == 72
    assert completed.stdout == b""
    assert completed.stderr == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_captures_late_background_thread_output(tmp_path):
    runtime_python = _fake_runtime(
        tmp_path,
        module_source='''import os
import threading
import time

def verify_gpu_qualification_v2_runtime_installation(**kwargs):
    writer = os.dup(1)
    def write_late():
        time.sleep(0.05)
        os.write(writer, b"THREAD-LEAK")
        os.close(writer)
    threading.Thread(target=write_late).start()
    return {"ok": len(kwargs) == 6}
''',
    )
    completed = subprocess.run(_verifier_command(runtime_python), capture_output=True)
    assert completed.returncode == 72
    assert completed.stdout == b""
    assert completed.stderr == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_bounds_an_unclosed_capture_writer(tmp_path):
    runtime_python = _fake_runtime(
        tmp_path,
        module_source='''import ctypes
import os
import time

def verify_gpu_qualification_v2_runtime_installation(**kwargs):
    child = ctypes.CDLL(None).fork()
    if child == 0:
        os.setsid()
        time.sleep(5.0)
        os._exit(0)
    return {"ok": len(kwargs) == 6}
''',
    )
    started = time.monotonic()
    completed = subprocess.run(
        _verifier_command(runtime_python),
        capture_output=True,
        timeout=4.0,
    )
    assert time.monotonic() - started < 3.5
    assert completed.returncode == 73
    assert completed.stdout == b""
    assert completed.stderr == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_fork_child_cannot_hold_protocol_or_capture_fds(tmp_path):
    runtime_python = _fake_runtime(
        tmp_path,
        module_source='''import os
import time

def verify_gpu_qualification_v2_runtime_installation(**kwargs):
    child = os.fork()
    if child == 0:
        time.sleep(3.0)
        os._exit(0)
    return {"ok": len(kwargs) == 6}
''',
    )
    completed = subprocess.run(
        _verifier_command(runtime_python),
        capture_output=True,
        timeout=1.5,
    )
    assert completed.returncode == 0
    assert completed.stdout == b'{"ok":true}\n'
    assert completed.stderr == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_worker_closes_inherited_parent_output_duplicates(tmp_path):
    bypass_read_fd, bypass_write_fd = os.pipe()
    inherited_fd = fcntl.fcntl(bypass_write_fd, fcntl.F_DUPFD, 100)
    os.close(bypass_write_fd)
    runtime_python = _fake_runtime(
        tmp_path,
        module_source=f'''import os

def verify_gpu_qualification_v2_runtime_installation(**kwargs):
    try:
        os.write({inherited_fd}, b"INHERITED-FD-BYPASS")
    except OSError:
        pass
    else:
        raise RuntimeError("inherited descriptor remained open")
    return {{"ok": len(kwargs) == 6}}
''',
    )
    try:
        completed = subprocess.run(
            _verifier_command(runtime_python),
            capture_output=True,
            pass_fds=(inherited_fd,),
        )
    finally:
        os.close(inherited_fd)
    bypass = os.read(bypass_read_fd, 1024)
    os.close(bypass_read_fd)
    assert completed.returncode == 0
    assert completed.stdout == b'{"ok":true}\n'
    assert completed.stderr == b""
    assert bypass == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_runtime_internal_fd_noise_corrupts_protocol_and_fails_closed(
    tmp_path,
):
    runtime_python = _fake_runtime(
        tmp_path,
        module_source='''import os

def verify_gpu_qualification_v2_runtime_installation(**kwargs):
    for descriptor in (3, 4, 5):
        try:
            os.write(descriptor, b"INTERNAL-FD-NOISE")
        except OSError:
            pass
    return {"ok": len(kwargs) == 6}
''',
    )
    completed = subprocess.run(_verifier_command(runtime_python), capture_output=True)
    assert completed.returncode == 72
    assert completed.stdout == b""
    assert completed.stderr == b""


def test_isolated_pip_check_has_only_the_exact_internal_target():
    command = isolated_runtime_pip_check_command(
        "/reviewed/runtime/bin/python",
        warning_policy=_WARNING_POLICY,
    )
    code_index = command.index("-c")
    assert command[code_index + 3 : code_index + 6] == [
        "pip._internal.cli.main",
        "pip/_internal/cli/main.py",
        "main",
    ]
    assert '["check"]' in command[code_index + 6]


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_pip_check_preserves_only_the_exact_target_output(tmp_path):
    runtime_root = tmp_path / "pip-runtime"
    runtime_python = runtime_root / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    shutil.copy2(sys.executable, runtime_python)
    pip_package = runtime_root / "lib/python3.11/site-packages/pip"
    cli_package = pip_package / "_internal" / "cli"
    cli_package.mkdir(parents=True)
    for package in (pip_package, pip_package / "_internal", cli_package):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (cli_package / "main.py").write_text(
        '''import os

def main(arguments):
    if arguments != ["check"]:
        return 2
    os.write(1, b"No broken requirements found.\\n")
    return 0
''',
        encoding="utf-8",
    )
    command = isolated_runtime_pip_check_command(
        runtime_python,
        warning_policy=_WARNING_POLICY,
    )
    completed = subprocess.run(command, capture_output=True)
    assert completed.returncode == 0
    assert completed.stdout == b"No broken requirements found.\n"
    assert completed.stderr == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_native_loader_hashes_loader_stdout_and_emits_one_record(tmp_path):
    torch_output = b"TORCH-LOADER\n"
    runtime_python = _fake_native_runtime(
        tmp_path,
        torch_output=torch_output,
    )
    environment, ld_library_path = _native_loader_environment(runtime_python)
    command = isolated_runtime_native_loader_command(
        runtime_python,
        expected_gpu_name="NVIDIA L4",
        expected_ld_library_path=ld_library_path,
        warning_policy=_WARNING_POLICY,
    )
    completed = subprocess.run(command, capture_output=True, env=environment)
    assert completed.returncode == 0
    assert completed.stderr == b""
    assert completed.stdout.count(b"\n") == 1
    record = json.loads(completed.stdout)
    loader_output = torch_output
    assert record == {
        "benchmark_executed": False,
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_synchronized": True,
        "cuda_tensor_result": 2.0,
        "gpu_compute_capability": "8.9",
        "gpu_name": "NVIDIA L4",
        "ld_library_path_sha256": hashlib.sha256(
            ld_library_path.encode("utf-8")
        ).hexdigest(),
        "loader_stderr_byte_count": 0,
        "loader_stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "loader_stdout_byte_count": len(loader_output),
        "loader_stdout_sha256": hashlib.sha256(loader_output).hexdigest(),
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
            "lib/python3.11/site-packages/vllm/_C_stable_libtorch.abi3.so"
        ),
    }


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_native_loader_hashes_libc_output_flushed_after_target(tmp_path):
    runtime_python = _fake_native_runtime(tmp_path)
    torch_init = (
        runtime_python.parent.parent
        / "lib/python3.11/site-packages/torch/__init__.py"
    )
    torch_init.write_text(
        "import ctypes\nctypes.CDLL(None).printf(b'C-BUFFERED')\n"
        + torch_init.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    environment, ld_library_path = _native_loader_environment(runtime_python)
    completed = subprocess.run(
        isolated_runtime_native_loader_command(
            runtime_python,
            expected_gpu_name="NVIDIA L4",
            expected_ld_library_path=ld_library_path,
            warning_policy=_WARNING_POLICY,
        ),
        capture_output=True,
        env=environment,
    )
    assert completed.returncode == 0
    assert completed.stderr == b""
    record = json.loads(completed.stdout)
    loader_output = b"C-BUFFERED"
    assert record["loader_stdout_byte_count"] == len(loader_output)
    assert record["loader_stdout_sha256"] == hashlib.sha256(loader_output).hexdigest()


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_native_loader_rejects_stderr_without_releasing_output(tmp_path):
    runtime_python = _fake_native_runtime(
        tmp_path,
        torch_output=b"SAFE-LOADER-OUTPUT\n",
    )
    torch_init = (
        runtime_python.parent.parent
        / "lib/python3.11/site-packages/torch/__init__.py"
    )
    torch_init.write_text(
        "import os\nos.write(2, b'LOADER-ERROR')\n"
        + torch_init.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    environment, ld_library_path = _native_loader_environment(runtime_python)
    command = isolated_runtime_native_loader_command(
        runtime_python,
        expected_gpu_name="NVIDIA L4",
        expected_ld_library_path=ld_library_path,
        warning_policy=_WARNING_POLICY,
    )
    completed = subprocess.run(command, capture_output=True, env=environment)
    assert completed.returncode == 72
    assert completed.stdout == b""
    assert completed.stderr == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_native_loader_rejects_loader_output_overflow(tmp_path):
    runtime_python = _fake_native_runtime(tmp_path)
    torch_init = (
        runtime_python.parent.parent
        / "lib/python3.11/site-packages/torch/__init__.py"
    )
    source = torch_init.read_text(encoding="utf-8")
    torch_init.write_text(
        source.replace("os.write(1, b'')", "os.write(1, b'x' * 1_048_577)"),
        encoding="utf-8",
    )
    environment, ld_library_path = _native_loader_environment(runtime_python)
    command = isolated_runtime_native_loader_command(
        runtime_python,
        expected_gpu_name="NVIDIA L4",
        expected_ld_library_path=ld_library_path,
        warning_policy=_WARNING_POLICY,
    )
    completed = subprocess.run(command, capture_output=True, env=environment)
    assert completed.returncode == 73
    assert completed.stdout == b""
    assert completed.stderr == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_native_loader_rejects_wrong_distribution_version(tmp_path):
    runtime_python = _fake_native_runtime(tmp_path)
    metadata = next(
        (
            runtime_python.parent.parent / "lib/python3.11/site-packages"
        ).glob("torch-*.dist-info/METADATA")
    )
    metadata.write_text(
        "Metadata-Version: 2.1\nName: torch\nVersion: 2.12.0+cu129\n",
        encoding="utf-8",
    )
    environment, ld_library_path = _native_loader_environment(runtime_python)
    command = isolated_runtime_native_loader_command(
        runtime_python,
        expected_gpu_name="NVIDIA L4",
        expected_ld_library_path=ld_library_path,
        warning_policy=_WARNING_POLICY,
    )
    completed = subprocess.run(command, capture_output=True, env=environment)
    assert completed.returncode == 70
    assert completed.stdout == b""
    assert completed.stderr == b""


@_REQUIRES_CPYTHON_311_RUNTIME
def test_isolated_native_loader_requires_exact_vllm_package_initializer(tmp_path):
    runtime_python = _fake_native_runtime(tmp_path)
    (
        runtime_python.parent.parent
        / "lib/python3.11/site-packages/vllm/__init__.py"
    ).unlink()
    environment, ld_library_path = _native_loader_environment(runtime_python)
    command = isolated_runtime_native_loader_command(
        runtime_python,
        expected_gpu_name="NVIDIA L4",
        expected_ld_library_path=ld_library_path,
        warning_policy=_WARNING_POLICY,
    )
    completed = subprocess.run(command, capture_output=True, env=environment)
    assert completed.returncode == 70
    assert completed.stdout == b""
    assert completed.stderr == b""


def test_native_loader_module_import_is_silent_and_does_not_import_runtime_deps(
    tmp_path,
):
    for package_name in ("torch", "vllm"):
        package = tmp_path / package_name
        package.mkdir()
        (package / "__init__.py").write_text(
            "import os\nos.write(1, b'UNTRUSTED-RUNTIME-IMPORT')\n",
            encoding="utf-8",
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(Path.cwd() / "src"), str(tmp_path))
    )
    script = '''import sys
import document_kv_cache._runtime_bootstrap_native_loader
if "torch" in sys.modules or "vllm" in sys.modules:
    raise SystemExit(3)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=environment,
    )
    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""
