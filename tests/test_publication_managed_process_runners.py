"""Exercise generated bootstrap control flow without installing a GPU runtime."""

import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from document_kv_cache.full_score_execution import FULL_SCORE_RUNNER_SCRIPT
from document_kv_cache.publication_bf16_handoff_generation import (
    PUBLICATION_BF16_HANDOFF_RUNNER_SCRIPT,
)
from document_kv_cache.publication_latency_execution import (
    PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT,
)
from document_kv_cache.publication_latency_handoff_generation import (
    PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT,
)


_RUNNERS = (
    pytest.param(
        PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT,
        "publication_latency_handoff_generation",
        "CACHET_LATENCY_HANDOFF_LOCKED_RUNTIME",
        id="q8",
    ),
    pytest.param(
        PUBLICATION_BF16_HANDOFF_RUNNER_SCRIPT,
        "publication_bf16_handoff_generation",
        "CACHET_LATENCY_HANDOFF_LOCKED_RUNTIME",
        id="bf16",
    ),
    pytest.param(
        PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT,
        "publication_latency_execution",
        "CACHET_LATENCY_SOURCE_CLOSURE_LOCKED_RUNTIME",
        id="latency-source-closure",
    ),
    pytest.param(
        FULL_SCORE_RUNNER_SCRIPT,
        "full_score_execution",
        "CACHET_FULL_SCORE_LOCKED_RUNTIME",
        id="full-score",
    ),
)
_CUDA_PARENT_ENV = "CACHET_GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_ATTESTATION"


@pytest.mark.parametrize("script,module,marker", _RUNNERS)
@pytest.mark.parametrize("child_exit", (0, 7), ids=("success", "failure"))
def test_locked_runtime_child_preserves_managed_parent_and_launch_binding(
    script, module, marker, child_exit, monkeypatch, tmp_path, capfd
):
    runner = tmp_path / "runner.py"
    runner.write_text(script, encoding="utf-8")
    namespace = {"__name__": "managed_runner_test"}
    exec(compile(script, str(runner), "exec"), namespace)
    source_closure = module == "publication_latency_execution"
    venv_dir = "/local_disk0/cachet-managed-process-test"
    venv_python = venv_dir + "/bin/python"
    torch_lib = venv_dir + "/lib/python3.11/site-packages/torch/lib"
    runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    argv = ["--runner-sha256", runner_sha, "--runtime-venv-dir", venv_dir]
    pins = {}
    paths = {}
    volume_paths = {}
    for name in (
        "package-wheel", "runtime-lock", "patched-vllm-wheel",
        "patched-flashinfer-wheel", "runtime-closure-manifest",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        pins[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        paths[name] = str(path)
        uri = "dbfs:/Volumes/test/runtime/" + name
        volume_paths[uri] = str(path)
        argv.extend(["--" + name + "-uri", uri, "--" + name + "-sha256", pins[name]])
    # Keep the actual runner/artifact digest checks; resolve only the test storage.
    namespace["_volume_path" if source_closure else "_cluster_path"] = (
        lambda uri: volume_paths.get(uri, uri)
    )
    if source_closure:
        request = tmp_path / "request.json"
        request.write_text("{}\n", encoding="utf-8")
        request_sha = hashlib.sha256(request.read_bytes()).hexdigest()
        closed_sha = "b" * 64
        volume_paths["dbfs:/Volumes/test/request.json"] = str(request)
        argv.extend([
            "--request-uri", "dbfs:/Volumes/test/request.json",
            "--request-file-sha256", request_sha,
            "--request-closed-record-sha256", closed_sha,
            "--coordinator-run-id", "12345",
        ])
        remaining = [
            "run-source-closure", "--request-path", str(request),
            "--expected-request-file-sha256", request_sha,
            "--expected-request-closed-record-sha256", closed_sha,
            "--coordinator-run-id", "12345",
        ]
    else:
        remaining = ["run-worker", "--payload", "space and ;$(literal) argument"]
        argv.extend(remaining)
    expected_argv = [venv_python, "-m", "document_kv_cache." + module, *remaining]

    parent_environment = {
        "PATH": "/usr/bin:/bin", "LD_LIBRARY_PATH": "/system-cuda/lib",
        "PYTHONPATH": "/untrusted/python", "PIP_INDEX_URL": "https://invalid.test",
        "VIRTUAL_ENV": "/untrusted/venv", _CUDA_PARENT_ENV: "stale-parent-value",
        "CACHET_TEST_PRESERVED": "yes",
    }

    def forbidden_exec(*args, **kwargs):
        pytest.fail("bootstrap replaced the managed task process")

    namespace["os"] = SimpleNamespace(**{
        **vars(os), "environ": parent_environment, "execve": forbidden_exec,
    })
    install_environment = namespace["_pip_subprocess_environment"]()
    install_environment["VIRTUAL_ENV"] = venv_dir
    install_environment["PATH"] = venv_dir + "/bin:" + parent_environment["PATH"]
    events = []
    namespace["_create_runtime_venv"] = lambda *args, **kwargs: events.append("venv")
    namespace["_capture_system_cuda_parent_attestation_json"] = lambda: "test-cuda-parent"
    namespace["_locked_runtime_torch_library_dir"] = lambda directory: torch_lib

    runtime_attestation = {"test_runtime": "verified"}

    def verify_runtime(**kwargs):
        assert kwargs["venv_python"] == venv_python
        for name, path in paths.items():
            assert kwargs[name.replace("-", "_")] == path
        assert kwargs["package_wheel_sha256"] == pins["package-wheel"]
        events.append("verify")
        return runtime_attestation

    namespace["_verify_locked_runtime"] = verify_runtime

    def pip_check(command, **kwargs):
        assert command == [venv_python, "-m", "pip", "check"]
        assert kwargs["environment"] == install_environment
        events.append("pip-check")
        return SimpleNamespace(
            returncode=0, stdout=b"No broken requirements found.\n", stderr=b"",
            timed_out=False, output_limit_exceeded=False,
        )

    namespace["_run_bounded_child"] = pip_check
    if module == "full_score_execution":
        identity = "cachet.full_score.locked_runtime.v2\0" + runner_sha + "".join(
            pins[name] for name in (
                "package-wheel", "runtime-lock", "patched-vllm-wheel",
                "patched-flashinfer-wheel", "runtime-closure-manifest",
            )
        )
        expected_marker = hashlib.sha256(identity.encode("ascii")).hexdigest()
    else:
        domain = "latency_source_closure" if source_closure else "latency_handoff"
        record = {
            "domain": "cachet.publication." + domain + ".runtime.v2",
            **{name.replace("-", "_") + "_sha256": value for name, value in pins.items()},
        }
        expected_marker = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    expected_environment = {**install_environment, marker: expected_marker}
    if source_closure:
        expected_environment["CACHET_LATENCY_SOURCE_CLOSURE_RUNTIME_ATTESTATION"] = (
            json.dumps(runtime_attestation, sort_keys=True, separators=(",", ":"))
        )
    else:
        expected_environment["LD_LIBRARY_PATH"] = torch_lib + ":/system-cuda/lib"
        expected_environment[_CUDA_PARENT_ENV] = "test-cuda-parent"

    parent_pid = os.getpid()
    observed_path = tmp_path / "child-observation.json"

    def checked_child(command, *, env):
        # No stdout/stderr redirection or shell override: output must stream to
        # the task, and shell metacharacters must remain an argument's contents.
        if command[:3] == [venv_python, "-m", "pip"]:
            assert env == install_environment
            events.append("install")
            return 0
        assert command == expected_argv
        assert env == expected_environment
        events.append("child")
        # The deployment venv is deliberately not installed in this CPU test.
        # Run a real child via check_call after verifying its exact launch binding.
        program = (
            "import json,os,pathlib,sys; "
            "print('managed-child-stdout',flush=True); "
            "print('managed-child-stderr',file=sys.stderr,flush=True); "
            "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),"
            "'ppid':os.getppid(),'marker':os.environ[sys.argv[2]]})); "
            "sys.exit(int(sys.argv[3]))"
        )
        return subprocess.check_call(
            [sys.executable, "-B", "-S", "-c", program,
             str(observed_path), marker, str(child_exit)],
            env=env,
        )

    namespace["subprocess"] = SimpleNamespace(check_call=checked_child)
    monkeypatch.setattr(sys, "argv", [str(runner), *argv])

    def bootstrap():
        return namespace["main"]() if source_closure else namespace["_bootstrap"](argv)

    if child_exit:
        with pytest.raises(subprocess.CalledProcessError) as failure:
            bootstrap()
        assert failure.value.returncode == child_exit
    else:
        assert bootstrap() is None
    assert os.getpid() == parent_pid
    observed = json.loads(observed_path.read_text(encoding="utf-8"))
    assert observed["pid"] != parent_pid
    assert observed["ppid"] == parent_pid
    assert observed["marker"] == expected_marker
    output = capfd.readouterr()
    assert "managed-child-stdout" in output.out
    assert "managed-child-stderr" in output.err
    expected_events = ["venv", *(["install"] * 4)]
    if source_closure:
        expected_events.append("pip-check")
    if module != "full_score_execution":
        expected_events.append("verify")
    assert events == [*expected_events, "child"]
    assert marker not in parent_environment
    assert parent_environment[_CUDA_PARENT_ENV] == "stale-parent-value"
