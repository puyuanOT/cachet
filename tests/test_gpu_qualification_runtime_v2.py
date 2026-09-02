from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

import document_kv_cache._gpu_qualification_sentinel_worker as sentinel_worker
import document_kv_cache._gpu_qualification_sentinels_v2 as runtime_v2
import document_kv_cache.gpu_qualification as qualification_v1
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PACKAGE_VERSION,
    FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256,
    FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
    FLASHINFER_PATCHED_WHEEL_SHA256,
    FLASHINFER_TARGET_MEMBER,
    FLASHINFER_TARGET_PATCHED_SHA256,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    GPU_QUALIFICATION_VLLM_VERSION,
    canonical_gpu_qualification_json,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_ARTIFACT_KEYS,
    GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION,
    GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION,
    GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
    GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS,
    GPUQualificationArtifactPinsV2,
    build_gpu_qualification_plan_v2,
    gpu_qualification_v2_runtime_closure,
    validate_gpu_runtime_verification_v2_record,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_ID,
    PUBLICATION_CAMPAIGN_LEDGER_ID,
    PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE,
    VLLM_PATCHED_MANIFEST_SHA256,
    VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_BASE_LOCK_FILENAME,
    VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SIZE,
)
from document_kv_cache.serving_env import (
    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
    GPU_RUNTIME_PYTHONWARNINGS,
    VLLM_PATCHED_WHEEL_SHA256_ENV,
    VLLM_PATCHED_WHEEL_URI_ENV,
)


_ROOT = Path(__file__).resolve().parents[1]
_LOCK_PATH = (
    _ROOT
    / "src"
    / "document_kv_cache"
    / "runtime_locks"
    / VLLM_RUNTIME_BASE_LOCK_FILENAME
)
_CLOSURE_PATH = (
    _ROOT
    / "databricks-runs"
    / "_campaign-inputs"
    / "vllm-0.27.1-runtime-closure"
    / "sha256"
    / RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
    / "vllm-0.27.1-flashinfer-0.6.16.post3-runtime-closure.json"
)
_PACKAGE_SHA256 = "a" * 64
_SOURCE_SHA256 = "b" * 64
_RUNNER_SHA256 = "c" * 64


def _bounded_stream_result(
    data: bytes,
    *,
    byte_count: int | None = None,
    digest: str | None = None,
    limit_exceeded: bool = False,
) -> runtime_v2._BoundedBinaryStreamResult:
    return runtime_v2._BoundedBinaryStreamResult(
        retained=data,
        byte_count=len(data) if byte_count is None else byte_count,
        sha256=sha256(data).hexdigest() if digest is None else digest,
        limit_exceeded=limit_exceeded,
    )


def _bounded_process_result(
    *,
    stdout: bytes = runtime_v2._FINAL_VERIFIER_PIP_CHECK_STDOUT,
    stderr: bytes = b"",
    returncode: int = 0,
    timed_out: bool = False,
    output_limit_exceeded: bool = False,
) -> runtime_v2._BoundedBinarySubprocessResult:
    return runtime_v2._BoundedBinarySubprocessResult(
        returncode=returncode,
        stdout=_bounded_stream_result(stdout, limit_exceeded=output_limit_exceeded),
        stderr=_bounded_stream_result(stderr),
        timed_out=timed_out,
    )


def _pins() -> GPUQualificationArtifactPinsV2:
    return GPUQualificationArtifactPinsV2(
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_sha256=GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        patched_flashinfer_wheel_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
        runtime_closure_manifest_sha256=RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        package_wheel_sha256=_PACKAGE_SHA256,
        cachet_source_tree_sha256=_SOURCE_SHA256,
        runner_sha256=_RUNNER_SHA256,
        input_bundle_sha256=GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    )


def _plan() -> dict[str, Any]:
    return build_gpu_qualification_plan_v2(
        campaign_id=PUBLICATION_CAMPAIGN_ID,
        campaign_record_sha256=PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
        campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS
        ),
        artifact_pins=_pins(),
    )


def _seal_v2(record: dict[str, Any]) -> None:
    record["closed_record_sha256"] = ""
    record["closed_record_sha256"] = sha256(
        canonical_gpu_qualification_json(record).encode("utf-8")
    ).hexdigest()


def _attestation(*, vllm_uri: str, flashinfer_uri: str) -> dict[str, Any]:
    closure = gpu_qualification_v2_runtime_closure()
    return {
        "base_lock_distribution_count": VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
        "base_lock_hash_count": VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
        "base_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "cachet_package_version": GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION,
        "flashinfer_annotation": GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION,
        "flashinfer_direct_url": flashinfer_uri,
        "flashinfer_import_ok": True,
        "closure_bound_flashinfer_manifest_closed_record_sha256": (
            FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
        ),
        "closure_bound_flashinfer_manifest_file_sha256": (
            FLASHINFER_PATCHED_MANIFEST_FILE_SHA256
        ),
        "closure_bound_vllm_manifest_file_sha256": VLLM_PATCHED_MANIFEST_SHA256,
        "flashinfer_member_sha256": FLASHINFER_TARGET_PATCHED_SHA256,
        "flashinfer_package_version": FLASHINFER_PACKAGE_VERSION,
        "flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
        "installed_distribution_count": 198,
        "ok": True,
        "packaged_base_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "pip_check_ok": True,
        "runtime_closure_closed_record_sha256": (
            RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
        ),
        "runtime_closure_file_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "unexpected_distributions": [],
        "vllm_direct_url": vllm_uri,
        "vllm_member_sha256": closure["vllm"]["member_sha256"],
        "vllm_package_version": GPU_QUALIFICATION_VLLM_VERSION,
        "vllm_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        "with_flashinfer_distribution_count": 196,
        "with_vllm_distribution_count": 197,
    }


def _artifact_paths(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir()
    paths: dict[str, Path] = {}
    for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        path = tmp_path / key
        if key == "input_bundle_sha256":
            path.mkdir()
        else:
            path.write_text(key, encoding="utf-8")
        paths[key] = path
    return paths


def test_worker_non_v2_attestation_dispatch_is_exact_legacy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = {"legacy": "exact"}
    calls = 0

    def legacy() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return marker

    monkeypatch.setattr(sentinel_worker, "_runtime_lock_attestation", legacy)
    assert sentinel_worker._runtime_lock_attestation_for_plan({}) is marker
    assert (
        sentinel_worker._runtime_lock_attestation_for_plan(
            {
                "record_type": "cachet.vllm_0271_gpu_qualification_plan.v1",
                "schema_version": 1,
            }
        )
        is marker
    )
    assert calls == 2


def test_runtime_installer_uses_exact_commands_environment_and_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pins = _pins()
    plan = _plan()
    job_id = plan["cloud_qualification"]["jobs"][0]["job_id"]
    planned_job = deepcopy(qualification_v1._plan_job(plan, job_id))
    artifact_paths = _artifact_paths(tmp_path / "artifacts")
    work_dir = tmp_path / "work"
    runtime_dir = work_dir / "runtime"
    runtime_python = runtime_dir / "bin" / "python"
    torch_library_dir = runtime_dir / "lib/python3.11/site-packages/torch/lib"
    events: list[str] = []
    subprocess_calls: list[tuple[list[str], dict[str, Any]]] = []
    identity = SimpleNamespace(file_binding="reviewed-python-binding")

    digest_by_path = {
        artifact_paths[key]: pins.to_record()[key]
        for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS
        if key != "input_bundle_sha256"
    }
    monkeypatch.setattr(runtime_v2, "_file_sha256", lambda path: digest_by_path[path])
    monkeypatch.setattr(runtime_v2, "_read_exact_runtime_closure", lambda _path: {})

    def create_venv(path: Path, *, copies: bool) -> None:
        assert copies is True
        (path / "bin").mkdir(parents=True)
        (path / "bin" / "python").write_bytes(b"python")
        torch_library_dir.mkdir(parents=True)

    attest_calls: list[tuple[str, str | None]] = []

    def attest(
        path: Path,
        *,
        expected_python_version: str,
        expected_file_binding: str | None = None,
    ) -> SimpleNamespace:
        assert path == runtime_dir
        assert expected_python_version == "3.11.11"
        attest_calls.append((expected_python_version, expected_file_binding))
        return identity

    def run(
        arguments: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        subprocess_calls.append((list(arguments), dict(kwargs)))
        if arguments[-1] == "check":
            events.append("pip-check")
        else:
            events.append(f"install-{len(subprocess_calls)}")
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(runtime_v2, "create_venv", create_venv)
    monkeypatch.setattr(runtime_v2, "_attest_isolated_python", attest)
    monkeypatch.setattr(
        runtime_v2,
        "_pip_subprocess_environment",
        lambda: {
            "FLASHINFER_LOGGING_LEVEL": "DEBUG",
            "LD_LIBRARY_PATH": "/ambient/reviewed-lib",
            "PYTHONWARNINGS": "ignore",
        },
    )
    monkeypatch.setattr(runtime_v2.subprocess, "run", run)

    vllm_uri = artifact_paths["patched_vllm_wheel_sha256"].resolve().as_uri()
    flashinfer_uri = (
        artifact_paths["patched_flashinfer_wheel_sha256"].resolve().as_uri()
    )
    package_uri = artifact_paths["package_wheel_sha256"].resolve().as_uri()

    original_launch_environment = runtime_v2._runtime_launch_environment

    def launch_environment(
        *,
        runtime_dir: Path,
        install_environment: dict[str, str],
    ) -> dict[str, str]:
        events.append("launch-environment")
        return original_launch_environment(
            runtime_dir=runtime_dir,
            install_environment=install_environment,
        )

    monkeypatch.setattr(
        runtime_v2, "_runtime_launch_environment", launch_environment
    )

    def final_verifier(
        observed_python: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append("final-verifier")
        assert observed_python == runtime_python
        assert kwargs == {
            "closure_path": artifact_paths["runtime_closure_manifest_sha256"],
            "environment": {
                "FLASHINFER_LOGGING_LEVEL": (
                    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL
                ),
                "LD_LIBRARY_PATH": (
                    f"{torch_library_dir}{os.pathsep}/ambient/reviewed-lib"
                ),
                "PYTHONSAFEPATH": "1",
                "PYTHONWARNINGS": GPU_RUNTIME_PYTHONWARNINGS,
            },
            "flashinfer_uri": flashinfer_uri,
            "package_sha256": _PACKAGE_SHA256,
            "package_uri": package_uri,
            "runtime_lock": artifact_paths["runtime_lock_sha256"],
            "vllm_uri": vllm_uri,
        }
        return _attestation(vllm_uri=vllm_uri, flashinfer_uri=flashinfer_uri)

    monkeypatch.setattr(runtime_v2, "_run_final_runtime_verifier", final_verifier)

    def verify_input(
        observed_python: Path,
        input_bundle: Path,
        *,
        expected_sha256: str,
        environment: dict[str, str],
    ) -> None:
        events.append("input-verifier")
        assert observed_python == runtime_python
        assert input_bundle == artifact_paths["input_bundle_sha256"]
        assert expected_sha256 == GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256
        assert environment["PYTHONSAFEPATH"] == "1"
        assert environment["FLASHINFER_LOGGING_LEVEL"] == "ERROR"
        assert (
            environment["PYTHONWARNINGS"] == GPU_RUNTIME_PYTHONWARNINGS
        )
        assert environment["LD_LIBRARY_PATH"] == (
            f"{torch_library_dir}{os.pathsep}/ambient/reviewed-lib"
        )

    monkeypatch.setattr(
        runtime_v2, "_verify_input_bundle_in_isolated_runtime", verify_input
    )
    monkeypatch.setattr(runtime_v2, "_make_site_packages_read_only", lambda _p: None)

    def worker(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        events.append("worker")
        output_path = Path(arguments[arguments.index("--output-json") + 1])
        output_path.write_text("{}\n", encoding="utf-8")
        assert kwargs["environment"]["PYTHONSAFEPATH"] == "1"
        assert kwargs["environment"]["FLASHINFER_LOGGING_LEVEL"] == "ERROR"
        assert (
            kwargs["environment"]["PYTHONWARNINGS"]
            == GPU_RUNTIME_PYTHONWARNINGS
        )
        assert kwargs["environment"]["LD_LIBRARY_PATH"] == (
            f"{torch_library_dir}{os.pathsep}/ambient/reviewed-lib"
        )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(runtime_v2, "_run_bounded_worker_process", worker)
    monkeypatch.setenv(VLLM_PATCHED_WHEEL_URI_ENV, "prior-uri")
    monkeypatch.setenv(VLLM_PATCHED_WHEEL_SHA256_ENV, "prior-sha")

    result = runtime_v2.run_gpu_qualification_sentinel_v2(
        plan_record=plan,
        planned_job=planned_job,
        artifact_paths=artifact_paths,
        work_dir=work_dir,
    )

    expected_commands = [
        [
            str(runtime_python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--only-binary",
            ":all:",
            "-r",
            str(artifact_paths["runtime_lock_sha256"]),
        ],
        [
            str(runtime_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            f"vllm @ {vllm_uri}#sha256={GPU_QUALIFICATION_PATCHED_WHEEL_SHA256}",
        ],
        [
            str(runtime_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "flashinfer-python @ "
            f"{flashinfer_uri}#sha256={FLASHINFER_PATCHED_WHEEL_SHA256}",
        ],
        [
            str(runtime_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            f"cachet-kv @ {package_uri}#sha256={_PACKAGE_SHA256}",
        ],
        [str(runtime_python), "-m", "pip", "check"],
    ]
    assert [call[0] for call in subprocess_calls] == expected_commands
    assert "--no-deps" not in subprocess_calls[0][0]
    assert all("--no-deps" in call[0] for call in subprocess_calls[1:4])
    for _arguments, kwargs in subprocess_calls:
        assert kwargs["check"] is True
        assert kwargs["cwd"] == runtime_dir
        assert kwargs["env"]["PYTHONSAFEPATH"] == "1"
        assert kwargs["env"]["FLASHINFER_LOGGING_LEVEL"] == "ERROR"
        assert kwargs["env"]["LD_LIBRARY_PATH"] == "/ambient/reviewed-lib"
        assert (
            kwargs["env"]["PYTHONWARNINGS"] == GPU_RUNTIME_PYTHONWARNINGS
        )
    assert [call[1]["timeout"] for call in subprocess_calls] == [3600] * 4 + [300]
    assert attest_calls == [("3.11.11", None), ("3.11.11", identity.file_binding)]
    assert events == [
        "install-1",
        "install-2",
        "install-3",
        "install-4",
        "pip-check",
        "launch-environment",
        "final-verifier",
        "input-verifier",
        "worker",
    ]
    assert result["measurements"] == {}
    runtime_verification = result["runtime_verification"]
    assert (
        runtime_verification["closed_record_sha256"]
        == sha256(
            canonical_gpu_qualification_json(
                {**runtime_verification, "closed_record_sha256": ""}
            ).encode("utf-8")
        ).hexdigest()
    )
    validate_gpu_runtime_verification_v2_record(
        runtime_verification,
        plan_record=plan,
        expected_job_id=job_id,
        expected_artifact_pins=pins,
    )

    validation_runtime = tmp_path / "validation-runtime"
    validation_torch_library = (
        validation_runtime / "lib/python3.11/site-packages/torch/lib"
    )
    validation_torch_library.mkdir(parents=True)
    install_environment = {
        "LD_LIBRARY_PATH": "/existing/one:/existing/two",
        "UNCHANGED": "yes",
    }
    assert original_launch_environment(
        runtime_dir=validation_runtime,
        install_environment=install_environment,
    ) == {
        "LD_LIBRARY_PATH": (
            f"{validation_torch_library}{os.pathsep}/existing/one:/existing/two"
        ),
        "UNCHANGED": "yes",
    }
    assert install_environment == {
        "LD_LIBRARY_PATH": "/existing/one:/existing/two",
        "UNCHANGED": "yes",
    }

    missing_runtime = tmp_path / "missing-torch-library-runtime"
    missing_runtime.mkdir()
    file_runtime = tmp_path / "file-torch-library-runtime"
    file_torch_library = file_runtime / "lib/python3.11/site-packages/torch/lib"
    file_torch_library.parent.mkdir(parents=True)
    file_torch_library.write_bytes(b"not-a-directory")
    linked_runtime = tmp_path / "linked-torch-library-runtime"
    linked_torch_library = (
        linked_runtime / "lib/python3.11/site-packages/torch/lib"
    )
    linked_torch_library.parent.mkdir(parents=True)
    linked_target = tmp_path / "linked-torch-library-target"
    linked_target.mkdir()
    linked_torch_library.symlink_to(linked_target, target_is_directory=True)
    escaping_runtime = tmp_path / "escaping-torch-library-runtime"
    escaping_torch = escaping_runtime / "lib/python3.11/site-packages/torch"
    escaping_torch.parent.mkdir(parents=True)
    escaping_target = tmp_path / "escaping-torch-target"
    (escaping_target / "lib").mkdir(parents=True)
    escaping_torch.symlink_to(escaping_target, target_is_directory=True)
    linked_runtime_root = tmp_path / "linked-runtime-root"
    linked_runtime_root.symlink_to(validation_runtime, target_is_directory=True)
    noncanonical_runtime = (
        validation_runtime / ".." / validation_runtime.name
    )
    for rejected_runtime in (
        missing_runtime,
        file_runtime,
        linked_runtime,
        escaping_runtime,
        linked_runtime_root,
        noncanonical_runtime,
    ):
        with pytest.raises(
            RuntimeError, match="v2 isolated torch library directory differs"
        ):
            original_launch_environment(
                runtime_dir=rejected_runtime,
                install_environment={},
            )


@pytest.mark.parametrize("tamper", ["plan", "job"])
def test_runtime_installer_rejects_nonexact_sealed_plan_or_job_before_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
) -> None:
    plan = _plan()
    job_id = plan["cloud_qualification"]["jobs"][0]["job_id"]
    planned_job = deepcopy(qualification_v1._plan_job(plan, job_id))
    if tamper == "plan":
        plan["cloud_qualification"]["jobs"][0]["gpu"] = "NVIDIA TAMPER"
        _seal_v2(plan)
    else:
        planned_job["gpu"] = "NVIDIA TAMPER"

    def unexpected_install(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("invalid plan/job reached installation")

    monkeypatch.setattr(runtime_v2.subprocess, "run", unexpected_install)
    monkeypatch.setattr(runtime_v2, "create_venv", unexpected_install)
    with pytest.raises(ValueError):
        runtime_v2.run_gpu_qualification_sentinel_v2(
            plan_record=plan,
            planned_job=planned_job,
            artifact_paths={
                key: tmp_path / key for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS
            },
            work_dir=tmp_path / "work",
        )


def test_standalone_verifier_pip_check_is_exact_bounded_binary_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "PIP_CONFIG_FILE": "/reviewed/pip.conf",
        "PYTHONWARNINGS": "ignore",
    }
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(runtime_v2, "_require_runtime_platform", lambda: None)
    monkeypatch.setattr(
        runtime_v2, "_pip_subprocess_environment", lambda: dict(environment)
    )

    def run(
        arguments: list[str], **kwargs: Any
    ) -> runtime_v2._BoundedBinarySubprocessResult:
        calls.append((list(arguments), dict(kwargs)))
        return _bounded_process_result()

    monkeypatch.setattr(runtime_v2, "_run_bounded_binary_subprocess", run)
    monkeypatch.setattr(runtime_v2, "_file_sha256", lambda _path: "0" * 64)
    with pytest.raises(RuntimeError, match="base lock SHA-256 differs"):
        runtime_v2.verify_gpu_qualification_v2_runtime_installation(
            runtime_lock="base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            runtime_closure_manifest="closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
        )

    assert calls == [
        (
            [runtime_v2.sys.executable, "-m", "pip", "check"],
            {
                "cwd": Path(runtime_v2.sys.prefix),
                "environment": {
                    **environment,
                    "FLASHINFER_LOGGING_LEVEL": (
                        GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL
                    ),
                    "PYTHONSAFEPATH": "1",
                    "PYTHONWARNINGS": GPU_RUNTIME_PYTHONWARNINGS,
                },
                "output_limit_bytes": (
                    runtime_v2._FINAL_VERIFIER_PROCESS_OUTPUT_LIMIT_BYTES
                ),
                "timeout_seconds": (
                    runtime_v2._FINAL_VERIFIER_INNER_PIP_TIMEOUT_SECONDS
                ),
            },
        )
    ]


def test_standalone_verifier_strips_private_pip_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setenv("PIP_INDEX_URL", "https://attacker.invalid/simple")
    monkeypatch.setenv("_PIP_USE_IMPORTLIB_METADATA", "0")
    monkeypatch.setenv("_pip_standalone_cert", "/attacker/ca.pem")
    monkeypatch.setattr(runtime_v2, "_require_runtime_platform", lambda: None)

    def run(
        _arguments: list[str], **kwargs: Any
    ) -> runtime_v2._BoundedBinarySubprocessResult:
        calls.append(dict(kwargs))
        return _bounded_process_result()

    monkeypatch.setattr(runtime_v2, "_run_bounded_binary_subprocess", run)
    monkeypatch.setattr(runtime_v2, "_file_sha256", lambda _path: "0" * 64)
    with pytest.raises(RuntimeError, match="base lock SHA-256 differs"):
        runtime_v2.verify_gpu_qualification_v2_runtime_installation(
            runtime_lock="base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            runtime_closure_manifest="closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
        )

    assert len(calls) == 1
    environment = calls[0]["environment"]
    assert {
        key: value
        for key, value in environment.items()
        if key.upper().startswith(("PIP_", "_PIP_"))
    } == {
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
    }
    assert environment["PYTHONSAFEPATH"] == "1"


def test_standalone_verifier_rejects_private_pip_warning_without_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = (
        b"DEPRECATION: Using the pkg_resources metadata backend is deprecated. "
        b"A possible replacement is to unset _PIP_USE_IMPORTLIB_METADATA.\n"
    )
    monkeypatch.setattr(runtime_v2, "_require_runtime_platform", lambda: None)
    monkeypatch.setattr(runtime_v2, "_pip_subprocess_environment", lambda: {})
    monkeypatch.setattr(
        runtime_v2,
        "_run_bounded_binary_subprocess",
        lambda *_args, **_kwargs: _bounded_process_result(stderr=warning),
    )

    with pytest.raises(RuntimeError, match="pip check output differs") as raised:
        runtime_v2.verify_gpu_qualification_v2_runtime_installation(
            runtime_lock="base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            runtime_closure_manifest="closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
        )

    assert "pkg_resources" not in str(raised.value)
    assert "_PIP_USE_IMPORTLIB_METADATA" not in str(raised.value)


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        (b"No broken requirements found.\r\n", b""),
        (b"No broken requirements found.\nextra\n", b""),
        (b"No broken requirements found.\n", b"stderr-secret"),
    ],
)
def test_standalone_verifier_rejects_pip_check_marker_mismatch_without_leak(
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes,
    stderr: bytes,
) -> None:
    monkeypatch.setattr(runtime_v2, "_require_runtime_platform", lambda: None)
    monkeypatch.setattr(runtime_v2, "_pip_subprocess_environment", lambda: {})
    monkeypatch.setattr(
        runtime_v2,
        "_run_bounded_binary_subprocess",
        lambda *_args, **_kwargs: _bounded_process_result(stdout=stdout, stderr=stderr),
    )
    with pytest.raises(RuntimeError, match="pip check output differs") as raised:
        runtime_v2.verify_gpu_qualification_v2_runtime_installation(
            runtime_lock="base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            runtime_closure_manifest="closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
        )
    assert "stderr-secret" not in str(raised.value)


def test_gpu_runtime_pinned_warning_prefix_policy_is_fail_closed() -> None:
    messages = (
        (
            "The cuda.cuda module is deprecated and will be removed in a future "
            "release, please switch to use the cuda.bindings.driver module instead."
        ),
        (
            "The cuda.cudart module is deprecated and will be removed in a future "
            "release, please switch to use the cuda.bindings.runtime module instead."
        ),
        (
            "The cuda.nvrtc module is deprecated and will be removed in a future "
            "release, please switch to use the cuda.bindings.nvrtc module instead."
        ),
    )
    transcript = b"".join(
        (
            "<frozen importlib._bootstrap_external>:1241: FutureWarning: "
            f"{message}\n"
        ).encode("utf-8")
        for message in messages
    )
    assert len(transcript) == 597
    assert (
        sha256(transcript).hexdigest()
        == "5ce62998bdbb2f2c0e3d268b77a220f1725deaf63143544be8c8bf24536dfdbe"
    )
    assert GPU_RUNTIME_PYTHONWARNINGS.split(",") == [
        "error",
        *(
            "ignore:"
            f"{message.partition(',')[0]}"
            ":FutureWarning:importlib._bootstrap_external:1241"
            for message in messages
        ),
    ]

    child_code = "\n".join(
        (
            "import warnings",
            f"messages = {messages!r}",
            "for message in messages:",
            "    warnings.warn_explicit(",
            "        message, FutureWarning,",
            '        "<frozen importlib._bootstrap_external>", 1241,',
            '        module="importlib._bootstrap_external",',
            "    )",
            "cases = (",
            "    (",
            '        "an unrelated future warning", FutureWarning,',
            '        "importlib._bootstrap_external", 1241,',
            "    ),",
            "    (messages[0], RuntimeWarning, ",
            '     "importlib._bootstrap_external", 1241),',
            "    (messages[0], FutureWarning, ",
            '     "another.module", 1241),',
            "    (messages[0], FutureWarning, ",
            '     "importlib._bootstrap_external", 1242),',
            "    (",
            '        "The cuda.cudaX module is deprecated and will be removed in a "',
            '        "future release, near-prefix mutation", FutureWarning,',
            '        "importlib._bootstrap_external", 1241,',
            "    ),",
            ")",
            "for message, category, module, lineno in cases:",
            "    try:",
            "        warnings.warn_explicit(",
            '            message, category, "<frozen importlib._bootstrap_external>",',
            "            lineno, module=module,",
            "        )",
            "    except Warning:",
            "        continue",
            '    raise AssertionError("non-allowlisted warning did not raise")',
            'print("policy-ok")',
        )
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["FLASHINFER_LOGGING_LEVEL"] = (
        GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL
    )
    environment["PYTHONWARNINGS"] = GPU_RUNTIME_PYTHONWARNINGS
    completed = subprocess.run(
        [runtime_v2.sys.executable, "-c", child_code],
        check=False,
        capture_output=True,
        env=environment,
    )
    assert (completed.returncode, completed.stdout, completed.stderr) == (
        0,
        b"policy-ok\n",
        b"",
    )


@pytest.mark.parametrize("failure", ["nonzero", "timeout"])
def test_standalone_verifier_pip_failure_is_fixed_and_does_not_leak(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    monkeypatch.setattr(runtime_v2, "_require_runtime_platform", lambda: None)
    monkeypatch.setattr(runtime_v2, "_pip_subprocess_environment", lambda: {})

    def run(*_args: Any, **_kwargs: Any) -> runtime_v2._BoundedBinarySubprocessResult:
        if failure == "timeout":
            return _bounded_process_result(
                stdout=b"stdout-secret",
                stderr=b"stderr-secret",
                timed_out=True,
            )
        return _bounded_process_result(
            returncode=17,
            stdout=b"stdout-secret",
            stderr=b"stderr-secret",
        )

    monkeypatch.setattr(runtime_v2, "_run_bounded_binary_subprocess", run)
    expected = "timed out" if failure == "timeout" else "pip check failed"
    with pytest.raises(RuntimeError, match=expected) as raised:
        runtime_v2.verify_gpu_qualification_v2_runtime_installation(
            runtime_lock="base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            runtime_closure_manifest="closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
        )
    diagnostic = str(raised.value)
    assert "command-secret" not in diagnostic
    assert "stdout-secret" not in diagnostic
    assert "stderr-secret" not in diagnostic


def test_bounded_binary_subprocess_stops_incremental_oversize_and_caps_retention(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_popen = subprocess.Popen
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def popen(arguments: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        calls.append((list(arguments), dict(kwargs)))
        return real_popen(arguments, **kwargs)

    monkeypatch.setattr(runtime_v2.subprocess, "Popen", popen)
    environment = {"PYTHONSAFEPATH": "1"}
    arguments = [
        runtime_v2.sys.executable,
        "-c",
        "import os\nchunk=b'x'*65536\nwhile True: os.write(1,chunk)",
    ]
    result = runtime_v2._run_bounded_binary_subprocess(
        arguments,
        timeout_seconds=5,
        output_limit_bytes=4096,
        environment=environment,
        cwd=tmp_path,
    )

    assert result.timed_out is False
    assert result.output_limit_exceeded is True
    assert result.stdout.byte_count > 4096
    assert result.stdout.retained == b"x" * 4096
    assert result.stdout.sha256 != runtime_v2._FINAL_VERIFIER_EMPTY_STREAM_SHA256
    assert result.stderr == _bounded_stream_result(b"")
    assert len(calls) == 1
    observed_arguments, kwargs = calls[0]
    assert observed_arguments == arguments
    assert kwargs == {
        "bufsize": 0,
        "cwd": tmp_path,
        "env": environment,
        "start_new_session": True,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "text": False,
    }


def test_bounded_binary_subprocess_timeout_terminates_then_kills_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_signal = runtime_v2._signal_bounded_subprocess_group
    signals: list[tuple[int, int]] = []

    def record_signal(process: subprocess.Popen[bytes], signal_number: int) -> None:
        signals.append((process.pid, signal_number))
        real_signal(process, signal_number)

    monkeypatch.setattr(runtime_v2, "_signal_bounded_subprocess_group", record_signal)
    result = runtime_v2._run_bounded_binary_subprocess(
        [
            runtime_v2.sys.executable,
            "-c",
            (
                "import signal,time;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "time.sleep(60)"
            ),
        ],
        timeout_seconds=1,
        output_limit_bytes=4096,
        environment={"PYTHONSAFEPATH": "1"},
        cwd=tmp_path,
    )

    assert result.timed_out is True
    assert result.output_limit_exceeded is False
    assert [signal_number for _pid, signal_number in signals[:2]] == [
        signal.SIGTERM,
        signal.SIGKILL,
    ]
    process_id = signals[0][0]
    assert (
        runtime_v2._bounded_subprocess_group_exists(SimpleNamespace(pid=process_id))
        is False
    )


def test_final_verifier_timeout_hierarchy_has_strict_cleanup_and_import_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runtime_v2._FINAL_VERIFIER_INNER_PIP_TIMEOUT_SECONDS == 180
    assert runtime_v2._FINAL_VERIFIER_OUTER_TIMEOUT_SECONDS == 300
    assert runtime_v2._FINAL_VERIFIER_INNER_CLEANUP_BUDGET_SECONDS == 10
    assert runtime_v2._FINAL_VERIFIER_POST_PIP_BUDGET_SECONDS == 60
    assert runtime_v2._FINAL_VERIFIER_REQUIRED_HIERARCHY_MARGIN_SECONDS == 90
    runtime_v2._require_final_verifier_timeout_hierarchy()

    monkeypatch.setattr(runtime_v2, "_FINAL_VERIFIER_INNER_PIP_TIMEOUT_SECONDS", 211.0)
    with pytest.raises(RuntimeError, match="timeout hierarchy differs"):
        runtime_v2._require_final_verifier_timeout_hierarchy()


def _process_or_group_exists(identifier: int, *, group: bool) -> bool:
    try:
        if group:
            os.killpg(identifier, 0)
        else:
            os.kill(identifier, 0)
    except (PermissionError, ProcessLookupError):
        return False
    return True


def test_outer_final_verifier_waits_for_nested_pip_group_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inner_marker = tmp_path / "inner-pid-pgid.txt"
    outer_marker = tmp_path / "outer-pid-pgid.txt"
    fake_python = tmp_path / "term-ignoring-python"
    fake_python.write_text(
        f"#!{runtime_v2.sys.executable}\n"
        "import os\n"
        "import signal\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "marker = os.environ['CACHET_TEST_INNER_MARKER']\n"
        "with open(marker, 'w', encoding='ascii') as stream:\n"
        "    stream.write(f'{os.getpid()} {os.getpgid(0)}\\n')\n"
        "    stream.flush()\n"
        "    os.fsync(stream.fileno())\n"
        "while True:\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)

    hook_dir = tmp_path / "site-hook"
    hook_dir.mkdir()
    (hook_dir / "sitecustomize.py").write_text(
        "import os\n"
        "import document_kv_cache._gpu_qualification_sentinels_v2 as runtime_v2\n"
        "runtime_v2._require_runtime_platform = lambda: None\n"
        "runtime_v2._FINAL_VERIFIER_INNER_PIP_TIMEOUT_SECONDS = 1.5\n"
        "runtime_v2.sys.executable = os.environ['CACHET_TEST_FAKE_PYTHON']\n"
        "marker = os.environ['CACHET_TEST_OUTER_MARKER']\n"
        "with open(marker, 'w', encoding='ascii') as stream:\n"
        "    stream.write(f'{os.getpid()} {os.getpgid(0)}\\n')\n"
        "    stream.flush()\n"
        "    os.fsync(stream.fileno())\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime_v2, "_FINAL_VERIFIER_INNER_PIP_TIMEOUT_SECONDS", 1.5)
    monkeypatch.setattr(runtime_v2, "_FINAL_VERIFIER_OUTER_TIMEOUT_SECONDS", 7.0)
    monkeypatch.setattr(runtime_v2, "_FINAL_VERIFIER_INNER_CLEANUP_BUDGET_SECONDS", 1.5)
    monkeypatch.setattr(runtime_v2, "_FINAL_VERIFIER_POST_PIP_BUDGET_SECONDS", 0.5)
    monkeypatch.setattr(
        runtime_v2, "_FINAL_VERIFIER_REQUIRED_HIERARCHY_MARGIN_SECONDS", 4.0
    )
    environment = {
        "CACHET_TEST_FAKE_PYTHON": str(fake_python),
        "CACHET_TEST_INNER_MARKER": str(inner_marker),
        "CACHET_TEST_OUTER_MARKER": str(outer_marker),
        "HOME": str(tmp_path),
        "LC_ALL": "C",
        "PATH": os.environ["PATH"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONEXECUTABLE": str(fake_python),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((str(hook_dir), str(_ROOT / "src"))),
        "PYTHONSAFEPATH": "1",
    }

    started = runtime_v2.monotonic()
    with pytest.raises(
        RuntimeError,
        match=r"rejected the installation \(pip_check/subprocess_timeout;",
    ):
        runtime_v2._run_final_runtime_verifier(
            Path(runtime_v2.sys.executable),
            runtime_lock=tmp_path / "unused-base.lock",
            vllm_uri="file:///unused-vllm.whl",
            flashinfer_uri="file:///unused-flashinfer.whl",
            closure_path=tmp_path / "unused-closure.json",
            package_uri="file:///unused-cachet.whl",
            package_sha256=_PACKAGE_SHA256,
            environment=environment,
        )
    assert runtime_v2.monotonic() - started < 7

    outer_pid, outer_pgid = (
        int(value) for value in outer_marker.read_text(encoding="ascii").split()
    )
    inner_pid, inner_pgid = (
        int(value) for value in inner_marker.read_text(encoding="ascii").split()
    )
    assert outer_pid == outer_pgid
    assert inner_pid == inner_pgid
    assert not _process_or_group_exists(outer_pid, group=False)
    assert not _process_or_group_exists(outer_pgid, group=True)
    assert not _process_or_group_exists(inner_pid, group=False)
    assert not _process_or_group_exists(inner_pgid, group=True)


def test_bounded_binary_accumulator_retains_only_its_cap_incrementally() -> None:
    accumulator = runtime_v2._BoundedBinaryAccumulator(7)
    assert accumulator.add(b"abc") is False
    assert accumulator.add(b"defgh") is True
    result = accumulator.result()
    assert result.retained == b"abcdefg"
    assert result.byte_count == 8
    assert result.sha256 == sha256(b"abcdefgh").hexdigest()
    assert result.limit_exceeded is True


class _Distribution:
    def __init__(
        self,
        name: str | None,
        version: str,
        *,
        direct_url: dict[str, Any] | None = None,
        root: Path | None = None,
        omit_name: bool = False,
    ) -> None:
        self.metadata = {} if omit_name else {"Name": name}
        self.version = version
        self._direct_url = direct_url
        self._root = root

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        if self._direct_url is None:
            return None
        return json.dumps(self._direct_url)

    def locate_file(self, name: str) -> Path:
        assert name == ""
        assert self._root is not None
        return self._root


def _patch_verifier_through_distribution_scan(
    monkeypatch: pytest.MonkeyPatch,
    distributions: list[_Distribution],
) -> Path:
    explicit_roots = {
        distribution._root
        for distribution in distributions
        if distribution._root is not None
    }
    assert len(explicit_roots) <= 1
    site_packages = (
        next(iter(explicit_roots))
        if explicit_roots
        else Path(runtime_v2.__file__).resolve().parents[1]
    )
    for distribution in distributions:
        if distribution._root is None:
            distribution._root = site_packages

    monkeypatch.setattr(runtime_v2, "_require_runtime_platform", lambda: None)
    monkeypatch.setattr(runtime_v2, "_pip_subprocess_environment", lambda: {})
    monkeypatch.setattr(
        runtime_v2,
        "_run_bounded_binary_subprocess",
        lambda *_args, **_kwargs: _bounded_process_result(),
    )
    monkeypatch.setattr(
        runtime_v2, "_file_sha256", lambda _path: VLLM_RUNTIME_BASE_LOCK_SHA256
    )
    monkeypatch.setattr(
        runtime_v2, "_base_lock_projection", lambda _path: ({"base-dist": "1"}, 1)
    )
    monkeypatch.setattr(runtime_v2, "VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT", 1)
    monkeypatch.setattr(runtime_v2, "VLLM_RUNTIME_BASE_LOCK_HASH_COUNT", 1)
    monkeypatch.setattr(
        runtime_v2,
        "_read_exact_runtime_closure",
        lambda _path: {
            "closed_record_sha256": RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
        },
    )
    monkeypatch.setattr(
        runtime_v2, "_isolated_runtime_site_packages", lambda: site_packages
    )

    def installed_distributions(*, path: list[str]) -> list[_Distribution]:
        assert path == [str(site_packages)]
        return distributions

    monkeypatch.setattr(
        runtime_v2.importlib.metadata, "distributions", installed_distributions
    )
    return site_packages


@pytest.mark.parametrize(
    ("distributions", "error"),
    [
        (
            [_Distribution("base-dist", "1"), _Distribution("base_dist", "1")],
            "duplicate installed distribution",
        ),
        ([_Distribution(None, "1", omit_name=True)], "no package name"),
        ([_Distribution("base-dist", "1")], "name closure differs"),
    ],
)
def test_standalone_verifier_rejects_duplicate_anonymous_or_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
    distributions: list[_Distribution],
    error: str,
) -> None:
    site_packages = _patch_verifier_through_distribution_scan(
        monkeypatch, distributions
    )
    with pytest.raises(RuntimeError, match=error):
        runtime_v2.verify_gpu_qualification_v2_runtime_installation(
            runtime_lock="base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            runtime_closure_manifest="closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
        )
    distributions[0]._root = site_packages.parent
    with pytest.raises(RuntimeError, match="outside private site-packages"):
        runtime_v2.verify_gpu_qualification_v2_runtime_installation(
            runtime_lock="base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            runtime_closure_manifest="closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
        )


def test_pep610_validation_checks_uri_archive_hash_and_origin_rehash(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "wheel+patched.whl"
    other = tmp_path / "other.whl"
    wheel.write_bytes(b"reviewed wheel")
    other.write_bytes(b"reviewed wheel")
    digest = sha256(b"reviewed wheel").hexdigest()
    value = {
        "archive_info": {"hashes": {"sha256": digest}},
        "url": wheel.resolve().as_uri(),
    }
    distribution = _Distribution("example", "1", direct_url=value)
    assert (
        runtime_v2._validate_direct_url(
            distribution,  # type: ignore[arg-type]
            expected_uri=wheel.resolve().as_uri(),
            expected_sha256=digest,
        )
        == wheel.resolve().as_uri()
    )

    distribution._direct_url = {**value, "url": other.resolve().as_uri()}
    with pytest.raises(RuntimeError, match="direct URL differs"):
        runtime_v2._validate_direct_url(
            distribution,  # type: ignore[arg-type]
            expected_uri=wheel.resolve().as_uri(),
            expected_sha256=digest,
        )

    distribution._direct_url = {
        **value,
        "archive_info": {"hashes": {"sha256": "0" * 64}},
    }
    with pytest.raises(RuntimeError, match="archive SHA-256 differs"):
        runtime_v2._validate_direct_url(
            distribution,  # type: ignore[arg-type]
            expected_uri=wheel.resolve().as_uri(),
            expected_sha256=digest,
        )

    distribution._direct_url = value
    wheel.write_bytes(b"tampered wheel")
    with pytest.raises(RuntimeError, match="source bytes differ"):
        runtime_v2._validate_direct_url(
            distribution,  # type: ignore[arg-type]
            expected_uri=wheel.resolve().as_uri(),
            expected_sha256=digest,
        )


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("member", "installed member differs"),
        ("origin", "imported FlashInfer module origin differs"),
        ("annotation", "postponed return annotation differs"),
    ],
)
def test_standalone_verifier_rejects_member_hash_or_import_annotation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
    error: str,
) -> None:
    flashinfer_root = tmp_path / "site-packages"
    flashinfer_member = flashinfer_root / FLASHINFER_TARGET_MEMBER
    flashinfer_member.parent.mkdir(parents=True)
    flashinfer_member.write_bytes(b"reviewed FlashInfer member")
    distributions = [
        _Distribution("base-dist", "1"),
        _Distribution("cachet-kv", GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION),
        _Distribution(
            "flashinfer-python",
            FLASHINFER_PACKAGE_VERSION,
            root=flashinfer_root,
        ),
        _Distribution("vllm", GPU_QUALIFICATION_VLLM_VERSION),
    ]
    _patch_verifier_through_distribution_scan(monkeypatch, distributions)
    monkeypatch.setattr(
        runtime_v2, "GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT", 4
    )
    monkeypatch.setattr(runtime_v2, "_uri_file_sha256", lambda _uri: _PACKAGE_SHA256)
    monkeypatch.setattr(
        runtime_v2,
        "_validate_direct_url",
        lambda _distribution, *, expected_uri, expected_sha256: expected_uri,
    )
    if failure == "member":

        def member_failure(*_args: Any, **_kwargs: Any) -> dict[str, str]:
            raise RuntimeError("v2 installed member differs: reviewed.py")

        monkeypatch.setattr(runtime_v2, "_installed_member_hashes", member_failure)
    else:
        monkeypatch.setattr(
            runtime_v2,
            "_installed_member_hashes",
            lambda _distribution, expected: dict(expected),
        )
        monkeypatch.setattr(
            runtime_v2,
            "_file_sha256",
            lambda path: (
                FLASHINFER_TARGET_PATCHED_SHA256
                if Path(path).name == Path(FLASHINFER_TARGET_MEMBER).name
                else VLLM_RUNTIME_BASE_LOCK_SHA256
            ),
        )
        function = SimpleNamespace(
            __annotations__={
                "return": (
                    "wrong"
                    if failure == "annotation"
                    else GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION
                )
            }
        )
        module_path = flashinfer_member
        if failure == "origin":
            module_path = tmp_path / "shadow" / "fd_exchange.py"
            module_path.parent.mkdir()
            module_path.write_bytes(b"shadow module")
        module = SimpleNamespace(_fd_ancillary=function, __file__=str(module_path))
        monkeypatch.setattr(runtime_v2.importlib, "import_module", lambda _name: module)

    with pytest.raises(RuntimeError, match=error):
        runtime_v2.verify_gpu_qualification_v2_runtime_installation(
            runtime_lock="base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            runtime_closure_manifest="closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
        )


def test_runtime_platform_requires_exact_python_3_11_11(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runtime_v2.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(runtime_v2.platform, "python_version", lambda: "3.11.11")
    monkeypatch.setattr(runtime_v2.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime_v2.platform, "libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setattr(runtime_v2.sys, "version_info", (3, 11, 11, "final", 0))
    monkeypatch.setattr(runtime_v2.sys, "platform", "linux")
    monkeypatch.setenv(
        "FLASHINFER_LOGGING_LEVEL", GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL
    )
    monkeypatch.setenv("PYTHONWARNINGS", GPU_RUNTIME_PYTHONWARNINGS)
    monkeypatch.setattr(
        runtime_v2.sys,
        "warnoptions",
        GPU_RUNTIME_PYTHONWARNINGS.split(","),
    )

    runtime_v2._require_runtime_platform()
    monkeypatch.setattr(runtime_v2.sys, "warnoptions", ["ignore"])
    with pytest.raises(RuntimeError, match="pinned CUDA warning startup policy"):
        runtime_v2._require_runtime_platform()
    monkeypatch.setattr(
        runtime_v2.sys,
        "warnoptions",
        GPU_RUNTIME_PYTHONWARNINGS.split(","),
    )
    monkeypatch.setattr(runtime_v2.platform, "python_version", lambda: "3.11.12")
    with pytest.raises(RuntimeError, match="requires Linux CPython3.11"):
        runtime_v2._require_runtime_platform()

    runtime_root = tmp_path.resolve() / "runtime"
    runtime_python = runtime_root / "bin/python"
    site_packages = runtime_root / "lib/python3.11/site-packages"
    verifier_source = site_packages / (
        "document_kv_cache/_gpu_qualification_sentinels_v2.py"
    )
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_bytes(b"reviewed copied interpreter")
    verifier_source.parent.mkdir(parents=True)
    verifier_source.write_bytes(b"reviewed verifier")
    monkeypatch.setattr(runtime_v2.sys, "prefix", str(runtime_root))
    monkeypatch.setattr(runtime_v2.sys, "base_prefix", "/reviewed/base-python")
    monkeypatch.setattr(runtime_v2.sys, "executable", str(runtime_python))
    monkeypatch.setattr(runtime_v2, "__file__", str(verifier_source))
    assert runtime_v2._isolated_runtime_site_packages() == site_packages

    monkeypatch.setattr(runtime_v2, "__file__", str(tmp_path / "ambient.py"))
    with pytest.raises(RuntimeError, match="runtime path identity differs"):
        runtime_v2._isolated_runtime_site_packages()


def test_real_runtime_closure_has_exact_identity_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    raw = _CLOSURE_PATH.read_bytes()
    assert len(raw) == RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE == 6634
    assert sha256(raw).hexdigest() == RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
    closure = runtime_v2._read_exact_runtime_closure(_CLOSURE_PATH)
    assert closure["closed_record_sha256"] == (
        RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
    )

    tampered = tmp_path / _CLOSURE_PATH.name
    tampered.write_bytes(raw[:-2] + b" \n")
    assert tampered.stat().st_size == 6634
    with pytest.raises(ValueError, match="file identity differs"):
        runtime_v2._read_exact_runtime_closure(tampered)


def test_real_base_lock_has_exact_projection_and_verifier_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = _LOCK_PATH.read_bytes()
    assert len(raw) == VLLM_RUNTIME_BASE_LOCK_SIZE == 376326
    assert sha256(raw).hexdigest() == VLLM_RUNTIME_BASE_LOCK_SHA256
    versions, hash_count = runtime_v2._base_lock_projection(_LOCK_PATH)
    assert len(versions) == VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT == 195
    assert hash_count == VLLM_RUNTIME_BASE_LOCK_HASH_COUNT == 4137
    assert not {"cachet-kv", "flashinfer-python", "vllm"} & versions.keys()

    tampered = tmp_path / VLLM_RUNTIME_BASE_LOCK_FILENAME
    tampered.write_bytes(raw[:-1] + (b" " if raw[-1:] != b" " else b"\n"))
    assert tampered.stat().st_size == 376326
    monkeypatch.setattr(runtime_v2, "_require_runtime_platform", lambda: None)
    monkeypatch.setattr(runtime_v2, "_pip_subprocess_environment", lambda: {})
    monkeypatch.setattr(
        runtime_v2,
        "_run_bounded_binary_subprocess",
        lambda *_args, **_kwargs: _bounded_process_result(),
    )
    with pytest.raises(RuntimeError, match="base lock SHA-256 differs"):
        runtime_v2.verify_gpu_qualification_v2_runtime_installation(
            runtime_lock=tampered,
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            runtime_closure_manifest=_CLOSURE_PATH,
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
        )


def _final_child_arguments() -> list[str]:
    return [
        "base.lock",
        "file:///vllm.whl",
        "file:///flashinfer.whl",
        "closure.json",
        "file:///cachet.whl",
        _PACKAGE_SHA256,
    ]


def _final_child_success_envelope(attestation: dict[str, Any]) -> dict[str, Any]:
    return {
        "attestation": attestation,
        "category": "none",
        "ok": True,
        "record_type": runtime_v2._FINAL_VERIFIER_CHILD_RECORD_TYPE,
        "schema_version": runtime_v2._FINAL_VERIFIER_CHILD_SCHEMA_VERSION,
        "stage": "complete",
        "stderr_bytes": 0,
        "stderr_sha256": runtime_v2._FINAL_VERIFIER_EMPTY_STREAM_SHA256,
        "stdout_bytes": 0,
        "stdout_sha256": runtime_v2._FINAL_VERIFIER_EMPTY_STREAM_SHA256,
    }


def test_final_verifier_child_success_envelope_is_canonical_and_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attestation = {"ok": True, "reviewed": "attestation"}

    def verify(**kwargs: Any) -> dict[str, Any]:
        callback = kwargs["stage_callback"]
        callback("complete")
        return attestation

    monkeypatch.setattr(
        runtime_v2, "_verify_gpu_qualification_v2_runtime_installation", verify
    )
    envelope = runtime_v2._final_runtime_verifier_child_envelope(
        _final_child_arguments()
    )
    assert envelope == _final_child_success_envelope(attestation)
    encoded = runtime_v2._canonical_final_runtime_verifier_child_envelope(envelope)
    assert encoded.endswith(b"\n")
    assert runtime_v2._parse_final_runtime_verifier_child_envelope(encoded) == (
        envelope
    )


@pytest.mark.parametrize(
    ("failure", "category"),
    [("nonzero", "subprocess_nonzero"), ("timeout", "subprocess_timeout")],
)
def test_final_verifier_child_failure_hashes_streams_and_never_leaks(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    category: str,
) -> None:
    captured_stdout = b"captured-stdout-secret"
    captured_stderr = b"captured-stderr-secret"

    def verify(**kwargs: Any) -> dict[str, Any]:
        callback = kwargs["stage_callback"]
        callback("pip_check")
        assert os.write(1, captured_stdout) == len(captured_stdout)
        assert os.write(2, captured_stderr) == len(captured_stderr)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                cmd=["exception-command-secret"],
                timeout=300,
                output=b"exception-stdout-secret",
                stderr=b"exception-stderr-secret",
            )
        raise subprocess.CalledProcessError(
            19,
            ["exception-command-secret"],
            output=b"exception-stdout-secret",
            stderr=b"exception-stderr-secret",
        )

    monkeypatch.setattr(
        runtime_v2, "_verify_gpu_qualification_v2_runtime_installation", verify
    )
    envelope = runtime_v2._final_runtime_verifier_child_envelope(
        _final_child_arguments()
    )
    encoded = runtime_v2._canonical_final_runtime_verifier_child_envelope(envelope)
    assert envelope["ok"] is False
    assert envelope["attestation"] is None
    assert envelope["stage"] == "pip_check"
    assert envelope["category"] == category
    assert envelope["stdout_bytes"] == len(captured_stdout)
    assert envelope["stdout_sha256"] == sha256(captured_stdout).hexdigest()
    assert envelope["stderr_bytes"] == len(captured_stderr)
    assert envelope["stderr_sha256"] == sha256(captured_stderr).hexdigest()
    for secret in (
        captured_stdout,
        captured_stderr,
        b"exception-command-secret",
        b"exception-stdout-secret",
        b"exception-stderr-secret",
    ):
        assert secret not in encoded
    assert runtime_v2._parse_final_runtime_verifier_child_envelope(encoded) == (
        envelope
    )


def test_final_verifier_child_incremental_oversize_is_bounded_and_contaminated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"captured-oversize-secret-" + b"x" * (
        runtime_v2._FINAL_VERIFIER_PROCESS_OUTPUT_LIMIT_BYTES + 4096
    )

    def verify(**kwargs: Any) -> dict[str, Any]:
        callback = kwargs["stage_callback"]
        callback("complete")
        offset = 0
        while offset < len(payload):
            offset += os.write(1, payload[offset:])
        return {"ok": True}

    monkeypatch.setattr(
        runtime_v2, "_verify_gpu_qualification_v2_runtime_installation", verify
    )
    envelope = runtime_v2._final_runtime_verifier_child_envelope(
        _final_child_arguments()
    )
    encoded = runtime_v2._canonical_final_runtime_verifier_child_envelope(envelope)
    assert envelope["ok"] is False
    assert envelope["attestation"] is None
    assert envelope["stage"] == "attestation"
    assert envelope["category"] == "verification_rejected"
    assert envelope["stdout_bytes"] == len(payload)
    assert envelope["stdout_sha256"] == sha256(payload).hexdigest()
    assert b"captured-oversize-secret" not in encoded
    assert runtime_v2._parse_final_runtime_verifier_child_envelope(encoded) == (
        envelope
    )


def test_final_verifier_child_invalid_arguments_have_exact_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_v2,
        "_verify_gpu_qualification_v2_runtime_installation",
        lambda **_kwargs: pytest.fail("invalid arguments reached verification"),
    )
    envelope = runtime_v2._final_runtime_verifier_child_envelope(["incomplete"])
    assert envelope["ok"] is False
    assert envelope["stage"] == "arguments"
    assert envelope["category"] == "invalid_arguments"
    encoded = runtime_v2._canonical_final_runtime_verifier_child_envelope(envelope)
    assert runtime_v2._parse_final_runtime_verifier_child_envelope(encoded) == (
        envelope
    )


@pytest.mark.parametrize("failure", ["import", "os"])
def test_final_verifier_child_maps_ordinary_import_or_os_error_to_rejection(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    def verify(**kwargs: Any) -> dict[str, Any]:
        callback = kwargs["stage_callback"]
        callback("flashinfer_import")
        if failure == "import":
            raise ImportError("import-error-secret")
        raise OSError("os-error-secret-path")

    monkeypatch.setattr(
        runtime_v2, "_verify_gpu_qualification_v2_runtime_installation", verify
    )
    envelope = runtime_v2._final_runtime_verifier_child_envelope(
        _final_child_arguments()
    )
    encoded = runtime_v2._canonical_final_runtime_verifier_child_envelope(envelope)
    assert envelope["ok"] is False
    assert envelope["stage"] == "flashinfer_import"
    assert envelope["category"] == "verification_rejected"
    assert b"import-error-secret" not in encoded
    assert b"os-error-secret-path" not in encoded
    assert runtime_v2._parse_final_runtime_verifier_child_envelope(encoded) == (
        envelope
    )


def test_final_verifier_child_classifies_pip_subprocess_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_v2, "_require_runtime_platform", lambda: None)
    monkeypatch.setattr(runtime_v2, "_pip_subprocess_environment", lambda: {})

    def fail_start(*_args: Any, **_kwargs: Any) -> None:
        raise runtime_v2._BoundedSubprocessStartFailure("start-failure-secret")

    monkeypatch.setattr(runtime_v2, "_run_bounded_binary_subprocess", fail_start)
    envelope = runtime_v2._final_runtime_verifier_child_envelope(
        _final_child_arguments()
    )
    encoded = runtime_v2._canonical_final_runtime_verifier_child_envelope(envelope)
    assert envelope["ok"] is False
    assert envelope["stage"] == "pip_check"
    assert envelope["category"] == "subprocess_start_failure"
    assert b"start-failure-secret" not in encoded
    assert runtime_v2._parse_final_runtime_verifier_child_envelope(encoded) == (
        envelope
    )


def test_final_verifier_parent_accepts_success_and_reports_rejection_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_python = tmp_path / "runtime" / "bin" / "python"
    attestation = {"ok": True, "reviewed": "attestation"}
    encoded = runtime_v2._canonical_final_runtime_verifier_child_envelope(
        _final_child_success_envelope(attestation)
    )
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(
        arguments: list[str], **kwargs: Any
    ) -> runtime_v2._BoundedBinarySubprocessResult:
        calls.append((list(arguments), dict(kwargs)))
        return _bounded_process_result(stdout=encoded)

    monkeypatch.setattr(runtime_v2, "_run_bounded_binary_subprocess", run)
    observed = runtime_v2._run_final_runtime_verifier(
        runtime_python,
        runtime_lock=tmp_path / "base.lock",
        vllm_uri="file:///vllm.whl",
        flashinfer_uri="file:///flashinfer.whl",
        closure_path=tmp_path / "closure.json",
        package_uri="file:///cachet.whl",
        package_sha256=_PACKAGE_SHA256,
        environment={"PYTHONSAFEPATH": "1"},
    )
    assert observed == attestation
    assert len(calls) == 1
    arguments, kwargs = calls[0]
    assert arguments[0:2] == [str(runtime_python), "-c"]
    assert "_final_runtime_verifier_child_main" in arguments[2]
    assert arguments[3:] == [
        str(tmp_path / "base.lock"),
        "file:///vllm.whl",
        "file:///flashinfer.whl",
        str(tmp_path / "closure.json"),
        "file:///cachet.whl",
        _PACKAGE_SHA256,
    ]
    assert kwargs == {
        "cwd": runtime_python.parent.parent,
        "environment": {"PYTHONSAFEPATH": "1"},
        "output_limit_bytes": (runtime_v2._FINAL_VERIFIER_PROCESS_OUTPUT_LIMIT_BYTES),
        "timeout_seconds": runtime_v2._FINAL_VERIFIER_OUTER_TIMEOUT_SECONDS,
    }

    stdout = b"authenticated-stdout-secret"
    stderr = b"authenticated-stderr-secret"
    rejected = runtime_v2._final_runtime_verifier_failure_envelope(
        stage="flashinfer_import",
        category="verification_rejected",
        stdout_bytes=len(stdout),
        stdout_sha256=sha256(stdout).hexdigest(),
        stderr_bytes=len(stderr),
        stderr_sha256=sha256(stderr).hexdigest(),
    )
    encoded_rejection = runtime_v2._canonical_final_runtime_verifier_child_envelope(
        rejected
    )
    monkeypatch.setattr(
        runtime_v2,
        "_run_bounded_binary_subprocess",
        lambda *_args, **_kwargs: _bounded_process_result(stdout=encoded_rejection),
    )
    with pytest.raises(RuntimeError, match="rejected the installation") as raised:
        runtime_v2._run_final_runtime_verifier(
            runtime_python,
            runtime_lock=tmp_path / "base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            closure_path=tmp_path / "closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
            environment={"PYTHONSAFEPATH": "1"},
        )
    diagnostic = str(raised.value)
    assert diagnostic == (
        "v2 final runtime verifier rejected the installation "
        f"(flashinfer_import/verification_rejected; stdout_bytes={len(stdout)}; "
        f"stdout_sha256={sha256(stdout).hexdigest()}; stderr_bytes={len(stderr)}; "
        f"stderr_sha256={sha256(stderr).hexdigest()})"
    )
    assert stdout.decode("ascii") not in diagnostic
    assert stderr.decode("ascii") not in diagnostic


def _malformed_final_child_outputs() -> list[tuple[str, bytes]]:
    empty_digest = runtime_v2._FINAL_VERIFIER_EMPTY_STREAM_SHA256
    envelope = runtime_v2._final_runtime_verifier_failure_envelope(
        stage="pip_check",
        category="verification_rejected",
        stdout_bytes=0,
        stdout_sha256=empty_digest,
        stderr_bytes=0,
        stderr_sha256=empty_digest,
    )
    canonical = runtime_v2._canonical_final_runtime_verifier_child_envelope(envelope)
    duplicate = canonical.replace(
        b'{"attestation":null,',
        b'{"attestation":null,"attestation":null,',
        1,
    )
    nan = canonical.replace(b'"attestation":null', b'"attestation":NaN', 1)
    extra = runtime_v2._canonical_final_runtime_verifier_child_envelope(
        {**envelope, "extra": None}
    )
    noncanonical = (json.dumps(envelope, indent=2, sort_keys=True) + "\n").encode()
    return [
        ("duplicate", duplicate),
        ("nan", nan),
        ("extra", extra),
        ("noncanonical", noncanonical),
        ("empty", b""),
        ("missing-lf", canonical[:-1]),
        ("prefix", b"polluted-prefix" + canonical),
        ("suffix", canonical + b"polluted-suffix"),
        ("two-objects", canonical + canonical),
    ]


@pytest.mark.parametrize(
    ("_case", "raw"),
    _malformed_final_child_outputs(),
    ids=[case for case, _raw in _malformed_final_child_outputs()],
)
def test_final_verifier_parser_rejects_malformed_or_polluted_output(
    _case: str,
    raw: bytes,
) -> None:
    with pytest.raises(RuntimeError, match="protocol failed"):
        runtime_v2._parse_final_runtime_verifier_child_envelope(raw)


@pytest.mark.parametrize(
    ("stage", "category"),
    [
        ("complete", "verification_rejected"),
        ("arguments", "subprocess_timeout"),
        ("platform", "invalid_arguments"),
        ("base_lock", "subprocess_nonzero"),
        ("attestation", "subprocess_start_failure"),
    ],
)
def test_final_verifier_parser_rejects_invalid_stage_category_relation(
    stage: str,
    category: str,
) -> None:
    empty_digest = runtime_v2._FINAL_VERIFIER_EMPTY_STREAM_SHA256
    envelope = runtime_v2._final_runtime_verifier_failure_envelope(
        stage=stage,
        category=category,
        stdout_bytes=0,
        stdout_sha256=empty_digest,
        stderr_bytes=0,
        stderr_sha256=empty_digest,
    )
    encoded = runtime_v2._canonical_final_runtime_verifier_child_envelope(envelope)
    with pytest.raises(RuntimeError, match="protocol failed"):
        runtime_v2._parse_final_runtime_verifier_child_envelope(encoded)


@pytest.mark.parametrize("failure", ["stderr", "oversize"])
def test_final_verifier_parent_rejects_stderr_or_oversize_without_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    encoded = runtime_v2._canonical_final_runtime_verifier_child_envelope(
        _final_child_success_envelope({"ok": True})
    )
    result = _bounded_process_result(
        stdout=encoded,
        stderr=b"parent-stderr-secret" if failure == "stderr" else b"",
    )
    if failure == "oversize":
        result = runtime_v2._BoundedBinarySubprocessResult(
            returncode=0,
            stdout=_bounded_stream_result(
                b"oversize-secret",
                byte_count=(runtime_v2._FINAL_VERIFIER_PROCESS_OUTPUT_LIMIT_BYTES + 1),
                digest="d" * 64,
                limit_exceeded=True,
            ),
            stderr=_bounded_stream_result(b""),
            timed_out=False,
        )
    monkeypatch.setattr(
        runtime_v2,
        "_run_bounded_binary_subprocess",
        lambda *_args, **_kwargs: result,
    )
    expected = "protocol failed" if failure == "stderr" else "exceeds its limit"
    with pytest.raises(RuntimeError, match=expected) as raised:
        runtime_v2._run_final_runtime_verifier(
            tmp_path / "runtime" / "bin" / "python",
            runtime_lock=tmp_path / "base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            closure_path=tmp_path / "closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
            environment={},
        )
    diagnostic = str(raised.value)
    assert "parent-stderr-secret" not in diagnostic
    assert "oversize-secret" not in diagnostic


@pytest.mark.parametrize("failure", ["nonzero", "timeout"])
def test_final_verifier_parent_process_failure_is_fixed_and_does_not_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    def run(*_args: Any, **_kwargs: Any) -> runtime_v2._BoundedBinarySubprocessResult:
        if failure == "timeout":
            return _bounded_process_result(
                stdout=b"parent-stdout-secret",
                stderr=b"parent-stderr-secret",
                timed_out=True,
            )
        return _bounded_process_result(
            returncode=23,
            stdout=b"parent-stdout-secret",
            stderr=b"parent-stderr-secret",
        )

    monkeypatch.setattr(runtime_v2, "_run_bounded_binary_subprocess", run)
    expected = "timed out" if failure == "timeout" else "process failed"
    with pytest.raises(RuntimeError, match=expected) as raised:
        runtime_v2._run_final_runtime_verifier(
            tmp_path / "runtime" / "bin" / "python",
            runtime_lock=tmp_path / "base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            closure_path=tmp_path / "closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
            environment={},
        )
    diagnostic = str(raised.value)
    assert "parent-command-secret" not in diagnostic
    assert "parent-stdout-secret" not in diagnostic
    assert "parent-stderr-secret" not in diagnostic
