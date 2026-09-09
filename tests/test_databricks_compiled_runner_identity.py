from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from document_kv_cache import databricks_engine_probe_job
from document_kv_cache import full_score_execution
from document_kv_cache import full_score_remote_control
from document_kv_cache import publication_bf16_handoff_generation
from document_kv_cache import publication_handoff_closure_coordinator
from document_kv_cache import publication_latency_execution
from document_kv_cache import publication_latency_handoff_generation


class _ReachedNextBootstrapStep(RuntimeError):
    pass


def _compiled_namespace(
    tmp_path: Path, name: str, script: str
) -> tuple[Path, dict[str, Any]]:
    path = tmp_path / name
    path.write_text(script, encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": f"{name}_test"}
    exec(compile(script, str(path), "exec"), namespace)
    assert "__file__" not in namespace
    return path, namespace


def _handoff_bootstrap_argv(runner_sha256: str) -> list[str]:
    digest = "1" * 64
    return [
        "--runner-sha256",
        runner_sha256,
        "--package-wheel-uri",
        "dbfs:/Volumes/catalog/schema/volume/cachet.whl",
        "--package-wheel-sha256",
        digest,
        "--runtime-lock-uri",
        "dbfs:/Volumes/catalog/schema/volume/runtime.lock",
        "--runtime-lock-sha256",
        digest,
        "--patched-vllm-wheel-uri",
        "dbfs:/Volumes/catalog/schema/volume/vllm.whl",
        "--patched-vllm-wheel-sha256",
        digest,
        "--patched-flashinfer-wheel-uri",
        "dbfs:/Volumes/catalog/schema/volume/flashinfer.whl",
        "--patched-flashinfer-wheel-sha256",
        digest,
        "--runtime-closure-manifest-uri",
        "dbfs:/Volumes/catalog/schema/volume/runtime.json",
        "--runtime-closure-manifest-sha256",
        digest,
        "--runtime-venv-dir",
        "/local_disk0/cachet-test-runtime",
    ]


@pytest.mark.parametrize(
    ("name", "script"),
    (
        (
            "q8_handoff.py",
            publication_latency_handoff_generation.PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT,
        ),
        (
            "bf16_handoff.py",
            publication_bf16_handoff_generation.PUBLICATION_BF16_HANDOFF_RUNNER_SCRIPT,
        ),
        (
            "handoff_closure.py",
            publication_handoff_closure_coordinator.PUBLICATION_HANDOFF_CLOSURE_RUNNER_SCRIPT,
        ),
    ),
)
def test_handoff_bootstraps_hash_the_compiled_python_file_without_file_global(
    tmp_path: Path,
    name: str,
    script: str,
) -> None:
    path, namespace = _compiled_namespace(tmp_path, name, script)
    runner_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def reached_artifact_validation(*_args: object, **_kwargs: object) -> str:
        raise _ReachedNextBootstrapStep

    namespace["_verified_path"] = reached_artifact_validation
    with pytest.raises(_ReachedNextBootstrapStep):
        namespace["_bootstrap"](_handoff_bootstrap_argv(runner_sha256))

    assert os.path.realpath(namespace["_bootstrap"].__code__.co_filename) == str(
        path.resolve()
    )


@pytest.mark.parametrize(
    ("name", "script"),
    (
        (
            "q8_handoff.py",
            publication_latency_handoff_generation.PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT,
        ),
        (
            "bf16_handoff.py",
            publication_bf16_handoff_generation.PUBLICATION_BF16_HANDOFF_RUNNER_SCRIPT,
        ),
        (
            "handoff_closure.py",
            publication_handoff_closure_coordinator.PUBLICATION_HANDOFF_CLOSURE_RUNNER_SCRIPT,
        ),
    ),
)
def test_handoff_bootstraps_reject_tampered_compiled_python_file(
    tmp_path: Path,
    name: str,
    script: str,
) -> None:
    path, namespace = _compiled_namespace(tmp_path, name, script)
    runner_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(script + "\n# tampered after compilation\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runner SHA-256 does not match"):
        namespace["_bootstrap"](_handoff_bootstrap_argv(runner_sha256))


def test_latency_worker_hashes_the_compiled_python_file_without_file_global(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, namespace = _compiled_namespace(
        tmp_path,
        "latency_worker.py",
        publication_latency_execution.PUBLICATION_LATENCY_RUNNER_SCRIPT,
    )
    calls: list[tuple[str, str]] = []

    def verified(candidate: str, _digest: str, label: str) -> str:
        calls.append((candidate, label))
        if label == "Cachet package wheel":
            raise _ReachedNextBootstrapStep
        return candidate

    namespace["_verified"] = verified
    monkeypatch.setattr(
        namespace["sys"],
        "argv",
        [
            str(path),
            "--job-record-json",
            "{}",
            "--expected-job-sha256",
            "0" * 64,
            "--runner-uri",
            "dbfs:/Volumes/catalog/schema/volume/latency.py",
            "--runner-sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "--package-wheel-uri",
            "dbfs:/Volumes/catalog/schema/volume/cachet.whl",
            "--package-wheel-sha256",
            "1" * 64,
            "--cloud-run-id",
            "1",
            "--task-run-id",
            "2",
        ],
    )
    with pytest.raises(_ReachedNextBootstrapStep):
        namespace["main"]()

    assert calls[1] == (
        str(path.resolve()),
        "executing publication latency runner",
    )


def test_latency_source_closure_hashes_compiled_file_without_file_global(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, namespace = _compiled_namespace(
        tmp_path,
        "latency_source_closure.py",
        publication_latency_execution.PUBLICATION_LATENCY_SOURCE_CLOSURE_RUNNER_SCRIPT,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def reached_artifact_validation(*_args: object, **_kwargs: object) -> str:
        raise _ReachedNextBootstrapStep

    namespace["_verified"] = reached_artifact_validation
    monkeypatch.setattr(
        namespace["sys"],
        "argv",
        [
            str(path),
            "--runner-sha256",
            digest,
            "--package-wheel-uri",
            "dbfs:/Volumes/catalog/schema/volume/cachet.whl",
            "--package-wheel-sha256",
            "1" * 64,
            "--runtime-lock-uri",
            "dbfs:/Volumes/catalog/schema/volume/runtime.lock",
            "--runtime-lock-sha256",
            "1" * 64,
            "--patched-vllm-wheel-uri",
            "dbfs:/Volumes/catalog/schema/volume/vllm.whl",
            "--patched-vllm-wheel-sha256",
            "1" * 64,
            "--patched-flashinfer-wheel-uri",
            "dbfs:/Volumes/catalog/schema/volume/flashinfer.whl",
            "--patched-flashinfer-wheel-sha256",
            "1" * 64,
            "--runtime-closure-manifest-uri",
            "dbfs:/Volumes/catalog/schema/volume/runtime.json",
            "--runtime-closure-manifest-sha256",
            "1" * 64,
            "--runtime-venv-dir",
            "/local_disk0/cachet-test-runtime",
            "--request-uri",
            "dbfs:/Volumes/catalog/schema/volume/request.json",
            "--request-file-sha256",
            "1" * 64,
            "--request-closed-record-sha256",
            "1" * 64,
            "--coordinator-run-id",
            "1",
        ],
    )
    with pytest.raises(_ReachedNextBootstrapStep):
        namespace["main"]()


def test_full_score_worker_hashes_compiled_file_without_file_global(
    tmp_path: Path,
) -> None:
    path, namespace = _compiled_namespace(
        tmp_path,
        "full_score_worker.py",
        full_score_execution.FULL_SCORE_RUNNER_SCRIPT,
    )
    verified_paths: list[str] = []

    def verified(candidate: str, _digest: str, _label: str) -> str:
        verified_paths.append(candidate)
        if len(verified_paths) > 1:
            raise _ReachedNextBootstrapStep
        return candidate

    namespace["_verified_path"] = verified
    with pytest.raises(_ReachedNextBootstrapStep):
        namespace["_bootstrap"](_handoff_bootstrap_argv("1" * 64))

    assert verified_paths[0] == str(path.resolve())


def test_full_score_remote_hashes_compiled_file_without_file_global(
    tmp_path: Path,
) -> None:
    path, namespace = _compiled_namespace(
        tmp_path,
        "full_score_remote.py",
        full_score_remote_control.FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SCRIPT,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    hashed_paths: list[str] = []
    real_sha256 = namespace["_sha256"]

    def sha256_then_stop(candidate: str) -> str:
        hashed_paths.append(candidate)
        if len(hashed_paths) > 1:
            raise _ReachedNextBootstrapStep
        return real_sha256(candidate)

    namespace["_sha256"] = sha256_then_stop
    with pytest.raises(_ReachedNextBootstrapStep):
        namespace["_main"](
            [
                "--runner-sha256",
                digest,
                "--package-wheel-uri",
                "dbfs:/Volumes/catalog/schema/volume/cachet.whl",
                "--package-wheel-sha256",
                "1" * 64,
                "--request-json",
                "dbfs:/Volumes/catalog/schema/volume/request.json",
                "--expected-request-file-sha256",
                "1" * 64,
                "--expected-request-record-sha256",
                "1" * 64,
            ]
        )

    assert hashed_paths[0] == str(path.resolve())


def test_engine_probe_reexecs_compiled_file_without_file_global(tmp_path: Path) -> None:
    path, namespace = _compiled_namespace(
        tmp_path,
        "engine_probe.py",
        databricks_engine_probe_job.ENGINE_PROBE_RUNNER_SCRIPT,
    )
    assert namespace["_runner_script_path"]() == str(path.resolve())
