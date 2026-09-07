import ast
import json
import os
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

import document_kv_cache.full_score_execution as full_score
import document_kv_cache.publication_bf16_handoff_generation as bf16_generation
import document_kv_cache.publication_latency_execution as latency_execution
import document_kv_cache.publication_latency_handoff_generation as q8_generation


_SYSTEM_CUDA_PARENT_ATTESTATION_ENV = (
    "CACHET_GPU_QUALIFICATION_SYSTEM_CUDA_PARENT_ATTESTATION"
)
_GPU_RUNNERS = (
    pytest.param(
        q8_generation.PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT,
        "_bootstrap",
        True,
        id="q8-handoff",
    ),
    pytest.param(
        bf16_generation.PUBLICATION_BF16_HANDOFF_RUNNER_SCRIPT,
        "_bootstrap",
        True,
        id="bf16-handoff",
    ),
    pytest.param(
        full_score.FULL_SCORE_RUNNER_SCRIPT,
        "_bootstrap",
        False,
        id="full-score",
    ),
)


def _compiled_runner_namespace(script: str, label: str) -> dict[str, object]:
    namespace: dict[str, object] = {"__name__": f"{label}_runner_test"}
    exec(compile(script, f"{label}_runner.py", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(("script", "entrypoint", "has_direct_verifier"), _GPU_RUNNERS)
def test_generated_gpu_runner_captures_and_revalidates_exact_parent_cuda(
    script: str,
    entrypoint: str,
    has_direct_verifier: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace = _compiled_runner_namespace(script, entrypoint)
    metadata = namespace["importlib"].metadata
    monkeypatch.setattr(metadata, "distributions", lambda: iter(()))
    with pytest.raises(RuntimeError, match="parent CUDA runtime distribution differs"):
        namespace["_capture_system_cuda_parent_attestation_json"]()

    payload = b"reviewed-generated-runner-libcudart"
    member = PurePosixPath("nvidia/cuda_runtime/lib/libcudart.so.12")
    distribution_root = tmp_path / "site-packages"
    member_path = distribution_root / member
    member_path.parent.mkdir(parents=True)
    member_path.write_bytes(payload)
    distribution = SimpleNamespace(
        metadata={"Name": "nvidia_cuda.runtime-cu12"},
        version="12.1.105",
        files=(member,),
        locate_file=lambda item: distribution_root / str(item),
    )
    monkeypatch.setattr(metadata, "distributions", lambda: iter((distribution,)))
    namespace["_SYSTEM_CUDA_PARENT_LIBCUDART_SIZE_BYTES"] = len(payload)
    namespace["_SYSTEM_CUDA_PARENT_LIBCUDART_SHA256"] = sha256(payload).hexdigest()

    monkeypatch.setenv(_SYSTEM_CUDA_PARENT_ATTESTATION_ENV, "hostile-ambient")
    install_environment = namespace["_pip_subprocess_environment"]()
    assert _SYSTEM_CUDA_PARENT_ATTESTATION_ENV not in install_environment

    raw = namespace["_capture_system_cuda_parent_attestation_json"]()
    record = json.loads(raw)
    assert raw == json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert record == {
        "distribution_name": "nvidia-cuda-runtime-cu12",
        "distribution_root": str(distribution_root),
        "distribution_version": "12.1.105",
        "libcudart_member": str(member),
        "libcudart_path": str(member_path),
        "libcudart_sha256": sha256(payload).hexdigest(),
        "libcudart_size_bytes": len(payload),
        "record_type": "cachet.gpu_qualification.system_cuda_parent_attestation.v1",
        "schema_version": 1,
    }

    monkeypatch.setenv(
        _SYSTEM_CUDA_PARENT_ATTESTATION_ENV,
        json.dumps(record, indent=2, sort_keys=True),
    )
    with pytest.raises(RuntimeError, match="attestation is not canonical"):
        namespace["_system_cuda_parent_attestation_json_from_environment"]()

    monkeypatch.setenv(_SYSTEM_CUDA_PARENT_ATTESTATION_ENV, raw)
    assert namespace["_system_cuda_parent_attestation_json_from_environment"]() == raw
    member_path.write_bytes(b"x" * len(payload))
    with pytest.raises(RuntimeError, match="member bytes differ"):
        namespace["_system_cuda_parent_attestation_json_from_environment"]()

    member_path.unlink()
    external_member = tmp_path / "external-libcudart.so.12"
    external_member.write_bytes(payload)
    member_path.symlink_to(external_member)
    with pytest.raises(RuntimeError, match="without following"):
        namespace["_system_cuda_parent_attestation_json_from_environment"]()

    assert entrypoint in namespace
    assert has_direct_verifier is any(
        verifier_name in script
        for verifier_name in (
            "_gpu_runtime_final_verifier_main",
            "verify_gpu_qualification_v2_runtime_installation",
        )
    )


@pytest.mark.parametrize(("script", "entrypoint", "has_direct_verifier"), _GPU_RUNNERS)
def test_generated_gpu_runner_threads_only_fresh_parent_cuda_after_install(
    script: str,
    entrypoint: str,
    has_direct_verifier: bool,
) -> None:
    entrypoint_body = script.split(f"def {entrypoint}", maxsplit=1)[1]
    assert (
        "venv_dir = _canonical_locked_runtime_venv_dir(args.runtime_venv_dir)"
        in entrypoint_body
    )
    capture = entrypoint_body.index("_capture_system_cuda_parent_attestation_json()")
    pip_environment = entrypoint_body.index("pip_environment =")
    create_venv = entrypoint_body.index("_create_runtime_venv(", pip_environment)
    assert capture < pip_environment < create_venv
    assert "env.pop(_SYSTEM_CUDA_PARENT_ATTESTATION_ENV, None)" in script
    assert (
        "env[_SYSTEM_CUDA_PARENT_ATTESTATION_ENV] = (\n"
        "        system_cuda_parent_attestation_json\n"
        "    )"
    ) in entrypoint_body
    inherited_validation = entrypoint_body.index(
        "_system_cuda_parent_attestation_json_from_environment()"
    )
    assert inherited_validation < capture
    inherited_torch_library_validation = entrypoint_body.index(
        "_require_locked_runtime_launch_environment("
    )
    assert inherited_torch_library_validation < capture
    launch_environment = entrypoint_body.index(
        "_locked_runtime_launch_environment(", capture
    )
    assert create_venv < launch_environment
    if has_direct_verifier:
        child_verifier = entrypoint_body.index("_verify_locked_runtime(")
        assert inherited_torch_library_validation < child_verifier < capture
        parent_verifier = entrypoint_body.index(
            "_verify_locked_runtime(", launch_environment
        )
        assert launch_environment < parent_verifier
        assert "env = dict(verifier_environment)" in entrypoint_body
        assert (
            entrypoint_body.count(
                "verifier_environment[_SYSTEM_CUDA_PARENT_ATTESTATION_ENV]"
            )
            == 2
        )
    else:
        child_import = entrypoint_body.index(
            "from document_kv_cache.full_score_execution import main"
        )
        assert inherited_torch_library_validation < child_import < capture
        assert "verifier_environment" not in entrypoint_body
        assert launch_environment < entrypoint_body.index("os.execve(")


@pytest.mark.parametrize(("script", "entrypoint", "has_direct_verifier"), _GPU_RUNNERS)
def test_generated_gpu_runner_binds_exact_venv_torch_library_directory(
    script: str,
    entrypoint: str,
    has_direct_verifier: bool,
    tmp_path: Path,
) -> None:
    del entrypoint, has_direct_verifier
    namespace = _compiled_runner_namespace(script, "torch_library")
    allowed_root = tmp_path / "local_disk0"
    allowed_root.mkdir()
    namespace["_LOCKED_RUNTIME_PARENT"] = allowed_root
    runtime_dir = allowed_root / "reviewed-runtime"
    torch_library_dir = runtime_dir / "lib/python3.11/site-packages/torch/lib"
    torch_library_dir.mkdir(parents=True)
    install_environment = {
        "LD_LIBRARY_PATH": "/ambient/reviewed-one:/ambient/reviewed-two",
        "SAFE": "1",
    }

    launch_environment = namespace["_locked_runtime_launch_environment"](
        venv_dir=str(runtime_dir),
        install_environment=install_environment,
    )

    assert launch_environment == {
        "LD_LIBRARY_PATH": (
            f"{torch_library_dir}{os.pathsep}"
            "/ambient/reviewed-one:/ambient/reviewed-two"
        ),
        "SAFE": "1",
    }
    assert install_environment == {
        "LD_LIBRARY_PATH": "/ambient/reviewed-one:/ambient/reviewed-two",
        "SAFE": "1",
    }
    assert (
        namespace["_require_locked_runtime_launch_environment"](
            venv_dir=str(runtime_dir),
            environment=launch_environment,
        )
        == launch_environment
    )
    with pytest.raises(
        RuntimeError, match="locked runtime torch library environment differs"
    ):
        namespace["_require_locked_runtime_launch_environment"](
            venv_dir=str(runtime_dir),
            environment={"LD_LIBRARY_PATH": "/ambient/only"},
        )

    absent_runtime = allowed_root / "absent-runtime"
    missing_runtime = allowed_root / "missing-torch-library"
    missing_runtime.mkdir()
    linked_runtime = allowed_root / "linked-runtime"
    linked_runtime.symlink_to(runtime_dir, target_is_directory=True)
    linked_torch_runtime = allowed_root / "linked-torch-runtime"
    linked_torch_library = (
        linked_torch_runtime / "lib/python3.11/site-packages/torch/lib"
    )
    linked_torch_library.parent.mkdir(parents=True)
    linked_target = tmp_path / "linked-torch-target"
    linked_target.mkdir()
    linked_torch_library.symlink_to(linked_target, target_is_directory=True)
    outside_runtime = tmp_path / "outside-runtime"
    (outside_runtime / "lib/python3.11/site-packages/torch/lib").mkdir(parents=True)
    noncanonical_runtime = f"{runtime_dir}/../{runtime_dir.name}"
    for rejected_runtime in (
        str(absent_runtime),
        str(missing_runtime),
        str(linked_runtime),
        str(linked_torch_runtime),
        str(outside_runtime),
        noncanonical_runtime,
    ):
        with pytest.raises(
            RuntimeError, match="locked runtime torch library directory differs"
        ):
            namespace["_locked_runtime_launch_environment"](
                venv_dir=rejected_runtime,
                install_environment={},
            )


@pytest.mark.parametrize(
    "script",
    (
        q8_generation.PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT,
        bf16_generation.PUBLICATION_BF16_HANDOFF_RUNNER_SCRIPT,
    ),
    ids=("q8-handoff", "bf16-handoff"),
)
def test_handoff_parent_uses_venv_for_independent_attestation_validation(
    script: str,
) -> None:
    tree = ast.parse(script)
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "document_kv_cache.gpu_qualification_v2"
        for node in ast.walk(tree)
    )
    assert "validate_gpu_qualification_v2_runtime_attestation as validate" in script
    assert (
        '[venv_python, "-c", validator, canonical_stdout.decode("utf-8")]'
        in script
    )
    assert "input=canonical_stdout" not in script
    assert 'completed.stdout != b"validated\\n"' in script
    assert "os._exit(0)" in script
    verifier = script.split("def _verify_locked_runtime", maxsplit=1)[1].split(
        "def _bootstrap", maxsplit=1
    )[0]
    assert verifier.count("environment=environment") == 2


def test_full_score_runtime_verifier_and_workers_preserve_bound_torch_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_library_path = (
        "/local_disk0/reviewed-runtime/"
        "lib/python3.11/site-packages/torch/lib:/ambient/reviewed"
    )
    monkeypatch.setattr(
        full_score.os,
        "environ",
        {"LD_LIBRARY_PATH": torch_library_path},
    )
    monkeypatch.setattr(
        full_score,
        "gpu_runtime_warning_environment_overrides",
        lambda: {},
    )
    runtime = SimpleNamespace(
        model_id="reviewed-model",
        model_revision="reviewed-revision",
        tokenizer_id="reviewed-tokenizer",
        tokenizer_revision="reviewed-tokenizer-revision",
    )

    environment = full_score._worker_environment(runtime)

    assert environment["LD_LIBRARY_PATH"] == torch_library_path


def test_ordinary_latency_runner_does_not_claim_cuda_parent_attestation() -> None:
    assert _SYSTEM_CUDA_PARENT_ATTESTATION_ENV not in (
        latency_execution.PUBLICATION_LATENCY_RUNNER_SCRIPT
    )
