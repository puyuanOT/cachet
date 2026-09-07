import ast
import json
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
    assert has_direct_verifier is (
        "verify_gpu_qualification_v2_runtime_installation" in script
    )


@pytest.mark.parametrize(("script", "entrypoint", "has_direct_verifier"), _GPU_RUNNERS)
def test_generated_gpu_runner_threads_only_fresh_parent_cuda_after_install(
    script: str,
    entrypoint: str,
    has_direct_verifier: bool,
) -> None:
    entrypoint_body = script.split(f"def {entrypoint}", maxsplit=1)[1]
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
    if has_direct_verifier:
        assert (
            entrypoint_body.count(
                "verifier_environment[_SYSTEM_CUDA_PARENT_ATTESTATION_ENV]"
            )
            == 2
        )
    else:
        assert "verifier_environment" not in entrypoint_body


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
    assert '[venv_python, "-c", validator]' in script
    assert "input=canonical_stdout" in script
    assert 'validated.stdout != "validated\\n"' in script


def test_ordinary_latency_runner_does_not_claim_cuda_parent_attestation() -> None:
    assert _SYSTEM_CUDA_PARENT_ATTESTATION_ENV not in (
        latency_execution.PUBLICATION_LATENCY_RUNNER_SCRIPT
    )
