import json
from pathlib import Path

import cachet.dependency_freshness as cachet_dependency_freshness
import document_kv_cache.dependency_freshness as public_dependency_freshness
from document_kv_cache.dependency_freshness import (
    DEPENDENCY_FRESHNESS_RECORD_TYPE,
    dependency_freshness_record_issues,
    dependency_freshness_to_record,
    evaluate_dependency_freshness,
    pyproject_direct_dependency_pins,
    serving_profile_runtime_pins,
    write_dependency_freshness_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / "release-ops"
    / "evidence"
    / "dependency-freshness"
    / "current"
    / "dependency-freshness-evidence.json"
)
CURRENT_LATEST_VERSIONS = {
    "poetry-core": "2.5.0",
    "packaging": "26.3",
    "databricks-sdk": "0.170.0",
    "pytest": "9.2.0",
    "vllm": "0.34.0",
    "torch": "2.13.0+cu129",
    "torchaudio": "2.11.0+cu129",
    "torchvision": "0.28.0+cu129",
    "triton": "3.9.0",
    "numba": "0.68.1",
    "torchcodec": "0.16.0+cu129",
    "pynvvideocodec": "2.2.4",
    "flashinfer-python": "0.7.5.post1",
    "flashinfer-cubin": "0.7.5.post1",
    "flashinfer-jit-cache": "0.7.5.post1+cu129",
    "apache-tvm-ffi": "0.1.16",
    "tilelang": "0.1.15",
    "nvidia-cudnn-frontend": "1.33.0",
    "nvtx": "0.2.17",
    "fastsafetensors": "0.5.0",
    "instanttensor": "0.2.3",
    "nvidia-cutlass-dsl": "4.8.1",
    "quack-kernels": "0.6.5",
    "tokenspeed-mla": "2.0.12",
    "humming-kernels": "0.1.20",
    "transformers": "5.24.0",
    "huggingface-hub": "2.4.0",
    "bitsandbytes": "0.50.2",
    "accelerate": "1.16.0",
    "opencv-python-headless": "5.1.0.95",
    "tokenizers": "0.23.3",
    "numpy": "2.5.5",
    "fastapi": "0.165.0",
    "prometheus-fastapi-instrumentator": "8.1.0",
    "sglang": "0.5.23",
}
VLLM_CLOSURE_HOLD_REASON = (
    "Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 "
    "hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G "
    "Databricks validation."
)
CURRENT_RUNTIME_PIN_ALLOWANCES = {
    "vllm": (
        "Held at the reviewed vLLM 0.30.0+cu129 serving profile: the E5M2-patched "
        "wheel, packaged hash lock, and sealed L4/A10G Databricks benchmark "
        "evidence target 0.30.0, and vLLM 0.34.0 publishes no CUDA 12.9 wheel; "
        "upgrading requires a new runtime port and fresh L4 and A10G Databricks "
        "validation."
    ),
    "sglang": (
        "Pinned to the latest g6/L4 Databricks-validated Cachet SGLang "
        "HiCache provider profile; upgrading requires a fresh native/live "
        "Databricks validation."
    ),
    **{
        package: VLLM_CLOSURE_HOLD_REASON
        for package in (
            "triton",
            "numba",
            "pynvvideocodec",
            "flashinfer-python",
            "flashinfer-cubin",
            "flashinfer-jit-cache",
            "apache-tvm-ffi",
            "tilelang",
            "nvidia-cudnn-frontend",
            "nvtx",
            "fastsafetensors",
            "instanttensor",
            "nvidia-cutlass-dsl",
            "tokenspeed-mla",
            "humming-kernels",
            "transformers",
            "huggingface-hub",
            "bitsandbytes",
            "accelerate",
            "opencv-python-headless",
            "tokenizers",
            "numpy",
            "fastapi",
            "prometheus-fastapi-instrumentator",
        )
    },
}
CURRENT_TRANSITIVE_OUTDATED = {
    "mypy": ("2.2.0", "2.4.1"),
    "ruff": ("0.15.21", "0.17.6"),
}
CURRENT_TRANSITIVE_ALLOWANCES = {
    "mypy": (
        "mypy==2.2.0 is an exact Poetry dev-group requirement, so the resolver "
        "keeps 2.2.0 even though PyPI has 2.4.1; upgrading adds type findings "
        "on the CI-checked files and is a separate lint-tooling change."
    ),
    "ruff": (
        "ruff==0.15.21 is an exact Poetry dev-group requirement, so the resolver "
        "keeps 0.15.21 even though PyPI has 0.17.6; upgrading adds lint findings "
        "on the CI-checked files and is a separate lint-tooling change."
    ),
}


def test_dependency_freshness_accepts_current_repo_direct_pins_with_explicit_runtime_holds():
    evidence = evaluate_dependency_freshness(
        pyproject_path=REPO_ROOT / "pyproject.toml",
        latest_versions=CURRENT_LATEST_VERSIONS,
        allowed_runtime_pins=CURRENT_RUNTIME_PIN_ALLOWANCES,
        transitive_outdated=CURRENT_TRANSITIVE_OUTDATED,
        allowed_transitive_outdated=CURRENT_TRANSITIVE_ALLOWANCES,
    )
    record = dependency_freshness_to_record(evidence)

    assert evidence.ok
    assert record["record_type"] == DEPENDENCY_FRESHNESS_RECORD_TYPE
    assert {pin["package"] for pin in record["direct_pins"]} == {
        "poetry-core",
        "packaging",
        "databricks-sdk",
        "pytest",
    }
    assert all(pin["current"] for pin in record["direct_pins"])
    assert any(
        pin["package"] == "sglang"
        and pin["current"] is False
        and pin["allowed"] is True
        and "Databricks-validated" in pin["allow_reason"]
        for pin in record["runtime_pins"]
    )
    assert any(
        pin["package"] == "vllm"
        and pin["pinned_version"] == "0.30.0+cu129"
        and pin["current"] is False
        and pin["allowed"] is True
        for pin in record["runtime_pins"]
    )
    assert record["transitive_outdated"] == [
        {
            "package": package,
            "locked_version": locked,
            "latest_version": latest,
            "allowed": True,
            "reason": CURRENT_TRANSITIVE_ALLOWANCES[package],
        }
        for package, (locked, latest) in sorted(CURRENT_TRANSITIVE_OUTDATED.items())
    ]
    assert record["issues"] == []
    assert dependency_freshness_record_issues(record) == ()


def test_dependency_freshness_rejects_stale_direct_pins_and_unallowed_runtime_pins(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[build-system]
requires = ["poetry-core==2.4.0"]
build-backend = "poetry.core.masonry.api"

[project]
dependencies = ["packaging>=26"]
""",
        encoding="utf-8",
    )

    pins, parse_issues = pyproject_direct_dependency_pins(pyproject)
    evidence = evaluate_dependency_freshness(
        pyproject_path=pyproject,
        latest_versions={"poetry-core": "2.4.1", "sglang": "0.5.13.post1"},
        runtime_pins=(("test-runtime", "sglang==0.5.10.post1"),),
    )

    assert pins == (("poetry-core", "2.4.0", "build-system.requires"),)
    assert parse_issues == ("project.dependencies: 'packaging>=26' must be an exact == pin",)
    assert not evidence.ok
    assert "direct dependency poetry-core pinned to 2.4.0, latest is 2.4.1" in evidence.issues
    assert (
        "runtime dependency sglang pinned to 0.5.10.post1, latest is 0.5.13.post1, "
        "and no valid allow reason was provided"
    ) in evidence.issues


def test_dependency_freshness_rejects_weak_allow_reasons(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[build-system]
requires = ["poetry-core==2.4.1"]
build-backend = "poetry.core.masonry.api"
""",
        encoding="utf-8",
    )

    evidence = evaluate_dependency_freshness(
        pyproject_path=pyproject,
        latest_versions={
            "poetry-core": "2.4.1",
            "sglang": "0.5.13.post1",
            "protobuf": "7.35.1",
        },
        runtime_pins=(("test-runtime", "sglang==0.5.10.post1"),),
        allowed_runtime_pins={"sglang": "temporary hold"},
        transitive_outdated={"protobuf": ("6.33.6", "7.35.1")},
        allowed_transitive_outdated={"protobuf": "temporary hold"},
    )
    record = dependency_freshness_to_record(evidence)

    assert not evidence.ok
    assert (
        "runtime dependency sglang pinned to 0.5.10.post1, latest is 0.5.13.post1, "
        "and no valid allow reason was provided"
    ) in evidence.issues
    assert (
        "transitive dependency protobuf locked to 6.33.6, latest is 7.35.1, "
        "and no valid allow reason was provided"
    ) in evidence.issues
    assert "runtime_pins[0].allow_reason" in " ".join(dependency_freshness_record_issues(record))
    assert "transitive_outdated[0].reason" in " ".join(dependency_freshness_record_issues(record))


def test_committed_dependency_freshness_evidence_matches_generator():
    committed_record = json.loads(CURRENT_EVIDENCE_PATH.read_text(encoding="utf-8"))
    evidence = evaluate_dependency_freshness(
        pyproject_path=REPO_ROOT / "pyproject.toml",
        latest_versions=CURRENT_LATEST_VERSIONS,
        allowed_runtime_pins=CURRENT_RUNTIME_PIN_ALLOWANCES,
        transitive_outdated=CURRENT_TRANSITIVE_OUTDATED,
        allowed_transitive_outdated=CURRENT_TRANSITIVE_ALLOWANCES,
    )
    generated_record = dependency_freshness_to_record(evidence)
    generated_record["pyproject_path"] = "pyproject.toml"

    assert committed_record == generated_record
    assert dependency_freshness_record_issues(committed_record) == ()


def test_dependency_freshness_cli_and_writer_emit_records(tmp_path, capsys):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[build-system]
requires = ["poetry-core==2.4.1"]
build-backend = "poetry.core.masonry.api"
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "dependency-freshness.json"
    evidence = evaluate_dependency_freshness(
        pyproject_path=pyproject,
        latest_versions={"poetry-core": "2.4.1"},
        runtime_pins=(),
    )

    write_dependency_freshness_json(evidence, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == dependency_freshness_to_record(evidence)

    assert public_dependency_freshness.main(
        [
            "--pyproject",
            str(pyproject),
            "--latest-version",
            "poetry-core=2.4.1",
            "--no-serving-profiles",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_dependency_freshness_cachet_facade_aliases_document_module():
    assert cachet_dependency_freshness is public_dependency_freshness
    assert cachet_dependency_freshness.evaluate_dependency_freshness is evaluate_dependency_freshness
    assert serving_profile_runtime_pins()[0][0].startswith("serving_env.")
