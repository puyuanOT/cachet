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
    "poetry-core": "2.4.1",
    "packaging": "26.2",
    "pyspark": "4.1.2",
    "databricks-sdk": "0.118.0",
    "pytest": "9.1.1",
    "vllm": "0.23.0",
    "transformers": "5.12.1",
    "huggingface-hub": "1.20.1",
    "bitsandbytes": "0.49.2",
    "accelerate": "1.14.0",
    "tokenizers": "0.23.1",
    "numpy": "2.5.0",
    "fastapi": "0.138.0",
    "prometheus-fastapi-instrumentator": "8.0.2",
    "sglang": "0.5.13.post1",
    "protobuf": "7.35.1",
}
CURRENT_RUNTIME_PIN_ALLOWANCES = {
    "sglang": (
        "Pinned to the latest g6/L4 Databricks-validated Cachet SGLang "
        "HiCache provider profile; upgrading requires a fresh native/live "
        "Databricks validation."
    ),
    "tokenizers": (
        "Pinned to the latest g6/L4 Databricks-validated vLLM 0.23.0 "
        "serving profile; upgrade after a fresh vLLM Databricks run."
    ),
    "numpy": (
        "Pinned to the latest g6/L4 Databricks-validated vLLM 0.23.0 "
        "serving profile; upgrade after a fresh vLLM Databricks run."
    ),
    "fastapi": (
        "Pinned to the latest g6/L4 Databricks-validated vLLM 0.23.0 "
        "serving profile; upgrade after a fresh vLLM Databricks run."
    ),
    "prometheus-fastapi-instrumentator": (
        "Pinned to the latest g6/L4 Databricks-validated vLLM 0.23.0 "
        "serving profile; upgrade after a fresh vLLM Databricks run."
    ),
}
CURRENT_TRANSITIVE_ALLOWANCES = {
    "protobuf": (
        "databricks-sdk==0.118.0 currently resolves protobuf <7.0, so "
        "Poetry keeps protobuf 6.33.6 even though PyPI has 7.35.1."
    ),
}


def test_dependency_freshness_accepts_current_repo_direct_pins_with_explicit_runtime_holds():
    evidence = evaluate_dependency_freshness(
        pyproject_path=REPO_ROOT / "pyproject.toml",
        latest_versions=CURRENT_LATEST_VERSIONS,
        allowed_runtime_pins=CURRENT_RUNTIME_PIN_ALLOWANCES,
        transitive_outdated={"protobuf": ("6.33.6", "7.35.1")},
        allowed_transitive_outdated=CURRENT_TRANSITIVE_ALLOWANCES,
    )
    record = dependency_freshness_to_record(evidence)

    assert evidence.ok
    assert record["record_type"] == DEPENDENCY_FRESHNESS_RECORD_TYPE
    assert {pin["package"] for pin in record["direct_pins"]} == {
        "poetry-core",
        "packaging",
        "pyspark",
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
    assert record["transitive_outdated"] == [
        {
            "package": "protobuf",
            "locked_version": "6.33.6",
            "latest_version": "7.35.1",
            "allowed": True,
            "reason": CURRENT_TRANSITIVE_ALLOWANCES["protobuf"],
        }
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
        transitive_outdated={"protobuf": ("6.33.6", "7.35.1")},
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
