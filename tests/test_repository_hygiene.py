from pathlib import Path

from document_kv_cache.repository_hygiene import (
    FORBIDDEN_TRACKED_ARTIFACT_PATTERNS,
    REQUIRED_GITIGNORE_PATTERNS,
    evaluate_repository_hygiene,
    evaluate_repository_hygiene_paths,
    repository_hygiene_to_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_covers_local_build_cache_and_secret_artifacts():
    ignored_lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert set(REQUIRED_GITIGNORE_PATTERNS).issubset(ignored_lines)


def test_no_generated_artifacts_are_tracked():
    evidence = evaluate_repository_hygiene(REPO_ROOT)
    violations = list(evidence.forbidden_tracked_paths)

    assert violations == []


def test_repository_hygiene_record_is_release_ready_for_current_repo():
    evidence = evaluate_repository_hygiene(REPO_ROOT)
    record = repository_hygiene_to_record(evidence)

    assert record["ok"] is True
    assert record["record_type"] == "document_kv.repository_hygiene.v1"
    assert record["missing_gitignore_patterns"] == []
    assert record["forbidden_tracked_paths"] == []
    assert record["tracked_path_count"] > 0
    assert set(record["required_gitignore_patterns"]) == set(REQUIRED_GITIGNORE_PATTERNS)
    assert set(record["forbidden_tracked_artifact_patterns"]) == set(FORBIDDEN_TRACKED_ARTIFACT_PATTERNS)
    assert record["issues"] == []


def test_repository_hygiene_reports_missing_gitignore_and_forbidden_tracked_artifacts(tmp_path):
    evidence = evaluate_repository_hygiene_paths(
        repository_root=tmp_path,
        tracked_paths=(
            ".env.example",
            "dist/document_kv_cache-0.2.0-py3-none-any.whl",
            "src/document_kv_cache/__pycache__/cache.pyc",
        ),
        gitignore_lines=(".venv/", "__pycache__/"),
    )
    record = repository_hygiene_to_record(evidence)

    assert record["ok"] is False
    assert ".env" in record["missing_gitignore_patterns"]
    assert ".env.example" not in record["forbidden_tracked_paths"]
    assert record["forbidden_tracked_paths"] == [
        "dist/document_kv_cache-0.2.0-py3-none-any.whl",
        "src/document_kv_cache/__pycache__/cache.pyc",
    ]
    assert any(issue.startswith("missing required .gitignore patterns:") for issue in record["issues"])
    assert any(issue.startswith("forbidden generated or secret-like tracked artifacts:") for issue in record["issues"])
