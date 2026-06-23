import subprocess
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
    assert ".ipynb_checkpoints/" in ignored_lines
    assert ".databricks/" in ignored_lines
    assert "databricks-runs/" in ignored_lines
    assert ".terraform/" in ignored_lines
    assert "terraform.tfstate" in ignored_lines


def test_no_generated_artifacts_are_tracked():
    evidence = evaluate_repository_hygiene(REPO_ROOT)
    violations = list(evidence.forbidden_tracked_paths)

    assert violations == []


def test_repository_hygiene_record_uses_current_repository_policy():
    evidence = evaluate_repository_hygiene(REPO_ROOT)
    record = repository_hygiene_to_record(evidence)

    assert record["record_type"] == "document_kv.repository_hygiene.v1"
    assert record["missing_gitignore_patterns"] == []
    assert record["forbidden_tracked_paths"] == []
    assert record["forbidden_untracked_paths"] == []
    assert record["missing_directory_documentation_paths"] == []
    assert "." in record["documentation_checked_directory_paths"]
    assert "src/document_kv_cache" in record["documentation_checked_directory_paths"]
    assert "tests" in record["documentation_checked_directory_paths"]
    assert record["tracked_path_count"] > 0
    assert record["untracked_path_count"] >= 0
    assert set(record["required_gitignore_patterns"]) == set(REQUIRED_GITIGNORE_PATTERNS)
    assert set(record["forbidden_tracked_artifact_patterns"]) == set(FORBIDDEN_TRACKED_ARTIFACT_PATTERNS)


def test_repository_hygiene_reports_dirty_tracked_paths_from_git(tmp_path):
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text("\n".join(REQUIRED_GITIGNORE_PATTERNS) + "\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", ".gitignore", "README.md")
    _git(tmp_path, "-c", "user.email=cachet@example.com", "-c", "user.name=Cachet CI", "commit", "-m", "init")

    (tmp_path / "README.md").write_text("dirty\n", encoding="utf-8")

    record = repository_hygiene_to_record(evaluate_repository_hygiene(tmp_path))

    assert record["ok"] is False
    assert record["dirty_tracked_paths"] == ["README.md"]
    assert record["forbidden_tracked_paths"] == []
    assert record["forbidden_untracked_paths"] == []
    assert any(issue == "dirty tracked paths: README.md" for issue in record["issues"])


def test_repository_hygiene_reports_untracked_forbidden_artifacts_from_git(tmp_path):
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text(
        "\n".join(pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern != "*.tmp") + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", ".gitignore", "README.md")
    _git(tmp_path, "-c", "user.email=cachet@example.com", "-c", "user.name=Cachet CI", "commit", "-m", "init")

    (tmp_path / "scratch.tmp").write_text("local artifact\n", encoding="utf-8")

    record = repository_hygiene_to_record(evaluate_repository_hygiene(tmp_path))

    assert record["ok"] is False
    assert record["untracked_path_count"] == 1
    assert record["forbidden_tracked_paths"] == []
    assert record["forbidden_untracked_paths"] == ["scratch.tmp"]
    assert any(issue == "forbidden generated or secret-like untracked artifacts: scratch.tmp" for issue in record["issues"])


def test_repository_hygiene_reports_missing_gitignore_and_forbidden_tracked_artifacts(tmp_path):
    (tmp_path / "README.md").write_text("documented root\n", encoding="utf-8")
    evidence = evaluate_repository_hygiene_paths(
        repository_root=tmp_path,
        tracked_paths=(
            ".env.example",
            "dist/cachet_kv-0.2.0-py3-none-any.whl",
            ".databricks/bundle/dev/state.json",
            ".terraform/providers/registry.terraform.io/hashicorp/databricks/provider",
            "databricks-runs/cachet_probe/run-status.json",
            "terraform.tfstate.backup",
            "notebooks/.ipynb_checkpoints/debug-checkpoint.ipynb",
            "src/document_kv_cache/__pycache__/cache.pyc",
        ),
        untracked_paths=(
            ".databricks/bundle/dev/state.json",
            ".terraform/providers/provider.bin",
            "databricks-runs/cachet_probe/submit-response.json",
            ".env.local",
            "notes.txt",
            "tmp/output.tmp",
            "terraform.tfstate",
        ),
        dirty_tracked_paths=("README.md", "src/document_kv_cache/repository_hygiene.py"),
        gitignore_lines=(".venv/", "__pycache__/"),
    )
    record = repository_hygiene_to_record(evidence)

    assert record["ok"] is False
    assert ".env" in record["missing_gitignore_patterns"]
    assert ".env.example" not in record["forbidden_tracked_paths"]
    assert record["documentation_checked_directory_paths"] == ["."]
    assert record["missing_directory_documentation_paths"] == []
    assert record["forbidden_tracked_paths"] == [
        ".databricks/bundle/dev/state.json",
        ".terraform/providers/registry.terraform.io/hashicorp/databricks/provider",
        "databricks-runs/cachet_probe/run-status.json",
        "dist/cachet_kv-0.2.0-py3-none-any.whl",
        "notebooks/.ipynb_checkpoints/debug-checkpoint.ipynb",
        "src/document_kv_cache/__pycache__/cache.pyc",
        "terraform.tfstate.backup",
    ]
    assert record["forbidden_untracked_paths"] == [
        ".databricks/bundle/dev/state.json",
        ".env.local",
        ".terraform/providers/provider.bin",
        "databricks-runs/cachet_probe/submit-response.json",
        "terraform.tfstate",
        "tmp/output.tmp",
    ]
    assert record["dirty_tracked_paths"] == [
        "README.md",
        "src/document_kv_cache/repository_hygiene.py",
    ]
    assert record["untracked_path_count"] == 7
    assert any(issue.startswith("missing required .gitignore patterns:") for issue in record["issues"])
    assert any(issue.startswith("forbidden generated or secret-like tracked artifacts:") for issue in record["issues"])
    assert any(issue.startswith("forbidden generated or secret-like untracked artifacts:") for issue in record["issues"])
    assert any(issue.startswith("dirty tracked paths:") for issue in record["issues"])


def test_repository_hygiene_reports_missing_directory_documentation(tmp_path):
    (tmp_path / "README.md").write_text("documented root\n", encoding="utf-8")
    (tmp_path / "documented").mkdir()
    (tmp_path / "documented" / "__init__.py").write_text('"""Documented package."""\n', encoding="utf-8")
    (tmp_path / "undocumented").mkdir()

    evidence = evaluate_repository_hygiene_paths(
        repository_root=tmp_path,
        tracked_paths=(
            "documented/__init__.py",
            "undocumented/source.py",
        ),
        gitignore_lines=REQUIRED_GITIGNORE_PATTERNS,
    )
    record = repository_hygiene_to_record(evidence)

    assert record["ok"] is False
    assert record["documentation_checked_directory_paths"] == [".", "documented", "undocumented"]
    assert record["missing_directory_documentation_paths"] == ["undocumented"]
    assert any(
        issue == "directories missing README.md or package docstring: undocumented"
        for issue in record["issues"]
    )


def test_repository_hygiene_skips_tooling_output_directories_for_documentation(tmp_path):
    (tmp_path / "README.md").write_text("documented root\n", encoding="utf-8")
    for relative_path in (
        ".tox/py/state.txt",
        ".nox/session/log.txt",
        "coverage/index.html",
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated\n", encoding="utf-8")

    evidence = evaluate_repository_hygiene_paths(
        repository_root=tmp_path,
        tracked_paths=(),
        untracked_paths=(
            ".tox/py/state.txt",
            ".nox/session/log.txt",
            "coverage/index.html",
        ),
        gitignore_lines=REQUIRED_GITIGNORE_PATTERNS,
    )
    record = repository_hygiene_to_record(evidence)

    assert record["ok"] is True
    assert record["documentation_checked_directory_paths"] == ["."]
    assert record["missing_directory_documentation_paths"] == []
    assert record["untracked_path_count"] == 3


def test_repository_hygiene_rejects_checkpoint_artifacts_without_requiring_directory_docs(tmp_path):
    (tmp_path / "README.md").write_text("documented root\n", encoding="utf-8")
    checkpoint = "notebooks/.ipynb_checkpoints/exploration-checkpoint.ipynb"
    path = tmp_path / checkpoint
    path.parent.mkdir(parents=True)
    path.write_text("generated\n", encoding="utf-8")

    evidence = evaluate_repository_hygiene_paths(
        repository_root=tmp_path,
        tracked_paths=(),
        untracked_paths=(checkpoint,),
        gitignore_lines=REQUIRED_GITIGNORE_PATTERNS,
    )
    record = repository_hygiene_to_record(evidence)

    assert record["ok"] is False
    assert record["forbidden_untracked_paths"] == [checkpoint]
    assert record["documentation_checked_directory_paths"] == ["."]
    assert record["missing_directory_documentation_paths"] == []


def test_repository_hygiene_rejects_databricks_runs_without_requiring_directory_docs(tmp_path):
    (tmp_path / "README.md").write_text("documented root\n", encoding="utf-8")
    run_output = "databricks-runs/cachet_probe/run-status.json"
    path = tmp_path / run_output
    path.parent.mkdir(parents=True)
    path.write_text('{"state": "TERMINATED"}\n', encoding="utf-8")

    evidence = evaluate_repository_hygiene_paths(
        repository_root=tmp_path,
        tracked_paths=(run_output,),
        untracked_paths=(run_output,),
        gitignore_lines=REQUIRED_GITIGNORE_PATTERNS,
    )
    record = repository_hygiene_to_record(evidence)

    assert record["ok"] is False
    assert record["forbidden_tracked_paths"] == [run_output]
    assert record["forbidden_untracked_paths"] == [run_output]
    assert record["documentation_checked_directory_paths"] == ["."]
    assert record["missing_directory_documentation_paths"] == []


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)
