"""Repository cleanliness evidence for Document KV Cache releases."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_kv_cache.storage import local_path


REPOSITORY_HYGIENE_RECORD_TYPE = "document_kv.repository_hygiene.v1"
REQUIRED_GITIGNORE_PATTERNS = (
    ".venv/",
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".hypothesis/",
    ".coverage",
    "htmlcov/",
    "dist/",
    "build/",
    "*.egg-info/",
    ".DS_Store",
    ".env",
    ".env.*",
    "!.env.example",
    ".envrc",
    "*.pem",
    "*.key",
    "*.secret",
    "*.secrets",
    "*.log",
    "*.tmp",
)
FORBIDDEN_TRACKED_ARTIFACT_PATTERNS = (
    ".venv/*",
    "*/.venv/*",
    "__pycache__/*",
    "*/__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".pytest_cache/*",
    "*/.pytest_cache/*",
    ".mypy_cache/*",
    "*/.mypy_cache/*",
    ".ruff_cache/*",
    "*/.ruff_cache/*",
    ".hypothesis/*",
    "*/.hypothesis/*",
    ".coverage",
    "*/.coverage",
    "htmlcov/*",
    "*/htmlcov/*",
    "dist/*",
    "*/dist/*",
    "build/*",
    "*/build/*",
    "*.whl",
    "*.tar.gz",
    "*.egg-info/*",
    "*/.DS_Store",
    ".DS_Store",
    ".env",
    ".env.*",
    "*/.env",
    "*/.env.*",
    ".envrc",
    "*/.envrc",
    "*.pem",
    "*.key",
    "*.secret",
    "*.secrets",
    "*.log",
    "*.tmp",
)
ALLOWED_FORBIDDEN_TRACKED_ARTIFACTS = frozenset({".env.example"})

__all__ = [
    "ALLOWED_FORBIDDEN_TRACKED_ARTIFACTS",
    "FORBIDDEN_TRACKED_ARTIFACT_PATTERNS",
    "REPOSITORY_HYGIENE_RECORD_TYPE",
    "REQUIRED_GITIGNORE_PATTERNS",
    "RepositoryHygieneEvidence",
    "evaluate_repository_hygiene",
    "evaluate_repository_hygiene_paths",
    "repository_hygiene_to_record",
    "write_repository_hygiene_json",
    "main",
]


@dataclass(frozen=True, slots=True)
class RepositoryHygieneEvidence:
    repository_root: str
    required_gitignore_patterns: tuple[str, ...]
    missing_gitignore_patterns: tuple[str, ...]
    forbidden_tracked_artifact_patterns: tuple[str, ...]
    forbidden_tracked_paths: tuple[str, ...]
    forbidden_untracked_paths: tuple[str, ...]
    dirty_tracked_paths: tuple[str, ...]
    tracked_path_count: int
    untracked_path_count: int
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.issues

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_gitignore_patterns",
            _string_tuple(self.required_gitignore_patterns, "required_gitignore_patterns"),
        )
        object.__setattr__(
            self,
            "missing_gitignore_patterns",
            _string_tuple(self.missing_gitignore_patterns, "missing_gitignore_patterns"),
        )
        object.__setattr__(
            self,
            "forbidden_tracked_artifact_patterns",
            _string_tuple(self.forbidden_tracked_artifact_patterns, "forbidden_tracked_artifact_patterns"),
        )
        object.__setattr__(
            self,
            "forbidden_tracked_paths",
            _string_tuple(self.forbidden_tracked_paths, "forbidden_tracked_paths"),
        )
        object.__setattr__(
            self,
            "forbidden_untracked_paths",
            _string_tuple(self.forbidden_untracked_paths, "forbidden_untracked_paths"),
        )
        object.__setattr__(
            self,
            "dirty_tracked_paths",
            _string_tuple(self.dirty_tracked_paths, "dirty_tracked_paths"),
        )
        if not isinstance(self.repository_root, str) or not self.repository_root:
            raise ValueError("repository_root must be non-empty")
        if type(self.tracked_path_count) is not int or self.tracked_path_count < 0:
            raise ValueError("tracked_path_count must be a non-negative integer")
        if type(self.untracked_path_count) is not int or self.untracked_path_count < 0:
            raise ValueError("untracked_path_count must be a non-negative integer")
        explicit_issues = _string_tuple(self.issues, "issues")
        semantic_issues = _semantic_issues(
            missing_gitignore_patterns=self.missing_gitignore_patterns,
            forbidden_tracked_paths=self.forbidden_tracked_paths,
            forbidden_untracked_paths=self.forbidden_untracked_paths,
            dirty_tracked_paths=self.dirty_tracked_paths,
        )
        object.__setattr__(self, "issues", _dedupe_strings((*explicit_issues, *semantic_issues)))


def evaluate_repository_hygiene(repository_root: str | Path = ".") -> RepositoryHygieneEvidence:
    root = local_path(str(repository_root)).resolve()
    gitignore_lines = _gitignore_lines(root)
    tracked_paths = _tracked_paths(root)
    untracked_paths = _untracked_paths(root)
    dirty_tracked_paths = _dirty_tracked_paths(root)
    return evaluate_repository_hygiene_paths(
        repository_root=root,
        tracked_paths=tracked_paths,
        untracked_paths=untracked_paths,
        dirty_tracked_paths=dirty_tracked_paths,
        gitignore_lines=gitignore_lines,
    )


def evaluate_repository_hygiene_paths(
    *,
    repository_root: str | Path,
    tracked_paths: Sequence[str],
    gitignore_lines: Sequence[str],
    untracked_paths: Sequence[str] = (),
    dirty_tracked_paths: Sequence[str] = (),
) -> RepositoryHygieneEvidence:
    root = local_path(str(repository_root))
    gitignore_patterns = _normalized_gitignore_patterns(gitignore_lines)
    missing_gitignore_patterns = tuple(
        pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in gitignore_patterns
    )
    normalized_tracked_paths = tuple(path for path in tracked_paths if isinstance(path, str) and path)
    normalized_dirty_tracked_paths = tuple(
        sorted(path for path in dirty_tracked_paths if isinstance(path, str) and path)
    )
    forbidden_tracked_paths = tuple(
        sorted(path for path in normalized_tracked_paths if _is_forbidden_tracked_artifact(path))
    )
    normalized_untracked_paths = tuple(path for path in untracked_paths if isinstance(path, str) and path)
    forbidden_untracked_paths = tuple(
        sorted(path for path in normalized_untracked_paths if _is_forbidden_tracked_artifact(path))
    )
    return RepositoryHygieneEvidence(
        repository_root=str(root),
        required_gitignore_patterns=REQUIRED_GITIGNORE_PATTERNS,
        missing_gitignore_patterns=missing_gitignore_patterns,
        forbidden_tracked_artifact_patterns=FORBIDDEN_TRACKED_ARTIFACT_PATTERNS,
        forbidden_tracked_paths=forbidden_tracked_paths,
        forbidden_untracked_paths=forbidden_untracked_paths,
        dirty_tracked_paths=normalized_dirty_tracked_paths,
        tracked_path_count=len(normalized_tracked_paths),
        untracked_path_count=len(normalized_untracked_paths),
    )


def repository_hygiene_to_record(evidence: RepositoryHygieneEvidence) -> dict[str, Any]:
    return {
        "record_type": REPOSITORY_HYGIENE_RECORD_TYPE,
        "ok": evidence.ok,
        "repository_root": evidence.repository_root,
        "tracked_path_count": evidence.tracked_path_count,
        "required_gitignore_patterns": list(evidence.required_gitignore_patterns),
        "missing_gitignore_patterns": list(evidence.missing_gitignore_patterns),
        "forbidden_tracked_artifact_patterns": list(evidence.forbidden_tracked_artifact_patterns),
        "forbidden_tracked_paths": list(evidence.forbidden_tracked_paths),
        "forbidden_untracked_paths": list(evidence.forbidden_untracked_paths),
        "dirty_tracked_paths": list(evidence.dirty_tracked_paths),
        "issues": list(evidence.issues),
        "untracked_path_count": evidence.untracked_path_count,
    }


def write_repository_hygiene_json(evidence: RepositoryHygieneEvidence, path: str | Path) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(repository_hygiene_to_record(evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _semantic_issues(
    *,
    missing_gitignore_patterns: Sequence[str],
    forbidden_tracked_paths: Sequence[str],
    forbidden_untracked_paths: Sequence[str],
    dirty_tracked_paths: Sequence[str],
) -> tuple[str, ...]:
    issues: list[str] = []
    if missing_gitignore_patterns:
        issues.append("missing required .gitignore patterns: " + ", ".join(missing_gitignore_patterns))
    if forbidden_tracked_paths:
        issues.append("forbidden generated or secret-like tracked artifacts: " + ", ".join(forbidden_tracked_paths))
    if forbidden_untracked_paths:
        issues.append(
            "forbidden generated or secret-like untracked artifacts: "
            + ", ".join(forbidden_untracked_paths)
        )
    if dirty_tracked_paths:
        issues.append("dirty tracked paths: " + ", ".join(dirty_tracked_paths))
    return tuple(issues)


def _gitignore_lines(repository_root: Path) -> tuple[str, ...]:
    gitignore_path = repository_root / ".gitignore"
    try:
        return tuple(gitignore_path.read_text(encoding="utf-8").splitlines())
    except OSError as exc:
        raise RuntimeError(f"Could not read .gitignore at {gitignore_path}: {exc}") from exc


def _tracked_paths(repository_root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return tuple(path for path in completed.stdout.decode("utf-8").split("\0") if path)


def _untracked_paths(repository_root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return tuple(path for path in completed.stdout.decode("utf-8").split("\0") if path)


def _dirty_tracked_paths(repository_root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", "-z", "HEAD", "--"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return tuple(sorted(path for path in completed.stdout.decode("utf-8").split("\0") if path))


def _normalized_gitignore_patterns(lines: Sequence[str]) -> frozenset[str]:
    return frozenset(line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#"))


def _is_forbidden_tracked_artifact(path: str) -> bool:
    if path in ALLOWED_FORBIDDEN_TRACKED_ARTIFACTS or any(
        path.endswith(f"/{allowed}") for allowed in ALLOWED_FORBIDDEN_TRACKED_ARTIFACTS
    ):
        return False
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in FORBIDDEN_TRACKED_ARTIFACT_PATTERNS)


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{field_name} must be a sequence of strings")
    return tuple(values)


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _write_or_print(record: Mapping[str, Any], output_json: str | None) -> None:
    if output_json:
        output_path = local_path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(record, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect repository hygiene readiness.")
    parser.add_argument("--repository-root", default=".", help="Repository root to inspect. Defaults to cwd.")
    parser.add_argument("--output-json", help="Write the hygiene status JSON to this path instead of stdout.")
    args = parser.parse_args(argv)

    try:
        evidence = evaluate_repository_hygiene(args.repository_root)
        record = repository_hygiene_to_record(evidence)
        _write_or_print(record, args.output_json)
    except Exception as exc:
        record = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        _write_or_print(record, args.output_json)
        return 1
    return 0 if record["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
