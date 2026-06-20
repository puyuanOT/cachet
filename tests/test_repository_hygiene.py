import fnmatch
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

IGNORED_ARTIFACT_PATTERNS = (
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


def test_gitignore_covers_local_build_cache_and_secret_artifacts():
    ignored_lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert set(IGNORED_ARTIFACT_PATTERNS).issubset(ignored_lines)


def test_no_generated_artifacts_are_tracked():
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    tracked_paths = completed.stdout.decode("utf-8").split("\0")
    violations = sorted(path for path in tracked_paths if path and _is_forbidden_tracked_artifact(path))

    assert violations == []


def _is_forbidden_tracked_artifact(path: str) -> bool:
    if path == ".env.example" or path.endswith("/.env.example"):
        return False
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in FORBIDDEN_TRACKED_ARTIFACT_PATTERNS)
