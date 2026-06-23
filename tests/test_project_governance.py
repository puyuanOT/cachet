import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from textwrap import dedent
import tomllib

import pytest

from document_kv_cache.databricks_runs import databricks_run_status_sidecar_issues
from document_kv_cache.release_bundle import STRICT_V1_RELEASE_REQUIRED_ARTIFACTS


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PACKAGE_NAME = "restaurant_kv_serving"
LEGACY_PACKAGE_PREFIX = f"{LEGACY_PACKAGE_NAME}."
DYNAMIC_LEGACY_IMPORT_CALLS = {
    "__import__",
    "import_module",
    "importlib.import_module",
    "importorskip",
    "pytest.importorskip",
}
STRING_RESOLVED_TARGET_CALLS = {
    "patch",
    "patch.dict",
    "patch.multiple",
    "unittest.mock.patch",
    "unittest.mock.patch.dict",
    "unittest.mock.patch.multiple",
}
STRING_RESOLVED_TARGET_SUFFIXES = (
    ".delattr",
    ".patch",
    ".patch.dict",
    ".patch.multiple",
    ".setattr",
)
LEGACY_STRING_ARGUMENT_NAMES = {
    "in_dict",
    "modname",
    "module",
    "name",
    "target",
}
_DYNAMIC_DICT_KEY = object()
GENERATED_OR_TOOLING_DIRS = {
    ".coverage",
    ".git",
    ".hypothesis",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "htmlcov",
}
REPOSITORY_TEXT_FILE_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
}
REPOSITORY_TEXT_FILE_NAMES = {
    ".env.example",
    ".gitignore",
    "Dockerfile",
    "Makefile",
}
SECRET_PATTERNS = {
    "databricks_pat": re.compile(r"dapi[A-Za-z0-9]{32,}"),
    "github_pat": re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
    "langsmith_token": re.compile(r"lsv2_pt_[A-Za-z0-9_]{20,}"),
    "openai_api_key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "pem_private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
STABLE_EXACT_VERSION_RE = re.compile(r"^(?:==)?(?:\d+!)?\d+(?:\.\d+)*(?:\.post\d+)?$")
EXACT_REQUIREMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+])?==(.*?)(?:;.*)?$")
DEPRECATED_TOOL_POETRY_METADATA_KEYS = {
    "authors",
    "classifiers",
    "dependencies",
    "description",
    "extras",
    "keywords",
    "name",
    "readme",
    "scripts",
    "version",
}
ALLOWED_LEGACY_TEST_REFERENCES = {
    "tests/test_benchmark_plan_executor.py": {
        "restaurant_kv_serving.benchmark_plan_executor",
    },
    "tests/test_benchmark_plan.py": {
        "restaurant_kv_serving.benchmark_plan",
    },
    "tests/test_benchmark_runner.py": {
        "restaurant_kv_serving.benchmark_runner",
        "restaurant_kv_serving.benchmark_runner.run_openai_compatible_v1_benchmark",
    },
    "tests/test_benchmark_handoffs.py": {
        "restaurant_kv_serving.benchmark_handoffs",
    },
    "tests/test_cache.py": {
        "restaurant_kv_serving.cache",
    },
    "tests/test_databricks_job.py": {
        "restaurant_kv_serving.databricks_job",
    },
    "tests/test_databricks_runs.py": {
        "restaurant_kv_serving.databricks_runs",
    },
    "tests/test_databricks_storage_benchmark_job.py": {
        "restaurant_kv_serving.databricks_storage_benchmark_job",
    },
    "tests/test_databricks_engine_probe_job.py": {
        "restaurant_kv_serving.databricks_engine_probe_job",
    },
    "tests/test_databricks_vllm_smoke_job.py": {
        "restaurant_kv_serving.databricks_vllm_smoke_job",
    },
    "tests/test_engine_adapters.py": {
        "restaurant_kv_serving.engine_adapters",
    },
    "tests/test_engine_probe.py": {
        "restaurant_kv_serving.engine_probe",
    },
    "tests/test_kvpack.py": {
        "restaurant_kv_serving.kvpack",
    },
    "tests/test_live_server.py": {
        "restaurant_kv_serving.live_server",
    },
    "tests/test_openai_compatible.py": {
        "restaurant_kv_serving.openai_compatible",
    },
    "tests/test_pr_evidence.py": {
        "restaurant_kv_serving.pr_evidence",
    },
    "tests/test_planner_materializer.py": {
        "restaurant_kv_serving.manifest",
        "restaurant_kv_serving.materializer",
        "restaurant_kv_serving.models",
        "restaurant_kv_serving.planner",
    },
    "tests/test_public_package.py": {
        "restaurant_kv_serving",
        "restaurant_kv_serving.benchmark_plan",
        "restaurant_kv_serving.benchmark_plan_executor",
        "restaurant_kv_serving.benchmark_runner",
        "restaurant_kv_serving.benchmarks",
        "restaurant_kv_serving.dataset_prep",
        "restaurant_kv_serving.engine",
        "restaurant_kv_serving.engine_adapters",
        "restaurant_kv_serving.engine_launch_config",
        "restaurant_kv_serving.engine_protocol",
        "restaurant_kv_serving.kvpack",
        "restaurant_kv_serving.live_server",
        "restaurant_kv_serving.native_probe_factories",
        "restaurant_kv_serving.openai_compatible",
        "restaurant_kv_serving.probe_fixtures",
        "restaurant_kv_serving.pr_evidence",
        "restaurant_kv_serving.release_bundle",
        "restaurant_kv_serving.release_evidence",
        "restaurant_kv_serving.serving_env",
        "restaurant_kv_serving.storage_benchmark",
        "restaurant_kv_serving.vllm_smoke",
        "restaurant_kv_serving.workflow",
    },
    "tests/test_probe_fixtures.py": {
        "restaurant_kv_serving.probe_fixtures",
    },
    "tests/test_storage.py": {
        "restaurant_kv_serving.storage",
    },
    "tests/test_storage_benchmark.py": {
        "restaurant_kv_serving.storage_benchmark",
    },
    "tests/test_release_evidence.py": {
        "restaurant_kv_serving.release_evidence",
    },
    "tests/test_release_bundle.py": {
        "restaurant_kv_serving.release_bundle",
    },
    "tests/test_scheduler.py": {
        "restaurant_kv_serving.scheduler",
    },
    "tests/test_serving_env.py": {
        "restaurant_kv_serving.serving_env",
    },
    "tests/test_vllm_smoke.py": {
        "restaurant_kv_serving.vllm_smoke",
    },
}
ALLOWED_LEGACY_SOURCE_REFERENCES = {}


def _is_ignored(path: Path) -> bool:
    return any(part in GENERATED_OR_TOOLING_DIRS or part.endswith(".egg-info") for part in path.parts)


def _package_docstring(path: Path) -> str | None:
    init_file = path / "__init__.py"
    if not init_file.exists():
        return None
    module = ast.parse(init_file.read_text(encoding="utf-8"))
    return ast.get_docstring(module)


def _git_known_directories() -> set[Path]:
    """Return tracked and untracked directories so local PR slices document new folders before commit."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    directories = {REPO_ROOT}
    for line in result.stdout.splitlines():
        relative_file = Path(line)
        if _is_ignored(relative_file):
            continue
        for relative_parent in relative_file.parents:
            directory = REPO_ROOT / relative_parent
            if directory == REPO_ROOT.parent or _is_ignored(relative_parent):
                continue
            directories.add(directory)
    return directories


def _is_legacy_reference(name: str) -> bool:
    return name == LEGACY_PACKAGE_NAME or name.startswith(LEGACY_PACKAGE_PREFIX)


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def _import_aliases(module: ast.Module) -> dict[str, str]:
    aliases = {}
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                target_name = alias.name if alias.asname else local_name
                aliases[local_name] = target_name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                aliases[local_name] = f"{node.module}.{alias.name}"
    return aliases


def _resolve_import_alias(name: str, aliases: dict[str, str]) -> str:
    parts = name.split(".")
    for prefix_length in range(len(parts), 0, -1):
        prefix = ".".join(parts[:prefix_length])
        if prefix not in aliases:
            continue
        suffix = ".".join(parts[prefix_length:])
        return f"{aliases[prefix]}.{suffix}" if suffix else aliases[prefix]
    return name


def _legacy_import_from_references(node: ast.ImportFrom) -> set[str]:
    if node.module is None:
        return set()
    if node.module == LEGACY_PACKAGE_NAME:
        return {
            f"{LEGACY_PACKAGE_NAME}.{alias.name}" if alias.name != "*" else f"{LEGACY_PACKAGE_NAME}.*"
            for alias in node.names
        }
    if node.module.startswith(LEGACY_PACKAGE_PREFIX):
        return {node.module}
    return set()


def _legacy_string_reference(value: object) -> str | None:
    if isinstance(value, str) and _is_legacy_reference(value):
        return value
    return None


def _legacy_string_references_in_call(node: ast.Call, aliases: dict[str, str]) -> set[str]:
    call_name = _dotted_name(node.func)
    if call_name is None:
        return set()
    call_name = _resolve_import_alias(call_name, aliases)
    is_dynamic_import = call_name in DYNAMIC_LEGACY_IMPORT_CALLS
    is_string_target = call_name in STRING_RESOLVED_TARGET_CALLS or call_name.endswith(STRING_RESOLVED_TARGET_SUFFIXES)
    if not is_dynamic_import and not is_string_target:
        return set()

    candidate_values = [getattr(node.args[0], "value", None)] if node.args else []
    candidate_values.extend(
        getattr(keyword.value, "value", None)
        for keyword in node.keywords
        if keyword.arg in LEGACY_STRING_ARGUMENT_NAMES
    )
    return {
        reference
        for value in candidate_values
        if (reference := _legacy_string_reference(value)) is not None
    }


def _legacy_references_in_test_module(path: Path) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _import_aliases(module)
    legacy_references = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_legacy_reference(alias.name):
                    legacy_references.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            legacy_references.update(_legacy_import_from_references(node))
        elif isinstance(node, ast.Call):
            legacy_references.update(_legacy_string_references_in_call(node, aliases))
    return legacy_references


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _literal_dict_key(value: ast.AST | None) -> object:
    if value is None:
        return _DYNAMIC_DICT_KEY
    try:
        literal_key = ast.literal_eval(value)
    except (ValueError, TypeError):
        return _DYNAMIC_DICT_KEY
    try:
        hash(literal_key)
    except TypeError:
        return _DYNAMIC_DICT_KEY
    return literal_key


def _literal_dict_key_entries(node: ast.Dict, flattened_dicts: set[int] | None = None) -> list[tuple[object, int]]:
    entries = []
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:
            if isinstance(value, ast.Dict):
                if flattened_dicts is not None:
                    flattened_dicts.add(id(value))
                entries.extend(_literal_dict_key_entries(value, flattened_dicts))
            continue
        literal_key = _literal_dict_key(key)
        if literal_key is _DYNAMIC_DICT_KEY:
            continue
        entries.append((literal_key, key.lineno))
    return entries


def _duplicate_literal_dict_keys(path: Path) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    duplicates: list[str] = []
    flattened_dicts: set[int] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Dict):
            continue
        if id(node) in flattened_dicts:
            continue
        seen = set()
        for literal_key, line_number in _literal_dict_key_entries(node, flattened_dicts):
            if literal_key in seen:
                duplicates.append(f"{_display_path(path)}:{line_number}:{literal_key!r}")
            else:
                seen.add(literal_key)
    return duplicates


def test_repository_directories_have_readme_or_package_docstring():
    missing_docs = []
    for path in sorted(_git_known_directories()):
        relative = path.relative_to(REPO_ROOT)
        if _is_ignored(relative):
            continue
        if (path / "README.md").exists() or _package_docstring(path):
            continue
        missing_docs.append(str(relative))

    assert missing_docs == []


def test_governance_directory_scan_skips_notebook_checkpoints():
    assert _is_ignored(Path("notebooks/.ipynb_checkpoints/exploration-checkpoint.ipynb"))


def test_source_layout_readme_reflects_document_owned_implementation():
    text = (REPO_ROOT / "src" / "README.md").read_text(encoding="utf-8")
    compact_text = " ".join(text.split())

    assert "Cachet, the document KV-cache library" in compact_text
    assert "distribution package is `cachet-kv`" in text
    assert "public product import namespace is the branded `cachet` facade" in compact_text
    assert "`cachet/` is the branded import facade" in text
    assert "`document_kv_cache/` is the canonical implementation and compatibility" in text
    assert "`restaurant_kv_serving/` remains in source only for migration-history tests" in compact_text
    assert "`vllm_kv_injection/` and `sglang_kv_injection/` are vendored engine-adapter" in compact_text
    assert "contains the current implementation" not in text


def test_packaged_template_root_readmes_explain_subfolders():
    template_roots = [
        REPO_ROOT / "src" / "document_kv_cache" / "templates",
        REPO_ROOT / "src" / "document_kv_cache" / "templates" / "databricks",
    ]

    for path in template_roots:
        text = (path / "README.md").read_text(encoding="utf-8")
        child_directories = [
            child.name
            for child in sorted(path.iterdir())
            if child.is_dir() and not _is_ignored(child.relative_to(REPO_ROOT))
        ]

        assert "This folder" in text
        assert child_directories
        for child_name in child_directories:
            assert f"`{child_name}/`" in text


def test_document_package_readme_lists_public_modules_and_console_scripts():
    import document_kv_cache

    text = (REPO_ROOT / "src" / "document_kv_cache" / "README.md").read_text(encoding="utf-8")
    package_dir = REPO_ROOT / "src" / "document_kv_cache"
    package_modules = {
        path.stem
        for path in package_dir.glob("*.py")
        if path.stem != "__init__" and not path.stem.startswith("_")
    }
    public_modules = sorted(document_kv_cache._PUBLIC_SUBMODULES)
    compatibility_only_modules = sorted(package_modules - set(public_modules))
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    document_scripts = sorted(
        name for name in pyproject["project"]["scripts"] if name.startswith("document-kv-")
    )
    cachet_scripts = sorted(
        name for name in pyproject["project"]["scripts"] if name.startswith("cachet-")
    )

    assert public_modules
    assert set(public_modules) <= package_modules
    assert compatibility_only_modules == ["scheduler"]
    assert document_scripts
    assert cachet_scripts
    for module_name in public_modules:
        assert f"`{module_name}.py`" in text
    for module_name in compatibility_only_modules:
        assert f"`{module_name}.py`" in text
    assert "Compatibility-Only Modules" in text
    for script_name in document_scripts:
        assert f"`{script_name}`" in text
    for script_name in cachet_scripts:
        assert f"`{script_name}`" in text
    assert "Cachet-branded aliases" in text
    assert "`templates/`" in text
    assert "`templates/databricks/`" in text
    assert "compatibility-preserving implementation package" in text
    assert "canonical implementation modules" in text
    assert "Public files in this package define the document-owned classes" in text
    assert "merge-settings" in text
    assert "auto-merge" in text
    assert "merged-branch cleanup" in text
    assert "wrappers over implementation modules in `restaurant_kv_serving`" not in text
    assert "real wrapper modules" not in text


def test_legacy_package_readme_describes_migration_shims_not_new_implementation_owner():
    text = (REPO_ROOT / "src" / "restaurant_kv_serving" / "README.md").read_text(encoding="utf-8")
    compact_text = " ".join(text.split())

    assert "migration shims for callers that have not yet moved to" in text
    assert "modules in this package forward to document-owned implementations" in compact_text
    assert "engine_adapters.py` is a compatibility facade over" in text
    assert "databricks_runs.py` is a compatibility wrapper over" in text
    assert "release_evidence.py` is a compatibility wrapper over" in text
    assert "implementation modules that have not" not in text
    assert "this package owns" not in compact_text


def test_legacy_compatibility_removal_gate_is_documented():
    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    docs_readme = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    matrix = (REPO_ROOT / "docs" / "v1-requirements-matrix.md").read_text(encoding="utf-8")
    gate = (REPO_ROOT / "docs" / "legacy-compatibility-removal.md").read_text(encoding="utf-8")
    compact_gate = " ".join(gate.split())

    assert "`docs/legacy-compatibility-removal.md`" in root_readme
    assert "`legacy-compatibility-removal.md`" in docs_readme
    assert "`docs/legacy-compatibility-removal.md`" in matrix
    assert "one repository and one distribution package" in compact_gate
    assert "`puyuanOT/cachet`" in gate
    assert "`cachet-kv`" in gate
    assert "primary import surface is the branded `cachet` facade" in compact_gate
    assert "`document_kv_cache` remains the canonical implementation namespace" in compact_gate
    assert "`restaurant_kv_serving` package and `restaurant-kv-*` console scripts" in compact_gate
    assert "New code must not add production dependencies on the legacy package" in compact_gate
    assert "`pyproject.toml` no longer packages `restaurant_kv_serving`" in compact_gate
    assert "`src/document_kv_cache/release_bundle.py` rejects" in compact_gate
    assert "`restaurant_kv_serving/__init__.py`" in gate
    assert "`restaurant_kv_serving/py.typed`" in gate
    assert "`tests/test_public_package.py` keeps source-only compatibility checks" in compact_gate
    assert "`tests/test_project_governance.py` keeps legacy references scoped" in compact_gate
    assert "PR evidence sidecars with Refactor-skill evidence" in compact_gate
    assert "completed GPT-5.5 review" in compact_gate
    assert "Downstream Databricks benchmark runners and QA jobs have migrated" in compact_gate
    assert "record type `document_kv.legacy_compatibility_migration.v1`" in compact_gate
    assert "python -m document_kv_cache.legacy_compatibility --validate-json" in compact_gate
    assert "`release`, `benchmark`, `storage`, `native_probe`, and `smoke`" in compact_gate
    assert "no checked runner uses `restaurant_kv_serving` imports" in compact_gate
    assert "`restaurant-kv-*` commands" in compact_gate
    assert "`legacy_migration_evidence` artifact role" in compact_gate
    assert "Current AWS g6/L4 release evidence" in compact_gate
    assert "optional AWS g5/A10G compatibility evidence" in compact_gate
    assert "strict release-bundle package-wheel gates are updated" in compact_gate
    assert "Keep `restaurant_kv_serving` absent from `pyproject.toml` package metadata" in compact_gate
    assert "Delete `src/restaurant_kv_serving` after the downstream migration evidence is attached" in compact_gate
    assert "Update `README.md`, `docs/v1-requirements-matrix.md`, `src/README.md`" in compact_gate
    assert "Run the focused governance, public-package, and release-bundle tests" in compact_gate
    assert "Refresh the tested wheel and strict release bundle before publication" in compact_gate


def test_legacy_restaurant_imports_in_tests_are_explicitly_scoped():
    actual = {
        str(path.relative_to(REPO_ROOT)): _legacy_references_in_test_module(path)
        for path in sorted((REPO_ROOT / "tests").rglob("*.py"))
        if not _is_ignored(path.relative_to(REPO_ROOT))
    }
    actual = {path: imports for path, imports in actual.items() if imports}

    assert actual == ALLOWED_LEGACY_TEST_REFERENCES


def test_production_source_does_not_depend_on_legacy_restaurant_package():
    source_files = sorted((REPO_ROOT / "src").rglob("*.py"))
    actual = {
        str(path.relative_to(REPO_ROOT)): _legacy_references_in_test_module(path)
        for path in source_files
        if "restaurant_kv_serving" not in path.relative_to(REPO_ROOT).parts
        and not _is_ignored(path.relative_to(REPO_ROOT))
    }
    actual = {path: imports for path, imports in actual.items() if imports}

    assert actual == ALLOWED_LEGACY_SOURCE_REFERENCES


def test_legacy_reference_scanner_detects_import_edges_and_string_targets(tmp_path):
    path = tmp_path / "sample_test.py"
    path.write_text(
        dedent(
            """
            import importlib
            import importlib as il
            import restaurant_kv_serving.cache as legacy_cache
            from importlib import import_module as load_module
            from restaurant_kv_serving import models
            from restaurant_kv_serving import *
            from restaurant_kv_serving.storage import DiskRangeReader
            from pytest import importorskip as skip_optional
            from unittest.mock import patch as mock_patch

            importlib.import_module("restaurant_kv_serving.dynamic")
            il.import_module("restaurant_kv_serving.alias_dynamic")
            load_module(name="restaurant_kv_serving.keyword_dynamic")
            pytest.importorskip("restaurant_kv_serving.optional")
            skip_optional(modname="restaurant_kv_serving.alias_optional")
            monkeypatch.setattr("restaurant_kv_serving.runner.hook", fake_hook)
            monkeypatch.delattr(target="restaurant_kv_serving.runner.old_hook")
            mock_patch(target="restaurant_kv_serving.mock.patch_target")
            mock_patch.dict("restaurant_kv_serving.mock.CONFIG", {})
            mock_patch.dict(in_dict="restaurant_kv_serving.mock.KEYWORD_CONFIG", values={})
            mock_patch.multiple(target="restaurant_kv_serving.mock.multi_target", hook=fake_hook)
            """
        ),
        encoding="utf-8",
    )

    assert _legacy_references_in_test_module(path) == {
        "restaurant_kv_serving.*",
        "restaurant_kv_serving.alias_dynamic",
        "restaurant_kv_serving.alias_optional",
        "restaurant_kv_serving.cache",
        "restaurant_kv_serving.dynamic",
        "restaurant_kv_serving.keyword_dynamic",
        "restaurant_kv_serving.mock.CONFIG",
        "restaurant_kv_serving.mock.KEYWORD_CONFIG",
        "restaurant_kv_serving.mock.multi_target",
        "restaurant_kv_serving.models",
        "restaurant_kv_serving.mock.patch_target",
        "restaurant_kv_serving.optional",
        "restaurant_kv_serving.runner.hook",
        "restaurant_kv_serving.runner.old_hook",
        "restaurant_kv_serving.storage",
    }


def test_duplicate_literal_dict_key_scanner_reports_silent_overwrites(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(
        dedent(
            """
            dynamic = "record_type"
            ok = {
                "record_type": "first",
                dynamic: "ignored",
                "record_type": "second",
                1: "one",
                1: "another",
                **{"spread": "first", "literal_unpack": "first"},
                **dynamic_source,
                "literal_unpack": "second",
                **{"inner_dup": "first", "inner_dup": "second"},
            }
            """
        ),
        encoding="utf-8",
    )

    assert _duplicate_literal_dict_keys(path) == [
        f"{_display_path(path)}:6:'record_type'",
        f"{_display_path(path)}:8:1",
        f"{_display_path(path)}:11:'literal_unpack'",
        f"{_display_path(path)}:12:'inner_dup'",
    ]


def test_python_source_files_do_not_repeat_literal_dict_keys():
    duplicates = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")) + sorted((REPO_ROOT / "tests").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT)
        if _is_ignored(relative):
            continue
        duplicates.extend(_duplicate_literal_dict_keys(path))

    assert duplicates == []


def test_contributing_doc_records_required_pr_workflow():
    text = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    compact_text = " ".join(text.split())

    assert "Direct pushes to `main`" in compact_text
    assert ".github/main-branch-protection.json" in text
    assert "one approving review" in text
    assert "Test and build" in text
    assert "force-pushes" in text
    assert "pull requests" in text
    assert "Refactor skill" in text
    assert "GPT-5.5 review" in text
    assert "what changed and why" in text
    assert "vLLM or SGLang" in text
    assert "Do not add a proprietary request scheduler" in text
    assert "custom serving solver" in text
    assert "handoff/adapter boundary" in text


def _markdown_section(text: str, heading: str) -> str:
    start_marker = f"## {heading}"
    start = text.index(start_marker)
    next_heading = text.find("\n## ", start + len(start_marker))
    return text[start:] if next_heading == -1 else text[start:next_heading]


def _first_python_fence_after(text: str, marker: str) -> str:
    start = text.index(marker)
    fence_start = text.index("```python", start)
    code_start = text.index("\n", fence_start) + 1
    fence_end = text.index("```", code_start)
    return text[code_start:fence_end]


def _first_bash_fence_after(text: str, marker: str) -> str:
    start = text.index(marker)
    fence_start = text.index("```bash", start)
    code_start = text.index("\n", fence_start) + 1
    fence_end = text.index("```", code_start)
    return text[code_start:fence_end]


def test_readme_documents_cachet_brand_and_scope():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    purpose = _markdown_section(text, "Purpose And Scope")
    compact_purpose = " ".join(purpose.split())

    assert text.startswith("# Cachet: Document KV Cache")
    assert "Cachet is a reusable document KV-cache orchestration package" in text
    compact_text = " ".join(text.split())

    assert "Cachet is the product brand and primary repository identity" in compact_text
    assert "The package publishes through the Cachet-branded `cachet-kv` distribution name" in compact_text
    assert "unrelated Cachet API client" in compact_text
    assert "branded `cachet` root and `cachet.<module>` import facades" in compact_text
    assert "canonical `document_kv_cache` implementation modules" in compact_text
    assert "applications that repeatedly serve long, mostly stable document context" in compact_purpose
    assert "Biography, HotpotQA, MusiQue, and Needle-in-a-Haystack" in compact_purpose
    assert "standard no-cache prefill baseline" in compact_purpose
    assert "vLLM, SGLang, or another established serving engine owns scheduling" in compact_purpose


def test_readme_cachet_first_quickstart_uses_branded_imports_and_cli():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = _markdown_section(text, "Cachet-First Quickstart")

    assert "from cachet import" in quickstart
    assert "from cachet.engine_launch_config import" in quickstart
    assert "cachet-benchmark-handoff-bundles --help" in quickstart
    assert "cachet-benchmark-plan --help" in quickstart
    assert "cachet-engine-launch-config build-vllm" in quickstart
    assert "cachet-databricks-runs payload-summary --help" in quickstart
    assert "Compatibility import paths and legacy command aliases" in quickstart
    assert "document_kv_cache" not in quickstart
    assert "document-kv-" not in quickstart
    assert "restaurant_kv_serving" not in quickstart
    assert "restaurant-kv-" not in quickstart

    example = _first_python_fence_after(text, "Use the Cachet-branded imports")
    namespace: dict[str, object] = {}
    exec(compile(example, "README.md", "exec"), namespace)

    document = namespace["document"]
    request = namespace["request"]
    layout = namespace["layout"]
    launch_config = namespace["vllm_launch_config"]

    assert document.document_id == "policy-handbook"
    assert request.selected_document_ids == (document.document_id,)
    assert layout.layout_version == "qwen3-v1"
    assert launch_config["kv_connector"] == "DocumentKVConnector"
    assert launch_config["kv_connector_extra_config"]["document_kv.backend"] == "vllm"


def test_readme_engine_adapter_handoff_example_uses_public_payload_reader():
    import document_kv_cache

    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    example = _first_python_fence_after(text, "For engine-specific integration code")
    tree = ast.parse(example)
    document_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "document_kv_cache"
        for alias in node.names
    }

    assert "adapter_storage" not in example
    assert "write_engine_adapter_handoff_bundle(" in example
    assert ".open(\"wb\")" not in example
    assert "payload = read_engine_adapter_payload(" in example
    assert "expected_bytes=record[\"payload_source\"][\"total_bytes\"]" in example
    assert document_imports
    missing_exports = sorted(name for name in document_imports if not hasattr(document_kv_cache, name))
    assert missing_exports == []


def test_readme_benchmark_plan_examples_include_release_actions_sidecars():
    root_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    legacy_text = (REPO_ROOT / "src" / "restaurant_kv_serving" / "README.md").read_text(encoding="utf-8")
    compact_root_text = " ".join(root_text.split())
    compact_legacy_text = " ".join(legacy_text.split())
    root_example = _first_bash_fence_after(root_text, "To run the V1 benchmark contract")
    legacy_example = _first_bash_fence_after(legacy_text, "`benchmark_plan.py` emits")

    for example in (root_example, legacy_example):
        assert "--engine-probe-output-json vllm=/data/vllm-engine-probe.json" in example
        assert "--engine-probe-actions-output-json vllm=/data/vllm-connector-actions.json" in example
        assert "--engine-probe-output-json sglang=/data/sglang-engine-probe.json" in example
        assert "--engine-probe-actions-output-json sglang=/data/sglang-connector-actions.json" in example
        assert "--release-evidence-output-json /data/release-evidence.json" in example

    assert "release evidence must include `--engine-probe-actions-output-json`" in compact_root_text
    assert "actions_output_json" in root_text
    assert "release evidence must include `--engine-probe-actions-output-json`" in compact_legacy_text
    assert "native probe and connector-action records already exist" in compact_root_text
    assert "Existing native probe and connector-action JSONs" in compact_legacy_text
    assert "--release-engine-actions-json" in compact_root_text
    assert "--release-engine-actions-json" in compact_legacy_text


def test_readme_model_profile_example_uses_portable_definition_artifact():
    import document_kv_cache

    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    example = _first_python_fence_after(text, "Future Qwen3.5, MiniMax")
    tree = ast.parse(example)
    document_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "document_kv_cache"
        for alias in node.names
    }

    assert "definition = ModelProfileDefinition(" in example
    assert "with_definition(definition)" in example
    assert "num_kv_heads=1" in example
    assert "Provider/Future-MQA-4B" in example
    assert "future-mqa-profile.json" in example
    assert document_imports
    missing_exports = sorted(name for name in document_imports if not hasattr(document_kv_cache, name))
    assert missing_exports == []


def test_project_metadata_uses_cachet_branded_distribution():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["name"] == "cachet-kv"
    assert "Cachet document KV-cache" in project["description"]
    assert "cachet" in project["keywords"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert "License :: OSI Approved :: Apache Software License" in project["classifiers"]


def test_readme_and_root_license_document_apache_2_license():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    license_section = _markdown_section(readme, "License")

    assert "Apache License 2.0" in license_section
    assert "`Apache-2.0` SPDX expression" in license_section
    assert "`LICENSE`" in license_section
    assert "`py.typed` markers" in license_section
    assert "`cachet`" in license_section
    assert "`document_kv_cache`" in license_section
    assert "`restaurant_kv_serving`" not in license_section
    assert license_text.startswith("Apache License\nVersion 2.0, January 2004")
    assert "https://www.apache.org/licenses/" in license_text


def test_project_metadata_exposes_repository_and_issue_urls():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["urls"] == {
        "Repository": "https://github.com/puyuanOT/cachet",
        "Issues": "https://github.com/puyuanOT/cachet/issues",
    }


def test_readme_development_commands_use_public_package_branding():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    development_section = _markdown_section(text, "Development")

    assert "poetry check --lock" in development_section
    assert "poetry install -E test" in development_section
    assert "poetry run pytest -q" in development_section
    assert "python -m pip install -e '.[test]'" in development_section
    assert "pytest tests -q" in development_section
    assert "restaurant-kv-serving[test]" not in text
    assert "restaurant-kv-serving/tests" not in text


def test_readme_avoids_workspace_local_script_references():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "../scripts" not in text
    assert "submit_document_kv_v1_benchmark.py" not in text
    assert "Workspace-specific automation can wrap this handoff" in text


def test_readme_manifest_schema_mentions_storage_layout():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    logical_model = _markdown_section(text, "Logical Model")
    manifest_start = logical_model.index("Manifest table:")
    fence_start = logical_model.rindex("```text", 0, manifest_start)
    fence_end = logical_model.index("```", manifest_start)
    manifest_table = logical_model[fence_start:fence_end]

    assert "layout_version\n  storage_layout" in manifest_table


def test_readme_remaining_work_keeps_serving_boundary_explicit():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    remaining_work = _markdown_section(text, "Remaining V1 Work")

    assert "connector action descriptors" in remaining_work
    assert "native engine block managers" in remaining_work
    assert "do not add a proprietary scheduler or custom solver" in remaining_work
    assert "Add vLLM and SGLang adapters" not in remaining_work


def test_v1_requirements_matrix_tracks_goal_evidence_and_remaining_gates():
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    docs_readme = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    matrix_text = (REPO_ROOT / "docs" / "v1-requirements-matrix.md").read_text(encoding="utf-8")
    compact_matrix = " ".join(matrix_text.split())
    remaining_release_gates = _markdown_section(matrix_text, "Remaining V1 Release Gates")
    compact_remaining_release_gates = " ".join(remaining_release_gates.split())
    stale_release_blockers = (
        "no-governance",
        "21 artifacts",
        "allow_auto_merge=false",
        "repository as private",
        "repository visibility as private",
        "still requires the GitHub governance sidecar",
        "still needs target AWS g6/L4 or Unity Catalog evidence in the release bundle",
        "previous complete g5-enriched strict bundle",
        "next complete strict bundle refresh",
        "PR #427 PR-evidence sidecar",
        "426398182137665",
        "315109189523858",
        "cachet_vllm_hot_payload_9ec0657_20260623_053557_repeat3_cache8g_current_main",
        "cachet_vllm_hot_payload_g5_01a6147_20260623_125720_repeat3_cache8g_current_main",
    )

    assert "docs/v1-requirements-matrix.md" in readme_text
    assert "`v1-requirements-matrix.md`" in docs_readme
    assert "Status values" in matrix_text
    assert "**Implemented:**" in matrix_text
    assert "**Bundle-refresh pending:**" in matrix_text
    assert "**Release-gated:**" in matrix_text
    assert "**Remaining:**" in matrix_text

    for required in (
        "vLLM/SGLang handoff boundary",
        "Memory, Disk, UC Volume, and routed readers",
        "AWS g6/L4",
        "g5/A10G compatibility evidence",
        "g5/A10G compatibility benchmark evidence",
        "`aws-g5-a10g`",
        "`g5.8xlarge`",
        "`qwen3:4b-instruct`",
        "Biography",
        "HotpotQA",
        "MusiQue",
        "NIAH",
        "`baseline_prefill` arm",
        "MQA/GQA",
        "KV Packet",
        "Qwen3.5",
        "MiniMax",
        "Cachet",
        "GPT-5.5 review",
        "Refactor skill",
        "one PR open",
        "complete strict release bundle",
        "strict V1 publication target",
        "compatibility_benchmark",
        "compatibility_databricks_run_status",
        "23 artifacts",
        "872615985402004",
        "566743786103032",
        "cachet_vllm_hot_payload_longcmp_388ea0a_20260623_160711_repeat3_cache8g_cachet_kv_current_main",
        "cachet_vllm_hot_payload_g5_longcmp_388ea0a_20260623_162302_repeat3_cache8g_cachet_kv_current_main",
        "real vLLM and SGLang native block managers",
        "docs/native-engine-integration.md",
        "`restaurant_kv_serving` compatibility directory",
    ):
        assert required in matrix_text

    assert "Release-gated | `databricks_job.py`" in compact_matrix
    assert "Run connector action descriptor validation" in matrix_text
    assert (
        "release evidence `ok=true` when paired with the current storage and native engine sidecars"
        in matrix_text
    )
    assert "cannot substitute for the strict release target" in matrix_text
    assert "`compatibility_benchmark` artifact role" in matrix_text
    assert "current complete g5-enriched strict bundle validates with 23 artifacts" in compact_matrix
    assert "compatibility Databricks run-status sidecar" in compact_matrix
    assert "GitHub governance is release-ready" in compact_matrix
    assert "auto-merge is enabled" in compact_matrix
    assert "Add native engine integration examples after connector probes land" not in matrix_text
    for stale_blocker in stale_release_blockers:
        assert stale_blocker not in matrix_text
    for _role, _minimum_count, label in STRICT_V1_RELEASE_REQUIRED_ARTIFACTS:
        assert label in compact_remaining_release_gates


def test_native_engine_integration_doc_examples_are_validated():
    text = (REPO_ROOT / "docs" / "native-engine-integration.md").read_text(encoding="utf-8")
    docs_readme = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "`native-engine-integration.md`" in docs_readme
    assert "docs/native-engine-integration.md" in readme_text
    for required in (
        "# Native Engine Integration",
        "cachet-engine-launch-config build-vllm",
        "cachet-engine-launch-config build-sglang",
        "--payload-cache-max-bytes 8589934592",
        "DocumentKVConnector",
        "DocumentKVHiCacheBackend",
        "vllm_kv_injection.vllm_native_provider:build_document_kv_provider",
        "sglang_kv_injection.sglang_dynamic_backend:build_document_kv_hicache_provider",
        "kv_transfer_params",
        "full logical prompt",
        "For SGLang, the current evidence covers",
        "Validate live decode-time prefix binding",
        "hardware target",
        "no-op providers",
        "in-memory test connectors",
        "aws-g6-l4",
        "g6.8xlarge",
        "aws-g5-a10g",
        "g5.8xlarge",
        "payload-summary",
        "benchmarks/databricks/2026-06-23-g6-l4-native-engine-probes/",
        "commit tokens",
    ):
        assert required in text

    example = _first_python_fence_after(text, "Validate launch configs in Python")
    namespace: dict[str, object] = {}
    exec(compile(example, "docs/native-engine-integration.md", "exec"), namespace)

    vllm_launch_config = namespace["vllm_launch_config"]
    sglang_launch_config = namespace["sglang_launch_config"]
    assert isinstance(vllm_launch_config, dict)
    assert isinstance(sglang_launch_config, dict)
    assert vllm_launch_config["kv_connector"] == "DocumentKVConnector"
    assert vllm_launch_config["kv_connector_extra_config"]["document_kv.backend"] == "vllm"
    assert json.loads(sglang_launch_config["hicache_storage_backend_extra_config"])[
        "document_kv.backend"
    ] == "sglang"


def test_standalone_benchmark_evidence_folders_track_current_databricks_runs():
    benchmark_root = REPO_ROOT / "benchmarks"
    databricks_root = benchmark_root / "databricks"
    root_readme = (benchmark_root / "README.md").read_text(encoding="utf-8")
    databricks_readme = (databricks_root / "README.md").read_text(encoding="utf-8")
    current_databricks_snapshot = (databricks_root / "CURRENT.md").read_text(encoding="utf-8")
    project_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    docs_readme = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    matrix_text = (REPO_ROOT / "docs" / "v1-requirements-matrix.md").read_text(encoding="utf-8")
    compact_project_readme = " ".join(project_readme.split())
    compact_snapshot = " ".join(current_databricks_snapshot.split())

    assert "[`benchmarks/`](benchmarks/README.md)" in project_readme
    assert "[`benchmarks/databricks/CURRENT.md`](benchmarks/databricks/CURRENT.md)" in project_readme
    assert "[`databricks/CURRENT.md`](databricks/CURRENT.md)" in root_readme
    assert "[`CURRENT.md`](CURRENT.md)" in databricks_readme
    assert "`../benchmarks/README.md`" in docs_readme
    assert "Standalone human-readable benchmark folders" in matrix_text
    assert "`pr-evidence/` tree" in compact_project_readme
    assert "without requiring readers to inspect `pr-evidence/`" in compact_snapshot
    assert "ignored local `databricks-runs/` output" in compact_snapshot
    assert "Raw local run directories stay under ignored `databricks-runs/`" in databricks_readme
    assert "Strict publication target | AWS g6/L4, `aws-g6-l4`, `g6.8xlarge`" in current_databricks_snapshot
    assert "Compatibility target | AWS g5/A10G, `aws-g5-a10g`, `g5.8xlarge`" in current_databricks_snapshot
    assert "`baseline_prefill`" in current_databricks_snapshot
    assert "`document_kv_cache`" in current_databricks_snapshot
    assert "does not replace the strict AWS g6/L4 publication target" in current_databricks_snapshot
    assert "instead of a package-owned serving scheduler" in current_databricks_snapshot
    assert "Release-bundle manifests are regenerated from these benchmark artifacts" in compact_snapshot

    expected_files = {
        "databricks/CURRENT.md",
        "README.md",
        "databricks/README.md",
        "databricks/2026-06-21-g6-l4-storage-readers/README.md",
        "databricks/2026-06-21-g6-l4-storage-readers/databricks_run_status.json",
        "databricks/2026-06-21-g6-l4-storage-readers/storage_benchmark.json",
        "databricks/2026-06-23-g5-a10g-v1-compatibility/README.md",
        "databricks/2026-06-23-g5-a10g-v1-compatibility/databricks_run_status.json",
        "databricks/2026-06-23-g5-a10g-v1-compatibility/v1_benchmark.json",
        "databricks/2026-06-23-g6-l4-native-engine-probes/README.md",
        "databricks/2026-06-23-g6-l4-native-engine-probes/databricks_run_status.json",
        "databricks/2026-06-23-g6-l4-native-engine-probes/sglang_connector_actions.json",
        "databricks/2026-06-23-g6-l4-native-engine-probes/sglang_engine_probe.json",
        "databricks/2026-06-23-g6-l4-native-engine-probes/vllm_connector_actions.json",
        "databricks/2026-06-23-g6-l4-native-engine-probes/vllm_engine_probe.json",
        "databricks/2026-06-23-g6-l4-v1/README.md",
        "databricks/2026-06-23-g6-l4-v1/databricks_run_status.json",
        "databricks/2026-06-23-g6-l4-v1/release_evidence.json",
        "databricks/2026-06-23-g6-l4-v1/v1_benchmark.json",
    }
    actual_files = {
        path.relative_to(benchmark_root).as_posix()
        for path in benchmark_root.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files

    expected_folders = {
        "2026-06-23-g6-l4-v1": {
            "run_id": 872615985402004,
            "hardware_target": "aws-g6-l4",
            "node_type": "g6.8xlarge",
        },
        "2026-06-23-g5-a10g-v1-compatibility": {
            "run_id": 566743786103032,
            "hardware_target": "aws-g5-a10g",
            "node_type": "g5.8xlarge",
        },
        "2026-06-21-g6-l4-storage-readers": {
            "run_id": 948365719597221,
            "hardware_target": "aws-g6-l4",
            "node_type": "g6.8xlarge",
        },
        "2026-06-23-g6-l4-native-engine-probes": {
            "run_id": 934698284395881,
            "hardware_target": "aws-g6-l4",
            "node_type": "g6.8xlarge",
        },
    }

    for folder_name, expected in expected_folders.items():
        folder = databricks_root / folder_name
        folder_readme = (folder / "README.md").read_text(encoding="utf-8")
        status = json.loads((folder / "databricks_run_status.json").read_text(encoding="utf-8"))
        summary = status["summary"]

        assert folder_name in root_readme
        assert str(expected["run_id"]) in root_readme
        assert str(expected["run_id"]) in folder_readme
        assert expected["hardware_target"] in folder_readme
        assert expected["node_type"] in folder_readme
        assert status["ok"] is True
        assert status["action"] == "get"
        assert summary["record_type"] == "document_kv.databricks_run_status.v1"
        assert summary["run_id"] == expected["run_id"]
        assert summary["terminal"] is True
        assert summary["succeeded"] is True
        assert summary["life_cycle_state"] == "TERMINATED"
        assert summary["result_state"] == "SUCCESS"
        assert summary["active_task_key"] is None
        assert summary["submit_payload"]["hardware_targets"] == [expected["hardware_target"]]
        assert summary["submit_payload"]["node_type_ids"] == [expected["node_type"]]
        assert databricks_run_status_sidecar_issues(
            status,
            expected_hardware_target=expected["hardware_target"],
            expected_node_type_id=expected["node_type"],
        ) == ()

    g6_benchmark = json.loads(
        (databricks_root / "2026-06-23-g6-l4-v1" / "v1_benchmark.json").read_text(encoding="utf-8")
    )
    assert g6_benchmark["record_type"] == "document_kv.benchmark_run.v1"
    assert g6_benchmark["suite"]["hardware_target"] == "aws-g6-l4"
    assert g6_benchmark["suite"]["model_id"] == "qwen3:4b-instruct"
    assert g6_benchmark["suite"]["datasets"] == ["biography", "hotpotqa", "musique", "niah"]
    assert len(g6_benchmark["measurements"]) == 24
    assert g6_benchmark["v1_evidence"]["ok"] is True
    assert min(row["ttft_speedup"] for row in g6_benchmark["comparisons"]) == pytest.approx(5.2668743391)
    assert max(row["ttft_speedup"] for row in g6_benchmark["comparisons"]) == pytest.approx(6.9722325045)

    g6_release_evidence = json.loads(
        (databricks_root / "2026-06-23-g6-l4-v1" / "release_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert g6_release_evidence["record_type"] == "document_kv.release_evidence.v1"
    assert g6_release_evidence["ok"] is True
    assert g6_release_evidence["issues"] == []
    assert g6_release_evidence["v1_benchmark_ok"] is True
    assert g6_release_evidence["storage_benchmark_ok"] is True
    assert set(g6_release_evidence["engine_probe_backends"]) == {"sglang", "vllm"}
    artifact_sources = g6_release_evidence["artifact_sources"]
    artifact_source_paths = {source["path"] for source in artifact_sources}
    assert not any(path.startswith("databricks-runs/") for path in artifact_source_paths)
    assert artifact_source_paths == {
        "benchmarks/databricks/2026-06-21-g6-l4-storage-readers/storage_benchmark.json",
        "benchmarks/databricks/2026-06-23-g6-l4-native-engine-probes/sglang_connector_actions.json",
        "benchmarks/databricks/2026-06-23-g6-l4-native-engine-probes/sglang_engine_probe.json",
        "benchmarks/databricks/2026-06-23-g6-l4-native-engine-probes/vllm_connector_actions.json",
        "benchmarks/databricks/2026-06-23-g6-l4-native-engine-probes/vllm_engine_probe.json",
        "benchmarks/databricks/2026-06-23-g6-l4-v1/v1_benchmark.json",
    }
    for source in artifact_sources:
        artifact_path = REPO_ROOT / source["path"]
        artifact_bytes = artifact_path.read_bytes()
        assert source["size_bytes"] == len(artifact_bytes)
        assert source["sha256"] == hashlib.sha256(artifact_bytes).hexdigest()

    g5_benchmark = json.loads(
        (
            databricks_root
            / "2026-06-23-g5-a10g-v1-compatibility"
            / "v1_benchmark.json"
        ).read_text(encoding="utf-8")
    )
    assert g5_benchmark["suite"]["hardware_target"] == "aws-g5-a10g"
    assert len(g5_benchmark["measurements"]) == 24
    assert g5_benchmark["v1_evidence"]["ok"] is True
    assert min(row["ttft_speedup"] for row in g5_benchmark["comparisons"]) == pytest.approx(4.6620010419)
    assert max(row["ttft_speedup"] for row in g5_benchmark["comparisons"]) == pytest.approx(6.0430383626)

    storage_benchmark = json.loads(
        (
            databricks_root
            / "2026-06-21-g6-l4-storage-readers"
            / "storage_benchmark.json"
        ).read_text(encoding="utf-8")
    )
    assert storage_benchmark["record_type"] == "document_kv.storage_benchmark.v1"
    assert storage_benchmark["uc_volume_is_real"] is True
    assert storage_benchmark["readers"] == ["memory", "disk", "unity_catalog"]
    assert storage_benchmark["release_storage_evidence"]["ok"] is True
    assert {row["reader_id"] for row in storage_benchmark["results"]} == {
        "disk",
        "memory",
        "unity_catalog",
    }
    assert all(row["errors"] == 0 for row in storage_benchmark["results"])

    native_probe_root = databricks_root / "2026-06-23-g6-l4-native-engine-probes"
    for backend, provider_factory in {
        "vllm": "vllm_kv_injection.vllm_native_provider:build_document_kv_provider",
        "sglang": "sglang_kv_injection.sglang_dynamic_backend:build_document_kv_hicache_provider",
    }.items():
        probe = json.loads((native_probe_root / f"{backend}_engine_probe.json").read_text(encoding="utf-8"))
        actions = json.loads(
            (native_probe_root / f"{backend}_connector_actions.json").read_text(encoding="utf-8")
        )

        assert probe["record_type"] == "document_kv.engine_kv_connector_probe.v1"
        assert probe["backend"] == backend
        assert probe["native_probe"] is True
        assert probe["payload_mode"] == "merged"
        assert probe["copied_tokens"] == 48
        assert probe["metadata"][f"{backend}_kv_injection.provider_factory"] == provider_factory
        assert actions["record_type"] == "document_kv.engine_kv_connector_actions.v1"
        assert actions["backend"] == backend


def test_readme_release_bundle_documents_artifact_validation_contracts():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    compact_text = " ".join(text.split())
    remaining_v1_work = _markdown_section(text, "Remaining V1 Work")
    compact_remaining_v1_work = " ".join(remaining_v1_work.split())
    stale_release_blockers = (
        "no-governance",
        "21 artifacts",
        "allow_auto_merge=false",
        "repository as private",
        "repository visibility as private",
        "still requires the GitHub governance sidecar",
        "still needs to be bundled before publication",
        "previous g5-enriched strict bundle",
        "next complete bundle refresh",
        "PR #427 evidence sidecar",
        "426398182137665",
        "315109189523858",
        "cachet_vllm_hot_payload_9ec0657_20260623_053557_repeat3_cache8g_current_main",
        "cachet_vllm_hot_payload_g5_01a6147_20260623_125720_repeat3_cache8g_current_main",
    )

    assert "package name/version for wheel artifacts" in compact_text
    assert "records the normalized package name and package version" in compact_text
    assert "current project version from `pyproject.toml` or installed package metadata" in compact_text
    assert "free of active task keys" in compact_text
    assert "task summaries carry non-empty `purpose` tags" in compact_text
    assert "summary arrays match the task summaries" in compact_text
    assert "V1 requirements matrix" in compact_text
    assert "--compatibility-databricks-run-status-json" in compact_text
    assert "AWS g5/A10G compatibility benchmark evidence" in compact_remaining_v1_work
    assert "--compatibility-benchmark-json" in compact_remaining_v1_work
    assert "g5-enriched strict bundle validates with 23 artifacts" in compact_remaining_v1_work
    assert "current release-gate PR evidence sidecar" in compact_remaining_v1_work
    assert "compatibility_databricks_run_status" in compact_remaining_v1_work
    assert "GitHub governance sidecar" in compact_remaining_v1_work
    assert "Current governance evidence is green" in compact_remaining_v1_work
    assert "566743786103032" in compact_remaining_v1_work
    assert "release evidence `ok=true`" in compact_remaining_v1_work
    assert (
        "does not change the strict V1 publication target from AWS g6/L4"
        in compact_remaining_v1_work
    )
    for stale_blocker in stale_release_blockers:
        assert stale_blocker not in text
    for _role, _minimum_count, label in STRICT_V1_RELEASE_REQUIRED_ARTIFACTS:
        assert label in compact_remaining_v1_work


def test_readme_native_probe_diagnostics_include_serving_environment_profile():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    serving_handoff = _markdown_section(text, "Serving Engine Handoff")
    compact_serving_handoff = " ".join(serving_handoff.split())

    assert "builtin_native_probe_factories_to_record()" in serving_handoff
    assert "ServingEngineConnector" in serving_handoff
    assert "prepare_and_submit_to_engine(" in serving_handoff
    assert "document-kv-serving-env" in serving_handoff
    assert "document-kv-native-probe-factories" in serving_handoff
    assert "fail closed" in compact_serving_handoff
    assert "pinned isolated serving-environment profile" in compact_serving_handoff
    assert "target engine versions and dependency constraints" in compact_serving_handoff
    assert "EngineKVInjectionPlan" in serving_handoff
    assert "layout-derived byte totals, block totals" in serving_handoff
    assert "before native block-manager calls" in compact_serving_handoff


def test_readme_workflow_api_shows_single_text_document_helper():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    storage_backends = _markdown_section(text, "Storage Backends")
    workflow_api = _markdown_section(text, "Workflow API")

    assert "MemoryRangeReader" in storage_backends
    assert "memory:" in storage_backends
    assert "SourceDocument.from_text(" in workflow_api
    assert "DocumentKVWorkflow.with_storage(" in workflow_api
    assert "DocumentKVRequest.for_text_document(" in workflow_api
    assert "memory:" in workflow_api
    assert "SourceDocument.from_texts(" in workflow_api
    assert "static_chunk_metadata=" in workflow_api
    assert "chunk_metadata=" in workflow_api
    assert "DocumentKVRequest.for_document_chunks(" in workflow_api
    assert "DocumentKVRequest.for_document_selection(" in workflow_api
    assert 'document_id="doc-a"' in workflow_api


def test_pull_request_template_captures_traceability_and_review_gates():
    text = (REPO_ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")

    for required in (
        "# What Changed",
        "# Why",
        "# Scope",
        "Name the touched boundaries",
        "# Verification",
        "## Refactor Skill Evidence",
        "## GPT-5.5 Review Evidence",
        "Refactor skill",
        "GPT-5.5 review",
        "Every new folder has a README or package docstring",
        "no proprietary scheduler or custom solver",
        "no-cache prefill baseline",
    ):
        assert required in text


def test_github_main_branch_protection_payload_requires_pr_review_and_ci():
    payload = json.loads((REPO_ROOT / ".github" / "main-branch-protection.json").read_text(encoding="utf-8"))
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    ci_job_names = re.findall(r"^    name: (.+)$", ci_text, flags=re.MULTILINE)

    assert ci_job_names == ["Test and build"]
    assert len(ci_job_names) == len(set(ci_job_names))
    assert payload["enforce_admins"] is True
    assert payload["required_linear_history"] is True
    assert payload["required_conversation_resolution"] is True
    assert payload["allow_force_pushes"] is False
    assert payload["allow_deletions"] is False
    assert payload["restrictions"] is None
    assert payload["required_status_checks"] == {
        "strict": True,
        "contexts": ["Test and build"],
    }
    assert payload["required_pull_request_reviews"] == {
        "dismiss_stale_reviews": True,
        "require_code_owner_reviews": False,
        "require_last_push_approval": True,
        "required_approving_review_count": 1,
    }


def test_github_docs_explain_branch_protection_application_and_plan_limit():
    text = (REPO_ROOT / ".github" / "README.md").read_text(encoding="utf-8")
    compact_text = " ".join(text.split())

    assert "`main-branch-protection.json`" in text
    assert "/branches/main/protection" in text
    assert "--data @.github/main-branch-protection.json" in text
    assert "Test and build" in text
    assert "direct pushes to `main` remain a process violation" in compact_text
    assert "private-repository branch protection" in text
    assert "squash or rebase merging enabled" in compact_text
    assert "auto-merge" in compact_text
    assert "delete head branches after merge" in compact_text or "merged PR branches are deleted" in compact_text


def test_github_ci_workflow_runs_pr_quality_gate():
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    for required in (
        "pull_request:",
        "push:",
        "branches:",
        "- main",
        "permissions:",
        "contents: read",
        "actions/checkout@v6",
        "actions/setup-python@v6",
        'python-version: "3.11"',
    ):
        assert required in text

    run_commands = [
        line.split("run:", maxsplit=1)[1].strip()
        for line in text.splitlines()
        if line.lstrip().startswith("run: ") and line.split("run:", maxsplit=1)[1].strip() != "|"
    ]
    assert run_commands == [
        "python -m pip install poetry==2.4.1",
        "poetry check",
        "poetry check --lock",
        "poetry install --dry-run",
        "poetry install --dry-run --extras databricks --extras test",
        "poetry install -E test",
        "poetry run pytest -q",
        "poetry build",
    ]


def test_github_ci_workflow_verifies_installed_console_scripts():
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "Verify console script entry points" in text
    assert 'tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]' in text
    assert "for script_name in sorted(scripts):" in text
    assert "shutil.which(script_name)" in text
    assert 'subprocess.run(' in text
    assert '[script_path, "--help"]' in text


def test_github_ci_workflow_smokes_built_wheel_imports():
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    workflow_readme = (REPO_ROOT / ".github" / "workflows" / "README.md").read_text(encoding="utf-8")
    compact_workflow_readme = " ".join(workflow_readme.split())

    assert "Verify PEP 517 wheel build" in text
    assert "python -m venv /tmp/cachet-pep517-wheel" in text
    assert "/tmp/cachet-pep517-wheel/bin/python -m pip wheel . --no-deps" in text
    assert 'Path("/tmp/cachet-pep517-wheelhouse").glob("cachet_kv-*.whl")' in text
    assert "Verify built wheel metadata" in text
    assert 'Path("dist").glob("cachet_kv-*.whl")' in text
    assert "def read_dist_info(archive, filename):" in text
    assert "assert len(roots) == 1" in text
    assert 'archive.read(f"{roots.pop()}/{filename}").decode("utf-8")' in text
    assert re.search(r"document_kv_cache-[^/\"']+\.dist-info/", text) is None
    assert "Requires-Python: >=3.11,<4.0" in text
    assert "Requires-Dist: packaging (==26.2)" in text
    assert "Tag: py3-none-any" in text
    assert "cachet-benchmark-plan=cachet.benchmark_plan:main" in text
    assert "cachet-pr-evidence=cachet.pr_evidence:main" in text
    assert "Verify built wheel import smoke" in text
    assert "python -m venv /tmp/cachet-wheel-smoke" in text
    assert "/tmp/cachet-wheel-smoke/bin/python -m pip install dist/cachet_kv-*.whl" in text
    assert "import cachet" in text
    assert "import document_kv_cache" in text
    assert 'legacy_package = "restaurant" "_kv_serving"' in text
    assert "find_spec(legacy_package) is None" in text
    assert "cachet.__all__ == document_kv_cache.__all__" in text
    assert 'not hasattr(cachet, "RestaurantKVRequest")' in text
    assert "assert cachet.storage is document_kv_cache.storage" in text
    assert "/tmp/cachet-wheel-smoke/bin/cachet-pr-evidence --help >/dev/null" in text
    assert "clean PEP 517 wheel build" in compact_workflow_readme
    assert "build with Cachet metadata" in compact_workflow_readme
    assert "verify the built wheel metadata" in compact_workflow_readme
    assert "built wheel into a fresh venv" in compact_workflow_readme
    assert "`cachet` and `document_kv_cache`" in compact_workflow_readme
    assert "`restaurant_kv_serving`" not in compact_workflow_readme


def test_gitignore_blocks_local_secrets_and_generated_artifacts():
    ignored = set((REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())

    for required in (
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
        ".coverage",
        ".ipynb_checkpoints/",
        "htmlcov/",
        ".ruff_cache/",
        ".hypothesis/",
        "dist/",
        "build/",
        "*.egg-info/",
    ):
        assert required in ignored


def test_gitignore_secret_patterns_work_with_git_check_ignore():
    candidate_paths = [
        ".env",
        ".env.local",
        ".envrc",
        "local.pem",
        "service.key",
        "openai.secret",
        "databricks.secrets",
        "logs/run.log",
        "tmp/output.tmp",
        ".coverage",
        ".ipynb_checkpoints/exploration-checkpoint.ipynb",
        "notebooks/.ipynb_checkpoints/databricks-checkpoint.ipynb",
        "htmlcov/index.html",
        ".ruff_cache/state",
        ".hypothesis/examples",
        "dist/package.whl",
        "build/temp.txt",
        "pkg.egg-info/PKG-INFO",
    ]
    completed = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(candidate_paths) + "\n",
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == candidate_paths


def test_repository_text_files_do_not_contain_committed_credentials():
    findings: list[str] = []

    for path in _iter_repository_text_files():
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        for secret_name, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                findings.append(f"{relative_path}:{line_number} matched {secret_name}")

    assert findings == []


def test_secret_patterns_match_representative_credential_shapes():
    samples = {
        "databricks_pat": "dapi" + ("a" * 32),
        "github_pat": "ghp_" + ("A" * 36),
        "langsmith_token": "lsv2_pt_" + ("a" * 24),
        "openai_api_key": "sk-" + "proj-" + ("A" * 28),
        "pem_private_key": "-----BEGIN " + "PRIVATE KEY-----",
    }

    for secret_name, sample in samples.items():
        assert SECRET_PATTERNS[secret_name].search(sample)


def _is_exact_stable_version_pin(version: str) -> bool:
    return STABLE_EXACT_VERSION_RE.fullmatch(version) is not None


def _dependency_version(spec: object) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict) and isinstance(spec.get("version"), str):
        return spec["version"]
    raise AssertionError(f"Dependency spec must include an exact version: {spec!r}")


def _iter_repository_text_files():
    for path in sorted(REPO_ROOT.rglob("*")):
        relative_parts = path.relative_to(REPO_ROOT).parts
        if any(part in GENERATED_OR_TOOLING_DIRS for part in relative_parts):
            continue
        if not path.is_file():
            continue
        if path.name in REPOSITORY_TEXT_FILE_NAMES or path.suffix in REPOSITORY_TEXT_FILE_SUFFIXES:
            yield path


def _requirement_name(requirement: str) -> str:
    return re.split(r"[\[<>=~!;,\s]", requirement, maxsplit=1)[0]


def _requirement_version(requirement: str) -> str:
    match = EXACT_REQUIREMENT_RE.fullmatch(requirement)
    if match is None:
        raise AssertionError(f"Requirement must use an exact == pin: {requirement!r}")
    return f"=={match.group(1)}"


def _collect_poetry_dependency_versions(pyproject: dict) -> dict[str, str]:
    dependency_versions = {}
    assert "dependency-groups" not in pyproject, "Use [tool.poetry.group.*.dependencies] for dependency groups"

    poetry_core_requirements = [
        requirement
        for requirement in pyproject["build-system"]["requires"]
        if _requirement_name(requirement) == "poetry-core"
    ]
    assert len(poetry_core_requirements) == 1
    dependency_versions["build-system.poetry-core"] = _requirement_version(poetry_core_requirements[0])

    poetry_config = pyproject["tool"]["poetry"]
    project_config = pyproject["project"]
    assert {"name", "version", "description", "authors"} <= set(project_config)
    deprecated_keys = DEPRECATED_TOOL_POETRY_METADATA_KEYS & set(poetry_config)
    assert deprecated_keys == set(), (
        f"Move deprecated [tool.poetry] keys to [project]: {sorted(deprecated_keys)}"
    )
    assert "dev-dependencies" not in poetry_config, (
        "[tool.poetry.dev-dependencies] is deprecated; use [tool.poetry.group.*.dependencies]"
    )

    for requirement in project_config.get("dependencies", ()):
        dependency_versions[f"project.dependencies.{_requirement_name(requirement)}"] = _requirement_version(requirement)
    for extra_name, requirements in project_config.get("optional-dependencies", {}).items():
        for requirement in requirements:
            dependency_versions[
                f"project.optional-dependencies.{extra_name}.{_requirement_name(requirement)}"
            ] = _requirement_version(requirement)

    dependency_tables = []
    dependency_tables.extend(
        (f"tool.poetry.group.{group_name}.dependencies", group_config.get("dependencies", {}))
        for group_name, group_config in poetry_config.get("group", {}).items()
    )
    for table_name, dependencies in dependency_tables:
        for name, spec in dependencies.items():
            if name == "python":
                continue
            dependency_versions[f"{table_name}.{name}"] = _dependency_version(spec)
    return dependency_versions


def test_exact_stable_version_pin_helper_rejects_ranges_and_prereleases():
    assert _is_exact_stable_version_pin("1.2.3")
    assert _is_exact_stable_version_pin("==1.2.3")
    assert _is_exact_stable_version_pin("1.2.3.post1")
    assert _is_exact_stable_version_pin("1!2.0.0")

    for version in ("^1.2.3", ">=1.2.3", "1.2.3rc1", "1.2.3.dev0", "1.2.3+local", "1.2.3, <2"):
        assert not _is_exact_stable_version_pin(version)


def test_poetry_dependency_collection_requires_build_backend_and_groups():
    pyproject = {
        "build-system": {"requires": ["poetry-core==2.4.1"]},
        "project": {
            "name": "example-package",
            "version": "1.2.3",
            "description": "Example package.",
            "authors": [{"name": "Example Team"}],
            "dependencies": ["requests==2.34.2"],
            "optional-dependencies": {
                "databricks": ["pyspark==4.1.2"],
            },
        },
        "tool": {
            "poetry": {
                "packages": [],
                "group": {
                    "dev": {
                        "dependencies": {
                            "ruff": "^0.8",
                        },
                    },
                },
            },
        },
    }

    assert _collect_poetry_dependency_versions(pyproject) == {
        "build-system.poetry-core": "==2.4.1",
        "project.dependencies.requests": "==2.34.2",
        "project.optional-dependencies.databricks.pyspark": "==4.1.2",
        "tool.poetry.group.dev.dependencies.ruff": "^0.8",
    }

    pyproject["build-system"]["requires"] = []

    with pytest.raises(AssertionError):
        _collect_poetry_dependency_versions(pyproject)

    pyproject["build-system"]["requires"] = ["poetry-core==2.4.1"]
    pyproject["tool"]["poetry"]["dev-dependencies"] = {"ruff": "^0.8"}

    with pytest.raises(AssertionError):
        _collect_poetry_dependency_versions(pyproject)

    del pyproject["tool"]["poetry"]["dev-dependencies"]
    pyproject["project"]["dependencies"] = ["requests>=2"]

    with pytest.raises(AssertionError):
        _collect_poetry_dependency_versions(pyproject)

    pyproject["project"]["dependencies"] = ["requests==2.34.2"]
    pyproject["project"]["optional-dependencies"] = {"dev": ["ruff>=0.8"]}

    with pytest.raises(AssertionError):
        _collect_poetry_dependency_versions(pyproject)

    pyproject["project"]["optional-dependencies"] = {"dev": ["ruff==0.8.0"]}
    pyproject["tool"]["poetry"]["scripts"] = {"old-script": "old.module:main"}

    with pytest.raises(AssertionError):
        _collect_poetry_dependency_versions(pyproject)

    del pyproject["tool"]["poetry"]["scripts"]
    pyproject["tool"]["poetry"]["version"] = "9.9.9"

    with pytest.raises(AssertionError):
        _collect_poetry_dependency_versions(pyproject)

    del pyproject["tool"]["poetry"]["version"]
    pyproject["dependency-groups"] = {"dev": ["ruff>=0.8"]}

    with pytest.raises(AssertionError):
        _collect_poetry_dependency_versions(pyproject)


def test_poetry_dependencies_use_exact_direct_pins():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    dependency_versions = _collect_poetry_dependency_versions(pyproject)
    assert dependency_versions
    for name, version in dependency_versions.items():
        assert _is_exact_stable_version_pin(version), f"{name} is not exactly pinned to a stable version: {version}"


def test_poetry_lockfile_tracks_direct_runtime_pins():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock_path = REPO_ROOT / "poetry.lock"

    assert lock_path.is_file()
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["metadata"]["lock-version"] == "2.1"
    assert lock["metadata"]["python-versions"] == pyproject["project"]["requires-python"]

    locked_versions = {package["name"]: f"=={package['version']}" for package in lock["package"]}
    direct_requirements = list(pyproject["project"].get("dependencies", ()))
    for requirements in pyproject["project"].get("optional-dependencies", {}).values():
        direct_requirements.extend(requirements)

    assert direct_requirements
    for requirement in direct_requirements:
        requirement_name = _requirement_name(requirement)
        assert locked_versions[requirement_name] == _requirement_version(requirement)

    assert lock["extras"] == {
        "databricks": ["databricks-sdk", "pyspark"],
        "test": ["pytest"],
    }


def test_runtime_packaging_pin_stays_databricks_ml_compatible():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependency_versions = _collect_poetry_dependency_versions(pyproject)

    assert dependency_versions["project.dependencies.packaging"] == "==26.2"


def test_poetry_metadata_keeps_conflicting_serving_engines_out_of_core_resolver():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"].get("dependencies", ())
    optional_dependencies = pyproject["project"].get("optional-dependencies", {})
    dependency_names = {_requirement_name(requirement) for requirement in dependencies}
    optional_dependency_names_by_extra = {
        extra_name: {_requirement_name(requirement) for requirement in requirements}
        for extra_name, requirements in optional_dependencies.items()
    }
    optional_dependency_names = set().union(*optional_dependency_names_by_extra.values())

    assert "vllm" not in dependency_names
    assert "sglang" not in dependency_names
    assert "vllm" not in optional_dependencies
    assert "sglang" not in optional_dependencies
    assert "serving" not in optional_dependencies
    assert "vllm-kv-injection" not in optional_dependency_names
    assert "sglang-kv-injection" not in optional_dependency_names


def test_public_and_adapter_packages_publish_pep561_markers():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = {
        include["path"]
        for include in pyproject["tool"]["poetry"]["include"]
        if include.get("format") == ["sdist", "wheel"]
    }
    marker_paths = (
        "src/cachet/py.typed",
        "src/document_kv_cache/py.typed",
        "src/vllm_kv_injection/py.typed",
        "src/sglang_kv_injection/py.typed",
    )

    for marker_path in marker_paths:
        assert (REPO_ROOT / marker_path).is_file()
        assert marker_path in includes
    assert (REPO_ROOT / "src" / "restaurant_kv_serving" / "py.typed").is_file()
    assert "src/restaurant_kv_serving/py.typed" not in includes
