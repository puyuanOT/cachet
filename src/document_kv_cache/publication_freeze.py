"""Deterministic source-freeze and local qualification-preflight tooling.

This module closes the two controller-side trust boundaries that precede the
vLLM 0.27.1 publication campaign.  It never submits a Databricks run and never
opens or mutates the GPU-hour ledger.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import site
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol, cast

from document_kv_cache.gpu_qualification import (
    GPUQualificationArtifactPins,
    build_local_preflight_evidence,
    validate_gpu_qualification_plan_record,
    validate_local_preflight_evidence_record,
)
from document_kv_cache.gpu_qualification_databricks import (
    GPU_QUALIFICATION_ARTIFACT_KEYS,
    GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT,
    GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256,
    GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES,
    _qualification_single_user_name_from_payloads,
    _verify_input_bundle_byte_closure,
    pins_from_plan_record,
    render_gpu_qualification_submit_payloads,
)
from document_kv_cache.databricks_runs import (
    DatabricksWorkspaceConfig,
    list_databricks_volume_directory,
    require_databricks_current_user_name,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_ID,
    validate_publication_campaign_plan_record,
)
from document_kv_cache.serving_env import (
    VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_LOCK_FILENAME,
    VLLM_RUNTIME_LOCK_SHA256,
)
from document_kv_cache.vllm_wheel_repack import (
    VLLM_PATCHED_WHEEL_MANIFEST_RECORD_TYPE,
    VLLM_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION,
)


PUBLICATION_SOURCE_CLOSURE_RECORD_TYPE: Final = (
    "cachet.publication_source_closure.v1"
)
PUBLICATION_SOURCE_CLOSURE_SCHEMA_VERSION: Final = 1
GPU_QUALIFICATION_LOCAL_CHECK_RECORD_TYPE: Final = (
    "cachet.gpu_qualification.local_check_evidence.v1"
)
GPU_QUALIFICATION_LOCAL_CHECK_SCHEMA_VERSION: Final = 1
PUBLICATION_FREEZE_BUILD_FRONTEND: Final = "build==1.2.2.post1"
PUBLICATION_FREEZE_BUILD_BACKEND: Final = "poetry-core==2.4.1"
PUBLICATION_FREEZE_PYTHON: Final = "CPython 3.11.16"
PUBLICATION_FREEZE_VLLM_VERSION: Final = "0.27.1+cu129"
PUBLICATION_FREEZE_OFFICIAL_VLLM_WHEEL_SHA256: Final = (
    "bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b"
)
PUBLICATION_FREEZE_PATCHED_VLLM_WHEEL_SHA256: Final = (
    "65120c48a9352b9eb65bab7a67090558d27af985ad366e469d3b87751073cff4"
)
PUBLICATION_FREEZE_PATCH_MANIFEST_SHA256: Final = (
    "14611e163e720f0fdeae6ef2704cecd9202eef6adc6336f892afd94a96726ef6"
)
PUBLICATION_FREEZE_RUNTIME_LOCK_INPUT_SHA256: Final = (
    "7be991a6cc37ebc84d76a9080e00d79206998eea652ec7d2413be59c19399019"
)
PUBLICATION_FREEZE_INPUT_BUNDLE_SHA256: Final = (
    "7ff6cf6a1553c0e844853d21de9780c75211f1be8304754da72e9cbebbd164ec"
)
PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_CLOSED_RECORD_SHA256: Final = (
    "404d0ed6ae2f169d1777034c81a057e2af131d805ecd9672900bfc7221871246"
)
PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_FILE_SHA256: Final = (
    "73f069467ab9c4e7071532b76efe6e2979e2aee7e7b05659c97d7c530a5a8dee"
)
PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_ID: Final = (
    "vllm-0271-publication-latency-handoffs-v1"
)
PUBLICATION_FREEZE_LATENCY_HANDOFF_WORKERS_SHA256: Final = (
    "d36983604b4de91446635d47918fe914b98872623615066aa335b8cb1ba31663"
)
PUBLICATION_FREEZE_MAX_ARTIFACT_BYTES: Final = 16 * 1024 * 1024
PUBLICATION_FREEZE_MAX_JSON_BYTES: Final = 16 * 1024 * 1024
PUBLICATION_FREEZE_MAX_CHECK_RECORD_BYTES: Final = 2 * 1024 * 1024
PUBLICATION_FREEZE_MAX_COMMAND_OUTPUT_BYTES: Final = 1024 * 1024
PUBLICATION_FREEZE_COMMAND_TAIL_BYTES: Final = 8 * 1024
PUBLICATION_FREEZE_COMMAND_TIMEOUT_SECONDS: Final = 30 * 60
PUBLICATION_FREEZE_EXPECTED_TEST_COUNT: Final = 3_334
_DEFAULT_PYTHON_EXECUTABLE: Final = Path(
    "/opt/homebrew/opt/python@3.11/bin/python3.11"
).resolve()
_DEFAULT_RUFF_EXECUTABLE: Final = Path(".venv/bin/ruff")
_DEFAULT_MYPY_EXECUTABLE: Final = Path(".venv/bin/mypy")
_GIT_EXECUTABLE: Final = Path("/usr/bin/git")
_GZIP_EXECUTABLE: Final = Path("/usr/bin/gzip")
_LATENCY_SEMANTIC_USER_HOME: Final = Path.home()
_LATENCY_SEMANTIC_UV_EXECUTABLE: Final = (
    _LATENCY_SEMANTIC_USER_HOME / ".local/bin/uv"
)
_LATENCY_SEMANTIC_UV_SHA256: Final = (
    "94151d6624054c3973829c82eb718db1afc55ef9fcee499cdd94bfb852fb99f9"
)
_LATENCY_SEMANTIC_UV_VERSION: Final = (
    "uv 0.11.6 (65950801c 2026-04-09 aarch64-apple-darwin)"
)
_LATENCY_SEMANTIC_UV_CACHE: Final = (
    _LATENCY_SEMANTIC_USER_HOME / ".cache/uv"
)
_LATENCY_SEMANTIC_HF_HUB_CACHE: Final = (
    _LATENCY_SEMANTIC_USER_HOME / ".cache/huggingface/hub"
)
_LATENCY_SEMANTIC_PYTHON_SHA256: Final = (
    "3494e1ea84d8b49a17c193a9825d34abd476b7f3c2075e58c14316696fe3bf6f"
)
_LATENCY_SEMANTIC_TOKENIZER_SNAPSHOT_SHA256: Final = (
    "21fbe7df7aab4932076e2eedc9ba9b1be4cfe7be2b742f05a01b2125d9186557"
)
_LATENCY_SEMANTIC_RUNTIME_DISTRIBUTIONS_SHA256: Final = (
    "87d25293be5ce9b36b30b085fbea6db17f5a074f6f18b340f954a982e2bc3d7a"
)
_LATENCY_SEMANTIC_RUNTIME_LOCK_RELATIVE_PATH: Final = Path(
    "src/document_kv_cache/runtime_locks/"
    "publication-latency-semantic-py311-macos-arm64.lock"
)
_LATENCY_SEMANTIC_RUNTIME_LOCK_SHA256: Final = (
    "8e26c54c74af9af63c5425e97581f3f9d1ecee00b28c7f151a607f673c14ccbb"
)
_LATENCY_SEMANTIC_RUNTIME_LOCK_BYTE_COUNT: Final = 30_963
_LATENCY_SEMANTIC_SITE_PACKAGES_SHA256: Final = (
    "26e45807542ce3fc12aac2f0d162db6268b8e964e389097b867af909ea78e780"
)
_LATENCY_SEMANTIC_SITE_PACKAGES_FILE_COUNT: Final = 4_753
_LATENCY_SEMANTIC_TOKENIZER_ID: Final = "Qwen/Qwen3-4B-Instruct-2507"
_LATENCY_SEMANTIC_TOKENIZER_REVISION: Final = (
    "cdbee75f17c01a7cc42f958dc650907174af0554"
)
_LATENCY_SEMANTIC_TOKENIZER_CLASS_MODULE: Final = (
    "transformers.models.qwen2.tokenization_qwen2"
)
_LATENCY_SEMANTIC_TOKENIZER_CLASS_NAME: Final = "Qwen2Tokenizer"
_LATENCY_SEMANTIC_TOKENIZER_VOCAB_SIZE: Final = 151_643
_LATENCY_SEMANTIC_ATTESTATION_RECORD_TYPE: Final = (
    "cachet.publication_latency_plan_semantic_validation.v1"
)
_LATENCY_SEMANTIC_PREPARED_INPUT_RELATIVE_PATH: Final = Path(
    "databricks-runs/vllm-0271-publication-prep/"
    "prepared-v3-sha256-"
    f"{PUBLICATION_FREEZE_INPUT_BUNDLE_SHA256}"
)
_FREEZE_HOME: Final = Path("/private/var/empty")
_FREEZE_TMPDIR: Final = Path("/private/var/tmp")
_FREEZE_PATH: Final = (
    "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
)

_SOURCE_FILE_ROLES: Final = (
    "cachet_package_wheel",
    "cachet_source_distribution",
    "git_source_archive",
    "gpu_qualification_bootstrap",
)
_SOURCE_REFERENCE_ROLES: Final = (
    "runtime_lock",
    "runtime_lock_input",
    "campaign_plan",
    "latency_handoff_plan",
    "full_score_inventory",
    "full_score_shard_plan",
)
_LATENCY_SEMANTIC_RUNTIME_DISTRIBUTIONS: Final = (
    ("annotated-doc", "0.0.5"),
    ("anyio", "4.14.2"),
    ("certifi", "2026.7.22"),
    ("click", "8.4.2"),
    ("filelock", "3.32.4"),
    ("fsspec", "2026.7.0"),
    ("h11", "0.16.0"),
    ("hf-xet", "1.6.0"),
    ("httpcore", "1.0.9"),
    ("httpx", "0.28.1"),
    ("huggingface-hub", "1.28.0"),
    ("idna", "3.19"),
    ("markdown-it-py", "4.2.0"),
    ("mdurl", "0.1.2"),
    ("numpy", "2.4.6"),
    ("packaging", "26.2"),
    ("pygments", "2.21.0"),
    ("pyyaml", "6.0.3"),
    ("regex", "2026.7.19"),
    ("rich", "15.0.0"),
    ("safetensors", "0.8.0"),
    ("shellingham", "1.5.4"),
    ("tokenizers", "0.22.2"),
    ("tqdm", "4.70.0"),
    ("transformers", "5.12.1"),
    ("typer", "0.27.1"),
    ("typing-extensions", "4.16.0"),
)
_LATENCY_SEMANTIC_TOKENIZER_SNAPSHOT_FILES: Final = (
    (
        "config.json",
        "../../blobs/6988f134db143052042f2bd6e0c897bc6a605189",
        727,
        "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba",
    ),
    (
        "merges.txt",
        "../../blobs/20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0",
        1_671_839,
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
    (
        "tokenizer.json",
        "../../blobs/aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        11_422_654,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    (
        "tokenizer_config.json",
        "../../blobs/51c1be0d9192e7f6e6596de71d0f07d58fbc32ac",
        9_377,
        "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3",
    ),
    (
        "vocab.json",
        "../../blobs/4783fe10ac3adce15ac8f358ef5462739852c569",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
)
_LOCAL_CHECK_IDS: Final = (
    "canonical_plan_schema",
    "runtime_lock_require_hashes",
    "patched_wheel_record_and_manifest",
    "source_runner_input_closure",
    "unit_tests",
    "ruff",
    "mypy",
)
_STATIC_ANALYSIS_TARGETS: Final = (
    "src/document_kv_cache/databricks_resource_ledger.py",
    "src/document_kv_cache/gpu_qualification.py",
    "src/document_kv_cache/gpu_qualification_databricks.py",
    "src/document_kv_cache/publication_campaign.py",
    "src/document_kv_cache/publication_latency_handoff_generation.py",
    "src/document_kv_cache/publication_bf16_handoff_generation.py",
    "src/document_kv_cache/publication_handoff_closure_coordinator.py",
    "src/document_kv_cache/publication_latency_execution.py",
    "src/document_kv_cache/full_score_execution.py",
    "src/document_kv_cache/full_score_remote_control.py",
    "src/document_kv_cache/publication_freeze.py",
)
_STATIC_ANALYSIS_EXCLUDED_TARGETS: Final = (
    "src/document_kv_cache/databricks_runs.py",
)
_RUFF_TARGETS: Final = _STATIC_ANALYSIS_TARGETS
_MYPY_TARGETS: Final = _STATIC_ANALYSIS_TARGETS
_COMMAND_CHECK_IDS: Final = ("unit_tests", "ruff", "mypy")
_PYTHON_CHECK_IDS: Final = _LOCAL_CHECK_IDS[:4]
_CHECK_INPUT_LABELS: Final = {
    "canonical_plan_schema": (
        "plan_json",
        "artifact_uris_json",
        "submit_payloads_json",
    ),
    "runtime_lock_require_hashes": (
        "runtime_lock",
        "runtime_lock_input",
        "package_wheel",
    ),
    "patched_wheel_record_and_manifest": (
        "official_vllm_wheel",
        "patched_vllm_wheel",
        "patched_vllm_manifest",
    ),
    "source_runner_input_closure": (
        "source_closure_json",
        "source_artifact_root",
        "package_wheel",
        "runner",
        "input_bundle",
    ),
    "unit_tests": ("repository_root",),
    "ruff": ("repository_root",),
    "mypy": ("repository_root",),
}
_SOURCE_KEYS: Final = frozenset(
    {
        "build",
        "campaign_id",
        "closed_record_sha256",
        "files",
        "git",
        "record_type",
        "references",
        "runtime",
        "schema_version",
    }
)
_SOURCE_BUILD_KEYS: Final = frozenset(
    {"build_backend", "build_frontend", "python", "source_date_epoch"}
)
_SOURCE_GIT_KEYS: Final = frozenset(
    {"branch", "commit", "commit_tree", "dirty"}
)
_SOURCE_FILE_KEYS: Final = frozenset(
    {"byte_count", "relative_path", "role", "sha256"}
)
_SOURCE_REFERENCE_KEYS: Final = frozenset(
    {"byte_count", "path", "role", "sha256"}
)
_SOURCE_RUNTIME_KEYS: Final = frozenset(
    {
        "official_vllm_wheel_sha256",
        "patched_vllm_wheel_sha256",
        "runtime_lock_sha256",
        "vllm_version",
    }
)
_CHECK_KEYS: Final = frozenset(
    {
        "check_id",
        "checked_at_utc",
        "command",
        "exit_code",
        "environment",
        "inputs",
        "plan_sha256",
        "record_type",
        "result",
        "schema_version",
        "status",
        "stderr",
        "stdout",
        "tool_identity",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class PublicationSourceClosureInputs:
    """Clean source tree, fresh artifact root, and closed references."""

    repository_root: Path
    artifact_output_root: Path
    runtime_lock: Path
    runtime_lock_input: Path
    campaign_plan: Path
    latency_handoff_plan: Path
    full_score_inventory: Path
    full_score_shard_plan: Path


@dataclass(frozen=True, slots=True)
class GPUQualificationLocalPreflightInputs:
    """Closed local artifacts consumed by all seven preflight checks."""

    repository_root: Path
    plan_json: Path
    artifact_uris_json: Path
    submit_payloads_json: Path
    source_closure_json: Path
    source_artifact_root: Path
    package_wheel: Path
    runner: Path
    input_bundle: Path
    runtime_lock: Path
    runtime_lock_input: Path
    official_vllm_wheel: Path
    patched_vllm_wheel: Path
    patched_vllm_manifest: Path
    python_executable: Path = _DEFAULT_PYTHON_EXECUTABLE
    ruff_executable: Path = _DEFAULT_RUFF_EXECUTABLE
    mypy_executable: Path = _DEFAULT_MYPY_EXECUTABLE


class CompletedCommand(Protocol):
    """The bounded subset of :class:`subprocess.CompletedProcess` we consume."""

    returncode: int
    stdout: bytes
    stderr: bytes


CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str]], CompletedCommand]


@dataclass(frozen=True, slots=True)
class _PackageBuildOutputs:
    wheel_name: str
    wheel_bytes: bytes
    sdist_name: str
    sdist_bytes: bytes


@dataclass(frozen=True, slots=True)
class _VerifiedTokenizerSnapshot:
    sha256: str
    files: tuple[tuple[str, str, bytes], ...]


def build_publication_source_closure(
    inputs: PublicationSourceClosureInputs,
) -> dict[str, Any]:
    """Build and fully validate one deterministic publication source closure."""

    if not isinstance(inputs, PublicationSourceClosureInputs):
        raise TypeError("inputs must be PublicationSourceClosureInputs")
    root = _regular_directory(inputs.repository_root, "repository_root")
    git = _git_identity(root)
    _require_freeze_toolchain()
    _require_freeze_build_system(root)
    _validate_publication_latency_handoff_reference(
        inputs.latency_handoff_plan,
        repository_root=root,
    )
    artifact_root = _create_directory_exclusive(
        inputs.artifact_output_root,
        "source artifact output root",
    )
    first_build, second_build = _build_package_twice(
        root,
        commit=cast(str, git["commit"]),
        source_date_epoch=cast(int, git["source_date_epoch"]),
    )
    _require_matching_build_outputs(first_build, second_build)
    package_wheel = artifact_root / first_build.wheel_name
    source_distribution = artifact_root / first_build.sdist_name
    _write_exclusive(package_wheel, first_build.wheel_bytes, "package wheel")
    _write_exclusive(
        source_distribution,
        first_build.sdist_bytes,
        "source distribution",
    )
    git_source_archive = artifact_root / f"cachet-{git['commit']}.tar.gz"
    _write_deterministic_git_archive(
        root,
        commit=cast(str, git["commit"]),
        output_path=git_source_archive,
    )
    bootstrap_runner = artifact_root / "gpu-qualification-bootstrap.py"
    _write_exclusive(
        bootstrap_runner,
        GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT.encode("utf-8"),
        "GPU qualification bootstrap",
    )
    files = (
        _source_file_record(package_wheel, "cachet_package_wheel"),
        _source_file_record(
            source_distribution, "cachet_source_distribution"
        ),
        _source_file_record(git_source_archive, "git_source_archive"),
        _source_file_record(
            bootstrap_runner, "gpu_qualification_bootstrap"
        ),
    )
    references = tuple(
        _source_reference_record(root, path, role)
        for path, role in (
            (inputs.runtime_lock, "runtime_lock"),
            (inputs.runtime_lock_input, "runtime_lock_input"),
            (inputs.campaign_plan, "campaign_plan"),
            (inputs.latency_handoff_plan, "latency_handoff_plan"),
            (inputs.full_score_inventory, "full_score_inventory"),
            (inputs.full_score_shard_plan, "full_score_shard_plan"),
        )
    )
    record: dict[str, Any] = {
        "build": {
            "build_backend": PUBLICATION_FREEZE_BUILD_BACKEND,
            "build_frontend": PUBLICATION_FREEZE_BUILD_FRONTEND,
            "python": PUBLICATION_FREEZE_PYTHON,
            "source_date_epoch": git["source_date_epoch"],
        },
        "campaign_id": PUBLICATION_CAMPAIGN_ID,
        "closed_record_sha256": "",
        "files": list(files),
        "git": {
            "branch": git["branch"],
            "commit": git["commit"],
            "commit_tree": git["commit_tree"],
            "dirty": False,
        },
        "record_type": PUBLICATION_SOURCE_CLOSURE_RECORD_TYPE,
        "references": list(references),
        "runtime": {
            "official_vllm_wheel_sha256": (
                PUBLICATION_FREEZE_OFFICIAL_VLLM_WHEEL_SHA256
            ),
            "patched_vllm_wheel_sha256": (
                PUBLICATION_FREEZE_PATCHED_VLLM_WHEEL_SHA256
            ),
            "runtime_lock_sha256": VLLM_RUNTIME_LOCK_SHA256,
            "vllm_version": PUBLICATION_FREEZE_VLLM_VERSION,
        },
        "schema_version": PUBLICATION_SOURCE_CLOSURE_SCHEMA_VERSION,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    _validate_publication_source_closure_record(
        record,
        repository_root=root,
        artifact_root=artifact_root,
        explicit_artifact_paths=None,
        verify_rebuild=False,
    )
    return record


def validate_publication_source_closure_record(
    record: Mapping[str, Any],
    *,
    repository_root: str | Path,
    artifact_root: str | Path,
    explicit_artifact_paths: Mapping[str, str | Path] | None = None,
) -> None:
    """Fail closed unless *record* and all referenced bytes match the freeze."""

    _validate_publication_source_closure_record(
        record,
        repository_root=repository_root,
        artifact_root=artifact_root,
        explicit_artifact_paths=explicit_artifact_paths,
        verify_rebuild=True,
    )


def _validate_publication_source_closure_record(
    record: Mapping[str, Any],
    *,
    repository_root: str | Path,
    artifact_root: str | Path,
    explicit_artifact_paths: Mapping[str, str | Path] | None,
    verify_rebuild: bool,
) -> None:

    normalized = _mapping_copy(record, "source closure")
    _require_exact_keys(normalized, _SOURCE_KEYS, "source closure")
    if normalized["record_type"] != PUBLICATION_SOURCE_CLOSURE_RECORD_TYPE:
        raise ValueError("source closure record_type differs")
    if normalized["schema_version"] != PUBLICATION_SOURCE_CLOSURE_SCHEMA_VERSION:
        raise ValueError("source closure schema_version differs")
    if normalized["campaign_id"] != PUBLICATION_CAMPAIGN_ID:
        raise ValueError("source closure campaign_id differs")
    _require_closed_digest(normalized, "source closure")
    root = _regular_directory(Path(repository_root), "repository_root")
    artifacts = _regular_directory(Path(artifact_root), "artifact_root")
    _require_freeze_toolchain()
    _require_freeze_build_system(root)
    build = _required_mapping(normalized, "build")
    _require_exact_keys(build, _SOURCE_BUILD_KEYS, "source closure build")
    expected_git = _git_identity(root)
    expected_build = {
        "build_backend": PUBLICATION_FREEZE_BUILD_BACKEND,
        "build_frontend": PUBLICATION_FREEZE_BUILD_FRONTEND,
        "python": PUBLICATION_FREEZE_PYTHON,
        "source_date_epoch": expected_git["source_date_epoch"],
    }
    if dict(build) != expected_build:
        raise ValueError("source closure build identity differs")
    git = _required_mapping(normalized, "git")
    _require_exact_keys(git, _SOURCE_GIT_KEYS, "source closure git")
    if dict(git) != {
        "branch": expected_git["branch"],
        "commit": expected_git["commit"],
        "commit_tree": expected_git["commit_tree"],
        "dirty": False,
    }:
        raise ValueError("source closure git identity differs")
    files = _mapping_sequence(normalized.get("files"), "source closure files")
    if tuple(item.get("role") for item in files) != _SOURCE_FILE_ROLES:
        raise ValueError("source closure file roles differ")
    observed_paths: set[str] = set()
    resolved_files: dict[str, Path] = {}
    explicit = dict(explicit_artifact_paths or {})
    for item in files:
        _require_exact_keys(item, _SOURCE_FILE_KEYS, "source closure file")
        relative = _safe_relative_path(item.get("relative_path"), "relative_path")
        if relative in observed_paths:
            raise ValueError("source closure repeats a file path")
        observed_paths.add(relative)
        candidate = (
            Path(explicit[relative]) if relative in explicit else artifacts / relative
        )
        candidate = _regular_file(candidate, f"source closure file {relative}")
        _require_file_binding(candidate, item, f"source closure file {relative}")
        resolved_files[cast(str, item["role"])] = candidate
    _validate_cachet_wheel(resolved_files["cachet_package_wheel"])
    _validate_sdist(resolved_files["cachet_source_distribution"])
    _validate_git_archive(
        resolved_files["git_source_archive"],
        repository_root=root,
        commit=cast(str, git["commit"]),
        source_date_epoch=cast(int, build["source_date_epoch"]),
    )
    if _file_sha256(resolved_files["gpu_qualification_bootstrap"]) != (
        GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256
    ):
        raise ValueError("source closure bootstrap runner differs")
    references = _mapping_sequence(
        normalized.get("references"), "source closure references"
    )
    if tuple(item.get("role") for item in references) != _SOURCE_REFERENCE_ROLES:
        raise ValueError("source closure reference roles differ")
    reference_paths: dict[str, Path] = {}
    for item in references:
        _require_exact_keys(item, _SOURCE_REFERENCE_KEYS, "source closure reference")
        relative = _safe_relative_path(item.get("path"), "reference path")
        candidate = _regular_file(root / relative, f"source reference {relative}")
        _require_file_binding(candidate, item, f"source reference {relative}")
        reference_paths[cast(str, item["role"])] = candidate
    if _file_sha256(reference_paths["runtime_lock"]) != VLLM_RUNTIME_LOCK_SHA256:
        raise ValueError("source closure runtime lock differs")
    if _file_sha256(reference_paths["runtime_lock_input"]) != (
        PUBLICATION_FREEZE_RUNTIME_LOCK_INPUT_SHA256
    ):
        raise ValueError("source closure runtime lock input differs")
    campaign = _read_canonical_json(
        reference_paths["campaign_plan"], pretty=True, label="campaign plan"
    )
    validate_publication_campaign_plan_record(campaign)
    _validate_publication_latency_handoff_reference(
        reference_paths["latency_handoff_plan"],
        repository_root=root,
    )
    runtime = _required_mapping(normalized, "runtime")
    _require_exact_keys(runtime, _SOURCE_RUNTIME_KEYS, "source closure runtime")
    if dict(runtime) != {
        "official_vllm_wheel_sha256": (
            PUBLICATION_FREEZE_OFFICIAL_VLLM_WHEEL_SHA256
        ),
        "patched_vllm_wheel_sha256": (
            PUBLICATION_FREEZE_PATCHED_VLLM_WHEEL_SHA256
        ),
        "runtime_lock_sha256": VLLM_RUNTIME_LOCK_SHA256,
        "vllm_version": PUBLICATION_FREEZE_VLLM_VERSION,
    }:
        raise ValueError("source closure runtime identity differs")
    if verify_rebuild:
        first_build, second_build = _build_package_twice(
            root,
            commit=cast(str, git["commit"]),
            source_date_epoch=cast(int, build["source_date_epoch"]),
        )
        _require_matching_build_outputs(first_build, second_build)
        _require_file_bytes(
            resolved_files["cachet_package_wheel"],
            first_build.wheel_bytes,
            "source closure package wheel versus clean-tree rebuild",
        )
        _require_file_bytes(
            resolved_files["cachet_source_distribution"],
            first_build.sdist_bytes,
            "source closure source distribution versus clean-tree rebuild",
        )


def _validate_publication_latency_handoff_reference(
    plan_path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Bind the retained latency plan to current exact-token semantics."""

    plan = _validated_frozen_latency_handoff_plan(plan_path)
    prepared_input_dir = _regular_directory(
        repository_root / _LATENCY_SEMANTIC_PREPARED_INPUT_RELATIVE_PATH,
        "latency semantic prepared input bundle",
    )
    observed = _run_publication_latency_semantic_subprocess(
        plan_path=plan_path,
        prepared_input_dir=prepared_input_dir,
        repository_root=repository_root,
    )
    expected = _publication_latency_semantic_attestation(plan)
    if observed != expected:
        raise ValueError(
            "source closure latency handoff semantic attestation differs"
        )
    return observed


def _validated_frozen_latency_handoff_plan(plan_path: Path) -> dict[str, Any]:
    candidate = _regular_file(plan_path, "latency handoff plan")
    plan = _read_canonical_json(
        candidate,
        pretty=True,
        label="latency handoff plan",
    )
    plan_digest = _required_sha256(
        plan.get("closed_record_sha256"),
        "latency handoff plan digest",
    )
    plan_payload = dict(plan)
    plan_payload.pop("closed_record_sha256", None)
    if plan_digest != hashlib.sha256(
        json.dumps(
            plan_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest():
        raise ValueError("latency handoff plan closed_record_sha256 is invalid")
    workers = _mapping_sequence(plan.get("workers"), "latency handoff workers")
    items = tuple(
        item
        for worker in workers
        for item in _mapping_sequence(
            worker.get("items"),
            "latency handoff worker items",
        )
    )
    segment_contract_count = sum(
        "segment_token_contracts" in item
        and "segment_token_contracts_sha256" in item
        for item in items
    )
    if len(items) != 384 or segment_contract_count != len(items):
        raise ValueError(
            "source closure latency handoff plan is semantically stale: "
            "segment_token_contracts coverage differs"
        )
    sharding = _required_mapping(plan, "sharding")
    coverage = _required_mapping(plan, "coverage")
    if (
        _file_sha256(candidate)
        != PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_FILE_SHA256
        or plan.get("closed_record_sha256")
        != PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_CLOSED_RECORD_SHA256
        or plan.get("plan_id") != PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_ID
        or plan.get("input_bundle_sha256")
        != PUBLICATION_FREEZE_INPUT_BUNDLE_SHA256
        or plan.get("workers_sha256")
        != PUBLICATION_FREEZE_LATENCY_HANDOFF_WORKERS_SHA256
        or sharding.get("worker_count") != 16
        or coverage.get("task_count") != 384
        or coverage.get("cache_prefix_generation_tokens") != 7_323_967
    ):
        raise ValueError(
            "source closure latency handoff plan differs from the frozen "
            "semantic plan"
        )
    return plan


def _publication_latency_semantic_attestation(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "input_bundle_sha256": PUBLICATION_FREEZE_INPUT_BUNDLE_SHA256,
        "plan_closed_record_sha256": (
            PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_CLOSED_RECORD_SHA256
        ),
        "plan_file_sha256": PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_FILE_SHA256,
        "python": PUBLICATION_FREEZE_PYTHON,
        "record_type": _LATENCY_SEMANTIC_ATTESTATION_RECORD_TYPE,
        "runtime_distributions_sha256": (
            _LATENCY_SEMANTIC_RUNTIME_DISTRIBUTIONS_SHA256
        ),
        "runtime_lock_sha256": _LATENCY_SEMANTIC_RUNTIME_LOCK_SHA256,
        "runtime_site_packages_sha256": (
            _LATENCY_SEMANTIC_SITE_PACKAGES_SHA256
        ),
        "schema_version": 1,
        "task_count": _required_mapping(plan, "coverage").get("task_count"),
        "tokenizer_id": _LATENCY_SEMANTIC_TOKENIZER_ID,
        "tokenizer_revision": _LATENCY_SEMANTIC_TOKENIZER_REVISION,
        "tokenizer_snapshot_sha256": (
            _LATENCY_SEMANTIC_TOKENIZER_SNAPSHOT_SHA256
        ),
        "validated": True,
        "worker_count": _required_mapping(plan, "sharding").get("worker_count"),
        "workers_sha256": PUBLICATION_FREEZE_LATENCY_HANDOFF_WORKERS_SHA256,
    }


def _run_publication_latency_semantic_subprocess(
    *,
    plan_path: Path,
    prepared_input_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    uv = _regular_file(
        _LATENCY_SEMANTIC_UV_EXECUTABLE,
        "latency semantic uv executable",
    )
    python = _regular_file(
        _DEFAULT_PYTHON_EXECUTABLE,
        "latency semantic Python executable",
    )
    if (
        _file_sha256(uv) != _LATENCY_SEMANTIC_UV_SHA256
        or _file_sha256(python) != _LATENCY_SEMANTIC_PYTHON_SHA256
    ):
        raise RuntimeError("latency semantic executable identity differs")
    lock_bytes = _verified_publication_latency_runtime_lock(
        repository_root / _LATENCY_SEMANTIC_RUNTIME_LOCK_RELATIVE_PATH
    )
    environment = _latency_semantic_environment(repository_root)
    version = subprocess.run(
        (str(uv), "--version"),
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        timeout=30,
    )
    _require_bounded_bytes(version.stdout, "latency semantic uv version stdout")
    _require_bounded_bytes(version.stderr, "latency semantic uv version stderr")
    if (
        version.returncode != 0
        or version.stderr
        or version.stdout.decode("utf-8", errors="strict").strip()
        != _LATENCY_SEMANTIC_UV_VERSION
    ):
        raise RuntimeError("latency semantic uv version differs")
    runtime_root = Path(
        tempfile.mkdtemp(prefix="cachet-latency-semantic-", dir=_FREEZE_TMPDIR)
    )
    try:
        observed_root = os.lstat(runtime_root)
        if (
            not stat.S_ISDIR(observed_root.st_mode)
            or stat.S_IMODE(observed_root.st_mode) != 0o700
        ):
            raise RuntimeError("latency semantic temporary root is not private")
        private_lock = runtime_root / "requirements.lock"
        _write_exclusive(
            private_lock,
            lock_bytes,
            "latency semantic private runtime lock",
        )
        runtime_venv = runtime_root / "runtime"
        _run_silent_latency_semantic_command(
            (
                str(uv),
                "venv",
                str(runtime_venv),
                "--quiet",
                "--offline",
                "--no-project",
                "--no-python-downloads",
                "--no-config",
                "--python",
                str(python),
            ),
            label="latency semantic uv venv",
            cwd=repository_root,
            environment=environment,
        )
        runtime_python = runtime_venv / "bin/python"
        _run_silent_latency_semantic_command(
            (
                str(uv),
                "pip",
                "install",
                "--quiet",
                "--offline",
                "--no-config",
                "--no-python-downloads",
                "--require-hashes",
                "--only-binary",
                ":all:",
                "--no-compile",
                "--strict",
                "--no-deps",
                "--python",
                str(runtime_python),
                "-r",
                str(private_lock),
            ),
            label="latency semantic hash-required install",
            cwd=repository_root,
            environment=environment,
        )
        completed = subprocess.run(
            (
                str(runtime_python),
                "-S",
                "-m",
                "document_kv_cache.publication_freeze",
                "latency-plan-semantic-check",
                "--plan-json",
                str(plan_path),
                "--prepared-input-dir",
                str(prepared_input_dir),
                "--runtime-lock",
                str(private_lock),
            ),
            cwd=repository_root,
            env=environment,
            check=False,
            capture_output=True,
            timeout=PUBLICATION_FREEZE_COMMAND_TIMEOUT_SECONDS,
        )
        _require_bounded_bytes(completed.stdout, "latency semantic stdout")
        _require_bounded_bytes(completed.stderr, "latency semantic stderr")
        if completed.returncode != 0 or completed.stderr:
            raise RuntimeError("latency handoff semantic validation failed")
        try:
            value = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "latency semantic output is not UTF-8 JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise RuntimeError("latency semantic output is not an object")
        record = _mapping_copy(value, "latency semantic output")
        expected_bytes = _canonical_json_bytes(record, pretty=False)
        if completed.stdout != expected_bytes:
            raise RuntimeError("latency semantic output is not canonical JSON")
        return record
    finally:
        shutil.rmtree(runtime_root)


def _run_silent_latency_semantic_command(
    command: Sequence[str],
    *,
    label: str,
    cwd: Path,
    environment: Mapping[str, str],
) -> None:
    completed = subprocess.run(
        tuple(command),
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        timeout=PUBLICATION_FREEZE_COMMAND_TIMEOUT_SECONDS,
    )
    _require_bounded_bytes(completed.stdout, f"{label} stdout")
    _require_bounded_bytes(completed.stderr, f"{label} stderr")
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        raise RuntimeError(f"{label} failed")


def _latency_semantic_environment(repository_root: Path) -> dict[str, str]:
    environment = _base_subprocess_environment()
    environment.update(
        {
            "DO_NOT_TRACK": "1",
            "HF_HUB_CACHE": str(_LATENCY_SEMANTIC_HF_HUB_CACHE),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str((repository_root / "src").resolve()),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "TRANSFORMERS_VERBOSITY": "error",
            "UV_CACHE_DIR": str(_LATENCY_SEMANTIC_UV_CACHE),
            "UV_NO_PROGRESS": "1",
            "UV_OFFLINE": "1",
            "UV_PYTHON_DOWNLOADS": "never",
        }
    )
    return environment


def _run_publication_latency_semantic_check(
    *,
    plan_path: Path,
    prepared_input_dir: Path,
    runtime_lock_path: Path,
) -> dict[str, Any]:
    site_packages = _require_publication_latency_semantic_runtime(
        runtime_lock_path
    )
    plan = _validated_frozen_latency_handoff_plan(plan_path)
    snapshot = _require_publication_latency_tokenizer_snapshot()
    from document_kv_cache.publication_latency_handoff_generation import (
        validate_publication_latency_handoff_generation_plan,
    )
    transformers = importlib.import_module("transformers")
    auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
    if auto_tokenizer is None:
        raise RuntimeError("latency semantic AutoTokenizer is unavailable")

    with _private_publication_latency_tokenizer_snapshot(snapshot) as private:
        tokenizer = auto_tokenizer.from_pretrained(
            str(private),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        if (
            type(tokenizer).__module__
            != _LATENCY_SEMANTIC_TOKENIZER_CLASS_MODULE
            or type(tokenizer).__name__
            != _LATENCY_SEMANTIC_TOKENIZER_CLASS_NAME
            or getattr(tokenizer, "is_fast", None) is not True
            or getattr(tokenizer, "name_or_path", None) != str(private)
            or getattr(tokenizer, "vocab_size", None)
            != _LATENCY_SEMANTIC_TOKENIZER_VOCAB_SIZE
        ):
            raise RuntimeError("latency semantic tokenizer identity differs")
        validate_publication_latency_handoff_generation_plan(
            plan,
            prepared_input_dir=prepared_input_dir,
            tokenizer=tokenizer,
        )
    _require_semantic_site_packages_closure(site_packages)
    attestation = _publication_latency_semantic_attestation(plan)
    if snapshot.sha256 != attestation["tokenizer_snapshot_sha256"]:
        raise RuntimeError("latency semantic tokenizer snapshot drift")
    return attestation


def _require_publication_latency_semantic_runtime(
    runtime_lock_path: Path,
) -> Path:
    _verified_publication_latency_runtime_lock(runtime_lock_path)
    if (
        f"{platform.python_implementation()} {platform.python_version()}"
        != PUBLICATION_FREEZE_PYTHON
        or sys.flags.no_site != 1
        or site.ENABLE_USER_SITE is not None
    ):
        raise RuntimeError("latency semantic Python runtime is not isolated")
    if any(
        name in sys.modules
        for name in ("numpy", "packaging", "tokenizers", "transformers")
    ):
        raise RuntimeError("latency semantic package imported before verification")
    executable = _regular_file(
        Path(sys.executable).resolve(),
        "latency semantic current Python executable",
    )
    if _file_sha256(executable) != _LATENCY_SEMANTIC_PYTHON_SHA256:
        raise RuntimeError("latency semantic Python executable differs")
    venv_root = _regular_directory(
        Path(sys.executable).parent.parent,
        "latency semantic private virtual environment",
    )
    site_packages = _regular_directory(
        venv_root / "lib/python3.11/site-packages",
        "latency semantic private site-packages",
    )
    observed: list[tuple[str, str]] = []
    distributions = tuple(
        importlib.metadata.distributions(path=[str(site_packages)])
    )
    for distribution in distributions:
        try:
            raw_name = distribution.metadata["Name"]
        except KeyError as exc:
            raise RuntimeError(
                "latency semantic distribution lacks a name"
            ) from exc
        if not isinstance(raw_name, str) or not raw_name:
            raise RuntimeError("latency semantic distribution lacks a name")
        observed.append(
            (_canonical_distribution_name(raw_name), distribution.version)
        )
    if tuple(sorted(observed)) != _LATENCY_SEMANTIC_RUNTIME_DISTRIBUTIONS:
        raise RuntimeError("latency semantic distribution closure differs")
    digest = hashlib.sha256(
        json.dumps(
            dict(observed),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if digest != _LATENCY_SEMANTIC_RUNTIME_DISTRIBUTIONS_SHA256:
        raise RuntimeError("latency semantic distribution digest differs")
    _require_semantic_distribution_records(
        distributions,
        virtual_environment=venv_root,
    )
    _require_semantic_site_packages_closure(site_packages)
    sys.path.insert(0, str(site_packages))
    return site_packages


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _verified_publication_latency_runtime_lock(path: Path) -> bytes:
    candidate = _regular_file(path, "latency semantic runtime lock")
    content = candidate.read_bytes()
    if (
        len(content) != _LATENCY_SEMANTIC_RUNTIME_LOCK_BYTE_COUNT
        or hashlib.sha256(content).hexdigest()
        != _LATENCY_SEMANTIC_RUNTIME_LOCK_SHA256
    ):
        raise RuntimeError("latency semantic runtime lock bytes differ")
    requirements = _parse_hash_locked_requirements(candidate)
    observed: list[tuple[str, str]] = []
    for line in content.decode("utf-8", errors="strict").splitlines():
        if line and line[0].isalnum() and "==" in line:
            name, remainder = line.split("==", 1)
            version = remainder.split(" ", 1)[0]
            observed.append((_canonical_distribution_name(name), version))
    if (
        tuple(observed) != _LATENCY_SEMANTIC_RUNTIME_DISTRIBUTIONS
        or tuple(requirements) != tuple(name for name, _version in observed)
        or any(hash_count < 1 for hash_count in requirements.values())
    ):
        raise RuntimeError("latency semantic runtime lock closure differs")
    return content


def _require_semantic_distribution_records(
    distributions: Sequence[importlib.metadata.Distribution],
    *,
    virtual_environment: Path,
) -> None:
    root = virtual_environment.resolve(strict=True)
    for distribution in distributions:
        files = distribution.files
        if files is None:
            raise RuntimeError("latency semantic distribution lacks RECORD")
        unhashed: list[str] = []
        for packaged in files:
            candidate = Path(str(distribution.locate_file(packaged))).resolve(
                strict=True
            )
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(
                    "latency semantic distribution escaped its environment"
                ) from exc
            content = _regular_file(
                candidate,
                "latency semantic installed distribution file",
            ).read_bytes()
            if packaged.hash is None:
                unhashed.append(str(packaged))
                continue
            if packaged.hash.mode != "sha256" or packaged.size is None:
                raise RuntimeError(
                    "latency semantic distribution RECORD is not SHA-256 closed"
                )
            observed = base64.urlsafe_b64encode(
                hashlib.sha256(content).digest()
            ).decode("ascii").rstrip("=")
            if observed != packaged.hash.value or len(content) != packaged.size:
                raise RuntimeError(
                    "latency semantic installed distribution bytes differ"
                )
        if len(unhashed) != 1 or not unhashed[0].endswith(".dist-info/RECORD"):
            raise RuntimeError(
                "latency semantic distribution RECORD coverage differs"
            )


def _semantic_site_packages_closure(
    site_packages: Path,
) -> tuple[int, int, str]:
    root = _regular_directory(
        site_packages,
        "latency semantic private site-packages",
    )
    rows: list[dict[str, Any]] = []
    record_count = 0
    for candidate in sorted(root.rglob("*")):
        observed = os.lstat(candidate)
        if stat.S_ISDIR(observed.st_mode):
            continue
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError("latency semantic site-packages contains a link")
        relative = candidate.relative_to(root).as_posix()
        if candidate.name == "RECORD" and candidate.parent.name.endswith(
            ".dist-info"
        ):
            record_count += 1
            continue
        content = candidate.read_bytes()
        rows.append(
            {
                "byte_count": len(content),
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    digest = hashlib.sha256(
        json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return len(rows), record_count, digest


def _require_semantic_site_packages_closure(
    site_packages: Path,
    *,
    expected_file_count: int = _LATENCY_SEMANTIC_SITE_PACKAGES_FILE_COUNT,
    expected_record_count: int = len(_LATENCY_SEMANTIC_RUNTIME_DISTRIBUTIONS),
    expected_sha256: str = _LATENCY_SEMANTIC_SITE_PACKAGES_SHA256,
) -> None:
    file_count, record_count, digest = _semantic_site_packages_closure(
        site_packages
    )
    if (
        record_count != expected_record_count
        or file_count != expected_file_count
        or digest != expected_sha256
    ):
        raise RuntimeError("latency semantic installed package closure differs")


def _require_publication_latency_tokenizer_snapshot() -> _VerifiedTokenizerSnapshot:
    cache_text = os.environ.get("HF_HUB_CACHE")
    if (
        not isinstance(cache_text, str)
        or not cache_text
        or not Path(cache_text).is_absolute()
    ):
        raise RuntimeError("latency semantic HF_HUB_CACHE is not absolute")
    hub = _regular_directory(Path(cache_text), "latency semantic HF Hub cache")
    model_root = _regular_directory(
        hub / "models--Qwen--Qwen3-4B-Instruct-2507",
        "latency semantic tokenizer model cache",
    )
    blob_root = _regular_directory(
        model_root / "blobs",
        "latency semantic tokenizer blob cache",
    )
    snapshot = _regular_directory(
        model_root / "snapshots" / _LATENCY_SEMANTIC_TOKENIZER_REVISION,
        "latency semantic tokenizer snapshot",
    )
    expected_names = {
        name for name, _target, _byte_count, _sha256 in (
            _LATENCY_SEMANTIC_TOKENIZER_SNAPSHOT_FILES
        )
    }
    if {item.name for item in snapshot.iterdir()} != expected_names:
        raise RuntimeError("latency semantic tokenizer snapshot coverage differs")
    files: list[tuple[str, str, bytes]] = []
    for name, target, byte_count, expected_sha256 in (
        _LATENCY_SEMANTIC_TOKENIZER_SNAPSHOT_FILES
    ):
        link = snapshot / name
        observed = os.lstat(link)
        if not stat.S_ISLNK(observed.st_mode) or os.readlink(link) != target:
            raise RuntimeError("latency semantic tokenizer snapshot link differs")
        resolved = link.resolve(strict=True)
        expected_blob = (blob_root / Path(target).name).resolve(strict=True)
        if resolved != expected_blob:
            raise RuntimeError("latency semantic tokenizer snapshot escaped its cache")
        blob_stat = os.lstat(resolved)
        if not stat.S_ISREG(blob_stat.st_mode):
            raise RuntimeError("latency semantic tokenizer blob is not regular")
        content = resolved.read_bytes()
        observed_sha256 = hashlib.sha256(content).hexdigest()
        if len(content) != byte_count or observed_sha256 != expected_sha256:
            raise RuntimeError("latency semantic tokenizer snapshot bytes differ")
        files.append((name, target, content))
    frozen_files = tuple(files)
    digest = _verified_tokenizer_snapshot_digest(frozen_files)
    if digest != _LATENCY_SEMANTIC_TOKENIZER_SNAPSHOT_SHA256:
        raise RuntimeError("latency semantic tokenizer snapshot digest differs")
    return _VerifiedTokenizerSnapshot(sha256=digest, files=frozen_files)


def _verified_tokenizer_snapshot_digest(
    files: Sequence[tuple[str, str, bytes]],
) -> str:
    rows = [
        {
            "byte_count": len(content),
            "name": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "symlink_target": target,
        }
        for name, target, content in files
    ]
    return hashlib.sha256(
        json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class _PrivateTokenizerSnapshot:
    def __init__(
        self,
        snapshot: _VerifiedTokenizerSnapshot,
        *,
        temporary_parent: Path = _FREEZE_TMPDIR,
    ) -> None:
        self._snapshot = snapshot
        self._temporary_parent = temporary_parent
        self._root: Path | None = None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        if self._root is not None:
            raise RuntimeError("private tokenizer snapshot cannot be reused")
        if (
            _verified_tokenizer_snapshot_digest(self._snapshot.files)
            != self._snapshot.sha256
        ):
            raise RuntimeError("captured tokenizer snapshot bytes differ")
        root = Path(
            tempfile.mkdtemp(
                prefix="cachet-tokenizer-snapshot-",
                dir=_regular_directory(
                    self._temporary_parent,
                    "private tokenizer temporary parent",
                ),
            )
        )
        self._root = root
        try:
            observed = os.lstat(root)
            if (
                not stat.S_ISDIR(observed.st_mode)
                or stat.S_IMODE(observed.st_mode) != 0o700
            ):
                raise RuntimeError("private tokenizer root permissions differ")
            path = root / "snapshot"
            path.mkdir(mode=0o700)
            for name, _target, content in self._snapshot.files:
                _write_exclusive(
                    path / name,
                    content,
                    f"private tokenizer {name}",
                )
            self.path = path
            _require_private_tokenizer_snapshot(path, self._snapshot)
            return path
        except BaseException:
            shutil.rmtree(root)
            self._root = None
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            if exc_type is None and self.path is not None:
                _require_private_tokenizer_snapshot(self.path, self._snapshot)
        finally:
            if self._root is not None:
                shutil.rmtree(self._root)


def _private_publication_latency_tokenizer_snapshot(
    snapshot: _VerifiedTokenizerSnapshot,
    *,
    temporary_parent: Path = _FREEZE_TMPDIR,
) -> _PrivateTokenizerSnapshot:
    return _PrivateTokenizerSnapshot(
        snapshot,
        temporary_parent=temporary_parent,
    )


def _require_private_tokenizer_snapshot(
    path: Path,
    snapshot: _VerifiedTokenizerSnapshot,
) -> None:
    root = _regular_directory(path, "private tokenizer snapshot")
    expected = {name: content for name, _target, content in snapshot.files}
    if {item.name for item in root.iterdir()} != set(expected):
        raise RuntimeError("private tokenizer snapshot coverage differs")
    for name, content in expected.items():
        candidate = _regular_file(root / name, f"private tokenizer {name}")
        if candidate.read_bytes() != content:
            raise RuntimeError("private tokenizer snapshot bytes changed")


def write_publication_source_closure_json(
    record: Mapping[str, Any],
    path: str | Path,
    *,
    repository_root: str | Path,
    artifact_root: str | Path,
    explicit_artifact_paths: Mapping[str, str | Path] | None = None,
) -> None:
    """Validate and exclusively write one pretty canonical source closure."""

    validate_publication_source_closure_record(
        record,
        repository_root=repository_root,
        artifact_root=artifact_root,
        explicit_artifact_paths=explicit_artifact_paths,
    )
    payload = _canonical_json_bytes(record, pretty=True)
    _write_exclusive(Path(path), payload, "source closure")


def run_gpu_qualification_local_preflight(
    inputs: GPUQualificationLocalPreflightInputs,
    output_root: str | Path,
) -> dict[str, Any]:
    """Execute all seven checks and seal evidence from their written bytes."""

    return _run_gpu_qualification_local_preflight(
        inputs,
        output_root,
        command_runner=_run_command,
        now=_utc_now,
    )


def validate_gpu_qualification_local_preflight_bundle(
    path: str | Path,
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    workspace_config: DatabricksWorkspaceConfig,
    require_fresh_workspace: bool = True,
) -> dict[str, Any]:
    """Validate the sidecar closure and freshly rerun all seven checks.

    The canonical entry point is ``local-preflight-evidence.json``.  Its seven
    sibling records remain durable audit evidence; authority comes from this
    function's non-injectable live replay against their bound clean Git tree,
    artifacts, and exact tool commands.
    """

    plan = _mapping_copy(plan_record, "qualification plan")
    submitted_payload_bytes = _canonical_submit_payload_closure_bytes(
        submit_payloads
    )
    if not isinstance(workspace_config, DatabricksWorkspaceConfig):
        raise TypeError("workspace_config must be DatabricksWorkspaceConfig")
    if type(require_fresh_workspace) is not bool:
        raise TypeError("require_fresh_workspace must be a bool")
    pins = pins_from_plan_record(plan)
    _require_fixed_publication_artifact_pins(pins)
    validate_gpu_qualification_plan_record(
        plan,
        expected_campaign_id=PUBLICATION_CAMPAIGN_ID,
        expected_artifact_pins=pins,
    )
    evidence_path = _canonical_preflight_evidence_path(Path(path))
    before = _preflight_bundle_file_hashes(evidence_path)
    evidence, inputs = _validate_preflight_bundle_structural(
        evidence_path,
        plan=plan,
    )
    _require_submit_payload_closure_binding(
        inputs,
        submitted_payload_bytes=submitted_payload_bytes,
    )
    single_user_name = _qualification_single_user_name_from_payloads(
        submit_payloads
    )
    with tempfile.TemporaryDirectory(prefix="cachet-gpuq-live-preflight-") as temp:
        replay_root = Path(temp).resolve() / "preflight"
        _run_gpu_qualification_local_preflight(
            inputs,
            replay_root,
            command_runner=_run_command,
            now=_utc_now,
        )
    _validate_live_workspace_and_remote_artifacts(
        workspace_config,
        inputs=inputs,
        plan=plan,
        single_user_name=single_user_name,
        require_fresh_workspace=require_fresh_workspace,
    )
    after = _preflight_bundle_file_hashes(evidence_path)
    if after != before:
        raise ValueError("local preflight bundle changed during live replay")
    final_evidence, final_inputs = _validate_preflight_bundle_structural(
        evidence_path,
        plan=plan,
    )
    if final_evidence != evidence or final_inputs != inputs:
        raise ValueError("local preflight bindings changed during live replay")
    _require_submit_payload_closure_binding(
        final_inputs,
        submitted_payload_bytes=submitted_payload_bytes,
    )
    return final_evidence


def _validate_preflight_bundle_structural(
    evidence_path: Path,
    *,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], GPUQualificationLocalPreflightInputs]:
    root = _regular_directory(evidence_path.parent, "preflight bundle root")
    plan_sha256 = _required_sha256(
        plan.get("closed_record_sha256"), "plan digest"
    )
    expected_names = {
        *(f"{check_id}.json" for check_id in _LOCAL_CHECK_IDS),
        "local-preflight-evidence.json",
    }
    if {item.name for item in root.iterdir()} != expected_names:
        raise ValueError("preflight directory does not have exact check coverage")
    records: dict[str, dict[str, Any]] = {}
    check_hashes: dict[str, str] = {}
    check_times: list[datetime] = []
    for check_id in _LOCAL_CHECK_IDS:
        path = root / f"{check_id}.json"
        record = _read_canonical_json(path, pretty=False, label=check_id)
        _validate_local_check_record(
            record,
            expected_check_id=check_id,
            expected_plan_sha256=plan_sha256,
        )
        if record["status"] != "passed" or record["exit_code"] != 0:
            raise ValueError(f"preflight check {check_id!r} did not pass")
        records[check_id] = record
        check_hashes[check_id] = _file_sha256(path)
        check_times.append(_parse_timestamp(record["checked_at_utc"], check_id))
    evidence = _read_canonical_json(
        evidence_path, pretty=False, label="local preflight evidence"
    )
    completion = validate_local_preflight_evidence_record(
        evidence,
        plan_sha256=plan_sha256,
    )
    observed = {
        cast(str, item["check_id"]): cast(str, item["evidence_sha256"])
        for item in cast(list[dict[str, Any]], evidence["checks"])
    }
    if observed != check_hashes:
        raise ValueError("local preflight evidence sidecar hashes differ")
    if check_times and completion < max(check_times):
        raise ValueError("local preflight completed before a check timestamp")
    inputs = _preflight_inputs_from_sidecars(records)
    _validate_sidecar_semantics(records, inputs=inputs, plan=plan)
    return evidence, inputs


def _canonical_submit_payload_closure_bytes(
    submit_payloads: Sequence[Mapping[str, Any]],
) -> bytes:
    if isinstance(submit_payloads, (str, bytes, bytearray)) or not isinstance(
        submit_payloads,
        Sequence,
    ):
        raise TypeError("submit_payloads must be a sequence of mappings")
    if len(submit_payloads) != 14:
        raise ValueError("submit_payloads must contain the exact fourteen-job closure")
    normalized: list[dict[str, Any]] = []
    for index, payload in enumerate(submit_payloads):
        normalized.append(_mapping_copy(payload, f"submit payload {index}"))
    try:
        return _canonical_json_bytes(normalized, pretty=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("submit_payloads are not canonical JSON values") from exc


def _require_submit_payload_closure_binding(
    inputs: GPUQualificationLocalPreflightInputs,
    *,
    submitted_payload_bytes: bytes,
) -> None:
    sidecar_payloads = _read_canonical_json_value(
        inputs.submit_payloads_json,
        label="preflight submit payloads",
    )
    if not isinstance(sidecar_payloads, list) or len(sidecar_payloads) != 14:
        raise ValueError("preflight submit payloads lack exact fourteen-job coverage")
    if _canonical_json_bytes(sidecar_payloads, pretty=False) != submitted_payload_bytes:
        raise ValueError(
            "submitted payload closure differs from preflight submit-payloads.json"
        )


def _validate_live_workspace_and_remote_artifacts(
    config: DatabricksWorkspaceConfig,
    *,
    inputs: GPUQualificationLocalPreflightInputs,
    plan: Mapping[str, Any],
    single_user_name: str,
    require_fresh_workspace: bool,
) -> None:
    from document_kv_cache.databricks_runs import (
        list_active_databricks_runs,
        list_databricks_node_types,
        stream_databricks_volume_file_sha256,
    )

    require_databricks_current_user_name(
        config,
        expected_user_name=single_user_name,
    )
    active_runs = list_active_databricks_runs(config, max_runs=256)
    if require_fresh_workspace and active_runs:
        raise ValueError("qualification launch requires zero direct active runs")
    node_types = list_databricks_node_types(config, max_node_types=1_024)
    observed_nodes = {
        item.get("node_type_id") for item in node_types if isinstance(item, Mapping)
    }
    required_nodes = {"g5.8xlarge", "g6.8xlarge", "g6e.4xlarge"}
    if not required_nodes.issubset(observed_nodes):
        raise ValueError("direct workspace node types lack qualification hardware")
    uris_record = _read_canonical_json(
        inputs.artifact_uris_json,
        pretty=False,
        label="artifact URI record",
    )
    uris = _required_mapping(uris_record, "artifact_uris")
    output_root = _required_string(uris_record, "output_root")
    output_parent = _parent_volume_directory_uri(output_root)
    output_entries = list_databricks_volume_directory(config, output_parent)
    if require_fresh_workspace and any(
        item.get("name") == output_root.rsplit("/", 1)[-1]
        for item in output_entries
    ):
        raise ValueError("qualification output root already exists in direct listing")
    pins = pins_from_plan_record(plan)
    _require_fixed_publication_artifact_pins(pins)
    expected_file_hashes = {
        "cachet_source_tree_sha256": pins.cachet_source_tree_sha256,
        "package_wheel_sha256": pins.package_wheel_sha256,
        "patched_vllm_wheel_sha256": pins.patched_vllm_wheel_sha256,
        "runner_sha256": pins.runner_sha256,
        "runtime_lock_sha256": pins.runtime_lock_sha256,
    }
    for role, expected_sha256 in expected_file_hashes.items():
        uri = _required_string(uris, role)
        observed = stream_databricks_volume_file_sha256(
            config,
            uri,
            max_bytes=1_073_741_824,
        )
        if (
            observed.get("dbfs_uri") != uri
            or observed.get("file_sha256") != expected_sha256
            or type(observed.get("size_bytes")) is not int
            or cast(int, observed["size_bytes"]) <= 0
        ):
            raise ValueError(f"direct UC Volume readback differs for {role!r}")
    source_uri = _required_string(uris, "cachet_source_tree_sha256")
    source_root_uri = _parent_volume_uri(source_uri)
    _require_remote_tree_matches_local(
        config,
        remote_root_uri=source_root_uri,
        local_root=inputs.source_artifact_root,
        label="source artifact tree",
        stream_file=stream_databricks_volume_file_sha256,
    )
    input_uri = _required_string(uris, "input_bundle_sha256")
    _require_remote_tree_matches_local(
        config,
        remote_root_uri=input_uri,
        local_root=inputs.input_bundle,
        label="input bundle",
        stream_file=stream_databricks_volume_file_sha256,
    )


def _parent_volume_uri(uri: str) -> str:
    if not uri.startswith("dbfs:/Volumes/") or uri.endswith("/") or "/" not in uri:
        raise ValueError("artifact URI is not one canonical UC Volume file")
    return uri.rsplit("/", 1)[0]


def _parent_volume_directory_uri(uri: str) -> str:
    if (
        not uri.startswith("dbfs:/Volumes/")
        or uri.endswith("/")
        or "/" not in uri
    ):
        raise ValueError("output root is not one canonical UC Volume directory")
    return uri.rsplit("/", 1)[0]


def _require_remote_tree_matches_local(
    config: DatabricksWorkspaceConfig,
    *,
    remote_root_uri: str,
    local_root: Path,
    label: str,
    stream_file: Callable[..., Mapping[str, Any]],
) -> None:
    local = _regular_directory(local_root, f"local {label}")
    local_rows = {
        path.relative_to(local).as_posix(): {
            "file_sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _regular_tree(local)
    }
    remote_rows: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, str]] = [(remote_root_uri, "")]
    seen_directories: set[str] = set()
    while pending:
        directory_uri, prefix = pending.pop()
        if directory_uri in seen_directories:
            raise RuntimeError(f"direct {label} listing repeated a directory")
        seen_directories.add(directory_uri)
        if len(seen_directories) > 256:
            raise RuntimeError(f"direct {label} listing exceeds directory cap")
        for entry in list_databricks_volume_directory(config, directory_uri):
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                raise RuntimeError(f"direct {label} listing has an invalid name")
            relative = f"{prefix}/{name}".lstrip("/")
            entry_path = entry.get("path")
            if not isinstance(entry_path, str):
                raise RuntimeError(f"direct {label} listing has an invalid path")
            if entry.get("is_directory") is True:
                pending.append((f"dbfs:{entry_path.removesuffix('/')}", relative))
                continue
            if entry.get("is_directory") is not False:
                raise RuntimeError(f"direct {label} listing has an invalid kind")
            if relative in remote_rows or len(remote_rows) >= 4_096:
                raise RuntimeError(f"direct {label} listing repeats or exceeds files")
            uri = f"dbfs:{entry_path}"
            observed = stream_file(config, uri, max_bytes=1_073_741_824)
            if (
                observed.get("dbfs_uri") != uri
                or not isinstance(observed.get("file_sha256"), str)
                or not _SHA256_RE.fullmatch(cast(str, observed["file_sha256"]))
                or type(observed.get("size_bytes")) is not int
                or cast(int, observed["size_bytes"]) < 0
            ):
                raise RuntimeError(
                    f"direct {label} file identity response differs"
                )
            remote_rows[relative] = {
                "file_sha256": observed.get("file_sha256"),
                "size_bytes": observed.get("size_bytes"),
            }
    if remote_rows != local_rows:
        raise ValueError(f"direct UC Volume {label} differs from local closed bytes")


def _run_gpu_qualification_local_preflight(
    inputs: GPUQualificationLocalPreflightInputs,
    output_root: str | Path,
    *,
    command_runner: CommandRunner,
    now: Callable[[], datetime],
    verify_source_rebuild: bool = True,
) -> dict[str, Any]:
    if not isinstance(inputs, GPUQualificationLocalPreflightInputs):
        raise TypeError("inputs must be GPUQualificationLocalPreflightInputs")
    _require_canonical_preflight_tool_paths(inputs)
    root = Path(output_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"preflight output root already exists: {root}")
    _create_directory_exclusive(root, "preflight output root")
    plan = _read_canonical_json(inputs.plan_json, pretty=False, label="qualification plan")
    pins = pins_from_plan_record(plan)
    _require_fixed_publication_artifact_pins(pins)
    validate_gpu_qualification_plan_record(
        plan,
        expected_campaign_id=PUBLICATION_CAMPAIGN_ID,
        expected_artifact_pins=pins,
    )
    plan_sha256 = _required_sha256(plan.get("closed_record_sha256"), "plan digest")
    checks: tuple[tuple[str, Callable[[], tuple[dict[str, Any], list[dict[str, Any]]]]], ...] = (
        (
            "canonical_plan_schema",
            lambda: _check_canonical_plan(inputs, plan, pins),
        ),
        (
            "runtime_lock_require_hashes",
            lambda: _check_runtime_lock(inputs, pins),
        ),
        (
            "patched_wheel_record_and_manifest",
            lambda: _check_patched_wheel(inputs, pins),
        ),
        (
            "source_runner_input_closure",
            lambda: _check_source_runner_inputs(
                inputs,
                plan,
                pins,
                verify_rebuild=verify_source_rebuild,
            ),
        ),
    )
    check_hashes: dict[str, str] = {}
    for check_id, check in checks:
        try:
            result, bindings = check()
        except BaseException as exc:
            failed = _local_check_record(
                check_id=check_id,
                plan_sha256=plan_sha256,
                checked_at=now(),
                command=("python-api", f"document_kv_cache.publication_freeze:{check_id}"),
                exit_code=1,
                environment=_python_api_environment_identity(),
                inputs=[],
                result={"error": _bounded_error(exc)},
                stdout=b"",
                stderr=str(exc).encode("utf-8", errors="replace"),
                tool_identity=_python_api_identity(),
                status="failed",
            )
            _write_check(root, failed)
            raise RuntimeError(f"local preflight check {check_id!r} failed") from exc
        passed = _local_check_record(
            check_id=check_id,
            plan_sha256=plan_sha256,
            checked_at=now(),
            command=("python-api", f"document_kv_cache.publication_freeze:{check_id}"),
            exit_code=0,
            environment=_python_api_environment_identity(),
            inputs=bindings,
            result=result,
            stdout=b"",
            stderr=b"",
            tool_identity=_python_api_identity(),
            status="passed",
        )
        check_hashes[check_id] = _write_check(root, passed)
    command_checks = _command_check_specs(inputs)
    environment = _preflight_environment(inputs.repository_root)
    for check_id, command, version_command, expected_version in command_checks:
        executable = _resolve_executable(inputs.repository_root, Path(command[0]))
        normalized_command = (str(executable), *command[1:])
        normalized_version = (str(executable), *version_command[1:])
        version = command_runner(normalized_version, inputs.repository_root, environment)
        _require_bounded_command(version, f"{check_id} version")
        version_text = version.stdout.decode("utf-8", errors="strict").strip()
        if version.returncode != 0 or version.stderr or version_text != expected_version:
            raise RuntimeError(f"{check_id} tool identity differs")
        completed = command_runner(normalized_command, inputs.repository_root, environment)
        _require_bounded_command(completed, check_id)
        if check_id == "unit_tests" and completed.returncode == 0:
            _require_exact_pytest_completion(completed.stdout)
        status = "passed" if completed.returncode == 0 else "failed"
        record = _local_check_record(
            check_id=check_id,
            plan_sha256=plan_sha256,
            checked_at=now(),
            command=normalized_command,
            exit_code=completed.returncode,
            environment=environment,
            inputs=[_repository_binding(inputs.repository_root)],
            result=_command_check_result(
                check_id,
                expected_version=expected_version,
                repository_root=inputs.repository_root,
            ),
            stdout=completed.stdout,
            stderr=completed.stderr,
            tool_identity={
                "executable_path": str(executable),
                "executable_sha256": _file_sha256(executable),
                "version": version_text,
                "version_command": list(normalized_version),
            },
            status=status,
        )
        check_hashes[check_id] = _write_check(root, record)
        if completed.returncode != 0:
            raise RuntimeError(
                f"local preflight command {check_id!r} returned "
                f"{completed.returncode}"
            )
    if tuple(check_hashes) != _LOCAL_CHECK_IDS:
        raise RuntimeError("local preflight did not execute exact check coverage")
    _validate_written_checks_before_seal(
        root,
        plan=plan,
        expected_hashes=check_hashes,
    )
    completed_at = now()
    evidence = build_local_preflight_evidence(
        plan_sha256=plan_sha256,
        completed_at_utc=_timestamp(completed_at),
        check_evidence_sha256=check_hashes,
    )
    evidence_path = root / "local-preflight-evidence.json"
    _write_exclusive(
        evidence_path,
        _canonical_json_bytes(evidence, pretty=False),
        "local preflight evidence",
    )
    try:
        validated, _validated_inputs = _validate_preflight_bundle_structural(
            evidence_path,
            plan=plan,
        )
    except BaseException:
        if evidence_path.is_file() and not evidence_path.is_symlink():
            evidence_path.unlink()
        raise
    return validated


def _validate_written_checks_before_seal(
    root: Path,
    *,
    plan: Mapping[str, Any],
    expected_hashes: Mapping[str, str],
) -> None:
    expected_names = {f"{check_id}.json" for check_id in _LOCAL_CHECK_IDS}
    if {item.name for item in root.iterdir()} != expected_names:
        raise RuntimeError("local preflight has extra or missing pre-seal sidecars")
    plan_sha256 = _required_sha256(
        plan.get("closed_record_sha256"), "plan digest"
    )
    records: dict[str, dict[str, Any]] = {}
    observed_hashes: dict[str, str] = {}
    for check_id in _LOCAL_CHECK_IDS:
        path = root / f"{check_id}.json"
        record = _read_canonical_json(path, pretty=False, label=check_id)
        _validate_local_check_record(
            record,
            expected_check_id=check_id,
            expected_plan_sha256=plan_sha256,
        )
        if record["status"] != "passed" or record["exit_code"] != 0:
            raise RuntimeError(f"pre-seal check {check_id!r} did not pass")
        records[check_id] = record
        observed_hashes[check_id] = _file_sha256(path)
    if observed_hashes != dict(expected_hashes):
        raise RuntimeError("pre-seal sidecar bytes changed after execution")
    inputs = _preflight_inputs_from_sidecars(records)
    _validate_sidecar_semantics(records, inputs=inputs, plan=plan)


def _check_canonical_plan(
    inputs: GPUQualificationLocalPreflightInputs,
    plan: Mapping[str, Any],
    pins: GPUQualificationArtifactPins,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    uris_record = _read_canonical_json(
        inputs.artifact_uris_json, pretty=False, label="artifact URI record"
    )
    if set(uris_record) != {"artifact_uris", "output_root", "plan_sha256"}:
        raise ValueError("artifact URI record has an open schema")
    if uris_record["plan_sha256"] != plan["closed_record_sha256"]:
        raise ValueError("artifact URI record plan digest differs")
    uris = _required_mapping(uris_record, "artifact_uris")
    if set(uris) != set(GPU_QUALIFICATION_ARTIFACT_KEYS):
        raise ValueError("artifact URI record does not cover exact roles")
    payloads_raw = _read_canonical_json_value(
        inputs.submit_payloads_json, label="submit payloads"
    )
    if not isinstance(payloads_raw, list):
        raise ValueError("submit payloads must be an array")
    runner_uri = _required_string(uris, "runner_sha256")
    package_uri = _required_string(uris, "package_wheel_sha256")
    patched_uri = _required_string(uris, "patched_vllm_wheel_sha256")
    single_user_name = _qualification_single_user_name_from_payloads(payloads_raw)
    expected = render_gpu_qualification_submit_payloads(
        plan,
        single_user_name=single_user_name,
        runner_uri=runner_uri,
        package_wheel_uri=package_uri,
        patched_vllm_wheel_uri=patched_uri,
        artifact_uris={key: _required_string(uris, key) for key in uris},
        output_root=_required_string(uris_record, "output_root"),
    )
    if payloads_raw != list(expected):
        raise ValueError("submit payload bytes differ from the reviewed renderer")
    sizes = [
        len(
            json.dumps(
                payload["tasks"][0]["spark_python_task"]["parameters"],
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for payload in expected
    ]
    if len(expected) != 14 or max(sizes) > GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES:
        raise ValueError("qualification payload count or parameter size differs")
    required_nodes = {"g5.8xlarge", "g6.8xlarge", "g6e.4xlarge"}
    return (
        {
            "authority_scope": "local_renderer_and_payload_size_only",
            "job_count": len(expected),
            "max_parameters_json_bytes": max(sizes),
            "min_parameters_json_bytes": min(sizes),
            "node_types": sorted(required_nodes),
            "artifact_pins": pins.to_record(),
        },
        _bindings(
            (inputs.plan_json, "plan_json"),
            (inputs.artifact_uris_json, "artifact_uris_json"),
            (inputs.submit_payloads_json, "submit_payloads_json"),
        ),
    )


def _check_runtime_lock(
    inputs: GPUQualificationLocalPreflightInputs,
    pins: GPUQualificationArtifactPins,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lock = _regular_file(inputs.runtime_lock, "runtime_lock")
    lock_input = _regular_file(inputs.runtime_lock_input, "runtime_lock_input")
    if _file_sha256(lock) != pins.runtime_lock_sha256:
        raise ValueError("runtime lock differs from qualification plan")
    if _file_sha256(lock_input) != PUBLICATION_FREEZE_RUNTIME_LOCK_INPUT_SHA256:
        raise ValueError("runtime lock input differs")
    requirements = _parse_hash_locked_requirements(lock)
    if len(requirements) != VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT:
        raise ValueError("runtime lock distribution count differs")
    with zipfile.ZipFile(inputs.package_wheel) as archive:
        packaged = archive.read(
            f"document_kv_cache/runtime_locks/{VLLM_RUNTIME_LOCK_FILENAME}"
        )
    if packaged != lock.read_bytes():
        raise ValueError("package wheel runtime lock differs")
    return (
        {
            "distribution_count": len(requirements),
            "hash_required": True,
            "lock_sha256": _file_sha256(lock),
            "lock_input_sha256": _file_sha256(lock_input),
            "packaged_lock_matches": True,
            "authority_scope": "local_runtime_lock_only",
        },
        _bindings(
            (lock, "runtime_lock"),
            (lock_input, "runtime_lock_input"),
            (inputs.package_wheel, "package_wheel"),
        ),
    )


def _check_patched_wheel(
    inputs: GPUQualificationLocalPreflightInputs,
    pins: GPUQualificationArtifactPins,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    official = _regular_file(inputs.official_vllm_wheel, "official_vllm_wheel")
    patched = _regular_file(inputs.patched_vllm_wheel, "patched_vllm_wheel")
    manifest_path = _regular_file(inputs.patched_vllm_manifest, "patched manifest")
    if _file_sha256(official) != PUBLICATION_FREEZE_OFFICIAL_VLLM_WHEEL_SHA256:
        raise ValueError("official vLLM wheel differs")
    if _file_sha256(patched) != pins.patched_vllm_wheel_sha256:
        raise ValueError("patched vLLM wheel differs from qualification plan")
    if _file_sha256(manifest_path) != PUBLICATION_FREEZE_PATCH_MANIFEST_SHA256:
        raise ValueError("patched vLLM manifest differs")
    manifest = _read_canonical_json(manifest_path, pretty=True, label="patched manifest")
    if (
        manifest.get("record_type") != VLLM_PATCHED_WHEEL_MANIFEST_RECORD_TYPE
        or manifest.get("schema_version")
        != VLLM_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION
        or manifest.get("source_wheel_sha256")
        != PUBLICATION_FREEZE_OFFICIAL_VLLM_WHEEL_SHA256
        or manifest.get("patched_wheel_sha256")
        != PUBLICATION_FREEZE_PATCHED_VLLM_WHEEL_SHA256
    ):
        raise ValueError("patched vLLM manifest identity differs")
    member_count, record_rows = _audit_wheel_record(patched)
    patch_closure = manifest.get("patch_closure")
    if not isinstance(patch_closure, list) or len(patch_closure) != 3:
        raise ValueError("patched vLLM patch closure differs")
    return (
        {
            "authority_scope": "local_wheel_and_manifest_only",
            "manifest_sha256": _file_sha256(manifest_path),
            "member_count": member_count,
            "patch_member_count": len(patch_closure),
            "record_rows_valid": record_rows,
            "source_wheel_sha256": _file_sha256(official),
        },
        _bindings(
            (official, "official_vllm_wheel"),
            (patched, "patched_vllm_wheel"),
            (manifest_path, "patched_vllm_manifest"),
        ),
    )


def _check_source_runner_inputs(
    inputs: GPUQualificationLocalPreflightInputs,
    plan: Mapping[str, Any],
    pins: GPUQualificationArtifactPins,
    *,
    verify_rebuild: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = _read_canonical_json(
        inputs.source_closure_json, pretty=True, label="source closure"
    )
    if verify_rebuild:
        validate_publication_source_closure_record(
            source,
            repository_root=inputs.repository_root,
            artifact_root=inputs.source_artifact_root,
        )
    else:
        _validate_publication_source_closure_record(
            source,
            repository_root=inputs.repository_root,
            artifact_root=inputs.source_artifact_root,
            explicit_artifact_paths=None,
            verify_rebuild=False,
        )
    source_digest = _file_sha256(inputs.source_closure_json)
    if source_digest != pins.cachet_source_tree_sha256:
        raise ValueError("source closure differs from qualification plan")
    if _file_sha256(inputs.package_wheel) != pins.package_wheel_sha256:
        raise ValueError("package wheel differs from qualification plan")
    if _file_sha256(inputs.runner) != pins.runner_sha256:
        raise ValueError("runner differs from qualification plan")
    input_digest = _input_bundle_closure_sha256(inputs.input_bundle)
    if input_digest != pins.input_bundle_sha256:
        raise ValueError("input bundle differs from qualification plan")
    source_files = _mapping_sequence(source.get("files"), "source closure files")
    return (
        {
            "authority_scope": "local_source_runner_and_input_only",
            "input_bundle_file_count": len(_regular_tree(inputs.input_bundle)),
            "input_bundle_sha256": input_digest,
            "package_wheel_sha256": _file_sha256(inputs.package_wheel),
            "runner_sha256": _file_sha256(inputs.runner),
            "source_closed_record_sha256": source["closed_record_sha256"],
            "source_file_count": len(source_files) + 1,
            "source_manifest_sha256": source_digest,
            "plan_sha256": plan["closed_record_sha256"],
        },
        _bindings(
            (inputs.source_closure_json, "source_closure_json"),
            (inputs.source_artifact_root, "source_artifact_root"),
            (inputs.package_wheel, "package_wheel"),
            (inputs.runner, "runner"),
            (inputs.input_bundle, "input_bundle"),
        ),
    )


def _canonical_preflight_evidence_path(path: Path) -> Path:
    raw = str(path)
    if not path.is_absolute() or path != Path(os.path.normpath(raw)):
        raise ValueError("local preflight evidence path must be normalized absolute")
    candidate = _regular_file(path, "local preflight evidence")
    if candidate.name != "local-preflight-evidence.json":
        raise ValueError(
            "local preflight evidence must use the canonical filename"
        )
    _regular_directory(candidate.parent, "preflight bundle root")
    return candidate


def _preflight_bundle_file_hashes(evidence_path: Path) -> dict[str, str]:
    root = _regular_directory(evidence_path.parent, "preflight bundle root")
    expected_names = {
        *(f"{check_id}.json" for check_id in _LOCAL_CHECK_IDS),
        "local-preflight-evidence.json",
    }
    if {item.name for item in root.iterdir()} != expected_names:
        raise ValueError("preflight directory does not have exact check coverage")
    return {
        name: _file_sha256(_regular_file(root / name, f"preflight {name}"))
        for name in sorted(expected_names)
    }


def _preflight_inputs_from_sidecars(
    records: Mapping[str, Mapping[str, Any]],
) -> GPUQualificationLocalPreflightInputs:
    if tuple(records) != _LOCAL_CHECK_IDS:
        raise ValueError("local preflight sidecars are not in canonical order")
    bindings_by_label: dict[str, Mapping[str, Any]] = {}
    for check_id in _LOCAL_CHECK_IDS:
        record = records[check_id]
        raw_bindings = record.get("inputs")
        if not isinstance(raw_bindings, list) or any(
            not isinstance(item, Mapping) for item in raw_bindings
        ):
            raise ValueError(f"preflight check {check_id!r} inputs are invalid")
        bindings = cast(list[Mapping[str, Any]], raw_bindings)
        observed_labels = tuple(item.get("label") for item in bindings)
        if observed_labels != _CHECK_INPUT_LABELS[check_id]:
            raise ValueError(
                f"preflight check {check_id!r} input coverage differs"
            )
        for binding in bindings:
            _validate_path_binding(binding)
            label = cast(str, binding["label"])
            prior = bindings_by_label.get(label)
            if prior is not None and dict(prior) != dict(binding):
                raise ValueError(f"preflight input {label!r} has conflicting bindings")
            bindings_by_label[label] = binding

    def bound_path(label: str) -> Path:
        binding = bindings_by_label.get(label)
        if binding is None:
            raise ValueError(f"preflight input {label!r} is missing")
        value = binding.get("path")
        if not isinstance(value, str):
            raise ValueError(f"preflight input {label!r} path is invalid")
        return Path(value)

    command_tools: dict[str, Path] = {}
    for check_id in _COMMAND_CHECK_IDS:
        tool = records[check_id].get("tool_identity")
        if not isinstance(tool, Mapping):
            raise ValueError(f"preflight check {check_id!r} tool identity is invalid")
        executable = tool.get("executable_path")
        if not isinstance(executable, str) or not Path(executable).is_absolute():
            raise ValueError(f"preflight check {check_id!r} executable is invalid")
        command_tools[check_id] = Path(executable)
    return GPUQualificationLocalPreflightInputs(
        repository_root=bound_path("repository_root"),
        plan_json=bound_path("plan_json"),
        artifact_uris_json=bound_path("artifact_uris_json"),
        submit_payloads_json=bound_path("submit_payloads_json"),
        source_closure_json=bound_path("source_closure_json"),
        source_artifact_root=bound_path("source_artifact_root"),
        package_wheel=bound_path("package_wheel"),
        runner=bound_path("runner"),
        input_bundle=bound_path("input_bundle"),
        runtime_lock=bound_path("runtime_lock"),
        runtime_lock_input=bound_path("runtime_lock_input"),
        official_vllm_wheel=bound_path("official_vllm_wheel"),
        patched_vllm_wheel=bound_path("patched_vllm_wheel"),
        patched_vllm_manifest=bound_path("patched_vllm_manifest"),
        python_executable=command_tools["unit_tests"],
        ruff_executable=command_tools["ruff"],
        mypy_executable=command_tools["mypy"],
    )


def _validate_sidecar_semantics(
    records: Mapping[str, Mapping[str, Any]],
    *,
    inputs: GPUQualificationLocalPreflightInputs,
    plan: Mapping[str, Any],
) -> None:
    _require_canonical_preflight_tool_paths(inputs)
    pins = pins_from_plan_record(plan)
    _require_fixed_publication_artifact_pins(pins)
    python_checks: tuple[
        tuple[
            str,
            Callable[[], tuple[dict[str, Any], list[dict[str, Any]]]],
        ],
        ...,
    ] = (
        (
            "canonical_plan_schema",
            lambda: _check_canonical_plan(inputs, plan, pins),
        ),
        (
            "runtime_lock_require_hashes",
            lambda: _check_runtime_lock(inputs, pins),
        ),
        (
            "patched_wheel_record_and_manifest",
            lambda: _check_patched_wheel(inputs, pins),
        ),
        (
            "source_runner_input_closure",
            lambda: _check_source_runner_inputs(
                inputs,
                plan,
                pins,
                verify_rebuild=False,
            ),
        ),
    )
    empty_output = _command_output_binding(b"")
    python_identity = _python_api_identity()
    for check_id, check in python_checks:
        record = records[check_id]
        expected_result, expected_bindings = check()
        expected_command = [
            "python-api",
            f"document_kv_cache.publication_freeze:{check_id}",
        ]
        if record.get("command") != expected_command:
            raise ValueError(f"preflight check {check_id!r} command differs")
        if record.get("inputs") != expected_bindings:
            raise ValueError(f"preflight check {check_id!r} inputs differ")
        if record.get("result") != expected_result:
            raise ValueError(f"preflight check {check_id!r} result differs")
        if record.get("tool_identity") != python_identity:
            raise ValueError(f"preflight check {check_id!r} tool identity differs")
        if record.get("environment") != _python_api_environment_identity():
            raise ValueError(f"preflight check {check_id!r} environment differs")
        if record.get("stdout") != empty_output or record.get("stderr") != empty_output:
            raise ValueError(f"preflight check {check_id!r} output differs")

    for (
        check_id,
        command,
        version_command,
        expected_version,
    ) in _command_check_specs(inputs):
        record = records[check_id]
        executable = _resolve_executable(inputs.repository_root, Path(command[0]))
        normalized_command = [str(executable), *command[1:]]
        normalized_version = [str(executable), *version_command[1:]]
        expected_tool = {
            "executable_path": str(executable),
            "executable_sha256": _file_sha256(executable),
            "version": expected_version,
            "version_command": normalized_version,
        }
        if record.get("command") != normalized_command:
            raise ValueError(f"preflight check {check_id!r} command differs")
        if record.get("inputs") != [_repository_binding(inputs.repository_root)]:
            raise ValueError(f"preflight check {check_id!r} inputs differ")
        if record.get("result") != _command_check_result(
            check_id,
            expected_version=expected_version,
            repository_root=inputs.repository_root,
        ):
            raise ValueError(f"preflight check {check_id!r} result differs")
        if record.get("tool_identity") != expected_tool:
            raise ValueError(f"preflight check {check_id!r} tool identity differs")
        if record.get("environment") != _preflight_environment(
            inputs.repository_root
        ):
            raise ValueError(f"preflight check {check_id!r} environment differs")
        _validate_command_output_binding(record.get("stdout"), f"{check_id} stdout")
        _validate_command_output_binding(record.get("stderr"), f"{check_id} stderr")
        if record.get("stderr") != empty_output:
            raise ValueError(f"preflight check {check_id!r} wrote stderr")


def _command_check_specs(
    inputs: GPUQualificationLocalPreflightInputs,
) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...], str], ...]:
    return (
        (
            "unit_tests",
            (
                str(inputs.python_executable),
                "-m",
                "pytest",
                "-q",
                "-c",
                "pyproject.toml",
                "-p",
                "no:cacheprovider",
                "tests",
            ),
            (str(inputs.python_executable), "-m", "pytest", "--version"),
            "pytest 8.0.0",
        ),
        (
            "ruff",
            (
                str(inputs.ruff_executable),
                "check",
                "--no-cache",
                "--config",
                "pyproject.toml",
                *_RUFF_TARGETS,
            ),
            (str(inputs.ruff_executable), "--version"),
            "ruff 0.15.21",
        ),
        (
            "mypy",
            (
                str(inputs.mypy_executable),
                "--strict",
                "--no-incremental",
                "--config-file",
                "pyproject.toml",
                *_MYPY_TARGETS,
            ),
            (str(inputs.mypy_executable), "--version"),
            "mypy 2.2.0 (compiled: yes)",
        ),
    )


def _require_canonical_preflight_tool_paths(
    inputs: GPUQualificationLocalPreflightInputs,
) -> None:
    observed_runtime = f"{platform.python_implementation()} {platform.python_version()}"
    if observed_runtime != PUBLICATION_FREEZE_PYTHON:
        raise RuntimeError(
            f"qualification preflight requires {PUBLICATION_FREEZE_PYTHON}, "
            f"found {observed_runtime}"
        )
    if Path(sys.executable).resolve() != _DEFAULT_PYTHON_EXECUTABLE.resolve():
        raise RuntimeError(
            "qualification preflight must run under the frozen Python executable"
        )
    expected = {
        "python": _DEFAULT_PYTHON_EXECUTABLE.resolve(),
        "ruff": (inputs.repository_root / _DEFAULT_RUFF_EXECUTABLE).resolve(),
        "mypy": (inputs.repository_root / _DEFAULT_MYPY_EXECUTABLE).resolve(),
    }
    observed = {
        "python": inputs.python_executable.resolve(),
        "ruff": inputs.ruff_executable.resolve(),
        "mypy": inputs.mypy_executable.resolve(),
    }
    if observed != expected:
        raise ValueError("qualification preflight tool paths differ from the freeze")


def _command_check_result(
    check_id: str,
    *,
    expected_version: str,
    repository_root: Path,
) -> dict[str, Any]:
    if check_id == "unit_tests":
        scope = "repository_test_suite"
        targets = ["tests"]
        excluded: list[str] = []
    elif check_id in {"ruff", "mypy"}:
        scope = "frozen_publication_surface"
        targets = list(_STATIC_ANALYSIS_TARGETS)
        excluded = list(_STATIC_ANALYSIS_EXCLUDED_TARGETS)
    else:
        raise ValueError("unknown command preflight check")
    return {
        "excluded_known_dirty_paths": excluded,
        "expected_tool_version": expected_version,
        "scope": scope,
        "target_paths": targets,
        "working_directory": str(repository_root.resolve()),
    }


def _require_fixed_publication_artifact_pins(
    pins: GPUQualificationArtifactPins,
) -> None:
    if pins.runtime_lock_sha256 != VLLM_RUNTIME_LOCK_SHA256:
        raise ValueError("qualification plan runtime-lock pin differs from publication")
    if pins.patched_vllm_wheel_sha256 != (
        PUBLICATION_FREEZE_PATCHED_VLLM_WHEEL_SHA256
    ):
        raise ValueError("qualification plan patched-wheel pin differs from publication")
    if pins.runner_sha256 != GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256:
        raise ValueError("qualification plan runner pin differs from publication")
    if pins.input_bundle_sha256 != PUBLICATION_FREEZE_INPUT_BUNDLE_SHA256:
        raise ValueError("qualification plan input-bundle pin differs from publication")


def _build_package_twice(
    repository_root: Path,
    *,
    commit: str,
    source_date_epoch: int,
) -> tuple[_PackageBuildOutputs, _PackageBuildOutputs]:
    temp_parent = _regular_directory(
        Path(tempfile.gettempdir()).resolve(),
        "canonical build temporary parent",
    )
    with tempfile.TemporaryDirectory(
        prefix="cachet-publication-build-",
        dir=temp_parent,
    ) as raw_temp:
        temp_root = _regular_directory(Path(raw_temp), "package build temporary root")
        first_root = _create_directory_exclusive(temp_root / "first", "first build")
        second_root = _create_directory_exclusive(
            temp_root / "second", "second build"
        )
        first = _run_isolated_package_build(
            repository_root,
            commit=commit,
            source_date_epoch=source_date_epoch,
            build_root=first_root,
        )
        second = _run_isolated_package_build(
            repository_root,
            commit=commit,
            source_date_epoch=source_date_epoch,
            build_root=second_root,
        )
    return first, second


def _run_isolated_package_build(
    repository_root: Path,
    *,
    commit: str,
    source_date_epoch: int,
    build_root: Path,
) -> _PackageBuildOutputs:
    source_root = _create_directory_exclusive(build_root / "source", "build source")
    dist_root = _create_directory_exclusive(build_root / "dist", "build dist")
    archive = subprocess.run(
        (
            str(_GIT_EXECUTABLE),
            "-C",
            str(repository_root),
            "archive",
            "--format=tar",
            commit,
        ),
        check=False,
        capture_output=True,
        env=_git_environment(),
        timeout=120,
    )
    _require_bounded_bytes(
        archive.stdout,
        "package build git archive",
        max_bytes=PUBLICATION_FREEZE_MAX_ARTIFACT_BYTES,
    )
    _require_bounded_bytes(archive.stderr, "package build git archive stderr")
    if archive.returncode != 0 or archive.stderr:
        raise RuntimeError("package build git archive failed")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as source_tar:
        members = source_tar.getmembers()
        if not members:
            raise RuntimeError("package build source archive is empty")
        for member in members:
            _safe_archive_name(member.name, "package build source archive")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError("package build source archive has an irregular entry")
        source_tar.extractall(source_root, members=members, filter="data")
    _regular_tree(source_root)
    environment = _build_environment(source_date_epoch)
    command = (
        str(_DEFAULT_PYTHON_EXECUTABLE),
        "-m",
        "build",
        "--installer",
        "pip",
        "--wheel",
        "--sdist",
        "--outdir",
        str(dist_root),
        str(source_root),
    )
    completed = subprocess.run(
        command,
        cwd=source_root,
        env=environment,
        check=False,
        capture_output=True,
        timeout=PUBLICATION_FREEZE_COMMAND_TIMEOUT_SECONDS,
    )
    _require_bounded_bytes(completed.stdout, "package build stdout")
    _require_bounded_bytes(completed.stderr, "package build stderr")
    if completed.returncode != 0:
        raise RuntimeError(
            f"deterministic package build returned {completed.returncode}"
        )
    outputs = _regular_tree(dist_root)
    if len(outputs) != 2:
        raise RuntimeError("package build did not emit exactly wheel and sdist")
    wheels = [path for path in outputs if path.suffix == ".whl"]
    sdists = [path for path in outputs if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("package build output kinds differ")
    wheel = wheels[0]
    sdist = sdists[0]
    if (
        wheel.stat().st_size > PUBLICATION_FREEZE_MAX_ARTIFACT_BYTES
        or sdist.stat().st_size > PUBLICATION_FREEZE_MAX_ARTIFACT_BYTES
    ):
        raise RuntimeError("package build output exceeds source-freeze byte cap")
    _validate_cachet_wheel(wheel)
    _validate_sdist(sdist)
    return _PackageBuildOutputs(
        wheel_name=wheel.name,
        wheel_bytes=wheel.read_bytes(),
        sdist_name=sdist.name,
        sdist_bytes=sdist.read_bytes(),
    )


def _build_environment(source_date_epoch: int) -> dict[str, str]:
    environment = _base_subprocess_environment()
    environment.update(
        {
            "PIP_CONFIG_FILE": "/dev/null",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_INDEX_URL": "https://pypi.org/simple",
            "PIP_NO_INPUT": "1",
            "SOURCE_DATE_EPOCH": str(source_date_epoch),
        }
    )
    return environment


def _base_subprocess_environment() -> dict[str, str]:
    _regular_directory(_FREEZE_HOME, "frozen subprocess HOME")
    _regular_directory(_FREEZE_TMPDIR, "frozen subprocess TMPDIR")
    return {
        "HOME": str(_FREEZE_HOME),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NO_COLOR": "1",
        "PATH": _FREEZE_PATH,
        "PYTHONHASHSEED": "0",
        "TMPDIR": str(_FREEZE_TMPDIR),
        "TZ": "UTC",
        "XDG_CACHE_HOME": str(_FREEZE_TMPDIR),
        "XDG_CONFIG_HOME": str(_FREEZE_HOME),
    }


def _git_environment() -> dict[str, str]:
    environment = _base_subprocess_environment()
    environment.update(
        {
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _require_matching_build_outputs(
    first: _PackageBuildOutputs,
    second: _PackageBuildOutputs,
) -> None:
    if first != second:
        raise ValueError(
            "two isolated clean-tree package builds did not produce identical bytes"
        )


def _require_file_bytes(path: Path, expected: bytes, label: str) -> None:
    candidate = _regular_file(path, label)
    if (
        candidate.stat().st_size != len(expected)
        or _file_sha256(candidate) != hashlib.sha256(expected).hexdigest()
    ):
        raise ValueError(f"{label} bytes differ")


def _write_deterministic_git_archive(
    repository_root: Path,
    *,
    commit: str,
    output_path: Path,
) -> None:
    prefix = f"cachet-{commit}/"
    archive = subprocess.run(
        (
            str(_GIT_EXECUTABLE),
            "-C",
            str(repository_root),
            "archive",
            "--format=tar",
            f"--prefix={prefix}",
            commit,
        ),
        check=False,
        capture_output=True,
        env=_git_environment(),
        timeout=120,
    )
    _require_bounded_bytes(
        archive.stdout,
        "git source archive",
        max_bytes=PUBLICATION_FREEZE_MAX_ARTIFACT_BYTES,
    )
    _require_bounded_bytes(archive.stderr, "git source archive stderr")
    if archive.returncode != 0 or archive.stderr:
        raise RuntimeError("git source archive generation failed")
    compressed = subprocess.run(
        (str(_GZIP_EXECUTABLE), "-n"),
        input=archive.stdout,
        check=False,
        capture_output=True,
        env=_base_subprocess_environment(),
        timeout=120,
    )
    _require_bounded_bytes(
        compressed.stdout,
        "compressed git source archive",
        max_bytes=PUBLICATION_FREEZE_MAX_ARTIFACT_BYTES,
    )
    _require_bounded_bytes(compressed.stderr, "gzip stderr")
    if compressed.returncode != 0 or compressed.stderr:
        raise RuntimeError("deterministic git source archive compression failed")
    _write_exclusive(output_path, compressed.stdout, "git source archive")


def _source_file_record(path: Path, role: str) -> dict[str, Any]:
    candidate = _regular_file(path, role)
    if candidate.stat().st_size > PUBLICATION_FREEZE_MAX_ARTIFACT_BYTES:
        raise ValueError(f"{role} exceeds the source-closure byte cap")
    return {
        "byte_count": candidate.stat().st_size,
        "relative_path": candidate.name,
        "role": role,
        "sha256": _file_sha256(candidate),
    }


def _source_reference_record(root: Path, path: Path, role: str) -> dict[str, Any]:
    candidate = _regular_file(path, role)
    try:
        relative = candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"source reference {role!r} is outside repository_root") from exc
    return {
        "byte_count": candidate.stat().st_size,
        "path": relative,
        "role": role,
        "sha256": _file_sha256(candidate),
    }


def _git_identity(root: Path) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError("source closure requires a clean Git worktree")
    branch = _git(root, "branch", "--show-current")
    if not branch:
        raise ValueError("source closure requires a named branch")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    epoch_text = _git(root, "show", "-s", "--format=%ct", "HEAD")
    try:
        epoch = int(epoch_text)
    except ValueError as exc:
        raise ValueError("git commit timestamp is invalid") from exc
    return {
        "branch": branch,
        "commit": _required_sha1(commit, "git commit"),
        "commit_tree": _required_sha1(tree, "git tree"),
        "source_date_epoch": epoch,
    }


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        (str(_GIT_EXECUTABLE), "-C", str(root), *args),
        check=False,
        capture_output=True,
        env=_git_environment(),
        timeout=60,
    )
    _require_bounded_bytes(completed.stdout, "git stdout")
    _require_bounded_bytes(completed.stderr, "git stderr")
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError(f"git {' '.join(args)} failed")
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _require_freeze_toolchain() -> None:
    observed_python = f"{platform.python_implementation()} {platform.python_version()}"
    if observed_python != PUBLICATION_FREEZE_PYTHON:
        raise RuntimeError(
            f"source closure requires {PUBLICATION_FREEZE_PYTHON}, found {observed_python}"
        )
    if importlib.metadata.version("build") != "1.2.2.post1":
        raise RuntimeError("source closure requires build==1.2.2.post1")


def _require_freeze_build_system(repository_root: Path) -> None:
    pyproject = _regular_file(repository_root / "pyproject.toml", "pyproject.toml")
    try:
        parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("pyproject.toml is invalid") from exc
    build_system = parsed.get("build-system")
    if build_system != {
        "requires": [PUBLICATION_FREEZE_BUILD_BACKEND],
        "build-backend": "poetry.core.masonry.api",
    }:
        raise ValueError("publication build-system identity differs")


def _validate_cachet_wheel(path: Path) -> None:
    from document_kv_cache.release_bundle import _package_wheel_issues

    issues = _package_wheel_issues(str(path), path.read_bytes(), expected_version="0.2.0")
    if issues:
        raise ValueError("Cachet wheel validation failed: " + "; ".join(issues))


def _validate_sdist(path: Path) -> None:
    if path.name != "cachet_kv-0.2.0.tar.gz":
        raise ValueError("source distribution filename differs")
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("source distribution is empty")
        for member in members:
            _safe_archive_name(member.name, "source distribution")
            if not member.name.startswith("cachet_kv-0.2.0/"):
                raise ValueError("source distribution root differs")
            if member.issym() or member.islnk():
                raise ValueError("source distribution cannot contain links")


def _validate_git_archive(
    path: Path,
    *,
    repository_root: Path,
    commit: str,
    source_date_epoch: int,
) -> None:
    expected_name = f"cachet-{commit}.tar.gz"
    if path.name != expected_name:
        raise ValueError("git source archive filename differs")
    prefix = f"cachet-{commit}/"
    archive = subprocess.run(
        (
            str(_GIT_EXECUTABLE),
            "-C",
            str(repository_root),
            "archive",
            "--format=tar",
            f"--prefix={prefix}",
            commit,
        ),
        check=False,
        capture_output=True,
        env=_git_environment(),
        timeout=120,
    )
    if archive.returncode != 0 or archive.stderr:
        raise RuntimeError("git archive reproduction failed")
    _require_bounded_bytes(
        archive.stdout,
        "git archive",
        max_bytes=PUBLICATION_FREEZE_MAX_ARTIFACT_BYTES,
    )
    compressed = subprocess.run(
        (str(_GZIP_EXECUTABLE), "-n"),
        input=archive.stdout,
        check=False,
        capture_output=True,
        env=_base_subprocess_environment(),
        timeout=120,
    )
    if compressed.returncode != 0 or compressed.stderr:
        raise RuntimeError("deterministic gzip reproduction failed")
    if compressed.stdout != path.read_bytes():
        raise ValueError("git source archive is not the deterministic commit archive")
    with tarfile.open(path, mode="r:gz") as tar:
        for member in tar.getmembers():
            _safe_archive_name(member.name, "git source archive")
            if member.mtime != source_date_epoch:
                raise ValueError("git source archive timestamp differs")


def _parse_hash_locked_requirements(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    requirements: dict[str, int] = {}
    current: str | None = None
    hash_count = 0
    for line in text.splitlines():
        if line and line[0].isalnum() and "==" in line:
            if current is not None:
                if hash_count == 0:
                    raise ValueError(f"runtime requirement {current!r} has no hash")
                requirements[current] = hash_count
            current = line.split("==", 1)[0].lower()
            if current in requirements:
                raise ValueError("runtime lock repeats a requirement")
            hash_count = line.count("--hash=sha256:")
        elif current is not None:
            hash_count += line.count("--hash=sha256:")
    if current is not None:
        if hash_count == 0:
            raise ValueError(f"runtime requirement {current!r} has no hash")
        requirements[current] = hash_count
    return requirements


def _audit_wheel_record(path: Path) -> tuple[int, int]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos if not item.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate members")
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            raise ValueError("wheel must contain one RECORD")
        rows = list(csv.reader(io.StringIO(archive.read(record_names[0]).decode("utf-8"))))
        if len(rows) != len(names):
            raise ValueError("wheel RECORD does not cover every file")
        by_name = {row[0]: row for row in rows if len(row) == 3}
        if set(by_name) != set(names):
            raise ValueError("wheel RECORD paths differ")
        for name in names:
            digest, size = by_name[name][1:]
            if name == record_names[0]:
                if digest or size:
                    raise ValueError("wheel RECORD self-row must be empty")
                continue
            content = archive.read(name)
            expected = "sha256=" + base64.urlsafe_b64encode(
                hashlib.sha256(content).digest()
            ).rstrip(b"=").decode("ascii")
            if digest != expected or size != str(len(content)):
                raise ValueError(f"wheel RECORD mismatch for {name!r}")
        return len(names), len(rows)


def _input_bundle_closure_sha256(path: Path) -> str:
    return _verify_input_bundle_byte_closure(
        _regular_directory(path, "input bundle"),
        expected_sha256=PUBLICATION_FREEZE_INPUT_BUNDLE_SHA256,
    )


def _local_check_record(
    *,
    check_id: str,
    plan_sha256: str,
    checked_at: datetime,
    command: Sequence[str],
    exit_code: int,
    environment: Mapping[str, str],
    inputs: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
    stdout: bytes,
    stderr: bytes,
    tool_identity: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    if check_id not in _LOCAL_CHECK_IDS:
        raise ValueError("unknown local check_id")
    if status not in {"passed", "failed"}:
        raise ValueError("local check status differs")
    if (status == "passed") != (exit_code == 0):
        raise ValueError("local check status and exit code differ")
    record: dict[str, Any] = {
        "check_id": check_id,
        "checked_at_utc": _timestamp(checked_at),
        "command": list(command),
        "environment": dict(environment),
        "exit_code": exit_code,
        "inputs": [dict(item) for item in inputs],
        "plan_sha256": plan_sha256,
        "record_type": GPU_QUALIFICATION_LOCAL_CHECK_RECORD_TYPE,
        "result": dict(result),
        "schema_version": GPU_QUALIFICATION_LOCAL_CHECK_SCHEMA_VERSION,
        "status": status,
        "stderr": _command_output_binding(stderr),
        "stdout": _command_output_binding(stdout),
        "tool_identity": dict(tool_identity),
    }
    _validate_local_check_record(
        record,
        expected_check_id=check_id,
        expected_plan_sha256=plan_sha256,
    )
    return record


def _validate_local_check_record(
    record: Mapping[str, Any],
    *,
    expected_check_id: str,
    expected_plan_sha256: str,
) -> None:
    _require_exact_keys(record, _CHECK_KEYS, "local check")
    if (
        record["record_type"] != GPU_QUALIFICATION_LOCAL_CHECK_RECORD_TYPE
        or record["schema_version"] != GPU_QUALIFICATION_LOCAL_CHECK_SCHEMA_VERSION
        or record["check_id"] != expected_check_id
        or record["plan_sha256"] != expected_plan_sha256
    ):
        raise ValueError("local check identity differs")
    _parse_timestamp(record["checked_at_utc"], expected_check_id)
    command = record["command"]
    if not isinstance(command, list) or not command or any(
        not isinstance(item, str) or not item for item in command
    ):
        raise ValueError("local check command is invalid")
    exit_code = record["exit_code"]
    if type(exit_code) is not int:
        raise ValueError("local check exit_code must be an integer")
    status = record["status"]
    if status not in {"passed", "failed"} or ((status == "passed") != (exit_code == 0)):
        raise ValueError("local check status differs from exit_code")
    if not isinstance(record["inputs"], list):
        raise ValueError("local check inputs must be an array")
    for field_name in (
        "environment",
        "result",
        "stdout",
        "stderr",
        "tool_identity",
    ):
        if not isinstance(record[field_name], Mapping):
            raise ValueError(f"local check {field_name} must be an object")
    _validate_command_output_binding(record["stdout"], "local check stdout")
    _validate_command_output_binding(record["stderr"], "local check stderr")
    payload = _canonical_json_bytes(record, pretty=False)
    if len(payload) > PUBLICATION_FREEZE_MAX_CHECK_RECORD_BYTES:
        raise ValueError("local check record exceeds its byte cap")


def _write_check(root: Path, record: Mapping[str, Any]) -> str:
    check_id = cast(str, record["check_id"])
    path = root / f"{check_id}.json"
    _write_exclusive(path, _canonical_json_bytes(record, pretty=False), check_id)
    reread = _read_canonical_json(path, pretty=False, label=check_id)
    if reread != dict(record):
        raise RuntimeError("local check changed during durable write")
    return _file_sha256(path)


def _run_command(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
) -> CompletedCommand:
    return cast(
        CompletedCommand,
        subprocess.run(
            tuple(command),
            cwd=cwd,
            env=dict(environment),
            check=False,
            capture_output=True,
            timeout=PUBLICATION_FREEZE_COMMAND_TIMEOUT_SECONDS,
        ),
    )


def _preflight_environment(repository_root: Path) -> dict[str, str]:
    config_path = _regular_file(
        repository_root / "pyproject.toml",
        "preflight pyproject.toml",
    )
    env = _base_subprocess_environment()
    env.update(
        {
            "MYPY_CONFIG_FILE": str(config_path),
            "MYPYPATH": "",
            "PYTHONPATH": str((repository_root / "src").resolve()),
            "PYTEST_ADDOPTS": "",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_PLUGINS": "",
            "RUFF_NO_CACHE": "1",
        }
    )
    return env


def _python_api_environment_identity() -> dict[str, str]:
    return {
        "execution_mode": "in_process",
        "subprocess_policy": "package_owned_allowlists",
    }


def _python_api_identity() -> dict[str, Any]:
    module_path = _regular_file(Path(__file__), "publication_freeze module")
    executable = _regular_file(Path(sys.executable).resolve(), "Python executable")
    return {
        "executable_path": str(executable),
        "executable_sha256": _file_sha256(executable),
        "module_path": str(module_path),
        "module_sha256": _file_sha256(module_path),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
    }


def _resolve_executable(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    return _regular_file(candidate, "preflight executable")


def _command_output_binding(content: bytes) -> dict[str, Any]:
    _require_bounded_bytes(content, "command output")
    tail = content[-PUBLICATION_FREEZE_COMMAND_TAIL_BYTES :]
    return {
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "utf8_tail": tail.decode("utf-8", errors="replace"),
    }


def _validate_command_output_binding(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding must be an object")
    _require_exact_keys(
        value,
        frozenset({"byte_count", "sha256", "utf8_tail"}),
        f"{label} binding",
    )
    byte_count = value.get("byte_count")
    tail = value.get("utf8_tail")
    if (
        type(byte_count) is not int
        or byte_count < 0
        or byte_count > PUBLICATION_FREEZE_MAX_COMMAND_OUTPUT_BYTES
        or not isinstance(tail, str)
        or len(tail.encode("utf-8")) > PUBLICATION_FREEZE_COMMAND_TAIL_BYTES * 3
    ):
        raise ValueError(f"{label} binding is invalid")
    _required_sha256(value.get("sha256"), f"{label} sha256")


def _path_binding(path: Path, label: str) -> dict[str, Any]:
    candidate = _absolute_normalized_path(path)
    _require_lstat_ancestors(candidate, label=label, include_leaf=False)
    try:
        observed = os.lstat(candidate)
    except FileNotFoundError as exc:
        raise ValueError(
            f"preflight input {label!r} is not a regular path"
        ) from exc
    if stat.S_ISREG(observed.st_mode):
        candidate = _regular_file(candidate, label)
        return {
            "byte_count": candidate.stat().st_size,
            "label": label,
            "path": str(candidate),
            "sha256": _file_sha256(candidate),
            "type": "file",
        }
    if stat.S_ISDIR(observed.st_mode):
        candidate = _regular_directory(candidate, label)
        tree = _regular_tree(candidate)
        return {
            "file_count": len(tree),
            "label": label,
            "path": str(candidate),
            "sha256": _tree_sha256(candidate, tree),
            "type": "directory",
        }
    raise ValueError(f"preflight input {label!r} is not a regular path")


def _repository_binding(path: Path) -> dict[str, Any]:
    root = _regular_directory(path, "repository_root")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("qualification preflight requires a clean Git worktree")
    identity = _git_identity(root)
    return {
        "branch": identity["branch"],
        "commit": identity["commit"],
        "commit_tree": identity["commit_tree"],
        "label": "repository_root",
        "path": str(root),
        "source_date_epoch": identity["source_date_epoch"],
        "type": "git_repository",
    }


def _validate_path_binding(binding: Mapping[str, Any]) -> None:
    label = binding.get("label")
    path_text = binding.get("path")
    binding_type = binding.get("type")
    if (
        not isinstance(label, str)
        or not label
        or not isinstance(path_text, str)
        or not path_text
        or not Path(path_text).is_absolute()
    ):
        raise ValueError("preflight input binding identity is invalid")
    if binding_type == "file":
        expected_keys = frozenset(
            {"byte_count", "label", "path", "sha256", "type"}
        )
        expected = _path_binding(Path(path_text), label)
    elif binding_type == "directory":
        expected_keys = frozenset(
            {"file_count", "label", "path", "sha256", "type"}
        )
        expected = _path_binding(Path(path_text), label)
    elif binding_type == "git_repository":
        expected_keys = frozenset(
            {
                "branch",
                "commit",
                "commit_tree",
                "label",
                "path",
                "source_date_epoch",
                "type",
            }
        )
        if label != "repository_root":
            raise ValueError("Git repository binding has an invalid label")
        expected = _repository_binding(Path(path_text))
    else:
        raise ValueError("preflight input binding type is invalid")
    _require_exact_keys(binding, expected_keys, f"preflight input {label}")
    if dict(binding) != expected:
        raise ValueError(f"preflight input {label!r} bytes differ")


def _bindings(*items: tuple[Path, str]) -> list[dict[str, Any]]:
    return [_path_binding(path, label) for path, label in items]


def _regular_tree(root: Path) -> tuple[Path, ...]:
    candidate = _regular_directory(root, "tree root")
    files: list[Path] = []
    for path in sorted(candidate.rglob("*")):
        _require_lstat_ancestors(path, label="tree entry", include_leaf=False)
        observed = os.lstat(path)
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"tree contains a symlink: {path}")
        if stat.S_ISREG(observed.st_mode):
            files.append(_regular_file(path, "tree file"))
        elif not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"tree contains an irregular entry: {path}")
    return tuple(files)


def _tree_sha256(root: Path, files: Sequence[Path]) -> str:
    rows = [
        {
            "byte_count": path.stat().st_size,
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
        }
        for path in files
    ]
    return hashlib.sha256(_canonical_json_bytes(rows, pretty=False).rstrip(b"\n")).hexdigest()


def _require_directory_bytes(first: Path, second: Path, label: str) -> None:
    first_root = _regular_directory(first, f"{label} first")
    second_root = _regular_directory(second, f"{label} second")
    if first_root.resolve() == second_root.resolve() or os.path.samefile(
        first_root, second_root
    ):
        raise ValueError(f"{label} requires independent paths")
    first_files = _regular_tree(first_root)
    second_files = _regular_tree(second_root)
    first_rows = {
        path.relative_to(first_root).as_posix(): (
            path.stat().st_size,
            _file_sha256(path),
        )
        for path in first_files
    }
    second_rows = {
        path.relative_to(second_root).as_posix(): (
            path.stat().st_size,
            _file_sha256(path),
        )
        for path in second_files
    }
    if first_rows != second_rows:
        raise ValueError(f"{label} bytes differ")


def _require_independent_bytes(first: Path, second: Path, label: str) -> None:
    first_file = _regular_file(first, f"{label} first")
    second_file = _regular_file(second, f"{label} second")
    if first_file.resolve() == second_file.resolve() or os.path.samefile(
        first_file, second_file
    ):
        raise ValueError(f"{label} requires independent paths")
    if (
        first_file.stat().st_size != second_file.stat().st_size
        or _file_sha256(first_file) != _file_sha256(second_file)
    ):
        raise ValueError(f"{label} bytes differ")


def _require_file_binding(path: Path, item: Mapping[str, Any], label: str) -> None:
    byte_count = item.get("byte_count")
    digest = item.get("sha256")
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError(f"{label} byte_count is invalid")
    _required_sha256(digest, f"{label} sha256")
    if path.stat().st_size != byte_count or _file_sha256(path) != digest:
        raise ValueError(f"{label} bytes differ")


def _read_canonical_json(path: Path, *, pretty: bool, label: str) -> dict[str, Any]:
    value = _read_canonical_json_value(path, label=label, pretty=pretty)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return cast(dict[str, Any], value)


def _read_canonical_json_value(
    path: Path,
    *,
    label: str,
    pretty: bool = False,
) -> Any:
    candidate = _regular_file(path, label)
    content = candidate.read_bytes()
    if len(content) > PUBLICATION_FREEZE_MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds its JSON byte cap")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError(f"{label} must be one newline-terminated JSON value")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if content != _canonical_json_bytes(value, pretty=pretty):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _read_json_value(path: Path, label: str) -> Any:
    candidate = _regular_file(path, label)
    content = candidate.read_bytes()
    if len(content) > PUBLICATION_FREEZE_MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds its JSON byte cap")
    try:
        return json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc


def _canonical_json_bytes(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    normalized = dict(record)
    normalized["closed_record_sha256"] = ""
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_closed_digest(record: Mapping[str, Any], label: str) -> None:
    digest = _required_sha256(record.get("closed_record_sha256"), f"{label} digest")
    if digest != _closed_record_sha256(record):
        raise ValueError(f"{label} closed_record_sha256 is invalid")


def _write_exclusive(path: Path, content: bytes, label: str) -> None:
    candidate = _absolute_normalized_path(path)
    parent, parent_descriptor, parent_identity = _open_verified_directory(
        candidate.parent,
        f"{label} parent",
    )
    if candidate == parent:
        os.close(parent_descriptor)
        raise ValueError(f"{label} cannot replace its parent directory")
    descriptor: int | None = None
    opened: os.stat_result | None = None
    try:
        try:
            os.stat(candidate.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"{label} already exists: {candidate}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(
            candidate.name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"{label} did not create a regular file")
        observed = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_dev != opened.st_dev
            or observed.st_ino != opened.st_ino
        ):
            raise RuntimeError(f"{label} path changed immediately after create")
        _require_open_directory_path(
            parent,
            parent_identity,
            f"{label} parent",
        )
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError(f"{label} write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        final_opened = os.fstat(descriptor)
        final_observed = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            final_opened.st_dev != opened.st_dev
            or final_opened.st_ino != opened.st_ino
            or not stat.S_ISREG(final_observed.st_mode)
            or final_observed.st_dev != opened.st_dev
            or final_observed.st_ino != opened.st_ino
        ):
            raise RuntimeError(f"{label} path changed during durable write")
        _require_open_directory_path(
            parent,
            parent_identity,
            f"{label} parent",
        )
        os.fsync(parent_descriptor)
    except BaseException:
        if opened is not None:
            try:
                observed = os.stat(
                    candidate.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                if (
                    observed.st_dev == opened.st_dev
                    and observed.st_ino == opened.st_ino
                ):
                    os.unlink(candidate.name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _regular_file(path: Path, label: str) -> Path:
    candidate = _absolute_normalized_path(path)
    _require_lstat_ancestors(candidate, label=label, include_leaf=False)
    try:
        observed = os.lstat(candidate)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} must be one regular file: {candidate}") from exc
    if stat.S_ISLNK(observed.st_mode):
        raise ValueError(f"{label} cannot traverse a symlink: {candidate}")
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{label} must be one regular file: {candidate}")
    return candidate


def _regular_directory(path: Path, label: str) -> Path:
    candidate = _absolute_normalized_path(path)
    _require_lstat_ancestors(candidate, label=label, include_leaf=False)
    try:
        observed = os.lstat(candidate)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} must be one regular directory: {candidate}") from exc
    if stat.S_ISLNK(observed.st_mode):
        raise ValueError(f"{label} cannot traverse a symlink: {candidate}")
    if not stat.S_ISDIR(observed.st_mode):
        raise ValueError(f"{label} must be one regular directory: {candidate}")
    return candidate


def _create_directory_exclusive(path: Path, label: str) -> Path:
    candidate = _absolute_normalized_path(path)
    parent, parent_descriptor, parent_identity = _open_verified_directory(
        candidate.parent,
        f"{label} parent",
    )
    if candidate == parent:
        os.close(parent_descriptor)
        raise ValueError(f"{label} cannot replace its parent directory")
    created: os.stat_result | None = None
    try:
        try:
            os.mkdir(candidate.name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise FileExistsError(f"{label} already exists: {candidate}") from exc
        created = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(created.st_mode):
            raise RuntimeError(f"{label} did not create a regular directory")
        _require_open_directory_path(
            parent,
            parent_identity,
            f"{label} parent",
        )
        os.fsync(parent_descriptor)
        _require_open_directory_path(
            parent,
            parent_identity,
            f"{label} parent",
        )
        observed = os.lstat(candidate)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_dev != created.st_dev
            or observed.st_ino != created.st_ino
        ):
            raise RuntimeError(f"{label} path changed during durable create")
        return candidate
    except BaseException:
        if created is not None:
            try:
                observed = os.stat(
                    candidate.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                if (
                    observed.st_dev == created.st_dev
                    and observed.st_ino == created.st_ino
                ):
                    os.rmdir(candidate.name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
        raise
    finally:
        os.close(parent_descriptor)


def _absolute_normalized_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.normpath(str(path))))


def _require_lstat_ancestors(
    path: Path,
    *,
    label: str,
    include_leaf: bool,
) -> None:
    candidate = _absolute_normalized_path(path)
    ancestors = tuple(reversed(candidate.parents))
    if include_leaf:
        ancestors = (*ancestors, candidate)
    for ancestor in ancestors:
        try:
            observed = os.lstat(ancestor)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"{label} cannot traverse a symlink: {ancestor}")
        if ancestor != candidate and not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"{label} has a non-directory ancestor: {ancestor}")


def _fsync_directory(path: Path) -> None:
    candidate, descriptor, identity = _open_verified_directory(
        path,
        "fsync directory",
    )
    try:
        os.fsync(descriptor)
        _require_open_directory_path(candidate, identity, "fsync directory")
    finally:
        os.close(descriptor)


def _open_verified_directory(
    path: Path,
    label: str,
) -> tuple[Path, int, os.stat_result]:
    candidate = _regular_directory(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(candidate, flags)
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISDIR(identity.st_mode):
            raise ValueError(f"{label} must be one regular directory: {candidate}")
        _require_open_directory_path(candidate, identity, label)
    except BaseException:
        os.close(descriptor)
        raise
    return candidate, descriptor, identity


def _require_open_directory_path(
    path: Path,
    identity: os.stat_result,
    label: str,
) -> None:
    _require_lstat_ancestors(path, label=label, include_leaf=True)
    observed = os.lstat(path)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_dev != identity.st_dev
        or observed.st_ino != identity.st_ino
    ):
        raise RuntimeError(f"{label} path changed after open")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_copy(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _required_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    observed = value.get(field_name)
    if not isinstance(observed, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], observed)


def _mapping_sequence(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{label} must be an array of objects")
    return tuple(cast(Mapping[str, Any], item) for item in value)


def _required_string(value: Mapping[str, Any], field_name: str) -> str:
    observed = value.get(field_name)
    if not isinstance(observed, str) or not observed:
        raise ValueError(f"{field_name} must be a non-empty string")
    return observed


def _required_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _required_sha1(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-1 digest")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{label} does not use the closed schema")


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{label} must be a canonical relative path")
    return value


def _safe_archive_name(value: str, label: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} contains an unsafe member")


def _require_bounded_command(completed: CompletedCommand, label: str) -> None:
    if type(completed.returncode) is not int:
        raise RuntimeError(f"{label} returned an invalid exit code")
    _require_bounded_bytes(completed.stdout, f"{label} stdout")
    _require_bounded_bytes(completed.stderr, f"{label} stderr")


def _require_exact_pytest_completion(stdout: bytes) -> None:
    try:
        output = stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError("unit-test output is not UTF-8") from exc
    expected = re.compile(
        rf"\n{PUBLICATION_FREEZE_EXPECTED_TEST_COUNT} passed in [^\n]+\n\Z"
    )
    if expected.search(output) is None:
        raise RuntimeError(
            "unit-test completion differs from the exact no-skip publication suite"
        )


def _require_bounded_bytes(
    content: bytes,
    label: str,
    *,
    max_bytes: int = PUBLICATION_FREEZE_MAX_COMMAND_OUTPUT_BYTES,
) -> None:
    if not isinstance(content, bytes):
        raise TypeError(f"{label} must be bytes")
    if len(content) > max_bytes:
        raise RuntimeError(f"{label} exceeds its byte cap")


def _bounded_error(exc: BaseException) -> dict[str, str]:
    text = str(exc)
    if len(text.encode("utf-8", errors="replace")) > PUBLICATION_FREEZE_COMMAND_TAIL_BYTES:
        text = text[: PUBLICATION_FREEZE_COMMAND_TAIL_BYTES] + "...[truncated]"
    return {"message": text, "type": type(exc).__name__}


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("preflight timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} timestamp must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} timestamp is invalid") from exc
    if _timestamp(parsed) != value:
        raise ValueError(f"{label} timestamp is not canonical")
    return parsed


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _source_inputs_from_args(args: argparse.Namespace) -> PublicationSourceClosureInputs:
    return PublicationSourceClosureInputs(
        repository_root=Path(args.repository_root),
        artifact_output_root=Path(args.artifact_output_root),
        runtime_lock=Path(args.runtime_lock),
        runtime_lock_input=Path(args.runtime_lock_input),
        campaign_plan=Path(args.campaign_plan),
        latency_handoff_plan=Path(args.latency_handoff_plan),
        full_score_inventory=Path(args.full_score_inventory),
        full_score_shard_plan=Path(args.full_score_shard_plan),
    )


def _preflight_inputs_from_args(
    args: argparse.Namespace,
) -> GPUQualificationLocalPreflightInputs:
    return GPUQualificationLocalPreflightInputs(
        **{
            field_name: Path(getattr(args, field_name))
            for field_name in GPUQualificationLocalPreflightInputs.__dataclass_fields__
        }
    )


def _add_path_arguments(
    parser: argparse.ArgumentParser,
    field_names: Sequence[str],
) -> None:
    for field_name in field_names:
        parser.add_argument(f"--{field_name.replace('_', '-')}", required=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one non-cloud publication freeze operation."""

    parser = argparse.ArgumentParser(
        description=(
            "Build source closure, validate frozen latency-plan semantics, or "
            "execute GPU qualification preflight."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    source_parser = subparsers.add_parser("source-closure")
    _add_path_arguments(
        source_parser,
        tuple(PublicationSourceClosureInputs.__dataclass_fields__),
    )
    source_parser.add_argument("--output-json", required=True)
    latency_parser = subparsers.add_parser("latency-plan-semantic-check")
    latency_parser.add_argument("--plan-json", required=True)
    latency_parser.add_argument("--prepared-input-dir", required=True)
    latency_parser.add_argument("--runtime-lock", required=True)
    preflight_parser = subparsers.add_parser("qualification-preflight")
    required_preflight = tuple(
        field_name
        for field_name, field in (
            GPUQualificationLocalPreflightInputs.__dataclass_fields__.items()
        )
        if field.default is MISSING and field.default_factory is MISSING
    )
    _add_path_arguments(preflight_parser, required_preflight)
    preflight_parser.add_argument(
        "--python-executable",
        default=str(_DEFAULT_PYTHON_EXECUTABLE),
    )
    preflight_parser.add_argument(
        "--ruff-executable",
        default=str(_DEFAULT_RUFF_EXECUTABLE),
    )
    preflight_parser.add_argument(
        "--mypy-executable",
        default=str(_DEFAULT_MYPY_EXECUTABLE),
    )
    preflight_parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "latency-plan-semantic-check":
        attestation = _run_publication_latency_semantic_check(
            plan_path=Path(args.plan_json),
            prepared_input_dir=Path(args.prepared_input_dir),
            runtime_lock_path=Path(args.runtime_lock),
        )
        sys.stdout.buffer.write(_canonical_json_bytes(attestation, pretty=False))
        return 0
    if args.command == "source-closure":
        source_inputs = _source_inputs_from_args(args)
        record = build_publication_source_closure(source_inputs)
        write_publication_source_closure_json(
            record,
            args.output_json,
            repository_root=source_inputs.repository_root,
            artifact_root=source_inputs.artifact_output_root,
        )
        print(
            json.dumps(
                {
                    "closed_record_sha256": record["closed_record_sha256"],
                    "file_sha256": _file_sha256(Path(args.output_json)),
                    "output_json": args.output_json,
                },
                sort_keys=True,
            )
        )
        return 0
    preflight_inputs = _preflight_inputs_from_args(args)
    evidence = run_gpu_qualification_local_preflight(
        preflight_inputs,
        args.output_root,
    )
    print(
        json.dumps(
            {
                "closed_record_sha256": evidence["closed_record_sha256"],
                "output_root": args.output_root,
                "plan_sha256": evidence["plan_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "GPU_QUALIFICATION_LOCAL_CHECK_RECORD_TYPE",
    "GPU_QUALIFICATION_LOCAL_CHECK_SCHEMA_VERSION",
    "GPUQualificationLocalPreflightInputs",
    "PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_CLOSED_RECORD_SHA256",
    "PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_FILE_SHA256",
    "PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_ID",
    "PUBLICATION_FREEZE_LATENCY_HANDOFF_WORKERS_SHA256",
    "PUBLICATION_SOURCE_CLOSURE_RECORD_TYPE",
    "PUBLICATION_SOURCE_CLOSURE_SCHEMA_VERSION",
    "PublicationSourceClosureInputs",
    "build_publication_source_closure",
    "main",
    "run_gpu_qualification_local_preflight",
    "validate_gpu_qualification_local_preflight_bundle",
    "validate_publication_source_closure_record",
    "write_publication_source_closure_json",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
