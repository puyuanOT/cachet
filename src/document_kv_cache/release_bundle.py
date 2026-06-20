"""Release bundle packaging for Document KV Cache artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from document_kv_cache.release_evidence import (
    RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE,
    RELEASE_EVIDENCE_RECORD_TYPE,
    REQUIRED_ENGINE_PROBE_BACKENDS,
    evaluate_release_evidence,
)
from document_kv_cache.benchmark_plan_executor import (
    BENCHMARK_PLAN_EXECUTION_RECORD_TYPE,
    BENCHMARK_PLAN_SOURCE_RECORD_TYPE,
)
from document_kv_cache.databricks_runs import (
    DATABRICKS_RUN_STATUS_RECORD_TYPE,
    DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE,
)
from document_kv_cache.github_governance import GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE
from document_kv_cache.native_probe_factories import (
    NATIVE_PROBE_FACTORIES_RECORD_TYPE,
    SGLANG_NATIVE_PROBE_FACTORY,
    VLLM_NATIVE_PROBE_FACTORY,
)
from document_kv_cache.pr_evidence import _PR_EVIDENCE_RECORD_KEYS, PR_EVIDENCE_RECORD_TYPE, evaluate_pr_evidence_record
from document_kv_cache.repository_hygiene import (
    FORBIDDEN_TRACKED_ARTIFACT_PATTERNS,
    REPOSITORY_HYGIENE_RECORD_TYPE,
    REQUIRED_GITIGNORE_PATTERNS,
)
from document_kv_cache.serving_env import serving_environment_profile, serving_environment_profile_to_record
from document_kv_cache.storage import local_path


__all__ = [
    "RELEASE_BUNDLE_RECORD_TYPE",
    "RELEASE_BUNDLE_MANIFEST_FILENAME",
    "RELEASE_BUNDLE_ARTIFACT_ROLES",
    "ReleaseBundleArtifact",
    "ReleaseBundle",
    "build_release_bundle",
    "release_bundle_to_record",
    "write_release_bundle_manifest_json",
    "main",
]


RELEASE_BUNDLE_RECORD_TYPE = "document_kv.release_bundle.v1"
RELEASE_BUNDLE_MANIFEST_FILENAME = "manifest.json"
RELEASE_BUNDLE_PACKAGE_NAME = "document-kv-cache"
RELEASE_BUNDLE_PACKAGE_LICENSE_EXPRESSION = "Apache-2.0"
RELEASE_BUNDLE_PACKAGE_LICENSE_FILE = "LICENSE"
RELEASE_BUNDLE_PACKAGE_TYPED_MARKER_PATHS = (
    "document_kv_cache/py.typed",
    "restaurant_kv_serving/py.typed",
)
RELEASE_BUNDLE_ARTIFACT_ROLES = (
    "v1_benchmark",
    "storage_benchmark",
    "engine_probe",
    "engine_connector_actions",
    "release_evidence",
    "preflight",
    "plan_execution",
    "databricks_run_status",
    "package_wheel",
    "pr_evidence",
    "github_governance",
    "repository_hygiene",
    "native_probe_factories",
)
STRICT_V1_RELEASE_REQUIRED_ARTIFACTS = (
    ("release_evidence", 1, "release evidence sidecar"),
    ("preflight", 1, "preflight sidecar"),
    ("engine_connector_actions", 2, "vLLM/SGLang connector action sidecars"),
    ("plan_execution", 1, "benchmark plan execution sidecar"),
    ("databricks_run_status", 1, "Databricks run-status sidecar"),
    ("package_wheel", 1, "tested package wheel"),
    ("pr_evidence", 1, "PR evidence sidecar"),
    ("github_governance", 1, "GitHub governance sidecar"),
    ("repository_hygiene", 1, "repository hygiene sidecar"),
    ("native_probe_factories", 1, "native probe factory diagnostics sidecar"),
)
STRICT_V1_RELEASE_REQUIRED_DATABRICKS_PURPOSES = (
    ("document-kv-v1-benchmark", "V1 benchmark Databricks run-status evidence"),
    ("document-kv-storage-benchmark", "storage-reader benchmark Databricks run-status evidence"),
    ("document-kv-engine-probe", "native engine probe Databricks run-status evidence"),
)
STRICT_V1_RELEASE_REQUIRED_NATIVE_PROBE_FACTORY_SUPPORT = (
    ("vllm", "vLLM native probe factory support"),
    ("sglang", "SGLang native probe factory support"),
)
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_WHEEL_FILENAME_RE = re.compile(
    r"^(?P<distribution>[A-Za-z0-9_.]+)-"
    r"(?P<version>[A-Za-z0-9_.!+]+)"
    r"(?:-(?P<build>[0-9][A-Za-z0-9_.]*))?"
    r"-(?P<python_tag>[A-Za-z0-9_.]+)"
    r"-(?P<abi_tag>[A-Za-z0-9_.]+)"
    r"-(?P<platform_tag>[A-Za-z0-9_.]+)\.whl$"
)
_DATABRICKS_RUN_STATUS_WRAPPER_KEYS = frozenset({"ok", "action", "summary"})
_DATABRICKS_RUN_STATUS_KEYS = frozenset(
    {
        "record_type",
        "run_id",
        "run_name",
        "run_page_url",
        "life_cycle_state",
        "result_state",
        "state_message",
        "start_time",
        "end_time",
        "terminal",
        "succeeded",
        "active_task_key",
        "task_count",
        "tasks",
        "cluster_id",
        "submit_payload",
    }
)
_DATABRICKS_RUN_STATUS_TASK_KEYS = frozenset(
    {
        "task_key",
        "run_id",
        "life_cycle_state",
        "result_state",
        "state_message",
        "cluster_id",
        "start_time",
        "end_time",
    }
)
_DATABRICKS_SUBMIT_PAYLOAD_KEYS = frozenset(
    {
        "record_type",
        "source_path",
        "sha256",
        "run_name",
        "task_count",
        "task_keys",
        "tasks",
        "node_type_ids",
        "driver_node_type_ids",
        "spark_versions",
        "data_security_modes",
        "single_node",
        "aws_g5_node_type",
    }
)
_DATABRICKS_SUBMIT_PAYLOAD_TASK_KEYS = frozenset(
    {
        "task_key",
        "node_type_id",
        "driver_node_type_id",
        "spark_version",
        "data_security_mode",
        "num_workers",
        "single_node",
        "purpose",
    }
)
_RELEASE_EVIDENCE_SIDECAR_KEYS = frozenset(
    {
        "record_type",
        "ok",
        "v1_benchmark_ok",
        "storage_benchmark_ok",
        "engine_probe_backends",
        "missing_engine_probe_backends",
        "duplicate_engine_probe_backends",
        "invalid_engine_probe_records",
        "engine_action_backends",
        "missing_engine_action_backends",
        "duplicate_engine_action_backends",
        "invalid_engine_action_records",
        "artifact_sources",
        "issues",
    }
)
_PREFLIGHT_SIDECAR_KEYS = frozenset(
    {
        "record_type",
        "ok",
        "required_engine_probe_backends",
        "required_engine_action_backends",
        "missing_paths",
        "unreadable_paths",
        "invalid_record_type_paths",
        "missing_engine_probe_backends",
        "missing_engine_action_backends",
        "input_files",
        "issues",
    }
)
_BENCHMARK_PLAN_EXECUTION_KEYS = frozenset({"record_type", "ok", "commands", "plan_source"})
_BENCHMARK_PLAN_EXECUTION_COMMAND_KEYS = frozenset({"name", "argv", "returncode", "skipped", "error"})
_BENCHMARK_PLAN_SOURCE_KEYS = frozenset(
    {
        "record_type",
        "path",
        "driver_path",
        "size_bytes",
        "sha256",
        "plan_version",
        "suite_id",
        "model_id",
        "hardware_target",
        "command_count",
    }
)
_GITHUB_GOVERNANCE_WRAPPER_KEYS = frozenset({"ok", "summary"})
_GITHUB_GOVERNANCE_SUMMARY_KEYS = frozenset(
    {
        "record_type",
        "ok",
        "repository",
        "default_branch",
        "branch",
        "private",
        "visibility",
        "archived",
        "disabled",
        "description",
        "homepage",
        "topics",
        "branch_protection",
        "open_pull_requests",
        "issues",
    }
)
_GITHUB_BRANCH_PROTECTION_KEYS = frozenset(
    {
        "enabled",
        "required_status_checks",
        "required_pull_request_reviews",
        "required_linear_history",
        "required_conversation_resolution",
        "enforce_admins",
        "allow_force_pushes",
        "allow_deletions",
    }
)
_GITHUB_REQUIRED_STATUS_CHECKS_KEYS = frozenset({"strict", "contexts"})
_GITHUB_REQUIRED_PULL_REQUEST_REVIEWS_KEYS = frozenset(
    {
        "dismiss_stale_reviews",
        "require_last_push_approval",
        "required_approving_review_count",
    }
)
_GITHUB_REQUIRED_REPOSITORY_DESCRIPTION_TERM = "cachet"
_GITHUB_REQUIRED_REPOSITORY_TOPICS = ("cachet", "kv-cache")
_GITHUB_OPEN_PULL_REQUESTS_KEYS = frozenset(
    {
        "checked",
        "total_count",
        "allowed_numbers",
        "allowed_count",
        "allowed",
        "unexpected_count",
        "unexpected",
        "truncated",
    }
)
_GITHUB_PULL_REQUEST_SUMMARY_KEYS = frozenset(
    {
        "number",
        "title",
        "draft",
        "html_url",
        "head_ref",
        "base_ref",
    }
)
_REPOSITORY_HYGIENE_KEYS = frozenset(
    {
        "record_type",
        "ok",
        "repository_root",
        "tracked_path_count",
        "required_gitignore_patterns",
        "missing_gitignore_patterns",
        "forbidden_tracked_artifact_patterns",
        "forbidden_tracked_paths",
        "forbidden_untracked_paths",
        "dirty_tracked_paths",
        "documentation_checked_directory_paths",
        "missing_directory_documentation_paths",
        "untracked_path_count",
        "issues",
    }
)
_NATIVE_PROBE_FACTORIES_KEYS = frozenset({"record_type", "factories"})
_NATIVE_PROBE_FACTORY_KEYS = frozenset(
    {
        "backend",
        "factory_path",
        "package_name",
        "package_importable",
        "package_version",
        "serving_environment_profile",
        "supported",
        "reason",
    }
)
_BUILTIN_NATIVE_PROBE_FACTORY_PATHS = {
    "vllm": VLLM_NATIVE_PROBE_FACTORY,
    "sglang": SGLANG_NATIVE_PROBE_FACTORY,
}


@dataclass(frozen=True, slots=True)
class _PreparedReleaseBundleArtifact:
    role: str
    source_path: str
    index: int
    payload: bytes
    record: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class _WheelPackageIdentity:
    package_name: str
    package_version: str


@dataclass(frozen=True, slots=True)
class ReleaseBundleArtifact:
    role: str
    source_path: str
    bundled_path: str
    size_bytes: int
    sha256: str
    record_type: str | None = None
    backend: str | None = None
    package_name: str | None = None
    package_version: str | None = None

    def __post_init__(self) -> None:
        if self.role not in RELEASE_BUNDLE_ARTIFACT_ROLES:
            raise ValueError(f"Unsupported release bundle artifact role {self.role!r}")
        for field_name in ("source_path", "bundled_path", "sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be non-empty")
        if not _SHA256_HEX_RE.fullmatch(self.sha256):
            raise ValueError("sha256 must be a 64-character lowercase hex digest")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        if self.record_type is not None and (not isinstance(self.record_type, str) or not self.record_type):
            raise ValueError("record_type must be non-empty when provided")
        if self.backend is not None and self.role not in ("engine_probe", "engine_connector_actions"):
            raise ValueError("backend can only be set for engine backend artifacts")
        if self.backend is not None and self.backend not in REQUIRED_ENGINE_PROBE_BACKENDS:
            raise ValueError(f"Unsupported artifact backend {self.backend!r}")
        if (self.package_name is None) != (self.package_version is None):
            raise ValueError("package_name and package_version must be provided together")
        if self.package_name is not None and self.role != "package_wheel":
            raise ValueError("package identity can only be set for package wheel artifacts")
        for field_name in ("package_name", "package_version"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field_name} must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class ReleaseBundle:
    output_dir: str
    manifest_path: str
    artifacts: tuple[ReleaseBundleArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be non-empty")
        if not isinstance(self.manifest_path, str) or not self.manifest_path:
            raise ValueError("manifest_path must be non-empty")
        if any(not isinstance(artifact, ReleaseBundleArtifact) for artifact in self.artifacts):
            raise TypeError("artifacts entries must be ReleaseBundleArtifact")
        object.__setattr__(self, "artifacts", tuple(self.artifacts))


def build_release_bundle(
    *,
    v1_benchmark_json: str | Path,
    storage_benchmark_json: str | Path,
    output_dir: str | Path,
    engine_probe_jsons: Sequence[str | Path] = (),
    engine_actions_jsons: Sequence[str | Path] = (),
    release_evidence_json: str | Path | None = None,
    preflight_json: str | Path | None = None,
    plan_execution_jsons: Sequence[str | Path] = (),
    databricks_run_status_jsons: Sequence[str | Path] = (),
    package_wheel: str | Path | None = None,
    pr_evidence_jsons: Sequence[str | Path] = (),
    github_governance_json: str | Path | None = None,
    repository_hygiene_json: str | Path | None = None,
    native_probe_factories_jsons: Sequence[str | Path] = (),
    require_complete_v1: bool = False,
    overwrite: bool = False,
) -> ReleaseBundle:
    if type(require_complete_v1) is not bool:
        raise ValueError("require_complete_v1 must be boolean")
    bundle_dir = local_path(str(output_dir))
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_dir / RELEASE_BUNDLE_MANIFEST_FILENAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Release bundle manifest already exists: {manifest_path}")
    sources = _release_bundle_sources(
        v1_benchmark_json=v1_benchmark_json,
        storage_benchmark_json=storage_benchmark_json,
        engine_probe_jsons=engine_probe_jsons,
        engine_actions_jsons=engine_actions_jsons,
        release_evidence_json=release_evidence_json,
        preflight_json=preflight_json,
        plan_execution_jsons=plan_execution_jsons,
        databricks_run_status_jsons=databricks_run_status_jsons,
        package_wheel=package_wheel,
        pr_evidence_jsons=pr_evidence_jsons,
        github_governance_json=github_governance_json,
        repository_hygiene_json=repository_hygiene_json,
        native_probe_factories_jsons=native_probe_factories_jsons,
    )
    prepared_artifacts = tuple(
        _prepare_release_bundle_artifact(role=role, source_path=source_path, index=index)
        for index, (role, source_path) in enumerate(sources)
    )
    expected_package_version = _release_bundle_expected_package_version() if require_complete_v1 else None
    if require_complete_v1 and expected_package_version is None:
        raise ValueError(
            "Strict V1 release bundle requires current project package version metadata "
            "from pyproject.toml or the installed package"
        )
    if require_complete_v1:
        _validate_strict_v1_release_bundle_completeness(prepared_artifacts)
    _validate_release_bundle_inputs(
        prepared_artifacts,
        expected_package_version=expected_package_version,
    )
    if require_complete_v1:
        _validate_strict_v1_databricks_purpose_coverage(prepared_artifacts)
        _validate_strict_v1_native_probe_factory_support(prepared_artifacts)
    _preflight_release_bundle_destinations(
        prepared_artifacts=prepared_artifacts,
        bundle_dir=bundle_dir,
        overwrite=overwrite,
    )
    artifacts = tuple(
        _copy_release_bundle_artifact(
            prepared=prepared,
            bundle_dir=bundle_dir,
            overwrite=overwrite,
        )
        for prepared in prepared_artifacts
    )
    bundle = ReleaseBundle(
        output_dir=str(bundle_dir),
        manifest_path=str(manifest_path),
        artifacts=artifacts,
    )
    write_release_bundle_manifest_json(bundle, manifest_path)
    return bundle


def release_bundle_to_record(bundle: ReleaseBundle) -> dict[str, Any]:
    return {
        "record_type": RELEASE_BUNDLE_RECORD_TYPE,
        "ok": True,
        "output_dir": bundle.output_dir,
        "manifest_path": bundle.manifest_path,
        "artifact_count": len(bundle.artifacts),
        "artifacts": [_artifact_to_record(artifact) for artifact in bundle.artifacts],
    }


def write_release_bundle_manifest_json(bundle: ReleaseBundle, path: str | Path) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(release_bundle_to_record(bundle), indent=2, sort_keys=True) + "\n")


def _release_bundle_sources(
    *,
    v1_benchmark_json: str | Path,
    storage_benchmark_json: str | Path,
    engine_probe_jsons: Sequence[str | Path],
    engine_actions_jsons: Sequence[str | Path],
    release_evidence_json: str | Path | None,
    preflight_json: str | Path | None,
    plan_execution_jsons: Sequence[str | Path],
    databricks_run_status_jsons: Sequence[str | Path],
    package_wheel: str | Path | None,
    pr_evidence_jsons: Sequence[str | Path],
    github_governance_json: str | Path | None,
    repository_hygiene_json: str | Path | None,
    native_probe_factories_jsons: Sequence[str | Path],
) -> tuple[tuple[str, str | Path], ...]:
    sources: list[tuple[str, str | Path]] = [
        ("v1_benchmark", v1_benchmark_json),
        ("storage_benchmark", storage_benchmark_json),
    ]
    sources.extend(("engine_probe", path) for path in engine_probe_jsons)
    sources.extend(("engine_connector_actions", path) for path in engine_actions_jsons)
    if release_evidence_json is not None:
        sources.append(("release_evidence", release_evidence_json))
    if preflight_json is not None:
        sources.append(("preflight", preflight_json))
    sources.extend(("plan_execution", path) for path in plan_execution_jsons)
    sources.extend(("databricks_run_status", path) for path in databricks_run_status_jsons)
    if package_wheel is not None:
        sources.append(("package_wheel", package_wheel))
    sources.extend(("pr_evidence", path) for path in pr_evidence_jsons)
    if github_governance_json is not None:
        sources.append(("github_governance", github_governance_json))
    if repository_hygiene_json is not None:
        sources.append(("repository_hygiene", repository_hygiene_json))
    sources.extend(("native_probe_factories", path) for path in native_probe_factories_jsons)
    return tuple(sources)


def _copy_release_bundle_artifact(
    *,
    prepared: _PreparedReleaseBundleArtifact,
    bundle_dir: Path,
    overwrite: bool,
) -> ReleaseBundleArtifact:
    local_source_path = local_path(prepared.source_path)
    destination = _bundle_destination(bundle_dir, prepared)
    if destination.exists() and not overwrite and not _same_path(local_source_path, destination):
        raise FileExistsError(f"Release bundle artifact already exists: {destination}")
    if not _same_path(local_source_path, destination):
        destination.write_bytes(prepared.payload)
    wheel_identity = _package_wheel_identity(prepared.payload) if prepared.role == "package_wheel" else None
    return ReleaseBundleArtifact(
        role=prepared.role,
        source_path=prepared.source_path,
        bundled_path=destination.relative_to(bundle_dir).as_posix(),
        size_bytes=len(prepared.payload),
        sha256=hashlib.sha256(prepared.payload).hexdigest(),
        record_type=_artifact_record_type(prepared),
        backend=(
            _optional_backend(prepared.record.get("backend"))
            if prepared.role in ("engine_probe", "engine_connector_actions") and prepared.record is not None
            else None
        ),
        package_name=wheel_identity.package_name if wheel_identity is not None else None,
        package_version=wheel_identity.package_version if wheel_identity is not None else None,
    )


def _prepare_release_bundle_artifact(
    *,
    role: str,
    source_path: str | Path,
    index: int,
) -> _PreparedReleaseBundleArtifact:
    source_path_text = str(source_path)
    payload = local_path(source_path_text).read_bytes()
    return _PreparedReleaseBundleArtifact(
        role=role,
        source_path=source_path_text,
        index=index,
        payload=payload,
        record=_artifact_record(role, payload, source_path_text),
    )


def _validate_release_bundle_inputs(
    artifacts: Sequence[_PreparedReleaseBundleArtifact],
    *,
    expected_package_version: str | None = None,
) -> None:
    v1_record = _single_record_for_role(artifacts, "v1_benchmark")
    storage_record = _single_record_for_role(artifacts, "storage_benchmark")
    engine_probe_records = tuple(artifact.record for artifact in artifacts if artifact.role == "engine_probe")
    engine_action_records = tuple(
        artifact.record for artifact in artifacts if artifact.role == "engine_connector_actions"
    )
    evidence = evaluate_release_evidence(
        v1_record,
        storage_record,
        engine_probe_records=engine_probe_records,
        engine_action_records=engine_action_records,
    )
    release_input_artifacts = tuple(
        artifact
        for artifact in artifacts
        if artifact.role in ("v1_benchmark", "storage_benchmark", "engine_probe", "engine_connector_actions")
    )
    issues = list(evidence.issues)
    for artifact in artifacts:
        if artifact.role == "release_evidence":
            if artifact.record is None:
                issues.append("release evidence sidecar must be JSON")
                continue
            issues.extend(_release_evidence_sidecar_issues(artifact.record, release_input_artifacts))
        elif artifact.role == "preflight":
            if artifact.record is None:
                issues.append("preflight sidecar must be JSON")
                continue
            issues.extend(_preflight_sidecar_issues(artifact.record, release_input_artifacts))
        elif artifact.role == "pr_evidence":
            if artifact.record is None:
                issues.append("PR evidence sidecar must be JSON")
                continue
            issues.extend(_pr_evidence_sidecar_issues(artifact.record))
        elif artifact.role == "github_governance":
            if artifact.record is None:
                issues.append("GitHub governance sidecar must be JSON")
                continue
            issues.extend(_github_governance_sidecar_issues(artifact.record))
        elif artifact.role == "repository_hygiene":
            if artifact.record is None:
                issues.append("repository hygiene sidecar must be JSON")
                continue
            issues.extend(_repository_hygiene_sidecar_issues(artifact.record))
        elif artifact.role == "plan_execution":
            if artifact.record is None:
                issues.append("benchmark plan execution sidecar must be JSON")
                continue
            issues.extend(_plan_execution_sidecar_issues(artifact.record))
        elif artifact.role == "databricks_run_status":
            if artifact.record is None:
                issues.append("Databricks run status sidecar must be JSON")
                continue
            issues.extend(_databricks_run_status_sidecar_issues(artifact.record))
        elif artifact.role == "package_wheel":
            issues.extend(
                _package_wheel_issues(
                    artifact.source_path,
                    artifact.payload,
                    expected_version=expected_package_version,
                )
            )
        elif artifact.role == "native_probe_factories":
            if artifact.record is None:
                issues.append("native probe factories sidecar must be JSON")
                continue
            issues.extend(_native_probe_factories_sidecar_issues(artifact.record))
    if issues:
        raise ValueError(f"Release bundle inputs are not release-ready: {'; '.join(issues)}")


def _validate_strict_v1_release_bundle_completeness(
    artifacts: Sequence[_PreparedReleaseBundleArtifact],
) -> None:
    role_counts = {role: 0 for role in RELEASE_BUNDLE_ARTIFACT_ROLES}
    for artifact in artifacts:
        role_counts[artifact.role] = role_counts.get(artifact.role, 0) + 1
    missing = [
        label
        for role, minimum_count, label in STRICT_V1_RELEASE_REQUIRED_ARTIFACTS
        if role_counts.get(role, 0) < minimum_count
    ]
    if missing:
        raise ValueError(
            "Strict V1 release bundle requires "
            f"{', '.join(missing)}"
        )


def _validate_strict_v1_databricks_purpose_coverage(
    artifacts: Sequence[_PreparedReleaseBundleArtifact],
) -> None:
    missing = _missing_strict_v1_databricks_purpose_labels(artifacts)
    if missing:
        raise ValueError(
            "Strict V1 release bundle requires "
            f"{', '.join(missing)}"
        )


def _missing_strict_v1_databricks_purpose_labels(
    artifacts: Sequence[_PreparedReleaseBundleArtifact],
) -> list[str]:
    observed_purposes: set[str] = set()
    for artifact in artifacts:
        if artifact.role != "databricks_run_status" or artifact.record is None:
            continue
        status_record = _databricks_run_status_record(artifact.record)
        if status_record is None:
            continue
        submit_payload = status_record.get("submit_payload")
        if not isinstance(submit_payload, Mapping):
            continue
        tasks = submit_payload.get("tasks")
        if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes, bytearray)):
            continue
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            purpose = task.get("purpose")
            if isinstance(purpose, str) and purpose:
                observed_purposes.add(purpose)
    return [
        label
        for purpose, label in STRICT_V1_RELEASE_REQUIRED_DATABRICKS_PURPOSES
        if purpose not in observed_purposes
    ]


def _validate_strict_v1_native_probe_factory_support(
    artifacts: Sequence[_PreparedReleaseBundleArtifact],
) -> None:
    missing = _missing_strict_v1_native_probe_factory_support_labels(artifacts)
    if missing:
        raise ValueError(
            "Strict V1 release bundle requires "
            f"{', '.join(missing)}"
        )


def _missing_strict_v1_native_probe_factory_support_labels(
    artifacts: Sequence[_PreparedReleaseBundleArtifact],
) -> list[str]:
    missing: list[str] = []
    for artifact in artifacts:
        if artifact.role != "native_probe_factories" or artifact.record is None:
            continue
        supported_backends = _supported_native_probe_factory_backends(artifact.record)
        missing.extend(
            f"{artifact.source_path}: {label}"
            for backend, label in STRICT_V1_RELEASE_REQUIRED_NATIVE_PROBE_FACTORY_SUPPORT
            if backend not in supported_backends
        )
    return _dedupe_strings(missing)


def _supported_native_probe_factory_backends(record: Mapping[str, Any]) -> set[str]:
    supported_backends: set[str] = set()
    factories = record.get("factories")
    if not isinstance(factories, Sequence) or isinstance(factories, (str, bytes, bytearray)):
        return supported_backends
    for factory in factories:
        if not isinstance(factory, Mapping):
            continue
        backend = factory.get("backend")
        if isinstance(backend, str) and factory.get("supported") is True:
            supported_backends.add(backend)
    return supported_backends


def _single_record_for_role(
    artifacts: Sequence[_PreparedReleaseBundleArtifact],
    role: str,
) -> Mapping[str, Any]:
    matches = [artifact.record for artifact in artifacts if artifact.role == role]
    if len(matches) != 1:
        raise ValueError(f"Release bundle requires exactly one {role} artifact")
    return matches[0]


def _release_evidence_sidecar_issues(
    record: Mapping[str, Any],
    release_input_artifacts: Sequence[_PreparedReleaseBundleArtifact],
) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(_unexpected_keys(record, _RELEASE_EVIDENCE_SIDECAR_KEYS, "release evidence sidecar"))
    if record.get("record_type") != RELEASE_EVIDENCE_RECORD_TYPE:
        issues.append(f"release evidence sidecar record_type must be {RELEASE_EVIDENCE_RECORD_TYPE!r}")
    if record.get("ok") is not True:
        issues.append("release evidence sidecar ok must be true")
    if record.get("v1_benchmark_ok") is not True:
        issues.append("release evidence sidecar v1_benchmark_ok must be true")
    if record.get("storage_benchmark_ok") is not True:
        issues.append("release evidence sidecar storage_benchmark_ok must be true")
    if not _matches_required_backend_set(record.get("engine_probe_backends")):
        issues.append("release evidence sidecar engine_probe_backends must match required backends")
    if not _matches_required_backend_set(record.get("engine_action_backends")):
        issues.append("release evidence sidecar engine_action_backends must match required backends")
    for field_name in (
        "missing_engine_probe_backends",
        "duplicate_engine_probe_backends",
        "invalid_engine_probe_records",
        "missing_engine_action_backends",
        "duplicate_engine_action_backends",
        "invalid_engine_action_records",
        "issues",
    ):
        if record.get(field_name) not in ([], ()):
            issues.append(f"release evidence sidecar {field_name} must be empty")
    if not _matches_expected_records(record.get("artifact_sources"), _expected_artifact_sources(release_input_artifacts)):
        issues.append("release evidence sidecar artifact_sources must match bundled release inputs")
    return tuple(issues)


def _preflight_sidecar_issues(
    record: Mapping[str, Any],
    release_input_artifacts: Sequence[_PreparedReleaseBundleArtifact],
) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(_unexpected_keys(record, _PREFLIGHT_SIDECAR_KEYS, "preflight sidecar"))
    if record.get("record_type") != RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE:
        issues.append(f"preflight sidecar record_type must be {RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE!r}")
    if record.get("ok") is not True:
        issues.append("preflight sidecar ok must be true")
    if not _matches_required_backend_set(record.get("required_engine_probe_backends")):
        issues.append("preflight sidecar required_engine_probe_backends must match required backends")
    if not _matches_required_backend_set(record.get("required_engine_action_backends")):
        issues.append("preflight sidecar required_engine_action_backends must match required backends")
    for field_name in (
        "missing_paths",
        "unreadable_paths",
        "invalid_record_type_paths",
        "missing_engine_probe_backends",
        "missing_engine_action_backends",
        "issues",
    ):
        if record.get(field_name) not in ([], ()):
            issues.append(f"preflight sidecar {field_name} must be empty")
    if not _matches_expected_records(record.get("input_files"), _expected_input_files(release_input_artifacts)):
        issues.append("preflight sidecar input_files must match bundled release inputs")
    return tuple(issues)


def _pr_evidence_sidecar_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    evidence = evaluate_pr_evidence_record(record)
    issues = list(evidence.issues)
    issues.extend(_unexpected_keys(record, _PR_EVIDENCE_RECORD_KEYS, "PR evidence sidecar"))
    if record.get("record_type") != PR_EVIDENCE_RECORD_TYPE:
        issues.append(f"PR evidence sidecar record_type must be {PR_EVIDENCE_RECORD_TYPE!r}")
    if record.get("ok") is not True:
        issues.append("PR evidence sidecar ok must be true")
    if evidence.gpt55_review_completed is not True:
        issues.append("PR evidence sidecar GPT-5.5 review must be completed")
    if evidence.refactor_skill_applied is not True:
        issues.append("PR evidence sidecar Refactor skill must be applied")
    return _dedupe_strings(issues)


def _github_governance_sidecar_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    governance_record = _github_governance_record(record)
    issues: list[str] = []
    issues.extend(_github_governance_container_key_issues(record))
    issues.extend(_github_governance_wrapper_field_issues(record))
    if governance_record is None:
        issues.append("GitHub governance sidecar must be a governance record or github_governance CLI output")
        return _dedupe_strings(issues)
    issues.extend(
        _unexpected_keys(
            governance_record,
            _GITHUB_GOVERNANCE_SUMMARY_KEYS,
            "GitHub governance sidecar summary",
        )
    )
    if governance_record.get("record_type") != GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE:
        issues.append(
            f"GitHub governance sidecar record_type must be {GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE!r}"
        )
    if governance_record.get("ok") is not True:
        issues.append("GitHub governance sidecar ok must be true")
    if not isinstance(governance_record.get("repository"), str) or not governance_record["repository"]:
        issues.append("GitHub governance sidecar repository must be non-empty")
    issues.extend(_required_str_field(governance_record, "default_branch", "GitHub governance sidecar summary"))
    if governance_record.get("default_branch") != "main":
        issues.append("GitHub governance sidecar default_branch must be 'main'")
    if governance_record.get("branch") != "main":
        issues.append("GitHub governance sidecar branch must be 'main'")
    if governance_record.get("private") is not False:
        issues.append("GitHub governance sidecar private must be false")
    if governance_record.get("visibility") != "public":
        issues.append("GitHub governance sidecar visibility must be 'public'")
    if governance_record.get("archived") is not False:
        issues.append("GitHub governance sidecar archived must be false")
    if governance_record.get("disabled") is not False:
        issues.append("GitHub governance sidecar disabled must be false")
    issues.extend(_github_repository_branding_issues(governance_record))
    issues.extend(_optional_str_field(governance_record, "homepage", "GitHub governance sidecar summary"))
    branch_protection = governance_record.get("branch_protection")
    if not isinstance(branch_protection, Mapping):
        issues.append("GitHub governance sidecar branch_protection must be an object")
    else:
        issues.extend(_github_branch_protection_issues(branch_protection))
    open_pull_requests = governance_record.get("open_pull_requests")
    if not isinstance(open_pull_requests, Mapping):
        issues.append("GitHub governance sidecar open_pull_requests must be an object")
    else:
        issues.extend(_github_open_pull_requests_issues(open_pull_requests))
    if governance_record.get("issues") != []:
        issues.append("GitHub governance sidecar issues must be an empty array")
    return _dedupe_strings(issues)


def _github_repository_branding_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    description = record.get("description")
    if not isinstance(description, str) or not description.strip():
        issues.append("GitHub governance sidecar summary.description must be a non-empty string")
    elif _GITHUB_REQUIRED_REPOSITORY_DESCRIPTION_TERM not in description.casefold():
        issues.append("GitHub governance sidecar summary.description must mention Cachet")
    topic_field_issues = _list_of_strings_field(record, "topics", "GitHub governance sidecar summary")
    issues.extend(topic_field_issues)
    if not topic_field_issues:
        topic_set = set(record["topics"])
        missing_topics = [topic for topic in _GITHUB_REQUIRED_REPOSITORY_TOPICS if topic not in topic_set]
        if missing_topics:
            issues.append(
                "GitHub governance sidecar summary.topics must include: " + ", ".join(missing_topics)
            )
    return tuple(issues)


def _github_governance_record(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if record.get("record_type") == GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE:
        return record
    summary = record.get("summary")
    if (
        isinstance(summary, Mapping)
        and summary.get("record_type") == GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE
    ):
        return summary
    return None


def _github_governance_container_key_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    if record.get("record_type") == GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE:
        return ()
    return _unexpected_keys(record, _GITHUB_GOVERNANCE_WRAPPER_KEYS, "GitHub governance sidecar wrapper")


def _github_governance_wrapper_field_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    if record.get("record_type") == GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE:
        return ()
    issues: list[str] = []
    if record.get("ok") is not True:
        issues.append("GitHub governance sidecar wrapper.ok must be true")
    if not isinstance(record.get("summary"), Mapping):
        issues.append("GitHub governance sidecar wrapper.summary must be an object")
    return tuple(issues)


def _github_branch_protection_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(
        _unexpected_keys(
            record,
            _GITHUB_BRANCH_PROTECTION_KEYS,
            "GitHub governance sidecar branch_protection",
        )
    )
    if record.get("enabled") is not True:
        issues.append("GitHub governance sidecar branch_protection.enabled must be true")
    if "error" in record or "error_status_code" in record:
        issues.append("GitHub governance sidecar branch_protection must not contain error fields")
    required_status_checks = record.get("required_status_checks")
    if not isinstance(required_status_checks, Mapping):
        issues.append("GitHub governance sidecar required_status_checks must be an object")
    else:
        issues.extend(
            _unexpected_keys(
                required_status_checks,
                _GITHUB_REQUIRED_STATUS_CHECKS_KEYS,
                "GitHub governance sidecar required_status_checks",
            )
        )
        if required_status_checks.get("strict") is not True:
            issues.append("GitHub governance sidecar required_status_checks.strict must be true")
        contexts = required_status_checks.get("contexts")
        if not isinstance(contexts, list):
            issues.append("GitHub governance sidecar required_status_checks.contexts must be an array of strings")
        elif any(not isinstance(context, str) or not context for context in contexts):
            issues.append("GitHub governance sidecar required_status_checks.contexts must be an array of strings")
        elif "Test and build" not in contexts:
            issues.append("GitHub governance sidecar required_status_checks.contexts must include 'Test and build'")
    pull_request_reviews = record.get("required_pull_request_reviews")
    if not isinstance(pull_request_reviews, Mapping):
        issues.append("GitHub governance sidecar required_pull_request_reviews must be an object")
    else:
        issues.extend(
            _unexpected_keys(
                pull_request_reviews,
                _GITHUB_REQUIRED_PULL_REQUEST_REVIEWS_KEYS,
                "GitHub governance sidecar required_pull_request_reviews",
            )
        )
        if pull_request_reviews.get("dismiss_stale_reviews") is not True:
            issues.append("GitHub governance sidecar required_pull_request_reviews.dismiss_stale_reviews must be true")
        if pull_request_reviews.get("require_last_push_approval") is not True:
            issues.append(
                "GitHub governance sidecar required_pull_request_reviews.require_last_push_approval must be true"
            )
        if pull_request_reviews.get("required_approving_review_count") != 1:
            issues.append(
                "GitHub governance sidecar required_pull_request_reviews.required_approving_review_count must be 1"
            )
    if record.get("required_linear_history") is not True:
        issues.append("GitHub governance sidecar required_linear_history must be true")
    if record.get("required_conversation_resolution") is not True:
        issues.append("GitHub governance sidecar required_conversation_resolution must be true")
    if record.get("enforce_admins") is not True:
        issues.append("GitHub governance sidecar enforce_admins must be true")
    if record.get("allow_force_pushes") is not False:
        issues.append("GitHub governance sidecar allow_force_pushes must be false")
    if record.get("allow_deletions") is not False:
        issues.append("GitHub governance sidecar allow_deletions must be false")
    return tuple(issues)


def _github_open_pull_requests_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(
        _unexpected_keys(
            record,
            _GITHUB_OPEN_PULL_REQUESTS_KEYS,
            "GitHub governance sidecar open_pull_requests",
        )
    )
    if record.get("checked") is not True:
        issues.append("GitHub governance sidecar open_pull_requests.checked must be true")
    if "error" in record or "error_status_code" in record:
        issues.append("GitHub governance sidecar open_pull_requests must not contain error fields")
    if type(record.get("total_count")) is not int or record.get("total_count") < 0:
        issues.append("GitHub governance sidecar open_pull_requests.total_count must be a non-negative integer")
    if type(record.get("unexpected_count")) is not int or record.get("unexpected_count") != 0:
        issues.append("GitHub governance sidecar open_pull_requests.unexpected_count must be 0")
    if record.get("truncated") is not False:
        issues.append("GitHub governance sidecar open_pull_requests.truncated must be false")
    issues.extend(
        _list_of_non_negative_ints_field(
            record,
            "allowed_numbers",
            "GitHub governance sidecar open_pull_requests",
        )
    )
    allowed_numbers = record.get("allowed_numbers")
    valid_allowed_numbers = (
        isinstance(allowed_numbers, Sequence)
        and not isinstance(allowed_numbers, (str, bytes, bytearray))
        and all(type(number) is int and number >= 0 for number in allowed_numbers)
    )
    if type(record.get("allowed_count")) is not int or record.get("allowed_count") < 0:
        issues.append("GitHub governance sidecar open_pull_requests.allowed_count must be a non-negative integer")
    allowed = record.get("allowed")
    if not isinstance(allowed, Sequence) or isinstance(allowed, (str, bytes, bytearray)):
        issues.append("GitHub governance sidecar open_pull_requests.allowed must be an array")
    else:
        if record.get("allowed_count") != len(allowed):
            issues.append("GitHub governance sidecar open_pull_requests.allowed_count must match allowed length")
        allowed_summary_numbers = []
        for index, pull_request in enumerate(allowed):
            issues.extend(_github_pull_request_summary_issues(pull_request, index=index))
            if isinstance(pull_request, Mapping) and type(pull_request.get("number")) is int:
                allowed_summary_numbers.append(pull_request["number"])
        if valid_allowed_numbers:
            unexpected_allowed_numbers = sorted(set(allowed_summary_numbers) - set(allowed_numbers))
            if unexpected_allowed_numbers:
                issues.append(
                    "GitHub governance sidecar open_pull_requests.allowed numbers must be listed in "
                    "allowed_numbers"
                )
            if record.get("allowed_count") == len(allowed) and sorted(allowed_summary_numbers) != sorted(allowed_numbers):
                issues.append(
                    "GitHub governance sidecar open_pull_requests.allowed must summarize every allowed number"
                )
    if (
        type(record.get("total_count")) is int
        and record.get("total_count") >= 0
        and type(record.get("allowed_count")) is int
        and record.get("allowed_count") >= 0
        and type(record.get("unexpected_count")) is int
        and record.get("unexpected_count") >= 0
        and record.get("total_count") != record.get("allowed_count") + record.get("unexpected_count")
    ):
        issues.append(
            "GitHub governance sidecar open_pull_requests.total_count must equal "
            "allowed_count plus unexpected_count"
        )
    if record.get("unexpected") != []:
        issues.append("GitHub governance sidecar open_pull_requests.unexpected must be an empty array")
    return tuple(issues)


def _github_pull_request_summary_issues(record: Any, *, index: int) -> tuple[str, ...]:
    label = f"GitHub governance sidecar open_pull_requests.allowed[{index}]"
    if not isinstance(record, Mapping):
        return (f"{label} must be an object",)
    issues: list[str] = []
    issues.extend(_unexpected_keys(record, _GITHUB_PULL_REQUEST_SUMMARY_KEYS, label))
    if type(record.get("number")) is not int or record.get("number") <= 0:
        issues.append(f"{label}.number must be a positive integer")
    issues.extend(_required_str_field(record, "title", label))
    issues.extend(_bool_field(record, "draft", label))
    issues.extend(_required_str_field(record, "html_url", label))
    issues.extend(_required_str_field(record, "head_ref", label))
    issues.extend(_required_str_field(record, "base_ref", label))
    return tuple(issues)


def _repository_hygiene_sidecar_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(_unexpected_keys(record, _REPOSITORY_HYGIENE_KEYS, "repository hygiene sidecar"))
    if record.get("record_type") != REPOSITORY_HYGIENE_RECORD_TYPE:
        issues.append(f"repository hygiene sidecar record_type must be {REPOSITORY_HYGIENE_RECORD_TYPE!r}")
    if record.get("ok") is not True:
        issues.append("repository hygiene sidecar ok must be true")
    if not isinstance(record.get("repository_root"), str) or not record["repository_root"]:
        issues.append("repository hygiene sidecar repository_root must be non-empty")
    if type(record.get("tracked_path_count")) is not int or record.get("tracked_path_count") <= 0:
        issues.append("repository hygiene sidecar tracked_path_count must be a positive integer")
    if type(record.get("untracked_path_count")) is not int or record.get("untracked_path_count") < 0:
        issues.append("repository hygiene sidecar untracked_path_count must be a non-negative integer")
    issues.extend(
        _exact_string_list_field(
            record,
            "required_gitignore_patterns",
            REQUIRED_GITIGNORE_PATTERNS,
            "repository hygiene sidecar",
        )
    )
    issues.extend(
        _exact_string_list_field(
            record,
            "forbidden_tracked_artifact_patterns",
            FORBIDDEN_TRACKED_ARTIFACT_PATTERNS,
            "repository hygiene sidecar",
        )
    )
    if record.get("missing_gitignore_patterns") != []:
        issues.append("repository hygiene sidecar missing_gitignore_patterns must be an empty array")
    if record.get("forbidden_tracked_paths") != []:
        issues.append("repository hygiene sidecar forbidden_tracked_paths must be an empty array")
    if record.get("forbidden_untracked_paths") != []:
        issues.append("repository hygiene sidecar forbidden_untracked_paths must be an empty array")
    if record.get("dirty_tracked_paths") != []:
        issues.append("repository hygiene sidecar dirty_tracked_paths must be an empty array")
    issues.extend(
        _list_of_strings_field(
            record,
            "documentation_checked_directory_paths",
            "repository hygiene sidecar",
        )
    )
    issues.extend(
        _list_of_strings_field(
            record,
            "missing_directory_documentation_paths",
            "repository hygiene sidecar",
        )
    )
    if record.get("missing_directory_documentation_paths") != []:
        issues.append("repository hygiene sidecar missing_directory_documentation_paths must be an empty array")
    if record.get("issues") != []:
        issues.append("repository hygiene sidecar issues must be an empty array")
    return _dedupe_strings(issues)


def _plan_execution_sidecar_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(_unexpected_keys(record, _BENCHMARK_PLAN_EXECUTION_KEYS, "benchmark plan execution sidecar"))
    if record.get("record_type") != BENCHMARK_PLAN_EXECUTION_RECORD_TYPE:
        issues.append(f"benchmark plan execution sidecar record_type must be {BENCHMARK_PLAN_EXECUTION_RECORD_TYPE!r}")
    if record.get("ok") is not True:
        issues.append("benchmark plan execution sidecar ok must be true")
    commands = record.get("commands")
    if not isinstance(commands, Sequence) or isinstance(commands, (str, bytes, bytearray)) or not commands:
        issues.append("benchmark plan execution sidecar commands must be a non-empty array")
    else:
        issues.extend(_plan_execution_command_issues(commands))
    plan_source = record.get("plan_source")
    if not isinstance(plan_source, Mapping):
        issues.append("benchmark plan execution sidecar plan_source must be an object")
    else:
        issues.extend(_plan_source_issues(plan_source))
    return _dedupe_strings(issues)


def _plan_execution_command_issues(commands: Sequence[Any]) -> tuple[str, ...]:
    issues: list[str] = []
    for index, command in enumerate(commands):
        if not isinstance(command, Mapping):
            issues.append(f"benchmark plan execution sidecar commands[{index}] must be an object")
            continue
        issues.extend(
            _unexpected_keys(
                command,
                _BENCHMARK_PLAN_EXECUTION_COMMAND_KEYS,
                f"benchmark plan execution sidecar commands[{index}]",
            )
        )
        if type(command.get("returncode")) is not int or command["returncode"] != 0:
            issues.append(f"benchmark plan execution sidecar commands[{index}].returncode must be 0")
        if command.get("error") is not None:
            issues.append(f"benchmark plan execution sidecar commands[{index}].error must be null")
    return tuple(issues)


def _native_probe_factories_sidecar_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(_unexpected_keys(record, _NATIVE_PROBE_FACTORIES_KEYS, "native probe factories sidecar"))
    if record.get("record_type") != NATIVE_PROBE_FACTORIES_RECORD_TYPE:
        issues.append(
            f"native probe factories sidecar record_type must be {NATIVE_PROBE_FACTORIES_RECORD_TYPE!r}"
        )
    factories = record.get("factories")
    if not isinstance(factories, Sequence) or isinstance(factories, (str, bytes, bytearray)) or not factories:
        issues.append("native probe factories sidecar factories must be a non-empty array")
        return _dedupe_strings(issues)

    backends: list[str] = []
    for index, factory in enumerate(factories):
        if not isinstance(factory, Mapping):
            issues.append(f"native probe factories sidecar factories[{index}] must be an object")
            continue
        issues.extend(_native_probe_factory_issues(factory, index=index))
        backend = factory.get("backend")
        if isinstance(backend, str):
            backends.append(backend)
    if set(backends) != set(REQUIRED_ENGINE_PROBE_BACKENDS) or len(backends) != len(set(backends)):
        issues.append("native probe factories sidecar backends must match required backends")
    return _dedupe_strings(issues)


def _native_probe_factory_issues(factory: Mapping[str, Any], *, index: int) -> tuple[str, ...]:
    label = f"native probe factories sidecar factories[{index}]"
    issues: list[str] = []
    issues.extend(_unexpected_keys(factory, _NATIVE_PROBE_FACTORY_KEYS, label))
    backend = factory.get("backend")
    if not isinstance(backend, str) or backend not in REQUIRED_ENGINE_PROBE_BACKENDS:
        issues.append(f"{label}.backend must be one of {list(REQUIRED_ENGINE_PROBE_BACKENDS)!r}")
    else:
        expected_factory_path = _BUILTIN_NATIVE_PROBE_FACTORY_PATHS.get(backend)
        if factory.get("factory_path") != expected_factory_path:
            issues.append(f"{label}.factory_path must match the built-in {backend} factory path")
    for field_name in ("backend", "factory_path", "package_name", "reason"):
        issues.extend(_required_str_field(factory, field_name, label))
    issues.extend(_optional_str_field(factory, "package_version", label))
    for field_name in ("package_importable", "supported"):
        issues.extend(_bool_field(factory, field_name, label))
    if factory.get("supported") is True:
        if factory.get("package_importable") is not True:
            issues.append(f"{label}.package_importable must be true when supported is true")
        if not isinstance(factory.get("package_version"), str) or not factory["package_version"]:
            issues.append(f"{label}.package_version must be non-empty when supported is true")
    serving_profile = factory.get("serving_environment_profile")
    if not isinstance(serving_profile, Mapping):
        issues.append(f"{label}.serving_environment_profile must be an object")
    elif isinstance(backend, str) and backend in REQUIRED_ENGINE_PROBE_BACKENDS:
        expected_profile = serving_environment_profile_to_record(serving_environment_profile(backend))
        if dict(serving_profile) != expected_profile:
            issues.append(f"{label}.serving_environment_profile must match the built-in {backend} profile")
    return tuple(issues)


def _plan_source_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(_unexpected_keys(record, _BENCHMARK_PLAN_SOURCE_KEYS, "benchmark plan execution sidecar plan_source"))
    if record.get("record_type") != BENCHMARK_PLAN_SOURCE_RECORD_TYPE:
        issues.append(f"benchmark plan execution sidecar plan_source.record_type must be {BENCHMARK_PLAN_SOURCE_RECORD_TYPE!r}")
    for field_name in ("path", "driver_path"):
        if not isinstance(record.get(field_name), str) or not record[field_name]:
            issues.append(f"benchmark plan execution sidecar plan_source.{field_name} must be non-empty")
    if type(record.get("size_bytes")) is not int or record["size_bytes"] <= 0:
        issues.append("benchmark plan execution sidecar plan_source.size_bytes must be a positive integer")
    if not isinstance(record.get("sha256"), str) or not _SHA256_HEX_RE.fullmatch(record["sha256"]):
        issues.append("benchmark plan execution sidecar plan_source.sha256 must be a 64-character lowercase hex digest")
    if "command_count" in record and (type(record["command_count"]) is not int or record["command_count"] <= 0):
        issues.append("benchmark plan execution sidecar plan_source.command_count must be a positive integer when present")
    return tuple(issues)


def _databricks_run_status_sidecar_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    status_record = _databricks_run_status_record(record)
    issues: list[str] = []
    if "response" in record:
        issues.append("Databricks run status sidecar must not include the raw Jobs API response")
    issues.extend(_databricks_run_status_container_key_issues(record))
    issues.extend(_databricks_run_status_wrapper_field_issues(record))
    if status_record is None:
        issues.append("Databricks run status sidecar must be a status record or databricks_runs get --summary output")
        return _dedupe_strings(issues)
    issues.extend(_unexpected_keys(status_record, _DATABRICKS_RUN_STATUS_KEYS, "Databricks run status sidecar summary"))
    issues.extend(_databricks_run_status_field_issues(status_record))
    if status_record.get("record_type") != DATABRICKS_RUN_STATUS_RECORD_TYPE:
        issues.append(f"Databricks run status sidecar record_type must be {DATABRICKS_RUN_STATUS_RECORD_TYPE!r}")
    if status_record.get("terminal") is not True:
        issues.append("Databricks run status sidecar terminal must be true")
    if status_record.get("succeeded") is not True:
        issues.append("Databricks run status sidecar succeeded must be true")
    if status_record.get("life_cycle_state") != "TERMINATED":
        issues.append("Databricks run status sidecar life_cycle_state must be 'TERMINATED'")
    if status_record.get("result_state") != "SUCCESS":
        issues.append("Databricks run status sidecar result_state must be 'SUCCESS'")
    run_id = status_record.get("run_id")
    if not ((type(run_id) is int and run_id >= 0) or (isinstance(run_id, str) and run_id)):
        issues.append("Databricks run status sidecar run_id must be a non-negative integer or non-empty string")
    task_count = status_record.get("task_count")
    tasks = status_record.get("tasks")
    if type(task_count) is not int or task_count <= 0:
        issues.append("Databricks run status sidecar task_count must be a positive integer")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes, bytearray)) or not tasks:
        issues.append("Databricks run status sidecar tasks must be a non-empty array")
    else:
        if type(task_count) is int and task_count > 0 and len(tasks) != task_count:
            issues.append("Databricks run status sidecar task_count must match tasks length")
        issues.extend(_databricks_run_status_task_issues(tasks))
    submit_payload = status_record.get("submit_payload")
    if not isinstance(submit_payload, Mapping):
        issues.append("Databricks run status sidecar submit_payload must be an object")
    else:
        issues.extend(_databricks_submit_payload_sidecar_issues(submit_payload, tasks=tasks))
    return _dedupe_strings(issues)


def _databricks_run_status_record(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if record.get("record_type") == DATABRICKS_RUN_STATUS_RECORD_TYPE:
        return record
    summary = record.get("summary")
    if (
        record.get("ok") is True
        and summary is not None
        and isinstance(summary, Mapping)
        and summary.get("record_type") == DATABRICKS_RUN_STATUS_RECORD_TYPE
    ):
        return summary
    return None


def _databricks_run_status_task_issues(tasks: Sequence[Any]) -> tuple[str, ...]:
    issues: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            issues.append(f"Databricks run status sidecar tasks[{index}] must be an object")
            continue
        issues.extend(
            _unexpected_keys(task, _DATABRICKS_RUN_STATUS_TASK_KEYS, f"Databricks run status sidecar tasks[{index}]")
        )
        issues.extend(_databricks_run_status_task_field_issues(task, index=index))
        if not isinstance(task.get("task_key"), str) or not task["task_key"]:
            issues.append(f"Databricks run status sidecar tasks[{index}].task_key must be non-empty")
        if task.get("life_cycle_state") != "TERMINATED":
            issues.append(f"Databricks run status sidecar tasks[{index}].life_cycle_state must be 'TERMINATED'")
        if task.get("result_state") != "SUCCESS":
            issues.append(f"Databricks run status sidecar tasks[{index}].result_state must be 'SUCCESS'")
    return tuple(issues)


def _databricks_submit_payload_sidecar_issues(
    record: Mapping[str, Any],
    *,
    tasks: Any,
) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(_unexpected_keys(record, _DATABRICKS_SUBMIT_PAYLOAD_KEYS, "Databricks run status sidecar submit_payload"))
    issues.extend(_databricks_submit_payload_field_issues(record))
    if record.get("record_type") != DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE:
        issues.append(
            f"Databricks run status sidecar submit_payload.record_type must be {DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE!r}"
        )
    source_path = record.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        issues.append("Databricks run status sidecar submit_payload.source_path must be non-empty")
    if not isinstance(record.get("sha256"), str) or not _SHA256_HEX_RE.fullmatch(record["sha256"]):
        issues.append("Databricks run status sidecar submit_payload.sha256 must be a 64-character lowercase hex digest")
    if record.get("single_node") is not True:
        issues.append("Databricks run status sidecar submit_payload.single_node must be true")
    if record.get("aws_g5_node_type") is not True:
        issues.append("Databricks run status sidecar submit_payload.aws_g5_node_type must be true")
    task_count = record.get("task_count")
    payload_tasks = record.get("tasks")
    if type(task_count) is not int or task_count <= 0:
        issues.append("Databricks run status sidecar submit_payload.task_count must be a positive integer")
    if not isinstance(payload_tasks, Sequence) or isinstance(payload_tasks, (str, bytes, bytearray)) or not payload_tasks:
        issues.append("Databricks run status sidecar submit_payload.tasks must be a non-empty array")
    else:
        if type(task_count) is int and task_count > 0 and len(payload_tasks) != task_count:
            issues.append("Databricks run status sidecar submit_payload.task_count must match tasks length")
        issues.extend(_databricks_submit_payload_task_issues(payload_tasks))
    data_security_modes = record.get("data_security_modes")
    if not isinstance(data_security_modes, Sequence) or isinstance(data_security_modes, (str, bytes, bytearray)):
        issues.append("Databricks run status sidecar submit_payload.data_security_modes must be an array")
    elif "SINGLE_USER" not in data_security_modes:
        issues.append("Databricks run status sidecar submit_payload.data_security_modes must include SINGLE_USER")
    if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes, bytearray)):
        status_task_keys = _task_key_list(tasks)
        payload_task_keys = _task_key_list(record.get("tasks"))
        if not status_task_keys:
            issues.append("Databricks run status sidecar tasks must include task keys")
        if not payload_task_keys:
            issues.append("Databricks run status sidecar submit_payload.tasks must include task keys")
        if status_task_keys and payload_task_keys and status_task_keys != payload_task_keys:
            issues.append("Databricks run status sidecar submit_payload.task_keys must match status task keys")
    if isinstance(record.get("task_keys"), Sequence) and not isinstance(record.get("task_keys"), (str, bytes, bytearray)):
        declared_task_keys = [key for key in record["task_keys"] if isinstance(key, str) and key]
        payload_task_keys = _task_key_list(record.get("tasks"))
        if declared_task_keys != payload_task_keys:
            issues.append("Databricks run status sidecar submit_payload.task_keys must match submit_payload.tasks")
    return tuple(issues)


def _databricks_submit_payload_task_issues(tasks: Sequence[Any]) -> tuple[str, ...]:
    issues: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            issues.append(f"Databricks run status sidecar submit_payload.tasks[{index}] must be an object")
            continue
        issues.extend(
            _unexpected_keys(
                task,
                _DATABRICKS_SUBMIT_PAYLOAD_TASK_KEYS,
                f"Databricks run status sidecar submit_payload.tasks[{index}]",
            )
        )
        issues.extend(_databricks_submit_payload_task_field_issues(task, index=index))
        if not isinstance(task.get("task_key"), str) or not task["task_key"]:
            issues.append(f"Databricks run status sidecar submit_payload.tasks[{index}].task_key must be non-empty")
        for field_name in ("node_type_id", "driver_node_type_id"):
            value = task.get(field_name)
            if not isinstance(value, str) or not value.startswith("g5."):
                issues.append(
                    f"Databricks run status sidecar submit_payload.tasks[{index}].{field_name} must be an AWS g5 node type"
                )
        if task.get("single_node") is not True:
            issues.append(f"Databricks run status sidecar submit_payload.tasks[{index}].single_node must be true")
        if task.get("data_security_mode") != "SINGLE_USER":
            issues.append(
                f"Databricks run status sidecar submit_payload.tasks[{index}].data_security_mode must be 'SINGLE_USER'"
            )
    return tuple(issues)


def _databricks_run_status_field_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(_required_str_field(record, "record_type", "Databricks run status sidecar"))
    issues.extend(_run_id_field_issues(record, "run_id", "Databricks run status sidecar"))
    for field_name in ("run_name", "run_page_url", "state_message", "active_task_key", "cluster_id"):
        issues.extend(_optional_str_field(record, field_name, "Databricks run status sidecar"))
    for field_name in ("life_cycle_state", "result_state"):
        issues.extend(_required_str_field(record, field_name, "Databricks run status sidecar"))
    for field_name in ("start_time", "end_time"):
        issues.extend(_optional_int_field(record, field_name, "Databricks run status sidecar"))
    for field_name in ("terminal", "succeeded"):
        issues.extend(_bool_field(record, field_name, "Databricks run status sidecar"))
    return tuple(issues)


def _databricks_run_status_task_field_issues(task: Mapping[str, Any], *, index: int) -> tuple[str, ...]:
    label = f"Databricks run status sidecar tasks[{index}]"
    issues: list[str] = []
    issues.extend(_required_str_field(task, "task_key", label))
    issues.extend(_run_id_field_issues(task, "run_id", label))
    for field_name in ("life_cycle_state", "result_state"):
        issues.extend(_required_str_field(task, field_name, label))
    for field_name in ("state_message", "cluster_id"):
        issues.extend(_optional_str_field(task, field_name, label))
    for field_name in ("start_time", "end_time"):
        issues.extend(_optional_int_field(task, field_name, label))
    return tuple(issues)


def _databricks_submit_payload_field_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    for field_name in ("record_type", "source_path"):
        issues.extend(_required_str_field(record, field_name, "Databricks run status sidecar submit_payload"))
    issues.extend(_optional_str_field(record, "run_name", "Databricks run status sidecar submit_payload"))
    for field_name in ("task_keys", "node_type_ids", "driver_node_type_ids", "spark_versions", "data_security_modes"):
        issues.extend(_list_of_strings_field(record, field_name, "Databricks run status sidecar submit_payload"))
    for field_name in ("single_node", "aws_g5_node_type"):
        issues.extend(_bool_field(record, field_name, "Databricks run status sidecar submit_payload"))
    return tuple(issues)


def _databricks_submit_payload_task_field_issues(task: Mapping[str, Any], *, index: int) -> tuple[str, ...]:
    label = f"Databricks run status sidecar submit_payload.tasks[{index}]"
    issues: list[str] = []
    for field_name in ("task_key", "node_type_id", "driver_node_type_id", "spark_version", "data_security_mode"):
        issues.extend(_required_str_field(task, field_name, label))
    issues.extend(_optional_str_field(task, "purpose", label))
    if type(task.get("num_workers")) is not int:
        issues.append(f"{label}.num_workers must be an integer")
    issues.extend(_bool_field(task, "single_node", label))
    return tuple(issues)


def _databricks_run_status_container_key_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    if record.get("record_type") == DATABRICKS_RUN_STATUS_RECORD_TYPE:
        return ()
    return _unexpected_keys(record, _DATABRICKS_RUN_STATUS_WRAPPER_KEYS, "Databricks run status sidecar wrapper")


def _databricks_run_status_wrapper_field_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    if record.get("record_type") == DATABRICKS_RUN_STATUS_RECORD_TYPE:
        return ()
    issues: list[str] = []
    if record.get("ok") is not True:
        issues.append("Databricks run status sidecar wrapper.ok must be true")
    if record.get("action") != "get":
        issues.append("Databricks run status sidecar wrapper.action must be 'get'")
    if not isinstance(record.get("summary"), Mapping):
        issues.append("Databricks run status sidecar wrapper.summary must be an object")
    return tuple(issues)


def _unexpected_keys(record: Mapping[str, Any], allowed_keys: frozenset[str], label: str) -> tuple[str, ...]:
    unexpected = sorted(str(key) for key in record if key not in allowed_keys)
    if not unexpected:
        return ()
    return (f"{label} has unsupported keys: {unexpected}",)


def _required_str_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if isinstance(value, str) and value:
        return ()
    return (f"{label}.{field_name} must be a non-empty string",)


def _optional_str_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if value is None or isinstance(value, str):
        return ()
    return (f"{label}.{field_name} must be a string or null",)


def _optional_int_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if value is None or type(value) is int:
        return ()
    return (f"{label}.{field_name} must be an integer or null",)


def _bool_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    if type(record.get(field_name)) is bool:
        return ()
    return (f"{label}.{field_name} must be boolean",)


def _run_id_field_issues(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if (type(value) is int and value >= 0) or (isinstance(value, str) and value):
        return ()
    return (f"{label}.{field_name} must be a non-negative integer or non-empty string",)


def _list_of_strings_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and all(
        isinstance(item, str) and item for item in value
    ):
        return ()
    return (f"{label}.{field_name} must be an array of non-empty strings",)


def _exact_string_list_field(
    record: Mapping[str, Any],
    field_name: str,
    expected: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    value = record.get(field_name)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and list(value) == list(expected)
    ):
        return ()
    return (f"{label}.{field_name} must match the current repository hygiene policy",)


def _list_of_non_negative_ints_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and all(
        type(item) is int and item >= 0 for item in value
    ):
        return ()
    return (f"{label}.{field_name} must be an array of non-negative integers",)


def _task_key_list(tasks: Any) -> list[str]:
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes, bytearray)):
        return []
    return [
        task["task_key"]
        for task in tasks
        if isinstance(task, Mapping) and isinstance(task.get("task_key"), str) and task["task_key"]
    ]


def _package_wheel_issues(
    source_path: str,
    payload: bytes,
    *,
    expected_version: str | None = None,
) -> tuple[str, ...]:
    issues: list[str] = []
    filename = Path(source_path).name
    filename_match = _WHEEL_FILENAME_RE.fullmatch(filename)
    if Path(filename).suffix != ".whl":
        issues.append("package wheel artifact source_path must end with .whl")
    elif filename_match is None:
        issues.append("package wheel artifact source_path must use a valid wheel filename")
    else:
        issues.extend(_wheel_filename_issues(filename_match))
    if not payload:
        issues.append("package wheel artifact must be non-empty")
    else:
        issues.extend(
            _wheel_zip_payload_issues(
                payload,
                filename_match=filename_match,
                expected_version=expected_version,
            )
        )
    return tuple(issues)


def _wheel_filename_issues(filename_match: re.Match[str]) -> tuple[str, ...]:
    issues: list[str] = []
    if canonicalize_name(filename_match.group("distribution")) != canonicalize_name(RELEASE_BUNDLE_PACKAGE_NAME):
        issues.append(f"package wheel artifact filename distribution must be {RELEASE_BUNDLE_PACKAGE_NAME!r}")
    if (
        filename_match.group("python_tag"),
        filename_match.group("abi_tag"),
        filename_match.group("platform_tag"),
    ) != ("py3", "none", "any"):
        issues.append("package wheel artifact filename tags must be py3-none-any")
    return tuple(issues)


def _wheel_zip_payload_issues(
    payload: bytes,
    *,
    filename_match: re.Match[str] | None,
    expected_version: str | None,
) -> tuple[str, ...]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as wheel_zip:
            corrupt_member = wheel_zip.testzip()
            if corrupt_member is not None:
                return (f"package wheel artifact zip member {corrupt_member!r} is corrupt",)
            names = wheel_zip.namelist()
            duplicate_member_issues = _duplicate_wheel_member_issues(names)
            dist_info_prefixes = tuple(sorted(_root_dist_info_prefixes(names)))
            metadata_names = tuple(name for name in names if _is_root_dist_info_file(name, "METADATA"))
            record_names = tuple(name for name in names if _is_root_dist_info_file(name, "RECORD"))
            wheel_names = tuple(name for name in names if _is_root_dist_info_file(name, "WHEEL"))
    except zipfile.BadZipFile:
        return ("package wheel artifact must be a valid wheel zip payload",)
    if duplicate_member_issues:
        return duplicate_member_issues
    if len(dist_info_prefixes) != 1:
        return ("package wheel artifact must contain exactly one root-level .dist-info directory",)
    dist_info_issues = _wheel_dist_info_prefix_issues(dist_info_prefixes[0], filename_match=filename_match)
    if dist_info_issues:
        return dist_info_issues
    if len(wheel_names) != 1:
        return ("package wheel artifact must contain .dist-info/WHEEL metadata",)
    if len(metadata_names) != 1:
        return ("package wheel artifact must contain exactly one .dist-info/METADATA file",)
    if len(record_names) != 1:
        return ("package wheel artifact must contain exactly one .dist-info/RECORD file",)
    license_issues = _wheel_license_file_issues(names, dist_info_prefix=dist_info_prefixes[0])
    if license_issues:
        return license_issues
    typed_marker_issues = _wheel_typed_marker_issues(names)
    if typed_marker_issues:
        return typed_marker_issues
    with zipfile.ZipFile(io.BytesIO(payload)) as wheel_zip:
        wheel_payload = wheel_zip.read(wheel_names[0])
        metadata_payload = wheel_zip.read(metadata_names[0])
        record_payload = wheel_zip.read(record_names[0])
        record_issues = _wheel_record_issues(
            record_payload,
            wheel_zip=wheel_zip,
            required_paths=(wheel_names[0], metadata_names[0], record_names[0]),
        )
    return (
        *_wheel_file_issues(wheel_payload),
        *_wheel_metadata_issues(
            metadata_payload,
            filename_match=filename_match,
            expected_version=expected_version,
        ),
        *record_issues,
    )


def _duplicate_wheel_member_issues(names: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name.endswith("/"):
            continue
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        duplicate_list = ", ".join(repr(name) for name in duplicates)
        return (f"package wheel artifact zip entries must not contain duplicate file paths: {duplicate_list}",)
    return ()


def _root_dist_info_prefixes(names: Sequence[str]) -> set[str]:
    prefixes = set()
    for name in names:
        first, separator, rest = name.partition("/")
        if separator and first.endswith(".dist-info") and rest:
            prefixes.add(first)
    return prefixes


def _is_root_dist_info_file(name: str, filename: str) -> bool:
    first, separator, rest = name.partition("/")
    return bool(separator and first.endswith(".dist-info") and rest == filename)


def _wheel_dist_info_prefix_issues(prefix: str, *, filename_match: re.Match[str] | None) -> tuple[str, ...]:
    if filename_match is None:
        return ()
    expected_prefix = f"{filename_match.group('distribution')}-{filename_match.group('version')}.dist-info"
    if prefix != expected_prefix:
        return ("package wheel artifact .dist-info directory must match wheel filename distribution and version",)
    return ()


def _wheel_metadata_issues(
    payload: bytes,
    *,
    filename_match: re.Match[str] | None,
    expected_version: str | None,
) -> tuple[str, ...]:
    try:
        metadata = _metadata_headers(payload.decode("utf-8"))
    except UnicodeDecodeError:
        return ("package wheel artifact METADATA must be UTF-8 text",)
    name = metadata.get("name")
    version = metadata.get("version")
    if name is None or canonicalize_name(name) != canonicalize_name(RELEASE_BUNDLE_PACKAGE_NAME):
        return (f"package wheel artifact METADATA Name must be {RELEASE_BUNDLE_PACKAGE_NAME!r}",)
    if not version:
        return ("package wheel artifact METADATA Version must be non-empty",)
    if filename_match is not None and not _wheel_versions_match(filename_match.group("version"), version):
        return ("package wheel artifact METADATA Version must match wheel filename",)
    if expected_version is not None and not _wheel_versions_match(expected_version, version):
        return (
            "package wheel artifact METADATA Version "
            f"must match current project version {expected_version!r}",
        )
    return _wheel_metadata_license_issues(metadata)


def _wheel_metadata_license_issues(metadata: Mapping[str, str]) -> tuple[str, ...]:
    issues: list[str] = []
    if metadata.get("license-expression") != RELEASE_BUNDLE_PACKAGE_LICENSE_EXPRESSION:
        issues.append(
            "package wheel artifact METADATA License-Expression "
            f"must be {RELEASE_BUNDLE_PACKAGE_LICENSE_EXPRESSION!r}"
        )
    if metadata.get("license-file") != RELEASE_BUNDLE_PACKAGE_LICENSE_FILE:
        issues.append(
            "package wheel artifact METADATA License-File "
            f"must be {RELEASE_BUNDLE_PACKAGE_LICENSE_FILE!r}"
        )
    return tuple(issues)


def _package_wheel_identity(payload: bytes) -> _WheelPackageIdentity | None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as wheel_zip:
            metadata_names = tuple(
                name for name in wheel_zip.namelist() if _is_root_dist_info_file(name, "METADATA")
            )
            if len(metadata_names) != 1:
                return None
            metadata = _metadata_headers(wheel_zip.read(metadata_names[0]).decode("utf-8"))
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError):
        return None
    name = metadata.get("name")
    version = metadata.get("version")
    if not name or not version:
        return None
    return _WheelPackageIdentity(
        package_name=canonicalize_name(name),
        package_version=version,
    )


def _wheel_license_file_issues(names: Sequence[str], *, dist_info_prefix: str) -> tuple[str, ...]:
    expected_path = f"{dist_info_prefix}/licenses/{RELEASE_BUNDLE_PACKAGE_LICENSE_FILE}"
    if expected_path not in names:
        return (f"package wheel artifact must contain license file {expected_path!r}",)
    return ()


def _wheel_typed_marker_issues(names: Sequence[str]) -> tuple[str, ...]:
    wheel_paths = set(names)
    return tuple(
        f"package wheel artifact must contain typed marker file {path!r}"
        for path in RELEASE_BUNDLE_PACKAGE_TYPED_MARKER_PATHS
        if path not in wheel_paths
    )


def _wheel_file_issues(payload: bytes) -> tuple[str, ...]:
    try:
        headers = _metadata_header_values(payload.decode("utf-8"))
    except UnicodeDecodeError:
        return ("package wheel artifact WHEEL metadata must be UTF-8 text",)
    issues: list[str] = []
    if not any(value for value in headers.get("wheel-version", ())):
        issues.append("package wheel artifact WHEEL metadata Wheel-Version must be non-empty")
    root_is_purelib = headers.get("root-is-purelib", ())
    if tuple(value.lower() for value in root_is_purelib) != ("true",):
        issues.append("package wheel artifact WHEEL metadata Root-Is-Purelib must be true")
    tags = headers.get("tag", ())
    if "py3-none-any" not in tags:
        issues.append("package wheel artifact WHEEL metadata Tag must include py3-none-any")
    return tuple(issues)


def _wheel_record_issues(
    payload: bytes,
    *,
    wheel_zip: zipfile.ZipFile,
    required_paths: Sequence[str],
) -> tuple[str, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return ("package wheel artifact RECORD must be UTF-8 text",)
    if not text.strip():
        return ("package wheel artifact RECORD must be non-empty",)

    issues: list[str] = []
    zip_names = frozenset(wheel_zip.namelist())
    recorded_paths: set[str] = set()
    package_payload_paths: set[str] = set()
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if len(row) != 3:
            issues.append("package wheel artifact RECORD rows must have path, hash, and size columns")
            continue
        path, hash_value, size_value = (column.strip() for column in row)
        if not path:
            issues.append("package wheel artifact RECORD rows must include a non-empty path")
            continue
        if path in recorded_paths:
            issues.append(f"package wheel artifact RECORD path {path!r} must be listed only once")
            continue
        recorded_paths.add(path)
        if path.startswith("document_kv_cache/") and not path.endswith("/"):
            package_payload_paths.add(path)
        if path not in zip_names:
            issues.append(f"package wheel artifact RECORD path {path!r} must exist in the wheel")
            continue
        issues.extend(_wheel_record_file_field_issues(wheel_zip, path, hash_value=hash_value, size_value=size_value))

    missing_required = tuple(path for path in required_paths if path not in recorded_paths)
    if missing_required:
        missing = ", ".join(repr(path) for path in missing_required)
        issues.append(f"package wheel artifact RECORD must list required wheel files: {missing}")
    unrecorded_paths = tuple(sorted(name for name in zip_names if not name.endswith("/") and name not in recorded_paths))
    if unrecorded_paths:
        missing = ", ".join(repr(path) for path in unrecorded_paths)
        issues.append(f"package wheel artifact RECORD must list every wheel file: {missing}")
    if not package_payload_paths:
        issues.append("package wheel artifact RECORD must list the document_kv_cache package payload")
    return tuple(issues)


def _wheel_record_file_field_issues(
    wheel_zip: zipfile.ZipFile,
    path: str,
    *,
    hash_value: str,
    size_value: str,
) -> tuple[str, ...]:
    is_record_file = path.endswith(".dist-info/RECORD")
    issues: list[str] = []
    if not is_record_file and not hash_value:
        issues.append(f"package wheel artifact RECORD path {path!r} must include a hash")
    if hash_value:
        issues.extend(_wheel_record_hash_issues(wheel_zip, path, hash_value))
    if not is_record_file and not size_value:
        issues.append(f"package wheel artifact RECORD path {path!r} must include a size")
    if size_value:
        try:
            parsed_size = int(size_value)
        except ValueError:
            issues.append(f"package wheel artifact RECORD path {path!r} size must be an integer")
        else:
            if parsed_size != wheel_zip.getinfo(path).file_size:
                issues.append(f"package wheel artifact RECORD path {path!r} size must match the wheel payload")
    return tuple(issues)


def _wheel_record_hash_issues(wheel_zip: zipfile.ZipFile, path: str, hash_value: str) -> tuple[str, ...]:
    algorithm, separator, encoded_digest = hash_value.partition("=")
    if separator != "=" or algorithm != "sha256" or not encoded_digest:
        return (f"package wheel artifact RECORD path {path!r} hash must use sha256=<urlsafe-base64-digest>",)
    actual_digest = base64.urlsafe_b64encode(hashlib.sha256(wheel_zip.read(path)).digest()).decode("ascii").rstrip("=")
    if encoded_digest != actual_digest:
        return (f"package wheel artifact RECORD path {path!r} hash must match the wheel payload",)
    return ()


def _release_bundle_expected_package_version() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject_version = _release_bundle_package_version_from_pyproject(pyproject_path)
    if pyproject_version is not None:
        return pyproject_version
    try:
        return package_metadata.version(RELEASE_BUNDLE_PACKAGE_NAME)
    except package_metadata.PackageNotFoundError:
        return None


def _release_bundle_package_version_from_pyproject(path: Path) -> str | None:
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    try:
        project = tomllib.loads(payload.decode("utf-8")).get("project")
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(project, Mapping):
        return None
    name = project.get("name")
    version = project.get("version")
    if not isinstance(name, str) or canonicalize_name(name) != canonicalize_name(RELEASE_BUNDLE_PACKAGE_NAME):
        return None
    if not isinstance(version, str) or not version:
        return None
    return version


def _wheel_versions_match(filename_version: str, metadata_version: str) -> bool:
    try:
        return Version(filename_version) == Version(metadata_version)
    except InvalidVersion:
        return filename_version == metadata_version


def _metadata_header_values(text: str) -> dict[str, tuple[str, ...]]:
    headers: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        if raw_line == "":
            break
        if raw_line[0].isspace() or ":" not in raw_line:
            continue
        name, value = raw_line.split(":", 1)
        headers.setdefault(name.strip().lower(), []).append(value.strip())
    return {name: tuple(values) for name, values in headers.items()}


def _metadata_headers(text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw_line in text.splitlines():
        if raw_line == "":
            break
        if raw_line[0].isspace() or ":" not in raw_line:
            continue
        name, value = raw_line.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    return headers


def _preflight_release_bundle_destinations(
    *,
    prepared_artifacts: Sequence[_PreparedReleaseBundleArtifact],
    bundle_dir: Path,
    overwrite: bool,
) -> None:
    destinations: dict[Path, str] = {}
    for prepared in prepared_artifacts:
        local_source_path = local_path(prepared.source_path)
        destination = _bundle_destination(bundle_dir, prepared)
        if destination in destinations:
            raise FileExistsError(
                f"Release bundle artifacts would collide at {destination}: "
                f"{destinations[destination]!r} and {prepared.source_path!r}"
            )
        destinations[destination] = prepared.source_path
        if destination.exists() and not overwrite and not _same_path(local_source_path, destination):
            raise FileExistsError(f"Release bundle artifact already exists: {destination}")


def _bundle_destination(bundle_dir: Path, prepared: _PreparedReleaseBundleArtifact) -> Path:
    return bundle_dir / _artifact_filename(prepared)


def _expected_artifact_sources(
    artifacts: Sequence[_PreparedReleaseBundleArtifact],
) -> tuple[dict[str, Any], ...]:
    return tuple(_artifact_source_identity(artifact) for artifact in artifacts)


def _artifact_source_identity(artifact: _PreparedReleaseBundleArtifact) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "role": artifact.role,
        "path": artifact.source_path,
        "size_bytes": len(artifact.payload),
        "sha256": hashlib.sha256(artifact.payload).hexdigest(),
    }
    if (record_type := _artifact_record_type(artifact)) is not None:
        identity["record_type"] = record_type
    if (
        artifact.role in ("engine_probe", "engine_connector_actions")
        and artifact.record is not None
        and (backend := _optional_backend(artifact.record.get("backend"))) is not None
    ):
        identity["backend"] = backend
    return identity


def _expected_input_files(
    artifacts: Sequence[_PreparedReleaseBundleArtifact],
) -> tuple[dict[str, Any], ...]:
    input_files = []
    for artifact in artifacts:
        identity = _artifact_source_identity(artifact)
        input_files.append(
            {
                "role": identity["role"],
                "path": identity["path"],
                "exists": True,
                "readable_json": True,
                **{key: value for key, value in identity.items() if key in ("record_type", "backend")},
            }
        )
    return tuple(input_files)


def _matches_expected_records(actual: Any, expected: Sequence[Mapping[str, Any]]) -> bool:
    if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes, bytearray)):
        return False
    if len(actual) != len(expected):
        return False
    return all(isinstance(item, Mapping) and _record_contains(item, expected_item) for item, expected_item in zip(actual, expected))


def _record_contains(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        if key in ("size_bytes", "sha256") and key not in actual:
            continue
        if actual.get(key) != value:
            return False
    return True


def _artifact_record(role: str, payload: bytes, source_path: str) -> Mapping[str, Any] | None:
    if role == "package_wheel":
        return None
    return _read_json_object(payload, source_path)


def _artifact_record_type(artifact: _PreparedReleaseBundleArtifact) -> str | None:
    if artifact.record is None:
        return None
    if artifact.role == "databricks_run_status":
        status_record = _databricks_run_status_record(artifact.record)
        return _optional_str(status_record.get("record_type")) if status_record is not None else None
    if artifact.role == "github_governance":
        governance_record = _github_governance_record(artifact.record)
        return _optional_str(governance_record.get("record_type")) if governance_record is not None else None
    return _optional_str(artifact.record.get("record_type"))


def _read_json_object(payload: bytes, source_path: str) -> Mapping[str, Any]:
    try:
        record = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Release bundle artifact {source_path!r} must be readable UTF-8 JSON") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"Release bundle artifact {source_path!r} JSON root must be an object")
    return record


def _artifact_filename(prepared: _PreparedReleaseBundleArtifact) -> str:
    role = prepared.role
    record = prepared.record
    index = prepared.index
    if role == "engine_probe":
        backend = _optional_backend(record.get("backend")) if record is not None else None
        backend = backend or f"record_{index + 1:02d}"
        return f"engine_probe_{index + 1:02d}_{backend}.json"
    if role == "engine_connector_actions":
        backend = _optional_backend(record.get("backend")) if record is not None else None
        backend = backend or f"record_{index + 1:02d}"
        return f"engine_connector_actions_{index + 1:02d}_{backend}.json"
    if role == "v1_benchmark":
        return "v1_benchmark.json"
    if role == "storage_benchmark":
        return "storage_benchmark.json"
    if role == "release_evidence":
        return "release_evidence.json"
    if role == "preflight":
        return "release_inputs.json"
    if role == "plan_execution":
        return f"plan_execution_{index + 1:02d}.json"
    if role == "databricks_run_status":
        return f"databricks_run_status_{index + 1:02d}.json"
    if role == "package_wheel":
        return Path(prepared.source_path).name
    if role == "pr_evidence":
        return f"pr_evidence_{index + 1:02d}.json"
    if role == "github_governance":
        return "github_governance.json"
    if role == "repository_hygiene":
        return "repository_hygiene.json"
    if role == "native_probe_factories":
        return f"native_probe_factories_{index + 1:02d}.json"
    raise ValueError(f"Unsupported release bundle artifact role {role!r}")


def _artifact_to_record(artifact: ReleaseBundleArtifact) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": artifact.role,
        "source_path": artifact.source_path,
        "bundled_path": artifact.bundled_path,
        "size_bytes": artifact.size_bytes,
        "sha256": artifact.sha256,
    }
    if artifact.record_type is not None:
        record["record_type"] = artifact.record_type
    if artifact.backend is not None:
        record["backend"] = artifact.backend
    if artifact.package_name is not None:
        record["package_name"] = artifact.package_name
    if artifact.package_version is not None:
        record["package_version"] = artifact.package_version
    return record


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_backend(value: Any) -> str | None:
    backend = _optional_str(value)
    return backend if backend in REQUIRED_ENGINE_PROBE_BACKENDS else None


def _matches_required_backend_set(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    return len(value) == len(REQUIRED_ENGINE_PROBE_BACKENDS) and set(value) == set(REQUIRED_ENGINE_PROBE_BACKENDS)


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a checksummed Document KV release evidence bundle.")
    parser.add_argument("--v1-benchmark-json", required=True)
    parser.add_argument("--storage-benchmark-json", required=True)
    parser.add_argument("--engine-probe-json", action="append", default=[])
    parser.add_argument("--engine-actions-json", action="append", default=[])
    parser.add_argument("--release-evidence-json")
    parser.add_argument("--preflight-json")
    parser.add_argument("--plan-execution-json", action="append", default=[])
    parser.add_argument("--databricks-run-status-json", action="append", default=[])
    parser.add_argument("--package-wheel")
    parser.add_argument("--pr-evidence-json", action="append", default=[])
    parser.add_argument("--github-governance-json")
    parser.add_argument("--repository-hygiene-json")
    parser.add_argument("--native-probe-factories-json", action="append", default=[])
    parser.add_argument(
        "--require-complete-v1",
        action="store_true",
        help=(
            "Require the full V1 release artifact set: release/preflight sidecars, "
            "vLLM/SGLang connector actions, plan execution, Databricks status for "
            "benchmark/storage/engine-probe runs, tested wheel, PR evidence, governance, "
            "repository hygiene, and supported native probe factory diagnostics."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    bundle = build_release_bundle(
        v1_benchmark_json=args.v1_benchmark_json,
        storage_benchmark_json=args.storage_benchmark_json,
        engine_probe_jsons=args.engine_probe_json,
        engine_actions_jsons=args.engine_actions_json,
        release_evidence_json=args.release_evidence_json,
        preflight_json=args.preflight_json,
        plan_execution_jsons=args.plan_execution_json,
        databricks_run_status_jsons=args.databricks_run_status_json,
        package_wheel=args.package_wheel,
        pr_evidence_jsons=args.pr_evidence_json,
        github_governance_json=args.github_governance_json,
        repository_hygiene_json=args.repository_hygiene_json,
        native_probe_factories_jsons=args.native_probe_factories_json,
        require_complete_v1=args.require_complete_v1,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    if args.output_json:
        write_release_bundle_manifest_json(bundle, args.output_json)
    else:
        print(json.dumps(release_bundle_to_record(bundle), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
