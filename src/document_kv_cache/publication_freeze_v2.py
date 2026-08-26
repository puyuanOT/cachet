"""Additive source-freeze and local-preflight authority for qualification v2.

The retained v1 source closure and seven-check preflight remain byte and API
stable.  This module owns the disjoint eight-artifact v2 closure and never
submits a Databricks run or mutates the GPU-hour ledger.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any, Final, Protocol, cast
import zipfile

from document_kv_cache.databricks_runs import (
    DatabricksWorkspaceConfig,
    list_databricks_volume_directory,
    require_databricks_current_user_name,
)
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256,
    FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
    FLASHINFER_PATCHED_MANIFEST_SIZE,
    FLASHINFER_PATCHED_WHEEL_SHA256,
    FLASHINFER_PATCHED_WHEEL_SIZE,
    FLASHINFER_SOURCE_WHEEL_SHA256,
    FLASHINFER_SOURCE_WHEEL_SIZE,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    canonical_gpu_qualification_json,
)
from document_kv_cache.gpu_qualification_databricks import (
    _qualification_single_user_name_from_payloads,
    _verify_input_bundle_byte_closure,
)
from document_kv_cache.gpu_qualification_databricks_v2 import (
    GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT,
    GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256,
    GPU_QUALIFICATION_V2_DATABRICKS_PARAMETERS_MAX_BYTES,
    render_gpu_qualification_submit_payloads_v2,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_ARTIFACT_KEYS,
    GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS,
    GPUQualificationArtifactPinsV2,
    build_local_preflight_evidence_v2,
    pins_from_gpu_qualification_plan_v2,
    validate_gpu_qualification_plan_v2_record,
    validate_local_preflight_evidence_v2_record,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_ID,
    validate_publication_campaign_plan_record,
)
import document_kv_cache.publication_freeze as freeze_v1
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE,
    VLLM_PATCHED_MANIFEST_SHA256,
    VLLM_PATCHED_MANIFEST_SIZE,
    VLLM_PATCHED_WHEEL_SHA256,
    VLLM_PATCHED_WHEEL_SIZE,
    VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_BASE_LOCK_FILENAME,
    VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SIZE,
    VLLM_RUNTIME_LOCK_SHA256,
    VLLM_RUNTIME_SOURCE_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_SOURCE_LOCK_HASH_COUNT,
    VLLM_RUNTIME_SOURCE_LOCK_SIZE,
    validate_flashinfer_direct_base_lock,
    validate_vllm_flashinfer_runtime_artifact_closure,
)


PUBLICATION_SOURCE_CLOSURE_V2_RECORD_TYPE: Final = (
    "cachet.publication_source_closure.v2"
)
PUBLICATION_SOURCE_CLOSURE_V2_SCHEMA_VERSION: Final = 2
GPU_QUALIFICATION_LOCAL_CHECK_V2_RECORD_TYPE: Final = (
    "cachet.gpu_qualification.local_check_evidence.v2"
)
GPU_QUALIFICATION_LOCAL_CHECK_V2_SCHEMA_VERSION: Final = 2

_SOURCE_FILE_ROLES: Final = (
    "cachet_package_wheel",
    "cachet_source_distribution",
    "git_source_archive",
    "gpu_qualification_bootstrap",
)
_SOURCE_REFERENCE_ROLES: Final = (
    "runtime_source_lock",
    "runtime_lock",
    "runtime_lock_input",
    "campaign_plan",
    "latency_handoff_plan",
    "full_score_inventory",
    "full_score_shard_plan",
    "patched_vllm_wheel",
    "patched_vllm_manifest",
    "pristine_flashinfer_wheel",
    "patched_flashinfer_wheel",
    "patched_flashinfer_manifest",
    "runtime_closure_manifest",
)
_PACKAGE_ROOTS: Final = (
    "cachet",
    "document_kv_cache",
    "sglang_kv_injection",
    "vllm_kv_injection",
)
_SOURCE_KEYS: Final = frozenset(
    {
        "build",
        "campaign_id",
        "closed_record_sha256",
        "files",
        "git",
        "package_payload_closure",
        "record_type",
        "references",
        "runtime",
        "schema_version",
    }
)
_SOURCE_RUNTIME_KEYS: Final = frozenset(
    {
        "base_lock",
        "flashinfer",
        "input_bundle_sha256",
        "runtime_closure",
        "source_lock",
        "vllm",
    }
)
_CHECK_KEYS: Final = frozenset(
    {
        "check_id",
        "checked_at_utc",
        "command",
        "environment",
        "exit_code",
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
_CHECK_INPUT_LABELS: Final = {
    "canonical_plan_schema": (
        "plan_json",
        "artifact_uris_json",
        "submit_payloads_json",
    ),
    "runtime_lock_require_hashes": (
        "runtime_source_lock",
        "runtime_lock",
        "package_wheel",
    ),
    "patched_wheel_record_and_manifest": (
        "patched_vllm_wheel",
        "patched_vllm_manifest",
        "pristine_flashinfer_wheel",
        "patched_flashinfer_wheel",
        "patched_flashinfer_manifest",
    ),
    "runtime_artifact_closure": (
        "runtime_source_lock",
        "runtime_lock",
        "patched_vllm_wheel",
        "patched_vllm_manifest",
        "pristine_flashinfer_wheel",
        "patched_flashinfer_wheel",
        "patched_flashinfer_manifest",
        "runtime_closure_manifest",
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
_COMMAND_CHECK_IDS: Final = ("unit_tests", "ruff", "mypy")
_PYTHON_CHECK_IDS: Final = GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS[:5]
_V2_STATIC_ANALYSIS_TARGETS: Final = tuple(
    dict.fromkeys(
        (
            *freeze_v1._STATIC_ANALYSIS_TARGETS,  # noqa: SLF001
            "src/document_kv_cache/gpu_qualification_v2.py",
            "src/document_kv_cache/gpu_qualification_databricks_v2.py",
            "src/document_kv_cache/flashinfer_wheel_repack.py",
            "src/document_kv_cache/runtime_artifact_closure.py",
            "src/document_kv_cache/publication_freeze_v2.py",
        )
    )
)


@dataclass(frozen=True, slots=True)
class PublicationSourceClosureInputsV2:
    """Clean source tree, output root, and all v2 runtime references."""

    repository_root: Path
    artifact_output_root: Path
    runtime_source_lock: Path
    runtime_lock: Path
    runtime_lock_input: Path
    campaign_plan: Path
    latency_handoff_plan: Path
    full_score_inventory: Path
    full_score_shard_plan: Path
    patched_vllm_wheel: Path
    patched_vllm_manifest: Path
    pristine_flashinfer_wheel: Path
    patched_flashinfer_wheel: Path
    patched_flashinfer_manifest: Path
    runtime_closure_manifest: Path


@dataclass(frozen=True, slots=True)
class GPUQualificationLocalPreflightInputsV2:
    """Exact local artifacts consumed by the eight v2 checks."""

    repository_root: Path
    plan_json: Path
    artifact_uris_json: Path
    submit_payloads_json: Path
    source_closure_json: Path
    source_artifact_root: Path
    package_wheel: Path
    runner: Path
    input_bundle: Path
    runtime_source_lock: Path
    runtime_lock: Path
    patched_vllm_wheel: Path
    patched_vllm_manifest: Path
    pristine_flashinfer_wheel: Path
    patched_flashinfer_wheel: Path
    patched_flashinfer_manifest: Path
    runtime_closure_manifest: Path
    python_executable: Path = freeze_v1._DEFAULT_PYTHON_EXECUTABLE  # noqa: SLF001
    ruff_executable: Path = freeze_v1._DEFAULT_RUFF_EXECUTABLE  # noqa: SLF001
    mypy_executable: Path = freeze_v1._DEFAULT_MYPY_EXECUTABLE  # noqa: SLF001


class CompletedCommand(Protocol):
    """Bounded command result used by the private test seam."""

    returncode: int
    stdout: bytes
    stderr: bytes


CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str]], CompletedCommand]


def build_publication_source_closure_v2(
    inputs: PublicationSourceClosureInputsV2,
) -> dict[str, Any]:
    """Build and validate one deterministic v2 publication source closure."""

    if not isinstance(inputs, PublicationSourceClosureInputsV2):
        raise TypeError("inputs must be PublicationSourceClosureInputsV2")
    root = freeze_v1._regular_directory(  # noqa: SLF001
        inputs.repository_root, "repository_root"
    )
    git = freeze_v1._git_identity(root)  # noqa: SLF001
    freeze_v1._require_freeze_toolchain()  # noqa: SLF001
    freeze_v1._require_freeze_build_system(root)  # noqa: SLF001
    _validate_source_references(inputs, root=root)
    first_build, second_build = freeze_v1._build_package_twice(  # noqa: SLF001
        root,
        commit=cast(str, git["commit"]),
        source_date_epoch=cast(int, git["source_date_epoch"]),
    )
    freeze_v1._require_matching_build_outputs(  # noqa: SLF001
        first_build, second_build
    )
    artifact_root = freeze_v1._create_directory_exclusive(  # noqa: SLF001
        inputs.artifact_output_root,
        "v2 source artifact output root",
    )
    package_wheel = artifact_root / first_build.wheel_name
    source_distribution = artifact_root / first_build.sdist_name
    freeze_v1._write_exclusive(  # noqa: SLF001
        package_wheel, first_build.wheel_bytes, "v2 package wheel"
    )
    freeze_v1._write_exclusive(  # noqa: SLF001
        source_distribution,
        first_build.sdist_bytes,
        "v2 source distribution",
    )
    git_source_archive = artifact_root / f"cachet-{git['commit']}.tar.gz"
    freeze_v1._write_deterministic_git_archive(  # noqa: SLF001
        root,
        commit=cast(str, git["commit"]),
        output_path=git_source_archive,
    )
    bootstrap_runner = artifact_root / "gpu-qualification-bootstrap-v2.py"
    freeze_v1._write_exclusive(  # noqa: SLF001
        bootstrap_runner,
        GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT.encode("utf-8"),
        "GPU qualification v2 bootstrap",
    )
    files = [
        freeze_v1._source_file_record(  # noqa: SLF001
            package_wheel, "cachet_package_wheel"
        ),
        freeze_v1._source_file_record(  # noqa: SLF001
            source_distribution, "cachet_source_distribution"
        ),
        freeze_v1._source_file_record(  # noqa: SLF001
            git_source_archive, "git_source_archive"
        ),
        freeze_v1._source_file_record(  # noqa: SLF001
            bootstrap_runner, "gpu_qualification_bootstrap"
        ),
    ]
    references = [
        freeze_v1._source_reference_record(root, path, role)  # noqa: SLF001
        for path, role in _source_reference_inputs(inputs)
    ]
    record: dict[str, Any] = {
        "build": {
            "build_backend": freeze_v1.PUBLICATION_FREEZE_BUILD_BACKEND,
            "build_frontend": freeze_v1.PUBLICATION_FREEZE_BUILD_FRONTEND,
            "python": freeze_v1.PUBLICATION_FREEZE_PYTHON,
            "source_date_epoch": git["source_date_epoch"],
        },
        "campaign_id": PUBLICATION_CAMPAIGN_ID,
        "closed_record_sha256": "",
        "files": files,
        "git": {
            "branch": git["branch"],
            "commit": git["commit"],
            "commit_tree": git["commit_tree"],
            "dirty": False,
        },
        "record_type": PUBLICATION_SOURCE_CLOSURE_V2_RECORD_TYPE,
        "package_payload_closure": _require_v2_package_source_equality(
            repository_root=root,
            package_wheel=package_wheel,
            source_distribution=source_distribution,
            git_source_archive=git_source_archive,
        ),
        "references": references,
        "runtime": _v2_runtime_identity(),
        "schema_version": PUBLICATION_SOURCE_CLOSURE_V2_SCHEMA_VERSION,
    }
    record["closed_record_sha256"] = freeze_v1._closed_record_sha256(  # noqa: SLF001
        record
    )
    _validate_publication_source_closure_v2_record(
        record,
        repository_root=root,
        artifact_root=artifact_root,
        explicit_artifact_paths=None,
        verify_rebuild=False,
    )
    return record


def validate_publication_source_closure_v2_record(
    record: Mapping[str, Any],
    *,
    repository_root: str | Path,
    artifact_root: str | Path,
    explicit_artifact_paths: Mapping[str, str | Path] | None = None,
) -> None:
    """Require the complete v2 source, package, and runtime closure."""

    _validate_publication_source_closure_v2_record(
        record,
        repository_root=repository_root,
        artifact_root=artifact_root,
        explicit_artifact_paths=explicit_artifact_paths,
        verify_rebuild=True,
    )


def _validate_publication_source_closure_v2_record(
    record: Mapping[str, Any],
    *,
    repository_root: str | Path,
    artifact_root: str | Path,
    explicit_artifact_paths: Mapping[str, str | Path] | None,
    verify_rebuild: bool,
) -> None:
    normalized = _mapping_copy(record, "v2 source closure")
    _require_exact_keys(normalized, _SOURCE_KEYS, "v2 source closure")
    if normalized.get("record_type") != PUBLICATION_SOURCE_CLOSURE_V2_RECORD_TYPE:
        raise ValueError("v2 source closure record_type differs")
    if (
        type(normalized.get("schema_version")) is not int
        or normalized.get("schema_version")
        != PUBLICATION_SOURCE_CLOSURE_V2_SCHEMA_VERSION
    ):
        raise ValueError("v2 source closure schema_version differs")
    if normalized.get("campaign_id") != PUBLICATION_CAMPAIGN_ID:
        raise ValueError("v2 source closure campaign_id differs")
    freeze_v1._require_closed_digest(normalized, "v2 source closure")  # noqa: SLF001
    root = freeze_v1._regular_directory(  # noqa: SLF001
        Path(repository_root), "repository_root"
    )
    artifacts = freeze_v1._regular_directory(  # noqa: SLF001
        Path(artifact_root), "artifact_root"
    )
    freeze_v1._require_freeze_toolchain()  # noqa: SLF001
    freeze_v1._require_freeze_build_system(root)  # noqa: SLF001
    expected_git = freeze_v1._git_identity(root)  # noqa: SLF001
    build = _required_mapping(normalized, "build")
    expected_build = {
        "build_backend": freeze_v1.PUBLICATION_FREEZE_BUILD_BACKEND,
        "build_frontend": freeze_v1.PUBLICATION_FREEZE_BUILD_FRONTEND,
        "python": freeze_v1.PUBLICATION_FREEZE_PYTHON,
        "source_date_epoch": expected_git["source_date_epoch"],
    }
    if set(build) != {"build_backend", "build_frontend", "python", "source_date_epoch"} or dict(build) != expected_build:
        raise ValueError("v2 source closure build identity differs")
    git = _required_mapping(normalized, "git")
    if set(git) != {"branch", "commit", "commit_tree", "dirty"} or dict(git) != {
        "branch": expected_git["branch"],
        "commit": expected_git["commit"],
        "commit_tree": expected_git["commit_tree"],
        "dirty": False,
    }:
        raise ValueError("v2 source closure Git identity differs")
    files = _mapping_sequence(normalized.get("files"), "v2 source closure files")
    if tuple(item.get("role") for item in files) != _SOURCE_FILE_ROLES:
        raise ValueError("v2 source closure file roles differ")
    explicit = dict(explicit_artifact_paths or {})
    observed_relative: set[str] = set()
    resolved_files: dict[str, Path] = {}
    for item in files:
        _require_exact_keys(
            item,
            frozenset({"byte_count", "relative_path", "role", "sha256"}),
            "v2 source closure file",
        )
        relative = freeze_v1._safe_relative_path(  # noqa: SLF001
            item.get("relative_path"), "relative_path"
        )
        if relative in observed_relative:
            raise ValueError("v2 source closure repeats a file path")
        observed_relative.add(relative)
        candidate = Path(explicit[relative]) if relative in explicit else artifacts / relative
        candidate = freeze_v1._regular_file(  # noqa: SLF001
            candidate, f"v2 source closure file {relative}"
        )
        freeze_v1._require_file_binding(  # noqa: SLF001
            candidate, item, f"v2 source closure file {relative}"
        )
        resolved_files[cast(str, item["role"])] = candidate
    freeze_v1._validate_cachet_wheel(  # noqa: SLF001
        resolved_files["cachet_package_wheel"]
    )
    freeze_v1._validate_sdist(  # noqa: SLF001
        resolved_files["cachet_source_distribution"]
    )
    freeze_v1._validate_git_archive(  # noqa: SLF001
        resolved_files["git_source_archive"],
        repository_root=root,
        commit=cast(str, git["commit"]),
        source_date_epoch=cast(int, build["source_date_epoch"]),
    )
    if freeze_v1._file_sha256(  # noqa: SLF001
        resolved_files["gpu_qualification_bootstrap"]
    ) != GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256:
        raise ValueError("v2 source closure bootstrap differs")
    package_payload_closure = _required_mapping(
        normalized, "package_payload_closure"
    )
    if dict(package_payload_closure) != _require_v2_package_source_equality(
        repository_root=root,
        package_wheel=resolved_files["cachet_package_wheel"],
        source_distribution=resolved_files["cachet_source_distribution"],
        git_source_archive=resolved_files["git_source_archive"],
    ):
        raise ValueError("v2 package payload closure differs")
    references = _mapping_sequence(
        normalized.get("references"), "v2 source closure references"
    )
    if tuple(item.get("role") for item in references) != _SOURCE_REFERENCE_ROLES:
        raise ValueError("v2 source closure reference roles differ")
    reference_paths: dict[str, Path] = {}
    for item in references:
        _require_exact_keys(
            item,
            frozenset({"byte_count", "path", "role", "sha256"}),
            "v2 source closure reference",
        )
        relative = freeze_v1._safe_relative_path(  # noqa: SLF001
            item.get("path"), "reference path"
        )
        candidate = freeze_v1._regular_file(  # noqa: SLF001
            root / relative, f"v2 source reference {relative}"
        )
        freeze_v1._require_file_binding(  # noqa: SLF001
            candidate, item, f"v2 source reference {relative}"
        )
        reference_paths[cast(str, item["role"])] = candidate
    _validate_reference_paths(reference_paths, repository_root=root)
    runtime = _required_mapping(normalized, "runtime")
    _require_exact_keys(runtime, _SOURCE_RUNTIME_KEYS, "v2 source runtime")
    if canonical_gpu_qualification_json(runtime) != canonical_gpu_qualification_json(
        _v2_runtime_identity()
    ):
        raise ValueError("v2 source closure runtime identity differs")
    if verify_rebuild:
        first_build, second_build = freeze_v1._build_package_twice(  # noqa: SLF001
            root,
            commit=cast(str, git["commit"]),
            source_date_epoch=cast(int, build["source_date_epoch"]),
        )
        freeze_v1._require_matching_build_outputs(  # noqa: SLF001
            first_build, second_build
        )
        freeze_v1._require_file_bytes(  # noqa: SLF001
            resolved_files["cachet_package_wheel"],
            first_build.wheel_bytes,
            "v2 package wheel versus clean-tree rebuild",
        )
        freeze_v1._require_file_bytes(  # noqa: SLF001
            resolved_files["cachet_source_distribution"],
            first_build.sdist_bytes,
            "v2 source distribution versus clean-tree rebuild",
        )


def write_publication_source_closure_v2_json(
    record: Mapping[str, Any],
    path: str | Path,
    *,
    repository_root: str | Path,
    artifact_root: str | Path,
    explicit_artifact_paths: Mapping[str, str | Path] | None = None,
) -> None:
    """Validate and exclusively write one canonical v2 source manifest."""

    validate_publication_source_closure_v2_record(
        record,
        repository_root=repository_root,
        artifact_root=artifact_root,
        explicit_artifact_paths=explicit_artifact_paths,
    )
    freeze_v1._write_exclusive(  # noqa: SLF001
        Path(path),
        freeze_v1._canonical_json_bytes(record, pretty=True),  # noqa: SLF001
        "v2 source closure",
    )


def run_gpu_qualification_local_preflight_v2(
    inputs: GPUQualificationLocalPreflightInputsV2,
    output_root: str | Path,
) -> dict[str, Any]:
    """Execute, persist, and seal all eight v2 local checks."""

    return _run_gpu_qualification_local_preflight_v2(
        inputs,
        output_root,
        command_runner=_run_command,
        now=_utc_now,
    )


def validate_gpu_qualification_local_preflight_bundle_v2(
    path: str | Path,
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    workspace_config: DatabricksWorkspaceConfig,
    require_fresh_workspace: bool = True,
) -> dict[str, Any]:
    """Replay and authenticate a closed eight-check v2 preflight bundle."""

    plan = _mapping_copy(plan_record, "v2 qualification plan")
    pins = pins_from_gpu_qualification_plan_v2(plan)
    _require_fixed_v2_pins(pins)
    validate_gpu_qualification_plan_v2_record(
        plan,
        expected_campaign_id=PUBLICATION_CAMPAIGN_ID,
        expected_artifact_pins=pins,
    )
    if not isinstance(workspace_config, DatabricksWorkspaceConfig):
        raise TypeError("workspace_config must be DatabricksWorkspaceConfig")
    if type(require_fresh_workspace) is not bool:
        raise TypeError("require_fresh_workspace must be a bool")
    submitted_payload_bytes = _canonical_submit_payload_closure_bytes(
        submit_payloads
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
    with tempfile.TemporaryDirectory(
        prefix="cachet-gpuq-v2-live-preflight-"
    ) as temporary:
        _run_gpu_qualification_local_preflight_v2(
            inputs,
            Path(temporary).resolve() / "preflight",
            command_runner=_run_command,
            now=_utc_now,
        )
    _validate_live_workspace_and_remote_artifacts_v2(
        workspace_config,
        inputs=inputs,
        plan=plan,
        single_user_name=single_user_name,
        require_fresh_workspace=require_fresh_workspace,
    )
    after = _preflight_bundle_file_hashes(evidence_path)
    if after != before:
        raise ValueError("v2 local preflight bundle changed during live replay")
    final_evidence, final_inputs = _validate_preflight_bundle_structural(
        evidence_path,
        plan=plan,
    )
    if final_evidence != evidence or final_inputs != inputs:
        raise ValueError("v2 local preflight bindings changed during replay")
    _require_submit_payload_closure_binding(
        final_inputs,
        submitted_payload_bytes=submitted_payload_bytes,
    )
    return final_evidence


def _run_gpu_qualification_local_preflight_v2(
    inputs: GPUQualificationLocalPreflightInputsV2,
    output_root: str | Path,
    *,
    command_runner: CommandRunner,
    now: Callable[[], datetime],
    verify_source_rebuild: bool = True,
) -> dict[str, Any]:
    if not isinstance(inputs, GPUQualificationLocalPreflightInputsV2):
        raise TypeError("inputs must be GPUQualificationLocalPreflightInputsV2")
    _require_canonical_preflight_tool_paths(inputs)
    root = Path(output_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"v2 preflight output root already exists: {root}")
    freeze_v1._create_directory_exclusive(  # noqa: SLF001
        root, "v2 preflight output root"
    )
    plan = freeze_v1._read_canonical_json(  # noqa: SLF001
        inputs.plan_json,
        pretty=False,
        label="v2 qualification plan",
    )
    pins = pins_from_gpu_qualification_plan_v2(plan)
    _require_fixed_v2_pins(pins)
    validate_gpu_qualification_plan_v2_record(
        plan,
        expected_campaign_id=PUBLICATION_CAMPAIGN_ID,
        expected_artifact_pins=pins,
    )
    plan_sha256 = _required_sha256(plan.get("closed_record_sha256"), "plan digest")
    python_checks = _python_check_specs(
        inputs,
        plan=plan,
        pins=pins,
        verify_source_rebuild=verify_source_rebuild,
    )
    check_hashes: dict[str, str] = {}
    for check_id, check in python_checks:
        try:
            result, bindings = check()
        except BaseException as exc:
            failed = _local_check_record(
                check_id=check_id,
                plan_sha256=plan_sha256,
                checked_at=now(),
                command=(
                    "python-api",
                    f"document_kv_cache.publication_freeze_v2:{check_id}",
                ),
                exit_code=1,
                environment=freeze_v1._python_api_environment_identity(),  # noqa: SLF001
                inputs=[],
                result={"error": freeze_v1._bounded_error(exc)},  # noqa: SLF001
                stdout=b"",
                stderr=str(exc).encode("utf-8", errors="replace"),
                tool_identity=_python_api_identity_v2(),
                status="failed",
            )
            _write_check(root, failed)
            raise RuntimeError(f"v2 local check {check_id!r} failed") from exc
        passed = _local_check_record(
            check_id=check_id,
            plan_sha256=plan_sha256,
            checked_at=now(),
            command=(
                "python-api",
                f"document_kv_cache.publication_freeze_v2:{check_id}",
            ),
            exit_code=0,
            environment=freeze_v1._python_api_environment_identity(),  # noqa: SLF001
            inputs=bindings,
            result=result,
            stdout=b"",
            stderr=b"",
            tool_identity=_python_api_identity_v2(),
            status="passed",
        )
        check_hashes[check_id] = _write_check(root, passed)
    environment = freeze_v1._preflight_environment(  # noqa: SLF001
        inputs.repository_root
    )
    for check_id, command, version_command, expected_version in _command_check_specs(
        inputs
    ):
        executable = freeze_v1._resolve_executable(  # noqa: SLF001
            inputs.repository_root, Path(command[0])
        )
        normalized_command = (str(executable), *command[1:])
        normalized_version = (str(executable), *version_command[1:])
        version = command_runner(
            normalized_version, inputs.repository_root, environment
        )
        freeze_v1._require_bounded_command(  # noqa: SLF001
            version, f"v2 {check_id} version"
        )
        version_text = version.stdout.decode("utf-8", errors="strict").strip()
        if version.returncode != 0 or version.stderr or version_text != expected_version:
            raise RuntimeError(f"v2 {check_id} tool identity differs")
        completed = command_runner(
            normalized_command, inputs.repository_root, environment
        )
        freeze_v1._require_bounded_command(completed, f"v2 {check_id}")  # noqa: SLF001
        if check_id == "unit_tests" and completed.returncode == 0:
            freeze_v1._require_exact_pytest_completion(completed.stdout)  # noqa: SLF001
        status = "passed" if completed.returncode == 0 else "failed"
        record = _local_check_record(
            check_id=check_id,
            plan_sha256=plan_sha256,
            checked_at=now(),
            command=normalized_command,
            exit_code=completed.returncode,
            environment=environment,
            inputs=[freeze_v1._repository_binding(inputs.repository_root)],  # noqa: SLF001
            result=_command_check_result(
                check_id,
                expected_version=expected_version,
                repository_root=inputs.repository_root,
            ),
            stdout=completed.stdout,
            stderr=completed.stderr,
            tool_identity={
                "executable_path": str(executable),
                "executable_sha256": freeze_v1._file_sha256(executable),  # noqa: SLF001
                "version": version_text,
                "version_command": list(normalized_version),
            },
            status=status,
        )
        check_hashes[check_id] = _write_check(root, record)
        if completed.returncode != 0:
            raise RuntimeError(
                f"v2 local command {check_id!r} returned {completed.returncode}"
            )
    if tuple(check_hashes) != GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS:
        raise RuntimeError("v2 preflight did not execute exact check coverage")
    _validate_written_checks_before_seal(
        root,
        plan=plan,
        expected_hashes=check_hashes,
    )
    evidence = build_local_preflight_evidence_v2(
        plan_sha256=plan_sha256,
        completed_at_utc=_timestamp(now()),
        check_evidence_sha256=check_hashes,
    )
    evidence_path = root / "local-preflight-evidence.json"
    freeze_v1._write_exclusive(  # noqa: SLF001
        evidence_path,
        freeze_v1._canonical_json_bytes(evidence, pretty=False),  # noqa: SLF001
        "v2 local preflight evidence",
    )
    try:
        validated, _ = _validate_preflight_bundle_structural(
            evidence_path,
            plan=plan,
        )
    except BaseException:
        if evidence_path.is_file() and not evidence_path.is_symlink():
            evidence_path.unlink()
        raise
    return validated


def _python_check_specs(
    inputs: GPUQualificationLocalPreflightInputsV2,
    *,
    plan: Mapping[str, Any],
    pins: GPUQualificationArtifactPinsV2,
    verify_source_rebuild: bool,
) -> tuple[
    tuple[str, Callable[[], tuple[dict[str, Any], list[dict[str, Any]]]]], ...
]:
    return (
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
            lambda: _check_patched_wheels(inputs, pins),
        ),
        (
            "runtime_artifact_closure",
            lambda: _check_runtime_artifact_closure(inputs, pins),
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


def _check_canonical_plan(
    inputs: GPUQualificationLocalPreflightInputsV2,
    plan: Mapping[str, Any],
    pins: GPUQualificationArtifactPinsV2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    uris_record = freeze_v1._read_canonical_json(  # noqa: SLF001
        inputs.artifact_uris_json,
        pretty=False,
        label="v2 artifact URI record",
    )
    if set(uris_record) != {"artifact_uris", "output_root", "plan_sha256"}:
        raise ValueError("v2 artifact URI record has an open schema")
    if uris_record.get("plan_sha256") != plan.get("closed_record_sha256"):
        raise ValueError("v2 artifact URI record plan digest differs")
    uris = _required_mapping(uris_record, "artifact_uris")
    if tuple(uris) != GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        raise ValueError("v2 artifact URI record lacks canonical eight roles")
    payloads_raw = freeze_v1._read_canonical_json_value(  # noqa: SLF001
        inputs.submit_payloads_json, label="v2 submit payloads"
    )
    if not isinstance(payloads_raw, list):
        raise ValueError("v2 submit payloads must be an array")
    single_user_name = _qualification_single_user_name_from_payloads(payloads_raw)
    expected = render_gpu_qualification_submit_payloads_v2(
        plan,
        single_user_name=single_user_name,
        artifact_uris={
            key: _required_string(uris, key)
            for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS
        },
        output_root=_required_string(uris_record, "output_root"),
    )
    if freeze_v1._canonical_json_bytes(  # noqa: SLF001
        payloads_raw, pretty=False
    ) != freeze_v1._canonical_json_bytes(list(expected), pretty=False):  # noqa: SLF001
        raise ValueError("v2 submit payload bytes differ from the renderer")
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
    if len(expected) != 14 or max(sizes) > GPU_QUALIFICATION_V2_DATABRICKS_PARAMETERS_MAX_BYTES:
        raise ValueError("v2 payload count or parameter size differs")
    return (
        {
            "artifact_pins": pins.to_record(),
            "authority_scope": "local_v2_renderer_and_payload_size_only",
            "job_count": len(expected),
            "max_parameters_json_bytes": max(sizes),
            "min_parameters_json_bytes": min(sizes),
            "node_types": ["g5.8xlarge", "g6.8xlarge", "g6e.4xlarge"],
        },
        freeze_v1._bindings(  # noqa: SLF001
            (inputs.plan_json, "plan_json"),
            (inputs.artifact_uris_json, "artifact_uris_json"),
            (inputs.submit_payloads_json, "submit_payloads_json"),
        ),
    )


def _check_runtime_lock(
    inputs: GPUQualificationLocalPreflightInputsV2,
    pins: GPUQualificationArtifactPinsV2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed = validate_flashinfer_direct_base_lock(
        inputs.runtime_source_lock,
        inputs.runtime_lock,
    )
    if observed.get("sha256") != pins.runtime_lock_sha256:
        raise ValueError("v2 runtime lock differs from the plan")
    package_wheel = freeze_v1._regular_file(  # noqa: SLF001
        inputs.package_wheel, "package_wheel"
    )
    with zipfile.ZipFile(package_wheel) as archive:
        packaged = archive.read(
            f"document_kv_cache/runtime_locks/{VLLM_RUNTIME_BASE_LOCK_FILENAME}"
        )
    lock = freeze_v1._regular_file(inputs.runtime_lock, "runtime_lock")  # noqa: SLF001
    if packaged != lock.read_bytes():
        raise ValueError("v2 package wheel base lock differs")
    return (
        {
            "authority_scope": "local_v2_runtime_lock_only",
            **dict(observed),
            "packaged_lock_matches": True,
        },
        freeze_v1._bindings(  # noqa: SLF001
            (inputs.runtime_source_lock, "runtime_source_lock"),
            (inputs.runtime_lock, "runtime_lock"),
            (inputs.package_wheel, "package_wheel"),
        ),
    )


def _check_patched_wheels(
    inputs: GPUQualificationLocalPreflightInputsV2,
    pins: GPUQualificationArtifactPinsV2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = _runtime_artifact_paths(inputs)
    _require_runtime_artifact_file_pins(paths, pins=pins)
    closure = validate_vllm_flashinfer_runtime_artifact_closure(
        source_lock=paths["runtime_source_lock"],
        base_lock=paths["runtime_lock"],
        vllm_wheel=paths["patched_vllm_wheel"],
        vllm_manifest=paths["patched_vllm_manifest"],
        pristine_flashinfer_wheel=paths["pristine_flashinfer_wheel"],
        patched_flashinfer_wheel=paths["patched_flashinfer_wheel"],
        flashinfer_manifest=paths["patched_flashinfer_manifest"],
        closure_manifest=paths["runtime_closure_manifest"],
    )
    vllm_members, vllm_rows = freeze_v1._audit_wheel_record(  # noqa: SLF001
        paths["patched_vllm_wheel"]
    )
    flashinfer_members, flashinfer_rows = freeze_v1._audit_wheel_record(  # noqa: SLF001
        paths["patched_flashinfer_wheel"]
    )
    return (
        {
            "authority_scope": "local_v2_wheels_and_manifests_only",
            "flashinfer_member_count": flashinfer_members,
            "flashinfer_record_rows_valid": flashinfer_rows,
            "runtime_closure_closed_record_sha256": closure[
                "closed_record_sha256"
            ],
            "vllm_member_count": vllm_members,
            "vllm_record_rows_valid": vllm_rows,
        },
        freeze_v1._bindings(  # noqa: SLF001
            (inputs.patched_vllm_wheel, "patched_vllm_wheel"),
            (inputs.patched_vllm_manifest, "patched_vllm_manifest"),
            (inputs.pristine_flashinfer_wheel, "pristine_flashinfer_wheel"),
            (inputs.patched_flashinfer_wheel, "patched_flashinfer_wheel"),
            (inputs.patched_flashinfer_manifest, "patched_flashinfer_manifest"),
        ),
    )


def _check_runtime_artifact_closure(
    inputs: GPUQualificationLocalPreflightInputsV2,
    pins: GPUQualificationArtifactPinsV2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = _runtime_artifact_paths(inputs)
    _require_runtime_artifact_file_pins(paths, pins=pins)
    result = validate_vllm_flashinfer_runtime_artifact_closure(
        source_lock=paths["runtime_source_lock"],
        base_lock=paths["runtime_lock"],
        vllm_wheel=paths["patched_vllm_wheel"],
        vllm_manifest=paths["patched_vllm_manifest"],
        pristine_flashinfer_wheel=paths["pristine_flashinfer_wheel"],
        patched_flashinfer_wheel=paths["patched_flashinfer_wheel"],
        flashinfer_manifest=paths["patched_flashinfer_manifest"],
        closure_manifest=paths["runtime_closure_manifest"],
    )
    return (
        {
            "authority_scope": "local_v2_complete_runtime_artifact_closure",
            **dict(result),
        },
        freeze_v1._bindings(  # noqa: SLF001
            (inputs.runtime_source_lock, "runtime_source_lock"),
            (inputs.runtime_lock, "runtime_lock"),
            (inputs.patched_vllm_wheel, "patched_vllm_wheel"),
            (inputs.patched_vllm_manifest, "patched_vllm_manifest"),
            (inputs.pristine_flashinfer_wheel, "pristine_flashinfer_wheel"),
            (inputs.patched_flashinfer_wheel, "patched_flashinfer_wheel"),
            (inputs.patched_flashinfer_manifest, "patched_flashinfer_manifest"),
            (inputs.runtime_closure_manifest, "runtime_closure_manifest"),
        ),
    )


def _check_source_runner_inputs(
    inputs: GPUQualificationLocalPreflightInputsV2,
    plan: Mapping[str, Any],
    pins: GPUQualificationArtifactPinsV2,
    *,
    verify_rebuild: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = freeze_v1._read_canonical_json(  # noqa: SLF001
        inputs.source_closure_json,
        pretty=True,
        label="v2 source closure",
    )
    _validate_publication_source_closure_v2_record(
        source,
        repository_root=inputs.repository_root,
        artifact_root=inputs.source_artifact_root,
        explicit_artifact_paths=None,
        verify_rebuild=verify_rebuild,
    )
    source_digest = freeze_v1._file_sha256(inputs.source_closure_json)  # noqa: SLF001
    if source_digest != pins.cachet_source_tree_sha256:
        raise ValueError("v2 source closure differs from the plan")
    if freeze_v1._file_sha256(inputs.package_wheel) != pins.package_wheel_sha256:  # noqa: SLF001
        raise ValueError("v2 package wheel differs from the plan")
    if freeze_v1._file_sha256(inputs.runner) != pins.runner_sha256:  # noqa: SLF001
        raise ValueError("v2 runner differs from the plan")
    input_digest = _verify_input_bundle_byte_closure(
        freeze_v1._regular_directory(inputs.input_bundle, "input bundle"),  # noqa: SLF001
        expected_sha256=GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    )
    if input_digest != pins.input_bundle_sha256:
        raise ValueError("v2 input bundle differs from the plan")
    return (
        {
            "authority_scope": "local_v2_source_runner_and_input_only",
            "input_bundle_file_count": len(
                freeze_v1._regular_tree(inputs.input_bundle)  # noqa: SLF001
            ),
            "input_bundle_sha256": input_digest,
            "package_wheel_sha256": freeze_v1._file_sha256(  # noqa: SLF001
                inputs.package_wheel
            ),
            "plan_sha256": plan["closed_record_sha256"],
            "runner_sha256": freeze_v1._file_sha256(inputs.runner),  # noqa: SLF001
            "source_closed_record_sha256": source["closed_record_sha256"],
            "source_manifest_sha256": source_digest,
        },
        freeze_v1._bindings(  # noqa: SLF001
            (inputs.source_closure_json, "source_closure_json"),
            (inputs.source_artifact_root, "source_artifact_root"),
            (inputs.package_wheel, "package_wheel"),
            (inputs.runner, "runner"),
            (inputs.input_bundle, "input_bundle"),
        ),
    )


def _validate_preflight_bundle_structural(
    evidence_path: Path,
    *,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], GPUQualificationLocalPreflightInputsV2]:
    root = freeze_v1._regular_directory(  # noqa: SLF001
        evidence_path.parent, "v2 preflight bundle root"
    )
    expected_names = {
        *(f"{check_id}.json" for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS),
        "local-preflight-evidence.json",
    }
    if {item.name for item in root.iterdir()} != expected_names:
        raise ValueError("v2 preflight directory lacks exact check coverage")
    plan_sha256 = _required_sha256(plan.get("closed_record_sha256"), "plan digest")
    records: dict[str, dict[str, Any]] = {}
    check_hashes: dict[str, str] = {}
    check_times: list[datetime] = []
    for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS:
        sidecar = root / f"{check_id}.json"
        record = freeze_v1._read_canonical_json(  # noqa: SLF001
            sidecar, pretty=False, label=f"v2 {check_id}"
        )
        _validate_local_check_record(
            record,
            expected_check_id=check_id,
            expected_plan_sha256=plan_sha256,
        )
        if record.get("status") != "passed" or record.get("exit_code") != 0:
            raise ValueError(f"v2 preflight check {check_id!r} did not pass")
        records[check_id] = record
        check_hashes[check_id] = freeze_v1._file_sha256(sidecar)  # noqa: SLF001
        check_times.append(
            freeze_v1._parse_timestamp(  # noqa: SLF001
                record.get("checked_at_utc"), check_id
            )
        )
    evidence = freeze_v1._read_canonical_json(  # noqa: SLF001
        evidence_path,
        pretty=False,
        label="v2 local preflight evidence",
    )
    completion = validate_local_preflight_evidence_v2_record(
        evidence,
        plan_sha256=plan_sha256,
    )
    raw_checks = evidence.get("checks")
    if not isinstance(raw_checks, list):
        raise ValueError("v2 local preflight evidence checks are invalid")
    observed = {
        _required_string(_mapping_copy(item, "v2 evidence check"), "check_id"):
        _required_string(_mapping_copy(item, "v2 evidence check"), "evidence_sha256")
        for item in raw_checks
    }
    if observed != check_hashes:
        raise ValueError("v2 preflight evidence sidecar hashes differ")
    if check_times and completion < max(check_times):
        raise ValueError("v2 preflight completed before a child check")
    inputs = _preflight_inputs_from_sidecars(records)
    _validate_sidecar_semantics(records, inputs=inputs, plan=plan)
    return evidence, inputs


def _validate_written_checks_before_seal(
    root: Path,
    *,
    plan: Mapping[str, Any],
    expected_hashes: Mapping[str, str],
) -> None:
    expected_names = {
        f"{check_id}.json" for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS
    }
    if {item.name for item in root.iterdir()} != expected_names:
        raise RuntimeError("v2 preflight has extra or missing child sidecars")
    plan_sha256 = _required_sha256(plan.get("closed_record_sha256"), "plan digest")
    records: dict[str, dict[str, Any]] = {}
    observed_hashes: dict[str, str] = {}
    for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS:
        path = root / f"{check_id}.json"
        record = freeze_v1._read_canonical_json(  # noqa: SLF001
            path, pretty=False, label=f"v2 {check_id}"
        )
        _validate_local_check_record(
            record,
            expected_check_id=check_id,
            expected_plan_sha256=plan_sha256,
        )
        if record.get("status") != "passed" or record.get("exit_code") != 0:
            raise RuntimeError(f"v2 pre-seal check {check_id!r} did not pass")
        records[check_id] = record
        observed_hashes[check_id] = freeze_v1._file_sha256(path)  # noqa: SLF001
    if observed_hashes != dict(expected_hashes):
        raise RuntimeError("v2 pre-seal sidecar bytes changed")
    inputs = _preflight_inputs_from_sidecars(records)
    _validate_sidecar_semantics(records, inputs=inputs, plan=plan)


def _preflight_inputs_from_sidecars(
    records: Mapping[str, Mapping[str, Any]],
) -> GPUQualificationLocalPreflightInputsV2:
    if tuple(records) != GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS:
        raise ValueError("v2 sidecars are not in canonical order")
    bindings_by_label: dict[str, Mapping[str, Any]] = {}
    for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS:
        record = records[check_id]
        raw_bindings = record.get("inputs")
        if not isinstance(raw_bindings, list) or any(
            not isinstance(item, Mapping) for item in raw_bindings
        ):
            raise ValueError(f"v2 check {check_id!r} inputs are invalid")
        bindings = cast(list[Mapping[str, Any]], raw_bindings)
        if tuple(item.get("label") for item in bindings) != _CHECK_INPUT_LABELS[
            check_id
        ]:
            raise ValueError(f"v2 check {check_id!r} input coverage differs")
        for binding in bindings:
            freeze_v1._validate_path_binding(binding)  # noqa: SLF001
            label = cast(str, binding["label"])
            prior = bindings_by_label.get(label)
            if prior is not None and dict(prior) != dict(binding):
                raise ValueError(f"v2 preflight input {label!r} conflicts")
            bindings_by_label[label] = binding

    def bound_path(label: str) -> Path:
        binding = bindings_by_label.get(label)
        if binding is None or not isinstance(binding.get("path"), str):
            raise ValueError(f"v2 preflight input {label!r} is missing")
        return Path(cast(str, binding["path"]))

    command_tools: dict[str, Path] = {}
    for check_id in _COMMAND_CHECK_IDS:
        tool = records[check_id].get("tool_identity")
        if not isinstance(tool, Mapping):
            raise ValueError(f"v2 check {check_id!r} tool identity is invalid")
        executable = tool.get("executable_path")
        if not isinstance(executable, str) or not Path(executable).is_absolute():
            raise ValueError(f"v2 check {check_id!r} executable is invalid")
        command_tools[check_id] = Path(executable)
    return GPUQualificationLocalPreflightInputsV2(
        repository_root=bound_path("repository_root"),
        plan_json=bound_path("plan_json"),
        artifact_uris_json=bound_path("artifact_uris_json"),
        submit_payloads_json=bound_path("submit_payloads_json"),
        source_closure_json=bound_path("source_closure_json"),
        source_artifact_root=bound_path("source_artifact_root"),
        package_wheel=bound_path("package_wheel"),
        runner=bound_path("runner"),
        input_bundle=bound_path("input_bundle"),
        runtime_source_lock=bound_path("runtime_source_lock"),
        runtime_lock=bound_path("runtime_lock"),
        patched_vllm_wheel=bound_path("patched_vllm_wheel"),
        patched_vllm_manifest=bound_path("patched_vllm_manifest"),
        pristine_flashinfer_wheel=bound_path("pristine_flashinfer_wheel"),
        patched_flashinfer_wheel=bound_path("patched_flashinfer_wheel"),
        patched_flashinfer_manifest=bound_path("patched_flashinfer_manifest"),
        runtime_closure_manifest=bound_path("runtime_closure_manifest"),
        python_executable=command_tools["unit_tests"],
        ruff_executable=command_tools["ruff"],
        mypy_executable=command_tools["mypy"],
    )


def _validate_sidecar_semantics(
    records: Mapping[str, Mapping[str, Any]],
    *,
    inputs: GPUQualificationLocalPreflightInputsV2,
    plan: Mapping[str, Any],
) -> None:
    _require_canonical_preflight_tool_paths(inputs)
    pins = pins_from_gpu_qualification_plan_v2(plan)
    _require_fixed_v2_pins(pins)
    python_checks = _python_check_specs(
        inputs,
        plan=plan,
        pins=pins,
        verify_source_rebuild=False,
    )
    empty_output = freeze_v1._command_output_binding(b"")  # noqa: SLF001
    python_identity = _python_api_identity_v2()
    for check_id, check in python_checks:
        record = records[check_id]
        expected_result, expected_bindings = check()
        if record.get("command") != [
            "python-api",
            f"document_kv_cache.publication_freeze_v2:{check_id}",
        ]:
            raise ValueError(f"v2 check {check_id!r} command differs")
        if record.get("inputs") != expected_bindings:
            raise ValueError(f"v2 check {check_id!r} inputs differ")
        if record.get("result") != expected_result:
            raise ValueError(f"v2 check {check_id!r} result differs")
        if record.get("tool_identity") != python_identity:
            raise ValueError(f"v2 check {check_id!r} tool identity differs")
        if record.get("environment") != freeze_v1._python_api_environment_identity():  # noqa: SLF001
            raise ValueError(f"v2 check {check_id!r} environment differs")
        if record.get("stdout") != empty_output or record.get("stderr") != empty_output:
            raise ValueError(f"v2 check {check_id!r} output differs")
    for check_id, command, version_command, expected_version in _command_check_specs(
        inputs
    ):
        record = records[check_id]
        executable = freeze_v1._resolve_executable(  # noqa: SLF001
            inputs.repository_root, Path(command[0])
        )
        normalized_command = [str(executable), *command[1:]]
        normalized_version = [str(executable), *version_command[1:]]
        expected_tool = {
            "executable_path": str(executable),
            "executable_sha256": freeze_v1._file_sha256(executable),  # noqa: SLF001
            "version": expected_version,
            "version_command": normalized_version,
        }
        if record.get("command") != normalized_command:
            raise ValueError(f"v2 check {check_id!r} command differs")
        if record.get("inputs") != [
            freeze_v1._repository_binding(inputs.repository_root)  # noqa: SLF001
        ]:
            raise ValueError(f"v2 check {check_id!r} inputs differ")
        if record.get("result") != _command_check_result(
            check_id,
            expected_version=expected_version,
            repository_root=inputs.repository_root,
        ):
            raise ValueError(f"v2 check {check_id!r} result differs")
        if record.get("tool_identity") != expected_tool:
            raise ValueError(f"v2 check {check_id!r} tool identity differs")
        if record.get("environment") != freeze_v1._preflight_environment(  # noqa: SLF001
            inputs.repository_root
        ):
            raise ValueError(f"v2 check {check_id!r} environment differs")
        freeze_v1._validate_command_output_binding(  # noqa: SLF001
            record.get("stdout"), f"v2 {check_id} stdout"
        )
        freeze_v1._validate_command_output_binding(  # noqa: SLF001
            record.get("stderr"), f"v2 {check_id} stderr"
        )
        if record.get("stderr") != empty_output:
            raise ValueError(f"v2 check {check_id!r} wrote stderr")


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
    if check_id not in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS:
        raise ValueError("unknown v2 local check_id")
    if status not in {"passed", "failed"} or ((status == "passed") != (exit_code == 0)):
        raise ValueError("v2 local check status and exit code differ")
    record: dict[str, Any] = {
        "check_id": check_id,
        "checked_at_utc": _timestamp(checked_at),
        "command": list(command),
        "environment": dict(environment),
        "exit_code": exit_code,
        "inputs": [dict(item) for item in inputs],
        "plan_sha256": plan_sha256,
        "record_type": GPU_QUALIFICATION_LOCAL_CHECK_V2_RECORD_TYPE,
        "result": dict(result),
        "schema_version": GPU_QUALIFICATION_LOCAL_CHECK_V2_SCHEMA_VERSION,
        "status": status,
        "stderr": freeze_v1._command_output_binding(stderr),  # noqa: SLF001
        "stdout": freeze_v1._command_output_binding(stdout),  # noqa: SLF001
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
    _require_exact_keys(record, _CHECK_KEYS, "v2 local check")
    if (
        record.get("record_type") != GPU_QUALIFICATION_LOCAL_CHECK_V2_RECORD_TYPE
        or type(record.get("schema_version")) is not int
        or record.get("schema_version") != GPU_QUALIFICATION_LOCAL_CHECK_V2_SCHEMA_VERSION
        or record.get("check_id") != expected_check_id
        or record.get("plan_sha256") != expected_plan_sha256
    ):
        raise ValueError("v2 local check identity differs")
    freeze_v1._parse_timestamp(record.get("checked_at_utc"), expected_check_id)  # noqa: SLF001
    command = record.get("command")
    if not isinstance(command, list) or not command or any(
        not isinstance(item, str) or not item for item in command
    ):
        raise ValueError("v2 local check command is invalid")
    exit_code = record.get("exit_code")
    if type(exit_code) is not int:
        raise ValueError("v2 local check exit_code must be an integer")
    status = record.get("status")
    if status not in {"passed", "failed"} or ((status == "passed") != (exit_code == 0)):
        raise ValueError("v2 local check status differs from exit_code")
    if not isinstance(record.get("inputs"), list):
        raise ValueError("v2 local check inputs must be an array")
    for field_name in ("environment", "result", "stdout", "stderr", "tool_identity"):
        if not isinstance(record.get(field_name), Mapping):
            raise ValueError(f"v2 local check {field_name} must be an object")
    freeze_v1._validate_command_output_binding(  # noqa: SLF001
        record.get("stdout"), "v2 local check stdout"
    )
    freeze_v1._validate_command_output_binding(  # noqa: SLF001
        record.get("stderr"), "v2 local check stderr"
    )
    if len(freeze_v1._canonical_json_bytes(record, pretty=False)) > (  # noqa: SLF001
        freeze_v1.PUBLICATION_FREEZE_MAX_CHECK_RECORD_BYTES
    ):
        raise ValueError("v2 local check exceeds its byte cap")


def _write_check(root: Path, record: Mapping[str, Any]) -> str:
    check_id = cast(str, record["check_id"])
    path = root / f"{check_id}.json"
    freeze_v1._write_exclusive(  # noqa: SLF001
        path,
        freeze_v1._canonical_json_bytes(record, pretty=False),  # noqa: SLF001
        f"v2 {check_id}",
    )
    reread = freeze_v1._read_canonical_json(  # noqa: SLF001
        path, pretty=False, label=f"v2 {check_id}"
    )
    if reread != dict(record):
        raise RuntimeError("v2 local check changed during durable write")
    return freeze_v1._file_sha256(path)  # noqa: SLF001


def _runtime_artifact_paths(
    inputs: GPUQualificationLocalPreflightInputsV2,
) -> dict[str, Path]:
    return {
        label: freeze_v1._regular_file(path, label)  # noqa: SLF001
        for label, path in (
            ("runtime_source_lock", inputs.runtime_source_lock),
            ("runtime_lock", inputs.runtime_lock),
            ("patched_vllm_wheel", inputs.patched_vllm_wheel),
            ("patched_vllm_manifest", inputs.patched_vllm_manifest),
            ("pristine_flashinfer_wheel", inputs.pristine_flashinfer_wheel),
            ("patched_flashinfer_wheel", inputs.patched_flashinfer_wheel),
            ("patched_flashinfer_manifest", inputs.patched_flashinfer_manifest),
            ("runtime_closure_manifest", inputs.runtime_closure_manifest),
        )
    }


def _require_runtime_artifact_file_pins(
    paths: Mapping[str, Path],
    *,
    pins: GPUQualificationArtifactPinsV2,
) -> None:
    expected = {
        "runtime_source_lock": (VLLM_RUNTIME_LOCK_SHA256, VLLM_RUNTIME_SOURCE_LOCK_SIZE),
        "runtime_lock": (pins.runtime_lock_sha256, VLLM_RUNTIME_BASE_LOCK_SIZE),
        "patched_vllm_wheel": (pins.patched_vllm_wheel_sha256, VLLM_PATCHED_WHEEL_SIZE),
        "patched_vllm_manifest": (VLLM_PATCHED_MANIFEST_SHA256, VLLM_PATCHED_MANIFEST_SIZE),
        "pristine_flashinfer_wheel": (FLASHINFER_SOURCE_WHEEL_SHA256, FLASHINFER_SOURCE_WHEEL_SIZE),
        "patched_flashinfer_wheel": (pins.patched_flashinfer_wheel_sha256, FLASHINFER_PATCHED_WHEEL_SIZE),
        "patched_flashinfer_manifest": (FLASHINFER_PATCHED_MANIFEST_FILE_SHA256, FLASHINFER_PATCHED_MANIFEST_SIZE),
        "runtime_closure_manifest": (pins.runtime_closure_manifest_sha256, RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE),
    }
    if tuple(paths) != tuple(expected):
        raise ValueError("v2 runtime artifact path coverage differs")
    for label, (expected_sha256, expected_size) in expected.items():
        path = paths[label]
        if (
            freeze_v1._file_sha256(path) != expected_sha256  # noqa: SLF001
            or path.stat().st_size != expected_size
        ):
            raise ValueError(f"v2 runtime artifact {label!r} differs")


def _command_check_specs(
    inputs: GPUQualificationLocalPreflightInputsV2,
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
                *_V2_STATIC_ANALYSIS_TARGETS,
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
                *_V2_STATIC_ANALYSIS_TARGETS,
            ),
            (str(inputs.mypy_executable), "--version"),
            "mypy 2.2.0 (compiled: yes)",
        ),
    )


def _command_check_result(
    check_id: str,
    *,
    expected_version: str,
    repository_root: Path,
) -> dict[str, Any]:
    if check_id == "unit_tests":
        scope = "repository_test_suite"
        targets = ["tests"]
    elif check_id in {"ruff", "mypy"}:
        scope = "frozen_publication_v2_surface"
        targets = list(_V2_STATIC_ANALYSIS_TARGETS)
    else:
        raise ValueError("unknown v2 command check")
    return {
        "excluded_known_dirty_paths": list(
            freeze_v1._STATIC_ANALYSIS_EXCLUDED_TARGETS  # noqa: SLF001
            if check_id in {"ruff", "mypy"}
            else ()
        ),
        "expected_tool_version": expected_version,
        "scope": scope,
        "target_paths": targets,
        "working_directory": str(repository_root.resolve()),
    }


def _require_canonical_preflight_tool_paths(
    inputs: GPUQualificationLocalPreflightInputsV2,
) -> None:
    observed_runtime = (
        f"{__import__('platform').python_implementation()} "
        f"{__import__('platform').python_version()}"
    )
    if observed_runtime != freeze_v1.PUBLICATION_FREEZE_PYTHON:
        raise RuntimeError(
            f"v2 preflight requires {freeze_v1.PUBLICATION_FREEZE_PYTHON}, "
            f"found {observed_runtime}"
        )
    expected_python = freeze_v1._DEFAULT_PYTHON_EXECUTABLE.resolve()  # noqa: SLF001
    if Path(__import__('sys').executable).resolve() != expected_python:
        raise RuntimeError("v2 preflight must run under the frozen Python executable")
    expected = {
        "python": expected_python,
        "ruff": (inputs.repository_root / freeze_v1._DEFAULT_RUFF_EXECUTABLE).resolve(),  # noqa: SLF001
        "mypy": (inputs.repository_root / freeze_v1._DEFAULT_MYPY_EXECUTABLE).resolve(),  # noqa: SLF001
    }
    observed = {
        "python": inputs.python_executable.resolve(),
        "ruff": inputs.ruff_executable.resolve(),
        "mypy": inputs.mypy_executable.resolve(),
    }
    if observed != expected:
        raise ValueError("v2 preflight tool paths differ from the freeze")


def _python_api_identity_v2() -> dict[str, Any]:
    identity = freeze_v1._python_api_identity()  # noqa: SLF001
    identity["module_path"] = str(Path(__file__).resolve())
    identity["module_sha256"] = freeze_v1._file_sha256(  # noqa: SLF001
        Path(__file__).resolve()
    )
    v1_path = Path(freeze_v1.__file__).resolve()
    identity["protocol_helper_module_path"] = str(v1_path)
    identity["protocol_helper_module_sha256"] = freeze_v1._file_sha256(v1_path)  # noqa: SLF001
    return identity


def _canonical_submit_payload_closure_bytes(
    submit_payloads: Sequence[Mapping[str, Any]],
) -> bytes:
    if isinstance(submit_payloads, (str, bytes, bytearray)) or not isinstance(
        submit_payloads, Sequence
    ):
        raise TypeError("v2 submit_payloads must be a sequence")
    if len(submit_payloads) != 14:
        raise ValueError("v2 submit_payloads must contain exactly fourteen jobs")
    normalized = [
        _mapping_copy(payload, f"v2 submit payload {index}")
        for index, payload in enumerate(submit_payloads)
    ]
    try:
        return freeze_v1._canonical_json_bytes(normalized, pretty=False)  # noqa: SLF001
    except (TypeError, ValueError) as exc:
        raise ValueError("v2 submit_payloads are not canonical JSON values") from exc


def _require_submit_payload_closure_binding(
    inputs: GPUQualificationLocalPreflightInputsV2,
    *,
    submitted_payload_bytes: bytes,
) -> None:
    sidecar_payloads = freeze_v1._read_canonical_json_value(  # noqa: SLF001
        inputs.submit_payloads_json,
        label="v2 preflight submit payloads",
    )
    if not isinstance(sidecar_payloads, list) or len(sidecar_payloads) != 14:
        raise ValueError("v2 preflight submit payloads lack exact coverage")
    if freeze_v1._canonical_json_bytes(  # noqa: SLF001
        sidecar_payloads, pretty=False
    ) != submitted_payload_bytes:
        raise ValueError("v2 submitted payload closure differs from preflight")


def _canonical_preflight_evidence_path(path: Path) -> Path:
    raw = str(path)
    if not path.is_absolute() or path != Path(os.path.normpath(raw)):
        raise ValueError("v2 preflight evidence path must be normalized absolute")
    candidate = freeze_v1._regular_file(  # noqa: SLF001
        path, "v2 local preflight evidence"
    )
    if candidate.name != "local-preflight-evidence.json":
        raise ValueError("v2 local preflight evidence filename differs")
    freeze_v1._regular_directory(candidate.parent, "v2 preflight bundle root")  # noqa: SLF001
    return candidate


def _preflight_bundle_file_hashes(evidence_path: Path) -> dict[str, str]:
    root = freeze_v1._regular_directory(  # noqa: SLF001
        evidence_path.parent, "v2 preflight bundle root"
    )
    expected_names = {
        *(f"{check_id}.json" for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS),
        "local-preflight-evidence.json",
    }
    if {item.name for item in root.iterdir()} != expected_names:
        raise ValueError("v2 preflight directory lacks exact check coverage")
    return {
        name: freeze_v1._file_sha256(  # noqa: SLF001
            freeze_v1._regular_file(root / name, f"v2 preflight {name}")  # noqa: SLF001
        )
        for name in sorted(expected_names)
    }


def _validate_live_workspace_and_remote_artifacts_v2(
    config: DatabricksWorkspaceConfig,
    *,
    inputs: GPUQualificationLocalPreflightInputsV2,
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
        raise ValueError("v2 qualification launch requires zero active runs")
    node_types = list_databricks_node_types(config, max_node_types=1_024)
    observed_nodes = {
        item.get("node_type_id") for item in node_types if isinstance(item, Mapping)
    }
    if not {"g5.8xlarge", "g6.8xlarge", "g6e.4xlarge"}.issubset(
        observed_nodes
    ):
        raise ValueError("v2 workspace lacks qualification node types")
    uris_record = freeze_v1._read_canonical_json(  # noqa: SLF001
        inputs.artifact_uris_json,
        pretty=False,
        label="v2 artifact URI record",
    )
    uris = _required_mapping(uris_record, "artifact_uris")
    if tuple(uris) != GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        raise ValueError("v2 remote URI role coverage differs")
    output_root = _required_string(uris_record, "output_root")
    output_parent = freeze_v1._parent_volume_directory_uri(output_root)  # noqa: SLF001
    output_entries = list_databricks_volume_directory(config, output_parent)
    if require_fresh_workspace and any(
        item.get("name") == output_root.rsplit("/", 1)[-1]
        for item in output_entries
    ):
        raise ValueError("v2 qualification output root already exists")
    pins = pins_from_gpu_qualification_plan_v2(plan)
    _require_fixed_v2_pins(pins)
    file_hashes = {
        key: getattr(pins, key)
        for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS
        if key != "input_bundle_sha256"
    }
    for role, expected_sha256 in file_hashes.items():
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
            raise ValueError(f"v2 direct UC readback differs for {role!r}")
    source_uri = _required_string(uris, "cachet_source_tree_sha256")
    freeze_v1._require_remote_tree_matches_local(  # noqa: SLF001
        config,
        remote_root_uri=freeze_v1._parent_volume_uri(source_uri),  # noqa: SLF001
        local_root=inputs.source_artifact_root,
        label="v2 source artifact tree",
        stream_file=stream_databricks_volume_file_sha256,
    )
    freeze_v1._require_remote_tree_matches_local(  # noqa: SLF001
        config,
        remote_root_uri=_required_string(uris, "input_bundle_sha256"),
        local_root=inputs.input_bundle,
        label="v2 input bundle",
        stream_file=stream_databricks_volume_file_sha256,
    )


def _require_fixed_v2_pins(pins: GPUQualificationArtifactPinsV2) -> None:
    expected = {
        "input_bundle_sha256": GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
        "patched_flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
        "patched_vllm_wheel_sha256": VLLM_PATCHED_WHEEL_SHA256,
        "runner_sha256": GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256,
        "runtime_closure_manifest_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "runtime_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
    }
    for field_name, expected_sha256 in expected.items():
        if getattr(pins, field_name) != expected_sha256:
            raise ValueError(f"v2 plan {field_name} differs from publication")


def _run_command(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
) -> CompletedCommand:
    return freeze_v1._run_command(command, cwd, environment)  # noqa: SLF001


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    return freeze_v1._timestamp(value)  # noqa: SLF001


def _required_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _required_string(value: Mapping[str, Any], field_name: str) -> str:
    raw = value.get(field_name)
    if not isinstance(raw, str) or not raw or raw.strip() != raw:
        raise ValueError(f"{field_name} must be a canonical nonempty string")
    return raw


__all__ = [
    "GPU_QUALIFICATION_LOCAL_CHECK_V2_RECORD_TYPE",
    "GPU_QUALIFICATION_LOCAL_CHECK_V2_SCHEMA_VERSION",
    "GPUQualificationLocalPreflightInputsV2",
    "PUBLICATION_SOURCE_CLOSURE_V2_RECORD_TYPE",
    "PUBLICATION_SOURCE_CLOSURE_V2_SCHEMA_VERSION",
    "PublicationSourceClosureInputsV2",
    "build_publication_source_closure_v2",
    "run_gpu_qualification_local_preflight_v2",
    "validate_gpu_qualification_local_preflight_bundle_v2",
    "validate_publication_source_closure_v2_record",
    "write_publication_source_closure_v2_json",
]


def _source_reference_inputs(
    inputs: PublicationSourceClosureInputsV2,
) -> tuple[tuple[Path, str], ...]:
    return (
        (inputs.runtime_source_lock, "runtime_source_lock"),
        (inputs.runtime_lock, "runtime_lock"),
        (inputs.runtime_lock_input, "runtime_lock_input"),
        (inputs.campaign_plan, "campaign_plan"),
        (inputs.latency_handoff_plan, "latency_handoff_plan"),
        (inputs.full_score_inventory, "full_score_inventory"),
        (inputs.full_score_shard_plan, "full_score_shard_plan"),
        (inputs.patched_vllm_wheel, "patched_vllm_wheel"),
        (inputs.patched_vllm_manifest, "patched_vllm_manifest"),
        (inputs.pristine_flashinfer_wheel, "pristine_flashinfer_wheel"),
        (inputs.patched_flashinfer_wheel, "patched_flashinfer_wheel"),
        (inputs.patched_flashinfer_manifest, "patched_flashinfer_manifest"),
        (inputs.runtime_closure_manifest, "runtime_closure_manifest"),
    )


def _validate_source_references(
    inputs: PublicationSourceClosureInputsV2,
    *,
    root: Path,
) -> None:
    paths = {
        role: freeze_v1._regular_file(path, role)  # noqa: SLF001
        for path, role in _source_reference_inputs(inputs)
    }
    _validate_reference_paths(paths, repository_root=root)


def _validate_reference_paths(
    paths: Mapping[str, Path],
    *,
    repository_root: Path,
) -> None:
    if tuple(paths) != _SOURCE_REFERENCE_ROLES:
        raise ValueError("v2 source reference coverage differs")
    if freeze_v1._file_sha256(paths["runtime_source_lock"]) != (  # noqa: SLF001
        VLLM_RUNTIME_LOCK_SHA256
    ):
        raise ValueError("v2 runtime source lock differs")
    if freeze_v1._file_sha256(paths["runtime_lock_input"]) != (  # noqa: SLF001
        freeze_v1.PUBLICATION_FREEZE_RUNTIME_LOCK_INPUT_SHA256
    ):
        raise ValueError("v2 runtime lock input differs")
    validate_publication_campaign_plan_record(
        freeze_v1._read_canonical_json(  # noqa: SLF001
            paths["campaign_plan"], pretty=True, label="campaign plan"
        )
    )
    freeze_v1._validate_publication_latency_handoff_reference(  # noqa: SLF001
        paths["latency_handoff_plan"],
        repository_root=repository_root,
    )
    validate_vllm_flashinfer_runtime_artifact_closure(
        source_lock=paths["runtime_source_lock"],
        base_lock=paths["runtime_lock"],
        vllm_wheel=paths["patched_vllm_wheel"],
        vllm_manifest=paths["patched_vllm_manifest"],
        pristine_flashinfer_wheel=paths["pristine_flashinfer_wheel"],
        patched_flashinfer_wheel=paths["patched_flashinfer_wheel"],
        flashinfer_manifest=paths["patched_flashinfer_manifest"],
        closure_manifest=paths["runtime_closure_manifest"],
    )


def _v2_runtime_identity() -> dict[str, Any]:
    return {
        "base_lock": {
            "byte_count": VLLM_RUNTIME_BASE_LOCK_SIZE,
            "distribution_count": VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
            "hash_count": VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
            "sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        },
        "flashinfer": {
            "manifest_closed_record_sha256": (
                FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
            ),
            "manifest_file_byte_count": FLASHINFER_PATCHED_MANIFEST_SIZE,
            "manifest_file_sha256": FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
            "patched_wheel_byte_count": FLASHINFER_PATCHED_WHEEL_SIZE,
            "patched_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
            "pristine_wheel_byte_count": FLASHINFER_SOURCE_WHEEL_SIZE,
            "pristine_wheel_sha256": FLASHINFER_SOURCE_WHEEL_SHA256,
        },
        "input_bundle_sha256": GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
        "runtime_closure": {
            "closed_record_sha256": RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
            "file_byte_count": RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE,
            "file_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        },
        "source_lock": {
            "byte_count": VLLM_RUNTIME_SOURCE_LOCK_SIZE,
            "distribution_count": VLLM_RUNTIME_SOURCE_LOCK_DISTRIBUTION_COUNT,
            "hash_count": VLLM_RUNTIME_SOURCE_LOCK_HASH_COUNT,
            "sha256": VLLM_RUNTIME_LOCK_SHA256,
        },
        "vllm": {
            "manifest_file_byte_count": VLLM_PATCHED_MANIFEST_SIZE,
            "manifest_file_sha256": VLLM_PATCHED_MANIFEST_SHA256,
            "wheel_byte_count": VLLM_PATCHED_WHEEL_SIZE,
            "wheel_sha256": VLLM_PATCHED_WHEEL_SHA256,
        },
    }


def _require_v2_package_source_equality(
    *,
    repository_root: Path,
    package_wheel: Path,
    source_distribution: Path,
    git_source_archive: Path,
) -> dict[str, Any]:
    workspace: dict[str, bytes] = {}
    for package_root in _PACKAGE_ROOTS:
        root = freeze_v1._regular_directory(  # noqa: SLF001
            repository_root / "src" / package_root,
            f"v2 package root {package_root}",
        )
        for path in freeze_v1._regular_tree(root):  # noqa: SLF001
            relative = path.relative_to(repository_root / "src").as_posix()
            if relative in workspace:
                raise ValueError("v2 workspace package payload repeats a path")
            workspace[relative] = path.read_bytes()
    with zipfile.ZipFile(package_wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("v2 package wheel repeats a member")
        wheel = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = info.filename
            if relative.split("/", 1)[0] not in _PACKAGE_ROOTS:
                continue
            if relative in wheel:
                raise ValueError("v2 package wheel repeats a payload path")
            wheel[relative] = archive.read(info)
    with tarfile.open(source_distribution, mode="r:gz") as archive:
        sdist: dict[str, bytes] = {}
        for member in archive.getmembers():
            parts = member.name.split("/")
            if len(parts) < 4 or parts[1] != "src" or parts[2] not in _PACKAGE_ROOTS:
                continue
            if not member.isfile():
                if member.isdir():
                    continue
                raise ValueError("v2 sdist package payload has an irregular member")
            relative = "/".join(parts[2:])
            if relative in sdist:
                raise ValueError("v2 sdist repeats a package payload path")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"v2 sdist member {relative!r} is unreadable")
            sdist[relative] = stream.read()
    with tarfile.open(git_source_archive, mode="r:gz") as archive:
        git: dict[str, bytes] = {}
        for member in archive.getmembers():
            parts = member.name.split("/")
            source_index = 0 if parts and parts[0] == "src" else 1
            if (
                len(parts) <= source_index + 2
                or parts[source_index] != "src"
                or parts[source_index + 1] not in _PACKAGE_ROOTS
            ):
                continue
            if not member.isfile():
                if member.isdir():
                    continue
                raise ValueError("v2 Git package payload has an irregular member")
            relative = "/".join(parts[source_index + 1 :])
            if relative in git:
                raise ValueError("v2 Git archive repeats a package payload path")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"v2 Git archive member {relative!r} is unreadable")
            git[relative] = stream.read()
    if not (set(workspace) == set(wheel) == set(sdist) == set(git)):
        raise ValueError("v2 package payload path coverage differs")
    for relative in sorted(workspace):
        if not (workspace[relative] == wheel[relative] == sdist[relative] == git[relative]):
            raise ValueError(f"v2 package source bytes differ for {relative!r}")
    rows = [
        [relative, sha256(workspace[relative]).hexdigest(), len(workspace[relative])]
        for relative in sorted(workspace)
    ]
    return {
        "file_count": len(rows),
        "tree_sha256": sha256(
            canonical_gpu_qualification_json({"files": rows}).encode("utf-8")
        ).hexdigest(),
    }


def _mapping_copy(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _required_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    raw = value.get(field_name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return raw


def _mapping_sequence(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{label} must be an array of objects")
    return tuple(cast(list[Mapping[str, Any]], value))


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{label} does not use the closed schema")
