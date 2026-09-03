from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any, Callable, Mapping
import zipfile

import pytest

import document_kv_cache.gpu_qualification as qualification_v1
import document_kv_cache.gpu_qualification_v2 as qualification_v2
import document_kv_cache.publication_inputs as full_score_inputs
import document_kv_cache.publication_freeze as freeze_v1
import document_kv_cache.publication_freeze_v2 as freeze_v2
from document_kv_cache.benchmarks import SUPPORTED_V1_DATASETS
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PATCHED_WHEEL_SHA256,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    canonical_gpu_qualification_json,
)
from document_kv_cache.gpu_qualification_databricks_v2 import (
    GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT,
    GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_ARTIFACT_KEYS,
    GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS,
    GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
    GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS,
    GPUQualificationArtifactPinsV2,
    build_gpu_qualification_plan_v2,
    build_local_preflight_evidence_v2,
    validate_local_preflight_evidence_v2_record,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_ID,
    PUBLICATION_CAMPAIGN_LEDGER_ID,
    PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
)
from document_kv_cache.publication_inputs import (
    build_full_score_shard_plan,
    full_score_inventory_to_record,
    load_full_score_inventory,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    VLLM_PATCHED_WHEEL_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _pins() -> GPUQualificationArtifactPinsV2:
    return GPUQualificationArtifactPinsV2(
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_sha256=VLLM_PATCHED_WHEEL_SHA256,
        patched_flashinfer_wheel_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
        runtime_closure_manifest_sha256=RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        package_wheel_sha256=_digest("v2-package-wheel"),
        cachet_source_tree_sha256=_digest("v2-source-closure"),
        runner_sha256=GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256,
        input_bundle_sha256=GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    )


def _plan() -> dict[str, Any]:
    return build_gpu_qualification_plan_v2(
        campaign_id=PUBLICATION_CAMPAIGN_ID,
        campaign_record_sha256=PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
        campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS
        ),
        artifact_pins=_pins(),
    )


def _check_hashes() -> dict[str, str]:
    return {
        check_id: _digest(f"v2-check:{check_id}")
        for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS
    }


def test_v2_local_evidence_has_exact_ordered_eight_check_seals() -> None:
    plan = _plan()
    completed = "2026-08-25T12:00:00Z"
    evidence = build_local_preflight_evidence_v2(
        plan_sha256=plan["closed_record_sha256"],
        completed_at_utc=completed,
        check_evidence_sha256=_check_hashes(),
    )

    assert evidence["record_type"] == (
        "cachet.vllm_0271_local_preflight_evidence.v2"
    )
    assert evidence["schema_version"] == 2
    assert evidence["scope"] == (
        "local_preflight_only_no_cloud_success_credit_v2"
    )
    assert [item["check_id"] for item in evidence["checks"]] == list(
        GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS
    )
    assert [item["evidence_sha256"] for item in evidence["checks"]] == list(
        _check_hashes().values()
    )
    assert validate_local_preflight_evidence_v2_record(
        evidence,
        plan_sha256=plan["closed_record_sha256"],
    ) == datetime(2026, 8, 25, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered"])
def test_v2_local_evidence_builder_rejects_noncanonical_coverage(
    mutation: str,
) -> None:
    hashes = _check_hashes()
    if mutation == "missing":
        hashes.pop("runtime_artifact_closure")
    elif mutation == "extra":
        hashes["unexpected"] = _digest("unexpected")
    else:
        hashes = dict(reversed(tuple(hashes.items())))

    with pytest.raises(ValueError, match="canonical eight-check coverage"):
        build_local_preflight_evidence_v2(
            plan_sha256=_digest("plan"),
            completed_at_utc="2026-08-25T12:00:00Z",
            check_evidence_sha256=hashes,
        )


def test_v2_local_evidence_rejects_tampered_or_resealed_child_order() -> None:
    evidence = build_local_preflight_evidence_v2(
        plan_sha256=_digest("plan"),
        completed_at_utc="2026-08-25T12:00:00Z",
        check_evidence_sha256=_check_hashes(),
    )
    tampered = deepcopy(evidence)
    tampered["checks"][3]["evidence_sha256"] = _digest("forged")
    with pytest.raises(ValueError, match="closed_record_sha256 differs"):
        validate_local_preflight_evidence_v2_record(
            tampered,
            plan_sha256=_digest("plan"),
        )

    resealed = deepcopy(evidence)
    resealed["checks"][2], resealed["checks"][3] = (
        resealed["checks"][3],
        resealed["checks"][2],
    )
    resealed["closed_record_sha256"] = qualification_v2._closed_record_sha256(
        resealed
    )
    with pytest.raises(ValueError, match="not canonical"):
        validate_local_preflight_evidence_v2_record(
            resealed,
            plan_sha256=_digest("plan"),
        )


def test_local_evidence_versions_cross_reject() -> None:
    v1_hashes = {
        check_id: _digest(f"v1:{check_id}")
        for check_id in qualification_v1._LOCAL_CHECK_IDS  # noqa: SLF001
    }
    v1_evidence = qualification_v1.build_local_preflight_evidence(
        plan_sha256=_digest("plan"),
        completed_at_utc="2026-08-25T12:00:00Z",
        check_evidence_sha256=v1_hashes,
    )
    with pytest.raises(ValueError, match="closed_record_sha256|record_type"):
        validate_local_preflight_evidence_v2_record(
            v1_evidence,
            plan_sha256=_digest("plan"),
        )

    v2_evidence = build_local_preflight_evidence_v2(
        plan_sha256=_digest("plan"),
        completed_at_utc="2026-08-25T12:00:00Z",
        check_evidence_sha256=_check_hashes(),
    )
    with pytest.raises(ValueError, match="closed_record_sha256|record_type"):
        qualification_v1.validate_local_preflight_evidence_record(
            v2_evidence,
            plan_sha256=_digest("plan"),
        )


_PACKAGE_PAYLOAD = {
    "cachet/__init__.py": b"CACHE = True\n",
    "document_kv_cache/runtime_locks/base.lock": b"locked==1 --hash=x\n",
    "sglang_kv_injection/__init__.py": b"SGLANG = True\n",
    "vllm_kv_injection/__init__.py": b"VLLM = True\n",
}


def _write_tar(path: Path, rows: Mapping[str, bytes]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, content in rows.items():
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            info.mtime = 1_777_777_777
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _write_wheel(path: Path, rows: Mapping[str, bytes]) -> None:
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, content in rows.items():
            archive.writestr(name, content)


def _package_representations(
    tmp_path: Path,
    *,
    wheel_rows: Mapping[str, bytes] | None = None,
) -> tuple[Path, Path, Path, Path]:
    repository = tmp_path / "repository"
    for package_root in freeze_v2._PACKAGE_ROOTS:
        (repository / "src" / package_root).mkdir(parents=True)
    for relative, content in _PACKAGE_PAYLOAD.items():
        target = repository / "src" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    wheel = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
    _write_wheel(wheel, wheel_rows or _PACKAGE_PAYLOAD)
    sdist = tmp_path / "cachet_kv-0.2.0.tar.gz"
    _write_tar(
        sdist,
        {
            f"cachet_kv-0.2.0/src/{relative}": content
            for relative, content in _PACKAGE_PAYLOAD.items()
        },
    )
    git_archive = tmp_path / f"cachet-{'1' * 40}.tar.gz"
    _write_tar(
        git_archive,
        {
            f"cachet-{'1' * 40}/src/{relative}": content
            for relative, content in _PACKAGE_PAYLOAD.items()
        },
    )
    return repository, wheel, sdist, git_archive


def test_package_payload_requires_exact_workspace_git_sdist_wheel_equality(
    tmp_path: Path,
) -> None:
    repository, wheel, sdist, git_archive = _package_representations(tmp_path)

    observed = freeze_v2._require_v2_package_source_equality(
        repository_root=repository,
        package_wheel=wheel,
        source_distribution=sdist,
        git_source_archive=git_archive,
    )
    rows = [
        [relative, hashlib.sha256(content).hexdigest(), len(content)]
        for relative, content in sorted(_PACKAGE_PAYLOAD.items())
    ]
    assert observed == {
        "file_count": 4,
        "tree_sha256": hashlib.sha256(
            canonical_gpu_qualification_json({"files": rows}).encode("utf-8")
        ).hexdigest(),
    }


@pytest.mark.parametrize("failure", ["missing", "changed"])
def test_package_payload_rejects_missing_or_changed_wheel_member(
    tmp_path: Path,
    failure: str,
) -> None:
    wheel_rows = dict(_PACKAGE_PAYLOAD)
    if failure == "missing":
        wheel_rows.pop("cachet/__init__.py")
        expected = "path coverage differs"
    else:
        wheel_rows["cachet/__init__.py"] = b"CACHE = False\n"
        expected = "source bytes differ"
    repository, wheel, sdist, git_archive = _package_representations(
        tmp_path,
        wheel_rows=wheel_rows,
    )

    with pytest.raises(ValueError, match=expected):
        freeze_v2._require_v2_package_source_equality(
            repository_root=repository,
            package_wheel=wheel,
            source_distribution=sdist,
            git_source_archive=git_archive,
        )


def _source_record_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Path, Path]:
    repository = tmp_path / "repository"
    artifact_root = tmp_path / "artifacts"
    repository.mkdir()
    artifact_root.mkdir()
    identity = {
        "branch": "codex/v2-source-test",
        "commit": "1" * 40,
        "commit_tree": "2" * 40,
        "source_date_epoch": 1_777_777_777,
    }
    monkeypatch.setattr(freeze_v1, "_git_identity", lambda _root: identity)
    monkeypatch.setattr(freeze_v1, "_require_freeze_toolchain", lambda: None)
    monkeypatch.setattr(
        freeze_v1, "_require_freeze_build_system", lambda _root: None
    )
    monkeypatch.setattr(freeze_v1, "_validate_cachet_wheel", lambda _path: None)
    monkeypatch.setattr(freeze_v1, "_validate_sdist", lambda _path: None)
    monkeypatch.setattr(
        freeze_v1, "_validate_git_archive", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        freeze_v2,
        "_require_v2_package_source_equality",
        lambda **_kwargs: {"file_count": 246, "tree_sha256": _digest("tree")},
    )
    monkeypatch.setattr(
        freeze_v2, "_validate_reference_paths", lambda *_args, **_kwargs: None
    )

    artifact_content = {
        "cachet_kv-0.2.0-py3-none-any.whl": b"wheel",
        "cachet_kv-0.2.0.tar.gz": b"sdist",
        f"cachet-{'1' * 40}.tar.gz": b"git archive",
        "gpu-qualification-bootstrap-v2.py": (
            GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT.encode("utf-8")
        ),
    }
    for name, content in artifact_content.items():
        (artifact_root / name).write_bytes(content)
    files = [
        freeze_v1._source_file_record(artifact_root / name, role)  # noqa: SLF001
        for name, role in zip(
            artifact_content,
            freeze_v2._SOURCE_FILE_ROLES,
            strict=True,
        )
    ]
    references = []
    for role in freeze_v2._SOURCE_REFERENCE_ROLES:
        path = repository / "references" / role
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{role}\n".encode("utf-8"))
        references.append(
            freeze_v1._source_reference_record(repository, path, role)  # noqa: SLF001
        )
    record: dict[str, Any] = {
        "build": {
            "build_backend": freeze_v1.PUBLICATION_FREEZE_BUILD_BACKEND,
            "build_frontend": freeze_v1.PUBLICATION_FREEZE_BUILD_FRONTEND,
            "python": freeze_v1.PUBLICATION_FREEZE_PYTHON,
            "source_date_epoch": identity["source_date_epoch"],
        },
        "campaign_id": PUBLICATION_CAMPAIGN_ID,
        "closed_record_sha256": "",
        "files": files,
        "git": {
            "branch": identity["branch"],
            "commit": identity["commit"],
            "commit_tree": identity["commit_tree"],
            "dirty": False,
        },
        "package_payload_closure": {
            "file_count": 246,
            "tree_sha256": _digest("tree"),
        },
        "record_type": freeze_v2.PUBLICATION_SOURCE_CLOSURE_V2_RECORD_TYPE,
        "references": references,
        "runtime": freeze_v2._v2_runtime_identity(),
        "schema_version": freeze_v2.PUBLICATION_SOURCE_CLOSURE_V2_SCHEMA_VERSION,
    }
    record["closed_record_sha256"] = freeze_v1._closed_record_sha256(  # noqa: SLF001
        record
    )
    return record, repository, artifact_root


def test_source_closure_v2_validates_seal_and_rejects_v1_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, repository, artifact_root = _source_record_fixture(
        tmp_path, monkeypatch
    )
    freeze_v2._validate_publication_source_closure_v2_record(
        record,
        repository_root=repository,
        artifact_root=artifact_root,
        explicit_artifact_paths=None,
        verify_rebuild=False,
    )

    tampered = deepcopy(record)
    tampered["package_payload_closure"]["file_count"] = 245
    with pytest.raises(
        ValueError,
        match="closed_record_sha256 (?:is invalid|differs)",
    ):
        freeze_v2._validate_publication_source_closure_v2_record(
            tampered,
            repository_root=repository,
            artifact_root=artifact_root,
            explicit_artifact_paths=None,
            verify_rebuild=False,
        )

    v1_identity = deepcopy(record)
    v1_identity["record_type"] = freeze_v1.PUBLICATION_SOURCE_CLOSURE_RECORD_TYPE
    v1_identity["closed_record_sha256"] = freeze_v1._closed_record_sha256(  # noqa: SLF001
        v1_identity
    )
    with pytest.raises(ValueError, match="record_type differs"):
        freeze_v2._validate_publication_source_closure_v2_record(
            v1_identity,
            repository_root=repository,
            artifact_root=artifact_root,
            explicit_artifact_paths=None,
            verify_rebuild=False,
        )

    public_calls = []

    def reject_static_public_validation(
        *_args: object,
        verify_rebuild: bool,
        latency_semantic_mode: freeze_v2._LatencySemanticValidationMode,
        **_kwargs: object,
    ) -> None:
        public_calls.append((latency_semantic_mode, verify_rebuild))
        if latency_semantic_mode is freeze_v2._LatencySemanticValidationMode.STATIC:
            raise ValueError("source reference changed during latency validation")

    monkeypatch.setattr(
        freeze_v2,
        "_validate_publication_source_closure_v2_record",
        reject_static_public_validation,
    )
    with pytest.raises(ValueError, match="source reference changed"):
        freeze_v2.validate_publication_source_closure_v2_record(
            record,
            repository_root=repository,
            artifact_root=artifact_root,
        )
    assert public_calls == [
        (freeze_v2._LatencySemanticValidationMode.EXECUTE, True),
        (freeze_v2._LatencySemanticValidationMode.STATIC, False),
    ]

    public_calls.clear()
    output = tmp_path / "must-not-write-source-closure.json"
    with pytest.raises(ValueError, match="source reference changed"):
        freeze_v2.write_publication_source_closure_v2_json(
            record,
            output,
            repository_root=repository,
            artifact_root=artifact_root,
        )
    assert public_calls == [
        (freeze_v2._LatencySemanticValidationMode.EXECUTE, True),
        (freeze_v2._LatencySemanticValidationMode.STATIC, False),
    ]
    assert not output.exists()


def test_source_closure_without_rebuild_never_enters_pypi_build_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, repository, artifact_root = _source_record_fixture(
        tmp_path, monkeypatch
    )
    build_calls: list[str] = []

    def forbidden_build(*_args: object, **_kwargs: object) -> None:
        build_calls.append("build_package_twice")

    def forbidden_pypi_environment(*_args: object, **_kwargs: object) -> None:
        build_calls.append("pypi_build_environment")

    monkeypatch.setattr(freeze_v1, "_build_package_twice", forbidden_build)
    monkeypatch.setattr(freeze_v1, "_build_environment", forbidden_pypi_environment)

    freeze_v2._validate_publication_source_closure_v2_record(
        record,
        repository_root=repository,
        artifact_root=artifact_root,
        explicit_artifact_paths=None,
        verify_rebuild=False,
        latency_semantic_mode=freeze_v2._LatencySemanticValidationMode.STATIC,
    )

    assert build_calls == []


def test_source_builder_reference_failure_leaves_no_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "must-not-exist"
    reference_paths = {}
    for field_name in freeze_v2.PublicationSourceClosureInputsV2.__dataclass_fields__:
        if field_name in {"repository_root", "artifact_output_root"}:
            continue
        path = repository / "references" / field_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{field_name}\n", encoding="utf-8")
        reference_paths[field_name] = path
    inputs = freeze_v2.PublicationSourceClosureInputsV2(
        repository_root=repository,
        artifact_output_root=output,
        **reference_paths,
    )
    monkeypatch.setattr(
        freeze_v1,
        "_git_identity",
        lambda _root: {
            "branch": "codex/test",
            "commit": "1" * 40,
            "commit_tree": "2" * 40,
            "source_date_epoch": 1_777_777_777,
        },
    )
    monkeypatch.setattr(freeze_v1, "_require_freeze_toolchain", lambda: None)
    monkeypatch.setattr(
        freeze_v1, "_require_freeze_build_system", lambda _root: None
    )

    original_file_sha256 = freeze_v1._file_sha256  # noqa: SLF001

    def pinned_file_sha256(path: Path) -> str:
        if path.name == "runtime_source_lock":
            return freeze_v2.VLLM_RUNTIME_LOCK_SHA256
        if path.name == "runtime_lock_input":
            return freeze_v1.PUBLICATION_FREEZE_RUNTIME_LOCK_INPUT_SHA256
        return original_file_sha256(path)

    monkeypatch.setattr(freeze_v1, "_file_sha256", pinned_file_sha256)
    monkeypatch.setattr(
        freeze_v1,
        "_read_canonical_json",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        freeze_v2,
        "validate_publication_campaign_plan_record",
        lambda _record: None,
    )
    monkeypatch.setattr(
        freeze_v1,
        "_validate_publication_latency_handoff_reference",
        lambda *_args, **_kwargs: None,
    )

    def reject_full_score(*_args: object, **_kwargs: object) -> None:
        raise ValueError("full-score semantic closure differs")

    monkeypatch.setattr(
        freeze_v2,
        "_validate_publication_full_score_references",
        reject_full_score,
    )
    monkeypatch.setattr(
        freeze_v2,
        "validate_vllm_flashinfer_runtime_artifact_closure",
        lambda **_kwargs: pytest.fail("runtime validation ran after full-score rejection"),
    )
    monkeypatch.setattr(
        freeze_v1,
        "_build_package_twice",
        lambda *_args, **_kwargs: pytest.fail("build ran after reference rejection"),
    )
    with pytest.raises(ValueError, match="full-score semantic closure differs"):
        freeze_v2.build_publication_source_closure_v2(inputs)
    assert not output.exists()

    observed_events: list[object] = []

    def reject_static_reference_replay(
        _inputs: freeze_v2.PublicationSourceClosureInputsV2,
        *,
        root: Path,
        latency_semantic_mode: freeze_v2._LatencySemanticValidationMode,
    ) -> None:
        assert root == repository
        observed_events.append(latency_semantic_mode)
        if latency_semantic_mode is freeze_v2._LatencySemanticValidationMode.STATIC:
            raise ValueError("source reference changed during latency validation")

    monkeypatch.setattr(
        freeze_v2,
        "_validate_source_references",
        reject_static_reference_replay,
    )
    monkeypatch.setattr(
        freeze_v1,
        "_build_package_twice",
        lambda *_args, **_kwargs: observed_events.append("build")
        or (object(), object()),
    )
    monkeypatch.setattr(
        freeze_v1,
        "_require_matching_build_outputs",
        lambda *_args: observed_events.append("match"),
    )
    with pytest.raises(ValueError, match="source reference changed"):
        freeze_v2.build_publication_source_closure_v2(inputs)
    assert observed_events == [
        freeze_v2._LatencySemanticValidationMode.EXECUTE,
        "build",
        "match",
        freeze_v2._LatencySemanticValidationMode.STATIC,
    ]
    assert not output.exists()


def test_latency_semantic_static_mode_pins_dependencies_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    plan_path = _touch(repository / "latency-plan.json", b"{}\n")
    prepared_input = (
        repository / freeze_v1._LATENCY_SEMANTIC_PREPARED_INPUT_RELATIVE_PATH
    )
    prepared_input.mkdir(parents=True)
    uv = _touch(tmp_path / "uv")
    python = _touch(tmp_path / "python")
    runtime_lock = _touch(repository / "latency-runtime.lock")
    monkeypatch.setattr(freeze_v1, "_LATENCY_SEMANTIC_UV_EXECUTABLE", uv)
    monkeypatch.setattr(freeze_v1, "_DEFAULT_PYTHON_EXECUTABLE", python)
    monkeypatch.setattr(
        freeze_v1,
        "_LATENCY_SEMANTIC_UV_SHA256",
        hashlib.sha256(uv.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        freeze_v1,
        "_LATENCY_SEMANTIC_PYTHON_SHA256",
        hashlib.sha256(python.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        freeze_v1,
        "_LATENCY_SEMANTIC_RUNTIME_LOCK_RELATIVE_PATH",
        runtime_lock.relative_to(repository),
    )
    events = []
    plan = {"coverage": {"task_count": 384}, "sharding": {"worker_count": 16}}
    attestation = {"validated": True}
    monkeypatch.setattr(
        freeze_v1,
        "_verified_publication_latency_runtime_lock",
        lambda path: events.append(("runtime_lock", path)) or b"lock\n",
    )
    monkeypatch.setattr(
        freeze_v1,
        "_validated_frozen_latency_handoff_plan",
        lambda path: events.append(("plan", path)) or plan,
    )
    monkeypatch.setattr(
        freeze_v1,
        "_input_bundle_closure_sha256",
        lambda path: events.append(("input", path)) or _digest("input"),
    )
    monkeypatch.setattr(
        freeze_v1,
        "_publication_latency_semantic_attestation",
        lambda observed: events.append(("attestation", observed)) or attestation,
    )
    monkeypatch.setattr(
        freeze_v1,
        "_validate_publication_latency_handoff_reference",
        lambda *_args, **_kwargs: pytest.fail("STATIC mode executed latency child"),
    )

    assert freeze_v2._validate_publication_latency_handoff_reference_v2(
        plan_path,
        repository_root=repository,
        mode=freeze_v2._LatencySemanticValidationMode.STATIC,
    ) == {"validated": True}
    assert events == [
        ("runtime_lock", runtime_lock),
        ("plan", plan_path),
        ("input", prepared_input),
        ("attestation", plan),
    ]

    execute_calls = []
    monkeypatch.setattr(
        freeze_v1,
        "_validate_publication_latency_handoff_reference",
        lambda path, *, repository_root: execute_calls.append(
            (path, repository_root)
        )
        or attestation,
    )
    assert freeze_v2._validate_publication_latency_handoff_reference_v2(
        plan_path,
        repository_root=repository,
        mode=freeze_v2._LatencySemanticValidationMode.EXECUTE,
    ) == attestation
    assert execute_calls == [(plan_path, repository)]


class _FullScoreCharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


def _full_score_source_record(dataset: str) -> dict[str, Any]:
    answer = f"answer-{dataset}"
    return {
        "dataset": dataset,
        "documents": [
            {
                "document_id": f"document-{dataset}",
                "text": f"natural source for {dataset}",
                "title": f"title-{dataset}",
            }
        ],
        "example_id": f"{dataset}-0",
        "expected_answer": answer,
        "query": f"question-{dataset}?",
        "references": [answer],
    }


def _portable_full_score_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    source_root = tmp_path / "full-score-sources"
    source_root.mkdir()
    source_paths = {}
    for dataset in SUPPORTED_V1_DATASETS:
        path = source_root / f"{dataset}.jsonl"
        path.write_text(
            json.dumps(_full_score_source_record(dataset)) + "\n",
            encoding="utf-8",
        )
        source_paths[dataset] = path
    inventory = load_full_score_inventory(
        source_paths,
        tokenizer=_FullScoreCharacterTokenizer(),
    )
    inventory_record = full_score_inventory_to_record(inventory)
    shard_plan_record = build_full_score_shard_plan(
        inventory,
        plan_id="full-score-v2-source-closure-test",
        max_workers=16,
        target_cache_prefix_tokens_per_shard=1,
    )
    inventory_path = tmp_path / "full-score-inventory.json"
    shard_plan_path = tmp_path / "full-score-shard-plan.json"
    inventory_path.write_bytes(
        freeze_v1._canonical_json_bytes(inventory_record, pretty=True)  # noqa: SLF001
    )
    shard_plan_path.write_bytes(
        freeze_v1._canonical_json_bytes(shard_plan_record, pretty=True)  # noqa: SLF001
    )
    cache_prefix_tokens = sum(item.cache_prefix_tokens for item in inventory.items)
    natural_prompt_tokens = sum(item.natural_prompt_tokens for item in inventory.items)
    authority = {
        "inventory_sha256": inventory.inventory_sha256,
        "item_count": len(inventory.items),
        "shard_plan_sha256": shard_plan_record["closed_record_sha256"],
        "shard_count": len(shard_plan_record["shards"]),
        "cache_prefix_tokens": cache_prefix_tokens,
        "natural_prompt_tokens": natural_prompt_tokens,
    }
    for name, value in {
        "_PUBLICATION_FULL_SCORE_INVENTORY_FILE_SHA256": hashlib.sha256(
            inventory_path.read_bytes()
        ).hexdigest(),
        "_PUBLICATION_FULL_SCORE_INVENTORY_FILE_SIZE": inventory_path.stat().st_size,
        "_PUBLICATION_FULL_SCORE_SHARD_PLAN_FILE_SHA256": hashlib.sha256(
            shard_plan_path.read_bytes()
        ).hexdigest(),
        "_PUBLICATION_FULL_SCORE_SHARD_PLAN_FILE_SIZE": (
            shard_plan_path.stat().st_size
        ),
        "FULL_SCORE_PUBLICATION_INVENTORY_SHA256": inventory.inventory_sha256,
        "FULL_SCORE_PUBLICATION_ITEM_COUNT": len(inventory.items),
        "FULL_SCORE_PUBLICATION_SHARD_PLAN_SHA256": shard_plan_record[
            "closed_record_sha256"
        ],
        "FULL_SCORE_PUBLICATION_SHARD_COUNT": len(shard_plan_record["shards"]),
        "FULL_SCORE_PUBLICATION_CACHE_PREFIX_TOKENS": cache_prefix_tokens,
        "FULL_SCORE_PUBLICATION_NATURAL_PROMPT_TOKENS": natural_prompt_tokens,
    }.items():
        monkeypatch.setattr(freeze_v2, name, value)
    campaign_record = {
        "budget": {
            "full_score_execution": {
                "cache_prefix_generation_tokens": cache_prefix_tokens,
                "example_count": len(inventory.items),
                "inventory_sha256": inventory.inventory_sha256,
                "natural_prompt_inference_tokens": natural_prompt_tokens,
                "shard_count": len(shard_plan_record["shards"]),
                "shard_plan_sha256": shard_plan_record["closed_record_sha256"],
            }
        },
        "full_score_program": {
            "datasets": list(SUPPORTED_V1_DATASETS),
            "max_natural_prompt_tokens": inventory.max_natural_prompt_tokens,
        },
    }
    return inventory_path, shard_plan_path, campaign_record, authority


def test_v2_full_score_references_require_complete_semantic_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, shard_plan, campaign, authority = _portable_full_score_references(
        tmp_path, monkeypatch
    )

    freeze_v2._validate_publication_full_score_references(
        inventory_path=inventory,
        shard_plan_path=shard_plan,
        campaign_record=campaign,
    )
    assert authority == {
        "inventory_sha256": freeze_v2.FULL_SCORE_PUBLICATION_INVENTORY_SHA256,
        "item_count": freeze_v2.FULL_SCORE_PUBLICATION_ITEM_COUNT,
        "shard_plan_sha256": freeze_v2.FULL_SCORE_PUBLICATION_SHARD_PLAN_SHA256,
        "shard_count": freeze_v2.FULL_SCORE_PUBLICATION_SHARD_COUNT,
        "cache_prefix_tokens": (freeze_v2.FULL_SCORE_PUBLICATION_CACHE_PREFIX_TOKENS),
        "natural_prompt_tokens": (
            freeze_v2.FULL_SCORE_PUBLICATION_NATURAL_PROMPT_TOKENS
        ),
    }


def test_v2_full_score_references_reject_cross_swap_and_resealed_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, shard_plan, campaign, _authority = _portable_full_score_references(
        tmp_path, monkeypatch
    )

    with pytest.raises(ValueError, match="inventory file differs"):
        freeze_v2._validate_publication_full_score_references(
            inventory_path=shard_plan,
            shard_plan_path=inventory,
            campaign_record=campaign,
        )

    canonical_shard_plan = shard_plan.read_bytes()
    noncanonical_shard_plan = (
        json.dumps(
            json.loads(canonical_shard_plan),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    shard_plan.write_bytes(noncanonical_shard_plan)
    monkeypatch.setattr(
        freeze_v2,
        "_PUBLICATION_FULL_SCORE_SHARD_PLAN_FILE_SHA256",
        hashlib.sha256(noncanonical_shard_plan).hexdigest(),
    )
    monkeypatch.setattr(
        freeze_v2,
        "_PUBLICATION_FULL_SCORE_SHARD_PLAN_FILE_SIZE",
        len(noncanonical_shard_plan),
    )
    with pytest.raises(ValueError, match="shard plan is not canonical JSON"):
        freeze_v2._validate_publication_full_score_references(
            inventory_path=inventory,
            shard_plan_path=shard_plan,
            campaign_record=campaign,
        )

    tampered = json.loads(canonical_shard_plan)
    tampered["coverage"]["identity_count"] += 1
    tampered["closed_record_sha256"] = full_score_inputs._closed_record_sha256(tampered)
    shard_plan.write_bytes(
        freeze_v1._canonical_json_bytes(tampered, pretty=True)  # noqa: SLF001
    )
    monkeypatch.setattr(
        freeze_v2,
        "_PUBLICATION_FULL_SCORE_SHARD_PLAN_FILE_SHA256",
        hashlib.sha256(shard_plan.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        freeze_v2,
        "_PUBLICATION_FULL_SCORE_SHARD_PLAN_FILE_SIZE",
        shard_plan.stat().st_size,
    )
    with pytest.raises(ValueError, match="does not match complete inventory"):
        freeze_v2._validate_publication_full_score_references(
            inventory_path=inventory,
            shard_plan_path=shard_plan,
            campaign_record=campaign,
        )


def _touch(path: Path, content: bytes | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if content is not None else f"{path.name}\n".encode())
    return path


def _preflight_inputs(
    tmp_path: Path,
) -> tuple[freeze_v2.GPUQualificationLocalPreflightInputsV2, dict[str, Any]]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _touch(repository / "pyproject.toml", b"[tool.pytest.ini_options]\n")
    plan = _plan()
    plan_path = repository / "gpu-qualification-plan-v2.json"
    plan_path.write_text(
        canonical_gpu_qualification_json(plan) + "\n",
        encoding="utf-8",
    )
    source_root = repository / "source-artifacts"
    input_bundle = repository / "input-bundle"
    source_root.mkdir()
    input_bundle.mkdir()
    _touch(source_root / "member.txt")
    _touch(input_bundle / "member.txt")
    tools = {
        name: _touch(repository / "tools" / name)
        for name in ("python", "ruff", "mypy")
    }
    inputs = freeze_v2.GPUQualificationLocalPreflightInputsV2(
        repository_root=repository,
        plan_json=plan_path,
        artifact_uris_json=_touch(repository / "artifact-uris.json", b"{}\n"),
        submit_payloads_json=_touch(repository / "submit-payloads.json", b"[]\n"),
        source_closure_json=_touch(repository / "source-closure.json", b"{}\n"),
        source_artifact_root=source_root,
        package_wheel=_touch(repository / "package.whl"),
        runner=_touch(repository / "runner.py"),
        input_bundle=input_bundle,
        runtime_source_lock=_touch(repository / "runtime-source.lock"),
        runtime_lock=_touch(repository / "runtime.lock"),
        patched_vllm_wheel=_touch(repository / "vllm.whl"),
        patched_vllm_manifest=_touch(repository / "vllm-manifest.json"),
        pristine_flashinfer_wheel=_touch(repository / "flashinfer-pristine.whl"),
        patched_flashinfer_wheel=_touch(repository / "flashinfer-patched.whl"),
        patched_flashinfer_manifest=_touch(
            repository / "flashinfer-manifest.json"
        ),
        runtime_closure_manifest=_touch(repository / "runtime-closure.json"),
        python_executable=tools["python"],
        ruff_executable=tools["ruff"],
        mypy_executable=tools["mypy"],
    )
    return inputs, plan


def test_runtime_artifact_check_dispatches_exact_eight_bound_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _plan_record = _preflight_inputs(tmp_path)
    observed: dict[str, Path] = {}
    monkeypatch.setattr(
        freeze_v2,
        "_require_runtime_artifact_file_pins",
        lambda _paths, *, pins: None,
    )

    def validate(**paths: Path) -> Mapping[str, Any]:
        observed.update(paths)
        return {
            "closed_record_sha256": RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
            "file_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
            "file_size": 6_634,
        }

    monkeypatch.setattr(
        freeze_v2,
        "validate_vllm_flashinfer_runtime_artifact_closure",
        validate,
    )
    result, bindings = freeze_v2._check_runtime_artifact_closure(inputs, _pins())

    assert tuple(observed) == (
        "source_lock",
        "base_lock",
        "vllm_wheel",
        "vllm_manifest",
        "pristine_flashinfer_wheel",
        "patched_flashinfer_wheel",
        "flashinfer_manifest",
        "closure_manifest",
    )
    assert tuple(item["label"] for item in bindings) == (
        "runtime_source_lock",
        "runtime_lock",
        "patched_vllm_wheel",
        "patched_vllm_manifest",
        "pristine_flashinfer_wheel",
        "patched_flashinfer_wheel",
        "patched_flashinfer_manifest",
        "runtime_closure_manifest",
    )
    assert result == {
        "authority_scope": "local_v2_complete_runtime_artifact_closure",
        "closed_record_sha256": RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
        "file_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "file_size": 6_634,
    }


@dataclass
class _Completed:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


def _clock() -> Callable[[], datetime]:
    current = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)

    def now() -> datetime:
        nonlocal current
        current += timedelta(seconds=1)
        return current

    return now


def _synthetic_python_check_specs(
    inputs: freeze_v2.GPUQualificationLocalPreflightInputsV2,
    **_kwargs: object,
) -> tuple[
    tuple[str, Callable[[], tuple[dict[str, Any], list[dict[str, Any]]]]], ...
]:
    paths = {
        field_name: getattr(inputs, field_name)
        for field_name in inputs.__dataclass_fields__
        if isinstance(getattr(inputs, field_name), Path)
    }
    checks = []
    for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS[:5]:
        bindings = freeze_v1._bindings(  # noqa: SLF001
            *(
                (paths[label], label)
                for label in freeze_v2._CHECK_INPUT_LABELS[check_id]
            )
        )
        result = {"check_id": check_id, "validated": True}
        checks.append(
            (
                check_id,
                lambda result=result, bindings=bindings: (result, bindings),
            )
        )
    return tuple(checks)


def _repository_binding(path: Path) -> dict[str, Any]:
    return {
        "branch": "codex/v2-preflight-test",
        "commit": "1" * 40,
        "commit_tree": "2" * 40,
        "label": "repository_root",
        "path": str(path.resolve()),
        "source_date_epoch": 1_777_777_777,
        "type": "git_repository",
    }


def _patch_synthetic_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    latency_modes: list[freeze_v2._LatencySemanticValidationMode] | None = None,
) -> None:
    monkeypatch.setattr(
        freeze_v2,
        "_require_canonical_preflight_tool_paths",
        lambda _inputs: None,
    )
    def observed_python_check_specs(
        inputs: freeze_v2.GPUQualificationLocalPreflightInputsV2,
        **kwargs: object,
    ) -> tuple[
        tuple[str, Callable[[], tuple[dict[str, Any], list[dict[str, Any]]]]],
        ...,
    ]:
        mode = kwargs.get("latency_semantic_mode")
        assert isinstance(mode, freeze_v2._LatencySemanticValidationMode)
        if latency_modes is not None:
            latency_modes.append(mode)
        return _synthetic_python_check_specs(inputs, **kwargs)

    monkeypatch.setattr(
        freeze_v2, "_python_check_specs", observed_python_check_specs
    )
    monkeypatch.setattr(freeze_v1, "_repository_binding", _repository_binding)
    monkeypatch.setattr(
        freeze_v1,
        "_python_api_identity",
        lambda: {"protocol_module": "publication_freeze_v1"},
    )
    monkeypatch.setattr(
        freeze_v2,
        "_python_api_identity_v2",
        lambda: {"protocol_module": "publication_freeze_v2"},
    )


def _runner(*, fail_check: str | None = None) -> freeze_v2.CommandRunner:
    def run(
        command: list[str] | tuple[str, ...],
        _cwd: Path,
        _environment: Mapping[str, str],
    ) -> _Completed:
        executable = Path(command[0]).name
        if "--version" in command:
            versions = {
                "python": "pytest 9.1.1\n",
                "ruff": "ruff 0.15.21\n",
                "mypy": "mypy 2.2.0 (compiled: yes)\n",
            }
            return _Completed(0, versions[executable].encode("utf-8"))
        if executable == fail_check:
            return _Completed(2, b"", f"{executable} failed\n".encode())
        if executable == "python":
            return _Completed(
                0,
                (
                    ".\n"
                    f"{freeze_v1.PUBLICATION_FREEZE_EXPECTED_TEST_COUNT} "
                    "passed in 1.00s\n"
                ).encode("utf-8"),
            )
        return _Completed(0, b"passed\n")

    return run


def _run_synthetic_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    latency_modes: list[freeze_v2._LatencySemanticValidationMode] | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    inputs, plan = _preflight_inputs(tmp_path)
    _patch_synthetic_preflight(monkeypatch, latency_modes=latency_modes)
    output = tmp_path / "preflight"
    evidence = freeze_v2._run_gpu_qualification_local_preflight_v2(
        inputs,
        output,
        command_runner=_runner(),
        now=_clock(),
    )
    return evidence, output, plan


def test_public_preflight_authoring_explicitly_enables_source_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _plan_record = _preflight_inputs(tmp_path)
    observed: dict[str, object] = {}
    sealed = {"sealed": True}

    def observe_authoring(
        *_args: object,
        **kwargs: object,
    ) -> dict[str, Any]:
        observed.update(kwargs)
        return sealed

    monkeypatch.setattr(
        freeze_v2,
        "_run_gpu_qualification_local_preflight_v2",
        observe_authoring,
    )

    assert freeze_v2.run_gpu_qualification_local_preflight_v2(
        inputs,
        tmp_path / "authoring-preflight",
    ) is sealed
    assert observed["verify_source_rebuild"] is True
    assert observed["final_latency_semantic_mode"] is (
        freeze_v2._LatencySemanticValidationMode.EXECUTE
    )


def test_preflight_writes_exact_eight_children_and_parent_file_seals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latency_modes = []
    evidence, output, _plan_record = _run_synthetic_preflight(
        tmp_path,
        monkeypatch,
        latency_modes=latency_modes,
    )
    assert latency_modes == [
        freeze_v2._LatencySemanticValidationMode.STATIC,
        freeze_v2._LatencySemanticValidationMode.STATIC,
        freeze_v2._LatencySemanticValidationMode.EXECUTE,
        freeze_v2._LatencySemanticValidationMode.STATIC,
    ]
    assert {path.name for path in output.iterdir()} == {
        *(f"{check_id}.json" for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS),
        "local-preflight-evidence.json",
    }
    assert [item["check_id"] for item in evidence["checks"]] == list(
        GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS
    )
    for item in evidence["checks"]:
        sidecar = output / f"{item['check_id']}.json"
        assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == item[
            "evidence_sha256"
        ]
        child = json.loads(sidecar.read_text(encoding="utf-8"))
        assert child["record_type"] == (
            "cachet.gpu_qualification.local_check_evidence.v2"
        )
        assert child["schema_version"] == 2
    unit_tests = json.loads(
        (output / "unit_tests.json").read_text(encoding="utf-8")
    )
    assert unit_tests["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    mypy = json.loads((output / "mypy.json").read_text(encoding="utf-8"))
    assert mypy["command"][1:] == [
        "--strict",
        "--no-incremental",
        "--cache-dir",
        "/dev/null",
        "--config-file",
        "pyproject.toml",
        *freeze_v2._V2_STATIC_ANALYSIS_TARGETS,
    ]
    assert (
        "src/document_kv_cache/publication_campaign_finalizer.py"
        in freeze_v2._V2_STATIC_ANALYSIS_TARGETS
    )
    assert (
        "src/document_kv_cache/publication_campaign_tables.py"
        in freeze_v2._V2_STATIC_ANALYSIS_TARGETS
    )
    assert (
        "src/document_kv_cache/gpu_qualification_sentinels.py"
        in freeze_v2._V2_STATIC_ANALYSIS_TARGETS
    )
    assert (
        "src/document_kv_cache/_gpu_qualification_sentinels_v2.py"
        in freeze_v2._V2_STATIC_ANALYSIS_TARGETS
    )
    assert (
        "src/document_kv_cache/_gpu_qualification_sentinel_worker.py"
        in freeze_v2._V2_STATIC_ANALYSIS_TARGETS
    )


def test_preflight_rejects_resealed_semantic_child_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, output, plan = _run_synthetic_preflight(tmp_path, monkeypatch)
    mypy_path = output / "mypy.json"
    original_mypy_bytes = mypy_path.read_bytes()
    evidence_path = output / "local-preflight-evidence.json"
    original_evidence_bytes = evidence_path.read_bytes()
    mypy = json.loads(original_mypy_bytes)
    del mypy["command"][3:5]
    mypy_path.write_bytes(
        freeze_v1._canonical_json_bytes(mypy, pretty=False)  # noqa: SLF001
    )
    resealed_mypy = build_local_preflight_evidence_v2(
        plan_sha256=plan["closed_record_sha256"],
        completed_at_utc=evidence["completed_at_utc"],
        check_evidence_sha256={
            check_id: hashlib.sha256(
                (output / f"{check_id}.json").read_bytes()
            ).hexdigest()
            for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS
        },
    )
    evidence_path.write_bytes(
        freeze_v1._canonical_json_bytes(resealed_mypy, pretty=False)  # noqa: SLF001
    )
    with pytest.raises(ValueError, match="mypy.*command differs"):
        freeze_v2._validate_preflight_bundle_structural(
            evidence_path,
            plan=plan,
        )
    mypy_path.write_bytes(original_mypy_bytes)
    evidence_path.write_bytes(original_evidence_bytes)

    child_path = output / "runtime_artifact_closure.json"
    child = json.loads(child_path.read_text(encoding="utf-8"))
    child["result"]["validated"] = False
    child_path.write_bytes(freeze_v1._canonical_json_bytes(child, pretty=False))  # noqa: SLF001
    resealed = build_local_preflight_evidence_v2(
        plan_sha256=plan["closed_record_sha256"],
        completed_at_utc=evidence["completed_at_utc"],
        check_evidence_sha256={
            check_id: hashlib.sha256(
                (output / f"{check_id}.json").read_bytes()
            ).hexdigest()
            for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS
        },
    )
    (output / "local-preflight-evidence.json").write_bytes(
        freeze_v1._canonical_json_bytes(resealed, pretty=False)  # noqa: SLF001
    )

    with pytest.raises(ValueError, match="runtime_artifact_closure.*result differs"):
        freeze_v2._validate_preflight_bundle_structural(
            output / "local-preflight-evidence.json",
            plan=plan,
        )

    replay_started = False

    def replay_must_not_start(*_args: object, **_kwargs: object) -> dict[str, Any]:
        nonlocal replay_started
        replay_started = True
        raise AssertionError("ephemeral live replay started before semantic validation")

    monkeypatch.setattr(
        freeze_v2,
        "_run_gpu_qualification_local_preflight_v2",
        replay_must_not_start,
    )
    with pytest.raises(ValueError, match="runtime_artifact_closure.*result differs"):
        freeze_v2.validate_gpu_qualification_local_preflight_bundle_v2(
            output / "local-preflight-evidence.json",
            plan_record=plan,
            submit_payloads=tuple({} for _ in range(14)),
            workspace_config=freeze_v2.DatabricksWorkspaceConfig(
                "https://workspace.example",
                "secret",
            ),
        )
    assert replay_started is False


def test_preflight_command_failure_never_writes_parent_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _plan_record = _preflight_inputs(tmp_path)
    _patch_synthetic_preflight(monkeypatch)
    output = tmp_path / "failed-preflight"

    with pytest.raises(RuntimeError, match="mypy.*returned 2"):
        freeze_v2._run_gpu_qualification_local_preflight_v2(
            inputs,
            output,
            command_runner=_runner(fail_check="mypy"),
            now=_clock(),
        )
    assert not (output / "local-preflight-evidence.json").exists()
    failed = json.loads((output / "mypy.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["exit_code"] == 2

    original_structural = freeze_v2._validate_preflight_bundle_structural
    execute_calls = 0

    def reject_latency_execution(
        evidence_path: Path,
        *,
        plan: Mapping[str, Any],
        latency_semantic_mode: freeze_v2._LatencySemanticValidationMode,
    ) -> tuple[dict[str, Any], freeze_v2.GPUQualificationLocalPreflightInputsV2]:
        nonlocal execute_calls
        if latency_semantic_mode is freeze_v2._LatencySemanticValidationMode.EXECUTE:
            execute_calls += 1
            raise RuntimeError("latency semantic child failed")
        return original_structural(
            evidence_path,
            plan=plan,
            latency_semantic_mode=latency_semantic_mode,
        )

    monkeypatch.setattr(
        freeze_v2,
        "_validate_preflight_bundle_structural",
        reject_latency_execution,
    )
    latency_failed_output = tmp_path / "latency-failed-preflight"
    with pytest.raises(RuntimeError, match="latency semantic child failed"):
        freeze_v2._run_gpu_qualification_local_preflight_v2(
            inputs,
            latency_failed_output,
            command_runner=_runner(),
            now=_clock(),
        )
    assert execute_calls == 1
    assert not (latency_failed_output / "local-preflight-evidence.json").exists()


def test_live_preflight_executes_latency_once_after_remote_and_detects_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, output, plan = _run_synthetic_preflight(tmp_path, monkeypatch)
    evidence_path = output / "local-preflight-evidence.json"
    original_structural = freeze_v2._validate_preflight_bundle_structural
    original_hashes = freeze_v2._preflight_bundle_file_hashes
    events: list[str] = []
    replay_source_rebuilds: list[bool] = []
    mutate_after_execute = False
    mutate_during_remote_at: int | None = None
    remote_failure_at: int | None = None
    remote_calls = 0
    ruff_path = output / "ruff.json"
    original_ruff_bytes = ruff_path.read_bytes()

    def observed_structural(
        path: Path,
        *,
        plan: Mapping[str, Any],
        latency_semantic_mode: freeze_v2._LatencySemanticValidationMode,
    ) -> tuple[dict[str, Any], freeze_v2.GPUQualificationLocalPreflightInputsV2]:
        events.append(f"structural:{latency_semantic_mode.value}")
        result = original_structural(
            path,
            plan=plan,
            latency_semantic_mode=latency_semantic_mode,
        )
        if (
            mutate_after_execute
            and latency_semantic_mode
            is freeze_v2._LatencySemanticValidationMode.EXECUTE
        ):
            ruff_path.write_bytes(ruff_path.read_bytes() + b" ")
        return result

    def observed_hashes(path: Path) -> dict[str, str]:
        events.append("hash")
        return original_hashes(path)

    def replay_without_latency(
        _inputs: freeze_v2.GPUQualificationLocalPreflightInputsV2,
        _output_root: str | Path,
        *,
        verify_source_rebuild: bool,
        final_latency_semantic_mode: freeze_v2._LatencySemanticValidationMode,
        **_kwargs: object,
    ) -> dict[str, Any]:
        replay_source_rebuilds.append(verify_source_rebuild)
        events.append(f"ephemeral:{final_latency_semantic_mode.value}")
        return evidence

    def validate_remote(*_args: object, **_kwargs: object) -> None:
        nonlocal remote_calls
        remote_calls += 1
        events.append("remote")
        if remote_calls == mutate_during_remote_at:
            ruff_path.write_bytes(ruff_path.read_bytes() + b" ")
        if remote_calls == remote_failure_at:
            raise RuntimeError("remote validation stopped")

    monkeypatch.setattr(
        freeze_v2,
        "_validate_preflight_bundle_structural",
        observed_structural,
    )
    monkeypatch.setattr(
        freeze_v2,
        "_preflight_bundle_file_hashes",
        observed_hashes,
    )
    monkeypatch.setattr(
        freeze_v2,
        "_run_gpu_qualification_local_preflight_v2",
        replay_without_latency,
    )
    monkeypatch.setattr(
        freeze_v2,
        "_validate_live_workspace_and_remote_artifacts_v2",
        validate_remote,
    )
    monkeypatch.setattr(
        freeze_v2,
        "_require_submit_payload_closure_binding",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        freeze_v2,
        "_qualification_single_user_name_from_payloads",
        lambda _payloads: "user@example.com",
    )
    submit_payloads = tuple({} for _ in range(14))
    config = freeze_v2.DatabricksWorkspaceConfig(
        "https://workspace.example",
        "secret",
    )

    assert freeze_v2.validate_gpu_qualification_local_preflight_bundle_v2(
        evidence_path,
        plan_record=plan,
        submit_payloads=submit_payloads,
        workspace_config=config,
    ) == evidence
    assert events == [
        "hash",
        "structural:static",
        "ephemeral:static",
        "remote",
        "hash",
        "structural:execute",
        "hash",
        "remote",
        "structural:static",
        "hash",
    ]

    events.clear()
    remote_calls = 0
    remote_failure_at = 1
    with pytest.raises(RuntimeError, match="remote validation stopped"):
        freeze_v2.validate_gpu_qualification_local_preflight_bundle_v2(
            evidence_path,
            plan_record=plan,
            submit_payloads=submit_payloads,
            workspace_config=config,
        )
    assert events == [
        "hash",
        "structural:static",
        "ephemeral:static",
        "remote",
    ]

    events.clear()
    remote_calls = 0
    remote_failure_at = 2
    with pytest.raises(RuntimeError, match="remote validation stopped"):
        freeze_v2.validate_gpu_qualification_local_preflight_bundle_v2(
            evidence_path,
            plan_record=plan,
            submit_payloads=submit_payloads,
            workspace_config=config,
        )
    assert events == [
        "hash",
        "structural:static",
        "ephemeral:static",
        "remote",
        "hash",
        "structural:execute",
        "hash",
        "remote",
    ]

    events.clear()
    remote_calls = 0
    remote_failure_at = None
    mutate_after_execute = True
    with pytest.raises(
        ValueError,
        match="changed during latency semantic validation",
    ):
        freeze_v2.validate_gpu_qualification_local_preflight_bundle_v2(
            evidence_path,
            plan_record=plan,
            submit_payloads=submit_payloads,
            workspace_config=config,
        )
    assert events == [
        "hash",
        "structural:static",
        "ephemeral:static",
        "remote",
        "hash",
        "structural:execute",
        "hash",
    ]

    ruff_path.write_bytes(original_ruff_bytes)
    events.clear()
    remote_calls = 0
    mutate_after_execute = False
    mutate_during_remote_at = 2
    with pytest.raises(ValueError):
        freeze_v2.validate_gpu_qualification_local_preflight_bundle_v2(
            evidence_path,
            plan_record=plan,
            submit_payloads=submit_payloads,
            workspace_config=config,
        )
    assert events == [
        "hash",
        "structural:static",
        "ephemeral:static",
        "remote",
        "hash",
        "structural:execute",
        "hash",
        "remote",
        "structural:static",
    ]
    ruff_path.write_bytes(original_ruff_bytes)
    assert replay_source_rebuilds == [False] * 5


def test_v2_artifact_and_check_role_constants_are_exact() -> None:
    assert GPU_QUALIFICATION_V2_ARTIFACT_KEYS == (
        "cachet_source_tree_sha256",
        "input_bundle_sha256",
        "package_wheel_sha256",
        "patched_flashinfer_wheel_sha256",
        "patched_vllm_wheel_sha256",
        "runner_sha256",
        "runtime_closure_manifest_sha256",
        "runtime_lock_sha256",
    )
    assert GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS == (
        "canonical_plan_schema",
        "runtime_lock_require_hashes",
        "patched_wheel_record_and_manifest",
        "runtime_artifact_closure",
        "source_runner_input_closure",
        "unit_tests",
        "ruff",
        "mypy",
    )
