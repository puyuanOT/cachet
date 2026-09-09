from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import cachet.publication_freeze as cachet_freeze
import document_kv_cache.databricks_runs as databricks_runs
import document_kv_cache.publication_campaign as campaign
import document_kv_cache.publication_freeze as freeze
from document_kv_cache.gpu_qualification import (
    GPUQualificationArtifactPins,
    build_gpu_qualification_plan,
    build_local_preflight_evidence,
    canonical_gpu_qualification_json,
)
from document_kv_cache.gpu_qualification_databricks import (
    GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_ID,
    PUBLICATION_CAMPAIGN_LEDGER_ID,
    PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS,
)
from document_kv_cache.release_bundle import RELEASE_BUNDLE_PACKAGE_CONSOLE_SCRIPTS
from document_kv_cache.serving_env import VLLM_RUNTIME_LOCK_SHA256


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SEMANTIC_RUNTIME_LOCK = (
    _REPOSITORY_ROOT / "src/document_kv_cache/runtime_locks/"
    "publication-latency-semantic-py311-macos-arm64.lock"
)
_WORKSPACE_CONFIG = freeze.DatabricksWorkspaceConfig(
    "https://dbc.example",
    "test-token",
)
_SINGLE_USER_NAME = "publication@example.com"


@pytest.fixture(autouse=True)
def _portable_subprocess_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bind real, empty test directories; retain production path validation and
    # environment sanitization without requiring the operator's macOS layout.
    for attribute, name in (("_FREEZE_HOME", "empty-home"), ("_FREEZE_TMPDIR", "temp")):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setattr(freeze, attribute, directory)


@pytest.fixture(autouse=True)
def _stub_exact_workspace_current_user(monkeypatch: pytest.MonkeyPatch) -> None:
    def require(_config, *, expected_user_name: str):
        assert expected_user_name == _SINGLE_USER_NAME
        return {"authenticated": True}

    monkeypatch.setattr(freeze, "require_databricks_current_user_name", require)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("implementation", "version", "build_version", "error"),
    (
        ("CPython", "3.11.16", "1.2.2.post1", None),
        ("PyPy", "3.11.16", "1.2.2.post1", "requires CPython"),
        ("CPython", "3.12.0", "1.2.2.post1", "requires CPython"),
        ("CPython", "3.11.16", "1.2.2", "requires build=="),
    ),
)
def test_source_freeze_requires_exact_toolchain(
    monkeypatch: pytest.MonkeyPatch,
    implementation: str,
    version: str,
    build_version: str,
    error: str | None,
) -> None:
    monkeypatch.setattr(
        freeze.platform, "python_implementation", lambda: implementation
    )
    monkeypatch.setattr(freeze.platform, "python_version", lambda: version)
    monkeypatch.setattr(
        freeze.importlib.metadata, "version", lambda _name: build_version
    )
    if error is None:
        freeze._require_freeze_toolchain()
    else:
        with pytest.raises(RuntimeError, match=error):
            freeze._require_freeze_toolchain()


@pytest.mark.parametrize("attribute", ("_FREEZE_HOME", "_FREEZE_TMPDIR"))
@pytest.mark.parametrize("kind", ("missing", "symlink"))
def test_subprocess_directory_bindings_retain_path_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    kind: str,
) -> None:
    invalid = tmp_path / "invalid-directory"
    if kind == "symlink":
        invalid.symlink_to(getattr(freeze, attribute), target_is_directory=True)
    monkeypatch.setattr(freeze, attribute, invalid)
    with pytest.raises(ValueError, match="regular directory|symlink"):
        freeze._base_subprocess_environment()


def _package_build_outputs(
    *, package_content: bytes = b"VALUE = 1\n"
) -> freeze._PackageBuildOutputs:
    """Small valid archives; no installed wheel or retained Git object is used."""
    dist_info = "cachet_kv-0.2.0.dist-info"
    members = {
        "cachet/__init__.py": package_content,
        "cachet/__init__.pyi": b"",
        "cachet/py.typed": b"",
        "document_kv_cache/__init__.py": package_content,
        "document_kv_cache/py.typed": b"",
        f"{dist_info}/METADATA": (
            b"Name: cachet-kv\nVersion: 0.2.0\nLicense-Expression: Apache-2.0\n"
            b"License-File: LICENSE\n"
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{dist_info}/licenses/LICENSE": b"Apache License 2.0\n",
        f"{dist_info}/entry_points.txt": (
            "[console_scripts]\n"
            + "".join(
                f"{name}={target}\n"
                for name, target in sorted(
                    RELEASE_BUNDLE_PACKAGE_CONSOLE_SCRIPTS.items()
                )
            )
        ).encode(),
    }
    record_lines = []
    for name, content in sorted(members.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .decode()
            .rstrip("=")
        )
        record_lines.append(f"{name},sha256={digest},{len(content)}")
    members[f"{dist_info}/RECORD"] = (
        "\n".join([*record_lines, f"{dist_info}/RECORD,,"]) + "\n"
    ).encode()
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    sdist = io.BytesIO()
    with gzip.GzipFile(fileobj=sdist, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            info = tarfile.TarInfo("cachet_kv-0.2.0/src/cachet/__init__.py")
            info.size = len(package_content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(package_content))
    return freeze._PackageBuildOutputs(
        "cachet_kv-0.2.0-py3-none-any.whl",
        wheel.getvalue(),
        "cachet_kv-0.2.0.tar.gz",
        sdist.getvalue(),
    )


def _source_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> freeze.PublicationSourceClosureInputs:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["'
        + freeze.PUBLICATION_FREEZE_BUILD_BACKEND
        + '"]\nbuild-backend = "poetry.core.masonry.api"\n',
        encoding="utf-8",
    )
    references = repository / "references"
    references.mkdir()
    runtime_paths = {}
    for role, filename in (
        ("runtime_lock", "vllm-0.27.1-cu129-py311-manylinux_2_35.lock"),
        ("runtime_lock_input", "vllm-0.27.1-cu129-py311-manylinux_2_35.in"),
    ):
        destination = references / filename
        destination.write_bytes(
            (
                _REPOSITORY_ROOT / "src/document_kv_cache/runtime_locks" / filename
            ).read_bytes()
        )
        runtime_paths[role] = destination
    campaign_path = references / "campaign.json"
    campaign.write_publication_campaign_plan_json(
        campaign.build_publication_campaign_plan(
            PUBLICATION_CAMPAIGN_ID,
            campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
            campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
            campaign_ledger_prefix=PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
            campaign_opening_terminal_gpu_hours=PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS,
        ),
        campaign_path,
    )
    other_paths = {}
    for role in (
        "latency_handoff_plan",
        "full_score_inventory",
        "full_score_shard_plan",
    ):
        path = references / f"{role}.json"
        path.write_bytes(
            freeze._canonical_json_bytes(
                {"synthetic_source_reference": role}, pretty=True
            )
        )
        other_paths[role] = path
    # Exercise actual clean-tree identity and Git archive reproduction with an
    # object that also exists in a shallow CI checkout. No old Git commit needed.
    freeze._git(repository, "init", "--quiet", "--initial-branch=codex/freeze-fixture")
    freeze._git(repository, "add", ".")
    freeze._git(
        repository,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "Synthetic source fixture",
    )
    outputs = _package_build_outputs()
    monkeypatch.setattr(freeze, "_require_freeze_toolchain", lambda: None)
    monkeypatch.setattr(
        freeze, "_build_package_twice", lambda *_args, **_kwargs: (outputs, outputs)
    )
    # Tokenizer/runtime semantic execution is tested separately. File closure,
    # package/tar validators, campaign schema and real Git reproduction stay on.
    monkeypatch.setattr(
        freeze,
        "_validate_publication_latency_handoff_reference",
        lambda *_args, **_kwargs: {},
    )
    return freeze.PublicationSourceClosureInputs(
        repository_root=repository,
        artifact_output_root=tmp_path / "source-artifacts",
        campaign_plan=campaign_path,
        **runtime_paths,
        **other_paths,
    )


def _latency_plan_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Synthetic reference-boundary records; external token replay is separate."""
    workers = [
        {
            "items": [
                {
                    "segment_token_contracts": [],
                    "segment_token_contracts_sha256": _digest("empty-contracts"),
                }
                for _ in range(24)
            ]
        }
        for _ in range(16)
    ]
    record = {
        "plan_id": "synthetic-latency-reference",
        "input_bundle_sha256": _digest("synthetic-inputs"),
        "workers_sha256": _digest(
            json.dumps(
                workers,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "workers": workers,
        "sharding": {"worker_count": 16},
        "coverage": {"task_count": 384, "cache_prefix_generation_tokens": 7_323_967},
    }
    record["closed_record_sha256"] = _digest(canonical_gpu_qualification_json(record))
    current = tmp_path / "current-plan.json"
    current.write_bytes(freeze._canonical_json_bytes(record, pretty=True))
    for name, value in {
        "PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_FILE_SHA256": hashlib.sha256(
            current.read_bytes()
        ).hexdigest(),
        "PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_CLOSED_RECORD_SHA256": record[
            "closed_record_sha256"
        ],
        "PUBLICATION_FREEZE_LATENCY_HANDOFF_PLAN_ID": record["plan_id"],
        "PUBLICATION_FREEZE_INPUT_BUNDLE_SHA256": record["input_bundle_sha256"],
        "PUBLICATION_FREEZE_LATENCY_HANDOFF_WORKERS_SHA256": record["workers_sha256"],
        "_LATENCY_SEMANTIC_PREPARED_INPUT_RELATIVE_PATH": Path("prepared"),
    }.items():
        monkeypatch.setattr(freeze, name, value)
    (tmp_path / "prepared").mkdir()
    stale_record = json.loads(current.read_bytes())
    del stale_record["workers"][0]["items"][0]["segment_token_contracts"]
    del stale_record["closed_record_sha256"]
    stale_record["closed_record_sha256"] = _digest(
        canonical_gpu_qualification_json(stale_record)
    )
    stale = tmp_path / "stale-plan.json"
    stale.write_bytes(freeze._canonical_json_bytes(stale_record, pretty=True))
    return current, stale


def test_source_closure_latency_semantics_reject_missing_contract_and_accept_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    current, stale = _latency_plan_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="semantically stale"):
        freeze._validate_publication_latency_handoff_reference(
            stale,
            repository_root=tmp_path,
        )

    plan = freeze._validated_frozen_latency_handoff_plan(current)
    expected = freeze._publication_latency_semantic_attestation(plan)
    monkeypatch.setattr(
        freeze,
        "_run_publication_latency_semantic_subprocess",
        lambda **_kwargs: expected,
    )
    assert freeze._validate_publication_latency_handoff_reference(
        current,
        repository_root=tmp_path,
    ) == expected


def test_latency_semantic_child_rejects_alternate_valid_plan_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    current, _stale = _latency_plan_fixture(tmp_path, monkeypatch)
    alternate = json.loads(current.read_text(encoding="utf-8"))
    alternate["plan_id"] = "alternate-semantically-valid-plan"
    payload = dict(alternate)
    payload.pop("closed_record_sha256")
    alternate["closed_record_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    alternate_path = tmp_path / "alternate-latency-plan.json"
    alternate_path.write_bytes(
        json.dumps(
            alternate,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )

    with pytest.raises(ValueError, match="frozen semantic plan"):
        monkeypatch.setattr(
            freeze,
            "_require_publication_latency_semantic_runtime",
            lambda _path: tmp_path,
        )
        freeze._run_publication_latency_semantic_check(
            plan_path=alternate_path,
            prepared_input_dir=_REPOSITORY_ROOT,
            runtime_lock_path=tmp_path / "unused.lock",
        )


def test_latency_semantic_runtime_lock_rejects_one_changed_byte(tmp_path: Path):
    content = bytearray(_SEMANTIC_RUNTIME_LOCK.read_bytes())
    content[-2] = ord("0") if content[-2] != ord("0") else ord("1")
    tampered = tmp_path / "tampered-runtime.lock"
    tampered.write_bytes(content)

    with pytest.raises(RuntimeError, match="runtime lock bytes differ"):
        freeze._verified_publication_latency_runtime_lock(tampered)


def test_latency_semantic_site_package_tamper_keeps_identity_but_rejects(
    tmp_path: Path,
):
    site_packages = tmp_path / "site-packages"
    package = site_packages / "example_package"
    metadata = site_packages / "example_package-1.0.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    module = package / "__init__.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    (metadata / "METADATA").write_text(
        "Name: example-package\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (metadata / "RECORD").write_text("", encoding="utf-8")
    file_count, record_count, digest = (
        freeze._semantic_site_packages_closure(site_packages)
    )
    freeze._require_semantic_site_packages_closure(
        site_packages,
        expected_file_count=file_count,
        expected_record_count=record_count,
        expected_sha256=digest,
    )

    module.write_text("VALUE = 2\n", encoding="utf-8")
    assert (metadata / "METADATA").read_text(encoding="utf-8") == (
        "Name: example-package\nVersion: 1.0\n"
    )
    with pytest.raises(RuntimeError, match="installed package closure differs"):
        freeze._require_semantic_site_packages_closure(
            site_packages,
            expected_file_count=file_count,
            expected_record_count=record_count,
            expected_sha256=digest,
        )


def test_private_tokenizer_uses_captured_bytes_after_shared_cache_mutates(
    tmp_path: Path,
):
    shared = tmp_path / "shared-tokenizer.json"
    shared.write_bytes(b"verified tokenizer bytes")
    files = (("tokenizer.json", "../../blobs/example", shared.read_bytes()),)
    snapshot = freeze._VerifiedTokenizerSnapshot(
        sha256=freeze._verified_tokenizer_snapshot_digest(files),
        files=files,
    )
    shared.write_bytes(b"mutated shared cache")

    with freeze._private_publication_latency_tokenizer_snapshot(
        snapshot,
        temporary_parent=tmp_path,
    ) as private:
        assert (private / "tokenizer.json").read_bytes() == (
            b"verified tokenizer bytes"
        )


def test_private_tokenizer_rejects_mutation_during_validation(tmp_path: Path):
    files = (("tokenizer.json", "../../blobs/example", b"verified bytes"),)
    snapshot = freeze._VerifiedTokenizerSnapshot(
        sha256=freeze._verified_tokenizer_snapshot_digest(files),
        files=files,
    )

    with pytest.raises(RuntimeError, match="snapshot bytes changed"):
        with freeze._private_publication_latency_tokenizer_snapshot(
            snapshot,
            temporary_parent=tmp_path,
        ) as private:
            (private / "tokenizer.json").write_bytes(b"mutated during validation")


def test_source_closure_cannot_issue_when_latency_semantic_authority_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs = _source_inputs(tmp_path, monkeypatch)

    def reject(*_args, **_kwargs):
        raise ValueError("semantic authority rejected latency plan")

    monkeypatch.setattr(
        freeze,
        "_validate_publication_latency_handoff_reference",
        reject,
    )
    with pytest.raises(ValueError, match="semantic authority rejected"):
        freeze.build_publication_source_closure(inputs)
    assert not inputs.artifact_output_root.exists()


def test_source_closure_builder_binds_current_bootstrap_and_reference_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs = _source_inputs(tmp_path, monkeypatch)

    record = freeze.build_publication_source_closure(inputs)

    bootstrap = next(
        item
        for item in record["files"]
        if item["role"] == "gpu_qualification_bootstrap"
    )
    runner_bytes = freeze.GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT.encode("utf-8")
    assert bootstrap["byte_count"] == len(runner_bytes)
    assert bootstrap["sha256"] == hashlib.sha256(runner_bytes).hexdigest()
    assert bootstrap["sha256"] == freeze.GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256
    assert (inputs.artifact_output_root / bootstrap["relative_path"]).read_bytes() == runner_bytes
    for reference in record["references"]:
        raw = (inputs.repository_root / reference["path"]).read_bytes()
        assert reference["byte_count"] == len(raw)
        assert reference["sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["runtime"]["runtime_lock_sha256"] == freeze.VLLM_RUNTIME_LOCK_SHA256
    output = inputs.artifact_output_root / "cachet-source-closure.json"
    freeze.write_publication_source_closure_json(
        record,
        output,
        repository_root=inputs.repository_root,
        artifact_root=inputs.artifact_output_root,
    )
    assert output.read_bytes() == freeze._canonical_json_bytes(record, pretty=True)
    with pytest.raises(FileExistsError):
        freeze.write_publication_source_closure_json(
            record,
            output,
            repository_root=inputs.repository_root,
            artifact_root=inputs.artifact_output_root,
        )


def test_source_closure_rejects_resealed_file_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs = _source_inputs(tmp_path, monkeypatch)
    record = freeze.build_publication_source_closure(inputs)
    record["files"][0]["sha256"] = _digest("forged-wheel")
    record["closed_record_sha256"] = freeze._closed_record_sha256(record)

    with pytest.raises(ValueError, match="bytes differ"):
        freeze.validate_publication_source_closure_record(
            record,
            repository_root=inputs.repository_root,
            artifact_root=inputs.artifact_output_root,
        )


def test_source_closure_requires_independent_build_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs = _source_inputs(tmp_path, monkeypatch)
    first = _package_build_outputs()
    second = freeze._PackageBuildOutputs(
        first.wheel_name,
        first.wheel_bytes + b"different",
        first.sdist_name,
        first.sdist_bytes,
    )
    monkeypatch.setattr(
        freeze,
        "_build_package_twice",
        lambda *_args, **_kwargs: (first, second),
    )

    with pytest.raises(ValueError, match="two isolated clean-tree package builds"):
        freeze.build_publication_source_closure(inputs)
    assert "independent_package_wheel" not in (
        freeze.PublicationSourceClosureInputs.__dataclass_fields__
    )


def test_source_validator_rejects_structurally_valid_unrelated_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs = _source_inputs(tmp_path, monkeypatch)
    record = freeze.build_publication_source_closure(inputs)
    unrelated = _package_build_outputs(package_content=b"UNRELATED_PACKAGE = True\n")
    wheel_path = inputs.artifact_output_root / record["files"][0]["relative_path"]
    wheel_path.write_bytes(unrelated.wheel_bytes)
    record["files"][0]["byte_count"] = wheel_path.stat().st_size
    record["files"][0]["sha256"] = hashlib.sha256(
        wheel_path.read_bytes()
    ).hexdigest()
    record["closed_record_sha256"] = freeze._closed_record_sha256(record)

    with pytest.raises(ValueError, match="clean-tree rebuild.*bytes differ"):
        freeze.validate_publication_source_closure_record(
            record,
            repository_root=inputs.repository_root,
            artifact_root=inputs.artifact_output_root,
        )


def _plan(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    pins = GPUQualificationArtifactPins(
        runtime_lock_sha256=VLLM_RUNTIME_LOCK_SHA256,
        patched_vllm_wheel_sha256=(
            freeze.PUBLICATION_FREEZE_PATCHED_VLLM_WHEEL_SHA256
        ),
        package_wheel_sha256=_digest("package-wheel"),
        cachet_source_tree_sha256=_digest("source-closure"),
        runner_sha256=GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256,
        input_bundle_sha256=freeze.PUBLICATION_FREEZE_INPUT_BUNDLE_SHA256,
    )
    plan = build_gpu_qualification_plan(
        campaign_id=PUBLICATION_CAMPAIGN_ID,
        campaign_record_sha256=PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
        campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
        ),
        artifact_pins=pins,
    )
    path = tmp_path / "gpu-qualification-plan.json"
    path.write_text(
        canonical_gpu_qualification_json(plan) + "\n",
        encoding="utf-8",
    )
    return plan, path


def _preflight_inputs(tmp_path: Path, plan_path: Path) -> freeze.GPUQualificationLocalPreflightInputs:
    executable_paths = []
    for name in ("python", "ruff", "mypy"):
        path = tmp_path / name
        path.write_text("tool\n", encoding="utf-8")
        executable_paths.append(path)
    def artifact_file(name: str) -> Path:
        path = tmp_path / name
        path.write_bytes(f"{name}\n".encode())
        return path

    def artifact_directory(name: str) -> Path:
        path = tmp_path / name
        path.mkdir()
        (path / "member.txt").write_text(f"{name}\n", encoding="utf-8")
        return path

    submit_payloads = tmp_path / "submit-payloads.json"
    submit_payloads.write_bytes(
        freeze._canonical_json_bytes(
            [
                {
                    "payload_index": index,
                    "tasks": [
                        {
                            "new_cluster": {
                                "data_security_mode": "SINGLE_USER",
                                "single_user_name": _SINGLE_USER_NAME,
                            }
                        }
                    ],
                }
                for index in range(14)
            ],
            pretty=False,
        )
    )

    return freeze.GPUQualificationLocalPreflightInputs(
        repository_root=_REPOSITORY_ROOT,
        plan_json=plan_path,
        artifact_uris_json=artifact_file("artifact-uris.json"),
        submit_payloads_json=submit_payloads,
        source_closure_json=artifact_file("source-closure.json"),
        source_artifact_root=artifact_directory("source-artifacts"),
        package_wheel=artifact_file("package.whl"),
        runner=artifact_file("runner.py"),
        input_bundle=artifact_directory("input-bundle"),
        runtime_lock=artifact_file("runtime.lock"),
        runtime_lock_input=artifact_file("runtime.in"),
        official_vllm_wheel=artifact_file("official.whl"),
        patched_vllm_wheel=artifact_file("patched.whl"),
        patched_vllm_manifest=artifact_file("patched-manifest.json"),
        python_executable=executable_paths[0],
        ruff_executable=executable_paths[1],
        mypy_executable=executable_paths[2],
    )


def _bound_submit_payloads(
    inputs: freeze.GPUQualificationLocalPreflightInputs,
) -> tuple[dict[str, Any], ...]:
    raw = json.loads(inputs.submit_payloads_json.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    return tuple(raw)


@dataclass
class _Completed:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


def _fake_runner(
    command: list[str] | tuple[str, ...],
    _cwd: Path,
    _environment: dict[str, str],
) -> _Completed:
    if "--version" in command:
        name = Path(command[0]).name
        versions = {
            "python": "pytest 9.1.1\n",
            "ruff": "ruff 0.15.21\n",
            "mypy": "mypy 2.2.0 (compiled: yes)\n",
        }
        return _Completed(0, versions[name].encode("utf-8"))
    if "pytest" in command:
        return _Completed(
            0,
            (
                ".\n"
                f"{freeze.PUBLICATION_FREEZE_EXPECTED_TEST_COUNT} "
                "passed in 1.00s\n"
            ).encode("utf-8"),
        )
    return _Completed(0, b"passed\n")


def _patch_in_process_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        freeze,
        "_require_canonical_preflight_tool_paths",
        lambda _inputs: None,
    )
    monkeypatch.setattr(
        freeze,
        "_validate_live_workspace_and_remote_artifacts",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        freeze,
        "_check_canonical_plan",
        lambda inputs, *_args, **_kwargs: (
            {"validated": True},
            freeze._bindings(
                (inputs.plan_json, "plan_json"),
                (inputs.artifact_uris_json, "artifact_uris_json"),
                (inputs.submit_payloads_json, "submit_payloads_json"),
            ),
        ),
    )
    monkeypatch.setattr(
        freeze,
        "_check_runtime_lock",
        lambda inputs, *_args: (
            {"validated": True},
            freeze._bindings(
                (inputs.runtime_lock, "runtime_lock"),
                (inputs.runtime_lock_input, "runtime_lock_input"),
                (inputs.package_wheel, "package_wheel"),
            ),
        ),
    )
    monkeypatch.setattr(
        freeze,
        "_check_patched_wheel",
        lambda inputs, *_args: (
            {"validated": True},
            freeze._bindings(
                (inputs.official_vllm_wheel, "official_vllm_wheel"),
                (inputs.patched_vllm_wheel, "patched_vllm_wheel"),
                (inputs.patched_vllm_manifest, "patched_vllm_manifest"),
            ),
        ),
    )
    monkeypatch.setattr(
        freeze,
        "_check_source_runner_inputs",
        lambda inputs, *_args, **_kwargs: (
            {"validated": True},
            freeze._bindings(
                (inputs.source_closure_json, "source_closure_json"),
                (inputs.source_artifact_root, "source_artifact_root"),
                (inputs.package_wheel, "package_wheel"),
                (inputs.runner, "runner"),
                (inputs.input_bundle, "input_bundle"),
            ),
        ),
    )
    monkeypatch.setattr(
        freeze,
        "_repository_binding",
        lambda path: {
            "branch": "codex/publication-freeze-test",
            "commit": "1" * 40,
            "commit_tree": "2" * 40,
            "label": "repository_root",
            "path": str(path.resolve()),
            "source_date_epoch": 1_777_777_777,
            "type": "git_repository",
        },
    )


def _clock() -> Any:
    current = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)

    def now() -> datetime:
        nonlocal current
        current += timedelta(seconds=1)
        return current

    return now


def test_preflight_runner_derives_hashes_and_detects_sidecar_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan, plan_path = _plan(tmp_path)
    inputs = _preflight_inputs(tmp_path, plan_path)
    _patch_in_process_checks(monkeypatch)
    monkeypatch.setattr(freeze, "_run_command", _fake_runner)
    output = tmp_path / "preflight"

    evidence = freeze._run_gpu_qualification_local_preflight(
        inputs,
        output,
        command_runner=_fake_runner,
        now=_clock(),
    )

    assert [item["check_id"] for item in evidence["checks"]] == list(
        freeze._LOCAL_CHECK_IDS
    )
    for item in evidence["checks"]:
        sidecar = output / f"{item['check_id']}.json"
        assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == item[
            "evidence_sha256"
        ]
    ruff = json.loads((output / "ruff.json").read_text(encoding="utf-8"))
    mypy = json.loads((output / "mypy.json").read_text(encoding="utf-8"))
    assert ruff["command"][1:2] == ["check"]
    assert ruff["command"][-len(freeze._STATIC_ANALYSIS_TARGETS) :] == list(
        freeze._STATIC_ANALYSIS_TARGETS
    )
    assert mypy["command"][1:] == [
        "--strict",
        "--no-incremental",
        "--cache-dir",
        "/dev/null",
        "--config-file",
        "pyproject.toml",
        *freeze._MYPY_TARGETS,
    ]
    assert ruff["environment"] == freeze._preflight_environment(
        inputs.repository_root
    )
    assert mypy["environment"] == freeze._preflight_environment(
        inputs.repository_root
    )
    assert ruff["result"]["excluded_known_dirty_paths"] == [
        "src/document_kv_cache/databricks_runs.py"
    ]
    freeze.validate_gpu_qualification_local_preflight_bundle(
        output / "local-preflight-evidence.json",
        plan_record=plan,
        submit_payloads=_bound_submit_payloads(inputs),
        workspace_config=_WORKSPACE_CONFIG,
    )
    sidecar = output / "ruff.json"
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    record["result"]["validated"] = False
    sidecar.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sidecar hashes differ"):
        freeze.validate_gpu_qualification_local_preflight_bundle(
            output / "local-preflight-evidence.json",
            plan_record=plan,
            submit_payloads=_bound_submit_payloads(inputs),
            workspace_config=_WORKSPACE_CONFIG,
        )


def test_preflight_bundle_rejects_resealed_semantic_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan, plan_path = _plan(tmp_path)
    inputs = _preflight_inputs(tmp_path, plan_path)
    _patch_in_process_checks(monkeypatch)
    monkeypatch.setattr(freeze, "_run_command", _fake_runner)
    output = tmp_path / "preflight"
    freeze._run_gpu_qualification_local_preflight(
        inputs,
        output,
        command_runner=_fake_runner,
        now=_clock(),
    )
    mypy_path = output / "mypy.json"
    original_mypy_bytes = mypy_path.read_bytes()
    evidence_path = output / "local-preflight-evidence.json"
    original_evidence_bytes = evidence_path.read_bytes()
    mypy = json.loads(mypy_path.read_text(encoding="utf-8"))
    del mypy["command"][3:5]
    mypy_path.write_text(
        json.dumps(mypy, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    old_evidence = json.loads(
        evidence_path.read_text(encoding="utf-8")
    )
    resealed = build_local_preflight_evidence(
        plan_sha256=plan["closed_record_sha256"],
        completed_at_utc=old_evidence["completed_at_utc"],
        check_evidence_sha256={
            check_id: hashlib.sha256(
                (output / f"{check_id}.json").read_bytes()
            ).hexdigest()
            for check_id in freeze._LOCAL_CHECK_IDS
        },
    )
    evidence_path.write_text(
        canonical_gpu_qualification_json(resealed) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mypy.*command differs"):
        freeze.validate_gpu_qualification_local_preflight_bundle(
            evidence_path,
            plan_record=plan,
            submit_payloads=_bound_submit_payloads(inputs),
            workspace_config=_WORKSPACE_CONFIG,
        )
    mypy_path.write_bytes(original_mypy_bytes)
    evidence_path.write_bytes(original_evidence_bytes)

    ruff_path = output / "ruff.json"
    ruff = json.loads(ruff_path.read_text(encoding="utf-8"))
    ruff["result"]["target_paths"] = ["."]
    ruff_path.write_text(
        json.dumps(ruff, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    resealed_ruff = build_local_preflight_evidence(
        plan_sha256=plan["closed_record_sha256"],
        completed_at_utc=old_evidence["completed_at_utc"],
        check_evidence_sha256={
            check_id: hashlib.sha256(
                (output / f"{check_id}.json").read_bytes()
            ).hexdigest()
            for check_id in freeze._LOCAL_CHECK_IDS
        },
    )
    evidence_path.write_text(
        canonical_gpu_qualification_json(resealed_ruff) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ruff.*result differs"):
        freeze.validate_gpu_qualification_local_preflight_bundle(
            evidence_path,
            plan_record=plan,
            submit_payloads=_bound_submit_payloads(inputs),
            workspace_config=_WORKSPACE_CONFIG,
        )


def test_preflight_bundle_rejects_resealed_environment_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan, plan_path = _plan(tmp_path)
    inputs = _preflight_inputs(tmp_path, plan_path)
    _patch_in_process_checks(monkeypatch)
    output = tmp_path / "preflight"
    freeze._run_gpu_qualification_local_preflight(
        inputs,
        output,
        command_runner=_fake_runner,
        now=_clock(),
    )
    ruff_path = output / "ruff.json"
    ruff = json.loads(ruff_path.read_text(encoding="utf-8"))
    ruff["environment"]["RUFF_CONFIG"] = "/tmp/attacker.toml"
    ruff_path.write_bytes(freeze._canonical_json_bytes(ruff, pretty=False))
    old_evidence = json.loads(
        (output / "local-preflight-evidence.json").read_text(encoding="utf-8")
    )
    resealed = build_local_preflight_evidence(
        plan_sha256=plan["closed_record_sha256"],
        completed_at_utc=old_evidence["completed_at_utc"],
        check_evidence_sha256={
            check_id: hashlib.sha256(
                (output / f"{check_id}.json").read_bytes()
            ).hexdigest()
            for check_id in freeze._LOCAL_CHECK_IDS
        },
    )
    (output / "local-preflight-evidence.json").write_text(
        canonical_gpu_qualification_json(resealed) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ruff.*environment differs"):
        freeze.validate_gpu_qualification_local_preflight_bundle(
            output / "local-preflight-evidence.json",
            plan_record=plan,
            submit_payloads=_bound_submit_payloads(inputs),
            workspace_config=_WORKSPACE_CONFIG,
        )


def test_preflight_bundle_live_replay_rejects_a_nonzero_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan, plan_path = _plan(tmp_path)
    inputs = _preflight_inputs(tmp_path, plan_path)
    _patch_in_process_checks(monkeypatch)
    output = tmp_path / "preflight"
    freeze._run_gpu_qualification_local_preflight(
        inputs,
        output,
        command_runner=_fake_runner,
        now=_clock(),
    )
    original_hashes = freeze._preflight_bundle_file_hashes(
        output / "local-preflight-evidence.json"
    )

    def fail_live_mypy(
        command: list[str] | tuple[str, ...],
        cwd: Path,
        environment: dict[str, str],
    ) -> _Completed:
        completed = _fake_runner(command, cwd, environment)
        if "--version" not in command and Path(command[0]).name == "mypy":
            return _Completed(1, b"", b"live type failure\n")
        return completed

    monkeypatch.setattr(freeze, "_run_command", fail_live_mypy)
    with pytest.raises(RuntimeError, match="mypy.*returned 1"):
        freeze.validate_gpu_qualification_local_preflight_bundle(
            output / "local-preflight-evidence.json",
            plan_record=plan,
            submit_payloads=_bound_submit_payloads(inputs),
            workspace_config=_WORKSPACE_CONFIG,
        )
    assert freeze._preflight_bundle_file_hashes(
        output / "local-preflight-evidence.json"
    ) == original_hashes


def test_preflight_rejects_alternate_consistent_submit_payload_closure_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan, plan_path = _plan(tmp_path)
    inputs = _preflight_inputs(tmp_path, plan_path)

    def rendered(label: str):
        uris = {
            role: f"dbfs:/Volumes/catalog/schema/volume/{label}/{role}"
            for role in freeze.GPU_QUALIFICATION_ARTIFACT_KEYS
        }
        payloads = freeze.render_gpu_qualification_submit_payloads(
            plan,
            single_user_name=_SINGLE_USER_NAME,
            runner_uri=uris["runner_sha256"],
            package_wheel_uri=uris["package_wheel_sha256"],
            patched_vllm_wheel_uri=uris["patched_vllm_wheel_sha256"],
            artifact_uris=uris,
            output_root=f"dbfs:/Volumes/catalog/schema/volume/results-{label}",
        )
        return uris, payloads

    uris_a, payloads_a = rendered("preflight-a")
    _uris_b, payloads_b = rendered("submitted-b")
    inputs.artifact_uris_json.write_bytes(
        freeze._canonical_json_bytes(
            {
                "artifact_uris": uris_a,
                "output_root": (
                    "dbfs:/Volumes/catalog/schema/volume/results-preflight-a"
                ),
                "plan_sha256": plan["closed_record_sha256"],
            },
            pretty=False,
        )
    )
    inputs.submit_payloads_json.write_bytes(
        freeze._canonical_json_bytes(list(payloads_a), pretty=False)
    )
    _patch_in_process_checks(monkeypatch)
    output = tmp_path / "preflight"
    freeze._run_gpu_qualification_local_preflight(
        inputs,
        output,
        command_runner=_fake_runner,
        now=_clock(),
    )

    def replay_must_not_start(*_args, **_kwargs):
        raise AssertionError("live check replay started before payload binding")

    monkeypatch.setattr(freeze, "_run_command", replay_must_not_start)
    with pytest.raises(ValueError, match="submitted payload closure differs"):
        freeze.validate_gpu_qualification_local_preflight_bundle(
            output / "local-preflight-evidence.json",
            plan_record=plan,
            submit_payloads=payloads_b,
            workspace_config=_WORKSPACE_CONFIG,
        )


def test_preflight_and_git_subprocesses_ignore_hostile_ambient_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "PYTEST_ADDOPTS",
        "--ignore=tests/test_publication_freeze.py",
    )
    monkeypatch.setenv("PYTEST_PLUGINS", "attacker_plugin")
    monkeypatch.setenv("MYPYPATH", str(tmp_path / "attacker-mypy"))
    monkeypatch.setenv("MYPY_CACHE_DIR", str(tmp_path / "attacker-mypy-cache"))
    monkeypatch.setenv("RUFF_CONFIG", str(tmp_path / "attacker-ruff.toml"))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker-git-dir"))
    monkeypatch.setenv("PIP_INDEX_URL", "https://attacker.invalid/simple")
    environment = freeze._preflight_environment(_REPOSITORY_ROOT)

    assert environment["PYTEST_ADDOPTS"] == ""
    assert environment["PYTEST_PLUGINS"] == ""
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["MYPYPATH"] == ""
    assert environment["RUFF_NO_CACHE"] == "true"
    assert "MYPY_CACHE_DIR" not in environment
    assert "RUFF_CONFIG" not in environment
    assert "GIT_DIR" not in freeze._git_environment()
    assert freeze._build_environment(1_777_777_777)["PIP_INDEX_URL"] == (
        "https://pypi.org/simple"
    )
    assert freeze._git(_REPOSITORY_ROOT, "rev-parse", "--show-toplevel") == str(
        _REPOSITORY_ROOT
    )

    collected = freeze._run_command(
        (
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-c",
            "pyproject.toml",
            "-p",
            "no:cacheprovider",
            "tests/test_publication_freeze.py",
        ),
        _REPOSITORY_ROOT,
        environment,
    )
    assert collected.returncode == 0
    assert b"test_preflight_and_git_subprocesses_ignore_hostile" in collected.stdout

    package_root = tmp_path / "package-root"
    package_root.mkdir()
    (package_root / "bytecode_probe.py").write_text(
        "PROBE = 'imported'\n",
        encoding="utf-8",
    )
    imported = freeze._run_command(
        (
            sys.executable,
            "-c",
            "import bytecode_probe",
        ),
        package_root,
        environment,
    )
    assert imported.returncode == 0
    assert imported.stdout == b""
    assert imported.stderr == b""
    assert not (package_root / "__pycache__").exists()

    with pytest.raises(RuntimeError, match="exact no-skip publication suite"):
        freeze._require_exact_pytest_completion(
            (
                f".\n{freeze.PUBLICATION_FREEZE_EXPECTED_TEST_COUNT - 1} "
                "passed, 1 skipped in 1.00s\n"
            ).encode("utf-8")
        )


def test_preflight_runner_does_not_seal_after_nonzero_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _plan_record, plan_path = _plan(tmp_path)
    inputs = _preflight_inputs(tmp_path, plan_path)
    _patch_in_process_checks(monkeypatch)
    monkeypatch.setattr(freeze, "_run_command", _fake_runner)

    def failing_runner(
        command: list[str] | tuple[str, ...],
        cwd: Path,
        environment: dict[str, str],
    ) -> _Completed:
        completed = _fake_runner(command, cwd, environment)
        if "--version" not in command and Path(command[0]).name == "mypy":
            return _Completed(2, b"", b"type failure\n")
        return completed

    output = tmp_path / "preflight"
    with pytest.raises(RuntimeError, match="mypy.*returned 2"):
        freeze._run_gpu_qualification_local_preflight(
            inputs,
            output,
            command_runner=failing_runner,
            now=_clock(),
        )
    assert not (output / "local-preflight-evidence.json").exists()
    assert json.loads((output / "mypy.json").read_text(encoding="utf-8"))[
        "status"
    ] == "failed"


def test_live_source_check_requires_clean_tree_rebuild_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan, plan_path = _plan(tmp_path)
    inputs = _preflight_inputs(tmp_path, plan_path)
    inputs.source_closure_json.write_text("{}\n", encoding="utf-8")

    def reject_handcrafted(*_args, **_kwargs):
        raise ValueError("clean-tree deterministic rebuild differs")

    monkeypatch.setattr(
        freeze,
        "validate_publication_source_closure_record",
        reject_handcrafted,
    )
    with pytest.raises(ValueError, match="clean-tree deterministic rebuild"):
        freeze._check_source_runner_inputs(
            inputs,
            plan,
            freeze.pins_from_plan_record(plan),
            verify_rebuild=True,
        )


def test_preflight_bundle_rejects_missing_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan, plan_path = _plan(tmp_path)
    inputs = _preflight_inputs(tmp_path, plan_path)
    _patch_in_process_checks(monkeypatch)
    output = tmp_path / "preflight"
    freeze._run_gpu_qualification_local_preflight(
        inputs,
        output,
        command_runner=_fake_runner,
        now=_clock(),
    )
    (output / "ruff.json").unlink()

    with pytest.raises(ValueError, match="exact check coverage"):
        freeze.validate_gpu_qualification_local_preflight_bundle(
            output / "local-preflight-evidence.json",
            plan_record=plan,
            submit_payloads=_bound_submit_payloads(inputs),
            workspace_config=_WORKSPACE_CONFIG,
        )


def test_live_workspace_uses_authenticated_state_and_remote_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan, plan_path = _plan(tmp_path)
    inputs = _preflight_inputs(tmp_path, plan_path)
    pins = freeze.pins_from_plan_record(plan)
    uris = {
        role: f"dbfs:/Volumes/catalog/schema/volume/{role}"
        for role in freeze.GPU_QUALIFICATION_ARTIFACT_KEYS
    }
    output_root = "dbfs:/Volumes/catalog/schema/volume/qualification-results"
    inputs.artifact_uris_json.write_text(
        canonical_gpu_qualification_json(
            {
                "artifact_uris": uris,
                "output_root": output_root,
                "plan_sha256": plan["closed_record_sha256"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    expected = {
        uris["cachet_source_tree_sha256"]: pins.cachet_source_tree_sha256,
        uris["package_wheel_sha256"]: pins.package_wheel_sha256,
        uris["patched_vllm_wheel_sha256"]: pins.patched_vllm_wheel_sha256,
        uris["runner_sha256"]: pins.runner_sha256,
        uris["runtime_lock_sha256"]: pins.runtime_lock_sha256,
    }
    monkeypatch.setattr(
        databricks_runs,
        "list_active_databricks_runs",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        databricks_runs,
        "list_databricks_node_types",
        lambda *_args, **_kwargs: tuple(
            {"node_type_id": value}
            for value in ("g5.8xlarge", "g6.8xlarge", "g6e.4xlarge")
        ),
    )
    listed_directories: list[str] = []

    def list_directory(_config, uri: str):
        listed_directories.append(uri)
        return ()

    monkeypatch.setattr(freeze, "list_databricks_volume_directory", list_directory)
    monkeypatch.setattr(
        freeze,
        "_require_remote_tree_matches_local",
        lambda *_args, **_kwargs: None,
    )

    def stream(_config, uri: str, *, max_bytes: int):
        assert max_bytes == 1_073_741_824
        return {
            "dbfs_uri": uri,
            "file_sha256": expected[uri],
            "size_bytes": 1,
        }

    monkeypatch.setattr(
        databricks_runs,
        "stream_databricks_volume_file_sha256",
        stream,
    )
    freeze._validate_live_workspace_and_remote_artifacts(
        _WORKSPACE_CONFIG,
        inputs=inputs,
        plan=plan,
        single_user_name=_SINGLE_USER_NAME,
        require_fresh_workspace=True,
    )
    assert listed_directories == [
        "dbfs:/Volumes/catalog/schema/volume"
    ]

    copied_local_digest = hashlib.sha256(inputs.patched_vllm_wheel.read_bytes()).hexdigest()

    def forged_stream(_config, uri: str, *, max_bytes: int):
        record = stream(_config, uri, max_bytes=max_bytes)
        if uri == uris["patched_vllm_wheel_sha256"]:
            record["file_sha256"] = copied_local_digest
        return record

    monkeypatch.setattr(
        databricks_runs,
        "stream_databricks_volume_file_sha256",
        forged_stream,
    )
    with pytest.raises(ValueError, match="patched_vllm_wheel_sha256"):
        freeze._validate_live_workspace_and_remote_artifacts(
            _WORKSPACE_CONFIG,
            inputs=inputs,
            plan=plan,
            single_user_name=_SINGLE_USER_NAME,
            require_fresh_workspace=True,
        )


def test_preflight_rejects_a_caller_selected_input_bundle_pin(tmp_path: Path):
    plan, _plan_path = _plan(tmp_path)
    pins = freeze.pins_from_plan_record(plan)
    forged = GPUQualificationArtifactPins(
        runtime_lock_sha256=pins.runtime_lock_sha256,
        patched_vllm_wheel_sha256=pins.patched_vllm_wheel_sha256,
        package_wheel_sha256=pins.package_wheel_sha256,
        cachet_source_tree_sha256=pins.cachet_source_tree_sha256,
        runner_sha256=pins.runner_sha256,
        input_bundle_sha256=_digest("caller-selected"),
    )

    with pytest.raises(ValueError, match="input-bundle pin differs"):
        freeze._require_fixed_publication_artifact_pins(forged)


def test_live_workspace_rejects_active_run_and_existing_output_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan, plan_path = _plan(tmp_path)
    inputs = _preflight_inputs(tmp_path, plan_path)
    uris = {
        role: f"dbfs:/Volumes/catalog/schema/volume/{role}"
        for role in freeze.GPU_QUALIFICATION_ARTIFACT_KEYS
    }
    inputs.artifact_uris_json.write_text(
        canonical_gpu_qualification_json(
            {
                "artifact_uris": uris,
                "output_root": (
                    "dbfs:/Volumes/catalog/schema/volume/qualification-results"
                ),
                "plan_sha256": plan["closed_record_sha256"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        databricks_runs,
        "list_active_databricks_runs",
        lambda *_args, **_kwargs: ({"life_cycle_state": "RUNNING", "run_id": 1},),
    )
    with pytest.raises(ValueError, match="zero direct active runs"):
        freeze._validate_live_workspace_and_remote_artifacts(
            _WORKSPACE_CONFIG,
            inputs=inputs,
            plan=plan,
            single_user_name=_SINGLE_USER_NAME,
            require_fresh_workspace=True,
        )

    monkeypatch.setattr(
        databricks_runs,
        "list_active_databricks_runs",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        databricks_runs,
        "list_databricks_node_types",
        lambda *_args, **_kwargs: tuple(
            {"node_type_id": value}
            for value in ("g5.8xlarge", "g6.8xlarge", "g6e.4xlarge")
        ),
    )
    monkeypatch.setattr(
        freeze,
        "list_databricks_volume_directory",
        lambda *_args, **_kwargs: (
            {
                "is_directory": True,
                "name": "qualification-results",
                "path": "/Volumes/catalog/schema/volume/qualification-results/",
            },
        ),
    )
    with pytest.raises(ValueError, match="output root already exists"):
        freeze._validate_live_workspace_and_remote_artifacts(
            _WORKSPACE_CONFIG,
            inputs=inputs,
            plan=plan,
            single_user_name=_SINGLE_USER_NAME,
            require_fresh_workspace=True,
        )


def test_authority_paths_reject_ancestor_and_tree_symlinks(tmp_path: Path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        freeze._write_exclusive(alias_parent / "evidence.json", b"{}\n", "evidence")
    assert not (real_parent / "evidence.json").exists()

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "regular.txt").write_text("ok\n", encoding="utf-8")
    (tree / "linked.txt").symlink_to(tree / "regular.txt")
    with pytest.raises(ValueError, match="tree contains a symlink"):
        freeze._regular_tree(tree)


def test_bundle_rejects_parent_alias_but_accepts_canonical_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(canonical_root, target_is_directory=True)
    plan, plan_path = _plan(canonical_root)
    inputs = _preflight_inputs(canonical_root, plan_path)
    _patch_in_process_checks(monkeypatch)
    monkeypatch.setattr(freeze, "_run_command", _fake_runner)
    output = canonical_root / "preflight"
    freeze._run_gpu_qualification_local_preflight(
        inputs,
        output,
        command_runner=_fake_runner,
        now=_clock(),
    )
    canonical = output / "local-preflight-evidence.json"
    freeze.validate_gpu_qualification_local_preflight_bundle(
        canonical,
        plan_record=plan,
        submit_payloads=_bound_submit_payloads(inputs),
        workspace_config=_WORKSPACE_CONFIG,
    )
    alias = alias_root / "preflight" / "local-preflight-evidence.json"
    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        freeze.validate_gpu_qualification_local_preflight_bundle(
            alias,
            plan_record=plan,
            submit_payloads=_bound_submit_payloads(inputs),
            workspace_config=_WORKSPACE_CONFIG,
        )


def test_cachet_publication_freeze_is_exact_module_alias():
    assert cachet_freeze is freeze
