from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import pytest

from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_SOURCE_WHEEL_SHA256,
    repack_flashinfer_0616_post3_wheel,
)
import document_kv_cache.runtime_artifact_closure as closure
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE,
    RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION,
    VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_BASE_LOCK_FILENAME,
    VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
    VLLM_RUNTIME_BASE_LOCK_VERSION_HASH_MAP_SHA256,
    VLLM_RUNTIME_BASE_LOCK_VERSION_MAP_SHA256,
    build_vllm_flashinfer_runtime_artifact_closure,
    derive_flashinfer_direct_base_lock,
    validate_flashinfer_direct_base_lock,
    validate_vllm_flashinfer_runtime_artifact_closure,
)


_ROOT = Path(__file__).parents[1]
_SOURCE_LOCK = (
    _ROOT
    / "src"
    / "document_kv_cache"
    / "runtime_locks"
    / closure.VLLM_RUNTIME_LOCK_FILENAME
)
_PRISTINE_FLASHINFER = (
    _ROOT
    / "databricks-runs"
    / "_campaign-inputs"
    / "flashinfer-python-0.6.16.post3"
    / "sha256"
    / FLASHINFER_SOURCE_WHEEL_SHA256
    / "flashinfer_python-0.6.16.post3-py3-none-any.whl"
)
_VLLM_ROOT = (
    _ROOT
    / "databricks-runs"
    / "_campaign-inputs"
    / "vllm-0.27.1"
    / "sha256"
    / closure.VLLM_PATCHED_WHEEL_SHA256
    / "artifacts"
    / "patched"
)
_VLLM_WHEEL = _VLLM_ROOT / closure.VLLM_PATCHED_WHEEL_FILENAME
_VLLM_MANIFEST = _VLLM_ROOT / closure.VLLM_PATCHED_MANIFEST_FILENAME


@dataclass(frozen=True)
class _Artifacts:
    base_lock: Path
    patched_flashinfer_wheel: Path
    flashinfer_manifest: Path
    first_closure: Path
    second_closure: Path


def _require_local_artifacts() -> None:
    missing = [
        path
        for path in (_SOURCE_LOCK, _PRISTINE_FLASHINFER, _VLLM_WHEEL, _VLLM_MANIFEST)
        if not path.is_file()
    ]
    if missing:
        pytest.skip(f"reviewed local artifacts are absent: {missing!r}")


@pytest.fixture(scope="module")
def reviewed_artifacts(tmp_path_factory: pytest.TempPathFactory) -> _Artifacts:
    _require_local_artifacts()
    root = tmp_path_factory.mktemp("runtime-artifact-closure")
    base_lock = derive_flashinfer_direct_base_lock(
        _SOURCE_LOCK,
        root / VLLM_RUNTIME_BASE_LOCK_FILENAME,
    )
    flashinfer = repack_flashinfer_0616_post3_wheel(
        _PRISTINE_FLASHINFER,
        root / "patched-flashinfer",
    )
    arguments = {
        "source_lock": _SOURCE_LOCK,
        "base_lock": base_lock,
        "vllm_wheel": _VLLM_WHEEL,
        "vllm_manifest": _VLLM_MANIFEST,
        "pristine_flashinfer_wheel": _PRISTINE_FLASHINFER,
        "patched_flashinfer_wheel": flashinfer.wheel_path,
        "flashinfer_manifest": flashinfer.manifest_path,
    }
    first = build_vllm_flashinfer_runtime_artifact_closure(
        **arguments,
        output_dir=root / "closure-a",
    )
    second = build_vllm_flashinfer_runtime_artifact_closure(
        **arguments,
        output_dir=root / "closure-b",
    )
    return _Artifacts(
        base_lock=base_lock,
        patched_flashinfer_wheel=flashinfer.wheel_path,
        flashinfer_manifest=flashinfer.manifest_path,
        first_closure=first.manifest_path,
        second_closure=second.manifest_path,
    )


def _closure_arguments(artifacts: _Artifacts) -> dict[str, Path]:
    return {
        "source_lock": _SOURCE_LOCK,
        "base_lock": artifacts.base_lock,
        "vllm_wheel": _VLLM_WHEEL,
        "vllm_manifest": _VLLM_MANIFEST,
        "pristine_flashinfer_wheel": _PRISTINE_FLASHINFER,
        "patched_flashinfer_wheel": artifacts.patched_flashinfer_wheel,
        "flashinfer_manifest": artifacts.flashinfer_manifest,
    }


def test_base_lock_is_exact_one_stanza_projection(tmp_path):
    _require_local_artifacts()
    target = derive_flashinfer_direct_base_lock(
        _SOURCE_LOCK,
        tmp_path / VLLM_RUNTIME_BASE_LOCK_FILENAME,
    )
    observed = validate_flashinfer_direct_base_lock(_SOURCE_LOCK, target)

    assert observed == {
        "bytes": 376_326,
        "distribution_count": VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
        "filename": VLLM_RUNTIME_BASE_LOCK_FILENAME,
        "hash_count": VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
        "removed_distribution": "flashinfer-python",
        "removed_stanza_sha256": sha256(closure._FLASHINFER_LOCK_STANZA).hexdigest(),
        "sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "version_hash_map_sha256": VLLM_RUNTIME_BASE_LOCK_VERSION_HASH_MAP_SHA256,
        "version_map_sha256": VLLM_RUNTIME_BASE_LOCK_VERSION_MAP_SHA256,
    }
    source = _SOURCE_LOCK.read_bytes()
    assert target.read_bytes() == source.replace(closure._FLASHINFER_LOCK_STANZA, b"", 1)
    assert target.read_bytes().count(b"flashinfer-python==") == 0


def test_base_lock_rejects_source_stanza_or_target_tamper(tmp_path):
    source = _SOURCE_LOCK.read_bytes()
    with pytest.raises(ValueError, match="source-lock identity differs"):
        closure._derived_base_lock_bytes(source + b"\n")
    with pytest.raises(ValueError, match="source-lock identity differs"):
        closure._derived_base_lock_bytes(
            source.replace(closure._FLASHINFER_LOCK_STANZA, b"", 1)
        )

    target = derive_flashinfer_direct_base_lock(
        _SOURCE_LOCK,
        tmp_path / VLLM_RUNTIME_BASE_LOCK_FILENAME,
    )
    target.write_bytes(target.read_bytes() + b"# tamper\n")
    with pytest.raises(ValueError, match="exceeds exact size"):
        validate_flashinfer_direct_base_lock(_SOURCE_LOCK, target)


def test_base_lock_failure_does_not_publish(tmp_path):
    bad_source = tmp_path / closure.VLLM_RUNTIME_LOCK_FILENAME
    bad_source.write_bytes(_SOURCE_LOCK.read_bytes() + b"tamper")
    target = tmp_path / VLLM_RUNTIME_BASE_LOCK_FILENAME
    with pytest.raises(ValueError, match="exceeds exact size"):
        derive_flashinfer_direct_base_lock(bad_source, target)
    assert not target.exists()


def test_composite_closure_is_deterministic_sealed_and_pending(reviewed_artifacts):
    assert reviewed_artifacts.first_closure.read_bytes() == (
        reviewed_artifacts.second_closure.read_bytes()
    )
    record = json.loads(reviewed_artifacts.first_closure.read_text())
    observed = validate_vllm_flashinfer_runtime_artifact_closure(
        **_closure_arguments(reviewed_artifacts),
        closure_manifest=reviewed_artifacts.first_closure,
    )

    assert record["record_type"] == RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE
    assert record["schema_version"] == RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION
    assert record["closed_record_sha256"] == observed["closed_record_sha256"]
    assert record["install_contract"]["distribution_counts"] == {
        "base_lock": 195,
        "separately_allowed_cachet": 1,
        "with_flashinfer": 196,
        "with_vllm": 197,
    }
    assert [item["artifact"] for item in record["install_contract"]["order"]] == [
        "base_lock",
        "vllm",
        "flashinfer",
        "cachet-kv",
    ]
    assert record["live_runtime_verification"]["state"] == "pending_not_executed"
    assert record["live_runtime_verification"]["passed"] == []
    assert record["live_runtime_verification"]["success_credit"] is False


def test_composite_closure_tamper_is_rejected(reviewed_artifacts, tmp_path):
    record = json.loads(reviewed_artifacts.first_closure.read_text())
    record["install_contract"]["distribution_counts"]["with_vllm"] = 198
    tampered = tmp_path / closure.RUNTIME_ARTIFACT_CLOSURE_FILENAME
    tampered.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="size, SHA-256, or stable stat differs"):
        validate_vllm_flashinfer_runtime_artifact_closure(
            **_closure_arguments(reviewed_artifacts),
            closure_manifest=tampered,
        )


def test_composite_rejects_vllm_manifest_tamper_before_output(
    reviewed_artifacts,
    tmp_path,
):
    bad_root = tmp_path / "bad-vllm"
    bad_root.mkdir()
    bad_manifest = bad_root / closure.VLLM_PATCHED_MANIFEST_FILENAME
    bad_manifest.write_bytes(_VLLM_MANIFEST.read_bytes() + b" ")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="exceeds exact size"):
        build_vllm_flashinfer_runtime_artifact_closure(
            **{
                **_closure_arguments(reviewed_artifacts),
                "vllm_manifest": bad_manifest,
            },
            output_dir=output,
        )
    assert not output.exists()


def test_resealed_live_success_claim_is_rejected(reviewed_artifacts):
    record = json.loads(reviewed_artifacts.first_closure.read_text())
    record["live_runtime_verification"]["state"] = "passed"
    record["live_runtime_verification"]["passed"] = ["pip_check"]
    unsigned = dict(record)
    unsigned.pop("closed_record_sha256")
    record["closed_record_sha256"] = closure._canonical_json_sha256(unsigned)
    with pytest.raises(ValueError, match="may not claim live verification"):
        closure._validate_runtime_closure_record(record)
