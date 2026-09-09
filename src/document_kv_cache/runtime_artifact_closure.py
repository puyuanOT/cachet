"""Seal the local vLLM plus patched-FlashInfer runtime artifact closure.

This stage owns only local artifact authority.  It derives the 195-package
base lock from the reviewed 196-package lock and binds that lock to the
unchanged patched vLLM wheel and the separately reviewed FlashInfer wheel.
No installer, qualification plan, or cloud transport is invoked here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping
from zipfile import ZipFile

from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PACKAGE_VERSION,
    FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256,
    FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
    FLASHINFER_PATCHED_MANIFEST_FILENAME,
    FLASHINFER_PATCHED_MANIFEST_SIZE,
    FLASHINFER_PATCHED_RECORD_SHA256,
    FLASHINFER_PATCHED_WHEEL_FILENAME,
    FLASHINFER_PATCHED_WHEEL_SHA256,
    FLASHINFER_PATCHED_WHEEL_SIZE,
    FLASHINFER_SOURCE_WHEEL_FILENAME,
    FLASHINFER_SOURCE_WHEEL_SHA256,
    FLASHINFER_SOURCE_WHEEL_SIZE,
    FLASHINFER_TARGET_MEMBER,
    FLASHINFER_TARGET_PATCHED_SHA256,
    validate_patched_flashinfer_wheel,
)
VLLM_PACKAGE_VERSION = "0.27.1+cu129"
VLLM_RUNTIME_LOCK_FILENAME = "vllm-0.27.1-cu129-py311-manylinux_2_35.lock"
VLLM_RUNTIME_LOCK_SHA256 = (
    "71c2c3e344ebdf1d8996adf2127a519328b6bad78a4eb7134c73e2a3f6115c44"
)
VLLM_RUNTIME_SOURCE_LOCK_SIZE = 376_593
VLLM_RUNTIME_SOURCE_LOCK_DISTRIBUTION_COUNT = 196
VLLM_RUNTIME_SOURCE_LOCK_HASH_COUNT = 4_138
VLLM_RUNTIME_BASE_LOCK_FILENAME = (
    "vllm-0.27.1-cu129-py311-manylinux_2_35-flashinfer-direct.lock"
)
VLLM_RUNTIME_BASE_LOCK_SHA256 = (
    "c4fc0e055f0838ff397012f52bd4c4f0d22426db8a5fc8faf01689510e258903"
)
VLLM_RUNTIME_BASE_LOCK_SIZE = 376_326
VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT = 195
VLLM_RUNTIME_BASE_LOCK_HASH_COUNT = 4_137
VLLM_RUNTIME_BASE_LOCK_VERSION_MAP_SHA256 = (
    "57ee2eb51449d71bd1e151fab5b196f86be29ca378642ee4789cd6561413f1b8"
)
VLLM_RUNTIME_BASE_LOCK_VERSION_HASH_MAP_SHA256 = (
    "28a4da095d20a10c9db8bc48a4987cf8d4a2bf4b2ff069a4b0121519e87fe4dd"
)
_FLASHINFER_LOCK_STANZA = (
    b"flashinfer-python==0.6.16.post3 \\\n"
    b"    --hash=sha256:caf686b9b079abe1c9d65ab505698bd325e8072de40afd822f2c74f2ac3bc601\n"
    b"    # via\n"
    b"    #   -r src/document_kv_cache/runtime_locks/"
    b"vllm-0.27.1-cu129-py311-manylinux_2_35.in\n"
    b"    #   vllm\n"
    b"    # from https://flashinfer.ai/whl/\n"
)

VLLM_PATCHED_WHEEL_FILENAME = (
    "vllm-0.27.1+cu129-1cachete5m265120c48a9352b9e-"
    "cp38-abi3-manylinux_2_28_x86_64.whl"
)
VLLM_PATCHED_WHEEL_SHA256 = (
    "65120c48a9352b9eb65bab7a67090558d27af985ad366e469d3b87751073cff4"
)
VLLM_PATCHED_WHEEL_SIZE = 537_751_595
VLLM_PATCHED_MANIFEST_FILENAME = (
    "vllm-0.27.1+cu129-1cachete5m265120c48a9352b9e-"
    "cp38-abi3-manylinux_2_28_x86_64.manifest.json"
)
VLLM_PATCHED_MANIFEST_SHA256 = (
    "14611e163e720f0fdeae6ef2704cecd9202eef6adc6336f892afd94a96726ef6"
)
VLLM_PATCHED_MANIFEST_SIZE = 2_615
_VLLM_PATCHED_MEMBER_CLOSURE: tuple[Mapping[str, str], ...] = (
    {
        "id": "vllm-qwen-attention-disable-e5m2-query-quant",
        "patched_sha256": (
            "5735acfb390cf344caeec950c2f286344bcd84721ce287e0a56701f2a18bc839"
        ),
        "reason": (
            "vLLM 0.27.1 constructs an E4M3-only query quantizer for fp8_e5m2; "
            "E5M2 queries must stay in the model compute dtype."
        ),
        "relative_path": "vllm/model_executor/layers/attention/attention.py",
        "source_sha256": (
            "dae6d1f09448adc1d67c776089e77e8b75378332166844a234cd8fa45c18195e"
        ),
    },
    {
        "id": "vllm-triton-reshape-cache-e5m2-closure",
        "patched_sha256": (
            "0682ca7bc56edf7cea5419188a81c78510b54192471472b160aa447ac0ceeb08"
        ),
        "reason": (
            "vLLM 0.27.1 otherwise views fp8_e5m2 pages through the platform "
            "E4M3 dtype and rejects the SM80-SM88 E5M2 software path."
        ),
        "relative_path": (
            "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py"
        ),
        "source_sha256": (
            "6cac51475b8c656992a21b2d150acd3a16a95a7ab7d49aab151a3ef13c24b80d"
        ),
    },
    {
        "id": "vllm-triton-attention-e5m2-closure",
        "patched_sha256": (
            "4dae0ff6c4ee8f11c1f195151a11673d595d457c413032e7bae7550913f94390"
        ),
        "reason": (
            "vLLM 0.27.1 otherwise binds Triton KV views to E4M3 and rejects "
            "fp8_e5m2 on A10G despite the targeted SM80+ E5M2 path."
        ),
        "relative_path": "vllm/v1/attention/backends/triton_attn.py",
        "source_sha256": (
            "20b4dd5f8c15cd2d6f9598268368f5fc572f2c980eb75327c5992100c49ff3ed"
        ),
    },
)

RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE = (
    "document_kv.vllm_flashinfer_runtime_artifact_closure.v1"
)
RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION = 1
RUNTIME_ARTIFACT_CLOSURE_FILENAME = (
    "vllm-0.27.1-flashinfer-0.6.16.post3-runtime-closure.json"
)

RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256 = (
    "b2cc4f90bf3e5e47ca23bc7b2117725faa9b114f0d1d803af6c89ae18ca05aaf"
)
RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256 = (
    "c13c25a4e116f15db31e2efdbaebdd2d76418c5e4eb2f72fb2af3d8b8090e7df"
)
RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE = 6_634

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ]+?)(?: \\)?$")
_HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})(?: \\)?$")


@dataclass(frozen=True, slots=True)
class RuntimeArtifactClosure:
    """One sealed local runtime-closure manifest."""

    manifest_path: Path
    closed_record_sha256: str
    file_sha256: str
    file_size: int


def derive_flashinfer_direct_base_lock(
    source_lock: str | Path,
    output_path: str | Path,
) -> Path:
    """Derive and exclusively publish the exact one-stanza lock projection."""

    source = Path(source_lock)
    target = Path(output_path)
    source_bytes = _read_exact_small_file(
        source,
        expected_name=VLLM_RUNTIME_LOCK_FILENAME,
        expected_sha256=VLLM_RUNTIME_LOCK_SHA256,
        expected_size=VLLM_RUNTIME_SOURCE_LOCK_SIZE,
        label="reviewed 196-distribution runtime lock",
    )
    target_bytes = _derived_base_lock_bytes(source_bytes)
    if target.name != VLLM_RUNTIME_BASE_LOCK_FILENAME:
        raise ValueError("derived runtime base-lock filename differs")
    _require_private_parent(target)
    if target.exists() or target.is_symlink():
        observed = _read_exact_small_file(
            target,
            expected_name=VLLM_RUNTIME_BASE_LOCK_FILENAME,
            expected_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
            expected_size=VLLM_RUNTIME_BASE_LOCK_SIZE,
            label="derived 195-distribution runtime lock",
        )
        if observed != target_bytes:
            raise FileExistsError("derived runtime base-lock path differs")
        return target
    _exclusive_write(target, target_bytes)
    return target


def validate_flashinfer_direct_base_lock(
    source_lock: str | Path,
    base_lock: str | Path,
) -> Mapping[str, Any]:
    """Prove the base lock is exactly the reviewed source minus one stanza."""

    source_bytes = _read_exact_small_file(
        Path(source_lock),
        expected_name=VLLM_RUNTIME_LOCK_FILENAME,
        expected_sha256=VLLM_RUNTIME_LOCK_SHA256,
        expected_size=VLLM_RUNTIME_SOURCE_LOCK_SIZE,
        label="reviewed 196-distribution runtime lock",
    )
    target_bytes = _read_exact_small_file(
        Path(base_lock),
        expected_name=VLLM_RUNTIME_BASE_LOCK_FILENAME,
        expected_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        expected_size=VLLM_RUNTIME_BASE_LOCK_SIZE,
        label="derived 195-distribution runtime lock",
    )
    if target_bytes != _derived_base_lock_bytes(source_bytes):
        raise ValueError("derived runtime base-lock byte projection differs")
    versions, hashes = _lock_projection(target_bytes)
    return {
        "filename": VLLM_RUNTIME_BASE_LOCK_FILENAME,
        "sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "bytes": VLLM_RUNTIME_BASE_LOCK_SIZE,
        "distribution_count": len(versions),
        "hash_count": sum(len(value) for value in hashes.values()),
        "version_map_sha256": _canonical_json_sha256(versions),
        "version_hash_map_sha256": _canonical_json_sha256(
            {
                name: {"hashes": sorted(hashes[name]), "version": versions[name]}
                for name in sorted(versions)
            }
        ),
        "removed_distribution": "flashinfer-python",
        "removed_stanza_sha256": sha256(_FLASHINFER_LOCK_STANZA).hexdigest(),
    }


def build_vllm_flashinfer_runtime_artifact_closure(
    *,
    source_lock: str | Path,
    base_lock: str | Path,
    vllm_wheel: str | Path,
    vllm_manifest: str | Path,
    pristine_flashinfer_wheel: str | Path,
    patched_flashinfer_wheel: str | Path,
    flashinfer_manifest: str | Path,
    output_dir: str | Path,
) -> RuntimeArtifactClosure:
    """Validate every local artifact and publish one sealed closure manifest."""

    record = _build_runtime_closure_record(
        source_lock=Path(source_lock),
        base_lock=Path(base_lock),
        vllm_wheel=Path(vllm_wheel),
        vllm_manifest=Path(vllm_manifest),
        pristine_flashinfer_wheel=Path(pristine_flashinfer_wheel),
        patched_flashinfer_wheel=Path(patched_flashinfer_wheel),
        flashinfer_manifest=Path(flashinfer_manifest),
    )
    content = _canonical_record_file_bytes(record)
    output_root = Path(output_dir)
    _require_or_create_private_directory(output_root)
    target = output_root / RUNTIME_ARTIFACT_CLOSURE_FILENAME
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
            raise FileExistsError("runtime-closure target exists with different bytes")
    else:
        _exclusive_write(target, content)
    validated = validate_vllm_flashinfer_runtime_artifact_closure(
        source_lock=source_lock,
        base_lock=base_lock,
        vllm_wheel=vllm_wheel,
        vllm_manifest=vllm_manifest,
        pristine_flashinfer_wheel=pristine_flashinfer_wheel,
        patched_flashinfer_wheel=patched_flashinfer_wheel,
        flashinfer_manifest=flashinfer_manifest,
        closure_manifest=target,
    )
    return RuntimeArtifactClosure(
        manifest_path=target,
        closed_record_sha256=_required_sha256(
            validated["closed_record_sha256"],
            "runtime closure closed digest",
        ),
        file_sha256=_required_sha256(
            validated["file_sha256"],
            "runtime closure file digest",
        ),
        file_size=_required_int(validated["file_size"], "runtime closure file size"),
    )


def validate_vllm_flashinfer_runtime_artifact_closure(
    *,
    source_lock: str | Path,
    base_lock: str | Path,
    vllm_wheel: str | Path,
    vllm_manifest: str | Path,
    pristine_flashinfer_wheel: str | Path,
    patched_flashinfer_wheel: str | Path,
    flashinfer_manifest: str | Path,
    closure_manifest: str | Path,
) -> Mapping[str, Any]:
    """Validate the sealed closure against every immutable local artifact."""

    expected = _build_runtime_closure_record(
        source_lock=Path(source_lock),
        base_lock=Path(base_lock),
        vllm_wheel=Path(vllm_wheel),
        vllm_manifest=Path(vllm_manifest),
        pristine_flashinfer_wheel=Path(pristine_flashinfer_wheel),
        patched_flashinfer_wheel=Path(patched_flashinfer_wheel),
        flashinfer_manifest=Path(flashinfer_manifest),
    )
    expected_bytes = _canonical_record_file_bytes(expected)
    observed = _read_exact_small_file(
        Path(closure_manifest),
        expected_name=RUNTIME_ARTIFACT_CLOSURE_FILENAME,
        expected_sha256=sha256(expected_bytes).hexdigest(),
        expected_size=len(expected_bytes),
        label="runtime artifact closure",
    )
    if observed != expected_bytes:
        raise ValueError("runtime artifact closure bytes differ")
    decoded = _canonical_record_from_bytes(observed)
    _validate_runtime_closure_record(decoded)
    closed = _required_sha256(
        decoded.get("closed_record_sha256"),
        "runtime closure closed_record_sha256",
    )
    file_digest = sha256(observed).hexdigest()
    _require_pinned_closure_output(closed, file_digest, len(observed))
    return {
        "closed_record_sha256": closed,
        "file_sha256": file_digest,
        "file_size": len(observed),
    }


def _derived_base_lock_bytes(source: bytes) -> bytes:
    if len(source) != VLLM_RUNTIME_SOURCE_LOCK_SIZE or sha256(source).hexdigest() != (
        VLLM_RUNTIME_LOCK_SHA256
    ):
        raise ValueError("reviewed runtime source-lock identity differs")
    if source.count(_FLASHINFER_LOCK_STANZA) != 1:
        raise ValueError("reviewed runtime source lock has no unique FlashInfer stanza")
    target = source.replace(_FLASHINFER_LOCK_STANZA, b"", 1)
    if len(target) != VLLM_RUNTIME_BASE_LOCK_SIZE or sha256(target).hexdigest() != (
        VLLM_RUNTIME_BASE_LOCK_SHA256
    ):
        raise RuntimeError("derived runtime base-lock identity differs")
    source_versions, source_hashes = _lock_projection(source)
    target_versions, target_hashes = _lock_projection(target)
    if (
        len(source_versions) != VLLM_RUNTIME_SOURCE_LOCK_DISTRIBUTION_COUNT
        or sum(len(value) for value in source_hashes.values())
        != VLLM_RUNTIME_SOURCE_LOCK_HASH_COUNT
        or len(target_versions) != VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT
        or sum(len(value) for value in target_hashes.values())
        != VLLM_RUNTIME_BASE_LOCK_HASH_COUNT
        or set(source_versions) - set(target_versions) != {"flashinfer-python"}
        or {name: source_versions[name] for name in target_versions} != target_versions
        or {name: source_hashes[name] for name in target_hashes} != target_hashes
        or _canonical_json_sha256(target_versions)
        != VLLM_RUNTIME_BASE_LOCK_VERSION_MAP_SHA256
    ):
        raise RuntimeError("derived runtime base-lock projection differs")
    target_closure = {
        name: {"hashes": sorted(target_hashes[name]), "version": target_versions[name]}
        for name in sorted(target_versions)
    }
    if _canonical_json_sha256(target_closure) != (
        VLLM_RUNTIME_BASE_LOCK_VERSION_HASH_MAP_SHA256
    ):
        raise RuntimeError("derived runtime base-lock hash closure differs")
    return target


def _lock_projection(raw: bytes) -> tuple[dict[str, str], dict[str, list[str]]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("runtime lock is not UTF-8") from exc
    versions: dict[str, str] = {}
    hashes: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        requirement = _REQUIREMENT_RE.fullmatch(line)
        if requirement is not None:
            name = re.sub(r"[-_.]+", "-", requirement.group(1)).lower()
            if name in versions:
                raise ValueError(f"runtime lock repeats distribution {name!r}")
            versions[name] = requirement.group(2)
            hashes[name] = []
            current = name
            continue
        digest = _HASH_RE.search(line.strip())
        if digest is not None:
            if current is None:
                raise ValueError("runtime lock hash has no distribution")
            hashes[current].append(digest.group(1))
    if not versions or any(not value for value in hashes.values()):
        raise ValueError("runtime lock distribution/hash closure is incomplete")
    if any(len(value) != len(set(value)) for value in hashes.values()):
        raise ValueError("runtime lock repeats a distribution hash")
    return dict(sorted(versions.items())), {
        name: sorted(hashes[name]) for name in sorted(hashes)
    }


def _build_runtime_closure_record(
    *,
    source_lock: Path,
    base_lock: Path,
    vllm_wheel: Path,
    vllm_manifest: Path,
    pristine_flashinfer_wheel: Path,
    patched_flashinfer_wheel: Path,
    flashinfer_manifest: Path,
) -> dict[str, Any]:
    lock = validate_flashinfer_direct_base_lock(source_lock, base_lock)
    vllm = _validate_vllm_artifacts(vllm_wheel, vllm_manifest)
    flashinfer = validate_patched_flashinfer_wheel(
        pristine_flashinfer_wheel,
        patched_flashinfer_wheel,
        flashinfer_manifest,
    )
    if (
        flashinfer.get("patched_wheel_sha256") != FLASHINFER_PATCHED_WHEEL_SHA256
        or flashinfer.get("patched_wheel_size") != FLASHINFER_PATCHED_WHEEL_SIZE
        or flashinfer.get("manifest_file_sha256")
        != FLASHINFER_PATCHED_MANIFEST_FILE_SHA256
        or flashinfer.get("manifest_closed_record_sha256")
        != FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
    ):
        raise RuntimeError("patched FlashInfer closure authority differs")
    record: dict[str, Any] = {
        "record_type": RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE,
        "schema_version": RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION,
        "closed_record_sha256": "",
        "target_runtime": {
            "python_implementation": "CPython",
            "python_version": "3.11",
            "operating_system": "linux",
            "machine": "x86_64",
            "libc": "glibc-2.35",
            "cuda": "12.9",
        },
        "artifacts": {
            "base_lock": dict(lock),
            "vllm": vllm,
            "flashinfer": {
                "distribution": "flashinfer-python",
                "version": FLASHINFER_PACKAGE_VERSION,
                "source_wheel_filename": FLASHINFER_SOURCE_WHEEL_FILENAME,
                "source_wheel_sha256": FLASHINFER_SOURCE_WHEEL_SHA256,
                "source_wheel_bytes": FLASHINFER_SOURCE_WHEEL_SIZE,
                "patched_wheel_filename": FLASHINFER_PATCHED_WHEEL_FILENAME,
                "patched_wheel_sha256": flashinfer["patched_wheel_sha256"],
                "patched_wheel_bytes": flashinfer["patched_wheel_size"],
                "patched_manifest_filename": FLASHINFER_PATCHED_MANIFEST_FILENAME,
                "patched_manifest_file_sha256": flashinfer["manifest_file_sha256"],
                "patched_manifest_file_bytes": FLASHINFER_PATCHED_MANIFEST_SIZE,
                "patched_manifest_closed_record_sha256": flashinfer[
                    "manifest_closed_record_sha256"
                ],
                "patched_member": {
                    "relative_path": FLASHINFER_TARGET_MEMBER,
                    "sha256": FLASHINFER_TARGET_PATCHED_SHA256,
                },
                "patched_record_sha256": FLASHINFER_PATCHED_RECORD_SHA256,
            },
        },
        "install_contract": {
            "order": [
                {
                    "step": 1,
                    "artifact": "base_lock",
                    "arguments": [
                        "--require-hashes",
                        "--only-binary",
                        ":all:",
                        "-r",
                        "{base_lock_path}",
                    ],
                },
                {
                    "step": 2,
                    "artifact": "vllm",
                    "arguments": [
                        "--no-deps",
                        "vllm @ {patched_vllm_wheel_uri}#sha256="
                        + VLLM_PATCHED_WHEEL_SHA256,
                    ],
                },
                {
                    "step": 3,
                    "artifact": "flashinfer",
                    "arguments": [
                        "--no-deps",
                        "flashinfer-python @ {patched_flashinfer_wheel_uri}#sha256="
                        + FLASHINFER_PATCHED_WHEEL_SHA256,
                    ],
                },
                {
                    "step": 4,
                    "artifact": "cachet-kv",
                    "arguments": [
                        "--no-deps",
                        "cachet-kv @ {package_wheel_uri}#sha256={package_wheel_sha256}",
                    ],
                    "binding": "qualification plan must supply the exact package pin",
                },
            ],
            "pep610": {
                "vllm": {
                    "distribution": "vllm",
                    "runtime_uri_binding": "patched_vllm_wheel_uri",
                    "archive_sha256": VLLM_PATCHED_WHEEL_SHA256,
                },
                "flashinfer-python": {
                    "distribution": "flashinfer-python",
                    "runtime_uri_binding": "patched_flashinfer_wheel_uri",
                    "archive_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
                },
                "require_normalized_url_equality": True,
                "require_archive_info_sha256": True,
            },
            "distribution_counts": {
                "base_lock": 195,
                "with_flashinfer": 196,
                "with_vllm": 197,
                "separately_allowed_cachet": 1,
            },
            "pip_check_required": True,
            "inherited_pip_environment_forbidden": True,
        },
        "live_runtime_verification": {
            "state": "pending_not_executed",
            "passed": [],
            "required": [
                "pip_check",
                "exact_196_dependency_version_equivalence",
                "vllm_pep610_url_and_sha256",
                "flashinfer_pep610_url_and_sha256",
                "flashinfer_fd_exchange_installed_member_sha256",
                "flashinfer.comm.fd_exchange CPython3.11 import",
            ],
            "success_credit": False,
        },
    }
    unsigned = dict(record)
    unsigned.pop("closed_record_sha256")
    record["closed_record_sha256"] = _canonical_json_sha256(unsigned)
    _validate_runtime_closure_record(record)
    return record


def _validate_vllm_artifacts(
    wheel_path: Path,
    manifest_path: Path,
) -> Mapping[str, Any]:
    manifest_bytes = _read_exact_small_file(
        manifest_path,
        expected_name=VLLM_PATCHED_MANIFEST_FILENAME,
        expected_sha256=VLLM_PATCHED_MANIFEST_SHA256,
        expected_size=VLLM_PATCHED_MANIFEST_SIZE,
        label="immutable patched vLLM manifest",
    )
    manifest = _canonical_record_from_bytes(manifest_bytes)
    expected_patch_closure = [
        {
            "id": _required_string(item.get("id"), "vLLM patch id"),
            "patched_sha256": _required_sha256(
                item.get("patched_sha256"),
                "vLLM patched member SHA",
            ),
            "reason": _required_string(item.get("reason"), "vLLM patch reason"),
            "relative_path": _required_string(
                item.get("relative_path"),
                "vLLM patch path",
            ),
            "source_sha256": _required_sha256(
                item.get("source_sha256"),
                "vLLM source member SHA",
            ),
        }
        for item in _VLLM_PATCHED_MEMBER_CLOSURE
    ]
    if (
        manifest.get("record_type") != "document_kv.vllm_patched_wheel_manifest.v2"
        or manifest.get("schema_version") != 2
        or manifest.get("package_version") != VLLM_PACKAGE_VERSION
        or manifest.get("patched_wheel_filename") != VLLM_PATCHED_WHEEL_FILENAME
        or manifest.get("patched_wheel_sha256") != VLLM_PATCHED_WHEEL_SHA256
        or manifest.get("patched_wheel_size") != VLLM_PATCHED_WHEEL_SIZE
        or manifest.get("patch_closure") != expected_patch_closure
    ):
        raise ValueError("immutable patched vLLM manifest closure differs")
    descriptor = _open_regular_nofollow(wheel_path, label="immutable patched vLLM wheel")
    try:
        before = os.fstat(descriptor)
        if wheel_path.name != VLLM_PATCHED_WHEEL_FILENAME:
            raise ValueError("immutable patched vLLM wheel filename differs")
        digest = sha256()
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if before.st_size != VLLM_PATCHED_WHEEL_SIZE or digest.hexdigest() != (
            VLLM_PATCHED_WHEEL_SHA256
        ):
            raise ValueError("immutable patched vLLM wheel identity differs")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
            with ZipFile(stream, "r") as archive:
                for item in expected_patch_closure:
                    member = _required_string(item["relative_path"], "vLLM member path")
                    content = archive.read(member)
                    if sha256(content).hexdigest() != item["patched_sha256"]:
                        raise ValueError(f"immutable patched vLLM member differs: {member}")
        after = os.fstat(descriptor)
        if _stable_stat(before) != _stable_stat(after):
            raise ValueError("immutable patched vLLM wheel changed while reading")
    finally:
        os.close(descriptor)
    return {
        "distribution": "vllm",
        "version": VLLM_PACKAGE_VERSION,
        "patched_wheel_filename": VLLM_PATCHED_WHEEL_FILENAME,
        "patched_wheel_sha256": VLLM_PATCHED_WHEEL_SHA256,
        "patched_wheel_bytes": VLLM_PATCHED_WHEEL_SIZE,
        "patched_manifest_filename": VLLM_PATCHED_MANIFEST_FILENAME,
        "patched_manifest_file_sha256": VLLM_PATCHED_MANIFEST_SHA256,
        "patched_manifest_file_bytes": VLLM_PATCHED_MANIFEST_SIZE,
        "member_closure": expected_patch_closure,
    }


def _validate_runtime_closure_record(record: Mapping[str, Any]) -> None:
    expected_keys = {
        "artifacts",
        "closed_record_sha256",
        "install_contract",
        "live_runtime_verification",
        "record_type",
        "schema_version",
        "target_runtime",
    }
    if set(record) != expected_keys:
        raise ValueError("runtime artifact closure keys differ")
    if (
        record.get("record_type") != RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE
        or record.get("schema_version") != RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION
    ):
        raise ValueError("runtime artifact closure schema differs")
    observed = _required_sha256(
        record.get("closed_record_sha256"),
        "runtime closure closed_record_sha256",
    )
    unsigned = dict(record)
    unsigned.pop("closed_record_sha256")
    if _canonical_json_sha256(unsigned) != observed:
        raise ValueError("runtime artifact closure closed digest differs")
    live = _required_mapping(record.get("live_runtime_verification"), "live verification")
    if (
        live.get("state") != "pending_not_executed"
        or live.get("passed") != []
        or live.get("success_credit") is not False
    ):
        raise ValueError("runtime closure may not claim live verification")


def _read_exact_small_file(
    path: Path,
    *,
    expected_name: str,
    expected_sha256: str,
    expected_size: int,
    label: str,
) -> bytes:
    if path.name != expected_name:
        raise ValueError(f"{label} filename differs")
    descriptor = _open_regular_nofollow(path, label=label)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > expected_size:
                raise ValueError(f"{label} exceeds exact size")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    if (
        _stable_stat(before) != _stable_stat(after)
        or len(content) != expected_size
        or sha256(content).hexdigest() != expected_sha256
    ):
        raise ValueError(f"{label} size, SHA-256, or stable stat differs")
    return content


def _open_regular_nofollow(path: Path, *, label: str) -> int:
    _require_no_symlink_path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode):
        os.close(descriptor)
        raise ValueError(f"{label} must be a regular non-symlink file")
    return descriptor


def _stable_stat(observed: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _require_no_symlink_path(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"path contains symlink component: {current}")


def _require_private_parent(path: Path) -> None:
    parent = path.parent
    _require_no_symlink_path(parent)
    if not parent.is_dir():
        raise FileNotFoundError(f"artifact parent does not exist: {parent}")


def _require_or_create_private_directory(path: Path) -> None:
    _require_no_symlink_path(path.parent)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise ValueError("runtime closure output root must be a regular directory")
    os.chmod(path, 0o700)


def _exclusive_write(path: Path, content: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_record_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_record_from_bytes(raw: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact record is not JSON") from exc
    if not isinstance(value, dict) or _canonical_record_file_bytes(value) != raw:
        raise ValueError("artifact record bytes are noncanonical")
    return value


def _require_pinned_closure_output(
    closed_record_sha256: str,
    file_sha256: str,
    file_size: int,
) -> None:
    if (
        closed_record_sha256,
        file_sha256,
        file_size,
    ) != (
        RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
        RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE,
    ):
        raise RuntimeError("runtime artifact closure output pins differ")


def _required_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _required_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _required_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


__all__ = [
    "RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256",
    "RUNTIME_ARTIFACT_CLOSURE_FILENAME",
    "RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256",
    "RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE",
    "RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE",
    "RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION",
    "RuntimeArtifactClosure",
    "VLLM_PATCHED_MANIFEST_SHA256",
    "VLLM_PATCHED_WHEEL_SHA256",
    "VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT",
    "VLLM_RUNTIME_BASE_LOCK_FILENAME",
    "VLLM_RUNTIME_BASE_LOCK_HASH_COUNT",
    "VLLM_RUNTIME_BASE_LOCK_SHA256",
    "VLLM_RUNTIME_BASE_LOCK_SIZE",
    "VLLM_RUNTIME_BASE_LOCK_VERSION_HASH_MAP_SHA256",
    "VLLM_RUNTIME_BASE_LOCK_VERSION_MAP_SHA256",
    "build_vllm_flashinfer_runtime_artifact_closure",
    "derive_flashinfer_direct_base_lock",
    "validate_flashinfer_direct_base_lock",
    "validate_vllm_flashinfer_runtime_artifact_closure",
]
