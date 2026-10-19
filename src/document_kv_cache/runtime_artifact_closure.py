"""Seal the local vLLM plus patched-FlashInfer runtime artifact closure.

This stage owns only local artifact authority.  It derives the 197-package
base lock from the reviewed 198-package lock and binds that lock to the
patched vLLM 0.30.0 wheel and the pristine FlashInfer 0.6.18.post1 wheel.
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

VLLM_PACKAGE_VERSION = "0.30.0+cu129"
VLLM_RUNTIME_LOCK_FILENAME = "vllm-0.30.0-cu129-py311-manylinux_2_35.lock"
VLLM_RUNTIME_LOCK_SHA256 = (
    "8b6f8618fff84070d44b35a2d6a54d5fc5831ecd8c375b5cbe9d847aeb3fae5d"
)
VLLM_RUNTIME_SOURCE_LOCK_SIZE = 385_393
VLLM_RUNTIME_SOURCE_LOCK_DISTRIBUTION_COUNT = 198
VLLM_RUNTIME_SOURCE_LOCK_HASH_COUNT = 4_237
VLLM_RUNTIME_BASE_LOCK_FILENAME = (
    "vllm-0.30.0-cu129-py311-manylinux_2_35-flashinfer-direct.lock"
)
VLLM_RUNTIME_BASE_LOCK_SHA256 = (
    "cf5f7dab9ee409f23436087d3ccd835a25b801c6d1538bc5eb20321dd1e4c1f4"
)
VLLM_RUNTIME_BASE_LOCK_SIZE = 385_126
VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT = 197
VLLM_RUNTIME_BASE_LOCK_HASH_COUNT = 4_236
VLLM_RUNTIME_BASE_LOCK_VERSION_MAP_SHA256 = (
    "805c53f6ea8ad77101549ade41245e30d7c1efd6e649cb59d4ad735b761045b1"
)
VLLM_RUNTIME_BASE_LOCK_VERSION_HASH_MAP_SHA256 = (
    "ffa58a9ad69208fe403c63ac14e1e69ca39c3a84768f91e596b16e0fcdc60c78"
)
_FLASHINFER_LOCK_STANZA = (
    b"flashinfer-python==0.6.18.post1 \\\n"
    b"    --hash=sha256:adb6d2952b47131138e108bd2999b1fe18b71f1a4c0b2684f168753db9563991\n"
    b"    # via\n"
    b"    #   -r src/document_kv_cache/runtime_locks/"
    b"vllm-0.30.0-cu129-py311-manylinux_2_35.in\n"
    b"    #   vllm\n"
    b"    # from https://flashinfer.ai/whl/\n"
)
FLASHINFER_RUNTIME_VERSION = "0.6.18.post1"
FLASHINFER_RUNTIME_WHEEL_FILENAME = (
    "flashinfer_python-0.6.18.post1-py3-none-any.whl"
)
FLASHINFER_RUNTIME_WHEEL_SHA256 = (
    "adb6d2952b47131138e108bd2999b1fe18b71f1a4c0b2684f168753db9563991"
)
FLASHINFER_RUNTIME_WHEEL_SIZE = 18_348_022
FLASHINFER_RUNTIME_MEMBER_SHA256 = (
    "b6b5883440287d05cd353e3c8c02201847b84bda7dc9ba3fa6d1b5d189658246"
)

VLLM_PATCHED_WHEEL_FILENAME = (
    "vllm-0.30.0+cu129-1cachete5m207efe2f8245ff6c1-"
    "cp38-abi3-manylinux_2_28_x86_64.whl"
)
VLLM_PATCHED_WHEEL_SHA256 = (
    "07efe2f8245ff6c16001cfa1cc940f4f82eaec9a75fa24b9e0bacf7643659247"
)
VLLM_PATCHED_WHEEL_SIZE = 537_513_535
VLLM_PATCHED_MANIFEST_FILENAME = (
    "vllm-0.30.0+cu129-1cachete5m207efe2f8245ff6c1-"
    "cp38-abi3-manylinux_2_28_x86_64.manifest.json"
)
VLLM_PATCHED_MANIFEST_SHA256 = (
    "8d2902856daff1ee0b1080deb6c36c0d7cc369c26df67a3498b07416ed1ffbc6"
)
VLLM_PATCHED_MANIFEST_SIZE = 2_609
_VLLM_PATCHED_MEMBER_CLOSURE: tuple[Mapping[str, str], ...] = (
    {
        "id": "vllm-qwen-attention-disable-e5m2-query-quant",
        "patched_sha256": (
            "8efa67985b30f0521867ca0dd373c88cca165ac7a546f6aa1b17cd08392058ce"
        ),
        "reason": (
            "vLLM 0.30.0 constructs an E4M3-only query quantizer for fp8_e5m2; "
            "E5M2 queries must stay in the model compute dtype."
        ),
        "relative_path": "vllm/model_executor/layers/attention/attention.py",
        "source_sha256": (
            "bc897e452f3aea603e353ca42a365d3be8836e9485d78cbf8cc4677baca2fb42"
        ),
    },
    {
        "id": "vllm-triton-reshape-cache-e5m2-closure",
        "patched_sha256": (
            "0682ca7bc56edf7cea5419188a81c78510b54192471472b160aa447ac0ceeb08"
        ),
        "reason": (
            "vLLM 0.30.0 otherwise views fp8_e5m2 pages through the platform "
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
            "a28a150bcead50012c1559406feed0bb0b3a370dd881a8f6c3009bc090c6923a"
        ),
        "reason": (
            "vLLM 0.30.0 otherwise binds Triton KV views to E4M3 and rejects "
            "fp8_e5m2 on A10G despite the targeted SM80+ E5M2 path."
        ),
        "relative_path": "vllm/v1/attention/backends/triton_attn.py",
        "source_sha256": (
            "1057663d110e1d5229495a799027b27277c1eca325d5b6780fa2259afbf58a1a"
        ),
    },
)

RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE = (
    "document_kv.vllm_flashinfer_runtime_artifact_closure.v1"
)
RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION = 1
RUNTIME_ARTIFACT_CLOSURE_FILENAME = (
    "vllm-0.30.0-flashinfer-0.6.18.post1-runtime-closure.json"
)

RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256 = (
    "fc15bbd35142de8c0c79ce51607e8c761db005859824e154658cc9e86153a271"
)
RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256 = (
    "193c095deeb1636a4cf5f873ed6b07bedf0f466c91e3df34417b357e9f125c03"
)
RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE = 6_332

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
        label="reviewed 198-distribution runtime lock",
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
            label="derived 197-distribution runtime lock",
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
        label="reviewed 198-distribution runtime lock",
    )
    target_bytes = _read_exact_small_file(
        Path(base_lock),
        expected_name=VLLM_RUNTIME_BASE_LOCK_FILENAME,
        expected_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        expected_size=VLLM_RUNTIME_BASE_LOCK_SIZE,
        label="derived 197-distribution runtime lock",
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
    flashinfer = _validate_pristine_flashinfer_wheel(pristine_flashinfer_wheel)
    del patched_flashinfer_wheel, flashinfer_manifest
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
                "version": FLASHINFER_RUNTIME_VERSION,
                "source_wheel_filename": FLASHINFER_RUNTIME_WHEEL_FILENAME,
                "source_wheel_sha256": FLASHINFER_RUNTIME_WHEEL_SHA256,
                "source_wheel_bytes": FLASHINFER_RUNTIME_WHEEL_SIZE,
                "patched_wheel_filename": FLASHINFER_RUNTIME_WHEEL_FILENAME,
                "patched_wheel_sha256": flashinfer["wheel_sha256"],
                "patched_wheel_bytes": flashinfer["wheel_bytes"],
                "patched_manifest_filename": None,
                "patched_manifest_file_sha256": None,
                "patched_manifest_file_bytes": 0,
                "patched_manifest_closed_record_sha256": None,
                "patched_member": {
                    "relative_path": "flashinfer/comm/fd_exchange.py",
                    "sha256": FLASHINFER_RUNTIME_MEMBER_SHA256,
                },
                "patched_record_sha256": None,
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
                        + FLASHINFER_RUNTIME_WHEEL_SHA256,
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
                    "archive_sha256": FLASHINFER_RUNTIME_WHEEL_SHA256,
                },
                "require_normalized_url_equality": True,
                "require_archive_info_sha256": True,
            },
            "distribution_counts": {
                "base_lock": 197,
                "with_flashinfer": 198,
                "with_vllm": 199,
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
                "exact_198_dependency_version_equivalence",
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


def _validate_pristine_flashinfer_wheel(wheel_path: Path) -> dict[str, Any]:
    """Accept the released FlashInfer wheel whose annotation import is already present."""

    digest = sha256()
    descriptor = _open_regular_nofollow(wheel_path, label="pristine FlashInfer wheel")
    try:
        before = os.fstat(descriptor)
        if wheel_path.name != FLASHINFER_RUNTIME_WHEEL_FILENAME:
            raise ValueError("pristine FlashInfer wheel filename differs")
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if before.st_size != FLASHINFER_RUNTIME_WHEEL_SIZE or digest.hexdigest() != (
            FLASHINFER_RUNTIME_WHEEL_SHA256
        ):
            raise ValueError("pristine FlashInfer wheel identity differs")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
            with ZipFile(stream, "r") as archive:
                member = archive.read("flashinfer/comm/fd_exchange.py")
        if sha256(member).hexdigest() != FLASHINFER_RUNTIME_MEMBER_SHA256:
            raise ValueError("pristine FlashInfer fd_exchange.py differs")
        if b"from __future__ import annotations\n" not in member:
            raise ValueError("pristine FlashInfer fd_exchange.py lacks future annotations")
        after = os.fstat(descriptor)
        if _stable_stat(before) != _stable_stat(after):
            raise ValueError("pristine FlashInfer wheel changed while reading")
    finally:
        os.close(descriptor)
    return {
        "wheel_sha256": FLASHINFER_RUNTIME_WHEEL_SHA256,
        "wheel_bytes": FLASHINFER_RUNTIME_WHEEL_SIZE,
    }


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
