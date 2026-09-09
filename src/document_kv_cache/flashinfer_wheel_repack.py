"""Deterministically patch the reviewed FlashInfer 0.6.16.post3 wheel.

The publication runtime cannot patch ``site-packages`` after installation.
This module therefore validates the complete pristine wheel closure, adds the
one reviewed future import to ``flashinfer/comm/fd_exchange.py``, rebuilds
``RECORD``, and emits a sealed, content-addressed artifact.  Public entry
points intentionally accept paths only; the source and patch authority is
package-owned.
"""

from __future__ import annotations

import ast
import base64
import csv
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import struct
import tempfile
from typing import Any, Mapping, Sequence
import unicodedata
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo


FLASHINFER_PACKAGE_NAME = "flashinfer-python"
FLASHINFER_PACKAGE_VERSION = "0.6.16.post3"
FLASHINFER_SOURCE_WHEEL_FILENAME = (
    "flashinfer_python-0.6.16.post3-py3-none-any.whl"
)
FLASHINFER_SOURCE_WHEEL_SHA256 = (
    "caf686b9b079abe1c9d65ab505698bd325e8072de40afd822f2c74f2ac3bc601"
)
FLASHINFER_SOURCE_WHEEL_SIZE = 15_836_034
FLASHINFER_SOURCE_MEMBER_COUNT = 4_957
FLASHINFER_SOURCE_TOTAL_UNCOMPRESSED_BYTES = 82_056_127
FLASHINFER_SOURCE_TOTAL_COMPRESSED_BYTES = 14_779_090
FLASHINFER_SOURCE_CONTENT_TREE_SHA256 = (
    "ac482b1516d8a3bf90afa88bce50ace8d549518ea1bcdbd6cc365e98b5250095"
)
FLASHINFER_SOURCE_ARCHIVE_TREE_SHA256 = (
    "27830718bee80506a1cb5468ca2d6d426a84fbcb93f9ef8663a7e0442f82f6ec"
)

FLASHINFER_TARGET_MEMBER = "flashinfer/comm/fd_exchange.py"
FLASHINFER_TARGET_SOURCE_SHA256 = (
    "6f9549238cc450efeb30aa740c0bdc2e6dfd4cfa29cee43a9ab010c90a407cee"
)
FLASHINFER_TARGET_SOURCE_SIZE = 9_656
FLASHINFER_TARGET_PATCH_OFFSET = 1_244
FLASHINFER_TARGET_PATCH_BYTES = b"from __future__ import annotations\n"
FLASHINFER_TARGET_PATCHED_SHA256 = (
    "05a4e1fa20c92b71de07f83695e8209c9f6d226072a6ea79a766af89c9fc3f25"
)
FLASHINFER_TARGET_PATCHED_SIZE = 9_691
FLASHINFER_TARGET_PREFIX_SHA256 = (
    "1360d8f7e528632b4951efaf2c3e1267ee3082dd18957164bfa69c4dd361b99b"
)
FLASHINFER_TARGET_SUFFIX_SHA256 = (
    "2e256746cc43e38447a96e850f9da01e60a3a0d3ac9d52cb9724a6da45d829f2"
)
FLASHINFER_TARGET_ANCHOR = b'"""\n\nimport array\n'
FLASHINFER_TARGET_ANNOTATION = (
    b"def _fd_ancillary(fd: int) -> tuple[tuple[int, int, array.array[int]]]:\n"
)

FLASHINFER_METADATA_MEMBER = (
    "flashinfer_python-0.6.16.post3.dist-info/METADATA"
)
FLASHINFER_METADATA_SHA256 = (
    "0560a3149cea4926be754e9a525640ebf3a11a2daed8fd7d65fc1baa05cf3385"
)
FLASHINFER_METADATA_SIZE = 11_565
FLASHINFER_WHEEL_MEMBER = "flashinfer_python-0.6.16.post3.dist-info/WHEEL"
FLASHINFER_WHEEL_MEMBER_SHA256 = (
    "35c5c429cac02f90aac2939994d9bea7bd09905d31cd129c3fe3374ef7c9e514"
)
FLASHINFER_WHEEL_MEMBER_SIZE = 97
FLASHINFER_RECORD_MEMBER = "flashinfer_python-0.6.16.post3.dist-info/RECORD"
FLASHINFER_SOURCE_RECORD_SHA256 = (
    "ce54586ff378266468ace4b9b2242de54d54dd9f4399b14bb9601384259b8640"
)
FLASHINFER_SOURCE_RECORD_SIZE = 623_912

FLASHINFER_PATCHED_WHEEL_MANIFEST_RECORD_TYPE = (
    "document_kv.flashinfer_patched_wheel_manifest.v1"
)
FLASHINFER_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION = 1

# Package-owned identities derived by the reviewed deterministic A/B build.
FLASHINFER_PATCHED_WHEEL_FILENAME = (
    "flashinfer_python-0.6.16.post3-"
    "1cachetpy31104e032c70234e876-py3-none-any.whl"
)
FLASHINFER_PATCHED_WHEEL_SHA256 = (
    "04e032c70234e8769f5ab7e787231c339a5b5230fca5f5b0b80f1a2a0ccad6ec"
)
FLASHINFER_PATCHED_WHEEL_SIZE = 83_113_106
FLASHINFER_PATCHED_CONTENT_TREE_SHA256 = (
    "0d64ea420d22e6a0f097fbc8f7db507d1b1757388db457daefdb330f0eecd13b"
)
FLASHINFER_PATCHED_ARCHIVE_TREE_SHA256 = (
    "bb889dc0c2411d1543f712f64cff2d354b1cb1c063616d9a4a348a8b1d8dcc34"
)
FLASHINFER_PATCHED_RECORD_SHA256 = (
    "dc9c8cf8c7c54682ca3908fa047e67f9b447fc86b38ba8b5e83032109dcb8756"
)
FLASHINFER_PATCHED_RECORD_SIZE = 623_912
FLASHINFER_PATCHED_MANIFEST_FILENAME = (
    "flashinfer_python-0.6.16.post3-"
    "1cachetpy31104e032c70234e876-py3-none-any.manifest.json"
)
FLASHINFER_PATCHED_MANIFEST_SIZE = 2_970
FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256 = (
    "60c5a3aa75914c8fb6d790802adb2a5291a5eaffc281ee17080a3609193e229d"
)
FLASHINFER_PATCHED_MANIFEST_FILE_SHA256 = (
    "4b5c3f726552697a2afa0c3f64655621d2201f9576de9306a986a696681d7303"
)

_FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MAX_ARCHIVE_MEMBERS = 5_000
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 1_000
_EXPECTED_PYTHON_IMPLEMENTATION = "CPython"
_EXPECTED_PYTHON_VERSION = "3.11.16"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RECORD_DIGEST_RE = re.compile(r"[A-Za-z0-9_-]{43}")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True, slots=True)
class PatchedFlashInferWheel:
    """Paths and identities produced by one reviewed deterministic build."""

    wheel_path: Path
    manifest_path: Path
    wheel_sha256: str
    wheel_size: int
    manifest_closed_record_sha256: str
    manifest_file_sha256: str


@dataclass(frozen=True, slots=True)
class _Member:
    name: str
    content: bytes
    compressed_size: int
    crc32: int
    timestamp: tuple[int, int, int, int, int, int]
    create_system: int
    external_attr: int
    internal_attr: int
    flag_bits: int
    compression: int

    @property
    def sha256(self) -> str:
        return sha256(self.content).hexdigest()

    @property
    def mode(self) -> int:
        return (self.external_attr >> 16) & 0xFFFF


@dataclass(frozen=True, slots=True)
class _Inspection:
    members: tuple[_Member, ...]
    content_tree_sha256: str
    archive_tree_sha256: str
    total_uncompressed_bytes: int
    total_compressed_bytes: int
    record_sha256: str
    record_size: int

    def member_map(self) -> dict[str, _Member]:
        return {member.name: member for member in self.members}


@dataclass(frozen=True, slots=True)
class _Build:
    wheel_name: str
    wheel_bytes: bytes
    wheel_sha256: str
    manifest_name: str
    manifest_bytes: bytes
    manifest: Mapping[str, Any]


def validate_pristine_flashinfer_wheel(
    source_wheel: str | Path,
) -> Mapping[str, Any]:
    """Validate and summarize the exact reviewed pristine release wheel."""

    source_path = Path(source_wheel)
    source_bytes = _read_exact_regular_file(
        source_path,
        expected_name=FLASHINFER_SOURCE_WHEEL_FILENAME,
        expected_sha256=FLASHINFER_SOURCE_WHEEL_SHA256,
        expected_size=FLASHINFER_SOURCE_WHEEL_SIZE,
        label="pristine FlashInfer wheel",
    )
    inspection = _inspect_wheel_bytes(source_bytes, label="pristine FlashInfer wheel")
    _require_pristine_authority(inspection)
    return _inspection_summary(inspection)


def repack_flashinfer_0616_post3_wheel(
    source_wheel: str | Path,
    output_dir: str | Path,
) -> PatchedFlashInferWheel:
    """Build twice and publish the deterministic patched wheel and manifest."""

    _require_governed_builder()
    source_path = Path(source_wheel)
    output_root = Path(output_dir)
    _require_or_create_private_directory(output_root)
    with tempfile.TemporaryDirectory(prefix=".flashinfer-build-a-", dir=output_root) as a_text:
        with tempfile.TemporaryDirectory(
            prefix=".flashinfer-build-b-", dir=output_root
        ) as b_text:
            first = _build_reviewed_wheel(source_path)
            second = _build_reviewed_wheel(source_path)
            if (
                first.wheel_name != second.wheel_name
                or first.wheel_bytes != second.wheel_bytes
                or first.manifest_name != second.manifest_name
                or first.manifest_bytes != second.manifest_bytes
            ):
                raise RuntimeError("independent FlashInfer A/B builds differ")
            first_wheel = Path(a_text) / first.wheel_name
            first_manifest = Path(a_text) / first.manifest_name
            second_wheel = Path(b_text) / second.wheel_name
            second_manifest = Path(b_text) / second.manifest_name
            _exclusive_write(first_wheel, first.wheel_bytes)
            _exclusive_write(first_manifest, first.manifest_bytes)
            _exclusive_write(second_wheel, second.wheel_bytes)
            _exclusive_write(second_manifest, second.manifest_bytes)
            if (
                first_wheel.read_bytes() != second_wheel.read_bytes()
                or first_manifest.read_bytes() != second_manifest.read_bytes()
            ):
                raise RuntimeError("on-disk FlashInfer A/B builds differ")
            _validate_build_files(source_path, first_wheel, first_manifest)
            _validate_build_files(source_path, second_wheel, second_manifest)
            target_wheel = output_root / first.wheel_name
            target_manifest = output_root / first.manifest_name
            _publish_identical(first_wheel, target_wheel)
            _publish_identical(first_manifest, target_manifest)

    validated = validate_patched_flashinfer_wheel(
        source_path,
        target_wheel,
        target_manifest,
    )
    return PatchedFlashInferWheel(
        wheel_path=target_wheel,
        manifest_path=target_manifest,
        wheel_sha256=_required_sha256(validated["patched_wheel_sha256"], "wheel SHA"),
        wheel_size=_required_int(validated["patched_wheel_size"], "wheel size"),
        manifest_closed_record_sha256=_required_sha256(
            validated["manifest_closed_record_sha256"],
            "manifest closed record SHA",
        ),
        manifest_file_sha256=_required_sha256(
            validated["manifest_file_sha256"],
            "manifest file SHA",
        ),
    )


def validate_patched_flashinfer_wheel(
    source_wheel: str | Path,
    patched_wheel: str | Path,
    manifest_path: str | Path,
) -> Mapping[str, Any]:
    """Validate the complete source-to-patched closure and sealed manifest."""

    _require_governed_builder()
    source_path = Path(source_wheel)
    patched_path = Path(patched_wheel)
    manifest_file = Path(manifest_path)
    expected = _build_reviewed_wheel(source_path)
    observed_wheel = _read_exact_regular_file(
        patched_path,
        expected_name=expected.wheel_name,
        expected_sha256=expected.wheel_sha256,
        expected_size=len(expected.wheel_bytes),
        label="patched FlashInfer wheel",
    )
    if observed_wheel != expected.wheel_bytes:
        raise ValueError("patched FlashInfer wheel bytes differ from deterministic build")
    observed_manifest = _read_exact_regular_file(
        manifest_file,
        expected_name=expected.manifest_name,
        expected_sha256=sha256(expected.manifest_bytes).hexdigest(),
        expected_size=len(expected.manifest_bytes),
        label="patched FlashInfer manifest",
    )
    if observed_manifest != expected.manifest_bytes:
        raise ValueError("patched FlashInfer manifest bytes differ from deterministic build")
    manifest = _canonical_manifest_from_bytes(observed_manifest)
    _validate_manifest_record(manifest)
    _require_pinned_derived_outputs(
        wheel_sha256=expected.wheel_sha256,
        wheel_size=len(expected.wheel_bytes),
        manifest_closed_record_sha256=_required_sha256(
            manifest.get("closed_record_sha256"),
            "manifest.closed_record_sha256",
        ),
        manifest_file_sha256=sha256(observed_manifest).hexdigest(),
    )
    return {
        "patched_wheel_sha256": expected.wheel_sha256,
        "patched_wheel_size": len(expected.wheel_bytes),
        "manifest_closed_record_sha256": manifest["closed_record_sha256"],
        "manifest_file_sha256": sha256(observed_manifest).hexdigest(),
    }


def _build_reviewed_wheel(source_path: Path) -> _Build:
    source_bytes = _read_exact_regular_file(
        source_path,
        expected_name=FLASHINFER_SOURCE_WHEEL_FILENAME,
        expected_sha256=FLASHINFER_SOURCE_WHEEL_SHA256,
        expected_size=FLASHINFER_SOURCE_WHEEL_SIZE,
        label="pristine FlashInfer wheel",
    )
    source = _inspect_wheel_bytes(source_bytes, label="pristine FlashInfer wheel")
    _require_pristine_authority(source)
    target_bytes = _patched_target_bytes(source.member_map()[FLASHINFER_TARGET_MEMBER].content)
    wheel_bytes = _deterministic_wheel_bytes(source, target_bytes)
    patched = _inspect_wheel_bytes(wheel_bytes, label="patched FlashInfer wheel")
    _require_patched_closure(source, patched, target_bytes)
    wheel_digest = sha256(wheel_bytes).hexdigest()
    wheel_name = _patched_wheel_name(wheel_digest)
    manifest = _manifest_record(
        source=source,
        patched=patched,
        wheel_name=wheel_name,
        wheel_sha256=wheel_digest,
        wheel_size=len(wheel_bytes),
    )
    manifest_name = wheel_name.removesuffix(".whl") + ".manifest.json"
    manifest_bytes = _canonical_manifest_file_bytes(manifest)
    patched_record = _required_mapping(manifest["patched_wheel"], "patched_wheel")
    if (
        wheel_name != FLASHINFER_PATCHED_WHEEL_FILENAME
        or wheel_digest != FLASHINFER_PATCHED_WHEEL_SHA256
        or len(wheel_bytes) != FLASHINFER_PATCHED_WHEEL_SIZE
        or patched.content_tree_sha256 != FLASHINFER_PATCHED_CONTENT_TREE_SHA256
        or patched.archive_tree_sha256 != FLASHINFER_PATCHED_ARCHIVE_TREE_SHA256
        or patched.record_sha256 != FLASHINFER_PATCHED_RECORD_SHA256
        or patched.record_size != FLASHINFER_PATCHED_RECORD_SIZE
        or manifest_name != FLASHINFER_PATCHED_MANIFEST_FILENAME
        or len(manifest_bytes) != FLASHINFER_PATCHED_MANIFEST_SIZE
        or manifest["closed_record_sha256"]
        != FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
        or sha256(manifest_bytes).hexdigest()
        != FLASHINFER_PATCHED_MANIFEST_FILE_SHA256
        or patched_record.get("sha256") != FLASHINFER_PATCHED_WHEEL_SHA256
    ):
        raise RuntimeError("complete derived FlashInfer artifact authority differs")
    return _Build(
        wheel_name=wheel_name,
        wheel_bytes=wheel_bytes,
        wheel_sha256=wheel_digest,
        manifest_name=manifest_name,
        manifest_bytes=manifest_bytes,
        manifest=manifest,
    )


def _inspect_wheel_bytes(raw: bytes, *, label: str) -> _Inspection:
    if not isinstance(raw, bytes):
        raise TypeError(f"{label} bytes must be bytes")
    _require_eocd_closure(raw, label=label)
    members: list[_Member] = []
    seen: set[str] = set()
    seen_casefold: set[str] = set()
    with ZipFile(io.BytesIO(raw), "r") as archive:
        infos = archive.infolist()
        if not 1 <= len(infos) <= _MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"{label} member count is unsafe")
        if archive.comment:
            raise ValueError(f"{label} archive comment is forbidden")
        start_dir = _required_int(getattr(archive, "start_dir"), f"{label} start_dir")
        ordered = sorted(infos, key=lambda item: item.header_offset)
        if len({item.header_offset for item in infos}) != len(infos):
            raise ValueError(f"{label} repeats local-header offsets")
        for index, info in enumerate(ordered):
            _validate_local_header(
                raw,
                info,
                next_offset=(
                    ordered[index + 1].header_offset
                    if index + 1 < len(ordered)
                    else start_dir
                ),
                label=label,
            )
        for info in infos:
            name = info.filename
            _validate_member_name(name)
            if info.orig_filename != name:
                raise ValueError(f"{label} contains a NUL-truncated member name")
            if name in seen or name.casefold() in seen_casefold:
                raise ValueError(f"{label} repeats or aliases member {name!r}")
            seen.add(name)
            seen_casefold.add(name.casefold())
            if info.is_dir() or name.endswith("/"):
                raise ValueError(f"{label} directory entries are forbidden")
            if info.flag_bits not in {0, 0x800}:
                if info.flag_bits & 0x1:
                    raise ValueError(f"{label} contains encrypted member {name!r}")
                raise ValueError(f"{label} member flags differ for {name!r}")
            if info.compress_type not in {ZIP_DEFLATED, ZIP_STORED}:
                raise ValueError(f"{label} member compression differs for {name!r}")
            if info.extra or info.comment:
                raise ValueError(f"{label} member extra/comment is forbidden for {name!r}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if info.create_system != 3 or stat.S_IFMT(mode) != stat.S_IFREG:
                raise ValueError(f"{label} contains link or special member {name!r}")
            if info.file_size and info.compress_size == 0:
                raise ValueError(f"{label} impossible compression for {name!r}")
            if (
                info.compress_size
                and info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO
            ):
                raise ValueError(f"{label} compression ratio is unsafe for {name!r}")
            content = archive.read(info)
            if len(content) != info.file_size:
                raise ValueError(f"{label} member size differs for {name!r}")
            members.append(
                _Member(
                    name=name,
                    content=content,
                    compressed_size=info.compress_size,
                    crc32=info.CRC,
                    timestamp=info.date_time,
                    create_system=info.create_system,
                    external_attr=info.external_attr,
                    internal_attr=info.internal_attr,
                    flag_bits=info.flag_bits,
                    compression=info.compress_type,
                )
            )
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise ValueError(f"{label} CRC failure in {bad_crc!r}")
    total_uncompressed = sum(len(member.content) for member in members)
    if total_uncompressed > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ValueError(f"{label} uncompressed size is unsafe")
    by_name = {member.name: member for member in members}
    _require_core_members(by_name, label=label)
    _validate_metadata(by_name, label=label)
    _validate_record(by_name, label=label)
    return _Inspection(
        members=tuple(members),
        content_tree_sha256=_content_tree_sha256(members),
        archive_tree_sha256=_archive_tree_sha256(members),
        total_uncompressed_bytes=total_uncompressed,
        total_compressed_bytes=sum(member.compressed_size for member in members),
        record_sha256=by_name[FLASHINFER_RECORD_MEMBER].sha256,
        record_size=len(by_name[FLASHINFER_RECORD_MEMBER].content),
    )


def _validate_local_header(
    raw: bytes,
    info: ZipInfo,
    *,
    next_offset: int,
    label: str,
) -> None:
    offset = info.header_offset
    if offset < 0 or offset + 30 > len(raw) or raw[offset : offset + 4] != b"PK\x03\x04":
        raise ValueError(f"{label} local header differs for {info.filename!r}")
    (
        _signature,
        _extract_version,
        flags,
        compression,
        _time,
        _date,
        crc32,
        compressed_size,
        file_size,
        name_length,
        extra_length,
    ) = struct.unpack_from("<IHHHHHIIIHH", raw, offset)
    name_start = offset + 30
    name_end = name_start + name_length
    data_start = name_end + extra_length
    data_end = data_start + info.compress_size
    if extra_length != 0 or data_end > next_offset or next_offset > len(raw):
        raise ValueError(f"{label} local member overlaps for {info.filename!r}")
    encoding = "utf-8" if flags & 0x800 else "cp437"
    try:
        local_name = raw[name_start:name_end].decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} local name is invalid") from exc
    if local_name != info.orig_filename:
        raise ValueError(f"{label} local/central name differs for {info.filename!r}")
    if flags != info.flag_bits or compression != info.compress_type:
        raise ValueError(f"{label} local/central flags differ for {info.filename!r}")
    if flags & 0x08:
        raise ValueError(f"{label} data descriptors are forbidden")
    if (crc32, compressed_size, file_size) != (
        info.CRC,
        info.compress_size,
        info.file_size,
    ):
        raise ValueError(f"{label} local/central metadata differs for {info.filename!r}")


def _require_eocd_closure(raw: bytes, *, label: str) -> None:
    offset = raw.rfind(b"PK\x05\x06")
    if offset < 0 or offset + 22 > len(raw):
        raise ValueError(f"{label} end-of-central-directory is missing")
    (
        _signature,
        disk,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_size,
    ) = struct.unpack_from("<IHHHHIIH", raw, offset)
    if disk or central_disk or disk_entries != total_entries:
        raise ValueError(f"{label} multi-disk ZIP is forbidden")
    if total_entries == 0xFFFF or central_size == 0xFFFFFFFF:
        raise ValueError(f"{label} ZIP64 archive is forbidden")
    if comment_size or offset + 22 != len(raw):
        raise ValueError(f"{label} comment or trailing data is forbidden")
    if central_offset + central_size != offset:
        raise ValueError(f"{label} central-directory bounds differ")


def _validate_member_name(name: str) -> None:
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or unicodedata.normalize("NFC", name) != name
    ):
        raise ValueError(f"unsafe wheel member name {name!r}")
    member = PurePosixPath(name)
    if (
        member.is_absolute()
        or member.as_posix() != name
        or not member.parts
        or any(part in {"", ".", ".."} for part in member.parts)
        or _DRIVE_RE.match(member.parts[0])
    ):
        raise ValueError(f"unsafe wheel member path {name!r}")


def _require_core_members(by_name: Mapping[str, _Member], *, label: str) -> None:
    expected = {
        FLASHINFER_TARGET_MEMBER,
        FLASHINFER_METADATA_MEMBER,
        FLASHINFER_WHEEL_MEMBER,
        FLASHINFER_RECORD_MEMBER,
    }
    missing = sorted(expected - set(by_name))
    if missing:
        raise ValueError(f"{label} is missing core members {missing!r}")
    dist_info_roots = {
        name.split("/", 1)[0]
        for name in by_name
        if ".dist-info/" in name
    }
    if dist_info_roots != {"flashinfer_python-0.6.16.post3.dist-info"}:
        raise ValueError(f"{label} dist-info closure differs")
    if any(name.endswith(("/RECORD.jws", "/RECORD.p7s")) for name in by_name):
        raise ValueError(f"{label} signed RECORD sidecars are forbidden")


def _validate_metadata(by_name: Mapping[str, _Member], *, label: str) -> None:
    metadata = BytesParser(policy=default).parsebytes(
        by_name[FLASHINFER_METADATA_MEMBER].content
    )
    if (
        metadata.get("Metadata-Version") != "2.4"
        or metadata.get("Name") != FLASHINFER_PACKAGE_NAME
        or metadata.get("Version") != FLASHINFER_PACKAGE_VERSION
        or metadata.get("Requires-Python") != "<4.0,>=3.10"
    ):
        raise ValueError(f"{label} METADATA identity differs")
    wheel = BytesParser(policy=default).parsebytes(by_name[FLASHINFER_WHEEL_MEMBER].content)
    if (
        wheel.get("Wheel-Version") != "1.0"
        or wheel.get("Root-Is-Purelib") != "true"
        or wheel.get_all("Tag", []) != ["py3-none-any"]
        or wheel.get("Generator") != "setuptools (0.6.16.post3)"
    ):
        raise ValueError(f"{label} WHEEL identity differs")


def _validate_record(by_name: Mapping[str, _Member], *, label: str) -> None:
    try:
        record_text = by_name[FLASHINFER_RECORD_MEMBER].content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} RECORD is not UTF-8") from exc
    rows = list(csv.reader(io.StringIO(record_text, newline="")))
    if len(rows) != len(by_name) or any(len(row) != 3 for row in rows):
        raise ValueError(f"{label} RECORD row count/shape differs")
    observed: set[str] = set()
    for path, digest_field, size_field in rows:
        if path in observed or path not in by_name:
            raise ValueError(f"{label} RECORD path differs for {path!r}")
        observed.add(path)
        content = by_name[path].content
        if path == FLASHINFER_RECORD_MEMBER:
            if digest_field or size_field:
                raise ValueError(f"{label} RECORD self row is not blank")
            continue
        if not digest_field.startswith("sha256="):
            raise ValueError(f"{label} RECORD hash algorithm differs for {path!r}")
        encoded = digest_field.removeprefix("sha256=")
        if not _RECORD_DIGEST_RE.fullmatch(encoded):
            raise ValueError(f"{label} RECORD digest differs for {path!r}")
        try:
            decoded = base64.urlsafe_b64decode(encoded + "=")
        except ValueError as exc:
            raise ValueError(f"{label} RECORD digest is invalid for {path!r}") from exc
        if decoded != sha256(content).digest() or size_field != str(len(content)):
            raise ValueError(f"{label} RECORD hash/size mismatch for {path!r}")
    if observed != set(by_name):
        raise ValueError(f"{label} RECORD coverage differs")


def _require_pristine_authority(inspection: _Inspection) -> None:
    by_name = inspection.member_map()
    expected_core = {
        FLASHINFER_TARGET_MEMBER: (
            FLASHINFER_TARGET_SOURCE_SHA256,
            FLASHINFER_TARGET_SOURCE_SIZE,
        ),
        FLASHINFER_METADATA_MEMBER: (
            FLASHINFER_METADATA_SHA256,
            FLASHINFER_METADATA_SIZE,
        ),
        FLASHINFER_WHEEL_MEMBER: (
            FLASHINFER_WHEEL_MEMBER_SHA256,
            FLASHINFER_WHEEL_MEMBER_SIZE,
        ),
        FLASHINFER_RECORD_MEMBER: (
            FLASHINFER_SOURCE_RECORD_SHA256,
            FLASHINFER_SOURCE_RECORD_SIZE,
        ),
    }
    for name, (expected_sha, expected_size) in expected_core.items():
        member = by_name[name]
        if member.sha256 != expected_sha or len(member.content) != expected_size:
            raise ValueError(f"pristine FlashInfer core member differs: {name}")
    if (
        len(inspection.members) != FLASHINFER_SOURCE_MEMBER_COUNT
        or inspection.total_uncompressed_bytes
        != FLASHINFER_SOURCE_TOTAL_UNCOMPRESSED_BYTES
        or inspection.total_compressed_bytes != FLASHINFER_SOURCE_TOTAL_COMPRESSED_BYTES
        or inspection.content_tree_sha256 != FLASHINFER_SOURCE_CONTENT_TREE_SHA256
        or inspection.archive_tree_sha256 != FLASHINFER_SOURCE_ARCHIVE_TREE_SHA256
    ):
        raise ValueError("pristine FlashInfer complete member inventory differs")
    target = by_name[FLASHINFER_TARGET_MEMBER].content
    if target.count(FLASHINFER_TARGET_ANCHOR) != 1:
        raise ValueError("FlashInfer future-import anchor must occur exactly once")
    if target.index(FLASHINFER_TARGET_ANCHOR) + 4 != FLASHINFER_TARGET_PATCH_OFFSET:
        raise ValueError("FlashInfer future-import offset differs")
    if target.count(FLASHINFER_TARGET_ANNOTATION) != 1:
        raise ValueError("FlashInfer array.array annotation differs")
    if target.count(FLASHINFER_TARGET_PATCH_BYTES) != 0:
        raise ValueError("pristine FlashInfer target already has future annotations")


def _patched_target_bytes(source: bytes) -> bytes:
    if len(source) != FLASHINFER_TARGET_SOURCE_SIZE or sha256(source).hexdigest() != (
        FLASHINFER_TARGET_SOURCE_SHA256
    ):
        raise ValueError("FlashInfer patch source bytes differ")
    prefix = source[:FLASHINFER_TARGET_PATCH_OFFSET]
    suffix = source[FLASHINFER_TARGET_PATCH_OFFSET:]
    if (
        sha256(prefix).hexdigest() != FLASHINFER_TARGET_PREFIX_SHA256
        or sha256(suffix).hexdigest() != FLASHINFER_TARGET_SUFFIX_SHA256
    ):
        raise ValueError("FlashInfer patch boundary differs")
    patched = prefix + FLASHINFER_TARGET_PATCH_BYTES + suffix
    if len(patched) != FLASHINFER_TARGET_PATCHED_SIZE or sha256(patched).hexdigest() != (
        FLASHINFER_TARGET_PATCHED_SHA256
    ):
        raise RuntimeError("FlashInfer patched member identity differs")
    _require_ast_only_future_import(source, patched)
    return patched


def _require_ast_only_future_import(source: bytes, patched: bytes) -> None:
    try:
        source_ast = ast.parse(source.decode("utf-8"), filename=FLASHINFER_TARGET_MEMBER)
        patched_ast = ast.parse(patched.decode("utf-8"), filename=FLASHINFER_TARGET_MEMBER)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError("FlashInfer target source is not valid UTF-8 Python") from exc
    future_nodes = [
        node
        for node in patched_ast.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    if len(future_nodes) != 1:
        raise ValueError("patched FlashInfer target must contain one future import")
    future = future_nodes[0]
    if future.level != 0 or [(item.name, item.asname) for item in future.names] != [
        ("annotations", None)
    ]:
        raise ValueError("patched FlashInfer future import differs")
    patched_ast.body.remove(future)
    if ast.dump(source_ast, include_attributes=False) != ast.dump(
        patched_ast,
        include_attributes=False,
    ):
        raise ValueError("FlashInfer patch changes AST beyond the future import")
    compile(patched.decode("utf-8"), FLASHINFER_TARGET_MEMBER, "exec")


def _deterministic_wheel_bytes(source: _Inspection, target_bytes: bytes) -> bytes:
    by_name = source.member_map()
    contents = {
        name: (target_bytes if name == FLASHINFER_TARGET_MEMBER else member.content)
        for name, member in by_name.items()
        if name != FLASHINFER_RECORD_MEMBER
    }
    contents[FLASHINFER_RECORD_MEMBER] = _record_bytes(contents)
    order = [
        *sorted(name for name in contents if name != FLASHINFER_RECORD_MEMBER),
        FLASHINFER_RECORD_MEMBER,
    ]
    output = io.BytesIO()
    with ZipFile(
        output,
        "w",
        compression=ZIP_STORED,
        allowZip64=False,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for name in order:
            source_member = by_name[name]
            info = ZipInfo(name, date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = ZIP_STORED
            info.create_system = 3
            info.external_attr = _canonical_external_attr(source_member.mode)
            info.internal_attr = 0
            info.flag_bits = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(
                info,
                contents[name],
                compress_type=ZIP_STORED,
            )
    return output.getvalue()


def _record_bytes(contents: Mapping[str, bytes]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name in sorted((*contents, FLASHINFER_RECORD_MEMBER)):
        if name == FLASHINFER_RECORD_MEMBER:
            writer.writerow((name, "", ""))
        else:
            content = contents[name]
            digest = base64.urlsafe_b64encode(sha256(content).digest()).rstrip(b"=")
            writer.writerow((name, f"sha256={digest.decode('ascii')}", str(len(content))))
    return output.getvalue().encode("utf-8")


def _require_patched_closure(
    source: _Inspection,
    patched: _Inspection,
    target_bytes: bytes,
) -> None:
    source_map = source.member_map()
    patched_map = patched.member_map()
    if set(source_map) != set(patched_map) or len(patched.members) != len(source.members):
        raise RuntimeError("patched FlashInfer member inventory differs")
    expected_order = [
        *sorted(name for name in source_map if name != FLASHINFER_RECORD_MEMBER),
        FLASHINFER_RECORD_MEMBER,
    ]
    if [member.name for member in patched.members] != expected_order:
        raise RuntimeError("patched FlashInfer member order differs")
    for name in sorted(source_map):
        observed = patched_map[name]
        if observed.timestamp != _FIXED_ZIP_TIMESTAMP:
            raise RuntimeError(f"patched FlashInfer timestamp differs for {name!r}")
        if (
            observed.create_system != 3
            or observed.external_attr != _canonical_external_attr(source_map[name].mode)
            or observed.internal_attr != 0
            or observed.flag_bits != 0
            or observed.compression != ZIP_STORED
        ):
            raise RuntimeError(f"patched FlashInfer ZIP metadata differs for {name!r}")
        if name not in {FLASHINFER_TARGET_MEMBER, FLASHINFER_RECORD_MEMBER} and (
            observed.content != source_map[name].content
        ):
            raise RuntimeError(f"unapproved FlashInfer member changed: {name!r}")
    if patched_map[FLASHINFER_TARGET_MEMBER].content != target_bytes:
        raise RuntimeError("patched FlashInfer target bytes differ")
    if (
        patched_map[FLASHINFER_TARGET_MEMBER].sha256
        != FLASHINFER_TARGET_PATCHED_SHA256
        or len(patched_map[FLASHINFER_TARGET_MEMBER].content)
        != FLASHINFER_TARGET_PATCHED_SIZE
    ):
        raise RuntimeError("patched FlashInfer target identity differs")


def _manifest_record(
    *,
    source: _Inspection,
    patched: _Inspection,
    wheel_name: str,
    wheel_sha256: str,
    wheel_size: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record_type": FLASHINFER_PATCHED_WHEEL_MANIFEST_RECORD_TYPE,
        "schema_version": FLASHINFER_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION,
        "closed_record_sha256": "",
        "package": {
            "name": FLASHINFER_PACKAGE_NAME,
            "version": FLASHINFER_PACKAGE_VERSION,
            "requires_python": ">=3.10,<4.0",
            "wheel_tag": "py3-none-any",
        },
        "source_wheel": {
            "filename": FLASHINFER_SOURCE_WHEEL_FILENAME,
            "sha256": FLASHINFER_SOURCE_WHEEL_SHA256,
            "bytes": FLASHINFER_SOURCE_WHEEL_SIZE,
            "member_count": len(source.members),
            "content_tree_sha256": source.content_tree_sha256,
            "archive_tree_sha256": source.archive_tree_sha256,
            "total_uncompressed_bytes": source.total_uncompressed_bytes,
            "total_compressed_bytes": source.total_compressed_bytes,
            "record_sha256": source.record_sha256,
            "record_bytes": source.record_size,
        },
        "patch": {
            "id": "flashinfer-python311-future-annotations",
            "relative_path": FLASHINFER_TARGET_MEMBER,
            "source_sha256": FLASHINFER_TARGET_SOURCE_SHA256,
            "source_bytes": FLASHINFER_TARGET_SOURCE_SIZE,
            "insertion_offset": FLASHINFER_TARGET_PATCH_OFFSET,
            "inserted_utf8": FLASHINFER_TARGET_PATCH_BYTES.decode("utf-8"),
            "inserted_bytes": len(FLASHINFER_TARGET_PATCH_BYTES),
            "prefix_sha256": FLASHINFER_TARGET_PREFIX_SHA256,
            "suffix_sha256": FLASHINFER_TARGET_SUFFIX_SHA256,
            "patched_sha256": FLASHINFER_TARGET_PATCHED_SHA256,
            "patched_bytes": FLASHINFER_TARGET_PATCHED_SIZE,
            "ast_change": "one __future__.annotations import only",
            "reason": (
                "CPython 3.11 evaluates array.array[int] at import time; future "
                "annotations preserves the source annotation without subscripting "
                "array.array during FlashInfer engine initialization."
            ),
        },
        "patched_wheel": {
            "filename": wheel_name,
            "sha256": wheel_sha256,
            "bytes": wheel_size,
            "member_count": len(patched.members),
            "content_tree_sha256": patched.content_tree_sha256,
            "archive_tree_sha256": patched.archive_tree_sha256,
            "total_uncompressed_bytes": patched.total_uncompressed_bytes,
            "total_compressed_bytes": patched.total_compressed_bytes,
            "record_sha256": patched.record_sha256,
            "record_bytes": patched.record_size,
        },
        "deterministic_build": {
            "algorithm": "cachet-flashinfer-wheel-repack-v1",
            "python_implementation": _EXPECTED_PYTHON_IMPLEMENTATION,
            "python_version": _EXPECTED_PYTHON_VERSION,
            "member_order": "UTF-8 lexical with RECORD last",
            "timestamp": "1980-01-01T00:00:00",
            "compression": "STORED",
            "compression_level": None,
            "regular_modes": "0644 or 0755 from source execute bit",
            "archive_comment": "",
            "member_extra_fields": "",
            "independent_build_count": 2,
        },
    }
    unsigned = dict(record)
    unsigned.pop("closed_record_sha256")
    record["closed_record_sha256"] = _canonical_json_sha256(unsigned)
    return record


def _validate_manifest_record(record: Mapping[str, Any]) -> None:
    expected_keys = {
        "closed_record_sha256",
        "deterministic_build",
        "package",
        "patch",
        "patched_wheel",
        "record_type",
        "schema_version",
        "source_wheel",
    }
    if set(record) != expected_keys:
        raise ValueError("patched FlashInfer manifest keys differ")
    if (
        record.get("record_type") != FLASHINFER_PATCHED_WHEEL_MANIFEST_RECORD_TYPE
        or record.get("schema_version")
        != FLASHINFER_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("patched FlashInfer manifest schema differs")
    observed = _required_sha256(
        record.get("closed_record_sha256"),
        "manifest.closed_record_sha256",
    )
    unsigned = dict(record)
    unsigned.pop("closed_record_sha256")
    if _canonical_json_sha256(unsigned) != observed:
        raise ValueError("patched FlashInfer manifest closed digest differs")


def _inspection_summary(inspection: _Inspection) -> Mapping[str, Any]:
    return {
        "member_count": len(inspection.members),
        "content_tree_sha256": inspection.content_tree_sha256,
        "archive_tree_sha256": inspection.archive_tree_sha256,
        "total_uncompressed_bytes": inspection.total_uncompressed_bytes,
        "total_compressed_bytes": inspection.total_compressed_bytes,
        "record_sha256": inspection.record_sha256,
        "record_bytes": inspection.record_size,
    }


def _content_tree_sha256(members: Sequence[_Member]) -> str:
    value = [
        [member.name, member.sha256, len(member.content)]
        for member in sorted(members, key=lambda item: item.name)
    ]
    return _canonical_json_sha256(value)


def _archive_tree_sha256(members: Sequence[_Member]) -> str:
    value = [
        {
            "compressed_size": member.compressed_size,
            "compression": member.compression,
            "crc32": f"{member.crc32:08x}",
            "create_system": member.create_system,
            "external_attr": member.external_attr,
            "flag_bits": member.flag_bits,
            "is_directory": False,
            "mode": f"{member.mode:06o}",
            "name": member.name,
            "sha256": member.sha256,
            "size": len(member.content),
            "timestamp": list(member.timestamp),
        }
        for member in sorted(members, key=lambda item: item.name)
    ]
    return _canonical_json_sha256(value)


def _patched_wheel_name(digest: str) -> str:
    _required_sha256(digest, "patched wheel digest")
    return (
        "flashinfer_python-0.6.16.post3-"
        f"1cachetpy311{digest[:16]}-py3-none-any.whl"
    )


def _canonical_external_attr(source_mode: int) -> int:
    permissions = 0o755 if source_mode & 0o111 else 0o644
    return (stat.S_IFREG | permissions) << 16


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


def _canonical_manifest_file_bytes(value: Mapping[str, Any]) -> bytes:
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


def _canonical_manifest_from_bytes(raw: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("patched FlashInfer manifest is not canonical JSON") from exc
    if not isinstance(value, dict) or _canonical_manifest_file_bytes(value) != raw:
        raise ValueError("patched FlashInfer manifest bytes are noncanonical")
    return value


def _read_exact_regular_file(
    path: Path,
    *,
    expected_name: str,
    expected_sha256: str,
    expected_size: int,
    label: str,
) -> bytes:
    if path.name != expected_name:
        raise ValueError(f"{label} filename differs")
    _require_no_symlink_path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} must be one regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max(expected_size, 64 * 1024 * 1024):
                raise ValueError(f"{label} exceeds its size authority")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"{label} changed while reading")
    content = b"".join(chunks)
    if len(content) != expected_size or sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"{label} size or SHA-256 differs")
    return content


def _require_no_symlink_path(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"path contains symlink component: {current}")


def _require_or_create_private_directory(path: Path) -> None:
    _require_no_symlink_path(path.parent)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise ValueError("FlashInfer output root must be a regular directory")
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


def _publish_identical(source: Path, target: Path) -> None:
    content = source.read_bytes()
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
            raise FileExistsError(f"artifact target exists with different bytes: {target}")
        return
    _exclusive_write(target, content)


def _validate_build_files(source: Path, wheel: Path, manifest: Path) -> None:
    expected = _build_reviewed_wheel(source)
    if wheel.read_bytes() != expected.wheel_bytes or manifest.read_bytes() != (
        expected.manifest_bytes
    ):
        raise RuntimeError("on-disk FlashInfer build validation failed")


def _require_governed_builder() -> None:
    if (
        platform.python_implementation() != _EXPECTED_PYTHON_IMPLEMENTATION
        or platform.python_version() != _EXPECTED_PYTHON_VERSION
    ):
        raise RuntimeError(
            "FlashInfer repack requires governed CPython 3.11.16"
        )


def _require_pinned_derived_outputs(
    *,
    wheel_sha256: str,
    wheel_size: int,
    manifest_closed_record_sha256: str,
    manifest_file_sha256: str,
) -> None:
    expected = (
        FLASHINFER_PATCHED_WHEEL_SHA256,
        FLASHINFER_PATCHED_WHEEL_SIZE,
        FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256,
        FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
    )
    observed = (
        wheel_sha256,
        wheel_size,
        manifest_closed_record_sha256,
        manifest_file_sha256,
    )
    if observed != expected:
        raise RuntimeError("derived FlashInfer artifact pins differ")


def _required_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _required_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _required_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


__all__ = [
    "FLASHINFER_METADATA_MEMBER",
    "FLASHINFER_METADATA_SHA256",
    "FLASHINFER_PACKAGE_NAME",
    "FLASHINFER_PACKAGE_VERSION",
    "FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256",
    "FLASHINFER_PATCHED_MANIFEST_FILE_SHA256",
    "FLASHINFER_PATCHED_MANIFEST_FILENAME",
    "FLASHINFER_PATCHED_MANIFEST_SIZE",
    "FLASHINFER_PATCHED_RECORD_SHA256",
    "FLASHINFER_PATCHED_RECORD_SIZE",
    "FLASHINFER_PATCHED_WHEEL_FILENAME",
    "FLASHINFER_PATCHED_WHEEL_MANIFEST_RECORD_TYPE",
    "FLASHINFER_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION",
    "FLASHINFER_PATCHED_WHEEL_SHA256",
    "FLASHINFER_PATCHED_WHEEL_SIZE",
    "FLASHINFER_RECORD_MEMBER",
    "FLASHINFER_SOURCE_CONTENT_TREE_SHA256",
    "FLASHINFER_SOURCE_WHEEL_FILENAME",
    "FLASHINFER_SOURCE_WHEEL_SHA256",
    "FLASHINFER_SOURCE_WHEEL_SIZE",
    "FLASHINFER_TARGET_MEMBER",
    "FLASHINFER_TARGET_PATCHED_SHA256",
    "FLASHINFER_TARGET_PATCHED_SIZE",
    "FLASHINFER_TARGET_PATCH_OFFSET",
    "FLASHINFER_TARGET_SOURCE_SHA256",
    "FLASHINFER_TARGET_SOURCE_SIZE",
    "FLASHINFER_WHEEL_MEMBER",
    "PatchedFlashInferWheel",
    "repack_flashinfer_0616_post3_wheel",
    "validate_patched_flashinfer_wheel",
    "validate_pristine_flashinfer_wheel",
]
