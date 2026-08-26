from __future__ import annotations

import array
import base64
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import struct
import warnings
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

import document_kv_cache.flashinfer_wheel_repack as repack
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PATCHED_WHEEL_MANIFEST_RECORD_TYPE,
    FLASHINFER_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION,
    FLASHINFER_SOURCE_CONTENT_TREE_SHA256,
    FLASHINFER_SOURCE_WHEEL_SHA256,
    FLASHINFER_TARGET_PATCHED_SHA256,
    FLASHINFER_TARGET_PATCHED_SIZE,
    FLASHINFER_TARGET_PATCH_OFFSET,
    repack_flashinfer_0616_post3_wheel,
    validate_patched_flashinfer_wheel,
    validate_pristine_flashinfer_wheel,
)


_REPOSITORY_ROOT = Path(__file__).parents[1]
_PRISTINE_WHEEL = (
    _REPOSITORY_ROOT
    / "databricks-runs"
    / "_campaign-inputs"
    / "flashinfer-python-0.6.16.post3"
    / "sha256"
    / FLASHINFER_SOURCE_WHEEL_SHA256
    / repack.FLASHINFER_SOURCE_WHEEL_FILENAME
)
_METADATA = (
    b"Metadata-Version: 2.4\n"
    b"Name: flashinfer-python\n"
    b"Version: 0.6.16.post3\n"
    b"Requires-Python: <4.0,>=3.10\n"
)
_WHEEL = (
    b"Wheel-Version: 1.0\n"
    b"Generator: setuptools (0.6.16.post3)\n"
    b"Root-Is-Purelib: true\n"
    b"Tag: py3-none-any\n"
)


def _pristine_wheel() -> Path:
    if not _PRISTINE_WHEEL.is_file():
        pytest.skip("reviewed pristine FlashInfer wheel is not present")
    return _PRISTINE_WHEEL


def _record_digest(content: bytes) -> str:
    digest = base64.urlsafe_b64encode(sha256(content).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


def _record_bytes(members: dict[str, bytes]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name in sorted((*members, repack.FLASHINFER_RECORD_MEMBER)):
        if name == repack.FLASHINFER_RECORD_MEMBER:
            writer.writerow((name, "", ""))
        else:
            content = members[name]
            writer.writerow((name, _record_digest(content), str(len(content))))
    return output.getvalue().encode()


def _info(name: str, *, mode: int = 0o100644) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = mode << 16
    return info


def _synthetic_wheel(
    *,
    extra_members: tuple[tuple[str, bytes, int], ...] = (),
    record_transform: object | None = None,
    archive_comment: bytes = b"",
) -> bytes:
    members = {
        repack.FLASHINFER_TARGET_MEMBER: b'"""test"""\n\nimport array\n',
        repack.FLASHINFER_METADATA_MEMBER: _METADATA,
        repack.FLASHINFER_WHEEL_MEMBER: _WHEEL,
        **{name: content for name, content, _mode in extra_members},
    }
    record = _record_bytes(members)
    if record_transform is not None:
        assert callable(record_transform)
        record = record_transform(record)
    members[repack.FLASHINFER_RECORD_MEMBER] = record
    modes = {name: mode for name, _content, mode in extra_members}
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        archive.comment = archive_comment
        for name, content in members.items():
            archive.writestr(
                _info(name, mode=modes.get(name, 0o100644)),
                content,
                compresslevel=9,
            )
    return output.getvalue()


@pytest.fixture(scope="module")
def reviewed_build(tmp_path_factory: pytest.TempPathFactory):
    source = _pristine_wheel()
    output = tmp_path_factory.mktemp("flashinfer-reviewed-build")
    return repack_flashinfer_0616_post3_wheel(source, output)


def test_pristine_wheel_matches_complete_reviewed_inventory():
    observed = validate_pristine_flashinfer_wheel(_pristine_wheel())

    assert observed == {
        "archive_tree_sha256": repack.FLASHINFER_SOURCE_ARCHIVE_TREE_SHA256,
        "content_tree_sha256": FLASHINFER_SOURCE_CONTENT_TREE_SHA256,
        "member_count": 4_957,
        "record_bytes": 623_912,
        "record_sha256": repack.FLASHINFER_SOURCE_RECORD_SHA256,
        "total_compressed_bytes": 14_779_090,
        "total_uncompressed_bytes": 82_056_127,
    }


def test_exact_member_patch_only_adds_future_annotations():
    with ZipFile(_pristine_wheel()) as archive:
        source = archive.read(repack.FLASHINFER_TARGET_MEMBER)

    patched = repack._patched_target_bytes(source)

    assert len(patched) == FLASHINFER_TARGET_PATCHED_SIZE
    assert sha256(patched).hexdigest() == FLASHINFER_TARGET_PATCHED_SHA256
    assert patched[:FLASHINFER_TARGET_PATCH_OFFSET] == source[:FLASHINFER_TARGET_PATCH_OFFSET]
    assert patched[FLASHINFER_TARGET_PATCH_OFFSET + 35 :] == source[
        FLASHINFER_TARGET_PATCH_OFFSET:
    ]


def test_exact_cpython311_annotation_failure_and_patched_success():
    function = (
        "def _fd_ancillary(fd: int) -> "
        "tuple[tuple[int, int, array.array[int]]]:\n"
        "    return ()\n"
    )
    with pytest.raises(TypeError, match="array.array.*not subscriptable"):
        exec(compile(function, "fd_exchange.py", "exec", dont_inherit=True), {"array": array})

    namespace: dict[str, object] = {"array": array}
    exec(
        compile(
            "from __future__ import annotations\n" + function,
            "fd_exchange.py",
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    observed = namespace["_fd_ancillary"]
    assert callable(observed)
    assert all(
        isinstance(value, str)
        for value in getattr(observed, "__annotations__").values()
    )


def test_double_build_is_byte_identical_and_record_closed(reviewed_build):
    source = _pristine_wheel()
    validated = validate_patched_flashinfer_wheel(
        source,
        reviewed_build.wheel_path,
        reviewed_build.manifest_path,
    )
    manifest = json.loads(reviewed_build.manifest_path.read_text())

    assert validated["patched_wheel_sha256"] == reviewed_build.wheel_sha256
    assert manifest["record_type"] == FLASHINFER_PATCHED_WHEEL_MANIFEST_RECORD_TYPE
    assert manifest["schema_version"] == FLASHINFER_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION
    assert manifest["deterministic_build"]["independent_build_count"] == 2
    with ZipFile(reviewed_build.wheel_path) as archive:
        names = archive.namelist()
        members = {name: archive.read(name) for name in names}
        assert names[-1] == repack.FLASHINFER_RECORD_MEMBER
        assert names[:-1] == sorted(names[:-1])
        assert {info.date_time for info in archive.infolist()} == {
            (1980, 1, 1, 0, 0, 0)
        }
    rows = {
        row[0]: row[1:]
        for row in csv.reader(
            io.StringIO(members[repack.FLASHINFER_RECORD_MEMBER].decode())
        )
    }
    assert set(rows) == set(members)
    for name, content in members.items():
        expected = (
            ["", ""]
            if name == repack.FLASHINFER_RECORD_MEMBER
            else [_record_digest(content), str(len(content))]
        )
        assert rows[name] == expected


@pytest.mark.parametrize(
    "name",
    ("../escape", "/absolute", "a\\b", "C:/drive", "a/./alias"),
)
def test_archive_rejects_unsafe_member_paths(name):
    raw = _synthetic_wheel(extra_members=((name, b"bad", 0o100644),))
    with pytest.raises(ValueError, match="unsafe wheel member"):
        repack._inspect_wheel_bytes(raw, label="test wheel")


def test_archive_rejects_casefold_aliases():
    raw = _synthetic_wheel(
        extra_members=(
            ("flashinfer/a.py", b"a", 0o100644),
            ("flashinfer/A.py", b"b", 0o100644),
        )
    )
    with pytest.raises(ValueError, match="repeats or aliases"):
        repack._inspect_wheel_bytes(raw, label="test wheel")


def test_archive_rejects_duplicate_members():
    raw = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(raw, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(_info(repack.FLASHINFER_TARGET_MEMBER), b"one")
            archive.writestr(_info(repack.FLASHINFER_TARGET_MEMBER), b"two")
    with pytest.raises(ValueError, match="repeats or aliases"):
        repack._inspect_wheel_bytes(raw.getvalue(), label="test wheel")


def test_archive_rejects_symlink_member():
    raw = _synthetic_wheel(
        extra_members=(("flashinfer/link", b"target", 0o120777),)
    )
    with pytest.raises(ValueError, match="link or special"):
        repack._inspect_wheel_bytes(raw, label="test wheel")


def test_archive_rejects_encrypted_flag():
    raw = bytearray(_synthetic_wheel())
    local = raw.find(b"PK\x03\x04")
    central = raw.find(b"PK\x01\x02")
    struct.pack_into("<H", raw, local + 6, 1)
    struct.pack_into("<H", raw, central + 8, 1)
    with pytest.raises(ValueError, match="encrypted"):
        repack._inspect_wheel_bytes(bytes(raw), label="test wheel")


def test_archive_rejects_trailing_data_and_comment():
    with pytest.raises(ValueError, match="comment or trailing data"):
        repack._inspect_wheel_bytes(_synthetic_wheel() + b"tail", label="test wheel")
    with pytest.raises(ValueError, match="comment or trailing data"):
        repack._inspect_wheel_bytes(
            _synthetic_wheel(archive_comment=b"comment"),
            label="test wheel",
        )


def test_archive_rejects_compression_bomb():
    raw = _synthetic_wheel(
        extra_members=(("flashinfer/zeros.bin", b"\0" * (2 * 1024 * 1024), 0o100644),)
    )
    with pytest.raises(ValueError, match="compression ratio is unsafe"):
        repack._inspect_wheel_bytes(raw, label="test wheel")


def test_archive_rejects_record_hash_and_self_row_tamper():
    wrong_hash = _synthetic_wheel(
        record_transform=lambda value: value.replace(b"sha256=", b"sha256=A", 1)
    )
    with pytest.raises(ValueError, match="RECORD digest differs"):
        repack._inspect_wheel_bytes(wrong_hash, label="test wheel")
    wrong_self = _synthetic_wheel(
        record_transform=lambda value: value.replace(
            f"{repack.FLASHINFER_RECORD_MEMBER},,\n".encode(),
            f"{repack.FLASHINFER_RECORD_MEMBER},sha256={'a' * 43},1\n".encode(),
        )
    )
    with pytest.raises(ValueError, match="RECORD self row is not blank"):
        repack._inspect_wheel_bytes(wrong_self, label="test wheel")


def test_patch_rejects_source_or_offset_tamper():
    with ZipFile(_pristine_wheel()) as archive:
        source = archive.read(repack.FLASHINFER_TARGET_MEMBER)
    with pytest.raises(ValueError, match="patch source bytes differ"):
        repack._patched_target_bytes(source + b" ")
    changed = bytearray(source)
    changed[FLASHINFER_TARGET_PATCH_OFFSET] ^= 1
    with pytest.raises(ValueError, match="patch source bytes differ"):
        repack._patched_target_bytes(bytes(changed))


def test_manifest_or_wheel_tamper_is_rejected(reviewed_build, tmp_path):
    source = _pristine_wheel()
    manifest = json.loads(reviewed_build.manifest_path.read_text())
    manifest["patch"]["insertion_offset"] += 1
    bad_manifest = tmp_path / reviewed_build.manifest_path.name
    bad_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="size or SHA-256 differs"):
        validate_patched_flashinfer_wheel(
            source,
            reviewed_build.wheel_path,
            bad_manifest,
        )

    wheel_bytes = bytearray(reviewed_build.wheel_path.read_bytes())
    wheel_bytes[-1] ^= 1
    bad_wheel = tmp_path / reviewed_build.wheel_path.name
    bad_wheel.write_bytes(wheel_bytes)
    with pytest.raises(ValueError, match="size or SHA-256 differs"):
        validate_patched_flashinfer_wheel(source, bad_wheel, reviewed_build.manifest_path)
