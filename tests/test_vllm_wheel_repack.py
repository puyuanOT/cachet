import base64
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from document_kv_cache.vllm_wheel_repack import (
    VLLM_PATCHED_WHEEL_MANIFEST_RECORD_TYPE,
    VLLM_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION,
    repack_vllm_0271_cu129_wheel,
)


SOURCE_WHEEL_NAME = (
    "vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
)
PATCH_PATHS = (
    "vllm/model_executor/layers/attention/attention.py",
    "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py",
    "vllm/v1/attention/backends/triton_attn.py",
)


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    return info


def _write_source_wheel(path: Path, sources: tuple[bytes, ...]) -> None:
    members = {
        **dict(zip(PATCH_PATHS, sources, strict=True)),
        "vllm-0.27.1+cu129.dist-info/METADATA": (
            b"Metadata-Version: 2.3\nName: vllm\nVersion: 0.27.1+cu129\n"
        ),
        "vllm-0.27.1+cu129.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
            b"Tag: cp38-abi3-manylinux_2_28_x86_64\n"
        ),
        "vllm-0.27.1+cu129.dist-info/RECORD": b"stale-record\n",
    }
    with ZipFile(path, "w") as archive:
        for name in sorted(members):
            archive.writestr(_zip_info(name), members[name], compresslevel=9)


def _closure(sources: tuple[bytes, ...]) -> tuple[dict[str, object], ...]:
    records = []
    for index, (member_name, source) in enumerate(
        zip(PATCH_PATHS, sources, strict=True)
    ):
        old = f"old-{index}"
        new = f"new-{index}"
        patched = source.decode().replace(old, new, 1).encode()
        records.append(
            {
                "id": f"patch-{index}",
                "relative_path": member_name,
                "source_sha256": sha256(source).hexdigest(),
                "patched_sha256": sha256(patched).hexdigest(),
                "replacements": ((old, new),),
                "reason": f"test patch {index}",
            }
        )
    return tuple(records)


def _file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _record_digest(content: bytes) -> str:
    encoded = base64.urlsafe_b64encode(sha256(content).digest()).rstrip(b"=")
    return f"sha256={encoded.decode()}"


def test_repack_is_deterministic_content_addressed_and_rebuilds_record(tmp_path):
    sources = tuple(f"prefix old-{index} suffix\n".encode() for index in range(3))
    source_wheel = tmp_path / SOURCE_WHEEL_NAME
    _write_source_wheel(source_wheel, sources)
    source_digest = _file_digest(source_wheel)
    closure = _closure(sources)

    first = repack_vllm_0271_cu129_wheel(
        source_wheel,
        tmp_path / "first",
        expected_source_wheel_sha256=source_digest,
        patch_closure=closure,
    )
    second = repack_vllm_0271_cu129_wheel(
        source_wheel,
        tmp_path / "second",
        expected_source_wheel_sha256=source_digest,
        patch_closure=closure,
    )

    assert first.wheel_sha256 == second.wheel_sha256
    assert first.wheel_path.name == second.wheel_path.name
    assert first.wheel_path.read_bytes() == second.wheel_path.read_bytes()
    assert f"1cachete5m2{first.wheel_sha256[:16]}" in first.wheel_path.name
    assert first.wheel_size == len(first.wheel_path.read_bytes())

    with ZipFile(first.wheel_path) as archive:
        members = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }
    for index, member_name in enumerate(PATCH_PATHS):
        assert f"new-{index}".encode() in members[member_name]
        assert f"old-{index}".encode() not in members[member_name]

    record_name = "vllm-0.27.1+cu129.dist-info/RECORD"
    record_rows = {
        row[0]: row[1:]
        for row in csv.reader(io.StringIO(members[record_name].decode()))
    }
    assert set(record_rows) == set(members)
    for name, content in members.items():
        if name == record_name:
            assert record_rows[name] == ["", ""]
        else:
            assert record_rows[name] == [_record_digest(content), str(len(content))]

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["record_type"] == VLLM_PATCHED_WHEEL_MANIFEST_RECORD_TYPE
    assert manifest["schema_version"] == VLLM_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION
    assert set(manifest["build_toolchain"]) == {
        "platform",
        "python_implementation",
        "python_version",
        "zlib_compile_version",
        "zlib_runtime_version",
    }
    assert manifest["source_wheel_sha256"] == source_digest
    assert manifest["patched_wheel_sha256"] == first.wheel_sha256
    assert [item["id"] for item in manifest["patch_closure"]] == [
        "patch-0",
        "patch-1",
        "patch-2",
    ]


def test_repack_rejects_unapproved_source_wheel_digest(tmp_path):
    sources = tuple(f"prefix old-{index} suffix\n".encode() for index in range(3))
    source_wheel = tmp_path / SOURCE_WHEEL_NAME
    _write_source_wheel(source_wheel, sources)

    with pytest.raises(ValueError, match="does not match approved cu129 asset"):
        repack_vllm_0271_cu129_wheel(
            source_wheel,
            tmp_path / "output",
            expected_source_wheel_sha256="0" * 64,
            patch_closure=_closure(sources),
        )


def test_repack_rejects_pristine_member_hash_mismatch(tmp_path):
    sources = tuple(f"prefix old-{index} suffix\n".encode() for index in range(3))
    source_wheel = tmp_path / SOURCE_WHEEL_NAME
    _write_source_wheel(source_wheel, sources)
    closure = list(_closure(sources))
    closure[1] = {**closure[1], "source_sha256": "f" * 64}

    with pytest.raises(ValueError, match="does not match approved pristine source"):
        repack_vllm_0271_cu129_wheel(
            source_wheel,
            tmp_path / "output",
            expected_source_wheel_sha256=_file_digest(source_wheel),
            patch_closure=tuple(closure),
        )
