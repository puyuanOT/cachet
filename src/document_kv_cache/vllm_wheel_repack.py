"""Deterministically repack the official vLLM 0.27.1 cu129 wheel.

The benchmark runtime must never mutate installed ``site-packages``.  This
module verifies the official release asset, applies Cachet's three approved
E5M2 source changes before installation, rebuilds wheel ``RECORD`` metadata,
and emits a content-addressed wheel plus a machine-readable closure manifest.
"""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import platform
from pathlib import Path, PurePosixPath
import zlib
from typing import Any, Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from document_kv_cache.serving_env import (
    VLLM_PACKAGE_VERSION,
    VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_LOCK_FILENAME,
    VLLM_RUNTIME_LOCK_SHA256,
    VLLM_VERSION,
    VLLM_WHEEL_FILENAME,
    VLLM_WHEEL_SHA256,
)
from document_kv_cache.vllm_runtime_contract_data import (
    VLLM_KV_CONNECTOR_V1_BASE_SOURCE_SHA256,
)

VLLM_PATCHED_WHEEL_MANIFEST_RECORD_TYPE = (
    "document_kv.vllm_patched_wheel_manifest.v2"
)
VLLM_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION = 2

# Full-file hashes are for the exact sources shipped by the official v0.27.1
# cu129 wheel.  Target hashes cover every byte after deterministic replacement.
VLLM_0271_E5M2_PATCH_CLOSURE: tuple[Mapping[str, object], ...] = (
    {
        "id": "vllm-qwen-attention-disable-e5m2-query-quant",
        "relative_path": "vllm/model_executor/layers/attention/attention.py",
        "source_sha256": "dae6d1f09448adc1d67c776089e77e8b75378332166844a234cd8fa45c18195e",
        "patched_sha256": "5735acfb390cf344caeec950c2f286344bcd84721ce287e0a56701f2a18bc839",
        "replacements": (
            (
                """        if (
            self.impl.supports_quant_query_input
            and (
                self.kv_cache_dtype.startswith("fp8") or self.kv_cache_dtype == "nvfp4"
            )
            and not self.kv_cache_dtype.endswith("per_token_head")
        ):
""",
                """        if (
            self.impl.supports_quant_query_input
            and self.kv_cache_dtype in {"fp8", "fp8_e4m3", "nvfp4"}
        ):
""",
            ),
        ),
        "reason": (
            "vLLM 0.27.1 constructs an E4M3-only query quantizer for "
            "fp8_e5m2; E5M2 queries must stay in the model compute dtype."
        ),
    },
    {
        "id": "vllm-triton-reshape-cache-e5m2-closure",
        "relative_path": "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py",
        "source_sha256": "6cac51475b8c656992a21b2d150acd3a16a95a7ab7d49aab151a3ef13c24b80d",
        "patched_sha256": "0682ca7bc56edf7cea5419188a81c78510b54192471472b160aa447ac0ceeb08",
        "replacements": (
            (
                """    if kv_cache_dtype.startswith("fp8"):
        return current_platform.has_device_capability(89) or current_platform.is_xpu()
""",
                """    if kv_cache_dtype == "fp8_e5m2":
        return current_platform.has_device_capability(80) or current_platform.is_xpu()
    if kv_cache_dtype.startswith("fp8"):
        return current_platform.has_device_capability(89) or current_platform.is_xpu()
""",
            ),
            (
                """    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if is_quantized_kv_cache(kv_cache_dtype)
        else key_cache.dtype
    )
""",
                """    kv_cache_torch_dtype = (
        torch.float8_e5m2
        if kv_cache_dtype == "fp8_e5m2"
        else current_platform.fp8_dtype()
        if is_quantized_kv_cache(kv_cache_dtype)
        else key_cache.dtype
    )
""",
            ),
            (
                """    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if is_quantized_kv_cache(kv_cache_dtype)
        else kv_cache.dtype
    )
""",
                """    kv_cache_torch_dtype = (
        torch.float8_e5m2
        if kv_cache_dtype == "fp8_e5m2"
        else current_platform.fp8_dtype()
        if is_quantized_kv_cache(kv_cache_dtype)
        else kv_cache.dtype
    )
""",
            ),
        ),
        "reason": (
            "vLLM 0.27.1 otherwise views fp8_e5m2 pages through the platform "
            "E4M3 dtype and rejects the SM80-SM88 E5M2 software path."
        ),
    },
    {
        "id": "vllm-triton-attention-e5m2-closure",
        "relative_path": "vllm/v1/attention/backends/triton_attn.py",
        "source_sha256": "20b4dd5f8c15cd2d6f9598268368f5fc572f2c980eb75327c5992100c49ff3ed",
        "patched_sha256": "4dae0ff6c4ee8f11c1f195151a11673d595d457c413032e7bae7550913f94390",
        "replacements": (
            (
                """            if self.kv_cache_dtype.startswith("fp8") and not (
                current_platform.has_device_capability(89)
            ):
""",
                """            if self.kv_cache_dtype == "fp8_e5m2":
                fp8_kv_supported = current_platform.has_device_capability(80)
            else:
                fp8_kv_supported = current_platform.has_device_capability(89)
            if self.kv_cache_dtype.startswith("fp8") and not fp8_kv_supported:
""",
            ),
            (
                """        self.fp8_dtype = current_platform.fp8_dtype()
""",
                """        self.fp8_dtype = (
            torch.float8_e5m2
            if kv_cache_dtype == "fp8_e5m2"
            else current_platform.fp8_dtype()
        )
""",
            ),
        ),
        "reason": (
            "vLLM 0.27.1 otherwise binds Triton KV views to E4M3 and rejects "
            "fp8_e5m2 on A10G despite the targeted SM80+ E5M2 path."
        ),
    },
)


@dataclass(frozen=True, slots=True)
class PatchedVLLMWheel:
    wheel_path: Path
    manifest_path: Path
    wheel_sha256: str
    wheel_size: int
    source_wheel_sha256: str


def repack_vllm_0271_cu129_wheel(
    source_wheel: str | Path,
    output_dir: str | Path,
    *,
    expected_source_wheel_sha256: str = VLLM_WHEEL_SHA256,
    patch_closure: Sequence[Mapping[str, object]] = VLLM_0271_E5M2_PATCH_CLOSURE,
) -> PatchedVLLMWheel:
    """Create a deterministic, content-addressed E5M2-patched vLLM wheel."""

    source_path = Path(source_wheel)
    if not source_path.is_file():
        raise FileNotFoundError(f"vLLM source wheel does not exist: {source_path}")
    source_digest = _file_sha256(source_path)
    if source_digest != expected_source_wheel_sha256:
        raise ValueError(
            f"vLLM source wheel SHA-256 {source_digest} does not match approved "
            f"cu129 asset {expected_source_wheel_sha256}"
        )
    if source_path.name != VLLM_WHEEL_FILENAME and expected_source_wheel_sha256 == VLLM_WHEEL_SHA256:
        raise ValueError(
            f"approved vLLM source wheel filename must be {VLLM_WHEEL_FILENAME!r}"
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_path = output_root / ".vllm-0.27.1-cu129-cachet-repack.tmp.whl"
    temporary_path.unlink(missing_ok=True)
    closure_records: list[dict[str, object]] = []
    patches_by_member = _patches_by_member(patch_closure)
    try:
        with ZipFile(source_path, "r") as source_archive:
            infos = source_archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("vLLM source wheel contains duplicate archive members")
            for name in names:
                _validate_wheel_member_name(name)
            signature_members = [
                name
                for name in names
                if name.endswith(("/RECORD.jws", "/RECORD.p7s"))
            ]
            if signature_members:
                raise ValueError(
                    "signed wheel RECORD files cannot be preserved after repacking: "
                    f"{signature_members!r}"
                )
            record_names = [
                name for name in names if name.endswith(".dist-info/RECORD")
            ]
            if len(record_names) != 1:
                raise ValueError(
                    "vLLM source wheel must contain exactly one dist-info RECORD, "
                    f"found {record_names!r}"
                )
            record_name = record_names[0]
            info_by_name = {
                info.filename: info for info in infos if not info.is_dir()
            }
            missing_patch_members = sorted(
                set(patches_by_member) - set(info_by_name)
            )
            if missing_patch_members:
                raise ValueError(
                    "vLLM source wheel is missing patch members "
                    f"{missing_patch_members!r}"
                )
            record_entries = _stream_repacked_wheel(
                source_archive,
                temporary_path,
                info_by_name=info_by_name,
                record_name=record_name,
                patches_by_member=patches_by_member,
                closure_records=closure_records,
            )
            if len(record_entries) != len(info_by_name) - 1:
                raise RuntimeError("repacked vLLM wheel did not cover every file member")
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    target_digest = _file_sha256(temporary_path)
    target_name = _content_addressed_wheel_name(source_path.name, target_digest)
    target_path = output_root / target_name
    if target_path.exists():
        if _file_sha256(target_path) != target_digest:
            raise FileExistsError(
                f"content-addressed wheel path already exists with different bytes: {target_path}"
            )
        temporary_path.unlink()
    else:
        temporary_path.replace(target_path)

    manifest = {
        "record_type": VLLM_PATCHED_WHEEL_MANIFEST_RECORD_TYPE,
        "schema_version": VLLM_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION,
        "base_version": VLLM_VERSION,
        "package_version": VLLM_PACKAGE_VERSION,
        "source_wheel_filename": source_path.name,
        "source_wheel_sha256": source_digest,
        "patched_wheel_filename": target_path.name,
        "patched_wheel_sha256": target_digest,
        "patched_wheel_size": target_path.stat().st_size,
        "record_member": record_name,
        "build_toolchain": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "zlib_compile_version": zlib.ZLIB_VERSION,
            "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        },
        "patch_closure": closure_records,
        "connector_base_source_sha256": VLLM_KV_CONNECTOR_V1_BASE_SOURCE_SHA256,
        "runtime_lock": {
            "filename": VLLM_RUNTIME_LOCK_FILENAME,
            "sha256": VLLM_RUNTIME_LOCK_SHA256,
            "locked_distribution_count": VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
            "excluded_package": "vllm",
        },
    }
    manifest_path = target_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PatchedVLLMWheel(
        wheel_path=target_path,
        manifest_path=manifest_path,
        wheel_sha256=target_digest,
        wheel_size=target_path.stat().st_size,
        source_wheel_sha256=source_digest,
    )


def validate_patched_vllm_member_bytes(
    member_name: str,
    content: bytes,
    *,
    patch_closure: Sequence[Mapping[str, object]] = VLLM_0271_E5M2_PATCH_CLOSURE,
) -> str:
    """Return the approved patched digest or fail closed for one member."""

    matching = [
        patch
        for patch in patch_closure
        if patch.get("relative_path") == member_name
    ]
    if len(matching) != 1:
        raise ValueError(f"no unique approved vLLM patch member for {member_name!r}")
    expected = _required_patch_digest(matching[0], "patched_sha256")
    observed = sha256(content).hexdigest()
    if observed != expected:
        raise ValueError(
            f"installed vLLM member {member_name} SHA-256 {observed} does not "
            f"match approved patched source {expected}"
        )
    return observed


def _apply_patch_replacements(
    source_bytes: bytes,
    patch: Mapping[str, object],
) -> bytes:
    text = source_bytes.decode("utf-8")
    replacements = patch.get("replacements")
    if not isinstance(replacements, tuple):
        raise TypeError("vLLM patch replacements must be a tuple")
    for replacement in replacements:
        if (
            not isinstance(replacement, tuple)
            or len(replacement) != 2
            or not all(isinstance(value, str) for value in replacement)
        ):
            raise TypeError("vLLM patch replacement must be an (old, new) string tuple")
        old, new = replacement
        if text.count(old) != 1:
            raise ValueError(
                f"approved replacement for {_required_patch_string(patch, 'id')} "
                f"must match exactly once, found {text.count(old)}"
            )
        text = text.replace(old, new, 1)
    return text.encode("utf-8")


def _patches_by_member(
    patch_closure: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    patches: dict[str, Mapping[str, object]] = {}
    for patch in patch_closure:
        member_name = _required_patch_string(patch, "relative_path")
        _validate_wheel_member_name(member_name)
        if member_name in patches:
            raise ValueError(f"vLLM patch closure repeats {member_name!r}")
        patches[member_name] = patch
    if not patches:
        raise ValueError("vLLM patch closure must not be empty")
    return patches


def _stream_repacked_wheel(
    source_archive: ZipFile,
    output_path: Path,
    *,
    info_by_name: Mapping[str, ZipInfo],
    record_name: str,
    patches_by_member: Mapping[str, Mapping[str, object]],
    closure_records: list[dict[str, object]],
) -> dict[str, tuple[str, int]]:
    record_entries: dict[str, tuple[str, int]] = {}
    closure_by_member: dict[str, dict[str, object]] = {}
    with ZipFile(
        output_path,
        "w",
        compression=ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as output_archive:
        output_archive.comment = b""
        for member_name in sorted(info_by_name):
            if member_name == record_name:
                continue
            source_info = info_by_name[member_name]
            target_info = _deterministic_zip_info(source_info)
            patch = patches_by_member.get(member_name)
            if patch is None:
                member_digest = sha256()
                member_size = 0
                with source_archive.open(source_info, "r") as source_stream:
                    with output_archive.open(target_info, "w") as target_stream:
                        for chunk in iter(
                            lambda: source_stream.read(4 * 1024 * 1024),
                            b"",
                        ):
                            member_digest.update(chunk)
                            member_size += len(chunk)
                            target_stream.write(chunk)
                record_entries[member_name] = (
                    _record_digest_from_hash(member_digest),
                    member_size,
                )
                continue

            source_bytes = source_archive.read(source_info)
            expected_source_digest = _required_patch_digest(
                patch,
                "source_sha256",
            )
            observed_source_digest = sha256(source_bytes).hexdigest()
            if observed_source_digest != expected_source_digest:
                raise ValueError(
                    f"vLLM wheel member {member_name} SHA-256 "
                    f"{observed_source_digest} does not match approved pristine "
                    f"source {expected_source_digest}"
                )
            patched_bytes = _apply_patch_replacements(source_bytes, patch)
            patched_digest = sha256(patched_bytes).hexdigest()
            expected_patched_digest = _required_patch_digest(
                patch,
                "patched_sha256",
            )
            if patched_digest != expected_patched_digest:
                raise RuntimeError(
                    f"patched vLLM wheel member {member_name} SHA-256 "
                    f"{patched_digest} does not match approved target "
                    f"{expected_patched_digest}"
                )
            with output_archive.open(target_info, "w") as target_stream:
                target_stream.write(patched_bytes)
            record_entries[member_name] = (
                _record_digest(patched_bytes),
                len(patched_bytes),
            )
            closure_by_member[member_name] = {
                "id": _required_patch_string(patch, "id"),
                "relative_path": member_name,
                "source_sha256": observed_source_digest,
                "patched_sha256": patched_digest,
                "reason": _required_patch_string(patch, "reason"),
            }

        record_bytes = _wheel_record_bytes(
            record_entries,
            record_name=record_name,
        )
        with output_archive.open(
            _deterministic_zip_info(info_by_name[record_name]),
            "w",
        ) as record_stream:
            record_stream.write(record_bytes)
    closure_records.extend(
        closure_by_member[member_name] for member_name in patches_by_member
    )
    return record_entries


def _wheel_record_bytes(
    record_entries: Mapping[str, tuple[str, int]],
    *,
    record_name: str,
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name in sorted((*record_entries, record_name)):
        if name == record_name:
            writer.writerow((name, "", ""))
            continue
        digest, size = record_entries[name]
        writer.writerow((name, digest, str(size)))
    return output.getvalue().encode("utf-8")


def _record_digest(content: bytes) -> str:
    return _record_digest_from_hash(sha256(content))


def _record_digest_from_hash(digest: Any) -> str:
    encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}"


def _deterministic_zip_info(source_info: ZipInfo) -> ZipInfo:
    info = ZipInfo(source_info.filename, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = source_info.compress_type
    info.create_system = source_info.create_system
    info.external_attr = source_info.external_attr
    info.internal_attr = source_info.internal_attr
    info.flag_bits = source_info.flag_bits & 0x800
    if info.compress_type == ZIP_DEFLATED:
        info._compresslevel = 9
    return info


def _content_addressed_wheel_name(source_name: str, digest: str) -> str:
    suffix = "-cp38-abi3-manylinux_2_28_x86_64.whl"
    if not source_name.endswith(suffix):
        raise ValueError(f"unexpected vLLM cu129 wheel tags in {source_name!r}")
    prefix = source_name[: -len(suffix)]
    return f"{prefix}-1cachete5m2{digest[:16]}{suffix}"


def _validate_wheel_member_name(name: str) -> None:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise ValueError(f"unsafe wheel archive member {name!r}")


def _required_patch_string(patch: Mapping[str, object], field_name: str) -> str:
    value = patch.get(field_name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"vLLM patch {field_name} must be a non-empty string")
    return value


def _required_patch_digest(patch: Mapping[str, object], field_name: str) -> str:
    value = _required_patch_string(patch, field_name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"vLLM patch {field_name} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the content-addressed Cachet vLLM 0.27.1 cu129 wheel."
    )
    parser.add_argument("source_wheel", help="Downloaded official cu129 release wheel")
    parser.add_argument("output_dir", help="Directory for the patched wheel and manifest")
    args = parser.parse_args(argv)
    result = repack_vllm_0271_cu129_wheel(args.source_wheel, args.output_dir)
    print(
        json.dumps(
            {
                "wheel_path": str(result.wheel_path),
                "manifest_path": str(result.manifest_path),
                "wheel_sha256": result.wheel_sha256,
                "wheel_size": result.wheel_size,
                "source_wheel_sha256": result.source_wheel_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PatchedVLLMWheel",
    "VLLM_PATCHED_WHEEL_MANIFEST_RECORD_TYPE",
    "VLLM_PATCHED_WHEEL_MANIFEST_SCHEMA_VERSION",
    "main",
    "repack_vllm_0271_cu129_wheel",
    "validate_patched_vllm_member_bytes",
]
