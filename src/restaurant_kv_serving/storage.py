"""Compatibility wrapper for :mod:`document_kv_cache.storage`."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Protocol

import document_kv_cache.storage as _document_module
from restaurant_kv_serving.models import ChunkRef


class RangeReader(Protocol):
    def read(self, ref: ChunkRef) -> bytes: ...


class MemoryRangeReader(_document_module.MemoryRangeReader):
    """Read cache shards from process memory."""

    def __init__(self, blobs: Mapping[str, bytes] | None = None) -> None:
        self._blobs = dict(blobs or {})

    def put(self, shard_uri: str, payload: bytes) -> None:
        self._blobs[shard_uri] = bytes(payload)

    def read(self, ref: ChunkRef) -> bytes:
        try:
            blob = self._blobs[ref.shard_uri]
        except KeyError as exc:
            raise FileNotFoundError(f"Missing in-memory shard {ref.shard_uri!r}") from exc
        payload = blob[ref.byte_offset : ref.byte_offset + ref.byte_length]
        return _validated_payload(ref, payload)


class DiskRangeReader(_document_module.DiskRangeReader):
    """Read cache shards from local disk, `disk:` URIs, or mounted paths."""

    def __init__(self, *, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else None

    def read(self, ref: ChunkRef) -> bytes:
        path = local_path(ref.shard_uri, root=self.root)
        with path.open("rb") as handle:
            handle.seek(ref.byte_offset)
            payload = handle.read(ref.byte_length)
        return _validated_payload(ref, payload)


class UnityCatalogVolumeRangeReader(DiskRangeReader, _document_module.UnityCatalogVolumeRangeReader):
    """Read shards from Databricks Unity Catalog Volumes."""

    def __init__(self, *, volume_root: str | Path | None = None) -> None:
        super().__init__(root=volume_root)

    def read(self, ref: ChunkRef) -> bytes:
        path = unity_catalog_volume_path(ref.shard_uri, root=self.root)
        with path.open("rb") as handle:
            handle.seek(ref.byte_offset)
            payload = handle.read(ref.byte_length)
        return _validated_payload(ref, payload)


class RoutedRangeReader(_document_module.RoutedRangeReader):
    """Dispatch reads by URI shape across memory, disk, and UC Volume readers."""

    def __init__(
        self,
        *,
        memory: MemoryRangeReader | None = None,
        disk: DiskRangeReader | None = None,
        unity_catalog: UnityCatalogVolumeRangeReader | None = None,
    ) -> None:
        self.memory = memory or MemoryRangeReader()
        self.disk = disk or DiskRangeReader()
        self.unity_catalog = unity_catalog or UnityCatalogVolumeRangeReader()

    def read(self, ref: ChunkRef) -> bytes:
        if ref.shard_uri.startswith("memory:") or ref.shard_uri.startswith("mem:"):
            return self.memory.read(ref)
        if ref.shard_uri.startswith("disk:"):
            return self.disk.read(ref)
        if ref.shard_uri.startswith("uc-volume:") or ref.shard_uri.startswith("/Volumes/"):
            return self.unity_catalog.read(ref)
        if self.unity_catalog.root is not None and _is_relative_uri(ref.shard_uri):
            return self.unity_catalog.read(ref)
        return self.disk.read(ref)


def local_path(uri: str, *, root: str | Path | None = None) -> Path:
    if uri.startswith("disk:"):
        return local_path(uri[len("disk:") :], root=root)
    if uri.startswith("file:"):
        return Path(uri[len("file:") :])
    if uri.startswith("dbfs:/"):
        return _join_confined(Path("/dbfs"), uri[len("dbfs:/") :], label="dbfs")
    if uri.startswith("uc-volume:"):
        return unity_catalog_volume_path(uri, root=root)
    path = Path(uri)
    if uri == "/Volumes" or uri.startswith("/Volumes/"):
        return unity_catalog_volume_path(uri, root=root)
    if path.is_absolute() or root is None:
        return path
    return _join_confined(Path(root), uri, label="disk")


def unity_catalog_volume_path(uri: str, *, root: str | Path | None = None) -> Path:
    if uri.startswith("uc-volume:"):
        return _validate_absolute_uc_path(
            _join_confined(Path("/Volumes"), uri[len("uc-volume:") :], label="uc-volume"),
            label="uc-volume",
        )
    path = Path(uri)
    if path.is_absolute():
        return _validate_absolute_uc_path(path, label="UC Volume path")
    if root is None:
        return _validate_absolute_uc_path(
            _join_confined(Path("/Volumes"), uri, label="UC Volume path"),
            label="UC Volume path",
        )
    return _join_confined(Path(root), uri, label="UC Volume relative path")


def is_real_uc_volume_root(value: str | Path | None) -> bool | None:
    """Return whether ``value`` is a root-confined Databricks UC Volume path."""

    if value is None:
        return None
    try:
        path = _validate_absolute_uc_path(Path(value), label="UC Volume root")
    except (TypeError, ValueError):
        return False
    return len(path.parts) >= 5


def _is_relative_uri(uri: str) -> bool:
    if ":" in uri:
        return False
    return not Path(uri).is_absolute()


def _join_confined(root: Path, raw_relative_path: str, *, label: str) -> Path:
    relative_path = _safe_relative_posix_path(raw_relative_path, label=label)
    path = root.joinpath(*relative_path.parts)
    try:
        path.relative_to(root)
    except ValueError as exc:  # pragma: no cover - protected by relative path validation.
        raise ValueError(f"{label} path must remain under {root}") from exc
    return path


def _safe_relative_posix_path(raw_path: str, *, label: str) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path must be non-empty")
    stripped_path = raw_path.lstrip("/")
    if not stripped_path:
        raise ValueError(f"{label} path must be non-empty")
    raw_parts = stripped_path.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"{label} path cannot contain empty, '.', or '..' components")
    return PurePosixPath(*raw_parts)


def _validate_absolute_uc_path(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute /Volumes/<catalog>/<schema>/<volume> path")
    parts = path.parts
    if len(parts) < 5 or parts[:2] != ("/", "Volumes"):
        raise ValueError(f"{label} must be under /Volumes/<catalog>/<schema>/<volume>")
    if any(part in {"", ".", ".."} for part in parts[2:]):
        raise ValueError(f"{label} cannot contain empty, '.', or '..' components")
    return path


def _validated_payload(ref: ChunkRef, payload: bytes) -> bytes:
    if len(payload) != ref.byte_length:
        raise ValueError(
            f"Short read for {ref.key.storage_key()}: expected {ref.byte_length} bytes, got {len(payload)}"
        )
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != ref.checksum:
        raise ValueError(f"Checksum mismatch for {ref.key.storage_key()}: expected {ref.checksum}, got {checksum}")
    return payload
