import hashlib
import importlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from textwrap import dedent

import pytest

from document_kv_cache.engine_protocol import KVStorageLayout
from document_kv_cache.kvpack import PackChunk, write_kvpack
from document_kv_cache.models import ChunkRef, DocumentChunkType, KVCacheKey
from document_kv_cache.storage import (
    DiskRangeReader,
    MemoryRangeReader,
    RoutedRangeReader,
    UnityCatalogVolumeRangeReader,
    local_path,
    unity_catalog_volume_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def key(chunk_id: str) -> KVCacheKey:
    return KVCacheKey.for_document(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_id="doc-1",
        chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
        chunk_id=chunk_id,
    )


def chunk_ref(**overrides) -> ChunkRef:
    values = {
        "key": key("a"),
        "shard_uri": "shard.kvpack",
        "byte_offset": 0,
        "byte_length": 1,
        "token_count": 1,
        "dtype": "int8",
        "layout_version": "v1",
        "checksum": hashlib.sha256(b"x").hexdigest(),
        "storage_layout": KVStorageLayout.SEPARATE_KEY_VALUE,
    }
    values.update(overrides)
    return ChunkRef(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"shard_uri": ""}, "shard_uri"),
        ({"byte_offset": -1}, "byte_offset"),
        ({"byte_offset": 1.5}, "byte_offset"),
        ({"byte_offset": True}, "byte_offset"),
        ({"byte_length": 0}, "byte_length"),
        ({"byte_length": "1"}, "byte_length"),
        ({"byte_length": True}, "byte_length"),
        ({"token_count": 0}, "token_count"),
        ({"token_count": "1"}, "token_count"),
        ({"token_count": True}, "token_count"),
        ({"dtype": ""}, "dtype"),
        ({"dtype": 123}, "dtype"),
        ({"layout_version": ""}, "layout_version"),
        ({"layout_version": 123}, "layout_version"),
        ({"storage_layout": "packed"}, "storage_layout"),
        ({"checksum": "abc"}, "checksum"),
        ({"checksum": "A" * 64}, "checksum"),
        ({"checksum": 123}, "checksum"),
    ),
)
def test_chunk_ref_rejects_invalid_storage_metadata(overrides, message):
    with pytest.raises(ValueError, match=message):
        chunk_ref(**overrides)


def test_memory_range_reader_reads_validated_byte_ranges(tmp_path):
    shard_path = tmp_path / "memory-source.kvpack"
    ref = write_kvpack(
        shard_path,
        [PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    reader = MemoryRangeReader({ref.shard_uri: shard_path.read_bytes()})

    assert reader.read(ref) == b"alpha"


def test_disk_range_reader_resolves_relative_paths_under_root(tmp_path):
    shard_dir = tmp_path / "shards"
    ref = write_kvpack(
        shard_dir / "disk.kvpack",
        [PackChunk(key=key("a"), payload=b"disk", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    relative_ref = replace(ref, shard_uri="disk.kvpack")

    assert DiskRangeReader(root=shard_dir).read(relative_ref) == b"disk"


def test_local_path_resolves_file_and_dbfs_uri_forms(tmp_path):
    file_path = tmp_path / "shard.kvpack"

    assert local_path(f"disk:{file_path}") == file_path
    assert local_path("disk:shard.kvpack", root=tmp_path) == tmp_path / "shard.kvpack"
    assert local_path(f"file:{file_path}") == file_path
    assert local_path("dbfs:/benchmarks/shard.kvpack").as_posix() == "/dbfs/benchmarks/shard.kvpack"


@pytest.mark.parametrize(
    "uri",
    (
        "dbfs:/../etc/passwd",
        "dbfs:/benchmarks/../secret.kvpack",
        "dbfs:/benchmarks//secret.kvpack",
        "dbfs:/benchmarks/./secret.kvpack",
    ),
)
def test_local_path_rejects_dbfs_paths_that_escape_root(uri):
    with pytest.raises(ValueError, match="cannot contain"):
        local_path(uri)


def test_unity_catalog_volume_reader_resolves_relative_paths_under_volume_root(tmp_path):
    volume_root = tmp_path / "Volumes" / "catalog" / "schema" / "volume"
    ref = write_kvpack(
        volume_root / "uc.kvpack",
        [PackChunk(key=key("a"), payload=b"uc-volume", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    relative_ref = replace(ref, shard_uri="uc.kvpack")

    assert UnityCatalogVolumeRangeReader(volume_root=volume_root).read(relative_ref) == b"uc-volume"
    assert unity_catalog_volume_path("uc-volume:/catalog/schema/volume/uc.kvpack").as_posix() == (
        "/Volumes/catalog/schema/volume/uc.kvpack"
    )
    assert local_path("uc-volume:/catalog/schema/volume/uc.kvpack").as_posix() == (
        "/Volumes/catalog/schema/volume/uc.kvpack"
    )
    assert unity_catalog_volume_path("uc-volume://catalog/schema/volume/uc.kvpack").as_posix() == (
        "/Volumes/catalog/schema/volume/uc.kvpack"
    )


@pytest.mark.parametrize(
    "uri",
    (
        "uc-volume:/../../etc/passwd",
        "uc-volume:/catalog/schema/volume/../secret.kvpack",
        "uc-volume:/catalog/schema/volume//secret.kvpack",
        "/Volumes/catalog/schema/volume/../secret.kvpack",
        "/tmp/Volumes/catalog/schema/volume/secret.kvpack",
    ),
)
def test_unity_catalog_volume_path_rejects_escape_or_non_uc_paths(uri):
    with pytest.raises(ValueError, match="/Volumes|cannot contain"):
        unity_catalog_volume_path(uri)


def test_unity_catalog_volume_reader_rejects_relative_paths_that_escape_root(tmp_path):
    with pytest.raises(ValueError, match="cannot contain"):
        unity_catalog_volume_path("../secret.kvpack", root=tmp_path / "Volumes" / "catalog" / "schema" / "volume")


def test_routed_range_reader_dispatches_memory_and_disk(tmp_path):
    disk_ref = write_kvpack(
        tmp_path / "routed.kvpack",
        [PackChunk(key=key("disk"), payload=b"disk", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    memory_ref = replace(disk_ref, shard_uri="memory:routed")
    memory = MemoryRangeReader({"memory:routed": b"disk"})
    reader = RoutedRangeReader(memory=memory, disk=DiskRangeReader())

    assert reader.read(memory_ref) == b"disk"
    assert reader.read(disk_ref) == b"disk"


def test_routed_range_reader_dispatches_relative_paths_to_configured_uc_root(tmp_path):
    volume_root = tmp_path / "Volumes" / "catalog" / "schema" / "volume"
    ref = write_kvpack(
        volume_root / "routed-uc.kvpack",
        [PackChunk(key=key("uc"), payload=b"uc-routed", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    relative_ref = replace(ref, shard_uri="routed-uc.kvpack")
    reader = RoutedRangeReader(unity_catalog=UnityCatalogVolumeRangeReader(volume_root=volume_root))

    assert reader.read(relative_ref) == b"uc-routed"


def test_routed_range_reader_disk_uri_overrides_configured_uc_root(tmp_path):
    disk_root = tmp_path / "local-disk"
    volume_root = tmp_path / "Volumes" / "catalog" / "schema" / "volume"
    ref = write_kvpack(
        disk_root / "local.kvpack",
        [PackChunk(key=key("local"), payload=b"local-routed", token_count=2, dtype="int8", layout_version="v1")],
        align_bytes=1,
    )[0]
    disk_ref = replace(ref, shard_uri="disk:local.kvpack")
    reader = RoutedRangeReader(
        disk=DiskRangeReader(root=disk_root),
        unity_catalog=UnityCatalogVolumeRangeReader(volume_root=volume_root),
    )

    assert reader.read(disk_ref) == b"local-routed"


def test_storage_public_and_legacy_modules_have_separate_ownership():
    public_storage = importlib.import_module("document_kv_cache.storage")
    legacy_storage = importlib.import_module("restaurant_kv_serving.storage")

    assert public_storage.DiskRangeReader.__module__ == "document_kv_cache.storage"
    assert legacy_storage.DiskRangeReader.__module__ == "restaurant_kv_serving.storage"
    assert issubclass(legacy_storage.MemoryRangeReader, public_storage.MemoryRangeReader)
    assert issubclass(legacy_storage.DiskRangeReader, public_storage.DiskRangeReader)
    assert issubclass(legacy_storage.UnityCatalogVolumeRangeReader, public_storage.UnityCatalogVolumeRangeReader)
    assert issubclass(legacy_storage.RoutedRangeReader, public_storage.RoutedRangeReader)
    assert legacy_storage.RangeReader.__module__ == "restaurant_kv_serving.storage"


def test_storage_star_import_surfaces_are_curated_for_document_and_preserved_for_legacy():
    public_namespace: dict[str, object] = {}
    legacy_namespace: dict[str, object] = {}

    exec("from document_kv_cache.storage import *", public_namespace)
    exec("from restaurant_kv_serving.storage import *", legacy_namespace)

    assert set(public_namespace) >= {
        "RangeReader",
        "MemoryRangeReader",
        "DiskRangeReader",
        "UnityCatalogVolumeRangeReader",
        "RoutedRangeReader",
        "local_path",
        "unity_catalog_volume_path",
        "is_real_uc_volume_root",
    }
    assert "hashlib" not in public_namespace
    assert set(legacy_namespace) >= {
        "hashlib",
        "Mapping",
        "Path",
        "PurePosixPath",
        "Protocol",
        "ChunkRef",
        "RangeReader",
        "MemoryRangeReader",
        "DiskRangeReader",
        "UnityCatalogVolumeRangeReader",
        "RoutedRangeReader",
        "local_path",
        "unity_catalog_volume_path",
        "is_real_uc_volume_root",
    }
    assert "_document_module" not in legacy_namespace


def test_legacy_storage_uses_legacy_helper_overrides(monkeypatch):
    legacy_storage = importlib.import_module("restaurant_kv_serving.storage")

    def fake_join_confined(root, raw_relative_path, *, label):
        assert root == Path("/dbfs")
        assert raw_relative_path == "benchmarks/shard.kvpack"
        assert label == "dbfs"
        return Path("/legacy-override/shard.kvpack")

    monkeypatch.setattr(legacy_storage, "_join_confined", fake_join_confined)

    assert legacy_storage.local_path("dbfs:/benchmarks/shard.kvpack") == Path("/legacy-override/shard.kvpack")


def test_legacy_storage_class_methods_ignore_public_class_overrides(monkeypatch):
    public_storage = importlib.import_module("document_kv_cache.storage")
    legacy_storage = importlib.import_module("restaurant_kv_serving.storage")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("legacy class leaked through public class method")

    monkeypatch.setattr(public_storage.MemoryRangeReader, "__init__", fail_if_called)
    monkeypatch.setattr(public_storage.MemoryRangeReader, "put", fail_if_called)
    monkeypatch.setattr(public_storage.DiskRangeReader, "__init__", fail_if_called)
    monkeypatch.setattr(public_storage.UnityCatalogVolumeRangeReader, "__init__", fail_if_called)

    memory_reader = legacy_storage.MemoryRangeReader()
    memory_reader.put("memory:payload", b"payload")
    disk_reader = legacy_storage.DiskRangeReader(root="/tmp")
    uc_reader = legacy_storage.UnityCatalogVolumeRangeReader(volume_root="/Volumes/catalog/schema/volume")

    assert memory_reader._blobs == {"memory:payload": b"payload"}
    assert disk_reader.root == Path("/tmp")
    assert uc_reader.root == Path("/Volumes/catalog/schema/volume")


def test_legacy_storage_ignores_public_helper_overrides_when_imported_later():
    script = dedent(
        """
        import importlib

        public_storage = importlib.import_module("document_kv_cache.storage")

        def fail_if_called(*args, **kwargs):
            raise AssertionError("legacy wrapper leaked through public helper")

        public_storage._join_confined = fail_if_called
        legacy_storage = importlib.import_module("restaurant_kv_serving.storage")
        assert legacy_storage.local_path("dbfs:/benchmarks/shard.kvpack").as_posix() == "/dbfs/benchmarks/shard.kvpack"
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
