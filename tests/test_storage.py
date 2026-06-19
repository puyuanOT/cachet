import hashlib
from dataclasses import replace

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
