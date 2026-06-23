import importlib

import pytest

from document_kv_cache.engine_protocol import KVStorageLayout
from document_kv_cache.kvpack import LocalRangeReader, PackChunk, write_kvpack, write_kvpack_bytes
from document_kv_cache.models import DocumentChunkType, KVCacheKey
from document_kv_cache.storage import MemoryRangeReader


def key(chunk_id: str) -> KVCacheKey:
    return KVCacheKey.for_document(
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        document_id="doc-1",
        chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
        chunk_id=chunk_id,
    )


def test_write_and_range_read_kvpack(tmp_path):
    refs = write_kvpack(
        tmp_path / "shard.kvpack",
        [
            PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="fp8", layout_version="v1"),
            PackChunk(key=key("b"), payload=b"bravo", token_count=3, dtype="fp8", layout_version="v1"),
        ],
        align_bytes=8,
    )

    reader = LocalRangeReader()

    assert [reader.read(ref) for ref in refs] == [b"alpha", b"bravo"]
    assert refs[1].byte_offset % 8 == 0
    assert [ref.storage_layout for ref in refs] == [
        KVStorageLayout.SEPARATE_KEY_VALUE,
        KVStorageLayout.SEPARATE_KEY_VALUE,
    ]


def test_write_kvpack_copies_storage_layout_to_manifest_refs(tmp_path):
    refs = write_kvpack(
        tmp_path / "separate.kvpack",
        [
            PackChunk(
                key=key("a"),
                payload=b"alpha",
                token_count=2,
                dtype="fp8",
                layout_version="v1",
                storage_layout=KVStorageLayout.SEPARATE_KEY_VALUE,
            )
        ],
        align_bytes=1,
    )

    assert refs[0].storage_layout == KVStorageLayout.SEPARATE_KEY_VALUE


def test_write_kvpack_bytes_builds_memory_readable_shard():
    shard_uri = "memory:shard"
    payload, refs = write_kvpack_bytes(
        shard_uri,
        [
            PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="fp8", layout_version="v1"),
            PackChunk(key=key("b"), payload=b"bravo", token_count=3, dtype="fp8", layout_version="v1"),
        ],
        align_bytes=8,
    )
    reader = MemoryRangeReader({shard_uri: payload})

    assert [reader.read(ref) for ref in refs] == [b"alpha", b"bravo"]
    assert refs[0].shard_uri == shard_uri
    assert refs[1].byte_offset % 8 == 0
    assert [ref.storage_layout for ref in refs] == [
        KVStorageLayout.SEPARATE_KEY_VALUE,
        KVStorageLayout.SEPARATE_KEY_VALUE,
    ]


def test_pack_chunk_rejects_invalid_payload_metadata():
    with pytest.raises(ValueError, match="payload"):
        PackChunk(key=key("empty"), payload=b"", token_count=1, dtype="fp8", layout_version="v1")

    with pytest.raises(ValueError, match="payload"):
        PackChunk(key=key("payload-type"), payload="abc", token_count=1, dtype="fp8", layout_version="v1")

    with pytest.raises(ValueError, match="token_count"):
        PackChunk(key=key("tokens"), payload=b"x", token_count=0, dtype="fp8", layout_version="v1")

    with pytest.raises(ValueError, match="token_count"):
        PackChunk(key=key("token-type"), payload=b"x", token_count="1", dtype="fp8", layout_version="v1")

    with pytest.raises(ValueError, match="token_count"):
        PackChunk(key=key("token-bool"), payload=b"x", token_count=True, dtype="fp8", layout_version="v1")

    with pytest.raises(ValueError, match="dtype"):
        PackChunk(key=key("dtype"), payload=b"x", token_count=1, dtype="", layout_version="v1")

    with pytest.raises(ValueError, match="dtype"):
        PackChunk(key=key("dtype-type"), payload=b"x", token_count=1, dtype=123, layout_version="v1")

    with pytest.raises(ValueError, match="layout_version"):
        PackChunk(key=key("layout"), payload=b"x", token_count=1, dtype="fp8", layout_version="")

    with pytest.raises(ValueError, match="layout_version"):
        PackChunk(key=key("layout-type"), payload=b"x", token_count=1, dtype="fp8", layout_version=123)

    with pytest.raises(ValueError, match="storage_layout"):
        PackChunk(
            key=key("storage-layout"),
            payload=b"x",
            token_count=1,
            dtype="fp8",
            layout_version="v1",
            storage_layout="packed",
        )


def test_pack_chunk_normalizes_bytes_like_payloads():
    chunk = PackChunk(key=key("bytearray"), payload=bytearray(b"abc"), token_count=1, dtype="fp8", layout_version="v1")

    assert chunk.payload == b"abc"
    assert isinstance(chunk.payload, bytes)


@pytest.mark.parametrize("align_bytes", (0, 1.5, True))
def test_write_kvpack_rejects_invalid_alignment_before_output_file_is_opened(tmp_path, align_bytes):
    output_path = tmp_path / "bad-align.kvpack"

    with pytest.raises(ValueError, match="align_bytes"):
        write_kvpack(
            output_path,
            [PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="fp8", layout_version="v1")],
            align_bytes=align_bytes,
        )

    assert not output_path.exists()


@pytest.mark.parametrize("align_bytes", (0, 1.5, True))
def test_write_kvpack_bytes_rejects_invalid_alignment(align_bytes):
    with pytest.raises(ValueError, match="align_bytes"):
        write_kvpack_bytes(
            "memory:bad-align",
            [PackChunk(key=key("a"), payload=b"alpha", token_count=2, dtype="fp8", layout_version="v1")],
            align_bytes=align_bytes,
        )


def test_invalid_pack_chunk_payload_fails_before_output_file_is_opened(tmp_path):
    output_path = tmp_path / "invalid-payload.kvpack"

    with pytest.raises(ValueError, match="payload"):
        write_kvpack(
            output_path,
            [PackChunk(key=key("invalid"), payload="abc", token_count=1, dtype="fp8", layout_version="v1")],
            align_bytes=1,
        )

    assert not output_path.exists()


def test_lazy_invalid_pack_chunk_fails_before_partial_output_is_opened(tmp_path):
    output_path = tmp_path / "lazy-invalid-payload.kvpack"

    def chunks():
        yield PackChunk(key=key("valid"), payload=b"alpha", token_count=2, dtype="fp8", layout_version="v1")
        yield PackChunk(key=key("invalid"), payload="abc", token_count=1, dtype="fp8", layout_version="v1")

    with pytest.raises(ValueError, match="payload"):
        write_kvpack(output_path, chunks(), align_bytes=1)

    assert not output_path.exists()
