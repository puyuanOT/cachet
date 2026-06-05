from restaurant_kv_serving.kvpack import LocalRangeReader, PackChunk, write_kvpack
from restaurant_kv_serving.models import ChunkType, KVCacheKey


def key(chunk_id: str) -> KVCacheKey:
    return KVCacheKey(
        model_id="qwen35-4b-w8a8",
        lora_id="selection",
        prompt_template_version="v1",
        restaurant_id="r1",
        chunk_type=ChunkType.REVIEW,
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

