import pytest

from vllm_kv_injection.block_mapping import (
    BlockSpan,
    map_segments_to_blocks,
    map_segments_to_reserved_blocks,
    plan_token_blocks,
)
from vllm_kv_injection.protocol import KVSegment


def test_plan_token_blocks_handles_partial_tail():
    spans = plan_token_blocks(total_tokens=10, block_size=4, starting_block_id=7)

    assert [(span.block_id, span.token_start, span.token_count) for span in spans] == [
        (7, 0, 4),
        (8, 4, 4),
        (9, 8, 2),
    ]


def test_map_segment_across_block_boundary():
    segment = KVSegment(
        document_id="doc-a",
        chunk_type="document_chunk",
        chunk_id="review-a",
        token_start=2,
        token_count=6,
        byte_start=0,
        byte_length=60,
    )

    mapping = map_segments_to_blocks((segment,), block_size=4, starting_block_id=10)

    segment_key = (0, "doc-a", "document_chunk", "review-a", "")

    assert [(span.block_id, span.block_offset, span.token_count) for span in mapping[segment_key]] == [
        (10, 2, 2),
        (11, 0, 4),
    ]
    assert [(span.block_id, span.block_offset, span.token_count) for span in mapping["review-a"]] == [
        (10, 2, 2),
        (11, 0, 4),
    ]
    assert len(mapping) == 1
    assert list(mapping.keys()) == [segment_key]


def test_map_segments_uses_document_aware_keys():
    first = KVSegment(
        document_id="doc-a",
        chunk_type="document_static",
        chunk_id="static",
        token_start=0,
        token_count=2,
        byte_start=0,
        byte_length=20,
    )
    second = KVSegment(
        document_id="doc-b",
        chunk_type="document_static",
        chunk_id="static",
        token_start=2,
        token_count=2,
        byte_start=20,
        byte_length=20,
    )

    mapping = map_segments_to_blocks((first, second), block_size=4)

    assert set(mapping) == {
        (0, "doc-a", "document_static", "static", ""),
        (1, "doc-b", "document_static", "static", ""),
    }
    assert mapping.segment_keys() == (
        (0, "doc-a", "document_static", "static", ""),
        (1, "doc-b", "document_static", "static", ""),
    )
    assert mapping.get("static") is None


def test_map_segments_to_reserved_blocks_uses_actual_physical_block_ids():
    segment = KVSegment(
        document_id="doc-a",
        chunk_type="document_chunk",
        chunk_id="review-a",
        token_start=2,
        token_count=5,
        byte_start=0,
        byte_length=50,
    )
    blocks = (
        BlockSpan(block_id=99, token_start=0, token_count=4, block_offset=0),
        BlockSpan(block_id=42, token_start=4, token_count=4, block_offset=0),
    )

    mapping = map_segments_to_reserved_blocks((segment,), blocks)

    assert [(span.block_id, span.block_offset, span.token_count) for span in mapping["review-a"]] == [
        (99, 2, 2),
        (42, 0, 3),
    ]


def test_map_segments_to_reserved_blocks_rejects_non_contiguous_logical_blocks():
    segment = KVSegment(
        document_id="doc-a",
        chunk_type="document_chunk",
        chunk_id="review-a",
        token_start=0,
        token_count=2,
        byte_start=0,
        byte_length=20,
    )
    blocks = (BlockSpan(block_id=99, token_start=1, token_count=4, block_offset=0),)

    with pytest.raises(ValueError, match="contiguous"):
        map_segments_to_reserved_blocks((segment,), blocks)


def test_legacy_segment_constructor_still_maps_by_unique_chunk_id():
    segment = KVSegment("static", 0, 2, 0, 20)

    mapping = map_segments_to_blocks((segment,), block_size=4)

    assert (0, "__legacy__", "legacy_chunk", "static", "") in mapping
    assert "static" in mapping
    assert mapping["static"] == mapping[(0, "__legacy__", "legacy_chunk", "static", "")]
    assert list(mapping.unique_chunk_ids()) == ["static"]
    assert len(mapping) == 1
