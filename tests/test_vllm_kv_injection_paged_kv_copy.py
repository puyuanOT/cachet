from __future__ import annotations

import pytest

from vllm_kv_injection.block_mapping import BlockSpan
from vllm_kv_injection.paged_kv_copy import inject_kv_cache_layer, slot_mapping_from_blocks

torch = pytest.importorskip("torch")


def test_slot_mapping_from_reserved_physical_blocks():
    blocks = (
        BlockSpan(block_id=2, token_start=0, token_count=2, block_offset=1),
        BlockSpan(block_id=0, token_start=2, token_count=3, block_offset=0),
    )

    slots = slot_mapping_from_blocks(blocks, block_size=4)

    assert slots.tolist() == [9, 10, 0, 1, 2]


def test_inject_standard_paged_kv_layer():
    dst = torch.zeros((3, 2, 4, 2), dtype=torch.float32)
    src = torch.arange(5 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2)
    slots = torch.tensor([9, 10, 0, 1, 2], dtype=torch.long)

    inject_kv_cache_layer(dst, src, slots, block_size=4, layout="standard")

    assert torch.equal(dst[2, :, 1], src[0])
    assert torch.equal(dst[2, :, 2], src[1])
    assert torch.equal(dst[0, :, 0], src[2])
    assert torch.equal(dst[0, :, 1], src[3])
    assert torch.equal(dst[0, :, 2], src[4])
    assert torch.count_nonzero(dst[1]) == 0


def test_inject_flat_paged_kv_layer():
    dst = torch.zeros((3, 4, 3), dtype=torch.float32)
    src = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3)
    slots = torch.tensor([9, 10, 0, 1, 2], dtype=torch.long)

    inject_kv_cache_layer(dst, src, slots, block_size=4, layout="flat")

    assert torch.equal(dst[2, 1], src[0])
    assert torch.equal(dst[2, 2], src[1])
    assert torch.equal(dst[0, 0], src[2])
    assert torch.equal(dst[0, 1], src[3])
    assert torch.equal(dst[0, 2], src[4])
    assert torch.count_nonzero(dst[1]) == 0


def test_inject_kv_cache_layer_infers_layout():
    standard = torch.zeros((1, 2, 2, 1), dtype=torch.float32)
    standard_src = torch.ones((2, 2, 1), dtype=torch.float32)
    flat = torch.zeros((1, 2, 1), dtype=torch.float32)
    flat_src = torch.ones((2, 1), dtype=torch.float32)
    slots = torch.tensor([0, 1], dtype=torch.long)

    inject_kv_cache_layer(standard, standard_src, slots, block_size=2)
    inject_kv_cache_layer(flat, flat_src, slots, block_size=2)

    assert torch.count_nonzero(standard) == 4
    assert torch.count_nonzero(flat) == 2


def test_inject_kv_cache_layer_validates_source_token_count():
    dst = torch.zeros((1, 2, 4, 2), dtype=torch.float32)
    src = torch.zeros((1, 2, 2), dtype=torch.float32)
    slots = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="first dimension"):
        inject_kv_cache_layer(dst, src, slots, block_size=4)


def test_inject_kv_cache_layer_rejects_negative_padded_slots():
    dst = torch.zeros((1, 2, 4, 2), dtype=torch.float32)
    src = torch.ones((2, 2, 2), dtype=torch.float32)
    slots = torch.tensor([0, -1], dtype=torch.long)

    with pytest.raises(ValueError, match="negative"):
        inject_kv_cache_layer(dst, src, slots, block_size=4)
    assert torch.count_nonzero(dst) == 0


def test_inject_kv_cache_layer_rejects_slots_outside_destination_cache():
    dst = torch.zeros((1, 2, 4, 2), dtype=torch.float32)
    src = torch.ones((1, 2, 2), dtype=torch.float32)
    slots = torch.tensor([4], dtype=torch.long)

    with pytest.raises(ValueError, match="outside"):
        inject_kv_cache_layer(dst, src, slots, block_size=4)
    assert torch.count_nonzero(dst) == 0


def test_inject_flat_paged_kv_layer_validates_block_size():
    dst = torch.zeros((1, 4, 3), dtype=torch.float32)
    src = torch.ones((2, 3), dtype=torch.float32)
    slots = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="block dimension"):
        inject_kv_cache_layer(dst, src, slots, block_size=2, layout="flat")


def test_slot_mapping_rejects_non_contiguous_logical_blocks():
    blocks = (BlockSpan(block_id=1, token_start=1, token_count=2, block_offset=0),)

    with pytest.raises(ValueError, match="contiguous"):
        slot_mapping_from_blocks(blocks, block_size=4)


def test_slot_mapping_rejects_block_span_outside_physical_page():
    blocks = (BlockSpan(block_id=1, token_start=0, token_count=2, block_offset=3),)

    with pytest.raises(ValueError, match="fit inside"):
        slot_mapping_from_blocks(blocks, block_size=4)
