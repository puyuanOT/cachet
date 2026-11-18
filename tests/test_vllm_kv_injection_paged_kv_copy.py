from __future__ import annotations

import pytest

from vllm_kv_injection.block_mapping import BlockSpan
from vllm_kv_injection.paged_kv_copy import (
    TRITON_PACKED_KV_LAYOUT,
    inject_kv_cache_layer,
    slot_mapping_from_blocks,
    validate_triton_packed_kv_cache_layer,
)

torch = pytest.importorskip("torch")


def test_slot_mapping_from_reserved_physical_blocks():
    blocks = (
        BlockSpan(block_id=2, token_start=0, token_count=2, block_offset=1),
        BlockSpan(block_id=0, token_start=2, token_count=3, block_offset=0),
    )

    slots = slot_mapping_from_blocks(blocks, block_size=4)

    assert slots.tolist() == [9, 10, 0, 1, 2]


def test_inject_triton_packed_layer_from_engine_independent_logical_source():
    # Destination is vLLM 0.27.1 logical [B,H,BS,2D]; source stays [T,2,H,D].
    dst = torch.zeros((3, 2, 4, 6), dtype=torch.float32)
    src = torch.arange(5 * 2 * 2 * 3, dtype=torch.float32).reshape(5, 2, 2, 3)
    slots = torch.tensor([9, 10, 0, 1, 2], dtype=torch.long)

    inject_kv_cache_layer(
        dst,
        src,
        slots,
        block_size=4,
        layout=TRITON_PACKED_KV_LAYOUT,
    )

    assert torch.equal(dst[2, :, 1, :3], src[0, 0])
    assert torch.equal(dst[2, :, 1, 3:], src[0, 1])
    assert torch.equal(dst[2, :, 2, :3], src[1, 0])
    assert torch.equal(dst[0, :, 0, :3], src[2, 0])
    assert torch.equal(dst[0, :, 0, 3:], src[2, 1])
    assert torch.count_nonzero(dst[1]) == 0


def test_inject_triton_packed_preserves_fp8_e5m2_raw_bytes():
    keys = torch.tensor([[[1.0, -2.0]], [[3.0, -4.0]]]).to(torch.float8_e5m2)
    values = torch.tensor([[[5.0, -6.0]], [[7.0, -8.0]]]).to(torch.float8_e5m2)
    src = torch.stack((keys.view(torch.uint8), values.view(torch.uint8)), dim=1)
    dst = torch.zeros((1, 1, 2, 4), dtype=torch.uint8)

    inject_kv_cache_layer(dst, src, torch.tensor([0, 1]), block_size=2)

    assert torch.equal(dst[0, 0, :, :2], keys.view(torch.uint8)[:, 0])
    assert torch.equal(dst[0, 0, :, 2:], values.view(torch.uint8)[:, 0])
    assert torch.equal(
        dst[0, 0, :, :2].view(torch.float8_e5m2).float(),
        keys[:, 0].float(),
    )
    assert torch.equal(
        dst[0, 0, :, 2:].view(torch.float8_e5m2).float(),
        values[:, 0].float(),
    )


def test_inject_triton_packed_preserves_bfloat16_values():
    src = torch.arange(16, dtype=torch.bfloat16).reshape(2, 2, 2, 2)
    dst = torch.zeros((1, 2, 2, 4), dtype=torch.bfloat16)

    inject_kv_cache_layer(dst, src, torch.tensor([0, 1]), block_size=2)

    assert torch.equal(dst[0, :, :, :2].transpose(0, 1), src[:, 0])
    assert torch.equal(dst[0, :, :, 2:].transpose(0, 1), src[:, 1])


def test_inject_triton_packed_respects_noncontiguous_nhd_physical_strides():
    # vLLM may expose logical [B,H,BS,2D] over physical [B,BS,H,2D].
    physical = torch.zeros((1, 2, 2, 4), dtype=torch.bfloat16)
    dst = physical.permute(0, 2, 1, 3)
    assert not dst.is_contiguous()
    src = torch.arange(16, dtype=torch.bfloat16).reshape(2, 2, 2, 2)

    inject_kv_cache_layer(dst, src, torch.tensor([0, 1]), block_size=2)

    assert torch.equal(physical[0, :, :, :2], src[:, 0])
    assert torch.equal(physical[0, :, :, 2:], src[:, 1])


def test_inject_kv_cache_layer_validates_source_token_count():
    dst = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    src = torch.zeros((1, 2, 1, 2), dtype=torch.float32)
    slots = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="first dimension"):
        inject_kv_cache_layer(dst, src, slots, block_size=4)


def test_inject_kv_cache_layer_rejects_negative_padded_slots():
    dst = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    src = torch.ones((2, 2, 1, 2), dtype=torch.float32)
    slots = torch.tensor([0, -1], dtype=torch.long)

    with pytest.raises(ValueError, match="negative"):
        inject_kv_cache_layer(dst, src, slots, block_size=4)
    assert torch.count_nonzero(dst) == 0


def test_inject_kv_cache_layer_rejects_slots_outside_destination_cache():
    dst = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    src = torch.ones((1, 2, 1, 2), dtype=torch.float32)
    slots = torch.tensor([4], dtype=torch.long)

    with pytest.raises(ValueError, match="outside"):
        inject_kv_cache_layer(dst, src, slots, block_size=4)
    assert torch.count_nonzero(dst) == 0


@pytest.mark.parametrize(
    "dst",
    [
        torch.zeros((1, 2, 4, 1, 2)),  # removed pre-0.26 K/V-axis ABI
        torch.zeros((1, 4, 3)),  # flat/MLA-like page
        torch.zeros((1, 1, 4, 5)),  # odd packed content
    ],
)
def test_inject_kv_cache_layer_fails_closed_on_unsupported_layouts(dst):
    src = torch.ones((1, 2, 1, 2))
    with pytest.raises(ValueError, match="vLLM 0.27.1 Triton|packed content"):
        inject_kv_cache_layer(dst, src, torch.tensor([0]), block_size=4)


def test_inject_kv_cache_layer_rejects_unapproved_layout_name():
    dst = torch.zeros((1, 1, 4, 4))
    src = torch.ones((1, 2, 1, 2))
    with pytest.raises(ValueError, match="Unsupported paged KV layout"):
        inject_kv_cache_layer(dst, src, torch.tensor([0]), block_size=4, layout="flat")


def test_validate_triton_packed_layer_checks_block_size():
    with pytest.raises(ValueError, match="block dimension"):
        validate_triton_packed_kv_cache_layer(
            torch.zeros((1, 1, 4, 4)),
            block_size=2,
        )


def test_slot_mapping_rejects_non_contiguous_logical_blocks():
    blocks = (BlockSpan(block_id=1, token_start=1, token_count=2, block_offset=0),)

    with pytest.raises(ValueError, match="contiguous"):
        slot_mapping_from_blocks(blocks, block_size=4)


def test_slot_mapping_rejects_block_span_outside_physical_page():
    blocks = (BlockSpan(block_id=1, token_start=0, token_count=2, block_offset=3),)

    with pytest.raises(ValueError, match="fit inside"):
        slot_mapping_from_blocks(blocks, block_size=4)
