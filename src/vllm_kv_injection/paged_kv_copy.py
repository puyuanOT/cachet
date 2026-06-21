from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from vllm_kv_injection.block_mapping import BlockSpan

PagedKVLayout = Literal["standard", "flat"]


def slot_mapping_from_blocks(
    blocks: Sequence[BlockSpan],
    *,
    block_size: int,
    device: object | None = None,
) -> object:
    """Return vLLM slot indices for already-allocated physical KV blocks.

    vLLM's standard paged KV layout addresses a token slot as
    ``physical_block_id * block_size + block_offset``. The resulting tensor is
    ordered by logical token position, so it can be used directly to copy a
    materialized document KV payload into vLLM-owned paged buffers.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    _validate_logical_blocks(blocks, block_size=block_size)

    torch = _torch()
    slot_ids: list[int] = []
    for block in blocks:
        start = block.block_id * block_size + block.block_offset
        slot_ids.extend(range(start, start + block.token_count))
    return torch.tensor(slot_ids, dtype=torch.long, device=device)


def inject_kv_cache_layer(
    dst_kv_cache_layer: object,
    src_kv_cache: object,
    slot_mapping: object,
    *,
    block_size: int,
    layout: PagedKVLayout | None = None,
) -> None:
    """Copy a materialized KV layer into a vLLM paged KV cache layer.

    Supported layouts:
    - ``standard``: ``[num_blocks, 2, block_size, ...]`` for normal K/V pages.
    - ``flat``: ``[num_blocks, block_size, ...]`` for MLA-like pages.

    ``layout=None`` infers ``standard`` when the second dimension is the K/V
    pair dimension, otherwise ``flat``.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    torch = _torch()
    if not torch.is_tensor(dst_kv_cache_layer):
        raise TypeError("dst_kv_cache_layer must be a torch.Tensor")
    if not torch.is_tensor(src_kv_cache):
        raise TypeError("src_kv_cache must be a torch.Tensor")
    if not torch.is_tensor(slot_mapping):
        raise TypeError("slot_mapping must be a torch.Tensor")
    if slot_mapping.ndim != 1:
        raise ValueError("slot_mapping must be one-dimensional")
    slot_mapping = slot_mapping.to(device=dst_kv_cache_layer.device, dtype=torch.long)
    if src_kv_cache.shape[0] != slot_mapping.numel():
        raise ValueError("src_kv_cache first dimension must match slot_mapping length")

    inferred_layout = layout or _infer_paged_kv_layout(dst_kv_cache_layer)
    _validate_slot_mapping_range(slot_mapping, dst_kv_cache_layer, block_size=block_size)
    if inferred_layout == "standard":
        _inject_standard_kv_layer(
            dst_kv_cache_layer,
            src_kv_cache,
            slot_mapping,
            block_size=block_size,
        )
        return
    if inferred_layout == "flat":
        _inject_flat_kv_layer(dst_kv_cache_layer, src_kv_cache, slot_mapping, block_size=block_size)
        return
    raise ValueError(f"Unsupported paged KV layout: {inferred_layout!r}")


def _infer_paged_kv_layout(dst_kv_cache_layer: Any) -> PagedKVLayout:
    if dst_kv_cache_layer.ndim >= 4 and dst_kv_cache_layer.shape[1] == 2:
        return "standard"
    if dst_kv_cache_layer.ndim >= 3:
        return "flat"
    raise ValueError("dst_kv_cache_layer has unsupported rank for a paged KV cache")


def _inject_standard_kv_layer(
    dst_kv_cache_layer: Any,
    src_kv_cache: Any,
    slot_mapping: Any,
    *,
    block_size: int,
) -> None:
    if dst_kv_cache_layer.ndim < 4 or dst_kv_cache_layer.shape[1] != 2:
        raise ValueError("standard paged KV layout must be [num_blocks, 2, block_size, ...]")
    if dst_kv_cache_layer.shape[2] != block_size:
        raise ValueError("dst_kv_cache_layer block dimension does not match block_size")
    expected_shape = (slot_mapping.numel(), 2, *dst_kv_cache_layer.shape[3:])
    if tuple(src_kv_cache.shape) != expected_shape:
        raise ValueError(f"src_kv_cache shape must be {expected_shape} for standard paged KV layout")
    block_indices = slot_mapping // block_size
    block_offsets = slot_mapping % block_size
    dst_kv_cache_layer[block_indices, :, block_offsets] = src_kv_cache.to(
        device=dst_kv_cache_layer.device,
        dtype=dst_kv_cache_layer.dtype,
    )


def _inject_flat_kv_layer(
    dst_kv_cache_layer: Any,
    src_kv_cache: Any,
    slot_mapping: Any,
    *,
    block_size: int,
) -> None:
    if dst_kv_cache_layer.ndim < 3:
        raise ValueError("flat paged KV layout must be [num_blocks, block_size, ...]")
    if dst_kv_cache_layer.shape[1] != block_size:
        raise ValueError("dst_kv_cache_layer block dimension does not match block_size")
    expected_shape = (slot_mapping.numel(),) + tuple(dst_kv_cache_layer.shape[2:])
    if tuple(src_kv_cache.shape) != expected_shape:
        raise ValueError(f"src_kv_cache shape must be {expected_shape} for flat paged KV layout")
    flattened = dst_kv_cache_layer.reshape(
        dst_kv_cache_layer.shape[0] * dst_kv_cache_layer.shape[1],
        *dst_kv_cache_layer.shape[2:],
    )
    flattened[slot_mapping] = src_kv_cache.to(
        device=dst_kv_cache_layer.device,
        dtype=dst_kv_cache_layer.dtype,
    )


def _validate_slot_mapping_range(slot_mapping: Any, dst_kv_cache_layer: Any, *, block_size: int) -> None:
    if slot_mapping.numel() == 0:
        return
    if bool((slot_mapping < 0).any().item()):
        raise ValueError("slot_mapping must not contain negative or padded slot ids")
    slot_capacity = dst_kv_cache_layer.shape[0] * block_size
    if bool((slot_mapping >= slot_capacity).any().item()):
        raise ValueError("slot_mapping contains slot ids outside the destination KV cache")


def _validate_logical_blocks(blocks: Sequence[BlockSpan], *, block_size: int) -> None:
    cursor = 0
    for block in blocks:
        if block.token_start != cursor:
            raise ValueError("Blocks must cover a contiguous logical token range")
        if block.token_count < 0:
            raise ValueError("Block token_count must be non-negative")
        if block.block_id < 0:
            raise ValueError("Block block_id must be non-negative")
        if block.block_offset < 0:
            raise ValueError("Block block_offset must be non-negative")
        if block.block_offset >= block_size:
            raise ValueError("Block block_offset must be smaller than block_size")
        if block.block_offset + block.token_count > block_size:
            raise ValueError("Block token span must fit inside one physical block")
        cursor += block.token_count


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency.
        raise RuntimeError("paged KV copy helpers require torch at runtime") from exc
    return torch
