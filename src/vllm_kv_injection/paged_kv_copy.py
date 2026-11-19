from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from vllm_kv_injection.block_mapping import BlockSpan


PagedKVLayout = Literal["triton_packed"]
TRITON_PACKED_KV_LAYOUT: PagedKVLayout = "triton_packed"


def slot_mapping_from_blocks(
    blocks: Sequence[BlockSpan],
    *,
    block_size: int,
    device: object | None = None,
) -> object:
    """Return vLLM slot indices for already-allocated physical KV blocks."""

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
    layout: PagedKVLayout = TRITON_PACKED_KV_LAYOUT,
    validate: bool = True,
) -> None:
    """Copy logical document KV into a vLLM 0.27.1 Triton cache page.

    Cachet artifacts remain engine-independent and expose one materialized layer
    as ``[token, K/V, kv_head, head_dim]``.  vLLM 0.27.1's Triton backend owns a
    packed logical page ``[block, kv_head, block_slot, 2 * head_dim]`` whose
    content dimension is ``[K | V]``.  This adapter is the only place where the
    two representations are coupled.

    No shape inference is performed.  In particular, the pre-0.26
    ``[block, K/V, block_slot, kv_head, head_dim]`` ABI and MLA-like flat pages
    are rejected instead of being guessed from ambiguous dimensions.

    ``validate`` runs the slot bounds check, which reads device tensors back to
    the host.  A caller reusing one mapping across layers may validate the first
    layer and pass ``validate=False`` for the remaining layers.
    """

    if layout != TRITON_PACKED_KV_LAYOUT:
        raise ValueError(f"Unsupported paged KV layout: {layout!r}")
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

    _validate_triton_packed_geometry(
        dst_kv_cache_layer,
        src_kv_cache,
        block_size=block_size,
    )
    if validate:
        _validate_slot_mapping_range(
            slot_mapping,
            dst_kv_cache_layer,
            block_size=block_size,
        )
    _inject_triton_packed_kv_layer(
        dst_kv_cache_layer,
        src_kv_cache,
        slot_mapping,
        block_size=block_size,
    )


def validate_triton_packed_kv_cache_layer(
    dst_kv_cache_layer: object,
    *,
    block_size: int | None = None,
) -> None:
    """Fail closed unless a registered cache has the vLLM 0.27.1 Triton ABI."""

    torch = _torch()
    if not torch.is_tensor(dst_kv_cache_layer):
        raise TypeError("registered vLLM KV cache layer must be a torch.Tensor")
    if dst_kv_cache_layer.ndim != 4:
        raise ValueError(
            "vLLM 0.27.1 Triton KV cache layer must have logical shape "
            "[num_blocks, num_kv_heads, block_size, 2 * head_dim]"
        )
    if any(int(dimension) <= 0 for dimension in dst_kv_cache_layer.shape):
        raise ValueError("vLLM 0.27.1 Triton KV cache dimensions must be positive")
    if int(dst_kv_cache_layer.shape[-1]) % 2:
        raise ValueError("vLLM 0.27.1 Triton KV cache packed content dimension must be even")
    if block_size is not None and int(dst_kv_cache_layer.shape[2]) != block_size:
        raise ValueError("destination Triton KV cache block dimension does not match block_size")


def _validate_triton_packed_geometry(
    dst_kv_cache_layer: Any,
    src_kv_cache: Any,
    *,
    block_size: int,
) -> None:
    validate_triton_packed_kv_cache_layer(
        dst_kv_cache_layer,
        block_size=block_size,
    )
    if src_kv_cache.ndim != 4 or int(src_kv_cache.shape[1]) != 2:
        raise ValueError(
            "logical source KV layer must have shape "
            "[num_tokens, 2, num_kv_heads, head_dim]"
        )
    num_kv_heads = int(dst_kv_cache_layer.shape[1])
    head_dim = int(dst_kv_cache_layer.shape[-1]) // 2
    expected_shape = (
        int(src_kv_cache.shape[0]),
        2,
        num_kv_heads,
        head_dim,
    )
    if tuple(src_kv_cache.shape) != expected_shape:
        raise ValueError(
            f"src_kv_cache shape must be {expected_shape} for the Triton packed KV layout"
        )


def _inject_triton_packed_kv_layer(
    dst_kv_cache_layer: Any,
    src_kv_cache: Any,
    slot_mapping: Any,
    *,
    block_size: int,
) -> None:
    block_indices = slot_mapping // block_size
    block_offsets = slot_mapping % block_size
    head_dim = int(dst_kv_cache_layer.shape[-1]) // 2
    source = _source_for_destination(src_kv_cache, dst_kv_cache_layer)
    dst_kv_cache_layer[block_indices, :, block_offsets, :head_dim] = source[:, 0]
    dst_kv_cache_layer[block_indices, :, block_offsets, head_dim:] = source[:, 1]


def _source_for_destination(src_kv_cache: Any, dst_kv_cache_layer: Any) -> Any:
    """Preserve raw FP8 bytes while otherwise matching the destination dtype."""

    torch = _torch()
    source = src_kv_cache.to(device=dst_kv_cache_layer.device)
    if source.dtype == dst_kv_cache_layer.dtype:
        return source
    # vLLM represents fp8_e4m3/fp8_e5m2 KV pages as uint8.  Cachet likewise
    # materializes persisted FP8 as raw uint8 bytes, so no numeric conversion is
    # allowed on this path.
    if source.dtype == torch.uint8 and dst_kv_cache_layer.dtype == torch.uint8:
        return source
    return source.to(dtype=dst_kv_cache_layer.dtype)


def _validate_slot_mapping_range(
    slot_mapping: Any,
    dst_kv_cache_layer: Any,
    *,
    block_size: int,
) -> None:
    if slot_mapping.numel() == 0:
        return
    if bool((slot_mapping < 0).any().item()):
        raise ValueError("slot_mapping must not contain negative or padded slot ids")
    slot_capacity = dst_kv_cache_layer.shape[0] * block_size
    if bool((slot_mapping >= slot_capacity).any().item()):
        raise ValueError("slot_mapping contains slot ids outside the destination KV cache")


def _validate_logical_blocks(
    blocks: Sequence[BlockSpan],
    *,
    block_size: int,
) -> None:
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
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("paged KV copy helpers require torch at runtime") from exc
    return torch
