"""Shared rotary-position-embedding (RoPE) re-alignment for pre-RoPE KV reuse.

Cachet can store document keys in PRE-RoPE form (for Qwen3, post QK-norm but
before rotary). A chunk cached this way is position-independent, so it can be
reused at any absolute offset: applying rotary at the chunk's true position
during injection reconstructs exactly the post-RoPE key that the serving engine's
paged cache expects. This fixes the multi-document positional-consistency problem
(each chunk is otherwise cached at local positions starting at 0).

The math matches the HuggingFace default (non-interleaved, "rotate-half") RoPE
used by the Llama/Qwen model family. The same helper is used at generation time
(to self-check that ``apply_rope(pre_rope_K, 0..L)`` reconstructs the model's own
post-RoPE keys) and at injection time (to rotate to the real offset), so the two
sides can never diverge. Rotation is always computed in float32 for precision,
then cast back to the key dtype.
"""

from __future__ import annotations

from typing import Any

__all__ = ["apply_rope_to_keys", "rope_cos_sin"]


def _torch() -> Any:
    import torch

    return torch


def rope_cos_sin(
    positions: Any,
    *,
    head_dim: int,
    rope_theta: float,
    device: Any = None,
    rotary_dim: int | None = None,
) -> tuple[Any, Any]:
    """Return ``(cos, sin)`` of shape ``[T, rotary_dim]`` (float32) for ``positions``.

    ``rotary_dim`` defaults to ``head_dim`` (full rotary). ``inv_freq`` uses the
    standard ``theta ** (-2i/rotary_dim)`` schedule over the rotary sub-dimension.
    """
    torch = _torch()
    rotary_dim = int(rotary_dim or head_dim)
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    if device is None:
        device = positions.device
    inv_freq = 1.0 / (
        float(rope_theta)
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    pos = positions.to(device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [T, rotary_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, rotary_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: Any) -> Any:
    torch = _torch()
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_to_keys(
    keys: Any,
    positions: Any = None,
    *,
    rope_theta: float | None = None,
    rotary_dim: int | None = None,
    cos: Any = None,
    sin: Any = None,
) -> Any:
    """Apply rotary position embedding to ``keys`` at ``positions``.

    ``keys`` is ``[T, H, D]`` (token, kv_head, head_dim); ``positions`` is ``[T]``
    of absolute logical positions. Returns a tensor of the same shape/dtype as
    ``keys`` with rotary applied to the first ``rotary_dim`` (default all) head
    dims. Rotation is done in float32. ``cos``/``sin`` (shape ``[T, rotary_dim]``,
    from :func:`rope_cos_sin`) may be passed to reuse a per-load precomputation.
    """
    torch = _torch()
    if keys.dim() != 3:
        raise ValueError(f"keys must be [T, H, D]; got shape {tuple(keys.shape)}")
    head_dim = keys.shape[-1]
    rd = int(rotary_dim or head_dim)
    if rd > head_dim:
        raise ValueError(f"rotary_dim {rd} exceeds head_dim {head_dim}")
    if cos is None or sin is None:
        if positions is None or rope_theta is None:
            raise ValueError("apply_rope_to_keys requires either (positions + rope_theta) or (cos + sin)")
        cos, sin = rope_cos_sin(
            positions, head_dim=head_dim, rope_theta=rope_theta, device=keys.device, rotary_dim=rd
        )
    cos = cos.to(torch.float32)[:, None, :]  # [T, 1, rd]
    sin = sin.to(torch.float32)[:, None, :]
    k32 = keys.to(torch.float32)
    if rd == head_dim:
        rotated = k32 * cos + _rotate_half(k32) * sin
        return rotated.to(keys.dtype)
    # Partial rotary: rotate the leading rd dims, pass the rest through.
    k_rot = k32[..., :rd]
    k_pass = k32[..., rd:]
    rotated = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat((rotated, k_pass), dim=-1).to(keys.dtype)
