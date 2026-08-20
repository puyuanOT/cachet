"""Property-based correctness tests for the shared RoPE re-alignment helper.

These validate the rotate-half RoPE against its defining invariants (norm
preservation and relative-position dot products) rather than a re-statement of
the same formula, so a convention bug (interleaved vs half, cos/sin swap, sign)
is caught locally before any on-cluster run.
"""

import pytest

from document_kv_cache.rope import apply_rope_to_keys, rope_cos_sin

torch = pytest.importorskip("torch")


THETA = 5_000_000.0  # Pinned Qwen3-4B-Instruct-2507 rope_theta


def test_position_zero_is_identity():
    keys = torch.randn(5, 8, 128, dtype=torch.float32)
    positions = torch.zeros(5, dtype=torch.long)
    out = apply_rope_to_keys(keys, positions, rope_theta=THETA)
    assert torch.allclose(out, keys, atol=1e-5)


def test_rope_preserves_per_head_norm():
    # A rotation preserves the L2 norm of each key vector.
    keys = torch.randn(7, 8, 128, dtype=torch.float32)
    positions = torch.arange(7, dtype=torch.long) * 13 + 4
    out = apply_rope_to_keys(keys, positions, rope_theta=THETA)
    assert torch.allclose(out.norm(dim=-1), keys.norm(dim=-1), atol=1e-4)


def test_rope_relative_position_dot_product_invariance():
    # RoPE's defining property: <rope(q,m), rope(k,n)> depends only on (m-n).
    head_dim = 128
    q = torch.randn(1, 1, head_dim, dtype=torch.float32)
    k = torch.randn(1, 1, head_dim, dtype=torch.float32)

    def score(mpos, npos):
        rq = apply_rope_to_keys(q, torch.tensor([mpos]), rope_theta=THETA)
        rk = apply_rope_to_keys(k, torch.tensor([npos]), rope_theta=THETA)
        return float((rq * rk).sum())

    # Same relative offset (m - n = 3) at different absolute positions.
    s1 = score(3, 0)
    s2 = score(103, 100)
    s3 = score(5003, 5000)
    assert abs(s1 - s2) < 1e-2
    assert abs(s1 - s3) < 1e-2


def test_rope_shifting_both_positions_preserves_scores():
    # Reusing a cached chunk at an offset must keep intra-chunk attention identical
    # to computing it at position 0 -> shifting q and k by the same delta is a no-op
    # on their dot product. This is exactly why re-alignment is correctness-preserving.
    head_dim = 128
    q = torch.randn(1, 4, head_dim)
    k = torch.randn(1, 4, head_dim)
    base_q = apply_rope_to_keys(q, torch.tensor([2]), rope_theta=THETA)
    base_k = apply_rope_to_keys(k, torch.tensor([0]), rope_theta=THETA)
    shift_q = apply_rope_to_keys(q, torch.tensor([2 + 4096]), rope_theta=THETA)
    shift_k = apply_rope_to_keys(k, torch.tensor([0 + 4096]), rope_theta=THETA)
    assert torch.allclose((base_q * base_k).sum(-1), (shift_q * shift_k).sum(-1), atol=1e-2)


def test_independent_pre_rope_documents_use_assembled_absolute_offsets():
    first_document = torch.randn(3, 8, 128, dtype=torch.float32)
    second_document = torch.randn(2, 8, 128, dtype=torch.float32)
    assembled = torch.cat((first_document, second_document), dim=0)

    expected = apply_rope_to_keys(
        assembled,
        torch.arange(5, dtype=torch.long),
        rope_theta=THETA,
    )
    independently_positioned = torch.cat(
        (
            apply_rope_to_keys(
                first_document,
                torch.arange(3, dtype=torch.long),
                rope_theta=THETA,
            ),
            apply_rope_to_keys(
                second_document,
                torch.arange(3, 5, dtype=torch.long),
                rope_theta=THETA,
            ),
        ),
        dim=0,
    )

    assert torch.allclose(independently_positioned, expected, atol=1e-6)


def test_cos_sin_reuse_matches_inline():
    keys = torch.randn(6, 8, 128, dtype=torch.float32)
    positions = torch.arange(6, dtype=torch.long) * 7
    cos, sin = rope_cos_sin(positions, head_dim=128, rope_theta=THETA)
    reused = apply_rope_to_keys(keys, positions, rope_theta=THETA, cos=cos, sin=sin)
    inline = apply_rope_to_keys(keys, positions, rope_theta=THETA)
    assert torch.allclose(reused, inline, atol=1e-6)


def test_dtype_roundtrip_upcasts_for_rotation():
    # Low-precision keys are rotated in float32 and cast back to the input dtype.
    keys = torch.randn(4, 8, 128).to(torch.bfloat16)
    positions = torch.arange(4, dtype=torch.long) * 11
    out = apply_rope_to_keys(keys, positions, rope_theta=THETA)
    assert out.dtype == torch.bfloat16
    assert out.shape == keys.shape


def test_partial_rotary_passes_through_tail():
    keys = torch.randn(4, 2, 128, dtype=torch.float32)
    positions = torch.arange(4, dtype=torch.long) + 1
    out = apply_rope_to_keys(keys, positions, rope_theta=THETA, rotary_dim=64)
    # Tail dims (64:) are untouched; leading dims (:64) are rotated (norm preserved).
    assert torch.allclose(out[..., 64:], keys[..., 64:], atol=1e-5)
    assert torch.allclose(out[..., :64].norm(dim=-1), keys[..., :64].norm(dim=-1), atol=1e-4)


def test_rejects_non_3d_keys():
    with pytest.raises(ValueError):
        apply_rope_to_keys(torch.randn(5, 128), torch.zeros(5, dtype=torch.long), rope_theta=THETA)


def test_provider_rerope_src_layer_ropes_keys_leaves_values():
    # The vLLM injection helper must rotate K (index 0) at the load's absolute
    # positions and leave V (index 1) byte-for-byte unchanged.
    from vllm_kv_injection.vllm_native_provider import _rerope_src_layer_keys

    T, H, D = 6, 8, 128
    src = torch.randn(T, 2, H, D, dtype=torch.float32)
    positions = torch.arange(4000, 4000 + T)
    cos, sin = rope_cos_sin(positions, head_dim=D, rope_theta=THETA, rotary_dim=D)
    out = _rerope_src_layer_keys(src.clone(), cos=cos, sin=sin, rope_theta=THETA, rotary_dim=D)
    assert out.shape == src.shape
    assert torch.allclose(out[:, 1], src[:, 1], atol=1e-6)  # V untouched
    assert torch.allclose(out[:, 0], apply_rope_to_keys(src[:, 0], positions, rope_theta=THETA), atol=1e-5)
    with pytest.raises(ValueError):
        _rerope_src_layer_keys(torch.randn(T, H, D), cos=cos, sin=sin, rope_theta=THETA, rotary_dim=D)


def test_provider_rerope_fp8_keys_stored_as_uint8():
    # fp8 KV is stored as raw uint8 bytes; the re-rope must bitcast uint8<->fp8 around
    # the rotation (not integer-convert the bytes). This reproduces the injection path
    # and would fail with the naive uint8.to(float32) bug.
    if not hasattr(torch, "float8_e5m2"):
        pytest.skip("float8_e5m2 unsupported")
    from vllm_kv_injection.vllm_native_provider import _rerope_src_layer_keys

    T, H, D = 6, 8, 128
    k_fp8 = torch.randn(T, H, D).to(torch.float8_e5m2)
    v_fp8 = torch.randn(T, H, D).to(torch.float8_e5m2)
    src_layer = torch.stack((k_fp8.view(torch.uint8), v_fp8.view(torch.uint8)), dim=1)  # [T,2,H,D] uint8
    positions = torch.arange(4000, 4000 + T)
    cos, sin = rope_cos_sin(positions, head_dim=D, rope_theta=THETA, rotary_dim=D)
    out = _rerope_src_layer_keys(
        src_layer, cos=cos, sin=sin, rope_theta=THETA, rotary_dim=D, payload_dtype="fp8_e5m2"
    )
    assert out.dtype == torch.uint8 and out.shape == src_layer.shape
    assert torch.equal(out[:, 1], v_fp8.view(torch.uint8))  # V bytes untouched
    # Decoded roped K must equal the reference rope of the decoded fp8 keys (fp8-coarse).
    out_k = out[:, 0].view(torch.float8_e5m2).to(torch.float32)
    ref_k = (
        apply_rope_to_keys(k_fp8.to(torch.float32), positions, rope_theta=THETA)
        .to(torch.float8_e5m2)
        .to(torch.float32)
    )
    assert torch.allclose(out_k, ref_k, atol=1e-1)


def test_provider_rope_cos_sin_for_load_uses_absolute_positions():
    from vllm_kv_injection.vllm_native_provider import _rope_cos_sin_for_load

    class _Load:
        source_token_start = 4000
        token_count = 6

    dst = torch.zeros(4, 2, 16, 8, 128)  # [blocks, 2, block_size, kv_heads, head_dim]
    cos, sin = _rope_cos_sin_for_load(_Load(), dst, rope_theta=THETA, rotary_dim=None)
    assert cos.shape == (6, 128) and sin.shape == (6, 128)
    ref_cos, ref_sin = rope_cos_sin(torch.arange(4000, 4006), head_dim=128, rope_theta=THETA, rotary_dim=128)
    assert torch.allclose(cos, ref_cos, atol=1e-6)
