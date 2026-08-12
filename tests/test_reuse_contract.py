from __future__ import annotations

import pytest

from document_kv_cache.engine_protocol import (
    KVKeyPositionEncoding,
    KVLayout,
    KVPayloadAxisOrder,
    KVStorageLayout,
)
from document_kv_cache.methods import method_spec
from document_kv_cache.models import CacheGenerationMethod
from document_kv_cache.reuse_contract import (
    ArtifactEncoding,
    PayloadDecodeStage,
    PositionHandling,
    TokenRecomputePolicy,
)


def layout(
    *,
    pre_rope: bool = False,
    key_position_encoding: KVKeyPositionEncoding | None = None,
    dtype: str = "float16",
    storage_layout: KVStorageLayout = KVStorageLayout.SEPARATE_KEY_VALUE,
    payload_axis_order: KVPayloadAxisOrder = KVPayloadAxisOrder.TOKEN_MAJOR,
) -> KVLayout:
    return KVLayout(
        model_id="model",
        lora_id="none",
        layout_version="v1",
        dtype=dtype,
        num_layers=2,
        block_size=16,
        bytes_per_token=32,
        num_query_heads=2,
        num_kv_heads=1,
        head_size=4,
        kv_stride_bytes=8,
        storage_layout=storage_layout,
        payload_axis_order=payload_axis_order,
        pre_rope=pre_rope,
        rope_theta=(
            10_000.0
            if pre_rope
            else None
        ),
        key_position_encoding=key_position_encoding,
    )


def test_vanilla_method_emits_raw_post_rope_reuse_plan() -> None:
    plan = method_spec(CacheGenerationMethod.VANILLA_PREFILL).reuse_plan()

    assert plan.artifact_format.encoding == ArtifactEncoding.RAW_KV
    assert plan.position_handling == PositionHandling.STORED_POST_ROPE
    assert plan.payload_decode_stage == PayloadDecodeStage.NONE
    assert plan.token_recompute_policy == TokenRecomputePolicy.NONE
    plan.validate_runtime_layout(layout())


def test_cacheblend_plan_requires_pre_rope_and_selective_recompute() -> None:
    plan = method_spec(CacheGenerationMethod.CACHEBLEND).reuse_plan()

    assert plan.position_handling == PositionHandling.REROPE_AT_INJECTION
    assert plan.requires_selective_recompute
    with pytest.raises(ValueError, match="pre-RoPE"):
        plan.validate_runtime_layout(layout())
    plan.validate_runtime_layout(layout(pre_rope=True))


def test_lmcache_plan_is_fully_engine_native() -> None:
    plan = method_spec(CacheGenerationMethod.LMCACHE).reuse_plan()

    assert not plan.requires_artifact
    assert plan.position_handling == PositionHandling.ENGINE_NATIVE
    assert plan.payload_decode_stage == PayloadDecodeStage.ENGINE_NATIVE
    assert plan.token_recompute_policy == TokenRecomputePolicy.ENGINE_NATIVE
