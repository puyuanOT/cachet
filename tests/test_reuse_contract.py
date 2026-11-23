from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from document_kv_cache.engine_protocol import (
    KVKeyPositionEncoding,
    KVLayout,
    KVPayloadAxisOrder,
    KVStorageLayout,
)
from document_kv_cache.methods import (
    DOCUMENT_KV_CACHE_ARM,
    ENGINE_NATIVE_EXECUTION,
    MethodSpec,
    method_spec,
)
from document_kv_cache.models import CacheGenerationMethod
from document_kv_cache.reuse_contract import (
    ENGINE_NATIVE_ARTIFACT_FORMAT,
    ArtifactEncoding,
    ArtifactFormat,
    PayloadDecodeStage,
    PositionHandling,
    ReusePlan,
    RuntimeOperationDescriptor,
    RuntimeOperationHandlerRegistry,
    RuntimeOperationPhase,
    RuntimeOperationResult,
    TokenRecomputePolicy,
    apply_runtime_operation_handlers,
    runtime_operation_config_digest,
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


def test_vanilla_method_emits_raw_pre_rope_reuse_plan() -> None:
    plan = method_spec(CacheGenerationMethod.VANILLA_PREFILL).reuse_plan()

    assert plan.artifact_format.encoding == ArtifactEncoding.RAW_KV
    assert plan.position_handling == PositionHandling.REROPE_AT_INJECTION
    assert plan.payload_decode_stage == PayloadDecodeStage.NONE
    assert plan.token_recompute_policy == TokenRecomputePolicy.NONE
    plan.validate_runtime_layout(layout(pre_rope=True))


def test_cacheblend_plan_requires_pre_rope_and_selective_recompute() -> None:
    method = method_spec(CacheGenerationMethod.CACHEBLEND)

    assert method.pre_rope
    assert method.selective_recompute
    with pytest.raises(NotImplementedError, match="cacheblend"):
        method.reuse_plan()


def test_engine_native_plan_is_fully_engine_native() -> None:
    plan = MethodSpec(
        method="example.engine_native",
        display_name="Engine-native example",
        arm_id=DOCUMENT_KV_CACHE_ARM,
        connector_mode="example_engine_native",
        pre_rope=False,
        selective_recompute=False,
        implemented=True,
        artifact_version="external-v1",
        execution_kind=ENGINE_NATIVE_EXECUTION,
        artifact_format=ENGINE_NATIVE_ARTIFACT_FORMAT,
        payload_decode_stage=PayloadDecodeStage.ENGINE_NATIVE,
        description="Application-provided engine-native connector method.",
    ).reuse_plan()

    assert not plan.requires_artifact
    assert plan.position_handling == PositionHandling.ENGINE_NATIVE
    assert plan.payload_decode_stage == PayloadDecodeStage.ENGINE_NATIVE
    assert plan.token_recompute_policy == TokenRecomputePolicy.ENGINE_NATIVE


def test_reuse_plan_record_round_trip_preserves_capability_identity() -> None:
    plan = method_spec(CacheGenerationMethod.VANILLA_PREFILL).reuse_plan()

    restored = ReusePlan.from_record(plan.to_record())

    assert restored == plan
    assert restored.capability_id == plan.capability_id


def test_reuse_plan_record_rejects_tampered_or_unknown_capabilities() -> None:
    plan = method_spec(CacheGenerationMethod.VANILLA_PREFILL).reuse_plan()
    tampered = plan.to_record()
    tampered["connector_mode"] = "tampered"

    with pytest.raises(ValueError, match="capability_id"):
        ReusePlan.from_record(tampered)

    unsupported = plan.to_record()
    unsupported["future_operation"] = "opaque"
    with pytest.raises(ValueError, match="unsupported keys"):
        ReusePlan.from_record(unsupported)


class _NonFiniteArtifactFormat(ArtifactFormat):
    def to_record(self) -> dict[str, Any]:
        return {**ArtifactFormat.to_record(self), "scale": float("nan")}


def test_reuse_plan_capability_id_rejects_non_finite_numbers() -> None:
    plan = method_spec(CacheGenerationMethod.VANILLA_PREFILL).reuse_plan()
    artifact_format = plan.artifact_format
    tainted = replace(
        plan,
        artifact_format=_NonFiniteArtifactFormat(
            format_id=artifact_format.format_id,
            version=artifact_format.version,
            encoding=artifact_format.encoding,
            bits_per_element=artifact_format.bits_per_element,
            storage_layouts=artifact_format.storage_layouts,
            payload_axis_orders=artifact_format.payload_axis_orders,
        ),
    )

    assert plan.capability_id == (
        "acb62c94d96e3ab2d835859fa9eff8e291434713f9430960d9a1bc0b9582865c"
    )
    with pytest.raises(ValueError, match="NaN or Infinity"):
        tainted.to_record()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_runtime_operation_config_digest_rejects_non_finite_numbers(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="NaN or Infinity"):
        runtime_operation_config_digest({"fraction": value})


def test_runtime_operation_config_digest_keeps_finite_digests_and_type_errors() -> None:
    circular: dict[str, Any] = {}
    circular["self"] = circular

    assert runtime_operation_config_digest(
        {"fraction": 0.25, "count": 1, "label": "x"}
    ) == "4e9a3a2a7bee3aa3e50bc9ca9c69fed475268c0ca68531815da479314a718a97"
    for config in (
        {"handler": object()},
        circular,
        {"handler": object(), "fraction": float("nan")},
    ):
        with pytest.raises(TypeError, match="JSON-serializable"):
            runtime_operation_config_digest(config)


def test_runtime_handler_registry_binds_exact_configuration_digest() -> None:
    descriptor = RuntimeOperationDescriptor(
        strategy_id="toy.selector",
        version="1",
        config_digest=runtime_operation_config_digest({"fraction": 0.25}),
    )
    mismatched = RuntimeOperationDescriptor(
        strategy_id=descriptor.strategy_id,
        version=descriptor.version,
        config_digest=runtime_operation_config_digest({"fraction": 0.5}),
    )
    registry = RuntimeOperationHandlerRegistry().with_handler(
        RuntimeOperationPhase.TOKEN_SELECT,
        descriptor,
        lambda _request: RuntimeOperationResult(selected_token_indices=(0,)),
    )

    with pytest.raises(ValueError, match="configuration digest"):
        registry.resolve(RuntimeOperationPhase.TOKEN_SELECT, mismatched)


def test_select_and_recompute_handlers_execute_in_declared_order() -> None:
    selector = RuntimeOperationDescriptor(
        strategy_id="toy.selector",
        version="1",
        config_digest=runtime_operation_config_digest({"count": 1}),
    )
    recomputer = RuntimeOperationDescriptor(
        strategy_id="toy.recomputer",
        version="1",
        config_digest=runtime_operation_config_digest({"mode": "identity"}),
    )
    plan = ReusePlan(
        method_id="toy",
        connector_mode="cachet",
        artifact_format=method_spec(
            CacheGenerationMethod.VANILLA_PREFILL
        ).artifact_format,
        position_handling=PositionHandling.STORED_POST_ROPE,
        payload_decode_stage=PayloadDecodeStage.NONE,
        token_recompute_policy=TokenRecomputePolicy.SELECTIVE,
        token_selector=selector,
        token_recomputer=recomputer,
    )
    calls: list[RuntimeOperationPhase] = []

    def select(request):
        calls.append(request.phase)
        return RuntimeOperationResult(selected_token_indices=(1,))

    def recompute(request):
        calls.append(request.phase)
        assert request.selected_token_indices == (1,)
        return RuntimeOperationResult(payload=request.payload)

    registry = (
        RuntimeOperationHandlerRegistry()
        .with_handler(RuntimeOperationPhase.TOKEN_SELECT, selector, select)
        .with_handler(
            RuntimeOperationPhase.TOKEN_RECOMPUTE,
            recomputer,
            recompute,
        )
    )
    runtime_layout = layout()
    payload = b"x" * (2 * runtime_layout.bytes_per_token)

    result = apply_runtime_operation_handlers(
        ReusePlan.from_record(plan.to_record()),
        payload,
        layout=runtime_layout,
        total_tokens=2,
        handler_registry=registry,
    )

    assert result.payload == payload
    assert result.selected_token_indices == (1,)
    assert calls == [
        RuntimeOperationPhase.TOKEN_SELECT,
        RuntimeOperationPhase.TOKEN_RECOMPUTE,
    ]
