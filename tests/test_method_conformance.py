from __future__ import annotations

import json
from dataclasses import replace

from document_kv_cache.engine_adapters import RuntimeOperationSupport, vllm_adapter_spec
from document_kv_cache.method_conformance import (
    inspect_method_conformance,
    main,
    method_conformance_to_record,
)
from document_kv_cache.methods import (
    CACHET_CONNECTOR_MODE,
    DOCUMENT_KV_CACHE_ARM,
    MethodSpec,
    method_spec,
)
from document_kv_cache.models import CacheGenerationMethod
from document_kv_cache.reference_method import METHOD_SPEC as REFERENCE_METHOD_SPEC
from document_kv_cache.reuse_contract import (
    RuntimeOperationDescriptor,
    RuntimeOperationHandlerRegistry,
    RuntimeOperationPhase,
    RuntimeOperationResult,
    TokenRecomputePolicy,
    runtime_operation_config_digest,
)


def test_builtin_artifact_method_loads_factory_without_model_instantiation() -> None:
    result = inspect_method_conformance(
        method_spec(CacheGenerationMethod.VANILLA_PREFILL)
    )

    assert result.ok
    assert result.factory_loaded
    assert not result.generator_instantiated
    record = method_conformance_to_record(result)
    assert record["method"]["reuse_plan"]["method_id"] == "vanilla_prefill"


def test_unimplemented_method_fails_closed_unless_explicitly_allowed() -> None:
    method = method_spec(CacheGenerationMethod.CACHEBLEND)

    assert not inspect_method_conformance(method).ok
    assert inspect_method_conformance(method, allow_unimplemented=True).ok


def test_engine_native_method_conforms_without_generator() -> None:
    result = inspect_method_conformance(method_spec(CacheGenerationMethod.LMCACHE))

    assert result.ok
    assert not result.factory_loaded


def test_method_conformance_cli_writes_machine_readable_record(tmp_path) -> None:
    output = tmp_path / "conformance.json"

    assert main(["--method-id", "lmcache", "--output-json", str(output)]) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["ok"] is True
    assert record["method"]["method_id"] == "lmcache"


def test_cpu_reference_method_exercises_strict_runtime_handoff_roundtrip() -> None:
    result = inspect_method_conformance(
        REFERENCE_METHOD_SPEC,
        instantiate_generator=True,
        exercise_runtime=True,
    )

    assert result.ok
    assert result.generator_instantiated
    assert result.workflow_exercised
    assert result.handoff_roundtrip


def test_cpu_custom_method_handlers_survive_and_execute_handoff_roundtrip() -> None:
    selector = RuntimeOperationDescriptor(
        strategy_id="cpu-toy.selector",
        version="1",
        config_digest=runtime_operation_config_digest({"tokens": 1}),
    )
    recomputer = RuntimeOperationDescriptor(
        strategy_id="cpu-toy.recomputer",
        version="1",
        config_digest=runtime_operation_config_digest({"mode": "identity"}),
    )
    method = MethodSpec(
        method="cpu_custom_runtime",
        display_name="CPU custom runtime",
        arm_id=DOCUMENT_KV_CACHE_ARM,
        connector_mode=CACHET_CONNECTOR_MODE,
        pre_rope=False,
        selective_recompute=True,
        implemented=True,
        generator_factory="document_kv_cache.reference_method:build_generator",
        token_selector=selector,
        token_recomputer=recomputer,
        description="CPU-only handler lifecycle conformance fixture.",
    )
    spec = replace(
        vllm_adapter_spec(),
        supported_token_recompute_policies=(
            TokenRecomputePolicy.NONE,
            TokenRecomputePolicy.SELECTIVE,
        ),
        supported_runtime_operations=(
            RuntimeOperationSupport(
                RuntimeOperationPhase.TOKEN_SELECT,
                selector.strategy_id,
                selector.version,
            ),
            RuntimeOperationSupport(
                RuntimeOperationPhase.TOKEN_RECOMPUTE,
                recomputer.strategy_id,
                recomputer.version,
            ),
        ),
    )
    calls: list[RuntimeOperationPhase] = []

    def select(request):
        calls.append(request.phase)
        return RuntimeOperationResult(selected_token_indices=(0,))

    def recompute(request):
        calls.append(request.phase)
        assert request.selected_token_indices == (0,)
        return RuntimeOperationResult(payload=request.payload)

    handlers = (
        RuntimeOperationHandlerRegistry()
        .with_handler(RuntimeOperationPhase.TOKEN_SELECT, selector, select)
        .with_handler(
            RuntimeOperationPhase.TOKEN_RECOMPUTE,
            recomputer,
            recompute,
        )
    )

    result = inspect_method_conformance(
        method,
        instantiate_generator=True,
        exercise_runtime=True,
        adapter_spec=spec,
        operation_handlers=handlers,
    )

    assert result.ok, result.issues
    assert result.runtime_operations_exercised
    assert calls == [
        RuntimeOperationPhase.TOKEN_SELECT,
        RuntimeOperationPhase.TOKEN_RECOMPUTE,
    ]
