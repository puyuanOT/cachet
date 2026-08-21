"""Tests for the method <-> pre-computation correspondence registry."""

import pytest

from document_kv_cache.methods import (
    BASELINE_PREFILL_ARM,
    CACHET_CONNECTOR_MODE,
    DOCUMENT_KV_CACHE_ARM,
    ENGINE_NATIVE_EXECUTION,
    FULL_PREFIX_HANDOFF_TOPOLOGY,
    HandoffTopologySpec,
    LMCACHE_CONNECTOR_MODE,
    METHOD_SPECS,
    MethodCodeStatus,
    MethodLifecycle,
    MethodRegistry,
    MethodSpec,
    MethodValidationStatus,
    NON_BENCHMARK_METHODS,
    PER_DOCUMENT_HANDOFF_TOPOLOGY,
    UpstreamReproductionStatus,
    default_method_registry,
    method_spec,
)
from document_kv_cache.models import CacheGenerationMethod


def test_every_cache_generation_method_has_a_spec_or_is_excluded():
    # Guard: any new CacheGenerationMethod must either declare a pre-computation
    # contract or be explicitly marked a non-benchmark label.
    for method in CacheGenerationMethod:
        assert method in METHOD_SPECS or method in NON_BENCHMARK_METHODS, method


def test_registry_keys_match_their_specs():
    for method, spec in METHOD_SPECS.items():
        assert isinstance(spec, MethodSpec)
        assert spec.method is method
        assert spec.display_name and spec.description


def test_method_spec_accepts_enum_and_string():
    assert method_spec(CacheGenerationMethod.CACHEBLEND).method is CacheGenerationMethod.CACHEBLEND
    assert method_spec("cacheblend").method is CacheGenerationMethod.CACHEBLEND


def test_method_spec_rejects_non_benchmark_methods():
    for method in NON_BENCHMARK_METHODS:
        with pytest.raises(KeyError):
            method_spec(method)


def test_implemented_specs_reference_valid_arms_and_connectors():
    from document_kv_cache.benchmarks import BASELINE_PREFILL_ARM as arm_baseline
    from document_kv_cache.benchmarks import CACHE_REUSE_ARM as arm_cache
    from document_kv_cache.vllm_smoke import KV_CONNECTOR_MODES

    # The literals in methods.py must stay in sync with their canonical definitions.
    assert BASELINE_PREFILL_ARM == arm_baseline
    assert DOCUMENT_KV_CACHE_ARM == arm_cache
    assert CACHET_CONNECTOR_MODE in KV_CONNECTOR_MODES
    valid_arms = {arm_baseline, arm_cache}
    for spec in METHOD_SPECS.values():
        assert spec.arm_id in valid_arms, spec
        assert spec.connector_mode in KV_CONNECTOR_MODES, spec


def test_builtin_executable_methods_are_registered_as_implemented():
    implemented = {method for method, spec in METHOD_SPECS.items() if spec.implemented}
    assert implemented == {
        CacheGenerationMethod.FULL_PREFIX_PREFILL,
        CacheGenerationMethod.VANILLA_PREFILL,
        CacheGenerationMethod.LMCACHE,
    }


def test_vanilla_kv_is_pre_rope_without_recompute():
    spec = method_spec(CacheGenerationMethod.VANILLA_PREFILL)
    assert spec.pre_rope is True
    assert spec.selective_recompute is False
    assert spec.artifact_version == "2"


def test_full_prefix_control_has_a_distinct_method_identity():
    spec = method_spec(CacheGenerationMethod.FULL_PREFIX_PREFILL)
    assert spec.method_id == "full_prefix_prefill"
    assert spec.pre_rope is False
    assert spec.implemented is True


def test_builtin_handoff_topology_is_method_owned_and_immutable():
    full_prefix = method_spec(CacheGenerationMethod.FULL_PREFIX_PREFILL)
    vanilla = method_spec(CacheGenerationMethod.VANILLA_PREFILL)

    assert full_prefix.handoff_topology == FULL_PREFIX_HANDOFF_TOPOLOGY
    assert vanilla.handoff_topology == PER_DOCUMENT_HANDOFF_TOPOLOGY
    with pytest.raises((AttributeError, TypeError)):
        vanilla.handoff_topology.segment_per_document = False  # type: ignore[misc,union-attr]


def test_custom_handoff_topology_can_leave_segmentation_method_defined():
    topology = HandoffTopologySpec(
        topology_id="custom.paired_windows",
        segment_per_document=None,
    )
    spec = MethodSpec(
        method="custom.paired_windows",
        display_name="Custom paired windows",
        arm_id=DOCUMENT_KV_CACHE_ARM,
        connector_mode=CACHET_CONNECTOR_MODE,
        pre_rope=False,
        selective_recompute=False,
        implemented=False,
        handoff_topology=topology,
        description="Custom method owns a non-document physical topology.",
    )

    spec.validate_handoff_generation_mode(segment_per_document=False)
    spec.validate_handoff_generation_mode(segment_per_document=True)


def test_cacheblend_requires_pre_rope_and_selective_recompute():
    spec = method_spec(CacheGenerationMethod.CACHEBLEND)
    assert spec.pre_rope is True
    assert spec.selective_recompute is True
    assert spec.implemented is False


def test_infoflow_builds_on_pre_rope():
    spec = method_spec(CacheGenerationMethod.INFOFLOW_KV)
    assert spec.pre_rope is True
    assert spec.implemented is False


def test_kv_packet_is_a_fail_closed_placeholder():
    spec = method_spec(CacheGenerationMethod.KV_PACKET)
    assert spec.implemented is False
    assert spec.generator_factory is None
    assert spec.lifecycle == MethodLifecycle(
        code_status=MethodCodeStatus.PLANNED,
        upstream_reproduction=UpstreamReproductionStatus.NOT_REPRODUCED,
    )
    with pytest.raises(NotImplementedError, match="kv_packet"):
        spec.require_implemented()
    with pytest.raises(NotImplementedError, match="kv_packet"):
        spec.reuse_plan()


def test_lmcache_is_an_engine_native_method():
    spec = method_spec(CacheGenerationMethod.LMCACHE)
    assert spec.connector_mode == LMCACHE_CONNECTOR_MODE
    assert spec.execution_kind == ENGINE_NATIVE_EXECUTION
    assert spec.generator_factory is None
    spec.require_implemented()


def test_default_registry_is_immutable_and_supports_custom_methods():
    registry = default_method_registry()
    assert isinstance(registry, MethodRegistry)
    with pytest.raises(TypeError):
        registry.specs["custom"] = method_spec(CacheGenerationMethod.VANILLA_PREFILL)  # type: ignore[index]

    custom = MethodSpec(
        method="opentable.experimental",
        display_name="OpenTable experimental",
        arm_id=DOCUMENT_KV_CACHE_ARM,
        connector_mode=CACHET_CONNECTOR_MODE,
        pre_rope=False,
        selective_recompute=False,
        implemented=False,
        description="Application-provided method scaffold.",
    )
    extended = registry.with_spec(custom)
    assert "opentable.experimental" not in registry
    assert extended.get("opentable.experimental") is custom


def test_unimplemented_method_fails_before_execution():
    with pytest.raises(NotImplementedError, match="cacheblend"):
        default_method_registry().get(
            CacheGenerationMethod.CACHEBLEND,
            require_implemented=True,
        )


def test_method_validates_generator_capabilities():
    class PreRopeGenerator:
        pre_rope = True

        def generate(self):
            raise AssertionError

    method_spec(CacheGenerationMethod.VANILLA_PREFILL).validate_generator(PreRopeGenerator())

    class PostRopeGenerator(PreRopeGenerator):
        pre_rope = False

    with pytest.raises(ValueError, match="requires pre_rope=True"):
        method_spec(CacheGenerationMethod.VANILLA_PREFILL).validate_generator(PostRopeGenerator())


def test_method_loads_declared_generator_factory() -> None:
    factory = method_spec(CacheGenerationMethod.VANILLA_PREFILL).load_generator_factory()

    assert callable(factory)
    assert factory.__name__ == "build_pre_rope_transformers_kv_chunk_generator"


def test_engine_native_method_has_no_generator_factory() -> None:
    with pytest.raises(ValueError, match="engine-native"):
        method_spec(CacheGenerationMethod.LMCACHE).load_generator_factory()


def test_method_spec_is_frozen():
    spec = method_spec(CacheGenerationMethod.VANILLA_PREFILL)
    with pytest.raises((AttributeError, TypeError)):
        spec.implemented = True  # type: ignore[misc]


def test_method_lifecycle_record_is_closed_and_round_trips() -> None:
    lifecycle = MethodLifecycle(
        code_status=MethodCodeStatus.RUNNABLE,
        upstream_reproduction=UpstreamReproductionStatus.REPRODUCED,
        engine_validation=MethodValidationStatus.PASSED,
        live_canary=MethodValidationStatus.PASSED,
        publication_evidence=MethodValidationStatus.PASSED,
    )

    assert MethodLifecycle.from_record(lifecycle.to_record()) == lifecycle
    unsupported = {**lifecycle.to_record(), "future_stage": "passed"}
    with pytest.raises(ValueError, match="unsupported keys"):
        MethodLifecycle.from_record(unsupported)


def test_method_lifecycle_cannot_overstate_runnable_or_evidence_state() -> None:
    with pytest.raises(ValueError, match="implemented must match"):
        MethodSpec(
            method="bad-lifecycle",
            display_name="Bad lifecycle",
            arm_id=DOCUMENT_KV_CACHE_ARM,
            connector_mode=CACHET_CONNECTOR_MODE,
            pre_rope=False,
            selective_recompute=False,
            implemented=False,
            lifecycle=MethodLifecycle(
                code_status=MethodCodeStatus.RUNNABLE,
                upstream_reproduction=UpstreamReproductionStatus.NOT_RECORDED,
            ),
            description="Invalid test fixture.",
        )
    with pytest.raises(ValueError, match="passing live canary"):
        MethodLifecycle(
            code_status=MethodCodeStatus.RUNNABLE,
            upstream_reproduction=UpstreamReproductionStatus.NOT_RECORDED,
            live_canary=MethodValidationStatus.PASSED,
        )
