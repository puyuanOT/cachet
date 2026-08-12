"""Tests for the method <-> pre-computation correspondence registry."""

import pytest

from document_kv_cache.methods import (
    BASELINE_PREFILL_ARM,
    CACHET_CONNECTOR_MODE,
    DOCUMENT_KV_CACHE_ARM,
    ENGINE_NATIVE_EXECUTION,
    LMCACHE_CONNECTOR_MODE,
    METHOD_SPECS,
    MethodRegistry,
    MethodSpec,
    NON_BENCHMARK_METHODS,
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


def test_vanilla_kv_is_post_rope_without_recompute():
    spec = method_spec(CacheGenerationMethod.VANILLA_PREFILL)
    assert spec.pre_rope is False
    assert spec.selective_recompute is False


def test_full_prefix_control_has_a_distinct_method_identity():
    spec = method_spec(CacheGenerationMethod.FULL_PREFIX_PREFILL)
    assert spec.method_id == "full_prefix_prefill"
    assert spec.pre_rope is False
    assert spec.implemented is True


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
    with pytest.raises(NotImplementedError, match="kv_packet"):
        spec.require_implemented()


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
    class PostRopeGenerator:
        pre_rope = False

        def generate(self):
            raise AssertionError

    method_spec(CacheGenerationMethod.VANILLA_PREFILL).validate_generator(PostRopeGenerator())

    class PreRopeGenerator(PostRopeGenerator):
        pre_rope = True

    with pytest.raises(ValueError, match="requires pre_rope=False"):
        method_spec(CacheGenerationMethod.VANILLA_PREFILL).validate_generator(PreRopeGenerator())


def test_method_loads_declared_generator_factory() -> None:
    factory = method_spec(CacheGenerationMethod.VANILLA_PREFILL).load_generator_factory()

    assert callable(factory)
    assert factory.__name__ == "build_transformers_kv_chunk_generator"


def test_engine_native_method_has_no_generator_factory() -> None:
    with pytest.raises(ValueError, match="engine-native"):
        method_spec(CacheGenerationMethod.LMCACHE).load_generator_factory()


def test_method_spec_is_frozen():
    spec = method_spec(CacheGenerationMethod.VANILLA_PREFILL)
    with pytest.raises((AttributeError, TypeError)):
        spec.implemented = True  # type: ignore[misc]
