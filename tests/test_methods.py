"""Tests for the method <-> pre-computation correspondence registry."""

import pytest

from document_kv_cache.methods import (
    BASELINE_PREFILL_ARM,
    CACHET_CONNECTOR_MODE,
    DOCUMENT_KV_CACHE_ARM,
    METHOD_SPECS,
    MethodSpec,
    NON_BENCHMARK_METHODS,
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


def test_only_vanilla_kv_is_implemented():
    implemented = {method for method, spec in METHOD_SPECS.items() if spec.implemented}
    assert implemented == {CacheGenerationMethod.VANILLA_PREFILL}


def test_vanilla_kv_is_post_rope_without_recompute():
    spec = method_spec(CacheGenerationMethod.VANILLA_PREFILL)
    assert spec.pre_rope is False
    assert spec.selective_recompute is False


def test_cacheblend_requires_pre_rope_and_selective_recompute():
    spec = method_spec(CacheGenerationMethod.CACHEBLEND)
    assert spec.pre_rope is True
    assert spec.selective_recompute is True
    assert spec.implemented is False


def test_infoflow_builds_on_pre_rope():
    spec = method_spec(CacheGenerationMethod.INFOFLOW_KV)
    assert spec.pre_rope is True
    assert spec.implemented is False


def test_kv_packet_is_planned_placeholder():
    spec = method_spec(CacheGenerationMethod.KV_PACKET)
    assert spec.implemented is False


def test_method_spec_is_frozen():
    spec = method_spec(CacheGenerationMethod.VANILLA_PREFILL)
    with pytest.raises((AttributeError, TypeError)):
        spec.implemented = True  # type: ignore[misc]
