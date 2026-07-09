"""Correspondence between benchmark methods and their KV pre-computation.

A benchmark method (a row in the Main Latency / Score tables) is defined by three
things that were previously scattered across the codebase: which serving arm it
uses, how its document KV is pre-computed (post-RoPE vs position-independent
pre-RoPE, and whether cross-chunk selective recomputation is required), and which
serving-engine connector transfers the KV at request time.

:data:`METHOD_SPECS` records that correspondence in one place so a future method
(KV Packet, CacheBlend, InfoFlow KV) declares its pre-computation contract here
instead of wiring env flags, layout stamps, and validator whitelists ad hoc. The
registry is declarative today (the operational pre-RoPE control is still the
generator ``pre_rope`` flag); implementing a method means filling in its routine
and flipping ``implemented`` to True.
"""

from __future__ import annotations

from dataclasses import dataclass

from document_kv_cache.models import CacheGenerationMethod

__all__ = [
    "MethodSpec",
    "method_spec",
    "METHOD_SPECS",
    "NON_BENCHMARK_METHODS",
    "BASELINE_PREFILL_ARM",
    "DOCUMENT_KV_CACHE_ARM",
    "CACHET_CONNECTOR_MODE",
]

# Benchmark arm ids (kept as literals to avoid importing the heavier benchmarks
# module; validated against it in tests).
BASELINE_PREFILL_ARM = "baseline_prefill"
DOCUMENT_KV_CACHE_ARM = "document_kv_cache"

# Serving-engine KV connector mode (mirrors vllm_smoke.CACHET_KV_CONNECTOR_MODE).
CACHET_CONNECTOR_MODE = "cachet"

# Cache-generation labels that are not stand-alone benchmark methods (no table row
# and no dedicated pre-computation contract of their own).
NON_BENCHMARK_METHODS: frozenset[CacheGenerationMethod] = frozenset(
    {CacheGenerationMethod.ADAPTER_TRAINED, CacheGenerationMethod.CUSTOM}
)


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """Declarative contract mapping a benchmark method to its KV pre-computation.

    Attributes:
        method: The cache-generation method label.
        display_name: Human-readable name used in the benchmark tables.
        arm_id: Benchmark arm the method reports under.
        connector_mode: Serving-engine KV connector that transfers the KV.
        pre_rope: Pre-computation stores position-independent pre-RoPE keys
            (re-roped to their true offset at injection) rather than post-RoPE keys.
        selective_recompute: Requires recomputing a subset of cross-chunk tokens
            with full context to recover multi-document quality.
        implemented: Whether the end-to-end pre-computation + serving path exists.
        description: What the method does and what (if anything) is still missing.
    """

    method: CacheGenerationMethod
    display_name: str
    arm_id: str
    connector_mode: str
    pre_rope: bool
    selective_recompute: bool
    implemented: bool
    description: str


METHOD_SPECS: dict[CacheGenerationMethod, MethodSpec] = {
    CacheGenerationMethod.VANILLA_PREFILL: MethodSpec(
        method=CacheGenerationMethod.VANILLA_PREFILL,
        display_name="vanilla KV",
        arm_id=DOCUMENT_KV_CACHE_ARM,
        connector_mode=CACHET_CONNECTOR_MODE,
        pre_rope=False,
        selective_recompute=False,
        implemented=True,
        description=(
            "Reuse per-document KV computed independently and stored post-RoPE; the "
            "connector reports the document/system prefix as already computed and "
            "hydrates it into GPU KV. Correct for single-document (true-prefix) reuse; "
            "multi-document quality is limited by missing cross-document attention."
        ),
    ),
    CacheGenerationMethod.KV_PACKET: MethodSpec(
        method=CacheGenerationMethod.KV_PACKET,
        display_name="KV Packet",
        arm_id=DOCUMENT_KV_CACHE_ARM,
        connector_mode=CACHET_CONNECTOR_MODE,
        pre_rope=False,
        selective_recompute=False,
        implemented=False,
        description=(
            "Planned: packed-Q4 document KV payloads with provider dequant (or native "
            "packed-Q4 serving-engine KV). The packed pre-computation layout is not yet "
            "defined."
        ),
    ),
    CacheGenerationMethod.CACHEBLEND: MethodSpec(
        method=CacheGenerationMethod.CACHEBLEND,
        display_name="CacheBlend",
        arm_id=DOCUMENT_KV_CACHE_ARM,
        connector_mode=CACHET_CONNECTOR_MODE,
        pre_rope=True,
        selective_recompute=True,
        implemented=False,
        description=(
            "Planned: store position-independent pre-RoPE keys (foundation implemented; "
            "re-roped to their true offset at injection) AND recompute a small fraction "
            "of high-divergence cross-chunk tokens with full context to recover "
            "multi-document quality. The selective-recompute step is not yet implemented."
        ),
    ),
    CacheGenerationMethod.INFOFLOW_KV: MethodSpec(
        method=CacheGenerationMethod.INFOFLOW_KV,
        display_name="InfoFlow KV",
        arm_id=DOCUMENT_KV_CACHE_ARM,
        connector_mode=CACHET_CONNECTOR_MODE,
        pre_rope=True,
        selective_recompute=False,
        implemented=False,
        description=(
            "Planned: recover cross-document information flow over reused KV. Expected to "
            "build on position-independent pre-RoPE keys; the information-flow recovery "
            "step is not yet defined."
        ),
    ),
}


def method_spec(method: CacheGenerationMethod | str) -> MethodSpec:
    """Return the :class:`MethodSpec` for a benchmark cache-generation method.

    Accepts an enum member or its string value. Raises ``KeyError`` for cache-
    generation labels that are not stand-alone benchmark methods (ADAPTER_TRAINED,
    CUSTOM) or for unknown methods.
    """
    key = method if isinstance(method, CacheGenerationMethod) else CacheGenerationMethod(method)
    return METHOD_SPECS[key]
