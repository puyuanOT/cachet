"""Executable contracts for benchmark methods and their KV pre-computation.

A benchmark method (a row in the Main Latency / Score tables) is defined by three
things that were previously scattered across the codebase: which serving arm it
uses, how its document KV is pre-computed (post-RoPE vs position-independent
pre-RoPE, and whether cross-chunk selective recomputation is required), and which
serving-engine connector transfers the KV at request time.

:data:`DEFAULT_METHOD_REGISTRY` records that correspondence in one place so a
future method declares its pre-computation and runtime contract here instead of
wiring env flags, layout stamps, and validator whitelists ad hoc.  The registry
also validates a generator's observable capabilities before artifact generation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any

from document_kv_cache.models import CacheGenerationMethod
from document_kv_cache.reuse_contract import (
    ENGINE_NATIVE_ARTIFACT_FORMAT,
    RAW_KV_ARTIFACT_FORMAT,
    ArtifactFormat,
    PayloadDecodeStage,
    PositionHandling,
    ReusePlan,
    TokenRecomputePolicy,
)

__all__ = [
    "MethodSpec",
    "MethodRegistry",
    "method_spec",
    "default_method_registry",
    "builtin_method_specs",
    "DEFAULT_METHOD_REGISTRY",
    "METHOD_SPECS",
    "NON_BENCHMARK_METHODS",
    "BASELINE_PREFILL_ARM",
    "DOCUMENT_KV_CACHE_ARM",
    "CACHET_CONNECTOR_MODE",
    "LMCACHE_CONNECTOR_MODE",
    "CACHET_ARTIFACT_EXECUTION",
    "ENGINE_NATIVE_EXECUTION",
]

# Benchmark arm ids (kept as literals to avoid importing the heavier benchmarks
# module; validated against it in tests).
BASELINE_PREFILL_ARM = "baseline_prefill"
DOCUMENT_KV_CACHE_ARM = "document_kv_cache"

# Serving-engine KV connector mode (mirrors vllm_smoke.CACHET_KV_CONNECTOR_MODE).
CACHET_CONNECTOR_MODE = "cachet"
LMCACHE_CONNECTOR_MODE = "lmcache"

CACHET_ARTIFACT_EXECUTION = "cachet_artifact"
ENGINE_NATIVE_EXECUTION = "engine_native"
_EXECUTION_KINDS = frozenset({CACHET_ARTIFACT_EXECUTION, ENGINE_NATIVE_EXECUTION})

# Cache-generation labels that are not stand-alone benchmark methods (no table row
# and no dedicated pre-computation contract of their own).
NON_BENCHMARK_METHODS: frozenset[CacheGenerationMethod] = frozenset(
    {CacheGenerationMethod.ADAPTER_TRAINED, CacheGenerationMethod.CUSTOM}
)


def _method_id(method: CacheGenerationMethod | str) -> str:
    method_id = method.value if isinstance(method, CacheGenerationMethod) else method
    _validate_identifier("method", method_id)
    return method_id


def _validate_identifier(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if "|" in value:
        raise ValueError(f"{field_name} must not contain '|'")


def _validate_factory_path(value: str) -> None:
    _validate_identifier("generator_factory", value)
    module_name, separator, attribute_name = value.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("generator_factory must use the 'module.path:callable_name' format")


def _validated_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in metadata.items():
        _validate_identifier("metadata key", key)
        if not isinstance(value, str):
            raise TypeError("metadata values must be strings")
        normalized[key] = value
    return normalized


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """Executable contract mapping a benchmark method to its KV implementation.

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
        artifact_version: Version of the method-specific artifact semantics.
        execution_kind: Whether Cachet creates an artifact or the serving engine
            owns the cache lifecycle.
        generator_factory: Import path for the default ``KVChunkGenerator`` factory.
        artifact_format: Persisted byte encoding, separate from runtime KV layout.
        payload_decode_stage: Where encoded artifact bytes become runtime KV.
        description: What the method does and what (if anything) is still missing.
    """

    method: CacheGenerationMethod | str
    display_name: str
    arm_id: str
    connector_mode: str
    pre_rope: bool
    selective_recompute: bool
    implemented: bool
    description: str
    artifact_version: str = "1"
    execution_kind: str = CACHET_ARTIFACT_EXECUTION
    generator_factory: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    artifact_format: ArtifactFormat = RAW_KV_ARTIFACT_FORMAT
    payload_decode_stage: PayloadDecodeStage = PayloadDecodeStage.NONE
    position_handling: PositionHandling | None = None

    def __post_init__(self) -> None:
        method_id = self.method_id
        _validate_identifier("method", method_id)
        _validate_identifier("artifact_version", self.artifact_version)
        _validate_identifier("arm_id", self.arm_id)
        _validate_identifier("connector_mode", self.connector_mode)
        if not isinstance(self.display_name, str) or not self.display_name:
            raise ValueError("display_name must be non-empty")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("description must be non-empty")
        if type(self.pre_rope) is not bool:
            raise ValueError("pre_rope must be a boolean")
        if type(self.selective_recompute) is not bool:
            raise ValueError("selective_recompute must be a boolean")
        if type(self.implemented) is not bool:
            raise ValueError("implemented must be a boolean")
        if self.execution_kind not in _EXECUTION_KINDS:
            raise ValueError(f"execution_kind must be one of {sorted(_EXECUTION_KINDS)}")
        if not isinstance(self.artifact_format, ArtifactFormat):
            raise TypeError("artifact_format must be an ArtifactFormat")
        decode_stage = PayloadDecodeStage(self.payload_decode_stage)
        object.__setattr__(self, "payload_decode_stage", decode_stage)
        if self.position_handling is None:
            position_handling = (
                PositionHandling.ENGINE_NATIVE
                if self.execution_kind == ENGINE_NATIVE_EXECUTION
                else PositionHandling.REROPE_AT_INJECTION
                if self.pre_rope
                else PositionHandling.STORED_POST_ROPE
            )
        else:
            position_handling = PositionHandling(self.position_handling)
        object.__setattr__(self, "position_handling", position_handling)
        if self.generator_factory is not None:
            _validate_factory_path(self.generator_factory)
        if self.execution_kind == ENGINE_NATIVE_EXECUTION and self.generator_factory is not None:
            raise ValueError("engine-native methods must not declare a generator_factory")
        if self.implemented and self.execution_kind == CACHET_ARTIFACT_EXECUTION and self.generator_factory is None:
            raise ValueError("implemented Cachet artifact methods require a generator_factory")
        if self.execution_kind == ENGINE_NATIVE_EXECUTION:
            if self.artifact_format != ENGINE_NATIVE_ARTIFACT_FORMAT:
                raise ValueError("engine-native methods require ENGINE_NATIVE_ARTIFACT_FORMAT")
            if decode_stage != PayloadDecodeStage.ENGINE_NATIVE:
                raise ValueError("engine-native methods require engine-native payload decoding")
            if position_handling != PositionHandling.ENGINE_NATIVE:
                raise ValueError("engine-native methods require engine-native position handling")
        elif self.artifact_format == ENGINE_NATIVE_ARTIFACT_FORMAT:
            raise ValueError("Cachet artifact methods require a persisted artifact format")
        elif position_handling == PositionHandling.ENGINE_NATIVE:
            raise ValueError("Cachet artifact methods cannot use engine-native position handling")
        if self.pre_rope and position_handling != PositionHandling.REROPE_AT_INJECTION:
            raise ValueError("pre-RoPE methods require re-rope position handling")
        if not self.pre_rope and position_handling == PositionHandling.REROPE_AT_INJECTION:
            raise ValueError("re-rope position handling requires pre_rope=True")
        object.__setattr__(self, "metadata", MappingProxyType(_validated_metadata(self.metadata)))

    @property
    def method_id(self) -> str:
        return self.method.value if isinstance(self.method, CacheGenerationMethod) else self.method

    def require_implemented(self) -> "MethodSpec":
        """Return this spec or fail before an unimplemented method can run."""

        if not self.implemented:
            raise NotImplementedError(
                f"Method {self.method_id!r} is registered but not implemented: {self.description}"
            )
        return self

    def validate_generator(self, generator: object, *, require_implemented: bool = True) -> None:
        """Validate method capabilities exposed by a chunk generator."""

        if require_implemented:
            self.require_implemented()
        assert self.position_handling is not None
        if self.execution_kind != CACHET_ARTIFACT_EXECUTION:
            raise ValueError(f"Method {self.method_id!r} is engine-native and does not use a Cachet generator")
        if not callable(getattr(generator, "generate", None)):
            raise TypeError("generator must expose a callable generate method")
        generator_pre_rope = getattr(generator, "pre_rope", False)
        if type(generator_pre_rope) is not bool:
            raise TypeError("generator.pre_rope must be a boolean when provided")
        if generator_pre_rope != self.pre_rope:
            raise ValueError(
                f"Method {self.method_id!r} requires pre_rope={self.pre_rope}, "
                f"but generator exposes pre_rope={generator_pre_rope}"
            )
        generator_position_handling = getattr(generator, "position_handling", None)
        if generator_position_handling is None:
            if not require_implemented:
                # Legacy/non-strict workflows may use a generic generator while
                # carrying an experimental method label. Strict registered
                # generation requires the executable position capability.
                return
            generator_position_handling = (
                PositionHandling.REROPE_AT_INJECTION
                if generator_pre_rope
                else PositionHandling.STORED_POST_ROPE
            )
        else:
            generator_position_handling = PositionHandling(
                generator_position_handling
            )
        if generator_position_handling != self.position_handling:
            raise ValueError(
                f"Method {self.method_id!r} requires "
                f"position_handling={self.position_handling.value!r}, but generator "
                f"exposes {generator_position_handling.value!r}"
            )

    def load_generator_factory(self) -> object:
        """Load the declared generator factory without mutating global state."""

        self.require_implemented()
        if self.execution_kind != CACHET_ARTIFACT_EXECUTION:
            raise ValueError(f"Method {self.method_id!r} is engine-native and has no generator factory")
        assert self.generator_factory is not None
        module_name, _, attribute_name = self.generator_factory.partition(":")
        try:
            factory = getattr(import_module(module_name), attribute_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"Could not load generator factory {self.generator_factory!r} "
                f"for method {self.method_id!r}"
            ) from exc
        if not callable(factory):
            raise TypeError(
                f"Generator factory {self.generator_factory!r} for method "
                f"{self.method_id!r} is not callable"
            )
        return factory

    def create_generator(self, *args: Any, **kwargs: Any) -> object:
        """Instantiate and capability-check this method's declared generator."""

        factory = self.load_generator_factory()
        assert callable(factory)
        generator = factory(*args, **kwargs)
        self.validate_generator(generator)
        return generator

    def reuse_plan(self) -> ReusePlan:
        """Emit the typed artifact-to-runtime operations for this method."""

        assert self.position_handling is not None
        if self.execution_kind == ENGINE_NATIVE_EXECUTION:
            recompute_policy = TokenRecomputePolicy.ENGINE_NATIVE
        else:
            recompute_policy = (
                TokenRecomputePolicy.SELECTIVE
                if self.selective_recompute
                else TokenRecomputePolicy.NONE
            )
        return ReusePlan(
            method_id=self.method_id,
            connector_mode=self.connector_mode,
            artifact_format=self.artifact_format,
            position_handling=self.position_handling,
            payload_decode_stage=self.payload_decode_stage,
            token_recompute_policy=recompute_policy,
        )


@dataclass(frozen=True, slots=True)
class MethodRegistry:
    """Immutable registry for built-in and application-provided KV methods."""

    specs: Mapping[str, MethodSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, MethodSpec] = {}
        for method_id, spec in self.specs.items():
            _validate_identifier("method registry key", method_id)
            if not isinstance(spec, MethodSpec):
                raise TypeError(f"Method registry entry {method_id!r} must be a MethodSpec")
            if method_id != spec.method_id:
                raise ValueError(
                    f"Method registry key {method_id!r} does not match spec method {spec.method_id!r}"
                )
            normalized[method_id] = spec
        object.__setattr__(self, "specs", MappingProxyType(normalized))

    def __contains__(self, method: CacheGenerationMethod | str) -> bool:
        return _method_id(method) in self.specs

    def __len__(self) -> int:
        return len(self.specs)

    def get(
        self,
        method: CacheGenerationMethod | str,
        *,
        require_implemented: bool = False,
    ) -> MethodSpec:
        method_id = _method_id(method)
        try:
            spec = self.specs[method_id]
        except KeyError as exc:
            supported = ", ".join(sorted(self.specs))
            raise KeyError(f"Unknown KV reuse method {method_id!r}; registered methods: {supported}") from exc
        return spec.require_implemented() if require_implemented else spec

    def with_spec(self, spec: MethodSpec, *, replace: bool = False) -> "MethodRegistry":
        if not isinstance(spec, MethodSpec):
            raise TypeError("spec must be a MethodSpec")
        entries = dict(self.specs)
        if spec.method_id in entries and entries[spec.method_id] != spec and not replace:
            raise ValueError(f"method {spec.method_id!r} is already registered")
        entries[spec.method_id] = spec
        return MethodRegistry(entries)

    def with_specs(self, specs: Iterable[MethodSpec], *, replace: bool = False) -> "MethodRegistry":
        registry = self
        for spec in specs:
            registry = registry.with_spec(spec, replace=replace)
        return registry

    @property
    def implemented_specs(self) -> tuple[MethodSpec, ...]:
        return tuple(spec for spec in self.specs.values() if spec.implemented)


_BUILTIN_METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        method=CacheGenerationMethod.FULL_PREFIX_PREFILL,
        display_name="full-prefix KV",
        arm_id=DOCUMENT_KV_CACHE_ARM,
        connector_mode=CACHET_CONNECTOR_MODE,
        pre_rope=False,
        selective_recompute=False,
        implemented=True,
        generator_factory=(
            "document_kv_cache.transformers_generator:"
            "build_transformers_kv_chunk_generator"
        ),
        description=(
            "Reuse one KV artifact generated from the complete logical prefix. "
            "This is an exact full-context cached-prefix control and is intentionally "
            "distinct from independently generated per-document segments."
        ),
    ),
    MethodSpec(
        method=CacheGenerationMethod.VANILLA_PREFILL,
        display_name="vanilla KV",
        arm_id=DOCUMENT_KV_CACHE_ARM,
        connector_mode=CACHET_CONNECTOR_MODE,
        pre_rope=False,
        selective_recompute=False,
        implemented=True,
        generator_factory=(
            "document_kv_cache.transformers_generator:"
            "build_transformers_kv_chunk_generator"
        ),
        description=(
            "Reuse per-document KV computed independently and stored post-RoPE; the "
            "connector reports the document/system prefix as already computed and "
            "hydrates it into GPU KV. Correct for single-document (true-prefix) reuse; "
            "multi-document quality is limited by missing cross-document attention."
        ),
    ),
    MethodSpec(
        method=CacheGenerationMethod.KV_PACKET,
        display_name="KV Packet",
        arm_id=DOCUMENT_KV_CACHE_ARM,
        connector_mode=CACHET_CONNECTOR_MODE,
        pre_rope=False,
        selective_recompute=False,
        implemented=False,
        description=(
            "Planned: no executable Cachet implementation is present. Pin and "
            "reproduce the upstream implementation before defining the artifact, "
            "positioning, and serving contracts."
        ),
    ),
    MethodSpec(
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
    MethodSpec(
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
    MethodSpec(
        method=CacheGenerationMethod.LMCACHE,
        display_name="LMCache",
        arm_id=DOCUMENT_KV_CACHE_ARM,
        connector_mode=LMCACHE_CONNECTOR_MODE,
        pre_rope=False,
        selective_recompute=False,
        implemented=True,
        artifact_version="external-v1",
        execution_kind=ENGINE_NATIVE_EXECUTION,
        artifact_format=ENGINE_NATIVE_ARTIFACT_FORMAT,
        payload_decode_stage=PayloadDecodeStage.ENGINE_NATIVE,
        description=(
            "Reuse engine-generated KV through LMCache's native vLLM connector. "
            "The serving engine owns cache creation, persistence, and loading; Cachet "
            "owns the shared experiment contract and evidence."
        ),
    ),
)

DEFAULT_METHOD_REGISTRY = MethodRegistry().with_specs(_BUILTIN_METHOD_SPECS)

# Compatibility view keyed by the historical enum. It is immutable so callers
# cannot silently change the process-wide method contract.
METHOD_SPECS: Mapping[CacheGenerationMethod, MethodSpec] = MappingProxyType(
    {
        CacheGenerationMethod(spec.method_id): spec
        for spec in _BUILTIN_METHOD_SPECS
    }
)


def method_spec(method: CacheGenerationMethod | str) -> MethodSpec:
    """Return the :class:`MethodSpec` for a benchmark cache-generation method.

    Accepts an enum member or its string value. Raises ``KeyError`` for cache-
    generation labels that are not stand-alone benchmark methods (ADAPTER_TRAINED,
    CUSTOM) or for unknown methods.
    """
    return DEFAULT_METHOD_REGISTRY.get(method)


def builtin_method_specs() -> Mapping[str, MethodSpec]:
    """Return the immutable built-in method mapping keyed by stable method id."""

    return DEFAULT_METHOD_REGISTRY.specs


def default_method_registry() -> MethodRegistry:
    """Return the immutable process-default method registry."""

    return DEFAULT_METHOD_REGISTRY
