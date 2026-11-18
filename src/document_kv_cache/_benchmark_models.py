from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Literal

from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    CACHE_REUSE_ARM,
    DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    BenchmarkArm,
    BenchmarkComparison,
    BenchmarkReportRow,
    BenchmarkSuite,
    DatasetMetricSpec,
    InferenceMeasurement,
)


class _FrozenList(list[Any]):
    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("evaluation JSON values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable  # type: ignore[assignment]
    __imul__ = _immutable  # type: ignore[assignment]
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


BENCHMARK_DECODE_SETTING_KEYS = frozenset(
    {
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "stop",
        "stop_token_ids",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "ignore_eos",
    }
)

# Every arm carries an authenticated identity for the static, non-decoding
# request customizations sent to its serving backend.  Empty configuration is
# still explicit so a missing identity cannot be confused with "no custom
# settings" in sanitized evidence.
EMPTY_REQUEST_CUSTOMIZATION_DIGEST = sha256(b"{}").hexdigest()


def _validate_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")


def _validate_positive_int(value: Any, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _validate_non_negative_finite_number(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be a non-negative finite number")


def _validate_positive_finite_number(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive finite number")


def _validate_sha256_digest(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _json_object_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        normalized[key] = _json_compatible_value(item, f"{field_name}.{key}")
    return normalized


def _json_compatible_value(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be JSON-compatible")
        return value
    if isinstance(value, Mapping):
        return _json_object_mapping(value, field_name)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return [
            _json_compatible_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{field_name} must be JSON-compatible")


def _deep_freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _deep_freeze_json_value(item) for key, item in value.items()}
    )


def _deep_freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _deep_freeze_json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return _FrozenList(_deep_freeze_json_value(item) for item in value)
    return value


def _validated_decode_settings(
    value: Mapping[str, Any],
    field_name: str = "decode_settings",
) -> dict[str, Any]:
    normalized = _json_object_mapping(value, field_name)
    unknown = sorted(set(normalized).difference(BENCHMARK_DECODE_SETTING_KEYS))
    if unknown:
        raise ValueError(
            f"{field_name} contains unsupported settings: {', '.join(unknown)}"
        )
    for key in ("top_p", "min_p"):
        item = normalized.get(key)
        if item is not None and (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not 0 <= float(item) <= 1
        ):
            raise ValueError(f"{field_name}.{key} must be a number in [0, 1]")
    top_k = normalized.get("top_k")
    if top_k is not None and (type(top_k) is not int or top_k < -1):
        raise ValueError(f"{field_name}.top_k must be an integer >= -1")
    seed = normalized.get("seed")
    if seed is not None and type(seed) is not int:
        raise ValueError(f"{field_name}.seed must be an integer")
    ignore_eos = normalized.get("ignore_eos")
    if ignore_eos is not None and type(ignore_eos) is not bool:
        raise ValueError(f"{field_name}.ignore_eos must be a boolean")
    stop = normalized.get("stop")
    if stop is not None and not (
        isinstance(stop, str)
        or (
            isinstance(stop, Sequence)
            and not isinstance(stop, (str, bytes, bytearray, memoryview))
            and all(isinstance(item, str) for item in stop)
        )
    ):
        raise ValueError(f"{field_name}.stop must be a string or list of strings")
    stop_token_ids = normalized.get("stop_token_ids")
    if stop_token_ids is not None and not (
        isinstance(stop_token_ids, Sequence)
        and not isinstance(
            stop_token_ids,
            (str, bytes, bytearray, memoryview),
        )
        and all(type(item) is int and item >= 0 for item in stop_token_ids)
    ):
        raise ValueError(
            f"{field_name}.stop_token_ids must be a list of non-negative integers"
        )
    for key in ("frequency_penalty", "presence_penalty"):
        item = normalized.get(key)
        if item is not None and (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{field_name}.{key} must be finite numeric")
    repetition_penalty = normalized.get("repetition_penalty")
    if repetition_penalty is not None and (
        isinstance(repetition_penalty, bool)
        or not isinstance(repetition_penalty, (int, float))
        or not math.isfinite(float(repetition_penalty))
        or float(repetition_penalty) <= 0
    ):
        raise ValueError(
            f"{field_name}.repetition_penalty must be positive finite numeric"
        )
    return normalized


def _decoding_config_digest(
    *,
    max_output_tokens: int | None,
    temperature: float | None,
    stream: bool | None,
    generation_seed: int | None,
    decode_settings: Mapping[str, Any],
) -> str:
    normalized_settings = _json_object_mapping(
        decode_settings,
        "decode_settings",
    )
    encoded = json.dumps(
        {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "stream": stream,
            "generation_seed": generation_seed,
            "decode_settings": normalized_settings,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


BENCHMARK_ARM_ENVIRONMENT_FIELDS = (
    "served_model_id",
    "canonical_model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "lora_id",
    "prompt_template_version",
    "engine_id",
    "engine_version",
    "serving_platform",
    "hardware_target",
    "hardware_fingerprint",
    "model_dtype",
    "model_quantization",
    "runtime_kv_dtype",
    "layout_version",
    "payload_axis_order",
    "block_size",
    "key_position_encoding",
    "rope_theta",
    "rope_rotary_dim",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "runtime_version",
    "storage_identity",
    "cache_state",
)

# A declared setting is an anchored experimental dimension, not always one raw
# provenance field. Honest hardware runs necessarily have different hardware
# fingerprints (and usually a different local-storage identity); changing serving
# platforms necessarily changes the engine identity. Fields outside the selected
# bundle remain invariant.
BENCHMARK_SETTING_DIMENSION_FIELDS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "hardware_target": frozenset(
            {"hardware_target", "hardware_fingerprint", "storage_identity"}
        ),
        "serving_platform": frozenset(
            {
                "serving_platform",
                "engine_id",
                "engine_version",
                "runtime_version",
            }
        ),
        "model_quantization": frozenset(
            {"model_quantization", "model_dtype", "served_model_id"}
        ),
    }
)


@dataclass(frozen=True, slots=True)
class BenchmarkArmEnvironment:
    """Actual immutable model, serving, hardware, and KV runtime for one arm."""

    served_model_id: str
    canonical_model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    lora_id: str
    prompt_template_version: str
    engine_id: str
    engine_version: str
    serving_platform: str
    hardware_target: str
    hardware_fingerprint: str
    model_dtype: str
    model_quantization: str
    runtime_kv_dtype: str
    layout_version: str
    payload_axis_order: str
    block_size: int | None
    key_position_encoding: str
    rope_theta: float | None
    rope_rotary_dim: int | None
    tensor_parallel_size: int | None
    pipeline_parallel_size: int | None
    runtime_version: str
    storage_identity: str
    cache_state: str

    def __post_init__(self) -> None:
        for field_name in BENCHMARK_ARM_ENVIRONMENT_FIELDS:
            value = getattr(self, field_name)
            if field_name in {
                "block_size",
                "rope_theta",
                "rope_rotary_dim",
                "tensor_parallel_size",
                "pipeline_parallel_size",
            }:
                if value is not None and field_name != "rope_theta":
                    _validate_positive_int(value, field_name)
                continue
            _validate_non_empty_string(value, field_name)
        if (self.rope_theta is None) != (self.rope_rotary_dim is None):
            raise ValueError(
                "rope_theta and rope_rotary_dim must be provided together"
            )
        if self.rope_theta is not None:
            _validate_positive_finite_number(self.rope_theta, "rope_theta")
            if self.rope_rotary_dim is None or self.rope_rotary_dim % 2:
                raise ValueError("rope_rotary_dim must be a positive even integer")
        if self.key_position_encoding == "pre_rope" and self.rope_theta is None:
            raise ValueError("pre_rope runtime environments require RoPE geometry")

    @property
    def has_unresolved_provenance(self) -> bool:
        return any(
            getattr(self, field_name) == "unresolved"
            for field_name in BENCHMARK_ARM_ENVIRONMENT_FIELDS
            if field_name
            not in {
                "block_size",
                "rope_theta",
                "rope_rotary_dim",
                "tensor_parallel_size",
                "pipeline_parallel_size",
            }
        ) or any(
            getattr(self, field_name) is None
            for field_name in (
                "block_size",
                "tensor_parallel_size",
                "pipeline_parallel_size",
            )
        )


@dataclass(frozen=True, slots=True)
class BenchmarkManifestContext:
    """Reproducibility metadata shared by every arm in one experiment."""

    model_revision: str = "unresolved"
    canonical_model_id: str = ""
    tokenizer_id: str = "unresolved"
    tokenizer_revision: str = "unresolved"
    lora_id: str = "base"
    engine_id: str = "unresolved"
    engine_version: str = "unresolved"
    serving_platform: str = "unresolved"
    model_dtype: str = "unresolved"
    model_quantization: str = "none"
    runtime_kv_dtype: str = "unresolved"
    layout_version: str = "unresolved"
    payload_axis_order: str = "unresolved"
    block_size: int | None = None
    key_position_encoding: str = "unresolved"
    rope_theta: float | None = None
    rope_rotary_dim: int | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    package_revisions: tuple[tuple[str, str], ...] = ()
    prompt_template_version: str = DEFAULT_V1_PROMPT_TEMPLATE_VERSION
    input_tokens_target: int | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = None
    generation_seed: int | None = None
    decode_settings: Mapping[str, Any] = field(default_factory=dict)
    hardware_fingerprint: str = "unresolved"
    runtime_id: str = "unresolved"
    runtime_version: str = "unresolved"
    storage_identity: str = "unresolved"
    cache_state: str = "unresolved"
    complete_dataset_split: bool = False
    measurement_scopes: tuple[str, ...] = ("latency", "quality")
    comparison_mode: Literal[
        "methods_same_setting",
        "single_method_setting_variation",
    ] = "methods_same_setting"
    varied_setting: str = ""
    reference_arm_id: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "lora_id",
            "engine_id",
            "engine_version",
            "serving_platform",
            "model_dtype",
            "model_quantization",
            "runtime_kv_dtype",
            "layout_version",
            "payload_axis_order",
            "key_position_encoding",
            "prompt_template_version",
            "hardware_fingerprint",
            "runtime_id",
            "runtime_version",
            "storage_identity",
            "cache_state",
        ):
            _validate_non_empty_string(getattr(self, field_name), field_name)
        if self.canonical_model_id:
            _validate_non_empty_string(self.canonical_model_id, "canonical_model_id")
        for field_name in (
            "block_size",
            "rope_rotary_dim",
            "tensor_parallel_size",
            "pipeline_parallel_size",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _validate_positive_int(value, field_name)
        if (self.rope_theta is None) != (self.rope_rotary_dim is None):
            raise ValueError(
                "rope_theta and rope_rotary_dim must be provided together"
            )
        if self.rope_theta is not None:
            _validate_positive_finite_number(self.rope_theta, "rope_theta")
            if self.rope_rotary_dim is None or self.rope_rotary_dim % 2:
                raise ValueError("rope_rotary_dim must be a positive even integer")
        if self.key_position_encoding == "pre_rope" and self.rope_theta is None:
            raise ValueError("pre_rope manifest context requires RoPE geometry")
        revisions = tuple(self.package_revisions)
        names: list[str] = []
        for name, revision in revisions:
            _validate_non_empty_string(name, "package_revisions name")
            _validate_non_empty_string(revision, f"package_revisions.{name}")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError(
                "package_revisions must not contain duplicate package names"
            )
        object.__setattr__(self, "package_revisions", tuple(sorted(revisions)))
        if self.input_tokens_target is not None:
            _validate_positive_int(self.input_tokens_target, "input_tokens_target")
        if self.max_output_tokens is not None:
            _validate_positive_int(self.max_output_tokens, "max_output_tokens")
        if self.temperature is not None:
            _validate_non_negative_finite_number(self.temperature, "temperature")
        if self.stream is not None and type(self.stream) is not bool:
            raise ValueError("stream must be a boolean when provided")
        if self.generation_seed is not None and type(self.generation_seed) is not int:
            raise ValueError("generation_seed must be an integer when provided")
        object.__setattr__(
            self,
            "decode_settings",
            _deep_freeze_json_mapping(
                _validated_decode_settings(self.decode_settings)
            ),
        )
        if type(self.complete_dataset_split) is not bool:
            raise ValueError("complete_dataset_split must be a boolean")
        scopes = tuple(self.measurement_scopes)
        if not scopes or any(
            scope not in {"latency", "quality", "resource"} for scope in scopes
        ):
            raise ValueError(
                "measurement_scopes must contain latency, quality, and/or resource"
            )
        if len(set(scopes)) != len(scopes):
            raise ValueError("measurement_scopes must not contain duplicates")
        object.__setattr__(self, "measurement_scopes", scopes)
        if self.comparison_mode not in {
            "methods_same_setting",
            "single_method_setting_variation",
        }:
            raise ValueError("unsupported comparison_mode")
        if self.comparison_mode == "single_method_setting_variation":
            _validate_non_empty_string(self.varied_setting, "varied_setting")
        elif self.varied_setting:
            raise ValueError(
                "varied_setting is only valid for setting-variation comparisons"
            )
        if self.reference_arm_id:
            _validate_non_empty_string(self.reference_arm_id, "reference_arm_id")


@dataclass(frozen=True, slots=True)
class BenchmarkArmManifest:
    arm_id: str
    implementation_kind: str
    uses_cache: bool
    method_id: str
    method_version: str
    method_config_digest: str
    artifact_ids: tuple[str, ...]
    variant_id: str
    connector_mode: str
    physical_transform_id: str
    physical_transform_version: str
    declared_physical_transform_config_digest: str
    physical_transform_config_digest: str
    request_customization_digest: str
    scorer_plugin_path: str
    offline_training_seconds: float | None
    offline_artifact_generation_seconds: float | None
    offline_checkpoint_load_seconds: float | None
    artifact_bytes: int | None
    offline_peak_memory_bytes: int | None
    source_revision: str
    checkpoint_identity: str
    setting_overrides: Mapping[str, Any]
    requires_cachet_handoff: bool
    runtime_environment: BenchmarkArmEnvironment

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_environment, BenchmarkArmEnvironment):
            raise TypeError("runtime_environment must be BenchmarkArmEnvironment")
        if (
            not isinstance(self.request_customization_digest, str)
            or len(self.request_customization_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.request_customization_digest
            )
        ):
            raise ValueError(
                "request_customization_digest must be a lowercase SHA-256 digest"
            )


@dataclass(frozen=True, slots=True)
class BenchmarkScorerManifest:
    dataset: str
    scorer_id: str
    version: str
    plugin_path: str
    publication_approved: bool
    metric_specs: tuple[DatasetMetricSpec, ...]
    prompt_plugin_path: str
    prompt_template_version: str


@dataclass(frozen=True, slots=True)
class BenchmarkExperimentManifest:
    experiment_id: str
    baseline_arm_id: str
    comparison_mode: str
    varied_setting: str
    sample_selection_digest: str
    dataset_sample_digests: tuple[tuple[str, str], ...]
    datasets: tuple[str, ...]
    example_count: int
    complete_dataset_split: bool
    measurement_scopes: tuple[str, ...]
    prompt_template_version: str
    scorer_identities: tuple[BenchmarkScorerManifest, ...]
    input_tokens_target: int | None
    output_tokens_target: int | None
    temperature: float | None
    stream: bool | None
    generation_seed: int | None
    decode_settings: Mapping[str, Any]
    decoding_config_digest: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    engine_id: str
    engine_version: str
    package_revisions: tuple[tuple[str, str], ...]
    hardware_target: str
    hardware_fingerprint: str
    runtime_id: str
    runtime_version: str
    storage_identity: str
    cache_state: str
    request_parallelism: int
    repeats: int
    warmups: int
    isolate_arms: bool
    order_mode: str
    shuffle: bool
    benchmark_seed: int | None
    arms: tuple[BenchmarkArmManifest, ...]
    execution_isolation_mode: Literal[
        "shared_process_sequential",
        "shared_process_concurrent",
        "separate_process_or_job",
    ] = "shared_process_sequential"
    source_execution_ids: tuple[tuple[str, str], ...] = ()
    resource_evidence_ids: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("input_tokens_target", "output_tokens_target"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_positive_int(value, field_name)
        if self.temperature is not None:
            _validate_non_negative_finite_number(self.temperature, "temperature")
        if self.stream is not None and type(self.stream) is not bool:
            raise ValueError("stream must be a boolean when provided")
        if self.generation_seed is not None and type(self.generation_seed) is not int:
            raise ValueError("generation_seed must be an integer when provided")
        normalized = _validated_decode_settings(self.decode_settings)
        expected_digest = _decoding_config_digest(
            max_output_tokens=self.output_tokens_target,
            temperature=self.temperature,
            stream=self.stream,
            generation_seed=self.generation_seed,
            decode_settings=normalized,
        )
        if self.decoding_config_digest != expected_digest:
            raise ValueError(
                "decoding_config_digest does not match the recorded decoding settings"
            )
        object.__setattr__(
            self,
            "decode_settings",
            _deep_freeze_json_mapping(normalized),
        )
        arm_ids = {arm.arm_id for arm in self.arms}
        resource_arm_ids: list[str] = []
        for arm_id, evidence_digest in self.resource_evidence_ids:
            _validate_non_empty_string(arm_id, "resource_evidence_ids arm_id")
            _validate_sha256_digest(
                evidence_digest,
                f"resource_evidence_ids.{arm_id}",
            )
            if arm_id not in arm_ids:
                raise ValueError(
                    "resource_evidence_ids references unknown arm "
                    f"{arm_id!r}"
                )
            resource_arm_ids.append(arm_id)
        if len(set(resource_arm_ids)) != len(resource_arm_ids):
            raise ValueError("resource_evidence_ids must not contain duplicate arm ids")
        object.__setattr__(
            self,
            "resource_evidence_ids",
            tuple(sorted(self.resource_evidence_ids)),
        )

    @property
    def has_unresolved_provenance(self) -> bool:
        required = (
            self.model_revision,
            self.tokenizer_id,
            self.tokenizer_revision,
            self.engine_id,
            self.engine_version,
            self.hardware_fingerprint,
            self.runtime_id,
            self.runtime_version,
            self.storage_identity,
            self.cache_state,
        )
        return any(value == "unresolved" for value in required) or any(
            arm.runtime_environment.has_unresolved_provenance for arm in self.arms
        )


@dataclass(frozen=True, slots=True)
class BenchmarkExecutionWindow:
    arm_id: str
    wall_seconds: float
    completion_tokens: int
    successful_requests: int
    started_at_seconds: float | None = None
    ended_at_seconds: float | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.arm_id, "arm_id")
        _validate_positive_finite_number(self.wall_seconds, "wall_seconds")
        if type(self.completion_tokens) is not int or self.completion_tokens < 0:
            raise ValueError("completion_tokens must be a non-negative integer")
        if type(self.successful_requests) is not int or self.successful_requests < 0:
            raise ValueError("successful_requests must be a non-negative integer")
        if (self.started_at_seconds is None) != (self.ended_at_seconds is None):
            raise ValueError(
                "started_at_seconds and ended_at_seconds must be provided together"
            )
        if self.started_at_seconds is not None:
            ended_at_seconds = self.ended_at_seconds
            if ended_at_seconds is None:  # pragma: no cover - paired above.
                raise ValueError("ended_at_seconds is required")
            _validate_non_negative_finite_number(
                self.started_at_seconds,
                "started_at_seconds",
            )
            _validate_positive_finite_number(
                ended_at_seconds,
                "ended_at_seconds",
            )
            if ended_at_seconds <= self.started_at_seconds:
                raise ValueError("ended_at_seconds must be after started_at_seconds")

    @property
    def aggregate_output_tokens_per_second(self) -> float:
        return self.completion_tokens / self.wall_seconds


@dataclass(frozen=True, slots=True)
class BenchmarkResourceEvidence:
    """Hash-bound resource measurements for one physical benchmark arm."""

    experiment_id: str
    arm_id: str
    execution_id_digest: str
    measurement_started_at_seconds: float
    measurement_ended_at_seconds: float
    sampling_interval_seconds: float
    first_sample_at_seconds: float
    last_sample_at_seconds: float
    max_sample_gap_seconds: float
    expected_sample_count: int
    sample_count: int
    error_count: int
    complete: bool
    telemetry_sha256: str
    peak_gpu_process_memory_bytes: int
    mean_gpu_utilization_percent: float
    peak_gpu_utilization_percent: float
    peak_process_tree_rss_bytes: int
    peak_host_memory_used_bytes: int
    source_revision: str
    source_tree_sha256: str
    wheel_sha256: str
    runner_sha256: str
    runtime_identity_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "arm_id",
            "source_revision",
        ):
            _validate_non_empty_string(getattr(self, field_name), field_name)
        for field_name in (
            "execution_id_digest",
            "telemetry_sha256",
            "source_tree_sha256",
            "wheel_sha256",
            "runner_sha256",
            "runtime_identity_sha256",
        ):
            _validate_sha256_digest(getattr(self, field_name), field_name)
        for field_name in (
            "measurement_started_at_seconds",
            "first_sample_at_seconds",
            "last_sample_at_seconds",
            "max_sample_gap_seconds",
            "mean_gpu_utilization_percent",
            "peak_gpu_utilization_percent",
        ):
            _validate_non_negative_finite_number(getattr(self, field_name), field_name)
        for field_name in (
            "measurement_ended_at_seconds",
            "sampling_interval_seconds",
        ):
            _validate_positive_finite_number(getattr(self, field_name), field_name)
        if self.measurement_ended_at_seconds <= self.measurement_started_at_seconds:
            raise ValueError(
                "measurement_ended_at_seconds must be after "
                "measurement_started_at_seconds"
            )
        if self.first_sample_at_seconds > self.last_sample_at_seconds:
            raise ValueError("first_sample_at_seconds must not exceed last_sample_at_seconds")
        for field_name in (
            "expected_sample_count",
            "sample_count",
        ):
            _validate_positive_int(getattr(self, field_name), field_name)
        if type(self.error_count) is not int or self.error_count < 0:
            raise ValueError("error_count must be a non-negative integer")
        for field_name in (
            "peak_gpu_process_memory_bytes",
            "peak_process_tree_rss_bytes",
            "peak_host_memory_used_bytes",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if type(self.complete) is not bool:
            raise ValueError("complete must be a boolean")
        if not 0 <= self.mean_gpu_utilization_percent <= 100:
            raise ValueError("mean_gpu_utilization_percent must be in [0, 100]")
        if not 0 <= self.peak_gpu_utilization_percent <= 100:
            raise ValueError("peak_gpu_utilization_percent must be in [0, 100]")
        if self.mean_gpu_utilization_percent > self.peak_gpu_utilization_percent:
            raise ValueError(
                "mean_gpu_utilization_percent must not exceed the peak"
            )
        if self.complete:
            if self.error_count:
                raise ValueError("complete resource evidence cannot contain errors")
            if self.sample_count < self.expected_sample_count:
                raise ValueError(
                    "complete resource evidence requires the expected sample count"
                )
            if (
                self.first_sample_at_seconds
                > self.measurement_started_at_seconds + self.sampling_interval_seconds
            ):
                raise ValueError(
                    "complete resource evidence does not cover the measurement start"
                )
            if (
                self.last_sample_at_seconds
                < self.measurement_ended_at_seconds - self.sampling_interval_seconds
            ):
                raise ValueError(
                    "complete resource evidence does not cover the measurement end"
                )
            if self.max_sample_gap_seconds > self.sampling_interval_seconds * 2:
                raise ValueError(
                    "complete resource evidence has a gap larger than two intervals"
                )


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    suite: BenchmarkSuite
    measurements: tuple[InferenceMeasurement, ...]
    report_rows: tuple[BenchmarkReportRow, ...]
    comparisons: tuple[BenchmarkComparison, ...]
    baseline_arm_id: str = BASELINE_PREFILL_ARM
    cache_arm_id: str = CACHE_REUSE_ARM
    request_parallelism: int = 1
    isolate_arms: bool = True
    arms: tuple[BenchmarkArm, ...] = ()
    repeats: int = 1
    shuffle: bool = False
    seed: int | None = None
    interleave_examples: bool = False
    prefix_cache_salt_mode: str = "static"
    warmups: int = 0
    experiment_manifest: BenchmarkExperimentManifest | None = None
    evidence_policy: Literal["smoke", "canary", "publication"] = "smoke"
    execution_windows: tuple[BenchmarkExecutionWindow, ...] = ()
    execution_isolation_mode: Literal[
        "shared_process_sequential",
        "shared_process_concurrent",
        "separate_process_or_job",
    ] = "shared_process_sequential"
    resource_evidence: tuple[BenchmarkResourceEvidence, ...] = ()

    def __post_init__(self) -> None:
        arm_ids: list[str] = []
        for evidence in self.resource_evidence:
            if not isinstance(evidence, BenchmarkResourceEvidence):
                raise TypeError(
                    "resource_evidence entries must be BenchmarkResourceEvidence"
                )
            arm_ids.append(evidence.arm_id)
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("resource_evidence must not contain duplicate arm ids")

    @property
    def cache_arm_ids(self) -> tuple[str, ...]:
        if self.arms:
            return tuple(
                arm.arm_id for arm in self.arms if arm.arm_id != self.baseline_arm_id
            )
        return (self.cache_arm_id,)

    @property
    def reference_arm_id(self) -> str:
        return self.baseline_arm_id


# Keep the historical public import path stable for callers that inspect these
# facade-exported classes.
for _public_class in (
    BenchmarkManifestContext,
    BenchmarkArmManifest,
    BenchmarkScorerManifest,
    BenchmarkExperimentManifest,
    BenchmarkExecutionWindow,
    BenchmarkResourceEvidence,
    BenchmarkRunResult,
):
    _public_class.__module__ = "document_kv_cache.benchmark_runner"
