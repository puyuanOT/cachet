from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from hashlib import sha256
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

if TYPE_CHECKING:
    from document_kv_cache.methods import MethodRegistry

from document_kv_cache._benchmark_datasets import (
    _validate_benchmark_jsonl_record as _validate_benchmark_jsonl_record,
    load_benchmark_jsonl,
    load_jsonl_suite,
    load_v1_jsonl_suite,
)
from document_kv_cache._benchmark_manifest import (
    BENCHMARK_EXPERIMENT_MANIFEST_RECORD_TYPE,
    _build_experiment_manifest,
    _kv_transfer_params_for_arm,
    _resolve_reference_arm_id,
    _sha256_json,
    _validate_comparison_design,
    benchmark_experiment_manifest_to_record,
)
from document_kv_cache._benchmark_records import (
    BENCHMARK_RESOURCE_EVIDENCE_RECORD_TYPE,
    BENCHMARK_RUN_RECORD_TYPE,
    benchmark_experiment_manifest_from_record,
    benchmark_gate_inputs_from_record,
    benchmark_record_aggregate_issues,
    benchmark_record_payload_digest,
    benchmark_resource_evidence_from_record,
    benchmark_resource_evidence_to_record,
    benchmark_run_result_from_record,
    benchmark_run_result_payload_to_record,
    benchmark_run_result_to_evidence_record,
    benchmark_run_result_to_record,
    merge_isolated_benchmark_run_records,
    write_benchmark_run_result_json,
)
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    CACHE_REUSE_ARM,
    DEFAULT_HARDWARE_TARGET,
    DEFAULT_V1_MODEL_ID,
    DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    DOCUMENT_KV_ARTIFACT_ID_PARAM,
    DOCUMENT_KV_CACHE_METHOD_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM,
    BenchmarkArm,
    BenchmarkExample,
    BenchmarkPromptParts,
    BenchmarkSuite,
    DatasetScoreContext,
    InferenceMeasurement,
    DatasetScorerRegistry,
    baseline_prefill_arm,
    build_prompt_parts,
    compare_to_baseline,
    document_kv_cache_arm,
    default_dataset_scorer_registry,
    final_answer_measurement_metadata,
    require_runnable_cachet_benchmark_arm,
    summarize_measurements,
    validate_v1_dataset,
    validate_v1_hardware_target,
)
from document_kv_cache._benchmark_models import (
    BENCHMARK_DECODE_SETTING_KEYS,
    EMPTY_REQUEST_CUSTOMIZATION_DIGEST,
    BenchmarkArmEnvironment,
    BenchmarkArmManifest,
    BenchmarkExecutionWindow,
    BenchmarkExperimentManifest,
    BenchmarkManifestContext,
    BenchmarkResourceEvidence,
    BenchmarkRunResult,
    BenchmarkScorerManifest,
    _deep_freeze_json_mapping,
    _json_object_mapping,
    _validate_non_empty_string,
    _validate_non_negative_finite_number,
    _validate_positive_finite_number,
    _validate_positive_int,
    _validate_sha256_digest,
    _validated_decode_settings,
)
from document_kv_cache.publication_inputs import (
    PublicationLatencyExample,
    project_publication_latency_request_order,
)


DEFAULT_OPENAI_COMPLETIONS_ENDPOINT = "/v1/completions"
PREFIX_CACHE_SALT_MODES = ("static", "per_request")

PUBLICATION_LATENCY_SCHEDULE_SHA256_METADATA_KEY = (
    "publication_latency_schedule_sha256"
)
PUBLICATION_LATENCY_REQUESTS_SHA256_METADATA_KEY = (
    "publication_latency_requests_sha256"
)
PUBLICATION_LATENCY_INPUT_BUNDLE_SHA256_METADATA_KEY = (
    "publication_latency_input_bundle_sha256"
)
PUBLICATION_LATENCY_SEED_SHA256_METADATA_KEY = "publication_latency_seed_sha256"
PUBLICATION_LATENCY_DEPLOYMENT_BLOCK_METADATA_KEY = (
    "publication_latency_deployment_block"
)
PUBLICATION_LATENCY_REQUEST_ID_METADATA_KEY = "publication_latency_request_id"
PUBLICATION_LATENCY_REQUEST_INDEX_METADATA_KEY = "publication_latency_request_index"
PUBLICATION_LATENCY_LOGICAL_KEY_SHA256_METADATA_KEY = (
    "publication_latency_logical_key_sha256"
)
PUBLICATION_LATENCY_LANE_METADATA_KEY = "publication_latency_lane"
PUBLICATION_LATENCY_LANE_POSITION_METADATA_KEY = "publication_latency_lane_position"


__all__ = [
    "BENCHMARK_RUN_RECORD_TYPE",
    "BENCHMARK_RESOURCE_EVIDENCE_RECORD_TYPE",
    "BENCHMARK_EXPERIMENT_MANIFEST_RECORD_TYPE",
    "DEFAULT_OPENAI_COMPLETIONS_ENDPOINT",
    "PUBLICATION_LATENCY_SCHEDULE_SHA256_METADATA_KEY",
    "PUBLICATION_LATENCY_REQUESTS_SHA256_METADATA_KEY",
    "PUBLICATION_LATENCY_INPUT_BUNDLE_SHA256_METADATA_KEY",
    "PUBLICATION_LATENCY_SEED_SHA256_METADATA_KEY",
    "PUBLICATION_LATENCY_DEPLOYMENT_BLOCK_METADATA_KEY",
    "PUBLICATION_LATENCY_REQUEST_ID_METADATA_KEY",
    "PUBLICATION_LATENCY_REQUEST_INDEX_METADATA_KEY",
    "PUBLICATION_LATENCY_LOGICAL_KEY_SHA256_METADATA_KEY",
    "PUBLICATION_LATENCY_LANE_METADATA_KEY",
    "PUBLICATION_LATENCY_LANE_POSITION_METADATA_KEY",
    "BenchmarkGeneration",
    "BenchmarkEngineRequest",
    "BenchmarkEngine",
    "BenchmarkRunResult",
    "BenchmarkExecutionWindow",
    "BenchmarkResourceEvidence",
    "BenchmarkArmManifest",
    "BenchmarkArmEnvironment",
    "BenchmarkScorerManifest",
    "BenchmarkExperimentManifest",
    "BenchmarkManifestContext",
    "OpenAICompatibleBenchmarkConfig",
    "OpenAICompatibleEngineFactory",
    "default_benchmark_arms",
    "run_benchmark_suite",
    "load_jsonl_suite",
    "load_v1_jsonl_suite",
    "load_benchmark_jsonl",
    "benchmark_run_result_payload_to_record",
    "benchmark_run_result_to_record",
    "benchmark_run_result_to_evidence_record",
    "benchmark_gate_inputs_from_record",
    "benchmark_record_payload_digest",
    "benchmark_resource_evidence_to_record",
    "benchmark_resource_evidence_from_record",
    "benchmark_experiment_manifest_to_record",
    "benchmark_experiment_manifest_from_record",
    "benchmark_run_result_from_record",
    "merge_isolated_benchmark_run_records",
    "benchmark_record_aggregate_issues",
    "parse_benchmark_arm_specs",
    "validate_arm_extra_body_contract",
    "write_benchmark_run_result_json",
    "run_openai_compatible_benchmark",
    "run_openai_compatible_v1_benchmark",
    "main",
]


@dataclass(frozen=True, slots=True)
class BenchmarkGeneration:
    output_text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_seconds: float
    time_to_completion_seconds: float
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_generation_text(self.output_text, "output_text")
        _validate_generation_non_negative_int(self.prompt_tokens, "prompt_tokens")
        _validate_generation_non_negative_int(
            self.completion_tokens, "completion_tokens"
        )
        _validate_generation_non_negative_finite_number(
            self.ttft_seconds, "ttft_seconds"
        )
        _validate_generation_non_negative_finite_number(
            self.time_to_completion_seconds,
            "time_to_completion_seconds",
        )
        if self.time_to_completion_seconds < self.ttft_seconds:
            raise ValueError(
                "time_to_completion_seconds must be greater than or equal to ttft_seconds"
            )
        object.__setattr__(self, "metadata", _generation_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class BenchmarkEngineRequest:
    suite_id: str
    model_id: str
    hardware_target: str
    example: BenchmarkExample
    arm: BenchmarkArm
    prompt_parts: BenchmarkPromptParts
    request_id: str | None = None
    kv_transfer_params: Mapping[str, Any] = field(default_factory=dict)
    repeat_index: int = 1

    def __post_init__(self) -> None:
        if self.request_id is not None:
            _validate_non_empty_string(self.request_id, "request_id")
        _validate_positive_int(self.repeat_index, "repeat_index")
        object.__setattr__(
            self,
            "kv_transfer_params",
            _deep_freeze_json_mapping(
                _json_object_mapping(self.kv_transfer_params, "kv_transfer_params")
            ),
        )

    @property
    def logical_prompt_text(self) -> str:
        return self.prompt_parts.prefill_prompt

    @property
    def prompt_text(self) -> str:
        return self.runtime_prompt_text

    @property
    def runtime_prompt_text(self) -> str:
        if self.arm.requires_cachet_handoff:
            runtime_prefix_text = self.kv_transfer_params.get(
                DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM
            )
            if isinstance(runtime_prefix_text, str) and runtime_prefix_text:
                return f"{runtime_prefix_text}{self.prompt_parts.cache_suffix_text}"
            return self.prompt_parts.cache_suffix_text
        return self.prompt_parts.prefill_prompt

    @property
    def cache_prefix_text(self) -> str:
        return self.prompt_parts.cache_prefix_text

    @property
    def cache_suffix_text(self) -> str:
        return self.prompt_parts.cache_suffix_text


class BenchmarkEngine(Protocol):
    def generate(self, request: BenchmarkEngineRequest) -> BenchmarkGeneration: ...


@dataclass(frozen=True, slots=True)
class _PublicationLatencyScheduleExecution:
    schedule_sha256: str
    requests_sha256: str
    input_bundle_sha256: str
    seed_sha256: str
    deployment_block: int
    projection: tuple[tuple[str, str, int], ...]
    request_ids: tuple[str, ...]
    lanes: tuple[tuple[int, ...], ...]
    lane_by_request_index: tuple[int, ...]
    lane_position_by_request_index: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OpenAICompatibleBenchmarkConfig:
    suite_id: str
    dataset_paths: Mapping[str, str | Path]
    base_url: str
    cache_base_url: str | None = None
    endpoint: str = DEFAULT_OPENAI_COMPLETIONS_ENDPOINT
    cache_endpoint: str | None = None
    model_id: str = DEFAULT_V1_MODEL_ID
    hardware_target: str = DEFAULT_HARDWARE_TARGET
    limit_per_dataset: int | None = None
    repeats: int = 1
    request_parallelism: int = 1
    arm_ids: tuple[str, ...] = ()
    arms: tuple[BenchmarkArm, ...] = ()
    arm_base_urls: Mapping[str, str] = field(default_factory=dict)
    arm_endpoints: Mapping[str, str] = field(default_factory=dict)
    arm_extra_bodies: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    shuffle: bool = False
    seed: int | None = None
    isolate_arms: bool = True
    api_key: str | None = None
    max_tokens: int = 128
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    stream: bool = True
    cache_runtime_prompt: bool = False
    prompt_token_accounting: Literal["logical", "server_usage"] = "logical"
    baseline_extra_body: Mapping[str, Any] = field(default_factory=dict)
    cache_extra_body: Mapping[str, Any] = field(default_factory=dict)
    prefix_cache_salt_mode: Literal["static", "per_request"] = "static"
    interleave_examples: bool = False
    warmups: int = 0
    evidence_policy: Literal["smoke", "canary", "publication"] = "smoke"
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
    generation_seed: int | None = None
    hardware_fingerprint: str = "unresolved"
    runtime_id: str = "unresolved"
    runtime_version: str = "unresolved"
    storage_identity: str = "unresolved"
    cache_state: str = "unresolved"
    complete_dataset_split: bool = False
    measurement_scopes: tuple[str, ...] = ("latency", "quality")
    comparison_mode: Literal[
        "methods_same_setting", "single_method_setting_variation"
    ] = "methods_same_setting"
    varied_setting: str = ""
    reference_arm_id: str = ""
    suite_contract: Literal["v1", "generalized"] = "v1"
    publication_latency_schedule_record: Mapping[str, Any] | None = None
    publication_latency_schedule_path: str | Path | None = None
    publication_latency_expected_input_bundle_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.suite_id, "suite_id")
        dataset_paths = _validated_dataset_paths(self.dataset_paths)
        if not dataset_paths:
            raise ValueError("dataset_paths must be non-empty")
        if self.suite_contract not in {"v1", "generalized"}:
            raise ValueError("suite_contract must be v1 or generalized")
        if self.suite_contract == "v1":
            for dataset in dataset_paths:
                validate_v1_dataset(dataset)
        _validate_non_empty_string(self.base_url, "base_url")
        if self.cache_base_url is not None:
            _validate_non_empty_string(self.cache_base_url, "cache_base_url")
        _validate_non_empty_string(self.endpoint, "endpoint")
        if self.cache_endpoint is not None:
            _validate_non_empty_string(self.cache_endpoint, "cache_endpoint")
        _validate_non_empty_string(self.model_id, "model_id")
        _validate_non_empty_string(self.hardware_target, "hardware_target")
        if self.suite_contract == "v1":
            validate_v1_hardware_target(self.hardware_target)
        if self.limit_per_dataset is not None and (
            type(self.limit_per_dataset) is not int or self.limit_per_dataset <= 0
        ):
            raise ValueError("limit_per_dataset must be positive when provided")
        _validate_positive_int(self.repeats, "repeats")
        _validate_positive_int(self.request_parallelism, "request_parallelism")
        object.__setattr__(self, "arm_ids", _validated_arm_ids(self.arm_ids))
        arms = tuple(self.arms)
        if self.arm_ids and arms:
            raise ValueError("arm_ids and arms are mutually exclusive")
        if any(not isinstance(arm, BenchmarkArm) for arm in arms):
            raise TypeError("arms entries must be BenchmarkArm")
        if len({arm.arm_id for arm in arms}) != len(arms):
            raise ValueError("arms must not contain duplicate arm ids")
        object.__setattr__(self, "arms", arms)
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer when provided")
        if type(self.shuffle) is not bool:
            raise ValueError("shuffle must be a boolean")
        if type(self.interleave_examples) is not bool:
            raise ValueError("interleave_examples must be a boolean")
        _validate_positive_int(self.max_tokens, "max_tokens")
        _validate_non_negative_finite_number(self.temperature, "temperature")
        _validate_positive_finite_number(self.timeout_seconds, "timeout_seconds")
        if type(self.stream) is not bool:
            raise ValueError("stream must be a boolean")
        if type(self.cache_runtime_prompt) is not bool:
            raise ValueError("cache_runtime_prompt must be a boolean")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ValueError("api_key must be a string when provided")
        if self.prompt_token_accounting not in {"logical", "server_usage"}:
            raise ValueError(
                "prompt_token_accounting must be 'logical' or 'server_usage'"
            )
        if self.prefix_cache_salt_mode not in PREFIX_CACHE_SALT_MODES:
            raise ValueError("prefix_cache_salt_mode must be 'static' or 'per_request'")
        if self.cache_runtime_prompt and self.cache_base_url is None:
            raise ValueError(
                "cache_runtime_prompt requires cache_base_url; pass the cache proxy URL explicitly"
            )
        object.__setattr__(self, "dataset_paths", MappingProxyType(dataset_paths))
        object.__setattr__(
            self,
            "baseline_extra_body",
            _deep_freeze_json_mapping(
                _json_object_mapping(self.baseline_extra_body, "baseline_extra_body")
            ),
        )
        object.__setattr__(
            self,
            "cache_extra_body",
            _deep_freeze_json_mapping(
                _json_object_mapping(self.cache_extra_body, "cache_extra_body")
            ),
        )
        object.__setattr__(
            self,
            "arm_base_urls",
            MappingProxyType(
                _validated_arm_string_mapping(self.arm_base_urls, "arm_base_urls", arms)
            ),
        )
        object.__setattr__(
            self,
            "arm_endpoints",
            MappingProxyType(
                _validated_arm_string_mapping(self.arm_endpoints, "arm_endpoints", arms)
            ),
        )
        object.__setattr__(
            self,
            "arm_extra_bodies",
            MappingProxyType(
                {
                    arm_id: _deep_freeze_json_mapping(body)
                    for arm_id, body in _validated_arm_json_mapping(
                        self.arm_extra_bodies,
                        "arm_extra_bodies",
                        arms,
                    ).items()
                }
            ),
        )
        _validate_comparable_decode_settings(
            self.baseline_extra_body,
            self.cache_extra_body,
            self.arm_extra_bodies,
            self.arms,
        )
        if type(self.warmups) is not int or self.warmups < 0:
            raise ValueError("warmups must be a non-negative integer")
        if self.evidence_policy not in {"smoke", "canary", "publication"}:
            raise ValueError("evidence_policy must be smoke, canary, or publication")
        _validate_publication_latency_config(self)
        # Validate all manifest metadata and normalize package/decode settings once.
        context = self.manifest_context
        object.__setattr__(self, "package_revisions", context.package_revisions)
        resolved_arms = self.arms or _benchmark_arms_for_ids(self.arm_ids)
        _validate_comparison_design(
            resolved_arms,
            comparison_mode=self.comparison_mode,
            varied_setting=self.varied_setting,
            reference_arm_id=self.reference_arm_id,
        )

    @property
    def manifest_context(self) -> BenchmarkManifestContext:
        decode_settings = _common_decode_settings(
            self.baseline_extra_body,
            self.cache_extra_body,
            self.arm_extra_bodies,
            self.arms,
        )
        return BenchmarkManifestContext(
            model_revision=self.model_revision,
            canonical_model_id=self.canonical_model_id,
            tokenizer_id=self.tokenizer_id,
            tokenizer_revision=self.tokenizer_revision,
            lora_id=self.lora_id,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            serving_platform=self.serving_platform,
            model_dtype=self.model_dtype,
            model_quantization=self.model_quantization,
            runtime_kv_dtype=self.runtime_kv_dtype,
            layout_version=self.layout_version,
            payload_axis_order=self.payload_axis_order,
            block_size=self.block_size,
            key_position_encoding=self.key_position_encoding,
            rope_theta=self.rope_theta,
            rope_rotary_dim=self.rope_rotary_dim,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            package_revisions=self.package_revisions,
            prompt_template_version=self.prompt_template_version,
            input_tokens_target=self.input_tokens_target,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=self.stream,
            generation_seed=self.generation_seed,
            decode_settings=decode_settings,
            hardware_fingerprint=self.hardware_fingerprint,
            runtime_id=self.runtime_id,
            runtime_version=self.runtime_version,
            storage_identity=self.storage_identity,
            cache_state=self.cache_state,
            complete_dataset_split=self.complete_dataset_split,
            measurement_scopes=self.measurement_scopes,
            comparison_mode=self.comparison_mode,
            varied_setting=self.varied_setting,
            reference_arm_id=self.reference_arm_id,
        )


OpenAICompatibleEngineFactory = Callable[
    [BenchmarkArm, OpenAICompatibleBenchmarkConfig], BenchmarkEngine
]


def _validated_dataset_paths(value: Any) -> dict[str, str | Path]:
    if not isinstance(value, Mapping):
        raise ValueError("dataset_paths must be a mapping")
    paths: dict[str, str | Path] = {}
    for dataset, path in value.items():
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("dataset_paths keys must be non-empty strings")
        if not isinstance(path, (str, Path)):
            raise ValueError(f"dataset_paths.{dataset} must be a path string or Path")
        paths[dataset] = path
    return paths


def _validate_publication_latency_config(
    config: OpenAICompatibleBenchmarkConfig,
) -> None:
    record = config.publication_latency_schedule_record
    raw_path = config.publication_latency_schedule_path
    expected_bundle_sha256 = (
        config.publication_latency_expected_input_bundle_sha256
    )
    if record is not None and raw_path is not None:
        raise ValueError(
            "publication_latency_schedule_record and "
            "publication_latency_schedule_path are mutually exclusive"
        )
    schedule_enabled = record is not None or raw_path is not None
    if schedule_enabled != (expected_bundle_sha256 is not None):
        raise ValueError(
            "a publication latency schedule and "
            "publication_latency_expected_input_bundle_sha256 must be provided "
            "together"
        )
    if not schedule_enabled:
        return
    assert expected_bundle_sha256 is not None
    _validate_sha256_digest(
        expected_bundle_sha256,
        "publication_latency_expected_input_bundle_sha256",
    )
    if config.shuffle or config.interleave_examples or config.seed is not None:
        raise ValueError(
            "a publication latency schedule owns request order; shuffle, "
            "interleave_examples, and benchmark seed must be disabled"
        )
    if record is not None:
        object.__setattr__(
            config,
            "publication_latency_schedule_record",
            _deep_freeze_json_mapping(
                _json_object_mapping(
                    record,
                    "publication_latency_schedule_record",
                )
            ),
        )
    if raw_path is not None:
        if not isinstance(raw_path, (str, Path)) or not str(raw_path):
            raise ValueError(
                "publication_latency_schedule_path must be a non-empty path"
            )
        object.__setattr__(
            config,
            "publication_latency_schedule_path",
            Path(raw_path).expanduser(),
        )


def _publication_latency_schedule_record_from_config(
    config: OpenAICompatibleBenchmarkConfig,
) -> Mapping[str, Any] | None:
    record = config.publication_latency_schedule_record
    if record is not None:
        return _json_object_mapping(
            record,
            "publication_latency_schedule_record",
        )
    path = config.publication_latency_schedule_path
    if path is None:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not load publication latency schedule from {path}: {exc}"
        ) from exc
    return _json_object_mapping(value, "publication_latency_schedule_path")


_RESERVED_ARM_EXTRA_BODY_FIELDS = frozenset(
    {
        "model",
        "prompt",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "stream",
        "stream_options",
        "request_id",
        "kv_transfer_params",
    }
)


def _validated_arm_string_mapping(
    value: Any,
    field_name: str,
    arms: Sequence[BenchmarkArm],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    known = {arm.arm_id for arm in arms} or {
        arm.arm_id for arm in default_benchmark_arms()
    }
    normalized: dict[str, str] = {}
    for arm_id, item in value.items():
        if arm_id not in known:
            raise ValueError(f"{field_name} references unknown arm {arm_id!r}")
        _validate_non_empty_string(item, f"{field_name}.{arm_id}")
        normalized[arm_id] = item
    return normalized


def _validated_arm_json_mapping(
    value: Any,
    field_name: str,
    arms: Sequence[BenchmarkArm],
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    known = {arm.arm_id for arm in arms} or {
        arm.arm_id for arm in default_benchmark_arms()
    }
    normalized: dict[str, Mapping[str, Any]] = {}
    for arm_id, item in value.items():
        if arm_id not in known:
            raise ValueError(f"{field_name} references unknown arm {arm_id!r}")
        normalized[arm_id] = _json_object_mapping(item, f"{field_name}.{arm_id}")
    return normalized


def _decode_settings(extra_body: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: extra_body[key]
        for key in sorted(BENCHMARK_DECODE_SETTING_KEYS.intersection(extra_body))
    }


def _request_customization_digest(
    extra_body: Mapping[str, Any],
    *,
    dynamic_cache_salt: bool,
) -> str:
    """Hash static backend request customizations without retaining raw values."""

    has_dynamic_cache_salt = (
        dynamic_cache_salt
        and isinstance(extra_body.get("cache_salt"), str)
        and bool(extra_body["cache_salt"])
    )
    customization = {
        key: value
        for key, value in extra_body.items()
        if key not in BENCHMARK_DECODE_SETTING_KEYS
        and not (has_dynamic_cache_salt and key == "cache_salt")
    }
    return _sha256_json(customization)


def _validated_request_customization_digests(
    arms: Sequence[BenchmarkArm],
    value: Mapping[str, str] | None,
    *,
    comparison_mode: str,
    varied_setting: str,
) -> dict[str, str]:
    arm_ids = {arm.arm_id for arm in arms}
    if value is None:
        normalized = {
            arm.arm_id: EMPTY_REQUEST_CUSTOMIZATION_DIGEST for arm in arms
        }
    else:
        if not isinstance(value, Mapping):
            raise TypeError("request_customization_digests must be a mapping or None")
        supplied_ids = set(value)
        if supplied_ids != arm_ids:
            missing = sorted(arm_ids.difference(supplied_ids))
            unexpected = sorted(
                str(item) for item in supplied_ids.difference(arm_ids)
            )
            raise ValueError(
                "request_customization_digests must cover every benchmark arm "
                f"exactly; missing={missing}, unexpected={unexpected}"
            )
        normalized = dict(value)
    for arm_id, digest in normalized.items():
        if (
            not isinstance(arm_id, str)
            or not arm_id
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "request_customization_digests must contain non-empty arm ids "
                "and lowercase SHA-256 digests"
            )
    if (
        comparison_mode == "single_method_setting_variation"
        and varied_setting != "serving_platform"
        and len(set(normalized.values())) > 1
    ):
        raise ValueError(
            "single_method_setting_variation requires invariant static request "
            "customizations unless serving_platform is varied"
        )
    return normalized


def _validate_comparable_decode_settings(
    baseline_extra_body: Mapping[str, Any],
    cache_extra_body: Mapping[str, Any],
    arm_extra_bodies: Mapping[str, Mapping[str, Any]],
    arms: Sequence[BenchmarkArm],
) -> None:
    for label, extra_body in (
        ("baseline_extra_body", baseline_extra_body),
        ("cache_extra_body", cache_extra_body),
        *(
            (f"arm_extra_bodies.{arm_id}", body)
            for arm_id, body in arm_extra_bodies.items()
        ),
    ):
        validate_arm_extra_body_contract(extra_body, label)
    settings = _active_decode_settings(
        baseline_extra_body,
        cache_extra_body,
        arm_extra_bodies,
        arms,
    )
    if settings and any(setting != settings[0] for setting in settings):
        raise ValueError(
            "comparable benchmark arms must use identical decode settings; "
            "method-specific request fields must not alter decoding"
        )


def validate_arm_extra_body_contract(
    extra_body: Mapping[str, Any],
    field_name: str,
) -> None:
    reserved = sorted(_RESERVED_ARM_EXTRA_BODY_FIELDS.intersection(extra_body))
    if reserved:
        raise ValueError(
            f"{field_name} must not override reserved request fields: "
            f"{', '.join(reserved)}"
        )
    custom_params = extra_body.get("custom_params")
    if isinstance(custom_params, Mapping) and "kv_transfer_params" in custom_params:
        raise ValueError(
            f"{field_name}.custom_params must not override reserved kv_transfer_params"
        )
    _validated_decode_settings(_decode_settings(extra_body), field_name)


def _common_decode_settings(
    baseline_extra_body: Mapping[str, Any],
    cache_extra_body: Mapping[str, Any],
    arm_extra_bodies: Mapping[str, Mapping[str, Any]],
    arms: Sequence[BenchmarkArm],
) -> Mapping[str, Any]:
    candidates = _active_decode_settings(
        baseline_extra_body,
        cache_extra_body,
        arm_extra_bodies,
        arms,
    )
    return next((candidate for candidate in candidates if candidate), {})


def _active_decode_settings(
    baseline_extra_body: Mapping[str, Any],
    cache_extra_body: Mapping[str, Any],
    arm_extra_bodies: Mapping[str, Mapping[str, Any]],
    arms: Sequence[BenchmarkArm],
) -> list[dict[str, Any]]:
    active_arms = tuple(arms) or default_benchmark_arms()
    return [
        _decode_settings(
            arm_extra_bodies.get(
                arm.arm_id,
                cache_extra_body if arm.uses_cache else baseline_extra_body,
            )
        )
        for arm in active_arms
    ]


def default_benchmark_arms() -> tuple[BenchmarkArm, ...]:
    return (baseline_prefill_arm(), document_kv_cache_arm())


def _validated_arm_ids(value: Any) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("arm_ids must be a sequence of benchmark arm ids")
    arm_ids: list[str] = []
    for index, arm_id in enumerate(value):
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError(f"arm_ids[{index}] must be a non-empty string")
        arm_ids.append(arm_id)
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError(f"arm_ids must not contain duplicates: {arm_ids}")
    known_arm_ids = {arm.arm_id for arm in default_benchmark_arms()}
    unknown = sorted(set(arm_ids).difference(known_arm_ids))
    if unknown:
        raise ValueError(f"Unknown benchmark arm ids: {unknown}")
    return tuple(arm_ids)


def _benchmark_arms_for_ids(arm_ids: Sequence[str] = ()) -> tuple[BenchmarkArm, ...]:
    arms = default_benchmark_arms()
    if not arm_ids:
        return arms
    by_id = {arm.arm_id: arm for arm in arms}
    return tuple(by_id[arm_id] for arm_id in arm_ids)


def _resolve_publication_latency_schedule(
    suite: BenchmarkSuite,
    *,
    arms: Sequence[BenchmarkArm],
    repeats: int,
    request_parallelism: int,
    isolate_arms: bool,
    shuffle: bool,
    seed: int | None,
    interleave_examples: bool,
    record: Mapping[str, Any] | None,
    expected_input_bundle_sha256: str | None,
) -> _PublicationLatencyScheduleExecution | None:
    if (record is None) != (expected_input_bundle_sha256 is None):
        raise ValueError(
            "publication_latency_schedule_record and "
            "publication_latency_expected_input_bundle_sha256 must be provided "
            "together"
        )
    if record is None:
        return None
    assert expected_input_bundle_sha256 is not None
    _validate_sha256_digest(
        expected_input_bundle_sha256,
        "publication_latency_expected_input_bundle_sha256",
    )
    if shuffle or interleave_examples or seed is not None:
        raise ValueError(
            "a publication latency schedule owns request order; shuffle, "
            "interleave_examples, and benchmark seed must be disabled"
        )
    if len(arms) > 1 and not isolate_arms:
        raise ValueError(
            "publication latency schedules require isolated method phases"
        )
    schedule_examples = tuple(
        PublicationLatencyExample(
            dataset=example.dataset,
            example_id=example.example_id,
        )
        for example in suite.examples
    )
    projection = project_publication_latency_request_order(
        record,
        examples=schedule_examples,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
    )
    expected_keys = {
        (example.dataset, example.example_id, repeat_index)
        for example in suite.examples
        for repeat_index in range(1, repeats + 1)
    }
    if len(projection) != len(expected_keys) or set(projection) != expected_keys:
        raise ValueError(
            "publication latency schedule membership must cover every suite example "
            "and configured repeat exactly once, with no extras"
        )

    lanes_by_parallelism = record.get("lanes")
    if not isinstance(lanes_by_parallelism, Mapping):
        raise ValueError("publication latency schedule lanes must be an object")
    raw_lanes = lanes_by_parallelism.get(str(request_parallelism))
    if not isinstance(raw_lanes, Sequence) or isinstance(
        raw_lanes, (str, bytes, bytearray)
    ):
        raise ValueError(
            "publication latency schedule does not define identity-sticky lanes "
            f"for request_parallelism={request_parallelism}"
        )
    if len(raw_lanes) != request_parallelism:
        raise ValueError(
            "publication latency schedule lane count does not match "
            "request_parallelism"
        )
    lanes: list[tuple[int, ...]] = []
    lane_by_request_index = [-1] * len(projection)
    lane_position_by_request_index = [-1] * len(projection)
    for lane_index, raw_lane in enumerate(raw_lanes):
        if not isinstance(raw_lane, Sequence) or isinstance(
            raw_lane, (str, bytes, bytearray)
        ):
            raise ValueError("publication latency schedule lane must be an array")
        lane: list[int] = []
        for lane_position, request_index in enumerate(raw_lane):
            if (
                type(request_index) is not int
                or request_index < 0
                or request_index >= len(projection)
                or lane_by_request_index[request_index] != -1
            ):
                raise ValueError(
                    "publication latency schedule lanes must contain each request "
                    "index exactly once"
                )
            lane.append(request_index)
            lane_by_request_index[request_index] = lane_index
            lane_position_by_request_index[request_index] = lane_position
        lanes.append(tuple(lane))
    if any(lane_index < 0 for lane_index in lane_by_request_index):
        raise ValueError(
            "publication latency schedule lanes are missing request indices"
        )

    raw_requests = record.get("requests")
    if not isinstance(raw_requests, Sequence) or isinstance(
        raw_requests, (str, bytes, bytearray)
    ):
        raise ValueError("publication latency schedule requests must be an array")
    request_ids: list[str] = []
    for request_index, raw_request in enumerate(raw_requests):
        if not isinstance(raw_request, Mapping):
            raise ValueError("publication latency schedule request must be an object")
        request_id = raw_request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(
                f"publication latency schedule request {request_index} has no request_id"
            )
        request_ids.append(request_id)

    schedule_sha256 = record.get("closed_record_sha256")
    requests_sha256 = record.get("requests_sha256")
    seed_sha256 = record.get("seed_sha256")
    input_bundle_sha256 = record.get("input_bundle_sha256")
    for field_name, value in (
        ("closed_record_sha256", schedule_sha256),
        ("requests_sha256", requests_sha256),
        ("seed_sha256", seed_sha256),
        ("input_bundle_sha256", input_bundle_sha256),
    ):
        _validate_sha256_digest(value, f"publication_latency_schedule.{field_name}")
    deployment_block = record.get("deployment_block")
    if type(deployment_block) is not int or deployment_block <= 0:
        raise ValueError(
            "publication latency schedule deployment_block must be positive"
        )
    assert isinstance(schedule_sha256, str)
    assert isinstance(requests_sha256, str)
    assert isinstance(seed_sha256, str)
    assert isinstance(input_bundle_sha256, str)
    return _PublicationLatencyScheduleExecution(
        schedule_sha256=schedule_sha256,
        requests_sha256=requests_sha256,
        input_bundle_sha256=input_bundle_sha256,
        seed_sha256=seed_sha256,
        deployment_block=deployment_block,
        projection=projection,
        request_ids=tuple(request_ids),
        lanes=tuple(lanes),
        lane_by_request_index=tuple(lane_by_request_index),
        lane_position_by_request_index=tuple(lane_position_by_request_index),
    )


def _build_default_benchmark_requests(
    suite: BenchmarkSuite,
    *,
    arms: Sequence[BenchmarkArm],
    repeats: int,
    shuffle: bool,
    seed: int | None,
    scorer_registry: DatasetScorerRegistry,
    allow_legacy_cache_params: bool,
) -> list[BenchmarkEngineRequest]:
    requests: list[BenchmarkEngineRequest] = []
    for example in suite.examples:
        prompt_parts = build_prompt_parts(
            example,
            scorer=scorer_registry.get(example.dataset),
        )
        arm_sequence = list(arms) * repeats
        if shuffle:
            random.Random(
                _example_seed(seed, example.dataset, example.example_id)
            ).shuffle(arm_sequence)
        repeat_indices_by_arm = {arm.arm_id: 0 for arm in arms}
        for arm in arm_sequence:
            repeat_indices_by_arm[arm.arm_id] += 1
            requests.append(
                _build_benchmark_engine_request(
                    suite,
                    example=example,
                    arm=arm,
                    prompt_parts=prompt_parts,
                    repeat_index=repeat_indices_by_arm[arm.arm_id],
                    allow_legacy_cache_params=allow_legacy_cache_params,
                )
            )
    return requests


def _build_publication_latency_requests(
    suite: BenchmarkSuite,
    *,
    arms: Sequence[BenchmarkArm],
    scorer_registry: DatasetScorerRegistry,
    allow_legacy_cache_params: bool,
    schedule: _PublicationLatencyScheduleExecution,
) -> list[BenchmarkEngineRequest]:
    examples_by_key = {
        (example.dataset, example.example_id): example for example in suite.examples
    }
    prompt_parts_by_key = {
        key: build_prompt_parts(
            example,
            scorer=scorer_registry.get(example.dataset),
        )
        for key, example in examples_by_key.items()
    }
    requests: list[BenchmarkEngineRequest] = []
    for dataset, example_id, repeat_index in schedule.projection:
        example = examples_by_key[(dataset, example_id)]
        prompt_parts = prompt_parts_by_key[(dataset, example_id)]
        for arm in arms:
            requests.append(
                _build_benchmark_engine_request(
                    suite,
                    example=example,
                    arm=arm,
                    prompt_parts=prompt_parts,
                    repeat_index=repeat_index,
                    allow_legacy_cache_params=allow_legacy_cache_params,
                )
            )
    return requests


def _build_benchmark_engine_request(
    suite: BenchmarkSuite,
    *,
    example: BenchmarkExample,
    arm: BenchmarkArm,
    prompt_parts: BenchmarkPromptParts,
    repeat_index: int,
    allow_legacy_cache_params: bool,
) -> BenchmarkEngineRequest:
    kv_transfer_params = _kv_transfer_params_for_arm(
        example,
        arm,
        allow_legacy=allow_legacy_cache_params,
    )
    return BenchmarkEngineRequest(
        suite_id=suite.suite_id,
        model_id=suite.model_id,
        hardware_target=suite.hardware_target,
        example=example,
        arm=arm,
        prompt_parts=prompt_parts,
        request_id=_request_id_for_arm(
            suite_id=suite.suite_id,
            example=example,
            arm=arm,
            repeat_index=repeat_index,
            kv_transfer_params=kv_transfer_params,
        ),
        kv_transfer_params=kv_transfer_params,
        repeat_index=repeat_index,
    )


def run_benchmark_suite(
    suite: BenchmarkSuite,
    engines: Mapping[str, BenchmarkEngine],
    *,
    arms: Sequence[BenchmarkArm] = default_benchmark_arms(),
    repeats: int = 1,
    request_parallelism: int = 1,
    shuffle: bool = False,
    seed: int | None = None,
    isolate_arms: bool = True,
    interleave_examples: bool = False,
    prefix_cache_salt_mode: str = "static",
    warmups: int = 0,
    scorer_registry: DatasetScorerRegistry | None = None,
    manifest_context: BenchmarkManifestContext | None = None,
    evidence_policy: Literal["smoke", "canary", "publication"] = "smoke",
    reference_arm_id: str | None = None,
    method_registry: MethodRegistry | None = None,
    request_customization_digests: Mapping[str, str] | None = None,
    publication_latency_schedule_record: Mapping[str, Any] | None = None,
    publication_latency_expected_input_bundle_sha256: str | None = None,
) -> BenchmarkRunResult:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    _validate_positive_int(request_parallelism, "request_parallelism")
    if prefix_cache_salt_mode not in PREFIX_CACHE_SALT_MODES:
        raise ValueError("prefix_cache_salt_mode must be 'static' or 'per_request'")
    if type(warmups) is not int or warmups < 0:
        raise ValueError("warmups must be a non-negative integer")
    if evidence_policy not in {"smoke", "canary", "publication"}:
        raise ValueError("evidence_policy must be smoke, canary, or publication")
    for arm in arms:
        require_runnable_cachet_benchmark_arm(
            arm,
            registry=method_registry,
            allow_unidentified_smoke=evidence_policy == "smoke",
        )
    _validate_engine_mapping(arms, engines)
    resolved_context = manifest_context or BenchmarkManifestContext()
    resolved_request_customization_digests = (
        _validated_request_customization_digests(
            arms,
            request_customization_digests,
            comparison_mode=resolved_context.comparison_mode,
            varied_setting=resolved_context.varied_setting,
        )
    )
    requested_reference = reference_arm_id or resolved_context.reference_arm_id
    resolved_reference_arm_id = _resolve_reference_arm_id(
        arms,
        requested_reference,
        comparison_mode=resolved_context.comparison_mode,
    )
    _validate_comparison_design(
        arms,
        comparison_mode=resolved_context.comparison_mode,
        varied_setting=resolved_context.varied_setting,
        reference_arm_id=resolved_reference_arm_id,
    )
    _validate_example_arm_params(suite, arms)
    allow_legacy_cache_params = (
        sum(1 for arm in arms if arm.requires_cachet_handoff) == 1
    )
    scorers = scorer_registry or default_dataset_scorer_registry()
    if not isinstance(scorers, DatasetScorerRegistry):
        raise TypeError("scorer_registry must be DatasetScorerRegistry")
    for dataset in suite.datasets:
        scorers.get(dataset)
    _validate_scorer_prompt_template_versions(
        suite,
        scorers,
        manifest_context=resolved_context,
    )
    publication_schedule = _resolve_publication_latency_schedule(
        suite,
        arms=arms,
        repeats=repeats,
        request_parallelism=request_parallelism,
        isolate_arms=isolate_arms,
        shuffle=shuffle,
        seed=seed,
        interleave_examples=interleave_examples,
        record=publication_latency_schedule_record,
        expected_input_bundle_sha256=(
            publication_latency_expected_input_bundle_sha256
        ),
    )
    if warmups:
        _run_benchmark_warmups(
            suite,
            arms,
            engines,
            scorers,
            warmups=warmups,
            allow_legacy_cache_params=allow_legacy_cache_params,
        )
    if publication_schedule is None:
        requests = _build_default_benchmark_requests(
            suite,
            arms=arms,
            repeats=repeats,
            shuffle=shuffle,
            seed=seed,
            scorer_registry=scorers,
            allow_legacy_cache_params=allow_legacy_cache_params,
        )
    else:
        requests = _build_publication_latency_requests(
            suite,
            arms=arms,
            scorer_registry=scorers,
            allow_legacy_cache_params=allow_legacy_cache_params,
            schedule=publication_schedule,
        )
    # Example interleaving (opt-in): the loop above emits each example's repeats
    # contiguously, so a request_parallelism=N wave would hydrate the SAME document
    # set N times concurrently. Round-robin the requests across examples (per arm)
    # so each concurrent wave draws from distinct documents while every hydrate stays
    # honestly cold (page-cache eviction + per-request cache_salt). Order is otherwise
    # unchanged, so the set of measurements is identical to the grouped ordering.
    if interleave_examples:
        requests = _interleave_requests_by_example(requests)
    # Arm isolation (default): run each arm's requests as a separate concurrency
    # phase instead of interleaving all arms through one shared executor. Co-scheduling
    # the cache arm alongside the baseline arm makes cache-arm requests queue behind
    # baseline full-prefill requests on the shared serving engine, which inflates the
    # measured cache-arm TTFT and hides Cachet's real speedup. Isolating arms measures
    # each arm the way it would actually be deployed (one arm per server). The set of
    # measurements is identical; only the execution order/contention differs.
    if isolate_arms and len(arms) > 1:
        measurements = []
        execution_windows: list[BenchmarkExecutionWindow] = []
        requests_by_arm: dict[str, list[BenchmarkEngineRequest]] = {}
        for request in requests:
            requests_by_arm.setdefault(request.arm.arm_id, []).append(request)
        for arm in arms:
            arm_requests = requests_by_arm.get(arm.arm_id)
            if arm_requests:
                window_started_at = time.time()
                window_started = time.monotonic()
                arm_measurements = _run_requests(
                    arm_requests,
                    engines,
                    scorer_registry=scorers,
                    request_parallelism=request_parallelism,
                    publication_schedule=publication_schedule,
                )
                window_seconds = max(time.monotonic() - window_started, 1e-12)
                window_ended_at = time.time()
                measurements.extend(arm_measurements)
                execution_windows.append(
                    _execution_window(
                        arm.arm_id,
                        arm_measurements,
                        wall_seconds=window_seconds,
                        started_at_seconds=window_started_at,
                        ended_at_seconds=window_ended_at,
                    )
                )
    else:
        window_started_at = time.time()
        window_started = time.monotonic()
        measurements = _run_requests(
            requests,
            engines,
            scorer_registry=scorers,
            request_parallelism=request_parallelism,
            publication_schedule=publication_schedule,
        )
        window_seconds = max(time.monotonic() - window_started, 1e-12)
        window_ended_at = time.time()
        if len(arms) == 1:
            execution_windows = [
                _execution_window(
                    arms[0].arm_id,
                    measurements,
                    wall_seconds=window_seconds,
                    started_at_seconds=window_started_at,
                    ended_at_seconds=window_ended_at,
                )
            ]
        else:
            execution_windows = [
                _execution_window(
                    "all_arms",
                    measurements,
                    wall_seconds=window_seconds,
                    started_at_seconds=window_started_at,
                    ended_at_seconds=window_ended_at,
                )
            ]
    aggregate_throughput = {
        window.arm_id: window.aggregate_output_tokens_per_second
        for window in execution_windows
    }
    report_rows = tuple(
        replace(
            row,
            aggregate_output_tokens_per_second=aggregate_throughput.get(row.arm_id),
        )
        for row in summarize_measurements(measurements)
    )
    baseline_arm_id = resolved_reference_arm_id
    cache_arm_ids = tuple(arm.arm_id for arm in arms if arm.arm_id != baseline_arm_id)
    cache_arm_id = cache_arm_ids[0] if cache_arm_ids else CACHE_REUSE_ARM
    comparisons = tuple(
        comparison
        for candidate_arm_id in cache_arm_ids
        for comparison in compare_to_baseline(
            report_rows,
            baseline_arm_id=baseline_arm_id,
            cache_arm_id=candidate_arm_id,
        )
    )
    experiment_manifest = _build_experiment_manifest(
        suite,
        arms=arms,
        measurements=measurements,
        scorer_registry=scorers,
        context=resolved_context,
        request_parallelism=request_parallelism,
        repeats=repeats,
        warmups=warmups,
        isolate_arms=isolate_arms if len(arms) > 1 else True,
        shuffle=shuffle,
        seed=seed,
        interleave_examples=interleave_examples,
        baseline_arm_id=baseline_arm_id,
        request_customization_digests=(
            resolved_request_customization_digests
        ),
    )
    return BenchmarkRunResult(
        suite=suite,
        measurements=tuple(measurements),
        report_rows=report_rows,
        comparisons=comparisons,
        baseline_arm_id=baseline_arm_id,
        cache_arm_id=cache_arm_id,
        request_parallelism=request_parallelism,
        isolate_arms=isolate_arms if len(arms) > 1 else True,
        arms=tuple(arms),
        repeats=repeats,
        shuffle=shuffle,
        seed=seed,
        interleave_examples=interleave_examples,
        prefix_cache_salt_mode=prefix_cache_salt_mode,
        warmups=warmups,
        experiment_manifest=experiment_manifest,
        evidence_policy=evidence_policy,
        execution_windows=tuple(execution_windows),
        execution_isolation_mode=(
            "shared_process_sequential"
            if isolate_arms or len(arms) == 1
            else "shared_process_concurrent"
        ),
    )


def _execution_window(
    arm_id: str,
    measurements: Sequence[InferenceMeasurement],
    *,
    wall_seconds: float,
    started_at_seconds: float,
    ended_at_seconds: float,
) -> BenchmarkExecutionWindow:
    successful = tuple(measurement for measurement in measurements if measurement.ok)
    return BenchmarkExecutionWindow(
        arm_id=arm_id,
        wall_seconds=wall_seconds,
        completion_tokens=sum(
            measurement.completion_tokens for measurement in successful
        ),
        successful_requests=len(successful),
        started_at_seconds=started_at_seconds,
        ended_at_seconds=ended_at_seconds,
    )


def _validate_scorer_prompt_template_versions(
    suite: BenchmarkSuite,
    scorer_registry: DatasetScorerRegistry,
    *,
    manifest_context: BenchmarkManifestContext,
) -> None:
    declared = {
        scorer_registry.get(dataset).prompt_template_version
        for dataset in suite.datasets
        if scorer_registry.get(dataset).prompt_template_version
    }
    if len(declared) > 1:
        raise ValueError(
            "benchmark scorers must declare one shared prompt_template_version; "
            f"got {sorted(declared)}"
        )
    if declared and next(iter(declared)) != manifest_context.prompt_template_version:
        raise ValueError(
            "scorer prompt_template_version must match "
            "manifest_context.prompt_template_version"
        )


def run_openai_compatible_v1_benchmark(
    config: OpenAICompatibleBenchmarkConfig,
    *,
    engine_factory: OpenAICompatibleEngineFactory | None = None,
    method_registry: MethodRegistry | None = None,
) -> BenchmarkRunResult:
    if config.suite_contract != "v1":
        raise ValueError(
            "run_openai_compatible_v1_benchmark requires suite_contract='v1'"
        )
    return run_openai_compatible_benchmark(
        config,
        scorer_registry=default_dataset_scorer_registry(),
        engine_factory=engine_factory,
        method_registry=method_registry,
    )


def run_openai_compatible_benchmark(
    config: OpenAICompatibleBenchmarkConfig,
    *,
    scorer_registry: DatasetScorerRegistry,
    engine_factory: OpenAICompatibleEngineFactory | None = None,
    method_registry: MethodRegistry | None = None,
) -> BenchmarkRunResult:
    """Run the generalized OpenAI-compatible path with an explicit scorer registry."""

    if not isinstance(scorer_registry, DatasetScorerRegistry):
        raise TypeError("scorer_registry must be DatasetScorerRegistry")
    suite_loader = (
        load_v1_jsonl_suite if config.suite_contract == "v1" else load_jsonl_suite
    )
    suite = suite_loader(
        suite_id=config.suite_id,
        paths=config.dataset_paths,
        model_id=config.model_id,
        hardware_target=config.hardware_target,
        limit_per_dataset=config.limit_per_dataset,
    )
    publication_latency_schedule_record = (
        _publication_latency_schedule_record_from_config(config)
    )
    factory = engine_factory or _openai_compatible_engine
    arms = config.arms or _benchmark_arms_for_ids(config.arm_ids)
    request_customization_digests = {
        arm.arm_id: _request_customization_digest(
            _effective_arm_extra_body(arm, config),
            dynamic_cache_salt=config.prefix_cache_salt_mode == "per_request",
        )
        for arm in arms
    }
    engines = {arm.arm_id: factory(arm, config) for arm in arms}
    return run_benchmark_suite(
        suite,
        engines,
        arms=arms,
        repeats=config.repeats,
        request_parallelism=config.request_parallelism,
        shuffle=config.shuffle,
        seed=config.seed,
        isolate_arms=config.isolate_arms,
        interleave_examples=config.interleave_examples,
        prefix_cache_salt_mode=config.prefix_cache_salt_mode,
        warmups=config.warmups,
        scorer_registry=scorer_registry,
        manifest_context=config.manifest_context,
        evidence_policy=config.evidence_policy,
        reference_arm_id=config.reference_arm_id or None,
        method_registry=method_registry,
        request_customization_digests=request_customization_digests,
        publication_latency_schedule_record=(
            publication_latency_schedule_record
        ),
        publication_latency_expected_input_bundle_sha256=(
            config.publication_latency_expected_input_bundle_sha256
        ),
    )


def _interleave_requests_by_example(
    requests: Sequence[BenchmarkEngineRequest],
) -> list[BenchmarkEngineRequest]:
    """Round-robin requests across (arm, example) groups, preserving each group's order.

    Groups are keyed by ``(arm_id, dataset, example_id)`` and consumed cyclically, so
    consecutive requests come from distinct examples. Grouping preserves first-seen
    order and each group keeps its internal (repeat) order, so the result is a stable
    permutation of the input with identical membership.
    """
    groups: dict[tuple[str, str, str], list[BenchmarkEngineRequest]] = {}
    for request in requests:
        key = (request.arm.arm_id, request.example.dataset, request.example.example_id)
        groups.setdefault(key, []).append(request)
    interleaved: list[BenchmarkEngineRequest] = []
    for cohort in zip_longest(*groups.values()):
        interleaved.extend(request for request in cohort if request is not None)
    return interleaved


def _run_requests(
    requests: Sequence[BenchmarkEngineRequest],
    engines: Mapping[str, BenchmarkEngine],
    *,
    scorer_registry: DatasetScorerRegistry,
    request_parallelism: int,
    publication_schedule: _PublicationLatencyScheduleExecution | None = None,
) -> list[InferenceMeasurement]:
    if publication_schedule is not None:
        return _run_publication_latency_requests(
            requests,
            engines,
            scorer_registry=scorer_registry,
            schedule=publication_schedule,
        )
    if request_parallelism == 1:
        return [
            _run_engine(
                request,
                engines[request.arm.arm_id],
                scorer_registry=scorer_registry,
            )
            for request in requests
        ]
    with ThreadPoolExecutor(max_workers=request_parallelism) as executor:
        return list(
            executor.map(
                lambda request: _run_engine(
                    request,
                    engines[request.arm.arm_id],
                    scorer_registry=scorer_registry,
                ),
                requests,
            )
        )


def _run_publication_latency_requests(
    requests: Sequence[BenchmarkEngineRequest],
    engines: Mapping[str, BenchmarkEngine],
    *,
    scorer_registry: DatasetScorerRegistry,
    schedule: _PublicationLatencyScheduleExecution,
) -> list[InferenceMeasurement]:
    actual_projection = tuple(
        (
            request.example.dataset,
            request.example.example_id,
            request.repeat_index,
        )
        for request in requests
    )
    if actual_projection != schedule.projection:
        raise ValueError(
            "publication latency execution requests must match the closed logical "
            "order exactly, without omissions, extras, or reordering"
        )

    def run_lane(
        lane_item: tuple[int, tuple[int, ...]],
    ) -> tuple[tuple[int, InferenceMeasurement], ...]:
        lane_index, request_indices = lane_item
        lane_measurements: list[tuple[int, InferenceMeasurement]] = []
        for lane_position, request_index in enumerate(request_indices):
            request = requests[request_index]
            measurement = _run_engine(
                request,
                engines[request.arm.arm_id],
                scorer_registry=scorer_registry,
            )
            lane_measurements.append(
                (
                    request_index,
                    _with_publication_latency_provenance(
                        measurement,
                        schedule=schedule,
                        request_index=request_index,
                        lane_index=lane_index,
                        lane_position=lane_position,
                    ),
                )
            )
        return tuple(lane_measurements)

    lane_items = tuple(enumerate(schedule.lanes))
    completed_lanes: tuple[tuple[tuple[int, InferenceMeasurement], ...], ...]
    if len(lane_items) == 1:
        completed_lanes = (run_lane(lane_items[0]),)
    else:
        # One closed-loop worker owns each identity-sticky lane.  Because every
        # identity's repeats are assigned to one lane, variable service times can
        # never make two requests for the same example overlap.
        with ThreadPoolExecutor(max_workers=len(lane_items)) as executor:
            completed_lanes = tuple(executor.map(run_lane, lane_items))
    ordered: list[InferenceMeasurement | None] = [None] * len(requests)
    for lane_measurements in completed_lanes:
        for request_index, measurement in lane_measurements:
            if ordered[request_index] is not None:
                raise RuntimeError(
                    "publication latency lane execution produced a duplicate request"
                )
            ordered[request_index] = measurement
    if any(measurement is None for measurement in ordered):
        raise RuntimeError(
            "publication latency lane execution omitted a scheduled request"
        )
    return [measurement for measurement in ordered if measurement is not None]


def _with_publication_latency_provenance(
    measurement: InferenceMeasurement,
    *,
    schedule: _PublicationLatencyScheduleExecution,
    request_index: int,
    lane_index: int,
    lane_position: int,
) -> InferenceMeasurement:
    expected_lane = schedule.lane_by_request_index[request_index]
    expected_lane_position = schedule.lane_position_by_request_index[request_index]
    if lane_index != expected_lane or lane_position != expected_lane_position:
        raise RuntimeError("publication latency lane provenance does not match schedule")
    dataset, example_id, repeat_index = schedule.projection[request_index]
    logical_key_sha256 = _sha256_json(
        {
            "dataset": dataset,
            "example_id": example_id,
            "repeat_index": repeat_index,
        }
    )
    metadata = dict(measurement.metadata)
    metadata.update(
        {
            PUBLICATION_LATENCY_SCHEDULE_SHA256_METADATA_KEY: (
                schedule.schedule_sha256
            ),
            PUBLICATION_LATENCY_REQUESTS_SHA256_METADATA_KEY: (
                schedule.requests_sha256
            ),
            PUBLICATION_LATENCY_INPUT_BUNDLE_SHA256_METADATA_KEY: (
                schedule.input_bundle_sha256
            ),
            PUBLICATION_LATENCY_SEED_SHA256_METADATA_KEY: schedule.seed_sha256,
            PUBLICATION_LATENCY_DEPLOYMENT_BLOCK_METADATA_KEY: str(
                schedule.deployment_block
            ),
            PUBLICATION_LATENCY_REQUEST_ID_METADATA_KEY: schedule.request_ids[
                request_index
            ],
            PUBLICATION_LATENCY_REQUEST_INDEX_METADATA_KEY: str(request_index),
            PUBLICATION_LATENCY_LOGICAL_KEY_SHA256_METADATA_KEY: (
                logical_key_sha256
            ),
            PUBLICATION_LATENCY_LANE_METADATA_KEY: str(lane_index),
            PUBLICATION_LATENCY_LANE_POSITION_METADATA_KEY: str(lane_position),
        }
    )
    return replace(measurement, metadata=metadata)


def _run_engine(
    request: BenchmarkEngineRequest,
    engine: BenchmarkEngine,
    *,
    scorer_registry: DatasetScorerRegistry,
) -> InferenceMeasurement:
    cache_method, artifact_id = _request_cache_identity(request)
    try:
        generation = engine.generate(request)
    except Exception as exc:  # pragma: no cover - exercised through tests with concrete exception type.
        return InferenceMeasurement(
            example_id=request.example.example_id,
            dataset=request.example.dataset,
            arm_id=request.arm.arm_id,
            prompt_tokens=0,
            completion_tokens=0,
            ttft_seconds=0.0,
            time_to_completion_seconds=0.0,
            output_text="",
            expected_answer=request.example.expected_answer,
            references=request.example.references,
            error=_exception_message(exc),
            metadata={"error_type": type(exc).__name__},
            cache_method=cache_method,
            artifact_id=artifact_id,
            variant_id=request.arm.variant_id,
            request_id=request.request_id or "",
            repeat_index=request.repeat_index,
        )
    scorer = scorer_registry.get(request.example.dataset)
    extraction = scorer.parse_answer(generation.output_text)
    scorer_output = (
        generation.output_text
        if extraction is None
        else extraction.extracted_answer
    )
    quality_scores = (
        scorer.zero_scores()
        if extraction is not None and not extraction.valid
        else scorer.score(
            DatasetScoreContext(
                dataset=request.example.dataset,
                example_id=request.example.example_id,
                output_text=scorer_output,
                references=request.example.references,
                metadata=request.example.metadata,
            )
        )
    )
    metadata = dict(generation.metadata)
    metadata.update(
        {
            "logical_prompt_sha256": sha256(
                request.logical_prompt_text.encode("utf-8")
            ).hexdigest(),
            "runtime_prompt_sha256": sha256(
                request.runtime_prompt_text.encode("utf-8")
            ).hexdigest(),
            "physical_transform_id": request.arm.physical_transform_id,
            "physical_transform_version": request.arm.physical_transform_version,
        }
    )
    if extraction is not None:
        metadata.update(final_answer_measurement_metadata(extraction))
    niah_cell_id = request.example.metadata.get("niah_cell_id")
    if request.example.dataset == "niah" and niah_cell_id is not None:
        metadata["niah_cell_id"] = niah_cell_id
    return InferenceMeasurement(
        example_id=request.example.example_id,
        dataset=request.example.dataset,
        arm_id=request.arm.arm_id,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        ttft_seconds=generation.ttft_seconds,
        time_to_completion_seconds=generation.time_to_completion_seconds,
        output_text=generation.output_text,
        expected_answer=request.example.expected_answer,
        references=request.example.references,
        metadata=metadata,
        cache_method=cache_method,
        artifact_id=artifact_id,
        variant_id=request.arm.variant_id,
        request_id=request.request_id or "",
        repeat_index=request.repeat_index,
        scorer_id=scorer.scorer_id,
        scorer_version=scorer.version,
        quality_scores=quality_scores,
    )


def _run_benchmark_warmups(
    suite: BenchmarkSuite,
    arms: Sequence[BenchmarkArm],
    engines: Mapping[str, BenchmarkEngine],
    scorer_registry: DatasetScorerRegistry,
    *,
    warmups: int,
    allow_legacy_cache_params: bool,
) -> None:
    example = suite.examples[0]
    prompt_parts = build_prompt_parts(
        example,
        scorer=scorer_registry.get(example.dataset),
    )
    for arm in arms:
        for warmup_index in range(1, warmups + 1):
            kv_transfer_params = _kv_transfer_params_for_arm(
                example,
                arm,
                allow_legacy=allow_legacy_cache_params,
            )
            request = BenchmarkEngineRequest(
                suite_id=f"{suite.suite_id}:warmup",
                model_id=suite.model_id,
                hardware_target=suite.hardware_target,
                example=example,
                arm=arm,
                prompt_parts=prompt_parts,
                request_id=_request_id_for_arm(
                    suite_id=f"{suite.suite_id}:warmup",
                    example=example,
                    arm=arm,
                    repeat_index=warmup_index,
                    kv_transfer_params=kv_transfer_params,
                ),
                kv_transfer_params=kv_transfer_params,
                repeat_index=warmup_index,
            )
            measurement = _run_engine(
                request,
                engines[arm.arm_id],
                scorer_registry=scorer_registry,
            )
            if not measurement.ok:
                raise RuntimeError(
                    f"warmup {warmup_index} for arm {arm.arm_id!r} failed: "
                    f"{measurement.error}"
                )


def _request_cache_identity(request: BenchmarkEngineRequest) -> tuple[str, str]:
    if not request.arm.uses_cache:
        return "", ""
    cache_method = request.kv_transfer_params.get(
        DOCUMENT_KV_CACHE_METHOD_PARAM,
        request.arm.cache_method,
    )
    artifact_id = request.kv_transfer_params.get(DOCUMENT_KV_ARTIFACT_ID_PARAM, "")
    if not isinstance(cache_method, str):
        raise ValueError(
            f"kv_transfer_params.{DOCUMENT_KV_CACHE_METHOD_PARAM} must be a string"
        )
    if not isinstance(artifact_id, str):
        raise ValueError(
            f"kv_transfer_params.{DOCUMENT_KV_ARTIFACT_ID_PARAM} must be a string"
        )
    if (
        request.arm.cache_method
        and cache_method
        and request.arm.cache_method != cache_method
    ):
        raise ValueError(
            f"Benchmark arm method {request.arm.cache_method!r} does not match "
            f"handoff method {cache_method!r}"
        )
    return cache_method, artifact_id


def _exception_message(exc: Exception) -> str:
    return str(exc) or type(exc).__name__


def _validate_generation_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")


def _validate_generation_non_negative_int(value: Any, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_generation_non_negative_finite_number(
    value: Any, field_name: str
) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be a non-negative finite number")


def _generation_metadata(metadata: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    normalized = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError(f"metadata.{key} must be a string")
        normalized[key] = value
    return MappingProxyType(normalized)


def _effective_arm_extra_body(
    arm: BenchmarkArm,
    config: OpenAICompatibleBenchmarkConfig,
) -> Mapping[str, Any]:
    return config.arm_extra_bodies.get(
        arm.arm_id,
        config.cache_extra_body if arm.uses_cache else config.baseline_extra_body,
    )


def _openai_compatible_engine(
    arm: BenchmarkArm, config: OpenAICompatibleBenchmarkConfig
) -> BenchmarkEngine:
    from document_kv_cache.openai_compatible import (  # Local import avoids an import cycle.
        OpenAICompatibleCompletionEngine,
        OpenAICompatibleEngineConfig,
    )

    default_base_url = (
        config.cache_base_url
        if arm.uses_cache and config.cache_base_url is not None
        else config.base_url
    )
    default_endpoint = (
        config.cache_endpoint
        if arm.uses_cache and config.cache_endpoint is not None
        else config.endpoint
    )
    base_url = config.arm_base_urls.get(arm.arm_id, default_base_url)
    endpoint = config.arm_endpoints.get(arm.arm_id, default_endpoint)
    prompt_text_mode: Literal["logical", "runtime"] = (
        "runtime" if arm.uses_cache and config.cache_runtime_prompt else "logical"
    )
    extra_body = dict(_effective_arm_extra_body(arm, config))
    if config.generation_seed is not None:
        configured_seed = extra_body.get("seed")
        if configured_seed is not None and configured_seed != config.generation_seed:
            raise ValueError(
                f"arm {arm.arm_id!r} extra_body seed conflicts with generation_seed"
            )
        extra_body["seed"] = config.generation_seed
    extra_body_factory = (
        _prefix_cache_salt_extra_body_factory(extra_body)
        if config.prefix_cache_salt_mode == "per_request"
        else None
    )
    return OpenAICompatibleCompletionEngine(
        OpenAICompatibleEngineConfig(
            base_url=_normalize_openai_base_url(base_url, endpoint=endpoint),
            endpoint=endpoint,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stream=config.stream,
            model_id=config.model_id,
            prompt_text_mode=prompt_text_mode,
            prompt_token_accounting=config.prompt_token_accounting,
            extra_body=extra_body,
        ),
        extra_body_factory=extra_body_factory,
    )


def _prefix_cache_salt_extra_body_factory(
    base_extra_body: Mapping[str, Any],
) -> Callable[[BenchmarkEngineRequest], Mapping[str, Any]]:
    base_cache_salt = base_extra_body.get("cache_salt")
    if not isinstance(base_cache_salt, str) or not base_cache_salt:
        return lambda request: {}

    def extra_body(request: BenchmarkEngineRequest) -> Mapping[str, Any]:
        return {
            "cache_salt": (
                f"{base_cache_salt}:"
                f"{request.suite_id}:"
                f"{request.example.dataset}:"
                f"{request.example.example_id}:"
                f"{request.arm.arm_id}:"
                f"repeat-{request.repeat_index}"
            )
        }

    return extra_body


def _normalize_openai_base_url(base_url: str, *, endpoint: str) -> str:
    stripped = base_url.rstrip("/")
    if endpoint == DEFAULT_OPENAI_COMPLETIONS_ENDPOINT and stripped.endswith("/v1"):
        return stripped[:-3]
    return stripped


def _validate_engine_mapping(
    arms: Sequence[BenchmarkArm], engines: Mapping[str, BenchmarkEngine]
) -> None:
    missing = [arm.arm_id for arm in arms if arm.arm_id not in engines]
    if missing:
        raise ValueError(f"Missing benchmark engines for arms: {missing}")
    arm_ids = [arm.arm_id for arm in arms]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError(f"Duplicate benchmark arm ids: {arm_ids}")


def _validate_example_arm_params(
    suite: BenchmarkSuite,
    arms: Sequence[BenchmarkArm],
) -> None:
    arms_by_id = {arm.arm_id: arm for arm in arms}
    cache_arm_ids = {arm.arm_id for arm in arms if arm.requires_cachet_handoff}
    for example in suite.examples:
        unknown = set(example.arm_kv_transfer_params).difference(arms_by_id)
        if unknown:
            raise ValueError(
                f"example {example.dataset}:{example.example_id} has request params "
                f"for unknown arms: {sorted(unknown)}"
            )
        non_cache = {
            arm_id
            for arm_id in example.arm_kv_transfer_params
            if not arms_by_id[arm_id].requires_cachet_handoff
        }
        if non_cache:
            raise ValueError(
                f"example {example.dataset}:{example.example_id} assigns KV transfer "
                f"params to non-cache arms: {sorted(non_cache)}"
            )
        if len(cache_arm_ids) > 1:
            missing = cache_arm_ids.difference(example.arm_kv_transfer_params)
            if missing:
                raise ValueError(
                    f"example {example.dataset}:{example.example_id} must declare distinct "
                    f"arm_kv_transfer_params for every cache arm; missing {sorted(missing)}"
                )
        for arm_id, params in example.arm_kv_transfer_params.items():
            declared_method = params.get(DOCUMENT_KV_CACHE_METHOD_PARAM)
            expected_method = arms_by_id[arm_id].cache_method
            if declared_method is not None and declared_method != expected_method:
                raise ValueError(
                    f"example {example.dataset}:{example.example_id} arm {arm_id!r} "
                    f"declares cache method {declared_method!r}, expected "
                    f"{expected_method!r}"
                )
        if len(cache_arm_ids) == 1 and example.kv_transfer_params:
            arm_id = next(iter(cache_arm_ids))
            declared_method = example.kv_transfer_params.get(
                DOCUMENT_KV_CACHE_METHOD_PARAM
            )
            expected_method = arms_by_id[arm_id].cache_method
            if declared_method is not None and declared_method != expected_method:
                raise ValueError(
                    f"example {example.dataset}:{example.example_id} legacy KV params "
                    f"declare cache method {declared_method!r}, expected "
                    f"{expected_method!r} for arm {arm_id!r}"
                )


def _arm_id_for_prefill(arms: Sequence[BenchmarkArm]) -> str:
    for arm in arms:
        if not arm.uses_cache:
            return arm.arm_id
    return BASELINE_PREFILL_ARM


def _arm_id_for_cache(arms: Sequence[BenchmarkArm]) -> str:
    for arm in arms:
        if arm.uses_cache:
            return arm.arm_id
    return CACHE_REUSE_ARM


def _request_id_for_arm(
    *,
    suite_id: str,
    example: BenchmarkExample,
    arm: BenchmarkArm,
    repeat_index: int,
    kv_transfer_params: Mapping[str, Any],
) -> str | None:
    if not arm.uses_cache:
        return None
    handoff_request_id = kv_transfer_params.get(DOCUMENT_KV_REQUEST_ID_PARAM)
    if handoff_request_id is None:
        return None
    if not isinstance(handoff_request_id, str) or not handoff_request_id:
        raise ValueError(
            f"kv_transfer_params.{DOCUMENT_KV_REQUEST_ID_PARAM} must be a non-empty string"
        )
    return (
        f"{suite_id}:"
        f"{example.dataset}:"
        f"{example.example_id}:"
        f"{arm.arm_id}:"
        f"repeat-{repeat_index}:"
        f"{handoff_request_id}"
    )


def _example_seed(seed: int | None, dataset: str, example_id: str) -> int:
    value = 0 if seed is None else seed
    for value_part in (dataset, "\0", example_id):
        for character in value_part:
            value = (value * 33 + ord(character)) & 0xFFFFFFFF
    return value


def main(argv: Sequence[str] | None = None) -> int:
    from document_kv_cache._benchmark_cli import main as _main

    return _main(argv)


def parse_benchmark_arm_specs(
    raw_specs: Sequence[Mapping[str, Any]],
    *,
    method_registry: MethodRegistry | None = None,
) -> tuple[
    tuple[BenchmarkArm, ...],
    Mapping[str, str],
    Mapping[str, str],
    Mapping[str, Mapping[str, Any]],
]:
    from document_kv_cache._benchmark_cli import (
        parse_benchmark_arm_specs as _parse_benchmark_arm_specs,
    )

    return _parse_benchmark_arm_specs(
        raw_specs,
        method_registry=method_registry,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
