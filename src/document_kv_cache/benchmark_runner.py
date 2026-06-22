from __future__ import annotations

import argparse
from itertools import islice
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    CACHE_REUSE_ARM,
    DEFAULT_HARDWARE_TARGET,
    DEFAULT_V1_MODEL_ID,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    BenchmarkArm,
    BenchmarkComparison,
    BenchmarkExample,
    BenchmarkPromptParts,
    BenchmarkReportRow,
    BenchmarkSuite,
    InferenceMeasurement,
    baseline_prefill_arm,
    build_prompt_parts,
    compare_to_baseline,
    document_kv_cache_arm,
    evaluate_v1_benchmark_evidence,
    summarize_measurements,
    validate_v1_dataset,
    validate_v1_hardware_target,
)
from document_kv_cache.models import DocumentChunkType
from document_kv_cache.storage import local_path
from document_kv_cache.workflow import SourceChunk, SourceDocument


DEFAULT_OPENAI_COMPLETIONS_ENDPOINT = "/v1/completions"
BENCHMARK_RUN_RECORD_TYPE = "document_kv.benchmark_run.v1"

__all__ = [
    "BENCHMARK_RUN_RECORD_TYPE",
    "DEFAULT_OPENAI_COMPLETIONS_ENDPOINT",
    "BenchmarkGeneration",
    "BenchmarkEngineRequest",
    "BenchmarkEngine",
    "BenchmarkRunResult",
    "OpenAICompatibleBenchmarkConfig",
    "OpenAICompatibleEngineFactory",
    "default_benchmark_arms",
    "run_benchmark_suite",
    "load_v1_jsonl_suite",
    "load_benchmark_jsonl",
    "benchmark_run_result_to_record",
    "write_benchmark_run_result_json",
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
        _validate_generation_non_negative_int(self.completion_tokens, "completion_tokens")
        _validate_generation_non_negative_finite_number(self.ttft_seconds, "ttft_seconds")
        _validate_generation_non_negative_finite_number(
            self.time_to_completion_seconds,
            "time_to_completion_seconds",
        )
        if self.time_to_completion_seconds < self.ttft_seconds:
            raise ValueError("time_to_completion_seconds must be greater than or equal to ttft_seconds")
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

    def __post_init__(self) -> None:
        if self.request_id is not None:
            _validate_non_empty_string(self.request_id, "request_id")
        object.__setattr__(
            self,
            "kv_transfer_params",
            _json_object_mapping(self.kv_transfer_params, "kv_transfer_params"),
        )

    @property
    def logical_prompt_text(self) -> str:
        return self.prompt_parts.prefill_prompt

    @property
    def prompt_text(self) -> str:
        return self.runtime_prompt_text

    @property
    def runtime_prompt_text(self) -> str:
        if self.arm.uses_cache:
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
class BenchmarkRunResult:
    suite: BenchmarkSuite
    measurements: tuple[InferenceMeasurement, ...]
    report_rows: tuple[BenchmarkReportRow, ...]
    comparisons: tuple[BenchmarkComparison, ...]
    baseline_arm_id: str = BASELINE_PREFILL_ARM
    cache_arm_id: str = CACHE_REUSE_ARM


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
    shuffle: bool = False
    seed: int | None = None
    api_key: str | None = None
    max_tokens: int = 128
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    stream: bool = True
    cache_runtime_prompt: bool = False
    prompt_token_accounting: Literal["logical", "server_usage"] = "logical"
    baseline_extra_body: Mapping[str, Any] = field(default_factory=dict)
    cache_extra_body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.suite_id, "suite_id")
        dataset_paths = _validated_dataset_paths(self.dataset_paths)
        if not dataset_paths:
            raise ValueError("dataset_paths must be non-empty")
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
        validate_v1_hardware_target(self.hardware_target)
        if self.limit_per_dataset is not None and (
            type(self.limit_per_dataset) is not int or self.limit_per_dataset <= 0
        ):
            raise ValueError("limit_per_dataset must be positive when provided")
        _validate_positive_int(self.repeats, "repeats")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer when provided")
        if type(self.shuffle) is not bool:
            raise ValueError("shuffle must be a boolean")
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
            raise ValueError("prompt_token_accounting must be 'logical' or 'server_usage'")
        if self.cache_runtime_prompt and self.cache_base_url is None:
            raise ValueError("cache_runtime_prompt requires cache_base_url; pass the cache proxy URL explicitly")
        object.__setattr__(self, "dataset_paths", dataset_paths)
        object.__setattr__(
            self,
            "baseline_extra_body",
            _json_object_mapping(self.baseline_extra_body, "baseline_extra_body"),
        )
        object.__setattr__(
            self,
            "cache_extra_body",
            _json_object_mapping(self.cache_extra_body, "cache_extra_body"),
        )


OpenAICompatibleEngineFactory = Callable[[BenchmarkArm, OpenAICompatibleBenchmarkConfig], BenchmarkEngine]


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
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return [_json_compatible_value(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{field_name} must be JSON-compatible")

def default_benchmark_arms() -> tuple[BenchmarkArm, ...]:
    return (baseline_prefill_arm(), document_kv_cache_arm())


def run_benchmark_suite(
    suite: BenchmarkSuite,
    engines: Mapping[str, BenchmarkEngine],
    *,
    arms: Sequence[BenchmarkArm] = default_benchmark_arms(),
    repeats: int = 1,
    shuffle: bool = False,
    seed: int | None = None,
) -> BenchmarkRunResult:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    _validate_engine_mapping(arms, engines)
    measurements: list[InferenceMeasurement] = []
    for example in suite.examples:
        prompt_parts = build_prompt_parts(example)
        arm_sequence = list(arms) * repeats
        if shuffle:
            random.Random(_example_seed(seed, example.dataset, example.example_id)).shuffle(arm_sequence)
        for arm in arm_sequence:
            request = BenchmarkEngineRequest(
                suite_id=suite.suite_id,
                model_id=suite.model_id,
                hardware_target=suite.hardware_target,
                example=example,
                arm=arm,
                prompt_parts=prompt_parts,
                request_id=_request_id_for_arm(example, arm),
                kv_transfer_params=_kv_transfer_params_for_arm(example, arm),
            )
            measurements.append(_run_engine(request, engines[arm.arm_id]))
    report_rows = summarize_measurements(measurements)
    baseline_arm_id = _arm_id_for_prefill(arms)
    cache_arm_id = _arm_id_for_cache(arms)
    return BenchmarkRunResult(
        suite=suite,
        measurements=tuple(measurements),
        report_rows=report_rows,
        comparisons=compare_to_baseline(
            report_rows,
            baseline_arm_id=baseline_arm_id,
            cache_arm_id=cache_arm_id,
        ),
        baseline_arm_id=baseline_arm_id,
        cache_arm_id=cache_arm_id,
    )


def load_v1_jsonl_suite(
    *,
    suite_id: str,
    paths: Mapping[str, str | Path],
    model_id: str | None = None,
    hardware_target: str | None = None,
    limit_per_dataset: int | None = None,
) -> BenchmarkSuite:
    examples: list[BenchmarkExample] = []
    for dataset, path in paths.items():
        validate_v1_dataset(dataset)
        dataset_examples = load_benchmark_jsonl(
            path,
            dataset=dataset,
            limit=limit_per_dataset,
            require_dataset=True,
        )
        if not dataset_examples:
            raise ValueError(f"Dataset {dataset!r} must include at least one example")
        examples.extend(dataset_examples)
    kwargs: dict[str, Any] = {
        "suite_id": suite_id,
        "examples": tuple(examples),
        "datasets": tuple(paths),
    }
    if model_id is not None:
        kwargs["model_id"] = model_id
    if hardware_target is not None:
        kwargs["hardware_target"] = hardware_target
    suite = BenchmarkSuite(**kwargs)
    if not suite.examples:
        raise ValueError("Benchmark suite must include at least one example")
    return suite


def load_benchmark_jsonl(
    path: str | Path,
    *,
    dataset: str | None = None,
    limit: int | None = None,
    require_dataset: bool = False,
) -> tuple[BenchmarkExample, ...]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    examples: list[BenchmarkExample] = []
    records = _iter_jsonl(path)
    if limit is not None:
        records = islice(records, limit)
    for record_index, (line_number, record) in enumerate(records, start=1):
        try:
            example = _example_from_record(
                record,
                default_dataset=dataset,
                record_index=record_index,
                require_dataset=require_dataset,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Benchmark JSONL line {line_number}: {exc}") from exc
        examples.append(example)
    return tuple(examples)


def run_openai_compatible_v1_benchmark(
    config: OpenAICompatibleBenchmarkConfig,
    *,
    engine_factory: OpenAICompatibleEngineFactory | None = None,
) -> BenchmarkRunResult:
    suite = load_v1_jsonl_suite(
        suite_id=config.suite_id,
        paths=config.dataset_paths,
        model_id=config.model_id,
        hardware_target=config.hardware_target,
        limit_per_dataset=config.limit_per_dataset,
    )
    factory = engine_factory or _openai_compatible_engine
    arms = default_benchmark_arms()
    engines = {arm.arm_id: factory(arm, config) for arm in arms}
    return run_benchmark_suite(
        suite,
        engines,
        arms=arms,
        repeats=config.repeats,
        shuffle=config.shuffle,
        seed=config.seed,
    )


def benchmark_run_result_to_record(result: BenchmarkRunResult) -> dict[str, Any]:
    return {
        "record_type": BENCHMARK_RUN_RECORD_TYPE,
        "suite": {
            "suite_id": result.suite.suite_id,
            "model_id": result.suite.model_id,
            "hardware_target": result.suite.hardware_target,
            "datasets": list(result.suite.datasets),
            "examples": len(result.suite.examples),
        },
        "measurements": [_measurement_to_record(measurement) for measurement in result.measurements],
        "report_rows": [_report_row_to_record(row) for row in result.report_rows],
        "comparisons": [_comparison_to_record(comparison) for comparison in result.comparisons],
        "v1_evidence": _v1_evidence_to_record(
            evaluate_v1_benchmark_evidence(
                result.report_rows,
                result.comparisons,
                baseline_arm_id=result.baseline_arm_id,
                cache_arm_id=result.cache_arm_id,
            )
        ),
    }


def write_benchmark_run_result_json(result: BenchmarkRunResult, path: str | Path) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark_run_result_to_record(result), indent=2, sort_keys=True) + "\n")


def _run_engine(request: BenchmarkEngineRequest, engine: BenchmarkEngine) -> InferenceMeasurement:
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
            error=_exception_message(exc),
            metadata={"error_type": type(exc).__name__},
        )
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
        metadata=dict(generation.metadata),
    )


def _exception_message(exc: Exception) -> str:
    return str(exc) or type(exc).__name__


def _validate_generation_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")


def _validate_generation_non_negative_int(value: Any, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_generation_non_negative_finite_number(value: Any, field_name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number")


def _generation_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    normalized = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError(f"metadata.{key} must be a string")
        normalized[key] = value
    return normalized


def _openai_compatible_engine(arm: BenchmarkArm, config: OpenAICompatibleBenchmarkConfig) -> BenchmarkEngine:
    from document_kv_cache.openai_compatible import (  # Local import avoids an import cycle.
        OpenAICompatibleCompletionEngine,
        OpenAICompatibleEngineConfig,
    )

    base_url = config.cache_base_url if arm.uses_cache and config.cache_base_url is not None else config.base_url
    endpoint = config.cache_endpoint if arm.uses_cache and config.cache_endpoint is not None else config.endpoint
    prompt_text_mode = "runtime" if arm.uses_cache and config.cache_runtime_prompt else "logical"
    extra_body = config.cache_extra_body if arm.uses_cache else config.baseline_extra_body
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
        )
    )


def _measurement_to_record(measurement: InferenceMeasurement) -> dict[str, Any]:
    return {
        "example_id": measurement.example_id,
        "dataset": measurement.dataset,
        "arm_id": measurement.arm_id,
        "prompt_tokens": measurement.prompt_tokens,
        "completion_tokens": measurement.completion_tokens,
        "ttft_seconds": measurement.ttft_seconds,
        "time_to_completion_seconds": measurement.time_to_completion_seconds,
        "output_text": measurement.output_text,
        "expected_answer": measurement.expected_answer,
        "exact_match": measurement.exact_match,
        "answer_found": measurement.answer_found,
        "error": measurement.error,
        "metadata": dict(measurement.metadata),
    }


def _report_row_to_record(row: BenchmarkReportRow) -> dict[str, Any]:
    return {
        "dataset": row.dataset,
        "arm_id": row.arm_id,
        "requests": row.requests,
        "errors": row.errors,
        "prompt_tokens_mean": row.prompt_tokens_mean,
        "completion_tokens_mean": row.completion_tokens_mean,
        "ttft": _latency_to_record(row.ttft),
        "time_to_completion": _latency_to_record(row.time_to_completion),
        "exact_match_rate": row.exact_match_rate,
        "answer_found_rate": row.answer_found_rate,
        "output_tokens_per_second": row.output_tokens_per_second,
    }


def _comparison_to_record(comparison: BenchmarkComparison) -> dict[str, Any]:
    return {
        "dataset": comparison.dataset,
        "baseline_arm_id": comparison.baseline_arm_id,
        "cache_arm_id": comparison.cache_arm_id,
        "ttft_speedup": comparison.ttft_speedup,
        "time_to_completion_speedup": comparison.time_to_completion_speedup,
        "exact_match_delta": comparison.exact_match_delta,
        "answer_found_delta": comparison.answer_found_delta,
    }


def _v1_evidence_to_record(evidence: Any) -> dict[str, Any]:
    return {
        "ok": evidence.ok,
        "required_datasets": list(evidence.required_datasets),
        "baseline_arm_id": evidence.baseline_arm_id,
        "cache_arm_id": evidence.cache_arm_id,
        "duplicate_required_datasets": list(evidence.duplicate_required_datasets),
        "duplicate_report_rows": list(evidence.duplicate_report_rows),
        "duplicate_comparisons": list(evidence.duplicate_comparisons),
        "missing_report_rows": list(evidence.missing_report_rows),
        "missing_comparisons": list(evidence.missing_comparisons),
        "comparisons_without_metrics": list(evidence.comparisons_without_metrics),
        "rows_without_successful_requests": list(evidence.rows_without_successful_requests),
        "rows_without_latency": list(evidence.rows_without_latency),
        "rows_without_quality": list(evidence.rows_without_quality),
        "unexpected_arms": list(evidence.unexpected_arms),
        "unexpected_datasets": list(evidence.unexpected_datasets),
        "issues": list(evidence.issues),
    }


def _latency_to_record(summary: Any) -> dict[str, Any]:
    return {
        "count": summary.count,
        "mean": summary.mean,
        "p50": summary.p50,
        "p95": summary.p95,
    }


def _normalize_openai_base_url(base_url: str, *, endpoint: str) -> str:
    stripped = base_url.rstrip("/")
    if endpoint == DEFAULT_OPENAI_COMPLETIONS_ENDPOINT and stripped.endswith("/v1"):
        return stripped[:-3]
    return stripped


def _validate_engine_mapping(arms: Sequence[BenchmarkArm], engines: Mapping[str, BenchmarkEngine]) -> None:
    missing = [arm.arm_id for arm in arms if arm.arm_id not in engines]
    if missing:
        raise ValueError(f"Missing benchmark engines for arms: {missing}")
    arm_ids = [arm.arm_id for arm in arms]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError(f"Duplicate benchmark arm ids: {arm_ids}")


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


def _kv_transfer_params_for_arm(example: BenchmarkExample, arm: BenchmarkArm) -> Mapping[str, Any]:
    if not arm.uses_cache:
        return {}
    return example.kv_transfer_params


def _request_id_for_arm(example: BenchmarkExample, arm: BenchmarkArm) -> str | None:
    if not arm.uses_cache:
        return None
    request_id = example.kv_transfer_params.get(DOCUMENT_KV_REQUEST_ID_PARAM)
    if request_id is None:
        return None
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(f"kv_transfer_params.{DOCUMENT_KV_REQUEST_ID_PARAM} must be a non-empty string")
    return request_id


def _example_seed(seed: int | None, dataset: str, example_id: str) -> int:
    value = 0 if seed is None else seed
    for value_part in (dataset, "\0", example_id):
        for character in value_part:
            value = (value * 33 + ord(character)) & 0xFFFFFFFF
    return value


def _iter_jsonl(path: str | Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with local_path(str(path)).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Benchmark JSONL line {line_number} is not valid JSON: {exc.msg}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"Benchmark JSONL line {line_number} must be an object")
            yield line_number, record


def _example_from_record(
    record: Mapping[str, Any],
    *,
    default_dataset: str | None,
    record_index: int,
    require_dataset: bool,
) -> BenchmarkExample:
    dataset = _string_field(record, "dataset", default=default_dataset)
    validate_v1_dataset(dataset)
    if require_dataset and default_dataset is not None and dataset != default_dataset:
        raise ValueError(
            f"dataset {dataset!r} does not match expected dataset {default_dataset!r}"
        )
    return BenchmarkExample(
        example_id=_string_field(record, "example_id", fallback_fields=("id",), default=f"{dataset}-{record_index}"),
        dataset=dataset,
        documents=tuple(_documents_from_record(record)),
        query=_string_field(record, "query", fallback_fields=("question",)),
        expected_answer=_optional_string_field(record, "expected_answer", fallback_fields=("answer", "target")),
        metadata=_string_mapping(record.get("metadata", {}), field_name="metadata"),
        kv_transfer_params=_json_object_mapping(record.get("kv_transfer_params", {}), "kv_transfer_params"),
    )


def _validate_benchmark_jsonl_record(
    record: Mapping[str, Any],
    *,
    dataset: str | None = None,
    record_index: int = 1,
    require_dataset: bool = False,
) -> None:
    _example_from_record(
        record,
        default_dataset=dataset,
        record_index=record_index,
        require_dataset=require_dataset,
    )


def _documents_from_record(record: Mapping[str, Any]) -> tuple[SourceDocument, ...]:
    raw_documents = record.get("documents")
    if raw_documents is None:
        raw_documents = record.get("contexts")
    if raw_documents is None:
        raw_documents = record.get("paragraphs")
    if raw_documents is None:
        raw_documents = record.get("context")
    if raw_documents is None:
        raise ValueError("Benchmark JSONL record must include documents, contexts, paragraphs, or context")
    normalized_documents = _normalize_raw_documents(raw_documents)
    documents = tuple(_document_from_record(document, index=index) for index, document in enumerate(normalized_documents))
    if not documents:
        raise ValueError("documents must contain at least one document")
    return documents


def _normalize_raw_documents(raw_documents: Any) -> tuple[Any, ...]:
    if isinstance(raw_documents, Mapping):
        return tuple({"document_id": key, "text": value} for key, value in raw_documents.items())
    if isinstance(raw_documents, str):
        return (raw_documents,)
    if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, bytes):
        raise ValueError("documents must be a sequence of document objects")
    if raw_documents and _looks_like_hotpot_context_pair(raw_documents[0]):
        return tuple(_hotpot_pair_to_document(item, index=index) for index, item in enumerate(raw_documents))
    return tuple(raw_documents)


def _looks_like_hotpot_context_pair(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and isinstance(value[1], Sequence)
        and not isinstance(value[1], (str, bytes))
    )


def _hotpot_pair_to_document(value: Any, *, index: int) -> Mapping[str, Any]:
    if not _looks_like_hotpot_context_pair(value):
        raise ValueError(f"HotpotQA context entry {index} must be [title, sentences]")
    title = _coerce_string(value[0], field_name=f"context {index} title")
    return {
        "document_id": title or f"doc-{index}",
        "title": title,
        "sentences": value[1],
    }


def _document_from_record(record: Any, *, index: int) -> SourceDocument:
    if isinstance(record, str):
        return SourceDocument.from_texts(document_id=f"doc-{index}", chunks={"text": record})
    if not isinstance(record, Mapping):
        raise ValueError("documents entries must be objects or strings")
    document_id = _string_field(record, "document_id", fallback_fields=("id", "title", "idx"), default=f"doc-{index}")
    metadata = _string_mapping(record.get("metadata", {}), field_name="document metadata")
    title = _optional_string_field(record, "title", fallback_fields=("name",))
    if title is not None and "title" not in metadata:
        metadata = {**metadata, "title": title}
    return SourceDocument(
        document_id=document_id,
        chunks=_chunks_from_record(record),
        metadata=metadata,
    )


def _chunks_from_record(record: Mapping[str, Any]) -> tuple[SourceChunk, ...]:
    chunks: list[SourceChunk] = []
    static_text = _optional_string_field(record, "static_text", fallback_fields=("summary",))
    if static_text is not None:
        chunks.append(
            SourceChunk(
                chunk_id="static",
                text=static_text,
                chunk_type=DocumentChunkType.DOCUMENT_STATIC,
            )
        )
    raw_chunks = record.get("chunks")
    if raw_chunks is None:
        raw_chunks = record.get("sentences")
    if raw_chunks is None:
        text = _optional_string_field(record, "text", fallback_fields=("body", "context", "paragraph_text"))
        if text is not None:
            raw_chunks = {"text": text}
    if raw_chunks is not None:
        chunks.extend(_iter_chunks(raw_chunks))
    if not chunks:
        raise ValueError("document record must include static_text, chunks, or text")
    return tuple(chunks)


def _iter_chunks(raw_chunks: Any) -> Iterable[SourceChunk]:
    if isinstance(raw_chunks, Mapping):
        for chunk_id, text in raw_chunks.items():
            yield SourceChunk(chunk_id=str(chunk_id), text=_coerce_string(text, field_name=f"chunk {chunk_id}"))
        return
    if isinstance(raw_chunks, Sequence) and not isinstance(raw_chunks, (str, bytes)):
        for index, chunk in enumerate(raw_chunks):
            yield _chunk_from_record(chunk, index=index)
        return
    raise ValueError("chunks must be a mapping or sequence")


def _chunk_from_record(record: Any, *, index: int) -> SourceChunk:
    if isinstance(record, str):
        return SourceChunk(chunk_id=f"chunk-{index}", text=record)
    if not isinstance(record, Mapping):
        raise ValueError("chunk entries must be objects or strings")
    chunk_type_value = _string_field(record, "chunk_type", default=DocumentChunkType.DOCUMENT_CHUNK.value)
    try:
        chunk_type = DocumentChunkType(chunk_type_value)
    except ValueError as exc:
        raise ValueError(f"Unsupported chunk_type {chunk_type_value!r}") from exc
    return SourceChunk(
        chunk_id=_string_field(record, "chunk_id", fallback_fields=("id", "idx"), default=f"chunk-{index}"),
        text=_string_field(record, "text", fallback_fields=("body", "context", "paragraph_text")),
        chunk_type=chunk_type,
        metadata=_string_mapping(record.get("metadata", {}), field_name="chunk metadata"),
    )


def _string_field(
    record: Mapping[str, Any],
    field_name: str,
    *,
    fallback_fields: Sequence[str] = (),
    default: str | None = None,
) -> str:
    value = _field_value(record, field_name, fallback_fields=fallback_fields, default=default)
    if value is None:
        expected = ", ".join((field_name, *fallback_fields))
        raise ValueError(f"Missing required field: {expected}")
    return _coerce_string(value, field_name=field_name)


def _optional_string_field(
    record: Mapping[str, Any],
    field_name: str,
    *,
    fallback_fields: Sequence[str] = (),
) -> str | None:
    value = _field_value(record, field_name, fallback_fields=fallback_fields)
    if value is None:
        return None
    return _coerce_string(value, field_name=field_name)


def _field_value(
    record: Mapping[str, Any],
    field_name: str,
    *,
    fallback_fields: Sequence[str],
    default: str | None = None,
) -> Any:
    for candidate in (field_name, *fallback_fields):
        if candidate in record and record[candidate] is not None:
            return record[candidate]
    return default


def _coerce_string(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    raise ValueError(f"{field_name} must be string-like")


def _string_mapping(value: Any, *, field_name: str) -> Mapping[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return {str(key): _coerce_string(item, field_name=f"{field_name}.{key}") for key, item in value.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the V1 document KV-cache benchmark against OpenAI-compatible vLLM/SGLang servers."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="Dataset JSONL path. Repeat for biography, hotpotqa, musique, and niah.",
    )
    parser.add_argument("--suite-id", default="v1-openai-compatible")
    parser.add_argument("--base-url", required=True, help="Baseline server base URL, for example http://localhost:8000")
    parser.add_argument("--cache-base-url", help="Optional KV-aware cache server/proxy URL. Defaults to --base-url.")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_OPENAI_COMPLETIONS_ENDPOINT,
        help="Completions endpoint appended to --base-url.",
    )
    parser.add_argument("--cache-endpoint", help="Optional endpoint appended to --cache-base-url for the cache arm.")
    parser.add_argument("--model-id", default=DEFAULT_V1_MODEL_ID)
    parser.add_argument("--hardware-target", default=DEFAULT_HARDWARE_TARGET)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument(
        "--cache-runtime-prompt",
        action="store_true",
        help="Send only the runtime suffix for the cache arm; requires a KV-aware proxy that binds cached prefixes.",
    )
    parser.add_argument(
        "--server-usage",
        action="store_true",
        help="Prefer server usage.prompt_tokens; metadata still reports whether usage was present.",
    )
    parser.add_argument("--baseline-extra-body-json", default="{}", help="JSON object merged into baseline requests.")
    parser.add_argument("--cache-extra-body-json", default="{}", help="JSON object merged into cache-arm requests.")
    parser.add_argument("--output-json", help="Write the full benchmark result JSON to this path instead of stdout.")
    args = parser.parse_args(argv)

    try:
        config = OpenAICompatibleBenchmarkConfig(
            suite_id=args.suite_id,
            dataset_paths=_dataset_paths_from_cli(args.dataset),
            base_url=args.base_url,
            cache_base_url=args.cache_base_url,
            endpoint=args.endpoint,
            cache_endpoint=args.cache_endpoint,
            model_id=args.model_id,
            hardware_target=args.hardware_target,
            limit_per_dataset=args.limit_per_dataset,
            repeats=args.repeats,
            shuffle=args.shuffle,
            seed=args.seed,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            stream=not args.no_stream,
            cache_runtime_prompt=args.cache_runtime_prompt,
            prompt_token_accounting="server_usage" if args.server_usage else "logical",
            baseline_extra_body=_json_object_option(args.baseline_extra_body_json, "--baseline-extra-body-json"),
            cache_extra_body=_json_object_option(args.cache_extra_body_json, "--cache-extra-body-json"),
        )
        result = run_openai_compatible_v1_benchmark(config)
        if args.output_json:
            write_benchmark_run_result_json(result, args.output_json)
        else:
            print(json.dumps(benchmark_run_result_to_record(result), indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, indent=2, sort_keys=True))
        return 1

    return 0 if not any(measurement.error for measurement in result.measurements) else 2


def _dataset_paths_from_cli(values: Sequence[str]) -> Mapping[str, Path]:
    dataset_paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset must use DATASET=PATH")
        dataset, raw_path = value.split("=", 1)
        validate_v1_dataset(dataset)
        if not raw_path:
            raise ValueError(f"--dataset {dataset}=PATH must include a path")
        if dataset in dataset_paths:
            raise ValueError(f"Duplicate dataset path for {dataset!r}")
        dataset_paths[dataset] = local_path(raw_path)
    return dataset_paths


def _json_object_option(raw_json: str, option_name: str) -> Mapping[str, Any]:
    value = json.loads(raw_json)
    if not isinstance(value, Mapping):
        raise ValueError(f"{option_name} must decode to a JSON object")
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
