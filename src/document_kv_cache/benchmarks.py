from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from html import escape

from document_kv_cache.workflow import SourceDocument


SUPPORTED_V1_DATASETS = ("biography", "hotpotqa", "musique", "niah")
DEFAULT_V1_MODEL_ID = "qwen3:4b-instruct"
SUPPORTED_V1_HARDWARE_TARGETS = ("aws-g6-l4", "aws-g5")
DEFAULT_HARDWARE_TARGET = "aws-g6-l4"
BASELINE_PREFILL_ARM = "baseline_prefill"
CACHE_REUSE_ARM = "document_kv_cache"
FINAL_ANSWER_CUE = "Answer:"

__all__ = [
    "SUPPORTED_V1_DATASETS",
    "SUPPORTED_V1_HARDWARE_TARGETS",
    "DEFAULT_V1_MODEL_ID",
    "DEFAULT_HARDWARE_TARGET",
    "BASELINE_PREFILL_ARM",
    "CACHE_REUSE_ARM",
    "BenchmarkDatasetSpec",
    "BenchmarkPromptParts",
    "FINAL_ANSWER_CUE",
    "BenchmarkExample",
    "BenchmarkSuite",
    "BenchmarkArm",
    "InferenceMeasurement",
    "LatencySummary",
    "BenchmarkReportRow",
    "BenchmarkComparison",
    "V1BenchmarkEvidence",
    "baseline_prefill_arm",
    "document_kv_cache_arm",
    "v1_dataset_specs",
    "dataset_spec",
    "build_prompt_parts",
    "build_prefill_prompt",
    "build_cache_prefix_text",
    "build_cache_suffix_text",
    "format_document_context",
    "summarize_measurements",
    "compare_to_baseline",
    "evaluate_v1_benchmark_evidence",
    "normalize_answer",
    "exact_match",
    "answer_found",
    "validate_v1_hardware_target",
    "validate_v1_dataset",
]


@dataclass(frozen=True, slots=True)
class BenchmarkDatasetSpec:
    dataset: str
    display_name: str
    task_instruction: str
    answer_instruction: str

    def __post_init__(self) -> None:
        validate_v1_dataset(self.dataset)
        if not self.display_name:
            raise ValueError("display_name must be non-empty")
        if not self.task_instruction:
            raise ValueError("task_instruction must be non-empty")
        if not self.answer_instruction:
            raise ValueError("answer_instruction must be non-empty")


@dataclass(frozen=True, slots=True)
class BenchmarkPromptParts:
    system_prompt: str
    document_context: str
    user_prompt: str

    @property
    def prefill_prompt(self) -> str:
        return _join_sections(self.system_prompt, self.document_context, self.user_prompt)

    @property
    def cache_prefix_text(self) -> str:
        return _join_sections(self.system_prompt, self.document_context)

    @property
    def cache_suffix_text(self) -> str:
        if not self.cache_prefix_text:
            return self.user_prompt
        return f"\n\n{self.user_prompt}"


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    example_id: str
    dataset: str
    documents: tuple[SourceDocument, ...]
    query: str
    expected_answer: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.example_id, "example_id")
        validate_v1_dataset(self.dataset)
        _validate_non_empty_str(self.query, "query")
        if self.expected_answer is not None:
            _validate_non_empty_str(self.expected_answer, "expected_answer")
        documents = _tuple_from_sequence(self.documents, "documents")
        if not documents:
            raise ValueError("documents must include at least one SourceDocument")
        for index, document in enumerate(documents):
            if not isinstance(document, SourceDocument):
                raise TypeError(f"documents[{index}] must be a SourceDocument")
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "metadata", _dict_from_str_mapping(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    suite_id: str
    examples: tuple[BenchmarkExample, ...]
    model_id: str = DEFAULT_V1_MODEL_ID
    hardware_target: str = DEFAULT_HARDWARE_TARGET
    datasets: tuple[str, ...] = SUPPORTED_V1_DATASETS

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.suite_id, "suite_id")
        _validate_non_empty_str(self.model_id, "model_id")
        _validate_non_empty_str(self.hardware_target, "hardware_target")
        validate_v1_hardware_target(self.hardware_target)
        examples = _tuple_from_sequence(self.examples, "examples")
        if not examples:
            raise ValueError("examples must include at least one BenchmarkExample")
        for index, example in enumerate(examples):
            if not isinstance(example, BenchmarkExample):
                raise TypeError(f"examples[{index}] must be a BenchmarkExample")
        duplicate_examples = _duplicate_labels(_example_key(example) for example in examples)
        if duplicate_examples:
            duplicate_ids = ", ".join(duplicate_examples)
            raise ValueError(f"examples contain duplicate dataset/example ids: {duplicate_ids}")
        datasets = _tuple_from_sequence(self.datasets, "datasets")
        if not datasets:
            raise ValueError("datasets must include at least one V1 dataset")
        for dataset in datasets:
            validate_v1_dataset(dataset)
        duplicate_datasets = _duplicate_labels(datasets)
        if duplicate_datasets:
            raise ValueError(f"datasets contain duplicate V1 dataset ids: {', '.join(duplicate_datasets)}")
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "datasets", datasets)
        example_datasets = {example.dataset for example in examples}
        missing = example_datasets.difference(datasets)
        if missing:
            raise ValueError(f"Examples reference datasets outside this suite: {sorted(missing)}")


@dataclass(frozen=True, slots=True)
class BenchmarkArm:
    arm_id: str
    uses_cache: bool
    description: str


@dataclass(frozen=True, slots=True)
class InferenceMeasurement:
    example_id: str
    dataset: str
    arm_id: str
    prompt_tokens: int
    completion_tokens: int
    ttft_seconds: float
    time_to_completion_seconds: float
    output_text: str
    expected_answer: str | None = None
    error: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.example_id, "example_id")
        validate_v1_dataset(self.dataset)
        _validate_non_empty_str(self.arm_id, "arm_id")
        _validate_non_negative_int(self.prompt_tokens, "prompt_tokens")
        _validate_non_negative_int(self.completion_tokens, "completion_tokens")
        _validate_non_negative_finite_number(self.ttft_seconds, "ttft_seconds")
        _validate_non_negative_finite_number(self.time_to_completion_seconds, "time_to_completion_seconds")
        if self.time_to_completion_seconds < self.ttft_seconds:
            raise ValueError("time_to_completion_seconds must be greater than or equal to ttft_seconds")
        _validate_str(self.output_text, "output_text")
        if self.expected_answer is not None:
            _validate_non_empty_str(self.expected_answer, "expected_answer")
        if self.error is not None:
            _validate_non_empty_str(self.error, "error")
        object.__setattr__(self, "metadata", _dict_from_str_mapping(self.metadata, "metadata"))

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def exact_match(self) -> bool | None:
        if self.expected_answer is None or not self.ok:
            return None
        return exact_match(self.output_text, self.expected_answer)

    @property
    def answer_found(self) -> bool | None:
        if self.expected_answer is None or not self.ok:
            return None
        return answer_found(self.output_text, self.expected_answer)


@dataclass(frozen=True, slots=True)
class LatencySummary:
    count: int
    mean: float | None
    p50: float | None
    p95: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkReportRow:
    dataset: str
    arm_id: str
    requests: int
    errors: int
    prompt_tokens_mean: float | None
    completion_tokens_mean: float | None
    ttft: LatencySummary
    time_to_completion: LatencySummary
    exact_match_rate: float | None
    answer_found_rate: float | None
    output_tokens_per_second: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    dataset: str
    baseline_arm_id: str
    cache_arm_id: str
    ttft_speedup: float | None
    time_to_completion_speedup: float | None
    exact_match_delta: float | None
    answer_found_delta: float | None


@dataclass(frozen=True, slots=True)
class V1BenchmarkEvidence:
    required_datasets: tuple[str, ...]
    baseline_arm_id: str
    cache_arm_id: str
    duplicate_required_datasets: tuple[str, ...]
    duplicate_report_rows: tuple[str, ...]
    duplicate_comparisons: tuple[str, ...]
    missing_report_rows: tuple[str, ...]
    missing_comparisons: tuple[str, ...]
    comparisons_without_metrics: tuple[str, ...]
    rows_without_successful_requests: tuple[str, ...]
    rows_without_latency: tuple[str, ...]
    rows_without_quality: tuple[str, ...]
    unexpected_datasets: tuple[str, ...] = ()
    unexpected_arms: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not (
            self.missing_report_rows
            or self.missing_comparisons
            or self.duplicate_required_datasets
            or self.duplicate_report_rows
            or self.duplicate_comparisons
            or self.comparisons_without_metrics
            or self.rows_without_successful_requests
            or self.rows_without_latency
            or self.rows_without_quality
            or self.unexpected_arms
            or self.unexpected_datasets
        )

    @property
    def issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.duplicate_required_datasets:
            issues.append(f"duplicate required datasets: {', '.join(self.duplicate_required_datasets)}")
        if self.duplicate_report_rows:
            issues.append(f"duplicate report rows: {', '.join(self.duplicate_report_rows)}")
        if self.duplicate_comparisons:
            issues.append(f"duplicate comparisons: {', '.join(self.duplicate_comparisons)}")
        if self.missing_report_rows:
            issues.append(f"missing report rows: {', '.join(self.missing_report_rows)}")
        if self.missing_comparisons:
            issues.append(f"missing comparisons: {', '.join(self.missing_comparisons)}")
        if self.comparisons_without_metrics:
            issues.append(
                "comparisons without speedup or quality deltas: "
                f"{', '.join(self.comparisons_without_metrics)}"
            )
        if self.rows_without_successful_requests:
            issues.append(f"rows without successful requests: {', '.join(self.rows_without_successful_requests)}")
        if self.rows_without_latency:
            issues.append(f"rows without latency evidence: {', '.join(self.rows_without_latency)}")
        if self.rows_without_quality:
            issues.append(f"rows without quality evidence: {', '.join(self.rows_without_quality)}")
        if self.unexpected_arms:
            issues.append(f"unexpected arms: {', '.join(self.unexpected_arms)}")
        if self.unexpected_datasets:
            issues.append(f"unexpected datasets: {', '.join(self.unexpected_datasets)}")
        return tuple(issues)


def baseline_prefill_arm() -> BenchmarkArm:
    return BenchmarkArm(
        arm_id=BASELINE_PREFILL_ARM,
        uses_cache=False,
        description="Standard inference prefill that recomputes all document tokens.",
    )


def document_kv_cache_arm() -> BenchmarkArm:
    return BenchmarkArm(
        arm_id=CACHE_REUSE_ARM,
        uses_cache=True,
        description="Inference path that reuses precomputed document KV cache.",
    )


def v1_dataset_specs() -> tuple[BenchmarkDatasetSpec, ...]:
    return tuple(_V1_DATASET_SPECS[dataset] for dataset in SUPPORTED_V1_DATASETS)


def dataset_spec(dataset: str) -> BenchmarkDatasetSpec:
    validate_v1_dataset(dataset)
    return _V1_DATASET_SPECS[dataset]


def build_prompt_parts(example: BenchmarkExample) -> BenchmarkPromptParts:
    spec = dataset_spec(example.dataset)
    return BenchmarkPromptParts(
        system_prompt=_system_prompt(spec),
        document_context=format_document_context(example.documents),
        user_prompt=_user_prompt(example, spec),
    )


def build_prefill_prompt(example: BenchmarkExample) -> str:
    return build_prompt_parts(example).prefill_prompt


def build_cache_prefix_text(example: BenchmarkExample) -> str:
    return build_prompt_parts(example).cache_prefix_text


def build_cache_suffix_text(example: BenchmarkExample) -> str:
    return build_prompt_parts(example).cache_suffix_text


def format_document_context(documents: Sequence[SourceDocument]) -> str:
    if not documents:
        raise ValueError("Benchmark examples must include at least one source document")
    return _join_sections("Documents:", *(_format_document(document) for document in documents))


def summarize_measurements(measurements: Iterable[InferenceMeasurement]) -> tuple[BenchmarkReportRow, ...]:
    grouped: dict[tuple[str, str], list[InferenceMeasurement]] = {}
    for measurement in measurements:
        grouped.setdefault((measurement.dataset, measurement.arm_id), []).append(measurement)
    rows = [_summarize_group(dataset, arm_id, group) for (dataset, arm_id), group in grouped.items()]
    return tuple(sorted(rows, key=lambda row: (row.dataset, row.arm_id)))


def compare_to_baseline(
    rows: Sequence[BenchmarkReportRow],
    *,
    baseline_arm_id: str = BASELINE_PREFILL_ARM,
    cache_arm_id: str = CACHE_REUSE_ARM,
) -> tuple[BenchmarkComparison, ...]:
    by_key = {(row.dataset, row.arm_id): row for row in rows}
    comparisons: list[BenchmarkComparison] = []
    datasets = sorted({row.dataset for row in rows})
    for dataset in datasets:
        baseline = by_key.get((dataset, baseline_arm_id))
        cache = by_key.get((dataset, cache_arm_id))
        if baseline is None or cache is None:
            continue
        comparisons.append(
            BenchmarkComparison(
                dataset=dataset,
                baseline_arm_id=baseline_arm_id,
                cache_arm_id=cache_arm_id,
                ttft_speedup=_speedup(baseline.ttft.p50, cache.ttft.p50),
                time_to_completion_speedup=_speedup(
                    baseline.time_to_completion.p50,
                    cache.time_to_completion.p50,
                ),
                exact_match_delta=_delta(cache.exact_match_rate, baseline.exact_match_rate),
                answer_found_delta=_delta(cache.answer_found_rate, baseline.answer_found_rate),
            )
        )
    return tuple(comparisons)


def evaluate_v1_benchmark_evidence(
    rows: Sequence[BenchmarkReportRow],
    comparisons: Sequence[BenchmarkComparison],
    *,
    required_datasets: Sequence[str] = SUPPORTED_V1_DATASETS,
    baseline_arm_id: str = BASELINE_PREFILL_ARM,
    cache_arm_id: str = CACHE_REUSE_ARM,
) -> V1BenchmarkEvidence:
    required = tuple(required_datasets)
    for dataset in required:
        validate_v1_dataset(dataset)
    duplicate_required_datasets = _duplicate_labels(required)
    unique_required = _dedupe_preserve_order(required)
    required_row_keys = tuple(
        (dataset, arm_id)
        for dataset in unique_required
        for arm_id in (baseline_arm_id, cache_arm_id)
    )
    rows_by_key, duplicate_report_rows = _report_rows_by_key(rows)
    comparisons_by_dataset, duplicate_comparisons = _comparisons_by_dataset(
        comparisons,
        baseline_arm_id=baseline_arm_id,
        cache_arm_id=cache_arm_id,
    )
    required_datasets_set = set(unique_required)
    expected_arms = {baseline_arm_id, cache_arm_id}
    observed_datasets = {row.dataset for row in rows}.union(comparison.dataset for comparison in comparisons)
    observed_arms = {row.arm_id for row in rows}.union(
        arm_id
        for comparison in comparisons
        for arm_id in (comparison.baseline_arm_id, comparison.cache_arm_id)
    )
    existing_required_rows = tuple(rows_by_key[key] for key in required_row_keys if key in rows_by_key)
    return V1BenchmarkEvidence(
        required_datasets=required,
        baseline_arm_id=baseline_arm_id,
        cache_arm_id=cache_arm_id,
        duplicate_required_datasets=duplicate_required_datasets,
        duplicate_report_rows=duplicate_report_rows,
        duplicate_comparisons=duplicate_comparisons,
        missing_report_rows=tuple(
            _row_key(dataset, arm_id)
            for dataset, arm_id in required_row_keys
            if (dataset, arm_id) not in rows_by_key
        ),
        missing_comparisons=tuple(dataset for dataset in unique_required if dataset not in comparisons_by_dataset),
        comparisons_without_metrics=tuple(
            dataset
            for dataset in unique_required
            if (comparison := comparisons_by_dataset.get(dataset)) is not None
            and _comparison_has_missing_metrics(comparison)
        ),
        rows_without_successful_requests=tuple(
            _row_key(row.dataset, row.arm_id)
            for row in existing_required_rows
            if row.ttft.count == 0
        ),
        rows_without_latency=tuple(
            _row_key(row.dataset, row.arm_id)
            for row in existing_required_rows
            if row.ttft.p50 is None or row.time_to_completion.p50 is None
        ),
        rows_without_quality=tuple(
            _row_key(row.dataset, row.arm_id)
            for row in existing_required_rows
            if row.exact_match_rate is None or row.answer_found_rate is None
        ),
        unexpected_arms=tuple(sorted(observed_arms.difference(expected_arms))),
        unexpected_datasets=tuple(sorted(observed_datasets.difference(required_datasets_set))),
    )


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def exact_match(output_text: str, expected_answer: str) -> bool:
    return normalize_answer(output_text) == normalize_answer(expected_answer)


def answer_found(output_text: str, expected_answer: str) -> bool:
    expected_tokens = normalize_answer(expected_answer).split()
    output_tokens = normalize_answer(output_text).split()
    if not expected_tokens:
        return False
    window = len(expected_tokens)
    for index in range(0, len(output_tokens) - window + 1):
        if output_tokens[index : index + window] == expected_tokens:
            return True
    return False


def validate_v1_dataset(dataset: str) -> None:
    if dataset not in SUPPORTED_V1_DATASETS:
        raise ValueError(f"Unsupported V1 dataset {dataset!r}; expected one of {SUPPORTED_V1_DATASETS}")


def validate_v1_hardware_target(hardware_target: str) -> None:
    if hardware_target not in SUPPORTED_V1_HARDWARE_TARGETS:
        raise ValueError(
            f"Unsupported V1 hardware target {hardware_target!r}; expected one of {SUPPORTED_V1_HARDWARE_TARGETS}"
        )


def _validate_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")


def _validate_str(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")


def _tuple_from_sequence(value: Sequence[object], field_name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return tuple(value)


def _dict_from_str_mapping(value: Mapping[str, str], field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if not isinstance(item, str):
            raise ValueError(f"{field_name}.{key} must be a string")
        normalized[key] = item
    return normalized


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_non_negative_finite_number(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number")


_V1_DATASET_SPECS: Mapping[str, BenchmarkDatasetSpec] = {
    "biography": BenchmarkDatasetSpec(
        dataset="biography",
        display_name="Biography",
        task_instruction="Answer biography questions using only the supplied document context.",
        answer_instruction="Return the shortest answer that fully resolves the question.",
    ),
    "hotpotqa": BenchmarkDatasetSpec(
        dataset="hotpotqa",
        display_name="HotpotQA",
        task_instruction="Answer multi-hop questions by combining the relevant facts in the supplied context.",
        answer_instruction="Return the final answer, not a chain-of-thought explanation.",
    ),
    "musique": BenchmarkDatasetSpec(
        dataset="musique",
        display_name="MusiQue",
        task_instruction="Answer compositional questions by using all necessary supporting documents in the context.",
        answer_instruction="Return the final answer, not a chain-of-thought explanation.",
    ),
    "niah": BenchmarkDatasetSpec(
        dataset="niah",
        display_name="Needle-in-a-Haystack",
        task_instruction="Find the hidden target statement in the supplied context.",
        answer_instruction="Return the exact needle or requested value from the context.",
    ),
}


def _system_prompt(spec: BenchmarkDatasetSpec) -> str:
    return _join_sections(
        f"Benchmark: {spec.display_name}",
        spec.task_instruction,
        "Use only the supplied document context. If the answer is absent, say you do not know.",
    )


def _user_prompt(example: BenchmarkExample, spec: BenchmarkDatasetSpec) -> str:
    return _join_sections(
        f"Question: {example.query}",
        spec.answer_instruction,
        FINAL_ANSWER_CUE,
    )


def _format_document(document: SourceDocument) -> str:
    title = document.metadata.get("title") or document.metadata.get("name") or document.document_id
    chunks = tuple(_format_chunk(chunk.chunk_id, chunk.chunk_type.value, chunk.text) for chunk in document.chunks)
    return _join_sections(
        f'[document id="{_attribute_text(document.document_id)}" title="{_attribute_text(title)}"]',
        *chunks,
        f'[/document id="{_attribute_text(document.document_id)}"]',
    )


def _format_chunk(chunk_id: str, chunk_type: str, text: str) -> str:
    return _join_sections(
        f'[chunk id="{_attribute_text(chunk_id)}" type="{_attribute_text(chunk_type)}"]',
        _quote_block_text(text),
        f'[/chunk id="{_attribute_text(chunk_id)}"]',
    )


def _summarize_group(dataset: str, arm_id: str, group: Sequence[InferenceMeasurement]) -> BenchmarkReportRow:
    if not group:
        raise ValueError("Cannot summarize an empty measurement group")
    ok = [measurement for measurement in group if measurement.ok]
    errors = len(group) - len(ok)
    prompt_tokens = [measurement.prompt_tokens for measurement in ok]
    completion_tokens = [measurement.completion_tokens for measurement in ok]
    ttft_values = [measurement.ttft_seconds for measurement in ok]
    ttc_values = [measurement.time_to_completion_seconds for measurement in ok]
    total_completion_tokens = sum(completion_tokens)
    total_ttc = sum(ttc_values)
    return BenchmarkReportRow(
        dataset=dataset,
        arm_id=arm_id,
        requests=len(group),
        errors=errors,
        prompt_tokens_mean=_mean(prompt_tokens),
        completion_tokens_mean=_mean(completion_tokens),
        ttft=_latency_summary(ttft_values),
        time_to_completion=_latency_summary(ttc_values),
        exact_match_rate=_rate(measurement.exact_match for measurement in ok),
        answer_found_rate=_rate(measurement.answer_found for measurement in ok),
        output_tokens_per_second=(total_completion_tokens / total_ttc) if total_ttc > 0 else None,
    )


def _latency_summary(values: Sequence[float]) -> LatencySummary:
    if not values:
        return LatencySummary(count=0, mean=None, p50=None, p95=None)
    sorted_values = sorted(values)
    return LatencySummary(
        count=len(sorted_values),
        mean=statistics.fmean(sorted_values),
        p50=_percentile(sorted_values, 0.50),
        p95=_percentile(sorted_values, 0.95),
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = percentile * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _mean(values: Sequence[int]) -> float | None:
    return statistics.fmean(values) if values else None


def _rate(values: Iterable[bool | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(1 for value in present if value) / len(present)


def _speedup(baseline_seconds: float | None, candidate_seconds: float | None) -> float | None:
    if baseline_seconds is None or candidate_seconds is None:
        return None
    if baseline_seconds <= 0 or candidate_seconds <= 0:
        return None
    return baseline_seconds / candidate_seconds


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _comparison_has_missing_metrics(comparison: BenchmarkComparison) -> bool:
    return (
        comparison.ttft_speedup is None
        or comparison.time_to_completion_speedup is None
        or comparison.exact_match_delta is None
        or comparison.answer_found_delta is None
    )


def _row_key(dataset: str, arm_id: str) -> str:
    return f"{dataset}:{arm_id}"


def _example_key(example: BenchmarkExample) -> str:
    return f"{example.dataset}:{example.example_id}"


def _comparison_key(comparison: BenchmarkComparison) -> str:
    return f"{comparison.dataset}:{comparison.baseline_arm_id}->{comparison.cache_arm_id}"


def _duplicate_labels(labels: Iterable[str]) -> tuple[str, ...]:
    seen = set()
    duplicate_seen = set()
    duplicates = []
    for label in labels:
        if label in seen and label not in duplicate_seen:
            duplicate_seen.add(label)
            duplicates.append(label)
        seen.add(label)
    return tuple(duplicates)


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def _report_rows_by_key(
    rows: Sequence[BenchmarkReportRow],
) -> tuple[dict[tuple[str, str], BenchmarkReportRow], tuple[str, ...]]:
    rows_by_key = {}
    duplicate_labels = []
    duplicate_seen = set()
    for row in rows:
        key = (row.dataset, row.arm_id)
        label = _row_key(row.dataset, row.arm_id)
        if key in rows_by_key:
            if label not in duplicate_seen:
                duplicate_seen.add(label)
                duplicate_labels.append(label)
            continue
        rows_by_key[key] = row
    return rows_by_key, tuple(duplicate_labels)


def _comparisons_by_dataset(
    comparisons: Sequence[BenchmarkComparison],
    *,
    baseline_arm_id: str,
    cache_arm_id: str,
) -> tuple[dict[str, BenchmarkComparison], tuple[str, ...]]:
    comparisons_by_dataset = {}
    duplicate_labels = []
    duplicate_seen = set()
    for comparison in comparisons:
        if comparison.baseline_arm_id != baseline_arm_id or comparison.cache_arm_id != cache_arm_id:
            continue
        label = _comparison_key(comparison)
        if comparison.dataset in comparisons_by_dataset:
            if label not in duplicate_seen:
                duplicate_seen.add(label)
                duplicate_labels.append(label)
            continue
        comparisons_by_dataset[comparison.dataset] = comparison
    return comparisons_by_dataset, tuple(duplicate_labels)


def _join_sections(*sections: str) -> str:
    return "\n\n".join(section for section in sections if section)


def _clean_inline_text(text: str) -> str:
    return " ".join(text.strip().split())


def _attribute_text(text: str) -> str:
    return escape(_clean_inline_text(text), quote=True)


def _quote_block_text(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(f"| {line}" for line in lines)
