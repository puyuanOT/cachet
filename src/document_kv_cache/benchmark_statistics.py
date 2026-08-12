"""Deterministic paired bootstrap statistics for benchmark comparisons."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from document_kv_cache.benchmarks import InferenceMeasurement

if TYPE_CHECKING:
    from document_kv_cache.benchmark_runner import BenchmarkRunResult


PAIRED_BENCHMARK_STATISTICS_RECORD_TYPE = "document_kv.paired_benchmark_statistics.v1"

__all__ = [
    "PAIRED_BENCHMARK_STATISTICS_RECORD_TYPE",
    "ConfidenceInterval",
    "PairedBenchmarkStatistics",
    "paired_benchmark_statistics",
    "paired_benchmark_statistics_to_record",
]


@runtime_checkable
class _BenchmarkRunResultLike(Protocol):
    measurements: Sequence[InferenceMeasurement]
    baseline_arm_id: str

    @property
    def cache_arm_ids(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    bootstrap_samples: int
    paired_samples: int
    independent_examples: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("estimate", "lower", "upper", "confidence_level"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if type(self.paired_samples) is not int or self.paired_samples <= 0:
            raise ValueError("paired_samples must be positive")
        if self.independent_examples is not None and (
            type(self.independent_examples) is not int
            or self.independent_examples <= 0
        ):
            raise ValueError("independent_examples must be positive when provided")
        if self.lower > self.estimate or self.estimate > self.upper:
            raise ValueError("confidence interval must contain its estimate")


@dataclass(frozen=True, slots=True)
class PairedBenchmarkStatistics:
    dataset: str
    baseline_arm_id: str
    cache_arm_id: str
    cache_method: str
    variant_id: str
    artifact_id: str
    paired_samples: int
    paired_examples: int
    missing_baseline_pairs: int
    missing_cache_pairs: int
    duplicate_pair_keys: tuple[str, ...]
    ttft_speedup: ConfidenceInterval | None
    time_to_completion_speedup: ConfidenceInterval | None
    exact_match_delta: ConfidenceInterval | None
    answer_found_delta: ConfidenceInterval | None
    quality_score_deltas: Mapping[str, ConfidenceInterval] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quality_score_deltas",
            MappingProxyType(dict(self.quality_score_deltas)),
        )

    @property
    def complete(self) -> bool:
        return (
            self.paired_samples > 0
            and self.missing_baseline_pairs == 0
            and self.missing_cache_pairs == 0
            and not self.duplicate_pair_keys
            and self.ttft_speedup is not None
            and self.time_to_completion_speedup is not None
        )


def paired_benchmark_statistics(
    result: "BenchmarkRunResult",
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> tuple[PairedBenchmarkStatistics, ...]:
    """Calculate paired request-level intervals for every dataset/cache arm."""

    if not isinstance(result, _BenchmarkRunResultLike):
        raise TypeError("result must be a BenchmarkRunResult")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, (int, float)):
        raise TypeError("confidence_level must be numeric")
    if not 0 < float(confidence_level) < 1:
        raise ValueError("confidence_level must be between zero and one")
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    cache_arms = result.cache_arm_ids
    datasets = sorted({measurement.dataset for measurement in result.measurements})
    rows: list[PairedBenchmarkStatistics] = []
    for cache_arm_id in cache_arms:
        for dataset in datasets:
            rows.append(
                _paired_statistics_for_arm(
                    result.measurements,
                    dataset=dataset,
                    baseline_arm_id=result.baseline_arm_id,
                    cache_arm_id=cache_arm_id,
                    confidence_level=float(confidence_level),
                    bootstrap_samples=bootstrap_samples,
                    seed=_derived_seed(seed, dataset, cache_arm_id),
                )
            )
    return tuple(rows)


def paired_benchmark_statistics_to_record(
    statistics_rows: Sequence[PairedBenchmarkStatistics],
) -> dict[str, Any]:
    rows = tuple(statistics_rows)
    for row in rows:
        if not isinstance(row, PairedBenchmarkStatistics):
            raise TypeError("statistics_rows entries must be PairedBenchmarkStatistics")
    return {
        "record_type": PAIRED_BENCHMARK_STATISTICS_RECORD_TYPE,
        "rows": [_paired_row_to_record(row) for row in rows],
    }


def _paired_statistics_for_arm(
    measurements: Sequence[InferenceMeasurement],
    *,
    dataset: str,
    baseline_arm_id: str,
    cache_arm_id: str,
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
) -> PairedBenchmarkStatistics:
    baseline, baseline_duplicates = _measurements_by_pair_key(
        measurements,
        dataset=dataset,
        arm_id=baseline_arm_id,
    )
    cache, cache_duplicates = _measurements_by_pair_key(
        measurements,
        dataset=dataset,
        arm_id=cache_arm_id,
    )
    baseline_keys = set(baseline)
    cache_keys = set(cache)
    common_keys = sorted(baseline_keys.intersection(cache_keys))
    keyed_pairs = tuple((key, (baseline[key], cache[key])) for key in common_keys)
    pairs = tuple(pair for _, pair in keyed_pairs)
    clusters_by_example: dict[str, list[tuple[InferenceMeasurement, InferenceMeasurement]]] = (
        defaultdict(list)
    )
    for (example_id, _repeat_index), pair in keyed_pairs:
        clusters_by_example[example_id].append(pair)
    measurement_clusters = tuple(
        tuple(clusters_by_example[example_id])
        for example_id in sorted(clusters_by_example)
    )
    cache_methods = {candidate.cache_method for _, candidate in pairs}
    artifact_ids = {candidate.artifact_id for _, candidate in pairs}
    variant_ids = {candidate.variant_id for _, candidate in pairs}
    cache_method = next(iter(cache_methods)) if len(cache_methods) == 1 else ""
    artifact_id = next(iter(artifact_ids)) if len(artifact_ids) == 1 else ""
    variant_id = next(iter(variant_ids)) if len(variant_ids) == 1 else ""
    duplicate_keys = tuple(sorted(baseline_duplicates.union(cache_duplicates)))
    quality_metric_names = sorted(
        {
            metric_name
            for baseline_measurement, cache_measurement in pairs
            for metric_name in set(baseline_measurement.quality_scores).intersection(
                cache_measurement.quality_scores
            )
        }
    )
    quality_score_deltas: dict[str, ConfidenceInterval] = {}
    for metric_name in quality_metric_names:
        interval = _quality_interval(
            measurement_clusters,
            quality=_quality_score_getter(metric_name),
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        if interval is not None:
            quality_score_deltas[metric_name] = interval
    return PairedBenchmarkStatistics(
        dataset=dataset,
        baseline_arm_id=baseline_arm_id,
        cache_arm_id=cache_arm_id,
        cache_method=cache_method,
        variant_id=variant_id,
        artifact_id=artifact_id,
        paired_samples=len(pairs),
        paired_examples=len(measurement_clusters),
        missing_baseline_pairs=len(cache_keys.difference(baseline_keys)),
        missing_cache_pairs=len(baseline_keys.difference(cache_keys)),
        duplicate_pair_keys=duplicate_keys,
        ttft_speedup=_clustered_bootstrap_interval(
            tuple(
                tuple(
                    (left.ttft_seconds, right.ttft_seconds)
                    for left, right in cluster
                )
                for cluster in measurement_clusters
            ),
            _speedup_estimator,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        time_to_completion_speedup=_clustered_bootstrap_interval(
            tuple(
                tuple(
                    (
                        left.time_to_completion_seconds,
                        right.time_to_completion_seconds,
                    )
                    for left, right in cluster
                )
                for cluster in measurement_clusters
            ),
            _speedup_estimator,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        exact_match_delta=_quality_interval(
            measurement_clusters,
            quality=lambda measurement: measurement.exact_match,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        answer_found_delta=_quality_interval(
            measurement_clusters,
            quality=lambda measurement: measurement.answer_found,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        quality_score_deltas=quality_score_deltas,
    )


def _quality_score_getter(
    metric_name: str,
) -> Callable[[InferenceMeasurement], float | None]:
    def get_quality_score(measurement: InferenceMeasurement) -> float | None:
        return measurement.quality_scores.get(metric_name)

    return get_quality_score


def _measurements_by_pair_key(
    measurements: Sequence[InferenceMeasurement],
    *,
    dataset: str,
    arm_id: str,
) -> tuple[dict[tuple[str, int], InferenceMeasurement], set[str]]:
    grouped: dict[tuple[str, int], list[InferenceMeasurement]] = defaultdict(list)
    for measurement in measurements:
        if measurement.dataset == dataset and measurement.arm_id == arm_id and measurement.ok:
            grouped[(measurement.example_id, measurement.repeat_index)].append(measurement)
    duplicates = {
        f"{example_id}:repeat-{repeat_index}"
        for (example_id, repeat_index), values in grouped.items()
        if len(values) > 1
    }
    unique = {key: values[0] for key, values in grouped.items() if len(values) == 1}
    return unique, duplicates


def _quality_interval(
    clusters: Sequence[Sequence[tuple[InferenceMeasurement, InferenceMeasurement]]],
    *,
    quality: Callable[[InferenceMeasurement], bool | float | None],
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
) -> ConfidenceInterval | None:
    quality_pairs: list[tuple[float, float]] = []
    for cluster in clusters:
        baseline_values: list[float] = []
        cache_values: list[float] = []
        for baseline, cache in cluster:
            baseline_value = quality(baseline)
            cache_value = quality(cache)
            if baseline_value is None or cache_value is None:
                continue
            baseline_values.append(float(baseline_value))
            cache_values.append(float(cache_value))
        if baseline_values and cache_values:
            quality_pairs.append(
                (statistics.fmean(baseline_values), statistics.fmean(cache_values))
            )
    return _clustered_bootstrap_interval(
        tuple((pair,) for pair in quality_pairs),
        _delta_estimator,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def _clustered_bootstrap_interval(
    clusters: Sequence[Sequence[tuple[float, float]]],
    estimator: Callable[[Sequence[tuple[float, float]]], float | None],
    *,
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
) -> ConfidenceInterval | None:
    nonempty = tuple(tuple(cluster) for cluster in clusters if cluster)
    if not nonempty:
        return None
    pairs = tuple(pair for cluster in nonempty for pair in cluster)
    estimate = estimator(pairs)
    if estimate is None:
        return None
    generator = random.Random(seed)
    estimates: list[float] = []
    cluster_count = len(nonempty)
    for _ in range(bootstrap_samples):
        sampled = tuple(
            pair
            for _ in range(cluster_count)
            for pair in nonempty[generator.randrange(cluster_count)]
        )
        value = estimator(sampled)
        if value is not None:
            estimates.append(value)
    if not estimates:
        return None
    estimates.sort()
    alpha = 1.0 - confidence_level
    lower = min(estimate, _percentile(estimates, alpha / 2.0))
    upper = max(estimate, _percentile(estimates, 1.0 - alpha / 2.0))
    return ConfidenceInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        paired_samples=len(pairs),
        independent_examples=cluster_count,
    )


def _bootstrap_interval(
    pairs: Sequence[tuple[float, float]],
    estimator: Callable[[Sequence[tuple[float, float]]], float | None],
    *,
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
) -> ConfidenceInterval | None:
    if not pairs:
        return None
    estimate = estimator(pairs)
    if estimate is None:
        return None
    generator = random.Random(seed)
    sample_count = len(pairs)
    estimates: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [pairs[generator.randrange(sample_count)] for _ in range(sample_count)]
        value = estimator(sample)
        if value is not None:
            estimates.append(value)
    if not estimates:
        return None
    estimates.sort()
    alpha = 1.0 - confidence_level
    lower = min(estimate, _percentile(estimates, alpha / 2.0))
    upper = max(estimate, _percentile(estimates, 1.0 - alpha / 2.0))
    return ConfidenceInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        paired_samples=sample_count,
    )


def _speedup_estimator(pairs: Sequence[tuple[float, float]]) -> float | None:
    if any(baseline <= 0 or candidate <= 0 for baseline, candidate in pairs):
        return None
    return statistics.median(
        baseline / candidate for baseline, candidate in pairs
    )


def _delta_estimator(pairs: Sequence[tuple[float, float]]) -> float:
    return statistics.fmean(right - left for left, right in pairs)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = percentile * (len(values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(values) - 1)
    weight = position - lower_index
    return values[lower_index] * (1 - weight) + values[upper_index] * weight


def _derived_seed(seed: int, dataset: str, cache_arm_id: str) -> int:
    digest = hashlib.sha256(f"{seed}|{dataset}|{cache_arm_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _paired_row_to_record(row: PairedBenchmarkStatistics) -> dict[str, Any]:
    return {
        "dataset": row.dataset,
        "baseline_arm_id": row.baseline_arm_id,
        "cache_arm_id": row.cache_arm_id,
        "cache_method": row.cache_method,
        "variant_id": row.variant_id,
        "artifact_id": row.artifact_id,
        "paired_samples": row.paired_samples,
        "paired_examples": row.paired_examples,
        "missing_baseline_pairs": row.missing_baseline_pairs,
        "missing_cache_pairs": row.missing_cache_pairs,
        "duplicate_pair_keys": list(row.duplicate_pair_keys),
        "complete": row.complete,
        "ttft_speedup": _interval_to_record(row.ttft_speedup),
        "time_to_completion_speedup": _interval_to_record(row.time_to_completion_speedup),
        "paired_median_ttft_speedup": _interval_to_record(row.ttft_speedup),
        "paired_median_time_to_completion_speedup": _interval_to_record(
            row.time_to_completion_speedup
        ),
        "exact_match_delta": _interval_to_record(row.exact_match_delta),
        "answer_found_delta": _interval_to_record(row.answer_found_delta),
        "quality_score_deltas": {
            metric_name: _interval_to_record(interval)
            for metric_name, interval in row.quality_score_deltas.items()
        },
    }


def _interval_to_record(interval: ConfidenceInterval | None) -> dict[str, Any] | None:
    if interval is None:
        return None
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "confidence_level": interval.confidence_level,
        "bootstrap_samples": interval.bootstrap_samples,
        "paired_samples": interval.paired_samples,
        "independent_examples": interval.independent_examples,
        "bootstrap_unit": "example" if interval.independent_examples is not None else "pair",
    }
