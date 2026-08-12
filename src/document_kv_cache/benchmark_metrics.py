"""Canonical benchmark metric formulas shared by runners and evidence gates."""

from __future__ import annotations

import math
from collections.abc import Iterable


__all__ = [
    "aggregate_decode_tokens_per_second",
    "request_decode_tokens_per_second",
    "latency_speedup",
    "quality_delta",
]


def request_decode_tokens_per_second(
    completion_tokens: int,
    ttft_seconds: float,
    time_to_completion_seconds: float,
) -> float | None:
    """Return decode throughput for one request.

    This is intentionally distinct from :func:`aggregate_decode_tokens_per_second`,
    which weights requests by their decode duration.
    """

    values = tuple(
        _validated_decode_sample(
            completion_tokens,
            ttft_seconds,
            time_to_completion_seconds,
        )
    )
    tokens, decode_seconds = values
    if decode_seconds <= 0:
        return None
    return tokens / decode_seconds


def aggregate_decode_tokens_per_second(
    samples: Iterable[tuple[int, float, float]],
) -> float | None:
    """Return total completion tokens divided by total post-TTFT decode time."""

    total_tokens = 0
    total_decode_seconds = 0.0
    observed = False
    for completion_tokens, ttft_seconds, time_to_completion_seconds in samples:
        tokens, decode_seconds = _validated_decode_sample(
            completion_tokens,
            ttft_seconds,
            time_to_completion_seconds,
        )
        observed = True
        total_tokens += tokens
        total_decode_seconds += decode_seconds
    if not observed or total_decode_seconds <= 0:
        return None
    return total_tokens / total_decode_seconds


def _validated_decode_sample(
    completion_tokens: int,
    ttft_seconds: float,
    time_to_completion_seconds: float,
) -> tuple[int, float]:
    if type(completion_tokens) is not int or completion_tokens < 0:
        raise ValueError("completion_tokens must be a non-negative integer")
    for field_name, value in (
        ("ttft_seconds", ttft_seconds),
        ("time_to_completion_seconds", time_to_completion_seconds),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{field_name} must be finite")
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative")
    if time_to_completion_seconds < ttft_seconds:
        raise ValueError("time_to_completion_seconds must be at least ttft_seconds")
    return completion_tokens, time_to_completion_seconds - ttft_seconds


def latency_speedup(
    baseline_seconds: float | None,
    candidate_seconds: float | None,
) -> float | None:
    if baseline_seconds is None or candidate_seconds is None:
        return None
    if baseline_seconds <= 0 or candidate_seconds <= 0:
        return None
    return baseline_seconds / candidate_seconds


def quality_delta(
    candidate_rate: float | None,
    baseline_rate: float | None,
) -> float | None:
    if candidate_rate is None or baseline_rate is None:
        return None
    return candidate_rate - baseline_rate
