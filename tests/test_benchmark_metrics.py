from __future__ import annotations

import pytest

from document_kv_cache.benchmark_metrics import (
    aggregate_decode_tokens_per_second,
    latency_speedup,
    quality_delta,
)


def test_aggregate_decode_throughput_excludes_ttft() -> None:
    assert aggregate_decode_tokens_per_second(
        (
            (20, 10.0, 30.0),
            (20, 20.0, 40.0),
        )
    ) == pytest.approx(1.0)


def test_aggregate_decode_throughput_rejects_invalid_timing() -> None:
    with pytest.raises(ValueError, match="at least"):
        aggregate_decode_tokens_per_second(((1, 2.0, 1.0),))


def test_canonical_comparison_formulas() -> None:
    assert latency_speedup(10.0, 2.0) == 5.0
    assert latency_speedup(None, 2.0) is None
    assert quality_delta(0.9, 0.8) == pytest.approx(0.1)
