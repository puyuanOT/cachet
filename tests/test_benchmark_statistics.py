from __future__ import annotations

import copy
import math
import random
import statistics
from collections import Counter

import pytest

from document_kv_cache.benchmark_statistics import five_deployment_paired_statistics


def _constant_blocks(value: float = math.log(2)) -> dict[int, dict[str, tuple[float, ...]]]:
    return {
        block: {f"example-{example}": (value, value, value) for example in range(4)}
        for block in range(1, 6)
    }


def test_five_deployment_constant_effect_and_explicit_sampling_metadata() -> None:
    result = five_deployment_paired_statistics(_constant_blocks(), bootstrap_samples=500)
    assert result["geometric_mean_speedup"] == pytest.approx(2)
    assert result["confidence_interval"] == {"lower": 2, "upper": 2}
    assert result["deployment_count"] == 5
    assert result["distinct_examples"] == 4
    assert result["paired_repeats_per_example"] == 3
    assert result["bootstrap_samples"] == 500
    assert result["estimand"] == (
        "geometric_mean_control_over_treatment_across_deployments_and_examples"
    )
    assert result["estimator"] == (
        "exponentiated_equal_weight_mean_of_paired_repeat_log_ratios"
    )
    assert result["bootstrap_unit"] == (
        "paired_crossed_deployment_and_example_with_paired_repeats_retained"
    )
    assert result["interval_scope"] == "pointwise_estimation_only"
    assert result["coverage_limitation"] == (
        "nominal_percentile_interval_no_exact_coverage_guarantee_with_five_deployments"
    )
    assert [item["block"] for item in result["deployment_effects"]] == list(range(1, 6))


def test_five_deployment_keeps_repeats_together() -> None:
    # Every example's paired repeats cancel. Resampling individual repeats
    # would falsely introduce uncertainty even though every cluster has mean 0.
    observations = {
        block: {f"example-{example}": (-2.0, 0.0, 2.0) for example in range(4)}
        for block in range(1, 6)
    }
    result = five_deployment_paired_statistics(observations, bootstrap_samples=500)
    assert result["geometric_mean_speedup"] == 1
    assert result["confidence_interval"] == {"lower": 1, "upper": 1}


def test_five_deployment_resamples_deployments_even_without_within_block_noise() -> None:
    observations = {
        block: {
            f"example-{example}": (math.log(2 ** (block - 1)),) * 3
            for example in range(4)
        }
        for block in range(1, 6)
    }
    result = five_deployment_paired_statistics(observations)
    assert result["geometric_mean_speedup"] == pytest.approx(4)
    assert result["confidence_interval"]["lower"] < 3
    assert result["confidence_interval"]["upper"] > 5
    assert [item["geometric_mean_speedup"] for item in result["deployment_effects"]] == (
        pytest.approx([1, 2, 4, 8, 16])
    )


@pytest.mark.parametrize("example_count,log_limit", [(4, 1.0), (32, 0.3125)])
def test_shared_example_effects_do_not_become_five_independent_example_samples(
    example_count: int, log_limit: float,
) -> None:
    # Every deployment observes the SAME example effects. The bootstrap log
    # variance is 1/N, not 1/(5N); nested example draws spuriously narrow this CI.
    logs = [-1.0] * (example_count // 2) + [1.0] * (example_count // 2)
    observations = {
        block: {
            f"example-{example:02}": (value,) * 3
            for example, value in enumerate(logs)
        }
        for block in range(1, 6)
    }
    result = five_deployment_paired_statistics(observations)
    assert result["geometric_mean_speedup"] == 1
    assert result["confidence_interval"] == pytest.approx({
        "lower": math.exp(-log_limit),
        "upper": math.exp(log_limit),
    })


def test_crossed_interaction_matches_independent_factor_weight_reference() -> None:
    # Nonadditive block/example interactions prevent either factor from being
    # averaged away. The reference weights every original cell by independently
    # drawn row and column multiplicities, rather than indexing sampled cells.
    cell_means = {
        block: {
            example: (block - 3) / 4 + (example - 2) * block / 7
            for example in range(4)
        }
        for block in range(1, 6)
    }
    observations = {
        block: {
            f"example-{example}": (value - 0.25, value, value + 0.25)
            for example, value in values.items()
        }
        for block, values in cell_means.items()
    }
    sample_count = 2_000
    seed = 19
    rng = random.Random(seed)
    reference_draws = []
    for _ in range(sample_count):
        row_counts = Counter(rng.randrange(1, 6) for _ in range(5))
        column_counts = Counter(rng.randrange(4) for _ in range(4))
        weighted_log_sum = math.fsum(
            row_counts[block] * column_counts[example] * value
            for block, values in cell_means.items()
            for example, value in values.items()
        )
        reference_draws.append(math.exp(weighted_log_sum / 20))
    # Inclusive stdlib quantiles implement type 7 without calling production's
    # percentile routine; 2.5% and 97.5% are cut points 1 and 39 out of 40.
    reference_quantiles = statistics.quantiles(reference_draws, n=40, method="inclusive")
    result = five_deployment_paired_statistics(
        observations, bootstrap_samples=sample_count, seed=seed,
    )
    assert result["confidence_interval"] == pytest.approx({
        "lower": reference_quantiles[0],
        "upper": reference_quantiles[-1],
    })
    assert result["geometric_mean_speedup"] == pytest.approx(math.exp(
        statistics.fmean(value for values in cell_means.values() for value in values.values())
    ))


def test_percentile_interval_is_not_clipped_to_include_point_estimate() -> None:
    observations = {
        block: {f"example-{example}": (float(block - 1),) for example in range(4)}
        for block in range(1, 6)
    }
    result = five_deployment_paired_statistics(observations, bootstrap_samples=1, seed=0)
    assert result["confidence_interval"]["lower"] > result["geometric_mean_speedup"]
    assert result["confidence_interval"]["upper"] == result["confidence_interval"]["lower"]


def test_five_deployment_statistics_ignore_mapping_insertion_order() -> None:
    observations = _constant_blocks()
    observations[3]["example-1"] = (0.0, 1.0, 2.0)
    reordered = {
        block: dict(reversed(list(observations[block].items())))
        for block in reversed(observations)
    }
    original = copy.deepcopy(observations)
    result = five_deployment_paired_statistics(observations, bootstrap_samples=500, seed=42)
    assert result == five_deployment_paired_statistics(
        reordered, bootstrap_samples=500, seed=42
    )
    assert observations == original


@pytest.mark.parametrize("mutation", ["block", "boolean_block", "example", "repeat", "empty", "few"])
def test_five_deployment_statistics_reject_incomplete_sampling_units(mutation: str) -> None:
    observations = _constant_blocks()
    if mutation == "block":
        del observations[5]
    elif mutation == "boolean_block":
        observations[True] = observations.pop(1)
    elif mutation == "example":
        observations[2]["different-example"] = observations[2].pop("example-1")
    elif mutation == "repeat":
        observations[2]["example-1"] = (1.0,)
    elif mutation == "empty":
        observations[2]["example-1"] = ()
    else:
        observations = {block: {"only-one": (1.0,)} for block in range(1, 6)}
    with pytest.raises(ValueError):
        five_deployment_paired_statistics(observations, bootstrap_samples=10)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, True, 1e308, -1e308])
def test_five_deployment_statistics_reject_nonfinite_or_unrepresentable_effects(
    invalid: float,
) -> None:
    observations = _constant_blocks()
    observations[2]["example-1"] = (invalid,) * 3
    with pytest.raises(ValueError):
        five_deployment_paired_statistics(observations, bootstrap_samples=10)


@pytest.mark.parametrize("samples", [0, -1, True])
def test_five_deployment_statistics_reject_invalid_draw_counts(samples: int) -> None:
    with pytest.raises(ValueError, match="bootstrap_samples"):
        five_deployment_paired_statistics(_constant_blocks(), bootstrap_samples=samples)


def test_five_deployment_statistics_reject_boolean_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        five_deployment_paired_statistics(_constant_blocks(), seed=True)
