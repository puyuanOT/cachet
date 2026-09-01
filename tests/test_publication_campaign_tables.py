from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from document_kv_cache import publication_campaign_tables as campaign_tables


_CACHE_KEYS = (
    "backend_bytes_read",
    "cold_read_attested_count",
    "eviction_requested_count",
    "eviction_succeeded_count",
    "expected_backend_bytes_read",
    "load_count",
    "mounted_path_load_count",
    "payload_cache_hit_count",
    "payload_cache_miss_count",
    "storage_materialization_count",
)
_PARSER_STATUSES = (
    "ok",
    "missing_block",
    "multiple_or_malformed_blocks",
    "extraneous_text",
    "nested_block",
    "empty_answer",
)
_METRICS = {
    "biography": ("exact_match",),
    "hotpotqa": ("exact_match", "f1"),
    "musique": ("answer_em", "answer_f1"),
    "niah": ("accuracy",),
}


def _cache_block(method: str, setting_id: str | None) -> dict[str, int]:
    zero = {key: 0 for key in _CACHE_KEYS}
    if method == "baseline_prefill":
        return zero
    if setting_id == "storage-ram":
        return {
            **zero,
            "load_count": 256,
            "payload_cache_hit_count": 256,
        }
    backend_bytes = 1024**3
    return {
        **zero,
        "backend_bytes_read": backend_bytes,
        "cold_read_attested_count": 0 if setting_id == "storage-uc" else 256,
        "eviction_requested_count": 256,
        "eviction_succeeded_count": 256,
        "expected_backend_bytes_read": backend_bytes,
        "load_count": 256,
        "mounted_path_load_count": 256 if setting_id == "storage-uc" else 0,
    }


def _cell(
    cell_id: str,
    method: str,
    tokens: int,
    concurrency: int,
    family: str | None,
    setting_id: str | None,
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for block_index in range(1, 6):
        cache = _cache_block(method, setting_id)
        job_suffix = (
            setting_id
            if setting_id is not None
            else f"{tokens // 1024}k-c{concurrency}-{method.removesuffix('_prefill')}"
        )
        blocks.append(
            {
                "cache_telemetry": cache,
                "configured_closed_loop_concurrency": concurrency,
                "deployment_block": block_index,
                "gpu_utilization_sample_count": 2,
                "job_id": f"block-{block_index:02d}-{job_suffix}",
                "mean_gpu_utilization_percent": 50.0,
                "observation_count": 256,
                "p50_decode_tokens_per_second": 32.0,
                "p50_time_to_completion_seconds": 3.0,
                "p50_ttft_seconds": 1.0,
                "p95_time_to_completion_seconds": 4.0,
                "p95_ttft_seconds": 2.0,
                "peak_gpu_process_memory_bytes": 1024**3,
                "peak_gpu_utilization_percent": 60.0,
                "peak_host_memory_used_bytes": 2 * 1024**3,
                "peak_process_tree_rss_bytes": 3 * 1024**3,
            }
        )
    cache = {
        key: sum(block["cache_telemetry"][key] for block in blocks)
        for key in _CACHE_KEYS
    }
    return {
        "cache_telemetry": cache,
        "cell_id": cell_id,
        "cell_kind": (
            "core_pooled_five_blocks"
            if setting_id is None
            else "auxiliary_pooled_five_blocks"
        ),
        "cell_sha256": "0" * 64,
        "comparison_family": family,
        "configured_closed_loop_concurrency": concurrency,
        "gpu_utilization_sample_count": 10,
        "input_tokens": tokens,
        "mean_gpu_utilization_percent": 50.0,
        "method_id": method,
        "observation_count": 1_280,
        "p50_decode_tokens_per_second": 32.0,
        "p50_time_to_completion_seconds": 3.0,
        "p50_ttft_seconds": 1.0,
        "p95_time_to_completion_seconds": 4.0,
        "p95_ttft_seconds": 2.0,
        "peak_gpu_process_memory_bytes": 1024**3,
        "peak_gpu_utilization_percent": 60.0,
        "peak_host_memory_used_bytes": 2 * 1024**3,
        "peak_process_tree_rss_bytes": 3 * 1024**3,
        "physical_blocks": blocks,
        "quantile_method": "empirical_nearest_rank",
        "request_parallelism": concurrency,
        "setting_id": setting_id,
    }


def _descriptive_cells() -> list[dict[str, Any]]:
    rows = [
        _cell(
            f"core-{method}-{tokens}-c{concurrency}",
            method,
            tokens,
            concurrency,
            None,
            None,
        )
        for tokens in (8_192, 16_384, 32_768)
        for concurrency in (1, 2, 4)
        for method in ("baseline_prefill", "vanilla_prefill")
    ]
    rows.extend(
        _cell(
            f"auxiliary-{setting_id}",
            "vanilla_prefill",
            16_384,
            4,
            family,
            setting_id,
        )
        for setting_id, family in (
            ("precision-bf16", "precision"),
            ("storage-disk", "storage"),
            ("storage-ram", "storage"),
            ("storage-uc", "storage"),
            ("hardware-a10g", "hardware"),
        )
    )
    return rows


def _estimands() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tokens in (8_192, 16_384, 32_768):
        for concurrency in (1, 2, 4):
            rows.append(
                _estimand(
                    f"method-{tokens}-c{concurrency}",
                    "method",
                    None,
                    f"core-baseline_prefill-{tokens}-c{concurrency}",
                    f"core-vanilla_prefill-{tokens}-c{concurrency}",
                    tokens,
                    concurrency,
                    128,
                )
            )
    for setting_id, family in (
        ("precision-bf16", "precision"),
        ("storage-ram", "storage"),
        ("storage-uc", "storage"),
        ("hardware-a10g", "hardware"),
    ):
        storage = family == "storage"
        rows.append(
            _estimand(
                f"auxiliary-{setting_id}",
                family,
                setting_id,
                (
                    "auxiliary-storage-disk"
                    if storage
                    else "core-vanilla_prefill-16384-c4"
                ),
                f"auxiliary-{setting_id}",
                16_384,
                4,
                8 if storage else 128,
            )
        )
    return rows


def _estimand(
    estimand_id: str,
    family: str,
    setting_id: str | None,
    control_cell_id: str,
    treatment_cell_id: str,
    input_tokens: int,
    concurrency: int,
    examples: int,
) -> dict[str, Any]:
    metric = {
        "confidence_interval_95": [0.9, 1.1],
        "geometric_mean_speedup": 1.0,
    }
    return {
        "comparison_family": family,
        "control_cell_id": control_cell_id,
        "deployment_block_count": 5,
        "estimand_id": estimand_id,
        "example_count_per_block": examples,
        "input_tokens": input_tokens,
        "metrics": {
            "ttft": deepcopy(metric),
            "time_to_completion": deepcopy(metric),
        },
        "paired_request_count": 1_280,
        "request_parallelism": concurrency,
        "setting_id": setting_id,
        "speedup_direction": "control_latency_divided_by_treatment_latency",
        "treatment_cell_id": treatment_cell_id,
    }


def _metric_summary(count: int, mean: float) -> dict[str, Any]:
    return {
        "example_count": count,
        "invalid_parser_score_sum": 0.0,
        "mean": mean,
        "sum": mean * count,
    }


def _paired_summary(count: int, mean: float) -> dict[str, Any]:
    return {
        "bootstrap_ci95": {
            "draws": 20_000,
            "lower": -0.1,
            "upper": 0.2,
        },
        "example_count": count,
        "mean": mean,
        "seed_sha256": "1" * 64,
    }


def _score_datasets() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset, metric_names in _METRICS.items():
        count = 10
        parser_counts = {status: 0 for status in _PARSER_STATUSES}
        parser_counts["ok"] = count
        output[dataset] = {
            "example_count": count,
            "methods": {
                "baseline_prefill": {
                    "example_count": count,
                    "metrics": {
                        metric: _metric_summary(count, 0.4)
                        for metric in metric_names
                    },
                    "parser_status_counts": deepcopy(parser_counts),
                },
                "vanilla_prefill": {
                    "example_count": count,
                    "metrics": {
                        metric: _metric_summary(count, 0.5)
                        for metric in metric_names
                    },
                    "parser_status_counts": deepcopy(parser_counts),
                },
            },
            "paired_vanilla_minus_baseline": {
                metric: _paired_summary(count, 0.1) for metric in metric_names
            },
        }
    return output


def _niah_grid() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for tokens in (8, 16, 32):
        for depth in (10, 50, 90):
            cell_id = f"niah-{tokens}k-depth-{depth:02d}"
            output[cell_id] = {
                "example_count": 10,
                "methods": {
                    "baseline_prefill": {
                        "accuracy": _metric_summary(10, 0.4)
                    },
                    "vanilla_prefill": {
                        "accuracy": _metric_summary(10, 0.5)
                    },
                },
                "paired_vanilla_minus_baseline": {
                    "accuracy": _paired_summary(10, 0.1)
                },
            }
    return output


def _protocol() -> dict[str, Any]:
    return {
        "complete_inventory_required": True,
        "input_length": {
            "max_natural_prompt_tokens": 32_768,
            "padding": False,
            "tokenizer_truncation": False,
        },
        "lifecycle": [
            "generate_q8_kv",
            "baseline_inference",
            "vanilla_inference",
            "validate_paired_outputs",
            "commit_durable_evidence",
            "delete_ephemeral_q8_kv",
        ],
        "max_tokens": 64,
        "methods": ["baseline_prefill", "vanilla_prefill"],
        "natural_eos": True,
        "passes_per_method": 1,
        "protocol_id": "cachet-vllm-0.27.1-complete-score-v1",
        "request_parallelism": 4,
        "temperature": 0.0,
    }


def _coverage() -> dict[str, Any]:
    return {
        "checked_cache_request_count": 101_573,
        "checked_distinct_example_count": 83_653,
        "cold_attested_request_count": 15_360,
        "full_score_cache_request_count": 83_653,
        "full_score_identity_count": 83_653,
        "full_score_passes_per_method": 1,
        "full_score_phase_count": 20,
        "full_score_shard_count": 160,
        "full_score_wave_count": 10,
        "latency_cache_request_count": 17_920,
        "latency_descriptive_cell_count": 23,
        "latency_estimand_count": 13,
        "latency_job_count": 115,
        "latency_request_count": 29_440,
        "methods": ["baseline_prefill", "vanilla_prefill"],
        "niah_cell_count": 9,
    }


def _report() -> dict[str, Any]:
    scorer_contracts = []
    for dataset, metrics in _METRICS.items():
        scorer_contracts.append(
            {
                "answer_parser_digest": "2" * 64,
                "answer_parser_id": "cachet.single_final_answer",
                "answer_parser_plugin_path": "module:parse",
                "answer_parser_version": "1",
                "dataset": dataset,
                "metric_names": list(metrics),
                "plugin_path": "module:score",
                "publication_approved": True,
                "scorer_id": f"synthetic.{dataset}",
                "scorer_version": "1",
            }
        )
    return {
        "campaign_id": "vllm-0271-publication-v1",
        "closed_record_sha256": "3" * 64,
        "coverage": _coverage(),
        "engine_version": "0.27.1",
        "latency": {
            "analysis": {},
            "descriptive_cells": _descriptive_cells(),
            "estimates": _estimands(),
        },
        "policy": "publication",
        "quality": {
            "aggregation_unit": "per_example_once_never_shard_means",
            "bootstrap": {},
            "datasets": _score_datasets(),
            "niah_grid": _niah_grid(),
            "protocol": _protocol(),
        },
        "scorer_contracts": scorer_contracts,
    }


def _document() -> str:
    parts = ["# Synthetic benchmark page\n"]
    for section in campaign_tables.PUBLICATION_TABLE_SECTION_ORDER:
        begin, end = campaign_tables.PUBLICATION_TABLE_REGION_MARKERS[section]
        parts.append(f"before {section}\n{begin}\nstale\n{end}\n")
    parts.append("after all regions\n")
    return "".join(parts)


@pytest.fixture(autouse=True)
def _accept_synthetic_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign_tables,
        "validate_vllm_0271_publication_report_pair",
        lambda _report_record, _gate_record: None,
    )


def test_render_is_deterministic_and_uses_frozen_row_orders() -> None:
    report = _report()
    report["quality"]["datasets"] = dict(
        reversed(list(report["quality"]["datasets"].items()))
    )
    before = deepcopy(report)
    first = campaign_tables.render_vllm_0271_publication_table_regions(
        report,
        {"synthetic": True},
    )
    second = campaign_tables.render_vllm_0271_publication_table_regions(
        deepcopy(report),
        {"synthetic": True},
    )

    assert dict(first) == dict(second)
    assert report == before
    assert tuple(first) == campaign_tables.PUBLICATION_TABLE_SECTION_ORDER
    assert first["core_latency"].count("| Baseline |") == 9
    assert first["core_latency"].count("| Vanilla&nbsp;KV |") == 9
    assert first["latency_estimands"].count("Five matched blocks") == 13
    assert first["dataset_scores"].count("\n| ") - 1 == 8
    assert first["dataset_scores"].startswith(
        "| Dataset | Governed metric | n |"
    )
    assert first["niah_grid"].count("\n| ") - 1 == 9
    assert first["resource_cache"].count("\n| ") - 1 == 23
    assert first["dataset_scores"].index("| Biography |") < first[
        "dataset_scores"
    ].index("| HotpotQA |")
    assert first["niah_grid"].count("| 8k |") == 3
    assert first["niah_grid"].count("| 16k |") == 3
    assert first["niah_grid"].count("| 32k |") == 3
    assert (
        "ok=10; missing_block=0; multiple_or_malformed_blocks=0; "
        "extraneous_text=0; nested_block=0; empty_answer=0"
        in first["dataset_scores"]
    )
    assert "| LongBench v2 | N/A (runner not implemented) |" in first[
        "dataset_scores"
    ]
    assert "| RULER | N/A (runner not implemented) |" in first[
        "dataset_scores"
    ]
    assert "1.000 GiB (1,073,741,824 bytes)" in first["resource_cache"]
    appendix = campaign_tables.render_vllm_0271_publication_appendix_readme(
        report,
        {"synthetic": True},
    )
    assert appendix.startswith("# vLLM 0.27.1 Publication Campaign\n")
    expected_digest = "3" * 64
    assert f"`{expected_digest}`" in appendix
    assert appendix.endswith("intentionally excluded.\n")


def test_marker_pairs_are_unique_and_frozen() -> None:
    markers = [
        marker
        for section in campaign_tables.PUBLICATION_TABLE_SECTION_ORDER
        for marker in campaign_tables.PUBLICATION_TABLE_REGION_MARKERS[section]
    ]

    assert len(markers) == 20
    assert len(set(markers)) == 20
    assert markers[0] == (
        "<!-- cachet:vllm-0271-publication-table:status:begin -->"
    )
    assert markers[-1] == (
        "<!-- cachet:vllm-0271-publication-table:resource-cache:end -->"
    )


def test_cachet_facade_exports_the_canonical_renderer() -> None:
    import cachet.publication_campaign_tables as cachet_tables

    assert (
        cachet_tables.render_vllm_0271_publication_table_regions
        is campaign_tables.render_vllm_0271_publication_table_regions
    )
    assert (
        cachet_tables.render_vllm_0271_publication_appendix_readme
        is campaign_tables.render_vllm_0271_publication_appendix_readme
    )


def test_render_delegates_to_the_public_exact_pair_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    gate = {"synthetic": True}
    calls: list[tuple[Any, Any]] = []

    def validate_pair(report_record: Any, gate_record: Any) -> None:
        calls.append((report_record, gate_record))

    monkeypatch.setattr(
        campaign_tables,
        "validate_vllm_0271_publication_report_pair",
        validate_pair,
    )

    campaign_tables.render_vllm_0271_publication_table_regions(report, gate)

    assert calls == [(report, gate)]
    assert calls[0][0] is not report
    assert calls[0][1] is not gate


def test_validator_and_caller_mutation_cannot_change_the_render_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    gate = {"synthetic": True}

    def mutate_after_snapshot(
        validation_report: dict[str, Any],
        validation_gate: dict[str, Any],
    ) -> None:
        validation_report["latency"]["descriptive_cells"][0][
            "p50_ttft_seconds"
        ] = 88.0
        validation_gate["synthetic"] = False
        report["latency"]["descriptive_cells"][0]["p50_ttft_seconds"] = 99.0
        gate["synthetic"] = False

    monkeypatch.setattr(
        campaign_tables,
        "validate_vllm_0271_publication_report_pair",
        mutate_after_snapshot,
    )

    regions = campaign_tables.render_vllm_0271_publication_table_regions(
        report,
        gate,
    )

    first_baseline = next(
        line
        for line in regions["core_latency"].splitlines()
        if line.startswith("| Baseline |")
    )
    assert "| 1.0000 |" in first_baseline
    assert "88.0000" not in first_baseline
    assert "99.0000" not in first_baseline


def test_replace_is_exact_idempotent_and_validatable() -> None:
    report = _report()
    gate = {"synthetic": True}
    original = _document()

    rendered = campaign_tables.replace_vllm_0271_publication_table_regions(
        original,
        report,
        gate,
    )

    assert rendered.startswith("# Synthetic benchmark page\n")
    assert rendered.endswith("after all regions\n")
    assert "\nstale\n" not in rendered
    assert (
        campaign_tables.replace_vllm_0271_publication_table_regions(
            rendered,
            report,
            gate,
        )
        == rendered
    )
    campaign_tables.validate_vllm_0271_publication_table_regions(
        rendered,
        report,
        gate,
    )


def test_region_validation_rejects_one_byte_table_tamper() -> None:
    report = _report()
    gate = {"synthetic": True}
    rendered = campaign_tables.replace_vllm_0271_publication_table_regions(
        _document(),
        report,
        gate,
    )
    tampered = rendered.replace("1.0000", "1.0001", 1)

    with pytest.raises(ValueError, match="not canonical"):
        campaign_tables.validate_vllm_0271_publication_table_regions(
            tampered,
            report,
            gate,
        )


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_renderer_rejects_non_json_or_nonfinite_numbers(value: Any) -> None:
    report = _report()
    report["latency"]["descriptive_cells"][0]["p50_ttft_seconds"] = value

    with pytest.raises(
        (TypeError, ValueError),
        match=r"(?:finite JSON number|non-JSON or non-finite value)",
    ):
        campaign_tables.render_vllm_0271_publication_table_regions(
            report,
            {"synthetic": True},
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["latency"]["descriptive_cells"].reverse(),
        lambda report: report["latency"]["estimates"].reverse(),
        lambda report: report["scorer_contracts"].reverse(),
    ],
)
def test_renderer_rejects_governed_row_order_drift(mutate: Any) -> None:
    report = _report()
    mutate(report)

    with pytest.raises(ValueError, match="order"):
        campaign_tables.render_vllm_0271_publication_table_regions(
            report,
            {"synthetic": True},
        )


def test_renderer_rejects_cell_schema_and_control_identity_tamper() -> None:
    extra = _report()
    extra["latency"]["descriptive_cells"][0]["unexpected"] = 1
    with pytest.raises(ValueError, match="schema drift"):
        campaign_tables.render_vllm_0271_publication_table_regions(
            extra,
            {"synthetic": True},
        )

    control = _report()
    control["latency"]["estimates"][0]["control_cell_id"] = (
        "core-vanilla_prefill-8192-c1"
    )
    with pytest.raises(ValueError, match="identity drift"):
        campaign_tables.render_vllm_0271_publication_table_regions(
            control,
            {"synthetic": True},
        )


def test_renderer_rejects_cache_sum_and_protocol_type_tamper() -> None:
    cache = _report()
    cache["latency"]["descriptive_cells"][1]["cache_telemetry"][
        "load_count"
    ] += 1
    with pytest.raises(ValueError, match="cache sum drift"):
        campaign_tables.render_vllm_0271_publication_table_regions(
            cache,
            {"synthetic": True},
        )

    protocol = _report()
    protocol["quality"]["protocol"]["temperature"] = 0
    with pytest.raises(ValueError, match="protocol drift"):
        campaign_tables.render_vllm_0271_publication_table_regions(
            protocol,
            {"synthetic": True},
        )

    for cold_count in (15_359, 16_641):
        cold = _report()
        cold["coverage"]["cold_attested_request_count"] = cold_count
        with pytest.raises(ValueError, match="cold-attestation coverage drift"):
            campaign_tables.render_vllm_0271_publication_table_regions(
                cold,
                {"synthetic": True},
            )

    invalid_parser_credit = _report()
    invalid_parser_credit["quality"]["datasets"]["biography"]["methods"][
        "baseline_prefill"
    ]["metrics"]["exact_match"]["invalid_parser_score_sum"] = 0.25
    with pytest.raises(ValueError, match="credits an invalid parsed answer"):
        campaign_tables.render_vllm_0271_publication_table_regions(
            invalid_parser_credit,
            {"synthetic": True},
        )


def test_marker_parser_rejects_duplicate_missing_and_reordered_regions() -> None:
    report = _report()
    gate = {"synthetic": True}
    document = _document()
    first = campaign_tables.PUBLICATION_TABLE_SECTION_ORDER[0]
    begin, end = campaign_tables.PUBLICATION_TABLE_REGION_MARKERS[first]

    for tampered in (
        document + begin + end,
        document.replace(begin, "", 1),
        document.replace(f"{begin}\nstale\n{end}", f"{end}\nstale\n{begin}"),
    ):
        with pytest.raises(ValueError, match="marker"):
            campaign_tables.replace_vllm_0271_publication_table_regions(
                tampered,
                report,
                gate,
            )
    for noncanonical in (
        document.replace("\n", "\r\n"),
        document + "\x00",
    ):
        with pytest.raises(ValueError, match="canonical LF text"):
            campaign_tables.replace_vllm_0271_publication_table_regions(
                noncanonical,
                report,
                gate,
            )


def test_quality_formatting_normalizes_zero_and_preserves_small_effects() -> None:
    report = _report()
    paired = report["quality"]["datasets"]["biography"][
        "paired_vanilla_minus_baseline"
    ]["exact_match"]
    paired["mean"] = -0.0
    paired["bootstrap_ci95"]["lower"] = -0.0
    vanilla = report["quality"]["datasets"]["biography"]["methods"][
        "vanilla_prefill"
    ]["metrics"]["exact_match"]
    vanilla["mean"] = 0.4
    vanilla["sum"] = 4.0

    regions = campaign_tables.render_vllm_0271_publication_table_regions(
        report,
        {"synthetic": True},
    )

    assert "-0.000000" not in regions["dataset_scores"]
    assert "0.000000" in regions["dataset_scores"]

    smallest_biography_delta = 1 / 72_831
    for direction, expected in ((1, "0.000014"), (-1, "-0.000014")):
        delta = direction * smallest_biography_delta
        paired["mean"] = delta
        paired["bootstrap_ci95"]["lower"] = -smallest_biography_delta
        paired["bootstrap_ci95"]["upper"] = smallest_biography_delta
        vanilla["mean"] = 0.4 + delta
        vanilla["sum"] = vanilla["mean"] * 10

        rendered = campaign_tables.render_vllm_0271_publication_table_regions(
            report,
            {"synthetic": True},
        )["dataset_scores"]
        biography_row = next(
            line for line in rendered.splitlines() if line.startswith("| Biography |")
        )
        assert f"| {expected} | [-0.000014, 0.000014] |" in biography_row
