from __future__ import annotations

import json
import shutil
import stat
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import document_kv_cache.publication_campaign_finalizer as campaign_finalizer
import document_kv_cache.publication_campaign_tables as campaign_tables
from document_kv_cache.benchmark_gates import (
    BENCHMARK_PUBLICATION_GATE_RECORD_TYPE,
)
from document_kv_cache.benchmarks import NIAH_CELL_IDS, SUPPORTED_V1_DATASETS


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _closed_record(record: dict[str, Any]) -> dict[str, Any]:
    closed = deepcopy(record)
    closed.pop("closed_record_sha256", None)
    digest = _canonical_sha256(closed)
    closed["closed_record_sha256"] = digest
    return closed


def _reclose(record: dict[str, Any]) -> None:
    record.pop("closed_record_sha256", None)
    record["closed_record_sha256"] = _canonical_sha256(record)


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _source_record(
    record_type: str,
    schema_version: int,
    *,
    label: str,
    **fields: Any,
) -> dict[str, Any]:
    return _closed_record(
        {
            "label": label,
            "record_type": record_type,
            "schema_version": schema_version,
            **fields,
        }
    )


def _ledger_prefix(*, count: int, label: str) -> dict[str, Any]:
    return {
        "cap_cluster_hours": 1024.0,
        "ledger_id": "synthetic-publication-ledger",
        "prefix_sha256": _digest(label),
        "reservation_count": count,
        "submission_receipt_count": count,
        "terminal_actual_count": count,
    }


def _latency_closed_record(record: dict[str, Any]) -> dict[str, Any]:
    closed = deepcopy(record)
    closed["closed_record_sha256"] = ""
    closed["closed_record_sha256"] = _canonical_sha256(closed)
    return closed


def _synthetic_cache_telemetry(
    *,
    method_id: str,
    observation_count: int,
    setting_id: str | None,
) -> dict[str, int]:
    telemetry = {
        "backend_bytes_read": 0,
        "cold_read_attested_count": 0,
        "eviction_requested_count": 0,
        "eviction_succeeded_count": 0,
        "expected_backend_bytes_read": 0,
        "load_count": 0,
        "mounted_path_load_count": 0,
        "payload_cache_hit_count": 0,
        "payload_cache_miss_count": 0,
        "storage_materialization_count": 0,
    }
    if method_id == "baseline_prefill":
        return telemetry
    telemetry["load_count"] = observation_count
    if setting_id == "storage-ram":
        telemetry["payload_cache_hit_count"] = observation_count
        return telemetry
    telemetry.update(
        {
            "backend_bytes_read": observation_count * 10,
            "cold_read_attested_count": (
                0 if setting_id == "storage-uc" else observation_count
            ),
            "eviction_requested_count": observation_count,
            "eviction_succeeded_count": observation_count,
            "expected_backend_bytes_read": observation_count * 10,
            "mounted_path_load_count": (
                observation_count if setting_id == "storage-uc" else 0
            ),
        }
    )
    return telemetry


def _synthetic_latency_descriptive_cells() -> list[dict[str, Any]]:
    coordinates: list[tuple[str, str, str | None, int, str, int, str | None]] = []
    for input_tokens in (8_192, 16_384, 32_768):
        for concurrency in (1, 2, 4):
            for method_id in ("baseline_prefill", "vanilla_prefill"):
                coordinates.append(
                    (
                        f"core-{method_id}-{input_tokens}-c{concurrency}",
                        "core_pooled_five_blocks",
                        None,
                        input_tokens,
                        method_id,
                        concurrency,
                        None,
                    )
                )
    for setting_id, family in (
        ("precision-bf16", "precision"),
        ("storage-disk", "storage"),
        ("storage-ram", "storage"),
        ("storage-uc", "storage"),
        ("hardware-a10g", "hardware"),
    ):
        coordinates.append(
            (
                f"auxiliary-{setting_id}",
                "auxiliary_pooled_five_blocks",
                family,
                16_384,
                "vanilla_prefill",
                4,
                setting_id,
            )
        )

    cells: list[dict[str, Any]] = []
    for (
        cell_id,
        cell_kind,
        comparison_family,
        input_tokens,
        method_id,
        concurrency,
        setting_id,
    ) in coordinates:
        blocks = []
        for deployment_block in range(1, 6):
            if cell_kind == "core_pooled_five_blocks":
                method_label = (
                    "baseline"
                    if method_id == "baseline_prefill"
                    else "vanilla"
                )
                job_id = (
                    f"block-{deployment_block:02d}-{input_tokens // 1024}k-"
                    f"c{concurrency}-{method_label}"
                )
            else:
                job_id = f"block-{deployment_block:02d}-{setting_id}"
            blocks.append(
                {
                    "cache_telemetry": _synthetic_cache_telemetry(
                        method_id=method_id,
                        observation_count=256,
                        setting_id=setting_id,
                    ),
                    "configured_closed_loop_concurrency": concurrency,
                    "deployment_block": deployment_block,
                    "gpu_utilization_sample_count": 10,
                    "job_id": job_id,
                    "mean_gpu_utilization_percent": 50.0,
                    "observation_count": 256,
                    "p50_decode_tokens_per_second": 100.0,
                    "p50_time_to_completion_seconds": 2.0,
                    "p50_ttft_seconds": 1.0,
                    "p95_time_to_completion_seconds": 3.0,
                    "p95_ttft_seconds": 1.5,
                    "peak_gpu_process_memory_bytes": 1_000 + deployment_block,
                    "peak_gpu_utilization_percent": 60.0,
                    "peak_host_memory_used_bytes": 2_000 + deployment_block,
                    "peak_process_tree_rss_bytes": 3_000 + deployment_block,
                }
            )
        pooled_telemetry = {
            key: sum(block["cache_telemetry"][key] for block in blocks)
            for key in blocks[0]["cache_telemetry"]
        }
        cell = {
            "cache_telemetry": pooled_telemetry,
            "cell_id": cell_id,
            "cell_kind": cell_kind,
            "cell_sha256": "",
            "comparison_family": comparison_family,
            "configured_closed_loop_concurrency": concurrency,
            "gpu_utilization_sample_count": 50,
            "input_tokens": input_tokens,
            "mean_gpu_utilization_percent": 50.0,
            "method_id": method_id,
            "observation_count": 1_280,
            "p50_decode_tokens_per_second": 100.0,
            "p50_time_to_completion_seconds": 2.0,
            "p50_ttft_seconds": 1.0,
            "p95_time_to_completion_seconds": 3.0,
            "p95_ttft_seconds": 1.5,
            "peak_gpu_process_memory_bytes": 1_005,
            "peak_gpu_utilization_percent": 60.0,
            "peak_host_memory_used_bytes": 2_005,
            "peak_process_tree_rss_bytes": 3_005,
            "physical_blocks": blocks,
            "quantile_method": "empirical_nearest_rank",
            "request_parallelism": concurrency,
            "setting_id": setting_id,
        }
        cell["cell_sha256"] = _canonical_sha256(cell)
        cells.append(cell)
    return cells


def _synthetic_latency_estimates() -> list[dict[str, Any]]:
    design: list[dict[str, Any]] = []
    for input_tokens in (8_192, 16_384, 32_768):
        for concurrency in (1, 2, 4):
            design.append(
                {
                    "comparison_family": "method",
                    "control_cell_id": (
                        f"core-baseline_prefill-{input_tokens}-c{concurrency}"
                    ),
                    "deployment_block_count": 5,
                    "estimand_id": f"method-{input_tokens}-c{concurrency}",
                    "example_count_per_block": 128,
                    "input_tokens": input_tokens,
                    "paired_request_count": 1_280,
                    "request_parallelism": concurrency,
                    "setting_id": None,
                    "speedup_direction": (
                        "control_latency_divided_by_treatment_latency"
                    ),
                    "treatment_cell_id": (
                        f"core-vanilla_prefill-{input_tokens}-c{concurrency}"
                    ),
                }
            )
    for setting_id, family in (
        ("precision-bf16", "precision"),
        ("storage-ram", "storage"),
        ("storage-uc", "storage"),
        ("hardware-a10g", "hardware"),
    ):
        storage = setting_id in {"storage-ram", "storage-uc"}
        design.append(
            {
                "comparison_family": family,
                "control_cell_id": (
                    "auxiliary-storage-disk"
                    if storage
                    else "core-vanilla_prefill-16384-c4"
                ),
                "deployment_block_count": 5,
                "estimand_id": f"auxiliary-{setting_id}",
                "example_count_per_block": 8 if storage else 128,
                "input_tokens": 16_384,
                "paired_request_count": 1_280,
                "request_parallelism": 4,
                "setting_id": setting_id,
                "speedup_direction": (
                    "control_latency_divided_by_treatment_latency"
                ),
                "treatment_cell_id": f"auxiliary-{setting_id}",
            }
        )
    return [
        {
            **item,
            "metrics": {
                metric: {
                    "confidence_interval_95": [0.9, 1.1],
                    "geometric_mean_speedup": 1.0,
                }
                for metric in ("ttft", "time_to_completion")
            },
        }
        for item in design
    ]


def _synthetic_latency_summary(
    *,
    collection_sha256: str,
    execution_plan_sha256: str,
) -> dict[str, Any]:
    return _latency_closed_record(
        {
            "analysis": {
                "bootstrap": "paired_hierarchical_deployment_block_and_example",
                "bootstrap_draws": 20_000,
                "confidence_intervals": "pointwise_95_percent",
                "decision_mode": "estimation_only",
                "null_hypothesis_rejections": False,
                "post_hoc_significance_claims": False,
                "descriptive_quantiles": "empirical_nearest_rank",
                "storage_workload": "2_examples_per_dataset_x_32_repeats",
            },
            "campaign_id": campaign_finalizer.PUBLICATION_CAMPAIGN_ID,
            "closed_record_sha256": "",
            "collection_sha256": collection_sha256,
            "descriptive_cell_count": 23,
            "descriptive_cells": _synthetic_latency_descriptive_cells(),
            "estimand_count": 13,
            "estimates": _synthetic_latency_estimates(),
            "execution_plan_sha256": execution_plan_sha256,
            "record_type": campaign_finalizer.PUBLICATION_LATENCY_SUMMARY_RECORD_TYPE,
            "schema_version": campaign_finalizer.PUBLICATION_LATENCY_SCHEMA_VERSION,
        }
    )


def _synthetic_metric_summary(*, count: int, mean: float) -> dict[str, Any]:
    return {
        "example_count": count,
        "invalid_parser_score_sum": 0.0,
        "mean": mean,
        "sum": mean * count,
    }


def _synthetic_paired_summary(
    *,
    count: int,
    dataset_stratum: str,
    inventory_sha256: str,
    metric: str,
    shard_plan_sha256: str,
) -> dict[str, Any]:
    delta = 0.5 - 0.4
    return {
        "bootstrap_ci95": {
            "draws": 20_000,
            "lower": delta - 0.01,
            "upper": delta + 0.01,
        },
        "example_count": count,
        "mean": delta,
        "seed_sha256": _canonical_sha256(
            {
                "dataset_stratum": dataset_stratum,
                "domain": "cachet.full_score.paired_bootstrap.seed.v1",
                "inventory_sha256": inventory_sha256,
                "metric": metric,
                "shard_plan_sha256": shard_plan_sha256,
            }
        ),
    }


def _synthetic_quality_tables(
    *,
    inventory_sha256: str,
    shard_plan_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    dataset_counts = {
        "biography": 72_831,
        "hotpotqa": 7_405,
        "musique": 2_417,
        "niah": 1_000,
    }
    metric_names = {
        scorer["dataset"]: tuple(scorer["metric_names"])
        for scorer in campaign_finalizer._expected_scorer_contracts()
    }
    datasets: dict[str, Any] = {}
    for dataset in SUPPORTED_V1_DATASETS:
        count = dataset_counts[dataset]
        datasets[dataset] = {
            "example_count": count,
            "methods": {
                method: {
                    "example_count": count,
                    "metrics": {
                        metric: _synthetic_metric_summary(
                            count=count,
                            mean=0.4 if method == "baseline_prefill" else 0.5,
                        )
                        for metric in metric_names[dataset]
                    },
                    "parser_status_counts": {
                        status: count if status == "ok" else 0
                        for status in campaign_finalizer.FINAL_ANSWER_PARSER_STATUSES
                    },
                }
                for method in campaign_finalizer.FULL_SCORE_METHODS
            },
            "paired_vanilla_minus_baseline": {
                metric: _synthetic_paired_summary(
                    count=count,
                    dataset_stratum=dataset,
                    inventory_sha256=inventory_sha256,
                    metric=metric,
                    shard_plan_sha256=shard_plan_sha256,
                )
                for metric in metric_names[dataset]
            },
        }
    niah_grid: dict[str, Any] = {}
    for index, cell_id in enumerate(NIAH_CELL_IDS):
        count = 112 if index == 0 else 111
        niah_grid[cell_id] = {
            "example_count": count,
            "methods": {
                method: {
                    metric: _synthetic_metric_summary(
                        count=count,
                        mean=0.4 if method == "baseline_prefill" else 0.5,
                    )
                    for metric in metric_names["niah"]
                }
                for method in campaign_finalizer.FULL_SCORE_METHODS
            },
            "paired_vanilla_minus_baseline": {
                metric: _synthetic_paired_summary(
                    count=count,
                    dataset_stratum=f"niah/{cell_id}",
                    inventory_sha256=inventory_sha256,
                    metric=metric,
                    shard_plan_sha256=shard_plan_sha256,
                )
                for metric in metric_names["niah"]
            },
        }
    return datasets, niah_grid, campaign_finalizer._expected_scorer_contracts()


class _SyntheticLedgerPrefix:
    def __init__(
        self,
        *,
        ledger_id: str,
        count: int,
        prefix_sha256: str,
    ) -> None:
        self.cap_cluster_hours = 1024.0
        self.ledger_id = ledger_id
        self.prefix_sha256 = prefix_sha256
        self.reservation_count = count
        self.submission_receipt_count = count
        self.terminal_actual_count = count

    def to_record(self) -> dict[str, Any]:
        return {
            "cap_cluster_hours": self.cap_cluster_hours,
            "ledger_id": self.ledger_id,
            "prefix_sha256": self.prefix_sha256,
            "reservation_count": self.reservation_count,
            "submission_receipt_count": self.submission_receipt_count,
            "terminal_actual_count": self.terminal_actual_count,
        }

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _SyntheticLedgerPrefix)
            and self.to_record() == other.to_record()
        )


def _synthetic_campaign_ledger_projection_case(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    ledger_id = "synthetic-publication-ledger"
    ledger_path = Path("synthetic-campaign-ledger-never-read.json")
    ledger_path_sha256 = _digest("synthetic-campaign-ledger-path")
    execution_plan_sha256 = _digest("full-score-execution-plan")
    latency_count = 115
    phase_count = campaign_finalizer.PUBLICATION_CAMPAIGN_FULL_SCORE_PHASES
    final_count = latency_count + phase_count
    latency_prefix = _SyntheticLedgerPrefix(
        ledger_id=ledger_id,
        count=latency_count,
        prefix_sha256=_digest("latency-terminal-prefix"),
    )
    final_prefix = _SyntheticLedgerPrefix(
        ledger_id=ledger_id,
        count=final_count,
        prefix_sha256=_digest("full-score-terminal-prefix"),
    )
    expected_workloads = [
        f"full-score:{execution_plan_sha256}:wave-{wave_index:03d}:{phase}"
        for wave_index in range(
            campaign_finalizer.PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES
        )
        for phase in ("producer", "consumer")
    ]
    latency_attempts = [
        f"latency-attempt-{index:03d}" for index in range(latency_count)
    ]
    phase_attempts = [f"full-score-attempt-{index:03d}" for index in range(phase_count)]
    reservations = [
        SimpleNamespace(
            attempt_id=attempt_id,
            workload_id=f"latency:job-{index:03d}",
        )
        for index, attempt_id in enumerate(latency_attempts)
    ] + [
        SimpleNamespace(attempt_id=attempt_id, workload_id=workload_id)
        for attempt_id, workload_id in zip(
            phase_attempts,
            expected_workloads,
            strict=True,
        )
    ]
    submission_receipts = [
        SimpleNamespace(attempt_id=attempt_id)
        for attempt_id in latency_attempts + phase_attempts
    ]
    terminal_actuals = [
        SimpleNamespace(attempt_id=attempt_id)
        for attempt_id in latency_attempts + phase_attempts
    ]
    live = SimpleNamespace(
        accounted_cluster_hours=900.0,
        active_reserved_cluster_hours=0.0,
        active_reserved_task_count=0,
        cap_cluster_hours=1024.0,
        ledger_id=ledger_id,
        remaining_cluster_hours=(
            campaign_finalizer.DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS
        ),
        reservations=reservations,
        submission_receipts=submission_receipts,
        terminal_actuals=terminal_actuals,
    )

    class FakeFullScorePhaseAuthorization:
        def __init__(self) -> None:
            self.execution_plan_sha256 = execution_plan_sha256
            self.ledger_path_sha256 = ledger_path_sha256
            self.ledger_prefix = final_prefix
            self.phase = "consumer"
            self.wave_index = (
                campaign_finalizer.PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES - 1
            )
            self.workspace_host_sha256 = _digest("workspace-host")
            self.user_name_sha256 = _digest("workspace-user")
            self.causal_closure_sha256 = _digest("final-consumer-causal")
            self.terminal_record_sha256 = _digest("final-consumer-terminal")

    latency_authorization = SimpleNamespace(
        ledger_path_sha256=ledger_path_sha256,
        ledger_prefix=latency_prefix,
        workspace_host_sha256=_digest("workspace-host"),
        user_name_sha256=_digest("workspace-user"),
    )
    final_consumer_authorization = FakeFullScorePhaseAuthorization()
    execution_plan = {"closed_record_sha256": execution_plan_sha256}
    aggregate = {
        "publication_lineage": {
            "authorization_sha256": final_consumer_authorization.causal_closure_sha256,
            "ledger_id": ledger_id,
            "ledger_path_sha256": ledger_path_sha256,
            "terminal_record_sha256": (
                final_consumer_authorization.terminal_record_sha256
            ),
            "terminal_prefix": final_prefix.to_record(),
        }
    }
    calls: dict[str, int] = {
        "current_prefix": 0,
        "historical_prefix": 0,
        "path_hash": 0,
        "prefix_from_record": 0,
        "read": 0,
    }

    def path_hash(observed_path: str | Path) -> str:
        assert observed_path == ledger_path
        calls["path_hash"] += 1
        return ledger_path_sha256

    def read_ledger(observed_path: str | Path) -> Any:
        assert observed_path == ledger_path
        calls["read"] += 1
        return live

    def historical_prefix(
        observed_live: Any,
        *,
        reservation_count: int,
        submission_receipt_count: int,
        terminal_actual_count: int,
    ) -> _SyntheticLedgerPrefix:
        assert observed_live is live
        assert (
            reservation_count,
            submission_receipt_count,
            terminal_actual_count,
        ) == (latency_count, latency_count, latency_count)
        calls["historical_prefix"] += 1
        return latency_prefix

    def current_prefix(observed_live: Any) -> _SyntheticLedgerPrefix:
        assert observed_live is live
        calls["current_prefix"] += 1
        return final_prefix

    def prefix_from_record(
        record: dict[str, Any],
    ) -> _SyntheticLedgerPrefix:
        assert record == final_prefix.to_record()
        calls["prefix_from_record"] += 1
        return final_prefix

    monkeypatch.setattr(
        campaign_finalizer,
        "FullScorePhaseAuthorization",
        FakeFullScorePhaseAuthorization,
    )
    monkeypatch.setattr(
        campaign_finalizer,
        "require_publication_latency_collection_authorization",
        lambda *_args, **_kwargs: latency_prefix,
    )
    monkeypatch.setattr(
        campaign_finalizer,
        "databricks_ledger_path_sha256",
        path_hash,
    )
    monkeypatch.setattr(
        campaign_finalizer,
        "read_databricks_cluster_hour_ledger_json",
        read_ledger,
    )
    monkeypatch.setattr(
        campaign_finalizer,
        "databricks_ledger_prefix_at_counts",
        historical_prefix,
    )
    monkeypatch.setattr(
        campaign_finalizer,
        "databricks_ledger_prefix",
        current_prefix,
    )
    monkeypatch.setattr(
        campaign_finalizer,
        "databricks_ledger_prefix_from_record",
        prefix_from_record,
    )
    return {
        "calls": calls,
        "expected_workloads": expected_workloads,
        "final_prefix": final_prefix,
        "latency_prefix": latency_prefix,
        "live": live,
        "projection_kwargs": {
            "latency_collection_authorization": latency_authorization,
            "latency_execution_plan_record": {
                "closed_record_sha256": _digest("latency-execution-plan")
            },
            "full_score_execution_plan_record": execution_plan,
            "full_score_aggregate_record": aggregate,
            "final_consumer_authorization": final_consumer_authorization,
            "ledger_path": ledger_path,
        },
    }


def _synthetic_report_inputs() -> dict[str, Any]:
    jobs = [
        {
            "job_id": f"latency-job-{index:03d}",
            "method_id": (
                "vanilla_prefill" if index < 70 else "baseline_prefill"
            ),
            "request_count": 256,
        }
        for index in range(115)
    ]
    results = [
        {
            "cache_telemetry": {
                "cold_read_attested_count": 256 if index < 60 else 0,
            },
            "job_id": job["job_id"],
        }
        for index, job in enumerate(jobs)
    ]
    latency_execution_plan = _source_record(
        campaign_finalizer.PUBLICATION_LATENCY_EXECUTION_PLAN_RECORD_TYPE,
        campaign_finalizer.PUBLICATION_LATENCY_SCHEMA_VERSION,
        label="latency-execution-plan",
        jobs=jobs,
        sources={
            "campaign": {
                "closed_record_sha256": (
                    campaign_finalizer.PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256
                ),
            }
        },
    )
    latency_collection = _source_record(
        campaign_finalizer.PUBLICATION_LATENCY_COLLECTION_RECORD_TYPE,
        campaign_finalizer.PUBLICATION_LATENCY_SCHEMA_VERSION,
        label="latency-collection",
        results=results,
    )
    latency_summary = _synthetic_latency_summary(
        collection_sha256=latency_collection["closed_record_sha256"],
        execution_plan_sha256=latency_execution_plan["closed_record_sha256"],
    )
    inventory = _source_record(
        campaign_finalizer.FULL_SCORE_INVENTORY_RECORD_TYPE,
        campaign_finalizer.FULL_SCORE_INVENTORY_SCHEMA_VERSION,
        label="full-score-inventory",
    )
    inventory["closed_record_sha256"] = (
        campaign_finalizer.FULL_SCORE_PUBLICATION_INVENTORY_SHA256
    )
    shard_plan = _source_record(
        campaign_finalizer.FULL_SCORE_SHARD_PLAN_RECORD_TYPE,
        campaign_finalizer.FULL_SCORE_SHARD_PLAN_SCHEMA_VERSION,
        label="full-score-shard-plan",
    )
    shard_plan["closed_record_sha256"] = (
        campaign_finalizer.FULL_SCORE_PUBLICATION_SHARD_PLAN_SHA256
    )
    execution_plan = _source_record(
        campaign_finalizer.FULL_SCORE_EXECUTION_PLAN_RECORD_TYPE,
        campaign_finalizer.FULL_SCORE_EXECUTION_PLAN_SCHEMA_VERSION,
        label="full-score-execution-plan",
    )
    execution_plan["closed_record_sha256"] = (
        campaign_finalizer.FULL_SCORE_PUBLICATION_EXECUTION_PLAN_SHA256
    )
    datasets, niah_grid, scorers = _synthetic_quality_tables(
        inventory_sha256=inventory["closed_record_sha256"],
        shard_plan_sha256=shard_plan["closed_record_sha256"],
    )
    aggregate = _source_record(
        campaign_finalizer.FULL_SCORE_AGGREGATE_RECORD_TYPE,
        campaign_finalizer.FULL_SCORE_AGGREGATE_SCHEMA_VERSION,
        label="full-score-aggregate",
        aggregation_unit="per_example_once_never_shard_means",
        authorization_scope=(
            campaign_finalizer.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
        ),
        bootstrap=campaign_finalizer._expected_full_score_bootstrap(),
        datasets=datasets,
        identity_count=83_653,
        methods=list(campaign_finalizer.FULL_SCORE_METHODS),
        niah_grid=niah_grid,
        passes_per_method=campaign_finalizer.FULL_SCORE_PASSES_PER_METHOD,
        protocol=campaign_finalizer._expected_full_score_protocol(),
        scorers=scorers,
        shard_count=160,
    )
    latency_prefix = _ledger_prefix(count=115, label="latency-prefix")
    full_score_prefix = _ledger_prefix(count=135, label="full-score-prefix")
    ledger = {
        "final_accounted_cluster_hours": 824.0,
        "final_active_reserved_cluster_hours": 0.0,
        "final_active_reserved_task_count": 0,
        "final_remaining_cluster_hours": 200.0,
        "full_score_terminal_prefix": full_score_prefix,
        "latency_terminal_prefix": latency_prefix,
        "ledger_id": "synthetic-publication-ledger",
        "ledger_path_sha256": _digest("ledger-path"),
        "required_unreserved_headroom_cluster_hours": 124.0,
    }
    return {
        "full_score_aggregate_record": aggregate,
        "full_score_execution_plan_record": execution_plan,
        "full_score_shard_plan_record": shard_plan,
        "inventory_record": inventory,
        "latency_collection_record": latency_collection,
        "latency_execution_plan_record": latency_execution_plan,
        "latency_summary_record": latency_summary,
        "ledger_record": ledger,
    }


def _synthetic_report() -> dict[str, Any]:
    return campaign_finalizer._build_report(**_synthetic_report_inputs())


def _rebind_latency_summary_projection(report: dict[str, Any]) -> None:
    latency = report["latency"]
    bindings = report["source_bindings"]
    summary = _latency_closed_record(
        {
            "analysis": deepcopy(latency["analysis"]),
            "campaign_id": campaign_finalizer.PUBLICATION_CAMPAIGN_ID,
            "closed_record_sha256": "",
            "collection_sha256": bindings["latency_collection"][
                "closed_record_sha256"
            ],
            "descriptive_cell_count": 23,
            "descriptive_cells": deepcopy(latency["descriptive_cells"]),
            "estimand_count": 13,
            "estimates": deepcopy(latency["estimates"]),
            "execution_plan_sha256": bindings["latency_execution_plan"][
                "closed_record_sha256"
            ],
            "record_type": (
                campaign_finalizer.PUBLICATION_LATENCY_SUMMARY_RECORD_TYPE
            ),
            "schema_version": campaign_finalizer.PUBLICATION_LATENCY_SCHEMA_VERSION,
        }
    )
    bindings["latency_summary"]["closed_record_sha256"] = summary[
        "closed_record_sha256"
    ]
    _reclose(report)


def _tamper_nested_report_projection(
    report: dict[str, Any],
    tamper: str,
) -> None:
    if tamper == "latency-estimate-control-cell":
        report["latency"]["estimates"][0]["control_cell_id"] = (
            "core-vanilla_prefill-8192-c1"
        )
        _rebind_latency_summary_projection(report)
        return
    if tamper == "latency-physical-block-schema":
        cell = report["latency"]["descriptive_cells"][0]
        cell["physical_blocks"][0]["unexpected"] = True
        cell["cell_sha256"] = ""
        cell["cell_sha256"] = _canonical_sha256(cell)
        _rebind_latency_summary_projection(report)
        return
    if tamper == "bootstrap-type":
        report["quality"]["bootstrap"]["draws"] = 20_000.0
    elif tamper == "protocol-lifecycle":
        report["quality"]["protocol"]["lifecycle"][0:2] = [
            "baseline_inference",
            "generate_q8_kv",
        ]
    elif tamper == "scorer-metric-identity":
        report["scorer_contracts"][0]["metric_names"].append(
            "unapproved_metric"
        )
    elif tamper == "dataset-method-schema":
        report["quality"]["datasets"]["biography"]["methods"][
            "baseline_prefill"
        ]["unexpected"] = True
    elif tamper == "dataset-metric-algebra":
        metric = next(
            iter(
                report["quality"]["datasets"]["biography"]["methods"][
                    "baseline_prefill"
                ]["metrics"]
            )
        )
        report["quality"]["datasets"]["biography"]["methods"][
            "baseline_prefill"
        ]["metrics"][metric]["mean"] = 0.41
    elif tamper == "parser-coverage":
        report["quality"]["datasets"]["hotpotqa"]["methods"][
            "vanilla_prefill"
        ]["parser_status_counts"]["ok"] -= 1
    elif tamper == "paired-ci-schema":
        metric = next(
            iter(
                report["quality"]["datasets"]["musique"][
                    "paired_vanilla_minus_baseline"
                ]
            )
        )
        report["quality"]["datasets"]["musique"][
            "paired_vanilla_minus_baseline"
        ][metric]["bootstrap_ci95"]["unexpected"] = True
    elif tamper == "paired-seed":
        metric = next(
            iter(
                report["quality"]["datasets"]["niah"][
                    "paired_vanilla_minus_baseline"
                ]
            )
        )
        report["quality"]["datasets"]["niah"][
            "paired_vanilla_minus_baseline"
        ][metric]["seed_sha256"] = "0" * 64
    elif tamper == "niah-cell-count":
        report["quality"]["niah_grid"][NIAH_CELL_IDS[0]][
            "example_count"
        ] += 1
    elif tamper == "niah-rollup-algebra":
        cell = report["quality"]["niah_grid"][NIAH_CELL_IDS[0]]
        metric = next(iter(cell["methods"]["baseline_prefill"]))
        count = cell["example_count"]
        for method, mean in (
            ("baseline_prefill", 0.3),
            ("vanilla_prefill", 0.4),
        ):
            summary = cell["methods"][method][metric]
            summary["mean"] = mean
            summary["sum"] = mean * count
    else:  # pragma: no cover - the parameter table below is closed.
        raise AssertionError(f"unknown nested report tamper: {tamper}")
    _reclose(report)


def _issued_finalization(
    report: dict[str, Any], gate: dict[str, Any]
) -> campaign_finalizer.PublicationCampaignFinalization:
    return campaign_finalizer.PublicationCampaignFinalization(
        report_record=report,
        gate_record=gate,
        _issuer=campaign_finalizer._FINALIZATION_ISSUER,
    )


def _load_with_synthetic_authority_placeholders(
    directory: Path,
) -> campaign_finalizer.PublicationCampaignFinalization:
    return campaign_finalizer.load_vllm_0271_publication_finalization(
        directory,
        latency_execution_plan_record=None,
        latency_collection_authorization=None,
        latency_summary_record=None,
        qualification_launch_authorization=None,
        handoff_serving_authorization=None,
        bf16_handoff_serving_authorization=None,
        source_closure_authorization=None,
        full_score_inventory=None,
        full_score_shard_plan_record=None,
        full_score_execution_plan_record=None,
        full_score_aggregate_record=None,
        final_consumer_authorization=None,
        remote_consumer_authorizations=(),
        compact_artifact_resolver=lambda _uri: directory,
        ledger_path=directory / "ledger.json",
    )


def test_finalization_returns_detached_nested_report_projections(
    tmp_path: Path,
) -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)
    finalization = _issued_finalization(report, gate)

    first_projection = finalization.report_record
    first_projection["latency"]["analysis"]["bootstrap_draws"] = 1
    first_projection["source_bindings"]["latency_summary"][
        "closed_record_sha256"
    ] = "0" * 64

    second_projection = finalization.report_record
    assert second_projection is not first_projection
    assert second_projection["latency"]["analysis"]["bootstrap_draws"] == 20_000
    assert second_projection["source_bindings"]["latency_summary"][
        "closed_record_sha256"
    ] == report["source_bindings"]["latency_summary"]["closed_record_sha256"]
    assert second_projection["closed_record_sha256"] == report[
        "closed_record_sha256"
    ]
    assert finalization.gate_record["benchmark_payload_digest"] == report[
        "closed_record_sha256"
    ]
    rendered = campaign_tables.render_vllm_0271_publication_table_regions(
        report,
        gate,
    )
    assert tuple(rendered) == campaign_tables.PUBLICATION_TABLE_SECTION_ORDER
    assert "| Dataset | Governed metric | n |" in rendered["dataset_scores"]
    appendix = campaign_tables.render_vllm_0271_publication_appendix_readme(
        report,
        gate,
    )
    assert report["closed_record_sha256"] in appendix

    repository_root = Path(__file__).resolve().parents[1]
    benchmark_root = tmp_path / "benchmarks"
    shutil.copytree(repository_root / "benchmarks", benchmark_root)
    root_readme_path = benchmark_root / "README.md"
    pending_readme = root_readme_path.read_bytes().decode("utf-8")
    published_readme = campaign_tables.replace_vllm_0271_publication_table_regions(
        pending_readme,
        report,
        gate,
    )
    root_readme_path.write_bytes(published_readme.encode("utf-8"))
    campaign_dir = benchmark_root / "appendix" / "vllm-0271-publication-v1"
    campaign_dir.mkdir()

    def canonical_pretty_json(value: dict[str, Any]) -> bytes:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

    (campaign_dir / "campaign-report.json").write_bytes(
        canonical_pretty_json(report)
    )
    (campaign_dir / "benchmark-publication-gate.json").write_bytes(
        canonical_pretty_json(gate)
    )
    (campaign_dir / "README.md").write_bytes(appendix.encode("utf-8"))

    campaign_finalizer.validate_vllm_0271_publication_report_pair(report, gate)
    campaign_tables.validate_vllm_0271_publication_table_regions(
        root_readme_path.read_bytes().decode("utf-8"),
        report,
        gate,
    )
    assert (campaign_dir / "README.md").read_bytes() == appendix.encode("utf-8")
    all_entries = list(benchmark_root.rglob("*"))
    assert not any(path.is_symlink() for path in all_entries)
    assert {
        str(path.relative_to(benchmark_root))
        for path in all_entries
        if path.is_file()
    } == {
        "README.md",
        "_template/README.md",
        "appendix/README.md",
        "appendix/vllm-0271-publication-v1/README.md",
        "appendix/vllm-0271-publication-v1/benchmark-publication-gate.json",
        "appendix/vllm-0271-publication-v1/campaign-report.json",
        "databricks/README.md",
        "native-engine/README.md",
        "sglang/README.md",
        "storage/README.md",
        "vllm/README.md",
    }
    published_markdown = "\n".join(
        path.read_bytes().decode("utf-8")
        for path in sorted(benchmark_root.rglob("*.md"))
    )
    assert "N/A (0.27.1 campaign pending)" not in published_markdown
    assert "N/A (0.27.1 full evaluation pending)" not in published_markdown
    for unsupported in (
        "KV&nbsp;Packet",
        "CacheBlend",
        "InfoFlow&nbsp;KV",
        "LongBench v2",
        "RULER",
        "Packed Q4",
        "Hybrid RAM/disk/Unity Catalog",
        "SGLang",
    ):
        assert unsupported in published_readme


def test_json_type_exact_equality_rejects_python_numeric_coercions() -> None:
    assert campaign_finalizer._json_type_exact_equal(
        {"publication_approved": True},
        {"publication_approved": True},
    )
    assert not campaign_finalizer._json_type_exact_equal(
        {"publication_approved": True},
        {"publication_approved": 1},
    )
    assert not campaign_finalizer._json_type_exact_equal(
        {"example_count": 1},
        {"example_count": 1.0},
    )


def test_campaign_ledger_projection_accepts_exact_twenty_phase_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _synthetic_campaign_ledger_projection_case(monkeypatch)

    projection = campaign_finalizer._validated_campaign_ledger_projection(
        **case["projection_kwargs"]
    )

    assert len(case["expected_workloads"]) == 20
    assert projection == {
        "final_accounted_cluster_hours": 900.0,
        "final_active_reserved_cluster_hours": 0.0,
        "final_active_reserved_task_count": 0,
        "final_remaining_cluster_hours": 124.0,
        "full_score_terminal_prefix": case["final_prefix"].to_record(),
        "latency_terminal_prefix": case["latency_prefix"].to_record(),
        "ledger_id": "synthetic-publication-ledger",
        "ledger_path_sha256": _digest("synthetic-campaign-ledger-path"),
        "required_unreserved_headroom_cluster_hours": 124.0,
    }
    assert case["calls"] == {
        "current_prefix": 1,
        "historical_prefix": 1,
        "path_hash": 1,
        "prefix_from_record": 1,
        "read": 1,
    }


def test_campaign_ledger_projection_rejects_final_authority_subclasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _synthetic_campaign_ledger_projection_case(monkeypatch)
    authority_type = campaign_finalizer.FullScorePhaseAuthorization

    class SmuggledFinalAuthority(authority_type):
        pass

    case["projection_kwargs"]["final_consumer_authorization"] = (
        SmuggledFinalAuthority()
    )
    with pytest.raises(TypeError, match="wrong authority type"):
        campaign_finalizer._validated_campaign_ledger_projection(
            **case["projection_kwargs"]
        )


@pytest.mark.parametrize(
    "field_name",
    ["authorization_sha256", "terminal_record_sha256"],
)
def test_campaign_ledger_projection_binds_final_authority_lineage(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    case = _synthetic_campaign_ledger_projection_case(monkeypatch)
    case["projection_kwargs"]["full_score_aggregate_record"][
        "publication_lineage"
    ][field_name] = _digest(f"tampered-{field_name}")
    with pytest.raises(ValueError, match="terminal prefix"):
        campaign_finalizer._validated_campaign_ledger_projection(
            **case["projection_kwargs"]
        )


@pytest.mark.parametrize("tamper", ["unrelated", "reordered"])
def test_campaign_ledger_projection_rejects_invalid_phase_suffix(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    case = _synthetic_campaign_ledger_projection_case(monkeypatch)
    suffix_start = case["latency_prefix"].reservation_count
    reservations = case["live"].reservations
    if tamper == "unrelated":
        reservations[suffix_start + 7].workload_id = "unrelated:append"
    else:
        reservations[suffix_start], reservations[suffix_start + 1] = (
            reservations[suffix_start + 1],
            reservations[suffix_start],
        )

    with pytest.raises(
        ValueError,
        match="full-score ledger suffix contains an unrelated append",
    ):
        campaign_finalizer._validated_campaign_ledger_projection(
            **case["projection_kwargs"]
        )


@pytest.mark.parametrize(
    ("active_hours", "active_tasks", "remaining_hours"),
    [
        pytest.param(0.5, 1, 124.0, id="active-reservation"),
        pytest.param(0.0, 0, 123.999, id="below-headroom"),
    ],
)
def test_campaign_ledger_projection_rejects_active_or_underfunded_ledger(
    monkeypatch: pytest.MonkeyPatch,
    active_hours: float,
    active_tasks: int,
    remaining_hours: float,
) -> None:
    case = _synthetic_campaign_ledger_projection_case(monkeypatch)
    case["live"].active_reserved_cluster_hours = active_hours
    case["live"].active_reserved_task_count = active_tasks
    case["live"].remaining_cluster_hours = remaining_hours

    with pytest.raises(
        ValueError,
        match="publication final ledger is active or below hard headroom",
    ):
        campaign_finalizer._validated_campaign_ledger_projection(
            **case["projection_kwargs"]
        )


def test_finalizer_replays_both_authorities_before_ledger_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_report_inputs()
    authoritative_latency = deepcopy(inputs["latency_summary_record"])
    authoritative_full_score = deepcopy(inputs["full_score_aggregate_record"])
    latency_prefix_record = deepcopy(
        inputs["ledger_record"]["latency_terminal_prefix"]
    )
    latency_collection = deepcopy(inputs["latency_collection_record"])
    latency_collection["ledger"] = {
        "ledger_path_sha256": inputs["ledger_record"]["ledger_path_sha256"],
        "ledger_prefix": latency_prefix_record,
    }
    _reclose(latency_collection)

    class FakeLedgerPrefix:
        def to_record(self) -> dict[str, Any]:
            return deepcopy(latency_prefix_record)

    class FakeLatencyCollectionAuthorization:
        def __init__(self) -> None:
            self.collection = latency_collection
            self.collection_sha256 = latency_collection[
                "closed_record_sha256"
            ]
            self.ledger_path_sha256 = inputs["ledger_record"][
                "ledger_path_sha256"
            ]
            self.ledger_prefix = FakeLedgerPrefix()

    class FakeFullScoreInventory:
        pass

    latency_authorization = FakeLatencyCollectionAuthorization()
    inventory = FakeFullScoreInventory()
    replayed: dict[str, dict[str, Any]] = {}
    ledger_calls: list[dict[str, Any]] = []
    build_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        campaign_finalizer,
        "PublicationLatencyCollectionAuthorization",
        FakeLatencyCollectionAuthorization,
    )
    monkeypatch.setattr(
        campaign_finalizer,
        "FullScoreInventory",
        FakeFullScoreInventory,
    )
    for validator_name in (
        "validate_publication_latency_execution_plan_record",
        "validate_publication_latency_collection_record",
        "validate_publication_latency_summary_record",
        "validate_full_score_inventory_record",
        "validate_full_score_shard_plan",
        "validate_full_score_aggregate_record",
    ):
        monkeypatch.setattr(
            campaign_finalizer,
            validator_name,
            lambda *_args, **_kwargs: None,
        )
    monkeypatch.setattr(
        campaign_finalizer,
        "full_score_inventory_to_record",
        lambda observed: (
            deepcopy(inputs["inventory_record"])
            if observed is inventory
            else pytest.fail("unexpected full-score inventory")
        ),
    )

    def replay_latency(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        record = deepcopy(authoritative_latency)
        replayed["latency"] = record
        return record

    def replay_full_score(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        record = deepcopy(authoritative_full_score)
        replayed["full_score"] = record
        return record

    def project_ledger(**kwargs: Any) -> dict[str, Any]:
        ledger_calls.append(kwargs)
        assert kwargs["full_score_aggregate_record"] is replayed["full_score"]
        return deepcopy(inputs["ledger_record"])

    original_build_report = campaign_finalizer._build_report

    def build_report(**kwargs: Any) -> dict[str, Any]:
        build_calls.append(kwargs)
        assert kwargs["latency_summary_record"] is replayed["latency"]
        assert kwargs["full_score_aggregate_record"] is replayed["full_score"]
        return original_build_report(**kwargs)

    monkeypatch.setattr(
        campaign_finalizer,
        "aggregate_publication_latency_campaign",
        replay_latency,
    )
    monkeypatch.setattr(
        campaign_finalizer,
        "aggregate_full_score_shard_evidence",
        replay_full_score,
    )
    monkeypatch.setattr(
        campaign_finalizer,
        "_validated_campaign_ledger_projection",
        project_ledger,
    )
    monkeypatch.setattr(campaign_finalizer, "_build_report", build_report)

    call_kwargs = {
        "latency_execution_plan_record": inputs[
            "latency_execution_plan_record"
        ],
        "latency_collection_authorization": latency_authorization,
        "latency_summary_record": deepcopy(authoritative_latency),
        "qualification_launch_authorization": object(),
        "handoff_serving_authorization": object(),
        "bf16_handoff_serving_authorization": object(),
        "source_closure_authorization": object(),
        "full_score_inventory": inventory,
        "full_score_shard_plan_record": inputs["full_score_shard_plan_record"],
        "full_score_execution_plan_record": inputs[
            "full_score_execution_plan_record"
        ],
        "full_score_aggregate_record": deepcopy(authoritative_full_score),
        "final_consumer_authorization": object(),
        "remote_consumer_authorizations": (),
        "compact_artifact_resolver": lambda _uri: Path("never-read"),
        "ledger_path": Path("never-read-ledger.json"),
    }

    latency_type_drift = deepcopy(authoritative_latency)
    latency_type_drift["analysis"]["bootstrap_draws"] = 20_000.0
    assert latency_type_drift == authoritative_latency
    with pytest.raises(
        ValueError,
        match="latency summary differs from authoritative campaign reaggregation",
    ):
        campaign_finalizer.finalize_vllm_0271_publication_campaign(
            **{**call_kwargs, "latency_summary_record": latency_type_drift}
        )

    full_score_type_drift = deepcopy(authoritative_full_score)
    full_score_type_drift["scorers"][0]["publication_approved"] = 1
    assert full_score_type_drift == authoritative_full_score
    with pytest.raises(
        ValueError,
        match=(
            "full-score aggregate differs from authoritative evidence "
            "reaggregation"
        ),
    ):
        campaign_finalizer.finalize_vllm_0271_publication_campaign(
            **{
                **call_kwargs,
                "full_score_aggregate_record": full_score_type_drift,
            }
        )

    finalization = campaign_finalizer.finalize_vllm_0271_publication_campaign(
        **call_kwargs
    )
    report = finalization.report_record
    gate = finalization.gate_record

    assert len(ledger_calls) == 1
    assert len(build_calls) == 1
    assert report["source_bindings"]["latency_summary"][
        "closed_record_sha256"
    ] == authoritative_latency["closed_record_sha256"]
    assert report["source_bindings"]["full_score_aggregate"][
        "closed_record_sha256"
    ] == authoritative_full_score["closed_record_sha256"]
    assert report["latency"] == {
        key: authoritative_latency[key]
        for key in ("analysis", "descriptive_cells", "estimates")
    }
    assert report["quality"] == {
        "aggregation_unit": authoritative_full_score["aggregation_unit"],
        "bootstrap": authoritative_full_score["bootstrap"],
        "datasets": authoritative_full_score["datasets"],
        "niah_grid": authoritative_full_score["niah_grid"],
        "protocol": authoritative_full_score["protocol"],
    }
    assert gate["benchmark_payload_digest"] == report["closed_record_sha256"]
    assert gate["ok"] is True


def test_synthetic_projection_is_deterministic_and_emits_standard_gate() -> None:
    inputs = _synthetic_report_inputs()
    expected_protocol = {
        "add_special_tokens": False,
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
        "prompt_text_mode": "logical",
        "protocol_id": "cachet-vllm-0.27.1-complete-score-v2",
        "request_parallelism": 4,
        "temperature": 0.0,
    }

    first = campaign_finalizer._build_report(**deepcopy(inputs))
    second = campaign_finalizer._build_report(**deepcopy(inputs))
    gate = campaign_finalizer._publication_gate_for_report(first)

    assert first == second
    assert first["closed_record_sha256"] == campaign_finalizer._closed_record_sha256(
        first
    )
    assert first["latency"] == {
        key: inputs["latency_summary_record"][key]
        for key in ("analysis", "descriptive_cells", "estimates")
    }
    assert first["quality"] == {
        "aggregation_unit": inputs["full_score_aggregate_record"][
            "aggregation_unit"
        ],
        "bootstrap": inputs["full_score_aggregate_record"]["bootstrap"],
        "datasets": inputs["full_score_aggregate_record"]["datasets"],
        "niah_grid": inputs["full_score_aggregate_record"]["niah_grid"],
        "protocol": expected_protocol,
    }
    assert first["scorer_contracts"] == inputs["full_score_aggregate_record"][
        "scorers"
    ]
    assert first["coverage"] == {
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
    assert gate == {
        "benchmark_payload_digest": first["closed_record_sha256"],
        "checked_cache_arms": ["vanilla_prefill"],
        "checked_cache_requests": 101_573,
        "checked_distinct_examples": 83_653,
        "cold_attested_requests": 15_360,
        "issues": [],
        "measurement_scopes": ["latency", "quality", "resource"],
        "ok": True,
        "policy": "publication",
        "record_type": BENCHMARK_PUBLICATION_GATE_RECORD_TYPE,
    }
    assert all(
        set(binding) == {
            "closed_record_sha256",
            "record_type",
            "schema_version",
        }
        for binding in first["source_bindings"].values()
    )


def test_public_report_pair_validator_accepts_exact_closed_pair() -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)

    campaign_finalizer.validate_vllm_0271_publication_report_pair(report, gate)

    for binding_name in (
        "campaign",
        "full_score_execution_plan",
        "full_score_inventory",
        "full_score_shard_plan",
    ):
        tampered = deepcopy(report)
        tampered["source_bindings"][binding_name][
            "closed_record_sha256"
        ] = _digest(f"unpinned-{binding_name}")
        _reclose(tampered)
        with pytest.raises(ValueError, match="pinned identity drift"):
            campaign_finalizer._validate_report_envelope(tampered)

    for cold_count in (15_359, 16_641):
        tampered = deepcopy(report)
        tampered["coverage"]["cold_attested_request_count"] = cold_count
        _reclose(tampered)
        with pytest.raises(ValueError, match="cold-attestation count is invalid"):
            campaign_finalizer._validate_report_envelope(tampered)

    expanded_cap = deepcopy(report)
    expanded_cap["ledger"]["latency_terminal_prefix"][
        "cap_cluster_hours"
    ] = 2_048.0
    expanded_cap["ledger"]["full_score_terminal_prefix"][
        "cap_cluster_hours"
    ] = 2_048.0
    expanded_cap["ledger"]["final_remaining_cluster_hours"] = 1_224.0
    _reclose(expanded_cap)
    with pytest.raises(
        ValueError,
        match="cap_cluster_hours must be a positive finite number no greater than",
    ):
        campaign_finalizer._validate_report_envelope(expanded_cap)

    invalid_parser_credit = deepcopy(report)
    invalid_parser_credit["quality"]["datasets"]["biography"]["methods"][
        "baseline_prefill"
    ]["metrics"]["exact_match"]["invalid_parser_score_sum"] = 0.25
    _reclose(invalid_parser_credit)
    with pytest.raises(ValueError, match="credits an invalid parsed answer"):
        campaign_finalizer._validate_report_envelope(invalid_parser_credit)

    assert "validate_vllm_0271_publication_report_pair" in (
        campaign_finalizer.__all__
    )


def test_public_report_pair_validator_rejects_type_drifted_gate() -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)
    gate["ok"] = 1

    with pytest.raises(
        ValueError,
        match="publication gate does not bind the supplied report",
    ):
        campaign_finalizer.validate_vllm_0271_publication_report_pair(
            report,
            gate,
        )


def test_public_report_pair_validator_rejects_nonfinite_json_input() -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)
    report["quality"]["protocol"]["temperature"] = float("nan")

    with pytest.raises(ValueError, match="Out of range float values"):
        campaign_finalizer.validate_vllm_0271_publication_report_pair(
            report,
            gate,
        )


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        pytest.param(
            "latency-estimate-control-cell",
            "latency estimand frozen-design drift",
            id="latency-explicit-pair-identity",
        ),
        pytest.param(
            "latency-physical-block-schema",
            "descriptive physical block schema is not closed",
            id="latency-nested-block-schema",
        ),
        pytest.param(
            "bootstrap-type",
            "quality bootstrap contract drift",
            id="bootstrap-type-exactness",
        ),
        pytest.param(
            "protocol-lifecycle",
            "full-score protocol drift",
            id="full-lifecycle-protocol",
        ),
        pytest.param(
            "scorer-metric-identity",
            "scorer/parser contract drift",
            id="scorer-metric-identity",
        ),
        pytest.param(
            "dataset-method-schema",
            "baseline_prefill must use a closed schema",
            id="dataset-method-schema",
        ),
        pytest.param(
            "dataset-metric-algebra",
            "mean/sum identity drift",
            id="metric-mean-sum-algebra",
        ),
        pytest.param(
            "parser-coverage",
            "parser-status coverage drift",
            id="parser-status-algebra",
        ),
        pytest.param(
            "paired-ci-schema",
            "bootstrap_ci95 must use a closed schema",
            id="paired-ci-schema",
        ),
        pytest.param(
            "paired-seed",
            "deterministic seed drift",
            id="paired-deterministic-seed",
        ),
        pytest.param(
            "niah-cell-count",
            "example count drift",
            id="niah-exact-cell-count",
        ),
        pytest.param(
            "niah-rollup-algebra",
            "NIAH cell/dataset metric drift",
            id="niah-cell-dataset-rollup",
        ),
    ],
)
def test_reclosed_nested_report_projection_tamper_is_rejected(
    tamper: str,
    match: str,
) -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)
    _tamper_nested_report_projection(report, tamper)

    with pytest.raises(ValueError, match=match):
        campaign_finalizer.validate_vllm_0271_publication_report_pair(
            report,
            gate,
        )


@pytest.mark.parametrize("forbidden_key", ["prompt", "run_id", "result_uri"])
def test_report_rejects_forbidden_nested_evidence(forbidden_key: str) -> None:
    report = _synthetic_report()
    report["quality"]["datasets"][SUPPORTED_V1_DATASETS[0]][forbidden_key] = (
        "must-not-publish"
    )
    _reclose(report)

    with pytest.raises(ValueError, match="forbidden in sanitized evidence"):
        campaign_finalizer._validate_report_envelope(report)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (
            ("schema_version",),
            True,
            "report envelope is invalid",
        ),
        (
            ("source_bindings", "latency_summary", "record_type"),
            "cachet.not_the_latency_summary.v1",
            "source binding latency_summary envelope drift",
        ),
        (
            ("source_bindings", "full_score_aggregate", "closed_record_sha256"),
            "not-a-sha256",
            "closed_record_sha256 must be a lowercase SHA-256 digest",
        ),
        (
            ("coverage", "checked_cache_request_count"),
            101_572,
            "report coverage is invalid",
        ),
        (
            ("coverage", "full_score_passes_per_method"),
            True,
            "report coverage is invalid",
        ),
        (
            (
                "ledger",
                "full_score_terminal_prefix",
                "terminal_actual_count",
            ),
            134,
            "report ledger closure is invalid",
        ),
        (
            ("ledger", "final_active_reserved_task_count"),
            1,
            "report ledger closure is invalid",
        ),
        (
            ("ledger", "final_active_reserved_task_count"),
            False,
            "report ledger closure is invalid",
        ),
        (
            ("ledger", "latency_terminal_prefix", "cap_cluster_hours"),
            1024,
            "report ledger closure is invalid",
        ),
    ],
)
def test_report_rejects_source_coverage_and_ledger_envelope_tamper(
    path: tuple[str, ...], value: Any, match: str
) -> None:
    report = _synthetic_report()
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _reclose(report)

    with pytest.raises(ValueError, match=match):
        campaign_finalizer._validate_report_envelope(report)


def test_report_rejects_nonfinite_projection_and_scalar_scorer() -> None:
    nonfinite_report = _synthetic_report()
    nonfinite_report["latency"]["analysis"]["nonfinite"] = float("nan")
    _reclose(nonfinite_report)
    with pytest.raises(ValueError, match="Out of range float values"):
        campaign_finalizer._validate_report_envelope(nonfinite_report)

    scalar_scorer_report = _synthetic_report()
    scalar_scorer_report["scorer_contracts"].append("ignored-scalar")
    _reclose(scalar_scorer_report)
    with pytest.raises(ValueError, match="scorer_contracts coverage is invalid"):
        campaign_finalizer._validate_report_envelope(scalar_scorer_report)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unexpected_top_level",), True),
        (("source_bindings", "latency_summary", "unexpected"), True),
        (("ledger", "unexpected"), "value"),
    ],
)
def test_report_rejects_extended_closed_schemas(
    path: tuple[str, ...], value: Any
) -> None:
    report = _synthetic_report()
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _reclose(report)

    with pytest.raises(ValueError, match="closed schema"):
        campaign_finalizer._validate_report_envelope(report)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("ok", False),
        ("ok", 1),
        ("issues", ["forged passing state"]),
        ("checked_cache_requests", 101_572),
        ("benchmark_payload_digest", "0" * 64),
        ("unexpected", True),
    ],
)
def test_writer_rejects_forged_or_extended_gate_before_writing(
    tmp_path: Path, field_name: str, value: Any
) -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)
    gate[field_name] = value
    finalization = _issued_finalization(report, gate)

    with pytest.raises(ValueError, match="does not bind the supplied report"):
        campaign_finalizer.write_vllm_0271_publication_finalization(
            finalization,
            tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_loader_rejects_json_type_drift_before_authority_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_report()
    expected_gate = campaign_finalizer._publication_gate_for_report(report)
    type_drift_gate = deepcopy(expected_gate)
    type_drift_gate["ok"] = 1
    report_path = tmp_path / campaign_finalizer.PUBLICATION_CAMPAIGN_REPORT_FILE_NAME
    gate_path = tmp_path / campaign_finalizer.PUBLICATION_CAMPAIGN_GATE_FILE_NAME
    report_path.write_bytes(
        campaign_finalizer._canonical_pretty_json_bytes(report)
    )
    gate_path.write_bytes(
        campaign_finalizer._canonical_pretty_json_bytes(type_drift_gate)
    )
    report_path.chmod(0o444)
    gate_path.chmod(0o444)
    def fail_authority_replay(**_kwargs: Any) -> Any:
        pytest.fail("gate drift must be rejected before authority replay")

    monkeypatch.setattr(
        campaign_finalizer,
        "finalize_vllm_0271_publication_campaign",
        fail_authority_replay,
    )

    with pytest.raises(ValueError, match="does not bind the supplied report"):
        _load_with_synthetic_authority_placeholders(tmp_path)


@pytest.mark.parametrize("checkout_mode", [0o600, 0o640, 0o644])
def test_loader_accepts_secure_git_mode_with_exact_authority_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkout_mode: int,
) -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)
    report_path = tmp_path / campaign_finalizer.PUBLICATION_CAMPAIGN_REPORT_FILE_NAME
    gate_path = tmp_path / campaign_finalizer.PUBLICATION_CAMPAIGN_GATE_FILE_NAME
    report_path.write_bytes(campaign_finalizer._canonical_pretty_json_bytes(report))
    gate_path.write_bytes(campaign_finalizer._canonical_pretty_json_bytes(gate))
    report_path.chmod(checkout_mode)
    gate_path.chmod(checkout_mode)
    expected = _issued_finalization(report, gate)
    monkeypatch.setattr(
        campaign_finalizer,
        "finalize_vllm_0271_publication_campaign",
        lambda **_kwargs: expected,
    )

    loaded = _load_with_synthetic_authority_placeholders(tmp_path)

    assert dict(loaded.report_record) == report
    assert dict(loaded.gate_record) == gate


def test_finalization_authority_cannot_be_constructed_by_external_issuer() -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)

    with pytest.raises(TypeError, match="must come from the campaign finalizer"):
        campaign_finalizer.PublicationCampaignFinalization(
            report_record=report,
            gate_record=gate,
            _issuer=object(),
        )


def test_writer_is_canonical_readable_exclusive_and_read_only(
    tmp_path: Path,
) -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)
    finalization = _issued_finalization(report, gate)

    report_path, gate_path = (
        campaign_finalizer.write_vllm_0271_publication_finalization(
            finalization,
            tmp_path,
        )
    )
    report_bytes = report_path.read_bytes()
    gate_bytes = gate_path.read_bytes()

    assert report_bytes == (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert gate_bytes == (
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(gate_path.stat().st_mode) == 0o444
    assert campaign_finalizer._read_canonical_json(
        report_path,
        "publication campaign report",
    ) == report
    assert campaign_finalizer._read_canonical_json(
        gate_path,
        "publication campaign gate",
    ) == gate

    with pytest.raises(FileExistsError):
        campaign_finalizer.write_vllm_0271_publication_finalization(
            finalization,
            tmp_path,
        )

    assert report_path.read_bytes() == report_bytes
    assert gate_path.read_bytes() == gate_bytes


def test_writer_requires_existing_evidence_directory(tmp_path: Path) -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)
    finalization = _issued_finalization(report, gate)
    missing_directory = tmp_path / "missing-evidence-directory"

    with pytest.raises(ValueError, match="must already exist as a real directory"):
        campaign_finalizer.write_vllm_0271_publication_finalization(
            finalization,
            missing_directory,
        )

    assert not missing_directory.exists()


def test_writer_recovers_exact_gate_after_report_serialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)
    finalization = _issued_finalization(report, gate)
    canonical_pretty_json_bytes = campaign_finalizer._canonical_pretty_json_bytes
    serialized_records: list[dict[str, Any]] = []

    def fail_report_serialization(record: dict[str, Any]) -> bytes:
        serialized_records.append(record)
        if len(serialized_records) == 2:
            raise TypeError("injected report serialization failure")
        return canonical_pretty_json_bytes(record)

    monkeypatch.setattr(
        campaign_finalizer,
        "_canonical_pretty_json_bytes",
        fail_report_serialization,
    )
    with pytest.raises(TypeError, match="injected report serialization failure"):
        campaign_finalizer.write_vllm_0271_publication_finalization(
            finalization,
            tmp_path,
        )

    report_path = tmp_path / campaign_finalizer.PUBLICATION_CAMPAIGN_REPORT_FILE_NAME
    gate_path = tmp_path / campaign_finalizer.PUBLICATION_CAMPAIGN_GATE_FILE_NAME
    assert serialized_records == [gate, report]
    assert not report_path.exists()
    assert gate_path.read_bytes() == canonical_pretty_json_bytes(gate)
    assert stat.S_IMODE(gate_path.stat().st_mode) == 0o444
    assert list(tmp_path.iterdir()) == [gate_path]

    monkeypatch.setattr(
        campaign_finalizer,
        "_canonical_pretty_json_bytes",
        canonical_pretty_json_bytes,
    )
    recovered_report, recovered_gate = (
        campaign_finalizer.write_vllm_0271_publication_finalization(
            finalization,
            tmp_path,
        )
    )
    assert recovered_report == report_path
    assert recovered_gate == gate_path
    assert report_path.read_bytes() == canonical_pretty_json_bytes(report)
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o444


@pytest.mark.parametrize("failure_point", ["temp_barrier", "final_link"])
def test_atomic_writer_removes_temporary_after_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)
    finalization = _issued_finalization(report, gate)
    if failure_point == "temp_barrier":
        durability_barrier = campaign_finalizer._durability_barrier
        barrier_calls = 0

        def fail_first_barrier(descriptor: int) -> None:
            nonlocal barrier_calls
            barrier_calls += 1
            if barrier_calls == 1:
                raise OSError("injected temporary durability-barrier failure")
            durability_barrier(descriptor)

        monkeypatch.setattr(
            campaign_finalizer,
            "_durability_barrier",
            fail_first_barrier,
        )
    else:

        def fail_link(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("injected final link failure")

        monkeypatch.setattr(campaign_finalizer.os, "link", fail_link)

    with pytest.raises(OSError, match="injected"):
        campaign_finalizer.write_vllm_0271_publication_finalization(
            finalization,
            tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_writer_closes_root_descriptor_when_unlock_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)
    finalization = _issued_finalization(report, gate)
    open_directory = campaign_finalizer._open_directory_no_symlinks
    release_lock = campaign_finalizer._release_publication_directory_lock
    root_descriptors: list[int] = []

    def capture_root_descriptor(path: Path, *, label: str) -> int:
        descriptor = open_directory(path, label=label)
        root_descriptors.append(descriptor)
        return descriptor

    def fail_after_unlock(descriptor: int) -> None:
        release_lock(descriptor)
        raise OSError("injected unlock failure")

    monkeypatch.setattr(
        campaign_finalizer,
        "_open_directory_no_symlinks",
        capture_root_descriptor,
    )
    monkeypatch.setattr(
        campaign_finalizer,
        "_release_publication_directory_lock",
        fail_after_unlock,
    )

    with pytest.raises(OSError, match="injected unlock failure"):
        campaign_finalizer.write_vllm_0271_publication_finalization(
            finalization,
            tmp_path,
        )

    assert len(root_descriptors) == 1
    with pytest.raises(OSError):
        campaign_finalizer.os.fstat(root_descriptors[0])


def test_writer_removes_only_narrow_owned_stale_temporary(
    tmp_path: Path,
) -> None:
    report = _synthetic_report()
    gate = campaign_finalizer._publication_gate_for_report(report)
    finalization = _issued_finalization(report, gate)
    stale_path = tmp_path / (
        f".{campaign_finalizer.PUBLICATION_CAMPAIGN_GATE_FILE_NAME}."
        f"{'a' * 32}.tmp"
    )
    unrelated_path = tmp_path / ".unrelated.tmp"
    stale_path.write_bytes(b"interrupted publication temporary")
    stale_path.chmod(0o600)
    unrelated_path.write_bytes(b"unrelated")

    report_path, gate_path = (
        campaign_finalizer.write_vllm_0271_publication_finalization(
            finalization,
            tmp_path,
        )
    )

    assert not stale_path.exists()
    assert unrelated_path.read_bytes() == b"unrelated"
    assert report_path.is_file()
    assert gate_path.is_file()


def test_canonical_reader_rejects_noncanonical_json_bytes(tmp_path: Path) -> None:
    path = tmp_path / "noncanonical.json"
    path.write_text(json.dumps(_synthetic_report()), encoding="utf-8")
    path.chmod(0o444)

    with pytest.raises(ValueError, match="bytes are not canonical"):
        campaign_finalizer._read_canonical_json(path, "synthetic report")


def test_canonical_reader_rejects_mutable_json_file(tmp_path: Path) -> None:
    path = tmp_path / "mutable.json"
    path.write_bytes(
        campaign_finalizer._canonical_pretty_json_bytes(_synthetic_report())
    )
    path.chmod(0o644)

    with pytest.raises(ValueError, match="must be a regular read-only file"):
        campaign_finalizer._read_canonical_json(path, "synthetic report")
