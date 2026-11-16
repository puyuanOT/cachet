"""Pure Markdown rendering for the governed vLLM 0.27.1 campaign report.

The renderer accepts only an exact, passing report/gate pair. It performs no
I/O: callers own the surrounding Markdown and replace its uniquely delimited
regions with :func:`replace_vllm_0271_publication_table_regions`.

Acceptance means the offline pair validator passed. That validator checks the
report's internal consistency and pinned source digests, not that its numbers
came from the campaign's evidence; the authority replay runs only in the
finalizing process (see :mod:`document_kv_cache.publication_campaign_finalizer`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_ENGINE_VERSION,
)
from document_kv_cache.publication_campaign_finalizer import (
    PUBLICATION_CAMPAIGN_REPORT_RECORD_TYPE,
    PUBLICATION_CAMPAIGN_VLLM_0300_REPORT_RECORD_TYPE,
    validate_vllm_0271_publication_report_pair,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_A10G_GPU_MEMORY_UTILIZATION,
    GPU_QUALIFICATION_A10G_MIN_PEAK_HEADROOM_BYTES,
    GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES,
)
from document_kv_cache.publication_latency_execution import (
    PUBLICATION_LATENCY_LEDGER_BINDINGS,
    PUBLICATION_LATENCY_LEDGER_RECEIPT_BINDING,
    PUBLICATION_LATENCY_RECOVERED_BALANCE_BINDING,
    PUBLICATION_LATENCY_RECOVERED_RECEIPT_BINDING,
)


_METHODS: Final[tuple[str, ...]] = ("baseline_prefill", "vanilla_prefill")
_PARSER_STATUS_ORDER: Final[tuple[str, ...]] = (
    "ok",
    "missing_block",
    "multiple_or_malformed_blocks",
    "extraneous_text",
    "nested_block",
    "empty_answer",
)
_CACHE_TELEMETRY_KEYS: Final[frozenset[str]] = frozenset(
    {
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
    }
)
_DESCRIPTIVE_METRIC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "configured_closed_loop_concurrency",
        "gpu_utilization_sample_count",
        "mean_gpu_utilization_percent",
        "observation_count",
        "p50_decode_tokens_per_second",
        "p50_time_to_completion_seconds",
        "p50_ttft_seconds",
        "p95_time_to_completion_seconds",
        "p95_ttft_seconds",
        "peak_gpu_process_memory_bytes",
        "peak_gpu_utilization_percent",
        "peak_host_memory_used_bytes",
        "peak_process_tree_rss_bytes",
    }
)
_DESCRIPTIVE_CELL_KEYS: Final[frozenset[str]] = (
    _DESCRIPTIVE_METRIC_KEYS
    | frozenset(
        {
            "cache_telemetry",
            "cell_id",
            "cell_kind",
            "cell_sha256",
            "comparison_family",
            "input_tokens",
            "method_id",
            "physical_blocks",
            "quantile_method",
            "request_parallelism",
            "setting_id",
        }
    )
)
_DESCRIPTIVE_BLOCK_KEYS: Final[frozenset[str]] = _DESCRIPTIVE_METRIC_KEYS | frozenset(
    {"cache_telemetry", "deployment_block", "job_id"}
)
_ESTIMATE_COMMON_KEYS: Final[frozenset[str]] = frozenset(
    {
        "comparison_family",
        "control_cell_id",
        "deployment_block_count",
        "estimand_id",
        "example_count_per_block",
        "input_tokens",
        "metrics",
        "paired_request_count",
        "request_parallelism",
        "setting_id",
        "speedup_direction",
        "treatment_cell_id",
    }
)
_QUALITY_KEYS: Final[frozenset[str]] = frozenset(
    {"aggregation_unit", "bootstrap", "datasets", "niah_grid", "protocol"}
)
_SCORER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "answer_parser_digest",
        "answer_parser_id",
        "answer_parser_plugin_path",
        "answer_parser_version",
        "dataset",
        "metric_names",
        "plugin_path",
        "publication_approved",
        "scorer_id",
        "scorer_version",
    }
)
_REPORT_LEDGER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "final_accounted_cluster_hours",
        "final_active_reserved_cluster_hours",
        "final_active_reserved_task_count",
        "final_remaining_cluster_hours",
        "full_score_terminal_prefix",
        "latency_terminal_prefix",
        "ledger_id",
        "ledger_path_sha256",
        "required_unreserved_headroom_cluster_hours",
    }
)
_VLLM_0300_REPORT_LEDGER_KEYS: Final[frozenset[str]] = _REPORT_LEDGER_KEYS | frozenset(
    {
        "cap_raises",
        "full_score_abandoned_attempts",
        "full_score_capacity_retries",
        "latency_recovered_balance",
        "qualification_capacity_retries",
    }
)
_CAP_RAISE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "from_cap_cluster_hours",
        "reservation_count",
        "submission_receipt_count",
        "terminal_actual_count",
        "to_cap_cluster_hours",
    }
)
_ATTEMPT_HOURS_KEYS: Final[frozenset[str]] = frozenset(
    {"attempt_count", "cluster_hours"}
)
_ABANDONED_ATTEMPT_KEYS: Final[frozenset[str]] = _ATTEMPT_HOURS_KEYS | frozenset(
    {"closed_consumer_phase_count"}
)
_LATENCY_PROVENANCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "execution_plan_count",
        "ledger_binding_counts",
        "plan_set_sha256",
        "superseded_attempt_count",
    }
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_LATENCY_CACHE_REQUEST_COUNT: Final = 17_920
_STORAGE_SETTING_CACHE_REQUEST_COUNT: Final = 1_280
# RAM-storage requests never attest a cold read and Unity Catalog requests may
# or may not, so one or two storage settings' requests carry no attestation.
_MAX_COLD_ATTESTED_REQUEST_COUNT: Final = (
    _LATENCY_CACHE_REQUEST_COUNT - _STORAGE_SETTING_CACHE_REQUEST_COUNT
)
_MIN_COLD_ATTESTED_REQUEST_COUNT: Final = (
    _MAX_COLD_ATTESTED_REQUEST_COUNT - _STORAGE_SETTING_CACHE_REQUEST_COUNT
)

_CORE_LATENCY_ROWS: Final[tuple[tuple[str, str, int, int], ...]] = tuple(
    (
        f"core-{method_id}-{tokens}-c{concurrency}",
        method_id,
        tokens,
        concurrency,
    )
    for tokens in (8_192, 16_384, 32_768)
    for concurrency in (1, 2, 4)
    for method_id in _METHODS
)
_AUXILIARY_CELL_ROWS: Final[tuple[tuple[str, str, str], ...]] = (
    ("auxiliary-precision-bf16", "precision-bf16", "precision"),
    ("auxiliary-storage-disk", "storage-disk", "storage"),
    ("auxiliary-storage-ram", "storage-ram", "storage"),
    ("auxiliary-storage-uc", "storage-uc", "storage"),
    ("auxiliary-hardware-a10g", "hardware-a10g", "hardware"),
)
_DESCRIPTIVE_CELL_ORDER: Final[tuple[str, ...]] = tuple(
    row[0] for row in _CORE_LATENCY_ROWS
) + tuple(row[0] for row in _AUXILIARY_CELL_ROWS)
_ESTIMAND_ROWS: Final[
    tuple[tuple[str, str, str | None, str, str, int, int, str, str], ...]
] = (
    *tuple(
        (
            f"method-{tokens}-c{concurrency}",
            "method",
            None,
            f"core-baseline_prefill-{tokens}-c{concurrency}",
            f"core-vanilla_prefill-{tokens}-c{concurrency}",
            tokens,
            concurrency,
            "Vanilla KV vs Baseline",
            f"{tokens // 1024}k, concurrency {concurrency}",
        )
        for tokens in (8_192, 16_384, 32_768)
        for concurrency in (1, 2, 4)
    ),
    (
        "auxiliary-precision-bf16",
        "precision",
        "precision-bf16",
        "core-vanilla_prefill-16384-c4",
        "auxiliary-precision-bf16",
        16_384,
        4,
        "BF16 payload/runtime KV vs Q8",
        "16k, concurrency 4, L4/NVMe",
    ),
    (
        "auxiliary-storage-ram",
        "storage",
        "storage-ram",
        "auxiliary-storage-disk",
        "auxiliary-storage-ram",
        16_384,
        4,
        "RAM vs Disk",
        "16k, concurrency 4, L4",
    ),
    (
        "auxiliary-storage-uc",
        "storage",
        "storage-uc",
        "auxiliary-storage-disk",
        "auxiliary-storage-uc",
        16_384,
        4,
        "Unity Catalog vs Disk",
        "16k, concurrency 4, L4",
    ),
    (
        "auxiliary-hardware-a10g",
        "hardware",
        "hardware-a10g",
        "core-vanilla_prefill-16384-c4",
        "auxiliary-hardware-a10g",
        16_384,
        4,
        "A10G vs L4",
        "16k, concurrency 4, local NVMe",
    ),
)
_SCORE_ROWS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("biography", "exact_match", "Biography", "Normalized-title exact match"),
    ("hotpotqa", "exact_match", "HotpotQA", "Answer exact match"),
    ("hotpotqa", "f1", "HotpotQA", "Answer F1"),
    (
        "musique",
        "answer_em",
        "MusiQue",
        "Official answer exact match, alias-max",
    ),
    (
        "musique",
        "answer_f1",
        "MusiQue",
        "Official answer F1, alias-max",
    ),
    ("niah", "accuracy", "NIAH", "Exact-value overall accuracy"),
)
_METRICS_BY_DATASET: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "biography": ("exact_match",),
        "hotpotqa": ("exact_match", "f1"),
        "musique": ("answer_em", "answer_f1"),
        "niah": ("accuracy",),
    }
)
_NIAH_ROWS: Final[tuple[tuple[str, str, str], ...]] = tuple(
    (
        f"niah-{tokens // 1024}k-depth-{depth:02d}",
        f"{tokens // 1024}k",
        f"{depth}%",
    )
    for tokens in (8_192, 16_384, 32_768)
    for depth in (10, 50, 90)
)

PUBLICATION_TABLE_SECTION_ORDER: Final[tuple[str, ...]] = (
    "status",
    "campaign_disclosures",
    "core_latency",
    "latency_estimands",
    "dataset_scores",
    "niah_grid",
    "precision",
    "storage",
    "hardware",
    "platform",
    "resource_cache",
)
"""Frozen order of governed Markdown regions in the public benchmark page."""


def _marker_pair(section: str) -> tuple[str, str]:
    stem = f"cachet:vllm-0271-publication-table:{section.replace('_', '-')}"
    return (f"<!-- {stem}:begin -->", f"<!-- {stem}:end -->")


PUBLICATION_TABLE_REGION_MARKERS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {section: _marker_pair(section) for section in PUBLICATION_TABLE_SECTION_ORDER}
)
"""Unique begin/end HTML comments for each governed table region."""


def render_vllm_0271_publication_table_regions(
    report_record: Mapping[str, Any],
    gate_record: Mapping[str, Any],
) -> Mapping[str, str]:
    """Validate one exact report/gate pair and render all governed table bodies."""

    report, _gate = _validated_pair_snapshots(report_record, gate_record)
    if report.get("engine_version") != PUBLICATION_CAMPAIGN_ENGINE_VERSION:
        raise ValueError(
            "publication table renderer requires vLLM "
            + PUBLICATION_CAMPAIGN_ENGINE_VERSION
        )
    if report.get("policy") != "publication":
        raise ValueError("publication table renderer requires publication policy")
    if report.get("campaign_id") != "vllm-0271-publication-v1":
        raise ValueError("publication campaign identity drift")

    coverage = _mapping_field(report, "coverage", "publication report")
    _validate_rendered_coverage(coverage)
    latency = _mapping_field(report, "latency", "publication report")
    _require_exact_keys(
        latency,
        frozenset({"analysis", "descriptive_cells", "estimates"}),
        "publication latency projection",
    )
    cells = _validated_descriptive_cells(
        _list_field(latency, "descriptive_cells", "publication latency projection")
    )
    estimates = _validated_estimands(
        _list_field(latency, "estimates", "publication latency projection")
    )
    quality = _mapping_field(report, "quality", "publication report")
    _require_exact_keys(quality, _QUALITY_KEYS, "publication quality projection")
    _validate_quality_protocol(
        _mapping_field(quality, "protocol", "publication quality projection")
    )
    datasets = _validated_score_datasets(
        _mapping_field(quality, "datasets", "publication quality projection")
    )
    niah_grid = _validated_niah_grid(
        _mapping_field(quality, "niah_grid", "publication quality projection")
    )
    _validate_scorer_contracts(report.get("scorer_contracts"))

    rendered = {
        "status": _render_status(coverage),
        "campaign_disclosures": _render_campaign_disclosures(report, coverage),
        "core_latency": _render_core_latency(cells),
        "latency_estimands": _render_latency_estimands(estimates),
        "dataset_scores": _render_dataset_scores(datasets),
        "niah_grid": _render_niah_grid(niah_grid),
        "precision": _render_precision(cells),
        "storage": _render_storage(cells),
        "hardware": _render_hardware(cells),
        "platform": _render_platform(),
        "resource_cache": _render_resource_cache(cells),
    }
    if tuple(rendered) != PUBLICATION_TABLE_SECTION_ORDER:
        raise RuntimeError("publication table renderer section order drift")
    return MappingProxyType(rendered)


def render_vllm_0271_publication_appendix_readme(
    report_record: Mapping[str, Any],
    gate_record: Mapping[str, Any],
) -> str:
    """Render the exact README committed beside one passing report/gate pair."""

    report, _gate = _validated_pair_snapshots(report_record, gate_record)
    if report.get("engine_version") != PUBLICATION_CAMPAIGN_ENGINE_VERSION:
        raise ValueError(
            "publication appendix requires vLLM "
            + PUBLICATION_CAMPAIGN_ENGINE_VERSION
        )
    if report.get("policy") != "publication":
        raise ValueError("publication appendix requires publication policy")
    if report.get("campaign_id") != "vllm-0271-publication-v1":
        raise ValueError("publication appendix campaign identity drift")
    digest = _sha256_field(
        report,
        "closed_record_sha256",
        "publication report",
    )
    return (
        f"# vLLM {PUBLICATION_CAMPAIGN_ENGINE_VERSION} Publication Campaign\n\n"
        "This directory contains the sanitized, content-addressed publication "
        "pair for `vllm-0271-publication-v1`.\n\n"
        "- [`campaign-report.json`](campaign-report.json) is the sealed campaign "
        f"report with closed-record SHA-256 `{digest}`.\n"
        "- [`benchmark-publication-gate.json`](benchmark-publication-gate.json) "
        "is the exact passing standard gate bound to that report digest.\n\n"
        "Raw Databricks responses, credentials, wheels, logs, generated "
        "datasets, prompt payloads, and scratch output are intentionally "
        "excluded.\n"
    )


def replace_vllm_0271_publication_table_regions(
    markdown: str,
    report_record: Mapping[str, Any],
    gate_record: Mapping[str, Any],
) -> str:
    """Replace every unique governed region and preserve all outside bytes."""

    rendered = render_vllm_0271_publication_table_regions(
        report_record,
        gate_record,
    )
    spans = _publication_region_spans(markdown)
    pieces: list[str] = []
    cursor = 0
    for section in PUBLICATION_TABLE_SECTION_ORDER:
        begin_at, _content_at, _content_end, end_after = spans[section]
        pieces.append(markdown[cursor:begin_at])
        begin, end = PUBLICATION_TABLE_REGION_MARKERS[section]
        pieces.append(f"{begin}\n{rendered[section]}\n{end}")
        cursor = end_after
    pieces.append(markdown[cursor:])
    result = "".join(pieces)
    result.encode("utf-8", errors="strict")
    return result


def validate_vllm_0271_publication_table_regions(
    markdown: str,
    report_record: Mapping[str, Any],
    gate_record: Mapping[str, Any],
) -> None:
    """Require every governed region to equal the canonical rendering exactly."""

    rendered = render_vllm_0271_publication_table_regions(
        report_record,
        gate_record,
    )
    spans = _publication_region_spans(markdown)
    for section in PUBLICATION_TABLE_SECTION_ORDER:
        _begin_at, content_at, content_end, _end_after = spans[section]
        expected = f"\n{rendered[section]}\n"
        if markdown[content_at:content_end] != expected:
            raise ValueError(
                f"publication table region {section!r} is not canonical"
            )


def _publication_region_spans(
    markdown: str,
) -> dict[str, tuple[int, int, int, int]]:
    if not isinstance(markdown, str):
        raise TypeError("markdown must be a string")
    if "\r" in markdown or "\x00" in markdown:
        raise ValueError("markdown must use canonical LF text without NUL bytes")
    markdown.encode("utf-8", errors="strict")
    spans: dict[str, tuple[int, int, int, int]] = {}
    previous_end = -1
    for section in PUBLICATION_TABLE_SECTION_ORDER:
        begin, end = PUBLICATION_TABLE_REGION_MARKERS[section]
        if markdown.count(begin) != 1 or markdown.count(end) != 1:
            raise ValueError(
                f"publication table region {section!r} must have one marker pair"
            )
        begin_at = markdown.index(begin)
        content_at = begin_at + len(begin)
        content_end = markdown.index(end)
        end_after = content_end + len(end)
        if begin_at <= previous_end or content_end < content_at:
            raise ValueError("publication table marker order or nesting drift")
        spans[section] = (begin_at, content_at, content_end, end_after)
        previous_end = end_after
    return spans


def _validate_rendered_coverage(coverage: Mapping[str, Any]) -> None:
    expected = {
        "checked_cache_request_count": 101_573,
        "checked_distinct_example_count": 83_653,
        "full_score_cache_request_count": 83_653,
        "full_score_identity_count": 83_653,
        "full_score_passes_per_method": 1,
        "full_score_phase_count": 20,
        "full_score_shard_count": 160,
        "full_score_wave_count": 10,
        "latency_cache_request_count": _LATENCY_CACHE_REQUEST_COUNT,
        "latency_descriptive_cell_count": 23,
        "latency_estimand_count": 13,
        "latency_job_count": 115,
        "latency_request_count": 29_440,
        "methods": list(_METHODS),
        "niah_cell_count": 9,
    }
    for key, value in expected.items():
        if type(coverage.get(key)) is not type(value) or coverage.get(key) != value:
            raise ValueError(f"publication table coverage field {key!r} drift")
    cold = _int_field(coverage, "cold_attested_request_count", "coverage")
    if not (
        _MIN_COLD_ATTESTED_REQUEST_COUNT
        <= cold
        <= _MAX_COLD_ATTESTED_REQUEST_COUNT
    ):
        raise ValueError("publication cold-attestation coverage drift")


def _validated_descriptive_cells(
    raw_cells: list[Any],
) -> dict[str, Mapping[str, Any]]:
    cells = _ordered_mapping_rows(
        raw_cells,
        id_field="cell_id",
        expected_ids=_DESCRIPTIVE_CELL_ORDER,
        label="publication descriptive cells",
    )
    core_specs = {
        cell_id: (method, tokens, concurrency)
        for cell_id, method, tokens, concurrency in _CORE_LATENCY_ROWS
    }
    auxiliary_specs = {
        cell_id: (setting_id, family)
        for cell_id, setting_id, family in _AUXILIARY_CELL_ROWS
    }
    for cell_id in _DESCRIPTIVE_CELL_ORDER:
        cell = cells[cell_id]
        _require_exact_keys(
            cell,
            _DESCRIPTIVE_CELL_KEYS,
            f"descriptive cell {cell_id}",
        )
        if cell.get("quantile_method") != "empirical_nearest_rank":
            raise ValueError(f"descriptive cell {cell_id} quantile policy drift")
        _sha256_field(cell, "cell_sha256", f"descriptive cell {cell_id}")
        if cell_id in core_specs:
            method_id, input_tokens, concurrency = core_specs[cell_id]
            expected_identity = {
                "cell_kind": "core_pooled_five_blocks",
                "comparison_family": None,
                "input_tokens": input_tokens,
                "method_id": method_id,
                "request_parallelism": concurrency,
                "setting_id": None,
            }
        else:
            setting_id, family = auxiliary_specs[cell_id]
            expected_identity = {
                "cell_kind": "auxiliary_pooled_five_blocks",
                "comparison_family": family,
                "input_tokens": 16_384,
                "method_id": "vanilla_prefill",
                "request_parallelism": 4,
                "setting_id": setting_id,
            }
        if any(
            cell.get(key) != value for key, value in expected_identity.items()
        ):
            raise ValueError(f"descriptive cell {cell_id} identity drift")
        _validate_descriptive_metrics(cell, f"descriptive cell {cell_id}")
        blocks = _list_field(
            cell,
            "physical_blocks",
            f"descriptive cell {cell_id}",
        )
        if len(blocks) != 5:
            raise ValueError(f"descriptive cell {cell_id} must contain five blocks")
        normalized_blocks: list[Mapping[str, Any]] = []
        for index, raw_block in enumerate(blocks, start=1):
            block = _mapping_value(
                raw_block,
                f"descriptive cell {cell_id} block",
            )
            _require_exact_keys(
                block,
                _DESCRIPTIVE_BLOCK_KEYS,
                f"descriptive cell {cell_id} block {index}",
            )
            if _int_field(block, "deployment_block", cell_id) != index:
                raise ValueError(f"descriptive cell {cell_id} block order drift")
            _string_field(block, "job_id", cell_id)
            _validate_descriptive_metrics(
                block,
                f"descriptive cell {cell_id} block {index}",
            )
            if block.get("configured_closed_loop_concurrency") != cell.get(
                "request_parallelism"
            ):
                raise ValueError(
                    f"descriptive cell {cell_id} block concurrency drift"
                )
            normalized_blocks.append(block)
        if len({block.get("job_id") for block in normalized_blocks}) != 5:
            raise ValueError(f"descriptive cell {cell_id} block identity reuse")
        if _int_field(cell, "observation_count", cell_id) != sum(
            _int_field(block, "observation_count", cell_id)
            for block in normalized_blocks
        ):
            raise ValueError(f"descriptive cell {cell_id} observation sum drift")
        sample_count = _int_field(
            cell,
            "gpu_utilization_sample_count",
            cell_id,
        )
        block_sample_count = sum(
            _int_field(block, "gpu_utilization_sample_count", cell_id)
            for block in normalized_blocks
        )
        if sample_count != block_sample_count:
            raise ValueError(
                f"descriptive cell {cell_id} utilization sample drift"
            )
        weighted_mean = sum(
            _number_field(block, "mean_gpu_utilization_percent", cell_id)
            * _int_field(block, "gpu_utilization_sample_count", cell_id)
            for block in normalized_blocks
        ) / block_sample_count
        if not _numbers_match(
            _number_field(cell, "mean_gpu_utilization_percent", cell_id),
            weighted_mean,
        ):
            raise ValueError(f"descriptive cell {cell_id} utilization mean drift")
        for peak_field in (
            "peak_gpu_process_memory_bytes",
            "peak_gpu_utilization_percent",
            "peak_host_memory_used_bytes",
            "peak_process_tree_rss_bytes",
        ):
            if not _numbers_match(
                _number_field(cell, peak_field, cell_id),
                max(
                    _number_field(block, peak_field, cell_id)
                    for block in normalized_blocks
                ),
            ):
                raise ValueError(
                    f"descriptive cell {cell_id} {peak_field} drift"
                )
        cell_cache = _mapping_field(cell, "cache_telemetry", cell_id)
        for key in _CACHE_TELEMETRY_KEYS:
            if _int_field(cell_cache, key, cell_id) != sum(
                _int_field(
                    _mapping_field(block, "cache_telemetry", cell_id),
                    key,
                    cell_id,
                )
                for block in normalized_blocks
            ):
                raise ValueError(f"descriptive cell {cell_id} cache sum drift")
    return cells


def _validate_descriptive_metrics(record: Mapping[str, Any], label: str) -> None:
    for field in (
        "configured_closed_loop_concurrency",
        "gpu_utilization_sample_count",
        "observation_count",
    ):
        if _int_field(record, field, label) <= 0:
            raise ValueError(f"{label}.{field} must be positive")
    for field in (
        "peak_gpu_process_memory_bytes",
        "peak_host_memory_used_bytes",
        "peak_process_tree_rss_bytes",
    ):
        if _int_field(record, field, label) < 0:
            raise ValueError(f"{label}.{field} must be nonnegative")
    for field in (
        "p50_decode_tokens_per_second",
        "p50_time_to_completion_seconds",
        "p50_ttft_seconds",
        "p95_time_to_completion_seconds",
        "p95_ttft_seconds",
    ):
        if _number_field(record, field, label) <= 0.0:
            raise ValueError(f"{label}.{field} must be positive")
    p50_ttft = _number_field(record, "p50_ttft_seconds", label)
    p95_ttft = _number_field(record, "p95_ttft_seconds", label)
    p50_ttc = _number_field(record, "p50_time_to_completion_seconds", label)
    p95_ttc = _number_field(record, "p95_time_to_completion_seconds", label)
    if (
        p50_ttft > p95_ttft
        or p50_ttc > p95_ttc
        or p50_ttft > p50_ttc
        or p95_ttft > p95_ttc
    ):
        raise ValueError(f"{label} quantile ordering drift")
    mean_utilization = _number_field(
        record,
        "mean_gpu_utilization_percent",
        label,
    )
    peak_utilization = _number_field(
        record,
        "peak_gpu_utilization_percent",
        label,
    )
    if not 0.0 <= mean_utilization <= peak_utilization <= 100.0:
        raise ValueError(f"{label} GPU utilization bounds drift")
    cache = _mapping_field(record, "cache_telemetry", label)
    _require_exact_keys(cache, _CACHE_TELEMETRY_KEYS, f"{label} cache telemetry")
    for field in _CACHE_TELEMETRY_KEYS:
        if _int_field(cache, field, label) < 0:
            raise ValueError(
                f"{label}.cache_telemetry.{field} must be nonnegative"
            )


def _validated_estimands(
    raw_estimates: list[Any],
) -> dict[str, Mapping[str, Any]]:
    expected_ids = tuple(row[0] for row in _ESTIMAND_ROWS)
    estimates = _ordered_mapping_rows(
        raw_estimates,
        id_field="estimand_id",
        expected_ids=expected_ids,
        label="publication latency estimands",
    )
    for (
        estimand_id,
        family,
        setting_id,
        control_cell_id,
        treatment_cell_id,
        input_tokens,
        concurrency,
        _comparison,
        _setting,
    ) in _ESTIMAND_ROWS:
        estimate = estimates[estimand_id]
        _require_exact_keys(
            estimate,
            _ESTIMATE_COMMON_KEYS,
            f"estimand {estimand_id}",
        )
        expected_identity = {
            "comparison_family": family,
            "control_cell_id": control_cell_id,
            "deployment_block_count": 5,
            "estimand_id": estimand_id,
            "input_tokens": input_tokens,
            "paired_request_count": 1_280,
            "request_parallelism": concurrency,
            "setting_id": setting_id,
            "speedup_direction": (
                "control_latency_divided_by_treatment_latency"
            ),
            "treatment_cell_id": treatment_cell_id,
        }
        if any(
            estimate.get(key) != value
            for key, value in expected_identity.items()
        ):
            raise ValueError(f"estimand {estimand_id} identity drift")
        expected_examples = (
            8 if setting_id in {"storage-ram", "storage-uc"} else 128
        )
        if (
            _int_field(estimate, "example_count_per_block", estimand_id)
            != expected_examples
        ):
            raise ValueError(f"estimand {estimand_id} example count drift")
        metrics = _mapping_field(estimate, "metrics", f"estimand {estimand_id}")
        _require_exact_keys(
            metrics,
            frozenset({"ttft", "time_to_completion"}),
            f"estimand {estimand_id} metrics",
        )
        for metric_name in ("ttft", "time_to_completion"):
            metric = _mapping_field(
                metrics,
                metric_name,
                f"estimand {estimand_id}",
            )
            _require_exact_keys(
                metric,
                frozenset(
                    {"confidence_interval_95", "geometric_mean_speedup"}
                ),
                f"estimand {estimand_id}.{metric_name}",
            )
            point = _number_field(
                metric,
                "geometric_mean_speedup",
                estimand_id,
            )
            interval = _list_field(
                metric,
                "confidence_interval_95",
                estimand_id,
            )
            if point <= 0.0 or len(interval) != 2:
                raise ValueError(
                    f"estimand {estimand_id}.{metric_name} is invalid"
                )
            lower = _finite_number(
                interval[0],
                f"estimand {estimand_id} CI lower",
            )
            upper = _finite_number(
                interval[1],
                f"estimand {estimand_id} CI upper",
            )
            if lower <= 0.0 or lower > upper:
                raise ValueError(
                    f"estimand {estimand_id}.{metric_name} CI is invalid"
                )
    return estimates


def _validate_quality_protocol(protocol: Mapping[str, Any]) -> None:
    expected_keys = frozenset(
        {
            "add_special_tokens",
            "complete_inventory_required",
            "input_length",
            "lifecycle",
            "max_tokens",
            "methods",
            "natural_eos",
            "passes_per_method",
            "prompt_text_mode",
            "protocol_id",
            "request_parallelism",
            "temperature",
        }
    )
    _require_exact_keys(protocol, expected_keys, "full-score protocol")
    input_length = _mapping_field(
        protocol,
        "input_length",
        "full-score protocol",
    )
    _require_exact_keys(
        input_length,
        frozenset(
            {"max_natural_prompt_tokens", "padding", "tokenizer_truncation"}
        ),
        "full-score input-length protocol",
    )
    expected = {
        "add_special_tokens": False,
        "complete_inventory_required": True,
        "lifecycle": [
            "generate_q8_kv",
            "baseline_inference",
            "vanilla_inference",
            "validate_paired_outputs",
            "commit_durable_evidence",
            "delete_ephemeral_q8_kv",
        ],
        "max_tokens": 64,
        "methods": list(_METHODS),
        "natural_eos": True,
        "passes_per_method": 1,
        "prompt_text_mode": "logical",
        "protocol_id": "cachet-vllm-0.27.1-complete-score-v2",
        "request_parallelism": 4,
        "temperature": 0.0,
    }
    if any(
        type(protocol.get(key)) is not type(value)
        or protocol.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("full-score protocol drift")
    expected_input = {
        "max_natural_prompt_tokens": 32_768,
        "padding": False,
        "tokenizer_truncation": False,
    }
    if any(
        type(input_length.get(key)) is not type(value)
        or input_length.get(key) != value
        for key, value in expected_input.items()
    ):
        raise ValueError("full-score input-length protocol drift")


def _validated_score_datasets(
    raw_datasets: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    _require_exact_keys(
        raw_datasets,
        frozenset(_METRICS_BY_DATASET),
        "score datasets",
    )
    datasets: dict[str, Mapping[str, Any]] = {}
    for dataset, expected_metrics in _METRICS_BY_DATASET.items():
        record = _mapping_field(raw_datasets, dataset, "score datasets")
        _require_exact_keys(
            record,
            frozenset(
                {"example_count", "methods", "paired_vanilla_minus_baseline"}
            ),
            f"score dataset {dataset}",
        )
        example_count = _int_field(record, "example_count", dataset)
        if example_count <= 0:
            raise ValueError(f"score dataset {dataset} must be nonempty")
        methods = _mapping_field(
            record,
            "methods",
            f"score dataset {dataset}",
        )
        _require_exact_keys(
            methods,
            frozenset(_METHODS),
            f"score dataset {dataset} methods",
        )
        method_means: dict[str, dict[str, float]] = {}
        for method in _METHODS:
            method_record = _mapping_field(
                methods,
                method,
                f"score dataset {dataset}",
            )
            _require_exact_keys(
                method_record,
                frozenset(
                    {"example_count", "metrics", "parser_status_counts"}
                ),
                f"score dataset {dataset}.{method}",
            )
            if (
                _int_field(method_record, "example_count", dataset)
                != example_count
            ):
                raise ValueError(f"score dataset {dataset}.{method} count drift")
            metrics = _mapping_field(
                method_record,
                "metrics",
                f"score dataset {dataset}.{method}",
            )
            _require_exact_keys(
                metrics,
                frozenset(expected_metrics),
                f"score dataset {dataset}.{method} metrics",
            )
            method_means[method] = {}
            for metric_name in expected_metrics:
                summary = _mapping_field(
                    metrics,
                    metric_name,
                    f"score dataset {dataset}.{method}",
                )
                mean = _validate_score_metric_summary(
                    summary,
                    expected_count=example_count,
                    label=f"score dataset {dataset}.{method}.{metric_name}",
                )
                method_means[method][metric_name] = mean
            _validate_parser_counts(
                _mapping_field(
                    method_record,
                    "parser_status_counts",
                    f"score dataset {dataset}.{method}",
                ),
                expected_count=example_count,
                label=f"score dataset {dataset}.{method}",
            )
        paired = _mapping_field(
            record,
            "paired_vanilla_minus_baseline",
            f"score dataset {dataset}",
        )
        _require_exact_keys(
            paired,
            frozenset(expected_metrics),
            f"score dataset {dataset} paired metrics",
        )
        for metric_name in expected_metrics:
            mean = _validate_paired_summary(
                _mapping_field(
                    paired,
                    metric_name,
                    f"score dataset {dataset} paired",
                ),
                expected_count=example_count,
                label=f"score dataset {dataset}.paired.{metric_name}",
            )
            expected_delta = (
                method_means["vanilla_prefill"][metric_name]
                - method_means["baseline_prefill"][metric_name]
            )
            if not _numbers_match(mean, expected_delta):
                raise ValueError(
                    f"score dataset {dataset}.{metric_name} delta drift"
                )
        datasets[dataset] = record
    return datasets


def _validated_niah_grid(
    raw_grid: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    expected_ids = tuple(row[0] for row in _NIAH_ROWS)
    _require_exact_keys(raw_grid, frozenset(expected_ids), "NIAH grid")
    grid: dict[str, Mapping[str, Any]] = {}
    for cell_id in expected_ids:
        cell = _mapping_field(raw_grid, cell_id, "NIAH grid")
        _require_exact_keys(
            cell,
            frozenset(
                {"example_count", "methods", "paired_vanilla_minus_baseline"}
            ),
            f"NIAH cell {cell_id}",
        )
        example_count = _int_field(cell, "example_count", cell_id)
        if example_count <= 0:
            raise ValueError(f"NIAH cell {cell_id} must be nonempty")
        methods = _mapping_field(cell, "methods", f"NIAH cell {cell_id}")
        _require_exact_keys(
            methods,
            frozenset(_METHODS),
            f"NIAH cell {cell_id} methods",
        )
        means: dict[str, float] = {}
        for method in _METHODS:
            metrics = _mapping_field(
                methods,
                method,
                f"NIAH cell {cell_id}",
            )
            _require_exact_keys(
                metrics,
                frozenset({"accuracy"}),
                f"NIAH cell {cell_id}.{method}",
            )
            means[method] = _validate_score_metric_summary(
                _mapping_field(
                    metrics,
                    "accuracy",
                    f"NIAH cell {cell_id}.{method}",
                ),
                expected_count=example_count,
                label=f"NIAH cell {cell_id}.{method}.accuracy",
            )
        paired = _mapping_field(
            cell,
            "paired_vanilla_minus_baseline",
            f"NIAH cell {cell_id}",
        )
        _require_exact_keys(
            paired,
            frozenset({"accuracy"}),
            f"NIAH cell {cell_id} paired",
        )
        delta = _validate_paired_summary(
            _mapping_field(
                paired,
                "accuracy",
                f"NIAH cell {cell_id} paired",
            ),
            expected_count=example_count,
            label=f"NIAH cell {cell_id}.paired.accuracy",
        )
        expected_delta = means["vanilla_prefill"] - means["baseline_prefill"]
        if not _numbers_match(delta, expected_delta):
            raise ValueError(f"NIAH cell {cell_id} delta drift")
        grid[cell_id] = cell
    return grid


def _validate_score_metric_summary(
    summary: Mapping[str, Any],
    *,
    expected_count: int,
    label: str,
) -> float:
    _require_exact_keys(
        summary,
        frozenset(
            {"example_count", "invalid_parser_score_sum", "mean", "sum"}
        ),
        label,
    )
    if _int_field(summary, "example_count", label) != expected_count:
        raise ValueError(f"{label} count drift")
    mean = _number_field(summary, "mean", label)
    total = _number_field(summary, "sum", label)
    invalid_parser_score_sum = _number_field(
        summary,
        "invalid_parser_score_sum",
        label,
    )
    if invalid_parser_score_sum != 0.0:
        raise ValueError(f"{label} credits an invalid parsed answer")
    if not 0.0 <= mean <= 1.0 or not 0.0 <= total <= expected_count:
        raise ValueError(f"{label} score bounds drift")
    if not _numbers_match(mean, total / expected_count):
        raise ValueError(f"{label} mean/sum drift")
    return mean


def _validate_paired_summary(
    summary: Mapping[str, Any],
    *,
    expected_count: int,
    label: str,
) -> float:
    _require_exact_keys(
        summary,
        frozenset(
            {"bootstrap_ci95", "example_count", "mean", "seed_sha256"}
        ),
        label,
    )
    if _int_field(summary, "example_count", label) != expected_count:
        raise ValueError(f"{label} count drift")
    _sha256_field(summary, "seed_sha256", label)
    mean = _number_field(summary, "mean", label)
    ci = _mapping_field(summary, "bootstrap_ci95", label)
    _require_exact_keys(
        ci,
        frozenset({"draws", "lower", "upper"}),
        f"{label} CI",
    )
    if _int_field(ci, "draws", label) <= 0:
        raise ValueError(f"{label} bootstrap draw count drift")
    lower = _number_field(ci, "lower", label)
    upper = _number_field(ci, "upper", label)
    if not -1.0 <= mean <= 1.0 or not -1.0 <= lower <= upper <= 1.0:
        raise ValueError(f"{label} bounds drift")
    return mean


def _validate_parser_counts(
    counts: Mapping[str, Any],
    *,
    expected_count: int,
    label: str,
) -> None:
    _require_exact_keys(
        counts,
        frozenset(_PARSER_STATUS_ORDER),
        f"{label} parser counts",
    )
    values = [
        _int_field(counts, status, label) for status in _PARSER_STATUS_ORDER
    ]
    if any(value < 0 for value in values) or sum(values) != expected_count:
        raise ValueError(f"{label} parser count closure drift")


def _validate_scorer_contracts(value: Any) -> None:
    if not isinstance(value, list) or len(value) != len(_METRICS_BY_DATASET):
        raise ValueError("scorer contracts must cover the four datasets")
    observed_datasets = [
        item.get("dataset") if isinstance(item, Mapping) else None
        for item in value
    ]
    if observed_datasets != list(_METRICS_BY_DATASET):
        raise ValueError("scorer contract order drift")
    for raw_contract in value:
        contract = _mapping_value(raw_contract, "scorer contract")
        dataset = _string_field(contract, "dataset", "scorer contract")
        _require_exact_keys(
            contract,
            _SCORER_KEYS,
            f"scorer contract {dataset}",
        )
        if contract.get("publication_approved") is not True:
            raise ValueError(
                f"scorer contract {dataset} is not publication approved"
            )
        metrics = contract.get("metric_names")
        if (
            not isinstance(metrics, list)
            or metrics != list(_METRICS_BY_DATASET[dataset])
        ):
            raise ValueError(f"scorer contract {dataset} metric order drift")
        _sha256_field(
            contract,
            "answer_parser_digest",
            f"scorer contract {dataset}",
        )
        for field in (
            "answer_parser_id",
            "answer_parser_plugin_path",
            "answer_parser_version",
            "plugin_path",
            "scorer_id",
            "scorer_version",
        ):
            _string_field(contract, field, f"scorer contract {dataset}")


def _render_status(coverage: Mapping[str, Any]) -> str:
    latency_jobs = _int_field(coverage, "latency_job_count", "coverage")
    scored = _int_field(coverage, "full_score_identity_count", "coverage")
    checked = _int_field(coverage, "checked_cache_request_count", "coverage")
    latency_cache = _int_field(coverage, "latency_cache_request_count", "coverage")
    full_score_cache = _int_field(
        coverage, "full_score_cache_request_count", "coverage"
    )
    cold = _int_field(coverage, "cold_attested_request_count", "coverage")
    return (
        "> **Status: vLLM "
        + PUBLICATION_CAMPAIGN_ENGINE_VERSION
        + " campaign published.** The sanitized campaign report and its "
        "report-bound publication gate passed. The gate is an integrity gate: "
        "it checks evidence bindings and fixed coverage counts, sets no "
        "quality or latency threshold, and latency results are "
        "estimation-only.\n>\n"
        f"> Coverage: {latency_jobs:,} latency jobs; {scored:,} distinct "
        f"full-score examples; {checked:,} Vanilla KV cache requests counted "
        f"by the gate ({latency_cache:,} latency and {full_score_cache:,} "
        f"full-score), of which {cold:,} latency requests are cold-read "
        "attested.\n>\n"
        "> Only latency requests carry cold-read attestations. The design "
        f"requires {_MIN_COLD_ATTESTED_REQUEST_COUNT:,} to "
        f"{_MAX_COLD_ATTESTED_REQUEST_COUNT:,} of the {latency_cache:,}: the "
        f"{_STORAGE_SETTING_CACHE_REQUEST_COUNT:,} RAM-storage requests are "
        "served from the in-memory payload cache and never read cold, and the "
        f"{_STORAGE_SETTING_CACHE_REQUEST_COUNT:,} Unity Catalog requests load "
        "through the mounted path and need not attest one."
    )


def _render_campaign_disclosures(
    report: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> str:
    ledger = _mapping_field(report, "ledger", "publication report")
    record_type = report.get("record_type")
    if record_type == PUBLICATION_CAMPAIGN_VLLM_0300_REPORT_RECORD_TYPE:
        _require_exact_keys(
            ledger,
            _VLLM_0300_REPORT_LEDGER_KEYS,
            "publication ledger projection",
        )
        rows = _multi_plan_disclosure_rows(report, ledger, coverage)
    elif record_type == PUBLICATION_CAMPAIGN_REPORT_RECORD_TYPE:
        _require_exact_keys(
            ledger,
            _REPORT_LEDGER_KEYS,
            "publication ledger projection",
        )
        rows = _single_plan_disclosure_rows(report, ledger, coverage)
    else:
        raise ValueError("publication report record type drift")
    return (
        "> **Campaign ledger disclosures.** Rendered from the sealed report's "
        "Databricks cluster-hour ledger projection and latency source fields. "
        "They change no metric on this page; they record the campaign's "
        "budget, retried and abandoned work, and how latency jobs are bound to "
        "the ledger.\n\n"
        + _markdown_table(
            ("Disclosure", "Report value", "Meaning"),
            ("---", "---", "---"),
            rows,
        )
    )


def _multi_plan_disclosure_rows(
    report: Mapping[str, Any],
    ledger: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> list[tuple[str, str, str]]:
    label = "publication ledger projection"
    cap_raises: list[str] = []
    for index, raw_raise in enumerate(_list_field(ledger, "cap_raises", label)):
        raise_label = f"publication ledger cap raise {index}"
        cap_raise = _mapping_value(raw_raise, raise_label)
        _require_exact_keys(cap_raise, _CAP_RAISE_KEYS, raise_label)
        from_cap = _number_field(cap_raise, "from_cap_cluster_hours", raise_label)
        to_cap = _number_field(cap_raise, "to_cap_cluster_hours", raise_label)
        reservations = _int_field(cap_raise, "reservation_count", raise_label)
        receipts = _int_field(cap_raise, "submission_receipt_count", raise_label)
        terminals = _int_field(cap_raise, "terminal_actual_count", raise_label)
        cap_raises.append(
            f"Raised from {_cluster_hours(from_cap, 0)} to "
            f"{_cluster_hours(to_cap, 0)} after "
            f"{_counted(reservations, 'reservation')}, "
            f"{_counted(receipts, 'receipt')}, and "
            f"{_counted(terminals, 'terminal actual')}"
        )
    final_cap = _number_field(
        _mapping_field(ledger, "full_score_terminal_prefix", label),
        "cap_cluster_hours",
        "full-score terminal prefix",
    )
    qualification_retries, qualification_hours = _attempt_hours(
        ledger, "qualification_capacity_retries", _ATTEMPT_HOURS_KEYS
    )
    full_score_retries, full_score_retry_hours = _attempt_hours(
        ledger, "full_score_capacity_retries", _ATTEMPT_HOURS_KEYS
    )
    abandoned, abandoned_hours = _attempt_hours(
        ledger, "full_score_abandoned_attempts", _ABANDONED_ATTEMPT_KEYS
    )
    closed_consumers = _int_field(
        _mapping_field(ledger, "full_score_abandoned_attempts", label),
        "closed_consumer_phase_count",
        "full-score abandoned attempts",
    )
    recovered_seeds, recovered_hours = _attempt_hours(
        ledger, "latency_recovered_balance", _ATTEMPT_HOURS_KEYS
    )
    phases = _int_field(coverage, "full_score_phase_count", "coverage")
    full_score_attempts = _full_score_attempts_after_latency(ledger)
    if full_score_attempts != phases + full_score_retries:
        raise ValueError("publication disclosure full-score attempt count drift")

    provenance = _mapping_field(report, "latency_provenance", "publication report")
    _require_exact_keys(
        provenance,
        _LATENCY_PROVENANCE_KEYS,
        "publication latency provenance",
    )
    binding_counts = _mapping_field(
        provenance,
        "ledger_binding_counts",
        "publication latency provenance",
    )
    _require_exact_keys(
        binding_counts,
        frozenset(PUBLICATION_LATENCY_LEDGER_BINDINGS),
        "publication latency ledger binding counts",
    )
    bound = {
        binding: _int_field(binding_counts, binding, "latency ledger binding counts")
        for binding in PUBLICATION_LATENCY_LEDGER_BINDINGS
    }
    latency_jobs = _int_field(coverage, "latency_job_count", "coverage")
    if sum(bound.values()) != latency_jobs:
        raise ValueError("publication disclosure latency binding count drift")
    plan_count = _int_field(
        provenance, "execution_plan_count", "publication latency provenance"
    )
    superseded = _int_field(
        provenance, "superseded_attempt_count", "publication latency provenance"
    )
    plan_set_sha256 = _sha256_field(
        provenance, "plan_set_sha256", "publication latency provenance"
    )
    return [
        (
            "Cluster-hour budget",
            "; ".join(cap_raises)
            if cap_raises
            else f"{_cluster_hours(final_cap, 0)} cap, never raised",
            "Budget extensions recorded in the campaign ledger; every ledger "
            "prefix before a raise keeps its earlier cap",
        ),
        _ledger_accounting_row(ledger, final_cap),
        (
            "GPU qualification capacity retries",
            f"{_counted(qualification_retries, 'attempt')}, "
            f"{_cluster_hours(qualification_hours)}",
            "Qualification jobs relaunched after a Databricks capacity failure",
        ),
        (
            "Full-score capacity retries",
            f"{_counted(full_score_retries, 'attempt')}, "
            f"{_cluster_hours(full_score_retry_hours)}",
            f"Phases relaunched after a Databricks capacity failure; the "
            f"{_counted(full_score_attempts, 'full-score ledger attempt')} "
            f"after the latency prefix cover the {_counted(phases, 'phase')}, "
            "and only each phase's "
            "final attempt is scored",
        ),
        (
            "Abandoned full-score attempts",
            f"{_counted(abandoned, 'attempt')}, "
            f"{_cluster_hours(abandoned_hours)}; "
            f"{_counted(closed_consumers, 'closed consumer phase')}",
            "Earlier full-score chains the campaign restarted over before the "
            "latency prefix; the waves their consumer phases scored are not "
            "used in this report",
        ),
        (
            "Latency recovered balance",
            f"{_counted(recovered_seeds, 'seed')}, "
            f"{_cluster_hours(recovered_hours)}",
            "Terminal-only ledger seeds restoring hours spent before the "
            "ledger was recreated",
        ),
        (
            "Latency jobs with their own ledger receipt",
            f"{bound[PUBLICATION_LATENCY_LEDGER_RECEIPT_BINDING]:,} of "
            f"{latency_jobs:,}",
            "Receipt recorded when the job was submitted",
        ),
        (
            "Latency jobs with a recovered receipt",
            f"{bound[PUBLICATION_LATENCY_RECOVERED_RECEIPT_BINDING]:,} of "
            f"{latency_jobs:,}",
            "Finished before the ledger was recreated; receipt re-recorded by "
            "run ID under a timeout-only stand-in payload",
        ),
        (
            "Latency jobs bound only by the recovered balance",
            f"{bound[PUBLICATION_LATENCY_RECOVERED_BALANCE_BINDING]:,} of "
            f"{latency_jobs:,}",
            "Finished before the ledger was recreated and have no ledger "
            "record; accepted only because the recovered-balance seeds cover "
            "the spent hours",
        ),
        (
            "Latency launch plans",
            f"{_counted(plan_count, 'execution plan')}; "
            f"{_counted(superseded, 'superseded attempt')}",
            "Jobs are joined across the plans they ran under (capacity retries "
            "and reopened cells); superseded attempts are not used. Plan-set "
            f"SHA-256 `{plan_set_sha256}`",
        ),
    ]


def _single_plan_disclosure_rows(
    report: Mapping[str, Any],
    ledger: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> list[tuple[str, str, str]]:
    final_cap = _number_field(
        _mapping_field(
            ledger,
            "full_score_terminal_prefix",
            "publication ledger projection",
        ),
        "cap_cluster_hours",
        "full-score terminal prefix",
    )
    phases = _int_field(coverage, "full_score_phase_count", "coverage")
    full_score_attempts = _full_score_attempts_after_latency(ledger)
    if full_score_attempts != phases:
        raise ValueError("publication disclosure full-score attempt count drift")
    plan_sha256 = _sha256_field(
        _mapping_field(
            _mapping_field(report, "source_bindings", "publication report"),
            "latency_execution_plan",
            "publication source bindings",
        ),
        "closed_record_sha256",
        "latency execution plan binding",
    )
    return [
        (
            "Cluster-hour budget",
            f"{_cluster_hours(final_cap, 0)} cap",
            "The 0.27.1 report schema requires the original cap at both "
            "report ledger prefixes",
        ),
        _ledger_accounting_row(ledger, final_cap),
        (
            "Full-score ledger attempts",
            f"{full_score_attempts:,} after the latency prefix for "
            f"{_counted(phases, 'phase')}",
            "The 0.27.1 report schema admits no capacity retry",
        ),
        (
            "Latency launch plans",
            "1 execution plan",
            f"Execution plan SHA-256 `{plan_sha256}`",
        ),
        (
            "Qualification retries, abandoned chains, recovered balance",
            "Not recorded",
            "The 0.27.1 report schema has no fields for them",
        ),
    ]


def _ledger_accounting_row(
    ledger: Mapping[str, Any],
    final_cap: float,
) -> tuple[str, str, str]:
    label = "publication ledger projection"
    accounted = _number_field(ledger, "final_accounted_cluster_hours", label)
    remaining = _number_field(ledger, "final_remaining_cluster_hours", label)
    active_tasks = _int_field(ledger, "final_active_reserved_task_count", label)
    headroom = _number_field(
        ledger, "required_unreserved_headroom_cluster_hours", label
    )
    return (
        "Final ledger accounting",
        f"{_cluster_hours(accounted)} accounted; {_cluster_hours(remaining)} "
        f"remaining of a {_cluster_hours(final_cap, 0)} cap; "
        f"{_counted(active_tasks, 'active reserved task')}",
        "The ledger closed with nothing reserved and at least "
        f"{_cluster_hours(headroom, 0)} unreserved",
    )


def _attempt_hours(
    ledger: Mapping[str, Any],
    field: str,
    keys: frozenset[str],
) -> tuple[int, float]:
    value = _mapping_field(ledger, field, "publication ledger projection")
    _require_exact_keys(value, keys, f"publication ledger {field}")
    return (
        _int_field(value, "attempt_count", field),
        _number_field(value, "cluster_hours", field),
    )


def _full_score_attempts_after_latency(ledger: Mapping[str, Any]) -> int:
    label = "publication ledger projection"
    return _int_field(
        _mapping_field(ledger, "full_score_terminal_prefix", label),
        "reservation_count",
        "full-score terminal prefix",
    ) - _int_field(
        _mapping_field(ledger, "latency_terminal_prefix", label),
        "reservation_count",
        "latency terminal prefix",
    )


def _counted(count: int, noun: str) -> str:
    return f"{count:,} {noun}" + ("" if count == 1 else "s")


def _cluster_hours(value: float, digits: int = 2) -> str:
    return f"{_finite_number(value, 'cluster hours'):,.{digits}f} h"


def _render_core_latency(cells: Mapping[str, Mapping[str, Any]]) -> str:
    rows: list[tuple[str, ...]] = []
    for cell_id, method, input_tokens, concurrency in _CORE_LATENCY_ROWS:
        cell = cells[cell_id]
        rows.append(
            (
                _method_label(method),
                f"{input_tokens // 1024}k",
                str(concurrency),
                _decimal(
                    _number_field(cell, "p50_ttft_seconds", cell_id),
                    4,
                ),
                _decimal(
                    _number_field(cell, "p95_ttft_seconds", cell_id),
                    4,
                ),
                _decimal(
                    _number_field(
                        cell,
                        "p50_time_to_completion_seconds",
                        cell_id,
                    ),
                    4,
                ),
                _decimal(
                    _number_field(
                        cell,
                        "p95_time_to_completion_seconds",
                        cell_id,
                    ),
                    4,
                ),
                _decimal(
                    _number_field(
                        cell,
                        "p50_decode_tokens_per_second",
                        cell_id,
                    ),
                    2,
                ),
                str(
                    _int_field(
                        cell,
                        "configured_closed_loop_concurrency",
                        cell_id,
                    )
                ),
                _gib(
                    _int_field(
                        cell,
                        "peak_gpu_process_memory_bytes",
                        cell_id,
                    )
                ),
                _gib(
                    _int_field(cell, "peak_host_memory_used_bytes", cell_id)
                ),
                _gib(
                    _int_field(cell, "peak_process_tree_rss_bytes", cell_id)
                ),
            )
        )
    return _markdown_table(
        (
            "Method",
            "Input context",
            "Concurrency setting",
            "P50 TTFT (s)",
            "P95 TTFT (s)",
            "P50 TTC (s, 256 toks)",
            "P95 TTC (s, 256 toks)",
            "P50 decode tok/s",
            "Configured closed-loop concurrency",
            "Peak GPU process memory",
            "Peak host memory",
            "Peak process-tree RSS",
        ),
        (
            "---",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
        ),
        rows,
    )


def _render_latency_estimands(
    estimates: Mapping[str, Mapping[str, Any]],
) -> str:
    rows: list[tuple[str, ...]] = []
    for estimand_id, *_identity, comparison, setting in _ESTIMAND_ROWS:
        estimate = estimates[estimand_id]
        metrics = _mapping_field(estimate, "metrics", estimand_id)
        ttft = _mapping_field(metrics, "ttft", estimand_id)
        ttc = _mapping_field(metrics, "time_to_completion", estimand_id)
        rows.append(
            (
                comparison,
                setting,
                _decimal(
                    _number_field(ttft, "geometric_mean_speedup", estimand_id),
                    4,
                ),
                _interval(
                    _list_field(ttft, "confidence_interval_95", estimand_id),
                    4,
                ),
                _decimal(
                    _number_field(ttc, "geometric_mean_speedup", estimand_id),
                    4,
                ),
                _interval(
                    _list_field(ttc, "confidence_interval_95", estimand_id),
                    4,
                ),
                "Five matched blocks",
            )
        )
    return _markdown_table(
        (
            "Treatment vs reference",
            "Setting",
            "TTFT geometric speedup",
            "TTFT 95% CI",
            "TTC geometric speedup",
            "TTC 95% CI",
            "Status",
        ),
        ("---", "---", "---:", "---:", "---:", "---:", "---"),
        rows,
    )


def _render_dataset_scores(
    datasets: Mapping[str, Mapping[str, Any]],
) -> str:
    rows: list[tuple[str, ...]] = []
    for dataset, metric_name, dataset_label, metric_label in _SCORE_ROWS:
        record = datasets[dataset]
        example_count = _int_field(record, "example_count", dataset)
        methods = _mapping_field(record, "methods", dataset)
        baseline = _mapping_field(methods, "baseline_prefill", dataset)
        vanilla = _mapping_field(methods, "vanilla_prefill", dataset)
        baseline_metric = _mapping_field(
            _mapping_field(baseline, "metrics", dataset),
            metric_name,
            dataset,
        )
        vanilla_metric = _mapping_field(
            _mapping_field(vanilla, "metrics", dataset),
            metric_name,
            dataset,
        )
        paired = _mapping_field(
            _mapping_field(
                record,
                "paired_vanilla_minus_baseline",
                dataset,
            ),
            metric_name,
            dataset,
        )
        ci = _mapping_field(paired, "bootstrap_ci95", dataset)
        rows.append(
            (
                dataset_label,
                metric_label,
                str(example_count),
                _quality_decimal(
                    _number_field(baseline_metric, "mean", dataset),
                ),
                _parser_counts(
                    _mapping_field(
                        baseline,
                        "parser_status_counts",
                        dataset,
                    )
                ),
                _quality_decimal(
                    _number_field(vanilla_metric, "mean", dataset),
                ),
                _parser_counts(
                    _mapping_field(
                        vanilla,
                        "parser_status_counts",
                        dataset,
                    )
                ),
                _quality_decimal(_number_field(paired, "mean", dataset)),
                _quality_ci_mapping(ci),
            )
        )
    unsupported = "N/A (runner not implemented)"
    rows.extend(
        (
            dataset,
            unsupported,
            unsupported,
            unsupported,
            unsupported,
            unsupported,
            unsupported,
            unsupported,
            unsupported,
        )
        for dataset in ("LongBench v2", "RULER")
    )
    return _markdown_table(
        (
            "Dataset",
            "Governed metric",
            "n",
            "Baseline",
            "Baseline parser-status counts",
            "Vanilla KV",
            "Vanilla parser-status counts",
            "Vanilla − Baseline",
            "Paired 95% CI",
        ),
        (
            "---",
            "---",
            "---:",
            "---:",
            "---",
            "---:",
            "---",
            "---:",
            "---:",
        ),
        rows,
    )


def _render_niah_grid(grid: Mapping[str, Mapping[str, Any]]) -> str:
    rows: list[tuple[str, ...]] = []
    for cell_id, context, position in _NIAH_ROWS:
        cell = grid[cell_id]
        methods = _mapping_field(cell, "methods", cell_id)
        baseline = _mapping_field(
            _mapping_field(methods, "baseline_prefill", cell_id),
            "accuracy",
            cell_id,
        )
        vanilla = _mapping_field(
            _mapping_field(methods, "vanilla_prefill", cell_id),
            "accuracy",
            cell_id,
        )
        paired = _mapping_field(
            _mapping_field(
                cell,
                "paired_vanilla_minus_baseline",
                cell_id,
            ),
            "accuracy",
            cell_id,
        )
        rows.append(
            (
                context,
                position,
                str(_int_field(cell, "example_count", cell_id)),
                _quality_decimal(_number_field(baseline, "mean", cell_id)),
                _quality_decimal(_number_field(vanilla, "mean", cell_id)),
                _quality_decimal(_number_field(paired, "mean", cell_id)),
                _quality_ci_mapping(
                    _mapping_field(paired, "bootstrap_ci95", cell_id),
                ),
            )
        )
    return _markdown_table(
        (
            "Context",
            "Needle position",
            "n",
            "Baseline accuracy",
            "Vanilla KV accuracy",
            "Vanilla − Baseline",
            "Paired 95% CI",
        ),
        ("---", "---:", "---:", "---:", "---:", "---:", "---:"),
        rows,
    )


def _render_precision(cells: Mapping[str, Mapping[str, Any]]) -> str:
    return _ablation_table(
        "Document KV payload/runtime KV",
        (
            (
                "Q8 / `fp8_e5m2`",
                cells["core-vanilla_prefill-16384-c4"],
                "Core 16k, concurrency-4 Vanilla anchor",
            ),
            (
                "bf16 / bf16",
                cells["auxiliary-precision-bf16"],
                "Implemented five-block refresh cell",
            ),
            (
                "Packed Q4",
                None,
                "N/A (not implemented); no packed-Q4 serving contract",
            ),
        ),
    )


def _render_storage(cells: Mapping[str, Mapping[str, Any]]) -> str:
    return _ablation_table(
        "Storage tier",
        (
            (
                "Local NVMe disk",
                cells["auxiliary-storage-disk"],
                "Five-block control; cold reads attested, page-cache residency unmeasured",
            ),
            (
                "RAM",
                cells["auxiliary-storage-ram"],
                "Prewarmed 16-GiB payload cache; priming also warmed the serving process",
            ),
            (
                "Unity Catalog mounted path",
                cells["auxiliary-storage-uc"],
                "OS eviction requested, residency unmeasured; backend cache unproven",
            ),
            (
                "Hybrid RAM/disk/Unity Catalog",
                None,
                "N/A (not implemented); combined serving policy unsupported",
            ),
        ),
    )


def _render_hardware(cells: Mapping[str, Mapping[str, Any]]) -> str:
    return _ablation_table(
        "Hardware",
        (
            (
                "AWS g6/L4, `g6.8xlarge`",
                cells["core-vanilla_prefill-16384-c4"],
                "Core 16k, concurrency-4 Vanilla anchor",
            ),
            (
                "AWS g5/A10G, `g5.8xlarge`",
                cells["auxiliary-hardware-a10g"],
                "Implemented five-block compatibility refresh cell; "
                "qualified vLLM `gpu_memory_utilization="
                f"{GPU_QUALIFICATION_A10G_GPU_MEMORY_UTILIZATION:.2f}` "
                "against an A10G-only "
                f"{GPU_QUALIFICATION_A10G_MIN_PEAK_HEADROOM_BYTES / 1024**3:.2f} GiB "
                "peak-headroom gate (other GPUs: "
                f"{GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES / 1024**3:.0f} GiB)",
            ),
        ),
    )


def _render_platform() -> str:
    return _markdown_table(
        ("Serving platform", "Result", "Status"),
        ("---", "---", "---"),
        (
            (
                "vLLM " + PUBLICATION_CAMPAIGN_ENGINE_VERSION,
                "Governed campaign published",
                "Report-bound integrity gate passed (no quality threshold)",
            ),
            (
                "SGLang",
                "N/A (Q8 pre-RoPE serving path not implemented)",
                "Unsupported for the main campaign",
            ),
        ),
    )


def _render_resource_cache(
    cells: Mapping[str, Mapping[str, Any]],
) -> str:
    rows: list[tuple[str, ...]] = []
    for cell_id in _DESCRIPTIVE_CELL_ORDER:
        cell = cells[cell_id]
        cache = _mapping_field(cell, "cache_telemetry", cell_id)
        backend = _gib(_int_field(cache, "backend_bytes_read", cell_id))
        expected_backend = _gib(
            _int_field(cache, "expected_backend_bytes_read", cell_id)
        )
        evictions = (
            f"{_int_field(cache, 'eviction_requested_count', cell_id)} / "
            f"{_int_field(cache, 'eviction_succeeded_count', cell_id)}"
        )
        payload_cache = (
            f"{_int_field(cache, 'payload_cache_hit_count', cell_id)} / "
            f"{_int_field(cache, 'payload_cache_miss_count', cell_id)}"
        )
        rows.append(
            (
                _resource_cell_label(cell_id),
                _decimal(
                    _number_field(
                        cell,
                        "mean_gpu_utilization_percent",
                        cell_id,
                    ),
                    2,
                ),
                _decimal(
                    _number_field(
                        cell,
                        "peak_gpu_utilization_percent",
                        cell_id,
                    ),
                    2,
                ),
                str(
                    _int_field(
                        cell,
                        "gpu_utilization_sample_count",
                        cell_id,
                    )
                ),
                _gib(
                    _int_field(
                        cell,
                        "peak_gpu_process_memory_bytes",
                        cell_id,
                    )
                ),
                _gib(
                    _int_field(cell, "peak_host_memory_used_bytes", cell_id)
                ),
                _gib(
                    _int_field(cell, "peak_process_tree_rss_bytes", cell_id)
                ),
                str(_int_field(cache, "load_count", cell_id)),
                f"{backend} / {expected_backend}",
                str(
                    _int_field(cache, "cold_read_attested_count", cell_id)
                ),
                evictions,
                str(_int_field(cache, "mounted_path_load_count", cell_id)),
                payload_cache,
                str(
                    _int_field(
                        cache,
                        "storage_materialization_count",
                        cell_id,
                    )
                ),
            )
        )
    return _markdown_table(
        (
            "Cell",
            "Mean GPU util (%)",
            "Peak GPU util (%)",
            "GPU util samples",
            "Peak GPU process memory",
            "Peak host memory",
            "Peak process-tree RSS",
            "Connector loads",
            "Backend bytes read / expected",
            "Cold-read attestations",
            "Evictions requested / succeeded",
            "Mounted-path loads",
            "Payload-cache hits / misses",
            "Storage materializations",
        ),
        (
            "---",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
        ),
        rows,
    )


def _ablation_table(
    first_header: str,
    rows: Sequence[tuple[str, Mapping[str, Any] | None, str]],
) -> str:
    rendered: list[tuple[str, ...]] = []
    for label, cell, status in rows:
        if cell is None:
            p50_ttft = "N/A (not implemented)"
            p50_ttc = "N/A (not implemented)"
        else:
            cell_id = _string_field(cell, "cell_id", "ablation cell")
            p50_ttft = _decimal(
                _number_field(cell, "p50_ttft_seconds", cell_id),
                4,
            )
            p50_ttc = _decimal(
                _number_field(
                    cell,
                    "p50_time_to_completion_seconds",
                    cell_id,
                ),
                4,
            )
        rendered.append((label, p50_ttft, p50_ttc, status))
    return _markdown_table(
        (first_header, "P50 TTFT (s)", "P50 TTC (s)", "Status"),
        ("---", "---:", "---:", "---"),
        rendered,
    )


def _resource_cell_label(cell_id: str) -> str:
    core = {row[0]: row for row in _CORE_LATENCY_ROWS}.get(cell_id)
    if core is not None:
        _cell_id, method, tokens, concurrency = core
        return f"{_method_label(method)} {tokens // 1024}k c{concurrency}"
    labels = {
        "auxiliary-precision-bf16": "BF16 / bf16, 16k c4 L4/NVMe",
        "auxiliary-storage-disk": "Disk, 16k c4 L4",
        "auxiliary-storage-ram": "RAM, 16k c4 L4",
        "auxiliary-storage-uc": "Unity Catalog, 16k c4 L4",
        "auxiliary-hardware-a10g": "A10G, 16k c4 local NVMe",
    }
    try:
        return labels[cell_id]
    except KeyError as exc:  # pragma: no cover - frozen internal guard.
        raise RuntimeError("unknown publication resource cell") from exc


def _method_label(method_id: str) -> str:
    if method_id == "baseline_prefill":
        return "Baseline"
    if method_id == "vanilla_prefill":
        return "Vanilla&nbsp;KV"
    raise RuntimeError("unknown frozen publication method")


def _parser_counts(counts: Mapping[str, Any]) -> str:
    return "; ".join(
        f"{status}={_int_field(counts, status, 'parser counts')}"
        for status in _PARSER_STATUS_ORDER
    )


def _quality_ci_mapping(ci: Mapping[str, Any]) -> str:
    lower = _quality_decimal(
        _number_field(ci, "lower", "quality confidence interval")
    )
    upper = _quality_decimal(
        _number_field(ci, "upper", "quality confidence interval")
    )
    return f"[{lower}, {upper}]"


def _interval(values: list[Any], digits: int) -> str:
    if len(values) != 2:
        raise ValueError("confidence interval must contain two values")
    lower = _decimal(
        _finite_number(values[0], "confidence interval lower"),
        digits,
    )
    upper = _decimal(
        _finite_number(values[1], "confidence interval upper"),
        digits,
    )
    return f"[{lower}, {upper}]"


def _decimal(value: float, digits: int) -> str:
    rendered = f"{_finite_number(value, 'numeric table cell'):.{digits}f}"
    if rendered.startswith("-") and float(rendered) == 0.0:
        return rendered[1:]
    return rendered


def _quality_decimal(value: float) -> str:
    numeric = _finite_number(value, "quality table cell")
    rendered = _decimal(numeric, 6)
    if numeric != 0.0 and float(rendered) == 0.0:
        return f"{numeric:.6e}"
    return rendered


def _gib(value: int) -> str:
    if type(value) is not int or value < 0:
        raise ValueError("byte count must be a nonnegative integer")
    return f"{_decimal(value / (1024**3), 3)} GiB ({value:,} bytes)"


def _markdown_table(
    headers: Sequence[str],
    alignments: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    width = len(headers)
    if width == 0 or len(alignments) != width:
        raise RuntimeError("Markdown table schema is invalid")
    normalized_rows = [tuple(row) for row in rows]
    if any(len(row) != width for row in normalized_rows):
        raise RuntimeError("Markdown table row width drift")
    cells = (
        *headers,
        *alignments,
        *(cell for row in normalized_rows for cell in row),
    )
    if any(
        not isinstance(cell, str)
        or "\n" in cell
        or "\r" in cell
        or "|" in cell
        for cell in cells
    ):
        raise ValueError("Markdown table cells must be one safe line")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row) + " |" for row in normalized_rows
    )
    return "\n".join(lines)


def _ordered_mapping_rows(
    values: list[Any],
    *,
    id_field: str,
    expected_ids: Sequence[str],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    rows = [_mapping_value(value, label) for value in values]
    observed_ids = [row.get(id_field) for row in rows]
    if observed_ids != list(expected_ids):
        raise ValueError(f"{label} order or coverage drift")
    return {
        expected_id: row
        for expected_id, row in zip(expected_ids, rows, strict=True)
    }


def _detached_json_object(value: Any, label: str) -> dict[str, Any]:
    detached = _detached_json_value(value, label)
    if not isinstance(detached, dict):
        raise TypeError(f"{label} must be a JSON object")
    return detached


def _validated_pair_snapshots(
    report_record: Mapping[str, Any],
    gate_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _detached_json_object(report_record, "publication report")
    gate = _detached_json_object(gate_record, "publication gate")
    validate_vllm_0271_publication_report_pair(
        _detached_json_object(report, "publication report validation copy"),
        _detached_json_object(gate, "publication gate validation copy"),
    )
    return report, gate


def _detached_json_value(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        detached: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} contains a non-string JSON key")
            if key in detached:
                raise ValueError(f"{label} contains a duplicate JSON key")
            detached[key] = _detached_json_value(item, f"{label}.{key}")
        return detached
    if type(value) is list:
        return [
            _detached_json_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"{label} contains a non-JSON or non-finite value")


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{label} schema drift")


def _mapping_value(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _mapping_field(
    value: Mapping[str, Any],
    field: str,
    label: str,
) -> Mapping[str, Any]:
    return _mapping_value(value.get(field), f"{label}.{field}")


def _list_field(
    value: Mapping[str, Any],
    field: str,
    label: str,
) -> list[Any]:
    result = value.get(field)
    if not isinstance(result, list):
        raise TypeError(f"{label}.{field} must be an array")
    return result


def _string_field(
    value: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise TypeError(f"{label}.{field} must be a nonempty string")
    return result


def _int_field(value: Mapping[str, Any], field: str, label: str) -> int:
    result = value.get(field)
    if type(result) is not int:
        raise TypeError(f"{label}.{field} must be an integer")
    return result


def _number_field(value: Mapping[str, Any], field: str, label: str) -> float:
    return _finite_number(value.get(field), f"{label}.{field}")


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise TypeError(f"{label} must be a finite JSON number")
    return float(value)


def _sha256_field(
    value: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    digest = value.get(field)
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label}.{field} must be a lowercase SHA-256 digest")
    return digest


def _numbers_match(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12 * max(1.0, abs(left), abs(right))


__all__ = [
    "PUBLICATION_TABLE_REGION_MARKERS",
    "PUBLICATION_TABLE_SECTION_ORDER",
    "render_vllm_0271_publication_appendix_readme",
    "render_vllm_0271_publication_table_regions",
    "replace_vllm_0271_publication_table_regions",
    "validate_vllm_0271_publication_table_regions",
]
