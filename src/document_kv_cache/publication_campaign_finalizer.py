"""Fail-closed finalization for the vLLM 0.27.1 publication campaign.

This leaf module recomputes the independent latency and full-score branches,
joins them to one uninterrupted ledger lineage, projects sanitized tables, and
binds the result to the standard benchmark publication-gate record.
"""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

from document_kv_cache.benchmark_gates import (
    BenchmarkPublicationGateResult,
    benchmark_publication_gate_to_record,
)
from document_kv_cache.benchmarks import (
    FINAL_ANSWER_PARSER_STATUSES,
    NIAH_CELL_IDS,
    SUPPORTED_V1_DATASETS,
    default_dataset_scorer_registry,
)
from document_kv_cache.databricks_resource_ledger import (
    DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS,
    MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    databricks_ledger_path_sha256,
    databricks_ledger_prefix,
    databricks_ledger_prefix_at_counts,
    databricks_ledger_prefix_from_record,
    read_databricks_cluster_hour_ledger_json,
)
from document_kv_cache.full_score_execution import (
    FULL_SCORE_AGGREGATE_RECORD_TYPE,
    FULL_SCORE_AGGREGATE_SCHEMA_VERSION,
    FULL_SCORE_EXECUTION_PLAN_RECORD_TYPE,
    FULL_SCORE_EXECUTION_PLAN_SCHEMA_VERSION,
    FULL_SCORE_MAX_TOKENS,
    FULL_SCORE_METHODS,
    FULL_SCORE_PASSES_PER_METHOD,
    FULL_SCORE_PROTOCOL_ID,
    FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE,
    FULL_SCORE_PUBLICATION_EXECUTION_PLAN_SHA256,
    FULL_SCORE_PUBLICATION_INVENTORY_SHA256,
    FULL_SCORE_PUBLICATION_SHARD_PLAN_SHA256,
    FULL_SCORE_REQUEST_PARALLELISM,
    FULL_SCORE_TEMPERATURE,
    FullScoreCompactArtifactResolver,
    FullScorePhaseAuthorization,
    aggregate_full_score_shard_evidence,
    validate_full_score_aggregate_record,
)
from document_kv_cache.gpu_qualification_databricks import (
    GPUQualificationLaunchAuthorization,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS,
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_ENGINE_VERSION,
    PUBLICATION_CAMPAIGN_FULL_SCORE_EXAMPLES,
    PUBLICATION_CAMPAIGN_FULL_SCORE_PHASES,
    PUBLICATION_CAMPAIGN_FULL_SCORE_SHARDS,
    PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES,
    PUBLICATION_CAMPAIGN_ID,
    PUBLICATION_CAMPAIGN_RECORD_TYPE,
    PUBLICATION_CAMPAIGN_SCHEMA_VERSION,
    PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS,
)
from document_kv_cache.publication_handoff_closure_coordinator import (
    PublicationHandoffRemoteClosureAuthorization,
)
from document_kv_cache.publication_inputs import (
    FULL_SCORE_INVENTORY_RECORD_TYPE,
    FULL_SCORE_INVENTORY_SCHEMA_VERSION,
    FULL_SCORE_MAX_NATURAL_PROMPT_TOKENS,
    FULL_SCORE_SHARD_PLAN_RECORD_TYPE,
    FULL_SCORE_SHARD_PLAN_SCHEMA_VERSION,
    FullScoreInventory,
    full_score_inventory_to_record,
    validate_full_score_inventory_record,
    validate_full_score_shard_plan,
)
from document_kv_cache.publication_latency_execution import (
    PUBLICATION_LATENCY_COLLECTION_RECORD_TYPE,
    PUBLICATION_LATENCY_EXECUTION_PLAN_RECORD_TYPE,
    PUBLICATION_LATENCY_SCHEMA_VERSION,
    PUBLICATION_LATENCY_SUMMARY_RECORD_TYPE,
    PublicationLatencyCollectionAuthorization,
    PublicationLatencySourceClosureAuthorization,
    aggregate_publication_latency_campaign,
    validate_publication_latency_collection_record,
    validate_publication_latency_execution_plan_record,
    validate_publication_latency_summary_record,
)


PUBLICATION_CAMPAIGN_REPORT_RECORD_TYPE = (
    "cachet.vllm_0271_publication_report.v1"
)
PUBLICATION_CAMPAIGN_REPORT_SCHEMA_VERSION = 1
PUBLICATION_CAMPAIGN_REPORT_FILE_NAME = "campaign-report.json"
PUBLICATION_CAMPAIGN_GATE_FILE_NAME = "benchmark-publication-gate.json"
_SEALED_PUBLICATION_FILE_MODES = frozenset({0o444})
_LOADABLE_PUBLICATION_FILE_MODES = frozenset(
    {0o400, 0o440, 0o444, 0o600, 0o640, 0o644}
)

_LATENCY_DESCRIPTIVE_CELL_COUNT = 23
_LATENCY_ESTIMAND_COUNT = 13
_LATENCY_REQUEST_COUNT = 29_440
_LATENCY_CACHE_REQUEST_COUNT = 17_920
_MIN_COLD_ATTESTED_REQUEST_COUNT = 15_360
_MAX_COLD_ATTESTED_REQUEST_COUNT = 16_640
_FULL_SCORE_CACHE_REQUEST_COUNT = PUBLICATION_CAMPAIGN_FULL_SCORE_EXAMPLES
_CHECKED_CACHE_REQUEST_COUNT = (
    _LATENCY_CACHE_REQUEST_COUNT + _FULL_SCORE_CACHE_REQUEST_COUNT
)
_NIAH_CELL_COUNT = 9
_FULL_SCORE_NIAH_EXAMPLES = 1_000
_FULL_SCORE_DATASET_COUNTS = (
    ("biography", 72_831),
    ("hotpotqa", 7_405),
    ("musique", 2_417),
    ("niah", _FULL_SCORE_NIAH_EXAMPLES),
)
_SCORER_CONTRACT_KEYS = frozenset(
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
_FINALIZATION_ISSUER = object()

_REPORT_KEYS = frozenset(
    {
        "campaign_id",
        "closed_record_sha256",
        "coverage",
        "engine_version",
        "latency",
        "ledger",
        "policy",
        "quality",
        "record_type",
        "schema_version",
        "scorer_contracts",
        "source_bindings",
    }
)
_SOURCE_BINDING_KEYS = frozenset(
    {"closed_record_sha256", "record_type", "schema_version"}
)
_SOURCE_BINDING_NAMES = frozenset(
    {
        "campaign",
        "full_score_aggregate",
        "full_score_execution_plan",
        "full_score_inventory",
        "full_score_shard_plan",
        "latency_collection",
        "latency_execution_plan",
        "latency_summary",
    }
)
_COVERAGE_KEYS = frozenset(
    {
        "checked_cache_request_count",
        "checked_distinct_example_count",
        "cold_attested_request_count",
        "full_score_cache_request_count",
        "full_score_identity_count",
        "full_score_passes_per_method",
        "full_score_phase_count",
        "full_score_shard_count",
        "full_score_wave_count",
        "latency_cache_request_count",
        "latency_descriptive_cell_count",
        "latency_estimand_count",
        "latency_job_count",
        "latency_request_count",
        "methods",
        "niah_cell_count",
    }
)
_LEDGER_KEYS = frozenset(
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
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "answer",
        "cloud_run_id",
        "coordinator_run_id",
        "log",
        "logs",
        "output",
        "principal",
        "prompt",
        "raw_output",
        "run_id",
        "single_user_name",
        "task_run_id",
        "uri",
    }
)


@dataclass(frozen=True, slots=True, init=False)
class PublicationCampaignFinalization:
    """Immutable in-process authority over one exact report/gate pair."""

    _report_bytes: bytes
    _gate_bytes: bytes

    def __init__(
        self,
        *,
        report_record: Mapping[str, Any],
        gate_record: Mapping[str, Any],
        _issuer: object,
    ) -> None:
        if _issuer is not _FINALIZATION_ISSUER:
            raise TypeError(
                "publication finalization must come from the campaign finalizer"
            )
        report = _json_object_copy(report_record, "publication campaign report")
        gate = _json_object_copy(gate_record, "publication campaign gate")
        object.__setattr__(self, "_report_bytes", _canonical_pretty_json_bytes(report))
        object.__setattr__(self, "_gate_bytes", _canonical_pretty_json_bytes(gate))

    @property
    def report_record(self) -> Mapping[str, Any]:
        """Return a detached read-only projection of the stored report bytes."""

        return MappingProxyType(json.loads(self._report_bytes))

    @property
    def gate_record(self) -> Mapping[str, Any]:
        """Return a detached read-only projection of the stored gate bytes."""

        return MappingProxyType(json.loads(self._gate_bytes))


def finalize_vllm_0271_publication_campaign(
    *,
    latency_execution_plan_record: Mapping[str, Any],
    latency_collection_authorization: PublicationLatencyCollectionAuthorization,
    latency_summary_record: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
    full_score_inventory: FullScoreInventory,
    full_score_shard_plan_record: Mapping[str, Any],
    full_score_execution_plan_record: Mapping[str, Any],
    full_score_aggregate_record: Mapping[str, Any],
    final_consumer_authorization: FullScorePhaseAuthorization,
    remote_consumer_authorizations: Sequence[object],
    compact_artifact_resolver: FullScoreCompactArtifactResolver,
    ledger_path: str | Path,
) -> PublicationCampaignFinalization:
    """Recompute both evidence branches and emit one digest-bound standard gate."""

    validate_publication_latency_execution_plan_record(latency_execution_plan_record)
    if not isinstance(
        latency_collection_authorization,
        PublicationLatencyCollectionAuthorization,
    ):
        raise TypeError(
            "latency_collection_authorization has the wrong authority type"
        )
    latency_collection = latency_collection_authorization.collection
    validate_publication_latency_collection_record(
        latency_collection,
        execution_plan_record=latency_execution_plan_record,
    )
    if (
        latency_collection.get("closed_record_sha256")
        != latency_collection_authorization.collection_sha256
    ):
        raise ValueError("latency collection authority payload drift")
    collection_ledger = _mapping(latency_collection, "ledger")
    if (
        collection_ledger.get("ledger_path_sha256")
        != latency_collection_authorization.ledger_path_sha256
        or not _json_type_exact_equal(
            dict(_mapping(collection_ledger, "ledger_prefix")),
            latency_collection_authorization.ledger_prefix.to_record(),
        )
    ):
        raise ValueError("latency collection authority ledger binding drift")

    validate_publication_latency_summary_record(
        latency_summary_record,
        expected_collection_sha256=(
            latency_collection_authorization.collection_sha256
        ),
        expected_execution_plan_sha256=_required_sha256(
            latency_execution_plan_record,
            "closed_record_sha256",
        ),
    )
    recomputed_latency_summary = aggregate_publication_latency_campaign(
        latency_collection_authorization,
        execution_plan_record=latency_execution_plan_record,
        qualification_launch_authorization=qualification_launch_authorization,
        handoff_serving_authorization=handoff_serving_authorization,
        bf16_handoff_serving_authorization=bf16_handoff_serving_authorization,
        source_closure_authorization=source_closure_authorization,
    )
    if not _json_type_exact_equal(
        dict(latency_summary_record),
        recomputed_latency_summary,
    ):
        raise ValueError(
            "latency summary differs from authoritative campaign reaggregation"
        )

    if not isinstance(full_score_inventory, FullScoreInventory):
        raise TypeError("full_score_inventory must be FullScoreInventory")
    inventory_record = full_score_inventory_to_record(full_score_inventory)
    validate_full_score_inventory_record(
        inventory_record,
        inventory=full_score_inventory,
    )
    validate_full_score_shard_plan(
        full_score_shard_plan_record,
        inventory=full_score_inventory,
    )
    validate_full_score_aggregate_record(
        full_score_aggregate_record,
        inventory=full_score_inventory,
        shard_plan=full_score_shard_plan_record,
        execution_plan=full_score_execution_plan_record,
        require_publication=True,
    )
    recomputed_full_score_aggregate = aggregate_full_score_shard_evidence(
        full_score_inventory,
        full_score_shard_plan_record,
        (),
        authorization_scope=FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE,
        execution_plan=full_score_execution_plan_record,
        final_consumer_authorization=final_consumer_authorization,
        ledger_path=ledger_path,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_artifact_resolver,
    )
    if not _json_type_exact_equal(
        dict(full_score_aggregate_record),
        recomputed_full_score_aggregate,
    ):
        raise ValueError(
            "full-score aggregate differs from authoritative evidence reaggregation"
        )

    ledger_record = _validated_campaign_ledger_projection(
        latency_collection_authorization=latency_collection_authorization,
        full_score_execution_plan_record=full_score_execution_plan_record,
        full_score_aggregate_record=recomputed_full_score_aggregate,
        final_consumer_authorization=final_consumer_authorization,
        ledger_path=ledger_path,
    )
    report = _build_report(
        latency_execution_plan_record=latency_execution_plan_record,
        latency_collection_record=latency_collection,
        latency_summary_record=recomputed_latency_summary,
        inventory_record=inventory_record,
        full_score_shard_plan_record=full_score_shard_plan_record,
        full_score_execution_plan_record=full_score_execution_plan_record,
        full_score_aggregate_record=recomputed_full_score_aggregate,
        ledger_record=ledger_record,
    )
    gate = _publication_gate_for_report(report)
    return PublicationCampaignFinalization(
        report_record=report,
        gate_record=gate,
        _issuer=_FINALIZATION_ISSUER,
    )


def validate_vllm_0271_publication_finalization(
    finalization: PublicationCampaignFinalization,
    *,
    latency_execution_plan_record: Mapping[str, Any],
    latency_collection_authorization: PublicationLatencyCollectionAuthorization,
    latency_summary_record: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
    full_score_inventory: FullScoreInventory,
    full_score_shard_plan_record: Mapping[str, Any],
    full_score_execution_plan_record: Mapping[str, Any],
    full_score_aggregate_record: Mapping[str, Any],
    final_consumer_authorization: FullScorePhaseAuthorization,
    remote_consumer_authorizations: Sequence[object],
    compact_artifact_resolver: FullScoreCompactArtifactResolver,
    ledger_path: str | Path,
) -> None:
    """Rebuild *finalization* from its authorities and require exact equality."""

    if not isinstance(finalization, PublicationCampaignFinalization):
        raise TypeError("finalization must be PublicationCampaignFinalization")
    validate_vllm_0271_publication_report_pair(
        finalization.report_record,
        finalization.gate_record,
    )
    expected = finalize_vllm_0271_publication_campaign(
        latency_execution_plan_record=latency_execution_plan_record,
        latency_collection_authorization=latency_collection_authorization,
        latency_summary_record=latency_summary_record,
        qualification_launch_authorization=qualification_launch_authorization,
        handoff_serving_authorization=handoff_serving_authorization,
        bf16_handoff_serving_authorization=bf16_handoff_serving_authorization,
        source_closure_authorization=source_closure_authorization,
        full_score_inventory=full_score_inventory,
        full_score_shard_plan_record=full_score_shard_plan_record,
        full_score_execution_plan_record=full_score_execution_plan_record,
        full_score_aggregate_record=full_score_aggregate_record,
        final_consumer_authorization=final_consumer_authorization,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_artifact_resolver,
        ledger_path=ledger_path,
    )
    if (
        not _json_type_exact_equal(
            dict(finalization.report_record),
            dict(expected.report_record),
        )
        or not _json_type_exact_equal(
            dict(finalization.gate_record),
            dict(expected.gate_record),
        )
    ):
        raise ValueError("publication campaign report/gate pair is not canonical")


def write_vllm_0271_publication_finalization(
    finalization: PublicationCampaignFinalization,
    directory: str | Path,
) -> tuple[Path, Path]:
    """Write an exact gate/report pair with the report as the commit file."""

    if not isinstance(finalization, PublicationCampaignFinalization):
        raise TypeError("finalization must be PublicationCampaignFinalization")
    validate_vllm_0271_publication_report_pair(
        finalization.report_record,
        finalization.gate_record,
    )
    root = Path(directory).expanduser().absolute()
    report_path = root / PUBLICATION_CAMPAIGN_REPORT_FILE_NAME
    gate_path = root / PUBLICATION_CAMPAIGN_GATE_FILE_NAME
    directory_descriptor = _open_directory_no_symlinks(
        root,
        label="publication evidence directory",
    )
    lock_descriptor = -1
    try:
        lock_descriptor = _acquire_publication_directory_lock(
            directory_descriptor,
            exclusive=True,
        )
        _remove_stale_publication_temporaries_at(directory_descriptor)
        report_exists = _matching_read_only_json_file_exists_at(
            directory_descriptor,
            PUBLICATION_CAMPAIGN_REPORT_FILE_NAME,
            finalization.report_record,
            label="publication campaign report",
        )
        gate_exists = _matching_read_only_json_file_exists_at(
            directory_descriptor,
            PUBLICATION_CAMPAIGN_GATE_FILE_NAME,
            finalization.gate_record,
            label="publication campaign gate",
        )
        if report_exists and not gate_exists:
            raise ValueError("publication report commit exists without its gate")
        if report_exists:
            raise FileExistsError("publication report/gate output already exists")
        if not gate_exists:
            _write_json_exclusive_at(
                directory_descriptor,
                PUBLICATION_CAMPAIGN_GATE_FILE_NAME,
                finalization.gate_record,
            )
        _write_json_exclusive_at(
            directory_descriptor,
            PUBLICATION_CAMPAIGN_REPORT_FILE_NAME,
            finalization.report_record,
        )
        if not (
            _matching_read_only_json_file_exists_at(
                directory_descriptor,
                PUBLICATION_CAMPAIGN_GATE_FILE_NAME,
                finalization.gate_record,
                label="publication campaign gate",
            )
            and _matching_read_only_json_file_exists_at(
                directory_descriptor,
                PUBLICATION_CAMPAIGN_REPORT_FILE_NAME,
                finalization.report_record,
                label="publication campaign report",
            )
        ):
            raise RuntimeError("publication report/gate commit revalidation failed")
        _require_directory_path_matches_descriptor(root, directory_descriptor)
    finally:
        try:
            if lock_descriptor >= 0:
                _release_publication_directory_lock(lock_descriptor)
        finally:
            os.close(directory_descriptor)
    return report_path, gate_path


def load_vllm_0271_publication_finalization(
    directory: str | Path,
    *,
    latency_execution_plan_record: Mapping[str, Any],
    latency_collection_authorization: PublicationLatencyCollectionAuthorization,
    latency_summary_record: Mapping[str, Any],
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    bf16_handoff_serving_authorization: PublicationHandoffRemoteClosureAuthorization,
    source_closure_authorization: PublicationLatencySourceClosureAuthorization,
    full_score_inventory: FullScoreInventory,
    full_score_shard_plan_record: Mapping[str, Any],
    full_score_execution_plan_record: Mapping[str, Any],
    full_score_aggregate_record: Mapping[str, Any],
    final_consumer_authorization: FullScorePhaseAuthorization,
    remote_consumer_authorizations: Sequence[object],
    compact_artifact_resolver: FullScoreCompactArtifactResolver,
    ledger_path: str | Path,
) -> PublicationCampaignFinalization:
    """Load canonical report/gate bytes and replay all source authorities."""

    root = Path(directory).expanduser().absolute()
    directory_descriptor = _open_directory_no_symlinks(
        root,
        label="publication evidence directory",
    )
    lock_descriptor = -1
    try:
        lock_descriptor = _acquire_publication_directory_lock(
            directory_descriptor,
            exclusive=False,
        )
        _reject_stale_publication_temporaries_at(directory_descriptor)
        report = _read_canonical_json_at(
            directory_descriptor,
            PUBLICATION_CAMPAIGN_REPORT_FILE_NAME,
            "publication campaign report",
            allowed_modes=_LOADABLE_PUBLICATION_FILE_MODES,
        )
        gate = _read_canonical_json_at(
            directory_descriptor,
            PUBLICATION_CAMPAIGN_GATE_FILE_NAME,
            "publication campaign gate",
            allowed_modes=_LOADABLE_PUBLICATION_FILE_MODES,
        )
        validate_vllm_0271_publication_report_pair(report, gate)
        finalization = PublicationCampaignFinalization(
            report_record=report,
            gate_record=gate,
            _issuer=_FINALIZATION_ISSUER,
        )
        validate_vllm_0271_publication_finalization(
            finalization,
            latency_execution_plan_record=latency_execution_plan_record,
            latency_collection_authorization=latency_collection_authorization,
            latency_summary_record=latency_summary_record,
            qualification_launch_authorization=qualification_launch_authorization,
            handoff_serving_authorization=handoff_serving_authorization,
            bf16_handoff_serving_authorization=(
                bf16_handoff_serving_authorization
            ),
            source_closure_authorization=source_closure_authorization,
            full_score_inventory=full_score_inventory,
            full_score_shard_plan_record=full_score_shard_plan_record,
            full_score_execution_plan_record=full_score_execution_plan_record,
            full_score_aggregate_record=full_score_aggregate_record,
            final_consumer_authorization=final_consumer_authorization,
            remote_consumer_authorizations=remote_consumer_authorizations,
            compact_artifact_resolver=compact_artifact_resolver,
            ledger_path=ledger_path,
        )
        if (
            not _json_type_exact_equal(
                report,
                _read_canonical_json_at(
                    directory_descriptor,
                    PUBLICATION_CAMPAIGN_REPORT_FILE_NAME,
                    "publication campaign report",
                    allowed_modes=_LOADABLE_PUBLICATION_FILE_MODES,
                ),
            )
            or not _json_type_exact_equal(
                gate,
                _read_canonical_json_at(
                    directory_descriptor,
                    PUBLICATION_CAMPAIGN_GATE_FILE_NAME,
                    "publication campaign gate",
                    allowed_modes=_LOADABLE_PUBLICATION_FILE_MODES,
                ),
            )
        ):
            raise ValueError("publication report/gate changed during authority replay")
        _require_directory_path_matches_descriptor(root, directory_descriptor)
        return finalization
    finally:
        try:
            if lock_descriptor >= 0:
                _release_publication_directory_lock(lock_descriptor)
        finally:
            os.close(directory_descriptor)


def _build_report(
    *,
    latency_execution_plan_record: Mapping[str, Any],
    latency_collection_record: Mapping[str, Any],
    latency_summary_record: Mapping[str, Any],
    inventory_record: Mapping[str, Any],
    full_score_shard_plan_record: Mapping[str, Any],
    full_score_execution_plan_record: Mapping[str, Any],
    full_score_aggregate_record: Mapping[str, Any],
    ledger_record: Mapping[str, Any],
) -> dict[str, Any]:
    jobs = _mapping_sequence(latency_execution_plan_record, "jobs")
    results = _mapping_sequence(latency_collection_record, "results")
    latency_request_count = sum(_required_int(job, "request_count") for job in jobs)
    latency_cache_request_count = sum(
        _required_int(job, "request_count")
        for job in jobs
        if job.get("method_id") == "vanilla_prefill"
    )
    cold_attested_request_count = sum(
        _required_nonnegative_int(
            _mapping(result, "cache_telemetry"),
            "cold_read_attested_count",
        )
        for result in results
    )
    if (
        len(jobs) != PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS
        or latency_request_count != _LATENCY_REQUEST_COUNT
        or latency_cache_request_count != _LATENCY_CACHE_REQUEST_COUNT
        or len(results) != len(jobs)
        or not (
            _MIN_COLD_ATTESTED_REQUEST_COUNT
            <= cold_attested_request_count
            <= _MAX_COLD_ATTESTED_REQUEST_COUNT
        )
    ):
        raise ValueError("publication latency coverage projection drift")

    descriptive_cells = _mapping_sequence(
        latency_summary_record,
        "descriptive_cells",
    )
    estimates = _mapping_sequence(latency_summary_record, "estimates")
    datasets = _mapping(full_score_aggregate_record, "datasets")
    niah_grid = _mapping(full_score_aggregate_record, "niah_grid")
    scorers = full_score_aggregate_record.get("scorers")
    if not isinstance(scorers, list) or not scorers:
        raise ValueError("full-score aggregate is missing scorer/parser contracts")
    if (
        full_score_aggregate_record.get("authorization_scope")
        != FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
        or full_score_aggregate_record.get("identity_count")
        != PUBLICATION_CAMPAIGN_FULL_SCORE_EXAMPLES
        or full_score_aggregate_record.get("shard_count")
        != PUBLICATION_CAMPAIGN_FULL_SCORE_SHARDS
        or full_score_aggregate_record.get("passes_per_method")
        != FULL_SCORE_PASSES_PER_METHOD
        or full_score_aggregate_record.get("methods") != list(FULL_SCORE_METHODS)
        or len(niah_grid) != _NIAH_CELL_COUNT
    ):
        raise ValueError("publication full-score coverage projection drift")

    campaign_binding = _mapping(
        _mapping(latency_execution_plan_record, "sources"),
        "campaign",
    )
    report: dict[str, Any] = {
        "campaign_id": PUBLICATION_CAMPAIGN_ID,
        "closed_record_sha256": "",
        "coverage": {
            "checked_cache_request_count": _CHECKED_CACHE_REQUEST_COUNT,
            "checked_distinct_example_count": (
                PUBLICATION_CAMPAIGN_FULL_SCORE_EXAMPLES
            ),
            "cold_attested_request_count": cold_attested_request_count,
            "full_score_cache_request_count": _FULL_SCORE_CACHE_REQUEST_COUNT,
            "full_score_identity_count": PUBLICATION_CAMPAIGN_FULL_SCORE_EXAMPLES,
            "full_score_passes_per_method": FULL_SCORE_PASSES_PER_METHOD,
            "full_score_phase_count": PUBLICATION_CAMPAIGN_FULL_SCORE_PHASES,
            "full_score_shard_count": PUBLICATION_CAMPAIGN_FULL_SCORE_SHARDS,
            "full_score_wave_count": PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES,
            "latency_cache_request_count": latency_cache_request_count,
            "latency_descriptive_cell_count": len(descriptive_cells),
            "latency_estimand_count": len(estimates),
            "latency_job_count": len(jobs),
            "latency_request_count": latency_request_count,
            "methods": list(FULL_SCORE_METHODS),
            "niah_cell_count": len(niah_grid),
        },
        "engine_version": PUBLICATION_CAMPAIGN_ENGINE_VERSION,
        "latency": {
            "analysis": _json_value_copy(latency_summary_record.get("analysis")),
            "descriptive_cells": _json_value_copy(descriptive_cells),
            "estimates": _json_value_copy(estimates),
        },
        "ledger": _json_value_copy(ledger_record),
        "policy": "publication",
        "quality": {
            "aggregation_unit": full_score_aggregate_record.get(
                "aggregation_unit"
            ),
            "bootstrap": _json_value_copy(
                full_score_aggregate_record.get("bootstrap")
            ),
            "datasets": _json_value_copy(datasets),
            "niah_grid": _json_value_copy(niah_grid),
            "protocol": _json_value_copy(
                full_score_aggregate_record.get("protocol")
            ),
        },
        "record_type": PUBLICATION_CAMPAIGN_REPORT_RECORD_TYPE,
        "schema_version": PUBLICATION_CAMPAIGN_REPORT_SCHEMA_VERSION,
        "scorer_contracts": _json_value_copy(scorers),
        "source_bindings": {
            "campaign": {
                "closed_record_sha256": _required_sha256(
                    campaign_binding,
                    "closed_record_sha256",
                ),
                "record_type": PUBLICATION_CAMPAIGN_RECORD_TYPE,
                "schema_version": PUBLICATION_CAMPAIGN_SCHEMA_VERSION,
            },
            "full_score_aggregate": _record_binding(
                full_score_aggregate_record,
                expected_record_type=FULL_SCORE_AGGREGATE_RECORD_TYPE,
                expected_schema_version=FULL_SCORE_AGGREGATE_SCHEMA_VERSION,
            ),
            "full_score_execution_plan": _record_binding(
                full_score_execution_plan_record,
                expected_record_type=FULL_SCORE_EXECUTION_PLAN_RECORD_TYPE,
                expected_schema_version=FULL_SCORE_EXECUTION_PLAN_SCHEMA_VERSION,
            ),
            "full_score_inventory": _record_binding(
                inventory_record,
                expected_record_type=FULL_SCORE_INVENTORY_RECORD_TYPE,
                expected_schema_version=FULL_SCORE_INVENTORY_SCHEMA_VERSION,
            ),
            "full_score_shard_plan": _record_binding(
                full_score_shard_plan_record,
                expected_record_type=FULL_SCORE_SHARD_PLAN_RECORD_TYPE,
                expected_schema_version=FULL_SCORE_SHARD_PLAN_SCHEMA_VERSION,
            ),
            "latency_collection": _record_binding(
                latency_collection_record,
                expected_record_type=PUBLICATION_LATENCY_COLLECTION_RECORD_TYPE,
                expected_schema_version=PUBLICATION_LATENCY_SCHEMA_VERSION,
            ),
            "latency_execution_plan": _record_binding(
                latency_execution_plan_record,
                expected_record_type=(
                    PUBLICATION_LATENCY_EXECUTION_PLAN_RECORD_TYPE
                ),
                expected_schema_version=PUBLICATION_LATENCY_SCHEMA_VERSION,
            ),
            "latency_summary": _record_binding(
                latency_summary_record,
                expected_record_type=PUBLICATION_LATENCY_SUMMARY_RECORD_TYPE,
                expected_schema_version=PUBLICATION_LATENCY_SCHEMA_VERSION,
            ),
        },
    }
    if (
        report["coverage"]["latency_descriptive_cell_count"]
        != _LATENCY_DESCRIPTIVE_CELL_COUNT
        or report["coverage"]["latency_estimand_count"]
        != _LATENCY_ESTIMAND_COUNT
    ):
        raise ValueError("latency table closure is incomplete")
    report["closed_record_sha256"] = _closed_record_sha256(report)
    _validate_report_envelope(report)
    return report


def _validated_campaign_ledger_projection(
    *,
    latency_collection_authorization: PublicationLatencyCollectionAuthorization,
    full_score_execution_plan_record: Mapping[str, Any],
    full_score_aggregate_record: Mapping[str, Any],
    final_consumer_authorization: FullScorePhaseAuthorization,
    ledger_path: str | Path,
) -> dict[str, Any]:
    if not isinstance(final_consumer_authorization, FullScorePhaseAuthorization):
        raise TypeError("final_consumer_authorization has the wrong authority type")
    execution_sha256 = _required_sha256(
        full_score_execution_plan_record,
        "closed_record_sha256",
    )
    path_sha256 = databricks_ledger_path_sha256(ledger_path)
    if (
        path_sha256 != latency_collection_authorization.ledger_path_sha256
        or path_sha256 != final_consumer_authorization.ledger_path_sha256
        or final_consumer_authorization.execution_plan_sha256 != execution_sha256
        or final_consumer_authorization.wave_index
        != PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES - 1
        or final_consumer_authorization.phase != "consumer"
    ):
        raise ValueError("campaign final ledger authority binding drift")
    live = read_databricks_cluster_hour_ledger_json(ledger_path)
    latency_prefix = latency_collection_authorization.ledger_prefix
    observed_latency_prefix = databricks_ledger_prefix_at_counts(
        live,
        reservation_count=latency_prefix.reservation_count,
        submission_receipt_count=latency_prefix.submission_receipt_count,
        terminal_actual_count=latency_prefix.terminal_actual_count,
    )
    if observed_latency_prefix != latency_prefix:
        raise ValueError("latency terminal prefix is not a live ledger prefix")
    final_prefix = databricks_ledger_prefix(live)
    lineage = _mapping(full_score_aggregate_record, "publication_lineage")
    lineage_prefix = databricks_ledger_prefix_from_record(
        _mapping(lineage, "terminal_prefix")
    )
    if (
        live.cap_cluster_hours != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
        or latency_prefix.cap_cluster_hours
        != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
        or final_prefix.cap_cluster_hours
        != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
        or live.ledger_id != latency_prefix.ledger_id
        or lineage.get("ledger_id") != live.ledger_id
        or lineage.get("ledger_path_sha256") != path_sha256
        or lineage_prefix != final_prefix
        or final_consumer_authorization.ledger_prefix != final_prefix
    ):
        raise ValueError("full-score terminal prefix is not the current live ledger")
    for field_name in (
        "reservation_count",
        "submission_receipt_count",
        "terminal_actual_count",
    ):
        if getattr(final_prefix, field_name) != getattr(latency_prefix, field_name) + (
            PUBLICATION_CAMPAIGN_FULL_SCORE_PHASES
        ):
            raise ValueError("full-score phase count does not close ledger lineage")
    expected_workloads = [
        f"full-score:{execution_sha256}:wave-{wave_index:03d}:{phase}"
        for wave_index in range(PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES)
        for phase in ("producer", "consumer")
    ]
    reservation_suffix = live.reservations[latency_prefix.reservation_count :]
    receipt_suffix = live.submission_receipts[
        latency_prefix.submission_receipt_count :
    ]
    terminal_suffix = live.terminal_actuals[latency_prefix.terminal_actual_count :]
    reservation_attempt_ids = [item.attempt_id for item in reservation_suffix]
    if (
        [item.workload_id for item in reservation_suffix] != expected_workloads
        or [item.attempt_id for item in receipt_suffix] != reservation_attempt_ids
        or [item.attempt_id for item in terminal_suffix] != reservation_attempt_ids
    ):
        raise ValueError("full-score ledger suffix contains an unrelated append")
    if (
        live.active_reserved_cluster_hours != 0.0
        or live.active_reserved_task_count != 0
        or live.remaining_cluster_hours
        < DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS
    ):
        raise ValueError("publication final ledger is active or below hard headroom")
    return {
        "final_accounted_cluster_hours": live.accounted_cluster_hours,
        "final_active_reserved_cluster_hours": live.active_reserved_cluster_hours,
        "final_active_reserved_task_count": live.active_reserved_task_count,
        "final_remaining_cluster_hours": live.remaining_cluster_hours,
        "full_score_terminal_prefix": final_prefix.to_record(),
        "latency_terminal_prefix": latency_prefix.to_record(),
        "ledger_id": live.ledger_id,
        "ledger_path_sha256": path_sha256,
        "required_unreserved_headroom_cluster_hours": (
            DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS
        ),
    }


def validate_vllm_0271_publication_report_pair(
    report_record: Mapping[str, Any],
    gate_record: Mapping[str, Any],
) -> None:
    """Validate one exact sanitized report and its standard publication gate."""

    report = _json_object_copy(report_record, "publication campaign report")
    gate = _json_object_copy(gate_record, "publication campaign gate")
    _validate_report_envelope(report)
    expected_gate = _publication_gate_for_report(report)
    if not _json_type_exact_equal(gate, expected_gate):
        raise ValueError("publication gate does not bind the supplied report")


def _publication_gate_for_report(record: Mapping[str, Any]) -> dict[str, Any]:
    _validate_report_envelope(record)
    coverage = _mapping(record, "coverage")
    return benchmark_publication_gate_to_record(
        BenchmarkPublicationGateResult(
            issues=(),
            checked_cache_arms=("vanilla_prefill",),
            checked_cache_requests=_required_int(
                coverage,
                "checked_cache_request_count",
            ),
            cold_attested_requests=_required_nonnegative_int(
                coverage,
                "cold_attested_request_count",
            ),
            policy="publication",
            checked_distinct_examples=_required_int(
                coverage,
                "checked_distinct_example_count",
            ),
            measurement_scopes=("latency", "quality", "resource"),
            benchmark_payload_digest=_required_sha256(
                record,
                "closed_record_sha256",
            ),
        )
    )


def _validate_report_envelope(record: Mapping[str, Any]) -> None:
    _require_exact_keys(record, _REPORT_KEYS, "publication campaign report")
    if (
        record.get("record_type") != PUBLICATION_CAMPAIGN_REPORT_RECORD_TYPE
        or not _json_type_exact_equal(
            record.get("schema_version"),
            PUBLICATION_CAMPAIGN_REPORT_SCHEMA_VERSION,
        )
        or record.get("campaign_id") != PUBLICATION_CAMPAIGN_ID
        or record.get("engine_version") != PUBLICATION_CAMPAIGN_ENGINE_VERSION
        or record.get("policy") != "publication"
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("publication campaign report envelope is invalid")
    _assert_sanitized(record)
    bindings = _mapping(record, "source_bindings")
    _require_exact_keys(bindings, _SOURCE_BINDING_NAMES, "report source bindings")
    expected_binding_envelopes = {
        "campaign": (
            PUBLICATION_CAMPAIGN_RECORD_TYPE,
            PUBLICATION_CAMPAIGN_SCHEMA_VERSION,
        ),
        "full_score_aggregate": (
            FULL_SCORE_AGGREGATE_RECORD_TYPE,
            FULL_SCORE_AGGREGATE_SCHEMA_VERSION,
        ),
        "full_score_execution_plan": (
            FULL_SCORE_EXECUTION_PLAN_RECORD_TYPE,
            FULL_SCORE_EXECUTION_PLAN_SCHEMA_VERSION,
        ),
        "full_score_inventory": (
            FULL_SCORE_INVENTORY_RECORD_TYPE,
            FULL_SCORE_INVENTORY_SCHEMA_VERSION,
        ),
        "full_score_shard_plan": (
            FULL_SCORE_SHARD_PLAN_RECORD_TYPE,
            FULL_SCORE_SHARD_PLAN_SCHEMA_VERSION,
        ),
        "latency_collection": (
            PUBLICATION_LATENCY_COLLECTION_RECORD_TYPE,
            PUBLICATION_LATENCY_SCHEMA_VERSION,
        ),
        "latency_execution_plan": (
            PUBLICATION_LATENCY_EXECUTION_PLAN_RECORD_TYPE,
            PUBLICATION_LATENCY_SCHEMA_VERSION,
        ),
        "latency_summary": (
            PUBLICATION_LATENCY_SUMMARY_RECORD_TYPE,
            PUBLICATION_LATENCY_SCHEMA_VERSION,
        ),
    }
    for name, binding in bindings.items():
        source_binding = _mapping_value(binding, f"source binding {name}")
        _require_exact_keys(
            source_binding,
            _SOURCE_BINDING_KEYS,
            f"source binding {name}",
        )
        expected_type, expected_schema = expected_binding_envelopes[name]
        if (
            source_binding.get("record_type") != expected_type
            or not _json_type_exact_equal(
                source_binding.get("schema_version"),
                expected_schema,
            )
        ):
            raise ValueError(f"source binding {name} envelope drift")
        _required_sha256(source_binding, "closed_record_sha256")
    expected_pinned_bindings = {
        "campaign": PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
        "full_score_execution_plan": (
            FULL_SCORE_PUBLICATION_EXECUTION_PLAN_SHA256
        ),
        "full_score_inventory": FULL_SCORE_PUBLICATION_INVENTORY_SHA256,
        "full_score_shard_plan": FULL_SCORE_PUBLICATION_SHARD_PLAN_SHA256,
    }
    for name, expected_sha256 in expected_pinned_bindings.items():
        if (
            _mapping(bindings, name).get("closed_record_sha256")
            != expected_sha256
        ):
            raise ValueError(f"source binding {name} pinned identity drift")
    coverage = _mapping(record, "coverage")
    _require_exact_keys(coverage, _COVERAGE_KEYS, "report coverage")
    expected_coverage = {
        "checked_cache_request_count": _CHECKED_CACHE_REQUEST_COUNT,
        "checked_distinct_example_count": PUBLICATION_CAMPAIGN_FULL_SCORE_EXAMPLES,
        "full_score_cache_request_count": _FULL_SCORE_CACHE_REQUEST_COUNT,
        "full_score_identity_count": PUBLICATION_CAMPAIGN_FULL_SCORE_EXAMPLES,
        "full_score_passes_per_method": FULL_SCORE_PASSES_PER_METHOD,
        "full_score_phase_count": PUBLICATION_CAMPAIGN_FULL_SCORE_PHASES,
        "full_score_shard_count": PUBLICATION_CAMPAIGN_FULL_SCORE_SHARDS,
        "full_score_wave_count": PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES,
        "latency_cache_request_count": _LATENCY_CACHE_REQUEST_COUNT,
        "latency_descriptive_cell_count": _LATENCY_DESCRIPTIVE_CELL_COUNT,
        "latency_estimand_count": _LATENCY_ESTIMAND_COUNT,
        "latency_job_count": PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS,
        "latency_request_count": _LATENCY_REQUEST_COUNT,
        "methods": list(FULL_SCORE_METHODS),
        "niah_cell_count": _NIAH_CELL_COUNT,
    }
    observed_coverage = {
        key: coverage.get(key)
        for key in expected_coverage
    }
    if not _json_type_exact_equal(observed_coverage, expected_coverage):
        raise ValueError("publication campaign report coverage is invalid")
    cold_count = _required_nonnegative_int(
        coverage,
        "cold_attested_request_count",
    )
    if not (
        _MIN_COLD_ATTESTED_REQUEST_COUNT
        <= cold_count
        <= _MAX_COLD_ATTESTED_REQUEST_COUNT
    ):
        raise ValueError("publication campaign cold-attestation count is invalid")
    latency = _mapping(record, "latency")
    _require_exact_keys(
        latency,
        frozenset({"analysis", "descriptive_cells", "estimates"}),
        "report latency projection",
    )
    if (
        len(_mapping_sequence(latency, "descriptive_cells"))
        != _LATENCY_DESCRIPTIVE_CELL_COUNT
        or len(_mapping_sequence(latency, "estimates")) != _LATENCY_ESTIMAND_COUNT
    ):
        raise ValueError("report latency table coverage is incomplete")
    latency_collection_sha256 = _required_sha256(
        _mapping(bindings, "latency_collection"),
        "closed_record_sha256",
    )
    latency_execution_plan_sha256 = _required_sha256(
        _mapping(bindings, "latency_execution_plan"),
        "closed_record_sha256",
    )
    reconstructed_latency_summary = {
        "analysis": _json_value_copy(latency.get("analysis")),
        "campaign_id": PUBLICATION_CAMPAIGN_ID,
        "closed_record_sha256": _required_sha256(
            _mapping(bindings, "latency_summary"),
            "closed_record_sha256",
        ),
        "collection_sha256": latency_collection_sha256,
        "descriptive_cell_count": _LATENCY_DESCRIPTIVE_CELL_COUNT,
        "descriptive_cells": _json_value_copy(
            latency.get("descriptive_cells")
        ),
        "estimand_count": _LATENCY_ESTIMAND_COUNT,
        "estimates": _json_value_copy(latency.get("estimates")),
        "execution_plan_sha256": latency_execution_plan_sha256,
        "record_type": PUBLICATION_LATENCY_SUMMARY_RECORD_TYPE,
        "schema_version": PUBLICATION_LATENCY_SCHEMA_VERSION,
    }
    validate_publication_latency_summary_record(
        reconstructed_latency_summary,
        expected_collection_sha256=latency_collection_sha256,
        expected_execution_plan_sha256=latency_execution_plan_sha256,
    )
    quality = _mapping(record, "quality")
    _require_exact_keys(
        quality,
        frozenset(
            {"aggregation_unit", "bootstrap", "datasets", "niah_grid", "protocol"}
        ),
        "report quality projection",
    )
    scorers = record.get("scorer_contracts")
    _validate_quality_projection(
        quality,
        scorers=scorers,
        inventory_sha256=_required_sha256(
            _mapping(bindings, "full_score_inventory"),
            "closed_record_sha256",
        ),
        shard_plan_sha256=_required_sha256(
            _mapping(bindings, "full_score_shard_plan"),
            "closed_record_sha256",
        ),
    )
    ledger = _mapping(record, "ledger")
    _require_exact_keys(ledger, _LEDGER_KEYS, "report ledger")
    latency_prefix_record = _mapping(ledger, "latency_terminal_prefix")
    full_score_prefix_record = _mapping(ledger, "full_score_terminal_prefix")
    latency_prefix = databricks_ledger_prefix_from_record(
        latency_prefix_record
    )
    full_score_prefix = databricks_ledger_prefix_from_record(
        full_score_prefix_record
    )
    if (
        not _json_type_exact_equal(
            dict(latency_prefix_record),
            latency_prefix.to_record(),
        )
        or not _json_type_exact_equal(
            dict(full_score_prefix_record),
            full_score_prefix.to_record(),
        )
        or ledger.get("ledger_id") != latency_prefix.ledger_id
        or full_score_prefix.ledger_id != latency_prefix.ledger_id
        or latency_prefix.cap_cluster_hours
        != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
        or full_score_prefix.cap_cluster_hours
        != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
        or full_score_prefix.cap_cluster_hours
        != latency_prefix.cap_cluster_hours
        or any(
            getattr(full_score_prefix, field_name)
            != getattr(latency_prefix, field_name)
            + PUBLICATION_CAMPAIGN_FULL_SCORE_PHASES
            for field_name in (
                "reservation_count",
                "submission_receipt_count",
                "terminal_actual_count",
            )
        )
        or not _json_type_exact_equal(
            ledger.get("final_active_reserved_cluster_hours"),
            0.0,
        )
        or not _json_type_exact_equal(
            ledger.get("final_active_reserved_task_count"),
            0,
        )
        or not _json_type_exact_equal(
            ledger.get("required_unreserved_headroom_cluster_hours"),
            DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS,
        )
    ):
        raise ValueError("report ledger closure is invalid")
    _required_sha256(ledger, "ledger_path_sha256")
    remaining = ledger.get("final_remaining_cluster_hours")
    accounted = ledger.get("final_accounted_cluster_hours")
    if (
        isinstance(remaining, bool)
        or not isinstance(remaining, (int, float))
        or not math.isfinite(float(remaining))
        or float(remaining) < DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS
        or isinstance(accounted, bool)
        or not isinstance(accounted, (int, float))
        or not math.isfinite(float(accounted))
        or float(accounted) < 0.0
        or not math.isclose(
            float(accounted) + float(remaining),
            full_score_prefix.cap_cluster_hours,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("report ledger accounting is invalid")


def _expected_full_score_protocol() -> dict[str, Any]:
    """Mirror the exact worker protocol at the finalization boundary."""

    return {
        "add_special_tokens": False,
        "complete_inventory_required": True,
        "input_length": {
            "max_natural_prompt_tokens": FULL_SCORE_MAX_NATURAL_PROMPT_TOKENS,
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
        "max_tokens": FULL_SCORE_MAX_TOKENS,
        "methods": list(FULL_SCORE_METHODS),
        "natural_eos": True,
        "passes_per_method": FULL_SCORE_PASSES_PER_METHOD,
        "prompt_text_mode": "logical",
        "protocol_id": FULL_SCORE_PROTOCOL_ID,
        "request_parallelism": FULL_SCORE_REQUEST_PARALLELISM,
        "temperature": FULL_SCORE_TEMPERATURE,
    }


def _expected_full_score_bootstrap() -> dict[str, Any]:
    return {
        "confidence_level": 0.95,
        "draws": PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS,
        "resampling_unit": "paired_example_within_dataset_or_niah_cell",
        "rng_algorithm": "cpython-3.11-random.Random-mt19937-choices-v1",
        "seed_domain": "cachet.full_score.paired_bootstrap.seed.v1",
        "stratification": "dataset; niah additionally by frozen 9-cell grid",
        "tail_algorithm": "linear_interpolated_empirical_quantile_type7_v1",
    }


def _expected_scorer_contracts() -> list[dict[str, Any]]:
    registry = default_dataset_scorer_registry()
    return [
        {
            "answer_parser_digest": scorer.answer_parser_digest,
            "answer_parser_id": scorer.answer_parser_id,
            "answer_parser_plugin_path": scorer.answer_parser_plugin_path,
            "answer_parser_version": scorer.answer_parser_version,
            "dataset": dataset,
            "metric_names": list(scorer.metric_names),
            "plugin_path": scorer.plugin_path,
            "publication_approved": scorer.publication_approved,
            "scorer_id": scorer.scorer_id,
            "scorer_version": scorer.version,
        }
        for dataset, scorer in registry.entries
    ]


def _validate_quality_projection(
    quality: Mapping[str, Any],
    *,
    scorers: Any,
    inventory_sha256: str,
    shard_plan_sha256: str,
) -> None:
    if quality.get("aggregation_unit") != "per_example_once_never_shard_means":
        raise ValueError("report quality aggregation-unit drift")
    if not _json_type_exact_equal(
        quality.get("bootstrap"),
        _expected_full_score_bootstrap(),
    ):
        raise ValueError("report quality bootstrap contract drift")
    if not _json_type_exact_equal(
        quality.get("protocol"),
        _expected_full_score_protocol(),
    ):
        raise ValueError("report quality full-score protocol drift")

    expected_scorers = _expected_scorer_contracts()
    if (
        not isinstance(scorers, list)
        or len(scorers) != len(expected_scorers)
        or any(not isinstance(item, Mapping) for item in scorers)
    ):
        raise ValueError("report scorer_contracts coverage is invalid")
    for index, scorer in enumerate(scorers):
        _require_exact_keys(
            scorer,
            _SCORER_CONTRACT_KEYS,
            f"report scorer_contracts[{index}]",
        )
    if not _json_type_exact_equal(scorers, expected_scorers):
        raise ValueError("report scorer/parser contract drift")

    dataset_counts = dict(_FULL_SCORE_DATASET_COUNTS)
    if sum(dataset_counts.values()) != PUBLICATION_CAMPAIGN_FULL_SCORE_EXAMPLES:
        raise RuntimeError("frozen full-score dataset counts do not close")
    metric_names_by_dataset = {
        item["dataset"]: tuple(item["metric_names"])
        for item in expected_scorers
    }
    datasets = _mapping(quality, "datasets")
    if frozenset(datasets) != frozenset(SUPPORTED_V1_DATASETS):
        raise ValueError("report quality dataset coverage is incomplete")
    dataset_method_summaries: dict[
        str, dict[str, dict[str, tuple[float, float]]]
    ] = {}
    dataset_paired_means: dict[str, dict[str, float]] = {}
    bootstrap_draws = _required_int(
        _mapping(quality, "bootstrap"),
        "draws",
    )
    for dataset in SUPPORTED_V1_DATASETS:
        dataset_record = _mapping_value(
            datasets.get(dataset),
            f"report quality dataset {dataset}",
        )
        _require_exact_keys(
            dataset_record,
            frozenset({"example_count", "methods", "paired_vanilla_minus_baseline"}),
            f"report quality dataset {dataset}",
        )
        expected_count = dataset_counts[dataset]
        if _required_int(dataset_record, "example_count") != expected_count:
            raise ValueError(f"report quality dataset {dataset} count drift")
        metric_names = metric_names_by_dataset[dataset]
        methods = _mapping(dataset_record, "methods")
        if frozenset(methods) != frozenset(FULL_SCORE_METHODS):
            raise ValueError(f"report quality dataset {dataset} method coverage drift")
        method_summaries = {
            method: _validate_quality_method_record(
                _mapping_value(
                    methods.get(method),
                    f"report quality dataset {dataset} method {method}",
                ),
                expected_count=expected_count,
                metric_names=metric_names,
                label=f"quality.datasets.{dataset}.{method}",
            )
            for method in FULL_SCORE_METHODS
        }
        dataset_method_summaries[dataset] = method_summaries
        dataset_paired_means[dataset] = _validate_quality_paired_metrics(
            _mapping(dataset_record, "paired_vanilla_minus_baseline"),
            dataset_stratum=dataset,
            expected_count=expected_count,
            metric_names=metric_names,
            method_summaries=method_summaries,
            inventory_sha256=inventory_sha256,
            shard_plan_sha256=shard_plan_sha256,
            bootstrap_draws=bootstrap_draws,
            label=f"quality.datasets.{dataset}.paired",
        )

    _validate_quality_niah_grid(
        _mapping(quality, "niah_grid"),
        metric_names=metric_names_by_dataset["niah"],
        dataset_method_summaries=dataset_method_summaries["niah"],
        dataset_paired_means=dataset_paired_means["niah"],
        inventory_sha256=inventory_sha256,
        shard_plan_sha256=shard_plan_sha256,
        bootstrap_draws=bootstrap_draws,
    )


def _validate_quality_method_record(
    record: Mapping[str, Any],
    *,
    expected_count: int,
    metric_names: Sequence[str],
    label: str,
) -> dict[str, tuple[float, float]]:
    _require_exact_keys(
        record,
        frozenset({"example_count", "metrics", "parser_status_counts"}),
        label,
    )
    if _required_int(record, "example_count") != expected_count:
        raise ValueError(f"{label} example count drift")
    metrics = _mapping(record, "metrics")
    if frozenset(metrics) != frozenset(metric_names):
        raise ValueError(f"{label} metric coverage drift")
    summaries = {
        metric: _validate_quality_metric_summary(
            _mapping_value(metrics.get(metric), f"{label}.metrics.{metric}"),
            expected_count=expected_count,
            label=f"{label}.metrics.{metric}",
        )
        for metric in metric_names
    }
    parser_counts = _mapping(record, "parser_status_counts")
    if frozenset(parser_counts) != frozenset(FINAL_ANSWER_PARSER_STATUSES):
        raise ValueError(f"{label} parser-status schema drift")
    observed_count = sum(
        _required_nonnegative_int(parser_counts, status)
        for status in FINAL_ANSWER_PARSER_STATUSES
    )
    if observed_count != expected_count:
        raise ValueError(f"{label} parser-status coverage drift")
    valid_count = _required_nonnegative_int(parser_counts, "ok")
    if any(
        total > valid_count and not _quality_numbers_match(total, float(valid_count))
        for _mean, total in summaries.values()
    ):
        raise ValueError(f"{label} credits an invalid parsed answer")
    return summaries


def _validate_quality_metric_summary(
    record: Mapping[str, Any],
    *,
    expected_count: int,
    label: str,
) -> tuple[float, float]:
    _require_exact_keys(
        record,
        frozenset(
            {"example_count", "invalid_parser_score_sum", "mean", "sum"}
        ),
        label,
    )
    if _required_int(record, "example_count") != expected_count:
        raise ValueError(f"{label} example count drift")
    mean = _finite_quality_number(record.get("mean"), f"{label}.mean")
    total = _finite_quality_number(record.get("sum"), f"{label}.sum")
    invalid_parser_score_sum = _finite_quality_number(
        record.get("invalid_parser_score_sum"),
        f"{label}.invalid_parser_score_sum",
    )
    if invalid_parser_score_sum != 0.0:
        raise ValueError(f"{label} credits an invalid parsed answer")
    if (
        not 0.0 <= mean <= 1.0
        or not 0.0 <= total <= expected_count
        or not _quality_numbers_match(mean, total / expected_count)
    ):
        raise ValueError(f"{label} mean/sum identity drift")
    return mean, total


def _validate_quality_paired_metrics(
    record: Mapping[str, Any],
    *,
    dataset_stratum: str,
    expected_count: int,
    metric_names: Sequence[str],
    method_summaries: Mapping[str, Mapping[str, tuple[float, float]]],
    inventory_sha256: str,
    shard_plan_sha256: str,
    bootstrap_draws: int,
    label: str,
) -> dict[str, float]:
    if frozenset(record) != frozenset(metric_names):
        raise ValueError(f"{label} metric coverage drift")
    means: dict[str, float] = {}
    for metric in metric_names:
        summary = _mapping_value(record.get(metric), f"{label}.{metric}")
        _require_exact_keys(
            summary,
            frozenset({"bootstrap_ci95", "example_count", "mean", "seed_sha256"}),
            f"{label}.{metric}",
        )
        if _required_int(summary, "example_count") != expected_count:
            raise ValueError(f"{label}.{metric} example count drift")
        mean = _finite_quality_number(summary.get("mean"), f"{label}.{metric}.mean")
        expected_mean = (
            method_summaries["vanilla_prefill"][metric][0]
            - method_summaries["baseline_prefill"][metric][0]
        )
        if not -1.0 <= mean <= 1.0 or not _quality_numbers_match(
            mean,
            expected_mean,
        ):
            raise ValueError(f"{label}.{metric} paired mean identity drift")
        expected_seed = _canonical_sha256(
            {
                "dataset_stratum": dataset_stratum,
                "domain": "cachet.full_score.paired_bootstrap.seed.v1",
                "inventory_sha256": inventory_sha256,
                "metric": metric,
                "shard_plan_sha256": shard_plan_sha256,
            }
        )
        if summary.get("seed_sha256") != expected_seed:
            raise ValueError(f"{label}.{metric} deterministic seed drift")
        interval = _mapping(summary, "bootstrap_ci95")
        _require_exact_keys(
            interval,
            frozenset({"draws", "lower", "upper"}),
            f"{label}.{metric}.bootstrap_ci95",
        )
        if _required_int(interval, "draws") != bootstrap_draws:
            raise ValueError(f"{label}.{metric} bootstrap draw drift")
        lower = _finite_quality_number(
            interval.get("lower"),
            f"{label}.{metric}.bootstrap_ci95.lower",
        )
        upper = _finite_quality_number(
            interval.get("upper"),
            f"{label}.{metric}.bootstrap_ci95.upper",
        )
        if not -1.0 <= lower <= upper <= 1.0:
            raise ValueError(f"{label}.{metric} bootstrap CI bounds drift")
        if expected_count == 1 and (
            not _quality_numbers_match(lower, mean)
            or not _quality_numbers_match(upper, mean)
        ):
            raise ValueError(f"{label}.{metric} singleton bootstrap CI drift")
        means[metric] = mean
    return means


def _validate_quality_niah_grid(
    grid: Mapping[str, Any],
    *,
    metric_names: Sequence[str],
    dataset_method_summaries: Mapping[str, Mapping[str, tuple[float, float]]],
    dataset_paired_means: Mapping[str, float],
    inventory_sha256: str,
    shard_plan_sha256: str,
    bootstrap_draws: int,
) -> None:
    if frozenset(grid) != frozenset(NIAH_CELL_IDS):
        raise ValueError("report quality/NIAH table coverage is incomplete")
    observed_count = 0
    metric_counts: dict[tuple[str, str], int] = {}
    metric_sums: dict[tuple[str, str], float] = {}
    paired_weighted_means = {metric: 0.0 for metric in metric_names}
    for cell_index, cell_id in enumerate(NIAH_CELL_IDS):
        cell = _mapping_value(grid.get(cell_id), f"quality.niah_grid.{cell_id}")
        _require_exact_keys(
            cell,
            frozenset({"example_count", "methods", "paired_vanilla_minus_baseline"}),
            f"quality.niah_grid.{cell_id}",
        )
        expected_cell_count = 112 if cell_index == 0 else 111
        cell_count = _required_int(cell, "example_count")
        if cell_count != expected_cell_count:
            raise ValueError(f"quality.niah_grid.{cell_id} example count drift")
        observed_count += cell_count
        methods = _mapping(cell, "methods")
        if frozenset(methods) != frozenset(FULL_SCORE_METHODS):
            raise ValueError(f"quality.niah_grid.{cell_id} method coverage drift")
        method_summaries: dict[str, dict[str, tuple[float, float]]] = {}
        for method in FULL_SCORE_METHODS:
            metrics = _mapping_value(
                methods.get(method),
                f"quality.niah_grid.{cell_id}.{method}",
            )
            if frozenset(metrics) != frozenset(metric_names):
                raise ValueError(
                    f"quality.niah_grid.{cell_id}.{method} metric coverage drift"
                )
            summaries = {
                metric: _validate_quality_metric_summary(
                    _mapping_value(
                        metrics.get(metric),
                        f"quality.niah_grid.{cell_id}.{method}.{metric}",
                    ),
                    expected_count=cell_count,
                    label=f"quality.niah_grid.{cell_id}.{method}.{metric}",
                )
                for metric in metric_names
            }
            method_summaries[method] = summaries
            for metric, (_mean, total) in summaries.items():
                key = (method, metric)
                metric_counts[key] = metric_counts.get(key, 0) + cell_count
                metric_sums[key] = metric_sums.get(key, 0.0) + total
        paired_means = _validate_quality_paired_metrics(
            _mapping(cell, "paired_vanilla_minus_baseline"),
            dataset_stratum=f"niah/{cell_id}",
            expected_count=cell_count,
            metric_names=metric_names,
            method_summaries=method_summaries,
            inventory_sha256=inventory_sha256,
            shard_plan_sha256=shard_plan_sha256,
            bootstrap_draws=bootstrap_draws,
            label=f"quality.niah_grid.{cell_id}.paired",
        )
        for metric, mean in paired_means.items():
            paired_weighted_means[metric] += mean * cell_count
    if observed_count != _FULL_SCORE_NIAH_EXAMPLES:
        raise ValueError("report quality NIAH cell counts do not close")
    for method in FULL_SCORE_METHODS:
        for metric in metric_names:
            key = (method, metric)
            if metric_counts.get(key) != _FULL_SCORE_NIAH_EXAMPLES or not (
                _quality_numbers_match(
                    metric_sums.get(key, 0.0),
                    dataset_method_summaries[method][metric][1],
                )
            ):
                raise ValueError("report quality NIAH cell/dataset metric drift")
    for metric in metric_names:
        if not _quality_numbers_match(
            paired_weighted_means[metric] / _FULL_SCORE_NIAH_EXAMPLES,
            dataset_paired_means[metric],
        ):
            raise ValueError("report quality NIAH paired cell/dataset drift")


def _finite_quality_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite numeric data")
    return float(value)


def _quality_numbers_match(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12 * max(1.0, abs(left), abs(right))


def _record_binding(
    record: Mapping[str, Any],
    *,
    expected_record_type: str,
    expected_schema_version: int,
) -> dict[str, Any]:
    if (
        record.get("record_type") != expected_record_type
        or not _json_type_exact_equal(
            record.get("schema_version"),
            expected_schema_version,
        )
    ):
        raise ValueError(f"{expected_record_type} source envelope drift")
    return {
        "closed_record_sha256": _required_sha256(record, "closed_record_sha256"),
        "record_type": expected_record_type,
        "schema_version": expected_schema_version,
    }


def _assert_sanitized(value: Any, *, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            if key in _FORBIDDEN_REPORT_KEYS or key.endswith("_uri"):
                raise ValueError(f"{path}.{key} is forbidden in sanitized evidence")
            _assert_sanitized(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_sanitized(item, path=f"{path}[{index}]")


def _write_json_exclusive_at(
    directory_descriptor: int,
    file_name: str,
    record: Mapping[str, Any],
) -> None:
    """Atomically publish complete immutable bytes without replacing a name."""

    _require_relative_entry_name(file_name)
    payload = _canonical_pretty_json_bytes(record)
    descriptor = -1
    temporary_name = ""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    for _attempt in range(128):
        candidate = f".{file_name}.{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                candidate,
                flags,
                0o600,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            continue
        temporary_name = candidate
        break
    if descriptor < 0:
        raise RuntimeError("could not allocate a publication temporary file")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            _durability_barrier(handle.fileno())
        os.link(
            temporary_name,
            file_name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        _durability_barrier(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
            _durability_barrier(directory_descriptor)


def _matching_read_only_json_file_exists_at(
    directory_descriptor: int,
    file_name: str,
    record: Mapping[str, Any],
    *,
    label: str,
) -> bool:
    """Accept only an exact immutable partial file for crash recovery."""

    try:
        raw = _read_immutable_file_at(
            directory_descriptor,
            file_name,
            label=label,
        )
    except FileNotFoundError:
        return False
    if raw != _canonical_pretty_json_bytes(record):
        raise ValueError(f"existing {label} differs from the canonical record")
    return True


def _read_canonical_json(path: Path, label: str) -> dict[str, Any]:
    absolute = path.expanduser().absolute()
    directory_descriptor = _open_directory_no_symlinks(
        absolute.parent,
        label=f"{label} directory",
    )
    try:
        record = _read_canonical_json_at(
            directory_descriptor,
            absolute.name,
            label,
        )
        _require_directory_path_matches_descriptor(
            absolute.parent,
            directory_descriptor,
        )
        return record
    finally:
        os.close(directory_descriptor)


def _read_canonical_json_at(
    directory_descriptor: int,
    file_name: str,
    label: str,
    *,
    allowed_modes: frozenset[int] = _SEALED_PUBLICATION_FILE_MODES,
) -> dict[str, Any]:
    raw = _read_immutable_file_at(
        directory_descriptor,
        file_name,
        label=label,
        allowed_modes=allowed_modes,
    )
    try:
        record = json.loads(raw, parse_constant=_reject_nonfinite_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid strict JSON") from exc
    if not isinstance(record, dict) or raw != _canonical_pretty_json_bytes(record):
        raise ValueError(f"{label} bytes are not canonical")
    return record


def _read_immutable_file_at(
    directory_descriptor: int,
    file_name: str,
    *,
    label: str,
    allowed_modes: frozenset[int] = _SEALED_PUBLICATION_FILE_MODES,
) -> bytes:
    _require_relative_entry_name(file_name)
    if not allowed_modes or any(type(mode) is not int for mode in allowed_modes):
        raise TypeError("allowed publication evidence modes are invalid")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            file_name,
            flags,
            dir_fd=directory_descriptor,
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"{label} could not be opened without following links") from exc
    try:
        opened_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or stat.S_IMODE(opened_stat.st_mode) not in allowed_modes
        ):
            raise ValueError(f"{label} must be a regular read-only file")
        first_read = _read_descriptor_bytes(descriptor)
        after_first_read_stat = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second_read = _read_descriptor_bytes(descriptor)
        after_second_read_stat = os.fstat(descriptor)
        try:
            named_stat = os.stat(
                file_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise ValueError(f"{label} changed while it was read") from exc
        if (
            not stat.S_ISREG(named_stat.st_mode)
            or stat.S_IMODE(named_stat.st_mode) not in allowed_modes
            or first_read != second_read
            or _stable_file_stat_identity(opened_stat)
            != _stable_file_stat_identity(after_first_read_stat)
            or _stable_file_stat_identity(opened_stat)
            != _stable_file_stat_identity(after_second_read_stat)
            or _stable_file_stat_identity(opened_stat)
            != _stable_file_stat_identity(named_stat)
        ):
            raise ValueError(f"{label} changed while it was read")
        return first_read
    finally:
        os.close(descriptor)


def _read_descriptor_bytes(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _stable_file_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _remove_stale_publication_temporaries_at(
    directory_descriptor: int,
) -> None:
    removed = False
    for file_name in os.listdir(directory_descriptor):
        if not _is_publication_temporary_name(file_name):
            continue
        try:
            file_stat = os.stat(
                file_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != os.geteuid()
            or stat.S_IMODE(file_stat.st_mode)
            not in {0o000, 0o200, 0o400, 0o444, 0o600}
        ):
            raise ValueError("publication temporary entry is unsafe")
        os.unlink(file_name, dir_fd=directory_descriptor)
        removed = True
    if removed:
        _durability_barrier(directory_descriptor)


def _reject_stale_publication_temporaries_at(
    directory_descriptor: int,
) -> None:
    if any(
        _is_publication_temporary_name(file_name)
        for file_name in os.listdir(directory_descriptor)
    ):
        raise ValueError("publication evidence directory has a stale temporary file")


def _is_publication_temporary_name(value: str) -> bool:
    if not isinstance(value, str):
        return False
    for final_name in (
        PUBLICATION_CAMPAIGN_GATE_FILE_NAME,
        PUBLICATION_CAMPAIGN_REPORT_FILE_NAME,
    ):
        prefix = f".{final_name}."
        if value.startswith(prefix) and value.endswith(".tmp"):
            token = value[len(prefix) : -len(".tmp")]
            return len(token) == 32 and all(
                character in "0123456789abcdef" for character in token
            )
    return False


def _durability_barrier(descriptor: int) -> None:
    full_fsync = getattr(fcntl, "F_FULLFSYNC", None)
    if full_fsync is not None:
        try:
            fcntl.fcntl(descriptor, full_fsync)
            return
        except OSError as exc:
            unsupported_errors = {
                errno.EINVAL,
                errno.ENOTTY,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if exc.errno not in unsupported_errors:
                raise
    os.fsync(descriptor)


def _open_directory_no_symlinks(path: Path, *, label: str) -> int:
    absolute = path.expanduser().absolute()
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(absolute.anchor, flags)
        for part in absolute.parts[1:]:
            _require_relative_entry_name(part)
            next_descriptor = os.open(
                part,
                flags,
                dir_fd=descriptor,
            )
            previous_descriptor = descriptor
            descriptor = next_descriptor
            os.close(previous_descriptor)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a real directory")
        return descriptor
    except (OSError, ValueError) as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(exc, ValueError):
            raise
        if exc.errno in {errno.ENOENT, errno.ENOTDIR}:
            raise ValueError(
                f"{label} must already exist as a real directory"
            ) from exc
        raise ValueError(f"{label} must contain no symlink components") from exc


def _require_directory_path_matches_descriptor(path: Path, descriptor: int) -> None:
    try:
        named_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError("publication evidence directory changed") from exc
    opened_stat = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(named_stat.st_mode)
        or (named_stat.st_dev, named_stat.st_ino)
        != (opened_stat.st_dev, opened_stat.st_ino)
    ):
        raise ValueError("publication evidence directory changed")


def _acquire_publication_directory_lock(
    directory_descriptor: int,
    *,
    exclusive: bool,
) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(".", flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise ValueError("publication evidence directory lock is unavailable") from exc
    try:
        opened_stat = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened_stat.st_mode)
            or (opened_stat.st_dev, opened_stat.st_ino)
            != (
                os.fstat(directory_descriptor).st_dev,
                os.fstat(directory_descriptor).st_ino,
            )
        ):
            raise ValueError("publication evidence directory lock drift")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise RuntimeError("publication evidence directory is busy") from exc
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _release_publication_directory_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _require_relative_entry_name(value: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or Path(value).is_absolute()
    ):
        raise ValueError("publication evidence entry name is invalid")


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _mapping(record: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    return _mapping_value(record.get(field_name), field_name)


def _mapping_value(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _mapping_sequence(
    record: Mapping[str, Any], field_name: str
) -> list[Mapping[str, Any]]:
    raw = record.get(field_name)
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be an array")
    values: list[Mapping[str, Any]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ValueError(f"{field_name}[{index}] must be an object")
        values.append(value)
    return values


def _required_int(record: Mapping[str, Any], field_name: str) -> int:
    value = record.get(field_name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _required_nonnegative_int(record: Mapping[str, Any], field_name: str) -> int:
    value = record.get(field_name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _required_sha256(record: Mapping[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_exact_keys(
    record: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(record)
    if actual != expected:
        raise ValueError(
            f"{label} must use a closed schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _json_object_copy(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} must be a mapping")
    value = _json_value_copy(dict(record))
    if not isinstance(value, dict):  # pragma: no cover - mapping above.
        raise TypeError(f"{label} must encode as an object")
    return value


def _json_value_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _json_type_exact_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int/float equality coercions."""

    try:
        return json.dumps(
            left,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ) == json.dumps(
            right,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return False


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("closed_record_sha256", None)
    return _canonical_sha256(payload)


def _canonical_pretty_json_bytes(record: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(record),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "PUBLICATION_CAMPAIGN_GATE_FILE_NAME",
    "PUBLICATION_CAMPAIGN_REPORT_FILE_NAME",
    "PUBLICATION_CAMPAIGN_REPORT_RECORD_TYPE",
    "PUBLICATION_CAMPAIGN_REPORT_SCHEMA_VERSION",
    "PublicationCampaignFinalization",
    "finalize_vllm_0271_publication_campaign",
    "load_vllm_0271_publication_finalization",
    "validate_vllm_0271_publication_finalization",
    "validate_vllm_0271_publication_report_pair",
    "write_vllm_0271_publication_finalization",
]
