from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

from document_kv_cache._benchmark_manifest import (
    BENCHMARK_EXPERIMENT_MANIFEST_RECORD_TYPE,
    VARIES_BY_ARM,
    _json_materialize,
    _runtime_environment_to_record,
    _resource_execution_id_digest,
    _resource_runtime_identity_digest,
    _sha256_json,
    _validate_arm_runtime_environments,
    _validate_comparison_design,
    benchmark_experiment_manifest_to_record,
)
from document_kv_cache._benchmark_models import (
    BENCHMARK_ARM_ENVIRONMENT_FIELDS,
    BenchmarkArmEnvironment,
    BenchmarkArmManifest,
    BenchmarkExecutionWindow,
    BenchmarkExperimentManifest,
    BenchmarkResourceEvidence,
    BenchmarkRunResult,
    BenchmarkScorerManifest,
    _deep_freeze_json_mapping,
    _json_object_mapping,
    _validate_non_empty_string,
)
from document_kv_cache.benchmarks import (
    CACHE_REUSE_ARM,
    BenchmarkArm,
    BenchmarkComparison,
    BenchmarkExample,
    BenchmarkOfflineCosts,
    BenchmarkReportRow,
    BenchmarkSuite,
    DatasetMetricSpec,
    InferenceMeasurement,
    compare_to_baseline,
    evaluate_v1_benchmark_evidence,
    summarize_measurements,
)
from document_kv_cache.storage import local_path
from document_kv_cache.workflow import SourceDocument


BENCHMARK_RUN_RECORD_TYPE = "document_kv.benchmark_run.v1"
BENCHMARK_RESOURCE_EVIDENCE_RECORD_TYPE = (
    "document_kv.benchmark_resource_evidence.v1"
)

_SANITIZED_DIGEST_METADATA_KEYS = frozenset(
    {
        "logical_prompt_sha256",
        "runtime_prompt_sha256",
        "request_payload_prompt_sha256",
        "prefix_cache_salt_sha256",
    }
)
_SANITIZED_INTEGER_METADATA_KEYS = frozenset(
    {
        "logical_prompt_tokens",
        "runtime_prompt_tokens",
        "server_usage_prompt_tokens",
        "gpu_memory_bytes",
        "cpu_memory_bytes",
        "storage_read_bytes",
    }
)
_SANITIZED_FLOAT_METADATA_KEYS = frozenset(
    {"gpu_utilization", "cpu_utilization", "energy_joules"}
)
_SANITIZED_BOOLEAN_METADATA_KEYS = frozenset(
    {
        "request_payload_add_special_tokens",
        "stream",
        "kv_transfer_params_attached",
        "prefix_cache_salt_attached",
        "server_usage_prompt_tokens_present",
    }
)
_SANITIZED_ENUM_METADATA_VALUES = {
    "server": frozenset({"openai-compatible"}),
    "request_mode": frozenset({"completion", "chat"}),
    "prompt_text_mode": frozenset({"logical", "runtime"}),
    "prompt_token_source": frozenset({"logical", "runtime", "server_usage"}),
}
SANITIZED_MEASUREMENT_METADATA_KEYS = frozenset(
    _SANITIZED_DIGEST_METADATA_KEYS
    | _SANITIZED_INTEGER_METADATA_KEYS
    | _SANITIZED_FLOAT_METADATA_KEYS
    | _SANITIZED_BOOLEAN_METADATA_KEYS
    | frozenset(_SANITIZED_ENUM_METADATA_VALUES)
)


def benchmark_run_result_payload_to_record(
    result: BenchmarkRunResult,
    *,
    sanitize_evidence: bool = False,
) -> dict[str, Any]:
    """Serialize the benchmark payload that an evidence gate attests.

    The gate itself is intentionally excluded so this representation has no
    recursive dependency and can be hashed identically at run and release time.
    """

    if not isinstance(result, BenchmarkRunResult):
        raise TypeError("result must be a BenchmarkRunResult")
    _validate_resource_evidence_bindings(result)
    if sanitize_evidence:
        result = _sanitized_gate_result(result)
    from document_kv_cache.benchmark_statistics import (
        paired_benchmark_statistics,
        paired_benchmark_statistics_to_record,
    )

    record: dict[str, Any] = {
        "record_type": BENCHMARK_RUN_RECORD_TYPE,
        "suite": {
            "suite_id": result.suite.suite_id,
            "model_id": result.suite.model_id,
            "hardware_target": result.suite.hardware_target,
            "datasets": list(result.suite.datasets),
            "examples": len(result.suite.examples),
            "request_parallelism": result.request_parallelism,
            "isolate_arms": result.isolate_arms,
            "repeats": result.repeats,
            "shuffle": result.shuffle,
            "seed": result.seed,
            "interleave_examples": result.interleave_examples,
            "prefix_cache_salt_mode": result.prefix_cache_salt_mode,
            "arms": [
                {
                    "arm_id": arm.arm_id,
                    "uses_cache": arm.uses_cache,
                    "cache_method": arm.cache_method,
                    "connector_mode": arm.connector_mode,
                    "variant_id": arm.variant_id,
                    "description": arm.description,
                }
                for arm in result.arms
            ],
        },
        "measurements": [
            _measurement_to_record(
                measurement,
                sanitize_evidence=sanitize_evidence,
            )
            for measurement in result.measurements
        ],
        "execution_windows": [
            {
                "arm_id": window.arm_id,
                "wall_seconds": window.wall_seconds,
                "started_at_seconds": window.started_at_seconds,
                "ended_at_seconds": window.ended_at_seconds,
                "completion_tokens": window.completion_tokens,
                "successful_requests": window.successful_requests,
                "aggregate_output_tokens_per_second": (
                    window.aggregate_output_tokens_per_second
                ),
            }
            for window in result.execution_windows
        ],
        "resource_evidence": [
            benchmark_resource_evidence_to_record(evidence)
            for evidence in result.resource_evidence
        ],
        "report_rows": [_report_row_to_record(row) for row in result.report_rows],
        "comparisons": [
            _comparison_to_record(comparison) for comparison in result.comparisons
        ],
        "paired_statistics": paired_benchmark_statistics_to_record(
            paired_benchmark_statistics(result)
        ),
        "v1_evidence": _v1_evidence_to_record(
            evaluate_v1_benchmark_evidence(
                _rows_for_arm_pair(
                    result.report_rows,
                    baseline_arm_id=result.baseline_arm_id,
                    cache_arm_id=result.cache_arm_id,
                ),
                _comparisons_for_arm_pair(
                    result.comparisons,
                    baseline_arm_id=result.baseline_arm_id,
                    cache_arm_id=result.cache_arm_id,
                ),
                baseline_arm_id=result.baseline_arm_id,
                cache_arm_id=result.cache_arm_id,
            )
        ),
        "v1_evidence_by_cache_arm": {
            cache_arm_id: _v1_evidence_to_record(
                evaluate_v1_benchmark_evidence(
                    _rows_for_arm_pair(
                        result.report_rows,
                        baseline_arm_id=result.baseline_arm_id,
                        cache_arm_id=cache_arm_id,
                    ),
                    _comparisons_for_arm_pair(
                        result.comparisons,
                        baseline_arm_id=result.baseline_arm_id,
                        cache_arm_id=cache_arm_id,
                    ),
                    baseline_arm_id=result.baseline_arm_id,
                    cache_arm_id=cache_arm_id,
                )
            )
            for cache_arm_id in result.cache_arm_ids
        },
    }
    if result.experiment_manifest is not None:
        record["experiment_manifest"] = benchmark_experiment_manifest_to_record(
            result.experiment_manifest
        )
    if sanitize_evidence:
        record["evidence_sanitized"] = True
    materialized = _json_materialize(record)
    if not isinstance(materialized, dict):  # pragma: no cover - defensive invariant.
        raise TypeError("benchmark payload must serialize to an object")
    return materialized


def benchmark_record_payload_digest(record: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 binding for a benchmark record payload."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    payload = {key: value for key, value in record.items() if key != "evidence_gate"}
    return _sha256_json(payload)


def benchmark_resource_evidence_to_record(
    evidence: BenchmarkResourceEvidence,
) -> dict[str, Any]:
    """Serialize one immutable arm-level resource evidence record."""

    if not isinstance(evidence, BenchmarkResourceEvidence):
        raise TypeError("evidence must be BenchmarkResourceEvidence")
    record: dict[str, Any] = {
        "record_type": BENCHMARK_RESOURCE_EVIDENCE_RECORD_TYPE,
        "experiment_id": evidence.experiment_id,
        "arm_id": evidence.arm_id,
        "execution_id_digest": evidence.execution_id_digest,
        "measurement_window": {
            "started_at_seconds": evidence.measurement_started_at_seconds,
            "ended_at_seconds": evidence.measurement_ended_at_seconds,
        },
        "sampling": {
            "interval_seconds": evidence.sampling_interval_seconds,
            "first_sample_at_seconds": evidence.first_sample_at_seconds,
            "last_sample_at_seconds": evidence.last_sample_at_seconds,
            "max_sample_gap_seconds": evidence.max_sample_gap_seconds,
            "expected_sample_count": evidence.expected_sample_count,
            "sample_count": evidence.sample_count,
            "error_count": evidence.error_count,
            "complete": evidence.complete,
        },
        "telemetry_sha256": evidence.telemetry_sha256,
        "metrics": {
            "peak_gpu_process_memory_bytes": (
                evidence.peak_gpu_process_memory_bytes
            ),
            "mean_gpu_utilization_percent": (
                evidence.mean_gpu_utilization_percent
            ),
            "peak_gpu_utilization_percent": (
                evidence.peak_gpu_utilization_percent
            ),
            "peak_process_tree_rss_bytes": evidence.peak_process_tree_rss_bytes,
            "peak_host_memory_used_bytes": (
                evidence.peak_host_memory_used_bytes
            ),
        },
        "software_identity": {
            "source_revision": evidence.source_revision,
            "source_tree_sha256": evidence.source_tree_sha256,
            "wheel_sha256": evidence.wheel_sha256,
            "runner_sha256": evidence.runner_sha256,
            "runtime_identity_sha256": evidence.runtime_identity_sha256,
        },
    }
    record["record_sha256"] = _sha256_json(record)
    return record


def benchmark_resource_evidence_from_record(
    record: Mapping[str, Any],
) -> BenchmarkResourceEvidence:
    """Parse and authenticate one arm-level resource evidence record."""

    if not isinstance(record, Mapping):
        raise TypeError("resource evidence record must be a mapping")
    _require_record_keys(
        record,
        {
            "record_type",
            "experiment_id",
            "arm_id",
            "execution_id_digest",
            "measurement_window",
            "sampling",
            "telemetry_sha256",
            "metrics",
            "software_identity",
            "record_sha256",
        },
        "resource_evidence",
    )
    if record.get("record_type") != BENCHMARK_RESOURCE_EVIDENCE_RECORD_TYPE:
        raise ValueError("unsupported resource evidence record_type")
    recorded_digest = _record_string(record, "record_sha256")
    payload = dict(record)
    payload.pop("record_sha256")
    if recorded_digest != _sha256_json(payload):
        raise ValueError("resource evidence record_sha256 does not match its payload")
    window = _record_mapping(record, "measurement_window")
    sampling = _record_mapping(record, "sampling")
    metrics = _record_mapping(record, "metrics")
    identity = _record_mapping(record, "software_identity")
    return BenchmarkResourceEvidence(
        experiment_id=_record_string(record, "experiment_id"),
        arm_id=_record_string(record, "arm_id"),
        execution_id_digest=_record_string(record, "execution_id_digest"),
        measurement_started_at_seconds=_record_number(
            window,
            "started_at_seconds",
        ),
        measurement_ended_at_seconds=_record_number(window, "ended_at_seconds"),
        sampling_interval_seconds=_record_number(sampling, "interval_seconds"),
        first_sample_at_seconds=_record_number(
            sampling,
            "first_sample_at_seconds",
        ),
        last_sample_at_seconds=_record_number(
            sampling,
            "last_sample_at_seconds",
        ),
        max_sample_gap_seconds=_record_number(
            sampling,
            "max_sample_gap_seconds",
        ),
        expected_sample_count=_record_int(sampling, "expected_sample_count"),
        sample_count=_record_int(sampling, "sample_count"),
        error_count=_record_int(sampling, "error_count"),
        complete=_record_bool(sampling, "complete"),
        telemetry_sha256=_record_string(record, "telemetry_sha256"),
        peak_gpu_process_memory_bytes=_record_int(
            metrics,
            "peak_gpu_process_memory_bytes",
        ),
        mean_gpu_utilization_percent=_record_number(
            metrics,
            "mean_gpu_utilization_percent",
        ),
        peak_gpu_utilization_percent=_record_number(
            metrics,
            "peak_gpu_utilization_percent",
        ),
        peak_process_tree_rss_bytes=_record_int(
            metrics,
            "peak_process_tree_rss_bytes",
        ),
        peak_host_memory_used_bytes=_record_int(
            metrics,
            "peak_host_memory_used_bytes",
        ),
        source_revision=_record_string(identity, "source_revision"),
        source_tree_sha256=_record_string(identity, "source_tree_sha256"),
        wheel_sha256=_record_string(identity, "wheel_sha256"),
        runner_sha256=_record_string(identity, "runner_sha256"),
        runtime_identity_sha256=_record_string(
            identity,
            "runtime_identity_sha256",
        ),
    )


def _validate_resource_evidence_bindings(result: BenchmarkRunResult) -> None:
    evidence_by_arm = {item.arm_id: item for item in result.resource_evidence}
    manifest = result.experiment_manifest
    if manifest is None:
        if evidence_by_arm:
            raise ValueError("resource evidence requires an experiment manifest")
        return
    evidence_ids = {
        arm_id: digest for arm_id, digest in manifest.resource_evidence_ids
    }
    actual_ids = {
        arm_id: benchmark_resource_evidence_to_record(evidence)["record_sha256"]
        for arm_id, evidence in evidence_by_arm.items()
    }
    if actual_ids != evidence_ids:
        raise ValueError(
            "experiment manifest resource_evidence_ids do not match resource evidence"
        )
    windows_by_arm = {
        window.arm_id: window
        for window in result.execution_windows
        if window.arm_id != "all_arms"
    }
    for arm_id, evidence in evidence_by_arm.items():
        if evidence.experiment_id != manifest.experiment_id:
            raise ValueError(
                f"resource evidence for arm {arm_id!r} changes experiment identity"
            )
        expected_execution_id = _resource_execution_id_digest(manifest, arm_id)
        if evidence.execution_id_digest != expected_execution_id:
            raise ValueError(
                f"resource evidence for arm {arm_id!r} changes execution identity"
            )
        expected_runtime_identity = _resource_runtime_identity_digest(
            manifest,
            arm_id,
            execution_id_digest=evidence.execution_id_digest,
        )
        if evidence.runtime_identity_sha256 != expected_runtime_identity:
            raise ValueError(
                f"resource evidence for arm {arm_id!r} changes runtime identity"
            )
        window = windows_by_arm.get(arm_id)
        if window is None:
            raise ValueError(
                f"resource evidence for arm {arm_id!r} has no execution window"
            )
        if window.started_at_seconds is None or window.ended_at_seconds is None:
            raise ValueError(
                f"resource evidence for arm {arm_id!r} requires a timestamped "
                "execution window"
            )
        if (
            evidence.measurement_started_at_seconds != window.started_at_seconds
            or evidence.measurement_ended_at_seconds != window.ended_at_seconds
        ):
            raise ValueError(
                f"resource evidence for arm {arm_id!r} changes the measurement window"
            )


def benchmark_run_result_to_record(
    result: BenchmarkRunResult,
    *,
    cache_state_attestations: Iterable[Any] = (),
    artifact_identities: Mapping[str, Any] | None = None,
    method_registry: Any = None,
    sanitize_evidence: bool = False,
) -> dict[str, Any]:
    attestations = tuple(cache_state_attestations)
    record = benchmark_run_result_payload_to_record(
        result,
        sanitize_evidence=sanitize_evidence,
    )
    identities = {} if artifact_identities is None else dict(artifact_identities)
    if attestations or identities:
        record["gate_inputs"] = {
            "artifact_identities": [
                identity.to_record()
                for _artifact_id, identity in sorted(identities.items())
            ],
            "cache_state_attestations": [
                _cache_state_attestation_to_record(
                    attestation,
                    sanitize_evidence=sanitize_evidence,
                )
                for attestation in attestations
            ],
        }
    benchmark_payload_digest = benchmark_record_payload_digest(record)
    from document_kv_cache.benchmark_gates import (
        benchmark_evidence_gate_to_record,
        evaluate_benchmark_evidence_gate,
    )

    gate_result = _sanitized_gate_result(result) if sanitize_evidence else result
    gate_attestations = (
        tuple(_sanitized_cache_state_attestation(item) for item in attestations)
        if sanitize_evidence
        else attestations
    )
    record["evidence_gate"] = benchmark_evidence_gate_to_record(
        evaluate_benchmark_evidence_gate(
            gate_result,
            policy=gate_result.evidence_policy,
            cache_state_attestations=gate_attestations,
            artifact_identities=identities,
            method_registry=method_registry,
            benchmark_payload_digest=benchmark_payload_digest,
        )
    )
    return record


def benchmark_run_result_to_evidence_record(
    result: BenchmarkRunResult,
    *,
    cache_state_attestations: Iterable[Any] = (),
    artifact_identities: Mapping[str, Any] | None = None,
    method_registry: Any = None,
) -> dict[str, Any]:
    """Serialize a gate-bound record without raw prompts, answers, or outputs."""

    return benchmark_run_result_to_record(
        result,
        cache_state_attestations=cache_state_attestations,
        artifact_identities=artifact_identities,
        method_registry=method_registry,
        sanitize_evidence=True,
    )


def benchmark_gate_inputs_from_record(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Reconstruct the external inputs needed to re-gate a benchmark record."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    raw_inputs = record.get("gate_inputs")
    if raw_inputs is None:
        return {}, ()
    if not isinstance(raw_inputs, Mapping):
        raise ValueError("gate_inputs must be an object")
    raw_identities = raw_inputs.get("artifact_identities", ())
    raw_attestations = raw_inputs.get("cache_state_attestations", ())
    for value, field_name in (
        (raw_identities, "artifact_identities"),
        (raw_attestations, "cache_state_attestations"),
    ):
        if not isinstance(value, Sequence) or isinstance(
            value,
            (str, bytes, bytearray),
        ):
            raise ValueError(f"gate_inputs.{field_name} must be an array")
    from document_kv_cache.artifact_identity import ArtifactIdentity
    from document_kv_cache.benchmark_gates import (
        CACHE_STATE_ATTESTATION_RECORD_TYPE,
        CacheStateAttestation,
    )

    identities: dict[str, Any] = {}
    for raw_identity in raw_identities:
        if not isinstance(raw_identity, Mapping):
            raise ValueError("artifact identity descriptor must be an object")
        identity = ArtifactIdentity.from_record(raw_identity)
        if identity.artifact_id in identities:
            raise ValueError("gate_inputs contains a duplicate artifact identity")
        identities[identity.artifact_id] = identity
    attestations: list[Any] = []
    for raw_attestation in raw_attestations:
        if not isinstance(raw_attestation, Mapping):
            raise ValueError("cache-state attestation must be an object")
        if raw_attestation.get("record_type") != CACHE_STATE_ATTESTATION_RECORD_TYPE:
            raise ValueError("cache-state attestation has an unsupported record_type")
        values = {
            key: value
            for key, value in raw_attestation.items()
            if key not in {"record_type", "cold_read_attested"}
        }
        attestations.append(CacheStateAttestation(**values))
    return identities, tuple(attestations)


def _arm_runtime_environment_from_record(
    arm: Mapping[str, Any],
    *,
    model_runtime: Mapping[str, Any],
    environment: Mapping[str, Any],
    workload: Mapping[str, Any],
    label: str,
) -> BenchmarkArmEnvironment:
    raw = arm.get("runtime_environment")
    if raw is None:
        # Additive V1 compatibility: older records used only global provenance.
        values: Mapping[str, Any] = {
            "served_model_id": model_runtime.get("model_id"),
            "canonical_model_id": model_runtime.get(
                "canonical_model_id",
                model_runtime.get("model_id"),
            ),
            "model_revision": model_runtime.get("model_revision"),
            "tokenizer_id": model_runtime.get("tokenizer_id"),
            "tokenizer_revision": model_runtime.get("tokenizer_revision"),
            "lora_id": model_runtime.get("lora_id", "base"),
            "prompt_template_version": model_runtime.get(
                "prompt_template_version",
                workload.get("prompt_template_version"),
            ),
            "engine_id": model_runtime.get("engine_id"),
            "engine_version": model_runtime.get("engine_version"),
            "serving_platform": model_runtime.get("serving_platform", "unresolved"),
            "hardware_target": environment.get("hardware_target"),
            "hardware_fingerprint": environment.get("hardware_fingerprint"),
            "model_dtype": model_runtime.get("model_dtype", "unresolved"),
            "model_quantization": model_runtime.get("model_quantization", "none"),
            "runtime_kv_dtype": model_runtime.get(
                "runtime_kv_dtype",
                "unresolved",
            ),
            "layout_version": model_runtime.get("layout_version", "unresolved"),
            "payload_axis_order": model_runtime.get(
                "payload_axis_order",
                "unresolved",
            ),
            "block_size": model_runtime.get("block_size"),
            "key_position_encoding": model_runtime.get(
                "key_position_encoding",
                "unresolved",
            ),
            "rope_theta": model_runtime.get("rope_theta"),
            "rope_rotary_dim": model_runtime.get("rope_rotary_dim"),
            "tensor_parallel_size": model_runtime.get("tensor_parallel_size"),
            "pipeline_parallel_size": model_runtime.get("pipeline_parallel_size"),
            "runtime_version": environment.get("runtime_version"),
            "storage_identity": environment.get("storage_identity"),
            "cache_state": environment.get("cache_state"),
        }
    else:
        values = _record_mapping_value(raw, label)
        _require_record_keys(
            values,
            set(BENCHMARK_ARM_ENVIRONMENT_FIELDS),
            label,
        )
    return BenchmarkArmEnvironment(
        served_model_id=_record_string(values, "served_model_id"),
        canonical_model_id=_record_string(values, "canonical_model_id"),
        model_revision=_record_string(values, "model_revision"),
        tokenizer_id=_record_string(values, "tokenizer_id"),
        tokenizer_revision=_record_string(values, "tokenizer_revision"),
        lora_id=_record_string(values, "lora_id"),
        prompt_template_version=_record_string(values, "prompt_template_version"),
        engine_id=_record_string(values, "engine_id"),
        engine_version=_record_string(values, "engine_version"),
        serving_platform=_record_string(values, "serving_platform"),
        hardware_target=_record_string(values, "hardware_target"),
        hardware_fingerprint=_record_string(values, "hardware_fingerprint"),
        model_dtype=_record_string(values, "model_dtype"),
        model_quantization=_record_string(values, "model_quantization"),
        runtime_kv_dtype=_record_string(values, "runtime_kv_dtype"),
        layout_version=_record_string(values, "layout_version"),
        payload_axis_order=_record_string(values, "payload_axis_order"),
        block_size=_record_optional_int(values, "block_size"),
        key_position_encoding=_record_string(values, "key_position_encoding"),
        rope_theta=_record_optional_number(values, "rope_theta"),
        rope_rotary_dim=_record_optional_int(values, "rope_rotary_dim"),
        tensor_parallel_size=_record_optional_int(values, "tensor_parallel_size"),
        pipeline_parallel_size=_record_optional_int(
            values,
            "pipeline_parallel_size",
        ),
        runtime_version=_record_string(values, "runtime_version"),
        storage_identity=_record_string(values, "storage_identity"),
        cache_state=_record_string(values, "cache_state"),
    )


def benchmark_experiment_manifest_from_record(
    record: Mapping[str, Any],
) -> BenchmarkExperimentManifest:
    """Parse the closed v1 experiment-manifest schema."""

    _require_record_keys(
        record,
        {
            "record_type",
            "manifest_version",
            "experiment_id",
            "comparison",
            "logical_workload",
            "decoding",
            "model_runtime",
            "environment",
            "execution",
            "arms",
            "has_unresolved_provenance",
        },
        "experiment_manifest",
    )
    if record.get("record_type") != BENCHMARK_EXPERIMENT_MANIFEST_RECORD_TYPE:
        raise ValueError("unsupported experiment manifest record_type")
    if record.get("manifest_version") != 1:
        raise ValueError("unsupported experiment manifest version")
    comparison = _record_mapping(record, "comparison")
    workload = _record_mapping(record, "logical_workload")
    decoding = _record_mapping(record, "decoding")
    model_runtime = _record_mapping(record, "model_runtime")
    environment = _record_mapping(record, "environment")
    execution = _record_mapping(record, "execution")
    raw_scorers = _record_sequence(workload, "scorers")
    scorer_identities: list[BenchmarkScorerManifest] = []
    for index, raw_scorer in enumerate(raw_scorers):
        scorer = _record_mapping_value(raw_scorer, f"scorers[{index}]")
        metrics = tuple(
            DatasetMetricSpec(
                metric_name=_record_string(
                    _record_mapping_value(raw_metric, f"metrics[{metric_index}]"),
                    "metric_name",
                ),
                direction=_record_string(
                    _record_mapping_value(raw_metric, f"metrics[{metric_index}]"),
                    "direction",
                ),  # type: ignore[arg-type]
                max_regression=_record_number(
                    _record_mapping_value(raw_metric, f"metrics[{metric_index}]"),
                    "max_regression",
                ),
            )
            for metric_index, raw_metric in enumerate(
                _record_sequence(scorer, "metrics")
            )
        )
        scorer_identities.append(
            BenchmarkScorerManifest(
                dataset=_record_string(scorer, "dataset"),
                scorer_id=_record_string(scorer, "scorer_id"),
                version=_record_string(scorer, "version"),
                plugin_path=_record_string(scorer, "plugin_path"),
                publication_approved=_record_bool(
                    scorer,
                    "publication_approved",
                ),
                metric_specs=metrics,
                prompt_plugin_path=_record_optional_string(
                    scorer,
                    "prompt_plugin_path",
                ),
                prompt_template_version=_record_optional_string(
                    scorer,
                    "prompt_template_version",
                ),
            )
        )
    arm_manifests: list[BenchmarkArmManifest] = []
    for index, raw_arm in enumerate(_record_sequence(record, "arms")):
        arm = _record_mapping_value(raw_arm, f"arms[{index}]")
        physical = _record_mapping(arm, "physical_transform")
        request_customization = _record_mapping(arm, "request_customization")
        _require_record_keys(
            request_customization,
            {"config_digest"},
            f"arms[{index}].request_customization",
        )
        offline = _record_mapping(arm, "offline_costs")
        arm_manifests.append(
            BenchmarkArmManifest(
                arm_id=_record_string(arm, "arm_id"),
                implementation_kind=_record_string(arm, "implementation_kind"),
                uses_cache=_record_bool(arm, "uses_cache"),
                method_id=_record_optional_string(arm, "method_id"),
                method_version=_record_optional_string(arm, "method_version"),
                method_config_digest=_record_optional_string(
                    arm,
                    "method_config_digest",
                ),
                artifact_ids=tuple(
                    _record_string_value(value, "artifact_ids entry")
                    for value in _record_sequence(arm, "artifact_ids")
                ),
                variant_id=_record_optional_string(arm, "variant_id"),
                connector_mode=_record_optional_string(arm, "connector_mode"),
                physical_transform_id=_record_string(physical, "transform_id"),
                physical_transform_version=_record_string(physical, "version"),
                declared_physical_transform_config_digest=(
                    _record_optional_string(physical, "declared_config_digest")
                ),
                physical_transform_config_digest=_record_string(
                    physical,
                    "config_digest",
                ),
                request_customization_digest=_record_string(
                    request_customization,
                    "config_digest",
                ),
                scorer_plugin_path=_record_optional_string(
                    arm,
                    "scorer_plugin_path",
                ),
                offline_training_seconds=_record_optional_number(
                    offline,
                    "training_seconds",
                ),
                offline_artifact_generation_seconds=_record_optional_number(
                    offline,
                    "artifact_generation_seconds",
                ),
                offline_checkpoint_load_seconds=_record_optional_number(
                    offline,
                    "checkpoint_load_seconds",
                ),
                artifact_bytes=_record_optional_int(offline, "artifact_bytes"),
                offline_peak_memory_bytes=_record_optional_int(
                    offline,
                    "peak_memory_bytes",
                ),
                source_revision=_record_optional_string(arm, "source_revision"),
                checkpoint_identity=_record_optional_string(
                    arm,
                    "checkpoint_identity",
                ),
                setting_overrides=_deep_freeze_json_mapping(
                    _json_object_mapping(
                        arm.get("setting_overrides", {}),
                        f"arms[{index}].setting_overrides",
                    )
                ),
                requires_cachet_handoff=(
                    _record_bool(arm, "requires_cachet_handoff")
                    if "requires_cachet_handoff" in arm
                    else (
                        _record_bool(arm, "uses_cache")
                        and _record_string(arm, "implementation_kind") == "cachet"
                    )
                ),
                runtime_environment=_arm_runtime_environment_from_record(
                    arm,
                    model_runtime=model_runtime,
                    environment=environment,
                    workload=workload,
                    label=f"arms[{index}].runtime_environment",
                ),
            )
        )
    dataset_digests = _record_mapping(workload, "dataset_sample_digests")
    package_revisions = _record_mapping(model_runtime, "package_revisions")
    manifest = BenchmarkExperimentManifest(
        experiment_id=_record_string(record, "experiment_id"),
        baseline_arm_id=_record_string(comparison, "baseline_arm_id"),
        comparison_mode=_record_string(comparison, "mode"),
        varied_setting=_record_optional_string(comparison, "varied_setting"),
        sample_selection_digest=_record_string(
            workload,
            "sample_selection_digest",
        ),
        dataset_sample_digests=tuple(
            (
                _record_string_value(dataset, "dataset"),
                _record_string_value(digest, "digest"),
            )
            for dataset, digest in dataset_digests.items()
        ),
        datasets=tuple(
            _record_string_value(dataset, "datasets entry")
            for dataset in _record_sequence(workload, "datasets")
        ),
        example_count=_record_int(workload, "example_count"),
        complete_dataset_split=_record_bool(workload, "complete_dataset_split"),
        measurement_scopes=tuple(
            _record_string_value(scope, "measurement_scopes entry")
            for scope in _record_sequence(workload, "measurement_scopes")
        ),
        prompt_template_version=_record_string(
            workload,
            "prompt_template_version",
        ),
        scorer_identities=tuple(scorer_identities),
        input_tokens_target=_record_optional_int(workload, "input_tokens_target"),
        output_tokens_target=_record_optional_int(workload, "output_tokens_target"),
        temperature=_record_optional_number(decoding, "temperature"),
        stream=_record_optional_bool(decoding, "stream"),
        generation_seed=_record_optional_int(decoding, "generation_seed"),
        decode_settings=_deep_freeze_json_mapping(
            _json_object_mapping(
                decoding.get("settings", {}),
                "decoding.settings",
            )
        ),
        decoding_config_digest=_record_string(decoding, "config_digest"),
        model_id=_record_string(model_runtime, "model_id"),
        model_revision=_record_string(model_runtime, "model_revision"),
        tokenizer_id=_record_string(model_runtime, "tokenizer_id"),
        tokenizer_revision=_record_string(model_runtime, "tokenizer_revision"),
        engine_id=_record_string(model_runtime, "engine_id"),
        engine_version=_record_string(model_runtime, "engine_version"),
        package_revisions=tuple(
            (
                _record_string_value(name, "package name"),
                _record_string_value(revision, "package revision"),
            )
            for name, revision in package_revisions.items()
        ),
        hardware_target=_record_string(environment, "hardware_target"),
        hardware_fingerprint=_record_string(environment, "hardware_fingerprint"),
        runtime_id=_record_string(environment, "runtime_id"),
        runtime_version=_record_string(environment, "runtime_version"),
        storage_identity=_record_string(environment, "storage_identity"),
        cache_state=_record_string(environment, "cache_state"),
        request_parallelism=_record_int(execution, "request_parallelism"),
        repeats=_record_int(execution, "repeats"),
        warmups=_record_int(execution, "warmups"),
        isolate_arms=_record_bool(execution, "isolate_arms"),
        order_mode=_record_string(execution, "order_mode"),
        shuffle=_record_bool(execution, "shuffle"),
        benchmark_seed=_record_optional_int(execution, "benchmark_seed"),
        arms=tuple(arm_manifests),
        execution_isolation_mode=_record_string(
            execution,
            "isolation_mode",
        ),  # type: ignore[arg-type]
        source_execution_ids=tuple(
            (
                _record_string(
                    _record_mapping_value(item, "source_execution_ids entry"),
                    "arm_id",
                ),
                _record_string(
                    _record_mapping_value(item, "source_execution_ids entry"),
                    "execution_id_digest",
                ),
            )
            for item in (
                _record_sequence(execution, "source_execution_ids")
                if "source_execution_ids" in execution
                else ()
            )
        ),
        resource_evidence_ids=tuple(
            (
                _record_string(
                    _record_mapping_value(item, "resource_evidence_ids entry"),
                    "arm_id",
                ),
                _record_string(
                    _record_mapping_value(item, "resource_evidence_ids entry"),
                    "resource_evidence_sha256",
                ),
            )
            for item in (
                _record_sequence(execution, "resource_evidence_ids")
                if "resource_evidence_ids" in execution
                else ()
            )
        ),
    )
    _validate_arm_runtime_environments(
        manifest.arms,
        comparison_mode=manifest.comparison_mode,
        varied_setting=manifest.varied_setting,
        reference_arm_id=manifest.baseline_arm_id,
    )
    return manifest


def benchmark_run_result_from_record(
    record: Mapping[str, Any],
    *,
    evidence_policy: Literal["smoke", "canary", "publication"] | None = None,
) -> BenchmarkRunResult:
    """Reconstruct typed benchmark state and recompute all derived aggregates."""

    if record.get("record_type") != BENCHMARK_RUN_RECORD_TYPE:
        raise ValueError("unsupported benchmark run record_type")
    manifest = benchmark_experiment_manifest_from_record(
        _record_mapping(record, "experiment_manifest")
    )
    suite_record = _record_mapping(record, "suite")
    descriptions = {
        _record_string(_record_mapping_value(raw, "suite arm"), "arm_id"): (
            _record_optional_string(
                _record_mapping_value(raw, "suite arm"), "description"
            )
            or "Recorded benchmark arm."
        )
        for raw in _record_sequence(suite_record, "arms")
    }
    arms = tuple(
        BenchmarkArm(
            arm_id=arm.arm_id,
            uses_cache=arm.uses_cache,
            description=descriptions.get(arm.arm_id, "Recorded benchmark arm."),
            cache_method=arm.method_id,
            connector_mode=arm.connector_mode,
            variant_id=arm.variant_id,
            implementation_kind=arm.implementation_kind,
            method_version=arm.method_version,
            method_config_digest=arm.method_config_digest,
            physical_transform_id=arm.physical_transform_id,
            physical_transform_version=arm.physical_transform_version,
            physical_transform_config_digest=(
                arm.declared_physical_transform_config_digest
            ),
            scorer_plugin_path=arm.scorer_plugin_path,
            offline_costs=BenchmarkOfflineCosts(
                training_seconds=arm.offline_training_seconds,
                artifact_generation_seconds=arm.offline_artifact_generation_seconds,
                checkpoint_load_seconds=arm.offline_checkpoint_load_seconds,
                artifact_bytes=arm.artifact_bytes,
                peak_memory_bytes=arm.offline_peak_memory_bytes,
            ),
            source_revision=arm.source_revision,
            checkpoint_identity=arm.checkpoint_identity,
            setting_overrides=arm.setting_overrides,
            requires_cachet_handoff=arm.requires_cachet_handoff,
        )
        for arm in manifest.arms
    )
    _validate_comparison_design(
        arms,
        comparison_mode=manifest.comparison_mode,
        varied_setting=manifest.varied_setting,
        reference_arm_id=manifest.baseline_arm_id,
    )
    measurements = tuple(
        _measurement_from_record(_record_mapping_value(raw, f"measurements[{index}]"))
        for index, raw in enumerate(_record_sequence(record, "measurements"))
    )
    arms_by_id = {arm.arm_id: arm for arm in manifest.arms}
    for measurement in measurements:
        arm = arms_by_id.get(measurement.arm_id)
        if arm is None:
            raise ValueError(
                f"measurement references unknown manifest arm {measurement.arm_id!r}"
            )
        expected_method = arm.method_id if arm.uses_cache else ""
        if measurement.cache_method != expected_method:
            raise ValueError(
                f"measurement arm {measurement.arm_id!r} cache_method "
                f"{measurement.cache_method!r} does not match manifest method "
                f"{expected_method!r}"
            )
        if measurement.variant_id != arm.variant_id:
            raise ValueError(
                f"measurement arm {measurement.arm_id!r} variant_id "
                f"{measurement.variant_id!r} does not match manifest variant "
                f"{arm.variant_id!r}"
            )
    examples: list[BenchmarkExample] = []
    by_example: dict[tuple[str, str], InferenceMeasurement] = {}
    for measurement in measurements:
        by_example.setdefault(
            (measurement.dataset, measurement.example_id),
            measurement,
        )
    for (dataset, example_id), measurement in sorted(by_example.items()):
        examples.append(
            BenchmarkExample(
                example_id=example_id,
                dataset=dataset,
                documents=(
                    SourceDocument.from_text(
                        document_id=f"record-{dataset}-{example_id}",
                        text="Recorded benchmark logical input.",
                    ),
                ),
                query="Recorded benchmark query.",
                expected_answer=measurement.expected_answer,
                references=measurement.references,
            )
        )
    if len(examples) != manifest.example_count:
        raise ValueError("benchmark measurements do not match manifest example_count")
    suite = BenchmarkSuite(
        suite_id=_record_string(suite_record, "suite_id"),
        examples=tuple(examples),
        model_id=_record_string(suite_record, "model_id"),
        hardware_target=_record_string(suite_record, "hardware_target"),
        datasets=tuple(
            _record_string_value(dataset, "suite datasets entry")
            for dataset in _record_sequence(suite_record, "datasets")
        ),
    )
    execution_windows = tuple(
        _execution_window_from_record(
            _record_mapping_value(raw, f"execution_windows[{index}]")
        )
        for index, raw in enumerate(record.get("execution_windows", ()))
    )
    resource_evidence = tuple(
        benchmark_resource_evidence_from_record(
            _record_mapping_value(raw, f"resource_evidence[{index}]")
        )
        for index, raw in enumerate(record.get("resource_evidence", ()))
    )
    _validate_execution_windows_against_measurements(
        execution_windows,
        measurements,
        arms=arms,
        isolation_mode=manifest.execution_isolation_mode,
    )
    aggregate_by_arm = {
        window.arm_id: window.aggregate_output_tokens_per_second
        for window in execution_windows
        if window.arm_id != "all_arms"
    }
    report_rows = tuple(
        replace(
            row,
            aggregate_output_tokens_per_second=aggregate_by_arm.get(row.arm_id),
        )
        for row in summarize_measurements(measurements)
    )
    comparison_arm_ids = tuple(
        arm.arm_id for arm in arms if arm.arm_id != manifest.baseline_arm_id
    )
    comparisons = tuple(
        comparison
        for arm_id in comparison_arm_ids
        for comparison in compare_to_baseline(
            report_rows,
            baseline_arm_id=manifest.baseline_arm_id,
            cache_arm_id=arm_id,
        )
    )
    resolved_policy = evidence_policy
    if resolved_policy is None:
        gate = record.get("evidence_gate")
        policy_value = gate.get("policy") if isinstance(gate, Mapping) else "smoke"
        if policy_value not in {"smoke", "canary", "publication"}:
            raise ValueError("benchmark evidence policy is invalid")
        resolved_policy = cast(
            Literal["smoke", "canary", "publication"],
            policy_value,
        )
    result = BenchmarkRunResult(
        suite=suite,
        measurements=measurements,
        report_rows=report_rows,
        comparisons=comparisons,
        baseline_arm_id=manifest.baseline_arm_id,
        cache_arm_id=(comparison_arm_ids[0] if comparison_arm_ids else CACHE_REUSE_ARM),
        request_parallelism=manifest.request_parallelism,
        isolate_arms=manifest.isolate_arms,
        arms=arms,
        repeats=manifest.repeats,
        shuffle=manifest.shuffle,
        seed=manifest.benchmark_seed,
        interleave_examples=manifest.order_mode == "example_interleaved",
        prefix_cache_salt_mode=_record_optional_string(
            suite_record,
            "prefix_cache_salt_mode",
        )
        or "static",
        warmups=manifest.warmups,
        experiment_manifest=manifest,
        evidence_policy=resolved_policy,
        execution_windows=execution_windows,
        execution_isolation_mode=manifest.execution_isolation_mode,
        resource_evidence=resource_evidence,
    )
    _validate_resource_evidence_bindings(result)
    return result


def benchmark_record_aggregate_issues(
    record: Mapping[str, Any],
) -> tuple[str, ...]:
    """Recompute canonical aggregates and report any record-level tampering."""

    try:
        result = benchmark_run_result_from_record(record)
    except (TypeError, ValueError) as exc:
        return (f"benchmark record cannot be reconstructed: {exc}",)
    issues: list[str] = []
    expected_rows = [_report_row_to_record(row) for row in result.report_rows]
    if record.get("report_rows") != expected_rows:
        issues.append("report_rows do not match raw benchmark measurements")
    expected_comparisons = [
        _comparison_to_record(comparison) for comparison in result.comparisons
    ]
    if record.get("comparisons") != expected_comparisons:
        issues.append("comparisons do not match recomputed report rows")
    from document_kv_cache.benchmark_statistics import (
        paired_benchmark_statistics,
        paired_benchmark_statistics_to_record,
    )

    expected_paired = paired_benchmark_statistics_to_record(
        paired_benchmark_statistics(result)
    )
    if record.get("paired_statistics") != expected_paired:
        issues.append("paired_statistics do not match raw benchmark measurements")
    return tuple(issues)


def _merged_environment_summary(
    raw_arms: Sequence[Any],
    field_name: str,
) -> Any:
    values = [
        _record_mapping(
            _record_mapping_value(raw_arm, "manifest arm"),
            "runtime_environment",
        ).get(field_name)
        for raw_arm in raw_arms
    ]
    if all(value == values[0] for value in values[1:]):
        return values[0]
    return VARIES_BY_ARM


def merge_isolated_benchmark_run_records(
    records: Sequence[Mapping[str, Any]],
    *,
    reference_arm_id: str,
    comparison_mode: Literal[
        "methods_same_setting",
        "single_method_setting_variation",
    ] = "methods_same_setting",
    varied_setting: str = "",
    policy: Literal["smoke", "canary", "publication"] = "canary",
    cache_state_attestations: Iterable[Any] = (),
    artifact_identities: Mapping[str, Any] | None = None,
    method_registry: Any = None,
) -> dict[str, Any]:
    """Merge independently valid one-arm jobs into one freshly gated comparison.

    A one-arm source necessarily names itself as its local reference.  The union-level
    comparison is therefore supplied explicitly here and is validated only after all
    physical arms have been collected.
    """

    source_records = tuple(records)
    if len(source_records) < 2:
        raise ValueError("isolated merge requires at least two benchmark records")
    _validate_non_empty_string(reference_arm_id, "reference_arm_id")
    payloads = [dict(record) for record in source_records]
    manifests = [_record_mapping(record, "experiment_manifest") for record in payloads]
    suites = [_record_mapping(record, "suite") for record in payloads]
    first_manifest = manifests[0]
    first_suite = suites[0]
    for index, (record, manifest, suite) in enumerate(
        zip(payloads, manifests, suites, strict=True)
    ):
        aggregate_issues = benchmark_record_aggregate_issues(record)
        if aggregate_issues:
            raise ValueError(
                f"isolated record {index} is not canonical: "
                + "; ".join(aggregate_issues)
            )
        manifest_arms = _record_sequence(manifest, "arms")
        suite_arms = _record_sequence(suite, "arms")
        if len(manifest_arms) != 1 or len(suite_arms) != 1:
            raise ValueError(
                f"isolated record {index} must contain exactly one physical arm"
            )
        manifest_arm_id = _record_string(
            _record_mapping_value(manifest_arms[0], "manifest arm"),
            "arm_id",
        )
        suite_arm_id = _record_string(
            _record_mapping_value(suite_arms[0], "suite arm"),
            "arm_id",
        )
        if manifest_arm_id != suite_arm_id:
            raise ValueError(
                f"isolated record {index} changes arm identity between suite and manifest"
            )
        measurement_arm_ids = {
            _record_string(
                _record_mapping_value(measurement, "measurement"),
                "arm_id",
            )
            for measurement in _record_sequence(record, "measurements")
        }
        if measurement_arm_ids != {manifest_arm_id}:
            raise ValueError(
                f"isolated record {index} contains measurements from another arm"
            )
    for index, (manifest, suite) in enumerate(
        zip(manifests[1:], suites[1:], strict=True),
        start=1,
    ):
        for field_name in ("logical_workload", "decoding"):
            if manifest.get(field_name) != first_manifest.get(field_name):
                raise ValueError(
                    f"isolated record {index} changes shared manifest {field_name}"
                )
        for field_name in (
            "suite_id",
            "datasets",
            "examples",
            "request_parallelism",
            "isolate_arms",
            "repeats",
            "shuffle",
            "seed",
            "interleave_examples",
            "prefix_cache_salt_mode",
        ):
            if suite.get(field_name) != first_suite.get(field_name):
                raise ValueError(
                    f"isolated record {index} changes suite field {field_name}"
                )
        model_runtime = _record_mapping(manifest, "model_runtime")
        first_model_runtime = _record_mapping(first_manifest, "model_runtime")
        if model_runtime.get("package_revisions") != first_model_runtime.get(
            "package_revisions"
        ):
            raise ValueError(
                f"isolated record {index} changes shared package revisions"
            )
        execution = _record_mapping(manifest, "execution")
        first_execution = _record_mapping(first_manifest, "execution")
        for field_name in (
            "request_parallelism",
            "repeats",
            "warmups",
            "shuffle",
            "benchmark_seed",
        ):
            if execution.get(field_name) != first_execution.get(field_name):
                raise ValueError(
                    f"isolated record {index} changes execution field {field_name}"
                )
    typed_manifests = tuple(
        benchmark_experiment_manifest_from_record(manifest) for manifest in manifests
    )
    merged_arms: list[Any] = []
    merged_suite_arms: list[Any] = []
    merged_measurements: list[Any] = []
    merged_windows: list[Any] = []
    merged_resource_evidence: list[Any] = []
    for record, manifest, suite, typed_manifest in zip(
        payloads,
        manifests,
        suites,
        typed_manifests,
        strict=True,
    ):
        raw_arm = dict(
            _record_mapping_value(_record_sequence(manifest, "arms")[0], "manifest arm")
        )
        raw_arm.setdefault(
            "runtime_environment",
            _runtime_environment_to_record(
                typed_manifest.arms[0].runtime_environment
            ),
        )
        merged_arms.append(raw_arm)
        merged_suite_arms.extend(_record_sequence(suite, "arms"))
        merged_measurements.extend(_record_sequence(record, "measurements"))
        merged_windows.extend(record.get("execution_windows", ()))
        merged_resource_evidence.extend(record.get("resource_evidence", ()))
    arm_ids = [
        _record_string(_record_mapping_value(arm, "manifest arm"), "arm_id")
        for arm in merged_arms
    ]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("isolated records contain duplicate arm ids")
    if reference_arm_id not in arm_ids:
        raise ValueError(
            f"reference_arm_id references unknown merged arm {reference_arm_id!r}"
        )
    merged_arm_manifests = tuple(manifest.arms[0] for manifest in typed_manifests)
    _validate_arm_runtime_environments(
        merged_arm_manifests,
        comparison_mode=comparison_mode,
        varied_setting=varied_setting,
        reference_arm_id=reference_arm_id,
    )
    source_runtime_ids = [
        _record_string(_record_mapping(manifest, "environment"), "runtime_id")
        for manifest in manifests
    ]
    if any(runtime_id == "unresolved" for runtime_id in source_runtime_ids):
        raise ValueError(
            "isolated records require a resolved execution-instance runtime_id"
        )
    if len(set(source_runtime_ids)) != len(source_runtime_ids):
        raise ValueError(
            "isolated records require distinct execution-instance runtime_id values"
        )
    source_execution_ids = [
        {
            "arm_id": arm_id,
            "execution_id_digest": sha256(runtime_id.encode("utf-8")).hexdigest(),
        }
        for arm_id, runtime_id in zip(arm_ids, source_runtime_ids, strict=True)
    ]
    resource_evidence_ids = [
        {
            "arm_id": _record_string(
                _record_mapping_value(item, "resource evidence"),
                "arm_id",
            ),
            "resource_evidence_sha256": _record_string(
                _record_mapping_value(item, "resource evidence"),
                "record_sha256",
            ),
        }
        for item in merged_resource_evidence
    ]
    merged_manifest = json.loads(json.dumps(first_manifest))
    merged_manifest["arms"] = merged_arms
    merged_manifest["comparison"] = {
        "mode": comparison_mode,
        "varied_setting": varied_setting or None,
        "baseline_arm_id": reference_arm_id,
        "reference_arm_id": reference_arm_id,
    }
    merged_manifest["has_unresolved_provenance"] = any(
        manifest.get("has_unresolved_provenance") is True for manifest in manifests
    )
    merged_manifest["environment"] = {
        **_record_mapping(merged_manifest, "environment"),
        "hardware_target": _merged_environment_summary(
            merged_arms,
            "hardware_target",
        ),
        "hardware_fingerprint": _merged_environment_summary(
            merged_arms,
            "hardware_fingerprint",
        ),
        "runtime_id": f"separate_jobs:{_sha256_json(sorted(source_runtime_ids))}",
        "runtime_version": _merged_environment_summary(
            merged_arms,
            "runtime_version",
        ),
        "storage_identity": _merged_environment_summary(
            merged_arms,
            "storage_identity",
        ),
        "cache_state": _merged_environment_summary(
            merged_arms,
            "cache_state",
        ),
    }
    model_runtime = _record_mapping(merged_manifest, "model_runtime")
    merged_manifest["model_runtime"] = {
        **model_runtime,
        **{
            field_name: _merged_environment_summary(merged_arms, environment_field)
            for field_name, environment_field in (
                ("model_id", "served_model_id"),
                ("canonical_model_id", "canonical_model_id"),
                ("model_revision", "model_revision"),
                ("tokenizer_id", "tokenizer_id"),
                ("tokenizer_revision", "tokenizer_revision"),
                ("lora_id", "lora_id"),
                ("prompt_template_version", "prompt_template_version"),
                ("engine_id", "engine_id"),
                ("engine_version", "engine_version"),
                ("serving_platform", "serving_platform"),
                ("model_dtype", "model_dtype"),
                ("model_quantization", "model_quantization"),
                ("runtime_kv_dtype", "runtime_kv_dtype"),
                ("layout_version", "layout_version"),
                ("payload_axis_order", "payload_axis_order"),
                ("block_size", "block_size"),
                ("key_position_encoding", "key_position_encoding"),
                ("tensor_parallel_size", "tensor_parallel_size"),
                ("pipeline_parallel_size", "pipeline_parallel_size"),
            )
        },
    }
    execution = _record_mapping(merged_manifest, "execution")
    merged_manifest["execution"] = {
        **execution,
        "isolate_arms": True,
        "order_mode": "physically_isolated_jobs",
        "isolation_mode": "separate_process_or_job",
        "source_execution_ids": source_execution_ids,
        "resource_evidence_ids": resource_evidence_ids,
    }
    merged_suite = json.loads(json.dumps(first_suite))
    merged_suite["arms"] = merged_suite_arms
    merged_suite["model_id"] = _merged_environment_summary(
        merged_arms,
        "served_model_id",
    )
    merged_suite["hardware_target"] = _merged_environment_summary(
        merged_arms,
        "hardware_target",
    )
    seed_record: dict[str, Any] = {
        "record_type": BENCHMARK_RUN_RECORD_TYPE,
        "suite": merged_suite,
        "experiment_manifest": merged_manifest,
        "measurements": merged_measurements,
        "execution_windows": merged_windows,
        "resource_evidence": merged_resource_evidence,
        "report_rows": [],
        "comparisons": [],
        "paired_statistics": {},
        "evidence_gate": {"policy": policy},
    }
    result = benchmark_run_result_from_record(seed_record, evidence_policy=policy)
    return benchmark_run_result_to_evidence_record(
        result,
        cache_state_attestations=cache_state_attestations,
        artifact_identities=artifact_identities,
        method_registry=method_registry,
    )


def _measurement_from_record(record: Mapping[str, Any]) -> InferenceMeasurement:
    measurement = InferenceMeasurement(
        example_id=_record_string(record, "example_id"),
        dataset=_record_string(record, "dataset"),
        arm_id=_record_string(record, "arm_id"),
        prompt_tokens=_record_int(record, "prompt_tokens"),
        completion_tokens=_record_int(record, "completion_tokens"),
        ttft_seconds=_record_number(record, "ttft_seconds"),
        time_to_completion_seconds=_record_number(
            record,
            "time_to_completion_seconds",
        ),
        output_text=_record_string_allow_empty(record, "output_text"),
        expected_answer=_record_nullable_string(record, "expected_answer"),
        error=_record_nullable_string(record, "error"),
        metadata=_record_string_mapping(record.get("metadata", {}), "metadata"),
        cache_method=_record_optional_string(record, "cache_method"),
        artifact_id=_record_optional_string(record, "artifact_id"),
        variant_id=_record_optional_string(record, "variant_id"),
        request_id=_record_optional_string(record, "request_id"),
        repeat_index=_record_int(record, "repeat_index"),
        scorer_id=_record_optional_string(record, "scorer_id"),
        scorer_version=_record_optional_string(record, "scorer_version"),
        quality_scores=_record_number_mapping(
            record.get("quality_scores", {}),
            "quality_scores",
        ),
        references=tuple(
            _record_string_value(reference, "references entry")
            for reference in record.get("references", ())
        ),
    )
    for field_name, reconstructed in (
        ("exact_match", measurement.exact_match),
        ("answer_found", measurement.answer_found),
    ):
        if field_name in record:
            recorded = _record_optional_bool(record, field_name)
            if recorded is not reconstructed:
                raise ValueError(
                    f"measurement {field_name} contradicts quality_scores/output"
                )
    return measurement


def _validate_execution_windows_against_measurements(
    windows: Sequence[BenchmarkExecutionWindow],
    measurements: Sequence[InferenceMeasurement],
    *,
    arms: Sequence[BenchmarkArm],
    isolation_mode: str,
) -> None:
    arm_ids = {arm.arm_id for arm in arms}
    window_ids = [window.arm_id for window in windows]
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("execution windows must not contain duplicate arm ids")
    isolated = isolation_mode in {
        "shared_process_sequential",
        "separate_process_or_job",
    }
    expected_ids = arm_ids if isolated or len(arm_ids) == 1 else {"all_arms"}
    if set(window_ids) != expected_ids:
        raise ValueError(
            "execution window coverage does not match the declared isolation mode"
        )
    for window in windows:
        scoped = tuple(
            measurement
            for measurement in measurements
            if window.arm_id == "all_arms" or measurement.arm_id == window.arm_id
        )
        successful = tuple(measurement for measurement in scoped if measurement.ok)
        if window.successful_requests != len(successful):
            raise ValueError(
                f"execution window {window.arm_id!r} successful_requests does not "
                "match raw measurements"
            )
        completion_tokens = sum(
            measurement.completion_tokens for measurement in successful
        )
        if window.completion_tokens != completion_tokens:
            raise ValueError(
                f"execution window {window.arm_id!r} completion_tokens does not "
                "match raw measurements"
            )


def _execution_window_from_record(
    record: Mapping[str, Any],
) -> BenchmarkExecutionWindow:
    window = BenchmarkExecutionWindow(
        arm_id=_record_string(record, "arm_id"),
        wall_seconds=_record_number(record, "wall_seconds"),
        completion_tokens=_record_int(record, "completion_tokens"),
        successful_requests=_record_int(record, "successful_requests"),
        started_at_seconds=_record_optional_number(record, "started_at_seconds"),
        ended_at_seconds=_record_optional_number(record, "ended_at_seconds"),
    )
    recorded_tps = _record_number(record, "aggregate_output_tokens_per_second")
    if not math.isclose(
        recorded_tps,
        window.aggregate_output_tokens_per_second,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("execution window aggregate throughput is inconsistent")
    return window


def _require_record_keys(
    record: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} must be an object")
    unknown = sorted(set(record).difference(allowed))
    if unknown:
        raise ValueError(f"{label} has unknown fields: {unknown}")


def _record_mapping(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _record_mapping_value(record.get(key), key)


def _record_mapping_value(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _record_sequence(record: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = record.get(key)
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise ValueError(f"{key} must be an array")
    return value


def _record_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    return _record_string_value(value, key)


def _record_string_allow_empty(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _record_string_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _record_optional_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _record_nullable_string(
    record: Mapping[str, Any],
    key: str,
) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string or null")
    return value


def _record_bool(record: Mapping[str, Any], key: str) -> bool:
    value = record.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _record_optional_bool(record: Mapping[str, Any], key: str) -> bool | None:
    value = record.get(key)
    if value is None:
        return None
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean or null")
    return value


def _record_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _record_optional_int(record: Mapping[str, Any], key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer or null")
    return value


def _record_number(record: Mapping[str, Any], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{key} must be finite")
    return float(value)


def _record_optional_number(record: Mapping[str, Any], key: str) -> float | None:
    if record.get(key) is None:
        return None
    return _record_number(record, key)


def _record_string_mapping(value: Any, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[_record_string_value(key, f"{label} key")] = _record_string_value(
            item,
            f"{label}.{key}",
        )
    return result


def _record_number_mapping(value: Any, label: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return {
        _record_string_value(key, f"{label} key"): _record_number(
            {"value": item},
            "value",
        )
        for key, item in value.items()
    }


def write_benchmark_run_result_json(
    result: BenchmarkRunResult,
    path: str | Path,
    *,
    cache_state_attestations: Iterable[Any] = (),
    artifact_identities: Mapping[str, Any] | None = None,
) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            benchmark_run_result_to_record(
                result,
                cache_state_attestations=cache_state_attestations,
                artifact_identities=artifact_identities,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _measurement_to_record(
    measurement: InferenceMeasurement,
    *,
    sanitize_evidence: bool = False,
) -> dict[str, Any]:
    quality_scores = dict(measurement.quality_scores)
    return {
        "example_id": measurement.example_id,
        "dataset": measurement.dataset,
        "arm_id": measurement.arm_id,
        "prompt_tokens": measurement.prompt_tokens,
        "completion_tokens": measurement.completion_tokens,
        "ttft_seconds": measurement.ttft_seconds,
        "time_to_completion_seconds": measurement.time_to_completion_seconds,
        "output_text": "" if sanitize_evidence else measurement.output_text,
        "expected_answer": (None if sanitize_evidence else measurement.expected_answer),
        "references": [] if sanitize_evidence else list(measurement.references),
        "exact_match": measurement.exact_match,
        "answer_found": measurement.answer_found,
        "error": (
            "redacted" if sanitize_evidence and measurement.error else measurement.error
        ),
        "metadata": (
            _sanitized_measurement_metadata(measurement.metadata)
            if sanitize_evidence
            else dict(measurement.metadata)
        ),
        "cache_method": measurement.cache_method,
        "artifact_id": measurement.artifact_id,
        "variant_id": measurement.variant_id,
        "request_id": (
            _sanitized_identifier(measurement.request_id)
            if sanitize_evidence and measurement.request_id
            else measurement.request_id
        ),
        "repeat_index": measurement.repeat_index,
        "scorer_id": measurement.scorer_id,
        "scorer_version": measurement.scorer_version,
        "quality_scores": quality_scores,
    }


def _sanitized_measurement_metadata(
    metadata: Mapping[str, str],
) -> dict[str, str]:
    """Return the closed, non-content-bearing measurement metadata projection."""

    sanitized: dict[str, str] = {}
    for key in _SANITIZED_DIGEST_METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        if not _is_sha256_digest(value):
            raise ValueError(f"measurement metadata {key} must be a SHA-256 digest")
        sanitized[key] = value
    cache_salt = metadata.get("prefix_cache_salt")
    if cache_salt is not None:
        if not isinstance(cache_salt, str) or not cache_salt:
            raise ValueError("measurement metadata prefix_cache_salt must be non-empty")
        cache_salt_digest = sha256(
            cache_salt.encode("utf-8")
        ).hexdigest()
        existing_digest = sanitized.get("prefix_cache_salt_sha256")
        if existing_digest is not None and existing_digest != cache_salt_digest:
            raise ValueError(
                "measurement metadata prefix_cache_salt digests do not match"
            )
        sanitized["prefix_cache_salt_sha256"] = cache_salt_digest
    for key in _SANITIZED_INTEGER_METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            normalized_int = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"measurement metadata {key} must be an integer") from exc
        if normalized_int < 0 or str(normalized_int) != value.strip():
            raise ValueError(
                f"measurement metadata {key} must be a canonical non-negative integer"
            )
        sanitized[key] = str(normalized_int)
    for key in _SANITIZED_FLOAT_METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            normalized_float = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"measurement metadata {key} must be numeric") from exc
        if not math.isfinite(normalized_float) or normalized_float < 0:
            raise ValueError(
                f"measurement metadata {key} must be finite and non-negative"
            )
        sanitized[key] = format(normalized_float, ".17g")
    for key in _SANITIZED_BOOLEAN_METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        if value not in {"true", "false"}:
            raise ValueError(f"measurement metadata {key} must be true or false")
        sanitized[key] = value
    for key, allowed in _SANITIZED_ENUM_METADATA_VALUES.items():
        value = metadata.get(key)
        if value is None:
            continue
        if value not in allowed:
            raise ValueError(
                f"measurement metadata {key} must be one of {sorted(allowed)!r}"
            )
        sanitized[key] = value
    return sanitized


def _cache_state_attestation_to_record(
    attestation: Any,
    *,
    sanitize_evidence: bool,
) -> dict[str, Any]:
    record = attestation.to_record()
    if not isinstance(record, Mapping):
        raise TypeError("cache-state attestation to_record() must return a mapping")
    materialized = dict(record)
    request_id = materialized.get("request_id")
    if sanitize_evidence:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("cache-state attestation request_id must be non-empty")
        materialized["request_id"] = _sanitized_identifier(request_id)
    return materialized


def _sanitized_gate_result(result: BenchmarkRunResult) -> BenchmarkRunResult:
    measurements = tuple(
        replace(
            measurement,
            output_text="",
            expected_answer=None,
            references=(),
            error=("redacted" if measurement.error else None),
            request_id=(
                _sanitized_identifier(measurement.request_id)
                if measurement.request_id
                else ""
            ),
            metadata=_sanitized_measurement_metadata(measurement.metadata),
        )
        for measurement in result.measurements
    )
    aggregate_by_arm = {
        row.arm_id: row.aggregate_output_tokens_per_second
        for row in result.report_rows
    }
    report_rows = tuple(
        replace(
            row,
            aggregate_output_tokens_per_second=aggregate_by_arm.get(row.arm_id),
        )
        for row in summarize_measurements(measurements)
    )
    comparisons = tuple(
        comparison
        for cache_arm_id in result.cache_arm_ids
        for comparison in compare_to_baseline(
            report_rows,
            baseline_arm_id=result.baseline_arm_id,
            cache_arm_id=cache_arm_id,
        )
    )
    return replace(
        result,
        measurements=measurements,
        report_rows=report_rows,
        comparisons=comparisons,
    )


def _sanitized_cache_state_attestation(attestation: Any) -> Any:
    request_id = getattr(attestation, "request_id", None)
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("cache-state attestation request_id must be non-empty")
    return replace(attestation, request_id=_sanitized_identifier(request_id))


def _sanitized_identifier(value: str) -> str:
    if _is_sha256_digest(value):
        return value
    return sha256(value.encode("utf-8")).hexdigest()


def sanitized_measurement_metadata_issues(value: object) -> tuple[str, ...]:
    """Validate metadata already marked as safe committed evidence."""

    if not isinstance(value, Mapping):
        return ("measurement metadata must be an object",)
    issues: list[str] = []
    unexpected = sorted(str(key) for key in value if key not in SANITIZED_MEASUREMENT_METADATA_KEYS)
    if unexpected:
        issues.append(f"measurement metadata contains non-sanitized keys: {unexpected}")
    for key, item in value.items():
        if key not in SANITIZED_MEASUREMENT_METADATA_KEYS:
            continue
        if not isinstance(item, str):
            issues.append(f"measurement metadata {key} must be a string")
            continue
        if key in _SANITIZED_DIGEST_METADATA_KEYS and not _is_sha256_digest(item):
            issues.append(f"measurement metadata {key} must be a SHA-256 digest")
        elif key in _SANITIZED_INTEGER_METADATA_KEYS:
            try:
                normalized = int(item)
            except ValueError:
                normalized = -1
            if normalized < 0 or str(normalized) != item:
                issues.append(
                    f"measurement metadata {key} must be a canonical non-negative integer"
                )
        elif key in _SANITIZED_FLOAT_METADATA_KEYS:
            try:
                normalized_float = float(item)
            except ValueError:
                normalized_float = math.nan
            if not math.isfinite(normalized_float) or normalized_float < 0:
                issues.append(
                    f"measurement metadata {key} must be finite and non-negative"
                )
        elif key in _SANITIZED_BOOLEAN_METADATA_KEYS and item not in {"true", "false"}:
            issues.append(f"measurement metadata {key} must be true or false")
        elif key in _SANITIZED_ENUM_METADATA_VALUES and item not in _SANITIZED_ENUM_METADATA_VALUES[key]:
            issues.append(
                f"measurement metadata {key} has an unsupported value"
            )
    return tuple(issues)


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        "pooled_request_output_tokens_per_second": row.output_tokens_per_second,
        "aggregate_output_tokens_per_second": (row.aggregate_output_tokens_per_second),
        "request_decode_tokens_per_second": _latency_to_record(
            row.request_decode_tokens_per_second
        ),
        "cache_method": row.cache_method,
        "artifact_id": row.artifact_id,
        "variant_id": row.variant_id,
        "unique_examples": row.unique_examples,
        "quality_score_means": dict(row.quality_score_means),
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
        "cache_method": comparison.cache_method,
        "artifact_id": comparison.artifact_id,
        "variant_id": comparison.variant_id,
        "quality_score_deltas": dict(comparison.quality_score_deltas),
    }


def _rows_for_arm_pair(
    rows: Sequence[BenchmarkReportRow],
    *,
    baseline_arm_id: str,
    cache_arm_id: str,
) -> tuple[BenchmarkReportRow, ...]:
    arm_ids = {baseline_arm_id, cache_arm_id}
    return tuple(row for row in rows if row.arm_id in arm_ids)


def _comparisons_for_arm_pair(
    comparisons: Sequence[BenchmarkComparison],
    *,
    baseline_arm_id: str,
    cache_arm_id: str,
) -> tuple[BenchmarkComparison, ...]:
    return tuple(
        comparison
        for comparison in comparisons
        if comparison.baseline_arm_id == baseline_arm_id
        and comparison.cache_arm_id == cache_arm_id
    )


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
        "rows_without_successful_requests": list(
            evidence.rows_without_successful_requests
        ),
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


for _public_function in (
    benchmark_run_result_payload_to_record,
    benchmark_record_payload_digest,
    benchmark_run_result_to_record,
    benchmark_run_result_to_evidence_record,
    benchmark_resource_evidence_to_record,
    benchmark_resource_evidence_from_record,
    benchmark_experiment_manifest_to_record,
    benchmark_experiment_manifest_from_record,
    benchmark_run_result_from_record,
    benchmark_record_aggregate_issues,
    merge_isolated_benchmark_run_records,
    write_benchmark_run_result_json,
):
    _public_function.__module__ = "document_kv_cache.benchmark_runner"
