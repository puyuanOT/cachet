from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from restaurant_kv_serving.benchmarks import (
    CACHE_REUSE_ARM,
    DEFAULT_HARDWARE_TARGET,
    DEFAULT_V1_MODEL_ID,
    SUPPORTED_V1_DATASETS,
    BASELINE_PREFILL_ARM,
)
from restaurant_kv_serving.benchmark_runner import BENCHMARK_RUN_RECORD_TYPE
from restaurant_kv_serving.engine_adapters import ServingBackend, validate_engine_kv_connector_probe_record
from restaurant_kv_serving.engine_protocol import (
    AttentionMechanism,
    KVLayout,
    KVStorageLayout,
    dtype_byte_width,
    kv_storage_layout_from_value,
)
from restaurant_kv_serving.engine_probe import (
    ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE,
    ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION,
)
from restaurant_kv_serving.model_profiles import get_model_profile
from restaurant_kv_serving.serving_env import serving_environment_profile
from restaurant_kv_serving.storage import is_real_uc_volume_root, local_path
from restaurant_kv_serving.storage_benchmark import RELEASE_STORAGE_BENCHMARK_READERS, STORAGE_BENCHMARK_RECORD_TYPE


RELEASE_EVIDENCE_RECORD_TYPE = "document_kv.release_evidence.v1"
RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE = "document_kv.release_evidence_inputs.v1"
REQUIRED_ENGINE_PROBE_BACKENDS = tuple(backend.value for backend in ServingBackend)
RELEASE_EVIDENCE_ARTIFACT_ROLES = ("v1_benchmark", "storage_benchmark", "engine_probe")


@dataclass(frozen=True, slots=True)
class ReleaseEvidenceArtifactSource:
    role: str
    path: str | Path
    record_type: str | None = None
    backend: str | None = None

    def __post_init__(self) -> None:
        if self.role not in RELEASE_EVIDENCE_ARTIFACT_ROLES:
            raise ValueError(f"Unsupported artifact role {self.role!r}")
        if isinstance(self.path, Path):
            object.__setattr__(self, "path", str(self.path))
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("path must be non-empty")
        if self.record_type is not None and (not isinstance(self.record_type, str) or not self.record_type):
            raise ValueError("record_type must be non-empty when provided")
        if self.backend is not None and self.role != "engine_probe":
            raise ValueError("backend can only be set for engine_probe artifact sources")
        if self.backend is not None and self.backend not in REQUIRED_ENGINE_PROBE_BACKENDS:
            raise ValueError(f"Unsupported artifact backend {self.backend!r}")


@dataclass(frozen=True, slots=True)
class ReleaseEvidence:
    v1_benchmark_ok: bool
    storage_benchmark_ok: bool
    engine_probe_backends: tuple[str, ...]
    missing_engine_probe_backends: tuple[str, ...]
    invalid_engine_probe_records: tuple[str, ...]
    issues: tuple[str, ...]
    artifact_sources: tuple[ReleaseEvidenceArtifactSource, ...] = ()
    duplicate_engine_probe_backends: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class ReleaseEvidenceInputFileStatus:
    role: str
    path: str
    exists: bool
    readable_json: bool
    record_type: str | None = None
    backend: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.role not in RELEASE_EVIDENCE_ARTIFACT_ROLES:
            raise ValueError(f"Unsupported artifact role {self.role!r}")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("path must be non-empty")
        if type(self.exists) is not bool:
            raise ValueError("exists must be boolean")
        if type(self.readable_json) is not bool:
            raise ValueError("readable_json must be boolean")
        if self.record_type is not None and (not isinstance(self.record_type, str) or not self.record_type):
            raise ValueError("record_type must be non-empty when provided")
        if self.backend is not None and self.role != "engine_probe":
            raise ValueError("backend can only be set for engine_probe input files")
        if self.backend is not None and self.backend not in REQUIRED_ENGINE_PROBE_BACKENDS:
            raise ValueError(f"Unsupported artifact backend {self.backend!r}")
        if self.error is not None and (not isinstance(self.error, str) or not self.error):
            raise ValueError("error must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class ReleaseEvidenceInputStatus:
    input_files: tuple[ReleaseEvidenceInputFileStatus, ...]
    missing_paths: tuple[str, ...]
    unreadable_paths: tuple[str, ...]
    missing_engine_probe_backends: tuple[str, ...]
    required_engine_probe_backends: tuple[str, ...] = REQUIRED_ENGINE_PROBE_BACKENDS

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def issues(self) -> tuple[str, ...]:
        issues = []
        if self.missing_paths:
            issues.append(f"missing input paths: {', '.join(self.missing_paths)}")
        if self.unreadable_paths:
            issues.append(f"unreadable input paths: {', '.join(self.unreadable_paths)}")
        if self.missing_engine_probe_backends:
            issues.append(f"missing engine probe backends: {', '.join(self.missing_engine_probe_backends)}")
        return tuple(issues)

    def __post_init__(self) -> None:
        if any(not isinstance(item, ReleaseEvidenceInputFileStatus) for item in self.input_files):
            raise TypeError("input_files entries must be ReleaseEvidenceInputFileStatus")
        object.__setattr__(self, "input_files", tuple(self.input_files))
        object.__setattr__(self, "missing_paths", _validated_str_tuple(self.missing_paths, "missing_paths"))
        object.__setattr__(self, "unreadable_paths", _validated_str_tuple(self.unreadable_paths, "unreadable_paths"))
        object.__setattr__(
            self,
            "missing_engine_probe_backends",
            _validated_backend_tuple(self.missing_engine_probe_backends, "missing_engine_probe_backends"),
        )
        object.__setattr__(
            self,
            "required_engine_probe_backends",
            _validated_required_backends(self.required_engine_probe_backends),
        )


def evaluate_release_evidence(
    v1_benchmark_record: Mapping[str, Any],
    storage_benchmark_record: Mapping[str, Any],
    *,
    engine_probe_records: Sequence[Mapping[str, Any]] = (),
    required_engine_probe_backends: Sequence[str] = REQUIRED_ENGINE_PROBE_BACKENDS,
    artifact_sources: Sequence[ReleaseEvidenceArtifactSource] = (),
) -> ReleaseEvidence:
    required_backends = _validated_required_backends(required_engine_probe_backends)
    artifact_source_tuple = _validated_artifact_sources(artifact_sources)
    issues: list[str] = []
    v1_issues = _v1_benchmark_issues(v1_benchmark_record)
    storage_issues = _storage_benchmark_issues(storage_benchmark_record)
    issues.extend(v1_issues)
    issues.extend(storage_issues)

    probe_backends, invalid_probe_records, duplicate_probe_backends = _engine_probe_evidence(engine_probe_records)
    missing_probe_backends = tuple(backend for backend in required_backends if backend not in probe_backends)
    if missing_probe_backends:
        issues.append(f"missing engine probe backends: {', '.join(missing_probe_backends)}")
    if duplicate_probe_backends:
        issues.append(f"duplicate engine probe backends: {', '.join(duplicate_probe_backends)}")
    issues.extend(f"invalid engine probe record: {issue}" for issue in invalid_probe_records)

    return ReleaseEvidence(
        v1_benchmark_ok=not v1_issues,
        storage_benchmark_ok=not storage_issues,
        engine_probe_backends=probe_backends,
        missing_engine_probe_backends=missing_probe_backends,
        duplicate_engine_probe_backends=duplicate_probe_backends,
        invalid_engine_probe_records=invalid_probe_records,
        issues=tuple(issues),
        artifact_sources=artifact_source_tuple,
    )


def evaluate_release_evidence_files(
    *,
    v1_benchmark_json: str | Path,
    storage_benchmark_json: str | Path,
    engine_probe_jsons: Sequence[str | Path] = (),
) -> ReleaseEvidence:
    v1_record = _read_json_record(v1_benchmark_json)
    storage_record = _read_json_record(storage_benchmark_json)
    engine_probe_records = tuple(_read_json_record(path) for path in engine_probe_jsons)
    return evaluate_release_evidence(
        v1_record,
        storage_record,
        engine_probe_records=engine_probe_records,
        artifact_sources=_artifact_sources_for_records(
            v1_benchmark_json=v1_benchmark_json,
            v1_record=v1_record,
            storage_benchmark_json=storage_benchmark_json,
            storage_record=storage_record,
            engine_probe_jsons=engine_probe_jsons,
            engine_probe_records=engine_probe_records,
        ),
    )


def inspect_release_evidence_input_files(
    *,
    v1_benchmark_json: str | Path,
    storage_benchmark_json: str | Path,
    engine_probe_jsons: Sequence[str | Path] = (),
    required_engine_probe_backends: Sequence[str] = REQUIRED_ENGINE_PROBE_BACKENDS,
) -> ReleaseEvidenceInputStatus:
    required_backends = _validated_required_backends(required_engine_probe_backends)
    input_files = [
        _inspect_release_evidence_input_file("v1_benchmark", v1_benchmark_json),
        _inspect_release_evidence_input_file("storage_benchmark", storage_benchmark_json),
    ]
    input_files.extend(_inspect_release_evidence_input_file("engine_probe", path) for path in engine_probe_jsons)
    missing_paths = tuple(status.path for status in input_files if not status.exists)
    unreadable_paths = tuple(status.path for status in input_files if status.exists and not status.readable_json)
    present_probe_backends = {
        status.backend
        for status in input_files
        if status.role == "engine_probe" and status.readable_json and status.backend is not None
    }
    return ReleaseEvidenceInputStatus(
        input_files=tuple(input_files),
        missing_paths=missing_paths,
        unreadable_paths=unreadable_paths,
        missing_engine_probe_backends=tuple(backend for backend in required_backends if backend not in present_probe_backends),
        required_engine_probe_backends=required_backends,
    )


def release_evidence_to_record(evidence: ReleaseEvidence) -> dict[str, Any]:
    return {
        "record_type": RELEASE_EVIDENCE_RECORD_TYPE,
        "ok": evidence.ok,
        "v1_benchmark_ok": evidence.v1_benchmark_ok,
        "storage_benchmark_ok": evidence.storage_benchmark_ok,
        "engine_probe_backends": list(evidence.engine_probe_backends),
        "missing_engine_probe_backends": list(evidence.missing_engine_probe_backends),
        "duplicate_engine_probe_backends": list(evidence.duplicate_engine_probe_backends),
        "invalid_engine_probe_records": list(evidence.invalid_engine_probe_records),
        "artifact_sources": [_artifact_source_to_record(source) for source in evidence.artifact_sources],
        "issues": list(evidence.issues),
    }


def release_evidence_input_status_to_record(status: ReleaseEvidenceInputStatus) -> dict[str, Any]:
    return {
        "record_type": RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE,
        "ok": status.ok,
        "required_engine_probe_backends": list(status.required_engine_probe_backends),
        "missing_paths": list(status.missing_paths),
        "unreadable_paths": list(status.unreadable_paths),
        "missing_engine_probe_backends": list(status.missing_engine_probe_backends),
        "input_files": [_input_file_status_to_record(input_file) for input_file in status.input_files],
        "issues": list(status.issues),
    }


def write_release_evidence_json(evidence: ReleaseEvidence, path: str | Path) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(release_evidence_to_record(evidence), indent=2, sort_keys=True) + "\n")


def write_release_evidence_input_status_json(status: ReleaseEvidenceInputStatus, path: str | Path) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(release_evidence_input_status_to_record(status), indent=2, sort_keys=True) + "\n")


def _validated_artifact_sources(
    artifact_sources: Sequence[ReleaseEvidenceArtifactSource],
) -> tuple[ReleaseEvidenceArtifactSource, ...]:
    validated = []
    for index, source in enumerate(artifact_sources):
        if not isinstance(source, ReleaseEvidenceArtifactSource):
            raise TypeError(f"artifact_sources[{index}] must be ReleaseEvidenceArtifactSource")
        validated.append(source)
    return tuple(validated)


def _artifact_sources_for_records(
    *,
    v1_benchmark_json: str | Path,
    v1_record: Mapping[str, Any],
    storage_benchmark_json: str | Path,
    storage_record: Mapping[str, Any],
    engine_probe_jsons: Sequence[str | Path],
    engine_probe_records: Sequence[Mapping[str, Any]],
) -> tuple[ReleaseEvidenceArtifactSource, ...]:
    sources = [
        ReleaseEvidenceArtifactSource(
            role="v1_benchmark",
            path=str(v1_benchmark_json),
            record_type=_optional_str(v1_record.get("record_type")),
        ),
        ReleaseEvidenceArtifactSource(
            role="storage_benchmark",
            path=str(storage_benchmark_json),
            record_type=_optional_str(storage_record.get("record_type")),
        ),
    ]
    for path, record in zip(engine_probe_jsons, engine_probe_records, strict=True):
        sources.append(
            ReleaseEvidenceArtifactSource(
                role="engine_probe",
                path=str(path),
                record_type=_optional_str(record.get("record_type")),
                backend=_optional_backend(record.get("backend")),
            )
        )
    return tuple(sources)


def _artifact_source_to_record(source: ReleaseEvidenceArtifactSource) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": source.role,
        "path": source.path,
    }
    if source.record_type is not None:
        record["record_type"] = source.record_type
    if source.backend is not None:
        record["backend"] = source.backend
    return record


def _input_file_status_to_record(status: ReleaseEvidenceInputFileStatus) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": status.role,
        "path": status.path,
        "exists": status.exists,
        "readable_json": status.readable_json,
    }
    if status.record_type is not None:
        record["record_type"] = status.record_type
    if status.backend is not None:
        record["backend"] = status.backend
    if status.error is not None:
        record["error"] = status.error
    return record


def _inspect_release_evidence_input_file(role: str, path: str | Path) -> ReleaseEvidenceInputFileStatus:
    path_text = str(path)
    try:
        local = local_path(path_text)
        if not local.exists():
            return ReleaseEvidenceInputFileStatus(
                role=role,
                path=path_text,
                exists=False,
                readable_json=False,
                error="path does not exist",
            )
        record = json.loads(local.read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError("JSON root must be an object")
        return ReleaseEvidenceInputFileStatus(
            role=role,
            path=path_text,
            exists=True,
            readable_json=True,
            record_type=_optional_str(record.get("record_type")),
            backend=_optional_backend(record.get("backend")) if role == "engine_probe" else None,
        )
    except Exception as exc:
        return ReleaseEvidenceInputFileStatus(
            role=role,
            path=path_text,
            exists=True,
            readable_json=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_backend(value: Any) -> str | None:
    backend = _optional_str(value)
    return backend if backend in REQUIRED_ENGINE_PROBE_BACKENDS else None


def _v1_benchmark_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    if record.get("record_type") != BENCHMARK_RUN_RECORD_TYPE:
        issues.append(f"v1 benchmark record_type must be {BENCHMARK_RUN_RECORD_TYPE!r}")
    suite = _mapping_or_issue(record, "suite", issues)
    evidence = _mapping_or_issue(record, "v1_evidence", issues)
    measurements = _sequence_or_issue(record, "measurements", issues)
    report_rows = _sequence_or_issue(record, "report_rows", issues)
    comparisons = _sequence_or_issue(record, "comparisons", issues)
    if suite is not None:
        if suite.get("hardware_target") != DEFAULT_HARDWARE_TARGET:
            issues.append(f"v1 benchmark hardware_target must be {DEFAULT_HARDWARE_TARGET!r}")
        if suite.get("model_id") != DEFAULT_V1_MODEL_ID:
            issues.append(f"v1 benchmark model_id must be {DEFAULT_V1_MODEL_ID!r}")
    if evidence is not None and evidence.get("ok") is not True:
        evidence_issues = evidence.get("issues")
        if isinstance(evidence_issues, list) and evidence_issues:
            issues.extend(f"v1 benchmark evidence: {issue}" for issue in evidence_issues)
        else:
            issues.append("v1 benchmark evidence is not release-ready")
    if evidence is not None:
        if tuple(evidence.get("required_datasets", ())) != SUPPORTED_V1_DATASETS:
            issues.append("v1 benchmark evidence required_datasets must match the V1 release datasets")
        for field_name in (
            "missing_report_rows",
            "missing_comparisons",
            "comparisons_without_metrics",
            "rows_without_successful_requests",
            "rows_without_latency",
            "rows_without_quality",
            "unexpected_datasets",
        ):
            if evidence.get(field_name) not in ([], ()):
                issues.append(f"v1 benchmark evidence {field_name} must be empty")
    if measurements is not None:
        _validate_v1_measurements(measurements, issues)
    if report_rows is not None:
        _validate_v1_report_rows(report_rows, issues)
    if comparisons is not None:
        _validate_v1_comparisons(comparisons, issues)
    return tuple(issues)


def _storage_benchmark_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    if record.get("record_type") != STORAGE_BENCHMARK_RECORD_TYPE:
        issues.append(f"storage benchmark record_type must be {STORAGE_BENCHMARK_RECORD_TYPE!r}")
    if not _matches_release_storage_readers(record.get("readers")):
        issues.append("storage benchmark readers must include exactly Memory, Disk, and Unity Catalog")
    if not _is_real_uc_volume_root(record.get("uc_volume_root")):
        issues.append("storage benchmark uc_volume_root must be a real /Volumes/<catalog>/<schema>/<volume> path")
    results = _sequence_or_issue(record, "results", issues)
    evidence = _mapping_or_issue(record, "release_storage_evidence", issues)
    if evidence is None:
        return tuple(issues)
    if evidence.get("ok") is not True:
        evidence_issues = evidence.get("issues")
        if isinstance(evidence_issues, list) and evidence_issues:
            issues.extend(f"storage benchmark evidence: {issue}" for issue in evidence_issues)
        else:
            issues.append("storage benchmark release evidence is not release-ready")
    if evidence.get("uc_volume_is_real") is not True:
        issues.append("storage benchmark must use a real /Volumes/<catalog>/<schema>/<volume> UC Volume path")
    if evidence.get("require_real_uc_volume") is not True:
        issues.append("storage benchmark release evidence must require a real UC Volume")
    if not _is_real_uc_volume_root(evidence.get("uc_volume_root")):
        issues.append("storage benchmark release evidence uc_volume_root must be a real UC Volume path")
    if not _matches_release_storage_readers(evidence.get("required_readers")):
        issues.append("storage benchmark release evidence required_readers must match the release readers")
    for field_name in (
        "missing_readers",
        "readers_with_errors",
        "readers_without_latency",
        "readers_without_throughput",
    ):
        if evidence.get(field_name) not in ([], ()):
            issues.append(f"storage benchmark release evidence {field_name} must be empty")
    if results is not None:
        _validate_storage_results(results, issues)
    return tuple(issues)


def _engine_probe_evidence(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    backends: list[str] = []
    duplicate_backends: list[str] = []
    invalid_records: list[str] = []
    for index, record in enumerate(records):
        backend = record.get("backend")
        label = str(backend) if isinstance(backend, str) and backend else f"record[{index}]"
        try:
            validate_engine_kv_connector_probe_record(record)
            _validate_release_engine_probe_record(record)
        except Exception as exc:
            invalid_records.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        if label in backends:
            if label not in duplicate_backends:
                duplicate_backends.append(label)
            continue
        backends.append(label)
    return tuple(backends), tuple(invalid_records), tuple(duplicate_backends)


def _validated_required_backends(backends: Sequence[str]) -> tuple[str, ...]:
    if not backends:
        raise ValueError("required_engine_probe_backends must be non-empty")
    return _validated_backend_tuple(backends, "required_engine_probe_backends")


def _validated_backend_tuple(backends: Sequence[str], field_name: str) -> tuple[str, ...]:
    normalized = []
    supported = set(REQUIRED_ENGINE_PROBE_BACKENDS)
    for backend in backends:
        if backend not in supported:
            raise ValueError(f"Unsupported {field_name} backend {backend!r}")
        if backend not in normalized:
            normalized.append(backend)
    return tuple(normalized)


def _validated_str_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of non-empty strings")
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        normalized.append(value)
    return tuple(normalized)


def _mapping_or_issue(record: Mapping[str, Any], key: str, issues: list[str]) -> Mapping[str, Any] | None:
    value = record.get(key)
    if isinstance(value, Mapping):
        return value
    issues.append(f"{key} must be a mapping")
    return None


def _sequence_or_issue(record: Mapping[str, Any], key: str, issues: list[str]) -> Sequence[Any] | None:
    value = record.get(key)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    issues.append(f"{key} must be a sequence")
    return None


def _validate_v1_report_rows(report_rows: Sequence[Any], issues: list[str]) -> None:
    if not report_rows:
        issues.append("v1 benchmark report_rows must be non-empty")
        return
    row_keys = set()
    for index, row in enumerate(report_rows):
        if not isinstance(row, Mapping):
            issues.append(f"v1 benchmark report_rows[{index}] must be a mapping")
            continue
        dataset = row.get("dataset")
        arm_id = row.get("arm_id")
        row_keys.add((dataset, arm_id))
        if not _is_positive_int(row.get("requests")):
            issues.append(f"v1 benchmark report row {dataset}:{arm_id} requests must be positive")
        if type(row.get("errors")) is not int or row["errors"] < 0:
            issues.append(f"v1 benchmark report row {dataset}:{arm_id} errors must be a non-negative integer")
        for metric_name in ("ttft", "time_to_completion"):
            metric = row.get(metric_name)
            if not isinstance(metric, Mapping) or metric.get("p50") is None or metric.get("p95") is None:
                issues.append(f"v1 benchmark report row {dataset}:{arm_id} must include {metric_name} p50/p95")
            else:
                _validate_latency_summary(metric, f"v1 benchmark report row {dataset}:{arm_id} {metric_name}", issues)
        if row.get("answer_found_rate") is None and row.get("exact_match_rate") is None:
            issues.append(f"v1 benchmark report row {dataset}:{arm_id} must include quality metrics")
    expected = {
        (dataset, arm_id)
        for dataset in SUPPORTED_V1_DATASETS
        for arm_id in (BASELINE_PREFILL_ARM, CACHE_REUSE_ARM)
    }
    missing = sorted(f"{dataset}:{arm_id}" for dataset, arm_id in expected.difference(row_keys))
    if missing:
        issues.append(f"v1 benchmark report_rows missing required rows: {', '.join(missing)}")


def _validate_v1_measurements(measurements: Sequence[Any], issues: list[str]) -> None:
    if not measurements:
        issues.append("v1 benchmark measurements must be non-empty")
        return
    measurement_keys = set()
    for index, measurement in enumerate(measurements):
        if not isinstance(measurement, Mapping):
            issues.append(f"v1 benchmark measurements[{index}] must be a mapping")
            continue
        dataset = measurement.get("dataset")
        arm_id = measurement.get("arm_id")
        measurement_keys.add((dataset, arm_id))
        if dataset not in SUPPORTED_V1_DATASETS:
            issues.append(f"v1 benchmark measurement {index} has unsupported dataset {dataset!r}")
        if arm_id not in (BASELINE_PREFILL_ARM, CACHE_REUSE_ARM):
            issues.append(f"v1 benchmark measurement {index} has unsupported arm_id {arm_id!r}")
        for field_name in ("prompt_tokens", "completion_tokens"):
            if type(measurement.get(field_name)) is not int or measurement[field_name] < 0:
                issues.append(f"v1 benchmark measurement {dataset}:{arm_id} {field_name} must be a non-negative integer")
        for field_name in ("ttft_seconds", "time_to_completion_seconds"):
            value = measurement.get(field_name)
            if not _is_non_negative_number(value):
                issues.append(
                    f"v1 benchmark measurement {dataset}:{arm_id} "
                    f"{field_name} must be a non-negative finite number"
                )
        ttft_seconds = measurement.get("ttft_seconds")
        time_to_completion_seconds = measurement.get("time_to_completion_seconds")
        if _is_non_negative_number(ttft_seconds) and _is_non_negative_number(time_to_completion_seconds):
            if time_to_completion_seconds < ttft_seconds:
                issues.append(
                    f"v1 benchmark measurement {dataset}:{arm_id} "
                    "time_to_completion_seconds must be greater than or equal to ttft_seconds"
                )
        if measurement.get("error") not in (None, ""):
            issues.append(f"v1 benchmark measurement {dataset}:{arm_id} must not have an error")
        if measurement.get("answer_found") is not None and type(measurement.get("answer_found")) is not bool:
            issues.append(f"v1 benchmark measurement {dataset}:{arm_id} answer_found must be boolean when present")
    expected = {
        (dataset, arm_id)
        for dataset in SUPPORTED_V1_DATASETS
        for arm_id in (BASELINE_PREFILL_ARM, CACHE_REUSE_ARM)
    }
    missing = sorted(f"{dataset}:{arm_id}" for dataset, arm_id in expected.difference(measurement_keys))
    if missing:
        issues.append(f"v1 benchmark measurements missing required dataset/arm pairs: {', '.join(missing)}")


def _validate_v1_comparisons(comparisons: Sequence[Any], issues: list[str]) -> None:
    if not comparisons:
        issues.append("v1 benchmark comparisons must be non-empty")
        return
    by_dataset = {}
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, Mapping):
            issues.append(f"v1 benchmark comparisons[{index}] must be a mapping")
            continue
        dataset = comparison.get("dataset")
        by_dataset[dataset] = comparison
        if comparison.get("baseline_arm_id") != BASELINE_PREFILL_ARM:
            issues.append(f"v1 benchmark comparison {dataset} baseline_arm_id must be {BASELINE_PREFILL_ARM!r}")
        if comparison.get("cache_arm_id") != CACHE_REUSE_ARM:
            issues.append(f"v1 benchmark comparison {dataset} cache_arm_id must be {CACHE_REUSE_ARM!r}")
        for metric_name in ("ttft_speedup", "time_to_completion_speedup", "exact_match_delta", "answer_found_delta"):
            if comparison.get(metric_name) is None:
                issues.append(f"v1 benchmark comparison {dataset} must include {metric_name}")
    missing = sorted(set(SUPPORTED_V1_DATASETS).difference(by_dataset))
    if missing:
        issues.append(f"v1 benchmark comparisons missing required datasets: {', '.join(missing)}")


def _validate_storage_results(results: Sequence[Any], issues: list[str]) -> None:
    if not results:
        issues.append("storage benchmark results must be non-empty")
        return
    by_reader = {}
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            issues.append(f"storage benchmark results[{index}] must be a mapping")
            continue
        reader_id = result.get("reader_id")
        by_reader[reader_id] = result
        if result.get("errors") != 0:
            issues.append(f"storage benchmark reader {reader_id} errors must be zero")
        for field_name in ("total_reads", "total_bytes", "parallelism"):
            if not _is_positive_int(result.get(field_name)):
                issues.append(f"storage benchmark reader {reader_id} {field_name} must be a positive integer")
        wall_seconds = result.get("wall_seconds")
        if not _is_positive_number(wall_seconds):
            issues.append(f"storage benchmark reader {reader_id} wall_seconds must be a positive finite number")
        if result.get("latency_p50_seconds") is None or result.get("latency_p95_seconds") is None:
            issues.append(f"storage benchmark reader {reader_id} must include latency p50/p95")
        else:
            _validate_latency_summary(
                {"p50": result.get("latency_p50_seconds"), "p95": result.get("latency_p95_seconds")},
                f"storage benchmark reader {reader_id} latency",
                issues,
            )
        throughput = result.get("throughput_bytes_per_second")
        if not _is_positive_number(throughput):
            issues.append(f"storage benchmark reader {reader_id} throughput must be a positive finite number")
    missing = tuple(reader for reader in RELEASE_STORAGE_BENCHMARK_READERS if reader not in by_reader)
    if missing:
        issues.append(f"storage benchmark results missing required readers: {', '.join(missing)}")


def _validate_release_engine_probe_record(record: Mapping[str, Any]) -> None:
    if record.get("model_id") != DEFAULT_V1_MODEL_ID:
        raise ValueError(f"Engine KV probe model_id must be {DEFAULT_V1_MODEL_ID!r}")
    if record.get("layout_version") != "qwen3-v1":
        raise ValueError("Engine KV probe layout_version must be 'qwen3-v1'")
    layout = _release_probe_layout(record)
    _validate_v1_qwen3_probe_layout(layout)
    backend = record["backend"]
    if record.get("connector_package") != backend:
        raise ValueError("Engine KV probe connector_package must match backend")
    if record.get("engine_version") == "unknown":
        raise ValueError("Engine KV probe engine_version must be a real native engine version")
    _validate_probe_serving_profile_metadata(record, backend)


def _validate_probe_serving_profile_metadata(record: Mapping[str, Any], backend: str) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Engine KV probe metadata must be a mapping")
    profile = serving_environment_profile(backend)
    package = metadata.get(ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE)
    version = metadata.get(ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION)
    if record.get("engine_version") != profile.engine_version:
        raise ValueError("Engine KV probe engine_version must match the backend serving profile")
    if package != profile.engine_package:
        raise ValueError(
            "Engine KV probe serving_engine_package metadata must match the backend serving profile"
        )
    if version != profile.engine_version:
        raise ValueError(
            "Engine KV probe serving_engine_version metadata must match the backend serving profile"
        )


def _release_probe_layout(record: Mapping[str, Any]) -> KVLayout:
    layout = record.get("layout")
    if not isinstance(layout, Mapping):
        raise ValueError("Engine KV probe must include a layout mapping")
    return KVLayout(
        model_id=_required_str(layout, "model_id"),
        lora_id=_required_str(layout, "lora_id"),
        layout_version=_required_str(layout, "layout_version"),
        dtype=_required_str(layout, "dtype"),
        num_layers=_required_positive_int(layout, "num_layers"),
        block_size=_required_positive_int(layout, "block_size"),
        bytes_per_token=_required_positive_int(layout, "bytes_per_token"),
        num_query_heads=_required_positive_int(layout, "num_query_heads"),
        num_kv_heads=_required_positive_int(layout, "num_kv_heads"),
        head_size=_required_positive_int(layout, "head_size"),
        kv_stride_bytes=_required_positive_int(layout, "kv_stride_bytes"),
        shares_kv_storage=_required_bool(layout, "shares_kv_storage"),
        storage_layout=kv_storage_layout_from_value(
            _required_str(layout, "storage_layout"),
            field_name="layout.storage_layout",
        ),
    )


def _validate_v1_qwen3_probe_layout(layout: KVLayout) -> None:
    profile = get_model_profile(DEFAULT_V1_MODEL_ID)
    layout.validate()
    if layout.model_id != DEFAULT_V1_MODEL_ID:
        raise ValueError(f"Engine KV probe layout model_id must be {DEFAULT_V1_MODEL_ID!r}")
    if layout.layout_version != profile.default_layout_version:
        raise ValueError(f"Engine KV probe layout_version must be {profile.default_layout_version!r}")
    if dtype_byte_width(layout.dtype) != dtype_byte_width(profile.default_dtype):
        raise ValueError("Engine KV probe layout dtype must use one byte per KV scalar for V1")
    if (
        layout.num_layers != profile.num_layers
        or layout.num_query_heads != profile.num_query_heads
        or layout.num_kv_heads != profile.num_kv_heads
        or layout.head_size != profile.head_size
        or layout.attention_mechanism != AttentionMechanism.GROUPED_QUERY
    ):
        raise ValueError("Engine KV probe layout must use the V1 Qwen3 GQA geometry")
    expected_bytes_per_token = profile.kv_scalars_per_token * dtype_byte_width(layout.dtype)
    if layout.bytes_per_token != expected_bytes_per_token:
        raise ValueError("Engine KV probe layout bytes_per_token must match the V1 Qwen3 GQA geometry")
    expected_stride = profile.head_size * dtype_byte_width(layout.dtype)
    if layout.kv_stride_bytes != expected_stride:
        raise ValueError("Engine KV probe layout kv_stride_bytes must match the V1 Qwen3 head stride")
    if layout.shares_kv_storage is not True or layout.storage_layout != KVStorageLayout.SHARED_KEY_VALUE:
        raise ValueError("Engine KV probe layout must use the V1 Qwen3 shared K/V storage layout")


def _is_real_uc_volume_root(value: Any) -> bool:
    return is_real_uc_volume_root(value) is True


def _matches_release_storage_readers(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return False
    if any(not isinstance(reader, str) or not reader for reader in value):
        return False
    return (
        len(value) == len(RELEASE_STORAGE_BENCHMARK_READERS)
        and set(value) == set(RELEASE_STORAGE_BENCHMARK_READERS)
    )


def _is_positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _is_non_negative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _is_positive_number(value: Any) -> bool:
    return _is_non_negative_number(value) and value > 0


def _validate_latency_summary(metric: Mapping[str, Any], label: str, issues: list[str]) -> None:
    p50 = metric.get("p50")
    p95 = metric.get("p95")
    if not _is_non_negative_number(p50):
        issues.append(f"{label} p50 must be a non-negative finite number")
    if not _is_non_negative_number(p95):
        issues.append(f"{label} p95 must be a non-negative finite number")
    if _is_non_negative_number(p50) and _is_non_negative_number(p95) and p95 < p50:
        issues.append(f"{label} p95 must be greater than or equal to p50")


def _required_str(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_positive_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if not _is_positive_int(value):
        raise ValueError(f"{key} must be a positive integer")
    return value


def _required_bool(record: Mapping[str, Any], key: str) -> bool:
    value = record.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be boolean")
    return value


def _read_json_record(path: str | Path) -> Mapping[str, Any]:
    record = json.loads(local_path(str(path)).read_text(encoding="utf-8"))
    if not isinstance(record, Mapping):
        raise ValueError(f"Release evidence input {path} must contain a JSON object")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Document KV Cache release evidence JSON artifacts.")
    parser.add_argument("--v1-benchmark-json", required=True)
    parser.add_argument("--storage-benchmark-json", required=True)
    parser.add_argument("--engine-probe-json", action="append", default=[])
    parser.add_argument("--output-json", help="Write the release evidence JSON to this path instead of stdout.")
    parser.add_argument("--preflight-output-json", help="Write release-evidence input file status JSON before validation.")
    parser.add_argument("--preflight-only", action="store_true", help="Only inspect input file availability and record types.")
    args = parser.parse_args(argv)

    try:
        if args.preflight_output_json or args.preflight_only:
            input_status = inspect_release_evidence_input_files(
                v1_benchmark_json=args.v1_benchmark_json,
                storage_benchmark_json=args.storage_benchmark_json,
                engine_probe_jsons=tuple(args.engine_probe_json),
            )
            if args.preflight_output_json:
                write_release_evidence_input_status_json(input_status, args.preflight_output_json)
            if args.preflight_only:
                if not args.preflight_output_json:
                    print(json.dumps(release_evidence_input_status_to_record(input_status), indent=2, sort_keys=True))
                return 0 if input_status.ok else 2
        evidence = evaluate_release_evidence_files(
            v1_benchmark_json=args.v1_benchmark_json,
            storage_benchmark_json=args.storage_benchmark_json,
            engine_probe_jsons=tuple(args.engine_probe_json),
        )
        if args.output_json:
            write_release_evidence_json(evidence, args.output_json)
        else:
            print(json.dumps(release_evidence_to_record(evidence), indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0 if evidence.ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
