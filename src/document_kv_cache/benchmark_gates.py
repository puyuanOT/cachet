"""Publication gates for fair, method-aware KV reuse benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from document_kv_cache.artifact_identity import ArtifactIdentity
from document_kv_cache.benchmark_runner import BenchmarkRunResult
from document_kv_cache.benchmark_statistics import paired_benchmark_statistics


BENCHMARK_PUBLICATION_GATE_RECORD_TYPE = "document_kv.benchmark_publication_gate.v1"
CACHE_STATE_ATTESTATION_RECORD_TYPE = "document_kv.cache_state_attestation.v1"

__all__ = [
    "BENCHMARK_PUBLICATION_GATE_RECORD_TYPE",
    "CACHE_STATE_ATTESTATION_RECORD_TYPE",
    "CacheStateAttestation",
    "BenchmarkPublicationGateConfig",
    "BenchmarkPublicationGateResult",
    "cache_state_attestation_from_vllm_telemetry",
    "evaluate_benchmark_publication_gate",
    "benchmark_publication_gate_to_record",
]


@dataclass(frozen=True, slots=True)
class CacheStateAttestation:
    """Mechanically observed state for one cache hydrate request."""

    request_id: str
    cache_method: str
    artifact_id: str
    source: str
    bytes_read: int
    payload_cache_hit: bool
    eviction_requested: bool
    eviction_succeeded: bool
    direct_io: bool = False
    expected_bytes: int | None = None
    expected_tokens: int | None = None
    loaded_tokens: int | None = None
    successful_loads: int = 1

    def __post_init__(self) -> None:
        for field_name in ("request_id", "cache_method", "artifact_id", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if type(self.bytes_read) is not int or self.bytes_read < 0:
            raise ValueError("bytes_read must be a non-negative integer")
        for field_name in ("expected_bytes", "expected_tokens", "loaded_tokens"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer when provided")
        if type(self.successful_loads) is not int or self.successful_loads < 0:
            raise ValueError("successful_loads must be a non-negative integer")
        for field_name in (
            "payload_cache_hit",
            "eviction_requested",
            "eviction_succeeded",
            "direct_io",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be a boolean")
        if self.eviction_succeeded and not self.eviction_requested:
            raise ValueError("eviction_succeeded requires eviction_requested")

    @property
    def cold_read_attested(self) -> bool:
        return (
            self.source in {"disk", "file", "local_path", "uri"}
            and self.bytes_read > 0
            and not self.payload_cache_hit
            and (self.direct_io or (self.eviction_requested and self.eviction_succeeded))
            and self.successful_loads == 1
            and (
                self.expected_bytes is None
                or self.bytes_read == self.expected_bytes
            )
            and (
                self.expected_tokens is None
                or self.loaded_tokens == self.expected_tokens
            )
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": CACHE_STATE_ATTESTATION_RECORD_TYPE,
            "request_id": self.request_id,
            "cache_method": self.cache_method,
            "artifact_id": self.artifact_id,
            "source": self.source,
            "bytes_read": self.bytes_read,
            "payload_cache_hit": self.payload_cache_hit,
            "eviction_requested": self.eviction_requested,
            "eviction_succeeded": self.eviction_succeeded,
            "direct_io": self.direct_io,
            "expected_bytes": self.expected_bytes,
            "expected_tokens": self.expected_tokens,
            "loaded_tokens": self.loaded_tokens,
            "successful_loads": self.successful_loads,
            "cold_read_attested": self.cold_read_attested,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkPublicationGateConfig:
    min_successful_requests_per_row: int = 1
    max_error_rate: float = 0.0
    max_exact_match_drop: float = 0.02
    max_answer_found_drop: float = 0.02
    require_method_identity: bool = True
    require_variant_identity: bool = True
    require_artifact_identity: bool = True
    require_resolved_artifact_identity: bool = True
    require_cold_attestation: bool = True
    require_unique_prefix_cache_salt: bool = True
    min_paired_samples: int = 1
    paired_confidence_level: float = 0.95
    paired_bootstrap_samples: int = 2_000
    paired_bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.min_successful_requests_per_row) is not int
            or self.min_successful_requests_per_row <= 0
        ):
            raise ValueError("min_successful_requests_per_row must be positive")
        if type(self.min_paired_samples) is not int or self.min_paired_samples <= 0:
            raise ValueError("min_paired_samples must be positive")
        for field_name in (
            "max_error_rate",
            "max_exact_match_drop",
            "max_answer_found_drop",
            "paired_confidence_level",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            if field_name == "paired_confidence_level":
                if not 0 < float(value) < 1:
                    raise ValueError("paired_confidence_level must be between zero and one")
            elif not 0 <= float(value) <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if type(self.paired_bootstrap_samples) is not int or self.paired_bootstrap_samples <= 0:
            raise ValueError("paired_bootstrap_samples must be positive")
        if type(self.paired_bootstrap_seed) is not int:
            raise ValueError("paired_bootstrap_seed must be an integer")
        for field_name in (
            "require_method_identity",
            "require_variant_identity",
            "require_artifact_identity",
            "require_resolved_artifact_identity",
            "require_cold_attestation",
            "require_unique_prefix_cache_salt",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be a boolean")


@dataclass(frozen=True, slots=True)
class BenchmarkPublicationGateResult:
    issues: tuple[str, ...]
    checked_cache_arms: tuple[str, ...]
    checked_cache_requests: int
    cold_attested_requests: int

    @property
    def ok(self) -> bool:
        return not self.issues


def evaluate_benchmark_publication_gate(
    result: BenchmarkRunResult,
    *,
    config: BenchmarkPublicationGateConfig | None = None,
    cache_state_attestations: Iterable[CacheStateAttestation] = (),
    artifact_identities: Mapping[str, ArtifactIdentity] | None = None,
) -> BenchmarkPublicationGateResult:
    """Evaluate whether a benchmark is safe to present as comparative evidence."""

    if not isinstance(result, BenchmarkRunResult):
        raise TypeError("result must be a BenchmarkRunResult")
    resolved = config or BenchmarkPublicationGateConfig()
    attestations = tuple(cache_state_attestations)
    for attestation in attestations:
        if not isinstance(attestation, CacheStateAttestation):
            raise TypeError("cache_state_attestations entries must be CacheStateAttestation")
    identities = {} if artifact_identities is None else dict(artifact_identities)
    for artifact_id, identity in identities.items():
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("artifact_identities keys must be non-empty strings")
        if not isinstance(identity, ArtifactIdentity):
            raise TypeError("artifact_identities values must be ArtifactIdentity")
        if artifact_id != identity.artifact_id:
            raise ValueError("artifact_identities key does not match ArtifactIdentity.artifact_id")

    issues: list[str] = []
    cache_arm_ids = result.cache_arm_ids
    cache_rows = tuple(row for row in result.report_rows if row.arm_id in cache_arm_ids)
    cache_measurements = tuple(
        measurement
        for measurement in result.measurements
        if measurement.arm_id in cache_arm_ids and measurement.ok
    )

    if resolved.require_unique_prefix_cache_salt:
        if result.prefix_cache_salt_mode != "per_request":
            issues.append("publication requires prefix_cache_salt_mode='per_request'")
        salts = [
            measurement.metadata.get("prefix_cache_salt", "")
            for measurement in cache_measurements
        ]
        missing_salt_requests = [
            measurement.request_id or f"{measurement.dataset}:{measurement.example_id}"
            for measurement, salt in zip(cache_measurements, salts, strict=True)
            if not salt
        ]
        for request_id in missing_salt_requests:
            issues.append(f"cache request {request_id} is missing a prefix cache salt")
        duplicate_salts = sorted(
            salt for salt in set(salts) if salt and salts.count(salt) > 1
        )
        for salt in duplicate_salts:
            issues.append(f"prefix cache salt {salt!r} is reused across cache requests")

    if not cache_arm_ids:
        issues.append("benchmark does not contain a cache arm")
    for row in cache_rows:
        row_label = f"{row.dataset}:{row.arm_id}"
        successes = row.requests - row.errors
        if successes < resolved.min_successful_requests_per_row:
            issues.append(
                f"{row_label} has {successes} successful requests; "
                f"requires {resolved.min_successful_requests_per_row}"
            )
        error_rate = row.errors / row.requests if row.requests else 1.0
        if error_rate > resolved.max_error_rate:
            issues.append(
                f"{row_label} error rate {error_rate:.6g} exceeds "
                f"{resolved.max_error_rate:.6g}"
            )
        if row.ttft.p50 is None or row.time_to_completion.p50 is None:
            issues.append(f"{row_label} is missing latency metrics")
        if row.exact_match_rate is None or row.answer_found_rate is None:
            issues.append(f"{row_label} is missing quality metrics")
        if resolved.require_method_identity and not row.cache_method:
            issues.append(f"{row_label} is missing cache_method identity")
        if resolved.require_variant_identity and not row.variant_id:
            issues.append(f"{row_label} is missing variant_id identity")
        if resolved.require_artifact_identity and not row.artifact_id:
            issues.append(f"{row_label} is missing artifact_id identity")

    paired_rows = paired_benchmark_statistics(
        result,
        confidence_level=resolved.paired_confidence_level,
        bootstrap_samples=resolved.paired_bootstrap_samples,
        seed=resolved.paired_bootstrap_seed,
    )
    for paired in paired_rows:
        row_label = f"{paired.dataset}:{paired.cache_arm_id}"
        if paired.paired_samples < resolved.min_paired_samples:
            issues.append(
                f"{row_label} has {paired.paired_samples} paired samples; "
                f"requires {resolved.min_paired_samples}"
            )
        if paired.missing_baseline_pairs:
            issues.append(f"{row_label} has {paired.missing_baseline_pairs} cache-only pairs")
        if paired.missing_cache_pairs:
            issues.append(f"{row_label} has {paired.missing_cache_pairs} baseline-only pairs")
        if paired.duplicate_pair_keys:
            issues.append(
                f"{row_label} has duplicate pair keys: {', '.join(paired.duplicate_pair_keys)}"
            )
        for metric_name in (
            "ttft_speedup",
            "time_to_completion_speedup",
            "exact_match_delta",
            "answer_found_delta",
        ):
            if getattr(paired, metric_name) is None:
                issues.append(f"{row_label} is missing paired {metric_name}")
        if (
            paired.exact_match_delta is not None
            and paired.exact_match_delta.lower < -resolved.max_exact_match_drop
        ):
            issues.append(
                f"{row_label} paired exact-match lower bound "
                f"{paired.exact_match_delta.lower:.6g} exceeds the allowed drop"
            )
        if (
            paired.answer_found_delta is not None
            and paired.answer_found_delta.lower < -resolved.max_answer_found_drop
        ):
            issues.append(
                f"{row_label} paired answer-found lower bound "
                f"{paired.answer_found_delta.lower:.6g} exceeds the allowed drop"
            )

    comparison_keys = {
        (comparison.dataset, comparison.cache_arm_id): comparison
        for comparison in result.comparisons
        if comparison.baseline_arm_id == result.baseline_arm_id
    }
    for row in cache_rows:
        key = (row.dataset, row.arm_id)
        comparison = comparison_keys.get(key)
        row_label = f"{row.dataset}:{row.arm_id}"
        if comparison is None:
            issues.append(f"{row_label} has no baseline comparison")
            continue
        if comparison.exact_match_delta is None:
            issues.append(f"{row_label} exact-match delta is missing")
        elif comparison.exact_match_delta < -resolved.max_exact_match_drop:
            issues.append(
                f"{row_label} exact-match drop {-comparison.exact_match_delta:.6g} exceeds "
                f"{resolved.max_exact_match_drop:.6g}"
            )
        if comparison.answer_found_delta is None:
            issues.append(f"{row_label} answer-found delta is missing")
        elif comparison.answer_found_delta < -resolved.max_answer_found_drop:
            issues.append(
                f"{row_label} answer-found drop {-comparison.answer_found_delta:.6g} exceeds "
                f"{resolved.max_answer_found_drop:.6g}"
            )

    observed_artifact_ids = {
        measurement.artifact_id for measurement in cache_measurements if measurement.artifact_id
    }
    if resolved.require_resolved_artifact_identity:
        for artifact_id in sorted(observed_artifact_ids):
            observed_identity = identities.get(artifact_id)
            if observed_identity is None:
                issues.append(f"artifact {artifact_id} has no submitted ArtifactIdentity descriptor")
            elif observed_identity.has_unresolved_fields:
                issues.append(f"artifact {artifact_id} contains unresolved identity fields")

    attestations_by_request: dict[str, CacheStateAttestation] = {}
    duplicate_attestations: set[str] = set()
    for attestation in attestations:
        if attestation.request_id in attestations_by_request:
            duplicate_attestations.add(attestation.request_id)
        attestations_by_request[attestation.request_id] = attestation
    for request_id in sorted(duplicate_attestations):
        issues.append(f"cache request {request_id} has duplicate cache-state attestations")

    cold_attested = 0
    if resolved.require_cold_attestation:
        for measurement in cache_measurements:
            label = measurement.request_id or (
                f"{measurement.dataset}:{measurement.example_id}:repeat-{measurement.repeat_index}"
            )
            if not measurement.request_id:
                issues.append(f"{label} is missing request_id required for cache-state attestation")
                continue
            observed_attestation = attestations_by_request.get(measurement.request_id)
            if observed_attestation is None:
                issues.append(f"cache request {measurement.request_id} has no cache-state attestation")
                continue
            if measurement.cache_method and observed_attestation.cache_method != measurement.cache_method:
                issues.append(f"cache request {measurement.request_id} attests a different cache_method")
            if measurement.artifact_id and observed_attestation.artifact_id != measurement.artifact_id:
                issues.append(f"cache request {measurement.request_id} attests a different artifact_id")
            if (
                observed_attestation.expected_bytes is None
                or observed_attestation.expected_tokens is None
                or observed_attestation.loaded_tokens is None
            ):
                issues.append(
                    f"cache request {measurement.request_id} is missing expected byte/token counts"
                )
            if observed_attestation.successful_loads != 1:
                issues.append(
                    f"cache request {measurement.request_id} must attest exactly one successful load"
                )
            if not observed_attestation.cold_read_attested:
                issues.append(f"cache request {measurement.request_id} is not mechanically attested cold")
            else:
                cold_attested += 1

    return BenchmarkPublicationGateResult(
        issues=tuple(issues),
        checked_cache_arms=cache_arm_ids,
        checked_cache_requests=len(cache_measurements),
        cold_attested_requests=cold_attested,
    )


def cache_state_attestation_from_vllm_telemetry(
    record: Mapping[str, Any],
) -> CacheStateAttestation:
    """Parse the provider's explicit cache-state attestation section."""

    if not isinstance(record, Mapping):
        raise TypeError("vLLM telemetry record must be a mapping")
    if record.get("record_type") != "document_kv.vllm_native_provider_load.v1":
        raise ValueError("unsupported vLLM telemetry record_type")
    state = record.get("cache_state_attestation")
    if not isinstance(state, Mapping):
        raise ValueError("vLLM telemetry is missing cache_state_attestation")
    return CacheStateAttestation(
        request_id=_required_string(record, "request_id"),
        cache_method=_required_string(state, "cache_method"),
        artifact_id=_required_string(state, "artifact_id"),
        source=_required_string(state, "source"),
        bytes_read=_required_nonnegative_int(state, "bytes_read"),
        payload_cache_hit=_required_bool(state, "payload_cache_hit"),
        eviction_requested=_required_bool(state, "eviction_requested"),
        eviction_succeeded=_required_bool(state, "eviction_succeeded"),
        direct_io=_required_bool(state, "direct_io"),
        expected_bytes=_optional_nonnegative_int(state, "expected_bytes"),
        expected_tokens=_optional_nonnegative_int(state, "expected_tokens"),
        loaded_tokens=_optional_nonnegative_int(state, "loaded_tokens"),
        successful_loads=_required_nonnegative_int(state, "successful_loads"),
    )


def benchmark_publication_gate_to_record(
    gate: BenchmarkPublicationGateResult,
) -> dict[str, Any]:
    if not isinstance(gate, BenchmarkPublicationGateResult):
        raise TypeError("gate must be a BenchmarkPublicationGateResult")
    return {
        "record_type": BENCHMARK_PUBLICATION_GATE_RECORD_TYPE,
        "ok": gate.ok,
        "issues": list(gate.issues),
        "checked_cache_arms": list(gate.checked_cache_arms),
        "checked_cache_requests": gate.checked_cache_requests,
        "cold_attested_requests": gate.cold_attested_requests,
    }


def _required_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_nonnegative_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _required_bool(record: Mapping[str, Any], key: str) -> bool:
    value = record.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _optional_nonnegative_int(record: Mapping[str, Any], key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} must be a non-negative integer when provided")
    return value
