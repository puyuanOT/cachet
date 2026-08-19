"""Publication gates for fair, method-aware KV reuse benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from document_kv_cache.artifact_identity import ArtifactIdentity
from document_kv_cache._benchmark_manifest import (
    VARIES_BY_ARM,
    _validate_arm_runtime_environments,
)
from document_kv_cache.benchmark_runner import BenchmarkRunResult
from document_kv_cache.benchmark_statistics import paired_benchmark_statistics
from document_kv_cache.benchmarks import SUPPORTED_V1_DATASETS
from document_kv_cache.methods import MethodRegistry, default_method_registry


BENCHMARK_PUBLICATION_GATE_RECORD_TYPE = "document_kv.benchmark_publication_gate.v1"
BENCHMARK_EVIDENCE_GATE_RECORD_TYPE = "document_kv.benchmark_evidence_gate.v1"
CACHE_STATE_ATTESTATION_RECORD_TYPE = "document_kv.cache_state_attestation.v1"

__all__ = [
    "BENCHMARK_PUBLICATION_GATE_RECORD_TYPE",
    "BENCHMARK_EVIDENCE_GATE_RECORD_TYPE",
    "CACHE_STATE_ATTESTATION_RECORD_TYPE",
    "CacheStateAttestation",
    "BenchmarkPublicationGateConfig",
    "BenchmarkPublicationGateResult",
    "cache_state_attestation_from_vllm_telemetry",
    "evaluate_benchmark_publication_gate",
    "evaluate_benchmark_evidence_gate",
    "benchmark_evidence_policy_config",
    "benchmark_publication_gate_to_record",
    "benchmark_evidence_gate_to_record",
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
    policy: Literal["smoke", "canary", "publication"] = "publication"
    min_successful_requests_per_row: int = 4
    min_distinct_examples_per_row: int = 4
    min_quality_examples_per_row: int = 50
    allow_complete_split_below_quality_min: bool = True
    max_error_rate: float = 0.0
    max_exact_match_drop: float = 0.02
    max_answer_found_drop: float = 0.02
    require_method_identity: bool = True
    require_variant_identity: bool = True
    require_artifact_identity: bool = True
    require_resolved_artifact_identity: bool = True
    require_cold_attestation: bool = True
    require_unique_prefix_cache_salt: bool = True
    min_paired_samples: int = 4
    min_paired_examples: int = 4
    paired_confidence_level: float = 0.95
    paired_bootstrap_samples: int = 2_000
    paired_bootstrap_seed: int = 0
    require_manifest: bool = True
    require_resolved_provenance: bool = True
    require_approved_scorer: bool = True
    require_logical_pairing: bool = True

    def __post_init__(self) -> None:
        if self.policy not in {"smoke", "canary", "publication"}:
            raise ValueError("policy must be smoke, canary, or publication")
        if (
            type(self.min_successful_requests_per_row) is not int
            or self.min_successful_requests_per_row <= 0
        ):
            raise ValueError("min_successful_requests_per_row must be positive")
        if type(self.min_paired_samples) is not int or self.min_paired_samples <= 0:
            raise ValueError("min_paired_samples must be positive")
        for field_name in (
            "min_distinct_examples_per_row",
            "min_quality_examples_per_row",
            "min_paired_examples",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be positive")
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
            "allow_complete_split_below_quality_min",
            "require_manifest",
            "require_resolved_provenance",
            "require_approved_scorer",
            "require_logical_pairing",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be a boolean")


@dataclass(frozen=True, slots=True)
class BenchmarkPublicationGateResult:
    issues: tuple[str, ...]
    checked_cache_arms: tuple[str, ...]
    checked_cache_requests: int
    cold_attested_requests: int
    policy: Literal["smoke", "canary", "publication"] = "publication"
    checked_distinct_examples: int = 0
    measurement_scopes: tuple[str, ...] = ()
    benchmark_payload_digest: str = ""

    def __post_init__(self) -> None:
        if self.benchmark_payload_digest and not _is_sha256_digest(
            self.benchmark_payload_digest
        ):
            raise ValueError("benchmark_payload_digest must be a SHA-256 digest")

    @property
    def ok(self) -> bool:
        return not self.issues


def benchmark_evidence_policy_config(
    policy: Literal["smoke", "canary", "publication"],
) -> BenchmarkPublicationGateConfig:
    """Return the rigorous default gate for an evidence maturity level."""

    if policy == "publication":
        return BenchmarkPublicationGateConfig()
    if policy == "canary":
        return BenchmarkPublicationGateConfig(
            policy="canary",
            min_successful_requests_per_row=2,
            min_distinct_examples_per_row=2,
            min_quality_examples_per_row=2,
            min_paired_samples=2,
            min_paired_examples=2,
            require_artifact_identity=False,
            require_resolved_artifact_identity=False,
            require_cold_attestation=False,
            require_resolved_provenance=False,
            require_approved_scorer=False,
        )
    if policy == "smoke":
        return BenchmarkPublicationGateConfig(
            policy="smoke",
            min_successful_requests_per_row=1,
            min_distinct_examples_per_row=1,
            min_quality_examples_per_row=1,
            min_paired_samples=1,
            min_paired_examples=1,
            max_exact_match_drop=1.0,
            max_answer_found_drop=1.0,
            require_method_identity=False,
            require_variant_identity=False,
            require_artifact_identity=False,
            require_resolved_artifact_identity=False,
            require_cold_attestation=False,
            require_unique_prefix_cache_salt=False,
            require_manifest=False,
            require_resolved_provenance=False,
            require_approved_scorer=False,
            require_logical_pairing=False,
        )
    raise ValueError("policy must be smoke, canary, or publication")


def evaluate_benchmark_evidence_gate(
    result: BenchmarkRunResult,
    *,
    policy: Literal["smoke", "canary", "publication"] = "canary",
    config: BenchmarkPublicationGateConfig | None = None,
    cache_state_attestations: Iterable[CacheStateAttestation] = (),
    artifact_identities: Mapping[str, ArtifactIdentity] | None = None,
    method_registry: MethodRegistry | None = None,
    benchmark_payload_digest: str | None = None,
) -> BenchmarkPublicationGateResult:
    resolved = config or benchmark_evidence_policy_config(policy)
    if resolved.policy != policy:
        raise ValueError("config.policy must match policy")
    return evaluate_benchmark_publication_gate(
        result,
        config=resolved,
        cache_state_attestations=cache_state_attestations,
        artifact_identities=artifact_identities,
        method_registry=method_registry,
        benchmark_payload_digest=benchmark_payload_digest,
    )


def evaluate_benchmark_publication_gate(
    result: BenchmarkRunResult,
    *,
    config: BenchmarkPublicationGateConfig | None = None,
    cache_state_attestations: Iterable[CacheStateAttestation] = (),
    artifact_identities: Mapping[str, ArtifactIdentity] | None = None,
    method_registry: MethodRegistry | None = None,
    benchmark_payload_digest: str | None = None,
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
    registry = default_method_registry() if method_registry is None else method_registry
    if not isinstance(registry, MethodRegistry):
        raise TypeError("method_registry must be a MethodRegistry or None")
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
    manifest = result.experiment_manifest
    manifest_arms = {
        arm.arm_id: arm for arm in manifest.arms
    } if manifest is not None else {}
    physical_cache_arm_ids = tuple(
        arm.arm_id
        for arm in result.arms
        if arm.uses_cache
    ) or cache_arm_ids
    physical_cache_rows = tuple(
        row for row in result.report_rows if row.arm_id in physical_cache_arm_ids
    )
    cache_measurements = tuple(
        measurement
        for measurement in result.measurements
        if measurement.arm_id in physical_cache_arm_ids and measurement.ok
    )
    active_scopes = set(
        manifest.measurement_scopes
        if manifest is not None
        else ("latency", "quality")
    )
    if resolved.policy == "smoke":
        # Smoke establishes execution, not statistical evidence.
        active_scopes = set()
    latency_claim = "latency" in active_scopes
    quality_claim = "quality" in active_scopes
    resource_claim = "resource" in active_scopes
    quality_specs_by_dataset = _quality_specs_by_dataset(
        result,
        config=resolved,
    ) if quality_claim else {}
    cold_latency_claim = latency_claim and (
        manifest is None
        or any(
            _is_cold_cache_state(arm.runtime_environment.cache_state)
            for arm in manifest.arms
            if arm.arm_id in physical_cache_arm_ids
        )
    )
    if resolved.require_manifest and manifest is None:
        issues.append(f"{resolved.policy} evidence requires an experiment manifest")
    if manifest is not None:
        if manifest.example_count != len(result.suite.examples):
            issues.append("experiment manifest example_count does not match the suite")
        if manifest.datasets != tuple(result.suite.datasets):
            issues.append("experiment manifest datasets do not match the suite")
        if (
            manifest.model_id != result.suite.model_id
            and manifest.model_id != VARIES_BY_ARM
        ):
            issues.append("experiment manifest model_id does not match the suite")
        if (
            manifest.hardware_target != result.suite.hardware_target
            and manifest.hardware_target != VARIES_BY_ARM
        ):
            issues.append("experiment manifest hardware_target does not match the suite")
        try:
            _validate_arm_runtime_environments(
                manifest.arms,
                comparison_mode=manifest.comparison_mode,
                varied_setting=manifest.varied_setting,
                reference_arm_id=manifest.baseline_arm_id,
            )
        except ValueError as exc:
            issues.append(f"invalid per-arm runtime environment contract: {exc}")
        if resolved.require_resolved_provenance and manifest.has_unresolved_provenance:
            issues.append("publication evidence contains unresolved experiment provenance")
        issues.extend(_measurement_arm_identity_issues(result))
        issues.extend(_logical_workload_token_issues(result))
        if resolved.require_approved_scorer and quality_claim:
            for scorer in manifest.scorer_identities:
                if not scorer.publication_approved:
                    issues.append(
                        f"dataset {scorer.dataset!r} scorer "
                        f"{scorer.scorer_id}@{scorer.version} is not publication-approved"
                    )
                if not scorer.plugin_path:
                    issues.append(
                        f"dataset {scorer.dataset!r} scorer "
                        f"{scorer.scorer_id}@{scorer.version} has no plugin path"
                    )
        if resolved.policy == "publication":
            scorers_by_dataset = {
                scorer.dataset: scorer for scorer in manifest.scorer_identities
            }
            for dataset in manifest.datasets:
                candidate_scorer = scorers_by_dataset.get(dataset)
                if candidate_scorer is None:
                    issues.append(
                        f"dataset {dataset!r} has no versioned scorer/prompt identity"
                    )
                    continue
                if dataset not in SUPPORTED_V1_DATASETS and (
                    not candidate_scorer.prompt_plugin_path
                    or not candidate_scorer.prompt_template_version
                ):
                    issues.append(
                        f"custom dataset {dataset!r} requires a versioned prompt "
                        "plugin path and template version"
                    )
            if (
                len(manifest.arms) > 1
                and manifest.execution_isolation_mode
                != "separate_process_or_job"
            ):
                issues.append(
                    "publication comparisons require separate_process_or_job "
                    "execution isolation"
                )
            if len(manifest.arms) > 1:
                source_execution_ids = tuple(manifest.source_execution_ids)
                source_execution_arms = tuple(
                    arm_id for arm_id, _digest in source_execution_ids
                )
                source_execution_digests = tuple(
                    digest for _arm_id, digest in source_execution_ids
                )
                expected_execution_arms = {arm.arm_id for arm in manifest.arms}
                if set(source_execution_arms) != expected_execution_arms:
                    issues.append(
                        "publication comparisons require one source execution "
                        "identity for every arm"
                    )
                if len(set(source_execution_arms)) != len(source_execution_arms):
                    issues.append(
                        "source execution identities contain duplicate arm ids"
                    )
                if (
                    len(source_execution_digests)
                    != len(set(source_execution_digests))
                    or any(
                        not _is_sha256_digest(value)
                        for value in source_execution_digests
                    )
                ):
                    issues.append(
                        "publication comparisons require distinct resolved source "
                        "execution identity digests"
                    )
            for arm in manifest.arms:
                if arm.implementation_kind == "cachet" and not arm.method_config_digest:
                    issues.append(
                        f"cache arm {arm.arm_id!r} is missing method_config_digest"
                    )
                if arm.implementation_kind in {"upstream", "external"}:
                    if not arm.source_revision:
                        issues.append(
                            f"external arm {arm.arm_id!r} is missing source_revision"
                        )
                    if not arm.checkpoint_identity:
                        issues.append(
                            f"external arm {arm.arm_id!r} is missing checkpoint_identity"
                        )
                    if not arm.method_config_digest:
                        issues.append(
                            f"external arm {arm.arm_id!r} is missing method_config_digest"
                        )

    if resolved.require_logical_pairing:
        issues.extend(_logical_pairing_issues(result))

    if resolved.require_unique_prefix_cache_salt and cold_latency_claim:
        if result.prefix_cache_salt_mode != "per_request":
            issues.append(
                f"{resolved.policy} evidence requires prefix_cache_salt_mode='per_request'"
            )
        salts = [
            _prefix_cache_salt_identity(measurement.metadata)
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
        for _salt in duplicate_salts:
            issues.append("a prefix cache salt is reused across cache requests")

    if not physical_cache_arm_ids and resolved.policy != "smoke":
        issues.append("benchmark does not contain a cache arm")
    checked_rows = (
        result.report_rows
        if resolved.policy == "smoke"
        else physical_cache_rows
    )
    distinct_examples_by_row = {
        (row.dataset, row.arm_id): len(
            {
                measurement.example_id
                for measurement in result.measurements
                if measurement.dataset == row.dataset
                and measurement.arm_id == row.arm_id
                and measurement.ok
            }
        )
        for row in checked_rows
    }
    for row in checked_rows:
        row_label = f"{row.dataset}:{row.arm_id}"
        successes = row.requests - row.errors
        if successes < resolved.min_successful_requests_per_row:
            issues.append(
                f"{row_label} has {successes} successful requests; "
                f"requires {resolved.min_successful_requests_per_row}"
            )
        distinct_examples = distinct_examples_by_row[(row.dataset, row.arm_id)]
        if distinct_examples < resolved.min_distinct_examples_per_row:
            issues.append(
                f"{row_label} has {distinct_examples} distinct examples; "
                f"requires {resolved.min_distinct_examples_per_row}"
            )
        error_rate = row.errors / row.requests if row.requests else 1.0
        if error_rate > resolved.max_error_rate:
            issues.append(
                f"{row_label} error rate {error_rate:.6g} exceeds "
                f"{resolved.max_error_rate:.6g}"
            )
        if latency_claim and (
            row.ttft.p50 is None or row.time_to_completion.p50 is None
        ):
            issues.append(f"{row_label} is missing latency metrics")
        for metric_name in quality_specs_by_dataset.get(row.dataset, {}):
            if _row_quality_metric_value(
                row,
                metric_name,
                legacy=manifest is None,
            ) is None:
                issues.append(f"{row_label} is missing quality metric {metric_name!r}")
                continue
            quality_examples = len(
                {
                    measurement.example_id
                    for measurement in result.measurements
                    if measurement.dataset == row.dataset
                    and measurement.arm_id == row.arm_id
                    and measurement.ok
                    and _measurement_quality_value(
                        measurement,
                        metric_name,
                        legacy=manifest is None,
                    )
                    is not None
                }
            )
            if (
                quality_examples < resolved.min_quality_examples_per_row
                and not (
                    resolved.allow_complete_split_below_quality_min
                    and manifest is not None
                    and manifest.complete_dataset_split
                )
            ):
                issues.append(
                    f"{row_label} has {quality_examples} unique examples for "
                    f"quality metric {metric_name!r}; requires "
                    f"{resolved.min_quality_examples_per_row} or a declared complete split"
                )
        if resolved.require_method_identity and not row.cache_method:
            issues.append(f"{row_label} is missing cache_method identity")
        if resolved.require_variant_identity and not row.variant_id:
            issues.append(f"{row_label} is missing variant_id identity")
        arm_manifest = manifest_arms.get(row.arm_id)
        requires_cachet_artifact = (
            arm_manifest is None or arm_manifest.implementation_kind == "cachet"
        )
        if (
            resolved.require_artifact_identity
            and requires_cachet_artifact
            and not row.artifact_id
        ):
            issues.append(f"{row_label} is missing artifact_id identity")

    paired_rows = paired_benchmark_statistics(
        result,
        confidence_level=resolved.paired_confidence_level,
        bootstrap_samples=resolved.paired_bootstrap_samples,
        seed=resolved.paired_bootstrap_seed,
    )
    for paired in (paired_rows if (latency_claim or quality_claim) else ()):
        row_label = f"{paired.dataset}:{paired.cache_arm_id}"
        if paired.paired_samples < resolved.min_paired_samples:
            issues.append(
                f"{row_label} has {paired.paired_samples} paired samples; "
                f"requires {resolved.min_paired_samples}"
            )
        if paired.paired_examples < resolved.min_paired_examples:
            issues.append(
                f"{row_label} has {paired.paired_examples} paired examples; "
                f"requires {resolved.min_paired_examples}"
            )
        if paired.missing_baseline_pairs:
            issues.append(f"{row_label} has {paired.missing_baseline_pairs} cache-only pairs")
        if paired.missing_cache_pairs:
            issues.append(f"{row_label} has {paired.missing_cache_pairs} baseline-only pairs")
        if paired.duplicate_pair_keys:
            issues.append(
                f"{row_label} has duplicate pair keys: {', '.join(paired.duplicate_pair_keys)}"
            )
        metric_names: tuple[str, ...] = (
            ("ttft_speedup", "time_to_completion_speedup")
            if latency_claim
            else ()
        )
        for metric_name in metric_names:
            if getattr(paired, metric_name) is None:
                issues.append(f"{row_label} is missing paired {metric_name}")
        for metric_name, (direction, tolerance) in quality_specs_by_dataset.get(
            paired.dataset,
            {},
        ).items():
            interval = _paired_quality_interval(
                paired,
                metric_name,
                legacy=manifest is None,
            )
            if interval is None:
                issues.append(f"{row_label} is missing paired quality metric {metric_name!r}")
                continue
            if direction == "higher_is_better" and interval.lower < -tolerance:
                issues.append(
                    f"{row_label} paired {metric_name!r} lower bound "
                    f"{interval.lower:.6g} exceeds allowed regression {tolerance:.6g}"
                )
            if direction == "lower_is_better" and interval.upper > tolerance:
                issues.append(
                    f"{row_label} paired {metric_name!r} upper bound "
                    f"{interval.upper:.6g} exceeds allowed regression {tolerance:.6g}"
                )

    comparison_keys = {
        (comparison.dataset, comparison.cache_arm_id): comparison
        for comparison in result.comparisons
        if comparison.baseline_arm_id == result.baseline_arm_id
    }
    for row in (cache_rows if (latency_claim or quality_claim) else ()):
        key = (row.dataset, row.arm_id)
        comparison = comparison_keys.get(key)
        row_label = f"{row.dataset}:{row.arm_id}"
        if comparison is None:
            issues.append(f"{row_label} has no baseline comparison")
            continue
        for metric_name in quality_specs_by_dataset.get(row.dataset, {}):
            delta = _comparison_quality_delta(
                comparison,
                metric_name,
                legacy=manifest is None,
            )
            if delta is None:
                issues.append(
                    f"{row_label} quality delta {metric_name!r} is missing"
                )
            elif manifest is None:
                tolerance = quality_specs_by_dataset[row.dataset][metric_name][1]
                if delta < -tolerance:
                    display_name = metric_name.replace("_", "-")
                    issues.append(
                        f"{row_label} {display_name} drop {-delta:.6g} exceeds "
                        f"{tolerance:.6g}"
                    )

    cachet_arm_ids = {
        arm_id
        for arm_id, arm in manifest_arms.items()
        if arm.implementation_kind == "cachet"
    } or set(physical_cache_arm_ids)
    observed_artifact_ids = {
        measurement.artifact_id
        for measurement in cache_measurements
        if measurement.arm_id in cachet_arm_ids and measurement.artifact_id
    }
    if manifest is not None and resolved.policy != "smoke":
        for arm_id in sorted(cachet_arm_ids):
            cachet_arm = manifest_arms.get(arm_id)
            if cachet_arm is None or cachet_arm.implementation_kind != "cachet":
                continue
            measured_arm_artifacts = {
                measurement.artifact_id
                for measurement in cache_measurements
                if measurement.arm_id == arm_id and measurement.artifact_id
            }
            if measured_arm_artifacts != set(cachet_arm.artifact_ids):
                issues.append(
                    f"Cachet arm {arm_id!r} measurement artifacts do not match "
                    "the experiment manifest"
                )
            issues.extend(
                _cachet_arm_contract_issues(
                    manifest,
                    cachet_arm,
                    identities=identities,
                    registry=registry,
                )
            )
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
    if resolved.require_cold_attestation and cold_latency_claim:
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

    if resource_claim:
        issues.extend(_resource_scope_issues(result))

    if benchmark_payload_digest is None:
        from document_kv_cache.benchmark_runner import (
            benchmark_record_payload_digest,
            benchmark_run_result_payload_to_record,
        )

        benchmark_payload_digest = benchmark_record_payload_digest(
            benchmark_run_result_payload_to_record(result)
        )
    elif not _is_sha256_digest(benchmark_payload_digest):
        raise ValueError("benchmark_payload_digest must be a SHA-256 digest")
    return BenchmarkPublicationGateResult(
        issues=tuple(issues),
        checked_cache_arms=cache_arm_ids,
        checked_cache_requests=len(cache_measurements),
        cold_attested_requests=cold_attested,
        policy=resolved.policy,
        checked_distinct_examples=len(
            {
                (measurement.dataset, measurement.example_id)
                for measurement in result.measurements
                if measurement.ok
            }
        ),
        measurement_scopes=tuple(sorted(active_scopes)),
        benchmark_payload_digest=benchmark_payload_digest,
    )


def cache_state_attestation_from_vllm_telemetry(
    record: Mapping[str, Any],
) -> CacheStateAttestation:
    """Parse provider state, preferring its explicit benchmark correlation id."""

    if not isinstance(record, Mapping):
        raise TypeError("vLLM telemetry record must be a mapping")
    if record.get("record_type") != "document_kv.vllm_native_provider_load.v1":
        raise ValueError("unsupported vLLM telemetry record_type")
    state = record.get("cache_state_attestation")
    if not isinstance(state, Mapping):
        raise ValueError("vLLM telemetry is missing cache_state_attestation")
    request_id = (
        _required_string(record, "request_id")
        if record.get("benchmark_request_id") is None
        else _required_string(record, "benchmark_request_id")
    )
    return CacheStateAttestation(
        request_id=request_id,
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
        "policy": gate.policy,
        "ok": gate.ok,
        "issues": list(gate.issues),
        "checked_cache_arms": list(gate.checked_cache_arms),
        "checked_cache_requests": gate.checked_cache_requests,
        "cold_attested_requests": gate.cold_attested_requests,
        "checked_distinct_examples": gate.checked_distinct_examples,
        "measurement_scopes": list(gate.measurement_scopes),
        "benchmark_payload_digest": gate.benchmark_payload_digest,
    }


def benchmark_evidence_gate_to_record(
    gate: BenchmarkPublicationGateResult,
) -> dict[str, Any]:
    record = benchmark_publication_gate_to_record(gate)
    record["record_type"] = BENCHMARK_EVIDENCE_GATE_RECORD_TYPE
    record["gate_version"] = 1
    return record


def _logical_pairing_issues(result: BenchmarkRunResult) -> tuple[str, ...]:
    baseline: dict[tuple[str, str, int], str] = {}
    issues: list[str] = []
    for measurement in result.measurements:
        if measurement.arm_id != result.baseline_arm_id or not measurement.ok:
            continue
        key = (measurement.dataset, measurement.example_id, measurement.repeat_index)
        digest = measurement.metadata.get("logical_prompt_sha256", "")
        if not digest:
            issues.append(
                f"baseline pair {measurement.dataset}:{measurement.example_id}:"
                f"repeat-{measurement.repeat_index} is missing logical_prompt_sha256"
            )
        baseline[key] = digest
    for measurement in result.measurements:
        if measurement.arm_id not in result.cache_arm_ids or not measurement.ok:
            continue
        key = (measurement.dataset, measurement.example_id, measurement.repeat_index)
        label = (
            f"{measurement.dataset}:{measurement.example_id}:"
            f"{measurement.arm_id}:repeat-{measurement.repeat_index}"
        )
        observed = measurement.metadata.get("logical_prompt_sha256", "")
        expected = baseline.get(key)
        if not observed:
            issues.append(f"cache pair {label} is missing logical_prompt_sha256")
        elif expected is not None and observed != expected:
            issues.append(f"cache pair {label} changes the logical prompt")
    return tuple(issues)


def _logical_workload_token_issues(
    result: BenchmarkRunResult,
) -> tuple[str, ...]:
    manifest = result.experiment_manifest
    if manifest is None:
        return ()
    issues: list[str] = []
    input_target = manifest.input_tokens_target
    output_target = manifest.output_tokens_target
    force_output_length = manifest.decode_settings.get("ignore_eos") is True
    if force_output_length and output_target is None:
        issues.append(
            "decoding setting ignore_eos=true requires an output_tokens_target"
        )
    for measurement in result.measurements:
        if not measurement.ok:
            continue
        label = (
            f"{measurement.dataset}:{measurement.example_id}:"
            f"{measurement.arm_id}:repeat-{measurement.repeat_index}"
        )
        if input_target is not None:
            raw_tokens = measurement.metadata.get("logical_prompt_tokens", "")
            try:
                observed_tokens = int(raw_tokens)
            except (TypeError, ValueError):
                observed_tokens = 0
            if observed_tokens <= 0:
                issues.append(
                    f"measurement {label} is missing positive "
                    "metadata.logical_prompt_tokens for the declared input target"
                )
            elif observed_tokens != input_target:
                issues.append(
                    f"measurement {label} logical_prompt_tokens={observed_tokens} "
                    f"does not match input_tokens_target={input_target}"
                )
        if (
            force_output_length
            and output_target is not None
            and measurement.completion_tokens != output_target
        ):
            issues.append(
                f"measurement {label} completion_tokens="
                f"{measurement.completion_tokens} does not match forced "
                f"output_tokens_target={output_target}"
            )
    return tuple(issues)


def _measurement_arm_identity_issues(
    result: BenchmarkRunResult,
) -> tuple[str, ...]:
    manifest = result.experiment_manifest
    if manifest is None:
        return ()
    arms_by_id = {arm.arm_id: arm for arm in manifest.arms}
    issues: list[str] = []
    for measurement in result.measurements:
        arm = arms_by_id.get(measurement.arm_id)
        if arm is None:
            issues.append(
                f"measurement references unknown manifest arm {measurement.arm_id!r}"
            )
            continue
        expected_method = arm.method_id if arm.uses_cache else ""
        if measurement.cache_method != expected_method:
            issues.append(
                f"measurement arm {measurement.arm_id!r} cache_method "
                f"{measurement.cache_method!r} does not match manifest method "
                f"{expected_method!r}"
            )
        if measurement.variant_id != arm.variant_id:
            issues.append(
                f"measurement arm {measurement.arm_id!r} variant_id "
                f"{measurement.variant_id!r} does not match manifest variant "
                f"{arm.variant_id!r}"
            )
    for row in result.report_rows:
        arm = arms_by_id.get(row.arm_id)
        if arm is None:
            issues.append(f"report row references unknown manifest arm {row.arm_id!r}")
            continue
        expected_method = arm.method_id if arm.uses_cache else ""
        if row.cache_method != expected_method:
            issues.append(
                f"report row arm {row.arm_id!r} cache_method {row.cache_method!r} "
                f"does not match manifest method {expected_method!r}"
            )
        if row.variant_id != arm.variant_id:
            issues.append(
                f"report row arm {row.arm_id!r} variant_id {row.variant_id!r} "
                f"does not match manifest variant {arm.variant_id!r}"
            )
    return tuple(issues)


def _is_cold_cache_state(value: str) -> bool:
    return value.strip().lower().replace("-", "_") in {
        "cold",
        "cold_cache",
        "cold_read",
    }


def _prefix_cache_salt_identity(metadata: Mapping[str, str]) -> str:
    sanitized = metadata.get("prefix_cache_salt_sha256", "")
    if sanitized:
        return sanitized
    return metadata.get("prefix_cache_salt", "")


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _quality_specs_by_dataset(
    result: BenchmarkRunResult,
    *,
    config: BenchmarkPublicationGateConfig,
) -> dict[str, dict[str, tuple[str, float]]]:
    manifest = result.experiment_manifest
    if manifest is None:
        return {
            dataset: {
                "exact_match": ("higher_is_better", config.max_exact_match_drop),
                "answer_found": ("higher_is_better", config.max_answer_found_drop),
            }
            for dataset in result.suite.datasets
        }
    return {
        scorer.dataset: {
            metric.metric_name: (metric.direction, metric.max_regression)
            for metric in scorer.metric_specs
        }
        for scorer in manifest.scorer_identities
    }


def _measurement_quality_value(
    measurement: Any,
    metric_name: str,
    *,
    legacy: bool,
) -> float | None:
    value = measurement.quality_scores.get(metric_name)
    if value is not None:
        return float(value)
    if legacy and metric_name == "exact_match":
        diagnostic = measurement.exact_match
        return None if diagnostic is None else float(diagnostic)
    if legacy and metric_name == "answer_found":
        diagnostic = measurement.answer_found
        return None if diagnostic is None else float(diagnostic)
    return None


def _row_quality_metric_value(
    row: Any,
    metric_name: str,
    *,
    legacy: bool,
) -> float | None:
    value = row.quality_score_means.get(metric_name)
    if value is not None:
        return float(value)
    if legacy and metric_name == "exact_match":
        rate = row.exact_match_rate
        return None if rate is None else float(rate)
    if legacy and metric_name == "answer_found":
        rate = row.answer_found_rate
        return None if rate is None else float(rate)
    return None


def _paired_quality_interval(
    paired: Any,
    metric_name: str,
    *,
    legacy: bool,
) -> Any:
    interval = paired.quality_score_deltas.get(metric_name)
    if interval is not None:
        return interval
    if legacy and metric_name == "exact_match":
        return paired.exact_match_delta
    if legacy and metric_name == "answer_found":
        return paired.answer_found_delta
    return None


def _comparison_quality_delta(
    comparison: Any,
    metric_name: str,
    *,
    legacy: bool,
) -> float | None:
    value = comparison.quality_score_deltas.get(metric_name)
    if value is not None:
        return float(value)
    if legacy and metric_name == "exact_match":
        delta = comparison.exact_match_delta
        return None if delta is None else float(delta)
    if legacy and metric_name == "answer_found":
        delta = comparison.answer_found_delta
        return None if delta is None else float(delta)
    return None


def _resource_scope_issues(result: BenchmarkRunResult) -> tuple[str, ...]:
    manifest = result.experiment_manifest
    if manifest is None:
        return ("resource evidence requires an experiment manifest",)
    resource_metadata_keys = {
        "gpu_memory_bytes",
        "cpu_memory_bytes",
        "storage_read_bytes",
        "gpu_utilization",
        "cpu_utilization",
        "energy_joules",
    }
    issues: list[str] = []
    for arm in manifest.arms:
        offline_values = (
            arm.offline_training_seconds,
            arm.offline_artifact_generation_seconds,
            arm.offline_checkpoint_load_seconds,
            arm.artifact_bytes,
            arm.offline_peak_memory_bytes,
        )
        has_request_resource = any(
            resource_metadata_keys.intersection(measurement.metadata)
            for measurement in result.measurements
            if measurement.arm_id == arm.arm_id
        )
        if not any(value is not None for value in offline_values) and not has_request_resource:
            issues.append(f"resource arm {arm.arm_id!r} has no resource measurements")
    return tuple(issues)


def _cachet_arm_contract_issues(
    manifest: Any,
    arm: Any,
    *,
    identities: Mapping[str, ArtifactIdentity],
    registry: MethodRegistry,
) -> tuple[str, ...]:
    issues: list[str] = []
    try:
        method = registry.get(arm.method_id, require_implemented=True)
    except (KeyError, NotImplementedError):
        return (
            f"Cachet arm {arm.arm_id!r} references an unknown or planned method "
            f"{arm.method_id!r}",
        )
    if arm.method_version != method.artifact_version:
        issues.append(
            f"Cachet arm {arm.arm_id!r} method_version does not match the registry"
        )
    if arm.connector_mode != method.connector_mode:
        issues.append(
            f"Cachet arm {arm.arm_id!r} connector_mode does not match the registry"
        )
    for artifact_id in arm.artifact_ids:
        identity = identities.get(artifact_id)
        if identity is None:
            continue
        environment = arm.runtime_environment
        expected = {
            "method_id": arm.method_id,
            "method_version": arm.method_version,
            "method_config_digest": arm.method_config_digest,
            "model_id": environment.canonical_model_id,
            "model_revision": environment.model_revision,
            "tokenizer_id": environment.tokenizer_id,
            "tokenizer_revision": environment.tokenizer_revision,
            "lora_id": environment.lora_id,
            "prompt_template_version": environment.prompt_template_version,
            "layout_version": environment.layout_version,
            "runtime_kv_dtype": environment.runtime_kv_dtype,
            "block_size": environment.block_size,
            "payload_axis_order": environment.payload_axis_order,
            "key_position_encoding": environment.key_position_encoding,
            "rope_theta": environment.rope_theta,
            "rope_rotary_dim": environment.rope_rotary_dim,
            "tensor_parallel_size": environment.tensor_parallel_size,
            "pipeline_parallel_size": environment.pipeline_parallel_size,
            "artifact_format_id": method.artifact_format.format_id,
            "artifact_format_version": method.artifact_format.version,
        }
        mismatches = [
            field_name
            for field_name, expected_value in expected.items()
            if getattr(identity, field_name) != expected_value
        ]
        if mismatches:
            issues.append(
                f"artifact {artifact_id} does not match Cachet arm {arm.arm_id!r}: "
                + ", ".join(mismatches)
            )
    return tuple(issues)


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
