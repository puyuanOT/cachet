"""Legacy compatibility migration evidence for Cachet release governance."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_kv_cache._hardware_targets import DEFAULT_HARDWARE_TARGET, SUPPORTED_V1_HARDWARE_TARGETS
from document_kv_cache.storage import local_path


LEGACY_COMPATIBILITY_MIGRATION_RECORD_TYPE = "document_kv.legacy_compatibility_migration.v1"
LEGACY_COMPATIBILITY_MIGRATION_VALIDATION_RECORD_TYPE = (
    "document_kv.legacy_compatibility_migration_validation.v1"
)
LEGACY_COMPATIBILITY_REQUIRED_JOB_CATEGORIES = (
    "release",
    "benchmark",
    "storage",
    "native_probe",
    "smoke",
)
LEGACY_COMPATIBILITY_ALLOWED_IMPORT_SURFACES = ("cachet", "document_kv_cache")
LEGACY_COMPATIBILITY_ALLOWED_COMMAND_PREFIXES = ("cachet-", "document-kv-")
_MIGRATION_RECORD_KEYS = frozenset(
    {
        "record_type",
        "ok",
        "checked_downstream_jobs",
        "release_evidence",
        "issues",
    }
)
_DOWNSTREAM_JOB_KEYS = frozenset(
    {
        "name",
        "category",
        "environment",
        "migrated_import_surface",
        "migrated_command_prefix",
        "legacy_imports_present",
        "legacy_console_scripts_present",
        "evidence_uri",
    }
)
_RELEASE_EVIDENCE_KEYS = frozenset(
    {
        "hardware_target",
        "evidence_uri",
        "runner_uses_legacy_facade",
    }
)

__all__ = [
    "LEGACY_COMPATIBILITY_MIGRATION_RECORD_TYPE",
    "LEGACY_COMPATIBILITY_MIGRATION_VALIDATION_RECORD_TYPE",
    "LEGACY_COMPATIBILITY_REQUIRED_JOB_CATEGORIES",
    "LEGACY_COMPATIBILITY_ALLOWED_IMPORT_SURFACES",
    "LEGACY_COMPATIBILITY_ALLOWED_COMMAND_PREFIXES",
    "LegacyCompatibilityMigrationEvidence",
    "evaluate_legacy_compatibility_migration_record",
    "evaluate_legacy_compatibility_migration_file",
    "legacy_compatibility_migration_to_record",
    "legacy_compatibility_migration_validation_to_record",
    "write_legacy_compatibility_migration_json",
    "main",
]


@dataclass(frozen=True, slots=True)
class LegacyCompatibilityMigrationEvidence:
    checked_downstream_jobs: tuple[dict[str, Any], ...]
    release_evidence: tuple[dict[str, Any], ...]
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.issues

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checked_downstream_jobs",
            tuple(dict(job) for job in self.checked_downstream_jobs),
        )
        object.__setattr__(
            self,
            "release_evidence",
            tuple(dict(evidence) for evidence in self.release_evidence),
        )
        object.__setattr__(self, "issues", _dedupe_strings(_string_sequence(self.issues, "issues")))


def evaluate_legacy_compatibility_migration_record(
    record: Mapping[str, Any],
) -> LegacyCompatibilityMigrationEvidence:
    if record.get("record_type") != LEGACY_COMPATIBILITY_MIGRATION_RECORD_TYPE:
        return _failed_migration_evidence(
            f"record_type must be {LEGACY_COMPATIBILITY_MIGRATION_RECORD_TYPE!r}"
        )
    issues: list[str] = []
    issues.extend(_unexpected_keys(record, _MIGRATION_RECORD_KEYS, "legacy migration evidence"))
    if record.get("ok") is not True:
        issues.append("legacy migration evidence ok must be true")
    checked_jobs = _mapping_sequence(record.get("checked_downstream_jobs"), "checked_downstream_jobs", issues)
    release_evidence = _mapping_sequence(record.get("release_evidence"), "release_evidence", issues)
    issues.extend(_checked_downstream_job_issues(checked_jobs))
    issues.extend(_release_evidence_issues(release_evidence))
    issues.extend(_record_issues_field(record))
    return LegacyCompatibilityMigrationEvidence(
        checked_downstream_jobs=checked_jobs,
        release_evidence=release_evidence,
        issues=tuple(issues),
    )


def evaluate_legacy_compatibility_migration_file(
    path: str | Path,
) -> LegacyCompatibilityMigrationEvidence:
    evidence_path = local_path(str(path))
    try:
        record = json.loads(evidence_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        return _failed_migration_evidence(f"{evidence_path} must contain valid UTF-8 JSON: {exc.reason}")
    except json.JSONDecodeError as exc:
        return _failed_migration_evidence(f"{evidence_path} must contain valid JSON: {exc.msg}")
    except OSError as exc:
        return _failed_migration_evidence(f"{evidence_path} could not be read: {exc}")
    if not isinstance(record, Mapping):
        return _failed_migration_evidence(f"{evidence_path} must contain a JSON object")
    return evaluate_legacy_compatibility_migration_record(record)


def legacy_compatibility_migration_to_record(
    evidence: LegacyCompatibilityMigrationEvidence,
) -> dict[str, Any]:
    return {
        "record_type": LEGACY_COMPATIBILITY_MIGRATION_RECORD_TYPE,
        "ok": evidence.ok,
        "checked_downstream_jobs": list(evidence.checked_downstream_jobs),
        "release_evidence": list(evidence.release_evidence),
        "issues": list(evidence.issues),
    }


def legacy_compatibility_migration_validation_to_record(
    evidence_by_file: Mapping[str, LegacyCompatibilityMigrationEvidence],
) -> dict[str, Any]:
    return {
        "record_type": LEGACY_COMPATIBILITY_MIGRATION_VALIDATION_RECORD_TYPE,
        "ok": bool(evidence_by_file) and all(evidence.ok for evidence in evidence_by_file.values()),
        "files": {
            path: legacy_compatibility_migration_to_record(evidence)
            for path, evidence in evidence_by_file.items()
        },
    }


def write_legacy_compatibility_migration_json(
    evidence: LegacyCompatibilityMigrationEvidence,
    path: str | Path,
) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(legacy_compatibility_migration_to_record(evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _checked_downstream_job_issues(jobs: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    issues: list[str] = []
    if not jobs:
        issues.append("checked_downstream_jobs must include at least one job for each required category")
        return tuple(issues)
    seen_categories: set[str] = set()
    seen_names: set[str] = set()
    for index, job in enumerate(jobs):
        label = f"checked_downstream_jobs[{index}]"
        issues.extend(_unexpected_keys(job, _DOWNSTREAM_JOB_KEYS, label))
        name = _required_str(job, "name", label, issues)
        if name:
            if name in seen_names:
                issues.append(f"{label}.name must be unique")
            seen_names.add(name)
        category = _required_str(job, "category", label, issues)
        if category:
            if category not in LEGACY_COMPATIBILITY_REQUIRED_JOB_CATEGORIES:
                issues.append(
                    f"{label}.category must be one of {LEGACY_COMPATIBILITY_REQUIRED_JOB_CATEGORIES!r}"
                )
            else:
                seen_categories.add(category)
        _required_str(job, "environment", label, issues)
        import_surface = _required_str(job, "migrated_import_surface", label, issues)
        if import_surface and import_surface not in LEGACY_COMPATIBILITY_ALLOWED_IMPORT_SURFACES:
            issues.append(
                f"{label}.migrated_import_surface must be one of "
                f"{LEGACY_COMPATIBILITY_ALLOWED_IMPORT_SURFACES!r}"
            )
        command_prefix = _required_str(job, "migrated_command_prefix", label, issues)
        if command_prefix and command_prefix not in LEGACY_COMPATIBILITY_ALLOWED_COMMAND_PREFIXES:
            issues.append(
                f"{label}.migrated_command_prefix must be one of "
                f"{LEGACY_COMPATIBILITY_ALLOWED_COMMAND_PREFIXES!r}"
            )
        _required_false(job, "legacy_imports_present", label, issues)
        _required_false(job, "legacy_console_scripts_present", label, issues)
        _required_str(job, "evidence_uri", label, issues)

    missing_categories = [
        category
        for category in LEGACY_COMPATIBILITY_REQUIRED_JOB_CATEGORIES
        if category not in seen_categories
    ]
    if missing_categories:
        issues.append(
            "checked_downstream_jobs must cover required categories: "
            + ", ".join(missing_categories)
        )
    return tuple(issues)


def _release_evidence_issues(release_evidence: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    issues: list[str] = []
    if not release_evidence:
        issues.append("release_evidence must include current AWS g6/L4 release evidence")
        return tuple(issues)
    seen_targets: set[str] = set()
    for index, evidence in enumerate(release_evidence):
        label = f"release_evidence[{index}]"
        issues.extend(_unexpected_keys(evidence, _RELEASE_EVIDENCE_KEYS, label))
        hardware_target = _required_str(evidence, "hardware_target", label, issues)
        if hardware_target:
            if hardware_target in seen_targets:
                issues.append(f"{label}.hardware_target must be unique")
            seen_targets.add(hardware_target)
            if hardware_target not in SUPPORTED_V1_HARDWARE_TARGETS:
                issues.append(f"{label}.hardware_target must be one of {SUPPORTED_V1_HARDWARE_TARGETS!r}")
        _required_str(evidence, "evidence_uri", label, issues)
        _required_false(evidence, "runner_uses_legacy_facade", label, issues)
    if DEFAULT_HARDWARE_TARGET not in seen_targets:
        issues.append(
            f"release_evidence must include the strict V1 hardware target {DEFAULT_HARDWARE_TARGET!r}"
        )
    return tuple(issues)


def _mapping_sequence(value: Any, field_name: str, issues: list[str]) -> tuple[dict[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        records: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                records.append(dict(item))
            else:
                issues.append(f"{field_name}[{index}] must be an object")
        return tuple(records)
    issues.append(f"{field_name} must be an array")
    return ()


def _required_str(record: Mapping[str, Any], field_name: str, label: str, issues: list[str]) -> str:
    value = record.get(field_name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    issues.append(f"{label}.{field_name} must be a non-empty string")
    return ""


def _required_false(record: Mapping[str, Any], field_name: str, label: str, issues: list[str]) -> None:
    if record.get(field_name) is not False:
        issues.append(f"{label}.{field_name} must be false")


def _record_issues_field(record: Mapping[str, Any]) -> tuple[str, ...]:
    value = record.get("issues", ())
    if value in ([], ()):
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        explicit_issues = [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
        return tuple(f"legacy migration evidence explicit issue: {issue}" for issue in explicit_issues)
    return ("legacy migration evidence issues must be an empty array",)


def _unexpected_keys(record: Mapping[str, Any], allowed_keys: frozenset[str], label: str) -> tuple[str, ...]:
    unexpected = sorted(str(key) for key in record if key not in allowed_keys)
    if not unexpected:
        return ()
    return (f"{label} has unsupported keys: {unexpected}",)


def _string_sequence(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of non-empty strings")
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings")
        normalized.append(value.strip())
    return tuple(normalized)


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _failed_migration_evidence(*issues: str) -> LegacyCompatibilityMigrationEvidence:
    return LegacyCompatibilityMigrationEvidence(
        checked_downstream_jobs=(),
        release_evidence=(),
        issues=issues,
    )


def _evaluate_files(paths: Sequence[str | Path]) -> dict[str, LegacyCompatibilityMigrationEvidence]:
    return {
        str(path): evaluate_legacy_compatibility_migration_file(path)
        for path in paths
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Cachet legacy compatibility migration evidence."
    )
    parser.add_argument(
        "--validate-json",
        action="append",
        default=[],
        metavar="PATH",
        help="Validate a legacy compatibility migration evidence JSON sidecar.",
    )
    parser.add_argument("--output-json", help="Write validation JSON instead of printing it.")
    args = parser.parse_args(argv)

    if not args.validate_json:
        parser.error("--validate-json is required")

    evidence_by_file = _evaluate_files(tuple(args.validate_json))
    record: dict[str, Any]
    if len(evidence_by_file) == 1:
        record = legacy_compatibility_migration_to_record(next(iter(evidence_by_file.values())))
    else:
        record = legacy_compatibility_migration_validation_to_record(evidence_by_file)
    if args.output_json:
        output_path = local_path(str(args.output_json))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
