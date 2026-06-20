"""Pull-request traceability evidence for Document KV Cache."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_kv_cache.storage import local_path


PR_EVIDENCE_RECORD_TYPE = "document_kv.pr_evidence.v1"
PR_EVIDENCE_VALIDATION_RECORD_TYPE = "document_kv.pr_evidence_validation.v1"
GPT55_REVIEW_OUTCOMES = ("clean", "findings_resolved")

__all__ = [
    "PR_EVIDENCE_RECORD_TYPE",
    "PR_EVIDENCE_VALIDATION_RECORD_TYPE",
    "GPT55_REVIEW_OUTCOMES",
    "PullRequestEvidence",
    "evaluate_pr_evidence",
    "evaluate_pr_evidence_directory",
    "evaluate_pr_evidence_file",
    "evaluate_pr_evidence_record",
    "pr_evidence_validation_to_record",
    "pr_evidence_to_record",
    "write_pr_evidence_json",
    "main",
]


@dataclass(frozen=True, slots=True)
class PullRequestEvidence:
    what_changed: tuple[str, ...]
    why: str
    scope: tuple[str, ...]
    verification: tuple[str, ...]
    refactor_skill_applied: bool
    gpt55_review_completed: bool
    gpt55_review_findings_resolved: bool
    gpt55_review_outcome: str
    gpt55_review_summary: str
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.issues

    def __post_init__(self) -> None:
        object.__setattr__(self, "what_changed", _string_tuple(self.what_changed, "what_changed"))
        object.__setattr__(self, "why", _string_or_empty(self.why, "why"))
        object.__setattr__(self, "scope", _string_tuple(self.scope, "scope"))
        object.__setattr__(self, "verification", _string_tuple(self.verification, "verification"))
        object.__setattr__(self, "gpt55_review_outcome", _string_or_empty(self.gpt55_review_outcome, "gpt55_review_outcome"))
        object.__setattr__(self, "gpt55_review_summary", _string_or_empty(self.gpt55_review_summary, "gpt55_review_summary"))
        for field_name in ("refactor_skill_applied", "gpt55_review_completed", "gpt55_review_findings_resolved"):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be boolean")
        explicit_issues = _string_tuple(self.issues, "issues")
        semantic_issues = _semantic_issues(
            what_changed=self.what_changed,
            why=self.why,
            scope=self.scope,
            verification=self.verification,
            refactor_skill_applied=self.refactor_skill_applied,
            gpt55_review_completed=self.gpt55_review_completed,
            gpt55_review_findings_resolved=self.gpt55_review_findings_resolved,
            gpt55_review_outcome=self.gpt55_review_outcome,
            gpt55_review_summary=self.gpt55_review_summary,
        )
        object.__setattr__(self, "issues", _dedupe_strings((*explicit_issues, *semantic_issues)))


def evaluate_pr_evidence(
    *,
    what_changed: Sequence[str],
    why: str,
    scope: Sequence[str],
    verification: Sequence[str],
    refactor_skill_applied: bool,
    gpt55_review_completed: bool,
    gpt55_review_findings_resolved: bool,
    gpt55_review_outcome: str,
    gpt55_review_summary: str,
) -> PullRequestEvidence:
    return PullRequestEvidence(
        what_changed=tuple(what_changed),
        why=why,
        scope=tuple(scope),
        verification=tuple(verification),
        refactor_skill_applied=refactor_skill_applied,
        gpt55_review_completed=gpt55_review_completed,
        gpt55_review_findings_resolved=gpt55_review_findings_resolved,
        gpt55_review_outcome=gpt55_review_outcome,
        gpt55_review_summary=gpt55_review_summary,
    )


def evaluate_pr_evidence_record(record: Mapping[str, Any]) -> PullRequestEvidence:
    if record.get("record_type") != PR_EVIDENCE_RECORD_TYPE:
        return PullRequestEvidence(
            what_changed=(),
            why="",
            scope=(),
            verification=(),
            refactor_skill_applied=False,
            gpt55_review_completed=False,
            gpt55_review_findings_resolved=False,
            gpt55_review_outcome="",
            gpt55_review_summary="",
            issues=(f"record_type must be {PR_EVIDENCE_RECORD_TYPE!r}",),
        )
    parsing_issues: list[str] = []
    what_changed = _record_string_sequence(record, "what_changed", parsing_issues)
    why = _record_string(record, "why", parsing_issues)
    scope = _record_string_sequence(record, "scope", parsing_issues)
    verification = _record_string_sequence(record, "verification", parsing_issues)
    refactor_skill_applied = _record_bool(record, "refactor_skill_applied", parsing_issues)
    gpt55_review_completed = _record_bool(record, "gpt55_review_completed", parsing_issues)
    gpt55_review_findings_resolved = _record_bool(record, "gpt55_review_findings_resolved", parsing_issues)
    gpt55_review_outcome = _record_string(record, "gpt55_review_outcome", parsing_issues)
    gpt55_review_summary = _record_string(record, "gpt55_review_summary", parsing_issues)
    return PullRequestEvidence(
        what_changed=what_changed,
        why=why,
        scope=scope,
        verification=verification,
        refactor_skill_applied=refactor_skill_applied,
        gpt55_review_completed=gpt55_review_completed,
        gpt55_review_findings_resolved=gpt55_review_findings_resolved,
        gpt55_review_outcome=gpt55_review_outcome,
        gpt55_review_summary=gpt55_review_summary,
        issues=tuple(parsing_issues),
    )


def evaluate_pr_evidence_file(path: str | Path) -> PullRequestEvidence:
    evidence_path = local_path(str(path))
    try:
        record = json.loads(evidence_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        return _failed_pr_evidence(f"{evidence_path} must contain valid UTF-8 JSON: {exc.reason}")
    except json.JSONDecodeError as exc:
        return _failed_pr_evidence(f"{evidence_path} must contain valid JSON: {exc.msg}")
    except OSError as exc:
        return _failed_pr_evidence(f"{evidence_path} could not be read: {exc}")
    if not isinstance(record, Mapping):
        return _failed_pr_evidence(f"{evidence_path} must contain a JSON object")
    return evaluate_pr_evidence_record(record)


def evaluate_pr_evidence_directory(
    directory: str | Path,
    *,
    pattern: str = "*.json",
    recursive: bool = True,
) -> dict[str, PullRequestEvidence]:
    evidence_dir = local_path(str(directory))
    if not evidence_dir.is_dir():
        return {str(evidence_dir): _failed_pr_evidence(f"{evidence_dir} must be a directory")}

    file_iter = evidence_dir.rglob(pattern) if recursive else evidence_dir.glob(pattern)
    evidence_files = [
        path
        for path in sorted(file_iter)
        if path.is_file() and not _is_pr_evidence_validation_record(path)
    ]
    if not evidence_files:
        return {str(evidence_dir): _failed_pr_evidence(f"{evidence_dir} has no PR evidence JSON files matching {pattern!r}")}
    return {
        str(path.relative_to(evidence_dir)): evaluate_pr_evidence_file(path)
        for path in evidence_files
    }


def pr_evidence_validation_to_record(evidence_by_file: Mapping[str, PullRequestEvidence]) -> dict[str, Any]:
    return {
        "record_type": PR_EVIDENCE_VALIDATION_RECORD_TYPE,
        "ok": bool(evidence_by_file) and all(evidence.ok for evidence in evidence_by_file.values()),
        "files": {
            path: pr_evidence_to_record(evidence)
            for path, evidence in evidence_by_file.items()
        },
    }


def pr_evidence_to_record(evidence: PullRequestEvidence) -> dict[str, Any]:
    return {
        "record_type": PR_EVIDENCE_RECORD_TYPE,
        "ok": evidence.ok,
        "what_changed": list(evidence.what_changed),
        "why": evidence.why,
        "scope": list(evidence.scope),
        "verification": list(evidence.verification),
        "refactor_skill_applied": evidence.refactor_skill_applied,
        "gpt55_review_completed": evidence.gpt55_review_completed,
        "gpt55_review_findings_resolved": evidence.gpt55_review_findings_resolved,
        "gpt55_review_outcome": evidence.gpt55_review_outcome,
        "gpt55_review_summary": evidence.gpt55_review_summary,
        "issues": list(evidence.issues),
    }


def write_pr_evidence_json(evidence: PullRequestEvidence, path: str | Path) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pr_evidence_to_record(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_record(record: Mapping[str, Any], path: str | Path) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _failed_pr_evidence(*issues: str) -> PullRequestEvidence:
    return PullRequestEvidence(
        what_changed=(),
        why="",
        scope=(),
        verification=(),
        refactor_skill_applied=False,
        gpt55_review_completed=False,
        gpt55_review_findings_resolved=False,
        gpt55_review_outcome="",
        gpt55_review_summary="",
        issues=issues,
    )


def _is_pr_evidence_validation_record(path: Path) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(record, Mapping) or record.get("record_type") != PR_EVIDENCE_VALIDATION_RECORD_TYPE:
        return False
    files = record.get("files")
    if record.get("ok") is not True or not isinstance(files, Mapping) or not files:
        return False
    return all(
        isinstance(path, str)
        and isinstance(evidence_record, Mapping)
        and evaluate_pr_evidence_record(evidence_record).ok
        for path, evidence_record in files.items()
    )


def _record_string_sequence(record: Mapping[str, Any], field_name: str, issues: list[str]) -> tuple[str, ...]:
    value = record.get(field_name, ())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized = []
        for index, item in enumerate(value):
            if isinstance(item, str) and item.strip():
                normalized.append(item.strip())
            else:
                issues.append(f"{field_name}[{index}] must be a non-empty string")
        return tuple(normalized)
    issues.append(f"{field_name} must be a sequence")
    return ()


def _record_string(record: Mapping[str, Any], field_name: str, issues: list[str]) -> str:
    value = record.get(field_name, "")
    if isinstance(value, str):
        return value
    issues.append(f"{field_name} must be a string")
    return ""


def _record_bool(record: Mapping[str, Any], field_name: str, issues: list[str]) -> bool:
    value = record.get(field_name, False)
    if type(value) is bool:
        return value
    issues.append(f"{field_name} must be boolean")
    return False


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of non-empty strings")
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings")
        normalized.append(value.strip())
    return tuple(normalized)


def _string_or_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value.strip()


def _semantic_issues(
    *,
    what_changed: tuple[str, ...],
    why: str,
    scope: tuple[str, ...],
    verification: tuple[str, ...],
    refactor_skill_applied: bool,
    gpt55_review_completed: bool,
    gpt55_review_findings_resolved: bool,
    gpt55_review_outcome: str,
    gpt55_review_summary: str,
) -> tuple[str, ...]:
    issues: list[str] = []
    if not what_changed:
        issues.append("what_changed must include at least one item")
    if not why:
        issues.append("why must be non-empty")
    if not scope:
        issues.append("scope must include at least one touched boundary")
    if not verification:
        issues.append("verification must include tests, builds, benchmarks, or an explicit not-applicable note")
    if not refactor_skill_applied:
        issues.append("Refactor skill must be applied during the PR slice")
    if not gpt55_review_completed:
        issues.append("GPT-5.5 review must be completed")
    if gpt55_review_outcome not in GPT55_REVIEW_OUTCOMES:
        issues.append("gpt55_review_outcome must be 'clean' or 'findings_resolved'")
    if gpt55_review_outcome == "findings_resolved" and not gpt55_review_findings_resolved:
        issues.append("GPT-5.5 findings must be resolved when gpt55_review_outcome is 'findings_resolved'")
    if not gpt55_review_summary:
        issues.append("gpt55_review_summary must describe findings and fixes, or state that the review was clean")
    return tuple(issues)


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _evaluate_pr_evidence_inputs(
    *,
    json_paths: Sequence[str | Path],
    directories: Sequence[str | Path],
) -> dict[str, PullRequestEvidence]:
    evidence_by_file: dict[str, PullRequestEvidence] = {}
    for json_path in json_paths:
        evidence_by_file[str(json_path)] = evaluate_pr_evidence_file(json_path)
    for directory in directories:
        directory_path = local_path(str(directory))
        directory_key = str(directory).rstrip("/")
        for relative_path, evidence in evaluate_pr_evidence_directory(directory).items():
            if relative_path == str(directory_path):
                evidence_by_file[relative_path] = evidence
            else:
                evidence_by_file[f"{directory_key}/{relative_path}"] = evidence
    return evidence_by_file


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit machine-checkable Document KV Cache pull-request evidence.")
    parser.add_argument("--what-changed", action="append", default=[])
    parser.add_argument("--why", default="")
    parser.add_argument("--scope", action="append", default=[])
    parser.add_argument("--verification", action="append", default=[])
    parser.add_argument("--refactor-skill-applied", action="store_true")
    parser.add_argument("--gpt55-review-completed", action="store_true")
    parser.add_argument("--gpt55-review-findings-resolved", action="store_true")
    parser.add_argument("--gpt55-review-outcome", default="", choices=("", *GPT55_REVIEW_OUTCOMES))
    parser.add_argument("--gpt55-review-summary", default="")
    parser.add_argument("--validate-json", action="append", default=[], metavar="PATH", help="Validate a PR evidence JSON sidecar.")
    parser.add_argument(
        "--validate-directory",
        action="append",
        default=[],
        metavar="DIR",
        help="Validate PR evidence JSON sidecars in a directory.",
    )
    parser.add_argument("--output-json", help="Write the PR evidence JSON to this path instead of stdout.")
    args = parser.parse_args(argv)

    try:
        if args.validate_json or args.validate_directory:
            evidence_by_file = _evaluate_pr_evidence_inputs(
                json_paths=tuple(args.validate_json),
                directories=tuple(args.validate_directory),
            )
            record = pr_evidence_validation_to_record(evidence_by_file)
            if args.output_json:
                _write_json_record(record, args.output_json)
            else:
                print(json.dumps(record, indent=2, sort_keys=True))
            return 0 if record["ok"] else 2

        evidence = evaluate_pr_evidence(
            what_changed=tuple(args.what_changed),
            why=args.why,
            scope=tuple(args.scope),
            verification=tuple(args.verification),
            refactor_skill_applied=args.refactor_skill_applied,
            gpt55_review_completed=args.gpt55_review_completed,
            gpt55_review_findings_resolved=args.gpt55_review_findings_resolved,
            gpt55_review_outcome=args.gpt55_review_outcome,
            gpt55_review_summary=args.gpt55_review_summary,
        )
        if args.output_json:
            write_pr_evidence_json(evidence, args.output_json)
        else:
            print(json.dumps(pr_evidence_to_record(evidence), indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0 if evidence.ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
