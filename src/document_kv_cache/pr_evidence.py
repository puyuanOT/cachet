"""Public document namespace for pull-request traceability evidence."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.pr_evidence",
    (
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
    ),
    globals(),
)

_main_bridge = LegacyMainBridge(
    public_namespace=globals(),
    legacy_module_name="restaurant_kv_serving.pr_evidence",
    hook_names=(
        "evaluate_pr_evidence",
        "evaluate_pr_evidence_directory",
        "evaluate_pr_evidence_file",
        "evaluate_pr_evidence_record",
        "pr_evidence_validation_to_record",
        "pr_evidence_to_record",
        "write_pr_evidence_json",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


__all__.append("main")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


del LegacyMainBridge, reexport_public
