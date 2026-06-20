"""Compatibility facade for :mod:`document_kv_cache.pr_evidence`."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_kv_cache._reexport import LegacyMainBridge
from document_kv_cache.pr_evidence import (
    GPT55_REVIEW_OUTCOMES,
    PR_EVIDENCE_RECORD_TYPE,
    PR_EVIDENCE_VALIDATION_RECORD_TYPE,
    PullRequestEvidence,
    _dedupe_strings,
    _evaluate_pr_evidence_inputs,
    _failed_pr_evidence,
    _is_pr_evidence_validation_record,
    _record_bool,
    _record_string,
    _record_string_sequence,
    _semantic_issues,
    _string_or_empty,
    _string_tuple,
    _write_json_record,
    evaluate_pr_evidence,
    evaluate_pr_evidence_directory,
    evaluate_pr_evidence_file,
    evaluate_pr_evidence_record,
    pr_evidence_to_record,
    pr_evidence_validation_to_record,
    write_pr_evidence_json,
)
from document_kv_cache.storage import local_path

_main_bridge = LegacyMainBridge(
    legacy_module_name="document_kv_cache.pr_evidence",
    public_namespace=globals(),
    hook_names=(
        "evaluate_pr_evidence",
        "evaluate_pr_evidence_directory",
        "evaluate_pr_evidence_file",
        "evaluate_pr_evidence_record",
        "pr_evidence_validation_to_record",
        "pr_evidence_to_record",
        "write_pr_evidence_json",
        "local_path",
        "_write_json_record",
        "_failed_pr_evidence",
        "_is_pr_evidence_validation_record",
        "_record_string_sequence",
        "_record_string",
        "_record_bool",
        "_string_tuple",
        "_string_or_empty",
        "_semantic_issues",
        "_dedupe_strings",
        "_evaluate_pr_evidence_inputs",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


del LegacyMainBridge
