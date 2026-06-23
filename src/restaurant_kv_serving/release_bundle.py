"""Compatibility facade for :mod:`document_kv_cache.release_bundle`."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_kv_cache.benchmark_plan_executor import (
    BENCHMARK_PLAN_EXECUTION_RECORD_TYPE,
    BENCHMARK_PLAN_SOURCE_RECORD_TYPE,
)
from document_kv_cache.databricks_runs import (
    DATABRICKS_RUN_STATUS_RECORD_TYPE,
    DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE,
)
from document_kv_cache.pr_evidence import PR_EVIDENCE_RECORD_TYPE, evaluate_pr_evidence_record
from document_kv_cache.release_bundle import (
    RELEASE_BUNDLE_ARTIFACT_ROLES,
    RELEASE_BUNDLE_MANIFEST_FILENAME,
    RELEASE_BUNDLE_PACKAGE_NAME,
    RELEASE_BUNDLE_RECORD_TYPE,
    STRICT_V1_RELEASE_HELP as _STRICT_V1_RELEASE_HELP,
    ReleaseBundle,
    ReleaseBundleArtifact,
    build_release_bundle,
    release_bundle_to_record,
)
from document_kv_cache.release_evidence import (
    RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE,
    RELEASE_EVIDENCE_RECORD_TYPE,
    REQUIRED_ENGINE_PROBE_BACKENDS,
    evaluate_release_evidence,
)
from document_kv_cache.storage import local_path


def write_release_bundle_manifest_json(bundle: ReleaseBundle, path: str | Path) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(release_bundle_to_record(bundle), indent=2, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a checksummed Document KV release evidence bundle.")
    parser.add_argument("--v1-benchmark-json", required=True)
    parser.add_argument("--compatibility-benchmark-json", action="append", default=[])
    parser.add_argument("--storage-benchmark-json", required=True)
    parser.add_argument("--engine-probe-json", action="append", default=[])
    parser.add_argument("--engine-actions-json", action="append", default=[])
    parser.add_argument("--release-evidence-json")
    parser.add_argument("--preflight-json")
    parser.add_argument("--plan-execution-json", action="append", default=[])
    parser.add_argument("--databricks-run-status-json", action="append", default=[])
    parser.add_argument("--package-wheel")
    parser.add_argument("--pr-evidence-json", action="append", default=[])
    parser.add_argument("--requirements-matrix-md")
    parser.add_argument("--github-governance-json")
    parser.add_argument("--repository-hygiene-json")
    parser.add_argument("--native-probe-factories-json", action="append", default=[])
    parser.add_argument(
        "--require-complete-v1",
        action="store_true",
        help=_STRICT_V1_RELEASE_HELP,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    bundle = build_release_bundle(
        v1_benchmark_json=args.v1_benchmark_json,
        storage_benchmark_json=args.storage_benchmark_json,
        compatibility_benchmark_jsons=args.compatibility_benchmark_json,
        engine_probe_jsons=args.engine_probe_json,
        engine_actions_jsons=args.engine_actions_json,
        release_evidence_json=args.release_evidence_json,
        preflight_json=args.preflight_json,
        plan_execution_jsons=args.plan_execution_json,
        databricks_run_status_jsons=args.databricks_run_status_json,
        package_wheel=args.package_wheel,
        pr_evidence_jsons=args.pr_evidence_json,
        requirements_matrix_md=args.requirements_matrix_md,
        github_governance_json=args.github_governance_json,
        repository_hygiene_json=args.repository_hygiene_json,
        native_probe_factories_jsons=args.native_probe_factories_json,
        require_complete_v1=args.require_complete_v1,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    if args.output_json:
        write_release_bundle_manifest_json(bundle, args.output_json)
    else:
        print(json.dumps(release_bundle_to_record(bundle), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
