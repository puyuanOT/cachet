"""Compatibility facade for :mod:`document_kv_cache.release_evidence`."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_kv_cache.benchmark_runner import BENCHMARK_RUN_RECORD_TYPE
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    CACHE_REUSE_ARM,
    DEFAULT_HARDWARE_TARGET,
    DEFAULT_V1_MODEL_ID,
    SUPPORTED_V1_DATASETS,
)
from document_kv_cache.engine_adapters import ServingBackend, validate_engine_kv_connector_probe_record
from document_kv_cache.engine_probe import (
    ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE,
    ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION,
)
from document_kv_cache.engine_protocol import (
    AttentionMechanism,
    KVLayout,
    KVStorageLayout,
    dtype_byte_width,
    kv_storage_layout_from_value,
)
from document_kv_cache.model_profiles import get_model_profile
from document_kv_cache.release_evidence import (
    RELEASE_EVIDENCE_ARTIFACT_ROLES,
    RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE,
    RELEASE_EVIDENCE_RECORD_TYPE,
    REQUIRED_ENGINE_PROBE_BACKENDS,
    ReleaseEvidence,
    ReleaseEvidenceArtifactSource,
    ReleaseEvidenceInputFileStatus,
    ReleaseEvidenceInputStatus,
    evaluate_release_evidence,
    evaluate_release_evidence_files,
    inspect_release_evidence_input_files,
    release_evidence_input_status_to_record,
    release_evidence_to_record,
    write_release_evidence_input_status_json,
    write_release_evidence_json,
)
from document_kv_cache.serving_env import serving_environment_profile
from document_kv_cache.storage import is_real_uc_volume_root, local_path
from document_kv_cache.storage_benchmark import RELEASE_STORAGE_BENCHMARK_READERS, STORAGE_BENCHMARK_RECORD_TYPE


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Document KV Cache release evidence JSON artifacts.")
    parser.add_argument("--v1-benchmark-json", required=True)
    parser.add_argument("--storage-benchmark-json", required=True)
    parser.add_argument("--engine-probe-json", action="append", default=[])
    parser.add_argument("--engine-actions-json", action="append", default=[])
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
                engine_actions_jsons=tuple(args.engine_actions_json),
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
            engine_actions_jsons=tuple(args.engine_actions_json),
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
