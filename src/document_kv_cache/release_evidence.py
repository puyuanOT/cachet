"""Public document namespace for release evidence validation."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.release_evidence",
    (
        "RELEASE_EVIDENCE_RECORD_TYPE",
        "RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE",
        "RELEASE_EVIDENCE_ARTIFACT_ROLES",
        "REQUIRED_ENGINE_PROBE_BACKENDS",
        "ReleaseEvidenceArtifactSource",
        "ReleaseEvidence",
        "ReleaseEvidenceInputFileStatus",
        "ReleaseEvidenceInputStatus",
        "evaluate_release_evidence",
        "evaluate_release_evidence_files",
        "inspect_release_evidence_input_files",
        "release_evidence_input_status_to_record",
        "release_evidence_to_record",
        "write_release_evidence_input_status_json",
        "write_release_evidence_json",
    ),
    globals(),
)

_main_bridge = LegacyMainBridge(
    public_namespace=globals(),
    legacy_module_name="restaurant_kv_serving.release_evidence",
    hook_names=(
        "evaluate_release_evidence_files",
        "inspect_release_evidence_input_files",
        "release_evidence_input_status_to_record",
        "release_evidence_to_record",
        "write_release_evidence_input_status_json",
        "write_release_evidence_json",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


__all__.append("main")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


del LegacyMainBridge, reexport_public
