"""Public document namespace for release evidence bundle packaging."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.release_bundle",
    (
        "RELEASE_BUNDLE_RECORD_TYPE",
        "RELEASE_BUNDLE_MANIFEST_FILENAME",
        "RELEASE_BUNDLE_ARTIFACT_ROLES",
        "ReleaseBundleArtifact",
        "ReleaseBundle",
        "build_release_bundle",
        "release_bundle_to_record",
        "write_release_bundle_manifest_json",
    ),
    globals(),
)

_main_bridge = LegacyMainBridge(
    public_namespace=globals(),
    legacy_module_name="restaurant_kv_serving.release_bundle",
    hook_names=(
        "build_release_bundle",
        "release_bundle_to_record",
        "write_release_bundle_manifest_json",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


__all__.append("main")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


del LegacyMainBridge, reexport_public
