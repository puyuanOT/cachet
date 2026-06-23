# Legacy Compatibility Removal Evidence

Cachet has one repository and one distribution package: `puyuanOT/cachet`
publishes the Cachet-branded `cachet-kv` wheel. The primary import surface is
the branded `cachet` facade, and `document_kv_cache` remains the canonical
implementation namespace used by existing evidence files and explicit document
KV-cache CLIs.

The `restaurant_kv_serving` package and `restaurant-kv-*` console scripts were
migration-only compatibility shims. They are no longer part of built
`cachet-kv` wheels or the source tree. New code must not add production
dependencies on the legacy package or legacy CLI aliases.

## Completed Compatibility Contract

The legacy facade has been removed from the built release package surface and
from `src/`. The removal stays release-gated by the evidence below so it does
not silently regress.

- `pyproject.toml` no longer packages `restaurant_kv_serving`, its `py.typed`
  marker, or `restaurant-kv-*` console scripts.
- `src/document_kv_cache/release_bundle.py` rejects
  `restaurant_kv_serving/__init__.py` import marker, the
  `restaurant_kv_serving/py.typed` marker, and every legacy console-script
  entry point in tested wheels.
- `tests/test_public_package.py` proves the public package surface and
  `pyproject.toml` metadata stay Cachet/document-owned with no legacy package
  or legacy console-script targets.
- `tests/test_project_governance.py` prevents new accidental
  `restaurant_kv_serving` dependencies from spreading.
- Current downstream migration evidence is tracked in
  `evidence/legacy-migration/current/legacy-migration-evidence.json`.
  It is generated from
  `evidence/legacy-migration/current/legacy-migration-scan-config.json`
  and validates with no legacy references across the `release`, `benchmark`,
  `storage`, `native_probe`, and `smoke` runner categories.
- Release PRs continue to include PR evidence sidecars with Refactor-skill
  evidence and completed GPT-5.5 review before the strict release bundle can use
  those changes.

## Removal Proof

The source-only `restaurant_kv_serving` directory must stay absent. The removal
is backed by these checks:

- Downstream Databricks benchmark runners and QA jobs have migrated from
  `restaurant_kv_serving` imports and `restaurant-kv-*` commands to `cachet`
  imports, `cachet-*` commands, or explicit `document_kv_cache` /
  `document-kv-*` names.
- Keep the current migration evidence artifact with record type
  `document_kv.legacy_compatibility_migration.v1` validates with
  `python -m document_kv_cache.legacy_compatibility --validate-json`. The
  artifact must cover `release`, `benchmark`, `storage`, `native_probe`, and
  `smoke` downstream job categories, and it must confirm no checked runner uses
  `restaurant_kv_serving` imports or `restaurant-kv-*` commands.
- Prefer generating the migration evidence from a scan config instead of
  hand-authoring it:

  ```bash
  python -m document_kv_cache.legacy_compatibility \
    --scan-config-json path/to/legacy-migration-scan-config.json \
    --output-json path/to/legacy-migration-evidence.json
  ```

  Scan configs use record type
  `document_kv.legacy_compatibility_scan_config.v1`, list
  `checked_downstream_jobs` with `checked_paths`, and reuse the same
  `release_evidence` entries that will be bundled for removal. The generated
  migration evidence keeps the existing
  `document_kv.legacy_compatibility_migration.v1` record type and adds optional
  per-job scan provenance through `checked_paths` and empty
  `legacy_reference_hits`.
- Current AWS g6/L4 release evidence and optional AWS g5/A10G compatibility
  evidence were generated from migrated runners or otherwise prove that the
  bundled Cachet wheel no longer needs the legacy facade for those paths.
- The strict release-bundle package-wheel gates are updated and reject legacy
  import markers, `py.typed` markers, and legacy console scripts in tested
  wheels.
- Documentation and public-package tests are updated in the same PR, so no
  release doc still promises a bundled legacy facade after it is removed.

## Regression Checklist

Future package-surface changes must keep this contract intact:

- Keep `restaurant_kv_serving` absent from `pyproject.toml` package metadata,
  package includes, and `restaurant-kv-*` scripts.
- Keep `src/restaurant_kv_serving` absent.
- Keep `src/document_kv_cache/release_bundle.py` rejecting
  `restaurant_kv_serving/__init__.py`, `restaurant_kv_serving/py.typed`, and
  legacy console-script entry points in tested wheels.
- Attach a PR evidence sidecar that records what changed, why removal is now
  safe, the downstream migration evidence, Refactor-skill usage, GPT-5.5 review,
  and verification.
- Bundle the migration evidence through the optional release-bundle
  `legacy_migration_evidence` artifact role when the removal changes the
  package surface.
- Run the focused governance, public-package, and release-bundle tests, then run
  the full test suite before merging.
- Refresh the tested wheel and strict release bundle before publication if a
  package-surface change affects the package artifact used for release
  evidence.
