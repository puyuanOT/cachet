# Legacy Compatibility Removal Gate

Cachet has one repository and one distribution package: `puyuanOT/cachet`
publishes the Cachet-branded `cachet-kv` wheel. The primary import surface is
the branded `cachet` facade, and `document_kv_cache` remains the canonical
implementation namespace used by existing evidence files and explicit document
KV-cache CLIs.

The `restaurant_kv_serving` package and `restaurant-kv-*` console scripts were
migration-only compatibility shims. They are no longer part of built
`cachet-kv` wheels. Source files remain temporarily for migration-history tests
and older local imports. New code must not add production dependencies on the
legacy package or legacy CLI aliases.

## Current Compatibility Contract

The legacy facade has been removed from the built release package surface. Its
remaining source-only cleanup stays release-gated until a dedicated deletion PR
proves no local test, documentation, or downstream runner still needs the
source shim.

- `pyproject.toml` no longer packages `restaurant_kv_serving`, its `py.typed`
  marker, or `restaurant-kv-*` console scripts.
- `src/document_kv_cache/release_bundle.py` rejects
  `restaurant_kv_serving/__init__.py` import marker, the
  `restaurant_kv_serving/py.typed` marker, and every legacy console-script
  entry point in tested wheels.
- `tests/test_public_package.py` keeps source-only compatibility checks for
  older local callers while installed-wheel CI proves the facade is absent.
- `tests/test_project_governance.py` keeps legacy references scoped to
  compatibility tests and prevents new accidental `restaurant_kv_serving`
  dependencies from spreading.
- Release PRs continue to include PR evidence sidecars with Refactor-skill
  evidence and completed GPT-5.5 review before the strict release bundle can use
  those changes.

## Removal Blockers

Do not delete the remaining source-only `restaurant_kv_serving` directory until
all of these blockers are closed:

- Downstream Databricks benchmark runners and QA jobs have migrated from
  `restaurant_kv_serving` imports and `restaurant-kv-*` commands to `cachet`
  imports, `cachet-*` commands, or explicit `document_kv_cache` /
  `document-kv-*` names.
- A current migration evidence artifact with record type
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

## Removal PR Checklist

The removal must be a normal reviewed pull request, not a direct push to
`main`.

- Keep `restaurant_kv_serving` absent from `pyproject.toml` package metadata,
  package includes, and `restaurant-kv-*` scripts.
- Delete `src/restaurant_kv_serving` after the downstream migration evidence is
  attached and source-only tests have been migrated or removed.
- Keep `src/document_kv_cache/release_bundle.py` rejecting
  `restaurant_kv_serving/__init__.py`, `restaurant_kv_serving/py.typed`, and
  legacy console-script entry points in tested wheels.
- Update `tests/test_public_package.py` and `tests/test_project_governance.py`
  to remove or replace legacy compatibility assertions and allowlists.
- Update `README.md`, `docs/v1-requirements-matrix.md`, `src/README.md`, and
  this document so the public docs describe the post-migration package surface.
- Attach a PR evidence sidecar that records what changed, why removal is now
  safe, the downstream migration evidence, Refactor-skill usage, GPT-5.5 review,
  and verification.
- Bundle the migration evidence through the optional release-bundle
  `legacy_migration_evidence` artifact role when the removal changes the
  package surface.
- Run the focused governance, public-package, and release-bundle tests, then run
  the full test suite before merging.
- Refresh the tested wheel and strict release bundle before publication if the
  removal changes the package artifact used for release evidence.
