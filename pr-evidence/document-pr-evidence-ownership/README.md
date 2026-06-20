# Document PR Evidence Ownership

## What Changed

- Moved the PR evidence implementation into `document_kv_cache.pr_evidence`.
- Replaced `restaurant_kv_serving.pr_evidence` with a compatibility facade that preserves the legacy star-import surface and routes legacy CLI hooks through `LegacyMainBridge`.
- Added regression tests for document-owned module identity, legacy star-import compatibility, and legacy CLI monkeypatch behavior.

## Why

PR traceability is part of the package governance surface, not restaurant-specific serving logic. Owning it under `document_kv_cache` keeps new users on the document-generic namespace while preserving legacy imports and console scripts during migration.

## Verification

- `poetry run pytest tests/test_pr_evidence.py tests/test_public_package.py tests/test_project_governance.py tests/test_release_bundle.py tests/test_benchmark_plan.py`
- `poetry run pytest`
- `poetry check`
- `poetry build`
- `git diff --check`
- Repository secret scan over the documented source, test, Databricks, evidence, and GitHub workflow paths; only the expected fake LangSmith governance fixtures were reported.

## Review

GPT-5.5 reviewer Volta initially requested changes for a legacy CLI monkeypatch regression. The facade now uses `LegacyMainBridge` for legacy `main`, with tests covering generation, validation, and hook restoration after errors. Volta re-reviewed and approved the corrected diff.
