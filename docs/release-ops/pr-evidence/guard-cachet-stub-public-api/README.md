# Guard Cachet Stub Public API Evidence

This evidence records the test-only guard that keeps the Cachet typing facade
aligned with the canonical `document_kv_cache` public API.

The PR verifies that `src/cachet/__init__.pyi` exports every
`document_kv_cache.__all__` name through explicit self-alias imports and that
each imported symbol exists on its declared source module.
