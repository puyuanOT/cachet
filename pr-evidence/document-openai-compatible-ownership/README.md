# Document OpenAI-Compatible Ownership

This PR-evidence sidecar covers the refactor slice that moves the
OpenAI-compatible completion benchmark engine from the legacy restaurant
package into `document_kv_cache.openai_compatible`.

The slice makes the document package the implementation owner for the
stdlib-only vLLM/SGLang OpenAI-compatible completion engine, prompt-text mode
selection, prompt-token accounting, streaming response parsing, and HTTP error
wrapping. `restaurant_kv_serving.openai_compatible` remains a compatibility
wrapper for older imports.

## Review

GPT-5.5 initially found one compatibility issue: callers that monkeypatched
`restaurant_kv_serving.openai_compatible.urlopen` would no longer intercept the
engine's HTTP call after the ownership move. The PR now preserves that migration
hook with a document-owned `_active_urlopen()` resolver and regression coverage
for both the legacy `urlopen` hook and the new document `_urlopen` hook.

The re-review found no issues. The only residual risk called out was intentional
precedence: if both hooks are patched simultaneously, the legacy hook wins so
existing callers keep their old behavior during migration.

## Verification

- `poetry run pytest tests/test_openai_compatible.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/openai_compatible.py src/restaurant_kv_serving/openai_compatible.py`
- `poetry run pytest -q`
- `poetry check`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry build`
- repository secret-pattern scan over `README.md CONTRIBUTING.md pyproject.toml src tests databricks pr-evidence .github`
