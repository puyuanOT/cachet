# Document Live Server Ownership

This PR-evidence sidecar covers the refactor slice that moves the live
OpenAI-compatible smoke-check implementation into
`document_kv_cache.live_server`.

The slice makes the document package the implementation owner for synthetic
live-check request construction, result serialization, OpenAI-compatible engine
selection, and the `document_kv_cache.live_server` CLI. The legacy
`restaurant_kv_serving.live_server` module remains a compatibility wrapper.

## Review

GPT-5.5 initially found one compatibility issue: legacy callers that monkeypatch
implementation globals such as `OpenAICompatibleCompletionEngine` would no
longer affect `restaurant_kv_serving.live_server.main()` after the move. The
wrapper now bridges all historical compatibility exports except `main` into the
document module while legacy `main()` runs, then restores the document module.

The re-review found no issues. The residual risk is the existing bridge pattern:
legacy `main()` temporarily mutates document-module globals during CLI
execution, which is acceptable for migration compatibility.

## Verification

- `poetry run pytest tests/test_live_server.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/live_server.py src/restaurant_kv_serving/live_server.py tests/test_live_server.py tests/test_public_package.py`
- `poetry run pytest -q`
- `poetry check`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- GPT-5.5 re-review after resolving the legacy engine-class monkeypatch finding
