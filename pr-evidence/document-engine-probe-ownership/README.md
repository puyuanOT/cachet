# Document Engine Probe Ownership

This PR-evidence sidecar covers the refactor slice that moves the native engine
KV-connector probe runner into `document_kv_cache.engine_probe`.

The slice makes the document package the implementation owner for handoff
loading, payload URI validation, native probe factory loading, probe evidence
generation, metadata provenance, and the CLI entry point. The legacy
`restaurant_kv_serving.engine_probe` module remains as a compatibility wrapper.

## Review

GPT-5.5 reviewed the ownership inversion and found two import-order isolation
issues in the legacy wrapper:

- public document monkeypatches installed before legacy import could become the
  legacy wrapper's default functions;
- public document metadata-constant patches installed before legacy import could
  leak into legacy provenance metadata.

The wrapper now loads a private pristine copy of the document implementation for
legacy defaults and explicitly takes legacy metadata constants from that pristine
copy. Legacy calls still overlay legacy monkeypatches after import, and
factory-result checks keep accepting document-owned `EngineKVProbeFactoryResult`
objects. GPT-5.5 re-reviewed the fixes and approved the branch.

## Verification

- `poetry run pytest tests/test_engine_probe.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/engine_probe.py src/restaurant_kv_serving/engine_probe.py src/document_kv_cache/__init__.py tests/test_engine_probe.py tests/test_public_package.py tests/test_project_governance.py`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check`
- `poetry build`
- `PYTHONPATH=src poetry run python -m document_kv_cache.pr_evidence --validate-directory pr-evidence`
- repository secret scan over README, package metadata, source, tests,
  Databricks helpers, PR evidence, and GitHub metadata
- GPT-5.5 review and re-review after resolving legacy import-order isolation
  findings
