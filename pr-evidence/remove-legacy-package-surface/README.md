# Remove Legacy Package Surface Evidence

This directory contains the pull-request evidence sidecar for removing the
legacy restaurant compatibility facade from built `cachet-kv` wheels while
keeping source-only migration tests.

Verification:

- `poetry run pytest tests/test_public_package.py tests/test_project_governance.py tests/test_release_bundle.py -q`
- `poetry check && poetry check --lock`
- `python -m compileall -q src/document_kv_cache/__init__.py src/document_kv_cache/release_bundle.py && git diff --check`
- built and inspected `cachet_kv-0.2.0-py3-none-any.whl`
- installed-wheel smoke for `cachet` and `document_kv_cache` with the legacy
  package absent
- `poetry run pytest -q`
- GPT-5.5 focused review with findings resolved
