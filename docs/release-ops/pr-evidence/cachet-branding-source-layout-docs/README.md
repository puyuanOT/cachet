# Cachet Branding And Source Layout Docs

This evidence covers the PR that introduces Cachet as the product brand while
keeping the Python distribution name `document-kv-cache` and import namespace
`document_kv_cache`.

The PR also updates `src/README.md` so the source tree documentation reflects
the current ownership model: `document_kv_cache/` is the canonical
implementation package, and `restaurant_kv_serving/` is migration-only
compatibility.

Verification:

- `poetry run pytest tests/test_project_governance.py -q`
- `poetry run pytest`
- `poetry check`
- `poetry build`
- `git diff --check`
- repository secret scan
- GPT-5.5 focused review

