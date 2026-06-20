# Release Bundle GitHub Governance Sidecar

This evidence covers the PR that teaches release bundles to accept and validate
the GitHub repository governance sidecar emitted by
`document_kv_cache.github_governance`.

Verification performed locally:

- `poetry run pytest`
- `poetry check`
- `git diff --check`
- `poetry build`
- Secret scan; only existing detector fixtures and prior evidence text matched,
  no live credentials found.

The GPT-5.5 review result is recorded in `pr-evidence.json` before merge.
