# Workflow Storage Factory

This PR adds `DocumentKVWorkflow.with_storage(...)`, a convenience constructor
for the standard Memory, Disk, and Unity Catalog reader stack plus CPU/local
cache tiers.

## Verification

- `poetry run pytest tests/test_workflow.py tests/test_project_governance.py tests/test_kvpack.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry install --dry-run`
- `poetry build`

## Review

GPT-5.5 initially found that relative `shard_uri` values did not round-trip
through configured disk or UC roots. The writer now uses the same storage-root
resolution contract as routed reads, regression tests cover disk and UC roots,
and GPT-5.5 approved the updated branch.
