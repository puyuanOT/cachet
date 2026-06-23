# Current Benchmark Evidence Refresh

This folder contains PR evidence for refreshing the release-readiness docs to
the current `cachet-kv` g6/L4 and g5/A10G Databricks benchmark runs.

The PR records the fresh benchmark evidence and release-evidence validation
without claiming that the complete strict release bundle has already been
rebuilt. The requirements matrix now distinguishes generated, validated
benchmark evidence from evidence that has been carried into a refreshed strict
bundle.

Verification:

- `git diff --check`
- `poetry run pytest tests/test_project_governance.py -q`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry check`
- `poetry check --lock`
- `poetry run pytest -q`
- release evidence validation over the fresh g6 and g5 benchmark artifacts
- GPT-5.5 focused review with findings resolved
