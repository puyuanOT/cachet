# Engine Probe Actions Sidecar PR Evidence

## What Changed

- Added an optional `actions_output_json` / `--actions-output-json` path to the
  engine KV probe runner so successful native probe runs can emit the validated
  `document_kv.engine_kv_connector_actions.v1` reserve/copy/bind/release
  descriptor next to the probe summary.
- Wired the optional sidecar through Databricks single-target and matrix probe
  job payload generation, including target JSON parsing aliases.
- Preserved static Databricks Asset Bundle compatibility by keeping the bundle
  template's required variable set unchanged.
- Exposed the sidecar writer through the document package and legacy
  compatibility wrapper, with regression tests for CLI, Databricks payload,
  legacy config pickle, public exports, and failure behavior.

## Verification

- `poetry run pytest tests/test_engine_probe.py tests/test_databricks_engine_probe_job.py tests/test_databricks_job.py tests/test_public_package.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry build`

## GPT-5.5 Review

The GPT-5.5 reviewer initially requested two fixes: keep the static Databricks
Asset Bundle template backwards-compatible, and write the sidecar only after the
native probe succeeds. Both were patched, covered by tests, and approved on
re-review.
