# Databricks Engine Probe Actions Template

This PR makes the standalone Databricks engine-probe Asset Bundle produce the
same connector-actions sidecar required by release evidence. The repo and
packaged bundle templates now expose `actions_output_json` and pass it to the
native probe runner as `--actions-output-json`.

Verification:

- `poetry run pytest tests/test_databricks_job.py tests/test_template_resources.py -q`
- `poetry run pytest tests/test_databricks_engine_probe_job.py tests/test_benchmark_plan.py -q`
- `python -m compileall -q src/document_kv_cache src/restaurant_kv_serving`
- `git diff --check`
- `poetry run pytest -q`

GPT-5.5 reviewed the diff, ran the focused bundle-template regression, and
approved it with no findings.
