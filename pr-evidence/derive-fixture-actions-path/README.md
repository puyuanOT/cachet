# Derive Fixture Actions Path PR Evidence

PR: https://github.com/puyuanOT/document-kv-cache/pull/297

## What Changed

- Fixture-backed planned engine probes derive `actions_output_json` from `qwen3-v1-fixture.actions.json` when omitted.
- Direct benchmark-plan probe commands avoid rewriting fixture-owned action sidecars.
- Databricks engine-probe matrix parameters use the same fixture-owned action suppression, including canonical path aliases.
- README docs describe the derived connector-action sidecar behavior.

## Review

GPT-5.5 high-reasoning review found a P1 double-writer issue in Databricks fixture target parameters, followed by a narrower P2 alias case. Both were fixed: Databricks runner parameters now suppress fixture-owned action sidecars, and the comparison uses canonical artifact-path equivalence. Final delta review reported no blocking issues.

## Verification

- `PYTHONPATH=src pytest tests/test_benchmark_plan.py::test_build_v1_benchmark_plan_derives_fixture_actions_output_json tests/test_benchmark_plan.py::test_main_derives_fixture_actions_for_release_planned_engine_probes tests/test_benchmark_plan.py::test_main_can_derive_planned_engine_probe_handoff_from_fixture_output_dir tests/test_benchmark_plan.py::test_engine_probe_targets_record_can_feed_databricks_matrix_helper -q`
- `PYTHONPATH=src pytest tests/test_databricks_engine_probe_job.py::test_databricks_engine_probe_matrix_payload_skips_fixture_owned_actions_output tests/test_databricks_engine_probe_job.py::test_databricks_engine_probe_matrix_payload_skips_fixture_owned_actions_alias -q`
- `PYTHONPATH=src pytest tests/test_databricks_engine_probe_job.py tests/test_benchmark_plan.py -q`
- `PYTHONPATH=src pytest -q`
- `python -m compileall -q src tests`
- `git diff --check`
