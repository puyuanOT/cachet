# README Benchmark Plan Actions Sidecars

This PR updates the public benchmark-plan recipe so release-ready planned
engine probes include `--engine-probe-actions-output-json` for both vLLM and
SGLang. The root README and legacy migration README now explain that release
evidence consumes planned probe and connector-action outputs together.

The docs also clarify the direct-record fallback: existing native probe JSONs
must be paired with existing connector-action JSONs through
`--release-engine-probe-json` and `--release-engine-actions-json`.

Verification:

- `poetry run pytest tests/test_project_governance.py::test_readme_benchmark_plan_examples_include_release_actions_sidecars tests/test_benchmark_plan.py::test_main_can_include_planned_engine_probes_and_release_evidence_validation -q`
- `poetry run pytest tests/test_project_governance.py tests/test_benchmark_plan.py -q`
- `python -m compileall -q src/document_kv_cache src/restaurant_kv_serving`
- `git diff --check`
- `poetry run pytest -q`

GPT-5.5 initially requested the direct-record fallback wording fix. After that
patch, GPT-5.5 re-reviewed the diff and approved it with no findings.
