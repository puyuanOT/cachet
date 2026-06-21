# Benchmark Plan Probe Fixtures

This directory contains PR evidence for wiring deterministic Qwen3 engine-probe
fixture generation into the benchmark plan.

Verification recorded here:

- `pytest -q tests/test_benchmark_plan.py tests/test_probe_fixtures.py tests/test_public_package.py`
- `pytest -q`
- `python -m build --wheel`

GPT-5.5 review found two fixture-planning gaps. The PR resolved them by
validating programmatic fixture handoff paths, checking fixture child artifact
collisions, and adding regression tests.
