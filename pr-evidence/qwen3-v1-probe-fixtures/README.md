# Qwen3 V1 Probe Fixtures

This sidecar documents the PR that adds deterministic Qwen3 V1 engine-probe
fixture generation. The fixture writer emits packed KV bytes, an external
payload, handoff JSON, and a small manifest through the existing
`DocumentKVService` and vLLM/SGLang adapter contracts.

Verification:

- `pytest -q tests/test_probe_fixtures.py tests/test_project_governance.py tests/test_public_package.py`
- `pytest -q`
- `python -m build --wheel`
