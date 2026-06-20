# Native Probe Serving Profile Diagnostics

This PR enriches built-in vLLM/SGLang native probe factory diagnostics with the
pinned isolated serving-environment profile for each backend. The factories
still fail closed until real block-manager adapters are implemented, but their
diagnostic record now tells operators which engine package, version, and
dependency constraints belong to each reserved probe entry point.

Verification:

- `poetry run pytest tests/test_native_probe_factories.py tests/test_serving_env.py tests/test_project_governance.py tests/test_public_package.py -q`
- `poetry run python` smoke for `builtin_native_probe_factories_to_record`
- `git diff --check`
- `poetry run python -m compileall -q src tests`
- `poetry run pytest -q`

GPT-5.5 review outcome: approved with no blocking findings.
