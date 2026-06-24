# Release Bundle Native Probe Factories Evidence

This directory records PR evidence for adding native-probe factory diagnostics to
release bundles and benchmark-plan release-bundle generation.

The GPT-5.5 review first requested stricter validation of embedded serving
environment profiles. The implementation now compares each native-probe factory
diagnostic profile to the package-owned built-in vLLM/SGLang profile record and
rejects malformed profile payloads.

Verification:

- `git diff --check`
- `poetry run python -m compileall -q src tests`
- `poetry run pytest -q`
