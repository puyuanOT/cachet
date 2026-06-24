# Native Probe Factory Diagnostics CLI

This PR adds `document-kv-native-probe-factories`, a document-branded command
that emits the built-in vLLM/SGLang native probe factory diagnostics record to
stdout or a JSON file. The command keeps the factories fail-closed while giving
operators a package-installed way to capture reserved probe entry points,
support status, and pinned isolated serving-environment profiles.

Verification:

- `poetry run pytest tests/test_native_probe_factories.py tests/test_public_package.py tests/test_project_governance.py -q`
- `git diff --check`
- `poetry run python -m compileall -q src tests`
- `poetry run pytest -q`
- GPT-5.5 spot checks for `python -m document_kv_cache.native_probe_factories --help` and stdout output.

GPT-5.5 review outcome: approved with no findings.
