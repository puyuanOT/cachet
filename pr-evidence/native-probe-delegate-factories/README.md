# Native Probe Delegate Factories

## What Changed

- Added public delegate environment variables for vLLM and SGLang native probe
  factories.
- Preserved stable built-in factory paths while delegating to backend-native
  adapter factories configured in the serving environment.
- Added `delegate_factory_path` to native probe diagnostics and validation,
  with fail-closed checks for missing packages, non-importable packages,
  unloadable delegates, built-in self-delegation, public facade aliases, and
  unhashable callable delegates.
- Exported the delegate environment constants through `document_kv_cache`,
  `cachet`, and the legacy `restaurant_kv_serving` compatibility namespace.
- Updated native probe documentation and release-bundle fixtures.

## Why

Serving integrations need stable Cachet/Document KV evidence contracts while
keeping engine-specific vLLM/SGLang block-manager code in the target serving
environment. Delegation lets native adapters plug into release diagnostics
without forking the public factory path or claiming support before a real
backend adapter is available.

## Verification

- `poetry run pytest tests/test_native_probe_factories.py tests/test_release_bundle.py tests/test_public_package.py -q` -> 81 passed
- `poetry run pytest -q` -> 1185 passed
- `poetry check` -> All set
- `git diff --check` -> passed
- `poetry run python -m compileall -q src tests` -> passed
- `poetry run python -m document_kv_cache.pr_evidence --validate-directory pr-evidence/native-probe-delegate-factories --output-json pr-evidence/native-probe-delegate-factories/pr-evidence-validation.json` -> ok
- `poetry run python -m document_kv_cache.pr_evidence --validate-directory pr-evidence --output-json pr-evidence/pr-evidence-sidecar-validation.json` -> ok
- GPT-5.5 review -> findings resolved, final approval
