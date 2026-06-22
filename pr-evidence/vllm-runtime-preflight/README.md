# vLLM Runtime Preflight Evidence

This evidence covers the Cachet vLLM native-runtime preflight added for PR
#314. The preflight combines installed-vLLM connector contract diagnostics,
the configured provider factory identity, and registered layer-name mapping
validation before provider-backed native probes can be treated as release-safe.

## Verification

- `python -m pytest -q tests/test_vllm_kv_injection_vllm_runtime_preflight.py tests/test_vllm_kv_injection_vllm_dynamic_connector.py tests/test_vllm_kv_injection_vllm_native_provider.py`
- `python -m pytest -q tests/test_vllm_kv_injection_vllm_runtime_preflight.py tests/test_vllm_kv_injection_vllm_native_provider.py tests/test_vllm_kv_injection_vllm_runtime_contract.py tests/test_vllm_kv_injection_probe.py tests/test_vllm_kv_injection_vllm_dynamic_connector.py tests/test_native_probe_factories.py tests/test_public_package.py tests/test_project_governance.py`
- `python -m pytest -q`
- `poetry check`
- `git diff --check`
