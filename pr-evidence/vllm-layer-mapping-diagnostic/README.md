# vLLM Layer Mapping Diagnostic Evidence

This evidence covers the Cachet vLLM provider diagnostic that serializes
registered KV cache layer-name mappings for Databricks and other runtime
preflights before provider-backed native probe launches.

## Verification

- `python -m pytest -q tests/test_vllm_kv_injection_vllm_native_provider.py`
- `python -m pytest -q tests/test_vllm_kv_injection_vllm_native_provider.py tests/test_vllm_kv_injection_probe.py tests/test_vllm_kv_injection_vllm_dynamic_connector.py tests/test_vllm_kv_injection_vllm_runtime_contract.py tests/test_vllm_kv_injection_vllm_transfer_config.py tests/test_native_probe_factories.py tests/test_public_package.py`
- `python -m pytest -q`
- `git diff --check`
