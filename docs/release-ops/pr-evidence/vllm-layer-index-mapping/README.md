# vLLM Layer Index Mapping Evidence

This evidence covers the provider-side vLLM native runtime change that maps
registered KV cache tensors to Cachet payload layers using vLLM layer names
instead of dictionary insertion order.

## Verification

- `python -m pytest -q tests/test_vllm_kv_injection_vllm_native_provider.py`
- `python -m pytest -q tests/test_vllm_kv_injection_vllm_native_provider.py tests/test_vllm_kv_injection_probe.py tests/test_vllm_kv_injection_vllm_dynamic_connector.py tests/test_vllm_kv_injection_vllm_runtime_contract.py tests/test_vllm_kv_injection_vllm_transfer_config.py tests/test_native_probe_factories.py`
- `python -m pytest -q`
- `git diff --check`
