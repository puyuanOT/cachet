# `cachet.adapters`

This package exposes Cachet-branded import aliases for the engine-specific
adapter packages vendored in this repository.

- `vllm.py` aliases the compatibility package `vllm_kv_injection`.
- `sglang.py` aliases the compatibility package `sglang_kv_injection`.

The compatibility package names remain importable because existing Databricks
probe metadata and launch configs reference those dotted paths.
