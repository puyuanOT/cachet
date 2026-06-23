# `cachet.adapters`

This package exposes Cachet-branded import aliases for the engine-specific
adapter packages vendored in this repository.

- `vllm.py` aliases the compatibility package `vllm_kv_injection`.
- `sglang.py` aliases the compatibility package `sglang_kv_injection`.

Nested imports such as `cachet.adapters.vllm.probe` and
`cachet.adapters.sglang.probe` resolve to the same module objects as the
vendored compatibility paths, so Cachet-branded imports do not split runtime
class identity from existing launch metadata.

The compatibility package names remain importable because existing Databricks
probe metadata and launch configs reference those dotted paths.
