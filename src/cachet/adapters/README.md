# `cachet.adapters`

This package exposes Cachet-branded import aliases for the engine-specific
adapter packages vendored in this repository.

- `vllm.py` aliases the compatibility package `vllm_kv_injection`.
- `sglang.py` aliases the compatibility package `sglang_kv_injection`.

Nested imports such as `cachet.adapters.vllm.probe` and
`cachet.adapters.sglang.probe` resolve to the same module objects as the
vendored compatibility paths, so Cachet-branded imports do not split runtime
class identity from existing launch metadata.

The same aliasing covers the production runtime modules:

- `cachet.adapters.vllm.vllm_native_provider` is the same module object as
  `vllm_kv_injection.vllm_native_provider`, including
  `DocumentKVNativeProvider` and `build_document_kv_provider`.
- `cachet.adapters.vllm.vllm_dynamic_connector` is the same module object as
  `vllm_kv_injection.vllm_dynamic_connector`, including `DocumentKVConnector`.
- `cachet.adapters.sglang.sglang_dynamic_backend` is the same module object as
  `sglang_kv_injection.sglang_dynamic_backend`, including
  `DocumentKVHiCacheBackend` and `build_document_kv_hicache_provider`.
- `cachet.adapters.sglang.sglang_request_metadata_bridge` is the same module
  object as `sglang_kv_injection.sglang_request_metadata_bridge`.

The compatibility package names remain importable because existing Databricks
probe metadata and launch configs reference those dotted paths.
