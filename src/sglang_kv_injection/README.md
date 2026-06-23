# `sglang_kv_injection`

This package defines the SGLang-facing document KV-cache integration contract.

- `protocol.py` re-exports `KVLayout`, `KVSegment`, and `KVCacheHandle` from `cachet-kv`.
- `record.py` defines `SGLangCacheRecord` and deterministic SGLang prefix-key construction.
- `connector.py` defines the payload-aware connector protocol and an in-memory test double.
- `sglang_adapter.py` converts a document `EngineReadyRequest` into connector stage/attach/release calls for a patched SGLang runtime.
- `sglang_dynamic_backend.py` exposes `DocumentKVHiCacheBackend`, the
  importable SGLang dynamic HiCache backend that delegates storage calls to a
  provider factory, plus Cachet's built-in memory/filesystem HiCache page
  provider. When SGLang supplies
  `HiCacheStorageExtraInfo.extra_info.custom_params.kv_transfer_params`, the
  backend normalizes it into `DocumentKVHiCacheRequestContext` and forwards it
  to provider batch methods that accept `document_kv_request_context`. Cachet's
  built-in page provider uses that context to validate SGLang handoffs, read the
  external payload, and hydrate full HiCache pages only when
  `document_kv.sglang_hicache_page_keys` matches SGLang's runtime hash keys.
- `probe.py` exposes strict debug and native probe wrappers, including the provider-backed `DocumentKVHiCacheProbeConnector` used to exercise Cachet's HiCache page provider without falling back to in-memory test doubles.
- `sglang_hicache_config.py` builds launch metadata for a patched SGLang HiCache dynamic storage backend.
- `sglang_runtime_preflight.py` validates provider-backed dynamic HiCache
  launch wiring and records whether the installed SGLang runtime bridges
  request `custom_params` into `HiCacheStorageExtraInfo.extra_info`; live
  SGLang cache-arm benchmarks require that bridge to be true.
- `sglang_runtime_contract.py` documents the runtime-cache bridge this package validates around the shared document handoff.

Keep this package close to SGLang internals and free of document retrieval, cache storage, CPU assembly, scheduling, or LoRA routing logic.
