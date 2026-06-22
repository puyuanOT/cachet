# `vllm_kv_injection`

This package defines the vLLM-facing document KV-cache integration contract.

- `protocol.py` re-exports `KVLayout`, `KVSegment`, and `KVCacheHandle` from `document-kv-cache`.
- `block_mapping.py` maps token spans and occurrence-aware document segment keys onto planned or already-reserved vLLM-style KV blocks.
- `connector.py` defines the payload-aware connector protocol and an in-memory test double.
- `paged_kv_copy.py` converts allocated vLLM physical blocks into slot mappings and copies materialized per-layer KV tensors into vLLM paged KV buffers.
- `vllm_adapter.py` converts a document `EngineReadyRequest` into connector reserve/inject/release calls and returns block mappings for a patched vLLM runtime.
- `vllm_dynamic_connector.py` exposes the vLLM V1 dynamic connector entrypoint and provider seam for materializing document KV into vLLM-owned paged buffers.
- `vllm_native_provider.py` supplies the first runtime-facing provider/factory and provider-backed native probe connector: it reads Cachet handoffs from vLLM `kv_transfer_params`, records block-aligned scheduler allocations, and synchronously loads materialized payload bytes into registered paged KV tensors.
- `vllm_runtime_contract.py` documents the vLLM V1 KV connector lifecycle that
  a native patched runtime must implement around the shared document handoff and
  can emit an installed-runtime drift diagnostic before native probe launches.
- `vllm_transfer_config.py` builds the vLLM launch config shape for a patched connector while reserving `document_kv.*` handoff keys.

Keep this package close to vLLM internals and free of document retrieval, cache storage, CPU assembly, scheduling, or LoRA routing logic.
