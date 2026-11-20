# Source Layout

This folder contains importable Python packages for Cachet, the document
KV-cache library. The distribution package is `cachet-kv`, while the public
product import namespace is the branded `cachet` facade.

- `cachet/` is the branded import facade for root Cachet APIs and public
  `cachet.<module>` aliases.
- `document_kv_cache/` is the canonical implementation namespace used by
  runners, evidence files, and CLI entry points.
- `vllm_kv_injection/` and `sglang_kv_injection/` are the vendored engine
  adapter packages. Native probe metadata and launch configs reference their
  dotted module paths.
