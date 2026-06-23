# Source Layout

This folder contains importable Python packages for Cachet, the document
KV-cache library. The distribution package is `cachet-kv`, while the public
product import namespace is the branded `cachet` facade.

- `cachet/` is the branded import facade for root Cachet APIs and public
  `cachet.<module>` aliases.
- `document_kv_cache/` is the canonical implementation and compatibility
  namespace used by existing runners, evidence files, and CLI entry points.
- `restaurant_kv_serving/` remains packaged as a migration-only compatibility
  layer for existing Databricks benchmark jobs and older imports.
- `vllm_kv_injection/` and `sglang_kv_injection/` are vendored engine-adapter
  compatibility packages. They stay importable under their existing names
  because native probe metadata references those dotted paths.
