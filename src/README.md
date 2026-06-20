# Source Layout

This folder contains importable Python packages for Cachet, the document
KV-cache library. The distribution package is `document-kv-cache`, and the
public import namespace is `document_kv_cache`.

- `document_kv_cache/` is the canonical implementation, public import, and CLI
  namespace for new users.
- `restaurant_kv_serving/` remains packaged as a migration-only compatibility
  layer for existing Databricks benchmark jobs and older imports.
