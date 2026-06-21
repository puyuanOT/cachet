# `cachet`

This package is the branded Cachet import facade. It delegates to the canonical
`document_kv_cache` implementation so users can write `import cachet` and
`import cachet.workflow` while runtime classes, functions, protocols, and
submodules keep their `document_kv_cache.*` module identity.

New user-facing examples should prefer `cachet` and `cachet.<module>` imports.
The underlying implementation namespace remains importable as
`document_kv_cache` during the migration so existing benchmark evidence,
Databricks runners, and downstream jobs keep working.

- `__init__.py` implements the lightweight runtime facade and registers public
  `cachet.<module>` aliases for the document-owned modules.
- `__init__.pyi` maps Cachet root exports to concrete `document_kv_cache.*`
  modules for static type checkers.
- `py.typed` marks the facade as a typed PEP 561 package.
