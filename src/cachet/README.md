# `cachet`

This package is the branded Cachet import facade. It delegates to the canonical
`document_kv_cache` implementation so users can write `import cachet` and
`import cachet.workflow` while runtime classes, functions, protocols, and
submodules keep their `document_kv_cache.*` module identity.

New user-facing examples should prefer `cachet` and `cachet.<module>` imports.
The underlying implementation namespace remains importable as
`document_kv_cache` during the migration so existing benchmark evidence,
Databricks runners, and downstream jobs keep working.

- `__init__.py` implements the lightweight root facade.
- `*.py` module facades re-export typed APIs from the document-owned modules,
  return the underlying `document_kv_cache.*` modules for normal imports, and
  forward `python -m cachet.<module>` for CLI-capable modules.
- `quickstart_local.py` is the packaged no-cloud local quickstart used by the
  root README and `examples/quickstart_local.py`.
- `__init__.pyi` maps Cachet root exports to concrete `document_kv_cache.*`
  modules for static type checkers.
- `py.typed` marks the facade as a typed PEP 561 package.
