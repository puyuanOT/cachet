# `cachet`

This package is the branded Cachet import facade. It delegates to the canonical
`document_kv_cache` implementation so users can write `import cachet` while the
runtime classes, functions, protocols, and submodules keep their
`document_kv_cache.*` module identity.

New user-facing examples may import root APIs from `cachet`. The facade does
not publish a parallel `cachet.<module>` tree; lower-level module examples
should still use explicit `document_kv_cache.<module>` imports until the legacy
restaurant compatibility package is removed.

- `__init__.py` implements the lightweight runtime facade.
- `__init__.pyi` maps Cachet root exports to concrete `document_kv_cache.*`
  modules for static type checkers.
- `py.typed` marks the facade as a typed PEP 561 package.
