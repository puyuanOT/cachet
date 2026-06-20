"""Compatibility facade for :mod:`document_kv_cache.engine_adapters`."""

from __future__ import annotations

from importlib import import_module as _import_module

_document_module = _import_module("document_kv_cache.engine_adapters")

for _name, _value in vars(_document_module).items():
    if _name != "__all__" and not _name.startswith("__"):
        globals()[_name] = _value

del _import_module
del _name
del _value
