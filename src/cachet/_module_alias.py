"""Private helpers for Cachet module facades."""

from __future__ import annotations

from importlib.abc import Loader, MetaPathFinder
from importlib import import_module
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
import sys
from types import ModuleType
from typing import Any

_ALIAS_MODULE_PREFIXES: dict[str, str] = {}
_ALIAS_FINDER: "_AliasSubmoduleFinder | None" = None


class _AliasSubmoduleFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        target_name = _target_module_name(fullname)
        if target_name is None:
            return None
        target_spec = find_spec(target_name)
        if target_spec is None:
            return None
        spec = ModuleSpec(
            fullname,
            _AliasSubmoduleLoader(alias_name=fullname, target_name=target_name),
            origin=target_spec.origin,
            is_package=target_spec.submodule_search_locations is not None,
        )
        if target_spec.submodule_search_locations is not None:
            spec.submodule_search_locations = list(target_spec.submodule_search_locations)
        return spec


class _AliasSubmoduleLoader(Loader):
    def __init__(self, *, alias_name: str, target_name: str) -> None:
        self._alias_name = alias_name
        self._target_name = target_name

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        module = import_module(self._target_name)
        sys.modules[self._alias_name] = module
        return module

    def exec_module(self, module: ModuleType) -> None:
        sys.modules[self._alias_name] = module


def install(alias_module_name: str, document_module_name: str) -> ModuleType:
    module = import_module(document_module_name)
    _install_submodule_alias(alias_module_name, document_module_name)
    sys.modules[alias_module_name] = module
    return module


def run_main(document_module_name: str) -> Any:
    module = import_module(document_module_name)
    return module.main()


def _install_submodule_alias(alias_module_name: str, document_module_name: str) -> None:
    existing_target = _ALIAS_MODULE_PREFIXES.get(alias_module_name)
    if existing_target is not None and existing_target != document_module_name:
        raise ValueError(
            f"module alias {alias_module_name!r} is already installed for {existing_target!r}"
        )
    _ALIAS_MODULE_PREFIXES[alias_module_name] = document_module_name
    _install_alias_finder()
    _alias_existing_target_modules(alias_module_name, document_module_name)


def _install_alias_finder() -> None:
    global _ALIAS_FINDER
    if _ALIAS_FINDER is None:
        _ALIAS_FINDER = _AliasSubmoduleFinder()
    if _ALIAS_FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _ALIAS_FINDER)


def _alias_existing_target_modules(alias_module_name: str, document_module_name: str) -> None:
    for module_name, module in tuple(sys.modules.items()):
        if module_name == document_module_name or module_name.startswith(f"{document_module_name}."):
            alias_name = f"{alias_module_name}{module_name[len(document_module_name):]}"
            sys.modules.setdefault(alias_name, module)


def _target_module_name(fullname: str) -> str | None:
    for alias_module_name, document_module_name in sorted(
        _ALIAS_MODULE_PREFIXES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if fullname == alias_module_name:
            return document_module_name
        if fullname.startswith(f"{alias_module_name}."):
            return f"{document_module_name}{fullname[len(alias_module_name):]}"
    return None
