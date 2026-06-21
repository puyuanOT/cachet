"""Private helpers for Cachet module facades."""

from __future__ import annotations

from importlib import import_module
import sys
from types import ModuleType
from typing import Any


def install(alias_module_name: str, document_module_name: str) -> ModuleType:
    module = import_module(document_module_name)
    sys.modules[alias_module_name] = module
    return module


def run_main(document_module_name: str) -> Any:
    module = import_module(document_module_name)
    return module.main()
