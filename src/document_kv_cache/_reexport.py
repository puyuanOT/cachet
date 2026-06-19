"""Internal helpers for document namespace wrapper modules."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from threading import RLock
from typing import Any


def reexport_public(legacy_module_name: str, public_names: tuple[str, ...], namespace: dict[str, Any]) -> list[str]:
    legacy_module = import_module(legacy_module_name)
    for name in public_names:
        namespace[name] = getattr(legacy_module, name)
    return list(public_names)


class LegacyMainBridge:
    """Call a legacy CLI main while honoring public-wrapper hook overrides."""

    def __init__(
        self,
        *,
        legacy_module_name: str,
        public_namespace: dict[str, Any],
        hook_names: tuple[str, ...],
    ) -> None:
        self._legacy_module = import_module(legacy_module_name)
        self._public_namespace = public_namespace
        self._hook_names = hook_names
        self._lock = RLock()

    def __call__(self, argv: Sequence[str] | None = None) -> int:
        with self._lock:
            previous_hooks = {name: getattr(self._legacy_module, name) for name in self._hook_names}
            for name in self._hook_names:
                setattr(self._legacy_module, name, self._public_namespace[name])
            try:
                return self._legacy_module.main(argv)
            finally:
                for name, previous_hook in previous_hooks.items():
                    setattr(self._legacy_module, name, previous_hook)
