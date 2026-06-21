"""Generate fail-closed native probe delegate scaffolds."""

from __future__ import annotations

import argparse
import keyword
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from document_kv_cache.engine_adapters import ServingBackend

__all__ = [
    "NativeProbeDelegateScaffoldConfig",
    "render_native_probe_delegate_module",
    "write_native_probe_delegate_module",
    "main",
]


@dataclass(frozen=True, slots=True)
class NativeProbeDelegateScaffoldConfig:
    """Inputs for rendering a backend-native probe delegate module skeleton."""

    backend: ServingBackend | str
    module_name: str | None = None
    class_name: str | None = None

    def __post_init__(self) -> None:
        backend = ServingBackend(self.backend)
        object.__setattr__(self, "backend", backend)
        module_name = (
            f"cachet_{backend.value}_native_probe"
            if self.module_name is None
            else self.module_name
        )
        class_name = _default_probe_class_name(backend) if self.class_name is None else self.class_name
        _validate_identifier_path(module_name, field_name="module_name")
        _validate_identifier(class_name, field_name="class_name")
        object.__setattr__(self, "module_name", module_name)
        object.__setattr__(self, "class_name", class_name)


def render_native_probe_delegate_module(config: NativeProbeDelegateScaffoldConfig) -> str:
    """Render a Python delegate module that declares the Cachet native-probe contracts."""

    if not isinstance(config, NativeProbeDelegateScaffoldConfig):
        raise TypeError("config must be a NativeProbeDelegateScaffoldConfig")
    backend = config.backend.value
    return f'''"""Cachet {backend} native probe delegate skeleton.

This module is intentionally fail-closed. Replace the NotImplementedError
methods with backend-native block-manager calls before using it as release
evidence.
"""

from __future__ import annotations

from typing import Any

from document_kv_cache.engine_adapters import (
    EngineKVBindAction,
    EngineKVReleaseAction,
    EngineKVReservationAction,
    EngineKVSegmentCopyAction,
)
from document_kv_cache.engine_probe import (
    EngineKVProbeFactoryContext,
    EngineKVProbeFactoryResult,
)
from document_kv_cache.native_probe_factories import (
    native_probe_adapter_contract_to_record,
    native_probe_runtime_contract_to_record,
)


DOCUMENT_KV_NATIVE_PROBE_CONTRACT = native_probe_adapter_contract_to_record()
DOCUMENT_KV_NATIVE_PROBE_RUNTIME_CONTRACT = native_probe_runtime_contract_to_record("{backend}")


class {config.class_name}:
    """Adapter-owned facade over the {backend} KV block manager."""

    def __init__(self, context: EngineKVProbeFactoryContext) -> None:
        self.context = context

    def reserve_kv_blocks(self, action: EngineKVReservationAction) -> Any:
        """Reserve {backend}-owned KV blocks for ``action.request_id``."""

        raise NotImplementedError("wire this method to the {backend} native KV block allocator")

    def import_kv_segment(
        self,
        reservation: Any,
        action: EngineKVSegmentCopyAction,
        payload: memoryview,
    ) -> None:
        """Copy or map one Cachet payload slice into the reserved {backend} blocks."""

        raise NotImplementedError("wire this method to the {backend} native KV import path")

    def bind_kv_handle(self, reservation: Any, action: EngineKVBindAction) -> None:
        """Bind the populated reservation to the pending {backend} request."""

        raise NotImplementedError("wire this method to the {backend} native request binding path")

    def release_kv_blocks(self, reservation: Any, action: EngineKVReleaseAction) -> None:
        """Release adapter-owned {backend} KV state after validation or decode."""

        raise NotImplementedError("wire this method to the {backend} native release path")


def build_probe(context: EngineKVProbeFactoryContext) -> EngineKVProbeFactoryResult:
    """Return the Cachet probe delegate consumed by the built-in {backend} factory."""

    observed_backend = getattr(context.backend, "value", context.backend)
    if observed_backend != "{backend}":
        raise ValueError(f"expected {backend} handoff context, got {{observed_backend!r}}")
    return EngineKVProbeFactoryResult(
        probe={config.class_name}(context),
        engine_version=_detect_engine_version(),
        native_probe=True,
        metadata={{
            "document_kv.adapter_scaffold": {backend!r},
            "document_kv.adapter_module": {config.module_name!r},
        }},
    )


build_probe.document_kv_native_probe_contract = DOCUMENT_KV_NATIVE_PROBE_CONTRACT
build_probe.document_kv_native_probe_runtime_contract = DOCUMENT_KV_NATIVE_PROBE_RUNTIME_CONTRACT


def _detect_engine_version() -> str:
    from importlib import metadata

    try:
        return metadata.version({backend!r})
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("install {backend} before running this native probe delegate") from exc
'''


def write_native_probe_delegate_module(
    config: NativeProbeDelegateScaffoldConfig,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a rendered native-probe delegate module to ``path``."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists; pass overwrite=True to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_native_probe_delegate_module(config), encoding="utf-8")
    return target


def _default_probe_class_name(backend: ServingBackend) -> str:
    return {
        ServingBackend.VLLM: "CachetVLLMBlockManagerProbe",
        ServingBackend.SGLANG: "CachetSGLangBlockManagerProbe",
    }[backend]


def _validate_identifier_path(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")
    for part in value.split("."):
        _validate_identifier(part, field_name=field_name)


def _validate_identifier(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(f"{field_name} must be a valid Python identifier")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Cachet native probe delegate skeleton.")
    parser.add_argument("--backend", required=True, choices=[backend.value for backend in ServingBackend])
    parser.add_argument("--output-file", required=True, help="Python file to write.")
    parser.add_argument("--module-name", help="Import module name for metadata, e.g. cachet_vllm_native_probe.")
    parser.add_argument("--class-name", help="Probe class name to render.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    args = parser.parse_args(argv)
    try:
        write_native_probe_delegate_module(
            NativeProbeDelegateScaffoldConfig(
                backend=args.backend,
                module_name=args.module_name,
                class_name=args.class_name,
            ),
            args.output_file,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
