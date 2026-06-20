"""Built-in native probe factory entry points for backend integrations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata, util
from pathlib import Path
from types import MappingProxyType
from typing import Any

from document_kv_cache.engine_adapters import ServingBackend
from document_kv_cache.engine_probe import EngineKVProbeFactoryContext
from document_kv_cache.serving_env import (
    serving_environment_profile,
    serving_environment_profile_to_record,
)

VLLM_NATIVE_PROBE_FACTORY = "document_kv_cache.native_probe_factories:vllm_native_probe_factory"
SGLANG_NATIVE_PROBE_FACTORY = "document_kv_cache.native_probe_factories:sglang_native_probe_factory"
NATIVE_PROBE_FACTORIES_RECORD_TYPE = "document_kv.native_probe_factories.v1"


class NativeProbeFactoryUnavailable(RuntimeError):
    """Raised when a built-in backend factory cannot construct a native probe."""


@dataclass(frozen=True, slots=True)
class NativeProbeFactoryInspection:
    """Import and support status for a packaged native probe factory."""

    backend: ServingBackend
    factory_path: str
    package_name: str
    package_importable: bool
    package_version: str | None
    supported: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _serving_backend(self.backend))
        if not self.factory_path:
            raise ValueError("factory_path must be non-empty")
        if not self.package_name:
            raise ValueError("package_name must be non-empty")
        if type(self.package_importable) is not bool:
            raise ValueError("package_importable must be a boolean")
        if self.package_version is not None and not self.package_version:
            raise ValueError("package_version must be non-empty when provided")
        if type(self.supported) is not bool:
            raise ValueError("supported must be a boolean")
        if not self.reason:
            raise ValueError("reason must be non-empty")


def builtin_native_probe_factory_path(backend: ServingBackend | str) -> str:
    """Return the public factory path reserved for a supported serving backend."""

    backend = _serving_backend(backend)
    if backend == ServingBackend.VLLM:
        return VLLM_NATIVE_PROBE_FACTORY
    if backend == ServingBackend.SGLANG:
        return SGLANG_NATIVE_PROBE_FACTORY
    raise ValueError(f"Unsupported serving backend {backend!r}")


def inspect_builtin_native_probe_factory(backend: ServingBackend | str) -> NativeProbeFactoryInspection:
    """Inspect the local environment for a built-in native probe target.

    The package currently owns the evidence and handoff contract. The actual
    vLLM/SGLang block-manager imports must be implemented by backend-specific
    adapters before these factories can produce release evidence.
    """

    backend = _serving_backend(backend)
    package_name = _backend_package_name(backend)
    version = _package_version(package_name)
    package_importable = util.find_spec(package_name) is not None
    if version is None and not package_importable:
        reason = f"{package_name!r} is not installed in this environment"
    elif version is None:
        reason = f"{package_name!r} is importable but package metadata is unavailable"
    else:
        reason = (
            f"{package_name} {version} is installed; a backend-native "
            "Document KV block-manager adapter is still required"
        )
    return NativeProbeFactoryInspection(
        backend=backend,
        factory_path=builtin_native_probe_factory_path(backend),
        package_name=package_name,
        package_importable=package_importable,
        package_version=version,
        supported=False,
        reason=reason,
    )


def native_probe_factory_inspection_to_record(
    inspection: NativeProbeFactoryInspection,
) -> dict[str, Any]:
    """Serialize an inspection result for diagnostics and release planning."""

    serving_profile = serving_environment_profile(inspection.backend)
    return {
        "backend": inspection.backend.value,
        "factory_path": inspection.factory_path,
        "package_name": inspection.package_name,
        "package_importable": inspection.package_importable,
        "package_version": inspection.package_version,
        "serving_environment_profile": serving_environment_profile_to_record(serving_profile),
        "supported": inspection.supported,
        "reason": inspection.reason,
    }


def inspect_builtin_native_probe_factories() -> tuple[NativeProbeFactoryInspection, ...]:
    """Inspect all built-in native probe factory entry points."""

    return tuple(
        inspect_builtin_native_probe_factory(backend)
        for backend in (ServingBackend.VLLM, ServingBackend.SGLANG)
    )


def builtin_native_probe_factories_to_record() -> dict[str, Any]:
    """Serialize all built-in factory inspections as a stable diagnostics record."""

    return {
        "record_type": NATIVE_PROBE_FACTORIES_RECORD_TYPE,
        "factories": [
            native_probe_factory_inspection_to_record(inspection)
            for inspection in inspect_builtin_native_probe_factories()
        ],
    }


def write_builtin_native_probe_factories_record_json(path: str | Path) -> None:
    """Write the built-in native factory diagnostics record to a JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(builtin_native_probe_factories_to_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def vllm_native_probe_factory(context: EngineKVProbeFactoryContext) -> Any:
    """Reserved vLLM factory entry point.

    This fails closed until a real vLLM block-manager adapter is implemented.
    """

    _validate_context_backend(context, ServingBackend.VLLM)
    raise NativeProbeFactoryUnavailable(inspect_builtin_native_probe_factory(ServingBackend.VLLM).reason)


def sglang_native_probe_factory(context: EngineKVProbeFactoryContext) -> Any:
    """Reserved SGLang factory entry point.

    This fails closed until a real SGLang block-manager adapter is implemented.
    """

    _validate_context_backend(context, ServingBackend.SGLANG)
    raise NativeProbeFactoryUnavailable(inspect_builtin_native_probe_factory(ServingBackend.SGLANG).reason)


def _validate_context_backend(context: EngineKVProbeFactoryContext, expected: ServingBackend) -> None:
    observed = _serving_backend(context.backend)
    if observed != expected:
        raise ValueError(
            f"{expected.value} native probe factory received {observed.value!r} handoff context"
        )


def _backend_package_name(backend: ServingBackend) -> str:
    if backend == ServingBackend.VLLM:
        return "vllm"
    if backend == ServingBackend.SGLANG:
        return "sglang"
    raise ValueError(f"Unsupported serving backend {backend!r}")


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _serving_backend(value: ServingBackend | str) -> ServingBackend:
    try:
        return value if isinstance(value, ServingBackend) else ServingBackend(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported serving backend {value!r}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Emit built-in native probe factory diagnostics for release planning."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect built-in vLLM/SGLang native probe factory entry points "
            "and their pinned isolated serving-environment profiles."
        )
    )
    parser.add_argument(
        "--output-json",
        help="Optional file path for the diagnostics JSON. Defaults to stdout.",
    )
    args = parser.parse_args(argv)

    if args.output_json:
        write_builtin_native_probe_factories_record_json(args.output_json)
    else:
        record = builtin_native_probe_factories_to_record()
        print(json.dumps(record, indent=2, sort_keys=True))
    return 0


__all__ = [
    "NativeProbeFactoryInspection",
    "NativeProbeFactoryUnavailable",
    "SGLANG_NATIVE_PROBE_FACTORY",
    "VLLM_NATIVE_PROBE_FACTORY",
    "builtin_native_probe_factories_to_record",
    "builtin_native_probe_factory_path",
    "inspect_builtin_native_probe_factories",
    "inspect_builtin_native_probe_factory",
    "main",
    "native_probe_factory_inspection_to_record",
    "sglang_native_probe_factory",
    "vllm_native_probe_factory",
    "write_builtin_native_probe_factories_record_json",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
