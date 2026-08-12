"""Fail-closed structural conformance checks for KV method plugins."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from document_kv_cache.methods import (
    CACHET_ARTIFACT_EXECUTION,
    MethodSpec,
    default_method_registry,
)


METHOD_CONFORMANCE_RECORD_TYPE = "document_kv.method_conformance.v1"

__all__ = [
    "METHOD_CONFORMANCE_RECORD_TYPE",
    "MethodConformanceResult",
    "inspect_method_conformance",
    "method_conformance_to_record",
    "load_method_spec",
    "main",
]


@dataclass(frozen=True, slots=True)
class MethodConformanceResult:
    method: MethodSpec
    issues: tuple[str, ...]
    factory_loaded: bool
    generator_instantiated: bool

    @property
    def ok(self) -> bool:
        return not self.issues


def inspect_method_conformance(
    method: MethodSpec,
    *,
    allow_unimplemented: bool = False,
    instantiate_generator: bool = False,
) -> MethodConformanceResult:
    if not isinstance(method, MethodSpec):
        raise TypeError("method must be a MethodSpec")
    if type(allow_unimplemented) is not bool:
        raise TypeError("allow_unimplemented must be a boolean")
    if type(instantiate_generator) is not bool:
        raise TypeError("instantiate_generator must be a boolean")

    issues: list[str] = []
    factory_loaded = False
    generator_instantiated = False
    try:
        method.reuse_plan()
    except Exception as exc:
        issues.append(f"reuse plan: {type(exc).__name__}: {exc}")
    try:
        default_method_registry().with_spec(method)
    except Exception as exc:
        issues.append(f"registry composition: {type(exc).__name__}: {exc}")
    if not method.implemented:
        if not allow_unimplemented:
            issues.append("method is not implemented")
        if instantiate_generator:
            issues.append("cannot instantiate an unimplemented method")
    elif method.execution_kind == CACHET_ARTIFACT_EXECUTION:
        try:
            method.load_generator_factory()
            factory_loaded = True
        except Exception as exc:
            issues.append(f"generator factory: {type(exc).__name__}: {exc}")
        if instantiate_generator and factory_loaded:
            try:
                method.create_generator()
                generator_instantiated = True
            except Exception as exc:
                issues.append(f"generator instantiation: {type(exc).__name__}: {exc}")
    elif instantiate_generator:
        issues.append("engine-native methods do not instantiate Cachet generators")
    return MethodConformanceResult(
        method=method,
        issues=tuple(issues),
        factory_loaded=factory_loaded,
        generator_instantiated=generator_instantiated,
    )


def method_conformance_to_record(result: MethodConformanceResult) -> dict[str, Any]:
    if not isinstance(result, MethodConformanceResult):
        raise TypeError("result must be a MethodConformanceResult")
    method = result.method
    return {
        "record_type": METHOD_CONFORMANCE_RECORD_TYPE,
        "ok": result.ok,
        "issues": list(result.issues),
        "method": {
            "method_id": method.method_id,
            "display_name": method.display_name,
            "implemented": method.implemented,
            "artifact_version": method.artifact_version,
            "execution_kind": method.execution_kind,
            "generator_factory": method.generator_factory,
            "reuse_plan": method.reuse_plan().to_record(),
        },
        "factory_loaded": result.factory_loaded,
        "generator_instantiated": result.generator_instantiated,
    }


def load_method_spec(path: str) -> MethodSpec:
    if not isinstance(path, str) or not path:
        raise ValueError("plugin path must be a non-empty module:attribute string")
    module_name, separator, attribute_name = path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("plugin path must use module.path:attribute")
    try:
        method = getattr(import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"could not load method plugin {path!r}") from exc
    if not isinstance(method, MethodSpec):
        raise TypeError(f"{path!r} does not reference a MethodSpec")
    return method


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a Cachet KV method contract.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--method-id", help="Built-in method ID.")
    source.add_argument("--plugin", help="Application MethodSpec as module.path:attribute.")
    parser.add_argument("--allow-unimplemented", action="store_true")
    parser.add_argument(
        "--instantiate-generator",
        action="store_true",
        help="Call a Cachet artifact method's zero-argument factory (may load a model).",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        method = (
            default_method_registry().get(args.method_id)
            if args.method_id is not None
            else load_method_spec(args.plugin)
        )
        result = inspect_method_conformance(
            method,
            allow_unimplemented=args.allow_unimplemented,
            instantiate_generator=args.instantiate_generator,
        )
        rendered = json.dumps(method_conformance_to_record(result), indent=2, sort_keys=True) + "\n"
        if args.output_json:
            target = Path(args.output_json)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except Exception as exc:
        parser.error(str(exc))
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
