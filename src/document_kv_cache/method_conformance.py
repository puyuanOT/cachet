"""Fail-closed structural conformance checks for KV method plugins."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from document_kv_cache.artifact_identity import method_config_digest
from document_kv_cache.engine_adapters import (
    EngineAdapterSpec,
    build_engine_adapter_request,
    build_engine_kv_injection_plan,
    engine_adapter_request_to_record,
    validate_engine_adapter_request_record,
    vllm_adapter_spec,
)
from document_kv_cache.engine_protocol import KVLayout
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.methods import (
    CACHET_ARTIFACT_EXECUTION,
    MethodSpec,
    default_method_registry,
)
from document_kv_cache.models import DocumentKVRequest
from document_kv_cache.workflow import (
    CacheBuildConfig,
    DocumentKVWorkflow,
    KVChunkGenerator,
    SourceDocument,
)
from document_kv_cache.reuse_contract import (
    RuntimeOperationHandlerRegistry,
    apply_runtime_operation_handlers,
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
    workflow_exercised: bool = False
    handoff_roundtrip: bool = False
    runtime_operations_exercised: bool = False

    @property
    def ok(self) -> bool:
        return not self.issues


def inspect_method_conformance(
    method: MethodSpec,
    *,
    allow_unimplemented: bool = False,
    instantiate_generator: bool = False,
    exercise_runtime: bool = False,
    adapter_spec: EngineAdapterSpec | None = None,
    operation_handlers: RuntimeOperationHandlerRegistry | None = None,
) -> MethodConformanceResult:
    if not isinstance(method, MethodSpec):
        raise TypeError("method must be a MethodSpec")
    if type(allow_unimplemented) is not bool:
        raise TypeError("allow_unimplemented must be a boolean")
    if type(instantiate_generator) is not bool:
        raise TypeError("instantiate_generator must be a boolean")
    if type(exercise_runtime) is not bool:
        raise TypeError("exercise_runtime must be a boolean")
    if adapter_spec is not None and not isinstance(adapter_spec, EngineAdapterSpec):
        raise TypeError("adapter_spec must be an EngineAdapterSpec or None")
    if operation_handlers is not None and not isinstance(
        operation_handlers,
        RuntimeOperationHandlerRegistry,
    ):
        raise TypeError(
            "operation_handlers must be a RuntimeOperationHandlerRegistry or None"
        )
    if exercise_runtime and not instantiate_generator:
        raise ValueError("exercise_runtime requires instantiate_generator=True")

    issues: list[str] = []
    factory_loaded = False
    generator_instantiated = False
    workflow_exercised = False
    handoff_roundtrip = False
    runtime_operations_exercised = False
    generator: object | None = None
    if method.implemented:
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
                generator = method.create_generator()
                generator_instantiated = True
            except Exception as exc:
                issues.append(f"generator instantiation: {type(exc).__name__}: {exc}")
        if exercise_runtime and generator_instantiated:
            assert generator is not None
            try:
                runtime_operations_exercised = _exercise_runtime_handoff(
                    method,
                    generator,
                    adapter_spec=adapter_spec,
                    operation_handlers=operation_handlers,
                )
                workflow_exercised = True
                handoff_roundtrip = True
            except Exception as exc:
                issues.append(f"runtime handoff: {type(exc).__name__}: {exc}")
    elif instantiate_generator:
        issues.append("engine-native methods do not instantiate Cachet generators")
    return MethodConformanceResult(
        method=method,
        issues=tuple(issues),
        factory_loaded=factory_loaded,
        generator_instantiated=generator_instantiated,
        workflow_exercised=workflow_exercised,
        handoff_roundtrip=handoff_roundtrip,
        runtime_operations_exercised=runtime_operations_exercised,
    )


def _exercise_runtime_handoff(
    method: MethodSpec,
    generator: object,
    *,
    adapter_spec: EngineAdapterSpec | None,
    operation_handlers: RuntimeOperationHandlerRegistry | None,
) -> bool:
    layout = getattr(generator, "layout", None)
    if not isinstance(layout, KVLayout):
        raise TypeError("runtime conformance requires generator.layout to be a KVLayout")
    registry = default_method_registry().with_spec(method)
    manifest = InMemoryManifestStore()
    workflow = DocumentKVWorkflow.with_storage(
        manifest=manifest,
        memory_blobs={},
        method_registry=registry,
    )
    if layout.storage_layout is None:
        raise ValueError("runtime conformance requires layout.storage_layout")
    prompt_template_version = "method-conformance-v1"
    config = CacheBuildConfig(
        model_id=layout.model_id,
        lora_id=layout.lora_id,
        prompt_template_version=prompt_template_version,
        dtype=layout.dtype,
        layout_version=layout.layout_version,
        cache_method=method.method_id,
        storage_layout=layout.storage_layout,
        payload_axis_order=layout.payload_axis_order,
        method_version=method.artifact_version,
        method_config_digest=method_config_digest({"conformance": True}),
        model_revision="method-conformance",
        tokenizer_id="method-conformance-tokenizer",
        tokenizer_revision="1",
        generator_family="method-conformance",
        generator_version="1",
        artifact_format_id=method.artifact_format.format_id,
        artifact_format_version=method.artifact_format.version,
        runtime_kv_dtype=layout.dtype,
    )
    document = SourceDocument.from_text(
        document_id="method-conformance-document",
        text="cachet",
    )
    generated = workflow.generate_cache(
        documents=(document,),
        generator=cast(KVChunkGenerator, generator),
        config=config,
        shard_uri=f"memory://method-conformance/{method.method_id}.kvpack",
        align_bytes=1,
    )
    request = DocumentKVRequest.for_text_document(
        request_id="method-conformance-request",
        task_id="method-conformance",
        model_id=layout.model_id,
        lora_id=layout.lora_id,
        prompt_template_version=prompt_template_version,
        document_id=document.document_id,
        artifact_identity=generated.artifact_identity,
    )
    ready = workflow.prepare_for_engine(
        request,
        layout=layout,
        cache_method=method.method_id,
    )
    resolved_adapter_spec = adapter_spec or vllm_adapter_spec()
    handlers = operation_handlers or RuntimeOperationHandlerRegistry()
    adapter_request = build_engine_adapter_request(
        ready,
        spec=resolved_adapter_spec,
        operation_handlers=handlers,
        method_registry=registry,
    )
    serialized = json.loads(
        json.dumps(
            engine_adapter_request_to_record(
                adapter_request,
                adapter_spec=resolved_adapter_spec,
                operation_handlers=handlers,
                method_registry=registry,
            )
        )
    )
    validate_engine_adapter_request_record(
        serialized,
        require_external_payload_uri=False,
        adapter_spec=resolved_adapter_spec,
        operation_handlers=handlers,
        method_registry=registry,
    )
    injection_plan = build_engine_kv_injection_plan(
        serialized,
        require_external_payload_uri=False,
        adapter_spec=resolved_adapter_spec,
        operation_handlers=handlers,
        method_registry=registry,
    )
    if injection_plan.reuse_plan.capability_id != method.reuse_plan().capability_id:
        raise ValueError("handoff changed the method reuse capability identity")
    if not isinstance(ready.payload, bytes):
        raise TypeError("runtime conformance requires one merged payload")
    operation_result = apply_runtime_operation_handlers(
        injection_plan.reuse_plan,
        ready.payload,
        layout=injection_plan.layout,
        total_tokens=injection_plan.total_tokens,
        handler_registry=handlers,
        metadata=injection_plan.metadata,
    )
    if operation_result.payload is None:
        raise ValueError("runtime operation pipeline did not produce payload bytes")
    return bool(injection_plan.reuse_plan.runtime_operations)


def method_conformance_to_record(result: MethodConformanceResult) -> dict[str, Any]:
    if not isinstance(result, MethodConformanceResult):
        raise TypeError("result must be a MethodConformanceResult")
    method = result.method
    assert method.lifecycle is not None
    reuse_plan = method.reuse_plan().to_record() if method.implemented else None
    return {
        "record_type": METHOD_CONFORMANCE_RECORD_TYPE,
        "ok": result.ok,
        "issues": list(result.issues),
        "method": {
            "method_id": method.method_id,
            "display_name": method.display_name,
            "implemented": method.implemented,
            "lifecycle": method.lifecycle.to_record(),
            "artifact_version": method.artifact_version,
            "execution_kind": method.execution_kind,
            "generator_factory": method.generator_factory,
            "reuse_plan": reuse_plan,
        },
        "factory_loaded": result.factory_loaded,
        "generator_instantiated": result.generator_instantiated,
        "workflow_exercised": result.workflow_exercised,
        "handoff_roundtrip": result.handoff_roundtrip,
        "runtime_operations_exercised": result.runtime_operations_exercised,
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
    parser.add_argument(
        "--exercise-runtime",
        action="store_true",
        help=(
            "Run strict generation, preparation, and a vLLM handoff round-trip. "
            "Requires a zero-argument CPU-safe generator factory."
        ),
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
            instantiate_generator=(
                args.instantiate_generator or args.exercise_runtime
            ),
            exercise_runtime=args.exercise_runtime,
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
