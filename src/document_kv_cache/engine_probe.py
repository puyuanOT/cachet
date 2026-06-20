"""CLI runner for validating engine-native KV connector handoffs."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from document_kv_cache.engine_adapters import (
    EngineAdapterRequest,
    EngineKVBlockManagerProbe,
    EngineKVConnectorProbeResult,
    EngineKVInjectionPlan,
    ServingBackend,
    build_engine_kv_connector_actions,
    build_engine_kv_injection_plan,
    engine_kv_connector_actions_to_record,
    engine_kv_connector_probe_result_to_record,
    probe_engine_kv_connector_actions,
    read_engine_adapter_request_json,
    validate_engine_kv_connector_probe_record,
    view_engine_adapter_payload,
    write_engine_adapter_request_json,
)
from document_kv_cache.serving_env import serving_environment_profile
from document_kv_cache.storage import local_path

_LOCAL_PAYLOAD_URI_SCHEMES = {"dbfs", "disk", "file", "uc-volume"}
__all__ = [
    "EngineKVProbeConfig",
    "ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND",
    "ENGINE_KV_PROBE_METADATA_HANDOFF_JSON",
    "ENGINE_KV_PROBE_METADATA_PAYLOAD_URI",
    "ENGINE_KV_PROBE_METADATA_PROBE_FACTORY",
    "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE",
    "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION",
    "EngineKVProbeFactory",
    "EngineKVProbeFactoryContext",
    "EngineKVProbeFactoryResult",
    "run_engine_kv_connector_probe",
    "read_engine_adapter_payload",
    "write_engine_adapter_handoff_bundle",
    "write_engine_adapter_payload",
    "write_engine_kv_connector_actions_record_json",
    "write_engine_kv_connector_probe_result_json",
    "load_engine_kv_probe_factory",
    "parse_args",
    "main",
]
ENGINE_KV_PROBE_METADATA_HANDOFF_JSON = "document_kv.handoff_json"
ENGINE_KV_PROBE_METADATA_PAYLOAD_URI = "document_kv.payload_uri"
ENGINE_KV_PROBE_METADATA_PROBE_FACTORY = "document_kv.probe_factory"
ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND = "document_kv.expected_backend"
ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE = "document_kv.serving_engine_package"
ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION = "document_kv.serving_engine_version"


@dataclass(frozen=True, slots=True)
class EngineKVProbeFactoryContext:
    """Validated handoff context passed to a backend-specific native probe factory."""

    backend: ServingBackend
    handoff_record: Mapping[str, Any]
    plan: EngineKVInjectionPlan
    payload_source_uri: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class EngineKVProbeFactoryResult:
    """Probe object plus engine metadata returned by a native adapter factory."""

    probe: EngineKVBlockManagerProbe
    engine_version: str
    native_probe: bool = True
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.engine_version:
            raise ValueError("engine_version must be non-empty")
        if type(self.native_probe) is not bool:
            raise TypeError("native_probe must be boolean")
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


EngineKVProbeFactory = Callable[
    [EngineKVProbeFactoryContext],
    EngineKVBlockManagerProbe | EngineKVProbeFactoryResult,
]


@dataclass(frozen=True, slots=True)
class EngineKVProbeConfig:
    """Inputs for producing one engine KV connector probe evidence record."""

    handoff_json: Path
    probe_factory: str
    output_json: Path | None = None
    expected_backend: ServingBackend | str | None = None
    payload_uri: str | None = None
    engine_version: str | None = None
    native_probe: bool = True
    metadata: Mapping[str, str] = field(default_factory=dict)
    actions_output_json: Path | None = None

    def __post_init__(self) -> None:
        if not self.probe_factory:
            raise ValueError("probe_factory must be non-empty")
        if type(self.native_probe) is not bool:
            raise TypeError("native_probe must be boolean")
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "handoff_json", Path(self.handoff_json))
        if self.output_json is not None:
            object.__setattr__(self, "output_json", Path(self.output_json))
        if self.actions_output_json is not None:
            object.__setattr__(self, "actions_output_json", Path(self.actions_output_json))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __reduce__(self):
        return (
            type(self),
            (
                self.handoff_json,
                self.probe_factory,
                self.output_json,
                self.expected_backend,
                self.payload_uri,
                self.engine_version,
                self.native_probe,
                dict(self.metadata),
                self.actions_output_json,
            ),
        )


def run_engine_kv_connector_probe(config: EngineKVProbeConfig) -> EngineKVConnectorProbeResult:
    """Load a handoff record and validate it against a native engine probe factory."""

    record = read_engine_adapter_request_json(
        config.handoff_json,
        expected_backend=config.expected_backend,
        require_external_payload_uri=config.payload_uri is None,
    )
    plan = build_engine_kv_injection_plan(
        record,
        expected_backend=config.expected_backend,
        require_external_payload_uri=config.payload_uri is None,
    )
    payload_uri = config.payload_uri or plan.payload_source_uri
    if payload_uri is None:
        raise ValueError("Engine KV probe requires a payload URI in the handoff record or config")

    payload = read_engine_adapter_payload(payload_uri, expected_bytes=plan.total_bytes)
    payload_or_segments = view_engine_adapter_payload(record, payload)
    actions = build_engine_kv_connector_actions(plan, payload_or_segments)

    factory = load_engine_kv_probe_factory(config.probe_factory)
    factory_context = EngineKVProbeFactoryContext(
        backend=plan.backend,
        handoff_record=record,
        plan=plan,
        payload_source_uri=payload_uri,
        metadata=config.metadata,
    )
    factory_output = factory(factory_context)
    if isinstance(factory_output, EngineKVProbeFactoryResult):
        probe = factory_output.probe
        native_probe = config.native_probe and factory_output.native_probe
        if native_probe and config.engine_version is not None:
            raise ValueError("engine_version override is not allowed for native factory-result probes")
        engine_version = (
            factory_output.engine_version
            if native_probe
            else config.engine_version or factory_output.engine_version
        )
        metadata = {
            **factory_output.metadata,
            **config.metadata,
            **_probe_trace_metadata(config, payload_uri=payload_uri, backend=plan.backend),
        }
    else:
        probe = factory_output
        engine_version = config.engine_version or getattr(probe, "engine_version", None)
        native_probe = config.native_probe
        metadata = {
            **config.metadata,
            **_probe_trace_metadata(config, payload_uri=payload_uri, backend=plan.backend),
        }

    if not isinstance(engine_version, str) or not engine_version:
        raise ValueError(
            "Engine KV probe requires a non-empty engine_version from the config, "
            "factory result, or probe.engine_version"
        )
    if native_probe and engine_version == "unknown":
        raise ValueError("Native engine KV probe evidence cannot use engine_version='unknown'")

    result = probe_engine_kv_connector_actions(
        actions,
        payload_or_segments,
        probe,
        engine_version=engine_version,
        native_probe=native_probe,
        metadata=metadata,
    )
    if native_probe:
        validate_engine_kv_connector_probe_record(engine_kv_connector_probe_result_to_record(result))
    if config.output_json is not None:
        write_engine_kv_connector_probe_result_json(result, config.output_json)
    if config.actions_output_json is not None:
        write_engine_kv_connector_actions_record_json(actions, config.actions_output_json)
    return result


def read_engine_adapter_payload(payload_uri: str, *, expected_bytes: int | None = None) -> bytes:
    """Read a materialized adapter payload from local disk, DBFS, or a UC Volume path."""

    _validate_local_payload_uri(payload_uri)
    payload = local_path(payload_uri).read_bytes()
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise ValueError(f"Engine adapter payload length {len(payload)} != expected {expected_bytes}")
    return payload


def write_engine_adapter_payload(request: EngineAdapterRequest, payload_uri: str) -> Path:
    """Write a materialized adapter payload to local disk, DBFS, or a UC Volume path."""

    if not isinstance(request, EngineAdapterRequest):
        raise TypeError("request must be an EngineAdapterRequest")
    _validate_local_payload_uri(payload_uri)
    request.ready_request.validate()
    output_path = local_path(payload_uri)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as payload_file:
        if isinstance(request.ready_request.payload, bytes):
            payload_file.write(request.ready_request.payload)
        else:
            for segment_payload in request.ready_request.payload:
                payload_file.write(segment_payload)
    payload_bytes = output_path.stat().st_size
    expected_bytes = request.ready_request.handle.total_bytes
    if payload_bytes != expected_bytes:
        raise ValueError(f"Engine adapter payload length {payload_bytes} != expected {expected_bytes}")
    return output_path


def write_engine_adapter_handoff_bundle(
    request: EngineAdapterRequest,
    handoff_json: str | Path,
    *,
    payload_uri: str,
    require_external_payload_uri: bool = True,
) -> tuple[Path, Path]:
    """Write a coordinated engine handoff JSON and payload file.

    Returns ``(handoff_path, payload_path)``. The payload is written first so a
    visible handoff record always points at materialized bytes.
    """

    payload_path = _resolved_local_path(payload_uri)
    handoff_path = _resolved_local_path(str(handoff_json))
    if _local_paths_collide(payload_path, handoff_path):
        raise ValueError("handoff_json and payload_uri must resolve to different files")
    payload_path = write_engine_adapter_payload(request, payload_uri)
    handoff_path = write_engine_adapter_request_json(
        request,
        handoff_json,
        payload_uri=payload_uri,
        require_external_payload_uri=require_external_payload_uri,
    )
    return handoff_path, payload_path


def _resolved_local_path(uri: str) -> Path:
    return local_path(uri).expanduser().resolve(strict=False)


def _local_paths_collide(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        return first.samefile(second)
    except FileNotFoundError:
        return False


def write_engine_kv_connector_actions_record_json(
    actions,
    path: str | Path,
) -> None:
    """Write the JSON descriptor for connector actions validated by a probe run."""

    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(engine_kv_connector_actions_to_record(actions), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_engine_kv_connector_probe_result_json(
    result: EngineKVConnectorProbeResult,
    path: str | Path,
) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(engine_kv_connector_probe_result_to_record(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_engine_kv_probe_factory(factory_path: str) -> EngineKVProbeFactory:
    """Load a probe factory from ``module:callable`` or ``module.callable`` syntax."""

    module_name, attribute_name = _split_factory_path(factory_path)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError(f"Engine KV probe factory {factory_path!r} is not callable")
    return factory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a native engine KV connector probe from a handoff JSON.")
    parser.add_argument(
        "--handoff-json",
        required=True,
        help="Path to document_kv.engine_adapter_request.v1 schema v2 JSON.",
    )
    parser.add_argument("--probe-factory", required=True, help="Dotted native probe factory, e.g. module:factory.")
    parser.add_argument("--output-json", help="Where to write the engine probe JSON record. Defaults to stdout.")
    parser.add_argument(
        "--actions-output-json",
        help="Optional path for the validated document_kv.engine_kv_connector_actions.v1 descriptor.",
    )
    parser.add_argument("--payload-uri", help="Override payload_source.uri from the handoff record.")
    parser.add_argument(
        "--engine-version",
        help=(
            "Fallback engine version for legacy probe objects or non-native debug probes. "
            "Native factory-result probes must report their own engine version."
        ),
    )
    parser.add_argument("--expected-backend", choices=[backend.value for backend in ServingBackend])
    parser.add_argument(
        "--allow-non-native-probe",
        action="store_true",
        help="Mark native_probe=false for local adapter debugging; release evidence rejects these records.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        metavar="KEY=VALUE",
        help="Additional string metadata to attach to the probe evidence record.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_engine_kv_connector_probe(
        EngineKVProbeConfig(
            handoff_json=Path(args.handoff_json),
            probe_factory=args.probe_factory,
            output_json=None,
            actions_output_json=args.actions_output_json,
            expected_backend=args.expected_backend,
            payload_uri=args.payload_uri,
            engine_version=args.engine_version,
            native_probe=not args.allow_non_native_probe,
            metadata=_parse_metadata_items(args.metadata or ()),
        )
    )
    if args.output_json:
        write_engine_kv_connector_probe_result_json(result, args.output_json)
    else:
        print(json.dumps(engine_kv_connector_probe_result_to_record(result), indent=2, sort_keys=True))
    return 0


def _split_factory_path(factory_path: str) -> tuple[str, str]:
    if ":" in factory_path:
        module_name, attribute_name = factory_path.split(":", maxsplit=1)
    else:
        module_name, _, attribute_name = factory_path.rpartition(".")
    if not module_name or not attribute_name:
        raise ValueError("probe_factory must use 'module:callable' or 'module.callable' syntax")
    return module_name, attribute_name


def _validate_local_payload_uri(payload_uri: str) -> None:
    if not isinstance(payload_uri, str) or not payload_uri:
        raise ValueError("payload_uri must be a non-empty string")
    if Path(payload_uri).is_absolute():
        return
    if ":" not in payload_uri:
        raise ValueError("payload_uri must be an absolute path or supported local URI")
    scheme = payload_uri.split(":", maxsplit=1)[0].lower()
    if scheme not in _LOCAL_PAYLOAD_URI_SCHEMES:
        raise ValueError(
            "Engine probe runner can read only absolute paths, disk:, file:, dbfs:, "
            f"or uc-volume: payload URIs, got {payload_uri!r}"
        )
    if scheme in {"disk", "file"}:
        target = payload_uri.split(":", maxsplit=1)[1]
        if not Path(target).is_absolute():
            raise ValueError("disk: and file: payload URIs must use absolute paths")
    if scheme == "dbfs" and not payload_uri.startswith("dbfs:/"):
        raise ValueError("dbfs payload URIs must use dbfs:/... paths")


def _parse_metadata_items(items: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError("metadata entries must use KEY=VALUE syntax")
        metadata[key] = value
    return metadata


def _probe_trace_metadata(
    config: EngineKVProbeConfig,
    *,
    payload_uri: str,
    backend: ServingBackend,
) -> dict[str, str]:
    profile = serving_environment_profile(backend)
    return {
        ENGINE_KV_PROBE_METADATA_HANDOFF_JSON: str(config.handoff_json),
        ENGINE_KV_PROBE_METADATA_PAYLOAD_URI: payload_uri,
        ENGINE_KV_PROBE_METADATA_PROBE_FACTORY: config.probe_factory,
        ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND: backend.value,
        ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE: profile.engine_package,
        ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION: profile.engine_version,
    }


def _validate_metadata_strings(metadata: Mapping[str, str]) -> None:
    invalid_entries = [
        key
        for key, value in metadata.items()
        if not isinstance(key, str) or not isinstance(value, str)
    ]
    if invalid_entries:
        raise TypeError("Engine KV probe metadata keys and values must be strings")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
