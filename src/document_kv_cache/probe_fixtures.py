"""Deterministic engine-probe fixture generation for native adapter work."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from document_kv_cache.admission import AdmissionQueue
from document_kv_cache.cache import ChunkCache
from document_kv_cache.engine_adapters import (
    EngineAdapterRequest,
    PayloadMode,
    ServingBackend,
    build_engine_adapter_request,
    build_engine_kv_connector_actions,
    build_engine_kv_injection_plan,
    engine_kv_connector_actions_to_record,
    read_engine_adapter_request_json,
    sglang_adapter_spec,
    validate_engine_kv_connector_actions_record,
    vllm_adapter_spec,
)
from document_kv_cache.engine_protocol import KVLayout
from document_kv_cache.engine_probe import write_engine_adapter_handoff_bundle
from document_kv_cache.kvpack import PackChunk, write_kvpack
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.materializer import KVMaterializer
from document_kv_cache.model_profiles import QWEN3_4B_INSTRUCT_PROFILE
from document_kv_cache.models import (
    DEFAULT_STATIC_CHUNK_ID,
    CacheGenerationMethod,
    DocumentChunkType,
    DocumentKVRequest,
    KVCacheKey,
)
from document_kv_cache.planner import CachePlanner
from document_kv_cache.service import DocumentKVService
from document_kv_cache.storage import DiskRangeReader, local_path

ENGINE_PROBE_FIXTURE_RECORD_TYPE = "document_kv.engine_probe_fixture.v1"
ENGINE_PROBE_FIXTURE_SCHEMA_VERSION = 1
DEFAULT_ENGINE_PROBE_FIXTURE_REQUEST_ID = "qwen3-v1-fixture-req"
DEFAULT_ENGINE_PROBE_FIXTURE_DOCUMENT_ID = "qwen3-v1-fixture-doc"
DEFAULT_ENGINE_PROBE_FIXTURE_TASK_ID = "engine-probe-fixture"
DEFAULT_ENGINE_PROBE_FIXTURE_TEMPLATE_VERSION = "qwen3-v1-fixture"
DEFAULT_ENGINE_PROBE_FIXTURE_CHUNK_IDS = ("section-1", "section-2")
DEFAULT_ENGINE_PROBE_FIXTURE_TOKENS_PER_SEGMENT = QWEN3_4B_INSTRUCT_PROFILE.default_block_size
DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES = MappingProxyType(
    {
        "pack": "qwen3-v1-fixture.kvpack",
        "payload": "qwen3-v1-fixture.payload.kv",
        "handoff": "qwen3-v1-fixture.handoff.json",
        "manifest": "qwen3-v1-fixture.manifest.json",
    }
)

__all__ = [
    "ENGINE_PROBE_FIXTURE_RECORD_TYPE",
    "ENGINE_PROBE_FIXTURE_SCHEMA_VERSION",
    "DEFAULT_ENGINE_PROBE_FIXTURE_REQUEST_ID",
    "EngineProbeFixtureConfig",
    "EngineProbeFixtureResult",
    "engine_probe_fixture_result_to_record",
    "write_qwen3_v1_engine_probe_fixture",
    "parse_args",
    "main",
]


@dataclass(frozen=True, slots=True)
class EngineProbeFixtureConfig:
    """Configuration for writing a deterministic Qwen3 V1 engine-probe fixture."""

    output_dir: str | Path
    backend: ServingBackend | str = ServingBackend.VLLM
    payload_mode: PayloadMode | str = PayloadMode.SEGMENTED
    request_id: str = DEFAULT_ENGINE_PROBE_FIXTURE_REQUEST_ID
    document_id: str = DEFAULT_ENGINE_PROBE_FIXTURE_DOCUMENT_ID
    task_id: str = DEFAULT_ENGINE_PROBE_FIXTURE_TASK_ID
    prompt_template_version: str = DEFAULT_ENGINE_PROBE_FIXTURE_TEMPLATE_VERSION
    lora_id: str = QWEN3_4B_INSTRUCT_PROFILE.default_lora_id
    dtype: str = QWEN3_4B_INSTRUCT_PROFILE.default_dtype
    tokens_per_segment: int = DEFAULT_ENGINE_PROBE_FIXTURE_TOKENS_PER_SEGMENT
    chunk_ids: Sequence[str] = DEFAULT_ENGINE_PROBE_FIXTURE_CHUNK_IDS
    include_static: bool = True
    static_chunk_id: str = DEFAULT_STATIC_CHUNK_ID
    adapter_ids: Sequence[str] = ()
    handle_uri: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", str(self.output_dir))
        object.__setattr__(self, "backend", _backend_from_value(self.backend))
        object.__setattr__(self, "payload_mode", _payload_mode_from_value(self.payload_mode))
        _validate_nonempty_string(self.request_id, field_name="request_id")
        _validate_nonempty_string(self.document_id, field_name="document_id")
        _validate_nonempty_string(self.task_id, field_name="task_id")
        _validate_nonempty_string(self.prompt_template_version, field_name="prompt_template_version")
        _validate_nonempty_string(self.lora_id, field_name="lora_id")
        _validate_nonempty_string(self.dtype, field_name="dtype")
        _validate_positive_int(self.tokens_per_segment, field_name="tokens_per_segment")
        object.__setattr__(self, "chunk_ids", _normalized_nonempty_string_tuple(self.chunk_ids, "chunk_ids"))
        object.__setattr__(self, "adapter_ids", _normalized_string_tuple(self.adapter_ids, "adapter_ids"))
        if type(self.include_static) is not bool:
            raise TypeError("include_static must be boolean")
        _validate_nonempty_string(self.static_chunk_id, field_name="static_chunk_id")
        if self.handle_uri is not None:
            _validate_nonempty_string(self.handle_uri, field_name="handle_uri")
        object.__setattr__(self, "metadata", MappingProxyType(_validated_metadata(self.metadata)))

    @property
    def segmented(self) -> bool:
        return self.payload_mode == PayloadMode.SEGMENTED


@dataclass(frozen=True, slots=True)
class EngineProbeFixtureResult:
    """Paths, URIs, and adapter request produced by fixture generation."""

    config: EngineProbeFixtureConfig
    layout: KVLayout
    adapter_request: EngineAdapterRequest
    pack_uri: str
    payload_uri: str
    handoff_uri: str
    manifest_uri: str
    pack_path: Path
    payload_path: Path
    handoff_json: Path
    manifest_json: Path
    pack_sha256: str
    payload_sha256: str
    handoff_sha256: str

    @property
    def total_tokens(self) -> int:
        return self.adapter_request.ready_request.handle.total_tokens

    @property
    def total_bytes(self) -> int:
        return self.adapter_request.ready_request.handle.total_bytes

    @property
    def segment_count(self) -> int:
        return len(self.adapter_request.ready_request.handle.segments)


def write_qwen3_v1_engine_probe_fixture(config: EngineProbeFixtureConfig) -> EngineProbeFixtureResult:
    """Write a Qwen3 V1 handoff/payload pair for native connector probes.

    The generated bytes are deterministic and sized according to the built-in
    Qwen3 4B Instruct grouped-query KV layout. They are not model outputs; they
    are fixture payloads for validating adapter reserve/copy/bind/release code.
    """

    if not isinstance(config, EngineProbeFixtureConfig):
        raise TypeError("config must be an EngineProbeFixtureConfig")
    output_dir_uri = _normalized_output_dir_uri(config.output_dir)
    output_dir = local_path(output_dir_uri)
    output_dir.mkdir(parents=True, exist_ok=True)

    layout = QWEN3_4B_INSTRUCT_PROFILE.to_layout(dtype=config.dtype, lora_id=config.lora_id)
    pack_uri = _output_uri(output_dir_uri, DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES["pack"])
    payload_uri = _output_uri(output_dir_uri, DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES["payload"])
    handoff_uri = _output_uri(output_dir_uri, DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES["handoff"])
    manifest_uri = _output_uri(output_dir_uri, DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES["manifest"])

    chunks = _fixture_pack_chunks(config, layout=layout)
    refs = write_kvpack(pack_uri, chunks, align_bytes=1)
    service = DocumentKVService(
        planner=CachePlanner(InMemoryManifestStore(refs)),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=0), reader=DiskRangeReader()),
        admission_queue=AdmissionQueue(max_pending_gpu_bytes=0),
    )
    ready = service.prepare_for_engine(
        _fixture_request(config),
        layout=layout,
        handle_uri=config.handle_uri,
        metadata=config.metadata,
        cache_method=CacheGenerationMethod.VANILLA_PREFILL,
        adapter_ids=config.adapter_ids,
        segmented=config.segmented,
    )
    adapter_request = build_engine_adapter_request(ready, spec=_adapter_spec(config.backend))
    handoff_path, payload_path = write_engine_adapter_handoff_bundle(
        adapter_request,
        handoff_uri,
        payload_uri=payload_uri,
        require_external_payload_uri=True,
    )
    pack_path = local_path(pack_uri)
    manifest_path = local_path(manifest_uri)
    result = EngineProbeFixtureResult(
        config=config,
        layout=layout,
        adapter_request=adapter_request,
        pack_uri=pack_uri,
        payload_uri=payload_uri,
        handoff_uri=handoff_uri,
        manifest_uri=manifest_uri,
        pack_path=pack_path,
        payload_path=payload_path,
        handoff_json=handoff_path,
        manifest_json=manifest_path,
        pack_sha256=_file_sha256(pack_path),
        payload_sha256=_file_sha256(payload_path),
        handoff_sha256=_file_sha256(handoff_path),
    )
    _write_fixture_manifest(result)
    _validate_written_fixture(result)
    return result


def engine_probe_fixture_result_to_record(result: EngineProbeFixtureResult) -> dict[str, Any]:
    """Return a JSON-serializable manifest for a generated fixture."""

    if not isinstance(result, EngineProbeFixtureResult):
        raise TypeError("result must be an EngineProbeFixtureResult")
    handle = result.adapter_request.ready_request.handle
    layout = handle.layout
    return {
        "record_type": ENGINE_PROBE_FIXTURE_RECORD_TYPE,
        "schema_version": ENGINE_PROBE_FIXTURE_SCHEMA_VERSION,
        "fixture": "qwen3-v1-engine-probe",
        "backend": result.adapter_request.backend.value,
        "payload_mode": result.adapter_request.payload_mode.value,
        "request_id": result.adapter_request.request_id,
        "model_id": layout.model_id,
        "lora_id": layout.lora_id,
        "layout_version": layout.layout_version,
        "dtype": layout.dtype,
        "storage_layout": layout.storage_layout.value,
        "shares_kv_storage": layout.shares_kv_storage,
        "num_layers": layout.num_layers,
        "num_query_heads": layout.num_query_heads,
        "num_kv_heads": layout.num_kv_heads,
        "head_size": layout.head_size,
        "kv_stride_bytes": layout.kv_stride_bytes,
        "block_size": layout.block_size,
        "bytes_per_token": layout.bytes_per_token,
        "total_tokens": handle.total_tokens,
        "total_bytes": handle.total_bytes,
        "total_blocks": build_engine_kv_injection_plan(
            read_engine_adapter_request_json(result.handoff_json, expected_backend=result.adapter_request.backend)
        ).total_blocks,
        "segment_count": len(handle.segments),
        "segments": [
            {
                "document_id": segment.document_id,
                "chunk_type": segment.chunk_type,
                "chunk_id": segment.chunk_id,
                "token_start": segment.token_start,
                "token_count": segment.token_count,
                "byte_start": segment.byte_start,
                "byte_length": segment.byte_length,
                "content_hash": segment.content_hash,
            }
            for segment in handle.segments
        ],
        "uris": {
            "pack": result.pack_uri,
            "payload": result.payload_uri,
            "handoff_json": result.handoff_uri,
            "manifest_json": result.manifest_uri,
        },
        "paths": {
            "pack": str(result.pack_path),
            "payload": str(result.payload_path),
            "handoff_json": str(result.handoff_json),
            "manifest_json": str(result.manifest_json),
        },
        "sha256": {
            "pack": result.pack_sha256,
            "payload": result.payload_sha256,
            "handoff_json": result.handoff_sha256,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a deterministic Qwen3 V1 engine-probe fixture.")
    parser.add_argument("--output-dir", required=True, help="Directory or local/DBFS/UC URI for fixture files.")
    parser.add_argument("--backend", choices=[backend.value for backend in ServingBackend], default=ServingBackend.VLLM)
    parser.add_argument(
        "--payload-mode",
        choices=[mode.value for mode in PayloadMode],
        default=PayloadMode.SEGMENTED,
    )
    parser.add_argument("--request-id", default=DEFAULT_ENGINE_PROBE_FIXTURE_REQUEST_ID)
    parser.add_argument("--document-id", default=DEFAULT_ENGINE_PROBE_FIXTURE_DOCUMENT_ID)
    parser.add_argument("--task-id", default=DEFAULT_ENGINE_PROBE_FIXTURE_TASK_ID)
    parser.add_argument("--prompt-template-version", default=DEFAULT_ENGINE_PROBE_FIXTURE_TEMPLATE_VERSION)
    parser.add_argument("--lora-id", default=QWEN3_4B_INSTRUCT_PROFILE.default_lora_id)
    parser.add_argument("--dtype", default=QWEN3_4B_INSTRUCT_PROFILE.default_dtype)
    parser.add_argument("--tokens-per-segment", type=int, default=DEFAULT_ENGINE_PROBE_FIXTURE_TOKENS_PER_SEGMENT)
    parser.add_argument(
        "--chunk-id",
        action="append",
        dest="chunk_ids",
        help="Document chunk id to include. May be repeated. Defaults to section-1 and section-2.",
    )
    parser.add_argument("--no-static", action="store_true", help="Do not include the document_static segment.")
    parser.add_argument("--static-chunk-id", default=DEFAULT_STATIC_CHUNK_ID)
    parser.add_argument("--adapter-id", action="append", dest="adapter_ids", help="Adapter id to bind; repeatable.")
    parser.add_argument("--handle-uri", help="Optional explicit handle URI in the adapter handoff.")
    parser.add_argument(
        "--metadata",
        action="append",
        metavar="KEY=VALUE",
        help="String metadata for the handle. document_kv.* and engine.* keys are reserved.",
    )
    parser.add_argument("--print-json", action="store_true", help="Print the fixture manifest JSON to stdout.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = write_qwen3_v1_engine_probe_fixture(
        EngineProbeFixtureConfig(
            output_dir=args.output_dir,
            backend=args.backend,
            payload_mode=args.payload_mode,
            request_id=args.request_id,
            document_id=args.document_id,
            task_id=args.task_id,
            prompt_template_version=args.prompt_template_version,
            lora_id=args.lora_id,
            dtype=args.dtype,
            tokens_per_segment=args.tokens_per_segment,
            chunk_ids=tuple(args.chunk_ids) if args.chunk_ids is not None else DEFAULT_ENGINE_PROBE_FIXTURE_CHUNK_IDS,
            include_static=not args.no_static,
            static_chunk_id=args.static_chunk_id,
            adapter_ids=tuple(args.adapter_ids or ()),
            handle_uri=args.handle_uri,
            metadata=_parse_metadata_items(args.metadata or ()),
        )
    )
    if args.print_json:
        print(json.dumps(engine_probe_fixture_result_to_record(result), indent=2, sort_keys=True))
    return 0


def _fixture_pack_chunks(config: EngineProbeFixtureConfig, *, layout: KVLayout) -> tuple[PackChunk, ...]:
    chunks: list[PackChunk] = []
    if config.include_static:
        chunks.append(
            _pack_chunk(
                config,
                layout=layout,
                chunk_type=DocumentChunkType.DOCUMENT_STATIC,
                chunk_id=config.static_chunk_id,
            )
        )
    for chunk_id in config.chunk_ids:
        chunks.append(
            _pack_chunk(
                config,
                layout=layout,
                chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
                chunk_id=chunk_id,
            )
        )
    return tuple(chunks)


def _pack_chunk(
    config: EngineProbeFixtureConfig,
    *,
    layout: KVLayout,
    chunk_type: DocumentChunkType,
    chunk_id: str,
) -> PackChunk:
    label = f"{config.request_id}|{config.document_id}|{chunk_type.value}|{chunk_id}|{layout.layout_version}"
    payload = _deterministic_payload(label, config.tokens_per_segment * layout.bytes_per_token)
    return PackChunk(
        key=KVCacheKey.for_document(
            model_id=layout.model_id,
            lora_id=layout.lora_id,
            prompt_template_version=config.prompt_template_version,
            document_id=config.document_id,
            chunk_type=chunk_type,
            chunk_id=chunk_id,
        ),
        payload=payload,
        token_count=config.tokens_per_segment,
        dtype=layout.dtype,
        layout_version=layout.layout_version,
        storage_layout=layout.storage_layout,
    )


def _fixture_request(config: EngineProbeFixtureConfig) -> DocumentKVRequest:
    return DocumentKVRequest.for_document_selection(
        request_id=config.request_id,
        task_id=config.task_id,
        model_id=QWEN3_4B_INSTRUCT_PROFILE.model_id,
        lora_id=config.lora_id,
        prompt_template_version=config.prompt_template_version,
        document_chunks={config.document_id: config.chunk_ids},
        include_static=config.include_static,
        static_chunk_id=config.static_chunk_id,
    )


def _adapter_spec(backend: ServingBackend):
    if backend == ServingBackend.VLLM:
        return vllm_adapter_spec()
    if backend == ServingBackend.SGLANG:
        return sglang_adapter_spec()
    raise ValueError(f"unsupported backend {backend!r}")


def _write_fixture_manifest(result: EngineProbeFixtureResult) -> None:
    result.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    result.manifest_json.write_text(
        json.dumps(engine_probe_fixture_result_to_record(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_written_fixture(result: EngineProbeFixtureResult) -> None:
    record = read_engine_adapter_request_json(
        result.handoff_json,
        expected_backend=result.adapter_request.backend,
        require_external_payload_uri=True,
    )
    plan = build_engine_kv_injection_plan(record, expected_backend=result.adapter_request.backend)
    if plan.total_tokens != result.total_tokens:
        raise ValueError("generated fixture plan total_tokens mismatch")
    if plan.total_bytes != result.total_bytes:
        raise ValueError("generated fixture plan total_bytes mismatch")
    payload = result.payload_path.read_bytes()
    actions = build_engine_kv_connector_actions(plan, _payload_for_connector_actions(result, payload))
    validate_engine_kv_connector_actions_record(engine_kv_connector_actions_to_record(actions))


def _payload_for_connector_actions(result: EngineProbeFixtureResult, payload: bytes) -> bytes | tuple[bytes, ...]:
    if result.adapter_request.payload_mode == PayloadMode.MERGED:
        return payload
    return tuple(
        payload[segment.byte_start : segment.byte_start + segment.byte_length]
        for segment in result.adapter_request.ready_request.handle.segments
    )


def _deterministic_payload(label: str, byte_count: int) -> bytes:
    _validate_positive_int(byte_count, field_name="byte_count")
    output = bytearray()
    counter = 0
    while len(output) < byte_count:
        output.extend(hashlib.sha256(f"{label}|{counter}".encode("utf-8")).digest())
        counter += 1
    return bytes(output[:byte_count])


def _output_uri(output_dir: str, filename: str) -> str:
    if output_dir.endswith("/"):
        return f"{output_dir}{filename}"
    if _has_uri_scheme(output_dir):
        return f"{output_dir}/{filename}"
    return str(Path(output_dir) / filename)


def _normalized_output_dir_uri(output_dir: str) -> str:
    if _has_uri_scheme(output_dir):
        return output_dir
    return str(Path(output_dir).expanduser().resolve(strict=False))


def _has_uri_scheme(value: str) -> bool:
    return ":" in value.split("/", maxsplit=1)[0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backend_from_value(value: ServingBackend | str) -> ServingBackend:
    try:
        return value if isinstance(value, ServingBackend) else ServingBackend(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported backend {value!r}") from exc


def _payload_mode_from_value(value: PayloadMode | str) -> PayloadMode:
    try:
        return value if isinstance(value, PayloadMode) else PayloadMode(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported payload_mode {value!r}") from exc


def _parse_metadata_items(items: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError("metadata entries must use KEY=VALUE syntax")
        metadata[key] = value
    return metadata


def _validated_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if not isinstance(value, str):
            raise TypeError("metadata values must be strings")
        if key.startswith(("document_kv.", "engine.")):
            raise ValueError("metadata keys must not use reserved document_kv.* or engine.* prefixes")
        normalized[key] = value
    return normalized


def _normalized_nonempty_string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    normalized = _normalized_string_tuple(values, field_name)
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one value")
    return normalized


def _normalized_string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    normalized = tuple(values)
    for value in normalized:
        _validate_nonempty_string(value, field_name=field_name)
        if "|" in value:
            raise ValueError(f"{field_name} entries must not contain '|'")
    return normalized


def _validate_nonempty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")
    if "|" in value:
        raise ValueError(f"{field_name} must not contain '|'")


def _validate_positive_int(value: object, *, field_name: str) -> None:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
