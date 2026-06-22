"""Attach Cachet handoff metadata to prepared V1 benchmark JSONL rows."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import string
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_kv_cache.benchmark_runner import _validate_benchmark_jsonl_record, load_benchmark_jsonl
from document_kv_cache.benchmarks import (
    BENCHMARK_CACHE_ARTIFACT_PREFIX,
    DEFAULT_V1_LORA_ID,
    DEFAULT_V1_MODEL_ID,
    DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    benchmark_cache_artifact_stem,
    benchmark_cache_request,
    benchmark_cache_source_document,
    validate_v1_dataset,
)
from document_kv_cache.engine_protocol import KVLayout, KVStorageLayout
from document_kv_cache.dataset_prep import write_v1_jsonl
from document_kv_cache.engine_adapters import (
    ServingBackend,
    build_engine_adapter_request,
    read_engine_adapter_request_json,
    sglang_adapter_spec,
    validate_engine_adapter_request_record,
    vllm_adapter_spec,
)
from document_kv_cache.engine_probe import _validate_local_payload_uri, write_engine_adapter_handoff_bundle
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.model_profiles import layout_for_model
from document_kv_cache.models import CacheGenerationMethod, ChunkRef
from document_kv_cache.storage import local_path
from document_kv_cache.workflow import (
    CacheBuildConfig,
    CacheGenerationResult,
    DocumentKVWorkflow,
    KVChunkGenerator,
)


BENCHMARK_HANDOFF_MANIFEST_RECORD_TYPE = "document_kv.benchmark_handoffs.v1"
BENCHMARK_HANDOFF_MANIFEST_SCHEMA_VERSION = 1
_TEMPLATE_FIELDS = frozenset({"dataset", "example_id"})
_BUNDLE_TEMPLATE_FIELDS = frozenset({"dataset", "example_id", "artifact_stem"})
_DEFAULT_BUNDLE_SHARD_FILENAME = "cachet-benchmark.kvpack"
_MANUAL_LAYOUT_REQUIRED_FIELDS = ("layout_version", "dtype", "num_layers", "block_size", "bytes_per_token")
_MANUAL_LAYOUT_TRIGGER_FIELDS = (
    "num_layers",
    "bytes_per_token",
    "num_query_heads",
    "num_kv_heads",
    "head_size",
)
_MANIFEST_KEYS = frozenset({"record_type", "schema_version", "entries"})
_ENTRY_KEYS = frozenset(
    {
        "dataset",
        "example_id",
        "request_id",
        "handoff_json",
        "handoff_record",
        "payload_uri",
    }
)

__all__ = [
    "BENCHMARK_HANDOFF_MANIFEST_RECORD_TYPE",
    "BENCHMARK_HANDOFF_MANIFEST_SCHEMA_VERSION",
    "BenchmarkHandoffEntry",
    "BenchmarkHandoffManifest",
    "BenchmarkHandoffBundleResult",
    "build_benchmark_handoff_manifest_from_jsonl",
    "benchmark_handoff_manifest_from_record",
    "benchmark_handoff_manifest_to_record",
    "generate_benchmark_handoff_bundles",
    "enrich_benchmark_jsonl_with_handoffs",
    "enrich_benchmark_records_with_handoffs",
    "load_benchmark_kv_chunk_generator",
    "read_benchmark_handoff_manifest_json",
    "write_benchmark_handoff_manifest_json",
    "bundle_main",
    "manifest_main",
    "main",
]


@dataclass(frozen=True, slots=True)
class BenchmarkHandoffEntry:
    """One Cachet handoff source keyed to a benchmark example."""

    dataset: str
    example_id: str
    request_id: str
    handoff_json: str | None = None
    handoff_record: Mapping[str, Any] | None = None
    payload_uri: str | None = None

    def __post_init__(self) -> None:
        validate_v1_dataset(_required_string(self.dataset, field_name="dataset"))
        _required_string(self.example_id, field_name="example_id")
        _required_string(self.request_id, field_name="request_id")
        handoff_json = self.handoff_json
        handoff_record = self.handoff_record
        if handoff_json is None and handoff_record is None:
            raise ValueError("handoff entry must include handoff_json or handoff_record")
        if handoff_json is not None and handoff_record is not None:
            raise ValueError("handoff entry must include only one of handoff_json or handoff_record")
        if handoff_json is not None:
            object.__setattr__(
                self,
                "handoff_json",
                _required_string(handoff_json, field_name="handoff_json"),
            )
        if handoff_record is not None:
            normalized_record = _json_object(handoff_record, field_name="handoff_record")
            validate_engine_adapter_request_record(
                normalized_record,
                require_external_payload_uri=self.payload_uri is None,
            )
            if normalized_record.get("request_id") != self.request_id:
                raise ValueError("handoff_record.request_id must match request_id")
            if self.payload_uri is None:
                _validate_handoff_record_payload_uri(normalized_record, field_name="handoff_record.payload_source.uri")
            object.__setattr__(self, "handoff_record", normalized_record)
        if self.payload_uri is not None:
            object.__setattr__(
                self,
                "payload_uri",
                _runtime_payload_uri(self.payload_uri, field_name="payload_uri"),
            )

    @property
    def key(self) -> tuple[str, str]:
        return (self.dataset, self.example_id)

    def kv_transfer_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            DOCUMENT_KV_REQUEST_ID_PARAM: self.request_id,
            DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM: "logical",
        }
        if self.handoff_json is not None:
            params[DOCUMENT_KV_HANDOFF_JSON_PARAM] = self.handoff_json
        else:
            assert self.handoff_record is not None
            params[DOCUMENT_KV_HANDOFF_RECORD_PARAM] = dict(self.handoff_record)
        if self.payload_uri is not None:
            params[DOCUMENT_KV_PAYLOAD_URI_PARAM] = self.payload_uri
        return params


@dataclass(frozen=True, slots=True)
class BenchmarkHandoffManifest:
    """Validated manifest of Cachet handoffs for prepared benchmark examples."""

    entries: tuple[BenchmarkHandoffEntry, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("handoff manifest entries must contain at least one item")
        for index, entry in enumerate(entries):
            if not isinstance(entry, BenchmarkHandoffEntry):
                raise TypeError(f"entries[{index}] must be a BenchmarkHandoffEntry")
        _entries_by_key(entries)
        object.__setattr__(self, "entries", entries)


@dataclass(frozen=True, slots=True)
class BenchmarkHandoffBundleResult:
    """Artifacts produced when prepared benchmark rows are materialized as Cachet handoffs."""

    manifest: BenchmarkHandoffManifest
    cache_generation: CacheGenerationResult
    shard_uri: str
    handoff_json_paths: tuple[str, ...]
    payload_uris: tuple[str, ...]
    cache_refs: tuple[ChunkRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, BenchmarkHandoffManifest):
            raise TypeError("manifest must be a BenchmarkHandoffManifest")
        if not isinstance(self.cache_generation, CacheGenerationResult):
            raise TypeError("cache_generation must be a CacheGenerationResult")
        object.__setattr__(self, "shard_uri", _required_string(self.shard_uri, field_name="shard_uri"))
        handoff_json_paths = _string_tuple(self.handoff_json_paths, field_name="handoff_json_paths")
        payload_uris = _string_tuple(self.payload_uris, field_name="payload_uris")
        cache_refs = tuple(self.cache_refs)
        for index, ref in enumerate(cache_refs):
            if not isinstance(ref, ChunkRef):
                raise TypeError(f"cache_refs[{index}] must be a ChunkRef")
        if len(handoff_json_paths) != len(self.manifest.entries):
            raise ValueError("handoff_json_paths must match manifest entry count")
        if len(payload_uris) != len(self.manifest.entries):
            raise ValueError("payload_uris must match manifest entry count")
        object.__setattr__(self, "handoff_json_paths", handoff_json_paths)
        object.__setattr__(self, "payload_uris", payload_uris)
        object.__setattr__(self, "cache_refs", cache_refs)


@dataclass(frozen=True, slots=True)
class _BenchmarkHandoffBundleTarget:
    example: Any
    request: Any
    handoff_json: str
    payload_uri: str


def benchmark_handoff_manifest_from_record(record: Mapping[str, Any]) -> BenchmarkHandoffManifest:
    """Parse a closed JSON manifest record into a validated manifest."""

    if not isinstance(record, Mapping):
        raise TypeError("handoff manifest must be an object")
    unexpected = sorted(str(key) for key in record if key not in _MANIFEST_KEYS)
    if unexpected:
        raise ValueError(f"handoff manifest has unsupported keys: {unexpected}")
    if record.get("record_type") != BENCHMARK_HANDOFF_MANIFEST_RECORD_TYPE:
        raise ValueError(f"record_type must be {BENCHMARK_HANDOFF_MANIFEST_RECORD_TYPE!r}")
    if record.get("schema_version") != BENCHMARK_HANDOFF_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {BENCHMARK_HANDOFF_MANIFEST_SCHEMA_VERSION}")
    raw_entries = record.get("entries")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes, bytearray)):
        raise ValueError("entries must be an array")
    entries = tuple(
        _entry_from_record(entry, index=index)
        for index, entry in enumerate(raw_entries)
    )
    return BenchmarkHandoffManifest(entries=entries)


def benchmark_handoff_manifest_to_record(manifest: BenchmarkHandoffManifest) -> dict[str, Any]:
    """Serialize a validated manifest to its stable JSON record."""

    if not isinstance(manifest, BenchmarkHandoffManifest):
        raise TypeError("manifest must be a BenchmarkHandoffManifest")
    return {
        "record_type": BENCHMARK_HANDOFF_MANIFEST_RECORD_TYPE,
        "schema_version": BENCHMARK_HANDOFF_MANIFEST_SCHEMA_VERSION,
        "entries": [
            _entry_to_record(entry)
            for entry in manifest.entries
        ],
    }


def read_benchmark_handoff_manifest_json(path: str | Path) -> BenchmarkHandoffManifest:
    """Read a benchmark handoff manifest JSON file."""

    try:
        record = json.loads(local_path(str(path)).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"handoff manifest JSON is invalid: {exc.msg}") from exc
    return benchmark_handoff_manifest_from_record(record)


def write_benchmark_handoff_manifest_json(
    manifest: BenchmarkHandoffManifest,
    path: str | Path,
) -> None:
    """Write a stable benchmark handoff manifest JSON file."""

    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(benchmark_handoff_manifest_to_record(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_benchmark_handoff_manifest_from_jsonl(
    input_jsonl: str | Path,
    *,
    handoff_json_template: str,
    dataset: str | None = None,
    payload_uri_template: str | None = None,
    expected_backend: ServingBackend | str | None = None,
    allow_missing: bool = False,
) -> BenchmarkHandoffManifest:
    """Build a strict handoff manifest by joining benchmark rows to handoff JSON files."""

    default_dataset = _default_dataset(dataset)
    entries: list[BenchmarkHandoffEntry] = []
    seen_record_keys: set[tuple[str, str]] = set()
    missing_keys: list[tuple[str, str]] = []
    for line_number, record in _iter_jsonl(input_jsonl):
        key = _record_key(record, default_dataset=default_dataset, line_number=line_number)
        if key in seen_record_keys:
            raise ValueError(f"Duplicate benchmark input rows for {_format_keys((key,))}")
        seen_record_keys.add(key)
        row_dataset, example_id = key
        handoff_json = _format_benchmark_handoff_template(
            handoff_json_template,
            dataset=row_dataset,
            example_id=example_id,
            field_name="handoff_json_template",
        )
        try:
            handoff_record = read_engine_adapter_request_json(
                handoff_json,
                expected_backend=expected_backend,
                require_external_payload_uri=payload_uri_template is None,
            )
        except FileNotFoundError as exc:
            if allow_missing:
                missing_keys.append(key)
                continue
            raise ValueError(f"Missing handoff JSON for {_format_keys((key,))}: {handoff_json}") from exc
        payload_uri = (
            _format_benchmark_handoff_template(
                payload_uri_template,
                dataset=row_dataset,
                example_id=example_id,
                field_name="payload_uri_template",
            )
            if payload_uri_template is not None
            else _payload_uri_from_handoff_record(handoff_record)
        )
        entries.append(
            BenchmarkHandoffEntry(
                dataset=row_dataset,
                example_id=example_id,
                request_id=_required_string(handoff_record.get("request_id"), field_name="handoff_record.request_id"),
                handoff_json=handoff_json,
                payload_uri=payload_uri,
            )
        )
    if missing_keys and not allow_missing:
        raise ValueError("Missing handoff JSON for " + _format_keys(missing_keys))
    return BenchmarkHandoffManifest(entries=tuple(entries))


def generate_benchmark_handoff_bundles(
    input_jsonl: str | Path,
    *,
    output_dir: str | Path,
    generator: KVChunkGenerator,
    layout: KVLayout,
    dataset: str | None = None,
    limit: int | None = None,
    backend: ServingBackend | str = ServingBackend.VLLM,
    manifest_json: str | Path | None = None,
    shard_uri: str | Path | None = None,
    handoff_json_template: str | None = None,
    payload_uri_template: str | None = None,
    model_id: str | None = None,
    lora_id: str | None = None,
    prompt_template_version: str = DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    cache_method: CacheGenerationMethod | str = CacheGenerationMethod.VANILLA_PREFILL,
    storage_layout: KVStorageLayout | str | None = None,
    segmented: bool = False,
    align_bytes: int = 4096,
    kv_gpu_bytes_per_payload_byte: float | None = None,
    prefix: str = BENCHMARK_CACHE_ARTIFACT_PREFIX,
    disk_root: str | Path | None = None,
    uc_volume_root: str | Path | None = None,
) -> BenchmarkHandoffBundleResult:
    """Generate Cachet handoff JSON/payload bundles for prepared benchmark rows.

    The caller supplies the KV chunk generator. The CLI intentionally has no
    built-in fake generator so benchmark evidence cannot silently use placeholder
    bytes.
    """

    _validate_generator(generator)
    if not isinstance(layout, KVLayout):
        raise TypeError("layout must be a KVLayout")
    layout.validate()
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    backend = _backend_from_value(backend)
    output_base = _bundle_output_base(output_dir)
    shard_uri_text = _default_bundle_shard_uri(output_base, shard_uri=shard_uri)
    resolved_model_id = _optional_string(model_id, default=layout.model_id, field_name="model_id")
    resolved_lora_id = _optional_string(lora_id, default=layout.lora_id, field_name="lora_id")
    resolved_template_version = _required_string(
        prompt_template_version,
        field_name="prompt_template_version",
    )
    if resolved_model_id != layout.model_id:
        raise ValueError("model_id must match layout.model_id")
    if resolved_lora_id != layout.lora_id:
        raise ValueError("lora_id must match layout.lora_id")
    config = CacheBuildConfig(
        model_id=resolved_model_id,
        lora_id=resolved_lora_id,
        prompt_template_version=resolved_template_version,
        dtype=layout.dtype,
        layout_version=layout.layout_version,
        cache_method=cache_method,
        storage_layout=layout.storage_layout if storage_layout is None else storage_layout,
    )
    if config.storage_layout != layout.storage_layout:
        raise ValueError("storage_layout must match layout.storage_layout")

    examples = load_benchmark_jsonl(
        input_jsonl,
        dataset=dataset,
        limit=limit,
        require_dataset=dataset is not None,
    )
    if not examples:
        raise ValueError("input_jsonl must contain at least one benchmark row")
    _validate_unique_benchmark_examples(examples)
    source_documents = tuple(
        benchmark_cache_source_document(example, prefix=prefix)
        for example in examples
    )
    bundle_targets = _benchmark_handoff_bundle_targets(
        examples,
        output_base=output_base,
        handoff_json_template=handoff_json_template,
        payload_uri_template=payload_uri_template,
        model_id=resolved_model_id,
        lora_id=resolved_lora_id,
        prompt_template_version=resolved_template_version,
        prefix=prefix,
    )
    _validate_bundle_output_artifact_paths(
        bundle_targets,
        input_jsonl=input_jsonl,
        manifest_json=manifest_json,
        shard_uri=shard_uri_text,
        disk_root=disk_root,
        uc_volume_root=uc_volume_root,
    )

    manifest_store = InMemoryManifestStore()
    workflow = DocumentKVWorkflow.with_storage(
        manifest=manifest_store,
        disk_root=disk_root,
        uc_volume_root=uc_volume_root,
    )
    cache_generation = workflow.generate_cache(
        documents=source_documents,
        generator=generator,
        config=config,
        shard_uri=shard_uri_text,
        align_bytes=align_bytes,
    )

    entries: list[BenchmarkHandoffEntry] = []
    handoff_json_paths: list[str] = []
    payload_uris: list[str] = []
    adapter_spec = _adapter_spec(backend)
    for target in bundle_targets:
        ready = workflow.prepare_for_engine(
            target.request,
            layout=layout,
            metadata={
                "cachet.benchmark.dataset": target.example.dataset,
                "cachet.benchmark.example_id": target.example.example_id,
            },
            cache_method=cache_generation.cache_method,
            training_artifacts=cache_generation.training_artifacts,
            segmented=segmented,
            kv_gpu_bytes_per_payload_byte=kv_gpu_bytes_per_payload_byte,
        )
        adapter_request = build_engine_adapter_request(ready, spec=adapter_spec)
        write_engine_adapter_handoff_bundle(
            adapter_request,
            target.handoff_json,
            payload_uri=target.payload_uri,
            require_external_payload_uri=True,
        )
        entries.append(
            BenchmarkHandoffEntry(
                dataset=target.example.dataset,
                example_id=target.example.example_id,
                request_id=target.request.request_id,
                handoff_json=target.handoff_json,
                payload_uri=target.payload_uri,
            )
        )
        handoff_json_paths.append(target.handoff_json)
        payload_uris.append(target.payload_uri)

    manifest = BenchmarkHandoffManifest(entries=tuple(entries))
    if manifest_json is not None:
        write_benchmark_handoff_manifest_json(manifest, manifest_json)
    return BenchmarkHandoffBundleResult(
        manifest=manifest,
        cache_generation=cache_generation,
        shard_uri=shard_uri_text,
        handoff_json_paths=tuple(handoff_json_paths),
        payload_uris=tuple(payload_uris),
        cache_refs=tuple(cache_generation.refs),
    )


def load_benchmark_kv_chunk_generator(factory_path: str) -> KVChunkGenerator:
    """Load a user-provided ``KVChunkGenerator`` factory from ``module:callable``."""

    module_name, attribute_name = _split_factory_path(
        _required_string(factory_path, field_name="generator_factory")
    )
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    candidate = factory
    if isinstance(candidate, type):
        candidate = candidate()
    elif not hasattr(candidate, "generate"):
        if not callable(candidate):
            raise TypeError("generator_factory must reference a KVChunkGenerator or zero-argument factory")
        candidate = candidate()
    _validate_generator(candidate)
    return candidate


def enrich_benchmark_jsonl_with_handoffs(
    input_jsonl: str | Path,
    manifest_json: str | Path,
    output_jsonl: str | Path,
    *,
    dataset: str | None = None,
    overwrite: bool = False,
    allow_missing: bool = False,
    allow_unmatched: bool = False,
) -> int:
    """Attach manifest handoffs to benchmark JSONL and write validated output."""

    manifest = read_benchmark_handoff_manifest_json(manifest_json)
    records = tuple(record for _, record in _iter_jsonl(input_jsonl))
    enriched = enrich_benchmark_records_with_handoffs(
        records,
        manifest,
        dataset=dataset,
        overwrite=overwrite,
        allow_missing=allow_missing,
        allow_unmatched=allow_unmatched,
    )
    return write_v1_jsonl(enriched, output_jsonl)


def enrich_benchmark_records_with_handoffs(
    records: Iterable[Mapping[str, Any]],
    manifest: BenchmarkHandoffManifest,
    *,
    dataset: str | None = None,
    overwrite: bool = False,
    allow_missing: bool = False,
    allow_unmatched: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Return benchmark records enriched with manifest-provided handoffs."""

    default_dataset = _default_dataset(dataset)
    by_key = _entries_by_key(manifest.entries)
    used_keys: set[tuple[str, str]] = set()
    seen_record_keys: set[tuple[str, str]] = set()
    missing_keys: list[tuple[str, str]] = []
    enriched_records: list[dict[str, Any]] = []

    for line_number, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"Record line {line_number} must be an object")
        key = _record_key(record, default_dataset=default_dataset, line_number=line_number)
        if key in seen_record_keys:
            raise ValueError(f"Duplicate benchmark input rows for {_format_keys((key,))}")
        seen_record_keys.add(key)
        entry = by_key.get(key)
        if entry is None:
            missing_keys.append(key)
            enriched_records.append(dict(record))
            continue
        if "kv_transfer_params" in record and not overwrite:
            raise ValueError(
                f"Record line {line_number} already has kv_transfer_params; pass overwrite=True to replace it"
            )
        used_keys.add(key)
        enriched = dict(record)
        if "dataset" not in enriched and default_dataset is not None:
            enriched["dataset"] = default_dataset
        enriched["kv_transfer_params"] = entry.kv_transfer_params()
        _validate_benchmark_jsonl_record(
            enriched,
            dataset=default_dataset,
            record_index=line_number,
            require_dataset=default_dataset is not None,
        )
        enriched_records.append(enriched)

    if missing_keys and not allow_missing:
        raise ValueError("Missing handoff manifest entries for " + _format_keys(missing_keys))
    unmatched_keys = sorted(set(by_key).difference(used_keys))
    if unmatched_keys and not allow_unmatched:
        raise ValueError("Unmatched handoff manifest entries for " + _format_keys(unmatched_keys))
    return tuple(enriched_records)


def bundle_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Cachet handoff bundles for prepared V1 benchmark JSONL.")
    parser.add_argument("--input-jsonl", required=True, help="Prepared benchmark JSONL whose rows need handoffs.")
    parser.add_argument("--output-dir", required=True, help="Local output directory for default bundle artifacts.")
    parser.add_argument(
        "--output-manifest-json",
        "--manifest-json",
        dest="manifest_json",
        required=True,
        help="Benchmark handoff manifest JSON output path.",
    )
    parser.add_argument(
        "--generator-factory",
        required=True,
        help="User-provided KVChunkGenerator object or zero-argument factory, as module:attribute.",
    )
    parser.add_argument("--dataset", help="Default dataset for JSONL rows without a dataset field.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of input rows to process.")
    parser.add_argument("--backend", choices=[backend.value for backend in ServingBackend], default=ServingBackend.VLLM.value)
    parser.add_argument("--shard-uri", help="Intermediate kvpack shard URI. Defaults under --output-dir.")
    parser.add_argument(
        "--handoff-json-template",
        help="Handoff JSON path template. Supports {dataset}, {example_id}, and {artifact_stem}.",
    )
    parser.add_argument(
        "--payload-uri-template",
        help="Payload URI template. Supports {dataset}, {example_id}, and {artifact_stem}.",
    )
    parser.add_argument("--model-id", default=DEFAULT_V1_MODEL_ID)
    parser.add_argument("--lora-id", default=DEFAULT_V1_LORA_ID)
    parser.add_argument("--prompt-template-version", default=DEFAULT_V1_PROMPT_TEMPLATE_VERSION)
    parser.add_argument("--layout-version", help="Override the built-in model profile layout version.")
    parser.add_argument("--dtype", help="Override the built-in model profile dtype.")
    parser.add_argument("--num-layers", type=int, help="Manual layout layer count for custom models.")
    parser.add_argument("--block-size", type=int, help="Override the built-in model profile block size.")
    parser.add_argument("--bytes-per-token", type=int, help="Manual layout bytes per token for custom models.")
    parser.add_argument("--num-query-heads", type=int)
    parser.add_argument("--num-kv-heads", type=int)
    parser.add_argument("--head-size", type=int)
    parser.add_argument("--kv-stride-bytes", type=int)
    parser.add_argument("--shares-kv-storage", action="store_true")
    parser.add_argument("--storage-layout", choices=[layout.value for layout in KVStorageLayout])
    parser.add_argument("--segmented", action="store_true", help="Write segmented payload handoffs.")
    parser.add_argument("--align-bytes", type=int, default=4096)
    parser.add_argument("--kv-gpu-bytes-per-payload-byte", type=float)
    parser.add_argument("--prefix", default=BENCHMARK_CACHE_ARTIFACT_PREFIX)
    parser.add_argument("--disk-root", help="Optional disk root for relative kvpack shard URIs.")
    parser.add_argument("--uc-volume-root", help="Optional UC Volume root for relative kvpack shard URIs.")
    try:
        args = parser.parse_args(argv)
        layout = _bundle_layout_from_args(args)
        generator = load_benchmark_kv_chunk_generator(args.generator_factory)
        result = generate_benchmark_handoff_bundles(
            args.input_jsonl,
            output_dir=args.output_dir,
            generator=generator,
            layout=layout,
            dataset=args.dataset,
            limit=args.limit,
            backend=args.backend,
            manifest_json=args.manifest_json,
            shard_uri=args.shard_uri,
            handoff_json_template=args.handoff_json_template,
            payload_uri_template=args.payload_uri_template,
            model_id=args.model_id,
            lora_id=args.lora_id,
            prompt_template_version=args.prompt_template_version,
            storage_layout=args.storage_layout,
            segmented=args.segmented,
            align_bytes=args.align_bytes,
            kv_gpu_bytes_per_payload_byte=args.kv_gpu_bytes_per_payload_byte,
            prefix=args.prefix,
            disk_root=args.disk_root,
            uc_volume_root=args.uc_volume_root,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "entries": len(result.manifest.entries),
                    "cache_refs": len(result.cache_refs),
                    "shard_uri": result.shard_uri,
                    "manifest_json": args.manifest_json,
                },
                sort_keys=True,
            )
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


def _bundle_layout_from_args(args: argparse.Namespace) -> KVLayout:
    if _uses_manual_layout(args):
        missing = tuple(field for field in _MANUAL_LAYOUT_REQUIRED_FIELDS if getattr(args, field) is None)
        if missing:
            required_flags = ", ".join(_field_to_flag(field) for field in _MANUAL_LAYOUT_REQUIRED_FIELDS)
            missing_flags = ", ".join(_field_to_flag(field) for field in missing)
            raise ValueError(
                "Manual benchmark layout requires "
                f"{required_flags}; missing {missing_flags}. "
                "Omit manual geometry to use the built-in model profile."
            )
        return KVLayout(
            model_id=args.model_id,
            lora_id=args.lora_id,
            layout_version=args.layout_version,
            dtype=args.dtype,
            num_layers=args.num_layers,
            block_size=args.block_size,
            bytes_per_token=args.bytes_per_token,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            kv_stride_bytes=args.kv_stride_bytes,
            shares_kv_storage=args.shares_kv_storage,
            storage_layout=args.storage_layout,
        )
    try:
        return layout_for_model(
            args.model_id,
            dtype=args.dtype,
            lora_id=args.lora_id,
            block_size=args.block_size,
            layout_version=args.layout_version,
            kv_stride_bytes=args.kv_stride_bytes,
            shares_kv_storage=True if args.shares_kv_storage else None,
            storage_layout=args.storage_layout,
        )
    except KeyError as exc:
        required_flags = ", ".join(_field_to_flag(field) for field in _MANUAL_LAYOUT_REQUIRED_FIELDS)
        raise ValueError(
            f"Unknown model profile {args.model_id!r}; pass complete manual layout flags: {required_flags}"
        ) from exc


def _uses_manual_layout(args: argparse.Namespace) -> bool:
    return any(getattr(args, field) is not None for field in _MANUAL_LAYOUT_TRIGGER_FIELDS)


def _field_to_flag(field: str) -> str:
    return "--" + field.replace("_", "-")


def manifest_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Cachet benchmark handoff manifest from handoff JSON files.")
    parser.add_argument("--input-jsonl", required=True, help="Prepared benchmark JSONL whose rows need handoffs.")
    parser.add_argument(
        "--handoff-json-template",
        required=True,
        help="Format template for handoff JSON paths. Supports {dataset} and {example_id}.",
    )
    parser.add_argument(
        "--output-json",
        "--output-manifest-json",
        dest="output_json",
        required=True,
        help="Benchmark handoff manifest JSON output path.",
    )
    parser.add_argument("--dataset", help="Default dataset for JSONL rows without a dataset field.")
    parser.add_argument(
        "--payload-uri-template",
        help="Optional payload URI override template. Supports {dataset} and {example_id}.",
    )
    parser.add_argument("--expected-backend", choices=[backend.value for backend in ServingBackend])
    parser.add_argument("--allow-missing", action="store_true", help="Skip rows whose handoff JSON is absent.")
    try:
        args = parser.parse_args(argv)
        manifest = build_benchmark_handoff_manifest_from_jsonl(
            args.input_jsonl,
            handoff_json_template=args.handoff_json_template,
            dataset=args.dataset,
            payload_uri_template=args.payload_uri_template,
            expected_backend=args.expected_backend,
            allow_missing=args.allow_missing,
        )
        write_benchmark_handoff_manifest_json(manifest, args.output_json)
        print(json.dumps({"ok": True, "entries": len(manifest.entries)}, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attach Cachet handoff metadata to V1 benchmark JSONL.")
    parser.add_argument("--input-jsonl", required=True, help="Prepared benchmark JSONL to enrich.")
    parser.add_argument(
        "--manifest-json",
        "--handoff-manifest-json",
        dest="manifest_json",
        required=True,
        help="Benchmark handoff manifest JSON.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Validated enriched JSONL output path.")
    parser.add_argument("--dataset", help="Default dataset for JSONL rows without a dataset field.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing kv_transfer_params.")
    parser.add_argument("--allow-missing", action="store_true", help="Leave rows without manifest entries unchanged.")
    parser.add_argument("--allow-unmatched", action="store_true", help="Permit manifest entries unused by the input JSONL.")
    try:
        args = parser.parse_args(argv)
        count = enrich_benchmark_jsonl_with_handoffs(
            args.input_jsonl,
            args.manifest_json,
            args.output_jsonl,
            dataset=args.dataset,
            overwrite=args.overwrite,
            allow_missing=args.allow_missing,
            allow_unmatched=args.allow_unmatched,
        )
        print(json.dumps({"ok": True, "records": count}, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


def _entry_from_record(record: object, *, index: int) -> BenchmarkHandoffEntry:
    if not isinstance(record, Mapping):
        raise ValueError(f"entries[{index}] must be an object")
    unexpected = sorted(str(key) for key in record if key not in _ENTRY_KEYS)
    if unexpected:
        raise ValueError(f"entries[{index}] has unsupported keys: {unexpected}")
    return BenchmarkHandoffEntry(
        dataset=_required_string(record.get("dataset"), field_name=f"entries[{index}].dataset"),
        example_id=_required_string(record.get("example_id"), field_name=f"entries[{index}].example_id"),
        request_id=_required_string(record.get("request_id"), field_name=f"entries[{index}].request_id"),
        handoff_json=record.get("handoff_json"),
        handoff_record=record.get("handoff_record"),
        payload_uri=record.get("payload_uri"),
    )


def _entry_to_record(entry: BenchmarkHandoffEntry) -> dict[str, Any]:
    record = {
        "dataset": entry.dataset,
        "example_id": entry.example_id,
        "request_id": entry.request_id,
    }
    if entry.handoff_json is not None:
        record["handoff_json"] = entry.handoff_json
    if entry.handoff_record is not None:
        record["handoff_record"] = dict(entry.handoff_record)
    if entry.payload_uri is not None:
        record["payload_uri"] = entry.payload_uri
    return record


def _entries_by_key(entries: Iterable[BenchmarkHandoffEntry]) -> dict[tuple[str, str], BenchmarkHandoffEntry]:
    by_key: dict[tuple[str, str], BenchmarkHandoffEntry] = {}
    duplicates: list[tuple[str, str]] = []
    for entry in entries:
        if entry.key in by_key:
            duplicates.append(entry.key)
            continue
        by_key[entry.key] = entry
    if duplicates:
        raise ValueError("Duplicate handoff manifest entries for " + _format_keys(duplicates))
    return by_key


def _payload_uri_from_handoff_record(record: Mapping[str, Any]) -> str:
    payload_source = record.get("payload_source")
    if not isinstance(payload_source, Mapping):
        raise ValueError("handoff_record.payload_source must be an object")
    return _runtime_payload_uri(payload_source.get("uri"), field_name="handoff_record.payload_source.uri")


def _format_benchmark_handoff_template(
    template: str,
    *,
    dataset: str,
    example_id: str,
    field_name: str,
) -> str:
    return _format_benchmark_template(
        template,
        field_name=field_name,
        supported_fields=_TEMPLATE_FIELDS,
        values={"dataset": dataset, "example_id": example_id},
    )


def _validate_benchmark_handoff_template(template: str, *, field_name: str) -> None:
    _validate_benchmark_template(
        template,
        field_name=field_name,
        supported_fields=_TEMPLATE_FIELDS,
    )


def _format_benchmark_bundle_template(
    template: str,
    *,
    dataset: str,
    example_id: str,
    artifact_stem: str,
    field_name: str,
) -> str:
    return _format_benchmark_template(
        template,
        field_name=field_name,
        supported_fields=_BUNDLE_TEMPLATE_FIELDS,
        values={"dataset": dataset, "example_id": example_id, "artifact_stem": artifact_stem},
    )


def _format_benchmark_template(
    template: str,
    *,
    field_name: str,
    supported_fields: frozenset[str],
    values: Mapping[str, str],
) -> str:
    _validate_benchmark_template(template, field_name=field_name, supported_fields=supported_fields)
    try:
        value = template.format(**values)
    except KeyError as exc:
        raise ValueError(f"{field_name} has unsupported placeholder {exc.args[0]!r}") from exc
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{field_name} is not a valid format template: {exc}") from exc
    return _required_string(value, field_name=field_name)


def _validate_benchmark_template(
    template: str,
    *,
    field_name: str,
    supported_fields: frozenset[str],
) -> None:
    try:
        parts = tuple(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid format template: {exc}") from exc
    for _literal_text, placeholder, format_spec, conversion in parts:
        if placeholder is None:
            continue
        if placeholder not in supported_fields:
            supported = " and ".join(f"{{{field}}}" for field in sorted(supported_fields))
            raise ValueError(
                f"{field_name} supports only {supported} placeholders; "
                f"got {{{placeholder}}}"
            )
        if format_spec or conversion is not None:
            supported = " and ".join(f"{{{field}}}" for field in sorted(supported_fields))
            raise ValueError(
                f"{field_name} supports only plain {supported} placeholders"
            )


def _iter_jsonl(path: str | Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with local_path(str(path)).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Benchmark JSONL line {line_number} is not valid JSON: {exc.msg}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"Benchmark JSONL line {line_number} must be an object")
            yield line_number, record


def _default_dataset(dataset: str | None) -> str | None:
    if dataset is None:
        return None
    validate_v1_dataset(_required_string(dataset, field_name="dataset"))
    return dataset


def _record_key(
    record: Mapping[str, Any],
    *,
    default_dataset: str | None,
    line_number: int,
) -> tuple[str, str]:
    dataset = record.get("dataset", default_dataset)
    if dataset != default_dataset and default_dataset is not None:
        raise ValueError(
            f"Record line {line_number} dataset {dataset!r} does not match expected {default_dataset!r}"
        )
    validate_v1_dataset(_required_string(dataset, field_name=f"Record line {line_number} dataset"))
    example_id = _required_string(
        record.get("example_id"),
        field_name=f"Record line {line_number} example_id",
    )
    return dataset, example_id


def _format_keys(keys: Iterable[tuple[str, str]]) -> str:
    return ", ".join(f"{dataset}/{example_id}" for dataset, example_id in sorted(set(keys)))


def _benchmark_handoff_bundle_targets(
    examples: Sequence[Any],
    *,
    output_base: Path,
    handoff_json_template: str | None,
    payload_uri_template: str | None,
    model_id: str,
    lora_id: str,
    prompt_template_version: str,
    prefix: str,
) -> tuple[_BenchmarkHandoffBundleTarget, ...]:
    targets: list[_BenchmarkHandoffBundleTarget] = []
    for example in examples:
        artifact_stem = benchmark_cache_artifact_stem(example, prefix=prefix)
        request = benchmark_cache_request(
            example,
            model_id=model_id,
            lora_id=lora_id,
            prompt_template_version=prompt_template_version,
            prefix=prefix,
        )
        targets.append(
            _BenchmarkHandoffBundleTarget(
                example=example,
                request=request,
                handoff_json=_bundle_handoff_json(
                    output_base,
                    handoff_json_template=handoff_json_template,
                    dataset=example.dataset,
                    example_id=example.example_id,
                    artifact_stem=artifact_stem,
                ),
                payload_uri=_bundle_payload_uri(
                    output_base,
                    payload_uri_template=payload_uri_template,
                    dataset=example.dataset,
                    example_id=example.example_id,
                    artifact_stem=artifact_stem,
                ),
            )
        )
    return tuple(targets)


def _validate_bundle_output_artifact_paths(
    targets: Sequence[_BenchmarkHandoffBundleTarget],
    *,
    input_jsonl: str | Path,
    manifest_json: str | Path | None,
    shard_uri: str,
    disk_root: str | Path | None,
    uc_volume_root: str | Path | None,
) -> None:
    named_paths: list[tuple[str, Path]] = [
        ("input_jsonl", _artifact_local_path(input_jsonl)),
        (
            "shard_uri",
            _shard_artifact_local_path(
                shard_uri,
                disk_root=disk_root,
                uc_volume_root=uc_volume_root,
            ),
        ),
    ]
    if manifest_json is not None:
        named_paths.append(("manifest_json", _artifact_local_path(manifest_json)))
    for target in targets:
        key = f"{target.example.dataset}/{target.example.example_id}"
        named_paths.extend(
            (
                (f"handoff_json for {key}", _artifact_local_path(target.handoff_json)),
                (f"payload_uri for {key}", _artifact_local_path(target.payload_uri)),
            )
        )
    duplicates = _duplicate_artifact_paths(named_paths)
    if duplicates:
        raise ValueError("Benchmark handoff bundle artifact paths collide: " + "; ".join(duplicates))


def _duplicate_artifact_paths(named_paths: Iterable[tuple[str, Path]]) -> tuple[str, ...]:
    seen: dict[Path, str] = {}
    duplicates: list[str] = []
    for label, path in named_paths:
        previous = seen.get(path)
        if previous is None:
            seen[path] = label
            continue
        duplicates.append(f"{label} collides with {previous} at {path}")
    return tuple(duplicates)


def _artifact_local_path(uri: str | Path, *, root: str | Path | None = None) -> Path:
    return local_path(str(uri), root=root).expanduser().resolve(strict=False)


def _shard_artifact_local_path(
    shard_uri: str,
    *,
    disk_root: str | Path | None,
    uc_volume_root: str | Path | None,
) -> Path:
    root: str | Path | None
    if _is_relative_artifact_uri(shard_uri):
        root = uc_volume_root if uc_volume_root is not None else disk_root
    elif shard_uri.startswith("disk:"):
        root = disk_root
    elif shard_uri.startswith("uc-volume:") or shard_uri == "/Volumes" or shard_uri.startswith("/Volumes/"):
        root = uc_volume_root
    else:
        root = None
    return _artifact_local_path(shard_uri, root=root)


def _is_relative_artifact_uri(uri: str) -> bool:
    return ":" not in uri and not Path(uri).is_absolute()


def _adapter_spec(backend: ServingBackend) -> Any:
    if backend == ServingBackend.VLLM:
        return vllm_adapter_spec()
    if backend == ServingBackend.SGLANG:
        return sglang_adapter_spec()
    raise ValueError(f"Unsupported backend {backend!r}")


def _backend_from_value(value: ServingBackend | str) -> ServingBackend:
    try:
        return value if isinstance(value, ServingBackend) else ServingBackend(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"backend must be one of {[backend.value for backend in ServingBackend]}") from exc


def _bundle_output_base(output_dir: str | Path) -> Path:
    return local_path(str(output_dir)).expanduser().resolve(strict=False)


def _default_bundle_shard_uri(output_base: Path, *, shard_uri: str | Path | None) -> str:
    if shard_uri is not None:
        return _required_string(str(shard_uri), field_name="shard_uri")
    return str(output_base / _DEFAULT_BUNDLE_SHARD_FILENAME)


def _bundle_handoff_json(
    output_base: Path,
    *,
    handoff_json_template: str | None,
    dataset: str,
    example_id: str,
    artifact_stem: str,
) -> str:
    template = handoff_json_template or str(output_base / "{dataset}" / "{artifact_stem}.handoff.json")
    return _format_benchmark_bundle_template(
        template,
        dataset=dataset,
        example_id=example_id,
        artifact_stem=artifact_stem,
        field_name="handoff_json_template",
    )


def _bundle_payload_uri(
    output_base: Path,
    *,
    payload_uri_template: str | None,
    dataset: str,
    example_id: str,
    artifact_stem: str,
) -> str:
    template = payload_uri_template or str(output_base / "{dataset}" / "{artifact_stem}.kv")
    payload_uri = _format_benchmark_bundle_template(
        template,
        dataset=dataset,
        example_id=example_id,
        artifact_stem=artifact_stem,
        field_name="payload_uri_template",
    )
    _validate_local_payload_uri(payload_uri)
    return payload_uri


def _split_factory_path(factory_path: str) -> tuple[str, str]:
    if ":" in factory_path:
        module_name, attribute_name = factory_path.split(":", maxsplit=1)
    else:
        module_name, _, attribute_name = factory_path.rpartition(".")
    if not module_name or not attribute_name:
        raise ValueError("generator_factory must use 'module:callable' or 'module.callable' syntax")
    return module_name, attribute_name


def _validate_generator(generator: object) -> None:
    if not hasattr(generator, "generate") or not callable(getattr(generator, "generate")):
        raise TypeError("generator must implement KVChunkGenerator.generate")


def _validate_unique_benchmark_examples(examples: Sequence[Any]) -> None:
    seen: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    for example in examples:
        key = (example.dataset, example.example_id)
        if key in seen:
            duplicates.append(key)
            continue
        seen.add(key)
    if duplicates:
        raise ValueError("Duplicate benchmark input rows for " + _format_keys(duplicates))


def _optional_string(value: str | None, *, default: str, field_name: str) -> str:
    if value is None:
        return default
    return _required_string(value, field_name=field_name)


def _string_tuple(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    normalized = tuple(values)
    for index, value in enumerate(normalized):
        _required_string(value, field_name=f"{field_name}[{index}]")
    return normalized


def _required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _runtime_payload_uri(value: object, *, field_name: str) -> str:
    uri = _required_string(value, field_name=field_name)
    try:
        _validate_local_payload_uri(uri)
    except ValueError as exc:
        raise ValueError(f"{field_name}: {exc}") from exc
    return uri


def _validate_handoff_record_payload_uri(record: Mapping[str, Any], *, field_name: str) -> None:
    payload_source = record.get("payload_source")
    if not isinstance(payload_source, Mapping):
        raise ValueError("handoff_record.payload_source must be an object")
    _runtime_payload_uri(payload_source.get("uri"), field_name=field_name)


def _json_object(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        normalized[key] = _json_value(item, field_name=f"{field_name}.{key}")
    return normalized


def _json_value(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be JSON-compatible")
        return value
    if isinstance(value, Mapping):
        return _json_object(value, field_name=field_name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return [_json_value(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{field_name} must be JSON-compatible")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
