"""Attach Cachet handoff metadata to prepared V1 benchmark JSONL rows."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_kv_cache.benchmark_runner import _validate_benchmark_jsonl_record
from document_kv_cache.benchmarks import (
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    validate_v1_dataset,
)
from document_kv_cache.dataset_prep import write_v1_jsonl
from document_kv_cache.engine_adapters import validate_engine_adapter_request_record
from document_kv_cache.engine_probe import _validate_local_payload_uri
from document_kv_cache.storage import local_path


BENCHMARK_HANDOFF_MANIFEST_RECORD_TYPE = "document_kv.benchmark_handoffs.v1"
BENCHMARK_HANDOFF_MANIFEST_SCHEMA_VERSION = 1
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
    "benchmark_handoff_manifest_from_record",
    "benchmark_handoff_manifest_to_record",
    "enrich_benchmark_jsonl_with_handoffs",
    "enrich_benchmark_records_with_handoffs",
    "read_benchmark_handoff_manifest_json",
    "write_benchmark_handoff_manifest_json",
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
