"""Content-addressed inputs for the published main latency protocol.

This module intentionally loads a tokenizer, never a causal model.  It selects a
fixed number of canonical examples per V1 dataset, losslessly tiles each source
document context into the protocol's document count, and adds deterministic
irrelevant padding.  Prepared Vanilla segments are accepted only when their
independently tokenized IDs compose to the exact leading cache-prefix IDs of the
logical prompt.
"""

from __future__ import annotations

import argparse
import heapq
import importlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

from document_kv_cache._benchmark_datasets import _example_from_record
from document_kv_cache.benchmarks import (
    DEFAULT_SYSTEM_PROMPT_POSITION,
    DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    SUPPORTED_V1_DATASETS,
    BenchmarkExample,
    benchmark_cache_prefix_segments,
    build_prompt_parts,
    resolve_system_prompt_position,
)
from document_kv_cache.storage import local_path


MAIN_LATENCY_INPUT_RECORD_TYPE = "cachet.main_latency_inputs"
MAIN_LATENCY_INPUT_SCHEMA_VERSION = 3
MAIN_LATENCY_TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"
MAIN_LATENCY_TOKENIZER_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MAIN_LATENCY_ADD_SPECIAL_TOKENS = False
MAIN_LATENCY_EXAMPLES_PER_DATASET = 32
MAIN_LATENCY_TARGET_SEGMENT_COUNTS: Mapping[int, int] = {
    8192: 4,
    16384: 8,
    32768: 16,
}
MAIN_LATENCY_PROVENANCE_FILENAME = "main-latency-inputs.provenance.json"

_CANONICAL_SOURCE_FIELDS = frozenset(
    {
        "dataset",
        "documents",
        "example_id",
        "expected_answer",
        "metadata",
        "query",
        "references",
    }
)
_TRANSFER_FIELDS = frozenset({"arm_kv_transfer_params", "kv_transfer_params"})
_PADDING_UNITS = (" padding", " x", "x", "0", ".")
_SOURCE_CHUNK_ID = "source-context"
_PADDING_CHUNK_ID = "length-padding"
_TRANSFORMATION_ID = "cachet.main_latency.lossless_context_tiling.v1"
_SELECTION_DOMAIN = "cachet.main_latency.content_hash_selection.v1"
_PROVENANCE_FIELDS = frozenset(
    {
        "bundle_sha256",
        "closed_record_sha256",
        "outputs",
        "outputs_sha256",
        "protocol",
        "record_type",
        "schema_version",
        "sources",
        "sources_sha256",
    }
)
_SOURCE_PROVENANCE_FIELDS = frozenset(
    {
        "byte_count",
        "dataset",
        "jsonl_sha256",
        "record_count",
        "selected_records",
        "selected_records_sha256",
    }
)
_SOURCE_SELECTION_PROVENANCE_FIELDS = frozenset(
    {
        "example_identity_sha256",
        "expected_answer_sha256",
        "query_sha256",
        "references_sha256",
        "selected_record_sha256",
        "selection_priority_sha256",
        "selection_record_index",
        "source_document_context_sha256",
    }
)
_OUTPUT_PROVENANCE_FIELDS = frozenset(
    {
        "byte_count",
        "dataset",
        "input_tokens_target",
        "jsonl_sha256",
        "record_count",
        "records",
        "records_sha256",
        "relative_path",
        "segment_count",
    }
)
_PREPARED_RECORD_PROVENANCE_FIELDS = frozenset(
    {
        "byte_count",
        "cache_prefix_sha256",
        "cache_prefix_token_count",
        "cache_prefix_token_ids_sha256",
        "cache_suffix_sha256",
        "dataset",
        "example_identity_sha256",
        "expected_answer_sha256",
        "input_tokens_target",
        "jsonl_row_sha256",
        "logical_prompt_sha256",
        "logical_prompt_token_ids_sha256",
        "prepared_document_context_sha256",
        "prepared_record_sha256",
        "query_sha256",
        "record_index",
        "references_sha256",
        "segment_count",
        "segments",
        "segments_sha256",
        "source_document_context_sha256",
        "source_record_sha256",
    }
)


class MainLatencyTokenizer(Protocol):
    """Tokenizer surface used by preparation and verification."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class MainLatencyInputFile:
    """One prepared dataset/context JSONL artifact."""

    dataset: str
    input_tokens_target: int
    segment_count: int
    jsonl_path: Path
    jsonl_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedMainLatencyInputs:
    """Paths and content identities for one complete prepared suite."""

    output_dir: Path
    provenance_json_path: Path
    provenance_sha256: str
    bundle_sha256: str
    files: tuple[MainLatencyInputFile, ...]


@dataclass(frozen=True, slots=True)
class _SourceRow:
    record: Mapping[str, Any]
    example: BenchmarkExample
    record_sha256: str
    source_record_index: int


@dataclass(frozen=True, slots=True)
class _SourceFile:
    dataset: str
    raw_sha256: str
    byte_count: int
    rows: tuple[_SourceRow, ...]


@dataclass(frozen=True, slots=True)
class _SegmentPadding:
    unit: str | None
    repetitions: int


@dataclass(frozen=True, slots=True)
class _PreparedRecord:
    record: Mapping[str, Any]
    example: BenchmarkExample
    source_row: _SourceRow
    source_context: str
    source_pieces: tuple[str, ...]
    paddings: tuple[_SegmentPadding, ...]


@dataclass(frozen=True, slots=True)
class _PreparedOutput:
    dataset: str
    target: int
    segment_count: int
    relative_path: str
    output_bytes: bytes
    records: tuple[_PreparedRecord, ...]


class _CandidateCannotSatisfyProtocol(ValueError):
    """A valid source row cannot satisfy one exact protocol setting."""


def prepare_main_latency_inputs(
    source_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    tokenizer: MainLatencyTokenizer | None = None,
    examples_per_dataset: int = MAIN_LATENCY_EXAMPLES_PER_DATASET,
) -> PreparedMainLatencyInputs:
    """Prepare all 12 canonical main-latency JSONL files and closed provenance.

    ``source_paths`` must contain canonical V1 JSONL for Biography, HotpotQA,
    MusiQue, and NIAH.  The requested number of stable example identities per
    dataset is reused across all three context targets.  Existing identical
    artifacts are accepted as a deterministic recheck; conflicting bytes fail
    rather than being overwritten.
    """

    _require_default_prompt_contract()
    selected_count = _validated_examples_per_dataset(examples_per_dataset)
    sources = _load_all_sources(source_paths)
    destination = local_path(str(output_dir))
    resolved_tokenizer = tokenizer or load_main_latency_tokenizer()
    prepared_outputs: list[_PreparedOutput] = []
    selected_rows: dict[str, tuple[_SourceRow, ...]] = {}
    for dataset in SUPPORTED_V1_DATASETS:
        rows, outputs = _select_and_prepare_dataset(
            sources[dataset],
            tokenizer=resolved_tokenizer,
            examples_per_dataset=selected_count,
        )
        selected_rows[dataset] = rows
        prepared_outputs.extend(outputs)
    ordered_outputs = tuple(
        sorted(
            prepared_outputs,
            key=lambda output: (output.target, output.dataset),
        )
    )
    provenance = _build_provenance(
        sources=sources,
        selected_rows=selected_rows,
        outputs=ordered_outputs,
        tokenizer=resolved_tokenizer,
        examples_per_dataset=selected_count,
    )
    provenance_bytes = _canonical_json_bytes(provenance, pretty=True)

    provenance_path = destination / MAIN_LATENCY_PROVENANCE_FILENAME
    artifacts = tuple(
        (
            destination / PurePosixPath(output.relative_path),
            output.output_bytes,
        )
        for output in ordered_outputs
    ) + ((provenance_path, provenance_bytes),)
    source_resolutions = {
        local_path(str(path)).resolve() for path in source_paths.values()
    }
    overlap = [
        path for path, _content in artifacts if path.resolve() in source_resolutions
    ]
    if overlap:
        raise ValueError("source JSONL paths must be separate from prepared artifacts")
    for path, content in artifacts:
        _require_absent_or_identical(path, content)
    for path, content in artifacts:
        _write_if_absent_or_identical(path, content)

    verified = verify_main_latency_inputs(
        destination,
        source_paths=source_paths,
        tokenizer=resolved_tokenizer,
        examples_per_dataset=selected_count,
    )
    expected_bundle = cast(str, provenance["bundle_sha256"])
    if verified.bundle_sha256 != expected_bundle:
        raise ValueError("prepared suite changed during deterministic verification")
    return verified


def verify_main_latency_inputs(
    output_dir: str | Path,
    *,
    source_paths: Mapping[str, str | Path] | None = None,
    tokenizer: MainLatencyTokenizer | None = None,
    examples_per_dataset: int = MAIN_LATENCY_EXAMPLES_PER_DATASET,
) -> PreparedMainLatencyInputs:
    """Verify canonical bytes, closed provenance, and every token invariant.

    Source JSONL paths are optional.  When supplied, their bytes and the selected
    canonical records are checked as well; without them, the closed provenance still
    verifies every emitted artifact and binds it to the recorded source digests.
    """

    _require_default_prompt_contract()
    selected_count = _validated_examples_per_dataset(examples_per_dataset)
    destination = local_path(str(output_dir))
    provenance_path = destination / MAIN_LATENCY_PROVENANCE_FILENAME
    provenance_bytes = provenance_path.read_bytes()
    provenance = _json_object_from_bytes(provenance_bytes, label="provenance JSON")
    if provenance_bytes != _canonical_json_bytes(provenance, pretty=True):
        raise ValueError("provenance JSON is not in canonical encoding")
    _verify_closed_provenance(provenance)
    resolved_tokenizer = tokenizer or load_main_latency_tokenizer()
    _verify_protocol_record(provenance, examples_per_dataset=selected_count)
    source_records = _source_records_from_provenance(
        provenance,
        examples_per_dataset=selected_count,
    )
    output_records = _output_records_from_provenance(
        provenance,
        examples_per_dataset=selected_count,
    )
    _verify_identity_reuse_across_targets(source_records, output_records)

    verified_files: list[MainLatencyInputFile] = []
    rebuilt_output_records: list[Mapping[str, Any]] = []
    for expected in output_records:
        dataset = _required_str(expected, "dataset")
        target = _required_int(expected, "input_tokens_target")
        segment_count = _required_int(expected, "segment_count")
        expected_path = _relative_output_path(dataset, target)
        if _required_str(expected, "relative_path") != expected_path:
            raise ValueError("provenance output path does not match the protocol")
        jsonl_path = destination / PurePosixPath(expected_path)
        output_bytes = jsonl_path.read_bytes()
        output_digest = sha256(output_bytes).hexdigest()
        if output_digest != _required_sha256(expected, "jsonl_sha256"):
            raise ValueError(f"prepared output digest mismatch for {dataset}/{target}")
        if len(output_bytes) != _required_int(expected, "byte_count"):
            raise ValueError(
                f"prepared output byte count mismatch for {dataset}/{target}"
            )
        records = _canonical_jsonl_records(output_bytes, label=expected_path)
        if len(records) != selected_count:
            raise ValueError(
                f"{expected_path} must contain exactly {selected_count} records"
            )
        described_records: list[Mapping[str, Any]] = []
        for record_index, record in enumerate(records, start=1):
            example = _canonical_example(
                record,
                dataset=dataset,
                record_index=record_index,
            )
            expected_records = _mapping_sequence(
                expected.get("records"),
                label=f"outputs[{dataset}/{target}].records",
            )
            source_record_sha256 = _required_sha256(
                expected_records[record_index - 1],
                "source_record_sha256",
            )
            row_bytes = _canonical_json_bytes(record, pretty=False) + b"\n"
            described_records.append(
                _describe_prepared_record(
                    record,
                    example=example,
                    dataset=dataset,
                    target=target,
                    segment_count=segment_count,
                    output_bytes=row_bytes,
                    source_record_sha256=source_record_sha256,
                    record_index=record_index,
                    tokenizer=resolved_tokenizer,
                )
            )
        rebuilt = _describe_prepared_output(
            dataset=dataset,
            target=target,
            segment_count=segment_count,
            relative_path=expected_path,
            output_bytes=output_bytes,
            records=described_records,
        )
        if rebuilt != expected:
            raise ValueError(
                f"prepared output provenance mismatch for {dataset}/{target}"
            )
        rebuilt_output_records.append(rebuilt)
        verified_files.append(
            MainLatencyInputFile(
                dataset=dataset,
                input_tokens_target=target,
                segment_count=segment_count,
                jsonl_path=jsonl_path,
                jsonl_sha256=output_digest,
            )
        )

    expected_outputs = _expected_output_keys()
    actual_outputs = {
        (
            _required_str(record, "dataset"),
            _required_int(record, "input_tokens_target"),
        )
        for record in output_records
    }
    if actual_outputs != expected_outputs or len(output_records) != len(
        expected_outputs
    ):
        raise ValueError("provenance does not contain the complete main-latency suite")
    if _canonical_sha256(rebuilt_output_records) != _required_sha256(
        provenance,
        "outputs_sha256",
    ):
        raise ValueError("outputs_sha256 does not match the closed output records")
    bundle_sha256 = _bundle_sha256(rebuilt_output_records)
    if bundle_sha256 != _required_sha256(provenance, "bundle_sha256"):
        raise ValueError("bundle_sha256 does not match the prepared artifacts")

    if source_paths is not None:
        sources = _load_all_sources(source_paths)
        _verify_sources_against_provenance(
            sources,
            source_records=source_records,
            output_records=output_records,
            examples_per_dataset=selected_count,
            tokenizer=resolved_tokenizer,
        )

    return PreparedMainLatencyInputs(
        output_dir=destination,
        provenance_json_path=provenance_path,
        provenance_sha256=sha256(provenance_bytes).hexdigest(),
        bundle_sha256=bundle_sha256,
        files=tuple(verified_files),
    )


def load_main_latency_tokenizer() -> MainLatencyTokenizer:
    """Load only the exact pinned tokenizer used by the main protocol."""

    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - caller environment dependent.
        raise RuntimeError(
            "main-latency input preparation requires Transformers for AutoTokenizer"
        ) from exc
    auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
    if auto_tokenizer is None:
        raise RuntimeError("Transformers does not expose AutoTokenizer")
    loaded = auto_tokenizer.from_pretrained(
        MAIN_LATENCY_TOKENIZER_ID,
        revision=MAIN_LATENCY_TOKENIZER_REVISION,
        trust_remote_code=False,
        use_fast=True,
    )
    if not callable(getattr(loaded, "encode", None)):
        raise TypeError("AutoTokenizer must return an object with encode()")
    return cast(MainLatencyTokenizer, loaded)


def main_latency_inputs_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for preparing or independently verifying the suite."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare or verify the content-addressed main latency inputs with the "
            "pinned Qwen tokenizer; no causal model is loaded."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="Repeat for biography, hotpotqa, musique, and niah canonical JSONL.",
    )
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument(
        "--examples-per-dataset",
        type=int,
        default=MAIN_LATENCY_EXAMPLES_PER_DATASET,
        help=(
            "Number of deterministic examples in each dataset JSONL "
            f"(default: {MAIN_LATENCY_EXAMPLES_PER_DATASET})."
        ),
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-dir", required=True)
    verify_parser.add_argument(
        "--source",
        action="append",
        metavar="DATASET=PATH",
        help="Optional; when used, repeat for all four canonical source JSONLs.",
    )
    verify_parser.add_argument(
        "--examples-per-dataset",
        type=int,
        default=MAIN_LATENCY_EXAMPLES_PER_DATASET,
        help=(
            "Expected number of examples in each dataset JSONL "
            f"(default: {MAIN_LATENCY_EXAMPLES_PER_DATASET})."
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_main_latency_inputs(
                _parse_source_specs(args.source),
                args.output_dir,
                examples_per_dataset=args.examples_per_dataset,
            )
        else:
            sources = None if not args.source else _parse_source_specs(args.source)
            result = verify_main_latency_inputs(
                args.output_dir,
                source_paths=sources,
                examples_per_dataset=args.examples_per_dataset,
            )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "ok": False,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "bundle_sha256": result.bundle_sha256,
                "examples_per_dataset": args.examples_per_dataset,
                "files": [
                    {
                        "dataset": item.dataset,
                        "input_tokens_target": item.input_tokens_target,
                        "jsonl_path": str(item.jsonl_path),
                        "jsonl_sha256": item.jsonl_sha256,
                        "segment_count": item.segment_count,
                    }
                    for item in result.files
                ],
                "ok": True,
                "provenance_json_path": str(result.provenance_json_path),
                "provenance_sha256": result.provenance_sha256,
                "tokenizer_id": MAIN_LATENCY_TOKENIZER_ID,
                "tokenizer_revision": MAIN_LATENCY_TOKENIZER_REVISION,
            },
            sort_keys=True,
        )
    )
    return 0


def _require_default_prompt_contract() -> None:
    position = resolve_system_prompt_position()
    if position != DEFAULT_SYSTEM_PROMPT_POSITION:
        raise ValueError(
            "main-latency input preparation requires the default start "
            "system-prompt position"
        )


def _validated_examples_per_dataset(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("examples_per_dataset must be a positive integer")
    return value


def _load_all_sources(
    source_paths: Mapping[str, str | Path],
) -> Mapping[str, _SourceFile]:
    if not isinstance(source_paths, Mapping):
        raise TypeError("source_paths must be a mapping")
    actual = set(source_paths)
    expected = set(SUPPORTED_V1_DATASETS)
    if actual != expected:
        raise ValueError(
            "source_paths must contain exactly "
            f"{sorted(expected)}; got {sorted(actual)}"
        )
    return {
        dataset: _load_source_file(dataset, source_paths[dataset])
        for dataset in SUPPORTED_V1_DATASETS
    }


def _load_source_file(dataset: str, path: str | Path) -> _SourceFile:
    source_path = local_path(str(path))
    raw = source_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{dataset} source JSONL must be UTF-8") from exc
    rows: list[_SourceRow] = []
    seen_identities: set[tuple[str, str]] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{dataset} source JSONL line {line_number} is invalid: {exc.msg}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ValueError(
                f"{dataset} source JSONL line {line_number} must be an object"
            )
        record = cast(
            Mapping[str, Any],
            json.loads(json.dumps(dict(value), ensure_ascii=False)),
        )
        transfer_fields = set(record).intersection(_TRANSFER_FIELDS)
        if transfer_fields:
            raise ValueError(
                f"{dataset} source JSONL line {line_number} contains KV transfer "
                f"metadata: {sorted(transfer_fields)}"
            )
        unknown = set(record).difference(_CANONICAL_SOURCE_FIELDS)
        if unknown:
            raise ValueError(
                f"{dataset} source JSONL line {line_number} has noncanonical "
                f"fields: {sorted(unknown)}"
            )
        for field_name in ("dataset", "documents", "example_id", "query"):
            if field_name not in record:
                raise ValueError(
                    f"{dataset} source JSONL line {line_number} requires {field_name}"
                )
        example = _canonical_example(
            record,
            dataset=dataset,
            record_index=len(rows) + 1,
        )
        if not example.references:
            raise ValueError(
                f"{dataset} source JSONL line {line_number} requires an answer "
                "or references"
            )
        identity = (example.dataset, example.example_id)
        if identity in seen_identities:
            raise ValueError(f"{dataset} source contains duplicate example identity")
        seen_identities.add(identity)
        rows.append(
            _SourceRow(
                record=record,
                example=example,
                record_sha256=_canonical_sha256(record),
                source_record_index=len(rows) + 1,
            )
        )
    if not rows:
        raise ValueError(f"{dataset} source JSONL must contain at least one record")
    return _SourceFile(
        dataset=dataset,
        raw_sha256=sha256(raw).hexdigest(),
        byte_count=len(raw),
        rows=tuple(rows),
    )


def _canonical_example(
    record: Mapping[str, Any],
    *,
    dataset: str,
    record_index: int,
) -> BenchmarkExample:
    try:
        return _example_from_record(
            record,
            default_dataset=dataset,
            record_index=record_index,
            require_dataset=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid canonical {dataset} record: {exc}") from exc


def _select_and_prepare_dataset(
    source: _SourceFile,
    *,
    tokenizer: MainLatencyTokenizer,
    examples_per_dataset: int,
) -> tuple[tuple[_SourceRow, ...], tuple[_PreparedOutput, ...]]:
    ordered = sorted(
        source.rows,
        key=lambda row: (_selection_priority_sha256(row), row.record_sha256),
    )
    failures: list[str] = []
    selected_rows: list[_SourceRow] = []
    prepared_by_target: dict[int, list[_PreparedRecord]] = {
        target: [] for target in MAIN_LATENCY_TARGET_SEGMENT_COUNTS
    }
    for row in ordered:
        prepared_records: dict[int, _PreparedRecord] = {}
        try:
            for target, segment_count in MAIN_LATENCY_TARGET_SEGMENT_COUNTS.items():
                prepared_records[target] = _prepare_record(
                    row,
                    target=target,
                    segment_count=segment_count,
                    tokenizer=tokenizer,
                )
        except _CandidateCannotSatisfyProtocol as exc:
            failures.append(str(exc))
            continue
        selected_rows.append(row)
        for target, prepared in prepared_records.items():
            prepared_by_target[target].append(prepared)
        if len(selected_rows) == examples_per_dataset:
            break
    if len(selected_rows) == examples_per_dataset:
        outputs: list[_PreparedOutput] = []
        for target, segment_count in MAIN_LATENCY_TARGET_SEGMENT_COUNTS.items():
            records = tuple(prepared_by_target[target])
            relative_path = _relative_output_path(source.dataset, target)
            output_bytes = b"".join(
                _canonical_json_bytes(prepared.record, pretty=False) + b"\n"
                for prepared in records
            )
            outputs.append(
                _PreparedOutput(
                    dataset=source.dataset,
                    target=target,
                    segment_count=segment_count,
                    relative_path=relative_path,
                    output_bytes=output_bytes,
                    records=records,
                )
            )
        return tuple(selected_rows), tuple(outputs)
    detail = failures[0] if failures else "no candidate records"
    raise ValueError(
        f"{source.dataset} source can produce only {len(selected_rows)} of "
        f"{examples_per_dataset} required examples for every main latency target: "
        f"{detail}"
    )


def _prepare_record(
    row: _SourceRow,
    *,
    target: int,
    segment_count: int,
    tokenizer: MainLatencyTokenizer,
) -> _PreparedRecord:
    source_context = build_prompt_parts(row.example).document_context
    source_pieces = _split_text_exactly(source_context, segment_count)
    record = _derived_record(row, source_pieces=source_pieces)
    example = _canonical_example(record, dataset=row.example.dataset, record_index=1)
    base_segments, base_segment_ids, base_prefix_ids, base_full_ids = (
        _validated_token_composition(example, tokenizer=tokenizer)
    )
    if len(base_segments) != segment_count:
        raise _CandidateCannotSatisfyProtocol(
            f"expected {segment_count} cache segments; got {len(base_segments)}"
        )
    online_tail_count = len(base_full_ids) - len(base_prefix_ids)
    desired_prefix_count = target - online_tail_count
    base_counts = tuple(len(token_ids) for token_ids in base_segment_ids)
    if desired_prefix_count < sum(base_counts):
        raise _CandidateCannotSatisfyProtocol(
            f"unpadded prompt exceeds the {target}-token target"
        )
    goals = _balanced_segment_goals(base_counts, desired_prefix_count)
    mutable_record = record
    paddings: list[_SegmentPadding] = []
    for index, goal in enumerate(goals):
        paddings.append(
            _pad_segment_to_exact_count(
                mutable_record,
                dataset=row.example.dataset,
                index=index,
                goal=goal,
                tokenizer=tokenizer,
            )
        )
    example = _canonical_example(
        mutable_record,
        dataset=row.example.dataset,
        record_index=1,
    )
    segments, segment_ids, prefix_ids, full_ids = _validated_token_composition(
        example,
        tokenizer=tokenizer,
    )
    if len(full_ids) != target:
        raise _CandidateCannotSatisfyProtocol(
            f"exact padding produced {len(full_ids)} tokens instead of {target}"
        )
    if tuple(len(ids) for ids in segment_ids) != goals:
        raise _CandidateCannotSatisfyProtocol(
            "segment token counts changed after exact padding"
        )
    if len(prefix_ids) != desired_prefix_count or len(segments) != segment_count:
        raise _CandidateCannotSatisfyProtocol(
            "prepared cache prefix changed after exact padding"
        )
    frozen_record = cast(
        Mapping[str, Any],
        json.loads(json.dumps(mutable_record, ensure_ascii=False)),
    )
    return _PreparedRecord(
        record=frozen_record,
        example=example,
        source_row=row,
        source_context=source_context,
        source_pieces=source_pieces,
        paddings=tuple(paddings),
    )


def _derived_record(
    row: _SourceRow,
    *,
    source_pieces: Sequence[str],
) -> dict[str, Any]:
    record = cast(
        dict[str, Any],
        json.loads(json.dumps(dict(row.record), ensure_ascii=False)),
    )
    identity_digest = _example_identity_sha256(row.example)
    count = len(source_pieces)
    width = max(2, len(str(count)))
    documents: list[dict[str, Any]] = []
    for index, source_piece in enumerate(source_pieces):
        ordinal = f"{index + 1:0{width}d}"
        documents.append(
            {
                "chunks": [
                    {"chunk_id": _SOURCE_CHUNK_ID, "text": source_piece},
                    {"chunk_id": _PADDING_CHUNK_ID, "text": ""},
                ],
                "document_id": (
                    f"cachet-main-{identity_digest[:16]}-segment-{ordinal}"
                ),
                "metadata": {
                    "cachet.main_latency.segment_count": str(count),
                    "cachet.main_latency.segment_index": str(index),
                    "cachet.main_latency.transformation": _TRANSFORMATION_ID,
                },
                "title": f"Source context segment {ordinal} of {count}",
            }
        )
    record["documents"] = documents
    return record


def _split_text_exactly(text: str, count: int) -> tuple[str, ...]:
    if not isinstance(text, str) or not text:
        raise ValueError("source document context must be non-empty")
    if type(count) is not int or count <= 0:
        raise ValueError("segment count must be a positive integer")
    bounds = tuple((len(text) * index) // count for index in range(count + 1))
    pieces = tuple(text[bounds[index] : bounds[index + 1]] for index in range(count))
    if "".join(pieces) != text:
        raise AssertionError("source context tiling must be lossless")
    return pieces


def _validated_token_composition(
    example: BenchmarkExample,
    *,
    tokenizer: MainLatencyTokenizer,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    parts = build_prompt_parts(example)
    segments = benchmark_cache_prefix_segments(example)
    segment_ids = tuple(_token_ids(tokenizer, text) for _chunk_id, text in segments)
    composed = tuple(token_id for ids in segment_ids for token_id in ids)
    prefix_ids = _token_ids(tokenizer, parts.cache_prefix_text)
    full_ids = _token_ids(tokenizer, parts.prefill_prompt)
    if composed != prefix_ids:
        raise _CandidateCannotSatisfyProtocol(
            "independently tokenized Vanilla segments do not compose to the "
            "logical cache prefix"
        )
    if len(prefix_ids) >= len(full_ids) or full_ids[: len(prefix_ids)] != prefix_ids:
        raise _CandidateCannotSatisfyProtocol(
            "logical cache prefix is not a strict leading token prefix of the prompt"
        )
    if len({text for _chunk_id, text in segments}) != len(segments):
        raise _CandidateCannotSatisfyProtocol(
            "prepared document segment texts are not distinct"
        )
    return segments, segment_ids, prefix_ids, full_ids


def _balanced_segment_goals(
    base_counts: Sequence[int],
    desired_total: int,
) -> tuple[int, ...]:
    goals = list(base_counts)
    remaining = desired_total - sum(goals)
    if remaining < 0:
        raise ValueError("desired segment total cannot be below the base total")
    heap = [(count, index) for index, count in enumerate(goals)]
    heapq.heapify(heap)
    for _ in range(remaining):
        count, index = heapq.heappop(heap)
        goals[index] = count + 1
        heapq.heappush(heap, (count + 1, index))
    return tuple(goals)


def _pad_segment_to_exact_count(
    record: dict[str, Any],
    *,
    dataset: str,
    index: int,
    goal: int,
    tokenizer: MainLatencyTokenizer,
) -> _SegmentPadding:
    counts: dict[tuple[str, int], int] = {}
    sentinel = _unused_padding_sentinel(record)
    _set_padding_text(record, index=index, text=sentinel)
    example = _canonical_example(record, dataset=dataset, record_index=1)
    segments = benchmark_cache_prefix_segments(example)
    if len(segments) <= index:
        raise _CandidateCannotSatisfyProtocol(
            "document segmentation collapsed during exact padding"
        )
    segment_template = segments[index][1]
    if segment_template.count(sentinel) != 1:
        raise _CandidateCannotSatisfyProtocol(
            "padding sentinel was not isolated in its document segment"
        )

    def count(unit: str, repetitions: int) -> int:
        key = (unit, repetitions)
        if key not in counts:
            candidate = segment_template.replace(
                sentinel,
                unit * repetitions,
                1,
            )
            counts[key] = len(_token_ids(tokenizer, candidate))
        return counts[key]

    for unit in _PADDING_UNITS:
        _set_padding_text(record, index=index, text="")
        zero_count = count(unit, 0)
        if zero_count == goal:
            return _SegmentPadding(unit=None, repetitions=0)
        if zero_count > goal:
            continue
        maximum = goal * 64
        low = 0
        high = 1
        while high <= maximum and count(unit, high) < goal:
            low = high
            high *= 2
        high = min(high, maximum)
        if count(unit, high) < goal:
            continue
        while low + 1 < high:
            middle = (low + high) // 2
            if count(unit, middle) < goal:
                low = middle
            else:
                high = middle
        start = max(0, low - 64)
        stop = min(maximum, high + 64)
        for repetitions in range(start, stop + 1):
            if count(unit, repetitions) == goal:
                _set_padding_text(
                    record,
                    index=index,
                    text=unit * repetitions,
                )
                return _SegmentPadding(unit=unit, repetitions=repetitions)
    _set_padding_text(record, index=index, text="")
    raise _CandidateCannotSatisfyProtocol(
        f"cannot pad document segment {index} to exactly {goal} tokens"
    )


def _unused_padding_sentinel(record: Mapping[str, Any]) -> str:
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
    sentinel = "\u241fCACHET_MAIN_LATENCY_PADDING_SENTINEL\u241f"
    while sentinel in serialized:
        sentinel += "_"
    return sentinel


def _set_padding_text(record: dict[str, Any], *, index: int, text: str) -> None:
    raw_documents = record.get("documents")
    if not isinstance(raw_documents, list) or index >= len(raw_documents):
        raise TypeError("prepared documents must be an array")
    raw_document = raw_documents[index]
    if not isinstance(raw_document, dict):
        raise TypeError("prepared document must be an object")
    raw_chunks = raw_document.get("chunks")
    if not isinstance(raw_chunks, list) or len(raw_chunks) != 2:
        raise TypeError("prepared document must contain source and padding chunks")
    raw_padding = raw_chunks[1]
    if not isinstance(raw_padding, dict):
        raise TypeError("prepared padding chunk must be an object")
    raw_padding["text"] = text


def _build_provenance(
    *,
    sources: Mapping[str, _SourceFile],
    selected_rows: Mapping[str, Sequence[_SourceRow]],
    outputs: Sequence[_PreparedOutput],
    tokenizer: MainLatencyTokenizer,
    examples_per_dataset: int,
) -> Mapping[str, Any]:
    source_records = [
        _describe_source_file(sources[dataset], selected_rows[dataset])
        for dataset in SUPPORTED_V1_DATASETS
    ]
    output_records = [
        _describe_prepared_output(
            dataset=output.dataset,
            target=output.target,
            segment_count=output.segment_count,
            relative_path=output.relative_path,
            output_bytes=output.output_bytes,
            records=[
                _describe_prepared_record(
                    prepared.record,
                    example=prepared.example,
                    dataset=output.dataset,
                    target=output.target,
                    segment_count=output.segment_count,
                    output_bytes=(
                        _canonical_json_bytes(prepared.record, pretty=False) + b"\n"
                    ),
                    source_record_sha256=prepared.source_row.record_sha256,
                    record_index=record_index,
                    tokenizer=tokenizer,
                )
                for record_index, prepared in enumerate(output.records, start=1)
            ],
        )
        for output in outputs
    ]
    record: dict[str, Any] = {
        "bundle_sha256": _bundle_sha256(output_records),
        "outputs": output_records,
        "outputs_sha256": _canonical_sha256(output_records),
        "protocol": _protocol_record(examples_per_dataset=examples_per_dataset),
        "record_type": MAIN_LATENCY_INPUT_RECORD_TYPE,
        "schema_version": MAIN_LATENCY_INPUT_SCHEMA_VERSION,
        "sources": source_records,
        "sources_sha256": _canonical_sha256(source_records),
    }
    record["closed_record_sha256"] = _canonical_sha256(record)
    return record


def _describe_source_file(
    source: _SourceFile,
    selected: Sequence[_SourceRow],
) -> Mapping[str, Any]:
    selected_records = [_describe_source_selection(row) for row in selected]
    return {
        "byte_count": source.byte_count,
        "dataset": source.dataset,
        "jsonl_sha256": source.raw_sha256,
        "record_count": len(source.rows),
        "selected_records": selected_records,
        "selected_records_sha256": _canonical_sha256(selected_records),
    }


def _describe_source_selection(selected: _SourceRow) -> Mapping[str, Any]:
    example = selected.example
    parts = build_prompt_parts(example)
    return {
        "example_identity_sha256": _example_identity_sha256(example),
        "expected_answer_sha256": _optional_text_sha256(example.expected_answer),
        "query_sha256": _text_sha256(example.query),
        "references_sha256": _canonical_sha256(list(example.references)),
        "selected_record_sha256": selected.record_sha256,
        "selection_priority_sha256": _selection_priority_sha256(selected),
        "selection_record_index": selected.source_record_index,
        "source_document_context_sha256": _text_sha256(parts.document_context),
    }


def _describe_prepared_output(
    *,
    dataset: str,
    target: int,
    segment_count: int,
    relative_path: str,
    output_bytes: bytes,
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    record_descriptions = list(records)
    return {
        "byte_count": len(output_bytes),
        "dataset": dataset,
        "input_tokens_target": target,
        "jsonl_sha256": sha256(output_bytes).hexdigest(),
        "record_count": len(record_descriptions),
        "records": record_descriptions,
        "records_sha256": _canonical_sha256(record_descriptions),
        "relative_path": relative_path,
        "segment_count": segment_count,
    }


def _describe_prepared_record(
    record: Mapping[str, Any],
    *,
    example: BenchmarkExample,
    dataset: str,
    target: int,
    segment_count: int,
    output_bytes: bytes,
    source_record_sha256: str,
    record_index: int,
    tokenizer: MainLatencyTokenizer,
) -> Mapping[str, Any]:
    if example.dataset != dataset:
        raise ValueError("prepared example dataset changed")
    raw_documents = _prepared_raw_documents(record, segment_count=segment_count)
    source_pieces: list[str] = []
    padding_values: list[tuple[str, _SegmentPadding]] = []
    for index, raw_document in enumerate(raw_documents):
        source_text, padding_text = _prepared_chunk_texts(raw_document, index=index)
        source_pieces.append(source_text)
        padding_values.append((padding_text, _padding_from_text(padding_text)))
    segments, segment_ids, prefix_ids, full_ids = _validated_token_composition(
        example,
        tokenizer=tokenizer,
    )
    if len(segments) != segment_count:
        raise ValueError(
            f"prepared {dataset}/{target} has {len(segments)} segments; "
            f"expected {segment_count}"
        )
    if len(full_ids) != target:
        raise ValueError(
            f"prepared {dataset}/{target} has {len(full_ids)} logical tokens"
        )
    if len({document.document_id for document in example.documents}) != segment_count:
        raise ValueError("prepared document identities are not distinct")
    parts = build_prompt_parts(example)
    segment_records: list[dict[str, Any]] = []
    for index, ((chunk_id, text), token_ids) in enumerate(
        zip(segments, segment_ids, strict=True)
    ):
        padding_text, padding = padding_values[index]
        segment_records.append(
            {
                "cache_chunk_id_sha256": _text_sha256(chunk_id),
                "document_id_sha256": _text_sha256(
                    example.documents[index].document_id
                ),
                "index": index,
                "padding": {
                    "text_sha256": _text_sha256(padding_text),
                    "unit_repetitions": padding.repetitions,
                    "unit_sha256": (
                        None if padding.unit is None else _text_sha256(padding.unit)
                    ),
                },
                "source_piece_sha256": _text_sha256(source_pieces[index]),
                "text_sha256": _text_sha256(text),
                "token_count": len(token_ids),
                "token_ids_sha256": _token_ids_sha256(token_ids),
            }
        )
    return {
        "byte_count": len(output_bytes),
        "cache_prefix_sha256": _text_sha256(parts.cache_prefix_text),
        "cache_prefix_token_count": len(prefix_ids),
        "cache_prefix_token_ids_sha256": _token_ids_sha256(prefix_ids),
        "cache_suffix_sha256": _text_sha256(parts.cache_suffix_text),
        "dataset": dataset,
        "example_identity_sha256": _example_identity_sha256(example),
        "expected_answer_sha256": _optional_text_sha256(example.expected_answer),
        "input_tokens_target": target,
        "jsonl_row_sha256": sha256(output_bytes).hexdigest(),
        "logical_prompt_sha256": _text_sha256(parts.prefill_prompt),
        "logical_prompt_token_ids_sha256": _token_ids_sha256(full_ids),
        "prepared_document_context_sha256": _text_sha256(parts.document_context),
        "prepared_record_sha256": _canonical_sha256(record),
        "query_sha256": _text_sha256(example.query),
        "record_index": record_index,
        "references_sha256": _canonical_sha256(list(example.references)),
        "segment_count": segment_count,
        "segments": segment_records,
        "segments_sha256": _canonical_sha256(segment_records),
        "source_document_context_sha256": _text_sha256("".join(source_pieces)),
        "source_record_sha256": source_record_sha256,
    }


def _prepared_raw_documents(
    record: Mapping[str, Any],
    *,
    segment_count: int,
) -> tuple[Mapping[str, Any], ...]:
    raw = record.get("documents")
    if not isinstance(raw, list) or len(raw) != segment_count:
        raise ValueError(
            f"prepared record must contain exactly {segment_count} documents"
        )
    documents: list[Mapping[str, Any]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ValueError(f"prepared document {index} must be an object")
        expected_metadata = {
            "cachet.main_latency.segment_count": str(segment_count),
            "cachet.main_latency.segment_index": str(index),
            "cachet.main_latency.transformation": _TRANSFORMATION_ID,
        }
        if value.get("metadata") != expected_metadata:
            raise ValueError(f"prepared document {index} metadata changed")
        documents.append(value)
    return tuple(documents)


def _prepared_chunk_texts(
    document: Mapping[str, Any],
    *,
    index: int,
) -> tuple[str, str]:
    raw_chunks = document.get("chunks")
    if not isinstance(raw_chunks, list) or len(raw_chunks) != 2:
        raise ValueError(f"prepared document {index} must contain exactly two chunks")
    values: list[str] = []
    for chunk_index, (raw_chunk, expected_id) in enumerate(
        zip(raw_chunks, (_SOURCE_CHUNK_ID, _PADDING_CHUNK_ID), strict=True)
    ):
        if not isinstance(raw_chunk, Mapping):
            raise ValueError(
                f"prepared document {index} chunk {chunk_index} must be an object"
            )
        if set(raw_chunk) != {"chunk_id", "text"}:
            raise ValueError(
                f"prepared document {index} chunk {chunk_index} is noncanonical"
            )
        if raw_chunk.get("chunk_id") != expected_id:
            raise ValueError(
                f"prepared document {index} chunk {chunk_index} identity changed"
            )
        text = raw_chunk.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"prepared document {index} chunk {chunk_index} text must be a string"
            )
        values.append(text)
    return values[0], values[1]


def _padding_from_text(text: str) -> _SegmentPadding:
    if not text:
        return _SegmentPadding(unit=None, repetitions=0)
    for unit in _PADDING_UNITS:
        if len(text) % len(unit) == 0:
            repetitions = len(text) // len(unit)
            if repetitions > 0 and unit * repetitions == text:
                return _SegmentPadding(unit=unit, repetitions=repetitions)
    raise ValueError("prepared padding text does not match a deterministic unit")


def _protocol_record(*, examples_per_dataset: int) -> Mapping[str, Any]:
    return {
        "datasets": list(SUPPORTED_V1_DATASETS),
        "prompt_contract": {
            "prompt_template_version": DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
            "system_prompt_position": DEFAULT_SYSTEM_PROMPT_POSITION,
        },
        "selection": {
            "domain": _SELECTION_DOMAIN,
            "identity_reused_across_targets": True,
            "ordering": "sha256_domain_dataset_identity_and_source_record",
            "selected_examples_per_dataset": examples_per_dataset,
        },
        "targets": [
            {
                "input_tokens_target": target,
                "segment_count": segment_count,
            }
            for target, segment_count in MAIN_LATENCY_TARGET_SEGMENT_COUNTS.items()
        ],
        "tokenizer": {
            "add_special_tokens": MAIN_LATENCY_ADD_SPECIAL_TOKENS,
            "tokenizer_id": MAIN_LATENCY_TOKENIZER_ID,
            "tokenizer_revision": MAIN_LATENCY_TOKENIZER_REVISION,
        },
        "transformation": {
            "document_context_tiling": "lossless_contiguous_unicode_codepoints",
            "id": _TRANSFORMATION_ID,
            "padding": "balanced_exact_token_count_irrelevant_units",
            "vanilla_composition": (
                "concatenated_independent_segment_token_ids_equal_logical_cache_prefix"
            ),
        },
    }


def _verify_protocol_record(
    provenance: Mapping[str, Any],
    *,
    examples_per_dataset: int,
) -> None:
    if provenance.get("record_type") != MAIN_LATENCY_INPUT_RECORD_TYPE:
        raise ValueError("unsupported main-latency provenance record_type")
    if provenance.get("schema_version") != MAIN_LATENCY_INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported main-latency provenance schema_version")
    if provenance.get("protocol") != _protocol_record(
        examples_per_dataset=examples_per_dataset
    ):
        raise ValueError("main-latency protocol pins do not match this implementation")


def _verify_closed_provenance(provenance: Mapping[str, Any]) -> None:
    if set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("provenance JSON does not use the closed schema")
    expected = _required_sha256(provenance, "closed_record_sha256")
    unsigned = dict(provenance)
    unsigned.pop("closed_record_sha256", None)
    if _canonical_sha256(unsigned) != expected:
        raise ValueError("closed_record_sha256 does not match provenance content")


def _source_records_from_provenance(
    provenance: Mapping[str, Any],
    *,
    examples_per_dataset: int,
) -> tuple[Mapping[str, Any], ...]:
    records = _mapping_sequence(provenance.get("sources"), label="sources")
    if [record.get("dataset") for record in records] != list(SUPPORTED_V1_DATASETS):
        raise ValueError("provenance sources are incomplete or out of canonical order")
    for index, record in enumerate(records):
        if set(record) != _SOURCE_PROVENANCE_FIELDS:
            raise ValueError(
                f"provenance sources[{index}] does not use the closed schema"
            )
        _required_int(record, "byte_count")
        _required_sha256(record, "jsonl_sha256")
        if _required_int(record, "record_count") == 0:
            raise ValueError("provenance source record_count must be positive")
        selected = _mapping_sequence(
            record.get("selected_records"),
            label=f"sources[{index}].selected_records",
        )
        if len(selected) != examples_per_dataset:
            raise ValueError(
                f"provenance sources[{index}] must select exactly "
                f"{examples_per_dataset} records"
            )
        selection_digests: set[str] = set()
        identity_digests: set[str] = set()
        for selected_index, selection in enumerate(selected):
            if set(selection) != _SOURCE_SELECTION_PROVENANCE_FIELDS:
                raise ValueError(
                    f"provenance sources[{index}].selected_records"
                    f"[{selected_index}] does not use the closed schema"
                )
            identity_digests.add(
                _required_sha256(selection, "example_identity_sha256")
            )
            expected_answer = selection.get("expected_answer_sha256")
            if expected_answer is not None:
                _required_sha256(selection, "expected_answer_sha256")
            _required_sha256(selection, "query_sha256")
            _required_sha256(selection, "references_sha256")
            selection_digests.add(
                _required_sha256(selection, "selected_record_sha256")
            )
            _required_sha256(selection, "selection_priority_sha256")
            if _required_int(selection, "selection_record_index") == 0:
                raise ValueError(
                    "provenance selection_record_index must be positive"
                )
            _required_sha256(selection, "source_document_context_sha256")
        if len(selection_digests) != examples_per_dataset:
            raise ValueError("provenance selected source records must be distinct")
        if len(identity_digests) != examples_per_dataset:
            raise ValueError("provenance selected example identities must be distinct")
        if _canonical_sha256(selected) != _required_sha256(
            record,
            "selected_records_sha256",
        ):
            raise ValueError("selected_records_sha256 does not match selections")
    if _canonical_sha256(records) != _required_sha256(provenance, "sources_sha256"):
        raise ValueError("sources_sha256 does not match the source records")
    return records


def _output_records_from_provenance(
    provenance: Mapping[str, Any],
    *,
    examples_per_dataset: int,
) -> tuple[Mapping[str, Any], ...]:
    records = _mapping_sequence(provenance.get("outputs"), label="outputs")
    expected_order = [
        (dataset, target)
        for target in MAIN_LATENCY_TARGET_SEGMENT_COUNTS
        for dataset in SUPPORTED_V1_DATASETS
    ]
    actual_order = [
        (record.get("dataset"), record.get("input_tokens_target")) for record in records
    ]
    if actual_order != expected_order:
        raise ValueError("provenance outputs are incomplete or out of canonical order")
    for index, record in enumerate(records):
        if set(record) != _OUTPUT_PROVENANCE_FIELDS:
            raise ValueError(
                f"provenance outputs[{index}] does not use the closed schema"
            )
        _required_int(record, "byte_count")
        _required_str(record, "dataset")
        _required_int(record, "input_tokens_target")
        _required_sha256(record, "jsonl_sha256")
        if _required_int(record, "record_count") != examples_per_dataset:
            raise ValueError(
                f"provenance outputs[{index}] must contain exactly "
                f"{examples_per_dataset} records"
            )
        _required_str(record, "relative_path")
        _required_int(record, "segment_count")
        prepared_records = _mapping_sequence(
            record.get("records"),
            label=f"outputs[{index}].records",
        )
        if len(prepared_records) != examples_per_dataset:
            raise ValueError(
                f"provenance outputs[{index}].records has invalid coverage"
            )
        identity_digests: set[str] = set()
        for record_index, prepared in enumerate(prepared_records, start=1):
            if set(prepared) != _PREPARED_RECORD_PROVENANCE_FIELDS:
                raise ValueError(
                    f"provenance outputs[{index}].records[{record_index - 1}] "
                    "does not use the closed schema"
                )
            if _required_int(prepared, "record_index") != record_index:
                raise ValueError("prepared provenance record_index is not canonical")
            identity_digests.add(
                _required_sha256(prepared, "example_identity_sha256")
            )
            _required_sha256(prepared, "source_record_sha256")
            _required_sha256(prepared, "jsonl_row_sha256")
        if len(identity_digests) != examples_per_dataset:
            raise ValueError("prepared output example identities must be distinct")
        if _canonical_sha256(prepared_records) != _required_sha256(
            record,
            "records_sha256",
        ):
            raise ValueError("records_sha256 does not match prepared records")
    return records


def _verify_sources_against_provenance(
    sources: Mapping[str, _SourceFile],
    *,
    source_records: Sequence[Mapping[str, Any]],
    output_records: Sequence[Mapping[str, Any]],
    examples_per_dataset: int,
    tokenizer: MainLatencyTokenizer,
) -> None:
    source_by_dataset = {
        _required_str(record, "dataset"): record for record in source_records
    }
    outputs_by_dataset: dict[str, list[Mapping[str, Any]]] = {
        dataset: [] for dataset in SUPPORTED_V1_DATASETS
    }
    for output in output_records:
        outputs_by_dataset[_required_str(output, "dataset")].append(output)
    rebuilt_sources: list[Mapping[str, Any]] = []
    for dataset in SUPPORTED_V1_DATASETS:
        source = sources[dataset]
        expected = source_by_dataset[dataset]
        if source.raw_sha256 != _required_sha256(expected, "jsonl_sha256"):
            raise ValueError(f"source JSONL digest mismatch for {dataset}")
        if source.byte_count != _required_int(expected, "byte_count"):
            raise ValueError(f"source JSONL byte count mismatch for {dataset}")
        if len(source.rows) != _required_int(expected, "record_count"):
            raise ValueError(f"source JSONL record count mismatch for {dataset}")
        expected_selections = _mapping_sequence(
            expected.get("selected_records"),
            label=f"sources[{dataset}].selected_records",
        )
        selected_rows, _outputs = _select_and_prepare_dataset(
            source,
            tokenizer=tokenizer,
            examples_per_dataset=examples_per_dataset,
        )
        expected_selection_digests = tuple(
            _required_sha256(selection, "selected_record_sha256")
            for selection in expected_selections
        )
        recomputed_selection_digests = tuple(
            row.record_sha256 for row in selected_rows
        )
        if recomputed_selection_digests != expected_selection_digests:
            raise ValueError(
                f"selected source records do not match content-hash selection for "
                f"{dataset}"
            )
        rebuilt = _describe_source_file(source, selected_rows)
        if rebuilt != expected:
            raise ValueError(f"selected source provenance mismatch for {dataset}")
        rebuilt_sources.append(rebuilt)
        dataset_outputs = outputs_by_dataset[dataset]
        if len(dataset_outputs) != len(MAIN_LATENCY_TARGET_SEGMENT_COUNTS):
            raise ValueError(f"prepared output coverage mismatch for {dataset}")
        source_selections = _mapping_sequence(
            rebuilt.get("selected_records"),
            label=f"rebuilt sources[{dataset}].selected_records",
        )
        if len(source_selections) != examples_per_dataset:
            raise ValueError(f"selected source coverage mismatch for {dataset}")
        for output in dataset_outputs:
            prepared_records = _mapping_sequence(
                output.get("records"),
                label=f"outputs[{dataset}].records",
            )
            if len(prepared_records) != examples_per_dataset:
                raise ValueError(
                    f"prepared output record coverage mismatch for {dataset}"
                )
            for prepared, selection in zip(
                prepared_records,
                source_selections,
                strict=True,
            ):
                for field_name in (
                    "example_identity_sha256",
                    "expected_answer_sha256",
                    "query_sha256",
                    "references_sha256",
                    "source_document_context_sha256",
                ):
                    if prepared.get(field_name) != selection.get(field_name):
                        raise ValueError(
                            f"prepared output changed source {field_name} for {dataset}"
                        )
                if prepared.get("source_record_sha256") != selection.get(
                    "selected_record_sha256"
                ):
                    raise ValueError(
                        f"prepared output source record changed for {dataset}"
                    )
    if _canonical_sha256(rebuilt_sources) != _canonical_sha256(source_records):
        raise ValueError("source provenance changed during verification")


def _verify_identity_reuse_across_targets(
    source_records: Sequence[Mapping[str, Any]],
    output_records: Sequence[Mapping[str, Any]],
) -> None:
    source_identities = {
        _required_str(source, "dataset"): tuple(
            _required_sha256(selected, "example_identity_sha256")
            for selected in _mapping_sequence(
                source.get("selected_records"),
                label="source selected_records",
            )
        )
        for source in source_records
    }
    for output in output_records:
        dataset = _required_str(output, "dataset")
        output_identities = tuple(
            _required_sha256(prepared, "example_identity_sha256")
            for prepared in _mapping_sequence(
                output.get("records"),
                label="output records",
            )
        )
        if output_identities != source_identities[dataset]:
            raise ValueError(
                f"prepared output identities are not reused across targets for "
                f"{dataset}"
            )


def _bundle_sha256(output_records: Sequence[Mapping[str, Any]]) -> str:
    manifest = [
        {
            "jsonl_sha256": _required_sha256(record, "jsonl_sha256"),
            "relative_path": _required_str(record, "relative_path"),
        }
        for record in output_records
    ]
    return _canonical_sha256(manifest)


def _expected_output_keys() -> set[tuple[str, int]]:
    return {
        (dataset, target)
        for target in MAIN_LATENCY_TARGET_SEGMENT_COUNTS
        for dataset in SUPPORTED_V1_DATASETS
    }


def _relative_output_path(dataset: str, target: int) -> str:
    if dataset not in SUPPORTED_V1_DATASETS:
        raise ValueError(f"unsupported main-latency dataset: {dataset}")
    if target not in MAIN_LATENCY_TARGET_SEGMENT_COUNTS:
        raise ValueError(f"unsupported main-latency token target: {target}")
    return f"{target}/{dataset}.jsonl"


def _parse_source_specs(values: Sequence[str]) -> Mapping[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        dataset, separator, path = value.partition("=")
        if not separator or not dataset or not path:
            raise ValueError("--source must use DATASET=PATH")
        if dataset in parsed:
            raise ValueError(f"duplicate --source dataset: {dataset}")
        parsed[dataset] = path
    expected = set(SUPPORTED_V1_DATASETS)
    if set(parsed) != expected:
        raise ValueError(
            f"--source must contain exactly {sorted(expected)}; got {sorted(parsed)}"
        )
    return parsed


def _token_ids(tokenizer: MainLatencyTokenizer, text: str) -> tuple[int, ...]:
    values = tokenizer.encode(
        text,
        add_special_tokens=MAIN_LATENCY_ADD_SPECIAL_TOKENS,
    )
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("tokenizer.encode() must return a sequence of token ids")
    token_ids = tuple(values)
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise ValueError("tokenizer.encode() returned an invalid token id")
    return token_ids


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return _canonical_sha256(list(token_ids))


def _example_identity_sha256(example: BenchmarkExample) -> str:
    return _canonical_sha256(
        {"dataset": example.dataset, "example_id": example.example_id}
    )


def _selection_priority_sha256(row: _SourceRow) -> str:
    """Return the source-order-independent publication selection priority."""

    return _canonical_sha256(
        {
            "dataset": row.example.dataset,
            "domain": _SELECTION_DOMAIN,
            "example_identity_sha256": _example_identity_sha256(row.example),
            "source_record_sha256": row.record_sha256,
        }
    )


def _text_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _optional_text_sha256(text: str | None) -> str | None:
    return None if text is None else _text_sha256(text)


def _canonical_jsonl_records(
    content: bytes,
    *,
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    if not content or not content.endswith(b"\n"):
        raise ValueError(f"{label} must be non-empty newline-terminated JSONL")
    raw_lines = content[:-1].split(b"\n")
    if any(not line for line in raw_lines):
        raise ValueError(f"{label} must not contain empty JSONL rows")
    records: list[Mapping[str, Any]] = []
    for record_index, line in enumerate(raw_lines, start=1):
        record = _json_object_from_bytes(
            line,
            label=f"{label} row {record_index}",
        )
        if line != _canonical_json_bytes(record, pretty=False):
            raise ValueError(
                f"{label} row {record_index} is not in canonical JSONL encoding"
            )
        records.append(record)
    return tuple(records)


def _json_object_from_bytes(content: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return cast(Mapping[str, Any], value)


def _mapping_sequence(value: Any, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"provenance {label} must be an array")
    records: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"provenance {label}[{index}] must be an object")
        records.append(item)
    return tuple(records)


def _required_str(record: Mapping[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"provenance {field_name} must be a non-empty string")
    return value


def _required_int(record: Mapping[str, Any], field_name: str) -> int:
    value = record.get(field_name)
    if type(value) is not int or value < 0:
        raise ValueError(f"provenance {field_name} must be a non-negative integer")
    return value


def _required_sha256(record: Mapping[str, Any], field_name: str) -> str:
    value = _required_str(record, field_name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"provenance {field_name} must be a lowercase SHA-256")
    return value


def _canonical_json_bytes(value: Any, *, pretty: bool) -> bytes:
    kwargs: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    suffix = "\n" if pretty else ""
    return (json.dumps(value, **kwargs) + suffix).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value, pretty=False)).hexdigest()


def _write_if_absent_or_identical(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file():
            raise ValueError(f"artifact path is not a file: {path}")
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite conflicting artifact: {path}")
        return
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _require_absent_or_identical(path: Path, content: bytes) -> None:
    if not path.exists():
        return
    if not path.is_file():
        raise ValueError(f"artifact path is not a file: {path}")
    if path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite conflicting artifact: {path}")


__all__ = [
    "MAIN_LATENCY_INPUT_RECORD_TYPE",
    "MAIN_LATENCY_INPUT_SCHEMA_VERSION",
    "MAIN_LATENCY_TOKENIZER_ID",
    "MAIN_LATENCY_TOKENIZER_REVISION",
    "MAIN_LATENCY_ADD_SPECIAL_TOKENS",
    "MAIN_LATENCY_EXAMPLES_PER_DATASET",
    "MAIN_LATENCY_TARGET_SEGMENT_COUNTS",
    "MAIN_LATENCY_PROVENANCE_FILENAME",
    "MainLatencyTokenizer",
    "MainLatencyInputFile",
    "PreparedMainLatencyInputs",
    "prepare_main_latency_inputs",
    "verify_main_latency_inputs",
    "load_main_latency_tokenizer",
    "main_latency_inputs_main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main_latency_inputs_main())
