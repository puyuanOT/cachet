"""Content-addressed matched score canary for Baseline and Vanilla KV.

This module deliberately prepares a small, descriptive canary.  It is not a
full-dataset or publication protocol.  Both arms consume the exact same
prepared JSONL files; only the Vanilla job enriches a private copy with
pre-RoPE per-document handoffs at run time.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

from document_kv_cache._benchmark_datasets import (
    _example_from_record,
    load_benchmark_jsonl,
)
from document_kv_cache.benchmarks import (
    DEFAULT_SYSTEM_PROMPT_POSITION,
    DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    SUPPORTED_V1_DATASETS,
    BenchmarkExample,
    benchmark_cache_prefix_segments,
    build_prompt_parts,
    method_benchmark_arm,
    resolve_system_prompt_position,
)
from document_kv_cache.dataset_prep import (
    REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID,
    REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION,
    load_representative_hotpotqa_tokenizer,
)
from document_kv_cache.storage import local_path
from document_kv_cache.workflow import SourceChunk, SourceDocument


SCORE_CANARY_RECORD_TYPE = "cachet.vanilla_score_canary"
SCORE_CANARY_SCHEMA_VERSION = 1
SCORE_CANARY_PROTOCOL_ID = "vanilla-score-canary-8k-n5-v1"
SCORE_CANARY_INPUT_TOKENS = 8192
SCORE_CANARY_EXAMPLES_PER_DATASET = 5
SCORE_CANARY_MAX_TOKENS = 64
SCORE_CANARY_REQUEST_PARALLELISM = 4
SCORE_CANARY_REPEATS = 1
SCORE_CANARY_ADD_SPECIAL_TOKENS = False
SCORE_CANARY_SELECTION_SEED = "cachet:vanilla-score-canary:selection:v1"
SCORE_CANARY_KV_BYTES_PER_TOKEN = 73_728

_PADDING_DOCUMENT_ID = "cachet-score-canary-length-padding"
_PADDING_DOCUMENT_TITLE = "Deterministic irrelevant score-canary length padding"
_PADDING_CHUNK_ID = "length-padding"
_PADDING_UNITS = (" padding", " x", "x", "0", ".", "\n")
_TRANSFER_FIELDS = frozenset({"kv_transfer_params", "arm_kv_transfer_params"})
_PRIMARY_METRICS = MappingProxyType(
    {
        "biography": "answer_found",
        "hotpotqa": "f1",
        "musique": "answer_found",
        "niah": "exact_match",
    }
)
_SCORER_IDENTITIES = MappingProxyType(
    {
        "biography": "cachet.answer_diagnostics@1",
        "hotpotqa": "hotpotqa.official_answer@hotpot_evaluate_v1@3635853403a8",
        "musique": "cachet.answer_diagnostics@1",
        "niah": "cachet.answer_diagnostics@1",
    }
)


class ScoreCanaryTokenizer(Protocol):
    """Tokenizer surface needed for exact prompt sizing."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class ScoreCanaryProtocol:
    """Immutable logical protocol; smaller values are useful only in tests."""

    protocol_id: str = SCORE_CANARY_PROTOCOL_ID
    input_tokens: int = SCORE_CANARY_INPUT_TOKENS
    examples_per_dataset: int = SCORE_CANARY_EXAMPLES_PER_DATASET
    max_tokens: int = SCORE_CANARY_MAX_TOKENS
    request_parallelism: int = SCORE_CANARY_REQUEST_PARALLELISM
    repeats: int = SCORE_CANARY_REPEATS
    selection_seed: str = SCORE_CANARY_SELECTION_SEED
    primary_metrics: Mapping[str, str] = field(
        default_factory=lambda: dict(_PRIMARY_METRICS)
    )

    def __post_init__(self) -> None:
        for name in ("protocol_id", "selection_seed"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "input_tokens",
            "examples_per_dataset",
            "max_tokens",
            "request_parallelism",
            "repeats",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        metrics = dict(self.primary_metrics)
        if set(metrics) != set(SUPPORTED_V1_DATASETS):
            raise ValueError("primary_metrics must name exactly all four V1 datasets")
        if any(not isinstance(metric, str) or not metric for metric in metrics.values()):
            raise ValueError("primary_metrics values must be non-empty strings")
        object.__setattr__(self, "primary_metrics", MappingProxyType(metrics))


DEFAULT_SCORE_CANARY_PROTOCOL = ScoreCanaryProtocol()


@dataclass(frozen=True, slots=True)
class PreparedScoreCanary:
    """Paths and digests for one prepared matched suite."""

    manifest_path: Path
    manifest_sha256: str
    suite_sha256: str
    dataset_paths: Mapping[str, Path]
    example_count: int


@dataclass(frozen=True, slots=True)
class ScoreCanaryValidation:
    """Successful validation summary."""

    manifest_path: Path
    manifest_sha256: str
    suite_sha256: str
    example_count: int


@dataclass(frozen=True, slots=True)
class _SourceRow:
    record: Mapping[str, Any]
    example: BenchmarkExample
    record_sha256: str
    selection_rank_sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedRow:
    record: Mapping[str, Any]
    example: BenchmarkExample
    source_record_sha256: str
    selection_rank_sha256: str
    unpadded_prompt_sha256: str
    unpadded_token_count: int
    padding_unit: str | None
    padding_repetitions: int
    token_composition: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _PaddingResult:
    filler: str
    unit: str
    repetitions: int
    example: BenchmarkExample


class _CandidateCannotSatisfyProtocol(ValueError):
    """Internal signal used to skip a content-ranked but incompatible row."""


def prepare_score_canary(
    sources: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    tokenizer: ScoreCanaryTokenizer | None = None,
    protocol: ScoreCanaryProtocol = DEFAULT_SCORE_CANARY_PROTOCOL,
) -> PreparedScoreCanary:
    """Prepare exact-length, content-addressed inputs for both isolated arms."""

    if not isinstance(protocol, ScoreCanaryProtocol):
        raise TypeError("protocol must be ScoreCanaryProtocol")
    if resolve_system_prompt_position() != DEFAULT_SYSTEM_PROMPT_POSITION:
        raise ValueError("score canary preparation requires system_prompt_position=start")
    source_paths = _validated_sources(sources)
    destination = local_path(str(output_dir))
    resolved_tokenizer = tokenizer or load_representative_hotpotqa_tokenizer()
    protocol_record = _protocol_record(protocol)
    protocol_sha256 = _canonical_sha256(protocol_record)

    prepared_datasets: dict[str, Mapping[str, Any]] = {}
    output_contents: dict[str, bytes] = {}
    output_names: dict[str, str] = {}
    for dataset in SUPPORTED_V1_DATASETS:
        source_path = source_paths[dataset]
        source_rows = _load_source_rows(
            source_path,
            dataset=dataset,
            selection_seed=protocol.selection_seed,
        )
        prepared_rows = _select_and_prepare_rows(
            source_rows,
            tokenizer=resolved_tokenizer,
            target=protocol.input_tokens,
            count=protocol.examples_per_dataset,
            dataset=dataset,
        )
        content = _jsonl_bytes(tuple(row.record for row in prepared_rows))
        content_sha256 = sha256(content).hexdigest()
        output_name = f"{dataset}-{content_sha256}.jsonl"
        output_contents[dataset] = content
        output_names[dataset] = output_name
        prepared_datasets[dataset] = _dataset_manifest_record(
            source_path=source_path,
            source_record_count=len(source_rows),
            output_name=output_name,
            output_content=content,
            prepared_rows=prepared_rows,
            protocol=protocol,
        )

    suite_sha256 = _suite_sha256(protocol_sha256, prepared_datasets)
    manifest: dict[str, Any] = {
        "datasets": prepared_datasets,
        "protocol": protocol_record,
        "protocol_sha256": protocol_sha256,
        "record_type": SCORE_CANARY_RECORD_TYPE,
        "schema_version": SCORE_CANARY_SCHEMA_VERSION,
        "suite_sha256": suite_sha256,
    }
    manifest_bytes = _canonical_json_bytes(manifest, pretty=True)
    manifest_destination = (
        local_path(str(manifest_path))
        if manifest_path is not None
        else destination / f"{protocol.protocol_id}-manifest.json"
    )
    source_resolutions = {path.resolve() for path in source_paths.values()}
    if manifest_destination.resolve() in source_resolutions:
        raise ValueError("manifest_path must not overwrite a source dataset")
    destination.mkdir(parents=True, exist_ok=True)
    dataset_paths: dict[str, Path] = {}
    for dataset in SUPPORTED_V1_DATASETS:
        output_path = destination / output_names[dataset]
        if output_path.resolve() in source_resolutions:
            raise ValueError("prepared output must not overwrite a source dataset")
        _atomic_write_bytes(output_path, output_contents[dataset])
        dataset_paths[dataset] = output_path
    _atomic_write_bytes(manifest_destination, manifest_bytes)

    validate_score_canary_manifest(
        manifest_destination,
        tokenizer=resolved_tokenizer,
        sources=source_paths,
    )
    return PreparedScoreCanary(
        manifest_path=manifest_destination,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        suite_sha256=suite_sha256,
        dataset_paths=MappingProxyType(dataset_paths),
        example_count=len(SUPPORTED_V1_DATASETS) * protocol.examples_per_dataset,
    )


def validate_score_canary_manifest(
    manifest_path: str | Path,
    *,
    tokenizer: ScoreCanaryTokenizer | None = None,
    sources: Mapping[str, str | Path] | None = None,
) -> ScoreCanaryValidation:
    """Fail closed if a prepared suite or its declared protocol has drifted."""

    path = local_path(str(manifest_path))
    raw_manifest = path.read_bytes()
    manifest = _json_object(raw_manifest, label="score canary manifest")
    expected_top_level = {
        "datasets",
        "protocol",
        "protocol_sha256",
        "record_type",
        "schema_version",
        "suite_sha256",
    }
    if set(manifest) != expected_top_level:
        raise ValueError("score canary manifest has an unexpected top-level schema")
    if manifest.get("record_type") != SCORE_CANARY_RECORD_TYPE:
        raise ValueError("score canary manifest record_type is invalid")
    if manifest.get("schema_version") != SCORE_CANARY_SCHEMA_VERSION:
        raise ValueError("score canary manifest schema_version is invalid")
    protocol_record = _required_mapping(manifest, "protocol")
    if manifest.get("protocol_sha256") != _canonical_sha256(protocol_record):
        raise ValueError("score canary protocol_sha256 does not match protocol")
    protocol = _protocol_from_record(protocol_record)
    if protocol_record != _protocol_record(protocol):
        raise ValueError("score canary protocol contains unsupported or drifted settings")
    dataset_records = _required_mapping(manifest, "datasets")
    if set(dataset_records) != set(SUPPORTED_V1_DATASETS):
        raise ValueError("score canary manifest must contain exactly all four datasets")
    expected_suite_sha256 = _suite_sha256(
        cast(str, manifest["protocol_sha256"]),
        cast(Mapping[str, Mapping[str, Any]], dataset_records),
    )
    if manifest.get("suite_sha256") != expected_suite_sha256:
        raise ValueError("score canary suite_sha256 does not match dataset outputs")
    if resolve_system_prompt_position() != DEFAULT_SYSTEM_PROMPT_POSITION:
        raise ValueError("score canary validation requires system_prompt_position=start")
    resolved_tokenizer = tokenizer or load_representative_hotpotqa_tokenizer()
    total = 0
    for dataset in SUPPORTED_V1_DATASETS:
        raw_dataset_record = dataset_records[dataset]
        if not isinstance(raw_dataset_record, Mapping):
            raise ValueError(f"datasets.{dataset} must be an object")
        total += _validate_dataset_output(
            path.parent,
            dataset=dataset,
            record=raw_dataset_record,
            protocol=protocol,
            tokenizer=resolved_tokenizer,
        )
    if sources is not None:
        source_paths = _validated_sources(sources)
        for dataset in SUPPORTED_V1_DATASETS:
            raw_dataset_record = cast(Mapping[str, Any], dataset_records[dataset])
            _validate_source_identity(
                source_paths[dataset],
                dataset=dataset,
                source_record=_required_mapping(raw_dataset_record, "source"),
            )
    return ScoreCanaryValidation(
        manifest_path=path,
        manifest_sha256=sha256(raw_manifest).hexdigest(),
        suite_sha256=expected_suite_sha256,
        example_count=total,
    )


def score_canary_main(argv: Sequence[str] | None = None) -> int:
    """CLI for production preparation and validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Canonical source as DATASET=JSONL_PATH; repeat for all four datasets.",
    )
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--manifest-path")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--manifest", required=True)
    validate_parser.add_argument(
        "--source",
        action="append",
        help="Optional DATASET=JSONL_PATH source identity checks; repeat four times.",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_score_canary(
                _parse_source_specs(args.source),
                args.output_dir,
                manifest_path=args.manifest_path,
            )
            response = {
                "dataset_paths": {
                    dataset: str(path)
                    for dataset, path in result.dataset_paths.items()
                },
                "example_count": result.example_count,
                "manifest_path": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "ok": True,
                "suite_sha256": result.suite_sha256,
            }
        else:
            source_specs = None
            if args.source is not None:
                source_specs = _parse_source_specs(args.source)
            validated = validate_score_canary_manifest(
                args.manifest,
                sources=source_specs,
            )
            response = {
                "example_count": validated.example_count,
                "manifest_path": str(validated.manifest_path),
                "manifest_sha256": validated.manifest_sha256,
                "ok": True,
                "suite_sha256": validated.suite_sha256,
            }
    except Exception as exc:
        print(
            json.dumps(
                {"error": str(exc), "error_type": type(exc).__name__, "ok": False},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(response, sort_keys=True))
    return 0


def _validated_sources(
    sources: Mapping[str, str | Path],
) -> Mapping[str, Path]:
    if not isinstance(sources, Mapping):
        raise TypeError("sources must be a mapping")
    if set(sources) != set(SUPPORTED_V1_DATASETS):
        raise ValueError("sources must name exactly all four V1 datasets")
    paths: dict[str, Path] = {}
    resolutions: set[Path] = set()
    for dataset in SUPPORTED_V1_DATASETS:
        path = local_path(str(sources[dataset]))
        if not path.is_file():
            raise ValueError(f"source dataset does not exist: {path}")
        resolution = path.resolve()
        if resolution in resolutions:
            raise ValueError("each dataset must use a distinct source file")
        resolutions.add(resolution)
        paths[dataset] = path
    return MappingProxyType(paths)


def _load_source_rows(
    path: Path,
    *,
    dataset: str,
    selection_seed: str,
) -> tuple[_SourceRow, ...]:
    rows: list[_SourceRow] = []
    identities: set[tuple[str, str]] = set()
    try:
        handle = path.open("r", encoding="utf-8")
        with handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{dataset} source line {line_number} is invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(raw_record, Mapping):
                    raise ValueError(f"{dataset} source line {line_number} must be an object")
                if _TRANSFER_FIELDS.intersection(raw_record):
                    raise ValueError(
                        f"{dataset} source line {line_number} contains KV transfer metadata"
                    )
                record = _json_materialize_mapping(raw_record)
                try:
                    example = _example_from_record(
                        record,
                        default_dataset=dataset,
                        record_index=len(rows) + 1,
                        require_dataset=True,
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{dataset} source line {line_number}: {exc}"
                    ) from exc
                if not example.references:
                    raise ValueError(
                        f"{dataset} source line {line_number} must include an expected answer"
                    )
                if any(
                    document.document_id == _PADDING_DOCUMENT_ID
                    for document in example.documents
                ):
                    raise ValueError(f"{dataset} source reserves the canary padding id")
                identity = (example.dataset, example.example_id)
                if identity in identities:
                    raise ValueError(f"{dataset} source contains duplicate example identities")
                identities.add(identity)
                record_sha256 = _canonical_sha256(record)
                rank = sha256(
                    (
                        f"{selection_seed}\0{dataset}\0{record_sha256}"
                    ).encode("utf-8")
                ).hexdigest()
                rows.append(
                    _SourceRow(
                        record=record,
                        example=example,
                        record_sha256=record_sha256,
                        selection_rank_sha256=rank,
                    )
                )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{dataset} source must be UTF-8") from exc
    if not rows:
        raise ValueError(f"{dataset} source must contain at least one record")
    return tuple(rows)


def _select_and_prepare_rows(
    source_rows: Sequence[_SourceRow],
    *,
    tokenizer: ScoreCanaryTokenizer,
    target: int,
    count: int,
    dataset: str,
) -> tuple[_PreparedRow, ...]:
    selected: list[_PreparedRow] = []
    for row in sorted(
        source_rows,
        key=lambda value: (
            value.selection_rank_sha256,
            value.record_sha256,
        ),
    ):
        prepared = _prepare_source_row(row, tokenizer=tokenizer, target=target)
        if prepared is not None:
            selected.append(prepared)
        if len(selected) == count:
            return tuple(selected)
    raise ValueError(
        f"{dataset} source cannot produce {count} exact {target}-token examples"
    )


def _prepare_source_row(
    row: _SourceRow,
    *,
    tokenizer: ScoreCanaryTokenizer,
    target: int,
) -> _PreparedRow | None:
    unpadded_prompt = build_prompt_parts(row.example).prefill_prompt
    unpadded_count = _token_count(tokenizer, unpadded_prompt)
    if unpadded_count > target:
        return None
    if unpadded_count == target:
        try:
            composition = _token_composition_attestation(
                row.example,
                tokenizer=tokenizer,
            )
        except _CandidateCannotSatisfyProtocol:
            return None
        return _PreparedRow(
            record=row.record,
            example=row.example,
            source_record_sha256=row.record_sha256,
            selection_rank_sha256=row.selection_rank_sha256,
            unpadded_prompt_sha256=sha256(unpadded_prompt.encode("utf-8")).hexdigest(),
            unpadded_token_count=unpadded_count,
            padding_unit=None,
            padding_repetitions=0,
            token_composition=composition,
        )
    padding = _exact_padding(row.example, tokenizer=tokenizer, target=target)
    if padding is None:
        return None
    record = _json_materialize_mapping(row.record)
    raw_documents = record.get("documents")
    if not isinstance(raw_documents, list):
        raise ValueError("canonical score source documents must be an array")
    raw_documents.append(_padding_document_record(padding.filler))
    prompt = build_prompt_parts(padding.example).prefill_prompt
    if _token_count(tokenizer, prompt) != target:
        raise ValueError("score canary exact-padding verification drifted")
    try:
        composition = _token_composition_attestation(
            padding.example,
            tokenizer=tokenizer,
        )
    except _CandidateCannotSatisfyProtocol:
        return None
    return _PreparedRow(
        record=record,
        example=padding.example,
        source_record_sha256=row.record_sha256,
        selection_rank_sha256=row.selection_rank_sha256,
        unpadded_prompt_sha256=sha256(unpadded_prompt.encode("utf-8")).hexdigest(),
        unpadded_token_count=unpadded_count,
        padding_unit=padding.unit,
        padding_repetitions=padding.repetitions,
        token_composition=composition,
    )


def _token_composition_attestation(
    example: BenchmarkExample,
    *,
    tokenizer: ScoreCanaryTokenizer,
) -> Mapping[str, Any]:
    """Mirror the live per-document handoff tokenizer-composition gate."""

    parts = build_prompt_parts(example)
    segments = benchmark_cache_prefix_segments(example)
    segment_token_ids = tuple(
        _token_ids(tokenizer, text) for _chunk_id, text in segments
    )
    composed_ids = tuple(
        token_id for token_ids in segment_token_ids for token_id in token_ids
    )
    prefix_ids = _token_ids(tokenizer, parts.cache_prefix_text)
    full_ids = _token_ids(tokenizer, parts.prefill_prompt)
    if composed_ids != prefix_ids:
        raise _CandidateCannotSatisfyProtocol(
            "independently tokenized Vanilla segments do not compose to the "
            "logical cache prefix"
        )
    if len(prefix_ids) >= len(full_ids) or full_ids[: len(prefix_ids)] != prefix_ids:
        raise _CandidateCannotSatisfyProtocol(
            "logical cache prefix is not a strict leading token prefix of the prompt"
        )
    segment_records = [
        {
            "chunk_id": chunk_id,
            "segment_index": index,
            "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
            "token_count": len(token_ids),
            "token_ids_sha256": _token_ids_sha256(token_ids),
        }
        for index, ((chunk_id, text), token_ids) in enumerate(
            zip(segments, segment_token_ids, strict=True)
        )
    ]
    return {
        "cache_prefix_is_strict_full_prompt_prefix": True,
        "cache_prefix_token_count": len(prefix_ids),
        "cache_prefix_token_ids_sha256": _token_ids_sha256(prefix_ids),
        "composed_equals_cache_prefix": True,
        "composed_token_count": len(composed_ids),
        "composed_token_ids_sha256": _token_ids_sha256(composed_ids),
        "full_prompt_token_count": len(full_ids),
        "full_prompt_token_ids_sha256": _token_ids_sha256(full_ids),
        "segment_count": len(segments),
        "segments": segment_records,
    }


def _exact_padding(
    example: BenchmarkExample,
    *,
    tokenizer: ScoreCanaryTokenizer,
    target: int,
) -> _PaddingResult | None:
    for unit in _PADDING_UNITS:
        result = _exact_repeated_unit(
            example,
            tokenizer=tokenizer,
            target=target,
            unit=unit,
        )
        if result is not None:
            return result
    return None


def _exact_repeated_unit(
    example: BenchmarkExample,
    *,
    tokenizer: ScoreCanaryTokenizer,
    target: int,
    unit: str,
) -> _PaddingResult | None:
    counts: dict[int, int] = {}

    def count(repetitions: int) -> int:
        if repetitions not in counts:
            candidate = _with_padding_document(example, unit * repetitions)
            counts[repetitions] = _token_count(
                tokenizer,
                build_prompt_parts(candidate).prefill_prompt,
            )
        return counts[repetitions]

    maximum = target * 32
    low = 0
    high = 1
    while high <= maximum and count(high) < target:
        low = high
        high *= 2
    high = min(high, maximum)
    if count(high) < target:
        return None
    while low + 1 < high:
        middle = (low + high) // 2
        if count(middle) < target:
            low = middle
        else:
            high = middle
    for repetitions in range(max(1, low - 16), min(maximum, high + 16) + 1):
        if count(repetitions) == target:
            filler = unit * repetitions
            return _PaddingResult(
                filler=filler,
                unit=unit,
                repetitions=repetitions,
                example=_with_padding_document(example, filler),
            )
    return None


def _with_padding_document(example: BenchmarkExample, filler: str) -> BenchmarkExample:
    document = SourceDocument(
        document_id=_PADDING_DOCUMENT_ID,
        chunks=(SourceChunk(chunk_id=_PADDING_CHUNK_ID, text=filler),),
        metadata={"title": _PADDING_DOCUMENT_TITLE},
    )
    return replace(example, documents=(*example.documents, document))


def _padding_document_record(filler: str) -> dict[str, Any]:
    return {
        "chunks": [{"chunk_id": _PADDING_CHUNK_ID, "text": filler}],
        "document_id": _PADDING_DOCUMENT_ID,
        "title": _PADDING_DOCUMENT_TITLE,
    }


def _protocol_record(protocol: ScoreCanaryProtocol) -> Mapping[str, Any]:
    total_examples = len(SUPPORTED_V1_DATASETS) * protocol.examples_per_dataset
    vanilla_bytes = total_examples * protocol.input_tokens * SCORE_CANARY_KV_BYTES_PER_TOKEN
    vanilla_arm = method_benchmark_arm(
        "vanilla_prefill",
        physical_transform_id="cachet.vanilla.per_document_segments",
    )
    vanilla_arm_record = {
        "arm_id": vanilla_arm.arm_id,
        "cache_method": vanilla_arm.cache_method,
        "connector_mode": vanilla_arm.connector_mode,
        "description": vanilla_arm.description,
        "implementation_kind": vanilla_arm.implementation_kind,
        "method_config_digest": vanilla_arm.method_config_digest,
        "method_version": vanilla_arm.method_version,
        "physical_transform_id": vanilla_arm.physical_transform_id,
        "physical_transform_version": vanilla_arm.physical_transform_version,
        "requires_cachet_handoff": vanilla_arm.requires_cachet_handoff,
        "uses_cache": vanilla_arm.uses_cache,
        "variant_id": vanilla_arm.variant_id,
    }
    dataset_cli = [
        item
        for dataset in SUPPORTED_V1_DATASETS
        for item in ("--dataset", f"{dataset}=<MANIFEST_DIR>/{dataset}-<SHA256>.jsonl")
    ]
    common_cli = [
        "--model-id",
        REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID,
        "--model-revision",
        REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION,
        "--tokenizer-revision",
        REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION,
        "--model-dtype",
        "bfloat16",
        "--model-quantization",
        "bitsandbytes",
        "--kv-cache-dtype",
        "fp8_e5m2",
        "--max-tokens",
        str(protocol.max_tokens),
        "--max-model-len",
        "8512",
        "--max-num-seqs",
        str(protocol.request_parallelism),
        "--gpu-memory-utilization",
        "0.9",
        "--hardware-target",
        "aws-g6-l4",
        "--benchmark-repeats",
        str(protocol.repeats),
        "--request-parallelism",
        str(protocol.request_parallelism),
        "--benchmark-interleave-examples",
        "--benchmark-evidence-policy",
        "canary",
        "--benchmark-prefix-cache-salt-mode",
        "per_request",
        "--payload-cache-max-bytes",
        "0",
        *dataset_cli,
    ]
    return {
        "arms": {
            "baseline": {
                "extra_cli_args": ["--benchmark-arm", "baseline_prefill"],
                "job_id": f"{protocol.protocol_id}:baseline",
                "task_timeout_seconds": 7200,
            },
            "vanilla": {
                "arm_spec": vanilla_arm_record,
                "extra_cli_args": [
                    "--benchmark-arm-spec-json",
                    "<CANONICAL_JSON_OF_ARM_SPEC>",
                    "--benchmark-handoff-generator-factory",
                    (
                        "document_kv_cache.transformers_generator:"
                        "build_pre_rope_transformers_kv_chunk_generator"
                    ),
                    "--benchmark-handoff-dtype",
                    "fp8_e5m2",
                    "--benchmark-handoff-align-bytes",
                    "4096",
                    "--benchmark-handoff-generation-timeout-seconds",
                    "10800",
                    "--benchmark-handoff-output-dir",
                    "<LOCAL_NVME_OUTPUT_DIR>/handoffs",
                    "--benchmark-handoff-chunk-per-document",
                    "--benchmark-handoff-cache-method",
                    "vanilla_prefill",
                ],
                "job_id": f"{protocol.protocol_id}:vanilla",
                "task_timeout_seconds": 14400,
            },
        },
        "cache_contract": {
            "artifact_dtype": "fp8_e5m2",
            "cold_page_cache_env": {"DOCUMENT_KV_EVICT_PAGE_CACHE": "1"},
            "key_position_encoding": "pre_rope",
            "nominal_vanilla_artifact_bytes": vanilla_bytes,
            "nominal_vanilla_artifact_gib": vanilla_bytes / 1024**3,
            "payload_cache_max_bytes": 0,
            "storage": "local_nvme",
        },
        "common_vllm_smoke_cli_args": common_cli,
        "dataset_order": list(SUPPORTED_V1_DATASETS),
        "decode": {
            "force_max_tokens": False,
            "max_tokens": protocol.max_tokens,
            "stop": "natural_eos",
            "stream": True,
            "temperature": 0.0,
            "top_p": "omitted; pinned vLLM 0.23.0 default",
        },
        "evidence": {
            "claim_scope": "descriptive_non_publication_canary",
            "inferential_claims_permitted": False,
            "paired_by": "dataset, example_id, prepared_prompt_sha256",
            "reporting": "per-arm dataset mean of the declared primary metric",
            "warning": (
                "n=5 per dataset is a smoke-sized diagnostic and must not be "
                "reported as a full-dataset score or publication result"
            ),
        },
        "examples_per_dataset": protocol.examples_per_dataset,
        "hardware": {
            "engine": "vllm",
            "engine_version": "0.23.0",
            "hardware_target": "aws-g6-l4",
            "node_type": "g6.8xlarge",
        },
        "input_tokens": protocol.input_tokens,
        "job_plan": {
            "isolated_job_count": 2,
            "max_reserved_cluster_hours": 6.0,
            "planning_cluster_hours": {"lower": 3.0, "upper": 4.0},
            "pre_rope_generation_calibration": {
                "observed_examples": 2,
                "observed_seconds": 735,
                "projected_seconds_for_vanilla_examples": 735 * total_examples / 2,
            },
        },
        "model": {
            "id": REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID,
            "revision": REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION,
            "served_weight_dtype": "bitsandbytes_4bit",
        },
        "primary_metrics": {
            dataset: {
                "metric": protocol.primary_metrics[dataset],
                "publication_approved_scorer": dataset == "hotpotqa",
                "role": "official" if dataset == "hotpotqa" else "diagnostic",
                "scorer_identity": _SCORER_IDENTITIES[dataset],
            }
            for dataset in SUPPORTED_V1_DATASETS
        },
        "prompt": {
            "add_special_tokens": SCORE_CANARY_ADD_SPECIAL_TOKENS,
            "prompt_template_version": DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
            "system_prompt_position": DEFAULT_SYSTEM_PROMPT_POSITION,
            "token_ids_digest": "sha256(canonical_json_integer_array)",
            "vanilla_segment_gate": (
                "independent_segment_tokens_equal_cache_prefix_tokens_and_cache_"
                "prefix_tokens_are_a_strict_full_prompt_prefix"
            ),
        },
        "protocol_id": protocol.protocol_id,
        "repeats": protocol.repeats,
        "request_parallelism": protocol.request_parallelism,
        "selection": {
            "algorithm": "ascending_sha256(seed\\0dataset\\0canonical_source_record_sha256)",
            "eligibility": (
                "expected_answer_present, unpadded_prompt_at_most_target, exact_"
                "padding_solution, and_vanilla_segment_token_composition_gate"
            ),
            "seed": protocol.selection_seed,
            "source_order_independent": True,
        },
        "tokenizer": {
            "add_special_tokens": SCORE_CANARY_ADD_SPECIAL_TOKENS,
            "id": REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID,
            "revision": REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION,
        },
        "total_examples_per_arm": total_examples,
    }


def _protocol_from_record(record: Mapping[str, Any]) -> ScoreCanaryProtocol:
    metrics_record = _required_mapping(record, "primary_metrics")
    metrics: dict[str, str] = {}
    for dataset in SUPPORTED_V1_DATASETS:
        dataset_metric = metrics_record.get(dataset)
        if not isinstance(dataset_metric, Mapping):
            raise ValueError(f"protocol.primary_metrics.{dataset} must be an object")
        metric = dataset_metric.get("metric")
        if not isinstance(metric, str) or not metric:
            raise ValueError(f"protocol.primary_metrics.{dataset}.metric is invalid")
        metrics[dataset] = metric
    selection = _required_mapping(record, "selection")
    return ScoreCanaryProtocol(
        protocol_id=_required_string(record, "protocol_id"),
        input_tokens=_required_positive_int(record, "input_tokens"),
        examples_per_dataset=_required_positive_int(record, "examples_per_dataset"),
        max_tokens=_required_positive_int(_required_mapping(record, "decode"), "max_tokens"),
        request_parallelism=_required_positive_int(record, "request_parallelism"),
        repeats=_required_positive_int(record, "repeats"),
        selection_seed=_required_string(selection, "seed"),
        primary_metrics=metrics,
    )


def _dataset_manifest_record(
    *,
    source_path: Path,
    source_record_count: int,
    output_name: str,
    output_content: bytes,
    prepared_rows: Sequence[_PreparedRow],
    protocol: ScoreCanaryProtocol,
) -> Mapping[str, Any]:
    examples: list[dict[str, Any]] = []
    for row in prepared_rows:
        prompt = build_prompt_parts(row.example).prefill_prompt
        examples.append(
            {
                "document_count": len(row.example.documents),
                "example_key_sha256": _canonical_sha256(
                    {"dataset": row.example.dataset, "example_id": row.example.example_id}
                ),
                "padding": {
                    "appended_document_count": int(row.padding_repetitions > 0),
                    "unit_repetitions": row.padding_repetitions,
                    "unit_sha256": (
                        None
                        if row.padding_unit is None
                        else sha256(row.padding_unit.encode("utf-8")).hexdigest()
                    ),
                },
                "prepared_prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                "prepared_record_sha256": _canonical_sha256(row.record),
                "prepared_token_count": protocol.input_tokens,
                "selection_rank_sha256": row.selection_rank_sha256,
                "source_record_sha256": row.source_record_sha256,
                "token_composition": row.token_composition,
                "unpadded_prompt_sha256": row.unpadded_prompt_sha256,
                "unpadded_token_count": row.unpadded_token_count,
            }
        )
    source_sha256, source_byte_count = _file_sha256_and_size(source_path)
    return {
        "examples": examples,
        "examples_sha256": _canonical_sha256(examples),
        "output": {
            "byte_count": len(output_content),
            "example_count": len(prepared_rows),
            "jsonl_sha256": sha256(output_content).hexdigest(),
            "path": output_name,
        },
        "primary_metric": protocol.primary_metrics[prepared_rows[0].example.dataset],
        "source": {
            "byte_count": source_byte_count,
            "jsonl_sha256": source_sha256,
            "record_count": source_record_count,
        },
    }


def _validate_dataset_output(
    manifest_dir: Path,
    *,
    dataset: str,
    record: Mapping[str, Any],
    protocol: ScoreCanaryProtocol,
    tokenizer: ScoreCanaryTokenizer,
) -> int:
    output = _required_mapping(record, "output")
    output_name = _required_string(output, "path")
    relative = Path(output_name)
    if relative.is_absolute() or relative.name != output_name:
        raise ValueError(f"datasets.{dataset}.output.path must be a filename")
    content = (manifest_dir / relative).read_bytes()
    expected_digest = _required_string(output, "jsonl_sha256")
    if sha256(content).hexdigest() != expected_digest:
        raise ValueError(f"datasets.{dataset} output SHA-256 mismatch")
    if len(content) != _required_positive_int(output, "byte_count"):
        raise ValueError(f"datasets.{dataset} output byte count mismatch")
    if output_name != f"{dataset}-{expected_digest}.jsonl":
        raise ValueError(f"datasets.{dataset} output filename is not content-addressed")
    raw_records = _jsonl_records(content, label=f"datasets.{dataset} output")
    expected_count = _required_positive_int(output, "example_count")
    if expected_count != protocol.examples_per_dataset or len(raw_records) != expected_count:
        raise ValueError(f"datasets.{dataset} output example count mismatch")
    examples = load_benchmark_jsonl(
        manifest_dir / relative,
        dataset=dataset,
        require_dataset=True,
    )
    if len({example.example_id for example in examples}) != expected_count:
        raise ValueError(f"datasets.{dataset} contains duplicate example identities")
    example_records = record.get("examples")
    if not isinstance(example_records, Sequence) or isinstance(
        example_records, (str, bytes, bytearray)
    ):
        raise ValueError(f"datasets.{dataset}.examples must be an array")
    if len(example_records) != expected_count:
        raise ValueError(f"datasets.{dataset}.examples count mismatch")
    if record.get("examples_sha256") != _canonical_sha256(example_records):
        raise ValueError(f"datasets.{dataset}.examples_sha256 mismatch")
    if record.get("primary_metric") != protocol.primary_metrics[dataset]:
        raise ValueError(f"datasets.{dataset}.primary_metric mismatch")
    for index, (raw_record, example, raw_attestation) in enumerate(
        zip(raw_records, examples, example_records, strict=True)
    ):
        if _TRANSFER_FIELDS.intersection(raw_record):
            raise ValueError(f"datasets.{dataset} output row {index} has KV metadata")
        if not isinstance(raw_attestation, Mapping):
            raise ValueError(f"datasets.{dataset}.examples[{index}] must be an object")
        if not example.references:
            raise ValueError(f"datasets.{dataset} output row {index} has no answer")
        if raw_attestation.get("prepared_record_sha256") != _canonical_sha256(raw_record):
            raise ValueError(f"datasets.{dataset} row {index} record hash mismatch")
        prompt = build_prompt_parts(example).prefill_prompt
        if _token_count(tokenizer, prompt) != protocol.input_tokens:
            raise ValueError(f"datasets.{dataset} row {index} token count mismatch")
        if raw_attestation.get("prepared_token_count") != protocol.input_tokens:
            raise ValueError(f"datasets.{dataset} row {index} token attestation mismatch")
        if raw_attestation.get("prepared_prompt_sha256") != sha256(
            prompt.encode("utf-8")
        ).hexdigest():
            raise ValueError(f"datasets.{dataset} row {index} prompt hash mismatch")
        if raw_attestation.get("document_count") != len(example.documents):
            raise ValueError(f"datasets.{dataset} row {index} document count mismatch")
        if raw_attestation.get("example_key_sha256") != _canonical_sha256(
            {"dataset": example.dataset, "example_id": example.example_id}
        ):
            raise ValueError(f"datasets.{dataset} row {index} identity hash mismatch")
        expected_composition = _token_composition_attestation(
            example,
            tokenizer=tokenizer,
        )
        if raw_attestation.get("token_composition") != expected_composition:
            raise ValueError(
                f"datasets.{dataset} row {index} token composition mismatch"
            )
        _validate_padding_attestation(
            dataset=dataset,
            index=index,
            raw_record=raw_record,
            raw_attestation=raw_attestation,
        )
    return expected_count


def _validate_padding_attestation(
    *,
    dataset: str,
    index: int,
    raw_record: Mapping[str, Any],
    raw_attestation: Mapping[str, Any],
) -> None:
    padding = raw_attestation.get("padding")
    if not isinstance(padding, Mapping):
        raise ValueError(f"datasets.{dataset} row {index} padding attestation is invalid")
    repetitions = padding.get("unit_repetitions")
    if type(repetitions) is not int or repetitions < 0:
        raise ValueError(f"datasets.{dataset} row {index} padding repetitions are invalid")
    appended_count = padding.get("appended_document_count")
    if appended_count != int(repetitions > 0):
        raise ValueError(f"datasets.{dataset} row {index} appended padding count is invalid")
    if repetitions == 0:
        if padding.get("unit_sha256") is not None:
            raise ValueError(f"datasets.{dataset} row {index} has a spurious padding unit")
        return
    unit_hash = padding.get("unit_sha256")
    units_by_hash = {
        sha256(unit.encode("utf-8")).hexdigest(): unit for unit in _PADDING_UNITS
    }
    if unit_hash not in units_by_hash:
        raise ValueError(f"datasets.{dataset} row {index} padding unit is invalid")
    raw_documents = raw_record.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError(f"datasets.{dataset} row {index} documents are invalid")
    expected_padding = _padding_document_record(
        units_by_hash[cast(str, unit_hash)] * repetitions
    )
    if raw_documents[-1] != expected_padding:
        raise ValueError(f"datasets.{dataset} row {index} padding document mismatch")


def _validate_source_identity(
    path: Path,
    *,
    dataset: str,
    source_record: Mapping[str, Any],
) -> None:
    digest, byte_count = _file_sha256_and_size(path)
    if source_record.get("jsonl_sha256") != digest:
        raise ValueError(f"datasets.{dataset} source SHA-256 mismatch")
    if source_record.get("byte_count") != byte_count:
        raise ValueError(f"datasets.{dataset} source byte count mismatch")
    count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if source_record.get("record_count") != count:
        raise ValueError(f"datasets.{dataset} source record count mismatch")


def _suite_sha256(
    protocol_sha256: str,
    datasets: Mapping[str, Mapping[str, Any]],
) -> str:
    outputs: dict[str, Any] = {}
    for dataset in SUPPORTED_V1_DATASETS:
        dataset_record = datasets.get(dataset)
        if not isinstance(dataset_record, Mapping):
            raise ValueError(f"datasets.{dataset} must be an object")
        output = dataset_record.get("output")
        if not isinstance(output, Mapping):
            raise ValueError(f"datasets.{dataset}.output must be an object")
        outputs[dataset] = {
            "example_count": output.get("example_count"),
            "jsonl_sha256": output.get("jsonl_sha256"),
        }
    return _canonical_sha256(
        {"dataset_outputs": outputs, "protocol_sha256": protocol_sha256}
    )


def _token_count(tokenizer: ScoreCanaryTokenizer, prompt: str) -> int:
    return len(_token_ids(tokenizer, prompt))


def _token_ids(
    tokenizer: ScoreCanaryTokenizer,
    prompt: str,
) -> tuple[int, ...]:
    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=SCORE_CANARY_ADD_SPECIAL_TOKENS,
    )
    if isinstance(token_ids, (str, bytes, bytearray)) or not isinstance(
        token_ids, Sequence
    ):
        raise TypeError("tokenizer.encode() must return a sequence of token ids")
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise ValueError("tokenizer.encode() returned an invalid token id")
    return tuple(token_ids)


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return _canonical_sha256(list(token_ids))


def _parse_source_specs(values: Sequence[str]) -> Mapping[str, str]:
    sources: dict[str, str] = {}
    for value in values:
        dataset, separator, path = value.partition("=")
        if not separator or not dataset or not path:
            raise ValueError("--source must use DATASET=JSONL_PATH")
        if dataset in sources:
            raise ValueError(f"duplicate --source dataset {dataset!r}")
        sources[dataset] = path
    if set(sources) != set(SUPPORTED_V1_DATASETS):
        raise ValueError("--source must be supplied exactly once for all four datasets")
    return MappingProxyType(sources)


def _jsonl_records(content: bytes, *, label: str) -> tuple[Mapping[str, Any], ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc
    records: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {line_number} is invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} line {line_number} must be an object")
        records.append(_json_materialize_mapping(value))
    return tuple(records)


def _json_object(content: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be a UTF-8 JSON object") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return _json_materialize_mapping(value)


def _json_materialize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(json.dumps(dict(value), ensure_ascii=False)),
    )


def _required_mapping(record: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = record.get(field_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _required_string(record: Mapping[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _required_positive_int(record: Mapping[str, Any], field_name: str) -> int:
    value = record.get(field_name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _file_sha256_and_size(path: Path) -> tuple[str, int]:
    digest = sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        _canonical_json_bytes(record, pretty=False) + b"\n" for record in records
    )


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return (json.dumps(value, **kwargs) + ("\n" if pretty else "")).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value, pretty=False)).hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


__all__ = [
    "SCORE_CANARY_RECORD_TYPE",
    "SCORE_CANARY_SCHEMA_VERSION",
    "SCORE_CANARY_PROTOCOL_ID",
    "SCORE_CANARY_INPUT_TOKENS",
    "SCORE_CANARY_EXAMPLES_PER_DATASET",
    "SCORE_CANARY_MAX_TOKENS",
    "SCORE_CANARY_REQUEST_PARALLELISM",
    "SCORE_CANARY_REPEATS",
    "SCORE_CANARY_ADD_SPECIAL_TOKENS",
    "SCORE_CANARY_SELECTION_SEED",
    "SCORE_CANARY_KV_BYTES_PER_TOKEN",
    "ScoreCanaryTokenizer",
    "ScoreCanaryProtocol",
    "DEFAULT_SCORE_CANARY_PROTOCOL",
    "PreparedScoreCanary",
    "ScoreCanaryValidation",
    "prepare_score_canary",
    "validate_score_canary_manifest",
    "score_canary_main",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI API.
    raise SystemExit(score_canary_main())
