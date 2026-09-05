"""Closed input schedules and shard plans for the publication campaign."""

from __future__ import annotations

import heapq
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from document_kv_cache._benchmark_datasets import _example_from_record
from document_kv_cache.benchmarks import (
    SUPPORTED_V1_DATASETS,
    benchmark_cache_prefix_segments,
    build_prompt_parts,
)
from document_kv_cache.main_latency_inputs import (
    MAIN_LATENCY_ADD_SPECIAL_TOKENS,
    MAIN_LATENCY_EXAMPLES_PER_DATASET,
    MAIN_LATENCY_TOKENIZER_ID,
    MAIN_LATENCY_TOKENIZER_REVISION,
    MainLatencyTokenizer,
    PreparedMainLatencyInputs,
    verify_main_latency_inputs,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS,
    PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM,
    PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE,
    PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL,
    PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET,
    PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE,
    PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL,
)


PUBLICATION_LATENCY_SCHEDULE_RECORD_TYPE = (
    "cachet.vllm_0271_publication_latency_schedule.v1"
)
PUBLICATION_LATENCY_SCHEDULE_SCHEMA_VERSION = 1
PUBLICATION_STORAGE_INPUTS_RECORD_TYPE = "cachet.publication_storage_inputs.v1"
PUBLICATION_STORAGE_INPUTS_SCHEMA_VERSION = 1
FULL_SCORE_INVENTORY_RECORD_TYPE = "cachet.full_score_inventory.v1"
FULL_SCORE_INVENTORY_SCHEMA_VERSION = 1
FULL_SCORE_SHARD_PLAN_RECORD_TYPE = "cachet.full_score_shard_plan.v1"
FULL_SCORE_SHARD_PLAN_SCHEMA_VERSION = 1
FULL_SCORE_MAX_WORKERS = 16
FULL_SCORE_MAX_NATURAL_PROMPT_TOKENS = 32_768

_LATENCY_SEED_DOMAIN = "cachet.publication.latency_schedule.seed.v1"
_STORAGE_LATENCY_SEED_DOMAIN = "cachet.publication.storage_latency_schedule.seed.v1"
_STORAGE_SELECTION_DOMAIN = "cachet.publication.storage_subset.selection.v1"
_LATENCY_ORDER_DOMAIN = "cachet.publication.latency_schedule.order.v1"
_FULL_SCORE_ORDER_DOMAIN = "cachet.publication.full_score.order.v1"
_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True, slots=True)
class PublicationLatencyExample:
    """One of the 128 identities shared by every publication latency cell."""

    dataset: str
    example_id: str

    def __post_init__(self) -> None:
        if self.dataset not in SUPPORTED_V1_DATASETS:
            raise ValueError(f"unsupported publication dataset: {self.dataset}")
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("example_id must be a non-empty string")

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {"dataset": self.dataset, "example_id": self.example_id}
        )


def select_publication_storage_examples(
    examples: Sequence[PublicationLatencyExample],
    *,
    input_bundle_sha256: str,
) -> tuple[PublicationLatencyExample, ...]:
    """Select the frozen two-per-dataset storage subset from all 128 inputs."""

    _require_sha256(input_bundle_sha256, field_name="input_bundle_sha256")
    full_examples = _validated_latency_examples(
        examples,
        examples_per_dataset=PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    )
    selected: list[PublicationLatencyExample] = []
    for dataset in SUPPORTED_V1_DATASETS:
        ranked = sorted(
            (example for example in full_examples if example.dataset == dataset),
            key=lambda example: (
                _storage_selection_rank_sha256(input_bundle_sha256, example),
                example.identity_sha256,
            ),
        )
        selected.extend(ranked[:PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET])
    return tuple(selected)


def load_publication_storage_selection_examples(
    source_paths: Mapping[str, str | Path],
) -> tuple[PublicationLatencyExample, ...]:
    """Load the exact 32/dataset source domain used by storage selection."""

    if set(source_paths) != set(SUPPORTED_V1_DATASETS):
        raise ValueError("storage input sources must contain all four datasets")
    examples: list[PublicationLatencyExample] = []
    for dataset in SUPPORTED_V1_DATASETS:
        source = Path(source_paths[dataset])
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"storage source is not a regular file: {source}")
        source_bytes = source.read_bytes()
        rows = source_bytes.splitlines(keepends=True)
        if len(rows) != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET:
            raise ValueError("storage source must contain exactly 32 rows per dataset")
        for line_number, row_bytes in enumerate(rows, 1):
            if not row_bytes.endswith(b"\n"):
                raise ValueError(f"storage source line {line_number} lacks a newline")
            raw = row_bytes[:-1]
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("storage source contains invalid JSON") from exc
            if (
                not isinstance(value, Mapping)
                or _canonical_json_bytes(value, pretty=False) != raw
            ):
                raise ValueError("storage source rows must be canonical JSON")
            if value.get("dataset") not in (None, dataset):
                raise ValueError("storage source row dataset does not match its file")
            example_id = value.get("example_id")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError("storage source row has no example_id")
            examples.append(PublicationLatencyExample(dataset, example_id))
    return _validated_latency_examples(
        examples,
        examples_per_dataset=PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    )


@dataclass(frozen=True, slots=True)
class FullScoreInventoryItem:
    """One complete natural-length score example and its token estimate."""

    dataset: str
    example_id: str
    natural_prompt_tokens: int
    cache_prefix_tokens: int
    natural_prompt_sha256: str
    natural_prompt_token_ids_sha256: str
    cache_prefix_sha256: str
    cache_prefix_token_ids_sha256: str
    segment_token_ids_sha256: str
    segment_count: int
    source_record_sha256: str

    def __post_init__(self) -> None:
        if self.dataset not in SUPPORTED_V1_DATASETS:
            raise ValueError(f"unsupported full-score dataset: {self.dataset}")
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("example_id must be a non-empty string")
        if type(self.natural_prompt_tokens) is not int or (
            self.natural_prompt_tokens <= 0
        ):
            raise ValueError("natural_prompt_tokens must be a positive integer")
        if type(self.cache_prefix_tokens) is not int or self.cache_prefix_tokens <= 0:
            raise ValueError("cache_prefix_tokens must be a positive integer")
        if self.cache_prefix_tokens > self.natural_prompt_tokens:
            raise ValueError("cache_prefix_tokens cannot exceed natural_prompt_tokens")
        if type(self.segment_count) is not int or self.segment_count <= 0:
            raise ValueError("segment_count must be a positive integer")
        for field_name in (
            "natural_prompt_sha256",
            "natural_prompt_token_ids_sha256",
            "cache_prefix_sha256",
            "cache_prefix_token_ids_sha256",
            "segment_token_ids_sha256",
            "source_record_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name=field_name)

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {"dataset": self.dataset, "example_id": self.example_id}
        )


@dataclass(frozen=True, slots=True)
class FullScoreDatasetSource:
    """Closed coverage of one complete canonical dataset JSONL."""

    dataset: str
    source_jsonl_sha256: str
    byte_count: int
    record_count: int
    identities_sha256: str
    source_records_sha256: str

    def __post_init__(self) -> None:
        if self.dataset not in SUPPORTED_V1_DATASETS:
            raise ValueError(f"unsupported full-score dataset: {self.dataset}")
        for field_name in (
            "source_jsonl_sha256",
            "identities_sha256",
            "source_records_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name=field_name)
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ValueError("byte_count must be a positive integer")
        if type(self.record_count) is not int or self.record_count <= 0:
            raise ValueError("record_count must be a positive integer")


@dataclass(frozen=True, slots=True)
class FullScoreInventory:
    """Every ID from every governed score dataset, without sampling."""

    sources: tuple[FullScoreDatasetSource, ...]
    items: tuple[FullScoreInventoryItem, ...]
    max_natural_prompt_tokens: int = FULL_SCORE_MAX_NATURAL_PROMPT_TOKENS

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        items = tuple(self.items)
        if tuple(source.dataset for source in sources) != tuple(SUPPORTED_V1_DATASETS):
            raise ValueError("full-score sources must cover all datasets in order")
        identities = [(item.dataset, item.example_id) for item in items]
        if len(set(identities)) != len(identities):
            raise ValueError("full-score inventory contains duplicate identities")
        counts = Counter(item.dataset for item in items)
        if (
            type(self.max_natural_prompt_tokens) is not int
            or self.max_natural_prompt_tokens <= 0
        ):
            raise ValueError("max_natural_prompt_tokens must be a positive integer")
        if any(
            item.natural_prompt_tokens > self.max_natural_prompt_tokens
            for item in items
        ):
            raise ValueError("full-score natural prompt exceeds max context")
        for source in sources:
            if counts[source.dataset] != source.record_count:
                raise ValueError(
                    f"full-score inventory coverage mismatch for {source.dataset}"
                )
            dataset_items = tuple(
                item for item in items if item.dataset == source.dataset
            )
            if _identity_closure_sha256(dataset_items) != source.identities_sha256:
                raise ValueError(
                    f"full-score identity closure mismatch for {source.dataset}"
                )
            if _source_record_closure_sha256(dataset_items) != (
                source.source_records_sha256
            ):
                raise ValueError(
                    f"full-score source-record closure mismatch for {source.dataset}"
                )
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "items", items)

    @property
    def inventory_sha256(self) -> str:
        return cast(
            str,
            full_score_inventory_to_record(self)["closed_record_sha256"],
        )


def build_publication_latency_block_schedule(
    *,
    campaign_id: str,
    deployment_block: int,
    input_bundle_sha256: str,
    examples: Sequence[PublicationLatencyExample],
) -> dict[str, Any]:
    """Build one method-independent, context-independent 256-request schedule."""

    return _build_publication_latency_block_schedule(
        campaign_id=campaign_id,
        deployment_block=deployment_block,
        input_bundle_sha256=input_bundle_sha256,
        examples=examples,
        examples_per_dataset=PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
        repeats_per_example=PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE,
        request_count=PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL,
        seed_domain=_LATENCY_SEED_DOMAIN,
        workload_id="main",
    )


def build_publication_storage_block_schedule(
    *,
    campaign_id: str,
    deployment_block: int,
    input_bundle_sha256: str,
    examples: Sequence[PublicationLatencyExample],
) -> dict[str, Any]:
    """Select and build the 8-identity x 32-repeat schedule for one block."""

    selected = select_publication_storage_examples(
        examples,
        input_bundle_sha256=input_bundle_sha256,
    )
    return _build_publication_storage_schedule_from_selected(
        campaign_id=campaign_id,
        deployment_block=deployment_block,
        input_bundle_sha256=input_bundle_sha256,
        selected_examples=selected,
    )


def _build_publication_storage_schedule_from_selected(
    *,
    campaign_id: str,
    deployment_block: int,
    input_bundle_sha256: str,
    selected_examples: Sequence[PublicationLatencyExample],
) -> dict[str, Any]:
    record = _build_publication_latency_block_schedule(
        campaign_id=campaign_id,
        deployment_block=deployment_block,
        input_bundle_sha256=input_bundle_sha256,
        examples=selected_examples,
        examples_per_dataset=PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET,
        repeats_per_example=PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE,
        request_count=PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL,
        seed_domain=_STORAGE_LATENCY_SEED_DOMAIN,
        workload_id="storage",
    )
    protocol = cast(dict[str, Any], record["protocol"])
    protocol["selection"] = _publication_storage_selection_record(
        selected_examples,
        input_bundle_sha256=input_bundle_sha256,
    )
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _build_publication_latency_block_schedule(
    *,
    campaign_id: str,
    deployment_block: int,
    input_bundle_sha256: str,
    examples: Sequence[PublicationLatencyExample],
    examples_per_dataset: int,
    repeats_per_example: int,
    request_count: int,
    seed_domain: str,
    workload_id: str,
) -> dict[str, Any]:
    """Build one exact balanced schedule for a frozen workload protocol."""

    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("campaign_id must be a non-empty string")
    if deployment_block not in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1):
        raise ValueError("deployment_block is outside the publication campaign")
    _require_sha256(input_bundle_sha256, field_name="input_bundle_sha256")
    ordered_examples = _validated_latency_examples(
        examples,
        examples_per_dataset=examples_per_dataset,
    )
    if request_count != (
        len(SUPPORTED_V1_DATASETS) * examples_per_dataset * repeats_per_example
    ):
        raise ValueError("latency schedule request count/protocol mismatch")
    seed_sha256 = _canonical_sha256(
        {
            "campaign_id": campaign_id,
            "deployment_block": deployment_block,
            "domain": seed_domain,
            "input_bundle_sha256": input_bundle_sha256,
        }
    )
    by_dataset = {
        dataset: tuple(
            example for example in ordered_examples if example.dataset == dataset
        )
        for dataset in SUPPORTED_V1_DATASETS
    }
    occurrences: dict[str, tuple[tuple[PublicationLatencyExample, int], ...]] = {}
    for dataset, dataset_examples in by_dataset.items():
        example_order = tuple(
            sorted(
                dataset_examples,
                key=lambda example: _latency_order_sha256(
                    seed_sha256,
                    "occurrence",
                    example.identity_sha256,
                ),
            )
        )
        occurrences[dataset] = tuple(
            (example, repeat_index)
            for repeat_index in range(1, repeats_per_example + 1)
            for example in example_order
        )

    wave_count = examples_per_dataset * repeats_per_example
    requests: list[dict[str, Any]] = []
    for wave_index in range(wave_count):
        dataset_order = sorted(
            SUPPORTED_V1_DATASETS,
            key=lambda dataset: _latency_order_sha256(
                seed_sha256,
                "dataset-wave",
                str(wave_index),
                dataset,
            ),
        )
        for dataset in dataset_order:
            example, repeat_index = occurrences[dataset][wave_index]
            request_index = len(requests)
            requests.append(
                {
                    "dataset": dataset,
                    "example_id": example.example_id,
                    "example_identity_sha256": example.identity_sha256,
                    "repeat_index": repeat_index,
                    "request_id": (
                        f"block-{deployment_block:02d}-request-{request_index:03d}"
                    ),
                    "request_index": request_index,
                    "wave_index": wave_index,
                }
            )

    lanes = {
        str(parallelism): _latency_lanes(
            ordered_examples,
            requests,
            seed_sha256=seed_sha256,
            parallelism=parallelism,
        )
        for parallelism in PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM
    }
    _validate_latency_admission_windows(requests)
    record: dict[str, Any] = {
        "campaign_id": campaign_id,
        "closed_record_sha256": "",
        "deployment_block": deployment_block,
        "input_bundle_sha256": input_bundle_sha256,
        "lanes": lanes,
        "protocol": {
            "closed_loop": True,
            "dataset_count_per_wave": len(SUPPORTED_V1_DATASETS),
            "datasets": list(SUPPORTED_V1_DATASETS),
            "duplicate_exclusion": "identity_sticky_single_flight_lane",
            "examples_per_dataset": examples_per_dataset,
            "logical_schedule_shared_across_methods": True,
            "logical_schedule_shared_across_parallelism": True,
            "repeats_per_example": repeats_per_example,
            "request_count": request_count,
            "request_parallelism": list(PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM),
            "think_time_seconds": 0,
            "workload_id": workload_id,
        },
        "record_type": PUBLICATION_LATENCY_SCHEDULE_RECORD_TYPE,
        "requests": requests,
        "requests_sha256": _canonical_sha256(requests),
        "schema_version": PUBLICATION_LATENCY_SCHEDULE_SCHEMA_VERSION,
        "seed_sha256": seed_sha256,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def load_verified_publication_latency_examples(
    output_dir: str | Path,
    *,
    tokenizer: MainLatencyTokenizer,
    source_paths: Mapping[str, str | Path] | None = None,
) -> tuple[PreparedMainLatencyInputs, tuple[PublicationLatencyExample, ...]]:
    """Load the 128 identities only after the entire 32-example bundle verifies."""

    verified = verify_main_latency_inputs(
        output_dir,
        source_paths=source_paths,
        tokenizer=tokenizer,
    )
    target_files = {
        artifact.dataset: artifact
        for artifact in verified.files
        if artifact.input_tokens_target == 8192
    }
    if set(target_files) != set(SUPPORTED_V1_DATASETS):
        raise ValueError("verified latency bundle has incomplete 8k identity coverage")
    examples: list[PublicationLatencyExample] = []
    for dataset in SUPPORTED_V1_DATASETS:
        raw_lines = (
            target_files[dataset].jsonl_path.read_text(encoding="utf-8").splitlines()
        )
        for raw_line in raw_lines:
            value = json.loads(raw_line)
            if not isinstance(value, Mapping):
                raise ValueError(
                    "verified latency JSONL unexpectedly contains non-object"
                )
            example_id = value.get("example_id")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError("verified latency JSONL has invalid example_id")
            examples.append(
                PublicationLatencyExample(dataset=dataset, example_id=example_id)
            )
    return verified, _validated_latency_examples(examples)


def build_verified_publication_latency_block_schedule(
    output_dir: str | Path,
    *,
    campaign_id: str,
    deployment_block: int,
    tokenizer: MainLatencyTokenizer,
    source_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build a schedule directly from a fail-closed verified latency bundle."""

    verified, examples = load_verified_publication_latency_examples(
        output_dir,
        tokenizer=tokenizer,
        source_paths=source_paths,
    )
    return build_publication_latency_block_schedule(
        campaign_id=campaign_id,
        deployment_block=deployment_block,
        input_bundle_sha256=verified.bundle_sha256,
        examples=examples,
    )


def validate_publication_latency_block_schedule(
    record: Mapping[str, Any],
    *,
    examples: Sequence[PublicationLatencyExample],
    expected_input_bundle_sha256: str,
) -> None:
    """Fail closed unless a schedule exactly matches its verified input bundle."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("latency schedule closed_record_sha256 is invalid")
    campaign_id = record.get("campaign_id")
    deployment_block = record.get("deployment_block")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("latency schedule campaign_id must be non-empty")
    if type(deployment_block) is not int:
        raise ValueError("latency schedule deployment_block must be an integer")
    protocol = record.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("latency schedule protocol must be an object")
    workload_id = protocol.get("workload_id")
    if workload_id == "main":
        expected = build_publication_latency_block_schedule(
            campaign_id=campaign_id,
            deployment_block=deployment_block,
            input_bundle_sha256=expected_input_bundle_sha256,
            examples=examples,
        )
    elif workload_id == "storage":
        counts = Counter(example.dataset for example in examples)
        if all(
            counts[dataset] == PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
            for dataset in SUPPORTED_V1_DATASETS
        ):
            expected = build_publication_storage_block_schedule(
                campaign_id=campaign_id,
                deployment_block=deployment_block,
                input_bundle_sha256=expected_input_bundle_sha256,
                examples=examples,
            )
        else:
            # Worker-side validation can authenticate the selected schedule's own
            # closure, but it is intentionally not an authority over which two
            # examples were chosen from the full 32/dataset source domain.
            expected = _build_publication_storage_schedule_from_selected(
                campaign_id=campaign_id,
                deployment_block=deployment_block,
                input_bundle_sha256=expected_input_bundle_sha256,
                selected_examples=examples,
            )
    else:
        raise ValueError("latency schedule workload_id is invalid")
    if dict(record) != expected:
        raise ValueError("latency schedule does not match the verified input bundle")


def validate_publication_storage_block_schedule(
    record: Mapping[str, Any],
    *,
    source_examples: Sequence[PublicationLatencyExample],
    expected_input_bundle_sha256: str,
) -> None:
    """Authorize a storage schedule only against the complete 32/dataset domain."""

    _validated_latency_examples(
        source_examples,
        examples_per_dataset=PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    )
    validate_publication_latency_block_schedule(
        record,
        examples=source_examples,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
    )


def project_publication_latency_request_order(
    record: Mapping[str, Any],
    *,
    examples: Sequence[PublicationLatencyExample],
    expected_input_bundle_sha256: str,
) -> tuple[tuple[str, str, int], ...]:
    """Return the exact request keys a benchmark runner must consume in order.

    The runner must dispatch this projection without omission, duplication, or
    reordering. Identity-sticky lanes in the same record additionally preserve
    single-flight identity exclusion under asynchronous closed-loop completion.
    """

    validate_publication_latency_block_schedule(
        record,
        examples=examples,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
    )
    requests = record.get("requests")
    if not isinstance(requests, list):
        raise ValueError("latency schedule requests must be an array")
    projection: list[tuple[str, str, int]] = []
    for request_index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            raise ValueError("latency schedule request must be an object")
        dataset = request.get("dataset")
        example_id = request.get("example_id")
        repeat_index = request.get("repeat_index")
        if (
            not isinstance(dataset, str)
            or not isinstance(example_id, str)
            or type(repeat_index) is not int
            or request.get("request_index") != request_index
        ):
            raise ValueError("latency schedule request key is invalid")
        projection.append((dataset, example_id, repeat_index))
    protocol = record.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("latency schedule protocol must be an object")
    request_count = protocol.get("request_count")
    if type(request_count) is not int or request_count <= 0:
        raise ValueError("latency schedule request_count is invalid")
    if len(projection) != request_count or len(set(projection)) != len(projection):
        raise ValueError("latency request-order projection is incomplete or duplicated")
    return tuple(projection)


def materialize_publication_storage_inputs(
    source_paths: Mapping[str, str | Path],
    schedule_records: Mapping[int, Mapping[str, Any]],
    output_dir: str | Path,
    *,
    expected_input_bundle_sha256: str,
) -> dict[str, Any]:
    """Write the exact 2/dataset storage subset as source-row-identical JSONL."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"storage input output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        record = _publication_storage_inputs_record(
            source_paths,
            schedule_records,
            destination,
            expected_input_bundle_sha256=expected_input_bundle_sha256,
            write=True,
        )
        _write_bytes_exclusive(
            destination / "publication-storage-inputs.json",
            _canonical_json_bytes(record, pretty=True),
        )
        directory = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if destination.exists() and not any(destination.iterdir()):
            destination.rmdir()
        raise
    validate_publication_storage_inputs_record(
        record,
        source_paths=source_paths,
        schedule_records=schedule_records,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
    )
    return record


def validate_publication_storage_inputs_record(
    record: Mapping[str, Any],
    *,
    source_paths: Mapping[str, str | Path],
    schedule_records: Mapping[int, Mapping[str, Any]],
    expected_input_bundle_sha256: str,
) -> None:
    """Re-read sources and outputs and authenticate exact byte/identity selection."""

    if record.get("record_type") != PUBLICATION_STORAGE_INPUTS_RECORD_TYPE:
        raise ValueError("storage input record_type is invalid")
    if record.get("schema_version") != PUBLICATION_STORAGE_INPUTS_SCHEMA_VERSION:
        raise ValueError("storage input schema_version is invalid")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("storage input record closure is invalid")
    output_root = record.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("storage input output_root is invalid")
    expected = _publication_storage_inputs_record(
        source_paths,
        schedule_records,
        Path(output_root),
        expected_input_bundle_sha256=expected_input_bundle_sha256,
        write=False,
    )
    if dict(record) != expected:
        raise ValueError("storage input files differ from exact source rows")


def _publication_storage_inputs_record(
    source_paths: Mapping[str, str | Path],
    schedule_records: Mapping[int, Mapping[str, Any]],
    output_dir: Path,
    *,
    expected_input_bundle_sha256: str,
    write: bool,
) -> dict[str, Any]:
    _require_sha256(
        expected_input_bundle_sha256,
        field_name="expected_input_bundle_sha256",
    )
    if set(source_paths) != set(SUPPORTED_V1_DATASETS):
        raise ValueError("storage input sources must contain all four datasets")
    if set(schedule_records) != set(
        range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1)
    ):
        raise ValueError("storage schedules must contain exactly five blocks")
    source_examples = load_publication_storage_selection_examples(source_paths)
    selected_examples = select_publication_storage_examples(
        source_examples,
        input_bundle_sha256=expected_input_bundle_sha256,
    )
    selection_record = _publication_storage_selection_record(
        selected_examples,
        input_bundle_sha256=expected_input_bundle_sha256,
    )
    identity_set = {
        (example.dataset, example.example_id) for example in selected_examples
    }
    schedule_bindings: list[dict[str, Any]] = []
    for block in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1):
        schedule = schedule_records[block]
        examples = _schedule_examples_from_requests(schedule)
        validate_publication_storage_block_schedule(
            schedule,
            source_examples=source_examples,
            expected_input_bundle_sha256=expected_input_bundle_sha256,
        )
        if (
            schedule.get("deployment_block") != block
            or _record_mapping(schedule, "protocol").get("workload_id") != "storage"
        ):
            raise ValueError("storage schedule block/protocol drift")
        observed = {(item.dataset, item.example_id) for item in examples}
        if observed != identity_set:
            raise ValueError(
                "storage schedule differs from deterministic eight-identity selection"
            )
        schedule_selection = _record_mapping(
            _record_mapping(schedule, "protocol"), "selection"
        )
        if dict(schedule_selection) != selection_record:
            raise ValueError("storage schedule selection closure drift")
        schedule_bindings.append(
            {
                "closed_record_sha256": _record_string(
                    schedule, "closed_record_sha256"
                ),
                "deployment_block": block,
                "requests_sha256": _record_string(schedule, "requests_sha256"),
                "selection_sha256": _record_string(
                    schedule_selection, "selection_sha256"
                ),
            }
        )
    files: list[dict[str, Any]] = []
    for dataset in SUPPORTED_V1_DATASETS:
        source = Path(source_paths[dataset])
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"storage source is not a regular file: {source}")
        source_bytes = source.read_bytes()
        rows_by_id: dict[str, bytes] = {}
        for line_number, row_bytes in enumerate(
            source_bytes.splitlines(keepends=True), 1
        ):
            if not row_bytes.endswith(b"\n"):
                raise ValueError(f"storage source line {line_number} lacks a newline")
            raw = row_bytes[:-1]
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("storage source contains invalid JSON") from exc
            if (
                not isinstance(value, Mapping)
                or _canonical_json_bytes(value, pretty=False) != raw
            ):
                raise ValueError("storage source rows must be canonical JSON")
            example_id = value.get("example_id")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError("storage source row has no example_id")
            if example_id in rows_by_id:
                raise ValueError("storage source contains duplicate example IDs")
            rows_by_id[example_id] = row_bytes
        selected_ids = sorted(
            example_id
            for selected_dataset, example_id in identity_set
            if selected_dataset == dataset
        )
        if len(selected_ids) != PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET:
            raise ValueError("storage schedule does not select two IDs per dataset")
        missing = sorted(set(selected_ids).difference(rows_by_id))
        if missing:
            raise ValueError(
                f"storage schedule IDs are missing from {dataset}: {missing}"
            )
        selected_rows = [rows_by_id[example_id] for example_id in selected_ids]
        output_bytes = b"".join(selected_rows)
        output_path = output_dir / f"storage-16384-{dataset}.jsonl"
        if write:
            _write_bytes_exclusive(output_path, output_bytes)
        elif (
            not output_path.is_file()
            or output_path.is_symlink()
            or output_path.read_bytes() != output_bytes
        ):
            raise ValueError(f"storage output {dataset} is not source-row identical")
        files.append(
            {
                "byte_count": len(output_bytes),
                "dataset": dataset,
                "identities": selected_ids,
                "record_count": len(selected_rows),
                "rows_sha256": _canonical_sha256(
                    [sha256(row).hexdigest() for row in selected_rows]
                ),
                "sha256": sha256(output_bytes).hexdigest(),
                "source_sha256": sha256(source_bytes).hexdigest(),
                "uri": str(output_path.absolute()),
            }
        )
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "files": files,
        "input_bundle_sha256": expected_input_bundle_sha256,
        "output_root": str(output_dir.absolute()),
        "record_type": PUBLICATION_STORAGE_INPUTS_RECORD_TYPE,
        "schedule_bindings": schedule_bindings,
        "schema_version": PUBLICATION_STORAGE_INPUTS_SCHEMA_VERSION,
        "selection_protocol": {
            **selection_record,
            "examples_per_dataset": PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET,
            "repeats_per_example": PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE,
            "request_count": PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL,
            "source_row_bytes_preserved": True,
        },
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _schedule_examples_from_requests(
    record: Mapping[str, Any],
) -> tuple[PublicationLatencyExample, ...]:
    requests = record.get("requests")
    if not isinstance(requests, list):
        raise ValueError("storage schedule requests must be an array")
    identities = {
        (request.get("dataset"), request.get("example_id"))
        for request in requests
        if isinstance(request, Mapping)
    }
    if any(
        not isinstance(dataset, str) or not isinstance(example_id, str)
        for dataset, example_id in identities
    ):
        raise ValueError("storage schedule contains an invalid identity")
    return tuple(
        PublicationLatencyExample(cast(str, dataset), cast(str, example_id))
        for dataset, example_id in sorted(identities)
    )


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def load_full_score_inventory(
    source_paths: Mapping[str, str | Path],
    *,
    tokenizer: MainLatencyTokenizer,
    max_natural_prompt_tokens: int = FULL_SCORE_MAX_NATURAL_PROMPT_TOKENS,
) -> FullScoreInventory:
    """Tokenize every natural prompt in all four complete score JSONLs."""

    if set(source_paths) != set(SUPPORTED_V1_DATASETS):
        raise ValueError("source_paths must contain exactly all score datasets")
    if type(max_natural_prompt_tokens) is not int or max_natural_prompt_tokens <= 0:
        raise ValueError("max_natural_prompt_tokens must be a positive integer")
    sources: list[FullScoreDatasetSource] = []
    items: list[FullScoreInventoryItem] = []
    for dataset in SUPPORTED_V1_DATASETS:
        path = Path(source_paths[dataset])
        raw = path.read_bytes()
        if not raw or not raw.endswith(b"\n"):
            raise ValueError(f"{dataset} source must be non-empty newline JSONL")
        raw_lines = raw[:-1].split(b"\n")
        if any(not line for line in raw_lines):
            raise ValueError(f"{dataset} source must not contain empty JSONL rows")
        dataset_items: list[FullScoreInventoryItem] = []
        seen_ids: set[str] = set()
        for record_index, raw_line in enumerate(raw_lines, start=1):
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid {dataset} JSONL row {record_index}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"{dataset} JSONL row {record_index} must be an object"
                )
            record = cast(Mapping[str, Any], value)
            try:
                example = _example_from_record(
                    record,
                    default_dataset=dataset,
                    record_index=record_index,
                    require_dataset=True,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid canonical {dataset} row {record_index}: {exc}"
                ) from exc
            if example.dataset != dataset:
                raise ValueError(f"{dataset} source contains a different dataset")
            if example.example_id in seen_ids:
                raise ValueError(f"{dataset} source contains duplicate example IDs")
            seen_ids.add(example.example_id)
            prompt_parts = build_prompt_parts(example)
            natural_prompt = prompt_parts.prefill_prompt
            cache_prefix = prompt_parts.cache_prefix_text
            natural_prompt_ids = _encoded_token_ids(tokenizer, natural_prompt)
            cache_prefix_ids = _encoded_token_ids(tokenizer, cache_prefix)
            segment_ids = tuple(
                _encoded_token_ids(tokenizer, segment_text)
                for _segment_id, segment_text in benchmark_cache_prefix_segments(
                    example
                )
            )
            composed_segment_ids = tuple(
                token_id for ids in segment_ids for token_id in ids
            )
            if composed_segment_ids != cache_prefix_ids:
                raise ValueError(
                    f"independent cache segments do not compose for {dataset}/"
                    f"{example.example_id}"
                )
            if natural_prompt_ids[: len(cache_prefix_ids)] != cache_prefix_ids:
                raise ValueError(
                    f"cache prefix is not the natural-prompt prefix for {dataset}/"
                    f"{example.example_id}"
                )
            if len(natural_prompt_ids) > max_natural_prompt_tokens:
                raise ValueError(
                    f"natural prompt exceeds {max_natural_prompt_tokens} tokens for "
                    f"{dataset}/{example.example_id}"
                )
            dataset_items.append(
                FullScoreInventoryItem(
                    dataset=dataset,
                    example_id=example.example_id,
                    natural_prompt_tokens=len(natural_prompt_ids),
                    cache_prefix_tokens=len(cache_prefix_ids),
                    natural_prompt_sha256=sha256(
                        natural_prompt.encode("utf-8")
                    ).hexdigest(),
                    natural_prompt_token_ids_sha256=_token_ids_sha256(
                        natural_prompt_ids
                    ),
                    cache_prefix_sha256=sha256(
                        cache_prefix.encode("utf-8")
                    ).hexdigest(),
                    cache_prefix_token_ids_sha256=_token_ids_sha256(cache_prefix_ids),
                    segment_token_ids_sha256=_segment_token_ids_sha256(segment_ids),
                    segment_count=len(segment_ids),
                    source_record_sha256=_canonical_sha256(record),
                )
            )
        ordered_dataset_items = tuple(
            sorted(dataset_items, key=_full_score_item_order_sha256)
        )
        sources.append(
            FullScoreDatasetSource(
                dataset=dataset,
                source_jsonl_sha256=sha256(raw).hexdigest(),
                byte_count=len(raw),
                record_count=len(ordered_dataset_items),
                identities_sha256=_identity_closure_sha256(ordered_dataset_items),
                source_records_sha256=_source_record_closure_sha256(
                    ordered_dataset_items
                ),
            )
        )
        items.extend(ordered_dataset_items)
    return FullScoreInventory(
        sources=tuple(sources),
        items=tuple(items),
        max_natural_prompt_tokens=max_natural_prompt_tokens,
    )


def full_score_inventory_to_record(inventory: FullScoreInventory) -> dict[str, Any]:
    """Return the canonical complete-ID closure used by shard plans."""

    if not isinstance(inventory, FullScoreInventory):
        raise TypeError("inventory must be a FullScoreInventory")
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "input_length_policy": {
            "max_natural_prompt_tokens": inventory.max_natural_prompt_tokens,
            "padding": False,
            "segment_token_id_digest_encoding": (
                "segment_count_and_lengths_then_uint64be_v1"
            ),
            "token_id_digest_encoding": "length_then_uint64be_v1",
            "tokenizer_truncation": False,
        },
        "items": [_full_score_item_record(item) for item in inventory.items],
        "record_type": FULL_SCORE_INVENTORY_RECORD_TYPE,
        "schema_version": FULL_SCORE_INVENTORY_SCHEMA_VERSION,
        "sources": [
            {
                "byte_count": source.byte_count,
                "dataset": source.dataset,
                "identities_sha256": source.identities_sha256,
                "record_count": source.record_count,
                "source_jsonl_sha256": source.source_jsonl_sha256,
                "source_records_sha256": source.source_records_sha256,
            }
            for source in inventory.sources
        ],
        "tokenizer": {
            "add_special_tokens": MAIN_LATENCY_ADD_SPECIAL_TOKENS,
            "tokenizer_id": MAIN_LATENCY_TOKENIZER_ID,
            "tokenizer_revision": MAIN_LATENCY_TOKENIZER_REVISION,
        },
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def validate_full_score_inventory_record(
    record: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
) -> None:
    """Fail closed unless a serialized inventory covers every loaded source ID."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("full-score inventory closed_record_sha256 is invalid")
    if dict(record) != full_score_inventory_to_record(inventory):
        raise ValueError("full-score inventory record does not match complete sources")


def build_full_score_shard_plan(
    inventory: FullScoreInventory,
    *,
    plan_id: str,
    max_workers: int,
    target_cache_prefix_tokens_per_shard: int,
) -> dict[str, Any]:
    """Partition every complete ID into token-capped streaming work shards."""

    if not isinstance(inventory, FullScoreInventory):
        raise TypeError("inventory must be a FullScoreInventory")
    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError("plan_id must be a non-empty string")
    if type(max_workers) is not int or not 1 <= max_workers <= FULL_SCORE_MAX_WORKERS:
        raise ValueError(f"max_workers must be between 1 and {FULL_SCORE_MAX_WORKERS}")
    if (
        type(target_cache_prefix_tokens_per_shard) is not int
        or target_cache_prefix_tokens_per_shard <= 0
    ):
        raise ValueError(
            "target_cache_prefix_tokens_per_shard must be a positive integer"
        )
    if not inventory.items:
        raise ValueError("full-score inventory must not be empty")

    inventory_sha256 = inventory.inventory_sha256
    ordered_items = tuple(
        sorted(
            inventory.items,
            key=lambda item: _canonical_sha256(
                {
                    "domain": _FULL_SCORE_ORDER_DOMAIN,
                    "inventory_sha256": inventory_sha256,
                    "item_identity_sha256": item.identity_sha256,
                    "source_record_sha256": item.source_record_sha256,
                }
            ),
        )
    )
    raw_shards: list[tuple[FullScoreInventoryItem, ...]] = []
    current: list[FullScoreInventoryItem] = []
    current_tokens = 0
    for item in ordered_items:
        if current and current_tokens + item.cache_prefix_tokens > (
            target_cache_prefix_tokens_per_shard
        ):
            raw_shards.append(tuple(current))
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += item.cache_prefix_tokens
        if current_tokens >= target_cache_prefix_tokens_per_shard:
            raw_shards.append(tuple(current))
            current = []
            current_tokens = 0
    if current:
        raw_shards.append(tuple(current))

    shard_specs = [
        {
            "items": shard_items,
            "items_sha256": _canonical_sha256(
                [_full_score_item_record(item) for item in shard_items]
            ),
            "cache_prefix_tokens": sum(
                item.cache_prefix_tokens for item in shard_items
            ),
            "natural_prompt_tokens": sum(
                item.natural_prompt_tokens for item in shard_items
            ),
            "shard_index": shard_index,
        }
        for shard_index, shard_items in enumerate(raw_shards)
    ]
    active_workers = min(max_workers, len(shard_specs))
    worker_heap: list[tuple[int, int]] = [
        (0, worker_index) for worker_index in range(active_workers)
    ]
    heapq.heapify(worker_heap)
    worker_by_shard: dict[int, int] = {}
    for shard in sorted(
        shard_specs,
        key=lambda value: (
            -cast(int, value["cache_prefix_tokens"]),
            cast(str, value["items_sha256"]),
        ),
    ):
        worker_tokens, worker_index = heapq.heappop(worker_heap)
        shard_index = cast(int, shard["shard_index"])
        worker_by_shard[shard_index] = worker_index
        heapq.heappush(
            worker_heap,
            (
                worker_tokens + cast(int, shard["cache_prefix_tokens"]),
                worker_index,
            ),
        )

    shards: list[dict[str, Any]] = []
    for shard in shard_specs:
        shard_index = cast(int, shard["shard_index"])
        items = cast(tuple[FullScoreInventoryItem, ...], shard["items"])
        shard_id = f"full-score-shard-{shard_index:05d}"
        shards.append(
            {
                "ephemeral_kv_artifact": (f"ephemeral://{plan_id}/{shard_id}/q8-kv"),
                "item_count": len(items),
                "items": [_full_score_item_record(item) for item in items],
                "items_sha256": shard["items_sha256"],
                "cache_prefix_tokens": shard["cache_prefix_tokens"],
                "natural_prompt_tokens": shard["natural_prompt_tokens"],
                "shard_id": shard_id,
                "shard_index": shard_index,
                "worker_index": worker_by_shard[shard_index],
            }
        )

    worker_loads = [
        {
            "cache_prefix_tokens": sum(
                cast(int, shard["cache_prefix_tokens"])
                for shard in shards
                if shard["worker_index"] == worker_index
            ),
            "natural_prompt_tokens": sum(
                cast(int, shard["natural_prompt_tokens"])
                for shard in shards
                if shard["worker_index"] == worker_index
            ),
            "shard_count": sum(
                1 for shard in shards if shard["worker_index"] == worker_index
            ),
            "worker_index": worker_index,
        }
        for worker_index in range(active_workers)
    ]
    coverage_items = sorted(
        (_full_score_item_record(item) for item in inventory.items),
        key=lambda item: (cast(str, item["dataset"]), cast(str, item["example_id"])),
    )
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "config": {
            "active_workers": active_workers,
            "max_inflight_shards_per_worker": 1,
            "max_workers": max_workers,
            "target_cache_prefix_tokens_per_shard": (
                target_cache_prefix_tokens_per_shard
            ),
        },
        "coverage": {
            "complete_inventory_required": True,
            "identity_count": len(coverage_items),
            "identities_sha256": _canonical_sha256(
                [
                    {
                        "dataset": item["dataset"],
                        "example_id": item["example_id"],
                    }
                    for item in coverage_items
                ]
            ),
            "cache_prefix_generation_tokens": sum(
                item.cache_prefix_tokens for item in inventory.items
            ),
            "natural_prompt_inference_tokens": sum(
                item.natural_prompt_tokens for item in inventory.items
            ),
            "no_duplicate_identities": True,
        },
        "inventory_sha256": inventory_sha256,
        "lifecycle": {
            "delete_condition": (
                "paired_outputs_validated_and_preserved_evidence_committed"
            ),
            "ephemeral_scope": "all_non_sample_q8_kv_for_one_shard",
            "preserve": [
                "inventory_and_shard_manifests",
                "baseline_and_vanilla_raw_outputs",
                "scores",
                "runtime_telemetry",
                "predeclared_sample_kv_artifacts",
            ],
            "steps": [
                "generate_q8_kv",
                "baseline_inference",
                "vanilla_inference",
                "validate_paired_outputs",
                "delete_ephemeral_q8_kv",
            ],
            "streaming": True,
        },
        "plan_id": plan_id,
        "record_type": FULL_SCORE_SHARD_PLAN_RECORD_TYPE,
        "schema_version": FULL_SCORE_SHARD_PLAN_SCHEMA_VERSION,
        "shards": shards,
        "shards_sha256": _canonical_sha256(shards),
        "worker_loads": worker_loads,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def validate_full_score_shard_plan(
    record: Mapping[str, Any],
    *,
    inventory: FullScoreInventory,
) -> None:
    """Fail closed on sampled, duplicated, rebalanced, or lifecycle-altered plans."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("full-score shard-plan closed_record_sha256 is invalid")
    config = record.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("full-score shard-plan config must be an object")
    plan_id = record.get("plan_id")
    max_workers = config.get("max_workers")
    target_cache_prefix_tokens_per_shard = config.get(
        "target_cache_prefix_tokens_per_shard"
    )
    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError("full-score shard-plan plan_id must be non-empty")
    if (
        type(max_workers) is not int
        or type(target_cache_prefix_tokens_per_shard) is not int
    ):
        raise ValueError("full-score shard-plan numeric config is invalid")
    expected = build_full_score_shard_plan(
        inventory,
        plan_id=plan_id,
        max_workers=max_workers,
        target_cache_prefix_tokens_per_shard=(target_cache_prefix_tokens_per_shard),
    )
    if dict(record) != expected:
        raise ValueError("full-score shard plan does not match complete inventory")


def _validated_latency_examples(
    examples: Sequence[PublicationLatencyExample],
    *,
    examples_per_dataset: int = PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
) -> tuple[PublicationLatencyExample, ...]:
    if examples_per_dataset == PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET and (
        MAIN_LATENCY_EXAMPLES_PER_DATASET != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
    ):
        raise ValueError("main-latency and publication campaign counts diverged")
    values = tuple(examples)
    if not all(isinstance(example, PublicationLatencyExample) for example in values):
        raise TypeError("examples must contain PublicationLatencyExample values")
    identities = [(example.dataset, example.example_id) for example in values]
    if len(set(identities)) != len(identities):
        raise ValueError("publication latency examples contain duplicate identities")
    counts = Counter(example.dataset for example in values)
    if any(
        counts[dataset] != examples_per_dataset for dataset in SUPPORTED_V1_DATASETS
    ):
        raise ValueError(
            f"publication latency requires exactly {examples_per_dataset} "
            "examples per dataset"
        )
    return tuple(
        sorted(values, key=lambda example: (example.dataset, example.identity_sha256))
    )


def _latency_lanes(
    examples: Sequence[PublicationLatencyExample],
    requests: Sequence[Mapping[str, Any]],
    *,
    seed_sha256: str,
    parallelism: int,
) -> list[list[int]]:
    ordered_identities = sorted(
        examples,
        key=lambda example: _latency_order_sha256(
            seed_sha256,
            "lane",
            str(parallelism),
            example.identity_sha256,
        ),
    )
    lane_by_identity = {
        example.identity_sha256: index % parallelism
        for index, example in enumerate(ordered_identities)
    }
    lanes: list[list[int]] = [[] for _index in range(parallelism)]
    for request in requests:
        identity = cast(str, request["example_identity_sha256"])
        lanes[lane_by_identity[identity]].append(cast(int, request["request_index"]))
    return lanes


def _validate_latency_admission_windows(
    requests: Sequence[Mapping[str, Any]],
) -> None:
    identities = tuple(
        (request.get("dataset"), request.get("example_id")) for request in requests
    )
    for parallelism in PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM:
        for start in range(0, len(identities) - parallelism + 1):
            window = identities[start : start + parallelism]
            if len(set(window)) != len(window):
                raise ValueError(
                    "latency schedule has a duplicate identity in a contiguous "
                    f"parallelism-{parallelism} admission window"
                )


def _latency_order_sha256(seed_sha256: str, *parts: str) -> str:
    return _canonical_sha256(
        {"domain": _LATENCY_ORDER_DOMAIN, "parts": list(parts), "seed": seed_sha256}
    )


def _storage_selection_rank_sha256(
    input_bundle_sha256: str,
    example: PublicationLatencyExample,
) -> str:
    return _canonical_sha256(
        {
            "dataset": example.dataset,
            "domain": _STORAGE_SELECTION_DOMAIN,
            "example_id": example.example_id,
            "example_identity_sha256": example.identity_sha256,
            "input_bundle_sha256": input_bundle_sha256,
        }
    )


def _publication_storage_selection_record(
    selected_examples: Sequence[PublicationLatencyExample],
    *,
    input_bundle_sha256: str,
) -> dict[str, Any]:
    _require_sha256(input_bundle_sha256, field_name="input_bundle_sha256")
    values = _validated_latency_examples(
        selected_examples,
        examples_per_dataset=PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET,
    )
    ordered: list[dict[str, Any]] = []
    for dataset in SUPPORTED_V1_DATASETS:
        dataset_values = sorted(
            (example for example in values if example.dataset == dataset),
            key=lambda example: (
                _storage_selection_rank_sha256(input_bundle_sha256, example),
                example.identity_sha256,
            ),
        )
        for selection_index, example in enumerate(dataset_values):
            ordered.append(
                {
                    "dataset": dataset,
                    "example_id": example.example_id,
                    "example_identity_sha256": example.identity_sha256,
                    "selection_index": selection_index,
                    "selection_rank_sha256": _storage_selection_rank_sha256(
                        input_bundle_sha256, example
                    ),
                }
            )
    body: dict[str, Any] = {
        "domain": _STORAGE_SELECTION_DOMAIN,
        "input_bundle_sha256": input_bundle_sha256,
        "ordered_identities": ordered,
        "selected_examples_per_dataset": (
            PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET
        ),
        "source_examples_per_dataset": PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    }
    return {**body, "selection_sha256": _canonical_sha256(body)}


def _full_score_item_order_sha256(item: FullScoreInventoryItem) -> str:
    return _canonical_sha256(
        {
            "domain": _FULL_SCORE_ORDER_DOMAIN,
            "identity_sha256": item.identity_sha256,
            "source_record_sha256": item.source_record_sha256,
        }
    )


def _full_score_item_record(item: FullScoreInventoryItem) -> dict[str, Any]:
    return {
        "cache_prefix_sha256": item.cache_prefix_sha256,
        "cache_prefix_token_ids_sha256": item.cache_prefix_token_ids_sha256,
        "cache_prefix_tokens": item.cache_prefix_tokens,
        "dataset": item.dataset,
        "example_id": item.example_id,
        "identity_sha256": item.identity_sha256,
        "natural_prompt_sha256": item.natural_prompt_sha256,
        "natural_prompt_token_ids_sha256": (item.natural_prompt_token_ids_sha256),
        "natural_prompt_tokens": item.natural_prompt_tokens,
        "segment_count": item.segment_count,
        "segment_token_ids_sha256": item.segment_token_ids_sha256,
        "source_record_sha256": item.source_record_sha256,
    }


def _identity_closure_sha256(items: Sequence[FullScoreInventoryItem]) -> str:
    return _canonical_sha256(
        sorted(
            (
                {"dataset": item.dataset, "example_id": item.example_id}
                for item in items
            ),
            key=lambda value: (value["dataset"], value["example_id"]),
        )
    )


def _source_record_closure_sha256(items: Sequence[FullScoreInventoryItem]) -> str:
    return _canonical_sha256(sorted(item.source_record_sha256 for item in items))


def _encoded_token_ids(
    tokenizer: MainLatencyTokenizer,
    text: str,
) -> tuple[int, ...]:
    values = tokenizer.encode(
        text,
        add_special_tokens=MAIN_LATENCY_ADD_SPECIAL_TOKENS,
    )
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("tokenizer.encode() must return a token sequence")
    token_ids = tuple(values)
    if not token_ids or any(
        type(token_id) is not int or token_id < 0 or token_id >= 2**64
        for token_id in token_ids
    ):
        raise ValueError("tokenizer.encode() returned invalid token IDs")
    return token_ids


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = sha256(b"cachet.token_ids.uint64be.v1\0")
    digest.update(len(token_ids).to_bytes(8, byteorder="big", signed=False))
    for token_id in token_ids:
        digest.update(token_id.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def _segment_token_ids_sha256(
    segments: Sequence[Sequence[int]],
) -> str:
    digest = sha256(b"cachet.segment_token_ids.uint64be.v1\0")
    digest.update(len(segments).to_bytes(8, byteorder="big", signed=False))
    for token_ids in segments:
        digest.update(len(token_ids).to_bytes(8, byteorder="big", signed=False))
        for token_id in token_ids:
            digest.update(token_id.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def _require_sha256(value: Any, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _record_mapping(record: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = record.get(field_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _record_string(record: Mapping[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _canonical_json_bytes(value: Any, *, pretty: bool) -> bytes:
    kwargs: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    suffix = "\n" if pretty else ""
    return (json.dumps(value, **kwargs) + suffix).encode("utf-8")


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("closed_record_sha256", None)
    return _canonical_sha256(payload)


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "FULL_SCORE_INVENTORY_RECORD_TYPE",
    "FULL_SCORE_INVENTORY_SCHEMA_VERSION",
    "FULL_SCORE_MAX_WORKERS",
    "FULL_SCORE_MAX_NATURAL_PROMPT_TOKENS",
    "FULL_SCORE_SHARD_PLAN_RECORD_TYPE",
    "FULL_SCORE_SHARD_PLAN_SCHEMA_VERSION",
    "PUBLICATION_LATENCY_SCHEDULE_RECORD_TYPE",
    "PUBLICATION_LATENCY_SCHEDULE_SCHEMA_VERSION",
    "PUBLICATION_STORAGE_INPUTS_RECORD_TYPE",
    "PUBLICATION_STORAGE_INPUTS_SCHEMA_VERSION",
    "FullScoreDatasetSource",
    "FullScoreInventory",
    "FullScoreInventoryItem",
    "PublicationLatencyExample",
    "build_full_score_shard_plan",
    "build_publication_latency_block_schedule",
    "build_publication_storage_block_schedule",
    "build_verified_publication_latency_block_schedule",
    "full_score_inventory_to_record",
    "load_full_score_inventory",
    "load_publication_storage_selection_examples",
    "load_verified_publication_latency_examples",
    "materialize_publication_storage_inputs",
    "project_publication_latency_request_order",
    "select_publication_storage_examples",
    "validate_full_score_shard_plan",
    "validate_full_score_inventory_record",
    "validate_publication_latency_block_schedule",
    "validate_publication_storage_block_schedule",
    "validate_publication_storage_inputs_record",
]
