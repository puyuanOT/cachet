import copy
import json
from collections import Counter, defaultdict
from hashlib import sha256

import pytest

import document_kv_cache.publication_inputs as publication_inputs
from document_kv_cache.benchmarks import SUPPORTED_V1_DATASETS
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM,
    PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL,
    PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL,
)
from document_kv_cache.publication_inputs import (
    FULL_SCORE_MAX_WORKERS,
    PublicationLatencyExample,
    build_full_score_shard_plan,
    build_publication_latency_block_schedule,
    build_publication_storage_block_schedule,
    full_score_inventory_to_record,
    load_full_score_inventory,
    materialize_publication_storage_inputs,
    project_publication_latency_request_order,
    select_publication_storage_examples,
    validate_full_score_shard_plan,
    validate_full_score_inventory_record,
    validate_publication_latency_block_schedule,
    validate_publication_storage_block_schedule,
    validate_publication_storage_inputs_record,
)


class _CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) for character in text]


class _BoundaryMergingTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        token_ids = []
        cursor = 0
        while cursor < len(text):
            if text.startswith("\n\n[document", cursor):
                token_ids.append(1_000_000)
                cursor += len("\n\n[document")
            else:
                token_ids.append(ord(text[cursor]))
                cursor += 1
        return token_ids


def _latency_examples():
    return tuple(
        PublicationLatencyExample(dataset, f"{dataset}-{index:02d}")
        for dataset in SUPPORTED_V1_DATASETS
        for index in range(PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET)
    )


def _storage_examples():
    return select_publication_storage_examples(
        _latency_examples(),
        input_bundle_sha256=sha256(b"verified-input-bundle").hexdigest(),
    )


def test_storage_schedule_is_exactly_eight_identities_by_32_repeats():
    bundle_sha256 = sha256(b"verified-input-bundle").hexdigest()
    schedule = build_publication_storage_block_schedule(
        campaign_id="publication-2026",
        deployment_block=1,
        input_bundle_sha256=bundle_sha256,
        examples=_latency_examples(),
    )

    assert schedule["protocol"]["workload_id"] == "storage"
    assert schedule["protocol"]["examples_per_dataset"] == 2
    assert schedule["protocol"]["repeats_per_example"] == 32
    assert len(schedule["requests"]) == PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL
    identities = Counter(
        (item["dataset"], item["example_id"]) for item in schedule["requests"]
    )
    assert len(identities) == 8
    assert set(identities.values()) == {32}
    validate_publication_latency_block_schedule(
        schedule,
        examples=_storage_examples(),
        expected_input_bundle_sha256=bundle_sha256,
    )
    validate_publication_storage_block_schedule(
        schedule,
        source_examples=_latency_examples(),
        expected_input_bundle_sha256=bundle_sha256,
    )


def test_storage_input_materializer_preserves_exact_source_rows(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source_paths = {}
    source_rows = {}
    for dataset in SUPPORTED_V1_DATASETS:
        rows = [
            {"dataset": dataset, "example_id": f"{dataset}-{index:02d}", "value": index}
            for index in range(32)
        ]
        raw_rows = [
            (
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                )
                + "\n"
            ).encode()
            for row in rows
        ]
        path = source_root / f"{dataset}.jsonl"
        path.write_bytes(b"".join(raw_rows))
        source_paths[dataset] = path
        source_rows[dataset] = raw_rows
    bundle_sha256 = sha256(b"verified-input-bundle").hexdigest()
    schedules = {
        block: build_publication_storage_block_schedule(
            campaign_id="publication-2026",
            deployment_block=block,
            input_bundle_sha256=bundle_sha256,
            examples=_latency_examples(),
        )
        for block in range(1, 6)
    }

    record = materialize_publication_storage_inputs(
        source_paths,
        schedules,
        tmp_path / "storage-inputs",
        expected_input_bundle_sha256=bundle_sha256,
    )

    assert [item["record_count"] for item in record["files"]] == [2, 2, 2, 2]
    assert (
        json.loads(
            (
                tmp_path / "storage-inputs" / "publication-storage-inputs.json"
            ).read_text()
        )
        == record
    )
    for item in record["files"]:
        dataset = item["dataset"]
        selected_rows = [
            source_rows[dataset][int(example_id.rsplit("-", 1)[1])]
            for example_id in item["identities"]
        ]
        assert (
            tmp_path / "storage-inputs" / f"storage-16384-{dataset}.jsonl"
        ).read_bytes() == b"".join(selected_rows)
    assert (
        len(
            {
                schedule["protocol"]["selection"]["selection_sha256"]
                for schedule in schedules.values()
            }
        )
        == 1
    )
    assert (
        record["selection_protocol"]["selection_sha256"]
        == schedules[1]["protocol"]["selection"]["selection_sha256"]
    )
    validate_publication_storage_inputs_record(
        record,
        source_paths=source_paths,
        schedule_records=schedules,
        expected_input_bundle_sha256=bundle_sha256,
    )


def test_storage_schedule_rejects_caller_selected_valid_subset():
    bundle_sha256 = sha256(b"verified-input-bundle").hexdigest()
    caller_selected = tuple(
        PublicationLatencyExample(dataset, f"{dataset}-{index:02d}")
        for dataset in SUPPORTED_V1_DATASETS
        for index in range(2)
    )
    forged = publication_inputs._build_publication_storage_schedule_from_selected(
        campaign_id="publication-2026",
        deployment_block=1,
        input_bundle_sha256=bundle_sha256,
        selected_examples=caller_selected,
    )

    validate_publication_latency_block_schedule(
        forged,
        examples=caller_selected,
        expected_input_bundle_sha256=bundle_sha256,
    )
    with pytest.raises(ValueError, match="verified input bundle"):
        validate_publication_storage_block_schedule(
            forged,
            source_examples=_latency_examples(),
            expected_input_bundle_sha256=bundle_sha256,
        )


def _score_record(dataset, index):
    answer = f"answer-{dataset}-{index}"
    return {
        "dataset": dataset,
        "documents": [
            {
                "document_id": f"document-{dataset}-{index}",
                "text": f"natural source {dataset} {index} " + ("x" * (index * 37)),
                "title": f"title-{index}",
            },
            {
                "document_id": f"document-extra-{dataset}-{index}",
                "text": f"second natural source {dataset} {index}",
                "title": f"extra-title-{index}",
            },
        ],
        "example_id": f"{dataset}-{index}",
        "expected_answer": answer,
        "query": f"question-{dataset}-{index}?",
        "references": [answer],
    }


def _write_full_score_sources(root, *, records_per_dataset=9):
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for dataset in SUPPORTED_V1_DATASETS:
        path = root / f"{dataset}.jsonl"
        rows = [_score_record(dataset, index) for index in range(records_per_dataset)]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        paths[dataset] = path
    return paths


def test_latency_schedule_is_balanced_reproducible_and_method_independent():
    examples = _latency_examples()
    bundle_sha256 = sha256(b"verified-input-bundle").hexdigest()

    schedule = build_publication_latency_block_schedule(
        campaign_id="publication-2026",
        deployment_block=1,
        input_bundle_sha256=bundle_sha256,
        examples=tuple(reversed(examples)),
    )
    repeated = build_publication_latency_block_schedule(
        campaign_id="publication-2026",
        deployment_block=1,
        input_bundle_sha256=bundle_sha256,
        examples=examples,
    )

    assert schedule == repeated
    requests = schedule["requests"]
    assert len(requests) == PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL == 256
    assert Counter(request["dataset"] for request in requests) == {
        dataset: 64 for dataset in SUPPORTED_V1_DATASETS
    }
    identities = Counter(
        (request["dataset"], request["example_id"]) for request in requests
    )
    assert set(identities.values()) == {2}
    assert len(identities) == 128
    for wave_start in range(0, len(requests), len(SUPPORTED_V1_DATASETS)):
        wave = requests[wave_start : wave_start + len(SUPPORTED_V1_DATASETS)]
        assert {request["dataset"] for request in wave} == set(SUPPORTED_V1_DATASETS)

    for parallelism in PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM:
        lanes = schedule["lanes"][str(parallelism)]
        assert len(lanes) == parallelism
        assert sorted(index for lane in lanes for index in lane) == list(range(256))
        assert {len(lane) for lane in lanes} == {256 // parallelism}
        identity_lanes = defaultdict(set)
        for lane_index, lane in enumerate(lanes):
            for request_index in lane:
                request = requests[request_index]
                identity_lanes[(request["dataset"], request["example_id"])].add(
                    lane_index
                )
        assert all(len(lane_ids) == 1 for lane_ids in identity_lanes.values())

    validate_publication_latency_block_schedule(
        schedule,
        examples=examples,
        expected_input_bundle_sha256=bundle_sha256,
    )
    projection = project_publication_latency_request_order(
        schedule,
        examples=examples,
        expected_input_bundle_sha256=bundle_sha256,
    )
    assert len(projection) == len(set(projection)) == 256
    projected_identities = [
        (dataset, example_id) for dataset, example_id, _ in projection
    ]
    for parallelism in PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM:
        assert all(
            len(set(projected_identities[start : start + parallelism])) == parallelism
            for start in range(len(projection) - parallelism + 1)
        )


def test_latency_schedule_uses_independent_blocks_and_fails_closed():
    examples = _latency_examples()
    bundle_sha256 = sha256(b"verified-input-bundle").hexdigest()
    first = build_publication_latency_block_schedule(
        campaign_id="publication-2026",
        deployment_block=1,
        input_bundle_sha256=bundle_sha256,
        examples=examples,
    )
    second = build_publication_latency_block_schedule(
        campaign_id="publication-2026",
        deployment_block=2,
        input_bundle_sha256=bundle_sha256,
        examples=examples,
    )

    assert first["seed_sha256"] != second["seed_sha256"]
    assert first["requests_sha256"] != second["requests_sha256"]

    tampered = copy.deepcopy(first)
    tampered["requests"][0]["repeat_index"] = 99
    with pytest.raises(ValueError, match="closed_record_sha256"):
        validate_publication_latency_block_schedule(
            tampered,
            examples=examples,
            expected_input_bundle_sha256=bundle_sha256,
        )

    different_examples = list(examples)
    different_examples[0] = PublicationLatencyExample(
        different_examples[0].dataset,
        "replacement-example",
    )
    with pytest.raises(ValueError, match="verified input bundle"):
        validate_publication_latency_block_schedule(
            first,
            examples=different_examples,
            expected_input_bundle_sha256=bundle_sha256,
        )


def test_latency_schedule_rejects_any_sampled_dataset():
    with pytest.raises(ValueError, match="exactly 32 examples per dataset"):
        build_publication_latency_block_schedule(
            campaign_id="publication-2026",
            deployment_block=1,
            input_bundle_sha256=sha256(b"bundle").hexdigest(),
            examples=_latency_examples()[:-1],
        )


def test_full_score_inventory_keeps_every_natural_prompt(tmp_path):
    source_paths = _write_full_score_sources(tmp_path, records_per_dataset=9)
    inventory = load_full_score_inventory(
        source_paths,
        tokenizer=_CharacterTokenizer(),
    )
    record = full_score_inventory_to_record(inventory)

    assert len(inventory.items) == 9 * len(SUPPORTED_V1_DATASETS)
    assert {source.record_count for source in inventory.sources} == {9}
    assert record["input_length_policy"] == {
        "max_natural_prompt_tokens": 32_768,
        "padding": False,
        "segment_token_id_digest_encoding": (
            "segment_count_and_lengths_then_uint64be_v1"
        ),
        "token_id_digest_encoding": "length_then_uint64be_v1",
        "tokenizer_truncation": False,
    }
    assert len(record["items"]) == len(inventory.items)
    assert all(
        {
            "natural_prompt_sha256",
            "natural_prompt_token_ids_sha256",
            "cache_prefix_sha256",
            "cache_prefix_token_ids_sha256",
            "segment_token_ids_sha256",
        }.issubset(item)
        for item in record["items"]
    )
    assert all(item.natural_prompt_tokens > 0 for item in inventory.items)
    assert all(item.cache_prefix_tokens > 0 for item in inventory.items)
    assert all(
        item.cache_prefix_tokens <= item.natural_prompt_tokens
        for item in inventory.items
    )
    assert len({(item.dataset, item.example_id) for item in inventory.items}) == len(
        inventory.items
    )
    validate_full_score_inventory_record(record, inventory=inventory)
    tampered = copy.deepcopy(record)
    tampered["items"].pop()
    with pytest.raises(ValueError, match="closed_record_sha256"):
        validate_full_score_inventory_record(tampered, inventory=inventory)


def test_full_score_shard_plan_is_complete_balanced_and_streaming(tmp_path):
    inventory = load_full_score_inventory(
        _write_full_score_sources(tmp_path, records_per_dataset=17),
        tokenizer=_CharacterTokenizer(),
    )
    target_tokens = 2_000
    plan = build_full_score_shard_plan(
        inventory,
        plan_id="full-score-publication-2026",
        max_workers=FULL_SCORE_MAX_WORKERS,
        target_cache_prefix_tokens_per_shard=target_tokens,
    )

    shards = plan["shards"]
    planned_items = [item for shard in shards for item in shard["items"]]
    planned_identities = [
        (item["dataset"], item["example_id"]) for item in planned_items
    ]
    inventory_identities = [(item.dataset, item.example_id) for item in inventory.items]
    assert Counter(planned_identities) == Counter(inventory_identities)
    assert set(Counter(planned_identities).values()) == {1}
    assert plan["coverage"]["identity_count"] == len(inventory.items)
    assert plan["coverage"]["no_duplicate_identities"] is True
    assert plan["coverage"]["cache_prefix_generation_tokens"] == sum(
        item.cache_prefix_tokens for item in inventory.items
    )
    assert plan["coverage"]["natural_prompt_inference_tokens"] == sum(
        item.natural_prompt_tokens for item in inventory.items
    )
    assert plan["config"]["active_workers"] <= FULL_SCORE_MAX_WORKERS
    assert all(
        shard["cache_prefix_tokens"] <= target_tokens or shard["item_count"] == 1
        for shard in shards
    )
    assert plan["lifecycle"]["steps"] == [
        "generate_q8_kv",
        "baseline_inference",
        "vanilla_inference",
        "validate_paired_outputs",
        "delete_ephemeral_q8_kv",
    ]
    assert plan["config"]["max_inflight_shards_per_worker"] == 1

    worker_loads = [worker["cache_prefix_tokens"] for worker in plan["worker_loads"]]
    assert sum(worker_loads) == plan["coverage"]["cache_prefix_generation_tokens"]
    assert max(worker_loads) - min(worker_loads) <= max(
        shard["cache_prefix_tokens"] for shard in shards
    )
    validate_full_score_shard_plan(plan, inventory=inventory)


def test_full_score_shard_plan_rejects_tamper_and_wrong_inventory(tmp_path):
    inventory = load_full_score_inventory(
        _write_full_score_sources(tmp_path / "first", records_per_dataset=4),
        tokenizer=_CharacterTokenizer(),
    )
    plan = build_full_score_shard_plan(
        inventory,
        plan_id="full-score-publication-2026",
        max_workers=4,
        target_cache_prefix_tokens_per_shard=800,
    )
    tampered = copy.deepcopy(plan)
    tampered["shards"][0]["items"].pop()
    with pytest.raises(ValueError, match="closed_record_sha256"):
        validate_full_score_shard_plan(tampered, inventory=inventory)

    different_inventory = load_full_score_inventory(
        _write_full_score_sources(tmp_path / "second", records_per_dataset=5),
        tokenizer=_CharacterTokenizer(),
    )
    with pytest.raises(ValueError, match="complete inventory"):
        validate_full_score_shard_plan(plan, inventory=different_inventory)


@pytest.mark.parametrize("max_workers", (0, FULL_SCORE_MAX_WORKERS + 1, True))
def test_full_score_shard_plan_caps_workers(tmp_path, max_workers):
    inventory = load_full_score_inventory(
        _write_full_score_sources(tmp_path, records_per_dataset=2),
        tokenizer=_CharacterTokenizer(),
    )
    with pytest.raises(ValueError, match="max_workers"):
        build_full_score_shard_plan(
            inventory,
            plan_id="invalid-workers",
            max_workers=max_workers,
            target_cache_prefix_tokens_per_shard=1_000,
        )


def test_full_score_inventory_rejects_noncomposing_segments_and_overflow(tmp_path):
    sources = _write_full_score_sources(tmp_path / "segments", records_per_dataset=1)
    with pytest.raises(ValueError, match="do not compose"):
        load_full_score_inventory(sources, tokenizer=_BoundaryMergingTokenizer())

    with pytest.raises(ValueError, match="natural prompt exceeds"):
        load_full_score_inventory(
            sources,
            tokenizer=_CharacterTokenizer(),
            max_natural_prompt_tokens=10,
        )
