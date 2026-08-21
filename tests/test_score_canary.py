import json
from hashlib import sha256
from pathlib import Path

import pytest

from document_kv_cache._benchmark_cli import parse_benchmark_arm_specs
from document_kv_cache._benchmark_datasets import load_benchmark_jsonl
from document_kv_cache.benchmarks import SUPPORTED_V1_DATASETS, build_prompt_parts
from document_kv_cache.score_canary import (
    SCORE_CANARY_EXAMPLES_PER_DATASET,
    SCORE_CANARY_INPUT_TOKENS,
    SCORE_CANARY_KV_BYTES_PER_TOKEN,
    SCORE_CANARY_MAX_TOKENS,
    SCORE_CANARY_PROTOCOL_ID,
    SCORE_CANARY_REPEATS,
    SCORE_CANARY_REQUEST_PARALLELISM,
    ScoreCanaryProtocol,
    prepare_score_canary,
    validate_score_canary_manifest,
)


class _WordTokenizer:
    def __init__(self):
        self.add_special_tokens_values = []

    def encode(self, text, *, add_special_tokens):
        self.add_special_tokens_values.append(add_special_tokens)
        return [
            sum((index + 1) * ord(character) for index, character in enumerate(token))
            for token in text.split()
        ]


class _BoundaryMergeTokenizer:
    """Character tokenizer with one merge that crosses each document boundary."""

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        marker = '\n\n[document id="'
        token_ids = []
        index = 0
        while index < len(text):
            if text.startswith(marker, index):
                token_ids.append(1_000_000)
                index += len(marker)
            else:
                token_ids.append(ord(text[index]))
                index += 1
        return token_ids


def _record(dataset: str, index: int):
    documents = [
        {
            "document_id": f"{dataset}-document-{index}",
            "text": f"Source text for {dataset} example {index} with relevant answer {index}.",
            "title": f"Title {index}",
        }
    ]
    if dataset in {"hotpotqa", "musique"}:
        documents.append(
            {
                "document_id": f"{dataset}-distractor-{index}",
                "text": f"Unrelated second document for example {index}.",
                "title": f"Distractor {index}",
            }
        )
    return {
        "dataset": dataset,
        "documents": documents,
        "example_id": f"{dataset}-{index:02d}",
        "expected_answer": f"answer-{index}",
        "metadata": {"source": "unit-test"},
        "query": f"What is answer {index}?",
    }


def _write_sources(root: Path, *, reverse: bool = False, transfer: bool = False):
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for dataset in SUPPORTED_V1_DATASETS:
        records = [_record(dataset, index) for index in range(8)]
        if transfer and dataset == "biography":
            records[0]["kv_transfer_params"] = {}
        if reverse:
            records.reverse()
        path = root / f"{dataset}.jsonl"
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        paths[dataset] = path
    return paths


def _test_protocol(*, count: int = 2):
    return ScoreCanaryProtocol(
        protocol_id=f"test-score-canary-n{count}",
        input_tokens=160,
        examples_per_dataset=count,
        max_tokens=13,
        request_parallelism=2,
        repeats=1,
        selection_seed="unit-test-selection-v1",
    )


def _manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value):
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(content).hexdigest()


def test_production_protocol_is_small_matched_non_publication_canary():
    protocol = ScoreCanaryProtocol()

    assert protocol.protocol_id == SCORE_CANARY_PROTOCOL_ID
    assert protocol.input_tokens == SCORE_CANARY_INPUT_TOKENS == 8192
    assert protocol.examples_per_dataset == SCORE_CANARY_EXAMPLES_PER_DATASET == 5
    assert protocol.max_tokens == SCORE_CANARY_MAX_TOKENS == 64
    assert protocol.request_parallelism == SCORE_CANARY_REQUEST_PARALLELISM == 4
    assert protocol.repeats == SCORE_CANARY_REPEATS == 1
    assert (
        len(SUPPORTED_V1_DATASETS)
        * protocol.examples_per_dataset
        * protocol.input_tokens
        * SCORE_CANARY_KV_BYTES_PER_TOKEN
        == 12_079_595_520
    )


def test_prepare_is_exact_content_addressed_and_runner_compatible(tmp_path):
    sources = _write_sources(tmp_path / "sources")
    tokenizer_a = _WordTokenizer()
    tokenizer_b = _WordTokenizer()
    protocol = _test_protocol()

    first = prepare_score_canary(
        sources,
        tmp_path / "first",
        tokenizer=tokenizer_a,
        protocol=protocol,
    )
    second = prepare_score_canary(
        sources,
        tmp_path / "second",
        tokenizer=tokenizer_b,
        protocol=protocol,
    )

    assert first.example_count == second.example_count == 8
    assert first.suite_sha256 == second.suite_sha256
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert tokenizer_a.add_special_tokens_values
    assert set(tokenizer_a.add_special_tokens_values) == {False}
    manifest = _manifest(first.manifest_path)
    assert manifest["protocol"]["job_plan"]["isolated_job_count"] == 2
    assert manifest["protocol"]["evidence"]["inferential_claims_permitted"] is False
    assert manifest["protocol"]["decode"] == {
        "force_max_tokens": False,
        "max_tokens": 13,
        "stop": "natural_eos",
        "stream": True,
        "temperature": 0.0,
        "top_p": "omitted; pinned vLLM 0.23.0 default",
    }
    assert {
        dataset: manifest["protocol"]["primary_metrics"][dataset]["metric"]
        for dataset in SUPPORTED_V1_DATASETS
    } == {
        "biography": "answer_found",
        "hotpotqa": "f1",
        "musique": "answer_found",
        "niah": "exact_match",
    }
    vanilla_arm = manifest["protocol"]["arms"]["vanilla"]["arm_spec"]
    arms, _urls, _endpoints, _bodies = parse_benchmark_arm_specs((vanilla_arm,))
    assert arms[0].cache_method == "vanilla_prefill"
    assert arms[0].requires_cachet_handoff is True
    assert arms[0].physical_transform_id == "cachet.vanilla.per_document_segments"

    for dataset, path in first.dataset_paths.items():
        output = manifest["datasets"][dataset]["output"]
        assert path.name == f"{dataset}-{output['jsonl_sha256']}.jsonl"
        examples = load_benchmark_jsonl(path, dataset=dataset, require_dataset=True)
        assert len(examples) == 2
        assert {
            len(
                tokenizer_a.encode(
                    build_prompt_parts(example).prefill_prompt,
                    add_special_tokens=False,
                )
            )
            for example in examples
        } == {160}
        assert all(not example.kv_transfer_params for example in examples)
        assert all(not example.arm_kv_transfer_params for example in examples)
        for attestation in manifest["datasets"][dataset]["examples"]:
            composition = attestation["token_composition"]
            assert composition["composed_equals_cache_prefix"] is True
            assert composition["cache_prefix_is_strict_full_prompt_prefix"] is True
            assert composition["composed_token_count"] == sum(
                segment["token_count"] for segment in composition["segments"]
            )
            assert (
                composition["composed_token_ids_sha256"]
                == composition["cache_prefix_token_ids_sha256"]
            )

    validated = validate_score_canary_manifest(
        first.manifest_path,
        tokenizer=_WordTokenizer(),
        sources=sources,
    )
    assert validated.example_count == 8
    assert validated.suite_sha256 == first.suite_sha256


def test_selection_is_independent_of_source_row_order(tmp_path):
    forward_sources = _write_sources(tmp_path / "forward")
    reverse_sources = _write_sources(tmp_path / "reverse", reverse=True)
    protocol = _test_protocol(count=3)

    forward = prepare_score_canary(
        forward_sources,
        tmp_path / "forward-output",
        tokenizer=_WordTokenizer(),
        protocol=protocol,
    )
    reverse = prepare_score_canary(
        reverse_sources,
        tmp_path / "reverse-output",
        tokenizer=_WordTokenizer(),
        protocol=protocol,
    )

    assert forward.suite_sha256 == reverse.suite_sha256
    for dataset in SUPPORTED_V1_DATASETS:
        assert (
            forward.dataset_paths[dataset].read_bytes()
            == reverse.dataset_paths[dataset].read_bytes()
        )
    forward_manifest = _manifest(forward.manifest_path)
    reverse_manifest = _manifest(reverse.manifest_path)
    assert any(
        forward_manifest["datasets"][dataset]["source"]["jsonl_sha256"]
        != reverse_manifest["datasets"][dataset]["source"]["jsonl_sha256"]
        for dataset in SUPPORTED_V1_DATASETS
    )


def test_validation_detects_tampered_output_and_protocol(tmp_path):
    sources = _write_sources(tmp_path / "sources")
    prepared = prepare_score_canary(
        sources,
        tmp_path / "output",
        tokenizer=_WordTokenizer(),
        protocol=_test_protocol(),
    )
    biography_path = prepared.dataset_paths["biography"]
    original = biography_path.read_bytes()
    biography_path.write_bytes(original + b"\n")

    with pytest.raises(ValueError, match="output SHA-256 mismatch"):
        validate_score_canary_manifest(
            prepared.manifest_path,
            tokenizer=_WordTokenizer(),
        )

    biography_path.write_bytes(original)
    original_manifest = prepared.manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    composition = manifest["datasets"]["biography"]["examples"][0][
        "token_composition"
    ]
    composition["segments"][0]["token_count"] += 1
    examples = manifest["datasets"]["biography"]["examples"]
    manifest["datasets"]["biography"]["examples_sha256"] = _canonical_sha256(
        examples
    )
    prepared.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="token composition mismatch"):
        validate_score_canary_manifest(
            prepared.manifest_path,
            tokenizer=_WordTokenizer(),
        )

    prepared.manifest_path.write_bytes(original_manifest)
    manifest = _manifest(prepared.manifest_path)
    manifest["protocol"]["decode"]["force_max_tokens"] = True
    manifest["protocol_sha256"] = "0" * 64
    prepared.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol_sha256"):
        validate_score_canary_manifest(
            prepared.manifest_path,
            tokenizer=_WordTokenizer(),
        )


def test_prepare_rejects_cross_segment_token_merge(tmp_path):
    sources = _write_sources(tmp_path / "sources")
    protocol = ScoreCanaryProtocol(
        protocol_id="test-boundary-merge",
        input_tokens=2000,
        examples_per_dataset=1,
        max_tokens=13,
        request_parallelism=1,
        repeats=1,
        selection_seed="boundary-merge-v1",
    )

    with pytest.raises(ValueError, match="cannot produce 1 exact"):
        prepare_score_canary(
            sources,
            tmp_path / "output",
            tokenizer=_BoundaryMergeTokenizer(),
            protocol=protocol,
        )


def test_prepare_rejects_transfer_metadata_and_insufficient_rows(tmp_path):
    sources = _write_sources(tmp_path / "transfer", transfer=True)
    with pytest.raises(ValueError, match="contains KV transfer metadata"):
        prepare_score_canary(
            sources,
            tmp_path / "transfer-output",
            tokenizer=_WordTokenizer(),
            protocol=_test_protocol(),
        )

    too_few = _write_sources(tmp_path / "too-few")
    biography = [_record("biography", 0)]
    too_few["biography"].write_text(
        json.dumps(biography[0], sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot produce 2 exact"):
        prepare_score_canary(
            too_few,
            tmp_path / "too-few-output",
            tokenizer=_WordTokenizer(),
            protocol=_test_protocol(),
        )
