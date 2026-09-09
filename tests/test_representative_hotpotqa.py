import json
from pathlib import Path

import pytest

from document_kv_cache._benchmark_datasets import load_benchmark_jsonl
from document_kv_cache.benchmarks import build_prompt_parts
from document_kv_cache.dataset_prep import (
    REPRESENTATIVE_HOTPOTQA_ADD_SPECIAL_TOKENS,
    REPRESENTATIVE_HOTPOTQA_RECORD_TYPE,
    REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID,
    REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION,
    load_representative_hotpotqa_tokenizer,
    prepare_representative_hotpotqa_jsonl,
    representative_hotpotqa_main,
)


class _WordTokenizer:
    def __init__(self):
        self.add_special_tokens_values = []

    def encode(self, text, *, add_special_tokens):
        self.add_special_tokens_values.append(add_special_tokens)
        return list(range(len(text.split())))


class _ImpossibleTokenizer:
    def encode(self, _text, *, add_special_tokens):
        assert add_special_tokens is False
        return []


def _official_record(example_id, *, document_count=2):
    return {
        "_id": example_id,
        "question": f"Secret question {example_id}?",
        "answer": f"Secret answer {example_id}",
        "context": [
            [
                f"Secret title {example_id}-{index}",
                [f"Secret document text {example_id}-{index}."],
            ]
            for index in range(document_count)
        ],
        "supporting_facts": [[f"Secret title {example_id}-0", 0]],
        "type": "bridge",
        "level": "medium",
    }


def _write_official_source(path: Path, *, records=None):
    values = records or [
        _official_record("example-b"),
        _official_record("example-a"),
        _official_record("example-c"),
    ]
    path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("target", [8192, 16384, 32768])
def test_prepare_representative_hotpotqa_is_exact_deterministic_and_sanitized(
    tmp_path,
    target,
):
    source = tmp_path / "hotpot_dev_distractor_v1.json"
    _write_official_source(source)
    tokenizer_a = _WordTokenizer()
    tokenizer_b = _WordTokenizer()

    first = prepare_representative_hotpotqa_jsonl(
        source,
        tmp_path / "first" / "hotpotqa.jsonl",
        provenance_json=tmp_path / "first" / "provenance.json",
        input_tokens_target=target,
        tokenizer=tokenizer_a,
    )
    second = prepare_representative_hotpotqa_jsonl(
        source,
        tmp_path / "second" / "hotpotqa.jsonl",
        provenance_json=tmp_path / "second" / "provenance.json",
        input_tokens_target=target,
        tokenizer=tokenizer_b,
    )

    assert first.jsonl_path.read_bytes() == second.jsonl_path.read_bytes()
    assert first.provenance_json_path.read_bytes() == second.provenance_json_path.read_bytes()
    assert first.jsonl_sha256 == second.jsonl_sha256
    examples = load_benchmark_jsonl(
        first.jsonl_path,
        dataset="hotpotqa",
        require_dataset=True,
    )
    assert [example.example_id for example in examples] == ["example-a", "example-b"]
    assert all(len(example.documents) >= 2 for example in examples)
    assert [example.expected_answer for example in examples] == [
        "Secret answer example-a",
        "Secret answer example-b",
    ]
    assert {
        len(
            tokenizer_a.encode(
                build_prompt_parts(example).prefill_prompt,
                add_special_tokens=False,
            )
        )
        for example in examples
    } == {target}
    assert tokenizer_a.add_special_tokens_values
    assert set(tokenizer_a.add_special_tokens_values) == {
        REPRESENTATIVE_HOTPOTQA_ADD_SPECIAL_TOKENS
    }

    provenance = json.loads(first.provenance_json_path.read_text(encoding="utf-8"))
    assert provenance["record_type"] == REPRESENTATIVE_HOTPOTQA_RECORD_TYPE
    assert provenance["input_tokens_target"] == target
    assert provenance["output"]["jsonl_sha256"] == first.jsonl_sha256
    assert provenance["tokenizer"] == {
        "add_special_tokens": False,
        "tokenizer_id": REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID,
        "tokenizer_revision": REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION,
    }
    assert {row["prepared_token_count"] for row in provenance["examples"]} == {
        target
    }
    serialized_provenance = json.dumps(provenance, sort_keys=True)
    for secret in (
        "example-a",
        "example-b",
        "Secret question",
        "Secret answer",
        "Secret title",
        "Secret document text",
        " padding",
        str(tmp_path),
    ):
        assert secret not in serialized_provenance


def test_prepare_representative_hotpotqa_preserves_reference_fields(tmp_path):
    source = tmp_path / "source.json"
    records = [_official_record("a"), _official_record("b")]
    for record in records:
        record["references"] = [record["answer"], f"Alternate {record['_id']}"]
    _write_official_source(source, records=records)

    result = prepare_representative_hotpotqa_jsonl(
        source,
        tmp_path / "hotpotqa.jsonl",
        input_tokens_target=8192,
        tokenizer=_WordTokenizer(),
    )

    output_rows = [
        json.loads(line)
        for line in result.jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert output_rows[0]["expected_answer"] == "Secret answer a"
    assert output_rows[0]["references"] == ["Secret answer a", "Alternate a"]
    assert output_rows[1]["expected_answer"] == "Secret answer b"
    assert output_rows[1]["references"] == ["Secret answer b", "Alternate b"]


def test_prepare_representative_hotpotqa_accepts_canonical_jsonl(tmp_path):
    source = tmp_path / "source.json"
    _write_official_source(source)
    first = prepare_representative_hotpotqa_jsonl(
        source,
        tmp_path / "prepared" / "hotpotqa.jsonl",
        input_tokens_target=8192,
        tokenizer=_WordTokenizer(),
    )

    second = prepare_representative_hotpotqa_jsonl(
        first.jsonl_path,
        tmp_path / "reprepared" / "hotpotqa.jsonl",
        input_tokens_target=8192,
        tokenizer=_WordTokenizer(),
    )

    assert second.jsonl_path.read_bytes() == first.jsonl_path.read_bytes()
    provenance = json.loads(second.provenance_json_path.read_text(encoding="utf-8"))
    assert provenance["source"]["format"] == "cachet.canonical_hotpotqa_jsonl"


@pytest.mark.parametrize("target", [0, 4096, 8192.0, True])
def test_prepare_representative_hotpotqa_rejects_noncanonical_target(tmp_path, target):
    source = tmp_path / "source.json"
    _write_official_source(source)

    with pytest.raises(ValueError, match="exactly one of"):
        prepare_representative_hotpotqa_jsonl(
            source,
            tmp_path / "hotpotqa.jsonl",
            input_tokens_target=target,
            tokenizer=_WordTokenizer(),
        )


def test_prepare_representative_hotpotqa_rejects_invalid_source(tmp_path):
    too_few = tmp_path / "too-few.json"
    _write_official_source(too_few, records=[_official_record("only")])
    with pytest.raises(ValueError, match="at least 2 distinct"):
        prepare_representative_hotpotqa_jsonl(
            too_few,
            tmp_path / "too-few.jsonl",
            input_tokens_target=8192,
            tokenizer=_WordTokenizer(),
        )

    one_document = tmp_path / "one-document.json"
    _write_official_source(
        one_document,
        records=[
            _official_record("a", document_count=1),
            _official_record("b"),
        ],
    )
    with pytest.raises(ValueError, match="at least two documents"):
        prepare_representative_hotpotqa_jsonl(
            one_document,
            tmp_path / "one-document.jsonl",
            input_tokens_target=8192,
            tokenizer=_WordTokenizer(),
        )

    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps([{"_id": "a"}, {"_id": "b"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        prepare_representative_hotpotqa_jsonl(
            malformed,
            tmp_path / "malformed.jsonl",
            input_tokens_target=8192,
            tokenizer=_WordTokenizer(),
        )


def test_prepare_representative_hotpotqa_fails_closed_when_exact_size_is_impossible(
    tmp_path,
):
    source = tmp_path / "source.json"
    _write_official_source(source)

    with pytest.raises(ValueError, match="cannot produce 2 exact 8192-token examples"):
        prepare_representative_hotpotqa_jsonl(
            source,
            tmp_path / "hotpotqa.jsonl",
            input_tokens_target=8192,
            tokenizer=_ImpossibleTokenizer(),
        )
    assert not (tmp_path / "hotpotqa.jsonl").exists()


def test_load_representative_hotpotqa_tokenizer_uses_exact_pin(monkeypatch):
    sentinel = _WordTokenizer()
    calls = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return sentinel

    class _Transformers:
        AutoTokenizer = _AutoTokenizer

    monkeypatch.setattr(
        "document_kv_cache._representative_hotpotqa.importlib.import_module",
        lambda name: _Transformers if name == "transformers" else None,
    )

    loaded = load_representative_hotpotqa_tokenizer()

    assert loaded is sentinel
    assert calls == [
        (
            REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID,
            {
                "revision": REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION,
                "trust_remote_code": False,
                "use_fast": True,
            },
        )
    ]


def test_representative_hotpotqa_cli_uses_tokenizer_only_path(
    tmp_path,
    monkeypatch,
    capsys,
):
    source = tmp_path / "source.json"
    _write_official_source(source)
    monkeypatch.setattr(
        "document_kv_cache._representative_hotpotqa.load_representative_hotpotqa_tokenizer",
        _WordTokenizer,
    )

    exit_code = representative_hotpotqa_main(
        [
            "--source",
            str(source),
            "--output-jsonl",
            str(tmp_path / "hotpotqa.jsonl"),
            "--input-tokens-target",
            "8192",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["input_tokens_target"] == 8192
    assert output["tokenizer_id"] == REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID
