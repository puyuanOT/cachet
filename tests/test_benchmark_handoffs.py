import json

import pytest

import cachet.benchmark_handoffs as cachet_benchmark_handoffs
import document_kv_cache.benchmark_handoffs as public_benchmark_handoffs
import restaurant_kv_serving.benchmark_handoffs as legacy_benchmark_handoffs
from document_kv_cache.benchmark_handoffs import (
    BENCHMARK_HANDOFF_MANIFEST_RECORD_TYPE,
    BENCHMARK_HANDOFF_MANIFEST_SCHEMA_VERSION,
    BenchmarkHandoffEntry,
    BenchmarkHandoffManifest,
    benchmark_handoff_manifest_to_record,
    enrich_benchmark_jsonl_with_handoffs,
    enrich_benchmark_records_with_handoffs,
)
from document_kv_cache.benchmark_runner import load_benchmark_jsonl
from document_kv_cache.benchmarks import (
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
)
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import (
    build_engine_adapter_request,
    engine_adapter_request_to_record,
    vllm_adapter_spec,
)
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment


def record(example_id="bio-1", *, dataset="biography", kv_transfer_params=None):
    row = {
        "dataset": dataset,
        "example_id": example_id,
        "query": "Who wrote notes?",
        "expected_answer": "Ada Lovelace",
        "documents": [
            {
                "document_id": "ada",
                "title": "Ada",
                "text": "Ada Lovelace wrote notes on the Analytical Engine.",
            }
        ],
    }
    if kv_transfer_params is not None:
        row["kv_transfer_params"] = kv_transfer_params
    return row


def manifest(*entries):
    return BenchmarkHandoffManifest(entries=tuple(entries))


def entry(example_id="bio-1", *, request_id="cachet-bio-1", **kwargs):
    return BenchmarkHandoffEntry(
        dataset="biography",
        example_id=example_id,
        request_id=request_id,
        handoff_json=f"/Volumes/catalog/schema/volume/cachet/{request_id}.handoff.json",
        payload_uri=f"uc-volume:/catalog/schema/volume/cachet/{request_id}.kv",
        **kwargs,
    )


def inline_handoff_record(*, request_id="cachet-bio-1", payload_uri=None):
    layout = KVLayout(
        model_id="tiny-test-model",
        lora_id="base",
        layout_version="standard-v1",
        dtype="int8",
        num_layers=1,
        block_size=2,
        bytes_per_token=4,
    )
    handle = KVCacheHandle(
        request_id=request_id,
        handle_uri=f"document-kv://{request_id}",
        layout=layout,
        segments=(KVSegment("doc-1", "document_static", "static", 0, 1, 0, 4),),
        total_tokens=1,
        total_bytes=4,
    )
    ready = EngineReadyRequest(handle=handle, payload=b"data", estimated_gpu_bytes=4)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    return engine_adapter_request_to_record(
        adapter_request,
        payload_uri=payload_uri or f"disk:/tmp/{request_id}.kv",
    )


def test_enrich_benchmark_jsonl_with_handoffs_writes_loadable_rows(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    manifest_path = tmp_path / "handoffs.json"
    output_path = tmp_path / "bio.enriched.jsonl"
    input_path.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    handoffs = manifest(entry())
    manifest_path.write_text(
        json.dumps(benchmark_handoff_manifest_to_record(handoffs)),
        encoding="utf-8",
    )

    count = enrich_benchmark_jsonl_with_handoffs(input_path, manifest_path, output_path)
    loaded = load_benchmark_jsonl(output_path, dataset="biography", require_dataset=True)

    assert count == 1
    assert loaded[0].kv_transfer_params == {
        DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
        DOCUMENT_KV_HANDOFF_JSON_PARAM: "/Volumes/catalog/schema/volume/cachet/cachet-bio-1.handoff.json",
        DOCUMENT_KV_PAYLOAD_URI_PARAM: "uc-volume:/catalog/schema/volume/cachet/cachet-bio-1.kv",
    }


def test_benchmark_handoff_manifest_record_is_stable():
    handoff_record = inline_handoff_record(request_id="cachet-bio-1")
    handoffs = manifest(
        BenchmarkHandoffEntry(
            dataset="biography",
            example_id="bio-1",
            request_id="cachet-bio-1",
            handoff_record=handoff_record,
        )
    )

    assert benchmark_handoff_manifest_to_record(handoffs) == {
        "record_type": BENCHMARK_HANDOFF_MANIFEST_RECORD_TYPE,
        "schema_version": BENCHMARK_HANDOFF_MANIFEST_SCHEMA_VERSION,
        "entries": [
            {
                "dataset": "biography",
                "example_id": "bio-1",
                "request_id": "cachet-bio-1",
                "handoff_record": handoff_record,
            }
        ],
    }


def test_enrich_records_rejects_duplicate_manifest_entries():
    with pytest.raises(ValueError, match="Duplicate handoff manifest entries for biography/bio-1"):
        manifest(entry(), entry())


def test_manifest_entry_rejects_runtime_unreadable_payload_uri():
    with pytest.raises(ValueError, match="payload_uri: payload_uri must be an absolute path"):
        BenchmarkHandoffEntry(
            dataset="biography",
            example_id="bio-1",
            request_id="cachet-bio-1",
            handoff_json="/Volumes/catalog/schema/volume/cachet/cachet-bio-1.handoff.json",
            payload_uri="not-a-uri-or-absolute-path",
        )


def test_enrich_records_rejects_duplicate_input_rows():
    with pytest.raises(ValueError, match="Duplicate benchmark input rows for biography/bio-1"):
        enrich_benchmark_records_with_handoffs(
            [record("bio-1"), record("bio-1")],
            manifest(entry("bio-1")),
        )


def test_enrich_records_rejects_missing_manifest_entries_by_default():
    with pytest.raises(ValueError, match="Missing handoff manifest entries for biography/bio-2"):
        enrich_benchmark_records_with_handoffs(
            [record("bio-1"), record("bio-2")],
            manifest(entry("bio-1")),
        )


def test_enrich_records_rejects_unmatched_manifest_entries_by_default():
    with pytest.raises(ValueError, match="Unmatched handoff manifest entries for biography/bio-2"):
        enrich_benchmark_records_with_handoffs(
            [record("bio-1")],
            manifest(entry("bio-1"), entry("bio-2", request_id="cachet-bio-2")),
            allow_missing=True,
        )


def test_enrich_records_rejects_existing_kv_transfer_params_without_overwrite():
    existing = {
        DOCUMENT_KV_REQUEST_ID_PARAM: "existing",
        DOCUMENT_KV_HANDOFF_JSON_PARAM: "/tmp/existing.handoff.json",
    }

    with pytest.raises(ValueError, match="already has kv_transfer_params"):
        enrich_benchmark_records_with_handoffs(
            [record(kv_transfer_params=existing)],
            manifest(entry()),
        )


def test_enrich_records_overwrites_existing_kv_transfer_params_when_requested():
    existing = {
        DOCUMENT_KV_REQUEST_ID_PARAM: "existing",
        DOCUMENT_KV_HANDOFF_JSON_PARAM: "/tmp/existing.handoff.json",
    }

    enriched = enrich_benchmark_records_with_handoffs(
        [record(kv_transfer_params=existing)],
        manifest(entry()),
        overwrite=True,
    )

    assert enriched[0]["kv_transfer_params"][DOCUMENT_KV_REQUEST_ID_PARAM] == "cachet-bio-1"


def test_inline_handoff_record_must_match_manifest_request_id():
    with pytest.raises(ValueError, match="handoff_record.request_id must match request_id"):
        BenchmarkHandoffEntry(
            dataset="biography",
            example_id="bio-1",
            request_id="cachet-bio-1",
            handoff_record=inline_handoff_record(request_id="different"),
        )


def test_inline_handoff_record_rejects_runtime_unreadable_embedded_payload_uri():
    with pytest.raises(ValueError, match="handoff_record.payload_source.uri"):
        BenchmarkHandoffEntry(
            dataset="biography",
            example_id="bio-1",
            request_id="cachet-bio-1",
            handoff_record=inline_handoff_record(
                request_id="cachet-bio-1",
                payload_uri="s3://bucket/cachet-bio-1.kv",
            ),
        )


def test_inline_handoff_record_accepts_runtime_readable_payload_override():
    handoff_record = inline_handoff_record(
        request_id="cachet-bio-1",
        payload_uri="s3://bucket/cachet-bio-1.kv",
    )

    entry = BenchmarkHandoffEntry(
        dataset="biography",
        example_id="bio-1",
        request_id="cachet-bio-1",
        handoff_record=handoff_record,
        payload_uri="disk:/tmp/cachet-bio-1.kv",
    )

    assert entry.kv_transfer_params()[DOCUMENT_KV_PAYLOAD_URI_PARAM] == "disk:/tmp/cachet-bio-1.kv"


def test_inline_handoff_entry_builds_kv_transfer_params():
    handoff_record = inline_handoff_record(request_id="cachet-bio-1")
    enriched = enrich_benchmark_records_with_handoffs(
        [record()],
        manifest(
            BenchmarkHandoffEntry(
                dataset="biography",
                example_id="bio-1",
                request_id="cachet-bio-1",
                handoff_record=handoff_record,
            )
        ),
    )

    assert enriched[0]["kv_transfer_params"] == {
        DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
        DOCUMENT_KV_HANDOFF_RECORD_PARAM: handoff_record,
    }


def test_main_writes_enriched_jsonl(tmp_path, capsys):
    input_path = tmp_path / "bio.jsonl"
    manifest_path = tmp_path / "handoffs.json"
    output_path = tmp_path / "bio.enriched.jsonl"
    input_path.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(benchmark_handoff_manifest_to_record(manifest(entry()))),
        encoding="utf-8",
    )

    exit_code = public_benchmark_handoffs.main(
        [
            "--input-jsonl",
            str(input_path),
            "--manifest-json",
            str(manifest_path),
            "--output-jsonl",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "records": 1}
    assert load_benchmark_jsonl(output_path, dataset="biography", require_dataset=True)


def test_cachet_and_legacy_facades_share_public_module():
    assert cachet_benchmark_handoffs is public_benchmark_handoffs
    assert legacy_benchmark_handoffs.BenchmarkHandoffManifest is BenchmarkHandoffManifest
    assert legacy_benchmark_handoffs.enrich_benchmark_jsonl_with_handoffs is enrich_benchmark_jsonl_with_handoffs
