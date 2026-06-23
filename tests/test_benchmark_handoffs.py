import json

import pytest

import cachet.benchmark_handoffs as cachet_benchmark_handoffs
import document_kv_cache.benchmark_handoffs as public_benchmark_handoffs
from document_kv_cache.benchmark_handoffs import (
    BENCHMARK_HANDOFF_MANIFEST_RECORD_TYPE,
    BENCHMARK_HANDOFF_MANIFEST_SCHEMA_VERSION,
    BenchmarkHandoffBundleResult,
    BenchmarkHandoffEntry,
    BenchmarkHandoffManifest,
    build_benchmark_handoff_manifest_from_jsonl,
    benchmark_handoff_manifest_to_record,
    enrich_benchmark_jsonl_with_handoffs,
    enrich_benchmark_records_with_handoffs,
    generate_benchmark_handoff_bundles,
    load_benchmark_kv_chunk_generator,
    read_benchmark_handoff_manifest_json,
)
from document_kv_cache.benchmark_runner import load_benchmark_jsonl
from document_kv_cache.benchmarks import (
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
)
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import (
    build_engine_adapter_request,
    engine_adapter_request_to_record,
    vllm_adapter_spec,
)
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment
from document_kv_cache.kvpack import PackChunk
from document_kv_cache.model_profiles import layout_for_model
from document_kv_cache.models import KVCacheKey


class AlignedGenerator:
    def generate(self, *, document, chunk, config, training_artifacts=None):
        payload = f"{document.document_id}:{chunk.chunk_id}:{chunk.text}".encode("utf-8")
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
            ),
            payload=payload,
            token_count=len(payload),
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


GENERATOR_MODULE_SOURCE = """
from document_kv_cache.kvpack import PackChunk
from document_kv_cache.models import KVCacheKey


class Generator:
    def generate(self, *, document, chunk, config, training_artifacts=None):
        payload = chunk.text.encode("utf-8")
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
            ),
            payload=payload,
            token_count=len(payload),
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


def factory():
    return Generator()
"""

PROFILE_GENERATOR_MODULE_SOURCE = """
from document_kv_cache.kvpack import PackChunk
from document_kv_cache.models import KVCacheKey


class Generator:
    def generate(self, *, document, chunk, config, training_artifacts=None):
        payload = b"q" * 73728
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
            ),
            payload=payload,
            token_count=1,
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


def factory():
    return Generator()
"""


def write_generator_module(tmp_path, module_name):
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(GENERATOR_MODULE_SOURCE, encoding="utf-8")
    return module_path


def write_profile_generator_module(tmp_path, module_name):
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(PROFILE_GENERATOR_MODULE_SOURCE, encoding="utf-8")
    return module_path


def tiny_layout():
    return KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version="toy-one-byte-v1",
        dtype="int8",
        num_layers=1,
        block_size=8,
        bytes_per_token=1,
    )


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


def write_handoff_json(path, *, request_id="cachet-bio-1", payload_uri=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inline_handoff_record(request_id=request_id, payload_uri=payload_uri)),
        encoding="utf-8",
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
        DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM: "logical",
        DOCUMENT_KV_HANDOFF_JSON_PARAM: "/Volumes/catalog/schema/volume/cachet/cachet-bio-1.handoff.json",
        DOCUMENT_KV_PAYLOAD_URI_PARAM: "uc-volume:/catalog/schema/volume/cachet/cachet-bio-1.kv",
    }


def test_build_manifest_from_jsonl_reads_handoff_records(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(
        json.dumps(record("bio-1")) + "\n" + json.dumps(record("bio-2")) + "\n",
        encoding="utf-8",
    )
    first_handoff = tmp_path / "handoffs" / "biography" / "bio-1.handoff.json"
    second_handoff = tmp_path / "handoffs" / "biography" / "bio-2.handoff.json"
    write_handoff_json(first_handoff, request_id="cachet-bio-1")
    write_handoff_json(second_handoff, request_id="cachet-bio-2")

    handoffs = build_benchmark_handoff_manifest_from_jsonl(
        input_path,
        handoff_json_template=str(tmp_path / "handoffs" / "{dataset}" / "{example_id}.handoff.json"),
        expected_backend="vllm",
    )

    assert benchmark_handoff_manifest_to_record(handoffs)["entries"] == [
        {
            "dataset": "biography",
            "example_id": "bio-1",
            "request_id": "cachet-bio-1",
            "handoff_json": str(first_handoff),
            "payload_uri": "disk:/tmp/cachet-bio-1.kv",
        },
        {
            "dataset": "biography",
            "example_id": "bio-2",
            "request_id": "cachet-bio-2",
            "handoff_json": str(second_handoff),
            "payload_uri": "disk:/tmp/cachet-bio-2.kv",
        },
    ]


def test_build_manifest_from_jsonl_can_override_payload_uri(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(json.dumps(record("bio-1")) + "\n", encoding="utf-8")
    handoff_path = tmp_path / "handoffs" / "biography" / "bio-1.handoff.json"
    write_handoff_json(
        handoff_path,
        request_id="cachet-bio-1",
        payload_uri="s3://bucket/cachet-bio-1.kv",
    )

    handoffs = build_benchmark_handoff_manifest_from_jsonl(
        input_path,
        handoff_json_template=str(tmp_path / "handoffs" / "{dataset}" / "{example_id}.handoff.json"),
        payload_uri_template="disk:/tmp/cachet/{dataset}/{example_id}.kv",
    )

    assert handoffs.entries[0].payload_uri == "disk:/tmp/cachet/biography/bio-1.kv"


def test_build_manifest_from_jsonl_rejects_missing_handoff_by_default(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(json.dumps(record("bio-1")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing handoff JSON for biography/bio-1"):
        build_benchmark_handoff_manifest_from_jsonl(
            input_path,
            handoff_json_template=str(tmp_path / "handoffs" / "{dataset}" / "{example_id}.handoff.json"),
        )


def test_build_manifest_from_jsonl_can_skip_missing_handoffs(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(
        json.dumps(record("bio-1")) + "\n" + json.dumps(record("bio-2")) + "\n",
        encoding="utf-8",
    )
    write_handoff_json(
        tmp_path / "handoffs" / "biography" / "bio-2.handoff.json",
        request_id="cachet-bio-2",
    )

    handoffs = build_benchmark_handoff_manifest_from_jsonl(
        input_path,
        handoff_json_template=str(tmp_path / "handoffs" / "{dataset}" / "{example_id}.handoff.json"),
        allow_missing=True,
    )

    assert [entry.example_id for entry in handoffs.entries] == ["bio-2"]


def test_build_manifest_from_jsonl_rejects_backend_mismatch(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(json.dumps(record("bio-1")) + "\n", encoding="utf-8")
    write_handoff_json(tmp_path / "handoffs" / "biography" / "bio-1.handoff.json")

    with pytest.raises(ValueError, match="does not match expected_backend"):
        build_benchmark_handoff_manifest_from_jsonl(
            input_path,
            handoff_json_template=str(tmp_path / "handoffs" / "{dataset}" / "{example_id}.handoff.json"),
            expected_backend="sglang",
        )


def test_generate_benchmark_handoff_bundles_writes_payloads_manifest_and_handoffs(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(
        json.dumps(record("Bio Example/1")) + "\n" + json.dumps(record("bio-2")) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "bundle-manifest.json"
    layout = KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version="toy-one-byte-v1",
        dtype="int8",
        num_layers=1,
        block_size=8,
        bytes_per_token=1,
    )

    result = generate_benchmark_handoff_bundles(
        input_path,
        output_dir=tmp_path / "bundles",
        generator=AlignedGenerator(),
        layout=layout,
        manifest_json=manifest_path,
        align_bytes=1,
    )

    assert isinstance(result, BenchmarkHandoffBundleResult)
    assert result.cache_generation.chunk_count == 2
    assert len(result.cache_refs) == 2
    assert (tmp_path / "bundles" / "cachet-benchmark.kvpack").exists()
    assert read_benchmark_handoff_manifest_json(manifest_path).entries == result.manifest.entries
    assert [entry.example_id for entry in result.manifest.entries] == ["Bio Example/1", "bio-2"]
    for index, entry in enumerate(result.manifest.entries):
        assert entry.request_id.startswith("cachet-biography-")
        assert entry.handoff_json is not None
        assert entry.payload_uri is not None
        assert "Bio Example/1" not in entry.handoff_json
        assert "Bio Example/1" not in entry.payload_uri
        assert entry.payload_uri == result.payload_uris[index]
        handoff_path = tmp_path / "bundles" / "biography" / f"{entry.request_id}.handoff.json"
        payload_path = tmp_path / "bundles" / "biography" / f"{entry.request_id}.kv"
        assert handoff_path.exists()
        assert payload_path.exists()
        handoff_record = json.loads(handoff_path.read_text(encoding="utf-8"))
        assert handoff_record["backend"] == "vllm"
        assert handoff_record["request_id"] == entry.request_id
        assert handoff_record["payload_source"]["uri"] == str(payload_path)
        assert handoff_record["handle"]["layout"]["bytes_per_token"] == 1
        assert handoff_record["handle"]["total_bytes"] == payload_path.stat().st_size


def test_generate_benchmark_handoff_bundles_can_emit_sglang_records(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(json.dumps(record("bio-1")) + "\n", encoding="utf-8")
    layout = KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version="toy-one-byte-v1",
        dtype="int8",
        num_layers=1,
        block_size=8,
        bytes_per_token=1,
    )

    result = generate_benchmark_handoff_bundles(
        input_path,
        output_dir=tmp_path / "bundles",
        generator=AlignedGenerator(),
        layout=layout,
        backend="sglang",
        align_bytes=1,
    )

    handoff_path = tmp_path / "bundles" / "biography" / f"{result.manifest.entries[0].request_id}.handoff.json"
    handoff_record = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert handoff_record["backend"] == "sglang"
    assert handoff_record["connector_package"] == "sglang"


def test_generate_benchmark_handoff_bundles_rejects_duplicate_input_rows(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(
        json.dumps(record("bio-1")) + "\n" + json.dumps(record("bio-1")) + "\n",
        encoding="utf-8",
    )
    layout = KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version="toy-one-byte-v1",
        dtype="int8",
        num_layers=1,
        block_size=8,
        bytes_per_token=1,
    )

    with pytest.raises(ValueError, match="Duplicate benchmark input rows for biography/bio-1"):
        generate_benchmark_handoff_bundles(
            input_path,
            output_dir=tmp_path / "bundles",
            generator=AlignedGenerator(),
            layout=layout,
            align_bytes=1,
        )


def test_generate_benchmark_handoff_bundles_rejects_duplicate_handoff_paths_before_writes(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(
        json.dumps(record("bio-1")) + "\n" + json.dumps(record("bio-2")) + "\n",
        encoding="utf-8",
    )
    handoff_path = tmp_path / "same.handoff.json"

    with pytest.raises(ValueError, match="handoff_json for biography/bio-2 collides"):
        generate_benchmark_handoff_bundles(
            input_path,
            output_dir=tmp_path / "bundles",
            generator=AlignedGenerator(),
            layout=tiny_layout(),
            handoff_json_template=str(handoff_path),
            align_bytes=1,
        )

    assert not handoff_path.exists()
    assert not (tmp_path / "bundles" / "cachet-benchmark.kvpack").exists()


def test_generate_benchmark_handoff_bundles_rejects_duplicate_payload_paths_before_writes(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(
        json.dumps(record("bio-1")) + "\n" + json.dumps(record("bio-2")) + "\n",
        encoding="utf-8",
    )
    payload_path = tmp_path / "same.kv"

    with pytest.raises(ValueError, match="payload_uri for biography/bio-2 collides"):
        generate_benchmark_handoff_bundles(
            input_path,
            output_dir=tmp_path / "bundles",
            generator=AlignedGenerator(),
            layout=tiny_layout(),
            payload_uri_template=str(payload_path),
            align_bytes=1,
        )

    assert not payload_path.exists()
    assert not (tmp_path / "bundles" / "cachet-benchmark.kvpack").exists()


def test_generate_benchmark_handoff_bundles_rejects_handoff_payload_manifest_and_shard_collisions(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(json.dumps(record("bio-1")) + "\n", encoding="utf-8")
    collision_path = tmp_path / "collision"

    with pytest.raises(ValueError, match="payload_uri for biography/bio-1 collides with handoff_json"):
        generate_benchmark_handoff_bundles(
            input_path,
            output_dir=tmp_path / "bundles",
            generator=AlignedGenerator(),
            layout=tiny_layout(),
            handoff_json_template=str(collision_path),
            payload_uri_template=str(collision_path),
            align_bytes=1,
        )
    with pytest.raises(ValueError, match="handoff_json for biography/bio-1 collides with manifest_json"):
        generate_benchmark_handoff_bundles(
            input_path,
            output_dir=tmp_path / "bundles",
            generator=AlignedGenerator(),
            layout=tiny_layout(),
            manifest_json=collision_path,
            handoff_json_template=str(collision_path),
            align_bytes=1,
        )
    with pytest.raises(ValueError, match="handoff_json for biography/bio-1 collides with shard_uri"):
        generate_benchmark_handoff_bundles(
            input_path,
            output_dir=tmp_path / "bundles",
            generator=AlignedGenerator(),
            layout=tiny_layout(),
            shard_uri=collision_path,
            handoff_json_template=str(collision_path),
            align_bytes=1,
        )
    with pytest.raises(ValueError, match="shard_uri collides with input_jsonl"):
        generate_benchmark_handoff_bundles(
            input_path,
            output_dir=tmp_path / "bundles",
            generator=AlignedGenerator(),
            layout=tiny_layout(),
            shard_uri=input_path,
            align_bytes=1,
        )

    assert not collision_path.exists()


def test_generate_benchmark_handoff_bundles_requires_generator_and_matching_model(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(json.dumps(record("bio-1")) + "\n", encoding="utf-8")
    layout = KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version="toy-one-byte-v1",
        dtype="int8",
        num_layers=1,
        block_size=8,
        bytes_per_token=1,
    )

    with pytest.raises(TypeError, match="generator must implement KVChunkGenerator.generate"):
        generate_benchmark_handoff_bundles(
            input_path,
            output_dir=tmp_path / "bundles",
            generator=object(),
            layout=layout,
        )
    with pytest.raises(ValueError, match="model_id must match layout.model_id"):
        generate_benchmark_handoff_bundles(
            input_path,
            output_dir=tmp_path / "bundles",
            generator=AlignedGenerator(),
            layout=layout,
            model_id="other-model",
        )


def test_load_benchmark_kv_chunk_generator_accepts_zero_arg_factory(tmp_path, monkeypatch):
    write_generator_module(tmp_path, "fixture_generator")
    monkeypatch.syspath_prepend(str(tmp_path))

    generator = load_benchmark_kv_chunk_generator("fixture_generator:factory")

    assert generator.__class__.__name__ == "Generator"


def test_load_benchmark_kv_chunk_generator_accepts_generator_class(tmp_path, monkeypatch):
    write_generator_module(tmp_path, "fixture_generator_class")
    monkeypatch.syspath_prepend(str(tmp_path))

    generator = load_benchmark_kv_chunk_generator("fixture_generator_class:Generator")

    assert generator.__class__.__name__ == "Generator"


def test_bundle_main_writes_manifest_from_generator_factory(tmp_path, monkeypatch, capsys):
    write_generator_module(tmp_path, "cli_generator")
    monkeypatch.syspath_prepend(str(tmp_path))
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(json.dumps(record("bio-1")) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "handoffs.json"

    exit_code = public_benchmark_handoffs.bundle_main(
        [
            "--input-jsonl",
            str(input_path),
            "--output-dir",
            str(tmp_path / "bundles"),
            "--output-manifest-json",
            str(manifest_path),
            "--generator-factory",
            "cli_generator:factory",
            "--layout-version",
            "toy-one-byte-v1",
            "--dtype",
            "int8",
            "--num-layers",
            "1",
            "--block-size",
            "8",
            "--bytes-per-token",
            "1",
            "--align-bytes",
            "1",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["ok"] is True
    assert output["entries"] == 1
    assert output["cache_refs"] == 1
    assert output["manifest_json"] == str(manifest_path)
    assert read_benchmark_handoff_manifest_json(manifest_path).entries[0].example_id == "bio-1"


def test_bundle_main_defaults_to_builtin_model_profile_layout(tmp_path, monkeypatch, capsys):
    write_profile_generator_module(tmp_path, "profile_cli_generator")
    monkeypatch.syspath_prepend(str(tmp_path))
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(json.dumps(record("bio-1")) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "handoffs.json"

    exit_code = public_benchmark_handoffs.bundle_main(
        [
            "--input-jsonl",
            str(input_path),
            "--output-dir",
            str(tmp_path / "bundles"),
            "--output-manifest-json",
            str(manifest_path),
            "--generator-factory",
            "profile_cli_generator:factory",
            "--align-bytes",
            "1",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    entry = read_benchmark_handoff_manifest_json(manifest_path).entries[0]
    handoff_path = tmp_path / "bundles" / "biography" / f"{entry.request_id}.handoff.json"
    handoff_record = json.loads(handoff_path.read_text(encoding="utf-8"))
    expected_layout = layout_for_model("qwen3:4b-instruct")

    assert exit_code == 0
    assert output["ok"] is True
    assert handoff_record["handle"]["layout"]["layout_version"] == expected_layout.layout_version
    assert handoff_record["handle"]["layout"]["num_layers"] == expected_layout.num_layers
    assert handoff_record["handle"]["layout"]["num_query_heads"] == expected_layout.num_query_heads
    assert handoff_record["handle"]["layout"]["num_kv_heads"] == expected_layout.num_kv_heads
    assert handoff_record["handle"]["layout"]["bytes_per_token"] == expected_layout.bytes_per_token
    assert handoff_record["handle"]["layout"]["storage_layout"] == "shared_key_value"
    assert handoff_record["handle"]["total_tokens"] == 1
    assert handoff_record["handle"]["total_bytes"] == expected_layout.bytes_per_token


def test_bundle_main_rejects_partial_manual_layout(tmp_path, monkeypatch, capsys):
    write_generator_module(tmp_path, "partial_cli_generator")
    monkeypatch.syspath_prepend(str(tmp_path))
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(json.dumps(record("bio-1")) + "\n", encoding="utf-8")

    exit_code = public_benchmark_handoffs.bundle_main(
        [
            "--input-jsonl",
            str(input_path),
            "--output-dir",
            str(tmp_path / "bundles"),
            "--output-manifest-json",
            str(tmp_path / "handoffs.json"),
            "--generator-factory",
            "partial_cli_generator:factory",
            "--num-layers",
            "1",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["error_type"] == "ValueError"
    assert "Manual benchmark layout requires" in output["error"]


@pytest.mark.parametrize(
    ("template", "error"),
    (
        ("{dataset[0]}/{example_id}.handoff.json", "supports only"),
        ("{dataset.__class__}/{example_id}.handoff.json", "supports only"),
        ("{dataset!r}/{example_id}.handoff.json", "plain"),
        ("{dataset:>10}/{example_id}.handoff.json", "plain"),
    ),
)
def test_build_manifest_from_jsonl_rejects_non_plain_template_fields(tmp_path, template, error):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(json.dumps(record("bio-1")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        build_benchmark_handoff_manifest_from_jsonl(
            input_path,
            handoff_json_template=template,
        )


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
        DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM: "logical",
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


def test_manifest_main_writes_manifest_json(tmp_path, capsys):
    input_path = tmp_path / "bio.jsonl"
    output_path = tmp_path / "handoffs.json"
    handoff_path = tmp_path / "handoffs" / "biography" / "bio-1.handoff.json"
    input_path.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    write_handoff_json(handoff_path)

    exit_code = public_benchmark_handoffs.manifest_main(
        [
            "--input-jsonl",
            str(input_path),
            "--handoff-json-template",
            str(tmp_path / "handoffs" / "{dataset}" / "{example_id}.handoff.json"),
            "--expected-backend",
            "vllm",
            "--output-json",
            str(output_path),
        ]
    )

    handoffs = read_benchmark_handoff_manifest_json(output_path)

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "entries": 1}
    assert handoffs.entries[0].handoff_json == str(handoff_path)
