import json
import shutil
from hashlib import sha256
from pathlib import Path

import pytest

import document_kv_cache.publication_handoff_artifacts as publication_artifacts
from document_kv_cache.artifact_identity import TokenContract
from document_kv_cache.benchmark_handoffs import (
    enrich_benchmark_records_with_handoffs,
    generate_benchmark_handoff_bundles,
)
from document_kv_cache.benchmarks import (
    DOCUMENT_KV_ARTIFACT_ID_PARAM,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    SUPPORTED_V1_DATASETS,
)
from document_kv_cache.engine_protocol import KVLayout
from document_kv_cache.kvpack import PackChunk
from document_kv_cache.models import CacheGenerationMethod, KVCacheKey
from document_kv_cache.publication_handoff_artifacts import (
    PUBLICATION_HANDOFF_BUNDLE_RECORD_TYPE,
    PUBLICATION_HANDOFF_STAGING_ATTESTATION_FILENAME,
    PUBLICATION_HANDOFF_STAGING_ATTESTATION_RECORD_TYPE,
    close_publication_latency_handoff_bundle,
    read_publication_latency_handoff_bundle,
    stage_publication_latency_handoff_bundle,
    validate_publication_latency_handoff_bundle,
    verify_staged_publication_latency_handoff_bundle,
    write_publication_latency_handoff_bundle,
)


class StrictPublicationGenerator:
    pre_rope = True
    rope_theta = 5_000_000.0
    rope_rotary_dim = 4
    add_special_tokens = False

    def __init__(self, layout):
        self.layout = layout

    def bind_layout(self, layout):
        self.layout = layout

    def generate(self, *, document, chunk, config, training_artifacts=None):
        del training_artifacts
        token_ids = tuple(chunk.text.encode("utf-8")) or (0,)
        payload = b"q" * (len(token_ids) * self.layout.bytes_per_token)
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
                content_hash=sha256(payload).hexdigest(),
                artifact_identity=config.artifact_identity_for(self.layout),
                token_contract=TokenContract.from_token_ids(
                    token_ids,
                    tokenizer_id=config.tokenizer_id,
                    tokenizer_revision=config.tokenizer_revision,
                    add_special_tokens=self.add_special_tokens,
                    prompt_template_version=config.prompt_template_version,
                ),
            ),
            payload=payload,
            token_count=len(token_ids),
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


def publication_layout():
    return KVLayout(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        lora_id="base",
        layout_version="qwen3-publication-test-v1",
        dtype="fp8_e5m2",
        num_layers=1,
        block_size=8,
        bytes_per_token=16,
        num_query_heads=1,
        num_kv_heads=1,
        head_size=4,
        kv_stride_bytes=8,
        storage_layout="separate_key_value",
        pre_rope=True,
        rope_theta=5_000_000.0,
        rope_rotary_dim=4,
    )


def source_record(dataset, *, segment_count=4):
    return {
        "dataset": dataset,
        "documents": [
            {
                "document_id": f"{dataset}-opaque-{index}",
                "metadata": {
                    "cachet.main_latency.segment_count": str(segment_count),
                    "cachet.main_latency.segment_index": str(index),
                    "cachet.main_latency.transformation": (
                        "cachet.main_latency.lossless_context_tiling.v1"
                    ),
                },
                "text": f"Café evidence {index} for the {dataset} publication example.",
            }
            for index in range(segment_count)
        ],
        "example_id": f"{dataset}-example-1",
        "expected_answer": "answer",
        "query": "What is the answer?",
    }


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def make_bundle(tmp_path, *, name="durable", context_tokens=8192):
    root = tmp_path / name
    source_path = tmp_path / f"{name}-source.jsonl"
    segment_count = {8192: 4, 16384: 8, 32768: 16}[context_tokens]
    records = [
        source_record(dataset, segment_count=segment_count)
        for dataset in SUPPORTED_V1_DATASETS
    ]
    write_jsonl(source_path, records)
    result = generate_benchmark_handoff_bundles(
        source_path,
        output_dir=root,
        generator=StrictPublicationGenerator(publication_layout()),
        layout=publication_layout(),
        cache_method=CacheGenerationMethod.VANILLA_PREFILL,
        segment_per_document=True,
        model_revision="model-revision-pinned",
        tokenizer_id="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_revision="tokenizer-revision-pinned",
        generator_version="generator-version-pinned",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        align_bytes=1,
    )
    (root / "cachet-benchmark.kvpack").unlink()
    dataset_paths = {}
    for record in records:
        dataset = record["dataset"]
        enriched = enrich_benchmark_records_with_handoffs(
            (record,),
            result.manifest,
            dataset=dataset,
            arm_id="vanilla",
            allow_unmatched=True,
        )
        path = root / "datasets" / f"{dataset}.jsonl"
        write_jsonl(path, enriched)
        dataset_paths[dataset] = path
    return root, dataset_paths


def close_bundle(root, dataset_paths, *, context_tokens=8192):
    return close_publication_latency_handoff_bundle(
        root,
        dataset_paths,
        context_tokens=context_tokens,
        input_bundle_sha256=sha256(b"main-latency-input-bundle").hexdigest(),
    )


@pytest.mark.parametrize("context_tokens", [8192, 16384, 32768])
def test_close_and_stage_publication_handoffs(context_tokens, tmp_path):
    root, dataset_paths = make_bundle(tmp_path, context_tokens=context_tokens)

    first = close_bundle(root, dataset_paths, context_tokens=context_tokens)
    second = close_bundle(root, dataset_paths, context_tokens=context_tokens)

    assert first == second
    assert first["record_type"] == PUBLICATION_HANDOFF_BUNDLE_RECORD_TYPE
    assert first["context_tokens"] == context_tokens
    assert str(root) not in json.dumps(first, sort_keys=True)
    assert len(first["datasets"]) == len(SUPPORTED_V1_DATASETS)
    identity = first["identity"]
    assert identity["topology_identity"] == {
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
    }
    assert identity["model_identity"]["model_revision"] == "model-revision-pinned"
    assert identity["tokenizer_identity"]["tokenizer_revision"] == (
        "tokenizer-revision-pinned"
    )
    assert identity["layout_identity"]["dtype"] == "fp8_e5m2"

    manifest_path = tmp_path / "reviewed-manifest.json"
    write_publication_latency_handoff_bundle(first, manifest_path)
    loaded = read_publication_latency_handoff_bundle(manifest_path)
    assert loaded == first

    stage_root = tmp_path / "nvme" / f"{context_tokens}"
    staged = stage_publication_latency_handoff_bundle(
        loaded,
        source_root=root,
        local_nvme_dir=stage_root,
    )
    verified = verify_staged_publication_latency_handoff_bundle(
        loaded,
        staged_root=stage_root,
    )

    assert verified.dataset_paths == staged.dataset_paths
    assert staged.attestation["record_type"] == (
        PUBLICATION_HANDOFF_STAGING_ATTESTATION_RECORD_TYPE
    )
    assert staged.attestation_path.name == (
        PUBLICATION_HANDOFF_STAGING_ATTESTATION_FILENAME
    )
    for dataset, staged_path in staged.dataset_paths.items():
        row = json.loads(staged_path.read_text(encoding="utf-8"))
        params = row["arm_kv_transfer_params"]["vanilla"]
        assert params[DOCUMENT_KV_HANDOFF_JSON_PARAM].startswith(str(stage_root))
        assert params[DOCUMENT_KV_PAYLOAD_URI_PARAM].startswith(str(stage_root))
        assert params[DOCUMENT_KV_REQUEST_ID_PARAM]
        assert params[DOCUMENT_KV_ARTIFACT_ID_PARAM] == identity["artifact_id"]
        source_row = json.loads(dataset_paths[dataset].read_text(encoding="utf-8"))
        source_params = source_row["arm_kv_transfer_params"]["vanilla"]
        assert {
            key: value
            for key, value in params.items()
            if key
            not in {DOCUMENT_KV_HANDOFF_JSON_PARAM, DOCUMENT_KV_PAYLOAD_URI_PARAM}
        } == {
            key: value
            for key, value in source_params.items()
            if key
            not in {DOCUMENT_KV_HANDOFF_JSON_PARAM, DOCUMENT_KV_PAYLOAD_URI_PARAM}
        }


def test_portable_identity_is_independent_of_source_root(tmp_path):
    first_root, first_paths = make_bundle(tmp_path, name="first")
    second_root, second_paths = make_bundle(tmp_path, name="second")

    first = close_bundle(first_root, first_paths)
    second = close_bundle(second_root, second_paths)

    assert first["closed_record_sha256"] != second["closed_record_sha256"]
    assert first["portable_bundle_sha256"] == second["portable_bundle_sha256"]


def test_closed_durable_bundle_can_move_before_staging(tmp_path):
    root, dataset_paths = make_bundle(tmp_path)
    manifest = close_bundle(root, dataset_paths)
    moved = tmp_path / "reviewed" / "moved-durable"
    moved.parent.mkdir()
    root.rename(moved)

    validate_publication_latency_handoff_bundle(manifest, bundle_root=moved)
    staged = stage_publication_latency_handoff_bundle(
        manifest,
        source_root=moved,
        local_nvme_dir=tmp_path / "moved-stage",
    )

    assert set(staged.dataset_paths) == set(SUPPORTED_V1_DATASETS)


def test_close_rejects_context_label_that_does_not_match_segment_topology(tmp_path):
    root, dataset_paths = make_bundle(tmp_path, context_tokens=8192)

    with pytest.raises(ValueError, match="16384 row must contain exactly 8"):
        close_bundle(root, dataset_paths, context_tokens=16384)


def test_close_rejects_extra_missing_duplicate_and_tampered_files(tmp_path):
    root, dataset_paths = make_bundle(tmp_path)
    (root / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="file closure mismatch"):
        close_bundle(root, dataset_paths)
    (root / "extra.txt").unlink()

    manifest = close_bundle(root, dataset_paths)
    payload = next(item for item in manifest["files"] if item["role"] == "payload")
    payload_path = root / payload["relative_name"]
    payload_path.write_bytes(payload_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="total_bytes|checksum"):
        validate_publication_latency_handoff_bundle(manifest, bundle_root=root)
    payload_path.write_bytes(payload_path.read_bytes()[:-8])

    missing = next(item for item in manifest["files"] if item["role"] == "handoff_json")
    (root / missing["relative_name"]).unlink()
    with pytest.raises(ValueError, match="missing handoff JSON"):
        validate_publication_latency_handoff_bundle(manifest, bundle_root=root)


def test_close_rejects_duplicate_reference_and_symlink(tmp_path):
    root, dataset_paths = make_bundle(tmp_path)
    biography = json.loads(dataset_paths["biography"].read_text(encoding="utf-8"))
    hotpot = json.loads(dataset_paths["hotpotqa"].read_text(encoding="utf-8"))
    hotpot_params = hotpot["arm_kv_transfer_params"]["vanilla"]
    biography_params = biography["arm_kv_transfer_params"]["vanilla"]
    original_handoff = hotpot_params[DOCUMENT_KV_HANDOFF_JSON_PARAM]
    hotpot_params[DOCUMENT_KV_HANDOFF_JSON_PARAM] = biography_params[
        DOCUMENT_KV_HANDOFF_JSON_PARAM
    ]
    write_jsonl(dataset_paths["hotpotqa"], (hotpot,))
    with pytest.raises(ValueError, match="payload URI does not match|request_id does not match|duplicate"):
        close_bundle(root, dataset_paths)
    hotpot_params[DOCUMENT_KV_HANDOFF_JSON_PARAM] = original_handoff
    write_jsonl(dataset_paths["hotpotqa"], (hotpot,))

    payload = Path(hotpot_params[DOCUMENT_KV_PAYLOAD_URI_PARAM])
    real_payload = payload.with_name(payload.name + ".real")
    payload.rename(real_payload)
    payload.symlink_to(real_payload)
    with pytest.raises(ValueError, match="symlink"):
        close_bundle(root, dataset_paths)


def test_close_and_manifest_validation_reject_traversal_and_unknown_keys(tmp_path):
    root, dataset_paths = make_bundle(tmp_path)
    outside = tmp_path / "outside.jsonl"
    shutil.copyfile(dataset_paths["biography"], outside)
    escaped = dict(dataset_paths)
    escaped["biography"] = outside
    with pytest.raises(ValueError, match="inside bundle_root"):
        close_bundle(root, escaped)

    biography = json.loads(dataset_paths["biography"].read_text(encoding="utf-8"))
    biography["arm_kv_transfer_params"]["vanilla"][
        DOCUMENT_KV_HANDOFF_JSON_PARAM
    ] = "../outside.json"
    write_jsonl(dataset_paths["biography"], (biography,))
    with pytest.raises(ValueError, match="traversal"):
        close_bundle(root, dataset_paths)

    root, dataset_paths = make_bundle(tmp_path, name="closed-record")

    manifest = close_bundle(root, dataset_paths)
    manifest["unexpected"] = True
    with pytest.raises(ValueError, match="keys are not closed"):
        validate_publication_latency_handoff_bundle(manifest, bundle_root=root)


def test_stage_rejects_collision_overlap_tampering_and_leaves_no_partial_target(
    tmp_path,
    monkeypatch,
):
    root, dataset_paths = make_bundle(tmp_path)
    manifest = close_bundle(root, dataset_paths)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        stage_publication_latency_handoff_bundle(
            manifest,
            source_root=root,
            local_nvme_dir=existing,
        )
    with pytest.raises(ValueError, match="overlap"):
        stage_publication_latency_handoff_bundle(
            manifest,
            source_root=root,
            local_nvme_dir=root / "nested-stage",
        )

    payload = next(item for item in manifest["files"] if item["role"] == "payload")
    payload_path = root / payload["relative_name"]
    original = payload_path.read_bytes()
    payload_path.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="total_bytes|checksum"):
        stage_publication_latency_handoff_bundle(
            manifest,
            source_root=root,
            local_nvme_dir=tmp_path / "tampered-stage",
        )
    payload_path.write_bytes(original)

    target = tmp_path / "failed-stage"
    calls = 0
    real_copyfile = publication_artifacts.shutil.copyfile

    def fail_second_copy(source, destination, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected copy failure")
        return real_copyfile(source, destination, **kwargs)

    monkeypatch.setattr(publication_artifacts.shutil, "copyfile", fail_second_copy)
    with pytest.raises(OSError, match="injected"):
        stage_publication_latency_handoff_bundle(
            manifest,
            source_root=root,
            local_nvme_dir=target,
        )
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.staging-*"))

    monkeypatch.setattr(publication_artifacts.shutil, "copyfile", real_copyfile)
    post_publish_target = tmp_path / "failed-post-publish-stage"

    def fail_post_publish_verification(*args, **kwargs):
        del args, kwargs
        raise ValueError("injected post-publish verification failure")

    monkeypatch.setattr(
        publication_artifacts,
        "verify_staged_publication_latency_handoff_bundle",
        fail_post_publish_verification,
    )
    with pytest.raises(ValueError, match="post-publish"):
        stage_publication_latency_handoff_bundle(
            manifest,
            source_root=root,
            local_nvme_dir=post_publish_target,
        )
    assert not post_publish_target.exists()


def test_stage_attestation_rejects_post_stage_tampering(tmp_path):
    root, dataset_paths = make_bundle(tmp_path)
    manifest = close_bundle(root, dataset_paths)
    stage_root = tmp_path / "stage"
    staged = stage_publication_latency_handoff_bundle(
        manifest,
        source_root=root,
        local_nvme_dir=stage_root,
    )
    staged.dataset_paths["musique"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="byte count mismatch|SHA-256 mismatch"):
        verify_staged_publication_latency_handoff_bundle(
            manifest,
            staged_root=stage_root,
        )


def test_stage_attestation_is_closed_against_field_tampering(tmp_path):
    root, dataset_paths = make_bundle(tmp_path)
    manifest = close_bundle(root, dataset_paths)
    stage_root = tmp_path / "stage"
    staged = stage_publication_latency_handoff_bundle(
        manifest,
        source_root=root,
        local_nvme_dir=stage_root,
    )
    attestation = json.loads(staged.attestation_path.read_text(encoding="utf-8"))
    attestation["files"][0]["source_sha256"] = sha256(b"wrong").hexdigest()
    staged.attestation_path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="closed_record_sha256"):
        verify_staged_publication_latency_handoff_bundle(
            manifest,
            staged_root=stage_root,
        )


def test_manifest_writer_refuses_overwrite(tmp_path):
    root, dataset_paths = make_bundle(tmp_path)
    manifest = close_bundle(root, dataset_paths)
    path = tmp_path / "manifest.json"
    write_publication_latency_handoff_bundle(manifest, path)

    with pytest.raises(FileExistsError, match="overwrite"):
        write_publication_latency_handoff_bundle(manifest, path)


def test_payload_verification_is_streaming_not_read_bytes(tmp_path, monkeypatch):
    root, dataset_paths = make_bundle(tmp_path)
    real_read_bytes = Path.read_bytes

    def reject_payload_read_bytes(path):
        if path.suffix == ".kv":
            raise AssertionError("KV payload must be hashed as a stream")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_payload_read_bytes)
    manifest = close_bundle(root, dataset_paths)
    staged = stage_publication_latency_handoff_bundle(
        manifest,
        source_root=root,
        local_nvme_dir=tmp_path / "streaming-stage",
    )

    assert staged.attestation_path.exists()
