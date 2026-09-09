"""Synthetic bytes exercise real handoff schemas; no model/GPU execution."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from document_kv_cache.artifact_identity import TokenContract
from document_kv_cache.benchmark_handoffs import (
    enrich_benchmark_records_with_handoffs,
    generate_benchmark_handoff_bundles,
)
from document_kv_cache.benchmarks import (
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
)
from document_kv_cache.canary_orchestration import (
    BASELINE_PREFILL_ARM,
    FULL_PREFIX_CANARY_ARM,
    REPRESENTATIVE_CANARY_MODEL_ID,
    REPRESENTATIVE_CANARY_MODEL_REVISION,
    VANILLA_CANARY_ARM,
)
from document_kv_cache.engine_adapters import read_engine_adapter_request_json
from document_kv_cache.engine_protocol import KVLayout
from document_kv_cache.kvpack import PackChunk
from document_kv_cache.model_profiles import QWEN3_4B_ROPE_ROTARY_DIM, QWEN3_4B_ROPE_THETA
from document_kv_cache.models import KVCacheKey
import document_kv_cache.representative_handoff_artifacts as artifacts
from document_kv_cache.representative_handoff_artifacts import (
    RepresentativeHandoffBindings,
    close_representative_handoff_bundle,
    stage_representative_handoff_bundle,
    validate_representative_handoff_bundle,
    verify_staged_representative_handoff_bundle,
)
from vllm_kv_injection.vllm_native_provider import KVTransferParamsDocumentKVSource


class FixtureGenerator:
    add_special_tokens = False

    def __init__(self, layout, pre_rope):
        self.layout = layout
        self.pre_rope = pre_rope
        self.rope_theta = QWEN3_4B_ROPE_THETA if pre_rope else None
        self.rope_rotary_dim = QWEN3_4B_ROPE_ROTARY_DIM if pre_rope else None

    def bind_layout(self, layout):
        self.layout = layout

    def generate(self, *, document, chunk, config, training_artifacts=None):
        tokens = tuple(chunk.text.encode()) or (0,)
        payload = b"x" * (len(tokens) * self.layout.bytes_per_token)
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id, lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id, chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id, content_hash=sha256(payload).hexdigest(),
                artifact_identity=config.artifact_identity_for(self.layout),
                token_contract=TokenContract.from_token_ids(
                    tokens, tokenizer_id=config.tokenizer_id,
                    tokenizer_revision=config.tokenizer_revision, add_special_tokens=False,
                    prompt_template_version=config.prompt_template_version,
                ),
            ),
            payload=payload, token_count=len(tokens), dtype=config.dtype,
            layout_version=config.layout_version, storage_layout=config.storage_layout,
        )


def bindings():
    return RepresentativeHandoffBindings(
        source_commit="a" * 40, package_wheel_sha256="b" * 64,
        native_runtime_closure_sha256="c" * 64, prepared_input_bundle_sha256="d" * 64,
    )


def test_public_cachet_facade_preserves_module_and_export_identity():
    import cachet
    import cachet.representative_handoff_artifacts as facade
    import document_kv_cache

    assert facade is artifacts
    assert cachet.representative_handoff_artifacts is artifacts
    assert document_kv_cache.representative_handoff_artifacts is artifacts
    assert "representative_handoff_artifacts" in dir(cachet)
    assert all(getattr(facade, name) is getattr(artifacts, name) for name in artifacts.__all__)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def token_counter(text):
    # A fixture tokenizer with registered context totals, not benchmark evidence.
    return 16384 if "context-16384" in text else 8192


def make_source(tmp_path, *, contexts=(8192, 16384), dtype="bfloat16", example_count=2, lora_id="base"):
    root = tmp_path / "source"
    root.mkdir()
    datasets = {}
    layout = KVLayout(
        model_id=REPRESENTATIVE_CANARY_MODEL_ID, lora_id=lora_id,
        layout_version="representative-test-v1", dtype=dtype, num_layers=1,
        block_size=8, bytes_per_token=512 if dtype == "bfloat16" else 256,
        num_query_heads=1, num_kv_heads=1, head_size=128,
        kv_stride_bytes=256 if dtype == "bfloat16" else 128,
        storage_layout="separate_key_value",
    )
    for context in contexts:
        rows = [
            {"dataset": "hotpotqa", "example_id": f"example-{i}",
             "documents": [
                 {"document_id": f"doc-{i}-{j}", "text": f"context-{context} evidence {i} {j}."}
                 for j in range(2)
             ], "query": "What?", "expected_answer": "answer"}
            for i in range(example_count)
        ]
        source = root / "generation-input.jsonl"
        write_rows(source, rows)
        combined = rows
        for arm, method, pre_rope in (
            (FULL_PREFIX_CANARY_ARM, "full_prefix_prefill", False),
            (VANILLA_CANARY_ARM, "vanilla_prefill", True),
        ):
            output = root / str(context) / method
            generated = generate_benchmark_handoff_bundles(
                source, output_dir=output, generator=FixtureGenerator(layout, pre_rope),
                layout=layout, cache_method=method, segment_per_document=pre_rope,
                segmented=pre_rope, model_revision=REPRESENTATIVE_CANARY_MODEL_REVISION,
                tokenizer_id=REPRESENTATIVE_CANARY_MODEL_ID,
                tokenizer_revision=REPRESENTATIVE_CANARY_MODEL_REVISION,
                generator_version="e" * 40, align_bytes=4096,
                lora_id=lora_id,
                prefix=f"representative-{context}-{method}",
            )
            (output / "cachet-benchmark.kvpack").unlink()
            combined = enrich_benchmark_records_with_handoffs(
                combined, generated.manifest, dataset="hotpotqa", arm_id=arm,
            )
        source.unlink()
        datasets[context] = root / str(context) / "hotpotqa.jsonl"
        write_rows(datasets[context], combined)
    return root, datasets


def close(root, datasets):
    return close_representative_handoff_bundle(
        root, datasets, bindings=bindings(), token_counter=token_counter, example_count=2,
    )


def validate(record, root, **kwargs):
    return validate_representative_handoff_bundle(
        record, bundle_root=root, expected_bindings=kwargs.get("expected_bindings", bindings()),
        expected_bundle_sha256=kwargs.get("expected_bundle_sha256", record["closed_record_sha256"]),
    )


def stage(record, root, target):
    return stage_representative_handoff_bundle(
        record, source_root=root, local_nvme_dir=target, expected_bindings=bindings(),
        expected_bundle_sha256=record["closed_record_sha256"],
    )


def verify(record, target):
    return verify_staged_representative_handoff_bundle(
        record, staged_root=target, expected_bindings=bindings(),
        expected_bundle_sha256=record["closed_record_sha256"],
    )


def test_one_set_preserves_exact_artifacts_and_request_identity_across_five_roots(tmp_path, monkeypatch):
    root, datasets = make_source(tmp_path)
    manifest = close(root, datasets)
    assert manifest == close(root, datasets)
    assert str(root) not in json.dumps(manifest)
    assert len(manifest["contexts"]) == 2
    for context in manifest["contexts"]:
        methods = context["methods"]
        assert methods[FULL_PREFIX_CANARY_ARM]["artifact_identity"]["key_position_encoding"] == "stored_post_rope"
        assert methods[VANILLA_CANARY_ARM]["artifact_identity"]["key_position_encoding"] == "pre_rope"
    original = {f["relative_name"]: (root / f["relative_name"]).read_bytes() for f in manifest["files"]}

    def no_generation(*args, **kwargs):
        raise AssertionError("serving deployment regenerated KV")

    monkeypatch.setattr(FixtureGenerator, "generate", no_generation)
    projections = []
    for group in range(1, 6):
        staged = stage(manifest, root, tmp_path / f"deployment-{group}")
        assert verify(manifest, staged.root).attestation == staged.attestation
        assert len(staged.dataset_paths) == 6
        assert staged.attestation["portable_identity_sha256"] == manifest["portable_identity_sha256"]
        assert all((staged.root / name).read_bytes() == value for name, value in original.items())
        for (context, arm), path in staged.dataset_paths.items():
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                assert "arm_kv_transfer_params" not in row
                if arm == BASELINE_PREFILL_ARM:
                    assert "kv_transfer_params" not in row
                else:
                    params = row["kv_transfer_params"]
                    handoff_path = Path(params[DOCUMENT_KV_HANDOFF_JSON_PARAM])
                    payload_path = Path(params[DOCUMENT_KV_PAYLOAD_URI_PARAM])
                    assert handoff_path.is_relative_to(staged.root)
                    assert payload_path.is_relative_to(staged.root)
                    handoff = read_engine_adapter_request_json(handoff_path, expected_backend="vllm")
                    assert sha256(payload_path.read_bytes()).hexdigest() == handoff["handle"]["payload_checksum"]
                    assert handoff["payload_source"]["uri"].startswith(str(root))
                    load = KVTransferParamsDocumentKVSource().get_load(
                        SimpleNamespace(request_id=f"deployment-{group}", kv_transfer_params=params),
                    )
                    assert load is not None
                    assert load.payload_uri == str(payload_path)
                    assert load.request_id == handoff["request_id"]
            assert f"context-{context}" in path.read_text()
        projections.append(staged.attestation["projections"])
    assert projections[0] != projections[1]  # Only local projections carry deployment paths.


def test_closed_source_can_move_before_consumption(tmp_path):
    root, datasets = make_source(tmp_path, contexts=(8192,))
    manifest = close(root, datasets)
    relocated = tmp_path / "relocated"
    root.rename(relocated)
    assert validate(manifest, relocated) == manifest
    assert stage(manifest, relocated, tmp_path / "node").attestation["bundle_closed_record_sha256"] == manifest["closed_record_sha256"]


@pytest.mark.parametrize("field", ("source_commit", "package_wheel_sha256", "native_runtime_closure_sha256", "prepared_input_bundle_sha256"))
def test_rejects_different_frozen_bindings(tmp_path, field):
    root, datasets = make_source(tmp_path, contexts=(8192,))
    manifest = close(root, datasets)
    different = replace(bindings(), **{field: "f" * (40 if field == "source_commit" else 64)})
    with pytest.raises(ValueError, match="bindings differ"):
        validate(manifest, root, expected_bindings=different)


@pytest.mark.parametrize("mutation", ("payload", "handoff_json", "combined_dataset", "missing", "extra", "symlink", "hardlink"))
def test_source_content_closure_rejects_mutation(tmp_path, mutation):
    root, datasets = make_source(tmp_path, contexts=(8192,))
    manifest = close(root, datasets)
    item = next(f for f in manifest["files"] if f["role"] == (mutation if mutation in {"payload", "handoff_json", "combined_dataset"} else "payload"))
    path = root / item["relative_name"]
    if mutation == "missing":
        path.unlink()
    elif mutation == "extra":
        (root / "extra.bin").write_bytes(b"extra")
    elif mutation == "symlink":
        external = tmp_path / "external.bin"
        path.rename(external)
        path.symlink_to(external)
    elif mutation == "hardlink":
        (root / "alias.bin").hardlink_to(path)
    else:
        path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises((ValueError, OSError)):
        stage(manifest, root, tmp_path / "node")
    assert not (tmp_path / "node").exists()


@pytest.mark.parametrize("mutation", ("payload", "handoff_json", "projection", "attestation", "extra"))
def test_staged_verification_rejects_mutation(tmp_path, mutation):
    root, datasets = make_source(tmp_path, contexts=(8192,))
    manifest = close(root, datasets)
    staged = stage(manifest, root, tmp_path / "node")
    if mutation in {"payload", "handoff_json"}:
        path = staged.root / next(f["relative_name"] for f in manifest["files"] if f["role"] == mutation)
    elif mutation == "projection":
        path = staged.dataset_paths[(8192, VANILLA_CANARY_ARM)]
    elif mutation == "attestation":
        path = staged.attestation_path
    else:
        path = staged.root / "extra.bin"
    path.write_bytes((path.read_bytes() if path.exists() else b"") + b"changed")
    with pytest.raises((ValueError, OSError)):
        verify(manifest, staged.root)


def test_rejects_fp8_artifacts_in_bf16_experiment(tmp_path):
    root, datasets = make_source(tmp_path, contexts=(8192,), dtype="fp8_e5m2")
    with pytest.raises(ValueError, match="BF16"):
        close(root, datasets)


def test_rejects_lora_adapter_in_base_model_experiment(tmp_path):
    root, datasets = make_source(tmp_path, contexts=(8192,), lora_id="adapter-v1")
    with pytest.raises(ValueError, match="BF16 identities"):
        close(root, datasets)


def test_rejects_arm_method_swap(tmp_path):
    root, datasets = make_source(tmp_path, contexts=(8192,))
    rows = [json.loads(line) for line in datasets[8192].read_text().splitlines()]
    params = rows[0]["arm_kv_transfer_params"]
    params[FULL_PREFIX_CANARY_ARM], params[VANILLA_CANARY_ARM] = params[VANILLA_CANARY_ARM], params[FULL_PREFIX_CANARY_ARM]
    write_rows(datasets[8192], rows)
    with pytest.raises(ValueError, match="wrong cache method"):
        close(root, datasets)


def test_rejects_wrong_context_token_count_and_default_membership(tmp_path):
    root, datasets = make_source(tmp_path, contexts=(8192,))
    with pytest.raises(ValueError, match="exact distinct"):
        close_representative_handoff_bundle(root, datasets, bindings=bindings(), token_counter=token_counter)
    with pytest.raises(ValueError, match="tokens; expected"):
        close_representative_handoff_bundle(root, datasets, bindings=bindings(), token_counter=lambda _: 8191, example_count=2)


def test_rejects_manifest_digest_drift_and_unknown_fields(tmp_path):
    root, datasets = make_source(tmp_path, contexts=(8192,))
    manifest = close(root, datasets)
    with pytest.raises(ValueError, match="SHA-256"):
        validate(manifest, root, expected_bundle_sha256="0" * 64)
    changed = deepcopy(manifest)
    changed["unknown"] = True
    with pytest.raises(ValueError, match="keys"):
        validate(changed, root)


def test_rejects_reference_outside_bundle_and_symlink_ancestor(tmp_path):
    root, datasets = make_source(tmp_path, contexts=(8192,))
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        close(alias, {8192: alias / "8192/hotpotqa.jsonl"})
    rows = [json.loads(line) for line in datasets[8192].read_text().splitlines()]
    rows[0]["arm_kv_transfer_params"][FULL_PREFIX_CANARY_ARM][DOCUMENT_KV_PAYLOAD_URI_PARAM] = str(tmp_path / "outside.bin")
    (tmp_path / "outside.bin").write_bytes(b"outside")
    write_rows(datasets[8192], rows)
    with pytest.raises(ValueError, match="inside|within|outside|relative"):
        close(root, datasets)


def test_stage_rejects_overlap_existing_target_and_cleans_partial_copy(tmp_path, monkeypatch):
    root, datasets = make_source(tmp_path, contexts=(8192,))
    manifest = close(root, datasets)
    with pytest.raises(ValueError, match="overlap"):
        stage(manifest, root, root / "node")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        stage(manifest, root, existing)
    original_copy = shutil.copyfile

    def corrupt_copy(source, target, **kwargs):
        result = original_copy(source, target, **kwargs)
        Path(target).write_bytes(Path(target).read_bytes() + b"corrupt")
        return result

    monkeypatch.setattr(artifacts.shutil, "copyfile", corrupt_copy)
    with pytest.raises(ValueError, match="byte count|SHA-256"):
        stage(manifest, root, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.staging-*"))
