"""Representative consumption transports verified bytes; no serving/model runs."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import sys

import pytest

from document_kv_cache.canary_orchestration import (
    BASELINE_PREFILL_ARM,
    VANILLA_CANARY_ARM,
    representative_canary_workload_manifest,
    validate_representative_canary_workload_payload,
)
from document_kv_cache.databricks_vllm_smoke_job import build_databricks_vllm_smoke_run_submit_payload, main
from document_kv_cache.representative_handoff_artifacts import (
    RepresentativeHandoffBindings,
    RepresentativeHandoffSourceV1,
    close_representative_handoff_bundle,
)
from document_kv_cache import vllm_smoke as runner
from test_representative_handoff_artifacts import make_source, token_counter
from test_representative_native_runtime_v2 import job_config, parameters, runtime_bundle


def source_config(index=0, **kwargs):
    workload = representative_canary_workload_manifest().workloads[index]
    native = runtime_bundle()
    source = RepresentativeHandoffSourceV1(
        manifest_uri="/Volumes/catalog/schema/volume/set/manifest.json",
        manifest_file_sha256="1" * 64, bundle_closed_record_sha256="2" * 64,
        source_root="/Volumes/catalog/schema/volume/set/immutable",
        local_stage_root=f"/local_disk0/{workload.workload_id}/immutable",
        context_tokens=8192 if "8k" in workload.profile_id else 16384,
        arm_id=workload.arm_id,
        bindings=RepresentativeHandoffBindings(
            source_commit="a" * 40, package_wheel_sha256=native.package_wheel_sha256,
            native_runtime_closure_sha256=native.runtime_closure_manifest_sha256,
            prepared_input_bundle_sha256="d" * 64,
        ),
    )
    return replace(source, **kwargs)


def consuming_job(index=0, *, source=None):
    return replace(
        job_config(index), representative_handoff_source=source or source_config(index),
        benchmark_handoff_generator_factory=None, benchmark_handoff_output_dir=None,
        benchmark_handoff_cache_method=None, benchmark_handoff_segment_per_document=False,
    )


def runner_config(job, monkeypatch):
    args = list(parameters(build_databricks_vllm_smoke_run_submit_payload(job)))
    for flag in ("--package-wheel-uri", "--package-wheel-sha256"):
        offset = args.index(flag)
        del args[offset:offset + 2]
    monkeypatch.setenv(runner.DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV, job.wheel_sha256)
    return runner.parse_args(args)


@pytest.mark.parametrize("index", (0, 1, 2, 3, 4, 5, 7, 8, 9))
def test_consumption_roundtrip_preserves_profile_arm_native_and_attempt(index, monkeypatch):
    config = consuming_job(index)
    payload = build_databricks_vllm_smoke_run_submit_payload(config)
    args = parameters(payload)
    assert not any(arg.startswith("--benchmark-handoff-") for arg in args)
    workload = representative_canary_workload_manifest().workloads[index]
    validate_representative_canary_workload_payload(workload, payload, attempt_id=config.submission_attempt_id)
    source = config.representative_handoff_source
    assert RepresentativeHandoffSourceV1.from_record(source.to_record()) == source
    parsed = runner_config(config, monkeypatch)
    assert parsed.representative_handoff_source == source
    assert parsed.handoff_generation is None
    assert parsed.native_runtime_v2 == config.native_runtime_v2
    assert parsed.benchmark_manifest_provenance == config.benchmark_manifest_provenance
    assert parsed.benchmark_manifest_provenance["key_position_encoding"] == (
        "pre_rope" if source.arm_id == VANILLA_CANARY_ARM else "stored_post_rope"
    )


@pytest.mark.parametrize("mutation", ("context", "arm", "wheel", "closure", "version", "unknown", "generator", "native", "attempt", "profile"))
def test_closed_payload_rejects_consumption_drift(mutation):
    config = consuming_job(1)
    payload = deepcopy(build_databricks_vllm_smoke_run_submit_payload(config))
    args = parameters(payload)
    offset = args.index("--representative-handoff-source-json") + 1
    source = json.loads(args[offset])
    if mutation in {"context", "arm", "wheel", "closure", "version", "unknown"}:
        if mutation == "context":
            source["context_tokens"] = 16384
        elif mutation == "arm":
            source["arm_id"] = BASELINE_PREFILL_ARM
        elif mutation in {"wheel", "closure"}:
            field = "package_wheel_sha256" if mutation == "wheel" else "native_runtime_closure_sha256"
            source["bindings"][field] = "0" * 64
        elif mutation == "version":
            source["record_type"] = "cachet.representative_handoff_source.v0"
        else:
            source["unrecognized"] = True
        args[offset] = json.dumps(source)
    elif mutation == "generator":
        args.extend(["--benchmark-handoff-generator-factory", "forbidden:factory"])
    elif mutation == "native":
        native_offset = args.index("--native-runtime-v2-json")
        del args[native_offset:native_offset + 2]
    elif mutation == "attempt":
        payload["idempotency_token"] = "wrong-attempt"
    else:
        args[args.index("--max-model-len") + 1] = "8192"
    workload = representative_canary_workload_manifest().workloads[1]
    with pytest.raises((ValueError, TypeError)):
        validate_representative_canary_workload_payload(workload, payload, attempt_id=config.submission_attempt_id)


@pytest.mark.parametrize("path", ("../manifest.json", "/Volumes/a/../b/manifest.json", "/Volumes/a//manifest.json", "https://example.org/manifest.json"))
def test_source_config_rejects_nonpersistent_or_unnormalized_paths(path):
    with pytest.raises(ValueError):
        source_config(manifest_uri=path)


@pytest.mark.parametrize("uri,expected", (
    ("dbfs:/Volumes/catalog/schema/volume/manifest.json", "/Volumes/catalog/schema/volume/manifest.json"),
    ("dbfs:/FileStore/set/manifest.json", "/dbfs/FileStore/set/manifest.json"),
    ("/Volumes/catalog/schema/volume/manifest.json", "/Volumes/catalog/schema/volume/manifest.json"),
))
def test_source_config_uses_real_dbfs_and_unity_catalog_mounts(uri, expected):
    source = source_config(manifest_uri=uri, source_root=uri.rsplit("/", 1)[0])
    assert str(source.local_path("manifest_uri")) == expected
    assert str(source.local_path("source_root")) == expected.rsplit("/", 1)[0]


@pytest.mark.parametrize("mutation", ("context", "arm", "wheel", "closure", "native", "unused_generator_flag"))
def test_runner_parser_independently_rejects_consumption_drift(monkeypatch, mutation):
    config = consuming_job(1)
    args = list(parameters(build_databricks_vllm_smoke_run_submit_payload(config)))
    for flag in ("--package-wheel-uri", "--package-wheel-sha256"):
        offset = args.index(flag)
        del args[offset:offset + 2]
    monkeypatch.setenv(runner.DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV, config.wheel_sha256)
    offset = args.index("--representative-handoff-source-json") + 1
    source = json.loads(args[offset])
    if mutation == "native":
        native_offset = args.index("--native-runtime-v2-json")
        del args[native_offset:native_offset + 2]
    elif mutation == "unused_generator_flag":
        args.append("--benchmark-handoff-dtype=bfloat16")
    else:
        if mutation == "context":
            source["context_tokens"] = 16384
        elif mutation == "arm":
            source["arm_id"] = BASELINE_PREFILL_ARM
        else:
            field = "package_wheel_sha256" if mutation == "wheel" else "native_runtime_closure_sha256"
            source["bindings"][field] = "0" * 64
        args[offset] = json.dumps(source)
    with pytest.raises((ValueError, SystemExit)):
        runner.parse_args(args)


def test_legacy_runner_preparation_keeps_generator_dispatch(monkeypatch):
    config = runner_config(job_config(1), monkeypatch)
    calls = []
    monkeypatch.setattr(runner, "prepare_generated_benchmark_handoffs", lambda cfg, paths: calls.append((cfg, paths)) or paths)
    paths = {"hotpotqa": config.local_dir / "hotpotqa.jsonl"}
    assert runner.prepare_benchmark_handoff_inputs(config, paths) is paths
    assert calls == [(config, paths)]


def test_databricks_parser_renders_exact_consumption_payload_without_submission(tmp_path, monkeypatch):
    config = consuming_job(2)
    expected = build_databricks_vllm_smoke_run_submit_payload(config)
    args = list(parameters(expected))
    for before, after in (("--package-wheel-uri", "--wheel-uri"), ("--package-wheel-sha256", "--wheel-sha256")):
        args[args.index(before)] = after
    output = tmp_path / "payload.json"
    args.extend([
        "--runner-python-file", config.runner_python_file,
        "--single-user-name", config.single_user_name,
        "--submission-attempt-id", config.submission_attempt_id,
        "--output-json", str(output),
    ])
    monkeypatch.setattr(sys, "argv", ["payload-renderer", *args])
    assert main() == 0
    assert json.loads(output.read_bytes()) == expected
    monkeypatch.setattr(sys, "argv", ["payload-renderer", *args, "--benchmark-handoff-dtype=bfloat16"])
    with pytest.raises(SystemExit):
        main()


def prepared_fixture(tmp_path, monkeypatch, *, index=0, example_count=32):
    root, datasets = make_source(tmp_path, contexts=(8192,), example_count=example_count)
    source = source_config(index)
    manifest = close_representative_handoff_bundle(
        root, datasets, bindings=source.bindings, token_counter=token_counter,
        example_count=example_count,
    )
    content = json.dumps(manifest, sort_keys=True).encode()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(content)
    source = replace(source, manifest_file_sha256=sha256(content).hexdigest(),
                     bundle_closed_record_sha256=manifest["closed_record_sha256"])
    config = runner_config(consuming_job(index, source=source), monkeypatch)
    config = replace(config, output_dir=tmp_path / "output")
    # Only translate cluster mounts to temporary test storage. The real stage,
    # hashing, schema validation and projection code all execute unchanged.
    paths = {"manifest_uri": manifest_path, "source_root": root,
             "local_stage_root": tmp_path / "node"}
    monkeypatch.setattr(RepresentativeHandoffSourceV1, "local_path", lambda self, field: paths[field])
    return config, datasets, manifest_path, paths


@pytest.mark.parametrize("index", (0, 1, 2))
def test_actual_runner_preparation_stages_without_generation(tmp_path, monkeypatch, index):
    config, datasets, _, paths = prepared_fixture(tmp_path, monkeypatch, index=index)

    def forbidden(*args, **kwargs):
        raise AssertionError("consuming runner invoked generation")

    monkeypatch.setattr(runner, "prepare_generated_benchmark_handoffs", forbidden)
    monkeypatch.setattr(runner, "generate_benchmark_handoff_bundles", forbidden)
    monkeypatch.setattr(runner, "load_benchmark_kv_chunk_generator", forbidden)
    selected = runner.prepare_benchmark_handoff_inputs(config, {"hotpotqa": datasets[8192]})
    rows = [json.loads(line) for line in selected["hotpotqa"].read_text().splitlines()]
    assert len(rows) == 32
    assert selected["hotpotqa"].is_relative_to(paths["local_stage_root"])
    assert config.representative_handoff_staging_attestation_path.read_bytes() == (
        paths["local_stage_root"] / "representative-handoff-staging.json"
    ).read_bytes()
    assert not config.prepared_handoff_generation_path.exists()
    coverage = runner.validate_prepared_benchmark_handoffs(config, selected)
    if config.representative_handoff_source.arm_id == BASELINE_PREFILL_ARM:
        assert coverage is None
        assert all("kv_transfer_params" not in row for row in rows)
    else:
        assert coverage["ok"] is True
        assert coverage["handoff_topology_attestation"] is not None
    assert len(runner.build_prompt_token_budget_rows(config, selected)) == 32


@pytest.mark.parametrize("mutation", ("manifest_hash", "record_hash", "membership", "logical_input", "payload"))
def test_runner_rejects_unverified_inputs_before_stage(tmp_path, monkeypatch, mutation):
    config, datasets, manifest_path, paths = prepared_fixture(
        tmp_path, monkeypatch, example_count=2 if mutation == "membership" else 32,
    )
    if mutation == "manifest_hash":
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    elif mutation == "record_hash":
        value = json.loads(manifest_path.read_bytes())
        value["portable_identity_sha256"] = "0" * 64
        content = json.dumps(value).encode()
        manifest_path.write_bytes(content)
        # Preserve all validated runtime config; change only the transport's
        # file pin, leaving the independently pinned closed record unchanged.
        object.__setattr__(config, "representative_handoff_source", replace(
            config.representative_handoff_source, manifest_file_sha256=sha256(content).hexdigest(),
        ))
    elif mutation == "logical_input":
        alternate = tmp_path / "alternate.jsonl"
        alternate.write_text(datasets[8192].read_text().replace('"query": "What?"', '"query": "Changed?"'))
        datasets[8192] = alternate
    elif mutation == "payload":
        value = json.loads(manifest_path.read_bytes())
        payload = paths["source_root"] / next(f["relative_name"] for f in value["files"] if f["role"] == "payload")
        payload.write_bytes(payload.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        runner.prepare_representative_handoff_inputs(config, {"hotpotqa": datasets[8192]})
    assert not paths["local_stage_root"].exists()
    assert not config.representative_handoff_staging_attestation_path.exists()
