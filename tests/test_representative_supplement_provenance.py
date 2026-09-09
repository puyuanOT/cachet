"""Pure provenance/transport tests; runtime records here are synthetic fixtures."""

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import pytest

from document_kv_cache import databricks_vllm_smoke_job as jobs
from document_kv_cache import publication_handoff_closure_coordinator as source_closure
from document_kv_cache import vllm_smoke as runner
from document_kv_cache._benchmark_manifest import _resource_software_identity_from_package_revisions
from document_kv_cache.canary_orchestration import (
    representative_canary_workload_manifest,
    validate_representative_canary_workload_payload,
)
from document_kv_cache.representative_handoff_artifacts import RepresentativeSupplementProvenanceV1
from document_kv_cache.gpu_qualification_databricks_v2 import (
    GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT,
    GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256,
)
from test_gpu_qualification_v2 import _valid_attestation
from test_representative_handoff_consumption import consuming_job, prepared_fixture, runner_config
from test_representative_native_runtime_v2 import parameters, runtime_bundle
from test_representative_runtime_qualification import reclose, unit_record


def test_shared_execution_settings_allow_only_distinct_resource_sidecar_ids():
    from document_kv_cache.canary_orchestration import REPRESENTATIVE_CANARY_ARM_IDS, _validate_shared_result_identity
    from test_canary_orchestration import _result_record

    records = {arm: _result_record(arm) for arm in REPRESENTATIVE_CANARY_ARM_IDS}
    for index, (arm, record) in enumerate(records.items()):
        record["experiment_manifest"]["execution"]["resource_evidence_ids"] = [
            {"arm_id": arm, "evidence_digest": str(index + 1) * 64},
        ]
    _validate_shared_result_identity(records)
    records[REPRESENTATIVE_CANARY_ARM_IDS[1]]["experiment_manifest"]["execution"]["request_parallelism"] = 2
    with pytest.raises(ValueError, match="manifest execution"):
        _validate_shared_result_identity(records)


def supplement(**kwargs):
    return replace(RepresentativeSupplementProvenanceV1(
        source_closure_uri="/Volumes/catalog/schema/volume/source/source-closure.json",
        source_tree_sha256="a" * 64, runner_sha256=jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SHA256,
    ), **kwargs)


def supplement_job(index=0, *, provenance=None):
    base = consuming_job(index)
    claim = provenance or supplement()
    return replace(
        base, representative_supplement_provenance=claim,
        runner_python_file=f"/Volumes/catalog/schema/volume/{claim.runner_sha256}/{jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_BASENAME}",
        benchmark_manifest_provenance={"input_tokens_target": base.representative_handoff_source.context_tokens},
    )


@pytest.mark.parametrize("index", (0, 1, 2, 3, 4, 5, 7, 8, 9))
def test_successor_scope_roundtrip_uses_exact_native_and_resource_identities(monkeypatch, index):
    job = supplement_job(index)
    payload = jobs.build_databricks_vllm_smoke_run_submit_payload(job)
    workload = representative_canary_workload_manifest().workloads[index]
    validate_representative_canary_workload_payload(workload, payload, attempt_id=job.submission_attempt_id)
    parsed = runner_config(job, monkeypatch)
    assert parsed.representative_supplement_provenance == job.representative_supplement_provenance
    assert parsed.benchmark_manifest_provenance == job.benchmark_manifest_provenance
    provenance = parsed.benchmark_manifest_provenance
    assert provenance["measurement_scopes"] == ("latency", "resource")
    packages = provenance["package_revisions"]
    assert len(packages) == 201  # 197 native distributions +4 software identity entries.
    assert packages["torch"] == "2.13.0+cu129"
    assert packages["vllm"] == "0.27.1+cu129"
    assert packages["flashinfer-python"] == "0.6.16.post3"
    assert "fastapi" in packages and "fastapi[standard]" not in packages
    assert _resource_software_identity_from_package_revisions(packages) == {
        "source_revision": parsed.representative_handoff_source.bindings.source_commit,
        "source_tree_sha256": supplement().source_tree_sha256,
        "wheel_sha256": parsed.native_runtime_v2.package_wheel_sha256,
        "runner_sha256": jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SHA256,
    }


@pytest.mark.parametrize("mutation", ("quality", "latency_only", "unknown", "runner", "old_native_version", "extra_package", "source_revision"))
def test_closed_supplement_rejects_scope_and_provenance_drift(mutation):
    job = supplement_job(1)
    payload = jobs.build_databricks_vllm_smoke_run_submit_payload(job)
    args = parameters(payload)
    flag = "--representative-supplement-provenance-json" if mutation in {"quality", "latency_only", "unknown", "runner"} else "--benchmark-manifest-provenance-json"
    offset = args.index(flag) + 1
    record = json.loads(args[offset])
    if mutation == "quality":
        record["measurement_scopes"] = ["latency", "quality", "resource"]
    elif mutation == "latency_only":
        record["measurement_scopes"] = ["latency"]
    elif mutation == "unknown":
        record["approval"] = True
    elif mutation == "runner":
        record["runner_sha256"] = "f" * 64
    else:
        key, value = {"old_native_version": ("vllm", "0.10.2"), "extra_package": ("unbound", "1"),
                      "source_revision": ("cachet-source", "git:" + "0" * 40)}[mutation]
        record["package_revisions"][key] = value
    if "closed_record_sha256" in record:
        record["closed_record_sha256"] = runner._closed_record_sha256(record)
    args[offset] = json.dumps(record)
    with pytest.raises(ValueError):
        validate_representative_canary_workload_payload(
            representative_canary_workload_manifest().workloads[1], payload, attempt_id=job.submission_attempt_id,
        )


def test_successor_requires_immutable_native_and_versioned_runner():
    job = supplement_job()
    for change in ({"representative_handoff_source": None}, {"native_runtime_v2": None},
                   {"runner_python_file": consuming_job().runner_python_file}):
        with pytest.raises(ValueError):
            replace(job, **change)
    with pytest.raises(ValueError):
        RepresentativeSupplementProvenanceV1.from_record({**supplement().to_record(), "runner_sha256": "0" * 64})


def source_runtime_fixture(tmp_path, monkeypatch):
    job = supplement_job()
    source = job.representative_handoff_source
    record = {
        "record_type": source_closure._PUBLICATION_SOURCE_CLOSURE_V2_RECORD_TYPE,
        "schema_version": 2, "git": {"commit": source.bindings.source_commit},
        "runtime": source_closure._native_v2_source_closure_runtime_identity(),
        # The existing source bundle has four inventory entries plus this
        # closure JSON. Its bootstrap role is the qualification runner, never
        # the independently observed supplement smoke wrapper.
        "files": [
            {"role": "cachet_package_wheel", "relative_path": "cachet_kv-0.2.0-py3-none-any.whl",
             "byte_count": 1, "sha256": source.bindings.package_wheel_sha256},
            {"role": "cachet_source_distribution", "relative_path": "cachet_kv-0.2.0.tar.gz",
             "byte_count": 1, "sha256": "b" * 64},
            {"role": "git_source_archive", "relative_path": f"cachet-{source.bindings.source_commit}.tar.gz",
             "byte_count": 1, "sha256": "c" * 64},
            {"role": "gpu_qualification_bootstrap", "relative_path": "gpu-qualification-bootstrap-v2.py",
             "byte_count": len(GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT.encode()),
             "sha256": GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256},
        ],
    }
    record["closed_record_sha256"] = runner._closed_record_sha256(record)
    path = tmp_path / "source-closure.json"
    content = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(content)
    job = supplement_job(provenance=supplement(source_tree_sha256=sha256(content).hexdigest()))
    config = runner_config(job, monkeypatch)
    monkeypatch.setattr(source_closure, "_cluster_path", lambda uri: path)
    monkeypatch.setenv(jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SHA256_ENV, jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SHA256)
    native = config.native_runtime_v2
    freeze = [f"{name}=={version}" for name, version in native.expected_package_revisions().items()
              if name not in {"vllm", "flashinfer-python"}]
    attestation = _valid_attestation()
    for name, artifact in (("vllm", "patched_vllm_wheel"), ("flashinfer-python", "patched_flashinfer_wheel"), ("cachet-kv", "package_wheel")):
        uri = native.local_path(artifact).resolve().as_uri()
        freeze.append(f"{name} @ {uri}#sha256={getattr(native, artifact + '_sha256')}")
        if name != "cachet-kv":
            attestation[("vllm" if name == "vllm" else "flashinfer") + "_direct_url"] = uri
    return config, path, freeze, attestation


def test_runtime_source_verification_replays_real_source_parser_and_native_validator(tmp_path, monkeypatch):
    config, path, freeze, attestation = source_runtime_fixture(tmp_path, monkeypatch)
    bootstrap = next(item for item in json.loads(path.read_bytes())["files"] if item["role"] == "gpu_qualification_bootstrap")
    assert bootstrap["sha256"] != config.representative_supplement_provenance.runner_sha256
    record = runner.verify_representative_supplement_provenance(config, runtime_attestation=attestation, installed_freeze=freeze)
    assert record["closed_record_sha256"] == runner._closed_record_sha256(record)
    assert record["installed_package_freeze"] == freeze
    assert record["verified_runtime_package_versions"] == runtime_bundle().expected_package_revisions()
    assert record["provenance"] == config.representative_supplement_provenance.to_record()


@pytest.mark.parametrize("mutation", ("source_bytes", "source_commit", "source_wheel", "source_runtime", "runner", "native_attestation", "freeze_version", "freeze_extra", "freeze_duplicate", "freeze_origin"))
def test_runtime_source_verification_rejects_unverified_observations(tmp_path, monkeypatch, mutation):
    config, path, freeze, attestation = source_runtime_fixture(tmp_path, monkeypatch)
    if mutation.startswith("source_"):
        if mutation == "source_bytes":
            path.write_bytes(path.read_bytes() + b" ")
        else:
            record = json.loads(path.read_bytes())
            if mutation == "source_commit":
                record["git"]["commit"] = "0" * 40
            elif mutation == "source_wheel":
                record["files"][0]["sha256"] = "0" * 64
            else:
                record["runtime"]["base_lock"]["sha256"] = "0" * 64
            record["closed_record_sha256"] = runner._closed_record_sha256(record)
            content = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
            path.write_bytes(content)
            # Independent expected source/runtime pins remain fixed even when
            # a caller presents a newly hashed but conflicting source record.
            object.__setattr__(config, "representative_supplement_provenance", replace(
                config.representative_supplement_provenance, source_tree_sha256=sha256(content).hexdigest(),
            ))
    elif mutation == "runner":
        monkeypatch.delenv(jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SHA256_ENV)
    elif mutation == "native_attestation":
        attestation["vllm_package_version"] = "0.10.2"
    elif mutation == "freeze_version":
        freeze[0] = freeze[0].split("==")[0] + "==wrong"
    elif mutation == "freeze_extra":
        freeze.append("unbound==1")
    elif mutation == "freeze_duplicate":
        freeze.append(freeze[0])
    else:
        freeze[-1] = "cachet-kv @ file:///different.whl"
    with pytest.raises((ValueError, RuntimeError)):
        runner.verify_representative_supplement_provenance(config, runtime_attestation=attestation, installed_freeze=freeze)


@pytest.mark.parametrize("mutation", (None, "bytes", "symlink", "inherited"))
def test_wrapper_observes_compiled_runner_without_file_or_argv_assumptions(tmp_path, monkeypatch, mutation):
    path = tmp_path / jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_BASENAME
    path.write_text(jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SCRIPT)
    if mutation == "bytes":
        path.write_text(path.read_text() + "\n# changed\n")
    elif mutation == "symlink":
        target = tmp_path / "real.py"
        path.rename(target)
        path.symlink_to(target)
    monkeypatch.delenv(jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SHA256_ENV, raising=False)
    if mutation == "inherited":
        monkeypatch.setenv(jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SHA256_ENV, jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SHA256)
    monkeypatch.setattr(sys, "argv", ["databricks-driver", "--representative-supplement-provenance-json", json.dumps(supplement().to_record())])
    calls = []
    monkeypatch.setattr(jobs.subprocess if hasattr(jobs, "subprocess") else runner.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(returncode=0))
    namespace = {"__name__": "__main__"}  # Databricks need not define __file__.
    if mutation is not None:
        with pytest.raises(ValueError):
            exec(compile(jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SCRIPT, str(path), "exec"), namespace)
        assert not calls
    else:
        with pytest.raises(SystemExit) as result:
            exec(compile(jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SCRIPT, str(path), "exec"), namespace)
        assert result.value.code == 0
        assert calls[0][1]["env"][jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SHA256_ENV] == sha256(path.read_bytes()).hexdigest()
    os.environ.pop(jobs.REPRESENTATIVE_SUPPLEMENT_RUNNER_SHA256_ENV, None)


def prepared_supplement(tmp_path, monkeypatch, index=0):
    config, datasets, _, _ = prepared_fixture(tmp_path, monkeypatch, index=index)
    return replace(
        config, representative_supplement_provenance=supplement(),
        benchmark_manifest_provenance={"input_tokens_target": 8192},
        local_root=tmp_path,
    ), {"hotpotqa": datasets[8192]}


def mocked_qualification_process(config, calls, *, mutation=None):
    def execute(command, **kwargs):
        calls.append("qualification")
        assert command[:5] == [str(config.venv_python), "-I", "-B", "-m",
                               "document_kv_cache.representative_runtime_qualification"]
        assert kwargs["timeout_seconds"] == 300
        assert kwargs["cwd"] == config.local_dir
        bindings = json.loads(Path(command[command.index("--bindings-json") + 1]).read_bytes())
        if mutation == "timeout":
            raise subprocess.TimeoutExpired(command, 300)
        if mutation == "missing_output":
            return ""
        # Synthetic CPU validator input; this is never live GPU evidence.
        record = unit_record()
        record["bindings"] = bindings
        if mutation == "binding":
            record["bindings"]["source_commit"] = "0" * 40
        elif mutation == "gpu_failure":
            record["cases"][0]["value_mismatch_count"] = 1
        output = Path(command[command.index("--output-json") + 1])
        output.write_text(json.dumps(reclose(record)))
        return ""
    return execute


@pytest.mark.parametrize("index", (0, 1, 2))
def test_runner_stages_then_qualifies_before_starting_each_serving_arm(tmp_path, monkeypatch, index):
    config, datasets = prepared_supplement(tmp_path, monkeypatch, index)
    calls = []
    monkeypatch.setattr(runner, "create_venv", lambda *args: None)
    monkeypatch.setattr(runner, "install_native_v2_runtime", lambda *args: {})
    monkeypatch.setattr(runner, "verify_vllm_runtime_patch_closure", lambda *args: {})
    monkeypatch.setattr(runner, "installed_versions", lambda *args: {})
    monkeypatch.setattr(runner, "installed_package_freeze", lambda *args: ())
    monkeypatch.setattr(runner, "verify_representative_supplement_provenance", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "cuda_wheel_env_paths", lambda *args: {})
    monkeypatch.setattr(runner, "probe_vllm_import", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "benchmark_dataset_paths", lambda *args: datasets)
    monkeypatch.setattr(runner, "server_env", lambda *args: {})
    monkeypatch.setattr(runner, "validate_prompt_token_budget", lambda *args: calls.append("budget"))
    monkeypatch.setattr(runner, "_run_checked_text_subprocess_with_term_cleanup", mocked_qualification_process(config, calls))

    def forbidden(*args, **kwargs):
        raise AssertionError("consumption invoked generation or legacy installation")

    for name in ("prepare_generated_benchmark_handoffs", "generate_benchmark_handoff_bundles",
                 "load_benchmark_kv_chunk_generator", "install_vllm", "install_document_kv_package"):
        monkeypatch.setattr(runner, name, forbidden)

    class ServingStart(Exception):
        pass

    def start(*args):
        calls.append("server")
        assert config.representative_handoff_staging_attestation_path.is_file()
        assert (config.output_dir / "representative-runtime-qualification.json").is_file()
        raise ServingStart

    monkeypatch.setattr(runner, "start_vllm_server", start)
    with pytest.raises(ServingStart):
        runner.run_vllm_smoke_benchmark(config)
    assert calls == ["budget", "qualification", "server"]
    actual = json.loads((config.output_dir / "representative-runtime-qualification-bindings.json").read_bytes())
    assert actual["arm_id"] == config.representative_handoff_source.arm_id
    assert actual["runner_sha256"] == supplement().runner_sha256
    assert actual["stage_attestation_closed_record_sha256"] == json.loads(
        config.representative_handoff_staging_attestation_path.read_bytes(),
    )["closed_record_sha256"]


@pytest.mark.parametrize("mutation", ("timeout", "missing_output", "binding", "gpu_failure", "stage_binding"))
def test_premeasurement_hook_fails_closed_on_missing_or_invalid_qualification(tmp_path, monkeypatch, mutation):
    config, datasets = prepared_supplement(tmp_path, monkeypatch)
    runner.prepare_benchmark_handoff_inputs(config, datasets)
    monkeypatch.setattr(runner, "server_env", lambda *args: {})
    calls = []
    monkeypatch.setattr(runner, "_run_checked_text_subprocess_with_term_cleanup", mocked_qualification_process(config, calls, mutation=mutation))
    if mutation == "stage_binding":
        path = config.representative_handoff_staging_attestation_path
        stage = json.loads(path.read_bytes())
        stage["bindings"]["source_commit"] = "0" * 40
        stage["closed_record_sha256"] = runner._closed_record_sha256(stage)
        path.write_text(json.dumps(stage))
    with pytest.raises((ValueError, FileNotFoundError, subprocess.TimeoutExpired)):
        runner.run_representative_premeasurement_qualification(config)
    assert calls == ([] if mutation == "stage_binding" else ["qualification"])


@pytest.mark.parametrize("ambient", (None, "0", "2"))
def test_supplement_server_environment_preserves_cold_request_eviction(monkeypatch, ambient):
    config = runner_config(supplement_job(), monkeypatch)
    monkeypatch.delenv("DOCUMENT_KV_EVICT_PAGE_CACHE", raising=False)
    monkeypatch.delenv("DOCUMENT_KV_PREFETCH_WORKERS", raising=False)
    if ambient is not None:
        monkeypatch.setenv("DOCUMENT_KV_PREFETCH_WORKERS", ambient)
    if ambient == "2":
        with pytest.raises(ValueError, match="DOCUMENT_KV_PREFETCH_WORKERS.*conflicts"):
            runner.server_env(config)
    else:
        environment = runner.server_env(config)
        assert environment["DOCUMENT_KV_EVICT_PAGE_CACHE"] == "1"
        assert environment["DOCUMENT_KV_PREFETCH_WORKERS"] == "0"
