import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

import document_kv_cache.gpu_qualification_databricks as qualification_job
import document_kv_cache.publication_bf16_handoff_generation as generation
import document_kv_cache.publication_handoff_closure_coordinator as closure_coordinator
import document_kv_cache.publication_latency_handoff_generation as q8_generation
from document_kv_cache.benchmarks import SUPPORTED_V1_DATASETS
from document_kv_cache.databricks_resource_ledger import (
    DatabricksClusterHourLedger,
    DatabricksRunAttemptReservationRequest,
    create_databricks_cluster_hour_ledger_json,
    databricks_ledger_path_sha256,
    databricks_ledger_prefix,
    read_databricks_cluster_hour_ledger_json,
    replay_databricks_run_attempt_batch_authorization_json,
    reserve_databricks_run_attempt_json,
)
from document_kv_cache.databricks_runs import DatabricksWorkspaceConfig
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256,
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    GPUQualificationSelection,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_EVIDENCE_RECORD_TYPE,
    GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE,
    GPUQualificationArtifactPinsV2,
)
from document_kv_cache.gpu_qualification_databricks import (
    GPUQualificationLaunchAuthorization,
)
from document_kv_cache.main_latency_inputs import (
    MAIN_LATENCY_TARGET_SEGMENT_COUNTS,
    MainLatencyInputFile,
    PreparedMainLatencyInputs,
)
from document_kv_cache.publication_bf16_handoff_generation import (
    PUBLICATION_BF16_HANDOFF_EXECUTION_MODE,
    PUBLICATION_BF16_HANDOFF_MAX_RESERVED_GPU_HOURS,
    PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS,
    PUBLICATION_BF16_HANDOFF_WORKER_COUNT,
    DatabricksPublicationBF16HandoffJobConfig,
    PublicationBF16HandoffGenerationResult,
    PublicationBF16HandoffServingAuthorization,
    authorize_publication_bf16_handoff_serving,
    build_databricks_publication_bf16_handoff_submit_payloads,
    build_publication_bf16_handoff_execution_config,
    build_publication_bf16_handoff_generation_plan,
    build_publication_bf16_handoff_resource_estimate,
    build_publication_bf16_handoff_worker_payloads,
    collect_publication_bf16_handoff_worker_attestation,
    publication_bf16_handoff_terminal_actual_gpu_seconds_from_ledger,
    publication_bf16_handoff_worker_attempt_id,
    reserve_and_submit_publication_bf16_handoff_worker,
    reserve_and_submit_publication_bf16_handoff_worker_wave,
    resume_publication_bf16_handoff_worker_wave,
    require_publication_bf16_handoff_submission_authorization,
    reserve_publication_bf16_handoff_worker_attempt_json,
    resolve_publication_bf16_handoff_bundle,
)
from document_kv_cache.publication_latency_handoff_generation import (
    PublicationLatencyHandoffGenerationResult,
    PublicationLatencyHandoffServingAuthorization,
    PublicationLatencyGeneratorHardwareQualificationV2,
)
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PATCHED_WHEEL_SHA256,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
)


@pytest.fixture(autouse=True)
def _authenticated_bf16_workspace(monkeypatch):
    monkeypatch.setattr(
        generation,
        "require_databricks_current_user_name",
        lambda workspace, *, expected_user_name, opener=None: {
            "workspace_host_sha256": sha256(
                workspace.normalized_host.encode("utf-8")
            ).hexdigest(),
            "user_name_sha256": sha256(expected_user_name.encode("utf-8")).hexdigest(),
        },
    )


def test_bf16_serving_resolver_rejects_authority_subclasses():
    class SmuggledAuthorization(PublicationBF16HandoffServingAuthorization):
        pass

    with pytest.raises(TypeError, match="PublicationBF16HandoffServingAuthorization"):
        resolve_publication_bf16_handoff_bundle(object.__new__(SmuggledAuthorization))


def test_bf16_predecessor_rejects_authority_subclasses_before_ledger_io():
    class SmuggledQualification(GPUQualificationLaunchAuthorization):
        pass

    with pytest.raises(TypeError, match="GPUQualificationLaunchAuthorization"):
        generation._require_matching_predecessor_ledger(
            "never-read-ledger.json",
            object.__new__(SmuggledQualification),
            object(),
            expected_input_bundle_sha256="a" * 64,
            expected_qualification_closed_record_sha256="b" * 64,
        )

    class SmuggledQ8Closure(
        closure_coordinator.PublicationHandoffRemoteClosureAuthorization
    ):
        pass

    with pytest.raises(
        TypeError,
        match="PublicationHandoffRemoteClosureAuthorization",
    ):
        generation._require_matching_predecessor_ledger(
            "never-read-ledger.json",
            object.__new__(GPUQualificationLaunchAuthorization),
            object.__new__(SmuggledQ8Closure),
            expected_input_bundle_sha256="a" * 64,
            expected_qualification_closed_record_sha256="b" * 64,
        )


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) for character in text]


class JsonHTTPResponse:
    status = 200

    def __init__(self, value):
        self._body = json.dumps(value).encode("utf-8")
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, amt=-1):
        if amt < 0:
            amt = len(self._body) - self._offset
        end = min(self._offset + amt, len(self._body))
        chunk = self._body[self._offset : end]
        self._offset = end
        return chunk


def _canonical_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _prepared_row(dataset, example_index, context_tokens):
    segment_count = MAIN_LATENCY_TARGET_SEGMENT_COUNTS[context_tokens]
    return {
        "dataset": dataset,
        "documents": [
            {
                "document_id": (
                    f"{dataset}-{example_index:02d}-{context_tokens}-{segment_index:02d}"
                ),
                "metadata": {},
                "text": (
                    f"evidence {dataset} {example_index:02d} "
                    f"{context_tokens} {segment_index:02d}."
                ),
                "title": f"segment {segment_index:02d}",
            }
            for segment_index in range(segment_count)
        ],
        "example_id": f"{dataset}-example-{example_index:02d}",
        "expected_answer": "answer",
        "query": "What is the answer?",
    }


def _fake_prepared_inputs(tmp_path):
    root = tmp_path / "prepared"
    files = []
    for context_tokens in MAIN_LATENCY_TARGET_SEGMENT_COUNTS:
        for dataset in SUPPORTED_V1_DATASETS:
            path = root / str(context_tokens) / f"{dataset}.jsonl"
            _canonical_jsonl(
                path,
                [_prepared_row(dataset, index, context_tokens) for index in range(32)],
            )
            files.append(
                MainLatencyInputFile(
                    dataset=dataset,
                    input_tokens_target=context_tokens,
                    segment_count=MAIN_LATENCY_TARGET_SEGMENT_COUNTS[context_tokens],
                    jsonl_path=path,
                    jsonl_sha256=sha256(path.read_bytes()).hexdigest(),
                )
            )
    provenance = root / "provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")
    return PreparedMainLatencyInputs(
        output_dir=root,
        provenance_json_path=provenance,
        provenance_sha256=sha256(provenance.read_bytes()).hexdigest(),
        bundle_sha256=GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
        files=tuple(files),
    )


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    value = _fake_prepared_inputs(tmp_path)
    monkeypatch.setattr(
        generation,
        "verify_main_latency_inputs",
        lambda *args, **kwargs: value,
    )
    return value


def _qualification(monkeypatch, prepared):
    selection = GPUQualificationSelection(
        attention_backend="TRITON_ATTN",
        gpu_memory_utilization=0.75,
        generation_hardware_id="aws-g6e-l40s",
        generation_databricks_node_type_id="g6e.4xlarge",
        generation_artifacts_sha256="7" * 64,
        generation_prefix_tokens_per_second=40.0,
        plan_sha256="8" * 64,
    )
    monkeypatch.setattr(
        q8_generation,
        "validate_gpu_qualification_evidence_v2_record",
        lambda *args, **kwargs: selection,
    )
    monkeypatch.setattr(
        q8_generation,
        "validate_gpu_qualification_plan_v2_record",
        lambda *args, **kwargs: None,
    )
    pins = GPUQualificationArtifactPinsV2(
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_sha256=GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        patched_flashinfer_wheel_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
        runtime_closure_manifest_sha256=RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        package_wheel_sha256="3" * 64,
        cachet_source_tree_sha256="4" * 64,
        runner_sha256="5" * 64,
        input_bundle_sha256=prepared.bundle_sha256,
    )
    return PublicationLatencyGeneratorHardwareQualificationV2(
        evidence_record={
            "closed_record_sha256": "6" * 64,
            "record_type": GPU_QUALIFICATION_V2_EVIDENCE_RECORD_TYPE,
        },
        plan_record={
            "closed_record_sha256": "8" * 64,
            "record_type": GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE,
        },
        expected_campaign_id="vllm-0271-publication-v1",
        expected_artifact_pins=pins,
        evidence_uri="dbfs:/qualification/evidence.json",
        evidence_file_sha256="9" * 64,
        plan_uri="dbfs:/qualification/plan.json",
        plan_file_sha256="a" * 64,
    )


def _authorization(
    qualification,
    *,
    ledger_path,
    plan_sha256=None,
    ledger_id="gpu-qualification-test-ledger",
):
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    return GPUQualificationLaunchAuthorization(
        selection=qualification.selection,
        plan_sha256=plan_sha256 or qualification.selection.plan_sha256,
        evidence_closed_record_sha256=(
            qualification.evidence_record["closed_record_sha256"]
        ),
        evidence_file_sha256=qualification.evidence_file_sha256,
        ledger_id=ledger_id,
        ledger_path_sha256=databricks_ledger_path_sha256(ledger_path),
        predecessor_prefix=databricks_ledger_prefix(ledger),
        producer_batch_prefix=databricks_ledger_prefix(ledger),
        ledger_prefix=databricks_ledger_prefix(ledger),
        causal_closure_sha256="c" * 64,
        _issuer=qualification_job._LAUNCH_AUTHORIZATION_ISSUER,
    )


def _q8_authorization(tmp_path, ledger_path, authorization, monkeypatch, prepared):
    del tmp_path
    prefix = databricks_ledger_prefix(
        read_databricks_cluster_hour_ledger_json(ledger_path)
    )
    lineage = {
        "ledger_id": authorization.ledger_id,
        "ledger_path_sha256": authorization.ledger_path_sha256,
        "predecessor_prefix": prefix.to_record(),
        "producer_batch_prefix": prefix.to_record(),
        "terminal_prefix": prefix.to_record(),
    }
    request = {
        "closed_record_sha256": "a" * 64,
        "ledger_lineage": lineage,
        "output_root_uri": ("dbfs:/Volumes/catalog/schema/volume/q8-authorized-result"),
        "result_uri": (
            "dbfs:/Volumes/catalog/schema/volume/q8-authorized-result/result.json"
        ),
        "stage": closure_coordinator.PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE,
    }
    execution = {
        "closed_record_sha256": "b" * 64,
        "file_sha256": "c" * 64,
        "record": {},
        "uri": (
            "dbfs:/Volumes/catalog/schema/volume/q8-authorized-result/"
            "publication-latency-handoff-execution.json"
        ),
    }
    result = {
        "closed_record_sha256": "d" * 64,
        "coordinator": {"run_id": "12345"},
        "execution": execution,
        "manifests": [],
        "pins": {
            "input_bundle_sha256": prepared.bundle_sha256,
            "qualification_closed_record_sha256": (
                authorization.evidence_closed_record_sha256
            ),
        },
    }
    monkeypatch.setattr(
        closure_coordinator, "_validate_closure_request", lambda _record: None
    )
    monkeypatch.setattr(
        closure_coordinator,
        "_validate_closure_result",
        lambda _record, *, request: None,
    )
    capability = closure_coordinator.PublicationHandoffRemoteClosureAuthorization(
        request=request,
        result=result,
        result_file_sha256=sha256(
            closure_coordinator._canonical_json_bytes(result, pretty=True)
        ).hexdigest(),
        coordinator_run_id="12345",
        control_plane_status_sha256="e" * 64,
        workspace_host_sha256=sha256(
            b"https://example.cloud.databricks.com"
        ).hexdigest(),
        user_name_sha256=sha256(b"publication@example.com").hexdigest(),
        _issuer=closure_coordinator._AUTHORIZATION_ISSUER,
    )
    return capability


def _launch_material(prepared, monkeypatch, *, ledger_path):
    plan = build_publication_bf16_handoff_generation_plan(
        prepared.output_dir,
        plan_id="bf16-handoffs-2026",
        tokenizer=CharacterTokenizer(),
    )
    config = build_publication_bf16_handoff_execution_config(
        vllm_bitsandbytes_loader_source_sha256=(
            GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
        )
    )
    qualification = _qualification(monkeypatch, prepared)
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="gpu-qualification-test-ledger",
    )
    authorization = _authorization(qualification, ledger_path=ledger_path)
    q8_authorization = _q8_authorization(
        prepared.output_dir.parent,
        ledger_path,
        authorization,
        monkeypatch,
        prepared,
    )
    payloads = build_publication_bf16_handoff_worker_payloads(
        plan,
        plan_uri="dbfs:/plans/bf16-plan.json",
        plan_file_sha256="d" * 64,
        prepared_input_uri="dbfs:/prepared/main-latency",
        prepared_provenance_file_sha256="e" * 64,
        prepared_provenance_closed_record_sha256="f" * 64,
        durable_output_root=(
            "dbfs:/Volumes/catalog/schema/volume/publication-bf16-handoffs/"
            f"{plan['closed_record_sha256']}"
        ),
        local_work_root_template="/local_disk0/bf16-worker-{worker_index}",
        source_revision="deadbeef" * 5,
        config=config,
        hardware_qualification=qualification,
    )
    job = DatabricksPublicationBF16HandoffJobConfig(
        runner_python_file="dbfs:/runner.py",
        worker_payload_uri_template="dbfs:/payloads/worker-{worker_index}.json",
        package_wheel_uri="dbfs:/cachet.whl",
        package_wheel_sha256="3" * 64,
        runtime_lock_uri="dbfs:/runtime.lock",
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_uri="dbfs:/vllm.whl",
        patched_vllm_wheel_sha256=GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        patched_flashinfer_wheel_uri="dbfs:/flashinfer.whl",
        patched_flashinfer_wheel_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
        runtime_closure_manifest_uri="dbfs:/runtime-closure.json",
        runtime_closure_manifest_sha256=RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        source_revision="deadbeef" * 5,
        cachet_source_tree_sha256="4" * 64,
        single_user_name="publication@example.com",
    )
    assert job.zone_id == "auto"
    return (
        plan,
        config,
        qualification,
        authorization,
        q8_authorization,
        payloads,
        job,
    )


def _terminal_run(submit_payload, worker_index, *, cluster_index=None):
    parent_run_id = 10_000 + worker_index
    cluster_id = (
        f"cluster-{cluster_index if cluster_index is not None else worker_index:02d}"
    )
    start = 1_000_000 + worker_index * 10_000
    return {
        "cluster_instance": {"cluster_id": cluster_id},
        "end_time": start + 1200,
        "original_attempt_run_id": parent_run_id,
        "repair_history": [],
        "run_id": parent_run_id,
        "run_name": submit_payload["run_name"],
        "start_time": start,
        "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
        "tasks": [
            {
                "attempt_number": 0,
                "cluster_instance": {"cluster_id": cluster_id},
                "end_time": start + 1100,
                "new_cluster": copy.deepcopy(submit_payload["tasks"][0]["new_cluster"]),
                "run_id": 20_000 + worker_index,
                "spark_python_task": copy.deepcopy(
                    submit_payload["tasks"][0]["spark_python_task"]
                ),
                "start_time": start + 100,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "SUCCESS",
                },
                "task_key": f"bf16_handoff_worker_{worker_index:02d}",
            }
        ],
    }


def test_plan_is_exact_16k_128_rows_and_resource_bound(prepared):
    plan = build_publication_bf16_handoff_generation_plan(
        prepared.output_dir,
        plan_id="bf16-handoffs-2026",
        tokenizer=CharacterTokenizer(),
    )
    items = [item for worker in plan["workers"] for item in worker["items"]]
    assert len(items) == 128
    assert plan["record_type"].endswith(".v2")
    assert plan["schema_version"] == 2
    assert {item["context_tokens"] for item in items} == {16384}
    assert len(plan["workers"]) == PUBLICATION_BF16_HANDOFF_WORKER_COUNT == 16
    assert plan["coverage"]["input_token_slots"] == 4 * 32 * 16384
    config = build_publication_bf16_handoff_execution_config(
        vllm_bitsandbytes_loader_source_sha256=(
            GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
        )
    )
    estimate = build_publication_bf16_handoff_resource_estimate(plan, config=config)
    assert config.layout.dtype == "bfloat16"
    assert config.layout.bytes_per_token == 147_456
    assert estimate["logical_payload_bytes"] == (
        plan["coverage"]["cache_prefix_generation_tokens"] * 147_456
    )
    assert estimate["reserved_gpu_hour_upper_bound"] == 80.0
    assert PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS == 18_000
    assert PUBLICATION_BF16_HANDOFF_MAX_RESERVED_GPU_HOURS == 80.0


def test_bf16_runner_inherits_native_v2_four_step_runtime_bootstrap() -> None:
    script = generation.PUBLICATION_BF16_HANDOFF_RUNNER_SCRIPT
    for flag in (
        "--runtime-lock-sha256",
        "--patched-vllm-wheel-sha256",
        "--patched-flashinfer-wheel-sha256",
        "--runtime-closure-manifest-sha256",
    ):
        assert flag in script
    assert 'variable_name.upper().startswith(("PIP_", "_PIP_"))' in script
    assert '[sys.executable, "-m", "venv", "--copies", venv_dir]' in script
    assert "verify_gpu_qualification_v2_runtime_installation" in script
    install_markers = (
        '"--require-hashes", "--only-binary", ":all:"',
        '"vllm", patched_vllm_wheel',
        '"flashinfer-python"',
        '"cachet-kv", package_wheel',
    )
    positions = tuple(script.index(marker) for marker in install_markers)
    assert positions == tuple(sorted(positions))


def test_bf16_launch_rejects_v1_payload_and_exact8_pin_tamper(
    prepared, monkeypatch, tmp_path
) -> None:
    plan, *_unused, payloads, _job = _launch_material(
        prepared,
        monkeypatch,
        ledger_path=tmp_path / "ledger.json",
    )
    legacy = copy.deepcopy(payloads[0])
    legacy["record_type"] = "cachet.publication_bf16_handoff_worker_payload.v1"
    legacy["schema_version"] = 1
    legacy["closed_record_sha256"] = generation._closed_record_sha256(legacy)
    with pytest.raises(ValueError, match="native-v2 worker payload"):
        generation.validate_publication_bf16_handoff_worker_payload(
            legacy,
            plan=plan,
        )

    for mutation in ("missing", "extra", "flashinfer"):
        tampered = copy.deepcopy(payloads[0])
        pins = tampered["generator_hardware_qualification"]["expected_artifact_pins"]
        if mutation == "missing":
            pins.pop("runtime_closure_manifest_sha256")
        elif mutation == "extra":
            pins["downstream_runner_sha256"] = "1" * 64
        else:
            pins["patched_flashinfer_wheel_sha256"] = "1" * 64
        tampered["closed_record_sha256"] = generation._closed_record_sha256(tampered)
        with pytest.raises(ValueError):
            generation.validate_publication_bf16_handoff_worker_payload(
                tampered,
                plan=plan,
            )


def test_bf16_qualification_runner_cannot_be_reused_as_downstream_runner(
    prepared, monkeypatch, tmp_path
) -> None:
    (
        _plan,
        _config,
        qualification,
        authorization,
        q8_authorization,
        payloads,
        job,
    ) = _launch_material(
        prepared,
        monkeypatch,
        ledger_path=tmp_path / "ledger.json",
    )
    assert qualification.expected_artifact_pins.runner_sha256 != job.runner_sha256
    conflated = copy.deepcopy(payloads)
    conflated[0]["generator_hardware_qualification"]["expected_artifact_pins"][
        "runner_sha256"
    ] = job.runner_sha256
    conflated[0]["closed_record_sha256"] = generation._closed_record_sha256(
        conflated[0]
    )
    with pytest.raises(ValueError, match="runner must remain distinct"):
        build_databricks_publication_bf16_handoff_submit_payloads(
            job,
            conflated,
            ledger_path=tmp_path / "ledger.json",
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
        )


def test_submit_is_capability_gated_exact_16x_l40s_five_hours(
    prepared,
    monkeypatch,
    tmp_path,
):
    ledger_path = tmp_path / "qualification-ledger.json"
    (
        plan,
        _config,
        qualification,
        authorization,
        q8_authorization,
        payloads,
        job,
    ) = _launch_material(
        prepared,
        monkeypatch,
        ledger_path=ledger_path,
    )
    wrong_ledger_path = tmp_path / "wrong-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        wrong_ledger_path,
        ledger_id="different-qualification-ledger",
    )
    submissions = build_databricks_publication_bf16_handoff_submit_payloads(
        job,
        payloads,
        ledger_path=ledger_path,
        qualification_launch_authorization=authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
    )
    with pytest.raises(
        ValueError, match="remote Q8 closure ledger predecessor binding drift"
    ):
        reserve_publication_bf16_handoff_worker_attempt_json(
            wrong_ledger_path,
            submissions[0],
            worker_payload=payloads[0],
            worker_index=0,
            attempt_id="different-ledger-reserve",
            job_config=job,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
        )

    opener_called = False

    def opener(*args, **kwargs):
        nonlocal opener_called
        opener_called = True
        raise AssertionError("cross-ledger submit must fail before cloud submission")

    with pytest.raises(
        ValueError, match="remote Q8 closure ledger predecessor binding drift"
    ):
        reserve_and_submit_publication_bf16_handoff_worker(
            DatabricksWorkspaceConfig(
                "https://example.cloud.databricks.com",
                "test-token",
            ),
            submissions[0],
            ledger_path=wrong_ledger_path,
            worker_payload=payloads[0],
            worker_index=0,
            attempt_id="different-ledger-submit",
            job_config=job,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
            opener=opener,
        )
    assert opener_called is False
    assert len(submissions) == 16
    assert sum(item["timeout_seconds"] for item in submissions) / 3600 == 80
    assert all(item["tasks"][0]["max_retries"] == 0 for item in submissions)
    assert all(
        item["tasks"][0]["new_cluster"]["node_type_id"] == "g6e.4xlarge"
        for item in submissions
    )
    assert {
        item["tasks"][0]["new_cluster"]["aws_attributes"]["zone_id"]
        for item in submissions
    } == {"auto"}
    with pytest.raises(TypeError, match="GPUQualificationLaunchAuthorization"):
        build_databricks_publication_bf16_handoff_submit_payloads(
            job,
            payloads,
            ledger_path=ledger_path,
            qualification_launch_authorization=object(),
            q8_handoff_remote_closure_authorization=q8_authorization,
        )
    with pytest.raises(
        ValueError, match="remote Q8 closure ledger predecessor binding drift"
    ):
        build_databricks_publication_bf16_handoff_submit_payloads(
            job,
            payloads,
            ledger_path=wrong_ledger_path,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
        )
    with pytest.raises(ValueError, match="authorization plan binding differs"):
        build_databricks_publication_bf16_handoff_submit_payloads(
            job,
            payloads,
            ledger_path=ledger_path,
            qualification_launch_authorization=_authorization(
                qualification,
                ledger_path=ledger_path,
                plan_sha256="1" * 64,
            ),
            q8_handoff_remote_closure_authorization=q8_authorization,
        )
    traversal_payloads = copy.deepcopy(payloads)
    traversal_payloads[0]["local_work_root"] = "/local_disk0/worker-00/.."
    traversal_payloads[0]["closed_record_sha256"] = generation._closed_record_sha256(
        traversal_payloads[0]
    )
    with pytest.raises(ValueError, match="direct child of /local_disk0"):
        build_databricks_publication_bf16_handoff_submit_payloads(
            job,
            traversal_payloads,
            ledger_path=ledger_path,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
        )
    assert plan["closed_record_sha256"] in payloads[0]["durable_output_root"]


def test_bf16_render_rejects_raw_or_legacy_local_q8_authority(
    prepared, monkeypatch, tmp_path
) -> None:
    ledger_path = tmp_path / "ledger.json"
    (
        _plan,
        _config,
        _qualification_record,
        authorization,
        remote_q8_authorization,
        payloads,
        job,
    ) = _launch_material(prepared, monkeypatch, ledger_path=ledger_path)
    local_root = tmp_path / "legacy-local-q8"
    local_root.mkdir()
    execution_path = local_root / "execution.json"
    execution_path.write_text("{}\n", encoding="utf-8")
    prefix = databricks_ledger_prefix(
        read_databricks_cluster_hour_ledger_json(ledger_path)
    )
    local_q8_authorization = PublicationLatencyHandoffServingAuthorization(
        result=PublicationLatencyHandoffGenerationResult(
            root=local_root,
            execution_record_path=execution_path,
            record={"closed_record_sha256": "b" * 64},
        ),
        ledger_id=authorization.ledger_id,
        ledger_path_sha256=authorization.ledger_path_sha256,
        predecessor_prefix=prefix,
        producer_batch_prefix=prefix,
        ledger_prefix=prefix,
        causal_closure_sha256="d" * 64,
        _issuer=q8_generation._SERVING_AUTHORIZATION_ISSUER,
    )
    for rejected in (
        local_q8_authorization,
        dict(remote_q8_authorization.result_record),
        dict(remote_q8_authorization.request_record),
    ):
        with pytest.raises(
            TypeError, match="PublicationHandoffRemoteClosureAuthorization"
        ):
            build_databricks_publication_bf16_handoff_submit_payloads(
                job,
                payloads,
                ledger_path=ledger_path,
                qualification_launch_authorization=authorization,
                q8_handoff_remote_closure_authorization=rejected,
            )


@pytest.mark.parametrize("binding", ["input", "qualification"])
def test_bf16_render_rejects_resealed_q8_binding_drift(
    binding, prepared, monkeypatch, tmp_path
) -> None:
    ledger_path = tmp_path / "ledger.json"
    (
        _plan,
        _config,
        _qualification_record,
        authorization,
        remote_q8_authorization,
        payloads,
        job,
    ) = _launch_material(prepared, monkeypatch, ledger_path=ledger_path)
    drifted = copy.deepcopy(payloads)
    for payload in drifted:
        if binding == "input":
            payload["input_bundle_sha256"] = "1" * 64
        else:
            payload["generator_hardware_qualification"][
                "evidence_closed_record_sha256"
            ] = "1" * 64
        payload["closed_record_sha256"] = generation._closed_record_sha256(payload)
    with pytest.raises(
        ValueError, match="remote handoff closure authorization binding"
    ):
        build_databricks_publication_bf16_handoff_submit_payloads(
            job,
            drifted,
            ledger_path=ledger_path,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=remote_q8_authorization,
        )


def test_single_worker_publication_route_is_nonauthorizing(
    prepared, monkeypatch, tmp_path
):
    ledger_path = tmp_path / "ledger.json"
    (
        _plan,
        _config,
        _qualification_record,
        authorization,
        q8_authorization,
        payloads,
        job,
    ) = _launch_material(prepared, monkeypatch, ledger_path=ledger_path)
    submissions = build_databricks_publication_bf16_handoff_submit_payloads(
        job,
        payloads,
        ledger_path=ledger_path,
        qualification_launch_authorization=authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
    )
    with pytest.raises(RuntimeError, match="nonpublication"):
        reserve_publication_bf16_handoff_worker_attempt_json(
            ledger_path,
            submissions[0],
            worker_payload=payloads[0],
            worker_index=0,
            attempt_id="bf16-worker-00",
            job_config=job,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
        )
    assert read_databricks_cluster_hour_ledger_json(ledger_path).reservations == ()


def test_sequential_bf16_reservations_cannot_mint_phase_submission_authority(
    prepared, monkeypatch, tmp_path
):
    ledger_path = tmp_path / "sequential-bf16-ledger.json"
    (
        _plan,
        _config,
        _qualification_record,
        authorization,
        q8_authorization,
        payloads,
        job,
    ) = _launch_material(prepared, monkeypatch, ledger_path=ledger_path)
    opening = read_databricks_cluster_hour_ledger_json(ledger_path)
    submissions = build_databricks_publication_bf16_handoff_submit_payloads(
        job,
        payloads,
        ledger_path=ledger_path,
        qualification_launch_authorization=authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
    )
    requests = []
    for index, submission in enumerate(submissions):
        attempt_id = publication_bf16_handoff_worker_attempt_id(
            payloads[index], worker_index=index
        )
        reserve_databricks_run_attempt_json(
            ledger_path,
            submission,
            attempt_id=attempt_id,
            workload_id=f"publication-bf16-handoff-worker-{index:02d}",
        )
        requests.append(
            DatabricksRunAttemptReservationRequest(
                attempt_id=attempt_id,
                workload_id=f"publication-bf16-handoff-worker-{index:02d}",
                submit_payload=submission,
            )
        )
    replayed = replay_databricks_run_attempt_batch_authorization_json(
        ledger_path,
        tuple(requests),
        expected_predecessor_prefix=databricks_ledger_prefix(opening),
    )
    with pytest.raises(
        TypeError, match="PublicationBF16HandoffSubmissionAuthorization"
    ):
        require_publication_bf16_handoff_submission_authorization(replayed)


def test_bf16_wave_resumes_lost_first_response_and_unclaimed_members(
    prepared, monkeypatch, tmp_path
):
    ledger_path = tmp_path / "bf16-resume-ledger.json"
    (
        _plan,
        _config,
        _qualification_record,
        authorization,
        q8_authorization,
        payloads,
        job,
    ) = _launch_material(prepared, monkeypatch, ledger_path=ledger_path)
    submissions = build_databricks_publication_bf16_handoff_submit_payloads(
        job,
        payloads,
        ledger_path=ledger_path,
        qualification_launch_authorization=authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
    )
    attempts = {
        index: publication_bf16_handoff_worker_attempt_id(
            payloads[index], worker_index=index
        )
        for index in range(16)
    }
    phase_root = tmp_path / "bf16-resume-phase"
    with pytest.raises(TimeoutError, match="lost response"):
        reserve_and_submit_publication_bf16_handoff_worker_wave(
            DatabricksWorkspaceConfig(
                "https://example.cloud.databricks.com",
                "test-token",
            ),
            submissions,
            ledger_path=ledger_path,
            worker_payloads=payloads,
            attempt_ids_by_worker=attempts,
            phase_lease_root=phase_root,
            job_config=job,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
            opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                TimeoutError("lost response")
            ),
        )
    (phase_root / "batch-reserved.json").unlink()
    next_run_id = 80_000

    def opener(_request, timeout):
        nonlocal next_run_id
        assert timeout > 0
        response = JsonHTTPResponse({"run_id": next_run_id})
        next_run_id += 1
        return response

    responses, batch = resume_publication_bf16_handoff_worker_wave(
        DatabricksWorkspaceConfig(
            "https://example.cloud.databricks.com",
            "test-token",
        ),
        submissions,
        ledger_path=ledger_path,
        worker_payloads=payloads,
        attempt_ids_by_worker=attempts,
        phase_lease_root=phase_root,
        job_config=job,
        qualification_launch_authorization=authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
        opener=opener,
    )
    assert len(responses) == 16
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert len(ledger.reservations) == len(ledger.submission_receipts) == 16
    assert (phase_root / "batch-reserved.json").is_file()
    assert batch.durable_output_root == payloads[0]["durable_output_root"]
    alternate_roots = [copy.deepcopy(item) for item in payloads]
    alternate_roots[0]["durable_output_root"] += "-alternate"
    with pytest.raises(ValueError, match="share one durable_output_root"):
        resume_publication_bf16_handoff_worker_wave(
            DatabricksWorkspaceConfig(
                "https://example.cloud.databricks.com", "test-token"
            ),
            submissions,
            ledger_path=ledger_path,
            worker_payloads=alternate_roots,
            attempt_ids_by_worker=attempts,
            phase_lease_root=phase_root,
            job_config=job,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
            opener=lambda *_args, **_kwargs: pytest.fail(
                "alternate BF16 output root must not POST"
            ),
        )


@pytest.mark.parametrize("route", ["reserve", "resume"])
@pytest.mark.parametrize("mutation", ["python_file", "runner_sha256"])
def test_bf16_wave_rejects_post_render_runner_substitution_before_side_effects(
    route, mutation, prepared, monkeypatch, tmp_path
) -> None:
    ledger_path = tmp_path / f"{route}-{mutation}-ledger.json"
    (
        _plan,
        _config,
        _qualification,
        authorization,
        q8_authorization,
        payloads,
        job,
    ) = _launch_material(prepared, monkeypatch, ledger_path=ledger_path)
    submissions = build_databricks_publication_bf16_handoff_submit_payloads(
        job,
        payloads,
        ledger_path=ledger_path,
        qualification_launch_authorization=authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
    )
    attempts = {
        index: publication_bf16_handoff_worker_attempt_id(
            payloads[index], worker_index=index
        )
        for index in range(16)
    }
    phase_root = tmp_path / f"{route}-{mutation}-phase"
    workspace = DatabricksWorkspaceConfig(
        "https://example.cloud.databricks.com",
        "test-token",
    )
    if route == "resume":
        with pytest.raises(TimeoutError, match="initial lost response"):
            reserve_and_submit_publication_bf16_handoff_worker_wave(
                workspace,
                submissions,
                ledger_path=ledger_path,
                worker_payloads=payloads,
                attempt_ids_by_worker=attempts,
                phase_lease_root=phase_root,
                job_config=job,
                qualification_launch_authorization=authorization,
                q8_handoff_remote_closure_authorization=q8_authorization,
                opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    TimeoutError("initial lost response")
                ),
            )
        phase_record = json.loads((phase_root / "phase-lease.json").read_text())
        assert phase_record["record_type"].endswith(".v2")
        workspace_record = json.loads(
            (phase_root / "workspace-authority.json").read_text()
        )
        assert workspace_record["record_type"].endswith(".v1")
        assert set(workspace_record) == {
            "closed_record_sha256",
            "phase_lease_record_sha256",
            "record_type",
            "schema_version",
            "user_name_sha256",
            "workspace_host_sha256",
        }
        assert phase_record["job_config"] == job.to_record()

    mutated = copy.deepcopy(submissions)
    spark_task = mutated[0]["tasks"][0]["spark_python_task"]
    if mutation == "python_file":
        spark_task["python_file"] = "dbfs:/attacker/substituted-runner.py"
        expected_error = "python_file"
    else:
        parameters = spark_task["parameters"]
        parameters[parameters.index("--runner-sha256") + 1] = "0" * 64
        expected_error = "runner SHA-256"
    before = read_databricks_cluster_hour_ledger_json(ledger_path)
    opener_called = False

    def reject_cloud(*_args, **_kwargs):
        nonlocal opener_called
        opener_called = True
        raise AssertionError("runner substitution reached Databricks")

    operation = (
        resume_publication_bf16_handoff_worker_wave
        if route == "resume"
        else reserve_and_submit_publication_bf16_handoff_worker_wave
    )
    with pytest.raises(ValueError, match=expected_error):
        operation(
            workspace,
            mutated,
            ledger_path=ledger_path,
            worker_payloads=payloads,
            attempt_ids_by_worker=attempts,
            phase_lease_root=phase_root,
            job_config=job,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
            opener=reject_cloud,
        )
    assert read_databricks_cluster_hour_ledger_json(ledger_path) == before
    assert opener_called is False
    if route == "reserve":
        assert not phase_root.exists()


@pytest.mark.parametrize("route", ["reserve", "resume"])
def test_bf16_wave_rechecks_runner_after_post_intent_before_cloud(
    route, prepared, monkeypatch, tmp_path
) -> None:
    ledger_path = tmp_path / f"{route}-post-intent-ledger.json"
    (
        _plan,
        _config,
        _qualification,
        authorization,
        q8_authorization,
        payloads,
        job,
    ) = _launch_material(prepared, monkeypatch, ledger_path=ledger_path)
    submissions = build_databricks_publication_bf16_handoff_submit_payloads(
        job,
        payloads,
        ledger_path=ledger_path,
        qualification_launch_authorization=authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
    )
    attempts = {
        index: publication_bf16_handoff_worker_attempt_id(
            payloads[index], worker_index=index
        )
        for index in range(16)
    }
    phase_root = tmp_path / f"{route}-post-intent-phase"
    workspace = DatabricksWorkspaceConfig(
        "https://example.cloud.databricks.com",
        "test-token",
    )
    if route == "resume":
        with pytest.raises(TimeoutError, match="initial lost response"):
            reserve_and_submit_publication_bf16_handoff_worker_wave(
                workspace,
                submissions,
                ledger_path=ledger_path,
                worker_payloads=payloads,
                attempt_ids_by_worker=attempts,
                phase_lease_root=phase_root,
                job_config=job,
                qualification_launch_authorization=authorization,
                q8_handoff_remote_closure_authorization=q8_authorization,
                opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    TimeoutError("initial lost response")
                ),
            )

    active_submissions = copy.deepcopy(submissions)
    parameters = active_submissions[0]["tasks"][0]["spark_python_task"]["parameters"]
    mutation_applied = False

    def mutate_runner_sha256() -> None:
        nonlocal mutation_applied
        parameters[parameters.index("--runner-sha256") + 1] = "0" * 64
        mutation_applied = True

    if route == "reserve":
        original_write = generation._write_json_exclusive

        def write_then_mutate(value, path):
            original_write(value, path)
            if Path(path).name == "worker-00.post-intent.json":
                mutate_runner_sha256()

        monkeypatch.setattr(generation, "_write_json_exclusive", write_then_mutate)
    else:
        original_read = generation._read_canonical_json_file

        def read_then_mutate(path, description):
            value = original_read(path, description)
            if Path(path).name == "worker-00.post-intent.json":
                mutate_runner_sha256()
            return value

        monkeypatch.setattr(generation, "_read_canonical_json_file", read_then_mutate)

    opener_called = False

    def reject_cloud(*_args, **_kwargs):
        nonlocal opener_called
        opener_called = True
        raise AssertionError("post-intent runner substitution reached Databricks")

    operation = (
        resume_publication_bf16_handoff_worker_wave
        if route == "resume"
        else reserve_and_submit_publication_bf16_handoff_worker_wave
    )
    with pytest.raises(ValueError, match="runner SHA-256"):
        operation(
            workspace,
            active_submissions,
            ledger_path=ledger_path,
            worker_payloads=payloads,
            attempt_ids_by_worker=attempts,
            phase_lease_root=phase_root,
            job_config=job,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
            opener=reject_cloud,
        )
    assert mutation_applied is True
    assert opener_called is False
    assert (phase_root / "worker-00.post-intent.json").is_file()
    assert not (phase_root / "worker-00.receipt.json").exists()


def test_raw_local_manifest_or_generation_result_cannot_authorize_serving(tmp_path):
    local_manifest = {
        "closed_record_sha256": "",
        "context_tokens": 16384,
        "record_type": "cachet.publication_handoff_bundle.v1",
    }
    local_manifest["closed_record_sha256"] = generation._closed_record_sha256(
        local_manifest
    )
    with pytest.raises(
        TypeError,
        match="PublicationBF16HandoffServingAuthorization",
    ):
        resolve_publication_bf16_handoff_bundle(local_manifest)
    raw_result = PublicationBF16HandoffGenerationResult(
        root=tmp_path,
        source_root=tmp_path / "source",
        manifest_path=tmp_path / "manifest.json",
        execution_record_path=tmp_path / "execution.json",
        manifest=local_manifest,
        record={"closed_record_sha256": "0" * 64},
    )
    with pytest.raises(
        TypeError,
        match="PublicationBF16HandoffServingAuthorization",
    ):
        resolve_publication_bf16_handoff_bundle(raw_result)


def test_post_close_replay_rejects_bf16_raw_evidence_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bf16-closed"
    source_root = root / "bundles" / f"16384-{'a' * 64}"
    source_root.mkdir(parents=True)
    (root / "manifests").mkdir()
    (root / "worker-records").mkdir()
    (root / "worker-results").mkdir()
    attestation_root = root / generation.PUBLICATION_BF16_HANDOFF_ATTESTATION_DIRECTORY
    attestation_root.mkdir()
    manifest_path = (
        root / "manifests" / generation.PUBLICATION_BF16_HANDOFF_MANIFEST_FILENAME
    )
    manifest_path.write_bytes(b"synthetic manifest\n")
    config = build_publication_bf16_handoff_execution_config(
        vllm_bitsandbytes_loader_source_sha256=(
            GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
        )
    )
    contract = generation._execution_config_record(config)
    segment_contract = {
        "chunk_id": "chunk",
        "token_count": 10,
        "token_ids_digest": "0" * 64,
    }
    identities = [
        (dataset, f"{dataset}-example-{index:02d}")
        for dataset in SUPPORTED_V1_DATASETS
        for index in range(32)
    ]
    assignments = [identities[index::16] for index in range(16)]
    plan_workers = []
    all_rows: list[dict] = []
    manifest_files: list[dict] = []
    manifest_entries: dict[str, list[dict]] = {
        dataset: [] for dataset in SUPPORTED_V1_DATASETS
    }
    qualification = {"evidence_closed_record_sha256": "6" * 64}
    observed_hardware = {"gpu_model": "NVIDIA L40S"}
    worker_results: list[dict] = []
    for worker_index, assigned in enumerate(assignments):
        items = []
        rows = []
        handoff_relative = f"worker-{worker_index:02d}/handoff.json"
        payload_relative = f"worker-{worker_index:02d}/payload.bin"
        handoff_path = source_root / handoff_relative
        payload_path = source_root / payload_relative
        handoff_path.parent.mkdir()
        handoff_path.write_bytes(f"handoff-{worker_index:02d}".encode())
        payload_path.write_bytes(f"payload-{worker_index:02d}".encode())
        for role, relative_name, path in (
            ("handoff_json", handoff_relative, handoff_path),
            ("payload", payload_relative, payload_path),
        ):
            manifest_files.append(
                {
                    "byte_count": path.stat().st_size,
                    "relative_name": relative_name,
                    "role": role,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
            )
        for dataset, example_id in assigned:
            task_id = f"task-{dataset}-{example_id}"
            items.append(
                {
                    "cache_prefix_tokens": 10,
                    "context_tokens": 16_384,
                    "dataset": dataset,
                    "example_id": example_id,
                    "input_token_slots": 10,
                    "segment_token_contracts": [segment_contract],
                    "task_id": task_id,
                }
            )
            row = {
                "arm_kv_transfer_params": {
                    generation.PUBLICATION_BF16_HANDOFF_ARM_ID: {
                        generation.DOCUMENT_KV_HANDOFF_JSON_PARAM: (
                            f"{root}/pending/16384/{handoff_relative}"
                        ),
                        generation.DOCUMENT_KV_PAYLOAD_URI_PARAM: (
                            f"{root}/pending/16384/{payload_relative}"
                        ),
                    }
                },
                "dataset": dataset,
                "example_id": example_id,
            }
            rows.append(row)
            all_rows.append(row)
            manifest_entries[dataset].append(
                {
                    "example_id": example_id,
                    "handoff_relative_name": handoff_relative,
                    "payload_relative_name": payload_relative,
                }
            )
        worker_id = f"bf16-worker-{worker_index:02d}"
        plan_workers.append(
            {
                "items": items,
                "worker_id": worker_id,
                "worker_index": worker_index,
            }
        )
        record_path = root / "worker-records" / f"worker-{worker_index:02d}.jsonl"
        generation._write_jsonl_exclusive(rows, record_path)
        bundle_files = []
        for relative_name, path in (
            (handoff_relative, handoff_path),
            (payload_relative, payload_path),
        ):
            bundle_files.append(
                {
                    "byte_count": path.stat().st_size,
                    "relative_path": f"pending/16384/{relative_name}",
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
            )
        task_ids = sorted(item["task_id"] for item in items)
        worker_result = {
            "accounting": {
                "cache_prefix_tokens": 80,
                "durable_byte_count": (
                    sum(item["byte_count"] for item in bundle_files)
                    + record_path.stat().st_size
                ),
                "durable_sync_seconds": 0.01,
                "generation_seconds": 0.02,
                "includes_generation_payload_hash_and_durable_sync": True,
                "input_token_slots": 80,
                "producer_metered_seconds": 0.05,
            },
            "bundle_files": bundle_files,
            "bundle_files_sha256": generation._canonical_sha256(bundle_files),
            "closed_record_sha256": "",
            "execution_contract": contract,
            "execution_mode": PUBLICATION_BF16_HANDOFF_EXECUTION_MODE,
            "generator_hardware": {
                "observed": observed_hardware,
                "qualification": qualification,
            },
            "input_bundle_sha256": "1" * 64,
            "partial_record_file": {
                "byte_count": record_path.stat().st_size,
                "record_count": len(rows),
                "relative_path": f"worker-records/worker-{worker_index:02d}.jsonl",
                "sha256": sha256(record_path.read_bytes()).hexdigest(),
            },
            "plan_closed_record_sha256": "2" * 64,
            "record_type": generation.PUBLICATION_BF16_HANDOFF_WORKER_RESULT_RECORD_TYPE,
            "schema_version": generation.PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION,
            "task_ids": task_ids,
            "task_ids_sha256": generation._canonical_sha256(task_ids),
            "worker_id": worker_id,
            "worker_index": worker_index,
        }
        worker_result["closed_record_sha256"] = generation._closed_record_sha256(
            worker_result
        )
        generation._write_json_exclusive(
            worker_result,
            root / "worker-results" / f"worker-{worker_index:02d}.json",
        )
        worker_results.append(worker_result)
        (attestation_root / f"worker-{worker_index:02d}.json").write_bytes(
            f"attestation-{worker_index:02d}".encode()
        )

    dataset_records = []
    for dataset in SUPPORTED_V1_DATASETS:
        rows = sorted(
            (row for row in all_rows if row["dataset"] == dataset),
            key=lambda row: row["example_id"],
        )
        relative_name = f"datasets/{dataset}.jsonl"
        dataset_path = source_root / relative_name
        generation._write_jsonl_exclusive(rows, dataset_path)
        manifest_files.append(
            {
                "byte_count": dataset_path.stat().st_size,
                "relative_name": relative_name,
                "role": "enriched_dataset",
                "sha256": sha256(dataset_path.read_bytes()).hexdigest(),
            }
        )
        dataset_records.append(
            {
                "dataset": dataset,
                "entries": manifest_entries[dataset],
                "relative_name": relative_name,
            }
        )
    manifest_files.sort(key=lambda item: item["relative_name"])
    manifest = {"datasets": dataset_records, "files": manifest_files}
    plan = {
        "closed_record_sha256": "2" * 64,
        "coverage": {
            "cache_prefix_generation_tokens": 1280,
            "input_token_slots": 1280,
        },
        "input_bundle_sha256": "1" * 64,
        "workers": plan_workers,
    }
    ledger = DatabricksClusterHourLedger(ledger_id="bf16-ledger")
    prefix = databricks_ledger_prefix(ledger)
    attempts = [
        {
            "actual_gpu_duration_seconds": 1.0,
            "attempt_id": f"attempt-{index:02d}",
            "worker_index": index,
        }
        for index in range(16)
    ]
    reconciliation = {
        "attempt_count": 16,
        "attempts": attempts,
        "attempts_sha256": generation._canonical_sha256(attempts),
        "cluster_ids_sha256": "3" * 64,
        "ledger_id": ledger.ledger_id,
        "parent_run_ids_sha256": "4" * 64,
        "task_run_ids_sha256": "5" * 64,
        "verification_source": "direct_databricks_runs_get",
    }
    authorizations = {}
    for index, worker_result in enumerate(worker_results):
        attestation_path = attestation_root / f"worker-{index:02d}.json"
        binding = generation.PublicationBF16HandoffAttestationBinding(
            worker_index=index,
            path=attestation_path,
            file_sha256=sha256(attestation_path.read_bytes()).hexdigest(),
            closed_record_sha256=sha256(f"closed-{index}".encode()).hexdigest(),
        )
        authorizations[index] = generation.PublicationBF16HandoffWorkerAuthorization(
            binding=binding,
            attempt_id=f"attempt-{index:02d}",
            ledger_id=ledger.ledger_id,
            ledger_path_sha256="7" * 64,
            producer_batch_prefix=prefix,
            control_plane_status_sha256="8" * 64,
            _issuer=generation._WORKER_AUTHORIZATION_ISSUER,
        )
    execution_workers = [
        {
            "charged_gpu_seconds": 1.0,
            "producer_metered_seconds": 0.05,
            "result_closed_record_sha256": worker_results[index][
                "closed_record_sha256"
            ],
            "result_relative_path": f"worker-results/worker-{index:02d}.json",
            "worker_index": index,
        }
        for index in range(16)
    ]
    durable_paths = [
        path
        for directory in (
            root / "bundles",
            root / "manifests",
            root / "worker-records",
            root / "worker-results",
            attestation_root,
        )
        for path in directory.rglob("*")
        if path.is_file()
    ]
    execution = {
        "accounting": {
            "charged_gpu_hours": 16.0 / 3600.0,
            "charged_gpu_seconds": 16.0,
            "coordinator_gpu_hours": 0.0,
            "coordinator_wall_seconds": 0.1,
            "cost_model": "sum_independent_one_gpu_worker_terminal_lifecycles",
            "durable_byte_count": sum(path.stat().st_size for path in durable_paths),
            "end_to_end_cache_prefix_tokens_per_gpu_second": 80.0,
            "end_to_end_input_token_slots_per_gpu_second": 80.0,
            "end_to_end_wall_seconds": 1.0,
            "full_launch_min_tokens_per_gpu_second": 35.0,
            "full_launch_throughput_gate_passed": True,
            "includes_bootstrap_generation_hash_and_durable_write": True,
            "payload_copy_count_during_closure": 0,
            "worker_count": 16,
        },
        "bundle": {
            "closed_record_sha256": "9" * 64,
            "context_tokens": 16_384,
            "manifest_file_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
            "manifest_relative_path": (
                f"manifests/{generation.PUBLICATION_BF16_HANDOFF_MANIFEST_FILENAME}"
            ),
            "portable_bundle_sha256": "a" * 64,
            "source_root_relative_path": f"bundles/16384-{'a' * 64}",
        },
        "closed_record_sha256": "",
        "coverage": {
            "cache_prefix_generation_tokens": 1280,
            "dataset_count": len(SUPPORTED_V1_DATASETS),
            "examples_per_dataset": 32,
            "generated_task_count": 128,
            "input_token_slots": 1280,
            "task_ids_sha256": generation._canonical_sha256(
                sorted(
                    item["task_id"]
                    for worker in plan_workers
                    for item in worker["items"]
                )
            ),
        },
        "execution_contract": contract,
        "execution_mode": PUBLICATION_BF16_HANDOFF_EXECUTION_MODE,
        "generator_hardware": {
            "hardware_target": "aws-g6e-l40s",
            "node_type_id": "g6e.4xlarge",
            "observed_hardware_identity_sha256": [
                generation._canonical_sha256(observed_hardware)
            ],
            "qualification_closed_record_sha256": "6" * 64,
        },
        "input_bundle_sha256": "1" * 64,
        "ledger_reconciliation": reconciliation,
        "plan_closed_record_sha256": "2" * 64,
        "record_type": generation.PUBLICATION_BF16_HANDOFF_EXECUTION_RECORD_TYPE,
        "schema_version": generation.PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION,
        "serving_reuse": {
            "regenerate_inside_timed_serving_jobs": False,
            "required_action": "validate_manifest_then_stage_content_addressed_bundle",
            "stage_target": "node_local_nvme",
        },
        "workers": execution_workers,
    }
    execution["closed_record_sha256"] = generation._closed_record_sha256(execution)
    execution_path = root / generation.PUBLICATION_BF16_HANDOFF_EXECUTION_FILENAME
    state = {
        "result": PublicationBF16HandoffGenerationResult(
            root=root,
            source_root=source_root,
            manifest_path=manifest_path,
            execution_record_path=execution_path,
            manifest=manifest,
            record=execution,
        )
    }
    monkeypatch.setattr(
        generation,
        "validate_publication_bf16_handoff_generation_plan",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation,
        "read_publication_bf16_handoff_generation_result",
        lambda _root: state["result"],
    )
    monkeypatch.setattr(
        generation,
        "read_publication_latency_handoff_bundle",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        generation,
        "_ledger_reconciliation",
        lambda *_args, **_kwargs: reconciliation,
    )
    monkeypatch.setattr(generation, "_validate_qualification_record", lambda _r: None)
    monkeypatch.setattr(generation, "_validate_observed_l40s_hardware", lambda _r: None)
    monkeypatch.setattr(
        generation, "_verify_bound_hardware_qualification_file", lambda _r: None
    )
    monkeypatch.setattr(generation, "_handoff_total_tokens", lambda _row: 10)
    monkeypatch.setattr(
        generation,
        "_handoff_segment_token_contracts",
        lambda _row: [segment_contract],
    )

    def read_attestation(binding, *, durable_output_root):
        del durable_output_root
        worker_result = worker_results[binding.worker_index]
        result_path = (
            root / "worker-results" / f"worker-{binding.worker_index:02d}.json"
        )
        return {
            "worker_result": {
                "closed_record_sha256": worker_result["closed_record_sha256"],
                "file_sha256": sha256(result_path.read_bytes()).hexdigest(),
            }
        }

    monkeypatch.setattr(
        generation, "read_publication_bf16_handoff_attestation", read_attestation
    )

    def replay():
        return generation._replay_closed_publication_bf16_handoff_generation(
            plan,
            prepared_input_dir=tmp_path / "prepared",
            durable_output_root=root,
            tokenizer=CharacterTokenizer(),
            config=config,
            ledger_snapshot=ledger,
            ledger_path_sha256="7" * 64,
            expected_producer_batch_prefix=prefix,
            attempt_ids_by_worker={
                index: f"attempt-{index:02d}" for index in range(16)
            },
            worker_authorizations=authorizations,
            _issuer=generation._POST_CLOSE_REPLAY_ISSUER,
        )

    assert replay().record == execution
    record_path = root / "worker-records" / "worker-00.jsonl"
    record_bytes = record_path.read_bytes()
    record_path.write_bytes(bytes([record_bytes[0] ^ 1]) + record_bytes[1:])
    with pytest.raises(ValueError, match="durable file binding drift"):
        replay()
    record_path.write_bytes(record_bytes)
    record_path.unlink()
    with pytest.raises(ValueError, match="durable file binding drift"):
        replay()
    record_path.write_bytes(record_bytes)
    extra_path = root / "worker-records" / "unexpected.jsonl"
    extra_path.write_bytes(record_bytes)
    with pytest.raises(ValueError, match="inventory drift"):
        replay()
    extra_path.unlink()
    resealed_execution = copy.deepcopy(execution)
    resealed_execution["unreviewed_resealed_field"] = True
    resealed_execution["closed_record_sha256"] = generation._closed_record_sha256(
        resealed_execution
    )
    state["result"] = PublicationBF16HandoffGenerationResult(
        root=root,
        source_root=source_root,
        manifest_path=manifest_path,
        execution_record_path=execution_path,
        manifest=manifest,
        record=resealed_execution,
    )
    with pytest.raises(ValueError, match="closed schema"):
        replay()


def test_serving_authorization_requires_q8_predecessor_authority(
    prepared,
    monkeypatch,
    tmp_path,
):
    qualification = _qualification(monkeypatch, prepared)
    ledger_path = tmp_path / "qualification-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="gpu-qualification-test-ledger",
    )
    authorization = _authorization(qualification, ledger_path=ledger_path)
    execution_path = tmp_path / "execution.json"
    manifest_path = tmp_path / "manifest.json"
    execution_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")
    reconciliation = {"ledger_id": authorization.ledger_id}
    result = PublicationBF16HandoffGenerationResult(
        root=tmp_path,
        source_root=tmp_path / "source",
        manifest_path=manifest_path,
        execution_record_path=execution_path,
        manifest={"closed_record_sha256": "2" * 64},
        record={
            "closed_record_sha256": "1" * 64,
            "generator_hardware": {
                "qualification_closed_record_sha256": (
                    authorization.evidence_closed_record_sha256
                )
            },
            "ledger_reconciliation": reconciliation,
        },
    )
    monkeypatch.setattr(
        generation,
        "read_publication_bf16_handoff_generation_result",
        lambda _root: result,
    )
    monkeypatch.setattr(
        generation,
        "_ledger_reconciliation",
        lambda *_args, **_kwargs: reconciliation,
    )
    with pytest.raises(TypeError, match="PublicationHandoffRemoteClosureAuthorization"):
        authorize_publication_bf16_handoff_serving(
            result,
            ledger_path=ledger_path,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=object(),
            submission_authorization=object(),
            attempt_ids_by_worker={},
            worker_authorizations={},
        )


def test_valid_bf16_serving_authority_resolves_with_genuine_phase_authorities(
    prepared, monkeypatch, tmp_path
):
    ledger_path = tmp_path / "serving-ledger.json"
    (
        _plan,
        _config,
        _qualification_record,
        qualification_authorization,
        q8_authorization,
        payloads,
        job,
    ) = _launch_material(prepared, monkeypatch, ledger_path=ledger_path)
    submissions = build_databricks_publication_bf16_handoff_submit_payloads(
        job,
        payloads,
        ledger_path=ledger_path,
        qualification_launch_authorization=qualification_authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
    )
    attempts = {
        index: publication_bf16_handoff_worker_attempt_id(
            payloads[index], worker_index=index
        )
        for index in range(16)
    }
    next_run = 50_000

    def submit(_request, timeout):
        nonlocal next_run
        assert timeout > 0
        response = JsonHTTPResponse({"run_id": next_run})
        next_run += 1
        return response

    _responses, submission_authorization = (
        reserve_and_submit_publication_bf16_handoff_worker_wave(
            DatabricksWorkspaceConfig("https://example.cloud.databricks.com", "token"),
            submissions,
            ledger_path=ledger_path,
            worker_payloads=payloads,
            attempt_ids_by_worker=attempts,
            phase_lease_root=tmp_path / "serving-phase",
            job_config=job,
            qualification_launch_authorization=qualification_authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
            opener=submit,
        )
    )
    live = read_databricks_cluster_hour_ledger_json(ledger_path)
    reservations = {item.attempt_id: item for item in live.reservations}
    reconciliation_attempts = [
        {
            "actual_gpu_duration_seconds": 1.0,
            "attempt_id": attempts[index],
            "attestation_closed_record_sha256": sha256(
                f"attestation-{index}".encode()
            ).hexdigest(),
            "cluster_id": f"cluster-{index}",
            "parent_run_id": str(50_000 + index),
            "submit_payload_sha256": reservations[
                attempts[index]
            ].submit_payload_sha256,
            "task_run_id": str(60_000 + index),
            "worker_index": index,
        }
        for index in range(16)
    ]
    reconciliation = {
        "attempt_count": 16,
        "attempts": reconciliation_attempts,
        "attempts_sha256": generation._canonical_sha256(reconciliation_attempts),
        "cluster_ids_sha256": "3" * 64,
        "ledger_id": live.ledger_id,
        "parent_run_ids_sha256": "4" * 64,
        "task_run_ids_sha256": "5" * 64,
        "verification_source": "direct_databricks_runs_get",
    }
    root = tmp_path / "serving-result"
    root.mkdir()
    execution_path = root / generation.PUBLICATION_BF16_HANDOFF_EXECUTION_FILENAME
    manifest_path = root / "manifest.json"
    execution_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")
    result = PublicationBF16HandoffGenerationResult(
        root=root,
        source_root=root / "bundle",
        manifest_path=manifest_path,
        execution_record_path=execution_path,
        manifest={"closed_record_sha256": "2" * 64},
        record={
            "closed_record_sha256": "1" * 64,
            "input_bundle_sha256": payloads[0]["input_bundle_sha256"],
            "generator_hardware": {
                "qualification_closed_record_sha256": (
                    qualification_authorization.evidence_closed_record_sha256
                )
            },
            "ledger_reconciliation": reconciliation,
        },
    )
    workers = {}
    for index in range(16):
        attestation_path = root / f"attestation-{index:02d}.json"
        attestation_path.write_text("{}\n", encoding="utf-8")
        workers[index] = generation.PublicationBF16HandoffWorkerAuthorization(
            binding=generation.PublicationBF16HandoffAttestationBinding(
                worker_index=index,
                path=attestation_path,
                file_sha256=sha256(attestation_path.read_bytes()).hexdigest(),
                closed_record_sha256=reconciliation_attempts[index][
                    "attestation_closed_record_sha256"
                ],
            ),
            attempt_id=attempts[index],
            ledger_id=live.ledger_id,
            ledger_path_sha256=(
                submission_authorization.batch_authorization.ledger_path_sha256
            ),
            producer_batch_prefix=(
                submission_authorization.batch_authorization.batch_prefix
            ),
            control_plane_status_sha256=sha256(f"control-{index}".encode()).hexdigest(),
            workspace_host_sha256=submission_authorization.workspace_host_sha256,
            user_name_sha256=submission_authorization.user_name_sha256,
            _issuer=generation._WORKER_AUTHORIZATION_ISSUER,
        )
    monkeypatch.setattr(
        generation,
        "read_publication_bf16_handoff_generation_result",
        lambda _root: result,
    )
    monkeypatch.setattr(
        generation,
        "_require_submission_q8_remote_closure",
        lambda *_args, **_kwargs: submission_authorization.batch_authorization,
    )
    monkeypatch.setattr(
        generation,
        "_require_matching_predecessor_ledger",
        lambda *_args, **_kwargs: live,
    )
    monkeypatch.setattr(
        generation, "require_databricks_ledger_prefix", lambda *_args: None
    )
    monkeypatch.setattr(
        generation,
        "require_databricks_batch_terminal_closure",
        lambda *_args, **_kwargs: (
            submission_authorization.batch_authorization.batch_prefix
        ),
    )
    monkeypatch.setattr(
        generation, "_ledger_reconciliation", lambda *_args, **_kwargs: reconciliation
    )
    serving = authorize_publication_bf16_handoff_serving(
        result,
        ledger_path=ledger_path,
        qualification_launch_authorization=qualification_authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
        submission_authorization=submission_authorization,
        attempt_ids_by_worker=attempts,
        worker_authorizations=workers,
    )
    assert resolve_publication_bf16_handoff_bundle(serving) == result


@pytest.mark.parametrize("duplicate_cluster", [False, True])
def test_direct_runs_get_attestations_close_all_16_unique_jobs(
    prepared,
    monkeypatch,
    tmp_path,
    duplicate_cluster,
):
    ledger_path = tmp_path / "ledger.json"
    (
        _plan,
        _config,
        _qualification_record,
        authorization,
        q8_authorization,
        payloads,
        job,
    ) = _launch_material(prepared, monkeypatch, ledger_path=ledger_path)
    submissions = build_databricks_publication_bf16_handoff_submit_payloads(
        job,
        payloads,
        ledger_path=ledger_path,
        qualification_launch_authorization=authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
    )
    durable_root = tmp_path / "durable"
    (durable_root / "worker-results").mkdir(parents=True)
    terminals = {}
    attempts = {
        index: publication_bf16_handoff_worker_attempt_id(
            payloads[index], worker_index=index
        )
        for index in range(16)
    }
    next_run_id = 10_000

    def opener(_request, timeout):
        nonlocal next_run_id
        assert timeout > 0
        response = JsonHTTPResponse({"run_id": next_run_id})
        next_run_id += 1
        return response

    submit_responses, batch_authorization = (
        reserve_and_submit_publication_bf16_handoff_worker_wave(
            DatabricksWorkspaceConfig(
                "https://example.cloud.databricks.com",
                "test-token",
            ),
            submissions,
            ledger_path=ledger_path,
            worker_payloads=payloads,
            attempt_ids_by_worker=attempts,
            phase_lease_root=tmp_path / "bf16-phase-lease",
            job_config=job,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
            opener=opener,
        )
    )
    assert len(read_databricks_cluster_hour_ledger_json(ledger_path).reservations) == 16
    (tmp_path / "bf16-phase-lease" / "batch-reserved.json").unlink()
    resumed, replayed_batch = resume_publication_bf16_handoff_worker_wave(
        DatabricksWorkspaceConfig(
            "https://example.cloud.databricks.com",
            "test-token",
        ),
        submissions,
        ledger_path=ledger_path,
        worker_payloads=payloads,
        attempt_ids_by_worker=attempts,
        phase_lease_root=tmp_path / "bf16-phase-lease",
        job_config=job,
        qualification_launch_authorization=authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
        opener=lambda *_args, **_kwargs: pytest.fail("completed wave must not POST"),
    )
    assert len(resumed) == 16
    assert (
        replayed_batch.batch_authorization.batch_prefix
        == batch_authorization.batch_authorization.batch_prefix
    )
    assert (tmp_path / "bf16-phase-lease" / "batch-reserved.json").is_file()
    mounted_output_root = batch_authorization.durable_output_root
    original_local_path = generation.local_path
    monkeypatch.setattr(
        generation,
        "local_path",
        lambda value: (
            str(durable_root)
            if str(value).rstrip("/") == mounted_output_root
            else original_local_path(value)
        ),
    )
    for index, submit_payload in enumerate(submissions):
        worker_result = {
            "closed_record_sha256": "",
            "execution_mode": PUBLICATION_BF16_HANDOFF_EXECUTION_MODE,
            "generator_hardware": {
                "qualification": copy.deepcopy(
                    payloads[index]["generator_hardware_qualification"]
                )
            },
            "record_type": generation.PUBLICATION_BF16_HANDOFF_WORKER_RESULT_RECORD_TYPE,
            "schema_version": generation.PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION,
            "worker_index": index,
        }
        worker_result["closed_record_sha256"] = generation._closed_record_sha256(
            worker_result
        )
        generation._write_json_exclusive(
            worker_result,
            durable_root / "worker-results" / f"worker-{index:02d}.json",
        )
        cluster_index = 14 if duplicate_cluster and index == 15 else index
        terminals[str(10_000 + index)] = _terminal_run(
            submit_payload,
            index,
            cluster_index=cluster_index,
        )
    monkeypatch.setattr(
        generation,
        "get_databricks_run",
        lambda _workspace, run_id: terminals[str(run_id)],
    )
    workspace = DatabricksWorkspaceConfig(
        host="https://example.cloud.databricks.com",
        token="test-token",
    )
    monkeypatch.setattr(
        generation,
        "require_databricks_current_user_name",
        lambda current_workspace, *, expected_user_name, opener=None: {
            "workspace_host_sha256": sha256(
                current_workspace.normalized_host.encode("utf-8")
            ).hexdigest(),
            "user_name_sha256": sha256(expected_user_name.encode("utf-8")).hexdigest(),
        },
    )
    original_terminal = terminals["10000"]
    for mutation in ("python_file", "ordered_parameters"):
        substituted_terminal = copy.deepcopy(original_terminal)
        observed_python_task = substituted_terminal["tasks"][0]["spark_python_task"]
        if mutation == "python_file":
            observed_python_task["python_file"] = "dbfs:/attacker.py"
        else:
            parameters = observed_python_task["parameters"]
            parameters[0], parameters[1] = parameters[1], parameters[0]
        terminals["10000"] = substituted_terminal
        with pytest.raises(ValueError, match="spark_python_task differs"):
            collect_publication_bf16_handoff_worker_attestation(
                workspace,
                submissions[0],
                submit_responses[0],
                ledger_path=ledger_path,
                durable_output_root=mounted_output_root,
                worker_index=0,
                attempt_id=attempts[0],
                q8_handoff_remote_closure_authorization=q8_authorization,
                submission_authorization=batch_authorization,
            )
    terminals["10000"] = original_terminal
    bindings = {}
    for index, submit_payload in enumerate(submissions):
        bindings[index] = collect_publication_bf16_handoff_worker_attestation(
            workspace,
            submit_payload,
            submit_responses[index],
            ledger_path=ledger_path,
            durable_output_root=mounted_output_root,
            worker_index=index,
            attempt_id=attempts[index],
            q8_handoff_remote_closure_authorization=q8_authorization,
            submission_authorization=batch_authorization,
        )
    bindings[0] = collect_publication_bf16_handoff_worker_attestation(
        workspace,
        submissions[0],
        submit_responses[0],
        ledger_path=ledger_path,
        durable_output_root=mounted_output_root,
        worker_index=0,
        attempt_id=attempts[0],
        q8_handoff_remote_closure_authorization=q8_authorization,
        submission_authorization=batch_authorization,
    )
    if duplicate_cluster:
        with pytest.raises(ValueError, match="not globally unique"):
            publication_bf16_handoff_terminal_actual_gpu_seconds_from_ledger(
                ledger_path,
                attempt_ids_by_worker=attempts,
                durable_output_root=mounted_output_root,
                worker_authorizations=bindings,
            )
    else:
        actuals = publication_bf16_handoff_terminal_actual_gpu_seconds_from_ledger(
            ledger_path,
            attempt_ids_by_worker=attempts,
            durable_output_root=mounted_output_root,
            worker_authorizations=bindings,
        )
        assert actuals == {index: 1.0 for index in range(16)}

        monkeypatch.setattr(generation, "local_path", original_local_path)
        mirror = tmp_path / "bf16-controller-mirror"
        uploaded = {}
        events = []
        tamper = {"result": False, "attestation": False}

        def mirror_download(_workspace, uri):
            if "/worker-results/" in uri:
                events.append("worker-result/get")
                if tamper["result"]:
                    return b"tampered remote result\n"
                return (durable_root / "worker-results" / "worker-00.json").read_bytes()
            events.append("attestation/get")
            if tamper["attestation"]:
                return b"tampered remote attestation\n"
            return uploaded[uri]

        def mirror_upload(_workspace, uri, path, **_kwargs):
            events.append("attestation/put")
            uploaded[uri] = Path(path).read_bytes()
            return {}

        monkeypatch.setattr(
            generation, "download_databricks_volume_file_bytes", mirror_download
        )
        monkeypatch.setattr(
            generation, "upload_databricks_volume_file_path_exclusive", mirror_upload
        )
        mirrored = collect_publication_bf16_handoff_worker_attestation(
            workspace,
            submissions[0],
            submit_responses[0],
            ledger_path=ledger_path,
            durable_output_root=batch_authorization.durable_output_root,
            worker_index=0,
            attempt_id=attempts[0],
            q8_handoff_remote_closure_authorization=q8_authorization,
            submission_authorization=batch_authorization,
            local_evidence_mirror_root=mirror,
        )
        assert str(mirrored.binding.path).startswith("/dbfs/Volumes/")
        with pytest.raises(
            ValueError, match="workspace/principal differs from submission"
        ):
            collect_publication_bf16_handoff_worker_attestation(
                DatabricksWorkspaceConfig("https://other-workspace.example", "token"),
                submissions[0],
                submit_responses[0],
                ledger_path=ledger_path,
                durable_output_root=batch_authorization.durable_output_root,
                worker_index=0,
                attempt_id=attempts[0],
                q8_handoff_remote_closure_authorization=q8_authorization,
                submission_authorization=batch_authorization,
                local_evidence_mirror_root=mirror,
            )
        monkeypatch.setattr(
            generation,
            "require_databricks_current_user_name",
            lambda _workspace, *, expected_user_name, opener=None: {
                "workspace_host_sha256": mirrored.workspace_host_sha256,
                "user_name_sha256": sha256(b"other-principal@example.com").hexdigest(),
            },
        )
        with pytest.raises(
            ValueError, match="workspace/principal differs from submission"
        ):
            collect_publication_bf16_handoff_worker_attestation(
                workspace,
                submissions[0],
                submit_responses[0],
                ledger_path=ledger_path,
                durable_output_root=batch_authorization.durable_output_root,
                worker_index=0,
                attempt_id=attempts[0],
                q8_handoff_remote_closure_authorization=q8_authorization,
                submission_authorization=batch_authorization,
                local_evidence_mirror_root=mirror,
            )
        monkeypatch.setattr(
            generation,
            "require_databricks_current_user_name",
            lambda current_workspace, *, expected_user_name, opener=None: {
                "workspace_host_sha256": sha256(
                    current_workspace.normalized_host.encode("utf-8")
                ).hexdigest(),
                "user_name_sha256": sha256(
                    expected_user_name.encode("utf-8")
                ).hexdigest(),
            },
        )
        replayed_mirrored = collect_publication_bf16_handoff_worker_attestation(
            workspace,
            submissions[0],
            submit_responses[0],
            ledger_path=ledger_path,
            durable_output_root=batch_authorization.durable_output_root,
            worker_index=0,
            attempt_id=attempts[0],
            q8_handoff_remote_closure_authorization=q8_authorization,
            submission_authorization=batch_authorization,
            local_evidence_mirror_root=mirror,
        )
        assert replayed_mirrored.binding == mirrored.binding
        tamper["result"] = True
        with pytest.raises(ValueError, match="not valid JSON"):
            collect_publication_bf16_handoff_worker_attestation(
                workspace,
                submissions[0],
                submit_responses[0],
                ledger_path=ledger_path,
                durable_output_root=batch_authorization.durable_output_root,
                worker_index=0,
                attempt_id=attempts[0],
                q8_handoff_remote_closure_authorization=q8_authorization,
                submission_authorization=batch_authorization,
                local_evidence_mirror_root=mirror,
            )
        tamper["result"] = False
        tamper["attestation"] = True
        with pytest.raises(RuntimeError, match="readback differs"):
            collect_publication_bf16_handoff_worker_attestation(
                workspace,
                submissions[0],
                submit_responses[0],
                ledger_path=ledger_path,
                durable_output_root=batch_authorization.durable_output_root,
                worker_index=0,
                attempt_id=attempts[0],
                q8_handoff_remote_closure_authorization=q8_authorization,
                submission_authorization=batch_authorization,
                local_evidence_mirror_root=mirror,
            )


def test_bf16_local_evidence_mirror_secure_replay_and_tamper(tmp_path):
    mirror = tmp_path / "bf16-mirror"
    root = generation._require_bf16_local_evidence_mirror_root(mirror, create=True)
    assert root.stat().st_mode & 0o777 == 0o700
    path = generation._write_bf16_exact_mirror_bytes(
        root, "worker-results/worker-00.json", b"exact\n"
    )
    assert path.stat().st_mode & 0o777 == 0o600
    assert generation._read_bf16_stable_regular_file(path) == b"exact\n"
    generation._write_bf16_exact_mirror_bytes(
        root, "worker-results/worker-00.json", b"exact\n"
    )
    with pytest.raises(FileExistsError, match="different bytes"):
        generation._write_bf16_exact_mirror_bytes(
            root, "worker-results/worker-00.json", b"tampered\n"
        )


def test_bf16_local_evidence_mirror_rejects_non_normalized(tmp_path):
    with pytest.raises(ValueError, match="absolute and normalized"):
        generation._require_bf16_local_evidence_mirror_root("relative", create=True)
    with pytest.raises(ValueError, match="absolute and normalized"):
        generation._require_bf16_local_evidence_mirror_root(
            f"{tmp_path}/../escape", create=True
        )


def test_bf16_wave_rejects_boolean_attempt_key(prepared, monkeypatch, tmp_path):
    ledger_path = tmp_path / "ledger.json"
    (
        _plan,
        _config,
        _qualification,
        authorization,
        q8_authorization,
        payloads,
        job,
    ) = _launch_material(prepared, monkeypatch, ledger_path=ledger_path)
    submissions = build_databricks_publication_bf16_handoff_submit_payloads(
        job,
        payloads,
        ledger_path=ledger_path,
        qualification_launch_authorization=authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
    )
    attempts = {
        index: publication_bf16_handoff_worker_attempt_id(
            payloads[index], worker_index=index
        )
        for index in range(1, 16)
    }
    attempts[False] = publication_bf16_handoff_worker_attempt_id(
        payloads[0], worker_index=0
    )
    with pytest.raises(ValueError, match="workers 0..15 exactly"):
        reserve_and_submit_publication_bf16_handoff_worker_wave(
            DatabricksWorkspaceConfig("https://example.cloud.databricks.com", "token"),
            submissions,
            ledger_path=ledger_path,
            worker_payloads=payloads,
            attempt_ids_by_worker=attempts,
            phase_lease_root=tmp_path / "must-not-create",
            job_config=job,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
        )


@pytest.mark.parametrize("drift", ["host", "principal"])
def test_bf16_wave_rejects_workspace_drift_from_q8_closure_before_state(
    prepared, monkeypatch, tmp_path, drift
):
    ledger_path = tmp_path / f"{drift}-ledger.json"
    (
        _plan,
        _config,
        _qualification,
        authorization,
        q8_authorization,
        payloads,
        job,
    ) = _launch_material(prepared, monkeypatch, ledger_path=ledger_path)
    submissions = build_databricks_publication_bf16_handoff_submit_payloads(
        job,
        payloads,
        ledger_path=ledger_path,
        qualification_launch_authorization=authorization,
        q8_handoff_remote_closure_authorization=q8_authorization,
    )
    attempts = {
        index: publication_bf16_handoff_worker_attempt_id(
            payloads[index], worker_index=index
        )
        for index in range(16)
    }
    monkeypatch.setattr(
        generation,
        "require_databricks_current_user_name",
        lambda _workspace, *, expected_user_name, opener=None: {
            "workspace_host_sha256": (
                sha256(b"https://other.example").hexdigest()
                if drift == "host"
                else q8_authorization.workspace_host_sha256
            ),
            "user_name_sha256": (
                sha256(b"other-principal@example.com").hexdigest()
                if drift == "principal"
                else q8_authorization.user_name_sha256
            ),
        },
    )
    phase_root = tmp_path / "must-not-create"
    with pytest.raises(ValueError, match="differs from Q8 closure"):
        reserve_and_submit_publication_bf16_handoff_worker_wave(
            DatabricksWorkspaceConfig("https://example.cloud.databricks.com", "token"),
            submissions,
            ledger_path=ledger_path,
            worker_payloads=payloads,
            attempt_ids_by_worker=attempts,
            phase_lease_root=phase_root,
            job_config=job,
            qualification_launch_authorization=authorization,
            q8_handoff_remote_closure_authorization=q8_authorization,
        )
    assert not phase_root.exists()
