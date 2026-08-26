import copy
import json
import os
from collections import Counter
from hashlib import sha256

import pytest

import document_kv_cache.gpu_qualification_databricks as qualification_job
import document_kv_cache.publication_latency_handoff_generation as generation
from document_kv_cache.artifact_identity import TokenContract
from document_kv_cache.benchmarks import SUPPORTED_V1_DATASETS
from document_kv_cache.engine_protocol import KVLayout
from document_kv_cache.kvpack import PackChunk
from document_kv_cache.main_latency_inputs import (
    MAIN_LATENCY_TARGET_SEGMENT_COUNTS,
    MAIN_LATENCY_TOKENIZER_ID,
    MAIN_LATENCY_TOKENIZER_REVISION,
    MainLatencyInputFile,
    PreparedMainLatencyInputs,
)
from document_kv_cache.model_profiles import QWEN3_4B_INSTRUCT_HF_MODEL_ID
from document_kv_cache.models import KVCacheKey
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256,
    GPUQualificationArtifactPins,
    GPUQualificationSelection,
)
from document_kv_cache.gpu_qualification_databricks import (
    GPUQualificationLaunchAuthorization,
)
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
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_INPUT_TOKEN_SLOTS,
)
from document_kv_cache.publication_latency_handoff_generation import (
    PUBLICATION_LATENCY_HANDOFF_TASK_COUNT,
    DatabricksPublicationLatencyHandoffJobConfig,
    PublicationLatencyGeneratorHardwareQualification,
    PublicationLatencyHandoffServingAuthorization,
    PublicationLatencyHandoffExecutionConfig,
    authorize_publication_latency_handoff_serving,
    build_databricks_publication_latency_handoff_worker_submit_payloads,
    build_publication_latency_handoff_databricks_attestation,
    build_publication_latency_handoff_worker_payloads,
    build_publication_latency_handoff_generation_plan,
    build_publication_latency_handoff_execution_config,
    close_publication_latency_handoff_generation_from_workers,
    execute_publication_latency_handoff_generation_plan_local_test_helper,
    publication_latency_handoff_terminal_actual_gpu_seconds_from_ledger,
    publication_latency_handoff_worker_attempt_id,
    require_publication_latency_full_launch_ready,
    require_publication_latency_handoff_submission_authorization,
    require_publication_latency_handoff_serving_authorization,
    read_publication_latency_handoff_generation_plan,
    resolve_publication_latency_serving_handoff_bundle,
    resolve_publication_latency_worker_handoff_bundle,
    run_publication_latency_handoff_worker,
    reconcile_publication_latency_handoff_worker_attempt_json,
    reserve_and_submit_publication_latency_handoff_worker_wave,
    resume_publication_latency_handoff_worker_wave,
    reserve_and_submit_publication_latency_handoff_worker,
    reserve_publication_latency_handoff_worker_attempt_json,
    validate_publication_latency_handoff_generation_plan,
    validate_publication_latency_handoff_worker_payload,
    write_publication_latency_handoff_worker_payloads,
    write_publication_latency_handoff_databricks_attestation,
    write_publication_latency_handoff_generation_plan,
)
from document_kv_cache.serving_env import VLLM_RUNTIME_LOCK_SHA256


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) for character in text]


class JsonHTTPResponse:
    status = 200

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")
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


class StrictQ8PreRopeGenerator:
    pre_rope = True
    rope_theta = 5_000_000.0
    rope_rotary_dim = 4
    add_special_tokens = False

    def __init__(self, layout, tracker, worker_index):
        self.layout = layout
        self.tracker = tracker
        self.worker_index = worker_index
        self.tracker["created"].append(worker_index)

    def bind_layout(self, layout):
        self.layout = layout

    def generate(self, *, document, chunk, config, training_artifacts=None):
        del training_artifacts
        self.tracker["calls"][self.worker_index] += 1
        token_ids = tuple(chunk.text.encode("ascii")) or (0,)
        payload = bytes([self.worker_index + 1]) * (
            len(token_ids) * self.layout.bytes_per_token
        )
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
                    add_special_tokens=False,
                    prompt_template_version=config.prompt_template_version,
                ),
            ),
            payload=payload,
            token_count=len(token_ids),
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )

    def close(self):
        self.tracker["closed"].append(self.worker_index)


def publication_layout():
    return KVLayout(
        model_id=QWEN3_4B_INSTRUCT_HF_MODEL_ID,
        lora_id="base",
        layout_version="qwen3-publication-generation-test-v1",
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


def canonical_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def prepared_row(dataset, example_index, context_tokens):
    segment_count = MAIN_LATENCY_TARGET_SEGMENT_COUNTS[context_tokens]
    return {
        "dataset": dataset,
        "documents": [
            {
                "document_id": (
                    f"{dataset}-{example_index:02d}-{context_tokens}-{segment_index:02d}"
                ),
                "metadata": {
                    "cachet.main_latency.segment_count": str(segment_count),
                    "cachet.main_latency.segment_index": str(segment_index),
                    "cachet.main_latency.transformation": (
                        "cachet.main_latency.lossless_context_tiling.v1"
                    ),
                },
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


def fake_prepared_inputs(tmp_path):
    root = tmp_path / "prepared"
    files = []
    for context_tokens in MAIN_LATENCY_TARGET_SEGMENT_COUNTS:
        for dataset in SUPPORTED_V1_DATASETS:
            path = root / str(context_tokens) / f"{dataset}.jsonl"
            canonical_jsonl(
                path,
                [prepared_row(dataset, index, context_tokens) for index in range(32)],
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
        bundle_sha256=sha256(b"verified-main-latency-bundle").hexdigest(),
        files=tuple(files),
    )


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    value = fake_prepared_inputs(tmp_path)
    monkeypatch.setattr(
        generation,
        "verify_main_latency_inputs",
        lambda *args, **kwargs: value,
    )
    return value


def test_generation_plan_is_exactly_once_and_token_balanced(prepared):
    plan = build_publication_latency_handoff_generation_plan(
        prepared.output_dir,
        plan_id="latency-handoffs-2026",
        tokenizer=CharacterTokenizer(),
    )

    items = [item for worker in plan["workers"] for item in worker["items"]]
    context_identities = Counter(
        (item["context_tokens"], item["dataset"], item["example_id"]) for item in items
    )
    cross_context_identities = Counter(
        (item["dataset"], item["example_id"]) for item in items
    )
    assert len(items) == PUBLICATION_LATENCY_HANDOFF_TASK_COUNT == 384
    assert set(context_identities.values()) == {1}
    assert set(cross_context_identities.values()) == {3}
    assert plan["coverage"]["input_token_slots"] == (
        PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_INPUT_TOKEN_SLOTS
    )
    assert plan["sharding"]["worker_count"] == 16
    assert all(
        worker["persistent_generator_instances"] == 1 for worker in plan["workers"]
    )
    worker_tokens = [worker["cache_prefix_tokens"] for worker in plan["workers"]]
    assert max(worker_tokens) - min(worker_tokens) <= max(
        item["cache_prefix_tokens"] for item in items
    )
    validate_publication_latency_handoff_generation_plan(
        plan,
        prepared_input_dir=prepared.output_dir,
        tokenizer=CharacterTokenizer(),
    )

    tampered = copy.deepcopy(plan)
    tampered["workers"][0]["items"].pop()
    with pytest.raises(ValueError, match="closed_record_sha256"):
        validate_publication_latency_handoff_generation_plan(
            tampered,
            prepared_input_dir=prepared.output_dir,
            tokenizer=CharacterTokenizer(),
        )


def test_generation_plan_round_trips_canonically_once(prepared, tmp_path):
    plan = build_publication_latency_handoff_generation_plan(
        prepared.output_dir,
        plan_id="latency-handoffs-2026",
        tokenizer=CharacterTokenizer(),
        worker_count=2,
    )
    path = tmp_path / "plan.json"
    assert write_publication_latency_handoff_generation_plan(plan, path) == path
    assert read_publication_latency_handoff_generation_plan(path) == plan
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_publication_latency_handoff_generation_plan(plan, path)


@pytest.mark.parametrize("worker_count", [0, 17, True])
def test_generation_plan_caps_persistent_workers(prepared, worker_count):
    with pytest.raises(ValueError, match="worker_count"):
        build_publication_latency_handoff_generation_plan(
            prepared.output_dir,
            plan_id="invalid-workers",
            tokenizer=CharacterTokenizer(),
            worker_count=worker_count,
        )


def test_executor_closes_content_addressed_bundles_for_serving_reuse(
    prepared,
    tmp_path,
):
    tokenizer = CharacterTokenizer()
    plan = build_publication_latency_handoff_generation_plan(
        prepared.output_dir,
        plan_id="latency-handoffs-2026",
        tokenizer=tokenizer,
        worker_count=2,
    )
    layout = publication_layout()
    tracker = {
        "created": [],
        "closed": [],
        "calls": Counter(),
    }

    def worker_factory(worker_index):
        return StrictQ8PreRopeGenerator(layout, tracker, worker_index)

    output = tmp_path / "published"
    result = execute_publication_latency_handoff_generation_plan_local_test_helper(
        plan,
        prepared_input_dir=prepared.output_dir,
        output_dir=output,
        tokenizer=tokenizer,
        worker_factory=worker_factory,
        config=PublicationLatencyHandoffExecutionConfig(
            layout=layout,
            model_revision="model-revision-pinned",
            generator_version="generator-version-pinned",
            vllm_bitsandbytes_loader_source_sha256=(
                GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
            ),
        ),
    )

    assert sorted(tracker["created"]) == [0, 1]
    assert sorted(tracker["closed"]) == [0, 1]
    assert all(tracker["calls"][worker_index] > 1 for worker_index in (0, 1))
    assert result.record["coverage"]["generated_task_count"] == 384
    assert result.record["coverage"]["no_duplicate_tasks"] is True
    assert result.record["accounting"]["charged_gpu_hours"] > 0
    assert (
        result.record["accounting"]["end_to_end_cache_prefix_tokens_per_gpu_second"] > 0
    )
    assert (
        result.record["serving_reuse"]["regenerate_inside_timed_serving_jobs"] is False
    )

    with pytest.raises(
        TypeError, match="PublicationLatencyHandoffServingAuthorization"
    ):
        resolve_publication_latency_serving_handoff_bundle(
            result,
            context_tokens=16_384,
        )
    with pytest.raises(ValueError, match="authenticated distributed execution"):
        resolve_publication_latency_worker_handoff_bundle(
            result,
            context_tokens=16_384,
        )

    with pytest.raises(
        TypeError, match="PublicationLatencyHandoffServingAuthorization"
    ):
        require_publication_latency_full_launch_ready(
            result,
            other_terminal_gpu_hours=0.0,
            current_active_reserved_gpu_hours=0.0,
            proposed_full_launch_reserved_gpu_hours=800.0,
        )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        execute_publication_latency_handoff_generation_plan_local_test_helper(
            plan,
            prepared_input_dir=prepared.output_dir,
            output_dir=output,
            tokenizer=tokenizer,
            worker_factory=worker_factory,
            config=PublicationLatencyHandoffExecutionConfig(
                layout=layout,
                model_revision="model-revision-pinned",
                generator_version="generator-version-pinned",
                vllm_bitsandbytes_loader_source_sha256=(
                    GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
                ),
            ),
        )


def test_execution_config_rejects_non_q8_or_post_rope_layout():
    q8 = publication_layout()
    with pytest.raises(ValueError, match="fp8_e5m2"):
        PublicationLatencyHandoffExecutionConfig(
            layout=KVLayout(
                model_id=q8.model_id,
                lora_id=q8.lora_id,
                layout_version=q8.layout_version,
                dtype="bf16",
                num_layers=1,
                block_size=8,
                bytes_per_token=32,
                num_query_heads=1,
                num_kv_heads=1,
                head_size=4,
                kv_stride_bytes=16,
                storage_layout="separate_key_value",
                pre_rope=True,
                rope_theta=5_000_000.0,
                rope_rotary_dim=4,
            ),
            model_revision="pinned",
            generator_version="pinned",
            vllm_bitsandbytes_loader_source_sha256=(
                GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
            ),
        )
    with pytest.raises(ValueError, match="pre-RoPE"):
        PublicationLatencyHandoffExecutionConfig(
            layout=KVLayout(
                model_id=q8.model_id,
                lora_id=q8.lora_id,
                layout_version=q8.layout_version,
                dtype="fp8_e5m2",
                num_layers=1,
                block_size=8,
                bytes_per_token=16,
                num_query_heads=1,
                num_kv_heads=1,
                head_size=4,
                kv_stride_bytes=8,
                storage_layout="separate_key_value",
            ),
            model_revision="pinned",
            generator_version="pinned",
            vllm_bitsandbytes_loader_source_sha256=(
                GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
            ),
        )


def test_generation_plan_binds_pinned_tokenizer(prepared):
    config = PublicationLatencyHandoffExecutionConfig(
        layout=publication_layout(),
        model_revision="pinned",
        generator_version="pinned",
        vllm_bitsandbytes_loader_source_sha256=(
            GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
        ),
    )
    assert config.tokenizer_id == MAIN_LATENCY_TOKENIZER_ID
    assert config.tokenizer_revision == MAIN_LATENCY_TOKENIZER_REVISION


def test_production_config_pins_q8_nf4_double_quant_and_loader_source(monkeypatch):
    config = build_publication_latency_handoff_execution_config(
        vllm_bitsandbytes_loader_source_sha256=(
            GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
        )
    )
    assert config.layout.dtype == "fp8_e5m2"
    assert config.layout.bytes_per_token == 73_728
    assert config.layout.pre_rope is True
    assert config.generator_device_map == "auto"
    assert config.generator_model_dtype == "bfloat16"
    assert config.generator_trust_remote_code is False
    assert dict(config.generator_quantization_config) == {
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_storage": "uint8",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "load_in_4bit": True,
    }
    namespace = {"__name__": "latency_handoff_runner_test"}
    exec(
        compile(
            generation.PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT,
            "publication_latency_handoff_runner.py",
            "exec",
        ),
        namespace,
    )
    monkeypatch.setenv("PIP_CONFIG_FILE", "/attacker/pip.conf")
    monkeypatch.setenv("PIP_INDEX_URL", "https://attacker.invalid/simple")
    monkeypatch.setenv("pip_no_index", "1")
    monkeypatch.setenv("PIP_REQUIREMENT", "/attacker/requirements.txt")
    monkeypatch.setenv("PYTHONHOME", "/attacker/python-home")
    monkeypatch.setenv("PYTHONPATH", "/attacker/python-path")
    monkeypatch.setenv("VIRTUAL_ENV", "/attacker/venv")
    environment = namespace["_pip_subprocess_environment"]()
    assert {
        key for key in environment if key.upper().startswith("PIP_")
    } == {
        "PIP_CONFIG_FILE",
        "PIP_DISABLE_PIP_VERSION_CHECK",
        "PIP_NO_INPUT",
    }
    assert environment["PIP_CONFIG_FILE"] == os.devnull
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert all(
        variable not in environment
        for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")
    )
    assert '"--extra-index-url"' not in (
        generation.PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT
    )


def fake_hardware_qualification(monkeypatch, prepared):
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
        generation,
        "validate_gpu_qualification_evidence_record",
        lambda *args, **kwargs: selection,
    )
    pins = GPUQualificationArtifactPins(
        runtime_lock_sha256=VLLM_RUNTIME_LOCK_SHA256,
        patched_vllm_wheel_sha256="2" * 64,
        package_wheel_sha256="3" * 64,
        cachet_source_tree_sha256="4" * 64,
        runner_sha256="5" * 64,
        input_bundle_sha256=prepared.bundle_sha256,
    )
    return PublicationLatencyGeneratorHardwareQualification(
        evidence_record={"closed_record_sha256": "6" * 64},
        plan_record={"closed_record_sha256": "8" * 64},
        expected_campaign_id="vllm-0271-publication-v1",
        expected_artifact_pins=pins,
        evidence_uri="dbfs:/qualification/evidence.json",
        evidence_file_sha256="9" * 64,
        plan_uri="dbfs:/qualification/plan.json",
        plan_file_sha256="a" * 64,
    )


def fake_launch_authorization(
    qualification,
    *,
    plan_sha256=None,
    evidence_closed_record_sha256=None,
    evidence_file_sha256=None,
    selection=None,
    ledger_id="gpu-qualification-test-ledger",
    ledger_path=None,
    ledger_prefix=None,
):
    resolved_prefix = ledger_prefix or (
        databricks_ledger_prefix(read_databricks_cluster_hour_ledger_json(ledger_path))
        if ledger_path is not None
        else databricks_ledger_prefix(DatabricksClusterHourLedger(ledger_id=ledger_id))
    )
    return GPUQualificationLaunchAuthorization(
        selection=selection or qualification.selection,
        plan_sha256=plan_sha256 or qualification.selection.plan_sha256,
        evidence_closed_record_sha256=(
            evidence_closed_record_sha256
            or qualification.evidence_record["closed_record_sha256"]
        ),
        evidence_file_sha256=(
            evidence_file_sha256 or qualification.evidence_file_sha256
        ),
        ledger_id=ledger_id,
        ledger_path_sha256=(
            databricks_ledger_path_sha256(ledger_path)
            if ledger_path is not None
            else "f" * 64
        ),
        predecessor_prefix=resolved_prefix,
        producer_batch_prefix=resolved_prefix,
        ledger_prefix=resolved_prefix,
        causal_closure_sha256="c" * 64,
        _issuer=qualification_job._LAUNCH_AUTHORIZATION_ISSUER,
    )


def production_launch_material(prepared, monkeypatch):
    plan = build_publication_latency_handoff_generation_plan(
        prepared.output_dir,
        plan_id="latency-handoff-authorization-tests",
        tokenizer=CharacterTokenizer(),
    )
    config = PublicationLatencyHandoffExecutionConfig(
        layout=publication_layout(),
        model_revision="model-revision-pinned",
        generator_version="generator-version-pinned",
        vllm_bitsandbytes_loader_source_sha256=(
            GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
        ),
    )
    qualification = fake_hardware_qualification(monkeypatch, prepared)
    authorization = fake_launch_authorization(qualification)
    payloads = list(
        build_publication_latency_handoff_worker_payloads(
            plan,
            plan_uri="dbfs:/plans/latency-handoff-plan.json",
            plan_file_sha256="d" * 64,
            prepared_input_uri="dbfs:/prepared/main-latency",
            prepared_provenance_file_sha256="e" * 64,
            prepared_provenance_closed_record_sha256="f" * 64,
            durable_output_root=(
                "dbfs:/Volumes/catalog/schema/volume/"
                "publication-latency-handoffs/"
                f"{plan['closed_record_sha256']}"
            ),
            local_work_root_template="/local_disk0/worker-{worker_index}",
            config=config,
            hardware_qualification=qualification,
        )
    )
    job_config = DatabricksPublicationLatencyHandoffJobConfig(
        runner_python_file="dbfs:/runner.py",
        worker_payload_uri_template="dbfs:/payloads/worker-{worker_index}.json",
        package_wheel_uri="dbfs:/cachet.whl",
        package_wheel_sha256="3" * 64,
        runtime_lock_uri="dbfs:/runtime.lock",
        runtime_lock_sha256=VLLM_RUNTIME_LOCK_SHA256,
        patched_vllm_wheel_uri="dbfs:/vllm.whl",
        patched_vllm_wheel_sha256="2" * 64,
        single_user_name="publication@example.com",
    )
    return plan, payloads, job_config, qualification, authorization


def successful_terminal_run(submit_payload, worker_index):
    parent_run_id = 10_000 + worker_index
    cluster_id = f"cluster-{worker_index:02d}"
    parent_start = 1_000_000 + worker_index * 10_000
    task_start = parent_start + 100
    task_end = task_start + 1_000
    return {
        "cluster_instance": {"cluster_id": cluster_id},
        "end_time": task_end + 100,
        "original_attempt_run_id": parent_run_id,
        "repair_history": [],
        "run_id": parent_run_id,
        "run_name": submit_payload["run_name"],
        "start_time": parent_start,
        "state": {
            "life_cycle_state": "TERMINATED",
            "result_state": "SUCCESS",
        },
        "tasks": [
            {
                "attempt_number": 0,
                "cluster_instance": {"cluster_id": cluster_id},
                "end_time": task_end,
                "new_cluster": copy.deepcopy(submit_payload["tasks"][0]["new_cluster"]),
                "run_id": 20_000 + worker_index,
                "start_time": task_start,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "SUCCESS",
                },
                "task_key": f"latency_handoff_worker_{worker_index:02d}",
            }
        ],
    }


def test_production_submit_render_rejects_missing_wrong_or_forged_authority(
    prepared,
    monkeypatch,
    tmp_path,
):
    (
        _plan,
        payloads,
        job_config,
        qualification,
        authorization,
    ) = production_launch_material(prepared, monkeypatch)
    ledger_path = tmp_path / "qualification-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id=authorization.ledger_id,
    )
    authorization = fake_launch_authorization(qualification, ledger_path=ledger_path)
    wrong_ledger_path = tmp_path / "wrong-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        wrong_ledger_path,
        ledger_id="different-qualification-ledger",
    )

    with pytest.raises(TypeError, match="qualification_launch_authorization"):
        build_databricks_publication_latency_handoff_worker_submit_payloads(
            job_config,
            payloads,
            ledger_path=ledger_path,
        )

    with pytest.raises(ValueError, match="differs from GPU qualification authority"):
        build_databricks_publication_latency_handoff_worker_submit_payloads(
            job_config,
            payloads,
            ledger_path=wrong_ledger_path,
            qualification_launch_authorization=authorization,
        )

    wrong_plan_authorization = fake_launch_authorization(
        qualification,
        plan_sha256="1" * 64,
        ledger_path=ledger_path,
    )
    with pytest.raises(ValueError, match="authorization plan binding differs"):
        build_databricks_publication_latency_handoff_worker_submit_payloads(
            job_config,
            payloads,
            ledger_path=ledger_path,
            qualification_launch_authorization=wrong_plan_authorization,
        )

    with pytest.raises(
        TypeError,
        match="GPUQualificationLaunchAuthorization",
    ):
        build_databricks_publication_latency_handoff_worker_submit_payloads(
            job_config,
            payloads,
            ledger_path=ledger_path,
            qualification_launch_authorization=qualification.selection,
        )

    for field_name, replacement, message in (
        ("generation_hardware_id", "aws-g6-l4", "must select L40S"),
        (
            "generation_prefix_tokens_per_second",
            41.0,
            "selection differs from worker payload",
        ),
        (
            "generation_artifacts_sha256",
            "0" * 64,
            "selection differs from worker payload",
        ),
    ):
        forged_payloads = copy.deepcopy(payloads)
        forged_qualification = forged_payloads[0]["generator_hardware_qualification"]
        forged_qualification[field_name] = replacement
        forged_payloads[0]["closed_record_sha256"] = generation._closed_record_sha256(
            forged_payloads[0]
        )
        with pytest.raises(ValueError, match=message):
            build_databricks_publication_latency_handoff_worker_submit_payloads(
                job_config,
                forged_payloads,
                ledger_path=ledger_path,
                qualification_launch_authorization=authorization,
            )


def test_reservation_and_submission_recheck_wrong_authority(
    prepared,
    monkeypatch,
    tmp_path,
):
    (
        _plan,
        payloads,
        job_config,
        qualification,
        authorization,
    ) = production_launch_material(prepared, monkeypatch)
    ledger_path = tmp_path / "wrong-authorization-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id=authorization.ledger_id,
    )
    authorization = fake_launch_authorization(qualification, ledger_path=ledger_path)
    wrong_ledger_path = tmp_path / "different-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        wrong_ledger_path,
        ledger_id="different-qualification-ledger",
    )
    submit_payloads = (
        build_databricks_publication_latency_handoff_worker_submit_payloads(
            job_config,
            payloads,
            ledger_path=ledger_path,
            qualification_launch_authorization=authorization,
        )
    )
    wrong_evidence_authorization = fake_launch_authorization(
        qualification,
        evidence_file_sha256="1" * 64,
        ledger_path=ledger_path,
    )

    with pytest.raises(ValueError, match="differs from GPU qualification authority"):
        reserve_publication_latency_handoff_worker_attempt_json(
            wrong_ledger_path,
            submit_payloads[0],
            worker_payload=payloads[0],
            worker_index=0,
            attempt_id="different-ledger-reserve",
            qualification_launch_authorization=authorization,
        )

    cross_ledger_opener_called = False

    def cross_ledger_opener(*args, **kwargs):
        nonlocal cross_ledger_opener_called
        cross_ledger_opener_called = True
        raise AssertionError("cross-ledger submit must fail before cloud submission")

    with pytest.raises(ValueError, match="differs from GPU qualification authority"):
        reserve_and_submit_publication_latency_handoff_worker(
            DatabricksWorkspaceConfig(
                "https://dbc.example/",
                "secret-token",
            ),
            submit_payloads[0],
            ledger_path=wrong_ledger_path,
            worker_payload=payloads[0],
            worker_index=0,
            attempt_id="different-ledger-submit",
            qualification_launch_authorization=authorization,
            opener=cross_ledger_opener,
        )
    assert cross_ledger_opener_called is False

    forged_submit_payload = copy.deepcopy(submit_payloads[0])
    parameters = forged_submit_payload["tasks"][0]["spark_python_task"]["parameters"]
    package_hash_index = parameters.index("--package-wheel-sha256") + 1
    parameters[package_hash_index] = "0" * 64
    with pytest.raises(ValueError, match="--package-wheel-sha256 differs"):
        reserve_publication_latency_handoff_worker_attempt_json(
            ledger_path,
            forged_submit_payload,
            worker_payload=payloads[0],
            worker_index=0,
            attempt_id="forged-artifact-reserve",
            qualification_launch_authorization=authorization,
        )

    with pytest.raises(ValueError, match="evidence binding differs"):
        reserve_publication_latency_handoff_worker_attempt_json(
            ledger_path,
            submit_payloads[0],
            worker_payload=payloads[0],
            worker_index=0,
            attempt_id="wrong-reserve",
            qualification_launch_authorization=wrong_evidence_authorization,
        )

    opener_called = False

    def opener(*args, **kwargs):
        nonlocal opener_called
        opener_called = True
        raise AssertionError("wrong authorization must fail before cloud submission")

    with pytest.raises(ValueError, match="evidence binding differs"):
        reserve_and_submit_publication_latency_handoff_worker(
            DatabricksWorkspaceConfig(
                "https://dbc.example/",
                "secret-token",
            ),
            submit_payloads[0],
            ledger_path=ledger_path,
            worker_payload=payloads[0],
            worker_index=0,
            attempt_id="wrong-submit",
            qualification_launch_authorization=wrong_evidence_authorization,
            opener=opener,
        )
    assert opener_called is False
    assert read_databricks_cluster_hour_ledger_json(ledger_path).reservations == ()


def test_single_worker_publication_route_is_nonauthorizing(
    prepared,
    monkeypatch,
    tmp_path,
):
    (
        _plan,
        payloads,
        job_config,
        qualification,
        authorization,
    ) = production_launch_material(prepared, monkeypatch)
    ledger_path = tmp_path / "toctou-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id=authorization.ledger_id,
    )
    authorization = fake_launch_authorization(qualification, ledger_path=ledger_path)
    submit_payloads = (
        build_databricks_publication_latency_handoff_worker_submit_payloads(
            job_config,
            payloads,
            ledger_path=ledger_path,
            qualification_launch_authorization=authorization,
        )
    )
    with pytest.raises(RuntimeError, match="nonpublication"):
        reserve_publication_latency_handoff_worker_attempt_json(
            ledger_path,
            submit_payloads[0],
            worker_payload=payloads[0],
            worker_index=0,
            attempt_id="toctou-reserve",
            qualification_launch_authorization=authorization,
        )
    assert read_databricks_cluster_hour_ledger_json(ledger_path).reservations == ()


def test_sequential_q8_reservations_cannot_mint_phase_submission_authority(
    prepared, monkeypatch, tmp_path
):
    (
        _plan,
        worker_payloads,
        job_config,
        qualification,
        _authorization,
    ) = production_launch_material(prepared, monkeypatch)
    ledger_path = tmp_path / "sequential-q8-ledger.json"
    opening = create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="gpu-qualification-test-ledger"
    )
    authorization = fake_launch_authorization(
        qualification, ledger_path=ledger_path
    )
    submissions = build_databricks_publication_latency_handoff_worker_submit_payloads(
        job_config,
        worker_payloads,
        ledger_path=ledger_path,
        qualification_launch_authorization=authorization,
    )
    requests = []
    for index, submission in enumerate(submissions):
        attempt_id = publication_latency_handoff_worker_attempt_id(
            worker_payloads[index], worker_index=index
        )
        reserve_databricks_run_attempt_json(
            ledger_path,
            submission,
            attempt_id=attempt_id,
            workload_id=f"publication-latency-handoff-worker-{index:02d}",
        )
        requests.append(
            DatabricksRunAttemptReservationRequest(
                attempt_id=attempt_id,
                workload_id=f"publication-latency-handoff-worker-{index:02d}",
                submit_payload=submission,
            )
        )
    replayed = replay_databricks_run_attempt_batch_authorization_json(
        ledger_path,
        tuple(requests),
        expected_predecessor_prefix=databricks_ledger_prefix(opening),
    )
    with pytest.raises(
        TypeError, match="PublicationLatencyHandoffSubmissionAuthorization"
    ):
        require_publication_latency_handoff_submission_authorization(replayed)


def test_q8_wave_resumes_lost_first_response_and_unclaimed_members(
    prepared, monkeypatch, tmp_path
):
    (
        _plan,
        worker_payloads,
        job_config,
        qualification,
        _authorization,
    ) = production_launch_material(prepared, monkeypatch)
    ledger_path = tmp_path / "q8-resume-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="gpu-qualification-test-ledger"
    )
    authorization = fake_launch_authorization(
        qualification, ledger_path=ledger_path
    )
    submissions = build_databricks_publication_latency_handoff_worker_submit_payloads(
        job_config,
        worker_payloads,
        ledger_path=ledger_path,
        qualification_launch_authorization=authorization,
    )
    attempts = {
        index: publication_latency_handoff_worker_attempt_id(
            worker_payloads[index], worker_index=index
        )
        for index in range(16)
    }
    phase_root = tmp_path / "q8-resume-phase"
    with pytest.raises(TimeoutError, match="lost response"):
        reserve_and_submit_publication_latency_handoff_worker_wave(
            DatabricksWorkspaceConfig("https://dbc.example", "token"),
            submissions,
            ledger_path=ledger_path,
            worker_payloads=worker_payloads,
            attempt_ids_by_worker=attempts,
            phase_lease_root=phase_root,
            qualification_launch_authorization=authorization,
            opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                TimeoutError("lost response")
            ),
        )
    (phase_root / "batch-reserved.json").unlink()
    next_run_id = 80_000

    def opener(_request, *, timeout):
        nonlocal next_run_id
        assert timeout > 0
        response = JsonHTTPResponse({"run_id": next_run_id})
        next_run_id += 1
        return response

    responses, batch = resume_publication_latency_handoff_worker_wave(
        DatabricksWorkspaceConfig("https://dbc.example", "token"),
        submissions,
        ledger_path=ledger_path,
        worker_payloads=worker_payloads,
        attempt_ids_by_worker=attempts,
        phase_lease_root=phase_root,
        qualification_launch_authorization=authorization,
        opener=opener,
    )
    assert len(responses) == 16
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert len(ledger.reservations) == len(ledger.submission_receipts) == 16
    assert (phase_root / "batch-reserved.json").is_file()
    assert batch.durable_output_root == worker_payloads[0]["durable_output_root"]
    alternate_roots = [copy.deepcopy(item) for item in worker_payloads]
    alternate_roots[0]["durable_output_root"] += "-alternate"
    with pytest.raises(ValueError, match="share one durable_output_root"):
        resume_publication_latency_handoff_worker_wave(
            DatabricksWorkspaceConfig("https://dbc.example", "token"),
            submissions,
            ledger_path=ledger_path,
            worker_payloads=alternate_roots,
            attempt_ids_by_worker=attempts,
            phase_lease_root=phase_root,
            qualification_launch_authorization=authorization,
            opener=lambda *_args, **_kwargs: pytest.fail(
                "alternate Q8 output root must not POST"
            ),
        )


def test_distributed_workers_close_without_copy_and_render_independent_jobs(
    prepared,
    monkeypatch,
    tmp_path,
):
    tokenizer = CharacterTokenizer()
    plan = build_publication_latency_handoff_generation_plan(
        prepared.output_dir,
        plan_id="latency-handoffs-distributed",
        tokenizer=tokenizer,
    )
    plan_path = tmp_path / "plan.json"
    write_publication_latency_handoff_generation_plan(plan, plan_path)
    provenance = {
        "closed_record_sha256": "b" * 64,
    }
    production_provenance_path = (
        prepared.output_dir / generation.MAIN_LATENCY_PROVENANCE_FILENAME
    )
    production_provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = PublicationLatencyHandoffExecutionConfig(
        layout=publication_layout(),
        model_revision="model-revision-pinned",
        generator_version="generator-version-pinned",
        vllm_bitsandbytes_loader_source_sha256=(
            GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
        ),
    )
    qualification = fake_hardware_qualification(monkeypatch, prepared)
    launch_authorization = fake_launch_authorization(qualification)
    durable_root = tmp_path / "durable"
    payloads = build_publication_latency_handoff_worker_payloads(
        plan,
        plan_uri=str(plan_path),
        plan_file_sha256=sha256(plan_path.read_bytes()).hexdigest(),
        prepared_input_uri=str(prepared.output_dir),
        prepared_provenance_file_sha256=sha256(
            production_provenance_path.read_bytes()
        ).hexdigest(),
        prepared_provenance_closed_record_sha256="b" * 64,
        durable_output_root=str(durable_root),
        local_work_root_template="/local_disk0/worker-{worker_index}",
        config=config,
        hardware_qualification=qualification,
    )
    assert len(payloads) == 16
    for index, payload in enumerate(payloads):
        assert payload["worker_index"] == index
        validate_publication_latency_handoff_worker_payload(payload, plan=plan)

    production_payloads = []
    for payload in payloads:
        production_payload = copy.deepcopy(payload)
        production_payload["durable_output_root"] = (
            "dbfs:/Volumes/catalog/schema/volume/"
            f"publication-latency-handoffs/{plan['closed_record_sha256']}"
        )
        production_payload.pop("closed_record_sha256")
        production_payload["closed_record_sha256"] = sha256(
            json.dumps(
                production_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        production_payloads.append(production_payload)
    ledger_path = tmp_path / "producer-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id=launch_authorization.ledger_id,
    )
    launch_authorization = fake_launch_authorization(
        qualification, ledger_path=ledger_path
    )
    databricks_payloads = (
        build_databricks_publication_latency_handoff_worker_submit_payloads(
            DatabricksPublicationLatencyHandoffJobConfig(
                runner_python_file="dbfs:/runner.py",
                worker_payload_uri_template="dbfs:/payloads/worker-{worker_index}.json",
                package_wheel_uri="dbfs:/cachet.whl",
                package_wheel_sha256="3" * 64,
                runtime_lock_uri="dbfs:/runtime.lock",
                runtime_lock_sha256=VLLM_RUNTIME_LOCK_SHA256,
                patched_vllm_wheel_uri="dbfs:/vllm.whl",
                patched_vllm_wheel_sha256="2" * 64,
                single_user_name="publication@example.com",
            ),
            production_payloads,
            ledger_path=ledger_path,
            qualification_launch_authorization=launch_authorization,
        )
    )
    assert len(databricks_payloads) == 16
    assert all(len(item["tasks"]) == 1 for item in databricks_payloads)
    assert all(item["tasks"][0]["max_retries"] == 0 for item in databricks_payloads)
    assert sum(item["timeout_seconds"] for item in databricks_payloads) / 3600 == 80
    attempt_ids = {
        index: publication_latency_handoff_worker_attempt_id(
            production_payloads[index], worker_index=index
        )
        for index in range(16)
    }
    submitted_wire_bodies = []
    next_run_id = 10_000

    def opener(request, *, timeout):
        nonlocal next_run_id
        assert timeout > 0
        submitted_wire_bodies.append(request.data)
        in_flight_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
        assert len(in_flight_ledger.reservations) == 16
        response = {"run_id": next_run_id}
        next_run_id += 1
        return JsonHTTPResponse(response)

    response_values, batch_authorization = (
        reserve_and_submit_publication_latency_handoff_worker_wave(
            DatabricksWorkspaceConfig(
                "https://dbc.example/",
                "secret-token",
            ),
            databricks_payloads,
            ledger_path=ledger_path,
            worker_payloads=production_payloads,
            attempt_ids_by_worker=attempt_ids,
            phase_lease_root=tmp_path / "q8-phase-lease",
            qualification_launch_authorization=launch_authorization,
            opener=opener,
        )
    )
    submit_responses = dict(enumerate(response_values))
    (tmp_path / "q8-phase-lease" / "batch-reserved.json").unlink()
    resumed, replayed_batch = resume_publication_latency_handoff_worker_wave(
        DatabricksWorkspaceConfig("https://dbc.example/", "secret-token"),
        databricks_payloads,
        ledger_path=ledger_path,
        worker_payloads=production_payloads,
        attempt_ids_by_worker=attempt_ids,
        phase_lease_root=tmp_path / "q8-phase-lease",
        qualification_launch_authorization=launch_authorization,
        opener=lambda *_args, **_kwargs: pytest.fail("completed wave must not POST"),
    )
    assert len(resumed) == 16
    assert (
        replayed_batch.batch_authorization.batch_prefix
        == batch_authorization.batch_authorization.batch_prefix
    )
    assert (tmp_path / "q8-phase-lease" / "batch-reserved.json").is_file()
    terminal_runs = {
        index: successful_terminal_run(submit_payload, index)
        for index, submit_payload in enumerate(databricks_payloads)
    }
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert ledger.active_reserved_cluster_hours == 80.0
    assert len(submitted_wire_bodies) == 16
    assert json.loads(submitted_wire_bodies[0]) == databricks_payloads[0]
    assert (
        ledger.reservations[0].submit_payload_sha256
        == sha256(submitted_wire_bodies[0]).hexdigest()
    )

    monkeypatch.setattr(
        generation,
        "_verify_bound_hardware_qualification_file",
        lambda binding: None,
    )
    monkeypatch.setattr(
        generation,
        "_apply_production_generator_environment",
        lambda config: None,
    )
    payload_paths = write_publication_latency_handoff_worker_payloads(
        payloads,
        tmp_path / "payloads",
    )
    tracker = {"created": [], "closed": [], "calls": Counter()}

    def worker_factory(worker_index):
        return StrictQ8PreRopeGenerator(
            config.layout,
            tracker,
            worker_index,
        )

    observed_hardware = {
        "cuda_device_count": 1,
        "cuda_device_name": "NVIDIA L40S",
        "cuda_device_total_memory_bytes": 48 * 1024**3,
        "cuda_major": 8,
        "cuda_minor": 9,
        "gpu_model": "NVIDIA L40S",
        "hardware_target": "aws-g6e-l40s",
        "node_type_id": "g6e.4xlarge",
    }
    for index, path in enumerate(payload_paths):
        run_publication_latency_handoff_worker(
            path,
            expected_worker_payload_sha256=sha256(path.read_bytes()).hexdigest(),
            tokenizer=tokenizer,
            worker_factory=worker_factory,
            local_work_root_override=tmp_path / f"local-worker-{index:02d}",
            hardware_probe=lambda: observed_hardware,
        )
    with pytest.raises(FileExistsError, match="not fresh"):
        run_publication_latency_handoff_worker(
            payload_paths[0],
            expected_worker_payload_sha256=sha256(
                payload_paths[0].read_bytes()
            ).hexdigest(),
            tokenizer=tokenizer,
            worker_factory=worker_factory,
            local_work_root_override=tmp_path / "fresh-gate-local",
            hardware_probe=lambda: observed_hardware,
        )
    retry_run = copy.deepcopy(terminal_runs[0])
    retry_run["tasks"][0]["attempt_number"] = 1
    with pytest.raises(ValueError, match="attempt_number=0"):
        build_publication_latency_handoff_databricks_attestation(
            databricks_payloads[0],
            submit_responses[0],
            retry_run,
            ledger_path=ledger_path,
            durable_output_root=durable_root,
            worker_index=0,
            attempt_id=attempt_ids[0],
            qualification_launch_authorization=launch_authorization,
            submission_authorization=batch_authorization,
        )
    repaired_run = copy.deepcopy(terminal_runs[0])
    repaired_run["repair_history"] = [{"id": 1}]
    with pytest.raises(ValueError, match="must not use a repaired run"):
        build_publication_latency_handoff_databricks_attestation(
            databricks_payloads[0],
            submit_responses[0],
            repaired_run,
            ledger_path=ledger_path,
            durable_output_root=durable_root,
            worker_index=0,
            attempt_id=attempt_ids[0],
            qualification_launch_authorization=launch_authorization,
            submission_authorization=batch_authorization,
        )
    mutated_submit_response = dict(submit_responses[0])
    mutated_submit_response["number_in_job"] = 1
    with pytest.raises(ValueError, match="reservation/submission receipt"):
        build_publication_latency_handoff_databricks_attestation(
            databricks_payloads[0],
            mutated_submit_response,
            terminal_runs[0],
            ledger_path=ledger_path,
            durable_output_root=durable_root,
            worker_index=0,
            attempt_id=attempt_ids[0],
            qualification_launch_authorization=launch_authorization,
            submission_authorization=batch_authorization,
        )

    attestation_records = {}
    attestations = {}
    for index in range(16):
        attestation_record = build_publication_latency_handoff_databricks_attestation(
            databricks_payloads[index],
            submit_responses[index],
            terminal_runs[index],
            ledger_path=ledger_path,
            durable_output_root=durable_root,
            worker_index=index,
            attempt_id=attempt_ids[index],
            qualification_launch_authorization=launch_authorization,
            submission_authorization=batch_authorization,
        )
        attestation = write_publication_latency_handoff_databricks_attestation(
            attestation_record,
            durable_root
            / generation.PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY
            / f"worker-{index:02d}.json",
        )
        assert (
            write_publication_latency_handoff_databricks_attestation(
                attestation_record,
                durable_root
                / generation.PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY
                / f"worker-{index:02d}.json",
            )
            == attestation
        )
        if index == 0:
            mutated_terminal_run = copy.deepcopy(terminal_runs[index])
            mutated_terminal_run["state"]["state_message"] = "digest drift"
            with pytest.raises(ValueError, match="differs from the attestation"):
                reconcile_publication_latency_handoff_worker_attempt_json(
                    ledger_path,
                    worker_index=index,
                    attempt_id=attempt_ids[index],
                    durable_output_root=durable_root,
                    attestation=attestation,
                    terminal_run=mutated_terminal_run,
                )
        ledger = reconcile_publication_latency_handoff_worker_attempt_json(
            ledger_path,
            worker_index=index,
            attempt_id=attempt_ids[index],
            durable_output_root=durable_root,
            attestation=attestation,
            terminal_run=terminal_runs[index],
        )
        attestation_records[index] = attestation_record
        attestations[index] = attestation
    assert ledger.active_reserved_cluster_hours == 0.0
    assert ledger.terminal_actual_cluster_hours == pytest.approx(16 / 3600)
    assert publication_latency_handoff_terminal_actual_gpu_seconds_from_ledger(
        ledger_path,
        attempt_ids_by_worker=attempt_ids,
        durable_output_root=durable_root,
        attestations_by_worker=attestations,
    ) == {index: 1.0 for index in range(16)}

    last_binding = attestations[15]
    last_bytes = last_binding.path.read_bytes()
    duplicate_identity = copy.deepcopy(attestation_records[15])
    duplicate_identity["cloud_execution"]["task_run_id"] = attestation_records[0][
        "cloud_execution"
    ]["task_run_id"]
    duplicate_identity["cloud_execution"]["cluster_id"] = attestation_records[0][
        "cloud_execution"
    ]["cluster_id"]
    duplicate_identity["closed_record_sha256"] = generation._closed_record_sha256(
        duplicate_identity
    )
    duplicate_bytes = (
        json.dumps(duplicate_identity, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    last_binding.path.write_bytes(duplicate_bytes)
    duplicate_bindings = dict(attestations)
    duplicate_bindings[15] = (
        generation.PublicationLatencyHandoffDatabricksAttestationBinding(
            worker_index=15,
            path=last_binding.path,
            file_sha256=sha256(duplicate_bytes).hexdigest(),
            closed_record_sha256=duplicate_identity["closed_record_sha256"],
        )
    )
    with pytest.raises(ValueError, match="unique parent-run, task-run, and cluster"):
        publication_latency_handoff_terminal_actual_gpu_seconds_from_ledger(
            ledger_path,
            attempt_ids_by_worker=attempt_ids,
            durable_output_root=durable_root,
            attestations_by_worker=duplicate_bindings,
        )
    last_binding.path.write_bytes(last_bytes)

    first_result = json.loads(
        (durable_root / "worker-results" / "worker-00.json").read_text()
    )
    first_file = first_result["bundle_files"][0]
    first_path = (
        durable_root
        / "pending"
        / str(first_file["context_tokens"])
        / first_file["relative_name"]
    )
    original = first_path.read_bytes()
    first_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="SHA-256 drift"):
        close_publication_latency_handoff_generation_from_workers(
            plan,
            prepared_input_dir=prepared.output_dir,
            durable_output_root=durable_root,
            tokenizer=tokenizer,
            config=config,
            ledger_path=ledger_path,
            attempt_ids_by_worker=attempt_ids,
            attestations_by_worker=attestations,
        )
    first_path.write_bytes(original)
    result = close_publication_latency_handoff_generation_from_workers(
        plan,
        prepared_input_dir=prepared.output_dir,
        durable_output_root=durable_root,
        tokenizer=tokenizer,
        config=config,
        ledger_path=ledger_path,
        attempt_ids_by_worker=attempt_ids,
        attestations_by_worker=attestations,
    )
    assert sorted(tracker["created"]) == list(range(16))
    assert sorted(tracker["closed"]) == list(range(16))
    assert result.record["accounting"]["cost_model"] == (
        "sum_independent_one_gpu_worker_terminal_lifecycles"
    )
    assert result.record["accounting"]["payload_copy_count_during_closure"] == 0
    assert result.record["coverage"]["generated_task_count"] == 384
    assert result.record["accounting"]["full_launch_throughput_gate_passed"] is True
    reconciled_attempts = result.record["ledger_reconciliation"]["attempts"]
    assert len({item["parent_run_id"] for item in reconciled_attempts}) == 16
    assert len({item["task_run_id"] for item in reconciled_attempts}) == 16
    assert len({item["cluster_id"] for item in reconciled_attempts}) == 16
    assert {item["verification_source"] for item in reconciled_attempts} == {
        "direct_databricks_runs_get"
    }
    def replay_closed_result():
        return generation._replay_closed_publication_latency_handoff_generation(
            plan,
            prepared_input_dir=prepared.output_dir,
            durable_output_root=durable_root,
            tokenizer=tokenizer,
            config=config,
            ledger_snapshot=read_databricks_cluster_hour_ledger_json(ledger_path),
            ledger_path_sha256=databricks_ledger_path_sha256(ledger_path),
            expected_producer_batch_prefix=(
                batch_authorization.batch_authorization.batch_prefix
            ),
            attempt_ids_by_worker=attempt_ids,
            attestations_by_worker=attestations,
            _issuer=generation._POST_CLOSE_REPLAY_ISSUER,
        )

    replayed = replay_closed_result()
    assert replayed.record == result.record
    worker_record_path = durable_root / "worker-records" / "worker-00" / "8192.jsonl"
    worker_record_bytes = worker_record_path.read_bytes()
    worker_record_path.write_bytes(
        bytes([worker_record_bytes[0] ^ 1]) + worker_record_bytes[1:]
    )
    with pytest.raises(ValueError, match="SHA-256 drift"):
        replay_closed_result()
    worker_record_path.write_bytes(worker_record_bytes)
    worker_record_path.unlink()
    with pytest.raises(ValueError):
        replay_closed_result()
    worker_record_path.write_bytes(worker_record_bytes)
    extra_record_path = durable_root / "worker-records" / "unexpected.jsonl"
    extra_record_path.write_bytes(worker_record_bytes)
    with pytest.raises(ValueError):
        replay_closed_result()
    extra_record_path.unlink()
    execution_bytes = result.execution_record_path.read_bytes()
    resealed_execution = json.loads(execution_bytes)
    resealed_execution["unreviewed_resealed_field"] = True
    resealed_execution["closed_record_sha256"] = generation._closed_record_sha256(
        resealed_execution
    )
    result.execution_record_path.write_bytes(
        generation._canonical_json_bytes(resealed_execution, pretty=True)
    )
    with pytest.raises(ValueError, match="closed schema"):
        replay_closed_result()
    result.execution_record_path.write_bytes(execution_bytes)
    assert (
        len(
            list(
                (
                    durable_root
                    / generation.PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY
                ).glob("worker-*.json")
            )
        )
        == 16
    )
    worker_bundle = resolve_publication_latency_worker_handoff_bundle(
        result,
        context_tokens=16_384,
    )
    assert worker_bundle.manifest_path.is_file()
    assert worker_bundle.manifest["identity"]["layout_identity"]["dtype"] == (
        "fp8_e5m2"
    )
    drifted_terminal_runs = copy.deepcopy(terminal_runs)
    drifted_terminal_runs[0]["state"]["state_message"] = "post-closure drift"
    monkeypatch.setattr(
        generation,
        "get_databricks_run",
        lambda _workspace, run_id: drifted_terminal_runs[int(run_id) - 10_000],
    )
    with pytest.raises(ValueError, match="differs from GPU qualification authority"):
        authorize_publication_latency_handoff_serving(
            DatabricksWorkspaceConfig("https://dbc.example/", "secret-token"),
            result,
            ledger_path=ledger_path,
            qualification_launch_authorization=fake_launch_authorization(
                qualification,
                ledger_id="different-qualification-ledger",
            ),
            submission_authorization=batch_authorization,
            attempt_ids_by_worker=attempt_ids,
            attestations_by_worker=attestations,
        )
    with pytest.raises(ValueError, match="differs from durable evidence"):
        authorize_publication_latency_handoff_serving(
            DatabricksWorkspaceConfig("https://dbc.example/", "secret-token"),
            result,
            ledger_path=ledger_path,
            qualification_launch_authorization=launch_authorization,
            submission_authorization=batch_authorization,
            attempt_ids_by_worker=attempt_ids,
            attestations_by_worker=attestations,
        )
    monkeypatch.setattr(
        generation,
        "get_databricks_run",
        lambda _workspace, run_id: terminal_runs[int(run_id) - 10_000],
    )
    serving_authorization = authorize_publication_latency_handoff_serving(
        DatabricksWorkspaceConfig("https://dbc.example/", "secret-token"),
        result,
        ledger_path=ledger_path,
        qualification_launch_authorization=launch_authorization,
        submission_authorization=batch_authorization,
        attempt_ids_by_worker=attempt_ids,
        attestations_by_worker=attestations,
    )
    assert isinstance(
        serving_authorization,
        PublicationLatencyHandoffServingAuthorization,
    )
    assert serving_authorization.ledger_id == launch_authorization.ledger_id
    assert (
        require_publication_latency_handoff_serving_authorization(
            serving_authorization,
            expected_execution_file_sha256=sha256(
                result.execution_record_path.read_bytes()
            ).hexdigest(),
            expected_input_bundle_sha256=prepared.bundle_sha256,
            expected_qualification_closed_record_sha256=(
                qualification.evidence_record["closed_record_sha256"]
            ),
        ).record["closed_record_sha256"]
        == result.record["closed_record_sha256"]
    )
    with pytest.raises(ValueError, match="differs from final artifact pins"):
        require_publication_latency_handoff_serving_authorization(
            serving_authorization,
            expected_execution_file_sha256="0" * 64,
            expected_input_bundle_sha256=prepared.bundle_sha256,
            expected_qualification_closed_record_sha256=(
                qualification.evidence_record["closed_record_sha256"]
            ),
        )
    bundle = resolve_publication_latency_serving_handoff_bundle(
        serving_authorization,
        context_tokens=16_384,
    )
    assert bundle.manifest_path.is_file()
    assert bundle.manifest["identity"]["layout_identity"]["dtype"] == "fp8_e5m2"
    projection = require_publication_latency_full_launch_ready(
        serving_authorization,
        other_terminal_gpu_hours=0.0,
        current_active_reserved_gpu_hours=0.0,
        proposed_full_launch_reserved_gpu_hours=800.0,
    )
    assert projection["projected_unreserved_gpu_hours"] >= 124.0
