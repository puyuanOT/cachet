import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

import cachet.full_score_execution as cachet_full_score
import document_kv_cache.full_score_execution as full_score
from document_kv_cache.serving_env import (
    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
    GPU_RUNTIME_PYTHONWARNINGS,
)
import document_kv_cache.full_score_remote_control as full_score_remote
import document_kv_cache.gpu_qualification_v2 as gpu_qualification_v2
import document_kv_cache.runtime_artifact_closure as runtime_artifact_closure
import document_kv_cache.flashinfer_wheel_repack as flashinfer_wheel_repack
from document_kv_cache.benchmark_handoffs import (
    BenchmarkHandoffEntry,
    BenchmarkHandoffManifest,
    write_benchmark_handoff_manifest_json,
)
from document_kv_cache.benchmarks import (
    NIAH_CELL_IDS,
    SUPPORTED_V1_DATASETS,
    default_dataset_scorer_registry,
)
from document_kv_cache.databricks_resource_ledger import (
    DatabricksClusterHourLedger,
    DatabricksClusterHourReservation,
    DatabricksClusterHourTerminalActual,
    DatabricksRunSubmissionReceipt,
    create_databricks_cluster_hour_ledger_json,
    databricks_cluster_hour_ledger_to_record,
    databricks_ledger_path_sha256,
    databricks_ledger_prefix,
    databricks_submit_payload_reservation,
    read_databricks_cluster_hour_ledger_json,
    record_databricks_run_submission_receipt_json,
    record_databricks_run_terminal_actual_json,
    record_databricks_verified_run_terminal_actual_json,
    reserve_databricks_run_attempt_json,
)
from document_kv_cache.databricks_runs import DatabricksWorkspaceConfig
from document_kv_cache.gpu_qualification import (
    GPUQualificationSelection,
    canonical_gpu_qualification_json,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    GPUQualificationArtifactPinsV2,
)
from document_kv_cache.publication_inputs import (
    build_full_score_shard_plan,
    load_full_score_inventory,
)
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PATCHED_WHEEL_SHA256,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    VLLM_PATCHED_WHEEL_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
)


_VALIDATE_PUBLICATION_FULL_SCORE_INPUTS = (
    full_score._validate_publication_full_score_inputs
)
_REQUIRE_SHARED_DBFS_PATH = full_score._require_shared_dbfs_path


class _CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) for character in text]


class _FakeDatabricksResponse:
    status = 200

    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, amt=-1):
        end = (
            len(self._payload)
            if amt < 0
            else min(len(self._payload), self._offset + amt)
        )
        chunk = self._payload[self._offset : end]
        self._offset = end
        return chunk


class _FakeDatabricksOpener:
    def __init__(self, payload):
        self._payload = payload
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        return _FakeDatabricksResponse(self._payload)


class _RoutingDatabricksOpener:
    def __init__(self, *, submit_payload, run_payload):
        self._submit_payload = submit_payload
        self._run_payload = run_payload
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        payload = (
            self._submit_payload
            if request.get_method() == "POST"
            else self._run_payload
        )
        return _FakeDatabricksResponse(payload)


class _AcceptedButResponseLostOpener:
    def __init__(self):
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        raise TimeoutError("accepted remotely; local response was lost")


@pytest.fixture(autouse=True)
def _bind_full_score_current_user(monkeypatch):
    def require_current_user(_workspace, *, expected_user_name, opener=None):
        assert expected_user_name == "researcher@example.com"
        return {
            "authenticated": True,
            "user_name_sha256": _digest(expected_user_name),
        }

    monkeypatch.setattr(
        full_score,
        "require_databricks_current_user_name",
        require_current_user,
    )
    monkeypatch.setattr(
        full_score_remote,
        "require_databricks_current_user_name",
        require_current_user,
    )


def _digest(label):
    return sha256(label.encode("utf-8")).hexdigest()


def _terminal_run_record(submit_payload, *, run_id):
    tasks = []
    for index, task in enumerate(submit_payload["tasks"]):
        tasks.append(
            {
                "cluster_instance": {"cluster_id": f"cluster-{run_id}-{index}"},
                "new_cluster": task["new_cluster"],
                "spark_python_task": copy.deepcopy(task["spark_python_task"]),
                "end_time": 2_500 + index,
                "run_id": run_id * 100 + index + 1,
                "start_time": 1_000 + index,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "SUCCESS",
                    "state_message": "",
                },
                "task_key": task["task_key"],
            }
        )
    return {
        "cluster_instance": {"cluster_id": f"parent-cluster-{run_id}"},
        "end_time": 3_000,
        "run_id": run_id,
        "run_name": submit_payload["run_name"],
        "run_page_url": f"https://example.invalid/run/{run_id}",
        "start_time": 900,
        "state": {
            "life_cycle_state": "TERMINATED",
            "result_state": "SUCCESS",
            "state_message": "",
        },
        "tasks": tasks,
    }


def _close(record):
    if record.get("record_type") in {
        full_score.FULL_SCORE_READY_SHARD_RECORD_TYPE,
        full_score.FULL_SCORE_SHARD_EVIDENCE_RECORD_TYPE,
    }:
        record.setdefault("runtime_verification", _runtime_verification())
    if (
        record.get("record_type")
        == full_score.FULL_SCORE_SHARD_EVIDENCE_RECORD_TYPE
    ):
        record.setdefault("protocol", full_score._full_score_protocol_record())
    record["closed_record_sha256"] = full_score._closed_record_sha256(record)
    return record


def _runtime_attestation():
    return {
        "base_lock_distribution_count": (
            runtime_artifact_closure.VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT
        ),
        "base_lock_hash_count": (
            runtime_artifact_closure.VLLM_RUNTIME_BASE_LOCK_HASH_COUNT
        ),
        "base_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "cachet_package_version": "0.2.0",
        "flashinfer_annotation": (
            gpu_qualification_v2.GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION
        ),
        "flashinfer_direct_url": "file:///dbfs/runtime/patched-flashinfer.whl",
        "flashinfer_import_ok": True,
        "closure_bound_flashinfer_manifest_closed_record_sha256": (
            flashinfer_wheel_repack.FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
        ),
        "closure_bound_flashinfer_manifest_file_sha256": (
            flashinfer_wheel_repack.FLASHINFER_PATCHED_MANIFEST_FILE_SHA256
        ),
        "closure_bound_vllm_manifest_file_sha256": (
            runtime_artifact_closure.VLLM_PATCHED_MANIFEST_SHA256
        ),
        "flashinfer_member_sha256": (
            flashinfer_wheel_repack.FLASHINFER_TARGET_PATCHED_SHA256
        ),
        "flashinfer_package_version": (
            flashinfer_wheel_repack.FLASHINFER_PACKAGE_VERSION
        ),
        "flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
        "installed_distribution_count": (
            gpu_qualification_v2.GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT
        ),
        "ok": True,
        "packaged_base_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "pip_check_ok": True,
        "runtime_closure_closed_record_sha256": (
            runtime_artifact_closure.RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
        ),
        "runtime_closure_file_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "unexpected_distributions": [],
        "vllm_direct_url": "file:///dbfs/runtime/patched-vllm.whl",
        "vllm_member_sha256": dict(gpu_qualification_v2._VLLM_PATCH_MEMBER_SHA256),
        "vllm_package_version": (gpu_qualification_v2.GPU_QUALIFICATION_VLLM_VERSION),
        "vllm_wheel_sha256": VLLM_PATCHED_WHEEL_SHA256,
        "with_flashinfer_distribution_count": (
            gpu_qualification_v2.GPU_QUALIFICATION_V2_WITH_FLASHINFER_DISTRIBUTION_COUNT
        ),
        "with_vllm_distribution_count": (
            gpu_qualification_v2.GPU_QUALIFICATION_V2_WITH_VLLM_DISTRIBUTION_COUNT
        ),
    }


def _runtime_verification():
    package_sha256 = _digest("cachet-wheel")
    artifacts = {
        "package_wheel_sha256": package_sha256,
        "package_wheel_uri": "dbfs:/runtime/cachet.whl",
        "patched_flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
        "patched_flashinfer_wheel_uri": "dbfs:/runtime/patched-flashinfer.whl",
        "patched_vllm_wheel_sha256": VLLM_PATCHED_WHEEL_SHA256,
        "patched_vllm_wheel_uri": "dbfs:/runtime/patched-vllm.whl",
        "runner_python_file": "dbfs:/runner/full-score.py",
        "runner_sha256": full_score.FULL_SCORE_RUNNER_SHA256,
        "runtime_closure_manifest_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "runtime_closure_manifest_uri": "dbfs:/runtime/closure.json",
        "runtime_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "runtime_lock_uri": "dbfs:/runtime/runtime.lock",
    }
    artifacts["locked_runtime_identity_sha256"] = (
        full_score._locked_runtime_identity_sha256(
            runner_sha256=artifacts["runner_sha256"],
            package_wheel_sha256=package_sha256,
            runtime_lock_sha256=artifacts["runtime_lock_sha256"],
            patched_vllm_wheel_sha256=artifacts["patched_vllm_wheel_sha256"],
            patched_flashinfer_wheel_sha256=(
                artifacts["patched_flashinfer_wheel_sha256"]
            ),
            runtime_closure_manifest_sha256=(
                artifacts["runtime_closure_manifest_sha256"]
            ),
        )
    )
    return full_score._runtime_verification_binding(
        _runtime_attestation(),
        artifacts=artifacts,
    )


def _score_record(dataset, index):
    answer = f"answer-{dataset}-{index}"
    return {
        "dataset": dataset,
        "documents": [
            {
                "document_id": f"doc-{dataset}-{index}",
                "text": f"source text {dataset} {index}",
                "title": f"title-{index}",
            }
        ],
        "example_id": f"{dataset}-{index}",
        "expected_answer": answer,
        "query": f"question {dataset} {index}?",
        "references": [answer],
    }


def _write_sources(root, counts):
    root.mkdir(parents=True)
    paths = {}
    for dataset in SUPPORTED_V1_DATASETS:
        path = root / f"{dataset}.jsonl"
        path.write_text(
            "".join(
                json.dumps(_score_record(dataset, index), sort_keys=True) + "\n"
                for index in range(counts[dataset])
            ),
            encoding="utf-8",
        )
        paths[dataset] = path
    return paths


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    paths = _write_sources(
        tmp_path / "sources",
        {dataset: 5 for dataset in SUPPORTED_V1_DATASETS},
    )
    inventory = load_full_score_inventory(paths, tokenizer=_CharacterTokenizer())
    shard_plan = build_full_score_shard_plan(
        inventory,
        plan_id="full-score-test",
        max_workers=16,
        target_cache_prefix_tokens_per_shard=1,
    )
    execution_plan = full_score.build_full_score_execution_plan(
        inventory,
        shard_plan,
    )
    assert [len(wave["shards"]) for wave in execution_plan["waves"]] == [16, 4]

    package_sha = _digest("cachet-wheel")
    patched_sha = VLLM_PATCHED_WHEEL_SHA256
    selection = GPUQualificationSelection(
        attention_backend="TRITON_ATTN",
        gpu_memory_utilization=0.80,
        generation_hardware_id="aws-g6e-l40s",
        generation_databricks_node_type_id="g6e.4xlarge",
        generation_artifacts_sha256=_digest("matched-generation-artifacts"),
        generation_prefix_tokens_per_second=51.25,
        plan_sha256=_digest("qualification-plan"),
    )
    monkeypatch.setattr(
        full_score,
        "validate_gpu_qualification_evidence_v2_record",
        lambda *args, **kwargs: selection,
    )
    monkeypatch.setattr(
        full_score,
        "_verify_bound_gpu_qualification",
        lambda *args, **kwargs: selection,
    )
    qualification_launch_authorization = SimpleNamespace(
        ledger_id="publication-full-score"
    )
    latency_collection_authorization = SimpleNamespace(
        collection_sha256=_digest("latency-collection")
    )

    def require_test_latency_collection(value, **kwargs):
        if value is not latency_collection_authorization:
            raise TypeError(
                "publication launch requires PublicationLatencyCollectionAuthorization"
            )
        return getattr(
            latency_collection_authorization,
            "ledger_prefix",
            databricks_ledger_prefix(
                read_databricks_cluster_hour_ledger_json(kwargs["ledger_path"])
            ),
        )

    monkeypatch.setattr(
        full_score,
        "require_publication_latency_collection_authorization",
        require_test_latency_collection,
    )

    def require_test_launch_authorization(value, **kwargs):
        if value is not qualification_launch_authorization:
            raise TypeError(
                "publication launch requires GPUQualificationLaunchAuthorization"
            )
        assert kwargs == {
            "expected_evidence_file_sha256": qualification_file_sha,
            "expected_plan_sha256": selection.plan_sha256,
        }
        return selection

    monkeypatch.setattr(
        full_score,
        "require_gpu_qualification_launch_authorization",
        require_test_launch_authorization,
    )
    pins = GPUQualificationArtifactPinsV2(
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_sha256=patched_sha,
        patched_flashinfer_wheel_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
        runtime_closure_manifest_sha256=RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        package_wheel_sha256=package_sha,
        cachet_source_tree_sha256=_digest("source-tree"),
        runner_sha256=_digest("qualification-runner"),
        input_bundle_sha256=GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    )
    qualification_plan = {"closed_record_sha256": selection.plan_sha256}
    qualification_evidence = {"closed_record_sha256": _digest("qualification-evidence")}
    qualification_file_sha = sha256(
        (canonical_gpu_qualification_json(qualification_evidence) + "\n").encode()
    ).hexdigest()
    qualification = full_score.FullScoreGPUQualificationConfig(
        campaign_id="publication-2026",
        plan_uri="dbfs:/qualification/plan.json",
        evidence_uri="dbfs:/qualification/evidence.json",
        evidence_file_sha256=qualification_file_sha,
        plan_record=qualification_plan,
        evidence_record=qualification_evidence,
        artifact_pins=pins,
    )
    runtime = full_score.FullScoreRuntimeConfig(
        python_executable="/local_disk0/cachet-full-score-runtime/bin/python",
        runtime_lock_uri="dbfs:/runtime/runtime.lock",
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_uri="dbfs:/runtime/patched-vllm.whl",
        patched_vllm_wheel_sha256=patched_sha,
        patched_flashinfer_wheel_uri="dbfs:/runtime/patched-flashinfer.whl",
        patched_flashinfer_wheel_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
        runtime_closure_manifest_uri="dbfs:/runtime/closure.json",
        runtime_closure_manifest_sha256=RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        vllm_wheel_install_spec=(
            "vllm @ file:///dbfs/runtime/patched-vllm.whl#sha256=" + patched_sha
        ),
        flashinfer_wheel_install_spec=(
            "flashinfer-python @ file:///dbfs/runtime/patched-flashinfer.whl"
            f"#sha256={FLASHINFER_PATCHED_WHEEL_SHA256}"
        ),
        kv_transfer_config={
            "kv_connector": "DocumentKVConnector",
            "kv_role": "kv_consumer",
            "kv_connector_extra_config": {
                "document_kv.payload_cache_max_bytes": 0,
                "document_kv.require_runtime_handshake": True,
            },
        },
    )
    bundle = full_score.FullScoreWorkerBundleConfig(
        inventory_uri="dbfs:/inputs/inventory.json",
        shard_plan_uri="dbfs:/inputs/shards.json",
        execution_plan_uri="dbfs:/inputs/execution.json",
        source_jsonl_uris={
            dataset: f"dbfs:/inputs/{dataset}.jsonl" for dataset in paths
        },
        durable_output_root="dbfs:/full-score/durable",
        ephemeral_root="/local_disk0/full-score",
        runtime=runtime,
        runner_python_file="dbfs:/runner/full-score.py",
        runner_sha256=full_score.FULL_SCORE_RUNNER_SHA256,
        package_wheel_uri="dbfs:/runtime/cachet.whl",
        package_wheel_sha256=package_sha,
        gpu_qualification=qualification,
    )
    payloads = full_score.build_full_score_worker_payloads(
        inventory,
        shard_plan,
        execution_plan,
        config=bundle,
    )
    job = full_score.DatabricksFullScoreJobConfig(
        runner_python_file=bundle.runner_python_file,
        runner_sha256=bundle.runner_sha256,
        worker_payload_uri_template="dbfs:/workers/{worker_index}.json",
        package_wheel_uri=bundle.package_wheel_uri,
        package_wheel_sha256=bundle.package_wheel_sha256,
        runtime_lock_uri=runtime.runtime_lock_uri,
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_uri=runtime.patched_vllm_wheel_uri,
        patched_vllm_wheel_sha256=runtime.patched_vllm_wheel_sha256,
        patched_flashinfer_wheel_uri=runtime.patched_flashinfer_wheel_uri,
        patched_flashinfer_wheel_sha256=runtime.patched_flashinfer_wheel_sha256,
        runtime_closure_manifest_uri=runtime.runtime_closure_manifest_uri,
        runtime_closure_manifest_sha256=runtime.runtime_closure_manifest_sha256,
        gpu_qualification=qualification,
        single_user_name="researcher@example.com",
    )
    worker_file_root = tmp_path / "worker-payloads"
    worker_file_root.mkdir()
    worker_files = {}
    for payload in payloads:
        worker_label = (
            f"wave-{payload['wave_index']:03d}-{payload['role']}-"
            f"{payload['worker_index']:02d}"
        )
        uri = job.worker_payload_uri_template.format(worker_index=worker_label)
        path = worker_file_root / f"{worker_label}.json"
        path.write_bytes(full_score._canonical_pretty_json_bytes(payload))
        worker_files[uri] = path
    monkeypatch.setattr(
        full_score,
        "_validate_publication_full_score_inputs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        full_score,
        "_require_shared_dbfs_path",
        lambda value, _field_name: str(value),
    )

    def governed_test_file(value, field_name):
        raw = str(value)
        path = worker_files.get(raw, Path(raw))
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{field_name} must be an existing regular test file")
        return path

    monkeypatch.setattr(full_score, "_governed_existing_file", governed_test_file)
    return {
        "bundle": bundle,
        "execution_plan": execution_plan,
        "inventory": inventory,
        "job": job,
        "latency_collection_authorization": latency_collection_authorization,
        "latency_execution_plan_record": {"test": "latency-plan"},
        "payloads": payloads,
        "qualification_launch_authorization": (
            qualification_launch_authorization
        ),
        "shard_plan": shard_plan,
        "tmp_path": tmp_path,
        "worker_files": worker_files,
    }


def _phase_payloads(campaign, wave_index, role):
    return [
        payload
        for payload in campaign["payloads"]
        if payload["wave_index"] == wave_index and payload["role"] == role
    ]


def _volume_campaign(campaign):
    volume = "dbfs:/Volumes/catalog/schema/volume"
    runtime = replace(
        campaign["bundle"].runtime,
        runtime_lock_uri=f"{volume}/runtime/runtime.lock",
        patched_vllm_wheel_uri=f"{volume}/runtime/patched-vllm.whl",
        patched_flashinfer_wheel_uri=f"{volume}/runtime/patched-flashinfer.whl",
        runtime_closure_manifest_uri=f"{volume}/runtime/closure.json",
        vllm_wheel_install_spec=(
            "vllm @ file:///Volumes/catalog/schema/volume/runtime/"
            f"patched-vllm.whl#sha256={campaign['bundle'].runtime.patched_vllm_wheel_sha256}"
        ),
        flashinfer_wheel_install_spec=(
            "flashinfer-python @ file:///Volumes/catalog/schema/volume/runtime/"
            "patched-flashinfer.whl#sha256="
            f"{campaign['bundle'].runtime.patched_flashinfer_wheel_sha256}"
        ),
    )
    qualification = replace(
        campaign["bundle"].gpu_qualification,
        plan_uri=f"{volume}/qualification/plan.json",
        evidence_uri=f"{volume}/qualification/evidence.json",
    )
    bundle = replace(
        campaign["bundle"],
        inventory_uri=f"{volume}/inputs/inventory.json",
        shard_plan_uri=f"{volume}/inputs/shards.json",
        execution_plan_uri=f"{volume}/inputs/execution.json",
        source_jsonl_uris={
            dataset: f"{volume}/inputs/{dataset}.jsonl"
            for dataset in SUPPORTED_V1_DATASETS
        },
        durable_output_root=f"{volume}/full-score",
        runtime=runtime,
        runner_python_file=f"{volume}/runtime/full-score-runner.py",
        package_wheel_uri=f"{volume}/runtime/cachet.whl",
        gpu_qualification=qualification,
    )
    payloads = full_score.build_full_score_worker_payloads(
        campaign["inventory"],
        campaign["shard_plan"],
        campaign["execution_plan"],
        config=bundle,
    )
    job = replace(
        campaign["job"],
        runner_python_file=bundle.runner_python_file,
        worker_payload_uri_template=f"{volume}/workers/{{worker_index}}.json",
        package_wheel_uri=bundle.package_wheel_uri,
        runtime_lock_uri=runtime.runtime_lock_uri,
        patched_vllm_wheel_uri=runtime.patched_vllm_wheel_uri,
        patched_flashinfer_wheel_uri=runtime.patched_flashinfer_wheel_uri,
        runtime_closure_manifest_uri=runtime.runtime_closure_manifest_uri,
        gpu_qualification=qualification,
    )
    worker_files = {}
    worker_root = campaign["tmp_path"] / "volume-worker-payloads"
    worker_root.mkdir(exist_ok=True)
    for payload in payloads:
        label = (
            f"wave-{payload['wave_index']:03d}-{payload['role']}-"
            f"{payload['worker_index']:02d}"
        )
        uri = job.worker_payload_uri_template.format(worker_index=label)
        path = worker_root / f"{label}.json"
        path.write_bytes(full_score._canonical_pretty_json_bytes(payload))
        worker_files[uri] = path
    return {
        **campaign,
        "bundle": bundle,
        "job": job,
        "payloads": payloads,
        "worker_files": worker_files,
    }


def _ready_records(campaign, wave_index):
    wave = campaign["execution_plan"]["waves"][wave_index]
    contract = campaign["payloads"][0]["generator_artifact_contract"]
    records = []
    for worker_index, shard in enumerate(wave["shards"]):
        records.append(
            _close(
                {
                    "closed_record_sha256": "",
                    "execution_plan_sha256": campaign["execution_plan"][
                        "closed_record_sha256"
                    ],
                    "generator_artifact_contract": contract,
                    "lifecycle": ["generate_q8_kv", "commit_ready_shard"],
                    "producer_hardware": {
                        "compute_capability": "8.9",
                        "gpu_count": 1,
                        "gpu_name": "NVIDIA L40S",
                        "hardware_target": "aws-g6e-l40s",
                        "node_type_id": "g6e.4xlarge",
                        "total_memory_bytes": 48 * 1024**3,
                    },
                    "ready_bytes": 1,
                    "ready_bytes_upper_bound": shard["ready_bytes_upper_bound"],
                    "record_type": full_score.FULL_SCORE_READY_SHARD_RECORD_TYPE,
                    "schema_version": full_score.FULL_SCORE_READY_SHARD_SCHEMA_VERSION,
                    "shard_id": shard["shard_id"],
                    "shard_items_sha256": shard["items_sha256"],
                    "wave_index": wave_index,
                    "worker_index": worker_index,
                }
            )
        )
    return records


def _wave_completion(campaign, wave_index):
    attestations = []
    wave = campaign["execution_plan"]["waves"][wave_index]
    for worker_index, shard in enumerate(wave["shards"]):
        attestations.append(
            _close(
                {
                    "closed_record_sha256": "",
                    "evidence_closed_record_sha256": _digest(
                        f"evidence-{shard['shard_id']}"
                    ),
                    "execution_plan_sha256": campaign["execution_plan"][
                        "closed_record_sha256"
                    ],
                    "lifecycle": [
                        "verify_ready_shard",
                        "baseline_inference",
                        "vanilla_inference",
                        "validate_paired_outputs",
                        "commit_durable_evidence",
                        "delete_ephemeral_q8_kv",
                    ],
                    "ready_shard_sha256": _digest(f"ready-{shard['shard_id']}"),
                    "record_type": (
                        full_score.FULL_SCORE_DELETION_ATTESTATION_RECORD_TYPE
                    ),
                    "schema_version": (
                        full_score.FULL_SCORE_DELETION_ATTESTATION_SCHEMA_VERSION
                    ),
                    "shard_id": shard["shard_id"],
                    "wave_index": wave_index,
                    "worker_index": worker_index,
                }
            )
        )
    record = full_score.build_full_score_wave_completion_record(
        campaign["execution_plan"],
        wave_index=wave_index,
        deletion_attestations=attestations,
    )
    record["authorization_scope"] = (
        full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
    )
    bindings = []
    for shard_id in record["shard_ids"]:
        directory = campaign["tmp_path"] / "governed-evidence" / shard_id
        directory.mkdir(parents=True)
        evidence = _close(
            {
                "authorization_scope": (
                    full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
                ),
                "closed_record_sha256": "",
                "execution_plan_sha256": campaign["execution_plan"][
                    "closed_record_sha256"
                ],
                "shard_id": shard_id,
                "wave_index": wave_index,
            }
        )
        deletion = _close(
            {
                "authorization_scope": (
                    full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
                ),
                "closed_record_sha256": "",
                "evidence_closed_record_sha256": evidence["closed_record_sha256"],
                "shard_id": shard_id,
            }
        )
        evidence_path = directory / "evidence.json"
        deletion_path = directory / "deletion-attestation.json"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        deletion_path.write_text(json.dumps(deletion), encoding="utf-8")
        bindings.append(
            {
                "deletion_file_sha256": sha256(deletion_path.read_bytes()).hexdigest(),
                "deletion_path": str(deletion_path),
                "evidence_file_sha256": sha256(evidence_path.read_bytes()).hexdigest(),
                "evidence_path": str(evidence_path),
                "shard_id": shard_id,
            }
        )
    record["governed_evidence_files"] = bindings
    return _close(record)


def _remote_wave_completion_authorization(campaign, wave_index):
    completion = _wave_completion(campaign, wave_index)
    durable_root = "dbfs:/Volumes/catalog/schema/volume/full-score"
    cas_root = campaign["tmp_path"] / f"remote-wave-{wave_index:03d}-cas"
    cas_root.mkdir()
    bindings = []
    compact_files = {}
    for shard_id in completion["shard_ids"]:
        shard = next(
            shard
            for shard in campaign["execution_plan"]["waves"][wave_index]["shards"]
            if shard["shard_id"] == shard_id
        )
        evidence = _close(
            {
                "authorization_scope": (
                    full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
                ),
                "closed_record_sha256": "",
                "durable_evidence_committed": True,
                "execution_plan_sha256": campaign["execution_plan"][
                    "closed_record_sha256"
                ],
                "method_wall_clock": "time.monotonic_ns",
                "method_wall_seconds": {
                    "baseline_prefill": 10.0,
                    "vanilla_prefill": 14.0,
                },
                "paired_examples": [
                    {
                        "dataset": item["dataset"],
                        "example_id": item["example_id"],
                        "methods": {
                            method: {"completion_tokens": 2}
                            for method in full_score.FULL_SCORE_METHODS
                        },
                    }
                    for item in shard["items"]
                ],
                "record_type": full_score.FULL_SCORE_SHARD_EVIDENCE_RECORD_TYPE,
                "schema_version": full_score.FULL_SCORE_SHARD_EVIDENCE_SCHEMA_VERSION,
                "scorers": full_score._scorer_contract_record(),
                "shard_id": shard_id,
                "shard_items_sha256": shard["items_sha256"],
                "wave_index": wave_index,
            }
        )
        deletion = _close(
            {
                "authorization_scope": (
                    full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
                ),
                "closed_record_sha256": "",
                "evidence_closed_record_sha256": evidence[
                    "closed_record_sha256"
                ],
                "execution_plan_sha256": campaign["execution_plan"][
                    "closed_record_sha256"
                ],
                "lifecycle": ["delete_ephemeral_q8_kv"],
                "record_type": (
                    full_score.FULL_SCORE_DELETION_ATTESTATION_RECORD_TYPE
                ),
                "schema_version": (
                    full_score.FULL_SCORE_DELETION_ATTESTATION_SCHEMA_VERSION
                ),
                "shard_id": shard_id,
                "wave_index": wave_index,
            }
        )
        evidence_path = cas_root / f"{shard_id}-evidence.json"
        deletion_path = cas_root / f"{shard_id}-deletion.json"
        evidence_path.write_bytes(
            full_score._canonical_pretty_json_bytes(evidence)
        )
        deletion_path.write_bytes(
            full_score._canonical_pretty_json_bytes(deletion)
        )
        evidence_uri = full_score_remote._consumer_evidence_artifact_uri(
            durable_root,
            wave_index=wave_index,
            shard_id=shard_id,
            filename="evidence.json",
        )
        deletion_uri = full_score_remote._consumer_evidence_artifact_uri(
            durable_root,
            wave_index=wave_index,
            shard_id=shard_id,
            filename="deletion-attestation.json",
        )
        compact_files[evidence_uri] = evidence_path
        compact_files[deletion_uri] = deletion_path
        bindings.append(
            {
                "deletion_file_sha256": sha256(
                    deletion_path.read_bytes()
                ).hexdigest(),
                "deletion_record_sha256": deletion["closed_record_sha256"],
                "deletion_uri": deletion_uri,
                "evidence_file_sha256": sha256(
                    evidence_path.read_bytes()
                ).hexdigest(),
                "evidence_record_sha256": evidence["closed_record_sha256"],
                "evidence_uri": evidence_uri,
                "shard_id": shard_id,
            }
        )
    completion["governed_evidence_files"] = [
        {
            "deletion_file_sha256": binding["deletion_file_sha256"],
            "deletion_path": binding["deletion_uri"],
            "evidence_file_sha256": binding["evidence_file_sha256"],
            "evidence_path": binding["evidence_uri"],
            "shard_id": binding["shard_id"],
        }
        for binding in bindings
    ]
    _close(completion)
    authorization = full_score_remote.FullScoreRemoteTreeAuthorization(
        action="consumer_evidence",
        execution_plan_sha256=campaign["execution_plan"][
            "closed_record_sha256"
        ],
        wave_index=wave_index,
        durable_output_root=durable_root,
        request_sha256=_digest(f"remote-request-{wave_index}"),
        result_uri=(
            f"{durable_root}/control/wave-{wave_index:03d}-completion.json"
        ),
        result_file_sha256=sha256(
            full_score._canonical_pretty_json_bytes(completion)
        ).hexdigest(),
        result_record_sha256=completion["closed_record_sha256"],
        result_record=completion,
        attestation_uri=(
            f"{durable_root}/control/wave-{wave_index:03d}-attestation.json"
        ),
        attestation_file_sha256=_digest(f"remote-attestation-file-{wave_index}"),
        attestation_record_sha256=_digest(
            f"remote-attestation-record-{wave_index}"
        ),
        coordinator_run_id=str(98_000 + wave_index),
        coordinator_run_record_sha256=_digest(f"remote-run-{wave_index}"),
        controller_authorization_record_sha256=_digest(
            f"remote-controller-authorization-{wave_index}"
        ),
        runs_get_receipt_record_sha256=_digest(
            f"remote-runs-get-{wave_index}"
        ),
        phase_terminal_record_sha256=_digest(
            f"remote-consumer-terminal-{wave_index}"
        ),
        evidence_bindings=bindings,
        _issuer=full_score_remote._REMOTE_AUTHORIZATION_ISSUER,
    )
    return completion, authorization, compact_files


def _producer_completion(campaign, wave_index):
    ready_records = _ready_records(campaign, wave_index)
    record = full_score.build_full_score_producer_phase_completion_record(
        campaign["execution_plan"],
        wave_index=wave_index,
        ready_shard_records=ready_records,
    )
    record["authorization_scope"] = (
        full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
    )
    ready_by_id = {item["shard_id"]: item for item in ready_records}
    bindings = []
    for item in record["ready_shards"]:
        ready_path = (
            campaign["tmp_path"]
            / "ready"
            / item["shard_id"]
            / "ready-record.json"
        )
        ready_path.parent.mkdir(parents=True)
        ready_path.write_text(
            json.dumps(ready_by_id[item["shard_id"]]),
            encoding="utf-8",
        )
        bindings.append(
            {
                "file_sha256": sha256(ready_path.read_bytes()).hexdigest(),
                "path": str(ready_path),
                "ready_record_sha256": item["ready_record_sha256"],
                "shard_id": item["shard_id"],
            }
        )
    record["ready_record_files"] = bindings
    return _close(record)


def _matched_blocks(campaign, through_wave):
    blocks = []
    for wave in campaign["execution_plan"]["waves"][: through_wave + 1]:
        for shard in wave["shards"]:
            block = {
                "authorization_scope": (
                    full_score.FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE
                ),
                "billed_gpu_seconds": {
                    "consumer_task": 30.0,
                    "producer": 12.0,
                },
                "billing_source_sha256": {
                    role: _digest(f"billing-{shard['shard_id']}-{role}")
                    for role in ("producer", "consumer_task")
                },
                "cache_prefix_tokens": shard["cache_prefix_tokens"],
                "closed_record_sha256": "",
                "consumer_task_diagnostics": {
                    "attribution": "indivisible_no_per_arm_billed_seconds",
                    "method_wall_clock": "time.monotonic_ns",
                    "method_wall_seconds": {
                        "baseline_prefill": 10.0,
                        "vanilla_prefill": 14.0,
                    },
                    "shared_or_unattributed_seconds": 6.0,
                },
                "deletion_attestation_sha256": _digest(
                    f"deletion-{shard['shard_id']}"
                ),
                "evidence_sha256": _digest(f"evidence-{shard['shard_id']}"),
                "execution_plan_sha256": campaign["execution_plan"][
                    "closed_record_sha256"
                ],
                "matched_status": "success_error_free",
                "natural_prompt_tokens": shard["natural_prompt_tokens"],
                "observed_completion_tokens": {
                    "baseline_prefill": shard["item_count"] * 3,
                    "vanilla_prefill": shard["item_count"] * 5,
                },
                "protocol_sha256": full_score._canonical_sha256(
                    full_score._full_score_protocol_record()
                ),
                "record_type": full_score.FULL_SCORE_MATCHED_BLOCK_RECORD_TYPE,
                "schema_version": full_score.FULL_SCORE_MATCHED_BLOCK_SCHEMA_VERSION,
                "shard_id": shard["shard_id"],
                "shard_items_sha256": shard["items_sha256"],
                "wave_index": wave["wave_index"],
            }
            blocks.append(_close(block))
    return blocks


def _reserve_wave_zero_producer(campaign, *, label):
    attempt_id = f"wave-000-producer-{label}"
    submit_payload = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "producer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=attempt_id,
    )
    ledger_path = campaign["tmp_path"] / f"{label}-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    campaign["latency_collection_authorization"].ledger_prefix = (
        databricks_ledger_prefix(
            read_databricks_cluster_hour_ledger_json(ledger_path)
        )
    )
    submit_path = campaign["tmp_path"] / f"{label}-submit.json"
    submit_path.write_bytes(full_score._canonical_pretty_json_bytes(submit_payload))
    run_id = 81_001
    opener = _RoutingDatabricksOpener(
        submit_payload={"run_id": run_id},
        run_payload=_terminal_run_record(submit_payload, run_id=run_id),
    )
    response, submission_authorization = (
        full_score.reserve_and_submit_governed_full_score_phase_attempt(
            DatabricksWorkspaceConfig(
                "https://dbc.example/", "secret-token"
            ),
            submit_payload,
            ledger_path=ledger_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=campaign[
                "latency_collection_authorization"
            ],
            latency_execution_plan_record=campaign[
                "latency_execution_plan_record"
            ],
            opener=opener,
        )
    )
    assert response == {"run_id": run_id}
    return {
        "ledger_path": ledger_path,
        "opener": opener,
        "run_path": campaign["tmp_path"] / f"{label}-runs-get.json",
        "submission_authorization": submission_authorization,
        "submit_path": submit_path,
        "submit_payload": submit_payload,
        "terminal_path": campaign["tmp_path"] / f"{label}-terminal.json",
    }


def _collect_wave_zero_producer(campaign, *, label):
    reserved = _reserve_wave_zero_producer(campaign, label=label)
    terminal, phase_authorization = (
        full_score.collect_governed_full_score_phase_attempt(
            DatabricksWorkspaceConfig(
                "https://dbc.example/", "secret-token"
            ),
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            ledger_path=reserved["ledger_path"],
            submission_authorization=reserved["submission_authorization"],
            submit_payload_path=reserved["submit_path"],
            control_plane_run_path=reserved["run_path"],
            terminal_record_path=reserved["terminal_path"],
            opener=reserved["opener"],
        )
    )
    return {
        **reserved,
        "phase_authorization": phase_authorization,
        "terminal": terminal,
    }


def test_facade_aliases_the_production_module():
    assert cachet_full_score.__all__ == full_score.__all__
    assert (
        cachet_full_score.build_full_score_execution_plan
        is full_score.build_full_score_execution_plan
    )


def test_production_path_rejects_a_valid_but_nonpublication_inventory(campaign):
    with pytest.raises(ValueError, match="publication full-score inventory closure"):
        _VALIDATE_PUBLICATION_FULL_SCORE_INPUTS(
            campaign["inventory"],
            campaign["shard_plan"],
            campaign["execution_plan"],
        )


def test_publication_path_rejects_a_reclosed_but_reordered_execution_plan(
    campaign,
    monkeypatch,
):
    inventory = campaign["inventory"]
    shard_plan = campaign["shard_plan"]
    execution_plan = campaign["execution_plan"]
    monkeypatch.setattr(
        full_score,
        "FULL_SCORE_PUBLICATION_INVENTORY_SHA256",
        inventory.inventory_sha256,
    )
    monkeypatch.setattr(
        full_score,
        "FULL_SCORE_PUBLICATION_SHARD_PLAN_SHA256",
        shard_plan["closed_record_sha256"],
    )
    monkeypatch.setattr(
        full_score,
        "FULL_SCORE_PUBLICATION_EXECUTION_PLAN_SHA256",
        execution_plan["closed_record_sha256"],
    )
    monkeypatch.setattr(
        full_score,
        "FULL_SCORE_PUBLICATION_ITEM_COUNT",
        len(inventory.items),
    )
    monkeypatch.setattr(
        full_score,
        "FULL_SCORE_PUBLICATION_SHARD_COUNT",
        len(shard_plan["shards"]),
    )
    monkeypatch.setattr(
        full_score,
        "FULL_SCORE_PUBLICATION_CACHE_PREFIX_TOKENS",
        sum(item.cache_prefix_tokens for item in inventory.items),
    )
    monkeypatch.setattr(
        full_score,
        "FULL_SCORE_PUBLICATION_NATURAL_PROMPT_TOKENS",
        sum(item.natural_prompt_tokens for item in inventory.items),
    )
    monkeypatch.setattr(
        full_score,
        "_FULL_SCORE_PUBLICATION_SOURCE_RECORDS",
        tuple(
            (
                source.dataset,
                source.byte_count,
                source.record_count,
                source.source_jsonl_sha256,
                source.source_records_sha256,
                source.identities_sha256,
            )
            for source in inventory.sources
        ),
    )
    _VALIDATE_PUBLICATION_FULL_SCORE_INPUTS(
        inventory,
        shard_plan,
        execution_plan,
    )
    assert execution_plan["record_type"] == "cachet.full_score_execution_plan.v2"
    assert execution_plan["schema_version"] == 2
    assert execution_plan["protocol"] == full_score._full_score_protocol_record()
    protocol_drift = copy.deepcopy(execution_plan)
    protocol_drift["protocol"]["add_special_tokens"] = True
    _close(protocol_drift)
    with pytest.raises(ValueError, match="execution-plan protocol drift"):
        full_score._validate_execution_plan(
            protocol_drift,
            inventory=inventory,
            shard_plan=shard_plan,
        )
    with pytest.raises(ValueError, match="execution-plan protocol drift"):
        full_score._validate_budget_execution_plan(protocol_drift)
    reordered = copy.deepcopy(execution_plan)
    reordered["waves"][0]["shards"].reverse()
    reordered["waves"][0]["shard_ids"].reverse()
    _close(reordered)
    with pytest.raises(ValueError, match="execution-plan closure drift"):
        _VALIDATE_PUBLICATION_FULL_SCORE_INPUTS(
            inventory,
            shard_plan,
            reordered,
        )


def test_worker_payloads_are_token_balanced_closed_and_persistent(
    campaign, monkeypatch
):
    monkeypatch.setenv("FLASHINFER_LOGGING_LEVEL", "DEBUG")
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    payloads = campaign["payloads"]
    assert len(_phase_payloads(campaign, 0, "producer")) == 16
    assert len(_phase_payloads(campaign, 0, "consumer")) == 16
    assert len(_phase_payloads(campaign, 1, "producer")) == 4
    for payload in payloads:
        full_score.validate_full_score_worker_payload(
            payload,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
        )
    producer_plan = full_score.render_full_score_worker_command_plan(
        _phase_payloads(campaign, 0, "producer")[0]
    )
    assert producer_plan["server"] is None
    assert {shard["operation"] for shard in producer_plan["shards"]} == {
        "persistent_generator_api_generate_and_close_ready_shard"
    }
    consumer_plan = full_score.render_full_score_worker_command_plan(
        _phase_payloads(campaign, 0, "consumer")[0]
    )
    assert consumer_plan["server"].count("--model") == 1
    assert "--trust-remote-code" not in consumer_plan["server"]
    for method in ("baseline", "vanilla"):
        command = consumer_plan["shards"][0][method]
        assert command[command.index("--request-parallelism") + 1] == "4"
        assert command[command.index("--max-tokens") + 1] == "64"
        assert command[command.index("--temperature") + 1] == "0"
        assert command[command.index("--repeats") + 1] == "1"
        extra_body_flag = (
            "--baseline-extra-body-json"
            if method == "baseline"
            else "--cache-extra-body-json"
        )
        extra_body = json.loads(command[command.index(extra_body_flag) + 1])
        assert extra_body["add_special_tokens"] is False
        assert "--cache-runtime-prompt" not in command
        assert not any("trunc" in argument or "padding" in argument for argument in command)
    environment = full_score._worker_environment(campaign["bundle"].runtime)
    assert (
        environment["FLASHINFER_LOGGING_LEVEL"]
        == GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL
    )
    assert environment["PYTHONWARNINGS"] == GPU_RUNTIME_PYTHONWARNINGS
    assert environment["CACHET_TRANSFORMERS_DEVICE_MAP"] == "auto"
    assert json.loads(environment["CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON"]) == (
        full_score.FULL_SCORE_GENERATOR_QUANTIZATION_CONFIG
    )


def test_publication_paths_are_confined_and_local_fixtures_are_nonauthorizing(
    campaign,
):
    with pytest.raises(ValueError, match="shared DBFS"):
        _REQUIRE_SHARED_DBFS_PATH("/tmp/not-durable", "durable_output_root")
    local_bundle = replace(
        campaign["bundle"],
        authorization_scope=full_score.FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
        durable_output_root=str(campaign["tmp_path"] / "local-durable"),
        ephemeral_root=str(campaign["tmp_path"] / "local-ephemeral"),
        source_jsonl_uris={
            dataset: str(campaign["tmp_path"] / f"{dataset}.jsonl")
            for dataset in SUPPORTED_V1_DATASETS
        },
    )
    local_payloads = full_score.build_full_score_worker_payloads(
        campaign["inventory"],
        campaign["shard_plan"],
        campaign["execution_plan"],
        config=local_bundle,
    )
    with pytest.raises(ValueError, match="rejects local-fixture payloads"):
        full_score.build_databricks_full_score_run_submit_payload(
            campaign["job"],
            [
                payload
                for payload in local_payloads
                if payload["wave_index"] == 0 and payload["role"] == "producer"
            ],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            attempt_id="local-fixture-rejected",
        )


def test_connector_proof_requires_one_exact_full_q8_load_per_vanilla_request(
    campaign,
):
    shard = campaign["execution_plan"]["waves"][0]["shards"][0]
    pairs = []
    telemetry = []
    for item in shard["items"]:
        suffix = f"{item['dataset']}-{item['example_id']}"
        artifact_id = f"artifact-{suffix}"
        vanilla_request_id = f"vanilla-{suffix}"
        pairs.append(
            {
                "dataset": item["dataset"],
                "example_id": item["example_id"],
                "methods": {
                    "baseline_prefill": {
                        "artifact_id": None,
                        "request_id": "",
                    },
                    "vanilla_prefill": {
                        "artifact_id": artifact_id,
                        "request_id": vanilla_request_id,
                    },
                },
            }
        )
        tokens = item["cache_prefix_tokens"]
        runtime_bytes = (
            tokens * full_score.FULL_SCORE_Q8_BYTES_PER_CACHE_PREFIX_TOKEN
        )
        telemetry.append(
            {
                "benchmark_request_id": vanilla_request_id,
                "cache_state_attestation": {
                    "artifact_id": artifact_id,
                    "cache_method": "vanilla_prefill",
                    "decoded_runtime_bytes": runtime_bytes,
                    "expected_runtime_bytes": runtime_bytes,
                    "expected_tokens": tokens,
                    "loaded_tokens": tokens,
                    "payload_cache_hit": False,
                    "successful_loads": 1,
                },
                "counts": {
                    "decoded_runtime_payload_bytes": runtime_bytes,
                    "expected_runtime_payload_bytes": runtime_bytes,
                    "handoff_total_tokens": tokens,
                    "layers_loaded": full_score.FULL_SCORE_MODEL_NUM_LAYERS,
                    "token_count": tokens,
                },
                "event": "load_request",
                "layout": {
                    "bytes_per_token": (
                        full_score.FULL_SCORE_Q8_BYTES_PER_CACHE_PREFIX_TOKEN
                    ),
                    "dtype": full_score.FULL_SCORE_KV_DTYPE,
                    "model_id": full_score.FULL_SCORE_MODEL_ID,
                    "num_layers": full_score.FULL_SCORE_MODEL_NUM_LAYERS,
                },
                "record_type": "document_kv.vllm_native_provider_load.v1",
                "provider_factory": (
                    "vllm_kv_injection.vllm_native_provider:"
                    "build_document_kv_provider"
                ),
                "success": True,
            }
        )
    telemetry_path = campaign["tmp_path"] / "connector-telemetry.jsonl"
    telemetry_path.write_text(
        "".join(json.dumps(record) + "\n" for record in telemetry),
        encoding="utf-8",
    )
    proof = full_score.build_full_score_connector_proof(
        telemetry_path,
        paired_examples=pairs,
        shard=shard,
    )
    assert proof["load_count"] == len(shard["items"])
    telemetry[0]["counts"]["layers_loaded"] -= 1
    telemetry_path.write_text(
        "".join(json.dumps(record) + "\n" for record in telemetry),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="count/layout coverage drift"):
        full_score.build_full_score_connector_proof(
            telemetry_path,
            paired_examples=pairs,
            shard=shard,
        )


def test_niah_measurement_cell_and_raw_protocol_must_replay_from_bound_source():
    key = ("niah", "niah-example")
    measurement = SimpleNamespace(
        arm_id=full_score.BASELINE_PREFILL_ARM,
        artifact_id="",
        cache_method=None,
        dataset=key[0],
        error=None,
        example_id=key[1],
        expected_answer="needle",
        metadata={
            "logical_prompt_sha256": _digest("natural-prompt"),
            "niah_cell_id": NIAH_CELL_IDS[1],
        },
        references=("needle",),
        repeat_index=1,
        request_id="",
    )
    example = SimpleNamespace(
        expected_answer="needle",
        metadata={"niah_cell_id": NIAH_CELL_IDS[0]},
        references=("needle",),
    )
    with pytest.raises(ValueError, match="differs from bound source"):
        full_score._validated_method_measurements(
            [measurement],
            method="baseline_prefill",
            shard_id="shard-000",
            expected_items={
                key: {"natural_prompt_sha256": _digest("natural-prompt")}
            },
            examples={key: example},
        )

    injected_baseline_artifact = copy.deepcopy(measurement)
    injected_baseline_artifact.artifact_id = _digest("injected-baseline-artifact")
    with pytest.raises(ValueError, match="unexpectedly declares an artifact ID"):
        full_score._validated_method_measurements(
            [injected_baseline_artifact],
            method="baseline_prefill",
            shard_id="shard-000",
            expected_items={
                key: {"natural_prompt_sha256": _digest("natural-prompt")}
            },
            examples={key: example},
        )

    protocol_source = _score_record("niah", 0)
    protocol_example = full_score._example_from_record(
        protocol_source,
        default_dataset="niah",
        record_index=1,
        require_dataset=True,
    )
    protocol_key = ("niah", protocol_example.example_id)
    protocol_examples = {protocol_key: protocol_example}

    def protocol_result(method):
        arm_id = (
            full_score.BASELINE_PREFILL_ARM
            if method == "baseline_prefill"
            else full_score.FULL_SCORE_VANILLA_ARM_ID
        )
        measurements = ()
        manifest = full_score._build_expected_full_score_benchmark_manifest(
            method=method,
            shard_id="shard-000",
            protocol_examples=protocol_examples,
            measurements=measurements,
        )
        return SimpleNamespace(
            baseline_arm_id=arm_id,
            evidence_policy="smoke",
            execution_isolation_mode="shared_process_sequential",
            experiment_manifest=manifest,
            interleave_examples=False,
            isolate_arms=True,
            prefix_cache_salt_mode="per_request",
            repeats=full_score.FULL_SCORE_PASSES_PER_METHOD,
            request_parallelism=full_score.FULL_SCORE_REQUEST_PARALLELISM,
            measurements=measurements,
            seed=None,
            shuffle=False,
            suite=SimpleNamespace(
                datasets=("niah",),
                examples=(object(),),
                hardware_target="aws-g6-l4",
                model_id=full_score.FULL_SCORE_SERVED_MODEL_NAME,
                suite_id=(
                    f"{full_score.FULL_SCORE_PROTOCOL_ID}:shard-000:{method}"
                ),
            ),
            warmups=0,
        )

    shard = {"shard_id": "shard-000"}
    expected_items = {
        protocol_key: {"natural_prompt_sha256": _digest("natural-prompt")}
    }
    for method in full_score.FULL_SCORE_METHODS:
        result = protocol_result(method)
        assert result.experiment_manifest.arms[0].request_customization_digest == (
            "440181b5f7930106194b542de751661bbd5662a071e7d10b64cf8172ac29774f"
        )
        full_score._validate_full_score_benchmark_protocol(
            result,
            method=method,
            shard=shard,
            expected_items=expected_items,
            protocol_examples=protocol_examples,
        )
        object.__setattr__(result.experiment_manifest, "temperature", 0.5)
        with pytest.raises(ValueError, match="complete manifest protocol drift"):
            full_score._validate_full_score_benchmark_protocol(
                result,
                method=method,
                shard=shard,
                expected_items=expected_items,
                protocol_examples=protocol_examples,
            )

        arm_drift = protocol_result(method)
        object.__setattr__(
            arm_drift.experiment_manifest.arms[0],
            "method_version",
            "substituted-method-version",
        )
        with pytest.raises(ValueError, match="complete manifest protocol drift"):
            full_score._validate_full_score_benchmark_protocol(
                arm_drift,
                method=method,
                shard=shard,
                expected_items=expected_items,
                protocol_examples=protocol_examples,
            )

        runtime_drift = protocol_result(method)
        object.__setattr__(
            runtime_drift.experiment_manifest.arms[0].runtime_environment,
            "lora_id",
            "substituted-lora",
        )
        with pytest.raises(ValueError, match="complete manifest protocol drift"):
            full_score._validate_full_score_benchmark_protocol(
                runtime_drift,
                method=method,
                shard=shard,
                expected_items=expected_items,
                protocol_examples=protocol_examples,
            )

    concurrency_drift = protocol_result("baseline_prefill")
    concurrency_drift.request_parallelism = 1
    with pytest.raises(ValueError, match="execution protocol drift"):
        full_score._validate_full_score_benchmark_protocol(
            concurrency_drift,
            method="baseline_prefill",
            shard=shard,
            expected_items=expected_items,
            protocol_examples=protocol_examples,
        )

    logical_prompt_sha256 = _digest("logical-prompt")
    runtime_prompt_sha256 = _digest("runtime-prompt")
    expected_cache_salt = "shard-000:vanilla:exact-cache-salt"
    vanilla_prompt_measurement = SimpleNamespace(
        metadata={
            "kv_transfer_params_attached": "true",
            "logical_prompt_sha256": logical_prompt_sha256,
            "logical_prompt_tokens": "20",
            "physical_transform_id": "cachet.vanilla.per_document_segments",
            "physical_transform_version": "1",
            "prefix_cache_salt": expected_cache_salt,
            "prefix_cache_salt_attached": "true",
            "prompt_text_mode": "logical",
            "prompt_token_source": "server_usage",
            "request_id": "request-1",
            "request_mode": "completion",
            "request_payload_endpoint": "/v1/completions",
            "request_payload_add_special_tokens": "false",
            "request_payload_keys": (
                "add_special_tokens,cache_salt,kv_transfer_params,max_tokens,"
                "model,prompt,request_id,stream,stream_options,temperature"
            ),
            "request_payload_kv_transfer_param_keys": "document_kv.request_id",
            "request_payload_max_token_fields": "max_tokens",
            "request_payload_max_tokens": "64",
            "request_payload_prompt_chars": "24",
            "request_payload_prompt_sha256": logical_prompt_sha256,
            "runtime_prompt_sha256": runtime_prompt_sha256,
            "runtime_prompt_tokens": "20",
            "server": "openai-compatible",
            "server_usage_prompt_tokens": "20",
            "server_usage_prompt_tokens_present": "true",
            "stream": "true",
        },
        prompt_tokens=20,
        request_id="request-1",
    )
    prompt_protocol = {
        "method": "vanilla_prefill",
        "expected_logical_prompt_sha256": logical_prompt_sha256,
        "expected_runtime_prompt_sha256": runtime_prompt_sha256,
        "expected_request_prompt_chars": 24,
        "expected_prefix_cache_salt": expected_cache_salt,
        "expected_kv_parameter_keys": "document_kv.request_id",
        "expected_logical_prompt_tokens": 20,
    }
    full_score._validate_full_score_measurement_prompt_protocol(
        vanilla_prompt_measurement,
        **prompt_protocol,
    )
    enabled_special_tokens = copy.deepcopy(vanilla_prompt_measurement)
    enabled_special_tokens.metadata["request_payload_add_special_tokens"] = "true"
    with pytest.raises(ValueError, match="prompt-delivery protocol drift"):
        full_score._validate_full_score_measurement_prompt_protocol(
            enabled_special_tokens,
            **prompt_protocol,
        )
    runtime_delivery = copy.deepcopy(vanilla_prompt_measurement)
    runtime_delivery.metadata["prompt_text_mode"] = "runtime"
    with pytest.raises(ValueError, match="prompt-delivery protocol drift"):
        full_score._validate_full_score_measurement_prompt_protocol(
            runtime_delivery,
            **prompt_protocol,
        )
    sent_runtime_prompt = copy.deepcopy(vanilla_prompt_measurement)
    sent_runtime_prompt.metadata[
        "request_payload_prompt_sha256"
    ] = runtime_prompt_sha256
    with pytest.raises(ValueError, match="prompt-delivery protocol drift"):
        full_score._validate_full_score_measurement_prompt_protocol(
            sent_runtime_prompt,
            **prompt_protocol,
        )
    noncanonical_usage = copy.deepcopy(vanilla_prompt_measurement)
    noncanonical_usage.metadata["server_usage_prompt_tokens"] = "04"
    with pytest.raises(ValueError, match="not canonical"):
        full_score._validate_full_score_measurement_prompt_protocol(
            noncanonical_usage,
            **prompt_protocol,
        )


def test_governed_niah_source_row_replays_from_inventory_hash(tmp_path):
    source = _score_record("niah", 97)
    source["metadata"] = {"niah_cell_id": NIAH_CELL_IDS[0]}
    path = tmp_path / "niah.jsonl"
    path.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    shard = {
        "item_count": 1,
        "items": [
            {
                "dataset": "niah",
                "example_id": source["example_id"],
                "source_record_sha256": full_score._canonical_sha256(source),
            }
        ],
    }
    records, examples = full_score._load_governed_ready_source_records(
        {"niah": path},
        shard=shard,
    )
    assert records[("niah", source["example_id"])] == source
    assert (
        examples[("niah", source["example_id"])].metadata["niah_cell_id"]
        == NIAH_CELL_IDS[0]
    )

    source["metadata"]["niah_cell_id"] = NIAH_CELL_IDS[1]
    path.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source-record hash drift"):
        full_score._load_governed_ready_source_records(
            {"niah": path},
            shard=shard,
        )


def test_governed_ready_and_handoff_replay_bind_source_artifact_and_files(
    campaign,
):
    wave = campaign["execution_plan"]["waves"][0]
    shard = wave["shards"][0]
    item = shard["items"][0]
    dataset = item["dataset"]
    example_id = item["example_id"]
    source_index = int(example_id.rsplit("-", 1)[1])
    source = _score_record(dataset, source_index)
    input_path = campaign["tmp_path"] / "preserved-input.jsonl"
    input_path.write_text(
        json.dumps(source, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert full_score._canonical_sha256(source) == item["source_record_sha256"]

    artifact_id = _digest(f"artifact-{dataset}-{example_id}")
    entry = BenchmarkHandoffEntry(
        dataset=dataset,
        example_id=example_id,
        request_id=f"handoff-{dataset}-{example_id}",
        handoff_json="/dbfs/full-score/ready/handoff.json",
        cache_method="vanilla_prefill",
        artifact_id=artifact_id,
    )
    manifest_path = campaign["tmp_path"] / "preserved-manifest.json"
    write_benchmark_handoff_manifest_json(
        BenchmarkHandoffManifest(entries=(entry,)),
        manifest_path,
    )
    enriched = dict(source)
    enriched["arm_kv_transfer_params"] = {
        full_score.FULL_SCORE_VANILLA_ARM_ID: entry.kv_transfer_params()
    }
    enriched_path = campaign["tmp_path"] / "preserved-enriched.jsonl"
    enriched_path.write_text(
        json.dumps(enriched, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    q8_raw = b"bound-q8-payload"
    ready_file_sources = {
        f"inputs/{dataset}.jsonl": input_path.read_bytes(),
        f"enriched/{dataset}.jsonl": enriched_path.read_bytes(),
        f"manifests/{dataset}.json": manifest_path.read_bytes(),
        f"q8-kv/{dataset}/cachet-benchmark.kvpack": q8_raw,
    }
    ready_files = [
        {
            "byte_count": len(raw),
            "relative_path": relative_path,
            "sha256": sha256(raw).hexdigest(),
        }
        for relative_path, raw in sorted(ready_file_sources.items())
    ]
    ready = _close(
        {
            "closed_record_sha256": "",
            "execution_plan_sha256": campaign["execution_plan"][
                "closed_record_sha256"
            ],
            "files": ready_files,
            "files_sha256": full_score._canonical_sha256(ready_files),
            "generator_artifact_contract": campaign["payloads"][0][
                "generator_artifact_contract"
            ],
            "inventory_sha256": campaign["inventory"].inventory_sha256,
            "lifecycle": ["generate_q8_kv", "commit_ready_shard"],
            "producer_hardware": {
                "compute_capability": "8.9",
                "gpu_count": 1,
                "gpu_name": full_score.FULL_SCORE_PRODUCER_GPU_NAME,
                "hardware_target": full_score.FULL_SCORE_PRODUCER_HARDWARE_TARGET,
                "node_type_id": full_score.FULL_SCORE_PRODUCER_NODE_TYPE_ID,
                "total_memory_bytes": 48 * 1024**3,
            },
            "ready_bytes": sum(record["byte_count"] for record in ready_files),
            "ready_bytes_upper_bound": shard["ready_bytes_upper_bound"],
            "record_type": full_score.FULL_SCORE_READY_SHARD_RECORD_TYPE,
            "schema_version": full_score.FULL_SCORE_READY_SHARD_SCHEMA_VERSION,
            "shard_id": shard["shard_id"],
            "shard_items_sha256": shard["items_sha256"],
            "shard_plan_sha256": campaign["shard_plan"]["closed_record_sha256"],
            "wave_index": 0,
            "worker_index": next(
                assignment["worker_index"]
                for assignment in wave["producer_assignments"]
                if shard["shard_id"] in assignment["shard_ids"]
            ),
        }
    )
    preserved = {}
    resolved = {}
    for evidence_name, path, ready_relative in (
        (f"input_{dataset}", input_path, f"inputs/{dataset}.jsonl"),
        (f"enriched_{dataset}", enriched_path, f"enriched/{dataset}.jsonl"),
        (
            f"handoff_manifest_{dataset}",
            manifest_path,
            f"manifests/{dataset}.json",
        ),
    ):
        ready_file = next(
            record
            for record in ready_files
            if record["relative_path"] == ready_relative
        )
        preserved[evidence_name] = {
            **ready_file,
            "relative_path": path.name,
        }
        resolved[evidence_name] = path
        evidence = {
            "preserved_files": preserved,
            "ready_shard_sha256": ready["closed_record_sha256"],
            "runtime_verification": ready["runtime_verification"],
            "wave_index": 0,
        }
    full_score._validate_governed_ready_manifest_replay(
        ready,
        evidence=evidence,
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        shard=shard,
        resolved_files=resolved,
        datasets=[dataset],
    )
    resealed_ready = copy.deepcopy(ready)
    resealed_runtime = resealed_ready["runtime_verification"]
    resealed_runtime["attestation"]["vllm_direct_url"] = (
        "file:///attacker/resealed-vllm.whl"
    )
    resealed_runtime["attestation_sha256"] = full_score._canonical_sha256(
        resealed_runtime["attestation"]
    )
    resealed_runtime["file_sha256"] = sha256(
        full_score._canonical_pretty_json_bytes(resealed_runtime["attestation"])
    ).hexdigest()
    _close(resealed_ready)
    resealed_evidence = {
        **evidence,
        "ready_shard_sha256": resealed_ready["closed_record_sha256"],
        "runtime_verification": resealed_runtime,
    }
    with pytest.raises(ValueError, match="vllm_direct_url differs"):
        full_score._validate_governed_ready_manifest_replay(
            resealed_ready,
            evidence=resealed_evidence,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            shard=shard,
            resolved_files=resolved,
            datasets=[dataset],
        )
    source_records, _examples = full_score._load_governed_ready_source_records(
        {dataset: input_path},
        shard=shard,
    )
    pairs = [
        {
            "dataset": dataset,
            "example_id": example_id,
            "methods": {"vanilla_prefill": {"artifact_id": artifact_id}},
        }
    ]
    full_score._validate_governed_handoff_replay(
        source_records=source_records,
        enriched_paths={dataset: enriched_path},
        manifest_paths={dataset: manifest_path},
        paired_examples=pairs,
    )

    tampered_evidence = copy.deepcopy(evidence)
    tampered_evidence["preserved_files"][f"input_{dataset}"]["sha256"] = _digest(
        "tampered-input"
    )
    with pytest.raises(ValueError, match="differs from ready-shard closure"):
        full_score._validate_governed_ready_manifest_replay(
            ready,
            evidence=tampered_evidence,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            shard=shard,
            resolved_files=resolved,
            datasets=[dataset],
        )
    pairs[0]["methods"]["vanilla_prefill"]["artifact_id"] = _digest(
        "different-artifact"
    )
    with pytest.raises(ValueError, match="differs from handoff manifest"):
        full_score._validate_governed_handoff_replay(
            source_records=source_records,
            enriched_paths={dataset: enriched_path},
            manifest_paths={dataset: manifest_path},
            paired_examples=pairs,
        )


def test_databricks_renders_two_independent_bounded_phases(campaign, monkeypatch):
    producers = _phase_payloads(campaign, 0, "producer")
    with pytest.raises(
        TypeError,
        match="publication launch requires GPUQualificationLaunchAuthorization",
    ):
        full_score.build_databricks_full_score_run_submit_payload(
            campaign["job"],
            producers,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            qualification_launch_authorization={},
            attempt_id="invalid-qualification-render",
        )
    producer_run = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        producers,
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id="wave-000-producer",
    )
    assert set(producer_run) == {
        "idempotency_token",
        "run_name",
        "tasks",
        "timeout_seconds",
    }
    assert len(producer_run["idempotency_token"]) == 64
    assert producer_run["timeout_seconds"] == 21_600
    assert len(producer_run["tasks"]) == 16
    assert all(task["timeout_seconds"] == 21_600 for task in producer_run["tasks"])
    assert all(task["max_retries"] == 0 for task in producer_run["tasks"])
    assert {task["new_cluster"]["node_type_id"] for task in producer_run["tasks"]} == {
        "g6e.4xlarge"
    }
    assert {
        task["new_cluster"]["data_security_mode"]
        for task in producer_run["tasks"]
    } == {"SINGLE_USER"}
    assert {
        task["new_cluster"]["single_user_name"]
        for task in producer_run["tasks"]
    } == {"researcher@example.com"}
    assert all("depends_on" not in task for task in producer_run["tasks"])
    assert all(
        set(task)
        == {
            "max_retries",
            "new_cluster",
            "spark_python_task",
            "task_key",
            "timeout_seconds",
        }
        for task in producer_run["tasks"]
    )
    producer_reservation = databricks_submit_payload_reservation(
        producer_run,
        attempt_id="wave-000-producer",
        workload_id="full-score",
    )
    assert producer_reservation.reserved_cluster_hours == 96.0
    mixed_principals = copy.deepcopy(producer_run)
    mixed_principals["tasks"][0]["new_cluster"]["single_user_name"] = (
        "attacker@example.com"
    )
    with pytest.raises(ValueError, match="share one exact principal"):
        full_score._validated_full_score_phase_submit_payload(
            campaign["execution_plan"],
            mixed_principals,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
        )
    for invalid_principal in (
        "researcher\x00@example.com",
        "researcher\x7f@example.com",
    ):
        with pytest.raises(ValueError, match="normalized non-empty string"):
            replace(
                campaign["job"],
                single_user_name=invalid_principal,
            )

    consumers = _phase_payloads(campaign, 0, "consumer")
    with pytest.raises(ValueError, match="producer-phase completion"):
        full_score.build_databricks_full_score_run_submit_payload(
            campaign["job"],
            consumers,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization={},
            attempt_id="wave-000-consumer-missing-completion",
        )
    producer_completion = _producer_completion(campaign, 0)
    producer_completion_path = (
        campaign["tmp_path"] / "wave-000-producer-completion.json"
    )
    producer_completion_path.write_bytes(
        full_score._canonical_pretty_json_bytes(producer_completion)
    )

    monkeypatch.setattr(
        full_score,
        "_validate_governed_producer_ready_phase",
        lambda *_args, **_kwargs: None,
    )
    consumer_run = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        consumers,
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id="wave-000-consumer",
        producer_phase_completion=producer_completion,
        producer_phase_completion_uri=str(producer_completion_path),
    )
    assert consumer_run["timeout_seconds"] == 21_600
    assert len(consumer_run["tasks"]) == 16
    assert {task["new_cluster"]["node_type_id"] for task in consumer_run["tasks"]} == {
        "g6.8xlarge"
    }
    assert all("depends_on" not in task for task in consumer_run["tasks"])
    assert all(
        "--producer-phase-completion-json"
        in task["spark_python_task"]["parameters"]
        for task in consumer_run["tasks"]
    )
    consumer_reservation = databricks_submit_payload_reservation(
        consumer_run,
        attempt_id="wave-000-consumer",
        workload_id="full-score",
    )
    assert consumer_reservation.reserved_cluster_hours == 96.0
    with pytest.raises(ValueError, match="exactly one phase"):
        full_score.build_databricks_full_score_run_submit_payload(
            campaign["job"],
            [*producers, *consumers],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            attempt_id="mixed-phase-rejected",
        )


def test_full_score_freezes_six_hour_task_and_run_timeout(campaign):
    assert full_score.FULL_SCORE_DATABRICKS_TASK_TIMEOUT_SECONDS == 21_600
    assert campaign["job"].run_timeout_seconds == 21_600
    assert campaign["bundle"].runtime.generator_timeout_seconds == 21_600.0
    assert full_score._command_timeout(
        ["python", "-m", "document_kv_cache.benchmark_runner"]
    ) == 21_600.0
    assert full_score.full_score_wave_worst_case_gpu_hours(
        _phase_payloads(campaign, 0, "producer")
    ) == 96.0
    with pytest.raises(ValueError, match="six hours"):
        replace(campaign["job"], run_timeout_seconds=14_400)
    with pytest.raises(ValueError, match="six hours"):
        replace(
            campaign["bundle"].runtime,
            generator_timeout_seconds=14_400.0,
        )
    with pytest.raises(ValueError, match="six hours"):
        full_score.full_score_wave_worst_case_gpu_hours(
            _phase_payloads(campaign, 0, "producer"),
            task_timeout_seconds=14_400,
        )
    with pytest.raises(ValueError, match="max_model_len is frozen"):
        replace(
            campaign["bundle"].runtime,
            max_model_len=campaign["bundle"].runtime.max_model_len + 1,
        )
    with pytest.raises(ValueError, match="max_num_seqs is frozen"):
        replace(
            campaign["bundle"].runtime,
            max_num_seqs=full_score.FULL_SCORE_REQUEST_PARALLELISM + 1,
        )
    transfer_with_fallback = copy.deepcopy(
        campaign["bundle"].runtime.kv_transfer_config
    )
    transfer_with_fallback["kv_connector_extra_config"][
        "document_kv.allow_silent_fallback"
    ] = True
    with pytest.raises(ValueError, match="extra config schema drift"):
        replace(
            campaign["bundle"].runtime,
            kv_transfer_config=transfer_with_fallback,
        )


def test_consumer_reservation_replays_completion_instead_of_trusting_closure(
    campaign,
    monkeypatch,
):
    completion = _producer_completion(campaign, 0)
    completion_path = campaign["tmp_path"] / "producer-completion-forged.json"
    completion_path.write_bytes(full_score._canonical_pretty_json_bytes(completion))
    monkeypatch.setattr(
        full_score,
        "_validate_governed_producer_ready_phase",
        lambda *_args, **_kwargs: None,
    )
    submit_payload = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "consumer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id="forged-consumer-attempt",
        producer_phase_completion=completion,
        producer_phase_completion_uri=str(completion_path),
    )
    forged = _close({"closed_record_sha256": "", "evil": "self-closed mapping"})
    completion_path.write_bytes(full_score._canonical_pretty_json_bytes(forged))
    for task in submit_payload["tasks"]:
        parameters = task["spark_python_task"]["parameters"]
        digest_index = parameters.index(
            "--expected-producer-phase-completion-sha256"
        ) + 1
        parameters[digest_index] = forged["closed_record_sha256"]
    ledger_path = campaign["tmp_path"] / "forged-consumer-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    before = ledger_path.read_bytes()
    with pytest.raises(ValueError, match="idempotency token drift"):
        full_score.reserve_governed_full_score_phase_attempt(
            ledger_path,
            submit_payload,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="consumer",
            attempt_id="forged-consumer-attempt",
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization={},
        )
    assert ledger_path.read_bytes() == before


def test_consumer_reservation_requires_exact_live_ready_trees(campaign, monkeypatch):
    completion = _producer_completion(campaign, 0)
    completion_path = campaign["tmp_path"] / "producer-completion-no-trees.json"
    completion_path.write_bytes(full_score._canonical_pretty_json_bytes(completion))
    with monkeypatch.context() as render_patch:
        render_patch.setattr(
            full_score,
            "_validate_governed_producer_ready_phase",
            lambda *_args, **_kwargs: None,
        )
        submit_payload = full_score.build_databricks_full_score_run_submit_payload(
            campaign["job"],
            _phase_payloads(campaign, 0, "consumer"),
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            attempt_id="consumer-without-exact-ready-trees",
            producer_phase_completion=completion,
            producer_phase_completion_uri=str(completion_path),
        )
    ledger_path = campaign["tmp_path"] / "consumer-no-ready-trees-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    before = ledger_path.read_bytes()
    with pytest.raises(ValueError, match="ready-shard directory"):
        full_score.reserve_governed_full_score_phase_attempt(
            ledger_path,
            submit_payload,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="consumer",
            attempt_id="consumer-without-exact-ready-trees",
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization={},
        )
    assert ledger_path.read_bytes() == before


def test_mac_consumer_render_and_submit_validation_use_remote_tree_authority(
    campaign,
    monkeypatch,
):
    completion = _producer_completion(campaign, 0)
    consumers = _phase_payloads(campaign, 0, "consumer")
    authority = object()
    authority_calls = []

    def require_remote(value, **kwargs):
        assert value is authority
        assert kwargs["completion_record"] == completion
        assert kwargs["completion_uri"] == "dbfs:/remote/control/completion.json"
        authority_calls.append(kwargs)
        return value

    monkeypatch.setattr(
        full_score_remote,
        "require_full_score_remote_ready_authorization",
        require_remote,
    )
    monkeypatch.setattr(
        full_score,
        "_validate_governed_producer_ready_phase",
        lambda *_args, **_kwargs: pytest.fail(
            "Mac controller must not traverse the remote ready tree"
        ),
    )
    submit = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        consumers,
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id="wave-000-consumer-remote-ready",
        producer_phase_completion=completion,
        producer_phase_completion_uri="dbfs:/remote/control/completion.json",
        remote_ready_authorization=authority,
        compact_artifact_resolver=lambda _uri: pytest.fail(
            "renderer must use the collected non-record authority"
        ),
    )
    compact_root = campaign["tmp_path"] / "compact-cas"
    compact_root.mkdir()
    resolved = {}
    for task, payload in zip(submit["tasks"], consumers, strict=True):
        parameters = task["spark_python_task"]["parameters"]
        uri = parameters[parameters.index("--worker-payload-json") + 1]
        path = compact_root / f"worker-{payload['worker_index']:02d}.json"
        path.write_bytes(full_score._canonical_pretty_json_bytes(payload))
        resolved[uri] = path
    completion_path = compact_root / "completion.json"
    completion_path.write_bytes(full_score._canonical_pretty_json_bytes(completion))
    resolved["dbfs:/remote/control/completion.json"] = completion_path

    bindings = full_score._validated_full_score_phase_submit_payload(
        campaign["execution_plan"],
        submit,
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        wave_index=0,
        phase="consumer",
        require_governed_consumer_ready_phase=True,
        remote_ready_authorization=authority,
        compact_artifact_resolver=resolved.__getitem__,
    )

    assert len(bindings) == 16
    assert len(authority_calls) == 2


def test_full_score_workers_use_volume_mounts_for_volume_uris():
    uri = "dbfs:/Volumes/catalog/schema/volume/durable/worker.json"

    assert full_score._cluster_path(uri) == Path(
        "/Volumes/catalog/schema/volume/durable/worker.json"
    )
    assert 'if uri.startswith("dbfs:/Volumes/")' in full_score.FULL_SCORE_RUNNER_SCRIPT
    assert 'return "/Volumes/"' in full_score.FULL_SCORE_RUNNER_SCRIPT


def test_governed_terminal_billing_and_wave_zero_reservation_are_file_bound(
    campaign,
):
    attempt_id = "wave-000-producer-attempt-001"
    submit_payload = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "producer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=attempt_id,
    )
    submit_path = campaign["tmp_path"] / "wave-000-producer-submit.json"
    submit_path.write_bytes(full_score._canonical_pretty_json_bytes(submit_payload))
    tampered_submit = copy.deepcopy(submit_payload)
    tampered_submit["tasks"][0]["spark_python_task"]["parameters"].append(
        "--unreviewed-argument"
    )
    with pytest.raises(ValueError, match="unexpected trailing parameters"):
        full_score._validated_full_score_phase_submit_payload(
            campaign["execution_plan"],
            tampered_submit,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
        )
    ledger_path = campaign["tmp_path"] / "cluster-hour-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    reserved = full_score.reserve_governed_full_score_phase_attempt(
        ledger_path,
        submit_payload,
        execution_plan=campaign["execution_plan"],
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        wave_index=0,
        phase="producer",
        attempt_id=attempt_id,
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        predecessor_authorization=campaign[
            "latency_collection_authorization"
        ],
        latency_execution_plan_record=campaign[
            "latency_execution_plan_record"
        ],
    )
    assert reserved[0].active_reserved_cluster_hours == 96.0
    with pytest.raises(ValueError, match="already reserved|intent binding drift"):
        full_score.reserve_governed_full_score_phase_attempt(
            ledger_path,
            submit_payload,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=campaign[
                "latency_collection_authorization"
            ],
            latency_execution_plan_record=campaign[
                "latency_execution_plan_record"
            ],
        )


    run_id = 42001
    record_databricks_run_submission_receipt_json(
        ledger_path,
        attempt_id=attempt_id,
        submit_response={"run_id": run_id},
    )
    run_tasks = []
    for index, task in enumerate(submit_payload["tasks"]):
        run_tasks.append(
            {
                "cluster_instance": {"cluster_id": f"cluster-{index}"},
                "new_cluster": task["new_cluster"],
                "spark_python_task": copy.deepcopy(task["spark_python_task"]),
                "end_time": 2_500 + index,
                "run_id": 50_000 + index,
                "start_time": 1_000 + index,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "SUCCESS",
                    "state_message": "",
                },
                "task_key": task["task_key"],
            }
        )
    run_record = {
        "cluster_instance": {"cluster_id": "full-score-run-cluster"},
        "end_time": 3_000,
        "run_id": run_id,
        "run_name": submit_payload["run_name"],
        "run_page_url": "https://example.invalid/run/42001",
        "start_time": 900,
        "state": {
            "life_cycle_state": "TERMINATED",
            "result_state": "SUCCESS",
            "state_message": "",
        },
        "tasks": run_tasks,
    }
    record_databricks_verified_run_terminal_actual_json(
        ledger_path,
        attempt_id=attempt_id,
        run_record=run_record,
    )
    run_path = campaign["tmp_path"] / "wave-000-producer-runs-get.json"
    run_path.write_text(json.dumps(run_record), encoding="utf-8")
    substituted_python_task = copy.deepcopy(run_record)
    substituted_python_task["tasks"][0]["spark_python_task"]["python_file"] = (
        "dbfs:/attacker/substituted-full-score-runner.py"
    )
    run_path.write_text(json.dumps(substituted_python_task), encoding="utf-8")
    with pytest.raises(ValueError, match="spark_python_task binding drift"):
        full_score.build_governed_full_score_phase_terminal_record(
            campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            submit_payload_path=submit_path,
            control_plane_run_path=run_path,
            ledger_path=ledger_path,
            submission_authorization=reserved[1],
        )
    run_path.write_text(json.dumps(run_record), encoding="utf-8")
    duplicate_cluster_run = copy.deepcopy(run_record)
    duplicate_cluster_run["tasks"][1]["cluster_instance"]["cluster_id"] = (
        duplicate_cluster_run["tasks"][0]["cluster_instance"]["cluster_id"]
    )
    run_path.write_text(json.dumps(duplicate_cluster_run), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct billed clusters"):
        full_score.build_governed_full_score_phase_terminal_record(
            campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            submit_payload_path=submit_path,
            control_plane_run_path=run_path,
            ledger_path=ledger_path,
            submission_authorization=reserved[1],
        )
    run_path.write_text(json.dumps(run_record), encoding="utf-8")
    terminal_path = campaign["tmp_path"] / "wave-000-producer-terminal.json"
    terminal = full_score.write_governed_full_score_phase_terminal_record(
        terminal_path,
        campaign["execution_plan"],
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        wave_index=0,
        phase="producer",
        attempt_id=attempt_id,
        submit_payload_path=submit_path,
        control_plane_run_path=run_path,
        ledger_path=ledger_path,
        submission_authorization=reserved[1],
    )
    assert terminal["billed_gpu_seconds"] == 24.0
    assert len(terminal["task_billing"]) == 16
    assert len({item["cluster_id"] for item in terminal["task_billing"]}) == 16
    assert len({item["task_run_id"] for item in terminal["task_billing"]}) == 16
    assert {item["shard_id"] for item in terminal["task_billing"]} == set(
        campaign["execution_plan"]["waves"][0]["shard_ids"]
    )
    tampered_run = copy.deepcopy(run_record)
    tampered_run["state"]["state_message"] = "changed after reconciliation"
    run_path.write_text(json.dumps(tampered_run), encoding="utf-8")
    with pytest.raises(ValueError, match="reconciliation drift"):
        full_score.load_governed_full_score_phase_terminal_record(
            terminal_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            ledger_path=ledger_path,
        )
    run_path.write_text(json.dumps(run_record), encoding="utf-8")
    worker_uri = terminal["task_billing"][0]["worker_payload_uri"]
    worker_path = campaign["worker_files"][worker_uri]
    worker_path.write_bytes(worker_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="worker payload file SHA-256 drift"):
        full_score.load_governed_full_score_phase_terminal_record(
            terminal_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            ledger_path=ledger_path,
        )


def test_wave_zero_reservation_requires_campaign_ledger_and_headroom(campaign):
    attempt_id = "wave-zero-admission-test"
    submit_payload = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "producer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=attempt_id,
    )
    wrong_ledger_path = campaign["tmp_path"] / "wrong-campaign-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        wrong_ledger_path,
        ledger_id="different-publication-campaign",
    )
    with pytest.raises(ValueError, match="qualification ledger"):
        full_score.reserve_governed_full_score_phase_attempt(
            wrong_ledger_path,
            submit_payload,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=campaign[
                "latency_collection_authorization"
            ],
            latency_execution_plan_record=campaign[
                "latency_execution_plan_record"
            ],
        )

    reservations = tuple(
        DatabricksClusterHourReservation(
            attempt_id=f"historical-{index:03d}",
            workload_id="historical-publication-work",
            submit_payload_sha256=f"{index + 1:064x}",
            run_timeout_seconds=43_200,
            task_timeout_seconds=(43_200,),
        )
        for index in range(68)
    )
    terminal_actuals = tuple(
        DatabricksClusterHourTerminalActual(
            attempt_id=reservation.attempt_id,
            terminal_state="succeeded",
            actual_cluster_duration_seconds=43_200,
        )
        for reservation in reservations
    )
    headroom_ledger = DatabricksClusterHourLedger(
        ledger_id="publication-full-score",
        reservations=reservations,
        terminal_actuals=terminal_actuals,
    )
    headroom_path = campaign["tmp_path"] / "exhausted-headroom-ledger.json"
    headroom_path.write_bytes(
        full_score._canonical_pretty_json_bytes(
            databricks_cluster_hour_ledger_to_record(headroom_ledger)
        )
    )
    with pytest.raises(ValueError, match="124.*headroom"):
        full_score.reserve_governed_full_score_phase_attempt(
            headroom_path,
            submit_payload,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=campaign[
                "latency_collection_authorization"
            ],
            latency_execution_plan_record=campaign[
                "latency_execution_plan_record"
            ],
        )
    before_generic = headroom_path.read_bytes()
    with pytest.raises(ValueError, match="124.*headroom"):
        reserve_databricks_run_attempt_json(
            headroom_path,
            submit_payload,
            attempt_id=attempt_id,
            workload_id=full_score._full_score_phase_workload_id(
                campaign["execution_plan"],
                wave_index=0,
                phase="producer",
            ),
        )
    assert headroom_path.read_bytes() == before_generic


def test_phase_reservations_require_active_zero_and_exact_one_shot_success(campaign):
    first_attempt = "wave-000-producer-failed-001"
    submit_payload = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "producer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=first_attempt,
    )
    ledger_path = campaign["tmp_path"] / "one-shot-phase-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    campaign["latency_collection_authorization"].ledger_prefix = (
        databricks_ledger_prefix(
            read_databricks_cluster_hour_ledger_json(ledger_path)
        )
    )
    common = {
        "execution_plan": campaign["execution_plan"],
        "inventory": campaign["inventory"],
        "shard_plan": campaign["shard_plan"],
        "wave_index": 0,
        "phase": "producer",
        "qualification_launch_authorization": campaign[
            "qualification_launch_authorization"
        ],
        "predecessor_authorization": campaign[
            "latency_collection_authorization"
        ],
        "latency_execution_plan_record": campaign[
            "latency_execution_plan_record"
        ],
    }
    full_score.reserve_governed_full_score_phase_attempt(
        ledger_path,
        submit_payload,
        attempt_id=first_attempt,
        **common,
    )
    active_bytes = ledger_path.read_bytes()
    with pytest.raises(ValueError, match="idempotency token drift"):
        full_score.reserve_governed_full_score_phase_attempt(
            ledger_path,
            submit_payload,
            attempt_id="wave-000-producer-overlap",
            **common,
        )
    assert ledger_path.read_bytes() == active_bytes
    record_databricks_run_terminal_actual_json(
        ledger_path,
        attempt_id=first_attempt,
        terminal_state="failed",
        actual_cluster_duration_seconds=60.0,
    )
    successful_attempt = "wave-000-producer-success-002"
    retry_payload = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "producer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=successful_attempt,
    )
    closed_bytes = ledger_path.read_bytes()
    with pytest.raises(ValueError, match="complete current ledger prefix"):
        full_score.reserve_governed_full_score_phase_attempt(
            ledger_path,
            retry_payload,
            attempt_id=successful_attempt,
            **common,
        )
    assert ledger_path.read_bytes() == closed_bytes


def test_governed_submit_couples_reservation_exact_wire_and_receipt(campaign):
    attempt_id = "wave-000-producer-coupled-001"
    submit_payload = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "producer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=attempt_id,
    )
    ledger_path = campaign["tmp_path"] / "coupled-submit-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    opener = _FakeDatabricksOpener({"run_id": 74001})
    with pytest.raises(
        TypeError,
        match="PublicationLatencyCollectionAuthorization",
    ):
        full_score.reserve_and_submit_governed_full_score_phase_attempt(
            DatabricksWorkspaceConfig("https://dbc.example/", "secret-token"),
            submit_payload,
            ledger_path=ledger_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization={},
            latency_execution_plan_record=campaign[
                "latency_execution_plan_record"
            ],
            opener=opener,
        )
    with pytest.raises(
        TypeError,
        match="publication launch requires GPUQualificationLaunchAuthorization",
    ):
        full_score.reserve_and_submit_governed_full_score_phase_attempt(
            DatabricksWorkspaceConfig("https://dbc.example/", "secret-token"),
            submit_payload,
            ledger_path=ledger_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization={},
            predecessor_authorization=campaign[
                "latency_collection_authorization"
            ],
            latency_execution_plan_record=campaign[
                "latency_execution_plan_record"
            ],
            opener=opener,
        )
    assert not read_databricks_cluster_hour_ledger_json(ledger_path).reservations
    assert not opener.requests
    response = full_score.reserve_and_submit_governed_full_score_phase_attempt(
        DatabricksWorkspaceConfig("https://dbc.example/", "secret-token"),
        submit_payload,
        ledger_path=ledger_path,
        execution_plan=campaign["execution_plan"],
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        wave_index=0,
        phase="producer",
        attempt_id=attempt_id,
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        predecessor_authorization=campaign[
            "latency_collection_authorization"
        ],
        latency_execution_plan_record=campaign[
            "latency_execution_plan_record"
        ],
        opener=opener,
    )
    assert response[0] == {"run_id": 74001}
    assert len(opener.requests) == 1
    assert json.loads(opener.requests[0].data.decode("utf-8")) == submit_payload
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert [item.attempt_id for item in ledger.reservations] == [attempt_id]
    assert [item.attempt_id for item in ledger.submission_receipts] == [attempt_id]
    assert ledger.submission_receipts[0].run_id == "74001"
    assert (
        ledger.submission_receipts[0].submit_payload_sha256
        == ledger.reservations[0].submit_payload_sha256
    )
    with pytest.raises(ValueError, match="already reserved|intent binding drift"):
        full_score.reserve_and_submit_governed_full_score_phase_attempt(
            DatabricksWorkspaceConfig("https://dbc.example/", "secret-token"),
            submit_payload,
            ledger_path=ledger_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=campaign[
                "latency_collection_authorization"
            ],
            latency_execution_plan_record=campaign[
                "latency_execution_plan_record"
            ],
            opener=opener,
        )
    assert len(opener.requests) == 1


def test_full_score_live_identity_fails_before_reservation_state(
    campaign,
    monkeypatch,
):
    attempt_id = "wave-000-producer-wrong-principal"
    submit_payload = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "producer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=attempt_id,
    )
    observed = []

    def reject_current_user(_workspace, *, expected_user_name, opener=None):
        observed.append((expected_user_name, opener))
        raise ValueError("Databricks current-user identity differs")

    monkeypatch.setattr(
        full_score,
        "require_databricks_current_user_name",
        reject_current_user,
    )
    ledger_path = campaign["tmp_path"] / "wrong-principal-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    ledger_before = ledger_path.read_bytes()

    with pytest.raises(ValueError, match="current-user identity differs"):
        full_score.reserve_and_submit_governed_full_score_phase_attempt(
            DatabricksWorkspaceConfig("https://dbc.example/", "secret-token"),
            submit_payload,
            ledger_path=ledger_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=campaign[
                "latency_collection_authorization"
            ],
            latency_execution_plan_record=campaign[
                "latency_execution_plan_record"
            ],
        )

    assert ledger_path.read_bytes() == ledger_before
    assert not ledger_path.with_name(
        f"{ledger_path.name}.full-score-phase-intents"
    ).exists()
    assert not ledger_path.with_name(
        f"{ledger_path.name}.full-score-phase-leases"
    ).exists()

    assert observed == [("researcher@example.com", None)]


def test_direct_collection_issues_ordered_path_bound_phase_authority(campaign):
    collected = _collect_wave_zero_producer(campaign, label="phase-chain")
    authorization = collected["phase_authorization"]
    ledger_path = collected["ledger_path"]
    terminal = collected["terminal"]
    assert "path" not in terminal["ledger"]
    assert terminal["ledger"]["ledger_path_sha256"] == (
        databricks_ledger_path_sha256(ledger_path)
    )
    prefix, _lineage = (
        full_score._require_full_score_phase_predecessor_authorization(
            authorization,
            execution_plan=campaign["execution_plan"],
            ledger_path=ledger_path,
            wave_index=0,
            phase="consumer",
        )
    )
    assert prefix == authorization.ledger_prefix
    with pytest.raises(TypeError, match="FullScorePhaseAuthorization"):
        full_score._require_full_score_phase_predecessor_authorization(
            dict(terminal),
            execution_plan=campaign["execution_plan"],
            ledger_path=ledger_path,
            wave_index=0,
            phase="consumer",
        )
    with pytest.raises(ValueError, match="ordering/path binding drift"):
        full_score._require_full_score_phase_predecessor_authorization(
            authorization,
            execution_plan=campaign["execution_plan"],
            ledger_path=ledger_path,
            wave_index=1,
            phase="consumer",
        )
    copied_path = campaign["tmp_path"] / "copied-phase-ledger.json"
    copied_path.write_bytes(ledger_path.read_bytes())
    with pytest.raises(ValueError, match="ordering/path binding drift"):
        full_score._require_full_score_phase_predecessor_authorization(
            authorization,
            execution_plan=campaign["execution_plan"],
            ledger_path=copied_path,
            wave_index=0,
            phase="consumer",
        )
    fresh_same_id = DatabricksClusterHourLedger(
        ledger_id="publication-full-score"
    )
    ledger_path.write_bytes(
        full_score._canonical_pretty_json_bytes(
            databricks_cluster_hour_ledger_to_record(fresh_same_id)
        )
    )
    with pytest.raises(ValueError, match="shorter than its authorized prefix"):
        full_score._require_full_score_phase_predecessor_authorization(
            authorization,
            execution_plan=campaign["execution_plan"],
            ledger_path=ledger_path,
            wave_index=0,
            phase="consumer",
        )
    with pytest.raises(ValueError, match="ledger path must be an existing"):
        full_score._require_full_score_phase_predecessor_authorization(
            authorization,
            execution_plan=campaign["execution_plan"],
            ledger_path="dbfs:/mutable/live-ledger.json",
            wave_index=0,
            phase="consumer",
        )


def test_terminal_replay_rejects_unrelated_batch_and_extended_prefix(campaign):
    collected = _collect_wave_zero_producer(campaign, label="forged-terminal")
    ledger_path = collected["ledger_path"]
    terminal_record = collected["terminal"]
    original = read_databricks_cluster_hour_ledger_json(ledger_path)
    unrelated_payload_sha256 = _digest("unrelated-submit-payload")
    unrelated_reservation = DatabricksClusterHourReservation(
        attempt_id="unrelated-attempt",
        workload_id="unrelated-workload",
        submit_payload_sha256=unrelated_payload_sha256,
        run_timeout_seconds=3_600,
        task_timeout_seconds=(3_600,),
    )
    unrelated_receipt = DatabricksRunSubmissionReceipt(
        attempt_id="unrelated-attempt",
        run_id="99001",
        submit_payload_sha256=unrelated_payload_sha256,
        submit_response_sha256=_digest("unrelated-submit-response"),
    )
    unrelated_terminal = DatabricksClusterHourTerminalActual(
        attempt_id="unrelated-attempt",
        terminal_state="succeeded",
        actual_cluster_duration_seconds=1.0,
        verification_source="direct_databricks_runs_get",
        run_id="99001",
        submit_payload_sha256=unrelated_payload_sha256,
        control_plane_status_sha256=_digest("unrelated-control-plane"),
    )

    extended = DatabricksClusterHourLedger(
        ledger_id=original.ledger_id,
        cap_cluster_hours=original.cap_cluster_hours,
        reservations=original.reservations + (unrelated_reservation,),
        submission_receipts=original.submission_receipts
        + (unrelated_receipt,),
        terminal_actuals=original.terminal_actuals + (unrelated_terminal,),
    )
    ledger_path.write_bytes(
        full_score._canonical_pretty_json_bytes(
            databricks_cluster_hour_ledger_to_record(extended)
        )
    )
    # A genuine terminal record remains replayable from its historical slice.
    assert full_score.load_governed_full_score_phase_terminal_record(
        collected["terminal_path"],
        execution_plan=campaign["execution_plan"],
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        ledger_path=ledger_path,
    ) == terminal_record
    extended_prefix_forgery = copy.deepcopy(terminal_record)
    extended_prefix_forgery["ledger"]["terminal_prefix"] = (
        databricks_ledger_prefix(extended).to_record()
    )
    _close(extended_prefix_forgery)
    extended_path = campaign["tmp_path"] / "extended-terminal-forgery.json"
    extended_path.write_bytes(
        full_score._canonical_pretty_json_bytes(extended_prefix_forgery)
    )
    with pytest.raises(ValueError, match="terminal-prefix transition"):
        full_score.load_governed_full_score_phase_terminal_record(
            extended_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            ledger_path=ledger_path,
        )

    empty = DatabricksClusterHourLedger(
        ledger_id=original.ledger_id,
        cap_cluster_hours=original.cap_cluster_hours,
    )
    unrelated_batch = DatabricksClusterHourLedger(
        ledger_id=original.ledger_id,
        cap_cluster_hours=original.cap_cluster_hours,
        reservations=(unrelated_reservation,),
    )
    target_after_unrelated = DatabricksClusterHourLedger(
        ledger_id=original.ledger_id,
        cap_cluster_hours=original.cap_cluster_hours,
        reservations=(unrelated_reservation,) + original.reservations,
        submission_receipts=(unrelated_receipt,) + original.submission_receipts,
        terminal_actuals=(unrelated_terminal,) + original.terminal_actuals,
    )
    ledger_path.write_bytes(
        full_score._canonical_pretty_json_bytes(
            databricks_cluster_hour_ledger_to_record(target_after_unrelated)
        )
    )
    unrelated_batch_forgery = copy.deepcopy(terminal_record)
    unrelated_batch_forgery["ledger"].update(
        {
            "predecessor_prefix": databricks_ledger_prefix(empty).to_record(),
            "batch_prefix": databricks_ledger_prefix(
                unrelated_batch
            ).to_record(),
            "terminal_prefix": databricks_ledger_prefix(
                target_after_unrelated
            ).to_record(),
        }
    )
    _close(unrelated_batch_forgery)
    unrelated_path = campaign["tmp_path"] / "unrelated-batch-forgery.json"
    unrelated_path.write_bytes(
        full_score._canonical_pretty_json_bytes(unrelated_batch_forgery)
    )
    with pytest.raises(ValueError, match="terminal-prefix transition"):
        full_score.load_governed_full_score_phase_terminal_record(
            unrelated_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            ledger_path=ledger_path,
        )


def test_terminal_phase_authority_reissues_after_controller_restart(campaign):
    collected = _collect_wave_zero_producer(campaign, label="terminal-restart")
    replayed = full_score.replay_governed_full_score_phase_authorization(
        collected["terminal_path"],
        execution_plan=campaign["execution_plan"],
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        ledger_path=collected["ledger_path"],
    )
    assert replayed == collected["phase_authorization"]
    prefix, lineage = (
        full_score._require_full_score_phase_predecessor_authorization(
            replayed,
            execution_plan=campaign["execution_plan"],
            ledger_path=collected["ledger_path"],
            wave_index=0,
            phase="consumer",
        )
    )
    assert prefix == replayed.ledger_prefix
    assert lineage["authorization_sha256"] == replayed.causal_closure_sha256
    intent_path = full_score._full_score_phase_intent_path(
        collected["ledger_path"],
        execution_plan_sha256=campaign["execution_plan"][
            "closed_record_sha256"
        ],
        wave_index=0,
        phase="producer",
    )
    intent_path.unlink()
    with pytest.raises(ValueError, match="durable intent"):
        full_score.replay_governed_full_score_phase_authorization(
            collected["terminal_path"],
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            ledger_path=collected["ledger_path"],
        )


def test_publication_aggregate_requires_current_final_consumer_authority(campaign):
    collected = _collect_wave_zero_producer(
        campaign,
        label="aggregate-final-consumer",
    )
    producer_authorization = collected["phase_authorization"]
    ledger_path = collected["ledger_path"]
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    batch_prefix = full_score.databricks_ledger_prefix_at_counts(
        ledger,
        reservation_count=(
            producer_authorization.predecessor_prefix.reservation_count + 1
        ),
        submission_receipt_count=(
            producer_authorization.predecessor_prefix.submission_receipt_count
        ),
        terminal_actual_count=(
            producer_authorization.predecessor_prefix.terminal_actual_count
        ),
    )
    causal_closure = full_score._canonical_sha256(
        {
            "batch_prefix": batch_prefix.to_record(),
            "ledger_path_sha256": databricks_ledger_path_sha256(ledger_path),
            "terminal_prefix": producer_authorization.ledger_prefix.to_record(),
            "terminal_record_sha256": (
                producer_authorization.terminal_record_sha256
            ),
        }
    )
    final_authorization = full_score.FullScorePhaseAuthorization(
        execution_plan_sha256=campaign["execution_plan"][
            "closed_record_sha256"
        ],
        wave_index=len(campaign["execution_plan"]["waves"]) - 1,
        phase="consumer",
        ledger_path_sha256=databricks_ledger_path_sha256(ledger_path),
        predecessor_prefix=producer_authorization.predecessor_prefix,
        ledger_prefix=producer_authorization.ledger_prefix,
        phase_lease_root=(
            ledger_path.with_name(f"{ledger_path.name}.full-score-phase-leases")
        ),
        terminal_record_sha256=(
            producer_authorization.terminal_record_sha256
        ),
        causal_closure_sha256=causal_closure,
        _issuer=full_score._FULL_SCORE_PHASE_AUTHORIZATION_ISSUER,
    )
    lineage = (
        full_score._require_full_score_final_consumer_aggregation_authorization(
            campaign["execution_plan"],
            final_authorization,
            ledger_path=ledger_path,
        )
    )
    assert lineage["predecessor_prefix"] == (
        final_authorization.predecessor_prefix.to_record()
    )
    assert lineage["batch_prefix"] == batch_prefix.to_record()
    assert lineage["terminal_prefix"] == (
        final_authorization.ledger_prefix.to_record()
    )
    with pytest.raises(TypeError, match="FullScorePhaseAuthorization"):
        full_score._require_full_score_final_consumer_aggregation_authorization(
            campaign["execution_plan"],
            dict(collected["terminal"]),
            ledger_path=ledger_path,
        )
    with pytest.raises(ValueError, match="final-wave consumer"):
        full_score._require_full_score_final_consumer_aggregation_authorization(
            campaign["execution_plan"],
            producer_authorization,
            ledger_path=ledger_path,
        )
    copied = campaign["tmp_path"] / "copied-aggregate-ledger.json"
    copied.write_bytes(ledger_path.read_bytes())
    with pytest.raises(ValueError, match="ledger path binding drift"):
        full_score._require_full_score_final_consumer_aggregation_authorization(
            campaign["execution_plan"],
            final_authorization,
            ledger_path=copied,
        )
    reserve_databricks_run_attempt_json(
        ledger_path,
        {
            "run_name": "post-final-extra-event",
            "tasks": [
                {
                    "max_retries": 0,
                    "new_cluster": {},
                    "task_key": "post_final_extra_event",
                    "timeout_seconds": 3_600,
                }
            ],
            "timeout_seconds": 3_600,
        },
        attempt_id="post-final-extra-event",
        workload_id="post-final-extra-event",
    )
    with pytest.raises(ValueError, match="complete current ledger prefix"):
        full_score._require_full_score_final_consumer_aggregation_authorization(
            campaign["execution_plan"],
            final_authorization,
            ledger_path=ledger_path,
        )


def test_terminal_collection_resumes_exact_crash_boundaries_and_races(
    campaign,
    monkeypatch,
):
    workspace = DatabricksWorkspaceConfig(
        "https://dbc.example/",
        "secret-token",
    )

    def collect(reserved):
        return full_score.collect_governed_full_score_phase_attempt(
            workspace,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            ledger_path=reserved["ledger_path"],
            submission_authorization=reserved["submission_authorization"],
            submit_payload_path=reserved["submit_path"],
            control_plane_run_path=reserved["run_path"],
            terminal_record_path=reserved["terminal_path"],
            opener=reserved["opener"],
        )

    wrong_identity = _reserve_wave_zero_producer(
        campaign,
        label="collector-wrong-principal",
    )
    wrong_identity_ledger = wrong_identity["ledger_path"].read_bytes()
    wrong_identity_request_count = len(wrong_identity["opener"].requests)
    observed_principals = []

    def reject_current_user(_workspace, *, expected_user_name, opener=None):
        observed_principals.append((expected_user_name, opener))
        raise ValueError("Databricks current-user identity differs")

    with monkeypatch.context() as identity_patch:
        identity_patch.setattr(
            full_score,
            "require_databricks_current_user_name",
            reject_current_user,
        )
        with pytest.raises(ValueError, match="current-user identity differs"):
            collect(wrong_identity)
    assert observed_principals == [
        ("researcher@example.com", wrong_identity["opener"])
    ]
    assert len(wrong_identity["opener"].requests) == wrong_identity_request_count
    assert wrong_identity["ledger_path"].read_bytes() == wrong_identity_ledger
    assert not wrong_identity["run_path"].exists()
    assert not wrong_identity["terminal_path"].exists()

    after_run_file = _reserve_wave_zero_producer(
        campaign,
        label="collector-after-run-file",
    )
    run_record = _terminal_run_record(
        after_run_file["submit_payload"],
        run_id=81_001,
    )
    after_run_file["run_path"].write_bytes(
        full_score._canonical_pretty_json_bytes(run_record)
    )
    first_record, first_authorization = collect(after_run_file)
    replayed_record, replayed_authorization = collect(after_run_file)
    assert replayed_record == first_record
    assert replayed_authorization == first_authorization

    after_terminal = _reserve_wave_zero_producer(
        campaign,
        label="collector-after-terminal",
    )
    run_record = _terminal_run_record(
        after_terminal["submit_payload"],
        run_id=81_001,
    )
    after_terminal["run_path"].write_bytes(
        full_score._canonical_pretty_json_bytes(run_record)
    )
    record_databricks_verified_run_terminal_actual_json(
        after_terminal["ledger_path"],
        attempt_id=after_terminal["submission_authorization"].attempt_id,
        run_record=run_record,
    )
    terminal_record, terminal_authorization = collect(after_terminal)
    assert terminal_record["ledger"]["terminal_prefix"] == (
        terminal_authorization.ledger_prefix.to_record()
    )

    concurrent = _reserve_wave_zero_producer(
        campaign,
        label="collector-concurrent",
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(collect, concurrent) for _index in range(2)]
    results = [future.result() for future in futures]
    assert results[0] == results[1]
    ledger = read_databricks_cluster_hour_ledger_json(concurrent["ledger_path"])
    assert len(ledger.terminal_actuals) == 1


def test_concurrent_duplicate_phase_launch_opens_exactly_once(campaign):
    attempt_id = "wave-000-producer-concurrent"
    submit_payload = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "producer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=attempt_id,
    )
    ledger_path = campaign["tmp_path"] / "concurrent-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    campaign["latency_collection_authorization"].ledger_prefix = (
        databricks_ledger_prefix(
            read_databricks_cluster_hour_ledger_json(ledger_path)
        )
    )
    opener = _FakeDatabricksOpener({"run_id": 91_001})

    def launch():
        return full_score.reserve_and_submit_governed_full_score_phase_attempt(
            DatabricksWorkspaceConfig(
                "https://dbc.example/", "secret-token"
            ),
            submit_payload,
            ledger_path=ledger_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=campaign[
                "latency_collection_authorization"
            ],
            latency_execution_plan_record=campaign[
                "latency_execution_plan_record"
            ],
            opener=opener,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(launch) for _index in range(2)]
    successes = [future.result() for future in futures if future.exception() is None]
    failures = [future.exception() for future in futures if future.exception()]
    assert len(successes) == 1
    assert len(failures) == 1
    assert len(opener.requests) == 1
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert len(ledger.reservations) == 1
    assert len(ledger.submission_receipts) == 1


def test_ambiguous_phase_post_recovery_is_exact_and_one_shot(
    campaign,
    monkeypatch,
):
    attempt_id = "wave-000-producer-recovery"
    submit_payload = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "producer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=attempt_id,
    )
    ledger_path = campaign["tmp_path"] / "recovery-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    campaign["latency_collection_authorization"].ledger_prefix = (
        databricks_ledger_prefix(
            read_databricks_cluster_hour_ledger_json(ledger_path)
        )
    )
    _ledger, submission_authorization = (
        full_score.reserve_governed_full_score_phase_attempt(
            ledger_path,
            submit_payload,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=campaign[
                "latency_collection_authorization"
            ],
            latency_execution_plan_record=campaign[
                "latency_execution_plan_record"
            ],
        )
    )
    rejected_opener = _FakeDatabricksOpener({"run_id": 91_999})
    observed_principals = []

    def reject_current_user(_workspace, *, expected_user_name, opener=None):
        observed_principals.append((expected_user_name, opener))
        raise ValueError("Databricks current-user identity differs")

    with monkeypatch.context() as identity_patch:
        identity_patch.setattr(
            full_score,
            "require_databricks_current_user_name",
            reject_current_user,
        )
        with pytest.raises(ValueError, match="current-user identity differs"):
            full_score.submit_governed_full_score_phase_attempt(
                DatabricksWorkspaceConfig(
                    "https://dbc.example/", "secret-token"
                ),
                submit_payload,
                ledger_path=ledger_path,
                submission_authorization=submission_authorization,
                opener=rejected_opener,
            )
        with pytest.raises(ValueError, match="current-user identity differs"):
            full_score.recover_governed_full_score_phase_attempt(
                DatabricksWorkspaceConfig(
                    "https://dbc.example/", "secret-token"
                ),
                submit_payload,
                ledger_path=ledger_path,
                submission_authorization=submission_authorization,
                opener=rejected_opener,
            )
    assert observed_principals == [
        ("researcher@example.com", rejected_opener),
        ("researcher@example.com", rejected_opener),
    ]
    assert rejected_opener.requests == []
    assert not ledger_path.with_name(f"{ledger_path.name}.post-claims").exists()

    lost = _AcceptedButResponseLostOpener()
    with pytest.raises(TimeoutError, match="response was lost"):
        full_score.submit_governed_full_score_phase_attempt(
            DatabricksWorkspaceConfig(
                "https://dbc.example/", "secret-token"
            ),
            submit_payload,
            ledger_path=ledger_path,
            submission_authorization=submission_authorization,
            opener=lost,
        )
    assert len(lost.requests) == 1
    recovery = _FakeDatabricksOpener({"run_id": 92_001})
    assert full_score.recover_governed_full_score_phase_attempt(
        DatabricksWorkspaceConfig("https://dbc.example/", "secret-token"),
        submit_payload,
        ledger_path=ledger_path,
        submission_authorization=submission_authorization,
        opener=recovery,
    ) == {"run_id": "92001"}
    assert len(recovery.requests) == 1
    no_second_post = _AcceptedButResponseLostOpener()
    assert full_score.recover_governed_full_score_phase_attempt(
        DatabricksWorkspaceConfig("https://dbc.example/", "secret-token"),
        submit_payload,
        ledger_path=ledger_path,
        submission_authorization=submission_authorization,
        opener=no_second_post,
    ) == {"run_id": "92001"}
    assert not no_second_post.requests
    missing_token = dict(submit_payload)
    missing_token.pop("idempotency_token")
    with pytest.raises(ValueError, match="payload binding drift"):
        full_score.recover_governed_full_score_phase_attempt(
            DatabricksWorkspaceConfig(
                "https://dbc.example/", "secret-token"
            ),
            missing_token,
            ledger_path=ledger_path,
            submission_authorization=submission_authorization,
        )
    changed_payload = copy.deepcopy(submit_payload)
    changed_payload["tasks"][0]["timeout_seconds"] -= 1
    with pytest.raises(ValueError, match="payload binding drift"):
        full_score.recover_governed_full_score_phase_attempt(
            DatabricksWorkspaceConfig(
                "https://dbc.example/", "secret-token"
            ),
            changed_payload,
            ledger_path=ledger_path,
            submission_authorization=submission_authorization,
        )
    intent_path = full_score._full_score_phase_intent_path(
        ledger_path,
        execution_plan_sha256=campaign["execution_plan"][
            "closed_record_sha256"
        ],
        wave_index=0,
        phase="producer",
    )
    intent_path.unlink()
    with pytest.raises(ValueError, match="durable intent"):
        full_score.recover_governed_full_score_phase_attempt(
            DatabricksWorkspaceConfig(
                "https://dbc.example/",
                "secret-token",
            ),
            submit_payload,
            ledger_path=ledger_path,
            submission_authorization=submission_authorization,
        )


def test_reserved_phase_resumes_after_controller_restart(
    campaign,
    monkeypatch,
):
    monkeypatch.setattr(
        full_score,
        "PublicationLatencyCollectionAuthorization",
        SimpleNamespace,
    )
    monkeypatch.setattr(
        full_score,
        "validate_publication_latency_collection_record",
        lambda *_args, **_kwargs: None,
    )
    campaign["latency_collection_authorization"].collection = {}
    workspace = DatabricksWorkspaceConfig(
        "https://dbc.example/",
        "secret-token",
    )

    for suffix, lose_first_response in (
        ("unclaimed", False),
        ("claimed", True),
    ):
        attempt_id = f"wave-000-producer-restart-{suffix}"
        submit_payload = full_score.build_databricks_full_score_run_submit_payload(
            campaign["job"],
            _phase_payloads(campaign, 0, "producer"),
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            attempt_id=attempt_id,
        )
        ledger_path = campaign["tmp_path"] / f"restart-{suffix}-ledger.json"
        create_databricks_cluster_hour_ledger_json(
            ledger_path,
            ledger_id="publication-full-score",
        )
        campaign["latency_collection_authorization"].ledger_prefix = (
            databricks_ledger_prefix(
                read_databricks_cluster_hour_ledger_json(ledger_path)
            )
        )
        campaign["latency_collection_authorization"].ledger_path_sha256 = (
            databricks_ledger_path_sha256(ledger_path)
        )
        _ledger, original_authorization = (
            full_score.reserve_governed_full_score_phase_attempt(
                ledger_path,
                submit_payload,
                execution_plan=campaign["execution_plan"],
                inventory=campaign["inventory"],
                shard_plan=campaign["shard_plan"],
                wave_index=0,
                phase="producer",
                attempt_id=attempt_id,
                qualification_launch_authorization=campaign[
                    "qualification_launch_authorization"
                ],
                predecessor_authorization=campaign[
                    "latency_collection_authorization"
                ],
                latency_execution_plan_record=campaign[
                    "latency_execution_plan_record"
                ],
            )
        )
        if lose_first_response:
            lost = _AcceptedButResponseLostOpener()
            with pytest.raises(TimeoutError, match="response was lost"):
                full_score.submit_governed_full_score_phase_attempt(
                    workspace,
                    submit_payload,
                    ledger_path=ledger_path,
                    submission_authorization=original_authorization,
                    opener=lost,
                )
            assert len(lost.requests) == 1
        if not lose_first_response:
            lease_path = full_score._full_score_phase_lease_path(
                ledger_path,
                execution_plan_sha256=campaign["execution_plan"][
                    "closed_record_sha256"
                ],
                wave_index=0,
                phase="producer",
            )
            lease_path.unlink()
            lease_root = lease_path.parent
            lease_root.rmdir()
            with pytest.raises(ValueError, match="post-batch lease"):
                full_score.replay_governed_full_score_phase_submission_authorization(
                    ledger_path,
                    submit_payload,
                    execution_plan=campaign["execution_plan"],
                    inventory=campaign["inventory"],
                    shard_plan=campaign["shard_plan"],
                    wave_index=0,
                    phase="producer",
                    attempt_id=attempt_id,
                    qualification_launch_authorization=campaign[
                        "qualification_launch_authorization"
                    ],
                    predecessor_authorization=campaign[
                        "latency_collection_authorization"
                    ],
                    latency_execution_plan_record=campaign[
                        "latency_execution_plan_record"
                    ],
                )
            assert not lease_root.exists()
            ledger_before_wrong_identity = ledger_path.read_bytes()
            rejected_opener = _FakeDatabricksOpener({"run_id": 92_999})
            observed_principals = []

            def reject_current_user(
                _workspace,
                *,
                expected_user_name,
                opener=None,
            ):
                observed_principals.append((expected_user_name, opener))
                raise ValueError("Databricks current-user identity differs")

            with monkeypatch.context() as identity_patch:
                identity_patch.setattr(
                    full_score,
                    "require_databricks_current_user_name",
                    reject_current_user,
                )
                with pytest.raises(
                    ValueError,
                    match="current-user identity differs",
                ):
                    full_score.resume_governed_full_score_phase_attempt(
                        workspace,
                        submit_payload,
                        ledger_path=ledger_path,
                        execution_plan=campaign["execution_plan"],
                        inventory=campaign["inventory"],
                        shard_plan=campaign["shard_plan"],
                        wave_index=0,
                        phase="producer",
                        attempt_id=attempt_id,
                        qualification_launch_authorization=campaign[
                            "qualification_launch_authorization"
                        ],
                        predecessor_authorization=campaign[
                            "latency_collection_authorization"
                        ],
                        latency_execution_plan_record=campaign[
                            "latency_execution_plan_record"
                        ],
                        opener=rejected_opener,
                    )
            assert observed_principals == [
                ("researcher@example.com", rejected_opener)
            ]
            assert rejected_opener.requests == []
            assert ledger_path.read_bytes() == ledger_before_wrong_identity
            assert not lease_root.exists()
            assert not ledger_path.with_name(
                f"{ledger_path.name}.post-claims"
            ).exists()

            intent_path = full_score._full_score_phase_intent_path(
                ledger_path,
                execution_plan_sha256=campaign["execution_plan"][
                    "closed_record_sha256"
                ],
                wave_index=0,
                phase="producer",
            )
            intent_bytes = intent_path.read_bytes()
            race_opener = _FakeDatabricksOpener({"run_id": 92_998})

            def authenticate_after_deleting_intent(
                _workspace,
                *,
                expected_user_name,
                opener=None,
            ):
                assert expected_user_name == "researcher@example.com"
                assert opener is race_opener
                intent_path.unlink()
                return {
                    "authenticated": True,
                    "user_name_sha256": _digest(expected_user_name),
                }

            with monkeypatch.context() as identity_patch:
                identity_patch.setattr(
                    full_score,
                    "require_databricks_current_user_name",
                    authenticate_after_deleting_intent,
                )
                with pytest.raises(
                    ValueError,
                    match="pre-reservation intent",
                ):
                    full_score.resume_governed_full_score_phase_attempt(
                        workspace,
                        submit_payload,
                        ledger_path=ledger_path,
                        execution_plan=campaign["execution_plan"],
                        inventory=campaign["inventory"],
                        shard_plan=campaign["shard_plan"],
                        wave_index=0,
                        phase="producer",
                        attempt_id=attempt_id,
                        qualification_launch_authorization=campaign[
                            "qualification_launch_authorization"
                        ],
                        predecessor_authorization=campaign[
                            "latency_collection_authorization"
                        ],
                        latency_execution_plan_record=campaign[
                            "latency_execution_plan_record"
                        ],
                        opener=race_opener,
                    )
            assert race_opener.requests == []
            assert ledger_path.read_bytes() == ledger_before_wrong_identity
            assert not lease_root.exists()
            assert not ledger_path.with_name(
                f"{ledger_path.name}.post-claims"
            ).exists()
            intent_path.write_bytes(intent_bytes)

            replayed = full_score.recover_governed_full_score_phase_reservation(
                ledger_path,
                submit_payload,
                execution_plan=campaign["execution_plan"],
                inventory=campaign["inventory"],
                shard_plan=campaign["shard_plan"],
                wave_index=0,
                phase="producer",
                attempt_id=attempt_id,
                qualification_launch_authorization=campaign[
                    "qualification_launch_authorization"
                ],
                predecessor_authorization=campaign[
                    "latency_collection_authorization"
                ],
                latency_execution_plan_record=campaign[
                    "latency_execution_plan_record"
                ],
            )
        else:
            replayed = (
                full_score.replay_governed_full_score_phase_submission_authorization(
                    ledger_path,
                    submit_payload,
                    execution_plan=campaign["execution_plan"],
                    inventory=campaign["inventory"],
                    shard_plan=campaign["shard_plan"],
                    wave_index=0,
                    phase="producer",
                    attempt_id=attempt_id,
                    qualification_launch_authorization=campaign[
                        "qualification_launch_authorization"
                    ],
                    predecessor_authorization=campaign[
                        "latency_collection_authorization"
                    ],
                    latency_execution_plan_record=campaign[
                        "latency_execution_plan_record"
                    ],
                )
            )
        assert replayed == original_authorization
        recovery = _FakeDatabricksOpener(
            {"run_id": 93_001 if not lose_first_response else 93_002}
        )
        response, resumed_authorization = (
            full_score.resume_governed_full_score_phase_attempt(
                workspace,
                submit_payload,
                ledger_path=ledger_path,
                execution_plan=campaign["execution_plan"],
                inventory=campaign["inventory"],
                shard_plan=campaign["shard_plan"],
                wave_index=0,
                phase="producer",
                attempt_id=attempt_id,
                qualification_launch_authorization=campaign[
                    "qualification_launch_authorization"
                ],
                predecessor_authorization=campaign[
                    "latency_collection_authorization"
                ],
                latency_execution_plan_record=campaign[
                    "latency_execution_plan_record"
                ],
                opener=recovery,
            )
        )
        assert response == {
            "run_id": "93001" if not lose_first_response else "93002"
        }
        assert resumed_authorization == original_authorization
        assert len(recovery.requests) == 1
        no_second_post = _AcceptedButResponseLostOpener()
        repeated, repeated_authorization = (
            full_score.resume_governed_full_score_phase_attempt(
                workspace,
                submit_payload,
                ledger_path=ledger_path,
                execution_plan=campaign["execution_plan"],
                inventory=campaign["inventory"],
                shard_plan=campaign["shard_plan"],
                wave_index=0,
                phase="producer",
                attempt_id=attempt_id,
                qualification_launch_authorization=campaign[
                    "qualification_launch_authorization"
                ],
                predecessor_authorization=campaign[
                    "latency_collection_authorization"
                ],
                latency_execution_plan_record=campaign[
                    "latency_execution_plan_record"
                ],
                opener=no_second_post,
            )
        )
        assert repeated == response
        assert repeated_authorization == original_authorization
        assert not no_second_post.requests


def test_governed_reservation_rejects_projected_max16_before_write(campaign):
    attempt_id = "active-cap-rejected-full-score"
    submit_payload = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "producer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=attempt_id,
    )
    ledger_path = campaign["tmp_path"] / "active-cap-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    seed_payload = {
        "run_name": "active-task-seed",
        "tasks": [
            {
                "max_retries": 0,
                "new_cluster": {},
                "task_key": "seed_task_00",
                "timeout_seconds": 3_600,
            }
        ],
        "timeout_seconds": 3_600,
    }
    reserve_databricks_run_attempt_json(
        ledger_path,
        seed_payload,
        attempt_id="active-task-seed",
        workload_id="active-task-seed",
    )
    before = ledger_path.read_bytes()
    assert (
        read_databricks_cluster_hour_ledger_json(
            ledger_path
        ).active_reserved_task_count
        == 1
    )
    with pytest.raises(ValueError, match="active task concurrency guard"):
        full_score.reserve_governed_full_score_phase_attempt(
            ledger_path,
            submit_payload,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=campaign[
                "latency_collection_authorization"
            ],
            latency_execution_plan_record=campaign[
                "latency_execution_plan_record"
            ],
        )
    assert ledger_path.read_bytes() == before


def test_live_p90_gate_replays_matched_blocks_and_authorizes_next_phase(campaign):
    wave_one_producers = _phase_payloads(campaign, 1, "producer")
    reservation = full_score.full_score_wave_worst_case_gpu_hours(
        wave_one_producers
    )
    assert reservation == 24.0
    blocks = _matched_blocks(campaign, 0)
    gate = full_score.build_full_score_live_p90_budget_admission(
        campaign["execution_plan"],
        blocks,
        next_wave_index=1,
        ledger_terminal_actual_gpu_hours=1.0,
        ledger_active_reserved_gpu_hours=0.0,
        next_wave_reserved_gpu_hours=reservation,
    )
    assert gate["admitted"] is True
    assert set(gate["formula"]) == {
        "admission",
        "consumer_task_scale",
        "matched_resampling",
        "producer_scale",
    }
    assert all(
        set(block["billed_gpu_seconds"]) == {"consumer_task", "producer"}
        for block in gate["completed_blocks"]
    )
    assert gate["bootstrap"]["draws"] == 10_000
    assert len(gate["bootstrap"]["resample_indices"]) == 10_000
    assert gate == full_score.build_full_score_live_p90_budget_admission(
        campaign["execution_plan"],
        blocks,
        next_wave_index=1,
        ledger_terminal_actual_gpu_hours=1.0,
        ledger_active_reserved_gpu_hours=0.0,
        next_wave_reserved_gpu_hours=reservation,
    )
    full_score.validate_full_score_live_p90_budget_admission(
        campaign["execution_plan"], gate
    )
    prior = _wave_completion(campaign, 0)
    local_gate_path = campaign["tmp_path"] / "local-p90-diagnostic.json"
    local_gate_path.write_bytes(full_score._canonical_pretty_json_bytes(gate))
    with pytest.raises(ValueError, match="prior wave must be reconciled"):
        full_score._build_databricks_full_score_run_submit_payload(
            campaign["job"],
            wave_one_producers,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            budget_admission=gate,
            idempotency_attempt_id="wave-001-producer-local-gate-a",
            publication_authorizing=True,
        )
    with pytest.raises(ValueError, match="rejects local-fixture"):
        full_score._build_databricks_full_score_run_submit_payload(
            campaign["job"],
            wave_one_producers,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            prior_wave_completion=prior,
            budget_admission=gate,
            idempotency_attempt_id="wave-001-producer-local-gate-b",
            publication_authorizing=True,
        )
    run = full_score.preview_local_fixture_databricks_full_score_run_submit_payload(
        campaign["job"],
        wave_one_producers,
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        prior_wave_completion=prior,
        budget_admission=gate,
    )
    assert len(run["tasks"]) == 4

    tampered = copy.deepcopy(gate)
    tampered["bootstrap"]["projected_remaining_gpu_hours_samples"][0] += 1
    _close(tampered)
    with pytest.raises(ValueError, match="does not replay"):
        full_score.validate_full_score_live_p90_budget_admission(
            campaign["execution_plan"], tampered
        )
    with pytest.raises(ValueError, match="every prior-wave matched block"):
        full_score.build_full_score_live_p90_budget_admission(
            campaign["execution_plan"],
            blocks[:-1],
            next_wave_index=1,
            ledger_terminal_actual_gpu_hours=1.0,
            ledger_active_reserved_gpu_hours=0.0,
            next_wave_reserved_gpu_hours=reservation,
        )
    allocated = copy.deepcopy(blocks)
    allocated[0]["billed_gpu_seconds"] = {
        "baseline_prefill": 10.0,
        "producer": 12.0,
        "vanilla_prefill": 20.0,
    }
    allocated[0]["billing_source_sha256"] = {
        role: _digest(f"allocated-{role}")
        for role in ("producer", *full_score.FULL_SCORE_METHODS)
    }
    _close(allocated[0])
    with pytest.raises(ValueError, match="incomplete role billing"):
        full_score.build_full_score_live_p90_budget_admission(
            campaign["execution_plan"],
            allocated,
            next_wave_index=1,
            ledger_terminal_actual_gpu_hours=1.0,
            ledger_active_reserved_gpu_hours=0.0,
            next_wave_reserved_gpu_hours=reservation,
        )


def test_remote_prior_wave_replay_uses_only_issuer_cas_on_mac(
    campaign,
    monkeypatch,
):
    completion, authorization, compact_files = (
        _remote_wave_completion_authorization(campaign, 0)
    )
    monkeypatch.setattr(
        full_score,
        "_validate_shard_evidence_record",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        full_score,
        "_validate_full_score_deletion_attestation",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        full_score,
        "_cluster_path",
        lambda *_args, **_kwargs: pytest.fail(
            "Mac prior-wave replay must not resolve /dbfs"
        ),
    )

    full_score._validate_prior_wave_completion(
        completion,
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        replay_raw_evidence=True,
        expected_wave_index=0,
        expected_execution_plan_sha256=campaign["execution_plan"][
            "closed_record_sha256"
        ],
        remote_consumer_authorization=authorization,
        compact_artifact_resolver=compact_files.__getitem__,
    )

    missing_uri = authorization.evidence_bindings[0]["deletion_uri"]
    missing = dict(compact_files)
    del missing[missing_uri]
    with pytest.raises(KeyError):
        full_score._validate_prior_wave_completion(
            completion,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            replay_raw_evidence=True,
            expected_wave_index=0,
            expected_execution_plan_sha256=campaign["execution_plan"][
                "closed_record_sha256"
            ],
            remote_consumer_authorization=authorization,
            compact_artifact_resolver=missing.__getitem__,
        )


def test_remote_cas_threads_wave0_into_wave1_render_reserve_and_replay(
    campaign,
    monkeypatch,
):
    completion, remote_authorization, compact_files = (
        _remote_wave_completion_authorization(campaign, 0)
    )
    monkeypatch.setattr(
        full_score,
        "_validate_shard_evidence_record",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        full_score,
        "_validate_full_score_deletion_attestation",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        full_score,
        "_cluster_path",
        lambda *_args, **_kwargs: pytest.fail(
            "Mac wave-boundary control must not resolve /dbfs"
        ),
    )

    ledger_path = campaign["tmp_path"] / "remote-wave-boundary-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    historical_payload = {
        "run_name": "remote-wave-boundary-history",
        "tasks": [
            {
                "max_retries": 0,
                "new_cluster": {},
                "task_key": "historical_task",
                "timeout_seconds": 3_600,
            }
        ],
        "timeout_seconds": 3_600,
    }
    reserve_databricks_run_attempt_json(
        ledger_path,
        historical_payload,
        attempt_id="remote-wave-boundary-history",
        workload_id="remote-wave-boundary-history",
    )
    historical_ledger = record_databricks_run_terminal_actual_json(
        ledger_path,
        attempt_id="remote-wave-boundary-history",
        terminal_state="succeeded",
        actual_cluster_duration_seconds=1_000.0,
    )
    predecessor_prefix = databricks_ledger_prefix(historical_ledger)
    predecessor_authorization = object()
    ledger_path_sha256 = databricks_ledger_path_sha256(ledger_path)

    def require_predecessor(value, **_kwargs):
        assert value is predecessor_authorization
        return predecessor_prefix, {
            "authorization_sha256": _digest("remote-prior-consumer"),
            "kind": "full_score_phase_terminal",
            "ledger_path_sha256": ledger_path_sha256,
            "ledger_prefix": predecessor_prefix.to_record(),
            "phase": "consumer",
            "terminal_record_sha256": _digest("remote-prior-terminal"),
            "wave_index": 0,
        }

    monkeypatch.setattr(
        full_score,
        "_require_full_score_phase_predecessor_authorization",
        require_predecessor,
    )
    durable_root = remote_authorization.durable_output_root
    wave_zero_shards = campaign["execution_plan"]["waves"][0]["shards"]
    producer_terminal = _close(
        {
            "closed_record_sha256": "",
            "ledger": {
                "ledger_path_sha256": ledger_path_sha256,
                "terminal_prefix": predecessor_prefix.to_record(),
            },
            "phase": "producer",
            "task_billing": [
                {
                    "billed_gpu_seconds": 12.0,
                    "durable_output_root": durable_root,
                    "shard_id": shard["shard_id"],
                }
                for shard in wave_zero_shards
            ],
            "wave_index": 0,
        }
    )
    consumer_terminal = _close(
        {
            "closed_record_sha256": "",
            "ledger": {
                "ledger_path_sha256": ledger_path_sha256,
                "predecessor_prefix": predecessor_prefix.to_record(),
                "terminal_prefix": predecessor_prefix.to_record(),
            },
            "phase": "consumer",
            "task_billing": [
                {
                    "billed_gpu_seconds": 30.0,
                    "durable_output_root": durable_root,
                    "shard_id": shard["shard_id"],
                }
                for shard in wave_zero_shards
            ],
            "wave_index": 0,
        }
    )
    producer_terminal_path = campaign["tmp_path"] / "producer-terminal.json"
    consumer_terminal_path = campaign["tmp_path"] / "consumer-terminal.json"
    producer_terminal_path.write_bytes(
        full_score._canonical_pretty_json_bytes(producer_terminal)
    )
    consumer_terminal_path.write_bytes(
        full_score._canonical_pretty_json_bytes(consumer_terminal)
    )
    compact_files[str(producer_terminal_path)] = producer_terminal_path
    compact_files[str(consumer_terminal_path)] = consumer_terminal_path

    def load_terminal(path, **_kwargs):
        return (
            producer_terminal
            if Path(path) == producer_terminal_path
            else consumer_terminal
        )

    monkeypatch.setattr(
        full_score,
        "load_governed_full_score_phase_terminal_record",
        load_terminal,
    )
    local_blocks = {
        block["shard_id"]: block for block in _matched_blocks(campaign, 0)
    }

    def build_matched(_execution_plan, *, shard_id, **_kwargs):
        return copy.deepcopy(local_blocks[shard_id])

    monkeypatch.setattr(
        full_score,
        "build_full_score_matched_billing_block",
        build_matched,
    )
    block_uris = []
    for shard in wave_zero_shards:
        shard_id = shard["shard_id"]
        evidence_directory = (
            f"{durable_root}/evidence/wave-000/{shard_id}"
        )
        block = full_score.build_governed_full_score_matched_billing_block(
            campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            evidence_dir=evidence_directory,
            producer_terminal_path=producer_terminal_path,
            consumer_terminal_path=consumer_terminal_path,
            ledger_path=ledger_path,
            remote_consumer_authorization=remote_authorization,
            compact_artifact_resolver=compact_files.__getitem__,
        )
        block_uri = f"{durable_root}/control/matched/{shard_id}.json"
        block_path = campaign["tmp_path"] / f"matched-{shard_id}.json"
        block_path.write_bytes(full_score._canonical_pretty_json_bytes(block))
        compact_files[block_uri] = block_path
        block_uris.append(block_uri)
    compact_files.update(campaign["worker_files"])

    wave_one_producers = _phase_payloads(campaign, 1, "producer")
    attempt_id = "remote-wave-001-producer"
    admission_uri = f"{durable_root}/control/wave-001-p90.json"
    admission_path = campaign["tmp_path"] / "remote-wave-001-p90.json"

    def publish_admission(uri, content):
        assert uri == admission_uri
        admission_path.write_bytes(content)
        compact_files[uri] = admission_path
        return admission_path

    rendered, admission = (
        full_score.prepare_governed_full_score_live_p90_phase_submission(
            admission_uri,
            campaign["job"],
            wave_one_producers,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            attempt_id=attempt_id,
            completed_block_paths=block_uris,
            prior_wave_completion=completion,
            ledger_path=ledger_path,
            predecessor_authorization=predecessor_authorization,
            remote_consumer_authorizations=[remote_authorization],
            compact_artifact_resolver=compact_files.__getitem__,
            compact_artifact_publisher=publish_admission,
        )
    )
    assert admission["next_submit_payload_sha256"]
    assert rendered == full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        wave_one_producers,
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=attempt_id,
        prior_wave_completion=completion,
        remote_consumer_authorizations=[remote_authorization],
        compact_artifact_resolver=compact_files.__getitem__,
        budget_admission_path=admission_uri,
        ledger_path=ledger_path,
        predecessor_authorization=predecessor_authorization,
    )
    _updated, submission_authorization = (
        full_score.reserve_governed_full_score_phase_attempt(
            ledger_path,
            rendered,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=1,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=predecessor_authorization,
            budget_admission_path=admission_uri,
            remote_consumer_authorizations=[remote_authorization],
            compact_artifact_resolver=compact_files.__getitem__,
        )
    )
    replayed = full_score.replay_governed_full_score_phase_submission_authorization(
        ledger_path,
        rendered,
        execution_plan=campaign["execution_plan"],
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        wave_index=1,
        phase="producer",
        attempt_id=attempt_id,
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        predecessor_authorization=predecessor_authorization,
        budget_admission_path=admission_uri,
        remote_consumer_authorizations=[remote_authorization],
        compact_artifact_resolver=compact_files.__getitem__,
    )
    assert replayed == submission_authorization


def test_stock_mac_files_cas_collects_terminals_and_writes_wave_one_gate(
    campaign,
    monkeypatch,
):
    local_worker_files = campaign["worker_files"]
    campaign = _volume_campaign(campaign)
    local_worker_files.update(campaign["worker_files"])
    workspace = DatabricksWorkspaceConfig(
        "https://dbc.example/",
        "secret-token",
    )
    remote_files = {
        uri: path.read_bytes() for uri, path in campaign["worker_files"].items()
    }
    download_calls = []

    def download(_workspace, uri, *, max_bytes):
        assert _workspace is workspace
        download_calls.append(uri)
        content = remote_files[uri]
        assert len(content) <= max_bytes
        return content

    def upload(_workspace, uri, content, *, max_bytes):
        assert _workspace is workspace
        assert len(content) <= max_bytes
        existing = remote_files.get(uri)
        if existing is not None and existing != content:
            raise ValueError("exclusive publication conflict")
        remote_files[uri] = content
        return {
            "created": existing is None,
            "dbfs_uri": uri,
            "file_sha256": sha256(content).hexdigest(),
            "size_bytes": len(content),
        }

    monkeypatch.setattr(
        full_score_remote,
        "download_databricks_volume_file_bytes",
        download,
    )
    monkeypatch.setattr(
        full_score_remote,
        "upload_databricks_volume_file_bytes_exclusive",
        upload,
    )
    monkeypatch.setattr(
        full_score,
        "_cluster_path",
        lambda *_args, **_kwargs: pytest.fail(
            "stock-Mac publication control must not resolve /dbfs or /Volumes"
        ),
    )
    monkeypatch.setattr(
        full_score,
        "_validate_shard_evidence_record",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        full_score,
        "_validate_full_score_deletion_attestation",
        lambda *args, **kwargs: None,
    )

    cas = full_score_remote.FullScoreCompactArtifactCAS(
        campaign["tmp_path"] / "stock-mac-cas"
    )
    compact_io = full_score_remote.FullScoreRemoteCompactArtifactIO(
        workspace,
        cas,
    )
    for uri in sorted(campaign["worker_files"]):
        compact_io.download(uri)

    ledger_path = campaign["tmp_path"] / "stock-mac-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    campaign["latency_collection_authorization"].ledger_prefix = (
        databricks_ledger_prefix(
            read_databricks_cluster_hour_ledger_json(ledger_path)
        )
    )
    durable_root = campaign["bundle"].durable_output_root
    before_malformed = len(download_calls)
    with pytest.raises(ValueError, match="exactly one phase"):
        full_score_remote.prepare_governed_full_score_remote_live_p90_phase_submission(
            workspace,
            cas=cas,
            path=f"{durable_root}/control/malformed-p90.json",
            config=campaign["job"],
            worker_payloads=(),
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            attempt_id="malformed-p90",
            completed_block_paths=[f"{durable_root}/control/missing.json"],
            prior_wave_completion={},
            ledger_path=ledger_path,
            predecessor_authorization=object(),
        )
    assert len(download_calls) == before_malformed

    producer_attempt_id = "stock-mac-wave-000-producer"
    producer_submit = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "producer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=producer_attempt_id,
        compact_artifact_resolver=cas.resolve,
    )
    producer_opener = _RoutingDatabricksOpener(
        submit_payload={"run_id": 81_001},
        run_payload=_terminal_run_record(producer_submit, run_id=81_001),
    )
    _response, producer_submission_authorization = (
        full_score.reserve_and_submit_governed_full_score_phase_attempt(
            workspace,
            producer_submit,
            ledger_path=ledger_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="producer",
            attempt_id=producer_attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=campaign[
                "latency_collection_authorization"
            ],
            latency_execution_plan_record=campaign[
                "latency_execution_plan_record"
            ],
            compact_artifact_resolver=cas.resolve,
            opener=producer_opener,
        )
    )
    producer_submit_uri = (
        f"{durable_root}/control/phase/wave-000-producer-submit.json"
    )
    producer_run_uri = (
        f"{durable_root}/control/phase/wave-000-producer-runs-get.json"
    )
    producer_terminal_uri = (
        f"{durable_root}/control/phase/wave-000-producer-terminal.json"
    )
    remote_files[producer_submit_uri] = (
        full_score._canonical_pretty_json_bytes(producer_submit)
    )
    downloads_before_wrong_identity = list(download_calls)
    producer_requests_before_wrong_identity = len(producer_opener.requests)

    def reject_remote_current_user(
        _workspace,
        *,
        expected_user_name,
        opener=None,
    ):
        assert expected_user_name == "researcher@example.com"
        assert opener is producer_opener
        raise ValueError("Databricks current-user identity differs")

    with monkeypatch.context() as identity_patch:
        identity_patch.setattr(
            full_score_remote,
            "require_databricks_current_user_name",
            reject_remote_current_user,
        )
        with pytest.raises(ValueError, match="current-user identity differs"):
            full_score_remote.collect_governed_full_score_remote_phase_attempt(
                workspace,
                cas=cas,
                execution_plan=campaign["execution_plan"],
                inventory=campaign["inventory"],
                shard_plan=campaign["shard_plan"],
                ledger_path=ledger_path,
                submission_authorization=producer_submission_authorization,
                submit_payload_uri=producer_submit_uri,
                control_plane_run_uri=producer_run_uri,
                terminal_record_uri=producer_terminal_uri,
                single_user_name="researcher@example.com",
                opener=producer_opener,
            )
    assert download_calls == downloads_before_wrong_identity
    assert len(producer_opener.requests) == producer_requests_before_wrong_identity
    with pytest.raises(ValueError, match="missing"):
        cas.resolve(producer_submit_uri)
    producer_terminal, producer_authorization = (
        full_score_remote.collect_governed_full_score_remote_phase_attempt(
            workspace,
            cas=cas,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            ledger_path=ledger_path,
            submission_authorization=producer_submission_authorization,
            submit_payload_uri=producer_submit_uri,
            control_plane_run_uri=producer_run_uri,
            terminal_record_uri=producer_terminal_uri,
            single_user_name="researcher@example.com",
            opener=producer_opener,
        )
    )

    producer_completion = _producer_completion(campaign, 0)
    producer_completion_uri = (
        f"{durable_root}/control/producer-ready/wave-000-completion.json"
    )
    producer_completion_bytes = full_score._canonical_pretty_json_bytes(
        producer_completion
    )
    remote_files[producer_completion_uri] = producer_completion_bytes
    compact_io.download(producer_completion_uri)
    ready_authorization = full_score_remote.FullScoreRemoteTreeAuthorization(
        action="producer_ready",
        execution_plan_sha256=campaign["execution_plan"][
            "closed_record_sha256"
        ],
        wave_index=0,
        durable_output_root=durable_root,
        request_sha256=_digest("stock-mac-producer-request"),
        result_uri=producer_completion_uri,
        result_file_sha256=sha256(producer_completion_bytes).hexdigest(),
        result_record_sha256=producer_completion["closed_record_sha256"],
        result_record=producer_completion,
        attestation_uri=(
            f"{durable_root}/control/producer-ready/wave-000-attestation.json"
        ),
        attestation_file_sha256=_digest("stock-mac-producer-attestation-file"),
        attestation_record_sha256=_digest(
            "stock-mac-producer-attestation-record"
        ),
        coordinator_run_id="91001",
        coordinator_run_record_sha256=_digest("stock-mac-producer-run"),
        controller_authorization_record_sha256=_digest(
            "stock-mac-producer-controller-authorization"
        ),
        runs_get_receipt_record_sha256=_digest(
            "stock-mac-producer-runs-get"
        ),
        phase_terminal_record_sha256=producer_terminal[
            "closed_record_sha256"
        ],
        evidence_bindings=(),
        _issuer=full_score_remote._REMOTE_AUTHORIZATION_ISSUER,
    )

    consumer_attempt_id = "stock-mac-wave-000-consumer"
    consumer_submit = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        _phase_payloads(campaign, 0, "consumer"),
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=consumer_attempt_id,
        producer_phase_completion=producer_completion,
        producer_phase_completion_uri=producer_completion_uri,
        remote_ready_authorization=ready_authorization,
        compact_artifact_resolver=cas.resolve,
    )
    consumer_opener = _RoutingDatabricksOpener(
        submit_payload={"run_id": 82_001},
        run_payload=_terminal_run_record(consumer_submit, run_id=82_001),
    )
    _response, consumer_submission_authorization = (
        full_score.reserve_and_submit_governed_full_score_phase_attempt(
            workspace,
            consumer_submit,
            ledger_path=ledger_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=0,
            phase="consumer",
            attempt_id=consumer_attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=producer_authorization,
            remote_ready_authorization=ready_authorization,
            compact_artifact_resolver=cas.resolve,
            opener=consumer_opener,
        )
    )
    consumer_submit_uri = (
        f"{durable_root}/control/phase/wave-000-consumer-submit.json"
    )
    consumer_run_uri = (
        f"{durable_root}/control/phase/wave-000-consumer-runs-get.json"
    )
    consumer_terminal_uri = (
        f"{durable_root}/control/phase/wave-000-consumer-terminal.json"
    )
    remote_files[consumer_submit_uri] = (
        full_score._canonical_pretty_json_bytes(consumer_submit)
    )
    consumer_terminal, consumer_authorization = (
        full_score_remote.collect_governed_full_score_remote_phase_attempt(
            workspace,
            cas=cas,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            ledger_path=ledger_path,
            submission_authorization=consumer_submission_authorization,
            submit_payload_uri=consumer_submit_uri,
            control_plane_run_uri=consumer_run_uri,
            terminal_record_uri=consumer_terminal_uri,
            single_user_name="researcher@example.com",
            opener=consumer_opener,
        )
    )

    wave_completion, fixture_evidence_authorization, evidence_files = (
        _remote_wave_completion_authorization(campaign, 0)
    )
    remote_files.update(
        {uri: path.read_bytes() for uri, path in evidence_files.items()}
    )
    evidence_authorization = full_score_remote.FullScoreRemoteTreeAuthorization(
        action=fixture_evidence_authorization.action,
        execution_plan_sha256=(
            fixture_evidence_authorization.execution_plan_sha256
        ),
        wave_index=fixture_evidence_authorization.wave_index,
        durable_output_root=fixture_evidence_authorization.durable_output_root,
        request_sha256=fixture_evidence_authorization.request_sha256,
        result_uri=fixture_evidence_authorization.result_uri,
        result_file_sha256=fixture_evidence_authorization.result_file_sha256,
        result_record_sha256=(
            fixture_evidence_authorization.result_record_sha256
        ),
        result_record=fixture_evidence_authorization.result_record,
        attestation_uri=fixture_evidence_authorization.attestation_uri,
        attestation_file_sha256=(
            fixture_evidence_authorization.attestation_file_sha256
        ),
        attestation_record_sha256=(
            fixture_evidence_authorization.attestation_record_sha256
        ),
        coordinator_run_id=fixture_evidence_authorization.coordinator_run_id,
        coordinator_run_record_sha256=(
            fixture_evidence_authorization.coordinator_run_record_sha256
        ),
        controller_authorization_record_sha256=(
            fixture_evidence_authorization.controller_authorization_record_sha256
        ),
        runs_get_receipt_record_sha256=(
            fixture_evidence_authorization.runs_get_receipt_record_sha256
        ),
        phase_terminal_record_sha256=consumer_terminal[
            "closed_record_sha256"
        ],
        evidence_bindings=fixture_evidence_authorization.evidence_bindings,
        _issuer=full_score_remote._REMOTE_AUTHORIZATION_ISSUER,
    )
    matched_block_uris = []
    matched_blocks = []
    for shard_id in wave_completion["shard_ids"]:
        evidence_directory = (
            f"{durable_root}/evidence/wave-000/{shard_id}"
        )
        block_uri = (
            f"{durable_root}/control/matched/wave-000/{shard_id}.json"
        )
        block = (
            full_score_remote.write_governed_full_score_remote_matched_billing_block(
                workspace,
                cas=cas,
                path=block_uri,
                execution_plan=campaign["execution_plan"],
                inventory=campaign["inventory"],
                shard_plan=campaign["shard_plan"],
                evidence_dir=evidence_directory,
                producer_terminal_uri=producer_terminal_uri,
                consumer_terminal_uri=consumer_terminal_uri,
                ledger_path=ledger_path,
                remote_consumer_authorization=evidence_authorization,
            )
        )
        matched_block_uris.append(block_uri)
        matched_blocks.append(block)

    wave_one_producers = _phase_payloads(campaign, 1, "producer")
    wave_one_attempt_id = "stock-mac-wave-001-producer"
    admission_uri = (
        f"{durable_root}/control/admission/wave-001-producer-p90.json"
    )
    rendered, admission = (
        full_score_remote.prepare_governed_full_score_remote_live_p90_phase_submission(
            workspace,
            cas=cas,
            path=admission_uri,
            config=campaign["job"],
            worker_payloads=wave_one_producers,
            execution_plan=campaign["execution_plan"],
            completed_block_paths=matched_block_uris,
            remote_consumer_authorizations=[evidence_authorization],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            attempt_id=wave_one_attempt_id,
            prior_wave_completion=wave_completion,
            ledger_path=ledger_path,
            predecessor_authorization=consumer_authorization,
        )
    )
    assert admission["next_submit_payload_sha256"]
    _ledger, submission_authorization = (
        full_score.reserve_governed_full_score_phase_attempt(
            ledger_path,
            rendered,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=1,
            phase="producer",
            attempt_id=wave_one_attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=consumer_authorization,
            budget_admission_path=admission_uri,
            remote_consumer_authorizations=[evidence_authorization],
            compact_artifact_resolver=cas.resolve,
        )
    )
    replayed = full_score.replay_governed_full_score_phase_submission_authorization(
        ledger_path,
        rendered,
        execution_plan=campaign["execution_plan"],
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        wave_index=1,
        phase="producer",
        attempt_id=wave_one_attempt_id,
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        predecessor_authorization=consumer_authorization,
        budget_admission_path=admission_uri,
        remote_consumer_authorizations=[evidence_authorization],
        compact_artifact_resolver=cas.resolve,
    )
    assert replayed == submission_authorization
    assert admission["admitted"] is True
    for uri in (
        producer_run_uri,
        producer_terminal_uri,
        consumer_run_uri,
        consumer_terminal_uri,
        admission_uri,
        *matched_block_uris,
    ):
        assert cas.resolve(uri).read_bytes() == remote_files[uri]

    wave_one_run_id = 83_001
    wave_one_opener = _RoutingDatabricksOpener(
        submit_payload={"run_id": wave_one_run_id},
        run_payload=_terminal_run_record(rendered, run_id=wave_one_run_id),
    )
    assert full_score.submit_governed_full_score_phase_attempt(
        workspace,
        rendered,
        ledger_path=ledger_path,
        submission_authorization=submission_authorization,
        opener=wave_one_opener,
    ) == {"run_id": wave_one_run_id}
    wave_one_submit_uri = (
        f"{durable_root}/control/phase/wave-001-producer-submit.json"
    )
    wave_one_run_uri = (
        f"{durable_root}/control/phase/wave-001-producer-runs-get.json"
    )
    wave_one_terminal_uri = (
        f"{durable_root}/control/phase/wave-001-producer-terminal.json"
    )
    remote_files[wave_one_submit_uri] = full_score._canonical_pretty_json_bytes(
        rendered
    )
    wave_one_terminal, wave_one_producer_authorization = (
        full_score_remote.collect_governed_full_score_remote_phase_attempt(
            workspace,
            cas=cas,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            ledger_path=ledger_path,
            submission_authorization=submission_authorization,
            submit_payload_uri=wave_one_submit_uri,
            control_plane_run_uri=wave_one_run_uri,
            terminal_record_uri=wave_one_terminal_uri,
            single_user_name="researcher@example.com",
            opener=wave_one_opener,
        )
    )
    wave_one_completion = _producer_completion(campaign, 1)
    wave_one_completion_uri = (
        f"{durable_root}/control/producer-ready/wave-001-completion.json"
    )
    wave_one_completion_bytes = full_score._canonical_pretty_json_bytes(
        wave_one_completion
    )
    remote_files[wave_one_completion_uri] = wave_one_completion_bytes
    wave_one_ready_authorization = full_score_remote.FullScoreRemoteTreeAuthorization(
        action="producer_ready",
        execution_plan_sha256=campaign["execution_plan"][
            "closed_record_sha256"
        ],
        wave_index=1,
        durable_output_root=durable_root,
        request_sha256=_digest("stock-mac-wave-one-producer-request"),
        result_uri=wave_one_completion_uri,
        result_file_sha256=sha256(wave_one_completion_bytes).hexdigest(),
        result_record_sha256=wave_one_completion["closed_record_sha256"],
        result_record=wave_one_completion,
        attestation_uri=(
            f"{durable_root}/control/producer-ready/wave-001-attestation.json"
        ),
        attestation_file_sha256=_digest(
            "stock-mac-wave-one-producer-attestation-file"
        ),
        attestation_record_sha256=_digest(
            "stock-mac-wave-one-producer-attestation-record"
        ),
        coordinator_run_id="91002",
        coordinator_run_record_sha256=_digest("stock-mac-wave-one-producer-run"),
        controller_authorization_record_sha256=_digest(
            "stock-mac-wave-one-producer-controller-authorization"
        ),
        runs_get_receipt_record_sha256=_digest(
            "stock-mac-wave-one-producer-runs-get"
        ),
        phase_terminal_record_sha256=wave_one_terminal[
            "closed_record_sha256"
        ],
        evidence_bindings=(),
        _issuer=full_score_remote._REMOTE_AUTHORIZATION_ISSUER,
    )
    wave_one_consumers = _phase_payloads(campaign, 1, "consumer")
    consumer_p90_uri = (
        f"{durable_root}/control/admission/wave-001-consumer-p90.json"
    )
    before_missing_ready = len(download_calls)
    with pytest.raises(ValueError, match="producer-ready authority"):
        full_score_remote.prepare_governed_full_score_remote_live_p90_phase_submission(
            workspace,
            cas=cas,
            path=consumer_p90_uri,
            config=campaign["job"],
            worker_payloads=wave_one_consumers,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            attempt_id="stock-mac-wave-001-consumer",
            completed_block_paths=matched_block_uris,
            prior_wave_completion=wave_completion,
            ledger_path=ledger_path,
            predecessor_authorization=wave_one_producer_authorization,
            producer_phase_completion=wave_one_completion,
            producer_phase_completion_uri=wave_one_completion_uri,
            remote_consumer_authorizations=[evidence_authorization],
        )
    assert len(download_calls) == before_missing_ready
    consumer_rendered, consumer_admission = (
        full_score_remote.prepare_governed_full_score_remote_live_p90_phase_submission(
            workspace,
            cas=cas,
            path=consumer_p90_uri,
            config=campaign["job"],
            worker_payloads=wave_one_consumers,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            attempt_id="stock-mac-wave-001-consumer",
            completed_block_paths=matched_block_uris,
            prior_wave_completion=wave_completion,
            ledger_path=ledger_path,
            predecessor_authorization=wave_one_producer_authorization,
            producer_phase_completion=wave_one_completion,
            producer_phase_completion_uri=wave_one_completion_uri,
            remote_ready_authorization=wave_one_ready_authorization,
            remote_consumer_authorizations=[evidence_authorization],
        )
    )
    assert consumer_admission["next_phase"] == "consumer"
    assert all(
        task["task_key"].startswith(
            f"{campaign['job'].task_key_prefix}_wave_001_consumer_"
        )
        for task in consumer_rendered["tasks"]
    )


def test_governed_p90_gate_binds_files_payload_ledger_and_is_one_shot(
    campaign,
    monkeypatch,
):
    blocks = []
    block_paths = []
    block_root = campaign["tmp_path"] / "matched-blocks"
    block_root.mkdir()
    for local_block in _matched_blocks(campaign, 0):
        block = copy.deepcopy(local_block)
        block["authorization_scope"] = (
            full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
        )
        block["governed_sources"] = {"fixture": "sealed-by-focused-test"}
        _close(block)
        path = block_root / f"{block['shard_id']}.json"
        path.write_bytes(full_score._canonical_pretty_json_bytes(block))
        blocks.append(block)
        block_paths.append(path)

    def load_test_block(path, **_kwargs):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            record.get("authorization_scope")
            != full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
            or record.get("closed_record_sha256")
            != full_score._closed_record_sha256(record)
        ):
            raise ValueError("test governed matched block closure drift")
        return record

    monkeypatch.setattr(
        full_score,
        "load_governed_full_score_matched_billing_block",
        load_test_block,
    )

    def load_test_evidence(directory, **_kwargs):
        root = Path(directory)
        return (
            json.loads((root / "evidence.json").read_text(encoding="utf-8")),
            json.loads(
                (root / "deletion-attestation.json").read_text(encoding="utf-8")
            ),
        )

    monkeypatch.setattr(
        full_score,
        "load_governed_full_score_shard_evidence",
        load_test_evidence,
    )
    live_ledger_path = campaign["tmp_path"] / "live-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        live_ledger_path,
        ledger_id="publication-full-score",
    )
    historical_payload = {
        "run_name": "historical-full-score",
        "tasks": [
            {
                "max_retries": 0,
                "new_cluster": {},
                "task_key": "historical_task",
                "timeout_seconds": 3_600,
            }
        ],
        "timeout_seconds": 3_600,
    }
    reserve_databricks_run_attempt_json(
        live_ledger_path,
        historical_payload,
        attempt_id="historical-attempt",
        workload_id="full-score-history",
    )
    historical_ledger = record_databricks_run_terminal_actual_json(
        live_ledger_path,
        attempt_id="historical-attempt",
        terminal_state="succeeded",
        actual_cluster_duration_seconds=1_000.0,
    )
    assert historical_ledger.active_reserved_cluster_hours == 0
    predecessor_prefix = databricks_ledger_prefix(historical_ledger)
    predecessor_authorization = object()
    ledger_path_sha256 = databricks_ledger_path_sha256(live_ledger_path)

    def require_test_predecessor(value, **_kwargs):
        if value is not predecessor_authorization:
            raise TypeError("test predecessor capability drift")
        return predecessor_prefix, {
            "authorization_sha256": _digest("prior-consumer-authorization"),
            "kind": "full_score_phase_terminal",
            "ledger_path_sha256": ledger_path_sha256,
            "ledger_prefix": predecessor_prefix.to_record(),
            "phase": "consumer",
            "terminal_record_sha256": _digest("prior-consumer-terminal"),
            "wave_index": 0,
        }

    monkeypatch.setattr(
        full_score,
        "_require_full_score_phase_predecessor_authorization",
        require_test_predecessor,
    )
    for block, path in zip(blocks, block_paths, strict=True):
        block["ledger_lineage"] = {
            "producer": {
                "ledger_path_sha256": ledger_path_sha256,
                "terminal_prefix": predecessor_prefix.to_record(),
            },
            "consumer": {
                "ledger_path_sha256": ledger_path_sha256,
                "predecessor_prefix": predecessor_prefix.to_record(),
                "terminal_prefix": predecessor_prefix.to_record(),
            },
        }
        _close(block)
        path.write_bytes(full_score._canonical_pretty_json_bytes(block))

    wave_one_producers = _phase_payloads(campaign, 1, "producer")
    reservation = full_score.full_score_wave_worst_case_gpu_hours(
        wave_one_producers
    )
    diagnostic = full_score.build_full_score_live_p90_budget_admission(
        campaign["execution_plan"],
        _matched_blocks(campaign, 0),
        next_wave_index=1,
        ledger_terminal_actual_gpu_hours=(
            historical_ledger.terminal_actual_cluster_hours
        ),
        ledger_active_reserved_gpu_hours=0.0,
        next_wave_reserved_gpu_hours=reservation,
    )
    prior = _wave_completion(campaign, 0)
    attempt_id = "wave-001-producer-attempt-001"
    candidate = (
        full_score.preview_local_fixture_databricks_full_score_run_submit_payload(
            campaign["job"],
            wave_one_producers,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
            prior_wave_completion=prior,
            budget_admission=diagnostic,
        )
    )
    candidate = full_score.bind_databricks_run_idempotency_token(
        candidate,
        attempt_id=attempt_id,
    )
    admission_path = campaign["tmp_path"] / "wave-001-producer-admission.json"
    admission = full_score.write_governed_full_score_live_p90_budget_admission(
        admission_path,
        campaign["execution_plan"],
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        completed_block_paths=block_paths,
        next_wave_index=1,
        next_phase="producer",
        attempt_id=attempt_id,
        next_submit_payload=candidate,
        ledger_path=live_ledger_path,
        predecessor_authorization=predecessor_authorization,
    )
    assert admission["admitted"] is True
    tampered = copy.deepcopy(blocks[0])
    tampered["billed_gpu_seconds"]["producer"] += 1
    block_paths[0].write_bytes(full_score._canonical_pretty_json_bytes(tampered))
    with pytest.raises(ValueError, match="closure drift"):
        full_score.load_governed_full_score_live_p90_budget_admission(
            admission_path,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            next_submit_payload=candidate,
            ledger_path=live_ledger_path,
            predecessor_authorization=predecessor_authorization,
        )
    block_paths[0].write_bytes(
        full_score._canonical_pretty_json_bytes(blocks[0])
    )
    rendered = full_score.build_databricks_full_score_run_submit_payload(
        campaign["job"],
        wave_one_producers,
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        attempt_id=attempt_id,
        prior_wave_completion=prior,
        budget_admission_path=admission_path,
        ledger_path=live_ledger_path,
        predecessor_authorization=predecessor_authorization,
    )
    assert rendered == candidate
    reserve_databricks_run_attempt_json(
        live_ledger_path,
        rendered,
        attempt_id=attempt_id,
        workload_id=full_score._full_score_phase_workload_id(
            campaign["execution_plan"],
            wave_index=1,
            phase="producer",
        ),
    )
    with pytest.raises(ValueError, match="nonzero replay requires"):
        full_score.replay_governed_full_score_phase_submission_authorization(
            live_ledger_path,
            rendered,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=1,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=predecessor_authorization,
        )
    with pytest.raises(ValueError, match="pre-reservation intent"):
        full_score.replay_governed_full_score_phase_submission_authorization(
            live_ledger_path,
            rendered,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=1,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=predecessor_authorization,
            budget_admission_path=admission_path,
        )
    live_ledger_path.write_bytes(
        full_score._canonical_pretty_json_bytes(
            databricks_cluster_hour_ledger_to_record(historical_ledger)
        )
    )
    reserved = full_score.reserve_governed_full_score_phase_attempt(
        live_ledger_path,
        rendered,
        execution_plan=campaign["execution_plan"],
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        wave_index=1,
        phase="producer",
        attempt_id=attempt_id,
        qualification_launch_authorization=campaign[
            "qualification_launch_authorization"
        ],
        predecessor_authorization=predecessor_authorization,
        budget_admission_path=admission_path,
    )
    assert reserved[0].active_reserved_cluster_hours == 24.0
    with pytest.raises(
        ValueError,
        match="live ledger changed while building the P90 admission",
    ):
        full_score.reserve_governed_full_score_phase_attempt(
            live_ledger_path,
            rendered,
            execution_plan=campaign["execution_plan"],
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            wave_index=1,
            phase="producer",
            attempt_id=attempt_id,
            qualification_launch_authorization=campaign[
                "qualification_launch_authorization"
            ],
            predecessor_authorization=predecessor_authorization,
            budget_admission_path=admission_path,
        )


def test_matched_billing_keeps_the_consumer_task_indivisible(campaign):
    shard = campaign["execution_plan"]["waves"][0]["shards"][0]
    evidence = {
        "closed_record_sha256": "",
        "durable_evidence_committed": True,
        "execution_plan_sha256": campaign["execution_plan"][
            "closed_record_sha256"
        ],
        "method_wall_clock": "time.monotonic_ns",
        "method_wall_seconds": {
            "baseline_prefill": 11.0,
            "vanilla_prefill": 13.0,
        },
        "paired_examples": [
            {
                "dataset": item["dataset"],
                "example_id": item["example_id"],
                "methods": {
                    method: {"completion_tokens": 2}
                    for method in full_score.FULL_SCORE_METHODS
                },
            }
            for item in shard["items"]
        ],
        "record_type": full_score.FULL_SCORE_SHARD_EVIDENCE_RECORD_TYPE,
        "schema_version": full_score.FULL_SCORE_SHARD_EVIDENCE_SCHEMA_VERSION,
        "scorers": full_score._scorer_contract_record(),
        "shard_id": shard["shard_id"],
        "shard_items_sha256": shard["items_sha256"],
        "wave_index": 0,
    }
    _close(evidence)
    deletion = {
        "closed_record_sha256": "",
        "evidence_closed_record_sha256": evidence["closed_record_sha256"],
        "execution_plan_sha256": campaign["execution_plan"][
            "closed_record_sha256"
        ],
        "lifecycle": ["commit_durable_evidence", "delete_ephemeral_q8_kv"],
        "record_type": full_score.FULL_SCORE_DELETION_ATTESTATION_RECORD_TYPE,
        "schema_version": full_score.FULL_SCORE_DELETION_ATTESTATION_SCHEMA_VERSION,
        "shard_id": shard["shard_id"],
        "wave_index": 0,
    }
    _close(deletion)
    block = full_score.build_full_score_matched_billing_block(
        campaign["execution_plan"],
        wave_index=0,
        shard_id=shard["shard_id"],
        evidence_record=evidence,
        deletion_attestation=deletion,
        producer_billed_gpu_seconds=20.0,
        consumer_task_billed_gpu_seconds=31.0,
        billing_source_sha256={
            "consumer_task": _digest("consumer-task-billing"),
            "producer": _digest("producer-task-billing"),
        },
    )
    assert block["billed_gpu_seconds"] == {
        "consumer_task": 31.0,
        "producer": 20.0,
    }
    assert block["consumer_task_diagnostics"] == {
        "attribution": "indivisible_no_per_arm_billed_seconds",
        "method_wall_clock": "time.monotonic_ns",
        "method_wall_seconds": {
            "baseline_prefill": 11.0,
            "vanilla_prefill": 13.0,
        },
        "shared_or_unattributed_seconds": 7.0,
    }


def test_state_machine_and_exclusive_writes_fail_closed(tmp_path):
    lifecycle = full_score.FullScoreShardLifecycle("consumer")
    with pytest.raises(RuntimeError, match="invalid full-score lifecycle"):
        lifecycle.advance("delete_ephemeral_q8_kv")
    for event in (
        "verify_ready_shard",
        "baseline_inference",
        "vanilla_inference",
        "validate_paired_outputs",
        "commit_durable_evidence",
        "delete_ephemeral_q8_kv",
    ):
        lifecycle.advance(event)
    assert lifecycle.state == "ephemeral_deleted"

    evidence = tmp_path / "evidence.json"
    full_score._exclusive_write_bytes(evidence, b"first\n")
    with pytest.raises(FileExistsError):
        full_score._exclusive_write_bytes(evidence, b"replacement\n")
    assert evidence.read_bytes() == b"first\n"

    source = tmp_path / "new-evidence.bin"
    source.write_bytes(b"new evidence")
    durable = tmp_path / "durable-evidence.bin"
    durable.write_bytes(b"reviewed evidence")
    with pytest.raises(FileExistsError):
        full_score._durable_copy(source, durable)
    assert durable.read_bytes() == b"reviewed evidence"

    durable_dir = tmp_path / "committed-shard"
    durable_dir.mkdir()
    execution_plan = _close({"closed_record_sha256": "", "plan": "test"})
    committed_evidence = _close(
        {
            "closed_record_sha256": "",
            "ready_shard_sha256": _digest("ready"),
            "shard_id": "shard-00000",
            "wave_index": 0,
            "worker_index": 0,
        }
    )
    payload = {
        "authorization_scope": full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
    }
    lifecycle_events = [
        "verify_ready_shard",
        "baseline_inference",
        "vanilla_inference",
        "validate_paired_outputs",
        "commit_durable_evidence",
        "delete_ephemeral_q8_kv",
    ]
    deletion = full_score._write_or_validate_deletion_attestation(
        durable_dir,
        payload=payload,
        evidence=committed_evidence,
        execution_plan=execution_plan,
        lifecycle=lifecycle_events,
    )
    assert deletion == full_score._write_or_validate_deletion_attestation(
        durable_dir,
        payload=payload,
        evidence=committed_evidence,
        execution_plan=execution_plan,
        lifecycle=lifecycle_events,
    )
    deletion_path = durable_dir / "deletion-attestation.json"
    tampered_deletion = copy.deepcopy(deletion)
    tampered_deletion["worker_index"] = 1
    deletion_path.write_text(json.dumps(tampered_deletion), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from recovery state"):
        full_score._write_or_validate_deletion_attestation(
            durable_dir,
            payload=payload,
            evidence=committed_evidence,
            execution_plan=execution_plan,
            lifecycle=lifecycle_events,
        )


def test_consumer_recovery_finishes_a_partially_deleted_ready_tree(
    campaign,
    monkeypatch,
):
    payload = copy.deepcopy(_phase_payloads(campaign, 0, "consumer")[0])
    payload["durable_output_root"] = str(campaign["tmp_path"] / "recovery-root")
    shard = payload["shards"][0]
    shard_id = shard["shard_id"]
    ready_dir = full_score._ready_shard_dir(payload, shard_id)
    ready_dir.mkdir(parents=True)
    (ready_dir / "partially-remaining-q8.bin").write_bytes(b"partial")
    durable_dir = (
        Path(payload["durable_output_root"]) / "evidence" / "wave-000" / shard_id
    )
    durable_dir.mkdir(parents=True)
    (durable_dir / "evidence.json").write_text("{}\n", encoding="utf-8")
    evidence = {
        "closed_record_sha256": _digest("committed-recovery-evidence"),
        "ready_shard_sha256": _digest("committed-recovery-ready"),
        "runtime_verification": _runtime_verification(),
        "shard_id": shard_id,
        "wave_index": 0,
        "worker_index": payload["worker_index"],
    }

    def load_recovery_evidence(path, **kwargs):
        assert Path(path) == durable_dir
        if kwargs["require_deletion"]:
            assert not ready_dir.exists()
            deletion = json.loads(
                (durable_dir / "deletion-attestation.json").read_text(
                    encoding="utf-8"
                )
            )
            return evidence, deletion
        return evidence, None

    monkeypatch.setattr(
        full_score,
        "load_governed_full_score_shard_evidence",
        load_recovery_evidence,
    )
    recovered = full_score._recover_committed_consumer_shard(
        payload,
        shard,
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
        runtime_verification=_runtime_verification(),
    )
    assert recovered == evidence
    assert not ready_dir.exists()
    deletion = json.loads(
        (durable_dir / "deletion-attestation.json").read_text(encoding="utf-8")
    )
    assert deletion["lifecycle"][-2:] == [
        "commit_durable_evidence",
        "delete_ephemeral_q8_kv",
    ]


def test_bootstrap_builds_and_reexecs_only_the_locked_runtime(monkeypatch):
    script = full_score.FULL_SCORE_RUNNER_SCRIPT
    namespace = {"__name__": "full_score_runner_test"}
    exec(compile(script, "full_score_runner.py", "exec"), namespace)
    monkeypatch.setenv("PIP_CONFIG_FILE", "/attacker/pip.conf")
    monkeypatch.setenv("PIP_INDEX_URL", "https://attacker.invalid/simple")
    monkeypatch.setenv("pip_no_index", "1")
    monkeypatch.setenv("PIP_REQUIREMENT", "/attacker/requirements.txt")
    monkeypatch.setenv("_PIP_STANDALONE_CERT", "/attacker/cert.pem")
    monkeypatch.setenv("PYTHONHOME", "/attacker/python-home")
    monkeypatch.setenv("PYTHONPATH", "/attacker/python-path")
    monkeypatch.setenv("VIRTUAL_ENV", "/attacker/venv")
    monkeypatch.setenv("FLASHINFER_LOGGING_LEVEL", "DEBUG")
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    environment = namespace["_pip_subprocess_environment"]()
    assert {key for key in environment if key.upper().startswith("PIP_")} == {
        "PIP_CONFIG_FILE",
        "PIP_DISABLE_PIP_VERSION_CHECK",
        "PIP_NO_INPUT",
    }
    assert environment["PIP_CONFIG_FILE"] == os.devnull
    assert (
        environment["FLASHINFER_LOGGING_LEVEL"]
        == GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL
    )
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"
    assert environment["PYTHONWARNINGS"] == GPU_RUNTIME_PYTHONWARNINGS
    assert "_PIP_STANDALONE_CERT" not in environment
    assert all(
        variable not in environment
        for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")
    )
    assert "--runner-sha256" in script
    assert '"--require-hashes"' in script
    assert '"--only-binary"' in script
    assert '"--extra-index-url"' not in script
    assert script.count('"--no-deps"') == 3
    assert script.index('"--require-hashes"') < script.index('"--no-deps", vllm_spec')
    assert script.index('"--no-deps", vllm_spec') < script.index(
        '"--no-deps", flashinfer_spec'
    )
    assert script.index('"--no-deps", flashinfer_spec') < script.index(
        '"--no-deps", package_spec'
    )
    assert '"venv", "--copies"' in script
    assert "runtime-closure-manifest-sha256" in script
    assert "patched-flashinfer-wheel-sha256" in script
    assert 'pip, "check"' not in script
    assert "os.execve(" in script


def test_native_v2_runtime_binding_rejects_attestation_and_artifact_drift():
    binding = _runtime_verification()
    full_score._validate_runtime_verification_binding(binding)

    tampered_attestation = copy.deepcopy(binding)
    tampered_attestation["attestation"]["pip_check_ok"] = False
    with pytest.raises(ValueError, match="attestation pip_check_ok differs"):
        full_score._validate_runtime_verification_binding(tampered_attestation)

    tampered_artifact = copy.deepcopy(binding)
    tampered_artifact["artifacts"]["package_wheel_sha256"] = _digest(
        "unreviewed-package"
    )
    with pytest.raises(ValueError, match="artifact binding identity drift"):
        full_score._validate_runtime_verification_binding(tampered_artifact)

    resealed_origin = copy.deepcopy(binding)
    resealed_origin["attestation"]["vllm_direct_url"] = (
        "file:///attacker/resealed-vllm.whl"
    )
    resealed_origin["attestation_sha256"] = full_score._canonical_sha256(
        resealed_origin["attestation"]
    )
    resealed_origin["file_sha256"] = sha256(
        full_score._canonical_pretty_json_bytes(resealed_origin["attestation"])
    ).hexdigest()
    with pytest.raises(ValueError, match="vllm_direct_url differs"):
        full_score._validate_runtime_verification_binding(resealed_origin)


def _bounded_stream(raw, *, limit_exceeded=False):
    return SimpleNamespace(
        retained=raw,
        byte_count=len(raw),
        sha256=sha256(raw).hexdigest(),
        limit_exceeded=limit_exceeded,
    )


def _bounded_verifier_result(
    stdout,
    *,
    stderr=b"",
    returncode=0,
    timed_out=False,
    output_limit_exceeded=False,
):
    return SimpleNamespace(
        stdout=_bounded_stream(stdout),
        stderr=_bounded_stream(stderr),
        returncode=returncode,
        timed_out=timed_out,
        output_limit_exceeded=output_limit_exceeded,
    )


def test_runtime_verifier_requires_bounded_canonical_stdout_and_empty_stderr(
    campaign,
    monkeypatch,
):
    runtime = campaign["bundle"].runtime
    bootstrap = campaign["payloads"][0]["bootstrap_artifacts"]
    canonical = full_score._canonical_pretty_json_bytes(_runtime_attestation())
    output_path = campaign["tmp_path"] / "native-v2-runtime-attestation.json"
    bounded_call = {}
    monkeypatch.setenv("FLASHINFER_LOGGING_LEVEL", "DEBUG")
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")

    def successful_verifier(arguments, **kwargs):
        bounded_call["count"] = bounded_call.get("count", 0) + 1
        bounded_call["arguments"] = arguments
        bounded_call.update(kwargs)
        return _bounded_verifier_result(canonical)

    monkeypatch.setattr(
        full_score,
        "_run_bounded_binary_subprocess",
        successful_verifier,
    )
    binding = full_score._run_runtime_verifier(
        runtime,
        bootstrap,
        output_path,
        runner=full_score._subprocess_command_runner,
    )
    assert output_path.read_bytes() == canonical
    assert binding["attestation"] == _runtime_attestation()
    assert (
        "verify_gpu_qualification_v2_runtime_installation"
        in (bounded_call["arguments"][2])
    )
    assert bounded_call["timeout_seconds"] == (
        full_score.FULL_SCORE_RUNTIME_VERIFIER_TIMEOUT_SECONDS
    )
    assert bounded_call["output_limit_bytes"] == (
        full_score.FULL_SCORE_RUNTIME_VERIFIER_OUTPUT_LIMIT_BYTES
    )
    assert (
        bounded_call["environment"]["FLASHINFER_LOGGING_LEVEL"]
        == GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL
    )
    assert (
        bounded_call["environment"]["PYTHONWARNINGS"]
        == GPU_RUNTIME_PYTHONWARNINGS
    )
    repeated = full_score._run_runtime_verifier(
        runtime,
        bootstrap,
        output_path,
        runner=full_score._subprocess_command_runner,
    )
    assert repeated == binding
    assert bounded_call["count"] == 2

    output_path.unlink()
    secret = b"must-not-leak"
    monkeypatch.setattr(
        full_score,
        "_run_bounded_binary_subprocess",
        lambda *args, **kwargs: _bounded_verifier_result(
            canonical,
            stderr=secret,
        ),
    )
    with pytest.raises(RuntimeError, match="wrote to stderr") as error:
        full_score._run_runtime_verifier(
            runtime,
            bootstrap,
            output_path,
            runner=full_score._subprocess_command_runner,
        )
    assert secret.decode() not in str(error.value)
    assert not output_path.exists()

    noncanonical = json.dumps(_runtime_attestation(), sort_keys=True).encode() + b"\n"
    monkeypatch.setattr(
        full_score,
        "_run_bounded_binary_subprocess",
        lambda *args, **kwargs: _bounded_verifier_result(noncanonical),
    )
    with pytest.raises(RuntimeError, match="output is not canonical"):
        full_score._run_runtime_verifier(
            runtime,
            bootstrap,
            output_path,
            runner=full_score._subprocess_command_runner,
        )

    origin_drift = _runtime_attestation()
    origin_drift["vllm_direct_url"] = "file:///attacker/substituted-vllm.whl"
    monkeypatch.setattr(
        full_score,
        "_run_bounded_binary_subprocess",
        lambda *args, **kwargs: _bounded_verifier_result(
            full_score._canonical_pretty_json_bytes(origin_drift)
        ),
    )
    with pytest.raises(ValueError, match="vllm_direct_url differs"):
        full_score._run_runtime_verifier(
            runtime,
            bootstrap,
            output_path,
            runner=full_score._subprocess_command_runner,
        )


def test_injected_runtime_verifier_requires_canonical_full_attestation(
    campaign,
):
    runtime = campaign["bundle"].runtime
    bootstrap = campaign["payloads"][0]["bootstrap_artifacts"]
    output_path = campaign["tmp_path"] / "injected-runtime-attestation.json"

    def noncanonical_runner(*_args, **_kwargs):
        output_path.write_text(
            json.dumps(_runtime_attestation(), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match="is not canonical"):
        full_score._run_runtime_verifier(
            runtime,
            bootstrap,
            output_path,
            runner=noncanonical_runner,
        )


def test_qualification_and_full_score_runners_are_independently_pinned(campaign):
    qualification_runner = campaign[
        "bundle"
    ].gpu_qualification.artifact_pins.runner_sha256
    assert qualification_runner != campaign["bundle"].runner_sha256
    assert campaign["bundle"].runner_sha256 == full_score.FULL_SCORE_RUNNER_SHA256
    full_score.validate_full_score_worker_payload(
        campaign["payloads"][0],
        inventory=campaign["inventory"],
        shard_plan=campaign["shard_plan"],
        execution_plan=campaign["execution_plan"],
    )
    conflated_pins = replace(
        campaign["bundle"].gpu_qualification.artifact_pins,
        runner_sha256=full_score.FULL_SCORE_RUNNER_SHA256,
    )
    conflated_qualification = replace(
        campaign["bundle"].gpu_qualification,
        artifact_pins=conflated_pins,
    )
    with pytest.raises(ValueError, match="runner identities must be distinct"):
        replace(
            campaign["bundle"],
            gpu_qualification=conflated_qualification,
        )
    with pytest.raises(ValueError, match="runner identities must be distinct"):
        replace(
            campaign["job"],
            gpu_qualification=conflated_qualification,
        )

    conflated_payload = copy.deepcopy(campaign["payloads"][0])
    conflated_payload["gpu_qualification"]["artifact_pins"]["runner_sha256"] = (
        full_score.FULL_SCORE_RUNNER_SHA256
    )
    _close(conflated_payload)
    with pytest.raises(ValueError, match="runner identities must be distinct"):
        full_score.validate_full_score_worker_payload(
            conflated_payload,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
        )

    substituted_payload = copy.deepcopy(campaign["payloads"][0])
    bootstrap = substituted_payload["bootstrap_artifacts"]
    bootstrap["package_wheel_uri"] = "dbfs:/attacker/substituted-cachet.whl"
    bootstrap["package_wheel_sha256"] = _digest("substituted-cachet-wheel")
    bootstrap["locked_runtime_identity_sha256"] = (
        full_score._locked_runtime_identity_sha256(
            runner_sha256=bootstrap["runner_sha256"],
            package_wheel_sha256=bootstrap["package_wheel_sha256"],
            runtime_lock_sha256=bootstrap["runtime_lock_sha256"],
            patched_vllm_wheel_sha256=bootstrap["patched_vllm_wheel_sha256"],
            patched_flashinfer_wheel_sha256=(
                bootstrap["patched_flashinfer_wheel_sha256"]
            ),
            runtime_closure_manifest_sha256=(
                bootstrap["runtime_closure_manifest_sha256"]
            ),
        )
    )
    _close(substituted_payload)
    with pytest.raises(ValueError, match="package wheel differs"):
        full_score.validate_full_score_worker_payload(
            substituted_payload,
            inventory=campaign["inventory"],
            shard_plan=campaign["shard_plan"],
            execution_plan=campaign["execution_plan"],
        )


def test_runtime_install_specs_bind_the_staged_wheel_uris(campaign):
    runtime = campaign["bundle"].runtime
    with pytest.raises(ValueError, match="vLLM install spec URI differs"):
        replace(
            runtime,
            vllm_wheel_install_spec=(
                "vllm @ file:///dbfs/attacker/substituted-vllm.whl#sha256="
                f"{runtime.patched_vllm_wheel_sha256}"
            ),
        )
    with pytest.raises(ValueError, match="FlashInfer install spec URI differs"):
        replace(
            runtime,
            flashinfer_wheel_install_spec=(
                "flashinfer-python @ file:///dbfs/attacker/substituted-flashinfer.whl"
                f"#sha256={runtime.patched_flashinfer_wheel_sha256}"
            ),
        )


def test_governed_paths_and_recursive_delete_reject_ancestor_symlinks(
    tmp_path,
    monkeypatch,
):
    dbfs_root = tmp_path / "dbfs"
    outside = tmp_path / "outside"
    dbfs_root.mkdir()
    (outside / "shard").mkdir(parents=True)
    governed_file = outside / "bound.json"
    governed_file.write_text("{}", encoding="utf-8")
    outside_payload = outside / "shard" / "payload.bin"
    outside_payload.write_bytes(b"must survive")
    (dbfs_root / "alias").symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        full_score,
        "_cluster_path",
        lambda value: dbfs_root / str(value).removeprefix("dbfs:/"),
    )
    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        full_score._governed_existing_file(
            "dbfs:/alias/bound.json",
            "adversarial governed input",
        )
    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        full_score._delete_directory_tree_no_follow(
            dbfs_root / "alias" / "shard",
            label="adversarial recursive delete",
        )
    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        full_score._atomic_write_bytes(
            dbfs_root / "alias" / "unreviewed-output.json",
            b"{}\n",
        )
    assert outside_payload.read_bytes() == b"must survive"
    assert not (outside / "unreviewed-output.json").exists()


def test_full_score_aggregate_uses_paired_examples_and_all_niah_cells(
    tmp_path, monkeypatch
):
    assert full_score.FULL_SCORE_AGGREGATE_RECORD_TYPE == (
        "cachet.full_score_aggregate.v2"
    )
    assert full_score.FULL_SCORE_AGGREGATE_SCHEMA_VERSION == 2
    counts = {"biography": 1, "hotpotqa": 1, "musique": 1, "niah": 1000}
    paths = _write_sources(tmp_path / "full-scores", counts)
    inventory = load_full_score_inventory(paths, tokenizer=_CharacterTokenizer())
    plan = build_full_score_shard_plan(
        inventory,
        plan_id="aggregate-test",
        max_workers=1,
        target_cache_prefix_tokens_per_shard=(
            sum(item.cache_prefix_tokens for item in inventory.items) + 1
        ),
    )
    monkeypatch.setattr(full_score, "PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS", 25)
    deltas = {"biography": -0.1, "hotpotqa": -0.2, "musique": -0.3, "niah": -0.4}
    registry = default_dataset_scorer_registry()
    niah_index = 0
    pairs = []
    for item in inventory.items:
        scorer = registry.get(item.dataset)
        baseline = {metric: 0.75 for metric in scorer.metric_names}
        vanilla = {
            metric: value + deltas[item.dataset] for metric, value in baseline.items()
        }
        cell_id = None
        if item.dataset == "niah":
            cell_id = NIAH_CELL_IDS[niah_index % len(NIAH_CELL_IDS)]
            niah_index += 1
        pairs.append(
            {
                "dataset": item.dataset,
                "example_id": item.example_id,
                "identity_sha256": item.identity_sha256,
                "methods": {
                    "baseline_prefill": {
                        "artifact_id": None,
                        "completion_tokens": 3,
                        "output_sha256": _digest(
                            f"baseline-output-{item.dataset}-{item.example_id}"
                        ),
                        "parser_status": "ok",
                        "parser_valid": True,
                        "quality_scores": baseline,
                        "request_id": "",
                        "scorer_id": scorer.scorer_id,
                        "scorer_version": scorer.version,
                    },
                    "vanilla_prefill": {
                        "artifact_id": f"artifact-{item.dataset}-{item.example_id}",
                        "completion_tokens": 3,
                        "output_sha256": _digest(
                            f"vanilla-output-{item.dataset}-{item.example_id}"
                        ),
                        "parser_status": "ok",
                        "parser_valid": True,
                        "quality_scores": vanilla,
                        "request_id": f"vanilla-{item.dataset}-{item.example_id}",
                        "scorer_id": scorer.scorer_id,
                        "scorer_version": scorer.version,
                    },
                },
                "natural_prompt_sha256": item.natural_prompt_sha256,
                "niah_cell_id": cell_id,
            }
        )
    shard = plan["shards"][0]
    evidence = _close(
        {
            "authorization_scope": (
                full_score.FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE
            ),
            "closed_record_sha256": "",
            "durable_evidence_committed": True,
            "inventory_sha256": inventory.inventory_sha256,
            "paired_examples": pairs,
            "record_type": full_score.FULL_SCORE_SHARD_EVIDENCE_RECORD_TYPE,
            "schema_version": full_score.FULL_SCORE_SHARD_EVIDENCE_SCHEMA_VERSION,
            "scorers": full_score._scorer_contract_record(),
            "shard_id": shard["shard_id"],
            "shard_items_sha256": shard["items_sha256"],
            "shard_plan_sha256": plan["closed_record_sha256"],
        }
    )
    execution_plan = full_score.build_full_score_execution_plan(inventory, plan)
    monkeypatch.setattr(
        full_score,
        "_validate_publication_full_score_inputs",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(TypeError, match="final-consumer"):
        full_score.aggregate_full_score_shard_evidence(
            inventory,
            plan,
            [evidence],
            authorization_scope=(
                full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
            ),
            execution_plan=execution_plan,
        )
    publication_evidence = copy.deepcopy(evidence)
    publication_evidence["authorization_scope"] = (
        full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
    )
    publication_evidence["execution_plan_sha256"] = execution_plan[
        "closed_record_sha256"
    ]
    publication_evidence["ready_shard_sha256"] = _digest("publication-ready")
    publication_evidence["wave_index"] = 0
    _close(publication_evidence)
    deletion = _close(
        {
            "authorization_scope": (
                full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
            ),
            "closed_record_sha256": "",
            "evidence_closed_record_sha256": publication_evidence[
                "closed_record_sha256"
            ],
            "execution_plan_sha256": execution_plan["closed_record_sha256"],
            "lifecycle": [
                "verify_ready_shard",
                "baseline_inference",
                "vanilla_inference",
                "validate_paired_outputs",
                "commit_durable_evidence",
                "delete_ephemeral_q8_kv",
            ],
            "ready_shard_sha256": publication_evidence[
                "ready_shard_sha256"
            ],
            "record_type": full_score.FULL_SCORE_DELETION_ATTESTATION_RECORD_TYPE,
            "schema_version": full_score.FULL_SCORE_DELETION_ATTESTATION_SCHEMA_VERSION,
            "shard_id": shard["shard_id"],
            "wave_index": 0,
        }
    )
    evidence_directory = tmp_path / "publication-evidence" / shard["shard_id"]
    evidence_directory.mkdir(parents=True)
    evidence_path = evidence_directory / "evidence.json"
    deletion_path = evidence_directory / "deletion-attestation.json"
    evidence_path.write_bytes(
        full_score._canonical_pretty_json_bytes(publication_evidence)
    )
    deletion_path.write_bytes(full_score._canonical_pretty_json_bytes(deletion))
    monkeypatch.setattr(
        full_score,
        "_validate_shard_evidence_record",
        lambda *_args, **_kwargs: None,
    )
    aggregate_submit_sha256 = _digest("aggregate-final-submit")
    aggregate_reservation = DatabricksClusterHourReservation(
        attempt_id="aggregate-final-consumer",
        workload_id="aggregate-final-consumer",
        submit_payload_sha256=aggregate_submit_sha256,
        run_timeout_seconds=3_600,
        task_timeout_seconds=(3_600,),
    )
    aggregate_receipt = DatabricksRunSubmissionReceipt(
        attempt_id="aggregate-final-consumer",
        run_id="97001",
        submit_payload_sha256=aggregate_submit_sha256,
        submit_response_sha256=_digest("aggregate-final-response"),
    )
    aggregate_terminal = DatabricksClusterHourTerminalActual(
        attempt_id="aggregate-final-consumer",
        terminal_state="succeeded",
        actual_cluster_duration_seconds=1.0,
        verification_source="direct_databricks_runs_get",
        run_id="97001",
        submit_payload_sha256=aggregate_submit_sha256,
        control_plane_status_sha256=_digest("aggregate-final-control-plane"),
    )
    aggregate_ledger = DatabricksClusterHourLedger(
        ledger_id="publication-full-score",
        reservations=(aggregate_reservation,),
        submission_receipts=(aggregate_receipt,),
        terminal_actuals=(aggregate_terminal,),
    )
    aggregate_ledger_path = tmp_path / "aggregate-final-ledger.json"
    aggregate_ledger_path.write_bytes(
        full_score._canonical_pretty_json_bytes(
            databricks_cluster_hour_ledger_to_record(aggregate_ledger)
        )
    )
    aggregate_predecessor = databricks_ledger_prefix(
        DatabricksClusterHourLedger(ledger_id="publication-full-score")
    )
    aggregate_batch = full_score.databricks_ledger_prefix_at_counts(
        aggregate_ledger,
        reservation_count=1,
        submission_receipt_count=0,
        terminal_actual_count=0,
    )
    aggregate_terminal_prefix = databricks_ledger_prefix(aggregate_ledger)
    aggregate_terminal_record_sha256 = _digest("aggregate-final-terminal-record")
    aggregate_causal_closure = full_score._canonical_sha256(
        {
            "batch_prefix": aggregate_batch.to_record(),
            "ledger_path_sha256": databricks_ledger_path_sha256(
                aggregate_ledger_path
            ),
            "terminal_prefix": aggregate_terminal_prefix.to_record(),
            "terminal_record_sha256": aggregate_terminal_record_sha256,
        }
    )
    final_consumer_authorization = full_score.FullScorePhaseAuthorization(
        execution_plan_sha256=execution_plan["closed_record_sha256"],
        wave_index=0,
        phase="consumer",
        ledger_path_sha256=databricks_ledger_path_sha256(
            aggregate_ledger_path
        ),
        predecessor_prefix=aggregate_predecessor,
        ledger_prefix=aggregate_terminal_prefix,
        phase_lease_root=aggregate_ledger_path.with_name(
            f"{aggregate_ledger_path.name}.full-score-phase-leases"
        ),
        terminal_record_sha256=aggregate_terminal_record_sha256,
        causal_closure_sha256=aggregate_causal_closure,
        _issuer=full_score._FULL_SCORE_PHASE_AUTHORIZATION_ISSUER,
    )
    durable_root = "dbfs:/Volumes/catalog/schema/volume/durable"
    evidence_uri = full_score_remote._consumer_evidence_artifact_uri(
        durable_root,
        wave_index=0,
        shard_id=shard["shard_id"],
        filename="evidence.json",
    )
    deletion_uri = full_score_remote._consumer_evidence_artifact_uri(
        durable_root,
        wave_index=0,
        shard_id=shard["shard_id"],
        filename="deletion-attestation.json",
    )
    evidence_binding = {
        "deletion_file_sha256": sha256(deletion_path.read_bytes()).hexdigest(),
        "deletion_record_sha256": deletion["closed_record_sha256"],
        "deletion_uri": deletion_uri,
        "evidence_file_sha256": sha256(evidence_path.read_bytes()).hexdigest(),
        "evidence_record_sha256": publication_evidence["closed_record_sha256"],
        "evidence_uri": evidence_uri,
        "shard_id": shard["shard_id"],
    }
    remote_authorization = SimpleNamespace(
        controller_authorization_record_sha256=_digest("remote-authorization"),
        coordinator_run_id="97101",
        evidence_bindings=(evidence_binding,),
        execution_plan_sha256=execution_plan["closed_record_sha256"],
        phase_terminal_record_sha256=aggregate_terminal_record_sha256,
        runs_get_receipt_record_sha256=_digest("remote-runs-get-receipt"),
        wave_index=0,
    )

    def require_remote_authorizations(authorizations, **_kwargs):
        assert list(authorizations) == [remote_authorization]
        return (remote_authorization,)

    monkeypatch.setattr(
        full_score_remote,
        "require_full_score_remote_consumer_evidence_authorizations",
        require_remote_authorizations,
    )
    compact_files = {evidence_uri: evidence_path, deletion_uri: deletion_path}
    monkeypatch.setattr(
        full_score,
        "_cluster_path",
        lambda *_args, **_kwargs: pytest.fail(
            "Mac publication aggregation must not resolve /dbfs"
        ),
    )
    publication_aggregate = full_score.aggregate_full_score_shard_evidence(
        inventory,
        plan,
        [],
        authorization_scope=(
            full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
        ),
        execution_plan=execution_plan,
        final_consumer_authorization=final_consumer_authorization,
        ledger_path=aggregate_ledger_path,
        remote_consumer_authorizations=[remote_authorization],
        compact_artifact_resolver=lambda uri: compact_files[uri],
    )
    aggregate_lineage = publication_aggregate["publication_lineage"]
    assert aggregate_lineage["ledger_path_sha256"] == (
        databricks_ledger_path_sha256(aggregate_ledger_path)
    )
    assert aggregate_lineage["predecessor_prefix"] == (
        aggregate_predecessor.to_record()
    )
    assert aggregate_lineage["batch_prefix"] == aggregate_batch.to_record()
    assert aggregate_lineage["terminal_prefix"] == (
        aggregate_terminal_prefix.to_record()
    )
    assert aggregate_lineage["terminal_record_sha256"] == (
        aggregate_terminal_record_sha256
    )
    assert aggregate_lineage["evidence"] == [
        {
            **evidence_binding,
            "authorization_record_sha256": (
                remote_authorization.controller_authorization_record_sha256
            ),
            "wave_index": 0,
        }
    ]
    assert aggregate_lineage["remote_consumer_authorizations"] == [
        {
            "authorization_record_sha256": (
                remote_authorization.controller_authorization_record_sha256
            ),
            "coordinator_run_id": remote_authorization.coordinator_run_id,
            "execution_plan_sha256": execution_plan["closed_record_sha256"],
            "phase_terminal_record_sha256": aggregate_terminal_record_sha256,
            "runs_get_receipt_record_sha256": (
                remote_authorization.runs_get_receipt_record_sha256
            ),
            "wave_index": 0,
        }
    ]
    assert publication_aggregate["scorers"] == full_score._scorer_contract_record()
    full_score.validate_full_score_aggregate_record(
        publication_aggregate,
        inventory=inventory,
        shard_plan=plan,
        execution_plan=execution_plan,
        require_publication=True,
    )
    remote_plan_drift = copy.deepcopy(publication_aggregate)
    remote_plan_drift["publication_lineage"][
        "remote_consumer_authorizations"
    ][0]["execution_plan_sha256"] = _digest("remote-plan-drift")
    _close(remote_plan_drift)
    with pytest.raises(ValueError, match="remote execution-plan authority drift"):
        full_score.validate_full_score_aggregate_record(
            remote_plan_drift,
            inventory=inventory,
            shard_plan=plan,
            execution_plan=execution_plan,
            require_publication=True,
        )
    original_evidence_bytes = evidence_path.read_bytes()
    evidence_path.write_bytes(original_evidence_bytes + b" ")
    with pytest.raises(ValueError, match="CAS evidence binding drift"):
        full_score.aggregate_full_score_shard_evidence(
            inventory,
            plan,
            [],
            authorization_scope=(
                full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
            ),
            execution_plan=execution_plan,
            final_consumer_authorization=final_consumer_authorization,
            ledger_path=aggregate_ledger_path,
            remote_consumer_authorizations=[remote_authorization],
            compact_artifact_resolver=lambda uri: compact_files[uri],
        )
    evidence_path.write_bytes(original_evidence_bytes)

    def missing_deletion(uri):
        if uri == deletion_uri:
            raise ValueError("compact artifact URI is not bound in the CAS")
        return compact_files[uri]

    with pytest.raises(ValueError, match="not bound in the CAS"):
        full_score.aggregate_full_score_shard_evidence(
            inventory,
            plan,
            [],
            authorization_scope=(
                full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
            ),
            execution_plan=execution_plan,
            final_consumer_authorization=final_consumer_authorization,
            ledger_path=aggregate_ledger_path,
            remote_consumer_authorizations=[remote_authorization],
            compact_artifact_resolver=missing_deletion,
        )
    aggregate = full_score.aggregate_full_score_shard_evidence(
        inventory,
        plan,
        [evidence],
        authorization_scope=full_score.FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE,
    )
    assert set(aggregate) == {
        "aggregation_unit",
        "authorization_scope",
        "bootstrap",
        "closed_record_sha256",
        "datasets",
        "identity_count",
        "inventory_sha256",
        "methods",
        "niah_grid",
        "passes_per_method",
        "protocol",
        "record_type",
        "schema_version",
        "scorers",
        "shard_count",
        "shard_plan_sha256",
    }
    assert aggregate["scorers"] == full_score._scorer_contract_record()
    assert aggregate["identity_count"] == 1003
    assert aggregate["datasets"]["niah"]["example_count"] == 1000
    assert set(aggregate["niah_grid"]) == set(NIAH_CELL_IDS)
    expected_statuses = {
        "ok",
        "missing_block",
        "multiple_or_malformed_blocks",
        "extraneous_text",
        "nested_block",
        "empty_answer",
    }
    for dataset, dataset_record in aggregate["datasets"].items():
        for method_record in dataset_record["methods"].values():
            counts_by_status = method_record["parser_status_counts"]
            assert set(counts_by_status) == expected_statuses
            assert counts_by_status["ok"] == counts[dataset]
            assert sum(counts_by_status.values()) == counts[dataset]
            assert all(
                summary["invalid_parser_score_sum"] == 0.0
                for summary in method_record["metrics"].values()
            )
    for dataset, delta in deltas.items():
        summaries = aggregate["datasets"][dataset][
            "paired_vanilla_minus_baseline"
        ]
        assert {round(value["mean"], 7) for value in summaries.values()} == {
            round(delta, 7)
        }
        assert all(value["bootstrap_ci95"]["draws"] == 25 for value in summaries.values())
    full_score.validate_full_score_aggregate_record(
        aggregate,
        inventory=inventory,
        shard_plan=plan,
    )
    aggregate_niah_redistribution = copy.deepcopy(aggregate)
    aggregate_niah_redistribution["niah_grid"][NIAH_CELL_IDS[0]][
        "example_count"
    ] -= 1
    aggregate_niah_redistribution["niah_grid"][NIAH_CELL_IDS[1]][
        "example_count"
    ] += 1
    _close(aggregate_niah_redistribution)
    with pytest.raises(ValueError, match="cell count distribution drift"):
        full_score.validate_full_score_aggregate_record(
            aggregate_niah_redistribution,
            inventory=inventory,
            shard_plan=plan,
        )
    with pytest.raises(ValueError, match="rejects local_fixture_only"):
        full_score.validate_full_score_aggregate_record(
            aggregate,
            inventory=inventory,
            shard_plan=plan,
            require_publication=True,
        )

    aggregate_scorer_drift = copy.deepcopy(aggregate)
    aggregate_scorer_drift["scorers"][0]["answer_parser_digest"] = _digest(
        "aggregate-parser-drift"
    )
    _close(aggregate_scorer_drift)
    with pytest.raises(ValueError, match="scorer/parser contract drift"):
        full_score.validate_full_score_aggregate_record(
            aggregate_scorer_drift,
            inventory=inventory,
            shard_plan=plan,
        )

    aggregate_scorer_type_drift = copy.deepcopy(aggregate)
    aggregate_scorer_type_drift["scorers"][0]["publication_approved"] = 1
    _close(aggregate_scorer_type_drift)
    with pytest.raises(ValueError, match="scorer/parser contract drift"):
        full_score.validate_full_score_aggregate_record(
            aggregate_scorer_type_drift,
            inventory=inventory,
            shard_plan=plan,
        )

    aggregate_parser_count_drift = copy.deepcopy(aggregate)
    aggregate_parser_count_drift["datasets"]["biography"]["methods"][
        "baseline_prefill"
    ]["parser_status_counts"]["ok"] = 0
    _close(aggregate_parser_count_drift)
    with pytest.raises(ValueError, match="parser-status coverage drift"):
        full_score.validate_full_score_aggregate_record(
            aggregate_parser_count_drift,
            inventory=inventory,
            shard_plan=plan,
        )

    aggregate_invalid_score_credit = copy.deepcopy(aggregate)
    invalid_credit_counts = aggregate_invalid_score_credit["datasets"][
        "biography"
    ]["methods"]["baseline_prefill"]["parser_status_counts"]
    invalid_credit_counts["ok"] = 0
    invalid_credit_counts["missing_block"] = 1
    _close(aggregate_invalid_score_credit)
    with pytest.raises(ValueError, match="credits invalid parsed answers"):
        full_score.validate_full_score_aggregate_record(
            aggregate_invalid_score_credit,
            inventory=inventory,
            shard_plan=plan,
        )

    aggregate_metric_drift = copy.deepcopy(aggregate)
    biography_metrics = aggregate_metric_drift["datasets"]["biography"][
        "methods"
    ]["baseline_prefill"]["metrics"]
    first_biography_metric = next(iter(biography_metrics.values()))
    first_biography_metric["sum"] = 0.5
    _close(aggregate_metric_drift)
    with pytest.raises(ValueError, match="metric mean/sum identity drift"):
        full_score.validate_full_score_aggregate_record(
            aggregate_metric_drift,
            inventory=inventory,
            shard_plan=plan,
        )

    aggregate_invalid_parser_sum = copy.deepcopy(aggregate)
    first_invalid_parser_summary = next(
        iter(
            aggregate_invalid_parser_sum["datasets"]["biography"]["methods"][
                "baseline_prefill"
            ]["metrics"].values()
        )
    )
    first_invalid_parser_summary["invalid_parser_score_sum"] = 0.25
    _close(aggregate_invalid_parser_sum)
    with pytest.raises(ValueError, match="credits an invalid parsed answer"):
        full_score.validate_full_score_aggregate_record(
            aggregate_invalid_parser_sum,
            inventory=inventory,
            shard_plan=plan,
        )

    aggregate_protocol_drift = copy.deepcopy(aggregate)
    aggregate_protocol_drift["protocol"]["temperature"] = 0.5
    _close(aggregate_protocol_drift)
    with pytest.raises(ValueError, match="protocol contract drift"):
        full_score.validate_full_score_aggregate_record(
            aggregate_protocol_drift,
            inventory=inventory,
            shard_plan=plan,
        )

    aggregate_paired_mean_drift = copy.deepcopy(aggregate)
    first_paired_summary = next(
        iter(
            aggregate_paired_mean_drift["datasets"]["biography"][
                "paired_vanilla_minus_baseline"
            ].values()
        )
    )
    first_paired_summary["mean"] = 0.0
    _close(aggregate_paired_mean_drift)
    with pytest.raises(ValueError, match="paired mean identity drift"):
        full_score.validate_full_score_aggregate_record(
            aggregate_paired_mean_drift,
            inventory=inventory,
            shard_plan=plan,
        )

    aggregate_seed_drift = copy.deepcopy(aggregate)
    first_biography_delta = next(
        iter(
            aggregate_seed_drift["datasets"]["biography"][
                "paired_vanilla_minus_baseline"
            ].values()
        )
    )
    first_biography_delta["seed_sha256"] = _digest("aggregate-seed-drift")
    _close(aggregate_seed_drift)
    with pytest.raises(ValueError, match="deterministic CI identity drift"):
        full_score.validate_full_score_aggregate_record(
            aggregate_seed_drift,
            inventory=inventory,
            shard_plan=plan,
        )

    aggregate_niah_coverage_drift = copy.deepcopy(aggregate)
    del aggregate_niah_coverage_drift["niah_grid"][NIAH_CELL_IDS[-1]]
    _close(aggregate_niah_coverage_drift)
    with pytest.raises(ValueError, match="exact nine-cell NIAH grid"):
        full_score.validate_full_score_aggregate_record(
            aggregate_niah_coverage_drift,
            inventory=inventory,
            shard_plan=plan,
        )

    scorer_drift = copy.deepcopy(evidence)
    scorer_drift["paired_examples"][0]["methods"]["baseline_prefill"][
        "scorer_id"
    ] = "unreviewed-scorer"
    _close(scorer_drift)
    with pytest.raises(ValueError, match="scorer identity drift"):
        full_score.aggregate_full_score_shard_evidence(
            inventory,
            plan,
            [scorer_drift],
            authorization_scope=(
                full_score.FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE
            ),
        )

    invalid_parser_score = copy.deepcopy(evidence)
    invalid_pair = next(
        pair
        for pair in invalid_parser_score["paired_examples"]
        if pair["dataset"] == "biography"
    )
    invalid_method = invalid_pair["methods"]["baseline_prefill"]
    invalid_method["parser_status"] = "missing_block"
    invalid_method["parser_valid"] = False
    _close(invalid_parser_score)
    with pytest.raises(ValueError, match="invalid parsed answers must receive zero"):
        full_score.aggregate_full_score_shard_evidence(
            inventory,
            plan,
            [invalid_parser_score],
            authorization_scope=(
                full_score.FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE
            ),
        )

    invalid_status = copy.deepcopy(evidence)
    invalid_status["paired_examples"][0]["methods"]["baseline_prefill"][
        "parser_status"
    ] = "unreviewed_status"
    _close(invalid_status)
    with pytest.raises(ValueError, match="outside the frozen states"):
        full_score.aggregate_full_score_shard_evidence(
            inventory,
            plan,
            [invalid_status],
            authorization_scope=(
                full_score.FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE
            ),
        )

    inconsistent_status = copy.deepcopy(evidence)
    inconsistent_status["paired_examples"][0]["methods"]["baseline_prefill"][
        "parser_status"
    ] = "empty_answer"
    _close(inconsistent_status)
    with pytest.raises(ValueError, match="validity/status are inconsistent"):
        full_score.aggregate_full_score_shard_evidence(
            inventory,
            plan,
            [inconsistent_status],
            authorization_scope=(
                full_score.FULL_SCORE_LOCAL_FIXTURE_AUTHORIZATION_SCOPE
            ),
        )
