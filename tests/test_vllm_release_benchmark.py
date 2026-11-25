"""CPU end-to-end check of the vLLM release benchmark pipeline.

The plan runs for real (dataset prep, handoff generation, enrichment, one runner
per arm, merge); only vLLM is replaced by a stub OpenAI-compatible server that
writes provider-shaped load telemetry for cache requests.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from document_kv_cache import vllm_smoke
from document_kv_cache._benchmark_cli import merge_isolated_main
from document_kv_cache.artifact_identity import ArtifactIdentity
from document_kv_cache.benchmark_plan import ISOLATED_ARM_COMMAND_PREFIX, ISOLATED_ARM_MERGE_COMMAND
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    CACHE_REUSE_ARM,
    DOCUMENT_KV_ARTIFACT_ID_PARAM,
    DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM,
    DOCUMENT_KV_CACHE_METHOD_PARAM,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    SUPPORTED_V1_DATASETS,
)
from document_kv_cache.release_bundle import _plan_execution_sidecar_issues
from document_kv_cache.release_evidence import _benchmark_evidence_gate_issues, _v1_benchmark_issues
from document_kv_cache.vllm_smoke import (
    VLLMReleaseBenchmarkConfig,
    VLLMSmokeBenchmarkConfig,
    build_release_benchmark_plan_config,
    build_vllm_server_args,
    parse_args,
    run_vllm_release_benchmark,
)


EXAMPLES_PER_DATASET = 4

CHUNK_TOKEN_GENERATOR_MODULE = '''
import hashlib
import os

from document_kv_cache.artifact_identity import TokenContract
from document_kv_cache.kvpack import PackChunk
from document_kv_cache.model_profiles import QWEN3_4B_ROPE_ROTARY_DIM, QWEN3_4B_ROPE_THETA
from document_kv_cache.models import KVCacheKey


class ChunkTokenPreRopeGenerator:
    """One token per chunk keeps real Qwen3-sized payloads small."""

    pre_rope = True
    rope_theta = QWEN3_4B_ROPE_THETA
    rope_rotary_dim = QWEN3_4B_ROPE_ROTARY_DIM
    layout = None

    def __init__(self):
        self.model_revision = os.environ.get("CACHET_TRANSFORMERS_MODEL_REVISION", "unresolved")
        self.tokenizer_revision = os.environ.get("CACHET_TRANSFORMERS_TOKENIZER_REVISION", "unresolved")

    def bind_layout(self, layout):
        self.layout = layout

    def generate(self, *, document, chunk, config, training_artifacts=None):
        token_ids = (len(chunk.text),)
        payload = bytes(len(token_ids) * self.layout.bytes_per_token)
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
                content_hash=hashlib.sha256(payload).hexdigest(),
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


def create_generator():
    return ChunkTokenPreRopeGenerator()
'''


class StubVLLMServer:
    """Serve streamed completions and record one cold load per cache request."""

    def __init__(self, port: int, telemetry_jsonl: Path | None, *, attest: bool = True) -> None:
        self.telemetry_jsonl = telemetry_jsonl
        self.attest = attest
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, b"")
                elif self.path == "/v1/models":
                    self._send(200, json.dumps({"data": [{"id": vllm_smoke.SERVED_MODEL_NAME}]}).encode())
                else:
                    self._send(404, b"")

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(body)
                params = body.get("kv_transfer_params") or {}
                time.sleep(0.004 if not params else 0.001)
                if params and server.telemetry_jsonl is not None:
                    server.record_load(body, params)
                chunks = [
                    {"choices": [{"index": 0, "text": "Final answer: unknown"}]},
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": len(body["prompt"].split()),
                            "completion_tokens": 3,
                        },
                    },
                ]
                payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                self._send(200, payload.encode(), content_type="text/event-stream")

            def _send(self, status, payload, *, content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        ThreadingHTTPServer.allow_reuse_address = True
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def record_load(self, body: dict, params: dict) -> None:
        assert self.telemetry_jsonl is not None
        handoff = json.loads(Path(params[DOCUMENT_KV_HANDOFF_JSON_PARAM]).read_text(encoding="utf-8"))
        artifact_id = params.get(DOCUMENT_KV_ARTIFACT_ID_PARAM) or ArtifactIdentity.from_record(
            handoff["handle"]["artifact_identity"]
        ).artifact_id
        tokens = sum(segment["token_count"] for segment in handoff["handle"]["segments"])
        record = {
            "record_type": "document_kv.vllm_native_provider_load.v1",
            "request_id": f"cmpl-{body['request_id']}-0",
            "benchmark_request_id": params[DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM],
            "cache_state_attestation": {
                "cache_method": params[DOCUMENT_KV_CACHE_METHOD_PARAM],
                "artifact_id": artifact_id,
                "source": "local_path",
                "bytes_read": tokens * 8,
                "payload_cache_hit": False,
                "eviction_requested": True,
                "eviction_succeeded": True,
                "direct_io": False,
                "expected_bytes": tokens * 8,
                "expected_tokens": tokens,
                "loaded_tokens": tokens,
                "successful_loads": 1,
            },
        }
        with self.telemetry_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"record_type": "document_kv_prefetch"}) + "\n")
            if self.attest:
                handle.write(json.dumps(record) + "\n")

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=10)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _write_raw_datasets(root: Path) -> tuple[str, ...]:
    specs = []
    root.mkdir(parents=True, exist_ok=True)
    for dataset in SUPPORTED_V1_DATASETS:
        path = root / f"{dataset}.jsonl"
        rows = (
            {
                "example_id": f"{dataset}-{index}",
                "query": f"Which city hosts landmark {index}?",
                "expected_answer": f"City{index}",
                "documents": [
                    {
                        "document_id": f"{dataset}-{index}-a",
                        "title": f"Landmark {index}",
                        "text": f"Landmark {index} stands in City{index} near the river.",
                    },
                    {
                        "document_id": f"{dataset}-{index}-b",
                        "title": f"River {index}",
                        "text": f"The river {index} flows past old bridges.",
                    },
                ],
            }
            for index in range(EXAMPLES_PER_DATASET)
        )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        specs.append(f"{dataset}={path}")
    return tuple(specs)


def _release_config(tmp_path: Path, *, port: int = 8000) -> VLLMSmokeBenchmarkConfig:
    return VLLMSmokeBenchmarkConfig(
        benchmark_id="release-e2e",
        output_dir=tmp_path / "output",
        local_root=tmp_path / "local",
        server_port=port,
        max_model_len=vllm_smoke.RELEASE_MAX_MODEL_LEN,
        release_benchmark=VLLMReleaseBenchmarkConfig(
            raw_dataset_specs=_write_raw_datasets(tmp_path / "raw"),
            generator_factory="cachet_release_chunk_generator:create_generator",
        ),
    )


def _metadata() -> dict[str, object]:
    return {
        "vllm_version_installed": "0.30.0",
        "transformers_version_installed": "5.24.0",
        "document_kv_cache_version_installed": "0.3.0",
        "torch_version_installed": "2.12.1",
    }


def _patch_release_runtime(monkeypatch, root: Path, *, attest: bool = True):
    (root / "cachet_release_chunk_generator.py").write_text(CHUNK_TOKEN_GENERATOR_MODULE, encoding="utf-8")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH")))),
    )
    config = _release_config(root, port=_free_port())
    config.venv_python.parent.mkdir(parents=True)
    config.venv_python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    config.venv_python.chmod(0o755)
    servers: list[tuple[str, StubVLLMServer]] = []
    budget_checks: list[dict[str, Path]] = []

    def start_stub(config, python_executable, log_path, *, telemetry_jsonl=None):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(" ".join(build_vllm_server_args(config, python_executable)), encoding="utf-8")
        server = StubVLLMServer(config.server_port, telemetry_jsonl, attest=attest)
        servers.append((log_path.name, server))
        return server

    monkeypatch.setattr(vllm_smoke, "start_vllm_server", start_stub)
    monkeypatch.setattr(vllm_smoke, "wait_for_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(vllm_smoke, "terminate_process", lambda server: server.stop())
    monkeypatch.setattr(
        vllm_smoke,
        "validate_prompt_token_budget",
        lambda config, dataset_paths: budget_checks.append(dataset_paths),
    )
    monkeypatch.setattr(vllm_smoke, "gpu_hardware_fingerprint", lambda: "nvidia:NVIDIA L4, 23034 MiB, 580.00")
    return config, servers, budget_checks


@pytest.fixture(scope="module")
def release_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("release-run")
    with pytest.MonkeyPatch.context() as monkeypatch:
        config, servers, budget_checks = _patch_release_runtime(monkeypatch, root)
        metadata = _metadata()
        results = run_vllm_release_benchmark(config, metadata)
    return SimpleNamespace(
        config=config,
        servers=servers,
        budget_checks=budget_checks,
        metadata=metadata,
        results=results,
    )


def test_release_benchmark_plan_publishes_a_cold_latency_gate_end_to_end(release_run):
    config = release_run.config
    results = release_run.results
    servers = release_run.servers
    budget_checks = release_run.budget_checks
    metadata = release_run.metadata

    assert [result.name for result in results][-3:] == [
        f"{ISOLATED_ARM_COMMAND_PREFIX}{BASELINE_PREFILL_ARM}",
        f"{ISOLATED_ARM_COMMAND_PREFIX}{CACHE_REUSE_ARM}",
        ISOLATED_ARM_MERGE_COMMAND,
    ]
    assert all(result.ok for result in results)
    assert [name for name, _server in servers] == [
        f"vllm-server-{BASELINE_PREFILL_ARM}.log",
        f"vllm-server-{CACHE_REUSE_ARM}.log",
    ]
    assert len(budget_checks) == 1
    baseline_requests = servers[0][1].requests
    cache_requests = servers[1][1].requests
    assert baseline_requests and not any("kv_transfer_params" in body for body in baseline_requests)
    assert cache_requests and all("kv_transfer_params" in body for body in cache_requests)

    merged = json.loads(config.benchmark_output_path.read_text(encoding="utf-8"))
    gate = json.loads(config.release_evidence_gate_path.read_text(encoding="utf-8"))
    measured_cache_requests = len(SUPPORTED_V1_DATASETS) * EXAMPLES_PER_DATASET
    assert gate == merged["evidence_gate"]
    assert gate["policy"] == "publication"
    assert gate["issues"] == []
    assert gate["ok"] is True
    assert gate["measurement_scopes"] == ["latency"]
    assert gate["checked_cache_requests"] == measured_cache_requests
    assert gate["cold_attested_requests"] == measured_cache_requests
    execution = merged["experiment_manifest"]["execution"]
    assert execution["isolation_mode"] == "separate_process_or_job"
    assert {item["arm_id"] for item in execution["source_execution_ids"]} == {
        BASELINE_PREFILL_ARM,
        CACHE_REUSE_ARM,
    }
    assert merged["experiment_manifest"]["has_unresolved_provenance"] is False
    assert _v1_benchmark_issues(merged) == ()
    assert _benchmark_evidence_gate_issues(merged, gate, method_registry=None) == ()

    plan_execution = json.loads(config.release_plan_execution_path.read_text(encoding="utf-8"))
    assert plan_execution["ok"] is True
    assert plan_execution["plan_source"]["benchmark_handoff_generation_datasets"] == list(SUPPORTED_V1_DATASETS)
    assert _plan_execution_sidecar_issues(
        plan_execution,
        strict_v1_target=True,
        expected_suite_id=merged["suite"]["suite_id"],
        expected_hardware_target=config.hardware_target,
    ) == ()
    assert metadata["release_plan_execution_path"] == str(config.release_plan_execution_path)
    assert (config.output_dir / f"vllm-telemetry-{CACHE_REUSE_ARM}.jsonl").exists()


def test_merge_isolated_cli_rejects_release_records_without_cold_attestations(release_run, tmp_path):
    config = release_run.config
    merge_argv = next(
        command["argv"]
        for command in json.loads(config.release_plan_path.read_text(encoding="utf-8"))["commands"]
        if command["name"] == ISOLATED_ARM_MERGE_COMMAND
    )
    arguments = list(merge_argv[merge_argv.index("merge-isolated") + 1 :])
    attestation_index = arguments.index("--cache-state-attestation-json")
    del arguments[attestation_index : attestation_index + 2]
    output_index = arguments.index("--output-json")
    arguments[output_index + 1] = str(tmp_path / "merged.json")
    gate_index = arguments.index("--evidence-gate-output-json")
    arguments[gate_index + 1] = str(tmp_path / "gate.json")

    assert merge_isolated_main(arguments) == 0

    gate = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert gate["ok"] is False
    assert gate["cold_attested_requests"] == 0
    assert any("has no cache-state attestation" in issue for issue in gate["issues"])


def test_merge_isolated_cli_sanitizes_the_merged_record_on_request(release_run, tmp_path):
    output_dir = release_run.config.output_dir
    output_json = tmp_path / "sanitized.json"

    exit_code = merge_isolated_main(
        [
            "--record-json",
            str(output_dir / f"v1-benchmark.{BASELINE_PREFILL_ARM}.json"),
            "--record-json",
            str(output_dir / f"v1-benchmark.{CACHE_REUSE_ARM}.json"),
            "--reference-arm-id",
            BASELINE_PREFILL_ARM,
            "--sanitize-evidence",
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    sanitized = json.loads(output_json.read_text(encoding="utf-8"))
    full = json.loads(release_run.config.benchmark_output_path.read_text(encoding="utf-8"))
    assert sanitized["evidence_sanitized"] is True
    assert "evidence_sanitized" not in full
    assert {measurement["expected_answer"] for measurement in sanitized["measurements"]} == {None}
    assert all(measurement["expected_answer"] for measurement in full["measurements"])


def test_merge_isolated_cli_reports_records_from_one_execution_as_an_error(release_run, tmp_path, capsys):
    baseline_record = release_run.config.output_dir / f"v1-benchmark.{BASELINE_PREFILL_ARM}.json"
    output_json = tmp_path / "merged.json"
    assert baseline_record.exists()

    exit_code = merge_isolated_main(
        [
            "--record-json",
            str(baseline_record),
            "--record-json",
            str(baseline_record),
            "--reference-arm-id",
            BASELINE_PREFILL_ARM,
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 1
    error = json.loads(capsys.readouterr().out)
    assert error["ok"] is False
    assert "duplicate arm ids" in error["error"]
    assert not output_json.exists()


def test_release_benchmark_fails_the_run_when_the_gate_fails(tmp_path, monkeypatch):
    config, _servers, _budget_checks = _patch_release_runtime(monkeypatch, tmp_path, attest=False)

    with pytest.raises(RuntimeError, match="release benchmark evidence gate failed"):
        run_vllm_release_benchmark(config, _metadata())

    plan_execution = json.loads(config.release_plan_execution_path.read_text(encoding="utf-8"))
    assert plan_execution["ok"] is True


def test_release_plan_config_matches_generated_vanilla_artifact_contract(tmp_path):
    config = _release_config(tmp_path)

    plan_config = build_release_benchmark_plan_config(
        config,
        engine_version="0.30.0",
        generator_version="5.24.0",
        package_revisions=("vllm=0.30.0",),
        hardware_fingerprint="nvidia:NVIDIA L4",
        runtime_id="release-e2e:run",
        runtime_version="databricks-runtime-16.4",
        serving_platform="databricks",
    )

    assert plan_config.evidence_policy == "publication"
    assert plan_config.measurement_scopes == ("latency",)
    assert plan_config.cache_state == "cold"
    assert plan_config.prefix_cache_salt_mode == "per_request"
    assert plan_config.isolated_arm_jobs is True
    assert plan_config.benchmark_handoff_cache_method == "vanilla_prefill"
    assert plan_config.benchmark_handoff_segment_per_document is True
    assert plan_config.key_position_encoding == "pre_rope"
    assert plan_config.model_revision == vllm_smoke.QWEN3_4B_INSTRUCT_HF_REVISION
    assert plan_config.tokenizer_revision == plan_config.model_revision
    assert plan_config.cache_state_attestation_jsons == (
        str(config.release_telemetry_path(CACHE_REUSE_ARM)),
    )
    assert all(
        Path(path.handoff_manifest_json).parent == config.release_handoff_dir
        for path in plan_config.dataset_paths
    )


def test_release_server_args_pin_revision_and_block_size(tmp_path):
    config = _release_config(tmp_path)

    args = build_vllm_server_args(
        config,
        config.venv_python,
        telemetry_jsonl=config.release_telemetry_path(CACHE_REUSE_ARM),
    )

    assert args[args.index("--revision") + 1] == vllm_smoke.QWEN3_4B_INSTRUCT_HF_REVISION
    assert args[args.index("--tokenizer-revision") + 1] == vllm_smoke.QWEN3_4B_INSTRUCT_HF_REVISION
    assert args[args.index("--block-size") + 1] == "16"
    assert args[args.index("--max-model-len") + 1] == str(vllm_smoke.RELEASE_MAX_MODEL_LEN)
    transfer_config = json.loads(args[args.index("--kv-transfer-config") + 1])
    assert str(config.release_telemetry_path(CACHE_REUSE_ARM)) in json.dumps(transfer_config)


def test_release_server_evicts_page_cache(tmp_path, monkeypatch):
    config = _release_config(tmp_path)
    launched = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            launched.update(kwargs)

    monkeypatch.setattr(vllm_smoke.subprocess, "Popen", FakePopen)

    vllm_smoke.start_vllm_server(config, config.venv_python, tmp_path / "server.log")

    assert launched["env"][vllm_smoke.DOCUMENT_KV_EVICT_PAGE_CACHE_ENV] == "1"


def test_parse_args_builds_release_benchmark_with_long_context(tmp_path):
    specs = _write_raw_datasets(tmp_path / "raw")

    config = parse_args(
        [
            "--benchmark-id",
            "release",
            "--output-dir",
            str(tmp_path / "output"),
            *(argument for spec in specs for argument in ("--release-raw-dataset", spec)),
            "--release-limit-per-dataset",
            "54",
        ]
    )

    assert config.release_benchmark is not None
    assert config.release_benchmark.limit_per_dataset == 54
    assert config.max_model_len == vllm_smoke.RELEASE_MAX_MODEL_LEN
    assert parse_args(["--benchmark-id", "smoke", "--output-dir", str(tmp_path)]).max_model_len == (
        vllm_smoke.DEFAULT_MAX_MODEL_LEN
    )


@pytest.mark.parametrize(
    ("option", "message"),
    (
        ("dataset_specs", "do not combine"),
        ("payload_cache_max_bytes", "payload cache to be disabled"),
    ),
)
def test_release_benchmark_rejects_conflicting_smoke_options(tmp_path, option, message):
    specs = _write_raw_datasets(tmp_path / "raw")
    overrides = {"dataset_specs": specs} if option == "dataset_specs" else {option: 1024}

    with pytest.raises(ValueError, match=message):
        VLLMSmokeBenchmarkConfig(
            benchmark_id="release",
            output_dir=tmp_path / "output",
            release_benchmark=VLLMReleaseBenchmarkConfig(raw_dataset_specs=specs),
            **overrides,
        )
