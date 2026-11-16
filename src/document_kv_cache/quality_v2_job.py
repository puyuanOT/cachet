"""Run one quality-v2 chunk end to end on a single-GPU Databricks node.

A chunk is a slice of one dataset.  The job generates the per-document Vanilla
KV for every example to local disk with the campaign's Transformers producer,
then boots the campaign's vLLM 0.30.0 runtime once and runs full prefill and the
cached arm over the same examples, and finally copies the raw benchmark records
and logs to the chunk's output directory.  The runner script that bootstraps the
locked runtime is the full-score runner with only its entry module renamed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from document_kv_cache.artifact_identity import RuntimeIdentity
from document_kv_cache.benchmarks import BASELINE_PREFILL_ARM
from document_kv_cache.full_score_execution import (
    FULL_SCORE_ATTENTION_BACKEND,
    FULL_SCORE_CONSUMER_NODE_TYPE_ID,
    FULL_SCORE_GENERATOR_FACTORY,
    FULL_SCORE_GENERATOR_QUANTIZATION,
    FULL_SCORE_GENERATOR_QUANTIZATION_CONFIG,
    FULL_SCORE_GENERATOR_VERSION,
    FULL_SCORE_KV_DTYPE,
    FULL_SCORE_MODEL_DTYPE,
    FULL_SCORE_MODEL_ID,
    FULL_SCORE_MODEL_QUANTIZATION,
    FULL_SCORE_REQUEST_PARALLELISM,
    FULL_SCORE_RUNNER_SCRIPT,
    FULL_SCORE_SERVED_MODEL_NAME,
    FULL_SCORE_VANILLA_ARM_ID,
    _full_score_handoff_layout,
    _vanilla_arm_spec,
)
from document_kv_cache.main_latency_inputs import MAIN_LATENCY_TOKENIZER_REVISION
from document_kv_cache.engine_protocol import KVKeyPositionEncoding, KVPayloadAxisOrder
from document_kv_cache.publication_campaign import PUBLICATION_CAMPAIGN_ENGINE_VERSION
from document_kv_cache.quality_v2 import (
    QUALITY_V2_MAX_TOKENS,
    QUALITY_V2_PROMPT_TEMPLATE_VERSION,
    QUALITY_V2_PROTOCOL_ID,
    quality_v2_protocol_record,
    quality_v2_scorer_registry,
)
from document_kv_cache.serving_env import gpu_runtime_warning_environment_overrides
from document_kv_cache.transformers_generator import (
    CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV,
    CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV,
    CACHET_TRANSFORMERS_DEVICE_ENV,
    CACHET_TRANSFORMERS_DEVICE_MAP_ENV,
    CACHET_TRANSFORMERS_MODEL_ID_ENV,
    CACHET_TRANSFORMERS_MODEL_REVISION_ENV,
    CACHET_TRANSFORMERS_PRE_ROPE_ENV,
    CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV,
    CACHET_TRANSFORMERS_QUANTIZATION_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_ID_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV,
    CACHET_TRANSFORMERS_TORCH_DTYPE_ENV,
    CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV,
)
from vllm_kv_injection.vllm_transfer_config import document_kv_transfer_config

__all__ = [
    "QUALITY_V2_CHUNK_RECORD_TYPE",
    "QUALITY_V2_GPU_MEMORY_UTILIZATION",
    "QUALITY_V2_MAX_MODEL_LEN",
    "QUALITY_V2_RUNNER_SCRIPT",
    "QUALITY_V2_RUNNER_SHA256",
    "QUALITY_V2_TASK_TIMEOUT_SECONDS",
    "main",
    "quality_v2_submit_payload",
    "quality_v2_vllm_server_command",
]

QUALITY_V2_CHUNK_RECORD_TYPE = "cachet.quality_v2_chunk_result.v1"
QUALITY_V2_MAX_MODEL_LEN = 32_768 + QUALITY_V2_MAX_TOKENS
QUALITY_V2_GPU_MEMORY_UTILIZATION = 0.80
QUALITY_V2_TASK_TIMEOUT_SECONDS = 4 * 3600
QUALITY_V2_SPARK_VERSION = "15.4.x-gpu-ml-scala2.12"
QUALITY_V2_RUNTIME_VENV_DIR = "/local_disk0/cachet-quality-v2-runtime"
QUALITY_V2_PRODUCER_SUB_BATCH_TOKENS = 400_000
_SERVER_HOST = "127.0.0.1"
_SERVER_PORT = 8000
_SERVER_START_TIMEOUT_SECONDS = 900.0
_REQUEST_TIMEOUT_SECONDS = 900.0

_RUNNER_MODULE = "document_kv_cache.full_score_execution"
if FULL_SCORE_RUNNER_SCRIPT.count(_RUNNER_MODULE) != 2:
    raise RuntimeError("full-score runner entry points changed; re-derive the v2 runner")
QUALITY_V2_RUNNER_SCRIPT = FULL_SCORE_RUNNER_SCRIPT.replace(
    _RUNNER_MODULE, "document_kv_cache.quality_v2_job"
).replace("cachet.full_score.locked_runtime.v2", "cachet.quality_v2.locked_runtime.v1")
QUALITY_V2_RUNNER_SHA256 = hashlib.sha256(
    QUALITY_V2_RUNNER_SCRIPT.encode("utf-8")
).hexdigest()


def _runtime_identity() -> RuntimeIdentity:
    layout = _full_score_handoff_layout(FULL_SCORE_MODEL_ID)
    if layout.key_position_encoding is None:
        raise ValueError("handoff layout must declare key positions")
    return RuntimeIdentity(
        model_id=FULL_SCORE_MODEL_ID,
        model_revision=MAIN_LATENCY_TOKENIZER_REVISION,
        tokenizer_id=FULL_SCORE_MODEL_ID,
        tokenizer_revision=MAIN_LATENCY_TOKENIZER_REVISION,
        lora_id=layout.lora_id,
        prompt_template_version=QUALITY_V2_PROMPT_TEMPLATE_VERSION,
        layout_version=layout.layout_version,
        kv_dtype=layout.dtype,
        block_size=layout.block_size,
        payload_axis_order=KVPayloadAxisOrder(layout.payload_axis_order).value,
        key_position_encoding=KVKeyPositionEncoding(layout.key_position_encoding).value,
        rope_theta=layout.rope_theta,
        rope_rotary_dim=layout.rope_rotary_dim,
    )


def quality_v2_vllm_server_command(
    python_executable: str, *, telemetry_jsonl: str
) -> list[str]:
    """The full-score consumer's vLLM launch with room for 256 decode tokens."""

    transfer = document_kv_transfer_config(
        kv_connector="DocumentKVConnector",
        kv_role="kv_consumer",
        payload_cache_max_bytes=0,
        telemetry_jsonl=telemetry_jsonl,
        runtime_identity=_runtime_identity(),
        require_runtime_handshake=True,
    )
    return [
        python_executable,
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        FULL_SCORE_MODEL_ID,
        "--served-model-name",
        FULL_SCORE_SERVED_MODEL_NAME,
        "--host",
        _SERVER_HOST,
        "--port",
        str(_SERVER_PORT),
        "--dtype",
        FULL_SCORE_MODEL_DTYPE,
        "--max-model-len",
        str(QUALITY_V2_MAX_MODEL_LEN),
        "--max-num-seqs",
        str(FULL_SCORE_REQUEST_PARALLELISM),
        "--gpu-memory-utilization",
        str(QUALITY_V2_GPU_MEMORY_UTILIZATION),
        "--kv-transfer-config",
        json.dumps(transfer, separators=(",", ":"), sort_keys=True),
        "--enable-prefix-caching",
        "--no-enable-log-requests",
        "--quantization",
        FULL_SCORE_MODEL_QUANTIZATION,
        "--revision",
        MAIN_LATENCY_TOKENIZER_REVISION,
        "--tokenizer-revision",
        MAIN_LATENCY_TOKENIZER_REVISION,
        "--kv-cache-dtype",
        FULL_SCORE_KV_DTYPE,
        "--attention-backend",
        FULL_SCORE_ATTENTION_BACKEND,
    ]


def _job_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(gpu_runtime_warning_environment_overrides())
    env.update(
        {
            CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV: "0",
            CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV: "head_major",
            CACHET_TRANSFORMERS_DEVICE_ENV: "cuda",
            CACHET_TRANSFORMERS_DEVICE_MAP_ENV: "auto",
            CACHET_TRANSFORMERS_MODEL_ID_ENV: FULL_SCORE_MODEL_ID,
            CACHET_TRANSFORMERS_MODEL_REVISION_ENV: MAIN_LATENCY_TOKENIZER_REVISION,
            CACHET_TRANSFORMERS_PRE_ROPE_ENV: "1",
            CACHET_TRANSFORMERS_QUANTIZATION_ENV: FULL_SCORE_GENERATOR_QUANTIZATION,
            CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV: json.dumps(
                FULL_SCORE_GENERATOR_QUANTIZATION_CONFIG,
                separators=(",", ":"),
                sort_keys=True,
            ),
            CACHET_TRANSFORMERS_TOKENIZER_ID_ENV: FULL_SCORE_MODEL_ID,
            CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV: MAIN_LATENCY_TOKENIZER_REVISION,
            CACHET_TRANSFORMERS_TORCH_DTYPE_ENV: FULL_SCORE_MODEL_DTYPE,
            CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV: "0",
            "VLLM_ATTENTION_BACKEND": FULL_SCORE_ATTENTION_BACKEND,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    return env


def quality_v2_submit_payload(
    *,
    run_name: str,
    task_key: str,
    runner_python_file: str,
    runner_arguments: Sequence[str],
    single_user_name: str,
    purpose: str,
    node_type_id: str = FULL_SCORE_CONSUMER_NODE_TYPE_ID,
    timeout_seconds: int = QUALITY_V2_TASK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """A one-task, single-node, on-demand runs/submit body (no retries)."""

    if Path(runner_python_file).parent.name != QUALITY_V2_RUNNER_SHA256:
        raise ValueError("runner_python_file must sit in a directory named by its SHA-256")
    return {
        "run_name": run_name,
        "timeout_seconds": timeout_seconds,
        "tasks": [
            {
                "task_key": task_key,
                "max_retries": 0,
                "timeout_seconds": timeout_seconds,
                "new_cluster": {
                    "aws_attributes": {"availability": "ON_DEMAND", "zone_id": "auto"},
                    "custom_tags": {"ResourceClass": "SingleNode", "purpose": purpose},
                    "data_security_mode": "SINGLE_USER",
                    "driver_node_type_id": node_type_id,
                    "node_type_id": node_type_id,
                    "num_workers": 0,
                    "single_user_name": single_user_name,
                    "spark_conf": {
                        "spark.databricks.cluster.profile": "singleNode",
                        "spark.master": "local[*]",
                    },
                    "spark_version": QUALITY_V2_SPARK_VERSION,
                },
                "spark_python_task": {
                    "python_file": runner_python_file,
                    "parameters": ["--runner-sha256", QUALITY_V2_RUNNER_SHA256, *runner_arguments],
                },
            }
        ],
    }


def _cluster_path(uri: str) -> Path:
    if uri.startswith("dbfs:/Volumes/"):
        return Path("/Volumes") / uri.removeprefix("dbfs:/Volumes/")
    return Path(uri)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def _produce(
    *, input_jsonl: Path, dataset: str, work_dir: Path, token_counts: Sequence[int]
) -> Path:
    """Generate Vanilla handoffs in token-bounded sub-batches; return enriched JSONL."""

    from document_kv_cache.benchmark_handoffs import (
        enrich_benchmark_jsonl_with_handoffs,
        generate_benchmark_handoff_bundles,
        load_benchmark_kv_chunk_generator,
    )
    from document_kv_cache.full_score_execution import _delete_intermediate_kvpack

    lines = input_jsonl.read_bytes().splitlines(keepends=True)
    if len(lines) != len(token_counts):
        raise ValueError("token counts do not match the chunk input")
    batches: list[list[bytes]] = [[]]
    batch_tokens = 0
    for line, tokens in zip(lines, token_counts, strict=True):
        if batches[-1] and batch_tokens + tokens > QUALITY_V2_PRODUCER_SUB_BATCH_TOKENS:
            batches.append([])
            batch_tokens = 0
        batches[-1].append(line)
        batch_tokens += tokens
    generator = load_benchmark_kv_chunk_generator(FULL_SCORE_GENERATOR_FACTORY)
    layout = _full_score_handoff_layout(FULL_SCORE_MODEL_ID)
    registry = quality_v2_scorer_registry()
    enriched_parts = []
    for index, batch in enumerate(batches):
        started = time.monotonic()
        sub_input = work_dir / "inputs" / f"sub-{index:03d}.jsonl"
        sub_input.parent.mkdir(parents=True, exist_ok=True)
        sub_input.write_bytes(b"".join(batch))
        output_dir = work_dir / "q8-kv" / f"sub-{index:03d}"
        manifest = work_dir / "manifests" / f"sub-{index:03d}.json"
        result = generate_benchmark_handoff_bundles(
            sub_input,
            output_dir=output_dir,
            generator=generator,
            layout=layout,
            dataset=dataset,
            backend="vllm",
            manifest_json=manifest,
            segmented=True,
            segment_per_document=True,
            cache_method="vanilla_prefill",
            model_id=FULL_SCORE_MODEL_ID,
            model_revision=MAIN_LATENCY_TOKENIZER_REVISION,
            tokenizer_id=FULL_SCORE_MODEL_ID,
            tokenizer_revision=MAIN_LATENCY_TOKENIZER_REVISION,
            generator_family="transformers",
            generator_version=FULL_SCORE_GENERATOR_VERSION,
            align_bytes=4096,
            require_artifact_contract=True,
            scorer_registry=registry,
        )
        _delete_intermediate_kvpack(result.shard_uri, output_dir=output_dir)
        enriched = work_dir / "enriched" / f"sub-{index:03d}.jsonl"
        enrich_benchmark_jsonl_with_handoffs(
            sub_input,
            manifest,
            enriched,
            dataset=dataset,
            arm_id=FULL_SCORE_VANILLA_ARM_ID,
            overwrite=True,
        )
        enriched_parts.append(enriched)
        _log(
            f"produced sub-batch {index + 1}/{len(batches)} ({len(batch)} examples) "
            f"in {time.monotonic() - started:.0f}s"
        )
    combined = work_dir / "enriched" / f"{dataset}.jsonl"
    with combined.open("wb") as handle:
        for part in enriched_parts:
            handle.write(part.read_bytes())
    return combined


def _wait_for_server(process: subprocess.Popen[Any], log_path: Path) -> None:
    deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
    base = f"http://{_SERVER_HOST}:{_SERVER_PORT}"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with {process.returncode}:\n{_tail(log_path)}")
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as response:
                served = {item.get("id") for item in json.loads(response.read()).get("data", [])}
            if FULL_SCORE_SERVED_MODEL_NAME in served:
                return
        except (urllib.error.URLError, TimeoutError, ValueError, ConnectionError):
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM did not become ready:\n{_tail(log_path)}")


def _tail(path: Path, limit: int = 6000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=60)


def _run_arm(
    *, arm: str, chunk_id: str, dataset: str, dataset_path: Path, output_json: Path
) -> dict[str, Any]:
    from document_kv_cache.benchmark_runner import (
        OpenAICompatibleBenchmarkConfig,
        benchmark_run_result_to_record,
        parse_benchmark_arm_specs,
        run_openai_compatible_benchmark,
    )

    base_url = f"http://{_SERVER_HOST}:{_SERVER_PORT}"
    salt = f"{QUALITY_V2_PROTOCOL_ID}:{chunk_id}:{arm}"
    common: dict[str, Any] = {
        "suite_id": f"{QUALITY_V2_PROTOCOL_ID}:{chunk_id}:{arm}",
        "dataset_paths": {dataset: dataset_path},
        "base_url": base_url,
        "cache_base_url": base_url,
        "model_id": FULL_SCORE_SERVED_MODEL_NAME,
        "canonical_model_id": FULL_SCORE_MODEL_ID,
        "model_revision": MAIN_LATENCY_TOKENIZER_REVISION,
        "tokenizer_id": FULL_SCORE_MODEL_ID,
        "tokenizer_revision": MAIN_LATENCY_TOKENIZER_REVISION,
        "hardware_target": "aws-g6-l4",
        "engine_id": "vllm",
        "engine_version": PUBLICATION_CAMPAIGN_ENGINE_VERSION,
        "serving_platform": "databricks-aws-single-gpu",
        "model_dtype": FULL_SCORE_MODEL_DTYPE,
        "model_quantization": FULL_SCORE_MODEL_QUANTIZATION,
        "runtime_kv_dtype": FULL_SCORE_KV_DTYPE,
        "max_tokens": QUALITY_V2_MAX_TOKENS,
        "temperature": 0.0,
        "timeout_seconds": _REQUEST_TIMEOUT_SECONDS,
        "repeats": 1,
        "request_parallelism": FULL_SCORE_REQUEST_PARALLELISM,
        "prompt_token_accounting": "server_usage",
        "prefix_cache_salt_mode": "per_request",
        "baseline_extra_body": {"add_special_tokens": False, "cache_salt": salt},
        "cache_extra_body": {"add_special_tokens": False, "cache_salt": salt},
        "runtime_id": f"{QUALITY_V2_PROTOCOL_ID}:{chunk_id}",
        "measurement_scopes": ("quality",),
        "prompt_template_version": QUALITY_V2_PROMPT_TEMPLATE_VERSION,
    }
    if arm == "baseline":
        config = OpenAICompatibleBenchmarkConfig(**common, arm_ids=(BASELINE_PREFILL_ARM,))
    else:
        arms, arm_base_urls, arm_endpoints, arm_extra_bodies = parse_benchmark_arm_specs(
            [_vanilla_arm_spec()]
        )
        config = OpenAICompatibleBenchmarkConfig(
            **common,
            arms=arms,
            arm_base_urls=arm_base_urls,
            arm_endpoints=arm_endpoints,
            arm_extra_bodies=arm_extra_bodies,
        )
    result = run_openai_compatible_benchmark(config, scorer_registry=quality_v2_scorer_registry())
    record = benchmark_run_result_to_record(result)
    output_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _environment_snapshot() -> dict[str, Any]:
    def run(command: Sequence[str]) -> str:
        try:
            return subprocess.run(
                list(command), capture_output=True, text=True, timeout=60, check=False
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"unavailable: {type(exc).__name__}"

    versions = {}
    for package in ("vllm", "torch", "transformers", "bitsandbytes", "cachet-kv"):
        try:
            from importlib.metadata import version

            versions[package] = version(package)
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            versions[package] = f"unavailable: {type(exc).__name__}"
    return {
        "disk": run(["df", "-B1", "/local_disk0"]),
        "gpu": run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "packages": versions,
        "python": sys.version,
    }


def run_chunk(*, chunk_json: str, output_dir: str, work_dir: str) -> dict[str, Any]:
    started = time.time()
    chunk = json.loads(_cluster_path(chunk_json).read_bytes())
    chunk_id = str(chunk["chunk_id"])
    dataset = str(chunk["dataset"])
    work = Path(work_dir) / chunk_id
    if work.exists():
        shutil.rmtree(work)
    (work / "inputs").mkdir(parents=True)
    local_input = work / "inputs" / f"{dataset}.jsonl"
    shutil.copyfile(_cluster_path(str(chunk["input_uri"])), local_input)
    if _sha256_file(local_input) != chunk["input_sha256"]:
        raise ValueError("chunk input SHA-256 mismatch")
    token_counts = [int(value) for value in chunk["prompt_tokens"]]
    timings: dict[str, float] = {}
    environment = _environment_snapshot()
    _log(f"chunk {chunk_id}: {len(token_counts)} {dataset} examples, {sum(token_counts)} tokens")

    stage = time.monotonic()
    producer_log = work / "producer.log"
    with producer_log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "document_kv_cache.quality_v2_job",
                "produce",
                "--input-jsonl",
                str(local_input),
                "--dataset",
                dataset,
                "--work-dir",
                str(work / "producer"),
                "--token-counts-json",
                json.dumps(token_counts),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=_job_environment(),
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"producer failed ({completed.returncode}):\n{_tail(producer_log)}")
    enriched = work / "producer" / "enriched" / f"{dataset}.jsonl"
    timings["produce_seconds"] = time.monotonic() - stage
    environment["disk_after_produce"] = subprocess.run(
        ["df", "-B1", "/local_disk0"], capture_output=True, text=True, check=False
    ).stdout.strip()

    server_log = work / "vllm-server.log"
    telemetry = work / "connector-telemetry.jsonl"
    stage = time.monotonic()
    with server_log.open("w", encoding="utf-8") as log_handle:
        server = subprocess.Popen(
            quality_v2_vllm_server_command(sys.executable, telemetry_jsonl=str(telemetry)),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=_job_environment(),
        )
        try:
            _wait_for_server(server, server_log)
            timings["server_start_seconds"] = time.monotonic() - stage
            for arm, path in (("baseline", local_input), ("vanilla", enriched)):
                stage = time.monotonic()
                _run_arm(
                    arm=arm,
                    chunk_id=chunk_id,
                    dataset=dataset,
                    dataset_path=path,
                    output_json=work / f"{arm}.json",
                )
                timings[f"{arm}_seconds"] = time.monotonic() - stage
                _log(f"{arm} arm done in {timings[f'{arm}_seconds']:.0f}s")
        finally:
            _terminate(server)

    destination = _cluster_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in (
        "baseline.json",
        "vanilla.json",
        "vllm-server.log",
        "connector-telemetry.jsonl",
        "producer.log",
    ):
        source = work / name
        if source.exists():
            shutil.copyfile(source, destination / name)
            files[name] = {"bytes": source.stat().st_size, "sha256": _sha256_file(source)}
    manifests = sorted((work / "producer" / "manifests").glob("sub-*.json"))
    files["handoff-manifests"] = {
        "count": len(manifests),
        "sha256": [_sha256_file(path) for path in manifests],
    }
    result = {
        "chunk_id": chunk_id,
        "dataset": dataset,
        "environment": environment,
        "examples": len(token_counts),
        "files": files,
        "input_sha256": chunk["input_sha256"],
        "prompt_tokens": sum(token_counts),
        "protocol": quality_v2_protocol_record(),
        "record_type": QUALITY_V2_CHUNK_RECORD_TYPE,
        "runner_sha256": QUALITY_V2_RUNNER_SHA256,
        "timings_seconds": timings,
        "wall_seconds": time.time() - started,
    }
    (destination / "chunk-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(work, ignore_errors=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one quality-v2 chunk.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-chunk")
    run.add_argument("--chunk-json", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--work-dir", default="/local_disk0/cachet-quality-v2")
    produce = subparsers.add_parser("produce")
    produce.add_argument("--input-jsonl", required=True)
    produce.add_argument("--dataset", required=True)
    produce.add_argument("--work-dir", required=True)
    produce.add_argument("--token-counts-json", required=True)
    args = parser.parse_args(argv)
    if args.command == "produce":
        _produce(
            input_jsonl=Path(args.input_jsonl),
            dataset=args.dataset,
            work_dir=Path(args.work_dir),
            token_counts=json.loads(args.token_counts_json),
        )
        return 0
    result = run_chunk(
        chunk_json=args.chunk_json, output_dir=args.output_dir, work_dir=args.work_dir
    )
    print(json.dumps({"chunk_id": result["chunk_id"], "ok": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
