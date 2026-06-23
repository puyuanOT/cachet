from pathlib import Path
import gc
import json
import os
import subprocess
import sys
from types import ModuleType
import urllib.error
import weakref

import pytest

import document_kv_cache.vllm_smoke as public_vllm_smoke
import restaurant_kv_serving.vllm_smoke as legacy_vllm_smoke
from document_kv_cache.serving_env import VLLM_SERVING_ENVIRONMENT_PROFILE
from document_kv_cache.vllm_smoke import (
    BASELINE_PREFIX_CACHE_SALT,
    CACHE_PREFIX_CACHE_SALT,
    DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV,
    FASTAPI_CONSTRAINT,
    HUGGINGFACE_HUB_CONSTRAINT,
    HF_MODEL_ID,
    NUMPY_CONSTRAINT,
    PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT,
    SERVED_MODEL_NAME,
    SMOKE_DATASETS,
    TOKENIZERS_CONSTRAINT,
    TRANSFORMERS_CONSTRAINT,
    VLLM_FIPS_OPENCV_OVERRIDE_CONSTRAINT,
    VLLM_VERSION,
    VLLMPreparedHandoffGenerationConfig,
    VLLMSmokeBenchmarkConfig,
    benchmark_dataset_paths,
    benchmark_failure_summary,
    build_benchmark_runner_args,
    build_metadata,
    build_prompt_token_budget_rows,
    build_vllm_native_provider_probe_record,
    build_vllm_server_args,
    cuda_wheel_env_paths,
    dataset_args,
    dependency_constraints,
    dependency_override_constraints,
    document_kv_transfer_config_for_smoke,
    document_kv_package_install_spec,
    install_document_kv_package,
    install_vllm,
    parse_args,
    parse_dataset_specs,
    prepare_generated_benchmark_handoffs,
    prepared_benchmark_handoff_coverage_record,
    run_prompt_token_budget_probe,
    run_vllm_smoke_benchmark,
    site_packages_dirs,
    smoke_dataset_records,
    validate_prepared_benchmark_handoffs,
)
from document_kv_cache.benchmarks import (
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
)
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import (
    build_engine_adapter_request,
    engine_adapter_request_to_record,
    sglang_adapter_spec,
    vllm_adapter_spec,
)
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment
from document_kv_cache.kvpack import PackChunk
from document_kv_cache.model_profiles import layout_for_model
from document_kv_cache.models import KVCacheKey
from vllm_kv_injection.vllm_dynamic_connector import DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY
from vllm_kv_injection.vllm_transfer_config import document_kv_transfer_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def prepared_dataset_paths(tmp_path, *, include_handoffs=True):
    paths = {}
    for dataset in SMOKE_DATASETS:
        request_id = f"cachet-{dataset}-1"
        handoff_path = tmp_path / "handoffs" / dataset / f"{dataset}-1.handoff.json"
        payload_uri = f"disk:{tmp_path / 'payloads' / dataset / f'{dataset}-1.kv'}"
        record = {
            "dataset": dataset,
            "example_id": f"{dataset}-1",
            "query": "Who is described?",
            "expected_answer": "Ada Lovelace",
            "documents": [{"document_id": "ada", "text": "Ada Lovelace biography"}],
        }
        if include_handoffs:
            write_handoff_json(handoff_path, request_id=request_id, payload_uri=payload_uri)
            record["kv_transfer_params"] = {
                DOCUMENT_KV_REQUEST_ID_PARAM: request_id,
                DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
                DOCUMENT_KV_PAYLOAD_URI_PARAM: payload_uri,
            }
        path = tmp_path / f"{dataset}.jsonl"
        path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        paths[dataset] = path
    return paths


def handoff_record(*, request_id: str, payload_uri: str, backend: str = "vllm") -> dict[str, object]:
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
    spec = vllm_adapter_spec() if backend == "vllm" else sglang_adapter_spec()
    adapter_request = build_engine_adapter_request(ready, spec=spec)
    return engine_adapter_request_to_record(adapter_request, payload_uri=payload_uri)


def write_handoff_json(path: Path, *, request_id: str, payload_uri: str, backend: str = "vllm") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(handoff_record(request_id=request_id, payload_uri=payload_uri, backend=backend), sort_keys=True),
        encoding="utf-8",
    )


class OneTokenBenchmarkKVGenerator:
    def generate(self, *, document, chunk, config, training_artifacts=None):
        del training_artifacts
        layout = layout_for_model(
            config.model_id,
            dtype=config.dtype,
            lora_id=config.lora_id,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
            ),
            payload=b"\0" * layout.bytes_per_token,
            token_count=1,
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


class TrackedOneTokenBenchmarkKVGenerator(OneTokenBenchmarkKVGenerator):
    last_ref = None

    def __init__(self) -> None:
        type(self).last_ref = weakref.ref(self)


def test_dependency_constraints_match_pinned_g5_vllm_stack():
    assert dependency_constraints() == list(VLLM_SERVING_ENVIRONMENT_PROFILE.dependency_constraints)
    assert all("==" in constraint for constraint in dependency_constraints())
    assert dependency_override_constraints() == [VLLM_FIPS_OPENCV_OVERRIDE_CONSTRAINT]
    assert VLLM_FIPS_OPENCV_OVERRIDE_CONSTRAINT == "opencv-python-headless==4.12.0.88"
    assert VLLM_VERSION == "0.23.0"
    assert TRANSFORMERS_CONSTRAINT == "transformers==5.12.1"
    assert HUGGINGFACE_HUB_CONSTRAINT == "huggingface-hub==1.20.1"
    assert TOKENIZERS_CONSTRAINT == "tokenizers==0.22.2"
    assert NUMPY_CONSTRAINT == "numpy==2.3.5"
    numpy_version = tuple(int(part) for part in NUMPY_CONSTRAINT.split("==", maxsplit=1)[1].split("."))
    assert (1, 25, 0) <= numpy_version < (2, 4, 0)
    assert FASTAPI_CONSTRAINT == "fastapi[standard]==0.136.0"
    fastapi_version = tuple(int(part) for part in FASTAPI_CONSTRAINT.split("==", maxsplit=1)[1].split("."))
    assert (0, 115, 0) <= fastapi_version < (0, 137, 0)
    assert PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT == "prometheus-fastapi-instrumentator==8.0.0"
    assert HF_MODEL_ID == "Qwen/Qwen3-4B-Instruct-2507"
    assert SERVED_MODEL_NAME == "qwen3:4b-instruct"


def test_smoke_dataset_records_cover_v1_release_datasets():
    records = smoke_dataset_records()

    assert set(records) == {"biography", "hotpotqa", "musique", "niah"}
    assert records["biography"]["expected_answer"] == "Katherine Johnson"
    assert records["hotpotqa"]["expected_answer"] == "Paris"
    assert records["musique"]["expected_answer"] == "Ada Lovelace"
    assert records["niah"]["expected_answer"] == "cerulean lantern"
    assert all(record["documents"] for record in records.values())


def test_document_kv_package_install_spec_prefers_config_then_env(monkeypatch, tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        package_install_spec="dbfs:/tmp/cachet/document_kv_cache.whl",
    )

    assert document_kv_package_install_spec(config) == "/dbfs/tmp/cachet/document_kv_cache.whl"

    monkeypatch.setenv(DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV, "dbfs:/tmp/cachet/from-env.whl")
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )

    assert document_kv_package_install_spec(config) == "/dbfs/tmp/cachet/from-env.whl"


def test_document_kv_package_install_spec_falls_back_to_source_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv(DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV, raising=False)
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )

    assert document_kv_package_install_spec(config) == str(REPO_ROOT)


def test_install_document_kv_package_uses_no_deps(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(public_vllm_smoke, "run", lambda argv: calls.append(argv))

    install_document_kv_package(tmp_path / "venv" / "bin" / "python", "/tmp/cachet.whl")

    assert calls == [
        [
            str(tmp_path / "venv" / "bin" / "python"),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "/tmp/cachet.whl",
        ]
    ]


def test_install_vllm_applies_fips_opencv_override_after_vllm_stack(monkeypatch, tmp_path):
    calls = []
    python = tmp_path / "venv" / "bin" / "python"
    monkeypatch.setattr(public_vllm_smoke, "run", lambda argv: calls.append(argv))

    install_vllm(python)

    assert calls == [
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        [str(python), "-m", "pip", "install", *dependency_constraints()],
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            *dependency_override_constraints(),
        ],
    ]


def test_vllm_server_args_use_qwen3_instruct_and_g5_safe_limits(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
    )
    args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")

    assert args[:4] == [str(tmp_path / "venv" / "bin" / "python"), "-u", "-m", "vllm.entrypoints.openai.api_server"]
    assert args[args.index("--model") + 1] == HF_MODEL_ID
    assert args[args.index("--served-model-name") + 1] == SERVED_MODEL_NAME
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "8123"
    assert args[args.index("--dtype") + 1] == "bfloat16"
    assert args[args.index("--max-model-len") + 1] == "4096"
    assert args[args.index("--max-num-seqs") + 1] == "2"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.85"
    assert json.loads(args[args.index("--kv-transfer-config") + 1]) == document_kv_transfer_config()
    assert "--trust-remote-code" in args
    assert "--no-enable-log-requests" in args
    assert "--disable-log-requests" not in args


def test_vllm_server_args_include_payload_cache_budget(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-cache-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        payload_cache_max_bytes=4096,
    )

    args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")
    decoded = json.loads(args[args.index("--kv-transfer-config") + 1])

    assert decoded == document_kv_transfer_config(payload_cache_max_bytes=4096)


def test_vllm_server_args_accept_full_benchmark_sizing_overrides(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="full-v1-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        max_model_len=32768,
        max_num_seqs=8,
        gpu_memory_utilization=0.72,
    )

    args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")

    assert args[args.index("--max-model-len") + 1] == "32768"
    assert args[args.index("--max-num-seqs") + 1] == "8"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.72"


def test_benchmark_runner_args_include_all_smoke_datasets(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        max_tokens=32,
        timeout_seconds=240,
        local_root=tmp_path / "local",
        server_port=8123,
    )
    dataset_paths = {name: tmp_path / f"{name}.jsonl" for name in smoke_dataset_records()}

    args = build_benchmark_runner_args(config, dataset_paths)

    assert args[:3] == [sys.executable, "-m", "document_kv_cache.benchmark_runner"]
    assert args[args.index("--suite-id") + 1] == "smoke-1"
    assert args[args.index("--base-url") + 1] == "http://127.0.0.1:8123"
    assert args[args.index("--model-id") + 1] == SERVED_MODEL_NAME
    assert args[args.index("--hardware-target") + 1] == "aws-g6-l4"
    assert args[args.index("--output-json") + 1] == str(tmp_path / "out" / "v1-benchmark.json")
    assert "--server-usage" in args
    assert "--cache-base-url" not in args
    assert "--cache-runtime-prompt" not in args
    assert dataset_args(dataset_paths) == [
        "--dataset",
        f"biography={tmp_path / 'biography.jsonl'}",
        "--dataset",
        f"hotpotqa={tmp_path / 'hotpotqa.jsonl'}",
        "--dataset",
        f"musique={tmp_path / 'musique.jsonl'}",
        "--dataset",
        f"niah={tmp_path / 'niah.jsonl'}",
    ]


def test_benchmark_runner_args_use_logical_cache_prompt_for_prepared_datasets(tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
        dataset_specs=specs,
    )

    args = build_benchmark_runner_args(config, parse_dataset_specs(specs))

    assert args[args.index("--cache-base-url") + 1] == "http://127.0.0.1:8123"
    assert "--cache-runtime-prompt" not in args
    assert json.loads(args[args.index("--baseline-extra-body-json") + 1]) == {
        "cache_salt": BASELINE_PREFIX_CACHE_SALT
    }
    assert json.loads(args[args.index("--cache-extra-body-json") + 1]) == {
        "cache_salt": CACHE_PREFIX_CACHE_SALT
    }


def test_prompt_token_budget_rows_use_full_logical_prompts(tmp_path):
    dataset_paths = {}
    for dataset in SMOKE_DATASETS:
        path = tmp_path / f"{dataset}.jsonl"
        path.write_text(
            (
                f'{{"dataset": "{dataset}", "example_id": "{dataset}-1", '
                '"query": "Who is described?", "expected_answer": "Ada Lovelace", '
                '"documents": [{"document_id": "ada", "text": "Ada Lovelace biography"}]}\n'
            ),
            encoding="utf-8",
        )
        dataset_paths[dataset] = path
    config = VLLMSmokeBenchmarkConfig(benchmark_id="smoke-1", output_dir=tmp_path / "out")

    rows = build_prompt_token_budget_rows(config, dataset_paths)

    assert {row["dataset"] for row in rows} == set(SMOKE_DATASETS)
    assert all("Documents:" in row["prompt"] for row in rows)
    assert all("Who is described?" in row["prompt"] for row in rows)


def test_validate_prompt_token_budget_writes_artifact_and_rejects_over_budget(monkeypatch, tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        max_model_len=32,
        max_tokens=4,
    )
    dataset_paths = {dataset: tmp_path / f"{dataset}.jsonl" for dataset in SMOKE_DATASETS}

    monkeypatch.setattr(
        public_vllm_smoke,
        "build_prompt_token_budget_rows",
        lambda cfg, paths: ({"dataset": "biography", "example_id": "bio-1", "prompt": "long prompt"},),
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "run_prompt_token_budget_probe",
        lambda *args, **kwargs: {
            "rows": [
                {
                    "dataset": "biography",
                    "example_id": "bio-1",
                    "prompt_tokens": 40,
                    "max_tokens": 4,
                    "total_tokens": 44,
                    "max_model_len": 32,
                }
            ],
            "over_budget": [
                {
                    "dataset": "biography",
                    "example_id": "bio-1",
                    "prompt_tokens": 40,
                    "max_tokens": 4,
                    "total_tokens": 44,
                    "max_model_len": 32,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="Prepared vLLM benchmark prompts exceed"):
        public_vllm_smoke.validate_prompt_token_budget(config, dataset_paths)

    record = json.loads(config.prompt_token_budget_path.read_text(encoding="utf-8"))
    assert record["over_budget"][0]["total_tokens"] == 44


def test_validate_prompt_token_budget_writes_failed_probe_artifact(monkeypatch, tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )
    dataset_paths = {dataset: tmp_path / f"{dataset}.jsonl" for dataset in SMOKE_DATASETS}
    monkeypatch.setattr(
        public_vllm_smoke,
        "build_prompt_token_budget_rows",
        lambda cfg, paths: ({"dataset": "biography", "example_id": "bio-1", "prompt": "prompt"},),
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "run_prompt_token_budget_probe",
        lambda *args, **kwargs: {
            "ok": False,
            "error_type": "TimeoutExpired",
            "error": "prompt token budget probe timed out after 180.0s",
            "rows": [],
            "over_budget": [],
        },
    )

    with pytest.raises(RuntimeError, match="Prompt token budget probe failed"):
        public_vllm_smoke.validate_prompt_token_budget(config, dataset_paths)

    record = json.loads(config.prompt_token_budget_path.read_text(encoding="utf-8"))
    assert record["ok"] is False
    assert record["error_type"] == "TimeoutExpired"


def test_run_prompt_token_budget_probe_returns_timeout_record(monkeypatch, tmp_path):
    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=3, output="partial out", stderr="partial err")

    monkeypatch.setattr(public_vllm_smoke.subprocess, "run", timeout_run)

    record = run_prompt_token_budget_probe(
        tmp_path / "python",
        tmp_path / "input.jsonl",
        model_id=HF_MODEL_ID,
        max_model_len=32,
        max_tokens=4,
        timeout_seconds=3,
    )

    assert record["ok"] is False
    assert record["error_type"] == "TimeoutExpired"
    assert "partial out" in record["stdout_tail"]
    assert "partial err" in record["stderr_tail"]


def test_run_prompt_token_budget_probe_returns_nonzero_record(monkeypatch, tmp_path):
    completed = subprocess.CompletedProcess(
        args=["python"],
        returncode=17,
        stdout="not json",
        stderr="tokenizer failed",
    )
    monkeypatch.setattr(public_vllm_smoke.subprocess, "run", lambda *args, **kwargs: completed)

    record = run_prompt_token_budget_probe(
        tmp_path / "python",
        tmp_path / "input.jsonl",
        model_id=HF_MODEL_ID,
        max_model_len=32,
        max_tokens=4,
        timeout_seconds=3,
    )

    assert record["ok"] is False
    assert record["returncode"] == 17
    assert record["error_type"] == "CalledProcessError"
    assert "tokenizer failed" in record["stderr_tail"]


def test_benchmark_failure_summary_reports_row_errors(tmp_path):
    output_path = tmp_path / "v1-benchmark.json"
    output_path.write_text(
        (
            '{"measurements": ['
            '{"dataset": "biography", "arm_id": "full_prefill", "error": "context overflow"},'
            '{"dataset": "hotpotqa", "arm_id": "full_prefill", "error": "server rejected request"},'
            '{"dataset": "musique", "arm_id": "cache_reuse", "error": "another failure"},'
            '{"dataset": "niah", "arm_id": "cache_reuse", "error": "last failure"}'
            "]}\n"
        ),
        encoding="utf-8",
    )

    summary = benchmark_failure_summary(output_path, limit=2)

    assert "4/4 errored measurements" in summary
    assert "biography/full_prefill: context overflow" in summary
    assert "hotpotqa/full_prefill: server rejected request" in summary
    assert "2 more" in summary


def test_run_benchmark_runner_reraises_with_failure_summary(monkeypatch, tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )
    config.output_dir.mkdir()
    config.benchmark_output_path.write_text(
        '{"measurements": [{"dataset": "biography", "arm_id": "full_prefill", "error": "too long"}]}\n',
        encoding="utf-8",
    )

    def fail_run(argv):
        raise subprocess.CalledProcessError(2, argv)

    monkeypatch.setattr(public_vllm_smoke, "run", fail_run)

    with pytest.raises(RuntimeError, match="biography/full_prefill: too long"):
        public_vllm_smoke.run_benchmark_runner(
            config,
            {dataset: tmp_path / f"{dataset}.jsonl" for dataset in SMOKE_DATASETS},
        )


def test_parse_dataset_specs_requires_complete_v1_dataset_set(tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)

    paths = parse_dataset_specs(specs)

    assert list(paths) == list(SMOKE_DATASETS)
    assert paths["biography"] == tmp_path / "biography.jsonl"

    with pytest.raises(ValueError, match="missing required V1 datasets"):
        parse_dataset_specs((f"biography={tmp_path / 'biography.jsonl'}",))
    with pytest.raises(ValueError, match="Unsupported V1 smoke dataset"):
        parse_dataset_specs(specs + (f"unknown={tmp_path / 'unknown.jsonl'}",))
    with pytest.raises(ValueError, match="duplicate dataset spec"):
        parse_dataset_specs(specs + (f"biography={tmp_path / 'other.jsonl'}",))
    with pytest.raises(ValueError, match="DATASET=JSONL_PATH"):
        parse_dataset_specs(("biography",))


def test_parse_dataset_specs_maps_dbfs_uris_to_cluster_paths():
    specs = tuple(f"{dataset}=dbfs:/benchmarks/v1/{dataset}.jsonl" for dataset in SMOKE_DATASETS)

    paths = parse_dataset_specs(specs)

    assert paths["biography"] == Path("/dbfs/benchmarks/v1/biography.jsonl")
    assert paths["niah"] == Path("/dbfs/benchmarks/v1/niah.jsonl")


def test_benchmark_dataset_paths_uses_prepared_specs_without_writing_smoke(monkeypatch, tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="full-v1-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
    )

    def fail_if_smoke_is_written(local_dir):
        raise AssertionError(f"unexpected smoke dataset write to {local_dir}")

    monkeypatch.setattr(public_vllm_smoke, "write_smoke_datasets", fail_if_smoke_is_written)

    assert benchmark_dataset_paths(config) == parse_dataset_specs(specs)


def test_prepare_generated_benchmark_handoffs_writes_enriched_prepared_inputs(tmp_path, monkeypatch):
    module = ModuleType("cachet_test_vllm_handoff_generator")
    module.build_generator = OneTokenBenchmarkKVGenerator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=False)
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-generated-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
        handoff_generation=VLLMPreparedHandoffGenerationConfig(
            generator_factory=f"{module.__name__}:build_generator",
            output_dir=tmp_path / "generated-handoffs",
            dtype="bfloat16",
            align_bytes=1,
        ),
    )

    generated_paths = prepare_generated_benchmark_handoffs(config, dataset_paths)
    coverage = validate_prepared_benchmark_handoffs(config, generated_paths)

    assert list(generated_paths) == list(SMOKE_DATASETS)
    assert coverage is not None
    assert coverage["ok"] is True
    generation = json.loads(config.prepared_handoff_generation_path.read_text(encoding="utf-8"))
    assert generation["ok"] is True
    assert generation["dtype"] == "bfloat16"
    assert generation["datasets"]["biography"]["entries"] == 1
    enriched = json.loads(generated_paths["biography"].read_text(encoding="utf-8"))
    assert enriched["kv_transfer_params"][DOCUMENT_KV_REQUEST_ID_PARAM].startswith("cachet-biography-biography-1-")
    handoff_json = Path(enriched["kv_transfer_params"][DOCUMENT_KV_HANDOFF_JSON_PARAM])
    payload_uri = enriched["kv_transfer_params"][DOCUMENT_KV_PAYLOAD_URI_PARAM]
    assert handoff_json.exists()
    assert payload_uri.startswith(str(tmp_path / "generated-handoffs" / "biography"))


def test_prepare_generated_benchmark_handoffs_releases_generator_before_cleanup(tmp_path, monkeypatch):
    module = ModuleType("cachet_test_vllm_handoff_generator_cleanup")
    module.build_generator = TrackedOneTokenBenchmarkKVGenerator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    released_after_generator_collectable = []

    def fake_release_handoff_generation_resources():
        gc.collect()
        generator_ref = TrackedOneTokenBenchmarkKVGenerator.last_ref
        released_after_generator_collectable.append(generator_ref is not None and generator_ref() is None)

    monkeypatch.setattr(
        public_vllm_smoke,
        "release_handoff_generation_resources",
        fake_release_handoff_generation_resources,
    )
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=False)
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-generated-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
        handoff_generation=VLLMPreparedHandoffGenerationConfig(
            generator_factory=f"{module.__name__}:build_generator",
            output_dir=tmp_path / "generated-handoffs",
            dtype="bfloat16",
            align_bytes=1,
        ),
    )

    prepare_generated_benchmark_handoffs(config, dataset_paths)

    assert released_after_generator_collectable == [True]


def test_prepare_generated_benchmark_handoffs_uses_vllm_venv_when_available(tmp_path, monkeypatch):
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=False)
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-generated-venv",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
        handoff_generation=VLLMPreparedHandoffGenerationConfig(
            generator_factory="module:factory",
            output_dir=tmp_path / "generated-handoffs",
            dtype="bfloat16",
            align_bytes=1,
            timeout_seconds=1234.0,
        ),
    )
    config.venv_python.parent.mkdir(parents=True)
    config.venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    generated_worker_paths = {
        dataset: tmp_path / "generated-handoffs" / f"{dataset}.handoffs.jsonl"
        for dataset in SMOKE_DATASETS
    }
    calls = []

    def fake_run(argv, *, check, capture_output, text, timeout, env):
        calls.append((argv, check, capture_output, text, timeout, env))
        assert argv[0] == str(config.venv_python)
        assert argv[1] == "-c"
        input_payload = json.loads(Path(argv[3]).read_text(encoding="utf-8"))
        assert input_payload["benchmark_id"] == "prepared-generated-venv"
        assert input_payload["handoff_generation"]["generator_factory"] == "module:factory"
        assert input_payload["handoff_generation"]["timeout_seconds"] == 1234.0
        Path(argv[4]).parent.mkdir(parents=True, exist_ok=True)
        for path in generated_worker_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        Path(argv[4]).write_text(
            json.dumps(
                {
                    "generated_paths": {
                        dataset: str(path)
                        for dataset, path in generated_worker_paths.items()
                    },
                    "record": {
                        "ok": True,
                        "dataset_source": "prepared",
                        "generator_python": str(config.venv_python),
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="worker ok", stderr="")

    monkeypatch.setattr(public_vllm_smoke.subprocess, "run", fake_run)

    generated_paths = prepare_generated_benchmark_handoffs(config, dataset_paths)

    assert generated_paths == generated_worker_paths
    generation = json.loads(config.prepared_handoff_generation_path.read_text(encoding="utf-8"))
    assert generation["ok"] is True
    assert generation["generator_python"] == str(config.venv_python)
    assert len(calls) == 1
    assert calls[0][4] == 1234.0
    assert calls[0][5]["HF_HOME"] == str(config.hf_cache_dir)


def test_prepared_benchmark_handoff_coverage_record_counts_enriched_rows(tmp_path):
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=True)
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
    )

    record = prepared_benchmark_handoff_coverage_record(config, dataset_paths)

    assert record["ok"] is True
    assert record["required"] is True
    assert record["examples"] == len(SMOKE_DATASETS)
    assert record["examples_with_kv_transfer_params"] == len(SMOKE_DATASETS)
    assert record["examples_with_loadable_handoff_references"] == len(SMOKE_DATASETS)
    assert record["missing_kv_transfer_params"] == []
    assert record["invalid_handoff_references"] == []
    assert record["datasets"] == {dataset: 1 for dataset in SMOKE_DATASETS}


def test_prepared_benchmark_handoff_coverage_treats_null_inline_record_as_absent(tmp_path):
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=True)
    record = json.loads(dataset_paths["biography"].read_text(encoding="utf-8"))
    record["kv_transfer_params"][DOCUMENT_KV_HANDOFF_RECORD_PARAM] = None
    dataset_paths["biography"].write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
    )

    coverage = prepared_benchmark_handoff_coverage_record(config, dataset_paths)

    assert coverage["ok"] is True
    assert coverage["invalid_handoff_references"] == []


def test_validate_prepared_benchmark_handoffs_writes_artifact_and_rejects_missing_params(tmp_path):
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=False)
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
    )

    with pytest.raises(ValueError, match="must be enriched with Cachet kv_transfer_params"):
        validate_prepared_benchmark_handoffs(config, dataset_paths)

    record = json.loads(config.prepared_handoff_coverage_path.read_text(encoding="utf-8"))
    assert record["ok"] is False
    assert record["examples_with_kv_transfer_params"] == 0
    assert record["examples_with_loadable_handoff_references"] == 0
    assert record["missing_kv_transfer_params"] == [f"{dataset}/{dataset}-1" for dataset in SMOKE_DATASETS]
    assert record["invalid_handoff_references"] == []


def test_validate_prepared_benchmark_handoffs_rejects_unloadable_handoff_references(tmp_path):
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=True)
    bad_handoff = tmp_path / "missing-handoff.json"
    bad_backend = tmp_path / "sglang-handoff.json"
    bad_request = tmp_path / "wrong-request.handoff.json"
    bad_payload_uri = tmp_path / "remote-payload.handoff.json"
    write_handoff_json(
        bad_backend,
        request_id="cachet-hotpotqa-1",
        payload_uri=f"disk:{tmp_path / 'payloads' / 'hotpotqa' / 'hotpotqa-1.kv'}",
        backend="sglang",
    )
    write_handoff_json(
        bad_request,
        request_id="different-request",
        payload_uri=f"disk:{tmp_path / 'payloads' / 'musique' / 'musique-1.kv'}",
    )
    write_handoff_json(
        bad_payload_uri,
        request_id="cachet-niah-1",
        payload_uri="s3://cachet-bucket/niah-1.kv",
    )
    replacements = {
        "biography": bad_handoff,
        "hotpotqa": bad_backend,
        "musique": bad_request,
        "niah": bad_payload_uri,
    }
    for dataset, handoff_path in replacements.items():
        record = json.loads(dataset_paths[dataset].read_text(encoding="utf-8"))
        record["kv_transfer_params"][DOCUMENT_KV_HANDOFF_JSON_PARAM] = str(handoff_path)
        if dataset == "niah":
            record["kv_transfer_params"].pop(DOCUMENT_KV_PAYLOAD_URI_PARAM)
        dataset_paths[dataset].write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
    )

    with pytest.raises(ValueError, match="invalid handoff references"):
        validate_prepared_benchmark_handoffs(config, dataset_paths)

    record = json.loads(config.prepared_handoff_coverage_path.read_text(encoding="utf-8"))
    invalid = record["invalid_handoff_references"]
    assert record["ok"] is False
    assert record["examples_with_kv_transfer_params"] == len(SMOKE_DATASETS)
    assert record["examples_with_loadable_handoff_references"] == 0
    assert [issue["dataset"] for issue in invalid] == ["biography", "hotpotqa", "musique", "niah"]
    assert invalid[0]["error_type"] == "FileNotFoundError"
    assert "expected_backend" in invalid[1]["error"]
    assert "request_id" in invalid[2]["error"]
    assert "Engine probe runner can read only" in invalid[3]["error"]


def test_validate_prepared_benchmark_handoffs_rejects_inline_non_vllm_handoff_record(tmp_path):
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=True)
    record = json.loads(dataset_paths["biography"].read_text(encoding="utf-8"))
    request_id = "cachet-biography-1"
    record["kv_transfer_params"] = {
        DOCUMENT_KV_REQUEST_ID_PARAM: request_id,
        DOCUMENT_KV_HANDOFF_RECORD_PARAM: handoff_record(
            request_id=request_id,
            payload_uri=f"disk:{tmp_path / 'payloads' / 'biography' / 'biography-1.kv'}",
            backend="sglang",
        ),
    }
    dataset_paths["biography"].write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
    )

    with pytest.raises(ValueError, match="invalid handoff references"):
        validate_prepared_benchmark_handoffs(config, dataset_paths)

    record = json.loads(config.prepared_handoff_coverage_path.read_text(encoding="utf-8"))
    invalid = record["invalid_handoff_references"]
    assert record["ok"] is False
    assert record["examples_with_loadable_handoff_references"] == len(SMOKE_DATASETS) - 1
    assert invalid == [
        {
            "dataset": "biography",
            "example_id": "biography-1",
            "error_type": "ValueError",
            "error": "Engine adapter handoff backend 'sglang' does not match expected_backend",
        }
    ]


def test_validate_prepared_benchmark_handoffs_skips_builtin_smoke(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )

    assert validate_prepared_benchmark_handoffs(config, {}) is None
    assert not config.prepared_handoff_coverage_path.exists()


def test_validate_prepared_benchmark_handoffs_writes_ok_artifact(tmp_path):
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=True)
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
    )

    record = validate_prepared_benchmark_handoffs(config, dataset_paths)

    assert record is not None
    assert record["ok"] is True
    assert json.loads(config.prepared_handoff_coverage_path.read_text(encoding="utf-8")) == record


def test_metadata_records_reproducible_smoke_context(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )

    metadata = build_metadata(config)

    assert metadata["benchmark_id"] == "smoke-1"
    assert metadata["hf_model_id"] == HF_MODEL_ID
    assert metadata["served_model_name"] == SERVED_MODEL_NAME
    assert metadata["server_bind_host"] == "127.0.0.1"
    assert metadata["server_client_host"] == "127.0.0.1"
    assert metadata["server_base_url"] == "http://127.0.0.1:8000"
    assert metadata["hf_home"] == str(tmp_path / "local" / "hf-cache")
    assert metadata["vllm_python"] == str(tmp_path / "local" / "document-kv-vllm-smoke-smoke-1" / "vllm-venv" / "bin" / "python")
    assert metadata["dependency_constraints"] == dependency_constraints()
    assert metadata["dataset_source"] == "smoke"
    assert metadata["dataset_specs"] == []
    assert metadata["cache_runtime_prompt"] is False
    assert metadata["cache_prompt_text_mode"] == "logical"
    assert metadata["prefix_cache_isolation"] is None
    assert metadata["requires_kv_transfer_params"] is False
    assert metadata["max_model_len"] == 4096
    assert metadata["max_num_seqs"] == 2
    assert metadata["gpu_memory_utilization"] == 0.85
    assert metadata["document_kv_package_install_spec"] == str(REPO_ROOT)
    assert metadata["dependency_override_constraints"] == dependency_override_constraints()
    assert metadata["vllm_server_env_overrides"] == {
        "PYTHONUNBUFFERED": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    assert metadata["vllm_kv_transfer_config"] == document_kv_transfer_config()


def test_metadata_records_payload_cache_budget(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-cache-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        payload_cache_max_bytes=4096,
    )

    metadata = build_metadata(config)

    assert metadata["vllm_kv_transfer_config"] == document_kv_transfer_config_for_smoke(config)
    assert metadata["vllm_kv_transfer_config"] == document_kv_transfer_config(payload_cache_max_bytes=4096)


def test_vllm_native_provider_probe_record_instantiates_default_provider():
    record = build_vllm_native_provider_probe_record()

    assert record["document_kv_native_provider_ok"] is True
    assert (
        record["document_kv_provider_factory"]
        == document_kv_transfer_config()["kv_connector_extra_config"][DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY]
    )
    assert (
        record["document_kv_provider_type"]
        == "vllm_kv_injection.vllm_native_provider.DocumentKVNativeProvider"
    )
    assert record["document_kv_connector_type"] == "vllm_kv_injection.vllm_dynamic_connector.DocumentKVConnector"
    assert record["document_kv_requires_native_runtime"] is True


def test_vllm_native_provider_probe_record_rejects_missing_provider_factory():
    config = document_kv_transfer_config(provider_factory=None)

    with pytest.raises(ValueError, match=DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY):
        build_vllm_native_provider_probe_record(config)


def test_vllm_native_provider_probe_record_rejects_non_native_provider(monkeypatch):
    class NonNativeProvider:
        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            return 0, False

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            return None

        def build_connector_meta(self, scheduler_output):
            return {}

        def register_kv_caches(self, kv_caches):
            return None

        def start_load_kv(self, forward_context, **kwargs):
            return None

        def wait_for_layer_load(self, layer_name):
            return None

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
            return None

        def wait_for_save(self):
            return None

        def request_finished(self, request, block_ids):
            return False, None

        def request_finished_all_groups(self, request, block_ids):
            return False, None

    module = ModuleType("document_kv_smoke_non_native_provider")
    module.build_provider = lambda *, vllm_config, extra_config: NonNativeProvider()
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(TypeError, match="native document KV provider"):
        build_vllm_native_provider_probe_record(
            document_kv_transfer_config(provider_factory=f"{module.__name__}:build_provider")
        )


def test_probe_vllm_import_records_native_provider_evidence(monkeypatch, tmp_path):
    completed = subprocess.CompletedProcess(
        args=["python"],
        returncode=0,
        stdout=(
            "probe warmup\n"
            '{"ok": true, "document_kv_native_provider_ok": true, '
            '"document_kv_provider_factory": "vllm_kv_injection.vllm_native_provider:build_document_kv_provider"}\n'
        ),
        stderr="",
    )
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert "build_vllm_native_provider_probe_record" in argv[2]
        return completed

    monkeypatch.setattr(public_vllm_smoke.subprocess, "run", fake_run)

    public_vllm_smoke.probe_vllm_import(
        tmp_path / "venv" / "bin" / "python",
        tmp_path / "probe.json",
        timeout_seconds=3,
        env={"HF_HOME": str(tmp_path / "hf-cache")},
    )

    record = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))
    assert record["ok"] is True
    assert record["document_kv_native_provider_ok"] is True
    assert (
        record["document_kv_provider_factory"]
        == "vllm_kv_injection.vllm_native_provider:build_document_kv_provider"
    )
    assert calls[0][1]["env"]["HF_HOME"] == str(tmp_path / "hf-cache")


def test_metadata_records_prepared_dataset_context(tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="full-v1-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        max_model_len=32768,
        max_num_seqs=8,
        gpu_memory_utilization=0.72,
        dataset_specs=specs,
    )

    metadata = build_metadata(config)

    assert metadata["dataset_source"] == "prepared"
    assert metadata["dataset_specs"] == list(specs)
    assert metadata["cache_runtime_prompt"] is False
    assert metadata["cache_prompt_text_mode"] == "logical"
    assert metadata["prefix_cache_isolation"] == {
        "baseline_cache_salt": BASELINE_PREFIX_CACHE_SALT,
        "cache_cache_salt": CACHE_PREFIX_CACHE_SALT,
    }
    assert metadata["requires_kv_transfer_params"] is True
    assert metadata["generates_prepared_handoffs"] is False
    assert metadata["benchmark_handoff_generation"] is None
    assert metadata["max_model_len"] == 32768
    assert metadata["max_num_seqs"] == 8
    assert metadata["gpu_memory_utilization"] == 0.72
    assert metadata["document_kv_package_install_spec"] == str(REPO_ROOT)


def test_parse_args_builds_config_with_overrides(tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    config = parse_args(
        [
            "--benchmark-id",
            "smoke-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--local-root",
            str(tmp_path / "local"),
            "--max-tokens",
            "16",
            "--timeout-seconds",
            "12.5",
            "--import-probe-timeout-seconds",
            "9",
            "--server-start-timeout-seconds",
            "30",
            "--server-host",
            "0.0.0.0",
            "--server-port",
            "8123",
            "--client-host",
            "127.0.0.1",
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "8",
            "--gpu-memory-utilization",
            "0.72",
            "--package-install-spec",
            str(tmp_path / "cachet.whl"),
            "--benchmark-handoff-generator-factory",
            "document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator",
            "--benchmark-handoff-output-dir",
            "dbfs:/tmp/cachet/generated-handoffs",
            "--benchmark-handoff-dtype",
            "bfloat16",
            "--benchmark-handoff-align-bytes",
            "1",
            *sum((["--dataset", spec] for spec in specs), []),
        ]
    )

    assert config == VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        max_tokens=16,
        timeout_seconds=12.5,
        import_probe_timeout_seconds=9,
        server_start_timeout_seconds=30,
        server_host="0.0.0.0",
        server_port=8123,
        client_host="127.0.0.1",
        max_model_len=32768,
        max_num_seqs=8,
        gpu_memory_utilization=0.72,
        dataset_specs=specs,
        package_install_spec=str(tmp_path / "cachet.whl"),
        handoff_generation=VLLMPreparedHandoffGenerationConfig(
            generator_factory="document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator",
            output_dir=Path("/dbfs/tmp/cachet/generated-handoffs"),
            dtype="bfloat16",
            align_bytes=1,
        ),
    )


def test_vllm_smoke_config_validates_before_runtime_setup(tmp_path):
    invalid_cases = [
        ({"benchmark_id": ""}, "benchmark_id must be non-empty"),
        ({"max_tokens": 0}, "max_tokens must be positive"),
        ({"timeout_seconds": 0}, "timeout_seconds must be positive"),
        ({"import_probe_timeout_seconds": 0}, "import_probe_timeout_seconds must be positive"),
        ({"server_start_timeout_seconds": 0}, "server_start_timeout_seconds must be positive"),
        ({"server_host": ""}, "server_host must be non-empty"),
        ({"server_port": 0}, "server_port must be between 1 and 65535"),
        ({"server_port": 65536}, "server_port must be between 1 and 65535"),
        ({"client_host": ""}, "client_host must be non-empty"),
        ({"max_model_len": 0}, "max_model_len must be positive"),
        ({"max_num_seqs": 0}, "max_num_seqs must be positive"),
        ({"gpu_memory_utilization": 0}, "gpu_memory_utilization must be in"),
        ({"gpu_memory_utilization": 1.1}, "gpu_memory_utilization must be in"),
        ({"payload_cache_max_bytes": -1}, "payload_cache_max_bytes must be a non-negative integer"),
        ({"dataset_specs": ("biography=/tmp/biography.jsonl",)}, "dataset specs missing required V1 datasets"),
        ({"package_install_spec": ""}, "package_install_spec must be non-empty"),
        (
            {
                "handoff_generation": VLLMPreparedHandoffGenerationConfig(
                    generator_factory="module:factory",
                    output_dir=tmp_path / "generated-handoffs",
                )
            },
            "requires prepared dataset specs",
        ),
    ]

    for overrides, message in invalid_cases:
        kwargs = {
            "benchmark_id": "smoke-1",
            "output_dir": tmp_path / "out",
            "local_root": tmp_path / "local",
        }
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=message):
            VLLMSmokeBenchmarkConfig(**kwargs)


def test_parse_args_rejects_invalid_values_before_setup(tmp_path):
    with pytest.raises(ValueError, match="server_port must be between"):
        parse_args(
            [
                "--benchmark-id",
                "smoke-1",
                "--output-dir",
                str(tmp_path / "out"),
                "--server-port",
                "0",
            ]
        )


def test_parse_args_maps_dbfs_output_dir_to_driver_filesystem():
    config = parse_args(["--benchmark-id", "smoke-1", "--output-dir", "dbfs:/benchmarks/cachet-smoke/output"])

    assert config.output_dir == Path("/dbfs/benchmarks/cachet-smoke/output")


def test_server_base_url_uses_client_host_not_bind_host(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_host="0.0.0.0",
        server_port=8123,
    )

    server_args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")

    assert server_args[server_args.index("--host") + 1] == "0.0.0.0"
    assert config.server_base_url == "http://127.0.0.1:8123"
    assert build_metadata(config)["server_bind_host"] == "0.0.0.0"
    assert build_metadata(config)["server_client_host"] == "127.0.0.1"


def test_server_env_forces_local_root_hf_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", "/slow-or-wrong")
    monkeypatch.setenv("CPATH", "/existing/include")
    monkeypatch.setenv("LIBRARY_PATH", "/existing/lib")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/ld-lib")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )
    site_packages = config.venv_dir / "lib" / "python3.12" / "site-packages"
    curand_include = site_packages / "nvidia" / "curand" / "include"
    curand_lib = site_packages / "nvidia" / "curand" / "lib"
    runtime_include = site_packages / "nvidia" / "cuda_runtime" / "include"
    runtime_lib = site_packages / "nvidia" / "cuda_runtime" / "lib"
    for path in (curand_include, curand_lib, runtime_include, runtime_lib):
        path.mkdir(parents=True)

    env = legacy_vllm_smoke.server_env(config)

    assert env["HF_HOME"] == str(tmp_path / "local" / "hf-cache")
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert site_packages_dirs(config) == [site_packages]
    assert cuda_wheel_env_paths(config) == {
        "include": [str(runtime_include), str(curand_include)],
        "library": [str(runtime_lib), str(curand_lib)],
    }
    assert env["CPATH"].split(os.pathsep) == [str(runtime_include), str(curand_include), "/existing/include"]
    assert env["LIBRARY_PATH"].split(os.pathsep) == [str(runtime_lib), str(curand_lib), "/existing/lib"]
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [str(runtime_lib), str(curand_lib), "/existing/ld-lib"]


class _FakeServer:
    returncode = None

    def poll(self):
        return None


class _FakeResponse:
    def __init__(self, *, status=200, payload=b""):
        self.status = status
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self._payload


def test_wait_for_server_requires_expected_served_model(monkeypatch, tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
    )
    requested_urls = []

    def fake_urlopen(url, timeout):
        requested_urls.append(url)
        if url.endswith("/health"):
            return _FakeResponse(status=200)
        if url.endswith("/v1/models"):
            return _FakeResponse(
                status=200,
                payload=b'{"data":[{"id":"qwen3:4b-instruct"}]}',
            )
        raise urllib.error.URLError(f"unexpected url {url}")

    monkeypatch.setattr(legacy_vllm_smoke.urllib.request, "urlopen", fake_urlopen)

    legacy_vllm_smoke.wait_for_server(
        _FakeServer(),
        tmp_path / "missing.log",
        config,
        timeout_seconds=1,
    )

    assert requested_urls == [
        "http://127.0.0.1:8123/health",
        "http://127.0.0.1:8123/v1/models",
    ]


def test_run_vllm_smoke_benchmark_orchestrates_and_cleans_up(monkeypatch, tmp_path):
    calls = []
    fake_server = object()
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
    )
    dataset_paths = {name: tmp_path / f"{name}.jsonl" for name in smoke_dataset_records()}

    monkeypatch.setattr(public_vllm_smoke, "create_venv", lambda path: calls.append(("create_venv", path)))
    monkeypatch.setattr(public_vllm_smoke, "install_vllm", lambda python: calls.append(("install_vllm", python)))
    monkeypatch.setattr(
        public_vllm_smoke,
        "install_document_kv_package",
        lambda python, install_spec: calls.append(("install_document_kv_package", python, install_spec)),
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "installed_versions",
        lambda python: {"vllm_version_installed": "0.23.0", "transformers_version_installed": "5.12.1"},
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "probe_vllm_import",
        lambda python, output, *, timeout_seconds, env: calls.append(
            ("probe_vllm_import", python, output, timeout_seconds, env["HF_HOME"])
        ),
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "write_smoke_datasets",
        lambda local_dir: calls.append(("write_smoke_datasets", local_dir)) or dataset_paths,
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "validate_prompt_token_budget",
        lambda cfg, paths: calls.append(("validate_prompt_token_budget", cfg.benchmark_id, paths)),
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "start_vllm_server",
        lambda cfg, python, log_path: calls.append(("start_vllm_server", cfg.server_base_url, python, log_path))
        or fake_server,
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "wait_for_server",
        lambda server, log_path, cfg, *, timeout_seconds: calls.append(
            ("wait_for_server", server, log_path, cfg.server_base_url, timeout_seconds)
        ),
    )
    monkeypatch.setattr(public_vllm_smoke, "run", lambda argv: calls.append(("run", argv)))
    monkeypatch.setattr(public_vllm_smoke, "terminate_process", lambda server: calls.append(("terminate", server)))
    monkeypatch.setattr(
        public_vllm_smoke,
        "copy_file_if_exists",
        lambda source, target: calls.append(("copy", source, target)),
    )

    run_vllm_smoke_benchmark(config)

    assert calls == [
        ("create_venv", config.venv_dir),
        ("install_vllm", config.venv_python),
        ("install_document_kv_package", config.venv_python, str(REPO_ROOT)),
        (
            "probe_vllm_import",
            config.venv_python,
            config.import_probe_path,
            180.0,
            str(tmp_path / "local" / "hf-cache"),
        ),
        ("write_smoke_datasets", config.local_dir),
        ("validate_prompt_token_budget", "smoke-1", dataset_paths),
        ("start_vllm_server", "http://127.0.0.1:8123", config.venv_python, config.server_log_path),
        ("wait_for_server", fake_server, config.server_log_path, "http://127.0.0.1:8123", 480.0),
        ("copy", config.server_log_path, config.server_log_copy_path),
        ("run", build_benchmark_runner_args(config, dataset_paths)),
        ("terminate", fake_server),
        ("copy", config.server_log_path, config.server_log_copy_path),
    ]
    metadata = build_metadata(config)
    assert metadata["server_base_url"] == "http://127.0.0.1:8123"
    assert metadata["hf_home"] == str(tmp_path / "local" / "hf-cache")


def test_legacy_vllm_smoke_run_respects_legacy_helper_monkeypatch(monkeypatch, tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )

    def fake_create_venv(path):
        raise RuntimeError(f"legacy hook used for {path.name}")

    monkeypatch.setattr(legacy_vllm_smoke, "create_venv", fake_create_venv)

    with pytest.raises(RuntimeError, match="legacy hook used"):
        legacy_vllm_smoke.run_vllm_smoke_benchmark(config)


def test_legacy_vllm_smoke_run_reaches_dataset_selection_without_wrapper_recursion(monkeypatch, tmp_path):
    calls = []
    fake_server = object()
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
    )
    dataset_paths = {name: tmp_path / f"{name}.jsonl" for name in smoke_dataset_records()}

    monkeypatch.setattr(legacy_vllm_smoke, "create_venv", lambda path: calls.append(("create_venv", path)))
    monkeypatch.setattr(legacy_vllm_smoke, "install_vllm", lambda python: calls.append(("install_vllm", python)))
    monkeypatch.setattr(
        legacy_vllm_smoke,
        "install_document_kv_package",
        lambda python, install_spec: calls.append(("install_document_kv_package", install_spec)),
    )
    monkeypatch.setattr(
        legacy_vllm_smoke,
        "installed_versions",
        lambda python: {"vllm_version_installed": "0.23.0"},
    )
    monkeypatch.setattr(
        legacy_vllm_smoke,
        "probe_vllm_import",
        lambda python, output, *, timeout_seconds, env: calls.append(("probe_vllm_import", output)),
    )
    monkeypatch.setattr(
        legacy_vllm_smoke,
        "write_smoke_datasets",
        lambda local_dir: calls.append(("write_smoke_datasets", local_dir)) or dataset_paths,
    )
    monkeypatch.setattr(
        legacy_vllm_smoke,
        "validate_prompt_token_budget",
        lambda cfg, paths: calls.append(("validate_prompt_token_budget", cfg.benchmark_id)),
    )
    monkeypatch.setattr(
        legacy_vllm_smoke,
        "start_vllm_server",
        lambda cfg, python, log_path: calls.append(("start_vllm_server", log_path)) or fake_server,
    )
    monkeypatch.setattr(
        legacy_vllm_smoke,
        "wait_for_server",
        lambda server, log_path, cfg, *, timeout_seconds: calls.append(("wait_for_server", timeout_seconds)),
    )
    monkeypatch.setattr(legacy_vllm_smoke, "run", lambda argv: calls.append(("run", argv)))
    monkeypatch.setattr(legacy_vllm_smoke, "terminate_process", lambda server: calls.append(("terminate", server)))
    monkeypatch.setattr(
        legacy_vllm_smoke,
        "copy_file_if_exists",
        lambda source, target: calls.append(("copy", source, target)),
    )

    legacy_vllm_smoke.run_vllm_smoke_benchmark(config)

    assert ("install_document_kv_package", str(REPO_ROOT)) in calls
    assert ("write_smoke_datasets", config.local_dir) in calls
    assert ("validate_prompt_token_budget", "smoke-1") in calls
    assert ("run", build_benchmark_runner_args(config, dataset_paths)) in calls
    assert calls[-2:] == [
        ("terminate", fake_server),
        ("copy", config.server_log_path, config.server_log_copy_path),
    ]


def test_legacy_vllm_smoke_direct_helper_respects_legacy_run_monkeypatch(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(legacy_vllm_smoke, "run", lambda argv: calls.append(argv))

    legacy_vllm_smoke.create_venv(tmp_path / "venv")

    assert calls == [[sys.executable, "-m", "venv", str(tmp_path / "venv")]]


def test_legacy_vllm_smoke_main_respects_legacy_run_monkeypatch(monkeypatch, tmp_path):
    called = {}

    def fake_run(config):
        called["config"] = config

    monkeypatch.setattr(legacy_vllm_smoke, "run_vllm_smoke_benchmark", fake_run)

    exit_code = legacy_vllm_smoke.main(
        [
            "--benchmark-id",
            "smoke-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--local-root",
            str(tmp_path / "local"),
        ]
    )

    assert exit_code == 0
    assert called["config"].benchmark_id == "smoke-1"
    assert called["config"].output_dir == tmp_path / "out"
    assert called["config"].local_root == tmp_path / "local"


def test_legacy_vllm_smoke_module_execution_shows_help():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, "-m", "restaurant_kv_serving.vllm_smoke", "--help"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run a Qwen3/vLLM V1 benchmark smoke on Databricks g5/g6." in completed.stdout


def test_public_vllm_smoke_main_respects_document_namespace_monkeypatch(monkeypatch, tmp_path):
    called = {}

    def fake_run(config):
        called["config"] = config

    monkeypatch.setattr(public_vllm_smoke, "run_vllm_smoke_benchmark", fake_run)

    exit_code = public_vllm_smoke.main(
        [
            "--benchmark-id",
            "smoke-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--local-root",
            str(tmp_path / "local"),
        ]
    )

    assert exit_code == 0
    assert called["config"].benchmark_id == "smoke-1"
    assert called["config"].output_dir == tmp_path / "out"
    assert called["config"].local_root == tmp_path / "local"
