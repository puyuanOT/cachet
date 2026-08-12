from pathlib import Path
import gc
import json
import subprocess
import sys
from types import ModuleType
import weakref

import pytest

import document_kv_cache.vllm_smoke as public_vllm_smoke
from document_kv_cache.artifact_identity import RuntimeIdentity
from document_kv_cache.canary_orchestration import (
    representative_canary_matrix,
    representative_vllm_comparison_suite_id,
)
from document_kv_cache.serving_env import VLLM_SERVING_ENVIRONMENT_PROFILE
from document_kv_cache.transformers_generator import (
    CACHET_TRANSFORMERS_DEVICE_MAP_ENV,
    CACHET_TRANSFORMERS_MODEL_ID_ENV,
    CACHET_TRANSFORMERS_MODEL_REVISION_ENV,
    CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV,
    CACHET_TRANSFORMERS_QUANTIZATION_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_ID_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV,
    CACHET_TRANSFORMERS_TORCH_DTYPE_ENV,
    CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV,
)
from document_kv_cache.vllm_smoke import (
    BASELINE_PREFIX_CACHE_SALT,
    CACHE_PREFIX_CACHE_SALT,
    DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV,
    DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV,
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
    write_lmcache_config,
    _build_lmcache_pass_args,
    dataset_args,
    dependency_constraints,
    dependency_override_constraints,
    document_kv_transfer_config_for_smoke,
    document_kv_package_install_spec,
    apply_vllm_runtime_patches,
    install_document_kv_package,
    install_vllm,
    kv_transfer_config_json,
    parse_args,
    parse_dataset_specs,
    prepare_generated_benchmark_handoffs,
    prewarm_cache_prefixes,
    prepared_benchmark_handoff_coverage_record,
    run_prompt_token_budget_probe,
    run_vllm_smoke_benchmark,
    server_env,
    smoke_dataset_records,
    validate_prepared_benchmark_handoffs,
    vllm_representative_workload_profile,
)
from document_kv_cache.benchmarks import (
    CACHE_REUSE_ARM,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
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
MODEL_REVISION = "a" * 40
REPRESENTATIVE_WHEEL_SHA256 = "f" * 64


@pytest.fixture(autouse=True)
def _verified_representative_wheel_sha256(monkeypatch):
    monkeypatch.setenv(
        DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV,
        REPRESENTATIVE_WHEEL_SHA256,
    )


def representative_vllm_kwargs(
    tmp_path: Path,
    *,
    profile_id: str = "vllm-8k-64-v1",
    arm_index: int = 0,
) -> dict[str, object]:
    profile = vllm_representative_workload_profile(profile_id)
    matrix = representative_canary_matrix()
    return {
        "benchmark_suite_id": representative_vllm_comparison_suite_id(
            hardware_target="aws-g6-l4",
            profile_id=profile.profile_id,
        ),
        "benchmark_runtime_id": "test-physical-run-1",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "representative_canary": True,
        "representative_workload_profile": profile.profile_id,
        "benchmark_arm_specs": (matrix.runs[arm_index].arm_spec,),
        "benchmark_evidence_policy": "canary",
        "benchmark_manifest_provenance": {
            "input_tokens_target": profile.input_tokens_target,
        },
        "max_tokens": profile.max_output_tokens,
        "max_model_len": profile.max_model_len,
        "max_num_seqs": profile.max_num_seqs,
        "gpu_memory_utilization": profile.gpu_memory_utilization,
        "force_max_tokens": True,
        "benchmark_repeats": profile.benchmark_repeats,
        "request_parallelism": profile.request_parallelism,
        "dataset_specs": tuple(
            f"{dataset}={tmp_path / f'{dataset}.jsonl'}"
            for dataset in SMOKE_DATASETS
        ),
    }


def stored_post_rope_runtime_identity() -> RuntimeIdentity:
    return RuntimeIdentity(
        model_id=HF_MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=HF_MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        lora_id="base",
        prompt_template_version="v1",
        layout_version="qwen3-v1",
        kv_dtype="bfloat16",
        block_size=16,
        payload_axis_order="token_major",
        key_position_encoding="stored_post_rope",
    )


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


def test_apply_vllm_runtime_patches_disables_e5m2_query_quant(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        kv_cache_dtype="fp8_e5m2",
    )
    attention_path = (
        config.venv_dir
        / "lib"
        / "python3.11"
        / "site-packages"
        / "vllm"
        / "model_executor"
        / "layers"
        / "attention"
        / "attention.py"
    )
    attention_path.parent.mkdir(parents=True)
    attention_path.write_text(
        """prefix
        if (
            self.impl.supports_quant_query_input
            and (
                self.kv_cache_dtype.startswith("fp8") or self.kv_cache_dtype == "nvfp4"
            )
            and not self.kv_cache_dtype.endswith("per_token_head")
        ):
suffix
""",
        encoding="utf-8",
    )
    reshape_path = (
        config.venv_dir
        / "lib"
        / "python3.11"
        / "site-packages"
        / "vllm"
        / "v1"
        / "attention"
        / "ops"
        / "triton_reshape_and_cache_flash.py"
    )
    reshape_path.parent.mkdir(parents=True)
    reshape_path.write_text(
        """import torch
    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if is_quantized_kv_cache(kv_cache_dtype)
        else key_cache.dtype
    )
middle
    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if is_quantized_kv_cache(kv_cache_dtype)
        else kv_cache.dtype
    )
""",
        encoding="utf-8",
    )
    triton_attn_path = (
        config.venv_dir
        / "lib"
        / "python3.11"
        / "site-packages"
        / "vllm"
        / "v1"
        / "attention"
        / "backends"
        / "triton_attn.py"
    )
    triton_attn_path.parent.mkdir(parents=True)
    triton_attn_path.write_text(
        """import torch
        self.fp8_dtype = current_platform.fp8_dtype()
""",
        encoding="utf-8",
    )

    patches = apply_vllm_runtime_patches(config)
    text = attention_path.read_text(encoding="utf-8")
    reshape_text = reshape_path.read_text(encoding="utf-8")
    triton_attn_text = triton_attn_path.read_text(encoding="utf-8")

    assert patches == [
        {
            "id": "vllm-qwen-attention-disable-e5m2-query-quant",
            "path": str(attention_path),
            "applied": True,
            "reason": (
                "vLLM 0.23.0 admits fp8_e5m2 KV cache in Triton metadata but its attention "
                "wrapper's query quantization path asserts only fp8/fp8_e4m3/nvfp4."
            ),
        },
        {
            "id": "vllm-triton-reshape-cache-use-e5m2-dtype",
            "path": str(reshape_path),
            "applied": True,
            "reason": (
                "vLLM 0.23.0 routes all quantized KV cache dtypes through current_platform.fp8_dtype(); "
                "on AWS g5/A10G that selects an E4M3 dtype even when --kv-cache-dtype=fp8_e5m2."
            ),
        },
        {
            "id": "vllm-triton-attn-use-e5m2-cache-view",
            "path": str(triton_attn_path),
            "applied": True,
            "reason": (
                "TritonAttentionImpl stores the platform default FP8 dtype and views quantized KV "
                "cache pages through it; on AWS g5/A10G this selects E4M3 for fp8_e5m2 KV pages."
            ),
        },
    ]
    assert 'self.kv_cache_dtype in {"fp8", "fp8_e4m3", "nvfp4"}' in text
    assert "startswith(\"fp8\")" not in text
    assert 'if kv_cache_dtype == "fp8_e5m2"' in reshape_text
    assert 'self.fp8_dtype = (\n            torch.float8_e5m2' in triton_attn_text
    assert apply_vllm_runtime_patches(config) == [
        {**patches[0], "applied": False},
        {**patches[1], "applied": False},
        {**patches[2], "applied": False},
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
    assert json.loads(args[args.index("--kv-transfer-config") + 1]) == document_kv_transfer_config_for_smoke(config)
    assert "--enable-prefix-caching" in args
    assert "--trust-remote-code" in args
    assert "--no-enable-log-requests" in args
    assert "--disable-log-requests" not in args


def test_vllm_server_args_pin_runtime_identity(tmp_path):
    identity = stored_post_rope_runtime_identity()
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="runtime-identity-live",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        runtime_identity=identity,
    )

    args = build_vllm_server_args(
        config,
        tmp_path / "venv" / "bin" / "python",
    )
    transfer_config = json.loads(
        args[args.index("--kv-transfer-config") + 1]
    )
    extra_config = transfer_config["kv_connector_extra_config"]

    assert args[args.index("--revision") + 1] == MODEL_REVISION
    assert (
        args[args.index("--tokenizer-revision") + 1]
        == MODEL_REVISION
    )
    assert extra_config["document_kv.runtime_identity"] == identity.to_record()
    assert extra_config["document_kv.require_runtime_handshake"] is True


def test_vllm_server_args_default_mode_uses_cachet_connector(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="cachet-mode",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )
    args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")
    decoded = json.loads(args[args.index("--kv-transfer-config") + 1])
    assert decoded["kv_connector"] == "DocumentKVConnector"
    assert "--enable-prefix-caching" in args


def test_vllm_server_args_lmcache_mode_uses_lmcache_connector(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="lmcache-mode",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        kv_connector_mode="lmcache",
    )
    args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")
    decoded = json.loads(args[args.index("--kv-transfer-config") + 1])
    assert decoded == {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
    # LMCache manages reuse; vLLM prefix caching must be off to avoid double caching.
    assert "--enable-prefix-caching" not in args


def test_build_multi_turn_followup_prompt_preserves_prior_conversation():
    from document_kv_cache.vllm_smoke import build_multi_turn_followup_prompt

    turn1_prompt = "Document A ... question 1"
    turn1_response = " answer one"
    followup = "question 2"
    result = build_multi_turn_followup_prompt(turn1_prompt, turn1_response, followup)
    # The exact prior text must remain a prefix so the engine reuses the resident KV.
    assert result.startswith(turn1_prompt + turn1_response)
    assert result.endswith("question 2\n")
    # Chaining a second follow-up keeps growing the same prefix.
    chained = build_multi_turn_followup_prompt(result, " answer two", "question 3")
    assert chained.startswith(result + " answer two")


def test_vllm_server_args_multi_mode_wraps_cachet_then_lmcache(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="multi-mode",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        kv_connector_mode="multi",
    )
    args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")
    decoded = json.loads(args[args.index("--kv-transfer-config") + 1])
    assert decoded["kv_connector"] == "MultiConnector"
    children = decoded["kv_connector_extra_config"]["connectors"]
    assert [child["kv_connector"] for child in children] == [
        "DocumentKVConnector",
        "LMCacheConnectorV1",
    ]
    # Hybrid path keeps vLLM prefix caching on for turn-2+ continuation.
    assert "--enable-prefix-caching" in args
    # MultiConnector prom-metrics path asserts on Cachet's stats-without-prom-metrics;
    # server-side stat logging is disabled for the hybrid arm to avoid it.
    assert "--disable-log-stats" in args


def test_write_lmcache_config_targets_disk_tier_with_odirect(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="lm-cfg",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        kv_connector_mode="lmcache",
        lmcache_local_dir=str(tmp_path / "store"),
        lmcache_max_disk_gb=64.0,
        lmcache_chunk_size=256,
    )
    path = write_lmcache_config(config)
    payload = json.loads(path.read_text())
    assert payload["local_cpu"] is False
    assert payload["local_disk"] == f"file://{tmp_path / 'store'}/"
    assert payload["max_local_disk_size"] == 64.0
    assert payload["chunk_size"] == 256
    assert payload["extra_config"]["use_odirect"] is True


def test_lmcache_pass_args_use_baseline_arm_without_cache_salt(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="lm-pass",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        kv_connector_mode="lmcache",
        request_parallelism=8,
        force_max_tokens=True,
    )
    dataset_paths = {"biography": tmp_path / "biography.jsonl"}
    args = _build_lmcache_pass_args(config, dataset_paths, tmp_path / "cold.json", "cold")
    assert args[args.index("--arm") + 1] == "baseline_prefill"
    assert args[args.index("--request-parallelism") + 1] == "8"
    assert args[args.index("--output-json") + 1] == str(tmp_path / "cold.json")
    assert "--cache-base-url" not in args
    assert "cache_salt" not in " ".join(args)
    # forced-decode parity with the Cachet arm
    assert json.loads(args[args.index("--baseline-extra-body-json") + 1]) == {"ignore_eos": True}


def test_parse_args_wires_lmcache_options(tmp_path):
    config = parse_args(
        [
            "--benchmark-id",
            "lm-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--kv-connector-mode",
            "lmcache",
            "--lmcache-max-disk-gb",
            "100",
            "--lmcache-chunk-size",
            "512",
            "--lmcache-local-dir",
            "/local_disk0/lm",
        ]
    )
    assert config.kv_connector_mode == "lmcache"
    assert config.lmcache_max_disk_gb == 100.0
    assert config.lmcache_chunk_size == 512
    assert config.lmcache_local_dir == "/local_disk0/lm"


def test_parse_args_wires_registered_representative_workload_profile(tmp_path):
    argv = [
        "--benchmark-id",
        "vllm-8k-64-v1",
        "--benchmark-suite-id",
        "g6-vllm-8k-64",
        "--runtime-id",
        "test-physical-run-1",
        "--output-dir",
        str(tmp_path / "out"),
        "--local-root",
        str(tmp_path / "local"),
        "--model-revision",
        MODEL_REVISION,
        "--tokenizer-revision",
        MODEL_REVISION,
        "--max-tokens",
        "64",
        "--max-model-len",
        "8512",
        "--benchmark-repeats",
        "3",
        "--benchmark-force-max-tokens",
        "--benchmark-arm-spec-json",
        json.dumps(dict(representative_canary_matrix().runs[0].arm_spec)),
        "--benchmark-evidence-policy",
        "canary",
        "--representative-canary",
        "--representative-workload-profile",
        "vllm-8k-64-v1",
        "--benchmark-manifest-provenance-json",
        json.dumps({"input_tokens_target": 8192}),
    ]
    for dataset in SMOKE_DATASETS:
        argv.extend(["--dataset", f"{dataset}={tmp_path / f'{dataset}.jsonl'}"])

    config = parse_args(argv)

    assert config.is_representative_submission is True
    assert config.representative_workload_profile is not None
    assert config.representative_workload_profile.profile_id == "vllm-8k-64-v1"


def test_vllm_config_rejects_unknown_kv_connector_mode(tmp_path):
    with pytest.raises(ValueError, match="kv_connector_mode"):
        VLLMSmokeBenchmarkConfig(
            benchmark_id="bad-mode",
            output_dir=tmp_path / "out",
            local_root=tmp_path / "local",
            kv_connector_mode="redis",
        )


def test_parse_args_wires_system_prompt_position(tmp_path):
    base = ["--benchmark-id", "sp-1", "--output-dir", str(tmp_path / "out")]

    default_config = parse_args(base)
    assert default_config.system_prompt_position == "start"
    assert build_metadata(default_config)["benchmark_system_prompt_position"] == "start"

    end_config = parse_args(base + ["--system-prompt-position", "end"])
    assert end_config.system_prompt_position == "end"
    assert build_metadata(end_config)["benchmark_system_prompt_position"] == "end"


def test_vllm_config_rejects_unknown_system_prompt_position(tmp_path):
    with pytest.raises(ValueError, match="system_prompt_position"):
        VLLMSmokeBenchmarkConfig(
            benchmark_id="bad-position",
            output_dir=tmp_path / "out",
            local_root=tmp_path / "local",
            system_prompt_position="middle",
        )


def test_vllm_server_args_omit_data_parallel_by_default(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="dp-default",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )
    args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")
    assert "--data-parallel-size" not in args


def test_vllm_server_args_emit_data_parallel_size_when_set(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="dp-8",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        data_parallel_size=8,
    )
    args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")
    assert args[args.index("--data-parallel-size") + 1] == "8"


def test_vllm_server_args_include_payload_cache_budget(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-cache-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        payload_cache_max_bytes=4096,
    )

    args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")
    decoded = json.loads(args[args.index("--kv-transfer-config") + 1])

    assert decoded == document_kv_transfer_config_for_smoke(config)


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


def test_vllm_server_args_accept_quantized_model_and_kv_overrides(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="q4-q8-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_dtype="float16",
        model_quantization="bitsandbytes",
        kv_cache_dtype="fp8",
        attention_backend="TRITON_ATTN",
    )

    args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")

    assert args[args.index("--model") + 1] == "Qwen/Qwen3-4B-Instruct-2507"
    assert args[args.index("--dtype") + 1] == "float16"
    assert args[args.index("--quantization") + 1] == "bitsandbytes"
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8"
    assert args[args.index("--attention-backend") + 1] == "TRITON_ATTN"


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
    assert args[args.index("--repeats") + 1] == "1"
    assert args[args.index("--request-parallelism") + 1] == "1"
    assert "--server-usage" in args
    assert "--cache-base-url" not in args
    assert "--cache-runtime-prompt" not in args
    assert "--arm" not in args
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


def test_benchmark_runner_args_preserve_configured_hardware_target(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-g5",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
        hardware_target="aws-g5-a10g",
    )
    dataset_paths = {name: tmp_path / f"{name}.jsonl" for name in smoke_dataset_records()}

    args = build_benchmark_runner_args(config, dataset_paths)

    assert args[args.index("--hardware-target") + 1] == "aws-g5-a10g"


def test_benchmark_runner_args_include_parallelism_and_selected_arm(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-g5-baseline",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
        request_parallelism=8,
        benchmark_arms=("baseline_prefill",),
    )
    dataset_paths = {name: tmp_path / f"{name}.jsonl" for name in smoke_dataset_records()}

    args = build_benchmark_runner_args(config, dataset_paths)

    assert args[args.index("--request-parallelism") + 1] == "8"
    assert args[args.index("--arm") + 1] == "baseline_prefill"


def test_benchmark_runner_args_use_cold_hydrate_cache_prompt_for_prepared_datasets(tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
        benchmark_repeats=3,
        dataset_specs=specs,
    )

    args = build_benchmark_runner_args(config, parse_dataset_specs(specs))

    assert args[args.index("--cache-base-url") + 1] == "http://127.0.0.1:8123"
    assert args[args.index("--repeats") + 1] == "3"
    assert "--cache-runtime-prompt" not in args
    assert json.loads(args[args.index("--baseline-extra-body-json") + 1]) == {
        "cache_salt": BASELINE_PREFIX_CACHE_SALT
    }
    assert json.loads(args[args.index("--cache-extra-body-json") + 1]) == {
        "cache_salt": CACHE_PREFIX_CACHE_SALT
    }
    assert args[args.index("--prefix-cache-salt-mode") + 1] == "per_request"


def test_benchmark_runner_args_can_share_static_prefix_cache_for_prepared_datasets(tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-static-salt",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
        dataset_specs=specs,
        prefix_cache_salt_mode="static",
    )

    args = build_benchmark_runner_args(config, parse_dataset_specs(specs))

    assert args[args.index("--prefix-cache-salt-mode") + 1] == "static"


def test_benchmark_runner_args_can_force_max_tokens_for_latency_protocol(tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-force-256",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
        dataset_specs=specs,
        max_tokens=256,
        force_max_tokens=True,
    )

    args = build_benchmark_runner_args(config, parse_dataset_specs(specs))

    assert json.loads(args[args.index("--baseline-extra-body-json") + 1]) == {
        "cache_salt": BASELINE_PREFIX_CACHE_SALT,
        "ignore_eos": True,
    }
    assert json.loads(args[args.index("--cache-extra-body-json") + 1]) == {
        "cache_salt": CACHE_PREFIX_CACHE_SALT,
        "ignore_eos": True,
    }
    assert args[args.index("--max-tokens") + 1] == "256"


def test_benchmark_runner_args_forward_arbitrary_arms_evidence_and_provenance(tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    matrix = representative_canary_matrix()
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="representative-canary-8k-64",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
        dataset_specs=specs,
        max_tokens=64,
        force_max_tokens=True,
        benchmark_arm_specs=tuple(run.arm_spec for run in matrix.runs),
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        benchmark_evidence_policy="canary",
        benchmark_manifest_provenance={
            "model_revision": "a" * 40,
            "tokenizer_id": "Qwen/Qwen3-4B-Instruct-2507",
            "tokenizer_revision": "a" * 40,
            "engine_id": "vllm",
            "engine_version": VLLM_VERSION,
            "input_tokens_target": 8192,
            "hardware_fingerprint": "g6.8xlarge-l4-128g",
            "measurement_scopes": ["latency", "resource"],
        },
    )

    args = build_benchmark_runner_args(config, parse_dataset_specs(specs))

    arm_specs = [
        json.loads(args[index + 1])
        for index, value in enumerate(args)
        if value == "--arm-spec-json"
    ]
    assert [arm["arm_id"] for arm in arm_specs] == [
        run.arm_id for run in matrix.runs
    ]
    assert "--arm" not in args
    assert args[args.index("--evidence-policy") + 1] == "canary"
    assert args[args.index("--input-tokens-target") + 1] == "8192"
    assert args[args.index("--hardware-fingerprint") + 1] == "g6.8xlarge-l4-128g"
    assert [
        args[index + 1]
        for index, value in enumerate(args)
        if value == "--measurement-scope"
    ] == ["latency", "resource"]
    # Fixed decode length remains a canonical smoke option, not an arbitrary
    # per-arm payload override.
    assert json.loads(args[args.index("--baseline-extra-body-json") + 1])[
        "ignore_eos"
    ] is True
    assert all("extra_body" not in arm for arm in arm_specs)


def test_vllm_representative_provenance_binds_resolved_rope_geometry(
    tmp_path,
    monkeypatch,
):
    class PreRopeLayout:
        lora_id = "base"
        layout_version = "qwen3-prerope-v1"
        payload_axis_order = "token_major"
        block_size = 16
        key_position_encoding = "pre_rope"
        rope_theta = 1_000_000.0
        rope_rotary_dim = 128

    monkeypatch.setattr(
        public_vllm_smoke,
        "layout_for_model",
        lambda *_args, **_kwargs: PreRopeLayout(),
    )
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="vllm-prerope-provenance",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        **representative_vllm_kwargs(tmp_path),
    )

    assert config.benchmark_manifest_provenance["rope_theta"] == 1_000_000.0
    assert config.benchmark_manifest_provenance["rope_rotary_dim"] == 128


@pytest.mark.parametrize(
    ("representative_canary", "profile_id"),
    [(True, None), (False, "vllm-8k-64-v1")],
)
def test_vllm_representative_flag_and_profile_are_atomic(
    tmp_path,
    representative_canary,
    profile_id,
):
    with pytest.raises(ValueError, match="must be provided together"):
        VLLMSmokeBenchmarkConfig(
            benchmark_id="partial-representative-label",
            output_dir=tmp_path / "out",
            local_root=tmp_path / "local",
            representative_canary=representative_canary,
            representative_workload_profile=profile_id,
        )


@pytest.mark.parametrize("profile_id", ["vllm-8k-64-v1", "vllm-16k-256-v1"])
def test_vllm_representative_profiles_accept_only_the_registered_workloads(
    tmp_path,
    profile_id,
):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id=profile_id,
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        **representative_vllm_kwargs(tmp_path, profile_id=profile_id),
    )

    assert config.is_representative_submission is True
    assert config.representative_workload_profile is not None
    assert config.representative_workload_profile.profile_id == profile_id
    assert build_metadata(config)["representative_workload_profile"] == profile_id
    assert config.benchmark_manifest_provenance["package_revisions"]["cachet-kv"] == (
        f"wheel-sha256:{REPRESENTATIVE_WHEEL_SHA256}"
    )


def test_vllm_representative_provenance_requires_verified_wheel_sha256(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV)

    with pytest.raises(ValueError, match="verified Cachet wheel SHA-256"):
        VLLMSmokeBenchmarkConfig(
            benchmark_id="representative-missing-wheel-digest",
            output_dir=tmp_path / "out",
            local_root=tmp_path / "local",
            **representative_vllm_kwargs(tmp_path),
        )


def test_vllm_representative_provenance_rejects_wheel_identity_tampering(tmp_path):
    kwargs = representative_vllm_kwargs(tmp_path)
    kwargs["benchmark_manifest_provenance"] = {
        "input_tokens_target": 8192,
        "package_revisions": {
            "cachet-kv": f"wheel-sha256:{'e' * 64}",
        },
    }

    with pytest.raises(ValueError, match="package_revisions"):
        VLLMSmokeBenchmarkConfig(
            benchmark_id="representative-tampered-wheel-digest",
            output_dir=tmp_path / "out",
            local_root=tmp_path / "local",
            **kwargs,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"benchmark_suite_id": "forged-suite"}, "comparison group"),
        ({"benchmark_runtime_id": "unresolved"}, "must be resolved"),
        ({"max_tokens": 32}, "max_tokens"),
        ({"max_model_len": 4096}, "max_model_len"),
        ({"benchmark_repeats": 2}, "benchmark_repeats"),
        ({"request_parallelism": 2}, "request_parallelism"),
        ({"force_max_tokens": False}, "force_max_tokens"),
        ({"prefix_cache_salt_mode": "static"}, "prefix_cache_salt_mode"),
        ({"cache_runtime_prompt": True}, "cache_runtime_prompt"),
        ({"payload_cache_max_bytes": 1}, "payload_cache_max_bytes"),
        (
            {"benchmark_manifest_provenance": {"input_tokens_target": 8193}},
            "input_tokens_target",
        ),
    ],
)
def test_vllm_representative_profile_rejects_workload_drift(
    tmp_path,
    override,
    message,
):
    kwargs = representative_vllm_kwargs(tmp_path)
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        VLLMSmokeBenchmarkConfig(
            benchmark_id="drifted-representative-workload",
            output_dir=tmp_path / "out",
            local_root=tmp_path / "local",
            **kwargs,
        )


def test_vllm_fixed_arm_without_explicit_profile_remains_generic(tmp_path):
    fixed_arm = representative_canary_matrix().runs[0].arm_spec
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="generic-fixed-arm",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "custom-local-root",
        benchmark_arm_specs=(fixed_arm,),
    )

    assert config.is_representative_submission is False
    assert config.representative_workload_profile is None


def test_vllm_representative_prompt_budget_requires_actual_multi_document_input(
    tmp_path,
):
    dataset_path = tmp_path / "hotpotqa.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "dataset": "hotpotqa",
                "example_id": "single-document",
                "query": "Who?",
                "documents": [
                    {"document_id": "doc-1", "text": "Only one document."}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    kwargs = representative_vllm_kwargs(tmp_path)
    kwargs.update(
        {
            "dataset_specs": (f"hotpotqa={dataset_path}",),
            "allow_dataset_subset": True,
        }
    )
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="single-document-representative",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        **kwargs,
    )

    with pytest.raises(ValueError, match="at least one prepared multi-document"):
        build_prompt_token_budget_rows(config, {"hotpotqa": dataset_path})


def test_vllm_smoke_config_rejects_legacy_and_arbitrary_arm_mix(tmp_path):
    matrix = representative_canary_matrix()

    with pytest.raises(ValueError, match="mutually exclusive"):
        VLLMSmokeBenchmarkConfig(
            benchmark_id="mixed-arms",
            output_dir=tmp_path / "out",
            local_root=tmp_path / "local",
            benchmark_arms=("baseline_prefill",),
            benchmark_arm_specs=(matrix.runs[0].arm_spec,),
        )


@pytest.mark.parametrize(
    ("arm_index", "cache_method", "segment_per_document"),
    (
        (1, "full_prefix_prefill", False),
        (2, "vanilla_prefill", True),
    ),
)
def test_representative_generated_handoff_costs_are_recorded_on_matching_arm(
    tmp_path,
    arm_index,
    cache_method,
    segment_per_document,
):
    kwargs = representative_vllm_kwargs(tmp_path, arm_index=arm_index)
    kwargs["handoff_generation"] = VLLMPreparedHandoffGenerationConfig(
        generator_factory="module:factory",
        output_dir=tmp_path / "handoffs",
        benchmark_handoff_segment_per_document=segment_per_document,
        cache_method=cache_method,
    )
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id=f"{cache_method}-costs",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        **kwargs,
    )
    config.output_dir.mkdir(parents=True)
    config.prepared_handoff_generation_path.write_text(
        json.dumps(
            {
                "artifact_generation_seconds": 12.5,
                "artifact_payload_bytes": 1000,
                "artifact_storage_bytes": 1200,
            }
        ),
        encoding="utf-8",
    )

    enriched = public_vllm_smoke._config_with_generated_handoff_offline_costs(
        config
    )

    assert enriched.benchmark_arm_specs[0]["offline_costs"] == {
        "artifact_generation_seconds": 12.5,
        "artifact_bytes": 1200,
    }


@pytest.mark.parametrize(
    "offline_costs",
    (
        {},
        {"artifact_generation_seconds": 12.5},
        {
            "artifact_generation_seconds": 12.5,
            "artifact_bytes": 1200,
            "training_seconds": 1.0,
        },
        {"artifact_generation_seconds": -0.1, "artifact_bytes": 1200},
        {"artifact_generation_seconds": float("inf"), "artifact_bytes": 1200},
        {"artifact_generation_seconds": float("nan"), "artifact_bytes": 1200},
        {"artifact_generation_seconds": True, "artifact_bytes": 1200},
        {"artifact_generation_seconds": 12.5, "artifact_bytes": -1},
        {"artifact_generation_seconds": 12.5, "artifact_bytes": False},
    ),
)
def test_fixed_representative_arm_rejects_invalid_offline_costs(
    tmp_path,
    offline_costs,
):
    spec = dict(
        representative_canary_matrix().run_for_arm(
            "document_kv_cache:full_prefix_prefill"
        ).arm_spec
    )
    spec["offline_costs"] = offline_costs

    assert not public_vllm_smoke._is_fixed_representative_arm_spec(spec)
    kwargs = representative_vllm_kwargs(tmp_path, arm_index=1)
    kwargs["benchmark_arm_specs"] = (spec,)
    with pytest.raises(ValueError):
        VLLMSmokeBenchmarkConfig(
            benchmark_id="invalid-offline-costs",
            output_dir=tmp_path / "out",
            local_root=tmp_path / "local",
            **kwargs,
        )


def test_fixed_representative_arm_rejects_tampered_identity_with_offline_costs(
    tmp_path,
):
    spec = dict(
        representative_canary_matrix().run_for_arm(
            "document_kv_cache:vanilla_prefill"
        ).arm_spec
    )
    spec["offline_costs"] = {
        "artifact_generation_seconds": 12.5,
        "artifact_bytes": 1200,
    }
    spec["physical_transform_version"] = "tampered"

    assert not public_vllm_smoke._is_fixed_representative_arm_spec(spec)
    kwargs = representative_vllm_kwargs(tmp_path, arm_index=2)
    kwargs["benchmark_arm_specs"] = (spec,)
    with pytest.raises(ValueError, match="benchmark_arm_specs"):
        VLLMSmokeBenchmarkConfig(
            benchmark_id="tampered-arm",
            output_dir=tmp_path / "out",
            local_root=tmp_path / "local",
            **kwargs,
        )


def test_prepared_handoff_coverage_validates_every_arbitrary_cache_arm(
    tmp_path,
    monkeypatch,
):
    matrix = representative_canary_matrix()
    dataset_path = tmp_path / "hotpotqa.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "dataset": "hotpotqa",
                "example_id": "example-1",
                "query": "Who?",
                "documents": [
                    {"document_id": "doc-1", "text": "one"},
                    {"document_id": "doc-2", "text": "two"},
                ],
                "arm_kv_transfer_params": {
                    matrix.runs[1].arm_id: {
                        "document_kv.request_id": "full",
                        "document_kv.handoff_json": "/tmp/full.json",
                    },
                    matrix.runs[2].arm_id: {
                        "document_kv.request_id": "vanilla",
                        "document_kv.handoff_json": "/tmp/vanilla.json",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="n-way-coverage",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=(f"hotpotqa={dataset_path}",),
        allow_dataset_subset=True,
        benchmark_arm_specs=tuple(run.arm_spec for run in matrix.runs),
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
    )
    observed = []

    def fake_issue(example, *, params, arm_id):
        observed.append((example.example_id, arm_id, params["document_kv.request_id"]))
        return None

    monkeypatch.setattr(
        public_vllm_smoke,
        "_prepared_handoff_reference_issue",
        fake_issue,
    )

    record = prepared_benchmark_handoff_coverage_record(
        config,
        {"hotpotqa": dataset_path},
    )

    assert record["ok"] is True
    assert record["cache_arm_ids"] == [matrix.runs[1].arm_id, matrix.runs[2].arm_id]
    assert observed == [
        ("example-1", matrix.runs[1].arm_id, "full"),
        ("example-1", matrix.runs[2].arm_id, "vanilla"),
    ]


def test_upstream_cache_arm_does_not_require_cachet_handoff_metadata(tmp_path):
    dataset_path = tmp_path / "hotpotqa.jsonl"
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="upstream-only",
        output_dir=tmp_path / "out",
        dataset_specs=(f"hotpotqa={dataset_path}",),
        allow_dataset_subset=True,
        benchmark_arm_specs=(
            {
                "arm_id": "upstream:author",
                "uses_cache": True,
                "description": "Author implementation",
                "cache_method": "author_method",
                "implementation_kind": "upstream",
                "requires_cachet_handoff": False,
            },
        ),
    )

    assert config.runs_document_kv_cache_arm is False
    assert config.requires_prepared_handoff_metadata is False


def test_benchmark_runner_args_can_use_runtime_cache_prompt_for_prepared_datasets(tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-runtime-prompt",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
        dataset_specs=specs,
        cache_runtime_prompt=True,
    )

    args = build_benchmark_runner_args(config, parse_dataset_specs(specs))

    assert args[args.index("--cache-base-url") + 1] == "http://127.0.0.1:8123"
    assert "--cache-runtime-prompt" in args
    assert args.index("--cache-runtime-prompt") < args.index("--dataset")


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


def test_validate_prompt_token_budget_enforces_exact_manifest_target(monkeypatch, tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="canary-8k",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        benchmark_manifest_provenance={
            "tokenizer_id": HF_MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "input_tokens_target": 8192,
        },
    )
    dataset_paths = {dataset: tmp_path / f"{dataset}.jsonl" for dataset in SMOKE_DATASETS}
    monkeypatch.setattr(
        public_vllm_smoke,
        "build_prompt_token_budget_rows",
        lambda cfg, paths: (
            {"dataset": "hotpotqa", "example_id": "hotpot-1", "prompt": "prompt"},
        ),
    )
    captured = {}

    def probe(*args, **kwargs):
        captured.update(kwargs)
        mismatch = {
            "dataset": "hotpotqa",
            "example_id": "hotpot-1",
            "logical_prompt_sha256": "a" * 64,
            "prompt_tokens": 8191,
        }
        return {
            "ok": True,
            "rows": [mismatch],
            "over_budget": [],
            "token_count_mismatches": [mismatch],
        }

    monkeypatch.setattr(public_vllm_smoke, "run_prompt_token_budget_probe", probe)

    with pytest.raises(ValueError, match="exact logical input token target"):
        public_vllm_smoke.validate_prompt_token_budget(config, dataset_paths)

    assert captured["tokenizer_id"] == HF_MODEL_ID
    assert captured["tokenizer_revision"] == MODEL_REVISION
    assert captured["add_special_tokens"] is False
    assert captured["expected_prompt_tokens"] == 8192
    record = json.loads(config.prompt_token_budget_path.read_text(encoding="utf-8"))
    assert record["token_count_mismatches"][0]["prompt_tokens"] == 8191


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
        model_revision=MODEL_REVISION,
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


def test_run_prompt_token_budget_probe_pins_tokenizer_and_records_contract(monkeypatch, tmp_path):
    captured = {}
    probe_record = {
        "rows": [
            {
                "dataset": "hotpotqa",
                "example_id": "hotpot-1",
                "logical_prompt_sha256": "a" * 64,
                "prompt_tokens": 8192,
            }
        ],
        "over_budget": [],
        "token_count_mismatches": [],
    }

    def completed_run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=json.dumps(probe_record),
            stderr="",
        )

    monkeypatch.setattr(public_vllm_smoke.subprocess, "run", completed_run)

    record = run_prompt_token_budget_probe(
        tmp_path / "python",
        tmp_path / "input.jsonl",
        model_id=HF_MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=HF_MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        add_special_tokens=False,
        expected_prompt_tokens=8192,
        max_model_len=16384,
        max_tokens=64,
        timeout_seconds=3,
    )

    assert captured["argv"][3:9] == [
        HF_MODEL_ID,
        MODEL_REVISION,
        HF_MODEL_ID,
        MODEL_REVISION,
        "false",
        "8192",
    ]
    assert record["model"] == {
        "model_id": HF_MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    assert record["tokenizer"] == {
        "tokenizer_id": HF_MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "add_special_tokens": False,
    }
    assert record["expected_prompt_tokens"] == 8192
    assert record["rows"][0]["logical_prompt_sha256"] == "a" * 64


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
    subset = parse_dataset_specs(
        (f"biography={tmp_path / 'biography.jsonl'}",),
        allow_subset=True,
    )
    assert subset == {"biography": tmp_path / "biography.jsonl"}
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
            require_artifact_contract=False,
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


def test_prepare_generated_benchmark_handoffs_limits_enriched_inputs(tmp_path, monkeypatch):
    module = ModuleType("cachet_test_vllm_limited_handoff_generator")
    module.build_generator = OneTokenBenchmarkKVGenerator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=False)
    for dataset, path in dataset_paths.items():
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "dataset": dataset,
                        "example_id": f"{dataset}-2",
                        "query": "Who is described?",
                        "expected_answer": "Grace Hopper",
                        "documents": [{"document_id": "grace", "text": "Grace Hopper biography"}],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-generated-limit-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
        handoff_generation=VLLMPreparedHandoffGenerationConfig(
            generator_factory=f"{module.__name__}:build_generator",
            output_dir=tmp_path / "generated-handoffs",
            dtype="bfloat16",
            align_bytes=1,
            limit=1,
            require_artifact_contract=False,
        ),
    )

    generated_paths = prepare_generated_benchmark_handoffs(config, dataset_paths)

    generation = json.loads(config.prepared_handoff_generation_path.read_text(encoding="utf-8"))
    assert generation["datasets"]["biography"]["entries"] == 1
    assert generation["datasets"]["biography"]["enriched_rows"] == 1
    assert sum(1 for _ in generated_paths["biography"].open(encoding="utf-8")) == 1
    limited_input = Path(generation["datasets"]["biography"]["generation_input_jsonl"])
    assert sum(1 for _ in limited_input.open(encoding="utf-8")) == 1


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
            require_artifact_contract=False,
        ),
    )

    prepare_generated_benchmark_handoffs(config, dataset_paths)

    assert released_after_generator_collectable == [True]


def test_prewarm_cache_prefixes_posts_kv_aware_prefix_prompts(tmp_path, monkeypatch):
    dataset_paths = prepared_dataset_paths(tmp_path)
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prewarm-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
        prewarm_cache_prefix=True,
        prefix_cache_salt_mode="static",
        timeout_seconds=12.5,
    )
    calls = []

    def fake_post_json(url, body, *, timeout_seconds):
        calls.append((url, body, timeout_seconds))
        return {"usage": {"prompt_tokens": 12, "completion_tokens": 1}}

    monkeypatch.setattr(public_vllm_smoke, "_post_json", fake_post_json)

    prewarm_cache_prefixes(config, dataset_paths)

    assert len(calls) == len(SMOKE_DATASETS)
    first_url, first_body, first_timeout = calls[0]
    assert first_url == f"{config.server_base_url}/v1/completions"
    assert first_timeout == 12.5
    assert first_body["model"] == SERVED_MODEL_NAME
    assert first_body["max_tokens"] == 1
    assert first_body["cache_salt"] == CACHE_PREFIX_CACHE_SALT
    assert first_body["request_id"].startswith("cachet-prewarm:prewarm-1:")
    assert "Ada Lovelace biography" in first_body["prompt"]
    assert first_body["prompt"].endswith("\n\nCache warmup.")
    assert first_body["kv_transfer_params"][DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM] == "logical"
    assert DOCUMENT_KV_HANDOFF_JSON_PARAM in first_body["kv_transfer_params"]
    record = json.loads(config.prewarm_cache_prefix_path.read_text(encoding="utf-8"))
    assert record["ok"] is True
    assert record["row_count"] == len(SMOKE_DATASETS)
    assert record["rows"][0]["prompt_tokens"] == 12


def test_prepare_generated_benchmark_handoffs_uses_vllm_venv_when_available(tmp_path, monkeypatch):
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=False)
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    representative_kwargs = representative_vllm_kwargs(tmp_path, arm_index=1)
    representative_kwargs["dataset_specs"] = specs
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-generated-venv",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_generation=VLLMPreparedHandoffGenerationConfig(
            generator_factory="module:factory",
            output_dir=tmp_path / "generated-handoffs",
            dtype="bfloat16",
            align_bytes=1,
            timeout_seconds=1234.0,
            limit=2,
        ),
        **representative_kwargs,
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
        assert input_payload["representative_canary"] is True
        assert input_payload["representative_workload_profile"] == "vllm-8k-64-v1"
        assert input_payload["max_tokens"] == 64
        assert input_payload["benchmark_repeats"] == 3
        assert input_payload["force_max_tokens"] is True
        assert input_payload["benchmark_manifest_provenance"][
            "input_tokens_target"
        ] == 8192
        assert len(input_payload["benchmark_arm_specs"]) == 1
        assert input_payload["handoff_generation"]["generator_factory"] == "module:factory"
        assert input_payload["handoff_generation"]["timeout_seconds"] == 1234.0
        assert input_payload["handoff_generation"]["limit"] == 2
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

    with pytest.raises(ValueError, match="Cachet per-arm or legacy kv_transfer_params"):
        validate_prepared_benchmark_handoffs(config, dataset_paths)

    record = json.loads(config.prepared_handoff_coverage_path.read_text(encoding="utf-8"))
    assert record["ok"] is False
    assert record["examples_with_kv_transfer_params"] == 0
    assert record["examples_with_loadable_handoff_references"] == 0
    assert record["missing_kv_transfer_params"] == [
        f"{dataset}/{dataset}-1:{CACHE_REUSE_ARM}"
        for dataset in SMOKE_DATASETS
    ]
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
            "arm_id": CACHE_REUSE_ARM,
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


def test_validate_prepared_benchmark_handoffs_skips_baseline_only_prepared_run(tmp_path):
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=False)
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-baseline-only",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
        benchmark_arms=("baseline_prefill",),
    )

    assert config.requires_prepared_handoff_metadata is False
    assert validate_prepared_benchmark_handoffs(config, dataset_paths) is None
    assert not config.prepared_handoff_coverage_path.exists()


def test_multi_mode_requires_prepared_handoff_metadata_for_baseline_only(tmp_path):
    dataset_paths = prepared_dataset_paths(tmp_path, include_handoffs=False)
    specs = tuple(f"{dataset}={path}" for dataset, path in dataset_paths.items())
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="prepared-multi-baseline-only",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
        benchmark_arms=("baseline_prefill",),
        kv_connector_mode="multi",
    )

    # The hybrid multi-turn probe always injects a Cachet turn-1 handoff, so a prepared
    # multi run must validate handoff coverage even with baseline-only client arms
    # (otherwise the multi-turn artifact is emitted without loadable handoffs).
    assert config.requires_prepared_handoff_metadata is True
    with pytest.raises(ValueError, match="kv_transfer_params"):
        validate_prepared_benchmark_handoffs(config, dataset_paths)


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
    assert metadata["model_dtype"] == "bfloat16"
    assert metadata["model_quantization"] is None
    assert metadata["kv_cache_dtype"] is None
    assert metadata["attention_backend"] is None
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
    assert metadata["benchmark_repeats"] == 1
    assert metadata["request_parallelism"] == 1
    assert metadata["benchmark_arms"] == []
    assert metadata["document_kv_package_install_spec"] == str(REPO_ROOT)
    assert metadata["dependency_override_constraints"] == dependency_override_constraints()
    assert metadata["vllm_server_env_overrides"] == {
        "PYTHONUNBUFFERED": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    assert metadata["document_kv_connector_telemetry_local_path"] == str(config.connector_telemetry_path)
    assert metadata["document_kv_connector_telemetry_path"] == str(config.connector_telemetry_copy_path)
    assert metadata["runtime_telemetry_local_path"] == str(config.runtime_telemetry_path)
    assert metadata["runtime_telemetry_path"] == str(config.runtime_telemetry_copy_path)
    assert metadata["runtime_telemetry_interval_seconds"] == 1.0
    assert metadata["vllm_kv_transfer_config"] == document_kv_transfer_config_for_smoke(config)


def test_metadata_records_payload_cache_budget(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-cache-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        payload_cache_max_bytes=4096,
    )

    metadata = build_metadata(config)

    assert metadata["vllm_kv_transfer_config"] == document_kv_transfer_config_for_smoke(config)
    assert metadata["vllm_kv_transfer_config"] == document_kv_transfer_config(
        payload_cache_max_bytes=4096,
        telemetry_jsonl=str(config.connector_telemetry_path),
    )


@pytest.mark.parametrize("mode", ["lmcache", "multi"])
def test_metadata_records_launched_transfer_config_for_connector_mode(tmp_path, mode):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-mode-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        kv_connector_mode=mode,
    )

    metadata = build_metadata(config)

    # Provenance must describe the connector the server is actually launched with
    # (kv_transfer_config_json tracks kv_connector_mode), not the Cachet config, so
    # provenance-driven reruns reproduce the same LMCache/MultiConnector server.
    assert metadata["vllm_kv_transfer_config"] == json.loads(kv_transfer_config_json(config))
    assert metadata["vllm_kv_transfer_config"] != document_kv_transfer_config_for_smoke(config)


def test_server_env_defaults_q4_handoff_generator_to_matching_transformers_config(tmp_path, monkeypatch):
    dataset_specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    for name in (
        CACHET_TRANSFORMERS_MODEL_ID_ENV,
        CACHET_TRANSFORMERS_TOKENIZER_ID_ENV,
        CACHET_TRANSFORMERS_TORCH_DTYPE_ENV,
        CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV,
        CACHET_TRANSFORMERS_QUANTIZATION_ENV,
        CACHET_TRANSFORMERS_DEVICE_MAP_ENV,
        CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-q4-handoff",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_dtype="bfloat16",
        model_quantization="bitsandbytes",
        dataset_specs=dataset_specs,
        handoff_generation=VLLMPreparedHandoffGenerationConfig(
            generator_factory="document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator",
            output_dir=tmp_path / "generated-handoffs",
            dtype="fp8_e5m2",
        ),
    )

    env = server_env(config)

    assert env[CACHET_TRANSFORMERS_MODEL_ID_ENV] == "Qwen/Qwen3-4B-Instruct-2507"
    assert env[CACHET_TRANSFORMERS_TOKENIZER_ID_ENV] == "Qwen/Qwen3-4B-Instruct-2507"
    assert env[CACHET_TRANSFORMERS_TORCH_DTYPE_ENV] == "bfloat16"
    assert env[CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV] == "true"
    assert env[CACHET_TRANSFORMERS_QUANTIZATION_ENV] == "bitsandbytes-4bit"
    assert env[CACHET_TRANSFORMERS_DEVICE_MAP_ENV] == "auto"
    assert json.loads(env[CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV]) == {
        "bnb_4bit_compute_dtype": "bfloat16",
    }


def test_representative_server_env_pins_generator_identity_and_page_cache(
    tmp_path,
    monkeypatch,
):
    for name in (
        CACHET_TRANSFORMERS_MODEL_ID_ENV,
        CACHET_TRANSFORMERS_MODEL_REVISION_ENV,
        CACHET_TRANSFORMERS_TOKENIZER_ID_ENV,
        CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV,
        "DOCUMENT_KV_EVICT_PAGE_CACHE",
    ):
        monkeypatch.delenv(name, raising=False)
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="representative-generator-env",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_generation=VLLMPreparedHandoffGenerationConfig(
            generator_factory=(
                "document_kv_cache.transformers_generator:"
                "build_transformers_kv_chunk_generator"
            ),
            output_dir=tmp_path / "generated-handoffs",
        ),
        **representative_vllm_kwargs(tmp_path, arm_index=1),
    )

    env = server_env(config)
    assert env[CACHET_TRANSFORMERS_MODEL_REVISION_ENV] == MODEL_REVISION
    assert env[CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV] == MODEL_REVISION
    assert env["DOCUMENT_KV_EVICT_PAGE_CACHE"] == "1"

    monkeypatch.setenv(CACHET_TRANSFORMERS_MODEL_REVISION_ENV, "wrong-revision")
    with pytest.raises(ValueError, match=CACHET_TRANSFORMERS_MODEL_REVISION_ENV):
        server_env(config)


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
        assert 'md.version("cachet-kv")' in argv[2]
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


def test_installed_versions_uses_cachet_distribution_name(monkeypatch, tmp_path):
    requested_packages = []

    def fake_installed_package_version(python_executable, package_name):
        requested_packages.append((python_executable, package_name))
        return f"{package_name}-version"

    monkeypatch.setattr(public_vllm_smoke, "installed_package_version", fake_installed_package_version)

    python_executable = tmp_path / "venv" / "bin" / "python"
    versions = public_vllm_smoke.installed_versions(python_executable)

    assert versions["document_kv_cache_version_installed"] == "cachet-kv-version"
    assert requested_packages == [
        (python_executable, "vllm"),
        (python_executable, "cachet-kv"),
        (python_executable, "transformers"),
        (python_executable, "torch"),
        (python_executable, "opencv-python-headless"),
    ]


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
    assert metadata["prewarm_cache_prefix"] is False
    assert metadata["cache_runtime_prompt"] is False
    assert metadata["cache_measurement_protocol"] == "cold_disk_to_gpu_hydrate"
    assert metadata["cache_prompt_text_mode"] == "logical"
    assert metadata["prefix_cache_isolation"] == {
        "baseline_cache_salt": BASELINE_PREFIX_CACHE_SALT,
        "cache_cache_salt": CACHE_PREFIX_CACHE_SALT,
        "cache_salt_mode": "per_request",
    }
    assert metadata["requires_kv_transfer_params"] is True
    assert metadata["generates_prepared_handoffs"] is False
    assert metadata["benchmark_handoff_generation"] is None
    assert metadata["max_model_len"] == 32768
    assert metadata["max_num_seqs"] == 8
    assert metadata["gpu_memory_utilization"] == 0.72
    assert metadata["document_kv_package_install_spec"] == str(REPO_ROOT)


def test_metadata_records_runtime_cache_prompt_mode(tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="full-v1-runtime",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
        cache_runtime_prompt=True,
    )

    metadata = build_metadata(config)

    assert metadata["cache_runtime_prompt"] is True
    assert metadata["cache_prompt_text_mode"] == "runtime"
    assert metadata["cache_measurement_protocol"] == "cold_disk_to_gpu_hydrate"


def test_metadata_marks_baseline_only_prepared_run_as_not_requiring_handoffs(tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="baseline-v1-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=specs,
        benchmark_arms=("baseline_prefill",),
    )

    metadata = build_metadata(config)

    assert metadata["dataset_source"] == "prepared"
    assert metadata["requires_kv_transfer_params"] is False
    assert metadata["benchmark_arms"] == ["baseline_prefill"]


def test_parse_args_builds_config_with_overrides(tmp_path):
    specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    config = parse_args(
        [
            "--benchmark-id",
            "smoke-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--model-id",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--model-dtype",
            "float16",
            "--model-quantization",
            "bitsandbytes",
            "--kv-cache-dtype",
            "fp8_e5m2",
            "--attention-backend",
            "TRITON_ATTN",
            "--local-root",
            str(tmp_path / "local"),
            "--max-tokens",
            "16",
            "--benchmark-force-max-tokens",
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
            "--hardware-target",
            "aws-g5-a10g",
            "--benchmark-repeats",
            "3",
            "--request-parallelism",
            "8",
            "--runtime-telemetry-interval-seconds",
            "2.5",
            "--benchmark-prewarm-cache-prefix",
            "--benchmark-cache-runtime-prompt",
            "--benchmark-prefix-cache-salt-mode",
            "static",
            "--benchmark-arm",
            "baseline_prefill",
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
            "--benchmark-handoff-generation-timeout-seconds",
            "1234",
            "--benchmark-handoff-limit",
            "2",
            *sum((["--dataset", spec] for spec in specs), []),
        ]
    )

    assert config == VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_dtype="float16",
        model_quantization="bitsandbytes",
        kv_cache_dtype="fp8_e5m2",
        attention_backend="TRITON_ATTN",
        local_root=tmp_path / "local",
        max_tokens=16,
        force_max_tokens=True,
        timeout_seconds=12.5,
        import_probe_timeout_seconds=9,
        server_start_timeout_seconds=30,
        server_host="0.0.0.0",
        server_port=8123,
        client_host="127.0.0.1",
        max_model_len=32768,
        max_num_seqs=8,
        gpu_memory_utilization=0.72,
        benchmark_repeats=3,
        request_parallelism=8,
        runtime_telemetry_interval_seconds=2.5,
        benchmark_arms=("baseline_prefill",),
        prewarm_cache_prefix=True,
        cache_runtime_prompt=True,
        prefix_cache_salt_mode="static",
        hardware_target="aws-g5-a10g",
        dataset_specs=specs,
        package_install_spec=str(tmp_path / "cachet.whl"),
        handoff_generation=VLLMPreparedHandoffGenerationConfig(
            generator_factory="document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator",
            output_dir=Path("/dbfs/tmp/cachet/generated-handoffs"),
            dtype="bfloat16",
            align_bytes=1,
            timeout_seconds=1234.0,
            limit=2,
        ),
    )


def test_parse_args_wires_strict_method_handoff_contract(tmp_path):
    specs = tuple(
        f"{dataset}={tmp_path / f'{dataset}.jsonl'}"
        for dataset in SMOKE_DATASETS
    )
    identity = stored_post_rope_runtime_identity()

    config = parse_args(
        [
            "--benchmark-id",
            "method-handoff-live",
            "--output-dir",
            str(tmp_path / "out"),
            "--model-revision",
            MODEL_REVISION,
            "--tokenizer-revision",
            MODEL_REVISION,
            "--runtime-identity-json",
            json.dumps(identity.to_record()),
            "--benchmark-handoff-generator-factory",
            "document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator",
            "--benchmark-handoff-chunk-per-document",
            "--benchmark-handoff-cache-method",
            "vanilla_prefill",
            "--benchmark-cache-runtime-prompt",
            *sum((["--dataset", spec] for spec in specs), []),
        ]
    )

    assert config.model_revision == MODEL_REVISION
    assert config.tokenizer_revision == MODEL_REVISION
    assert config.runtime_identity == identity
    assert config.handoff_generation is not None
    assert config.handoff_generation.cache_method == "vanilla_prefill"
    assert (
        config.handoff_generation.benchmark_handoff_segment_per_document
        is True
    )
    assert config.handoff_generation.require_artifact_contract is True
    runner_args = build_benchmark_runner_args(
        config,
        parse_dataset_specs(specs),
    )
    assert "add_special_tokens" not in json.loads(
        runner_args[runner_args.index("--baseline-extra-body-json") + 1]
    )
    assert "add_special_tokens" not in json.loads(
        runner_args[runner_args.index("--cache-extra-body-json") + 1]
    )


def test_handoff_artifact_contract_is_strict_by_default_with_legacy_opt_out(tmp_path):
    assert VLLMPreparedHandoffGenerationConfig(
        generator_factory="module:factory",
        output_dir=tmp_path / "strict",
    ).require_artifact_contract is True
    legacy = VLLMPreparedHandoffGenerationConfig(
        generator_factory="module:factory",
        output_dir=tmp_path / "legacy",
        require_artifact_contract=False,
    )
    specs = tuple(
        f"{dataset}={tmp_path / f'{dataset}.jsonl'}"
        for dataset in SMOKE_DATASETS
    )

    with pytest.raises(ValueError, match="canary and publication"):
        VLLMSmokeBenchmarkConfig(
            benchmark_id="legacy-canary",
            output_dir=tmp_path / "out",
            dataset_specs=specs,
            benchmark_evidence_policy="canary",
            handoff_generation=legacy,
        )

    parsed = parse_args(
        [
            "--benchmark-id",
            "legacy-debug",
            "--output-dir",
            str(tmp_path / "parsed"),
            "--benchmark-handoff-generator-factory",
            "module:factory",
            "--benchmark-handoff-allow-legacy-artifact-contract",
            *sum((["--dataset", spec] for spec in specs), []),
        ]
    )
    assert parsed.handoff_generation is not None
    assert parsed.handoff_generation.require_artifact_contract is False


def test_vllm_smoke_config_validates_before_runtime_setup(tmp_path):
    dataset_specs = tuple(f"{dataset}={tmp_path / f'{dataset}.jsonl'}" for dataset in SMOKE_DATASETS)
    with pytest.raises(ValueError, match="benchmark_handoff_timeout_seconds"):
        VLLMPreparedHandoffGenerationConfig(
            generator_factory="module:factory",
            output_dir=tmp_path / "generated-handoffs",
            timeout_seconds=0,
        )
    with pytest.raises(ValueError, match="benchmark_handoff_limit"):
        VLLMPreparedHandoffGenerationConfig(
            generator_factory="module:factory",
            output_dir=tmp_path / "generated-handoffs",
            limit=-1,
        )
    with pytest.raises(ValueError, match="vanilla_prefill.*segment per document"):
        VLLMPreparedHandoffGenerationConfig(
            generator_factory="module:factory",
            output_dir=tmp_path / "generated-handoffs",
            cache_method="vanilla_prefill",
        )
    with pytest.raises(ValueError, match="full_prefix_prefill.*full-prefix segment"):
        VLLMPreparedHandoffGenerationConfig(
            generator_factory="module:factory",
            output_dir=tmp_path / "generated-handoffs",
            cache_method="full_prefix_prefill",
            benchmark_handoff_segment_per_document=True,
        )
    invalid_cases = [
        ({"benchmark_id": ""}, "benchmark_id must be non-empty"),
        ({"model_id": ""}, "model_id must be non-empty"),
        ({"model_dtype": ""}, "model_dtype must be non-empty"),
        ({"model_quantization": ""}, "model_quantization must be non-empty"),
        ({"kv_cache_dtype": ""}, "kv_cache_dtype must be non-empty"),
        (
            {"hardware_target": "aws-g5-a10g", "kv_cache_dtype": "fp8"},
            "fp8_e5m2",
        ),
        (
            {"hardware_target": "aws-g5-a10g", "kv_cache_dtype": "fp8_e4m3"},
            "fp8_e5m2",
        ),
        ({"attention_backend": ""}, "attention_backend must be non-empty"),
        ({"max_tokens": 0}, "max_tokens must be positive"),
        ({"force_max_tokens": "yes"}, "force_max_tokens must be a boolean"),
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
        ({"benchmark_repeats": 0}, "benchmark_repeats must be a positive integer"),
        ({"request_parallelism": 0}, "request_parallelism must be a positive integer"),
        ({"runtime_telemetry_interval_seconds": 0}, "runtime_telemetry_interval_seconds must be positive"),
        ({"benchmark_arms": ("unknown",)}, "Unknown benchmark arms"),
        ({"allow_dataset_subset": "yes"}, "allow_dataset_subset must be a boolean"),
        ({"prefix_cache_salt_mode": "dynamic"}, "prefix_cache_salt_mode"),
        (
            {"prewarm_cache_prefix": True},
            "benchmark_prewarm_cache_prefix requires prepared dataset specs",
        ),
        (
                {
                    "prewarm_cache_prefix": True,
                    "prefix_cache_salt_mode": "per_request",
                    "dataset_specs": dataset_specs,
                },
            "requires prefix_cache_salt_mode='static'",
        ),
        (
            {"cache_runtime_prompt": True},
            "benchmark_cache_runtime_prompt requires prepared dataset specs",
        ),
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

    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        hardware_target="aws-g5-a10g",
        kv_cache_dtype="fp8_e5m2",
    )
    assert config.kv_cache_dtype == "fp8_e5m2"

    subset_config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-subset-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=(f"biography={tmp_path / 'biography.jsonl'}",),
        allow_dataset_subset=True,
    )
    assert benchmark_dataset_paths(subset_config) == {"biography": tmp_path / "biography.jsonl"}


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
    class FakeRuntimeTelemetrySampler:
        def __init__(self, output_path, *, process_pid, interval_seconds):
            self.output_path = output_path
            calls.append(("runtime_telemetry_init", output_path, process_pid, interval_seconds))

        def start(self):
            calls.append(("runtime_telemetry_start", self.output_path))
            return self

        def stop(self):
            calls.append(("runtime_telemetry_stop", self.output_path))

    monkeypatch.setattr(public_vllm_smoke, "RuntimeTelemetrySampler", FakeRuntimeTelemetrySampler)
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
        ("runtime_telemetry_init", config.runtime_telemetry_path, None, 1.0),
        ("runtime_telemetry_start", config.runtime_telemetry_path),
        ("wait_for_server", fake_server, config.server_log_path, "http://127.0.0.1:8123", 480.0),
        ("copy", config.server_log_path, config.server_log_copy_path),
        ("run", build_benchmark_runner_args(config, dataset_paths)),
        ("terminate", fake_server),
        ("runtime_telemetry_stop", config.runtime_telemetry_path),
        ("copy", config.server_log_path, config.server_log_copy_path),
        ("copy", config.connector_telemetry_path, config.connector_telemetry_copy_path),
        ("copy", config.runtime_telemetry_path, config.runtime_telemetry_copy_path),
    ]
    metadata = build_metadata(config)
    assert metadata["server_base_url"] == "http://127.0.0.1:8123"
    assert metadata["hf_home"] == str(tmp_path / "local" / "hf-cache")


def test_run_lmcache_cold_benchmark_preflights_prompt_token_budget(monkeypatch, tmp_path):
    calls = []
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="lmcache-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        kv_connector_mode="lmcache",
    )
    dataset_paths = {name: tmp_path / f"{name}.jsonl" for name in smoke_dataset_records()}
    lmcache_cfg = tmp_path / "lmcache-config.json"
    lmcache_cfg.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(public_vllm_smoke, "create_venv", lambda path: None)
    monkeypatch.setattr(public_vllm_smoke, "install_vllm", lambda python: None)
    monkeypatch.setattr(public_vllm_smoke, "install_lmcache", lambda python, version: "0.3.10")
    monkeypatch.setattr(public_vllm_smoke, "install_document_kv_package", lambda python, install_spec: None)
    monkeypatch.setattr(public_vllm_smoke, "apply_vllm_runtime_patches", lambda cfg: [])
    monkeypatch.setattr(public_vllm_smoke, "installed_versions", lambda python: {"vllm_version_installed": "0.23.0"})
    monkeypatch.setattr(public_vllm_smoke, "write_lmcache_config", lambda cfg: lmcache_cfg)
    monkeypatch.setattr(
        public_vllm_smoke,
        "probe_lmcache_import",
        lambda python, output, *, timeout_seconds, env: None,
    )
    monkeypatch.setattr(public_vllm_smoke, "benchmark_dataset_paths", lambda cfg: dataset_paths)
    monkeypatch.setattr(
        public_vllm_smoke,
        "validate_prompt_token_budget",
        lambda cfg, paths: calls.append(("validate_prompt_token_budget", paths)),
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "_run_lmcache_two_pass",
        lambda cfg, paths, warm, measure: calls.append(("_run_lmcache_two_pass", paths)),
    )

    public_vllm_smoke.run_lmcache_cold_benchmark(config)

    # The context-budget preflight must run before the expensive warm+measure passes
    # so over-budget prompts fail fast instead of surfacing as late request errors.
    assert ("validate_prompt_token_budget", dataset_paths) in calls
    assert calls.index(("validate_prompt_token_budget", dataset_paths)) < calls.index(
        ("_run_lmcache_two_pass", dataset_paths)
    )


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
