import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
from types import ModuleType

import pytest

import document_kv_cache.sglang_smoke as public_sglang_smoke
from document_kv_cache.benchmark_runner import BenchmarkGeneration
from document_kv_cache.benchmarks import (
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM,
    SUPPORTED_V1_DATASETS,
)
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import (
    build_engine_adapter_request,
    engine_adapter_request_to_record,
    read_engine_adapter_request_json,
    sglang_adapter_spec,
)
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment
from document_kv_cache.kvpack import PackChunk
from document_kv_cache.model_profiles import layout_for_model
from document_kv_cache.models import KVCacheKey
from document_kv_cache.sglang_smoke import (
    CACHET_MODEL_ID,
    DEFAULT_SGLANG_HICACHE_PAGE_SIZE,
    DEFAULT_SGLANG_PREPARED_HICACHE_PAGE_SIZE,
    DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_POLICY,
    DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_THRESHOLD,
    DEFAULT_SGLANG_LIVE_HANDOFF_GENERATOR_FACTORY,
    DEFAULT_SGLANG_LIVE_CHECK_EXTRA_BODY,
    DEFAULT_SGLANG_LIVE_CHECK_PROMPT_FORMAT,
    DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE,
    DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE,
    DEFAULT_SGLANG_FLUSH_CACHE_BEFORE_CACHE_ARM,
    DEFAULT_SGLANG_FLUSH_CACHE_BEFORE_CANARY,
    DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT_SECONDS,
    DEFAULT_SGLANG_LIVE_BENCHMARK_REPEATS,
    DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV,
    HF_MODEL_ID,
    SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE,
    SGLANG_GENERATED_HANDOFF_EXPLICIT_FIELDS_UNSUPPORTED_MESSAGE,
    SERVED_MODEL_NAME,
    SGLangLiveHandoffGenerationConfig,
    SGLANG_QUALITY_CANARY_ANSWER,
    SGLANG_DEPENDENCY_CONSTRAINTS,
    SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE,
    SGLANG_LIVE_BENCHMARK_RECORD_TYPE,
    SGLANG_PREPARED_V1_LIVE_BENCHMARK_SCOPE,
    SGLANG_VERSION,
    SGLangSmokeBenchmarkConfig,
    benchmark_dataset_paths,
    build_metadata,
    flush_sglang_cache,
    build_sglang_quality_canary_request,
    build_sglang_hicache_provider_probe_record,
    build_sglang_server_args,
    dependency_constraints,
    document_kv_package_install_spec,
    install_document_kv_package,
    install_sglang,
    parse_args,
    prepared_sglang_benchmark_handoff_coverage_record,
    prepare_generated_live_handoff,
    require_sglang_cache_hit,
    run_live_checks,
    run_sglang_live_benchmark,
    run_sglang_quality_canary,
    sglang_cache_hit_validation_record,
    sglang_cached_token_counts,
    sglang_prefill_token_counts,
    sglang_hicache_config_for_smoke,
    validate_prepared_sglang_benchmark_handoffs,
)
from document_kv_cache.storage import local_path
from document_kv_cache.workflow import SourceDocument
from sglang_kv_injection.hicache_keys import sglang_hicache_page_keys
from sglang_kv_injection.sglang_dynamic_backend import (
    DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY,
    DocumentKVHiCachePageProvider,
    DocumentKVHiCacheRequestContext,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def handoff_record(
    *,
    request_id: str,
    payload_uri: str,
    block_size: int = 2,
    total_tokens: int = 4,
) -> dict[str, object]:
    total_bytes = total_tokens * 4
    layout = KVLayout(
        model_id="tiny-test-model",
        lora_id="base",
        layout_version="standard-v1",
        dtype="int8",
        num_layers=1,
        block_size=block_size,
        bytes_per_token=4,
    )
    handle = KVCacheHandle(
        request_id=request_id,
        handle_uri=f"document-kv://{request_id}",
        layout=layout,
        segments=(
            KVSegment(
                "doc-1",
                "document_static",
                "static",
                0,
                total_tokens,
                0,
                total_bytes,
            ),
        ),
        total_tokens=total_tokens,
        total_bytes=total_bytes,
    )
    ready = EngineReadyRequest(
        handle=handle,
        payload=b"\0" * total_bytes,
        estimated_gpu_bytes=total_bytes,
    )
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    return engine_adapter_request_to_record(adapter_request, payload_uri=payload_uri)


def write_handoff_json(
    path: Path,
    *,
    request_id: str,
    payload_uri: str,
    block_size: int = 2,
    total_tokens: int = 4,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            handoff_record(
                request_id=request_id,
                payload_uri=payload_uri,
                block_size=block_size,
                total_tokens=total_tokens,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_prepared_sglang_v1_datasets(
    root: Path,
    *,
    missing_page_keys_for: str | None = None,
    page_key_count: int = 2,
) -> tuple[str, ...]:
    layout = layout_for_model(CACHET_MODEL_ID)
    total_tokens = layout.block_size * 2
    specs: list[str] = []
    for dataset in SUPPORTED_V1_DATASETS:
        request_id = f"cachet-{dataset}-prepared-1"
        handoff_path = root / "handoffs" / dataset / "example-1.handoff.json"
        payload_uri = f"disk:{root / 'payloads' / dataset / 'example-1.kv'}"
        write_handoff_json(
            handoff_path,
            request_id=request_id,
            payload_uri=payload_uri,
            block_size=layout.block_size,
            total_tokens=total_tokens,
        )
        kv_transfer_params: dict[str, object] = {
            DOCUMENT_KV_REQUEST_ID_PARAM: request_id,
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
            DOCUMENT_KV_PAYLOAD_URI_PARAM: payload_uri,
        }
        if dataset != missing_page_keys_for:
            kv_transfer_params[DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM] = [
                f"{dataset}-page-{index}"
                for index in range(1, page_key_count + 1)
            ]
        dataset_path = root / "datasets" / f"{dataset}.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "example_id": f"{dataset}-example-1",
                    "documents": [
                        {
                            "document_id": f"{dataset}-doc-1",
                            "text": f"{dataset} document contains answer-{dataset}.",
                        }
                    ],
                    "query": f"What answer is in the {dataset} document?",
                    "expected_answer": f"answer-{dataset}",
                    "kv_transfer_params": kv_transfer_params,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        specs.append(f"{dataset}={dataset_path}")
    return tuple(specs)


class FakeLiveResult:
    def __init__(
        self,
        *,
        ok: bool,
        request_id: str | None,
        prompt_text_mode: str,
        cache_arm: bool,
        request_mode: str = "completion",
        server_usage_prompt_tokens: int | None = None,
    ) -> None:
        self.ok = ok
        self.request_id = request_id
        self.prompt_text_mode = prompt_text_mode
        self.request_mode = request_mode
        self.cache_arm = cache_arm
        self.server_usage_prompt_tokens = server_usage_prompt_tokens

    def to_record(self):
        record = {
            "ok": self.ok,
            "request_id": self.request_id,
            "prompt_text_mode": self.prompt_text_mode,
            "request_mode": self.request_mode,
            "arm_id": "document_kv_cache" if self.cache_arm else "baseline_prefill",
        }
        if self.server_usage_prompt_tokens is not None:
            record["prompt_tokens"] = self.server_usage_prompt_tokens
            record["metadata"] = {
                "server_usage_prompt_tokens": str(self.server_usage_prompt_tokens),
                "server_usage_prompt_tokens_present": "true",
            }
        return record


class FakeCanaryEngine:
    def __init__(self, output_text: str | None = None) -> None:
        self.output_text = (
            f"The token is {SGLANG_QUALITY_CANARY_ANSWER}."
            if output_text is None
            else output_text
        )
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return BenchmarkGeneration(
            output_text=self.output_text,
            prompt_tokens=7,
            completion_tokens=3,
            ttft_seconds=0.1,
            time_to_completion_seconds=0.2,
            metadata={
                "server": "fake-canary",
                "prompt_token_source": "server_usage",
                "request_payload_endpoint": "/v1/chat/completions",
            },
        )


def fake_canary_record(*, ok: bool = True) -> dict[str, object]:
    return {
        "ok": ok,
        "label": "model_quality_canary",
        "model_id": CACHET_MODEL_ID,
        "served_model_name": SERVED_MODEL_NAME,
        "answer_found": ok,
        "output_text": SGLANG_QUALITY_CANARY_ANSWER if ok else "wrong",
        "metadata": {},
    }


def fake_flush_record(
    *, ok: bool = True, reason: str = "before_model_quality_canary"
) -> dict[str, object]:
    return {
        "ok": ok,
        "reason": reason,
        "endpoint": "/flush_cache",
        "timeout_seconds": DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT_SECONDS,
        "http_status": 200 if ok else 400,
        "response_text_tail": "Cache flushed." if ok else "Flush cache failed.",
    }


class TinyTokenizer:
    token_ids = (11, 12, 13, 14)

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert text
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return {"input_ids": [list(self.token_ids)]}

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **template_kwargs
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert template_kwargs == {
            "reasoning_effort": "none",
            "thinking": False,
            "enable_thinking": False,
        }
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        return rendered + "<|im_start|>assistant\n"


class TinySGLangLiveKVGenerator:
    tokenizer = TinyTokenizer()
    add_special_tokens = False

    def generate(self, *, document, chunk, config, training_artifacts=None):
        del training_artifacts
        layout = layout_for_model(
            config.model_id,
            dtype=config.dtype,
            lora_id=config.lora_id,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )
        token_count = len(self.tokenizer.token_ids)
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
            ),
            payload=b"\0" * (token_count * layout.bytes_per_token),
            token_count=token_count,
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


class PromptFormatTokenizer:
    source_token_ids = (21, 22, 23, 24)
    runtime_token_ids = (21, 22, 23, 24, 25, 26)
    plain_token_ids = (91, 92, 93, 94)
    seen_texts: list[str] = []
    seen_templates: list[list[dict[str, str]]] = []
    seen_template_kwargs: list[dict[str, object]] = []

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert text
        assert return_tensors == "pt"
        assert add_special_tokens is False
        self.seen_texts.append(text)
        if text.startswith("<|im_start|>system") and "<|im_start|>assistant" in text:
            token_ids = self.runtime_token_ids
        elif text.startswith("<|im_start|>system"):
            token_ids = self.source_token_ids
        elif text.startswith("Benchmark:"):
            token_ids = self.plain_token_ids
        else:
            token_ids = (7, 8, 9, 10)
        return {"input_ids": [list(token_ids)]}

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **template_kwargs
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        self.seen_templates.append([dict(message) for message in messages])
        self.seen_template_kwargs.append(dict(template_kwargs))
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        return rendered + "<|im_start|>assistant\n"


class PromptFormatSGLangLiveKVGenerator:
    tokenizer = PromptFormatTokenizer()
    add_special_tokens = False

    def generate(self, *, document, chunk, config, training_artifacts=None):
        del training_artifacts
        layout = layout_for_model(
            config.model_id,
            dtype=config.dtype,
            lora_id=config.lora_id,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )
        token_ids = self.tokenizer(
            chunk.text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
            ),
            payload=b"\0" * (len(token_ids) * layout.bytes_per_token),
            token_count=len(token_ids),
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


class BoundaryDriftTokenizer:
    prefix_text = "prefix-alone"
    stable_text = "stable-prefix"

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert text
        assert return_tensors == "pt"
        assert add_special_tokens is False
        if text == self.prefix_text:
            token_ids = [11, 12, 13, 14, 15]
        elif text == self.stable_text:
            token_ids = [11, 12, 13, 14]
        else:
            token_ids = [11, 12, 13, 14, 99, 20]
        return {"input_ids": [token_ids]}

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        if list(token_ids) == [11, 12, 13, 14]:
            return self.stable_text
        return "not-round-trippable"


class BoundaryDriftSGLangLiveKVGenerator:
    tokenizer = BoundaryDriftTokenizer()
    add_special_tokens = False

    def generate(self, *, document, chunk, config, training_artifacts=None):
        del training_artifacts
        layout = layout_for_model(
            config.model_id,
            dtype=config.dtype,
            lora_id=config.lora_id,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )
        token_ids = self.tokenizer(
            chunk.text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
            ),
            payload=b"\0" * (len(token_ids) * layout.bytes_per_token),
            token_count=len(token_ids),
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


class PageAlignmentFallbackTokenizer:
    non_page_text = "lossless-nine-token-prefix"
    small_page_text = "lossless-five-token-prefix"

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert text
        assert return_tensors == "pt"
        assert add_special_tokens is False
        if text == self.non_page_text:
            token_ids = list(range(1, 10))
        elif text == self.small_page_text:
            token_ids = list(range(1, 6))
        else:
            token_ids = [99]
        return {"input_ids": [token_ids]}

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        token_ids = list(token_ids)
        if token_ids == list(range(1, 9 + 1)):
            return self.non_page_text
        if token_ids == list(range(1, 5 + 1)):
            return self.small_page_text
        return "not-round-trippable"


class PageAlignmentFallbackSGLangLiveKVGenerator:
    tokenizer = PageAlignmentFallbackTokenizer()
    add_special_tokens = False


def test_dependency_constraints_match_pinned_sglang_stack():
    assert dependency_constraints() == list(SGLANG_DEPENDENCY_CONSTRAINTS)
    assert dependency_constraints() == ["sglang==0.5.10.post1"]
    assert SGLANG_VERSION == "0.5.10.post1"
    assert HF_MODEL_ID == "Qwen/Qwen3-4B-Instruct-2507"
    assert CACHET_MODEL_ID == "qwen3:4b-instruct"
    assert SERVED_MODEL_NAME == "qwen3-4b-instruct"
    assert ":" not in SERVED_MODEL_NAME


def test_sglang_smoke_cache_arm_requires_handoff_and_page_keys(tmp_path):
    with pytest.raises(ValueError) as exc:
        SGLangSmokeBenchmarkConfig(benchmark_id="sglang-1", output_dir=tmp_path / "out")

    assert str(exc.value) == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE

    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(
        handoff_path, request_id="cachet-live-sglang-1", payload_uri=payload_uri
    )
    with pytest.raises(ValueError) as exc:
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            handoff_json=str(handoff_path),
            payload_uri=payload_uri,
            request_id="cachet-live-sglang-1",
            sglang_hicache_page_keys=(),
        )

    assert str(exc.value) == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE

    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        handoff_json=str(handoff_path),
        payload_uri=payload_uri,
        request_id="cachet-live-sglang-1",
        sglang_hicache_page_keys=("page-a", "page-b"),
    )

    assert config.baseline_only is False
    assert config.sglang_hicache_page_keys == ("page-a", "page-b")


def test_sglang_smoke_accepts_generated_live_handoff_without_explicit_fields(tmp_path):
    generation = SGLangLiveHandoffGenerationConfig(
        output_dir=tmp_path / "generated-live",
        generator_factory="module:factory",
        page_size=2,
    )
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        handoff_generation=generation,
    )

    assert config.handoff_generation == generation
    assert config.hicache_page_size == 2
    assert config.sglang_hicache_page_keys == ()
    assert (
        config.live_handoff_generation_path
        == tmp_path / "out" / "sglang-live-handoff-generation.json"
    )
    assert config.live_check_request_mode == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
    assert config.live_check_temperature == DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE
    assert DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE == 0.0
    assert config.live_check_extra_body == DEFAULT_SGLANG_LIVE_CHECK_EXTRA_BODY
    assert set(config.live_check_extra_body) == {
        "reasoning_effort",
        "chat_template_kwargs",
    }
    assert config.live_check_extra_body["reasoning_effort"] == "none"
    assert config.live_check_extra_body["chat_template_kwargs"] == {
        "thinking": False,
        "enable_thinking": False,
    }
    assert config.sglang_attention_backend is None
    assert config.sglang_sampling_backend is None
    assert config.sglang_enable_deterministic_inference is False
    assert (
        config.flush_cache_before_cache_arm
        is DEFAULT_SGLANG_FLUSH_CACHE_BEFORE_CACHE_ARM
    )
    assert config.flush_cache_before_canary is DEFAULT_SGLANG_FLUSH_CACHE_BEFORE_CANARY
    assert (
        config.flush_cache_timeout_seconds
        == DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT_SECONDS
    )
    assert config.live_benchmark_repeats == DEFAULT_SGLANG_LIVE_BENCHMARK_REPEATS
    assert (
        config.live_benchmark_output_path
        == tmp_path / "out" / "sglang-live-benchmark.json"
    )


def test_sglang_smoke_rejects_invalid_live_check_options(tmp_path):
    with pytest.raises(ValueError, match="live_check_request_mode"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            live_check_request_mode="responses",
        )

    with pytest.raises(ValueError, match="live_check_temperature"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            live_check_temperature=True,
        )

    with pytest.raises(ValueError, match="live_check_extra_body"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            live_check_extra_body=[],
        )

    with pytest.raises(ValueError, match="live_check_extra_body"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            live_check_extra_body={"bad": object()},
        )

    with pytest.raises(ValueError, match="flush_cache_before_cache_arm"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            flush_cache_before_cache_arm="yes",
        )

    with pytest.raises(ValueError, match="flush_cache_before_canary"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            flush_cache_before_canary="yes",
        )

    with pytest.raises(ValueError, match="flush_cache_timeout_seconds"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            flush_cache_timeout_seconds=0,
        )

    with pytest.raises(ValueError, match="live_benchmark_repeats"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            live_benchmark_repeats=-1,
        )

    with pytest.raises(ValueError, match="live_benchmark_repeats"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            live_benchmark_repeats=1,
        )


def test_sglang_smoke_accepts_prepared_v1_dataset_specs_for_live_benchmark(tmp_path):
    dataset_specs = write_prepared_sglang_v1_datasets(tmp_path / "prepared")

    config = parse_args(
        [
            "--benchmark-id",
            "sglang-prepared-v1",
            "--output-dir",
            str(tmp_path / "out"),
            "--live-benchmark-repeats",
            "1",
            *[
                item
                for spec in dataset_specs
                for item in ("--dataset", spec)
            ],
        ]
    )

    assert config.uses_prepared_datasets is True
    assert config.dataset_specs == dataset_specs
    assert config.live_benchmark_repeats == 1
    assert config.hicache_page_size == DEFAULT_SGLANG_PREPARED_HICACHE_PAGE_SIZE
    assert config.prepared_handoff_coverage_path == (
        tmp_path / "out" / "prepared-sglang-handoff-coverage.json"
    )
    assert tuple(benchmark_dataset_paths(config)) == SUPPORTED_V1_DATASETS


def test_sglang_smoke_rejects_invalid_prepared_v1_dataset_specs(tmp_path):
    dataset_specs = write_prepared_sglang_v1_datasets(tmp_path / "prepared")

    with pytest.raises(ValueError, match="missing required V1 datasets"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-prepared-v1",
            output_dir=tmp_path / "out",
            dataset_specs=dataset_specs[:-1],
            live_benchmark_repeats=1,
        )

    with pytest.raises(ValueError, match="cache-arm SGLang live benchmark"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-prepared-v1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            dataset_specs=dataset_specs,
            live_benchmark_repeats=0,
        )

    with pytest.raises(ValueError, match="dataset specs require live_benchmark_repeats"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-prepared-v1",
            output_dir=tmp_path / "out",
            dataset_specs=dataset_specs,
        )

    with pytest.raises(ValueError, match="must not be combined"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-prepared-v1",
            output_dir=tmp_path / "out",
            dataset_specs=dataset_specs,
            live_benchmark_repeats=1,
            handoff_generation=SGLangLiveHandoffGenerationConfig(
                output_dir=tmp_path / "generated-live"
            ),
        )


def test_prepared_sglang_benchmark_handoff_coverage_requires_page_keys(tmp_path):
    dataset_specs = write_prepared_sglang_v1_datasets(tmp_path / "prepared")
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-prepared-v1",
        output_dir=tmp_path / "out",
        dataset_specs=dataset_specs,
        live_benchmark_repeats=1,
    )

    record = prepared_sglang_benchmark_handoff_coverage_record(
        config,
        benchmark_dataset_paths(config),
    )

    assert record["ok"] is True
    assert record["release_v1_suite"] is True
    assert record["datasets"] == {dataset: 1 for dataset in SUPPORTED_V1_DATASETS}
    assert record["examples"] == len(SUPPORTED_V1_DATASETS)
    assert record["examples_with_kv_transfer_params"] == len(SUPPORTED_V1_DATASETS)
    assert record["examples_with_loadable_sglang_handoff_references"] == len(
        SUPPORTED_V1_DATASETS
    )
    assert record["missing_kv_transfer_params"] == []
    assert record["invalid_handoff_references"] == []
    validated = validate_prepared_sglang_benchmark_handoffs(
        config,
        benchmark_dataset_paths(config),
    )
    assert validated == record
    assert json.loads(
        config.prepared_handoff_coverage_path.read_text(encoding="utf-8")
    ) == record

    invalid_specs = write_prepared_sglang_v1_datasets(
        tmp_path / "prepared-missing-page-keys",
        missing_page_keys_for="niah",
    )
    invalid_config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-prepared-v1",
        output_dir=tmp_path / "invalid-out",
        dataset_specs=invalid_specs,
        live_benchmark_repeats=1,
    )

    invalid_record = prepared_sglang_benchmark_handoff_coverage_record(
        invalid_config,
        benchmark_dataset_paths(invalid_config),
    )

    assert invalid_record["ok"] is False
    assert invalid_record["missing_kv_transfer_params"] == []
    assert len(invalid_record["invalid_handoff_references"]) == 1
    invalid_reference = invalid_record["invalid_handoff_references"][0]
    assert invalid_reference["dataset"] == "niah"
    assert invalid_reference["example_id"] == "niah-example-1"
    assert DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM in invalid_reference["error"]

    with pytest.raises(ValueError, match="must be enriched with Cachet"):
        validate_prepared_sglang_benchmark_handoffs(
            invalid_config,
            benchmark_dataset_paths(invalid_config),
        )
    assert json.loads(
        invalid_config.prepared_handoff_coverage_path.read_text(encoding="utf-8")
    ) == invalid_record

    mismatch_config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-prepared-v1",
        output_dir=tmp_path / "mismatch-out",
        dataset_specs=dataset_specs,
        live_benchmark_repeats=1,
        hicache_page_size=1,
    )
    mismatch_record = prepared_sglang_benchmark_handoff_coverage_record(
        mismatch_config,
        benchmark_dataset_paths(mismatch_config),
    )
    assert mismatch_record["ok"] is False
    assert len(mismatch_record["invalid_handoff_references"]) == len(
        SUPPORTED_V1_DATASETS
    )
    assert "must match handoff handle.layout.block_size" in mismatch_record[
        "invalid_handoff_references"
    ][0]["error"]

    short_page_key_specs = write_prepared_sglang_v1_datasets(
        tmp_path / "prepared-short-page-keys",
        page_key_count=1,
    )
    short_page_key_config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-prepared-v1",
        output_dir=tmp_path / "short-page-keys-out",
        dataset_specs=short_page_key_specs,
        live_benchmark_repeats=1,
    )
    short_page_key_record = prepared_sglang_benchmark_handoff_coverage_record(
        short_page_key_config,
        benchmark_dataset_paths(short_page_key_config),
    )
    assert short_page_key_record["ok"] is False
    assert len(short_page_key_record["invalid_handoff_references"]) == len(
        SUPPORTED_V1_DATASETS
    )
    assert "has 1 keys" in short_page_key_record["invalid_handoff_references"][0][
        "error"
    ]


def test_parse_args_accepts_live_check_extra_body_json(tmp_path):
    config = parse_args(
        [
            "--benchmark-id",
            "sglang-extra-body",
            "--output-dir",
            str(tmp_path / "out"),
            "--baseline-only",
            "--live-check-extra-body-json",
            '{"reasoning_effort":"none","chat_template_kwargs":{"thinking":false}}',
        ]
    )

    assert config.live_check_extra_body == {
        "reasoning_effort": "none",
        "chat_template_kwargs": {"thinking": False},
    }
    assert public_sglang_smoke._sglang_live_check_chat_template_kwargs(
        config.live_check_extra_body
    ) == {
        "reasoning_effort": "none",
        "thinking": False,
        "enable_thinking": False,
    }


def test_parse_args_can_disable_cache_flushes(tmp_path):
    config = parse_args(
        [
            "--benchmark-id",
            "sglang-no-flush",
            "--output-dir",
            str(tmp_path / "out"),
            "--baseline-only",
            "--no-flush-cache-before-cache-arm",
            "--no-flush-cache-before-canary",
            "--flush-cache-timeout-seconds",
            "12.5",
        ]
    )

    assert config.flush_cache_before_cache_arm is False
    assert config.flush_cache_before_canary is False
    assert config.flush_cache_timeout_seconds == 12.5


def test_sglang_smoke_validates_server_backend_controls(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
        sglang_attention_backend=" triton ",
        sglang_sampling_backend=" pytorch ",
        sglang_enable_deterministic_inference=True,
    )

    assert config.sglang_attention_backend == "triton"
    assert config.sglang_sampling_backend == "pytorch"
    assert config.sglang_enable_deterministic_inference is True

    with pytest.raises(ValueError, match="sglang_attention_backend"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            sglang_attention_backend="flash-attention",
        )

    with pytest.raises(ValueError, match="sglang_sampling_backend"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            sglang_sampling_backend="flash-attention",
        )

    with pytest.raises(ValueError, match="sglang_attention_backend"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            sglang_enable_deterministic_inference=True,
        )

    with pytest.raises(ValueError, match="sglang_attention_backend"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            sglang_attention_backend="flashinfer",
            sglang_enable_deterministic_inference=True,
        )

    with pytest.raises(ValueError, match="sglang_sampling_backend"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            sglang_attention_backend="triton",
            sglang_sampling_backend="flashinfer",
            sglang_enable_deterministic_inference=True,
        )


def test_sglang_smoke_rejects_generated_live_handoff_with_explicit_fields(tmp_path):
    generation = SGLangLiveHandoffGenerationConfig(
        output_dir=tmp_path / "generated-live",
        generator_factory="module:factory",
    )

    with pytest.raises(ValueError) as exc:
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            handoff_generation=generation,
            sglang_hicache_page_keys=("page-a",),
        )

    assert (
        str(exc.value) == SGLANG_GENERATED_HANDOFF_EXPLICIT_FIELDS_UNSUPPORTED_MESSAGE
    )

    with pytest.raises(
        ValueError, match="must match live handoff generation page_size"
    ):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            handoff_generation=generation,
            hicache_page_size=2,
        )


def test_sglang_smoke_accepts_baseline_only_without_handoff_fields(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )

    assert config.baseline_only is True
    assert config.sglang_hicache_page_keys == ()
    assert config.local_dir == Path("/local_disk0/document-kv-sglang-smoke-sglang-1")


def test_sglang_smoke_rejects_invalid_hicache_prefetch_threshold(tmp_path):
    with pytest.raises(ValueError, match="hicache_storage_prefetch_threshold"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            hicache_storage_prefetch_threshold=0,
        )

    with pytest.raises(ValueError, match="hicache_storage_prefetch_threshold"):
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            hicache_storage_prefetch_threshold=True,
        )


def test_sglang_smoke_rejects_handoff_fields_for_baseline_only(tmp_path):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    write_handoff_json(
        handoff_path,
        request_id="cachet-live-sglang-1",
        payload_uri=f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}",
    )

    with pytest.raises(ValueError) as exc:
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            handoff_json=str(handoff_path),
        )

    assert str(exc.value) == SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE

    with pytest.raises(ValueError) as exc:
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            sglang_hicache_page_keys=(),
        )

    assert str(exc.value) == SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE


def test_sglang_smoke_rejects_generated_live_handoff_for_baseline_only(tmp_path):
    with pytest.raises(ValueError) as exc:
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            handoff_generation=SGLangLiveHandoffGenerationConfig(
                output_dir=tmp_path / "generated-live"
            ),
        )

    assert str(exc.value) == SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE


def test_prepare_generated_live_handoff_writes_runtime_handoff_inputs(
    tmp_path, monkeypatch
):
    module = ModuleType("cachet_test_sglang_live_handoff_generator")
    module.build_generator = TinySGLangLiveKVGenerator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-generated-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_generation=SGLangLiveHandoffGenerationConfig(
            output_dir=tmp_path / "generated-live",
            generator_factory=f"{module.__name__}:build_generator",
            dtype="bfloat16",
            align_bytes=1,
            page_size=2,
        ),
    )

    runtime_config = prepare_generated_live_handoff(config)

    assert runtime_config.handoff_generation is None
    assert runtime_config.request_id == "cachet-live-sglang-generated-1"
    assert runtime_config.sglang_hicache_page_keys == sglang_hicache_page_keys(
        TinyTokenizer.token_ids,
        page_size=2,
    )
    assert runtime_config.handoff_json is not None
    assert Path(runtime_config.handoff_json).exists()
    assert runtime_config.payload_uri is not None
    assert local_path(runtime_config.payload_uri).exists()
    handoff = read_engine_adapter_request_json(
        runtime_config.handoff_json, require_external_payload_uri=False
    )
    layout = handoff["handle"]["layout"]
    assert layout["block_size"] == 2
    provider = DocumentKVHiCachePageProvider()
    context = DocumentKVHiCacheRequestContext(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: runtime_config.request_id,
            DOCUMENT_KV_HANDOFF_JSON_PARAM: runtime_config.handoff_json,
            DOCUMENT_KV_PAYLOAD_URI_PARAM: runtime_config.payload_uri,
            DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: list(
                runtime_config.sglang_hicache_page_keys
            ),
        },
        request_id=runtime_config.request_id,
        handoff_json=runtime_config.handoff_json,
        payload_uri=runtime_config.payload_uri,
        sglang_hicache_page_keys=runtime_config.sglang_hicache_page_keys,
    )
    hydrated = provider.batch_get_v1(
        [runtime_config.sglang_hicache_page_keys[0]],
        document_kv_request_context=context,
    )
    payload = local_path(runtime_config.payload_uri).read_bytes()
    assert hydrated == [payload[: 2 * layout["bytes_per_token"]]]
    generation = json.loads(
        config.live_handoff_generation_path.read_text(encoding="utf-8")
    )
    assert generation["ok"] is True
    assert (
        generation["live_check_request_mode"] == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
    )
    assert generation["chat_template_rendered"] is True
    assert generation["live_check_chat_template_kwargs"] == {
        "reasoning_effort": "none",
        "thinking": False,
        "enable_thinking": False,
    }
    assert generation["cache_prefix_tokens"] == len(TinyTokenizer.token_ids)
    assert generation["sglang_hicache_page_size"] == 2


def test_prepare_generated_live_handoff_uses_live_prompt_format_cache_prefix(
    tmp_path, monkeypatch
):
    module = ModuleType("cachet_test_sglang_live_handoff_prompt_format_generator")
    module.build_generator = PromptFormatSGLangLiveKVGenerator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    PromptFormatTokenizer.seen_texts.clear()
    PromptFormatTokenizer.seen_templates.clear()
    PromptFormatTokenizer.seen_template_kwargs.clear()
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-generated-qwen-chat",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_generation=SGLangLiveHandoffGenerationConfig(
            output_dir=tmp_path / "generated-live",
            generator_factory=f"{module.__name__}:build_generator",
            dtype="bfloat16",
            align_bytes=1,
            page_size=2,
        ),
    )

    runtime_config = prepare_generated_live_handoff(config)

    assert config.live_check_prompt_format == DEFAULT_SGLANG_LIVE_CHECK_PROMPT_FORMAT
    assert config.live_check_request_mode == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
    assert runtime_config.sglang_hicache_page_keys == sglang_hicache_page_keys(
        PromptFormatTokenizer.source_token_ids,
        page_size=2,
    )
    assert PromptFormatTokenizer.seen_texts
    assert PromptFormatTokenizer.seen_templates
    assert PromptFormatTokenizer.seen_template_kwargs == [
        {
            "reasoning_effort": "none",
            "thinking": False,
            "enable_thinking": False,
        }
    ]
    assert PromptFormatTokenizer.seen_templates[0][0]["role"] == "system"
    assert PromptFormatTokenizer.seen_templates[0][1]["role"] == "user"
    assert all(
        text.startswith("<|im_start|>system")
        for text in PromptFormatTokenizer.seen_texts
    )
    assert not any(
        text.startswith("Benchmark:") for text in PromptFormatTokenizer.seen_texts
    )
    generation = json.loads(
        config.live_handoff_generation_path.read_text(encoding="utf-8")
    )
    assert generation["live_check_request_mode"] == "chat"
    assert generation["live_handoff_prompt_format"] == "plain"
    assert generation["chat_template_rendered"] is True
    assert generation["live_check_chat_template_kwargs"] == {
        "reasoning_effort": "none",
        "thinking": False,
        "enable_thinking": False,
    }
    assert generation["cache_prefix_tokens"] == len(
        PromptFormatTokenizer.source_token_ids
    )
    assert generation["runtime_prompt_tokens"] == len(
        PromptFormatTokenizer.runtime_token_ids
    )
    assert generation["cache_prefix_token_stable"] is True


def test_prepare_generated_live_handoff_uses_token_stable_runtime_prefix(
    tmp_path, monkeypatch
):
    module = ModuleType("cachet_test_sglang_live_handoff_drift_generator")
    module.build_generator = BoundaryDriftSGLangLiveKVGenerator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original_source_document = public_sglang_smoke._live_handoff_cache_source_document

    def drifted_source_document(live_request, *, prefix):
        source_document = original_source_document(live_request, prefix=prefix)
        source_chunk = source_document.chunks[0]
        return SourceDocument.from_text(
            document_id=source_document.document_id,
            text=BoundaryDriftTokenizer.prefix_text,
            chunk_id=source_chunk.chunk_id,
            chunk_type=source_chunk.chunk_type,
            metadata=source_document.metadata,
            chunk_metadata=source_chunk.metadata,
        )

    monkeypatch.setattr(
        public_sglang_smoke,
        "_live_handoff_cache_source_document",
        drifted_source_document,
    )
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-token-stable-generated",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        live_check_request_mode="completion",
        handoff_generation=SGLangLiveHandoffGenerationConfig(
            output_dir=tmp_path / "generated-live",
            generator_factory=f"{module.__name__}:build_generator",
            dtype="bfloat16",
            align_bytes=1,
            page_size=2,
        ),
    )

    runtime_config = prepare_generated_live_handoff(config)

    assert runtime_config.sglang_hicache_page_keys == sglang_hicache_page_keys(
        (11, 12, 13, 14),
        page_size=2,
    )
    handoff = read_engine_adapter_request_json(
        runtime_config.handoff_json, require_external_payload_uri=False
    )
    assert handoff["handle"]["total_tokens"] == 4
    generation = json.loads(
        config.live_handoff_generation_path.read_text(encoding="utf-8")
    )
    assert generation["cache_prefix_tokens"] == 4
    assert generation["cache_prefix_source_tokens"] == 5
    assert generation["runtime_prompt_tokens"] == 6
    assert generation["cache_prefix_full_pages"] == 2
    assert generation["cache_prefix_token_stable_ratio"] == 0.8
    assert generation["cache_prefix_token_stable"] is False
    assert generation["cache_prefix_token_stable_truncated"] is True
    assert generation["cache_prefix_chars"] == len(BoundaryDriftTokenizer.stable_text)
    assert generation["cache_prefix_source_chars"] == len(
        BoundaryDriftTokenizer.prefix_text
    )


def test_token_stable_handoff_rejects_tiny_runtime_prefix():
    source_document = SourceDocument.from_text(
        document_id="cachet-live-short-prefix",
        text=BoundaryDriftTokenizer.prefix_text,
    )

    with pytest.raises(RuntimeError, match="token-stable prefix is too small"):
        public_sglang_smoke._token_stable_handoff_source_document(
            BoundaryDriftSGLangLiveKVGenerator(),
            source_document=source_document,
            source_token_ids=(11, 12, 13, 14, 15),
            runtime_token_ids=(11, 99, 20),
            page_size=1,
        )


def test_token_stable_handoff_rejects_non_page_aligned_decode_fallback():
    source_document = SourceDocument.from_text(
        document_id="cachet-live-page-align",
        text="source-prefix",
    )

    with pytest.raises(RuntimeError, match="token-stable prefix is too small"):
        public_sglang_smoke._token_stable_handoff_source_document(
            PageAlignmentFallbackSGLangLiveKVGenerator(),
            source_document=source_document,
            source_token_ids=tuple(range(1, 12)),
            runtime_token_ids=(*range(1, 11), 99),
            page_size=5,
        )


def test_prepare_generated_live_handoff_uses_sglang_venv_when_available(
    tmp_path, monkeypatch
):
    generation = SGLangLiveHandoffGenerationConfig(
        output_dir=tmp_path / "generated-live",
        generator_factory="module:factory",
        dtype="bfloat16",
        align_bytes=1,
        page_size=3,
        timeout_seconds=1234.0,
    )
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-generated-venv",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        live_check_prompt_format="plain",
        live_check_temperature=0.25,
        handoff_generation=generation,
    )
    config.venv_python.parent.mkdir(parents=True)
    config.venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    calls = []

    def fake_run(argv, *, check, capture_output, text, timeout, env):
        calls.append((argv, check, capture_output, text, timeout, env))
        assert argv[0] == str(config.venv_python)
        assert argv[1] == "-c"
        assert "DEFAULT_SGLANG_LIVE_CHECK_EXTRA_BODY,\n" in argv[2]
        assert "DEFAULT_SGLANG_LIVE_CHECK_PROMPT_FORMAT,\n" in argv[2]
        assert "DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE,\n" in argv[2]
        assert "DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE,\n" in argv[2]
        input_payload = json.loads(Path(argv[3]).read_text(encoding="utf-8"))
        assert input_payload["benchmark_id"] == "sglang-generated-venv"
        assert input_payload["live_check_prompt_format"] == "plain"
        assert input_payload["live_check_temperature"] == 0.25
        assert (
            input_payload["live_check_extra_body"]
            == DEFAULT_SGLANG_LIVE_CHECK_EXTRA_BODY
        )
        assert (
            input_payload["live_check_request_mode"]
            == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
        )
        assert (
            input_payload["handoff_generation"]["generator_factory"] == "module:factory"
        )
        assert input_payload["handoff_generation"]["sglang_hicache_page_size"] == 3
        Path(argv[4]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[4]).write_text(
            json.dumps(
                {
                    "ok": True,
                    "request_id": "cachet-live-sglang-generated-venv",
                    "handoff_json": str(
                        tmp_path / "generated-live" / "sglang-live.handoff.json"
                    ),
                    "payload_uri": f"disk:{tmp_path / 'generated-live' / 'sglang-live.kv'}",
                    "sglang_hicache_page_keys": ["page-a"],
                    "generator_python": str(config.venv_python),
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="worker ok", stderr="")

    monkeypatch.setattr(public_sglang_smoke.subprocess, "run", fake_run)

    runtime_config = prepare_generated_live_handoff(config)

    assert runtime_config.handoff_json == str(
        tmp_path / "generated-live" / "sglang-live.handoff.json"
    )
    assert runtime_config.sglang_hicache_page_keys == ("page-a",)
    generation_record = json.loads(
        config.live_handoff_generation_path.read_text(encoding="utf-8")
    )
    assert generation_record["generator_python"] == str(config.venv_python)
    assert len(calls) == 1
    assert calls[0][4] == 1234.0
    assert calls[0][5]["HF_HOME"] == str(config.hf_cache_dir)


def test_document_kv_package_install_spec_prefers_config_then_env(
    monkeypatch, tmp_path
):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
        package_install_spec="dbfs:/tmp/cachet/cachet_kv.whl",
    )

    assert document_kv_package_install_spec(config) == "/dbfs/tmp/cachet/cachet_kv.whl"

    monkeypatch.setenv(
        DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV, "dbfs:/tmp/cachet/from-env.whl"
    )
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )

    assert document_kv_package_install_spec(config) == "/dbfs/tmp/cachet/from-env.whl"


def test_document_kv_package_install_spec_falls_back_to_source_checkout(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV, raising=False)
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )

    assert document_kv_package_install_spec(config) == str(REPO_ROOT)


def test_install_sglang_and_cachet_package_use_pinned_constraints(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(public_sglang_smoke, "run", lambda argv: calls.append(argv))
    python = tmp_path / "venv" / "bin" / "python"

    install_sglang(python)
    install_document_kv_package(python, "/tmp/cachet.whl")

    assert calls == [
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ],
        [str(python), "-m", "pip", "install", *dependency_constraints()],
        [str(python), "-m", "pip", "install", "--no-deps", "/tmp/cachet.whl"],
    ]


def test_sglang_server_args_use_qwen3_and_hicache_backend(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
        server_port=8123,
        context_length=8192,
        mem_fraction_static=0.72,
        hicache_page_store_uri=f"disk:{tmp_path / 'hicache-pages'}",
        hicache_page_size=2,
        hicache_size_gb=4,
        hicache_write_policy="write_through_selective",
    )

    args = build_sglang_server_args(config, tmp_path / "venv" / "bin" / "python")

    assert args[:4] == [
        str(tmp_path / "venv" / "bin" / "python"),
        "-u",
        "-m",
        "sglang.launch_server",
    ]
    assert args[args.index("--model-path") + 1] == HF_MODEL_ID
    assert args[args.index("--served-model-name") + 1] == SERVED_MODEL_NAME
    assert ":" not in args[args.index("--served-model-name") + 1]
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "8123"
    assert args[args.index("--context-length") + 1] == "8192"
    assert args[args.index("--mem-fraction-static") + 1] == "0.72"
    assert "--attention-backend" not in args
    assert "--sampling-backend" not in args
    assert "--enable-deterministic-inference" not in args
    assert "--enable-hierarchical-cache" in args
    assert args[args.index("--hicache-storage-backend") + 1] == "dynamic"
    assert args[args.index("--page-size") + 1] == "2"
    assert args[args.index("--hicache-size") + 1] == "4"
    assert args[args.index("--hicache-storage-prefetch-policy") + 1] == (
        DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_POLICY
    )
    assert args[args.index("--hicache-write-policy") + 1] == "write_through_selective"
    extra_config = json.loads(
        args[args.index("--hicache-storage-backend-extra-config") + 1]
    )
    assert extra_config[DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY].endswith(
        "/hicache-pages"
    )
    assert extra_config[DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY]
    assert extra_config["document_kv.requires_native_runtime"] is True
    assert (
        extra_config["prefetch_threshold"]
        == DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_THRESHOLD
    )


def test_sglang_server_args_include_backend_controls(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
        sglang_attention_backend="triton",
        sglang_sampling_backend="pytorch",
        sglang_enable_deterministic_inference=True,
    )

    args = build_sglang_server_args(config, tmp_path / "venv" / "bin" / "python")

    assert args[args.index("--attention-backend") + 1] == "triton"
    assert args[args.index("--sampling-backend") + 1] == "pytorch"
    assert "--enable-deterministic-inference" in args


def test_sglang_hicache_provider_probe_rejects_noop_launch_config():
    launch_config = sglang_hicache_config_for_smoke(
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1", output_dir=Path("/tmp/out"), baseline_only=True
        )
    )
    extra_config = json.loads(launch_config["hicache_storage_backend_extra_config"])
    extra_config.pop(DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY)
    launch_config["hicache_storage_backend_extra_config"] = json.dumps(extra_config)

    with pytest.raises(ValueError, match="provider_factory"):
        build_sglang_hicache_provider_probe_record(launch_config)


def test_sglang_hicache_provider_probe_accepts_builtin_provider(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
        hicache_page_store_uri=f"disk:{tmp_path / 'pages'}",
    )

    record = build_sglang_hicache_provider_probe_record(
        sglang_hicache_config_for_smoke(config)
    )

    assert record["document_kv_hicache_provider_ok"] is True
    assert record["document_kv_requires_native_runtime"] is True
    assert record["document_kv_provider_type"].endswith("DocumentKVHiCachePageProvider")


def test_sglang_smoke_cli_accepts_handoff_cache_arm(monkeypatch, tmp_path):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(
        handoff_path, request_id="cachet-live-sglang-1", payload_uri=payload_uri
    )
    seen_configs = []

    monkeypatch.setattr(
        public_sglang_smoke,
        "run_sglang_live_smoke",
        lambda config: seen_configs.append(config),
    )

    exit_code = public_sglang_smoke.main(
        [
            "--benchmark-id",
            "sglang-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--handoff-json",
            str(handoff_path),
            "--payload-uri",
            payload_uri,
            "--request-id",
            "cachet-live-sglang-1",
            "--sglang-hicache-page-keys-json",
            '["page-a","page-b"]',
        ]
    )

    assert exit_code == 0
    assert len(seen_configs) == 1
    assert seen_configs[0].sglang_hicache_page_keys == ("page-a", "page-b")


def test_sglang_smoke_cli_accepts_generated_live_handoff(monkeypatch, tmp_path):
    seen_configs = []
    monkeypatch.setattr(
        public_sglang_smoke,
        "run_sglang_live_smoke",
        lambda config: seen_configs.append(config),
    )

    exit_code = public_sglang_smoke.main(
        [
            "--benchmark-id",
            "sglang-generated-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--generate-live-handoff",
            "--live-handoff-output-dir",
            str(tmp_path / "generated-live"),
            "--live-handoff-generator-factory",
            "module:factory",
            "--live-handoff-dtype",
            "float16",
            "--live-handoff-align-bytes",
            "8",
            "--sglang-hicache-page-size",
            "2",
            "--live-check-prompt-format",
            "plain",
            "--live-check-request-mode",
            "completion",
            "--live-check-temperature",
            "0.25",
            "--hicache-storage-prefetch-threshold",
            "3",
            "--live-handoff-generation-timeout-seconds",
            "12.5",
        ]
    )

    assert exit_code == 0
    assert len(seen_configs) == 1
    generation = seen_configs[0].handoff_generation
    assert generation is not None
    assert generation.output_dir == tmp_path / "generated-live"
    assert generation.generator_factory == "module:factory"
    assert generation.dtype == "float16"
    assert generation.align_bytes == 8
    assert generation.page_size == 2
    assert generation.timeout_seconds == 12.5
    assert seen_configs[0].live_check_prompt_format == "plain"
    assert seen_configs[0].live_check_request_mode == "completion"
    assert seen_configs[0].live_check_temperature == 0.25
    assert seen_configs[0].hicache_page_size == 2
    assert (
        seen_configs[0].hicache_storage_prefetch_policy
        == DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_POLICY
    )
    assert seen_configs[0].hicache_storage_prefetch_threshold == 3


def test_build_metadata_records_custom_params_transport(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
        hardware_target="aws-g5-a10g",
    )

    metadata = build_metadata(config)

    assert metadata["model_id"] == CACHET_MODEL_ID
    assert metadata["served_model_name"] == SERVED_MODEL_NAME
    assert metadata["hardware_target"] == "aws-g5-a10g"
    assert metadata["kv_transfer_params_transport"] == "custom_params"
    assert metadata["sglang_attention_backend"] is None
    assert metadata["sglang_sampling_backend"] is None
    assert metadata["sglang_enable_deterministic_inference"] is False
    assert metadata["sglang_server_backend_options"] == {
        "attention_backend": None,
        "sampling_backend": None,
        "enable_deterministic_inference": False,
    }
    assert metadata["cache_prompt_text_mode"] == "logical"
    assert (
        metadata["live_check_prompt_format"] == DEFAULT_SGLANG_LIVE_CHECK_PROMPT_FORMAT
    )
    assert metadata["live_check_request_mode"] == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
    assert metadata["live_check_temperature"] == DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE
    assert metadata["live_check_extra_body"] == DEFAULT_SGLANG_LIVE_CHECK_EXTRA_BODY
    assert metadata["live_check_chat_template_kwargs"] == {
        "reasoning_effort": "none",
        "thinking": False,
        "enable_thinking": False,
    }
    assert metadata["flush_cache_before_cache_arm"] is True
    assert metadata["flush_cache_before_canary"] is True
    assert metadata["flush_cache_timeout_seconds"] == (
        DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT_SECONDS
    )
    assert metadata["live_benchmark_repeats"] == DEFAULT_SGLANG_LIVE_BENCHMARK_REPEATS
    assert metadata["live_benchmark_output_path"] is None
    assert metadata["requires_kv_transfer_params"] is False
    assert metadata["cache_arm_supported"] is False
    assert metadata["cache_arm_blocker"] == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE
    assert metadata["live_request_metadata_bridge_required"] is True
    assert metadata["live_request_metadata_bridge_ok"] is False
    assert metadata["generates_live_handoff"] is False
    assert metadata["live_handoff_generation"] is None


def test_build_metadata_records_cache_arm_support_for_handoff_smoke(tmp_path):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(
        handoff_path, request_id="cachet-live-sglang-1", payload_uri=payload_uri
    )
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        handoff_json=str(handoff_path),
        payload_uri=payload_uri,
        request_id="cachet-live-sglang-1",
        sglang_hicache_page_keys=("page-a",),
    )

    metadata = build_metadata(config)

    assert metadata["requires_kv_transfer_params"] is True
    assert metadata["cache_arm_supported"] is True
    assert metadata["cache_arm_blocker"] is None
    assert metadata["live_request_metadata_bridge_ok"] is False
    assert metadata["sglang_import_probe_ok"] is False

    generated_config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-generated-1",
        output_dir=tmp_path / "generated-out",
        handoff_generation=SGLangLiveHandoffGenerationConfig(
            output_dir=tmp_path / "generated-live"
        ),
    )
    metadata = build_metadata(generated_config)

    assert metadata["generates_live_handoff"] is True
    assert (
        metadata["live_handoff_generation"]["generator_factory"]
        == DEFAULT_SGLANG_LIVE_HANDOFF_GENERATOR_FACTORY
    )
    assert (
        metadata["live_handoff_generation"]["sglang_hicache_page_size"]
        == DEFAULT_SGLANG_HICACHE_PAGE_SIZE
    )

    metadata = build_metadata(
        config,
        import_probe_record={
            "ok": True,
            "document_kv_request_metadata_bridge_ok": True,
            "document_kv_request_metadata_bridge": {"ok": True},
        },
    )

    assert metadata["live_request_metadata_bridge_ok"] is True
    assert metadata["sglang_import_probe_ok"] is True
    assert metadata["document_kv_request_metadata_bridge"] == {"ok": True}


def test_sglang_quality_canary_request_uses_baseline_prompt_contract(tmp_path):
    request = build_sglang_quality_canary_request(
        model_id=SERVED_MODEL_NAME,
        hardware_target="aws-g6-l4",
    )

    assert request.suite_id == "sglang-quality-canary"
    assert request.model_id == SERVED_MODEL_NAME
    assert request.arm.uses_cache is False
    assert request.request_id is None
    assert request.kv_transfer_params == {}
    assert SGLANG_QUALITY_CANARY_ANSWER in request.logical_prompt_text
    assert request.prompt_text == request.logical_prompt_text


def test_run_sglang_quality_canary_records_model_sanity_result(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )
    engine = FakeCanaryEngine()

    record = run_sglang_quality_canary(config, engine=engine)

    assert record["ok"] is True
    assert record["model_id"] == CACHET_MODEL_ID
    assert record["served_model_name"] == SERVED_MODEL_NAME
    assert record["request_mode"] == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
    assert record["prompt_format"] == DEFAULT_SGLANG_LIVE_CHECK_PROMPT_FORMAT
    assert record["expected_answer"] == SGLANG_QUALITY_CANARY_ANSWER
    assert record["answer_found"] is True
    assert record["metadata"]["request_payload_endpoint"] == "/v1/chat/completions"
    assert engine.requests[0].suite_id == "sglang-quality-canary"


def test_run_sglang_quality_canary_records_failed_model_sanity_result(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )

    record = run_sglang_quality_canary(
        config, engine=FakeCanaryEngine(output_text="wrong token")
    )

    assert record["ok"] is False
    assert record["answer_found"] is False
    assert record["output_text"] == "wrong token"


def test_flush_sglang_cache_posts_runtime_flush_endpoint(monkeypatch, tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
        flush_cache_timeout_seconds=12.5,
    )
    seen = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b"Cache flushed.\n"

        def getcode(self):
            return self.status

    def fake_urlopen(request, *, timeout):
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(public_sglang_smoke.urllib.request, "urlopen", fake_urlopen)

    record = flush_sglang_cache(config, reason="before_model_quality_canary")

    assert record == {
        "endpoint": "/flush_cache",
        "http_status": 200,
        "ok": True,
        "reason": "before_model_quality_canary",
        "response_text_tail": "Cache flushed.\n",
        "timeout_seconds": 12.5,
    }
    assert seen == {
        "method": "POST",
        "timeout": 12.5,
        "url": f"{config.server_base_url}/flush_cache?timeout=12.5",
    }


def test_flush_sglang_cache_records_http_failure(monkeypatch, tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )

    def fake_urlopen(request, *, timeout):
        del request, timeout
        raise urllib.error.HTTPError(
            url=f"{config.server_base_url}/flush_cache",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b"Flush cache failed.\n"),
        )

    monkeypatch.setattr(public_sglang_smoke.urllib.request, "urlopen", fake_urlopen)

    record = flush_sglang_cache(config, reason="before_model_quality_canary")

    assert record["ok"] is False
    assert record["endpoint"] == "/flush_cache"
    assert record["http_status"] == 400
    assert record["response_text_tail"] == "Flush cache failed.\n"
    assert record["error_type"] == "HTTPError"


def test_run_live_checks_runs_baseline_only_and_records_cache_arm_blocker(
    monkeypatch, tmp_path
):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )
    seen_configs = []
    events = []

    def fake_live_check(live_config):
        seen_configs.append(live_config)
        events.append("cache" if live_config.use_cache_arm else "baseline")
        return FakeLiveResult(
            ok=True,
            request_id=live_config.kv_transfer_params.get(DOCUMENT_KV_REQUEST_ID_PARAM),
            prompt_text_mode=live_config.prompt_text_mode,
            request_mode=live_config.request_mode,
            cache_arm=live_config.use_cache_arm,
        )

    monkeypatch.setattr(
        public_sglang_smoke, "run_openai_compatible_live_check", fake_live_check
    )

    def fake_flush(live_config, *, reason):
        events.append(f"flush:{reason}")
        return fake_flush_record(reason=reason)

    monkeypatch.setattr(public_sglang_smoke, "flush_sglang_cache", fake_flush)

    def fake_canary(live_config):
        events.append("canary")
        return fake_canary_record()

    monkeypatch.setattr(
        public_sglang_smoke,
        "run_sglang_quality_canary",
        fake_canary,
    )

    record = run_live_checks(config)

    assert record["ok"] is True
    assert record["model_id"] == CACHET_MODEL_ID
    assert record["served_model_name"] == SERVED_MODEL_NAME
    assert len(seen_configs) == 1
    assert events == ["baseline", "flush:before_model_quality_canary", "canary"]
    assert seen_configs[0].model_id == SERVED_MODEL_NAME
    assert seen_configs[0].use_cache_arm is False
    assert seen_configs[0].temperature == DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE
    assert seen_configs[0].extra_body == DEFAULT_SGLANG_LIVE_CHECK_EXTRA_BODY
    assert seen_configs[0].prompt_text_mode == "logical"
    assert seen_configs[0].request_mode == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
    assert record["cache"] is None
    assert record["cache_arm_cache_flush"] is None
    assert record["model_quality_canary"]["ok"] is True
    assert record["canary_cache_flush"]["ok"] is True
    assert record["flush_cache_before_cache_arm"] is True
    assert record["flush_cache_before_canary"] is True
    assert record["flush_cache_timeout_seconds"] == (
        DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT_SECONDS
    )
    assert record["requires_kv_transfer_params"] is False
    assert record["cache_arm_supported"] is False
    assert record["cache_arm_blocker"] == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE
    assert record["live_request_metadata_bridge_required"] is True
    assert record["live_request_metadata_bridge_ok"] is False
    assert record["live_check_request_mode"] == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
    assert record["cache_prefill_log_start_index"] is None
    written = json.loads(config.live_smoke_output_path.read_text(encoding="utf-8"))
    assert written["cache"] is None
    assert written["cache_arm_cache_flush"] is None
    assert written["model_quality_canary"]["ok"] is True
    assert written["canary_cache_flush"]["ok"] is True
    assert written["baseline"]["model_id"] == CACHET_MODEL_ID
    assert written["baseline"]["served_model_name"] == SERVED_MODEL_NAME


def test_run_live_checks_runs_handoff_cache_arm(monkeypatch, tmp_path):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(
        handoff_path, request_id="cachet-live-sglang-1", payload_uri=payload_uri
    )
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_json=str(handoff_path),
        payload_uri=payload_uri,
        request_id="cachet-live-sglang-1",
        sglang_hicache_page_keys=("page-a", "page-b"),
    )
    config.server_log_path.parent.mkdir(parents=True, exist_ok=True)
    config.server_log_path.write_text(
        "[INFO] Prefill batch, #new-token: 6, #cached-token: 0\n"
        "[INFO] Prefill batch, #new-token: 1, #cached-token: 0\n",
        encoding="utf-8",
    )
    seen_configs = []
    events = []

    def fake_live_check(live_config):
        seen_configs.append(live_config)
        events.append("cache" if live_config.use_cache_arm else "baseline")
        return FakeLiveResult(
            ok=True,
            request_id=live_config.kv_transfer_params.get(DOCUMENT_KV_REQUEST_ID_PARAM),
            prompt_text_mode=live_config.prompt_text_mode,
            request_mode=live_config.request_mode,
            cache_arm=live_config.use_cache_arm,
        )

    monkeypatch.setattr(
        public_sglang_smoke, "run_openai_compatible_live_check", fake_live_check
    )

    def fake_flush(live_config, *, reason):
        events.append(f"flush:{reason}")
        return fake_flush_record(reason=reason)

    monkeypatch.setattr(public_sglang_smoke, "flush_sglang_cache", fake_flush)

    def fake_canary(live_config):
        events.append("canary")
        return fake_canary_record()

    monkeypatch.setattr(
        public_sglang_smoke,
        "run_sglang_quality_canary",
        fake_canary,
    )

    record = run_live_checks(
        config,
        import_probe_record={
            "ok": True,
            "document_kv_request_metadata_bridge_ok": True,
            "document_kv_request_metadata_bridge": {"ok": True},
        },
    )

    assert record["ok"] is True
    assert record["cache_prefill_log_start_index"] == 2
    assert record["model_id"] == CACHET_MODEL_ID
    assert record["served_model_name"] == SERVED_MODEL_NAME
    assert len(seen_configs) == 2
    assert events == [
        "baseline",
        "flush:before_cache_arm",
        "cache",
        "flush:before_model_quality_canary",
        "canary",
    ]
    assert seen_configs[0].model_id == SERVED_MODEL_NAME
    assert seen_configs[1].model_id == SERVED_MODEL_NAME
    assert seen_configs[0].use_cache_arm is False
    assert seen_configs[1].use_cache_arm is True
    assert seen_configs[0].temperature == DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE
    assert seen_configs[1].temperature == DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE
    assert seen_configs[0].extra_body == DEFAULT_SGLANG_LIVE_CHECK_EXTRA_BODY
    assert seen_configs[1].extra_body == DEFAULT_SGLANG_LIVE_CHECK_EXTRA_BODY
    assert seen_configs[1].kv_transfer_params_transport == "custom_params"
    assert seen_configs[1].prompt_text_mode == "logical"
    assert seen_configs[0].request_mode == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
    assert seen_configs[1].request_mode == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
    assert (
        seen_configs[1].kv_transfer_params[DOCUMENT_KV_REQUEST_ID_PARAM]
        == "cachet-live-sglang-1"
    )
    assert seen_configs[1].kv_transfer_params[
        DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM
    ] == [
        "page-a",
        "page-b",
    ]
    assert record["cache"]["arm_id"] == "document_kv_cache"
    assert record["cache_arm_cache_flush"]["ok"] is True
    assert record["cache_arm_cache_flush"]["reason"] == "before_cache_arm"
    assert record["canary_cache_flush"]["ok"] is True
    assert record["model_quality_canary"]["ok"] is True
    assert record["baseline"]["model_id"] == CACHET_MODEL_ID
    assert record["cache"]["model_id"] == CACHET_MODEL_ID
    assert record["baseline"]["served_model_name"] == SERVED_MODEL_NAME
    assert record["cache"]["served_model_name"] == SERVED_MODEL_NAME
    assert record["cache_arm_supported"] is True
    assert record["cache_arm_blocker"] is None
    assert record["requires_kv_transfer_params"] is True
    assert record["live_request_metadata_bridge_ok"] is True
    assert record["live_check_prompt_format"] == DEFAULT_SGLANG_LIVE_CHECK_PROMPT_FORMAT
    assert record["live_check_request_mode"] == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
    assert record["live_check_temperature"] == DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE
    assert record["live_check_extra_body"] == DEFAULT_SGLANG_LIVE_CHECK_EXTRA_BODY
    assert record["flush_cache_before_cache_arm"] is True
    assert record["flush_cache_before_canary"] is True
    assert record["live_check_chat_template_kwargs"] == {
        "reasoning_effort": "none",
        "thinking": False,
        "enable_thinking": False,
    }
    written = json.loads(config.live_smoke_output_path.read_text(encoding="utf-8"))
    assert written["cache"]["request_id"] == "cachet-live-sglang-1"
    assert written["cache_arm_cache_flush"]["ok"] is True


def test_run_live_checks_can_skip_cache_arm_cache_flush(monkeypatch, tmp_path):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(
        handoff_path, request_id="cachet-live-sglang-1", payload_uri=payload_uri
    )
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_json=str(handoff_path),
        payload_uri=payload_uri,
        request_id="cachet-live-sglang-1",
        sglang_hicache_page_keys=("page-a", "page-b"),
        flush_cache_before_cache_arm=False,
    )
    events = []

    def fake_live_check(live_config):
        events.append("cache" if live_config.use_cache_arm else "baseline")
        return FakeLiveResult(
            ok=True,
            request_id=live_config.kv_transfer_params.get(DOCUMENT_KV_REQUEST_ID_PARAM),
            prompt_text_mode=live_config.prompt_text_mode,
            request_mode=live_config.request_mode,
            cache_arm=live_config.use_cache_arm,
        )

    monkeypatch.setattr(
        public_sglang_smoke, "run_openai_compatible_live_check", fake_live_check
    )

    def fake_flush(live_config, *, reason):
        events.append(f"flush:{reason}")
        return fake_flush_record(reason=reason)

    monkeypatch.setattr(public_sglang_smoke, "flush_sglang_cache", fake_flush)
    monkeypatch.setattr(
        public_sglang_smoke,
        "run_sglang_quality_canary",
        lambda live_config: fake_canary_record(),
    )

    record = run_live_checks(
        config,
        import_probe_record={
            "ok": True,
            "document_kv_request_metadata_bridge_ok": True,
            "document_kv_request_metadata_bridge": {"ok": True},
        },
    )

    assert record["ok"] is True
    assert events == ["baseline", "cache", "flush:before_model_quality_canary"]
    assert record["flush_cache_before_cache_arm"] is False
    assert record["cache_arm_cache_flush"] == {
        "ok": None,
        "reason": "before_cache_arm",
        "endpoint": "/flush_cache",
        "skipped": True,
        "skip_reason": "disabled",
    }
    written = json.loads(config.live_smoke_output_path.read_text(encoding="utf-8"))
    assert written["cache_arm_cache_flush"]["skipped"] is True


def test_run_sglang_live_benchmark_writes_repeat_measurements(
    monkeypatch, tmp_path
):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(
        handoff_path, request_id="cachet-live-sglang-1", payload_uri=payload_uri
    )
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-benchmark-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_json=str(handoff_path),
        payload_uri=payload_uri,
        request_id="cachet-live-sglang-1",
        sglang_hicache_page_keys=("page-a", "page-b"),
        live_benchmark_repeats=2,
    )
    config.server_log_path.parent.mkdir(parents=True, exist_ok=True)
    events = []

    def fake_flush(live_config, *, reason):
        events.append(f"flush:{reason}")
        return fake_flush_record(reason=reason)

    def fake_live_check(live_config):
        cache_arm = live_config.use_cache_arm
        events.append("cache" if cache_arm else "baseline")
        prompt_tokens = 205
        completion_tokens = 1
        if cache_arm:
            with config.server_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "[INFO] Prefill batch, #new-token: 30, #cached-token: 175\n"
                )
            ttft_seconds = 0.1
            time_to_completion_seconds = 0.2
        else:
            with config.server_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "[INFO] Prefill batch, #new-token: 205, #cached-token: 0\n"
                )
            ttft_seconds = 0.4
            time_to_completion_seconds = 0.6
        request = public_sglang_smoke.build_live_server_check_request(
            model_id=SERVED_MODEL_NAME,
            hardware_target=live_config.hardware_target,
            use_cache_arm=cache_arm,
            prompt_format=public_sglang_smoke._live_check_request_prompt_format(
                live_config.prompt_format,
                request_mode=live_config.request_mode,
            ),
            kv_transfer_params=(
                live_config.kv_transfer_params if cache_arm else None
            ),
        )
        return public_sglang_smoke.LiveServerCheckResult(
            request=request,
            generation=BenchmarkGeneration(
                output_text="otkv7391",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_seconds=ttft_seconds,
                time_to_completion_seconds=time_to_completion_seconds,
                metadata={
                    "prompt_token_source": "server_usage",
                    "server_usage_prompt_tokens": str(prompt_tokens),
                },
            ),
            prompt_text_mode=live_config.prompt_text_mode,
            request_mode=live_config.request_mode,
            prompt_format=live_config.prompt_format,
            answer_found=True,
        )

    monkeypatch.setattr(public_sglang_smoke, "flush_sglang_cache", fake_flush)
    monkeypatch.setattr(
        public_sglang_smoke, "run_openai_compatible_live_check", fake_live_check
    )

    record = run_sglang_live_benchmark(config)

    assert record["ok"] is True
    assert record["record_type"] == SGLANG_LIVE_BENCHMARK_RECORD_TYPE
    assert record["suite"] == {
        "suite_id": "sglang-live-synthetic-niah",
        "scope": "live_synthetic_niah",
        "datasets": ["niah"],
        "examples": 1,
        "repeats": 2,
        "release_v1_suite": False,
    }
    assert events == [
        "flush:before_live_benchmark_baseline_niah/live-niah-synthetic-nonce_repeat_1",
        "baseline",
        "flush:before_live_benchmark_cache_niah/live-niah-synthetic-nonce_repeat_1",
        "cache",
        "flush:before_live_benchmark_baseline_niah/live-niah-synthetic-nonce_repeat_2",
        "baseline",
        "flush:before_live_benchmark_cache_niah/live-niah-synthetic-nonce_repeat_2",
        "cache",
    ]
    assert len(record["measurements"]) == 4
    assert [row["requests"] for row in record["report_rows"]] == [2, 2]
    assert record["comparisons"][0]["dataset"] == "niah"
    assert record["comparisons"][0]["ttft_speedup"] == 4.0
    assert len(record["cache_hit_validations"]) == 2
    assert all(item["ok"] is True for item in record["cache_hit_validations"])
    assert [
        item["cache_request_cached_tokens"]
        for item in record["cache_hit_validations"]
    ] == [175, 175]
    written = json.loads(config.live_benchmark_output_path.read_text(encoding="utf-8"))
    assert written == record


def test_run_sglang_live_benchmark_uses_prepared_v1_dataset_requests(
    monkeypatch, tmp_path
):
    dataset_specs = write_prepared_sglang_v1_datasets(tmp_path / "prepared")
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-prepared-v1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        dataset_specs=dataset_specs,
        live_benchmark_repeats=1,
    )
    config.server_log_path.parent.mkdir(parents=True, exist_ok=True)
    events = []

    def fake_flush(live_config, *, reason):
        del live_config
        events.append(f"flush:{reason}")
        return fake_flush_record(reason=reason)

    def fake_benchmark_request(live_config, *, request, use_cache_arm):
        events.append(
            f"{request.example.dataset}:{'cache' if use_cache_arm else 'baseline'}"
        )
        prompt_tokens = 40
        completion_tokens = 2
        expected_answer = request.example.expected_answer or ""
        if use_cache_arm:
            with live_config.server_log_path.open("a", encoding="utf-8") as handle:
                handle.write("[INFO] Prefill batch, #new-token: 8, #cached-token: 32\n")
            ttft_seconds = 0.1
            time_to_completion_seconds = 0.2
        else:
            with live_config.server_log_path.open("a", encoding="utf-8") as handle:
                handle.write("[INFO] Prefill batch, #new-token: 40, #cached-token: 0\n")
            ttft_seconds = 0.4
            time_to_completion_seconds = 0.6
        return public_sglang_smoke.LiveServerCheckResult(
            request=request,
            generation=BenchmarkGeneration(
                output_text=f"The answer is {expected_answer}.",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_seconds=ttft_seconds,
                time_to_completion_seconds=time_to_completion_seconds,
                metadata={
                    "prompt_token_source": "server_usage",
                    "server_usage_prompt_tokens": str(prompt_tokens),
                },
            ),
            prompt_text_mode=(
                live_config.cache_prompt_text_mode if use_cache_arm else "logical"
            ),
            request_mode=live_config.live_check_request_mode,
            prompt_format=live_config.live_check_prompt_format,
            answer_found=True,
        )

    monkeypatch.setattr(public_sglang_smoke, "flush_sglang_cache", fake_flush)
    monkeypatch.setattr(
        public_sglang_smoke,
        "_run_sglang_live_benchmark_request",
        fake_benchmark_request,
    )

    record = run_sglang_live_benchmark(config)

    assert record["ok"] is True
    assert record["suite"] == {
        "suite_id": "sglang-prepared-v1",
        "scope": SGLANG_PREPARED_V1_LIVE_BENCHMARK_SCOPE,
        "datasets": list(SUPPORTED_V1_DATASETS),
        "examples": len(SUPPORTED_V1_DATASETS),
        "repeats": 1,
        "release_v1_suite": True,
    }
    assert len(record["measurements"]) == len(SUPPORTED_V1_DATASETS) * 2
    assert len(record["comparisons"]) == len(SUPPORTED_V1_DATASETS)
    assert {
        comparison["dataset"]
        for comparison in record["comparisons"]
    } == set(SUPPORTED_V1_DATASETS)
    assert all(
        comparison["ttft_speedup"] == 4.0
        for comparison in record["comparisons"]
    )
    assert len(record["cache_hit_validations"]) == len(SUPPORTED_V1_DATASETS)
    assert all(item["ok"] is True for item in record["cache_hit_validations"])
    assert {
        item["dataset"]
        for item in record["cache_hit_validations"]
    } == set(SUPPORTED_V1_DATASETS)
    assert [
        item["minimum_cached_tokens"]
        for item in record["cache_hit_validations"]
    ] == [32, 32, 32, 32]
    assert [
        item["cache_request_cached_tokens"]
        for item in record["cache_hit_validations"]
    ] == [32, 32, 32, 32]
    assert events == [
        item
        for dataset in SUPPORTED_V1_DATASETS
        for item in (
            f"flush:before_live_benchmark_baseline_{dataset}/{dataset}-example-1_repeat_1",
            f"{dataset}:baseline",
            f"flush:before_live_benchmark_cache_{dataset}/{dataset}-example-1_repeat_1",
            f"{dataset}:cache",
        )
    ]
    written = json.loads(config.live_benchmark_output_path.read_text(encoding="utf-8"))
    assert written == record


def test_run_sglang_live_benchmark_records_request_failures(monkeypatch, tmp_path):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(
        handoff_path, request_id="cachet-live-sglang-1", payload_uri=payload_uri
    )
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-benchmark-failure",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_json=str(handoff_path),
        payload_uri=payload_uri,
        request_id="cachet-live-sglang-1",
        sglang_hicache_page_keys=("page-a",),
        live_benchmark_repeats=1,
    )
    config.server_log_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        public_sglang_smoke,
        "flush_sglang_cache",
        lambda live_config, *, reason: fake_flush_record(reason=reason),
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "run_openai_compatible_live_check",
        lambda live_config: (_ for _ in ()).throw(TimeoutError("timed out")),
    )

    record = run_sglang_live_benchmark(config)

    assert record["ok"] is False
    assert len(record["measurements"]) == 2
    assert record["measurements"][0]["error"] == "timed out"
    assert record["measurements"][1]["error"] == "timed out"
    assert record["cache_hit_validations"][0]["ok"] is False
    assert any("timed out" in issue for issue in record["issues"])
    written = json.loads(config.live_benchmark_output_path.read_text(encoding="utf-8"))
    assert written["ok"] is False


def test_run_live_checks_requires_bridge_probe_for_cache_arm(monkeypatch, tmp_path):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(
        handoff_path, request_id="cachet-live-sglang-1", payload_uri=payload_uri
    )
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        handoff_json=str(handoff_path),
        payload_uri=payload_uri,
        request_id="cachet-live-sglang-1",
        sglang_hicache_page_keys=("page-a", "page-b"),
    )

    monkeypatch.setattr(
        public_sglang_smoke,
        "run_openai_compatible_live_check",
        lambda live_config: FakeLiveResult(
            ok=True,
            request_id=live_config.kv_transfer_params.get(DOCUMENT_KV_REQUEST_ID_PARAM),
            prompt_text_mode=live_config.prompt_text_mode,
            request_mode=live_config.request_mode,
            cache_arm=live_config.use_cache_arm,
        ),
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "run_sglang_quality_canary",
        lambda live_config: fake_canary_record(),
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "flush_sglang_cache",
        lambda live_config, *, reason: fake_flush_record(reason=reason),
    )

    with pytest.raises(RuntimeError, match="request metadata bridge not verified"):
        run_live_checks(config)

    written = json.loads(config.live_smoke_output_path.read_text(encoding="utf-8"))
    assert written["ok"] is False
    assert written["cache_arm_cache_flush"]["ok"] is True
    assert written["canary_cache_flush"]["ok"] is True
    assert written["model_quality_canary"]["ok"] is True
    assert "request metadata bridge not verified" in written["issues"]


def test_run_live_checks_records_cache_arm_cache_flush_failure(monkeypatch, tmp_path):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(
        handoff_path, request_id="cachet-live-sglang-1", payload_uri=payload_uri
    )
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        handoff_json=str(handoff_path),
        payload_uri=payload_uri,
        request_id="cachet-live-sglang-1",
        sglang_hicache_page_keys=("page-a", "page-b"),
    )

    monkeypatch.setattr(
        public_sglang_smoke,
        "run_openai_compatible_live_check",
        lambda live_config: FakeLiveResult(
            ok=True,
            request_id=live_config.kv_transfer_params.get(DOCUMENT_KV_REQUEST_ID_PARAM),
            prompt_text_mode=live_config.prompt_text_mode,
            request_mode=live_config.request_mode,
            cache_arm=live_config.use_cache_arm,
        ),
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "run_sglang_quality_canary",
        lambda live_config: fake_canary_record(),
    )

    def fake_flush(live_config, *, reason):
        return fake_flush_record(ok=reason != "before_cache_arm", reason=reason)

    monkeypatch.setattr(public_sglang_smoke, "flush_sglang_cache", fake_flush)

    with pytest.raises(RuntimeError, match="cache-arm cache flush failed"):
        run_live_checks(
            config,
            import_probe_record={
                "ok": True,
                "document_kv_request_metadata_bridge_ok": True,
                "document_kv_request_metadata_bridge": {"ok": True},
            },
        )

    written = json.loads(config.live_smoke_output_path.read_text(encoding="utf-8"))
    assert written["ok"] is False
    assert written["cache_arm_cache_flush"]["ok"] is False
    assert written["cache_arm_cache_flush"]["reason"] == "before_cache_arm"
    assert written["canary_cache_flush"]["ok"] is True
    assert "cache-arm cache flush failed" in written["issues"]


def test_run_live_checks_records_model_quality_canary_failure(monkeypatch, tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )

    monkeypatch.setattr(
        public_sglang_smoke,
        "run_sglang_quality_canary",
        lambda live_config: fake_canary_record(ok=False),
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "flush_sglang_cache",
        lambda live_config, *, reason: fake_flush_record(reason=reason),
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "run_openai_compatible_live_check",
        lambda live_config: FakeLiveResult(
            ok=True,
            request_id=None,
            prompt_text_mode=live_config.prompt_text_mode,
            request_mode=live_config.request_mode,
            cache_arm=live_config.use_cache_arm,
        ),
    )

    with pytest.raises(RuntimeError, match="model quality canary failed"):
        run_live_checks(config)

    written = json.loads(config.live_smoke_output_path.read_text(encoding="utf-8"))
    assert written["ok"] is False
    assert written["canary_cache_flush"]["ok"] is True
    assert written["model_quality_canary"]["ok"] is False
    assert "model quality canary failed" in written["issues"]


def test_run_live_checks_records_canary_cache_flush_failure(monkeypatch, tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )

    monkeypatch.setattr(
        public_sglang_smoke,
        "run_sglang_quality_canary",
        lambda live_config: fake_canary_record(),
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "flush_sglang_cache",
        lambda live_config, *, reason: fake_flush_record(ok=False, reason=reason),
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "run_openai_compatible_live_check",
        lambda live_config: FakeLiveResult(
            ok=True,
            request_id=None,
            prompt_text_mode=live_config.prompt_text_mode,
            request_mode=live_config.request_mode,
            cache_arm=live_config.use_cache_arm,
        ),
    )

    with pytest.raises(RuntimeError, match="model quality canary cache flush failed"):
        run_live_checks(config)

    written = json.loads(config.live_smoke_output_path.read_text(encoding="utf-8"))
    assert written["ok"] is False
    assert written["model_quality_canary"]["ok"] is True
    assert written["canary_cache_flush"]["ok"] is False
    assert "model quality canary cache flush failed" in written["issues"]


def test_sglang_cache_hit_validation_reads_cache_arm_cached_tokens(tmp_path):
    log_path = tmp_path / "sglang-server.log"
    log_path.write_text(
        "[INFO] Prefill batch, #new-token: 6, #cached-token: 0\n"
        "[INFO] Prefill batch, #new-token: 1, #cached-token: 0\n"
        "[INFO] Prefill batch, #new-token: 25, #cached-token: 150\n"
        "[INFO] Prefill batch, #new-token: 1, #cached-token: 174\n",
        encoding="utf-8",
    )

    assert sglang_cached_token_counts(log_path.read_text(encoding="utf-8")) == (
        0,
        0,
        150,
        174,
    )
    assert sglang_prefill_token_counts(log_path.read_text(encoding="utf-8")) == (
        {"new_tokens": 6, "cached_tokens": 0, "total_prompt_tokens": 6},
        {"new_tokens": 1, "cached_tokens": 0, "total_prompt_tokens": 1},
        {"new_tokens": 25, "cached_tokens": 150, "total_prompt_tokens": 175},
        {"new_tokens": 1, "cached_tokens": 174, "total_prompt_tokens": 175},
    )
    record = sglang_cache_hit_validation_record(
        log_path, cache_request_prompt_tokens=175
    )

    assert record["ok"] is True
    assert record["cached_token_counts"] == [0, 0, 150, 174]
    assert record["cache_request_prompt_tokens"] == 175
    assert record["cache_request_prefill_index"] == 2
    assert record["cache_request_cached_tokens"] == 150
    assert record["issue"] is None


def test_sglang_cache_hit_validation_requires_generated_prefix_minimum(tmp_path):
    log_path = tmp_path / "sglang-server.log"
    log_path.write_text(
        "[INFO] Prefill batch, #new-token: 46, #cached-token: 128\n",
        encoding="utf-8",
    )

    record = sglang_cache_hit_validation_record(
        log_path,
        cache_request_prompt_tokens=174,
        minimum_cached_tokens=150,
    )

    assert record["ok"] is False
    assert record["minimum_cached_tokens"] == 150
    assert record["cache_request_cached_tokens"] == 128
    assert (
        record["issue"]
        == "SGLang cache arm cached fewer tokens than the generated handoff prefix"
    )


def test_sglang_cache_hit_validation_rejects_later_baseline_warm_hit(tmp_path):
    log_path = tmp_path / "sglang-server.log"
    log_path.write_text(
        "[INFO] Prefill batch, #new-token: 174, #cached-token: 0\n"
        "[INFO] Prefill batch, #new-token: 1, #cached-token: 173\n",
        encoding="utf-8",
    )

    record = sglang_cache_hit_validation_record(
        log_path, cache_request_prompt_tokens=174
    )

    assert record["ok"] is False
    assert record["cached_token_counts"] == [0, 173]
    assert record["cache_request_prompt_tokens"] == 174
    assert record["cache_request_prompt_match_field"] == "total_prompt_tokens"
    assert record["cache_request_prefill_index"] == 0
    assert record["cache_request_cached_tokens"] == 0
    assert record["issue"] == "SGLang cache arm reported zero cached tokens"


def test_sglang_cache_hit_validation_matches_cache_arm_after_warmups(tmp_path):
    log_path = tmp_path / "sglang-server.log"
    log_path.write_text(
        "[INFO] Prefill batch, #new-token: 6, #cached-token: 0\n"
        "[INFO] Prefill batch, #new-token: 1, #cached-token: 0\n"
        "[INFO] Prefill batch, #new-token: 174, #cached-token: 0\n"
        "[INFO] Prefill batch, #new-token: 1, #cached-token: 173\n",
        encoding="utf-8",
    )

    record = sglang_cache_hit_validation_record(
        log_path,
        cache_request_prompt_tokens=174,
        cache_request_prefill_start_index=2,
    )

    assert record["ok"] is False
    assert record["cache_request_prefill_start_index"] == 2
    assert record["cache_request_prefill_index"] == 2
    assert record["cache_request_cached_tokens"] == 0
    assert record["prefill_token_counts"] == [
        {"new_tokens": 6, "cached_tokens": 0, "total_prompt_tokens": 6},
        {"new_tokens": 1, "cached_tokens": 0, "total_prompt_tokens": 1},
        {"new_tokens": 174, "cached_tokens": 0, "total_prompt_tokens": 174},
        {"new_tokens": 1, "cached_tokens": 173, "total_prompt_tokens": 174},
    ]
    assert record["issue"] == "SGLang cache arm reported zero cached tokens"


def test_sglang_cache_hit_validation_uses_log_offset_to_skip_matching_warmup_total(
    tmp_path,
):
    log_path = tmp_path / "sglang-server.log"
    log_path.write_text(
        "[INFO] Prefill batch, #new-token: 174, #cached-token: 0\n"
        "[INFO] Prefill batch, #new-token: 24, #cached-token: 150\n",
        encoding="utf-8",
    )

    record = sglang_cache_hit_validation_record(
        log_path,
        cache_request_prompt_tokens=174,
        cache_request_prefill_start_index=1,
    )

    assert record["ok"] is True
    assert record["cache_request_prefill_start_index"] == 1
    assert record["cache_request_prefill_index"] == 1
    assert record["cache_request_cached_tokens"] == 150


def test_sglang_cache_hit_validation_matches_runtime_prompt_mode_by_new_tokens(
    tmp_path,
):
    log_path = tmp_path / "sglang-server.log"
    log_path.write_text(
        "[INFO] Prefill batch, #new-token: 25, #cached-token: 150\n",
        encoding="utf-8",
    )

    record = sglang_cache_hit_validation_record(
        log_path,
        cache_request_prompt_tokens=25,
        cache_prompt_text_mode="runtime",
    )

    assert record["ok"] is True
    assert record["cache_prompt_text_mode"] == "runtime"
    assert record["cache_request_prompt_match_field"] == "new_tokens"
    assert record["cache_request_prefill_index"] == 0
    assert record["cache_request_cached_tokens"] == 150


def test_sglang_cache_hit_validation_requires_cache_request_prompt_tokens(tmp_path):
    log_path = tmp_path / "sglang-server.log"
    log_path.write_text(
        "[INFO] Prefill batch, #new-token: 2, #cached-token: 173\n",
        encoding="utf-8",
    )

    record = sglang_cache_hit_validation_record(log_path)

    assert record["ok"] is False
    assert record["cache_request_prefill_index"] is None
    assert record["cache_request_cached_tokens"] is None
    assert (
        record["issue"]
        == "SGLang cache request prompt token count unavailable; cannot verify cache-arm cached tokens"
    )


def test_sglang_cache_hit_validation_rejects_missing_cache_request_prompt_total(
    tmp_path,
):
    log_path = tmp_path / "sglang-server.log"
    log_path.write_text(
        "[INFO] Prefill batch, #new-token: 2, #cached-token: 173\n",
        encoding="utf-8",
    )

    record = sglang_cache_hit_validation_record(
        log_path, cache_request_prompt_tokens=174
    )

    assert record["ok"] is False
    assert record["cache_request_prefill_index"] is None
    assert record["cache_request_cached_tokens"] is None
    assert (
        record["issue"]
        == "SGLang server log did not report cache request prompt token count"
    )


def test_require_sglang_cache_hit_rejects_zero_cache_tokens(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_generation=SGLangLiveHandoffGenerationConfig(
            output_dir=tmp_path / "generated-live"
        ),
        live_benchmark_repeats=1,
    )
    config.server_log_path.parent.mkdir(parents=True, exist_ok=True)
    config.server_log_path.write_text(
        "[INFO] Prefill batch, #new-token: 174, #cached-token: 0\n"
        "[INFO] Prefill batch, #new-token: 174, #cached-token: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="zero cached tokens"):
        require_sglang_cache_hit(
            config,
            {
                "ok": True,
                "issues": [],
                "cache": {
                    "metadata": {
                        "server_usage_prompt_tokens": "174",
                    },
                    "prompt_tokens": 92,
                },
            },
        )

    written = json.loads(config.live_smoke_output_path.read_text(encoding="utf-8"))
    assert written["ok"] is False
    assert written["sglang_cache_hit_validation"]["cache_request_cached_tokens"] == 0
    assert "SGLang cache arm reported zero cached tokens" in written["issues"]


def test_require_sglang_cache_hit_rejects_partial_generated_handoff_hit(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_generation=SGLangLiveHandoffGenerationConfig(
            output_dir=tmp_path / "generated-live"
        ),
    )
    config.server_log_path.parent.mkdir(parents=True, exist_ok=True)
    config.server_log_path.write_text(
        "[INFO] Prefill batch, #new-token: 46, #cached-token: 128\n",
        encoding="utf-8",
    )
    public_sglang_smoke.write_json(
        config.live_handoff_generation_path,
        {"ok": True, "cache_prefix_tokens": 150},
    )

    with pytest.raises(RuntimeError, match="fewer tokens"):
        require_sglang_cache_hit(
            config,
            {
                "ok": True,
                "issues": [],
                "cache_prefill_log_start_index": 0,
                "cache": {
                    "metadata": {
                        "server_usage_prompt_tokens": "174",
                    },
                    "prompt_tokens": 92,
                },
            },
        )

    written = json.loads(config.live_smoke_output_path.read_text(encoding="utf-8"))
    assert written["ok"] is False
    assert written["sglang_cache_hit_validation"]["minimum_cached_tokens"] == 150
    assert written["sglang_cache_hit_validation"]["cache_request_cached_tokens"] == 128
    assert (
        "SGLang cache arm cached fewer tokens than the generated handoff prefix"
        in written["issues"]
    )


def test_require_sglang_cache_hit_requires_server_usage_prompt_tokens(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_generation=SGLangLiveHandoffGenerationConfig(
            output_dir=tmp_path / "generated-live"
        ),
    )
    config.server_log_path.parent.mkdir(parents=True, exist_ok=True)
    config.server_log_path.write_text(
        "[INFO] Prefill batch, #new-token: 1, #cached-token: 173\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="prompt token count unavailable"):
        require_sglang_cache_hit(
            config,
            {
                "ok": True,
                "issues": [],
                "cache": {
                    "prompt_tokens": 92,
                },
            },
        )

    written = json.loads(config.live_smoke_output_path.read_text(encoding="utf-8"))
    assert written["ok"] is False
    assert (
        written["sglang_cache_hit_validation"]["issue"]
        == "SGLang cache request prompt token count unavailable; cannot verify cache-arm cached tokens"
    )


def test_run_sglang_live_smoke_uses_generated_handoff_for_live_checks(
    monkeypatch, tmp_path
):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-generated-run",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_generation=SGLangLiveHandoffGenerationConfig(
            output_dir=tmp_path / "generated-live"
        ),
    )
    events = []
    runtime_config = SGLangSmokeBenchmarkConfig(
        benchmark_id=config.benchmark_id,
        output_dir=config.output_dir,
        local_root=config.local_root,
        handoff_json=str(tmp_path / "generated-live" / "sglang-live.handoff.json"),
        payload_uri=f"disk:{tmp_path / 'generated-live' / 'sglang-live.kv'}",
        request_id="cachet-live-sglang-generated-run",
        sglang_hicache_page_keys=("page-a",),
        live_benchmark_repeats=1,
    )

    class FakeServer:
        pass

    def fake_prepare(seen_config):
        events.append("prepare")
        assert seen_config is config
        return runtime_config

    def fake_start(seen_config, python_executable, log_path):
        events.append("start")
        assert seen_config is runtime_config
        assert python_executable == runtime_config.venv_python
        assert log_path == runtime_config.server_log_path
        return FakeServer()

    def fake_run_live_checks(seen_config, *, import_probe_record):
        events.append("live")
        assert seen_config is runtime_config
        assert import_probe_record["document_kv_request_metadata_bridge_ok"] is True
        runtime_config.server_log_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_config.server_log_path.write_text(
            "[INFO] Prefill batch, #new-token: 25, #cached-token: 150\n"
            "[INFO] Prefill batch, #new-token: 174, #cached-token: 0\n",
            encoding="utf-8",
        )
        return {
            "ok": True,
            "issues": [],
            "cache_prefill_log_start_index": 0,
            "cache": {
                "metadata": {
                    "server_usage_prompt_tokens": "175",
                },
                "prompt_tokens": 92,
            },
        }

    monkeypatch.setattr(
        public_sglang_smoke, "create_venv", lambda path: events.append("venv")
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "install_sglang",
        lambda python: events.append("install-sglang"),
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "install_document_kv_package",
        lambda python, spec: events.append("install-cachet"),
    )
    monkeypatch.setattr(public_sglang_smoke, "installed_versions", lambda python: {})
    monkeypatch.setattr(
        public_sglang_smoke,
        "probe_sglang_import",
        lambda *args, **kwargs: {
            "ok": True,
            "document_kv_request_metadata_bridge_ok": True,
            "document_kv_request_metadata_bridge": {"ok": True},
        },
    )
    monkeypatch.setattr(
        public_sglang_smoke, "prepare_generated_live_handoff", fake_prepare
    )
    monkeypatch.setattr(public_sglang_smoke, "start_sglang_server", fake_start)
    monkeypatch.setattr(
        public_sglang_smoke,
        "wait_for_sglang_server",
        lambda *args, **kwargs: events.append("wait"),
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "copy_file_if_exists",
        lambda *args, **kwargs: events.append("copy"),
    )
    monkeypatch.setattr(public_sglang_smoke, "run_live_checks", fake_run_live_checks)
    monkeypatch.setattr(
        public_sglang_smoke,
        "run_sglang_live_benchmark",
        lambda seen_config: (
            events.append("benchmark")
            or {"ok": True, "issues": [], "benchmark_id": seen_config.benchmark_id}
        ),
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "terminate_process",
        lambda server: events.append("terminate"),
    )

    public_sglang_smoke.run_sglang_live_smoke(config)

    assert events.index("prepare") < events.index("start") < events.index("live")
    assert events.index("live") < events.index("benchmark") < events.index("terminate")
    metadata = json.loads(config.metadata_path.read_text(encoding="utf-8"))
    assert metadata["generated_live_handoff"] is True
    assert metadata["generated_live_handoff_page_keys"] == 1
    record = json.loads(
        runtime_config.live_smoke_output_path.read_text(encoding="utf-8")
    )
    assert record["sglang_cache_hit_validation"]["cache_request_cached_tokens"] == 150


def test_run_sglang_live_smoke_records_cache_hit_when_quality_fails(
    monkeypatch, tmp_path
):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-quality-failure-run",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        handoff_generation=SGLangLiveHandoffGenerationConfig(
            output_dir=tmp_path / "generated-live"
        ),
    )
    runtime_config = SGLangSmokeBenchmarkConfig(
        benchmark_id=config.benchmark_id,
        output_dir=config.output_dir,
        local_root=config.local_root,
        handoff_json=str(tmp_path / "generated-live" / "sglang-live.handoff.json"),
        payload_uri=f"disk:{tmp_path / 'generated-live' / 'sglang-live.kv'}",
        request_id="cachet-live-sglang-quality-failure-run",
        sglang_hicache_page_keys=("page-a",),
    )

    class FakeServer:
        pass

    def fake_run_live_checks(seen_config, *, import_probe_record):
        assert seen_config is runtime_config
        assert import_probe_record["document_kv_request_metadata_bridge_ok"] is True
        runtime_config.server_log_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_config.server_log_path.write_text(
            "[INFO] Prefill batch, #new-token: 46, #cached-token: 128\n",
            encoding="utf-8",
        )
        public_sglang_smoke.write_json(
            runtime_config.live_smoke_output_path,
            {
                "ok": False,
                "issues": ["baseline live check failed", "cache-arm live check failed"],
                "cache_prefill_log_start_index": 0,
                "cache": {
                    "metadata": {
                        "server_usage_prompt_tokens": "174",
                    },
                    "prompt_tokens": 92,
                },
            },
        )
        raise RuntimeError(
            "SGLang live smoke failed: baseline live check failed; cache-arm live check failed"
        )

    monkeypatch.setattr(public_sglang_smoke, "create_venv", lambda path: None)
    monkeypatch.setattr(public_sglang_smoke, "install_sglang", lambda python: None)
    monkeypatch.setattr(
        public_sglang_smoke, "install_document_kv_package", lambda python, spec: None
    )
    monkeypatch.setattr(public_sglang_smoke, "installed_versions", lambda python: {})
    monkeypatch.setattr(
        public_sglang_smoke,
        "probe_sglang_import",
        lambda *args, **kwargs: {
            "ok": True,
            "document_kv_request_metadata_bridge_ok": True,
            "document_kv_request_metadata_bridge": {"ok": True},
        },
    )
    monkeypatch.setattr(
        public_sglang_smoke,
        "prepare_generated_live_handoff",
        lambda seen_config: runtime_config,
    )
    monkeypatch.setattr(
        public_sglang_smoke, "start_sglang_server", lambda *args, **kwargs: FakeServer()
    )
    monkeypatch.setattr(
        public_sglang_smoke, "wait_for_sglang_server", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        public_sglang_smoke, "copy_file_if_exists", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(public_sglang_smoke, "run_live_checks", fake_run_live_checks)
    monkeypatch.setattr(public_sglang_smoke, "terminate_process", lambda server: None)

    with pytest.raises(RuntimeError, match="baseline live check failed"):
        public_sglang_smoke.run_sglang_live_smoke(config)

    record = json.loads(
        runtime_config.live_smoke_output_path.read_text(encoding="utf-8")
    )
    assert record["ok"] is False
    assert record["issues"] == [
        "baseline live check failed",
        "cache-arm live check failed",
    ]
    assert record["sglang_cache_hit_validation"]["ok"] is True
    assert record["sglang_cache_hit_validation"]["cache_request_cached_tokens"] == 128


def test_parse_args_builds_baseline_only_config(tmp_path):
    config = parse_args(
        [
            "--benchmark-id",
            "sglang-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--baseline-only",
            "--hardware-target",
            "aws-g5-a10g",
            "--sglang-attention-backend",
            "triton",
            "--sglang-sampling-backend",
            "pytorch",
            "--sglang-enable-deterministic-inference",
            "--no-stream",
        ]
    )

    assert config.baseline_only is True
    assert config.hardware_target == "aws-g5-a10g"
    assert config.sglang_attention_backend == "triton"
    assert config.sglang_sampling_backend == "pytorch"
    assert config.sglang_enable_deterministic_inference is True
    assert config.stream is False
    assert config.cache_prompt_text_mode == "logical"
    assert config.live_check_temperature == DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE
    assert config.live_benchmark_repeats == DEFAULT_SGLANG_LIVE_BENCHMARK_REPEATS

    with pytest.raises(ValueError) as exc:
        parse_args(
            [
                "--benchmark-id",
                "sglang-1",
                "--output-dir",
                str(tmp_path / "out"),
                "--baseline-only",
                "--sglang-hicache-page-keys-json",
                "[]",
            ]
        )

    assert str(exc.value) == SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE


def test_parse_args_accepts_live_benchmark_repeats_for_cache_arm(tmp_path):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(
        handoff_path, request_id="cachet-live-sglang-1", payload_uri=payload_uri
    )

    config = parse_args(
        [
            "--benchmark-id",
            "sglang-benchmark-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--handoff-json",
            str(handoff_path),
            "--payload-uri",
            payload_uri,
            "--request-id",
            "cachet-live-sglang-1",
            "--sglang-hicache-page-keys-json",
            '["page-a"]',
            "--live-benchmark-repeats",
            "3",
        ]
    )

    assert config.live_benchmark_repeats == 3

    with pytest.raises(ValueError, match="live_benchmark_repeats"):
        parse_args(
            [
                "--benchmark-id",
                "sglang-baseline-1",
                "--output-dir",
                str(tmp_path / "out"),
                "--baseline-only",
                "--live-benchmark-repeats",
                "1",
            ]
        )


def test_probe_sglang_import_writes_timeout_artifact(monkeypatch, tmp_path):
    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["python"], timeout=3, output="partial out", stderr="partial err"
        )

    monkeypatch.setattr(public_sglang_smoke.subprocess, "run", timeout_run)
    output_path = tmp_path / "probe.json"

    with pytest.raises(RuntimeError, match="timed out"):
        public_sglang_smoke.probe_sglang_import(
            tmp_path / "python",
            output_path,
            launch_config_path=tmp_path / "launch.json",
            timeout_seconds=3,
        )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["ok"] is False
    assert record["error_type"] == "TimeoutExpired"
    assert "partial out" in record["stdout_tail"]
    assert "partial err" in record["stderr_tail"]


def test_probe_sglang_import_requires_request_metadata_bridge(monkeypatch, tmp_path):
    def successful_probe_without_bridge(argv, **kwargs):
        payload = {
            "ok": True,
            "document_kv_request_metadata_bridge_ok": False,
            "document_kv_request_metadata_bridge": {
                "ok": False,
                "reason": "missing patch",
            },
        }
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(payload) + "\n", stderr=""
        )

    monkeypatch.setattr(
        public_sglang_smoke.subprocess, "run", successful_probe_without_bridge
    )
    launch_config_path = tmp_path / "launch.json"
    launch_config_path.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "probe.json"

    with pytest.raises(RuntimeError, match="request metadata bridge"):
        public_sglang_smoke.probe_sglang_import(
            tmp_path / "python",
            output_path,
            launch_config_path=launch_config_path,
            timeout_seconds=3,
        )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["ok"] is False
    assert record["error_type"] == "SGLangRequestMetadataBridgeUnavailable"
