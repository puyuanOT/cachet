import json
from urllib.error import HTTPError

import pytest

import document_kv_cache.openai_compatible as openai_module
import restaurant_kv_serving.openai_compatible as legacy_openai_module
from document_kv_cache.benchmark_runner import BenchmarkEngineRequest
from document_kv_cache.benchmarks import BenchmarkExample, build_prompt_parts, document_kv_cache_arm
from document_kv_cache.openai_compatible import (
    OpenAICompatibleCompletionEngine,
    OpenAICompatibleEngineConfig,
    WhitespaceTokenCounter,
)
from document_kv_cache.workflow import SourceDocument


class FakeClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        if not self.values:
            raise AssertionError("FakeClock exhausted")
        return self.values.pop(0)


class FakeStreamResponse:
    def __init__(self, lines=None):
        self.lines = lines or [
            b'data: {"choices":[{"text":"Ada"}]}\n',
            b"\n",
            b'data: {"choices":[{"text":" Lovelace"}]}\n',
            b"\n",
            b'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":2}}\n',
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


class FakeJSONResponse:
    def __init__(self, data=None):
        self.data = data or {
            "choices": [{"text": "Ada Lovelace"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2},
        }
        self.closed = False

    def read(self):
        return json.dumps(self.data).encode("utf-8")

    def close(self):
        self.closed = True


class CapturingEngine(OpenAICompatibleCompletionEngine):
    def __init__(self, *args, response, **kwargs):
        super().__init__(*args, **kwargs)
        self.response = response
        self.payloads = []
        self.headers = {}

    def _post_json(self, payload):
        self.payloads.append(dict(payload))
        self.headers = self._headers()
        return self.response


class FixedPromptTokenCounter:
    def __init__(self, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens

    def count(self, text):
        if "Documents:" in text:
            return self.prompt_tokens
        return len(text.split())


class FakeErrorBody:
    def __init__(self):
        self.closed = False

    def read(self):
        return b"bad request body"

    def close(self):
        self.closed = True


def benchmark_request() -> BenchmarkEngineRequest:
    example = BenchmarkExample(
        example_id="bio-1",
        dataset="biography",
        documents=(
            SourceDocument.from_texts(
                document_id="doc-1",
                static_text="Ada Lovelace biography",
                chunks={"p1": "Lovelace wrote notes."},
            ),
        ),
        query="Who wrote notes?",
        expected_answer="Ada Lovelace",
    )
    return BenchmarkEngineRequest(
        suite_id="suite",
        model_id="qwen3:4b-instruct",
        hardware_target="aws-g5",
        example=example,
        arm=document_kv_cache_arm(),
        prompt_parts=build_prompt_parts(example),
    )


def test_streaming_completion_engine_measures_ttft_and_uses_logical_prompt_by_default():
    response = FakeStreamResponse()
    engine = CapturingEngine(
        OpenAICompatibleEngineConfig(
            base_url="http://localhost:8000",
            api_key="token",
            max_tokens=32,
            extra_body={"top_p": 0.9},
            extra_headers={"X-Test": "yes"},
        ),
        response=response,
        token_counter=FixedPromptTokenCounter(99),
        clock=FakeClock([0.0, 0.25, 0.75]),
    )

    generation = engine.generate(benchmark_request())

    assert generation.output_text == "Ada Lovelace"
    assert generation.prompt_tokens == 99
    assert generation.completion_tokens == 2
    assert generation.ttft_seconds == pytest.approx(0.25)
    assert generation.time_to_completion_seconds == pytest.approx(0.75)
    assert generation.metadata["prompt_token_source"] == "logical"
    assert generation.metadata["prompt_text_mode"] == "logical"
    request_body = engine.payloads[0]
    assert request_body["prompt"] == benchmark_request().logical_prompt_text
    assert request_body["model"] == "qwen3:4b-instruct"
    assert request_body["stream"] is True
    assert request_body["stream_options"] == {"include_usage": True}
    assert request_body["top_p"] == 0.9
    assert engine.headers["Authorization"] == "Bearer token"
    assert engine.headers["X-Test"] == "yes"
    assert response.closed is True


def test_runtime_prompt_mode_uses_cache_suffix_for_kv_aware_proxy():
    engine = CapturingEngine(
        OpenAICompatibleEngineConfig(
            base_url="http://localhost:8000",
            stream=False,
            prompt_text_mode="runtime",
            prompt_token_accounting="server_usage",
        ),
        response=FakeJSONResponse(),
        clock=FakeClock([1.0, 2.0]),
    )

    generation = engine.generate(benchmark_request())

    assert generation.prompt_tokens == 12
    assert generation.metadata["prompt_token_source"] == "server_usage"
    assert generation.metadata["prompt_text_mode"] == "runtime"
    assert engine.payloads[0]["prompt"] == benchmark_request().cache_suffix_text


def test_non_streaming_completion_engine_uses_usage_and_total_latency():
    response = FakeJSONResponse()
    engine = CapturingEngine(
        OpenAICompatibleEngineConfig(
            base_url="http://localhost:8000",
            stream=False,
            model_id="served-qwen",
            prompt_token_accounting="server_usage",
        ),
        response=response,
        clock=FakeClock([10.0, 12.5]),
    )

    generation = engine.generate(benchmark_request())

    assert generation.output_text == "Ada Lovelace"
    assert generation.prompt_tokens == 12
    assert generation.completion_tokens == 2
    assert generation.metadata["prompt_token_source"] == "server_usage"
    assert generation.ttft_seconds == pytest.approx(2.5)
    assert generation.time_to_completion_seconds == pytest.approx(2.5)
    assert engine.payloads[0]["model"] == "served-qwen"
    assert engine.payloads[0]["stream"] is False
    assert response.closed is True


def test_missing_usage_uses_logical_prompt_and_output_fallback_counts():
    request = benchmark_request()
    engine = CapturingEngine(
        OpenAICompatibleEngineConfig(
            base_url="http://localhost:8000",
            stream=False,
        ),
        response=FakeJSONResponse({"choices": [{"text": "Ada Lovelace"}]}),
        clock=FakeClock([10.0, 12.5]),
    )

    generation = engine.generate(request)

    assert generation.prompt_tokens == WhitespaceTokenCounter().count(request.logical_prompt_text)
    assert generation.completion_tokens == 2
    assert generation.metadata["prompt_token_source"] == "logical"


def test_server_usage_accounting_labels_missing_usage_fallback():
    request = benchmark_request()
    engine = CapturingEngine(
        OpenAICompatibleEngineConfig(
            base_url="http://localhost:8000",
            stream=False,
            prompt_token_accounting="server_usage",
        ),
        response=FakeJSONResponse({"choices": [{"text": "Ada Lovelace"}]}),
        clock=FakeClock([10.0, 12.5]),
    )

    generation = engine.generate(request)

    assert generation.prompt_tokens == WhitespaceTokenCounter().count(request.logical_prompt_text)
    assert generation.metadata["prompt_token_source"] == "logical_fallback"


def test_streaming_error_payload_raises_and_closes_response():
    response = FakeStreamResponse(
        [
            b'data: {"error":{"message":"bad request"}}\n',
            b"\n",
        ]
    )
    engine = CapturingEngine(
        OpenAICompatibleEngineConfig(base_url="http://localhost:8000"),
        response=response,
        clock=FakeClock([0.0]),
    )

    with pytest.raises(RuntimeError, match="bad request"):
        engine.generate(benchmark_request())
    assert response.closed is True


def test_non_streaming_error_payload_raises_and_closes_response():
    response = FakeJSONResponse({"error": {"message": "bad request"}})
    engine = CapturingEngine(
        OpenAICompatibleEngineConfig(base_url="http://localhost:8000", stream=False),
        response=response,
        clock=FakeClock([0.0]),
    )

    with pytest.raises(RuntimeError, match="bad request"):
        engine.generate(benchmark_request())
    assert response.closed is True


def test_http_error_body_is_closed_after_wrapping(monkeypatch):
    body = FakeErrorBody()

    def raise_http_error(*args, **kwargs):
        raise HTTPError("http://localhost:8000/v1/completions", 400, "bad request", {}, body)

    monkeypatch.setattr(legacy_openai_module, "urlopen", raise_http_error)
    engine = OpenAICompatibleCompletionEngine(
        OpenAICompatibleEngineConfig(base_url="http://localhost:8000", stream=False),
        clock=FakeClock([0.0]),
    )

    with pytest.raises(RuntimeError, match="HTTP 400: bad request body"):
        engine.generate(benchmark_request())
    assert body.closed is True


def test_document_urlopen_hook_still_wraps_http_errors(monkeypatch):
    body = FakeErrorBody()

    def raise_http_error(*args, **kwargs):
        raise HTTPError("http://localhost:8000/v1/completions", 400, "bad request", {}, body)

    monkeypatch.setattr(openai_module, "_urlopen", raise_http_error)
    engine = OpenAICompatibleCompletionEngine(
        OpenAICompatibleEngineConfig(base_url="http://localhost:8000", stream=False),
        clock=FakeClock([0.0]),
    )

    with pytest.raises(RuntimeError, match="HTTP 400: bad request body"):
        engine.generate(benchmark_request())
    assert body.closed is True


def test_whitespace_token_counter_is_available_as_local_fallback():
    assert WhitespaceTokenCounter().count("Ada Lovelace wrote notes") == 4


def test_legacy_module_reexports_document_owned_engine():
    assert OpenAICompatibleCompletionEngine.__module__ == "document_kv_cache.openai_compatible"
    assert legacy_openai_module.OpenAICompatibleCompletionEngine is OpenAICompatibleCompletionEngine
    assert legacy_openai_module.OpenAICompatibleEngineConfig is OpenAICompatibleEngineConfig
    assert set(openai_module.__all__) < set(legacy_openai_module.__all__)
    assert "urlopen" in legacy_openai_module.__all__
