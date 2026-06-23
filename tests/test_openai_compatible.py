import json
import math
from urllib.error import HTTPError

import pytest

import document_kv_cache.openai_compatible as openai_module
import restaurant_kv_serving.openai_compatible as legacy_openai_module
from document_kv_cache.benchmark_runner import BenchmarkEngineRequest
from document_kv_cache.benchmarks import (
    DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    BenchmarkExample,
    build_prompt_parts,
    document_kv_cache_arm,
)
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


def benchmark_request(kv_transfer_params=None, *, repeat_index: int = 1) -> BenchmarkEngineRequest:
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
        kv_transfer_params=kv_transfer_params or {},
    )
    return BenchmarkEngineRequest(
        suite_id="suite",
        model_id="qwen3:4b-instruct",
        hardware_target="aws-g6-l4",
        example=example,
        arm=document_kv_cache_arm(),
        prompt_parts=build_prompt_parts(example),
        request_id=example.kv_transfer_params.get(DOCUMENT_KV_REQUEST_ID_PARAM),
        kv_transfer_params=example.kv_transfer_params,
        repeat_index=repeat_index,
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
    assert generation.metadata["logical_prompt_tokens"] == "99"
    assert generation.metadata["runtime_prompt_tokens"] == "99"
    assert generation.metadata["kv_transfer_params_attached"] == "false"
    assert "request_id" not in generation.metadata
    request_body = engine.payloads[0]
    assert request_body["prompt"] == benchmark_request().logical_prompt_text
    assert request_body["model"] == "qwen3:4b-instruct"
    assert request_body["stream"] is True
    assert request_body["stream_options"] == {"include_usage": True}
    assert request_body["top_p"] == 0.9
    assert "request_id" not in request_body
    assert "kv_transfer_params" not in request_body
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
    assert int(generation.metadata["logical_prompt_tokens"]) > int(generation.metadata["runtime_prompt_tokens"])
    assert engine.payloads[0]["prompt"] == benchmark_request().cache_suffix_text


def test_runtime_prompt_mode_sends_request_kv_transfer_params():
    kv_transfer_params = {
        DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
        "document_kv.handoff_json": "/Volumes/catalog/schema/volume/cachet/bio-1.handoff.json",
        "document_kv.payload_uri": "uc-volume:/catalog/schema/volume/cachet/bio-1.kv",
    }
    engine = CapturingEngine(
        OpenAICompatibleEngineConfig(
            base_url="http://localhost:8000",
            stream=False,
            prompt_text_mode="runtime",
            extra_body={"cache_salt": "cachet-document-kv-cache"},
        ),
        response=FakeJSONResponse(),
        clock=FakeClock([1.0, 2.0]),
    )

    generation = engine.generate(benchmark_request(kv_transfer_params=kv_transfer_params))

    assert engine.payloads[0]["prompt"] == benchmark_request().cache_suffix_text
    assert engine.payloads[0]["request_id"] == "cachet-bio-1"
    assert engine.payloads[0]["kv_transfer_params"] == {
        **kv_transfer_params,
        DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM: "runtime",
    }
    assert engine.payloads[0]["cache_salt"] == "cachet-document-kv-cache"
    assert generation.metadata["kv_transfer_params_attached"] == "true"
    assert generation.metadata["request_id"] == "cachet-bio-1"
    assert generation.metadata["prefix_cache_salt_attached"] == "true"
    assert generation.metadata["prefix_cache_salt"] == "cachet-document-kv-cache"


def test_extra_body_factory_can_vary_prefix_cache_salt_per_request():
    engine = CapturingEngine(
        OpenAICompatibleEngineConfig(
            base_url="http://localhost:8000",
            stream=False,
            extra_body={"cache_salt": "static-salt", "top_p": 0.9},
        ),
        extra_body_factory=lambda request: {"cache_salt": f"dynamic-repeat-{request.repeat_index}"},
        response=FakeJSONResponse(),
        clock=FakeClock([1.0, 2.0, 3.0, 4.0]),
    )

    first = engine.generate(benchmark_request(repeat_index=1))
    second = engine.generate(benchmark_request(repeat_index=2))

    assert engine.payloads[0]["cache_salt"] == "dynamic-repeat-1"
    assert engine.payloads[1]["cache_salt"] == "dynamic-repeat-2"
    assert engine.payloads[0]["top_p"] == 0.9
    assert engine.payloads[1]["top_p"] == 0.9
    assert first.metadata["prefix_cache_salt"] == "dynamic-repeat-1"
    assert second.metadata["prefix_cache_salt"] == "dynamic-repeat-2"


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
    assert generation.metadata["logical_prompt_tokens"] == generation.metadata["runtime_prompt_tokens"]
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
    assert generation.metadata["logical_prompt_tokens"] == generation.metadata["runtime_prompt_tokens"]


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
    assert generation.metadata["logical_prompt_tokens"] == generation.metadata["runtime_prompt_tokens"]


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


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("base_url", "", "base_url must be non-empty"),
        ("endpoint", "", "endpoint must be non-empty"),
        ("api_key", 123, "api_key must be a string"),
        ("timeout_seconds", math.inf, "timeout_seconds must be a positive finite number"),
        ("max_tokens", True, "max_tokens must be positive"),
        ("temperature", math.nan, "temperature must be a non-negative finite number"),
        ("stream", 1, "stream must be a boolean"),
        ("include_usage", 0, "include_usage must be a boolean"),
        ("model_id", "", "model_id must be non-empty"),
        ("prompt_text_mode", "full", "prompt_text_mode"),
        ("prompt_token_accounting", "usage", "prompt_token_accounting"),
    ],
)
def test_openai_compatible_engine_config_rejects_invalid_public_fields(field_name, value, message):
    kwargs = {"base_url": "http://localhost:8000"}
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=message):
        OpenAICompatibleEngineConfig(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"extra_body": []}, "extra_body must be a mapping"),
        ({"extra_body": {"": 1}}, "extra_body keys"),
        ({"extra_body": {"temperature": math.nan}}, "extra_body.temperature"),
        ({"extra_body": {"bad": object()}}, "extra_body.bad"),
        ({"extra_body": {"nested": {"bad": object()}}}, "extra_body.nested.bad"),
        ({"extra_headers": []}, "extra_headers must be a mapping"),
        ({"extra_headers": {"": "value"}}, "extra_headers keys"),
        ({"extra_headers": {"X-Test": 1}}, "extra_headers.X-Test"),
    ],
)
def test_openai_compatible_engine_config_rejects_invalid_mappings(overrides, message):
    with pytest.raises(ValueError, match=message):
        OpenAICompatibleEngineConfig(base_url="http://localhost:8000", **overrides)


def test_openai_compatible_engine_config_normalizes_json_body_tuples():
    config = OpenAICompatibleEngineConfig(
        base_url="http://localhost:8000",
        extra_body={"guided_choice": ("yes", "no")},
    )

    assert config.extra_body == {"guided_choice": ["yes", "no"]}


def test_legacy_module_reexports_document_owned_engine():
    assert OpenAICompatibleCompletionEngine.__module__ == "document_kv_cache.openai_compatible"
    assert legacy_openai_module.OpenAICompatibleCompletionEngine is OpenAICompatibleCompletionEngine
    assert legacy_openai_module.OpenAICompatibleEngineConfig is OpenAICompatibleEngineConfig
    assert set(openai_module.__all__) < set(legacy_openai_module.__all__)
    assert "urlopen" in legacy_openai_module.__all__
