import json

import pytest

import document_kv_cache.live_server as public_live_server
import document_kv_cache.openai_compatible as openai_compatible_module
from document_kv_cache.benchmark_runner import BenchmarkGeneration
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    CACHE_REUSE_ARM,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM,
)
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import (
    build_engine_adapter_request,
    engine_adapter_request_to_record,
    sglang_adapter_spec,
    vllm_adapter_spec,
)
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment
from document_kv_cache.live_server import (
    DEFAULT_LIVE_CHECK_ANSWER,
    LIVE_CHECK_SUITE_ID,
    LiveServerCheckConfig,
    build_live_server_check_request,
    live_check_kv_transfer_params,
    main,
    run_openai_compatible_live_check,
)
from document_kv_cache.quality_v2 import quality_v2_prompt_parts

QWEN3_ASSISTANT_OPEN = "<|im_end|>\n<|im_start|>assistant\n"


def render_qwen3_chat_template(messages) -> str:
    rendered = "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    )
    return rendered + "<|im_start|>assistant\n"


class FakeEngine:
    def __init__(
        self,
        output_text: str = f"The code is <final_answer>{DEFAULT_LIVE_CHECK_ANSWER}</final_answer>.",
    ) -> None:
        self.output_text = output_text
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return BenchmarkGeneration(
            output_text=self.output_text,
            prompt_tokens=123,
            completion_tokens=len(self.output_text.split()),
            ttft_seconds=0.5,
            time_to_completion_seconds=1.25,
            metadata={"engine": "fake"},
        )


def handoff_record(
    *, request_id: str, payload_uri: str, backend: str = "sglang"
) -> dict[str, object]:
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
    spec = sglang_adapter_spec() if backend == "sglang" else vllm_adapter_spec()
    adapter_request = build_engine_adapter_request(ready, spec=spec)
    return engine_adapter_request_to_record(adapter_request, payload_uri=payload_uri)


def write_handoff_json(
    path, *, request_id: str, payload_uri: str, backend: str = "sglang"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            handoff_record(
                request_id=request_id, payload_uri=payload_uri, backend=backend
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_build_live_server_check_request_defaults_to_baseline_full_prompt_contract():
    request = build_live_server_check_request(model_id="qwen3:4b-instruct")

    assert request.suite_id == LIVE_CHECK_SUITE_ID
    assert request.arm.arm_id == BASELINE_PREFILL_ARM
    assert request.logical_prompt_text == request.prompt_text
    assert DEFAULT_LIVE_CHECK_ANSWER in request.logical_prompt_text
    assert DEFAULT_LIVE_CHECK_ANSWER not in request.cache_suffix_text
    assert (
        request.cache_prefix_text + request.cache_suffix_text
        == request.logical_prompt_text
    )


def test_build_live_server_check_request_sends_the_quality_v2_chat_turn():
    request = build_live_server_check_request(use_cache_arm=True)

    assert request.prompt_parts == quality_v2_prompt_parts(request.example)
    assert request.logical_prompt_text.startswith("<|im_start|>user\n")
    assert request.logical_prompt_text.endswith(QWEN3_ASSISTANT_OPEN)
    assert "<|im_start|>system" not in request.logical_prompt_text
    assert request.cache_prefix_text.startswith("<|im_start|>user\n")
    assert "<|im_end|>\n<|im_start|>assistant" in request.cache_suffix_text
    assert DEFAULT_LIVE_CHECK_ANSWER in request.cache_prefix_text
    assert DEFAULT_LIVE_CHECK_ANSWER not in request.cache_suffix_text
    assert (
        request.cache_prefix_text + request.cache_suffix_text
        == request.logical_prompt_text
    )


def test_build_live_server_check_request_can_use_cache_arm_for_kv_aware_proxy():
    request = build_live_server_check_request(use_cache_arm=True)

    assert request.arm.arm_id == CACHE_REUSE_ARM
    assert request.prompt_text == request.cache_suffix_text
    assert request.logical_prompt_text != request.runtime_prompt_text
    assert DEFAULT_LIVE_CHECK_ANSWER not in request.runtime_prompt_text


def test_live_check_kv_transfer_params_from_sglang_handoff_json(tmp_path):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(
        handoff_path,
        request_id="cachet-live-sglang-1",
        payload_uri=payload_uri,
        backend="sglang",
    )

    params = live_check_kv_transfer_params(
        handoff_json=str(handoff_path),
        expected_backend="sglang",
    )
    request = build_live_server_check_request(
        use_cache_arm=True, kv_transfer_params=params
    )

    assert params == {
        DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-live-sglang-1",
        DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
        DOCUMENT_KV_PAYLOAD_URI_PARAM: payload_uri,
    }
    assert request.arm.arm_id == CACHE_REUSE_ARM
    assert request.request_id == "cachet-live-sglang-1"
    assert request.kv_transfer_params == params


def test_live_check_kv_transfer_params_accepts_inline_handoff_record(tmp_path):
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    record = handoff_record(
        request_id="cachet-live-inline-1",
        payload_uri=payload_uri,
        backend="sglang",
    )

    params = live_check_kv_transfer_params(
        handoff_record=record,
        payload_uri=f"disk:{tmp_path / 'payloads' / 'override.kv'}",
        expected_backend="sglang",
    )

    assert params[DOCUMENT_KV_REQUEST_ID_PARAM] == "cachet-live-inline-1"
    assert params[DOCUMENT_KV_HANDOFF_RECORD_PARAM] == record
    assert params[DOCUMENT_KV_PAYLOAD_URI_PARAM].endswith("/override.kv")


def test_live_check_kv_transfer_params_can_attach_sglang_page_keys(tmp_path):
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    record = handoff_record(
        request_id="cachet-live-inline-1",
        payload_uri=payload_uri,
        backend="sglang",
    )

    params = live_check_kv_transfer_params(
        handoff_record=record,
        sglang_hicache_page_keys=("page-a", "page-b"),
        expected_backend="sglang",
    )

    assert params[DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM] == ["page-a", "page-b"]


def test_live_check_kv_transfer_params_rejects_malformed_sglang_page_keys(tmp_path):
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    record = handoff_record(
        request_id="cachet-live-inline-1",
        payload_uri=payload_uri,
        backend="sglang",
    )

    with pytest.raises(ValueError, match="sglang_hicache_page_keys must be a sequence"):
        live_check_kv_transfer_params(
            handoff_record=record,
            sglang_hicache_page_keys="page-a",
            expected_backend="sglang",
        )


def test_live_check_kv_transfer_params_rejects_page_keys_without_handoff():
    with pytest.raises(
        ValueError,
        match="sglang_hicache_page_keys require handoff_json or handoff_record",
    ):
        live_check_kv_transfer_params(sglang_hicache_page_keys=("page-a",))


def test_live_check_kv_transfer_params_rejects_backend_mismatch(tmp_path):
    handoff_path = tmp_path / "handoffs" / "vllm-live.handoff.json"
    write_handoff_json(
        handoff_path,
        request_id="cachet-live-vllm-1",
        payload_uri=f"disk:{tmp_path / 'payloads' / 'vllm-live.kv'}",
        backend="vllm",
    )

    with pytest.raises(ValueError, match="does not match expected_backend"):
        live_check_kv_transfer_params(
            handoff_json=str(handoff_path), expected_backend="sglang"
        )


def test_build_live_server_check_request_requires_cache_arm_for_handoff_params(
    tmp_path,
):
    params = {
        DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-live-1",
        DOCUMENT_KV_HANDOFF_RECORD_PARAM: handoff_record(
            request_id="cachet-live-1",
            payload_uri=f"disk:{tmp_path / 'payloads' / 'live.kv'}",
        ),
    }

    with pytest.raises(ValueError, match="kv_transfer_params require use_cache_arm"):
        build_live_server_check_request(kv_transfer_params=params)


def test_run_openai_compatible_live_check_returns_json_ready_record():
    engine = FakeEngine()

    result = run_openai_compatible_live_check(
        LiveServerCheckConfig(base_url="http://localhost:8000"),
        engine=engine,
    )

    assert result.ok is True
    assert result.answer_found is True
    assert engine.requests[0].arm.arm_id == BASELINE_PREFILL_ARM
    assert result.to_record() == {
        "ok": True,
        "suite_id": LIVE_CHECK_SUITE_ID,
        "model_id": "qwen3:4b-instruct",
        "hardware_target": "aws-g6-l4",
        "dataset": "niah",
        "arm_id": BASELINE_PREFILL_ARM,
        "request_id": None,
        "prompt_text_mode": "logical",
        "request_mode": "completion",
        "kv_transfer_params_present": False,
        "kv_transfer_param_keys": [],
        "logical_prompt_chars": len(engine.requests[0].logical_prompt_text),
        "runtime_prompt_chars": len(engine.requests[0].runtime_prompt_text),
        "prompt_tokens": 123,
        "prompt_token_source": "unknown",
        "completion_tokens": 4,
        "ttft_seconds": 0.5,
        "time_to_completion_seconds": 1.25,
        "answer_found": True,
        "output_text": f"The code is <final_answer>{DEFAULT_LIVE_CHECK_ANSWER}</final_answer>.",
        "metadata": {"engine": "fake"},
    }


def test_run_openai_compatible_live_check_attaches_kv_transfer_params(tmp_path):
    engine = FakeEngine()
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    params = live_check_kv_transfer_params(
        handoff_record=handoff_record(
            request_id="cachet-live-sglang-2",
            payload_uri=payload_uri,
            backend="sglang",
        ),
        expected_backend="sglang",
    )

    result = run_openai_compatible_live_check(
        LiveServerCheckConfig(
            base_url="http://localhost:8000",
            use_cache_arm=True,
            kv_transfer_params=params,
        ),
        engine=engine,
    )

    assert engine.requests[0].arm.arm_id == CACHE_REUSE_ARM
    assert engine.requests[0].request_id == "cachet-live-sglang-2"
    assert engine.requests[0].kv_transfer_params == params
    assert result.to_record()["request_id"] == "cachet-live-sglang-2"
    assert result.to_record()["kv_transfer_params_present"] is True
    assert result.to_record()["kv_transfer_param_keys"] == sorted(params)


def test_chat_live_check_serves_the_quality_v2_prompt_and_records_chat_mode():
    engine = FakeEngine()

    result = run_openai_compatible_live_check(
        LiveServerCheckConfig(base_url="http://localhost:8000", request_mode="chat"),
        engine=engine,
    )

    request = engine.requests[0]
    assert request.prompt_parts == quality_v2_prompt_parts(request.example)
    assert result.to_record()["request_mode"] == "chat"
    assert "prompt_format" not in result.to_record()


def test_chat_live_check_messages_render_the_quality_v2_prompt():
    config = LiveServerCheckConfig(base_url="http://localhost:8000", request_mode="chat")
    request = build_live_server_check_request()
    factory = public_live_server._live_check_chat_messages_factory(config)

    assert factory is not None
    messages = factory(request)
    assert [message["role"] for message in messages] == ["user"]
    assert render_qwen3_chat_template(messages) == request.logical_prompt_text


class _JSONResponse:
    def __init__(self, data):
        self.data = data

    def read(self):
        return json.dumps(self.data).encode("utf-8")

    def close(self):
        pass


def _post_live_check(monkeypatch, **config_kwargs):
    bodies = []

    def fake_urlopen(request, *, timeout):
        bodies.append(json.loads(request.data))
        return _JSONResponse(
            {"choices": [{"text": f"<final_answer>{DEFAULT_LIVE_CHECK_ANSWER}</final_answer>"}]}
        )

    monkeypatch.setattr(openai_compatible_module, "_urlopen", fake_urlopen)
    result = run_openai_compatible_live_check(
        LiveServerCheckConfig(base_url="http://localhost:8000", stream=False, **config_kwargs)
    )
    (body,) = bodies
    return result, body


def test_chat_live_check_sends_one_user_message_without_chat_markers(monkeypatch):
    result, body = _post_live_check(monkeypatch, request_mode="chat")

    assert "prompt" not in body
    (message,) = body["messages"]
    assert message["role"] == "user"
    assert "<|im_start|>" not in message["content"]
    assert "<|im_end|>" not in message["content"]
    assert render_qwen3_chat_template(body["messages"]) == result.request.logical_prompt_text


def test_completion_live_check_sends_the_quality_v2_prompt_text(monkeypatch):
    result, body = _post_live_check(monkeypatch)

    assert "messages" not in body
    assert body["prompt"] == quality_v2_prompt_parts(result.request.example).prefill_prompt
    assert body["prompt"].startswith("<|im_start|>user\n")
    assert body["prompt"].endswith("<|im_end|>\n<|im_start|>assistant\n")


def test_cli_rejects_the_removed_prompt_format_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--base-url", "http://localhost:8000", "--prompt-format", "plain"])

    assert exc.value.code == 2
    assert "unrecognized arguments: --prompt-format plain" in capsys.readouterr().err


def test_live_server_check_config_rejects_invalid_kv_transfer_transport():
    with pytest.raises(ValueError, match="kv_transfer_params_transport"):
        LiveServerCheckConfig(
            base_url="http://localhost:8000",
            kv_transfer_params_transport="query",  # type: ignore[arg-type]
        )


def test_live_server_check_config_rejects_invalid_temperature():
    with pytest.raises(ValueError, match="temperature"):
        LiveServerCheckConfig(
            base_url="http://localhost:8000",
            temperature=float("nan"),
        )


def test_live_server_check_config_rejects_invalid_request_mode():
    with pytest.raises(ValueError, match="request_mode"):
        LiveServerCheckConfig(
            base_url="http://localhost:8000",
            request_mode="responses",  # type: ignore[arg-type]
        )


def test_live_check_reports_failed_quality_when_expected_answer_is_missing():
    result = run_openai_compatible_live_check(
        LiveServerCheckConfig(base_url="http://localhost:8000"),
        engine=FakeEngine(output_text="I do not know."),
    )

    assert result.ok is False
    assert result.answer_found is False


@pytest.mark.parametrize(
    ("output_text", "expected_found"),
    [
        (f"<final_answer>{DEFAULT_LIVE_CHECK_ANSWER}</final_answer>", True),
        (
            "<final_answer>...</final_answer> Checking the document. "
            f"<final_answer>{DEFAULT_LIVE_CHECK_ANSWER}</final_answer>",
            True,
        ),
        (f"The hidden live-check verification code is {DEFAULT_LIVE_CHECK_ANSWER}.", False),
        (
            f"The hidden live-check verification code is {DEFAULT_LIVE_CHECK_ANSWER}. "
            "<final_answer>unknown</final_answer>",
            False,
        ),
        (f"<final_answer>{DEFAULT_LIVE_CHECK_ANSWER}", False),
    ],
    ids=(
        "final-answer-block",
        "after-echoed-placeholder",
        "echoed-needle-without-block",
        "echoed-needle-outside-block",
        "unclosed-block",
    ),
)
def test_live_check_judges_the_answer_the_quality_v2_parser_extracts(
    output_text, expected_found
):
    result = run_openai_compatible_live_check(
        LiveServerCheckConfig(base_url="http://localhost:8000"),
        engine=FakeEngine(output_text=output_text),
    )

    assert result.answer_found is expected_found
    assert result.ok is expected_found
    assert result.to_record()["answer_found"] is expected_found


def test_runtime_prompt_mode_requires_cache_arm():
    with pytest.raises(ValueError, match="requires use_cache_arm"):
        LiveServerCheckConfig(
            base_url="http://localhost:8000",
            prompt_text_mode="runtime",
        )


def test_kv_transfer_params_require_cache_arm():
    with pytest.raises(ValueError, match="kv_transfer_params require use_cache_arm"):
        LiveServerCheckConfig(
            base_url="http://localhost:8000",
            kv_transfer_params={
                DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-live-1",
                DOCUMENT_KV_HANDOFF_JSON_PARAM: "/tmp/cachet-live.handoff.json",
            },
        )


def test_cli_reports_json_error_for_runtime_prompt_without_cache_arm(capsys):
    exit_code = main(["--base-url", "http://localhost:8000", "--runtime-prompt"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error_type"] == "ValueError"
    assert "requires use_cache_arm" in payload["error"]


def test_cli_reports_json_error_for_handoff_without_cache_arm(tmp_path, capsys):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    write_handoff_json(
        handoff_path,
        request_id="cachet-live-sglang-3",
        payload_uri=f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}",
        backend="sglang",
    )

    exit_code = main(
        [
            "--base-url",
            "http://localhost:8000",
            "--handoff-json",
            str(handoff_path),
            "--expected-backend",
            "sglang",
            "--kv-transfer-params-transport",
            "custom-params",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error_type"] == "ValueError"
    assert "kv_transfer_params require use_cache_arm" in payload["error"]


def test_cli_sends_validated_handoff_params_for_cache_arm(
    tmp_path, monkeypatch, capsys
):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(
        handoff_path,
        request_id="cachet-live-sglang-4",
        payload_uri=payload_uri,
        backend="sglang",
    )
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return run_openai_compatible_live_check(config, engine=FakeEngine())

    monkeypatch.setattr(
        public_live_server, "run_openai_compatible_live_check", fake_run
    )

    exit_code = main(
        [
            "--base-url",
            "http://localhost:8000",
            "--cache-arm",
            "--handoff-json",
            str(handoff_path),
            "--expected-backend",
            "sglang",
            "--kv-transfer-params-transport",
            "custom-params",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    config = captured["config"]
    assert exit_code == 0
    assert config.use_cache_arm is True
    assert config.kv_transfer_params_transport == "custom_params"
    assert config.kv_transfer_params == {
        DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-live-sglang-4",
        DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
        DOCUMENT_KV_PAYLOAD_URI_PARAM: payload_uri,
    }
    assert payload["ok"] is True
    assert payload["request_id"] == "cachet-live-sglang-4"
    assert payload["kv_transfer_params_present"] is True
