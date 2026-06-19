import json
from types import SimpleNamespace

import pytest

import document_kv_cache.live_server as public_live_server
import restaurant_kv_serving.live_server as legacy_live_server
from document_kv_cache.benchmark_runner import BenchmarkGeneration
from document_kv_cache.benchmarks import BASELINE_PREFILL_ARM, CACHE_REUSE_ARM
from document_kv_cache.live_server import (
    DEFAULT_LIVE_CHECK_ANSWER,
    LIVE_CHECK_SUITE_ID,
    LiveServerCheckConfig,
    build_live_server_check_request,
    main,
    run_openai_compatible_live_check,
)


class FakeEngine:
    def __init__(self, output_text: str = f"The code is {DEFAULT_LIVE_CHECK_ANSWER}.") -> None:
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


def test_build_live_server_check_request_defaults_to_baseline_full_prompt_contract():
    request = build_live_server_check_request(model_id="qwen3:4b-instruct")

    assert request.suite_id == LIVE_CHECK_SUITE_ID
    assert request.arm.arm_id == BASELINE_PREFILL_ARM
    assert request.logical_prompt_text == request.prompt_text
    assert DEFAULT_LIVE_CHECK_ANSWER in request.logical_prompt_text
    assert DEFAULT_LIVE_CHECK_ANSWER not in request.cache_suffix_text
    assert request.cache_prefix_text + request.cache_suffix_text == request.logical_prompt_text


def test_build_live_server_check_request_can_use_cache_arm_for_kv_aware_proxy():
    request = build_live_server_check_request(use_cache_arm=True)

    assert request.arm.arm_id == CACHE_REUSE_ARM
    assert request.prompt_text == request.cache_suffix_text
    assert request.logical_prompt_text != request.runtime_prompt_text
    assert DEFAULT_LIVE_CHECK_ANSWER not in request.runtime_prompt_text


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
        "hardware_target": "aws-g5",
        "dataset": "niah",
        "arm_id": BASELINE_PREFILL_ARM,
        "prompt_text_mode": "logical",
        "logical_prompt_chars": len(engine.requests[0].logical_prompt_text),
        "runtime_prompt_chars": len(engine.requests[0].runtime_prompt_text),
        "prompt_tokens": 123,
        "prompt_token_source": "unknown",
        "completion_tokens": 4,
        "ttft_seconds": 0.5,
        "time_to_completion_seconds": 1.25,
        "answer_found": True,
        "output_text": f"The code is {DEFAULT_LIVE_CHECK_ANSWER}.",
        "metadata": {"engine": "fake"},
    }


def test_live_check_reports_failed_quality_when_expected_answer_is_missing():
    result = run_openai_compatible_live_check(
        LiveServerCheckConfig(base_url="http://localhost:8000"),
        engine=FakeEngine(output_text="I do not know."),
    )

    assert result.ok is False
    assert result.answer_found is False


def test_runtime_prompt_mode_requires_cache_arm():
    with pytest.raises(ValueError, match="requires use_cache_arm"):
        LiveServerCheckConfig(
            base_url="http://localhost:8000",
            prompt_text_mode="runtime",
        )


def test_cli_reports_json_error_for_runtime_prompt_without_cache_arm(capsys):
    exit_code = main(["--base-url", "http://localhost:8000", "--runtime-prompt"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error_type"] == "ValueError"
    assert "requires use_cache_arm" in payload["error"]


def test_public_live_server_main_respects_document_namespace_monkeypatch(monkeypatch, capsys):
    original_legacy_run = legacy_live_server.run_openai_compatible_live_check

    def fake_run(config):
        assert config.base_url == "http://localhost:8000"
        return SimpleNamespace(ok=True, to_record=lambda: {"ok": True, "source": "public-hook"})

    monkeypatch.setattr(public_live_server, "run_openai_compatible_live_check", fake_run)

    exit_code = public_live_server.main(["--base-url", "http://localhost:8000"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "source": "public-hook"}
    assert legacy_live_server.run_openai_compatible_live_check is original_legacy_run
