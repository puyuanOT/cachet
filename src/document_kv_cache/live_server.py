"""Live OpenAI-compatible smoke checks for document KV-cache serving."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from document_kv_cache.benchmark_runner import BenchmarkEngine, BenchmarkEngineRequest, BenchmarkGeneration
from document_kv_cache.benchmarks import (
    DEFAULT_HARDWARE_TARGET,
    DEFAULT_V1_MODEL_ID,
    BenchmarkExample,
    answer_found,
    baseline_prefill_arm,
    build_prompt_parts,
    document_kv_cache_arm,
)
from document_kv_cache.openai_compatible import (
    OpenAICompatibleCompletionEngine,
    OpenAICompatibleEngineConfig,
    PromptTextMode,
    PromptTokenAccounting,
)
from document_kv_cache.workflow import SourceDocument

__all__ = [
    "LIVE_CHECK_SUITE_ID",
    "DEFAULT_LIVE_CHECK_ANSWER",
    "LiveServerCheckConfig",
    "LiveServerCheckResult",
    "build_live_server_check_request",
    "run_openai_compatible_live_check",
    "main",
]

LIVE_CHECK_SUITE_ID = "openai-compatible-live-check"
DEFAULT_LIVE_CHECK_ANSWER = "otkv7391"


@dataclass(frozen=True, slots=True)
class LiveServerCheckConfig:
    base_url: str
    model_id: str = DEFAULT_V1_MODEL_ID
    hardware_target: str = DEFAULT_HARDWARE_TARGET
    use_cache_arm: bool = False
    prompt_text_mode: PromptTextMode = "logical"
    prompt_token_accounting: PromptTokenAccounting = "logical"
    stream: bool = True
    max_tokens: int = 32
    timeout_seconds: float = 120.0
    expected_answer: str = DEFAULT_LIVE_CHECK_ANSWER
    api_key: str | None = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must be non-empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.prompt_text_mode == "runtime" and not self.use_cache_arm:
            raise ValueError("prompt_text_mode='runtime' requires use_cache_arm=True")


@dataclass(frozen=True, slots=True)
class LiveServerCheckResult:
    request: BenchmarkEngineRequest
    generation: BenchmarkGeneration
    prompt_text_mode: PromptTextMode
    answer_found: bool

    @property
    def ok(self) -> bool:
        return bool(self.generation.output_text.strip()) and self.answer_found

    def to_record(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "suite_id": self.request.suite_id,
            "model_id": self.request.model_id,
            "hardware_target": self.request.hardware_target,
            "dataset": self.request.example.dataset,
            "arm_id": self.request.arm.arm_id,
            "prompt_text_mode": self.prompt_text_mode,
            "logical_prompt_chars": len(self.request.logical_prompt_text),
            "runtime_prompt_chars": len(self.request.runtime_prompt_text),
            "prompt_tokens": self.generation.prompt_tokens,
            "prompt_token_source": self.generation.metadata.get("prompt_token_source", "unknown"),
            "completion_tokens": self.generation.completion_tokens,
            "ttft_seconds": self.generation.ttft_seconds,
            "time_to_completion_seconds": self.generation.time_to_completion_seconds,
            "answer_found": self.answer_found,
            "output_text": self.generation.output_text,
            "metadata": dict(self.generation.metadata),
        }


def build_live_server_check_request(
    *,
    model_id: str = DEFAULT_V1_MODEL_ID,
    hardware_target: str = DEFAULT_HARDWARE_TARGET,
    use_cache_arm: bool = False,
    expected_answer: str = DEFAULT_LIVE_CHECK_ANSWER,
) -> BenchmarkEngineRequest:
    example = BenchmarkExample(
        example_id="live-niah-synthetic-nonce",
        dataset="niah",
        documents=(
            SourceDocument.from_texts(
                document_id="live-synthetic-needle",
                static_text=(
                    "This synthetic document exists only to verify live KV-cache serving. "
                    f"The hidden live-check verification code is {expected_answer}."
                ),
                chunks={
                    "distractor": "The surrounding haystack contains ordinary filler text with no useful answer."
                },
            ),
        ),
        query="What is the hidden live-check verification code?",
        expected_answer=expected_answer,
    )
    return BenchmarkEngineRequest(
        suite_id=LIVE_CHECK_SUITE_ID,
        model_id=model_id,
        hardware_target=hardware_target,
        example=example,
        arm=document_kv_cache_arm() if use_cache_arm else baseline_prefill_arm(),
        prompt_parts=build_prompt_parts(example),
    )


def run_openai_compatible_live_check(
    config: LiveServerCheckConfig,
    *,
    engine: BenchmarkEngine | None = None,
) -> LiveServerCheckResult:
    request = build_live_server_check_request(
        model_id=config.model_id,
        hardware_target=config.hardware_target,
        use_cache_arm=config.use_cache_arm,
        expected_answer=config.expected_answer,
    )
    active_engine = engine or OpenAICompatibleCompletionEngine(
        OpenAICompatibleEngineConfig(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            max_tokens=config.max_tokens,
            stream=config.stream,
            prompt_text_mode=config.prompt_text_mode,
            prompt_token_accounting=config.prompt_token_accounting,
            model_id=config.model_id,
            extra_body=config.extra_body,
        )
    )
    generation = active_engine.generate(request)
    return LiveServerCheckResult(
        request=request,
        generation=generation,
        prompt_text_mode=config.prompt_text_mode,
        answer_found=answer_found(generation.output_text, config.expected_answer),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a live OpenAI-compatible vLLM/SGLang smoke check.")
    parser.add_argument("--base-url", required=True, help="Server base URL, for example http://localhost:8000")
    parser.add_argument("--model-id", default=DEFAULT_V1_MODEL_ID)
    parser.add_argument("--hardware-target", default=DEFAULT_HARDWARE_TARGET)
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--cache-arm", action="store_true", help="Build the request with the document KV-cache arm id.")
    parser.add_argument(
        "--runtime-prompt",
        action="store_true",
        help="Send only the runtime suffix; requires --cache-arm and a KV-aware proxy that binds cached prefixes.",
    )
    parser.add_argument(
        "--server-usage",
        action="store_true",
        help="Prefer server usage.prompt_tokens; prompt_token_source reports whether usage was present.",
    )
    parser.add_argument("--extra-body-json", default="{}", help="Additional JSON fields merged into the request body.")
    args = parser.parse_args(argv)

    try:
        extra_body = json.loads(args.extra_body_json)
        if not isinstance(extra_body, Mapping):
            raise ValueError("--extra-body-json must decode to a JSON object")
        result = run_openai_compatible_live_check(
            LiveServerCheckConfig(
                base_url=args.base_url,
                model_id=args.model_id,
                hardware_target=args.hardware_target,
                api_key=args.api_key,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
                stream=not args.no_stream,
                use_cache_arm=args.cache_arm,
                prompt_text_mode="runtime" if args.runtime_prompt else "logical",
                prompt_token_accounting="server_usage" if args.server_usage else "logical",
                extra_body=extra_body,
            )
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, indent=2, sort_keys=True))
        return 1

    print(json.dumps(result.to_record(), indent=2, sort_keys=True))
    return 0 if result.ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
