"""Live OpenAI-compatible smoke checks for document KV-cache serving."""

from __future__ import annotations

import argparse
import math
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from document_kv_cache.benchmark_runner import (
    BenchmarkEngine,
    BenchmarkEngineRequest,
    BenchmarkGeneration,
)
from document_kv_cache.benchmarks import (
    DEFAULT_HARDWARE_TARGET,
    DEFAULT_V1_MODEL_ID,
    BenchmarkExample,
    BenchmarkPromptParts,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM,
    answer_found,
    baseline_prefill_arm,
    build_prompt_parts,
    document_kv_cache_arm,
)
from document_kv_cache.engine_adapters import (
    ServingBackend,
    read_engine_adapter_request_json,
    validate_engine_adapter_request_record,
)
from document_kv_cache.openai_compatible import (
    KVTransferParamsTransport,
    OpenAICompatibleCompletionEngine,
    OpenAICompatibleEngineConfig,
    OpenAICompatibleRequestMode,
    PromptTextMode,
    PromptTokenAccounting,
)
from document_kv_cache.workflow import SourceDocument

__all__ = [
    "LIVE_CHECK_SUITE_ID",
    "DEFAULT_LIVE_CHECK_ANSWER",
    "DEFAULT_LIVE_CHECK_PROMPT_FORMAT",
    "LiveCheckPromptFormat",
    "LiveServerCheckConfig",
    "LiveServerCheckResult",
    "build_live_server_check_request",
    "live_check_kv_transfer_params",
    "run_openai_compatible_live_check",
    "main",
]

LIVE_CHECK_SUITE_ID = "openai-compatible-live-check"
DEFAULT_LIVE_CHECK_ANSWER = "otkv7391"
LiveCheckPromptFormat = Literal["plain", "qwen3_chat"]
DEFAULT_LIVE_CHECK_PROMPT_FORMAT: LiveCheckPromptFormat = "plain"
_QWEN3_CHAT_SYSTEM_PROMPT = (
    "You are a precise benchmark extraction assistant. "
    "Return only the requested answer, with no explanation."
)


@dataclass(frozen=True, slots=True)
class LiveServerCheckConfig:
    base_url: str
    model_id: str = DEFAULT_V1_MODEL_ID
    hardware_target: str = DEFAULT_HARDWARE_TARGET
    use_cache_arm: bool = False
    prompt_text_mode: PromptTextMode = "logical"
    prompt_token_accounting: PromptTokenAccounting = "logical"
    kv_transfer_params_transport: KVTransferParamsTransport = "top_level"
    request_mode: OpenAICompatibleRequestMode = "completion"
    stream: bool = True
    max_tokens: int = 32
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    expected_answer: str = DEFAULT_LIVE_CHECK_ANSWER
    prompt_format: LiveCheckPromptFormat = DEFAULT_LIVE_CHECK_PROMPT_FORMAT
    api_key: str | None = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    kv_transfer_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must be non-empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a non-negative finite number")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.prompt_text_mode == "runtime" and not self.use_cache_arm:
            raise ValueError("prompt_text_mode='runtime' requires use_cache_arm=True")
        if self.kv_transfer_params_transport not in {"top_level", "custom_params"}:
            raise ValueError(
                "kv_transfer_params_transport must be 'top_level' or 'custom_params'"
            )
        if self.request_mode not in {"completion", "chat"}:
            raise ValueError("request_mode must be 'completion' or 'chat'")
        _validate_live_check_prompt_format(self.prompt_format)
        if not isinstance(self.kv_transfer_params, Mapping):
            raise ValueError("kv_transfer_params must be a mapping")
        object.__setattr__(self, "kv_transfer_params", dict(self.kv_transfer_params))
        if self.kv_transfer_params and not self.use_cache_arm:
            raise ValueError("kv_transfer_params require use_cache_arm=True")


@dataclass(frozen=True, slots=True)
class LiveServerCheckResult:
    request: BenchmarkEngineRequest
    generation: BenchmarkGeneration
    prompt_text_mode: PromptTextMode
    request_mode: OpenAICompatibleRequestMode
    prompt_format: LiveCheckPromptFormat
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
            "request_id": self.request.request_id,
            "prompt_text_mode": self.prompt_text_mode,
            "request_mode": self.request_mode,
            "prompt_format": self.prompt_format,
            "kv_transfer_params_present": bool(self.request.kv_transfer_params),
            "kv_transfer_param_keys": sorted(self.request.kv_transfer_params),
            "logical_prompt_chars": len(self.request.logical_prompt_text),
            "runtime_prompt_chars": len(self.request.runtime_prompt_text),
            "prompt_tokens": self.generation.prompt_tokens,
            "prompt_token_source": self.generation.metadata.get(
                "prompt_token_source", "unknown"
            ),
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
    prompt_format: LiveCheckPromptFormat = DEFAULT_LIVE_CHECK_PROMPT_FORMAT,
    kv_transfer_params: Mapping[str, Any] | None = None,
) -> BenchmarkEngineRequest:
    _validate_live_check_prompt_format(prompt_format)
    params = {} if kv_transfer_params is None else dict(kv_transfer_params)
    if params and not use_cache_arm:
        raise ValueError("kv_transfer_params require use_cache_arm=True")
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
        kv_transfer_params=params,
    )
    arm = document_kv_cache_arm() if use_cache_arm else baseline_prefill_arm()
    request_params = example.kv_transfer_params if arm.uses_cache else {}
    prompt_parts = _live_check_prompt_parts(example, prompt_format=prompt_format)
    return BenchmarkEngineRequest(
        suite_id=LIVE_CHECK_SUITE_ID,
        model_id=model_id,
        hardware_target=hardware_target,
        example=example,
        arm=arm,
        prompt_parts=prompt_parts,
        request_id=_request_id_from_kv_transfer_params(request_params),
        kv_transfer_params=request_params,
    )


def live_check_kv_transfer_params(
    *,
    handoff_json: str | None = None,
    handoff_record: Mapping[str, Any] | None = None,
    request_id: str | None = None,
    payload_uri: str | None = None,
    sglang_hicache_page_keys: Sequence[str] = (),
    expected_backend: ServingBackend | str | None = None,
) -> dict[str, Any]:
    """Build strict Cachet handoff params for live endpoint smoke checks."""

    page_keys = _string_tuple(
        sglang_hicache_page_keys, field_name="sglang_hicache_page_keys"
    )
    if handoff_json and handoff_record is not None:
        raise ValueError(
            "live check handoff params must use only one of handoff_json or handoff_record"
        )
    if handoff_json is None and handoff_record is None:
        if request_id is not None or payload_uri is not None or page_keys:
            raise ValueError(
                "request_id, payload_uri, and sglang_hicache_page_keys require handoff_json or handoff_record"
            )
        return {}
    if handoff_json is not None:
        record = read_engine_adapter_request_json(
            handoff_json,
            expected_backend=expected_backend,
            require_external_payload_uri=payload_uri is None,
        )
        params: dict[str, Any] = {
            DOCUMENT_KV_REQUEST_ID_PARAM: _resolve_request_id(request_id, record),
            DOCUMENT_KV_HANDOFF_JSON_PARAM: handoff_json,
        }
    else:
        if not isinstance(handoff_record, Mapping):
            raise ValueError("handoff_record must be a JSON object")
        record = dict(handoff_record)
        validate_engine_adapter_request_record(
            record,
            expected_backend=expected_backend,
            require_external_payload_uri=payload_uri is None,
        )
        params = {
            DOCUMENT_KV_REQUEST_ID_PARAM: _resolve_request_id(request_id, record),
            DOCUMENT_KV_HANDOFF_RECORD_PARAM: record,
        }
    resolved_payload_uri = _resolve_payload_uri(payload_uri, record)
    if resolved_payload_uri is not None:
        params[DOCUMENT_KV_PAYLOAD_URI_PARAM] = resolved_payload_uri
    if page_keys:
        params[DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM] = list(page_keys)
    build_live_server_check_request(use_cache_arm=True, kv_transfer_params=params)
    return params


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
        prompt_format=_live_check_request_prompt_format(
            config.prompt_format, request_mode=config.request_mode
        ),
        kv_transfer_params=config.kv_transfer_params,
    )
    active_engine = engine or OpenAICompatibleCompletionEngine(
        OpenAICompatibleEngineConfig(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stream=config.stream,
            prompt_text_mode=config.prompt_text_mode,
            prompt_token_accounting=config.prompt_token_accounting,
            kv_transfer_params_transport=config.kv_transfer_params_transport,
            request_mode=config.request_mode,
            model_id=config.model_id,
            extra_body=config.extra_body,
        ),
        chat_messages_factory=_live_check_chat_messages_factory(config),
    )
    generation = active_engine.generate(request)
    return LiveServerCheckResult(
        request=request,
        generation=generation,
        prompt_text_mode=config.prompt_text_mode,
        request_mode=config.request_mode,
        prompt_format=config.prompt_format,
        answer_found=answer_found(generation.output_text, config.expected_answer),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a live OpenAI-compatible vLLM/SGLang smoke check."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Server base URL, for example http://localhost:8000",
    )
    parser.add_argument("--model-id", default=DEFAULT_V1_MODEL_ID)
    parser.add_argument("--hardware-target", default=DEFAULT_HARDWARE_TARGET)
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--prompt-format",
        choices=("plain", "qwen3_chat"),
        default=DEFAULT_LIVE_CHECK_PROMPT_FORMAT,
    )
    parser.add_argument(
        "--request-mode", choices=("completion", "chat"), default="completion"
    )
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument(
        "--cache-arm",
        action="store_true",
        help="Build the request with the document KV-cache arm id.",
    )
    parser.add_argument(
        "--runtime-prompt",
        action="store_true",
        help="Send only the runtime suffix; requires --cache-arm and a KV-aware proxy that binds cached prefixes.",
    )
    parser.add_argument(
        "--server-usage",
        action="store_true",
        help=(
            "Record server usage.prompt_tokens in metadata when present; "
            "reported prompt_tokens still follow the logical/runtime prompt context."
        ),
    )
    parser.add_argument(
        "--kv-transfer-params-transport",
        choices=("top-level", "custom-params"),
        default="top-level",
        help=(
            "Where to attach Cachet handoff params in the OpenAI-compatible request. "
            "Use top-level for vLLM and custom-params for SGLang."
        ),
    )
    parser.add_argument(
        "--extra-body-json",
        default="{}",
        help="Additional JSON fields merged into the request body.",
    )
    parser.add_argument(
        "--handoff-json",
        help="Validated Cachet engine-adapter handoff JSON path to send through kv_transfer_params.",
    )
    parser.add_argument(
        "--handoff-record-json",
        help="Inline Cachet engine-adapter handoff JSON object to send through kv_transfer_params.",
    )
    parser.add_argument(
        "--payload-uri",
        help="Optional adapter-readable payload URI override for the live handoff.",
    )
    parser.add_argument(
        "--request-id",
        help="Optional request id override; must match the handoff record request id when provided.",
    )
    parser.add_argument(
        "--sglang-hicache-page-keys-json",
        help=(
            "Optional JSON array of expected SGLang HiCache page keys to attach to kv_transfer_params."
        ),
    )
    parser.add_argument(
        "--expected-backend",
        choices=[ServingBackend.VLLM.value, ServingBackend.SGLANG.value],
        help="Validate the live handoff against a specific serving backend before sending it.",
    )
    args = parser.parse_args(argv)

    try:
        extra_body = json.loads(args.extra_body_json)
        if not isinstance(extra_body, Mapping):
            raise ValueError("--extra-body-json must decode to a JSON object")
        handoff_record = _json_object_option(
            args.handoff_record_json, "--handoff-record-json"
        )
        sglang_hicache_page_keys = _json_string_array_option(
            args.sglang_hicache_page_keys_json,
            "--sglang-hicache-page-keys-json",
        )
        kv_transfer_params = live_check_kv_transfer_params(
            handoff_json=args.handoff_json,
            handoff_record=handoff_record,
            request_id=args.request_id,
            payload_uri=args.payload_uri,
            sglang_hicache_page_keys=sglang_hicache_page_keys,
            expected_backend=args.expected_backend,
        )
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
                prompt_format=args.prompt_format,
                request_mode=args.request_mode,
                prompt_text_mode="runtime" if args.runtime_prompt else "logical",
                prompt_token_accounting=(
                    "server_usage" if args.server_usage else "logical"
                ),
                kv_transfer_params_transport=args.kv_transfer_params_transport.replace(
                    "-", "_"
                ),
                extra_body=extra_body,
                kv_transfer_params=kv_transfer_params,
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc), "error_type": type(exc).__name__},
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    print(json.dumps(result.to_record(), indent=2, sort_keys=True))
    return 0 if result.ok else 2


def _request_id_from_kv_transfer_params(params: Mapping[str, Any]) -> str | None:
    if not params:
        return None
    request_id = params.get(DOCUMENT_KV_REQUEST_ID_PARAM)
    if isinstance(request_id, str) and request_id:
        return request_id
    return None


def _live_check_request_prompt_format(
    prompt_format: LiveCheckPromptFormat,
    *,
    request_mode: OpenAICompatibleRequestMode,
) -> LiveCheckPromptFormat:
    if request_mode == "chat" and prompt_format == "qwen3_chat":
        return "plain"
    return prompt_format


def _live_check_chat_messages_factory(
    config: LiveServerCheckConfig,
) -> Callable[[BenchmarkEngineRequest], Sequence[Mapping[str, Any]]] | None:
    if config.request_mode != "chat" or config.prompt_format != "qwen3_chat":
        return None

    def factory(request: BenchmarkEngineRequest) -> Sequence[Mapping[str, Any]]:
        return _live_check_chat_messages(
            request, prompt_text_mode=config.prompt_text_mode
        )

    return factory


def _live_check_chat_messages(
    request: BenchmarkEngineRequest,
    *,
    prompt_text_mode: PromptTextMode = "logical",
) -> tuple[dict[str, str], ...]:
    content = (
        request.prompt_text
        if prompt_text_mode == "runtime"
        else request.logical_prompt_text
    )
    return (
        {"role": "system", "content": _QWEN3_CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    )


def _live_check_prompt_parts(
    example: BenchmarkExample,
    *,
    prompt_format: LiveCheckPromptFormat,
) -> BenchmarkPromptParts:
    parts = build_prompt_parts(example)
    if prompt_format == "plain":
        return parts
    if prompt_format == "qwen3_chat":
        return BenchmarkPromptParts(
            system_prompt=(
                "<|im_start|>system\n"
                "You are a precise benchmark extraction assistant. "
                "Return only the requested answer, with no explanation.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{parts.system_prompt}"
            ),
            document_context=parts.document_context,
            user_prompt=f"{parts.user_prompt}\n<|im_end|>\n<|im_start|>assistant\n",
        )
    raise ValueError(f"unsupported live check prompt_format {prompt_format!r}")


def _validate_live_check_prompt_format(value: object) -> None:
    if value not in {"plain", "qwen3_chat"}:
        raise ValueError("prompt_format must be 'plain' or 'qwen3_chat'")


def _resolve_request_id(request_id: str | None, record: Mapping[str, Any]) -> str:
    record_request_id = record.get("request_id")
    if not isinstance(record_request_id, str) or not record_request_id:
        raise ValueError("handoff record request_id must be a non-empty string")
    if request_id is not None and request_id != record_request_id:
        raise ValueError("request_id must match the handoff record request_id")
    return record_request_id


def _resolve_payload_uri(
    payload_uri: str | None, record: Mapping[str, Any]
) -> str | None:
    if payload_uri is not None:
        return payload_uri
    payload_source = record.get("payload_source")
    if not isinstance(payload_source, Mapping):
        return None
    uri = payload_source.get("uri")
    if uri is None:
        return None
    if not isinstance(uri, str) or not uri:
        raise ValueError(
            "handoff record payload_source.uri must be a non-empty string when present"
        )
    return uri


def _json_object_option(
    value: str | None, option_name: str
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    decoded = json.loads(value)
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{option_name} must decode to a JSON object")
    return decoded


def _json_string_array_option(value: str | None, option_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    decoded = json.loads(value)
    return _string_tuple(decoded, field_name=option_name)


def _string_tuple(values: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    if not isinstance(values, Sequence):
        raise ValueError(f"{field_name} must be a sequence of strings")
    items = tuple(values)
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field_name}[{index}] must be a non-empty string")
    return items


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
