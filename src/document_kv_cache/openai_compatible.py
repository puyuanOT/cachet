"""OpenAI-compatible completion engine for document KV-cache benchmarks."""

from __future__ import annotations

import json
import time
import urllib.error as _urlerror
import urllib.parse as _urlparse
import urllib.request as _urlrequest
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from document_kv_cache.benchmark_runner import (
    BenchmarkEngineRequest,
    BenchmarkGeneration,
)
from document_kv_cache.benchmarks import DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM

__all__ = [
    "TokenCounter",
    "PromptTextMode",
    "PromptTokenAccounting",
    "KVTransferParamsTransport",
    "OpenAICompatibleRequestMode",
    "WhitespaceTokenCounter",
    "OpenAICompatibleEngineConfig",
    "OpenAICompatibleCompletionEngine",
]

_urlopen = _urlrequest.urlopen


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


PromptTextMode = Literal["logical", "runtime"]
PromptTokenAccounting = Literal["logical", "server_usage"]
KVTransferParamsTransport = Literal["top_level", "custom_params"]
OpenAICompatibleRequestMode = Literal["completion", "chat"]

_COMPLETIONS_ENDPOINT = "/v1/completions"
_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"


@dataclass(frozen=True, slots=True)
class WhitespaceTokenCounter:
    def count(self, text: str) -> int:
        return len(text.split())


@dataclass(frozen=True, slots=True)
class OpenAICompatibleEngineConfig:
    base_url: str
    endpoint: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 120.0
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = True
    include_usage: bool = True
    model_id: str | None = None
    prompt_text_mode: PromptTextMode = "logical"
    prompt_token_accounting: PromptTokenAccounting = "logical"
    kv_transfer_params_transport: KVTransferParamsTransport = "top_level"
    request_mode: OpenAICompatibleRequestMode = "completion"
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.base_url, "base_url")
        if self.request_mode not in {"completion", "chat"}:
            raise ValueError("request_mode must be 'completion' or 'chat'")
        endpoint = (
            _default_endpoint(self.request_mode)
            if self.endpoint is None
            else self.endpoint
        )
        _validate_non_empty_string(endpoint, "endpoint")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ValueError("api_key must be a string when provided")
        _validate_positive_finite_number(self.timeout_seconds, "timeout_seconds")
        _validate_positive_int(self.max_tokens, "max_tokens")
        _validate_non_negative_finite_number(self.temperature, "temperature")
        if type(self.stream) is not bool:
            raise ValueError("stream must be a boolean")
        if type(self.include_usage) is not bool:
            raise ValueError("include_usage must be a boolean")
        if self.model_id is not None:
            _validate_non_empty_string(self.model_id, "model_id")
        if self.prompt_text_mode not in {"logical", "runtime"}:
            raise ValueError("prompt_text_mode must be 'logical' or 'runtime'")
        if self.prompt_token_accounting not in {"logical", "server_usage"}:
            raise ValueError(
                "prompt_token_accounting must be 'logical' or 'server_usage'"
            )
        if self.kv_transfer_params_transport not in {"top_level", "custom_params"}:
            raise ValueError(
                "kv_transfer_params_transport must be 'top_level' or 'custom_params'"
            )
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(
            self, "extra_body", _json_object_mapping(self.extra_body, "extra_body")
        )
        object.__setattr__(
            self, "extra_headers", _string_mapping(self.extra_headers, "extra_headers")
        )


def _default_endpoint(request_mode: OpenAICompatibleRequestMode) -> str:
    if request_mode == "chat":
        return _CHAT_COMPLETIONS_ENDPOINT
    return _COMPLETIONS_ENDPOINT


def _validate_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")


def _validate_positive_int(value: Any, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _validate_non_negative_finite_number(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be a non-negative finite number")


def _validate_positive_finite_number(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive finite number")


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if not isinstance(item, str):
            raise ValueError(f"{field_name}.{key} must be a string")
        normalized[key] = item
    return normalized


def _json_object_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        normalized[key] = _json_compatible_value(item, f"{field_name}.{key}")
    return normalized


def _json_compatible_value(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be JSON-compatible")
        return value
    if isinstance(value, Mapping):
        return _json_object_mapping(value, field_name)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [
            _json_compatible_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{field_name} must be JSON-compatible")


def _chat_message_mapping(value: Any, field_name: str) -> dict[str, Any]:
    message = _json_object_mapping(value, field_name)
    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or not role:
        raise ValueError(f"{field_name}.role must be a non-empty string")
    if not isinstance(content, str):
        raise ValueError(f"{field_name}.content must be a string")
    return message


class OpenAICompatibleCompletionEngine:
    """BenchmarkEngine for vLLM/SGLang OpenAI-compatible completion servers."""

    def __init__(
        self,
        config: OpenAICompatibleEngineConfig,
        *,
        extra_body_factory: (
            Callable[[BenchmarkEngineRequest], Mapping[str, Any]] | None
        ) = None,
        chat_messages_factory: (
            Callable[[BenchmarkEngineRequest], Sequence[Mapping[str, Any]]] | None
        ) = None,
        token_counter: TokenCounter | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.extra_body_factory = extra_body_factory
        self.chat_messages_factory = chat_messages_factory
        self.token_counter = token_counter or WhitespaceTokenCounter()
        self.clock = clock

    def generate(self, request: BenchmarkEngineRequest) -> BenchmarkGeneration:
        started = self.clock()
        payload = self._payload(request)
        response = self._post_json(payload)
        try:
            if payload["stream"]:
                return self._stream_generation(
                    request, response, started, cache_salt=payload.get("cache_salt")
                )
            return self._completion_generation(
                request, response, started, cache_salt=payload.get("cache_salt")
            )
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    def _payload(self, request: BenchmarkEngineRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model_id or request.model_id,
            "temperature": self.config.temperature,
            "stream": self.config.stream,
        }
        if self.config.request_mode == "chat":
            payload["messages"] = self._chat_messages(request)
            payload["max_completion_tokens"] = self.config.max_tokens
        else:
            payload["prompt"] = self._prompt_text(request)
            payload["max_tokens"] = self.config.max_tokens
        if self.config.stream and self.config.include_usage:
            payload["stream_options"] = {"include_usage": True}
        payload.update(self._extra_body(request))
        if request.kv_transfer_params:
            kv_transfer_params = dict(request.kv_transfer_params)
            kv_transfer_params[DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM] = (
                self.config.prompt_text_mode
            )
            if self.config.kv_transfer_params_transport == "custom_params":
                custom_params = _json_object_mapping(
                    payload.get("custom_params") or {}, "custom_params"
                )
                custom_params["kv_transfer_params"] = kv_transfer_params
                payload["custom_params"] = custom_params
            else:
                if request.request_id:
                    payload["request_id"] = request.request_id
                payload["kv_transfer_params"] = kv_transfer_params
        return payload

    def _extra_body(self, request: BenchmarkEngineRequest) -> dict[str, Any]:
        extra_body = dict(self.config.extra_body)
        if self.extra_body_factory is not None:
            extra_body.update(
                _json_object_mapping(
                    self.extra_body_factory(request), "extra_body_factory"
                )
            )
        return extra_body

    def _prompt_text(self, request: BenchmarkEngineRequest) -> str:
        if self.config.prompt_text_mode == "runtime":
            return request.prompt_text
        return request.logical_prompt_text

    def _chat_messages(self, request: BenchmarkEngineRequest) -> list[dict[str, Any]]:
        if self.chat_messages_factory is None:
            raw_messages: Sequence[Mapping[str, Any]] = (
                {"role": "user", "content": self._prompt_text(request)},
            )
        else:
            raw_messages = self.chat_messages_factory(request)
        if isinstance(raw_messages, (str, bytes, bytearray)) or not isinstance(
            raw_messages, Sequence
        ):
            raise ValueError(
                "chat_messages_factory must return a sequence of JSON object messages"
            )
        messages = [
            _chat_message_mapping(message, f"chat_messages[{index}]")
            for index, message in enumerate(raw_messages)
        ]
        if not messages:
            raise ValueError("chat_messages_factory must return at least one message")
        return messages

    def _post_json(self, payload: Mapping[str, Any]) -> Any:
        endpoint = self.config.endpoint
        if endpoint is None:
            raise RuntimeError("OpenAI-compatible endpoint was not configured")
        request = _urlrequest.Request(
            _urlparse.urljoin(
                self.config.base_url.rstrip("/") + "/", endpoint.lstrip("/")
            ),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            return _active_urlopen()(request, timeout=self.config.timeout_seconds)
        except _urlerror.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            finally:
                close = getattr(exc, "close", None)
                if callable(close):
                    close()
            raise RuntimeError(
                f"OpenAI-compatible server returned HTTP {exc.code}: {detail}"
            ) from exc
        except _urlerror.URLError as exc:
            raise RuntimeError(
                f"OpenAI-compatible server request failed: {exc.reason}"
            ) from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.config.extra_headers}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _stream_generation(
        self,
        request: BenchmarkEngineRequest,
        response: Any,
        started: float,
        *,
        cache_salt: Any,
    ) -> BenchmarkGeneration:
        first_token_at: float | None = None
        output_parts: list[str] = []
        usage: Mapping[str, Any] = {}
        for event in _iter_sse_events(response):
            if event == "[DONE]":
                break
            data = json.loads(event)
            _raise_for_api_error(data)
            usage = data.get("usage") or usage
            delta = _choice_text(data)
            if delta:
                if first_token_at is None:
                    first_token_at = self.clock()
                output_parts.append(delta)
        completed = self.clock()
        output_text = "".join(output_parts)
        ttft = (first_token_at or completed) - started
        prompt_tokens, prompt_token_source, token_metadata = self._prompt_token_count(
            request, usage
        )
        return BenchmarkGeneration(
            output_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=_usage_count(
                usage, "completion_tokens", self.token_counter.count(output_text)
            ),
            ttft_seconds=ttft,
            time_to_completion_seconds=completed - started,
            metadata=self._generation_metadata(
                request,
                stream=True,
                prompt_token_source=prompt_token_source,
                token_metadata=token_metadata,
                cache_salt=cache_salt,
            ),
        )

    def _completion_generation(
        self,
        request: BenchmarkEngineRequest,
        response: Any,
        started: float,
        *,
        cache_salt: Any,
    ) -> BenchmarkGeneration:
        data = json.loads(response.read().decode("utf-8"))
        _raise_for_api_error(data)
        completed = self.clock()
        output_text = _choice_text(data)
        usage = data.get("usage") or {}
        prompt_tokens, prompt_token_source, token_metadata = self._prompt_token_count(
            request, usage
        )
        return BenchmarkGeneration(
            output_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=_usage_count(
                usage, "completion_tokens", self.token_counter.count(output_text)
            ),
            ttft_seconds=completed - started,
            time_to_completion_seconds=completed - started,
            metadata=self._generation_metadata(
                request,
                stream=False,
                prompt_token_source=prompt_token_source,
                token_metadata=token_metadata,
                cache_salt=cache_salt,
            ),
        )

    def _generation_metadata(
        self,
        request: BenchmarkEngineRequest,
        *,
        stream: bool,
        prompt_token_source: str,
        token_metadata: Mapping[str, str],
        cache_salt: Any,
    ) -> dict[str, str]:
        metadata = {
            "server": "openai-compatible",
            "stream": "true" if stream else "false",
            "request_mode": self.config.request_mode,
            "prompt_text_mode": self.config.prompt_text_mode,
            "prompt_token_source": prompt_token_source,
            "kv_transfer_params_attached": (
                "true" if request.kv_transfer_params else "false"
            ),
            **token_metadata,
        }
        if isinstance(cache_salt, str) and cache_salt:
            metadata["prefix_cache_salt_attached"] = "true"
            metadata["prefix_cache_salt"] = cache_salt
        if request.request_id:
            metadata["request_id"] = request.request_id
        return metadata

    def _prompt_token_count(
        self,
        request: BenchmarkEngineRequest,
        usage: Mapping[str, Any],
    ) -> tuple[int, str, dict[str, str]]:
        logical_count = self.token_counter.count(request.logical_prompt_text)
        runtime_count = self.token_counter.count(self._prompt_text(request))
        context_count = (
            runtime_count
            if self.config.prompt_text_mode == "runtime"
            else logical_count
        )
        context_source = self.config.prompt_text_mode
        metadata = {
            "logical_prompt_tokens": str(logical_count),
            "runtime_prompt_tokens": str(runtime_count),
        }
        if self.config.prompt_token_accounting == "server_usage":
            value = usage.get("prompt_tokens")
            if isinstance(value, int) and value >= 0:
                metadata["server_usage_prompt_tokens"] = str(value)
                metadata["server_usage_prompt_tokens_present"] = "true"
            else:
                metadata["server_usage_prompt_tokens_present"] = "false"
            return context_count, context_source, metadata
        return context_count, context_source, metadata


def _iter_sse_events(response: Any) -> Iterator[str]:
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data_lines.append(line.removeprefix("data:").strip())
    if data_lines:
        yield "\n".join(data_lines)


def _active_urlopen() -> Callable[..., Any]:
    return _urlopen


def _choice_text(data: Mapping[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return str(choice["text"])
    message = choice.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    delta = choice.get("delta")
    if isinstance(delta, Mapping):
        return str(delta.get("content") or "")
    return ""


def _usage_count(usage: Mapping[str, Any], key: str, fallback: int) -> int:
    value = usage.get(key)
    if isinstance(value, int) and value >= 0:
        return value
    return fallback


def _raise_for_api_error(data: Mapping[str, Any]) -> None:
    error = data.get("error")
    if not error:
        return
    if isinstance(error, Mapping):
        message = error.get("message") or error.get("type") or error
    else:
        message = error
    raise RuntimeError(f"OpenAI-compatible server returned error: {message}")
