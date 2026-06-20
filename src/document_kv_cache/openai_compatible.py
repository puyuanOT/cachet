"""OpenAI-compatible completion engine for document KV-cache benchmarks."""

from __future__ import annotations

import json
import sys
import time
import urllib.error as _urlerror
import urllib.parse as _urlparse
import urllib.request as _urlrequest
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from document_kv_cache.benchmark_runner import BenchmarkEngineRequest, BenchmarkGeneration

__all__ = [
    "TokenCounter",
    "PromptTextMode",
    "PromptTokenAccounting",
    "WhitespaceTokenCounter",
    "OpenAICompatibleEngineConfig",
    "OpenAICompatibleCompletionEngine",
]

_stdlib_urlopen = _urlrequest.urlopen
_urlopen = _stdlib_urlopen


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


PromptTextMode = Literal["logical", "runtime"]
PromptTokenAccounting = Literal["logical", "server_usage"]


@dataclass(frozen=True, slots=True)
class WhitespaceTokenCounter:
    def count(self, text: str) -> int:
        return len(text.split())


@dataclass(frozen=True, slots=True)
class OpenAICompatibleEngineConfig:
    base_url: str
    endpoint: str = "/v1/completions"
    api_key: str | None = None
    timeout_seconds: float = 120.0
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = True
    include_usage: bool = True
    model_id: str | None = None
    prompt_text_mode: PromptTextMode = "logical"
    prompt_token_accounting: PromptTokenAccounting = "logical"
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.base_url, "base_url")
        _validate_non_empty_string(self.endpoint, "endpoint")
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
            raise ValueError("prompt_token_accounting must be 'logical' or 'server_usage'")
        object.__setattr__(self, "extra_body", _json_object_mapping(self.extra_body, "extra_body"))
        object.__setattr__(self, "extra_headers", _string_mapping(self.extra_headers, "extra_headers"))


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
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return [_json_compatible_value(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{field_name} must be JSON-compatible")


class OpenAICompatibleCompletionEngine:
    """BenchmarkEngine for vLLM/SGLang OpenAI-compatible completion servers."""

    def __init__(
        self,
        config: OpenAICompatibleEngineConfig,
        *,
        token_counter: TokenCounter | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.token_counter = token_counter or WhitespaceTokenCounter()
        self.clock = clock

    def generate(self, request: BenchmarkEngineRequest) -> BenchmarkGeneration:
        started = self.clock()
        payload = self._payload(request)
        response = self._post_json(payload)
        try:
            if payload["stream"]:
                return self._stream_generation(request, response, started)
            return self._completion_generation(request, response, started)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    def _payload(self, request: BenchmarkEngineRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model_id or request.model_id,
            "prompt": self._prompt_text(request),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": self.config.stream,
        }
        if self.config.stream and self.config.include_usage:
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.config.extra_body)
        return payload

    def _prompt_text(self, request: BenchmarkEngineRequest) -> str:
        if self.config.prompt_text_mode == "runtime":
            return request.prompt_text
        return request.logical_prompt_text

    def _post_json(self, payload: Mapping[str, Any]) -> Any:
        request = _urlrequest.Request(
            _urlparse.urljoin(self.config.base_url.rstrip("/") + "/", self.config.endpoint.lstrip("/")),
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
            raise RuntimeError(f"OpenAI-compatible server returned HTTP {exc.code}: {detail}") from exc
        except _urlerror.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible server request failed: {exc.reason}") from exc

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
        prompt_tokens, prompt_token_source, token_metadata = self._prompt_token_count(request, usage)
        return BenchmarkGeneration(
            output_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=_usage_count(usage, "completion_tokens", self.token_counter.count(output_text)),
            ttft_seconds=ttft,
            time_to_completion_seconds=completed - started,
            metadata={
                "server": "openai-compatible",
                "stream": "true",
                "prompt_text_mode": self.config.prompt_text_mode,
                "prompt_token_source": prompt_token_source,
                **token_metadata,
            },
        )

    def _completion_generation(
        self,
        request: BenchmarkEngineRequest,
        response: Any,
        started: float,
    ) -> BenchmarkGeneration:
        data = json.loads(response.read().decode("utf-8"))
        _raise_for_api_error(data)
        completed = self.clock()
        output_text = _choice_text(data)
        usage = data.get("usage") or {}
        prompt_tokens, prompt_token_source, token_metadata = self._prompt_token_count(request, usage)
        return BenchmarkGeneration(
            output_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=_usage_count(usage, "completion_tokens", self.token_counter.count(output_text)),
            ttft_seconds=completed - started,
            time_to_completion_seconds=completed - started,
            metadata={
                "server": "openai-compatible",
                "stream": "false",
                "prompt_text_mode": self.config.prompt_text_mode,
                "prompt_token_source": prompt_token_source,
                **token_metadata,
            },
        )

    def _prompt_token_count(
        self,
        request: BenchmarkEngineRequest,
        usage: Mapping[str, Any],
    ) -> tuple[int, str, dict[str, str]]:
        logical_count = self.token_counter.count(request.logical_prompt_text)
        runtime_count = self.token_counter.count(self._prompt_text(request))
        metadata = {
            "logical_prompt_tokens": str(logical_count),
            "runtime_prompt_tokens": str(runtime_count),
        }
        if self.config.prompt_token_accounting == "server_usage":
            value = usage.get("prompt_tokens")
            if isinstance(value, int) and value >= 0:
                return value, "server_usage", metadata
            return logical_count, "logical_fallback", metadata
        return logical_count, "logical", metadata


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
    # Preserve the old module-level monkeypatch hook while the legacy namespace exists.
    legacy_module = sys.modules.get("restaurant_kv_serving.openai_compatible")
    legacy_urlopen = getattr(legacy_module, "urlopen", None)
    if legacy_urlopen is not None and legacy_urlopen is not _stdlib_urlopen:
        return legacy_urlopen
    return _urlopen


def _choice_text(data: Mapping[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return str(choice["text"])
    delta = choice.get("delta") or {}
    if isinstance(delta, Mapping):
        return str(delta.get("content") or "")
    message = choice.get("message") or {}
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
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
