"""Compatibility wrapper for :mod:`document_kv_cache.openai_compatible`."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from document_kv_cache._reexport import reexport_public
from document_kv_cache.benchmark_runner import BenchmarkEngineRequest, BenchmarkGeneration

__all__ = reexport_public(
    "document_kv_cache.openai_compatible",
    (
        "TokenCounter",
        "PromptTextMode",
        "PromptTokenAccounting",
        "WhitespaceTokenCounter",
        "OpenAICompatibleEngineConfig",
        "OpenAICompatibleCompletionEngine",
    ),
    globals(),
)

__all__ += [
    "json",
    "time",
    "Callable",
    "Iterator",
    "Mapping",
    "dataclass",
    "field",
    "Any",
    "Literal",
    "Protocol",
    "HTTPError",
    "URLError",
    "urljoin",
    "Request",
    "urlopen",
    "BenchmarkEngineRequest",
    "BenchmarkGeneration",
]

del reexport_public
