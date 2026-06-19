"""Compatibility wrapper for :mod:`document_kv_cache.live_server`."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from document_kv_cache._reexport import LegacyMainBridge, reexport_public
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

__all__ = reexport_public(
    "document_kv_cache.live_server",
    (
        "LIVE_CHECK_SUITE_ID",
        "DEFAULT_LIVE_CHECK_ANSWER",
        "LiveServerCheckConfig",
        "LiveServerCheckResult",
        "build_live_server_check_request",
        "run_openai_compatible_live_check",
        "main",
    ),
    globals(),
)

__all__ += [
    "argparse",
    "json",
    "Mapping",
    "Sequence",
    "dataclass",
    "field",
    "Any",
    "BenchmarkEngine",
    "BenchmarkEngineRequest",
    "BenchmarkGeneration",
    "DEFAULT_HARDWARE_TARGET",
    "DEFAULT_V1_MODEL_ID",
    "BenchmarkExample",
    "answer_found",
    "baseline_prefill_arm",
    "build_prompt_parts",
    "document_kv_cache_arm",
    "OpenAICompatibleCompletionEngine",
    "OpenAICompatibleEngineConfig",
    "PromptTextMode",
    "PromptTokenAccounting",
    "SourceDocument",
]

_main_bridge = LegacyMainBridge(
    legacy_module_name="document_kv_cache.live_server",
    public_namespace=globals(),
    hook_names=tuple(name for name in __all__ if name != "main"),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del LegacyMainBridge
del reexport_public
