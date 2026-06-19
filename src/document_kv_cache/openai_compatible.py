"""Public document namespace for OpenAI-compatible benchmark engines."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.openai_compatible",
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

del reexport_public
