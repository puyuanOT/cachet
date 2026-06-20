"""Compatibility wrapper for :mod:`document_kv_cache.serving_env`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from document_kv_cache.serving_env import (
    FASTAPI_CONSTRAINT,
    HUGGINGFACE_HUB_CONSTRAINT,
    NUMPY_CONSTRAINT,
    PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT,
    SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE,
    SGLANG_DEPENDENCY_CONSTRAINTS,
    SGLANG_SERVING_ENVIRONMENT_PROFILE,
    SGLANG_VERSION,
    TOKENIZERS_CONSTRAINT,
    TRANSFORMERS_CONSTRAINT,
    VLLM_DEPENDENCY_CONSTRAINTS,
    VLLM_SERVING_ENVIRONMENT_PROFILE,
    VLLM_VERSION,
    ServingEnvironmentProfile,
    serving_environment_profile,
    serving_environment_profile_to_record,
    serving_environment_profiles,
    serving_environment_profiles_to_record,
)
from restaurant_kv_serving.engine_adapters import ServingBackend

__all__ = [
    "FASTAPI_CONSTRAINT",
    "HUGGINGFACE_HUB_CONSTRAINT",
    "NUMPY_CONSTRAINT",
    "PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT",
    "SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE",
    "SGLANG_DEPENDENCY_CONSTRAINTS",
    "SGLANG_SERVING_ENVIRONMENT_PROFILE",
    "SGLANG_VERSION",
    "ServingEnvironmentProfile",
    "TOKENIZERS_CONSTRAINT",
    "TRANSFORMERS_CONSTRAINT",
    "VLLM_DEPENDENCY_CONSTRAINTS",
    "VLLM_SERVING_ENVIRONMENT_PROFILE",
    "VLLM_VERSION",
    "serving_environment_profile",
    "serving_environment_profile_to_record",
    "serving_environment_profiles",
    "serving_environment_profiles_to_record",
]
