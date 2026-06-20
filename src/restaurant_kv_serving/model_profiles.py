"""Compatibility wrapper for :mod:`document_kv_cache.model_profiles`."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from document_kv_cache.model_profiles import (
    DTYPE_BYTE_WIDTHS,
    MODEL_PROFILE_RECORD_TYPE,
    QWEN3_4B_INSTRUCT_PROFILE,
    AttentionMechanism,
    KVLayout,
    KVModelProfile,
    KVStorageLayout,
    ModelProfileDefinition,
    ModelProfileRegistry,
    builtin_model_profiles,
    default_model_profile_registry,
    dtype_byte_width,
    get_model_profile,
    kv_storage_layout_from_value,
    layout_for_model,
    model_profile_definition_from_record,
    model_profile_definition_to_record,
    read_model_profile_definition_json,
    write_model_profile_definition_json,
)
from restaurant_kv_serving.storage import local_path
