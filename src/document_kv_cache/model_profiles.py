"""Public document namespace for model KV-layout profiles."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.model_profiles",
    (
        "DTYPE_BYTE_WIDTHS",
        "AttentionMechanism",
        "KVStorageLayout",
        "KVLayout",
        "dtype_byte_width",
        "kv_storage_layout_from_value",
        "KVModelProfile",
        "MODEL_PROFILE_RECORD_TYPE",
        "ModelProfileDefinition",
        "ModelProfileRegistry",
        "QWEN3_4B_INSTRUCT_PROFILE",
        "builtin_model_profiles",
        "default_model_profile_registry",
        "get_model_profile",
        "layout_for_model",
        "model_profile_definition_from_record",
        "model_profile_definition_to_record",
        "read_model_profile_definition_json",
        "write_model_profile_definition_json",
    ),
    globals(),
)

del reexport_public
