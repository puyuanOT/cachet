"""String constants for Cachet's vLLM native provider path."""

DOCUMENT_KV_HANDOFF_JSON_PARAM = "document_kv.handoff_json"
DOCUMENT_KV_HANDOFF_RECORD_PARAM = "document_kv.handoff_record"
DOCUMENT_KV_PAYLOAD_URI_PARAM = "document_kv.payload_uri"
DOCUMENT_KV_REQUEST_ID_PARAM = "document_kv.request_id"
DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM = "document_kv.prompt_text_mode"
DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY = "document_kv.handoff_source_factory"
DOCUMENT_KV_NATIVE_PROVIDER_FACTORY = "vllm_kv_injection.vllm_native_provider:build_document_kv_provider"

__all__ = [
    "DOCUMENT_KV_HANDOFF_JSON_PARAM",
    "DOCUMENT_KV_HANDOFF_RECORD_PARAM",
    "DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY",
    "DOCUMENT_KV_NATIVE_PROVIDER_FACTORY",
    "DOCUMENT_KV_PAYLOAD_URI_PARAM",
    "DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM",
    "DOCUMENT_KV_REQUEST_ID_PARAM",
]
