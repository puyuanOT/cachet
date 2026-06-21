import pytest

from document_kv_cache.native_probe_factories import native_probe_adapter_contract_to_record
from vllm_kv_injection.vllm_runtime_contract import (
    VLLM_KV_CONNECTOR_V1_CONTRACT,
    VLLM_KV_CONNECTOR_V1_CONTRACT_RECORD_TYPE,
    VLLM_KV_CONNECTOR_V1_CONTRACT_SCHEMA_VERSION,
    VLLM_KV_CONNECTOR_V1_DOC_URL,
    VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS,
    VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
    VLLM_KV_CONNECTOR_V1_RUNTIME,
    validate_vllm_kv_connector_v1_contract_record,
    validate_vllm_kv_connector_v1_methods,
    vllm_kv_connector_v1_contract_record_issues,
    vllm_kv_connector_v1_contract_to_record,
    vllm_kv_connector_v1_method_issues,
)


class CompleteVLLMKVConnectorV1:
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return None, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return None

    def build_connector_meta(self, scheduler_output):
        return {}

    def register_kv_caches(self, kv_caches):
        return None

    def start_load_kv(self, forward_context, **kwargs):
        return None

    def wait_for_layer_load(self, layer_name):
        return None

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return None

    def wait_for_save(self):
        return None

    def request_finished(self, request, block_ids):
        return False, None

    def request_finished_all_groups(self, request, block_ids):
        return False, None


def test_vllm_kv_connector_v1_contract_record_documents_runtime_lifecycle():
    record = vllm_kv_connector_v1_contract_to_record(
        handoff_contract=native_probe_adapter_contract_to_record(),
    )

    assert record == {
        "record_type": VLLM_KV_CONNECTOR_V1_CONTRACT_RECORD_TYPE,
        "schema_version": VLLM_KV_CONNECTOR_V1_CONTRACT_SCHEMA_VERSION,
        "runtime": VLLM_KV_CONNECTOR_V1_RUNTIME,
        "doc_url": VLLM_KV_CONNECTOR_V1_DOC_URL,
        "required_methods": list(VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS),
        "optional_methods": list(VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS),
        "handoff_contract": native_probe_adapter_contract_to_record(),
    }
    assert "get_num_new_matched_tokens" in record["required_methods"]
    assert "start_load_kv" in record["required_methods"]
    assert "wait_for_layer_load" in record["required_methods"]
    assert "save_kv_layer" in record["required_methods"]
    assert "wait_for_save" in record["required_methods"]
    assert "request_finished" in record["required_methods"]
    assert "request_finished_all_groups" in record["required_methods"]
    assert len(record["optional_methods"]) == len(set(record["optional_methods"]))
    assert "build_connector_worker_meta" in record["optional_methods"]
    assert "build_kv_connector_stats" in record["optional_methods"]
    assert "build_prom_metrics" in record["optional_methods"]
    assert "get_required_kvcache_layout" in record["optional_methods"]
    assert "handle_preemptions" in record["optional_methods"]
    assert "has_pending_push_work" in record["optional_methods"]
    assert "register_cross_layers_kv_cache" in record["optional_methods"]
    assert "update_connector_output" in record["optional_methods"]
    validate_vllm_kv_connector_v1_contract_record(record)


def test_vllm_kv_connector_v1_contract_record_rejects_shape_drift():
    record = vllm_kv_connector_v1_contract_to_record()
    record["required_methods"] = ["reserve", "inject", "release"]
    record["extra"] = True

    issues = vllm_kv_connector_v1_contract_record_issues(record)

    assert "vLLM V1 KV connector contract required_methods must match the package contract" in issues
    assert any("unsupported keys" in issue and "extra" in issue for issue in issues)
    with pytest.raises(ValueError, match="required_methods"):
        validate_vllm_kv_connector_v1_contract_record(record)


def test_validate_vllm_kv_connector_v1_methods_accepts_required_lifecycle():
    validate_vllm_kv_connector_v1_methods(CompleteVLLMKVConnectorV1())
    assert vllm_kv_connector_v1_method_issues(CompleteVLLMKVConnectorV1()) == ()


def test_validate_vllm_kv_connector_v1_methods_reports_missing_hooks():
    class IncompleteConnector:
        def start_load_kv(self, forward_context, **kwargs):
            return None

    issues = vllm_kv_connector_v1_method_issues(IncompleteConnector())

    assert len(issues) == 1
    assert "get_num_new_matched_tokens" in issues[0]
    assert "wait_for_layer_load" in issues[0]
    assert "request_finished" in issues[0]
    assert "request_finished_all_groups" in issues[0]
    with pytest.raises(TypeError, match="get_num_new_matched_tokens"):
        validate_vllm_kv_connector_v1_methods(IncompleteConnector())


def test_runtime_contract_is_exported_from_package_root():
    import vllm_kv_injection

    assert vllm_kv_injection.VLLM_KV_CONNECTOR_V1_RUNTIME == VLLM_KV_CONNECTOR_V1_RUNTIME
    assert vllm_kv_injection.VLLM_KV_CONNECTOR_V1_CONTRACT == VLLM_KV_CONNECTOR_V1_CONTRACT
    assert (
        vllm_kv_injection.vllm_kv_connector_v1_contract_to_record()
        == vllm_kv_connector_v1_contract_to_record()
    )
