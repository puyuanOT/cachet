from __future__ import annotations

import pickle
from types import SimpleNamespace

from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import (
    build_engine_adapter_request,
    build_engine_kv_connector_actions,
    build_engine_kv_injection_plan,
    engine_adapter_request_to_record,
    vllm_adapter_spec,
    view_engine_adapter_payload,
)
from document_kv_cache.engine_probe import write_engine_adapter_handoff_bundle
from vllm_kv_injection.protocol import KVCacheHandle, KVLayout, KVSegment
from vllm_kv_injection.vllm_dynamic_connector import (
    DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY,
    DocumentKVConnector,
)
import vllm_kv_injection.vllm_runtime_preflight as vllm_runtime_preflight
from vllm_kv_injection.vllm_native_provider import (
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
    DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE,
    DOCUMENT_KV_VLLM_LAYER_MAPPING_SCHEMA_VERSION,
    DocumentKVHandoffLoad,
    DocumentKVNativeProvider,
    DocumentKVNativeProbeConnector,
    KVTransferParamsDocumentKVSource,
    document_kv_vllm_layer_index_from_name,
    document_kv_vllm_layer_mapping_record_issues,
    document_kv_vllm_layer_mapping_to_record,
    document_kv_vllm_probe_layer_names,
    inspect_document_kv_vllm_layer_mapping,
    validate_document_kv_vllm_layer_mapping_record,
)
from vllm_kv_injection.vllm_runtime_contract import (
    VLLMInstalledKVConnectorContract,
    VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS,
    VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
    installed_vllm_kv_connector_v1_contract_to_record,
)
from vllm_kv_injection.vllm_runtime_preflight import (
    validate_document_kv_vllm_runtime_preflight_record,
)

import pytest

torch = pytest.importorskip("torch")


def layout() -> KVLayout:
    return KVLayout(
        model_id="tiny-test-model",
        lora_id="base",
        layout_version="standard-v1",
        dtype="int8",
        num_layers=2,
        block_size=2,
        bytes_per_token=8,
        num_query_heads=1,
        num_kv_heads=1,
        head_size=2,
        kv_stride_bytes=2,
    )


def handle() -> KVCacheHandle:
    return KVCacheHandle(
        request_id="req-1",
        handle_uri="document-kv://req-1",
        layout=layout(),
        segments=(
            KVSegment("doc-a", "document_static", "static", 0, 2, 0, 16),
            KVSegment("doc-a", "document_chunk", "chunk-a", 2, 1, 16, 8),
        ),
        total_tokens=3,
        total_bytes=24,
    )


def payload() -> bytes:
    return bytes(
        (
            1,
            2,
            3,
            4,
            11,
            12,
            13,
            14,
            5,
            6,
            7,
            8,
            15,
            16,
            17,
            18,
            9,
            10,
            19,
            20,
            21,
            22,
            23,
            24,
        )
    )


def ready_request() -> EngineReadyRequest:
    return EngineReadyRequest(handle=handle(), payload=payload(), estimated_gpu_bytes=24)


def extended_ready_request() -> EngineReadyRequest:
    extended_handle = KVCacheHandle(
        request_id="req-1",
        handle_uri="document-kv://req-1",
        layout=layout(),
        segments=(
            KVSegment("doc-a", "document_static", "static", 0, 2, 0, 16),
            KVSegment("doc-a", "document_chunk", "chunk-a", 2, 1, 16, 8),
            KVSegment("doc-a", "document_chunk", "chunk-b", 3, 2, 24, 16),
        ),
        total_tokens=5,
        total_bytes=40,
    )
    return EngineReadyRequest(
        handle=extended_handle,
        payload=bytes(range(1, 41)),
        estimated_gpu_bytes=40,
    )


def matching_installed_contract() -> dict:
    return installed_vllm_kv_connector_v1_contract_to_record(
        VLLMInstalledKVConnectorContract(
            package_version="0.23.0",
            importable=True,
            installed_methods=tuple(
                sorted(
                    (
                        *VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
                        *VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS,
                    )
                )
            ),
            installed_properties=("prefer_cross_layer_blocks", "role"),
        )
    )


def handoff_load() -> DocumentKVHandoffLoad:
    return _handoff_load_from_ready_request(ready_request())


def extended_handoff_load() -> DocumentKVHandoffLoad:
    return _handoff_load_from_ready_request(extended_ready_request())


def _handoff_load_from_ready_request(request: EngineReadyRequest) -> DocumentKVHandoffLoad:
    adapter_request = build_engine_adapter_request(request, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(adapter_request, payload_uri="disk:/tmp/cachet-req-1.kv")
    plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
    payload_view = view_engine_adapter_payload(record, request.payload)
    actions = build_engine_kv_connector_actions(plan, payload_view)
    return DocumentKVHandoffLoad(actions=actions, payload=request.payload)


class StaticHandoffSource:
    def __init__(self, load: DocumentKVHandoffLoad | None) -> None:
        self.load = load
        self.requests: list[str] = []

    def get_load(self, request):
        self.requests.append(request.request_id)
        return self.load


class AllocatedBlocks:
    def __init__(self, block_ids: list[int]) -> None:
        self.block_ids = block_ids

    def get_block_ids(self):
        return (self.block_ids,)


def scheduler_output(block_ids: list[int]):
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="req-1", block_ids=(block_ids,))],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
    )


def cached_scheduler_output(block_ids: list[int]):
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-1"],
            resumed_req_ids=set(),
            new_block_ids=[(block_ids,)],
        ),
    )


def resumed_cached_scheduler_output(block_ids: list[int]):
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-1"],
            resumed_req_ids={"req-1"},
            new_block_ids=[(block_ids,)],
        ),
    )


def test_native_provider_records_matched_token_allocation_metadata():
    source = StaticHandoffSource(handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(request_id="req-1", num_tokens=3, kv_transfer_params={})

    assert provider.get_num_new_matched_tokens(request, 0) == (2, False)
    provider.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = provider.build_connector_meta(scheduler_output([5, 7]))

    assert source.requests == ["req-1"]
    assert len(meta.loads) == 1
    load = meta.loads[0]
    assert load.request_id == "req-1"
    assert load.source_token_start == 0
    assert load.token_count == 2
    assert [(block.block_id, block.token_start, block.token_count, block.block_offset) for block in load.blocks] == [
        (5, 0, 2, 0),
    ]
    pickle.loads(pickle.dumps(meta))


def test_native_provider_records_cached_request_allocation_metadata():
    source = StaticHandoffSource(handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(request_id="req-1", num_tokens=3, kv_transfer_params={})

    assert provider.get_num_new_matched_tokens(request, 0) == (2, False)
    provider.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = provider.build_connector_meta(cached_scheduler_output([11, 13]))

    assert len(meta.loads) == 1
    load = meta.loads[0]
    assert [(block.block_id, block.token_start, block.token_count, block.block_offset) for block in load.blocks] == [
        (11, 0, 2, 0),
    ]


def test_native_provider_treats_cached_request_new_blocks_as_relative_metadata():
    source = StaticHandoffSource(extended_handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert provider.get_num_new_matched_tokens(request, 2) == (2, False)
    provider.update_state_after_alloc(request, AllocatedBlocks([3, 5, 7]), 2)
    meta = provider.build_connector_meta(cached_scheduler_output([101]))

    assert len(meta.loads) == 1
    load = meta.loads[0]
    assert load.source_token_start == 2
    assert load.token_count == 2
    assert [(block.block_id, block.token_start, block.token_count, block.block_offset) for block in load.blocks] == [
        (101, 0, 2, 0),
    ]


def test_native_provider_treats_resumed_cached_request_blocks_as_full_metadata():
    source = StaticHandoffSource(extended_handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert provider.get_num_new_matched_tokens(request, 2) == (2, False)
    provider.update_state_after_alloc(request, AllocatedBlocks([3, 5, 7]), 2)
    meta = provider.build_connector_meta(resumed_cached_scheduler_output([101, 103, 105]))

    assert len(meta.loads) == 1
    load = meta.loads[0]
    assert load.source_token_start == 2
    assert load.token_count == 2
    assert [(block.block_id, block.token_start, block.token_count, block.block_offset) for block in load.blocks] == [
        (103, 0, 2, 0),
    ]


def test_native_provider_rejects_allocations_missing_from_scheduler_output():
    source = StaticHandoffSource(handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(request_id="req-1", num_tokens=3, kv_transfer_params={})

    assert provider.get_num_new_matched_tokens(request, 0) == (2, False)
    provider.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)

    with pytest.raises(ValueError, match="scheduled vLLM block ids"):
        provider.build_connector_meta(
            SimpleNamespace(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
            )
        )


def test_native_provider_rejects_duplicate_scheduler_block_metadata():
    source = StaticHandoffSource(handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(request_id="req-1", num_tokens=3, kv_transfer_params={})

    assert provider.get_num_new_matched_tokens(request, 0) == (2, False)
    provider.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)

    with pytest.raises(ValueError, match="duplicate scheduled vLLM block ids"):
        provider.build_connector_meta(
            SimpleNamespace(
                scheduled_new_reqs=[SimpleNamespace(req_id="req-1", block_ids=([5, 7],))],
                scheduled_cached_reqs=SimpleNamespace(req_ids=["req-1"], new_block_ids=[([5, 7],)]),
            )
        )


def test_native_provider_reports_only_block_aligned_prefix_tokens():
    source = StaticHandoffSource(handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(request_id="req-1", num_tokens=3, kv_transfer_params={})

    assert provider.get_num_new_matched_tokens(request, 1) == (0, False)


def test_native_provider_copies_materialized_payload_into_registered_paged_kv_layers():
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=3, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    layer_0 = torch.zeros((8, 2, 2, 1, 2), dtype=torch.int8)
    layer_1 = torch.zeros((8, 2, 2, 1, 2), dtype=torch.int8)

    connector.register_kv_caches({"layer.0": layer_0, "layer.1": layer_1})
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert torch.equal(layer_0[5, :, 0], torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert torch.equal(layer_0[5, :, 1], torch.tensor([[[5, 6]], [[7, 8]]], dtype=torch.int8))
    assert torch.equal(layer_0[7, :, 0], torch.zeros((2, 1, 2), dtype=torch.int8))
    assert torch.equal(layer_1[5, :, 0], torch.tensor([[[11, 12]], [[13, 14]]], dtype=torch.int8))
    assert torch.equal(layer_1[5, :, 1], torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8))
    assert torch.equal(layer_1[7, :, 0], torch.zeros((2, 1, 2), dtype=torch.int8))
    assert connector.get_kv_connector_stats()["document_kv_layers_loaded"] == 2
    assert connector.take_events() == [{"event": "document_kv_loaded", "request_id": "req-1"}]


def test_native_provider_maps_vllm_layer_names_independently_of_registration_order():
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=3, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    layer_0 = torch.zeros((8, 2, 2, 1, 2), dtype=torch.int8)
    layer_1 = torch.zeros((8, 2, 2, 1, 2), dtype=torch.int8)

    connector.register_kv_caches(
        {
            "model.layers.1.self_attn.attn": layer_1,
            "model.layers.0.self_attn.attn": layer_0,
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert torch.equal(layer_0[5, :, 0], torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert torch.equal(layer_1[5, :, 0], torch.tensor([[[11, 12]], [[13, 14]]], dtype=torch.int8))


def test_native_provider_rejects_unparseable_registered_vllm_layer_names():
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))

    with pytest.raises(ValueError, match="Cannot determine vLLM layer index"):
        provider.register_kv_caches(
            {
                "attention_a": torch.zeros((8, 2, 2, 1, 2), dtype=torch.int8),
                "attention_b": torch.zeros((8, 2, 2, 1, 2), dtype=torch.int8),
            }
        )


def test_native_provider_rejects_duplicate_registered_vllm_layer_indices():
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))

    with pytest.raises(ValueError, match="Duplicate vLLM layer index"):
        provider.register_kv_caches(
            {
                "model.layers.0.self_attn.attn": torch.zeros((8, 2, 2, 1, 2), dtype=torch.int8),
                "decoder.layers.0.self_attn.attn": torch.zeros((8, 2, 2, 1, 2), dtype=torch.int8),
            }
        )


def test_vllm_layer_mapping_diagnostic_accepts_runtime_layer_names():
    inspection = inspect_document_kv_vllm_layer_mapping(
        [
            "model.layers.1.self_attn.attn",
            "model.layers.0.self_attn.attn",
        ]
    )

    assert inspection.ok is True
    assert inspection.layer_indices == {
        "model.layers.1.self_attn.attn": 1,
        "model.layers.0.self_attn.attn": 0,
    }
    assert document_kv_vllm_layer_index_from_name("language_model.model.layers.12.self_attn") == 12
    assert document_kv_vllm_layer_index_from_name("attention") is None

    record = document_kv_vllm_layer_mapping_to_record(inspection)

    assert record == {
        "record_type": DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE,
        "schema_version": DOCUMENT_KV_VLLM_LAYER_MAPPING_SCHEMA_VERSION,
        "runtime": "vllm-kv-connector-v1",
        "layer_names": [
            "model.layers.1.self_attn.attn",
            "model.layers.0.self_attn.attn",
        ],
        "layer_indices": {
            "model.layers.1.self_attn.attn": 1,
            "model.layers.0.self_attn.attn": 0,
        },
        "unresolved_layer_names": [],
        "duplicate_layer_indices": {},
        "ok": True,
    }
    validate_document_kv_vllm_layer_mapping_record(record)


def test_vllm_probe_layer_names_match_layer_mapping_contract():
    names = document_kv_vllm_probe_layer_names(layout())

    assert names == ("probe.layer.0", "probe.layer.1")
    assert document_kv_vllm_layer_mapping_to_record(names)["ok"] is True


def test_vllm_layer_mapping_preflight_rejects_unresolved_and_duplicate_names():
    record = document_kv_vllm_layer_mapping_to_record(
        [
            "attention_without_index",
            "model.layers.0.self_attn.attn",
            "decoder.layers.0.self_attn.attn",
        ]
    )

    assert record["ok"] is False
    assert record["layer_indices"] == {
        "model.layers.0.self_attn.attn": 0,
        "decoder.layers.0.self_attn.attn": 0,
    }
    assert record["unresolved_layer_names"] == ["attention_without_index"]
    assert record["duplicate_layer_indices"] == {
        "0": [
            "decoder.layers.0.self_attn.attn",
            "model.layers.0.self_attn.attn",
        ]
    }
    assert "ok must be true for a safe vLLM layer mapping preflight" in (
        document_kv_vllm_layer_mapping_record_issues(record)
    )
    with pytest.raises(ValueError, match="ok must be true"):
        validate_document_kv_vllm_layer_mapping_record(record)


def test_vllm_layer_mapping_record_rejects_inconsistent_derived_fields():
    record = document_kv_vllm_layer_mapping_to_record(["model.layers.0.self_attn.attn"])
    record["ok"] = False
    record["layer_indices"] = {"model.layers.0.self_attn.attn": 2}
    record["unexpected"] = True

    issues = document_kv_vllm_layer_mapping_record_issues(record)

    assert any("unsupported keys" in issue and "unexpected" in issue for issue in issues)
    assert "layer_indices must match layer_names" in issues
    assert "ok must match layer_names" in issues
    with pytest.raises(ValueError, match="layer_indices"):
        validate_document_kv_vllm_layer_mapping_record(record)


def test_vllm_layer_mapping_diagnostic_is_exported_from_cachet_adapter_facade():
    import cachet.adapters.vllm as cachet_vllm
    import vllm_kv_injection

    assert (
        cachet_vllm.inspect_document_kv_vllm_layer_mapping
        is vllm_kv_injection.inspect_document_kv_vllm_layer_mapping
    )
    assert (
        cachet_vllm.document_kv_vllm_layer_mapping_to_record(["model.layers.0.self_attn.attn"])["ok"]
        is True
    )
    assert cachet_vllm.document_kv_vllm_probe_layer_names(layout()) == ("probe.layer.0", "probe.layer.1")


def test_native_provider_handshake_metadata_includes_runtime_preflight(monkeypatch):
    monkeypatch.setattr(
        vllm_runtime_preflight,
        "installed_vllm_kv_connector_v1_contract_to_record",
        matching_installed_contract,
    )
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(None))
    connector = DocumentKVConnector(provider=provider)
    layer_0 = torch.zeros((8, 2, 2, 1, 2), dtype=torch.int8)
    layer_1 = torch.zeros((8, 2, 2, 1, 2), dtype=torch.int8)

    connector.register_kv_caches(
        {
            "model.layers.0.self_attn.attn": layer_0,
            "model.layers.1.self_attn.attn": layer_1,
        }
    )
    record = connector.get_handshake_metadata()

    assert record is not None
    assert record["ok"] is True
    assert record["layer_mapping"] == provider.vllm_layer_mapping_record()
    assert record["layer_mapping"]["layer_indices"] == {
        "model.layers.0.self_attn.attn": 0,
        "model.layers.1.self_attn.attn": 1,
    }
    validate_document_kv_vllm_runtime_preflight_record(record)


def test_kv_transfer_params_source_reads_cachet_handoff_bundle(tmp_path):
    adapter_request = build_engine_adapter_request(ready_request(), spec=vllm_adapter_spec())
    handoff_path, _payload_path = write_engine_adapter_handoff_bundle(
        adapter_request,
        tmp_path / "handoff.json",
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    request = SimpleNamespace(
        request_id="req-1",
        kv_transfer_params={DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path)},
    )

    load = KVTransferParamsDocumentKVSource().get_load(request)

    assert load is not None
    assert load.request_id == "req-1"
    assert load.total_tokens == 3
    assert load.payload == payload()


def test_native_provider_factory_is_release_safe_provider_wiring():
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY: DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
            }
        )
    )

    connector = DocumentKVConnector(vllm_config=vllm_config)

    assert isinstance(connector.provider, DocumentKVNativeProvider)
    assert connector.provider.document_kv_native_provider is True


def test_native_probe_connector_does_not_require_vllm_base_config(monkeypatch):
    def fail_base_init(*args, **kwargs):
        raise AssertionError("probe connector must not initialize DocumentKVConnector base")

    monkeypatch.setattr(
        "vllm_kv_injection.vllm_dynamic_connector.DocumentKVConnector.__init__",
        fail_base_init,
    )

    connector = DocumentKVNativeProbeConnector()

    assert isinstance(connector.provider, DocumentKVNativeProvider)
