from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import replace
from types import SimpleNamespace

from document_kv_cache.benchmark_handoffs import generate_benchmark_handoff_bundles
from document_kv_cache.artifact_identity import TokenContract
from document_kv_cache.cache import ChunkCache
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import (
    RuntimeOperationSupport,
    build_engine_adapter_request,
    build_engine_kv_connector_actions,
    build_engine_kv_injection_plan,
    engine_adapter_request_to_record,
    engine_kv_connector_actions_to_record,
    vllm_adapter_spec,
    view_engine_adapter_payload,
)
from document_kv_cache.reuse_contract import (
    ArtifactEncoding,
    PACKED_Q4_ARTIFACT_FORMAT,
    PayloadDecodeStage,
    PositionHandling,
    RuntimeOperationDescriptor,
    RuntimeOperationHandlerRegistry,
    RuntimeOperationPhase,
    RuntimeOperationResult,
    runtime_operation_config_digest,
)
from document_kv_cache.engine_probe import write_engine_adapter_handoff_bundle
from document_kv_cache.kvpack import PackChunk
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.materializer import KVMaterializer
from document_kv_cache.methods import MethodRegistry, MethodSpec, method_spec
from document_kv_cache.models import CacheGenerationMethod, DocumentKVRequest, KVCacheKey
from document_kv_cache.rope import apply_rope_to_keys
from document_kv_cache.serving_env import VLLM_PACKAGE_VERSION
from document_kv_cache.vllm_runtime_contract_data import (
    VLLM_KV_CONNECTOR_V1_BASE_SOURCE_SHA256,
)
from document_kv_cache.storage import DiskRangeReader
from document_kv_cache.workflow import (
    CacheBuildConfig,
    DocumentKVWorkflow,
    SourceDocument,
)
from vllm_kv_injection.protocol import KVCacheHandle, KVLayout, KVSegment
from vllm_kv_injection.vllm_dynamic_connector import (
    DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY,
    DocumentKVConnector,
)
from vllm_kv_injection.block_mapping import BlockSpan
import vllm_kv_injection.vllm_native_provider as vllm_native_provider
import vllm_kv_injection.vllm_runtime_preflight as vllm_runtime_preflight
from vllm_kv_injection.vllm_native_provider import (
    DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
    DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY,
    DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    DOCUMENT_KV_SEGMENTED_LOAD_STRATEGY_ENV,
    DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY,
    DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE,
    DOCUMENT_KV_VLLM_LAYER_MAPPING_SCHEMA_VERSION,
    DocumentKVConnectorMetadata,
    DocumentKVHandoffLoad,
    DocumentKVLoadRequest,
    DocumentKVNativeProvider,
    DocumentKVNativeProbeConnector,
    KVTransferParamsDocumentKVSource,
    build_document_kv_provider,
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
        cache_method="full_prefix_prefill",
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


def segmented_ready_request() -> EngineReadyRequest:
    return EngineReadyRequest(handle=handle(), payload=(payload()[:16], payload()[16:]), estimated_gpu_bytes=24)


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
        cache_method="full_prefix_prefill",
    )
    return EngineReadyRequest(
        handle=extended_handle,
        payload=bytes(range(1, 41)),
        estimated_gpu_bytes=40,
    )


def matching_installed_contract() -> dict:
    return installed_vllm_kv_connector_v1_contract_to_record(
        VLLMInstalledKVConnectorContract(
            package_version=VLLM_PACKAGE_VERSION,
            importable=True,
            base_source_sha256=VLLM_KV_CONNECTOR_V1_BASE_SOURCE_SHA256,
            installed_methods=tuple(
                sorted(
                    (
                        *VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
                        *VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS,
                    )
                )
            ),
            installed_properties=(
                "prefer_cross_layer_blocks",
                "requires_kv_delivery",
                "role",
            ),
        )
    )


def handoff_load() -> DocumentKVHandoffLoad:
    return _handoff_load_from_ready_request(ready_request())


def handoff_load_with_content_hashes(
    content_hashes: tuple[str, ...] = ("hash-doc-a-static", "hash-doc-a-chunk-a"),
) -> DocumentKVHandoffLoad:
    load = handoff_load()
    if len(content_hashes) != len(load.actions.copies):
        raise ValueError("content_hashes must match copy count")
    return DocumentKVHandoffLoad(
        actions=replace(
            load.actions,
            copies=tuple(
                replace(copy, content_hash=content_hash)
                for copy, content_hash in zip(load.actions.copies, content_hashes, strict=True)
            ),
        ),
        payload=load.payload,
    )


def extended_handoff_load() -> DocumentKVHandoffLoad:
    return _handoff_load_from_ready_request(extended_ready_request())


def segmented_handoff_load() -> DocumentKVHandoffLoad:
    request = segmented_ready_request()
    adapter_request = build_engine_adapter_request(request, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(
        adapter_request,
        payload_uri="disk:/tmp/cachet-req-1.kv",
    )
    plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
    actions = build_engine_kv_connector_actions(plan, request.payload)
    return DocumentKVHandoffLoad(actions=actions, payload=request.payload)


def _handoff_load_from_ready_request(request: EngineReadyRequest) -> DocumentKVHandoffLoad:
    adapter_request = build_engine_adapter_request(request, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(adapter_request, payload_uri="disk:/tmp/cachet-req-1.kv")
    plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
    payload_view = view_engine_adapter_payload(record, request.payload)
    actions = build_engine_kv_connector_actions(plan, payload_view)
    return DocumentKVHandoffLoad(actions=actions, payload=request.payload)


def _packed_handoff_load(
    *,
    encoded: bytes,
    descriptor: RuntimeOperationDescriptor,
):
    if len(encoded) < 2:
        raise ValueError("packed provider fixture requires at least two bytes")
    base = handoff_load()
    first_length = len(encoded) // 2
    second_length = len(encoded) - first_length
    method = MethodSpec(
        method="cpu_packed_provider_fixture",
        display_name="CPU packed provider fixture",
        arm_id="document_kv_cache",
        connector_mode="cachet",
        pre_rope=False,
        selective_recompute=False,
        implemented=True,
        description="CPU-only provider boundary fixture.",
        generator_factory="fixture:generator",
        artifact_format=PACKED_Q4_ARTIFACT_FORMAT,
        payload_decode_stage=PayloadDecodeStage.PROVIDER,
        payload_decoder=descriptor,
    )
    method_registry = MethodRegistry().with_spec(method)
    reuse_plan = method.reuse_plan()
    first_copy, second_copy = base.actions.copies
    actions = replace(
        base.actions,
        copies=(
            replace(
                first_copy,
                source_byte_start=0,
                source_byte_length=first_length,
                global_byte_start=0,
                global_byte_end=first_length,
            ),
            replace(
                second_copy,
                source_byte_start=first_length,
                source_byte_length=second_length,
                global_byte_start=first_length,
                global_byte_end=len(encoded),
            ),
        ),
        bind=replace(
            base.actions.bind,
            cache_method=method.method_id,
            metadata={
                **base.actions.bind.metadata,
                "document_kv.total_bytes": str(len(encoded)),
                "document_kv.cache_method": method.method_id,
                "document_kv.reuse_capability_id": reuse_plan.capability_id,
            },
        ),
        reuse_plan=reuse_plan,
    )
    spec = replace(
        vllm_adapter_spec(),
        supported_artifact_encodings=(
            ArtifactEncoding.RAW_KV,
            ArtifactEncoding.PACKED_Q4,
        ),
        supported_payload_decode_stages=(
            PayloadDecodeStage.NONE,
            PayloadDecodeStage.PROVIDER,
        ),
        supported_runtime_operations=(
            RuntimeOperationSupport(
                RuntimeOperationPhase.PAYLOAD_DECODE,
                descriptor.strategy_id,
                descriptor.version,
            ),
        ),
    )
    return (
        DocumentKVHandoffLoad(
            actions=actions,
            payload=encoded,
            method_registry=method_registry,
        ),
        spec,
        method_registry,
    )


class StaticHandoffSource:
    def __init__(self, load: DocumentKVHandoffLoad | None) -> None:
        self.load = load
        self.requests: list[str] = []

    def get_load(self, request):
        self.requests.append(request.request_id)
        return self.load


class RequestHandoffSource:
    def __init__(self, loads_by_request_id: dict[str, DocumentKVHandoffLoad]) -> None:
        self.loads_by_request_id = dict(loads_by_request_id)

    def get_load(self, request):
        return self.loads_by_request_id.get(request.request_id)


class AllocatedBlocks:
    def __init__(self, block_ids: list[int]) -> None:
        self.block_ids = block_ids

    def get_block_ids(self):
        return (self.block_ids,)


class TwoTokenBenchmarkGenerator:
    payload = bytes((1, 2, 3, 4, 11, 12, 13, 14, 5, 6, 7, 8, 15, 16, 17, 18))

    def generate(self, *, document, chunk, config, training_artifacts=None):
        del training_artifacts
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
                content_hash=hashlib.sha256(self.payload).hexdigest(),
                artifact_identity=config.artifact_identity_for(layout()),
                token_contract=TokenContract.from_token_ids(
                    (1, 2),
                    tokenizer_id=config.tokenizer_id,
                    tokenizer_revision=config.tokenizer_revision,
                    add_special_tokens=False,
                    prompt_template_version=config.prompt_template_version,
                ),
            ),
            payload=self.payload,
            token_count=2,
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


class PackedPipelineGenerator:
    """CPU-only encoded-artifact fixture with an authenticated identity."""

    pre_rope = False
    position_handling = PositionHandling.STORED_POST_ROPE

    _PAYLOADS = {
        "static": (b"pack", (1, 2)),
        "chunk-a": (b"d", (3,)),
    }

    def generate(self, *, document, chunk, config, training_artifacts=None):
        del training_artifacts
        encoded, token_ids = self._PAYLOADS[chunk.chunk_id]
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
                content_hash=hashlib.sha256(encoded).hexdigest(),
                artifact_identity=config.artifact_identity_for(layout()),
                token_contract=TokenContract.from_token_ids(
                    token_ids,
                    tokenizer_id=config.tokenizer_id,
                    tokenizer_revision=config.tokenizer_revision,
                    add_special_tokens=False,
                    prompt_template_version=config.prompt_template_version,
                ),
            ),
            payload=encoded,
            token_count=len(token_ids),
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


def benchmark_jsonl_record() -> dict[str, object]:
    return {
        "dataset": "biography",
        "example_id": "bio-provider-1",
        "query": "Who wrote notes?",
        "expected_answer": "Ada Lovelace",
        "documents": [
            {
                "document_id": "ada",
                "title": "Ada",
                "text": "Ada Lovelace wrote notes on the Analytical Engine.",
            }
        ],
    }


def scheduler_output(block_ids: list[int], *, request_id: str = "req-1"):
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=request_id, block_ids=(block_ids,))],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
    )


def logical_kv_slot(layer, block_id: int, block_offset: int):
    """Expose one packed vLLM 0.27.1 slot as logical [K/V, H, D]."""

    head_dim = layer.shape[-1] // 2
    return torch.stack(
        (
            layer[block_id, :, block_offset, :head_dim],
            layer[block_id, :, block_offset, head_dim:],
        ),
        dim=0,
    )


def hydrate_two_token_load(
    provider: DocumentKVNativeProvider,
) -> tuple[DocumentKVConnector, object, object]:
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})
    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([4, 6]), 2)
    meta = connector.build_connector_meta(scheduler_output([4, 6]))
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    connector.register_kv_caches(
        {
            "model.layers.0.self_attn.attn": layer_0,
            "model.layers.1.self_attn.attn": layer_1,
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())
    return connector, layer_0, layer_1


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
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

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


def test_native_provider_rejects_planned_method_before_payload_io(monkeypatch):
    ready = ready_request()
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(adapter_request)
    assert adapter_request.reuse_plan is not None
    forged_plan = replace(adapter_request.reuse_plan, method_id="kv_packet")
    record["reuse_plan"] = forged_plan.to_record()
    record["handle"]["cache_method"] = "kv_packet"
    record["metadata"]["document_kv.cache_method"] = "kv_packet"
    record["metadata"][
        "document_kv.reuse_capability_id"
    ] = forged_plan.capability_id
    payload_reads = []

    def unexpected_payload_read(*args, **kwargs):
        payload_reads.append((args, kwargs))
        raise AssertionError("payload must not be read for a planned method")

    monkeypatch.setattr(
        vllm_native_provider,
        "read_engine_adapter_payload",
        unexpected_payload_read,
    )
    provider = DocumentKVNativeProvider()
    request = SimpleNamespace(
        request_id="req-1",
        num_tokens=5,
        kv_transfer_params={
            DOCUMENT_KV_HANDOFF_RECORD_PARAM: record,
            DOCUMENT_KV_PAYLOAD_URI_PARAM: "disk:/not-read/planned.kv",
        },
    )

    with pytest.raises(ValueError, match="not a runnable registered Cachet method"):
        provider.get_num_new_matched_tokens(request, 0)
    assert payload_reads == []


def test_native_provider_does_not_rematch_request_with_pending_allocation():
    source = StaticHandoffSource(handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert provider.get_num_new_matched_tokens(request, 0) == (2, False)
    provider.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)

    assert provider.get_num_new_matched_tokens(request, 0) == (0, False)
    assert source.requests == ["req-1"]


def test_native_provider_does_not_rematch_active_external_request_until_release():
    source = StaticHandoffSource(handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert provider.get_num_new_matched_tokens(request, 0) == (2, False)
    provider.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    provider.build_connector_meta(scheduler_output([5, 7]))

    assert provider.get_num_new_matched_tokens(request, 0) == (0, False)
    provider.request_finished(request, [5, 7])
    assert provider.get_num_new_matched_tokens(request, 0) == (2, False)


def test_native_provider_records_cached_request_allocation_metadata():
    source = StaticHandoffSource(handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

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
    request = SimpleNamespace(request_id="req-1", num_tokens=7, kv_transfer_params={})

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
    request = SimpleNamespace(request_id="req-1", num_tokens=7, kv_transfer_params={})

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
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

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
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

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
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert provider.get_num_new_matched_tokens(request, 1) == (0, False)


def test_native_provider_rejects_suffix_only_prompt_length():
    source = StaticHandoffSource(handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(request_id="req-1", num_tokens=3, kv_transfer_params={})

    with pytest.raises(ValueError, match="full logical prompt"):
        provider.get_num_new_matched_tokens(request, 0)


def test_native_provider_rejects_runtime_prompt_mode_fail_closed():
    # vLLM's scheduler can only accept external KV for a prefix visible in the request.
    # A suffix-only runtime prompt cannot preserve that positional/token contract.
    source = StaticHandoffSource(handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(
        request_id="req-1",
        num_tokens=1,
        kv_transfer_params={DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM: "runtime"},
    )

    with pytest.raises(ValueError, match="runtime.*unsupported.*full logical prompt"):
        provider.get_num_new_matched_tokens(request, 0)


def test_native_provider_matches_aligned_external_prefix_for_logical_prompt_mode():
    source = StaticHandoffSource(handoff_load())
    provider = DocumentKVNativeProvider(source=source)
    request = SimpleNamespace(
        request_id="req-1",
        num_tokens=5,
        kv_transfer_params={DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM: "logical"},
    )

    matched, load_async = provider.get_num_new_matched_tokens(request, 0)
    assert (matched, load_async) == (2, False)
    assert matched < request.num_tokens


def test_native_provider_copies_materialized_payload_into_registered_paged_kv_layers():
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)

    connector.register_kv_caches({"layer.0": layer_0, "layer.1": layer_1})
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert torch.equal(logical_kv_slot(layer_0, 5, 0), torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_0, 5, 1), torch.tensor([[[5, 6]], [[7, 8]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_0, 7, 0), torch.zeros((2, 1, 2), dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 5, 0), torch.tensor([[[11, 12]], [[13, 14]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 5, 1), torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 7, 0), torch.zeros((2, 1, 2), dtype=torch.int8))
    assert connector.get_kv_connector_stats()["document_kv_layers_loaded"] == 2
    assert connector.take_events() == [{"event": "document_kv_loaded", "request_id": "req-1"}]


def test_native_provider_decodes_packed_payload_before_tensor_view_and_copy():
    encoded = b"pkd-byts"
    decoded = payload()
    descriptor = RuntimeOperationDescriptor(
        strategy_id="cpu-toy.vllm-decoder",
        version="1",
        config_digest=runtime_operation_config_digest({"codec": "fixture"}),
    )
    load, spec, method_registry = _packed_handoff_load(
        encoded=encoded,
        descriptor=descriptor,
    )
    calls = []

    def decode(request):
        calls.append(request)
        assert request.payload == encoded
        return RuntimeOperationResult(payload=decoded)

    handlers = RuntimeOperationHandlerRegistry().with_handler(
        RuntimeOperationPhase.PAYLOAD_DECODE,
        descriptor,
        decode,
    )
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(load),
        adapter_spec=spec,
        operation_handlers=handlers,
        method_registry=method_registry,
    )
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    connector.register_kv_caches({"layer.0": layer_0, "layer.1": layer_1})
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert len(calls) == 1
    assert torch.equal(
        logical_kv_slot(layer_0, 5, 0),
        torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8),
    )
    assert torch.equal(
        logical_kv_slot(layer_1, 5, 1),
        torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8),
    )


def test_vanilla_two_segment_handoff_applies_global_positions_and_preserves_values():
    rope_theta = 5_000_000.0
    pre_rope_layout = KVLayout(
        model_id="tiny-pre-rope-model",
        lora_id="base",
        layout_version="pre-rope-v2",
        dtype="float32",
        num_layers=1,
        block_size=2,
        bytes_per_token=32,
        num_query_heads=1,
        num_kv_heads=1,
        head_size=4,
        kv_stride_bytes=16,
        shares_kv_storage=False,
        storage_layout="separate_key_value",
        pre_rope=True,
        rope_theta=rope_theta,
        rope_rotary_dim=4,
        key_position_encoding="pre_rope",
    )
    pre_rope_keys = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [
            [[11.0, 12.0, 13.0, 14.0]],
            [[21.0, 22.0, 23.0, 24.0]],
            [[31.0, 32.0, 33.0, 34.0]],
            [[41.0, 42.0, 43.0, 44.0]],
        ],
        dtype=torch.float32,
    )
    token_major = torch.stack((pre_rope_keys, values), dim=1).unsqueeze(1)
    payload_bytes = bytes(
        token_major.contiguous().view(torch.uint8).flatten().tolist()
    )
    pre_rope_handle = KVCacheHandle(
        request_id="req-vanilla",
        handle_uri="document-kv://req-vanilla",
        layout=pre_rope_layout,
        segments=(
            KVSegment("doc-a", "document_chunk", "doc-a", 0, 2, 0, 64),
            KVSegment("doc-b", "document_chunk", "doc-b", 2, 2, 64, 64),
        ),
        total_tokens=4,
        total_bytes=len(payload_bytes),
        cache_method="vanilla_prefill",
        payload_checksum=hashlib.sha256(payload_bytes).hexdigest(),
    )
    ready = EngineReadyRequest(
        handle=pre_rope_handle,
        payload=payload_bytes,
        estimated_gpu_bytes=len(payload_bytes),
        reuse_plan=method_spec(CacheGenerationMethod.VANILLA_PREFILL).reuse_plan(),
    )
    load = _handoff_load_from_ready_request(ready)
    assert len(load.actions.copies) == 2
    assert [copy.token_start for copy in load.actions.copies] == [0, 2]
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(load))
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(
        request_id="req-vanilla",
        num_tokens=6,
        kv_transfer_params={},
    )

    assert connector.get_num_new_matched_tokens(request, 0) == (4, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 4)
    metadata = connector.build_connector_meta(
        scheduler_output([5, 7], request_id="req-vanilla")
    )
    destination = torch.zeros((8, 1, 2, 8), dtype=torch.float32)
    connector.register_kv_caches({"layer.0": destination})
    connector.bind_connector_metadata(metadata)
    connector.start_load_kv(SimpleNamespace())

    loaded_keys = torch.stack(
        (
            logical_kv_slot(destination, 5, 0)[0],
            logical_kv_slot(destination, 5, 1)[0],
            logical_kv_slot(destination, 7, 0)[0],
            logical_kv_slot(destination, 7, 1)[0],
        )
    )
    loaded_values = torch.stack(
        (
            logical_kv_slot(destination, 5, 0)[1],
            logical_kv_slot(destination, 5, 1)[1],
            logical_kv_slot(destination, 7, 0)[1],
            logical_kv_slot(destination, 7, 1)[1],
        )
    )
    expected_keys = apply_rope_to_keys(
        pre_rope_keys,
        torch.arange(4),
        rope_theta=rope_theta,
        rotary_dim=4,
    )
    local_document_two_keys = apply_rope_to_keys(
        pre_rope_keys[2:],
        torch.arange(2),
        rope_theta=rope_theta,
        rotary_dim=4,
    )
    assert torch.allclose(loaded_keys, expected_keys)
    assert not torch.allclose(loaded_keys[2:], local_document_two_keys)
    assert torch.equal(loaded_values, values)


def test_strict_packed_artifact_pipeline_decodes_stored_bytes_before_cpu_copy(
    tmp_path,
):
    descriptor = RuntimeOperationDescriptor(
        strategy_id="cpu-toy.pipeline-decoder",
        version="1",
        config_digest=runtime_operation_config_digest({"codec": "fixture"}),
    )
    method = MethodSpec(
        method="cpu_packed_fixture",
        display_name="CPU packed fixture",
        arm_id="document_kv_cache",
        connector_mode="cachet",
        pre_rope=False,
        selective_recompute=False,
        implemented=True,
        description="CPU-only encoded artifact regression fixture.",
        generator_factory="fixture:generator",
        artifact_format=PACKED_Q4_ARTIFACT_FORMAT,
        payload_decode_stage=PayloadDecodeStage.PROVIDER,
        payload_decoder=descriptor,
    )
    method_registry = MethodRegistry().with_spec(method)
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(
            cache=ChunkCache(cpu_max_bytes=4096),
            reader=DiskRangeReader(),
        ),
        method_registry=method_registry,
    )
    config = CacheBuildConfig(
        model_id=layout().model_id,
        lora_id=layout().lora_id,
        prompt_template_version="v1",
        dtype="uint8",
        runtime_kv_dtype=layout().dtype,
        layout_version=layout().layout_version,
        cache_method=method.method_id,
        artifact_format_id=method.artifact_format.format_id,
        artifact_format_version=method.artifact_format.version,
    )
    document = SourceDocument.from_texts(
        document_id="doc-a",
        static_text="two tokens",
        chunks={"chunk-a": "one"},
    )
    generated = workflow.generate_cache(
        documents=(document,),
        generator=PackedPipelineGenerator(),
        config=config,
        shard_uri=tmp_path / "packed.kvpack",
        align_bytes=1,
    )
    assert generated.artifact_identity is not None
    request = DocumentKVRequest(
        request_id="req-1",
        task_id="cpu-packed",
        model_id=layout().model_id,
        lora_id=layout().lora_id,
        prompt_template_version="v1",
        document_chunks={"doc-a": ("chunk-a",)},
        artifact_identity=generated.artifact_identity,
    )
    ready = workflow.prepare_for_engine(request, layout=layout())
    assert ready.payload == b"packd"
    assert ready.handle.total_bytes == 5
    assert ready.handle.total_tokens == 3
    assert ready.estimated_gpu_bytes == len(payload())
    assert ready.handle.artifact_identity is not None
    assert ready.handle.artifact_identity.kv_dtype == "uint8"
    assert ready.handle.artifact_identity.runtime_kv_dtype == "int8"

    spec = replace(
        vllm_adapter_spec(),
        supported_artifact_encodings=(
            ArtifactEncoding.RAW_KV,
            ArtifactEncoding.PACKED_Q4,
        ),
        supported_payload_decode_stages=(
            PayloadDecodeStage.NONE,
            PayloadDecodeStage.PROVIDER,
        ),
        supported_runtime_operations=(
            RuntimeOperationSupport(
                RuntimeOperationPhase.PAYLOAD_DECODE,
                descriptor.strategy_id,
                descriptor.version,
            ),
        ),
    )
    decode_calls = []

    def decode(operation_request):
        decode_calls.append(operation_request)
        assert operation_request.payload == b"packd"
        return RuntimeOperationResult(payload=payload())

    handlers = RuntimeOperationHandlerRegistry().with_handler(
        RuntimeOperationPhase.PAYLOAD_DECODE,
        descriptor,
        decode,
    )
    adapter_request = build_engine_adapter_request(
        ready,
        spec=spec,
        operation_handlers=handlers,
        method_registry=method_registry,
    )
    record = engine_adapter_request_to_record(
        adapter_request,
        adapter_spec=spec,
        operation_handlers=handlers,
        method_registry=method_registry,
    )
    plan = build_engine_kv_injection_plan(
        record,
        require_external_payload_uri=False,
        adapter_spec=spec,
        operation_handlers=handlers,
        method_registry=method_registry,
    )
    stored_payload = view_engine_adapter_payload(
        record,
        ready.payload,
        adapter_spec=spec,
        operation_handlers=handlers,
        method_registry=method_registry,
    )
    actions = build_engine_kv_connector_actions(
        plan,
        stored_payload,
        method_registry=method_registry,
    )
    load = DocumentKVHandoffLoad(
        actions=actions,
        payload=ready.payload,
        method_registry=method_registry,
    )
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(load),
        adapter_spec=spec,
        operation_handlers=handlers,
        method_registry=method_registry,
    )
    connector = DocumentKVConnector(provider=provider)
    runtime_request = SimpleNamespace(
        request_id="req-1",
        num_tokens=5,
        prompt_token_ids=(1, 2, 3, 9, 9),
        kv_transfer_params={},
    )

    assert connector.get_num_new_matched_tokens(runtime_request, 0) == (2, False)
    connector.update_state_after_alloc(
        runtime_request,
        AllocatedBlocks([5, 7]),
        2,
    )
    metadata = connector.build_connector_meta(scheduler_output([5, 7]))
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    connector.register_kv_caches({"layer.0": layer_0, "layer.1": layer_1})
    connector.bind_connector_metadata(metadata)
    connector.start_load_kv(SimpleNamespace())

    assert len(decode_calls) == 1
    assert torch.equal(
        logical_kv_slot(layer_0, 5, 0),
        torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8),
    )
    assert torch.equal(
        logical_kv_slot(layer_1, 5, 1),
        torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8),
    )


def test_strict_raw_generation_rejects_distinct_persisted_and_runtime_dtypes(
    tmp_path,
):
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(
            cache=ChunkCache(cpu_max_bytes=4096),
            reader=DiskRangeReader(),
        ),
    )
    shard_path = tmp_path / "invalid-raw.kvpack"

    with pytest.raises(
        ValueError,
        match="raw-KV generation requires persisted dtype",
    ):
        workflow.generate_cache(
            documents=(
                SourceDocument.from_texts(
                    document_id="doc-a",
                    static_text="two tokens",
                    chunks={"chunk-a": "one"},
                ),
            ),
            generator=PackedPipelineGenerator(),
                config=CacheBuildConfig(
                    model_id=layout().model_id,
                    lora_id=layout().lora_id,
                    prompt_template_version="v1",
                    dtype="uint8",
                    runtime_kv_dtype="int8",
                    layout_version=layout().layout_version,
                    cache_method="full_prefix_prefill",
                ),
            shard_uri=shard_path,
            align_bytes=1,
        )

    assert not shard_path.exists()


def test_packed_cache_state_attestation_compares_physical_stored_bytes():
    descriptor = RuntimeOperationDescriptor(
        strategy_id="cpu-toy.telemetry-decoder",
        version="1",
        config_digest=runtime_operation_config_digest({"codec": "fixture"}),
    )
    handoff, spec, method_registry = _packed_handoff_load(
        encoded=b"pkd-byts",
        descriptor=descriptor,
    )
    handlers = RuntimeOperationHandlerRegistry().with_handler(
        RuntimeOperationPhase.PAYLOAD_DECODE,
        descriptor,
        lambda request: RuntimeOperationResult(payload=payload()),
    )
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(handoff),
        adapter_spec=spec,
        operation_handlers=handlers,
        method_registry=method_registry,
    )
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})
    assert provider.get_num_new_matched_tokens(request, 0) == (2, False)
    provider.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    load = provider.build_connector_meta(scheduler_output([5, 7])).loads[0]
    observation = {
        "source": "local_path",
        "bytes_read": len(b"pkd-byts"),
        "payload_cache_hit": False,
        "eviction_requested": True,
        "eviction_succeeded": True,
        "direct_io": False,
    }

    attestation = vllm_native_provider._cache_state_attestation_record(
        load,
        cache_state_observation=observation,
        successful=True,
        decoded_runtime_bytes=len(payload()),
    )

    assert attestation["expected_bytes"] == len(b"pkd-byts")
    assert attestation["expected_stored_bytes"] == len(b"pkd-byts")
    assert attestation["expected_runtime_bytes"] == len(payload())
    assert attestation["decoded_runtime_bytes"] == len(payload())
    assert attestation["cold_read_attested"] is True
    assert vllm_native_provider._cache_state_attestation_record(
        load,
        cache_state_observation={
            **observation,
            "bytes_read": len(payload()),
        },
        successful=True,
        decoded_runtime_bytes=len(payload()),
    )["cold_read_attested"] is False


def test_native_provider_rejects_decoder_config_digest_mismatch_before_load():
    descriptor = RuntimeOperationDescriptor(
        strategy_id="cpu-toy.vllm-decoder",
        version="1",
        config_digest=runtime_operation_config_digest({"codec": "expected"}),
    )
    load, spec, method_registry = _packed_handoff_load(
        encoded=b"pkd-byts",
        descriptor=descriptor,
    )
    wrong_descriptor = replace(
        descriptor,
        config_digest=runtime_operation_config_digest({"codec": "wrong"}),
    )
    handlers = RuntimeOperationHandlerRegistry().with_handler(
        RuntimeOperationPhase.PAYLOAD_DECODE,
        wrong_descriptor,
        lambda request: RuntimeOperationResult(payload=request.payload),
    )
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(load),
        adapter_spec=spec,
        operation_handlers=handlers,
        method_registry=method_registry,
    )
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    with pytest.raises(ValueError, match="configuration digest does not match"):
        provider.get_num_new_matched_tokens(request, 0)


def test_native_provider_loads_uri_payload_via_mmap_matches_inline(
    tmp_path,
    monkeypatch,
):
    # The no-cache cold-hydrate path memory-maps the payload file (lazily) instead of
    # reading it into a bytes object. Loading from a file URI must inject exactly the
    # same KV values as the inline-bytes path.
    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    inline = handoff_load()
    uri_load = DocumentKVHandoffLoad(actions=inline.actions, payload_uri=str(payload_path))
    telemetry_path = tmp_path / "telemetry.jsonl"
    observed_memoryview: list[bool] = []
    original_payload_tensor_view = vllm_native_provider._payload_tensor_view

    def inspect_payload_tensor_view(raw_payload, load):
        observed_memoryview.append(isinstance(raw_payload, memoryview))
        return original_payload_tensor_view(raw_payload, load)

    monkeypatch.setattr(
        vllm_native_provider,
        "_payload_tensor_view",
        inspect_payload_tensor_view,
    )

    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(uri_load),
        telemetry_jsonl=str(telemetry_path),
    )
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)

    connector.register_kv_caches({"layer.0": layer_0, "layer.1": layer_1})
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert torch.equal(logical_kv_slot(layer_0, 5, 0), torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_0, 5, 1), torch.tensor([[[5, 6]], [[7, 8]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 5, 0), torch.tensor([[[11, 12]], [[13, 14]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 5, 1), torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8))
    assert connector.get_kv_connector_stats()["document_kv_layers_loaded"] == 2
    assert observed_memoryview == [True]


def test_native_provider_evicts_page_cache_before_mmap_when_enabled(tmp_path, monkeypatch):
    # With DOCUMENT_KV_EVICT_PAGE_CACHE=1 the cold-hydrate path must drop the payload
    # file from the OS page cache (posix_fadvise DONTNEED) before mapping it, so the
    # host->device copy reads cold from disk. Injected KV must still be correct.
    calls: list[tuple[int, int, int, int]] = []
    sync_calls: list[int] = []

    def fake_fadvise(fd, offset, length, advice):
        calls.append((fd, offset, length, advice))

    monkeypatch.setattr(os, "posix_fadvise", fake_fadvise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(os, "sync", lambda: sync_calls.append(1), raising=False)
    monkeypatch.setenv("DOCUMENT_KV_EVICT_PAGE_CACHE", "1")

    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    inline = handoff_load()
    uri_load = DocumentKVHandoffLoad(actions=inline.actions, payload_uri=str(payload_path))
    telemetry_path = tmp_path / "cold-telemetry.jsonl"

    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(uri_load),
        telemetry_jsonl=str(telemetry_path),
    )
    assert provider._evict_page_cache is True
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    connector.register_kv_caches({"layer.0": layer_0, "layer.1": layer_1})
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert len(calls) == 1
    assert calls[0][3] == 4  # POSIX_FADV_DONTNEED
    assert sync_calls == [1]  # one-time flush before eviction so reads are fully cold
    assert torch.equal(logical_kv_slot(layer_0, 5, 0), torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert connector.get_kv_connector_stats()["document_kv_layers_loaded"] == 2
    cache_state = json.loads(telemetry_path.read_text(encoding="utf-8"))[
        "cache_state_attestation"
    ]
    assert cache_state["source"] == "local_path"
    assert cache_state["bytes_read"] == len(payload())
    assert cache_state["eviction_requested"] is True
    assert cache_state["eviction_succeeded"] is True
    assert cache_state["cold_read_attested"] is True


def test_read_payload_view_streams_full_file(tmp_path):
    # The plain buffered reader must return every byte of a multi-chunk payload
    # (the loop reads in 8 MiB chunks, so exercise more than one chunk boundary).
    data = bytes((i * 7 + 3) % 256 for i in range(200_000))
    path = tmp_path / "payload.kv"
    path.write_bytes(data)

    view = vllm_native_provider._read_payload_view(str(path), expected_bytes=len(data))
    assert bytes(view) == data


def test_read_payload_view_raises_on_size_mismatch(tmp_path):
    path = tmp_path / "payload.kv"
    path.write_bytes(b"abcd")
    with pytest.raises(ValueError, match="!= expected"):
        vllm_native_provider._read_payload_view(str(path), expected_bytes=8)


def test_read_payload_view_evicts_page_cache_when_requested(tmp_path, monkeypatch):
    # evict_page_cache drops the file from the OS page cache before the buffered read
    # so the read streams cold from disk (honest cold-hydrate measurement).
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(os, "posix_fadvise", lambda *a: calls.append(a), raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)

    path = tmp_path / "payload.kv"
    path.write_bytes(b"hello cachet")
    view = vllm_native_provider._read_payload_view(str(path), evict_page_cache=True)

    assert bytes(view) == b"hello cachet"
    assert len(calls) == 1
    assert calls[0][3] == 4  # POSIX_FADV_DONTNEED


def test_advise_sequential_readahead_issues_madvise_hints():
    # The cold-read path should hint the kernel to read the mapping ahead
    # sequentially so the host->device copy pulls large I/Os instead of faulting
    # page-by-page. We can't observe throughput in a unit test, but we can assert
    # the madvise hints are issued (when the platform defines them).
    advises: list[int] = []

    class _FakeMapping:
        def madvise(self, option, *args):
            advises.append(option)

    vllm_native_provider._advise_sequential_readahead(_FakeMapping())

    expected = [
        getattr(vllm_native_provider.mmap, name)
        for name in ("MADV_SEQUENTIAL", "MADV_WILLNEED")
        if getattr(vllm_native_provider.mmap, name, None) is not None
    ]
    assert advises == expected


def test_advise_sequential_readahead_tolerates_missing_madvise():
    # No madvise attribute (e.g. plain-read fallback view) must be a no-op.
    class _NoMadvise:
        pass

    vllm_native_provider._advise_sequential_readahead(_NoMadvise())


def test_native_provider_does_not_evict_page_cache_by_default(tmp_path, monkeypatch):
    calls: list[object] = []
    monkeypatch.setattr(os, "posix_fadvise", lambda *a: calls.append(a), raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.delenv("DOCUMENT_KV_EVICT_PAGE_CACHE", raising=False)

    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    inline = handoff_load()
    uri_load = DocumentKVHandoffLoad(actions=inline.actions, payload_uri=str(payload_path))

    provider = DocumentKVNativeProvider(source=StaticHandoffSource(uri_load))
    assert provider._evict_page_cache is False
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})
    connector.get_num_new_matched_tokens(request, 0)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    connector.register_kv_caches(
        {
            "layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
            "layer.1": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert calls == []


def test_native_provider_prefetches_payloads_when_workers_enabled(tmp_path, monkeypatch):
    # With DOCUMENT_KV_PREFETCH_WORKERS>0 the connector warms the payload into the
    # OS page cache from a background pool as soon as the step's loads are bound, so
    # the on-critical-path host->device copy streams from cache. When eviction is
    # also enabled the prefetch performs the (single) cold read; the critical-path
    # mmap must NOT re-evict the pages the prefetch just warmed.
    fadvise_calls: list[object] = []
    monkeypatch.setattr(os, "posix_fadvise", lambda *a: fadvise_calls.append(a), raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(os, "sync", lambda: None, raising=False)
    monkeypatch.setenv("DOCUMENT_KV_EVICT_PAGE_CACHE", "1")
    monkeypatch.setenv("DOCUMENT_KV_PREFETCH_WORKERS", "2")

    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    inline = handoff_load()
    uri_load = DocumentKVHandoffLoad(actions=inline.actions, payload_uri=str(payload_path))

    provider = DocumentKVNativeProvider(source=StaticHandoffSource(uri_load))
    assert provider._prefetch_workers == 2
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    connector.register_kv_caches({"layer.0": layer_0, "layer.1": layer_1})

    connector.bind_connector_metadata(meta)
    assert provider._stats_prefetch_submitted == 1
    # Deterministically finish the background prefetch before hydrating so the
    # non-blocking reap observes it complete (the pool is otherwise racy in tests).
    for future in list(provider._prefetch_futures.values()):
        future.result(timeout=30)
    connector.start_load_kv(SimpleNamespace())

    # Exactly one eviction (from the background prefetch), consumed future, correct KV.
    assert len(fadvise_calls) == 1
    assert provider._prefetch_futures == {}
    assert torch.equal(logical_kv_slot(layer_0, 5, 0), torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 5, 1), torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8))
    assert connector.get_kv_connector_stats()["document_kv_layers_loaded"] == 2


def test_native_provider_prefetch_caps_concurrent_reads_at_inflight_limit(monkeypatch):
    # A wave of prefetches may be submitted at once, but concurrent NVMe reads must
    # be bounded to the disk sweet spot (default 4) to avoid over-subscription; the
    # surplus queues on the executor.
    monkeypatch.setenv("DOCUMENT_KV_PREFETCH_WORKERS", "8")
    monkeypatch.delenv("DOCUMENT_KV_PREFETCH_MAX_INFLIGHT", raising=False)
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    assert provider._prefetch_max_inflight == 4
    pool = provider._ensure_prefetch_pool()
    assert pool._max_workers == 4  # min(workers=8, max_inflight=4)

    monkeypatch.setenv("DOCUMENT_KV_PREFETCH_MAX_INFLIGHT", "2")
    provider2 = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    assert provider2._ensure_prefetch_pool()._max_workers == 2


def test_native_provider_no_prefetch_pool_by_default(monkeypatch):
    monkeypatch.delenv("DOCUMENT_KV_PREFETCH_WORKERS", raising=False)
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    assert provider._prefetch_workers == 0
    provider._submit_prefetch(())
    assert provider._prefetch_pool is None
    assert provider._stats_prefetch_submitted == 0


def test_native_provider_reports_phase_timing_metrics(monkeypatch):
    ticks = iter(
        [
            100,
            200,
            210,
            300,
            320,
            400,
            430,
            500,
            540,
            600,
            650,
            700,
        ]
    )

    monkeypatch.setattr(vllm_native_provider.time, "perf_counter_ns", lambda: next(ticks))
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    connector.register_kv_caches(
        {
            "layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
            "layer.1": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    stats = connector.get_kv_connector_stats()
    assert stats["document_kv_loads_started"] == 1
    assert stats["document_kv_layers_loaded"] == 2
    assert stats["document_kv_load_error_blocks"] == 0
    assert stats["document_kv_payload_materialize_ns"] == 10
    assert stats["document_kv_payload_merge_ns"] == 20
    assert stats["document_kv_payload_view_ns"] == 30
    assert stats["document_kv_layer_load_ns"] == 90
    assert connector.get_kv_connector_stats() is None


def test_native_provider_writes_per_load_telemetry_jsonl(tmp_path):
    telemetry_path = tmp_path / "connector-telemetry.jsonl"
    original_load = handoff_load()
    linked_load = DocumentKVHandoffLoad(
        actions=replace(
            original_load.actions,
            bind=replace(
                original_load.actions.bind,
                metadata={
                    **original_load.actions.bind.metadata,
                    DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM: "benchmark-request-1",
                },
            ),
        ),
        payload=original_load.payload,
    )
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(linked_load),
        telemetry_jsonl=str(telemetry_path),
    )
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    connector.register_kv_caches(
        {
            "layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
            "layer.1": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["record_type"] == "document_kv.vllm_native_provider_load.v1"
    assert row["success"] is True
    assert row["request_id"] == "req-1"
    assert row["benchmark_request_id"] == "benchmark-request-1"
    assert row["counts"]["token_count"] == 2
    assert row["counts"]["handoff_total_tokens"] == 3
    assert row["counts"]["layers_loaded"] == 2
    assert row["counts"]["expected_payload_bytes"] == len(payload())
    assert row["layout"]["dtype"] == "int8"
    assert row["payload"]["source"] == "inline"
    assert row["cache_state_attestation"] == {
        "cache_method": "full_prefix_prefill",
        "artifact_id": "",
        "source": "inline",
        "bytes_read": 0,
        "payload_cache_hit": False,
        "eviction_requested": False,
        "eviction_succeeded": False,
        "direct_io": False,
        "expected_bytes": len(payload()),
        "expected_stored_bytes": len(payload()),
        "expected_runtime_bytes": len(payload()),
        "decoded_runtime_bytes": len(payload()),
        "expected_tokens": 2,
        "loaded_tokens": 2,
        "successful_loads": 1,
        "cold_read_attested": False,
    }


def test_native_provider_telemetry_always_records_wall_clock(tmp_path):
    telemetry_path = tmp_path / "connector-telemetry.jsonl"
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(handoff_load()),
        telemetry_jsonl=str(telemetry_path),
    )
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})
    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    connector.register_kv_caches(
        {
            "layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
            "layer.1": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    row = json.loads(telemetry_path.read_text(encoding="utf-8").splitlines()[0])
    # Wall-clock stamps are always recorded so a serialization timeline can be
    # reconstructed across concurrent loads; stage splits stay off by default.
    assert isinstance(row["wall_clock"]["start_s"], float)
    assert isinstance(row["wall_clock"]["end_s"], float)
    assert row["wall_clock"]["end_s"] >= row["wall_clock"]["start_s"]
    assert "h2d" not in row["timings_ns"]
    assert "scatter" not in row["timings_ns"]


def test_native_provider_telemetry_splits_h2d_and_scatter_when_profiling(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMENT_KV_PROFILE_STAGES", "1")
    telemetry_path = tmp_path / "connector-telemetry.jsonl"
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(handoff_load()),
        telemetry_jsonl=str(telemetry_path),
    )
    assert provider._profile_stages is True
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})
    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    connector.register_kv_caches(
        {
            "layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
            "layer.1": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    row = json.loads(telemetry_path.read_text(encoding="utf-8").splitlines()[0])
    timings = row["timings_ns"]
    assert "h2d" in timings
    assert "scatter" in timings
    assert timings["h2d"] >= 0
    assert timings["scatter"] >= 0


def test_native_provider_telemetry_labels_local_payload_paths(tmp_path):
    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    handoff = handoff_load()
    load = DocumentKVLoadRequest(
        request_id=handoff.request_id,
        actions_record=engine_kv_connector_actions_to_record(handoff.actions),
        payload=None,
        blocks=(BlockSpan(block_id=1, token_start=0, token_count=2, block_offset=0),),
        source_token_start=0,
        token_count=2,
        payload_uri=str(payload_path),
    )

    record = vllm_native_provider._load_telemetry_record(
        load,
        provider_factory=DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
        payload_cache_enabled=False,
        total_ns=1,
        payload_materialize_ns=1,
        payload_merge_ns=0,
        payload_view_ns=0,
        layer_load_ns=0,
        layers_loaded=0,
        payload_cache_hits=0,
        payload_cache_misses=0,
        error_type=None,
        error_message=None,
    )

    assert record["payload"]["source"] == "uri"
    assert record["payload"]["uri_scheme"] == "local_path"
    assert record["payload"]["uri_tail"] == "req-1.kv"


def test_native_provider_reuses_payload_uri_cache_for_hot_documents(tmp_path, monkeypatch):
    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    payload_uri = f"disk:{payload_path}"
    read_calls = []
    original_read_payload = vllm_native_provider.read_engine_adapter_payload

    def counting_read_payload(uri, *, expected_bytes=None):
        read_calls.append((uri, expected_bytes))
        return original_read_payload(uri, expected_bytes=expected_bytes)

    monkeypatch.setattr(vllm_native_provider, "read_engine_adapter_payload", counting_read_payload)
    load = DocumentKVHandoffLoad(actions=handoff_load_with_content_hashes().actions, payload_uri=payload_uri)
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(load),
        payload_cache_max_bytes=len(payload()) * 2,
    )
    connector = DocumentKVConnector(provider=provider)
    connector.register_kv_caches(
        {
            "layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
            "layer.1": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
        }
    )

    for request_id, block_ids in (("req-1", [4, 6]), ("req-2", [5, 7])):
        request = SimpleNamespace(request_id=request_id, num_tokens=5, kv_transfer_params={})
        assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
        connector.update_state_after_alloc(request, AllocatedBlocks(block_ids), 2)
        meta = connector.build_connector_meta(scheduler_output(block_ids, request_id=request_id))
        connector.bind_connector_metadata(meta)
        connector.start_load_kv(SimpleNamespace())

    stats = connector.get_kv_connector_stats()
    assert read_calls == [(payload_uri, len(payload()))]
    assert stats["document_kv_loads_started"] == 2
    assert stats["document_kv_layers_loaded"] == 4
    assert stats["document_kv_payload_cache_misses"] == 1
    assert stats["document_kv_payload_cache_hits"] == 1
    assert connector.get_kv_connector_stats() is None


def test_native_provider_payload_uri_cache_skips_oversized_payloads(tmp_path, monkeypatch):
    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    payload_uri = f"disk:{payload_path}"
    read_calls = []
    original_read_payload = vllm_native_provider.read_engine_adapter_payload

    def counting_read_payload(uri, *, expected_bytes=None):
        read_calls.append((uri, expected_bytes))
        return original_read_payload(uri, expected_bytes=expected_bytes)

    monkeypatch.setattr(vllm_native_provider, "read_engine_adapter_payload", counting_read_payload)
    load = DocumentKVHandoffLoad(actions=handoff_load_with_content_hashes().actions, payload_uri=payload_uri)
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(load), payload_cache_max_bytes=1)
    connector = DocumentKVConnector(provider=provider)
    connector.register_kv_caches({"layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8)})

    for request_id, block_ids in (("req-1", [4, 6]), ("req-2", [5, 7])):
        request = SimpleNamespace(request_id=request_id, num_tokens=5, kv_transfer_params={})
        assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
        connector.update_state_after_alloc(request, AllocatedBlocks(block_ids), 2)
        meta = connector.build_connector_meta(scheduler_output(block_ids, request_id=request_id))
        connector.bind_connector_metadata(meta)
        connector.start_load_kv(SimpleNamespace())

    stats = connector.get_kv_connector_stats()
    assert read_calls == [(payload_uri, len(payload())), (payload_uri, len(payload()))]
    assert stats["document_kv_payload_cache_misses"] == 2
    assert stats["document_kv_payload_cache_hits"] == 0


def test_native_provider_payload_uri_cache_rejects_unhashed_copy_actions(tmp_path):
    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    payload_uri = f"disk:{payload_path}"
    load = DocumentKVHandoffLoad(actions=handoff_load().actions, payload_uri=payload_uri)
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(load),
        payload_cache_max_bytes=len(payload()) * 2,
    )
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([4, 6]), 2)
    connector.bind_connector_metadata(connector.build_connector_meta(scheduler_output([4, 6])))
    connector.register_kv_caches({"layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8)})
    with pytest.raises(ValueError, match="content_hash"):
        connector.start_load_kv(SimpleNamespace())


def test_native_provider_payload_uri_cache_misses_when_content_identity_changes(tmp_path, monkeypatch):
    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    payload_uri = f"disk:{payload_path}"
    read_calls = []
    original_read_payload = vllm_native_provider.read_engine_adapter_payload

    def counting_read_payload(uri, *, expected_bytes=None):
        read_calls.append((uri, expected_bytes))
        return original_read_payload(uri, expected_bytes=expected_bytes)

    monkeypatch.setattr(vllm_native_provider, "read_engine_adapter_payload", counting_read_payload)
    load_1 = DocumentKVHandoffLoad(
        actions=handoff_load_with_content_hashes(("old-static", "old-chunk")).actions,
        payload_uri=payload_uri,
    )
    load_2 = DocumentKVHandoffLoad(
        actions=handoff_load_with_content_hashes(("new-static", "new-chunk")).actions,
        payload_uri=payload_uri,
    )
    provider = DocumentKVNativeProvider(
        source=RequestHandoffSource({"req-1": load_1, "req-2": load_2}),
        payload_cache_max_bytes=len(payload()) * 2,
    )
    connector = DocumentKVConnector(provider=provider)
    connector.register_kv_caches({"layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8)})

    for request_id, block_ids in (("req-1", [4, 6]), ("req-2", [5, 7])):
        request = SimpleNamespace(request_id=request_id, num_tokens=5, kv_transfer_params={})
        assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
        connector.update_state_after_alloc(request, AllocatedBlocks(block_ids), 2)
        meta = connector.build_connector_meta(scheduler_output(block_ids, request_id=request_id))
        connector.bind_connector_metadata(meta)
        if request_id == "req-2":
            payload_path.write_bytes(bytes(reversed(payload())))
        connector.start_load_kv(SimpleNamespace())

    stats = connector.get_kv_connector_stats()
    assert read_calls == [(payload_uri, len(payload())), (payload_uri, len(payload()))]
    assert stats["document_kv_payload_cache_misses"] == 2
    assert stats["document_kv_payload_cache_hits"] == 0


def test_native_provider_reset_cache_clears_payload_uri_cache(tmp_path, monkeypatch):
    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    payload_uri = f"disk:{payload_path}"
    read_calls = []
    original_read_payload = vllm_native_provider.read_engine_adapter_payload

    def counting_read_payload(uri, *, expected_bytes=None):
        read_calls.append((uri, expected_bytes))
        return original_read_payload(uri, expected_bytes=expected_bytes)

    monkeypatch.setattr(vllm_native_provider, "read_engine_adapter_payload", counting_read_payload)
    load = DocumentKVHandoffLoad(actions=handoff_load_with_content_hashes().actions, payload_uri=payload_uri)
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(load),
        payload_cache_max_bytes=len(payload()) * 2,
    )
    connector = DocumentKVConnector(provider=provider)
    connector.register_kv_caches({"layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8)})

    for request_id, block_ids in (("req-1", [4, 6]), ("req-2", [5, 7])):
        request = SimpleNamespace(request_id=request_id, num_tokens=5, kv_transfer_params={})
        assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
        connector.update_state_after_alloc(request, AllocatedBlocks(block_ids), 2)
        meta = connector.build_connector_meta(scheduler_output(block_ids, request_id=request_id))
        connector.bind_connector_metadata(meta)
        if request_id == "req-2":
            assert connector.reset_cache() is True
        connector.start_load_kv(SimpleNamespace())

    stats = connector.get_kv_connector_stats()
    assert read_calls == [(payload_uri, len(payload())), (payload_uri, len(payload()))]
    assert stats["document_kv_payload_cache_misses"] == 2
    assert stats["document_kv_payload_cache_hits"] == 0


def test_native_provider_reuses_one_payload_tensor_view_per_load(monkeypatch):
    calls = []
    original_frombuffer = torch.frombuffer

    def counting_frombuffer(*args, **kwargs):
        calls.append((args, kwargs))
        return original_frombuffer(*args, **kwargs)

    monkeypatch.setattr(torch, "frombuffer", counting_frombuffer)
    load = handoff_load()
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(load))
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    connector.register_kv_caches(
        {
            "layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
            "layer.1": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert len(calls) == 1
    assert calls[0][0][0] is load.payload
    assert isinstance(calls[0][0][0], bytes)


def test_payload_tensor_view_rejects_layer_major_payload():
    # The provider read path reshapes payloads as token-major, so a layer-major
    # payload must fail loudly here instead of being silently misread and
    # corrupting GPU KV.
    layer_major_layout = KVLayout(
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
        payload_axis_order="layer_major",
    )
    load = SimpleNamespace(
        actions=SimpleNamespace(
            reservation=SimpleNamespace(layout=layer_major_layout, total_tokens=3)
        )
    )

    with pytest.raises(ValueError, match="token-major"):
        vllm_native_provider._payload_tensor_view(payload(), load)


def test_native_provider_reuses_slot_mapping_for_layers_on_same_device(monkeypatch):
    calls = []
    original_slot_mapping_from_blocks = vllm_native_provider.slot_mapping_from_blocks

    def counting_slot_mapping_from_blocks(*args, **kwargs):
        calls.append((args, kwargs))
        return original_slot_mapping_from_blocks(*args, **kwargs)

    monkeypatch.setattr(vllm_native_provider, "slot_mapping_from_blocks", counting_slot_mapping_from_blocks)
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    connector.register_kv_caches({"layer.0": layer_0, "layer.1": layer_1})
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert len(calls) == 1
    assert calls[0][1]["device"] == layer_0.device
    assert calls[0][1]["block_size"] == layout().block_size


def test_native_provider_consumes_bound_load_metadata_after_successful_load():
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    connector.register_kv_caches(
        {
            "layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
            "layer.1": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())
    connector.start_load_kv(SimpleNamespace())

    stats = connector.get_kv_connector_stats()
    assert stats["document_kv_loads_started"] == 1
    assert stats["document_kv_layers_loaded"] == 2
    assert connector.get_kv_connector_stats() is None
    assert connector.take_events() == [{"event": "document_kv_loaded", "request_id": "req-1"}]


def test_native_provider_skips_rebound_duplicate_load_metadata_after_successful_load():
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    connector.register_kv_caches(
        {
            "layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
            "layer.1": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
        }
    )

    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())
    connector.clear_connector_metadata()
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    stats = connector.get_kv_connector_stats()
    assert stats["document_kv_loads_started"] == 1
    assert stats["document_kv_layers_loaded"] == 2
    assert connector.get_kv_connector_stats() is None
    assert connector.take_events() == [{"event": "document_kv_loaded", "request_id": "req-1"}]


def test_native_provider_skips_duplicate_load_without_dropping_new_load_in_same_metadata():
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    connector = DocumentKVConnector(provider=provider)
    req_1 = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})
    req_2 = SimpleNamespace(request_id="req-2", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(req_1, 0) == (2, False)
    connector.update_state_after_alloc(req_1, AllocatedBlocks([5, 7]), 2)
    meta_1 = connector.build_connector_meta(scheduler_output([5, 7], request_id="req-1"))
    assert connector.get_num_new_matched_tokens(req_2, 0) == (2, False)
    connector.update_state_after_alloc(req_2, AllocatedBlocks([6, 8]), 2)
    meta_2 = connector.build_connector_meta(scheduler_output([6, 8], request_id="req-2"))
    connector.register_kv_caches(
        {
            "layer.0": torch.zeros((9, 1, 2, 4), dtype=torch.int8),
            "layer.1": torch.zeros((9, 1, 2, 4), dtype=torch.int8),
        }
    )

    connector.bind_connector_metadata(meta_1)
    connector.start_load_kv(SimpleNamespace())
    assert connector.get_kv_connector_stats()["document_kv_loads_started"] == 1
    assert connector.take_events() == [{"event": "document_kv_loaded", "request_id": "req-1"}]
    mixed_meta = DocumentKVConnectorMetadata(loads=(meta_1.loads[0], meta_2.loads[0]))
    connector.bind_connector_metadata(mixed_meta)
    connector.start_load_kv(SimpleNamespace())

    stats = connector.get_kv_connector_stats()
    assert stats["document_kv_loads_started"] == 1
    assert stats["document_kv_layers_loaded"] == 2
    assert connector.get_kv_connector_stats() is None
    assert connector.take_events() == [{"event": "document_kv_loaded", "request_id": "req-2"}]


def test_native_provider_releases_loaded_identity_for_finished_request_ids():
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    connector.register_kv_caches(
        {
            "layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
            "layer.1": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
        }
    )

    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())
    assert connector.get_finished({"req-1"}) == (None, None)
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    stats = connector.get_kv_connector_stats()
    assert stats["document_kv_loads_started"] == 2
    assert stats["document_kv_layers_loaded"] == 4
    assert connector.get_kv_connector_stats() is None


def test_native_provider_records_load_error_blocks_for_payload_view_failures(monkeypatch):
    def fail_payload_tensor_view(*args, **kwargs):
        del args, kwargs
        raise ValueError("payload view failed")

    monkeypatch.setattr(vllm_native_provider, "_payload_tensor_view", fail_payload_tensor_view)
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    connector.register_kv_caches({"layer.0": torch.zeros((8, 1, 2, 4), dtype=torch.int8)})
    connector.bind_connector_metadata(meta)

    with pytest.raises(ValueError, match="payload view failed"):
        connector.start_load_kv(SimpleNamespace())

    assert connector.get_block_ids_with_load_errors() == {5}
    assert connector.get_kv_connector_stats()["document_kv_load_error_blocks"] == 1
    assert connector.get_kv_connector_stats() is None


def test_native_provider_maps_vllm_layer_names_independently_of_registration_order():
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))
    connector = DocumentKVConnector(provider=provider)
    request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([5, 7]), 2)
    meta = connector.build_connector_meta(scheduler_output([5, 7]))
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)

    connector.register_kv_caches(
        {
            "model.layers.1.self_attn.attn": layer_1,
            "model.layers.0.self_attn.attn": layer_0,
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert torch.equal(logical_kv_slot(layer_0, 5, 0), torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 5, 0), torch.tensor([[[11, 12]], [[13, 14]]], dtype=torch.int8))


def test_native_provider_rejects_unparseable_registered_vllm_layer_names():
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))

    with pytest.raises(ValueError, match="Cannot determine vLLM layer index"):
        provider.register_kv_caches(
            {
                "attention_a": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
                "attention_b": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
            }
        )


def test_native_provider_rejects_duplicate_registered_vllm_layer_indices():
    provider = DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))

    with pytest.raises(ValueError, match="Duplicate vLLM layer index"):
        provider.register_kv_caches(
            {
                "model.layers.0.self_attn.attn": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
                "decoder.layers.0.self_attn.attn": torch.zeros((8, 1, 2, 4), dtype=torch.int8),
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
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)

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


def test_kv_transfer_params_source_builds_lazy_cachet_handoff_load(tmp_path, monkeypatch):
    adapter_request = build_engine_adapter_request(ready_request(), spec=vllm_adapter_spec())
    payload_uri = f"disk:{tmp_path / 'req-1.kv'}"
    handoff_path, _payload_path = write_engine_adapter_handoff_bundle(
        adapter_request,
        tmp_path / "handoff.json",
        payload_uri=payload_uri,
    )
    request = SimpleNamespace(
        request_id="req-1",
        kv_transfer_params={DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path)},
    )

    def fail_scheduler_payload_read(*args, **kwargs):
        del args, kwargs
        raise AssertionError("scheduler-side source must not materialize KV payload bytes")

    monkeypatch.setattr(vllm_native_provider, "read_engine_adapter_payload", fail_scheduler_payload_read)
    load = KVTransferParamsDocumentKVSource().get_load(request)

    assert load is not None
    assert load.request_id == "req-1"
    assert load.total_tokens == 3
    assert load.payload is None
    assert load.payload_uri == payload_uri


def test_kv_transfer_params_source_uses_cachet_request_id_for_wrapped_vllm_request(tmp_path):
    adapter_request = build_engine_adapter_request(ready_request(), spec=vllm_adapter_spec())
    payload_uri = f"disk:{tmp_path / 'req-1.kv'}"
    handoff_path, _payload_path = write_engine_adapter_handoff_bundle(
        adapter_request,
        tmp_path / "handoff.json",
        payload_uri=payload_uri,
    )
    request = SimpleNamespace(
        request_id="cmpl-req-1-0",
        kv_transfer_params={
            DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM: "benchmark-request-1",
            DOCUMENT_KV_REQUEST_ID_PARAM: "req-1",
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
        },
    )

    load = KVTransferParamsDocumentKVSource().get_load(request)

    assert load is not None
    assert load.request_id == "req-1"
    assert (
        load.actions.bind.metadata[DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM]
        == "benchmark-request-1"
    )
    assert load.payload is None
    assert load.payload_uri == payload_uri


def test_kv_transfer_params_source_rejects_cachet_request_id_mismatch(tmp_path):
    adapter_request = build_engine_adapter_request(ready_request(), spec=vllm_adapter_spec())
    handoff_path, _payload_path = write_engine_adapter_handoff_bundle(
        adapter_request,
        tmp_path / "handoff.json",
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    request = SimpleNamespace(
        request_id="cmpl-req-1-0",
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: "other-req",
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
        },
    )

    with pytest.raises(ValueError, match="document_kv.request_id must match handoff request_id"):
        KVTransferParamsDocumentKVSource().get_load(request)


def test_kv_transfer_params_source_rejects_unbound_benchmark_request_metadata(
    tmp_path,
):
    adapter_request = build_engine_adapter_request(
        ready_request(),
        spec=vllm_adapter_spec(),
    )
    handoff_path, _payload_path = write_engine_adapter_handoff_bundle(
        adapter_request,
        tmp_path / "handoff.json",
        payload_uri=f"disk:{tmp_path / 'req-1.kv'}",
    )
    handoff_record = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff_record["metadata"][DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM] = (
        "forged-request-id"
    )
    handoff_path.write_text(
        json.dumps(handoff_record),
        encoding="utf-8",
    )
    request = SimpleNamespace(
        request_id="cmpl-req-1-0",
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: "req-1",
            DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path),
        },
    )

    with pytest.raises(
        ValueError,
        match="reserved connector action metadata.*requires explicit",
    ):
        KVTransferParamsDocumentKVSource().get_load(request)


def test_benchmark_handoff_bundle_feeds_vllm_native_provider_load_path(tmp_path):
    input_path = tmp_path / "bio.jsonl"
    input_path.write_text(json.dumps(benchmark_jsonl_record()) + "\n", encoding="utf-8")
    result = generate_benchmark_handoff_bundles(
        input_path,
        output_dir=tmp_path / "bundles",
        generator=TwoTokenBenchmarkGenerator(),
        layout=layout(),
        align_bytes=1,
    )
    entry = result.manifest.entries[0]
    runtime_request_id = f"cmpl-{entry.request_id}-0"
    connector = DocumentKVConnector(provider=DocumentKVNativeProvider())
    runtime_params = entry.kv_transfer_params()
    runtime_request = SimpleNamespace(
        request_id=runtime_request_id,
        num_tokens=3,
        prompt_token_ids=(1, 2, 3),
        kv_transfer_params=runtime_params,
    )
    with pytest.raises(ValueError, match="runtime.*unsupported.*full logical prompt"):
        connector.get_num_new_matched_tokens(runtime_request, 0)
    logical_params = dict(runtime_params)
    logical_params[DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM] = "logical"
    logical_params[DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM] = entry.request_id
    request = SimpleNamespace(
        request_id=runtime_request_id,
        num_tokens=3,
        prompt_token_ids=(1, 2, 3),
        kv_transfer_params=logical_params,
    )
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([4]), 2)
    meta = connector.build_connector_meta(scheduler_output([4], request_id=runtime_request_id))
    assert meta.loads[0].request_id == runtime_request_id
    assert meta.loads[0].actions.reservation.request_id == runtime_request_id
    assert (
        meta.loads[0].actions.bind.metadata[DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM]
        == entry.request_id
    )
    assert meta.loads[0].payload is None
    assert meta.loads[0].payload_uri == entry.payload_uri
    connector.register_kv_caches(
        {
            "model.layers.0.self_attn.attn": layer_0,
            "model.layers.1.self_attn.attn": layer_1,
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert torch.equal(logical_kv_slot(layer_0, 4, 0), torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_0, 4, 1), torch.tensor([[[5, 6]], [[7, 8]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 4, 0), torch.tensor([[[11, 12]], [[13, 14]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 4, 1), torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8))
    assert connector.get_kv_connector_stats()["document_kv_layers_loaded"] == 2


def test_segmented_handoff_bundle_feeds_lazy_vllm_native_provider_load_path(tmp_path):
    adapter_request = build_engine_adapter_request(segmented_ready_request(), spec=vllm_adapter_spec())
    payload_uri = f"disk:{tmp_path / 'req-1.kv'}"
    handoff_path, _payload_path = write_engine_adapter_handoff_bundle(
        adapter_request,
        tmp_path / "handoff.json",
        payload_uri=payload_uri,
    )
    connector = DocumentKVConnector(provider=DocumentKVNativeProvider())
    request = SimpleNamespace(
        request_id="req-1",
        num_tokens=5,
        kv_transfer_params={DOCUMENT_KV_HANDOFF_JSON_PARAM: str(handoff_path)},
    )
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)

    assert connector.get_num_new_matched_tokens(request, 0) == (2, False)
    connector.update_state_after_alloc(request, AllocatedBlocks([4, 6]), 2)
    meta = connector.build_connector_meta(scheduler_output([4, 6]))
    assert meta.loads[0].payload is None
    assert meta.loads[0].payload_uri == payload_uri
    meta = pickle.loads(pickle.dumps(meta))
    connector.register_kv_caches(
        {
            "model.layers.0.self_attn.attn": layer_0,
            "model.layers.1.self_attn.attn": layer_1,
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert torch.equal(logical_kv_slot(layer_0, 4, 0), torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_0, 4, 1), torch.tensor([[[5, 6]], [[7, 8]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 4, 0), torch.tensor([[[11, 12]], [[13, 14]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 4, 1), torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8))
    assert connector.get_kv_connector_stats()["document_kv_layers_loaded"] == 2


@pytest.mark.parametrize("strategy", ("auto", "direct"))
def test_canonical_segmented_uri_uses_direct_global_snapshot_and_hashes_once(
    tmp_path,
    monkeypatch,
    strategy,
):
    monkeypatch.setenv(DOCUMENT_KV_SEGMENTED_LOAD_STRATEGY_ENV, strategy)
    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    inline = segmented_handoff_load()
    actions = replace(
        inline.actions,
        reservation=replace(
            inline.actions.reservation,
            payload_checksum=hashlib.sha256(payload()).hexdigest(),
        ),
    )
    telemetry_path = tmp_path / "direct-telemetry.jsonl"
    checksum_calls: list[int] = []
    tensor_view_payload_types: list[type[object]] = []
    original_verify = vllm_native_provider._verify_payload_checksum
    original_payload_tensor_view = vllm_native_provider._payload_tensor_view

    def count_checksum(action_record, raw_payload):
        checksum_calls.append(len(raw_payload))
        return original_verify(action_record, raw_payload)

    def inspect_payload_tensor_view(raw_payload, load):
        tensor_view_payload_types.append(type(raw_payload))
        return original_payload_tensor_view(raw_payload, load)

    monkeypatch.setattr(
        vllm_native_provider,
        "_verify_payload_checksum",
        count_checksum,
    )
    monkeypatch.setattr(
        vllm_native_provider,
        "_payload_tensor_view",
        inspect_payload_tensor_view,
    )
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(
            DocumentKVHandoffLoad(actions=actions, payload_uri=str(payload_path))
        ),
        telemetry_jsonl=str(telemetry_path),
    )

    _connector, layer_0, layer_1 = hydrate_two_token_load(provider)

    assert checksum_calls == [len(payload())]
    assert tensor_view_payload_types == [bytes]
    assert torch.equal(logical_kv_slot(layer_0, 4, 0), torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 4, 1), torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8))
    row = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert row["payload_loading"] == {
        "configured_segmented_strategy": strategy,
        "selected_strategy": "direct_global_snapshot",
        "payload_mode": "segmented",
        "canonical_segmented_global_view": True,
        "legacy_fallback_reason": None,
        "checksum_validation_count": 1,
        "snapshot_copy_bytes": len(payload()),
        "reassembly_copy_bytes": 0,
        "copy_metadata_retained": True,
        "copy_count": 2,
    }


def test_direct_global_snapshot_reuses_owned_payload_cache_bytes():
    cached_payload = payload()
    inline = segmented_handoff_load()
    actions = replace(
        inline.actions,
        reservation=replace(
            inline.actions.reservation,
            payload_checksum=hashlib.sha256(cached_payload).hexdigest(),
        ),
    )
    load = SimpleNamespace(
        actions=actions,
        payload=None,
        payload_uri="disk:/cache/req-1.kv",
    )

    def cached_reader(_payload_uri, *, expected_bytes, actions):
        assert expected_bytes == len(cached_payload)
        assert actions is load.actions
        return cached_payload

    materialized = vllm_native_provider._materialized_payload(
        load,
        payload_reader=cached_reader,
        segmented_load_strategy="auto",
    )

    assert materialized.selected_strategy == "direct_global_snapshot"
    assert materialized.payload is cached_payload
    assert materialized.snapshot_copy_bytes == 0
    assert materialized.checksum_validation_count == 1


def test_segmented_uri_legacy_switch_preserves_two_copy_path(tmp_path, monkeypatch):
    monkeypatch.setenv(DOCUMENT_KV_SEGMENTED_LOAD_STRATEGY_ENV, "legacy")
    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    inline = segmented_handoff_load()
    actions = replace(
        inline.actions,
        reservation=replace(
            inline.actions.reservation,
            payload_checksum=hashlib.sha256(payload()).hexdigest(),
        ),
    )
    telemetry_path = tmp_path / "legacy-telemetry.jsonl"
    checksum_calls: list[int] = []
    original_verify = vllm_native_provider._verify_payload_checksum

    def count_checksum(action_record, raw_payload):
        checksum_calls.append(sum(map(len, raw_payload)) if isinstance(raw_payload, tuple) else len(raw_payload))
        return original_verify(action_record, raw_payload)

    monkeypatch.setattr(
        vllm_native_provider,
        "_verify_payload_checksum",
        count_checksum,
    )
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(
            DocumentKVHandoffLoad(actions=actions, payload_uri=str(payload_path))
        ),
        telemetry_jsonl=str(telemetry_path),
    )

    _connector, layer_0, layer_1 = hydrate_two_token_load(provider)

    assert checksum_calls == [len(payload()), len(payload())]
    assert torch.equal(logical_kv_slot(layer_0, 4, 0), torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 4, 1), torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8))
    row = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert row["payload_loading"] == {
        "configured_segmented_strategy": "legacy",
        "selected_strategy": "legacy_segment_remerge",
        "payload_mode": "segmented",
        "canonical_segmented_global_view": True,
        "legacy_fallback_reason": "configured_legacy",
        "checksum_validation_count": 2,
        "snapshot_copy_bytes": 0,
        "reassembly_copy_bytes": 2 * len(payload()),
        "copy_metadata_retained": True,
        "copy_count": 2,
    }


def test_noncanonical_segmented_uri_auto_uses_strict_legacy_fallback(
    tmp_path,
):
    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    inline = segmented_handoff_load()
    noncanonical_actions = replace(
        inline.actions,
        copies=tuple(
            replace(copy, source_byte_start=4)
            for copy in inline.actions.copies
        ),
    )
    telemetry_path = tmp_path / "fallback-telemetry.jsonl"
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(
            DocumentKVHandoffLoad(
                actions=noncanonical_actions,
                payload_uri=str(payload_path),
            )
        ),
        telemetry_jsonl=str(telemetry_path),
    )

    _connector, layer_0, layer_1 = hydrate_two_token_load(provider)

    assert torch.equal(logical_kv_slot(layer_0, 4, 0), torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 4, 1), torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8))
    row = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert row["payload_loading"]["configured_segmented_strategy"] == "auto"
    assert row["payload_loading"]["selected_strategy"] == "legacy_segment_remerge"
    assert row["payload_loading"]["canonical_segmented_global_view"] is False
    assert row["payload_loading"]["legacy_fallback_reason"] == "source_byte_start_is_not_zero"
    assert row["payload_loading"]["reassembly_copy_bytes"] == 2 * len(payload())


def test_direct_global_snapshot_is_stable_after_checksum_when_file_mutates(
    tmp_path,
    monkeypatch,
):
    payload_path = tmp_path / "req-1.kv"
    original_payload = payload()
    mutated_payload = bytes(reversed(original_payload))
    payload_path.write_bytes(original_payload)
    inline = segmented_handoff_load()
    actions = replace(
        inline.actions,
        reservation=replace(
            inline.actions.reservation,
            payload_checksum=hashlib.sha256(original_payload).hexdigest(),
        ),
    )
    telemetry_path = tmp_path / "snapshot-race-telemetry.jsonl"
    checksum_calls = 0
    original_verify = vllm_native_provider._verify_payload_checksum

    def mutate_backing_file_after_checksum(action_record, raw_payload):
        nonlocal checksum_calls
        checksum_calls += 1
        original_verify(action_record, raw_payload)
        with payload_path.open("r+b") as payload_file:
            assert payload_file.write(mutated_payload) == len(mutated_payload)
            payload_file.flush()

    monkeypatch.setattr(
        vllm_native_provider,
        "_verify_payload_checksum",
        mutate_backing_file_after_checksum,
    )
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(
            DocumentKVHandoffLoad(actions=actions, payload_uri=str(payload_path))
        ),
        telemetry_jsonl=str(telemetry_path),
    )

    _connector, layer_0, layer_1 = hydrate_two_token_load(provider)

    assert checksum_calls == 1
    assert payload_path.read_bytes() == mutated_payload
    assert torch.equal(
        logical_kv_slot(layer_0, 4, 0),
        torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8),
    )
    assert torch.equal(
        logical_kv_slot(layer_1, 4, 1),
        torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8),
    )
    row = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert row["payload_loading"]["selected_strategy"] == (
        "direct_global_snapshot"
    )
    assert row["payload_loading"]["checksum_validation_count"] == 1
    assert row["payload_loading"]["snapshot_copy_bytes"] == len(original_payload)
    assert row["payload_loading"]["reassembly_copy_bytes"] == 0


def test_noncanonical_segmented_uri_direct_mode_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv(DOCUMENT_KV_SEGMENTED_LOAD_STRATEGY_ENV, "direct")
    payload_path = tmp_path / "req-1.kv"
    payload_path.write_bytes(payload())
    inline = segmented_handoff_load()
    noncanonical_actions = replace(
        inline.actions,
        copies=tuple(
            replace(copy, payload_index=len(inline.actions.copies) - index - 1)
            for index, copy in enumerate(inline.actions.copies)
        ),
    )
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(
            DocumentKVHandoffLoad(
                actions=noncanonical_actions,
                payload_uri=str(payload_path),
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="direct segmented payload loading requires canonical Cachet copy metadata",
    ):
        hydrate_two_token_load(provider)


def test_direct_mode_rejects_inline_segmented_tuple(monkeypatch):
    monkeypatch.setenv(DOCUMENT_KV_SEGMENTED_LOAD_STRATEGY_ENV, "direct")
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(segmented_handoff_load())
    )

    with pytest.raises(
        ValueError,
        match="requires one flat external payload buffer",
    ):
        hydrate_two_token_load(provider)


def test_direct_eligibility_explicitly_requires_cumulative_global_ranges():
    actions = segmented_handoff_load().actions
    first_copy = actions.copies[0]
    noncumulative_actions = replace(
        actions,
        copies=(
            replace(
                first_copy,
                global_byte_start=8,
                global_byte_end=8 + first_copy.source_byte_length,
            ),
            actions.copies[1],
        ),
    )

    assert (
        vllm_native_provider._canonical_segmented_global_view_issue(
            noncumulative_actions
        )
        == "global_byte_start_is_not_cumulative"
    )


def test_native_provider_rejects_unknown_segmented_load_strategy(monkeypatch):
    monkeypatch.setenv(DOCUMENT_KV_SEGMENTED_LOAD_STRATEGY_ENV, "fast-ish")

    with pytest.raises(ValueError, match="must be one of 'auto', 'direct', or 'legacy'"):
        DocumentKVNativeProvider(source=StaticHandoffSource(handoff_load()))


def test_lazy_segmented_payload_uri_respects_copy_payload_index(tmp_path):
    request = segmented_ready_request()
    payload_uri = f"disk:{tmp_path / 'req-1.kv'}"
    (tmp_path / "req-1.kv").write_bytes(b"".join(request.payload))
    adapter_request = build_engine_adapter_request(request, spec=vllm_adapter_spec())
    record = engine_adapter_request_to_record(adapter_request, payload_uri=payload_uri)
    plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
    actions = build_engine_kv_connector_actions(plan, request.payload)
    reordered_actions = replace(
        actions,
        copies=(
            replace(actions.copies[0], payload_index=1),
            replace(actions.copies[1], payload_index=0),
        ),
    )
    provider = DocumentKVNativeProvider(
        source=StaticHandoffSource(DocumentKVHandoffLoad(actions=reordered_actions, payload_uri=payload_uri))
    )
    connector = DocumentKVConnector(provider=provider)
    vllm_request = SimpleNamespace(request_id="req-1", num_tokens=5, kv_transfer_params={})
    layer_0 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)
    layer_1 = torch.zeros((8, 1, 2, 4), dtype=torch.int8)

    assert connector.get_num_new_matched_tokens(vllm_request, 0) == (2, False)
    connector.update_state_after_alloc(vllm_request, AllocatedBlocks([4, 6]), 2)
    meta = connector.build_connector_meta(scheduler_output([4, 6]))
    connector.register_kv_caches(
        {
            "model.layers.0.self_attn.attn": layer_0,
            "model.layers.1.self_attn.attn": layer_1,
        }
    )
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(SimpleNamespace())

    assert torch.equal(logical_kv_slot(layer_0, 4, 0), torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_0, 4, 1), torch.tensor([[[5, 6]], [[7, 8]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 4, 0), torch.tensor([[[11, 12]], [[13, 14]]], dtype=torch.int8))
    assert torch.equal(logical_kv_slot(layer_1, 4, 1), torch.tensor([[[15, 16]], [[17, 18]]], dtype=torch.int8))


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


def test_native_provider_factory_accepts_payload_cache_budget():
    provider = build_document_kv_provider(
        vllm_config=None,
        extra_config={DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY: 4096},
    )

    assert isinstance(provider, DocumentKVNativeProvider)


def test_native_provider_factory_accepts_telemetry_jsonl_path(tmp_path):
    telemetry_path = tmp_path / "connector.jsonl"
    provider = build_document_kv_provider(
        vllm_config=None,
        extra_config={DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY: str(telemetry_path)},
    )

    assert isinstance(provider, DocumentKVNativeProvider)
    assert provider.telemetry_jsonl == str(telemetry_path)


@pytest.mark.parametrize("value", [-1, True, "4096"])
def test_native_provider_factory_rejects_invalid_payload_cache_budget(value):
    with pytest.raises((TypeError, ValueError), match=DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY):
        build_document_kv_provider(
            vllm_config=None,
            extra_config={DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY: value},
        )


@pytest.mark.parametrize("value", ["", " "])
def test_native_provider_factory_rejects_invalid_telemetry_jsonl(value):
    with pytest.raises(ValueError, match=DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY):
        build_document_kv_provider(
            vllm_config=None,
            extra_config={DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY: value},
        )


def test_native_probe_connector_does_not_require_vllm_base_config(monkeypatch):
    def fail_base_init(*args, **kwargs):
        raise AssertionError("probe connector must not initialize DocumentKVConnector base")

    monkeypatch.setattr(
        "vllm_kv_injection.vllm_dynamic_connector.DocumentKVConnector.__init__",
        fail_base_init,
    )

    connector = DocumentKVNativeProbeConnector()

    assert isinstance(connector.provider, DocumentKVNativeProvider)
