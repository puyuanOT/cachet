import pytest

from document_kv_cache.admission import AdmissionQueue
from document_kv_cache.cache import ChunkCache
from document_kv_cache.engine_protocol import KVLayout, KVStorageLayout
from document_kv_cache.kvpack import PackChunk, write_kvpack
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.materializer import KVMaterializer
from document_kv_cache.models import CacheGenerationMethod, DocumentChunkType, DocumentKVRequest, KVCacheKey
from document_kv_cache.planner import CachePlanner
from document_kv_cache.service import DocumentKVService
from document_kv_cache.storage import DiskRangeReader
from document_kv_cache.workflow import (
    CacheAdapterArtifact,
    CacheBuildConfig,
    DocumentKVWorkflow,
    SourceChunk,
    SourceDocument,
    TrainingArtifacts,
)


class EchoGenerator:
    def generate(
        self,
        *,
        document: SourceDocument,
        chunk: SourceChunk,
        config: CacheBuildConfig,
        training_artifacts: TrainingArtifacts | None = None,
    ) -> PackChunk:
        suffix = ""
        if training_artifacts is not None and training_artifacts.adapter_ids:
            suffix = f"|{training_artifacts.adapter_ids[0]}"
        payload = f"{document.document_id}:{chunk.chunk_id}:{chunk.text}{suffix}".encode("utf-8")
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
            ),
            payload=payload,
            token_count=max(1, len(chunk.text.split())),
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


class ByteAlignedGenerator(EchoGenerator):
    def generate(self, **kwargs) -> PackChunk:
        pack_chunk = super().generate(**kwargs)
        return PackChunk(
            key=pack_chunk.key,
            payload=pack_chunk.payload,
            token_count=len(pack_chunk.payload),
            dtype=pack_chunk.dtype,
            layout_version=pack_chunk.layout_version,
            storage_layout=pack_chunk.storage_layout,
        )


class RecordingTrainer:
    def __init__(self) -> None:
        self.document_ids: tuple[str, ...] = ()

    def fit(self, documents, config: CacheBuildConfig) -> TrainingArtifacts:
        self.document_ids = tuple(document.document_id for document in documents)
        return TrainingArtifacts(adapter_ids=(f"{config.model_id}-adapter",), metadata={"trained": "true"})


class ArtifactTrainer:
    def fit(self, documents, config: CacheBuildConfig) -> TrainingArtifacts:
        adapter = CacheAdapterArtifact(
            adapter_id=f"{config.model_id}-kv-packet",
            artifact_uri="/Volumes/catalog/schema/volume/adapters/qwen3-kv-packet.safetensors",
            cache_method=CacheGenerationMethod.KV_PACKET,
            metadata={"rank": "8"},
        )
        return TrainingArtifacts(metadata={"trained": "true"}, adapter_artifacts=(adapter,))


class WrongDocumentGenerator(EchoGenerator):
    def generate(self, **kwargs) -> PackChunk:
        pack_chunk = super().generate(**kwargs)
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=pack_chunk.key.model_id,
                lora_id=pack_chunk.key.lora_id,
                prompt_template_version=pack_chunk.key.prompt_template_version,
                document_id="wrong-doc",
                chunk_type=pack_chunk.key.chunk_type,
                chunk_id=pack_chunk.key.chunk_id,
            ),
            payload=pack_chunk.payload,
            token_count=pack_chunk.token_count,
            dtype=pack_chunk.dtype,
            layout_version=pack_chunk.layout_version,
            storage_layout=pack_chunk.storage_layout,
        )


class WrongStorageLayoutGenerator(EchoGenerator):
    def generate(self, **kwargs) -> PackChunk:
        pack_chunk = super().generate(**kwargs)
        return PackChunk(
            key=pack_chunk.key,
            payload=pack_chunk.payload,
            token_count=pack_chunk.token_count,
            dtype=pack_chunk.dtype,
            layout_version=pack_chunk.layout_version,
            storage_layout=KVStorageLayout.SHARED_KEY_VALUE,
        )


def config(*, cache_method: CacheGenerationMethod = CacheGenerationMethod.VANILLA_PREFILL) -> CacheBuildConfig:
    return CacheBuildConfig(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        dtype="int8",
        layout_version="toy-one-byte-v1",
        cache_method=cache_method,
    )


def request_for(document_id: str) -> DocumentKVRequest:
    return DocumentKVRequest(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_chunks={document_id: ["section-1"]},
    )


def one_byte_layout() -> KVLayout:
    return KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version="toy-one-byte-v1",
        dtype="int8",
        num_layers=1,
        block_size=16,
        bytes_per_token=1,
    )


def test_source_chunk_validates_and_normalizes_public_inputs():
    chunk = SourceChunk(
        chunk_id="p1",
        text="body",
        chunk_type="document_static",
        metadata={"source": "fixture"},
    )

    assert chunk.chunk_id == "p1"
    assert chunk.chunk_type == DocumentChunkType.DOCUMENT_STATIC
    assert chunk.metadata == {"source": "fixture"}

    with pytest.raises(ValueError, match="chunk_id must be a non-empty string"):
        SourceChunk(chunk_id="", text="body")
    with pytest.raises(TypeError, match="text must be a string"):
        SourceChunk(chunk_id="p1", text=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="chunk_type must be one of"):
        SourceChunk(chunk_id="p1", text="body", chunk_type="bad-type")
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        SourceChunk(chunk_id="p1", text="body", metadata=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metadata keys must be non-empty strings"):
        SourceChunk(chunk_id="p1", text="body", metadata={"": "fixture"})
    with pytest.raises(ValueError, match="metadata.source must be a string"):
        SourceChunk(chunk_id="p1", text="body", metadata={"source": 1})  # type: ignore[dict-item]


def test_source_document_validates_and_normalizes_public_inputs():
    chunk = SourceChunk(chunk_id="p1", text="body")
    document = SourceDocument(document_id="doc-a", chunks=[chunk], metadata={"title": "Doc A"})  # type: ignore[arg-type]

    assert document.document_id == "doc-a"
    assert document.chunks == (chunk,)
    assert document.metadata == {"title": "Doc A"}

    with pytest.raises(ValueError, match="document_id must be a non-empty string"):
        SourceDocument(document_id="", chunks=(chunk,))
    with pytest.raises(ValueError, match="chunks must contain at least one SourceChunk"):
        SourceDocument(document_id="doc-a", chunks=())
    with pytest.raises(TypeError, match="chunks entries must be SourceChunk instances"):
        SourceDocument(document_id="doc-a", chunks=("not-a-chunk",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metadata.title must be a string"):
        SourceDocument(document_id="doc-a", chunks=(chunk,), metadata={"title": object()})  # type: ignore[dict-item]


def test_source_document_from_text_builds_single_document_chunk():
    document = SourceDocument.from_text(
        document_id="doc-a",
        text="one long document",
        metadata={"title": "Doc A"},
        chunk_metadata={"section": "body"},
    )

    assert document.document_id == "doc-a"
    assert document.metadata == {"title": "Doc A"}
    assert len(document.chunks) == 1
    assert document.chunks[0] == SourceChunk(
        chunk_id="document",
        text="one long document",
        chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
        metadata={"section": "body"},
    )


def test_source_document_from_text_accepts_custom_chunk_identity_and_type():
    document = SourceDocument.from_text(
        document_id="doc-a",
        text="static profile",
        chunk_id="profile",
        chunk_type="document_static",
    )

    assert document.chunks == (
        SourceChunk(
            chunk_id="profile",
            text="static profile",
            chunk_type=DocumentChunkType.DOCUMENT_STATIC,
        ),
    )


def test_source_document_from_text_reuses_source_chunk_validation():
    with pytest.raises(TypeError, match="text must be a string"):
        SourceDocument.from_text(document_id="doc-a", text=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="chunk_id must be a non-empty string"):
        SourceDocument.from_text(document_id="doc-a", text="body", chunk_id="")
    with pytest.raises(ValueError, match="chunk_type must be one of"):
        SourceDocument.from_text(document_id="doc-a", text="body", chunk_type="bad-type")
    with pytest.raises(ValueError, match="chunk_metadata.source must be a string"):
        SourceDocument.from_text(document_id="doc-a", text="body", chunk_metadata={"source": 1})  # type: ignore[dict-item]


def test_source_document_from_texts_validates_helper_inputs():
    document = SourceDocument.from_texts(
        document_id="doc-a",
        static_text="static context",
        chunks={"p1": "body"},
        metadata={"title": "Doc A"},
    )

    assert [chunk.chunk_id for chunk in document.chunks] == ["static", "p1"]
    assert document.chunks[0].chunk_type == DocumentChunkType.DOCUMENT_STATIC
    assert document.metadata == {"title": "Doc A"}

    with pytest.raises(TypeError, match="chunks must be a mapping"):
        SourceDocument.from_texts(document_id="doc-a", chunks=["body"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        SourceDocument.from_texts(document_id="doc-a", chunks={"p1": "body"}, metadata=[])  # type: ignore[arg-type]


def test_workflow_generates_registers_and_prepares_cache(tmp_path):
    manifest = InMemoryManifestStore()
    workflow = DocumentKVWorkflow(
        manifest=manifest,
        materializer=KVMaterializer(
            cache=ChunkCache(cpu_max_bytes=4096),
            reader=DiskRangeReader(),
        ),
    )
    document = SourceDocument.from_texts(
        document_id="doc-a",
        static_text="static context",
        chunks={"section-1": "hello world"},
    )

    result = workflow.generate_cache(
        documents=(document,),
        generator=EchoGenerator(),
        config=config(),
        shard_uri=tmp_path / "doc-cache.kvpack",
        align_bytes=1,
    )
    materialized = workflow.prepare(request_for("doc-a"))

    assert result.document_ids == ("doc-a",)
    assert result.chunk_count == 2
    assert result.total_bytes == len(materialized.payload)
    assert result.cache_method == CacheGenerationMethod.VANILLA_PREFILL
    assert b"doc-a:static:static context" in materialized.payload
    assert b"doc-a:section-1:hello world" in materialized.payload
    assert manifest.keys_for_document("doc-a")


def test_workflow_generates_and_prepares_single_text_document(tmp_path):
    manifest = InMemoryManifestStore()
    workflow = DocumentKVWorkflow(
        manifest=manifest,
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )
    document = SourceDocument.from_text(document_id="doc-a", text="hello from one text document")
    request = DocumentKVRequest.for_text_document(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_id="doc-a",
    )

    result = workflow.generate_cache(
        documents=(document,),
        generator=EchoGenerator(),
        config=config(),
        shard_uri=tmp_path / "one-text-cache.kvpack",
        align_bytes=1,
    )
    materialized = workflow.prepare(request)

    assert result.document_ids == ("doc-a",)
    assert result.chunk_count == 1
    assert request.document_chunks == {"doc-a": ("document",)}
    assert materialized.payload == b"doc-a:document:hello from one text document"


def test_workflow_generates_and_prepares_selected_document_chunks(tmp_path):
    manifest = InMemoryManifestStore()
    workflow = DocumentKVWorkflow(
        manifest=manifest,
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )
    document = SourceDocument.from_texts(
        document_id="doc-a",
        static_text="profile",
        chunks={"review-1": "first review", "review-2": "second review"},
    )
    request = DocumentKVRequest.for_document_chunks(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_id="doc-a",
        chunk_ids=("review-2",),
    )

    result = workflow.generate_cache(
        documents=(document,),
        generator=EchoGenerator(),
        config=config(),
        shard_uri=tmp_path / "selected-document-chunks.kvpack",
        align_bytes=1,
    )
    materialized = workflow.prepare(request)

    assert result.chunk_count == 3
    assert request.document_chunks == {"doc-a": ("review-2",)}
    assert b"doc-a:static:profile" in materialized.payload
    assert b"doc-a:review-2:second review" in materialized.payload
    assert b"doc-a:review-1:first review" not in materialized.payload


def test_workflow_invokes_optional_training_adapter(tmp_path):
    trainer = RecordingTrainer()
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )
    document = SourceDocument.from_texts(document_id="doc-a", static_text="static", chunks={"section-1": "body"})

    result = workflow.generate_cache(
        documents=(document,),
        generator=EchoGenerator(),
        config=config(),
        shard_uri=tmp_path / "trained-cache.kvpack",
        trainer=trainer,
        align_bytes=1,
    )
    materialized = workflow.prepare(request_for("doc-a"))

    assert trainer.document_ids == ("doc-a",)
    assert result.training_artifacts == TrainingArtifacts(
        adapter_ids=("qwen3:4b-instruct-adapter",),
        metadata={"trained": "true"},
    )
    assert result.cache_method == CacheGenerationMethod.ADAPTER_TRAINED
    assert b"|qwen3:4b-instruct-adapter" in materialized.payload


def test_workflow_records_training_adapter_artifacts_and_derives_engine_adapters(tmp_path):
    manifest = InMemoryManifestStore()
    materializer = KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader())
    workflow = DocumentKVWorkflow(manifest=manifest, materializer=materializer)
    document = SourceDocument.from_texts(document_id="doc-a", static_text="static", chunks={"section-1": "body"})

    result = workflow.generate_cache(
        documents=(document,),
        generator=ByteAlignedGenerator(),
        config=config(cache_method=CacheGenerationMethod.KV_PACKET),
        shard_uri=tmp_path / "artifact-cache.kvpack",
        trainer=ArtifactTrainer(),
        align_bytes=1,
    )
    ready = workflow.prepare_for_engine(
        request_for("doc-a"),
        layout=one_byte_layout(),
        cache_method=result.cache_method,
        training_artifacts=result.training_artifacts,
    )

    assert result.adapter_ids == ("qwen3:4b-instruct-kv-packet",)
    assert result.training_artifacts is not None
    assert result.training_artifacts.adapter_ids == result.adapter_ids
    assert result.training_artifacts.adapter_artifacts[0].cache_method == "kv_packet"
    assert result.training_artifacts.adapter_artifacts[0].metadata == {"rank": "8"}
    assert ready.handle.cache_method == "kv_packet"
    assert ready.handle.adapter_ids == result.adapter_ids


def test_workflow_records_non_vanilla_cache_generation_method(tmp_path):
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )
    document = SourceDocument.from_texts(document_id="doc-a", static_text="static", chunks={"section-1": "body"})

    result = workflow.generate_cache(
        documents=(document,),
        generator=EchoGenerator(),
        config=config(cache_method=CacheGenerationMethod.KV_PACKET),
        shard_uri=tmp_path / "kv-packet-cache.kvpack",
        align_bytes=1,
    )

    assert result.cache_method == CacheGenerationMethod.KV_PACKET
    assert result.training_artifacts is None


def test_training_artifacts_validate_adapter_artifact_identity():
    adapter = CacheAdapterArtifact(adapter_id="packet-adapter", artifact_uri="/Volumes/cache/packet.safetensors")

    with pytest.raises(ValueError, match="adapter_ids must match"):
        TrainingArtifacts(adapter_ids=("other-adapter",), adapter_artifacts=(adapter,))


def test_training_artifacts_reject_bare_string_adapter_ids():
    with pytest.raises(ValueError, match="not a string"):
        TrainingArtifacts(adapter_ids="packet-adapter")  # type: ignore[arg-type]


def test_workflow_rejects_generator_key_mismatch(tmp_path):
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )
    document = SourceDocument.from_texts(document_id="doc-a", static_text="static", chunks={"section-1": "body"})

    with pytest.raises(ValueError, match="document_id"):
        workflow.generate_cache(
            documents=(document,),
            generator=WrongDocumentGenerator(),
            config=config(),
            shard_uri=tmp_path / "bad-cache.kvpack",
            align_bytes=1,
        )


def test_workflow_rejects_generator_storage_layout_mismatch(tmp_path):
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )
    document = SourceDocument.from_texts(document_id="doc-a", static_text="static", chunks={"section-1": "body"})

    with pytest.raises(ValueError, match="storage_layout"):
        workflow.generate_cache(
            documents=(document,),
            generator=WrongStorageLayoutGenerator(),
            config=config(),
            shard_uri=tmp_path / "bad-storage-layout.kvpack",
            align_bytes=1,
        )


def test_workflow_can_enqueue_prepared_request(tmp_path):
    manifest = InMemoryManifestStore()
    materializer = KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader())
    service = DocumentKVService(
        planner=CachePlanner(manifest),
        materializer=materializer,
        admission_queue=AdmissionQueue(max_pending_gpu_bytes=4096),
    )
    workflow = DocumentKVWorkflow(manifest=manifest, materializer=materializer, service=service)
    document = SourceDocument.from_texts(document_id="doc-a", static_text="static", chunks={"section-1": "body"})
    workflow.generate_cache(
        documents=(document,),
        generator=EchoGenerator(),
        config=config(),
        shard_uri=tmp_path / "enqueue-cache.kvpack",
        align_bytes=1,
    )

    assert workflow.prepare_and_enqueue(request_for("doc-a"))


def test_workflow_prepare_uses_injected_service_dependencies(tmp_path):
    service_refs = write_kvpack(
        tmp_path / "service-prepare.kvpack",
        [
            PackChunk(
                key=KVCacheKey.for_document(
                    model_id="qwen3:4b-instruct",
                    lora_id="base",
                    prompt_template_version="v1",
                    document_id="doc-a",
                    chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
                    chunk_id="section-1",
                ),
                payload=b"service-bytes",
                token_count=len(b"service-bytes"),
                dtype="int8",
                layout_version="toy-one-byte-v1",
            )
        ],
        align_bytes=1,
    )
    service = DocumentKVService(
        planner=CachePlanner(InMemoryManifestStore(service_refs)),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
        admission_queue=AdmissionQueue(max_pending_gpu_bytes=4096),
    )
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
        service=service,
    )

    materialized = workflow.prepare(
        DocumentKVRequest(
            request_id="req-1",
            task_id="qa",
            model_id="qwen3:4b-instruct",
            lora_id="base",
            prompt_template_version="v1",
            document_chunks={"doc-a": ["section-1"]},
            include_static=False,
        ),
    )

    assert materialized.payload == b"service-bytes"


def test_workflow_prepares_engine_ready_request(tmp_path):
    manifest = InMemoryManifestStore()
    materializer = KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader())
    service = DocumentKVService(
        planner=CachePlanner(manifest),
        materializer=materializer,
        admission_queue=AdmissionQueue(max_pending_gpu_bytes=4096),
        kv_gpu_bytes_per_payload_byte=2.0,
    )
    workflow = DocumentKVWorkflow(manifest=manifest, materializer=materializer, service=service)
    document = SourceDocument.from_texts(document_id="doc-a", static_text="static", chunks={"section-1": "body"})
    workflow.generate_cache(
        documents=(document,),
        generator=ByteAlignedGenerator(),
        config=config(cache_method=CacheGenerationMethod.KV_PACKET),
        shard_uri=tmp_path / "engine-cache.kvpack",
        align_bytes=1,
    )

    ready = workflow.prepare_for_engine(
        request_for("doc-a"),
        layout=one_byte_layout(),
        metadata={"engine": "vllm"},
        cache_method=CacheGenerationMethod.KV_PACKET,
        adapter_ids=("qa-lora",),
        segmented=True,
    )

    assert ready.handle.cache_method == "kv_packet"
    assert ready.handle.metadata == {"engine": "vllm"}
    assert ready.handle.adapter_ids == ("qa-lora",)
    assert isinstance(ready.payload, tuple)
    assert ready.estimated_gpu_bytes == 2 * ready.handle.total_bytes


def test_workflow_engine_handoff_uses_injected_service_dependencies(tmp_path):
    service_refs = write_kvpack(
        tmp_path / "service-only.kvpack",
        [
            PackChunk(
                key=KVCacheKey.for_document(
                    model_id="qwen3:4b-instruct",
                    lora_id="base",
                    prompt_template_version="v1",
                    document_id="doc-a",
                    chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
                    chunk_id="section-1",
                ),
                payload=b"service-bytes",
                token_count=len(b"service-bytes"),
                dtype="int8",
                layout_version="toy-one-byte-v1",
            )
        ],
        align_bytes=1,
    )
    service = DocumentKVService(
        planner=CachePlanner(InMemoryManifestStore(service_refs)),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
        admission_queue=AdmissionQueue(max_pending_gpu_bytes=4096),
    )
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
        service=service,
    )

    ready = workflow.prepare_for_engine(
        DocumentKVRequest(
            request_id="req-1",
            task_id="qa",
            model_id="qwen3:4b-instruct",
            lora_id="base",
            prompt_template_version="v1",
            document_chunks={"doc-a": ["section-1"]},
            include_static=False,
        ),
        layout=one_byte_layout(),
        segmented=True,
    )

    assert ready.payload == (b"service-bytes",)
    assert ready.handle.total_bytes == len(b"service-bytes")


def test_workflow_requires_service_before_enqueue():
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )

    with pytest.raises(RuntimeError, match="without a DocumentKVService"):
        workflow.prepare_and_enqueue(request_for("doc-a"))


def test_workflow_prepares_engine_ready_request_without_service(tmp_path):
    manifest = InMemoryManifestStore()
    materializer = KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader())
    workflow = DocumentKVWorkflow(manifest=manifest, materializer=materializer)
    document = SourceDocument.from_texts(document_id="doc-a", static_text="static", chunks={"section-1": "body"})
    workflow.generate_cache(
        documents=(document,),
        generator=ByteAlignedGenerator(),
        config=config(),
        shard_uri=tmp_path / "engine-cache-no-service.kvpack",
        align_bytes=1,
    )

    ready = workflow.prepare_for_engine(request_for("doc-a"), layout=one_byte_layout())

    assert ready.handle.cache_method == "vanilla_prefill"
    assert ready.estimated_gpu_bytes == ready.handle.total_bytes


def test_workflow_engine_handoff_accepts_explicit_gpu_multiplier(tmp_path):
    manifest = InMemoryManifestStore()
    materializer = KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader())
    workflow = DocumentKVWorkflow(manifest=manifest, materializer=materializer)
    document = SourceDocument.from_texts(document_id="doc-a", static_text="static", chunks={"section-1": "body"})
    workflow.generate_cache(
        documents=(document,),
        generator=ByteAlignedGenerator(),
        config=config(),
        shard_uri=tmp_path / "engine-cache-explicit-multiplier.kvpack",
        align_bytes=1,
    )

    ready = workflow.prepare_for_engine(
        request_for("doc-a"),
        layout=one_byte_layout(),
        kv_gpu_bytes_per_payload_byte=3.0,
    )

    assert ready.estimated_gpu_bytes == 3 * ready.handle.total_bytes


@pytest.mark.parametrize(
    ("multiplier", "error_type", "message"),
    [
        (True, TypeError, "kv_gpu_bytes_per_payload_byte must be numeric"),
        ("1.0", TypeError, "kv_gpu_bytes_per_payload_byte must be numeric"),
        (float("nan"), ValueError, "kv_gpu_bytes_per_payload_byte must be finite"),
        (float("inf"), ValueError, "kv_gpu_bytes_per_payload_byte must be finite"),
        (-1.0, ValueError, "kv_gpu_bytes_per_payload_byte must be non-negative"),
    ],
)
def test_workflow_engine_handoff_rejects_invalid_gpu_multiplier(multiplier, error_type, message):
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )

    with pytest.raises(error_type, match=message):
        workflow.prepare_for_engine(
            request_for("doc-a"),
            layout=one_byte_layout(),
            kv_gpu_bytes_per_payload_byte=multiplier,
        )


def test_workflow_engine_handoff_rejects_bare_string_adapter_ids():
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )

    with pytest.raises(ValueError, match="not a string"):
        workflow.prepare_for_engine(
            request_for("doc-a"),
            layout=one_byte_layout(),
            adapter_ids="packet-adapter",  # type: ignore[arg-type]
        )
