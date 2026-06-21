import pytest

from document_kv_cache.admission import AdmissionQueue
from document_kv_cache.cache import CacheTier, ChunkCache
from document_kv_cache.engine_protocol import KVLayout, KVStorageLayout
from document_kv_cache.kvpack import PackChunk, write_kvpack, write_kvpack_bytes
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.materializer import KVMaterializer
from document_kv_cache.models import CacheGenerationMethod, DocumentChunkType, DocumentKVRequest, KVCacheKey
from document_kv_cache.planner import CachePlanner
from document_kv_cache.service import DocumentKVService
from document_kv_cache.storage import DiskRangeReader, MemoryRangeReader, RoutedRangeReader
from document_kv_cache.workflow import (
    CacheAdapterArtifact,
    CacheBuildConfig,
    CacheGenerationResult,
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


class RecordingConnector:
    def __init__(self) -> None:
        self.submitted = []
        self.released: list[str] = []

    def submit(self, request) -> None:
        self.submitted.append(request)

    def release(self, request_id: str) -> None:
        self.released.append(request_id)


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


def result_ref(tmp_path, *, document_id: str = "doc-a", chunk_id: str = "section-1", filename: str = "result-ref.kvpack"):
    return write_kvpack(
        tmp_path / filename,
        [
            PackChunk(
                key=KVCacheKey.for_document(
                    model_id="qwen3:4b-instruct",
                    lora_id="base",
                    prompt_template_version="v1",
                    document_id=document_id,
                    chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
                    chunk_id=chunk_id,
                ),
                payload=b"result-bytes",
                token_count=2,
                dtype="int8",
                layout_version="toy-one-byte-v1",
            )
        ],
        align_bytes=1,
    )[0]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"model_id": ""}, "model_id"),
        ({"lora_id": ""}, "lora_id"),
        ({"prompt_template_version": ""}, "prompt_template_version"),
        ({"dtype": ""}, "dtype"),
        ({"layout_version": ""}, "layout_version"),
        ({"cache_method": ""}, "cache_method"),
        ({"cache_method": object()}, "cache_method"),
    ),
)
def test_cache_build_config_rejects_invalid_identity_fields(overrides, message):
    values = {
        "model_id": "qwen3:4b-instruct",
        "lora_id": "base",
        "prompt_template_version": "v1",
        "dtype": "int8",
        "layout_version": "toy-one-byte-v1",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        CacheBuildConfig(**values)


def test_cache_build_config_normalizes_known_cache_method_strings():
    cfg = CacheBuildConfig(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        dtype="int8",
        layout_version="toy-one-byte-v1",
        cache_method="kv_packet",
    )

    assert cfg.cache_method is CacheGenerationMethod.KV_PACKET


def test_cache_build_config_preserves_non_empty_custom_cache_method_strings():
    cfg = CacheBuildConfig(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        dtype="int8",
        layout_version="toy-one-byte-v1",
        cache_method="vendor_custom_method",
    )

    assert cfg.cache_method == "vendor_custom_method"


def test_cache_generation_result_normalizes_public_fields(tmp_path):
    ref = result_ref(tmp_path)
    training_artifacts = TrainingArtifacts(adapter_ids=("packet-adapter",))

    result = CacheGenerationResult(
        refs=[ref],  # type: ignore[arg-type]
        document_ids=["doc-a"],  # type: ignore[arg-type]
        chunk_count=1,
        total_bytes=ref.byte_length,
        training_artifacts=training_artifacts,
        cache_method="kv_packet",
    )

    assert result.refs == (ref,)
    assert result.document_ids == ("doc-a",)
    assert result.training_artifacts is training_artifacts
    assert result.cache_method is CacheGenerationMethod.KV_PACKET
    assert result.adapter_ids == ("packet-adapter",)


def test_cache_generation_result_derives_document_id_order_from_refs(tmp_path):
    first_ref = result_ref(tmp_path, document_id="doc-a", chunk_id="section-1", filename="doc-a-1.kvpack")
    second_ref = result_ref(tmp_path, document_id="doc-b", chunk_id="section-1", filename="doc-b-1.kvpack")
    repeated_first_ref = result_ref(tmp_path, document_id="doc-a", chunk_id="section-2", filename="doc-a-2.kvpack")

    result = CacheGenerationResult(
        refs=(first_ref, second_ref, repeated_first_ref),
        document_ids=("doc-a", "doc-b"),
        chunk_count=3,
        total_bytes=first_ref.byte_length + second_ref.byte_length + repeated_first_ref.byte_length,
    )

    assert result.document_ids == ("doc-a", "doc-b")


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"refs": "not-refs"}, "refs"),
        ({"refs": [object()]}, "refs entries"),
        ({"document_ids": "doc-a"}, "document_ids"),
        ({"document_ids": [""]}, "document_ids"),
        ({"document_ids": ("doc-b",)}, "document_ids must match refs document_id order"),
        ({"document_ids": ("doc-a", "doc-a")}, "document_ids must match refs document_id order"),
        ({"chunk_count": -1}, "chunk_count"),
        ({"chunk_count": True}, "chunk_count"),
        ({"chunk_count": 2}, "chunk_count must match"),
        ({"total_bytes": -1}, "total_bytes"),
        ({"total_bytes": True}, "total_bytes"),
        ({"total_bytes_delta": 1}, "total_bytes must match"),
        ({"training_artifacts": object()}, "training_artifacts"),
        ({"cache_method": ""}, "cache_method"),
    ),
)
def test_cache_generation_result_rejects_invalid_public_fields(tmp_path, overrides, message):
    ref = result_ref(tmp_path)
    values = {
        "refs": (ref,),
        "document_ids": ("doc-a",),
        "chunk_count": 1,
        "total_bytes": ref.byte_length,
    }
    if "total_bytes_delta" in overrides:
        overrides = dict(overrides)
        values["total_bytes"] = ref.byte_length + overrides.pop("total_bytes_delta")
    values.update(overrides)

    with pytest.raises((TypeError, ValueError), match=message):
        CacheGenerationResult(**values)


def test_cache_generation_result_allows_empty_generation_result():
    result = CacheGenerationResult(
        refs=(),
        document_ids=(),
        chunk_count=0,
        total_bytes=0,
        cache_method="vendor_custom_method",
    )

    assert result.refs == ()
    assert result.document_ids == ()
    assert result.cache_method == "vendor_custom_method"


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


def test_source_document_rejects_duplicate_chunk_identities():
    first = SourceChunk(chunk_id="section-1", text="first")
    duplicate = SourceChunk(chunk_id="section-1", text="second")

    with pytest.raises(ValueError, match="duplicate chunk identities") as exc_info:
        SourceDocument(document_id="doc-a", chunks=(first, duplicate))

    assert "document_chunk:section-1" in str(exc_info.value)


def test_source_document_allows_same_chunk_id_for_different_chunk_types():
    static = SourceChunk(chunk_id="shared", text="profile", chunk_type=DocumentChunkType.DOCUMENT_STATIC)
    body = SourceChunk(chunk_id="shared", text="body", chunk_type=DocumentChunkType.DOCUMENT_CHUNK)

    document = SourceDocument(document_id="doc-a", chunks=(static, body))

    assert document.chunks == (static, body)


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
        static_chunk_id="profile",
        chunks={"p1": "body"},
        metadata={"title": "Doc A"},
        chunk_metadata={"p1": {"source": "review", "rank": "1"}},
        static_chunk_metadata={"source": "profile"},
    )

    assert [chunk.chunk_id for chunk in document.chunks] == ["profile", "p1"]
    assert document.chunks[0].chunk_type == DocumentChunkType.DOCUMENT_STATIC
    assert document.chunks[0].metadata == {"source": "profile"}
    assert document.chunks[1].metadata == {"source": "review", "rank": "1"}
    assert document.metadata == {"title": "Doc A"}

    with pytest.raises(TypeError, match="chunks must be a mapping"):
        SourceDocument.from_texts(document_id="doc-a", chunks=["body"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        SourceDocument.from_texts(document_id="doc-a", chunks={"p1": "body"}, metadata=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="chunk_metadata must be a mapping"):
        SourceDocument.from_texts(
            document_id="doc-a",
            chunks={"p1": "body"},
            chunk_metadata=[],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="chunk_metadata keys must be non-empty strings"):
        SourceDocument.from_texts(
            document_id="doc-a",
            chunks={"p1": "body"},
            chunk_metadata={"": {"source": "review"}},
        )
    with pytest.raises(TypeError, match="chunk_metadata.p1 must be a mapping"):
        SourceDocument.from_texts(
            document_id="doc-a",
            chunks={"p1": "body"},
            chunk_metadata={"p1": []},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="chunk_metadata.p1.source must be a string"):
        SourceDocument.from_texts(
            document_id="doc-a",
            chunks={"p1": "body"},
            chunk_metadata={"p1": {"source": 1}},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="chunk_metadata contains unknown chunk ids: missing"):
        SourceDocument.from_texts(
            document_id="doc-a",
            chunks={"p1": "body"},
            chunk_metadata={"missing": {"source": "review"}},
        )
    with pytest.raises(ValueError, match="static_chunk_metadata requires static_text"):
        SourceDocument.from_texts(
            document_id="doc-a",
            chunks={"p1": "body"},
            static_chunk_metadata={"source": "profile"},
        )
    with pytest.raises(ValueError, match="static_chunk_metadata.source must be a string"):
        SourceDocument.from_texts(
            document_id="doc-a",
            static_text="static context",
            chunks={"p1": "body"},
            static_chunk_metadata={"source": 1},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="chunk_id must be a non-empty string"):
        SourceDocument.from_texts(
            document_id="doc-a",
            static_text="body",
            static_chunk_id="",
            chunks={"p1": "body"},
        )


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


def test_workflow_with_storage_wires_routed_reader_and_local_cache(tmp_path):
    manifest = InMemoryManifestStore()
    workflow = DocumentKVWorkflow.with_storage(
        manifest=manifest,
        cpu_cache_bytes=1,
        local_cache_dir=tmp_path / "local-cache",
        local_cache_bytes=4096,
    )
    document = SourceDocument.from_texts(
        document_id="doc-a",
        static_text="static context",
        chunks={"section-1": "hello world"},
    )

    workflow.generate_cache(
        documents=(document,),
        generator=EchoGenerator(),
        config=config(),
        shard_uri=tmp_path / "factory-cache.kvpack",
        align_bytes=1,
    )
    first = workflow.prepare(request_for("doc-a"))
    second = workflow.prepare(request_for("doc-a"))

    assert b"doc-a:static:static context" in first.payload
    assert b"doc-a:section-1:hello world" in first.payload
    assert first.segment_tiers == (CacheTier.COLD_STORAGE, CacheTier.COLD_STORAGE)
    assert second.payload == first.payload
    assert second.segment_tiers == (CacheTier.LOCAL_DISK, CacheTier.LOCAL_DISK)
    assert workflow.materializer.cache.stats().cold_misses == 2
    assert workflow.materializer.cache.stats().local_hits == 2


def test_workflow_with_storage_generates_relative_shards_under_disk_root(tmp_path):
    disk_root = tmp_path / "disk-root"
    manifest = InMemoryManifestStore()
    workflow = DocumentKVWorkflow.with_storage(
        manifest=manifest,
        cpu_cache_bytes=4096,
        disk_root=disk_root,
    )
    document = SourceDocument.from_text(document_id="doc-a", text="hello from disk root")
    request = DocumentKVRequest.for_text_document(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_id="doc-a",
    )

    workflow.generate_cache(
        documents=(document,),
        generator=EchoGenerator(),
        config=config(),
        shard_uri="relative-disk-cache.kvpack",
        align_bytes=1,
    )
    materialized = workflow.prepare(request)

    assert (disk_root / "relative-disk-cache.kvpack").exists()
    assert materialized.payload == b"doc-a:document:hello from disk root"


def test_workflow_with_storage_generates_relative_shards_under_uc_volume_root(tmp_path):
    uc_volume_root = tmp_path / "Volumes" / "catalog" / "schema" / "volume"
    manifest = InMemoryManifestStore()
    workflow = DocumentKVWorkflow.with_storage(
        manifest=manifest,
        cpu_cache_bytes=4096,
        uc_volume_root=uc_volume_root,
    )
    document = SourceDocument.from_text(document_id="doc-a", text="hello from uc root")
    request = DocumentKVRequest.for_text_document(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_id="doc-a",
    )

    workflow.generate_cache(
        documents=(document,),
        generator=EchoGenerator(),
        config=config(),
        shard_uri="relative-uc-cache.kvpack",
        align_bytes=1,
    )
    materialized = workflow.prepare(request)

    assert (uc_volume_root / "relative-uc-cache.kvpack").exists()
    assert materialized.payload == b"doc-a:document:hello from uc root"


def test_workflow_with_storage_generates_memory_shards_in_process(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = InMemoryManifestStore()
    workflow = DocumentKVWorkflow.with_storage(manifest=manifest, cpu_cache_bytes=4096)
    document = SourceDocument.from_text(document_id="doc-a", text="hello from memory")
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
        shard_uri="memory:doc-a",
        align_bytes=1,
    )
    materialized = workflow.prepare(request)

    assert result.refs[0].shard_uri == "memory:doc-a"
    assert materialized.payload == b"doc-a:document:hello from memory"
    assert not (tmp_path / "memory:doc-a").exists()


def test_workflow_with_storage_generates_mem_alias_shards_in_process(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = InMemoryManifestStore()
    workflow = DocumentKVWorkflow.with_storage(manifest=manifest, cpu_cache_bytes=4096)
    document = SourceDocument.from_text(document_id="doc-a", text="hello from mem alias")

    result = workflow.generate_cache(
        documents=(document,),
        generator=EchoGenerator(),
        config=config(),
        shard_uri="mem:doc-a",
        align_bytes=1,
    )
    materialized = workflow.prepare(
        DocumentKVRequest.for_text_document(
            request_id="req-1",
            task_id="qa",
            model_id="qwen3:4b-instruct",
            lora_id="base",
            prompt_template_version="v1",
            document_id="doc-a",
        )
    )

    assert result.refs[0].shard_uri == "mem:doc-a"
    assert materialized.payload == b"doc-a:document:hello from mem alias"
    assert not (tmp_path / "mem:doc-a").exists()


def test_workflow_with_storage_populates_injected_service_memory_reader(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = InMemoryManifestStore()
    service = DocumentKVService(
        planner=CachePlanner(manifest),
        materializer=KVMaterializer(
            cache=ChunkCache(cpu_max_bytes=4096),
            reader=RoutedRangeReader(memory=MemoryRangeReader()),
        ),
        admission_queue=AdmissionQueue(max_pending_gpu_bytes=4096),
    )
    workflow = DocumentKVWorkflow.with_storage(
        manifest=manifest,
        cpu_cache_bytes=4096,
        service=service,
    )
    document = SourceDocument.from_text(document_id="doc-a", text="service memory")

    workflow.generate_cache(
        documents=(document,),
        generator=EchoGenerator(),
        config=config(),
        shard_uri="memory:service-doc-a",
        align_bytes=1,
    )
    materialized = workflow.prepare(
        DocumentKVRequest.for_text_document(
            request_id="req-1",
            task_id="qa",
            model_id="qwen3:4b-instruct",
            lora_id="base",
            prompt_template_version="v1",
            document_id="doc-a",
        )
    )

    assert materialized.payload == b"doc-a:document:service memory"
    assert materialized.segment_tiers == (CacheTier.COLD_STORAGE,)
    assert not (tmp_path / "memory:service-doc-a").exists()


def test_workflow_with_storage_rejects_memory_shard_when_injected_service_reader_cannot_load_it(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    manifest = InMemoryManifestStore()
    service = DocumentKVService(
        planner=CachePlanner(manifest),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
        admission_queue=AdmissionQueue(max_pending_gpu_bytes=4096),
    )
    workflow = DocumentKVWorkflow.with_storage(
        manifest=manifest,
        cpu_cache_bytes=4096,
        service=service,
    )
    document = SourceDocument.from_text(document_id="doc-a", text="service disk reader")

    with pytest.raises(ValueError, match="active materializer"):
        workflow.generate_cache(
            documents=(document,),
            generator=EchoGenerator(),
            config=config(),
            shard_uri="memory:service-doc-a",
            align_bytes=1,
        )

    assert not (tmp_path / "memory:service-doc-a").exists()


def test_workflow_with_storage_copies_preloaded_memory_blobs_to_injected_service_reader():
    chunk = PackChunk(
        key=KVCacheKey.for_document(
            model_id="qwen3:4b-instruct",
            lora_id="base",
            prompt_template_version="v1",
            document_id="doc-a",
            chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
            chunk_id="section-1",
        ),
        payload=b"preloaded-service-memory",
        token_count=len(b"preloaded-service-memory"),
        dtype="int8",
        layout_version="toy-one-byte-v1",
    )
    payload, refs = write_kvpack_bytes("memory:preloaded-service", [chunk], align_bytes=1)
    manifest = InMemoryManifestStore(refs)
    service = DocumentKVService(
        planner=CachePlanner(manifest),
        materializer=KVMaterializer(
            cache=ChunkCache(cpu_max_bytes=4096),
            reader=RoutedRangeReader(memory=MemoryRangeReader()),
        ),
        admission_queue=AdmissionQueue(max_pending_gpu_bytes=4096),
    )
    workflow = DocumentKVWorkflow.with_storage(
        manifest=manifest,
        cpu_cache_bytes=4096,
        memory_blobs={"memory:preloaded-service": payload},
        service=service,
    )
    request = DocumentKVRequest(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_chunks={"doc-a": ["section-1"]},
        include_static=False,
    )

    materialized = workflow.prepare(request)

    assert materialized.payload == b"preloaded-service-memory"


def test_manual_workflow_generates_memory_shards_when_materializer_can_read_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = InMemoryManifestStore()
    memory_reader = MemoryRangeReader()
    workflow = DocumentKVWorkflow(
        manifest=manifest,
        materializer=KVMaterializer(
            cache=ChunkCache(cpu_max_bytes=4096),
            reader=RoutedRangeReader(memory=memory_reader),
        ),
    )
    document = SourceDocument.from_text(document_id="doc-a", text="manual memory")

    workflow.generate_cache(
        documents=(document,),
        generator=EchoGenerator(),
        config=config(),
        shard_uri="memory:doc-a",
        align_bytes=1,
    )
    materialized = workflow.prepare(
        DocumentKVRequest.for_text_document(
            request_id="req-1",
            task_id="qa",
            model_id="qwen3:4b-instruct",
            lora_id="base",
            prompt_template_version="v1",
            document_id="doc-a",
        )
    )

    assert materialized.payload == b"doc-a:document:manual memory"
    assert not (tmp_path / "memory:doc-a").exists()


def test_manual_workflow_rejects_explicit_memory_writer_when_active_materializer_cannot_read_memory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
        memory_writer=MemoryRangeReader(),
    )
    document = SourceDocument.from_text(document_id="doc-a", text="explicit writer")

    with pytest.raises(ValueError, match="active materializer"):
        workflow.generate_cache(
            documents=(document,),
            generator=EchoGenerator(),
            config=config(),
            shard_uri="memory:doc-a",
            align_bytes=1,
        )

    assert not (tmp_path / "memory:doc-a").exists()


def test_manual_workflow_rejects_memory_shard_uri_without_memory_storage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )
    document = SourceDocument.from_text(document_id="doc-a", text="manual memory")

    with pytest.raises(ValueError, match="active materializer"):
        workflow.generate_cache(
            documents=(document,),
            generator=EchoGenerator(),
            config=config(),
            shard_uri="memory:doc-a",
            align_bytes=1,
        )

    assert not (tmp_path / "memory:doc-a").exists()


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
        static_chunk_id="profile",
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
        static_chunk_id="profile",
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
    assert b"doc-a:profile:profile" in materialized.payload
    assert b"doc-a:review-2:second review" in materialized.payload
    assert b"doc-a:review-1:first review" not in materialized.payload


def test_workflow_generates_and_prepares_multi_document_selection(tmp_path):
    manifest = InMemoryManifestStore()
    workflow = DocumentKVWorkflow(
        manifest=manifest,
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )
    documents = (
        SourceDocument.from_texts(
            document_id="doc-a",
            static_text="profile a",
            static_chunk_id="profile",
            chunks={"review-1": "first a", "review-2": "second a"},
        ),
        SourceDocument.from_texts(
            document_id="doc-b",
            static_text="profile b",
            static_chunk_id="profile",
            chunks={"review-1": "first b", "review-2": "second b"},
        ),
    )
    request = DocumentKVRequest.for_document_selection(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_chunks={"doc-a": ("review-2",), "doc-b": ("review-1",)},
        static_chunk_id="profile",
    )

    result = workflow.generate_cache(
        documents=documents,
        generator=EchoGenerator(),
        config=config(),
        shard_uri=tmp_path / "multi-document-selection.kvpack",
        align_bytes=1,
    )
    materialized = workflow.prepare(request)

    assert result.document_ids == ("doc-a", "doc-b")
    assert request.selected_document_ids == ("doc-a", "doc-b")
    assert b"doc-a:profile:profile a" in materialized.payload
    assert b"doc-a:review-2:second a" in materialized.payload
    assert b"doc-b:profile:profile b" in materialized.payload
    assert b"doc-b:review-1:first b" in materialized.payload
    assert b"doc-a:review-1:first a" not in materialized.payload
    assert b"doc-b:review-2:second b" not in materialized.payload


def test_workflow_rejects_duplicate_generation_document_ids_before_training(tmp_path):
    trainer = RecordingTrainer()
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )
    documents = (
        SourceDocument.from_text(document_id="doc-a", text="first"),
        SourceDocument.from_text(document_id="doc-a", text="second"),
    )

    with pytest.raises(ValueError, match="documents contain duplicate document ids: doc-a"):
        workflow.generate_cache(
            documents=documents,
            generator=EchoGenerator(),
            config=config(),
            shard_uri=tmp_path / "duplicate-documents.kvpack",
            trainer=trainer,
            align_bytes=1,
        )

    assert trainer.document_ids == ()
    assert not (tmp_path / "duplicate-documents.kvpack").exists()
    assert workflow.manifest.keys_for_document("doc-a") == []


def test_workflow_rejects_non_source_document_generation_entries(tmp_path):
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )

    with pytest.raises(TypeError, match="documents entries must be SourceDocument"):
        workflow.generate_cache(
            documents=(object(),),  # type: ignore[arg-type]
            generator=EchoGenerator(),
            config=config(),
            shard_uri=tmp_path / "bad-document-entry.kvpack",
            align_bytes=1,
        )


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


def test_workflow_prepares_and_submits_engine_ready_request(tmp_path):
    manifest = InMemoryManifestStore()
    materializer = KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader())
    workflow = DocumentKVWorkflow(manifest=manifest, materializer=materializer)
    connector = RecordingConnector()
    document = SourceDocument.from_texts(document_id="doc-a", static_text="static", chunks={"section-1": "body"})

    result = workflow.generate_cache(
        documents=(document,),
        generator=ByteAlignedGenerator(),
        config=config(cache_method=CacheGenerationMethod.KV_PACKET),
        shard_uri=tmp_path / "submit-cache.kvpack",
        trainer=ArtifactTrainer(),
        align_bytes=1,
    )
    ready = workflow.prepare_and_submit_to_engine(
        request_for("doc-a"),
        connector=connector,
        layout=one_byte_layout(),
        cache_method=result.cache_method,
        training_artifacts=result.training_artifacts,
        segmented=True,
    )

    assert connector.submitted == [ready]
    assert ready.payload == (
        b"doc-a:static:static|qwen3:4b-instruct-kv-packet",
        b"doc-a:section-1:body|qwen3:4b-instruct-kv-packet",
    )
    assert ready.handle.cache_method == "kv_packet"
    assert ready.handle.adapter_ids == ("qwen3:4b-instruct-kv-packet",)
    assert connector.released == []


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


def test_workflow_preserves_custom_cache_method_into_engine_handle(tmp_path):
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(cache=ChunkCache(cpu_max_bytes=4096), reader=DiskRangeReader()),
    )
    document = SourceDocument.from_texts(document_id="doc-a", static_text="static", chunks={"section-1": "body"})

    result = workflow.generate_cache(
        documents=(document,),
        generator=ByteAlignedGenerator(),
        config=CacheBuildConfig(
            model_id="qwen3:4b-instruct",
            lora_id="base",
            prompt_template_version="v1",
            dtype="int8",
            layout_version="toy-one-byte-v1",
            cache_method="vendor_custom_method",
        ),
        shard_uri=tmp_path / "custom-method-cache.kvpack",
        align_bytes=1,
    )
    ready = workflow.prepare_for_engine(
        request_for("doc-a"),
        layout=one_byte_layout(),
        cache_method=result.cache_method,
    )

    assert result.cache_method == "vendor_custom_method"
    assert ready.handle.cache_method == "vendor_custom_method"


def test_training_artifacts_validate_adapter_artifact_identity():
    adapter = CacheAdapterArtifact(adapter_id="packet-adapter", artifact_uri="/Volumes/cache/packet.safetensors")

    with pytest.raises(ValueError, match="adapter_ids must match"):
        TrainingArtifacts(adapter_ids=("other-adapter",), adapter_artifacts=(adapter,))

    with pytest.raises(ValueError, match="adapter_ids entries must be unique"):
        TrainingArtifacts(adapter_ids=("packet-adapter", "packet-adapter"))

    with pytest.raises(ValueError, match="adapter_artifacts adapter_id entries must be unique"):
        TrainingArtifacts(
            adapter_artifacts=(
                adapter,
                CacheAdapterArtifact(
                    adapter_id="packet-adapter",
                    artifact_uri="/Volumes/cache/packet-rerun.safetensors",
                ),
            )
        )


def test_training_artifact_metadata_uses_string_maps():
    artifact = CacheAdapterArtifact(
        adapter_id="packet-adapter",
        artifact_uri="/Volumes/cache/packet.safetensors",
        metadata={"rank": "8"},
    )
    training_artifacts = TrainingArtifacts(
        adapter_artifacts=(artifact,),
        metadata={"trained": "true"},
    )

    assert artifact.metadata == {"rank": "8"}
    assert training_artifacts.metadata == {"trained": "true"}

    with pytest.raises(TypeError, match="metadata must be a mapping"):
        CacheAdapterArtifact(
            adapter_id="packet-adapter",
            artifact_uri="/Volumes/cache/packet.safetensors",
            metadata=[],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="metadata keys must be non-empty strings"):
        TrainingArtifacts(adapter_ids=("packet-adapter",), metadata={"": "true"})
    with pytest.raises(ValueError, match="metadata.rank must be a string"):
        CacheAdapterArtifact(
            adapter_id="packet-adapter",
            artifact_uri="/Volumes/cache/packet.safetensors",
            metadata={"rank": 8},  # type: ignore[dict-item]
        )


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

    with pytest.raises(ValueError, match="adapter_ids entries must be unique"):
        workflow.prepare_for_engine(
            request_for("doc-a"),
            layout=one_byte_layout(),
            adapter_ids=("packet-adapter", "packet-adapter"),
        )
