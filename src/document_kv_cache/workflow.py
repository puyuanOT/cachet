"""End-to-end document cache generation and serving workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from document_kv_cache.cache import ChunkCache
from document_kv_cache.engine import (
    EngineReadyRequest,
    KVLayout,
    ServingEngineConnector,
    _normalize_gpu_byte_multiplier,
    build_engine_ready_request,
)
from document_kv_cache.engine_protocol import KVStorageLayout, kv_storage_layout_from_value
from document_kv_cache.kvpack import PackChunk, write_kvpack, write_kvpack_bytes
from document_kv_cache.manifest import ManifestStore
from document_kv_cache.materializer import KVMaterializer, MaterializedKV, SegmentedMaterializedKV
from document_kv_cache.models import (
    DEFAULT_STATIC_CHUNK_ID,
    CacheGenerationMethod,
    ChunkRef,
    DocumentChunkType,
    DocumentKVRequest,
)
from document_kv_cache.planner import CachePlanner
from document_kv_cache.service import DocumentKVService
from document_kv_cache.storage import (
    DiskRangeReader,
    MemoryRangeReader,
    RoutedRangeReader,
    UnityCatalogVolumeRangeReader,
    local_path,
    unity_catalog_volume_path,
)


@dataclass(frozen=True, slots=True)
class SourceChunk:
    chunk_id: str
    text: str
    chunk_type: DocumentChunkType | str = DocumentChunkType.DOCUMENT_CHUNK
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _non_empty_string("chunk_id", self.chunk_id))
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        object.__setattr__(self, "chunk_type", _document_chunk_type(self.chunk_type))
        object.__setattr__(self, "metadata", _metadata_dict("metadata", self.metadata))


@dataclass(frozen=True, slots=True)
class SourceDocument:
    document_id: str
    chunks: tuple[SourceChunk, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _non_empty_string("document_id", self.document_id))
        chunks = tuple(self.chunks)
        if not chunks:
            raise ValueError("chunks must contain at least one SourceChunk")
        for chunk in chunks:
            if not isinstance(chunk, SourceChunk):
                raise TypeError("chunks entries must be SourceChunk instances")
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "metadata", _metadata_dict("metadata", self.metadata))

    @classmethod
    def from_text(
        cls,
        *,
        document_id: str,
        text: str,
        chunk_id: str = "document",
        chunk_type: DocumentChunkType | str = DocumentChunkType.DOCUMENT_CHUNK,
        metadata: Mapping[str, str] | None = None,
        chunk_metadata: Mapping[str, str] | None = None,
    ) -> "SourceDocument":
        return cls(
            document_id=document_id,
            chunks=(
                SourceChunk(
                    chunk_id=chunk_id,
                    text=text,
                    chunk_type=chunk_type,
                    metadata={} if chunk_metadata is None else _metadata_dict("chunk_metadata", chunk_metadata),
                ),
            ),
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_texts(
        cls,
        *,
        document_id: str,
        chunks: Mapping[str, str],
        static_text: str | None = None,
        static_chunk_id: str = DEFAULT_STATIC_CHUNK_ID,
        metadata: Mapping[str, str] | None = None,
        chunk_metadata: Mapping[str, Mapping[str, str]] | None = None,
        static_chunk_metadata: Mapping[str, str] | None = None,
    ) -> "SourceDocument":
        if not isinstance(chunks, Mapping):
            raise TypeError("chunks must be a mapping")
        normalized_chunk_metadata = _chunk_metadata_map(chunk_metadata)
        unknown_metadata_chunk_ids = tuple(chunk_id for chunk_id in normalized_chunk_metadata if chunk_id not in chunks)
        if unknown_metadata_chunk_ids:
            raise ValueError(
                "chunk_metadata contains unknown chunk ids: " + ", ".join(unknown_metadata_chunk_ids)
            )
        if static_text is None and static_chunk_metadata is not None:
            raise ValueError("static_chunk_metadata requires static_text")
        source_chunks: list[SourceChunk] = []
        if static_text is not None:
            source_chunks.append(
                SourceChunk(
                    chunk_id=static_chunk_id,
                    text=static_text,
                    chunk_type=DocumentChunkType.DOCUMENT_STATIC,
                    metadata={} if static_chunk_metadata is None else _metadata_dict(
                        "static_chunk_metadata",
                        static_chunk_metadata,
                    ),
                )
            )
        source_chunks.extend(
            SourceChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=normalized_chunk_metadata.get(chunk_id, {}),
            )
            for chunk_id, text in chunks.items()
        )
        return cls(
            document_id=document_id,
            chunks=tuple(source_chunks),
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True, slots=True)
class CacheBuildConfig:
    model_id: str
    lora_id: str
    prompt_template_version: str
    dtype: str
    layout_version: str
    cache_method: CacheGenerationMethod | str = CacheGenerationMethod.VANILLA_PREFILL
    storage_layout: KVStorageLayout | str = KVStorageLayout.SEPARATE_KEY_VALUE

    def __post_init__(self) -> None:
        for field_name in ("model_id", "lora_id", "prompt_template_version", "dtype", "layout_version"):
            object.__setattr__(self, field_name, _non_empty_string(field_name, getattr(self, field_name)))
        object.__setattr__(self, "cache_method", _cache_generation_method(self.cache_method))
        object.__setattr__(
            self,
            "storage_layout",
            kv_storage_layout_from_value(self.storage_layout, field_name="storage_layout"),
        )


@dataclass(frozen=True, slots=True)
class CacheAdapterArtifact:
    adapter_id: str
    artifact_uri: str
    cache_method: CacheGenerationMethod | str = CacheGenerationMethod.ADAPTER_TRAINED
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        adapter_id = _non_empty_string("adapter_id", self.adapter_id)
        artifact_uri = _non_empty_string("artifact_uri", self.artifact_uri)
        cache_method = _cache_method_value(self.cache_method)
        if not cache_method:
            raise ValueError("cache_method must be non-empty")
        object.__setattr__(self, "adapter_id", adapter_id)
        object.__setattr__(self, "artifact_uri", artifact_uri)
        object.__setattr__(self, "cache_method", cache_method)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    adapter_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, str] = field(default_factory=dict)
    adapter_artifacts: tuple[CacheAdapterArtifact, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        adapter_ids = _non_empty_string_tuple("adapter_ids", self.adapter_ids)
        adapter_artifacts = tuple(self.adapter_artifacts)
        for artifact in adapter_artifacts:
            if not isinstance(artifact, CacheAdapterArtifact):
                raise TypeError("adapter_artifacts entries must be CacheAdapterArtifact instances")
        if adapter_artifacts:
            artifact_ids = tuple(artifact.adapter_id for artifact in adapter_artifacts)
            if adapter_ids and adapter_ids != artifact_ids:
                raise ValueError("adapter_ids must match adapter_artifacts adapter_id order")
            adapter_ids = artifact_ids
        object.__setattr__(self, "adapter_ids", adapter_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "adapter_artifacts", adapter_artifacts)


class TrainingAdapter(Protocol):
    def fit(self, documents: Sequence[SourceDocument], config: CacheBuildConfig) -> TrainingArtifacts: ...


class KVChunkGenerator(Protocol):
    def generate(
        self,
        *,
        document: SourceDocument,
        chunk: SourceChunk,
        config: CacheBuildConfig,
        training_artifacts: TrainingArtifacts | None = None,
    ) -> PackChunk: ...


@dataclass(frozen=True, slots=True)
class CacheGenerationResult:
    refs: tuple[ChunkRef, ...]
    document_ids: tuple[str, ...]
    chunk_count: int
    total_bytes: int
    training_artifacts: TrainingArtifacts | None = None
    cache_method: CacheGenerationMethod | str = CacheGenerationMethod.VANILLA_PREFILL

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        if self.training_artifacts is None:
            return ()
        return self.training_artifacts.adapter_ids


class DocumentKVWorkflow:
    def __init__(
        self,
        *,
        manifest: ManifestStore,
        materializer: KVMaterializer,
        planner: CachePlanner | None = None,
        service: DocumentKVService | None = None,
        shard_path_resolver: Callable[[str], Path] | None = None,
        memory_writer: MemoryRangeReader | None = None,
        memory_writers: Sequence[MemoryRangeReader] = (),
    ) -> None:
        self.manifest = manifest
        self.planner = planner or CachePlanner(manifest)
        self.materializer = materializer
        self.service = service
        self.shard_path_resolver = shard_path_resolver
        inferred_memory_writer = _active_memory_reader_for(materializer, service)
        self._memory_generation_supported = inferred_memory_writer is not None
        self.memory_writers = _dedupe_memory_writers(
            tuple(
                writer
                for writer in (inferred_memory_writer, memory_writer, *memory_writers)
                if writer is not None
            )
        )
        self.memory_writer = self.memory_writers[0] if self.memory_writers else None

    @classmethod
    def with_storage(
        cls,
        *,
        manifest: ManifestStore,
        cpu_cache_bytes: int = 0,
        local_cache_dir: str | Path | None = None,
        local_cache_bytes: int | None = None,
        disk_root: str | Path | None = None,
        uc_volume_root: str | Path | None = None,
        memory_blobs: Mapping[str, bytes] | None = None,
        planner: CachePlanner | None = None,
        service: DocumentKVService | None = None,
    ) -> "DocumentKVWorkflow":
        """Build a workflow with the standard memory/disk/UC reader stack."""
        shard_path_resolver = _storage_shard_path_resolver(
            disk_root=disk_root,
            uc_volume_root=uc_volume_root,
        )
        memory_reader = MemoryRangeReader(memory_blobs)
        service_memory_reader = _memory_reader_for_materializer(service.materializer) if service is not None else None
        if memory_blobs is not None and service_memory_reader is not None:
            for shard_uri, payload in memory_blobs.items():
                service_memory_reader.put(shard_uri, payload)
        return cls(
            manifest=manifest,
            planner=planner,
            materializer=KVMaterializer(
                cache=ChunkCache(
                    cpu_max_bytes=cpu_cache_bytes,
                    local_dir=local_cache_dir,
                    local_max_bytes=local_cache_bytes,
                ),
                reader=RoutedRangeReader(
                    memory=memory_reader,
                    disk=DiskRangeReader(root=disk_root),
                    unity_catalog=UnityCatalogVolumeRangeReader(volume_root=uc_volume_root),
                ),
            ),
            service=service,
            shard_path_resolver=shard_path_resolver,
            memory_writers=_memory_writers_for(
                memory_reader,
                service=service,
                service_memory_reader=service_memory_reader,
            ),
        )

    def generate_cache(
        self,
        *,
        documents: Sequence[SourceDocument],
        generator: KVChunkGenerator,
        config: CacheBuildConfig,
        shard_uri: str | Path,
        trainer: TrainingAdapter | None = None,
        align_bytes: int = 4096,
    ) -> CacheGenerationResult:
        training_artifacts = trainer.fit(documents, config) if trainer is not None else None
        cache_method = _effective_cache_method(config, trainer)
        pack_chunks = tuple(self._iter_pack_chunks(documents, generator, config, training_artifacts))
        refs = self._write_pack_chunks(shard_uri, pack_chunks, align_bytes=align_bytes)
        self.manifest.put_many(refs)
        return CacheGenerationResult(
            refs=refs,
            document_ids=tuple(document.document_id for document in documents),
            chunk_count=len(pack_chunks),
            total_bytes=sum(ref.byte_length for ref in refs),
            training_artifacts=training_artifacts,
            cache_method=cache_method,
        )

    def prepare(self, request: DocumentKVRequest, *, segmented: bool = False) -> MaterializedKV | SegmentedMaterializedKV:
        planner, materializer = self._preparation_dependencies()
        plan = planner.build_plan(request)
        if segmented:
            return materializer.materialize_segmented(plan)
        return materializer.materialize(plan)

    def prepare_and_enqueue(self, request: DocumentKVRequest) -> bool:
        return self._require_service().prepare_and_enqueue(request)

    def prepare_for_engine(
        self,
        request: DocumentKVRequest,
        *,
        layout: KVLayout,
        handle_uri: str | None = None,
        metadata: Mapping[str, str] | None = None,
        cache_method: CacheGenerationMethod | str = CacheGenerationMethod.VANILLA_PREFILL,
        adapter_ids: tuple[str, ...] = (),
        training_artifacts: TrainingArtifacts | None = None,
        segmented: bool = False,
        kv_gpu_bytes_per_payload_byte: float | None = None,
    ) -> EngineReadyRequest:
        gpu_byte_multiplier = self._engine_gpu_byte_multiplier(kv_gpu_bytes_per_payload_byte)
        engine_adapter_ids = _engine_adapter_ids(adapter_ids, training_artifacts)
        planner, materializer = self._preparation_dependencies()
        plan = planner.build_plan(request)
        materialized = materializer.materialize_segmented(plan) if segmented else materializer.materialize(plan)
        return build_engine_ready_request(
            materialized,
            layout=layout,
            handle_uri=handle_uri,
            metadata=metadata,
            cache_method=cache_method,
            adapter_ids=engine_adapter_ids,
            kv_gpu_bytes_per_payload_byte=gpu_byte_multiplier,
        )

    def prepare_and_submit_to_engine(
        self,
        request: DocumentKVRequest,
        *,
        connector: ServingEngineConnector,
        layout: KVLayout,
        handle_uri: str | None = None,
        metadata: Mapping[str, str] | None = None,
        cache_method: CacheGenerationMethod | str = CacheGenerationMethod.VANILLA_PREFILL,
        adapter_ids: tuple[str, ...] = (),
        training_artifacts: TrainingArtifacts | None = None,
        segmented: bool = False,
        kv_gpu_bytes_per_payload_byte: float | None = None,
    ) -> EngineReadyRequest:
        ready = self.prepare_for_engine(
            request,
            layout=layout,
            handle_uri=handle_uri,
            metadata=metadata,
            cache_method=cache_method,
            adapter_ids=adapter_ids,
            training_artifacts=training_artifacts,
            segmented=segmented,
            kv_gpu_bytes_per_payload_byte=kv_gpu_bytes_per_payload_byte,
        )
        connector.submit(ready)
        return ready

    def _require_service(self) -> DocumentKVService:
        if self.service is None:
            raise RuntimeError("DocumentKVWorkflow was created without a DocumentKVService")
        return self.service

    def _engine_gpu_byte_multiplier(self, override: float | None) -> float:
        if override is not None:
            multiplier = override
        elif self.service is not None:
            multiplier = self.service.kv_gpu_bytes_per_payload_byte
        else:
            multiplier = 1.0
        return _normalize_gpu_byte_multiplier(multiplier)

    def _preparation_dependencies(self) -> tuple[CachePlanner, KVMaterializer]:
        if self.service is not None:
            return self.service.planner, self.service.materializer
        return self.planner, self.materializer

    def _iter_pack_chunks(
        self,
        documents: Sequence[SourceDocument],
        generator: KVChunkGenerator,
        config: CacheBuildConfig,
        training_artifacts: TrainingArtifacts | None,
    ) -> Iterable[PackChunk]:
        for document in documents:
            for chunk in document.chunks:
                pack_chunk = generator.generate(
                    document=document,
                    chunk=chunk,
                    config=config,
                    training_artifacts=training_artifacts,
                )
                self._validate_pack_chunk(document, chunk, config, pack_chunk)
                yield pack_chunk

    def _write_pack_chunks(
        self,
        shard_uri: str | Path,
        pack_chunks: Sequence[PackChunk],
        *,
        align_bytes: int,
    ) -> tuple[ChunkRef, ...]:
        shard_uri_text = str(shard_uri)
        if _is_memory_storage_uri(shard_uri_text):
            if not self._memory_generation_supported:
                raise ValueError("memory shard URIs require the active materializer to use memory storage")
            if not self.memory_writers:
                raise ValueError("memory shard URIs require at least one memory writer")
            payload, refs = write_kvpack_bytes(shard_uri_text, pack_chunks, align_bytes=align_bytes)
            for memory_writer in self.memory_writers:
                memory_writer.put(shard_uri_text, payload)
            return tuple(refs)
        return tuple(
            write_kvpack(
                shard_uri,
                pack_chunks,
                align_bytes=align_bytes,
                path_resolver=self.shard_path_resolver,
            )
        )

    @staticmethod
    def _validate_pack_chunk(
        document: SourceDocument,
        chunk: SourceChunk,
        config: CacheBuildConfig,
        pack_chunk: PackChunk,
    ) -> None:
        key = pack_chunk.key
        expected = {
            "model_id": config.model_id,
            "lora_id": config.lora_id,
            "prompt_template_version": config.prompt_template_version,
            "document_id": document.document_id,
            "chunk_type": chunk.chunk_type.value,
            "chunk_id": chunk.chunk_id,
            "dtype": config.dtype,
            "layout_version": config.layout_version,
            "storage_layout": config.storage_layout.value,
        }
        actual = {
            "model_id": key.model_id,
            "lora_id": key.lora_id,
            "prompt_template_version": key.prompt_template_version,
            "document_id": key.document_id,
            "chunk_type": key.chunk_type.value,
            "chunk_id": key.chunk_id,
            "dtype": pack_chunk.dtype,
            "layout_version": pack_chunk.layout_version,
            "storage_layout": pack_chunk.storage_layout.value,
        }
        mismatches = [name for name, value in expected.items() if actual[name] != value]
        if mismatches:
            details = ", ".join(f"{name}: expected {expected[name]!r}, got {actual[name]!r}" for name in mismatches)
            raise ValueError(f"Generated chunk does not match source/config ({details})")


def _effective_cache_method(config: CacheBuildConfig, trainer: TrainingAdapter | None) -> CacheGenerationMethod | str:
    if trainer is not None and config.cache_method == CacheGenerationMethod.VANILLA_PREFILL:
        return CacheGenerationMethod.ADAPTER_TRAINED
    return config.cache_method


def _engine_adapter_ids(
    adapter_ids: tuple[str, ...],
    training_artifacts: TrainingArtifacts | None,
) -> tuple[str, ...]:
    explicit_adapter_ids = _non_empty_string_tuple("adapter_ids", adapter_ids)
    if training_artifacts is None:
        return explicit_adapter_ids
    artifact_adapter_ids = training_artifacts.adapter_ids
    if explicit_adapter_ids and explicit_adapter_ids != artifact_adapter_ids:
        raise ValueError("adapter_ids must match training_artifacts adapter_ids")
    return explicit_adapter_ids or artifact_adapter_ids


def _cache_method_value(cache_method: CacheGenerationMethod | str) -> str:
    if isinstance(cache_method, CacheGenerationMethod):
        return cache_method.value
    return str(cache_method)


def _cache_generation_method(cache_method: CacheGenerationMethod | str) -> CacheGenerationMethod | str:
    if isinstance(cache_method, CacheGenerationMethod):
        return cache_method
    cache_method_text = _non_empty_string("cache_method", cache_method)
    try:
        return CacheGenerationMethod(cache_method_text)
    except ValueError:
        return cache_method_text


def _document_chunk_type(chunk_type: DocumentChunkType | str) -> DocumentChunkType:
    if isinstance(chunk_type, DocumentChunkType):
        return chunk_type
    try:
        return DocumentChunkType(str(chunk_type))
    except ValueError as exc:
        raise ValueError(f"chunk_type must be one of {[chunk_type.value for chunk_type in DocumentChunkType]}") from exc


def _metadata_dict(name: str, metadata: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError(f"{name}.{key} must be a string")
        normalized[key] = value
    return normalized


def _chunk_metadata_map(
    chunk_metadata: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]]:
    if chunk_metadata is None:
        return {}
    if not isinstance(chunk_metadata, Mapping):
        raise TypeError("chunk_metadata must be a mapping")
    normalized: dict[str, dict[str, str]] = {}
    for chunk_id, metadata in chunk_metadata.items():
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("chunk_metadata keys must be non-empty strings")
        normalized[chunk_id] = _metadata_dict(f"chunk_metadata.{chunk_id}", metadata)
    return normalized


def _storage_shard_path_resolver(
    *,
    disk_root: str | Path | None,
    uc_volume_root: str | Path | None,
) -> Callable[[str], Path]:
    def resolve(shard_uri: str) -> Path:
        if shard_uri.startswith("uc-volume:") or shard_uri == "/Volumes" or shard_uri.startswith("/Volumes/"):
            return unity_catalog_volume_path(shard_uri, root=uc_volume_root)
        if uc_volume_root is not None and _is_relative_storage_uri(shard_uri):
            return unity_catalog_volume_path(shard_uri, root=uc_volume_root)
        return local_path(shard_uri, root=disk_root)

    return resolve


def _is_relative_storage_uri(shard_uri: str) -> bool:
    return ":" not in shard_uri and not Path(shard_uri).is_absolute()


def _is_memory_storage_uri(shard_uri: str) -> bool:
    return shard_uri.startswith("memory:") or shard_uri.startswith("mem:")


def _memory_writers_for(
    primary: MemoryRangeReader,
    *,
    service: DocumentKVService | None,
    service_memory_reader: MemoryRangeReader | None,
) -> tuple[MemoryRangeReader, ...]:
    if service is None:
        return (primary,)
    if service_memory_reader is None:
        return ()
    return _dedupe_memory_writers((primary, service_memory_reader))


def _active_memory_reader_for(
    materializer: KVMaterializer,
    service: DocumentKVService | None,
) -> MemoryRangeReader | None:
    if service is not None:
        return _memory_reader_for_materializer(service.materializer)
    return _memory_reader_for_materializer(materializer)


def _memory_reader_for_materializer(materializer: KVMaterializer) -> MemoryRangeReader | None:
    reader = materializer.reader
    if isinstance(reader, MemoryRangeReader):
        return reader
    memory = getattr(reader, "memory", None)
    if isinstance(memory, MemoryRangeReader):
        return memory
    return None


def _dedupe_memory_writers(writers: Sequence[MemoryRangeReader]) -> tuple[MemoryRangeReader, ...]:
    deduped: list[MemoryRangeReader] = []
    seen: set[int] = set()
    for writer in writers:
        writer_id = id(writer)
        if writer_id in seen:
            continue
        seen.add(writer_id)
        deduped.append(writer)
    return tuple(deduped)


def _non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _non_empty_string_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of non-empty strings, not a string")
    tuple_values = tuple(values)
    if any(not isinstance(value, str) or not value for value in tuple_values):
        raise ValueError(f"{name} entries must be non-empty strings")
    return tuple_values


__all__ = [
    "SourceChunk",
    "SourceDocument",
    "CacheBuildConfig",
    "CacheAdapterArtifact",
    "TrainingArtifacts",
    "TrainingAdapter",
    "KVChunkGenerator",
    "CacheGenerationResult",
    "DocumentKVWorkflow",
]
