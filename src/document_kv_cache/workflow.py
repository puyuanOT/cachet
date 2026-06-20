"""End-to-end document cache generation and serving workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from document_kv_cache.engine import EngineReadyRequest, KVLayout, _normalize_gpu_byte_multiplier, build_engine_ready_request
from document_kv_cache.engine_protocol import KVStorageLayout, kv_storage_layout_from_value
from document_kv_cache.kvpack import PackChunk, write_kvpack
from document_kv_cache.manifest import ManifestStore
from document_kv_cache.materializer import KVMaterializer, MaterializedKV, SegmentedMaterializedKV
from document_kv_cache.models import CacheGenerationMethod, ChunkRef, DocumentChunkType, DocumentKVRequest
from document_kv_cache.planner import CachePlanner
from document_kv_cache.service import DocumentKVService


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
    def from_texts(
        cls,
        *,
        document_id: str,
        chunks: Mapping[str, str],
        static_text: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> "SourceDocument":
        if not isinstance(chunks, Mapping):
            raise TypeError("chunks must be a mapping")
        source_chunks: list[SourceChunk] = []
        if static_text is not None:
            source_chunks.append(
                SourceChunk(
                    chunk_id="static",
                    text=static_text,
                    chunk_type=DocumentChunkType.DOCUMENT_STATIC,
                )
            )
        source_chunks.extend(SourceChunk(chunk_id=chunk_id, text=text) for chunk_id, text in chunks.items())
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
    cache_method: CacheGenerationMethod = CacheGenerationMethod.VANILLA_PREFILL
    storage_layout: KVStorageLayout | str = KVStorageLayout.SEPARATE_KEY_VALUE

    def __post_init__(self) -> None:
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
    cache_method: CacheGenerationMethod = CacheGenerationMethod.VANILLA_PREFILL

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
    ) -> None:
        self.manifest = manifest
        self.planner = planner or CachePlanner(manifest)
        self.materializer = materializer
        self.service = service

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
        refs = tuple(write_kvpack(shard_uri, pack_chunks, align_bytes=align_bytes))
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


def _effective_cache_method(config: CacheBuildConfig, trainer: TrainingAdapter | None) -> CacheGenerationMethod:
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
