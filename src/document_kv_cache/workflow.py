"""End-to-end document cache generation and serving workflows."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from document_kv_cache.artifact_identity import (
    ArtifactIdentity,
    UNRESOLVED_IDENTITY,
    method_config_digest,
)
from document_kv_cache.cache import ChunkCache
from document_kv_cache.engine import (
    EngineReadyRequest,
    KVLayout,
    ServingEngineConnector,
    _normalize_gpu_byte_multiplier,
    build_engine_ready_request,
)
from document_kv_cache.engine_protocol import (
    KVStorageLayout,
    kv_payload_axis_order_from_value,
    kv_storage_layout_from_value,
)
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
from document_kv_cache.methods import MethodRegistry, MethodSpec, default_method_registry
from document_kv_cache.planner import CachePlanner
from document_kv_cache.reuse_contract import ArtifactEncoding, PositionHandling, ReusePlan
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
        duplicate_chunk_ids = _duplicate_source_chunk_ids(chunks)
        if duplicate_chunk_ids:
            raise ValueError("chunks contain duplicate chunk identities: " + ", ".join(duplicate_chunk_ids))
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
    """Identity used to generate persisted method artifacts.

    ``dtype`` is the persisted artifact element dtype. ``runtime_kv_dtype`` is
    the dtype after any provider decoder has produced engine-ready KV; it
    defaults to ``dtype`` for raw-KV compatibility. Physical stored byte counts
    come from generated :class:`PackChunk` payloads rather than from the runtime
    layout geometry.
    """

    model_id: str
    lora_id: str
    prompt_template_version: str
    dtype: str
    layout_version: str
    # A build config has no runtime layout from which to infer positional
    # semantics.  Keep the omission-safe default aligned with the default
    # (post-RoPE) generator; Vanilla/pre-RoPE generation must opt in through
    # its registered method path.
    cache_method: CacheGenerationMethod | str = CacheGenerationMethod.FULL_PREFIX_PREFILL
    storage_layout: KVStorageLayout | str = KVStorageLayout.SEPARATE_KEY_VALUE
    payload_axis_order: str = "token_major"
    method_version: str | None = None
    method_config_digest: str = field(default_factory=lambda: method_config_digest({}))
    model_revision: str = UNRESOLVED_IDENTITY
    tokenizer_id: str | None = None
    tokenizer_revision: str = UNRESOLVED_IDENTITY
    generator_family: str = "transformers"
    generator_version: str = UNRESOLVED_IDENTITY
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    artifact_format_id: str = "raw_kv"
    artifact_format_version: str = "1"
    runtime_kv_dtype: str | None = None

    def __post_init__(self) -> None:
        cache_method = _cache_generation_method(self.cache_method)
        object.__setattr__(self, "cache_method", cache_method)
        method_version = self.method_version
        if method_version is None:
            try:
                method_version = default_method_registry().get(
                    cache_method,
                ).artifact_version
            except KeyError:
                # Application-defined methods remain usable without registering a
                # process-global MethodSpec. Their historical default is v1, while
                # callers can always stamp an explicit custom version.
                method_version = "1"
        object.__setattr__(self, "method_version", method_version)
        for field_name in (
            "model_id",
            "lora_id",
            "prompt_template_version",
            "dtype",
            "layout_version",
            "method_version",
            "model_revision",
            "tokenizer_revision",
            "generator_family",
            "generator_version",
            "artifact_format_id",
            "artifact_format_version",
        ):
            object.__setattr__(self, field_name, _non_empty_string(field_name, getattr(self, field_name)))
        tokenizer_id = self.model_id if self.tokenizer_id is None else self.tokenizer_id
        object.__setattr__(self, "tokenizer_id", _non_empty_string("tokenizer_id", tokenizer_id))
        runtime_kv_dtype = self.dtype if self.runtime_kv_dtype is None else self.runtime_kv_dtype
        object.__setattr__(
            self,
            "runtime_kv_dtype",
            _non_empty_string("runtime_kv_dtype", runtime_kv_dtype),
        )
        _sha256_digest("method_config_digest", self.method_config_digest)
        for field_name in ("tensor_parallel_size", "pipeline_parallel_size"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        object.__setattr__(
            self,
            "storage_layout",
            kv_storage_layout_from_value(self.storage_layout, field_name="storage_layout"),
        )
        object.__setattr__(
            self,
            "payload_axis_order",
            kv_payload_axis_order_from_value(self.payload_axis_order, field_name="payload_axis_order").value,
        )

    def artifact_identity_for(self, layout: KVLayout) -> ArtifactIdentity:
        """Build the exact stored-artifact and runtime-layout identity.

        ``dtype`` describes the persisted artifact elements.  The supplied
        :class:`KVLayout` describes the decoded KV presented to the serving
        runtime, so its dtype is compared with ``runtime_kv_dtype`` instead of
        assuming that persisted and runtime representations are identical.
        """

        if not isinstance(layout, KVLayout):
            raise TypeError("layout must be a KVLayout")
        layout.validate()
        expected = {
            "model_id": self.model_id,
            "lora_id": self.lora_id,
            "layout_version": self.layout_version,
            "runtime_kv_dtype": self.runtime_kv_dtype,
            "payload_axis_order": self.payload_axis_order,
        }
        actual = {
            "model_id": layout.model_id,
            "lora_id": layout.lora_id,
            "layout_version": layout.layout_version,
            "runtime_kv_dtype": layout.dtype,
            "payload_axis_order": layout.payload_axis_order.value,
        }
        mismatches = [name for name, value in expected.items() if actual[name] != value]
        if mismatches:
            raise ValueError(
                "layout does not match CacheBuildConfig identity: " + ", ".join(mismatches)
            )
        return ArtifactIdentity(
            method_id=_cache_method_value(self.cache_method),
            method_version=self.method_version,
            method_config_digest=self.method_config_digest,
            model_id=self.model_id,
            model_revision=self.model_revision,
            tokenizer_id=self.tokenizer_id,
            tokenizer_revision=self.tokenizer_revision,
            lora_id=self.lora_id,
            prompt_template_version=self.prompt_template_version,
            layout_version=self.layout_version,
            kv_dtype=self.dtype,
            block_size=layout.block_size,
            payload_axis_order=self.payload_axis_order,
            key_position_encoding=layout.key_position_encoding.value,
            rope_theta=layout.rope_theta,
            rope_rotary_dim=layout.rope_rotary_dim,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            generator_family=self.generator_family,
            generator_version=self.generator_version,
            artifact_format_id=self.artifact_format_id,
            artifact_format_version=self.artifact_format_version,
            runtime_kv_dtype=self.runtime_kv_dtype,
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
        object.__setattr__(self, "metadata", _metadata_dict("metadata", self.metadata))


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    adapter_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, str] = field(default_factory=dict)
    adapter_artifacts: tuple[CacheAdapterArtifact, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        adapter_ids = _non_empty_unique_string_tuple("adapter_ids", self.adapter_ids)
        adapter_artifacts = tuple(self.adapter_artifacts)
        for artifact in adapter_artifacts:
            if not isinstance(artifact, CacheAdapterArtifact):
                raise TypeError("adapter_artifacts entries must be CacheAdapterArtifact instances")
        if adapter_artifacts:
            artifact_ids = _non_empty_unique_string_tuple(
                "adapter_artifacts adapter_id",
                (artifact.adapter_id for artifact in adapter_artifacts),
            )
            if adapter_ids and adapter_ids != artifact_ids:
                raise ValueError("adapter_ids must match adapter_artifacts adapter_id order")
            adapter_ids = artifact_ids
        object.__setattr__(self, "adapter_ids", adapter_ids)
        object.__setattr__(self, "metadata", _metadata_dict("metadata", self.metadata))
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
    refs: tuple[ChunkRef, ...] | Sequence[ChunkRef]
    document_ids: tuple[str, ...] | Sequence[str]
    chunk_count: int
    total_bytes: int
    training_artifacts: TrainingArtifacts | None = None
    # Results constructed without refs/identity cannot infer a pre-RoPE
    # contract.  Default to the post-RoPE full-prefix method and require
    # Vanilla callers to name it (normal workflow generation always does).
    cache_method: CacheGenerationMethod | str = CacheGenerationMethod.FULL_PREFIX_PREFILL
    artifact_identity: ArtifactIdentity | None = None
    reuse_plan: ReusePlan | None = None

    def __post_init__(self) -> None:
        refs = _chunk_ref_tuple(self.refs)
        document_ids = _non_empty_string_tuple("document_ids", self.document_ids)
        expected_document_ids = _document_ids_for_refs(refs)
        if document_ids != expected_document_ids:
            raise ValueError("document_ids must match refs document_id order")
        if type(self.chunk_count) is not int or self.chunk_count < 0:
            raise ValueError("chunk_count must be a non-negative integer")
        if self.chunk_count != len(refs):
            raise ValueError("chunk_count must match refs length")
        expected_total_bytes = sum(ref.byte_length for ref in refs)
        if type(self.total_bytes) is not int or self.total_bytes < 0:
            raise ValueError("total_bytes must be a non-negative integer")
        if self.total_bytes != expected_total_bytes:
            raise ValueError("total_bytes must match refs byte_length sum")
        if self.training_artifacts is not None and not isinstance(self.training_artifacts, TrainingArtifacts):
            raise TypeError("training_artifacts must be TrainingArtifacts or None")
        object.__setattr__(self, "refs", refs)
        object.__setattr__(self, "document_ids", document_ids)
        object.__setattr__(self, "cache_method", _cache_generation_method(self.cache_method))
        ref_identities = {ref.key.artifact_identity for ref in refs}
        if len(ref_identities) > 1:
            raise ValueError("refs must share one artifact_identity")
        inferred_identity = next(iter(ref_identities), None)
        if self.artifact_identity is not None and not isinstance(
            self.artifact_identity,
            ArtifactIdentity,
        ):
            raise TypeError("artifact_identity must be an ArtifactIdentity or None")
        if self.artifact_identity is not None and self.artifact_identity != inferred_identity:
            raise ValueError("artifact_identity must match refs")
        artifact_identity = self.artifact_identity or inferred_identity
        if artifact_identity is not None:
            if artifact_identity.method_id != _cache_method_value(self.cache_method):
                raise ValueError("artifact_identity.method_id must match cache_method")
        object.__setattr__(self, "artifact_identity", artifact_identity)
        if self.reuse_plan is not None:
            if not isinstance(self.reuse_plan, ReusePlan):
                raise TypeError("reuse_plan must be a ReusePlan or None")
            if self.reuse_plan.method_id != _cache_method_value(self.cache_method):
                raise ValueError("reuse_plan.method_id must match cache_method")
            if artifact_identity is not None:
                artifact_format = self.reuse_plan.artifact_format
                if (
                    artifact_identity.artifact_format_id != artifact_format.format_id
                    or artifact_identity.artifact_format_version != artifact_format.version
                ):
                    raise ValueError(
                        "reuse_plan artifact format must match artifact_identity"
                    )

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        if self.training_artifacts is None:
            return ()
        return self.training_artifacts.adapter_ids

    @property
    def artifact_id(self) -> str | None:
        if self.artifact_identity is None:
            return None
        return self.artifact_identity.artifact_id


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
        method_registry: MethodRegistry | None = None,
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
        self.method_registry = (
            default_method_registry() if method_registry is None else method_registry
        )
        if not isinstance(self.method_registry, MethodRegistry):
            raise TypeError("method_registry must be a MethodRegistry")

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
        method_registry: MethodRegistry | None = None,
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
            method_registry=method_registry,
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
        require_registered_method: bool = True,
    ) -> CacheGenerationResult:
        documents_tuple = _source_documents_tuple(documents)
        method = self._validate_method_generator(
            config,
            generator,
            require_registered_method=require_registered_method,
        )
        training_artifacts = trainer.fit(documents_tuple, config) if trainer is not None else None
        cache_method = config.cache_method
        pack_chunks = self._iter_pack_chunks(
            documents_tuple,
            generator,
            config,
            training_artifacts,
            require_artifact_contract=require_registered_method,
            registered_method=method if require_registered_method else None,
        )
        refs = self._write_pack_chunks(shard_uri, pack_chunks, align_bytes=align_bytes)
        self.manifest.put_many(refs)
        return CacheGenerationResult(
            refs=refs,
            document_ids=tuple(document.document_id for document in documents_tuple),
            chunk_count=len(refs),
            total_bytes=sum(ref.byte_length for ref in refs),
            training_artifacts=training_artifacts,
            cache_method=cache_method,
            artifact_identity=_artifact_identity_for_refs(refs),
            reuse_plan=None if method is None else method.reuse_plan(),
        )

    def _validate_method_generator(
        self,
        config: CacheBuildConfig,
        generator: KVChunkGenerator,
        *,
        require_registered_method: bool,
    ) -> MethodSpec | None:
        try:
            spec = self.method_registry.get(
                config.cache_method,
                require_implemented=require_registered_method,
            )
        except KeyError:
            if require_registered_method:
                raise
            return None
        if not require_registered_method and not spec.implemented:
            # Explicit legacy generation may preserve an experimental label, but
            # an unimplemented MethodSpec must never mint an executable ReusePlan.
            return None
        if require_registered_method and config.method_version != spec.artifact_version:
            raise ValueError(
                f"CacheBuildConfig method_version {config.method_version!r} does not match "
                f"registered {spec.method_id!r} artifact_version {spec.artifact_version!r}"
            )
        if require_registered_method:
            configured_format = (config.artifact_format_id, config.artifact_format_version)
            registered_format = (
                spec.artifact_format.format_id,
                spec.artifact_format.version,
            )
            if configured_format != registered_format:
                raise ValueError(
                    f"CacheBuildConfig artifact format {configured_format!r} does not match "
                    f"registered {spec.method_id!r} format {registered_format!r}"
                )
            if (
                spec.artifact_format.encoding == ArtifactEncoding.RAW_KV
                and config.dtype != config.runtime_kv_dtype
            ):
                raise ValueError(
                    "raw-KV generation requires persisted dtype to match "
                    "runtime_kv_dtype"
                )
        spec.validate_generator(
            generator,
            require_implemented=require_registered_method,
        )
        return spec

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
        cache_method: CacheGenerationMethod | str | None = None,
        adapter_ids: tuple[str, ...] = (),
        training_artifacts: TrainingArtifacts | None = None,
        segmented: bool = False,
        kv_gpu_bytes_per_payload_byte: float | None = None,
        require_registered_method: bool = True,
    ) -> EngineReadyRequest:
        gpu_byte_multiplier = self._engine_gpu_byte_multiplier(kv_gpu_bytes_per_payload_byte)
        engine_adapter_ids = _engine_adapter_ids(adapter_ids, training_artifacts)
        planner, materializer = self._preparation_dependencies()
        plan = planner.build_plan(request)
        materialized = materializer.materialize_segmented(plan) if segmented else materializer.materialize(plan)
        artifact_method_ids = {
            identity.method_id
            for segment in materialized.plan.segments
            if (identity := segment.ref.key.artifact_identity) is not None
        }
        artifact_identities = {
            identity
            for segment in materialized.plan.segments
            if (identity := segment.ref.key.artifact_identity) is not None
        }
        if len(artifact_method_ids) > 1:
            raise ValueError(
                "materialized KV segments contain multiple cache methods"
            )
        resolved_cache_method: CacheGenerationMethod | str = (
            next(iter(artifact_method_ids))
            if cache_method is None and artifact_method_ids
            else (
                CacheGenerationMethod.VANILLA_PREFILL
                if layout.pre_rope
                else CacheGenerationMethod.FULL_PREFIX_PREFILL
            )
            if cache_method is None
            else cache_method
        )
        try:
            method = self.method_registry.get(
                resolved_cache_method,
                require_implemented=require_registered_method,
            )
        except KeyError:
            if require_registered_method:
                raise
            method = None
        reuse_plan = None
        if method is not None:
            if artifact_method_ids and method.method_id not in artifact_method_ids:
                raise ValueError(
                    "cache_method does not match materialized artifact identity"
                )
            reuse_plan = method.reuse_plan()
            reuse_plan.validate_runtime_layout(layout)
            for identity in artifact_identities:
                if identity.method_version != method.artifact_version:
                    raise ValueError(
                        "materialized artifact method_version does not match the "
                        "registered method contract"
                    )
                if (
                    identity.artifact_format_id
                    != reuse_plan.artifact_format.format_id
                    or identity.artifact_format_version
                    != reuse_plan.artifact_format.version
                ):
                    raise ValueError(
                        "materialized artifact format does not match the registered "
                        "reuse plan"
                    )
        return build_engine_ready_request(
            materialized,
            layout=layout,
            handle_uri=handle_uri,
            metadata=metadata,
            cache_method=resolved_cache_method,
            adapter_ids=engine_adapter_ids,
            kv_gpu_bytes_per_payload_byte=gpu_byte_multiplier,
            reuse_plan=reuse_plan,
            allow_legacy_reuse_plan=not require_registered_method,
            method_registry=self.method_registry,
        )

    def prepare_and_submit_to_engine(
        self,
        request: DocumentKVRequest,
        *,
        connector: ServingEngineConnector,
        layout: KVLayout,
        handle_uri: str | None = None,
        metadata: Mapping[str, str] | None = None,
        cache_method: CacheGenerationMethod | str | None = None,
        adapter_ids: tuple[str, ...] = (),
        training_artifacts: TrainingArtifacts | None = None,
        segmented: bool = False,
        kv_gpu_bytes_per_payload_byte: float | None = None,
        require_registered_method: bool = True,
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
            require_registered_method=require_registered_method,
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
        *,
        require_artifact_contract: bool,
        registered_method: MethodSpec | None,
    ) -> Iterable[PackChunk]:
        for document in documents:
            for chunk in document.chunks:
                pack_chunk = generator.generate(
                    document=document,
                    chunk=chunk,
                    config=config,
                    training_artifacts=training_artifacts,
                )
                self._validate_pack_chunk(
                    document,
                    chunk,
                    config,
                    pack_chunk,
                    require_artifact_contract=require_artifact_contract,
                    registered_method=registered_method,
                )
                yield pack_chunk

    def _write_pack_chunks(
        self,
        shard_uri: str | Path,
        pack_chunks: Iterable[PackChunk],
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
        *,
        require_artifact_contract: bool,
        registered_method: MethodSpec | None,
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
        if require_artifact_contract:
            missing = [
                name
                for name, value in (
                    ("content_hash", key.content_hash),
                    ("artifact_identity", key.artifact_identity),
                    ("token_contract", key.token_contract),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "Registered method generation requires: " + ", ".join(missing)
                )
        if key.content_hash:
            payload_digest = hashlib.sha256(pack_chunk.payload).hexdigest()
            if key.content_hash != payload_digest:
                raise ValueError("Generated chunk content_hash does not match payload bytes")
        if key.token_contract is not None and key.token_contract.token_count != pack_chunk.token_count:
            raise ValueError("Generated chunk token contract count does not match token_count")
        if key.artifact_identity is not None:
            identity = key.artifact_identity
            identity_expected = {
                "method_id": _cache_method_value(config.cache_method),
                "method_version": config.method_version,
                "method_config_digest": config.method_config_digest,
                "model_revision": config.model_revision,
                "tokenizer_id": config.tokenizer_id,
                "tokenizer_revision": config.tokenizer_revision,
                "generator_family": config.generator_family,
                "generator_version": config.generator_version,
                "tensor_parallel_size": config.tensor_parallel_size,
                "pipeline_parallel_size": config.pipeline_parallel_size,
                "kv_dtype": config.dtype,
                "layout_version": config.layout_version,
                "payload_axis_order": config.payload_axis_order,
                "artifact_format_id": config.artifact_format_id,
                "artifact_format_version": config.artifact_format_version,
                "runtime_kv_dtype": config.runtime_kv_dtype,
            }
            identity_mismatches = [
                name
                for name, value in identity_expected.items()
                if getattr(identity, name) != value
            ]
            if identity_mismatches:
                raise ValueError(
                    "Generated chunk artifact identity does not match config: "
                    + ", ".join(identity_mismatches)
                )
            if registered_method is not None:
                position_handling = registered_method.position_handling
                if position_handling == PositionHandling.REROPE_AT_INJECTION:
                    expected_key_position_encoding = "pre_rope"
                elif position_handling == PositionHandling.STORED_POST_ROPE:
                    expected_key_position_encoding = "stored_post_rope"
                else:
                    raise ValueError(
                        "Registered Cachet artifact generation cannot use "
                        f"position handling {position_handling!r}"
                    )
                if identity.key_position_encoding != expected_key_position_encoding:
                    raise ValueError(
                        "Generated chunk artifact identity position encoding does "
                        f"not match registered method {registered_method.method_id!r}: "
                        f"expected {expected_key_position_encoding!r}, got "
                        f"{identity.key_position_encoding!r}"
                    )


def _engine_adapter_ids(
    adapter_ids: tuple[str, ...],
    training_artifacts: TrainingArtifacts | None,
) -> tuple[str, ...]:
    explicit_adapter_ids = _non_empty_unique_string_tuple("adapter_ids", adapter_ids)
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


def _chunk_ref_tuple(refs: Iterable[ChunkRef]) -> tuple[ChunkRef, ...]:
    if isinstance(refs, (str, bytes, bytearray)):
        raise TypeError("refs must be a sequence of ChunkRef instances")
    refs_tuple = tuple(refs)
    for ref in refs_tuple:
        if not isinstance(ref, ChunkRef):
            raise TypeError("refs entries must be ChunkRef instances")
    return refs_tuple


def _document_ids_for_refs(refs: Sequence[ChunkRef]) -> tuple[str, ...]:
    seen: set[str] = set()
    document_ids: list[str] = []
    for ref in refs:
        document_id = ref.key.document_id
        if document_id in seen:
            continue
        seen.add(document_id)
        document_ids.append(document_id)
    return tuple(document_ids)


def _artifact_identity_for_refs(
    refs: Sequence[ChunkRef],
) -> ArtifactIdentity | None:
    identities = {ref.key.artifact_identity for ref in refs}
    if len(identities) > 1:
        raise ValueError("refs must share one artifact_identity")
    return next(iter(identities), None)


def _source_documents_tuple(documents: Sequence[SourceDocument]) -> tuple[SourceDocument, ...]:
    if isinstance(documents, (str, bytes, bytearray)):
        raise TypeError("documents must be a sequence of SourceDocument instances")
    documents_tuple = tuple(documents)
    for document in documents_tuple:
        if not isinstance(document, SourceDocument):
            raise TypeError("documents entries must be SourceDocument instances")
    duplicate_document_ids = _duplicate_document_ids(documents_tuple)
    if duplicate_document_ids:
        raise ValueError("documents contain duplicate document ids: " + ", ".join(duplicate_document_ids))
    return documents_tuple


def _duplicate_document_ids(documents: Sequence[SourceDocument]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    duplicate_labels: set[str] = set()
    for document in documents:
        document_id = document.document_id
        if document_id not in seen:
            seen.add(document_id)
            continue
        if document_id not in duplicate_labels:
            duplicates.append(document_id)
            duplicate_labels.add(document_id)
    return tuple(duplicates)


def _document_chunk_type(chunk_type: DocumentChunkType | str) -> DocumentChunkType:
    if isinstance(chunk_type, DocumentChunkType):
        return chunk_type
    try:
        return DocumentChunkType(str(chunk_type))
    except ValueError as exc:
        raise ValueError(f"chunk_type must be one of {[chunk_type.value for chunk_type in DocumentChunkType]}") from exc


def _duplicate_source_chunk_ids(chunks: Sequence[SourceChunk]) -> tuple[str, ...]:
    seen: set[tuple[DocumentChunkType, str]] = set()
    duplicates: list[str] = []
    duplicate_labels: set[str] = set()
    for chunk in chunks:
        identity = (chunk.chunk_type, chunk.chunk_id)
        if identity not in seen:
            seen.add(identity)
            continue
        label = f"{chunk.chunk_type.value}:{chunk.chunk_id}"
        if label not in duplicate_labels:
            duplicates.append(label)
            duplicate_labels.add(label)
    return tuple(duplicates)


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


def _sha256_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _non_empty_string_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of non-empty strings, not a string")
    tuple_values = tuple(values)
    if any(not isinstance(value, str) or not value for value in tuple_values):
        raise ValueError(f"{name} entries must be non-empty strings")
    return tuple_values


def _non_empty_unique_string_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    tuple_values = _non_empty_string_tuple(name, values)
    if len(set(tuple_values)) != len(tuple_values):
        raise ValueError(f"{name} entries must be unique")
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
