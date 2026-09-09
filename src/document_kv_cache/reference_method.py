"""CPU-only reference KV method for plugin and workflow conformance tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from document_kv_cache.artifact_identity import TokenContract
from document_kv_cache.engine_protocol import KVLayout
from document_kv_cache.kvpack import PackChunk
from document_kv_cache.methods import (
    CACHET_CONNECTOR_MODE,
    DOCUMENT_KV_CACHE_ARM,
    MethodRegistry,
    MethodSpec,
)
from document_kv_cache.model_profiles import layout_for_model
from document_kv_cache.models import DocumentChunkType, KVCacheKey
from document_kv_cache.workflow import (
    CacheBuildConfig,
    SourceChunk,
    SourceDocument,
    TrainingArtifacts,
)


REFERENCE_METHOD_ID = "cpu_reference"

__all__ = [
    "REFERENCE_METHOD_ID",
    "CPUReferenceKVGenerator",
    "METHOD_SPEC",
    "build_generator",
    "register",
]


@dataclass(frozen=True, slots=True)
class CPUReferenceKVGenerator:
    """Generate deterministic shape-correct bytes without model inference."""

    layout: KVLayout = field(
        default_factory=lambda: layout_for_model("qwen3:4b-instruct", dtype="int8")
    )
    pre_rope: bool = False

    def generate(
        self,
        *,
        document: SourceDocument,
        chunk: SourceChunk,
        config: CacheBuildConfig,
        training_artifacts: TrainingArtifacts | None = None,
    ) -> PackChunk:
        del training_artifacts
        token_ids = tuple(chunk.text.encode("utf-8")) or (0,)
        payload_bytes = len(token_ids) * self.layout.bytes_per_token
        seed = hashlib.sha256(
            f"{document.document_id}|{chunk.chunk_id}|{chunk.text}".encode("utf-8")
        ).digest()
        payload = (seed * ((payload_bytes + len(seed) - 1) // len(seed)))[:payload_bytes]
        content_hash = hashlib.sha256(payload).hexdigest()
        token_contract = TokenContract.from_token_ids(
            token_ids,
            tokenizer_id=config.tokenizer_id or config.model_id,
            tokenizer_revision=config.tokenizer_revision,
            add_special_tokens=False,
            prompt_template_version=config.prompt_template_version,
        )
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=DocumentChunkType(chunk.chunk_type),
                chunk_id=chunk.chunk_id,
                content_hash=content_hash,
                artifact_identity=config.artifact_identity_for(self.layout),
                token_contract=token_contract,
            ),
            payload=payload,
            token_count=len(token_ids),
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


def build_generator() -> CPUReferenceKVGenerator:
    return CPUReferenceKVGenerator()


METHOD_SPEC = MethodSpec(
    method=REFERENCE_METHOD_ID,
    display_name="CPU reference KV",
    arm_id=DOCUMENT_KV_CACHE_ARM,
    connector_mode=CACHET_CONNECTOR_MODE,
    pre_rope=False,
    selective_recompute=False,
    implemented=True,
    artifact_version="1",
    generator_factory="document_kv_cache.reference_method:build_generator",
    description=(
        "Deterministic CPU-only raw KV-shaped bytes for method plugin, identity, "
        "streaming, and handoff conformance; not latency or quality evidence."
    ),
    metadata={"evidence_scope": "conformance_only"},
)


def register(registry: MethodRegistry) -> MethodRegistry:
    return registry.with_spec(METHOD_SPEC)
