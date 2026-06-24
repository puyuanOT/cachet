"""Packaged local Cachet quickstart with no cloud, GPU, or model download."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from cachet import (
    CacheBuildConfig,
    DocumentKVRequest,
    DocumentKVWorkflow,
    InMemoryManifestStore,
    KVCacheKey,
    SourceChunk,
    SourceDocument,
    TrainingArtifacts,
)
from cachet.engine_protocol import KVLayout, KVStorageLayout
from cachet.kvpack import PackChunk


MODEL_ID = "toy-local-model"
LORA_ID = "base"
LAYOUT_VERSION = "toy-local-v1"
PROMPT_TEMPLATE_VERSION = "v1"


class ToyKVGenerator:
    """Generate deterministic fake KV payloads for local plumbing tests."""

    def generate(
        self,
        *,
        document: SourceDocument,
        chunk: SourceChunk,
        config: CacheBuildConfig,
        training_artifacts: TrainingArtifacts | None = None,
    ) -> PackChunk:
        if training_artifacts is not None:
            raise ValueError("ToyKVGenerator does not use training artifacts")
        payload = (
            f"toy-kv|model={config.model_id}|document={document.document_id}|"
            f"chunk={chunk.chunk_id}|text={chunk.text}"
        ).encode("utf-8")
        content_hash = hashlib.sha256(payload).hexdigest()
        key = KVCacheKey.for_document(
            model_id=config.model_id,
            lora_id=config.lora_id,
            prompt_template_version=config.prompt_template_version,
            document_id=document.document_id,
            chunk_type=chunk.chunk_type,
            chunk_id=chunk.chunk_id,
            content_hash=content_hash,
        )
        return PackChunk(
            key=key,
            payload=payload,
            token_count=len(payload),
            dtype=config.dtype,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )


def main() -> None:
    layout = KVLayout(
        model_id=MODEL_ID,
        lora_id=LORA_ID,
        layout_version=LAYOUT_VERSION,
        dtype="uint8",
        num_layers=1,
        block_size=16,
        bytes_per_token=1,
        storage_layout=KVStorageLayout.SEPARATE_KEY_VALUE,
    )
    config = CacheBuildConfig(
        model_id=layout.model_id,
        lora_id=layout.lora_id,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        dtype=layout.dtype,
        layout_version=layout.layout_version,
        storage_layout=layout.storage_layout,
    )

    document = SourceDocument.from_texts(
        document_id="policy-handbook",
        static_text="Company policy handbook.",
        chunks={
            "vacation": "Employees earn paid time off every month.",
            "security": "Security reviews are required before production launch.",
        },
    )
    request = DocumentKVRequest.for_document_chunks(
        request_id="req-local",
        task_id="answer-question",
        model_id=config.model_id,
        lora_id=config.lora_id,
        prompt_template_version=config.prompt_template_version,
        document_id=document.document_id,
        chunk_ids=("vacation", "security"),
    )

    generator = ToyKVGenerator()
    with tempfile.TemporaryDirectory(prefix="cachet-quickstart-") as tmp:
        workspace = Path(tmp)
        workflow = DocumentKVWorkflow.with_storage(
            manifest=InMemoryManifestStore(),
            cpu_cache_bytes=8 * 1024 * 1024,
            local_cache_dir=workspace / "local-cache",
            local_cache_bytes=32 * 1024 * 1024,
            disk_root=workspace / "kvpacks",
        )

        disk_result = workflow.generate_cache(
            documents=(document,),
            generator=generator,
            config=config,
            shard_uri="policy-handbook.kvpack",
            align_bytes=1,
        )
        memory_result = workflow.generate_cache(
            documents=(
                SourceDocument.from_text(
                    document_id="memory-note",
                    text=(
                        "Memory-backed shards are useful for tests and "
                        "short-lived payloads."
                    ),
                ),
            ),
            generator=generator,
            config=config,
            shard_uri="memory:quickstart-note.kvpack",
            align_bytes=1,
        )

        materialized = workflow.prepare(request)
        ready = workflow.prepare_for_engine(
            request,
            layout=layout,
            cache_method=disk_result.cache_method,
            metadata={"example": "quickstart_local"},
        )

        print(f"generated chunks: {disk_result.chunk_count + memory_result.chunk_count}")
        print(f"materialized bytes: {len(materialized.payload)}")
        print(
            "materialized tiers: "
            + ", ".join(tier.value for tier in materialized.segment_tiers)
        )
        print(f"engine handle: {ready.handle.handle_uri}")
        print(f"engine segments: {len(ready.handle.segments)}")
        print(f"disk shard: {workspace / 'kvpacks' / 'policy-handbook.kvpack'}")


if __name__ == "__main__":
    main()
