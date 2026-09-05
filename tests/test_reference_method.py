from __future__ import annotations

from document_kv_cache.artifact_identity import method_config_digest
from document_kv_cache.cache import ChunkCache
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.materializer import KVMaterializer
from document_kv_cache.reference_method import (
    METHOD_SPEC,
    build_generator,
    register,
)
from document_kv_cache.storage import DiskRangeReader
from document_kv_cache.workflow import (
    CacheBuildConfig,
    DocumentKVWorkflow,
    SourceDocument,
)
from document_kv_cache.methods import default_method_registry


def test_cpu_reference_method_runs_strict_generation_without_model_weights(tmp_path) -> None:
    generator = build_generator()
    layout = generator.layout
    registry = register(default_method_registry())
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(
            cache=ChunkCache(cpu_max_bytes=32 * 1024 * 1024),
            reader=DiskRangeReader(),
        ),
        method_registry=registry,
    )
    config = CacheBuildConfig(
        model_id=layout.model_id,
        lora_id=layout.lora_id,
        prompt_template_version="reference-v1",
        dtype=layout.dtype,
        layout_version=layout.layout_version,
        cache_method=METHOD_SPEC.method_id,
        storage_layout=layout.storage_layout,
        payload_axis_order=layout.payload_axis_order,
        method_version=METHOD_SPEC.artifact_version,
        method_config_digest=method_config_digest({"reference": True}),
        model_revision="cpu-reference",
        tokenizer_id="utf8-bytes",
        tokenizer_revision="1",
        generator_family="cachet_cpu_reference",
        generator_version="1",
        artifact_format_id=METHOD_SPEC.artifact_format.format_id,
        artifact_format_version=METHOD_SPEC.artifact_format.version,
    )

    result = workflow.generate_cache(
        documents=(
            SourceDocument.from_text(document_id="doc-1", text="cachet"),
        ),
        generator=generator,
        config=config,
        shard_uri=tmp_path / "reference.kvpack",
        align_bytes=1,
        require_registered_method=True,
    )

    assert result.cache_method == METHOD_SPEC.method_id
    assert result.artifact_identity is not None
    assert not result.artifact_identity.has_unresolved_fields
    assert result.refs[0].key.token_contract is not None
    assert result.refs[0].byte_length == result.refs[0].token_count * layout.bytes_per_token
