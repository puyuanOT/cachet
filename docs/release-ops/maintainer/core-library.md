# Core Library

Maintainer reference for the manifest, storage, cache, workflow, and
model-profile layer in `document_kv_cache`, which the `cachet` facade
re-exports. The user path is [`../../getting-started.md`](../../getting-started.md)
and [`../../concepts.md`](../../concepts.md); per-module ownership is listed in
[`src/document_kv_cache/README.md`](../../../src/document_kv_cache/README.md).

## Logical Model

Each document can be represented by stable chunks and selectable chunks:

```text
task_prefix_cache
+ document_static_cache     # metadata, description, source summary, etc.
+ document_chunk_cache      # selected sections, passages, reviews, facts, etc.
+ user/task suffix
```

`DocumentChunkRole` maps concrete chunk types onto the canonical roles
`TASK_PREFIX`, `STATIC`, and `CONTENT`, so manifests and planners order task
prefixes, static document context, and selected content consistently.

Physical storage uses large packed files, not one file per chunk:

```text
UC Volume, disk, or object-store mounted path:
  shard_000001.kvpack
  shard_000002.kvpack

Manifest table:
  model_id
  lora_id
  prompt_template_version
  document_id
  chunk_type
  chunk_id
  shard_uri
  byte_offset
  byte_length
  token_count
  dtype
  layout_version
  storage_layout
  checksum
```

The package is storage-format-first and tensor-runtime-agnostic: it
materializes ordered byte ranges, and the vendored `vllm_kv_injection` and
`sglang_kv_injection` adapters interpret them as engine KV blocks.

## Storage Backends

The materializer consumes a small `RangeReader` protocol. Readers can also
implement `RangeBatchReader.read_many()` so a materialization plan can load many
ranges from the same packed shard with one open file handle per shard. Current
implementations are:

- `MemoryRangeReader` for hot shards already resident in process memory.
- `DiskRangeReader` for local NVMe, `disk:` / `file:` URIs, and filesystem-mounted paths.
- `UnityCatalogVolumeRangeReader` for Databricks UC Volumes exposed under `/Volumes`.
- `RoutedRangeReader` for URI-based dispatch across the three readers; `disk:` forces
  the local disk reader even when a UC Volume root is configured.

`DocumentKVWorkflow.with_storage(...)` can also generate directly into the
configured `MemoryRangeReader` when `generate_cache(...)` receives a `memory:`
or `mem:` shard URI. Use that path for hot ephemeral shards; UC Volume or disk
URIs remain the durable storage targets.

## Tiered Cache

`ChunkCache` keeps hot chunks in a byte-bounded CPU LRU and can persist cold
misses into a local disk tier; the configuration example is in
[`../../production.md`](../../production.md#hierarchical-document-kv-persistence).
`ChunkCacheStats` reports CPU hits, local disk hits, cold misses, and tier
sizes. Use `get_or_load_with_tier()` when a direct cache probe needs the serving
tier (`cpu`, `local_disk`, or `cold_storage`) for latency attribution;
`MaterializedKV`, `SegmentedMaterializedKV`, and `EngineReadyRequest` expose the
same attribution as `segment_tiers`. Local files are keyed by logical chunk
identity plus checksum, so a regenerated chunk cannot reuse stale local bytes.
Measure reader behavior with the storage-reader benchmark in
[`benchmarks.md`](benchmarks.md#storage-reader-benchmark).

## Workflow API

`DocumentKVWorkflow` coordinates the data flow described in
[`../../concepts.md`](../../concepts.md#data-flow). Minimal shape:

```python
generator = build_pre_rope_transformers_kv_chunk_generator()
layout = layout_for_model(
    "qwen3:4b-instruct",
    dtype="bfloat16",
    pre_rope=True,
    rope_theta=generator.rope_theta,
    rope_rotary_dim=generator.rope_rotary_dim,
    shares_kv_storage=False,
    storage_layout=KVStorageLayout.SEPARATE_KEY_VALUE,
)
generator.bind_layout(layout)
workflow = DocumentKVWorkflow.with_storage(
    manifest=manifest,
    cpu_cache_bytes=2 * 1024**3,
    local_cache_dir="/local_disk0/cachet-kv",
    local_cache_bytes=200 * 1024**3,
    uc_volume_root="/Volumes/catalog/schema/volume",
)
document = SourceDocument.from_text(
    document_id="doc-a",
    text="Long source document text...",
)
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
    generator=generator,
    config=CacheBuildConfig(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        dtype=layout.dtype,
        layout_version="qwen3-v1",
        storage_layout=layout.storage_layout,
        cache_method=CacheGenerationMethod.VANILLA_PREFILL,
    ),
    shard_uri="/Volumes/catalog/schema/volume/shard_000001.kvpack",
)
prepared = workflow.prepare(request)
ready = workflow.prepare_for_engine(
    request,
    layout=layout,
    cache_method=result.cache_method,
)
```

When `with_storage(...)` is used, relative `shard_uri` values passed to
`generate_cache(...)` are written under `uc_volume_root` when it is configured;
otherwise they resolve under `disk_root` or the current filesystem path. The
manifest keeps the logical URI so the routed reader can load the same shard
during `prepare(...)`. A `memory:` or `mem:` shard URI stays in process memory
and is not written to disk.

For documents that are already split into static context and reusable content
chunks, use `SourceDocument.from_texts(...)` with
`DocumentKVRequest.for_document_chunks(...)`:

```python
document = SourceDocument.from_texts(
    document_id="doc-a",
    static_text="Document profile...",
    static_chunk_id="profile",
    static_chunk_metadata={"source": "metadata"},
    chunks={"review-1": "First reusable chunk", "review-2": "Second reusable chunk"},
    chunk_metadata={
        "review-1": {"source": "review"},
        "review-2": {"source": "review"},
    },
)
request = DocumentKVRequest.for_document_chunks(
    request_id="req-1",
    task_id="qa",
    model_id="qwen3:4b-instruct",
    lora_id="base",
    prompt_template_version="v1",
    document_id="doc-a",
    static_chunk_id="profile",
    chunk_ids=("review-2",),
)
```

Use `DocumentKVRequest.for_document_selection(...)` when one request selects
chunks from more than one document:

```python
request = DocumentKVRequest.for_document_selection(
    request_id="req-1",
    task_id="qa",
    model_id="qwen3:4b-instruct",
    lora_id="base",
    prompt_template_version="v1",
    document_chunks={"doc-a": ("review-2",), "doc-b": ("review-1",)},
    static_chunk_id="profile",
)
```

The generator is injected so the orchestration and storage contract stay
stable across generators. `CacheGenerationMethod` labels the generation path
(for example `VANILLA_PREFILL`, `ADAPTER_TRAINED`, or `KV_PACKET`) without
changing the manifest key schema. Training-aware generators can return
`TrainingArtifacts` with `adapter_ids` or `CacheAdapterArtifact` entries;
`prepare_for_engine(..., training_artifacts=result.training_artifacts)` derives
the engine adapter IDs from them, while explicit `adapter_ids` remain available
for callers that manage LoRA or packet adapters separately.

## Model Profiles

`KVModelProfile` centralizes attention geometry so cache generators, manifests,
and engine adapters derive the same `KVLayout`. V1 ships
`QWEN3_4B_INSTRUCT_PROFILE`, exposed as `qwen3:4b-instruct` and the canonical
`Qwen/Qwen3-4B-Instruct-2507` Hugging Face model id. `KVLayout` records a
`KVStorageLayout` (`separate_key_value`, `interleaved_key_value`, or
`shared_key_value`); cache generators and manifest refs persist it, and serving
validation rejects a request whose persisted chunks disagree with the engine
handoff layout.

Future Qwen3.5, MiniMax, or adapter-trained KV Packet integrations should add a
profile first, then implement a generator or engine adapter against the same
layout contract. Use a caller-owned `ModelProfileRegistry` for profiles that are
not V1 built-ins, or ship a portable `ModelProfileDefinition` JSON artifact with
an external model bundle. That artifact has a closed top-level schema; put
bundle annotations in `metadata`. The example uses schematic MQA-style geometry;
real bundles must use measured values from the target model:

```python
from document_kv_cache import (
    KVModelProfile,
    ModelProfileDefinition,
    default_model_profile_registry,
    write_model_profile_definition_json,
)

future_mqa_profile = KVModelProfile(
    model_id="future-mqa:4b",
    architecture="FutureMQAForCausalLM",
    num_layers=28,
    num_query_heads=32,
    num_kv_heads=1,
    head_size=128,
    max_context_tokens=65536,
    default_layout_version="future-mqa-v1",
    default_shares_kv_storage=False,
    default_storage_layout="separate_key_value",
    metadata={"attention": "mqa", "status": "future-extension"},
)
definition = ModelProfileDefinition(
    profile=future_mqa_profile,
    aliases=("Provider/Future-MQA-4B",),
)
registry = default_model_profile_registry().with_definition(definition)
layout = registry.layout_for_model("Provider/Future-MQA-4B", dtype="int8")

write_model_profile_definition_json(
    definition,
    "future-mqa-profile.json",
)
```
