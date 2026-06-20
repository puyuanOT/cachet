# Cachet: Document KV Cache

Cachet is a reusable document KV-cache orchestration package for long-context
LLM serving. It is being split out of the restaurant KV-cache experiments into
an open-source library for arbitrary textual documents. Cachet owns the work
outside the inference engine:

- document/chunk manifest lookup
- packed KV shard reading
- dynamic shard loading from Memory, Disk, and Unity Catalog Volumes
- hot CPU RAM and local disk cache tiers
- CPU-side materialization of selected document chunks
- engine-ready KV handles for vLLM, SGLang, or other KV-injection connectors
- admission metadata for external serving-engine connectors
- local tests and Databricks benchmark support

The package publishes as `document-kv-cache` and exposes the public
`document_kv_cache` import path. Cachet is the product brand; the Python
distribution and import names stay explicit for package discovery and backward
compatibility. The legacy `restaurant_kv_serving` package and
restaurant-specific aliases are still bundled as compatibility shims for
existing benchmark jobs. New code should use the document-generic names:
`DocumentKVRequest`, `DocumentChunkType.DOCUMENT_STATIC`,
`DocumentChunkType.DOCUMENT_CHUNK`, `KVCacheKey.for_document`, and manifest
`keys_for_document` methods. Core identifiers are stored as `document_id`;
legacy `restaurant_id` and `selected_restaurants` accessors remain aliases for
old callers.

## Purpose And Scope

Cachet targets applications that repeatedly serve long, mostly stable document
context with short request-specific suffixes: retrieval-heavy assistants,
semantic filtering, compliance review, internal knowledge tools, and benchmark
suites that compare cached-context serving against ordinary prefill. The V1
release focuses on Qwen3 4B Instruct on AWS g5-class hardware, with Biography,
HotpotQA, MusiQue, and Needle-in-a-Haystack benchmarks measuring quality and
latency against a standard no-cache prefill baseline.

The package deliberately stops at the engine handoff boundary. vLLM, SGLang, or
another established serving engine owns scheduling, decode, LoRA execution, and
native KV block management. Cachet provides the manifest, storage, materialized
payload, admission metadata, benchmark evidence, and adapter contracts that let
those engines reuse precomputed document context safely.

## Logical Model

Each document can be represented by stable chunks and selectable chunks:

```text
task_prefix_cache
+ document_static_cache     # metadata, description, source summary, etc.
+ document_chunk_cache      # selected sections, passages, reviews, facts, etc.
+ user/task suffix
```

The public `DocumentChunkRole` helper maps concrete chunk-type aliases onto the
canonical roles `TASK_PREFIX`, `STATIC`, and `CONTENT`. Generic document chunk
types and legacy restaurant chunk types share those roles, so manifests and
planners order task prefixes, static document context, and selected content
consistently while downstream jobs migrate away from restaurant-specific names.

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

The current implementation is storage-format-first and tensor-runtime-agnostic. It materializes ordered byte ranges; the serving integration repo owns interpretation as vLLM or SGLang KV blocks.

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

## Tiered Cache

`ChunkCache` keeps hot document chunks in a byte-bounded CPU LRU and can persist
cold misses into a local disk tier. The local tier is useful for Databricks or
serving nodes where UC Volumes or remote storage are durable but too slow to hit
for every request.

```python
cache = ChunkCache(
    cpu_max_bytes=8 * 1024**3,
    local_dir="/local_disk0/document-kv-cache",
    local_max_bytes=200 * 1024**3,
)
materializer = KVMaterializer(
    cache=cache,
    reader=RoutedRangeReader(),
)
```

The cache reports `ChunkCacheStats` with CPU hits, local disk hits, cold misses,
and current tier sizes. Call `get_or_load_with_tier()` when direct cache probes
need the payload plus the serving tier (`cpu`, `local_disk`, or `cold_storage`)
for latency attribution. Normal materialization also carries those tiers:
`MaterializedKV`, `SegmentedMaterializedKV`, and `EngineReadyRequest` expose
`segment_tiers` parallel to their ordered KV segments. Local files are keyed by
logical chunk identity plus checksum, so regenerating a chunk cannot
accidentally reuse stale local bytes from an older shard. When a reader supports
`read_many()`, normal materialization batches cold misses while preserving the
same CPU/local/cold tier attribution as single-chunk loads.

To measure storage-reader behavior on the target hardware, run the synthetic
reader benchmark. It writes one packed shard, then reports p50/p95 read latency
and throughput for Memory, Disk, and Unity Catalog Volume readers:

```bash
python -m document_kv_cache.storage_benchmark \
  --workspace-dir /local_disk0/document-kv-storage-benchmark \
  --uc-volume-root /Volumes/catalog/schema/volume/document-kv-storage-benchmark \
  --chunk-count 64 \
  --chunk-bytes 1048576 \
  --repeats 4 \
  --parallelism 8 \
  --output-json storage-benchmark.json
```

Omit `--uc-volume-root` for a local UC-reader smoke test, or repeat `--reader`
to benchmark only selected backends such as `--reader memory --reader disk`.
The JSON output includes `storage_evidence` for the readers selected in that
run and `release_storage_evidence` for the strict Memory + Disk + real Unity
Catalog Volume release gate. Both blocks report missing readers, reader errors,
absent latency or throughput metrics, and whether the UC root is backed by a
`/Volumes/<catalog>/<schema>/<volume>` path when required.

For managed Databricks execution on the target AWS g5 hardware, generate a tiny
storage-benchmark runner and `runs/submit` payload:

```bash
python -m document_kv_cache.databricks_storage_benchmark_job \
  --workspace-dir /local_disk0/document-kv-storage-benchmark \
  --benchmark-output-json /Volumes/catalog/schema/volume/storage/storage-benchmark.json \
  --uc-volume-root /Volumes/catalog/schema/volume/storage \
  --runner-python-file dbfs:/benchmarks/run_storage_benchmark.py \
  --runner-script-output run_storage_benchmark.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --output-json databricks-storage-benchmark-submit.json
```

## Workflow API

`DocumentKVWorkflow` coordinates the package-level path from source documents to serving-ready KV:

1. describe source text with `SourceDocument` and `SourceChunk`
2. optionally run a `TrainingAdapter`
3. generate `PackChunk` objects through a user-provided `KVChunkGenerator`
4. write a packed shard with manifest `ChunkRef` entries
5. prepare or enqueue `DocumentKVRequest` objects for serving

Minimal shape:

```python
layout = layout_for_model("qwen3:4b-instruct")
workflow = DocumentKVWorkflow(manifest=manifest, materializer=materializer)
result = workflow.generate_cache(
    documents=(document,),
    generator=generator,
    config=CacheBuildConfig(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        dtype="int8",
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

The generator is intentionally injected. Real implementations can use Qwen, vLLM prefill workers, or future KV Packet training while this package keeps the orchestration and storage contract stable.
`CacheGenerationMethod` labels the generation path as `VANILLA_PREFILL`,
`ADAPTER_TRAINED`, `KV_PACKET`, or `CUSTOM` so benchmark reports and downstream
engine adapters can distinguish training-free caches from adapter-trained or
packet-style caches without changing the manifest key schema.
`DocumentKVRequest` freezes the selected document/chunk map at construction
time: mapping values become immutable tuples and the mapping itself is read-only.
This prevents a queued request from changing if caller-owned lists or dicts are
mutated while the materializer or engine adapter is still preparing the payload.
Training-aware generators can return `TrainingArtifacts` with either simple
`adapter_ids` or richer `CacheAdapterArtifact` entries that record the adapter
URI, method, and metadata. `prepare_for_engine(..., training_artifacts=result.training_artifacts)`
derives the engine adapter IDs from those artifacts, while explicit
`adapter_ids` remain available for callers that already manage LoRA or packet
adapters separately.

## Serving Engine Handoff

`DocumentKVService.prepare_for_engine` turns a document request into an `EngineReadyRequest`:

```python
ready = service.prepare_for_engine(
    request,
    layout=qwen3_layout,
    metadata={"engine": "vllm"},
    adapter_ids=("selection-lora",),
)
```

The `EngineReadyRequest` contains the loaded payload, per-segment cache-tier
attribution, and a validated `KVCacheHandle` with contiguous token and byte
segments. vLLM and SGLang integrations should consume that handle, reserve
engine-native KV blocks, and inject or map the payload into their existing
schedulers. The document package intentionally stops at this boundary; it does
not implement an alternative serving engine.

For engine-specific integration code, wrap the ready request in an adapter
contract:

```python
from pathlib import Path

from document_kv_cache import (
    build_engine_adapter_request,
    build_engine_kv_connector_actions,
    build_engine_kv_injection_plan,
    engine_kv_connector_actions_from_record,
    engine_kv_connector_actions_to_record,
    engine_adapter_request_to_record,
    probe_engine_kv_connector_actions,
    read_engine_adapter_payload,
    read_engine_adapter_request_json,
    view_engine_adapter_payload,
    vllm_adapter_spec,
    write_engine_adapter_request_json,
)

adapter_request = build_engine_adapter_request(
    ready,
    spec=vllm_adapter_spec(),
)
handoff_record = engine_adapter_request_to_record(adapter_request)
payload_path = Path("/local_disk0/document-kv-cache/req-123.kv")
payload_path.parent.mkdir(parents=True, exist_ok=True)
with payload_path.open("wb") as payload_file:
    if isinstance(ready.payload, bytes):
        payload_file.write(ready.payload)
    else:
        for segment_payload in ready.payload:
            payload_file.write(segment_payload)
write_engine_adapter_request_json(
    adapter_request,
    "req-123-handoff.json",
    payload_uri=f"disk:{payload_path}",
)

# Inside the vLLM/SGLang connector process:
record = read_engine_adapter_request_json(
    "req-123-handoff.json",
    expected_backend="vllm",
)
payload = read_engine_adapter_payload(
    record["payload_source"]["uri"],
    expected_bytes=record["payload_source"]["total_bytes"],
)
payload_or_segments = view_engine_adapter_payload(record, payload)
injection_plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
actions = build_engine_kv_connector_actions(injection_plan, payload_or_segments)
actions_record = engine_kv_connector_actions_to_record(actions)
actions = engine_kv_connector_actions_from_record(actions_record, expected_backend="vllm")

reservation = vllm_block_manager.reserve(
    request_id=actions.reservation.request_id,
    block_count=actions.reservation.total_blocks,
)
for copy in actions.copies:
    source = payload_or_segments if copy.payload_index is None else payload_or_segments[copy.payload_index]
    engine_kv_importer.copy(
        reservation,
        source[copy.source_byte_start : copy.source_byte_end],
        token_start=copy.token_start,
        block_range=(copy.first_block_index, copy.last_block_index_exclusive),
    )
vllm_scheduler.bind_document_kv(reservation, actions.bind)
```

Adapter smoke tests can wrap the real native block manager in the small
`EngineKVBlockManagerProbe` protocol and call
`probe_engine_kv_connector_actions(actions, payload_or_segments, probe)`. That
validation path exercises native reservation, KV import, handle binding, and
release using the exact descriptors that production serving will consume, but it
does not schedule decode; vLLM or SGLang still owns request scheduling, LoRA
routing, token generation, and final cleanup.
`engine_kv_connector_actions_to_record` emits the JSON-compatible
`document_kv.engine_kv_connector_actions.v1` descriptor for out-of-process native
adapters. The record contains the request id, backend, reservation geometry,
copy offsets, bind metadata, and release action, but not raw KV payload bytes.
Use `engine_kv_connector_actions_from_record` or
`validate_engine_kv_connector_actions_record` at the engine boundary to reject a
stale backend, schema, unsupported action field, payload offset, or
layout-derived block map before making native block-manager calls.
Serialize successful probe summaries with
`engine_kv_connector_probe_result_to_record` so release validation can verify
that both native backends exercised reservation, import, bind, and release with
the full Qwen3 GQA layout metadata and a real native engine version. Release
evidence rejects records that only claim `qwen3:4b-instruct` without the
matching one-byte Qwen3 GQA geometry. Engine probe records include
`schema_version` `2`; schema v2 makes the nested `layout` mapping mandatory.
For a reproducible CLI handoff, expose a small native factory in the engine
adapter package and run:

```bash
python -m document_kv_cache.engine_probe \
  --handoff-json req-123-handoff.json \
  --probe-factory my_vllm_adapter.probes:build_probe \
  --expected-backend vllm \
  --actions-output-json vllm-connector-actions.json \
  --output-json vllm-engine-probe.json
```

For managed Databricks execution on the target AWS g5 hardware, generate a tiny
engine-probe runner and `runs/submit` payload:

```bash
python -m document_kv_cache.databricks_engine_probe_job \
  --handoff-json /Volumes/catalog/schema/volume/probes/vllm-handoff.json \
  --probe-factory my_vllm_adapter.probes:build_probe \
  --probe-output-json /Volumes/catalog/schema/volume/probes/vllm-engine-probe.json \
  --actions-output-json /Volumes/catalog/schema/volume/probes/vllm-connector-actions.json \
  --payload-uri /Volumes/catalog/schema/volume/probes/vllm-payload.kv \
  --runner-python-file dbfs:/benchmarks/run_engine_probe.py \
  --runner-script-output run_engine_probe.py \
  --expected-backend vllm \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --release-safe \
  --output-json databricks-engine-probe-submit.json
```

For release runs, prefer a two-backend probe target file and one release-safe
Databricks payload so vLLM and SGLang exercise the same descriptor contract on
the same AWS g5 policy:

```json
{
  "probes": [
    {
      "backend": "vllm",
      "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
      "probe_factory": "my_vllm_adapter.probes:build_probe",
      "output_json": "/Volumes/catalog/schema/volume/probes/vllm-engine-probe.json",
      "actions_output_json": "/Volumes/catalog/schema/volume/probes/vllm-connector-actions.json",
      "payload_uri": "/Volumes/catalog/schema/volume/probes/vllm-payload.kv"
    },
    {
      "backend": "sglang",
      "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
      "probe_factory": "my_sglang_adapter.probes:build_probe",
      "output_json": "/Volumes/catalog/schema/volume/probes/sglang-engine-probe.json",
      "actions_output_json": "/Volumes/catalog/schema/volume/probes/sglang-connector-actions.json",
      "payload_uri": "/Volumes/catalog/schema/volume/probes/sglang-payload.kv"
    }
  ]
}
```

```bash
python -m document_kv_cache.databricks_engine_probe_job \
  --backend-config-json engine-probe-targets.json \
  --runner-python-file dbfs:/benchmarks/run_engine_probe.py \
  --runner-script-output run_engine_probe.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --release-safe \
  --output-json databricks-engine-probes-submit.json
```

In `--release-safe` matrix mode, the submit payload must contain exactly one
native vLLM task and one native SGLang task. Debug fallbacks such as
`engine_version` overrides or `allow_non_native_probe` are rejected before the
Databricks job is written.
When `--actions-output-json` or a target `actions_output_json` is present, the
runner also writes the validated `document_kv.engine_kv_connector_actions.v1`
reserve/copy/bind/release descriptor next to the probe evidence. This sidecar is
useful for auditing exactly which native block-manager actions were exercised.

If the workspace uses Databricks Asset Bundles, the same native-probe contract is
available as a standalone template under `databricks/engine-probe/databricks.yml`.

The factory receives an `EngineKVProbeFactoryContext` containing the validated
handoff record, injection plan, backend, and payload URI. It should return an
`EngineKVProbeFactoryResult` with a real `EngineKVBlockManagerProbe`, the native
engine version, and optional metadata. Debug/test factories should return
`native_probe=False`, which forces the output record to remain non-releaseable
even if the CLI default is native. The runner reads local disk, DBFS, or UC
Volume payload URIs, builds the reserve/copy/bind/release descriptors, executes
the probe, and writes a `document_kv.engine_kv_connector_probe.v1` JSON record.
Probe records include runner-owned metadata for the handoff JSON, payload URI,
probe factory, and expected backend, so native evidence can be traced back to the
exact adapter boundary that produced it.
`--allow-non-native-probe` exists only for local adapter debugging; release
evidence rejects those records. Use `--release-safe` on the Databricks
`runs/submit` helper to fail fast if debug-only probe flags are present.
The built-in reserved vLLM/SGLang probe factories fail closed until real
block-manager adapters exist, but
`builtin_native_probe_factories_to_record()` reports that status together with
the pinned isolated serving-environment profile for each backend. Use that
diagnostic record when preparing native adapter work so the target engine
versions and dependency constraints stay tied to the probe entry points:

```bash
document-kv-serving-env \
  --output-json serving-environment-profiles.json

document-kv-native-probe-factories \
  --output-json native-probe-factories.json
```

`EngineAdapterRequest` records the target backend (`vllm` or `sglang`), payload
mode (`merged` or `segmented`), expected external package, required injection
steps, and namespaced metadata such as `document_kv.handle_uri` and
`engine.kv_injection_method`. `engine_adapter_request_to_record` turns that plan
into a JSON-serializable handoff artifact with `record_type`
`document_kv.engine_adapter_request.v1` and `schema_version` `2`. The record
contains the handle URI, payload source descriptor, model layout, token/byte
segment boundaries, per-segment cache-tier attribution, adapter ids, required
steps, and estimated GPU bytes. It intentionally omits raw KV payload bytes.
`write_engine_adapter_request_json`
therefore requires an adapter-readable `payload_uri` (or an external
`handle_uri`) by default; use the in-process `EngineAdapterRequest` directly
when the connector already has access to `ready.payload`.
`read_engine_adapter_request_json` validates the schema, expected backend,
payload source, layout geometry, and contiguous byte/token segments before a
connector consumes the record. `read_engine_adapter_payload` reads absolute
local paths, `disk:`, `file:`, `dbfs:`, or UC Volume payload URIs and checks the
expected byte length before the connector views the payload.
`view_engine_adapter_payload` validates the loaded byte length and returns
either one `memoryview` over the merged payload or per-segment `memoryview`
slices matching the handle boundaries. Use `split_engine_adapter_payload` only
when a connector needs independent `bytes` objects and can afford that extra
copy. `build_engine_kv_injection_plan` derives a
reference reservation/copy map from the validated record: total native KV blocks
to reserve from `layout.block_size`, source byte spans, destination token spans,
block-index ranges, adapter ids, and engine metadata. Segment block ranges may
overlap when a document segment starts or ends inside an engine block; reserve
using the plan-level `total_blocks`, then use segment bindings to map byte spans
onto the affected token/block spans. `build_engine_kv_connector_actions` turns
that plan plus the loaded payload into reserve/copy/bind/release descriptors.
Those descriptors carry source offsets, payload indexes, token spans, and block
ranges, but not raw KV bytes, so adapters can slice or stream from their
existing payload buffer without duplicating large caches in Python objects. The
optional `engine_kv_connector_actions_to_record` step serializes those
descriptors as `document_kv.engine_kv_connector_actions.v1` for native adapters
that cross a process boundary; the matching parser validates the record back
into the same reserve/copy/bind/release dataclasses before the engine touches
its block manager.
The actual adapter remains a thin integration layer inside the vLLM or SGLang
serving process: it translates the descriptors to native block-manager calls,
binds the handle to the request, and lets the serving engine own scheduling,
decode, LoRA routing, and release. The nested `layout` record also carries
`storage_layout`, so adapters can distinguish separate key/value planes from
interleaved key/value payloads and shared K/V base storage when mapping bytes
into engine-native blocks. Treat `storage_layout` as a compatibility guard, not
a complete tensor-stride language: the exact byte order is defined by
`layout_version` plus the model profile. The V1 `qwen3-v1` payload contract is
token-major, then layer-major, then logical key/value plane, then KV-head, with
head elements contiguous according to `kv_stride_bytes`.

## Model Profiles

`KVModelProfile` centralizes model attention geometry so cache generators,
manifests, and engine adapters derive the same `KVLayout`. V1 includes
`QWEN3_4B_INSTRUCT_PROFILE`, exposed through both `qwen3:4b-instruct` and
`Qwen/Qwen3-4B` aliases. A profile records the query-head count, KV-head count,
head size, layer count, context limit, default dtype, and layout version, then
derives bytes per cached token for MHA, GQA, or MQA-style caches. `KVLayout`
also records a `KVStorageLayout`: `separate_key_value` for distinct key/value
planes, `interleaved_key_value` for connector-specific K/V interleaving, and
`shared_key_value` when K and V views share a base allocation and the adapter
must preserve that relationship. Cache generators and manifest refs persist the
same `storage_layout`; serving validation rejects a request if persisted chunks
and the engine handoff layout disagree.

```python
from document_kv_cache import layout_for_model

layout = layout_for_model(
    "qwen3:4b-instruct",
    dtype="int8",
    lora_id="selection-lora",
)
```

Future Qwen3.5, MiniMax, or adapter-trained KV Packet integrations should add a
profile first, then implement a generator or engine adapter against the same
layout contract. Use a caller-owned `ModelProfileRegistry` for profiles that are
not part of the V1 built-ins, or ship a portable `ModelProfileDefinition` JSON
artifact when the profile belongs to an external model bundle. The JSON artifact
uses a closed top-level schema; place external bundle annotations in `metadata`
rather than adding ad hoc top-level fields. The example below uses a schematic
MQA-style future profile; real model bundles should replace the
illustrative geometry with measured values from the target model:

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

## Benchmark Contract

The V1 benchmark surface targets Biography, HotpotQA, MusiQue, and NIAH on AWS g5 with Qwen3 4B Instruct. It defines a common schema for comparing the no-cache prefill baseline with document KV-cache reuse:

- `BenchmarkExample` captures one dataset example, query, expected answer, and selected source documents.
- `BenchmarkDatasetSpec` records the canonical V1 instruction style for Biography, HotpotQA, MusiQue, and NIAH.
- `build_prompt_parts` splits each example into `system_prompt`, `document_context`, and `user_prompt`.
- `build_prefill_prompt` concatenates all three parts for the no-cache baseline.
- `build_cache_prefix_text` returns the text that should be represented by cached KV, while `build_cache_suffix_text` returns the query suffix appended at inference time.
- `load_benchmark_jsonl` and `load_v1_jsonl_suite` load normalized JSONL plus common HotpotQA-style `context` pairs and MusiQue-style `paragraphs`, without adding a hard dependency on any one dataset host.
- `normalize_v1_record`, `convert_v1_jsonl`, and `build_niah_record` prepare raw Biography, HotpotQA, MusiQue, and synthetic/source NIAH rows into that normalized JSONL contract.
- `build_v1_benchmark_plan` and the `benchmark_plan` CLI emit a portable command plan that prepares all four V1 datasets, runs the OpenAI-compatible benchmark on AWS g5/Qwen3, and can append storage-reader benchmarking on the same node.
- `benchmark_plan_executor` and `databricks_job` let managed job runners execute that plan on single-node AWS g5 Databricks clusters; `databricks_runs` can submit/check those payloads using credentials supplied only through environment variables.
- `run_benchmark_suite` executes caller-provided baseline and KV-cache engines against the same logical prompt parts and emits `InferenceMeasurement` rows. Cache engines receive the runtime suffix as `prompt_text`; the full logical prompt remains available as `logical_prompt_text`.
- `InferenceMeasurement` records prompt tokens, completion tokens, TTFT, time-to-completion, generated text, expected answer, and errors. OpenAI-compatible V1 measurements also carry `logical_prompt_tokens` and `runtime_prompt_tokens` metadata so release evidence proves the no-cache baseline saw the full prompt and the KV-cache arm generated from a smaller runtime suffix.
- `summarize_measurements` produces per-dataset/per-arm latency and quality rows.
- `compare_to_baseline` reports cache speedups and quality deltas against `baseline_prefill`.
- `evaluate_v1_benchmark_evidence` marks whether the report is release-evaluable:
  all four V1 datasets, baseline/cache rows, comparisons with speedups and
  quality deltas, successful latency measurements, and quality rates must be
  present.

The package intentionally does not own model execution. Databricks jobs, vLLM adapters, SGLang adapters, or local harnesses provide `BenchmarkEngine` implementations; the package keeps dataset parsing, prompt pairing, error capture, summaries, and baseline comparison consistent.

For a vLLM or SGLang server exposing the OpenAI-compatible completions API, use the stdlib-only streaming engine:

```python
engine = OpenAICompatibleCompletionEngine(
    OpenAICompatibleEngineConfig(
        base_url="http://localhost:8000",
        max_tokens=128,
        stream=True,
    )
)
```

Streaming responses provide TTFT; non-streaming responses report TTFT equal to total completion latency. If the server omits usage fields, the engine falls back to a simple token counter that callers can replace with a tokenizer-aware implementation.

By default this engine posts the full logical prompt, which is the correct behavior for ordinary OpenAI-compatible vLLM/SGLang servers and for platform-managed prefix caching. Set `prompt_text_mode="runtime"` only when the server endpoint is a KV-aware adapter or proxy that binds the cached prefix out of band and expects only the runtime suffix in the `prompt` field. The engine records both logical and runtime prompt-token counts in measurement metadata; strict V1 release evidence rejects cache measurements whose runtime prompt is not smaller than the logical prompt.

To run the V1 benchmark contract against existing OpenAI-compatible vLLM or SGLang servers, generate a reproducible command plan for all four datasets:

```bash
python -m document_kv_cache.benchmark_plan \
  --raw-dataset biography=/raw/biography.jsonl \
  --raw-dataset hotpotqa=/raw/hotpotqa.jsonl \
  --raw-dataset musique=/raw/musique.jsonl \
  --raw-dataset niah=/raw/niah.jsonl \
  --prepared-dir /data/v1-prepared \
  --base-url http://localhost:8000 \
  --storage-benchmark-workspace-dir /local_disk0/document-kv-storage-benchmark \
  --storage-benchmark-uc-volume-root /Volumes/catalog/schema/volume/document-kv-storage-benchmark \
  --engine-probe-handoff-json vllm=/data/vllm-handoff.json \
  --engine-probe-output-json vllm=/data/vllm-engine-probe.json \
  --engine-probe-actions-output-json vllm=/data/vllm-connector-actions.json \
  --engine-probe-handoff-json sglang=/data/sglang-handoff.json \
  --engine-probe-output-json sglang=/data/sglang-engine-probe.json \
  --engine-probe-actions-output-json sglang=/data/sglang-connector-actions.json \
  --engine-probe-use-builtin-factories \
  --release-evidence-output-json /data/release-evidence.json \
  --release-preflight-output-json /data/release-inputs.json \
  --release-bundle-output-dir /data/document-kv-release-bundle \
  --release-bundle-output-json /data/release-bundle-manifest.json \
  --release-bundle-databricks-run-status-json /data/databricks-run-status.json \
  --release-bundle-package-wheel /data/dist/document_kv_cache-0.2.0-py3-none-any.whl \
  --release-bundle-pr-evidence-json /data/pr-evidence/release-provenance.json \
  --github-governance-output-json /data/github-governance.json \
  --repository-hygiene-output-json /data/repository-hygiene.json \
  --native-probe-factories-output-json /data/native-probe-factories.json \
  --engine-probe-targets-output-json /data/engine-probe-targets.json \
  --engine-probe-targets-release-safe \
  --plan-output-json /data/v1-plan.json \
  --plan-output-sh /data/run-v1-benchmark.sh
```

The generated shell script runs `dataset_prep` for each raw file, invokes
`benchmark_runner`, and, when `--storage-benchmark-workspace-dir` is provided,
appends a `document_kv_cache.storage_benchmark` command. The storage command
captures Memory and Disk reader latency/throughput on the same AWS g5 node, and
adds the Unity Catalog reader only when `--storage-benchmark-uc-volume-root`
points at a real UC Volume. It writes `<suite-id>-storage-benchmark.json` under
`--prepared-dir` unless `--storage-benchmark-output-json` is set. Backend-keyed
`--engine-probe-*` options append native `document_kv_cache.engine_probe`
commands for vLLM and SGLang handoffs; release evidence automatically consumes
those planned probe and connector-action outputs. Planned probes consumed by
release evidence must include `--engine-probe-actions-output-json` for each
backend and cannot use debug-only `--engine-probe-engine-version` or
`--allow-non-native-engine-probe`; if debug probe commands are also needed, pass
separate native records directly with repeatable `--release-engine-probe-json`
and `--release-engine-actions-json`.
The benchmark plan executor treats generated plan JSON as a closed execution
schema: unsupported top-level plan keys or per-command keys are rejected before
any command is run.
`--engine-probe-use-builtin-factories` fills missing planned factories with
package-owned vLLM/SGLang factory paths. Those factories are stable release-plan
targets but still fail closed until the backend-native block-manager adapter is
available; pass explicit `--engine-probe-factory BACKEND=MODULE:CALLABLE` to use
a downstream adapter module.
Add `--github-governance-output-json` to the benchmark plan to emit the
matching `document_kv.github_repository_governance.v1` sidecar using
environment-provided GitHub credentials; when release-bundle assembly is also
enabled, the generated sidecar is included in the bundle automatically. Use
`--release-bundle-github-governance-json` only when the sidecar was generated
outside the command plan.
Add `--repository-hygiene-output-json` to the benchmark plan to emit the
matching `document_kv.repository_hygiene.v1` sidecar; when release-bundle
assembly is also enabled, the generated sidecar is included in the bundle
automatically. Use `--release-bundle-repository-hygiene-json` only when the
sidecar was generated outside the command plan.
Add `--native-probe-factories-output-json` to the benchmark plan to emit the
matching `document_kv.native_probe_factories.v1` diagnostics sidecar; when
release-bundle assembly is also enabled, the generated sidecar is included in
the bundle automatically.
Add `--release-preflight-output-json` to the benchmark plan to emit the
matching `document_kv.release_evidence_inputs.v1` sidecar before release
validation; when release-bundle assembly is also enabled, the generated
preflight sidecar is included in the bundle automatically. Use
`--release-bundle-preflight-json` only when the preflight sidecar was generated
outside the command plan.
If native probe and connector-action records already exist, skip the planned
probe flags and pass them directly with repeatable `--release-engine-probe-json`
and `--release-engine-actions-json`.
`--engine-probe-targets-output-json` writes a
`document_kv.engine_probe_targets.v1` sidecar whose `probes` array is directly
accepted by `document_kv_cache.databricks_engine_probe_job --backend-config-json`.
The Databricks engine-probe job treats this target file as a closed schema:
unsupported top-level target keys or per-probe keys are rejected before a
run-submit payload is produced.
Use `--engine-probe-targets-release-safe` for release jobs; it requires exactly
one native vLLM probe and one native SGLang probe and rejects debug-only planned
probe settings before writing the target file. Release-safe targets also carry
each backend's `actions_output_json` path so the Databricks probe jobs write the
sidecars required by release evidence.
When release-evidence flags are supplied, the plan appends
`document_kv_cache.release_evidence` last so the V1 benchmark, storage
benchmark, and vLLM/SGLang probe artifacts are validated together. Release
evidence also checks that each native probe reports the pinned runtime engine
version and carries the serving-engine package/version metadata from
`document_kv_cache.serving_env`. Add
`--release-bundle-output-dir` to append `document_kv_cache.release_bundle`
after that validation step; the bundle command copies the validated V1,
storage, engine-probe, release-evidence artifacts plus any generated or supplied preflight,
Databricks run-status, wheel, PR-evidence, GitHub governance, and native-probe
factory diagnostics sidecars into a checksummed handoff directory. The benchmark
plan executor writes `plan-execution.json` only after the planned commands
finish, so the final strict release bundle should be built as a separate
post-execution command with
`--plan-execution-json /data/plan-execution.json --require-complete-v1`, or by
supplying `--release-bundle-plan-execution-json` together with
`--release-bundle-require-complete-v1` when planning a follow-up bundle command.
Strict follow-up bundle plans are checked before commands are emitted: the
planner requires the preflight, plan-execution, Databricks run-status, tested
wheel, PR evidence, GitHub governance, repository hygiene, and native probe
factory diagnostics sidecars to be supplied or generated by the plan.
For ad hoc local conversion, use `dataset_prep`
directly:

```bash
python -m document_kv_cache.dataset_prep \
  --dataset hotpotqa \
  --input-jsonl /data/raw_hotpotqa.jsonl \
  --output-jsonl /data/hotpotqa.jsonl

python -m document_kv_cache.dataset_prep \
  --dataset niah \
  --haystack-file /data/haystack.txt \
  --needle-answer "blue lantern" \
  --count 100 \
  --output-jsonl /data/niah.jsonl
```

Then run the benchmark manually or execute the generated shell script:

```bash
python -m document_kv_cache.benchmark_runner \
  --dataset biography=/data/biography.jsonl \
  --dataset hotpotqa=/data/hotpotqa.jsonl \
  --dataset musique=/data/musique.jsonl \
  --dataset niah=/data/niah.jsonl \
  --base-url http://localhost:8000 \
  --model-id qwen3:4b-instruct \
  --hardware-target aws-g5 \
  --output-json v1-results.json
```

For a self-contained Databricks AWS g5 smoke of the actual vLLM server path, use
`document_kv_cache.vllm_smoke` from a GPU task. It creates an isolated vLLM
environment on local NVMe, installs the pinned serving dependency stack,
starts `Qwen/Qwen3-4B-Instruct-2507` as `qwen3:4b-instruct`, runs one tiny
Biography/HotpotQA/MusiQue/NIAH example through the OpenAI-compatible benchmark
runner, and writes `metadata.json`, `vllm-import-probe.json`,
`vllm-server.log`, and `v1-benchmark.json` to the chosen output directory:

```bash
python -m document_kv_cache.vllm_smoke \
  --benchmark-id v1_vllm_smoke_001 \
  --output-dir /Volumes/catalog/schema/volume/document-kv-v1-smoke
```

To launch that same smoke through a Databricks managed task, upload the wheel
and generated runner script, then emit a single-node AWS g5 `runs/submit`
payload:

```bash
python -m document_kv_cache.databricks_vllm_smoke_job \
  --benchmark-id v1_vllm_smoke_001 \
  --output-dir /Volumes/catalog/schema/volume/document-kv-v1-smoke \
  --runner-python-file dbfs:/benchmarks/run_vllm_smoke.py \
  --runner-script-output run_vllm_smoke.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --output-json databricks-vllm-smoke-submit.json
```

This emits suite metadata, per-request measurements, per-dataset quality/latency
summaries, cache-vs-baseline comparisons, and a `v1_evidence` block. Smoke runs
against one dataset are allowed, but `v1_evidence.ok` is false until all required
V1 datasets and metrics are present. With the default `/v1/completions`
endpoint, the runner accepts server roots such as `http://localhost:8000` or
OpenAI-style bases such as `http://localhost:8000/v1` without generating
`/v1/v1/completions`. Custom `--endpoint` and `--cache-endpoint` values are
appended to the exact base URL you provide. The baseline arm posts the full
prompt. The cache arm also posts the full logical prompt by default; add
`--cache-base-url` and `--cache-runtime-prompt` only when that explicit cache
endpoint is a KV-aware adapter or proxy that binds the cached prefix out of band.

After the real V1 benchmark, storage benchmark, and native vLLM/SGLang probe
runs complete, validate the collected release artifacts:

```bash
python -m document_kv_cache.release_evidence \
  --v1-benchmark-json v1-results.json \
  --storage-benchmark-json storage-benchmark.json \
  --engine-probe-json vllm-probe.json \
  --engine-probe-json sglang-probe.json \
  --engine-actions-json vllm-connector-actions.json \
  --engine-actions-json sglang-connector-actions.json \
  --output-json release-evidence.json
```

To check which release inputs are present before strict validation, add
`--preflight-only` or write a sidecar preflight record:

```bash
python -m document_kv_cache.release_evidence \
  --v1-benchmark-json v1-results.json \
  --storage-benchmark-json storage-benchmark.json \
  --engine-probe-json vllm-probe.json \
  --engine-actions-json vllm-connector-actions.json \
  --preflight-only \
  --preflight-output-json release-inputs.json
```

The preflight sidecar keeps malformed JSON, missing paths, and wrong
role/record-type assignments separate. A readable JSON file with
`document_kv.engine_kv_connector_actions.v1` in an `--engine-probe-json` slot is
reported under `invalid_record_type_paths` instead of being treated as a valid
probe input. Release bundles treat release-evidence and preflight sidecars as
closed schemas, rejecting unsupported top-level keys before copying artifacts
into the bundle.

The command returns exit code `0` only when the V1 benchmark is AWS g5/Qwen3,
the storage benchmark has strict Memory + Disk + real UC Volume evidence, and
exactly one native engine probe record plus one connector action descriptor is
present for each vLLM/SGLang backend. V1 comparison rows must include finite
numeric quality deltas and positive finite latency speedups. Each native probe
must also report the pinned runtime engine version and include serving-engine
package/version metadata matching the `document_kv_cache.serving_env` backend
profile. Each
action descriptor must validate against the reserve/copy/bind/release schema and
the same one-byte Qwen3 GQA layout contract. For each backend, release evidence
also cross-checks that the connector-action sidecar describes the same request,
block count, copied token count, copied byte count, copied segment count, payload
mode, and exact layout record as the native probe record it audits.
The V1 benchmark artifact must identify itself with
`record_type=document_kv.benchmark_run.v1`; storage, engine-probe, and
connector-action artifacts have matching record-type checks. Successful V1
measurement rows must carry positive prompt and completion token counts, and
report rows must carry positive prompt-token, completion-token, and
output-throughput summaries, so zero-token or summary-only benchmark stubs
cannot pass release evidence. The release JSON output also includes
`artifact_sources` entries for the exact V1 benchmark, storage benchmark,
engine-probe, and connector-action files that were evaluated, including
`size_bytes` and `sha256` fingerprints, so release records remain auditable
after artifacts are copied into durable storage. Release bundles still re-run
validation for older sidecars without fingerprints; when fingerprints are
present, they must match the bundled input payloads.

To create that durable handoff directory, bundle the validated inputs together
with checksums:

```bash
python -m document_kv_cache.release_bundle \
  --v1-benchmark-json v1-results.json \
  --storage-benchmark-json storage-benchmark.json \
  --engine-probe-json vllm-probe.json \
  --engine-probe-json sglang-probe.json \
  --engine-actions-json vllm-connector-actions.json \
  --engine-actions-json sglang-connector-actions.json \
  --release-evidence-json release-evidence.json \
  --preflight-json release-inputs.json \
  --plan-execution-json plan-execution.json \
  --databricks-run-status-json databricks-run-status.json \
  --package-wheel dist/document_kv_cache-0.2.0-py3-none-any.whl \
  --pr-evidence-json pr-evidence/release-provenance.json \
  --github-governance-json github-governance.json \
  --repository-hygiene-json repository-hygiene.json \
  --native-probe-factories-json native-probe-factories.json \
  --require-complete-v1 \
  --output-dir document-kv-release-bundle \
  --output-json release-bundle-manifest.json
```

The bundle directory contains byte-for-byte copies of each JSON artifact plus a
`manifest.json` with the original path, bundled relative path, record type,
backend where applicable, package name/version for wheel artifacts, size, and
SHA-256 for every artifact. Add
`--plan-execution-json` to include the command execution summary that identifies
the exact benchmark plan JSON, `--package-wheel` to include the exact wheel
tested on the target AWS g5 runtime, and repeat `--pr-evidence-json` to carry PR
traceability records alongside the benchmark, storage, engine-probe,
connector-action, release-evidence, and preflight artifacts. Add
`--require-complete-v1` for release publishing; this strict mode refuses to
build a V1 bundle unless the release evidence, preflight, vLLM/SGLang connector
action, benchmark-plan execution, Databricks run-status, tested wheel, PR
evidence, GitHub governance, repository hygiene, and native-probe factory
diagnostics sidecars are all present; the native factory diagnostics must also
report supported built-in vLLM and SGLang factory entry points.
Benchmark-plan execution sidecars are validated as closed schemas, including
each command entry and the embedded plan-source provenance record, before they
are copied into the release bundle. Add
`--github-governance-json` to
include the repository visibility and branch-protection sidecar emitted by
`document_kv_cache.github_governance`; the bundle rejects it unless the
governance record is release-ready. Add `--repository-hygiene-json` to include
the `.gitignore`, tracked-artifact, and untracked-artifact sidecar emitted by
`document_kv_cache.repository_hygiene`; release bundles reject it unless no
generated, build, cache, or secret-like artifacts are tracked or exposed as
untracked and no tracked paths differ from `HEAD`. Repeat
`--native-probe-factories-json` for
`document_kv.native_probe_factories.v1` diagnostics emitted by
`document_kv_cache.native_probe_factories`; the bundle validates that the
diagnostics cover the built-in vLLM and SGLang native factory entry points, and
strict V1 release bundles require both entries to be supported.
Repeat
`--databricks-run-status-json` for compact `databricks_runs get --summary`
outputs, or extracted inner `summary` records, from the managed runs whose
outputs are included in the bundle; release bundles require those status
sidecars to be terminal, successful, and attached to a hashed single-node AWS g5
`SINGLE_USER` submit-payload summary. Do not use the `--include-response` debug
flag for release-bundle status sidecars; bundles reject raw Jobs API responses.
PR evidence sidecars must be valid `document_kv.pr_evidence.v1` records with
Refactor-skill evidence, completed GPT-5.5 review, and any GPT-5.5 findings
marked resolved before they can enter the release bundle. They are also
validated as closed schemas so ad hoc review notes or debug payloads stay out of
the release handoff.
Wheel artifacts must be valid wheels for `document-kv-cache` with non-empty
package metadata, and the wheel filename, root `.dist-info` directory, and
`METADATA` name/version must describe the same normalized package identity. The
release bundle manifest records the normalized package name and package version
for the copied wheel so the tested package can be audited without reopening the
wheel archive. Strict V1 release bundles also require the wheel version to match
the current project version from `pyproject.toml` or installed package metadata.
The wheel's `WHEEL` metadata must also describe a pure Python `py3-none-any`
artifact, and the wheel `RECORD` manifest must list the package payload plus the
required `.dist-info` metadata with matching hashes and sizes. Wheel zip members
must use unique file paths so the `RECORD` manifest cannot hide duplicate
installed files.

Capture repository governance status as a separate release-readiness sidecar.
The command reads the GitHub token from an environment variable, records
repository visibility, `main` branch protection state, and open pull-request
pressure, and returns non-zero until the repository is public, the required
`Test and build` protection is active, and no unexpected pull requests remain
open:

```bash
export GITHUB_TOKEN=...
python -m document_kv_cache.github_governance \
  --repository OWNER/document-kv-cache \
  --output-json github-governance.json
```

When producing a sidecar from inside the current release PR, pass
`--allow-open-pull-request-number <PR_NUMBER>` so that one active pull request is
recorded but not treated as stale release pressure.

If GitHub reports that private-repository branch protection is unavailable, the
sidecar records `ok=false` and the release remains process-only rather than
enforced by GitHub settings.

Capture repository hygiene as a separate release-readiness sidecar:

```bash
python -m document_kv_cache.repository_hygiene \
  --repository-root . \
  --output-json repository-hygiene.json
```

The sidecar records the required `.gitignore` patterns, tracked and untracked
path counts, tracked generated or secret-like artifact paths, and untracked
generated or secret-like paths that Git still exposes as non-ignored files. It
also records dirty tracked paths so release handoffs prove they were produced
from a clean tracked worktree. It returns non-zero until all required ignore
patterns are present, no forbidden artifact paths are tracked or exposed as
untracked, and no tracked files differ from `HEAD`.

For Databricks-managed execution, upload the package wheel, the generated benchmark plan JSON, and a small runner script, then generate a single-node AWS g5 `runs/submit` payload:

```bash
python -m document_kv_cache.databricks_job \
  --plan-json-uri dbfs:/benchmarks/v1-plan.json \
  --runner-python-file dbfs:/benchmarks/run_plan.py \
  --runner-script-output run_plan.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --output-json databricks-run-submit.json
```

Submit or inspect a generated payload with env-provided credentials, keeping
tokens out of command arguments and JSON artifacts:

```bash
export DATABRICKS_HOST=https://dbc-...cloud.databricks.com
export DATABRICKS_TOKEN=...
python -m document_kv_cache.databricks_runs \
  --output-json databricks-submit-response.json \
  submit \
  --payload-json databricks-run-submit.json

python -m document_kv_cache.databricks_runs \
  --output-json databricks-run-status.json \
  get \
  --run-id 123456789 \
  --summary \
  --submit-payload-json databricks-run-submit.json
```

When the managed runner is configured with `--execution-result-json-uri`, the
result record includes a `plan_source` block with the original plan path, driver
path, byte size, SHA-256, suite/model/hardware fields when present, and command
count. This lets release operators tie Databricks command execution back to the
exact benchmark plan JSON that was uploaded.
The `databricks_runs get --summary` output adds a compact
`document_kv.databricks_run_status.v1` block with run/task lifecycle states,
result state, active task key, cluster ids, timing, and success/terminal flags,
plus a sanitized `document_kv.databricks_run_submit_payload.v1` hash and cluster
summary when `--submit-payload-json` is supplied, so polling automation does not
need to parse or persist the full Jobs API response. Add `--include-response`
only when debugging requires the raw `runs/get` payload.

Workspace-specific automation can still POST the payload itself after applying
the organization’s auth, cluster policy, and asset-upload conventions. Teams
that manage jobs declaratively can use the reference Databricks Asset Bundle in
`databricks/databricks.yml`; it mirrors the same single-node AWS g5 benchmark
contract without embedding workspace credentials or paths.
Standalone bundles for the runtime smoke, storage-reader benchmark, and native
engine-probe evidence live under `databricks/vllm-smoke/`,
`databricks/storage-benchmark/`, and `databricks/engine-probe/`.
Installed package users can list or extract those packaged templates without
cloning the repository:

```bash
document-kv-templates list --prefix databricks
document-kv-templates extract \
  --prefix databricks/storage-benchmark \
  --output-dir ./document-kv-templates
```

The emitted cluster and bundle default to `SINGLE_USER` access mode so the
storage benchmark can read and write real Unity Catalog Volume paths under
`/Volumes/...`. The Python helper requires an explicit `--single-user-name` for
`SINGLE_USER` runs; the Asset Bundle defaults that variable to
`${workspace.current_user.userName}`.

To run the native engine probes as their own managed Databricks job from the
planned target file:

```bash
python -m document_kv_cache.databricks_engine_probe_job \
  --backend-config-json /data/engine-probe-targets.json \
  --runner-python-file dbfs:/benchmarks/run_engine_probe.py \
  --runner-script-output run_engine_probe.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --release-safe \
  --output-json /data/databricks-engine-probes-submit.json
```

Workspace-specific automation can wrap this handoff by building or reusing the
wheel, writing the plan and runner, uploading those small artifacts to the
workspace storage layer, selecting a real UC Volume path for the storage-reader
benchmark, and submitting the single-node AWS g5 job. Raw
benchmark datasets should already be visible to the cluster as `disk:...`,
`file:...`, `dbfs:/...`, `/dbfs/...`, `/Volumes/...`, `uc-volume:/...`, or `uc-volume://...` paths; the
command plan normalizes those storage URIs on the Databricks driver before
reading or writing JSONL artifacts.

To verify a live endpoint without adding custom serving code, run the OpenAI-compatible smoke check against a vLLM or SGLang server:

```bash
python -m document_kv_cache.live_server \
  --base-url http://localhost:8000 \
  --model-id qwen3:4b-instruct
```

The command prints a JSON record with TTFT, time-to-completion, token counts, `prompt_token_source`, answer-found quality, and the prompt mode used. Add `--cache-arm --runtime-prompt` only for a KV-aware proxy that injects the cached prefix out of band; ordinary OpenAI-compatible servers should use the default full-prompt mode.

## Development

This package uses Poetry metadata with exact direct dependency pins: Python
`>=3.11,<4.0`, `poetry-core==2.4.1`, optional Databricks extras
`pyspark==4.1.2` and `databricks-sdk==0.118.0`, and test extra
`pytest==9.1.0`. vLLM and SGLang are intentionally not Poetry extras because
current engine releases pin incompatible Torch/Transformers stacks; install the
target engine in its own serving environment. The Databricks vLLM smoke helper
creates one such isolated local-NVMe environment and pins `vllm==0.23.0`.
`document_kv_cache.serving_env` records the exact helper profiles for vLLM and
SGLang so future smoke/probe jobs share the same install boundary.

```bash
poetry install -E test
poetry run pytest -q
```

For pip-only local workflows from the repository root:

```bash
python -m pip install -e '.[test]'
pytest tests -q
```

## License

Cachet is distributed under the Apache License 2.0. The repository includes
the full license text in `LICENSE`, and the Poetry package metadata includes
the `Apache-2.0` SPDX expression plus the license file in built wheels and
source distributions. Release-bundle validation also requires the built wheel
to carry `py.typed` markers for both `document_kv_cache` and the legacy
`restaurant_kv_serving` compatibility package so downstream users keep inline
type annotations after installation.

## Remaining V1 Work

- Run and publish the complete release bundle from target AWS g5/UC runs, including the V1 benchmark, storage-reader benchmark, native engine probes, connector-action sidecars, release evidence, preflight sidecar, GitHub governance sidecar, repository hygiene sidecar, native-probe factory diagnostics, plan execution record, Databricks run-status sidecars, tested package wheel, and PR-evidence sidecars.
- Run the connector action descriptors validation probe against native engine block managers in vLLM and SGLang.
- Keep serving integrations inside established engines; do not add a proprietary scheduler or custom solver.
- Remove the legacy `restaurant_kv_serving` compatibility package after downstream jobs migrate.
