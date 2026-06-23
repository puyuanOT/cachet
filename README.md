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

The package publishes through the transitional `document-kv-cache` distribution
name and exposes the branded `cachet` root and `cachet.<module>` import
facades over the canonical `document_kv_cache` implementation modules. Cachet
is the product brand and primary repository identity; the distribution name
stays explicit for package discovery and backward compatibility until the
package-index migration is complete. Installed wheels expose `cachet-*` CLI
aliases for the primary Cachet workflow commands as well as explicit
`document-kv-*` command names. The legacy `restaurant_kv_serving` package and
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
release focuses on Qwen3 4B Instruct on plain AWS g6/L4 Databricks hardware, with Biography,
HotpotQA, MusiQue, and Needle-in-a-Haystack benchmarks measuring quality and
latency against a standard no-cache prefill baseline.

The package deliberately stops at the engine handoff boundary. vLLM, SGLang, or
another established serving engine owns scheduling, decode, LoRA execution, and
native KV block management. Cachet provides the manifest, storage, materialized
payload, admission metadata, benchmark evidence, and adapter contracts that let
those engines reuse precomputed document context safely. Cachet now vendors the
thin vLLM/SGLang adapter packages in this repository while preserving their
existing `vllm_kv_injection.*` and `sglang_kv_injection.*` import paths for
Databricks probe metadata and launch-config compatibility.

The current implementation and release gaps are tracked in
`docs/v1-requirements-matrix.md`. Treat that matrix as the audit map for the V1
open-source package goal: it distinguishes repository-implemented requirements
from target AWS g6/L4/Unity Catalog evidence that has been bundled for the
current release and must be kept fresh before each publication.

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

`DocumentKVWorkflow.with_storage(...)` can also generate directly into the
configured `MemoryRangeReader` when `generate_cache(...)` receives a `memory:`
or `mem:` shard URI. Use that path for hot ephemeral shards; UC Volume or disk
URIs remain the durable storage targets.

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

For managed Databricks execution on the target AWS g6/L4 hardware, generate a tiny
storage-benchmark runner and `runs/submit` payload:

```bash
mkdir -p databricks-runs/storage-benchmark
python -m document_kv_cache.databricks_storage_benchmark_job \
  --workspace-dir /local_disk0/document-kv-storage-benchmark \
  --benchmark-output-json /Volumes/catalog/schema/volume/storage/storage-benchmark.json \
  --uc-volume-root /Volumes/catalog/schema/volume/storage \
  --runner-python-file dbfs:/benchmarks/run_storage_benchmark.py \
  --runner-script-output databricks-runs/storage-benchmark/run_storage_benchmark.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --output-json databricks-runs/storage-benchmark/databricks-storage-benchmark-submit.json
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
workflow = DocumentKVWorkflow.with_storage(
    manifest=manifest,
    cpu_cache_bytes=2 * 1024**3,
    local_cache_dir="/local_disk0/document-kv-cache",
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

If the integration already has a vLLM or SGLang connector object implementing
the small `ServingEngineConnector` protocol, use
`prepare_and_submit_to_engine(...)` to keep preparation and handoff together:

```python
ready = workflow.prepare_and_submit_to_engine(
    request,
    connector=vllm_connector,
    layout=qwen3_layout,
    metadata={"engine": "vllm"},
    adapter_ids=("selection-lora",),
    segmented=True,
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
    write_engine_adapter_handoff_bundle,
)

adapter_request = build_engine_adapter_request(
    ready,
    spec=vllm_adapter_spec(),
)
handoff_record = engine_adapter_request_to_record(adapter_request)
payload_path = Path("/local_disk0/document-kv-cache/req-123.kv")
handoff_path, written_payload_path = write_engine_adapter_handoff_bundle(
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

For managed Databricks execution on the target AWS g6/L4 hardware, generate a tiny
engine-probe runner and `runs/submit` payload:

```bash
mkdir -p databricks-runs/engine-probe-vllm
python -m document_kv_cache.databricks_engine_probe_job \
  --provider-backed-vllm-native-probe \
  --fixture-output-dir /Volumes/catalog/schema/volume/probes/vllm-fixture \
  --probe-output-json /Volumes/catalog/schema/volume/probes/vllm-engine-probe.json \
  --actions-output-json /Volumes/catalog/schema/volume/probes/vllm-fixture/qwen3-v1-fixture.actions.json \
  --vllm-runtime-preflight-output-json /Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-runtime-preflight.json \
  --vllm-runtime-preflight-layer-names-json /Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-layer-names.json \
  --native-probe-factories-output-json /Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-native-probe-factories.json \
  --runner-python-file dbfs:/benchmarks/run_engine_probe.py \
  --runner-script-output databricks-runs/engine-probe-vllm/run_engine_probe.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --release-safe \
  --output-json databricks-runs/engine-probe-vllm/databricks-engine-probe-submit.json
```

Release-safe single-backend engine-probe jobs derive backend-specific task keys
(`document_kv_engine_probe_vllm` or `document_kv_engine_probe_sglang`) when
`--task-key` is omitted, so split vLLM/SGLang Databricks status sidecars remain
strict-bundle-ready without manual task-key overrides.

`--provider-backed-vllm-native-probe` is the preferred single-backend QA probe
for the built-in vLLM path. It sets the Cachet vLLM probe factory, the
`vllm_kv_injection.probe:build_native_connector_probe` delegate, the
provider-backed connector metadata, `--expected-backend vllm`, and Cachet's
pinned vLLM serving dependency profile. The fixture option writes deterministic
Qwen3 handoff/payload/action artifacts on the target node before the native
probe runs; when a fixture is requested, the preset defaults the fixture payload
mode to the native adapter contract (`merged`) and rejects conflicting modes.
In release-safe mode, the preset also requires
`--native-probe-factories-output-json` plus a vLLM runtime preflight output path
and a layer-name JSON source from the runtime registration check; the
Databricks runner writes the native-factory diagnostics from inside the
installed engine runtime after preflight succeeds and before starting the native
probe. The preset rejects debug fallback flags and extra wheels so the
provider-backed adapter modules come only from the Cachet wheel. When runtime
packages or wheels are present, the
generated runner installs them into a local driver venv
(`/local_disk0/cachet/engine-probe-venv` by default) and re-execs the probe
inside that venv so Databricks ML image packages do not leak into the serving
runtime import path. If the task Python lacks stdlib `venv`/`ensurepip`
support, the runner falls back to the pinned `virtualenv` bootstrap package.
`--provider-backed-sglang-native-probe` is the matching single-backend QA probe
for Cachet's built-in provider-backed SGLang HiCache path. It sets the Cachet
SGLang probe factory, the
`sglang_kv_injection.probe:build_native_connector_probe` delegate, the
provider-backed connector metadata, `--expected-backend sglang`, and the pinned
SGLang serving dependency profile. In release-safe mode, the preset requires
`--native-probe-factories-output-json` and both SGLang runtime preflight sidecars:
`--sglang-runtime-preflight-output-json` and
`--sglang-runtime-preflight-launch-config-json`.

For release runs, prefer a two-backend probe target file and one release-safe
Databricks payload so vLLM and SGLang exercise the same descriptor contract on
the same AWS g6/L4 policy:

```json
{
  "probes": [
    {
      "backend": "vllm",
      "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
      "probe_factory": "document_kv_cache.native_probe_factories:vllm_native_probe_factory",
      "output_json": "/Volumes/catalog/schema/volume/probes/vllm-engine-probe.json",
      "actions_output_json": "/Volumes/catalog/schema/volume/probes/vllm-connector-actions.json",
      "payload_uri": "/Volumes/catalog/schema/volume/probes/vllm-payload.kv",
      "vllm_runtime_preflight_output_json": "/Volumes/catalog/schema/volume/probes/vllm-runtime-preflight.json",
      "vllm_runtime_preflight_layer_names_json": "/Volumes/catalog/schema/volume/probes/vllm-layer-names.json",
      "native_probe_factories_output_json": "/Volumes/catalog/schema/volume/probes/vllm-native-probe-factories.json",
      "native_probe_delegate_factory": "vllm_kv_injection.probe:build_native_connector_probe",
      "metadata": [
        "vllm_kv_injection.connector_factory=vllm_kv_injection.probe:build_document_kv_native_probe_connector"
      ],
      "pip_packages": [
        "vllm==0.23.0",
        "transformers==5.12.1",
        "huggingface-hub==1.20.1",
        "tokenizers==0.22.2",
        "numpy==2.3.5",
        "fastapi[standard]==0.136.0",
        "prometheus-fastapi-instrumentator==8.0.0"
      ]
    },
    {
      "backend": "sglang",
      "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
      "probe_factory": "document_kv_cache.native_probe_factories:sglang_native_probe_factory",
      "output_json": "/Volumes/catalog/schema/volume/probes/sglang-engine-probe.json",
      "actions_output_json": "/Volumes/catalog/schema/volume/probes/sglang-connector-actions.json",
      "payload_uri": "/Volumes/catalog/schema/volume/probes/sglang-payload.kv",
      "native_probe_delegate_factory": "sglang_kv_injection.probe:build_native_connector_probe",
      "metadata": [
        "sglang_kv_injection.connector_factory=sglang_kv_injection.probe:build_document_kv_hicache_probe_connector"
      ],
      "sglang_runtime_preflight_output_json": "/Volumes/catalog/schema/volume/probes/sglang-runtime-preflight.json",
      "sglang_runtime_preflight_launch_config_json": "/Volumes/catalog/schema/volume/probes/sglang-launch-config.json",
      "native_probe_factories_output_json": "/Volumes/catalog/schema/volume/probes/sglang-native-probe-factories.json",
      "pip_packages": ["sglang==0.5.10.post1"]
    }
  ]
}
```

When the target uses Cachet's built-in reserved factory path, add
`native_probe_delegate_factory` to that backend's target record. The Databricks
payload helper maps it to the matching cluster `spark_env_vars` entry,
`DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY` or
`DOCUMENT_KV_SGLANG_NATIVE_PROBE_FACTORY`, rather than passing the delegate path
as a runner argument. This keeps release target JSON stable while letting the
managed serving environment provide the actual backend-native block-manager
adapter.

```bash
mkdir -p databricks-runs/engine-probe-matrix
python -m document_kv_cache.databricks_engine_probe_job \
  --backend-config-json engine-probe-targets.json \
  --runner-python-file dbfs:/benchmarks/run_engine_probe.py \
  --runner-script-output databricks-runs/engine-probe-matrix/run_engine_probe.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --release-safe \
  --output-json databricks-runs/engine-probe-matrix/databricks-engine-probes-submit.json
```

In `--release-safe` matrix mode, the submit payload must contain exactly one
native vLLM task and one native SGLang task, and each target must carry
`native_probe_factories_output_json` so the runner records
`document_kv.native_probe_factories.v1` diagnostics inside that backend's
isolated runtime. Debug fallbacks such as `engine_version` overrides or
`allow_non_native_probe` are rejected before the Databricks job is written.
Add `--serial-tasks` when GPU capacity or cost constraints favor requesting
only one backend probe cluster at a time; the generated matrix payload adds
Databricks task dependencies so each backend runs after the previous task.
Use per-target `pip_packages` in the backend target file for engine runtime
packages such as vLLM and SGLang, because current releases pin incompatible
runtime stacks. The Cachet wheel supplies the built-in
`vllm_kv_injection`/`sglang_kv_injection` adapter modules, so the runner
installs those PyPI package specs before the Cachet wheel and any explicitly
requested custom extension wheels.
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
The built-in reserved vLLM/SGLang probe factories fail closed unless the target
serving environment has the backend package installed and points the matching
delegate environment variable at a native factory:
`DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY=module:callable` or
`DOCUMENT_KV_SGLANG_NATIVE_PROBE_FACTORY=module:callable`. Cachet's wheel ships
provider-backed delegate paths for vLLM and SGLang HiCache. The SGLang path
exercises Cachet's runtime-facing HiCache provider and launch preflight; live
decode-time prefix binding still needs validation in the installed SGLang
runtime before it can produce benchmark evidence.
When using the adapter-package delegates
`vllm_kv_injection.probe:build_native_connector_probe` or
`sglang_kv_injection.probe:build_native_connector_probe`, also add target
metadata for the backend-native connector factory, for example
`"metadata": ["vllm_kv_injection.connector_factory=vllm_kv_injection.probe:build_document_kv_native_probe_connector"]`
for the built-in provider-backed vLLM path, or
`"metadata": ["sglang_kv_injection.connector_factory=sglang_kv_injection.probe:build_document_kv_hicache_probe_connector"]`
for the built-in provider-backed SGLang HiCache path.
The Databricks target parser rejects those known delegates without the matching
connector-factory metadata so release-safe jobs fail before requesting GPU
capacity.
Use `document-kv-native-probe-scaffold` to generate a fail-closed delegate
module that declares the required Cachet contracts and backend guard checks:

```bash
document-kv-native-probe-scaffold \
  --backend vllm \
  --output-file cachet_vllm_native_probe.py
```

The generated methods raise `NotImplementedError` until a backend-native adapter
replaces them with real block reservation, payload import, request binding, and
release calls; scaffold output is therefore a starting point for a vLLM/SGLang
adapter package rather than release evidence.
Delegate factories must also declare the exact Document KV adapter contract
they implement. Set either a module-level
`DOCUMENT_KV_NATIVE_PROBE_CONTRACT = native_probe_adapter_contract_to_record()`
constant or a callable-level `document_kv_native_probe_contract` attribute.
Inspection reports the delegate as unsupported when the declaration is missing
or does not match the built-in handoff/probe/action contract, so backend forks
fail fast before a Databricks GPU probe starts.
`builtin_native_probe_factories_to_record()` reports that status together with
the pinned isolated serving-environment profile for each backend and the
configured delegate path, if present. The same diagnostic record also includes
the expected `adapter_contract` block and the delegate's declared
`delegate_adapter_contract`/`delegate_adapter_contract_valid` status. The
contract names the required engine handoff record,
connector-action record, probe record, `qwen3-v1` layout, merged payload mode,
and `native_probe=true` release requirement. Its vLLM runtime contract keeps
required hooks narrow while listing the wider current V1 base connector surface
as optional diagnostics, so backend forks can detect lifecycle drift without
turning optional hooks into hard release blockers. Use that diagnostic record when
preparing native adapter work so the target engine versions and dependency
constraints stay tied to the probe entry points, the native delegate entry
points stay visible, and the descriptor contract remains explicit:

```bash
document-kv-serving-env \
  --output-json serving-environment-profiles.json

document-kv-native-probe-factories \
  --output-json native-probe-factories.json
```

Call `validate_native_probe_factories_record` or
`native_probe_factories_record_issues` before bundling externally supplied
diagnostics; release bundles reuse the same validator for the
`document_kv.native_probe_factories.v1` sidecar.

`EngineAdapterRequest` records the target backend (`vllm` or `sglang`), payload
mode (`merged` or `segmented`), expected external package, required injection
steps, and namespaced metadata such as `document_kv.handle_uri` and
`engine.kv_injection_method`. `engine_adapter_request_to_record` turns that plan
into a JSON-serializable handoff artifact with `record_type`
`document_kv.engine_adapter_request.v1` and `schema_version` `2`. The record
contains the handle URI, payload source descriptor, model layout, token/byte
segment boundaries, per-segment cache-tier attribution, adapter ids, required
steps, and estimated GPU bytes. It intentionally omits raw KV payload bytes.
`write_engine_adapter_handoff_bundle` writes the already materialized payload
bytes for a validated adapter request plus the JSON handoff record that points
at them. Payload bytes go to an absolute local path, `disk:`, `file:`, `dbfs:`,
or UC Volume URI, preserving the merged byte stream that connector records
reference; the JSON writer therefore records an adapter-readable `payload_uri`
(or an external `handle_uri`) by default. Lower-level
`write_engine_adapter_payload` and `write_engine_adapter_request_json` helpers
remain available for workflows that intentionally manage the two artifacts
separately, and in-process connectors can consume `EngineAdapterRequest`
directly when they already have access to `ready.payload`.
The handoff writer accepts ordinary filesystem paths plus `disk:`, `file:`,
`dbfs:`, `/dbfs/...`, `/Volumes/...`, and `uc-volume:` paths so Databricks jobs
can place handoff records beside local-NVMe or Unity Catalog payloads.
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
block-index ranges, adapter ids, and engine metadata. The resulting
`EngineKVInjectionPlan` re-validates layout-derived byte totals, block totals,
and segment token/byte/block spans so adapters fail before native block-manager
calls if a handoff is malformed. Segment block ranges may
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
head elements contiguous according to `kv_stride_bytes`. `kv_stride_bytes`
includes any backend alignment padding for a KV head, so layout byte math uses
`num_layers * num_kv_heads * kv_stride_bytes * 2` rather than assuming every
engine stores exactly `head_size * dtype_width` bytes per head.

## Model Profiles

`KVModelProfile` centralizes model attention geometry so cache generators,
manifests, and engine adapters derive the same `KVLayout`. V1 includes
`QWEN3_4B_INSTRUCT_PROFILE`, exposed through `qwen3:4b-instruct` and the
canonical `Qwen/Qwen3-4B-Instruct-2507` Hugging Face model id, while
`Qwen/Qwen3-4B` remains a compatibility alias. A profile records the query-head
count, KV-head count, head size, layer count, context limit, default dtype, and
layout version, then derives bytes per cached token for MHA, GQA, or MQA-style
caches. `KVLayout` also records a `KVStorageLayout`: `separate_key_value` for
distinct key/value planes, `interleaved_key_value` for connector-specific K/V
interleaving, and `shared_key_value` when K and V views share a base allocation
and the adapter must preserve that relationship. Cache generators and manifest
refs persist the same `storage_layout`; serving validation rejects a request if
persisted chunks and the engine handoff layout disagree.

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
rather than adding ad hoc top-level fields. Profiles also carry the default
K/V storage layout for derived handoffs, so model bundles can choose shared,
separate, or interleaved key/value storage once instead of relying on every
caller to pass the same override. The example below uses a schematic MQA-style
future profile with separate K/V storage; real model bundles should replace the
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

## Benchmark Contract

The V1 benchmark surface targets Biography, HotpotQA, MusiQue, and NIAH with
Qwen3 4B Instruct on the default AWS g6/L4 release target and the explicit AWS
g5/A10G compatibility target. The default and release QA target remains
`aws-g6-l4` on plain AWS `g6.8xlarge` Databricks hardware; Cachet also models
the explicit non-default `aws-g5-a10g` target for `g5.8xlarge` compatibility
runs. It defines a common schema for comparing the no-cache prefill baseline
with document KV-cache reuse:

- `BenchmarkExample` captures one dataset example, query, expected answer, and selected source documents.
- `BenchmarkDatasetSpec` records the canonical V1 instruction style for Biography, HotpotQA, MusiQue, and NIAH.
- `build_prompt_parts` splits each example into `system_prompt`, `document_context`, and `user_prompt`.
- `build_prefill_prompt` concatenates all three parts for the no-cache baseline.
- `build_cache_prefix_text` returns the text that should be represented by cached KV, while `build_cache_suffix_text` returns the query suffix appended at inference time.
- `benchmark_cache_source_document` and `benchmark_cache_request` turn a V1
  example into the exact Cachet source document/request pair for that cached
  prefix, with stable request/artifact IDs for generated handoff bundles.
- `load_benchmark_jsonl` and `load_v1_jsonl_suite` load normalized JSONL
  plus common HotpotQA-style `context` pairs and MusiQue-style `paragraphs`,
  without adding a hard dependency on any one dataset host. Malformed benchmark
  rows fail with the physical JSONL line number so managed AWS g6/L4 runs can
  trace dataset issues back to source files.
- `normalize_v1_record`, `convert_v1_jsonl`, and `build_niah_record` prepare raw Biography, HotpotQA, MusiQue, and synthetic/source NIAH rows into that normalized JSONL contract.
- `benchmark_handoffs` generates per-row Cachet handoff bundles from prepared
  JSONL via a caller-supplied `KVChunkGenerator`, and enriches benchmark rows
  from a closed `(dataset, example_id)` manifest, so Databricks inputs can
  reference real adapter handoff and payload artifacts without hand-editing
  JSONL.
- `build_v1_benchmark_plan` and the `benchmark_plan` CLI emit a portable command plan that prepares all four V1 datasets, runs the OpenAI-compatible benchmark on AWS g6/L4/Qwen3, and can append storage-reader benchmarking on the same node.
- `benchmark_plan_executor` and `databricks_job` let managed job runners execute that plan on single-node AWS g6/L4 Databricks clusters; `databricks_runs` can submit/check those payloads using credentials supplied only through environment variables.
- `run_benchmark_suite` executes caller-provided baseline and KV-cache engines against the same logical prompt parts and emits `InferenceMeasurement` rows. Cache engines can post the full logical prompt for native serving connectors, or the runtime suffix for an explicit KV-aware proxy; both prompt views remain available on each benchmark request.
- `InferenceMeasurement` records prompt tokens, completion tokens, TTFT, time-to-completion, generated text, expected answer, and errors. OpenAI-compatible V1 measurements also carry `logical_prompt_tokens`, `runtime_prompt_tokens`, `prompt_text_mode`, and `kv_transfer_params_attached` metadata so release evidence proves the no-cache baseline saw the full prompt and the KV-cache arm attached native transfer params in its declared prompt mode.
- `summarize_measurements` produces per-dataset/per-arm latency and quality rows.
- `compare_to_baseline` reports cache speedups and quality deltas against `baseline_prefill`.
- `evaluate_v1_benchmark_evidence` marks whether the report is release-evaluable:
  all four V1 datasets, only the expected baseline/cache arms, comparisons with
  speedups and quality deltas, successful latency measurements, and quality
  rates must be present.

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

By default this engine posts the full logical prompt, which is the correct behavior for ordinary OpenAI-compatible vLLM/SGLang servers, native vLLM connector scheduling, and platform-managed prefix caching. Set `prompt_text_mode="runtime"` only when the server endpoint is a KV-aware adapter or proxy that binds the cached prefix out of band and expects only the runtime suffix in the `prompt` field. The engine records both logical and runtime prompt-token counts in measurement metadata; strict V1 release evidence accepts either logical native-cache measurements or runtime proxy-cache measurements when KV transfer params are attached.

Generate per-example handoff JSON and payload files from prepared benchmark rows
by supplying the runtime KV generator factory. Cachet does not ship a fake
benchmark generator; the factory must return a `KVChunkGenerator` whose payload
geometry matches the supplied model layout. For vanilla prefill generation in a
Transformers environment, Cachet provides
`document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator`.
Use a floating KV dtype, such as `--dtype bfloat16`, for model-produced payloads.
Set `CACHET_TRANSFORMERS_TRUST_REMOTE_CODE=true` only for model repositories that
require custom Transformers code. The generator expects Hugging Face
head-major cache tensors by default; custom token-major runtimes can set
`CACHET_TRANSFORMERS_CACHE_AXIS_ORDER=token_major`.
Default bundle filenames use a stable path-safe `{artifact_stem}` derived from
`(dataset, example_id)`:

```bash
CACHET_TRANSFORMERS_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 \
CACHET_TRANSFORMERS_DEVICE=cuda \
cachet-benchmark-handoff-bundles \
  --input-jsonl /data/v1-prepared/biography.jsonl \
  --output-dir /Volumes/catalog/schema/volume/cachet/handoffs \
  --output-manifest-json /data/handoffs/biography-manifest.json \
  --generator-factory document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator \
  --dtype bfloat16
```

The bundle CLI derives the default `qwen3:4b-instruct` layout from the built-in
model profile. For custom models, pass the complete manual layout flags:
`--layout-version`, `--dtype`, `--num-layers`, `--block-size`, and
`--bytes-per-token`; add `--num-query-heads`, `--num-kv-heads`, `--head-size`,
`--kv-stride-bytes`, `--shares-kv-storage`, and `--storage-layout` when the
serving adapter needs attention geometry.

If handoff JSON files already exist, build the closed manifest that the
benchmark plan consumes. The template supports `{dataset}` and `{example_id}`,
and each referenced handoff JSON is validated before the manifest is written:

```bash
cachet-benchmark-handoff-manifest \
  --input-jsonl /data/v1-prepared/biography.jsonl \
  --handoff-json-template '/Volumes/catalog/schema/volume/cachet/handoffs/{dataset}/{example_id}.handoff.json' \
  --expected-backend vllm \
  --output-json /data/handoffs/biography-manifest.json
```

To run the V1 benchmark contract against existing OpenAI-compatible vLLM or SGLang servers, generate a reproducible command plan for all four datasets:

```bash
python -m document_kv_cache.benchmark_plan \
  --raw-dataset biography=/raw/biography.jsonl \
  --raw-dataset hotpotqa=/raw/hotpotqa.jsonl \
  --raw-dataset musique=/raw/musique.jsonl \
  --raw-dataset niah=/raw/niah.jsonl \
  --prepared-dir /data/v1-prepared \
  --benchmark-handoff-manifest-json biography=/data/handoffs/biography-manifest.json \
  --benchmark-handoff-manifest-json hotpotqa=/data/handoffs/hotpotqa-manifest.json \
  --benchmark-handoff-manifest-json musique=/data/handoffs/musique-manifest.json \
  --benchmark-handoff-manifest-json niah=/data/handoffs/niah-manifest.json \
  --benchmark-handoff-generator-factory document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator \
  --benchmark-handoff-dtype bfloat16 \
  --benchmark-handoff-output-dir /data/handoffs/generated \
  --base-url http://localhost:8000 \
  --storage-benchmark-workspace-dir /local_disk0/document-kv-storage-benchmark \
  --storage-benchmark-uc-volume-root /Volumes/catalog/schema/volume/document-kv-storage-benchmark \
  --engine-probe-handoff-json vllm=/data/vllm-handoff.json \
  --engine-probe-output-json vllm=/data/vllm-engine-probe.json \
  --engine-probe-actions-output-json vllm=/data/vllm-connector-actions.json \
  --engine-probe-native-probe-factories-output-json vllm=/data/vllm-native-probe-factories.json \
  --engine-probe-handoff-json sglang=/data/sglang-handoff.json \
  --engine-probe-output-json sglang=/data/sglang-engine-probe.json \
  --engine-probe-actions-output-json sglang=/data/sglang-connector-actions.json \
  --engine-probe-native-probe-factories-output-json sglang=/data/sglang-native-probe-factories.json \
  --engine-probe-use-builtin-factories \
  --release-evidence-output-json /data/release-evidence.json \
  --release-preflight-output-json /data/release-inputs.json \
  --release-bundle-output-dir /data/document-kv-release-bundle \
  --release-bundle-output-json /data/release-bundle-manifest.json \
  --release-bundle-databricks-run-status-json /data/databricks-run-status-benchmark.json \
  --release-bundle-databricks-run-status-json /data/databricks-run-status-storage.json \
  --release-bundle-databricks-run-status-json /data/databricks-run-status-vllm-engine-probe.json \
  --release-bundle-databricks-run-status-json /data/databricks-run-status-sglang-engine-probe.json \
  --release-bundle-package-wheel /data/dist/document_kv_cache-0.2.0-py3-none-any.whl \
  --release-bundle-compatibility-benchmark-json /data/g5-v1-results.json \
  --release-bundle-pr-evidence-json /data/pr-evidence/release-provenance.json \
  --github-governance-output-json /data/github-governance.json \
  --repository-hygiene-output-json /data/repository-hygiene.json \
  --native-probe-factories-output-json /data/native-probe-factories.json \
  --engine-launch-config-output-dir /data/engine-launch-configs \
  --engine-probe-targets-output-json /data/engine-probe-targets.json \
  --engine-probe-targets-release-safe \
  --plan-output-json /data/v1-plan.json \
  --plan-output-sh /data/run-v1-benchmark.sh
```

The generated shell script runs `dataset_prep` for each raw file, optionally
generates per-dataset handoff bundles when
`--benchmark-handoff-generator-factory` is supplied, enriches any dataset
configured with handoffs by invoking `document_kv_cache.benchmark_handoffs`,
then invokes `benchmark_runner` against the enriched JSONL. Unless
`--benchmark-handoff-output-jsonl DATASET=PATH` is provided, enriched inputs are
written as `<prepared-dir>/<dataset>.handoffs.jsonl`. When manifests are not
provided explicitly, generated handoff manifests default under
`--benchmark-handoff-output-dir`.
When `--storage-benchmark-workspace-dir` is provided, the plan also appends a
`document_kv_cache.storage_benchmark` command. The storage command captures
Memory and Disk reader latency/throughput on the same AWS g6/L4 node, and adds
the Unity Catalog reader only when `--storage-benchmark-uc-volume-root` points
at a real UC Volume. It writes `<suite-id>-storage-benchmark.json` under
`--prepared-dir` unless `--storage-benchmark-output-json` is set.
Backend-keyed
`--engine-probe-*` options append native `document_kv_cache.engine_probe`
commands for vLLM and SGLang handoffs; release evidence automatically consumes
those planned probe and connector-action outputs. Planned probes consumed by
release evidence must include `--engine-probe-actions-output-json` for each
backend and cannot use debug-only `--engine-probe-engine-version` or
`--allow-non-native-engine-probe`; if debug probe commands are also needed, pass
separate native records directly with repeatable `--release-engine-probe-json`
and `--release-engine-actions-json`.
Use `--engine-probe-fixture-output-dir BACKEND=DIR` when the plan should first
write a deterministic Qwen3 V1 probe fixture with
`document_kv_cache.probe_fixtures`; if `--engine-probe-handoff-json` is omitted
for that backend, the plan derives it as
`DIR/qwen3-v1-fixture.handoff.json`. Add
`--engine-probe-fixture-payload-mode BACKEND=merged|segmented` to select the
fixture payload format. The fixture directory includes the `.kvpack`, payload,
handoff JSON, and `qwen3-v1-fixture.actions.json` connector-action sidecar so
native adapter work can inspect the exact reserve/copy/bind/release descriptors
before running a backend block-manager probe.
When `--engine-probe-use-builtin-factories` selects Cachet's built-in native
probe factories, fixture-backed planned probes default to the native adapter
contract (`merged`).
When `--engine-probe-actions-output-json` is omitted for a fixture-backed probe,
the plan uses that derived `qwen3-v1-fixture.actions.json` path as the release
connector-action sidecar and avoids rewriting it from the probe command.
The benchmark plan executor treats generated plan JSON as a closed execution
schema: unsupported top-level plan keys or per-command keys are rejected before
any command is run.
`--engine-probe-use-builtin-factories` fills missing planned factories with
package-owned vLLM/SGLang factory paths. Those factories are stable release-plan
targets: the vLLM path can use Cachet's provider-backed delegate with strict
connector-factory metadata and runtime preflight, and the SGLang path can use
Cachet's provider-backed HiCache delegate with strict connector-factory metadata
plus a passing SGLang runtime preflight for its dynamic HiCache launch config
and provider factory. Pass explicit
`--engine-probe-factory BACKEND=MODULE:CALLABLE` to use a downstream adapter
module.
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
For Databricks engine-probe target JSON, use
`--engine-probe-native-probe-factories-output-json BACKEND=PATH` to make the
managed probe runner emit the same diagnostics sidecar from inside each
backend-specific isolated serving runtime.
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
When a planned probe uses `--engine-probe-fixture-output-dir`, the target JSON
also carries `fixture_output_dir` and `fixture_payload_mode`; the generated
Databricks runner writes the deterministic Qwen3 fixture first, including the
validated connector-action sidecar, then runs the native engine probe against
the derived handoff JSON.
Use `--engine-probe-native-delegate-factory BACKEND=MODULE:CALLABLE` when the
planned target should run a built-in reserved vLLM/SGLang probe factory through
a backend-native delegate on Databricks. The target JSON carries the delegate
path, and the Databricks payload helper injects the corresponding
`DOCUMENT_KV_*_NATIVE_PROBE_FACTORY` environment variable into that backend's
single-node task cluster.
Use `--engine-probe-targets-release-safe` for release jobs; it requires exactly
one native vLLM probe and one native SGLang probe and rejects debug-only planned
probe settings before writing the target file. Release-safe targets also carry
each backend's `actions_output_json` path so the Databricks probe jobs write the
sidecars required by release evidence.
Release-safe targets must also carry each backend's
`native_probe_factories_output_json`; generate it with
`--engine-probe-native-probe-factories-output-json BACKEND=PATH` so split vLLM
and SGLang probe tasks produce native-factory diagnostics from the runtime that
actually imports that backend.
Provider-backed vLLM Databricks targets must also carry
`vllm_runtime_preflight_output_json` and
`vllm_runtime_preflight_layer_names_json` so the runner validates the installed
vLLM contract and layer mapping before the native probe starts. Use
`--engine-probe-vllm-runtime-preflight-output-json` and
`--engine-probe-vllm-runtime-preflight-layer-names-json` when generating target
JSON from `document_kv_cache.benchmark_plan`.
Release-safe SGLang Databricks targets must carry
`sglang_runtime_preflight_output_json` and
`sglang_runtime_preflight_launch_config_json` so the runner validates the
installed dynamic HiCache surface and provider-backed launch config before the
native probe starts. Use
`--engine-probe-sglang-runtime-preflight-output-json` and
`--engine-probe-sglang-runtime-preflight-launch-config-json` when generating
target JSON.
Preflight sidecar paths may use Databricks `dbfs:/...` URIs; the managed runner
normalizes those file arguments to the driver-local `/dbfs/...` FUSE path before
calling the runtime preflight CLIs, while preserving inline JSON layer-name or
launch-config values.
When release-evidence flags are supplied, the plan appends
`document_kv_cache.release_evidence` last so the V1 benchmark, storage
benchmark, and vLLM/SGLang probe artifacts are validated together. Release
evidence also checks that each native probe reports the pinned runtime engine
version and carries the serving-engine package/version metadata from
`document_kv_cache.serving_env`. Add
`--release-bundle-output-dir` to append `document_kv_cache.release_bundle`
after that validation step; the bundle command copies the validated V1,
storage, engine-probe, release-evidence artifacts plus any generated or supplied preflight,
Databricks run-status, wheel, PR-evidence, V1 requirements matrix, GitHub
governance, and native-probe factory diagnostics sidecars into a checksummed
handoff directory. The benchmark plan executor writes `plan-execution.json` only
after the planned commands finish, so the final strict release bundle should be
built as a separate post-execution command with
`--plan-execution-json /data/plan-execution.json --requirements-matrix-md docs/v1-requirements-matrix.md --require-complete-v1`,
or by supplying `--release-bundle-plan-execution-json` together with
`--release-bundle-requirements-matrix-md` and
`--release-bundle-require-complete-v1` when planning a follow-up bundle command.
Strict follow-up bundle plans are checked before commands are emitted: the
planner requires the preflight, plan-execution, Databricks run-status, tested
wheel, PR evidence, V1 requirements matrix, GitHub governance, repository
hygiene, native probe factory diagnostics, and engine launch config sidecars to
be present. Use `--engine-launch-config-output-dir` to make the plan generate
the vLLM/SGLang launch-config sidecars before bundle assembly; explicit
`--release-bundle-engine-launch-config-json` paths remain supported, but if
both are provided they must point at the generated files to avoid duplicate
backend evidence.
For ad hoc standalone use, generate those sidecars with the same package helper
so the JSON shape matches the validator used by strict release bundles:

```bash
document-kv-engine-launch-config build-vllm \
  --output-json /data/engine-launch/vllm.json

document-kv-engine-launch-config build-sglang \
  --output-json /data/engine-launch/sglang.json
```

The builder writes only validated sidecars. Optional `--extra-config KEY=VALUE`
entries are preserved for adapter-specific metadata, but reserved
`document_kv.*` handoff fields are generated by Cachet and cannot be
overridden. The vLLM sidecar includes
`document_kv.provider_factory=vllm_kv_injection.vllm_native_provider:build_document_kv_provider`
by default, so `DocumentKVConnector` resolves the provider-backed runtime path
instead of falling back to its safe no-op provider.
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
  --hardware-target aws-g6-l4 \
  --output-json v1-results.json
```

For a self-contained Databricks AWS g6/L4 smoke of the actual vLLM server path, use
`document_kv_cache.vllm_smoke` from a GPU task. It creates an isolated vLLM
environment on local NVMe, installs the pinned serving dependency stack,
installs Cachet into that same vLLM environment, starts
`Qwen/Qwen3-4B-Instruct-2507` as `qwen3:4b-instruct` with Cachet's
`DocumentKVConnector` `KVTransferConfig`, runs one tiny
Biography/HotpotQA/MusiQue/NIAH example through the OpenAI-compatible benchmark
runner, and writes `metadata.json`, `vllm-import-probe.json`,
`vllm-server.log`, and `v1-benchmark.json` to the chosen output directory.
The import probe instantiates `DocumentKVConnector` from the same
`KVTransferConfig` passed to vLLM and fails unless the configured provider
factory resolves to native document-KV wiring.

```bash
python -m document_kv_cache.vllm_smoke \
  --benchmark-id v1_vllm_smoke_001 \
  --output-dir /Volumes/catalog/schema/volume/document-kv-v1-smoke
```

For release-grade or long-context evidence, prepare one canonical JSONL file per
V1 dataset. You can either enrich those files ahead of time with the benchmark
handoff tooling or let this runner generate Transformers-backed Cachet handoff
bundles before vLLM starts. The same helper still owns the isolated vLLM
environment and Databricks-local model cache, but it no longer writes the tiny
built-in smoke files. Prepared mode writes `prepared-handoff-generation.json`
when generation is enabled, always writes `prepared-handoff-coverage.json`, fails
before model startup when any row lacks handoff params or points at an
unreadable/non-vLLM Cachet handoff, and runs the cache arm with the full logical
prompt plus `kv_transfer_params` against the same native vLLM server. vLLM's V1
connector scheduler must see the logical prefix token positions before it can
allocate external KV blocks, so suffix-only cache prompts are intentionally not
used for native vLLM evidence:

```bash
python -m document_kv_cache.vllm_smoke \
  --benchmark-id v1_vllm_prepared_001 \
  --output-dir /Volumes/catalog/schema/volume/document-kv-v1-prepared \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.9 \
  --max-tokens 100 \
  --benchmark-handoff-generator-factory document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator \
  --benchmark-handoff-output-dir /Volumes/catalog/schema/volume/document-kv-v1-prepared/handoffs \
  --benchmark-handoff-dtype bfloat16 \
  --dataset biography=/Volumes/catalog/schema/volume/v1/biography.jsonl \
  --dataset hotpotqa=/Volumes/catalog/schema/volume/v1/hotpotqa.jsonl \
  --dataset musique=/Volumes/catalog/schema/volume/v1/musique.jsonl \
  --dataset niah=/Volumes/catalog/schema/volume/v1/niah.jsonl
```

Prepared dataset mode requires exactly Biography, HotpotQA, MusiQue, and NIAH.
The metadata records `dataset_source`, `dataset_specs`, `max_model_len`,
`max_num_seqs`, `gpu_memory_utilization`, the Cachet package install spec, and
the vLLM `KVTransferConfig` so release evidence can distinguish tiny smoke
checks from full or long-context benchmark runs.

To launch that same smoke through a Databricks managed task, upload the wheel
and generated runner script, then emit a single-node AWS g6/L4 `runs/submit`
payload. The runner installs the uploaded wheel into the Databricks driver
process and forwards the same wheel path to the isolated vLLM environment:

```bash
mkdir -p databricks-runs/vllm-smoke
python -m document_kv_cache.databricks_vllm_smoke_job \
  --benchmark-id v1_vllm_smoke_001 \
  --output-dir /Volumes/catalog/schema/volume/document-kv-v1-smoke \
  --runner-python-file dbfs:/benchmarks/run_vllm_smoke.py \
  --runner-script-output databricks-runs/vllm-smoke/run_vllm_smoke.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --output-json databricks-runs/vllm-smoke/databricks-vllm-smoke-submit.json
```

The Databricks submit helper accepts the same prepared dataset and sizing flags,
so the managed task can run either the built-in smoke or a prepared long-context
benchmark on the target AWS g6/L4 profile. Use `--hardware-target` to derive
the default Databricks node family (`aws-g6-l4` -> `g6.8xlarge`,
`aws-g5-a10g` -> `g5.8xlarge`); pass `--node-type-id` only when a workspace
cluster policy requires a specific matching size:

```bash
mkdir -p databricks-runs/vllm-prepared
python -m document_kv_cache.databricks_vllm_smoke_job \
  --benchmark-id v1_vllm_prepared_001 \
  --output-dir /Volumes/catalog/schema/volume/document-kv-v1-prepared \
  --runner-python-file dbfs:/benchmarks/run_vllm_smoke.py \
  --runner-script-output databricks-runs/vllm-prepared/run_vllm_smoke.py \
  --wheel-uri dbfs:/benchmarks/cachet-v1/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --hardware-target aws-g6-l4 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.9 \
  --max-tokens 100 \
  --benchmark-handoff-generator-factory document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator \
  --benchmark-handoff-output-dir /Volumes/catalog/schema/volume/document-kv-v1-prepared/handoffs \
  --benchmark-handoff-dtype bfloat16 \
  --spark-env-var CACHET_TRANSFORMERS_DEVICE=cuda \
  --spark-env-var CACHET_TRANSFORMERS_TORCH_DTYPE=bfloat16 \
  --dataset biography=dbfs:/benchmarks/v1/biography.jsonl \
  --dataset hotpotqa=dbfs:/benchmarks/v1/hotpotqa.jsonl \
  --dataset musique=dbfs:/benchmarks/v1/musique.jsonl \
  --dataset niah=dbfs:/benchmarks/v1/niah.jsonl \
  --output-json databricks-runs/vllm-prepared/databricks-vllm-prepared-submit.json
```

This emits suite metadata, per-request measurements, per-dataset quality/latency
summaries, cache-vs-baseline comparisons, and a `v1_evidence` block. Smoke runs
against one dataset are allowed, but `v1_evidence.ok` is false until all required
V1 datasets, expected arms, and metrics are present. With the default `/v1/completions`
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

The command returns exit code `0` only when the V1 benchmark targets supported AWS g6/L4 hardware with Qwen3,
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
  --compatibility-benchmark-json g5-v1-results.json \
  --storage-benchmark-json storage-benchmark.json \
  --engine-probe-json vllm-probe.json \
  --engine-probe-json sglang-probe.json \
  --engine-actions-json vllm-connector-actions.json \
  --engine-actions-json sglang-connector-actions.json \
  --engine-launch-config-json vllm-launch-config.json \
  --engine-launch-config-json sglang-launch-config.json \
  --release-evidence-json release-evidence.json \
  --preflight-json release-inputs.json \
  --plan-execution-json plan-execution.json \
  --databricks-run-status-json databricks-run-status-benchmark.json \
  --databricks-run-status-json databricks-run-status-storage.json \
  --databricks-run-status-json databricks-run-status-vllm-engine-probe.json \
  --databricks-run-status-json databricks-run-status-sglang-engine-probe.json \
  --package-wheel dist/document_kv_cache-0.2.0-py3-none-any.whl \
  --pr-evidence-json pr-evidence/release-provenance.json \
  --requirements-matrix-md docs/v1-requirements-matrix.md \
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
tested on the target AWS g6/L4 runtime, and repeat `--pr-evidence-json` to carry PR
traceability records alongside the benchmark, storage, engine-probe,
connector-action, engine-launch-config, release-evidence, and preflight
artifacts. Add
`--compatibility-benchmark-json` to carry non-default supported V1 benchmark
evidence such as AWS g5/A10G compatibility runs; the bundle validates each
compatibility benchmark against the same storage/native probe/action sidecars
and rejects the strict AWS g6/L4 release target in that compatibility role. Add
`--require-complete-v1` for release publishing; this strict mode refuses to
build a V1 bundle unless the release evidence, preflight, vLLM/SGLang native
engine-probe, connector-action, and engine-launch-config sidecars,
benchmark-plan execution, Databricks run-status sidecars for benchmark,
storage, and engine-probe runs, tested wheel, PR evidence, V1 requirements
matrix, GitHub governance, repository hygiene, and native-probe factory
diagnostics sidecars are all present; the release bundle also verifies exactly
one Databricks sidecar for the benchmark and storage purposes plus one vLLM and
one SGLang engine-probe status, either split across backend runs or grouped in
an engine-probe matrix run. The native factory diagnostics must also report
supported built-in vLLM and SGLang factory entry points across the bundled
diagnostics sidecars.
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
strict V1 release bundles require both entries to be supported across the
bundled diagnostics.
Repeat
`--databricks-run-status-json` for compact `databricks_runs get --summary`
outputs, or extracted inner `summary` records, from the managed runs whose
outputs are included in the bundle; release bundles require those status
sidecars to be terminal, successful, free of active task keys, and attached to a
hashed single-node AWS g6/L4 `SINGLE_USER` submit-payload summary whose task
summaries carry non-empty `purpose` tags and whose summary arrays match the task
summaries. Do not use the `--include-response` debug flag for release-bundle
status sidecars; bundles reject raw Jobs API responses.
PR evidence sidecars must be valid `document_kv.pr_evidence.v1` records with
the GitHub pull-request number and canonical owner/repo pull-request URL,
Refactor-skill evidence, completed GPT-5.5 review, and any GPT-5.5 findings
marked resolved before they can enter the release bundle. When a GitHub
governance sidecar is bundled, each PR URL must also point at the governed
repository. They are also validated as closed schemas so ad hoc review notes or
debug payloads stay out of the release handoff.
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
The command reads the GitHub token from `GITHUB_TOKEN` or, for local developer
runs, falls back to `gh auth token` unless `--no-gh-auth-token-fallback` is set.
It records repository visibility, branded repository metadata, `main` branch
protection state, and open pull-request pressure. It returns non-zero until the
repository is public, the repository name is `cachet`, the description is
non-empty and mentions Cachet, any homepage URL also mentions Cachet, the
repository topics include `cachet` and `kv-cache`, the required
`Test and build` protection is active, branch protection applies to
administrators, repository merge settings allow squash or rebase merges, GitHub
auto-merge is enabled, merged PR branches are deleted automatically, and no
unexpected pull requests remain open:

```bash
python -m document_kv_cache.github_governance \
  --repository OWNER/cachet \
  --output-json github-governance.json
```

Use an explicit `GITHUB_TOKEN` in CI and Databricks jobs. For hermetic checks
that should fail instead of reading the local GitHub CLI keyring, add
`--no-gh-auth-token-fallback`.

When producing a sidecar from inside the current release PR, pass
`--allow-open-pull-request-number <PR_NUMBER>` so that one active pull request is
recorded with its sanitized PR summary but not treated as stale release
pressure.
Regenerate GitHub governance sidecars before strict release assembly whenever
the package version changes its governance schema; strict release bundles reject
older sidecars that do not include the merge-settings block.

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
generated or secret-like paths that Git still exposes as non-ignored files,
including notebook checkpoint folders produced by exploratory Databricks or
Jupyter work. Databricks `runs/submit` sidecars, task logs, and status JSONs
belong under the ignored `databricks-runs/` directory or in explicit release
bundles; repository hygiene rejects that directory if it becomes tracked or
visible as untracked source. It also records dirty tracked paths and the
directories that require `README.md` or package docstring documentation, so
release handoffs prove they were produced from a clean, documented worktree. It
returns non-zero until all required ignore patterns are present, no forbidden
artifact paths are tracked or exposed as untracked, every non-generated
tracked/untracked directory is documented, and no tracked files differ from
`HEAD`.

For Databricks-managed execution, upload the package wheel, the generated benchmark plan JSON, and a small runner script, then generate a single-node AWS g6/L4 `runs/submit` payload. New integrations should prefer the generic `DatabricksSingleNodeGPUClusterConfig`, `build_single_node_gpu_cluster`, and `validate_aws_single_node_gpu_type` helper names; the older `g5` names remain compatibility aliases for existing callers. The Databricks payload CLIs accept `--hardware-target aws-g6-l4` or `--hardware-target aws-g5-a10g` and derive the default node type from the shared V1 hardware profile unless `--node-type-id` is explicitly supplied.
When generating `v1-plan.json` for the provider-backed native probe paths,
include
`--engine-probe-metadata vllm=vllm_kv_injection.connector_factory=vllm_kv_injection.probe:build_document_kv_native_probe_connector`
and
`--engine-probe-metadata sglang=sglang_kv_injection.connector_factory=sglang_kv_injection.probe:build_document_kv_hicache_probe_connector`
so the plan carries the strict connector-factory metadata before a Databricks
GPU job is submitted.

```bash
mkdir -p databricks-runs/managed-plan
python -m document_kv_cache.databricks_job \
  --plan-json-uri dbfs:/benchmarks/v1-plan.json \
  --runner-python-file dbfs:/benchmarks/run_plan.py \
  --runner-script-output databricks-runs/managed-plan/run_plan.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --hardware-target aws-g6-l4 \
  --spark-env-var CACHET_TRANSFORMERS_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 \
  --spark-env-var CACHET_TRANSFORMERS_DEVICE=cuda \
  --spark-env-var CACHET_TRANSFORMERS_TORCH_DTYPE=bfloat16 \
  --vllm-native-probe-delegate-factory vllm_kv_injection.probe:build_native_connector_probe \
  --sglang-native-probe-delegate-factory my_sglang_adapter.probes:build_probe \
  --output-json databricks-runs/managed-plan/databricks-run-submit.json
```

Use `--spark-env-var` only for non-secret runtime configuration, such as the
Transformers generator model id, device, dtype, or cache-axis order. The helper
rejects secret-looking env names and Databricks PAT-shaped values so generated
payloads stay suitable for release evidence.

Set the native-probe delegate factory flags only when the benchmark plan uses
Cachet's built-in reserved vLLM or SGLang native probe factories. The Databricks
job helper writes those paths as cluster `spark_env_vars`
(`DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY` and
`DOCUMENT_KV_SGLANG_NATIVE_PROBE_FACTORY`) instead of passing them to the
benchmark runner, so plan execution arguments remain stable.

Dry-run a staged payload locally before any Databricks credentials or network
requests are involved:

```bash
cachet-databricks-runs \
  --output-json databricks-runs/managed-plan/databricks-run-submit-summary.json \
  payload-summary \
  --payload-json databricks-runs/managed-plan/databricks-run-submit.json \
  --expected-hardware-target aws-g6-l4 \
  --expected-node-type-id g6.8xlarge

cachet-databricks-runs \
  --output-json databricks-runs/managed-plan/databricks-stage-submit-plan.json \
  stage-and-submit \
  --payload-json databricks-runs/managed-plan/databricks-run-submit.json \
  --artifact v1-plan.json=dbfs:/benchmarks/v1-plan.json \
  --artifact databricks-runs/managed-plan/run_plan.py=dbfs:/benchmarks/run_plan.py \
  --require-payload-dbfs-artifacts \
  --dry-run
```

For payloads that intentionally write generated DBFS outputs during the run,
such as fixture-backed native engine probes, use
`--require-payload-staged-dbfs-artifacts` instead. It still checks DBFS runner,
wheel, plan, and SGLang launch-config inputs, but does not require upload
artifacts for output JSONs or generated fixture paths.

Submit or inspect a generated payload with env-provided credentials, keeping
tokens out of command arguments and JSON artifacts. If the workspace is already
configured in `~/.databrickscfg`, pass `--profile PROFILE_NAME` instead of
exporting `DATABRICKS_HOST` and `DATABRICKS_TOKEN`; use `--config-file PATH`
when the profile lives outside the default config file. Profiles with `host`
and `token` are parsed directly by default. Profiles with `auth_type` and no
static token are resolved through the optional Databricks SDK. For OAuth or
Databricks CLI profiles that also contain a transient static token, pass
`--profile-auth-mode sdk` to force SDK profile auth after refreshing the profile
with the Databricks CLI. Use `--profile-auth-mode static` when a profile must
fail unless its static token is present. Install the `databricks` extra before
using SDK-backed OAuth or Databricks CLI auth profiles:

```bash
export DATABRICKS_HOST=https://dbc-...cloud.databricks.com
export DATABRICKS_TOKEN=...

cachet-databricks-runs \
  --output-json databricks-runs/managed-plan/databricks-stage-submit-response.json \
  stage-and-submit \
  --payload-json databricks-runs/managed-plan/databricks-run-submit.json \
  --artifact v1-plan.json=dbfs:/benchmarks/v1-plan.json \
  --artifact databricks-runs/managed-plan/run_plan.py=dbfs:/benchmarks/run_plan.py \
  --overwrite \
  --require-payload-dbfs-artifacts

cachet-databricks-runs \
  --output-json databricks-runs/managed-plan/databricks-run-status.json \
  get \
  --run-id 123456789 \
  --summary \
  --submit-payload-json databricks-runs/managed-plan/databricks-run-submit.json \
  --expected-hardware-target aws-g6-l4 \
  --expected-node-type-id g6.8xlarge
```

Equivalent profile-based submit:

```bash
cachet-databricks-runs \
  --profile QA \
  --output-json databricks-runs/managed-plan/databricks-auth-check.json \
  auth-check

cachet-databricks-runs \
  --profile QA_OAUTH \
  --profile-auth-mode sdk \
  --output-json databricks-runs/managed-plan/databricks-auth-check.json \
  auth-check

cachet-databricks-runs \
  --profile QA \
  --output-json databricks-runs/managed-plan/databricks-stage-submit-response.json \
  stage-and-submit \
  --payload-json databricks-runs/managed-plan/databricks-run-submit.json \
  --artifact v1-plan.json=dbfs:/benchmarks/v1-plan.json \
  --artifact databricks-runs/managed-plan/run_plan.py=dbfs:/benchmarks/run_plan.py \
  --overwrite \
  --require-payload-dbfs-artifacts \
  --preflight-auth-check
```

`auth-check` performs a non-mutating current-user request before staging or
submitting. Its output records only the endpoint, HTTP status, response key
names, and a SHA-256 hash of the workspace host, so it can be attached to launch
handoffs without writing tokens or user details. The `stage-and-submit`
`--preflight-auth-check` flag runs the same check after local artifact
validation and before any DBFS upload or run submission.

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
Use `validate_databricks_run_status_sidecar` or
`databricks_run_status_sidecar_issues` to preflight release-oriented run-status
sidecars before assembling a release bundle. Pass
`expected_hardware_target="aws-g6-l4"` and
`expected_node_type_id="g6.8xlarge"` for strict V1 release checks so stale
non-g6 status sidecars, or sidecars from non-default g6 sizes, cannot be reused
as release evidence.

Workspace-specific automation can still POST the payload itself after applying
the organization’s auth, cluster policy, and asset-upload conventions. Teams
that manage jobs declaratively can use the reference Databricks Asset Bundle in
`databricks/databricks.yml`; it mirrors the same single-node AWS g6/L4 benchmark
contract without embedding workspace credentials or paths. The bundle exposes
non-secret `transformers_*` variables for generator runtime settings and maps
them to `CACHET_TRANSFORMERS_*` cluster `spark_env_vars`.
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
  --runner-script-output /data/run_engine_probe.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --release-safe \
  --output-json /data/databricks-engine-probes-submit.json
```

Workspace-specific automation can wrap this handoff by building or reusing the
wheel, writing the plan and runner, uploading those small artifacts to the
workspace storage layer, selecting a real UC Volume path for the storage-reader
benchmark, and submitting the single-node AWS g6/L4 job. Raw
benchmark datasets should already be visible to the cluster as `disk:...`,
`file:...`, `dbfs:/...`, `/dbfs/...`, `/Volumes/...`, `uc-volume:/...`, or `uc-volume://...` paths; the
command plan normalizes those storage URIs on the Databricks driver before
reading or writing JSONL artifacts.
Backend target JSON may include per-target `pip_packages`; use that for
backend-specific runtime packages instead of installing every engine into every
task.
Release-safe backend target JSON must include per-target
`native_probe_factories_output_json`; the runner writes that sidecar after any
runtime preflight succeeds and before the native engine probe starts, and stops
without running the probe if diagnostics fail.
Provider-backed vLLM release targets must include
`vllm_runtime_preflight_output_json` and
`vllm_runtime_preflight_layer_names_json`; the Databricks runner executes the
strict vLLM runtime preflight before the native engine probe and returns its
nonzero exit code without starting the probe if validation fails.
Release-safe SGLang targets must include
`sglang_runtime_preflight_output_json` and
`sglang_runtime_preflight_launch_config_json`; the Databricks runner executes
`cachet-sglang-runtime-preflight` against that provider-backed HiCache launch
config before the native engine probe and returns its nonzero exit code without
starting the probe if validation fails.
For Databricks-managed probes, `dbfs:/...` preflight sidecar paths are
normalized to `/dbfs/...` on the driver before the preflight CLI reads or writes
them; inline JSON layer-name or launch-config arguments are passed through
unchanged.

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
`pytest==9.1.1`. Raw vLLM and SGLang runtimes are intentionally not Poetry
extras because their current dependency graphs conflict in one resolver; use
the vendored adapter code from the Cachet wheel and install exactly one serving
runtime in an isolated environment. The Databricks vLLM smoke helper creates
one such isolated local-NVMe environment and pins `vllm==0.23.0`.
`document_kv_cache.serving_env` records the exact helper profiles for vLLM and
SGLang so future smoke/probe jobs share the same install boundary.
The committed `poetry.lock` records the resolver output for the base package,
Databricks extras, and test extras; CI validates it with `poetry check --lock`.

```bash
poetry check --lock
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
to carry `py.typed` markers for `cachet`, `document_kv_cache`, and the legacy
`restaurant_kv_serving` compatibility package so downstream users keep inline
type annotations after installation.

## Remaining V1 Work

- Keep the complete strict release bundle refreshed from the target AWS
  g6/L4/UC evidence set. The benchmark, storage-reader, and vLLM/SGLang native
  engine-probe Databricks runs have succeeded, and release evidence over the
  benchmark, storage, connector action descriptors, and native engine block
  managers is green. The native engine block managers remain owned by vLLM and
  SGLang rather than Cachet. The current g5-enriched strict bundle validates with 22
  artifacts: release evidence sidecar, preflight sidecar, vLLM/SGLang native
  engine probe sidecars, vLLM/SGLang connector action sidecars, vLLM/SGLang
  engine launch config sidecars, benchmark plan execution sidecar, Databricks
  run-status sidecars for benchmark, storage, and vLLM/SGLang engine-probe
  runs, tested package wheel, PR evidence sidecar, V1 requirements matrix,
  GitHub governance sidecar, repository hygiene sidecar, native probe factory
  diagnostics sidecar entries from both runtime environments, and the current
  `aws-g5-a10g` benchmark carried through the `compatibility_benchmark` role.
- Keep the current AWS g5/A10G compatibility benchmark evidence with the
  release handoff: QA Databricks run `315109189523858` on `g5.8xlarge`
  completed Biography, HotpotQA, MusiQue, and NIAH with no benchmark errors,
  current `v1_evidence.ok=true`, TTFT speedups of 4.55x-6.07x, and release
  evidence `ok=true` when validated with the current storage and native
  vLLM/SGLang probe/action artifacts. Carry it in release handoffs with
  `--compatibility-benchmark-json`; this compatibility evidence does not change
  the strict V1 publication target from AWS g6/L4.
- Keep GitHub governance release-ready before each public release. Current
  governance evidence is green: the repository is public, `allow_auto_merge`
  reports enabled, merged head branches are deleted automatically, `main`
  branch protection requires the `Test and build` check, branch protection
  applies to administrators, force-pushes/deletions are blocked, and no
  unexpected pull requests are open.
- Continue running connector action descriptors validation against native
  engine block managers in vLLM and SGLang whenever adapter contracts, runtime
  pins, or launch configs change.
- Keep serving integrations inside established engines; do not add a proprietary scheduler or custom solver.
- Remove the legacy `restaurant_kv_serving` compatibility package after downstream jobs migrate.
