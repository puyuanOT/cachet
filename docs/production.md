# Production

This page is for integrators moving from the local quickstart to real serving.
It intentionally avoids release-bundle and PR-evidence details; release
operators should use [`release-ops/README.md`](release-ops/README.md).

## Production Checklist

1. Choose the model and layout.
2. Generate real KV payloads for stable document chunks.
3. Store packed shards in a location visible to serving workers.
4. Store manifest rows in a durable table or service.
5. Materialize requests with `DocumentKVWorkflow`.
6. Hand the payload and metadata to vLLM, SGLang, or a custom adapter.
7. Compare against a no-cache baseline before treating speedups as real.

## vLLM

Cachet ships vLLM adapter modules in the same `cachet-kv` distribution. The
serving environment still installs and runs vLLM. The native path is:

```text
Cachet handoff -> vLLM KV transfer params -> Cachet vLLM provider -> vLLM paged KV blocks
```

Start with [`native-engine-integration.md`](native-engine-integration.md) when
you are wiring real vLLM block allocation and load behavior.

For serving-engine KV offload under GPU memory pressure, use vLLM's native
offloading connector. Cachet exposes a validated config builder:

```bash
cachet-engine-launch-config build-vllm-offload \
  --cpu-bytes-to-use 34359738368 \
  --block-size 64 \
  --secondary-fs-root /local_disk0/vllm-kv-offload
```

The output is a vLLM `--kv-transfer-config` JSON object for platform-managed
runtime KV offload. It is intentionally separate from `build-vllm`, which emits
Cachet's document-KV import connector config.

## SGLang

Cachet ships SGLang adapter modules for HiCache-style handoff metadata. The
serving environment still installs and runs SGLang. The native path is:

```text
Cachet handoff -> SGLang request metadata -> Cachet HiCache provider -> SGLang prefix binding
```

The current SGLang evidence validates live cache-hit correctness and quality.
Treat performance results separately from correctness until your prompt lengths,
cache-hit sizes, and runtime configuration match your workload.

SGLang HiCache host-pool and policy knobs can be emitted with the same launch
config helper:

```bash
cachet-engine-launch-config build-sglang \
  --hicache-size-gb 64 \
  --page-size 8 \
  --hicache-write-policy write_through_selective \
  --hicache-storage-prefetch-policy timeout
```

For code that launches SGLang directly, `cachet.build_sglang_hicache_server_args`
returns the equivalent CLI flags.

## Hierarchical Document-KV Persistence

Cachet document KV payloads can use a separate hierarchy before they are handed
to the serving engine:

```python
from cachet import ChunkCache, KVMaterializer, RoutedRangeReader

reader = RoutedRangeReader()
cache = ChunkCache(
    cpu_max_bytes=8 * 1024**3,
    local_dir="/local_disk0/cachet/document-kv",
    local_max_bytes=256 * 1024**3,
    local_promotion_threshold=2,
)
materializer = KVMaterializer(cache=cache, reader=reader)
```

With `local_promotion_threshold=2`, a cold chunk referenced by a `uc-volume:`,
`/Volumes/...`, `disk:`, or `memory:` shard URI is served from its authoritative
storage on first access, then promoted into the local-disk tier after it becomes
popular. Local-disk files are evicted with LRU once `local_max_bytes` is full.
CPU RAM remains the fastest byte cache and is governed by `cpu_max_bytes`.

## Runtime Verification

Run the runtime KV offload probe before enabling these paths in production:

```bash
cachet-runtime-kv-offload-probe \
  --work-dir /local_disk0/cachet-runtime-kv-offload-probe \
  --output-json /local_disk0/cachet-runtime-kv-offload-probe/probe.json
```

The probe validates the vLLM offload config shape, SGLang HiCache launch args,
and Cachet document-KV promotion/eviction behavior from cold storage, CPU RAM,
and local disk. On an environment where the serving runtime is installed, add
`--require-vllm-offloading-import` or `--require-sglang-package` to make the
probe fail closed when the platform support is missing.

## Databricks

Databricks is one supported production and benchmark environment, not a
requirement for using Cachet. Use it when you need managed GPU jobs, Unity
Catalog Volumes, or reproducible benchmark runs on the target hardware.

Databricks job templates live under [`../databricks/`](../databricks/). The
public quickstart does not require them. The ad hoc runner in
[`../databricks/runtime-kv-offload-probe/`](../databricks/runtime-kv-offload-probe/)
executes the runtime verification probe from an installed wheel on a managed
cluster.

## Stable User Commands

Most production users start with Python. The local example module is stable for
new users:

```bash
python -m cachet.quickstart_local
```

The user-facing CLI that is useful outside release operations is:

```bash
cachet-engine-launch-config --help
```

Maintainer-only commands for release bundles, PR evidence, repository hygiene,
GitHub governance, Databricks job generation, and benchmark publication are
documented in [`release-ops/README.md`](release-ops/README.md).

## Benchmarks

Use [`../benchmarks/README.md`](../benchmarks/) for the current
human-readable benchmark summary. Keep raw run output out of the source tree;
promote only compact, sanitized reports when they support a public claim.
