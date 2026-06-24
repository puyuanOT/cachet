# Databricks Benchmark Evidence

These folders contain the current sanitized Databricks benchmark artifacts for
Cachet. They are meant for release and benchmark review, not for PR process
bookkeeping.

For human-facing reports, start with the standalone benchmark folders and their
dated report subfolders: [`../vllm/`](../vllm/),
[`../sglang/`](../sglang/), [`../storage/`](../storage/), and
[`../native-engine/`](../native-engine/). Use [`CURRENT.md`](CURRENT.md) for the
current Databricks artifact snapshot, then use each dated folder README here
for source-artifact details. The JSON files are validated source records used to
audit the report claims:

- `document_kv.benchmark_run.v1` for V1 latency and quality benchmarks.
- `cachet.sglang_live_benchmark.v1` for SGLang synthetic live endpoint
  measurements and prepared V1 live attempts.
- `document_kv.storage_benchmark.v1` for Memory, Disk, and Unity Catalog reader
  benchmarks.
- `document_kv.databricks_run_status.v1` summaries for terminal successful QA
  Databricks runs.
- `document_kv.engine_kv_connector_probe.v1` and
  `document_kv.engine_kv_connector_actions.v1` for native vLLM/SGLang connector
  probes.

Some records are canonical release-bundle inputs, such as
`document_kv.benchmark_run.v1`; raw prepared SGLang live V1 records are
dedicated release-bundle inputs through the `sglang_live_v1_benchmark` role.
Synthetic SGLang live benchmark and integration records remain separate from
release-bundle consumption unless validators explicitly ingest those scoped
record types.

Folders containing `v1_benchmark.json` are full V1 latency and quality
benchmark reports. SGLang live benchmark folders under `../sglang/` can be
cited as scoped synthetic live endpoint measurements, prepared V1 live
benchmark evidence, or pre-publication prepared attempt evidence when they
include sanitized `cachet.sglang_live_benchmark.v1` records. Prepared
`scope=live_v1_release` records remain a distinct SGLang live report surface
rather than canonical `document_kv.benchmark_run.v1` release-bundle inputs, but
strict bundles validate raw sidecars through `sglang_live_v1_benchmark`. The
standalone report JSON can be compacted for human review. The native-engine
probe folder proves
provider-backed vLLM/SGLang integration against engine-owned KV block managers,
but it does not publish latency, throughput, or quality benchmark results.

Live-readiness folders, such as the generated-handoff SGLang smoke under
[`../sglang/`](../sglang/), can track pending or failed Databricks attempts
before a terminal successful benchmark result exists. Keep those files out of
release-bundle inputs until they are promoted to terminal successful benchmark
artifacts.

Raw local run directories stay under ignored `databricks-runs/`.
