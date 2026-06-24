# Databricks Benchmark Evidence

These folders contain the current sanitized Databricks benchmark artifacts for
Cachet. They are meant for release and benchmark review, not for PR process
bookkeeping.

For human-facing reports, start with the standalone benchmark folders:
[`../vllm/`](../vllm/), [`../sglang/`](../sglang/),
[`../storage/`](../storage/), and [`../native-engine/`](../native-engine/).
Use [`CURRENT.md`](CURRENT.md) for the current Databricks artifact snapshot,
then use each dated folder README for run-specific source details. The JSON
files are the validated records copied from the strict release-bundle evidence
set:

- `document_kv.benchmark_run.v1` for V1 latency and quality benchmarks.
- `document_kv.storage_benchmark.v1` for Memory, Disk, and Unity Catalog reader
  benchmarks.
- `document_kv.databricks_run_status.v1` summaries for terminal successful QA
  Databricks runs.
- `document_kv.engine_kv_connector_probe.v1` and
  `document_kv.engine_kv_connector_actions.v1` for native vLLM/SGLang connector
  probes.

Only folders containing `v1_benchmark.json` should be cited as latency and
quality benchmark reports. The native-engine probe folder proves provider-backed
vLLM/SGLang integration against engine-owned KV block managers, but it does not
publish SGLang latency, throughput, or quality benchmark results.

Pending live-readiness folders, such as the generated-handoff SGLang smoke under
[`../sglang/`](../sglang/), can be tracked before a Databricks run reaches a
terminal state. Keep those files out of release-bundle inputs until they are
promoted to terminal successful benchmark artifacts.

Raw local run directories stay under ignored `databricks-runs/`.
