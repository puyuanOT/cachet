# Databricks Benchmark Evidence

These folders contain the current sanitized Databricks benchmark artifacts for
Cachet. They are meant for release and benchmark review, not for PR process
bookkeeping.

Start with [`CURRENT.md`](CURRENT.md) for the human-readable current benchmark
and release-evidence snapshot, then use the per-folder README for run-specific
details. The JSON files are the validated records copied from the strict
release-bundle evidence set:

- `document_kv.benchmark_run.v1` for V1 latency and quality benchmarks.
- `document_kv.storage_benchmark.v1` for Memory, Disk, and Unity Catalog reader
  benchmarks.
- `document_kv.databricks_run_status.v1` summaries for terminal successful QA
  Databricks runs.
- `document_kv.engine_kv_connector_probe.v1` and
  `document_kv.engine_kv_connector_actions.v1` for native vLLM/SGLang connector
  probes.

Raw local run directories stay under ignored `databricks-runs/`.
