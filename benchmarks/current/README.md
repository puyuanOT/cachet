# Current Benchmark Results

This folder is the human-facing front door for Cachet benchmark results. Use it
when you want the current Databricks benchmark answer without reading
`pr-evidence/` or local `databricks-runs/` scratch output.

## Read These First

| Result | Folder | Databricks run | Target | Human result |
| --- | --- | --- | --- | --- |
| vLLM strict V1 benchmark | [`../vllm/2026-06-23-g6-l4-v1/`](../vllm/2026-06-23-g6-l4-v1/) | `872615985402004` | `aws-g6-l4` / `g6.8xlarge` | `ok=true`; 24 measurements; 5.27x-6.97x TTFT speedup; unchanged quality |
| vLLM compatibility benchmark | [`../vllm/2026-06-23-g5-a10g-v1-compatibility/`](../vllm/2026-06-23-g5-a10g-v1-compatibility/) | `566743786103032` | `aws-g5-a10g` / `g5.8xlarge` | `ok=true`; 24 measurements; 4.66x-6.04x TTFT speedup; unchanged quality |
| SGLang prepared V1 live benchmark | [`../sglang/2026-06-24-g6-l4-prepared-v1-release-suite-success/`](../sglang/2026-06-24-g6-l4-prepared-v1-release-suite-success/) | `48413356233422` | `aws-g6-l4` / `g6.8xlarge` | `ok=true`; 16 measurements; 8/8 Cachet cache-hit validations; unchanged quality; no speedup on short prompts |
| SGLang synthetic live benchmark | [`../sglang/2026-06-24-g6-l4-live-benchmark-synthetic-niah-success/`](../sglang/2026-06-24-g6-l4-live-benchmark-synthetic-niah-success/) | `238535418152934` | `aws-g6-l4` / `g6.8xlarge` | `ok=true`; two Cachet-backed cache repeats with 175 cached tokens; unchanged quality; tiny-prompt scoped evidence |
| Storage-reader benchmark | [`../storage/2026-06-21-g6-l4-storage-readers/`](../storage/2026-06-21-g6-l4-storage-readers/) | `948365719597221` | `aws-g6-l4` / `g6.8xlarge` | Memory, Disk, and real Unity Catalog readers passed with zero reader errors |
| Native engine integration evidence | [`../native-engine/2026-06-23-g6-l4-native-engine-probes/`](../native-engine/2026-06-23-g6-l4-native-engine-probes/) | `934698284395881` | `aws-g6-l4` / `g6.8xlarge` | Provider-backed vLLM and SGLang native probes succeeded; integration evidence, not latency evidence |

## Artifact Boundary

The linked dated folders are the standalone benchmark report folders. They
contain concise `README.md` summaries plus compact sanitized JSON records
beside each report. Those JSON records are the durable benchmark evidence to
cite and audit.

Use [`../databricks/CURRENT.md`](../databricks/CURRENT.md) when you need the
Databricks release-source mirror and run-status snapshot. Use `pr-evidence/`
only for PR validation and release-audit traceability, not as the benchmark
report surface.

Do not commit Databricks tokens, raw Jobs API responses, package wheels, logs,
generated datasets, handoff payload blobs, or local scratch output here.
