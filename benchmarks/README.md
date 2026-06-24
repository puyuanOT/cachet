# Benchmark Reports

This directory contains curated, human-readable Cachet benchmark reports and
the sanitized artifacts that back them. It is separate from `pr-evidence/`,
which records PR review and validation history, and from `databricks-runs/`,
which remains ignored local scratch output.

Start with the standalone report folders:

| Folder | Purpose | Current status |
| --- | --- | --- |
| [`vllm/`](vllm/) | vLLM latency and quality benchmark report | Published g6/L4 target and g5/A10G compatibility results |
| [`sglang/`](sglang/) | SGLang benchmark status and live smoke folders | Generated-handoff Databricks smokes reached a full 175-token external cache hit covering a generated Qwen-chat prefix through the chat-completions path with deterministic no-thinking controls, explicit Triton/PyTorch backend controls, a minimal no-thinking request body, and a passing post-flush model-quality canary; latest blocker is failed Qwen3/SGLang baseline and cache-arm quality; live latency and quality benchmark pending |
| [`storage/`](storage/) | Storage-reader benchmark report | Published Memory, Disk, and Unity Catalog results |
| [`native-engine/`](native-engine/) | Native connector integration evidence | Published provider-backed vLLM and SGLang probes |

The Databricks artifact index remains available at
[`databricks/CURRENT.md`](databricks/CURRENT.md). Each benchmark run gets a
dated folder under `databricks/` with a human-readable `README.md` plus the
sanitized JSON artifacts needed to audit the claim.

Use this boundary when adding or reviewing benchmark output:

- `benchmarks/vllm/`, `benchmarks/sglang/`, `benchmarks/storage/`, and
  `benchmarks/native-engine/` are standalone human-readable report folders.
- Pending live-readiness runs can live under the relevant engine folder, such
  as `benchmarks/sglang/2026-06-23-g6-l4-live-handoff-smoke/`, when they are
  useful to review before a terminal benchmark result exists.
- `benchmarks/databricks/CURRENT.md` is the current human-readable summary.
- `benchmarks/databricks/<date>-<target>-<purpose>/README.md` is the
  run-specific human-readable source-artifact report.
- Folders with `v1_benchmark.json` are latency and quality benchmark reports.
- The native-engine probe folder is integration evidence only; it is not a
  vLLM or SGLang latency/quality benchmark report.
- JSON files beside each report are sanitized, schema-validated source records
  for the claims in that report.
- `pr-evidence/` is PR validation and release-audit material, not the benchmark
  report surface.
- `databricks-runs/` is ignored local scratch output and must not be used as the
  durable benchmark location.

Do not commit Databricks tokens, raw Jobs API responses, package wheels, logs,
generated dataset payloads, or cluster-local scratch output here.

## Current Databricks Evidence

| Folder | Purpose | Databricks run | Target | Result |
| --- | --- | --- | --- | --- |
| [`databricks/2026-06-23-g6-l4-v1`](databricks/2026-06-23-g6-l4-v1/) | Strict V1 benchmark target | `872615985402004` | `aws-g6-l4` / `g6.8xlarge` | `ok=true`; 24 measurements; 5.27x-6.97x TTFT speedups |
| [`databricks/2026-06-23-g5-a10g-v1-compatibility`](databricks/2026-06-23-g5-a10g-v1-compatibility/) | Non-default compatibility benchmark | `566743786103032` | `aws-g5-a10g` / `g5.8xlarge` | `ok=true`; 24 measurements; 4.66x-6.04x TTFT speedups |
| [`databricks/2026-06-21-g6-l4-storage-readers`](databricks/2026-06-21-g6-l4-storage-readers/) | Memory, Disk, and Unity Catalog storage readers | `948365719597221` | `aws-g6-l4` / `g6.8xlarge` | `ok=true`; real UC Volume; zero reader errors |
| [`databricks/2026-06-23-g6-l4-native-engine-probes`](databricks/2026-06-23-g6-l4-native-engine-probes/) | vLLM and SGLang native connector probes, not latency benchmarks | `934698284395881` | `aws-g6-l4` / `g6.8xlarge` | `ok=true`; provider-backed native probes succeeded |

SGLang live-readiness failures are tracked in standalone folders under
[`sglang/`](sglang/), including the latest canary-flush cache-hit quality
failure at
[`sglang/2026-06-24-g6-l4-live-handoff-smoke-canary-flush-cache-hit-quality-failure/`](sglang/2026-06-24-g6-l4-live-handoff-smoke-canary-flush-cache-hit-quality-failure/).

The strict V1 publication target remains AWS g6/L4. The g5 folder is retained
only as compatibility evidence and cannot replace the g6/L4 release target.
