# Benchmark Reports

This directory contains durable, human-readable Cachet benchmark reports and
the sanitized artifacts that back them. It is separate from `pr-evidence/`,
which records PR review and validation history, and from `databricks-runs/`,
which remains ignored local scratch output.

Start with the standalone report folders:

| Folder | Purpose | Current status |
| --- | --- | --- |
| [`vllm/`](vllm/) | vLLM latency and quality benchmark report | Published g6/L4 target and g5/A10G compatibility results |
| [`sglang/`](sglang/) | SGLang benchmark status and live benchmark folders | Prepared SGLang V1 live benchmark now passes on g6/L4 with 16 rows, 8/8 validated Cachet-backed cache hits, and unchanged quality; the Cachet arm is slower on the short prepared prompts, so this is a correctness and integration benchmark result rather than a speedup result |
| [`storage/`](storage/) | Storage-reader benchmark report | Published Memory, Disk, and Unity Catalog results |
| [`native-engine/`](native-engine/) | Native connector integration evidence | Published provider-backed vLLM and SGLang probes |
| [`_template/`](./_template/) | New standalone benchmark report template | Use for future human-readable run folders |

Category folders own the human-facing report pages. Dated subfolders such as
[`vllm/2026-06-23-g6-l4-v1/`](vllm/2026-06-23-g6-l4-v1/),
[`storage/2026-06-21-g6-l4-storage-readers/`](storage/2026-06-21-g6-l4-storage-readers/),
and
[`native-engine/2026-06-23-g6-l4-native-engine-probes/`](native-engine/2026-06-23-g6-l4-native-engine-probes/)
are the standalone benchmark reports to read and cite. The Databricks artifact
index remains available at [`databricks/CURRENT.md`](databricks/CURRENT.md);
its dated folders hold the sanitized JSON source records needed to audit the
claims.

Use this boundary when adding or reviewing benchmark output:

- `benchmarks/vllm/`, `benchmarks/sglang/`, `benchmarks/storage/`, and
  `benchmarks/native-engine/` are standalone human-readable report folders.
- New benchmark runs should get a dated standalone folder with a concise
  `README.md` before they are cited as results. Start from
  [`_template/README.md`](_template/README.md) so every folder includes the
  target, run id, scope, result, and artifact boundary.
- Dated folders under the category directories are durable benchmark reports,
  not temporary package-development evidence.
- Pending live-readiness runs can live under the relevant engine folder, such
  as `benchmarks/sglang/2026-06-23-g6-l4-live-handoff-smoke/`, when they are
  useful to review before a terminal benchmark result exists.
- `benchmarks/databricks/CURRENT.md` is the current human-readable summary.
- `benchmarks/databricks/<date>-<target>-<purpose>/README.md` is the
  run-specific human-readable source-artifact report.
- Folders with `v1_benchmark.json` are latency and quality benchmark reports.
- SGLang `sglang-live-benchmark.json` files with `scope=live_synthetic_niah`
  are synthetic live endpoint measurements and do not replace full V1 benchmark
  reports.
- SGLang `sglang-live-benchmark.json` files with `scope=live_v1_release` and
  `release_v1_suite=true` are prepared live V1 SGLang benchmark evidence for
  rows that carry validated Cachet handoffs and SGLang HiCache page keys. They
  remain a distinct live SGLang report surface and are not a replacement for
  canonical `v1_benchmark.json` release-bundle inputs until release validation
  explicitly consumes that record type.
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

SGLang synthetic live benchmark evidence is tracked in
[`sglang/2026-06-24-g6-l4-live-benchmark-synthetic-niah-success/`](sglang/2026-06-24-g6-l4-live-benchmark-synthetic-niah-success/):
run `238535418152934` on `aws-g6-l4` / `g6.8xlarge`, `ok=true`, two
baseline/cache repeats, 175 cached tokens in each cache repeat, unchanged
quality, and no speedup on the tiny synthetic prompt.

The latest prepared SGLang V1 Databricks benchmark is tracked in
[`sglang/2026-06-24-g6-l4-prepared-v1-release-suite-success/`](sglang/2026-06-24-g6-l4-prepared-v1-release-suite-success/):
run `48413356233422` on `aws-g6-l4` / `g6.8xlarge`. It imported SGLang on an
NVIDIA L4, validated the production HiCache provider factory and request
metadata bridge, generated prepared handoffs for Biography, HotpotQA, MusiQue,
and NIAH, validated prepared handoff coverage, launched SGLang, wrote 16 live
measurement rows, and passed 8/8 cache-hit validations. Quality deltas were
`0.0`, but the Cachet arm was slower on the short prepared prompts.

The previous prepared SGLang V1 Databricks attempt is tracked in
[`sglang/2026-06-24-g6-l4-prepared-v1-padded-token-validation-failure/`](sglang/2026-06-24-g6-l4-prepared-v1-padded-token-validation-failure/):
run `918882025776007` on `aws-g6-l4` / `g6.8xlarge`. It wrote 16 live
measurement rows but failed publication validation before PR #500 fixed padded
SGLang prompt-token matching.

The first prepared SGLang V1 Databricks attempt is tracked in
[`sglang/2026-06-24-g6-l4-prepared-v1-config-swap-failure/`](sglang/2026-06-24-g6-l4-prepared-v1-config-swap-failure/):
run `514040136831626` on `aws-g6-l4` / `g6.8xlarge`. It generated all prepared
handoffs but failed before server launch because the generated prepared dataset
config swap preserved single-request handoff fields.

SGLang live-readiness evidence is tracked in standalone folders under
[`sglang/`](sglang/), including the current baseline-isolated successful smoke
at
[`sglang/2026-06-24-g6-l4-live-handoff-smoke-baseline-isolated-success/`](sglang/2026-06-24-g6-l4-live-handoff-smoke-baseline-isolated-success/)
with a full 175-token external cache hit, plus earlier failed smokes that
document the blocker history.

The strict V1 publication target remains AWS g6/L4. The g5 folder is retained
only as compatibility evidence and cannot replace the g6/L4 release target.
