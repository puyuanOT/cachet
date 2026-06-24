# SGLang g6/L4 Prepared V1 Attempt: Config Swap Failure

This folder tracks the first Databricks attempt to run the prepared SGLang V1
benchmark path with generated handoffs for all four V1 datasets. It is
standalone benchmark-readiness evidence, not `pr-evidence/` and not ignored
local `databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `514040136831626` |
| Task run | `799338455029344` |
| Run name | `Cachet SGLang prepared V1 sglang-prepared-v1-g6-054b721-20260624_122945` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `054b721` |
| Package wheel | `cachet_kv-0.2.0-py3-none-any.whl`, sha256 `1bcb3b3f9cff17dfa5a9a6267a9341aa1749377f843afeff43e3846d095fb5b1` |
| Mode | Prepared SGLang V1 benchmark datasets, generated SGLang handoff bundles, plain completion requests, two live benchmark repeats, `--attention-backend triton`, `--sampling-backend pytorch`, `--enable-deterministic-inference`, `hicache_storage_prefetch_policy=wait_complete`, page size `16`, `bfloat16` handoff generation |
| Current state | `FAILED`; import probe and prepared handoff generation passed, but the runner failed before SGLang server launch and before benchmark rows were written |
| Benchmark result | Not published; no SGLang V1 latency or throughput measurements were produced |

## Scope

This run proves that the Databricks g6/L4 environment can install the Cachet
wheel, import SGLang `0.5.10.post1`, see an NVIDIA L4, wire the production
SGLang HiCache provider factory, install the request metadata bridge, and
generate SGLang prepared handoff JSONLs for Biography, HotpotQA, MusiQue, and
NIAH.

It does not prove live SGLang benchmark latency or throughput. The runner
failed after successful prepared handoff generation and before server launch
with:

```text
prepared SGLang benchmark datasets must not be combined with single live handoff fields
```

The diagnosis is that `run_sglang_live_smoke` swapped the original dataset
specs for generated prepared-handoff dataset specs using `dataclasses.replace`,
but preserved normalized empty single-request handoff fields from the original
config. Prepared dataset specs then correctly rejected the mixed configuration.
The next code PR should clear the single-request handoff fields during that
generated prepared-dataset config swap, add a regression test, and rerun this
Databricks target.

## Artifacts

- [`failed_run.json`](failed_run.json) contains a compact, sanitized snapshot
  of the terminal Databricks state, import probe, prepared handoff generation,
  launch configuration, and failure diagnosis.

Raw Databricks API responses, package wheels, driver logs, generated datasets,
handoff payloads, page-key lists, prompt text, task-output blobs, and local
scratch outputs stay out of this folder.
