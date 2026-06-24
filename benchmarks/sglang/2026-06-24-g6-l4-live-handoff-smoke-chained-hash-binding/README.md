# SGLang g6/L4 Live Handoff Smoke: Chained Hash Binding Blocker

This folder tracks the latest generated-handoff SGLang live smoke on the
target Databricks hardware. It is standalone benchmark-readiness evidence, not
`pr-evidence/` and not ignored local `databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `476430354490832` |
| Task run | `331512280797427` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `73f051b` |
| Mode | Generated live Cachet handoff, SGLang HiCache page keys, logical prompt text, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete`, split runtime-key binding from PR #464 |
| Current state | `FAILED`; the PR #464 wheel was present and hydrated the first 128 runtime pages, but the later 46-key query still missed because it is chained from SGLang's runtime `last_hash` |
| Benchmark result | Not published; no SGLang latency, throughput, or quality result was produced from this run |

## Scope

This run proves the split runtime-key binding code from PR #464 was present in
the Databricks wheel and that the remaining miss is narrower than the previous
partial page-binding blocker. SGLang launched on an NVIDIA L4, the import probe
passed, the request metadata bridge passed, the generated live handoff produced
150 SGLang HiCache page keys, and the provider hydrated the first 128 runtime
HiCache keys.

The second 46-key storage query still missed. Upstream SGLang builds that later
query from the previous host-prefix `last_hash`, so its runtime keys do not
equal a plain contiguous slice of the generated `sglang_hicache_page_keys`.
Cachet therefore needs to propagate `operation.last_hash` through
`HiCacheStorageExtraInfo.extra_info` and use it as the anchor for later split
queries. This folder is the evidence for that blocker, not a benchmark result.

## Promotion Criteria

Promote a replacement smoke run from readiness evidence to a published SGLang
benchmark only if all of these are true:

- A new Databricks run terminates with `result_state=SUCCESS` on `g6.8xlarge`.
- The SGLang live smoke writes `sglang-live-smoke.json` with `ok=true`.
- The cache arm uses logical prompt text or an equivalent validated virtual
  prefix binding path.
- Split SGLang HiCache storage queries hydrate all matching generated handoff
  pages, including later chunks chained from SGLang's runtime `last_hash`.
- The cache-arm request reports positive SGLang cached-token validation.
- The positive cached-token validation is matched to the cache request's
  prompt-token total, not to warmup requests or later baseline prefix-cache
  reuse.
- Latency, throughput, and quality numbers are recorded in the same benchmark
  schema used by the vLLM report.

## Artifacts

- [`failed_run.json`](failed_run.json) contains a compact, sanitized snapshot
  of the terminal failed Databricks run state and blocker.

Raw Databricks API responses, package wheels, driver logs, generated payloads,
page-key lists, and local scratch outputs stay out of this folder.
