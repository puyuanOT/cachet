# SGLang g6/L4 Live Handoff Smoke: Batch Prior-Hash Metadata Blocker

This folder tracks the latest generated-handoff SGLang live smoke on the
target Databricks hardware. It is standalone benchmark-readiness evidence, not
`pr-evidence/` and not ignored local `databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `73938470896039` |
| Task run | `298112994701441` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `90de5c5` |
| Mode | Generated live Cachet handoff, SGLang HiCache page keys, logical prompt text, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete`, operation `last_hash` metadata from PR #465 |
| Current state | `FAILED`; the PR #465 wheel was present, but the later 46-key query still logged `last_hash_present=False` because SGLang keeps the per-batch prior hash in a local `_storage_hit_query` variable |
| Benchmark result | Not published; no SGLang latency, throughput, or quality result was produced from this run |

## Scope

This run proves that forwarding `operation.last_hash` alone is not enough for
the live SGLang 0.5.10.post1 path. SGLang launched on an NVIDIA L4, the import
probe passed, the request metadata bridge passed, the generated live handoff
produced 150 SGLang HiCache page keys, and the provider hydrated the first 128
runtime HiCache keys.

The second 46-key storage query still missed, and both storage-query log lines
reported `last_hash_present=False`. Upstream SGLang updates a local
`last_hash` variable while computing each batch of runtime hashes inside
`_storage_hit_query`; it does not write that per-batch prior hash back to the
operation before constructing `HiCacheStorageExtraInfo`. Cachet therefore needs
the bridge to track the first `prior_hash` passed to SGLang's `get_hash_str`
for each storage batch and forward that value through `extra_info`.

This folder is evidence for that blocker, not a benchmark result.

## Promotion Criteria

Promote a replacement smoke run from readiness evidence to a published SGLang
benchmark only if all of these are true:

- A new Databricks run terminates with `result_state=SUCCESS` on `g6.8xlarge`.
- The SGLang live smoke writes `sglang-live-smoke.json` with `ok=true`.
- The cache arm uses logical prompt text or an equivalent validated virtual
  prefix binding path.
- Split SGLang HiCache storage queries hydrate all matching generated handoff
  pages, including later chunks anchored by the batch `prior_hash` used to
  compute SGLang's chained runtime keys.
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
