# SGLang g6/L4 Live Handoff Smoke: Attach-Time Hash Tracking Gate

This folder tracks the generated-handoff SGLang live smoke launched after PR
#466 merged on the target Databricks hardware. It is standalone
benchmark-readiness evidence, not `pr-evidence/` and not ignored local
`databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `672750124167579` |
| Task run | `178988370465155` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `1c7084f` |
| Mode | Generated live Cachet handoff, logical prompt text, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete`, batch prior-hash metadata bridge from PR #466 |
| Current state | `FAILED`; the provider factory loaded, but the SGLang request metadata bridge failed during import probe because the hash-tracking installability check looked for `get_hash_str` in the wrong controller lifecycle hook |
| Benchmark result | Not published; no live handoff generation, SGLang server launch, latency, throughput, or quality result was produced from this run |

## Scope

This run proves that the strict PR #466 bridge gate was pointed at the wrong
SGLang controller lifecycle method. The PR #466 wheel was present and the
provider factory loaded. SGLang 0.5.10.post1 assigns
`self.get_hash_str` inside `HiCacheController.attach_storage_backend`, before
the dynamic backend factory constructs Cachet's `DocumentKVHiCacheBackend`.
Cachet's bridge installation was called from that backend constructor, so the
current controller instance already had `get_hash_str`, but the install-time
preflight rejected the class because it inspected `__init__` instead of the
storage attach path and `_storage_hit_query`.

The follow-up fix must treat `attach_storage_backend` plus a
`_storage_hit_query` implementation that calls `self.get_hash_str` as the
patchable SGLang shape. The runtime `_storage_hit_query` wrapper must still
wrap the current instance before storage queries run, because the active attach
call has already passed the assignment point by the time Cachet's backend is
constructed.

This folder is evidence for that blocker, not a benchmark result.

## Promotion Criteria

Promote a replacement smoke run from readiness evidence to a published SGLang
benchmark only if all of these are true:

- A new Databricks run terminates with `result_state=SUCCESS` on `g6.8xlarge`.
- The import probe reports `document_kv_request_metadata_bridge_ok=true`.
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
