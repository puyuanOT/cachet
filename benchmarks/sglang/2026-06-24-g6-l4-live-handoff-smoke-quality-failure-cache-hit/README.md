# SGLang g6/L4 Live Handoff Smoke: Cache Hit With Quality Failure

This folder tracks the generated-handoff SGLang live smoke launched after PR
#467 merged on the target Databricks hardware. It is standalone
benchmark-readiness evidence, not `pr-evidence/` and not ignored local
`databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `348824841142825` |
| Task run | `401615265653490` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `6a697b3` |
| Mode | Generated live Cachet handoff, logical prompt text, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete`, attach-time SGLang hash tracking from PR #467 |
| Current state | `FAILED`; import probe, request metadata bridge, and generated live handoff passed; SGLang reported a positive cache-arm hit for the first 128 external pages, but the later 46-key split query still missed and both live quality checks failed |
| Benchmark result | Not published; no SGLang latency, throughput, or quality benchmark result is claimed from this run |

## Scope

This run proves that the PR #467 attach-time hash tracking fix reached the live
SGLang runtime. The provider factory loaded, the request metadata bridge
reported `controller_hash_tracking_patched=true`, and generated live handoff
creation produced 150 SGLang HiCache page keys. SGLang then reported a positive 128-token external cache hit for the cache-arm prefill, out of 174 prompt tokens.

The run is still not publishable. The server log shows the first 128 generated
pages hydrated, followed by a later 46-key `binding_miss` with
`last_hash_present=true`. The live smoke also failed the quality gate for both
the cache arm and the baseline arm, so this evidence must not be cited as an
SGLang latency, throughput, or quality benchmark.

The next SGLang serving fix must handle the later split HiCache query and make
the live quality check reliable on Qwen3/SGLang before promotion.

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
