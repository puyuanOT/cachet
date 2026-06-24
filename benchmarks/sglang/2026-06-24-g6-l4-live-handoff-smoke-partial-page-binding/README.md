# SGLang g6/L4 Live Handoff Smoke: Partial Page-Binding Blocker

This folder tracks the latest generated-handoff SGLang live smoke on the
target Databricks hardware. It is standalone benchmark-readiness evidence, not
`pr-evidence/` and not ignored local `databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `521023980659718` |
| Task run | `893706825011644` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `a0249d0` |
| Mode | Generated live Cachet handoff, SGLang HiCache page keys, logical prompt text, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete` |
| Current state | `FAILED`; SGLang reported 128 cached tokens from Cachet handoff pages, but Cachet missed the later 46-key storage query and both live checks failed |
| Benchmark result | Not published; no SGLang latency, throughput, or quality result was produced from this run |

## Scope

This run proves the previous zero-cache-hit blocker was cleared enough for
SGLang to wait on Cachet-backed storage pages. SGLang launched with
`qwen3-4b-instruct`, the import probe passed, the request metadata bridge
passed, the live handoff generator produced 150 SGLang HiCache page keys, and
the provider factory was the production Cachet SGLang HiCache provider.

The run still is not a benchmark result. Cachet hydrated the first 128 runtime
HiCache keys from the generated handoff and SGLang reported 128 cached tokens
for the cache-arm prefill. The same request then issued a second 46-key storage
query. Cachet treated that query as starting at the beginning of the expected
page-key list, missed the binding, and did not hydrate the remaining generated
page keys. The partial 128-of-150 page injection left both live checks
generating repeated text instead of the expected answer.

The next engineering fix is to bind later SGLang HiCache storage queries
against contiguous slices of the generated `sglang_hicache_page_keys`, hydrate
only the matching remaining handoff pages, and leave suffix-only runtime keys
to SGLang's normal compute path.

## Promotion Criteria

Promote a replacement smoke run from readiness evidence to a published SGLang
benchmark only if all of these are true:

- A new Databricks run terminates with `result_state=SUCCESS` on `g6.8xlarge`.
- The SGLang live smoke writes `sglang-live-smoke.json` with `ok=true`.
- The cache arm uses logical prompt text or an equivalent validated virtual
  prefix binding path.
- Split SGLang HiCache storage queries hydrate all matching generated handoff
  pages, including later chunks.
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
