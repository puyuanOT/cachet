# Evidence Policy

Cachet keeps benchmark results, release evidence, and PR traceability in the
repository so release claims remain auditable. These folders should stay
navigable: human-facing summaries belong in `README.md` files, while bulky raw
service output stays outside Git.

## Evidence Qualification

Every benchmark result must be labeled at one of these levels. The label
describes the strength of the evidence, not whether the software is useful.

| Level | Appropriate use | Minimum expectation |
| --- | --- | --- |
| Smoke | Execution and correctness debugging | A small run that proves the path executes; it makes no comparative performance or quality claim |
| Canary | Regression detection and directional engineering decisions | Complete paired arms over the declared examples, a reproducible setting, recorded failures, and sanitized evidence; it is not a publication claim |
| Publication | Public performance or quality claims | Meaningful distinct-sample coverage, example-clustered uncertainty, the dataset's approved versioned scorer, immutable provenance, and a passing publication gate; cold-cache claims also require request-correlated cold-state attestation |

A private Databricks or DBFS path is useful provenance, but it is not durable
publication evidence by itself. If the compact measurements, identities,
statistics, and gate result are not available as sanitized committed records,
the corresponding table must say **provisional / non-publication-qualified**.
This remains true even when the private job succeeded or the raw sample count
is large.

Clear Databricks single-user principals are private control-plane material.
Execution plans, job records, submit-payload snapshots, and controller leases
that contain `single_user_name` must remain in ignored or access-restricted
control roots and must not be copied into committed benchmark evidence. Public
sanitized evidence may retain only the SHA-256 principal attestation returned
by the live current-user check.

The default V1 registry contains four publication-approved, versioned scorer
contracts. All four require the shared versioned final-answer parser and emit
only their declared metrics:

- Biography uses Cachet's normalized-title exact match for the versioned entity
  identification task. Its normalizer preserves name-significant punctuation.
- HotpotQA ports only the answer EM/F1 portion of the official
  [`hotpot_evaluate_v1.py`](https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py)
  script, pinned to upstream commit
  `3635853403a8735609ee997664e1528f4480762a`. Cachet does not claim
  supporting-fact or joint metrics because the serving benchmark does not
  collect supporting-fact predictions.
- MusiQue ports the official v1.0 answer EM/F1 implementation at pinned commit
  `922ac98f19a201998dbdae6d7f2887a5258dbdeb`, maximized over the preserved
  answer aliases. Cachet does not claim support-index or answerability-group
  metrics.
- NIAH uses exact requested-value accuracy over the frozen 8k/16k/32k by
  10%/50%/90% needle-position grid. The first canonical cell contains 112
  examples and each of the other eight contains 111; redistribution is not an
  equivalent 1,000-example grid.

The generic built-in answer diagnostic remains diagnostic and cannot satisfy a
publication gate in place of one of these registered scorer contracts.

One-arm physical latency jobs are sealed as `smoke` component evidence because
a Baseline-only job contains no cache arm and cannot truthfully make a
comparative publication claim. This does not relax the publication gate. The
vLLM 0.27.1 campaign finalizer must independently revalidate all 115 component
records, exactly reaggregate the latency summary and complete paired full-score
aggregate, prove their uninterrupted shared-ledger lineage, and emit one
sanitized `cachet.vllm_0271_publication_report.v1`. The only passing campaign
gate is the exact `document_kv.benchmark_publication_gate.v1` record whose
`benchmark_payload_digest` equals that report's `closed_record_sha256`.
Each preserved full-score raw run must also replay the frozen suite, isolated
arm, model/runtime, 64-token, temperature-zero, concurrency-four, one-pass
benchmark manifest. The replay rebuilds the complete arm and runtime manifest
from the governed natural and enriched inputs, binds Vanilla to runtime-prompt
delivery, and matches the SHA-256 of the exact prompt serialized by the client
along with server-usage token accounting. That validated protocol is carried
through every shard
evidence record, the aggregate, and the public report; it is not reconstructed
from a label at publication time. Every metric summary carries an
`invalid_parser_score_sum` audit value that must be exactly zero in the raw
reaggregation, aggregate validator, report validator, and table renderer.

The campaign report and gate are an inseparable promotion pair. The report may
contain only whitelisted aggregate tables, scorer/parser contracts, ledger
accounting, and content-addressed source bindings. It must not copy artifact
URIs, principals, Jobs API identifiers, prompts, answers, raw outputs, or logs.
Missing, duplicated, tampered, locally scoped, interleaved-ledger, active-ledger,
or below-headroom inputs fail closed and remain provisional.
The public pair additionally pins the frozen campaign, inventory, shard-plan,
and execution-plan digests, requires 15,360–16,640 cold-read attestations, and
accepts only the authorized 1,024-hour ledger cap.
Each file is published from complete immutable temporary bytes; the gate is
published first and the report last as the pair's commit file. An exact
read-only gate-only interruption is retryable, while a report without its gate
or any mismatched partial fails closed. Authority-replaying loads accept the
writer's sealed `0444` mode or Git checkout modes made owner-only/group-readable
by a secure umask (`0600`, `0640`, or `0644`, including read-only variants);
crash recovery accepts only the sealed mode, and every load still requires
stable canonical bytes plus complete source and ledger replay.

Human-readable campaign tables are a deterministic projection of that exact
pair, not a second hand-edited evidence source. The publication-table renderer
uses fixed row identities and order, type-exact finite numeric formatting, and
named Markdown regions. Promotion must byte-compare every governed region with
fresh renderer output. The benchmark surface is valid only in one of two
states: entirely pending with no report folder, or finalized with the complete
report/gate/README trio. A partial pair, mixed pending and populated cells,
unexpected row, or one-cell transcription difference fails closed.
The child README is renderer-owned too: it names both exact JSON files and the
validated report digest, so a hand-written or stale appendix index cannot
silently accompany an otherwise valid pair.

## Fair Comparison Design

There are two valid comparison shapes:

1. A **method comparison** holds one immutable experiment setting fixed and
   changes only the method arm. The setting includes the exact model and
   tokenizer revisions, engine version, hardware shape, model and KV
   precision, storage boundary, dataset sample identities, logical input and
   output budgets, prompt and decoding configuration, concurrency, cache
   state, repetitions, ordering, and seeds.
2. A **one-variable ablation** holds the method and every other setting fixed
   while changing one declared factor, such as hardware, quantization, storage
   tier, or serving platform. If several factors change, report a separate
   setting comparison rather than calling it an ablation.

The typed `arms[].runtime_environment` snapshot is authoritative for these
checks. A method comparison requires identical per-arm snapshots. A
one-variable ablation requires each arm's `setting_overrides` value to equal
the corresponding typed snapshot field, and exactly that one field may differ
from the reference arm. The older top-level model and environment fields are
compatibility summaries; an isolated merged run records `varies_by_arm` there
when necessary instead of presenting the first arm's environment as shared.

The shared decoding record stores the closed, normalized settings object
(`top_p`, stops, penalties, `ignore_eos`, and related supported keys) beside
its digest; a digest without its JSON preimage is not reproducible evidence.
When an input-token target is declared, every successful arm measurement must
report that exact logical prompt count. When `ignore_eos=true`, every
successful measurement must also reach the declared output-token target.
Static non-decoding server customizations are bound per arm by a canonical
digest, while raw values stay out of sanitized evidence and per-request cache
salts are authenticated separately. One-variable comparisons keep this digest
fixed except when the declared factor is the serving platform.

Methods may legitimately transform the physical input. Fairness therefore
means that every arm starts from the same logical documents, query, expected
answer, and decode budget; it does not require byte-identical physical prompts.
Evidence must record the transformation name and version, physical token
counts or identities, inserted or removed content, and the resulting artifact
identity. An unrecorded transformation invalidates a method comparison.

Keep offline and online costs separate. Training, artifact generation,
checkpoint loading, artifact size, and peak generation resources are offline
costs. TTFT, time to completion, request throughput, and serving resources are
online costs. Each table must state whether storage-to-CPU and CPU-to-GPU
hydration occur inside or outside the measured online request boundary.

## Representative Canary Submission Contract

Representative smoke jobs are a narrow, versioned subset of ordinary canary
experiments. The `--representative-canary` label and a registered
`--representative-workload-profile` must be supplied together; neither a
canonical arm nor `--benchmark-evidence-policy canary` implicitly applies the
representative label. Generic pinned canaries therefore remain usable for
other hardware and exploratory matrices.

The registered vLLM profiles are `vllm-8k-64-v1` and
`vllm-16k-256-v1`. They bind the logical input target and forced output budget
to 8,192/64 and 16,384/256 tokens respectively, with three repeats, request
parallelism one, prepared multi-document input, per-request prefix-cache salts,
and disabled prewarm, runtime-prompt, and process payload caches. The SGLang
profile `sglang-4k-32-v1` binds a 4,096-token context, 32 output tokens, two
repeats, the `triton` attention backend, the `pytorch` sampling backend, and
deterministic inference.

Each representative vLLM method arm runs in a separate Databricks task; the
three results are aggregated only after the isolated jobs finish. Representative
jobs use only the exact `g6.8xlarge` or `g5.8xlarge` node for the declared
hardware target, keep handoff roots under `/local_disk0`, set
`DOCUMENT_KV_EVICT_PAGE_CACHE=1`, use immutable model/tokenizer revisions, and
submit with a four-hour hard timeout and zero task retries.

## Folder Boundaries

| Path | Keep | Do not keep |
| --- | --- | --- |
| [`../benchmarks/`](../benchmarks/) | Human-readable benchmark reports, compact sanitized JSON records, current benchmark index | Raw Databricks responses, task logs, package wheels, generated datasets |
| [`release-ops/evidence/`](release-ops/evidence/) | Durable release-governance records that are not benchmark reports and not PR sidecars | PR traceability records, benchmark reports, local scratch output |
| [`release-ops/pr-evidence/`](release-ops/pr-evidence/) | Valid `document_kv.pr_evidence.v1` sidecars and validation summaries | Benchmark results, runtime logs, Databricks credentials |
| `../databricks-runs/` | Ignored local scratch output only | Tracked source, durable benchmark reports, release artifacts |
| Release bundles | Explicit publication handoff artifacts copied into durable storage | Unreviewed local worktree output or credentials |

## What To Commit

Commit small, sanitized records when they directly support a durable claim:

- benchmark report JSON beside a standalone dated benchmark `README.md`
- Databricks run-status sidecars after secrets and raw response bodies are
  removed
- dependency freshness or legacy migration records under `docs/release-ops/evidence/`
- PR evidence sidecars under `docs/release-ops/pr-evidence/`
- generated release-bundle manifests only when they are intended as durable
  release handoff material

Every committed artifact should answer one question clearly: what claim does
this prove?

For benchmark claims, a sanitized evidence set should be sufficient to audit
the result without access to the originating workspace. It should identify the
immutable experiment setting and method arms, logical sample set and scorer,
method-specific physical transformations, artifact identities, successful and
failed measurements, paired statistics, resource/offline-cost records where
claimed, cache-state attestations where relevant, and the evidence-level or
publication-gate decision. Run IDs and private storage paths may be included as
supplementary provenance, but cannot replace these records.

## What To Keep Out Of Git

Never commit credentials, Databricks tokens, OAuth material, raw Jobs API
responses, cluster logs, package wheels, generated datasets, `.env` files,
notebook checkpoints, or local temporary directories. Keep exploratory run
payloads and task status files under ignored `databricks-runs/` until a compact
sanitized record is promoted to `benchmarks/`, `docs/release-ops/evidence/`,
`docs/release-ops/pr-evidence/`, or a release bundle.

## Promotion Checklist

Before promoting output from `databricks-runs/` into a tracked folder:

- redact tokens, hosts with embedded credentials, raw headers, and request
  bodies that are not part of the audited schema
- replace raw logs with a concise result summary and schema-validated JSON
- choose the right durable folder from the boundary table above
- add or update the nearest `README.md` so a person can understand the record
  without opening every JSON file
- run repository hygiene and the focused validation command for the artifact
  type

## Why Evidence Stays

The evidence folders are intentionally separate from package implementation
code. They let maintainers prove that a release was built from a clean
worktree, tested on the target hardware, reviewed through the PR process, and
published with current dependency and governance sidecars. The goal is not to
make users browse those records day to day; the goal is to keep release claims
verifiable when someone needs to audit them.
