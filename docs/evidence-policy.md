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

The built-in HotpotQA publication scorer ports only the answer EM/F1 portion of
the official
[`hotpot_evaluate_v1.py`](https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py)
script, pinned to upstream commit
`3635853403a8735609ee997664e1528f4480762a`. Cachet does not claim supporting-fact
or joint HotpotQA metrics because the serving benchmark does not collect
supporting-fact predictions. Other built-in V1 dataset scorers remain diagnostic
and cannot satisfy a publication gate until a versioned dataset-approved scorer
is registered.

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
