# Evidence Policy

Cachet keeps benchmark results, release evidence, and PR traceability in the
repository so release claims remain auditable. These folders should stay
navigable: human-facing summaries belong in `README.md` files, while bulky raw
service output stays outside Git.

## Folder Boundaries

| Path | Keep | Do not keep |
| --- | --- | --- |
| [`../benchmarks/`](../benchmarks/) | Human-readable benchmark reports, compact sanitized JSON records, current benchmark index | Raw Databricks responses, task logs, package wheels, generated datasets |
| [`../evidence/`](../evidence/) | Durable release-governance records that are not benchmark reports and not PR sidecars | PR traceability records, benchmark reports, local scratch output |
| [`../pr-evidence/`](../pr-evidence/) | Valid `document_kv.pr_evidence.v1` sidecars and validation summaries | Benchmark results, runtime logs, Databricks credentials |
| `../databricks-runs/` | Ignored local scratch output only | Tracked source, durable benchmark reports, release artifacts |
| Release bundles | Explicit publication handoff artifacts copied into durable storage | Unreviewed local worktree output or credentials |

## What To Commit

Commit small, sanitized records when they directly support a durable claim:

- benchmark report JSON beside a standalone dated benchmark `README.md`
- Databricks run-status sidecars after secrets and raw response bodies are
  removed
- dependency freshness or legacy migration records under `evidence/`
- PR evidence sidecars under `pr-evidence/`
- generated release-bundle manifests only when they are intended as durable
  release handoff material

Every committed artifact should answer one question clearly: what claim does
this prove?

## What To Keep Out Of Git

Never commit credentials, Databricks tokens, OAuth material, raw Jobs API
responses, cluster logs, package wheels, generated datasets, `.env` files,
notebook checkpoints, or local temporary directories. Keep exploratory run
payloads and task status files under ignored `databricks-runs/` until a compact
sanitized record is promoted to `benchmarks/`, `evidence/`, `pr-evidence/`, or
a release bundle.

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
