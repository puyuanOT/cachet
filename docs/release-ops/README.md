# Release Operations

This folder is for maintainers and release operators. New users should start
with the root [`README.md`](../../README.md), [`../getting-started.md`](../getting-started.md),
and [`../concepts.md`](../concepts.md).

Release-ops material includes:

- strict release-bundle assembly and validation;
- PR traceability sidecars;
- repository hygiene and GitHub governance records;
- Databricks benchmark/job publication details;
- dependency freshness and legacy migration evidence;
- historical maintainer reference material moved out of the public README.

## Maintainer References

| Path | Purpose |
| --- | --- |
| [`maintainer-reference.md`](maintainer-reference.md) | Historical detailed root README content, including benchmark, Databricks, release-bundle, and governance workflows |
| [`pr-evidence/`](pr-evidence/) | Machine-readable PR traceability sidecars |
| [`evidence/`](evidence/) | Durable non-benchmark release-governance evidence |
| [`maintainer-release-checklist.md`](maintainer-release-checklist.md) | Human checklist for internal release gates |

## CLI Audience Boundary

Stable user commands documented in the beginner path:

- `python -m cachet.quickstart_local`
- `cachet-engine-launch-config`

Maintainer and serving-operator commands:

| Group | Commands |
| --- | --- |
| Benchmark planning and execution | `cachet-benchmark-plan`, `cachet-benchmark-handoffs`, `cachet-benchmark-handoff-manifest`, `cachet-benchmark-handoff-bundles`, `cachet-run-benchmark-plan`, `cachet-storage-benchmark` |
| Databricks job generation and run inspection | `cachet-databricks-job`, `cachet-databricks-runs`, `cachet-storage-benchmark-databricks-job`, `cachet-engine-probe-databricks-job`, `cachet-vllm-smoke-databricks-job`, `cachet-sglang-smoke-databricks-job` |
| Native probes and serving smoke checks | `cachet-native-probe-scaffold`, `cachet-serving-env`, `cachet-native-probe-factories`, `cachet-engine-probe`, `cachet-engine-probe-fixture`, `cachet-runtime-kv-offload-probe`, `cachet-vllm-runtime-preflight`, `cachet-sglang-runtime-preflight`, `cachet-vllm-smoke`, `cachet-sglang-smoke` |
| Release and governance | `cachet-release-evidence`, `cachet-release-bundle`, `cachet-pr-evidence`, `cachet-dependency-freshness`, `cachet-github-governance`, `cachet-repository-hygiene` |
| Template extraction | `cachet-templates` |

The `document-kv-*` console scripts are compatibility aliases for the same
command families. Prefer `cachet-*` in new documentation, and keep both alias
families out of the first-touch user path unless the command is explicitly
stable for users.

Keep credentials, raw service responses, generated wheels, logs, and local run
scratch output out of Git. Use ignored `databricks-runs/` for local work and
promote only compact sanitized records.
