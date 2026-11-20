# Evidence

This directory holds durable, machine-readable Cachet evidence that is not a
Databricks benchmark report and is not a pull-request traceability sidecar.

Use this boundary when adding evidence:

- `benchmarks/` is for human-readable benchmark summaries plus their sanitized
  benchmark JSON records.
- `../pr-evidence/` is for `document_kv.pr_evidence.v1` pull-request audit
  sidecars.
- `./` is for release-governance evidence with other record types, such
  as dependency freshness.

Current evidence families:

- [`dependency-freshness/`](dependency-freshness/) records direct package pins,
  isolated serving-profile pins, and resolver-held transitive drift.

Do not commit tokens, raw service responses, logs, wheels, generated datasets,
or local scratch output here.
