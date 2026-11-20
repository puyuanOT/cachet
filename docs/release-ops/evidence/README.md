# Evidence

This directory holds durable, machine-readable Cachet release-governance
evidence that is not a benchmark report.

Use this boundary when adding evidence:

- `benchmarks/` is for human-readable benchmark summaries plus their sanitized
  benchmark JSON records.
- `./` is for release-governance records such as dependency freshness.
- Per-release sidecars such as PR evidence, GitHub governance, and repository
  hygiene are generated at bundle time and travel in the release bundle; they
  are not committed here.

Current evidence families:

- [`dependency-freshness/`](dependency-freshness/) records direct package pins,
  isolated serving-profile pins, and resolver-held transitive drift.

Do not commit tokens, raw service responses, logs, wheels, generated datasets,
or local scratch output here.
