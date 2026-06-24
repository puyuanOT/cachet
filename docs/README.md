# Project Documentation

This folder contains user, integrator, serving-maintainer, and release-operator
documentation for Cachet. The wheel publishes as the Cachet-branded `cachet-kv`
distribution, while the repository and import surface are Cachet-branded. Start
with the root `README.md` and the beginner docs before opening
release-operation material.

- `getting-started.md` is the beginner path for installing Cachet and running a
  local no-cloud example.
- `concepts.md` defines the KV-cache vocabulary used throughout the project.
- `production.md` explains the vLLM, SGLang, and managed-cloud deployment path.
- `release-ops/` keeps maintainer-only release machinery, audit records, and
  historical detailed references away from the first-touch user path.
- `v1-requirements-matrix.md` maps the generalized document KV-cache goal to
  current repository evidence and remaining V1 release gaps.
- `repo-map.md` is the human navigation map for source packages, benchmarks,
  maintainer evidence, Databricks templates, and ignored local run output.
- `evidence-policy.md` defines which machine-readable records belong in
  `benchmarks/`, `docs/release-ops/evidence/`,
  `docs/release-ops/pr-evidence/`, release bundles, or ignored
  `databricks-runs/` scratch space.
- `legacy-compatibility-removal.md` records the machine-checkable proof that
  the source-only `restaurant_kv_serving` compatibility directory was removed;
  the current downstream migration evidence lives under
  `release-ops/evidence/legacy-migration/current/`.
- `release-ops/evidence/dependency-freshness/current/README.md` explains the
  current direct dependency, serving-profile, and resolver-held transitive
  freshness evidence.
- `native-engine-integration.md` shows how Cachet handoff artifacts and
  launch-config sidecars connect to provider-backed vLLM and SGLang runtimes.
- `../benchmarks/current/README.md` is the human-facing current benchmark
  index; `../benchmarks/README.md` indexes the standalone, human-readable
  benchmark report folders and their Databricks source-artifact records.
