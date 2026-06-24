# Project Documentation

This folder contains project-level planning and release-readiness documents for
Cachet. The wheel publishes as the Cachet-branded `cachet-kv` distribution,
while the repository and import surface are Cachet-branded. These files
complement the package API documentation in `README.md` and the module ownership
notes under `src/`.

- `v1-requirements-matrix.md` maps the generalized document KV-cache goal to
  current repository evidence and remaining V1 release gaps.
- `repo-map.md` is the human navigation map for source packages, benchmarks,
  durable evidence, PR traceability, Databricks templates, and ignored local
  run output.
- `evidence-policy.md` defines which machine-readable records belong in
  `benchmarks/`, `evidence/`, `pr-evidence/`, release bundles, or ignored
  `databricks-runs/` scratch space.
- `legacy-compatibility-removal.md` records the machine-checkable proof that
  the source-only `restaurant_kv_serving` compatibility directory was removed;
  the current downstream migration evidence lives under
  `../evidence/legacy-migration/current/`.
- `../evidence/dependency-freshness/current/README.md` explains the current
  direct dependency, serving-profile, and resolver-held transitive freshness
  evidence.
- `native-engine-integration.md` shows how Cachet handoff artifacts and
  launch-config sidecars connect to provider-backed vLLM and SGLang runtimes.
- `../benchmarks/current/README.md` is the human-facing current benchmark
  index; `../benchmarks/README.md` indexes the standalone, human-readable
  benchmark report folders and their Databricks source-artifact records.
