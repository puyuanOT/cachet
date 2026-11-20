# Project Documentation

This folder contains user, integrator, serving-maintainer, and release-operator
documentation for Cachet. The wheel publishes as the Cachet-branded `cachet-kv`
distribution, while the repository and import surface are Cachet-branded. Start
with the root `README.md` and the beginner docs before opening
release-operation material.

- `getting-started.md` is the beginner path for installing Cachet and running a
  local no-cloud example.
- `concepts.md` defines the KV-cache vocabulary used throughout the project.
- `adding-a-kv-method.md` is the method-author path for scaffolding, contracts,
  serving integration, tests, and benchmark publication gates.
- `production.md` explains the vLLM, SGLang, and managed-cloud deployment path.
- `release-ops/` keeps maintainer-only release machinery and governance records
  away from the first-touch user path.
- `v1-requirements-matrix.md` maps the generalized document KV-cache goal to
  current repository evidence and remaining V1 release gaps.
- `repo-map.md` is the human navigation map for source packages, benchmarks,
  maintainer evidence, Databricks templates, and ignored local run output.
- `evidence-policy.md` defines which machine-readable records belong in
  `benchmarks/`, `docs/release-ops/evidence/`, release bundles, or ignored
  `databricks-runs/` scratch space.
- `release-ops/evidence/dependency-freshness/current/README.md` explains the
  current direct dependency, serving-profile, and resolver-held transitive
  freshness evidence.
- `native-engine-integration.md` shows how Cachet handoff artifacts and
  launch-config sidecars connect to provider-backed vLLM and SGLang runtimes.
- `../benchmarks/README.md` is the human-facing research-style benchmark page
  for the published vLLM 0.30.0 campaign. Its sealed records live under
  `../benchmarks/appendix/`, and `../benchmarks/verify_sealed_records.py`
  checks their integrity; full re-validation runs at the
  `vllm-0300-campaign-archive` tag.
