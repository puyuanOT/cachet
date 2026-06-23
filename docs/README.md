# Project Documentation

This folder contains project-level planning and release-readiness documents for
Cachet. The wheel publishes as the Cachet-branded `cachet-kv` distribution,
while the repository and import surface are Cachet-branded. These files
complement the package API documentation in `README.md` and the module ownership
notes under `src/`.

- `v1-requirements-matrix.md` maps the generalized document KV-cache goal to
  current repository evidence and remaining V1 release gaps.
- `legacy-compatibility-removal.md` records the machine-checkable gate for
  deleting the remaining source-only `restaurant_kv_serving` compatibility
  directory after local compatibility tests migrate; the current downstream
  migration evidence lives under
  `../evidence/legacy-migration/current/`.
- `native-engine-integration.md` shows how Cachet handoff artifacts and
  launch-config sidecars connect to provider-backed vLLM and SGLang runtimes.
- `../benchmarks/README.md` indexes the standalone, human-readable Databricks
  benchmark evidence folders.
