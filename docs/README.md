# Project Documentation

This folder contains project-level planning and release-readiness documents for
Cachet. The wheel publishes as the Cachet-branded `cachet-kv` distribution,
while the repository and import surface are Cachet-branded. These files
complement the package API documentation in `README.md` and the module ownership
notes under `src/`.

- `v1-requirements-matrix.md` maps the generalized document KV-cache goal to
  current repository evidence and remaining V1 release gaps.
- `legacy-compatibility-removal.md` records the machine-checkable gate for
  removing the migration-only `restaurant_kv_serving` package and
  `restaurant-kv-*` console scripts after downstream jobs migrate.
- `../benchmarks/README.md` indexes the standalone, human-readable Databricks
  benchmark evidence folders.
