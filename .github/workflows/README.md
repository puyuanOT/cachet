# GitHub Workflows

This folder contains repository CI workflows.

- `ci.yml` runs the pull-request quality gate and repeats the same safety check
  on pushes to `main`: validate Poetry package metadata, dry-run dependency
  resolution for the base and optional dependency sets, install the package with
  test extras, verify every installed console script from `[project.scripts]`
  responds to `--help`, run the full pytest suite, verify a clean PEP 517 wheel
  build with Cachet metadata and entry points, build the source and wheel
  distributions with Poetry, verify the built wheel metadata, then install the
  built wheel into a fresh venv and smoke-test the `cachet`,
  `document_kv_cache`, and `restaurant_kv_serving` import namespaces plus a
  Cachet CLI alias.
