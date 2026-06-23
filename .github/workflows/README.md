# GitHub Workflows

This folder contains repository CI workflows.

- `ci.yml` runs the pull-request quality gate and repeats the same safety check
  on pushes to `main`: validate Poetry package metadata and the committed
  lockfile, dry-run dependency resolution for the base and optional dependency
  sets, install the package with test extras, verify every installed console
  script from `[project.scripts]` responds to `--help`, run the full pytest
  suite, verify a clean PEP 517 wheel
  build with Cachet metadata and entry points, build the source and wheel
  distributions with Poetry, verify the built wheel metadata, then install the
  built wheel into a fresh venv and smoke-test the `cachet` and
  `document_kv_cache` import namespaces, verify the legacy restaurant facade is
  absent from installed wheels, and exercise a Cachet CLI alias.
