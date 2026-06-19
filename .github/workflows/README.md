# GitHub Workflows

This folder contains repository CI workflows.

- `ci.yml` runs the pull-request quality gate and repeats the same safety check
  on pushes to `main`: validate Poetry package metadata, dry-run dependency
  resolution for the base and optional dependency sets, install the package with
  test extras, run the full pytest suite, and build the source and wheel
  distributions.
