# GitHub Workflows

This folder contains the repository CI workflow. `ci.yml` runs for every pull
request and for pushes to `main`; duplicate runs for the same ref are cancelled.

## Test Matrix

The `test` job runs independently on Python 3.11 and 3.12. Each matrix job:

1. installs Poetry 2.4.1;
2. validates package metadata and the committed lockfile with `poetry check`
   and `poetry check --lock`;
3. dry-runs both the base dependency set and the `databricks` + `test` optional
   dependency set;
4. installs the package with test extras and installs the CPU-only PyTorch
   2.12.1 contract dependency from the PyTorch CPU index;
5. runs focused Ruff and strict mypy checks over the method, artifact,
   benchmark manifest/runner/statistics, publication-gate, representative-input
   preparation, canary,
   Databricks resource-guard, scaffold, conformance, and reference-method
   contracts listed in the workflow;
6. verifies that every `[project.scripts]` console entry point is installed and
   responds to `--help`, then runs the full pytest suite;
7. builds a clean PEP 517 wheel and validates its metadata and entry points;
8. builds the source and wheel distributions with Poetry and validates the
   wheel metadata again; and
9. installs the built wheel into a fresh virtual environment, checks the
   `cachet` and `document_kv_cache` import surfaces, verifies that the removed
   legacy facade is absent, and exercises a packaged Cachet CLI.

Ruff/mypy targets are intentionally explicit. When framework integration files
change, add them only after the exact strict commands pass in a clean Python
3.11 and 3.12 environment; an imported-module failure is still a CI failure.

## Required Gate

The `gate` job is named `Test and build`, runs even when a matrix job fails or
is cancelled, and succeeds only when every Python matrix job succeeds. Branch
protection requires this aggregate check, so a passing job for only one Python
version cannot merge a pull request.

Together, the matrix verifies a clean PEP 517 wheel build with Cachet metadata,
uses Poetry to verify the built wheel metadata, installs the built wheel into a
fresh venv, and smoke-tests both `cachet` and `document_kv_cache`.
