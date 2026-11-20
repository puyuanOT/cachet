# Development

Maintainer reference for dependency pins, isolated serving runtimes,
dependency freshness evidence, CI, and the repository rules that tests
enforce. Day-to-day commands are in [`AGENTS.md`](../../../AGENTS.md) and
[`CONTRIBUTING.md`](../../../CONTRIBUTING.md).

## Dependencies

- `pyproject.toml` uses PEP 621 metadata with exact direct pins: Python
  `>=3.11,<4.0`, build backend `poetry-core==2.4.1`, runtime `packaging==26.3`,
  the `databricks` extra (`pyspark==4.1.2`, `databricks-sdk==0.118.0`), the
  `test` extra (`pytest==9.1.1`), and the Poetry dev group (`mypy==2.2.0`,
  `ruff==0.15.21`).
- The committed `poetry.lock` records resolver output for the base package and
  both extras; CI validates it with `poetry check --lock`.
- vLLM and SGLang are not Poetry extras because their dependency graphs
  conflict in one resolver. The vendored `vllm_kv_injection` and
  `sglang_kv_injection` adapters ship in the Cachet wheel; install exactly one
  serving runtime per isolated environment.
- `document_kv_cache.serving_env` records the exact vLLM and SGLang profiles
  that smoke and probe jobs share. `cachet-vllm-smoke` installs the pinned
  official vLLM `0.30.0+cu129` wheel and the profile's exact pins from the
  CUDA 12.9 indexes. Release-safe provider-backed vLLM probes install the
  E5M2-patched wheel built by `vllm_wheel_repack.py` plus the packaged hash
  lock (see [`databricks.md`](databricks.md#engine-probe-jobs)).

## Dependency Freshness

Current freshness evidence lives under
`docs/release-ops/evidence/dependency-freshness/current/` and is produced by
`cachet-dependency-freshness`. Direct metadata pins must match the supplied
latest stable versions; non-latest serving-profile pins and resolver-held
transitive packages need an explicit compatibility or Databricks-validation
reason (`--allow-runtime-pin`, `--allow-transitive-outdated`) before the
evidence passes. Strict release bundles require this sidecar.

## CI

`.github/workflows/ci.yml` runs on Python 3.11 and 3.12, and the aggregate
`Test and build` job is the required branch-protection check. In order it
runs `poetry check`, `poetry check --lock`, dry-run installs of the base and
extras dependency sets, `poetry install -E test`, the CPU `torch==2.12.1`
install, `ruff check` and `mypy` on an explicit file list, `--help` for every
`[project.scripts]` entry, `poetry run pytest -q`, a PEP 517 wheel build,
`poetry build`, built-wheel metadata checks, and a clean-venv import smoke.

- Ruff and mypy (strict) cover only the files listed in `ci.yml`. Keep a module
  on that list clean when you edit it, and add new typed contract modules to
  both steps.
- Torch-dependent tests skip when torch is missing; install the CPU wheel
  locally to run them.

## Repository Rules

- Implementation lives in `src/document_kv_cache/`. `src/cachet/` holds one
  re-export alias module per public module, `cachet.__all__` mirrors
  `document_kv_cache.__all__`, and `src/cachet/__init__.pyi` must re-export
  exactly those names.
- Register every command in `[project.scripts]` under both a `cachet-*` name
  and a `document-kv-*` alias, list it in the
  [release-ops command table](../README.md#cli-audience-boundary), and prefer
  `cachet-*` in documentation.
- `databricks/**/databricks.yml` must stay byte-identical to the packaged
  copies under `src/document_kv_cache/templates/databricks/`.
- Every Git-visible directory needs a `README.md` or a package docstring.
- `tests/test_project_governance.py` pins exact phrases in the README,
  `CONTRIBUTING.md`, and these maintainer docs. When moving documented
  content, point the assertion at the new file instead of deleting it.
- Sealed records under `benchmarks/` are integrity-checked by
  `python benchmarks/verify_sealed_records.py`; do not edit them by hand.

## License

Cachet is Apache-2.0. `LICENSE` holds the full text, the package metadata
declares the SPDX expression and ships the license file in wheels and
sdists, and release bundles require the `cachet` and `document_kv_cache`
`py.typed` markers in the wheel.
