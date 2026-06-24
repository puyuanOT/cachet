# CI Console Script Entrypoint Guard

This PR makes CI verify the installed command-line surface. After
`poetry install -E test`, the workflow reads `[project.scripts]` from
`pyproject.toml`, resolves each console script in the Poetry environment, and
runs `--help` with `check=True`.

The guard protects the documented open-source UX: if a script is renamed,
misconfigured, or imports a broken module at startup, CI fails before the
package is released.

Verification:

- `poetry install -E test` plus the console-script probe
- `poetry run pytest tests/test_project_governance.py tests/test_public_package.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry build`

GPT-5.5 review approved the diff with no findings and independently reran the
console-script probe.
