# Public UX Pass Evidence

This folder stores maintainer traceability for the public-facing Cachet repo UX
pass. The public user path is the root `README.md`, `docs/getting-started.md`,
`docs/concepts.md`, `docs/production.md`, and `examples/quickstart_local.py`.

The sidecar records the reviewer findings and the fixes:

- new public files are tracked with the PR;
- installed users can run `python -m cachet.quickstart_local`;
- release-ops docs classify stable user commands separately from maintainer and
  compatibility CLI surfaces.

## Verification

- `PYTHONPATH=src python -m cachet.quickstart_local`
- `python examples/quickstart_local.py`
- `poetry build -f wheel`, install the built wheel in a fresh venv, then
  `python -m cachet.quickstart_local`
- `poetry run pytest tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `poetry check --lock`
- `git diff --check`
- changed-file secret scan
