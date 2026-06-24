# Cachet Vendor Engine Adapters Evidence

This directory contains the pull-request evidence sidecar for vendoring the
vLLM and SGLang adapter packages into the Cachet repository while preserving
their compatibility import paths.

Verification:

- `poetry install --dry-run`
- targeted public/governance and adapter test slices
- `python -m pytest -q`
- `poetry build`
- installed-wheel adapter import smoke
- GitHub CI `Test and build`
- GPT-5.5 focused review with findings resolved
