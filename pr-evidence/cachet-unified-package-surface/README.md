# Cachet Unified Package Surface Evidence

This directory contains the pull-request evidence sidecar for unifying the
Cachet repository and import surface while preserving `document_kv_cache`
compatibility.

Verification:

- `python -m pytest tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m pytest -q`
- `poetry build`
- installed-wheel import and `python -m cachet.benchmark_plan --help` smoke
- GitHub CI `Test and build`
- GPT-5.5 focused review with findings resolved
