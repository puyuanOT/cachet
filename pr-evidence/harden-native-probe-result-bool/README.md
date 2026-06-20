# Harden Native Probe Result Bool

This PR requires `native_probe` to be an actual boolean on engine probe
configuration and factory-result dataclasses in both the document-owned API and
the legacy compatibility namespace.

## Verification

- `poetry run pytest tests/test_engine_probe.py::test_engine_probe_native_probe_flags_must_be_boolean -q`
- `poetry run pytest tests/test_engine_probe.py -q`
- `poetry run pytest -q`
- `poetry build`

## Review

GPT-5.5 approved the focused bool-hardening diff with no requested changes.
