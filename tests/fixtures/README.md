# Test Fixtures

This directory contains immutable compatibility fixtures used by contract tests.

- `engine_adapter_handoff_v2.json` is a literal legacy handoff record. Tests use
  it only to prove that schema-v2 raw-KV records require explicit compatibility
  opt-in; new writers emit the current strict schema.
