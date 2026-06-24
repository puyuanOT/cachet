# Strict V1 Databricks Purpose Tests

This PR hardens release-bundle test coverage around strict Databricks evidence.
It verifies every required Databricks purpose failure, direct status-record
inputs, and the public plus legacy CLI help text for `--require-complete-v1`.

The fixture run names intentionally differ from the required purpose strings so
the tests prove strict mode reads `submit_payload.tasks[].purpose` rather than
run-level names.
