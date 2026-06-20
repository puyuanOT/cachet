# Engine Submit Handoff

This PR adds `prepare_and_submit_to_engine(...)` helpers on the service and
workflow APIs. The helpers prepare a validated `EngineReadyRequest`, submit it
to a caller-provided `ServingEngineConnector`, and return the submitted request
for observability.

## Verification

- `poetry run pytest tests/test_engine.py tests/test_workflow.py tests/test_project_governance.py tests/test_public_package.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry install --dry-run`
- `poetry build`

## Review

GPT-5.5 approved with no findings and confirmed that the helper preserves the
serving boundary: connector submit failures stay with the connector, and the
package does not add custom scheduling or decode logic.
