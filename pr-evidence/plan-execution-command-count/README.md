# Plan Execution Command Count

This PR tightens release-bundle validation for benchmark plan execution
sidecars. When `plan_source.command_count` is present, it must match the number
of command result records in the sidecar.

The regression test mutates a release-ready execution sidecar so the plan source
claims an extra command. Strict validation now rejects that mismatch, preventing
truncated or stale benchmark execution sidecars from looking complete.
