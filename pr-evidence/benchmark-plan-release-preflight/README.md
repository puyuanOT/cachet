# Benchmark Plan Release Preflight

This evidence records the PR that lets benchmark plans generate the
`document_kv.release_evidence_inputs.v1` release preflight sidecar with
`--release-preflight-output-json`.

The generated preflight command runs before release validation and is bundled
automatically when release-bundle assembly is enabled. Explicit
`--release-bundle-preflight-json` remains available for sidecars produced
outside the benchmark plan.

