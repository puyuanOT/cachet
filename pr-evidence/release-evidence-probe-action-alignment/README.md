# Release Evidence Probe/Action Alignment

This PR-evidence sidecar covers the release-readiness validation slice that
tightens native engine probe and connector-action alignment.

The slice makes strict V1 release evidence reject vLLM/SGLang sidecar pairs that
disagree on inferred payload mode or the exact KV layout record, so a native
probe cannot audit one handoff while the connector-action artifact describes a
different payload shape or layout.
