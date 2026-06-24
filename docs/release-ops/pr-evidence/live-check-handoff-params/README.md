# Live Check Handoff Params PR Evidence

This folder records traceability evidence for PR #445, which adds validated
Cachet handoff parameters to the OpenAI-compatible live endpoint smoke path.
That lets future vLLM/SGLang live checks send real `kv_transfer_params` on the
cache arm instead of only labeling the request as cache-enabled.

The JSON sidecar is machine-checkable `document_kv.pr_evidence.v1` evidence for
release-bundle traceability. It is not benchmark output; benchmark reports live
under `benchmarks/`.
