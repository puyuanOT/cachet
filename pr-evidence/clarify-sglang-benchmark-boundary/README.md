# Clarify SGLang Benchmark Boundary PR Evidence

This folder records traceability evidence for PR #444, which clarifies the
human-facing Databricks benchmark surface so readers can distinguish published
vLLM latency/quality benchmark evidence from SGLang native HiCache probe and
connector-action integration evidence.

The JSON sidecar is machine-checkable `document_kv.pr_evidence.v1` evidence for
release-bundle traceability. It is not benchmark output; benchmark reports live
under `benchmarks/`.
