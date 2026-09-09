# Benchmark Appendix

This appendix accepts at most one publication result folder for the frozen
vLLM 0.27.1 campaign:
`vllm-0271-publication-v1/`. Its absence means that no campaign result is
published. Its presence is valid only when it contains exactly the canonical
`README.md`, `campaign-report.json`, and `benchmark-publication-gate.json`, the
report/gate pair passes exact validation, and the governed tables in the
[benchmark root](../) byte-match the deterministic renderer.

Superseded result folders are not retained as active or historical comparison
points. Do not commit raw Databricks responses, credentials, wheels, logs,
generated datasets, prompt payloads, or local scratch output. The report shape
and sanitization boundary are documented in
[`../_template/README.md`](../_template/README.md).
