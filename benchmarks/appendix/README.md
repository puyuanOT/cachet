# Benchmark Appendix

This appendix holds three sealed result folders:

| Folder | Contents |
| --- | --- |
| [`vllm-0271-publication-v1/`](vllm-0271-publication-v1/) | The campaign's `campaign-report.json` and `benchmark-publication-gate.json`, and their rendered `README.md`. The campaign ran on vLLM 0.30.0; the folder name keeps the identifier of the frozen 0.27.1 design. |
| [`vllm-0300-quality-v2/`](vllm-0300-quality-v2/) | `quality-v2-report.json`, the paired answer-quality report that replaces the campaign's score table as the measure of answer quality. |
| [`vllm-0300-auxiliary-latency-rerun/`](vllm-0300-auxiliary-latency-rerun/) | `auxiliary-latency-rerun.json`, the plan-coherent re-run of the storage, BF16, and A10G latency contrasts. |

The campaign folder is valid only when it contains exactly the canonical
`README.md`, `campaign-report.json`, and `benchmark-publication-gate.json`, the
report/gate pair passes exact validation, and the governed tables in the
[benchmark root](../) byte-match the deterministic renderer. The two later
records are validated the same way against their own regions in the benchmark
root, and neither changes the campaign report or gate.

Superseded result folders are not retained as active or historical comparison
points. Do not commit raw Databricks responses, credentials, wheels, logs,
generated datasets, prompt payloads, or local scratch output. The report shape
and sanitization boundary are documented in
[`../_template/README.md`](../_template/README.md).
