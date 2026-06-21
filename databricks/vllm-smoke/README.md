# vLLM Smoke Bundle

This folder contains a standalone Databricks Asset Bundle for the smallest
runtime check: starting Qwen3 through vLLM on a single AWS g6/L4 node and running
the built-in V1 smoke examples.

Use this bundle before the full benchmark-plan bundle when you only want to
verify that the pinned vLLM stack, model download, GPU runtime, and output
artifact path work in a target workspace.

For release evidence, use the same runner with prepared V1 JSONL files instead
of the built-in smoke records. Pass all four dataset flags (`biography`,
`hotpotqa`, `musique`, and `niah`) together with the desired server sizing, for
example `--max-model-len 32768 --max-num-seqs 1 --gpu-memory-utilization 0.9`.
Prepared mode writes `dataset_source=prepared` and the sizing values into
`metadata.json`, making long-context benchmark artifacts distinguishable from
the tiny runtime smoke.
