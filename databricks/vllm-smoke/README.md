# vLLM Smoke Bundle

This folder contains a standalone Databricks Asset Bundle for the smallest
runtime check: starting Qwen3 through vLLM on a single AWS g5/g6 node and running
the built-in V1 smoke examples.

Use this bundle before the full benchmark-plan bundle when you only want to
verify that the pinned vLLM stack, model download, GPU runtime, and output
artifact path work in a target workspace.
