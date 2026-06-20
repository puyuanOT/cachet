# Qwen3 Instruct Profile Alias Evidence

This evidence supports the PR that makes the V1 Qwen3 Instruct Hugging Face
model id explicit in Cachet model profile metadata, aliases, and the vLLM smoke
helper.

The branch keeps the existing base `Qwen/Qwen3-4B` profile alias for
compatibility while making `Qwen/Qwen3-4B-Instruct-2507` the canonical HF model
id for the `qwen3:4b-instruct` profile used by V1 AWS g5 release evidence.
