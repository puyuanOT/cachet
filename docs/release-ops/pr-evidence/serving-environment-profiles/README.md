# Serving Environment Profiles PR Evidence

This folder records PR evidence for the slice that centralized serving-engine
install metadata into `serving_env.py`. The profile module keeps vLLM and SGLang
out of the core Poetry resolver while giving Databricks smoke/probe jobs a
shared one-engine-per-environment install contract.

Local verification covers the package, wrapper API, metadata serialization, and
non-engine transitive pins. Full engine package installation still belongs to
Databricks/GPU smoke jobs because local macOS and cross-platform pip dry-runs do
not faithfully resolve the CUDA serving stack.
