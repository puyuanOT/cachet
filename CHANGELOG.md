# Changelog

## Unreleased

- vLLM 0.30.0 (CUDA 12.9) serving runtime with pinned, hash-locked runtime
  lock files shipped in the wheel.
- Registered method framework: method lifecycles, artifact, token, and layout
  contracts, N-way evaluation arms, paired statistics, and a method scaffold
  and conformance checker.
- Vanilla v2: per-document pre-RoPE keys with absolute-position injection, and
  a single-pass segmented vLLM cold loader.
- Governed Databricks publication pipeline: resource ledger, GPU
  qualification, latency, full-score, and finalizer stages, plus a source
  freeze.
- New `cachet.*` facades and `cachet-*` console scripts, each with a
  `document-kv-*` twin.
- `packaging` pinned to 26.3.
- Benchmark records: the vLLM 0.30.0 campaign report, the quality v2 report,
  and the plan-coherent auxiliary latency re-run.

## 0.2.0

- Cachet-branded `cachet-kv` distribution and `cachet` import facade.
- Document-oriented request, manifest, storage, materialization, and workflow
  APIs.
- Local memory and disk storage readers plus Unity Catalog Volume support.
- vLLM and SGLang adapter modules shipped in the same distribution.
- Human-readable benchmark summaries in `benchmarks/README.md`.
- Public quickstart and beginner documentation.
