# Changelog

## Unreleased

- vLLM 0.30.0 (CUDA 12.9) serving profile, with the pinned hash-locked vLLM
  runtime lock shipped in the wheel, and `vllm_wheel_repack` for building the
  E5M2-patched vLLM wheel.
- Registered method framework: method lifecycles, artifact, token, and layout
  contracts, N-way evaluation arms, paired statistics, benchmark gates and
  metrics, Quality v2 scoring, runtime telemetry, and a method scaffold and
  conformance checker (`cachet-method-scaffold`, `cachet-method-conformance`).
- Vanilla v2: per-document pre-RoPE keys with absolute-position injection, and
  a single-pass segmented vLLM cold loader.
- Decode throughput is now aggregated as total completion tokens over total
  post-TTFT decode time; the method `vanilla` is renamed `vanilla_prefill`, and
  connector actions take an explicit reuse plan.
- The packaged vLLM Databricks smoke runner installs the official vLLM
  `0.30.0+cu129` wheel from the CUDA 12.9 indexes, and both the vLLM and SGLang
  smoke runners emit handoffs that carry the registered artifact and token
  contracts. The vLLM runner passes the registered cache arm that those
  handoffs declare. On 2026-11-18 the
  vLLM runner passed on L4 and A10G and the SGLang runner passed on A10G.
- `packaging` pinned to 26.3.
- Benchmark records: the vLLM 0.30.0 campaign report and gate, the Quality v2
  report, and the plan-coherent auxiliary latency re-run, plus
  `benchmarks/verify_sealed_records.py`, a dependency-free integrity check of
  those records and the README regions rendered from them.
- The campaign machinery that produced the records (resource ledger, GPU
  qualification, latency, full-score, and finalizer stages, source freeze,
  LMCache/hybrid launchers, and the full offline validators) is not part of
  the package; it is preserved at the `vllm-0300-campaign-archive` tag.
- Removed the superseded `benchmarks/current/` summary, the earlier result
  folders under `benchmarks/{databricks,native-engine,sglang,storage,vllm}/`
  (their index READMEs remain), and six PR-evidence folders for superseded
  benchmark results under `docs/release-ops/pr-evidence/`.
- Sanitized benchmark evidence keeps the final-answer parser identity, status,
  and NIAH cell id, and records a SHA-256 of the extracted answer instead of
  the answer, so scored results can pass the publication gate after
  sanitization.
- Scorer manifests can declare a non-v1 final-answer parser (the Quality v2
  scorers do), and the publication gate checks measurements against the
  declared parser. Manifests for v1 scorers are unchanged.
- An integral `rope_theta` now yields the same artifact id as the equal float,
  and reuse-plan and runtime-operation digests reject NaN and Infinity.

## 0.2.0

- Cachet-branded `cachet-kv` distribution and `cachet` import facade.
- Document-oriented request, manifest, storage, materialization, and workflow
  APIs.
- Local memory and disk storage readers plus Unity Catalog Volume support.
- vLLM and SGLang adapter modules shipped in the same distribution.
- Human-readable benchmark summaries in `benchmarks/README.md`.
- Public quickstart and beginner documentation.
