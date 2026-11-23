# Release Notes

## 0.3.0

Cachet 0.3.0 moves the serving profile to vLLM 0.30.0 and makes Quality
Benchmark v2 the only answer-quality protocol.

Highlights:

- vLLM 0.30.0 (CUDA 12.9) serving profile with a hash-locked runtime shipped in
  the wheel; vLLM and SGLang handoffs carry registered artifact and token
  contracts;
- registered method framework with N-way evaluation arms, paired statistics,
  benchmark gates, a method scaffold, and a conformance checker;
- Quality v2 prompting, decoding, parsing, and scoring on every benchmark path,
  with NIAH staged as the natural-paragraph grid;
- sealed vLLM 0.30.0 campaign and Quality v2 records in `benchmarks/`, checked
  by `benchmarks/verify_sealed_records.py`;
- strict release bundles that re-verify the publication benchmark gate and
  record which model reviewed the release PR.

Breaking changes:

- the `document-kv-*` console scripts are removed; use the `cachet-*` names;
- the v1 prompt, scorers, and synthetic NIAH generator are removed, and the new
  prompt template changes artifact ids and cache keys, so handoffs generated
  under the v1 template must be regenerated;
- legacy modules, readers for older handoff and sidecar formats, and the
  `gpt55_review_*` PR-evidence fields are removed; see `CHANGELOG.md`.

Known limitations:

- the serving profile is pinned to vLLM 0.30.0 and the validated SGLang
  provider profile; newer engine releases need a fresh Databricks validation;
- APIs are alpha and may change before a stable 1.0 release.

## 0.2.0

Cachet 0.2.0 is the first public-facing alpha release candidate for document
KV-cache orchestration.

Highlights:

- install with `pip install cachet-kv`;
- import from `cachet`;
- run `python -m cachet.quickstart_local` or
  `python examples/quickstart_local.py` without cloud or GPU access;
- use memory/disk storage locally and advanced vLLM/SGLang adapters in serving
  environments;
- read current benchmark summaries from `benchmarks/README.md`.

Known limitations:

- local quickstart payloads are toy bytes, not real model KV tensors;
- production serving still requires a real model-aware generator and serving
  engine integration;
- APIs are alpha and may change before a stable 1.0 release.
