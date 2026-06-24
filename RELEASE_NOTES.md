# Release Notes

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
- read current benchmark summaries from `benchmarks/current/README.md`.

Known limitations:

- local quickstart payloads are toy bytes, not real model KV tensors;
- production serving still requires a real model-aware generator and serving
  engine integration;
- APIs are alpha and may change before a stable 1.0 release.
