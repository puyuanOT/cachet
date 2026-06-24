# Native Engine Integration

Cachet integrates with established serving engines at their native KV-transfer
boundary. The package generates document KV handoff metadata, payloads, and
launch-config sidecars; vLLM and SGLang still own scheduling, allocation,
decode, routing, and cleanup.

Use this guide when wiring the current provider-backed vLLM or SGLang paths into
a deployment. For benchmark evidence, start with the standalone folders under
`benchmarks/vllm/`, `benchmarks/sglang/`, `benchmarks/storage/`, and
`benchmarks/native-engine/`; `benchmarks/databricks/` mirrors the Databricks
status and release-source records for audit.

## Launch Configs

Generate one launch-config sidecar per engine. These sidecars are also accepted
by strict release-bundle validation.

```bash
cachet-engine-launch-config build-vllm \
  --payload-cache-max-bytes 8589934592 \
  --output-json /data/engine-launch/vllm.json

cachet-engine-launch-config build-sglang \
  --output-json /data/engine-launch/sglang.json
```

The vLLM sidecar selects `DocumentKVConnector` from
`vllm_kv_injection.vllm_dynamic_connector` and the native provider factory
`vllm_kv_injection.vllm_native_provider:build_document_kv_provider`. The SGLang
sidecar selects the dynamic HiCache `DocumentKVHiCacheBackend` and the provider
factory
`sglang_kv_injection.sglang_dynamic_backend:build_document_kv_hicache_provider`.

Validate launch configs in Python before handing them to an engine wrapper or a
managed benchmark plan:

```python
import json

from document_kv_cache.engine_launch_config import (
    DEFAULT_SGLANG_DOCUMENT_KV_PROVIDER_FACTORY,
    DEFAULT_VLLM_DOCUMENT_KV_PROVIDER_FACTORY,
    build_sglang_launch_config,
    build_vllm_launch_config,
    validate_engine_launch_config_record,
)

vllm_launch_config = build_vllm_launch_config(
    payload_cache_max_bytes=8 * 1024**3,
)
sglang_launch_config = build_sglang_launch_config()

validate_engine_launch_config_record(vllm_launch_config, expected_backend="vllm")
validate_engine_launch_config_record(sglang_launch_config, expected_backend="sglang")

assert vllm_launch_config["kv_connector"] == "DocumentKVConnector"
assert (
    vllm_launch_config["kv_connector_extra_config"]["document_kv.provider_factory"]
    == DEFAULT_VLLM_DOCUMENT_KV_PROVIDER_FACTORY
)

sglang_extra = json.loads(sglang_launch_config["hicache_storage_backend_extra_config"])
assert sglang_extra["class_name"] == "DocumentKVHiCacheBackend"
assert sglang_extra["document_kv.provider_factory"] == DEFAULT_SGLANG_DOCUMENT_KV_PROVIDER_FACTORY
```

Do not use no-op providers, in-memory test connectors, or debug probe factories
as release evidence. Strict probes and release bundles must carry provider
factory metadata for the native runtime path.

## Request Flow

1. Generate or read Cachet handoff bundles for the documents that should be
   reused. Benchmark jobs usually use `cachet-benchmark-handoff-bundles` or the
   `benchmark_plan` handoff generation flags.
2. Start vLLM or SGLang through the engine's normal deployment mechanism, using
   the launch-config sidecar for that engine.
3. For the vLLM native provider path, send OpenAI-compatible requests with the
   full logical prompt and `kv_transfer_params` on the cache arm. Native vLLM
   scheduling needs the logical prefix token positions before it can claim
   external matched tokens and allocate runtime KV blocks.
4. Keep the payload URI and handoff record stable for the request id. Cachet's
   native provider uses those fields to load assembled KV into the engine's
   allocated block mappings.

For SGLang, the current evidence covers the runtime-facing dynamic HiCache
provider, launch config, preflight, native probe, and connector action
descriptors. Validate live decode-time prefix binding in the target SGLang
deployment before treating it as benchmark evidence. The SGLang smoke helper can
prepare that readiness run with `--generate-live-handoff`, which creates the
live synthetic Cachet handoff and matching SGLang HiCache page-key metadata
inside the same isolated runtime used to launch SGLang.

Cachet does not run a proprietary request scheduler. If an integration needs a
different batching, decode, routing, or cleanup policy, implement it inside the
serving engine or the engine adapter package rather than inside Cachet.

## Databricks Preflight

For QA, target the default AWS g6/L4 hardware target: `aws-g6-l4` on
`g6.8xlarge`. The explicit `aws-g5-a10g` hardware target on `g5.8xlarge` is
compatibility evidence only.

Dry-run the generated payload before submitting a GPU job:

```bash
cachet-databricks-runs \
  --output-json databricks-runs/managed-plan/databricks-run-submit-summary.json \
  payload-summary \
  --payload-json databricks-runs/managed-plan/databricks-run-submit.json \
  --expected-hardware-target aws-g6-l4 \
  --expected-node-type-id g6.8xlarge
```

Credentials must stay in Databricks profiles or environment variables. Do not
commit tokens, raw Jobs API responses, task logs, package wheels, generated
dataset payloads, or local `databricks-runs/` output.

## Evidence

The current provider-backed evidence is tracked in:

- `benchmarks/vllm/2026-06-23-g6-l4-v1/`
- `benchmarks/vllm/2026-06-23-g5-a10g-v1-compatibility/`
- `benchmarks/storage/2026-06-21-g6-l4-storage-readers/`
- `benchmarks/native-engine/2026-06-23-g6-l4-native-engine-probes/`

Refresh those standalone folders, the matching `benchmarks/databricks/` mirrors,
the strict release bundle, and `docs/v1-requirements-matrix.md` whenever
benchmark code, runtime pins, connector contracts, launch-config fields, or
package wheel identity change.
