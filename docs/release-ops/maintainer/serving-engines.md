# Serving Engines

Maintainer reference for the vLLM/SGLang boundary: engine-ready requests,
adapter handoff records, connector actions, native probes, launch configs, and
live smoke checks. Deployment wiring is in
[`../../native-engine-integration.md`](../../native-engine-integration.md); the
vendored adapters are described in
[`src/vllm_kv_injection/README.md`](../../../src/vllm_kv_injection/README.md) and
[`src/sglang_kv_injection/README.md`](../../../src/sglang_kv_injection/README.md).
Submitting probes and smokes as managed jobs is covered in
[`databricks.md`](databricks.md).

Cachet stops at the engine handoff boundary and does not implement an
alternative serving engine. The adapter is a thin layer inside the vLLM or
SGLang process: it translates connector actions into native block-manager
calls, binds the handle to the request, and leaves scheduling, decode, LoRA
routing, and release to the engine.

## Serving Engine Handoff

`DocumentKVService.prepare_for_engine` turns a document request into an `EngineReadyRequest`:

```python
ready = service.prepare_for_engine(
    request,
    layout=qwen3_layout,
    metadata={"engine": "vllm"},
    adapter_ids=("selection-lora",),
)
```

If the integration already has a vLLM or SGLang connector object implementing
the small `ServingEngineConnector` protocol, use
`prepare_and_submit_to_engine(...)` to keep preparation and handoff together:

```python
ready = workflow.prepare_and_submit_to_engine(
    request,
    connector=vllm_connector,
    layout=qwen3_layout,
    metadata={"engine": "vllm"},
    adapter_ids=("selection-lora",),
    segmented=True,
)
```

The `EngineReadyRequest` contains the loaded payload, per-segment cache-tier
attribution, and a validated `KVCacheHandle` with contiguous token and byte
segments. Engine integrations consume that handle, reserve engine-native KV
blocks, and inject or map the payload into their existing schedulers.

For engine-specific integration code, wrap the ready request in an adapter
contract:

```python
from pathlib import Path

from document_kv_cache import (
    build_engine_adapter_request,
    build_engine_kv_connector_actions,
    build_engine_kv_injection_plan,
    engine_kv_connector_actions_from_record,
    engine_kv_connector_actions_to_record,
    engine_adapter_request_to_record,
    probe_engine_kv_connector_actions,
    read_engine_adapter_payload,
    read_engine_adapter_request_json,
    view_engine_adapter_payload,
    vllm_adapter_spec,
    write_engine_adapter_handoff_bundle,
)

adapter_request = build_engine_adapter_request(
    ready,
    spec=vllm_adapter_spec(),
)
handoff_record = engine_adapter_request_to_record(adapter_request)
payload_path = Path("/local_disk0/cachet-kv/req-123.kv")
handoff_path, written_payload_path = write_engine_adapter_handoff_bundle(
    adapter_request,
    "req-123-handoff.json",
    payload_uri=f"disk:{payload_path}",
)

# Inside the vLLM/SGLang connector process:
record = read_engine_adapter_request_json(
    "req-123-handoff.json",
    expected_backend="vllm",
)
payload = read_engine_adapter_payload(
    record["payload_source"]["uri"],
    expected_bytes=record["payload_source"]["total_bytes"],
)
payload_or_segments = view_engine_adapter_payload(record, payload)
injection_plan = build_engine_kv_injection_plan(record, expected_backend="vllm")
actions = build_engine_kv_connector_actions(injection_plan, payload_or_segments)
actions_record = engine_kv_connector_actions_to_record(actions)
actions = engine_kv_connector_actions_from_record(actions_record, expected_backend="vllm")

reservation = vllm_block_manager.reserve(
    request_id=actions.reservation.request_id,
    block_count=actions.reservation.total_blocks,
)
for copy in actions.copies:
    source = payload_or_segments if copy.payload_index is None else payload_or_segments[copy.payload_index]
    engine_kv_importer.copy(
        reservation,
        source[copy.source_byte_start : copy.source_byte_end],
        token_start=copy.token_start,
        block_range=(copy.first_block_index, copy.last_block_index_exclusive),
    )
vllm_scheduler.bind_document_kv(reservation, actions.bind)
```

### Handoff Records And Connector Actions

- `engine_adapter_request_to_record` writes a
  `document_kv.engine_adapter_request.v1` record (current `schema_version` 4;
  version 2 records are still readable) with the handle URI, payload source,
  model layout, token/byte segment boundaries, per-segment cache tiers,
  adapter ids, and estimated GPU bytes. It never contains raw KV bytes.
- `write_engine_adapter_handoff_bundle` writes the payload before the JSON
  record that points at it. Use the lower-level `write_engine_adapter_payload`
  and `write_engine_adapter_request_json` only when the two artifacts are
  managed separately.
- `read_engine_adapter_request_json` validates schema, backend, payload source,
  layout geometry, and contiguous segments; `read_engine_adapter_payload`
  checks the expected byte length. `view_engine_adapter_payload` returns
  `memoryview` slices; use `split_engine_adapter_payload` only when a connector
  needs independent `bytes` and can afford the copy.
- `build_engine_kv_injection_plan` returns an `EngineKVInjectionPlan` that
  re-validates layout-derived byte totals, block totals, and segment
  token/byte/block spans so adapters fail before native block-manager calls if
  a handoff is malformed. Segment block ranges may overlap when a segment
  starts or ends inside an engine block: reserve the plan-level `total_blocks`,
  then map byte spans through the segment bindings.
- `engine_kv_connector_actions_to_record` emits the
  `document_kv.engine_kv_connector_actions.v1` reserve/copy/bind/release
  descriptor for out-of-process adapters. Parse it back with
  `engine_kv_connector_actions_from_record` or
  `validate_engine_kv_connector_actions_record` at the engine boundary so a
  stale backend, schema, unsupported field, payload offset, or block map is
  rejected before native block-manager calls.
- Treat the layout's `storage_layout` as a compatibility guard, not a
  tensor-stride language. Byte order comes from `layout_version`, the model
  profile, and `payload_axis_order`: `token_major` (the default) is
  `[token, layer, K/V, kv_head, head_dim]` and `layer_major` is
  `[layer, token, K/V, kv_head, head_dim]`. `kv_stride_bytes` includes backend
  alignment padding, so byte math uses
  `num_layers * num_kv_heads * kv_stride_bytes * 2`.

### Native Probes

Adapter smoke tests wrap the native block manager in the
`EngineKVBlockManagerProbe` protocol and call
`probe_engine_kv_connector_actions(actions, payload_or_segments, probe)`. That
exercises reservation, KV import, handle binding, and release with the exact
production descriptors, but does not schedule decode. Serialize results with
`engine_kv_connector_probe_result_to_record`; probe records use
`schema_version` `2`, which makes the nested `layout` mapping mandatory, and
release evidence rejects records that claim `qwen3:4b-instruct` without the
matching one-byte Qwen3 GQA geometry.

For a reproducible CLI handoff, expose a native factory in the adapter package
and run:

```bash
python -m document_kv_cache.engine_probe \
  --handoff-json req-123-handoff.json \
  --probe-factory my_vllm_adapter.probes:build_probe \
  --expected-backend vllm \
  --actions-output-json vllm-connector-actions.json \
  --output-json vllm-engine-probe.json
```

The factory receives an `EngineKVProbeFactoryContext` and returns an
`EngineKVProbeFactoryResult` with a real `EngineKVBlockManagerProbe` and the
native engine version. Debug factories must return `native_probe=False`, and
`--allow-non-native-probe` exists only for local adapter debugging; release
evidence rejects both.

The built-in reserved vLLM/SGLang probe factories fail closed unless the
serving environment has the backend installed and points the matching delegate
variable at a native factory: `DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY=module:callable`
or `DOCUMENT_KV_SGLANG_NATIVE_PROBE_FACTORY=module:callable`. The wheel ships
provider-backed delegates for vLLM and SGLang HiCache. When using
`vllm_kv_injection.probe:build_native_connector_probe` or
`sglang_kv_injection.probe:build_native_connector_probe`, also pass the
connector-factory metadata
`vllm_kv_injection.connector_factory=vllm_kv_injection.probe:build_document_kv_native_probe_connector`
or
`sglang_kv_injection.connector_factory=sglang_kv_injection.probe:build_document_kv_hicache_probe_connector`;
the Databricks target parser rejects those delegates without it. The SGLang
path covers the HiCache provider and launch preflight; validate live
decode-time prefix binding in the installed SGLang runtime before treating it
as benchmark evidence.

Generate a fail-closed delegate module with
`cachet-native-probe-scaffold --backend vllm --output-file cachet_vllm_native_probe.py`.
Its methods raise `NotImplementedError` until a native adapter replaces them,
so scaffold output is never release evidence. Delegates must declare the
adapter contract they implement, either as a module-level
`DOCUMENT_KV_NATIVE_PROBE_CONTRACT = native_probe_adapter_contract_to_record()`
or a callable-level `document_kv_native_probe_contract` attribute; inspection
reports the delegate as unsupported when the declaration is missing or stale.

`builtin_native_probe_factories_to_record()` reports that status together with
the pinned isolated serving-environment profile for each backend, the
configured delegate path, and the expected and declared adapter contracts. Use
it when preparing native adapter work so the target engine versions and
dependency constraints stay tied to the probe entry points:

```bash
cachet-serving-env \
  --output-json serving-environment-profiles.json

cachet-native-probe-factories \
  --output-json native-probe-factories.json
```

Call `validate_native_probe_factories_record` or
`native_probe_factories_record_issues` before bundling externally supplied
`document_kv.native_probe_factories.v1` diagnostics; release bundles use the
same validator.

## Launch Configs

Build launch-config sidecars with `cachet-engine-launch-config build-vllm` and
`build-sglang` as shown in
[`../../native-engine-integration.md`](../../native-engine-integration.md#launch-configs).
`--extra-config KEY=VALUE` entries are preserved for adapter metadata, but
reserved `document_kv.*` keys are generated by Cachet and cannot be overridden.

## Live Endpoint Smoke

Check a running OpenAI-compatible vLLM or SGLang server without custom serving
code:

```bash
python -m document_kv_cache.live_server \
  --base-url http://localhost:8000 \
  --model-id qwen3:4b-instruct
```

Ordinary OpenAI-compatible servers use the default full-prompt mode. Add
`--cache-arm --runtime-prompt` only for a KV-aware proxy that injects the cached
prefix out of band. For native engine paths, pass a validated handoff so the
request carries the same `kv_transfer_params` as benchmark cache-arm requests.
Use `--kv-transfer-params-transport custom-params` for SGLang and keep the
default `top-level` transport for vLLM:

```bash
python -m document_kv_cache.live_server \
  --base-url http://localhost:8000 \
  --model-id qwen3:4b-instruct \
  --cache-arm \
  --runtime-prompt \
  --kv-transfer-params-transport custom-params \
  --handoff-json /Volumes/catalog/schema/volume/cachet/live/sglang.handoff.json \
  --sglang-hicache-page-keys-json '["first-runtime-page-key", "second-runtime-page-key"]' \
  --expected-backend sglang
```

## Serving Smoke Helpers

`cachet-vllm-smoke` and `cachet-sglang-smoke` run on a GPU node. Each creates an
isolated serving environment from the pinned `document_kv_cache.serving_env`
profile, installs Cachet into it, starts Qwen3 4B Instruct, and writes
`metadata.json`, an import probe, the server log, and the run result to
`--output-dir`.

```bash
python -m document_kv_cache.vllm_smoke \
  --benchmark-id v1_vllm_smoke_001 \
  --output-dir /Volumes/catalog/schema/volume/cachet-v1-smoke
```

- Without `--dataset`, the vLLM helper runs one built-in example per V1
  dataset. Prepared mode requires one `--dataset NAME=PATH` for each of
  `biography`, `hotpotqa`, `musique`, and `niah`, and can generate handoffs
  first with `--benchmark-handoff-generator-factory`,
  `--benchmark-handoff-output-dir`, and `--benchmark-handoff-dtype`. It writes
  `prepared-handoff-coverage.json` and fails before model startup when a row
  lacks loadable vLLM handoff params.
- The vLLM cache arm sends the full logical prompt plus `kv_transfer_params`.
  vLLM's connector scheduler needs the logical prefix token positions before it
  can allocate external KV blocks, so do not use suffix-only cache prompts for
  native vLLM evidence.
- The SGLang helper is the live readiness gate before publishing SGLang
  numbers. `--baseline-only` output is provider and server bring-up evidence,
  not benchmark evidence. For cache-arm readiness prefer
  `--generate-live-handoff`, which builds the live handoff and exact HiCache
  page keys inside the isolated runtime; otherwise pass a validated
  `--handoff-json` plus `--sglang-hicache-page-keys-json`.
- Outside `--baseline-only`, the SGLang helper runs a clean baseline and
  flushes the prefix cache before the cache arm; it also flushes before the
  model-quality canary. Use `--no-flush-cache-before-cache-arm` or
  `--no-flush-cache-before-canary` only for diagnostics.
- SGLang page keys must match the KV layout block size (`16` for the built-in
  Qwen3 V1 layout). Prepared SGLang runs default to that page size and fail
  before server startup when a handoff uses a different block size or its page
  keys do not cover every full payload page.

```bash
python -m document_kv_cache.sglang_smoke \
  --benchmark-id v1_sglang_live_001 \
  --output-dir /Volumes/catalog/schema/volume/cachet-sglang-live \
  --baseline-only \
  --hardware-target aws-g6-l4
```
