# Adding A KV Reuse Method

Cachet methods are immutable, application-owned plugins. A method author adds a
`MethodSpec`, a generator factory for Cachet-produced artifacts (or an
engine-native connector mode), contract tests, and benchmark evidence. Runtime
flags alone do not define a new method.

## Bootstrap A Plugin

From a source checkout:

```bash
poetry run cachet-method-scaffold \
  --method-id my_reuse_method \
  --display-name "My Reuse Method" \
  --output-file experiments/my_reuse_method.py
```

Add `--pre-rope` when the artifact stores position-independent keys and
`--selective-recompute` when serving must recompute chosen tokens with full
context. The generated spec has `implemented=False`; this is deliberate. It
prevents an incomplete path from entering benchmark tables.

The generated module exposes:

- `METHOD_SPEC`, the stable method and artifact-semantics contract;
- `build_generator`, the importable generator factory;
- `register(registry)`, which returns a new immutable registry.

Do not mutate the process-wide default registry. Compose an application
registry explicitly:

```python
from document_kv_cache.methods import default_method_registry
from experiments.my_reuse_method import register

registry = register(default_method_registry())
method = registry.get("my_reuse_method")
```

Run the structural conformance check while the scaffold is still fail-closed:

```bash
poetry run cachet-method-conformance \
  --plugin experiments.my_reuse_method:METHOD_SPEC \
  --allow-unimplemented
```

After enabling the method, remove `--allow-unimplemented`. The command checks
registry composition, emits the typed `ReusePlan`, and verifies that an
implemented artifact method's factory is importable. Add
`--instantiate-generator` only for a lightweight factory; the built-in
Transformers factory can load model weights.

Cachet includes a CPU-only end-to-end extension test that needs no model
weights:

```bash
poetry run cachet-method-conformance \
  --plugin document_kv_cache.reference_method:METHOD_SPEC \
  --instantiate-generator
poetry run pytest tests/test_reference_method.py -q
```

The reference generator emits deterministic KV-shaped bytes solely to exercise
identity, token, streaming, and handoff contracts. Its output is never valid
latency or model-quality evidence.

## Implement The Generator

For a Cachet artifact method, `generate` returns one validated `PackChunk` per
source chunk. Its key must carry:

- an `ArtifactIdentity` with method version, model/tokenizer revisions,
  topology, generator version, runtime dtype, and payload axis order;
- a `TokenContract` built from the exact token IDs used to generate the KV;
- the source content hash and layout metadata.

Set `implemented=True` only after `MethodSpec.create_generator` succeeds,
generation produces these contracts, and an established serving-engine
connector consumes the artifact. Engine-native methods instead declare
`execution_kind="engine_native"` and must not declare a generator factory.

## Required Tests

At minimum, add CPU tests for:

1. immutable registry composition and duplicate registration;
2. generator capability matching (`pre_rope`);
3. exact artifact and token identity;
4. payload checksum corruption;
5. runtime compatibility rejection;
6. method-aware benchmark serialization;
7. quality and cache-state publication gates.

GPU integration tests should prove that the target vLLM or SGLang version
hydrates engine-owned KV blocks and produces the same answer as full prefill.

## OpenTable GPU Escalation Path

Use the same order for team experiments:

1. clean checkout: `poetry install -E test`;
2. CPU reference and method conformance;
3. one pinned-revision Transformers artifact with
   `examples/transformers_kv_generation.py`;
4. local engine probe and handoff validation;
5. sanitized Databricks vLLM/SGLang smoke jobs;
6. paired N-way benchmark runs and publication-gate evaluation.

Keep Databricks tokens and workspace identifiers in the approved secret store.
Generated payloads, raw prompts, service responses, and local run records stay
under ignored scratch paths. Only sanitized identity, telemetry, statistics,
and gate records move into benchmark evidence.

## Benchmark Evidence

Every cache request must carry the same `cache_method` and `artifact_id` from
handoff through telemetry and report rows. Use paired examples and repeats
against `baseline_prefill`. Publication requires:

- successful latency and quality measurements;
- quality deltas within the declared thresholds;
- a fully resolved `ArtifactIdentity`;
- a request-correlated cache-state attestation;
- successful page-cache eviction or direct I/O for claims labeled cold.

`evaluate_benchmark_publication_gate` fails closed when any of those records are
missing. A prefetch or payload-cache hit is warm evidence and cannot be
published as cold disk-to-GPU latency.

## Review Checklist

- The method ID and artifact version will remain stable.
- Storage encoding is not confused with the serving engine's runtime KV dtype.
- Unsupported engines and layouts fail before generation or injection.
- Generation is streamed to an atomic shard; it does not retain all payloads.
- New dependencies are optional and versioned in the environment evidence.
- Benchmark records are sanitized and contain no prompt or customer data.
