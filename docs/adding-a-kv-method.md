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

## Declare The Physical Handoff Topology

Physical segmentation is method semantics, not a benchmark-wrapper toggle.
Declare `MethodSpec.handoff_topology` with an immutable
`HandoffTopologySpec`. Cachet's full-prefix control requires
`segment_per_document=False`; vanilla KV requires
`segment_per_document=True`. The public handoff generator rejects a conflicting
flag before it creates an output directory, writes a manifest, or invokes the
generator.

Custom methods can set `segment_per_document=None` for a method-owned topology
that is not expressible as either built-in mode. They must then validate and
attest their custom topology in their own generation path. Do not reuse a
built-in method ID to bypass its declared segmentation contract.

## Track Lifecycle And Evidence Separately

Every `MethodSpec` carries a closed `MethodLifecycle` record. Code readiness is
either `planned` or `runnable`; independent fields record upstream
reproduction, engine validation, live-canary evidence, and publication
evidence. Do not use `implemented=True` as a claim that an upstream result was
reproduced or that publication evidence exists. It remains a compatibility
view of runnable code and must agree with `lifecycle.code_status`.

A planned method cannot produce a `ReusePlan`. Keep proposed algorithms such as
selective recomputation out of executable descriptors until an implementation
and handler exist. Promote each evidence field only after its corresponding
record has passed. A passing live canary requires passing engine validation,
and publication evidence requires a passing canary. Upstream reproduction
remains independent because it may be `not_applicable` for a Cachet-native
control method.

## Declare Runtime Customizations

Method-owned provider decoding, token selection, and token recomputation use
immutable `RuntimeOperationDescriptor` values containing a strategy ID,
version, and canonical configuration digest. The descriptors are authenticated
inside `ReusePlan` and survive handoff and connector-action round trips.

The serving application injects an immutable
`RuntimeOperationHandlerRegistry`. Each binding is exact for phase, strategy,
version, and configuration digest. A backend must advertise that exact
phase/strategy/version and the registry must resolve the authenticated digest
before any payload read or KV injection. This keeps custom configuration
application-owned without process-global mutation. Handlers receive the
descriptor, including its verified digest, plus the immutable request context.

Provider decoding runs before the runtime KV tensor/page view. Selection and
recomputation remain distinct hooks and run in that order. Cachet supplies the
contract and fail-closed dispatch boundary; each method supplies its own
algorithm and configuration resolver.

## Required Tests

At minimum, add CPU tests for:

1. immutable registry composition and duplicate registration;
2. generator capability matching (`pre_rope`);
3. exact artifact and token identity;
4. payload checksum corruption;
5. runtime compatibility rejection;
6. method-aware benchmark serialization;
7. quality and cache-state publication gates.
8. operation-handler round trips and configuration-digest mismatch rejection
   when the method declares runtime customizations.
9. correct and incorrect physical handoff topology, including rejection before
   any output write.

GPU integration tests should prove that the target vLLM or SGLang version
hydrates engine-owned KV blocks and produces the same answer as full prefill.

## OpenTable GPU Escalation Path

Use the same order for team experiments:

1. pin the author's upstream repository and reproduce its documented result
   before changing Cachet;
2. preserve that upstream implementation as a separately identified reference
   arm and record its environment;
3. clean Cachet checkout: `poetry install -E test`;
4. CPU reference and method conformance;
5. one pinned-revision Transformers artifact with
   `examples/transformers_kv_generation.py`;
6. local engine probe and handoff validation;
7. sanitized Databricks vLLM/SGLang smoke jobs;
8. paired N-way runs of baseline, upstream reference, and Cachet integration;
9. optimize only after the integrated result is functionally and statistically
   comparable with the reproduced reference.

Keep Databricks tokens and workspace identifiers in the approved secret store.
Generated payloads, raw prompts, service responses, and local run records stay
under ignored scratch paths. Only sanitized identity, telemetry, statistics,
and gate records move into benchmark evidence.

Method-specific prompt or token transformations are allowed, but they must be
versioned and recorded. Comparisons hold the logical documents, query, expected
answer, and decode budget fixed while accounting for the physical tokens each
arm actually serves. Training, preprocessing, artifact generation, checkpoint
loading, and artifact footprint are offline costs; report them separately from
online TTFT, TTC, throughput, and serving resources.

For a custom dataset, register one versioned `DatasetScorer` with both its
metric function and `prompt_function`, then pass the same immutable
`DatasetScorerRegistry` to `run_benchmark_suite` (or
`run_openai_compatible_benchmark`) and `generate_benchmark_handoff_bundles`.
This makes the cached prefix come from the scorer-owned logical prompt instead
of the built-in V1 template. All custom scorers in one run/bundle must share a
prompt-template version, and that version must match the manifest and artifact
identity. The checked-in remote benchmark-plan CLI remains deliberately
V1-closed; custom scorer code is an explicit programmatic integration rather
than a dynamically imported command-line plugin.

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

Use the evidence levels in [`evidence-policy.md`](evidence-policy.md): smoke for
execution-only checks, canary for reproducible paired engineering evidence, and
publication only for sanitized evidence that satisfies the publication gate.
A private workspace or DBFS path can supplement provenance but cannot be the
only evidence behind a public comparison.

## Review Checklist

- The method ID and artifact version will remain stable.
- Storage encoding is not confused with the serving engine's runtime KV dtype.
- Unsupported engines and layouts fail before generation or injection.
- Generation is streamed to an atomic shard; it does not retain all payloads.
- New dependencies are optional and versioned in the environment evidence.
- Benchmark records are sanitized and contain no prompt or customer data.
