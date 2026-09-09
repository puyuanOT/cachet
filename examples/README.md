# Examples

This folder contains small, runnable examples for people trying Cachet locally.

- `quickstart_local.py` runs without cloud services, GPUs, vLLM, or SGLang. It
  uses a toy KV generator, memory storage, and a temporary disk shard to show
  the document -> KV payload -> engine handoff flow.
- `custom_method_registry.py` shows how a team-owned method composes an
  immutable registry without changing Cachet's process-wide defaults.
- `transformers_kv_generation.py` is an opt-in, real-model artifact generation
  example. It downloads pinned model/tokenizer revisions and may require a GPU;
  it is not part of the safe CPU quickstart.

After installing the package:

```bash
python -m cachet.quickstart_local
```

From a source checkout:

```bash
python examples/quickstart_local.py
python examples/custom_method_registry.py
```

After installing the serving/model dependencies, real generation requires
explicit revisions:

```bash
python examples/transformers_kv_generation.py \
  --model-revision <commit> \
  --tokenizer-revision <commit>
```

Examples should stay self-contained and safe to run from a source checkout.
