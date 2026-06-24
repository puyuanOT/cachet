# Examples

This folder contains small, runnable examples for people trying Cachet locally.

- `quickstart_local.py` runs without cloud services, GPUs, vLLM, or SGLang. It
  uses a toy KV generator, memory storage, and a temporary disk shard to show
  the document -> KV payload -> engine handoff flow.

After installing the package:

```bash
python -m cachet.quickstart_local
```

From a source checkout:

```bash
python examples/quickstart_local.py
```

Examples should stay self-contained and safe to run from a source checkout.
