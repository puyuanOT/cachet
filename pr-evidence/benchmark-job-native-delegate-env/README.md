# Benchmark job native delegate env

This PR adds optional backend-specific native probe delegate factory paths to
the full V1 Databricks benchmark job helper and Asset Bundle template. The
delegate paths are emitted as cluster environment variables consumed by Cachet's
built-in native vLLM/SGLang probe factories, while benchmark runner parameters
remain stable.

The PR evidence JSON is generated after GPT-5.5 review completes.
