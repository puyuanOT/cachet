# Current Dependency Freshness Evidence

This folder contains the current generated
`document_kv.dependency_freshness.v1` evidence for Cachet dependency policy.

The evidence distinguishes three dependency surfaces:

- Direct package metadata pins in `pyproject.toml`. These must be exact `==`
  pins and must match the latest stable package-index version supplied to the
  evidence generator.
- Isolated vLLM/SGLang serving-profile pins. These must be exact pins. A
  non-latest runtime pin is accepted only when the record carries an explicit
  compatibility or Databricks-validation hold reason.
- Resolved transitive drift from `poetry show --outdated --all`. Drift is
  accepted only when the record carries an explicit resolver-constraint reason.

Current direct pins are fresh: `poetry-core==2.4.1`, `packaging==26.3`,
`pyspark==4.1.2`, `databricks-sdk==0.118.0`, and `pytest==9.1.1`.

The vLLM profile is the reviewed `0.27.1+cu129` publication closure: its direct
CUDA/runtime packages are exact pins, while all transitive packages are bound
by the packaged 196-distribution hash lock. Current runtime holds are
intentionally visible. The vLLM companion pins for `tokenizers`, `numpy`,
`fastapi`, and `prometheus-fastapi-instrumentator`, plus the SGLang runtime
pin, may move only after fresh L4/A10G Databricks validation.

Current transitive drift is `protobuf`: Poetry resolves `protobuf==6.33.6`
because `databricks-sdk==0.118.0` constrains protobuf below `7.0`, while the
latest package-index release supplied to the record is `7.35.1`.

Files:

- `dependency-freshness-evidence.json`: generated freshness evidence record.

Regenerate from the repository root with:

```bash
python -m document_kv_cache.dependency_freshness \
  --pyproject pyproject.toml \
  --latest-version poetry-core=2.4.1 \
  --latest-version packaging=26.3 \
  --latest-version pyspark=4.1.2 \
  --latest-version databricks-sdk=0.118.0 \
  --latest-version pytest=9.1.1 \
  --latest-version vllm=0.27.1+cu129 \
  --latest-version torch=2.13.0+cu129 \
  --latest-version torchaudio=2.11.0+cu129 \
  --latest-version torchvision=0.28.0+cu129 \
  --latest-version triton=3.7.1 \
  --latest-version numba=0.65.0 \
  --latest-version torchcodec=0.16.0+cu129 \
  --latest-version PyNvVideoCodec=2.0.4 \
  --latest-version flashinfer-python=0.6.16.post3 \
  --latest-version flashinfer-cubin=0.6.16.post3 \
  --latest-version flashinfer-jit-cache=0.6.16.post3+cu129 \
  --latest-version apache-tvm-ffi=0.1.11 \
  --latest-version tilelang=0.1.12 \
  --latest-version nvidia-cudnn-frontend=1.27.0 \
  --latest-version nvtx=0.2.15 \
  --latest-version fastsafetensors=0.3.3 \
  --latest-version nvidia-cutlass-dsl=4.6.0 \
  --latest-version quack-kernels=0.6.1 \
  --latest-version tokenspeed-mla=0.1.8 \
  --latest-version humming-kernels=0.1.10 \
  --latest-version transformers=5.12.1 \
  --latest-version huggingface-hub=1.20.1 \
  --latest-version bitsandbytes=0.49.2 \
  --latest-version accelerate=1.14.0 \
  --latest-version opencv-python-headless=4.13.0.92 \
  --latest-version tokenizers=0.23.1 \
  --latest-version numpy=2.5.0 \
  --latest-version fastapi=0.138.0 \
  --latest-version prometheus-fastapi-instrumentator=8.0.2 \
  --latest-version sglang=0.5.13.post1 \
  --latest-version protobuf=7.35.1 \
  --allow-runtime-pin 'sglang=Pinned to the latest g6/L4 Databricks-validated Cachet SGLang HiCache provider profile; upgrading requires a fresh native/live Databricks validation.' \
  --allow-runtime-pin 'tokenizers=Pinned by the reviewed vLLM 0.27.1 CUDA 12.9 hash lock; upgrade only after fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'numpy=Pinned by the reviewed vLLM 0.27.1 CUDA 12.9 hash lock; upgrade only after fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'fastapi=Pinned by the reviewed vLLM 0.27.1 CUDA 12.9 hash lock; upgrade only after fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'prometheus-fastapi-instrumentator=Pinned by the reviewed vLLM 0.27.1 CUDA 12.9 hash lock; upgrade only after fresh L4 and A10G Databricks validation.' \
  --outdated-package protobuf=6.33.6:7.35.1 \
  --allow-transitive-outdated 'protobuf=databricks-sdk==0.118.0 currently resolves protobuf <7.0, so Poetry keeps protobuf 6.33.6 even though PyPI has 7.35.1.' \
  --output-json docs/release-ops/evidence/dependency-freshness/current/dependency-freshness-evidence.json
```

Do not commit package-index credentials, service tokens, raw resolver logs,
virtual environments, wheels, or local scratch output here.
