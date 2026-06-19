"""Public wrapper for :mod:`restaurant_kv_serving.vllm_smoke`."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.vllm_smoke",
    (
        "VLLM_VERSION",
        "TRANSFORMERS_CONSTRAINT",
        "HUGGINGFACE_HUB_CONSTRAINT",
        "TOKENIZERS_CONSTRAINT",
        "NUMPY_CONSTRAINT",
        "FASTAPI_CONSTRAINT",
        "PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT",
        "HF_MODEL_ID",
        "SERVED_MODEL_NAME",
        "SERVER_BASE_URL",
        "SMOKE_DATASETS",
        "VLLMSmokeBenchmarkConfig",
        "run_vllm_smoke_benchmark",
        "build_metadata",
        "dependency_constraints",
        "build_vllm_server_args",
        "build_benchmark_runner_args",
        "write_smoke_datasets",
        "smoke_dataset_records",
        "dataset_args",
        "parse_args",
        "main",
    ),
    globals(),
)


_main_bridge = LegacyMainBridge(
    legacy_module_name="restaurant_kv_serving.vllm_smoke",
    public_namespace=globals(),
    hook_names=("run_vllm_smoke_benchmark",),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del LegacyMainBridge
del reexport_public
