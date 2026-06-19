"""Public package facade for Document KV Cache.

The implementation still lives in :mod:`restaurant_kv_serving` while existing
Databricks jobs migrate. This facade gives new users the document-generic import
path and console script targets without duplicating implementation modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LEGACY_PACKAGE = "restaurant_kv_serving"
_LEGACY_ROOT_EXPORTS = frozenset(
    {
        "ChunkType",
        "RestaurantKVRequest",
        "RestaurantKVService",
    }
)
_PUBLIC_SUBMODULES = frozenset(
    {
        "admission",
        "benchmark_plan",
        "benchmark_plan_executor",
        "benchmark_runner",
        "benchmarks",
        "cache",
        "databricks_engine_probe_job",
        "databricks_job",
        "databricks_runs",
        "databricks_storage_benchmark_job",
        "dataset_prep",
        "engine",
        "engine_adapters",
        "engine_probe",
        "engine_protocol",
        "kvpack",
        "live_server",
        "manifest",
        "materializer",
        "model_profiles",
        "models",
        "native_probe_factories",
        "openai_compatible",
        "planner",
        "pr_evidence",
        "release_bundle",
        "release_evidence",
        "service",
        "serving_env",
        "storage",
        "storage_benchmark",
        "template_resources",
        "vllm_smoke",
        "databricks_vllm_smoke_job",
        "workflow",
    }
)

_legacy_package = import_module(_LEGACY_PACKAGE)
__all__ = [
    name
    for name in getattr(_legacy_package, "__all__", ())
    if name not in _LEGACY_ROOT_EXPORTS
]


def __getattr__(name: str) -> Any:
    if name in _PUBLIC_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    return getattr(_legacy_package, name)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_PUBLIC_SUBMODULES})
