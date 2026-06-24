# Document Benchmark Plan Executor Ownership PR Evidence

This folder records PR evidence for the slice that moved benchmark plan
execution into the public `document_kv_cache.benchmark_plan_executor`
namespace while keeping `restaurant_kv_serving.benchmark_plan_executor` as a
legacy compatibility facade for downstream imports, star imports, private
helpers, and CLI monkeypatch hooks.
