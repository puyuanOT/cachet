"""Public wrapper for :mod:`restaurant_kv_serving.engine_probe`."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.engine_probe",
    (
        "EngineKVProbeConfig",
        "ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND",
        "ENGINE_KV_PROBE_METADATA_HANDOFF_JSON",
        "ENGINE_KV_PROBE_METADATA_PAYLOAD_URI",
        "ENGINE_KV_PROBE_METADATA_PROBE_FACTORY",
        "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE",
        "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION",
        "EngineKVProbeFactory",
        "EngineKVProbeFactoryContext",
        "EngineKVProbeFactoryResult",
        "run_engine_kv_connector_probe",
        "read_engine_adapter_payload",
        "write_engine_kv_connector_probe_result_json",
        "load_engine_kv_probe_factory",
        "parse_args",
        "main",
    ),
    globals(),
)

_main_bridge = LegacyMainBridge(
    legacy_module_name="restaurant_kv_serving.engine_probe",
    public_namespace=globals(),
    hook_names=("run_engine_kv_connector_probe", "write_engine_kv_connector_probe_result_json"),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


del LegacyMainBridge, reexport_public
