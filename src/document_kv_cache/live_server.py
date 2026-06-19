"""Public wrapper for :mod:`restaurant_kv_serving.live_server`."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.live_server",
    (
        "LIVE_CHECK_SUITE_ID",
        "DEFAULT_LIVE_CHECK_ANSWER",
        "LiveServerCheckConfig",
        "LiveServerCheckResult",
        "build_live_server_check_request",
        "run_openai_compatible_live_check",
        "main",
    ),
    globals(),
)


_main_bridge = LegacyMainBridge(
    legacy_module_name="restaurant_kv_serving.live_server",
    public_namespace=globals(),
    hook_names=("LiveServerCheckConfig", "run_openai_compatible_live_check"),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del LegacyMainBridge
del reexport_public
