"""Worker-side entrypoint for Databricks engine KV-connector probes."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from document_kv_cache.storage import local_path

_SUPPORTED_BACKENDS = ("vllm", "sglang")
_FIXTURE_PAYLOAD_MODES = ("merged", "segmented")
_DEFAULT_FIXTURE_PAYLOAD_MODE = "segmented"

__all__ = ["run_engine_probe_task"]


def run_engine_probe_task(argv: Sequence[str] | None = None) -> int:
    """Run an optional generated fixture step, then the native engine probe."""

    runner_argv = list(sys.argv[1:] if argv is None else argv)
    runner_args, engine_probe_argv = _split_fixture_runner_args(runner_argv)
    if runner_args.fixture_output_dir is not None:
        from document_kv_cache import probe_fixtures

        fixture_exit_code = probe_fixtures.main(
            [
                "--output-dir",
                runner_args.fixture_output_dir,
                "--backend",
                runner_args.fixture_backend,
                "--payload-mode",
                runner_args.fixture_payload_mode,
            ]
        )
        if fixture_exit_code:
            return fixture_exit_code
    if runner_args.vllm_runtime_preflight_output_json is not None:
        preflight_exit_code = _run_vllm_runtime_preflight(runner_args)
        if preflight_exit_code:
            return preflight_exit_code
    if runner_args.sglang_runtime_preflight_output_json is not None:
        preflight_exit_code = _run_sglang_runtime_preflight(runner_args)
        if preflight_exit_code:
            return preflight_exit_code
    from document_kv_cache import engine_probe

    return engine_probe.main(engine_probe_argv)


def _split_fixture_runner_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fixture-output-dir")
    parser.add_argument("--fixture-backend", choices=_SUPPORTED_BACKENDS)
    parser.add_argument(
        "--fixture-payload-mode",
        choices=_FIXTURE_PAYLOAD_MODES,
        default=_DEFAULT_FIXTURE_PAYLOAD_MODE,
    )
    parser.add_argument("--vllm-runtime-preflight-output-json")
    parser.add_argument("--vllm-runtime-preflight-layer-names-json")
    parser.add_argument("--sglang-runtime-preflight-output-json")
    parser.add_argument("--sglang-runtime-preflight-launch-config-json")
    fixture_args, engine_probe_argv = parser.parse_known_args(argv)
    if fixture_args.fixture_output_dir is not None and fixture_args.fixture_backend is None:
        raise ValueError("--fixture-backend is required when --fixture-output-dir is provided")
    if fixture_args.fixture_output_dir is None and fixture_args.fixture_backend is not None:
        raise ValueError("--fixture-backend requires --fixture-output-dir")
    if (
        fixture_args.vllm_runtime_preflight_output_json is None
        and fixture_args.vllm_runtime_preflight_layer_names_json is not None
    ):
        raise ValueError(
            "--vllm-runtime-preflight-layer-names-json requires --vllm-runtime-preflight-output-json"
        )
    if (
        fixture_args.vllm_runtime_preflight_output_json is not None
        and fixture_args.vllm_runtime_preflight_layer_names_json is None
    ):
        raise ValueError(
            "--vllm-runtime-preflight-output-json requires --vllm-runtime-preflight-layer-names-json"
        )
    if (
        fixture_args.sglang_runtime_preflight_output_json is None
        and fixture_args.sglang_runtime_preflight_launch_config_json is not None
    ):
        raise ValueError(
            "--sglang-runtime-preflight-launch-config-json requires --sglang-runtime-preflight-output-json"
        )
    if (
        fixture_args.sglang_runtime_preflight_output_json is not None
        and fixture_args.sglang_runtime_preflight_launch_config_json is None
    ):
        raise ValueError(
            "--sglang-runtime-preflight-output-json requires --sglang-runtime-preflight-launch-config-json"
        )
    return fixture_args, engine_probe_argv


def _run_vllm_runtime_preflight(runner_args: argparse.Namespace) -> int:
    from vllm_kv_injection import vllm_runtime_preflight

    return vllm_runtime_preflight.main(
        [
            "--layer-names-json",
            _cluster_preflight_file_argument(runner_args.vllm_runtime_preflight_layer_names_json),
            "--output-json",
            _cluster_preflight_file_argument(runner_args.vllm_runtime_preflight_output_json),
        ]
    )


def _run_sglang_runtime_preflight(runner_args: argparse.Namespace) -> int:
    from sglang_kv_injection import sglang_runtime_preflight

    return sglang_runtime_preflight.main(
        [
            "--launch-config-json",
            _cluster_preflight_file_argument(runner_args.sglang_runtime_preflight_launch_config_json),
            "--output-json",
            _cluster_preflight_file_argument(runner_args.sglang_runtime_preflight_output_json),
        ]
    )


def _cluster_preflight_file_argument(value: str) -> str:
    stripped_value = value.lstrip()
    if stripped_value.startswith(("{", "[")):
        return value
    scheme = _uri_scheme(value)
    if scheme in {"dbfs", "disk", "file", "uc-volume"} or value == "/Volumes" or value.startswith("/Volumes/"):
        return str(local_path(value))
    return value


def _uri_scheme(uri: str) -> str | None:
    head = uri.split("/", maxsplit=1)[0]
    if ":" not in head:
        return None
    scheme, _separator, _rest = head.partition(":")
    return scheme or None
