"""Databricks runner for Cachet runtime KV offload probe evidence."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _cluster_file_path(uri: str) -> str:
    if uri.startswith("dbfs:/"):
        return "/dbfs/" + uri.removeprefix("dbfs:/").lstrip("/")
    return uri


def _install_package_wheel(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--package-wheel-uri")
    args, remaining = parser.parse_known_args(argv)
    if args.package_wheel_uri:
        package_wheel_path = _cluster_file_path(args.package_wheel_uri)
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_wheel_path])
        os.environ["DOCUMENT_KV_PACKAGE_INSTALL_SPEC"] = package_wheel_path
    return remaining


if __name__ == "__main__":
    remaining_args = _install_package_wheel(sys.argv[1:])
    from document_kv_cache.runtime_kv_offload_probe import main

    exit_code = main(remaining_args)
    if exit_code:
        raise SystemExit(exit_code)
