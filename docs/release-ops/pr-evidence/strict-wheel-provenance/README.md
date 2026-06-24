# Strict Wheel Provenance

This PR tightens release-bundle validation for the tested Cachet package wheel.
Package wheel filenames must now identify the `document-kv-cache` distribution,
and strict V1 release bundles require the wheel metadata version to match the
current project version from `pyproject.toml` or installed package metadata.

The change prevents an otherwise valid but stale or wrong-distribution wheel
from being copied into an auditable V1 release bundle.
