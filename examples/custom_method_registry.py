"""Compose an application-owned KV method registry without global mutation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


repo_src = Path(__file__).resolve().parents[1] / "src"
if repo_src.is_dir():
    sys.path.insert(0, str(repo_src))

from document_kv_cache.methods import (  # noqa: E402
    CACHET_CONNECTOR_MODE,
    DOCUMENT_KV_CACHE_ARM,
    MethodSpec,
    default_method_registry,
)


METHOD_SPEC = MethodSpec(
    method="example_reuse",
    display_name="Example Reuse",
    arm_id=DOCUMENT_KV_CACHE_ARM,
    connector_mode=CACHET_CONNECTOR_MODE,
    pre_rope=False,
    selective_recompute=False,
    implemented=False,
    generator_factory=__name__ + ":build_generator",
    description="Runnable registry example; artifact generation is intentionally incomplete.",
)


class ExampleGenerator:
    pre_rope = False

    def generate(self, *, document, chunk, config, training_artifacts):
        raise NotImplementedError("implement PackChunk generation before enabling this method")


def build_generator() -> ExampleGenerator:
    return ExampleGenerator()


def main() -> int:
    default = default_method_registry()
    application_registry = default.with_spec(METHOD_SPEC)
    registered = application_registry.get(METHOD_SPEC.method_id)
    print(
        json.dumps(
            {
                "method_id": registered.method_id,
                "implemented": registered.implemented,
                "default_registry_unchanged": METHOD_SPEC.method_id not in default,
                "application_registry_size": len(application_registry),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
