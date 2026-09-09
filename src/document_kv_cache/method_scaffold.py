"""Generate a fail-closed Cachet KV method plugin skeleton."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


__all__ = [
    "MethodScaffoldConfig",
    "render_method_plugin",
    "write_method_plugin",
    "main",
]

_METHOD_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class MethodScaffoldConfig:
    method_id: str
    display_name: str
    pre_rope: bool = False
    selective_recompute: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not _METHOD_ID_PATTERN.fullmatch(self.method_id):
            raise ValueError("method_id must use lowercase letters, digits, and underscores")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display_name must be a non-empty string")
        if type(self.pre_rope) is not bool:
            raise ValueError("pre_rope must be a boolean")
        if type(self.selective_recompute) is not bool:
            raise ValueError("selective_recompute must be a boolean")

    @property
    def class_name(self) -> str:
        return "".join(part.capitalize() for part in self.method_id.split("_")) + "Generator"


def render_method_plugin(config: MethodScaffoldConfig) -> str:
    if not isinstance(config, MethodScaffoldConfig):
        raise TypeError("config must be a MethodScaffoldConfig")
    return f'''"""Cachet method plugin for {config.display_name}.

The plugin starts fail-closed. Implement ``generate`` and its tests, then set
``implemented=True`` only when artifact generation and serving are both wired.
"""

from __future__ import annotations

from document_kv_cache.methods import (
    CACHET_ARTIFACT_EXECUTION,
    CACHET_CONNECTOR_MODE,
    DOCUMENT_KV_CACHE_ARM,
    MethodRegistry,
    MethodSpec,
)


class {config.class_name}:
    pre_rope = {config.pre_rope!r}

    def generate(self, *, document, chunk, config, training_artifacts):
        raise NotImplementedError(
            "return a validated document_kv_cache.kvpack.PackChunk"
        )


def build_generator(**kwargs):
    if kwargs:
        raise TypeError(f"unsupported generator options: {{sorted(kwargs)}}")
    return {config.class_name}()


METHOD_SPEC = MethodSpec(
    method={config.method_id!r},
    display_name={config.display_name!r},
    arm_id=DOCUMENT_KV_CACHE_ARM,
    connector_mode=CACHET_CONNECTOR_MODE,
    pre_rope={config.pre_rope!r},
    selective_recompute={config.selective_recompute!r},
    implemented=False,
    artifact_version="1",
    execution_kind=CACHET_ARTIFACT_EXECUTION,
    generator_factory=__name__ + ":build_generator",
    description="TODO: document preparation, reuse, and correctness semantics.",
)


def register(registry: MethodRegistry) -> MethodRegistry:
    return registry.with_spec(METHOD_SPEC)
'''


def write_method_plugin(
    config: MethodScaffoldConfig,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists; pass overwrite=True to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_method_plugin(config), encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Cachet KV method plugin skeleton.")
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--pre-rope", action="store_true")
    parser.add_argument("--selective-recompute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        write_method_plugin(
            MethodScaffoldConfig(
                method_id=args.method_id,
                display_name=args.display_name,
                pre_rope=args.pre_rope,
                selective_recompute=args.selective_recompute,
            ),
            args.output_file,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
