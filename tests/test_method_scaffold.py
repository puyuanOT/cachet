from __future__ import annotations

import importlib.util

import pytest

from document_kv_cache.method_scaffold import (
    MethodScaffoldConfig,
    main,
    render_method_plugin,
    write_method_plugin,
)
from document_kv_cache.methods import default_method_registry


def test_method_scaffold_renders_fail_closed_registry_plugin(tmp_path) -> None:
    config = MethodScaffoldConfig(
        method_id="example_reuse",
        display_name="Example Reuse",
        pre_rope=True,
        selective_recompute=True,
    )
    target = write_method_plugin(config, tmp_path / "example_reuse.py")
    spec = importlib.util.spec_from_file_location("example_reuse", target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    registered = module.register(default_method_registry())

    method = registered.get("example_reuse")
    assert method.pre_rope is True
    assert method.selective_recompute is True
    assert method.implemented is False
    assert "implemented=False" in render_method_plugin(config)
    with pytest.raises(NotImplementedError, match="not implemented"):
        method.require_implemented()


def test_method_scaffold_refuses_to_overwrite(tmp_path) -> None:
    target = tmp_path / "method.py"
    target.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_method_plugin(
            MethodScaffoldConfig("example_reuse", "Example Reuse"),
            target,
        )


def test_method_scaffold_cli_writes_module(tmp_path) -> None:
    target = tmp_path / "method.py"

    assert (
        main(
            [
                "--method-id",
                "example_reuse",
                "--display-name",
                "Example Reuse",
                "--output-file",
                str(target),
            ]
        )
        == 0
    )
    assert "METHOD_SPEC = MethodSpec(" in target.read_text(encoding="utf-8")


@pytest.mark.parametrize("method_id", ("Example", "two-words", "2fast", ""))
def test_method_scaffold_rejects_unstable_method_ids(method_id: str) -> None:
    with pytest.raises(ValueError, match="method_id"):
        MethodScaffoldConfig(method_id, "Example")
