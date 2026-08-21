from __future__ import annotations

import importlib.util
from pathlib import Path

from document_kv_cache.engine_protocol import KVStorageLayout
from document_kv_cache.reuse_contract import PositionHandling


_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "transformers_kv_generation.py"
)
_EXAMPLE_SPEC = importlib.util.spec_from_file_location(
    "cachet_transformers_kv_generation_example",
    _EXAMPLE_PATH,
)
assert _EXAMPLE_SPEC is not None and _EXAMPLE_SPEC.loader is not None
transformers_kv_generation = importlib.util.module_from_spec(_EXAMPLE_SPEC)
_EXAMPLE_SPEC.loader.exec_module(transformers_kv_generation)


def test_transformers_example_builds_vanilla_v2_pre_rope_contract(
    monkeypatch,
) -> None:
    observed_pre_rope: list[bool] = []

    class FakeGenerator:
        pre_rope = True
        position_handling = PositionHandling.REROPE_AT_INJECTION
        rope_theta = 5_000_000.0
        rope_rotary_dim = 128
        layout = None

        def bind_layout(self, layout) -> None:
            self.layout = layout

        def generate(self) -> None:
            raise AssertionError("the construction regression must not generate KV")

    generator = FakeGenerator()

    def fake_from_pretrained(config, *, pre_rope=False):
        del config
        observed_pre_rope.append(pre_rope)
        return generator

    monkeypatch.setattr(
        transformers_kv_generation.TransformersKVChunkGenerator,
        "from_pretrained",
        fake_from_pretrained,
    )

    method, built_generator, layout = (
        transformers_kv_generation._build_vanilla_v2_generator_and_layout(
            transformers_kv_generation.TransformersKVGeneratorConfig(
                model_id="Qwen/Qwen3-4B-Instruct-2507",
                torch_dtype="bfloat16",
            ),
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            dtype="bfloat16",
        )
    )

    assert observed_pre_rope == [True]
    assert method.method_id == "vanilla_prefill"
    assert method.artifact_version == "2"
    assert built_generator is generator
    assert generator.layout is layout
    assert layout.pre_rope is True
    assert layout.key_position_encoding.value == "pre_rope"
    assert layout.rope_theta == 5_000_000.0
    assert layout.rope_rotary_dim == 128
    assert layout.shares_kv_storage is False
    assert layout.storage_layout == KVStorageLayout.SEPARATE_KEY_VALUE
