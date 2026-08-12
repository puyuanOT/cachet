"""Opt-in real Transformers KV generation example.

This downloads model weights and may require a GPU. It is not part of the
CPU-local quickstart.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path


repo_src = Path(__file__).resolve().parents[1] / "src"
if repo_src.is_dir():
    sys.path.insert(0, str(repo_src))

from document_kv_cache.artifact_identity import method_config_digest  # noqa: E402
from document_kv_cache.cache import ChunkCache  # noqa: E402
from document_kv_cache.manifest import InMemoryManifestStore  # noqa: E402
from document_kv_cache.materializer import KVMaterializer  # noqa: E402
from document_kv_cache.methods import method_spec  # noqa: E402
from document_kv_cache.model_profiles import layout_for_model  # noqa: E402
from document_kv_cache.models import CacheGenerationMethod  # noqa: E402
from document_kv_cache.storage import DiskRangeReader  # noqa: E402
from document_kv_cache.transformers_generator import (  # noqa: E402
    TransformersKVChunkGenerator,
    TransformersKVGeneratorConfig,
)
from document_kv_cache.workflow import (  # noqa: E402
    CacheBuildConfig,
    DocumentKVWorkflow,
    SourceDocument,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--text", default="Cachet stores reusable document KV.")
    parser.add_argument("--output", default="databricks-runs/transformers-example.kvpack")
    args = parser.parse_args()

    method = method_spec(CacheGenerationMethod.VANILLA_PREFILL)
    layout = layout_for_model(args.model_id, dtype=args.torch_dtype)
    generator = TransformersKVChunkGenerator.from_pretrained(
        TransformersKVGeneratorConfig(
            model_id=args.model_id,
            tokenizer_id=args.model_id,
            device=args.device,
            torch_dtype=args.torch_dtype,
            model_kwargs={"revision": args.model_revision},
            tokenizer_kwargs={"revision": args.tokenizer_revision},
        ),
        layout=layout,
    )
    config = CacheBuildConfig(
        model_id=layout.model_id,
        lora_id=layout.lora_id,
        prompt_template_version="transformers-example-v1",
        dtype=layout.dtype,
        layout_version=layout.layout_version,
        cache_method=method.method_id,
        storage_layout=layout.storage_layout,
        payload_axis_order=layout.payload_axis_order,
        method_version=method.artifact_version,
        method_config_digest=method_config_digest(
            {
                "add_special_tokens": False,
                "torch_dtype": args.torch_dtype,
            }
        ),
        model_revision=args.model_revision,
        tokenizer_id=args.model_id,
        tokenizer_revision=args.tokenizer_revision,
        generator_family="transformers",
        generator_version=metadata.version("transformers"),
        artifact_format_id=method.artifact_format.format_id,
        artifact_format_version=method.artifact_format.version,
    )
    workflow = DocumentKVWorkflow(
        manifest=InMemoryManifestStore(),
        materializer=KVMaterializer(
            cache=ChunkCache(cpu_max_bytes=1),
            reader=DiskRangeReader(),
        ),
    )
    result = workflow.generate_cache(
        documents=(
            SourceDocument.from_text(document_id="example-document", text=args.text),
        ),
        generator=generator,
        config=config,
        shard_uri=args.output,
        require_registered_method=True,
    )
    print(
        json.dumps(
            {
                "artifact_id": result.artifact_id,
                "cache_method": result.cache_method,
                "chunks": result.chunk_count,
                "output": str(Path(args.output)),
                "total_bytes": result.total_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
