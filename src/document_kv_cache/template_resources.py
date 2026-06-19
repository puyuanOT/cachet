from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


PACKAGED_TEMPLATE_PACKAGE = "document_kv_cache.templates"
TEMPLATE_RESOURCE_RECORD_TYPE = "document_kv.template_resources.v1"
IGNORED_TEMPLATE_RESOURCE_NAMES = frozenset({"__init__.py"})
SUPPORTED_TEMPLATE_RESOURCE_SUFFIXES = frozenset({".md", ".yml"})


@dataclass(frozen=True, slots=True)
class TemplateResource:
    name: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")


def list_template_resources(prefix: str | None = None) -> tuple[TemplateResource, ...]:
    normalized_prefix = _normalized_prefix(prefix)
    root = resources.files(PACKAGED_TEMPLATE_PACKAGE)
    return tuple(
        resource
        for resource in _walk_template_resources(root, Path())
        if normalized_prefix is None or resource.name == normalized_prefix or resource.name.startswith(f"{normalized_prefix}/")
    )


def read_template_resource(name: str) -> str:
    resource_path = _resource_path_for_name(name)
    return resource_path.read_text(encoding="utf-8")


def extract_template_resources(
    output_dir: str | Path,
    *,
    prefix: str | None = None,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    target_root = Path(output_dir)
    if target_root.exists() and not target_root.is_dir():
        raise ValueError("output_dir must be a directory")
    target_root.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for resource in list_template_resources(prefix):
        source_path = _resource_path_for_name(resource.name)
        target_path = target_root / resource.name
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"{target_path} already exists; pass overwrite=True to replace it")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open("rb") as source_file:
            with target_path.open("wb") as target_file:
                shutil.copyfileobj(source_file, target_file)
        written_paths.append(target_path)
    return tuple(written_paths)


def template_resources_to_record(resources_: Sequence[TemplateResource]) -> dict[str, object]:
    return {
        "record_type": TEMPLATE_RESOURCE_RECORD_TYPE,
        "resources": [
            {
                "name": resource.name,
                "size_bytes": resource.size_bytes,
            }
            for resource in resources_
        ],
    }


def _walk_template_resources(root: resources.abc.Traversable, relative_root: Path) -> tuple[TemplateResource, ...]:
    items = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.name in IGNORED_TEMPLATE_RESOURCE_NAMES:
            continue
        relative_path = relative_root / child.name
        if child.is_dir():
            items.extend(_walk_template_resources(child, relative_path))
            continue
        if child.is_file() and child.name.endswith(tuple(SUPPORTED_TEMPLATE_RESOURCE_SUFFIXES)):
            items.append(TemplateResource(name=relative_path.as_posix(), size_bytes=len(child.read_bytes())))
    return tuple(items)


def _resource_path_for_name(name: str) -> resources.abc.Traversable:
    normalized_name = _validated_resource_name(name)
    by_name = {resource.name: resource for resource in list_template_resources()}
    try:
        resource = by_name[normalized_name]
    except KeyError as exc:
        raise ValueError(f"Unknown template resource {name!r}") from exc
    root = resources.files(PACKAGED_TEMPLATE_PACKAGE)
    path = root
    for part in normalized_name.split("/"):
        path = path.joinpath(part)
    if not path.is_file():
        raise ValueError(f"Template resource {name!r} is not a file")
    return path


def _normalized_prefix(prefix: str | None) -> str | None:
    if prefix is None:
        return None
    return _validated_resource_name(prefix)


def _validated_resource_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("template resource name must be non-empty")
    path = Path(name)
    if path.is_absolute():
        raise ValueError("template resource name must be relative")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise ValueError("template resource name cannot be empty or escape the template root")
    return Path(*parts).as_posix()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List, show, or extract packaged Document KV Cache templates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List packaged template resources.")
    list_parser.add_argument("--prefix", help="Optional relative resource prefix, such as databricks/storage-benchmark.")
    list_parser.add_argument("--output-json", action="store_true", help="Print a JSON resource list.")

    show_parser = subparsers.add_parser("show", help="Print one packaged template resource.")
    show_parser.add_argument("name", help="Relative template resource name.")

    extract_parser = subparsers.add_parser("extract", help="Copy packaged template resources to a directory.")
    extract_parser.add_argument("--output-dir", required=True, help="Directory to receive templates.")
    extract_parser.add_argument("--prefix", help="Optional relative resource prefix, such as databricks/storage-benchmark.")
    extract_parser.add_argument("--overwrite", action="store_true", help="Replace existing files.")

    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            template_resources = list_template_resources(args.prefix)
            if args.output_json:
                print(json.dumps(template_resources_to_record(template_resources), indent=2, sort_keys=True))
            else:
                for resource in template_resources:
                    print(resource.name)
        elif args.command == "show":
            print(read_template_resource(args.name), end="")
        elif args.command == "extract":
            written_paths = extract_template_resources(
                args.output_dir,
                prefix=args.prefix,
                overwrite=args.overwrite,
            )
            for path in written_paths:
                print(path.as_posix())
        else:  # pragma: no cover - argparse enforces this branch.
            raise ValueError(f"Unsupported command {args.command!r}")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
