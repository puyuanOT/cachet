import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from document_kv_cache.template_resources import (
    PACKAGED_TEMPLATE_PACKAGE,
    SUPPORTED_TEMPLATE_RESOURCE_SUFFIXES,
    TEMPLATE_RESOURCE_RECORD_TYPE,
    TemplateResource,
    extract_template_resources,
    list_template_resources,
    main,
    read_template_resource,
    template_resources_to_record,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TEMPLATE_RESOURCE_NAMES = [
    "README.md",
    "databricks/README.md",
    "databricks/databricks.yml",
    "databricks/engine-probe/README.md",
    "databricks/engine-probe/databricks.yml",
    "databricks/storage-benchmark/README.md",
    "databricks/storage-benchmark/databricks.yml",
    "databricks/vllm-smoke/README.md",
    "databricks/vllm-smoke/databricks.yml",
]


def test_list_template_resources_reports_packaged_databricks_templates():
    resources = list_template_resources()
    names = [resource.name for resource in resources]

    assert PACKAGED_TEMPLATE_PACKAGE == "document_kv_cache.templates"
    assert SUPPORTED_TEMPLATE_RESOURCE_SUFFIXES == {".md", ".yml"}
    assert names == EXPECTED_TEMPLATE_RESOURCE_NAMES
    assert all(resource.size_bytes > 0 for resource in resources)


def test_template_resources_can_be_filtered_read_and_serialized():
    resources = list_template_resources("databricks/storage-benchmark")
    record = template_resources_to_record(resources)
    bundle_text = read_template_resource("databricks/storage-benchmark/databricks.yml")

    assert [resource.name for resource in resources] == [
        "databricks/storage-benchmark/README.md",
        "databricks/storage-benchmark/databricks.yml",
    ]
    assert record["record_type"] == TEMPLATE_RESOURCE_RECORD_TYPE
    assert record["resources"][0]["name"] == "databricks/storage-benchmark/README.md"
    assert "document-kv-storage-benchmark" in bundle_text


def test_packaged_databricks_readme_explains_installed_wheel_extraction():
    readme_text = read_template_resource("databricks/README.md")

    assert "document-kv-templates list --prefix databricks" in readme_text
    assert "document-kv-templates extract" in readme_text
    assert "--prefix databricks" in readme_text
    assert "--output-dir ./document-kv-templates" in readme_text
    assert "document-kv-templates/databricks/" in readme_text
    assert "document-kv-templates/databricks/storage-benchmark/" in readme_text
    assert "document-kv-templates/databricks/engine-probe/" in readme_text
    assert "document-kv-templates/databricks/vllm-smoke/" in readme_text


def test_template_resource_validates_names():
    with pytest.raises(ValueError, match="relative"):
        read_template_resource("/databricks/databricks.yml")

    with pytest.raises(ValueError, match="escape"):
        read_template_resource("../pyproject.toml")

    with pytest.raises(ValueError, match="Unknown"):
        read_template_resource("databricks/missing.yml")

    with pytest.raises(ValueError, match="size_bytes"):
        TemplateResource(name="databricks/databricks.yml", size_bytes=-1)


def test_extract_template_resources_preserves_relative_paths_and_requires_overwrite(tmp_path):
    written = extract_template_resources(tmp_path / "templates", prefix="databricks/storage-benchmark")

    assert sorted(path.relative_to(tmp_path / "templates").as_posix() for path in written) == [
        "databricks/storage-benchmark/README.md",
        "databricks/storage-benchmark/databricks.yml",
    ]
    assert "document-kv-storage-benchmark" in (
        tmp_path / "templates" / "databricks" / "storage-benchmark" / "databricks.yml"
    ).read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        extract_template_resources(tmp_path / "templates", prefix="databricks/storage-benchmark")

    overwritten = extract_template_resources(
        tmp_path / "templates",
        prefix="databricks/storage-benchmark",
        overwrite=True,
    )
    assert len(overwritten) == len(written)


def test_template_resources_cli_lists_shows_and_extracts(tmp_path, capsys):
    assert main(["list", "--prefix", "databricks/storage-benchmark"]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert listed == [
        "databricks/storage-benchmark/README.md",
        "databricks/storage-benchmark/databricks.yml",
    ]

    assert main(["list", "--prefix", "databricks/storage-benchmark", "--output-json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["record_type"] == TEMPLATE_RESOURCE_RECORD_TYPE

    assert main(["show", "databricks/storage-benchmark/databricks.yml"]) == 0
    assert "document-kv-storage-benchmark" in capsys.readouterr().out

    output_dir = tmp_path / "extracted"
    assert main(["extract", "--prefix", "databricks/storage-benchmark", "--output-dir", str(output_dir)]) == 0
    extracted = capsys.readouterr().out.splitlines()
    assert extracted == [
        (output_dir / "databricks" / "storage-benchmark" / "README.md").as_posix(),
        (output_dir / "databricks" / "storage-benchmark" / "databricks.yml").as_posix(),
    ]


def test_template_resources_module_executes_with_python_m(tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.template_resources",
            "list",
            "--prefix",
            "databricks/storage-benchmark",
            "--output-json",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    record = json.loads(completed.stdout)
    assert completed.stderr == ""
    assert record["record_type"] == TEMPLATE_RESOURCE_RECORD_TYPE
    assert [resource["name"] for resource in record["resources"]] == [
        "databricks/storage-benchmark/README.md",
        "databricks/storage-benchmark/databricks.yml",
    ]
