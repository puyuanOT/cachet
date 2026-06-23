import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cachet_pyproject_packages_vllm_adapter_without_raw_engine_extra():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "document-kv-cache"
    assert "vllm_kv_injection" in {
        package["include"]
        for package in pyproject["tool"]["poetry"]["packages"]
    }
    assert "vllm" not in pyproject["project"]["optional-dependencies"]
    assert {
        "path": "src/vllm_kv_injection/py.typed",
        "format": ["sdist", "wheel"],
    } in pyproject["tool"]["poetry"]["include"]


def test_vllm_adapter_readme_keeps_engine_boundary_in_cachet_monorepo():
    package_readme = (REPO_ROOT / "src" / "vllm_kv_injection" / "README.md").read_text(encoding="utf-8").lower()
    cachet_adapters_readme = (REPO_ROOT / "src" / "cachet" / "adapters" / "README.md").read_text(encoding="utf-8")
    compact_cachet_adapters_readme = " ".join(cachet_adapters_readme.split())

    assert "patched vllm runtime" in package_readme
    assert "keep this package close to vllm internals" in package_readme
    assert "vllm.py` aliases the compatibility package `vllm_kv_injection`" in cachet_adapters_readme
    assert "`cachet.adapters.vllm.probe`" in cachet_adapters_readme
    assert "same module objects as the vendored compatibility paths" in compact_cachet_adapters_readme
    for out_of_scope_term in (
        "document retrieval",
        "cache storage",
        "cpu assembly",
        "scheduling",
        "lora routing",
    ):
        assert out_of_scope_term in package_readme


def test_vllm_package_root_import_does_not_import_runtime_modules():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import vllm_kv_injection; "
                "print(json.dumps({"
                "'vllm': 'vllm' in sys.modules, "
                "'dynamic': 'vllm_kv_injection.vllm_dynamic_connector' in sys.modules, "
                "'native_provider': 'vllm_kv_injection.vllm_native_provider' in sys.modules, "
                "'probe': 'vllm_kv_injection.probe' in sys.modules"
                "}, sort_keys=True))"
            ),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "dynamic": False,
        "native_provider": False,
        "probe": False,
        "vllm": False,
    }


def test_probe_fixtures_import_does_not_import_vllm_runtime_modules():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import document_kv_cache.probe_fixtures; "
                "print(json.dumps({"
                "'vllm': 'vllm' in sys.modules, "
                "'dynamic': 'vllm_kv_injection.vllm_dynamic_connector' in sys.modules, "
                "'native_provider': 'vllm_kv_injection.vllm_native_provider' in sys.modules, "
                "'layer_mapping': 'vllm_kv_injection.vllm_layer_mapping' in sys.modules"
                "}, sort_keys=True))"
            ),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "dynamic": False,
        "layer_mapping": True,
        "native_provider": False,
        "vllm": False,
    }


def test_vllm_runtime_preflight_import_does_not_import_runtime_modules():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import vllm_kv_injection.vllm_runtime_preflight; "
                "print(json.dumps({"
                "'vllm': 'vllm' in sys.modules, "
                "'dynamic': 'vllm_kv_injection.vllm_dynamic_connector' in sys.modules, "
                "'native_provider': 'vllm_kv_injection.vllm_native_provider' in sys.modules, "
                "'runtime_contract': 'vllm_kv_injection.vllm_runtime_contract' in sys.modules"
                "}, sort_keys=True))"
            ),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "dynamic": False,
        "native_provider": False,
        "runtime_contract": True,
        "vllm": False,
    }


def test_cachet_vllm_adapter_facade_import_does_not_import_native_provider_modules():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import cachet.adapters.vllm; "
                "print(json.dumps({"
                "'vllm': 'vllm' in sys.modules, "
                "'native_provider': 'vllm_kv_injection.vllm_native_provider' in sys.modules, "
                "'probe': 'vllm_kv_injection.probe' in sys.modules"
                "}, sort_keys=True))"
            ),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "native_provider": False,
        "probe": False,
        "vllm": False,
    }
