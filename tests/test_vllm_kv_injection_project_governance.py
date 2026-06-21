from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cachet_pyproject_packages_vllm_adapter_with_exact_optional_engine_pin():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "document-kv-cache"
    assert "vllm_kv_injection" in {
        package["include"]
        for package in pyproject["tool"]["poetry"]["packages"]
    }
    assert pyproject["project"]["optional-dependencies"]["vllm"] == [
        "vllm==0.23.0; python_version < '3.15'"
    ]
    assert {
        "path": "src/vllm_kv_injection/py.typed",
        "format": ["sdist", "wheel"],
    } in pyproject["tool"]["poetry"]["include"]


def test_vllm_adapter_readme_keeps_engine_boundary_in_cachet_monorepo():
    package_readme = (REPO_ROOT / "src" / "vllm_kv_injection" / "README.md").read_text(encoding="utf-8").lower()
    cachet_adapters_readme = (REPO_ROOT / "src" / "cachet" / "adapters" / "README.md").read_text(encoding="utf-8")

    assert "patched vllm runtime" in package_readme
    assert "keep this package close to vllm internals" in package_readme
    assert "vllm.py` aliases the compatibility package `vllm_kv_injection`" in cachet_adapters_readme
    for out_of_scope_term in (
        "document retrieval",
        "cache storage",
        "cpu assembly",
        "scheduling",
        "lora routing",
    ):
        assert out_of_scope_term in package_readme
