from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cachet_pyproject_packages_sglang_adapter_without_raw_engine_extra():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "cachet-kv"
    assert "sglang_kv_injection" in {
        package["include"]
        for package in pyproject["tool"]["poetry"]["packages"]
    }
    assert "sglang" not in pyproject["project"]["optional-dependencies"]
    assert {
        "path": "src/sglang_kv_injection/py.typed",
        "format": ["sdist", "wheel"],
    } in pyproject["tool"]["poetry"]["include"]


def test_sglang_adapter_readme_keeps_engine_boundary_in_cachet_monorepo():
    package_readme = (REPO_ROOT / "src" / "sglang_kv_injection" / "README.md").read_text(encoding="utf-8").lower()
    cachet_adapters_readme = (REPO_ROOT / "src" / "cachet" / "adapters" / "README.md").read_text(encoding="utf-8")
    compact_cachet_adapters_readme = " ".join(cachet_adapters_readme.split())

    assert "patched sglang runtime" in package_readme
    assert "keep this package close to sglang internals" in package_readme
    assert "sglang.py` aliases the compatibility package `sglang_kv_injection`" in cachet_adapters_readme
    assert "`cachet.adapters.sglang.probe`" in cachet_adapters_readme
    assert "same module objects as the vendored compatibility paths" in compact_cachet_adapters_readme
    for out_of_scope_term in (
        "document retrieval",
        "cache storage",
        "cpu assembly",
        "scheduling",
        "lora routing",
    ):
        assert out_of_scope_term in package_readme
