"""Private, import-safe native torch/vLLM bootstrap attestation target."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.metadata as importlib_metadata
import importlib.util
import os
import stat
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, NoReturn


_RECORD_TYPE: Final = "cachet.runtime_bootstrap_canary_native_loader.v1"
_SCHEMA_VERSION: Final = 1
_TORCH_DISTRIBUTION_VERSION: Final = "2.13.0+cu129"
_TORCH_SOURCE_VERSION: Final = "2.13.0+cu129"
_TORCH_CUDA_VERSION: Final = "12.9"
_VLLM_DISTRIBUTION_VERSION: Final = "0.27.1+cu129"
_VLLM_NATIVE_MODULE: Final = "vllm._C_stable_libtorch"
_TORCH_ORIGIN: Final = "lib/python3.11/site-packages/torch/__init__.py"
_VLLM_NATIVE_ORIGIN: Final = (
    "lib/python3.11/site-packages/vllm/_C_stable_libtorch.abi3.so"
)
_EXPECTED_GPU_NAMES: Final = frozenset({"NVIDIA L40S", "NVIDIA L4"})
_EXPECTED_GPU_CAPABILITY: Final = (8, 9)
_EXPECTED_GPU_CAPABILITY_TEXT: Final = "8.9"
_OFFLINE_ENVIRONMENT: Final = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1",
}
_BLOCKED_NETWORK_AUDIT_EVENTS: Final = frozenset(
    {
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.spawn",
        "os.system",
        "socket.bind",
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "socket.getnameinfo",
        "socket.sendmsg",
        "socket.sendto",
        "subprocess.Popen",
    }
)

_ProtocolExit = Callable[[dict[str, object]], NoReturn]
_SOURCE_FILE_LOADER: Final = importlib.machinery.SourceFileLoader
_EXTENSION_FILE_LOADER: Final = importlib.machinery.ExtensionFileLoader
_MODULE_FROM_SPEC: Final = importlib.util.module_from_spec
_SPEC_FROM_FILE_LOCATION: Final = importlib.util.spec_from_file_location


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _require_exact_regular_file(path: Path, *, label: str) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or path.resolve(strict=True) != path
    ):
        _fail(f"{label} is not an exact canonical regular file")


def _require_exact_directory(path: Path, *, label: str) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISDIR(info.st_mode)
        or path.resolve(strict=True) != path
    ):
        _fail(f"{label} is not an exact canonical directory")


def _exact_source_module_origin(
    module: object,
    expected: Path,
    *,
    label: str,
    module_name: str,
) -> str:
    origin = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None)
    if (
        type(origin) is not str
        or Path(origin) != expected
        or getattr(spec, "name", None) != module_name
        or getattr(spec, "origin", None) != str(expected)
        or type(loader) is not _SOURCE_FILE_LOADER
        or getattr(loader, "name", None) != module_name
        or getattr(loader, "path", None) != str(expected)
    ):
        _fail(f"{label} origin does not match the locked runtime")
    _require_exact_regular_file(expected, label=f"{label} origin")
    return expected.relative_to(Path(sys.prefix)).as_posix()


def _load_exact_vllm_native(site_packages: Path) -> tuple[object, str]:
    vllm_name = "vllm"
    vllm_path = site_packages / "vllm" / "__init__.py"
    native_path = site_packages / "vllm" / "_C_stable_libtorch.abi3.so"
    if vllm_name in sys.modules or _VLLM_NATIVE_MODULE in sys.modules:
        _fail("vLLM modules were preloaded")
    vllm_package = importlib.import_module(vllm_name)
    _exact_source_module_origin(
        vllm_package,
        vllm_path,
        label="vLLM package",
        module_name=vllm_name,
    )
    if _VLLM_NATIVE_MODULE in sys.modules:
        _fail("vLLM package preloaded the native module")
    _require_exact_regular_file(native_path, label="vLLM native module origin")
    loader = _EXTENSION_FILE_LOADER(_VLLM_NATIVE_MODULE, str(native_path))
    spec = _SPEC_FROM_FILE_LOCATION(
        _VLLM_NATIVE_MODULE,
        native_path,
        loader=loader,
    )
    if (
        spec is None
        or spec.name != _VLLM_NATIVE_MODULE
        or spec.origin != str(native_path)
        or type(spec.loader) is not _EXTENSION_FILE_LOADER
    ):
        _fail("vLLM native module spec differs")
    try:
        native_module = _MODULE_FROM_SPEC(spec)
        sys.modules[_VLLM_NATIVE_MODULE] = native_module
        loader.exec_module(native_module)
    except BaseException:
        sys.modules.pop(_VLLM_NATIVE_MODULE, None)
        raise
    if (
        getattr(native_module, "__spec__", None) is not spec
        or getattr(native_module, "__loader__", None) is not loader
        or getattr(native_module, "__file__", None) != str(native_path)
    ):
        _fail("vLLM native module loader provenance differs")
    return native_module, native_path.relative_to(Path(sys.prefix)).as_posix()


def _install_network_denial_audit() -> Callable[[], bool]:
    network_accessed = [False]

    def deny_network_or_process(event: str, _arguments: tuple[object, ...]) -> None:
        if event in _BLOCKED_NETWORK_AUDIT_EVENTS:
            network_accessed[0] = True
            _fail("network or child-process access is forbidden")

    sys.addaudithook(deny_network_or_process)
    return lambda: network_accessed[0]


def _exact_distribution_version(
    distribution_name: str,
    *,
    expected_version: str,
    site_packages: Path,
) -> str:
    distributions = list(importlib_metadata.distributions(name=distribution_name))
    if len(distributions) != 1:
        _fail(f"{distribution_name} distribution identity is ambiguous")
    distribution = distributions[0]
    distribution_root = Path(str(distribution.locate_file("")))
    if (
        distribution_root != site_packages
        or distribution_root.resolve(strict=True) != site_packages
        or distribution.version != expected_version
    ):
        _fail(f"{distribution_name} distribution provenance mismatch")
    return distribution.version


def _validate_arguments(arguments: Sequence[str]) -> tuple[str, str, Path]:
    if (
        len(arguments) != 2
        or any(type(argument) is not str for argument in arguments)
    ):
        _fail("native-loader target requires exactly two string arguments")
    expected_gpu_name, expected_ld_library_path = arguments
    if expected_gpu_name not in _EXPECTED_GPU_NAMES:
        _fail("native-loader GPU is outside the exact allowlist")
    runtime_root = Path(sys.prefix)
    if (
        not runtime_root.is_absolute()
        or runtime_root.resolve(strict=True) != runtime_root
    ):
        _fail("runtime root is not canonical")
    site_packages = runtime_root / "lib" / "python3.11" / "site-packages"
    torch_lib = site_packages / "torch" / "lib"
    ld_components = expected_ld_library_path.split(os.pathsep)
    if (
        not expected_ld_library_path
        or "\x00" in expected_ld_library_path
        or any(not component for component in ld_components)
        or ld_components[0] != str(torch_lib)
        or os.environ.get("LD_LIBRARY_PATH") != expected_ld_library_path
    ):
        _fail("LD_LIBRARY_PATH does not match the exact expected value")
    _require_exact_directory(site_packages, label="runtime site-packages")
    _require_exact_directory(torch_lib, label="runtime torch/lib")
    return expected_gpu_name, expected_ld_library_path, site_packages


def main(
    arguments: Sequence[str],
    *,
    protocol_exit: _ProtocolExit,
) -> NoReturn:
    """Load only the fixed native modules, validate them, and seal the record."""

    if not callable(protocol_exit):
        _fail("native-loader protocol finalizer is not callable")
    (
        expected_gpu_name,
        expected_ld_library_path,
        site_packages,
    ) = _validate_arguments(arguments)
    if any(
        os.environ.get(name) != value
        for name, value in _OFFLINE_ENVIRONMENT.items()
    ):
        _fail("native-loader offline environment is not exact")

    network_accessed = _install_network_denial_audit()
    torch = importlib.import_module("torch")
    vllm_native, vllm_native_origin = _load_exact_vllm_native(site_packages)

    torch_distribution_version = _exact_distribution_version(
        "torch",
        expected_version=_TORCH_DISTRIBUTION_VERSION,
        site_packages=site_packages,
    )
    vllm_distribution_version = _exact_distribution_version(
        "vllm",
        expected_version=_VLLM_DISTRIBUTION_VERSION,
        site_packages=site_packages,
    )
    torch_source_version = getattr(torch, "__version__", None)
    torch_version = getattr(torch, "version", None)
    torch_cuda_version = getattr(torch_version, "cuda", None)
    if (
        torch_distribution_version != _TORCH_DISTRIBUTION_VERSION
        or torch_source_version != _TORCH_SOURCE_VERSION
        or torch_cuda_version != _TORCH_CUDA_VERSION
        or vllm_distribution_version != _VLLM_DISTRIBUTION_VERSION
    ):
        _fail("native-loader distribution or source version mismatch")

    torch_origin = _exact_source_module_origin(
        torch,
        site_packages / "torch" / "__init__.py",
        label="torch",
        module_name="torch",
    )
    if getattr(vllm_native, "__name__", None) != _VLLM_NATIVE_MODULE:
        _fail("vLLM native module identity differs")
    if torch_origin != _TORCH_ORIGIN or vllm_native_origin != _VLLM_NATIVE_ORIGIN:
        _fail("native-loader relative origin mismatch")

    cuda = getattr(torch, "cuda", None)
    if cuda is None or cuda.is_available() is not True:
        _fail("CUDA is unavailable")
    cuda_device_count = cuda.device_count()
    gpu_name = cuda.get_device_name(0)
    gpu_capability = cuda.get_device_capability(0)
    if (
        type(cuda_device_count) is not int
        or cuda_device_count != 1
        or type(gpu_name) is not str
        or gpu_name != expected_gpu_name
        or gpu_capability != _EXPECTED_GPU_CAPABILITY
    ):
        _fail("CUDA device identity does not match the exact canary contract")

    tensor = torch.tensor(1.0, device="cuda") + 1.0
    cuda_tensor_result = tensor.item()
    cuda.synchronize()
    if type(cuda_tensor_result) is not float or cuda_tensor_result != 2.0:
        _fail("CUDA tensor probe result mismatch")
    if network_accessed():
        _fail("network access was observed")

    protocol_exit(
        {
            "benchmark_executed": False,
            "cuda_available": True,
            "cuda_device_count": cuda_device_count,
            "cuda_synchronized": True,
            "cuda_tensor_result": cuda_tensor_result,
            "gpu_compute_capability": _EXPECTED_GPU_CAPABILITY_TEXT,
            "gpu_name": gpu_name,
            "ld_library_path_sha256": hashlib.sha256(
                expected_ld_library_path.encode("utf-8")
            ).hexdigest(),
            "model_loaded": False,
            "network_accessed": False,
            "record_type": _RECORD_TYPE,
            "schema_version": _SCHEMA_VERSION,
            "torch_cuda_version": torch_cuda_version,
            "torch_distribution_version": torch_distribution_version,
            "torch_lib_is_first": True,
            "torch_origin": torch_origin,
            "torch_source_version": torch_source_version,
            "vllm_distribution_version": vllm_distribution_version,
            "vllm_native_module": _VLLM_NATIVE_MODULE,
            "vllm_native_origin": vllm_native_origin,
        }
    )
    _fail("native-loader protocol finalizer returned")


__all__ = ["main"]
