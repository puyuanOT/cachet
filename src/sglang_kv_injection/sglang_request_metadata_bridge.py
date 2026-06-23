"""Runtime bridge from SGLang request metadata to HiCache storage calls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import functools
import importlib
import inspect
import threading
from typing import Any

DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_RECORD_TYPE = (
    "sglang_kv_injection.request_metadata_bridge.v1"
)
DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SCHEMA_VERSION = 1
DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SOURCE = (
    "sglang_kv_injection.sglang_request_metadata_bridge"
)

_SCHEDULER_MODULE = "sglang.srt.managers.scheduler"
_CACHE_CONTROLLER_MODULE = "sglang.srt.managers.cache_controller"
_SCHEDULER_CLASS = "Scheduler"
_CACHE_CONTROLLER_CLASS = "HiCacheController"
_HICACHE_STORAGE_EXTRA_INFO_NAME = "HiCacheStorageExtraInfo"
_PATCH_MARKER = "__document_kv_request_metadata_bridge_patched__"
_ORIGINAL_ATTR = "__document_kv_request_metadata_bridge_original__"
_REQUEST_METADATA_REGISTRY_ATTR = "_document_kv_request_metadata_by_rid"
_OPERATION_EXTRA_INFO_ATTR = "document_kv_extra_info"
_SGLANG_REQ_BACKREF_KEY = "__req__"
_SGLANG_CUSTOM_PARAMS_KEY = "custom_params"
_SGLANG_KV_TRANSFER_PARAMS_KEY = "kv_transfer_params"
_THREAD_CONTEXT = threading.local()
_MISSING = object()

__all__ = [
    "DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_RECORD_TYPE",
    "DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SCHEMA_VERSION",
    "DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SOURCE",
    "SGLangRequestMetadataBridgeStatus",
    "install_sglang_request_metadata_bridge",
    "sglang_request_metadata_bridge_status_to_record",
]


@dataclass(frozen=True, slots=True)
class SGLangRequestMetadataBridgeStatus:
    """Status for Cachet's runtime request-metadata bridge patch."""

    installed: bool
    scheduler_prefetch_patched: bool = False
    controller_prefetch_patched: bool = False
    hicache_storage_extra_info_factory_patched: bool = False
    storage_hit_query_patched: bool = False
    page_transfer_patched: bool = False
    patched_modules: tuple[str, ...] = ()
    error: str | None = None


def install_sglang_request_metadata_bridge() -> SGLangRequestMetadataBridgeStatus:
    """Install Cachet's SGLang request-to-HiCache metadata bridge.

    The pinned SGLang OpenAI path stores request `custom_params` on
    `req.sampling_params`, while HiCache storage calls receive only
    `HiCacheStorageExtraInfo`. This bridge keeps the version-sensitive wiring in
    one place: it registers Cachet `custom_params` by request id at scheduler
    prefetch time, attaches them to the corresponding prefetch operation before
    SGLang queues it, and injects them into `HiCacheStorageExtraInfo.extra_info`
    while SGLang performs hit queries and page transfers.
    """

    try:
        scheduler_module = importlib.import_module(_SCHEDULER_MODULE)
        cache_controller_module = importlib.import_module(_CACHE_CONTROLLER_MODULE)
        scheduler_cls = getattr(scheduler_module, _SCHEDULER_CLASS)
        controller_cls = getattr(cache_controller_module, _CACHE_CONTROLLER_CLASS)
        hicache_storage_extra_info = getattr(
            cache_controller_module,
            _HICACHE_STORAGE_EXTRA_INFO_NAME,
        )
    except Exception as exc:
        return _bridge_status_error(f"{type(exc).__name__}: {exc}")

    try:
        scheduler_patched = _patch_scheduler_prefetch(scheduler_cls)
        controller_patched = _patch_controller_prefetch(controller_cls)
        extra_info_patched = _patch_hicache_storage_extra_info_factory(
            cache_controller_module,
            hicache_storage_extra_info,
        )
        storage_hit_query_patched = _patch_operation_context_method(controller_cls, "_storage_hit_query")
        page_transfer_patched = _patch_operation_context_method(controller_cls, "_page_transfer")
    except Exception as exc:
        return _bridge_status_error(f"{type(exc).__name__}: {exc}")

    installed = all(
        (
            scheduler_patched,
            controller_patched,
            extra_info_patched,
            storage_hit_query_patched,
            page_transfer_patched,
        )
    )
    return SGLangRequestMetadataBridgeStatus(
        installed=installed,
        scheduler_prefetch_patched=scheduler_patched,
        controller_prefetch_patched=controller_patched,
        hicache_storage_extra_info_factory_patched=extra_info_patched,
        storage_hit_query_patched=storage_hit_query_patched,
        page_transfer_patched=page_transfer_patched,
        patched_modules=(_SCHEDULER_MODULE, _CACHE_CONTROLLER_MODULE) if installed else (),
    )


def sglang_request_metadata_bridge_status_to_record(
    status: SGLangRequestMetadataBridgeStatus | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a request-metadata bridge status for runtime preflight evidence."""

    if isinstance(status, Mapping):
        return dict(status)
    if status is None:
        status = SGLangRequestMetadataBridgeStatus(
            installed=False,
            error="request metadata bridge was not installed",
        )
    record: dict[str, Any] = {
        "record_type": DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_RECORD_TYPE,
        "schema_version": DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SCHEMA_VERSION,
        "source": DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SOURCE,
        "installed": status.installed,
        "scheduler_prefetch_patched": status.scheduler_prefetch_patched,
        "controller_prefetch_patched": status.controller_prefetch_patched,
        "hicache_storage_extra_info_factory_patched": (
            status.hicache_storage_extra_info_factory_patched
        ),
        "storage_hit_query_patched": status.storage_hit_query_patched,
        "page_transfer_patched": status.page_transfer_patched,
        "patched_modules": sorted(set(status.patched_modules)),
    }
    if status.error:
        record["error"] = status.error
    record["ok"] = _request_metadata_bridge_installed(record)
    return record


def _bridge_status_error(error: str) -> SGLangRequestMetadataBridgeStatus:
    return SGLangRequestMetadataBridgeStatus(installed=False, error=error)


def _patch_scheduler_prefetch(scheduler_cls: type) -> bool:
    original = getattr(scheduler_cls, "_prefetch_kvcache", None)
    if not callable(original):
        raise TypeError("SGLang Scheduler._prefetch_kvcache is not callable")
    if getattr(original, _PATCH_MARKER, False) is True:
        return True

    @functools.wraps(original)
    def patched_prefetch_kvcache(self: object, req: object) -> object:
        registration = _register_request_metadata_from_req(self, req)
        try:
            return original(self, req)
        finally:
            if registration is not None:
                controller, request_id, metadata = registration
                _discard_unconsumed_request_metadata(controller, request_id, metadata)

    setattr(patched_prefetch_kvcache, _PATCH_MARKER, True)
    setattr(patched_prefetch_kvcache, _ORIGINAL_ATTR, original)
    setattr(scheduler_cls, "_prefetch_kvcache", patched_prefetch_kvcache)
    return True


def _patch_controller_prefetch(controller_cls: type) -> bool:
    original = getattr(controller_cls, "prefetch", None)
    if not callable(original):
        raise TypeError("SGLang HiCacheController.prefetch is not callable")
    if getattr(original, _PATCH_MARKER, False) is True:
        return True
    if not _signature_has_parameters(
        original,
        ("self", "request_id", "host_indices", "new_input_tokens", "last_hash", "prefix_keys"),
    ):
        raise TypeError("SGLang HiCacheController.prefetch signature is not supported")

    @functools.wraps(original)
    def patched_prefetch(
        self: object,
        request_id: str,
        host_indices: object,
        new_input_tokens: list[int],
        last_hash: str | None = None,
        prefix_keys: list[str] | None = None,
    ) -> object:
        request_metadata = _pop_request_metadata(self, request_id)
        if request_metadata is None:
            return original(
                self,
                request_id,
                host_indices,
                new_input_tokens,
                last_hash,
                prefix_keys,
            )
        prefetch_queue = getattr(self, "prefetch_queue", None)
        original_put = getattr(prefetch_queue, "put", None)
        if not callable(original_put):
            raise TypeError("SGLang HiCacheController.prefetch_queue.put is not callable")

        def put_with_metadata(item: object, *args: object, **kwargs: object) -> object:
            if getattr(item, "request_id", None) == request_id:
                setattr(item, _OPERATION_EXTRA_INFO_ATTR, request_metadata)
            return original_put(item, *args, **kwargs)

        try:
            setattr(prefetch_queue, "put", put_with_metadata)
        except Exception as exc:
            raise TypeError("SGLang HiCacheController.prefetch_queue.put cannot be patched") from exc
        try:
            return original(
                self,
                request_id,
                host_indices,
                new_input_tokens,
                last_hash,
                prefix_keys,
            )
        finally:
            setattr(prefetch_queue, "put", original_put)

    setattr(patched_prefetch, _PATCH_MARKER, True)
    setattr(patched_prefetch, _ORIGINAL_ATTR, original)
    setattr(controller_cls, "prefetch", patched_prefetch)
    return True


def _patch_hicache_storage_extra_info_factory(
    cache_controller_module: object,
    original_factory: object,
) -> bool:
    if getattr(original_factory, _PATCH_MARKER, False) is True:
        return True
    if not callable(original_factory):
        raise TypeError("SGLang HiCacheStorageExtraInfo is not callable")

    @functools.wraps(original_factory)
    def cachet_hicache_storage_extra_info(*args: object, **kwargs: object) -> object:
        request_metadata = _current_operation_request_metadata()
        if request_metadata is not None:
            extra_info = _explicit_extra_info(args=args, kwargs=kwargs)
            if extra_info is _MISSING:
                kwargs["extra_info"] = request_metadata
            elif extra_info is None:
                if len(args) >= 2:
                    args = (args[0], request_metadata, *args[2:])
                else:
                    kwargs["extra_info"] = request_metadata
        return original_factory(*args, **kwargs)

    setattr(cachet_hicache_storage_extra_info, _PATCH_MARKER, True)
    setattr(cachet_hicache_storage_extra_info, _ORIGINAL_ATTR, original_factory)
    setattr(cache_controller_module, _HICACHE_STORAGE_EXTRA_INFO_NAME, cachet_hicache_storage_extra_info)
    return True


def _patch_operation_context_method(controller_cls: type, method_name: str) -> bool:
    original = getattr(controller_cls, method_name, None)
    if not callable(original):
        raise TypeError(f"SGLang HiCacheController.{method_name} is not callable")
    if getattr(original, _PATCH_MARKER, False) is True:
        return True

    @functools.wraps(original)
    def patched_method(self: object, operation: object, *args: object, **kwargs: object) -> object:
        previous = getattr(_THREAD_CONTEXT, "operation", None)
        _THREAD_CONTEXT.operation = operation
        try:
            return original(self, operation, *args, **kwargs)
        finally:
            if previous is None:
                try:
                    delattr(_THREAD_CONTEXT, "operation")
                except AttributeError:
                    pass
            else:
                _THREAD_CONTEXT.operation = previous

    setattr(patched_method, _PATCH_MARKER, True)
    setattr(patched_method, _ORIGINAL_ATTR, original)
    setattr(controller_cls, method_name, patched_method)
    return True


def _register_request_metadata_from_req(
    scheduler: object,
    req: object,
) -> tuple[object, str, Mapping[str, object]] | None:
    metadata = _request_metadata_from_req(req)
    if metadata is None:
        return None
    request_id = getattr(req, "rid", None)
    if not isinstance(request_id, str) or not request_id:
        return None
    controller = _cache_controller_from_scheduler(scheduler)
    if controller is None:
        return None
    registry = getattr(controller, _REQUEST_METADATA_REGISTRY_ATTR, None)
    if not isinstance(registry, dict):
        registry = {}
        setattr(controller, _REQUEST_METADATA_REGISTRY_ATTR, registry)
    registry[request_id] = metadata
    return controller, request_id, metadata


def _request_metadata_from_req(req: object) -> dict[str, object] | None:
    sampling_params = getattr(req, "sampling_params", None)
    custom_params = getattr(sampling_params, _SGLANG_CUSTOM_PARAMS_KEY, None)
    if not isinstance(custom_params, Mapping):
        return None
    kv_transfer_params = custom_params.get(_SGLANG_KV_TRANSFER_PARAMS_KEY)
    if not isinstance(kv_transfer_params, Mapping):
        return None
    filtered_custom_params = {
        key: value
        for key, value in custom_params.items()
        if key != _SGLANG_REQ_BACKREF_KEY
    }
    return {_SGLANG_CUSTOM_PARAMS_KEY: filtered_custom_params}


def _cache_controller_from_scheduler(scheduler: object) -> object | None:
    tree_cache = getattr(scheduler, "tree_cache", None)
    if tree_cache is None:
        return None
    controller = getattr(tree_cache, "cache_controller", None)
    return controller


def _pop_request_metadata(controller: object, request_id: object) -> Mapping[str, object] | None:
    if not isinstance(request_id, str) or not request_id:
        return None
    registry = getattr(controller, _REQUEST_METADATA_REGISTRY_ATTR, None)
    if not isinstance(registry, dict):
        return None
    metadata = registry.pop(request_id, None)
    if not isinstance(metadata, Mapping):
        return None
    return dict(metadata)


def _discard_unconsumed_request_metadata(
    controller: object,
    request_id: str,
    metadata: Mapping[str, object],
) -> None:
    registry = getattr(controller, _REQUEST_METADATA_REGISTRY_ATTR, None)
    if not isinstance(registry, dict):
        return
    if registry.get(request_id) is metadata:
        registry.pop(request_id, None)


def _current_operation_request_metadata() -> Mapping[str, object] | None:
    operation = getattr(_THREAD_CONTEXT, "operation", None)
    if operation is None:
        return None
    metadata = getattr(operation, _OPERATION_EXTRA_INFO_ATTR, None)
    if not isinstance(metadata, Mapping):
        return None
    return metadata


def _explicit_extra_info(
    *,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> object:
    if len(args) >= 2:
        return args[1]
    return kwargs.get("extra_info", _MISSING)


def _signature_has_parameters(handler: object, names: tuple[str, ...]) -> bool:
    try:
        parameters = tuple(inspect.signature(handler).parameters)
    except (TypeError, ValueError):
        return False
    return parameters[: len(names)] == names


def _request_metadata_bridge_installed(record: Mapping[str, Any]) -> bool:
    return (
        record.get("installed") is True
        and record.get("scheduler_prefetch_patched") is True
        and record.get("controller_prefetch_patched") is True
        and record.get("hicache_storage_extra_info_factory_patched") is True
        and record.get("storage_hit_query_patched") is True
        and record.get("page_transfer_patched") is True
    )
