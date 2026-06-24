"""Runtime bridge from SGLang request metadata to HiCache storage calls."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import functools
import importlib
import inspect
import logging
import threading
from typing import Any

DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_RECORD_TYPE = (
    "sglang_kv_injection.request_metadata_bridge.v1"
)
DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SCHEMA_VERSION = 3
DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SOURCE = (
    "sglang_kv_injection.sglang_request_metadata_bridge"
)

_SCHEDULER_MODULE = "sglang.srt.managers.scheduler"
_CACHE_CONTROLLER_MODULE = "sglang.srt.managers.cache_controller"
_SCHEDULER_CLASS = "Scheduler"
_CACHE_CONTROLLER_CLASS = "HiCacheController"
_PREFETCH_OPERATION_CLASS = "PrefetchOperation"
_HICACHE_STORAGE_EXTRA_INFO_NAME = "HiCacheStorageExtraInfo"
_PATCH_MARKER = "__document_kv_request_metadata_bridge_patched__"
_ORIGINAL_ATTR = "__document_kv_request_metadata_bridge_original__"
_HASH_TRACKING_PATCH_MARKER = "__document_kv_hicache_hash_tracking_patched__"
_HASH_TRACKING_ORIGINAL_ATTR = "__document_kv_hicache_hash_tracking_original__"
_REQUEST_METADATA_REGISTRY_ATTR = "_document_kv_request_metadata_by_rid"
_OPERATION_EXTRA_INFO_ATTR = "document_kv_extra_info"
_SGLANG_REQ_BACKREF_KEY = "__req__"
_SGLANG_CUSTOM_PARAMS_KEY = "custom_params"
_SGLANG_KV_TRANSFER_PARAMS_KEY = "kv_transfer_params"
_DOCUMENT_KV_REQUEST_ID_PARAM = "document_kv.request_id"
_DOCUMENT_KV_HANDOFF_JSON_PARAM = "document_kv.handoff_json"
_DOCUMENT_KV_HANDOFF_RECORD_PARAM = "document_kv.handoff_record"
_DOCUMENT_KV_PAYLOAD_URI_PARAM = "document_kv.payload_uri"
_DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM = "document_kv.sglang_hicache_page_keys"
DOCUMENT_KV_SGLANG_HICACHE_LAST_HASH_EXTRA_INFO_KEY = "document_kv.sglang_hicache_last_hash"
_THREAD_CONTEXT = threading.local()
_MISSING = object()
_LOGGER = logging.getLogger(__name__)

__all__ = [
    "DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_RECORD_TYPE",
    "DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SCHEMA_VERSION",
    "DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SOURCE",
    "DOCUMENT_KV_SGLANG_HICACHE_LAST_HASH_EXTRA_INFO_KEY",
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
    controller_hash_tracking_patched: bool = False
    prefetch_operation_patched: bool = False
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
        prefetch_operation_cls = getattr(cache_controller_module, _PREFETCH_OPERATION_CLASS)
        hicache_storage_extra_info = getattr(
            cache_controller_module,
            _HICACHE_STORAGE_EXTRA_INFO_NAME,
        )
    except Exception as exc:
        return _bridge_status_error(f"{type(exc).__name__}: {exc}")

    try:
        scheduler_patched = _patch_scheduler_prefetch(scheduler_cls)
        controller_patched = _patch_controller_prefetch(controller_cls)
        controller_hash_tracking_patched = _patch_controller_hash_tracking(controller_cls)
        prefetch_operation_patched = _patch_prefetch_operation(prefetch_operation_cls)
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
            controller_hash_tracking_patched,
            prefetch_operation_patched,
            extra_info_patched,
            storage_hit_query_patched,
            page_transfer_patched,
        )
    )
    return SGLangRequestMetadataBridgeStatus(
        installed=installed,
        scheduler_prefetch_patched=scheduler_patched,
        controller_prefetch_patched=controller_patched,
        controller_hash_tracking_patched=controller_hash_tracking_patched,
        prefetch_operation_patched=prefetch_operation_patched,
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
        "controller_hash_tracking_patched": status.controller_hash_tracking_patched,
        "prefetch_operation_patched": status.prefetch_operation_patched,
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
        restore_put = _patch_prefetch_queue_put(self, request_id, request_metadata)
        previous = getattr(_THREAD_CONTEXT, "prefetch_metadata", None)
        _THREAD_CONTEXT.prefetch_metadata = (request_id, request_metadata)
        try:
            operation = original(
                self,
                request_id,
                host_indices,
                new_input_tokens,
                last_hash,
                prefix_keys,
            )
            _attach_operation_request_metadata(operation, request_id, request_metadata)
            return operation
        finally:
            if restore_put is not None:
                restore_put()
            if previous is None:
                try:
                    delattr(_THREAD_CONTEXT, "prefetch_metadata")
                except AttributeError:
                    pass
            else:
                _THREAD_CONTEXT.prefetch_metadata = previous

    setattr(patched_prefetch, _PATCH_MARKER, True)
    setattr(patched_prefetch, _ORIGINAL_ATTR, original)
    setattr(controller_cls, "prefetch", patched_prefetch)
    return True


def _patch_controller_hash_tracking(controller_cls: type) -> bool:
    original = getattr(controller_cls, "__init__", None)
    if not callable(original):
        raise TypeError("SGLang HiCacheController.__init__ is not callable")
    if getattr(original, _HASH_TRACKING_PATCH_MARKER, False) is True:
        return True
    if not _controller_hash_tracking_installable(controller_cls):
        raise TypeError("SGLang HiCacheController.get_hash_str is not patchable")

    @functools.wraps(original)
    def patched_controller_init(self: object, *args: object, **kwargs: object) -> None:
        original(self, *args, **kwargs)
        if not _ensure_hicache_hash_tracking(self):
            raise TypeError("SGLang HiCacheController.get_hash_str is not patchable")
        return None

    setattr(patched_controller_init, _HASH_TRACKING_PATCH_MARKER, True)
    setattr(patched_controller_init, _HASH_TRACKING_ORIGINAL_ATTR, original)
    setattr(controller_cls, "__init__", patched_controller_init)
    return True


def _controller_hash_tracking_installable(controller_cls: type) -> bool:
    if callable(getattr(controller_cls, "get_hash_str", None)):
        return True
    controller_init = getattr(controller_cls, "__init__", None)
    try:
        source = inspect.getsource(controller_init)
    except (OSError, TypeError):
        return False
    return "get_hash_str" in source and "self.get_hash_str" in source


def _patch_prefetch_operation(prefetch_operation_cls: type) -> bool:
    original = getattr(prefetch_operation_cls, "__init__", None)
    if not callable(original):
        raise TypeError("SGLang PrefetchOperation.__init__ is not callable")
    if getattr(original, _PATCH_MARKER, False) is True:
        return True

    @functools.wraps(original)
    def patched_prefetch_operation_init(self: object, *args: object, **kwargs: object) -> None:
        original(self, *args, **kwargs)
        prefetch_metadata = _current_prefetch_request_metadata()
        if prefetch_metadata is None:
            return None
        request_id, request_metadata = prefetch_metadata
        _attach_operation_request_metadata(self, request_id, request_metadata)
        return None

    setattr(patched_prefetch_operation_init, _PATCH_MARKER, True)
    setattr(patched_prefetch_operation_init, _ORIGINAL_ATTR, original)
    setattr(prefetch_operation_cls, "__init__", patched_prefetch_operation_init)
    return True


def _patch_prefetch_queue_put(
    controller: object,
    request_id: str,
    request_metadata: Mapping[str, object],
) -> object | None:
    prefetch_queue = getattr(controller, "prefetch_queue", None)
    original_put = getattr(prefetch_queue, "put", None)
    if not callable(original_put):
        return None

    def put_with_metadata(item: object, *args: object, **kwargs: object) -> object:
        _attach_operation_request_metadata(item, request_id, request_metadata)
        return original_put(item, *args, **kwargs)

    try:
        setattr(prefetch_queue, "put", put_with_metadata)
    except Exception:
        return None

    def restore_put() -> None:
        setattr(prefetch_queue, "put", original_put)

    return restore_put


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
        previous_operation = getattr(_THREAD_CONTEXT, "operation", _MISSING)
        previous_batch_hash_priors = getattr(_THREAD_CONTEXT, "batch_hash_priors", _MISSING)
        _THREAD_CONTEXT.operation = operation
        if method_name == "_storage_hit_query":
            if not _ensure_hicache_hash_tracking(self):
                raise TypeError("SGLang HiCacheController.get_hash_str is not patchable")
            _THREAD_CONTEXT.batch_hash_priors = []
        try:
            return original(self, operation, *args, **kwargs)
        finally:
            if method_name == "_storage_hit_query":
                _restore_thread_context_attr("batch_hash_priors", previous_batch_hash_priors)
            _restore_thread_context_attr("operation", previous_operation)

    setattr(patched_method, _PATCH_MARKER, True)
    setattr(patched_method, _ORIGINAL_ATTR, original)
    setattr(controller_cls, method_name, patched_method)
    return True


def _ensure_hicache_hash_tracking(controller: object) -> bool:
    get_hash_str = getattr(controller, "get_hash_str", None)
    if getattr(get_hash_str, _HASH_TRACKING_PATCH_MARKER, False) is True:
        return True
    if not callable(get_hash_str):
        return False

    @functools.wraps(get_hash_str)
    def tracked_get_hash_str(*args: object, **kwargs: object) -> object:
        priors = getattr(_THREAD_CONTEXT, "batch_hash_priors", None)
        if isinstance(priors, list):
            priors.append(_prior_hash_argument(args=args, kwargs=kwargs))
        return get_hash_str(*args, **kwargs)

    try:
        setattr(tracked_get_hash_str, _HASH_TRACKING_PATCH_MARKER, True)
        setattr(tracked_get_hash_str, _HASH_TRACKING_ORIGINAL_ATTR, get_hash_str)
        setattr(controller, "get_hash_str", tracked_get_hash_str)
    except Exception:
        return False
    return True


def _restore_thread_context_attr(name: str, previous: object) -> None:
    if previous is _MISSING:
        try:
            delattr(_THREAD_CONTEXT, name)
        except AttributeError:
            pass
        return
    setattr(_THREAD_CONTEXT, name, previous)


def _prior_hash_argument(
    *,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> object:
    if len(args) >= 2:
        return args[1]
    return kwargs.get("prior_hash")


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
    _log_request_metadata_registered(metadata)
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


def _log_request_metadata_registered(metadata: Mapping[str, object]) -> None:
    kv_transfer_params = _kv_transfer_params_from_request_metadata(metadata)
    page_keys = kv_transfer_params.get(_DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM)
    page_key_count = len(page_keys) if _is_non_string_sequence(page_keys) else 0
    _LOGGER.info(
        "Cachet SGLang request metadata bridge: event=request_registered "
        "has_kv_transfer_params=%s has_request_id=%s has_handoff_json=%s "
        "has_inline_handoff=%s has_payload_uri=%s sglang_hicache_page_key_count=%d",
        bool(kv_transfer_params),
        isinstance(kv_transfer_params.get(_DOCUMENT_KV_REQUEST_ID_PARAM), str),
        isinstance(kv_transfer_params.get(_DOCUMENT_KV_HANDOFF_JSON_PARAM), str),
        isinstance(kv_transfer_params.get(_DOCUMENT_KV_HANDOFF_RECORD_PARAM), Mapping),
        isinstance(kv_transfer_params.get(_DOCUMENT_KV_PAYLOAD_URI_PARAM), str),
        page_key_count,
    )


def _kv_transfer_params_from_request_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    custom_params = metadata.get(_SGLANG_CUSTOM_PARAMS_KEY)
    if not isinstance(custom_params, Mapping):
        return {}
    kv_transfer_params = custom_params.get(_SGLANG_KV_TRANSFER_PARAMS_KEY)
    if not isinstance(kv_transfer_params, Mapping):
        return {}
    return kv_transfer_params


def _is_non_string_sequence(value: object) -> bool:
    return not isinstance(value, (str, bytes, bytearray)) and isinstance(value, Iterable)


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
    return _operation_request_metadata_with_runtime_context(operation, metadata)


def _operation_request_metadata_with_runtime_context(
    operation: object,
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    enriched = dict(metadata)
    last_hash = _pop_current_batch_prior_hash()
    if last_hash is _MISSING:
        last_hash = getattr(operation, "last_hash", None)
    if isinstance(last_hash, str) and last_hash:
        enriched[DOCUMENT_KV_SGLANG_HICACHE_LAST_HASH_EXTRA_INFO_KEY] = last_hash
    return enriched


def _pop_current_batch_prior_hash() -> object:
    priors = getattr(_THREAD_CONTEXT, "batch_hash_priors", None)
    if not isinstance(priors, list):
        return _MISSING
    if not priors:
        return _MISSING
    first_prior = priors[0]
    priors.clear()
    return first_prior


def _current_prefetch_request_metadata() -> tuple[str, Mapping[str, object]] | None:
    prefetch_metadata = getattr(_THREAD_CONTEXT, "prefetch_metadata", None)
    if not isinstance(prefetch_metadata, tuple) or len(prefetch_metadata) != 2:
        return None
    request_id, request_metadata = prefetch_metadata
    if not isinstance(request_id, str) or not isinstance(request_metadata, Mapping):
        return None
    return request_id, request_metadata


def _attach_operation_request_metadata(
    operation: object,
    request_id: str,
    request_metadata: Mapping[str, object],
) -> None:
    if getattr(operation, "request_id", request_id) != request_id:
        return
    setattr(operation, _OPERATION_EXTRA_INFO_ATTR, dict(request_metadata))


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
        and record.get("controller_hash_tracking_patched") is True
        and record.get("prefetch_operation_patched") is True
        and record.get("hicache_storage_extra_info_factory_patched") is True
        and record.get("storage_hit_query_patched") is True
        and record.get("page_transfer_patched") is True
    )
