"""Closed, reusable Vanilla handoff artifacts for publication latency runs.

Offline KV generation is intentionally separate from deployment-block timing.  This
module closes the generated inputs and their handoff/payload files into a portable
manifest, then stages that reviewed bundle on node-local storage.  Absolute source
paths are never part of the manifest identity: path-bearing JSON values are reduced
to validated POSIX names before request identities are hashed.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, cast

from document_kv_cache.artifact_identity import ArtifactIdentity, TokenContract
from document_kv_cache.benchmarks import (
    DOCUMENT_KV_ARTIFACT_ID_PARAM,
    DOCUMENT_KV_CACHE_METHOD_PARAM,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    SUPPORTED_V1_DATASETS,
)
from document_kv_cache.engine_adapters import validate_engine_adapter_request_record
from document_kv_cache.main_latency_inputs import (
    MAIN_LATENCY_ADD_SPECIAL_TOKENS,
    MAIN_LATENCY_TARGET_SEGMENT_COUNTS,
)
from document_kv_cache.storage import local_path


PUBLICATION_HANDOFF_BUNDLE_RECORD_TYPE = (
    "cachet.publication_latency_handoff_bundle.v1"
)
PUBLICATION_HANDOFF_BUNDLE_SCHEMA_VERSION = 1
PUBLICATION_HANDOFF_STAGING_ATTESTATION_RECORD_TYPE = (
    "cachet.publication_latency_handoff_staging.v1"
)
PUBLICATION_HANDOFF_STAGING_ATTESTATION_SCHEMA_VERSION = 1
PUBLICATION_HANDOFF_STAGING_ATTESTATION_FILENAME = (
    "publication-latency-handoff-staging.attestation.json"
)

_SHA256_LENGTH = 64
_BUNDLE_KEYS = frozenset(
    {
        "closed_record_sha256",
        "context_tokens",
        "datasets",
        "files",
        "files_sha256",
        "identity",
        "input_bundle_sha256",
        "portable_bundle_sha256",
        "record_type",
        "request_closure_sha256",
        "schema_version",
        "token_closure_sha256",
    }
)
_FILE_KEYS = frozenset({"byte_count", "relative_name", "role", "sha256"})
_DATASET_KEYS = frozenset(
    {
        "byte_count",
        "dataset",
        "entries",
        "normalized_records_sha256",
        "relative_name",
        "row_count",
        "sha256",
    }
)
_ENTRY_KEYS = frozenset(
    {
        "artifact_id",
        "cache_method",
        "dataset",
        "example_id",
        "handoff_relative_name",
        "handoff_identity_sha256",
        "handoff_sha256",
        "payload_relative_name",
        "payload_sha256",
        "request_id",
        "request_identity_sha256",
        "row_index",
        "token_identity_sha256",
        "transfer_scope",
    }
)
_IDENTITY_KEYS = frozenset(
    {
        "artifact_id",
        "artifact_identity",
        "artifact_identity_sha256",
        "layout_identity",
        "layout_identity_sha256",
        "model_identity",
        "model_identity_sha256",
        "tokenizer_identity",
        "tokenizer_identity_sha256",
        "topology_identity",
        "topology_identity_sha256",
    }
)
_ATTESTATION_KEYS = frozenset(
    {
        "bundle_closed_record_sha256",
        "closed_record_sha256",
        "context_tokens",
        "datasets",
        "files",
        "input_bundle_sha256",
        "record_type",
        "schema_version",
        "staged_root",
    }
)
_STAGED_FILE_KEYS = frozenset(
    {
        "byte_count",
        "path_rewritten",
        "relative_name",
        "role",
        "sha256",
        "source_byte_count",
        "source_sha256",
    }
)
_STAGED_DATASET_KEYS = frozenset(
    {
        "byte_count",
        "dataset",
        "normalized_records_sha256",
        "relative_name",
        "sha256",
    }
)
_PATH_TOKEN_PREFIX = "bundle-relative:"
_MAIN_LATENCY_TRANSFORMATION_ID = (
    "cachet.main_latency.lossless_context_tiling.v1"
)

__all__ = [
    "PUBLICATION_HANDOFF_BUNDLE_RECORD_TYPE",
    "PUBLICATION_HANDOFF_BUNDLE_SCHEMA_VERSION",
    "PUBLICATION_HANDOFF_STAGING_ATTESTATION_FILENAME",
    "PUBLICATION_HANDOFF_STAGING_ATTESTATION_RECORD_TYPE",
    "PUBLICATION_HANDOFF_STAGING_ATTESTATION_SCHEMA_VERSION",
    "StagedPublicationLatencyHandoffBundle",
    "close_publication_latency_handoff_bundle",
    "read_publication_latency_handoff_bundle",
    "stage_publication_latency_handoff_bundle",
    "validate_publication_latency_handoff_bundle",
    "verify_staged_publication_latency_handoff_bundle",
    "write_publication_latency_handoff_bundle",
]


@dataclass(frozen=True, slots=True)
class StagedPublicationLatencyHandoffBundle:
    """Verified node-local paths plus their closed staging attestation."""

    root: Path
    attestation_path: Path
    dataset_path_items: tuple[tuple[str, Path], ...]
    attestation: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "attestation_path", Path(self.attestation_path))
        items = tuple((dataset, Path(path)) for dataset, path in self.dataset_path_items)
        if tuple(dataset for dataset, _path in items) != tuple(SUPPORTED_V1_DATASETS):
            raise ValueError("dataset_path_items must cover publication datasets in order")
        object.__setattr__(self, "dataset_path_items", items)
        object.__setattr__(self, "attestation", MappingProxyType(dict(self.attestation)))

    @property
    def dataset_paths(self) -> dict[str, Path]:
        """Return runner-ready enriched dataset paths in governed order."""

        return dict(self.dataset_path_items)


@dataclass(frozen=True, slots=True)
class _TransferBinding:
    scope: str
    params: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _AnalyzedBinding:
    entry: dict[str, Any]
    artifact_identity: dict[str, Any]
    layout_identity: dict[str, Any]


def close_publication_latency_handoff_bundle(
    bundle_root: str | Path,
    enriched_dataset_paths: Mapping[str, str | Path],
    *,
    context_tokens: int,
    input_bundle_sha256: str,
) -> dict[str, Any]:
    """Close one 8k/16k/32k enriched dataset set and every referenced artifact.

    ``bundle_root`` must contain exactly the four enriched JSONL files and the
    handoff/payload files they reference.  The returned record contains no source
    root, so it remains meaningful after the reviewed durable directory is moved.
    """

    root = _existing_real_directory(bundle_root, field_name="bundle_root")
    target = _validated_context_tokens(context_tokens)
    input_digest = _required_sha256(
        input_bundle_sha256,
        field_name="input_bundle_sha256",
    )
    dataset_paths = _validated_dataset_paths(
        root,
        enriched_dataset_paths,
    )
    record = _build_bundle_record(
        root,
        dataset_paths,
        context_tokens=target,
        input_bundle_sha256=input_digest,
    )
    validate_publication_latency_handoff_bundle(record, bundle_root=root)
    return record


def validate_publication_latency_handoff_bundle(
    record: Mapping[str, Any],
    *,
    bundle_root: str | Path,
) -> None:
    """Re-authenticate a closed durable bundle without trusting stored paths."""

    manifest = _validated_bundle_record(record)
    root = _existing_real_directory(bundle_root, field_name="bundle_root")
    dataset_paths = {
        cast(str, dataset_record["dataset"]): root
        / _validated_relative_name(
            dataset_record["relative_name"],
            field_name="datasets.relative_name",
        )
        for dataset_record in _mapping_sequence(manifest["datasets"], "datasets")
    }
    rebuilt = _build_bundle_record(
        root,
        dataset_paths,
        context_tokens=cast(int, manifest["context_tokens"]),
        input_bundle_sha256=cast(str, manifest["input_bundle_sha256"]),
        expected_manifest=manifest,
    )
    if rebuilt != manifest:
        raise ValueError("publication handoff bundle does not match its closed manifest")


def write_publication_latency_handoff_bundle(
    record: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Write one canonical manifest, refusing every overwrite."""

    manifest = _validated_bundle_record(record)
    target = Path(path).expanduser().absolute()
    _reject_symlink_path(target, include_leaf=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite publication manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(_canonical_json_bytes(manifest, pretty=True))
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite publication manifest: {target}"
        ) from None
    return target


def read_publication_latency_handoff_bundle(path: str | Path) -> dict[str, Any]:
    """Read a canonical closed manifest record."""

    source = _existing_regular_file(Path(path), field_name="manifest")
    content = source.read_bytes()
    record = _json_object(content, field_name="publication handoff manifest")
    if content != _canonical_json_bytes(record, pretty=True):
        raise ValueError("publication handoff manifest is not canonical JSON")
    return _validated_bundle_record(record)


def stage_publication_latency_handoff_bundle(
    record: Mapping[str, Any],
    *,
    source_root: str | Path,
    local_nvme_dir: str | Path,
) -> StagedPublicationLatencyHandoffBundle:
    """Copy, verify, path-rewrite, and atomically publish one node-local bundle.

    The final directory must not exist.  Work happens in a sibling temporary
    directory and is removed on any failure, so a failed call cannot leave a
    seemingly usable partial stage.
    """

    manifest = _validated_bundle_record(record)
    source = _existing_real_directory(source_root, field_name="source_root")
    validate_publication_latency_handoff_bundle(manifest, bundle_root=source)
    target = _nonexistent_output_directory(local_nvme_dir)
    _reject_overlap(source, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(target.parent, include_leaf=True)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    published = False
    try:
        source_files = _manifest_files_by_name(manifest)
        for relative_name, source_file in source_files.items():
            relative = _validated_relative_name(
                relative_name,
                field_name="files.relative_name",
            )
            source_path = source / relative
            destination_path = temporary / relative
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if destination_path.exists() or destination_path.is_symlink():
                raise ValueError(f"staging path collision: {relative_name}")
            shutil.copyfile(source_path, destination_path, follow_symlinks=False)
            _verify_file_record(destination_path, source_file, label="copied file")

        dataset_records = {
            cast(str, item["dataset"]): item
            for item in _mapping_sequence(manifest["datasets"], "datasets")
        }
        for dataset in SUPPORTED_V1_DATASETS:
            dataset_record = dataset_records[dataset]
            relative = _validated_relative_name(
                dataset_record["relative_name"],
                field_name="datasets.relative_name",
            )
            staged_path = temporary / relative
            records = _canonical_jsonl_records(
                staged_path.read_bytes(),
                field_name=f"source {dataset} JSONL",
            )
            rewritten = _rewrite_dataset_records_for_stage(
                records,
                dataset_record=dataset_record,
                final_root=target,
            )
            rewritten_bytes = _canonical_jsonl_bytes(rewritten)
            rewrite_path = staged_path.with_name(staged_path.name + ".rewrite")
            with rewrite_path.open("xb") as handle:
                handle.write(rewritten_bytes)
            os.replace(rewrite_path, staged_path)

        attestation = _build_staging_attestation(
            manifest,
            temporary_root=temporary,
            final_root=target,
        )
        attestation_path = temporary / PUBLICATION_HANDOFF_STAGING_ATTESTATION_FILENAME
        if attestation_path.exists():
            raise ValueError("staging attestation path collides with bundle content")
        with attestation_path.open("xb") as handle:
            handle.write(_canonical_json_bytes(attestation, pretty=True))

        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite local NVMe stage: {target}")
        temporary_inode = temporary.stat().st_ino
        os.rename(temporary, target)
        try:
            result = verify_staged_publication_latency_handoff_bundle(
                manifest,
                staged_root=target,
            )
        except BaseException:
            try:
                if (
                    target.is_dir()
                    and not target.is_symlink()
                    and target.stat().st_ino == temporary_inode
                ):
                    shutil.rmtree(target)
            finally:
                published = False
            raise
        published = True
        return result
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def verify_staged_publication_latency_handoff_bundle(
    record: Mapping[str, Any],
    *,
    staged_root: str | Path,
) -> StagedPublicationLatencyHandoffBundle:
    """Verify a completed stage, including its destination-specific attestation."""

    manifest = _validated_bundle_record(record)
    root = _existing_real_directory(staged_root, field_name="staged_root")
    attestation_path = root / PUBLICATION_HANDOFF_STAGING_ATTESTATION_FILENAME
    attestation_bytes = _existing_regular_file(
        attestation_path,
        field_name="staging attestation",
    ).read_bytes()
    attestation = _json_object(attestation_bytes, field_name="staging attestation")
    if attestation_bytes != _canonical_json_bytes(attestation, pretty=True):
        raise ValueError("staging attestation is not canonical JSON")
    _validate_attestation(attestation, manifest=manifest, staged_root=root)
    _verify_staged_tree(root, manifest=manifest, attestation=attestation)
    _verify_staged_semantics(root, manifest=manifest)
    dataset_paths = tuple(
        (
            dataset,
            root
            / _validated_relative_name(
                next(
                    item["relative_name"]
                    for item in _mapping_sequence(manifest["datasets"], "datasets")
                    if item["dataset"] == dataset
                ),
                field_name="datasets.relative_name",
            ),
        )
        for dataset in SUPPORTED_V1_DATASETS
    )
    return StagedPublicationLatencyHandoffBundle(
        root=root,
        attestation_path=attestation_path,
        dataset_path_items=dataset_paths,
        attestation=attestation,
    )


def _build_bundle_record(
    root: Path,
    dataset_paths: Mapping[str, Path],
    *,
    context_tokens: int,
    input_bundle_sha256: str,
    expected_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_entries = _expected_entry_map(expected_manifest)
    datasets: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any]] = {}
    shared_artifact_identity: dict[str, Any] | None = None
    shared_layout_identity: dict[str, Any] | None = None

    for dataset in SUPPORTED_V1_DATASETS:
        dataset_path = _existing_regular_file(
            dataset_paths[dataset],
            field_name=f"{dataset} enriched JSONL",
        )
        dataset_relative = _relative_name_for_path(
            root,
            dataset_path,
            field_name=f"{dataset} enriched JSONL",
        )
        dataset_bytes = dataset_path.read_bytes()
        records = _canonical_jsonl_records(
            dataset_bytes,
            field_name=f"{dataset} enriched JSONL",
        )
        if not records:
            raise ValueError(f"{dataset} enriched JSONL must not be empty")
        _add_file_record(
            files,
            dataset_relative,
            role="enriched_dataset",
            content=dataset_bytes,
        )
        normalized_records: list[dict[str, Any]] = []
        dataset_entries: list[dict[str, Any]] = []
        seen_example_ids: set[str] = set()
        for row_index, row in enumerate(records):
            row_dataset = _required_string(row, "dataset")
            if row_dataset != dataset:
                raise ValueError(
                    f"{dataset} row {row_index} declares dataset {row_dataset!r}"
                )
            example_id = _required_string(row, "example_id")
            if example_id in seen_example_ids:
                raise ValueError(f"duplicate example_id in {dataset}: {example_id}")
            seen_example_ids.add(example_id)
            _validate_context_topology(row, context_tokens=context_tokens)
            bindings = _transfer_bindings(row, row_index=row_index)
            normalized_row = copy.deepcopy(dict(row))
            for binding in bindings:
                expected = expected_entries.get((dataset, row_index, binding.scope))
                analyzed = _analyze_binding(
                    root,
                    dataset=dataset,
                    example_id=example_id,
                    row_index=row_index,
                    binding=binding,
                    files=files,
                    expected_entry=expected,
                )
                dataset_entries.append(analyzed.entry)
                _rewrite_binding_paths(
                    normalized_row,
                    binding.scope,
                    handoff_value=_portable_path(
                        cast(str, analyzed.entry["handoff_relative_name"])
                    ),
                    payload_value=_portable_path(
                        cast(str, analyzed.entry["payload_relative_name"])
                    ),
                )
                if shared_artifact_identity is None:
                    shared_artifact_identity = analyzed.artifact_identity
                    shared_layout_identity = analyzed.layout_identity
                elif (
                    analyzed.artifact_identity != shared_artifact_identity
                    or analyzed.layout_identity != shared_layout_identity
                ):
                    raise ValueError(
                        "publication handoff bundle mixes artifact or layout identities"
                    )
            normalized_records.append(normalized_row)

        datasets.append(
            {
                "byte_count": len(dataset_bytes),
                "dataset": dataset,
                "entries": dataset_entries,
                "normalized_records_sha256": _canonical_sha256(normalized_records),
                "relative_name": dataset_relative,
                "row_count": len(records),
                "sha256": sha256(dataset_bytes).hexdigest(),
            }
        )

    if shared_artifact_identity is None or shared_layout_identity is None:
        raise ValueError("publication handoff bundle contains no handoff artifacts")
    ordered_files = [files[name] for name in sorted(files)]
    _verify_exact_source_tree(root, expected_relative_names=set(files))
    identity = _identity_closure(
        shared_artifact_identity,
        shared_layout_identity,
    )
    entries = [
        entry
        for dataset_record in datasets
        for entry in cast(list[dict[str, Any]], dataset_record["entries"])
    ]
    portable_core = {
        "context_tokens": context_tokens,
        "datasets": [
            {
                "dataset": item["dataset"],
                "entries": [
                    _portable_entry_identity(entry)
                    for entry in cast(list[Mapping[str, Any]], item["entries"])
                ],
                "normalized_records_sha256": item["normalized_records_sha256"],
                "relative_name": item["relative_name"],
                "row_count": item["row_count"],
            }
            for item in datasets
        ],
        "identity": identity,
        "input_bundle_sha256": input_bundle_sha256,
    }
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "context_tokens": context_tokens,
        "datasets": datasets,
        "files": ordered_files,
        "files_sha256": _canonical_sha256(ordered_files),
        "identity": identity,
        "input_bundle_sha256": input_bundle_sha256,
        "portable_bundle_sha256": _canonical_sha256(portable_core),
        "record_type": PUBLICATION_HANDOFF_BUNDLE_RECORD_TYPE,
        "request_closure_sha256": _canonical_sha256(
            [entry["request_identity_sha256"] for entry in entries]
        ),
        "schema_version": PUBLICATION_HANDOFF_BUNDLE_SCHEMA_VERSION,
        "token_closure_sha256": _canonical_sha256(
            [entry["token_identity_sha256"] for entry in entries]
        ),
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _analyze_binding(
    root: Path,
    *,
    dataset: str,
    example_id: str,
    row_index: int,
    binding: _TransferBinding,
    files: dict[str, dict[str, Any]],
    expected_entry: Mapping[str, Any] | None,
) -> _AnalyzedBinding:
    params = binding.params
    if DOCUMENT_KV_HANDOFF_RECORD_PARAM in params:
        raise ValueError("publication bundles require external handoff JSON artifacts")
    handoff_reference = _required_string(params, DOCUMENT_KV_HANDOFF_JSON_PARAM)
    payload_reference = _required_string(params, DOCUMENT_KV_PAYLOAD_URI_PARAM)
    if expected_entry is None:
        handoff_path = _path_from_initial_reference(
            root,
            handoff_reference,
            field_name=DOCUMENT_KV_HANDOFF_JSON_PARAM,
        )
        payload_path = _path_from_initial_reference(
            root,
            payload_reference,
            field_name=DOCUMENT_KV_PAYLOAD_URI_PARAM,
        )
        handoff_relative = _relative_name_for_path(
            root,
            handoff_path,
            field_name=DOCUMENT_KV_HANDOFF_JSON_PARAM,
        )
        payload_relative = _relative_name_for_path(
            root,
            payload_path,
            field_name=DOCUMENT_KV_PAYLOAD_URI_PARAM,
        )
    else:
        _require_exact_keys(expected_entry, _ENTRY_KEYS, "manifest entry")
        handoff_relative = str(
            _validated_relative_name(
                expected_entry["handoff_relative_name"],
                field_name="entry.handoff_relative_name",
            )
        )
        payload_relative = str(
            _validated_relative_name(
                expected_entry["payload_relative_name"],
                field_name="entry.payload_relative_name",
            )
        )
        _require_reference_suffix(
            handoff_reference,
            handoff_relative,
            field_name=DOCUMENT_KV_HANDOFF_JSON_PARAM,
        )
        _require_reference_suffix(
            payload_reference,
            payload_relative,
            field_name=DOCUMENT_KV_PAYLOAD_URI_PARAM,
        )
        handoff_path = root / PurePosixPath(handoff_relative)
        payload_path = root / PurePosixPath(payload_relative)

    if handoff_relative == payload_relative:
        raise ValueError("handoff and payload relative names collide")
    handoff_bytes = _existing_regular_file(
        handoff_path,
        field_name="handoff JSON",
    ).read_bytes()
    handoff = _json_object(handoff_bytes, field_name="handoff JSON")
    if handoff_bytes != _canonical_handoff_json_bytes(handoff):
        raise ValueError("handoff JSON is not canonical JSON")
    validate_engine_adapter_request_record(
        handoff,
        expected_backend="vllm",
        require_external_payload_uri=True,
    )
    payload_path = _existing_regular_file(
        payload_path,
        field_name="payload",
    )
    payload_byte_count, payload_sha256 = _file_sha256_and_size(payload_path)
    handle = _required_mapping(handoff, "handle")
    payload_source = _required_mapping(handoff, "payload_source")
    handoff_payload_reference = _required_string(payload_source, "uri")
    if expected_entry is None:
        handoff_payload_path = _path_from_initial_reference(
            root,
            handoff_payload_reference,
            field_name="handoff.payload_source.uri",
        )
        if handoff_payload_path != payload_path:
            raise ValueError("handoff payload URI does not match enriched payload URI")
    else:
        _require_reference_suffix(
            handoff_payload_reference,
            payload_relative,
            field_name="handoff.payload_source.uri",
        )
    if _required_nonnegative_int(handle, "total_bytes") != payload_byte_count:
        raise ValueError("handoff total_bytes does not match payload byte count")
    if _required_nonnegative_int(payload_source, "total_bytes") != payload_byte_count:
        raise ValueError("payload_source.total_bytes does not match payload byte count")
    if _required_sha256(handle, "payload_checksum") != payload_sha256:
        raise ValueError("handoff payload checksum does not match payload bytes")
    if _required_sha256(payload_source, "checksum") != payload_sha256:
        raise ValueError("payload source checksum does not match payload bytes")

    request_id = _required_string(params, DOCUMENT_KV_REQUEST_ID_PARAM)
    if request_id != _required_string(handoff, "request_id"):
        raise ValueError("enriched request_id does not match handoff")
    cache_method = _required_string(params, DOCUMENT_KV_CACHE_METHOD_PARAM)
    if cache_method != _required_string(handle, "cache_method"):
        raise ValueError("enriched cache_method does not match handoff")
    artifact_record = _required_mapping(handle, "artifact_identity")
    artifact = ArtifactIdentity.from_record(artifact_record)
    if artifact.has_unresolved_fields:
        raise ValueError("publication artifact identity contains unresolved fields")
    artifact_id = artifact.artifact_id
    if _required_string(params, DOCUMENT_KV_ARTIFACT_ID_PARAM) != artifact_id:
        raise ValueError("enriched artifact_id does not match handoff identity")
    layout = dict(_required_mapping(handle, "layout"))
    _validate_layout_against_artifact(layout, artifact)
    token_identity = _token_identity(handle, artifact=artifact)

    handoff_sha256 = sha256(handoff_bytes).hexdigest()
    _add_file_record(
        files,
        handoff_relative,
        role="handoff_json",
        content=handoff_bytes,
    )
    _add_measured_file_record(
        files,
        payload_relative,
        role="payload",
        byte_count=payload_byte_count,
        digest=payload_sha256,
    )
    normalized_params = copy.deepcopy(dict(params))
    normalized_params[DOCUMENT_KV_HANDOFF_JSON_PARAM] = _portable_path(
        handoff_relative
    )
    normalized_params[DOCUMENT_KV_PAYLOAD_URI_PARAM] = _portable_path(
        payload_relative
    )
    normalized_handoff = copy.deepcopy(dict(handoff))
    cast(dict[str, Any], normalized_handoff["payload_source"])["uri"] = (
        _portable_path(payload_relative)
    )
    request_identity = {
        "artifact_id": artifact_id,
        "cache_method": cache_method,
        "dataset": dataset,
        "example_id": example_id,
        "handoff": normalized_handoff,
        "params": normalized_params,
        "request_id": request_id,
        "transfer_scope": binding.scope,
    }
    entry = {
        "artifact_id": artifact_id,
        "cache_method": cache_method,
        "dataset": dataset,
        "example_id": example_id,
        "handoff_relative_name": handoff_relative,
        "handoff_identity_sha256": _canonical_sha256(normalized_handoff),
        "handoff_sha256": handoff_sha256,
        "payload_relative_name": payload_relative,
        "payload_sha256": payload_sha256,
        "request_id": request_id,
        "request_identity_sha256": _canonical_sha256(request_identity),
        "row_index": row_index,
        "token_identity_sha256": _canonical_sha256(token_identity),
        "transfer_scope": binding.scope,
    }
    return _AnalyzedBinding(
        entry=entry,
        artifact_identity=dict(artifact_record),
        layout_identity=layout,
    )


def _identity_closure(
    artifact_record: Mapping[str, Any],
    layout_record: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = ArtifactIdentity.from_record(artifact_record)
    topology = {
        "pipeline_parallel_size": artifact.pipeline_parallel_size,
        "tensor_parallel_size": artifact.tensor_parallel_size,
    }
    model = {
        "lora_id": artifact.lora_id,
        "model_id": artifact.model_id,
        "model_revision": artifact.model_revision,
    }
    tokenizer = {
        "prompt_template_version": artifact.prompt_template_version,
        "tokenizer_id": artifact.tokenizer_id,
        "tokenizer_revision": artifact.tokenizer_revision,
    }
    artifact_dict = dict(artifact_record)
    layout_dict = dict(layout_record)
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_identity": artifact_dict,
        "artifact_identity_sha256": _canonical_sha256(artifact_dict),
        "layout_identity": layout_dict,
        "layout_identity_sha256": _canonical_sha256(layout_dict),
        "model_identity": model,
        "model_identity_sha256": _canonical_sha256(model),
        "tokenizer_identity": tokenizer,
        "tokenizer_identity_sha256": _canonical_sha256(tokenizer),
        "topology_identity": topology,
        "topology_identity_sha256": _canonical_sha256(topology),
    }


def _token_identity(
    handle: Mapping[str, Any],
    *,
    artifact: ArtifactIdentity,
) -> dict[str, Any]:
    segments = _mapping_sequence(handle.get("segments"), "handoff.handle.segments")
    if not segments:
        raise ValueError("publication handoff must contain token segments")
    described: list[dict[str, Any]] = []
    token_total = 0
    for index, segment in enumerate(segments):
        contract_record = _required_mapping(segment, "token_contract")
        contract = TokenContract.from_record(contract_record)
        if (
            contract.tokenizer_id != artifact.tokenizer_id
            or contract.tokenizer_revision != artifact.tokenizer_revision
            or contract.prompt_template_version != artifact.prompt_template_version
            or contract.add_special_tokens is not MAIN_LATENCY_ADD_SPECIAL_TOKENS
        ):
            raise ValueError(f"segment {index} token contract does not match artifact")
        token_count = _required_positive_int(segment, "token_count")
        if contract.token_count != token_count:
            raise ValueError(f"segment {index} token count does not match contract")
        token_total += token_count
        described.append(
            {
                "chunk_id": _required_string(segment, "chunk_id"),
                "chunk_type": _required_string(segment, "chunk_type"),
                "document_id": _required_string(segment, "document_id"),
                "token_contract": dict(contract_record),
                "token_count": token_count,
                "token_end": _required_nonnegative_int(segment, "token_end"),
                "token_start": _required_nonnegative_int(segment, "token_start"),
            }
        )
    if token_total != _required_positive_int(handle, "total_tokens"):
        raise ValueError("handoff segment token counts do not match total_tokens")
    return {"segments": described, "total_tokens": token_total}


def _validate_layout_against_artifact(
    layout: Mapping[str, Any],
    artifact: ArtifactIdentity,
) -> None:
    comparisons = {
        "block_size": artifact.block_size,
        "dtype": artifact.runtime_kv_dtype,
        "key_position_encoding": artifact.key_position_encoding,
        "layout_version": artifact.layout_version,
        "lora_id": artifact.lora_id,
        "model_id": artifact.model_id,
        "payload_axis_order": artifact.payload_axis_order,
        "rope_rotary_dim": artifact.rope_rotary_dim,
        "rope_theta": artifact.rope_theta,
    }
    for field_name, expected in comparisons.items():
        actual = layout.get(field_name)
        if actual != expected:
            raise ValueError(
                f"handoff layout {field_name} does not match artifact identity"
            )


def _build_staging_attestation(
    manifest: Mapping[str, Any],
    *,
    temporary_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    source_files = _manifest_files_by_name(manifest)
    staged_files: list[dict[str, Any]] = []
    dataset_relative_names = {
        cast(str, item["relative_name"])
        for item in _mapping_sequence(manifest["datasets"], "datasets")
    }
    for relative_name in sorted(source_files):
        source_record = source_files[relative_name]
        staged_path = temporary_root / PurePosixPath(relative_name)
        staged_path = _existing_regular_file(staged_path, field_name="staged file")
        byte_count, digest = _file_sha256_and_size(staged_path)
        staged_files.append(
            {
                "byte_count": byte_count,
                "path_rewritten": relative_name in dataset_relative_names,
                "relative_name": relative_name,
                "role": source_record["role"],
                "sha256": digest,
                "source_byte_count": source_record["byte_count"],
                "source_sha256": source_record["sha256"],
            }
        )
    dataset_records = {
        cast(str, item["dataset"]): item
        for item in _mapping_sequence(manifest["datasets"], "datasets")
    }
    staged_datasets: list[dict[str, Any]] = []
    for dataset in SUPPORTED_V1_DATASETS:
        source_dataset = dataset_records[dataset]
        relative_name = cast(str, source_dataset["relative_name"])
        content = (temporary_root / PurePosixPath(relative_name)).read_bytes()
        records = _canonical_jsonl_records(
            content,
            field_name=f"staged {dataset} JSONL",
        )
        normalized = _normalize_staged_records(
            records,
            dataset_record=source_dataset,
            final_root=final_root,
        )
        normalized_digest = _canonical_sha256(normalized)
        if normalized_digest != source_dataset["normalized_records_sha256"]:
            raise ValueError(f"staged {dataset} request identities changed")
        staged_datasets.append(
            {
                "byte_count": len(content),
                "dataset": dataset,
                "normalized_records_sha256": normalized_digest,
                "relative_name": relative_name,
                "sha256": sha256(content).hexdigest(),
            }
        )
    attestation: dict[str, Any] = {
        "bundle_closed_record_sha256": manifest["closed_record_sha256"],
        "closed_record_sha256": "",
        "context_tokens": manifest["context_tokens"],
        "datasets": staged_datasets,
        "files": staged_files,
        "input_bundle_sha256": manifest["input_bundle_sha256"],
        "record_type": PUBLICATION_HANDOFF_STAGING_ATTESTATION_RECORD_TYPE,
        "schema_version": PUBLICATION_HANDOFF_STAGING_ATTESTATION_SCHEMA_VERSION,
        "staged_root": str(final_root),
    }
    attestation["closed_record_sha256"] = _closed_record_sha256(attestation)
    return attestation


def _validate_attestation(
    attestation: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    staged_root: Path,
) -> None:
    _require_exact_keys(attestation, _ATTESTATION_KEYS, "staging attestation")
    if attestation.get("record_type") != PUBLICATION_HANDOFF_STAGING_ATTESTATION_RECORD_TYPE:
        raise ValueError("invalid staging attestation record_type")
    if attestation.get("schema_version") != PUBLICATION_HANDOFF_STAGING_ATTESTATION_SCHEMA_VERSION:
        raise ValueError("invalid staging attestation schema_version")
    if attestation.get("closed_record_sha256") != _closed_record_sha256(attestation):
        raise ValueError("invalid staging attestation closed_record_sha256")
    for field_name in (
        "bundle_closed_record_sha256",
        "context_tokens",
        "input_bundle_sha256",
    ):
        manifest_name = (
            "closed_record_sha256"
            if field_name == "bundle_closed_record_sha256"
            else field_name
        )
        if attestation.get(field_name) != manifest.get(manifest_name):
            raise ValueError(f"staging attestation {field_name} mismatch")
    if attestation.get("staged_root") != str(staged_root):
        raise ValueError("staging attestation staged_root mismatch")
    staged_files = _mapping_sequence(attestation.get("files"), "attestation.files")
    if any(set(item) != _STAGED_FILE_KEYS for item in staged_files):
        raise ValueError("staging attestation file records are not closed")
    staged_file_names = [
        _required_string(item, "relative_name") for item in staged_files
    ]
    if staged_file_names != sorted(staged_file_names) or len(staged_file_names) != len(
        set(staged_file_names)
    ):
        raise ValueError("staging attestation file names must be unique and sorted")
    staged_datasets = _mapping_sequence(
        attestation.get("datasets"),
        "attestation.datasets",
    )
    if any(set(item) != _STAGED_DATASET_KEYS for item in staged_datasets):
        raise ValueError("staging attestation dataset records are not closed")
    if tuple(item.get("dataset") for item in staged_datasets) != tuple(
        SUPPORTED_V1_DATASETS
    ):
        raise ValueError("staging attestation dataset coverage is invalid")
    manifest_datasets = {
        cast(str, item["dataset"]): item
        for item in _mapping_sequence(manifest["datasets"], "manifest.datasets")
    }
    staged_files_by_name = {
        cast(str, item["relative_name"]): item for item in staged_files
    }
    for staged_dataset in staged_datasets:
        dataset = _required_string(staged_dataset, "dataset")
        manifest_dataset = manifest_datasets[dataset]
        relative_name = _required_string(staged_dataset, "relative_name")
        if (
            relative_name != manifest_dataset["relative_name"]
            or staged_dataset.get("normalized_records_sha256")
            != manifest_dataset["normalized_records_sha256"]
        ):
            raise ValueError("staging attestation dataset identity mismatch")
        staged_file = staged_files_by_name.get(relative_name)
        if staged_file is None or (
            staged_dataset.get("sha256") != staged_file.get("sha256")
            or staged_dataset.get("byte_count") != staged_file.get("byte_count")
        ):
            raise ValueError("staging attestation dataset file binding mismatch")


def _verify_staged_tree(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    attestation: Mapping[str, Any],
) -> None:
    source_files = _manifest_files_by_name(manifest)
    staged_records = {
        _required_string(item, "relative_name"): item
        for item in _mapping_sequence(attestation["files"], "attestation.files")
    }
    if set(staged_records) != set(source_files):
        raise ValueError("staging attestation file coverage mismatch")
    expected_names = set(staged_records) | {
        PUBLICATION_HANDOFF_STAGING_ATTESTATION_FILENAME
    }
    _verify_exact_source_tree(root, expected_relative_names=expected_names)
    for relative_name, staged_record in staged_records.items():
        _verify_file_record(
            root / _validated_relative_name(relative_name, field_name="relative_name"),
            staged_record,
            label="staged file",
        )
        source_record = source_files[relative_name]
        if (
            staged_record.get("source_sha256") != source_record.get("sha256")
            or staged_record.get("source_byte_count")
            != source_record.get("byte_count")
            or staged_record.get("role") != source_record.get("role")
        ):
            raise ValueError("staging attestation source file binding mismatch")
        rewritten = source_record["role"] == "enriched_dataset"
        if staged_record.get("path_rewritten") is not rewritten:
            raise ValueError("staging attestation path_rewritten flag mismatch")
        if not rewritten and (
            staged_record.get("sha256") != source_record.get("sha256")
            or staged_record.get("byte_count") != source_record.get("byte_count")
        ):
            raise ValueError("non-dataset staged artifact changed during copy")


def _verify_staged_semantics(
    root: Path,
    *,
    manifest: Mapping[str, Any],
) -> None:
    dataset_records = {
        cast(str, item["dataset"]): item
        for item in _mapping_sequence(manifest["datasets"], "datasets")
    }
    for dataset in SUPPORTED_V1_DATASETS:
        dataset_record = dataset_records[dataset]
        relative = _validated_relative_name(
            dataset_record["relative_name"],
            field_name="dataset.relative_name",
        )
        records = _canonical_jsonl_records(
            (root / relative).read_bytes(),
            field_name=f"staged {dataset} JSONL",
        )
        normalized = _normalize_staged_records(
            records,
            dataset_record=dataset_record,
            final_root=root,
        )
        if _canonical_sha256(normalized) != dataset_record["normalized_records_sha256"]:
            raise ValueError(f"staged {dataset} normalized records mismatch")


def _rewrite_dataset_records_for_stage(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset_record: Mapping[str, Any],
    final_root: Path,
) -> tuple[dict[str, Any], ...]:
    entries = _dataset_entry_map(dataset_record)
    rewritten: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        row = copy.deepcopy(dict(record))
        for binding in _transfer_bindings(row, row_index=row_index):
            entry = entries.get((row_index, binding.scope))
            if entry is None:
                raise ValueError("manifest is missing a staged transfer binding")
            handoff = final_root / _validated_relative_name(
                entry["handoff_relative_name"],
                field_name="entry.handoff_relative_name",
            )
            payload = final_root / _validated_relative_name(
                entry["payload_relative_name"],
                field_name="entry.payload_relative_name",
            )
            _rewrite_binding_paths(
                row,
                binding.scope,
                handoff_value=str(handoff),
                payload_value=str(payload),
            )
        rewritten.append(row)
    if set(entries) != {
        (row_index, binding.scope)
        for row_index, row in enumerate(records)
        for binding in _transfer_bindings(row, row_index=row_index)
    }:
        raise ValueError("manifest has unmatched staged transfer bindings")
    return tuple(rewritten)


def _normalize_staged_records(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset_record: Mapping[str, Any],
    final_root: Path,
) -> tuple[dict[str, Any], ...]:
    entries = _dataset_entry_map(dataset_record)
    normalized: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        row = copy.deepcopy(dict(record))
        for binding in _transfer_bindings(row, row_index=row_index):
            entry = entries.get((row_index, binding.scope))
            if entry is None:
                raise ValueError("manifest is missing a staged transfer binding")
            handoff_relative = cast(str, entry["handoff_relative_name"])
            payload_relative = cast(str, entry["payload_relative_name"])
            expected_handoff = final_root / PurePosixPath(handoff_relative)
            expected_payload = final_root / PurePosixPath(payload_relative)
            if _reference_path(binding.params[DOCUMENT_KV_HANDOFF_JSON_PARAM]) != expected_handoff:
                raise ValueError("staged handoff path does not point into staged root")
            if _reference_path(binding.params[DOCUMENT_KV_PAYLOAD_URI_PARAM]) != expected_payload:
                raise ValueError("staged payload path does not point into staged root")
            _rewrite_binding_paths(
                row,
                binding.scope,
                handoff_value=_portable_path(handoff_relative),
                payload_value=_portable_path(payload_relative),
            )
        normalized.append(row)
    return tuple(normalized)


def _validated_bundle_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("publication handoff bundle must be a mapping")
    manifest = copy.deepcopy(dict(record))
    _require_exact_keys(manifest, _BUNDLE_KEYS, "publication handoff bundle")
    if manifest.get("record_type") != PUBLICATION_HANDOFF_BUNDLE_RECORD_TYPE:
        raise ValueError("invalid publication handoff bundle record_type")
    if manifest.get("schema_version") != PUBLICATION_HANDOFF_BUNDLE_SCHEMA_VERSION:
        raise ValueError("invalid publication handoff bundle schema_version")
    _validated_context_tokens(manifest.get("context_tokens"))
    _required_sha256(manifest, "input_bundle_sha256")
    if manifest.get("closed_record_sha256") != _closed_record_sha256(manifest):
        raise ValueError("invalid publication handoff bundle closed_record_sha256")
    datasets = _mapping_sequence(manifest.get("datasets"), "datasets")
    if tuple(item.get("dataset") for item in datasets) != tuple(SUPPORTED_V1_DATASETS):
        raise ValueError("publication handoff datasets must cover governed datasets in order")
    all_entries: list[Mapping[str, Any]] = []
    for dataset in datasets:
        _require_exact_keys(dataset, _DATASET_KEYS, "dataset record")
        _validated_relative_name(dataset.get("relative_name"), field_name="dataset.relative_name")
        _required_sha256(dataset, "sha256")
        _required_sha256(dataset, "normalized_records_sha256")
        _required_positive_int(dataset, "byte_count")
        _required_positive_int(dataset, "row_count")
        entries = _mapping_sequence(dataset.get("entries"), "dataset.entries")
        if not entries:
            raise ValueError("dataset entries must not be empty")
        for entry in entries:
            _require_exact_keys(entry, _ENTRY_KEYS, "manifest entry")
            for field_name in (
                "handoff_sha256",
                "handoff_identity_sha256",
                "payload_sha256",
                "request_identity_sha256",
                "token_identity_sha256",
            ):
                _required_sha256(entry, field_name)
            _validated_relative_name(
                entry.get("handoff_relative_name"),
                field_name="entry.handoff_relative_name",
            )
            _validated_relative_name(
                entry.get("payload_relative_name"),
                field_name="entry.payload_relative_name",
            )
            _required_nonnegative_int(entry, "row_index")
        all_entries.extend(entries)
    files = _mapping_sequence(manifest.get("files"), "files")
    file_names: list[str] = []
    for file_record in files:
        _require_exact_keys(file_record, _FILE_KEYS, "file record")
        file_names.append(
            str(
                _validated_relative_name(
                    file_record.get("relative_name"),
                    field_name="file.relative_name",
                )
            )
        )
        if file_record.get("role") not in {"enriched_dataset", "handoff_json", "payload"}:
            raise ValueError("invalid publication bundle file role")
        _required_sha256(file_record, "sha256")
        _required_positive_int(file_record, "byte_count")
    if file_names != sorted(file_names) or len(file_names) != len(set(file_names)):
        raise ValueError("publication bundle file names must be unique and sorted")
    if manifest.get("files_sha256") != _canonical_sha256(files):
        raise ValueError("invalid publication bundle files_sha256")
    identity = _required_mapping(manifest, "identity")
    _require_exact_keys(identity, _IDENTITY_KEYS, "identity closure")
    _verify_identity_closure(identity)
    if manifest.get("request_closure_sha256") != _canonical_sha256(
        [entry["request_identity_sha256"] for entry in all_entries]
    ):
        raise ValueError("invalid publication bundle request closure")
    if manifest.get("token_closure_sha256") != _canonical_sha256(
        [entry["token_identity_sha256"] for entry in all_entries]
    ):
        raise ValueError("invalid publication bundle token closure")
    portable_core = {
        "context_tokens": manifest["context_tokens"],
        "datasets": [
            {
                "dataset": item["dataset"],
                "entries": [
                    _portable_entry_identity(entry)
                    for entry in _mapping_sequence(item["entries"], "dataset.entries")
                ],
                "normalized_records_sha256": item["normalized_records_sha256"],
                "relative_name": item["relative_name"],
                "row_count": item["row_count"],
            }
            for item in datasets
        ],
        "identity": identity,
        "input_bundle_sha256": manifest["input_bundle_sha256"],
    }
    if manifest.get("portable_bundle_sha256") != _canonical_sha256(portable_core):
        raise ValueError("invalid publication bundle portable identity")
    return manifest


def _portable_entry_identity(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude only the raw handoff digest whose source URI is path-bearing."""

    return {
        "artifact_id": entry["artifact_id"],
        "cache_method": entry["cache_method"],
        "dataset": entry["dataset"],
        "example_id": entry["example_id"],
        "handoff_identity_sha256": entry["handoff_identity_sha256"],
        "handoff_relative_name": entry["handoff_relative_name"],
        "payload_relative_name": entry["payload_relative_name"],
        "payload_sha256": entry["payload_sha256"],
        "request_id": entry["request_id"],
        "request_identity_sha256": entry["request_identity_sha256"],
        "row_index": entry["row_index"],
        "token_identity_sha256": entry["token_identity_sha256"],
        "transfer_scope": entry["transfer_scope"],
    }


def _verify_identity_closure(identity: Mapping[str, Any]) -> None:
    artifact_record = _required_mapping(identity, "artifact_identity")
    artifact = ArtifactIdentity.from_record(artifact_record)
    if artifact.has_unresolved_fields:
        raise ValueError("publication artifact identity contains unresolved fields")
    expected = _identity_closure(
        artifact_record,
        _required_mapping(identity, "layout_identity"),
    )
    if dict(identity) != expected:
        raise ValueError("publication bundle identity closure is invalid")


def _validated_dataset_paths(
    root: Path,
    paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    if set(paths) != set(SUPPORTED_V1_DATASETS):
        raise ValueError("enriched_dataset_paths must cover all publication datasets")
    resolved: dict[str, Path] = {}
    for dataset in SUPPORTED_V1_DATASETS:
        path = Path(paths[dataset]).expanduser()
        if not path.is_absolute():
            path = root / path
        path = _existing_regular_file(path, field_name=f"{dataset} enriched JSONL")
        _relative_name_for_path(root, path, field_name=f"{dataset} enriched JSONL")
        resolved[dataset] = path
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("enriched dataset files must be distinct")
    return resolved


def _validate_context_topology(
    record: Mapping[str, Any],
    *,
    context_tokens: int,
) -> None:
    expected_count = MAIN_LATENCY_TARGET_SEGMENT_COUNTS[context_tokens]
    documents = record.get("documents")
    if not isinstance(documents, list) or len(documents) != expected_count:
        raise ValueError(
            f"publication {context_tokens} row must contain exactly "
            f"{expected_count} prepared document segments"
        )
    document_ids: set[str] = set()
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise ValueError("prepared document segments must be objects")
        document_id = _required_string(document, "document_id")
        if document_id in document_ids:
            raise ValueError("prepared document segment ids must be unique")
        document_ids.add(document_id)
        metadata = _required_mapping(document, "metadata")
        expected_metadata = {
            "cachet.main_latency.segment_count": str(expected_count),
            "cachet.main_latency.segment_index": str(index),
            "cachet.main_latency.transformation": _MAIN_LATENCY_TRANSFORMATION_ID,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("prepared document segment metadata does not match context")


def _transfer_bindings(
    record: Mapping[str, Any],
    *,
    row_index: int,
) -> tuple[_TransferBinding, ...]:
    direct = record.get("kv_transfer_params")
    per_arm = record.get("arm_kv_transfer_params")
    if direct and per_arm:
        raise ValueError(f"row {row_index} mixes direct and per-arm transfer params")
    bindings: list[_TransferBinding] = []
    if direct is not None:
        if not isinstance(direct, Mapping) or not direct:
            raise ValueError(f"row {row_index} kv_transfer_params must be a non-empty object")
        bindings.append(_TransferBinding("kv_transfer_params", direct))
    if per_arm is not None:
        if not isinstance(per_arm, Mapping) or not per_arm:
            raise ValueError(f"row {row_index} arm_kv_transfer_params must be a non-empty object")
        for arm_id in sorted(per_arm):
            params = per_arm[arm_id]
            if not isinstance(arm_id, str) or not arm_id:
                raise ValueError("arm ids must be non-empty strings")
            if not isinstance(params, Mapping) or not params:
                raise ValueError(f"row {row_index} arm {arm_id!r} params are invalid")
            bindings.append(_TransferBinding(f"arm_kv_transfer_params/{arm_id}", params))
    if not bindings:
        raise ValueError(f"row {row_index} has no external handoff transfer params")
    return tuple(bindings)


def _rewrite_binding_paths(
    record: dict[str, Any],
    scope: str,
    *,
    handoff_value: str,
    payload_value: str,
) -> None:
    if scope == "kv_transfer_params":
        params = cast(dict[str, Any], record["kv_transfer_params"])
    elif scope.startswith("arm_kv_transfer_params/"):
        arm_id = scope.removeprefix("arm_kv_transfer_params/")
        arms = cast(dict[str, Any], record["arm_kv_transfer_params"])
        params = cast(dict[str, Any], arms[arm_id])
    else:
        raise ValueError(f"unsupported transfer scope: {scope}")
    params[DOCUMENT_KV_HANDOFF_JSON_PARAM] = handoff_value
    params[DOCUMENT_KV_PAYLOAD_URI_PARAM] = payload_value


def _expected_entry_map(
    manifest: Mapping[str, Any] | None,
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    if manifest is None:
        return {}
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for dataset in _mapping_sequence(manifest.get("datasets"), "datasets"):
        dataset_name = _required_string(dataset, "dataset")
        for entry in _mapping_sequence(dataset.get("entries"), "dataset.entries"):
            key = (
                dataset_name,
                _required_nonnegative_int(entry, "row_index"),
                _required_string(entry, "transfer_scope"),
            )
            if key in result:
                raise ValueError("duplicate manifest transfer entry")
            result[key] = entry
    return result


def _dataset_entry_map(
    dataset_record: Mapping[str, Any],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    for entry in _mapping_sequence(dataset_record.get("entries"), "dataset.entries"):
        key = (
            _required_nonnegative_int(entry, "row_index"),
            _required_string(entry, "transfer_scope"),
        )
        if key in result:
            raise ValueError("duplicate dataset transfer entry")
        result[key] = entry
    return result


def _add_file_record(
    files: dict[str, dict[str, Any]],
    relative_name: str,
    *,
    role: str,
    content: bytes,
) -> None:
    _add_measured_file_record(
        files,
        relative_name,
        role=role,
        byte_count=len(content),
        digest=sha256(content).hexdigest(),
    )


def _add_measured_file_record(
    files: dict[str, dict[str, Any]],
    relative_name: str,
    *,
    role: str,
    byte_count: int,
    digest: str,
) -> None:
    _validated_relative_name(relative_name, field_name="relative_name")
    if type(byte_count) is not int or byte_count <= 0:
        raise ValueError("bundle artifact byte_count must be a positive integer")
    _required_sha256(digest, field_name="bundle artifact SHA-256")
    if relative_name == PUBLICATION_HANDOFF_STAGING_ATTESTATION_FILENAME:
        raise ValueError("bundle file collides with reserved staging attestation name")
    if relative_name in files:
        raise ValueError(f"duplicate publication bundle file: {relative_name}")
    files[relative_name] = {
        "byte_count": byte_count,
        "relative_name": relative_name,
        "role": role,
        "sha256": digest,
    }


def _manifest_files_by_name(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        _required_string(item, "relative_name"): item
        for item in _mapping_sequence(manifest.get("files"), "files")
    }


def _verify_file_record(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    source = _existing_regular_file(path, field_name=label)
    byte_count, digest = _file_sha256_and_size(source)
    if byte_count != _required_positive_int(record, "byte_count"):
        raise ValueError(f"{label} byte count mismatch: {path}")
    if digest != _required_sha256(record, "sha256"):
        raise ValueError(f"{label} SHA-256 mismatch: {path}")


def _file_sha256_and_size(path: Path) -> tuple[int, str]:
    """Hash large KV payloads with bounded memory and mutation detection."""

    digest = sha256()
    byte_count = 0
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(handle.fileno())
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ValueError(f"file changed while hashing: {path}")
    if byte_count != after.st_size:
        raise ValueError(f"file size changed while hashing: {path}")
    return byte_count, digest.hexdigest()


def _verify_exact_source_tree(
    root: Path,
    *,
    expected_relative_names: set[str],
) -> None:
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    inodes: dict[tuple[int, int], str] = {}
    for directory, names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            if child.is_symlink():
                raise ValueError(f"publication bundle contains symlink: {child}")
            if not child.is_dir():
                raise ValueError(f"publication bundle contains non-directory: {child}")
            actual_directories.add(_relative_name_for_path(root, child, field_name="directory"))
        for filename in filenames:
            child = directory_path / filename
            if child.is_symlink():
                raise ValueError(f"publication bundle contains symlink: {child}")
            info = child.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"publication bundle contains non-regular file: {child}")
            relative = _relative_name_for_path(root, child, field_name="file")
            inode = (info.st_dev, info.st_ino)
            previous = inodes.get(inode)
            if previous is not None:
                raise ValueError(
                    f"publication bundle contains duplicate hard-linked files: {previous}, {relative}"
                )
            inodes[inode] = relative
            actual_files.add(relative)
    if actual_files != expected_relative_names:
        missing = sorted(expected_relative_names - actual_files)
        extra = sorted(actual_files - expected_relative_names)
        raise ValueError(f"publication bundle file closure mismatch; missing={missing}, extra={extra}")
    expected_directories = {
        str(parent)
        for name in expected_relative_names
        for parent in _relative_parents(PurePosixPath(name))
    }
    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)
        extra = sorted(actual_directories - expected_directories)
        raise ValueError(
            f"publication bundle directory closure mismatch; missing={missing}, extra={extra}"
        )


def _relative_parents(path: PurePosixPath) -> tuple[PurePosixPath, ...]:
    parents: list[PurePosixPath] = []
    current = path.parent
    while str(current) != ".":
        parents.append(current)
        current = current.parent
    return tuple(parents)


def _path_from_initial_reference(
    root: Path,
    value: str,
    *,
    field_name: str,
) -> Path:
    path = _reference_path(value)
    if not path.is_absolute():
        if ".." in path.parts:
            raise ValueError(f"{field_name} contains path traversal")
        path = root / path
    path = _existing_regular_file(path, field_name=field_name)
    _relative_name_for_path(root, path, field_name=field_name)
    return path


def _reference_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact reference must be a non-empty string")
    return Path(local_path(value)).expanduser()


def _require_reference_suffix(
    value: str,
    relative_name: str,
    *,
    field_name: str,
) -> None:
    path = _reference_path(value)
    relative = _validated_relative_name(relative_name, field_name=field_name)
    if tuple(path.parts[-len(relative.parts) :]) != tuple(relative.parts):
        raise ValueError(f"{field_name} does not match its portable relative name")


def _relative_name_for_path(root: Path, path: Path, *, field_name: str) -> str:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be inside bundle_root") from exc
    pure = _validated_relative_name(relative.as_posix(), field_name=field_name)
    return str(pure)


def _validated_relative_name(value: object, *, field_name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty POSIX relative name")
    if "\\" in value:
        raise ValueError(f"{field_name} must not contain backslashes")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} contains traversal or is not normalized")
    return path


def _portable_path(relative_name: str) -> str:
    return _PATH_TOKEN_PREFIX + str(
        _validated_relative_name(relative_name, field_name="portable path")
    )


def _existing_real_directory(value: str | Path, *, field_name: str) -> Path:
    path = Path(value).expanduser().absolute()
    _reject_symlink_path(path, include_leaf=True)
    if not path.is_dir():
        raise ValueError(f"{field_name} must be an existing directory: {path}")
    return path.resolve(strict=True)


def _existing_regular_file(path: Path, *, field_name: str) -> Path:
    candidate = Path(path).expanduser().absolute()
    _reject_symlink_path(candidate, include_leaf=True)
    try:
        info = candidate.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"missing {field_name}: {candidate}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{field_name} must be a regular file: {candidate}")
    return candidate.resolve(strict=True)


def _reject_symlink_path(path: Path, *, include_leaf: bool) -> None:
    candidate = path.absolute()
    parts = candidate.parents
    for parent in reversed(parts):
        if parent.exists() and parent.is_symlink():
            raise ValueError(f"symlink paths are forbidden: {parent}")
    if include_leaf and candidate.is_symlink():
        raise ValueError(f"symlink paths are forbidden: {candidate}")


def _nonexistent_output_directory(value: str | Path) -> Path:
    target = Path(value).expanduser().absolute()
    _reject_symlink_path(target, include_leaf=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite local NVMe stage: {target}")
    if target.name in {"", ".", ".."}:
        raise ValueError("local_nvme_dir must name a new directory")
    return target.resolve(strict=False)


def _reject_overlap(source: Path, target: Path) -> None:
    resolved_target = target.resolve(strict=False)
    if _is_relative_to(resolved_target, source) or _is_relative_to(source, resolved_target):
        raise ValueError("source_root and local_nvme_dir must not overlap")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _canonical_jsonl_records(
    content: bytes,
    *,
    field_name: str,
) -> tuple[Mapping[str, Any], ...]:
    if not content or not content.endswith(b"\n"):
        raise ValueError(f"{field_name} must be non-empty newline-terminated JSONL")
    lines = content[:-1].split(b"\n")
    if any(not line for line in lines):
        raise ValueError(f"{field_name} must not contain blank rows")
    records: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines):
        record = _json_object(line, field_name=f"{field_name} row {index}")
        if line + b"\n" != _canonical_jsonl_bytes((record,)):
            raise ValueError(f"{field_name} row {index} is not canonical JSONL")
        records.append(record)
    return tuple(records)


def _canonical_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(record), sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")


def _canonical_handoff_json_bytes(record: Mapping[str, Any]) -> bytes:
    """Mirror ``write_engine_adapter_request_json`` byte-for-byte."""

    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _json_object(content: bytes, *, field_name: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _canonical_json_bytes(value: Any, *, pretty: bool) -> bytes:
    options: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    suffix = "\n" if pretty else ""
    return (json.dumps(value, **options) + suffix).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value, pretty=False)).hexdigest()


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    normalized = dict(record)
    normalized["closed_record_sha256"] = ""
    return _canonical_sha256(normalized)


def _mapping_sequence(value: object, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{index}] must be an object")
        result.append(item)
    return tuple(result)


def _required_mapping(record: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = record.get(field_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _required_string(record: Mapping[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _required_sha256(record: Mapping[str, Any] | str, field_name: str) -> str:
    value = record if isinstance(record, str) else record.get(field_name)
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _required_nonnegative_int(record: Mapping[str, Any], field_name: str) -> int:
    value = record.get(field_name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _required_positive_int(record: Mapping[str, Any], field_name: str) -> int:
    value = _required_nonnegative_int(record, field_name)
    if value == 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_exact_keys(
    record: Mapping[str, Any],
    expected: frozenset[str],
    field_name: str,
) -> None:
    if set(record) != expected:
        missing = sorted(expected - set(record))
        extra = sorted(set(record) - expected)
        raise ValueError(f"{field_name} keys are not closed; missing={missing}, extra={extra}")


def _validated_context_tokens(value: object) -> int:
    if type(value) is not int or value not in MAIN_LATENCY_TARGET_SEGMENT_COUNTS:
        raise ValueError("context_tokens must be exactly 8192, 16384, or 32768")
    return value
