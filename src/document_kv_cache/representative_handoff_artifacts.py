"""Immutable BF16 representative handoffs, independent of serving deployments.

This content format is separate from the four-dataset publication bundle. It
reuses its file/request verification primitives without changing that format.
Token counts are measured by the pinned tokenizer at closure; consumers verify
the sealed inputs, token contracts and exact bytes, without regenerating KV.
These functions do not submit work or grant publication/ledger authority.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from document_kv_cache import publication_handoff_artifacts as shared
from document_kv_cache.artifact_identity import ArtifactIdentity
from document_kv_cache.benchmark_handoffs import BenchmarkHandoffEntry, BenchmarkHandoffManifest
from document_kv_cache.benchmark_runner import load_benchmark_jsonl
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    DOCUMENT_KV_ARTIFACT_ID_PARAM,
    DOCUMENT_KV_CACHE_METHOD_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
)
from document_kv_cache.canary_orchestration import (
    FULL_PREFIX_CANARY_ARM,
    REPRESENTATIVE_CANARY_MODEL_ID,
    REPRESENTATIVE_CANARY_MODEL_REVISION,
    VANILLA_CANARY_ARM,
    _logical_sample_digest,
    _logical_token_rows,
    _project_records,
    _validate_combined_params,
    _validate_manifest_pair,
    build_handoff_topology_attestation,
)

REPRESENTATIVE_HANDOFF_BUNDLE_RECORD_TYPE = "cachet.representative_handoff_bundle.v1"
REPRESENTATIVE_HANDOFF_STAGING_RECORD_TYPE = "cachet.representative_handoff_staging.v1"
REPRESENTATIVE_HANDOFF_STAGING_FILENAME = "representative-handoff-staging.json"
_PROJECTIONS = "_representative_stage"
_ARMS = (BASELINE_PREFILL_ARM, FULL_PREFIX_CANARY_ARM, VANILLA_CANARY_ARM)
_BINDING_KEYS = {
    "source_commit", "package_wheel_sha256", "native_runtime_closure_sha256",
    "prepared_input_bundle_sha256",
}
_RECORD_KEYS = {
    "record_type", "schema_version", "bindings", "example_count", "contexts",
    "files", "portable_identity_sha256", "closed_record_sha256",
}
REPRESENTATIVE_HANDOFF_SOURCE_RECORD_TYPE = "cachet.representative_handoff_source.v1"
REPRESENTATIVE_SUPPLEMENT_PROVENANCE_RECORD_TYPE = "cachet.representative_supplement_provenance.v1"

__all__ = [
    "REPRESENTATIVE_HANDOFF_BUNDLE_RECORD_TYPE",
    "REPRESENTATIVE_HANDOFF_STAGING_RECORD_TYPE",
    "REPRESENTATIVE_HANDOFF_STAGING_FILENAME",
    "REPRESENTATIVE_HANDOFF_SOURCE_RECORD_TYPE",
    "RepresentativeHandoffBindings",
    "RepresentativeHandoffSourceV1",
    "RepresentativeSupplementProvenanceV1",
    "REPRESENTATIVE_SUPPLEMENT_PROVENANCE_RECORD_TYPE",
    "StagedRepresentativeHandoffs",
    "close_representative_handoff_bundle",
    "validate_representative_handoff_bundle",
    "stage_representative_handoff_bundle",
    "verify_staged_representative_handoff_bundle",
]


@dataclass(frozen=True, slots=True)
class RepresentativeHandoffBindings:
    """Frozen external source/runtime/preparation references, checked by callers.

The exact enriched input files and their logical identities are additionally
hashed inside the bundle; these references do not replace content verification.
"""

    source_commit: str
    package_wheel_sha256: str
    native_runtime_closure_sha256: str
    prepared_input_bundle_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_commit, str)
            or len(self.source_commit) != 40
            or any(c not in "0123456789abcdef" for c in self.source_commit)
        ):
            raise ValueError("source_commit must be a lowercase Git SHA-1")
        for name in sorted(_BINDING_KEYS - {"source_commit"}):
            shared._required_sha256(getattr(self, name), name)

    def to_record(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StagedRepresentativeHandoffs:
    root: Path
    attestation_path: Path
    dataset_paths: Mapping[tuple[int, str], Path]
    attestation: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RepresentativeHandoffSourceV1:
    """Closed transport settings; all three arms consume the same immutable set."""

    manifest_uri: str
    manifest_file_sha256: str
    bundle_closed_record_sha256: str
    source_root: str
    local_stage_root: str
    context_tokens: int
    arm_id: str
    bindings: RepresentativeHandoffBindings

    def __post_init__(self) -> None:
        for field in ("manifest_file_sha256", "bundle_closed_record_sha256"):
            shared._required_sha256(getattr(self, field), field)
        if not isinstance(self.bindings, RepresentativeHandoffBindings):
            raise TypeError("bindings must be RepresentativeHandoffBindings")
        if type(self.context_tokens) is not int or self.context_tokens not in {8192, 16384}:
            raise ValueError("representative source context must be 8192 or 16384")
        if self.arm_id not in _ARMS:
            raise ValueError("representative source arm must be Baseline/full-prefix/Vanilla")
        for field in ("manifest_uri", "source_root", "local_stage_root"):
            value = getattr(self, field)
            if not isinstance(value, str) or value != value.strip() or "\\" in value:
                raise ValueError(f"{field} must be a normalized cluster path")
            path_text = value.removeprefix("dbfs:")
            if str(Path(path_text)) != path_text or ".." in Path(path_text).parts:
                raise ValueError(f"{field} must be a normalized cluster path")
            if field == "local_stage_root":
                if not value.startswith("/local_disk0/"):
                    raise ValueError("local_stage_root must be under /local_disk0")
            elif not value.startswith(("dbfs:/", "/dbfs/", "/Volumes/")):
                raise ValueError(f"{field} must use persistent cluster storage")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RepresentativeHandoffSourceV1:
        fields = frozenset(cls.__dataclass_fields__)
        shared._require_exact_keys(record, fields | {"record_type"}, "representative source")
        if record["record_type"] != REPRESENTATIVE_HANDOFF_SOURCE_RECORD_TYPE:
            raise ValueError("unsupported representative source record_type")
        shared._require_exact_keys(record["bindings"], frozenset(_BINDING_KEYS), "bindings")
        values = {name: record[name] for name in fields}
        values["bindings"] = RepresentativeHandoffBindings(**record["bindings"])
        return cls(**values)

    def to_record(self) -> dict[str, Any]:
        return {"record_type": REPRESENTATIVE_HANDOFF_SOURCE_RECORD_TYPE, **asdict(self)}

    def validate_runtime(
        self, *, context_tokens: int, arm_id: str, package_wheel_sha256: str,
        runtime_closure_manifest_sha256: str,
    ) -> None:
        if (self.context_tokens, self.arm_id) != (context_tokens, arm_id):
            raise ValueError("representative source context/arm differs from workload profile")
        if (
            self.bindings.package_wheel_sha256 != package_wheel_sha256
            or self.bindings.native_runtime_closure_sha256 != runtime_closure_manifest_sha256
        ):
            raise ValueError("representative source wheel/runtime closure bindings differ")

    def local_path(self, field: str) -> Path:
        if field not in {"manifest_uri", "source_root", "local_stage_root"}:
            raise ValueError("unknown representative source path")
        value = getattr(self, field)
        if value.startswith("dbfs:/Volumes/"):
            return Path(value.removeprefix("dbfs:"))
        return Path("/dbfs" + value.removeprefix("dbfs:") if value.startswith("dbfs:") else value)

    def read_manifest(self) -> dict[str, Any]:
        path = shared._existing_regular_file(self.local_path("manifest_uri"), field_name="manifest")
        content = path.read_bytes()
        if sha256(content).hexdigest() != self.manifest_file_sha256:
            raise ValueError("representative handoff manifest file SHA-256 differs")
        record = json.loads(content)
        return _manifest(record, self.bindings, self.bundle_closed_record_sha256)


@dataclass(frozen=True, slots=True)
class RepresentativeSupplementProvenanceV1:
    """Caller-pinned source/runner metadata for latency and resource claims.

    source_tree_sha256 is the existing source-closure JSON's file SHA256, not a
    Git tree object ID. The associated handoff source supplies commit/wheel and
    native-runtime pins. This record grants no publication authority.
    """

    source_closure_uri: str
    source_tree_sha256: str
    runner_sha256: str
    measurement_scopes: tuple[str, ...] = ("latency", "resource")

    def __post_init__(self) -> None:
        for name in ("source_tree_sha256", "runner_sha256"):
            shared._required_sha256(getattr(self, name), name)
        uri = self.source_closure_uri
        if (
            not isinstance(uri, str) or not uri.startswith(("dbfs:/", "/dbfs/", "/Volumes/"))
            or uri != uri.strip() or "\\" in uri
            or str(Path(uri.removeprefix("dbfs:"))) != uri.removeprefix("dbfs:")
            or ".." in Path(uri.removeprefix("dbfs:")).parts
        ):
            raise ValueError("source_closure_uri must be a normalized persistent cluster path")
        if self.measurement_scopes != ("latency", "resource"):
            raise ValueError("supplement measurement scopes must be exactly latency/resource")

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "record_type": REPRESENTATIVE_SUPPLEMENT_PROVENANCE_RECORD_TYPE,
            "source_closure_uri": self.source_closure_uri,
            "source_tree_sha256": self.source_tree_sha256,
            "runner_sha256": self.runner_sha256,
            "measurement_scopes": list(self.measurement_scopes),
        }
        record["closed_record_sha256"] = shared._closed_record_sha256(record)
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RepresentativeSupplementProvenanceV1:
        shared._require_exact_keys(record, frozenset({
            "record_type", "source_closure_uri", "source_tree_sha256", "runner_sha256",
            "measurement_scopes", "closed_record_sha256",
        }), "supplement provenance")
        if record["record_type"] != REPRESENTATIVE_SUPPLEMENT_PROVENANCE_RECORD_TYPE:
            raise ValueError("unsupported supplement provenance version")
        if record["closed_record_sha256"] != shared._closed_record_sha256(record):
            raise ValueError("supplement provenance closed SHA-256 differs")
        if record["measurement_scopes"] != ["latency", "resource"]:
            raise ValueError("supplement measurement scopes must be exactly latency/resource")
        return cls(record["source_closure_uri"], record["source_tree_sha256"], record["runner_sha256"])

    def software_identity_packages(self, bindings: RepresentativeHandoffBindings) -> dict[str, str]:
        return {
            "cachet-source": "git:" + bindings.source_commit,
            "cachet-source-tree": "sha256:" + self.source_tree_sha256,
            "cachet-kv": "wheel-sha256:" + bindings.package_wheel_sha256,
            "cachet-runner": "sha256:" + self.runner_sha256,
        }


def close_representative_handoff_bundle(
    bundle_root: str | Path,
    combined_dataset_paths: Mapping[int, str | Path],
    *,
    bindings: RepresentativeHandoffBindings,
    token_counter: Callable[[str], int],
    example_count: int = 32,
) -> dict[str, Any]:
    """Close exact HotpotQA inputs and both methods for each 8k/16k context.

The source tree must contain only the combined JSONLs and referenced external
handoff/payload files. Save the returned manifest outside that immutable tree.
The tokenizer callback must use the frozen model tokenizer, without special
tokens. A smaller explicit example_count supports diagnostics, not publication.
"""
    if not callable(token_counter):
        raise TypeError("token_counter must be callable")
    root = shared._existing_real_directory(bundle_root, field_name="bundle_root")
    paths = _context_paths(root, combined_dataset_paths)
    return _build(root, paths, bindings, example_count, token_counter=token_counter)


def validate_representative_handoff_bundle(
    record: Mapping[str, Any],
    *,
    bundle_root: str | Path,
    expected_bindings: RepresentativeHandoffBindings,
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    """Verify a caller-pinned manifest against all immutable source bytes."""
    manifest = _manifest(record, expected_bindings, expected_bundle_sha256)
    root = shared._existing_real_directory(bundle_root, field_name="bundle_root")
    _verify_content(manifest, root)
    return manifest


def stage_representative_handoff_bundle(
    record: Mapping[str, Any],
    *,
    source_root: str | Path,
    local_nvme_dir: str | Path,
    expected_bindings: RepresentativeHandoffBindings,
    expected_bundle_sha256: str,
) -> StagedRepresentativeHandoffs:
    """Copy exact artifact bytes and atomically publish local arm projections.

Handoff JSONs, payloads and original combined inputs are never rewritten. Only
new one-arm JSONLs contain local path overrides. No tokenizer or generator runs.
"""
    manifest = validate_representative_handoff_bundle(
        record, bundle_root=source_root, expected_bindings=expected_bindings,
        expected_bundle_sha256=expected_bundle_sha256,
    )
    source = shared._existing_real_directory(source_root, field_name="source_root")
    target = shared._nonexistent_output_directory(local_nvme_dir)
    shared._reject_overlap(source, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shared._reject_symlink_path(target.parent, include_leaf=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    published_inode: int | None = None
    try:
        for item in manifest["files"]:
            relative = shared._validated_relative_name(item["relative_name"], field_name="file")
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination, follow_symlinks=False)
            shared._verify_file_record(destination, item, label="copied artifact")
        _verify_content(manifest, temporary)
        projections = _projection_bytes(manifest, temporary, final_root=target)
        for name, content in projections.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
        attestation = _attestation(manifest, target, projections)
        with (temporary / REPRESENTATIVE_HANDOFF_STAGING_FILENAME).open("xb") as stream:
            stream.write(shared._canonical_json_bytes(attestation, pretty=True))
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite stage: {target}")
        inode = temporary.stat().st_ino
        os.rename(temporary, target)
        published_inode = inode
        result = verify_staged_representative_handoff_bundle(
            manifest, staged_root=target, expected_bindings=expected_bindings,
            expected_bundle_sha256=expected_bundle_sha256,
        )
        published_inode = None
        return result
    finally:
        if published_inode is not None and target.is_dir() and not target.is_symlink():
            if target.stat().st_ino == published_inode:
                shutil.rmtree(target)
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_staged_representative_handoff_bundle(
    record: Mapping[str, Any],
    *,
    staged_root: str | Path,
    expected_bindings: RepresentativeHandoffBindings,
    expected_bundle_sha256: str,
) -> StagedRepresentativeHandoffs:
    """Recheck exact artifact bytes and every local projection/attestation."""
    manifest = _manifest(record, expected_bindings, expected_bundle_sha256)
    root = shared._existing_real_directory(staged_root, field_name="staged_root")
    projections = _projection_bytes(manifest, root, final_root=root)
    extra = set(projections) | {REPRESENTATIVE_HANDOFF_STAGING_FILENAME}
    _verify_content(manifest, root, extra_files=extra)
    for name, content in projections.items():
        path = shared._existing_regular_file(root / name, field_name="projection")
        if path.read_bytes() != content:
            raise ValueError("staged projection differs from immutable request identity")
    attestation_path = shared._existing_regular_file(
        root / REPRESENTATIVE_HANDOFF_STAGING_FILENAME, field_name="staging attestation",
    )
    attestation = _attestation(manifest, root, projections)
    if attestation_path.read_bytes() != shared._canonical_json_bytes(attestation, pretty=True):
        raise ValueError("staging attestation differs from verified content")
    paths = {
        (context["context_tokens"], arm): root / _projection_name(context["context_tokens"], arm)
        for context in manifest["contexts"] for arm in _ARMS
    }
    return StagedRepresentativeHandoffs(root, attestation_path, paths, attestation)


def _manifest(
    record: Mapping[str, Any], bindings: RepresentativeHandoffBindings, expected: str,
) -> dict[str, Any]:
    if not isinstance(bindings, RepresentativeHandoffBindings):
        raise TypeError("expected_bindings must be RepresentativeHandoffBindings")
    shared._required_sha256(expected, "expected_bundle_sha256")
    shared._require_exact_keys(record, frozenset(_RECORD_KEYS), "representative bundle")
    if record["record_type"] != REPRESENTATIVE_HANDOFF_BUNDLE_RECORD_TYPE:
        raise ValueError("unsupported representative bundle record_type")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("unsupported representative bundle schema_version")
    if record["bindings"] != bindings.to_record():
        raise ValueError("source/native/prepared bindings differ")
    if record["closed_record_sha256"] != expected or shared._closed_record_sha256(record) != expected:
        raise ValueError("representative bundle SHA-256 differs from pinned identity")
    result = copy.deepcopy(dict(record))
    contexts = shared._mapping_sequence(result["contexts"], "contexts")
    counts = [c.get("context_tokens") for c in contexts]
    if (
        not counts or any(type(c) is not int or c not in {8192, 16384} for c in counts)
        or counts != sorted(set(int(c) for c in counts if type(c) is int))
    ):
        raise ValueError("contexts must be unique ordered 8192/16384 entries")
    return result


def _context_paths(root: Path, paths: Mapping[int, str | Path]) -> dict[int, Path]:
    if not paths or any(type(c) is not int or c not in {8192, 16384} for c in paths):
        raise ValueError("context paths must use 8192 and/or 16384")
    result = {}
    for context in sorted(paths):
        path = shared._existing_regular_file(Path(paths[context]), field_name="combined JSONL")
        shared._relative_name_for_path(root, path, field_name="combined JSONL")
        result[context] = path
    return result


def _verify_content(
    manifest: Mapping[str, Any], root: Path, *, extra_files: set[str] | None = None,
) -> None:
    paths = {
        context["context_tokens"]: root / shared._validated_relative_name(
            context["relative_name"], field_name="context dataset",
        ) for context in manifest["contexts"]
    }
    bindings = RepresentativeHandoffBindings(**manifest["bindings"])
    rebuilt = _build(
        root, paths, bindings, manifest["example_count"], expected=manifest,
        extra_files=extra_files,
    )
    if rebuilt != manifest:
        raise ValueError("representative content differs from its closed manifest")


def _build(
    root: Path, paths: Mapping[int, Path], bindings: RepresentativeHandoffBindings,
    example_count: int, *, token_counter: Callable[[str], int] | None = None,
    expected: Mapping[str, Any] | None = None, extra_files: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(bindings, RepresentativeHandoffBindings):
        raise TypeError("bindings must be RepresentativeHandoffBindings")
    if type(example_count) is not int or example_count < 2:
        raise ValueError("example_count must be an integer of at least two")
    files: dict[str, dict[str, Any]] = {}
    contexts: list[dict[str, Any]] = []
    expected_contexts = {} if expected is None else {
        c["context_tokens"]: c for c in expected["contexts"]
    }
    for context, path in sorted(paths.items()):
        path = shared._existing_regular_file(path, field_name="combined JSONL")
        relative = shared._relative_name_for_path(root, path, field_name="combined JSONL")
        content = path.read_bytes()
        records = shared._canonical_jsonl_records(content, field_name="combined JSONL")
        examples = load_benchmark_jsonl(path, dataset="hotpotqa", require_dataset=True)
        if len(examples) != example_count or any(len(e.documents) < 2 for e in examples):
            raise ValueError("exact distinct multi-document HotpotQA membership is required")
        if len({e.example_id for e in examples}) != example_count:
            raise ValueError("duplicate representative example_id")
        _validate_combined_params(records)
        if any("kv_transfer_params" in row for row in records):
            raise ValueError("combined inputs must contain only per-arm handoffs")
        shared._add_file_record(files, relative, role="combined_dataset", content=content)
        previous = expected_contexts.get(context)
        counter: Callable[[str], int]
        if previous is not None:
            # These are previously measured, digest-pinned counts, not invented
            # tokenization. Recheck every prompt hash against its closed count.
            counts = {r["logical_prompt_sha256"]: r["logical_tokens"] for r in previous["logical_token_counts"]}
            if any(type(value) is not int or value != context for value in counts.values()):
                raise ValueError("sealed logical token counts must match the context")

            def closed_count(text: str) -> int:
                digest = sha256(text.encode("utf-8")).hexdigest()
                if digest not in counts:
                    raise ValueError("logical prompt differs from sealed token counts")
                return int(counts[digest])

            counter = closed_count
        else:
            assert token_counter is not None
            counter = token_counter
        token_rows = _logical_token_rows(examples, token_counter=counter, input_tokens_target=context)
        methods: dict[str, dict[str, Any]] = {}
        manifests = {}
        normalized = copy.deepcopy([dict(row) for row in records])
        for arm in _ARMS[1:]:
            entries, handoffs = [], []
            previous_entries = {} if previous is None else {
                e["row_index"]: e for e in previous["methods"][arm]["entries"]
            }
            identity: dict[str, Any] | None = None
            layout: dict[str, Any] | None = None
            for index, row in enumerate(records):
                params = row["arm_kv_transfer_params"][arm]
                scope = "arm_kv_transfer_params/" + arm
                analyzed = shared._analyze_binding(
                    root, dataset="hotpotqa", example_id=examples[index].example_id,
                    row_index=index, binding=shared._TransferBinding(scope, params), files=files,
                    expected_entry=previous_entries.get(index),
                )
                artifact = ArtifactIdentity.from_record(analyzed.artifact_identity)
                _representative_identity(artifact)
                if identity is None:
                    identity, layout = analyzed.artifact_identity, analyzed.layout_identity
                elif identity != analyzed.artifact_identity or layout != analyzed.layout_identity:
                    raise ValueError("method sub-bundle contains mixed artifact/layout identities")
                entry = analyzed.entry
                entries.append(entry)
                handoffs.append(BenchmarkHandoffEntry(
                    dataset="hotpotqa", example_id=examples[index].example_id,
                    request_id=params[DOCUMENT_KV_REQUEST_ID_PARAM],
                    handoff_json=str(root / entry["handoff_relative_name"]),
                    payload_uri=str(root / entry["payload_relative_name"]),
                    cache_method=params[DOCUMENT_KV_CACHE_METHOD_PARAM],
                    artifact_id=params[DOCUMENT_KV_ARTIFACT_ID_PARAM],
                ))
                shared._rewrite_binding_paths(
                    normalized[index], scope,
                    handoff_value=shared._portable_path(entry["handoff_relative_name"]),
                    payload_value=shared._portable_path(entry["payload_relative_name"]),
                )
            handoff_manifest = BenchmarkHandoffManifest(tuple(handoffs))
            manifests[arm] = handoff_manifest
            methods[arm] = {
                "artifact_identity": identity, "layout_identity": layout, "entries": entries,
                "topology_attestation": build_handoff_topology_attestation(
                    path, handoff_manifest, token_counter=counter,
                ),
            }
        _validate_manifest_pair(
            manifests[FULL_PREFIX_CANARY_ARM], manifests[VANILLA_CANARY_ARM],
            document_counts={(e.dataset, e.example_id): len(e.documents) for e in examples},
        )
        contexts.append({
            "context_tokens": context, "relative_name": relative,
            "logical_sample_digest": _logical_sample_digest(examples),
            "logical_token_counts": token_rows, "methods": methods,
            "normalized_records_sha256": shared._canonical_sha256(normalized),
        })
    for name in files:
        if name == REPRESENTATIVE_HANDOFF_STAGING_FILENAME or name.split("/")[0] == _PROJECTIONS:
            raise ValueError("source file collides with reserved staging paths")
    shared._verify_exact_source_tree(root, expected_relative_names=set(files) | (extra_files or set()))
    portable = [
        {"context_tokens": c["context_tokens"], "logical_sample_digest": c["logical_sample_digest"],
         "requests": [e["request_identity_sha256"] for method in c["methods"].values() for e in method["entries"]]}
        for c in contexts
    ]
    result = {
        "record_type": REPRESENTATIVE_HANDOFF_BUNDLE_RECORD_TYPE, "schema_version": 1,
        "bindings": bindings.to_record(), "example_count": example_count, "contexts": contexts,
        "files": [files[name] for name in sorted(files)],
        "portable_identity_sha256": shared._canonical_sha256(portable),
    }
    result["closed_record_sha256"] = shared._closed_record_sha256(result)
    return result


def _representative_identity(identity: ArtifactIdentity) -> None:
    expected = {
        "model_id": REPRESENTATIVE_CANARY_MODEL_ID,
        "tokenizer_id": REPRESENTATIVE_CANARY_MODEL_ID,
        "model_revision": REPRESENTATIVE_CANARY_MODEL_REVISION,
        "tokenizer_revision": REPRESENTATIVE_CANARY_MODEL_REVISION,
        "lora_id": "base",
        "kv_dtype": "bfloat16", "runtime_kv_dtype": "bfloat16",
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
    }
    if identity.has_unresolved_fields or any(getattr(identity, k) != v for k, v in expected.items()):
        raise ValueError("representative artifact requires pinned Qwen3 BF16 identities and TP/PP=1")


def _projection_name(context: int, arm: str) -> str:
    return f"{_PROJECTIONS}/{context}/{arm.replace(':', '-')}.jsonl"


def _projection_bytes(manifest: Mapping[str, Any], root: Path, *, final_root: Path) -> dict[str, bytes]:
    projections = {}
    for context in manifest["contexts"]:
        path = root / shared._validated_relative_name(context["relative_name"], field_name="dataset")
        path = shared._existing_regular_file(path, field_name="combined JSONL")
        records = shared._canonical_jsonl_records(path.read_bytes(), field_name="combined JSONL")
        entries = [e for method in context["methods"].values() for e in method["entries"]]
        rewritten = shared._rewrite_dataset_records_for_stage(
            records, dataset_record={"entries": entries}, final_root=final_root,
        )
        for arm in _ARMS:
            projections[_projection_name(context["context_tokens"], arm)] = shared._canonical_jsonl_bytes(
                _project_records(rewritten, arm),
            )
    return projections


def _attestation(manifest: Mapping[str, Any], root: Path, projections: Mapping[str, bytes]) -> dict[str, Any]:
    result = {
        "record_type": REPRESENTATIVE_HANDOFF_STAGING_RECORD_TYPE, "schema_version": 1,
        "bundle_closed_record_sha256": manifest["closed_record_sha256"],
        "portable_identity_sha256": manifest["portable_identity_sha256"],
        "bindings": manifest["bindings"], "staged_root": str(root),
        "immutable_files": manifest["files"],
        "contexts": manifest["contexts"],
        "projections": [
            {"relative_name": name, "byte_count": len(content), "sha256": sha256(content).hexdigest()}
            for name, content in sorted(projections.items())
        ],
    }
    result["closed_record_sha256"] = shared._closed_record_sha256(result)
    return result
