from __future__ import annotations

import math
import os
import re
import statistics
import string
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from html import escape
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from document_kv_cache.methods import MethodRegistry

from document_kv_cache._hardware_targets import (
    DEFAULT_HARDWARE_TARGET,
    SUPPORTED_V1_HARDWARE_TARGETS,
    validate_v1_hardware_target as _validate_v1_hardware_target,
)
from document_kv_cache.benchmark_metrics import (
    aggregate_decode_tokens_per_second,
    latency_speedup,
    quality_delta,
    request_decode_tokens_per_second,
)
from document_kv_cache.models import DocumentKVRequest
from document_kv_cache.workflow import SourceDocument


SUPPORTED_V1_DATASETS = ("biography", "hotpotqa", "musique", "niah")
DEFAULT_V1_MODEL_ID = "qwen3:4b-instruct"
DEFAULT_V1_LORA_ID = "base"
DEFAULT_V1_PROMPT_TEMPLATE_VERSION = "v2-final-answer"
BENCHMARK_CACHE_PREFIX_CHUNK_ID = "cache_prefix"
BENCHMARK_CACHE_ARTIFACT_PREFIX = "cachet"
BASELINE_PREFILL_ARM = "baseline_prefill"
CACHE_REUSE_ARM = "document_kv_cache"
DOCUMENT_KV_REQUEST_ID_PARAM = "document_kv.request_id"
DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM = "document_kv.benchmark_request_id"
DOCUMENT_KV_HANDOFF_JSON_PARAM = "document_kv.handoff_json"
DOCUMENT_KV_HANDOFF_RECORD_PARAM = "document_kv.handoff_record"
DOCUMENT_KV_PAYLOAD_URI_PARAM = "document_kv.payload_uri"
DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM = "document_kv.prompt_text_mode"
DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM = "document_kv.runtime_prefix_text"
DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM = "document_kv.sglang_hicache_page_keys"
DOCUMENT_KV_CACHE_METHOD_PARAM = "document_kv.cache_method"
DOCUMENT_KV_ARTIFACT_ID_PARAM = "document_kv.artifact_id"
FINAL_ANSWER_CUE = "<final_answer>answer</final_answer>"
FINAL_ANSWER_PARSER_ID = "cachet.single_final_answer"
FINAL_ANSWER_PARSER_VERSION = "1"
FINAL_ANSWER_PARSER_PLUGIN_PATH = (
    "document_kv_cache.benchmarks:extract_single_final_answer"
)
FINAL_ANSWER_PARSER_STATUSES = (
    "ok",
    "missing_block",
    "multiple_or_malformed_blocks",
    "extraneous_text",
    "nested_block",
    "empty_answer",
)
FINAL_ANSWER_PARSER_CONTRACT = (
    "utf8-trim;case-sensitive-lowercase-tags;whole-output-match;"
    "exactly-one-nonempty-final_answer-block;no-nested-answer-tags;"
    "statuses="
    + ",".join(FINAL_ANSWER_PARSER_STATUSES)
)
FINAL_ANSWER_PARSER_DIGEST = sha256(
    FINAL_ANSWER_PARSER_CONTRACT.encode("utf-8")
).hexdigest()
FINAL_ANSWER_METADATA_PREFIX = "cachet.score.final_answer_parser"
FINAL_ANSWER_EXTRACTED_METADATA_KEY = "cachet.score.extracted_answer"
FINAL_ANSWER_NO_EXTRACTION_VALUE = "<no-valid-answer>"
FINAL_ANSWER_PARSER_ID_METADATA_KEY = f"{FINAL_ANSWER_METADATA_PREFIX}.id"
FINAL_ANSWER_PARSER_VERSION_METADATA_KEY = f"{FINAL_ANSWER_METADATA_PREFIX}.version"
FINAL_ANSWER_PARSER_PLUGIN_METADATA_KEY = f"{FINAL_ANSWER_METADATA_PREFIX}.plugin_path"
FINAL_ANSWER_PARSER_DIGEST_METADATA_KEY = f"{FINAL_ANSWER_METADATA_PREFIX}.digest"
FINAL_ANSWER_PARSER_VALID_METADATA_KEY = f"{FINAL_ANSWER_METADATA_PREFIX}.valid"
FINAL_ANSWER_PARSER_STATUS_METADATA_KEY = f"{FINAL_ANSWER_METADATA_PREFIX}.status"
DEFAULT_V1_PROMPT_PLUGIN_PATH = "document_kv_cache.benchmarks:_default_prompt_parts"
BIOGRAPHY_TITLE_NORMALIZER_ID = "nfkc_casefold_ws_terminal_punctuation_v2"
BIOGRAPHY_SCORER_VERSION = (
    f"biography_entity_identification_v2+{BIOGRAPHY_TITLE_NORMALIZER_ID}+"
    f"final_answer_v1@"
    f"{FINAL_ANSWER_PARSER_DIGEST[:12]}"
)
HOTPOTQA_SCORER_VERSION = (
    f"hotpot_evaluate_v1@3635853403a8+final_answer_v1@"
    f"{FINAL_ANSWER_PARSER_DIGEST[:12]}"
)
MUSIQUE_OFFICIAL_COMMIT = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
MUSIQUE_ANSWER_SCORER_SHA256 = (
    "10368f619b4d5ef5d83748c05a96c0afd332a14ab5c010740c98d58dfaefe974"
)
MUSIQUE_SCORER_VERSION = (
    f"evaluate_v1.0@{MUSIQUE_OFFICIAL_COMMIT}+answer.py@"
    f"{MUSIQUE_ANSWER_SCORER_SHA256[:12]}+final_answer_v1@"
    f"{FINAL_ANSWER_PARSER_DIGEST[:12]}"
)
NIAH_SCORER_VERSION = (
    f"cachet_niah_grid_v1+final_answer_v1@{FINAL_ANSWER_PARSER_DIGEST[:12]}"
)
NIAH_CELL_IDS = tuple(
    f"niah-{context_tokens // 1024}k-depth-{round(position * 100):02d}"
    for context_tokens in (8192, 16384, 32768)
    for position in (0.1, 0.5, 0.9)
)
# Controls whether the system/task guidance prompt is placed at the start of the
# cached document prefix (baked into the cached KV) or after the documents so it is
# recomputed online. "end" moves the guidance out of the cached KV, so it is prefilled
# online with full attention over the injected document KV.
CACHET_BENCHMARK_SYSTEM_PROMPT_POSITION_ENV = "CACHET_BENCHMARK_SYSTEM_PROMPT_POSITION"
SYSTEM_PROMPT_POSITIONS = ("start", "end")
DEFAULT_SYSTEM_PROMPT_POSITION = "start"
BENCHMARK_ARM_IMPLEMENTATION_KINDS = ("baseline", "cachet", "upstream", "external")


class _FrozenList(list[Any]):
    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("evaluation JSON values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable  # type: ignore[assignment]
    __imul__ = _immutable  # type: ignore[assignment]
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

__all__ = [
    "SUPPORTED_V1_DATASETS",
    "SUPPORTED_V1_HARDWARE_TARGETS",
    "DEFAULT_V1_MODEL_ID",
    "DEFAULT_V1_LORA_ID",
    "DEFAULT_V1_PROMPT_TEMPLATE_VERSION",
    "BENCHMARK_CACHE_PREFIX_CHUNK_ID",
    "BENCHMARK_CACHE_ARTIFACT_PREFIX",
    "DEFAULT_HARDWARE_TARGET",
    "BASELINE_PREFILL_ARM",
    "CACHE_REUSE_ARM",
    "DOCUMENT_KV_REQUEST_ID_PARAM",
    "DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM",
    "DOCUMENT_KV_HANDOFF_JSON_PARAM",
    "DOCUMENT_KV_HANDOFF_RECORD_PARAM",
    "DOCUMENT_KV_PAYLOAD_URI_PARAM",
    "DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM",
    "DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM",
    "DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM",
    "DOCUMENT_KV_CACHE_METHOD_PARAM",
    "DOCUMENT_KV_ARTIFACT_ID_PARAM",
    "CACHET_BENCHMARK_SYSTEM_PROMPT_POSITION_ENV",
    "SYSTEM_PROMPT_POSITIONS",
    "DEFAULT_SYSTEM_PROMPT_POSITION",
    "BENCHMARK_ARM_IMPLEMENTATION_KINDS",
    "resolve_system_prompt_position",
    "BenchmarkDatasetSpec",
    "BenchmarkPromptParts",
    "FINAL_ANSWER_CUE",
    "FINAL_ANSWER_PARSER_ID",
    "FINAL_ANSWER_PARSER_VERSION",
    "FINAL_ANSWER_PARSER_PLUGIN_PATH",
    "FINAL_ANSWER_PARSER_STATUSES",
    "FINAL_ANSWER_PARSER_CONTRACT",
    "FINAL_ANSWER_PARSER_DIGEST",
    "FINAL_ANSWER_EXTRACTED_METADATA_KEY",
    "FINAL_ANSWER_NO_EXTRACTION_VALUE",
    "FINAL_ANSWER_PARSER_ID_METADATA_KEY",
    "FINAL_ANSWER_PARSER_VERSION_METADATA_KEY",
    "FINAL_ANSWER_PARSER_PLUGIN_METADATA_KEY",
    "FINAL_ANSWER_PARSER_DIGEST_METADATA_KEY",
    "FINAL_ANSWER_PARSER_VALID_METADATA_KEY",
    "FINAL_ANSWER_PARSER_STATUS_METADATA_KEY",
    "BIOGRAPHY_SCORER_VERSION",
    "BIOGRAPHY_TITLE_NORMALIZER_ID",
    "HOTPOTQA_SCORER_VERSION",
    "MUSIQUE_OFFICIAL_COMMIT",
    "MUSIQUE_ANSWER_SCORER_SHA256",
    "MUSIQUE_SCORER_VERSION",
    "NIAH_SCORER_VERSION",
    "NIAH_CELL_IDS",
    "BenchmarkExample",
    "BenchmarkSuite",
    "BenchmarkArm",
    "BenchmarkOfflineCosts",
    "DatasetScorer",
    "DatasetMetricSpec",
    "DatasetScoreContext",
    "FinalAnswerExtraction",
    "DatasetScorerRegistry",
    "InferenceMeasurement",
    "LatencySummary",
    "BenchmarkReportRow",
    "BenchmarkComparison",
    "V1BenchmarkEvidence",
    "baseline_prefill_arm",
    "document_kv_cache_arm",
    "external_benchmark_arm",
    "method_benchmark_arm",
    "require_runnable_cachet_benchmark_arm",
    "default_dataset_scorer_registry",
    "diagnostic_answer_scores",
    "extract_single_final_answer",
    "final_answer_measurement_metadata",
    "biography_entity_identification_scores",
    "normalize_biography_title",
    "hotpotqa_official_answer_scores",
    "musique_official_answer_scores",
    "niah_exact_value_scores",
    "niah_cell_identity",
    "v1_dataset_specs",
    "dataset_spec",
    "build_prompt_parts",
    "build_prefill_prompt",
    "build_cache_prefix_text",
    "build_cache_suffix_text",
    "benchmark_cache_artifact_stem",
    "benchmark_cache_document_id",
    "benchmark_cache_source_document",
    "benchmark_cache_request",
    "format_document_context",
    "summarize_measurements",
    "compare_to_baseline",
    "evaluate_v1_benchmark_evidence",
    "normalize_answer",
    "exact_match",
    "answer_found",
    "validate_v1_hardware_target",
    "validate_v1_dataset",
]


@dataclass(frozen=True, slots=True)
class DatasetScoreContext:
    dataset: str
    example_id: str
    output_text: str
    references: tuple[str, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.example_id, "example_id")
        if not isinstance(self.output_text, str):
            raise TypeError("output_text must be a string")
        references = tuple(self.references)
        if any(not isinstance(reference, str) or not reference for reference in references):
            raise ValueError("references must contain non-empty strings")
        object.__setattr__(self, "references", references)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_dict_from_str_mapping(self.metadata, "metadata")),
        )


ScoreFunction = Callable[[DatasetScoreContext], Mapping[str, float]]
PromptFunction = Callable[["BenchmarkExample"], "BenchmarkPromptParts"]
AnswerParserFunction = Callable[[str], "FinalAnswerExtraction"]


@dataclass(frozen=True, slots=True)
class FinalAnswerExtraction:
    """Auditable result of the publication answer-output parser."""

    raw_output: str
    extracted_answer: str
    valid: bool
    status: str
    parser_id: str = FINAL_ANSWER_PARSER_ID
    parser_version: str = FINAL_ANSWER_PARSER_VERSION
    parser_plugin_path: str = FINAL_ANSWER_PARSER_PLUGIN_PATH
    parser_digest: str = FINAL_ANSWER_PARSER_DIGEST

    def __post_init__(self) -> None:
        if not isinstance(self.raw_output, str):
            raise TypeError("raw_output must be a string")
        if not isinstance(self.extracted_answer, str):
            raise TypeError("extracted_answer must be a string")
        if type(self.valid) is not bool:
            raise TypeError("valid must be a boolean")
        for field_name in (
            "status",
            "parser_id",
            "parser_version",
            "parser_plugin_path",
            "parser_digest",
        ):
            _validate_non_empty_str(getattr(self, field_name), field_name)
        if self.status not in FINAL_ANSWER_PARSER_STATUSES:
            raise ValueError("status is outside the frozen final-answer parser states")
        if self.valid != (self.status == "ok"):
            raise ValueError("parser validity must be true exactly for status='ok'")
        if self.valid and (self.status != "ok" or not self.extracted_answer):
            raise ValueError("valid extraction requires status='ok' and a non-empty answer")
        if not self.valid and self.extracted_answer:
            raise ValueError("invalid extraction must not expose an extracted answer")
        if len(self.parser_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.parser_digest
        ):
            raise ValueError("parser_digest must be a lowercase SHA-256 digest")


def extract_single_final_answer(output_text: str) -> FinalAnswerExtraction:
    """Parse exactly one whole-output ``<final_answer>`` block.

    The tag spelling is intentionally case-sensitive. Any prose outside the block,
    empty content, nested answer tags, duplicate blocks, or malformed tags is an
    invalid answer and therefore receives zero for every registered metric.
    """

    if not isinstance(output_text, str):
        raise TypeError("output_text must be a string")
    open_tag = "<final_answer>"
    close_tag = "</final_answer>"
    open_count = output_text.count(open_tag)
    close_count = output_text.count(close_tag)
    if open_count == 0 and close_count == 0:
        return FinalAnswerExtraction(output_text, "", False, "missing_block")
    if open_count != 1 or close_count != 1:
        return FinalAnswerExtraction(output_text, "", False, "multiple_or_malformed_blocks")
    stripped = output_text.strip()
    if not stripped.startswith(open_tag) or not stripped.endswith(close_tag):
        return FinalAnswerExtraction(output_text, "", False, "extraneous_text")
    answer = stripped[len(open_tag) : -len(close_tag)].strip()
    if "<final_answer" in answer or "</final_answer" in answer:
        return FinalAnswerExtraction(output_text, "", False, "nested_block")
    if not answer:
        return FinalAnswerExtraction(output_text, "", False, "empty_answer")
    return FinalAnswerExtraction(output_text, answer, True, "ok")


def final_answer_measurement_metadata(
    extraction: FinalAnswerExtraction,
) -> Mapping[str, str]:
    if not isinstance(extraction, FinalAnswerExtraction):
        raise TypeError("extraction must be a FinalAnswerExtraction")
    return MappingProxyType(
        {
            FINAL_ANSWER_EXTRACTED_METADATA_KEY: (
                extraction.extracted_answer
                if extraction.valid
                else FINAL_ANSWER_NO_EXTRACTION_VALUE
            ),
            FINAL_ANSWER_PARSER_ID_METADATA_KEY: extraction.parser_id,
            FINAL_ANSWER_PARSER_VERSION_METADATA_KEY: extraction.parser_version,
            FINAL_ANSWER_PARSER_PLUGIN_METADATA_KEY: extraction.parser_plugin_path,
            FINAL_ANSWER_PARSER_DIGEST_METADATA_KEY: extraction.parser_digest,
            FINAL_ANSWER_PARSER_VALID_METADATA_KEY: str(extraction.valid).lower(),
            FINAL_ANSWER_PARSER_STATUS_METADATA_KEY: extraction.status,
        }
    )


@dataclass(frozen=True, slots=True)
class DatasetMetricSpec:
    """Comparison semantics for one versioned dataset metric."""

    metric_name: str
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"
    max_regression: float = 0.02

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.metric_name, "metric_name")
        if self.direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError(
                "direction must be higher_is_better or lower_is_better"
            )
        if (
            isinstance(self.max_regression, bool)
            or not isinstance(self.max_regression, (int, float))
            or not math.isfinite(float(self.max_regression))
            or self.max_regression < 0
        ):
            raise ValueError("max_regression must be a non-negative finite number")


@dataclass(frozen=True, slots=True)
class DatasetScorer:
    """Versioned scorer used for one or more benchmark datasets."""

    scorer_id: str
    version: str
    metric_names: tuple[str, ...]
    score_function: ScoreFunction = field(repr=False, compare=False)
    publication_approved: bool = False
    plugin_path: str = ""
    metric_specs: tuple[DatasetMetricSpec, ...] = ()
    prompt_function: PromptFunction | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    prompt_plugin_path: str = ""
    prompt_template_version: str = ""
    answer_parser_function: AnswerParserFunction | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    answer_parser_id: str = ""
    answer_parser_version: str = ""
    answer_parser_plugin_path: str = ""
    answer_parser_digest: str = ""

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.scorer_id, "scorer_id")
        _validate_non_empty_str(self.version, "version")
        names = tuple(self.metric_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("metric_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("metric_names must not contain duplicates")
        if not callable(self.score_function):
            raise TypeError("score_function must be callable")
        if type(self.publication_approved) is not bool:
            raise ValueError("publication_approved must be a boolean")
        if not isinstance(self.plugin_path, str):
            raise TypeError("plugin_path must be a string")
        specs = tuple(self.metric_specs) or tuple(
            DatasetMetricSpec(metric_name=name) for name in names
        )
        if any(not isinstance(spec, DatasetMetricSpec) for spec in specs):
            raise TypeError("metric_specs entries must be DatasetMetricSpec")
        if tuple(spec.metric_name for spec in specs) != names:
            raise ValueError(
                "metric_specs must declare each metric_name once and in the same order"
            )
        if self.prompt_function is not None and not callable(self.prompt_function):
            raise TypeError("prompt_function must be callable when provided")
        for field_name in ("prompt_plugin_path", "prompt_template_version"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")
        if self.prompt_function is not None:
            _validate_non_empty_str(self.prompt_plugin_path, "prompt_plugin_path")
            _validate_non_empty_str(
                self.prompt_template_version,
                "prompt_template_version",
            )
        parser_function = self.answer_parser_function
        if self.publication_approved and parser_function is None:
            parser_function = extract_single_final_answer
            object.__setattr__(self, "answer_parser_function", parser_function)
        if parser_function is not None:
            if not callable(parser_function):
                raise TypeError("answer_parser_function must be callable when provided")
            parser_defaults = {
                "answer_parser_id": FINAL_ANSWER_PARSER_ID,
                "answer_parser_version": FINAL_ANSWER_PARSER_VERSION,
                "answer_parser_plugin_path": FINAL_ANSWER_PARSER_PLUGIN_PATH,
                "answer_parser_digest": FINAL_ANSWER_PARSER_DIGEST,
            }
            for field_name, default in parser_defaults.items():
                if not getattr(self, field_name):
                    object.__setattr__(self, field_name, default)
                _validate_non_empty_str(getattr(self, field_name), field_name)
            if len(self.answer_parser_digest) != 64 or any(
                character not in "0123456789abcdef"
                for character in self.answer_parser_digest
            ):
                raise ValueError(
                    "answer_parser_digest must be a lowercase SHA-256 digest"
                )
        elif any(
            getattr(self, field_name)
            for field_name in (
                "answer_parser_id",
                "answer_parser_version",
                "answer_parser_plugin_path",
                "answer_parser_digest",
            )
        ):
            raise ValueError(
                "answer parser identity requires answer_parser_function"
            )
        if self.publication_approved:
            _validate_non_empty_str(self.plugin_path, "plugin_path")
            if self.prompt_function is None:
                if not self.prompt_plugin_path:
                    object.__setattr__(
                        self,
                        "prompt_plugin_path",
                        DEFAULT_V1_PROMPT_PLUGIN_PATH,
                    )
                if not self.prompt_template_version:
                    object.__setattr__(
                        self,
                        "prompt_template_version",
                        DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
                    )
            _validate_non_empty_str(self.prompt_plugin_path, "prompt_plugin_path")
            _validate_non_empty_str(
                self.prompt_template_version,
                "prompt_template_version",
            )
        object.__setattr__(self, "metric_names", names)
        object.__setattr__(self, "metric_specs", specs)

    @property
    def identity(self) -> str:
        return f"{self.scorer_id}@{self.version}"

    def score(self, context: DatasetScoreContext) -> Mapping[str, float]:
        if not isinstance(context, DatasetScoreContext):
            raise TypeError("context must be DatasetScoreContext")
        raw = self.score_function(context)
        if not isinstance(raw, Mapping):
            raise TypeError("score_function must return a mapping")
        if not raw:
            return MappingProxyType({})
        scores: dict[str, float] = {}
        for metric_name in self.metric_names:
            value = raw.get(metric_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"scorer {self.identity} must return finite numeric metric {metric_name!r}"
                )
            scores[metric_name] = float(value)
        unexpected = set(raw).difference(self.metric_names)
        if unexpected:
            raise ValueError(
                f"scorer {self.identity} returned undeclared metrics: {sorted(unexpected)}"
            )
        return MappingProxyType(scores)

    def prompt_parts(self, example: "BenchmarkExample") -> "BenchmarkPromptParts":
        if self.prompt_function is None:
            return _default_prompt_parts(example)
        prompt_parts = self.prompt_function(example)
        if not isinstance(prompt_parts, BenchmarkPromptParts):
            raise TypeError("prompt_function must return BenchmarkPromptParts")
        return prompt_parts

    def parse_answer(self, raw_output: str) -> FinalAnswerExtraction | None:
        if self.answer_parser_function is None:
            return None
        extraction = self.answer_parser_function(raw_output)
        if not isinstance(extraction, FinalAnswerExtraction):
            raise TypeError(
                "answer_parser_function must return FinalAnswerExtraction"
            )
        expected_identity = (
            self.answer_parser_id,
            self.answer_parser_version,
            self.answer_parser_plugin_path,
            self.answer_parser_digest,
        )
        observed_identity = (
            extraction.parser_id,
            extraction.parser_version,
            extraction.parser_plugin_path,
            extraction.parser_digest,
        )
        if observed_identity != expected_identity:
            raise ValueError(
                "answer parser result identity does not match the scorer contract"
            )
        return extraction

    def zero_scores(self) -> Mapping[str, float]:
        return MappingProxyType({metric_name: 0.0 for metric_name in self.metric_names})


@dataclass(frozen=True, slots=True)
class DatasetScorerRegistry:
    """Immutable dataset-to-scorer registry."""

    entries: tuple[tuple[str, DatasetScorer], ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(self.entries)
        datasets: list[str] = []
        for dataset, scorer in normalized:
            _validate_non_empty_str(dataset, "dataset")
            if not isinstance(scorer, DatasetScorer):
                raise TypeError("scorer registry values must be DatasetScorer")
            datasets.append(dataset)
        if len(set(datasets)) != len(datasets):
            raise ValueError("scorer registry must not contain duplicate datasets")
        object.__setattr__(self, "entries", normalized)

    def register(self, dataset: str, scorer: DatasetScorer) -> "DatasetScorerRegistry":
        _validate_non_empty_str(dataset, "dataset")
        if not isinstance(scorer, DatasetScorer):
            raise TypeError("scorer must be a DatasetScorer")
        return DatasetScorerRegistry(
            tuple((key, value) for key, value in self.entries if key != dataset)
            + ((dataset, scorer),)
        )

    def get(self, dataset: str) -> DatasetScorer:
        _validate_non_empty_str(dataset, "dataset")
        for candidate, scorer in self.entries:
            if candidate == dataset:
                return scorer
        raise KeyError(f"No scorer is registered for dataset {dataset!r}")

    def identities(self, datasets: Sequence[str]) -> tuple[tuple[str, str, bool], ...]:
        return tuple(
            (
                dataset,
                self.get(dataset).identity,
                self.get(dataset).publication_approved,
            )
            for dataset in datasets
        )


def diagnostic_answer_scores(context: DatasetScoreContext) -> Mapping[str, float]:
    """Common answer diagnostics; not a substitute for an official dataset metric."""

    if not context.references:
        return {}
    expected_answer = context.references[0]
    return {
        "exact_match": float(exact_match(context.output_text, expected_answer)),
        "answer_found": float(answer_found(context.output_text, expected_answer)),
    }


def default_dataset_scorer_registry() -> DatasetScorerRegistry:
    shared: dict[str, Any] = {
        "publication_approved": True,
        "prompt_function": _default_prompt_parts,
        "prompt_plugin_path": DEFAULT_V1_PROMPT_PLUGIN_PATH,
        "prompt_template_version": DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
        "answer_parser_function": extract_single_final_answer,
        "answer_parser_id": FINAL_ANSWER_PARSER_ID,
        "answer_parser_version": FINAL_ANSWER_PARSER_VERSION,
        "answer_parser_plugin_path": FINAL_ANSWER_PARSER_PLUGIN_PATH,
        "answer_parser_digest": FINAL_ANSWER_PARSER_DIGEST,
    }
    biography = DatasetScorer(
        scorer_id="cachet.biography_entity_identification",
        version=BIOGRAPHY_SCORER_VERSION,
        metric_names=("exact_match",),
        metric_specs=(DatasetMetricSpec("exact_match", max_regression=1.0),),
        score_function=biography_entity_identification_scores,
        plugin_path=(
            "document_kv_cache.benchmarks:biography_entity_identification_scores"
        ),
        **shared,
    )
    hotpotqa = DatasetScorer(
        scorer_id="hotpotqa.official_answer",
        version=HOTPOTQA_SCORER_VERSION,
        metric_names=("exact_match", "f1"),
        metric_specs=(
            DatasetMetricSpec("exact_match", max_regression=1.0),
            DatasetMetricSpec("f1", max_regression=1.0),
        ),
        score_function=hotpotqa_official_answer_scores,
        plugin_path=(
            "document_kv_cache.benchmarks:hotpotqa_official_answer_scores"
        ),
        **shared,
    )
    musique = DatasetScorer(
        scorer_id="musique.official_answer",
        version=MUSIQUE_SCORER_VERSION,
        metric_names=("answer_em", "answer_f1"),
        metric_specs=(
            DatasetMetricSpec("answer_em", max_regression=1.0),
            DatasetMetricSpec("answer_f1", max_regression=1.0),
        ),
        score_function=musique_official_answer_scores,
        plugin_path="document_kv_cache.benchmarks:musique_official_answer_scores",
        **shared,
    )
    niah = DatasetScorer(
        scorer_id="cachet.niah_exact_value",
        version=NIAH_SCORER_VERSION,
        metric_names=("accuracy",),
        metric_specs=(DatasetMetricSpec("accuracy", max_regression=1.0),),
        score_function=niah_exact_value_scores,
        plugin_path="document_kv_cache.benchmarks:niah_exact_value_scores",
        **shared,
    )
    return DatasetScorerRegistry(
        (
            ("biography", biography),
            ("hotpotqa", hotpotqa),
            ("musique", musique),
            ("niah", niah),
        )
    )


def biography_entity_identification_scores(
    context: DatasetScoreContext,
) -> Mapping[str, float]:
    """Normalized-title exact match for Cachet's versioned biography task."""

    if not isinstance(context, DatasetScoreContext):
        raise TypeError("context must be a DatasetScoreContext")
    if not context.references:
        return MappingProxyType({"exact_match": 0.0})
    prediction = normalize_biography_title(context.output_text)
    exact = any(
        prediction == normalize_biography_title(reference)
        for reference in context.references
    )
    return MappingProxyType({"exact_match": float(exact)})


def normalize_biography_title(value: str) -> str:
    """Normalize an entity title without erasing name-significant punctuation.

    Unlike SQuAD-style QA normalization, this contract preserves articles,
    apostrophes, and internal hyphens. It applies Unicode NFKC, case-folding,
    whitespace collapse, and removes only surrounding terminal punctuation.
    """

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    start = 0
    end = len(normalized)
    while start < end and (
        normalized[start].isspace()
        or unicodedata.category(normalized[start]).startswith("P")
    ):
        start += 1
    while end > start and (
        normalized[end - 1].isspace()
        or unicodedata.category(normalized[end - 1]).startswith("P")
    ):
        end -= 1
    trimmed = normalized[start:end].strip()
    # WikiBio includes punctuation-only entity names such as ``!!!``. Erasing
    # those would make the task impossible and conflate unrelated predictions.
    return trimmed or normalized


def hotpotqa_official_answer_scores(
    context: DatasetScoreContext,
) -> Mapping[str, float]:
    """Return the official HotpotQA answer EM/F1 metrics for one prediction.

    This is an answer-only port of ``hotpot_evaluate_v1.py`` pinned by the
    scorer version above. Cachet does not claim the script's supporting-fact or
    joint metrics because its generation contract does not collect supporting
    fact predictions.
    """

    if not isinstance(context, DatasetScoreContext):
        raise TypeError("context must be a DatasetScoreContext")
    if not context.references:
        return MappingProxyType({"exact_match": 0.0, "f1": 0.0})
    prediction = _hotpotqa_normalize_answer(context.output_text)
    ground_truth = _hotpotqa_normalize_answer(context.references[0])
    exact = float(prediction == ground_truth)
    if (
        prediction in {"yes", "no", "noanswer"}
        or ground_truth in {"yes", "no", "noanswer"}
    ) and prediction != ground_truth:
        return MappingProxyType({"exact_match": exact, "f1": 0.0})
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    shared = sum(common.values())
    if shared == 0 or not prediction_tokens or not ground_truth_tokens:
        f1 = 0.0
    else:
        precision = shared / len(prediction_tokens)
        recall = shared / len(ground_truth_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    return MappingProxyType({"exact_match": exact, "f1": f1})


def musique_official_answer_scores(
    context: DatasetScoreContext,
) -> Mapping[str, float]:
    """Port MuSiQue v1.0 answer EM/F1, maximizing over answer aliases.

    This is the answer-only part of ``metrics.answer.AnswerMetric`` at
    ``MUSIQUE_OFFICIAL_COMMIT``. Cachet does not collect predicted support
    indices or answerability groups, so it does not claim those official metrics.
    """

    if not isinstance(context, DatasetScoreContext):
        raise TypeError("context must be a DatasetScoreContext")
    if not context.references:
        return MappingProxyType({"answer_em": 0.0, "answer_f1": 0.0})
    exact_scores = tuple(
        _musique_compute_exact(reference, context.output_text)
        for reference in context.references
    )
    f1_scores = tuple(
        _musique_compute_f1(reference, context.output_text)
        for reference in context.references
    )
    return MappingProxyType(
        {
            "answer_em": float(max(exact_scores)),
            "answer_f1": float(max(f1_scores)),
        }
    )


def _musique_normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in set(string.punctuation)
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _musique_compute_exact(gold: str, prediction: str) -> int:
    return int(_musique_normalize_answer(gold) == _musique_normalize_answer(prediction))


def _musique_compute_f1(gold: str, prediction: str) -> float:
    gold_tokens = _musique_normalize_answer(gold).split() if gold else []
    prediction_tokens = (
        _musique_normalize_answer(prediction).split() if prediction else []
    )
    common = Counter(gold_tokens) & Counter(prediction_tokens)
    shared = sum(common.values())
    if not gold_tokens or not prediction_tokens:
        return float(gold_tokens == prediction_tokens)
    if shared == 0:
        return 0.0
    precision = shared / len(prediction_tokens)
    recall = shared / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def niah_cell_identity(context_token_target: int, needle_position: float) -> str:
    if type(context_token_target) is not int or context_token_target not in {
        8192,
        16384,
        32768,
    }:
        raise ValueError("context_token_target must be one of 8192, 16384, or 32768")
    if isinstance(needle_position, bool) or not isinstance(
        needle_position, (int, float)
    ):
        raise TypeError("needle_position must be numeric")
    matched = next(
        (
            candidate
            for candidate in (0.1, 0.5, 0.9)
            if math.isclose(float(needle_position), candidate, abs_tol=1e-12)
        ),
        None,
    )
    if matched is None:
        raise ValueError("needle_position must be one of 0.1, 0.5, or 0.9")
    return f"niah-{context_token_target // 1024}k-depth-{round(matched * 100):02d}"


def niah_exact_value_scores(
    context: DatasetScoreContext,
) -> Mapping[str, float]:
    """Case-sensitive exact requested-value accuracy for the frozen NIAH grid."""

    if not isinstance(context, DatasetScoreContext):
        raise TypeError("context must be a DatasetScoreContext")
    cell_id = context.metadata.get("niah_cell_id")
    if cell_id is not None and cell_id not in NIAH_CELL_IDS:
        raise ValueError(
            "niah_cell_id metadata must identify a recognized publication grid cell"
        )
    if not context.references:
        return MappingProxyType({"accuracy": 0.0})
    prediction = context.output_text.strip()
    expected = context.references[0].strip()
    return MappingProxyType({"accuracy": float(prediction == expected)})


def _hotpotqa_normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


@dataclass(frozen=True, slots=True)
class BenchmarkDatasetSpec:
    dataset: str
    display_name: str
    task_instruction: str
    answer_instruction: str

    def __post_init__(self) -> None:
        validate_v1_dataset(self.dataset)
        if not self.display_name:
            raise ValueError("display_name must be non-empty")
        if not self.task_instruction:
            raise ValueError("task_instruction must be non-empty")
        if not self.answer_instruction:
            raise ValueError("answer_instruction must be non-empty")


@dataclass(frozen=True, slots=True)
class BenchmarkPromptParts:
    system_prompt: str
    document_context: str
    user_prompt: str
    system_prompt_position: str = DEFAULT_SYSTEM_PROMPT_POSITION

    def __post_init__(self) -> None:
        if self.system_prompt_position not in SYSTEM_PROMPT_POSITIONS:
            raise ValueError(
                f"system_prompt_position must be one of {SYSTEM_PROMPT_POSITIONS}; "
                f"got {self.system_prompt_position!r}"
            )

    @property
    def _ordered_sections(self) -> tuple[str, str, str]:
        # "end" places the guidance after the documents so it is not part of the cached
        # prefix and is recomputed online; "start" keeps it as the leading cached section.
        if self.system_prompt_position == "end":
            return (self.document_context, self.system_prompt, self.user_prompt)
        return (self.system_prompt, self.document_context, self.user_prompt)

    @property
    def prefill_prompt(self) -> str:
        return _join_sections(*self._ordered_sections)

    @property
    def cache_prefix_text(self) -> str:
        if self.system_prompt_position == "end":
            prefix = self.document_context
            has_suffix = bool(self.system_prompt or self.user_prompt)
        else:
            prefix = _join_sections(self.system_prompt, self.document_context)
            has_suffix = bool(self.user_prompt)
        # Own the existing section separator on the cached side of the split.
        # Keeping ``\n\n`` with the preceding section makes tokenizers that merge
        # a closing delimiter with the following newlines produce the same leading
        # token sequence for the standalone cached prefix and the full prompt.
        return f"{prefix}\n\n" if prefix and has_suffix else prefix

    @property
    def cache_suffix_text(self) -> str:
        prefix = self.cache_prefix_text
        if not prefix:
            return self.prefill_prompt
        # cache_prefix_text is always a leading substring of prefill_prompt, so the
        # online suffix is exactly the remainder (keeps prefix + suffix == prefill).
        return self.prefill_prompt[len(prefix) :]


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    example_id: str
    dataset: str
    documents: tuple[SourceDocument, ...]
    query: str
    expected_answer: str | None = None
    references: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    kv_transfer_params: Mapping[str, Any] = field(default_factory=dict)
    arm_kv_transfer_params: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.example_id, "example_id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.query, "query")
        if self.expected_answer is not None:
            _validate_non_empty_str(self.expected_answer, "expected_answer")
        references = tuple(self.references)
        if any(not isinstance(reference, str) or not reference for reference in references):
            raise ValueError("references must contain non-empty strings")
        if self.expected_answer is not None:
            if references and references[0] != self.expected_answer:
                raise ValueError(
                    "expected_answer must equal the first reference when both are provided"
                )
            if not references:
                references = (self.expected_answer,)
        elif references:
            object.__setattr__(self, "expected_answer", references[0])
        object.__setattr__(self, "references", references)
        documents = _tuple_from_sequence(self.documents, "documents")
        if not documents:
            raise ValueError("documents must include at least one SourceDocument")
        for index, document in enumerate(documents):
            if not isinstance(document, SourceDocument):
                raise TypeError(f"documents[{index}] must be a SourceDocument")
        object.__setattr__(self, "documents", documents)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_dict_from_str_mapping(self.metadata, "metadata")),
        )
        kv_transfer_params = _dict_from_json_object_mapping(self.kv_transfer_params, "kv_transfer_params")
        _validate_kv_transfer_params(kv_transfer_params)
        object.__setattr__(
            self,
            "kv_transfer_params",
            _deep_freeze_mapping(kv_transfer_params),
        )
        if not isinstance(self.arm_kv_transfer_params, Mapping):
            raise TypeError("arm_kv_transfer_params must be a mapping")
        arm_params: dict[str, Mapping[str, Any]] = {}
        for arm_id, raw_params in self.arm_kv_transfer_params.items():
            _validate_non_empty_str(arm_id, "arm_kv_transfer_params arm id")
            params = _dict_from_json_object_mapping(
                raw_params,
                f"arm_kv_transfer_params.{arm_id}",
            )
            _validate_kv_transfer_params(params)
            arm_params[arm_id] = _deep_freeze_mapping(params)
        object.__setattr__(
            self,
            "arm_kv_transfer_params",
            MappingProxyType(arm_params),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    suite_id: str
    examples: tuple[BenchmarkExample, ...]
    model_id: str = DEFAULT_V1_MODEL_ID
    hardware_target: str = DEFAULT_HARDWARE_TARGET
    datasets: tuple[str, ...] = SUPPORTED_V1_DATASETS

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.suite_id, "suite_id")
        _validate_non_empty_str(self.model_id, "model_id")
        _validate_non_empty_str(self.hardware_target, "hardware_target")
        examples = _tuple_from_sequence(self.examples, "examples")
        if not examples:
            raise ValueError("examples must include at least one BenchmarkExample")
        for index, example in enumerate(examples):
            if not isinstance(example, BenchmarkExample):
                raise TypeError(f"examples[{index}] must be a BenchmarkExample")
        duplicate_examples = _duplicate_labels(_example_key(example) for example in examples)
        if duplicate_examples:
            duplicate_ids = ", ".join(duplicate_examples)
            raise ValueError(f"examples contain duplicate dataset/example ids: {duplicate_ids}")
        datasets = _tuple_from_sequence(self.datasets, "datasets")
        if not datasets:
            raise ValueError("datasets must include at least one dataset")
        for dataset in datasets:
            _validate_non_empty_str(dataset, "dataset")
        duplicate_datasets = _duplicate_labels(datasets)
        if duplicate_datasets:
            raise ValueError(f"datasets contain duplicate dataset ids: {', '.join(duplicate_datasets)}")
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "datasets", datasets)
        example_datasets = {example.dataset for example in examples}
        missing = example_datasets.difference(datasets)
        if missing:
            raise ValueError(f"Examples reference datasets outside this suite: {sorted(missing)}")


@dataclass(frozen=True, slots=True)
class BenchmarkOfflineCosts:
    """Method preparation costs kept outside the online serving boundary."""

    training_seconds: float | None = None
    artifact_generation_seconds: float | None = None
    checkpoint_load_seconds: float | None = None
    artifact_bytes: int | None = None
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "training_seconds",
            "artifact_generation_seconds",
            "checkpoint_load_seconds",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _validate_non_negative_finite_number(value, field_name)
        for field_name in ("artifact_bytes", "peak_memory_bytes"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_non_negative_int(value, field_name)


@dataclass(frozen=True, slots=True)
class BenchmarkArm:
    arm_id: str
    uses_cache: bool
    description: str
    cache_method: str = ""
    connector_mode: str = ""
    variant_id: str = ""
    implementation_kind: Literal["baseline", "cachet", "upstream", "external"] | str = ""
    method_version: str = ""
    method_config_digest: str = ""
    physical_transform_id: str = "identity"
    physical_transform_version: str = "1"
    physical_transform_config_digest: str = ""
    scorer_plugin_path: str = ""
    offline_costs: BenchmarkOfflineCosts = field(default_factory=BenchmarkOfflineCosts)
    source_revision: str = ""
    checkpoint_identity: str = ""
    setting_overrides: Mapping[str, Any] = field(default_factory=dict)
    runtime_environment_overrides: Mapping[str, Any] = field(default_factory=dict)
    requires_cachet_handoff: bool | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.arm_id, "arm_id")
        if type(self.uses_cache) is not bool:
            raise ValueError("uses_cache must be a boolean")
        _validate_non_empty_str(self.description, "description")
        for field_name in (
            "cache_method",
            "connector_mode",
            "variant_id",
            "method_version",
            "method_config_digest",
            "physical_transform_id",
            "physical_transform_version",
            "physical_transform_config_digest",
            "scorer_plugin_path",
            "source_revision",
            "checkpoint_identity",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
        implementation_kind = self.implementation_kind or (
            "cachet" if self.uses_cache else "baseline"
        )
        if implementation_kind not in BENCHMARK_ARM_IMPLEMENTATION_KINDS:
            raise ValueError(
                "implementation_kind must be one of "
                f"{BENCHMARK_ARM_IMPLEMENTATION_KINDS}"
            )
        object.__setattr__(self, "implementation_kind", implementation_kind)
        requires_cachet_handoff = self.requires_cachet_handoff
        if requires_cachet_handoff is None:
            requires_cachet_handoff = self.uses_cache and implementation_kind == "cachet"
        if type(requires_cachet_handoff) is not bool:
            raise ValueError("requires_cachet_handoff must be a boolean")
        if requires_cachet_handoff and not self.uses_cache:
            raise ValueError("requires_cachet_handoff requires uses_cache")
        object.__setattr__(self, "requires_cachet_handoff", requires_cachet_handoff)
        if not self.uses_cache and self.cache_method:
            raise ValueError("non-cache benchmark arms must not declare cache_method")
        if not self.physical_transform_id:
            raise ValueError("physical_transform_id must be non-empty")
        if not self.physical_transform_version:
            raise ValueError("physical_transform_version must be non-empty")
        for field_name in ("method_config_digest", "physical_transform_config_digest"):
            value = getattr(self, field_name)
            if value and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest when provided")
        if not isinstance(self.offline_costs, BenchmarkOfflineCosts):
            raise TypeError("offline_costs must be BenchmarkOfflineCosts")
        setting_overrides = _dict_from_json_object_mapping(
            self.setting_overrides,
            "setting_overrides",
        )
        object.__setattr__(
            self,
            "setting_overrides",
            _deep_freeze_mapping(setting_overrides),
        )
        runtime_environment_overrides = _dict_from_json_object_mapping(
            self.runtime_environment_overrides,
            "runtime_environment_overrides",
        )
        object.__setattr__(
            self,
            "runtime_environment_overrides",
            _deep_freeze_mapping(runtime_environment_overrides),
        )


@dataclass(frozen=True, slots=True)
class InferenceMeasurement:
    example_id: str
    dataset: str
    arm_id: str
    prompt_tokens: int
    completion_tokens: int
    ttft_seconds: float
    time_to_completion_seconds: float
    output_text: str
    expected_answer: str | None = None
    error: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    cache_method: str = ""
    artifact_id: str = ""
    variant_id: str = ""
    request_id: str = ""
    repeat_index: int = 1
    scorer_id: str = ""
    scorer_version: str = ""
    quality_scores: Mapping[str, float] = field(default_factory=dict)
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.example_id, "example_id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.arm_id, "arm_id")
        _validate_non_negative_int(self.prompt_tokens, "prompt_tokens")
        _validate_non_negative_int(self.completion_tokens, "completion_tokens")
        _validate_non_negative_finite_number(self.ttft_seconds, "ttft_seconds")
        _validate_non_negative_finite_number(self.time_to_completion_seconds, "time_to_completion_seconds")
        if self.time_to_completion_seconds < self.ttft_seconds:
            raise ValueError("time_to_completion_seconds must be greater than or equal to ttft_seconds")
        _validate_str(self.output_text, "output_text")
        if self.expected_answer is not None:
            _validate_non_empty_str(self.expected_answer, "expected_answer")
        references = tuple(self.references)
        if any(not isinstance(reference, str) or not reference for reference in references):
            raise ValueError("references must contain non-empty strings")
        if self.expected_answer is not None:
            if references and references[0] != self.expected_answer:
                raise ValueError(
                    "expected_answer must equal the first reference when both are provided"
                )
            if not references:
                references = (self.expected_answer,)
        elif references:
            object.__setattr__(self, "expected_answer", references[0])
        object.__setattr__(self, "references", references)
        if self.error is not None:
            _validate_non_empty_str(self.error, "error")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_dict_from_str_mapping(self.metadata, "metadata")),
        )
        for field_name in ("cache_method", "artifact_id", "variant_id"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")
        if not isinstance(self.request_id, str):
            raise TypeError("request_id must be a string")
        if type(self.repeat_index) is not int or self.repeat_index <= 0:
            raise ValueError("repeat_index must be a positive integer")
        for field_name in ("scorer_id", "scorer_version"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")
        quality_scores: dict[str, float] = {}
        if not isinstance(self.quality_scores, Mapping):
            raise TypeError("quality_scores must be a mapping")
        for metric_name, value in self.quality_scores.items():
            if not isinstance(metric_name, str) or not metric_name:
                raise ValueError("quality_scores keys must be non-empty strings")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"quality_scores.{metric_name} must be finite numeric")
            quality_scores[metric_name] = float(value)
        object.__setattr__(self, "quality_scores", MappingProxyType(quality_scores))

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def exact_match(self) -> bool | None:
        explicit = self.quality_scores.get("exact_match")
        if explicit is not None:
            return explicit >= 0.5
        if self.expected_answer is None or not self.ok:
            return None
        return exact_match(self.output_text, self.expected_answer)

    @property
    def answer_found(self) -> bool | None:
        explicit = self.quality_scores.get("answer_found")
        if explicit is not None:
            return explicit >= 0.5
        if self.expected_answer is None or not self.ok:
            return None
        return answer_found(self.output_text, self.expected_answer)


@dataclass(frozen=True, slots=True)
class LatencySummary:
    count: int
    mean: float | None
    p50: float | None
    p95: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkReportRow:
    dataset: str
    arm_id: str
    requests: int
    errors: int
    prompt_tokens_mean: float | None
    completion_tokens_mean: float | None
    ttft: LatencySummary
    time_to_completion: LatencySummary
    exact_match_rate: float | None
    answer_found_rate: float | None
    output_tokens_per_second: float | None
    cache_method: str = ""
    artifact_id: str = ""
    variant_id: str = ""
    unique_examples: int = 0
    quality_score_means: Mapping[str, float] = field(default_factory=dict)
    request_decode_tokens_per_second: LatencySummary = field(
        default_factory=lambda: LatencySummary(count=0, mean=None, p50=None, p95=None)
    )
    aggregate_output_tokens_per_second: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quality_score_means",
            MappingProxyType(dict(self.quality_score_means)),
        )
        if self.aggregate_output_tokens_per_second is not None:
            _validate_non_negative_finite_number(
                self.aggregate_output_tokens_per_second,
                "aggregate_output_tokens_per_second",
            )


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    dataset: str
    baseline_arm_id: str
    cache_arm_id: str
    ttft_speedup: float | None
    time_to_completion_speedup: float | None
    exact_match_delta: float | None
    answer_found_delta: float | None
    cache_method: str = ""
    artifact_id: str = ""
    variant_id: str = ""
    quality_score_deltas: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quality_score_deltas",
            MappingProxyType(dict(self.quality_score_deltas)),
        )


@dataclass(frozen=True, slots=True)
class V1BenchmarkEvidence:
    required_datasets: tuple[str, ...]
    baseline_arm_id: str
    cache_arm_id: str
    duplicate_required_datasets: tuple[str, ...]
    duplicate_report_rows: tuple[str, ...]
    duplicate_comparisons: tuple[str, ...]
    missing_report_rows: tuple[str, ...]
    missing_comparisons: tuple[str, ...]
    comparisons_without_metrics: tuple[str, ...]
    rows_without_successful_requests: tuple[str, ...]
    rows_without_latency: tuple[str, ...]
    rows_without_quality: tuple[str, ...]
    unexpected_datasets: tuple[str, ...] = ()
    unexpected_arms: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not (
            self.missing_report_rows
            or self.missing_comparisons
            or self.duplicate_required_datasets
            or self.duplicate_report_rows
            or self.duplicate_comparisons
            or self.comparisons_without_metrics
            or self.rows_without_successful_requests
            or self.rows_without_latency
            or self.rows_without_quality
            or self.unexpected_arms
            or self.unexpected_datasets
        )

    @property
    def issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.duplicate_required_datasets:
            issues.append(f"duplicate required datasets: {', '.join(self.duplicate_required_datasets)}")
        if self.duplicate_report_rows:
            issues.append(f"duplicate report rows: {', '.join(self.duplicate_report_rows)}")
        if self.duplicate_comparisons:
            issues.append(f"duplicate comparisons: {', '.join(self.duplicate_comparisons)}")
        if self.missing_report_rows:
            issues.append(f"missing report rows: {', '.join(self.missing_report_rows)}")
        if self.missing_comparisons:
            issues.append(f"missing comparisons: {', '.join(self.missing_comparisons)}")
        if self.comparisons_without_metrics:
            issues.append(
                "comparisons without speedup or quality deltas: "
                f"{', '.join(self.comparisons_without_metrics)}"
            )
        if self.rows_without_successful_requests:
            issues.append(f"rows without successful requests: {', '.join(self.rows_without_successful_requests)}")
        if self.rows_without_latency:
            issues.append(f"rows without latency evidence: {', '.join(self.rows_without_latency)}")
        if self.rows_without_quality:
            issues.append(f"rows without quality evidence: {', '.join(self.rows_without_quality)}")
        if self.unexpected_arms:
            issues.append(f"unexpected arms: {', '.join(self.unexpected_arms)}")
        if self.unexpected_datasets:
            issues.append(f"unexpected datasets: {', '.join(self.unexpected_datasets)}")
        return tuple(issues)


def baseline_prefill_arm() -> BenchmarkArm:
    return BenchmarkArm(
        arm_id=BASELINE_PREFILL_ARM,
        uses_cache=False,
        description="Standard inference prefill that recomputes all document tokens.",
        implementation_kind="baseline",
        physical_transform_id="identity",
    )


def document_kv_cache_arm() -> BenchmarkArm:
    return BenchmarkArm(
        arm_id=CACHE_REUSE_ARM,
        uses_cache=True,
        description="Inference path that reuses precomputed document KV cache.",
        connector_mode="cachet",
        implementation_kind="cachet",
        physical_transform_id="cachet.prefix_reuse",
    )


def external_benchmark_arm(
    arm_id: str,
    *,
    description: str,
    implementation_kind: Literal["upstream", "external"] = "upstream",
    uses_cache: bool = True,
    method: str = "",
    method_version: str = "",
    method_config_digest: str = "",
    variant_id: str = "default",
    physical_transform_id: str,
    physical_transform_version: str,
    physical_transform_config_digest: str,
    offline_costs: BenchmarkOfflineCosts | None = None,
    source_revision: str,
    checkpoint_identity: str,
    setting_overrides: Mapping[str, Any] | None = None,
) -> BenchmarkArm:
    """Describe an author/upstream or other externally executed comparison arm."""

    return BenchmarkArm(
        arm_id=arm_id,
        uses_cache=uses_cache,
        description=description,
        cache_method=method if uses_cache else "",
        variant_id=variant_id,
        implementation_kind=implementation_kind,
        method_version=method_version,
        method_config_digest=method_config_digest,
        physical_transform_id=physical_transform_id,
        physical_transform_version=physical_transform_version,
        physical_transform_config_digest=physical_transform_config_digest,
        offline_costs=offline_costs or BenchmarkOfflineCosts(),
        source_revision=source_revision,
        checkpoint_identity=checkpoint_identity,
        setting_overrides=setting_overrides or {},
    )


def method_benchmark_arm(
    method: str,
    *,
    arm_id: str | None = None,
    variant_id: str = "default",
    registry: Any | None = None,
    method_config_digest: str = "",
    physical_transform_id: str | None = None,
    physical_transform_version: str = "1",
    physical_transform_config_digest: str = "",
    offline_costs: BenchmarkOfflineCosts | None = None,
    setting_overrides: Mapping[str, Any] | None = None,
) -> BenchmarkArm:
    """Create a cache arm directly from the executable method registry."""

    from document_kv_cache.methods import (
        CACHET_ARTIFACT_EXECUTION,
        MethodRegistry,
        default_method_registry,
    )

    resolved_registry = default_method_registry() if registry is None else registry
    if not isinstance(resolved_registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    spec = resolved_registry.get(method, require_implemented=True)
    if not method_config_digest:
        from document_kv_cache.artifact_identity import (
            method_config_digest as digest_method_config,
        )

        method_config_digest = digest_method_config({})
    return BenchmarkArm(
        arm_id=arm_id or f"{CACHE_REUSE_ARM}:{spec.method_id}",
        uses_cache=True,
        description=spec.description,
        cache_method=spec.method_id,
        connector_mode=spec.connector_mode,
        variant_id=variant_id,
        implementation_kind="cachet",
        method_version=spec.artifact_version,
        method_config_digest=method_config_digest,
        physical_transform_id=(
            physical_transform_id or f"cachet.{spec.method_id}.runtime_input"
        ),
        physical_transform_version=physical_transform_version,
        physical_transform_config_digest=physical_transform_config_digest,
        offline_costs=offline_costs or BenchmarkOfflineCosts(),
        setting_overrides=setting_overrides or {},
        requires_cachet_handoff=(
            spec.execution_kind == CACHET_ARTIFACT_EXECUTION
        ),
    )


def require_runnable_cachet_benchmark_arm(
    arm: BenchmarkArm,
    *,
    registry: MethodRegistry | None = None,
    allow_unidentified_smoke: bool = False,
) -> None:
    """Fail before a Cachet-labeled arm can claim an unregistered method."""

    from document_kv_cache.methods import (
        CACHET_ARTIFACT_EXECUTION,
        MethodRegistry,
        default_method_registry,
    )

    if not isinstance(arm, BenchmarkArm):
        raise TypeError("arm must be a BenchmarkArm")
    if arm.implementation_kind != "cachet":
        return
    resolved_registry = default_method_registry() if registry is None else registry
    if not isinstance(resolved_registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry or None")
    if not arm.uses_cache:
        raise ValueError("Cachet benchmark arms must use cache")
    if not arm.cache_method:
        if allow_unidentified_smoke:
            return
        raise ValueError("Cachet benchmark arms must declare cache_method")
    try:
        method = resolved_registry.get(arm.cache_method, require_implemented=True)
    except (KeyError, NotImplementedError) as exc:
        raise ValueError(
            f"Cachet benchmark arm {arm.arm_id!r} must name a registered runnable "
            "Cachet method"
        ) from exc
    if arm.method_version != method.artifact_version:
        raise ValueError(
            f"Cachet benchmark arm {arm.arm_id!r} method_version must match "
            f"{method.artifact_version!r}"
        )
    if arm.connector_mode != method.connector_mode:
        raise ValueError(
            f"Cachet benchmark arm {arm.arm_id!r} connector_mode must match "
            f"{method.connector_mode!r}"
        )
    if not arm.method_config_digest:
        raise ValueError(
            f"Cachet benchmark arm {arm.arm_id!r} must declare method_config_digest"
        )
    expected_handoff = method.execution_kind == CACHET_ARTIFACT_EXECUTION
    if arm.requires_cachet_handoff is not expected_handoff:
        raise ValueError(
            f"Cachet benchmark arm {arm.arm_id!r} requires_cachet_handoff must be "
            f"{str(expected_handoff).lower()} for the registered execution kind"
        )


def v1_dataset_specs() -> tuple[BenchmarkDatasetSpec, ...]:
    return tuple(_V1_DATASET_SPECS[dataset] for dataset in SUPPORTED_V1_DATASETS)


def dataset_spec(dataset: str) -> BenchmarkDatasetSpec:
    validate_v1_dataset(dataset)
    return _V1_DATASET_SPECS[dataset]


def resolve_system_prompt_position() -> str:
    """Resolve the system-prompt placement from the environment (default ``start``)."""

    value = os.environ.get(
        CACHET_BENCHMARK_SYSTEM_PROMPT_POSITION_ENV, DEFAULT_SYSTEM_PROMPT_POSITION
    ).strip().lower()
    if value not in SYSTEM_PROMPT_POSITIONS:
        raise ValueError(
            f"{CACHET_BENCHMARK_SYSTEM_PROMPT_POSITION_ENV} must be one of "
            f"{SYSTEM_PROMPT_POSITIONS}; got {value!r}"
        )
    return value


def build_prompt_parts(
    example: BenchmarkExample,
    *,
    scorer: DatasetScorer | None = None,
) -> BenchmarkPromptParts:
    if scorer is not None:
        if not isinstance(scorer, DatasetScorer):
            raise TypeError("scorer must be DatasetScorer when provided")
        return scorer.prompt_parts(example)
    return _default_prompt_parts(example)


def _default_prompt_parts(example: BenchmarkExample) -> BenchmarkPromptParts:
    if example.dataset not in SUPPORTED_V1_DATASETS:
        raise ValueError(
            f"Dataset {example.dataset!r} requires a registered scorer with a "
            "versioned prompt_function"
        )
    spec = dataset_spec(example.dataset)
    return BenchmarkPromptParts(
        system_prompt=_system_prompt(spec),
        document_context=format_document_context(example.documents),
        user_prompt=_user_prompt(example, spec),
        system_prompt_position=resolve_system_prompt_position(),
    )


def build_prefill_prompt(
    example: BenchmarkExample,
    *,
    scorer: DatasetScorer | None = None,
) -> str:
    return build_prompt_parts(example, scorer=scorer).prefill_prompt


def build_cache_prefix_text(
    example: BenchmarkExample,
    *,
    scorer: DatasetScorer | None = None,
) -> str:
    return build_prompt_parts(example, scorer=scorer).cache_prefix_text


def build_cache_suffix_text(
    example: BenchmarkExample,
    *,
    scorer: DatasetScorer | None = None,
) -> str:
    return build_prompt_parts(example, scorer=scorer).cache_suffix_text


def benchmark_cache_artifact_stem(
    example: BenchmarkExample,
    *,
    prefix: str = BENCHMARK_CACHE_ARTIFACT_PREFIX,
) -> str:
    """Return a stable path-safe stem for benchmark cache artifacts."""

    benchmark_example = _benchmark_example(example)
    prefix_slug = _artifact_slug(prefix, field_name="prefix")
    label = _artifact_slug(
        f"{benchmark_example.dataset}-{benchmark_example.example_id}",
        field_name="dataset/example_id",
    )
    digest = sha256(f"{benchmark_example.dataset}\0{benchmark_example.example_id}".encode("utf-8")).hexdigest()[:12]
    max_label_chars = max(8, 96 - len(prefix_slug) - len(digest) - 2)
    label = label[:max_label_chars].rstrip("-") or "example"
    return f"{prefix_slug}-{label}-{digest}"


def benchmark_cache_document_id(
    example: BenchmarkExample,
    *,
    prefix: str = BENCHMARK_CACHE_ARTIFACT_PREFIX,
) -> str:
    """Return the synthetic Cachet document id for this example's cached prefix."""

    return benchmark_cache_artifact_stem(example, prefix=prefix)


def _cache_prefix_chunk_id(index: int) -> str:
    return f"{BENCHMARK_CACHE_PREFIX_CHUNK_ID}-{index}"


def benchmark_cache_prefix_segments(
    example: BenchmarkExample,
    *,
    scorer: DatasetScorer | None = None,
) -> tuple[tuple[str, str], ...]:
    """Tile the exact cache-prefix text into one contiguous ``(chunk_id, text)`` per document.

    The concatenation of the returned texts equals ``build_cache_prefix_text(example)``
    byte-for-byte. Tokenizer-backed vLLM handoff generation additionally requires the
    independently tokenized segments to compose to an exact leading token prefix of the
    logical prompt before an artifact can be written. The leading system prompt +
    "Documents:" header rides on the first document's segment. Falls back to a single
    segment when there is one document or the per-document offsets cannot be located.
    """

    benchmark_example = _benchmark_example(example)
    prefix = build_cache_prefix_text(benchmark_example, scorer=scorer)
    documents = benchmark_example.documents
    if len(documents) <= 1:
        return ((BENCHMARK_CACHE_PREFIX_CHUNK_ID, prefix),)
    starts: list[int] = []
    cursor = 0
    for document in documents:
        formatted = _format_document(document)
        index = prefix.find(formatted, cursor)
        if index < 0:
            return ((BENCHMARK_CACHE_PREFIX_CHUNK_ID, prefix),)
        starts.append(index)
        cursor = index + len(formatted)
    bounds = [0, *starts[1:], len(prefix)]
    return tuple(
        (_cache_prefix_chunk_id(i), prefix[bounds[i] : bounds[i + 1]])
        for i in range(len(documents))
    )


def benchmark_cache_source_document(
    example: BenchmarkExample,
    *,
    document_id: str | None = None,
    chunk_id: str = BENCHMARK_CACHE_PREFIX_CHUNK_ID,
    prefix: str = BENCHMARK_CACHE_ARTIFACT_PREFIX,
    segment_per_document: bool = False,
    scorer: DatasetScorer | None = None,
) -> SourceDocument:
    """Represent the exact V1 benchmark cache prefix as a Cachet source document.

    With ``segment_per_document`` the prefix is split into one KV chunk per document
    (tiling the exact prefix text), so the handoff assembles N independently-prefilled
    document segments instead of one monolithic prefix chunk.
    """

    benchmark_example = _benchmark_example(example)
    resolved_document_id = document_id or benchmark_cache_document_id(benchmark_example, prefix=prefix)
    document_metadata = {
        "cachet.benchmark.dataset": benchmark_example.dataset,
        "cachet.benchmark.example_id": benchmark_example.example_id,
        "cachet.benchmark.role": "cache_prefix",
    }
    if segment_per_document:
        segments = benchmark_cache_prefix_segments(
            benchmark_example,
            scorer=scorer,
        )
        if len(segments) > 1:
            return SourceDocument.from_texts(
                document_id=resolved_document_id,
                chunks={segment_chunk_id: text for segment_chunk_id, text in segments},
                metadata=document_metadata,
                chunk_metadata={
                    segment_chunk_id: {"cachet.benchmark.prompt_part": f"document_segment_{index}"}
                    for index, (segment_chunk_id, _text) in enumerate(segments)
                },
            )
    return SourceDocument.from_text(
        document_id=resolved_document_id,
        text=build_cache_prefix_text(benchmark_example, scorer=scorer),
        chunk_id=chunk_id,
        metadata=document_metadata,
        chunk_metadata={
            "cachet.benchmark.prompt_part": "system_prompt_and_document_context",
        },
    )


def benchmark_cache_request(
    example: BenchmarkExample,
    *,
    model_id: str = DEFAULT_V1_MODEL_ID,
    lora_id: str = DEFAULT_V1_LORA_ID,
    prompt_template_version: str = DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    request_id: str | None = None,
    task_id: str | None = None,
    document_id: str | None = None,
    chunk_id: str = BENCHMARK_CACHE_PREFIX_CHUNK_ID,
    prefix: str = BENCHMARK_CACHE_ARTIFACT_PREFIX,
    segment_per_document: bool = False,
    scorer: DatasetScorer | None = None,
) -> DocumentKVRequest:
    """Build the Cachet request that materializes this example's cached prefix.

    With ``segment_per_document`` the request selects the N per-document prefix chunks
    (in order) so ``CachePlanner`` emits an N-segment contiguous plan. Chunk ids must
    match :func:`benchmark_cache_source_document`.
    """

    benchmark_example = _benchmark_example(example)
    resolved_document_id = document_id or benchmark_cache_document_id(benchmark_example, prefix=prefix)
    resolved_request_id = request_id or benchmark_cache_artifact_stem(benchmark_example, prefix=prefix)
    if segment_per_document:
        segments = benchmark_cache_prefix_segments(
            benchmark_example,
            scorer=scorer,
        )
        if len(segments) > 1:
            return DocumentKVRequest.for_document_chunks(
                request_id=resolved_request_id,
                task_id=task_id or f"v1-benchmark-{benchmark_example.dataset}",
                model_id=model_id,
                lora_id=lora_id,
                prompt_template_version=prompt_template_version,
                document_id=resolved_document_id,
                chunk_ids=tuple(segment_chunk_id for segment_chunk_id, _text in segments),
                include_static=False,
            )
    return DocumentKVRequest.for_text_document(
        request_id=resolved_request_id,
        task_id=task_id or f"v1-benchmark-{benchmark_example.dataset}",
        model_id=model_id,
        lora_id=lora_id,
        prompt_template_version=prompt_template_version,
        document_id=resolved_document_id,
        chunk_id=chunk_id,
    )


def format_document_context(documents: Sequence[SourceDocument]) -> str:
    if not documents:
        raise ValueError("Benchmark examples must include at least one source document")
    return _join_sections("Documents:", *(_format_document(document) for document in documents))


def summarize_measurements(measurements: Iterable[InferenceMeasurement]) -> tuple[BenchmarkReportRow, ...]:
    grouped: dict[tuple[str, str], list[InferenceMeasurement]] = {}
    for measurement in measurements:
        grouped.setdefault((measurement.dataset, measurement.arm_id), []).append(measurement)
    rows = [_summarize_group(dataset, arm_id, group) for (dataset, arm_id), group in grouped.items()]
    return tuple(sorted(rows, key=lambda row: (row.dataset, row.arm_id)))


def compare_to_baseline(
    rows: Sequence[BenchmarkReportRow],
    *,
    baseline_arm_id: str = BASELINE_PREFILL_ARM,
    cache_arm_id: str = CACHE_REUSE_ARM,
) -> tuple[BenchmarkComparison, ...]:
    by_key = {(row.dataset, row.arm_id): row for row in rows}
    comparisons: list[BenchmarkComparison] = []
    datasets = sorted({row.dataset for row in rows})
    for dataset in datasets:
        baseline = by_key.get((dataset, baseline_arm_id))
        cache = by_key.get((dataset, cache_arm_id))
        if baseline is None or cache is None:
            continue
        comparisons.append(
            BenchmarkComparison(
                dataset=dataset,
                baseline_arm_id=baseline_arm_id,
                cache_arm_id=cache_arm_id,
                ttft_speedup=latency_speedup(baseline.ttft.p50, cache.ttft.p50),
                time_to_completion_speedup=latency_speedup(
                    baseline.time_to_completion.p50,
                    cache.time_to_completion.p50,
                ),
                exact_match_delta=quality_delta(cache.exact_match_rate, baseline.exact_match_rate),
                answer_found_delta=quality_delta(cache.answer_found_rate, baseline.answer_found_rate),
                cache_method=cache.cache_method,
                artifact_id=cache.artifact_id,
                variant_id=cache.variant_id,
                quality_score_deltas={
                    metric_name: cache.quality_score_means[metric_name]
                    - baseline.quality_score_means[metric_name]
                    for metric_name in sorted(
                        set(baseline.quality_score_means).intersection(
                            cache.quality_score_means
                        )
                    )
                },
            )
        )
    return tuple(comparisons)


def evaluate_v1_benchmark_evidence(
    rows: Sequence[BenchmarkReportRow],
    comparisons: Sequence[BenchmarkComparison],
    *,
    required_datasets: Sequence[str] = SUPPORTED_V1_DATASETS,
    baseline_arm_id: str = BASELINE_PREFILL_ARM,
    cache_arm_id: str = CACHE_REUSE_ARM,
) -> V1BenchmarkEvidence:
    required = tuple(required_datasets)
    for dataset in required:
        validate_v1_dataset(dataset)
    duplicate_required_datasets = _duplicate_labels(required)
    unique_required = _dedupe_preserve_order(required)
    required_row_keys = tuple(
        (dataset, arm_id)
        for dataset in unique_required
        for arm_id in (baseline_arm_id, cache_arm_id)
    )
    rows_by_key, duplicate_report_rows = _report_rows_by_key(rows)
    comparisons_by_dataset, duplicate_comparisons = _comparisons_by_dataset(
        comparisons,
        baseline_arm_id=baseline_arm_id,
        cache_arm_id=cache_arm_id,
    )
    required_datasets_set = set(unique_required)
    expected_arms = {baseline_arm_id, cache_arm_id}
    observed_datasets = {row.dataset for row in rows}.union(comparison.dataset for comparison in comparisons)
    observed_arms = {row.arm_id for row in rows}.union(
        arm_id
        for comparison in comparisons
        for arm_id in (comparison.baseline_arm_id, comparison.cache_arm_id)
    )
    existing_required_rows = tuple(rows_by_key[key] for key in required_row_keys if key in rows_by_key)
    return V1BenchmarkEvidence(
        required_datasets=required,
        baseline_arm_id=baseline_arm_id,
        cache_arm_id=cache_arm_id,
        duplicate_required_datasets=duplicate_required_datasets,
        duplicate_report_rows=duplicate_report_rows,
        duplicate_comparisons=duplicate_comparisons,
        missing_report_rows=tuple(
            _row_key(dataset, arm_id)
            for dataset, arm_id in required_row_keys
            if (dataset, arm_id) not in rows_by_key
        ),
        missing_comparisons=tuple(dataset for dataset in unique_required if dataset not in comparisons_by_dataset),
        comparisons_without_metrics=tuple(
            dataset
            for dataset in unique_required
            if (comparison := comparisons_by_dataset.get(dataset)) is not None
            and _comparison_has_missing_metrics(comparison)
        ),
        rows_without_successful_requests=tuple(
            _row_key(row.dataset, row.arm_id)
            for row in existing_required_rows
            if row.ttft.count == 0
        ),
        rows_without_latency=tuple(
            _row_key(row.dataset, row.arm_id)
            for row in existing_required_rows
            if row.ttft.p50 is None or row.time_to_completion.p50 is None
        ),
        rows_without_quality=tuple(
            _row_key(row.dataset, row.arm_id)
            for row in existing_required_rows
            if row.exact_match_rate is None or row.answer_found_rate is None
        ),
        unexpected_arms=tuple(sorted(observed_arms.difference(expected_arms))),
        unexpected_datasets=tuple(sorted(observed_datasets.difference(required_datasets_set))),
    )


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def exact_match(output_text: str, expected_answer: str) -> bool:
    return normalize_answer(output_text) == normalize_answer(expected_answer)


def answer_found(output_text: str, expected_answer: str) -> bool:
    expected_tokens = normalize_answer(expected_answer).split()
    output_tokens = normalize_answer(output_text).split()
    if not expected_tokens:
        return False
    window = len(expected_tokens)
    for index in range(0, len(output_tokens) - window + 1):
        if output_tokens[index : index + window] == expected_tokens:
            return True
    return False


def validate_v1_dataset(dataset: str) -> None:
    if dataset not in SUPPORTED_V1_DATASETS:
        raise ValueError(f"Unsupported V1 dataset {dataset!r}; expected one of {SUPPORTED_V1_DATASETS}")


def validate_v1_hardware_target(hardware_target: str) -> None:
    _validate_v1_hardware_target(hardware_target)


def _benchmark_example(example: BenchmarkExample) -> BenchmarkExample:
    if not isinstance(example, BenchmarkExample):
        raise TypeError("example must be a BenchmarkExample")
    return example


def _artifact_slug(value: str, *, field_name: str) -> str:
    _validate_non_empty_str(value, field_name)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._").lower()
    if not slug:
        raise ValueError(f"{field_name} must contain at least one path-safe character")
    return slug


def _validate_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")


def _validate_str(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")


_SequenceItem = TypeVar("_SequenceItem")


def _tuple_from_sequence(
    value: Sequence[_SequenceItem],
    field_name: str,
) -> tuple[_SequenceItem, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return tuple(value)


def _dict_from_str_mapping(value: Mapping[str, str], field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if not isinstance(item, str):
            raise ValueError(f"{field_name}.{key} must be a string")
        normalized[key] = item
    return normalized


def _dict_from_json_object_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        normalized[key] = _json_compatible_value(item, f"{field_name}.{key}")
    return normalized


def _json_compatible_value(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be JSON-compatible")
        return value
    if isinstance(value, Mapping):
        return _dict_from_json_object_mapping(value, field_name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return [_json_compatible_value(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{field_name} must be JSON-compatible")


def _deep_freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _deep_freeze_value(item) for key, item in value.items()}
    )


def _deep_freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _deep_freeze_mapping(value)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return _FrozenList(_deep_freeze_value(item) for item in value)
    return value


def _validate_kv_transfer_params(kv_transfer_params: Mapping[str, Any]) -> None:
    if not kv_transfer_params:
        return
    request_id = kv_transfer_params.get(DOCUMENT_KV_REQUEST_ID_PARAM)
    if request_id is None:
        raise ValueError(
            f"kv_transfer_params.{DOCUMENT_KV_REQUEST_ID_PARAM} is required when kv_transfer_params are provided"
        )
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(f"kv_transfer_params.{DOCUMENT_KV_REQUEST_ID_PARAM} must be a non-empty string")
    handoff_json = kv_transfer_params.get(DOCUMENT_KV_HANDOFF_JSON_PARAM)
    handoff_record = kv_transfer_params.get(DOCUMENT_KV_HANDOFF_RECORD_PARAM)
    if handoff_json is None and handoff_record is None:
        raise ValueError(
            "kv_transfer_params must include "
            f"{DOCUMENT_KV_HANDOFF_JSON_PARAM} or {DOCUMENT_KV_HANDOFF_RECORD_PARAM}"
        )
    if handoff_json is not None and handoff_record is not None:
        raise ValueError(
            "kv_transfer_params must include only one of "
            f"{DOCUMENT_KV_HANDOFF_JSON_PARAM} or {DOCUMENT_KV_HANDOFF_RECORD_PARAM}"
        )
    if handoff_json is not None and (not isinstance(handoff_json, str) or not handoff_json):
        raise ValueError(f"kv_transfer_params.{DOCUMENT_KV_HANDOFF_JSON_PARAM} must be a non-empty string")
    payload_uri = kv_transfer_params.get(DOCUMENT_KV_PAYLOAD_URI_PARAM)
    if payload_uri is not None:
        _validate_runtime_payload_uri(
            payload_uri,
            field_name=f"kv_transfer_params.{DOCUMENT_KV_PAYLOAD_URI_PARAM}",
        )
    _validate_optional_string_sequence(
        kv_transfer_params.get(DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM),
        field_name=f"kv_transfer_params.{DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM}",
    )
    runtime_prefix_text = kv_transfer_params.get(DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM)
    if runtime_prefix_text is not None and not isinstance(runtime_prefix_text, str):
        raise ValueError(f"kv_transfer_params.{DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM} must be a string")
    for parameter in (DOCUMENT_KV_CACHE_METHOD_PARAM, DOCUMENT_KV_ARTIFACT_ID_PARAM):
        value = kv_transfer_params.get(parameter)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"kv_transfer_params.{parameter} must be a non-empty string")
    if handoff_record is not None:
        if not isinstance(handoff_record, Mapping):
            raise ValueError(f"kv_transfer_params.{DOCUMENT_KV_HANDOFF_RECORD_PARAM} must be an object")
        _validate_inline_handoff_record(
            handoff_record,
            request_id=request_id,
            payload_uri_override=payload_uri,
        )
        handle = handoff_record.get("handle")
        if isinstance(handle, Mapping):
            cache_method = kv_transfer_params.get(DOCUMENT_KV_CACHE_METHOD_PARAM)
            if cache_method is not None and cache_method != handle.get("cache_method"):
                raise ValueError(
                    f"kv_transfer_params.{DOCUMENT_KV_CACHE_METHOD_PARAM} must match handoff handle"
                )
            artifact_id = kv_transfer_params.get(DOCUMENT_KV_ARTIFACT_ID_PARAM)
            artifact_identity = handle.get("artifact_identity")
            if artifact_id is not None and isinstance(artifact_identity, Mapping):
                from document_kv_cache.artifact_identity import ArtifactIdentity

                if artifact_id != ArtifactIdentity.from_record(artifact_identity).artifact_id:
                    raise ValueError(
                        f"kv_transfer_params.{DOCUMENT_KV_ARTIFACT_ID_PARAM} must match handoff handle"
                    )


def _validate_runtime_payload_uri(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    from document_kv_cache.engine_probe import _validate_local_payload_uri

    try:
        _validate_local_payload_uri(value)
    except ValueError as exc:
        raise ValueError(f"{field_name}: {exc}") from exc


def _validate_optional_string_sequence(value: object, *, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of strings")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field_name}[{index}] must be a non-empty string")


def _validate_inline_handoff_record(
    handoff_record: Mapping[str, Any],
    *,
    request_id: str,
    payload_uri_override: object,
) -> None:
    from document_kv_cache.engine_adapters import validate_engine_adapter_request_record

    validate_engine_adapter_request_record(
        handoff_record,
        require_external_payload_uri=payload_uri_override is None,
    )
    handoff_request_id = handoff_record.get("request_id")
    if handoff_request_id != request_id:
        raise ValueError(
            f"kv_transfer_params.{DOCUMENT_KV_HANDOFF_RECORD_PARAM}.request_id must match "
            f"kv_transfer_params.{DOCUMENT_KV_REQUEST_ID_PARAM}"
        )
    if payload_uri_override is None:
        _validate_inline_handoff_payload_uri(handoff_record)


def _validate_inline_handoff_payload_uri(handoff_record: Mapping[str, Any]) -> None:
    payload_source = handoff_record.get("payload_source")
    if not isinstance(payload_source, Mapping):
        raise ValueError(f"kv_transfer_params.{DOCUMENT_KV_HANDOFF_RECORD_PARAM}.payload_source must be an object")
    _validate_runtime_payload_uri(
        payload_source.get("uri"),
        field_name=f"kv_transfer_params.{DOCUMENT_KV_HANDOFF_RECORD_PARAM}.payload_source.uri",
    )


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_non_negative_finite_number(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number")


_V1_DATASET_SPECS: Mapping[str, BenchmarkDatasetSpec] = {
    "biography": BenchmarkDatasetSpec(
        dataset="biography",
        display_name="Biography Entity Identification",
        task_instruction=(
            "Identify the entity described by the supplied biography context. "
            "Document identifiers and titles are intentionally opaque."
        ),
        answer_instruction="Return only the normalized entity title as the answer value.",
    ),
    "hotpotqa": BenchmarkDatasetSpec(
        dataset="hotpotqa",
        display_name="HotpotQA",
        task_instruction="Answer multi-hop questions by combining the relevant facts in the supplied context.",
        answer_instruction="Return the final answer, not a chain-of-thought explanation.",
    ),
    "musique": BenchmarkDatasetSpec(
        dataset="musique",
        display_name="MusiQue",
        task_instruction="Answer compositional questions by using all necessary supporting documents in the context.",
        answer_instruction="Return the final answer, not a chain-of-thought explanation.",
    ),
    "niah": BenchmarkDatasetSpec(
        dataset="niah",
        display_name="Needle-in-a-Haystack",
        task_instruction="Find the hidden target statement in the supplied context.",
        answer_instruction="Return the exact needle or requested value from the context.",
    ),
}


def _system_prompt(spec: BenchmarkDatasetSpec) -> str:
    return _join_sections(
        f"Benchmark: {spec.display_name}",
        spec.task_instruction,
        "Use only the supplied document context. If the answer is absent, say you do not know.",
        (
            "Your entire response must contain exactly one non-empty block of the "
            "form <final_answer>answer</final_answer>, with no text outside it."
        ),
    )


def _user_prompt(example: BenchmarkExample, spec: BenchmarkDatasetSpec) -> str:
    return _join_sections(
        f"Question: {example.query}",
        spec.answer_instruction,
        f"Required response form: {FINAL_ANSWER_CUE}",
    )


def _format_document(document: SourceDocument) -> str:
    title = document.metadata.get("title") or document.metadata.get("name") or document.document_id
    chunks = tuple(
        _format_chunk(
            chunk.chunk_id,
            (
                chunk.chunk_type.value
                if hasattr(chunk.chunk_type, "value")
                else chunk.chunk_type
            ),
            chunk.text,
        )
        for chunk in document.chunks
    )
    return _join_sections(
        f'[document id="{_attribute_text(document.document_id)}" title="{_attribute_text(title)}"]',
        *chunks,
        f'[/document id="{_attribute_text(document.document_id)}"]',
    )


def _format_chunk(chunk_id: str, chunk_type: str, text: str) -> str:
    return _join_sections(
        f'[chunk id="{_attribute_text(chunk_id)}" type="{_attribute_text(chunk_type)}"]',
        _quote_block_text(text),
        f'[/chunk id="{_attribute_text(chunk_id)}"]',
    )


def _summarize_group(dataset: str, arm_id: str, group: Sequence[InferenceMeasurement]) -> BenchmarkReportRow:
    if not group:
        raise ValueError("Cannot summarize an empty measurement group")
    ok = [measurement for measurement in group if measurement.ok]
    errors = len(group) - len(ok)
    prompt_tokens = [measurement.prompt_tokens for measurement in ok]
    completion_tokens = [measurement.completion_tokens for measurement in ok]
    ttft_values = [measurement.ttft_seconds for measurement in ok]
    ttc_values = [measurement.time_to_completion_seconds for measurement in ok]
    request_decode_rates = [
        rate
        for measurement in ok
        if (
            rate := request_decode_tokens_per_second(
                measurement.completion_tokens,
                measurement.ttft_seconds,
                measurement.time_to_completion_seconds,
            )
        )
        is not None
    ]
    cache_methods = {measurement.cache_method for measurement in group}
    artifact_ids = {measurement.artifact_id for measurement in group}
    variant_ids = {measurement.variant_id for measurement in group}
    if len(cache_methods) != 1:
        raise ValueError(f"Benchmark arm {arm_id!r} mixes cache methods: {sorted(cache_methods)}")
    if len(artifact_ids) != 1:
        raise ValueError(f"Benchmark arm {arm_id!r} mixes artifact identities: {sorted(artifact_ids)}")
    if len(variant_ids) != 1:
        raise ValueError(f"Benchmark arm {arm_id!r} mixes variant identities: {sorted(variant_ids)}")
    return BenchmarkReportRow(
        dataset=dataset,
        arm_id=arm_id,
        requests=len(group),
        errors=errors,
        prompt_tokens_mean=_mean(prompt_tokens),
        completion_tokens_mean=_mean(completion_tokens),
        ttft=_latency_summary(ttft_values),
        time_to_completion=_latency_summary(ttc_values),
        exact_match_rate=_unique_example_rate(ok, "exact_match"),
        answer_found_rate=_unique_example_rate(ok, "answer_found"),
        output_tokens_per_second=aggregate_decode_tokens_per_second(
            (
                (
                    measurement.completion_tokens,
                    measurement.ttft_seconds,
                    measurement.time_to_completion_seconds,
                )
                for measurement in ok
            )
        ),
        cache_method=next(iter(cache_methods)),
        artifact_id=next(iter(artifact_ids)),
        variant_id=next(iter(variant_ids)),
        unique_examples=len({measurement.example_id for measurement in ok}),
        quality_score_means=_unique_example_quality_means(ok),
        request_decode_tokens_per_second=_latency_summary(request_decode_rates),
    )


def _latency_summary(values: Sequence[float]) -> LatencySummary:
    if not values:
        return LatencySummary(count=0, mean=None, p50=None, p95=None)
    sorted_values = sorted(values)
    return LatencySummary(
        count=len(sorted_values),
        mean=statistics.fmean(sorted_values),
        p50=_percentile(sorted_values, 0.50),
        p95=_percentile(sorted_values, 0.95),
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = percentile * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _mean(values: Sequence[int]) -> float | None:
    return statistics.fmean(values) if values else None


def _rate(values: Iterable[bool | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(1 for value in present if value) / len(present)


def _unique_example_rate(
    measurements: Sequence[InferenceMeasurement],
    property_name: Literal["exact_match", "answer_found"],
) -> float | None:
    by_example: dict[str, list[bool]] = {}
    for measurement in measurements:
        value = getattr(measurement, property_name)
        if value is not None:
            by_example.setdefault(measurement.example_id, []).append(value)
    if not by_example:
        return None
    # Repeats estimate one example's success probability; examples remain the
    # independent quality units and therefore receive equal weight.
    return statistics.fmean(
        statistics.fmean(float(value) for value in values)
        for values in by_example.values()
    )


def _unique_example_quality_means(
    measurements: Sequence[InferenceMeasurement],
) -> Mapping[str, float]:
    metric_examples: dict[str, dict[str, list[float]]] = {}
    for measurement in measurements:
        for metric_name, value in measurement.quality_scores.items():
            metric_examples.setdefault(metric_name, {}).setdefault(
                measurement.example_id, []
            ).append(value)
    return {
        metric_name: statistics.fmean(
            statistics.fmean(values) for values in example_values.values()
        )
        for metric_name, example_values in sorted(metric_examples.items())
    }


def _comparison_has_missing_metrics(comparison: BenchmarkComparison) -> bool:
    return (
        comparison.ttft_speedup is None
        or comparison.time_to_completion_speedup is None
        or comparison.exact_match_delta is None
        or comparison.answer_found_delta is None
    )


def _row_key(dataset: str, arm_id: str) -> str:
    return f"{dataset}:{arm_id}"


def _example_key(example: BenchmarkExample) -> str:
    return f"{example.dataset}:{example.example_id}"


def _comparison_key(comparison: BenchmarkComparison) -> str:
    return f"{comparison.dataset}:{comparison.baseline_arm_id}->{comparison.cache_arm_id}"


def _duplicate_labels(labels: Iterable[str]) -> tuple[str, ...]:
    seen = set()
    duplicate_seen = set()
    duplicates = []
    for label in labels:
        if label in seen and label not in duplicate_seen:
            duplicate_seen.add(label)
            duplicates.append(label)
        seen.add(label)
    return tuple(duplicates)


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def _report_rows_by_key(
    rows: Sequence[BenchmarkReportRow],
) -> tuple[dict[tuple[str, str], BenchmarkReportRow], tuple[str, ...]]:
    rows_by_key = {}
    duplicate_labels = []
    duplicate_seen = set()
    for row in rows:
        key = (row.dataset, row.arm_id)
        label = _row_key(row.dataset, row.arm_id)
        if key in rows_by_key:
            if label not in duplicate_seen:
                duplicate_seen.add(label)
                duplicate_labels.append(label)
            continue
        rows_by_key[key] = row
    return rows_by_key, tuple(duplicate_labels)


def _comparisons_by_dataset(
    comparisons: Sequence[BenchmarkComparison],
    *,
    baseline_arm_id: str,
    cache_arm_id: str,
) -> tuple[dict[str, BenchmarkComparison], tuple[str, ...]]:
    comparisons_by_dataset = {}
    duplicate_labels = []
    duplicate_seen = set()
    for comparison in comparisons:
        if comparison.baseline_arm_id != baseline_arm_id or comparison.cache_arm_id != cache_arm_id:
            continue
        label = _comparison_key(comparison)
        if comparison.dataset in comparisons_by_dataset:
            if label not in duplicate_seen:
                duplicate_seen.add(label)
                duplicate_labels.append(label)
            continue
        comparisons_by_dataset[comparison.dataset] = comparison
    return comparisons_by_dataset, tuple(duplicate_labels)


def _join_sections(*sections: str) -> str:
    return "\n\n".join(section for section in sections if section)


def _clean_inline_text(text: str) -> str:
    return " ".join(text.strip().split())


def _attribute_text(text: str) -> str:
    return escape(_clean_inline_text(text), quote=True)


def _quote_block_text(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(f"| {line}" for line in lines)
