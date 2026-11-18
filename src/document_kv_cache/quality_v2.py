"""Quality benchmark protocol v2 for the vLLM 0.30.0 publication.

The governed full-score campaign (protocol ``cachet-vllm-0.27.1-complete-score-v2``)
sent raw completions without the model's chat template, stopped decoding at 64
tokens, and scored only an exact whole-output ``<final_answer>`` block.  Under that
request shape the instruct model echoed instruction text before its answer, so the
strict parser rejected almost every output for both arms and the published quality
numbers do not measure answer quality.

v2 keeps the model, runtime, datasets, answer normalisation and the per-document
Vanilla cache method, and changes only what the v1 audit showed to be broken:

* the same benchmark prompt sections are wrapped in the Qwen3 chat template (a
  single user turn, exactly as ``apply_chat_template`` renders it);
* decoding may run to 256 tokens;
* answers come from a versioned tolerant parser (the first closed
  ``<final_answer>`` block that is not the echoed placeholder), and strict
  whole-output compliance is reported separately as a format metric;
* biography references and predictions are compared without WikiBio's
  ``-lrb- ... -rrb-`` disambiguators;
* NIAH uses natural-text haystacks with a numeric needle at a paragraph boundary.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any

from document_kv_cache._benchmark_datasets import _example_from_record
from document_kv_cache.benchmarks import (
    HOTPOTQA_SCORER_VERSION,
    MUSIQUE_SCORER_VERSION,
    NIAH_CELL_IDS,
    BenchmarkExample,
    BenchmarkPromptParts,
    DatasetMetricSpec,
    DatasetScoreContext,
    DatasetScorer,
    DatasetScorerRegistry,
    FinalAnswerExtraction,
    dataset_spec,
    extract_single_final_answer,
    format_document_context,
    hotpotqa_official_answer_scores,
    musique_official_answer_scores,
    niah_cell_identity,
    normalize_biography_title,
)

__all__ = [
    "QUALITY_V2_PROTOCOL_ID",
    "QUALITY_V2_PROMPT_TEMPLATE_VERSION",
    "QUALITY_V2_PROMPT_PLUGIN_PATH",
    "QUALITY_V2_MAX_TOKENS",
    "QUALITY_V2_PARSER_ID",
    "QUALITY_V2_PARSER_VERSION",
    "QUALITY_V2_PARSER_PLUGIN_PATH",
    "QUALITY_V2_PARSER_CONTRACT",
    "QUALITY_V2_PARSER_DIGEST",
    "QUALITY_V2_NIAH_SEED",
    "QUALITY_V2_HEADLINE_DATASETS",
    "QUALITY_V2_DIAGNOSTIC_DATASETS",
    "QUALITY_V2_MAX_REGRESSION",
    "QUALITY_V2_README_MARKERS",
    "QUALITY_V2_REPORT_RECORD_TYPE",
    "QUALITY_V2_REPORT_SCHEMA_VERSION",
    "QualityV2DatasetSummary",
    "biography_v3_entity_scores",
    "build_quality_v2_niah_records",
    "extract_first_final_answer",
    "niah_v2_exact_value_scores",
    "quality_v2_diagnostics",
    "quality_v2_primary_metric",
    "quality_v2_prompt_parts",
    "quality_v2_protocol_record",
    "quality_v2_report_record",
    "quality_v2_scorer_registry",
    "render_quality_v2_readme_region",
    "strip_biography_disambiguator",
    "summarize_quality_v2_pairs",
    "validate_quality_v2_readme_region",
    "validate_quality_v2_report_record",
]

QUALITY_V2_PROTOCOL_ID = "cachet-vllm-0.30.0-quality-v2"
QUALITY_V2_PROMPT_TEMPLATE_VERSION = "v3-qwen3-chat-final-answer"
QUALITY_V2_PROMPT_PLUGIN_PATH = "document_kv_cache.quality_v2:quality_v2_prompt_parts"
QUALITY_V2_MAX_TOKENS = 256
QUALITY_V2_NIAH_SEED = 20_261_115
QUALITY_V2_HEADLINE_DATASETS = ("hotpotqa", "musique", "niah")
QUALITY_V2_DIAGNOSTIC_DATASETS = ("biography",)
# Non-inferiority margin: on a headline dataset's primary metric the lower end of
# the 95% paired bootstrap interval of (cached - full prefill) must be >= -0.02.
QUALITY_V2_MAX_REGRESSION = 0.02

_CHAT_USER_OPEN = "<|im_start|>user\n"
_CHAT_ASSISTANT_OPEN = "<|im_end|>\n<|im_start|>assistant\n"

QUALITY_V2_PARSER_ID = "cachet.first_final_answer"
QUALITY_V2_PARSER_VERSION = "2"
QUALITY_V2_PARSER_PLUGIN_PATH = "document_kv_cache.quality_v2:extract_first_final_answer"
_PLACEHOLDER_ANSWERS = ("", "...", "answer", "your answer")
QUALITY_V2_PARSER_CONTRACT = (
    "utf8-trim;case-sensitive-lowercase-tags;"
    "first-closed-innermost-final_answer-block;"
    "skip-placeholders=" + "|".join(_PLACEHOLDER_ANSWERS) + ";"
    "text-outside-block-ignored;"
    "statuses=ok,missing_block,multiple_or_malformed_blocks,empty_answer"
)
QUALITY_V2_PARSER_DIGEST = sha256(QUALITY_V2_PARSER_CONTRACT.encode("utf-8")).hexdigest()
_INNERMOST_BLOCK = re.compile(
    r"<final_answer>((?:(?!<final_answer>).)*?)</final_answer>", re.S
)
_PARSER_SUFFIX = f"first_final_answer_v2@{QUALITY_V2_PARSER_DIGEST[:12]}"
_BIOGRAPHY_DISAMBIGUATOR = re.compile(r"-lrb-.*?-rrb-|\([^()]*\)")


def extract_first_final_answer(output_text: str) -> FinalAnswerExtraction:
    """Return the first closed ``<final_answer>`` block that is a real answer.

    Text outside the block is ignored, nested opening tags resolve to the innermost
    block, and the echoed ``<final_answer>answer</final_answer>`` cue (or another
    placeholder) is skipped.  An unclosed trailing block is not an answer.
    """

    if not isinstance(output_text, str):
        raise TypeError("output_text must be a string")
    identity = {
        "parser_id": QUALITY_V2_PARSER_ID,
        "parser_version": QUALITY_V2_PARSER_VERSION,
        "parser_plugin_path": QUALITY_V2_PARSER_PLUGIN_PATH,
        "parser_digest": QUALITY_V2_PARSER_DIGEST,
    }
    if "<final_answer>" not in output_text:
        return FinalAnswerExtraction(output_text, "", False, "missing_block", **identity)
    saw_closed_block = False
    for match in _INNERMOST_BLOCK.finditer(output_text):
        saw_closed_block = True
        answer = match.group(1).strip()
        if answer.casefold() not in _PLACEHOLDER_ANSWERS:
            return FinalAnswerExtraction(output_text, answer, True, "ok", **identity)
    status = "empty_answer" if saw_closed_block else "multiple_or_malformed_blocks"
    return FinalAnswerExtraction(output_text, "", False, status, **identity)


def quality_v2_prompt_parts(example: BenchmarkExample) -> BenchmarkPromptParts:
    """Wrap the v1 benchmark sections in a single Qwen3 chat user turn.

    The chat markers ride on the leading cached section and on the online suffix,
    so per-document segmentation of the document context is unchanged.
    """

    if not isinstance(example, BenchmarkExample):
        raise TypeError("example must be a BenchmarkExample")
    spec = dataset_spec(example.dataset)
    instructions = "\n\n".join(
        (
            f"Benchmark: {spec.display_name}",
            spec.task_instruction,
            "Use only the supplied document context. If the answer is absent, "
            "say you do not know.",
            "Your entire response must contain exactly one non-empty block of the "
            "form <final_answer>answer</final_answer>, with no text outside it.",
        )
    )
    question = "\n\n".join(
        (
            f"Question: {example.query}",
            spec.answer_instruction,
            "Required response form: <final_answer>answer</final_answer>",
        )
    )
    return BenchmarkPromptParts(
        system_prompt=_CHAT_USER_OPEN + instructions,
        document_context=format_document_context(example.documents),
        user_prompt=question + _CHAT_ASSISTANT_OPEN,
        system_prompt_position="start",
    )


def strip_biography_disambiguator(value: str) -> str:
    """Drop WikiBio ``-lrb- ... -rrb-`` (and parenthesised) disambiguators."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return " ".join(_BIOGRAPHY_DISAMBIGUATOR.sub(" ", value).split())


def biography_v3_entity_scores(context: DatasetScoreContext) -> Mapping[str, float]:
    """Biography exact match on titles without disambiguators."""

    if not isinstance(context, DatasetScoreContext):
        raise TypeError("context must be a DatasetScoreContext")
    if not context.references:
        return MappingProxyType({"exact_match": 0.0})
    prediction = normalize_biography_title(
        strip_biography_disambiguator(context.output_text)
    )
    exact = any(
        prediction == normalize_biography_title(strip_biography_disambiguator(reference))
        for reference in context.references
    )
    return MappingProxyType({"exact_match": float(exact)})


def niah_v2_exact_value_scores(context: DatasetScoreContext) -> Mapping[str, float]:
    """Exact needle value after trimming whitespace and terminal punctuation."""

    if not isinstance(context, DatasetScoreContext):
        raise TypeError("context must be a DatasetScoreContext")
    cell_id = context.metadata.get("niah_cell_id")
    if cell_id is not None and cell_id not in NIAH_CELL_IDS:
        raise ValueError("niah_cell_id metadata must identify a NIAH grid cell")
    if not context.references:
        return MappingProxyType({"accuracy": 0.0})
    prediction = context.output_text.strip().strip(".,;:!?\"'` ")
    expected = context.references[0].strip()
    return MappingProxyType({"accuracy": float(prediction == expected)})


def quality_v2_scorer_registry() -> DatasetScorerRegistry:
    shared: dict[str, Any] = {
        "publication_approved": True,
        "prompt_function": quality_v2_prompt_parts,
        "prompt_plugin_path": QUALITY_V2_PROMPT_PLUGIN_PATH,
        "prompt_template_version": QUALITY_V2_PROMPT_TEMPLATE_VERSION,
        "answer_parser_function": extract_first_final_answer,
        "answer_parser_id": QUALITY_V2_PARSER_ID,
        "answer_parser_version": QUALITY_V2_PARSER_VERSION,
        "answer_parser_plugin_path": QUALITY_V2_PARSER_PLUGIN_PATH,
        "answer_parser_digest": QUALITY_V2_PARSER_DIGEST,
    }
    biography = DatasetScorer(
        scorer_id="cachet.biography_entity_identification",
        version=(
            "biography_entity_identification_v3+strip_lrb_rrb_disambiguator+"
            "nfkc_casefold_ws_terminal_punctuation_v2+" + _PARSER_SUFFIX
        ),
        metric_names=("exact_match",),
        metric_specs=(DatasetMetricSpec("exact_match", max_regression=1.0),),
        score_function=biography_v3_entity_scores,
        plugin_path="document_kv_cache.quality_v2:biography_v3_entity_scores",
        **shared,
    )
    hotpotqa = DatasetScorer(
        scorer_id="hotpotqa.official_answer",
        version=HOTPOTQA_SCORER_VERSION.split("+final_answer_v1@")[0]
        + "+"
        + _PARSER_SUFFIX,
        metric_names=("exact_match", "f1"),
        metric_specs=(
            DatasetMetricSpec("exact_match", max_regression=QUALITY_V2_MAX_REGRESSION),
            DatasetMetricSpec("f1", max_regression=QUALITY_V2_MAX_REGRESSION),
        ),
        score_function=hotpotqa_official_answer_scores,
        plugin_path="document_kv_cache.benchmarks:hotpotqa_official_answer_scores",
        **shared,
    )
    musique = DatasetScorer(
        scorer_id="musique.official_answer",
        version=MUSIQUE_SCORER_VERSION.split("+final_answer_v1@")[0]
        + "+"
        + _PARSER_SUFFIX,
        metric_names=("answer_em", "answer_f1"),
        metric_specs=(
            DatasetMetricSpec("answer_em", max_regression=QUALITY_V2_MAX_REGRESSION),
            DatasetMetricSpec("answer_f1", max_regression=QUALITY_V2_MAX_REGRESSION),
        ),
        score_function=musique_official_answer_scores,
        plugin_path="document_kv_cache.benchmarks:musique_official_answer_scores",
        **shared,
    )
    niah = DatasetScorer(
        scorer_id="cachet.niah_exact_value",
        version="cachet_niah_grid_v2+numeric_needle+" + _PARSER_SUFFIX,
        metric_names=("accuracy",),
        metric_specs=(
            DatasetMetricSpec("accuracy", max_regression=QUALITY_V2_MAX_REGRESSION),
        ),
        score_function=niah_v2_exact_value_scores,
        plugin_path="document_kv_cache.quality_v2:niah_v2_exact_value_scores",
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


def quality_v2_protocol_record() -> dict[str, Any]:
    registry = quality_v2_scorer_registry()
    return {
        "diagnostic_datasets": list(QUALITY_V2_DIAGNOSTIC_DATASETS),
        "headline_datasets": list(QUALITY_V2_HEADLINE_DATASETS),
        "max_regression": QUALITY_V2_MAX_REGRESSION,
        "max_tokens": QUALITY_V2_MAX_TOKENS,
        "niah_seed": QUALITY_V2_NIAH_SEED,
        "parser": {
            "contract": QUALITY_V2_PARSER_CONTRACT,
            "digest": QUALITY_V2_PARSER_DIGEST,
            "id": QUALITY_V2_PARSER_ID,
            "plugin_path": QUALITY_V2_PARSER_PLUGIN_PATH,
            "version": QUALITY_V2_PARSER_VERSION,
        },
        "prompt_plugin_path": QUALITY_V2_PROMPT_PLUGIN_PATH,
        "prompt_template_version": QUALITY_V2_PROMPT_TEMPLATE_VERSION,
        "protocol_id": QUALITY_V2_PROTOCOL_ID,
        "scorers": {
            dataset: {"identity": scorer.identity, "plugin_path": scorer.plugin_path}
            for dataset, scorer in registry.entries
        },
        "temperature": 0.0,
    }


_NIAH_TARGETS = (8192, 16384, 32768)
_NIAH_DEPTHS = (0.1, 0.5, 0.9)
_NIAH_QUERY = "What is the special magic number mentioned in the document?"
_NIAH_TARGET_SLACK_TOKENS = 384


def build_quality_v2_niah_records(
    paragraphs: Sequence[str],
    *,
    count_tokens: Callable[[str], int],
    item_count: int = 1000,
    seed: int = QUALITY_V2_NIAH_SEED,
) -> list[dict[str, Any]]:
    """Build the v2 NIAH grid from natural paragraphs.

    Items cycle through the nine (context length, depth) cells.  Each haystack is
    a seeded sample of distinct paragraphs; the needle is its own paragraph at the
    requested depth.  ``count_tokens`` must count tokens of the full chat-framed
    prompt text; each prompt lands in ``[target - 384, target]`` tokens.
    """

    pool = tuple(
        dict.fromkeys(
            " ".join(paragraph.split())
            for paragraph in paragraphs
            if paragraph and "magic number" not in paragraph.casefold()
        )
    )
    if len(pool) < 1000:
        raise ValueError("NIAH v2 needs at least 1000 distinct paragraphs")
    if item_count <= 0:
        raise ValueError("item_count must be positive")
    cells = tuple((target, depth) for depth in _NIAH_DEPTHS for target in _NIAH_TARGETS)
    paragraph_tokens: dict[str, int] = {}

    def tokens_of(paragraph: str) -> int:
        cached = paragraph_tokens.get(paragraph)
        if cached is None:
            cached = paragraph_tokens[paragraph] = count_tokens(paragraph) + 1
        return cached

    records = []
    for index in range(item_count):
        target, depth = cells[index % len(cells)]
        rng = random.Random(f"{seed}:{index}")
        value = str(rng.randrange(1_000_000, 10_000_000))
        needle = f"The special magic number is {value}."
        order = list(range(len(pool)))
        rng.shuffle(order)
        candidates = iter(pool[position] for position in order if value not in pool[position])
        chosen: list[str] = []
        estimate = 0
        budget = target - 200
        for paragraph in candidates:
            chosen.append(paragraph)
            estimate += tokens_of(paragraph)
            if estimate >= budget:
                break

        def render(selected: Sequence[str]) -> tuple[dict[str, Any], str]:
            insert_at = round(depth * len(selected))
            body = [*selected[:insert_at], needle, *selected[insert_at:]]
            cell_id = niah_cell_identity(target, depth)
            record = {
                "dataset": "niah",
                "documents": [{"document_id": "haystack", "text": "\n\n".join(body)}],
                "example_id": f"niah-v2-{seed}-{index:05d}",
                "expected_answer": value,
                "metadata": {
                    "benchmark_task": "niah_exact_value",
                    "benchmark_task_version": "2",
                    "context_token_target": str(target),
                    "haystack_source": "hotpotqa+musique context paragraphs",
                    "needle_paragraph_index": str(insert_at),
                    "needle_position": f"{depth:.3f}",
                    "needle_text": needle,
                    "niah_cell_id": cell_id,
                    "seed": str(seed),
                    "source": "synthetic_niah_grid_v2",
                },
                "query": _NIAH_QUERY,
                "references": [value],
            }
            example = _example_from_record(
                record, default_dataset="niah", record_index=index + 1, require_dataset=True
            )
            return record, quality_v2_prompt_parts(example).prefill_prompt

        record, prompt = render(chosen)
        total = count_tokens(prompt)
        while total > target and len(chosen) > 1:
            excess = total - target
            while excess > 0 and len(chosen) > 1:
                excess -= tokens_of(chosen.pop())
            record, prompt = render(chosen)
            total = count_tokens(prompt)
        for paragraph in candidates:
            if total >= target - _NIAH_TARGET_SLACK_TOKENS // 2:
                break
            if tokens_of(paragraph) > target - total:
                continue
            chosen.append(paragraph)
            record, prompt = render(chosen)
            total = count_tokens(prompt)
            if total > target:
                chosen.pop()
                record, prompt = render(chosen)
                total = count_tokens(prompt)
        if not target - _NIAH_TARGET_SLACK_TOKENS <= total <= target:
            raise ValueError(
                f"NIAH v2 item {index} has {total} prompt tokens for target {target}"
            )
        record["metadata"]["prompt_tokens"] = str(total)
        records.append(record)
    return records


def quality_v2_primary_metric(dataset: str) -> str:
    return {
        "biography": "exact_match",
        "hotpotqa": "exact_match",
        "musique": "answer_em",
        "niah": "accuracy",
    }[dataset]


@dataclass(frozen=True, slots=True)
class QualityV2DatasetSummary:
    dataset: str
    pairs: int
    metrics: Mapping[str, Mapping[str, float]]
    strict_format_rate: Mapping[str, float]
    tolerant_parse_rate: Mapping[str, float]
    token_cap_rate: Mapping[str, float]
    mean_completion_tokens: Mapping[str, float]
    errors: Mapping[str, int]

    def to_record(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                {
                    "dataset": self.dataset,
                    "errors": dict(self.errors),
                    "mean_completion_tokens": dict(self.mean_completion_tokens),
                    "metrics": {key: dict(value) for key, value in self.metrics.items()},
                    "pairs": self.pairs,
                    "strict_format_rate": dict(self.strict_format_rate),
                    "token_cap_rate": dict(self.token_cap_rate),
                    "tolerant_parse_rate": dict(self.tolerant_parse_rate),
                },
                sort_keys=True,
            )
        )


def summarize_quality_v2_pairs(
    baseline_measurements: Iterable[Mapping[str, Any]],
    cached_measurements: Iterable[Mapping[str, Any]],
    *,
    max_tokens: int = QUALITY_V2_MAX_TOKENS,
    bootstrap_draws: int = 10_000,
    seed: int = 20_261_115,
) -> dict[str, QualityV2DatasetSummary]:
    """Pair the two arms by (dataset, example_id) and summarise each dataset.

    Every metric gets both arm means and the paired cached-minus-baseline
    difference with a 95% percentile bootstrap interval over examples.
    """

    def index(rows: Iterable[Mapping[str, Any]], label: str) -> dict[tuple[str, str], Mapping[str, Any]]:
        indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in rows:
            key = (str(row["dataset"]), str(row["example_id"]))
            if key in indexed:
                raise ValueError(f"duplicate {label} measurement for {key}")
            indexed[key] = row
        return indexed

    baseline = index(baseline_measurements, "baseline")
    cached = index(cached_measurements, "cached")
    if set(baseline) != set(cached):
        missing = sorted(set(baseline) ^ set(cached))[:5]
        raise ValueError(f"arms are not paired; first unmatched keys: {missing}")
    by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(baseline):
        by_dataset[key[0]].append(key)
    summaries = {}
    for dataset, keys in sorted(by_dataset.items()):
        arms = {"baseline": [baseline[key] for key in keys], "cached": [cached[key] for key in keys]}
        metric_names = sorted(
            {
                name
                for rows in arms.values()
                for row in rows
                for name in (row.get("quality_scores") or {})
            }
        )
        metrics: dict[str, dict[str, float]] = {}
        for name in metric_names:
            base_values = [_score(row, name) for row in arms["baseline"]]
            cached_values = [_score(row, name) for row in arms["cached"]]
            diffs = [c - b for b, c in zip(base_values, cached_values, strict=True)]
            low, high = _bootstrap_mean_interval(
                diffs, draws=bootstrap_draws, seed=f"{seed}:{dataset}:{name}"
            )
            metrics[name] = {
                "baseline": statistics.fmean(base_values),
                "cached": statistics.fmean(cached_values),
                "difference": statistics.fmean(diffs),
                "difference_ci95_low": low,
                "difference_ci95_high": high,
                "cached_better": float(sum(d > 0 for d in diffs)),
                "cached_worse": float(sum(d < 0 for d in diffs)),
            }
        summaries[dataset] = QualityV2DatasetSummary(
            dataset=dataset,
            pairs=len(keys),
            metrics=metrics,
            strict_format_rate={
                arm: statistics.fmean(
                    float(extract_single_final_answer(str(row.get("output_text") or "")).valid)
                    for row in rows
                )
                for arm, rows in arms.items()
            },
            tolerant_parse_rate={
                arm: statistics.fmean(
                    float(extract_first_final_answer(str(row.get("output_text") or "")).valid)
                    for row in rows
                )
                for arm, rows in arms.items()
            },
            token_cap_rate={
                arm: statistics.fmean(
                    float(int(row.get("completion_tokens") or 0) >= max_tokens)
                    for row in rows
                )
                for arm, rows in arms.items()
            },
            mean_completion_tokens={
                arm: statistics.fmean(float(row.get("completion_tokens") or 0) for row in rows)
                for arm, rows in arms.items()
            },
            errors={arm: sum(1 for row in rows if row.get("error")) for arm, rows in arms.items()},
        )
    return summaries


def _score(row: Mapping[str, Any], name: str) -> float:
    if row.get("error"):
        return 0.0
    value = (row.get("quality_scores") or {}).get(name, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"non-numeric {name} score")
    return float(value)


def _bootstrap_mean_interval(
    values: Sequence[float], *, draws: int, seed: str
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap needs at least one value")
    if draws <= 0:
        return (statistics.fmean(values), statistics.fmean(values))
    count = len(values)
    rng = random.Random(seed)
    population = list(values)
    means = sorted(math.fsum(rng.choices(population, k=count)) / count for _ in range(draws))
    return (means[int(0.025 * (draws - 1))], means[int(math.ceil(0.975 * (draws - 1)))])


QUALITY_V2_REPORT_RECORD_TYPE = "cachet.quality_v2_report.v1"
QUALITY_V2_REPORT_SCHEMA_VERSION = 1
QUALITY_V2_README_MARKERS = (
    "<!-- cachet:vllm-0300-quality-v2:begin -->",
    "<!-- cachet:vllm-0300-quality-v2:end -->",
)
_QUALITY_V2_BOOTSTRAP_SEED = 20_261_115
_QUALITY_V2_PARSER_STATUSES = (
    "ok",
    "missing_block",
    "multiple_or_malformed_blocks",
    "empty_answer",
)
_DOCUMENT_MARKUP = re.compile(r"\[/?(?:document|chunk) id=")
_REPORT_KEYS = frozenset(
    {
        "bootstrap",
        "closed_record_sha256",
        "diagnostics",
        "execution",
        "protocol",
        "record_type",
        "schema_version",
        "summaries",
        "verdicts",
    }
)
_EXECUTION_REQUIRED_KEYS = frozenset(
    {"chunks", "package_wheel_sha256", "plan_sha256", "vanilla_arm_id"}
)
_DATASET_LABELS = {
    "biography": "Biography",
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
    "niah": "NIAH",
}
_METRIC_LABELS = {
    "accuracy": "Exact-value accuracy",
    "answer_em": "Answer exact match, alias-max",
    "answer_f1": "Answer F1, alias-max",
    "exact_match": "Exact match",
    "f1": "Answer F1",
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _closed_sha256(record: Mapping[str, Any]) -> str:
    body = dict(record)
    body["closed_record_sha256"] = ""
    return sha256(_canonical(body).encode("utf-8")).hexdigest()


def _mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def quality_v2_diagnostics(
    baseline_measurements: Iterable[Mapping[str, Any]],
    cached_measurements: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-dataset parse outcomes, NIAH grid cells and the biography reference split."""

    by_key: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for arm, rows in (("baseline", baseline_measurements), ("cached", cached_measurements)):
        for row in rows:
            by_key[(str(row["dataset"]), str(row["example_id"]))][arm] = row
    if any(set(pair) != {"baseline", "cached"} for pair in by_key.values()):
        raise ValueError("diagnostics need paired baseline and cached measurements")
    diagnostics: dict[str, Any] = {}
    for dataset in sorted({key[0] for key in by_key}):
        pairs = [by_key[key] for key in sorted(by_key) if key[0] == dataset]
        metric = quality_v2_primary_metric(dataset)
        item: dict[str, Any] = {}
        for arm in ("baseline", "cached"):
            rows = [pair[arm] for pair in pairs]
            statuses = {status: 0 for status in _QUALITY_V2_PARSER_STATUSES}
            for row in rows:
                statuses[extract_first_final_answer(str(row.get("output_text") or "")).status] += 1
            item[arm] = {
                "document_markup_in_output_rate": _mean_or_zero(
                    float(bool(_DOCUMENT_MARKUP.search(str(row.get("output_text") or ""))))
                    for row in rows
                ),
                "mean_prompt_tokens": _mean_or_zero(
                    float(row.get("prompt_tokens") or 0) for row in rows
                ),
                "parser_statuses": statuses,
            }
        if dataset == "biography":
            split: dict[str, Any] = {}
            for label, disambiguated in (
                ("disambiguated_reference", True),
                ("plain_reference", False),
            ):
                subset = [
                    pair
                    for pair in pairs
                    if any("-lrb-" in str(ref) for ref in pair["baseline"].get("references") or [])
                    is disambiguated
                ]
                split[label] = {
                    "pairs": len(subset),
                    **{
                        arm: _mean_or_zero(_score(pair[arm], metric) for pair in subset)
                        for arm in ("baseline", "cached")
                    },
                }
            item["reference_split"] = split
        if dataset == "niah":
            cells: dict[str, list[dict[str, Mapping[str, Any]]]] = defaultdict(list)
            for pair in pairs:
                cell_id = (pair["baseline"].get("metadata") or {}).get("niah_cell_id")
                if cell_id not in NIAH_CELL_IDS:
                    raise ValueError("NIAH measurement is missing its grid cell")
                cells[str(cell_id)].append(pair)
            item["cells"] = {
                cell_id: {
                    "pairs": len(cells[cell_id]),
                    **{
                        arm: _mean_or_zero(_score(pair[arm], metric) for pair in cells[cell_id])
                        for arm in ("baseline", "cached")
                    },
                }
                for cell_id in NIAH_CELL_IDS
                if cell_id in cells
            }
        diagnostics[dataset] = item
    return diagnostics


def _quality_v2_verdicts(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    verdicts = {}
    for dataset in QUALITY_V2_HEADLINE_DATASETS:
        name = quality_v2_primary_metric(dataset)
        metric = summaries[dataset]["metrics"][name]
        verdicts[dataset] = {
            "difference": metric["difference"],
            "difference_ci95_high": metric["difference_ci95_high"],
            "difference_ci95_low": metric["difference_ci95_low"],
            "meets_non_inferiority": metric["difference_ci95_low"] >= -QUALITY_V2_MAX_REGRESSION,
            "metric": name,
        }
    return verdicts


def quality_v2_report_record(
    baseline_measurements: Iterable[Mapping[str, Any]],
    cached_measurements: Iterable[Mapping[str, Any]],
    *,
    execution: Mapping[str, Any],
    bootstrap_draws: int = 10_000,
) -> dict[str, Any]:
    """Seal one complete quality-v2 run into a publishable report record.

    ``execution`` carries run provenance: the plan and package wheel digests, the
    cached arm id, and one entry per chunk with its dataset and example count.
    """

    baseline = list(baseline_measurements)
    cached = list(cached_measurements)
    summaries = {
        dataset: summary.to_record()
        for dataset, summary in summarize_quality_v2_pairs(
            baseline,
            cached,
            bootstrap_draws=bootstrap_draws,
            seed=_QUALITY_V2_BOOTSTRAP_SEED,
        ).items()
    }
    if set(summaries) != {*QUALITY_V2_HEADLINE_DATASETS, *QUALITY_V2_DIAGNOSTIC_DATASETS}:
        raise ValueError("quality-v2 report must cover every protocol dataset")
    record: dict[str, Any] = {
        "bootstrap": {
            "draws": bootstrap_draws,
            "interval": "percentile_95_over_paired_examples",
            "seed": _QUALITY_V2_BOOTSTRAP_SEED,
        },
        "closed_record_sha256": "",
        "diagnostics": quality_v2_diagnostics(baseline, cached),
        "execution": json.loads(_canonical(dict(execution))),
        "protocol": quality_v2_protocol_record(),
        "record_type": QUALITY_V2_REPORT_RECORD_TYPE,
        "schema_version": QUALITY_V2_REPORT_SCHEMA_VERSION,
        "summaries": summaries,
        "verdicts": _quality_v2_verdicts(summaries),
    }
    record["closed_record_sha256"] = _closed_sha256(record)
    validate_quality_v2_report_record(record)
    return record


def validate_quality_v2_report_record(record: Mapping[str, Any]) -> None:
    """Fail unless ``record`` is a complete, self-consistent quality-v2 report."""

    if not isinstance(record, Mapping) or set(record) != _REPORT_KEYS:
        raise ValueError("quality-v2 report has the wrong fields")
    if record["record_type"] != QUALITY_V2_REPORT_RECORD_TYPE:
        raise ValueError("quality-v2 report record_type is invalid")
    if record["schema_version"] != QUALITY_V2_REPORT_SCHEMA_VERSION:
        raise ValueError("quality-v2 report schema_version is invalid")
    if record["closed_record_sha256"] != _closed_sha256(record):
        raise ValueError("quality-v2 report closure is invalid")
    if record["protocol"] != quality_v2_protocol_record():
        raise ValueError("quality-v2 report protocol differs from this library")
    datasets = set(QUALITY_V2_HEADLINE_DATASETS) | set(QUALITY_V2_DIAGNOSTIC_DATASETS)
    summaries = record["summaries"]
    diagnostics = record["diagnostics"]
    if set(summaries) != datasets or set(diagnostics) != datasets:
        raise ValueError("quality-v2 report must cover every protocol dataset")
    for dataset, summary in summaries.items():
        pairs = summary.get("pairs")
        if type(pairs) is not int or pairs <= 0:
            raise ValueError(f"quality-v2 {dataset} summary has no pairs")
        if quality_v2_primary_metric(dataset) not in summary.get("metrics", {}):
            raise ValueError(f"quality-v2 {dataset} summary lacks its primary metric")
        for arm in ("baseline", "cached"):
            statuses = diagnostics[dataset][arm]["parser_statuses"]
            if set(statuses) != set(_QUALITY_V2_PARSER_STATUSES) or sum(statuses.values()) != pairs:
                raise ValueError(f"quality-v2 {dataset} parser statuses do not cover the pairs")
    if record["verdicts"] != _quality_v2_verdicts(summaries):
        raise ValueError("quality-v2 report verdicts do not follow from its summaries")
    if sum(cell["pairs"] for cell in diagnostics["niah"]["cells"].values()) != (
        summaries["niah"]["pairs"]
    ):
        raise ValueError("quality-v2 NIAH grid does not cover every pair")
    if sum(item["pairs"] for item in diagnostics["biography"]["reference_split"].values()) != (
        summaries["biography"]["pairs"]
    ):
        raise ValueError("quality-v2 biography reference split does not cover every pair")
    execution = record["execution"]
    if not isinstance(execution, Mapping) or not _EXECUTION_REQUIRED_KEYS <= set(execution):
        raise ValueError("quality-v2 report execution provenance is incomplete")
    examples: dict[str, int] = defaultdict(int)
    for chunk in execution["chunks"]:
        examples[str(chunk["dataset"])] += int(chunk["examples"])
    if dict(examples) != {dataset: summary["pairs"] for dataset, summary in summaries.items()}:
        raise ValueError("quality-v2 chunk examples do not match the scored pairs")


def _rate_pair(values: Mapping[str, float]) -> str:
    return f"{100 * values['baseline']:.1f}% / {100 * values['cached']:.1f}%"


def render_quality_v2_readme_region(record: Mapping[str, Any]) -> str:
    """Render the governed README tables for one sealed quality-v2 report."""

    validate_quality_v2_report_record(record)
    summaries = record["summaries"]
    diagnostics = record["diagnostics"]
    verdicts = record["verdicts"]
    margin = f"−{QUALITY_V2_MAX_REGRESSION:.2f}"
    lines = [
        "",
        f"Report `closed_record_sha256`: `{record['closed_record_sha256']}`. "
        f"Protocol `{record['protocol']['protocol_id']}`, parser "
        f"`{QUALITY_V2_PARSER_ID}` v{QUALITY_V2_PARSER_VERSION}, "
        f"{record['bootstrap']['draws']:,} paired bootstrap draws.",
        "",
        "| Dataset | Role | Metric | Pairs | Full prefill | Vanilla KV "
        "| Vanilla − full prefill | Paired 95% CI "
        f"| Non-inferiority (CI low ≥ {margin}) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    ordered = [*QUALITY_V2_HEADLINE_DATASETS, *QUALITY_V2_DIAGNOSTIC_DATASETS]
    for dataset in ordered:
        summary = summaries[dataset]
        primary = quality_v2_primary_metric(dataset)
        names = [primary, *sorted(name for name in summary["metrics"] if name != primary)]
        for name in names:
            metric = summary["metrics"][name]
            if dataset in verdicts and name == primary:
                role = "Headline, primary"
                verdict = "Meets" if verdicts[dataset]["meets_non_inferiority"] else "Does not meet"
            elif dataset in verdicts:
                role = "Headline, secondary"
                verdict = "Not assessed"
            else:
                role = "Diagnostic"
                verdict = "Not assessed"
            lines.append(
                f"| {_DATASET_LABELS[dataset]} | {role} | {_METRIC_LABELS.get(name, name)} "
                f"| {summary['pairs']:,} | {metric['baseline']:.3f} | {metric['cached']:.3f} "
                f"| {metric['difference']:+.3f} "
                f"| [{metric['difference_ci95_low']:+.3f}, {metric['difference_ci95_high']:+.3f}] "
                f"| {verdict} |"
            )
    lines += [
        "",
        "| Dataset | Strict whole-output format (full / Vanilla) "
        "| Answer extracted (full / Vanilla) | Reached 256-token cap (full / Vanilla) "
        "| Mean completion tokens (full / Vanilla) | Document markup echoed (full / Vanilla) "
        "| Request errors (full / Vanilla) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in ordered:
        summary = summaries[dataset]
        markup = {
            arm: diagnostics[dataset][arm]["document_markup_in_output_rate"]
            for arm in ("baseline", "cached")
        }
        tokens = summary["mean_completion_tokens"]
        errors = summary["errors"]
        lines.append(
            f"| {_DATASET_LABELS[dataset]} | {_rate_pair(summary['strict_format_rate'])} "
            f"| {_rate_pair(summary['tolerant_parse_rate'])} "
            f"| {_rate_pair(summary['token_cap_rate'])} "
            f"| {tokens['baseline']:.1f} / {tokens['cached']:.1f} | {_rate_pair(markup)} "
            f"| {errors['baseline']} / {errors['cached']} |"
        )
    lines += [
        "",
        "| NIAH cell | Pairs | Full prefill accuracy | Vanilla KV accuracy | Vanilla − full prefill |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    niah_cells = diagnostics["niah"]["cells"]
    for cell_id in (cell_id for cell_id in NIAH_CELL_IDS if cell_id in niah_cells):
        cell = niah_cells[cell_id]
        _, context, _, depth = cell_id.split("-")
        lines.append(
            f"| {context}, needle at {int(depth)}% | {cell['pairs']:,} | {cell['baseline']:.3f} "
            f"| {cell['cached']:.3f} | {cell['cached'] - cell['baseline']:+.3f} |"
        )
    lines += [
        "",
        "| Biography reference | Pairs | Full prefill exact match | Vanilla KV exact match |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, title in (
        ("disambiguated_reference", "Had a WikiBio `-lrb- … -rrb-` disambiguator"),
        ("plain_reference", "Plain title"),
    ):
        item = diagnostics["biography"]["reference_split"][label]
        lines.append(
            f"| {title} | {item['pairs']:,} | {item['baseline']:.3f} | {item['cached']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def validate_quality_v2_readme_region(readme: str, record: Mapping[str, Any]) -> None:
    """Fail unless ``readme`` holds exactly the rendered quality-v2 region."""

    begin, end = QUALITY_V2_README_MARKERS
    if readme.count(begin) != 1 or readme.count(end) != 1:
        raise ValueError("README must contain one quality-v2 region")
    start = readme.index(begin) + len(begin)
    stop = readme.index(end)
    if stop < start or readme[start:stop] != render_quality_v2_readme_region(record):
        raise ValueError("README quality-v2 region differs from the sealed report")
