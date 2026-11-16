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
    "QualityV2DatasetSummary",
    "biography_v3_entity_scores",
    "build_quality_v2_niah_records",
    "extract_first_final_answer",
    "niah_v2_exact_value_scores",
    "quality_v2_primary_metric",
    "quality_v2_prompt_parts",
    "quality_v2_protocol_record",
    "quality_v2_scorer_registry",
    "strip_biography_disambiguator",
    "summarize_quality_v2_pairs",
]

QUALITY_V2_PROTOCOL_ID = "cachet-vllm-0.30.0-quality-v2"
QUALITY_V2_PROMPT_TEMPLATE_VERSION = "v3-qwen3-chat-final-answer"
QUALITY_V2_PROMPT_PLUGIN_PATH = "document_kv_cache.quality_v2:quality_v2_prompt_parts"
QUALITY_V2_MAX_TOKENS = 256
QUALITY_V2_NIAH_SEED = 20_261_115
QUALITY_V2_HEADLINE_DATASETS = ("hotpotqa", "musique", "niah")
QUALITY_V2_DIAGNOSTIC_DATASETS = ("biography",)
# Largest accepted drop of the cached arm below full prefill on a headline
# dataset's primary metric, as the upper end of a 95% paired bootstrap interval.
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
    try:
        import numpy
    except ImportError:
        rng = random.Random(seed)
        means = sorted(
            sum(values[rng.randrange(count)] for _ in range(count)) / count
            for _ in range(draws)
        )
    else:
        generator = numpy.random.default_rng(
            int.from_bytes(sha256(seed.encode("utf-8")).digest()[:8], "big")
        )
        array = numpy.asarray(values, dtype=numpy.float64)
        means = []
        for start in range(0, draws, 500):
            block = generator.integers(0, count, size=(min(500, draws - start), count))
            means.extend(array[block].mean(axis=1).tolist())
        means.sort()
    return (means[int(0.025 * (draws - 1))], means[int(math.ceil(0.975 * (draws - 1)))])
