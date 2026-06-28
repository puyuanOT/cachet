from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import zipfile
from typing import Any

from document_kv_cache.dataset_prep import build_niah_record, write_v1_jsonl


HOTPOTQA_REPO_ID = "hotpotqa/hotpot_qa"
HOTPOTQA_VALIDATION_FILE = "distractor/validation-00000-of-00001.parquet"
MUSIQUE_REPO_ID = "voidful/MuSiQue"
MUSIQUE_ANSWERABLE_DEV_FILE = "musique_ans_v1.0_dev.jsonl"
WIKIBIO_REPO_ID = "michaelauli/wiki_bio"
WIKIBIO_ARCHIVE_FILE = "data/wikipedia-biography-dataset.zip"
WIKIBIO_SPLIT_NAMES = {
    "train": "train",
    "validation": "valid",
    "valid": "valid",
    "test": "test",
}
DEFAULT_NIAH_SAMPLE_COUNT = 1000
DEFAULT_NIAH_CONTEXT_TOKEN_TARGETS = (8192, 16384, 32768)
DEFAULT_NIAH_NEEDLE_POSITIONS = (0.1, 0.5, 0.9)
DATASET_FILENAMES = {
    "biography": "biography.jsonl",
    "hotpotqa": "hotpotqa.jsonl",
    "musique": "musique.jsonl",
    "niah": "niah.jsonl",
}

__all__ = [
    "HOTPOTQA_REPO_ID",
    "HOTPOTQA_VALIDATION_FILE",
    "MUSIQUE_REPO_ID",
    "MUSIQUE_ANSWERABLE_DEV_FILE",
    "WIKIBIO_REPO_ID",
    "WIKIBIO_ARCHIVE_FILE",
    "DEFAULT_NIAH_SAMPLE_COUNT",
    "DEFAULT_NIAH_CONTEXT_TOKEN_TARGETS",
    "DEFAULT_NIAH_NEEDLE_POSITIONS",
    "DatasetSourceStagingConfig",
    "stage_full_benchmark_datasets",
    "stage_hotpotqa_parquet",
    "stage_musique_jsonl",
    "stage_wikibio_zip",
    "stage_niah_jsonl",
    "main",
]


@dataclass(frozen=True, slots=True)
class DatasetSourceStagingConfig:
    output_dir: Path
    cache_dir: Path | None = None
    hotpotqa_parquet: Path | None = None
    musique_jsonl: Path | None = None
    wikibio_zip: Path | None = None
    wikibio_split: str = "validation"
    niah_sample_count: int = DEFAULT_NIAH_SAMPLE_COUNT
    niah_context_token_targets: tuple[int, ...] = DEFAULT_NIAH_CONTEXT_TOKEN_TARGETS
    niah_needle_positions: tuple[float, ...] = DEFAULT_NIAH_NEEDLE_POSITIONS
    seed: int = 20260628
    limit_per_dataset: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        for field_name in ("hotpotqa_parquet", "musique_jsonl", "wikibio_zip"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value))
        if self.wikibio_split not in WIKIBIO_SPLIT_NAMES:
            raise ValueError(f"wikibio_split must be one of {sorted(WIKIBIO_SPLIT_NAMES)}")
        if self.niah_sample_count <= 0:
            raise ValueError("niah_sample_count must be positive")
        if not self.niah_context_token_targets or any(target <= 0 for target in self.niah_context_token_targets):
            raise ValueError("niah_context_token_targets must contain positive integers")
        if not self.niah_needle_positions or any(
            position <= 0.0 or position >= 1.0 for position in self.niah_needle_positions
        ):
            raise ValueError("niah_needle_positions must be between 0 and 1")
        if self.limit_per_dataset is not None and self.limit_per_dataset <= 0:
            raise ValueError("limit_per_dataset must be positive when provided")


def stage_full_benchmark_datasets(config: DatasetSourceStagingConfig) -> dict[str, Any]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    hotpotqa_path = config.hotpotqa_parquet or _hf_download(
        HOTPOTQA_REPO_ID,
        HOTPOTQA_VALIDATION_FILE,
        cache_dir=config.cache_dir,
    )
    musique_path = config.musique_jsonl or _hf_download(
        MUSIQUE_REPO_ID,
        MUSIQUE_ANSWERABLE_DEV_FILE,
        cache_dir=config.cache_dir,
    )
    wikibio_path = config.wikibio_zip or _hf_download(
        WIKIBIO_REPO_ID,
        WIKIBIO_ARCHIVE_FILE,
        cache_dir=config.cache_dir,
    )

    datasets = {
        "biography": stage_wikibio_zip(
            wikibio_path,
            output_dir / DATASET_FILENAMES["biography"],
            split=config.wikibio_split,
            limit=config.limit_per_dataset,
        ),
        "hotpotqa": stage_hotpotqa_parquet(
            hotpotqa_path,
            output_dir / DATASET_FILENAMES["hotpotqa"],
            limit=config.limit_per_dataset,
        ),
        "musique": stage_musique_jsonl(
            musique_path,
            output_dir / DATASET_FILENAMES["musique"],
            limit=config.limit_per_dataset,
        ),
        "niah": stage_niah_jsonl(
            output_dir / DATASET_FILENAMES["niah"],
            sample_count=_effective_niah_sample_count(config),
            context_token_targets=config.niah_context_token_targets,
            needle_positions=config.niah_needle_positions,
            seed=config.seed,
        ),
    }
    record = {
        "record_type": "document_kv.full_benchmark_dataset_sources.v1",
        "datasets": datasets,
        "dataset_specs": [f"{dataset}={output_dir / filename}" for dataset, filename in DATASET_FILENAMES.items()],
        "source_files": {
            "biography": str(wikibio_path),
            "hotpotqa": str(hotpotqa_path),
            "musique": str(musique_path),
            "niah": "synthetic",
        },
        "source_refs": {
            "biography": {
                "repo_id": WIKIBIO_REPO_ID,
                "file": WIKIBIO_ARCHIVE_FILE,
                "split": config.wikibio_split,
            },
            "hotpotqa": {
                "repo_id": HOTPOTQA_REPO_ID,
                "file": HOTPOTQA_VALIDATION_FILE,
                "split": "validation",
            },
            "musique": {
                "repo_id": MUSIQUE_REPO_ID,
                "file": MUSIQUE_ANSWERABLE_DEV_FILE,
                "split": "answerable_dev",
            },
            "niah": {
                "kind": "synthetic_needle_grid",
                "sample_count": datasets["niah"]["records"],
                "context_token_targets": list(config.niah_context_token_targets),
                "needle_positions": list(config.niah_needle_positions),
                "seed": config.seed,
            },
        },
        "limit_per_dataset": config.limit_per_dataset,
    }
    (output_dir / "dataset-source-metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def stage_hotpotqa_parquet(
    parquet_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on caller environment.
        raise RuntimeError("stage_hotpotqa_parquet requires pyarrow") from exc

    rows = pq.read_table(parquet_path).to_pylist()
    records = []
    for row in _limit(rows, limit):
        context = row["context"]
        titles = context["title"]
        sentences = context["sentences"]
        records.append(
            {
                "dataset": "hotpotqa",
                "example_id": row["id"],
                "query": row["question"],
                "expected_answer": row["answer"],
                "documents": [
                    {
                        "document_id": title,
                        "title": title,
                        "chunks": list(document_sentences),
                    }
                    for title, document_sentences in zip(titles, sentences, strict=True)
                ],
                "metadata": {
                    "source": "hotpotqa/hotpot_qa",
                    "split": "validation",
                    "type": str(row.get("type", "")),
                    "level": str(row.get("level", "")),
                },
            }
        )
    count = write_v1_jsonl(records, output_path)
    return {"dataset": "hotpotqa", "records": count, "path": str(output_path), "split": "validation"}


def stage_musique_jsonl(
    jsonl_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    records = []
    for row in _limit(_iter_jsonl(jsonl_path), limit):
        if row.get("answerable") is False:
            continue
        records.append(
            {
                "dataset": "musique",
                "example_id": row["id"],
                "query": row["question"],
                "expected_answer": row["answer"],
                "documents": [
                    {
                        "document_id": str(paragraph.get("idx", paragraph_index)),
                        "title": str(paragraph.get("title", "")),
                        "text": paragraph["paragraph_text"],
                        "metadata": {"is_supporting": str(bool(paragraph.get("is_supporting", False))).lower()},
                    }
                    for paragraph_index, paragraph in enumerate(row["paragraphs"])
                ],
                "metadata": {
                    "source": "voidful/MuSiQue",
                    "split": "answerable_dev",
                    "answerable": str(bool(row.get("answerable", True))).lower(),
                },
            }
        )
    count = write_v1_jsonl(records, output_path)
    return {"dataset": "musique", "records": count, "path": str(output_path), "split": "answerable_dev"}


def stage_wikibio_zip(
    zip_path: str | Path,
    output_path: str | Path,
    *,
    split: str = "validation",
    limit: int | None = None,
) -> dict[str, Any]:
    split_name = WIKIBIO_SPLIT_NAMES.get(split)
    if split_name is None:
        raise ValueError(f"split must be one of {sorted(WIKIBIO_SPLIT_NAMES)}")
    with zipfile.ZipFile(zip_path) as archive:
        records = _wikibio_records_from_archive(archive, split_name=split_name, limit=limit)
        count = write_v1_jsonl(records, output_path)
    return {"dataset": "biography", "records": count, "path": str(output_path), "split": split}


def stage_niah_jsonl(
    output_path: str | Path,
    *,
    sample_count: int = DEFAULT_NIAH_SAMPLE_COUNT,
    context_token_targets: Sequence[int] = DEFAULT_NIAH_CONTEXT_TOKEN_TARGETS,
    needle_positions: Sequence[float] = DEFAULT_NIAH_NEEDLE_POSITIONS,
    seed: int = 20260628,
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not context_token_targets or any(target <= 0 for target in context_token_targets):
        raise ValueError("context_token_targets must contain positive integers")
    if not needle_positions or any(position <= 0.0 or position >= 1.0 for position in needle_positions):
        raise ValueError("needle_positions must be between 0 and 1")
    records = (
        _niah_record(
            index=index,
            seed=seed,
            context_token_target=context_token_targets[index % len(context_token_targets)],
            needle_position=needle_positions[(index // len(context_token_targets)) % len(needle_positions)],
        )
        for index in range(sample_count)
    )
    count = write_v1_jsonl(records, output_path)
    return {
        "dataset": "niah",
        "records": count,
        "path": str(output_path),
        "kind": "synthetic_needle_grid",
        "context_token_targets": list(context_token_targets),
        "needle_positions": list(needle_positions),
        "seed": seed,
    }


def _wikibio_records_from_archive(
    archive: zipfile.ZipFile,
    *,
    split_name: str,
    limit: int | None,
) -> Iterable[Mapping[str, Any]]:
    base = f"wikipedia-biography-dataset/{split_name}/{split_name}"
    with archive.open(f"{base}.id") as id_file, archive.open(f"{base}.nb") as nb_file, archive.open(
        f"{base}.sent"
    ) as sent_file, archive.open(f"{base}.title") as title_file:
        emitted = 0
        for raw_id, raw_nb, raw_title in zip(id_file, nb_file, title_file):
            if limit is not None and emitted >= limit:
                break
            example_id = raw_id.decode("utf-8", errors="replace").strip()
            sentence_count = int(raw_nb.decode("utf-8", errors="replace").strip())
            title = _decode_wikibio_title(raw_title)
            sentences = [
                sent_file.readline().decode("utf-8", errors="replace").strip()
                for _ in range(sentence_count)
            ]
            document_text = " ".join(sentence for sentence in sentences if sentence)
            if not example_id or not title or not document_text:
                continue
            emitted += 1
            yield {
                "dataset": "biography",
                "example_id": example_id,
                "query": "Which person is described in the biography?",
                "expected_answer": title,
                "documents": [
                    {
                        "document_id": example_id,
                        "title": title,
                        "text": document_text,
                    }
                ],
                "metadata": {
                    "source": "michaelauli/wiki_bio",
                    "split": split_name,
                    "article_title": title,
                },
            }


def _niah_record(
    *,
    index: int,
    seed: int,
    context_token_target: int,
    needle_position: float,
) -> Mapping[str, Any]:
    answer = f"cachet-needle-{seed}-{index:05d}"
    needle = f"The hidden answer is {answer}."
    haystack = _synthetic_haystack(
        needle=needle,
        approx_token_target=context_token_target,
        needle_position=needle_position,
    )
    return build_niah_record(
        example_id=f"niah-{seed}-{index:05d}",
        haystack_text=haystack,
        needle_answer=answer,
        needle_text=needle,
        metadata={
            "source": "synthetic_niah_grid",
            "context_token_target": str(context_token_target),
            "needle_position": f"{needle_position:.3f}",
            "seed": str(seed),
        },
    )


def _synthetic_haystack(*, needle: str, approx_token_target: int, needle_position: float) -> str:
    target_chars = max(512, (approx_token_target - 512) * 4)
    filler = (
        "This deterministic filler sentence is irrelevant to the hidden answer. "
        "It exists only to control the benchmark context length. "
    )
    insert_at = int(target_chars * needle_position)
    before = _repeat_to_length(filler, insert_at)
    after = _repeat_to_length(filler, max(0, target_chars - insert_at - len(needle)))
    return before + needle + after


def _repeat_to_length(text: str, target_length: int) -> str:
    if target_length <= 0:
        return ""
    repeats = target_length // len(text) + 1
    return (text * repeats)[:target_length]


def _decode_wikibio_title(raw_title: bytes) -> str:
    return raw_title.decode("utf-8", errors="replace").strip().replace("_", " ")


def _effective_niah_sample_count(config: DatasetSourceStagingConfig) -> int:
    if config.limit_per_dataset is None:
        return config.niah_sample_count
    return min(config.niah_sample_count, config.limit_per_dataset)


def _hf_download(repo_id: str, filename: str, *, cache_dir: Path | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on caller environment.
        raise RuntimeError("Dataset staging requires huggingface_hub when source paths are not provided") from exc
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=cache_dir))


def _iter_jsonl(path: str | Path) -> Iterable[Mapping[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL line {line_number} must be an object")
            yield row


def _limit(values: Iterable[Any], limit: int | None) -> Iterable[Any]:
    if limit is None:
        yield from values
        return
    if limit <= 0:
        raise ValueError("limit must be positive when provided")
    for index, value in enumerate(values):
        if index >= limit:
            break
        yield value


def _int_tuple(values: Sequence[str] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        return default
    parsed = tuple(int(value) for value in values)
    if any(value <= 0 for value in parsed):
        raise ValueError("integer values must be positive")
    return parsed


def _float_tuple(values: Sequence[str] | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return default
    parsed = tuple(float(value) for value in values)
    if any(value <= 0.0 or value >= 1.0 for value in parsed):
        raise ValueError("needle positions must be between 0 and 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> DatasetSourceStagingConfig:
    parser = argparse.ArgumentParser(description="Stage full Cachet benchmark datasets as canonical V1 JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--hotpotqa-parquet")
    parser.add_argument("--musique-jsonl")
    parser.add_argument("--wikibio-zip")
    parser.add_argument("--wikibio-split", default="validation", choices=sorted(WIKIBIO_SPLIT_NAMES))
    parser.add_argument("--niah-sample-count", type=int, default=DEFAULT_NIAH_SAMPLE_COUNT)
    parser.add_argument("--niah-context-token-target", action="append", default=None)
    parser.add_argument("--niah-needle-position", action="append", default=None)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        help="Debug/canary limit. Omit for full benchmark-score datasets.",
    )
    args = parser.parse_args(argv)
    return DatasetSourceStagingConfig(
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        hotpotqa_parquet=Path(args.hotpotqa_parquet) if args.hotpotqa_parquet else None,
        musique_jsonl=Path(args.musique_jsonl) if args.musique_jsonl else None,
        wikibio_zip=Path(args.wikibio_zip) if args.wikibio_zip else None,
        wikibio_split=args.wikibio_split,
        niah_sample_count=args.niah_sample_count,
        niah_context_token_targets=_int_tuple(
            args.niah_context_token_target,
            DEFAULT_NIAH_CONTEXT_TOKEN_TARGETS,
        ),
        niah_needle_positions=_float_tuple(args.niah_needle_position, DEFAULT_NIAH_NEEDLE_POSITIONS),
        seed=args.seed,
        limit_per_dataset=args.limit_per_dataset,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        record = stage_full_benchmark_datasets(parse_args(argv))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True, **record}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
