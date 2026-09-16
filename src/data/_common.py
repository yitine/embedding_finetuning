"""Shared utilities for loading and preparing MTEB datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from datasets import DatasetDict, load_dataset


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_split(dataset_name: str, split: str) -> Any:
    """Load a named split, falling back to the first available split."""
    dataset = load_dataset(dataset_name)
    if isinstance(dataset, DatasetDict):
        if split in dataset:
            return dataset[split]
        if not dataset:
            raise ValueError(f"Dataset {dataset_name!r} has no splits")
        return next(iter(dataset.values()))
    return dataset


def load_config(dataset_name: str, config_name: str) -> Any:
    """Load the first split from a named HuggingFace dataset config."""
    dataset = load_dataset(dataset_name, name=config_name)
    if isinstance(dataset, DatasetDict):
        if not dataset:
            raise ValueError(f"Dataset config {dataset_name!r}/{config_name!r} has no splits")
        return next(iter(dataset.values()))
    return dataset


def records(dataset: Any) -> list[dict[str, Any]]:
    """Convert a HuggingFace dataset or iterable of mappings to plain records."""
    return [dict(row) for row in dataset]


def extract_text_mapping(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Extract ID-to-text mappings from MTEB query or corpus records."""
    mapping: dict[str, str] = {}
    for row in rows:
        identifier = identifier_from_row(row)
        text = text_from_row(row)
        if identifier is not None and text is not None:
            mapping[identifier] = text
    return mapping


def first_value(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def text_from_row(row: Mapping[str, Any]) -> str | None:
    value = first_value(row, ("text", "contents", "passage", "document", "query"))
    if isinstance(value, Mapping):
        value = first_value(value, ("text", "contents", "title"))
    return None if value is None else str(value)


def identifier_from_row(row: Mapping[str, Any]) -> str | None:
    value = first_value(row, ("_id", "id", "doc_id", "document_id", "corpus_id", "query_id", "query-id"))
    return None if value is None else str(value)


def extract_query_corpus_qrels(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Extract query, corpus, and relevance mappings from MTEB-style rows."""
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for row in rows:
        nested_corpus = row.get("corpus")
        if isinstance(nested_corpus, Mapping):
            for doc_id, document in nested_corpus.items():
                corpus[str(doc_id)] = str(document.get("text", document) if isinstance(document, Mapping) else document)

        nested_queries = row.get("queries")
        if isinstance(nested_queries, Mapping):
            for query_id, query in nested_queries.items():
                queries[str(query_id)] = str(query)

        query_id = first_value(row, ("query_id", "query-id", "qid"))
        query_text = first_value(row, ("query", "question"))
        if query_id is not None and query_text is not None:
            queries[str(query_id)] = str(query_text)

        doc_id = first_value(row, ("corpus_id", "corpus-id", "doc_id", "document_id", "passage_id"))
        doc_text = first_value(row, ("passage", "document", "contents", "text"))
        if doc_id is not None and doc_text is not None and query_id is not None:
            corpus[str(doc_id)] = str(doc_text)
        score = first_value(row, ("score", "relevance", "label"))
        if query_id is not None and doc_id is not None and score is not None:
            qrels.setdefault(str(query_id), {})[str(doc_id)] = int(score)

        row_text = text_from_row(row)
        if row_text is not None and query_text is None:
            row_id = identifier_from_row(row)
            if row_id is not None:
                corpus[row_id] = row_text

    return queries, corpus, qrels


def save_processed(name: str, split: str, payload: Mapping[str, Any]) -> Path:
    """Save a normalized dataset payload as JSON under data/processed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{name}_{split}.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return output_path


def sample_in_batch_negatives(
    positive_doc_ids: Iterable[str],
    batch_doc_ids: Iterable[str],
    *,
    min_negatives: int = 7,
    max_negatives: int = 15,
    seed: int = 42,
) -> list[str]:
    """Sample 7-15 other batch documents as negatives for one positive."""
    if not 0 < min_negatives <= max_negatives:
        raise ValueError("Require 0 < min_negatives <= max_negatives")

    positives = {str(doc_id) for doc_id in positive_doc_ids}
    candidates = [str(doc_id) for doc_id in dict.fromkeys(batch_doc_ids) if str(doc_id) not in positives]
    if not candidates:
        return []

    sample_size = min(max_negatives, max(min_negatives, len(candidates)))
    sample_size = min(sample_size, len(candidates))
    generator = random.Random(seed)
    return generator.sample(candidates, sample_size)


def build_in_batch_negatives(
    positives: Mapping[str, Iterable[str]],
    batch_query_ids: Iterable[str],
    *,
    min_negatives: int = 7,
    max_negatives: int = 15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Build negatives for every positive in a batch of query IDs."""
    batch_doc_ids = [
        str(doc_id)
        for query_id in batch_query_ids
        for doc_id in positives.get(str(query_id), [])
    ]
    return {
        str(query_id): sample_in_batch_negatives(
            positives.get(str(query_id), []),
            batch_doc_ids,
            min_negatives=min_negatives,
            max_negatives=max_negatives,
            seed=seed + index,
        )
        for index, query_id in enumerate(batch_query_ids)
    }
