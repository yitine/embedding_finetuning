"""Load and normalize the MTEB NFCorpus dataset."""

from __future__ import annotations

from typing import Any

from ._common import (
    extract_query_corpus_qrels,
    extract_text_mapping,
    first_value,
    load_config,
    load_split,
    records,
    save_processed,
)

DATASET_NAME = "mteb/nfcorpus"


def _positive_ids(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key, score in value.items() if float(score) > 0]
    if isinstance(value, (list, tuple, set)):
        return [str(item.get("id", item.get("doc_id"))) if isinstance(item, dict) else str(item) for item in value]
    return []


def load_nfcorpus(split: str = "train") -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    """Load NFCorpus and return ``(queries, corpus, positives)``.

    The normalized result is also written to ``data/processed`` as JSON. Positive
    document IDs are grouped by query ID; training code can use the shared
    ``sample_in_batch_negatives`` helper for in-batch negatives.
    """
    rows = records(load_split(DATASET_NAME, split))
    queries = extract_text_mapping(records(load_config(DATASET_NAME, "queries")))
    corpus = extract_text_mapping(records(load_config(DATASET_NAME, "corpus")))
    _, _, qrels = extract_query_corpus_qrels(rows)
    positives: dict[str, list[str]] = {
        query_id: [doc_id for doc_id, score in docs.items() if score > 0]
        for query_id, docs in qrels.items()
    }

    for row in rows:
        query_id = first_value(row, ("query_id", "query-id", "qid"))
        positive_value = first_value(row, ("positives", "positive", "positive_passages"))
        if query_id is not None and positive_value is not None:
            positives.setdefault(str(query_id), []).extend(_positive_ids(positive_value))

    positives = {query_id: sorted(set(doc_ids)) for query_id, doc_ids in positives.items()}
    save_processed(
        "nfcorpus",
        split,
        {"queries": queries, "corpus": corpus, "positives": positives},
    )
    return queries, corpus, positives


__all__ = ["load_nfcorpus"]
