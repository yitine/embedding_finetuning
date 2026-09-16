"""Load and normalize the MTEB SciFact dataset."""

from __future__ import annotations

from ._common import (
    extract_query_corpus_qrels,
    extract_text_mapping,
    load_config,
    load_split,
    records,
    save_processed,
)

DATASET_NAME = "mteb/scifact"


def load_scifact(split: str = "test") -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Load SciFact and return ``(queries, corpus, qrels)``.

    The normalized result is also written to ``data/processed`` as JSON. Qrels
    map each query ID to document IDs and integer relevance scores.
    """
    rows = records(load_split(DATASET_NAME, split))
    queries = extract_text_mapping(records(load_config(DATASET_NAME, "queries")))
    corpus = extract_text_mapping(records(load_config(DATASET_NAME, "corpus")))
    _, _, qrels = extract_query_corpus_qrels(rows)
    save_processed(
        "scifact",
        split,
        {"queries": queries, "corpus": corpus, "qrels": qrels},
    )
    return queries, corpus, qrels


__all__ = ["load_scifact"]
