"""Run dense-retrieval baselines for NFCorpus and SciFact."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable
from tqdm import tqdm
import faiss

# Avoid native thread/process conflicts between PyTorch, tokenizers, and FAISS.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "baseline"
TOP_K = 100
EVAL_K = 10




def select_device(requested_device: str | None = None) -> str:
    """Select the best available device (CUDA > MPS > CPU)."""
    if requested_device:
        return requested_device
    
    try:
        import torch
        
        if torch.cuda.is_available():
            return "cuda"
        if sys.platform == "darwin" and hasattr(torch.backends, "mps"):
            try:
                if torch.backends.mps.is_available():
                    return "mps"
            except (AttributeError, RuntimeError):
                pass  # MPS check failed, fall back to CPU
    except (ImportError, AttributeError):
        pass
    
    return "cpu"


def require_dependency(module_name: str, *, install_hint: str) -> Any:
    """Import a dependency or raise a user-friendly install hint."""
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing dependency '{module_name}'. Install it with: {install_hint}"
        ) from exc


def load_model_config() -> dict[str, dict[str, Any]]:
    yaml = require_dependency("yaml", install_hint="pip install pyyaml")
    
    try:
        with CONFIG_PATH.open() as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse {CONFIG_PATH}: {e}") from e
    
    models = config.get("models", {})
    if not models:
        raise ValueError(f"No models found in {CONFIG_PATH}")
    return models


def load_processed_dataset(dataset_name: str) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    path = PROCESSED_DIR / f"{dataset_name}_test.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate it first with "
            f"load_{dataset_name}(\"test\") or the corresponding data loader."
        )

    with path.open() as handle:
        payload = json.load(handle)
    relevance_key = "positives" if dataset_name == "nfcorpus" else "qrels"
    return payload["queries"], payload["corpus"], payload[relevance_key]


def relevance_set(relevance: Any) -> set[str]:
    """Convert graded relevance judgments to binary relevance.

    Any document with a relevance score > 0 is considered relevant.
    This intentionally discards relevance levels and is used for
    binary metrics such as Recall@10, MAP@10, and MRR@10.
    """
    if isinstance(relevance, dict):
        return {str(doc_id) for doc_id, score in relevance.items() if float(score) > 0}
    return {str(doc_id) for doc_id in relevance}


def dcg(relevances: Iterable[int]) -> float:
    return sum(relevance / np.log2(rank + 2) for rank, relevance in enumerate(relevances))

    

    
def query_metrics(
    retrieved_ids: list[str],
    relevance: Any,  
    k: int = EVAL_K,
) -> dict[str, float]:
    relevant_ids = relevance_set(relevance)  
    ranked = retrieved_ids[:k]
    hits = [int(doc_id in relevant_ids) for doc_id in ranked]
    relevant_count = len(relevant_ids)

    # NDCG uses original graded relevance scores
    scores = relevance if isinstance(relevance, dict) else {
        doc_id: 1 for doc_id in relevant_ids
    }
    # DCG: reward relevant documents, with higher ranks weighted more.
    gains = [2 ** float(scores.get(doc_id, 0)) - 1 for doc_id in ranked]
    # assume all relevant documents are ranked in the optimal order,
    ideal = sorted(
        (2 ** float(score) - 1 for score in scores.values()),
        reverse=True,
    )[:k]
    ideal_dcg = dcg(ideal)
    ndcg = dcg(gains) / ideal_dcg if ideal_dcg else 0.0

    
    recall = sum(hits) / relevant_count if relevant_count else 0.0

    average_precision = 0.0
    hit_count = 0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            hit_count += 1
            average_precision += hit_count / rank
    average_precision = average_precision / relevant_count if relevant_count else 0.0

    first_hit = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
    reciprocal_rank = 1.0 / first_hit if first_hit is not None else 0.0
    return {
        "ndcg_at_10": ndcg,
        "recall_at_10": recall,
        "map_at_10": average_precision,
        "mrr_at_10": reciprocal_rank,
    }  


def average_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "map_at_10": 0.0, "mrr_at_10": 0.0}
    return {
        metric: float(np.mean([result[metric] for result in per_query]))
        for metric in ("ndcg_at_10", "recall_at_10", "map_at_10", "mrr_at_10")
    }


def encode(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    faiss = require_dependency("faiss", install_hint="pip install faiss-cpu")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32", copy=False)
    faiss.normalize_L2(embeddings)
    return embeddings


def evaluate_dataset(
    model: SentenceTransformer,
    dataset_name: str,
    *,
    corpus_batch_size: int = 64,
    query_batch_size: int = 32,
) -> dict[str, float]:
    faiss = require_dependency("faiss", install_hint="pip install faiss-cpu")
  

    queries, corpus, relevance = load_processed_dataset(dataset_name)
    corpus_ids = list(corpus)
    query_ids = [query_id for query_id in queries if query_id in relevance]
    if not corpus_ids or not query_ids:
        raise ValueError(f"{dataset_name} contains no corpus/query relevance data")

    corpus_embeddings = encode(model, [corpus[doc_id] for doc_id in corpus_ids], corpus_batch_size)
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_embeddings)

    query_embeddings = encode(model, [queries[query_id] for query_id in query_ids], query_batch_size)
    _, indices = index.search(query_embeddings, min(TOP_K, len(corpus_ids)))
    per_query = []
    for query_id, retrieved_indices in tqdm(
        zip(query_ids, indices), total=len(query_ids), desc=f"Metrics {dataset_name}"
    ):
        retrieved_ids = [corpus_ids[index] for index in retrieved_indices if index >= 0]
        per_query.append(query_metrics(retrieved_ids, relevance[query_id]))
    return average_metrics(per_query)


def write_results(results: dict[str, dict[str, dict[str, float]]]) -> None:
    import pandas as pd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "results.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")

    rows = []
    for model_name, datasets in results.items():
        for dataset_name, metrics in datasets.items():
            rows.append({"model": model_name, "dataset": dataset_name, **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    markdown_rows = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        markdown_rows.append(
            "| " + " | ".join(
                f"{value:.4f}" if isinstance(value, float) else str(value)
                for value in row
            ) + " |"
        )
    (RESULTS_DIR / "results.md").write_text("\n".join(markdown_rows) + "\n")

    print("\nBaseline results")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved results to {RESULTS_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=None, help="Model key from configs/model_config.yaml")
    parser.add_argument("--dataset", choices=("nfcorpus", "scifact"), help="Evaluate only one dataset")
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        help="Execution device; macOS defaults to CPU to avoid native runtime conflicts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = load_model_config()
    if args.model and args.model not in models:
        raise ValueError(f"Unknown model {args.model!r}; choose from {', '.join(models)}")

    model_names = [args.model] if args.model else list(models)
    dataset_names = [args.dataset] if args.dataset else ["nfcorpus", "scifact"]
    device = select_device(args.device)
    if device == "cpu":
        try:
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except (ImportError, RuntimeError):
            pass
    print(f"Using device: {device}")

    results: dict[str, dict[str, dict[str, float]]] = {}
    for model_key in model_names:
        model_config = models[model_key]
        print(f"\nLoading {model_key}: {model_config['model_name']}")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_config["model_name"], device=device)
        if "max_seq_length" in model_config:
            model.max_seq_length = model_config["max_seq_length"]
        results[model_key] = {
            dataset_name: evaluate_dataset(model, dataset_name)
            for dataset_name in dataset_names
        }

    write_results(results)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
