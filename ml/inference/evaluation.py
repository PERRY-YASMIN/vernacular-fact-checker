from __future__ import annotations

from typing import Dict, Iterable, Sequence

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def evaluate_claim_detection(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def evaluate_retrieval(relevant_ids: Sequence[set[str]], retrieved_ids: Sequence[Sequence[str]], k: int = 5) -> Dict[str, float]:
    hits = 0
    mrr_total = 0.0
    recall_total = 0.0

    for rel, ret in zip(relevant_ids, retrieved_ids):
        top_k = list(ret)[:k]
        if any(doc_id in rel for doc_id in top_k):
            hits += 1

        rank = 0
        for i, doc_id in enumerate(top_k, start=1):
            if doc_id in rel:
                rank = i
                break
        if rank:
            mrr_total += 1.0 / rank

        if rel:
            found = sum(1 for doc_id in top_k if doc_id in rel)
            recall_total += found / len(rel)

    n = max(len(relevant_ids), 1)
    return {
        f"hit_rate@{k}": float(hits / n),
        f"mrr@{k}": float(mrr_total / n),
        f"recall@{k}": float(recall_total / n),
    }


def evaluate_verification(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_end_to_end(claim_metrics: Dict[str, float], retrieval_metrics: Dict[str, float], verification_metrics: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update({f"claim_{k}": v for k, v in claim_metrics.items()})
    out.update({f"retrieval_{k}": v for k, v in retrieval_metrics.items()})
    out.update({f"verification_{k}": v for k, v in verification_metrics.items()})
    return out
