from __future__ import annotations

from typing import Any

from .claim_detector import extract_claims
from .embedder import embed_text
from .fluff_filter import clean_text
from .retrieval_pipeline import retrieve_facts
from .verifier import verify_claim


def fact_check_text(text: str) -> dict[str, Any]:
    """Run end-to-end fact checking for all extracted claims in text."""
    normalized_text = clean_text(text)
    claims = extract_claims(normalized_text)

    results = []
    retrieval_hits = 0
    verdict_confidences = []
    for claim in claims:
        _ = embed_text(claim)
        retrieved = retrieve_facts(claim, k=5)
        verification = verify_claim(claim, retrieved)
        if retrieved:
            retrieval_hits += 1
        verdict_confidences.append(float(verification["confidence"]))
        evidence = [
            {
                "id": r.fact.id,
                "claim": r.fact.claim,
                "language": r.fact.language,
                "score": float(r.score),
                "vector_score": float(getattr(r, "vector_score", r.score)),
                "bm25_score": float(getattr(r, "bm25_score", 0.0)),
                "rerank_score": float(getattr(r, "rerank_score", r.score)),
            }
            for r in retrieved
        ]

        results.append(
            {
                "claim": claim,
                "verdict": verification["verdict"],
                "confidence": float(verification["confidence"]),
                "evidence": evidence,
            }
        )

    n_claims = len(results)
    avg_conf = sum(verdict_confidences) / max(len(verdict_confidences), 1)
    stage_metrics = {
        "claim_detection": {"n_claims": n_claims},
        "retrieval": {"hit_rate": retrieval_hits / max(n_claims, 1)},
        "verification": {"avg_confidence": avg_conf},
    }
    return {"claims": results, "stage_metrics": stage_metrics}
