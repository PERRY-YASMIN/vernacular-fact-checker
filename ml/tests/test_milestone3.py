import numpy as np

from ml.inference.evaluation import (
    evaluate_claim_detection,
    evaluate_end_to_end,
    evaluate_retrieval,
    evaluate_verification,
)
from ml.inference.fluff_filter import clean_text
from ml.inference.retrieval_pipeline import Fact, _build_bm25_stats, retrieve_for_claim


def test_multilingual_clean_text_normalizes_indic_digits_and_noise():
    text = "आज पेट्रोल १२३ रुपये है!!! कृपया सबको बताएं https://x.com"
    cleaned = clean_text(text)
    assert "123" in cleaned
    assert "https" not in cleaned


def test_hybrid_retrieval_returns_ranked_results(monkeypatch):
    facts = [
        Fact(id="f1", claim="petrol price in delhi is around 95 rupees", language="en"),
        Fact(id="f2", claim="lpg subsidy was reduced", language="en"),
    ]
    fact_embeddings = np.array([[0.9, 0.1], [0.1, 0.9]], dtype="float32")
    bm25_stats = _build_bm25_stats(facts)

    monkeypatch.setattr(
        "ml.inference.retrieval_pipeline.embed_texts",
        lambda texts: np.array([[1.0, 0.0]], dtype="float32"),
    )

    out = retrieve_for_claim(
        "petrol price in delhi",
        fact_embeddings=fact_embeddings,
        facts=facts,
        bm25_stats=bm25_stats,
        top_k=2,
    )
    assert len(out) >= 1
    assert out[0].fact.id == "f1"


def test_stage_metric_helpers():
    claim_m = evaluate_claim_detection([1, 1, 0], [1, 0, 0])
    retrieval_m = evaluate_retrieval([{"a"}], [["a", "b"]], k=2)
    verify_m = evaluate_verification(["Supported", "Refuted"], ["Supported", "Supported"])
    report = evaluate_end_to_end(claim_m, retrieval_m, verify_m)

    assert "claim_f1" in report
    assert "retrieval_hit_rate@2" in report
    assert "verification_accuracy" in report
