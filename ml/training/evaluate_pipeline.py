from __future__ import annotations

import json
from pathlib import Path

from ml.inference.evaluation import (
    evaluate_claim_detection,
    evaluate_end_to_end,
    evaluate_retrieval,
    evaluate_verification,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Offline evaluation driver; replace with real eval data wiring."""
    claim_metrics = evaluate_claim_detection([1, 1, 0, 1], [1, 0, 0, 1])
    retrieval_metrics = evaluate_retrieval(
        relevant_ids=[{"fact1"}, {"fact2", "fact3"}],
        retrieved_ids=[["fact1", "fact8"], ["fact5", "fact2", "fact3"]],
        k=3,
    )
    verification_metrics = evaluate_verification(
        ["Supported", "Refuted", "NotEnoughEvidence"],
        ["Supported", "Refuted", "Refuted"],
    )

    report = evaluate_end_to_end(claim_metrics, retrieval_metrics, verification_metrics)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
