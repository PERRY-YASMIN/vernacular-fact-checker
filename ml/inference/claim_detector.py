from __future__ import annotations

from functools import lru_cache
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml import config
from .fluff_filter import clean_text, split_sentences, transliterate_if_needed

VERB_HINTS = [
    " is ",
    " are ",
    " was ",
    " were ",
    " has ",
    " have ",
    " था ",
    " थी ",
    " थे ",
    " है ",
    " हैं ",
    " होगा ",
    " रही ",
    " रहे ",
]


def _is_potential_claim(sentence: str) -> bool:
    s = f" {sentence.strip()} "
    if len(s) < 20:
        return False
    if any(ch.isdigit() for ch in s):
        return True
    return any(hint in s for hint in VERB_HINTS)


@lru_cache(maxsize=1)
def _load_claim_detector_model():
    model_path = config.CLAIM_DETECTOR_MODEL_PATH
    model_name = str(model_path) if model_path.exists() else config.CLAIM_DETECTOR_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = config.get_device()
    if device == "cuda":
        model = model.to(torch.device("cuda"))
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def _score_claim_probability(sentence: str) -> float:
    tokenizer, model = _load_claim_detector_model()
    device = config.get_device()

    enc = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
    if device == "cuda":
        enc = {k: v.to(torch.device("cuda")) for k, v in enc.items()}
    logits = model(**enc).logits

    if logits.shape[-1] == 1:
        prob = torch.sigmoid(logits[0, 0]).item()
        return float(prob)

    probs = torch.softmax(logits, dim=-1)[0]
    # Assume binary classifier where label-1 corresponds to "claim".
    claim_idx = 1 if probs.numel() > 1 else 0
    return float(probs[claim_idx].item())


def extract_claims(text: str) -> List[str]:
    """Extract sentence-level claim candidates from raw post text."""
    normalized = transliterate_if_needed(clean_text(text))
    sentences = split_sentences(normalized)

    use_model = config.CLAIM_DETECTOR_MODEL_PATH.exists()
    claims: List[str] = []
    for sentence in sentences:
        # Use transformer model only after it has been fine-tuned and saved locally.
        keep = False
        if use_model:
            try:
                score = _score_claim_probability(sentence)
                keep = score >= config.CLAIM_DETECTOR_THRESHOLD
            except Exception:
                keep = _is_potential_claim(sentence)
        else:
            keep = _is_potential_claim(sentence)
        if keep:
            claims.append(sentence)

    if not claims and normalized:
        claims.append(normalized)
    return claims
