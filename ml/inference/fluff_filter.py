from __future__ import annotations

import re
import unicodedata
from typing import List

import regex as reg
from unidecode import unidecode

FLUFF_PATTERNS = [
    r"\bplease\s+share\b",
    r"\bshare\s+this\b",
    r"\bforward\s+this\b",
    r"\bज्यादा से ज्यादा लोगों तक\b",
    r"\bसबको बताएं\b",
    r"\blike\s+and\s+share\b",
    r"\bsubscribe\s+to\s+my\b",
]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
WHITESPACE_RE = re.compile(r"\s+")
EMOJI_RE = reg.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}")
PUNCT_RE = re.compile(r"[!?]{2,}")
CONTROL_RE = re.compile(r"[\u200b-\u200f\ufeff]")

INDIC_DIGIT_MAP = str.maketrans(
    {
        "०": "0",
        "१": "1",
        "२": "2",
        "३": "3",
        "४": "4",
        "५": "5",
        "६": "6",
        "७": "7",
        "८": "8",
        "९": "9",
        "০": "0",
        "১": "1",
        "২": "2",
        "৩": "3",
        "৪": "4",
        "৫": "5",
        "৬": "6",
        "৭": "7",
        "৮": "8",
        "৯": "9",
    }
)


def _basic_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.strip())
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = CONTROL_RE.sub(" ", text)
    text = text.translate(INDIC_DIGIT_MAP)
    text = PUNCT_RE.sub("!", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _remove_fluff(text: str) -> str:
    lowered = text.lower()
    for pat in FLUFF_PATTERNS:
        lowered = re.sub(pat, " ", lowered)
    lowered = WHITESPACE_RE.sub(" ", lowered)
    return lowered.strip()


def clean_text(text: str) -> str:
    """Normalize user text before claim extraction and retrieval."""
    out = _basic_normalize(text)
    out = _remove_fluff(out)
    if all(ord(ch) < 128 for ch in out):
        out = unidecode(out)
    return out.strip()


def transliterate_if_needed(text: str) -> str:
    """Best-effort transliteration fallback for mixed/noisy social media text."""
    # Skip transliteration when text is mostly non-Latin; preserve native script semantics.
    if not text:
        return text
    latin_ratio = sum(1 for ch in text if "a" <= ch.lower() <= "z") / max(len(text), 1)
    if latin_ratio < 0.4:
        return text
    return unidecode(text)


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"[.!?।]+", text)
    return [p.strip() for p in parts if p.strip()]
