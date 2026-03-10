from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ml import config
from .embedder import embed_texts


@dataclass
class Fact:
    id: str
    claim: str
    language: str


@dataclass
class RetrievedFact:
    fact: Fact
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def load_facts(path: Path | None = None) -> List[Fact]:
    if path is None:
        path = config.VERIFIED_FACTS_PATH
    facts: List[Fact] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        facts.append(Fact(id=obj["id"], claim=obj["claim"], language=obj.get("language", "unk")))
    return facts


def build_fact_index(facts: List[Fact]) -> Tuple[np.ndarray, List[Fact]]:
    texts = [f.claim for f in facts]
    embeddings = embed_texts(texts)
    return embeddings, facts


def _build_bm25_stats(facts: List[Fact]) -> dict:
    docs = [_tokenize(f.claim) for f in facts]
    doc_freq: Dict[str, int] = {}
    for doc in docs:
        for token in set(doc):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    avgdl = float(sum(len(d) for d in docs) / max(len(docs), 1))
    return {"docs": docs, "doc_freq": doc_freq, "avgdl": avgdl}


@lru_cache(maxsize=1)
def _cached_index() -> Tuple[np.ndarray, List[Fact], dict]:
    facts = load_facts()
    embeddings, facts = build_fact_index(facts)
    bm25_stats = _build_bm25_stats(facts)
    return embeddings, facts, bm25_stats


def _bm25_scores(query: str, bm25_stats: dict) -> np.ndarray:
    docs: List[List[str]] = bm25_stats["docs"]
    doc_freq: Dict[str, int] = bm25_stats["doc_freq"]
    avgdl = bm25_stats["avgdl"]
    q_tokens = _tokenize(query)
    n_docs = len(docs)
    scores = np.zeros(n_docs, dtype="float32")

    for i, doc in enumerate(docs):
        if not doc:
            continue
        doc_len = len(doc)
        term_counts: Dict[str, int] = {}
        for token in doc:
            term_counts[token] = term_counts.get(token, 0) + 1

        score = 0.0
        for token in q_tokens:
            tf = term_counts.get(token, 0)
            if tf == 0:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + config.BM25_K1 * (1.0 - config.BM25_B + config.BM25_B * doc_len / max(avgdl, 1e-8))
            score += idf * ((tf * (config.BM25_K1 + 1.0)) / max(denom, 1e-8))
        scores[i] = float(score)

    return scores


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    mn = float(scores.min())
    mx = float(scores.max())
    if mx - mn < 1e-8:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def _rerank(claim_text: str, retrieved: List[RetrievedFact]) -> List[RetrievedFact]:
    q_tokens = set(_tokenize(claim_text))
    reranked: List[RetrievedFact] = []
    for item in retrieved:
        d_tokens = set(_tokenize(item.fact.claim))
        overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
        rerank_score = 0.8 * item.score + 0.2 * overlap
        reranked.append(
            RetrievedFact(
                fact=item.fact,
                score=float(rerank_score),
                vector_score=item.vector_score,
                bm25_score=item.bm25_score,
                rerank_score=float(rerank_score),
            )
        )
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked


def retrieve_for_claim(
    claim_text: str,
    fact_embeddings: np.ndarray,
    facts: List[Fact],
    bm25_stats: dict,
    top_k: int | None = None,
) -> List[RetrievedFact]:
    if top_k is None:
        top_k = config.TOP_K_FACTS

    claim_vec = embed_texts([claim_text])
    vector_scores = cosine_similarity(claim_vec, fact_embeddings)[0]
    bm25_scores = _bm25_scores(claim_text, bm25_stats)

    norm_vector = _normalize_scores(vector_scores)
    norm_bm25 = _normalize_scores(bm25_scores)
    hybrid_scores = config.HYBRID_ALPHA * norm_vector + (1.0 - config.HYBRID_ALPHA) * norm_bm25

    preselect = max(top_k, config.RERANK_TOP_N)
    idx_sorted = np.argsort(-hybrid_scores)[:preselect]

    results: List[RetrievedFact] = []
    for idx in idx_sorted:
        score = float(hybrid_scores[idx])
        if score < config.MIN_SIMILARITY and float(norm_vector[idx]) < config.MIN_SIMILARITY:
            continue
        results.append(
            RetrievedFact(
                fact=facts[idx],
                score=score,
                vector_score=float(vector_scores[idx]),
                bm25_score=float(bm25_scores[idx]),
            )
        )

    reranked = _rerank(claim_text, results)
    return reranked[:top_k]


def retrieve_facts(claim: str, k: int = 5) -> List[RetrievedFact]:
    """Retrieve top-k fact candidates for a claim."""
    fact_embeddings, facts, bm25_stats = _cached_index()
    return retrieve_for_claim(claim, fact_embeddings, facts, bm25_stats=bm25_stats, top_k=k)
