"""Scoring sub-package — similarity, overlap, claim extraction, and backends.

Exposed helpers (re-exported for convenience):

    from sf_hallucinate.scoring import (
        split_sentences,
        extract_claims,
        sentence_similarity,
        token_f1,
        hybrid_similarity,
        find_best_match,
        create_backend,
        detect_contradiction,
        tokenize,
    )
"""
from sf_hallucinate.scoring.backends import create_backend
from sf_hallucinate.scoring.claims import extract_claims, split_sentences
from sf_hallucinate.scoring.contradiction import detect_contradiction
from sf_hallucinate.scoring.languages import tokenize
from sf_hallucinate.scoring.overlap import token_f1
from sf_hallucinate.scoring.similarity import (
    find_best_match,
    hybrid_similarity,
    sentence_similarity,
)

__all__ = [
    "split_sentences",
    "extract_claims",
    "sentence_similarity",
    "token_f1",
    "hybrid_similarity",
    "find_best_match",
    "create_backend",
    "detect_contradiction",
    "tokenize",
]
