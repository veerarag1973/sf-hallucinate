"""Sentence-embedding similarity backend.

Uses `sentence-transformers <https://www.sbert.net/>`_ for dense vector
cosine similarity.  Requires the optional ``embedding`` extra::

    pip install sf-hallucinate[embedding]

The embedding model is configurable via :attr:`EvalConfig.embedding_model`
(default ``"all-MiniLM-L6-v2"``).
"""
from __future__ import annotations

import math
from typing import Any

from sf_hallucinate._types import EvalConfig
from sf_hallucinate.scoring.backends import ClaimScore
from sf_hallucinate.scoring.contradiction import detect_contradiction


class EmbeddingBackend:
    """Dense cosine similarity using sentence-transformer embeddings."""

    def __init__(self, config: EvalConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'embedding' similarity backend requires sentence-transformers. "
                "Install with: pip install sf-hallucinate[embedding]"
            ) from exc

        self._model: Any = SentenceTransformer(config.embedding_model)
        self._language = config.language
        self._detect_contradictions = config.detect_contradictions

    def score_claim(
        self,
        claim: str,
        reference_sentences: list[str],
    ) -> ClaimScore:
        """Score *claim* against *reference_sentences* using embedding cosine."""
        if not reference_sentences:
            return ClaimScore(similarity=0.0, best_match="")

        # Encode claim + all references in one batch for efficiency
        texts = [claim] + reference_sentences
        embeddings = self._model.encode(texts, convert_to_numpy=True)

        claim_emb = embeddings[0]
        ref_embs = embeddings[1:]

        best_sim = -1.0
        best_idx = 0
        for i, ref_emb in enumerate(ref_embs):
            sim = self._cosine_similarity(claim_emb, ref_emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        # Normalise to [0, 1]
        best_sim = max(0.0, min(1.0, best_sim))
        best_match = reference_sentences[best_idx]

        # Contradiction detection (heuristic fallback)
        contradiction = False
        c_score = 0.0
        if self._detect_contradictions and best_match:
            contradiction, c_score = detect_contradiction(
                claim, best_match, language=self._language
            )

        if contradiction and c_score >= 0.6:
            best_sim = min(best_sim, 0.15)

        return ClaimScore(
            similarity=round(best_sim, 6),
            best_match=best_match,
            contradiction_detected=contradiction,
            contradiction_score=c_score,
        )

    @staticmethod
    def _cosine_similarity(a: Any, b: Any) -> float:
        """Cosine similarity between two numpy vectors."""
        dot = float(sum(ai * bi for ai, bi in zip(a, b)))
        mag_a = math.sqrt(float(sum(ai * ai for ai in a)))
        mag_b = math.sqrt(float(sum(bi * bi for bi in b)))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)
