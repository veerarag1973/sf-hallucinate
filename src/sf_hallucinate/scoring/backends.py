"""Pluggable similarity backends for claim scoring.

Three built-in backends are provided:

* **hybrid** (default) — TF-IDF cosine + token F1 blend (zero dependencies).
* **embedding** — sentence-transformer dense cosine similarity.
* **llm-nli** — LLM-as-judge Natural Language Inference.

Custom backends must implement the :class:`SimilarityBackend` protocol.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Protocol, runtime_checkable, runtime_checkable

from sf_hallucinate._types import EvalConfig
from sf_hallucinate.scoring.contradiction import detect_contradiction
from sf_hallucinate.scoring.languages import (
    is_cjk_language,
    tokenize as lang_tokenize,
)
from sf_hallucinate.scoring.overlap import token_f1
from sf_hallucinate.scoring.similarity import find_best_match

# ---------------------------------------------------------------------------
# Intermediate claim score
# ---------------------------------------------------------------------------


class ClaimScore:
    """Result of scoring a single claim against reference sentences.

    Not a frozen dataclass for flexibility — created fresh per-claim.
    """

    __slots__ = (
        "similarity",
        "best_match",
        "entailment_label",
        "contradiction_detected",
        "contradiction_score",
        "confidence",
    )

    def __init__(
        self,
        *,
        similarity: float,
        best_match: str,
        entailment_label: str = "",
        contradiction_detected: bool = False,
        contradiction_score: float = 0.0,
        confidence: float = 1.0,
    ) -> None:
        self.similarity = similarity
        self.best_match = best_match
        self.entailment_label = entailment_label
        self.contradiction_detected = contradiction_detected
        self.contradiction_score = contradiction_score
        self.confidence = confidence


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
@runtime_checkable
class SimilarityBackend(Protocol):
    """Protocol for pluggable similarity backends."""

    def score_claim(
        self,
        claim: str,
        reference_sentences: list[str],
    ) -> ClaimScore:
        """Score a single claim against reference sentences."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Hybrid backend (default)
# ---------------------------------------------------------------------------


class HybridBackend:
    """TF-IDF cosine + token F1 hybrid with optional contradiction detection.

    For English this delegates to the optimised stdlib functions in
    :mod:`sf_hallucinate.scoring.similarity`.  For other languages it uses
    language-aware tokenisation from :mod:`sf_hallucinate.scoring.languages`.
    """

    def __init__(self, config: EvalConfig) -> None:
        self._tfidf_weight = config.tfidf_weight
        self._language = config.language
        self._detect_contradictions = config.detect_contradictions

    def score_claim(
        self,
        claim: str,
        reference_sentences: list[str],
    ) -> ClaimScore:
        if not reference_sentences:
            return ClaimScore(similarity=0.0, best_match="")

        if self._language == "en":
            # Fast path: existing optimised English implementation
            sim, best = find_best_match(
                claim, reference_sentences, tfidf_weight=self._tfidf_weight
            )
        else:
            sim, best = self._find_best_match_multilang(claim, reference_sentences)

        contradiction = False
        c_score = 0.0
        if self._detect_contradictions and best:
            contradiction, c_score = detect_contradiction(
                claim, best, language=self._language
            )

        # If contradiction detected with high confidence, penalise similarity
        if contradiction and c_score >= 0.6:
            sim = min(sim, 0.15)

        return ClaimScore(
            similarity=round(sim, 6),
            best_match=best,
            contradiction_detected=contradiction,
            contradiction_score=c_score,
        )

    # ------------------------------------------------------------------
    # Multi-language scoring
    # ------------------------------------------------------------------

    def _find_best_match_multilang(
        self,
        claim: str,
        reference_sentences: list[str],
    ) -> tuple[float, str]:
        best_score = 0.0
        best_sent = ""
        for sent in reference_sentences:
            score = self._hybrid_similarity_multilang(claim, sent)
            if score > best_score:
                best_score = score
                best_sent = sent
        return best_score, best_sent

    def _hybrid_similarity_multilang(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        cosine = self._tfidf_cosine(a, b)
        f1 = self._token_f1_multilang(a, b)
        return self._tfidf_weight * cosine + (1.0 - self._tfidf_weight) * f1

    def _tfidf_cosine(self, a: str, b: str) -> float:
        tok_a = lang_tokenize(a, language=self._language)
        tok_b = lang_tokenize(b, language=self._language)
        if not tok_a or not tok_b:
            return 0.0
        # IDF from the 2-doc corpus
        docs = [tok_a, tok_b]
        n = len(docs)
        df: dict[str, int] = {}
        for doc in docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        idf = {t: math.log((n + 1) / (f + 1)) + 1.0 for t, f in df.items()}
        # TF-IDF vectors
        vec_a = self._tfidf_vec(tok_a, idf)
        vec_b = self._tfidf_vec(tok_b, idf)
        return self._cosine(vec_a, vec_b)

    @staticmethod
    def _tfidf_vec(
        tokens: list[str], idf: dict[str, float]
    ) -> dict[str, float]:
        if not tokens:
            return {}
        n = len(tokens)
        tf = {t: c / n for t, c in Counter(tokens).items()}
        return {t: v * idf.get(t, 1.0) for t, v in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in a)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return min(1.0, max(0.0, dot / (mag_a * mag_b)))

    def _token_f1_multilang(self, hypothesis: str, reference: str) -> float:
        hyp_tokens = lang_tokenize(
            hypothesis, language=self._language, remove_stop_words=True
        )
        ref_tokens = lang_tokenize(
            reference, language=self._language, remove_stop_words=True
        )
        if not hyp_tokens or not ref_tokens:
            return 0.0
        hyp_set = set(hyp_tokens)
        ref_set = set(ref_tokens)
        common = hyp_set & ref_set
        if not common:
            return 0.0
        precision = len(common) / len(hyp_set)
        recall = len(common) / len(ref_set)
        return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_backend(config: EvalConfig) -> Any:
    """Create a :class:`SimilarityBackend` instance from *config*.

    Parameters
    ----------
    config:
        Evaluation config controlling backend selection.

    Returns
    -------
    SimilarityBackend
        Concrete backend instance.

    Raises
    ------
    ValueError
        For unknown ``similarity_backend`` values.
    """
    name = config.similarity_backend

    if name == "hybrid":
        return HybridBackend(config)

    if name == "embedding":
        from sf_hallucinate.scoring.embedding import EmbeddingBackend

        return EmbeddingBackend(config)

    if name == "llm-nli":
        from sf_hallucinate.scoring.nli import LLMNLIBackend

        return LLMNLIBackend(config)

    raise ValueError(f"Unknown similarity backend: {name!r}")
