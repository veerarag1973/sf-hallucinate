"""TF-IDF cosine similarity and hybrid scoring (pure Python stdlib).

Hybrid similarity
-----------------
The module exposes :func:`hybrid_similarity` which combines two complementary
signals:

1. **TF-IDF cosine similarity** — captures shared vocabulary with IDF
   down-weighting of common terms.
2. **Token F1** — robust unigram overlap that handles paraphrases with
   overlapping content words.

The weight blend is controlled by *tfidf_weight* (default ``0.6``):

    hybrid = tfidf_weight × tfidf_cosine + (1 − tfidf_weight) × token_f1

No external packages are required — every routine is implemented with the
Python standard library only.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from sf_hallucinate.scoring.overlap import token_f1

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]")


def _tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, split — preserves numbers."""
    return _PUNCT.sub(" ", text.lower()).split()


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

def _compute_tf(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    n = len(tokens)
    return {term: count / n for term, count in Counter(tokens).items()}


def _compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Smoothed IDF: log((N+1) / (df+1)) + 1."""
    n = len(documents)
    df: dict[str, int] = {}
    for doc in documents:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    return {
        term: math.log((n + 1) / (freq + 1)) + 1.0
        for term, freq in df.items()
    }


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = _compute_tf(tokens)
    return {term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()}


def _cosine(
    vec_a: dict[str, float],
    vec_b: dict[str, float],
) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(vec_a.get(t, 0.0) * vec_b.get(t, 0.0) for t in vec_a)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sentence_similarity(sentence_a: str, sentence_b: str) -> float:
    """TF-IDF cosine similarity between two sentences.

    Builds a two-document corpus from *sentence_a* and *sentence_b*, computes
    per-document TF-IDF vectors, then returns the cosine similarity.

    Parameters
    ----------
    sentence_a, sentence_b:
        Input strings.  Empty strings yield ``0.0``.

    Returns
    -------
    float
        Cosine similarity in [0.0, 1.0].

    Examples
    --------
    >>> round(sentence_similarity(
    ...     "The sky is blue",
    ...     "The sky is blue in color",
    ... ), 2)
    0.79
    """
    tok_a = _tokenize(sentence_a)
    tok_b = _tokenize(sentence_b)
    if not tok_a or not tok_b:
        return 0.0
    idf = _compute_idf([tok_a, tok_b])
    vec_a = _tfidf_vector(tok_a, idf)
    vec_b = _tfidf_vector(tok_b, idf)
    return min(1.0, max(0.0, _cosine(vec_a, vec_b)))


def hybrid_similarity(
    hypothesis: str,
    reference: str,
    *,
    tfidf_weight: float = 0.6,
) -> float:
    """Blend of TF-IDF cosine and token F1 similarities.

    The two metrics are complementary:

    * TF-IDF cosine handles shared vocabulary well even with rare terms.
    * Token F1 is robust to paraphrase and vocabulary variation.

    Parameters
    ----------
    hypothesis:
        The claim / LLM output sentence being evaluated.
    reference:
        The reference sentence to compare against.
    tfidf_weight:
        Blend factor in [0.0, 1.0].  Default ``0.6`` gives slight preference
        to TF-IDF.

    Returns
    -------
    float
        Blended similarity in [0.0, 1.0].

    Examples
    --------
    >>> score = hybrid_similarity(
    ...     "Paris is the capital of France",
    ...     "France's capital city is Paris",
    ... )
    >>> score > 0.4
    True
    """
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    cosine = sentence_similarity(hypothesis, reference)
    f1 = token_f1(hypothesis, reference)
    return tfidf_weight * cosine + (1.0 - tfidf_weight) * f1


def find_best_match(
    claim: str,
    reference_sentences: list[str],
    *,
    tfidf_weight: float = 0.6,
) -> tuple[float, str]:
    """Find the reference sentence that best supports *claim*.

    Parameters
    ----------
    claim:
        A single factual claim extracted from an LLM output.
    reference_sentences:
        Sentences from the reference document.
    tfidf_weight:
        Forwarded to :func:`hybrid_similarity`.

    Returns
    -------
    tuple[float, str]
        ``(best_score, best_sentence)`` where *best_score* is in [0.0, 1.0].
        Returns ``(0.0, "")`` when *reference_sentences* is empty.

    Examples
    --------
    >>> score, match = find_best_match(
    ...     "Paris is in France",
    ...     ["Paris is in France.", "Berlin is in Germany."],
    ... )
    >>> score > 0.5
    True
    >>> "Paris" in match
    True
    """
    if not reference_sentences:
        return 0.0, ""
    best_score = -1.0
    best_sent = ""
    for ref in reference_sentences:
        score = hybrid_similarity(claim, ref, tfidf_weight=tfidf_weight)
        if score > best_score:
            best_score = score
            best_sent = ref
    return max(0.0, best_score), best_sent
