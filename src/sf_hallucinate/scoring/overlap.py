"""Token-overlap metrics (no external dependencies).

Implements:

* :func:`tokenize`        — lower-case, strip punctuation, split.
* :func:`token_f1`        — precision/recall/F1 over unigram bags.
* :func:`bigram_f1`       — precision/recall/F1 over bigram bags.
* :func:`jaccard`         — set-based Jaccard index.
"""
from __future__ import annotations

import re
from collections import Counter

# Stop words excluded from overlap calculations (English only).
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "it",
        "its",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "and",
        "or",
        "but",
        "with",
        "as",
        "by",
        "from",
        "that",
        "this",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "whether",
        "if",
        "then",
        "than",
        "such",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "each",
        "about",
        "also",
        "just",
        "however",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "most",
        "other",
        "more",
        "very",
        "only",
        "same",
        "own",
        "up",
        "out",
        "over",
        "under",
        "again",
        "there",
        "here",
        "they",
        "them",
        "their",
        "he",
        "she",
        "we",
        "us",
        "i",
        "me",
        "my",
        "your",
        "you",
        "his",
        "her",
    }
)

_PUNCT = re.compile(r"[^\w\s]")


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Lower-case, strip punctuation, split on whitespace.

    Parameters
    ----------
    text:
        Input string.
    remove_stopwords:
        When ``True`` (default), common English stop-words are removed before
        returning.

    Returns
    -------
    list[str]
        Filtered token list (may be empty).
    """
    text = _PUNCT.sub(" ", text.lower())
    tokens = text.split()
    if remove_stopwords:
        return [t for t in tokens if t not in _STOP_WORDS]
    return tokens


def _f1(counter_a: Counter[str], counter_b: Counter[str]) -> float:
    """Internal F1 helper over arbitrary Counter bags."""
    if not counter_a or not counter_b:
        return 0.0
    overlap = sum((counter_a & counter_b).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(counter_a.values())
    recall = overlap / sum(counter_b.values())
    return 2.0 * precision * recall / (precision + recall)


def token_f1(
    hypothesis: str,
    reference: str,
    *,
    remove_stopwords: bool = True,
) -> float:
    """Compute token-level F1 between *hypothesis* and *reference*.

    Uses unigram bag-of-words overlap (same as SQuAD token F1).  Stop words
    are excluded by default so content words drive the score.

    Parameters
    ----------
    hypothesis:
        The string being evaluated (LLM output / claim).
    reference:
        The gold-standard string (reference sentence).
    remove_stopwords:
        When ``True``, common English stop words are excluded.

    Returns
    -------
    float
        F1 score in [0.0, 1.0].  Returns 0.0 when either string is empty
        after tokenisation.

    Examples
    --------
    >>> round(token_f1("Paris is in France", "France's capital is Paris"), 2)
    0.67
    """
    h_tokens = Counter(tokenize(hypothesis, remove_stopwords=remove_stopwords))
    r_tokens = Counter(tokenize(reference, remove_stopwords=remove_stopwords))
    return _f1(h_tokens, r_tokens)


def bigram_f1(hypothesis: str, reference: str) -> float:
    """Compute bigram F1 between *hypothesis* and *reference*.

    Captures phrase-level matches that unigram F1 misses.

    Returns
    -------
    float
        F1 score in [0.0, 1.0].
    """
    def bigrams(tokens: list[str]) -> list[tuple[str, str]]:
        return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

    h_bg = Counter(bigrams(tokenize(hypothesis)))
    r_bg = Counter(bigrams(tokenize(reference)))
    return _f1(h_bg, r_bg)  # type: ignore[arg-type]


def jaccard(text_a: str, text_b: str, *, remove_stopwords: bool = True) -> float:
    """Set-based Jaccard similarity between the token sets of two strings.

    Returns
    -------
    float
        |A ∩ B| / |A ∪ B|, in [0.0, 1.0].  Returns 0.0 when both sets are
        empty.
    """
    set_a = set(tokenize(text_a, remove_stopwords=remove_stopwords))
    set_b = set(tokenize(text_b, remove_stopwords=remove_stopwords))
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
