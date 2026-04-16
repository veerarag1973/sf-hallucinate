"""Heuristic contradiction detection between claims and reference sentences.

Provides lexical heuristics to detect obvious contradictions without
requiring an LLM:

1. **Negation asymmetry** — one text has a negation where the other does not.
2. **Antonym pairs** — known antonym pairs appear across claim and reference.
3. **Numeric discrepancy** — different numbers in otherwise similar sentences.

These heuristics are intentionally conservative: false negatives are
acceptable (the LLM-NLI backend provides higher recall), but false
positives should be rare.
"""
from __future__ import annotations

import re
from typing import Tuple

from sf_hallucinate.scoring.languages import tokenize

# ---------------------------------------------------------------------------
# Negation word lists (English-centric; other languages use these too since
# the claim/reference are expected to be in the same language)
# ---------------------------------------------------------------------------

_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "none", "cannot",
})

_NEGATION_CONTRACTIONS: frozenset[str] = frozenset({
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "couldn't", "shouldn't", "can't", "hasn't",
    "haven't", "hadn't", "mustn't", "needn't", "shan't", "mightn't",
    # Expanded forms (after lowering, apostrophes stripped)
    "isnt", "arent", "wasnt", "werent", "dont", "doesnt", "didnt",
    "wont", "wouldnt", "couldnt", "shouldnt", "cant", "hasnt",
    "havent", "hadnt", "mustnt", "neednt", "shant", "mightnt",
})

_ALL_NEGATIONS: frozenset[str] = _NEGATION_WORDS | _NEGATION_CONTRACTIONS

# ---------------------------------------------------------------------------
# Antonym pairs
# ---------------------------------------------------------------------------

_ANTONYM_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"increase", "decrease"}),
    frozenset({"true", "false"}),
    frozenset({"correct", "incorrect"}),
    frozenset({"possible", "impossible"}),
    frozenset({"agree", "disagree"}),
    frozenset({"legal", "illegal"}),
    frozenset({"safe", "unsafe"}),
    frozenset({"safe", "dangerous"}),
    frozenset({"known", "unknown"}),
    frozenset({"present", "absent"}),
    frozenset({"success", "failure"}),
    frozenset({"accept", "reject"}),
    frozenset({"allow", "deny"}),
    frozenset({"allow", "forbid"}),
    frozenset({"always", "never"}),
    frozenset({"before", "after"}),
    frozenset({"begin", "end"}),
    frozenset({"better", "worse"}),
    frozenset({"big", "small"}),
    frozenset({"large", "small"}),
    frozenset({"buy", "sell"}),
    frozenset({"open", "close"}),
    frozenset({"open", "closed"}),
    frozenset({"gain", "loss"}),
    frozenset({"high", "low"}),
    frozenset({"rise", "fall"}),
    frozenset({"positive", "negative"}),
    frozenset({"include", "exclude"}),
    frozenset({"internal", "external"}),
    frozenset({"maximum", "minimum"}),
    frozenset({"majority", "minority"}),
    frozenset({"win", "lose"}),
    frozenset({"alive", "dead"}),
    frozenset({"appear", "disappear"}),
    frozenset({"guilty", "innocent"}),
})

# Pre-compute a lookup dict for fast antonym checking
_ANTONYM_LOOKUP: dict[str, set[str]] = {}
for _pair in _ANTONYM_PAIRS:
    _words = list(_pair)
    for _w in _words:
        _ANTONYM_LOOKUP.setdefault(_w, set()).update(_words)
        _ANTONYM_LOOKUP[_w].discard(_w)

# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"""
    (?:^|(?<=\s))               # start of string or whitespace
    -?                          # optional negative sign
    \d{1,3}(?:,\d{3})*         # integer with optional comma grouping
    (?:\.\d+)?                  # optional decimal part
    (?:%|                       # percent
       \s*(?:million|billion|trillion|thousand|hundred|
            k\b|m\b|b\b))?     # magnitude suffix
    """,
    re.IGNORECASE | re.VERBOSE,
)

_MAGNITUDE: dict[str, float] = {
    "hundred": 100,
    "thousand": 1_000,
    "k": 1_000,
    "million": 1_000_000,
    "m": 1_000_000,
    "billion": 1_000_000_000,
    "b": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}


def _parse_number(match_str: str) -> float | None:
    """Attempt to parse a matched number string into a float."""
    s = match_str.strip().lower()
    # Strip percent
    s = s.rstrip("%").strip()
    # Check for magnitude suffix
    mag = 1.0
    for suffix, mult in _MAGNITUDE.items():
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
            mag = mult
            break
    # Remove commas
    s = s.replace(",", "")
    try:
        return float(s) * mag
    except ValueError:
        return None


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from *text*."""
    nums: list[float] = []
    for m in _NUMBER_RE.finditer(text):
        val = _parse_number(m.group())
        if val is not None:
            nums.append(val)
    return nums


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _has_negation(tokens: list[str]) -> bool:
    """Check if any token is a negation word."""
    return bool(set(tokens) & _ALL_NEGATIONS)


def _check_negation_asymmetry(
    claim_tokens: list[str],
    ref_tokens: list[str],
) -> bool:
    """``True`` when exactly one side has a negation."""
    return _has_negation(claim_tokens) != _has_negation(ref_tokens)


def _check_antonyms(
    claim_tokens: set[str],
    ref_tokens: set[str],
) -> bool:
    """``True`` when an antonym pair straddles claim and reference."""
    for token in claim_tokens:
        antonyms = _ANTONYM_LOOKUP.get(token)
        if antonyms and antonyms & ref_tokens:
            return True
    return False


def _check_numeric_contradiction(claim: str, reference: str) -> bool:
    """``True`` when claim and reference contain conflicting numbers.

    Two numbers are considered conflicting when:
    - There is at least one number in each text.
    - The closest-magnitude pair differs by more than 10%.
    """
    claim_nums = _extract_numbers(claim)
    ref_nums = _extract_numbers(reference)
    if not claim_nums or not ref_nums:
        return False

    for cn in claim_nums:
        for rn in ref_nums:
            if cn == 0 and rn == 0:
                continue
            denom = max(abs(cn), abs(rn), 1e-9)
            if abs(cn - rn) / denom > 0.10:
                return True
    return False


def detect_contradiction(
    claim: str,
    reference_sentence: str,
    *,
    language: str = "en",
) -> Tuple[bool, float]:
    """Detect whether *claim* contradicts *reference_sentence*.

    Parameters
    ----------
    claim:
        A factual claim extracted from LLM output.
    reference_sentence:
        The best-matching sentence from the reference document.
    language:
        Language code for tokenisation.

    Returns
    -------
    tuple[bool, float]
        ``(is_contradiction, contradiction_score)`` where
        *contradiction_score* is in [0.0, 1.0].  Higher values indicate
        stronger evidence of contradiction.
    """
    claim_tokens = tokenize(claim, language=language, remove_stop_words=False)
    ref_tokens = tokenize(reference_sentence, language=language, remove_stop_words=False)

    claim_set = set(claim_tokens)
    ref_set = set(ref_tokens)

    signals: list[float] = []

    # Signal 1: Negation asymmetry (strong signal)
    if _check_negation_asymmetry(claim_tokens, ref_tokens):
        # Only flag if the sentences share substantial vocabulary
        overlap = len(claim_set & ref_set) / max(len(claim_set | ref_set), 1)
        if overlap > 0.2:
            signals.append(0.8)

    # Signal 2: Antonym pairs (moderate signal)
    if _check_antonyms(claim_set, ref_set):
        signals.append(0.6)

    # Signal 3: Numeric discrepancy (moderate signal)
    if _check_numeric_contradiction(claim, reference_sentence):
        signals.append(0.7)

    if not signals:
        return False, 0.0

    score = min(1.0, max(signals))
    return True, round(score, 4)
