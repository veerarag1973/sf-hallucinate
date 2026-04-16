"""Sentence-level claim extraction from free-form text.

Design goals
------------
* Zero dependencies — pure Python stdlib only.
* Robust to abbreviations (Dr., e.g., U.S.A., …).
* Filters out questions, exclamations, and fragments too short to be claims.
* Handles bulleted / numbered lists by treating each item as a sentence.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Known abbreviations that should NOT trigger sentence splits
# ---------------------------------------------------------------------------
_ABBREVS: frozenset[str] = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "fig",
        "ref",
        "no",
        "vol",
        "approx",
        "dept",
        "est",
        "al",
        "inc",
        "corp",
        "ltd",
        "co",
        "st",
        "ave",
        "blvd",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    }
)

# Matches "word." at end of word — used to check for abbreviations
_TRAILING_DOT = re.compile(r"\b([A-Za-z]{1,5})\.$")

# Bullet / list prefix patterns  (-, *, •, 1., 2), a., …)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)]\s*|[a-z][.)]\s*)\s*", re.IGNORECASE)

# Pure meta / transitional phrases to skip (not factual claims)
_META_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:in summary|in conclusion|to summarise|to summarize|"
        r"here is|here are|the following|as follows|note that|"
        r"please note|importantly|for example|for instance|"
        r"as mentioned|as stated|as noted|see below|see above)",
        re.IGNORECASE,
    ),
)


def _is_abbreviation_boundary(text: str, dot_pos: int) -> bool:
    """Return True if the '.' at *dot_pos* is part of a known abbreviation."""
    before = text[:dot_pos]
    m = _TRAILING_DOT.search(before)
    if m and m.group(1).lower() in _ABBREVS:
        return True
    # Single uppercase letter (initials: "J. Smith")
    if m and len(m.group(1)) == 1 and m.group(1).isupper():
        return True
    # Decimal numbers  e.g. "3.14"
    if dot_pos > 0 and text[dot_pos - 1].isdigit():
        # check if next char is digit too
        if dot_pos + 1 < len(text) and text[dot_pos + 1].isdigit():
            return True
    return False


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences with basic abbreviation awareness.

    Parameters
    ----------
    text:
        Arbitrary free-form text (LLM output, reference document, …).

    Returns
    -------
    list[str]
        Non-empty sentence strings in document order.  Leading/trailing
        whitespace is stripped from each item.

    Examples
    --------
    >>> split_sentences("The sky is blue. It was always so.")
    ['The sky is blue.', 'It was always so.']
    >>> split_sentences("Dr. Smith said it was fine.  Really fine.")
    ['Dr. Smith said it was fine.', 'Really fine.']
    """
    if not text or not text.strip():
        return []

    # Pre-process: normalise line-endings and collapse blank lines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on blank lines first — paragraphs are always independent
    paragraphs = re.split(r"\n\s*\n", text)
    sentences: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Each line in a paragraph that starts with a list prefix is a
        # separate sentence.
        lines = para.split("\n")
        if all(_LIST_PREFIX.match(ln) for ln in lines if ln.strip()):
            for ln in lines:
                clean = _LIST_PREFIX.sub("", ln).strip()
                if clean:
                    sentences.append(clean)
            continue

        # Otherwise, use punctuation-based splitting with abbreviation guard
        buf: list[str] = []
        i = 0
        while i < len(para):
            ch = para[i]
            buf.append(ch)
            if ch in ".!?":
                # Look ahead
                rest = para[i + 1 :]
                # Is the next non-space char an uppercase letter or end?
                m = re.match(r"\s+([A-Z\"\'])", rest)
                if m or i == len(para) - 1:
                    if ch == "." and _is_abbreviation_boundary(para, i):
                        # Not a real sentence boundary
                        i += 1
                        continue
                    # Commit current buffer as a sentence
                    sent = "".join(buf).strip()
                    if sent:
                        sentences.append(sent)
                    buf = []
                    # Skip the whitespace that follows
                    if m:
                        i += m.start(1)  # rewind to the uppercase letter
            i += 1

        # Flush any remaining text
        remainder = "".join(buf).strip()
        if remainder:
            sentences.append(remainder)

    return [s.strip() for s in sentences if s.strip()]


def _is_meta_sentence(sentence: str) -> bool:
    """Return True when a sentence is a transitional / meta phrase."""
    return any(p.match(sentence) for p in _META_PATTERNS)


def extract_claims(text: str, *, min_length: int = 15) -> list[str]:
    """Extract factual claims from *text*.

    A *claim* is defined as a declarative sentence that:

    * is not a question (does not end with ``?``),
    * is at least *min_length* characters long,
    * is not a purely transitional / meta phrase (e.g. "In summary, …").

    Parameters
    ----------
    text:
        LLM output to analyse.
    min_length:
        Minimum character length (stripped) for a sentence to qualify as a
        claim.  Default ``15``.

    Returns
    -------
    list[str]
        Ordered list of claim strings.

    Examples
    --------
    >>> claims = extract_claims(
    ...     "The Eiffel Tower is in Paris. Is it tall? Yes."
    ... )
    >>> claims
    ['The Eiffel Tower is in Paris.']
    """
    sentences = split_sentences(text)
    claims: list[str] = []
    for sent in sentences:
        s = sent.strip()
        if len(s) < min_length:
            continue
        if s.endswith("?"):
            continue
        if _is_meta_sentence(s):
            continue
        claims.append(s)
    return claims
