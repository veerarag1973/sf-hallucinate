"""Multi-language tokenisation and stop-word filtering.

Supports ten languages: ``en``, ``es``, ``fr``, ``de``, ``pt``, ``ru``,
``zh``, ``ja``, ``ko``, ``ar``.

For Latin-script languages, tokenisation is whitespace-based with
punctuation removal and optional stop-word filtering.  For CJK scripts
(Chinese, Japanese, Korean), character bigrams are used since word
boundaries are not marked by spaces.
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# CJK character detection
# ---------------------------------------------------------------------------

_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),     # CJK Unified Ideographs
    (0x3400, 0x4DBF),     # CJK Extension A
    (0x3040, 0x309F),     # Hiragana
    (0x30A0, 0x30FF),     # Katakana
    (0xAC00, 0xD7AF),     # Hangul Syllables
    (0xF900, 0xFAFF),     # CJK Compatibility Ideographs
)


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def is_cjk_language(language: str) -> bool:
    """Return ``True`` if *language* uses CJK script."""
    return language in ("zh", "ja", "ko")


# ---------------------------------------------------------------------------
# Punctuation & normalisation
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _normalise(text: str) -> str:
    """Unicode NFC normalisation and case folding."""
    return unicodedata.normalize("NFC", text).lower()


# ---------------------------------------------------------------------------
# Stop-word dictionaries
# ---------------------------------------------------------------------------

STOP_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset({
        "a", "an", "the", "is", "it", "its", "in", "on", "at", "to", "of",
        "for", "and", "or", "but", "with", "as", "by", "from", "that",
        "this", "was", "are", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "shall", "can", "not", "no", "so", "than", "very",
        "just", "also", "about", "which", "who", "what", "when", "where",
        "how", "all", "each", "both", "more", "other", "some", "such",
        "only", "same", "into", "through", "during", "before", "after",
    }),
    "es": frozenset({
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
        "al", "y", "en", "que", "es", "por", "con", "para", "como", "más",
        "pero", "su", "sus", "le", "se", "lo", "ya", "era", "ser", "está",
        "fue", "son", "no", "si", "muy", "este", "esta", "estos", "estas",
        "todo", "todos", "otra", "otro", "hay", "tan", "entre", "sin",
    }),
    "fr": frozenset({
        "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
        "et", "en", "que", "qui", "est", "dans", "pour", "pas", "sur",
        "par", "avec", "ce", "cette", "ces", "son", "sa", "ses", "il",
        "elle", "nous", "vous", "ils", "elles", "on", "ne", "se", "si",
        "mais", "ou", "où", "plus", "tout", "très", "aussi", "même",
    }),
    "de": frozenset({
        "der", "die", "das", "ein", "eine", "und", "ist", "in", "von",
        "zu", "den", "mit", "auf", "für", "an", "es", "dem", "nicht",
        "als", "auch", "er", "sie", "dass", "nach", "wird", "bei", "einer",
        "um", "am", "sind", "noch", "wie", "einem", "über", "so", "zum",
        "war", "hat", "nur", "oder", "aber", "vor", "zur", "bis", "mehr",
    }),
    "pt": frozenset({
        "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das",
        "em", "no", "na", "nos", "nas", "por", "para", "com", "que", "se",
        "não", "mais", "como", "mas", "ao", "ele", "ela", "seu", "sua",
        "ou", "quando", "muito", "também", "foi", "são", "está", "ser",
        "tem", "já", "entre", "depois", "sem", "mesmo", "aos", "seus",
    }),
    "ru": frozenset({
        "и", "в", "не", "на", "я", "что", "он", "с", "это", "а", "то",
        "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
        "вы", "за", "бы", "по", "только", "её", "мне", "было", "вот",
        "от", "меня", "ещё", "нет", "о", "из", "ему", "теперь", "когда",
        "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть",
    }),
    "zh": frozenset({
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
        "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
        "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
    }),
    "ja": frozenset({
        "の", "に", "は", "を", "た", "が", "で", "て", "と", "し",
        "れ", "さ", "ある", "いる", "も", "する", "から", "な", "こと",
        "として", "い", "や", "れる", "など", "なっ", "ない", "この",
        "ため", "その", "あっ", "よう", "また", "もの", "という", "あり",
    }),
    "ko": frozenset({
        "이", "그", "저", "것", "수", "를", "에", "의", "는", "은",
        "가", "로", "하", "고", "도", "를", "만", "다", "에서", "와",
        "한", "있", "들", "그리고", "또는", "하지만", "그러나", "때문에",
    }),
    "ar": frozenset({
        "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "التي",
        "الذي", "هو", "هي", "كان", "قد", "لا", "ما", "أن", "بين",
        "كل", "ذلك", "بعد", "عند", "لم", "إذا", "حتى", "أو", "ثم",
        "هل", "لكن", "أي", "فقط", "غير", "نحو", "منذ", "خلال", "حول",
    }),
}

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(sorted(STOP_WORDS.keys()))


# ---------------------------------------------------------------------------
# Tokenisers
# ---------------------------------------------------------------------------

def _tokenize_cjk(text: str) -> list[str]:
    """Character bigram tokenisation for CJK text.

    Returns overlapping bigrams of CJK characters.  Non-CJK tokens
    (numbers, Latin fragments) are kept as whole words.
    """
    normalised = _normalise(text)
    normalised = _PUNCT.sub(" ", normalised)

    cjk_buf: list[str] = []
    tokens: list[str] = []

    for ch in normalised:
        if _is_cjk_char(ch):
            cjk_buf.append(ch)
        else:
            # Flush CJK bigrams
            if cjk_buf:
                for i in range(len(cjk_buf) - 1):
                    tokens.append(cjk_buf[i] + cjk_buf[i + 1])
                if len(cjk_buf) == 1:
                    tokens.append(cjk_buf[0])
                cjk_buf = []
            # Non-CJK: treat as Latin
            if ch.strip():
                # Will be collected below from whitespace split
                pass

    # Final CJK flush
    if cjk_buf:
        for i in range(len(cjk_buf) - 1):
            tokens.append(cjk_buf[i] + cjk_buf[i + 1])
        if len(cjk_buf) == 1:
            tokens.append(cjk_buf[0])

    # Also add Latin-style tokens
    latin_tokens = _PUNCT.sub(" ", normalised).split()
    for t in latin_tokens:
        if t and not any(_is_cjk_char(c) for c in t):
            tokens.append(t)

    return tokens


def _tokenize_latin(text: str) -> list[str]:
    """Space-based tokenisation for Latin, Cyrillic, Arabic scripts."""
    normalised = _normalise(text)
    return _PUNCT.sub(" ", normalised).split()


def tokenize(
    text: str,
    *,
    language: str = "en",
    remove_stop_words: bool = False,
) -> list[str]:
    """Tokenise *text* for the given *language*.

    Parameters
    ----------
    text:
        Input text.
    language:
        ISO 639-1 language code.  Default ``"en"``.
    remove_stop_words:
        When ``True``, remove language-specific stop words.

    Returns
    -------
    list[str]
        Lowercased tokens.  Empty strings and single characters (for Latin
        scripts) are filtered out.
    """
    if is_cjk_language(language):
        tokens = _tokenize_cjk(text)
    else:
        tokens = _tokenize_latin(text)

    # Filter short tokens for non-CJK
    if not is_cjk_language(language):
        tokens = [t for t in tokens if len(t) > 1]

    if remove_stop_words:
        stops = STOP_WORDS.get(language, STOP_WORDS["en"])
        tokens = [t for t in tokens if t not in stops]

    return tokens
