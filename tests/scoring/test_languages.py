"""Tests for sf_hallucinate.scoring.languages — multi-language tokenisation."""
from __future__ import annotations

import pytest

from sf_hallucinate.scoring.languages import (
    STOP_WORDS,
    SUPPORTED_LANGUAGES,
    is_cjk_language,
    tokenize,
)


class TestSupportedLanguages:
    def test_all_languages_have_stop_words(self) -> None:
        for lang in SUPPORTED_LANGUAGES:
            assert lang in STOP_WORDS
            assert len(STOP_WORDS[lang]) > 10

    def test_supported_languages_sorted(self) -> None:
        assert list(SUPPORTED_LANGUAGES) == sorted(SUPPORTED_LANGUAGES)


class TestIsCjkLanguage:
    @pytest.mark.parametrize("lang", ["zh", "ja", "ko"])
    def test_cjk_languages(self, lang: str) -> None:
        assert is_cjk_language(lang) is True

    @pytest.mark.parametrize("lang", ["en", "es", "fr", "de", "ru", "ar"])
    def test_non_cjk_languages(self, lang: str) -> None:
        assert is_cjk_language(lang) is False


class TestTokenize:
    def test_english_basic(self) -> None:
        tokens = tokenize("The sky is blue.", language="en")
        assert "sky" in tokens
        assert "blue" in tokens

    def test_english_stop_word_removal(self) -> None:
        tokens = tokenize(
            "The sky is blue.", language="en", remove_stop_words=True
        )
        assert "the" not in tokens
        assert "is" not in tokens
        assert "sky" in tokens

    def test_english_no_single_chars(self) -> None:
        tokens = tokenize("I am a person.", language="en")
        # Single char "i" and "a" should be filtered
        assert all(len(t) > 1 for t in tokens)

    def test_spanish(self) -> None:
        tokens = tokenize(
            "El cielo es azul.", language="es", remove_stop_words=True
        )
        assert "cielo" in tokens
        assert "azul" in tokens
        assert "el" not in tokens

    def test_french(self) -> None:
        tokens = tokenize(
            "Le ciel est bleu.", language="fr", remove_stop_words=True
        )
        assert "ciel" in tokens
        assert "bleu" in tokens

    def test_german(self) -> None:
        tokens = tokenize(
            "Der Himmel ist blau.", language="de", remove_stop_words=True
        )
        assert "himmel" in tokens
        assert "blau" in tokens

    def test_chinese_bigrams(self) -> None:
        tokens = tokenize("今天天气很好", language="zh")
        # Should produce character bigrams
        assert len(tokens) > 0
        # Check a bigram exists
        assert "今天" in tokens

    def test_chinese_stop_words(self) -> None:
        tokens = tokenize(
            "我是一个人", language="zh", remove_stop_words=True
        )
        assert "我" not in [t for t in tokens if len(t) == 1]

    def test_japanese(self) -> None:
        tokens = tokenize("東京は日本の首都です", language="ja")
        assert len(tokens) > 0

    def test_russian(self) -> None:
        tokens = tokenize(
            "Небо голубое и красивое.", language="ru", remove_stop_words=True
        )
        assert "небо" in tokens
        assert "и" not in tokens

    def test_arabic(self) -> None:
        tokens = tokenize(
            "السماء زرقاء اليوم", language="ar", remove_stop_words=True
        )
        assert len(tokens) > 0

    def test_empty_text(self) -> None:
        assert tokenize("", language="en") == []

    def test_punctuation_removed(self) -> None:
        tokens = tokenize("Hello, world!", language="en")
        assert "hello" in tokens
        assert "world" in tokens

    def test_default_language_is_english(self) -> None:
        tokens = tokenize("Hello world")
        assert "hello" in tokens
