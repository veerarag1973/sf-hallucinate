"""Tests for sf_hallucinate.scoring.overlap (token F1, bigram F1, Jaccard)."""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sf_hallucinate.scoring.overlap import bigram_f1, jaccard, token_f1, tokenize


class TestTokenize:
    def test_lowercases(self) -> None:
        assert "hello" in tokenize("Hello World")

    def test_removes_punctuation(self) -> None:
        tokens = tokenize("Hello, world!")
        assert "," not in tokens
        assert "!" not in tokens

    def test_removes_stopwords_by_default(self) -> None:
        tokens = tokenize("the sky is blue")
        # "the", "is" are stop words
        assert "the" not in tokens
        assert "is" not in tokens
        assert "sky" in tokens
        assert "blue" in tokens

    def test_keeps_stopwords_when_disabled(self) -> None:
        tokens = tokenize("the sky is blue", remove_stopwords=False)
        assert "the" in tokens
        assert "is" in tokens

    def test_empty_returns_empty(self) -> None:
        assert tokenize("") == []

    def test_only_stopwords_returns_empty(self) -> None:
        # All tokens are stop words
        tokens = tokenize("the is a")
        assert tokens == []


class TestTokenF1:
    def test_identical_strings(self) -> None:
        score = token_f1("Paris France capital city", "Paris France capital city")
        assert score == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self) -> None:
        score = token_f1("alpha beta gamma", "delta epsilon zeta")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        score = token_f1("Paris capital France", "France capital Paris city")
        assert 0.0 < score <= 1.0

    def test_empty_strings_return_zero(self) -> None:
        assert token_f1("", "some text") == pytest.approx(0.0)
        assert token_f1("some text", "") == pytest.approx(0.0)
        assert token_f1("", "") == pytest.approx(0.0)

    def test_symmetric_approximately(self) -> None:
        a = "Paris is the capital of France"
        b = "France capital Paris city"
        # F1 is symmetric by definition
        assert token_f1(a, b) == pytest.approx(token_f1(b, a))

    def test_result_in_unit_interval(self) -> None:
        score = token_f1(
            "The Eiffel Tower is in Paris",
            "The tower is in France",
        )
        assert 0.0 <= score <= 1.0

    def test_squad_style_example(self) -> None:
        # Classic SQuAD token F1 verification
        score = token_f1("Paris is in France", "France's capital is Paris")
        assert score > 0.3  # moderate overlap

    @given(st.text(min_size=1, max_size=200), st.text(min_size=1, max_size=200))
    @settings(max_examples=100)
    def test_always_in_unit_interval(self, a: str, b: str) -> None:
        score = token_f1(a, b)
        assert 0.0 <= score <= 1.0


class TestBigramF1:
    def test_identical_strings(self) -> None:
        score = bigram_f1("Paris France capital", "Paris France capital")
        assert score == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self) -> None:
        score = bigram_f1("alpha beta gamma", "delta epsilon zeta")
        assert score == pytest.approx(0.0)

    def test_partial_bigram_overlap(self) -> None:
        score = bigram_f1("Paris France capital city", "France capital Paris")
        # Bigrams: (Paris, France), (France, capital), (capital, city) vs
        #          (France, capital), (capital, Paris)
        assert 0.0 <= score <= 1.0

    def test_empty_returns_zero(self) -> None:
        assert bigram_f1("", "text") == pytest.approx(0.0)
        assert bigram_f1("word", "") == pytest.approx(0.0)

    def test_single_word_returns_zero(self) -> None:
        # Single word → no bigrams
        assert bigram_f1("Paris", "France") == pytest.approx(0.0)

    @given(st.text(min_size=0, max_size=200), st.text(min_size=0, max_size=200))
    @settings(max_examples=100)
    def test_always_in_unit_interval(self, a: str, b: str) -> None:
        score = bigram_f1(a, b)
        assert 0.0 <= score <= 1.0


class TestJaccard:
    def test_identical_sets(self) -> None:
        score = jaccard("Paris France tower", "Paris France tower")
        assert score == pytest.approx(1.0)

    def test_disjoint_sets_return_zero(self) -> None:
        score = jaccard("alpha beta gamma", "delta epsilon zeta")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        score = jaccard("Paris France", "Paris Germany")
        # |{Paris} ∩ {Paris, Germany}| / |{Paris, France, Germany}| = 1/3 ≈ 0.33
        assert 0.0 < score < 1.0

    def test_empty_returns_zero(self) -> None:
        # Both sets empty after stop-word removal
        assert jaccard("the a is", "is the a") == pytest.approx(0.0)

    @given(st.text(min_size=0, max_size=200), st.text(min_size=0, max_size=200))
    @settings(max_examples=100)
    def test_always_in_unit_interval(self, a: str, b: str) -> None:
        score = jaccard(a, b)
        assert 0.0 <= score <= 1.0
