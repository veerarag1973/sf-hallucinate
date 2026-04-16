"""Tests for sf_hallucinate.scoring.similarity (TF-IDF + hybrid)."""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sf_hallucinate.scoring.similarity import (
    _cosine,
    _tokenize,
    find_best_match,
    hybrid_similarity,
    sentence_similarity,
)


class TestTokenize:
    def test_lowercases(self) -> None:
        assert _tokenize("Hello WORLD") == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        tokens = _tokenize("Hello, world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_preserves_numbers(self) -> None:
        tokens = _tokenize("built in 1889")
        assert "1889" in tokens

    def test_empty_returns_empty(self) -> None:
        assert _tokenize("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert _tokenize("   \t\n  ") == []


class TestCosine:
    def test_identical_vectors(self) -> None:
        v = {"a": 1.0, "b": 2.0}
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        v1 = {"a": 1.0}
        v2 = {"b": 1.0}
        assert _cosine(v1, v2) == pytest.approx(0.0)

    def test_empty_vectors(self) -> None:
        assert _cosine({}, {}) == 0.0
        assert _cosine({"a": 1.0}, {}) == 0.0

    def test_zero_magnitude(self) -> None:
        assert _cosine({"a": 0.0}, {"a": 0.0}) == 0.0


class TestSentenceSimilarity:
    def test_identical_sentences(self) -> None:
        score = sentence_similarity("the sky is blue", "the sky is blue")
        assert score == pytest.approx(1.0)

    def test_completely_different(self) -> None:
        score = sentence_similarity("alpha beta gamma", "delta epsilon zeta")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        score = sentence_similarity("Paris is in France", "France is a country")
        assert 0.0 < score < 1.0

    def test_empty_sentences_return_zero(self) -> None:
        assert sentence_similarity("", "some text") == 0.0
        assert sentence_similarity("some text", "") == 0.0
        assert sentence_similarity("", "") == 0.0

    def test_result_in_unit_interval(self) -> None:
        score = sentence_similarity(
            "The Eiffel Tower is 330 metres tall.",
            "The structure stands 330 metres.",
        )
        assert 0.0 <= score <= 1.0

    def test_high_overlap_sentence(self) -> None:
        score = sentence_similarity(
            "The sky is blue in color",
            "The sky is blue",
        )
        assert score > 0.6

    @given(st.text(min_size=1, max_size=200), st.text(min_size=1, max_size=200))
    @settings(max_examples=100)
    def test_always_in_unit_interval(self, a: str, b: str) -> None:
        score = sentence_similarity(a, b)
        assert 0.0 <= score <= 1.0


class TestHybridSimilarity:
    def test_empty_strings_return_zero(self) -> None:
        assert hybrid_similarity("", "reference") == 0.0
        assert hybrid_similarity("output", "") == 0.0

    def test_identical_high_score(self) -> None:
        s = "Paris is the capital of France"
        score = hybrid_similarity(s, s)
        assert score > 0.9

    def test_paraphrase_overlap(self) -> None:
        score = hybrid_similarity(
            "Paris is the capital of France",
            "France's capital city is Paris",
        )
        assert score > 0.4

    def test_tfidf_weight_extremes(self) -> None:
        a = "Paris France capital"
        b = "France capital Paris city"
        score_pure_tfidf = hybrid_similarity(a, b, tfidf_weight=1.0)
        score_pure_f1 = hybrid_similarity(a, b, tfidf_weight=0.0)
        # Both should be positive
        assert score_pure_tfidf > 0.0
        assert score_pure_f1 > 0.0

    def test_hallucinated_claim_low_score(self) -> None:
        score = hybrid_similarity(
            "The Eiffel Tower is in Berlin Germany",
            "The Eiffel Tower is in Paris France",
        )
        # Paris != Berlin — score should be lower than identical case
        identical = hybrid_similarity(
            "The Eiffel Tower is in Paris France",
            "The Eiffel Tower is in Paris France",
        )
        assert score < identical

    @given(st.text(min_size=1, max_size=200), st.text(min_size=1, max_size=200))
    @settings(max_examples=100)
    def test_always_in_unit_interval(self, a: str, b: str) -> None:
        score = hybrid_similarity(a, b)
        assert 0.0 <= score <= 1.0


class TestFindBestMatch:
    def test_empty_reference_sentences(self) -> None:
        score, sent = find_best_match("some claim", [])
        assert score == 0.0
        assert sent == ""

    def test_exact_match(self) -> None:
        refs = ["The sky is blue.", "Water is wet.", "Fire is hot."]
        score, best = find_best_match("The sky is blue.", refs)
        assert score > 0.8
        assert "sky" in best.lower() or "blue" in best.lower()

    def test_best_match_is_highest(self) -> None:
        refs = [
            "Paris is in France.",         # matches
            "Berlin is in Germany.",       # doesn't match "Paris"
            "London is in England.",       # doesn't match
        ]
        score, best = find_best_match("Paris is the capital of France.", refs)
        assert "Paris" in best or "France" in best

    def test_returns_nonnegative_score(self) -> None:
        score, _ = find_best_match("completely random text xyz", ["abc def ghi"])
        assert score >= 0.0

    def test_single_reference_sentence(self) -> None:
        score, sent = find_best_match("Paris is in France.", ["Paris is in France."])
        assert score > 0.8
        assert sent == "Paris is in France."
