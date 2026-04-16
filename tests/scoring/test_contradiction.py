"""Tests for sf_hallucinate.scoring.contradiction — heuristic contradiction detection."""
from __future__ import annotations

import pytest

from sf_hallucinate.scoring.contradiction import (
    _check_antonyms,
    _check_negation_asymmetry,
    _check_numeric_contradiction,
    _extract_numbers,
    _has_negation,
    detect_contradiction,
)


class TestHasNegation:
    def test_with_not(self) -> None:
        assert _has_negation(["the", "tower", "is", "not", "tall"]) is True

    def test_without_negation(self) -> None:
        assert _has_negation(["the", "tower", "is", "tall"]) is False

    def test_with_contraction(self) -> None:
        assert _has_negation(["the", "tower", "isn't", "tall"]) is True

    def test_with_never(self) -> None:
        assert _has_negation(["i", "never", "said", "that"]) is True


class TestNegationAsymmetry:
    def test_one_side_negated(self) -> None:
        assert _check_negation_asymmetry(
            ["the", "tower", "is", "not", "tall"],
            ["the", "tower", "is", "tall"],
        ) is True

    def test_both_negated(self) -> None:
        assert _check_negation_asymmetry(
            ["the", "tower", "is", "not", "tall"],
            ["the", "building", "is", "not", "short"],
        ) is False

    def test_neither_negated(self) -> None:
        assert _check_negation_asymmetry(
            ["the", "tower", "is", "tall"],
            ["the", "building", "is", "tall"],
        ) is False


class TestCheckAntonyms:
    def test_increase_decrease(self) -> None:
        assert _check_antonyms({"increase", "in"}, {"decrease", "in"}) is True

    def test_no_antonyms(self) -> None:
        assert _check_antonyms({"tall", "tower"}, {"short", "building"}) is False

    def test_true_false(self) -> None:
        assert _check_antonyms({"true", "statement"}, {"false", "claim"}) is True


class TestExtractNumbers:
    def test_simple_integer(self) -> None:
        nums = _extract_numbers("The tower is 330 metres tall.")
        assert 330.0 in nums

    def test_decimal(self) -> None:
        nums = _extract_numbers("It weighs 3.14 kg.")
        assert any(abs(n - 3.14) < 0.01 for n in nums)

    def test_comma_separated(self) -> None:
        nums = _extract_numbers("Population is 1,000,000.")
        assert 1_000_000.0 in nums

    def test_magnitude_million(self) -> None:
        nums = _extract_numbers("Revenue of 5 million.")
        assert any(abs(n - 5_000_000) < 1 for n in nums)

    def test_no_numbers(self) -> None:
        assert _extract_numbers("No numbers here.") == []


class TestNumericContradiction:
    def test_different_numbers(self) -> None:
        assert _check_numeric_contradiction(
            "The tower is 330 metres tall.",
            "The tower is 150 metres tall.",
        ) is True

    def test_same_numbers(self) -> None:
        assert _check_numeric_contradiction(
            "The tower is 330 metres tall.",
            "The tower is 330 metres tall.",
        ) is False

    def test_no_numbers_in_claim(self) -> None:
        assert _check_numeric_contradiction(
            "The tower is tall.",
            "The tower is 330 metres tall.",
        ) is False


class TestDetectContradiction:
    def test_clear_negation_contradiction(self) -> None:
        is_contra, score = detect_contradiction(
            "The Earth is not round.",
            "The Earth is round and orbits the Sun.",
        )
        assert is_contra is True
        assert score > 0.5

    def test_no_contradiction(self) -> None:
        is_contra, score = detect_contradiction(
            "The Earth is round.",
            "The Earth is round and orbits the Sun.",
        )
        assert is_contra is False
        assert score == 0.0

    def test_numeric_contradiction(self) -> None:
        is_contra, score = detect_contradiction(
            "The population is 5 million.",
            "The population is 3 million.",
        )
        assert is_contra is True
        assert score > 0.5

    def test_antonym_contradiction(self) -> None:
        is_contra, score = detect_contradiction(
            "The stock price will increase sharply.",
            "The stock price will decrease gradually.",
        )
        assert is_contra is True
        assert score > 0.0

    def test_unrelated_sentences(self) -> None:
        is_contra, _ = detect_contradiction(
            "The sky is blue.",
            "Fish live in water.",
        )
        # Unrelated: negation asymmetry won't trigger (no shared vocab)
        assert is_contra is False

    def test_language_parameter_accepted(self) -> None:
        # Should not error for non-English
        is_contra, score = detect_contradiction(
            "El cielo no es azul.",
            "El cielo es azul.",
            language="es",
        )
        # Negation detection still works (uses same token set)
        assert isinstance(is_contra, bool)
        assert isinstance(score, float)
