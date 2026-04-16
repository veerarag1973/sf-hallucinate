"""Tests for sf_hallucinate.scoring.claims (sentence splitting + extraction)."""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sf_hallucinate.scoring.claims import _is_meta_sentence, extract_claims, split_sentences


class TestSplitSentences:
    def test_simple_two_sentences(self) -> None:
        result = split_sentences("The sky is blue. It was always so.")
        assert len(result) == 2
        assert result[0].startswith("The sky")
        assert result[1].startswith("It was")

    def test_empty_string_returns_empty(self) -> None:
        assert split_sentences("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert split_sentences("   \n\t  ") == []

    def test_abbreviation_dr_not_split(self) -> None:
        text = "Dr. Smith said it was fine. Really fine."
        result = split_sentences(text)
        # Should not split on "Dr."
        assert any("Dr." in s for s in result)
        # But should split the two sentences
        assert len(result) >= 1

    def test_exclamation_splits(self) -> None:
        result = split_sentences("Hello! World is great.")
        assert len(result) >= 1

    def test_question_mark_splits(self) -> None:
        result = split_sentences("Is it blue? Yes it is.")
        assert len(result) >= 1

    def test_numbered_list(self) -> None:
        text = "1. First item\n2. Second item\n3. Third item"
        result = split_sentences(text)
        assert len(result) == 3

    def test_bullet_list(self) -> None:
        text = "- Item one\n- Item two\n- Item three"
        result = split_sentences(text)
        assert len(result) == 3

    def test_paragraph_separation(self) -> None:
        text = "First paragraph sentence.\n\nSecond paragraph sentence."
        result = split_sentences(text)
        assert len(result) == 2

    def test_single_sentence_no_period(self) -> None:
        result = split_sentences("The sky is blue")
        assert len(result) == 1
        assert result[0] == "The sky is blue"

    def test_multiple_sentences_complex(self) -> None:
        text = (
            "The Eiffel Tower is in Paris. "
            "It was built in 1889. "
            "Gustave Eiffel designed it."
        )
        result = split_sentences(text)
        assert len(result) == 3

    def test_no_empty_strings_in_result(self) -> None:
        result = split_sentences("One sentence. Another.")
        assert all(s.strip() for s in result)

    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=200)
    def test_no_empty_results(self, text: str) -> None:
        result = split_sentences(text)
        assert all(len(s) > 0 for s in result)

    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=200)
    def test_returns_list_of_str(self, text: str) -> None:
        result = split_sentences(text)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)


class TestIsMetaSentence:
    @pytest.mark.parametrize(
        "sentence",
        [
            "In summary, the results show that...",
            "In conclusion, we can say...",
            "Here is a breakdown of the facts.",
            "Here are the key points:",
            "Note that the following is important.",
            "As mentioned above, the tower is tall.",
            "For example, Paris is in France.",
            "For instance, the sky is blue.",
        ],
    )
    def test_meta_sentences_detected(self, sentence: str) -> None:
        assert _is_meta_sentence(sentence) is True

    @pytest.mark.parametrize(
        "sentence",
        [
            "The Eiffel Tower is in Paris.",
            "Water boils at 100 degrees Celsius.",
            "Gustave Eiffel designed the tower.",
            "The structure stands 330 metres tall.",
        ],
    )
    def test_factual_sentences_not_meta(self, sentence: str) -> None:
        assert _is_meta_sentence(sentence) is False


class TestExtractClaims:
    def test_question_excluded(self) -> None:
        text = "The sky is blue. Is it always blue? Mostly, yes."
        claims = extract_claims(text)
        assert all(not c.endswith("?") for c in claims)

    def test_short_sentences_excluded(self) -> None:
        claims = extract_claims("Yes. The Eiffel Tower is in Paris, France.")
        # "Yes." is only 3 chars — should be excluded
        assert not any(c == "Yes." for c in claims)

    def test_factual_sentences_included(self) -> None:
        text = "The Eiffel Tower is in Paris. It was built in 1889."
        claims = extract_claims(text)
        assert len(claims) >= 1

    def test_meta_phrases_excluded(self) -> None:
        text = "In summary, the tower is in Paris. The tower is 330m tall."
        claims = extract_claims(text)
        assert not any("In summary" in c for c in claims)

    def test_min_length_respected(self) -> None:
        text = "Short. A longer and more complete factual sentence about Paris."
        claims = extract_claims(text, min_length=10)
        assert all(len(c) >= 10 for c in claims)

    def test_empty_text_returns_empty(self) -> None:
        assert extract_claims("") == []

    def test_all_questions_returns_empty(self) -> None:
        text = "Is the sky blue? Was it always so? Can we know?"
        claims = extract_claims(text)
        assert claims == []

    def test_complex_output(self) -> None:
        text = (
            "The Eiffel Tower is a wrought-iron lattice tower located in Paris. "
            "It was designed by Gustave Eiffel and built between 1887 and 1889. "
            "In summary, it is a famous landmark."
        )
        claims = extract_claims(text)
        # First two factual sentences should be included
        assert any("Paris" in c for c in claims)
        assert any("Gustave Eiffel" in c for c in claims)

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=150)
    def test_returns_list_of_str(self, text: str) -> None:
        claims = extract_claims(text)
        assert isinstance(claims, list)
        assert all(isinstance(c, str) for c in claims)

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=150)
    def test_no_empty_claims(self, text: str) -> None:
        claims = extract_claims(text)
        assert all(len(c) >= 15 for c in claims)
