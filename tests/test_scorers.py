"""Tests for sf_hallucinate.scorers — AnswerRelevancy and ContextRelevancy."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from sf_hallucinate._types import EvalConfig, ScorerResult
from sf_hallucinate.scorers.answer_relevancy import AnswerRelevancyScorer
from sf_hallucinate.scorers.context_relevancy import ContextRelevancyScorer


class TestAnswerRelevancyScorer:
    def test_hybrid_backend(self) -> None:
        config = EvalConfig(similarity_backend="hybrid")
        scorer = AnswerRelevancyScorer(
            question="What is the capital of France?",
            config=config,
        )
        result = scorer.score(
            "The capital of France is Paris.",
            "Paris is the capital of France.",
        )
        assert isinstance(result, ScorerResult)
        assert result.faithfulness_score > 0.0

    def test_hybrid_unrelated_answer(self) -> None:
        config = EvalConfig(similarity_backend="hybrid")
        scorer = AnswerRelevancyScorer(
            question="What is the capital of France?",
            config=config,
        )
        result = scorer.score(
            "Fish live in water and need oxygen.",
            "Paris is the capital of France.",
        )
        # Score should be relatively low for unrelated answer
        assert result.faithfulness_score < 0.7

    def test_hybrid_empty_question_tokens(self) -> None:
        config = EvalConfig(similarity_backend="hybrid")
        scorer = AnswerRelevancyScorer(
            question="the a an",  # all stop words
            config=config,
        )
        result = scorer.score("Some answer.", "Reference.")
        assert result.faithfulness_score == 0.0

    def test_hybrid_empty_answer(self) -> None:
        config = EvalConfig(similarity_backend="hybrid")
        scorer = AnswerRelevancyScorer(
            question="What is Python?",
            config=config,
        )
        result = scorer.score("", "Reference text.")
        assert result.faithfulness_score == 0.0

    def test_default_config(self) -> None:
        scorer = AnswerRelevancyScorer(question="What is X?")
        assert scorer.config is not None
        assert scorer.name == "answer_relevancy"

    def test_score_batch(self) -> None:
        config = EvalConfig(similarity_backend="hybrid")
        scorer = AnswerRelevancyScorer(
            question="What is the capital of France?",
            config=config,
        )
        results = scorer.score_batch(
            ["Paris is the capital.", "I don't know."],
            ["Paris is the capital.", "Paris info."],
        )
        assert len(results) == 2
        assert all(isinstance(r, ScorerResult) for r in results)

    def test_score_batch_length_mismatch(self) -> None:
        scorer = AnswerRelevancyScorer(question="Q?")
        with pytest.raises(ValueError, match="same length"):
            scorer.score_batch(["a", "b"], ["c"])

    @patch("sf_hallucinate._llm.call_chat_completion")
    def test_llm_backend(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps({
            "relevancy_score": 0.85,
            "explanation": "The answer directly addresses the question.",
        })

        config = EvalConfig(
            similarity_backend="llm-nli",
            llm_api_key="test-key",
        )
        scorer = AnswerRelevancyScorer(
            question="What is the capital of France?",
            config=config,
        )
        result = scorer.score(
            "The capital of France is Paris.",
            "Paris is the capital of France.",
        )
        assert isinstance(result, ScorerResult)
        assert result.faithfulness_score == pytest.approx(0.85, abs=0.01)

    @patch("sf_hallucinate._llm.call_chat_completion")
    def test_llm_backend_json_parse_failure(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "not valid json at all"
        config = EvalConfig(
            similarity_backend="llm-nli",
            llm_api_key="test-key",
        )
        scorer = AnswerRelevancyScorer(question="Q?", config=config)
        result = scorer.score("Answer", "Reference")
        # Should fall back to 0.5 on parse failure
        assert result.faithfulness_score == pytest.approx(0.5, abs=0.01)

    @patch("sf_hallucinate._llm.call_chat_completion")
    def test_llm_result_metadata(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps({"relevancy_score": 0.7})
        config = EvalConfig(
            similarity_backend="llm-nli",
            llm_api_key="test-key",
        )
        scorer = AnswerRelevancyScorer(question="Q?", config=config)
        result = scorer.score("A", "R")
        assert result.metadata["backend"] == "llm-nli"
        assert result.confidence == 0.9  # llm-nli gives 0.9

    def test_hybrid_result_confidence(self) -> None:
        scorer = AnswerRelevancyScorer(question="What is X?")
        result = scorer.score("X is Y.", "X is Y.")
        assert result.confidence == 0.5  # hybrid gives 0.5

    def test_scorer_name(self) -> None:
        config = EvalConfig(similarity_backend="hybrid")
        scorer = AnswerRelevancyScorer(
            question="What is X?",
            config=config,
        )
        result = scorer.score("Answer text", "Reference text")
        assert result.metadata["scorer"] == "answer_relevancy"


class TestContextRelevancyScorer:
    def test_hybrid_backend(self) -> None:
        config = EvalConfig(similarity_backend="hybrid")
        scorer = ContextRelevancyScorer(
            question="What is the capital of France?",
            config=config,
        )
        result = scorer.score(
            "The capital of France is Paris.",
            "Paris is the capital and largest city of France.",
        )
        assert isinstance(result, ScorerResult)
        assert result.faithfulness_score > 0.0

    def test_hybrid_empty_tokens(self) -> None:
        config = EvalConfig(similarity_backend="hybrid")
        scorer = ContextRelevancyScorer(
            question="the a an",  # all stop words
            config=config,
        )
        result = scorer.score("Answer.", "Some context.")
        assert result.faithfulness_score == 0.0

    def test_hybrid_empty_reference(self) -> None:
        config = EvalConfig(similarity_backend="hybrid")
        scorer = ContextRelevancyScorer(question="What?", config=config)
        result = scorer.score("Answer", "")
        assert result.faithfulness_score == 0.0

    def test_default_config(self) -> None:
        scorer = ContextRelevancyScorer(question="What?")
        assert scorer.config is not None
        assert scorer.name == "context_relevancy"

    def test_score_batch(self) -> None:
        config = EvalConfig(similarity_backend="hybrid")
        scorer = ContextRelevancyScorer(
            question="What is Python?",
            config=config,
        )
        results = scorer.score_batch(
            ["Python is a language.", "I don't know."],
            ["Python is a programming language.", "Java info."],
        )
        assert len(results) == 2
        assert all(isinstance(r, ScorerResult) for r in results)

    def test_score_batch_length_mismatch(self) -> None:
        scorer = ContextRelevancyScorer(question="Q?")
        with pytest.raises(ValueError, match="same length"):
            scorer.score_batch(["a"], ["b", "c"])

    @patch("sf_hallucinate._llm.call_chat_completion")
    def test_llm_backend(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps({
            "relevancy_score": 0.9,
            "explanation": "The context directly supports answering the question.",
        })

        config = EvalConfig(
            similarity_backend="llm-nli",
            llm_api_key="test-key",
        )
        scorer = ContextRelevancyScorer(
            question="What is the capital of France?",
            config=config,
        )
        result = scorer.score(
            "The capital of France is Paris.",
            "Paris is the capital city of France.",
        )
        assert isinstance(result, ScorerResult)
        assert result.faithfulness_score == pytest.approx(0.9, abs=0.01)

    @patch("sf_hallucinate._llm.call_chat_completion")
    def test_llm_backend_json_parse_failure(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "invalid json"
        config = EvalConfig(
            similarity_backend="llm-nli",
            llm_api_key="test-key",
        )
        scorer = ContextRelevancyScorer(question="Q?", config=config)
        result = scorer.score("Answer", "Context")
        assert result.faithfulness_score == pytest.approx(0.5, abs=0.01)

    @patch("sf_hallucinate._llm.call_chat_completion")
    def test_llm_result_metadata(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps({"relevancy_score": 0.8})
        config = EvalConfig(
            similarity_backend="llm-nli",
            llm_api_key="test-key",
        )
        scorer = ContextRelevancyScorer(question="Q?", config=config)
        result = scorer.score("A", "R")
        assert result.metadata["backend"] == "llm-nli"
        assert result.metadata["scorer"] == "context_relevancy"
        assert result.confidence == 0.9

    def test_hybrid_result_confidence(self) -> None:
        scorer = ContextRelevancyScorer(question="What is X?")
        result = scorer.score("X is Y.", "X is Y.")
        assert result.confidence == 0.5

    def test_scorer_name(self) -> None:
        config = EvalConfig(similarity_backend="hybrid")
        scorer = ContextRelevancyScorer(
            question="What is X?",
            config=config,
        )
        result = scorer.score("Answer text", "Reference text")
        assert result.metadata["scorer"] == "context_relevancy"
