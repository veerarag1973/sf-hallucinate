"""Tests for sf_hallucinate.scoring.nli — LLM NLI backend (mocked)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from sf_hallucinate._types import EvalConfig
from sf_hallucinate.scoring.backends import ClaimScore
from sf_hallucinate.scoring.nli import LLMNLIBackend, _parse_llm_response


class TestParseLlmResponse:
    def test_clean_json_array(self) -> None:
        raw = json.dumps([
            {"label": "entailment", "confidence": 0.9},
            {"label": "contradiction", "confidence": 0.8},
        ])
        results = _parse_llm_response(raw, num_claims=2)
        assert len(results) == 2
        assert results[0]["label"] == "entailment"
        assert results[1]["label"] == "contradiction"

    def test_json_inside_markdown(self) -> None:
        raw = "Here is the result:\n```json\n" + json.dumps([
            {"label": "neutral", "confidence": 0.5},
        ]) + "\n```"
        results = _parse_llm_response(raw, num_claims=1)
        assert len(results) == 1
        assert results[0]["label"] == "neutral"

    def test_pads_missing_items(self) -> None:
        raw = json.dumps([
            {"label": "entailment", "confidence": 0.9},
        ])
        results = _parse_llm_response(raw, num_claims=3)
        assert len(results) == 3
        assert results[1]["label"] == "neutral"
        assert results[2]["label"] == "neutral"

    def test_invalid_json_fallback(self) -> None:
        raw = "This is not valid JSON at all"
        results = _parse_llm_response(raw, num_claims=2)
        assert len(results) == 2
        assert all(r["label"] == "neutral" for r in results)


class TestLLMNLIBackend:
    def test_requires_api_key(self) -> None:
        import os
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key"):
                config = EvalConfig(
                    similarity_backend="llm-nli",
                    llm_api_key=None,
                )
                LLMNLIBackend(config)
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old

    @patch("sf_hallucinate.scoring.nli.call_chat_completion")
    def test_score_claim(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps([
            {"label": "entailment", "confidence": 0.95},
        ])

        config = EvalConfig(
            similarity_backend="llm-nli",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini",
        )
        backend = LLMNLIBackend(config)
        result = backend.score_claim(
            "Paris is the capital of France.",
            ["Paris is the capital of France."],
        )

        assert isinstance(result, ClaimScore)
        assert result.similarity > 0.9
        assert result.entailment_label == "entailment"
        assert result.contradiction_detected is False

    @patch("sf_hallucinate.scoring.nli.call_chat_completion")
    def test_score_contradiction(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps([
            {"label": "contradiction", "confidence": 0.9},
        ])

        config = EvalConfig(
            similarity_backend="llm-nli",
            llm_api_key="test-key",
        )
        backend = LLMNLIBackend(config)
        result = backend.score_claim(
            "The sky is green.",
            ["The sky is blue."],
        )

        assert result.similarity < 0.1
        assert result.entailment_label == "contradiction"
        assert result.contradiction_detected is True

    @patch("sf_hallucinate.scoring.nli.call_chat_completion")
    def test_score_claims_batch(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps([
            {"label": "entailment", "confidence": 0.9},
            {"label": "neutral", "confidence": 0.6},
            {"label": "contradiction", "confidence": 0.85},
        ])

        config = EvalConfig(
            similarity_backend="llm-nli",
            llm_api_key="test-key",
        )
        backend = LLMNLIBackend(config)
        results = backend.score_claims_batch(
            ["Claim A", "Claim B", "Claim C"],
            ["Reference sentence 1.", "Reference sentence 2."],
        )

        assert len(results) == 3
        assert results[0].entailment_label == "entailment"
        assert results[1].entailment_label == "neutral"
        assert results[2].entailment_label == "contradiction"
        assert results[2].contradiction_detected is True

    @patch("sf_hallucinate.scoring.nli.call_chat_completion")
    def test_empty_references(self, mock_llm: MagicMock) -> None:
        config = EvalConfig(
            similarity_backend="llm-nli",
            llm_api_key="test-key",
        )
        backend = LLMNLIBackend(config)
        result = backend.score_claim("Any claim", [])

        assert result.similarity == 0.0
        assert result.best_match == ""
        mock_llm.assert_not_called()
