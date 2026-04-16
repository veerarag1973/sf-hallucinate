"""Tests for sf_hallucinate.scoring.embedding — EmbeddingBackend (mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sf_hallucinate._types import EvalConfig
from sf_hallucinate.scoring.backends import ClaimScore


class TestEmbeddingBackend:
    def test_import_error_when_missing(self) -> None:
        """EmbeddingBackend raises ImportError when sentence-transformers is absent."""
        config = EvalConfig(similarity_backend="embedding")
        with pytest.raises(ImportError, match="sentence-transformers"):
            from sf_hallucinate.scoring.embedding import EmbeddingBackend
            # Force re-import scenario
            import sys
            with patch.dict(sys.modules, {"sentence_transformers": None}):
                EmbeddingBackend(config)

    @patch("sf_hallucinate.scoring.embedding.EmbeddingBackend.__init__", return_value=None)
    def test_score_claim_with_mock(self, mock_init: MagicMock) -> None:
        """Test scoring logic with a mocked model."""
        import numpy as np

        from sf_hallucinate.scoring.embedding import EmbeddingBackend

        backend = EmbeddingBackend.__new__(EmbeddingBackend)
        backend._language = "en"
        backend._detect_contradictions = False

        # Mock the model
        mock_model = MagicMock()
        # Claim vector, then two reference vectors
        mock_model.encode.return_value = [
            [1.0, 0.0, 0.0],  # claim
            [0.9, 0.1, 0.0],  # ref 1 (similar)
            [0.0, 0.0, 1.0],  # ref 2 (different)
        ]
        backend._model = mock_model

        result = backend.score_claim(
            "Paris is in France.",
            ["Paris is in France.", "Berlin is in Germany."],
        )

        assert isinstance(result, ClaimScore)
        assert result.similarity > 0.5
        assert result.best_match == "Paris is in France."

    @patch("sf_hallucinate.scoring.embedding.EmbeddingBackend.__init__", return_value=None)
    def test_empty_references(self, mock_init: MagicMock) -> None:
        from sf_hallucinate.scoring.embedding import EmbeddingBackend

        backend = EmbeddingBackend.__new__(EmbeddingBackend)
        backend._language = "en"
        backend._detect_contradictions = False
        backend._model = MagicMock()

        result = backend.score_claim("Any claim", [])
        assert result.similarity == 0.0
        assert result.best_match == ""
