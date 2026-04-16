"""Tests for the SpanForge integration module.

These tests verify the integration layer in isolation using mocked spanforge
objects so the test suite works regardless of whether spanforge is installed.
"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from sf_hallucinate._types import ClaimResult, EvalConfig, ScorerResult


def _make_result(passed: bool = True) -> ScorerResult:
    cr = ClaimResult(claim="Paris is in France.", best_match="Paris is in France.", similarity=0.9, grounded=True)
    return ScorerResult(
        hallucination_risk=0.1 if passed else 0.9,
        faithfulness_score=0.9 if passed else 0.1,
        grounded_claim_count=1,
        total_claim_count=1,
        claim_results=(cr,),
        threshold=0.5,
        passed=passed,
        metadata={"scorer": "faithfulness", "config": {}},
    )


class TestEmitEvalEventWithMockedSpanforge:
    """Test emit_eval_event() with spanforge mocked out."""

    def _make_spanforge_mock(self) -> MagicMock:
        mock_sf = MagicMock()
        mock_sf.get_config.return_value = None  # no active config
        mock_event = MagicMock()
        mock_sf.Event = MagicMock(return_value=mock_event)
        return mock_sf

    def test_emit_creates_event(self) -> None:
        mock_sf = self._make_spanforge_mock()
        mock_sf.emit = MagicMock()

        with patch.dict(sys.modules, {"spanforge": mock_sf}):
            # Re-import to pick up mock
            if "sf_hallucinate.integration.spanforge" in sys.modules:
                del sys.modules["sf_hallucinate.integration.spanforge"]
            try:
                from sf_hallucinate.integration.spanforge import emit_eval_event
                emit_eval_event(
                    output="The Eiffel Tower is in Paris.",
                    reference="The Eiffel Tower is in Paris, France.",
                    result=_make_result(passed=True),
                )
                mock_sf.Event.assert_called_once()
            finally:
                if "sf_hallucinate.integration.spanforge" in sys.modules:
                    del sys.modules["sf_hallucinate.integration.spanforge"]

    def test_emit_with_active_config_calls_emit(self) -> None:
        mock_sf = self._make_spanforge_mock()
        mock_sf.emit = MagicMock()

        with patch.dict(sys.modules, {"spanforge": mock_sf}):
            if "sf_hallucinate.integration.spanforge" in sys.modules:
                del sys.modules["sf_hallucinate.integration.spanforge"]
            try:
                from sf_hallucinate.integration.spanforge import emit_eval_event
                emit_eval_event(
                    output="test output",
                    reference="test reference",
                    result=_make_result(),
                )
                mock_sf.Event.assert_called_once()
                mock_sf.emit.assert_called_once()
            finally:
                if "sf_hallucinate.integration.spanforge" in sys.modules:
                    del sys.modules["sf_hallucinate.integration.spanforge"]

    def test_emit_swallows_spanforge_errors(self) -> None:
        mock_sf = self._make_spanforge_mock()
        mock_sf.Event = MagicMock(side_effect=RuntimeError("spanforge error"))

        with patch.dict(sys.modules, {"spanforge": mock_sf}):
            if "sf_hallucinate.integration.spanforge" in sys.modules:
                del sys.modules["sf_hallucinate.integration.spanforge"]
            try:
                from sf_hallucinate.integration.spanforge import emit_eval_event
                # Should NOT raise
                emit_eval_event(
                    output="test",
                    reference="ref",
                    result=_make_result(),
                )
            finally:
                if "sf_hallucinate.integration.spanforge" in sys.modules:
                    del sys.modules["sf_hallucinate.integration.spanforge"]
