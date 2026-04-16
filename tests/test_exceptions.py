"""Tests for sf_hallucinate._exceptions."""
from __future__ import annotations

import pytest

from sf_hallucinate._exceptions import (
    EmptyOutputError,
    EmptyReferenceError,
    HallucinationRiskExceeded,
    SfHallucinateError,
)
from sf_hallucinate._types import ClaimResult, ScorerResult


def _make_result(risk: float = 0.7, threshold: float = 0.5) -> ScorerResult:
    cr = ClaimResult(claim="c", best_match="m", similarity=0.3, grounded=False)
    return ScorerResult(
        hallucination_risk=risk,
        faithfulness_score=1.0 - risk,
        grounded_claim_count=0,
        total_claim_count=1,
        claim_results=(cr,),
        threshold=threshold,
        passed=False,
        metadata={},
    )


class TestHallucinationRiskExceeded:
    def test_is_sf_base(self) -> None:
        exc = HallucinationRiskExceeded(_make_result())
        assert isinstance(exc, SfHallucinateError)

    def test_attaches_result(self) -> None:
        result = _make_result(risk=0.8, threshold=0.4)
        exc = HallucinationRiskExceeded(result)
        assert exc.result is result
        assert exc.result.hallucination_risk == pytest.approx(0.8)

    def test_message_contains_risk(self) -> None:
        exc = HallucinationRiskExceeded(_make_result(risk=0.75, threshold=0.5))
        assert "0.750" in str(exc)
        assert "0.500" in str(exc)

    def test_raiseable(self) -> None:
        with pytest.raises(HallucinationRiskExceeded) as exc_info:
            raise HallucinationRiskExceeded(_make_result())
        assert exc_info.value.result.passed is False


class TestEmptyReferenceError:
    def test_is_sf_base(self) -> None:
        assert isinstance(EmptyReferenceError(), SfHallucinateError)

    def test_message(self) -> None:
        assert "Reference" in str(EmptyReferenceError())


class TestEmptyOutputError:
    def test_is_sf_base(self) -> None:
        assert isinstance(EmptyOutputError(), SfHallucinateError)

    def test_message(self) -> None:
        assert "output" in str(EmptyOutputError()).lower()
