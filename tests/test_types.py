"""Tests for sf_hallucinate._types (EvalConfig, ScorerResult, ClaimResult)."""
from __future__ import annotations

import dataclasses

import pytest

from sf_hallucinate._types import ClaimResult, EvalConfig, ScorerResult


class TestClaimResult:
    def test_basic_construction(self) -> None:
        cr = ClaimResult(
            claim="Paris is in France.",
            best_match="Paris is in France.",
            similarity=0.95,
            grounded=True,
        )
        assert cr.claim == "Paris is in France."
        assert cr.similarity == 0.95
        assert cr.grounded is True

    def test_frozen(self) -> None:
        cr = ClaimResult(claim="x", best_match="y", similarity=0.5, grounded=True)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cr.claim = "changed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        cr = ClaimResult(claim="x", best_match="y", similarity=0.5, grounded=True)
        d = cr.to_dict()
        assert d["claim"] == "x"
        assert d["similarity"] == 0.5
        assert d["grounded"] is True

    def test_ungrounded(self) -> None:
        cr = ClaimResult(claim="x", best_match="", similarity=0.1, grounded=False)
        assert cr.grounded is False

    def test_new_fields_default(self) -> None:
        cr = ClaimResult(claim="x", best_match="y", similarity=0.5, grounded=True)
        assert cr.contradiction_detected is False
        assert cr.entailment_label == ""
        assert cr.confidence == 1.0

    def test_new_fields_custom(self) -> None:
        cr = ClaimResult(
            claim="x",
            best_match="y",
            similarity=0.1,
            grounded=False,
            contradiction_detected=True,
            entailment_label="contradiction",
            confidence=0.85,
        )
        assert cr.contradiction_detected is True
        assert cr.entailment_label == "contradiction"
        assert cr.confidence == 0.85


class TestEvalConfig:
    def test_defaults(self) -> None:
        cfg = EvalConfig()
        assert cfg.threshold == 0.5
        assert cfg.grounding_threshold == 0.25
        assert cfg.min_claim_length == 15
        assert cfg.fail_on_threshold is False
        assert cfg.scorer_name == "faithfulness"
        assert cfg.tfidf_weight == 0.6

    def test_custom_values(self) -> None:
        cfg = EvalConfig(threshold=0.3, grounding_threshold=0.4, tfidf_weight=0.8)
        assert cfg.threshold == 0.3
        assert cfg.grounding_threshold == 0.4
        assert cfg.tfidf_weight == 0.8

    def test_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            EvalConfig(threshold=1.5)

    def test_threshold_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            EvalConfig(threshold=-0.1)

    def test_grounding_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="grounding_threshold"):
            EvalConfig(grounding_threshold=2.0)

    def test_min_claim_length_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="min_claim_length"):
            EvalConfig(min_claim_length=0)

    def test_tfidf_weight_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="tfidf_weight"):
            EvalConfig(tfidf_weight=1.5)

    def test_boundary_values_accepted(self) -> None:
        cfg = EvalConfig(threshold=0.0, grounding_threshold=0.0, tfidf_weight=0.0)
        assert cfg.threshold == 0.0
        cfg2 = EvalConfig(threshold=1.0, grounding_threshold=1.0, tfidf_weight=1.0)
        assert cfg2.threshold == 1.0

    def test_frozen(self) -> None:
        cfg = EvalConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.threshold = 0.9  # type: ignore[misc]

    def test_new_backend_defaults(self) -> None:
        cfg = EvalConfig()
        assert cfg.similarity_backend == "hybrid"
        assert cfg.embedding_model == "all-MiniLM-L6-v2"
        assert cfg.llm_model == "gpt-4o-mini"
        assert cfg.llm_api_key is None
        assert cfg.language == "en"
        assert cfg.detect_contradictions is True

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="similarity_backend"):
            EvalConfig(similarity_backend="bad")

    def test_invalid_language_raises(self) -> None:
        with pytest.raises(ValueError, match="language"):
            EvalConfig(language="xx")


class TestScorerResult:
    def _make_result(
        self,
        risk: float = 0.3,
        *,
        grounded: int = 7,
        total: int = 10,
        passed: bool = True,
    ) -> ScorerResult:
        cr = ClaimResult(claim="x", best_match="y", similarity=0.7, grounded=True)
        return ScorerResult(
            hallucination_risk=risk,
            faithfulness_score=round(1.0 - risk, 6),
            grounded_claim_count=grounded,
            total_claim_count=total,
            claim_results=tuple([cr] * total),
            threshold=0.5,
            passed=passed,
            metadata={"scorer": "faithfulness"},
        )

    def test_grounding_rate(self) -> None:
        r = self._make_result(grounded=7, total=10)
        assert r.grounding_rate == pytest.approx(0.7)

    def test_grounding_rate_no_claims(self) -> None:
        cr_empty: tuple[ClaimResult, ...] = ()
        r = ScorerResult(
            hallucination_risk=0.0,
            faithfulness_score=1.0,
            grounded_claim_count=0,
            total_claim_count=0,
            claim_results=cr_empty,
            threshold=0.5,
            passed=True,
            metadata={},
        )
        assert r.grounding_rate == 1.0

    def test_ungrounded_claims(self) -> None:
        grounded_cr = ClaimResult(claim="g", best_match="g", similarity=0.8, grounded=True)
        ungrd_cr = ClaimResult(claim="u", best_match="", similarity=0.1, grounded=False)
        r = ScorerResult(
            hallucination_risk=0.5,
            faithfulness_score=0.5,
            grounded_claim_count=1,
            total_claim_count=2,
            claim_results=(grounded_cr, ungrd_cr),
            threshold=0.5,
            passed=True,
            metadata={},
        )
        assert len(r.ungrounded_claims) == 1
        assert r.ungrounded_claims[0].claim == "u"

    def test_to_dict_structure(self) -> None:
        r = self._make_result()
        d = r.to_dict()
        assert "hallucination_risk" in d
        assert "faithfulness_score" in d
        assert "grounding_rate" in d
        assert "claim_results" in d
        assert isinstance(d["claim_results"], list)
        # New fields
        assert "confidence" in d
        assert "contradiction_count" in d

    def test_frozen(self) -> None:
        r = self._make_result()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            r.hallucination_risk = 0.9  # type: ignore[misc]
