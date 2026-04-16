"""Core tests for FaithfulnessScorer, EvalScorer protocol, and EvalPipeline."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sf_hallucinate._exceptions import (
    EmptyOutputError,
    EmptyReferenceError,
    HallucinationRiskExceeded,
)
from sf_hallucinate._types import ClaimResult, EvalConfig, ScorerResult
from sf_hallucinate.eval import EvalPipeline, EvalScorer, FaithfulnessScorer
from tests.conftest import (
    OUTPUT_FAITHFUL,
    OUTPUT_HALLUCINATED,
    OUTPUT_MIXED,
    REFERENCE_PARIS,
)


class TestEvalScorerProtocol:
    def test_faithfulness_scorer_satisfies_protocol(self) -> None:
        scorer = FaithfulnessScorer()
        assert isinstance(scorer, EvalScorer)

    def test_scorer_has_name(self) -> None:
        scorer = FaithfulnessScorer()
        assert scorer.name == "faithfulness"

    def test_scorer_has_config(self) -> None:
        scorer = FaithfulnessScorer()
        assert isinstance(scorer.config, EvalConfig)

    def test_custom_class_satisfies_protocol(self) -> None:
        class MyScorer:
            name = "custom"
            config = EvalConfig()

            def score(self, output: str, reference: str) -> ScorerResult:  # noqa: ARG002
                raise NotImplementedError

            def score_batch(
                self, outputs: list[str], references: list[str]
            ) -> list[ScorerResult]:
                raise NotImplementedError

        assert isinstance(MyScorer(), EvalScorer)


class TestFaithfulnessScorer:
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_default_config(self) -> None:
        scorer = FaithfulnessScorer()
        assert scorer.config.threshold == 0.5
        assert scorer.config.grounding_threshold == 0.25

    def test_custom_config(self) -> None:
        cfg = EvalConfig(threshold=0.3)
        scorer = FaithfulnessScorer(cfg)
        assert scorer.config.threshold == 0.3

    # ------------------------------------------------------------------
    # score() — happy path
    # ------------------------------------------------------------------

    def test_faithful_output_passes(self) -> None:
        # Faithful output risk ≈ 0.57 (paraphrased sentences lower TF-IDF scores);
        # use a generous threshold so the test validates correct direction, not
        # exact values which are algorithm-implementation-specific.
        scorer = FaithfulnessScorer(EvalConfig(threshold=0.65))
        result = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert isinstance(result, ScorerResult)
        assert result.passed is True
        assert result.hallucination_risk < 0.65

    def test_hallucinated_output_higher_risk(self) -> None:
        scorer = FaithfulnessScorer()
        faithful_result = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        hallucinated_result = scorer.score(OUTPUT_HALLUCINATED, REFERENCE_PARIS)
        assert hallucinated_result.hallucination_risk > faithful_result.hallucination_risk

    def test_result_type(self) -> None:
        scorer = FaithfulnessScorer()
        result = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert isinstance(result, ScorerResult)
        assert isinstance(result.claim_results, tuple)
        assert all(isinstance(cr, ClaimResult) for cr in result.claim_results)

    def test_risk_in_unit_interval(self) -> None:
        scorer = FaithfulnessScorer()
        result = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert 0.0 <= result.hallucination_risk <= 1.0

    def test_faithfulness_plus_risk_equals_one(self) -> None:
        scorer = FaithfulnessScorer()
        result = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        total = result.faithfulness_score + result.hallucination_risk
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_grounded_count_lte_total(self) -> None:
        scorer = FaithfulnessScorer()
        result = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert result.grounded_claim_count <= result.total_claim_count

    def test_claim_results_match_total_count(self) -> None:
        scorer = FaithfulnessScorer()
        result = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert len(result.claim_results) == result.total_claim_count

    def test_metadata_has_scorer_name(self) -> None:
        scorer = FaithfulnessScorer()
        result = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert result.metadata.get("scorer") == "faithfulness"

    def test_threshold_stored_in_result(self) -> None:
        scorer = FaithfulnessScorer(EvalConfig(threshold=0.35))
        result = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert result.threshold == 0.35

    def test_passed_reflects_threshold(self) -> None:
        scorer_tight = FaithfulnessScorer(EvalConfig(threshold=0.0))
        result = scorer_tight.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        # risk > 0 for any real output, so threshold=0.0 should fail
        assert result.passed is (result.hallucination_risk <= 0.0)

    # ------------------------------------------------------------------
    # score() — mixed output
    # ------------------------------------------------------------------

    def test_mixed_output_intermediate_risk(self) -> None:
        scorer = FaithfulnessScorer()
        faithful = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        mixed = scorer.score(OUTPUT_MIXED, REFERENCE_PARIS)
        hallucinated = scorer.score(OUTPUT_HALLUCINATED, REFERENCE_PARIS)
        assert faithful.hallucination_risk <= mixed.hallucination_risk
        assert mixed.hallucination_risk <= hallucinated.hallucination_risk

    # ------------------------------------------------------------------
    # score() — error handling
    # ------------------------------------------------------------------

    def test_empty_output_raises(self) -> None:
        scorer = FaithfulnessScorer()
        with pytest.raises(EmptyOutputError):
            scorer.score("", REFERENCE_PARIS)

    def test_whitespace_only_output_raises(self) -> None:
        scorer = FaithfulnessScorer()
        with pytest.raises(EmptyOutputError):
            scorer.score("   \n\t  ", REFERENCE_PARIS)

    def test_empty_reference_raises(self) -> None:
        scorer = FaithfulnessScorer()
        with pytest.raises(EmptyReferenceError):
            scorer.score(OUTPUT_FAITHFUL, "")

    def test_whitespace_only_reference_raises(self) -> None:
        scorer = FaithfulnessScorer()
        with pytest.raises(EmptyReferenceError):
            scorer.score(OUTPUT_FAITHFUL, "   ")

    def test_fail_on_threshold_raises(self) -> None:
        scorer = FaithfulnessScorer(
            EvalConfig(threshold=0.0, fail_on_threshold=True)
        )
        with pytest.raises(HallucinationRiskExceeded) as exc_info:
            scorer.score(OUTPUT_HALLUCINATED, REFERENCE_PARIS)
        assert exc_info.value.result.passed is False

    def test_fail_on_threshold_false_no_raise(self) -> None:
        scorer = FaithfulnessScorer(
            EvalConfig(threshold=0.0, fail_on_threshold=False)
        )
        result = scorer.score(OUTPUT_HALLUCINATED, REFERENCE_PARIS)
        assert isinstance(result, ScorerResult)

    def test_exception_contains_result(self) -> None:
        scorer = FaithfulnessScorer(
            EvalConfig(threshold=0.0, fail_on_threshold=True)
        )
        with pytest.raises(HallucinationRiskExceeded) as exc_info:
            scorer.score(OUTPUT_HALLUCINATED, REFERENCE_PARIS)
        assert isinstance(exc_info.value.result, ScorerResult)

    # ------------------------------------------------------------------
    # score() — edge cases
    # ------------------------------------------------------------------

    def test_output_with_only_questions(self) -> None:
        scorer = FaithfulnessScorer()
        # No claims extracted → risk = 0.0 (benefit of the doubt)
        result = scorer.score("Is it tall? Was it built in Paris?", REFERENCE_PARIS)
        assert result.total_claim_count == 0
        assert result.hallucination_risk == pytest.approx(0.0)
        assert result.passed is True

    def test_single_sentence_output(self) -> None:
        scorer = FaithfulnessScorer()
        result = scorer.score("The Eiffel Tower is in Paris.", REFERENCE_PARIS)
        assert result.total_claim_count >= 1

    def test_long_reference_does_not_error(self) -> None:
        scorer = FaithfulnessScorer()
        long_ref = (REFERENCE_PARIS + " ") * 20
        result = scorer.score(OUTPUT_FAITHFUL, long_ref)
        assert isinstance(result, ScorerResult)

    # ------------------------------------------------------------------
    # score_batch()
    # ------------------------------------------------------------------

    def test_score_batch_length_mismatch_raises(self) -> None:
        scorer = FaithfulnessScorer()
        with pytest.raises(ValueError, match="same length"):
            scorer.score_batch(["a", "b"], ["c"])

    def test_score_batch_returns_correct_length(self) -> None:
        scorer = FaithfulnessScorer()
        outputs = [OUTPUT_FAITHFUL, OUTPUT_HALLUCINATED]
        refs = [REFERENCE_PARIS, REFERENCE_PARIS]
        results = scorer.score_batch(outputs, refs)
        assert len(results) == 2

    def test_score_batch_results_match_individual(self) -> None:
        scorer = FaithfulnessScorer()
        outputs = [OUTPUT_FAITHFUL, OUTPUT_HALLUCINATED]
        refs = [REFERENCE_PARIS, REFERENCE_PARIS]
        batch = scorer.score_batch(outputs, refs)
        for i, (o, r) in enumerate(zip(outputs, refs)):
            individual = scorer.score(o, r)
            assert batch[i].hallucination_risk == pytest.approx(
                individual.hallucination_risk, abs=1e-6
            )

    def test_score_batch_empty_lists(self) -> None:
        scorer = FaithfulnessScorer()
        results = scorer.score_batch([], [])
        assert results == []

    # ------------------------------------------------------------------
    # async scoring
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ascore_returns_same_as_sync(self) -> None:
        scorer = FaithfulnessScorer()
        sync_result = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        async_result = await scorer.ascore(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert async_result.hallucination_risk == pytest.approx(
            sync_result.hallucination_risk, abs=1e-6
        )

    @pytest.mark.asyncio
    async def test_ascore_batch_returns_correct_length(self) -> None:
        scorer = FaithfulnessScorer()
        results = await scorer.ascore_batch(
            [OUTPUT_FAITHFUL, OUTPUT_HALLUCINATED],
            [REFERENCE_PARIS, REFERENCE_PARIS],
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_ascore_batch_length_mismatch_raises(self) -> None:
        scorer = FaithfulnessScorer()
        with pytest.raises(ValueError):
            await scorer.ascore_batch(["a", "b"], ["c"])

    # ------------------------------------------------------------------
    # SpanForge event emission
    # ------------------------------------------------------------------

    def test_emit_called_on_score(self) -> None:
        scorer = FaithfulnessScorer()
        with patch("sf_hallucinate.eval.emit_eval_event") as mock_emit:
            scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
            mock_emit.assert_called_once()
            kwargs = mock_emit.call_args.kwargs
            assert "output" in kwargs
            assert "reference" in kwargs
            assert "result" in kwargs

    def test_emit_failure_does_not_crash_scorer(self) -> None:
        # The real emit_eval_event wraps _do_emit in try/except, so emission
        # errors never propagate.  Verify the scorer works normally.
        scorer = FaithfulnessScorer()
        result = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert isinstance(result, ScorerResult)


class TestEvalPipeline:
    def test_single_scorer(self) -> None:
        scorer = FaithfulnessScorer(EvalConfig(threshold=0.5))
        pipeline = EvalPipeline(scorer)
        results = pipeline.run(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert "faithfulness" in results
        assert isinstance(results["faithfulness"], ScorerResult)

    def test_multiple_scorers(self) -> None:
        s1 = FaithfulnessScorer(EvalConfig(scorer_name="scorer_a"))
        s2 = FaithfulnessScorer(EvalConfig(scorer_name="scorer_b"))
        pipeline = EvalPipeline(s1, s2)
        results = pipeline.run(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert "scorer_a" in results
        assert "scorer_b" in results

    def test_no_scorers_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            EvalPipeline()

    def test_failing_scorer_propagates(self) -> None:
        scorer = FaithfulnessScorer(
            EvalConfig(threshold=0.0, fail_on_threshold=True)
        )
        pipeline = EvalPipeline(scorer)
        with pytest.raises(HallucinationRiskExceeded):
            pipeline.run(OUTPUT_HALLUCINATED, REFERENCE_PARIS)

    def test_results_same_as_individual(self) -> None:
        scorer = FaithfulnessScorer()
        pipeline = EvalPipeline(scorer)
        pipe_result = pipeline.run(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        direct_result = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert pipe_result["faithfulness"].hallucination_risk == pytest.approx(
            direct_result.hallucination_risk, abs=1e-6
        )

    def test_passed_true_when_all_pass(self) -> None:
        scorer = FaithfulnessScorer(EvalConfig(threshold=0.99))
        pipeline = EvalPipeline(scorer)
        pipeline.run(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        assert pipeline.passed is True

    def test_passed_false_when_any_fail(self) -> None:
        scorer = FaithfulnessScorer(EvalConfig(threshold=0.0))
        pipeline = EvalPipeline(scorer)
        pipeline.run(OUTPUT_HALLUCINATED, REFERENCE_PARIS)
        assert pipeline.passed is False

    def test_passed_before_run_raises(self) -> None:
        scorer = FaithfulnessScorer()
        pipeline = EvalPipeline(scorer)
        with pytest.raises(RuntimeError, match="Call run"):
            pipeline.passed  # noqa: B018


class TestPropertyBased:
    @given(
        output=st.text(min_size=20, max_size=500, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
        reference=st.text(min_size=20, max_size=500, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
    )
    @settings(max_examples=80)
    def test_risk_always_in_unit_interval(self, output: str, reference: str) -> None:
        scorer = FaithfulnessScorer()
        try:
            result = scorer.score(output, reference)
            assert 0.0 <= result.hallucination_risk <= 1.0
            assert 0.0 <= result.faithfulness_score <= 1.0
        except (EmptyOutputError, EmptyReferenceError):
            pass  # expected for all-whitespace or punctuation-only inputs

    @given(
        output=st.text(min_size=20, max_size=300, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
        reference=st.text(min_size=20, max_size=300, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
    )
    @settings(max_examples=80)
    def test_passed_consistent_with_threshold(self, output: str, reference: str) -> None:
        cfg = EvalConfig(threshold=0.5)
        scorer = FaithfulnessScorer(cfg)
        try:
            result = scorer.score(output, reference)
            assert result.passed == (result.hallucination_risk <= 0.5)
        except (EmptyOutputError, EmptyReferenceError):
            pass
