"""Custom exception hierarchy for sf-hallucinate."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sf_hallucinate._types import ScorerResult


class SfHallucinateError(Exception):
    """Base class for all sf-hallucinate exceptions."""


class HallucinationRiskExceeded(SfHallucinateError):
    """Raised by :class:`~sf_hallucinate.eval.FaithfulnessScorer` when
    ``EvalConfig.fail_on_threshold`` is ``True`` and the computed
    ``hallucination_risk`` exceeds the configured ``threshold``.

    The full :class:`~sf_hallucinate._types.ScorerResult` is attached so
    callers can inspect per-claim details.

    Example::

        from sf_hallucinate import FaithfulnessScorer, EvalConfig
        from sf_hallucinate import HallucinationRiskExceeded

        scorer = FaithfulnessScorer(
            EvalConfig(threshold=0.3, fail_on_threshold=True)
        )
        try:
            result = scorer.score(llm_output, reference)
        except HallucinationRiskExceeded as exc:
            print(exc.result.hallucination_risk)
            print(exc.result.ungrounded_claims)
            raise SystemExit(1)
    """

    def __init__(self, result: "ScorerResult") -> None:
        self.result = result
        super().__init__(
            f"Hallucination risk {result.hallucination_risk:.3f} exceeds "
            f"threshold {result.threshold:.3f} "
            f"({result.grounded_claim_count}/{result.total_claim_count} claims grounded)"
        )


class EmptyReferenceError(SfHallucinateError):
    """Raised when an empty string is supplied as the reference document."""

    def __init__(self) -> None:
        super().__init__(
            "Reference document must not be empty. "
            "Faithfulness scoring requires a non-empty reference to check claims against."
        )


class EmptyOutputError(SfHallucinateError):
    """Raised when an empty string is supplied as the LLM output to score."""

    def __init__(self) -> None:
        super().__init__(
            "LLM output must not be empty. "
            "There are no claims to extract from an empty string."
        )
