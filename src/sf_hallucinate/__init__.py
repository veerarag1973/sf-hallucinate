"""sf-hallucinate — faithfulness scorer for LLM outputs.

Public API surface:

    from sf_hallucinate import FaithfulnessScorer, EvalConfig, ScorerResult
    from sf_hallucinate import ClaimResult, HallucinationRiskExceeded
    from sf_hallucinate.eval import EvalScorer            # Protocol
    from sf_hallucinate.scorers import AnswerRelevancyScorer, ContextRelevancyScorer
"""
from __future__ import annotations

from sf_hallucinate._exceptions import HallucinationRiskExceeded
from sf_hallucinate._types import ClaimResult, EvalConfig, ScorerResult
from sf_hallucinate.eval import EvalScorer, FaithfulnessScorer
from sf_hallucinate.scorers import AnswerRelevancyScorer, ContextRelevancyScorer

__all__ = [
    # protocol
    "EvalScorer",
    # scorers
    "FaithfulnessScorer",
    "AnswerRelevancyScorer",
    "ContextRelevancyScorer",
    # config / result types
    "EvalConfig",
    "ScorerResult",
    "ClaimResult",
    # exceptions
    "HallucinationRiskExceeded",
]

__version__ = "1.1.0"
