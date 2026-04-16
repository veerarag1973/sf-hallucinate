"""Immutable data-classes that carry scoring results through the pipeline.

All types are ``frozen=True`` so results are safe to cache, hash, and pass
across thread / async boundaries without defensive copying.
"""
from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ClaimResult:
    """Score for a single factual claim extracted from an LLM output.

    Attributes:
        claim:              The claim sentence as extracted from the output.
        best_match:         The reference sentence that best supports (or
                            contradicts) this claim.
        similarity:         Hybrid similarity score in [0.0, 1.0].  Computed
                            as 0.6 × TF-IDF cosine + 0.4 × token F1.
        grounded:           ``True`` when *similarity* >= ``EvalConfig.
                            grounding_threshold``.  Grounded claims are
                            considered factually supported by the reference.
    """

    claim: str
    best_match: str
    similarity: float
    grounded: bool
    contradiction_detected: bool = False
    entailment_label: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ScorerResult:
    """Aggregate hallucination-risk result for one LLM output / reference pair.

    Attributes:
        hallucination_risk:     Float in [0.0, 1.0].  0.0 = fully grounded,
                                1.0 = fully hallucinated.
        faithfulness_score:     ``1.0 - hallucination_risk``.
        grounded_claim_count:   Number of claims where *similarity* >=
                                ``EvalConfig.grounding_threshold``.
        total_claim_count:      Total claims extracted from the LLM output.
        claim_results:          Per-claim breakdown (immutable tuple).
        threshold:              The configured pipeline-failure threshold.
        passed:                 ``True`` when *hallucination_risk* <=
                                *threshold*.
        metadata:               Free-form dict for scorer name, config
                                snapshot, and any extra diagnostics.
    """

    hallucination_risk: float
    faithfulness_score: float
    grounded_claim_count: int
    total_claim_count: int
    claim_results: tuple[ClaimResult, ...]
    threshold: float
    passed: bool
    metadata: dict[str, Any]
    confidence: float = 1.0
    contradiction_count: int = 0

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def grounding_rate(self) -> float:
        """Fraction of claims that are grounded.  1.0 when no claims exist."""
        if self.total_claim_count == 0:
            return 1.0
        return self.grounded_claim_count / self.total_claim_count

    @property
    def ungrounded_claims(self) -> tuple[ClaimResult, ...]:
        """Subset of *claim_results* that are NOT grounded."""
        return tuple(cr for cr in self.claim_results if not cr.grounded)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            "hallucination_risk": self.hallucination_risk,
            "faithfulness_score": self.faithfulness_score,
            "grounded_claim_count": self.grounded_claim_count,
            "total_claim_count": self.total_claim_count,
            "grounding_rate": self.grounding_rate,
            "claim_results": [cr.to_dict() for cr in self.claim_results],
            "threshold": self.threshold,
            "passed": self.passed,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "contradiction_count": self.contradiction_count,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ScorerResult("
            f"risk={self.hallucination_risk:.3f}, "
            f"faithfulness={self.faithfulness_score:.3f}, "
            f"claims={self.grounded_claim_count}/{self.total_claim_count}, "
            f"passed={self.passed})"
        )


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    """Configuration bundle for :class:`~sf_hallucinate.eval.FaithfulnessScorer`.

    Attributes:
        threshold:              Hallucination-risk value in [0.0, 1.0] above
                                which the pipeline is considered to have
                                *failed*.  Default ``0.5``.
        grounding_threshold:    Minimum hybrid similarity for a claim to be
                                labelled *grounded*.  Default ``0.25``.
        min_claim_length:       Minimum character length for a sentence to be
                                treated as a claim.  Shorter strings are
                                skipped.  Default ``15``.
        fail_on_threshold:      When ``True``, :meth:`FaithfulnessScorer.score`
                                raises :exc:`HallucinationRiskExceeded` if the
                                result does not pass.  Default ``False``.
        scorer_name:            Label stored in ``ScorerResult.metadata``.
                                Default ``"faithfulness"``.
        tfidf_weight:           Weight for the TF-IDF cosine component of the
                                hybrid similarity formula (the remainder goes to
                                token F1).  Must be in [0.0, 1.0].
                                Default ``0.6``.
    """

    threshold: float = 0.5
    grounding_threshold: float = 0.25
    min_claim_length: int = 15
    fail_on_threshold: bool = False
    scorer_name: str = "faithfulness"
    tfidf_weight: float = 0.6
    similarity_backend: str = "hybrid"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    language: str = "en"
    detect_contradictions: bool = True

    _VALID_BACKENDS: tuple[str, ...] = dataclasses.field(
        default=("hybrid", "embedding", "llm-nli"),
        init=False,
        repr=False,
        compare=False,
    )
    _VALID_LANGUAGES: tuple[str, ...] = dataclasses.field(
        default=("ar", "de", "en", "es", "fr", "ja", "ko", "pt", "ru", "zh"),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if not 0.0 <= self.grounding_threshold <= 1.0:
            raise ValueError(
                f"grounding_threshold must be in [0, 1], got {self.grounding_threshold}"
            )
        if self.min_claim_length < 1:
            raise ValueError(
                f"min_claim_length must be >= 1, got {self.min_claim_length}"
            )
        if not 0.0 <= self.tfidf_weight <= 1.0:
            raise ValueError(
                f"tfidf_weight must be in [0, 1], got {self.tfidf_weight}"
            )
        if self.similarity_backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"similarity_backend must be one of {self._VALID_BACKENDS}, "
                f"got {self.similarity_backend!r}"
            )
        if self.language not in self._VALID_LANGUAGES:
            raise ValueError(
                f"language must be one of {self._VALID_LANGUAGES}, "
                f"got {self.language!r}"
            )
