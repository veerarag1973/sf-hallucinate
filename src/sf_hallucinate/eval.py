"""SpanForge-compatible EvalScorer protocol and built-in FaithfulnessScorer.

Platform source: ``eval.py → EvalScorer protocol``

Overview
--------
This module defines:

* :class:`EvalScorer` — a :pep:`544` ``Protocol`` that all scorers must satisfy.
  Aligned with the SpanForge ``llm.eval.*`` namespace so any scorer built here
  integrates transparently into a SpanForge pipeline.

* :class:`FaithfulnessScorer` — built-in scorer that implements the faithfulness
  evaluation strategy:

  1. **Claim extraction** — every declarative sentence in the LLM output is
     treated as a factual claim.
  2. **Sentence similarity** — each claim is checked against every sentence in
     the reference document using a hybrid TF-IDF cosine + token-F1 metric.
  3. **Grounding decision** — a claim is *grounded* when its best-match
     similarity meets the configurable ``grounding_threshold``.
  4. **Aggregate scoring** — ``faithfulness_score = mean(best-match scores)``;
     ``hallucination_risk = 1 − faithfulness_score``.
  5. **Pipeline gate** — optionally raises :exc:`HallucinationRiskExceeded`
     when ``hallucination_risk`` exceeds the configured ``threshold``.

* :class:`EvalPipeline` — chains multiple :class:`EvalScorer` instances,
  returns a mapping of scorer-name → :class:`~sf_hallucinate._types.ScorerResult`.

SpanForge integration
---------------------
The scorer automatically emits an ``llm.eval.faithfulness.scored`` event
after every :meth:`FaithfulnessScorer.score` call.  This integrates seamlessly
with SpanForge trace contexts, audit chains, and compliance evidence packages.
See :mod:`sf_hallucinate.integration.spanforge` for details.

Dependencies
------------
Requires ``spanforge>=2.0.3`` as the sole runtime dependency.  All scoring
algorithms use the Python standard library only.
"""
from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, Protocol, runtime_checkable

from sf_hallucinate._exceptions import (
    EmptyOutputError,
    EmptyReferenceError,
    HallucinationRiskExceeded,
)
from sf_hallucinate._types import ClaimResult, EvalConfig, ScorerResult
from sf_hallucinate.integration.spanforge import emit_eval_event
from sf_hallucinate.scoring.backends import ClaimScore, create_backend
from sf_hallucinate.scoring.claims import extract_claims, split_sentences


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EvalScorer(Protocol):
    """Protocol that every scorer must satisfy.

    SpanForge discovers scorers registered under the
    ``spanforge.eval_scorers`` entry-point group.  Any class that implements
    this protocol can be registered without subclassing a base class.

    Attributes
    ----------
    name:
        Short human-readable identifier used in log messages and
        ``ScorerResult.metadata``.
    config:
        :class:`~sf_hallucinate._types.EvalConfig` instance controlling
        thresholds and algorithm parameters.
    """

    name: str
    config: EvalConfig

    def score(self, output: str, reference: str) -> ScorerResult:
        """Score *output* against *reference*, returning a :class:`ScorerResult`.

        Parameters
        ----------
        output:
            The LLM-generated text to evaluate.
        reference:
            The ground-truth / source document against which claims are
            checked.

        Returns
        -------
        ScorerResult
            Full result including per-claim breakdown and aggregate risk.

        Raises
        ------
        EmptyOutputError
            When *output* is blank.
        EmptyReferenceError
            When *reference* is blank.
        HallucinationRiskExceeded
            When ``config.fail_on_threshold`` is ``True`` and the result
            does not pass.
        """
        ...  # pragma: no cover

    def score_batch(
        self,
        outputs: list[str],
        references: list[str],
    ) -> list[ScorerResult]:
        """Score multiple output/reference pairs.

        Parameters
        ----------
        outputs:
            List of LLM-generated texts.
        references:
            Corresponding reference documents — must be the same length.

        Returns
        -------
        list[ScorerResult]
            One result per pair, in the same order.

        Raises
        ------
        ValueError
            When *outputs* and *references* have different lengths.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Built-in scorer
# ---------------------------------------------------------------------------

class FaithfulnessScorer:
    """Scores LLM outputs for faithfulness against a reference document.

    Implements the :class:`EvalScorer` protocol.  Usable standalone or
    embedded inside a SpanForge trace/pipeline.

    Parameters
    ----------
    config:
        Optional :class:`~sf_hallucinate._types.EvalConfig`.  When omitted,
        sensible defaults are used (``threshold=0.5``,
        ``grounding_threshold=0.25``).

    Examples
    --------
    Basic usage::

        from sf_hallucinate import FaithfulnessScorer, EvalConfig

        scorer = FaithfulnessScorer(EvalConfig(threshold=0.4))
        result = scorer.score(
            output="The Eiffel Tower is in Berlin.",
            reference="The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        )
        print(result.hallucination_risk)   # high — Berlin ≠ Paris
        print(result.passed)               # False

    Pipeline gate::

        from sf_hallucinate import FaithfulnessScorer, EvalConfig, HallucinationRiskExceeded

        scorer = FaithfulnessScorer(EvalConfig(fail_on_threshold=True))
        try:
            result = scorer.score(llm_output, reference_doc)
        except HallucinationRiskExceeded as exc:
            # Fail the pipeline
            raise SystemExit(1) from exc

    Async usage::

        result = await scorer.ascore(output, reference)

    Batch evaluation::

        results = scorer.score_batch(outputs, references)
    """

    # `name` is exposed as a property so EvalPipeline keying respects
    # EvalConfig.scorer_name overrides (e.g. when the same scorer type is
    # registered under different names in a multi-scorer pipeline).
    @property
    def name(self) -> str:  # type: ignore[override]
        """Scorer identifier — defaults to ``config.scorer_name``."""
        return self.config.scorer_name

    def __init__(self, config: EvalConfig | None = None) -> None:
        self.config: EvalConfig = config if config is not None else EvalConfig()

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score(self, output: str, reference: str) -> ScorerResult:
        """Score *output* against *reference*.

        Implements :meth:`EvalScorer.score`.
        """
        if not output or not output.strip():
            raise EmptyOutputError()
        if not reference or not reference.strip():
            raise EmptyReferenceError()

        claims = extract_claims(output, min_length=self.config.min_claim_length)
        ref_sentences = split_sentences(reference)

        claim_results = self._score_claims(claims, ref_sentences)
        result = self._aggregate(claim_results)

        # Emit a SpanForge llm.eval.* event
        emit_eval_event(output=output, reference=reference, result=result)

        if self.config.fail_on_threshold and not result.passed:
            raise HallucinationRiskExceeded(result)

        return result

    async def ascore(self, output: str, reference: str) -> ScorerResult:
        """Async wrapper around :meth:`score`.

        Offloads the CPU-bound computation to a thread-pool executor so it
        does not block the event loop.

        Parameters
        ----------
        output:
            LLM-generated text to evaluate.
        reference:
            Source / ground-truth document.

        Returns
        -------
        ScorerResult
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.score, output, reference)

    def score_batch(
        self,
        outputs: list[str],
        references: list[str],
    ) -> list[ScorerResult]:
        """Score multiple output/reference pairs.

        Implements :meth:`EvalScorer.score_batch`.

        Parameters
        ----------
        outputs:
            LLM-generated texts.
        references:
            Corresponding reference documents (same length as *outputs*).

        Raises
        ------
        ValueError
            When lists are not the same length.
        """
        if len(outputs) != len(references):
            raise ValueError(
                f"outputs and references must be the same length, "
                f"got {len(outputs)} vs {len(references)}"
            )
        return [self.score(o, r) for o, r in zip(outputs, references)]

    async def ascore_batch(
        self,
        outputs: list[str],
        references: list[str],
    ) -> list[ScorerResult]:
        """Async batch scoring — runs all pairs concurrently.

        Parameters
        ----------
        outputs:
            LLM-generated texts.
        references:
            Corresponding reference documents.

        Returns
        -------
        list[ScorerResult]
            In the same order as *outputs*.
        """
        if len(outputs) != len(references):
            raise ValueError(
                f"outputs and references must be the same length, "
                f"got {len(outputs)} vs {len(references)}"
            )
        coros = [self.ascore(o, r) for o, r in zip(outputs, references)]
        return list(await asyncio.gather(*coros))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_claims(
        self,
        claims: list[str],
        ref_sentences: list[str],
    ) -> list[ClaimResult]:
        backend = create_backend(self.config)

        # Batch scoring path for backends that support it (e.g. LLM NLI)
        if hasattr(backend, "score_claims_batch"):
            claim_scores: list[ClaimScore] = backend.score_claims_batch(
                claims, ref_sentences
            )
        else:
            claim_scores = [
                backend.score_claim(c, ref_sentences) for c in claims
            ]

        results: list[ClaimResult] = []
        for claim, cs in zip(claims, claim_scores):
            # Contradiction overrides grounding
            grounded = (
                cs.similarity >= self.config.grounding_threshold
                and not cs.contradiction_detected
            )
            results.append(
                ClaimResult(
                    claim=claim,
                    best_match=cs.best_match,
                    similarity=round(cs.similarity, 6),
                    grounded=grounded,
                    contradiction_detected=cs.contradiction_detected,
                    entailment_label=cs.entailment_label,
                    confidence=round(cs.confidence, 6),
                )
            )
        return results

    def _aggregate(self, claim_results: list[ClaimResult]) -> ScorerResult:
        if not claim_results:
            # No claims extracted → benefit of the doubt (fully grounded)
            faithfulness = 1.0
            risk = 0.0
            grounded_count = 0
            contradiction_count = 0
            confidence = 1.0
        else:
            total = len(claim_results)
            faithfulness = sum(cr.similarity for cr in claim_results) / total
            faithfulness = min(1.0, max(0.0, faithfulness))
            risk = round(1.0 - faithfulness, 6)
            grounded_count = sum(1 for cr in claim_results if cr.grounded)
            contradiction_count = sum(
                1 for cr in claim_results if cr.contradiction_detected
            )
            confidence = self._compute_confidence(claim_results)

        passed = risk <= self.config.threshold

        # Redact sensitive fields from config snapshot
        config_dict = dataclasses.asdict(self.config)
        for key in ("llm_api_key", "_VALID_BACKENDS", "_VALID_LANGUAGES"):
            config_dict.pop(key, None)

        return ScorerResult(
            hallucination_risk=risk,
            faithfulness_score=round(1.0 - risk, 6),
            grounded_claim_count=grounded_count,
            total_claim_count=len(claim_results),
            claim_results=tuple(claim_results),
            threshold=self.config.threshold,
            passed=passed,
            metadata={
                "scorer": self.config.scorer_name,
                "config": config_dict,
            },
            confidence=round(confidence, 4),
            contradiction_count=contradiction_count,
        )

    @staticmethod
    def _compute_confidence(claim_results: list[ClaimResult]) -> float:
        """Calibrate confidence in the aggregate score.

        Combines three signals:
        * **count factor** — more claims → more reliable.
        * **consistency** — low variance in scores → more confident.
        * **backend confidence** — average per-claim confidence.
        """
        n = len(claim_results)
        if n == 0:
            return 1.0

        sims = [cr.similarity for cr in claim_results]
        mean_sim = sum(sims) / n

        # Factor 1: claim count (saturates at 5 claims)
        count_factor = min(1.0, n / 5.0)

        # Factor 2: consistency (inverse variance)
        variance = sum((s - mean_sim) ** 2 for s in sims) / n
        consistency = 1.0 / (1.0 + variance * 4.0)

        # Factor 3: per-claim backend confidence
        backend_conf = sum(cr.confidence for cr in claim_results) / n

        return count_factor * 0.3 + consistency * 0.3 + backend_conf * 0.4

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FaithfulnessScorer("
            f"threshold={self.config.threshold}, "
            f"grounding_threshold={self.config.grounding_threshold})"
        )


# ---------------------------------------------------------------------------
# EvalPipeline
# ---------------------------------------------------------------------------

class EvalPipeline:
    """Chains multiple :class:`EvalScorer` instances into one evaluation pass.

    Each scorer sees the same *output* / *reference* pair.  Results are
    collected in a dict keyed by scorer name.  The pipeline as a whole
    *passes* only when **every** scorer passes.

    Parameters
    ----------
    scorers:
        One or more :class:`EvalScorer`-compatible objects.

    Examples
    --------
    ::

        from sf_hallucinate import FaithfulnessScorer, EvalConfig
        from sf_hallucinate.eval import EvalPipeline

        pipeline = EvalPipeline(
            FaithfulnessScorer(EvalConfig(threshold=0.4)),
        )
        results = pipeline.run(output, reference)
        print(results["faithfulness"].passed)
    """

    def __init__(self, *scorers: EvalScorer) -> None:
        if not scorers:
            raise ValueError("EvalPipeline requires at least one scorer.")
        self._scorers = scorers
        self._last_results: dict[str, ScorerResult] | None = None

    def run(
        self,
        output: str,
        reference: str,
    ) -> dict[str, ScorerResult]:
        """Run all scorers and return a name → result mapping.

        Parameters
        ----------
        output:
            LLM-generated text.
        reference:
            Reference document.

        Returns
        -------
        dict[str, ScorerResult]
            One entry per scorer.

        Raises
        ------
        HallucinationRiskExceeded
            Propagated from any scorer whose ``fail_on_threshold`` is
            ``True`` and whose result does not pass.
        """
        self._last_results = {
            s.name: s.score(output, reference) for s in self._scorers
        }
        return self._last_results

    @property
    def passed(self) -> bool:
        """``True`` when the most recent :meth:`run` had all scorers passing.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called yet.
        """
        if self._last_results is None:
            raise RuntimeError("Call run() before checking passed.")
        return all(r.passed for r in self._last_results.values())



