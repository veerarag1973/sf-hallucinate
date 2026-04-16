"""SpanForge ``llm.eval.*`` event emission for faithfulness scoring.

Bridges :class:`~sf_hallucinate._types.ScorerResult` to SpanForge's
``llm.eval.*`` event namespace (``EvalScenarioPayload`` shape) so results
appear in trace viewers, audit chains, and compliance evidence packages.

Event type emitted
------------------
``llm.eval.faithfulness.scored``

Payload shape (``EvalScenarioPayload``-compatible)
--------------------------------------------------
::

    {
        "scenario_id":       str,    # ULID from SpanForge
        "evaluator":         str,    # "sf-hallucinate/faithfulness"
        "score":             float,  # faithfulness_score (0→1)
        "label":             str,    # "pass" | "fail"
        "hallucination_risk": float,
        "grounded_claims":   int,
        "total_claims":      int,
        "grounding_rate":    float,
        "threshold":         float,
        "scorer_config":     dict,
    }

Usage
-----
The emitter is called automatically by :class:`~sf_hallucinate.eval.FaithfulnessScorer`
after every :meth:`~sf_hallucinate.eval.FaithfulnessScorer.score` call.  You
do not need to call it directly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sf_hallucinate._types import ScorerResult

import spanforge  # type: ignore[import-untyped]
from spanforge import Event  # type: ignore[import-untyped]

_EVAL_EVENT_TYPE = "llm.eval.faithfulness.scored"
_EVALUATOR_ID = "sf-hallucinate/faithfulness"


def emit_eval_event(
    *,
    output: str,
    reference: str,
    result: "ScorerResult",
) -> None:
    """Emit an ``llm.eval.faithfulness.scored`` SpanForge event.

    Parameters
    ----------
    output:
        The LLM-generated text that was scored.
    reference:
        The reference document used for grounding.
    result:
        The :class:`~sf_hallucinate._types.ScorerResult` from the scorer.
    """
    try:
        _do_emit(output=output, reference=reference, result=result)
    except Exception:  # noqa: BLE001
        # Never let event emission fail a scoring call.
        pass


def _do_emit(
    *,
    output: str,
    reference: str,
    result: "ScorerResult",
) -> None:
    """Internal emitter — may raise; caller wraps in try/except."""
    payload: dict[str, Any] = {
        "evaluator": _EVALUATOR_ID,
        "score": result.faithfulness_score,
        "label": "pass" if result.passed else "fail",
        "hallucination_risk": result.hallucination_risk,
        "grounded_claims": result.grounded_claim_count,
        "total_claims": result.total_claim_count,
        "grounding_rate": result.grounding_rate,
        "threshold": result.threshold,
        "scorer_config": result.metadata.get("config", {}),
        # Truncate strings to avoid bloating the audit log
        "output_preview": output[:200] if output else "",
        "reference_preview": reference[:200] if reference else "",
    }

    event = Event(
        event_type=_EVAL_EVENT_TYPE,
        source=f"{_EVALUATOR_ID}@1.0.0",
        payload=payload,
    )

    # Use the public module-level emit helper (SpanForge 2.0+)
    spanforge.emit(event)
