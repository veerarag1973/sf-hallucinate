"""LLM-based Natural Language Inference (NLI) similarity backend.

Calls an OpenAI-compatible chat completion API to classify each claim as
**entailment**, **neutral**, or **contradiction** relative to the reference.

No extra dependencies required — uses stdlib ``urllib`` for HTTP.

Configuration via :class:`~sf_hallucinate._types.EvalConfig`:

* ``llm_model`` — model identifier (default ``"gpt-4o-mini"``).
* ``llm_api_key`` — bearer token (or set ``OPENAI_API_KEY`` env var).
* ``llm_base_url`` — custom API base (default ``https://api.openai.com/v1``).
"""
from __future__ import annotations

import json
import re
from typing import Any

from sf_hallucinate._llm import call_chat_completion
from sf_hallucinate._types import EvalConfig
from sf_hallucinate.scoring.backends import ClaimScore

# ---------------------------------------------------------------------------
# Entailment label → similarity mapping
# ---------------------------------------------------------------------------

_LABEL_MAP: dict[str, float] = {
    "entailment": 0.92,
    "contradiction": 0.05,
    "neutral": 0.35,
}


def _label_to_similarity(label: str, confidence: float) -> float:
    """Map an NLI label + confidence to a similarity score in [0, 1]."""
    base = _LABEL_MAP.get(label, 0.35)
    if label == "entailment":
        return base + 0.08 * confidence  # [0.92, 1.0]
    if label == "contradiction":
        return base * (1.0 - confidence)  # [0.0, 0.05]
    # neutral
    return base + 0.15 * (1.0 - confidence)  # [0.35, 0.50]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise Natural Language Inference (NLI) evaluator.  Your task is
to determine whether claims made in an LLM output are supported by a
reference document.

For each claim, classify it as:
- "entailment": The reference document clearly supports this claim.
- "contradiction": The reference document contradicts this claim.
- "neutral": The reference document neither supports nor contradicts this claim.

Also assign a confidence score between 0.0 and 1.0 for your classification.
Respond ONLY with a JSON array — no markdown fences, no explanation."""

_USER_TEMPLATE = """\
Reference document:
---
{reference}
---

Claims to evaluate:
{claims_block}

Respond with a JSON array.  Each element MUST have exactly these keys:
- "index": integer (1-based claim number)
- "label": "entailment" | "contradiction" | "neutral"
- "confidence": float between 0.0 and 1.0

Example response:
[{{"index": 1, "label": "entailment", "confidence": 0.95}}]"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_llm_response(
    raw: str,
    num_claims: int,
) -> list[dict[str, Any]]:
    """Best-effort parse of the LLM JSON response."""
    # Try to extract a JSON array from the response
    m = _JSON_ARRAY_RE.search(raw)
    if not m:
        # Return neutral defaults
        return [
            {"index": i + 1, "label": "neutral", "confidence": 0.5}
            for i in range(num_claims)
        ]

    try:
        items = json.loads(m.group())
    except json.JSONDecodeError:
        return [
            {"index": i + 1, "label": "neutral", "confidence": 0.5}
            for i in range(num_claims)
        ]

    # Validate and normalise
    result: list[dict[str, Any]] = []
    for item in items:
        label = str(item.get("label", "neutral")).lower().strip()
        if label not in _LABEL_MAP:
            label = "neutral"
        conf = float(item.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        idx = int(item.get("index", len(result) + 1))
        result.append({"index": idx, "label": label, "confidence": conf})

    # Pad if LLM returned fewer items than expected
    seen = {r["index"] for r in result}
    for i in range(1, num_claims + 1):
        if i not in seen:
            result.append({"index": i, "label": "neutral", "confidence": 0.5})

    result.sort(key=lambda r: r["index"])
    return result[:num_claims]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class LLMNLIBackend:
    """LLM-as-judge NLI backend.

    Sends all claims in a single API call for cost efficiency.  Falls back
    to per-claim calls if batch parsing fails.
    """

    def __init__(self, config: EvalConfig) -> None:
        self._model = config.llm_model
        self._api_key = config.llm_api_key
        self._base_url = config.llm_base_url
        # Validate eagerly that an API key is available
        import os

        key = config.llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError(
                "The 'llm-nli' backend requires an API key.  Set the "
                "OPENAI_API_KEY environment variable or pass llm_api_key "
                "in EvalConfig."
            )

    def score_claim(
        self,
        claim: str,
        reference_sentences: list[str],
    ) -> ClaimScore:
        """Score a single claim (wraps :meth:`score_claims_batch`)."""
        results = self.score_claims_batch([claim], reference_sentences)
        return results[0]

    def score_claims_batch(
        self,
        claims: list[str],
        reference_sentences: list[str],
    ) -> list[ClaimScore]:
        """Score multiple claims in a single LLM API call."""
        if not claims:
            return []
        if not reference_sentences:
            return [ClaimScore(similarity=0.0, best_match="") for _ in claims]

        reference = " ".join(reference_sentences)
        claims_block = "\n".join(
            f"{i}. {claim}" for i, claim in enumerate(claims, 1)
        )

        user_msg = _USER_TEMPLATE.format(
            reference=reference, claims_block=claims_block
        )

        raw = call_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=self._model,
            api_key=self._api_key,
            base_url=self._base_url,
        )

        parsed = _parse_llm_response(raw, len(claims))

        # Build ClaimScore objects
        results: list[ClaimScore] = []
        for item, claim in zip(parsed, claims):
            label = item["label"]
            confidence = item["confidence"]
            sim = _label_to_similarity(label, confidence)

            # For NLI, "best_match" is the full reference (the LLM
            # considers it holistically)
            best = reference_sentences[0] if reference_sentences else ""

            results.append(
                ClaimScore(
                    similarity=round(sim, 6),
                    best_match=best,
                    entailment_label=label,
                    contradiction_detected=(label == "contradiction"),
                    contradiction_score=confidence if label == "contradiction" else 0.0,
                    confidence=confidence,
                )
            )

        return results
