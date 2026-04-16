"""Context relevancy scorer — rates how relevant retrieved context is for a question.

Supports three modes via ``EvalConfig.similarity_backend``:

* **hybrid** — lexical overlap between question terms and context (basic).
* **embedding** — cosine similarity between question and context embeddings.
* **llm-nli** — LLM-as-judge relevancy rating (recommended).

The *question* is bound at construction time so the scorer satisfies the
:class:`~sf_hallucinate.eval.EvalScorer` protocol.
"""
from __future__ import annotations

import json

from sf_hallucinate._types import ClaimResult, EvalConfig, ScorerResult
from sf_hallucinate.scoring.languages import tokenize


class ContextRelevancyScorer:
    """Score how relevant the retrieved context is for answering a question.

    Parameters
    ----------
    question:
        The user question the context should help answer.
    config:
        Optional :class:`EvalConfig`.
    """

    def __init__(
        self,
        *,
        question: str,
        config: EvalConfig | None = None,
    ) -> None:
        self.question = question
        self.config: EvalConfig = config if config is not None else EvalConfig()

    @property
    def name(self) -> str:
        return "context_relevancy"

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, output: str, reference: str) -> ScorerResult:
        """Score *reference* (the context) for relevancy to *self.question*.

        Parameters
        ----------
        output:
            LLM answer (available for additional context but not scored).
        reference:
            The retrieved context to evaluate.

        Returns
        -------
        ScorerResult
        """
        backend = self.config.similarity_backend

        if backend == "llm-nli":
            relevancy = self._score_llm(reference)
        elif backend == "embedding":
            relevancy = self._score_embedding(reference)
        else:
            relevancy = self._score_hybrid(reference)

        relevancy = max(0.0, min(1.0, relevancy))
        risk = round(1.0 - relevancy, 6)
        passed = risk <= self.config.threshold

        claim = ClaimResult(
            claim=self.question,
            best_match=reference[:200],
            similarity=round(relevancy, 6),
            grounded=passed,
        )

        return ScorerResult(
            hallucination_risk=risk,
            faithfulness_score=round(relevancy, 6),
            grounded_claim_count=1 if passed else 0,
            total_claim_count=1,
            claim_results=(claim,),
            threshold=self.config.threshold,
            passed=passed,
            metadata={
                "scorer": "context_relevancy",
                "question": self.question,
                "backend": backend,
            },
            confidence=0.9 if backend == "llm-nli" else 0.5,
        )

    def score_batch(
        self,
        outputs: list[str],
        references: list[str],
    ) -> list[ScorerResult]:
        if len(outputs) != len(references):
            raise ValueError(
                f"outputs and references must be the same length, "
                f"got {len(outputs)} vs {len(references)}"
            )
        return [self.score(o, r) for o, r in zip(outputs, references)]

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _score_hybrid(self, reference: str) -> float:
        """Lexical overlap between question and context."""
        lang = self.config.language
        q_tokens = set(
            tokenize(self.question, language=lang, remove_stop_words=True)
        )
        c_tokens = set(
            tokenize(reference, language=lang, remove_stop_words=True)
        )
        if not q_tokens or not c_tokens:
            return 0.0
        common = q_tokens & c_tokens
        recall = len(common) / len(q_tokens)
        precision = len(common) / len(c_tokens)
        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)

    def _score_embedding(self, reference: str) -> float:
        """Embedding cosine similarity between question and context."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Embedding backend requires sentence-transformers. "
                "Install with: pip install sf-hallucinate[embedding]"
            ) from exc

        import math

        model = SentenceTransformer(self.config.embedding_model)
        embeddings = model.encode([self.question, reference], convert_to_numpy=True)
        a, b = embeddings[0], embeddings[1]
        dot = float(sum(ai * bi for ai, bi in zip(a, b)))
        mag_a = math.sqrt(float(sum(ai * ai for ai in a)))
        mag_b = math.sqrt(float(sum(bi * bi for bi in b)))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (mag_a * mag_b)))

    def _score_llm(self, reference: str) -> float:
        """LLM-as-judge context relevancy rating."""
        from sf_hallucinate._llm import call_chat_completion

        system = (
            "You are a context relevancy evaluator.  Rate how relevant "
            "the retrieved context is for answering the given question.  "
            "Respond ONLY with JSON — no markdown fences."
        )
        user = (
            f"Question: {self.question}\n\n"
            f"Context: {reference[:3000]}\n\n"
            "Rate the context's relevancy on a scale from 0.0 to 1.0:\n"
            "- 1.0: Context directly answers the question\n"
            "- 0.7-0.9: Context contains most needed information\n"
            "- 0.4-0.6: Context partially relevant\n"
            "- 0.1-0.3: Context barely relates\n"
            "- 0.0: Context completely irrelevant\n\n"
            'Respond with JSON: {"relevancy_score": <float>}'
        )

        raw = call_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
        )

        try:
            data = json.loads(raw)
            return float(data.get("relevancy_score", 0.5))
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.5
