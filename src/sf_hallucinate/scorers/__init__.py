"""Additional evaluation scorers beyond faithfulness.

Provides:

* :class:`AnswerRelevancyScorer` — rates how well an answer addresses a question.
* :class:`ContextRelevancyScorer` — rates how relevant retrieved context is
  for answering a question.
"""
from sf_hallucinate.scorers.answer_relevancy import AnswerRelevancyScorer
from sf_hallucinate.scorers.context_relevancy import ContextRelevancyScorer

__all__ = ["AnswerRelevancyScorer", "ContextRelevancyScorer"]
