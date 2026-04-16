"""SpanForge integration package.

Emits ``llm.eval.faithfulness.scored`` events into the active SpanForge trace
context, connecting hallucination risk scores to the full compliance pipeline.

Requires ``spanforge>=2.0.3`` (declared as a mandatory dependency in
``pyproject.toml``).
"""
from sf_hallucinate.integration.spanforge import emit_eval_event

__all__ = ["emit_eval_event"]
