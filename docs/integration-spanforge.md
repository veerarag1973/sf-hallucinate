# SpanForge Integration

`sf-hallucinate` integrates transparently with SpanForge 2.0.3+. When SpanForge is installed, every `FaithfulnessScorer.score()` call automatically emits a structured event into the SpanForge trace pipeline.

---

## Event emitted

**Event type:** `llm.eval.faithfulness.scored`

**Source:** `sf-hallucinate/faithfulness@1.1.0`

---

## Payload shape (`EvalScenarioPayload`-compatible)

```json
{
  "evaluator":          "sf-hallucinate/faithfulness",
  "score":              0.588,
  "label":              "pass",
  "hallucination_risk": 0.412,
  "grounded_claims":    2,
  "total_claims":       3,
  "grounding_rate":     0.667,
  "threshold":          0.5,
  "confidence":         0.82,
  "contradiction_count": 0,
  "backend":            "hybrid",
  "scorer_config":      { "tfidf_weight": 0.6, "grounding_threshold": 0.25 },
  "output_preview":     "The Eiffel Tower is in Paris...",
  "reference_preview":  "Paris is the capital city of France..."
}
```

`output_preview` and `reference_preview` are truncated to 200 characters to avoid bloating audit logs.

---

## How dispatch works

The emitter uses SpanForge's event dispatch hierarchy:

1. **Active configuration** — if `spanforge.get_config()` returns a non-`None` config, events are dispatched through the active trace context via `spanforge._stream._dispatch()`.
2. **Module-level emit** — falls back to `spanforge.emit(event)` for SpanForge 2.0+ module-level API.

Any exception during emission is silently swallowed — event emission **never** causes a scoring call to fail.

---

## Required dependency

SpanForge is a required dependency, declared in `pyproject.toml`:

```toml
dependencies = ["spanforge>=2.0.3"]
```

SpanForge is always available at runtime — events are emitted on every `score()` call. Emission errors are still swallowed to ensure scoring never fails due to event infrastructure.

---

## Entry-point registration

`sf-hallucinate` registers `FaithfulnessScorer` under the SpanForge scorer discovery group so it is found automatically by SpanForge pipelines:

```toml
# pyproject.toml
[project.entry-points."spanforge.eval_scorers"]
faithfulness = "sf_hallucinate.eval:FaithfulnessScorer"
answer_relevancy = "sf_hallucinate.scorers.answer_relevancy:AnswerRelevancyScorer"
context_relevancy = "sf_hallucinate.scorers.context_relevancy:ContextRelevancyScorer"
```

SpanForge scans this group at runtime to build its scorer registry. Once `sf-hallucinate` is installed, SpanForge-managed pipelines can reference `"faithfulness"`, `"answer_relevancy"`, or `"context_relevancy"` by name without any manual wiring.

---

## Viewing events

With SpanForge 2.0.3, events appear in:

- **Trace viewer** — under the `llm.eval.*` namespace
- **Audit chain** — each event is hashed and appended to the current run's evidence package
- **Compliance reports** — `label: "pass" / "fail"` feeds directly into SpanForge's policy engine

---

## Example: explicit SpanForge context

```python
import spanforge
from sf_hallucinate.eval import FaithfulnessScorer

# Activate a SpanForge config with your trace endpoint
spanforge.configure(endpoint="https://my-trace-server/ingest", api_key="...")

scorer = FaithfulnessScorer()
result = scorer.score(llm_output, reference_doc)
# ↑ automatically emits llm.eval.faithfulness.scored to your trace endpoint
```

---

## Disabling emission

If you want to suppress SpanForge events temporarily (e.g. in unit tests), uninstall `spanforge` from the test environment, or mock the emitter:

```python
from unittest import mock
import sf_hallucinate.eval as eval_mod

with mock.patch.object(eval_mod.FaithfulnessScorer, "_emit", lambda *a, **kw: None):
    result = scorer.score(output, reference)
```
