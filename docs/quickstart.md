# Quickstart

Get up and running with `sf-hallucinate` in under 5 minutes.

---

## Installation

```bash
pip install sf-hallucinate
```

Requires Python 3.9+. Installs `spanforge>=2.0.3` automatically.

### Optional extras

```bash
# Embedding similarity backend
pip install sf-hallucinate[embedding]

# All optional dependencies
pip install sf-hallucinate[all]
```

---

## 1. Score a single output

```python
from sf_hallucinate.eval import FaithfulnessScorer
from sf_hallucinate._types import EvalConfig

reference = """
Paris is the capital city of France. The Eiffel Tower was completed in 1889
and stands 330 metres tall. It is made of iron and receives about 7 million
visitors per year.
"""

output = """
The Eiffel Tower is located in Paris, France. Built in the late 19th century,
the iron structure is one of the most visited monuments in the world.
"""

scorer = FaithfulnessScorer()            # default threshold = 0.5
result = scorer.score(output, reference)

print(f"Hallucination risk : {result.hallucination_risk:.3f}")
print(f"Faithfulness score : {result.faithfulness_score:.3f}")
print(f"Grounded claims    : {result.grounded_claim_count}/{result.total_claim_count}")
print(f"Passed             : {result.passed}")
```

---

## 2. Inspect per-claim results

```python
for claim_result in result.claim_results:
    status = "✓" if claim_result.grounded else "✗"
    print(f"  {status} [{claim_result.similarity:.2f}] {claim_result.claim}")
    print(f"       → {claim_result.best_match}")
```

---

## 3. Fail the pipeline on high risk

```python
from sf_hallucinate._exceptions import HallucinationRiskExceeded
from sf_hallucinate._types import EvalConfig

config = EvalConfig(threshold=0.3, fail_on_threshold=True)
scorer = FaithfulnessScorer(config)

try:
    result = scorer.score(output, reference)
except HallucinationRiskExceeded as exc:
    print(f"Pipeline blocked! Risk = {exc.result.hallucination_risk:.3f}")
```

---

## 4. Run multiple scorers with EvalPipeline

```python
from sf_hallucinate.eval import EvalPipeline, FaithfulnessScorer
from sf_hallucinate._types import EvalConfig

strict  = FaithfulnessScorer(EvalConfig(threshold=0.3, scorer_name="strict"))
lenient = FaithfulnessScorer(EvalConfig(threshold=0.7, scorer_name="lenient"))

pipeline = EvalPipeline(strict, lenient)
results  = pipeline.run(output, reference)

for name, result in results.items():
    print(f"{name}: risk={result.hallucination_risk:.3f}  passed={result.passed}")
```

---

## 5. Batch scoring

```python
outputs    = ["Output A ...", "Output B ...", "Output C ..."]
references = ["Reference A ...", "Reference B ...", "Reference C ..."]

results = scorer.score_batch(outputs, references)
for r in results:
    print(r.passed, r.hallucination_risk)
```

---

## 6. Async usage

```python
import asyncio
from sf_hallucinate.eval import FaithfulnessScorer

async def evaluate():
    scorer = FaithfulnessScorer()
    result = await scorer.ascore(output, reference)
    print(result.passed)

asyncio.run(evaluate())
```

---

## 7. CLI — quick check

```bash
sf-hallucinate score \
    --output "The Eiffel Tower is in Berlin." \
    --reference "The Eiffel Tower is in Paris, France." \
    --threshold 0.4
```

Exit code `0` = passed, `1` = failed or error.

---

## 8. Pluggable backends

```python
from sf_hallucinate.eval import FaithfulnessScorer
from sf_hallucinate._types import EvalConfig

# Embedding similarity (requires sf-hallucinate[embedding])
config = EvalConfig(similarity_backend="embedding", embedding_model="all-MiniLM-L6-v2")
scorer = FaithfulnessScorer(config)
result = scorer.score(output, reference)

# LLM-NLI (requires an OpenAI-compatible API key)
config = EvalConfig(
    similarity_backend="llm-nli",
    llm_model="gpt-4o-mini",
    llm_api_key="sk-...",
)
scorer = FaithfulnessScorer(config)
result = scorer.score(output, reference)
```

---

## 9. Multi-language scoring

```python
config = EvalConfig(language="es")
scorer = FaithfulnessScorer(config)

result = scorer.score(
    "La Torre Eiffel está en París.",
    "La Torre Eiffel está ubicada en París, Francia.",
)
print(result.faithfulness_score)
```

Supported languages: `ar`, `de`, `en`, `es`, `fr`, `ja`, `ko`, `pt`, `ru`, `zh`.

---

## 10. Contradiction detection

Contradiction detection is enabled by default. Inspect results:

```python
config = EvalConfig(detect_contradictions=True)
scorer = FaithfulnessScorer(config)

result = scorer.score(
    "The tower is 200 metres tall.",
    "The Eiffel Tower stands 330 metres tall.",
)

print(f"Contradictions found: {result.contradiction_count}")
for cr in result.claim_results:
    if cr.contradiction_detected:
        print(f"  ⚠ {cr.claim}")
```

Disable with `EvalConfig(detect_contradictions=False)` or `--no-contradiction-detection` on the CLI.

---

## 11. Confidence scores

Every `ScorerResult` includes a calibrated confidence:

```python
result = scorer.score(output, reference)
print(f"Confidence: {result.confidence:.2f}")
```

Higher confidence means more claims, consistent scores, and higher backend confidence.

---

## 12. Answer & Context Relevancy scorers

```python
from sf_hallucinate.scorers.answer_relevancy import AnswerRelevancyScorer
from sf_hallucinate.scorers.context_relevancy import ContextRelevancyScorer

question = "What is the height of the Eiffel Tower?"

answer_scorer  = AnswerRelevancyScorer(question=question)
context_scorer = ContextRelevancyScorer(question=question)

answer_result  = answer_scorer.score(output, reference)
context_result = context_scorer.score(output, reference)

print(f"Answer relevancy : {answer_result.faithfulness_score:.3f}")
print(f"Context relevancy: {context_result.faithfulness_score:.3f}")
```

---

## Next steps

- [API Reference](api-reference.md) — full class and method documentation
- [CLI Reference](cli.md) — all sub-commands and flags
- [Algorithms](algorithms.md) — understand the scoring math
- [SpanForge Integration](integration-spanforge.md) — trace events and entry-points
