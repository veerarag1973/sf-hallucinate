# sf-hallucinate

**Hallucination detection and faithfulness scoring for LLM outputs.**

`sf-hallucinate` is a Python library (and CLI tool) that scores how *faithfully* an LLM-generated answer is grounded in a reference document. Built on top of [SpanForge 2.0.3](https://github.com/spanforge/spanforge), it emits structured `llm.eval.*` trace events, integrates into evaluation pipelines, and provides a standalone command-line interface for CI/CD usage.

---

## Features

| Feature | Detail |
|---|---|
| **EvalScorer Protocol** | PEP 544 `@runtime_checkable` protocol — plug in any custom scorer |
| **FaithfulnessScorer** | Built-in claim-by-claim faithfulness grading |
| **Pluggable backends** | `hybrid` (default), `embedding`, `llm-nli` — swap via `EvalConfig.similarity_backend` |
| **Embedding similarity** | Dense cosine similarity using `sentence-transformers` |
| **LLM-NLI backend** | Natural Language Inference via any OpenAI-compatible API |
| **Contradiction detection** | Heuristic detection of negation, antonym, and numeric contradictions |
| **Confidence calibration** | Per-result confidence score from claim count, consistency, and backend signals |
| **Multi-language** | 10 languages (ar, de, en, es, fr, ja, ko, pt, ru, zh) with CJK bigram tokenization |
| **Answer relevancy** | `AnswerRelevancyScorer` — rates how well an answer addresses a question |
| **Context relevancy** | `ContextRelevancyScorer` — rates how relevant retrieved context is |
| **Hybrid similarity** | 0.6 × TF-IDF cosine + 0.4 × token F1 |
| **Claim extraction** | Abbreviation-aware sentence splitter; filters questions and meta-sentences |
| **Pipeline gate** | Optionally raises `HallucinationRiskExceeded` on threshold breach |
| **EvalPipeline** | Chains multiple scorers, returns `dict[name → ScorerResult]` |
| **SpanForge integration** | Emits `llm.eval.faithfulness.scored` events automatically |
| **CLI** | `score`, `score-file`, `batch` sub-commands with human/JSON/JSONL output |
| **Async API** | `ascore()` / `ascore_batch()` for async LLM pipelines |
| **Minimal dependencies** | Only requires `spanforge>=2.0.3`; optional extras for embedding and LLM backends |

---

## Quick installation

```bash
pip install sf-hallucinate

# With embedding backend support
pip install sf-hallucinate[embedding]

# All optional dependencies
pip install sf-hallucinate[all]
```

Or from source:

```bash
git clone https://github.com/veerarag1973/sf-hallucinate
cd sf-hallucinate
pip install -e ".[all]"
```

---

## 30-second example

```python
from sf_hallucinate.eval import FaithfulnessScorer
from sf_hallucinate._types import EvalConfig

reference = """
Paris is the capital city of France, located in northern France.
The Eiffel Tower, completed in 1889, stands 330 metres tall and is
made of iron. It receives about 7 million visitors annually.
"""

output = """
The Eiffel Tower is located in Paris, France. Built in the late 19th century,
the iron structure is one of the most visited monuments in the world.
"""

scorer = FaithfulnessScorer(EvalConfig(threshold=0.5))
result = scorer.score(output, reference)

print(f"Risk:         {result.hallucination_risk:.3f}")
print(f"Faithfulness: {result.faithfulness_score:.3f}")
print(f"Passed:       {result.passed}")
```

---

## Documentation

| Page | Description |
|---|---|
| [Quickstart](quickstart.md) | 5-minute walkthrough with code examples |
| [API Reference](api-reference.md) | All public classes, methods, and types |
| [Algorithms](algorithms.md) | Hybrid similarity, backends, contradiction detection, confidence calibration |
| [CLI Reference](cli.md) | All sub-commands, flags, and exit codes |
| [SpanForge Integration](integration-spanforge.md) | Event payload, trace contexts, entry-points |
| [Contributing](contributing.md) | Dev setup, running tests, code style |
| [Changelog](changelog.md) | Version history |

---

## At a glance: scoring pipeline

```
LLM output
    │
    ▼
extract_claims()              ← sentence splitter + meta-filter
    │
    ▼
create_backend(config)        ← hybrid / embedding / llm-nli
    │
    ▼  for each claim:
backend.score_claim()         ← similarity + entailment label
    │
    ├─ detect_contradiction()  ← negation / antonym / numeric (optional)
    │
    ▼
aggregate scores              ← mean(best-match scores)
    │
    ▼
faithfulness_score = mean
hallucination_risk = 1 − faithfulness_score
confidence = calibrated(count, consistency, backend)
    │
    ├─ passed?  (risk ≤ threshold)
    └─ SpanForge event emitted  →  llm.eval.faithfulness.scored
```

---

## License

MIT — see [LICENSE](../LICENSE).
