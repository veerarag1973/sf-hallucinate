# sf-hallucinate

**Hallucination detection and faithfulness scoring for LLM outputs.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage: 92%](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)]()
[![Tests: 311 passed](https://img.shields.io/badge/tests-311%20passed-brightgreen.svg)]()
[![SpanForge: 2.0.3](https://img.shields.io/badge/SpanForge-2.0.3-blueviolet.svg)]()

`sf-hallucinate` scores how *faithfully* an LLM-generated answer is grounded in a reference document. It extracts factual claims from the output, checks each claim against the reference using pluggable similarity backends, and produces a hallucination risk score in `[0.0, 1.0]`.

Built on [SpanForge 2.0.3](https://github.com/spanforge/spanforge) — emits structured `llm.eval.*` trace events, integrates with the SpanForge audit chain, and is discoverable via the `spanforge.eval_scorers` entry-point group.

---

## Features

| | |
|---|---|
| **EvalScorer Protocol** | PEP 544 `@runtime_checkable` — plug in any custom scorer without subclassing |
| **FaithfulnessScorer** | Claim-by-claim faithfulness grading with per-sentence breakdown |
| **Pluggable backends** | `hybrid` (default), `embedding`, or `llm-nli` similarity backends |
| **Hybrid similarity** | 0.6 × TF-IDF cosine + 0.4 × token F1 — robust to paraphrases |
| **Embedding backend** | Dense cosine similarity via `sentence-transformers` (optional) |
| **LLM-NLI backend** | Natural Language Inference via any OpenAI-compatible API |
| **Contradiction detection** | Negation asymmetry, antonym pairs, and numeric discrepancy detection |
| **Confidence calibration** | 3-signal formula: claim count × 0.3 + consistency × 0.3 + backend confidence × 0.4 |
| **Multi-language support** | 10 languages: en, es, fr, de, pt, ru, zh, ja, ko, ar with CJK bigram tokenization |
| **Answer relevancy scorer** | Rates how well an LLM answer addresses the question |
| **Context relevancy scorer** | Rates how relevant retrieved context is for the question |
| **Smart claim extraction** | Abbreviation-aware splitter; filters questions and meta-sentences |
| **Pipeline gate** | Optionally raises `HallucinationRiskExceeded` when threshold is breached |
| **EvalPipeline** | Chains multiple scorers, returns `dict[scorer_name → ScorerResult]` |
| **SpanForge integration** | Auto-emits `llm.eval.faithfulness.scored` events |
| **CLI** | `score`, `score-file`, `batch` with human / JSON / JSONL output |
| **Async API** | `ascore()` / `ascore_batch()` for async LLM pipelines |
| **Minimal dependencies** | Only requires `spanforge>=2.0.3`; scoring algorithms are pure stdlib |

---

## Installation

```bash
pip install sf-hallucinate
```

This installs `spanforge>=2.0.3` automatically as a required dependency.

### Optional extras

```bash
# Embedding similarity backend (sentence-transformers)
pip install sf-hallucinate[embedding]

# All optional dependencies
pip install sf-hallucinate[all]
```

From source:

```bash
git clone https://github.com/veerarag1973/sf-hallucinate
cd sf-hallucinate
pip install -e .
```

---

## Quick example

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

scorer = FaithfulnessScorer(EvalConfig(threshold=0.5))
result = scorer.score(output, reference)

print(f"Hallucination risk : {result.hallucination_risk:.3f}")  # e.g. 0.431
print(f"Faithfulness score : {result.faithfulness_score:.3f}")  # e.g. 0.569
print(f"Grounded claims    : {result.grounded_claim_count}/{result.total_claim_count}")
print(f"Passed             : {result.passed}")
```

### Per-claim breakdown

```python
for cr in result.claim_results:
    status = "✓" if cr.grounded else "✗"
    print(f"  {status} [{cr.similarity:.2f}] {cr.claim}")
```

### Fail the pipeline on high risk

```python
from sf_hallucinate._exceptions import HallucinationRiskExceeded

config = EvalConfig(threshold=0.3, fail_on_threshold=True)
scorer = FaithfulnessScorer(config)

try:
    result = scorer.score(output, reference)
except HallucinationRiskExceeded as exc:
    print(f"Blocked! Risk = {exc.result.hallucination_risk:.3f}")
```

### Run multiple scorers

```python
from sf_hallucinate.eval import EvalPipeline

pipeline = EvalPipeline(
    FaithfulnessScorer(EvalConfig(threshold=0.3, scorer_name="strict")),
    FaithfulnessScorer(EvalConfig(threshold=0.7, scorer_name="lenient")),
)
results = pipeline.run(output, reference)
# results["strict"].passed, results["lenient"].passed
```

### Async

```python
import asyncio

async def evaluate():
    result = await scorer.ascore(output, reference)
    print(result.passed)

asyncio.run(evaluate())
```

---

## Pluggable similarity backends

Choose a backend via `EvalConfig(similarity_backend="...")`:

| Backend | Requires | Best for |
|---|---|---|
| `hybrid` (default) | Nothing extra | Fast, offline scoring |
| `embedding` | `pip install sf-hallucinate[embedding]` | Semantic similarity |
| `llm-nli` | OpenAI-compatible API key | Highest accuracy |

```python
from sf_hallucinate.eval import FaithfulnessScorer
from sf_hallucinate._types import EvalConfig

# Embedding backend
config = EvalConfig(
    similarity_backend="embedding",
    embedding_model="all-MiniLM-L6-v2",
)
scorer = FaithfulnessScorer(config)

# LLM-NLI backend
config = EvalConfig(
    similarity_backend="llm-nli",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
)
scorer = FaithfulnessScorer(config)
```

---

## Contradiction detection

Automatically detects claims that contradict the reference using three heuristic signals:

- **Negation asymmetry** — detects "not" / negation mismatches (confidence 0.8)
- **Antonym pairs** — 35 common antonym pairs (confidence 0.6)
- **Numeric discrepancy** — mismatched numbers in similar contexts (confidence 0.7)

```python
config = EvalConfig(detect_contradictions=True)  # enabled by default
scorer = FaithfulnessScorer(config)
result = scorer.score(
    "The tower is 200 metres tall.",
    "The tower is 330 metres tall.",
)
print(result.contradiction_count)  # 1
```

---

## Multi-language support

Supports 10 languages with language-aware tokenization:

`en` · `es` · `fr` · `de` · `pt` · `ru` · `zh` · `ja` · `ko` · `ar`

CJK languages use character bigram tokenization. All languages include stop word removal.

```python
config = EvalConfig(language="es")
scorer = FaithfulnessScorer(config)
result = scorer.score(
    "París es la capital de Francia.",
    "París es la capital y la ciudad más grande de Francia.",
)
```

---

## Answer & context relevancy scorers

Beyond faithfulness, evaluate how well an answer addresses the question and how relevant the retrieved context is:

```python
from sf_hallucinate.scorers import AnswerRelevancyScorer, ContextRelevancyScorer
from sf_hallucinate._types import EvalConfig

# Answer relevancy — does the answer address the question?
scorer = AnswerRelevancyScorer(
    question="What is the capital of France?",
    config=EvalConfig(similarity_backend="hybrid"),
)
result = scorer.score("The capital of France is Paris.", "Paris is the capital.")
print(f"Answer relevancy: {result.faithfulness_score:.3f}")

# Context relevancy — is the retrieved context useful for the question?
scorer = ContextRelevancyScorer(
    question="What is the capital of France?",
    config=EvalConfig(similarity_backend="hybrid"),
)
result = scorer.score("answer text", "Paris is the capital city of France.")
print(f"Context relevancy: {result.faithfulness_score:.3f}")
```

---

## Confidence calibration

Every `ScorerResult` includes a `confidence` field (0.0–1.0) computed from three signals:

$$\text{confidence} = 0.3 \times \text{claim\_count\_signal} + 0.3 \times \text{consistency} + 0.4 \times \text{backend\_confidence}$$

- **Claim count signal** — saturates at 5 claims (more claims = more reliable)
- **Consistency** — inverse std-dev of per-claim scores
- **Backend confidence** — mean confidence from the similarity backend

---

## CLI

```bash
# Single pair
sf-hallucinate score \
    --output "The Eiffel Tower is in Berlin." \
    --reference "The Eiffel Tower is in Paris, France." \
    --threshold 0.4

# With LLM-NLI backend
sf-hallucinate score \
    --output "Paris is the capital." \
    --reference "Paris is the capital of France." \
    --backend llm-nli \
    --llm-api-key sk-... \
    --llm-model gpt-4o-mini

# Multi-language scoring
sf-hallucinate score \
    --output "París es la capital." \
    --reference "París es la capital de Francia." \
    --language es

# From files
sf-hallucinate score-file \
    --output answer.txt \
    --reference source.txt \
    --format json

# Batch JSONL  (each line: {"output": "...", "reference": "..."})
sf-hallucinate batch \
    --input pairs.jsonl \
    --threshold 0.5 \
    --fail-on-any \
    --format jsonl
```

### New CLI flags (v1.1.0)

| Flag | Description |
|---|---|
| `--backend` | Similarity backend: `hybrid`, `embedding`, `llm-nli` |
| `--embedding-model` | Model name for embedding backend |
| `--llm-model` | Model name for LLM-NLI backend |
| `--llm-api-key` | API key for LLM-NLI backend |
| `--language` | Language code (en, es, fr, de, pt, ru, zh, ja, ko, ar) |
| `--no-contradiction-detection` | Disable contradiction detection |

**Exit codes:** `0` = all passed · `1` = failed or error · `130` = interrupted

---

## Scoring pipeline

```
LLM output
    │
    ▼
extract_claims()          ← sentence splitter + meta-filter
    │
    ▼  for each claim:
SimilarityBackend         ← hybrid / embedding / llm-nli
    │
    ├─ contradiction?     ← negation + antonyms + numerics
    │
    ▼
faithfulness_score = mean(best-match scores)
hallucination_risk = 1 − faithfulness_score
confidence = calibrated from 3 signals
    │
    ├─ passed?  (risk ≤ threshold)
    └─ SpanForge event  →  llm.eval.faithfulness.scored
```

**Hybrid similarity formula:**

$$\text{hybrid}(c, r) = 0.6 \times \text{tfidf\_cosine}(c, r) + 0.4 \times \text{token\_f1}(c, r)$$

See [docs/algorithms.md](docs/algorithms.md) for the full mathematical derivation.

---

## SpanForge integration

When `spanforge>=2.0.3` is installed, every `score()` call emits:

```json
{
  "event_type": "llm.eval.faithfulness.scored",
  "source":     "sf-hallucinate/faithfulness@1.0.0",
  "payload": {
    "evaluator":          "sf-hallucinate/faithfulness",
    "score":              0.588,
    "label":              "pass",
    "hallucination_risk": 0.412,
    "grounded_claims":    2,
    "total_claims":       3,
    "threshold":          0.5
  }
}
```

`FaithfulnessScorer` is registered under the `spanforge.eval_scorers` entry-point group and is auto-discovered by SpanForge pipelines.

See [docs/integration-spanforge.md](docs/integration-spanforge.md) for full details.

---

## Documentation

| | |
|---|---|
| [Quickstart](docs/quickstart.md) | 5-minute walkthrough |
| [API Reference](docs/api-reference.md) | All classes, methods, and types |
| [Algorithms](docs/algorithms.md) | Math behind the scoring |
| [CLI Reference](docs/cli.md) | Sub-commands, flags, exit codes |
| [SpanForge Integration](docs/integration-spanforge.md) | Events, entry-points, trace contexts |
| [Contributing](docs/contributing.md) | Dev setup, tests, code style |
| [Changelog](CHANGELOG.md) | Version history |

---

## License

MIT — see [LICENSE](LICENSE).
Scores LLM outputs against reference documents for factual grounding. Produces a hallucination risk score per output. Configurable threshold. Fails pipeline if score exceeds threshold.
