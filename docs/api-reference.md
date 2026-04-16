# API Reference

Full reference for all public classes, methods, and types in `sf-hallucinate`.

---

## Module: `sf_hallucinate.eval`

### `EvalScorer` (Protocol)

```python
@runtime_checkable
class EvalScorer(Protocol):
    name: str
    config: EvalConfig

    def score(self, output: str, reference: str) -> ScorerResult: ...
    def score_batch(
        self, outputs: list[str], references: list[str]
    ) -> list[ScorerResult]: ...
```

A PEP 544 `@runtime_checkable` protocol. Any class implementing `name`, `config`, `score()`, and `score_batch()` is considered a valid `EvalScorer`. No subclassing required.

SpanForge discovers scorers registered under the `spanforge.eval_scorers` entry-point group automatically.

---

### `FaithfulnessScorer`

```python
class FaithfulnessScorer:
    def __init__(self, config: EvalConfig | None = None) -> None: ...
```

Built-in faithfulness scorer. Satisfies `EvalScorer`.

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` (property) | Returns `config.scorer_name`. Defaults to `"faithfulness"`. |
| `config` | `EvalConfig` | Algorithm and threshold configuration. |

**Methods**

#### `score(output, reference) → ScorerResult`

```python
def score(self, output: str, reference: str) -> ScorerResult
```

Score a single LLM output against a reference document.

- Extracts declarative claims from `output`.
- Finds the best-matching sentence in `reference` for each claim using hybrid similarity.
- Computes aggregate `faithfulness_score` and `hallucination_risk`.
- Emits a `llm.eval.faithfulness.scored` SpanForge event (if SpanForge is installed).

**Raises:** `EmptyOutputError`, `EmptyReferenceError`, `HallucinationRiskExceeded` (when `config.fail_on_threshold=True` and risk exceeds threshold).

---

#### `score_batch(outputs, references) → list[ScorerResult]`

```python
def score_batch(
    self, outputs: list[str], references: list[str]
) -> list[ScorerResult]
```

Score multiple pairs in sequence. Lengths of `outputs` and `references` must match.

**Raises:** `ValueError` if lengths differ.

---

#### `ascore(output, reference) → ScorerResult`

```python
async def ascore(self, output: str, reference: str) -> ScorerResult
```

Async wrapper around `score()`. Uses `asyncio.get_event_loop().run_in_executor()` internally.

---

#### `ascore_batch(outputs, references) → list[ScorerResult]`

```python
async def ascore_batch(
    self, outputs: list[str], references: list[str]
) -> list[ScorerResult]
```

Async wrapper around `score_batch()`.

---

### `EvalPipeline`

```python
class EvalPipeline:
    def __init__(self, *scorers: EvalScorer) -> None: ...
    def run(self, output: str, reference: str) -> dict[str, ScorerResult]: ...
```

Chains multiple scorers. Requires at least one scorer (raises `ValueError` otherwise).

`run()` executes each scorer in order and returns a `dict` keyed by each scorer's `name`. Exceptions from individual scorers propagate immediately and halt the pipeline.

**Example**

```python
pipeline = EvalPipeline(
    FaithfulnessScorer(EvalConfig(threshold=0.3, scorer_name="strict")),
    FaithfulnessScorer(EvalConfig(threshold=0.7, scorer_name="lenient")),
)
results = pipeline.run(output, reference)
# results["strict"].passed, results["lenient"].passed
```

---

## Module: `sf_hallucinate._types`

### `EvalConfig`

```python
@dataclasses.dataclass(frozen=True)
class EvalConfig:
    threshold: float = 0.5
    grounding_threshold: float = 0.25
    min_claim_length: int = 15
    fail_on_threshold: bool = False
    scorer_name: str = "faithfulness"
    tfidf_weight: float = 0.6
    similarity_backend: str = "hybrid"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    language: str = "en"
    detect_contradictions: bool = True
```

Frozen dataclass — safe to hash, cache, and share across threads.

| Field | Default | Description |
|---|---|---|
| `threshold` | `0.5` | Pipeline fails when `hallucination_risk > threshold`. Must be in `[0.0, 1.0]`. |
| `grounding_threshold` | `0.25` | A claim is *grounded* when its best-match similarity ≥ this value. |
| `min_claim_length` | `15` | Claims shorter than this (characters) are discarded. |
| `fail_on_threshold` | `False` | When `True`, `score()` raises `HallucinationRiskExceeded` on failure. |
| `scorer_name` | `"faithfulness"` | Key used in `EvalPipeline` results and `ScorerResult.metadata`. |
| `tfidf_weight` | `0.6` | Weight for TF-IDF cosine in the hybrid similarity formula. |
| `similarity_backend` | `"hybrid"` | Backend selection: `"hybrid"`, `"embedding"`, or `"llm-nli"`. |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Model name for the embedding backend. |
| `llm_model` | `"gpt-4o-mini"` | Model name for the LLM-NLI backend. |
| `llm_api_key` | `None` | API key for LLM backends. Falls back to `OPENAI_API_KEY` env var. |
| `llm_base_url` | `None` | Custom API base URL for LLM backends. |
| `language` | `"en"` | Language code: `ar`, `de`, `en`, `es`, `fr`, `ja`, `ko`, `pt`, `ru`, `zh`. |
| `detect_contradictions` | `True` | Enable heuristic contradiction detection. |

**Validation** (raises `ValueError` on construction):
- `threshold` must be in `[0.0, 1.0]`
- `grounding_threshold` must be in `[0.0, 1.0]`
- `min_claim_length` must be ≥ 1
- `tfidf_weight` must be in `[0.0, 1.0]`
- `similarity_backend` must be one of `("hybrid", "embedding", "llm-nli")`
- `language` must be one of `("ar", "de", "en", "es", "fr", "ja", "ko", "pt", "ru", "zh")`

---

### `ScorerResult`

```python
@dataclasses.dataclass(frozen=True)
class ScorerResult:
    hallucination_risk: float
    faithfulness_score: float
    grounded_claim_count: int
    total_claim_count: int
    claim_results: tuple[ClaimResult, ...]
    threshold: float
    passed: bool
    metadata: dict[str, Any]
    confidence: float = 1.0
    contradiction_count: int = 0
```

Frozen, hashable aggregate result.

| Field | Type | Default | Description |
|---|---|---|---|
| `hallucination_risk` | `float` | — | Risk score in `[0.0, 1.0]`. |
| `faithfulness_score` | `float` | — | `1.0 - hallucination_risk`. |
| `grounded_claim_count` | `int` | — | Claims meeting grounding threshold. |
| `total_claim_count` | `int` | — | Total claims extracted. |
| `claim_results` | `tuple[ClaimResult, ...]` | — | Per-claim breakdown. |
| `threshold` | `float` | — | Configured risk threshold. |
| `passed` | `bool` | — | `True` when risk ≤ threshold. |
| `metadata` | `dict[str, Any]` | — | Scorer name, config, diagnostics. |
| `confidence` | `float` | `1.0` | Calibrated confidence score (3-signal formula). |
| `contradiction_count` | `int` | `0` | Number of contradictions detected. |

**Properties**

| Property | Type | Description |
|---|---|---|
| `grounding_rate` | `float` | `grounded_claim_count / total_claim_count`. Returns `1.0` when no claims. |
| `ungrounded_claims` | `tuple[ClaimResult, ...]` | Subset of `claim_results` where `grounded=False`. |

**Methods**

- `to_dict() → dict[str, Any]` — JSON-serialisable representation (includes `claim_results` as list of dicts).

---

### `ClaimResult`

```python
@dataclasses.dataclass(frozen=True)
class ClaimResult:
    claim: str
    best_match: str
    similarity: float
    grounded: bool
    contradiction_detected: bool = False
    entailment_label: str = ""
    confidence: float = 1.0
```

Score for one extracted claim.

| Field | Default | Description |
|---|---|---|
| `claim` | — | The claim sentence extracted from the LLM output. |
| `best_match` | — | The reference sentence that scored highest against this claim. |
| `similarity` | — | Similarity score in `[0.0, 1.0]`. |
| `grounded` | — | `True` when `similarity >= EvalConfig.grounding_threshold`. |
| `contradiction_detected` | `False` | Whether the claim contradicts the reference. |
| `entailment_label` | `""` | NLI label from LLM backend (`"entailment"`, `"contradiction"`, `"neutral"`). |
| `confidence` | `1.0` | Per-claim confidence from the backend. |

- `to_dict() → dict[str, Any]`

---

## Module: `sf_hallucinate._exceptions`

### `SfHallucinateError`

Base exception for all `sf-hallucinate` errors.

---

### `HallucinationRiskExceeded`

```python
class HallucinationRiskExceeded(SfHallucinateError):
    def __init__(self, result: ScorerResult) -> None: ...
    result: ScorerResult
```

Raised by `FaithfulnessScorer.score()` when `config.fail_on_threshold=True` and the computed risk exceeds `config.threshold`. The full `ScorerResult` is attached as `.result`.

---

### `EmptyOutputError`

Raised when the LLM output string is blank or whitespace-only.

---

### `EmptyReferenceError`

Raised when the reference string is blank or whitespace-only.

---

## Module: `sf_hallucinate.scoring.similarity`

### `hybrid_similarity(hypothesis, reference, *, tfidf_weight=0.6) → float`

```python
def hybrid_similarity(
    hypothesis: str,
    reference: str,
    *,
    tfidf_weight: float = 0.6,
) -> float
```

Compute the hybrid similarity between two sentences.

Returns a float in `[0.0, 1.0]`. Empty strings return `0.0`.

---

### `find_best_match(claim, ref_sentences) → tuple[str, float]`

```python
def find_best_match(
    claim: str,
    ref_sentences: list[str],
) -> tuple[str, float]
```

Find the reference sentence with the highest hybrid similarity to `claim`.

Returns `("", 0.0)` when `ref_sentences` is empty.

---

## Module: `sf_hallucinate.scoring.claims`

### `extract_claims(text) → list[str]`

```python
def extract_claims(text: str) -> list[str]
```

Extract declarative factual claims from `text`.

Filters out:
- Questions (end with `?`)
- Meta-sentences (e.g. *"In summary, ..."*, *"As mentioned above, ..."*)
- Sentences shorter than `EvalConfig.min_claim_length` (15 chars by default)

---

### `split_sentences(text) → list[str]`

```python
def split_sentences(text: str) -> list[str]
```

Abbreviation-aware sentence splitter using only the standard library. Correctly handles `Dr.`, `Mr.`, `U.S.`, `e.g.`, `i.e.`, etc.

---

## Module: `sf_hallucinate.scoring.overlap`

### `token_f1(hypothesis, reference) → float`

Unigram F1 overlap between two strings (after stopword filtering). Returns `0.0` for empty inputs.

### `bigram_f1(hypothesis, reference) → float`

Bigram F1 overlap.

### `jaccard(hypothesis, reference) → float`

Jaccard similarity on token sets.

### `tokenize(text) → list[str]`

Lower-case, strip punctuation, remove stopwords.

---

## Module: `sf_hallucinate.scoring.backends`

### `SimilarityBackend` (Protocol)

```python
@runtime_checkable
class SimilarityBackend(Protocol):
    def score_claim(
        self, claim: str, reference_sentences: list[str]
    ) -> ClaimScore: ...
```

Protocol for pluggable similarity backends. Any class implementing `score_claim()` satisfies this protocol.

---

### `ClaimScore`

```python
class ClaimScore:
    similarity: float
    best_match: str
    entailment_label: str = ""
    contradiction_detected: bool = False
    contradiction_score: float = 0.0
    confidence: float = 1.0
```

Intermediate result from a backend's `score_claim()`. Bridges backends to `ClaimResult`.

---

### `HybridBackend`

```python
class HybridBackend:
    def __init__(self, config: EvalConfig) -> None: ...
    def score_claim(self, claim: str, reference_sentences: list[str]) -> ClaimScore: ...
```

Default backend. For English, delegates to optimised `find_best_match()`. For other languages, uses language-aware TF-IDF cosine + token F1 with `scoring.languages.tokenize()`. Integrates contradiction detection when `config.detect_contradictions=True`.

---

### `create_backend(config) → SimilarityBackend`

```python
def create_backend(config: EvalConfig) -> SimilarityBackend
```

Factory function. Creates a backend instance based on `config.similarity_backend`:
- `"hybrid"` → `HybridBackend`
- `"embedding"` → `EmbeddingBackend` (lazy import)
- `"llm-nli"` → `LLMNLIBackend` (lazy import)

Raises `ValueError` for unknown backend names.

---

## Module: `sf_hallucinate.scoring.embedding`

### `EmbeddingBackend`

```python
class EmbeddingBackend:
    def __init__(self, config: EvalConfig) -> None: ...
    def score_claim(self, claim: str, reference_sentences: list[str]) -> ClaimScore: ...
```

Dense cosine similarity using `sentence-transformers`. Batch-encodes claim + all references in one call. Requires `pip install sf-hallucinate[embedding]`.

Raises `ImportError` if `sentence-transformers` is not installed.

---

## Module: `sf_hallucinate.scoring.nli`

### `LLMNLIBackend`

```python
class LLMNLIBackend:
    def __init__(self, config: EvalConfig) -> None: ...
    def score_claim(self, claim: str, reference_sentences: list[str]) -> ClaimScore: ...
    def score_claims_batch(
        self, claims: list[str], reference_sentences: list[str]
    ) -> list[ClaimScore]: ...
```

Natural Language Inference via any OpenAI-compatible chat completion API. Sends structured prompts and parses JSON responses with fallback. Label mapping:
- `entailment` → 0.92+ similarity
- `contradiction` → 0.05 × (1 − confidence)
- `neutral` → 0.35+ similarity

`score_claims_batch()` sends all claims in one API call for efficiency.

Raises `ValueError` if no API key is available.

---

## Module: `sf_hallucinate.scoring.contradiction`

### `detect_contradiction(claim, reference_sentence, *, language="en") → tuple[bool, float]`

```python
def detect_contradiction(
    claim: str,
    reference_sentence: str,
    *,
    language: str = "en",
) -> tuple[bool, float]
```

Heuristic contradiction detection. Returns `(is_contradiction, confidence)`.

Three detection signals:
- **Negation asymmetry** (confidence 0.8) — one sentence has negation, the other doesn't, with >20% vocabulary overlap.
- **Antonym pairs** (confidence 0.6) — 35 common antonym pairs (e.g. true/false, increase/decrease).
- **Numeric discrepancy** (confidence 0.7) — different numbers in sentences with shared context.

---

## Module: `sf_hallucinate.scoring.languages`

### `tokenize(text, *, language="en", remove_stop_words=False) → list[str]`

```python
def tokenize(
    text: str,
    *,
    language: str = "en",
    remove_stop_words: bool = False,
) -> list[str]
```

Language-aware tokenizer. Applies Unicode NFC normalisation. CJK languages (zh, ja, ko) use character bigram tokenization. Supports 10 languages with built-in stop word lists.

### `is_cjk_language(language) → bool`

Returns `True` for `zh`, `ja`, `ko`.

---

## Module: `sf_hallucinate.scorers`

### `AnswerRelevancyScorer`

```python
class AnswerRelevancyScorer:
    def __init__(self, *, question: str, config: EvalConfig | None = None) -> None: ...
    @property
    def name(self) -> str: ...            # "answer_relevancy"
    def score(self, output: str, reference: str) -> ScorerResult: ...
    def score_batch(self, outputs: list[str], references: list[str]) -> list[ScorerResult]: ...
```

Rates how well an LLM answer addresses a given question. The `question` is bound at construction time so the scorer satisfies the `EvalScorer` protocol.

Backend modes:
- **hybrid** — lexical F1 overlap between question and answer tokens.
- **embedding** — cosine similarity between question and answer embeddings.
- **llm-nli** — LLM-as-judge relevancy rating (recommended, returns JSON with `relevancy_score`).

---

### `ContextRelevancyScorer`

```python
class ContextRelevancyScorer:
    def __init__(self, *, question: str, config: EvalConfig | None = None) -> None: ...
    @property
    def name(self) -> str: ...            # "context_relevancy"
    def score(self, output: str, reference: str) -> ScorerResult: ...
    def score_batch(self, outputs: list[str], references: list[str]) -> list[ScorerResult]: ...
```

Rates how relevant retrieved context is for answering a given question. Same interface and backend support as `AnswerRelevancyScorer`, but scores the `reference` (context) rather than the `output` (answer).

---

## Module: `sf_hallucinate._llm`

### `call_chat_completion(messages, *, model, api_key=None, base_url=None, temperature=0.0, timeout=120) → str`

```python
def call_chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    timeout: int = 120,
) -> str
```

Calls an OpenAI-compatible chat completion endpoint using stdlib `urllib.request`. Returns the assistant message content string.

- Falls back to `OPENAI_API_KEY` env var when `api_key` is `None`.
- Raises `ValueError` when no API key is available.
- Raises `RuntimeError` on HTTP errors or connection failures.
