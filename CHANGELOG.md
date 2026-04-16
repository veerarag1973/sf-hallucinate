# Changelog

All notable changes to `sf-hallucinate` are documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and
[Conventional Commits](https://www.conventionalcommits.org/).

---

## [1.1.0] — 2026-04-16

### Added

#### Pluggable similarity backends

- **`SimilarityBackend` protocol** (`scoring/backends.py`) — `@runtime_checkable` protocol with `score_claim()` method; factory function `create_backend()` instantiates by name.
- **`HybridBackend`** — default backend; English fast-path delegates to existing `find_best_match()`; non-English uses language-aware TF-IDF/F1 cosine hybrid. Contradiction detection integrated.
- **`EmbeddingBackend`** (`scoring/embedding.py`) — dense cosine similarity via `sentence-transformers`. Requires optional `[embedding]` extra.
- **`LLMNLIBackend`** (`scoring/nli.py`) — Natural Language Inference via any OpenAI-compatible chat completion API. Maps entailment/contradiction/neutral labels to similarity scores. Batch scoring in a single API call.

#### Contradiction detection

- **`scoring/contradiction.py`** — heuristic contradiction detector with three signals:
  - Negation asymmetry detection (confidence 0.8)
  - Antonym pair matching — 35 common antonym pairs (confidence 0.6)
  - Numeric discrepancy detection (confidence 0.7)
- Contradiction-detected claims are penalised to similarity ≤ 0.15.
- `detect_contradictions` config flag (default `True`) to enable/disable.

#### Multi-language support

- **`scoring/languages.py`** — language-aware tokenizer with stop word lists for 10 languages: `en`, `es`, `fr`, `de`, `pt`, `ru`, `zh`, `ja`, `ko`, `ar`.
- CJK languages (zh, ja, ko) use character bigram tokenization.
- Unicode NFC normalisation applied before tokenization.

#### Confidence calibration

- `ScorerResult.confidence` field — 3-signal calibrated confidence score:
  - Claim count signal (0.3 weight, saturates at 5 claims)
  - Consistency signal (0.3 weight, inverse std-dev of per-claim scores)
  - Backend confidence signal (0.4 weight, mean confidence from similarity backend)
- `ScorerResult.contradiction_count` field — count of contradictions detected.

#### New scorers

- **`AnswerRelevancyScorer`** (`scorers/answer_relevancy.py`) — rates how well an LLM answer addresses a given question. Supports hybrid, embedding, and llm-nli backends.
- **`ContextRelevancyScorer`** (`scorers/context_relevancy.py`) — rates how relevant retrieved context is for answering a question. Same backend support.

#### LLM API utility

- **`_llm.py`** — stdlib `urllib.request`-based OpenAI-compatible chat completion caller. No additional HTTP dependencies. Falls back to `OPENAI_API_KEY` env var.

#### New `ClaimResult` fields

- `contradiction_detected: bool` — whether the claim contradicts the reference.
- `entailment_label: str` — NLI label from LLM backend (entailment/contradiction/neutral).
- `confidence: float` — per-claim confidence from the backend.

#### New `EvalConfig` fields

- `similarity_backend: str` — `"hybrid"` (default), `"embedding"`, or `"llm-nli"`.
- `embedding_model: str` — model name for embedding backend (default `"all-MiniLM-L6-v2"`).
- `llm_model: str` — model name for LLM-NLI backend (default `"gpt-4o-mini"`).
- `llm_api_key: str | None` — API key for LLM backends.
- `llm_base_url: str | None` — custom API base URL.
- `language: str` — language code (default `"en"`).
- `detect_contradictions: bool` — enable/disable contradiction detection (default `True`).

#### CLI enhancements

- `--backend` flag — select similarity backend (`hybrid`, `embedding`, `llm-nli`).
- `--embedding-model` flag — model name for embedding backend.
- `--llm-model` flag — model name for LLM-NLI backend.
- `--llm-api-key` flag — API key for LLM backends.
- `--language` flag — language code for multi-language scoring.
- `--no-contradiction-detection` flag — disable contradiction detection.

#### Optional dependencies

- `[embedding]` extra — installs `sentence-transformers>=2.2`.
- `[all]` extra — installs all optional dependencies.

#### Tests

- **311 tests** across 16 test files; **92% line + branch coverage** (target: 90%).
- 7 new test files for backends, languages, contradiction, embedding, NLI, LLM utility, and scorers.

### Changed

- `eval.py` — `_score_claims()` refactored to use pluggable `SimilarityBackend` via factory pattern instead of direct `find_best_match()` calls.
- `eval.py` — `_aggregate()` now computes `contradiction_count` and calibrated `confidence`.
- Version bumped to **1.1.0** across `__init__.py`, `cli.py`, and `pyproject.toml`.

---

## [1.0.0] — 2026-04-16

### Added

#### Core protocol and scorer

- **`EvalScorer` protocol** (`eval.py`) — PEP 544 `@runtime_checkable` protocol defining the contract for all scorers: `name`, `config`, `score()`, and `score_batch()`. Aligns with the SpanForge `llm.eval.*` namespace for zero-friction integration.
- **`FaithfulnessScorer`** — built-in scorer implementing the faithfulness evaluation strategy:
  - Claim extraction from LLM outputs (abbreviation-aware sentence splitter, meta-sentence filter)
  - Per-claim best-match search against reference sentences using hybrid similarity
  - Aggregate `faithfulness_score` and `hallucination_risk` computation
  - Configurable `fail_on_threshold` pipeline gate (raises `HallucinationRiskExceeded`)
  - Synchronous (`score`, `score_batch`) and async (`ascore`, `ascore_batch`) APIs
- **`EvalPipeline`** — chains multiple `EvalScorer` instances; returns `dict[scorer_name → ScorerResult]`

#### Scoring algorithms (pure Python stdlib, no additional dependencies beyond SpanForge)

- **`scoring/similarity.py`** — TF-IDF cosine similarity with smoothed IDF; `hybrid_similarity()` combining TF-IDF cosine (weight 0.6) and token F1 (weight 0.4); `find_best_match()` to locate the highest-scoring reference sentence for a claim
- **`scoring/claims.py`** — `split_sentences()` (abbreviation-aware, handles `Dr.`, `Mr.`, `U.S.`, `e.g.`, etc.); `extract_claims()` (filters questions, meta-phrases, short sentences)
- **`scoring/overlap.py`** — `token_f1()`, `bigram_f1()`, `jaccard()`, `tokenize()` (stopword filtering)

#### Data types and exceptions

- **`_types.py`** — frozen dataclasses `EvalConfig`, `ScorerResult`, `ClaimResult` with `to_dict()` helpers and derived properties (`grounding_rate`, `ungrounded_claims`)
- **`_exceptions.py`** — `SfHallucinateError` base, `HallucinationRiskExceeded` (carries full `ScorerResult`), `EmptyOutputError`, `EmptyReferenceError`

#### SpanForge integration

- **`integration/spanforge.py`** — emits `llm.eval.faithfulness.scored` events automatically after each `score()` call; dispatches through active SpanForge config if available; errors are swallowed so scoring is never interrupted by event infrastructure
- **Entry-point registration** — `FaithfulnessScorer` registered under `spanforge.eval_scorers` for auto-discovery by SpanForge pipelines

#### CLI (`sf-hallucinate` / `python -m sf_hallucinate`)

- `score` sub-command — score a single output/reference pair from inline strings
- `score-file` sub-command — score from UTF-8 text files
- `batch` sub-command — score a JSONL file of `{"output": ..., "reference": ...}` pairs
- Output formats: `human` (visual risk bar + per-claim breakdown), `json`, `jsonl`
- Exit codes: `0` pass, `1` fail/error, `130` interrupted
- `--fail-on-any` flag for batch mode
- `_quiet_stdout()` context manager to prevent SpanForge pretty-print output from polluting machine-readable JSON/JSONL streams

#### Tests

- **195 tests** across 9 test files; **95.25% line + branch coverage** (target: 90%)
- Property-based tests using [Hypothesis](https://hypothesis.readthedocs.io/) for scoring invariants
- Async tests using `pytest-asyncio`
- Full CLI coverage including JSON output validation, exit code verification, and `KeyboardInterrupt` handling

#### Documentation

- `docs/index.md` — feature overview and pipeline diagram
- `docs/quickstart.md` — 5-minute guide with 7 worked examples
- `docs/api-reference.md` — full reference for all public classes, methods, and types
- `docs/algorithms.md` — mathematical derivation of hybrid similarity and faithfulness scoring
- `docs/cli.md` — all sub-commands, flags, exit codes, and output shapes
- `docs/integration-spanforge.md` — SpanForge event payload, dispatch hierarchy, entry-point registration
- `docs/contributing.md` — dev setup, test running, code style, PR guide

---

[1.1.0]: https://github.com/veerarag1973/sf-hallucinate/releases/tag/v1.1.0
[1.0.0]: https://github.com/veerarag1973/sf-hallucinate/releases/tag/v1.0.0
