# Changelog

All notable changes to `sf-hallucinate` are documented here.

This project follows [Semantic Versioning](https://semver.org/) and
[Conventional Commits](https://www.conventionalcommits.org/).

See the full [CHANGELOG.md](../CHANGELOG.md) at the repository root.

---

## Quick summary

### v1.1.0 (2026-04-16)

- Pluggable similarity backends: `hybrid` (default), `embedding`, `llm-nli`
- Contradiction detection (negation, antonyms, numeric discrepancies)
- Multi-language support (10 languages including CJK bigram tokenization)
- Confidence calibration (3-signal formula)
- `AnswerRelevancyScorer` and `ContextRelevancyScorer`
- LLM API utility (`_llm.py`) using stdlib `urllib`
- New CLI flags: `--backend`, `--language`, `--llm-model`, `--llm-api-key`, etc.
- 311 tests, 92% coverage

### v1.0.0 (2026-04-16)

- Initial release with `FaithfulnessScorer`, hybrid TF-IDF + token F1 similarity
- `EvalScorer` protocol, `EvalPipeline`, CLI, SpanForge integration
- 195 tests, 95% coverage
