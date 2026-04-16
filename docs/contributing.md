# Contributing

Thank you for your interest in contributing to `sf-hallucinate`!

---

## Prerequisites

- Python 3.9 or newer
- [Git](https://git-scm.com/)
- [pip](https://pip.pypa.io/en/stable/)

---

## Getting started

```bash
# 1. Clone the repository
git clone https://github.com/veerarag1973/sf-hallucinate
cd sf-hallucinate

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# 3. Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

---

## Running the test suite

```bash
# All tests with coverage (must reach 90%)
python -m pytest tests/ -v --cov

# A single test file
python -m pytest tests/test_eval.py -v

# A specific test
python -m pytest tests/test_eval.py::TestFaithfulnessScorer::test_faithful_output_passes -v

# Exclude slow hypothesis tests
python -m pytest tests/ -v -m "not slow"
```

The coverage threshold is enforced by `pytest --cov-fail-under=90` in `pyproject.toml`. All CI runs must reach 90%. Current: **311 tests, 92.05% coverage**.

---

## Code style

`sf-hallucinate` uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for lint errors
ruff check src/ tests/

# Auto-fix and format
ruff check --fix src/ tests/
ruff format src/ tests/
```

Key rules enforced:
- PEP 8 naming conventions
- No bare `except:` — use `except Exception:` at minimum
- No unused imports
- Type annotations on all public functions

---

## Type checking

```bash
mypy src/
```

The project targets `mypy --strict` compatibility. Type: `ignore` comments are acceptable only for untyped third-party packages (e.g. `spanforge`).

---

## Project layout

```
sf-hallucinate/
├── src/
│   └── sf_hallucinate/
│       ├── __init__.py              # Public re-exports
│       ├── _types.py                # Frozen dataclasses
│       ├── _exceptions.py           # Exception hierarchy
│       ├── _llm.py                  # OpenAI-compatible chat completion utility
│       ├── eval.py                  # EvalScorer protocol, FaithfulnessScorer, EvalPipeline
│       ├── cli.py                   # Standalone CLI
│       ├── scoring/
│       │   ├── __init__.py
│       │   ├── backends.py          # SimilarityBackend protocol, HybridBackend, create_backend()
│       │   ├── claims.py            # Claim extraction, sentence splitting
│       │   ├── contradiction.py     # Heuristic contradiction detection
│       │   ├── embedding.py         # EmbeddingBackend (sentence-transformers)
│       │   ├── languages.py         # Multi-language tokenizer and stop words
│       │   ├── nli.py               # LLMNLIBackend (OpenAI-compatible NLI)
│       │   ├── overlap.py           # Token/bigram F1, Jaccard
│       │   └── similarity.py        # TF-IDF cosine, hybrid similarity
│       ├── scorers/
│       │   ├── __init__.py
│       │   ├── answer_relevancy.py  # AnswerRelevancyScorer
│       │   └── context_relevancy.py # ContextRelevancyScorer
│       └── integration/
│           └── spanforge.py         # SpanForge event emission
├── tests/
│   ├── conftest.py                  # Shared fixtures and corpus constants
│   ├── test_types.py
│   ├── test_exceptions.py
│   ├── test_eval.py
│   ├── test_integration.py
│   ├── test_cli.py
│   ├── test_llm.py
│   ├── test_answer_relevancy.py
│   ├── test_context_relevancy.py
│   └── scoring/
│       ├── test_backends.py
│       ├── test_claims.py
│       ├── test_contradiction.py
│       ├── test_embedding.py
│       ├── test_languages.py
│       ├── test_nli.py
│       ├── test_overlap.py
│       └── test_similarity.py
├── docs/                            # Documentation (Markdown)
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

---

## Writing tests

- Tests live in `tests/` and mirror the `src/` layout.
- Use `pytest` fixtures from `tests/conftest.py` where applicable.
- Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/).
- Async tests use `@pytest.mark.asyncio` from `pytest-asyncio`.
- Keep test functions focused: one assertion per logical concern.

---

## Submitting a pull request

1. Fork the repository and create a feature branch: `git checkout -b feat/my-feature`
2. Make your changes and write tests.
3. Run `ruff check`, `mypy src/`, and `python -m pytest tests/` — all must pass.
4. Commit using [Conventional Commits](https://www.conventionalcommits.org/): `feat: add bigram scorer`.
5. Open a pull request against `main`. Describe what changed and why.

---

## Adding a custom scorer

Implement the `EvalScorer` protocol:

```python
from sf_hallucinate.eval import EvalScorer
from sf_hallucinate._types import EvalConfig, ScorerResult

class MyScorer:
    name = "my-scorer"
    config = EvalConfig()

    def score(self, output: str, reference: str) -> ScorerResult:
        ...

    def score_batch(
        self, outputs: list[str], references: list[str]
    ) -> list[ScorerResult]:
        return [self.score(o, r) for o, r in zip(outputs, references)]
```

To register it as a SpanForge entry-point, add to your `pyproject.toml`:

```toml
[project.entry-points."spanforge.eval_scorers"]
my-scorer = "mypackage.scorer:MyScorer"
```
