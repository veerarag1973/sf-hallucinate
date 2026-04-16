"""Shared pytest fixtures and helpers."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from sf_hallucinate._types import EvalConfig
from sf_hallucinate.eval import FaithfulnessScorer


# ---------------------------------------------------------------------------
# Well-known test corpora
# ---------------------------------------------------------------------------

# A reference document that states factual claims clearly
REFERENCE_PARIS = (
    "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. "
    "It was designed by the engineer Gustave Eiffel and built between 1887 and 1889. "
    "The tower stands 330 metres tall and was the world's tallest man-made structure "
    "from 1889 until 1930. It attracts millions of tourists every year."
)

# Faithful LLM output (almost every claim is grounded)
OUTPUT_FAITHFUL = (
    "The Eiffel Tower stands in Paris, France. "
    "It was constructed between 1887 and 1889. "
    "Gustave Eiffel designed the tower. "
    "The structure is 330 metres in height."
)

# Hallucinated LLM output (claims contradicting the reference)
OUTPUT_HALLUCINATED = (
    "The Eiffel Tower is located in Berlin, Germany. "
    "It was built in 1950 by the architect Ludwig Mies van der Rohe. "
    "The tower is made of reinforced concrete and stands 150 metres tall."
)

# Mixed output
OUTPUT_MIXED = (
    "The Eiffel Tower is in Paris. "
    "It was built by Julius Caesar in 100 BC. "
    "The tower is 330 metres tall."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_scorer() -> FaithfulnessScorer:
    return FaithfulnessScorer()


@pytest.fixture()
def strict_scorer() -> FaithfulnessScorer:
    return FaithfulnessScorer(EvalConfig(threshold=0.2, grounding_threshold=0.3))


@pytest.fixture()
def failing_scorer() -> FaithfulnessScorer:
    return FaithfulnessScorer(EvalConfig(fail_on_threshold=True, threshold=0.3))


@pytest.fixture()
def tmp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture()
def tmp_output_file(tmp_dir: Path) -> Path:
    p = tmp_dir / "output.txt"
    p.write_text(OUTPUT_FAITHFUL, encoding="utf-8")
    return p


@pytest.fixture()
def tmp_reference_file(tmp_dir: Path) -> Path:
    p = tmp_dir / "reference.txt"
    p.write_text(REFERENCE_PARIS, encoding="utf-8")
    return p


@pytest.fixture()
def tmp_batch_jsonl(tmp_dir: Path) -> Path:
    pairs = [
        {"output": OUTPUT_FAITHFUL, "reference": REFERENCE_PARIS},
        {"output": OUTPUT_HALLUCINATED, "reference": REFERENCE_PARIS},
    ]
    p = tmp_dir / "pairs.jsonl"
    p.write_text(
        "\n".join(json.dumps(pair) for pair in pairs),
        encoding="utf-8",
    )
    return p
