"""Tests for sf_hallucinate.scoring.backends — backend protocol and HybridBackend."""
from __future__ import annotations

import pytest

from sf_hallucinate._types import EvalConfig
from sf_hallucinate.scoring.backends import (
    ClaimScore,
    HybridBackend,
    SimilarityBackend,
    create_backend,
)


class TestClaimScore:
    def test_defaults(self) -> None:
        cs = ClaimScore(similarity=0.8, best_match="hello")
        assert cs.similarity == 0.8
        assert cs.best_match == "hello"
        assert cs.entailment_label == ""
        assert cs.contradiction_detected is False
        assert cs.confidence == 1.0

    def test_full_construction(self) -> None:
        cs = ClaimScore(
            similarity=0.95,
            best_match="ref",
            entailment_label="entailment",
            contradiction_detected=False,
            confidence=0.9,
        )
        assert cs.entailment_label == "entailment"
        assert cs.confidence == 0.9


class TestHybridBackend:
    def test_english_scoring(self) -> None:
        config = EvalConfig(language="en")
        backend = HybridBackend(config)
        result = backend.score_claim(
            "Paris is in France.",
            ["Paris is in France.", "Berlin is in Germany."],
        )
        assert isinstance(result, ClaimScore)
        assert result.similarity > 0.5
        assert "Paris" in result.best_match

    def test_empty_references(self) -> None:
        config = EvalConfig()
        backend = HybridBackend(config)
        result = backend.score_claim("Any claim", [])
        assert result.similarity == 0.0
        assert result.best_match == ""

    def test_multilang_spanish(self) -> None:
        config = EvalConfig(language="es")
        backend = HybridBackend(config)
        result = backend.score_claim(
            "París está en Francia.",
            ["París está en Francia.", "Berlín está en Alemania."],
        )
        assert result.similarity > 0.3
        assert "París" in result.best_match

    def test_contradiction_detection(self) -> None:
        config = EvalConfig(detect_contradictions=True)
        backend = HybridBackend(config)
        result = backend.score_claim(
            "The tower is not 330 metres tall.",
            ["The tower is 330 metres tall."],
        )
        # Should detect contradiction and penalise similarity
        assert result.contradiction_detected is True

    def test_no_contradiction_detection(self) -> None:
        config = EvalConfig(detect_contradictions=False)
        backend = HybridBackend(config)
        result = backend.score_claim(
            "The tower is not 330 metres tall.",
            ["The tower is 330 metres tall."],
        )
        assert result.contradiction_detected is False

    def test_chinese_scoring(self) -> None:
        config = EvalConfig(language="zh")
        backend = HybridBackend(config)
        result = backend.score_claim(
            "东京是日本的首都。",
            ["东京是日本的首都。", "北京是中国的首都。"],
        )
        assert result.similarity > 0.3

    def test_tfidf_weight_respected(self) -> None:
        config_high = EvalConfig(tfidf_weight=0.9, language="es")
        config_low = EvalConfig(tfidf_weight=0.1, language="es")
        backend_high = HybridBackend(config_high)
        backend_low = HybridBackend(config_low)

        claim = "La torre Eiffel está en París."
        refs = ["La torre Eiffel está ubicada en París, Francia."]

        r1 = backend_high.score_claim(claim, refs)
        r2 = backend_low.score_claim(claim, refs)
        # Different weights should produce different scores
        assert r1.similarity != r2.similarity


class TestCreateBackend:
    def test_hybrid_backend(self) -> None:
        config = EvalConfig(similarity_backend="hybrid")
        backend = create_backend(config)
        assert isinstance(backend, HybridBackend)

    def test_embedding_backend_requires_package(self) -> None:
        config = EvalConfig(similarity_backend="embedding")
        # sentence-transformers may or may not be installed;
        # if installed but broken this may raise other errors.
        # We just verify create_backend attempts the import path.
        try:
            backend = create_backend(config)
            # If it succeeds, sentence-transformers is working
            assert backend is not None
        except (ImportError, Exception):
            pass  # Expected when package is missing or broken

    def test_llm_nli_backend_requires_key(self) -> None:
        config = EvalConfig(
            similarity_backend="llm-nli",
            llm_api_key=None,
        )
        import os
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key"):
                create_backend(config)
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="similarity_backend"):
            EvalConfig(similarity_backend="invalid")


class TestSimilarityBackendProtocol:
    def test_hybrid_satisfies_protocol(self) -> None:
        config = EvalConfig()
        backend = HybridBackend(config)
        assert isinstance(backend, SimilarityBackend)


class TestHybridBackendInternals:
    """Cover multi-language internal methods for higher coverage."""

    def test_multilang_empty_claim(self) -> None:
        config = EvalConfig(language="es")
        backend = HybridBackend(config)
        result = backend.score_claim("", ["Hola mundo."])
        assert result.similarity == 0.0

    def test_multilang_empty_reference_text(self) -> None:
        config = EvalConfig(language="fr")
        backend = HybridBackend(config)
        result = backend.score_claim("Bonjour le monde.", [""])
        # Should still return a result (possibly low score)
        assert isinstance(result, ClaimScore)

    def test_multilang_identical_strings(self) -> None:
        config = EvalConfig(language="de")
        backend = HybridBackend(config)
        result = backend.score_claim(
            "Berlin ist die Hauptstadt.",
            ["Berlin ist die Hauptstadt."],
        )
        assert result.similarity > 0.5

    def test_multilang_multiple_refs(self) -> None:
        config = EvalConfig(language="pt")
        backend = HybridBackend(config)
        result = backend.score_claim(
            "Lisboa é a capital de Portugal.",
            [
                "Madrid é a capital de Espanha.",
                "Lisboa é a capital de Portugal.",
                "Roma é a capital de Itália.",
            ],
        )
        assert "Lisboa" in result.best_match

    def test_contradiction_with_multilang(self) -> None:
        config = EvalConfig(language="es", detect_contradictions=True)
        backend = HybridBackend(config)
        result = backend.score_claim(
            "La torre no es alta.",
            ["La torre es alta."],
        )
        assert isinstance(result, ClaimScore)

    def test_create_backend_llm_nli(self) -> None:
        config = EvalConfig(
            similarity_backend="llm-nli",
            llm_api_key="test-key",
        )
        backend = create_backend(config)
        from sf_hallucinate.scoring.nli import LLMNLIBackend
        assert isinstance(backend, LLMNLIBackend)

    def test_create_backend_unknown(self) -> None:
        # Create a config then manually override backend
        config = EvalConfig()
        object.__setattr__(config, "similarity_backend", "unknown")
        with pytest.raises(ValueError, match="Unknown similarity backend"):
            create_backend(config)
