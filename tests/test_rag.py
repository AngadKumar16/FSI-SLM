"""Tests for the RAG layer that run without FAISS or an embedding model.

Covers query construction (no label leakage), passage formatting, and the
RAG prompt assembly format contract.
"""

import numpy as np

from Data.schema import FlowRegime
from Data.synthetic import generate_experiment
from features.extract import extract_features
from rag.prompt_assembly import _format_passages, assemble_rag_prompt
from rag.retrieve import RetrievalResult, build_query


def _record(regime=FlowRegime.GALLOPING, seed=3):
    raw = generate_experiment(regime, rng=np.random.default_rng(seed))
    return extract_features(raw)


class TestBuildQuery:
    def test_query_contains_measured_features(self):
        fr = _record()
        q = build_query(fr)
        assert "Reynolds number" in q
        assert "Strouhal number" in q
        assert "Shedding frequency" in q

    def test_no_ground_truth_leakage(self):
        """The retrieval query must not depend on the answer key: real
        experiments have no label, so train and inference retrieval must see
        identical queries for identical measurements."""
        fr = _record()
        q_with_label = build_query(fr)
        fr_no_label = fr.model_copy(update={"ground_truth_regime": None})
        q_without_label = build_query(fr_no_label)
        assert q_with_label == q_without_label
        assert "galloping" not in q_with_label.lower()


class TestPassageFormatting:
    def _passages(self):
        return [
            RetrievalResult(
                passage_text="Lock-in occurs when shedding synchronizes.",
                score=0.91,
                source="williamson1988.pdf",
                page=4,
                chunk_id=12,
            ),
            RetrievalResult(
                passage_text="Galloping is a movement-induced instability.",
                score=0.72,
                source="parkinson1989.pdf",
                page=2,
                chunk_id=7,
            ),
        ]

    def test_format_passages_block(self):
        block = _format_passages(self._passages())
        assert block.startswith("\n[REFERENCE PASSAGES]")
        assert block.rstrip().endswith("[/REFERENCE PASSAGES]")
        assert "williamson1988.pdf" in block
        assert "Passage 1" in block and "Passage 2" in block

    def test_format_passages_empty(self):
        assert _format_passages([]) == ""

    def test_assemble_rag_prompt_structure(self):
        fr = _record()
        prompt = assemble_rag_prompt(fr, self._passages())
        # Feature record precedes passages; generation cue at the end.
        assert prompt.index("FSI_FEATURE_RECORD") < prompt.index("[REFERENCE PASSAGES]")
        assert prompt.rstrip().endswith("<|assistant|>")

    def test_assemble_without_passages_omits_block(self):
        fr = _record()
        prompt = assemble_rag_prompt(fr, [])
        assert "[REFERENCE PASSAGES]" not in prompt
