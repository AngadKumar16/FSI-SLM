"""Retrieval from FAISS index for RAG-augmented inference.

Given a ``FeatureRecord``, builds a query embedding, searches the FAISS index,
and returns the top-k most similar passages with metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from data.schema import FeatureRecord


@dataclass
class RetrievalResult:
    """A single retrieved passage with relevance metadata.

    Attributes
    ----------
    passage_text : str
        The text content of the retrieved passage.
    score : float
        Cosine similarity score (inner product of normalized embeddings).
    source : str
        Source filename of the passage.
    page : int
        Page number within the source document.
    chunk_id : int
        Chunk index within the source document.
    """
    passage_text: str
    score: float
    source: str
    page: int
    chunk_id: int


def build_query(feature_record: FeatureRecord) -> str:
    """Build a text query from a FeatureRecord for semantic retrieval.

    The query concatenates key physical parameters to find passages about similar
    flow conditions and regimes.

    Parameters
    ----------
    feature_record : FeatureRecord
        The experiment feature record.

    Returns
    -------
    str
        A text query string for embedding-based retrieval.
    """
    c = feature_record.conditions
    p = feature_record.piv
    s = feature_record.structure

    regime_label = ""
    if feature_record.ground_truth_regime is not None:
        regime_label = feature_record.ground_truth_regime.value

    query = (
        f"Flow regime: {regime_label}. "
        f"Reynolds number: {c.reynolds_number:.4g}, "
        f"angle of attack: {c.angle_of_attack_deg:.1f} deg, "
        f"reduced velocity: {c.reduced_velocity:.2f}. "
        f"Shedding frequency: {p.shedding_frequency_hz:.2f} Hz, "
        f"Strouhal number: {p.strouhal_number:.4f}. "
        f"Structural natural frequency: {s.natural_frequency_hz:.2f} Hz, "
        f"RMS amplitude: {s.rms_amplitude_over_c:.4f}c. "
        f"Separation at x/c = {p.separation_location_x_over_c:.2f}, "
        f"wake width: {p.wake_width_over_c:.2f}c."
    )
    return query


def retrieve(
    query: str,
    index,
    metadata: List[dict],
    model,
    top_k: int = 3,
    score_threshold: float = 0.3,
) -> List[RetrievalResult]:
    """Search the FAISS index and return the top-k most relevant passages.

    Parameters
    ----------
    query : str
        Text query string.
    index : faiss.Index
        The FAISS index to search.
    metadata : list[dict]
        List of metadata dicts corresponding to index positions.
    model : SentenceTransformer
        The embedding model.
    top_k : int
        Number of top results to return.
    score_threshold : float
        Minimum similarity score for a result to be included.

    Returns
    -------
    list[RetrievalResult]
        List of retrieval results sorted by score descending.
    """
    if index.ntotal == 0:
        return []

    # Clip top_k to index size
    k = min(top_k, index.ntotal)

    # Encode query
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    # Search
    scores, indices = index.search(query_embedding, k)

    results: List[RetrievalResult] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        if score < score_threshold:
            continue
        m = metadata[idx]
        results.append(
            RetrievalResult(
                passage_text=m["text"],
                score=float(score),
                source=m["source"],
                page=m.get("page", 0),
                chunk_id=m.get("chunk_id", int(idx)),
            )
        )

    return results


__all__ = ["RetrievalResult", "build_query", "retrieve"]