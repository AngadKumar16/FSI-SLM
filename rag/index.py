"""FAISS index builder and loader for the paper corpus.

Embeds passages using sentence-transformers, builds a FAISS index (Flat or IVF),
and persists both the index and metadata store to disk.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def build_index(
    passages: List[dict],
    model_name: str = "all-MiniLM-L6-v2",
    index_type: str = "Flat",
    output_dir: str = "artifacts/rag_index",
) -> None:
    """Embed passages and build a FAISS index, then save to disk.

    Parameters
    ----------
    passages : list[dict]
        List of passage dicts with keys ``text``, ``source``, ``page``, ``chunk_id``.
    model_name : str
        Sentence-transformers model name for embeddings.
    index_type : str
        FAISS index type: ``"Flat"`` (exact search) or ``"IVF"`` (approximate).
    output_dir : str
        Directory to save the index and metadata.
    """
    from sentence_transformers import SentenceTransformer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[index] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [p["text"] for p in passages]
    print(f"[index] Encoding {len(texts)} passages...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    dim = embeddings.shape[1]

    import faiss

    if index_type == "IVF":
        nlist = min(int(np.sqrt(len(passages))), 100)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings.astype(np.float32))
        index.add(embeddings.astype(np.float32))
        index.nprobe = min(nlist, 10)
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

    # Save index
    index_path = out / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"[index] Index saved to {index_path} (dim={dim}, n_passages={index.ntotal})")

    # Save metadata
    metadata = [
        {
            "text": p["text"],
            "source": p["source"],
            "page": p.get("page", 0),
            "chunk_id": p.get("chunk_id", i),
        }
        for i, p in enumerate(passages)
    ]
    meta_path = out / "metadata.json"
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[index] Metadata saved to {meta_path}")

    # Save model reference
    model_path = out / "model_name.txt"
    with model_path.open("w") as f:
        f.write(model_name)
    print(f"[index] Done.")


def load_index(
    index_dir: str = "artifacts/rag_index",
) -> Tuple:
    """Load a persisted FAISS index, metadata, and embedding model.

    Parameters
    ----------
    index_dir : str
        Directory containing the saved index and metadata.

    Returns
    -------
    tuple
        ``(index, metadata, model)`` where:
        - ``index`` is a ``faiss.Index``
        - ``metadata`` is a list of dicts
        - ``model`` is a ``SentenceTransformer`` instance
    """
    from sentence_transformers import SentenceTransformer

    idx_path = Path(index_dir) / "index.faiss"
    meta_path = Path(index_dir) / "metadata.json"
    model_path = Path(index_dir) / "model_name.txt"

    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {idx_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_path}")

    import faiss

    index = faiss.read_index(str(idx_path))

    with meta_path.open("r") as f:
        metadata = json.load(f)

    model_name = "all-MiniLM-L6-v2"
    if model_path.exists():
        model_name = model_path.read_text().strip()

    print(f"[index] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[index] Loaded index with {index.ntotal} passages, dim={index.d}")
    return index, metadata, model


__all__ = ["build_index", "load_index"]