"""Corpus loader: extract text from PDFs and chunk into passages.

Walks a directory (default ``Data/Papers/``), extracts text from PDF files using
``pdfminer.six``, and chunks the text into overlapping passages for embedding
and indexing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List


def _extract_text_from_pdf(path: str) -> str:
    """Extract raw text from a PDF file using pdfminer.

    Parameters
    ----------
    path : str
        Path to the PDF file.

    Returns
    -------
    str
        Extracted text content.
    """
    from pdfminer.high_level import extract_text

    return extract_text(path)


def _chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 128,
) -> List[dict]:
    """Split text into overlapping chunks of approximately ``chunk_size`` tokens.

    Token count is approximated as ``len(text.split())`` (whitespace tokenization).
    Each chunk is a dict with keys ``text``, ``chunk_id``.

    Parameters
    ----------
    text : str
        The full text to chunk.
    chunk_size : int
        Approximate number of tokens per chunk.
    overlap : int
        Approximate number of overlapping tokens between consecutive chunks.

    Returns
    -------
    list[dict]
        List of chunk dicts with keys ``text`` and ``chunk_id``.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[dict] = []
    stride = chunk_size - overlap
    if stride < 1:
        stride = 1

    for i in range(0, len(words), stride):
        chunk_words = words[i : i + chunk_size]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words)
        # Skip chunks that are mostly whitespace/empty
        if len(chunk_text.strip()) < 10:
            continue
        chunks.append({"text": chunk_text, "chunk_id": len(chunks)})

    return chunks


def load_corpus(
    directory: str = "Data/Papers",
    chunk_size: int = 256,
    chunk_overlap: int = 128,
) -> List[dict]:
    """Walk a directory, extract text from PDFs, and chunk into passages.

    Parameters
    ----------
    directory : str
        Path to the directory containing PDF files.
    chunk_size : int
        Approximate number of tokens per chunk.
    chunk_overlap : int
        Approximate number of overlapping tokens between consecutive chunks.

    Returns
    -------
    list[dict]
        List of passage dicts with keys:
        - ``text``: the chunk text
        - ``source``: the source filename
        - ``page``: page number (approximate, from pdfminer metadata if available)
        - ``chunk_id``: unique chunk index within the source document
    """
    corpus_dir = Path(directory)
    if not corpus_dir.exists():
        print(f"[corpus_loader] Warning: directory '{directory}' does not exist. Returning empty corpus.")
        return []

    passages: List[dict] = []
    pdf_files = sorted(corpus_dir.glob("*.pdf")) + sorted(corpus_dir.glob("*.PDF"))

    if not pdf_files:
        print(f"[corpus_loader] Warning: no PDF files found in '{directory}'. Returning empty corpus.")
        return []

    for pdf_path in pdf_files:
        try:
            print(f"[corpus_loader] Extracting: {pdf_path.name}")
            text = _extract_text_from_pdf(str(pdf_path))
            # Clean text: collapse whitespace, remove null bytes
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                print(f"[corpus_loader] Warning: no text extracted from {pdf_path.name}")
                continue

            chunks = _chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            for chunk in chunks:
                passages.append({
                    "text": chunk["text"],
                    "source": pdf_path.name,
                    "page": 0,  # pdfminer doesn't easily give per-chunk page numbers
                    "chunk_id": chunk["chunk_id"],
                })
            print(f"[corpus_loader]   -> {len(chunks)} chunks from {pdf_path.name}")
        except Exception as e:
            print(f"[corpus_loader] Error processing {pdf_path.name}: {e}")

    print(f"[corpus_loader] Total: {len(passages)} passages from {len(pdf_files)} files")
    return passages


__all__ = ["load_corpus"]