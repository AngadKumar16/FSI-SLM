"""Unified inference runner for trained Phi-4-mini-instruct + LoRA adapter.

This module owns the "load model + adapter, build prompt, generate" pipeline
so that ``cli.py:eval``, ``cli.py:infer``, and ``serve/fastapi_app.py`` all
share one code path. The rule-based ``Data/describe.py`` is still available
as a fallback when no adapter exists or when running in ``--no-model`` mode.

Public API
----------
- :class:`InferenceConfig` -- paths and generation params, mirrors YAML.
- :class:`InferenceRunner` -- loads the model + adapter once, exposes
  :meth:`generate` for a single FeatureRecord (with optional RAG) and
  :meth:`generate_batch` for evaluating a list of records efficiently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Data.schema import FeatureRecord
from train.tokenization.prompt_template import SYSTEM_PROMPT, render_chat


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    """Paths and parameters for inference.

    Defaults match the project's standard layout but every field is
    overridable from the CLI.
    """

    model_name: str = "./artifacts/models/Phi-4-mini-instruct"
    adapter_path: Optional[str] = "artifacts/adapters"
    rag_index_dir: str = "artifacts/rag_index"
    max_tokens: int = 350
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 3           # RAG top_k
    score_threshold: float = 0.3
    seed: int = 42


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class InferenceRunner:
    """Holds a loaded MLX model + optional adapter + RAG index.

    Lazy-loads everything on first use so importing this module is cheap.
    Use as a context manager if you want explicit cleanup, or just call
    :meth:`close` when done.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._hf_tokenizer = None
        self._rag_index = None
        self._rag_metadata = None
        self._rag_model = None

    # ------------------------------------------------------------------ #
    # Lazy loaders
    # ------------------------------------------------------------------ #
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        # Defer heavy imports so this module stays import-cheap.
        from mlx_lm.utils import load as mlx_load

        adapter = self.config.adapter_path
        if adapter is not None and not Path(adapter, "adapters.safetensors").exists():
            print(f"[infer] WARNING: adapter not found at {adapter}, using base model")
            adapter = None
        if adapter is not None:
            print(f"[infer] Loading base model + adapter from {adapter}...")
        else:
            print(f"[infer] Loading base model (no adapter)...")
        loaded = mlx_load(
            self.config.model_name,
            adapter_path=adapter,
        )
        self._model = loaded[0]
        self._tokenizer = loaded[1]
        # mlx-lm returns a TokenizerWrapper; for chat-template + hf utils we
        # need the underlying PreTrainedTokenizer.
        self._hf_tokenizer = getattr(self._tokenizer, "_tokenizer", self._tokenizer)

    def _ensure_rag(self) -> bool:
        """Load FAISS index if available. Returns True if loaded."""
        if self._rag_index is not None:
            return True
        idx_path = Path(self.config.rag_index_dir) / "index.faiss"
        if not idx_path.exists():
            return False
        from rag.index import load_index

        self._rag_index, self._rag_metadata, self._rag_model = load_index(
            self.config.rag_index_dir
        )
        return True

    # ------------------------------------------------------------------ #
    # Prompt building
    # ------------------------------------------------------------------ #
    def build_prompt(
        self,
        feature_record: FeatureRecord,
        retrieved: Optional[list] = None,
    ) -> str:
        """Render the chat-template prompt, optionally with RAG passages."""
        self._ensure_model()
        if retrieved is not None:
            from rag.prompt_assembly import assemble_rag_prompt

            return assemble_rag_prompt(feature_record, retrieved)
        return render_chat(
            SYSTEM_PROMPT,
            feature_record.to_prompt_block(),
            tokenizer=self._hf_tokenizer,
            add_generation_prompt=True,
        )

    def retrieve(
        self, feature_record: FeatureRecord, top_k: Optional[int] = None
    ) -> List:
        """Run RAG retrieval. Returns [] if no index is available."""
        if not self._ensure_rag():
            return []
        from rag.retrieve import build_query, retrieve

        return retrieve(
            query=build_query(feature_record),
            index=self._rag_index,
            metadata=self._rag_metadata,
            model=self._rag_model,
            top_k=top_k or self.config.top_k,
            score_threshold=self.config.score_threshold,
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(
        self,
        feature_record: FeatureRecord,
        use_rag: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate one interpretation. Returns raw text."""
        self._ensure_model()
        retrieved = self.retrieve(feature_record) if use_rag else None
        prompt = self.build_prompt(feature_record, retrieved=retrieved)
        from mlx_lm import generate as mlx_generate

        text = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens or self.config.max_tokens,
            verbose=False,
        )
        return text

    def generate_batch(
        self,
        feature_records: List[FeatureRecord],
        use_rag: bool = False,
        max_tokens: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """Generate interpretations for a list of records.

        Sequentially -- MLX is single-device so batching here just means
        iterating. tqdm-style progress if ``show_progress``.
        """
        out: List[str] = []
        n = len(feature_records)
        for i, fr in enumerate(feature_records):
            if show_progress and n > 1:
                print(f"[infer] {i + 1}/{n} ({fr.experiment_id})", flush=True)
            out.append(self.generate(fr, use_rag=use_rag, max_tokens=max_tokens))
        return out

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._hf_tokenizer = None
        self._rag_index = None
        self._rag_metadata = None
        self._rag_model = None

    def __enter__(self) -> "InferenceRunner":
        return self

    def __exit__(self, *args) -> None:
        self.close()


__all__ = ["InferenceConfig", "InferenceRunner"]