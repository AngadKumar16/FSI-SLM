"""Assemble SFT examples and split them BY EXPERIMENT.

Each example is a chat triple ``{system, user, assistant}`` where:

* ``system``    -- the fixed FSI-interpreter instruction (defines the task and
                   the required output structure).
* ``user``      -- the serialized feature record (verbatim ``to_prompt_block``),
                   optionally wrapped with retrieved RAG passages at *inference*
                   time. For SFT we train WITHOUT RAG passages first (see README).
* ``assistant`` -- the deterministic target description from :mod:`data.describe`.

Phi-4-mini-instruct is sensitive to prompt formatting, so we apply its
*documented* chat template. When ``transformers`` + the model tokenizer are
available we call ``tokenizer.apply_chat_template`` (the ground truth). When they
are not (e.g. building the dataset on a CPU box with no model download), we fall
back to :func:`apply_phi4_chat_template`, a hand-written formatter that matches
Phi-4-mini's template exactly:

    <|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>{assistant}<|end|>

The split is BY ``experiment_id`` (held-out runs) so no run leaks across
train/val/test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from Data.describe import describe
from Data.schema import FeatureRecord
from Data.synthetic import DEFAULT_SEED, generate_dataset
from features.extract import extract_features
from train.tokenization.prompt_template import (
    PHI4_BOS,
    PHI4_END,
    SYSTEM_PROMPT,
    apply_chat_template,
    render_chat,
)

# Backwards-compatible alias: callers historically imported
# ``apply_phi4_chat_template`` from this module.
apply_phi4_chat_template = apply_chat_template


# --------------------------------------------------------------------------- #
# Example construction
# --------------------------------------------------------------------------- #
@dataclass
class SFTExample:
    experiment_id: str
    system: str
    user: str
    assistant: str
    text: str  # fully templated training string

    def to_json(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "system": self.system,
            "user": self.user,
            "assistant": self.assistant,
            "text": self.text,
        }


def build_example(fr: FeatureRecord, tokenizer=None, paraphrase: bool = False) -> SFTExample:
    """Build a single SFT example from a feature record."""
    user = fr.to_prompt_block()  # the experiment's own numbers, verbatim
    assistant = describe(fr, paraphrase_with_api=paraphrase)
    text = render_chat(SYSTEM_PROMPT, user, assistant, tokenizer=tokenizer)
    return SFTExample(
        experiment_id=fr.experiment_id,
        system=SYSTEM_PROMPT,
        user=user,
        assistant=assistant,
        text=text,
    )


def build_rag_example(
    fr: FeatureRecord,
    retrieved: list,
    tokenizer=None,
    paraphrase: bool = False,
    max_passage_chars: Optional[int] = 600,
) -> SFTExample:
    """Build a SFT example that includes retrieved paper passages.

    The user message is augmented with a ``[REFERENCE PASSAGES]`` block
    formatted the same way ``rag.prompt_assembly.assemble_rag_prompt``
    does, so the model sees the same prompt format it will encounter at
    inference time. This teaches the model to ground its answers in
    papers without overriding the experiment's own numbers.

    Parameters
    ----------
    fr : FeatureRecord
        The experiment.
    retrieved : list[RetrievalResult]
        Pre-retrieved passages from the FAISS index.
    tokenizer : optional
        Tokenizer for the chat template.
    paraphrase : bool
        Whether to paraphrase the assistant target.
    max_passage_chars : int, optional
        Cap each retrieved passage to this many characters before formatting.
        Keeps the whole training example within ``max_seq_length`` so we
        don't lose the assistant target to truncation. ``None`` disables.

    Returns
    -------
    SFTExample
    """
    import dataclasses
    from rag.prompt_assembly import _format_passages

    base_user = fr.to_prompt_block()
    if max_passage_chars is None:
        truncated = retrieved
    else:
        truncated = []
        for r in retrieved:
            text = r.passage_text or ""
            if len(text) > max_passage_chars:
                text = text[:max_passage_chars].rsplit(" ", 1)[0] + "..."
            truncated.append(dataclasses.replace(r, passage_text=text))
    ref_block = _format_passages(truncated) if truncated else ""
    user = f"{base_user}\n\n{ref_block}" if ref_block else base_user
    assistant = describe(fr, paraphrase_with_api=paraphrase)
    text = render_chat(SYSTEM_PROMPT, user, assistant, tokenizer=tokenizer)
    return SFTExample(
        experiment_id=fr.experiment_id,
        system=SYSTEM_PROMPT,
        user=user,
        assistant=assistant,
        text=text,
    )


def split_by_experiment(
    feature_records: List[FeatureRecord],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[FeatureRecord]]:
    """Split feature records into train/val/test BY experiment id (no leakage)."""
    ids = sorted({fr.experiment_id for fr in feature_records})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])

    buckets: Dict[str, List[FeatureRecord]] = {"train": [], "val": [], "test": []}
    for fr in feature_records:
        if fr.experiment_id in train_ids:
            buckets["train"].append(fr)
        elif fr.experiment_id in val_ids:
            buckets["val"].append(fr)
        else:
            buckets["test"].append(fr)
    return buckets


def build_dataset(
    out_dir: str = "artifacts/dataset",
    n_per_regime: int = 40,
    seed: int = DEFAULT_SEED,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    paraphrase: bool = False,
    tokenizer=None,
    rag_index_dir: Optional[str] = None,
    rag_top_k: int = 3,
    rag_score_threshold: float = 0.3,
    rag_fraction: float = 0.0,
    rag_max_passage_chars: Optional[int] = 600,
) -> Dict[str, Path]:
    """Generate synthetic data, extract features, build examples, split, and write.

    With ``rag_fraction > 0``, a fraction of training examples (rounded) will
    have retrieved paper passages added to the user message -- the rest stay
    in the original no-RAG format. This trains the model to use retrieved
    context when present and ignore it when not.

    Writes ``{train,val,test}.jsonl`` plus ``feature_records_{split}.jsonl``
    (the raw feature records with ground truth, used by the evaluator).
    Returns a map of split -> path for the chat JSONL files.

    Parameters
    ----------
    rag_index_dir : str, optional
        Path to a FAISS index directory. Required when ``rag_fraction > 0``.
    rag_top_k : int
        Number of passages to retrieve per RAG example.
    rag_score_threshold : float
        Minimum similarity score for a retrieved passage.
    rag_fraction : float
        Fraction in [0, 1] of training examples that should be augmented with
        retrieved passages. Default 0 (no RAG).
    rag_max_passage_chars : int, optional
        Cap each retrieved passage to this many characters before formatting.
        Keeps the training example within ``max_seq_length`` so we don't lose
        the assistant target to truncation.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_experiments = generate_dataset(n_per_regime=n_per_regime, seed=seed)
    feature_records = [extract_features(r) for r in raw_experiments]
    buckets = split_by_experiment(feature_records, ratios=ratios, seed=seed)

    # Lazy RAG setup
    rag_index = rag_metadata = rag_model = None
    if rag_fraction > 0:
        if not rag_index_dir:
            raise ValueError("rag_index_dir is required when rag_fraction > 0")
        from rag.index import load_index
        from rag.retrieve import build_query, retrieve

        rag_index, rag_metadata, rag_model = load_index(rag_index_dir)
        print(f"[build_dataset] RAG: loaded index from {rag_index_dir}")

    paths: Dict[str, Path] = {}
    for split, records in buckets.items():
        chat_path = out / f"{split}.jsonl"
        feat_path = out / f"feature_records_{split}.jsonl"

        # Decide which records in this split get RAG passages.
        # Apply the fraction only to the training split; val/test stay no-RAG
        # so eval can compare fairly.
        if split == "train" and rag_fraction > 0 and rag_index is not None:
            n_rag = max(1, int(round(len(records) * rag_fraction)))
            rag_indices = set(range(n_rag))  # first n_rag are RAG-augmented
            print(
                f"[build_dataset] {split}: {n_rag}/{len(records)} examples will "
                f"include RAG passages"
            )
        else:
            rag_indices = set()

        with chat_path.open("w") as cf, feat_path.open("w") as ff:
            for i, fr in enumerate(records):
                ff.write(fr.model_dump_json() + "\n")
                if i in rag_indices:
                    from rag.retrieve import build_query, retrieve
                    retrieved = retrieve(
                        query=build_query(fr),
                        index=rag_index,
                        metadata=rag_metadata,
                        model=rag_model,
                        top_k=rag_top_k,
                        score_threshold=rag_score_threshold,
                    )
                    if retrieved:
                        ex = build_rag_example(
                            fr, retrieved, tokenizer=tokenizer, paraphrase=paraphrase,
                            max_passage_chars=rag_max_passage_chars,
                        )
                    else:
                        ex = build_example(
                            fr, tokenizer=tokenizer, paraphrase=paraphrase
                        )
                else:
                    ex = build_example(
                        fr, tokenizer=tokenizer, paraphrase=paraphrase
                    )
                cf.write(json.dumps(ex.to_json()) + "\n")
        paths[split] = chat_path
        print(f"[build_dataset] {split}: {len(records)} examples -> {chat_path}")
    return paths


__all__ = [
    "SYSTEM_PROMPT",
    "apply_phi4_chat_template",
    "render_chat",
    "build_example",
    "build_dataset",
    "split_by_experiment",
    "SFTExample",
]
