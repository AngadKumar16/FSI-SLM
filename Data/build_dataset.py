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

from data.describe import describe
from data.schema import FeatureRecord
from data.synthetic import DEFAULT_SEED, generate_dataset
from features.extract import extract_features

# --------------------------------------------------------------------------- #
# Phi-4-mini-instruct prompt format
# --------------------------------------------------------------------------- #
# These tokens are part of Phi-4-mini-instruct's documented chat template. A turn
# is "<|role|>" + content + "<|end|>". add_generation_prompt appends a bare
# "<|assistant|>" to cue generation. Do NOT change these without re-checking the
# model card -- the model is format-sensitive.
PHI4_BOS = "<|endoftext|>"
PHI4_END = "<|end|>"

SYSTEM_PROMPT = (
    "You are an expert in fluid-structure interaction (FSI) and aeroelasticity. "
    "You are given a serialized feature record from a single airfoil/hydrofoil/"
    "membrane experiment. The feature record is the ONLY source of truth for this "
    "experiment's measured numbers. Interpret it and respond with a structured "
    "description that states, in this order: (1) the flow regime "
    "(attached, separated, VIV lock-in, galloping, or stall flutter); "
    "(2) the dominant frequency in Hz and the Strouhal number St; "
    "(3) the lock-in status (compare shedding frequency to the structural natural "
    "frequency); (4) the amplitude trend; (5) the mode shape; and (6) a one-line "
    "mechanism note. If reference passages from the literature are provided, you "
    "may use them for domain grounding only -- never let them override the "
    "experiment's own measured values."
)


def apply_phi4_chat_template(
    system: str,
    user: str,
    assistant: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    """Hand-written Phi-4-mini chat template (offline fallback).

    If ``assistant`` is provided, a full training string is returned (ending with
    ``<|end|>``). If ``assistant`` is None and ``add_generation_prompt`` is True,
    the string ends with a bare ``<|assistant|>`` ready for generation.
    """
    parts = [f"<|system|>{system}{PHI4_END}", f"<|user|>{user}{PHI4_END}"]
    if assistant is not None:
        parts.append(f"<|assistant|>{assistant}{PHI4_END}")
    elif add_generation_prompt:
        parts.append("<|assistant|>")
    return "".join(parts)


def render_chat(
    system: str,
    user: str,
    assistant: Optional[str] = None,
    add_generation_prompt: bool = False,
    tokenizer=None,
) -> str:
    """Render a chat into a single string using the REAL tokenizer if given.

    Prefers ``tokenizer.apply_chat_template`` (authoritative); otherwise uses the
    documented offline fallback above.
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt and assistant is None,
        )
    return apply_phi4_chat_template(system, user, assistant, add_generation_prompt)


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
) -> Dict[str, Path]:
    """Generate synthetic data, extract features, build examples, split, and write.

    Writes ``{train,val,test}.jsonl`` plus ``feature_records_{split}.jsonl`` (the
    raw feature records with ground truth, used by the evaluator). Returns a map of
    split -> path for the chat JSONL files.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_experiments = generate_dataset(n_per_regime=n_per_regime, seed=seed)
    feature_records = [extract_features(r) for r in raw_experiments]
    buckets = split_by_experiment(feature_records, ratios=ratios, seed=seed)

    paths: Dict[str, Path] = {}
    for split, records in buckets.items():
        chat_path = out / f"{split}.jsonl"
        feat_path = out / f"feature_records_{split}.jsonl"
        with chat_path.open("w") as cf, feat_path.open("w") as ff:
            for fr in records:
                ex = build_example(fr, tokenizer=tokenizer, paraphrase=paraphrase)
                cf.write(json.dumps(ex.to_json()) + "\n")
                ff.write(fr.model_dump_json() + "\n")
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
