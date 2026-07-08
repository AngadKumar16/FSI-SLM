"""LoRA fine-tuning for Phi-4-mini-instruct on Apple Silicon.

This module is a thin wrapper around ``mlx_lm.lora`` (the official Apple MLX
training entry point). It exists to:

1. Hold a project-friendly :class:`TrainingConfig` dataclass that mirrors the
   YAML config in ``configs/default.yaml``.
2. Convert the JSONL files produced by ``Data/build_dataset.py`` (which use
   the flat ``{system, user, assistant, text}`` schema) into the
   ``messages`` format that ``mlx_lm.tuner.datasets.ChatDataset`` expects,
   so we can use ``mask_prompt=True`` and train only on the assistant
   response.
3. Expose :func:`train` and :func:`merge_weights` with the project's
   vocabulary (``adapter_path``, ``lora_rank``, ``lora_alpha``, etc.).

We deliberately do *not* re-implement the training loop. As of mlx-lm 0.31
the public API is:

* :func:`mlx_lm.utils.load` -- load base model + tokenizer, optionally with
  ``adapter_path=`` to apply a previously saved adapter.
* :func:`mlx_lm.tuner.linear_to_lora_layers` -- wrap linear layers in LoRA.
  The signature is ``(model, num_layers, config)`` where ``config`` is a
  dict with ``rank``, ``scale`` (== alpha), ``dropout``, and optional
  ``keys``.
* :func:`mlx_lm.tuner.datasets.create_dataset` -- load local JSONL with
  ``{train,valid,test}.jsonl`` into ChatDataset/CompletionsDataset/TextDataset.
* :func:`mlx_lm.lora.train_model` -- the actual training loop, which calls
  :func:`mlx_lm.tuner.train` under the hood. Handles JIT compile, gradient
  checkpointing, distributed init, validation, and saving ``adapter_config.json``
  + ``adapters.safetensors``.
* :func:`mlx_lm.fuse` (via the CLI or programmatically) -- merge LoRA into
  base weights for a deployment-ready checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    """Training configuration mirroring the YAML config keys.

    Attributes
    ----------
    model_name : str
        HuggingFace model name or path.
    adapter_path : str
        Directory in which to save LoRA adapter weights and ``adapter_config.json``.
    dataset_dir : str
        Directory containing ``{train,val,test}.jsonl`` (produced by
        ``Data/build_dataset.py``).
    checkpoint_dir : str
        Directory in which to save full checkpoints (``adapters.safetensors`` +
        a numbered copy) at ``save_every`` intervals.
    resume_from_checkpoint : str or None
        Path to an ``adapters.safetensors`` file to resume from, or None.
    batch_size : int
        Per-device batch size.
    learning_rate : float
        Peak learning rate (after warmup).
    epochs : int
        Number of training epochs.
    iters : int
        Absolute iteration count (overrides ``epochs`` if > 0). This is what
        the underlying ``mlx_lm.tuner.TrainingArgs.iters`` field expects.
    lora_rank : int
        LoRA rank (r).
    lora_alpha : float
        LoRA alpha. In MLX's API this is called ``scale`` and is applied
        directly to the LoRA delta -- not normalized by rank. We pass it
        through unchanged.
    lora_dropout : float
        LoRA dropout probability.
    max_seq_length : int
        Maximum sequence length for tokenization.
    grad_accum_steps : int
        Number of gradient accumulation steps.
    warmup_steps : int
        Number of linear warmup steps (LR ramps from 0 to peak).
    save_every : int
        Save a numbered checkpoint every N iterations. Final adapter is
        always saved to ``adapter_path/adapters.safetensors``.
    eval_every : int
        Run validation every N iterations.
    val_batches : int
        Number of validation batches per evaluation (-1 = entire val set).
    seed : int
        PRNG seed.
    mask_prompt : bool
        If True, only the assistant response contributes to the loss
        (recommended). If False, the loss covers the full sequence.
    grad_checkpoint : bool
        Trade compute for memory by recomputing activations in backward.
    num_layers : int
        Number of transformer layers (counting from the last) to apply LoRA
        to. ``-1`` means all layers.
    clear_cache_threshold : int
        Free the Metal allocator cache between iterations when its size
        exceeds this many bytes. ``0`` disables. Try ~1 GB
        (``1073741824``) if you hit OOM with longer sequences.
    """

    model_name: str = "microsoft/Phi-4-mini-instruct"
    adapter_path: str = "artifacts/adapters"
    dataset_dir: str = "artifacts/dataset"
    checkpoint_dir: str = "artifacts/checkpoints"
    resume_from_checkpoint: Optional[str] = None
    batch_size: int = 4
    learning_rate: float = 2.0e-4
    epochs: int = 3
    iters: int = 0  # if > 0, overrides epochs
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    max_seq_length: int = 2048
    grad_accum_steps: int = 2
    warmup_steps: int = 50
    save_every: int = 500
    eval_every: int = 200
    val_batches: int = 25
    seed: int = 42
    mask_prompt: bool = True
    grad_checkpoint: bool = False
    num_layers: int = -1  # -1 = all layers
    clear_cache_threshold: int = 0  # bytes; free Metal cache when exceeded. 0 disables.


# --------------------------------------------------------------------------- #
# Dataset conversion
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file into a list of dicts."""
    out: List[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _to_messages(rec: dict) -> Optional[dict]:
    """Convert one ``build_dataset`` record to the ``messages`` chat format.

    The build_dataset format has flat ``system``, ``user``, ``assistant`` keys
    plus a pre-rendered ``text``. We drop ``text`` (the library will re-render
    via the tokenizer's chat template) and emit::

        {"messages": [
            {"role": "system", "content": ...},
            {"role": "user", "content": ...},
            {"role": "assistant", "content": ...},
        ]}

    Records missing any of the three roles are skipped.
    """
    system = rec.get("system")
    user = rec.get("user")
    assistant = rec.get("assistant")
    if not (system and user and assistant):
        return None
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def convert_split(src_path: Path, dst_path: Path) -> int:
    """Convert a ``build_dataset`` JSONL split into the ``messages`` format.

    Parameters
    ----------
    src_path : Path
        Input JSONL (``train.jsonl``, ``val.jsonl``, or ``test.jsonl``) with
        flat ``system/user/assistant/text`` records.
    dst_path : Path
        Output JSONL with ``messages`` records. Will be overwritten.

    Returns
    -------
    int
        Number of records written.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with dst_path.open("w") as fout:
        for rec in _read_jsonl(src_path):
            converted = _to_messages(rec)
            if converted is None:
                continue
            fout.write(json.dumps(converted) + "\n")
            n_written += 1
    return n_written


def load_dataset(path: str, tokenizer=None) -> List[Dict]:
    """Backwards-compatible loader: load a JSONL and return raw records.

    The training entry point (``train``) does NOT use this function directly;
    instead it relies on ``mlx_lm.tuner.datasets.create_dataset`` which
    handles tokenization via the chat template. This function is kept for
    callers that want to inspect a raw JSONL file (e.g. evaluation scripts).

    Parameters
    ----------
    path : str
        Path to the JSONL file.
    tokenizer : optional
        Unused here, kept for backwards compatibility with the old signature.

    Returns
    -------
    list[dict]
        List of raw record dicts.
    """
    del tokenizer  # unused
    return _read_jsonl(Path(path))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _build_lora_args(config: TrainingConfig):
    """Construct an ``argparse.Namespace`` matching ``mlx_lm.lora`` expectations.

    The library's training entry point was designed for a CLI; we synthesize
    a Namespace so we can call :func:`mlx_lm.lora.train_model` programmatically.
    """
    import argparse
    import types

    # Convert "epochs" into "iters" -- the library works in iterations.
    # We need to know the dataset size to convert; we look it up here.
    train_path = Path(config.dataset_dir) / "train.jsonl"
    n_train = sum(1 for _ in train_path.open()) if train_path.exists() else 0
    if config.iters > 0:
        iters = config.iters
    elif n_train > 0 and config.batch_size > 0:
        # ceil(n_train / batch_size) * epochs -- one epoch = one full pass
        iters = max(1, -(-n_train // config.batch_size) * config.epochs)
    else:
        iters = 100  # safe fallback

    # mlx-lm's LR schedule advances once per *optimizer* update, not per
    # iteration, so warmup/decay lengths must be divided by grad_accum_steps.
    # Decay spans the whole run so the LR never peaks near the end.
    accum = max(1, config.grad_accum_steps)
    total_opt_steps = max(1, -(-iters // accum))
    warmup_opt_steps = min(
        max(1, -(-config.warmup_steps // accum)) if config.warmup_steps > 0 else 0,
        total_opt_steps // 2,
    )

    return types.SimpleNamespace(
        # Model + data
        model=config.model_name,
        data=config.dataset_dir,
        train=True,
        test=False,
        seed=config.seed,
        # LoRA structure
        fine_tune_type="lora",
        num_layers=config.num_layers,
        lora_parameters={
            "rank": config.lora_rank,
            "scale": config.lora_alpha,
            "dropout": config.lora_dropout,
        },
        # Optimization
        optimizer="adamw",
        optimizer_config={"adamw": {}},
        learning_rate=config.learning_rate,
        lr_schedule=(
            {
                "name": "cosine_decay",
                "arguments": [
                    config.learning_rate,
                    max(1, total_opt_steps - warmup_opt_steps),
                ],
                "warmup": warmup_opt_steps,
                "warmup_init": 0.0,
            }
            if warmup_opt_steps > 0
            else None
        ),
        # Batching
        batch_size=config.batch_size,
        iters=iters,
        max_seq_length=config.max_seq_length,
        grad_accumulation_steps=config.grad_accum_steps,
        grad_checkpoint=config.grad_checkpoint,
        clear_cache_threshold=config.clear_cache_threshold,
        # Logging / saving
        steps_per_report=10,
        steps_per_eval=config.eval_every,
        val_batches=config.val_batches,
        save_every=config.save_every,
        adapter_path=config.adapter_path,
        resume_adapter_file=config.resume_from_checkpoint,
        # Prompt masking
        mask_prompt=config.mask_prompt,
        # Reporting (no-op for now)
        report_to=None,
        project_name=None,
        # Unused by train_model but referenced in some library paths
        config=None,
    )


def train(config: TrainingConfig) -> None:
    """Run LoRA training on the JSONL dataset under ``config.dataset_dir``.

    This is a thin wrapper around :func:`mlx_lm.lora.train_model`, which
    handles the entire training loop (JIT compile, grad accumulation,
    validation, checkpointing, adapter save).

    The dataset files produced by ``Data/build_dataset.py`` have flat
    ``{system, user, assistant}`` keys; we convert them on the fly into the
    ``messages`` chat format that ``mlx_lm.tuner.datasets.ChatDataset``
    expects, so ``mask_prompt=True`` actually works.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    """
    import mlx.core as mx
    from mlx_lm.lora import train_model
    from mlx_lm.tuner.datasets import load_dataset as mlx_load_dataset
    from mlx_lm.utils import load as mlx_load

    # Set seed before any model load
    mx.random.seed(config.seed)
    np.random.seed(config.seed)

    # ------------------------------------------------------------------ #
    # Validate dataset
    # ------------------------------------------------------------------ #
    dataset_dir = Path(config.dataset_dir)
    train_path = dataset_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {train_path}. "
            "Run `python cli.py build-dataset` first."
        )

    # ------------------------------------------------------------------ #
    # Convert dataset to messages format in-place. We write a sibling
    # directory so the original build_dataset output is untouched and
    # re-runs don't have to reconvert. mlx-lm expects ``valid.jsonl`` (not
    # ``val.jsonl``) so we write both names for the validation split.
    # ------------------------------------------------------------------ #
    converted_dir = dataset_dir / "_mlx_messages"
    converted_dir.mkdir(exist_ok=True)
    split_aliases = {"val": "valid"}  # build_dataset uses "val", mlx-lm wants "valid"
    for split in ("train", "val", "test"):
        src = dataset_dir / f"{split}.jsonl"
        if not src.exists():
            continue
        # Always write under the canonical name ("val") so re-runs are deterministic.
        dst = converted_dir / f"{split}.jsonl"
        n = convert_split(src, dst)
        print(f"[train] Converted {n} {split} examples -> {dst}")
        # Mirror to the alias if needed.
        if split in split_aliases:
            alias = converted_dir / f"{split_aliases[split]}.jsonl"
            alias.write_text(dst.read_text())
            print(f"[train]   (also wrote {alias.name} for mlx-lm compatibility)")

    # Point the args at the converted dir
    config_for_call = TrainingConfig(
        **{**config.__dict__, "dataset_dir": str(converted_dir)}
    )
    args = _build_lora_args(config_for_call)

    # ------------------------------------------------------------------ #
    # Load model + tokenizer (with optional resume adapter)
    # ------------------------------------------------------------------ #
    print(f"[train] Loading model: {config.model_name}")
    loaded = mlx_load(config.model_name)
    # mlx_lm.utils.load returns either (model, tokenizer) or
    # (model, tokenizer, config) depending on return_config. We only need
    # the first two.
    model = loaded[0]
    tokenizer_wrapper = loaded[1]
    # mlx_lm returns a TokenizerWrapper, but tuner.datasets.load_dataset
    # type-annotates a PreTrainedTokenizer. The wrapper's _tokenizer is the
    # real HuggingFace tokenizer (verified by inspecting tokenizer_utils.py).
    hf_tokenizer = getattr(tokenizer_wrapper, "_tokenizer", tokenizer_wrapper)

    # ------------------------------------------------------------------ #
    # Load datasets via the library's loader (handles tokenization + masking)
    # ------------------------------------------------------------------ #
    print(f"[train] Loading datasets from {converted_dir}")
    train_set, valid_set, test_set = mlx_load_dataset(args, hf_tokenizer)  # type: ignore[arg-type]
    if not train_set:
        raise RuntimeError(
            "No training examples loaded after conversion. "
            "Check that build_dataset produced non-empty train.jsonl."
        )
    print(
        f"[train] Loaded {len(train_set)} train, "
        f"{len(valid_set)} val, {len(test_set)} test examples"
    )

    # ------------------------------------------------------------------ #
    # Delegate the actual training loop
    # ------------------------------------------------------------------ #
    adapter_path = Path(config.adapter_path)
    print(
        f"[train] Starting LoRA training: "
        f"rank={config.lora_rank}, alpha={config.lora_alpha}, "
        f"dropout={config.lora_dropout}, "
        f"layers={config.num_layers}, "
        f"batch_size={config.batch_size}, "
        f"grad_accum={config.grad_accum_steps}, "
        f"max_seq_length={config.max_seq_length}"
    )
    print(f"[train] Adapter output: {adapter_path}")
    print(f"[train] Iters: {args.iters}, warmup: {config.warmup_steps}")

    train_model(
        args=args,
        model=model,
        train_set=train_set,
        valid_set=valid_set,
    )

    print(f"[train] Training complete. Adapter saved to {adapter_path}")


# --------------------------------------------------------------------------- #
# Adapter merge
# --------------------------------------------------------------------------- #
def merge_weights(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    dequantize: bool = False,
) -> None:
    """Merge LoRA adapter weights into the base model and save.

    Thin wrapper around :mod:`mlx_lm.fuse`, which loads the base model,
    applies the LoRA layers, calls ``.fuse()`` on each, and saves the
    resulting fully-merged model.

    Parameters
    ----------
    base_model_name : str
        HuggingFace model name or path (must match the model the adapter
        was trained on).
    adapter_path : str
        Path to a directory containing ``adapters.safetensors`` and
        ``adapter_config.json`` (produced by :func:`train`).
    output_path : str
        Directory to save the merged model.
    dequantize : bool
        If True and the base model is quantized, also dequantize on merge.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.utils import load as mlx_load, save as mlx_save

    adapter_dir = Path(adapter_path)
    if not (adapter_dir / "adapters.safetensors").exists():
        raise FileNotFoundError(
            f"Adapters not found at {adapter_dir}. "
            "Did training complete successfully?"
        )
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"adapter_config.json not found at {adapter_dir}. "
            "The adapter was saved by an older / incompatible version."
        )

    print(f"[merge] Loading base model: {base_model_name}")
    print(f"[merge] Loading adapters from: {adapter_dir}")
    loaded = mlx_load(
        base_model_name, adapter_path=str(adapter_dir), return_config=True
    )
    # mlx_load returns (model, tokenizer, config) when return_config=True
    if len(loaded) == 3:
        model, tokenizer, config = loaded
    else:  # pragma: no cover - defensive
        model, tokenizer = loaded
        config = {}

    # Fuse every LoRA layer
    fused_linears = [
        (n, m.fuse(dequantize=dequantize))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]
    if not fused_linears:
        raise RuntimeError(
            "No fuseable LoRA layers found in the model after loading "
            "the adapter. Did you load the right adapter?"
        )
    model.update_modules(tree_unflatten(fused_linears))
    print(f"[merge] Fused {len(fused_linears)} LoRA layers into base weights")

    # Save via the library so config / tokenizer come along
    out = Path(output_path)
    print(f"[merge] Saving merged model to {out}")
    mlx_save(str(out), base_model_name, model, tokenizer, config, donate_model=False)
    print(f"[merge] Done.")


__all__ = ["TrainingConfig", "load_dataset", "train", "merge_weights", "convert_split"]