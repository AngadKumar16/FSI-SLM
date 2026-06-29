"""MLX-based LoRA fine-tuning for Phi-4-mini-instruct on Apple Silicon.

This module provides the training loop for supervised fine-tuning (SFT) using
MLX (Apple's machine learning framework for Apple Silicon). It loads a base model
(e.g. Phi-4-mini-instruct), applies LoRA adapters, trains on a JSONL dataset
produced by ``Data/build_dataset.py``, and saves the adapter weights.

Key features:
- LoRA with configurable rank, alpha, and dropout
- Checkpointing and resume capability
- Gradient accumulation
- Warmup steps
- Periodic evaluation on validation set
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TrainingConfig:
    """Training configuration mirroring the YAML config keys.

    Attributes
    ----------
    model_name : str
        HuggingFace model name or path.
    adapter_path : str
        Path to save LoRA adapter weights.
    checkpoint_dir : str
        Path to save training checkpoints.
    resume_from_checkpoint : str or None
        Path to a checkpoint to resume from, or None.
    batch_size : int
        Per-device batch size.
    learning_rate : float
        Peak learning rate.
    epochs : int
        Number of training epochs.
    lora_rank : int
        LoRA rank (r).
    lora_alpha : float
        LoRA alpha scaling factor.
    lora_dropout : float
        LoRA dropout probability.
    max_seq_length : int
        Maximum sequence length for tokenization.
    grad_accum_steps : int
        Number of gradient accumulation steps.
    warmup_steps : int
        Number of linear warmup steps.
    save_every : int
        Save checkpoint every N steps.
    eval_every : int
        Run evaluation every N steps.
    seed : int
        Random seed.
    """
    model_name: str = "microsoft/Phi-4-mini-instruct"
    adapter_path: str = "artifacts/adapters"
    checkpoint_dir: str = "artifacts/checkpoints"
    resume_from_checkpoint: Optional[str] = None
    batch_size: int = 4
    learning_rate: float = 2.0e-4
    epochs: int = 3
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    max_seq_length: int = 2048
    grad_accum_steps: int = 2
    warmup_steps: int = 50
    save_every: int = 500
    eval_every: int = 200
    seed: int = 42


def load_dataset(path: str, tokenizer) -> List[Dict]:
    """Load a JSONL dataset and tokenize the ``text`` field.

    Parameters
    ----------
    path : str
        Path to the JSONL file (each line is a dict with a ``text`` key).
    tokenizer : mlx_lm.tokenizer_utils.TokenizerWrapper
        The MLX tokenizer.

    Returns
    -------
    list[dict]
        List of tokenized examples with keys ``input_ids`` and ``labels``.
    """
    examples = []
    max_seq_length = getattr(tokenizer, "max_seq_length", 2048)
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "")
            if not text:
                continue
            # Tokenize
            tokens = tokenizer.encode(text)
            # Truncate to max_seq_length
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
            examples.append({"input_ids": tokens, "labels": tokens.copy()})
    return examples


def _linear_warmup(step: int, warmup_steps: int) -> float:
    """Linear warmup multiplier: 0 -> 1 over warmup_steps."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def train(config: TrainingConfig) -> None:
    """Run MLX-based LoRA training.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm.utils import load, save_config
    from mlx_lm.tuner import create_lora_parameters, linear_to_lora_layers

    from train.tokenization.prompt_template import SYSTEM_PROMPT, render_chat

    # Set seed
    mx.random.seed(config.seed)
    np.random.seed(config.seed)

    # Create output directories
    adapter_dir = Path(config.adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load model and tokenizer
    # ------------------------------------------------------------------ #
    print(f"[train] Loading model: {config.model_name}")
    model, tokenizer = load(config.model_name)
    model.eval()

    # Apply LoRA to all linear layers in the transformer
    print(f"[train] Applying LoRA (rank={config.lora_rank}, alpha={config.lora_alpha})")
    linear_to_lora_layers(
        model,
        config.lora_rank,
        config.lora_alpha,
        config.lora_dropout,
    )

    # Create trainable parameters
    trainable_params = create_lora_parameters(model)
    num_params = sum(p.size for p in trainable_params.values())
    print(f"[train] Trainable parameters: {num_params:,}")

    # ------------------------------------------------------------------ #
    # Load datasets
    # ------------------------------------------------------------------ #
    train_path = Path(config.adapter_path).parent.parent / "dataset" / "train.jsonl"
    val_path = Path(config.adapter_path).parent.parent / "dataset" / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {train_path}. "
            "Run 'python -m Data.build_dataset' first."
        )

    print(f"[train] Loading training data from {train_path}")
    train_examples = load_dataset(str(train_path), tokenizer)
    print(f"[train] Loaded {len(train_examples)} training examples")

    val_examples = []
    if val_path.exists():
        val_examples = load_dataset(str(val_path), tokenizer)
        print(f"[train] Loaded {len(val_examples)} validation examples")

    # ------------------------------------------------------------------ #
    # Optimizer and scheduler
    # ------------------------------------------------------------------ #
    # Use a learning rate schedule with linear warmup
    def lr_schedule(step: int) -> float:
        scale = _linear_warmup(step, config.warmup_steps)
        return config.learning_rate * scale

    optimizer = optim.AdamW(learning_rate=lr_schedule(0))

    # ------------------------------------------------------------------ #
    # Training state
    # ------------------------------------------------------------------ #
    start_step = 0
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    # Resume from checkpoint if specified
    if config.resume_from_checkpoint:
        ckpt_path = Path(config.resume_from_checkpoint)
        if ckpt_path.exists():
            print(f"[train] Resuming from checkpoint: {ckpt_path}")
            model.load_weights(str(ckpt_path / "adapters.safetensors"))
            # Load optimizer state if available
            opt_state_path = ckpt_path / "optimizer_state.pkl"
            if opt_state_path.exists():
                import pickle
                with open(opt_state_path, "rb") as f:
                    state = pickle.load(f)
                optimizer.state = state["optimizer"]
                global_step = state.get("global_step", 0)
                start_step = global_step
                print(f"[train] Resumed at step {global_step}")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    total_steps = len(train_examples) * config.epochs // config.batch_size
    accum_counter = 0
    accum_grads = None

    print(f"[train] Starting training for {config.epochs} epochs ({total_steps} steps)")
    print(f"[train] batch_size={config.batch_size}, grad_accum={config.grad_accum_steps}")

    for epoch in range(start_epoch, config.epochs):
        # Shuffle training examples
        np.random.shuffle(train_examples)

        for batch_start in range(0, len(train_examples), config.batch_size):
            batch = train_examples[batch_start : batch_start + config.batch_size]

            # Skip incomplete batches
            if len(batch) < config.batch_size:
                continue

            # Prepare batch tensors
            batch_input_ids = []
            batch_labels = []
            for ex in batch:
                batch_input_ids.append(ex["input_ids"])
                batch_labels.append(ex["labels"])

            # Pad sequences to max length in batch
            max_len = max(len(ids) for ids in batch_input_ids)
            max_len = min(max_len, config.max_seq_length)

            padded_ids = []
            padded_labels = []
            for ids, labs in zip(batch_input_ids, batch_labels):
                pad_len = max_len - len(ids)
                if pad_len > 0:
                    padded_ids.append(ids + [tokenizer.eos_token_id] * pad_len)
                    padded_labels.append(labs + [-100] * pad_len)
                else:
                    padded_ids.append(ids[:max_len])
                    padded_labels.append(labs[:max_len])

            # Convert to MLX arrays
            x = mx.array(padded_ids)
            y = mx.array(padded_labels)

            # Define loss function over trainable params
            def loss_fn(params):
                model.update(params)
                logits = model(x)
                # Cross-entropy loss
                B, T, V = logits.shape
                logits = logits.reshape(-1, V)
                targets = y.reshape(-1)
                loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
                return loss

            # Compute loss and gradients
            loss_and_grad_fn = nn.value_and_grad(loss_fn)
            loss_val, grads = loss_and_grad_fn(trainable_params)

            # Accumulate gradients
            if accum_grads is None:
                accum_grads = {k: v for k, v in grads.items()}
            else:
                for k in accum_grads:
                    accum_grads[k] = accum_grads[k] + grads[k]

            accum_counter += 1

            # Gradient accumulation step
            if accum_counter >= config.grad_accum_steps:
                # Scale gradients by accumulation steps
                for k in accum_grads:
                    accum_grads[k] = accum_grads[k] / config.grad_accum_steps

                # Update learning rate with warmup
                lr = lr_schedule(global_step)
                optimizer.learning_rate = mx.array(lr)

                # Apply gradients
                optimizer.update(model, accum_grads)
                mx.eval(model.parameters())
                mx.eval(optimizer.state)

                # Reset accumulation
                accum_grads = None
                accum_counter = 0
                global_step += 1

                # Logging
                if global_step % 10 == 0:
                    print(
                        f"[train] Epoch {epoch + 1}/{config.epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {loss_val.item():.4f} | "
                        f"LR: {lr:.2e}"
                    )

                # Evaluation
                if config.eval_every > 0 and global_step % config.eval_every == 0:
                    val_loss = _evaluate(model, val_examples, tokenizer, config)
                    print(f"[train] Validation loss: {val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_adapter(model, str(adapter_dir / "best"))
                        print(f"[train] New best model saved (val_loss={val_loss:.4f})")

                # Save checkpoint
                if config.save_every > 0 and global_step % config.save_every == 0:
                    ckpt_path = ckpt_dir / f"step_{global_step}"
                    _save_checkpoint(model, optimizer, global_step, str(ckpt_path))
                    print(f"[train] Checkpoint saved at step {global_step}")

    # ------------------------------------------------------------------ #
    # Save final adapter
    # ------------------------------------------------------------------ #
    _save_adapter(model, str(adapter_dir))
    print(f"[train] Training complete. Adapter saved to {adapter_dir}")

    # Save training config
    config_path = adapter_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model_name": config.model_name,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_seq_length": config.max_seq_length,
            "global_step": global_step,
        }, f, indent=2)


def _evaluate(model, val_examples, tokenizer, config: TrainingConfig) -> float:
    """Evaluate the model on validation examples.

    Parameters
    ----------
    model : nn.Module
        The MLX model.
    val_examples : list[dict]
        List of tokenized validation examples.
    tokenizer : TokenizerWrapper
        The MLX tokenizer.
    config : TrainingConfig
        Training configuration.

    Returns
    -------
    float
        Average validation loss.
    """
    import mlx.core as mx
    import mlx.nn as nn

    if not val_examples:
        return float("inf")

    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch_start in range(0, len(val_examples), config.batch_size):
        batch = val_examples[batch_start : batch_start + config.batch_size]
        if len(batch) < config.batch_size:
            continue

        batch_input_ids = [ex["input_ids"] for ex in batch]
        batch_labels = [ex["labels"] for ex in batch]

        max_len = min(max(len(ids) for ids in batch_input_ids), config.max_seq_length)
        padded_ids = []
        padded_labels = []
        for ids, labs in zip(batch_input_ids, batch_labels):
            pad_len = max_len - len(ids)
            if pad_len > 0:
                padded_ids.append(ids + [tokenizer.eos_token_id] * pad_len)
                padded_labels.append(labs + [-100] * pad_len)
            else:
                padded_ids.append(ids[:max_len])
                padded_labels.append(labs[:max_len])

        x = mx.array(padded_ids)
        y = mx.array(padded_labels)

        logits = model(x)
        B, T, V = logits.shape
        logits = logits.reshape(-1, V)
        targets = y.reshape(-1)
        loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def _save_adapter(model, path: str) -> None:
    """Save LoRA adapter weights.

    Parameters
    ----------
    model : nn.Module
        The MLX model with LoRA layers.
    path : str
        Output directory path.
    """
    from mlx_lm.tuner import get_lora_layers

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    lora_layers = get_lora_layers(model)
    weights = {k: v for k, v in lora_layers.items() if v is not None}

    import mlx.core as mx
    mx.save_safetensors(str(out / "adapters.safetensors"), weights)
    print(f"[train] Adapter saved to {out / 'adapters.safetensors'}")


def _save_checkpoint(model, optimizer, global_step: int, path: str) -> None:
    """Save a full training checkpoint.

    Parameters
    ----------
    model : nn.Module
        The MLX model.
    optimizer : optim.Optimizer
        The MLX optimizer.
    global_step : int
        Current global training step.
    path : str
        Output directory path.
    """
    import pickle

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    _save_adapter(model, str(out))

    opt_state = {"optimizer": optimizer.state, "global_step": global_step}
    with open(out / "optimizer_state.pkl", "wb") as f:
        pickle.dump(opt_state, f)

    print(f"[train] Checkpoint saved to {out}")


def merge_weights(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
) -> None:
    """Merge LoRA adapter weights into the base model and save.

    Parameters
    ----------
    base_model_name : str
        HuggingFace model name or path for the base model.
    adapter_path : str
        Path to the saved LoRA adapter weights.
    output_path : str
        Path to save the merged model.
    """
    from mlx_lm.utils import load, save_config
    from mlx_lm.tuner import linear_to_lora_layers, get_lora_layers

    import mlx.core as mx

    print(f"[merge] Loading base model: {base_model_name}")
    model, tokenizer = load(base_model_name)

    # Load adapter config
    adapter_dir = Path(adapter_path)
    config_path = adapter_dir / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_cfg = json.load(f)
        lora_rank = train_cfg.get("lora_rank", 8)
        lora_alpha = train_cfg.get("lora_alpha", 16.0)
        lora_dropout = train_cfg.get("lora_dropout", 0.05)
    else:
        lora_rank = 8
        lora_alpha = 16.0
        lora_dropout = 0.05

    # Apply LoRA structure
    linear_to_lora_layers(model, lora_rank, lora_alpha, lora_dropout)

    # Load adapter weights
    adapter_weights = mx.load(str(adapter_dir / "adapters.safetensors"))
    lora_layers = get_lora_layers(model)
    for k, v in adapter_weights.items():
        if k in lora_layers:
            lora_layers[k] = v

    # Update model with merged weights
    model.update(lora_layers)

    # Save merged model
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(out / "model.safetensors"))
    save_config(str(out), config=model.config)
    print(f"[merge] Merged model saved to {out}")


__all__ = ["TrainingConfig", "train", "merge_weights"]