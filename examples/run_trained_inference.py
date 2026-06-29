"""Run inference with the trained Phi-4-mini-instruct LoRA adapter.

Demonstrates the trained model end-to-end:
1. Load a FeatureRecord.
2. Render the chat-template prompt.
3. Call mlx_lm.generate with the adapter.
4. Print the model's interpretation.

Usage:
    python examples/run_trained_inference.py [--adapter-path PATH] [--record-id ID]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="./artifacts/models/Phi-4-mini-instruct",
        help="Base model path or HF repo.",
    )
    parser.add_argument(
        "--adapter-path",
        default="./artifacts/adapters_test",
        help="Path to LoRA adapter directory.",
    )
    parser.add_argument(
        "--record-id",
        default=None,
        help="Experiment ID to load from artifacts/dataset/feature_records_test.jsonl "
        "(default: first record).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--feature-records",
        default="artifacts/dataset/feature_records_test.jsonl",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load the FeatureRecord
    # ------------------------------------------------------------------ #
    fr_path = Path(args.feature_records)
    if not fr_path.exists():
        raise SystemExit(
            f"Feature records not found at {fr_path}. "
            "Run `python cli.py build-dataset` first."
        )

    with fr_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    if args.record_id is not None:
        rec = next((r for r in records if r["experiment_id"] == args.record_id), None)
        if rec is None:
            raise SystemExit(
                f"Record {args.record_id!r} not found. "
                f"Available: {[r['experiment_id'] for r in records[:5]]}..."
            )
    else:
        rec = records[0]
    print(f"[infer] Using experiment_id={rec['experiment_id']}")

    # ------------------------------------------------------------------ #
    # 2. Build the prompt
    # ------------------------------------------------------------------ #
    from Data.schema import FeatureRecord
    from train.tokenization.prompt_template import (
        SYSTEM_PROMPT,
        render_chat,
    )

    fr = FeatureRecord.model_validate(rec)
    user_block = fr.to_prompt_block()

    # Use the wrapped tokenizer (already mlx-format aware) for the chat template.
    from mlx_lm.utils import load as mlx_load
    _, tokenizer_wrapper = mlx_load(args.model, adapter_path=args.adapter_path)
    hf_tok = getattr(tokenizer_wrapper, "_tokenizer", tokenizer_wrapper)

    prompt = render_chat(
        SYSTEM_PROMPT,
        user_block,
        tokenizer=hf_tok,
        add_generation_prompt=True,
    )

    print("=" * 60)
    print("PROMPT (truncated to 800 chars):")
    print(prompt[:800] + ("..." if len(prompt) > 800 else ""))
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 3. Generate with the trained model + adapter
    # ------------------------------------------------------------------ #
    from mlx_lm import generate

    # Reload the model fresh with adapter so generation uses the trained weights.
    print(f"[infer] Loading model + adapter from {args.adapter_path}...")
    model, tokenizer = mlx_load(args.model, adapter_path=args.adapter_path)

    print("[infer] Generating...")
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        verbose=False,
    )

    print("=" * 60)
    print("MODEL OUTPUT:")
    print(response)
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 4. Validate the output
    # ------------------------------------------------------------------ #
    from eval.validate_output import validate_output

    result = validate_output(response)
    print()
    print(f"[infer] Validation: valid={result.valid}")
    print(f"[infer] Parsed fields: {result.parsed_fields}")
    if result.errors:
        print(f"[infer] Errors: {result.errors}")
    if result.warnings:
        print(f"[infer] Warnings: {result.warnings}")

    # ------------------------------------------------------------------ #
    # 5. Compare to ground truth
    # ------------------------------------------------------------------ #
    print()
    print("[infer] Ground truth regime:", fr.ground_truth_regime)
    print("[infer] Ground truth lock-in:", fr.lock_in_status().value)


if __name__ == "__main__":
    main()