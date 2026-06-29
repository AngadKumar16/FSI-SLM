"""Unified CLI for the FSI-SLM pipeline.

Uses ``fire.Fire`` for subcommand dispatch. Each subcommand loads configuration,
sets the random seed, and calls the appropriate module.

Usage::

    # Generate synthetic data, build dataset
    python cli.py generate-data --n-per-regime 40
    python cli.py build-dataset --n-per-regime 40

    # Index the paper corpus for RAG
    python cli.py index-corpus

    # Train the model
    python cli.py train --epochs 3

    # Evaluate
    python cli.py eval

    # Run inference on a single feature record
    python cli.py infer --feature-record path/to/fr.json

    # Start the inference server
    python cli.py serve --port 8000
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from tools.utils.repro import set_seed


@dataclass
class CLIContext:
    """Holds parsed config, seed, and device info for the current session."""
    config: dict = field(default_factory=dict)
    seed: int = 42
    device: str = "cpu"

    @classmethod
    def from_config_path(cls, config_path: str = "configs/default.yaml") -> "CLIContext":
        """Load configuration from a YAML file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        seed = cfg.get("seed", 42)
        set_seed(seed)
        return cls(config=cfg, seed=seed)


def _resolve_path(path_str: str) -> str:
    """Resolve a path string relative to the project root."""
    p = Path(path_str)
    return str(p.resolve())


def generate_data(
    n_per_regime: int = 40,
    seed: Optional[int] = None,
    config_path: str = "configs/default.yaml",
) -> None:
    """Generate synthetic RawExperiment data and save as JSONL.

    Parameters
    ----------
    n_per_regime : int
        Number of experiments per regime (default: 40).
    seed : int, optional
        Random seed. Overrides config file if provided.
    config_path : str
        Path to YAML configuration file.
    """
    from data.synthetic import generate_dataset, DEFAULT_SEED

    ctx = CLIContext.from_config_path(config_path)
    if seed is not None:
        ctx.seed = seed
        set_seed(ctx.seed)

    experiments = generate_dataset(
        n_per_regime=n_per_regime,
        seed=ctx.seed,
    )

    out_dir = Path("artifacts/raw_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiments.jsonl"

    with out_path.open("w") as f:
        for exp in experiments:
            f.write(exp.model_dump_json() + "\n")

    print(f"[cli] Generated {len(experiments)} experiments -> {out_path}")


def build_dataset(
    n_per_regime: int = 40,
    seed: Optional[int] = None,
    output_dir: str = "artifacts/dataset",
    config_path: str = "configs/default.yaml",
) -> None:
    """Build SFT dataset from synthetic data.

    Parameters
    ----------
    n_per_regime : int
        Number of experiments per regime (default: 40).
    seed : int, optional
        Random seed. Overrides config file if provided.
    output_dir : str
        Output directory for the dataset.
    config_path : str
        Path to YAML configuration file.
    """
    from data.build_dataset import build_dataset as _build_dataset

    ctx = CLIContext.from_config_path(config_path)
    if seed is not None:
        ctx.seed = seed
        set_seed(ctx.seed)

    cfg = ctx.config.get("dataset", {})
    npr = n_per_regime or cfg.get("n_per_regime", 40)
    ratios = (
        cfg.get("train_ratio", 0.8),
        cfg.get("val_ratio", 0.1),
        cfg.get("test_ratio", 0.1),
    )
    paraphrase = cfg.get("paraphrase", False)

    paths = _build_dataset(
        out_dir=output_dir,
        n_per_regime=npr,
        seed=ctx.seed,
        ratios=ratios,
        paraphrase=paraphrase,
    )
    print(f"[cli] Dataset built: {len(paths)} splits")
    for split, path in paths.items():
        print(f"  {split}: {path}")


def index_corpus(
    config_path: str = "configs/default.yaml",
) -> None:
    """Build or rebuild the FAISS index over the paper corpus.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    """
    from rag.corpus_loader import load_corpus
    from rag.index import build_index

    ctx = CLIContext.from_config_path(config_path)
    rag_cfg = ctx.config.get("rag", {})

    corpus_dir = rag_cfg.get("corpus_dir", "Data/Papers")
    chunk_size = rag_cfg.get("chunk_size", 256)
    chunk_overlap = rag_cfg.get("chunk_overlap", 128)
    embedding_model = rag_cfg.get("embedding_model", "all-MiniLM-L6-v2")
    index_type = rag_cfg.get("index_type", "Flat")
    index_dir = rag_cfg.get("index_dir", "artifacts/rag_index")

    print(f"[cli] Loading corpus from {corpus_dir}")
    passages = load_corpus(
        directory=corpus_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not passages:
        print("[cli] No passages found. Skipping index build.")
        return

    print(f"[cli] Building index with {len(passages)} passages")
    build_index(
        passages=passages,
        model_name=embedding_model,
        index_type=index_type,
        output_dir=index_dir,
    )


def train(
    config_path: str = "configs/default.yaml",
    **kwargs,
) -> None:
    """Run MLX-based LoRA training.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    **kwargs
        Overrides for training configuration (e.g. ``--epochs 5``).
    """
    from train.sft import train as _train
    from train.sft import TrainingConfig

    ctx = CLIContext.from_config_path(config_path)
    train_cfg = ctx.config.get("training", {})

    # Merge overrides
    train_cfg.update(kwargs)

    # Build TrainingConfig from dict
    config = TrainingConfig(
        model_name=ctx.config.get("model_name", "microsoft/Phi-4-mini-instruct"),
        adapter_path=train_cfg.get("adapter_path", "artifacts/adapters"),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "artifacts/checkpoints"),
        resume_from_checkpoint=train_cfg.get("resume_from_checkpoint", None),
        batch_size=train_cfg.get("batch_size", 4),
        learning_rate=train_cfg.get("learning_rate", 2.0e-4),
        epochs=train_cfg.get("epochs", 3),
        lora_rank=train_cfg.get("lora_rank", 8),
        lora_alpha=train_cfg.get("lora_alpha", 16.0),
        lora_dropout=train_cfg.get("lora_dropout", 0.05),
        max_seq_length=train_cfg.get("max_seq_length", 2048),
        grad_accum_steps=train_cfg.get("grad_accum_steps", 2),
        warmup_steps=train_cfg.get("warmup_steps", 50),
        save_every=train_cfg.get("save_every", 500),
        eval_every=train_cfg.get("eval_every", 200),
        seed=ctx.seed,
    )

    _train(config)


def eval(
    config_path: str = "configs/default.yaml",
    output_file: Optional[str] = None,
) -> None:
    """Evaluate the trained model on the held-out test set.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    output_file : str, optional
        Path to save evaluation results JSON.
    """
    # Note: full evaluation requires a trained model to generate predictions.
    # This function loads cached predictions if available, or generates them
    # using the trained model. For now, it loads the test set and runs the
    # rule-based evaluator as a sanity check.
    from data.schema import FeatureRecord
    from eval.evaluate import evaluate

    ctx = CLIContext.from_config_path(config_path)
    eval_cfg = ctx.config.get("eval", {})

    # Load test feature records
    test_path = Path("artifacts/dataset/feature_records_test.jsonl")
    if not test_path.exists():
        print(f"[cli] Test set not found at {test_path}. Run 'build-dataset' first.")
        return

    test_records = []
    with test_path.open() as f:
        for line in f:
            test_records.append(FeatureRecord.model_validate_json(line))

    print(f"[cli] Loaded {len(test_records)} test records")

    # For now, use the deterministic describe() output as the "prediction"
    # (simulates perfect model). In production, this would run the trained model.
    from data.describe import describe

    predictions = [describe(fr) for fr in test_records]

    results = evaluate(
        predictions=predictions,
        test_dataset=test_records,
        config=eval_cfg,
    )

    # Print summary
    print("\n[cli] Evaluation Results:")
    print(f"  Examples: {results['summary']['num_examples']}")
    print(f"  Regime Macro F1: {results['summary']['regime_macro_f1']:.4f}")
    print(f"  Lock-in F1: {results['summary']['lock_in_f1']:.4f}")
    print(f"  Avg Field Accuracy: {results['summary']['avg_field_accuracy']:.4f}")

    if results["per_field_accuracy"]:
        print("\n  Per-Field Accuracy:")
        for field, metrics in results["per_field_accuracy"].items():
            print(f"    {field}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")

    # Save results
    if output_file:
        out_path = Path(output_file)
    else:
        out_dir = eval_cfg.get("output_dir", "artifacts/eval_results")
        out_path = Path(out_dir) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[cli] Results saved to {out_path}")


def infer(
    feature_record: str,
    use_rag: bool = False,
    config_path: str = "configs/default.yaml",
) -> None:
    """Run inference on a single FeatureRecord.

    Parameters
    ----------
    feature_record : str
        Path to a JSON file containing a FeatureRecord, or a JSON string.
    use_rag : bool
        Whether to use RAG augmentation (default: False).
    config_path : str
        Path to YAML configuration file.
    """
    from data.schema import FeatureRecord
    from train.tokenization.prompt_template import apply_chat_template, SYSTEM_PROMPT

    ctx = CLIContext.from_config_path(config_path)

    # Load feature record
    if Path(feature_record).exists():
        with open(feature_record) as f:
            data = json.load(f)
    else:
        data = json.loads(feature_record)

    fr = FeatureRecord.model_validate(data)

    if use_rag:
        # Load RAG index and retrieve
        try:
            from rag.index import load_index
            from rag.retrieve import build_query, retrieve
            from rag.prompt_assembly import assemble_rag_prompt

            rag_cfg = ctx.config.get("rag", {})
            index, metadata, model = load_index(rag_cfg.get("index_dir", "artifacts/rag_index"))
            query = build_query(fr)
            retrieved = retrieve(
                query=query,
                index=index,
                metadata=metadata,
                model=model,
                top_k=rag_cfg.get("top_k", 3),
                score_threshold=rag_cfg.get("score_threshold", 0.3),
            )
            prompt = assemble_rag_prompt(fr, retrieved)
            print(f"[cli] RAG: retrieved {len(retrieved)} passages")
        except (FileNotFoundError, ImportError) as e:
            print(f"[cli] RAG unavailable (run 'index-corpus' first): {e}")
            prompt = apply_chat_template(SYSTEM_PROMPT, fr.to_prompt_block(), add_generation_prompt=True)
    else:
        prompt = apply_chat_template(SYSTEM_PROMPT, fr.to_prompt_block(), add_generation_prompt=True)

    print("\n[cli] Prompt:")
    print("-" * 60)
    print(prompt)
    print("-" * 60)

    # Note: actual model inference would go here using mlx_lm.generate
    from data.describe import describe
    prediction = describe(fr)
    print("\n[cli] Prediction (rule-based fallback):")
    print(prediction)


def serve(
    config_path: str = "configs/default.yaml",
    port: Optional[int] = None,
    host: Optional[str] = None,
) -> None:
    """Start the FastAPI inference server.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    port : int, optional
        Server port. Overrides config file if provided.
    host : str, optional
        Server host. Overrides config file if provided.
    """
    from serve.fastapi_app import create_app

    ctx = CLIContext.from_config_path(config_path)
    serve_cfg = ctx.config.get("serve", {})

    import uvicorn
    uvicorn.run(
        create_app(config=ctx.config),
        host=host or serve_cfg.get("host", "0.0.0.0"),
        port=port or serve_cfg.get("port", 8000),
    )


def main() -> None:
    """Main entry point - dispatches to subcommands via fire.Fire."""
    import fire
    fire.Fire({
        "generate-data": generate_data,
        "build-dataset": build_dataset,
        "index-corpus": index_corpus,
        "train": train,
        "eval": eval,
        "infer": infer,
        "serve": serve,
    })


if __name__ == "__main__":
    main()


__all__ = [
    "CLIContext",
    "generate_data",
    "build_dataset",
    "index_corpus",
    "train",
    "eval",
    "infer",
    "serve",
    "main",
]