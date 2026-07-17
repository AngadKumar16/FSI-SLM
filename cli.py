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

    # Launch the Gradio web UI (provide data, interpret, ask questions)
    python cli.py ui --port 7860
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

# transformers tries to import TensorFlow when present; a broken TF install
# segfaults RAG/serving paths. Nothing here needs TF -- torch/MLX only.
os.environ.setdefault("USE_TF", "0")

# Ensure the project root is on sys.path so that ``Data`` (the package
# directory) is importable when this file is invoked as ``python cli.py``
# from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
    from Data.synthetic import generate_dataset, DEFAULT_SEED

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
    rag_index_dir: Optional[str] = None,
    rag_fraction: float = 0.0,
    rag_top_k: int = 3,
    rag_score_threshold: float = 0.3,
    rag_max_passage_chars: int = 600,
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
    rag_index_dir : str, optional
        Path to a FAISS index (e.g. ``artifacts/rag_index``). When provided
        together with ``rag_fraction > 0``, a fraction of training examples
        will be augmented with retrieved passages.
    rag_fraction : float
        Fraction in [0, 1] of training examples to augment with RAG.
        Default 0 (no RAG).
    rag_top_k : int
        Number of passages to retrieve per RAG example.
    rag_score_threshold : float
        Minimum similarity score for a retrieved passage.
    """
    from Data.build_dataset import build_dataset as _build_dataset

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
        rag_index_dir=rag_index_dir,
        rag_fraction=rag_fraction,
        rag_top_k=rag_top_k,
        rag_score_threshold=rag_score_threshold,
        rag_max_passage_chars=rag_max_passage_chars,
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
        dataset_dir=train_cfg.get("dataset_dir", "artifacts/dataset"),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "artifacts/checkpoints"),
        resume_from_checkpoint=train_cfg.get("resume_from_checkpoint", None),
        batch_size=train_cfg.get("batch_size", 4),
        learning_rate=train_cfg.get("learning_rate", 2.0e-4),
        epochs=train_cfg.get("epochs", 3),
        iters=train_cfg.get("iters", 0),
        lora_rank=train_cfg.get("lora_rank", 8),
        lora_alpha=train_cfg.get("lora_alpha", 16.0),
        lora_dropout=train_cfg.get("lora_dropout", 0.05),
        max_seq_length=train_cfg.get("max_seq_length", 2048),
        grad_accum_steps=train_cfg.get("grad_accum_steps", 2),
        warmup_steps=train_cfg.get("warmup_steps", 50),
        save_every=train_cfg.get("save_every", 500),
        eval_every=train_cfg.get("eval_every", 200),
        val_batches=train_cfg.get("val_batches", 25),
        seed=ctx.seed,
        mask_prompt=train_cfg.get("mask_prompt", True),
        grad_checkpoint=train_cfg.get("grad_checkpoint", False),
        num_layers=train_cfg.get("num_layers", -1),
    )

    _train(config)


def _build_inference_config(ctx: "CLIContext"):
    """Build an InferenceConfig from the CLI YAML config."""
    from inference.runner import InferenceConfig

    cfg = ctx.config.get("inference", {})
    return InferenceConfig(
        model_name=cfg.get("model_name", "./artifacts/models/Phi-4-mini-instruct"),
        adapter_path=cfg.get("adapter_path", "artifacts/adapters"),
        rag_index_dir=cfg.get("rag_index_dir", "artifacts/rag_index"),
        max_tokens=cfg.get("max_tokens", 350),
        temperature=cfg.get("temperature", 0.0),
        top_p=cfg.get("top_p", 1.0),
        top_k=cfg.get("rag_top_k", 3),
        score_threshold=cfg.get("rag_score_threshold", 0.3),
        seed=ctx.seed,
    )


def eval(
    config_path: str = "configs/default.yaml",
    output_file: Optional[str] = None,
    use_model: bool = True,
    use_rag: bool = False,
    baseline: bool = False,
    max_tokens: Optional[int] = None,
) -> None:
    """Evaluate on the held-out test set.

    Three modes:

    - ``--baseline`` (default if no adapter exists): use the rule-based
      ``Data/describe.py`` for every record. This is the perfect-oracle
      baseline and lets you sanity-check the evaluator.
    - ``--use-model`` (default when an adapter exists): load the trained
      LoRA adapter and generate a prediction for every test record.
    - ``--use-rag``: additionally retrieve top-k passages from the FAISS
      index and inject them into the prompt (only meaningful with
      ``--use-model``).

    Results are written to ``artifacts/eval_results/eval_results.json``
    (or ``--output-file``).
    """
    from Data.schema import FeatureRecord
    from eval.evaluate import evaluate

    ctx = CLIContext.from_config_path(config_path)
    eval_cfg = ctx.config.get("eval", {})

    # Load test set
    test_path = Path("artifacts/dataset/feature_records_test.jsonl")
    if not test_path.exists():
        print(f"[cli] Test set not found at {test_path}. Run 'build-dataset' first.")
        return

    test_records: List[FeatureRecord] = []
    with test_path.open() as f:
        for line in f:
            test_records.append(FeatureRecord.model_validate_json(line))

    print(f"[cli] Loaded {len(test_records)} test records")

    # Decide which predictor to use
    predictions: List[str]
    predictor_name: str

    adapter_exists = (
        ctx.config.get("inference", {}).get("adapter_path")
        and Path(
            ctx.config["inference"]["adapter_path"], "adapters.safetensors"
        ).exists()
    )

    if baseline or (not use_model) or (not adapter_exists and not baseline):
        if not adapter_exists and use_model and not baseline:
            print(
                "[cli] No adapter found at "
                f"{ctx.config.get('inference', {}).get('adapter_path')}. "
                "Falling back to rule-based describe()."
            )
        from Data.describe import describe

        predictions = [describe(fr) for fr in test_records]
        predictor_name = "rule-based"
    else:
        from inference.runner import InferenceRunner

        inf_cfg = _build_inference_config(ctx)
        if max_tokens is not None:
            inf_cfg.max_tokens = max_tokens
        print(
            f"[cli] Evaluating with trained model "
            f"(adapter={inf_cfg.adapter_path}, RAG={use_rag}, "
            f"max_tokens={inf_cfg.max_tokens})..."
        )
        with InferenceRunner(inf_cfg) as runner:
            predictions = runner.generate_batch(
                test_records, use_rag=use_rag, show_progress=True
            )
        predictor_name = (
            f"trained-model+adapter({'rag' if use_rag else 'no-rag'})"
        )

    print(f"[cli] Running metrics on {len(predictions)} predictions from {predictor_name}")
    results = evaluate(
        predictions=predictions,
        test_dataset=test_records,
        config=eval_cfg,
    )
    results["meta"] = {
        "predictor": predictor_name,
        "num_records": len(test_records),
        "config_path": config_path,
    }

    # Print summary
    print("\n[cli] Evaluation Results:")
    print(f"  Predictor: {predictor_name}")
    print(f"  Examples: {results['summary']['num_examples']}")
    print(f"  Regime Macro F1: {results['summary']['regime_macro_f1']:.4f}")
    print(f"  Lock-in F1: {results['summary']['lock_in_f1']:.4f}")
    print(f"  Avg Field Accuracy: {results['summary']['avg_field_accuracy']:.4f}")

    if results["per_field_accuracy"]:
        print("\n  Per-Field Accuracy:")
        for field, metrics in results["per_field_accuracy"].items():
            print(
                f"    {field}: {metrics['accuracy']:.2%} "
                f"({metrics['correct']}/{metrics['total']})"
            )

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

    # Presentation-ready markdown report next to the JSON.
    from eval.report import render_markdown_report

    report_path = out_path.with_suffix(".md")
    report_path.write_text(render_markdown_report(results))
    print(f"[cli] Markdown report saved to {report_path}")


def infer(
    feature_record: str,
    use_rag: bool = False,
    use_model: bool = True,
    baseline: bool = False,
    config_path: str = "configs/default.yaml",
    max_tokens: Optional[int] = None,
    json_output: bool = False,
    stream: bool = False,
) -> str:
    """Run inference on a single FeatureRecord.

    Parameters
    ----------
    feature_record : str
        Path to a JSON file containing a FeatureRecord, or a JSON string.
    use_rag : bool
        Whether to use RAG augmentation (default: False).
    use_model : bool
        Use the trained LoRA adapter (default: True). Ignored if ``--baseline``
        is set.
    baseline : bool
        Force the rule-based ``Data/describe.py`` instead of the trained model.
    config_path : str
        Path to YAML configuration file.
    max_tokens : int, optional
        Override ``inference.max_tokens`` from the config.
    json_output : bool
        Print a machine-readable JSON object (instead of pretty text).
    stream : bool
        Stream tokens to stdout as they are generated (token-by-token).

    Returns
    -------
    str
        The interpretation text. Also prints to stdout for human inspection.
    """
    from Data.schema import FeatureRecord

    ctx = CLIContext.from_config_path(config_path)

    # Load feature record
    if Path(feature_record).exists():
        with open(feature_record) as f:
            data = json.load(f)
    else:
        data = json.loads(feature_record)

    fr = FeatureRecord.model_validate(data)

    # Decide which predictor
    adapter_path = ctx.config.get("inference", {}).get("adapter_path")
    adapter_exists = (
        adapter_path and Path(adapter_path, "adapters.safetensors").exists()
    )

    if baseline or (not use_model) or (not adapter_exists):
        if not adapter_exists and use_model and not baseline:
            print(
                f"[cli] No adapter found at {adapter_path}; falling back to "
                "rule-based describe()."
            )
        from Data.describe import describe

        prediction = describe(fr)
        predictor_name = "rule-based"
        retrieved: list = []
    else:
        from inference.runner import InferenceRunner

        inf_cfg = _build_inference_config(ctx)
        if max_tokens is not None:
            inf_cfg.max_tokens = max_tokens
        with InferenceRunner(inf_cfg) as runner:
            if use_rag:
                retrieved = runner.retrieve(fr)
                print(f"[cli] RAG: retrieved {len(retrieved)} passages")
            else:
                retrieved = []

            if stream:
                print(f"\n[cli] Streaming prediction ({predictor_name_to_label(adapter_path, use_rag)}):", flush=True)
                print("-" * 60, flush=True)
                pieces: list = []
                for response in runner.stream(fr, retrieved=retrieved, max_tokens=max_tokens):
                    piece = getattr(response, "text", "") or ""
                    pieces.append(piece)
                    if not json_output:
                        print(piece, end="", flush=True)
                if not json_output:
                    print()  # final newline
                prediction = "".join(pieces)
                print("-" * 60, flush=True)
            else:
                prediction = runner.generate(fr, retrieved=retrieved, max_tokens=max_tokens)
        predictor_name = predictor_name_to_label(adapter_path, use_rag)

    # Validate
    from eval.validate_output import validate_output, parse_structured_fields

    result = validate_output(prediction)
    parsed = parse_structured_fields(prediction)

    if json_output:
        out = {
            "experiment_id": fr.experiment_id,
            "interpretation": prediction,
            "predictor": predictor_name,
            "regime": parsed.get("regime"),
            "used_rag": use_rag and bool(retrieved),
            "retrieved_passages": [
                {
                    "text": r.passage_text,
                    "source": r.source,
                    "score": r.score,
                    "page": r.page,
                }
                for r in retrieved
            ],
            "validator": {
                "valid": result.valid,
                "errors": list(result.errors),
                "warnings": list(result.warnings),
            },
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"\n[cli] Prediction ({predictor_name}):")
        print(prediction)
        print(f"\n[cli] Validator: valid={result.valid}")
        if result.errors:
            print(f"  errors:   {result.errors}")
        if result.warnings:
            print(f"  warnings: {result.warnings}")

    return prediction


def predictor_name_to_label(adapter_path: str, use_rag: bool) -> str:
    return f"trained-model+adapter({'rag' if use_rag else 'no-rag'})"


def serve(
    config_path: str = "configs/default.yaml",
    port: Optional[int] = None,
    host: Optional[str] = None,
    eager_load: bool = False,
    workers: int = 1,
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
    eager_load : bool
        Load model + adapter at startup. Slow startup (~30s) but first
        request is fast. Default: lazy-load on first /infer.
    workers : int
        Number of uvicorn worker processes. Each worker loads the model
        independently into its own memory. Keep at 1 on a single Mac M4
        Pro; raise only if you have many GBs of RAM.
    """
    from serve.fastapi_app import create_app

    ctx = CLIContext.from_config_path(config_path)
    serve_cfg = ctx.config.get("serve", {})

    import uvicorn
    uvicorn.run(
        create_app(config=ctx.config, eager_load=eager_load),
        host=host or serve_cfg.get("host", "0.0.0.0"),
        port=port or serve_cfg.get("port", 8000),
        workers=workers,
    )


def ui(
    config_path: str = "configs/default.yaml",
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the Gradio web UI.

    The UI lets a user provide real FSI data (upload a RawExperiment JSON or
    fill the manual form), get the structured interpretation, and ask free-text
    questions about the record.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    host : str
        Bind address (default 127.0.0.1; use 0.0.0.0 to expose on the LAN).
    port : int
        Server port (default 7860).
    share : bool
        Create a temporary public Gradio share link.
    """
    from serve.gradio_app import create_ui

    ctx = CLIContext.from_config_path(config_path)
    demo = create_ui(config=ctx.config)
    demo.launch(server_name=host, server_port=port, share=share)

def mcp(config_path: str = "configs/default.yaml") -> None:
    """Start the MCP (Model Context Protocol) server over stdio.

    Exposes interpret_experiment, ask_question, search_literature,
    validate_interpretation, and generate_synthetic_experiment as MCP tools
    for clients like Claude Desktop and Claude Code. Register via the
    project's .mcp.json. The model is lazy-loaded on first inference call.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file (also settable via FSI_SLM_CONFIG).
    """
    os.environ.setdefault("FSI_SLM_CONFIG", _resolve_path(config_path))
    from serve.mcp_server import main as mcp_main

    mcp_main()


def demo(seed: int = 0, sweep: bool = False) -> None:
    """End-to-end terminal demo: no model, no downloads, runs in seconds.

    For each of the five flow regimes: generate a synthetic experiment,
    extract features, produce the rule-based interpretation, and validate it.
    Prints a summary table (true vs predicted regime, St, lock-in, validator
    verdict). With ``--sweep``, also prints a reduced-velocity regime sweep
    for the VIV record showing the predicted lock-in window.

    Parameters
    ----------
    seed : int
        Seed for the synthetic generator (deterministic per seed).
    sweep : bool
        Also demo the what-if parameter sweep (regime map vs U*).
    """
    import numpy as np

    from Data.describe import classify_regime_confidence, describe
    from Data.schema import FlowRegime
    from Data.synthetic import generate_experiment
    from eval.validate_output import validate_output
    from features.extract import extract_features

    print(f"[demo] seed={seed} — generate → extract → interpret → validate\n")
    header = f"{'true regime':<15} {'predicted':<15} {'conf':<6} {'St':<7} {'lock-in':<12} valid"
    print(header)
    print("-" * len(header))

    correct = 0
    viv_record = None
    for i, regime in enumerate(FlowRegime):
        raw = generate_experiment(regime, rng=np.random.default_rng(seed + i))
        fr = extract_features(raw)
        pred, conf, _ = classify_regime_confidence(fr)
        ok = validate_output(describe(fr)).valid
        if pred == regime:
            correct += 1
        if regime == FlowRegime.VIV_LOCKIN:
            viv_record = fr
        print(
            f"{regime.value:<15} {pred.value:<15} {conf:<6.2f} "
            f"{fr.piv.strouhal_number:<7.3f} {fr.lock_in_status().value:<12} {ok}"
        )

    print(f"\n[demo] {correct}/{len(FlowRegime)} regimes recovered by the rule-based classifier")

    if sweep and viv_record is not None:
        from features.sweep import find_transitions, sweep_parameter

        print("\n[demo] What-if sweep: reduced velocity U* ∈ [2, 12] on the VIV record")
        points = sweep_parameter(viv_record, "reduced_velocity", n=11)
        for pt in points:
            print(
                f"  U*={pt.value:5.2f}  ->  {pt.regime.value:<15} "
                f"(conf {pt.confidence:.2f}, lock-in: {pt.lock_in})"
            )
        for value, frm, to in find_transitions(points):
            print(f"[demo] transition at U*≈{value:.2f}: {frm.value} -> {to.value}")


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
        "ui": ui,
        "mcp": mcp,
        "demo": demo,
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
    "ui",
    "mcp",
    "demo",
    "main",
]