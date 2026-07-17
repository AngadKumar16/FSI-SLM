"""MCP (Model Context Protocol) server for the FSI interpreter.

Exposes the trained model + RAG pipeline as MCP tools so any MCP client
(Claude Desktop, Claude Code, custom agents) can call the FSI interpreter
directly. Runs over stdio.

Tools
-----
* ``interpret_experiment``          -- structured interpretation of a FeatureRecord
* ``ask_question``                  -- free-text Q&A grounded in a FeatureRecord
* ``search_literature``             -- FAISS retrieval over the FSI paper corpus
* ``validate_interpretation``       -- parse + physics-check generated text (no model)
* ``generate_synthetic_experiment`` -- demo FeatureRecord for a chosen flow regime

Resources
---------
* ``fsi://schema/feature_record``   -- FeatureRecord JSON schema

Launch::

    python cli.py mcp                 # or: python -m serve.mcp_server

Register with an MCP client via the project's ``.mcp.json``. The heavy model
is lazy-loaded on the first inference call; retrieval-only and validation
tools never load it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# transformers tries to import TensorFlow when present; a broken TF install
# segfaults the whole server. Nothing here needs TF -- torch/MLX only.
os.environ.setdefault("USE_TF", "0")

# Ensure project root is importable when launched as `python -m serve.mcp_server`
# from an arbitrary working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "fsi-slm",
    instructions=(
        "Fluid-structure interaction (FSI) experiment interpreter backed by a "
        "LoRA-tuned Phi-4-mini model and a FAISS literature index. Provide a "
        "FeatureRecord (see fsi://schema/feature_record) to interpret real "
        "experiments, or generate a synthetic one to try the pipeline."
    ),
)

# --------------------------------------------------------------------------- #
# Lazy shared state
# --------------------------------------------------------------------------- #
_state: Dict[str, Any] = {"config": None, "runner": None, "rag": None}


def _load_config() -> dict:
    if _state["config"] is None:
        import yaml

        path = os.environ.get("FSI_SLM_CONFIG", str(_ROOT / "configs" / "default.yaml"))
        with open(path) as f:
            _state["config"] = yaml.safe_load(f) or {}
    return _state["config"]


def _get_runner():
    """Build the shared InferenceRunner (heavy model load deferred inside it)."""
    if _state["runner"] is None:
        from inference.runner import InferenceConfig, InferenceRunner

        config = _load_config()
        cfg = config.get("inference", {})
        rag_cfg = config.get("rag", {})

        def _abs(p: str) -> str:
            return p if os.path.isabs(p) else str(_ROOT / p)

        _state["runner"] = InferenceRunner(InferenceConfig(
            model_name=_abs(cfg.get("model_name", "./artifacts/models/Phi-4-mini-instruct")),
            adapter_path=_abs(cfg.get("adapter_path", "artifacts/adapters")),
            rag_index_dir=_abs(cfg.get("rag_index_dir", rag_cfg.get("index_dir", "artifacts/rag_index"))),
            max_tokens=cfg.get("max_tokens", 350),
            top_k=cfg.get("rag_top_k", rag_cfg.get("top_k", 3)),
            score_threshold=cfg.get("rag_score_threshold", rag_cfg.get("score_threshold", 0.3)),
        ))
    return _state["runner"]


def _get_rag():
    """Load the FAISS index + embedder without touching the LLM."""
    if _state["rag"] is None:
        config = _load_config()
        rag_cfg = config.get("rag", {})
        index_dir = rag_cfg.get("index_dir", "artifacts/rag_index")
        if not os.path.isabs(index_dir):
            index_dir = str(_ROOT / index_dir)
        if not (Path(index_dir) / "index.faiss").exists():
            raise FileNotFoundError(
                f"No RAG index at {index_dir}. Build it with: python cli.py index-corpus"
            )
        from rag.index import load_index

        _state["rag"] = load_index(index_dir)
    return _state["rag"]


def _build_record(feature_record: Dict[str, Any]):
    from Data.schema import FeatureRecord

    return FeatureRecord.model_validate(feature_record)


def _citations(retrieved: List) -> List[Dict[str, Any]]:
    return [
        {
            "source": r.source,
            "page": r.page,
            "score": round(float(r.score), 4),
            "passage": r.passage_text.strip(),
        }
        for r in retrieved
    ]


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def interpret_experiment(
    feature_record: Dict[str, Any],
    use_rag: bool = False,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the trained FSI interpreter on one experiment.

    Call this when you have a FeatureRecord (extracted features of an FSI
    experiment: Strouhal number, shedding/natural frequencies, force
    coefficients, ...) and want the structured interpretation: flow regime,
    lock-in assessment, and physical explanation. Set use_rag=true to ground
    the output in retrieved literature passages (slower; needs the FAISS
    index). Loads the local model on first call (~30s).
    """
    from eval.validate_output import validate_output

    fr = _build_record(feature_record)
    runner = _get_runner()
    retrieved = runner.retrieve(fr) if use_rag else []
    text = runner.generate(fr, use_rag=bool(retrieved), max_tokens=max_tokens)
    result = validate_output(text)
    return {
        "interpretation": text,
        "regime": result.parsed_fields.get("regime"),
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "used_rag": bool(retrieved),
        "citations": _citations(retrieved),
    }


@mcp.tool()
def ask_question(
    feature_record: Dict[str, Any],
    question: str,
    use_rag: bool = True,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Answer a free-text question about one FSI experiment.

    Unlike interpret_experiment (fixed structured output), this answers
    open-ended questions ("why is this locked in?", "what happens if U
    increases 20%?") grounded in the given FeatureRecord, optionally with
    retrieved literature. Loads the base model on first call.
    """
    fr = _build_record(feature_record)
    runner = _get_runner()
    answer, retrieved = runner.answer(
        fr, question, use_rag=use_rag, max_tokens=max_tokens
    )
    return {"answer": answer, "citations": _citations(retrieved)}


@mcp.tool()
def search_literature(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search the indexed FSI paper corpus (FAISS + sentence-transformers).

    Returns the top passages with source, page, and similarity score. Cheap:
    loads only the embedding model and index, never the LLM. Use for
    literature lookup on vortex shedding, VIV lock-in, galloping, stall
    flutter, etc.
    """
    from rag.retrieve import retrieve

    index, metadata, model = _get_rag()
    results = retrieve(
        query=query,
        index=index,
        metadata=metadata,
        model=model,
        top_k=top_k,
        score_threshold=0.0,
    )
    return _citations(results)


@mcp.tool()
def validate_interpretation(text: str) -> Dict[str, Any]:
    """Validate a generated FSI interpretation without any model.

    Parses the structured claims (regime, frequencies, Strouhal number,
    lock-in) out of the text and checks numeric consistency (reported St vs
    f_shed*c/U, lock-in logic vs claimed frequencies) and schema conformance.
    Use to check outputs from interpret_experiment or any other source.
    """
    from dataclasses import asdict

    from eval.validate_output import validate_output

    return asdict(validate_output(text))


@mcp.tool()
def generate_synthetic_experiment(regime: str = "viv_lockin", seed: int = 0) -> Dict[str, Any]:
    """Generate a physically-plausible synthetic FeatureRecord for demoing.

    regime must be one of: attached, separated, viv_lockin, galloping,
    stall_flutter. Returns a FeatureRecord dict you can pass straight to
    interpret_experiment or ask_question. Deterministic for a given
    (regime, seed).
    """
    import numpy as np

    from Data.schema import FlowRegime
    from Data.synthetic import generate_experiment
    from features.extract import extract_features

    try:
        flow_regime = FlowRegime(regime)
    except ValueError:
        valid = ", ".join(r.value for r in FlowRegime)
        raise ValueError(f"Unknown regime '{regime}'. Valid regimes: {valid}")

    raw = generate_experiment(flow_regime, rng=np.random.default_rng(seed))
    fr = extract_features(raw)
    return json.loads(fr.model_dump_json())


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #
@mcp.resource("fsi://schema/feature_record")
def feature_record_schema() -> str:
    """JSON schema for the FeatureRecord accepted by the inference tools."""
    from Data.schema import FeatureRecord

    return json.dumps(FeatureRecord.model_json_schema(), indent=2)


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
