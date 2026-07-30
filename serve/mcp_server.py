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
* ``predict_flow``                  -- U-Net surrogate: geometry + alpha -> Cl/Cd/Cm, fields
* ``predict_polar``                 -- Cl/Cd polar and stall angle for a section
* ``digital_twin_record``           -- geometry -> full FeatureRecord + data-quality report
* ``check_data_quality``            -- instrumentation-fault screening on raw channels
* ``surrogate_status``              -- dataset/checkpoint state

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


def _abs(p: str) -> str:
    """Resolve a path against the project root.

    MCP clients launch the server from arbitrary working directories, so every
    path an agent passes in has to be anchored here rather than to ``os.getcwd``.
    """
    return p if os.path.isabs(p) else str(_ROOT / p)


def _get_runner():
    """Build the shared InferenceRunner (heavy model load deferred inside it)."""
    if _state["runner"] is None:
        from inference.runner import InferenceConfig, InferenceRunner

        config = _load_config()
        cfg = config.get("inference", {})
        rag_cfg = config.get("rag", {})

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
    from features.extract import check_physical_consistency

    fr = _build_record(feature_record)
    runner = _get_runner()
    retrieved = runner.retrieve(fr) if use_rag else []
    text = runner.generate(fr, retrieved=retrieved, max_tokens=max_tokens)
    result = validate_output(text)
    return {
        "interpretation": text,
        "regime": result.parsed_fields.get("regime"),
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "record_consistency_warnings": check_physical_consistency(fr),
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
def check_experiment_consistency(feature_record: Dict[str, Any]) -> Dict[str, Any]:
    """Physics-check a FeatureRecord without loading any model.

    Runs two independent validators: (1) range_warnings — every field vs its
    documented physical range (units-bug detector, e.g. Hz vs rad/s); and
    (2) check_physical_consistency — cross-field physics (reported St vs
    f_shed*c/U, lock-in label vs actual frequency ratio, damping ratio vs
    spectral bandwidth, coherence phase). Instant; use before trusting or
    interpreting any record.
    """
    from features.extract import check_physical_consistency

    fr = _build_record(feature_record)
    range_warns = fr.range_warnings()
    physics_warns = check_physical_consistency(fr)
    return {
        "consistent": not (range_warns or physics_warns),
        "range_warnings": range_warns,
        "physics_warnings": physics_warns,
    }


@mcp.tool()
def sweep_regime_map(
    feature_record: Dict[str, Any],
    param: str = "reduced_velocity",
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    n: int = 15,
) -> Dict[str, Any]:
    """What-if analysis: sweep one parameter and map the predicted regimes.

    Varies ``param`` (reduced_velocity, angle_of_attack_deg, or
    natural_frequency_hz) across [lo, hi] (sensible defaults when omitted),
    rescales physically coupled quantities, and classifies the regime at each
    point with the transparent rule-based classifier. Returns the per-point
    regimes plus detected transition boundaries (e.g. the lock-in window).
    No model load — instant.
    """
    from features.sweep import find_transitions, sweep_parameter

    fr = _build_record(feature_record)
    points = sweep_parameter(fr, param=param, lo=lo, hi=hi, n=n)
    return {
        "param": param,
        "points": [
            {
                "value": round(pt.value, 4),
                "regime": pt.regime.value,
                "confidence": round(pt.confidence, 3),
                "shedding_frequency_hz": round(pt.shedding_frequency_hz, 3),
                "lock_in": pt.lock_in,
            }
            for pt in points
        ],
        "transitions": [
            {"value": round(v, 4), "from": frm.value, "to": to.value}
            for v, frm, to in find_transitions(points)
        ],
    }


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


@mcp.tool()
def predict_flow(
    airfoil: str,
    alpha_deg: float,
    reynolds: float = 5.0e5,
    return_fields: bool = False,
) -> Dict[str, Any]:
    """Predict lift, drag, moment and deflection for an airfoil, instantly.

    Runs the trained U-Net surrogate on the given section and incidence -- no
    mesh, no solver, milliseconds. Returns Cl, Cd, Cm, peak deflection and
    trailing-edge twist; set return_fields for the velocity, pressure, load and
    displacement maps (downsampled 4x). Falls back to the panel solver if no
    checkpoint is trained yet, and says which it used in "source".

    airfoil is a path to a .dat file, e.g. Data/Airfoils/4/n0009sm.dat.txt.
    """
    from sensor.api import predict_case

    return predict_case(
        _abs(airfoil), alpha_deg, reynolds, return_fields=return_fields
    )


@mcp.tool()
def predict_polar(
    airfoil: str,
    alpha_min: float = -4.0,
    alpha_max: float = 16.0,
    n_points: int = 11,
    reynolds: float = 5.0e5,
) -> Dict[str, Any]:
    """Lift/drag polar for a section: Cl(alpha), Cd(alpha), Cl_max and its angle.

    Use this to compare candidate sections or to find the stall angle before
    committing to a test condition.
    """
    import numpy as np

    from sensor.api import sweep_alpha

    alphas = np.linspace(alpha_min, alpha_max, max(2, int(n_points))).tolist()
    return sweep_alpha(_abs(airfoil), alphas, reynolds)


@mcp.tool()
def digital_twin_record(
    airfoil: str,
    alpha_deg: float,
    chord_m: float = 0.1,
    span_m: float = 0.3,
    u_inf_m_s: float = 5.0,
    inject_faults: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a full FSI experiment record from geometry alone.

    Runs the surrogate, derives wake width, separation point, shedding
    frequency and structural response, and returns a FeatureRecord ready for
    interpret_experiment -- plus a data-quality report screening every channel
    for instrumentation faults (PIV dropout, clipping, mains pickup, drift,
    spikes, aliasing).

    inject_faults deliberately corrupts the synthesised signals to test the
    detector: any of clipping, dropout, mains, drift, spike, flatline.

    The time series are a 1-DOF instrument model driven by steady surrogate
    output, not measurements; the returned "provenance" block says so.
    """
    from sensor.api import twin_record

    return twin_record(
        _abs(airfoil), alpha_deg, chord_m=chord_m, span_m=span_m,
        u_inf_m_s=u_inf_m_s, inject_faults=tuple(inject_faults or ()),
    )


@mcp.tool()
def check_data_quality(feature_record: Dict[str, Any] = None,
                       signals: Dict[str, List[float]] = None,
                       sampling_rate_hz: float = 400.0) -> Dict[str, Any]:
    """Screen raw experiment channels for instrumentation faults.

    Pass ``signals`` as a mapping of channel name to samples (displacement,
    wake probe, cl, cd, cm). Reports PIV dropout and spurious vectors, ADC
    clipping and quantisation, spikes, drift, flatlines, poor spectral
    signal-to-noise, aliasing and 50/60 Hz mains pickup, each with a severity
    and a plain-language explanation of what it invalidates.

    This is instrumentation screening, not physics checking -- use
    check_experiment_consistency for the latter.
    """
    from features.noise import analyse_signals

    if not signals:
        raise ValueError("provide `signals`: {channel_name: [samples, ...]}")
    report = analyse_signals(signals, sampling_rate_hz)
    out = report.to_dict()
    out["prompt_block"] = report.to_prompt_block()
    return out


@mcp.tool()
def surrogate_status() -> Dict[str, Any]:
    """Is the 1,926-case dataset generated and the U-Net trained? Which epoch?"""
    from sensor.api import checkpoint_info, dataset_summary

    return {"dataset": dataset_summary(), "surrogate": checkpoint_info()}


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
