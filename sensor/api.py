"""One shared entry point for the surrogate, used by the CLI, REST and MCP.

Every surface (``cli.py``, ``serve/fastapi_app.py``, ``serve/mcp_server.py``)
calls these functions rather than reaching into :mod:`sensor.predict` or
:mod:`sensor.bridge` directly, so the three cannot drift apart and the
predictor is loaded once per process instead of once per request.

Everything returned here is plain JSON-safe Python: arrays become nested lists
only when explicitly requested, because a 128 x 128 field is 16 k numbers and
an agent asking "what is the lift" should not be handed a megabyte.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .bridge import TwinConfig, digital_twin
from .geometry import load_dat
from .predict import DEFAULT_CHECKPOINT, SurrogatePredictor
from .sdf import Grid
from .simulate import CasePrep, simulate_case

_PREDICTORS: Dict[str, SurrogatePredictor] = {}


def get_predictor(
    checkpoint: Union[str, Path] = DEFAULT_CHECKPOINT,
    device: str = "auto",
) -> Optional[SurrogatePredictor]:
    """Process-wide cached predictor. ``None`` when no checkpoint exists yet."""
    key = f"{checkpoint}|{device}"
    if key in _PREDICTORS:
        return _PREDICTORS[key]
    if not Path(checkpoint).exists():
        return None
    _PREDICTORS[key] = SurrogatePredictor(checkpoint, device)
    return _PREDICTORS[key]


def _downsample_fields(fields: Dict[str, np.ndarray], factor: int) -> Dict[str, list]:
    out = {}
    for k, v in fields.items():
        a = np.asarray(v)
        if factor > 1:
            h, w = a.shape
            a = a[: h - h % factor, : w - w % factor]
            a = a.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
        out[k] = np.round(a.astype(float), 4).tolist()
    return out


def predict_case(
    airfoil: str,
    alpha_deg: float,
    reynolds: float = 5.0e5,
    checkpoint: Union[str, Path] = DEFAULT_CHECKPOINT,
    device: str = "auto",
    return_fields: bool = False,
    field_downsample: int = 4,
    fallback_to_solver: bool = True,
) -> dict:
    """Coefficients (and optionally fields) for one case.

    Falls back to the physics solver when no trained checkpoint is present, so
    the endpoint is useful before the network is trained and the caller can tell
    which happened from ``source``.
    """
    grid = Grid()
    predictor = get_predictor(checkpoint, device)
    if predictor is None:
        if not fallback_to_solver:
            raise FileNotFoundError(
                f"no surrogate checkpoint at {checkpoint}; train one with "
                f"`python cli.py sensor-train`"
            )
        af = load_dat(airfoil)
        case = simulate_case(CasePrep.build(af), alpha_deg, reynolds, grid)
        result = {
            "source": "solver",
            "airfoil": af.name,
            "alpha_deg": float(alpha_deg),
            "reynolds": float(reynolds),
            "coeffs": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in case["coeffs"].items()},
            "resolution": [grid.ny, grid.nx],
        }
        if return_fields:
            names = ("u", "v", "p", "load", "disp")
            result["fields"] = _downsample_fields(
                {n: case["target"][i] for i, n in enumerate(names)}, field_downsample
            )
        return result

    pred = predictor.predict(airfoil, alpha_deg, reynolds, grid)
    result = {
        "source": "unet",
        "checkpoint": str(checkpoint),
        "airfoil": pred["airfoil"],
        "alpha_deg": float(alpha_deg),
        "reynolds": float(reynolds),
        "coeffs": pred["coeffs"],
        "resolution": pred["resolution"],
    }
    if return_fields:
        result["fields"] = _downsample_fields(pred["fields"], field_downsample)
    return result


def sweep_alpha(
    airfoil: str,
    alphas: Sequence[float] = (-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16),
    reynolds: float = 5.0e5,
    checkpoint: Union[str, Path] = DEFAULT_CHECKPOINT,
    device: str = "auto",
    use_solver: bool = False,
) -> dict:
    """Lift and drag polar. The cheapest useful thing to ask a surrogate."""
    grid = Grid()
    predictor = None if use_solver else get_predictor(checkpoint, device)
    af = load_dat(airfoil)
    prep = CasePrep.build(af) if predictor is None else None

    rows = []
    for a in alphas:
        if predictor is None:
            c = simulate_case(prep, float(a), reynolds, grid, with_modal=False)["coeffs"]
            rows.append({"alpha_deg": float(a), "cl": c["cl"], "cd": c["cd"],
                         "cm": c["cm"], "f_sep": c["f_sep"], "source": "solver"})
        else:
            c = predictor.predict(af, float(a), reynolds, grid)["coeffs"]
            rows.append({"alpha_deg": float(a), **{k: float(v) for k, v in c.items()},
                         "source": "unet"})

    cls = [r["cl"] for r in rows]
    k = int(np.argmax(cls))
    return {
        "airfoil": af.name,
        "reynolds": float(reynolds),
        "source": rows[0]["source"] if rows else "none",
        "points": rows,
        "cl_max": float(cls[k]) if cls else None,
        "alpha_cl_max_deg": float(rows[k]["alpha_deg"]) if rows else None,
    }


def twin_record(
    airfoil: str,
    alpha_deg: float,
    chord_m: float = 0.1,
    span_m: float = 0.3,
    u_inf_m_s: float = 5.0,
    damping_ratio: float = 0.01,
    use_surrogate: bool = True,
    checkpoint: Union[str, Path] = DEFAULT_CHECKPOINT,
    device: str = "auto",
    inject_faults: Sequence[str] = (),
    seed: int = 0,
) -> dict:
    """Full twin pass: geometry -> fields -> FeatureRecord + data-quality report.

    This is the object the language model interprets, so it is also the natural
    payload for an agent that wants to go straight from "NACA 0009 at 12
    degrees" to an interpretation.
    """
    cfg = TwinConfig(
        chord_m=chord_m, span_m=span_m, u_inf_m_s=u_inf_m_s,
        damping_ratio=damping_ratio,
    )
    predictor = get_predictor(checkpoint, device) if use_surrogate else None
    out = digital_twin(
        airfoil, alpha_deg, cfg, predictor=predictor,
        inject_faults=inject_faults, seed=seed,
    )
    return {
        "feature_record": json.loads(out["feature_record"].model_dump_json()),
        "noise_report": out["noise_report"].to_dict(),
        "data_quality_block": out["noise_report"].to_prompt_block(),
        "den_hartog_margin": out["den_hartog_margin"],
        "regime_rule_based": out["feature_record"].ground_truth_regime.value
        if out["feature_record"].ground_truth_regime else None,
        "provenance": out["provenance"],
    }


def dataset_summary(out_dir: str = "artifacts/cfd") -> dict:
    """What has actually been generated, for a status endpoint."""
    p = Path(out_dir)
    summary_path = p / "summary.json"
    if not summary_path.exists():
        return {"generated": False, "dir": str(p)}
    summary = json.loads(summary_path.read_text())
    summary["generated"] = True
    npz_dir = p / "npz"
    summary["npz_files"] = len(list(npz_dir.glob("*.npz"))) if npz_dir.exists() else 0
    return summary


def checkpoint_info(checkpoint: Union[str, Path] = DEFAULT_CHECKPOINT) -> dict:
    """Metadata about the trained surrogate without loading it for inference."""
    path = Path(checkpoint)
    if not path.exists():
        return {"trained": False, "checkpoint": str(path)}
    import torch

    ck = torch.load(str(path), map_location="cpu")
    hist = ck.get("history", [])
    return {
        "trained": True,
        "checkpoint": str(path),
        "epoch": ck.get("epoch"),
        "config": ck.get("config"),
        "downsample": ck.get("downsample", 1),
        "best_val_loss": ck.get("best"),
        "last_val": hist[-1]["val"] if hist else None,
        "scalar_names": ck.get("scalar_names"),
    }
