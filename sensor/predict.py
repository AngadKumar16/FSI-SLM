"""Run the trained surrogate on a new (shape, incidence, Reynolds) case.

    from sensor.predict import SurrogatePredictor
    p = SurrogatePredictor("artifacts/sensor/run/best.pt")
    out = p.predict("Data/Airfoils/4/n0009sm.dat.txt", alpha_deg=8.0)
    out["coeffs"]["cl"], out["fields"]["u"]

The predictor owns the denormalisation and the masking, so callers get physical
units and zeros outside each channel's own region -- the same convention the
solver writes. ``compare_to_solver`` re-runs the panel solver on the same case
and reports per-channel errors, which is how the surrogate is validated on the
held-out NACA 0009.

Two things worth knowing.

``fidelity=``
    With the ``v2`` input contract the fidelity code is an input channel, so a
    network trained on a mix can be *asked* which solver to imitate. Requesting
    ``"rans"`` from a checkpoint that never saw a RANS case is extrapolation and
    says so in the returned metadata rather than failing quietly.

``tta=``
    Test-time mirror averaging. The prediction and the prediction of the
    mirrored case are averaged back together; the mirror is an exact symmetry of
    the equations (:mod:`sensor.symmetry`), so the averaging cancels the part of
    the error that is antisymmetric noise and leaves the physics untouched. One
    extra forward pass, no retraining.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from .data import SensorCache
from .geometry import Airfoil, load_dat
from .model import UNet, UNetConfig, pick_device
from .sdf import FIDELITY_CODES, Grid, InputSpec, build_input
from .simulate import CasePrep, SCALAR_NAMES, TARGET_CHANNELS, simulate_case
from .symmetry import (
    normalised_field_parity,
    normalised_input_parity,
    normalised_scalar_parity,
)

DEFAULT_CHECKPOINT = "artifacts/sensor/run/best.pt"


class LegacyCheckpointError(RuntimeError):
    """A checkpoint from before the architecture rewrite. Retraining is required.

    Raised as its own type so callers -- the test suite, the figure scripts, the
    twin bridge -- can tell "this file predates the current model" apart from
    "this file is corrupt", and say so.
    """


class SurrogatePredictor:
    """Loaded U-Net plus the normalisation and input contract it was trained with."""

    def __init__(
        self,
        checkpoint: Union[str, Path] = DEFAULT_CHECKPOINT,
        device: str = "auto",
        tta: bool = True,
    ):
        import torch

        from .model import load_checkpoint

        self.device = pick_device(device)
        ck = load_checkpoint(checkpoint, map_location=self.device)
        self.spec = InputSpec.from_dict(ck.get("input_spec"))
        self.cfg = UNetConfig.from_dict(ck["config"])
        self.model = UNet(self.cfg).to(self.device)
        # Prefer the exponential moving average, which is what was validated and
        # what "best" was selected on. Older checkpoints have only "model".
        weights = ck.get("weights") or ck["model"]
        try:
            self.model.load_state_dict(weights)
        except RuntimeError as exc:
            raise LegacyCheckpointError(
                f"{checkpoint} was written by the previous architecture "
                f"(transposed-convolution decoder, single-layer head, no "
                f"attention) and its tensors do not map onto the current one. "
                f"Retrain: `python cli.py sensor-cache && python cli.py "
                f"sensor-train`. Original error: {exc}"
            ) from None
        self.model.eval()
        self.stats = ck["stats"]
        self.downsample = int(ck.get("downsample", 1))
        self.scalar_names = ck.get("scalar_names", list(SCALAR_NAMES))
        self.epoch = int(ck.get("epoch", -1))
        self.checkpoint = str(checkpoint)
        self.trained_fidelities = ck.get("fidelities")
        self.tta = tta
        self._torch = torch

        # Affine mirror transforms in *normalised* space; see sensor.symmetry.
        self._x_par, self._x_shift = normalised_input_parity(
            self.spec, self.stats["x_mean"], self.stats["x_std"]
        )
        self._y_par, self._y_shift = normalised_field_parity(
            self.stats["y_mean"], self.stats["y_std"]
        )
        self._s_par, self._s_shift = normalised_scalar_parity(
            self.stats["s_mean"], self.stats["s_std"]
        )

    # ---------------------------------------------------------------- helpers

    def _normalise_input(self, x: np.ndarray):
        torch = self._torch
        mean = np.asarray(self.stats["x_mean"], np.float32)[:, None, None]
        std = np.asarray(self.stats["x_std"], np.float32)[:, None, None]
        t = torch.from_numpy(((x - mean) / std).astype(np.float32))[None]
        if self.downsample > 1:
            t = torch.nn.functional.avg_pool2d(t, self.downsample)
        return t.to(self.device)

    def _denormalise(self, y: np.ndarray, s: np.ndarray):
        y_mean = np.asarray(self.stats["y_mean"], np.float32)[:, None, None]
        y_std = np.asarray(self.stats["y_std"], np.float32)[:, None, None]
        s_mean = np.asarray(self.stats["s_mean"], np.float32)
        s_std = np.asarray(self.stats["s_std"], np.float32)
        return y * y_std + y_mean, s * s_std + s_mean

    def _mirror(self, t, par, shift):
        """Affine mirror on a normalised ``(B, C, H, W)`` or ``(B, C)`` tensor.

        Involutive: the shift is nonzero only on channels whose parity is -1,
        and for those the second application cancels it exactly. That is what
        makes it safe to use the same call to fold the prediction back.
        """
        torch = self._torch
        p = torch.as_tensor(par, dtype=t.dtype, device=t.device)
        sh = torch.as_tensor(shift, dtype=t.dtype, device=t.device)
        if t.dim() == 4:
            return torch.flip(t, dims=(-2,)) * p.view(1, -1, 1, 1) + sh.view(1, -1, 1, 1)
        return t * p.view(1, -1) + sh.view(1, -1)

    def _forward(self, t):
        """Forward pass, optionally averaged with the mirrored case."""
        torch = self._torch
        with torch.no_grad():
            y, s = self.model(t)
            if not self.tta:
                return y, s
            ym, sm = self.model(self._mirror(t, self._x_par, self._x_shift))
            y = 0.5 * (y + self._mirror(ym, self._y_par, self._y_shift))
            s = 0.5 * (s + self._mirror(sm, self._s_par, self._s_shift))
        return y, s

    # ------------------------------------------------------------------- main

    def predict(
        self,
        airfoil: Union[str, Path, Airfoil],
        alpha_deg: float,
        reynolds: float = 5.0e5,
        grid: Optional[Grid] = None,
        fidelity: str = "panel",
    ) -> dict:
        """Fields and coefficients for one case, in physical (nondim) units."""
        grid = grid or Grid()
        af = airfoil if isinstance(airfoil, Airfoil) else load_dat(airfoil)
        prep = CasePrep.build(af)

        x = build_input(
            prep.airfoil, alpha_deg, reynolds, grid, self.spec, fidelity=fidelity
        ).astype(np.float32)
        y, s = self._forward(self._normalise_input(x))
        y = y[0].cpu().numpy()
        s = s[0].cpu().numpy()
        y, s = self._denormalise(y, s)

        # Masks come from the SDF the network was given, downsampled the same
        # way, so predicted fields are zeroed exactly where the targets are.
        sdf = x[self.spec.index("sdf")]
        if self.downsample > 1:
            sdf = _avg_pool_np(sdf, self.downsample)
        fluid = (sdf > 0).astype(np.float32)
        solid = 1.0 - fluid
        y[:3] *= fluid
        y[3:] *= solid

        fields = {name: y[i] for i, name in enumerate(TARGET_CHANNELS)}
        coeffs = {name: float(s[i]) for i, name in enumerate(self.scalar_names)}
        out = {
            "fields": fields,
            "coeffs": coeffs,
            "fluid_mask": fluid,
            "solid_mask": solid,
            "sdf": sdf,
            "alpha_deg": float(alpha_deg),
            "reynolds": float(reynolds),
            "airfoil": af.name,
            "resolution": list(sdf.shape),
            "source": "unet",
            "input_spec": self.spec.name,
            "fidelity": fidelity,
            "tta": bool(self.tta),
            "checkpoint": self.checkpoint,
        }
        if "fidelity" not in self.spec.names and fidelity != "panel":
            out["warning"] = (
                f"checkpoint was trained on the {self.spec.name} input contract, "
                f"which has no fidelity channel; the request for "
                f"'{fidelity}' had no effect"
            )
        elif self.trained_fidelities and fidelity not in self.trained_fidelities:
            out["warning"] = (
                f"no {fidelity} rows were in this checkpoint's training set "
                f"({sorted(self.trained_fidelities)}); the fidelity channel is "
                f"being extrapolated"
            )
        return out

    def compare_to_solver(
        self,
        airfoil: Union[str, Path, Airfoil],
        alpha_deg: float,
        reynolds: float = 5.0e5,
        grid: Optional[Grid] = None,
    ) -> dict:
        """Surrogate vs panel solver on the same case: per-channel errors."""
        grid = grid or Grid()
        af = airfoil if isinstance(airfoil, Airfoil) else load_dat(airfoil)
        pred = self.predict(af, alpha_deg, reynolds, grid)
        truth = simulate_case(CasePrep.build(af), alpha_deg, reynolds, grid)

        tgt = truth["target"]
        fluid = truth["mask"]
        solid = truth["solid_mask"]
        if self.downsample > 1:
            tgt = np.stack([_avg_pool_np(c, self.downsample) for c in tgt])
            fluid = (_avg_pool_np(fluid, self.downsample) > 0.999).astype(np.float32)
            solid = (_avg_pool_np(solid, self.downsample) > 0.999).astype(np.float32)

        errs = {}
        for i, name in enumerate(TARGET_CHANNELS):
            m = fluid if i < 3 else solid
            w = m.sum() + 1e-8
            diff = (pred["fields"][name] - tgt[i]) * m
            rms_true = np.sqrt((tgt[i] ** 2 * m).sum() / w)
            errs[name] = {
                "mae": float(np.abs(diff).sum() / w),
                "rmse": float(np.sqrt((diff ** 2).sum() / w)),
                "relative_rmse": float(
                    np.sqrt((diff ** 2).sum() / w) / (rms_true + 1e-8)
                ),
            }

        truth_scalars = {
            "cl": truth["coeffs"]["cl"], "cd": truth["coeffs"]["cd"],
            "cm": truth["coeffs"]["cm"], "w_max": truth["coeffs"]["w_max"],
            "twist_te_deg": truth["coeffs"]["twist_te_deg"],
        }
        scalar_err = {
            k: {
                "pred": pred["coeffs"].get(k),
                "true": truth_scalars[k],
                "abs_error": abs(pred["coeffs"].get(k, 0.0) - truth_scalars[k]),
            }
            for k in truth_scalars
        }
        return {"fields": errs, "scalars": scalar_err,
                "alpha_deg": alpha_deg, "airfoil": af.name}


def _converged_npz(cache: SensorCache) -> Dict[str, bool]:
    """Map npz path -> convergence flag, read from each file's own metadata."""
    out: Dict[str, bool] = {}
    for case in cache.cases:
        path = case.get("npz")
        if not path or not Path(path).exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
        except Exception:
            continue
        conv = meta.get("converged")
        # None means the backend has nothing to converge (the panel solve is
        # closed-form), which counts as usable.
        out[path] = True if conv is None else bool(conv)
    return out


def _avg_pool_np(a: np.ndarray, k: int) -> np.ndarray:
    h, w = a.shape[-2:]
    a = a[..., : h - h % k, : w - w % k]
    return a.reshape(*a.shape[:-2], h // k, k, w // k, k).mean(axis=(-3, -1))


def evaluate_split(
    checkpoint: Union[str, Path] = DEFAULT_CHECKPOINT,
    cache_dir: str = "artifacts/sensor/cache",
    split: str = "test",
    device: str = "auto",
    limit: Optional[int] = None,
    out_path: Optional[Union[str, Path]] = None,
    fidelity: Optional[str] = None,
    converged_only: bool = False,
    tta: bool = True,
) -> dict:
    """Score the surrogate over a whole split, straight from the cache.

    Reports per-channel relative RMSE on the fields and, for the coefficients,
    both MAE and R^2 -- MAE alone hides the fact that Cd spans two decades.

    ``fidelity`` restricts to ``"panel"``, ``"rans"`` or ``"urans"`` rows.
    Scoring a mixed split gives a number that is neither: on panel rows the
    network is reproducing one distribution, on RANS rows another. Report them
    apart.

    ``converged_only`` additionally drops rows whose residuals never fell, which
    for a *steady* solver on separated flow is most of the post-stall sweep.
    Comparing against a limit-cycle snapshot measures nothing. URANS rows carry
    a statistical-stationarity flag in the same field, so the same switch works
    for them.
    """
    import torch

    p = SurrogatePredictor(checkpoint, device, tta=tta)
    cache = SensorCache(cache_dir)
    if cache.spec.names != p.spec.names:
        raise ValueError(
            f"cache was built with the {cache.spec.name} input contract "
            f"({cache.spec.n_channels} channels) but the checkpoint expects "
            f"{p.spec.name} ({p.spec.n_channels}). Rebuild the cache with "
            f"`python cli.py sensor-cache` or score a matching checkpoint."
        )
    idx = cache.indices(split, fidelity=fidelity)
    if converged_only:
        conv = _converged_npz(cache)
        idx = np.asarray(
            [i for i in idx if conv.get(cache.cases[int(i)]["npz"], True)],
            dtype=np.int64,
        )
    if limit:
        idx = idx[:limit]
    if idx.size == 0:
        raise ValueError(
            f"no cases in split '{split}'"
            + (f" at fidelity '{fidelity}'" if fidelity else "")
            + (" that converged" if converged_only else "")
        )

    x_mean = np.asarray(p.stats["x_mean"], np.float32)[:, None, None]
    x_std = np.asarray(p.stats["x_std"], np.float32)[:, None, None]

    ch_num = {c: 0.0 for c in TARGET_CHANNELS}
    ch_den = {c: 0.0 for c in TARGET_CHANNELS}
    ch_mae = {c: 0.0 for c in TARGET_CHANNELS}
    ch_n = {c: 0.0 for c in TARGET_CHANNELS}
    s_pred, s_true = [], []

    for j in idx:
        xi = np.asarray(cache.inputs[j], np.float32)
        yi = np.asarray(cache.targets[j], np.float32)
        mi = np.asarray(cache.masks[j], np.float32)
        si = np.asarray(cache.scalars[j], np.float32)

        t = torch.from_numpy(((xi - x_mean) / x_std).astype(np.float32))[None]
        if p.downsample > 1:
            t = torch.nn.functional.avg_pool2d(t, p.downsample)
            yi = np.stack([_avg_pool_np(c, p.downsample) for c in yi])
            mi = np.stack([(_avg_pool_np(c, p.downsample) > 0.999).astype(np.float32)
                           for c in mi])
        yp, sp = p._forward(t.to(p.device))
        yp = yp[0].cpu().numpy()
        sp = sp[0].cpu().numpy()
        yp, sp = p._denormalise(yp, sp)

        for i, name in enumerate(TARGET_CHANNELS):
            m = mi[0] if i < 3 else mi[1]
            d = (yp[i] - yi[i]) * m
            ch_num[name] += float((d ** 2).sum())
            ch_den[name] += float((yi[i] ** 2 * m).sum())
            ch_mae[name] += float(np.abs(d).sum())
            ch_n[name] += float(m.sum())
        s_pred.append(sp)
        s_true.append(si)

    s_pred = np.asarray(s_pred)
    s_true = np.asarray(s_true)
    scalars = {}
    for i, name in enumerate(p.scalar_names):
        err = s_pred[:, i] - s_true[:, i]
        ss_res = float((err ** 2).sum())
        ss_tot = float(((s_true[:, i] - s_true[:, i].mean()) ** 2).sum())
        scalars[name] = {
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "r2": float(1.0 - ss_res / (ss_tot + 1e-12)),
        }

    fields = {
        name: {
            "relative_rmse": float(np.sqrt(ch_num[name] / (ch_den[name] + 1e-12))),
            "mae": float(ch_mae[name] / (ch_n[name] + 1e-12)),
        }
        for name in TARGET_CHANNELS
    }
    result = {
        "split": split,
        "fidelity": fidelity or "all",
        "converged_only": converged_only,
        "tta": bool(tta),
        "input_spec": p.spec.name,
        "n_cases": int(len(idx)),
        "checkpoint": str(checkpoint),
        "downsample": p.downsample,
        "epoch": p.epoch,
        "fields": fields,
        "scalars": scalars,
    }
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result
