"""Figures for the surrogate: flow fields, polars, dataset coverage, accuracy.

    python -m sensor.figures --out poster/generated

Every figure is generated from the artefacts actually on disk -- the manifest,
the cache, the checkpoint -- so a figure cannot show a result the repository
does not contain. If the surrogate is untrained the model panels are skipped
rather than faked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .geometry import load_dat  # noqa: E402
from .sdf import Grid  # noqa: E402
from .simulate import CasePrep, simulate_case  # noqa: E402

FOIL_0009 = "Data/Airfoils/4/n0009sm.dat.txt"
CMAP = "RdBu_r"


def _style(ax, title=""):
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_fields(out: Path, alpha_deg: float = 12.0):
    """Five target channels for one case, plus the section outline."""
    grid = Grid()
    prep = CasePrep.build(load_dat(FOIL_0009))
    case = simulate_case(prep, alpha_deg, 5.0e5, grid)
    poly = case["polygon"]
    t = case["target"]
    names = ["u / U", "v / U", "p / rho U^2", "load  dCp", "displacement w/c"]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.1))
    extent = [*grid.xlim, *grid.ylim]
    for i, (ax, name) in enumerate(zip(axes, names)):
        data = t[i]
        region = case["mask"] if i < 3 else case["solid_mask"]
        vals = data[region > 0]
        lim = np.percentile(np.abs(vals), 99) if vals.size else 1.0
        centre = 1.0 if i == 0 else 0.0
        im = ax.imshow(data, origin="lower", extent=extent, cmap=CMAP,
                       vmin=centre - lim, vmax=centre + lim, aspect="equal")
        ax.plot(poly[:, 0], poly[:, 1], "k-", lw=0.8)
        _style(ax, name)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle(
        f"Surrogate targets: NACA 0009, alpha = {alpha_deg:g} deg, Re = 5e5 "
        f"(panel + Beddoes-Leishman + wake; chordwise beam)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_sensor_fields.png", dpi=160)
    plt.close(fig)


def fig_separation_series(out: Path):
    """Streamwise velocity across incidence: attached -> separated -> stalled."""
    grid = Grid()
    prep = CasePrep.build(load_dat(FOIL_0009))
    alphas = [2.0, 8.0, 12.0, 16.0, 20.0]
    fig, axes = plt.subplots(1, len(alphas), figsize=(16, 3.1))
    extent = [*grid.xlim, *grid.ylim]
    for ax, a in zip(axes, alphas):
        case = simulate_case(prep, a, 5.0e5, grid)
        u = case["target"][0]
        im = ax.imshow(u, origin="lower", extent=extent, cmap=CMAP,
                       vmin=0.0, vmax=1.6, aspect="equal")
        ax.plot(case["polygon"][:, 0], case["polygon"][:, 1], "k-", lw=0.8)
        c = case["coeffs"]
        _style(ax, f"a = {a:g} deg   Cl = {c['cl']:.2f}   f = {c['f_sep']:.2f}")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("Separation growth with incidence (u / U)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_sensor_separation.png", dpi=160)
    plt.close(fig)


def fig_polar(out: Path):
    """Lift and drag polars against published reference points."""
    grid = Grid(nx=64, ny=64)
    foils = {
        "NACA 0009": "Data/Airfoils/4/n0009sm.dat.txt",
        "NACA 0012": "Data/Airfoils/4/n0012.dat.txt",
        "NACA 4412": "Data/Airfoils/4/naca4412.dat.txt",
    }
    alphas = np.arange(-4, 21, 2.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for label, path in foils.items():
        prep = CasePrep.build(load_dat(path))
        cl, cd = [], []
        for a in alphas:
            c = simulate_case(prep, float(a), 5.0e5, grid, with_modal=False)["coeffs"]
            cl.append(c["cl"])
            cd.append(c["cd"])
        ax1.plot(alphas, cl, "-o", ms=3, label=label)
        ax2.plot(alphas, cd, "-o", ms=3, label=label)

    # published anchors (Abbott & von Doenhoff / Sheldahl & Klimas, Re = 5e5)
    ax1.plot([8, 13], [0.85, 1.28], "k*", ms=11, label="NACA 0012 reference")
    ax2.plot([0, 20], [0.010, 0.30], "k*", ms=11, label="NACA 0012 reference")

    ax1.set_xlabel("angle of attack [deg]")
    ax1.set_ylabel("$C_l$")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.set_xlabel("angle of attack [deg]")
    ax2.set_ylabel("$C_d$")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    fig.suptitle("Solver polars vs published data, Re = 5e5", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_sensor_polar.png", dpi=160)
    plt.close(fig)


def fig_dataset(out: Path, manifest: str = "artifacts/cfd/manifest.jsonl"):
    """What the 1,926 cases actually cover."""
    path = Path(manifest)
    if not path.exists():
        return
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    splits = json.loads(Path("artifacts/cfd/splits.json").read_text())
    where = {f: s for s, fs in splits.items() for f in fs}

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    colours = {"train": "#4C72B0", "val": "#DD8452", "test": "#C44E52"}

    ax = axes[0]
    for split, colour in colours.items():
        sub = [r for r in rows if where.get(r["foil"]) == split]
        ax.scatter([r["thickness"] for r in sub], [r["alpha_deg"] for r in sub],
                   s=4, alpha=0.5, c=colour, label=f"{split} ({len(sub)})")
    ax.set_xlabel("max thickness t/c")
    ax.set_ylabel("angle of attack [deg]")
    ax.set_title(f"{len(rows)} cases, {len({r['foil'] for r in rows})} sections",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for split, colour in colours.items():
        sub = [r for r in rows if where.get(r["foil"]) == split]
        ax.scatter([r["cd"] for r in sub], [r["cl"] for r in sub],
                   s=4, alpha=0.5, c=colour)
    ax.set_xscale("log")
    ax.set_xlabel("$C_d$")
    ax.set_ylabel("$C_l$")
    ax.set_title("coefficient coverage", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    series = sorted({r["series"] for r in rows})
    counts = [sum(1 for r in rows if r["series"] == s) for s in series]
    ax.bar([f"{s}-series" for s in series], counts, color="#4C72B0")
    deep = sum(1 for r in rows if r.get("deep_stall"))
    ax.set_title(f"by family  ({deep} cases tagged deep_stall)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out / "fig_sensor_dataset.png", dpi=160)
    plt.close(fig)


def fig_surrogate_accuracy(
    out: Path,
    checkpoint: str = "artifacts/sensor/run/best.pt",
    cache_dir: str = "artifacts/sensor/cache",
):
    """Predicted vs true coefficients on the held-out foils, and a field triple."""
    if not Path(checkpoint).exists():
        return
    from .predict import LegacyCheckpointError, SurrogatePredictor, _avg_pool_np
    from .data import SensorCache
    import torch

    try:
        p = SurrogatePredictor(checkpoint)
    except LegacyCheckpointError as exc:
        print(f"skipping fig_surrogate_accuracy: {exc}")
        return
    cache = SensorCache(cache_dir)
    # The cache stores the input tensor, so a cache built under a different
    # channel contract than the checkpoint expects is a silent broadcast error
    # nine lines down. Say so instead.
    if cache.spec.names != p.spec.names:
        print(
            f"skipping fig_surrogate_accuracy: cache is {cache.spec.name} "
            f"({cache.spec.n_channels} channels), checkpoint expects "
            f"{p.spec.name} ({p.spec.n_channels}). Rebuild with "
            f"`python cli.py sensor-cache`."
        )
        return
    idx = cache.indices("test")
    if idx.size == 0:
        return

    x_mean = np.asarray(p.stats["x_mean"], np.float32)[:, None, None]
    x_std = np.asarray(p.stats["x_std"], np.float32)[:, None, None]
    preds, trues = [], []
    for j in idx:
        xi = np.asarray(cache.inputs[j], np.float32)
        t = torch.from_numpy(((xi - x_mean) / x_std).astype(np.float32))[None]
        if p.downsample > 1:
            t = torch.nn.functional.avg_pool2d(t, p.downsample)
        # Through the predictor's own forward, so the figure shows the model as
        # released -- with test-time mirror averaging -- rather than a variant
        # of it that only this file uses.
        _, sp = p._forward(t.to(p.device))
        sp = sp[0].cpu().numpy()
        sp = sp * np.asarray(p.stats["s_std"], np.float32) + np.asarray(
            p.stats["s_mean"], np.float32)
        preds.append(sp)
        trues.append(np.asarray(cache.scalars[j], np.float32))
    preds = np.asarray(preds)
    trues = np.asarray(trues)

    names = p.scalar_names[:3]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.scatter(trues[:, i], preds[:, i], s=6, alpha=0.4, c="#4C72B0")
        lo = float(min(trues[:, i].min(), preds[:, i].min()))
        hi = float(max(trues[:, i].max(), preds[:, i].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ss_res = float(((preds[:, i] - trues[:, i]) ** 2).sum())
        ss_tot = float(((trues[:, i] - trues[:, i].mean()) ** 2).sum())
        ax.set_title(f"{name}   $R^2$ = {1 - ss_res / (ss_tot + 1e-12):.3f}", fontsize=9)
        ax.set_xlabel("solver")
        ax.set_ylabel("U-Net")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Held-out sections ({len(idx)} cases, {len(cache.splits['test'])} unseen "
        f"shapes incl. NACA 0009)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_sensor_accuracy.png", dpi=160)
    plt.close(fig)

    # field comparison on NACA 0009
    grid = Grid()
    pred = p.predict(FOIL_0009, 12.0, 5.0e5, grid)
    truth = simulate_case(CasePrep.build(load_dat(FOIL_0009)), 12.0, 5.0e5, grid)
    tgt = truth["target"][0]
    if p.downsample > 1:
        tgt = _avg_pool_np(tgt, p.downsample)
    up = pred["fields"]["u"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    extent = [*grid.xlim, *grid.ylim]
    for ax, data, name, kw in [
        (axes[0], tgt, "solver  u/U", dict(vmin=0, vmax=1.6, cmap=CMAP)),
        (axes[1], up, "U-Net  u/U", dict(vmin=0, vmax=1.6, cmap=CMAP)),
        (axes[2], up - tgt, "error", dict(vmin=-0.2, vmax=0.2, cmap=CMAP)),
    ]:
        im = ax.imshow(data, origin="lower", extent=extent, aspect="equal", **kw)
        _style(ax, name)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("NACA 0009 at 12 deg -- held out of training", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_sensor_field_compare.png", dpi=160)
    plt.close(fig)


def fig_training_curve(out: Path, history: str = "artifacts/sensor/run/history.json"):
    path = Path(history)
    if not path.exists():
        return
    hist = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot([h["epoch"] for h in hist], [h["train"]["loss"] for h in hist],
            label="train")
    ax.plot([h["epoch"] for h in hist], [h["val"]["loss"] for h in hist], label="val")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked loss")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title("U-Net training", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_sensor_training.png", dpi=160)
    plt.close(fig)


def fig_rans_validation(out: Path, manifest: str = "artifacts/cfd/manifest.jsonl"):
    """RANS vs panel polars on NACA 0009, with convergence marked.

    The point of the figure is the *split*: the converged branch is a genuine
    validation, the rest is a steady solver limit-cycling on separated flow.
    Drawing them the same way would be dishonest, so non-converged points are
    hollow and carry their oscillation amplitude as an error bar.
    """
    path = Path(manifest)
    if not path.exists():
        return
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rans = sorted((r for r in rows if r.get("fidelity") == "rans"),
                  key=lambda r: r["alpha_deg"])
    if not rans:
        return
    foil = rans[0]["foil"]
    # Compute the panel polar directly rather than reading it from the
    # manifest: the merge replaces panel rows wherever a RANS case exists at
    # the same angle, so the manifest only retains the non-overlapping angles
    # and the comparison curve would come out full of holes.
    panel = []
    dat = rans[0].get("dat")
    if dat and Path(dat).exists():
        prep = CasePrep.build(load_dat(dat))
        grid = Grid(nx=64, ny=64)
        for a in np.arange(-8.0, 20.5, 1.0):
            c = simulate_case(prep, float(a), 5.0e5, grid, with_modal=False)["coeffs"]
            panel.append({"alpha_deg": float(a), "cl": c["cl"], "cd": c["cd"]})

    ok = [r for r in rans if r.get("converged")]
    bad = [r for r in rans if not r.get("converged")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, label in ((axes[0], "cl", "$C_l$"), (axes[1], "cd", "$C_d$")):
        if panel:
            ax.plot([r["alpha_deg"] for r in panel], [r[key] for r in panel],
                    "-", color="#4C72B0", lw=1.5, label="panel (training target)")
        err = [r.get(f"{key}_std") or 0.0 for r in bad]
        ax.errorbar([r["alpha_deg"] for r in bad], [r[key] for r in bad], yerr=err,
                    fmt="o", mfc="none", color="#C44E52", ms=6, capsize=3,
                    label="RANS, not converged")
        ax.plot([r["alpha_deg"] for r in ok], [r[key] for r in ok],
                "s", color="#C44E52", ms=6, label="RANS, converged")
        ax.set_xlabel("angle of attack [deg]")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[1].set_yscale("log")

    # published anchors for NACA 0009 at Re = 5e5
    axes[1].plot([0], [0.007], "k*", ms=12, label="published")
    axes[1].legend(fontsize=7)

    ax = axes[2]
    ax.semilogy([r["alpha_deg"] for r in rans],
                [r.get("residual_max") or 1e-8 for r in rans], "o-", color="#55A868")
    ax.axhline(1e-4, color="k", ls="--", lw=1)
    ax.text(-7, 1.4e-4, "convergence tolerance", fontsize=7)
    ax.set_xlabel("angle of attack [deg]")
    ax.set_ylabel("max initial residual")
    ax.set_title(f"{len(ok)} of {len(rans)} cases converged", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Steady RANS (k-omega SST) on {foil}, Re = 5e5 -- held out of training. "
        f"Post-stall cases limit-cycle and are excluded.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_rans_validation.png", dpi=160)
    plt.close(fig)


def fig_fidelity_transfer(
    out: Path,
    panel_json: str = "artifacts/sensor/eval_test_panel.json",
    rans_json: str = "artifacts/sensor/eval_test_rans.json",
):
    """How the panel-trained surrogate holds up when the truth changes solver."""
    pp, rp = Path(panel_json), Path(rans_json)
    if not (pp.exists() and rp.exists()):
        return
    panel = json.loads(pp.read_text())
    rans = json.loads(rp.read_text())

    fields = ["u", "v", "p", "load", "disp"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(fields))
    ax1.bar(x - 0.2, [panel["fields"][f]["relative_rmse"] for f in fields], 0.4,
            label=f"vs panel (n={panel['n_cases']})", color="#4C72B0")
    ax1.bar(x + 0.2, [rans["fields"][f]["relative_rmse"] for f in fields], 0.4,
            label=f"vs RANS (n={rans['n_cases']})", color="#C44E52")
    ax1.set_xticks(x)
    ax1.set_xticklabels(fields)
    ax1.set_ylabel("relative RMSE")
    ax1.set_title("field channels", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    # MAE only: R^2 is meaningless for the near-zero-variance scalars at n = 7
    scal = ["cl", "cd", "cm", "w_max", "twist_te_deg"]
    x = np.arange(len(scal))
    pv = [panel["scalars"][s]["mae"] for s in scal]
    rv = [rans["scalars"][s]["mae"] for s in scal]
    ax2.bar(x - 0.2, pv, 0.4, label="vs panel", color="#4C72B0")
    ax2.bar(x + 0.2, rv, 0.4, label="vs RANS", color="#C44E52")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scal, rotation=15)
    ax2.set_ylabel("mean absolute error")
    ax2.set_title("integrated quantities (MAE; $R^2$ omitted, n=7)", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y", which="both")

    fig.suptitle(
        "Cross-fidelity transfer: one network, trained on panel targets only, "
        "scored against each solver on the held-out section", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_fidelity_transfer.png", dpi=160)
    plt.close(fig)


def fig_urans_shedding(out: Path, manifest: str = "artifacts/cfd/manifest.jsonl"):
    """What the unsteady runs bought: a measured shedding frequency past stall.

    Left: mean lift with the RMS fluctuation as an error bar, against the steady
    branch. Right: the projected-height Strouhal number, which is the number the
    bluff-body literature quotes and which should sit near 0.2 if the wake is
    really shedding a von Karman street. Cases whose statistics did not settle
    are drawn hollow rather than dropped -- how many of them there are is part of
    the result.
    """
    path = Path(manifest)
    if not path.exists():
        return
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    urans = sorted(
        (r for r in rows if r.get("fidelity") == "urans" and r.get("ok")),
        key=lambda r: r["alpha_deg"],
    )
    if not urans:
        return
    steady = sorted(
        (r for r in rows if r.get("fidelity") == "rans" and r.get("ok")
         and r.get("converged")),
        key=lambda r: r["alpha_deg"],
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    a = np.array([r["alpha_deg"] for r in urans])
    cl = np.array([r["cl"] for r in urans], dtype=float)
    rms = np.array([r.get("cl_std") or 0.0 for r in urans], dtype=float)
    ok = np.array([bool(r.get("converged")) for r in urans])

    ax1.errorbar(a[ok], cl[ok], yerr=rms[ok], fmt="o", color="#C44E52", capsize=3,
                 label="URANS, statistics settled")
    if (~ok).any():
        ax1.errorbar(a[~ok], cl[~ok], yerr=rms[~ok], fmt="o", mfc="none",
                     color="#C44E52", capsize=3, label="URANS, not settled")
    if steady:
        ax1.plot([r["alpha_deg"] for r in steady], [r["cl"] for r in steady],
                 "s-", color="#4C72B0", label="steady RANS, converged")
    ax1.set_xlabel(r"$\alpha$ [deg]")
    ax1.set_ylabel(r"$C_l$")
    ax1.set_title("mean lift, error bar = RMS fluctuation", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    st = np.array([r.get("strouhal_projected") or np.nan for r in urans], dtype=float)
    ax2.plot(a[ok], st[ok], "o", color="#C44E52")
    if (~ok).any():
        ax2.plot(a[~ok], st[~ok], "o", mfc="none", color="#C44E52")
    ax2.axhspan(0.15, 0.25, color="grey", alpha=0.2,
                label="bluff-body band, St $\\approx$ 0.2")
    ax2.set_xlabel(r"$\alpha$ [deg]")
    ax2.set_ylabel(r"$St_d = f\,d/U$,   $d = c(\sin|\alpha| + t/c)$")
    ax2.set_title("measured shedding, projected height", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Unsteady RANS past stall: the steady solver has no fixed point here, "
        "and only the unsteady run measures a frequency", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_urans_shedding.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="poster/generated")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for fn in (fig_fields, fig_separation_series, fig_polar, fig_dataset,
               fig_surrogate_accuracy, fig_training_curve,
               fig_rans_validation, fig_fidelity_transfer, fig_urans_shedding):
        try:
            fn(out)
            print(f"  {fn.__name__} ok", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad figure must not stop the rest
            print(f"  {fn.__name__} FAILED: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
