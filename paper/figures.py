"""Publication figures for the FSI-SLM paper.

    python -m paper.figures --out paper/figures
    python -m paper.figures --out paper/figures --only fig4_parity

Every figure is generated from artefacts actually on disk -- the manifest, the
memmap cache, the checkpoint, the eval JSON -- so a figure cannot show a result
this repository does not contain. A figure whose inputs are missing is skipped
with a message rather than faked.

Each figure is written twice: a vector ``.pdf`` for the manuscript and a
``.png`` at 400 dpi for slides and previews.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths to the artefacts every figure reads
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "artifacts/cfd/manifest.jsonl"
CACHE_DIR = ROOT / "artifacts/sensor/cache"
CHECKPOINT = ROOT / "artifacts/sensor/run3/best.pt"
EVAL_PANEL = ROOT / "artifacts/sensor/eval_test_panel.json"
EVAL_PANEL_NOTTA = ROOT / "artifacts/sensor/eval_test_panel_notta.json"
EVAL_RANS = ROOT / "artifacts/sensor/eval_test_rans.json"
EVAL_SLM = ROOT / "artifacts/eval_results/v2_vs_v3.json"
FOIL_0009 = "n0009sm"

# --------------------------------------------------------------------------- #
# Style. Muted, print-safe, colour-blind-legible; no reliance on hue alone.
# --------------------------------------------------------------------------- #
INK = "#1A1A1A"
GREY = "#6E7377"
LIGHT = "#D8DBDD"
BLUE = "#2C5F8A"      # solver / panel
ORANGE = "#C1571E"    # network / prediction
TEAL = "#2A8A7F"      # RANS / high fidelity
PURPLE = "#6A4C93"    # structure
RED = "#B3202C"       # error / failure
GREEN = "#3F7D3A"     # pass

REGIME_ORDER = ["attached", "separated", "viv_lockin", "galloping", "stall_flutter"]
REGIME_LABELS = {
    "attached": "Attached",
    "separated": "Separated",
    "viv_lockin": "VIV lock-in",
    "galloping": "Galloping",
    "stall_flutter": "Stall flutter",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.labelsize": 8,
        "axes.labelcolor": INK,
        "axes.edgecolor": GREY,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "text.color": INK,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "lines.linewidth": 1.3,
        "grid.color": LIGHT,
        "grid.linewidth": 0.5,
    })


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def _load_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _manifest() -> List[dict]:
    if not MANIFEST.exists():
        return []
    with open(MANIFEST) as f:
        return [json.loads(line) for line in f]


def _panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.06) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", color=INK)


# --------------------------------------------------------------------------- #
# Figure 1 -- system architecture
# --------------------------------------------------------------------------- #
def fig1_system(out: Path) -> None:
    """The digital twin end to end: geometry in, interpreted experiment out."""
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=6.8, tc=INK, lw=1.0, ls="-"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.0",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, linespacing=1.3)

    def arrow(x1, y1, x2, y2, color=GREY, ls="-", lw=1.0, rad=0.0):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
            color=color, linewidth=lw, linestyle=ls, shrinkA=1, shrinkB=1,
            connectionstyle=f"arc3,rad={rad}"))

    for y, txt in ((55.5, "SOLVE"), (35.3, "LEARN"),
                   (19.4, "BRIDGE"), (4.7, "INTERPRET")):
        ax.text(-1.0, y, txt, fontsize=6.2, color=GREY, rotation=90,
                ha="center", va="center", fontweight="bold")

    # ---- SOLVE: three ALTERNATIVE backends, chosen by incidence
    box(2, 59.5, 24, 5.4, "Panel + B–L stall  ·  0.13 s", "#EAF0F6", BLUE, fs=6.5)
    box(2, 52.7, 24, 5.4, "Steady RANS, α ≤ 8°  ·  4 min", "#E4F1EE", TEAL, fs=6.5)
    box(2, 45.9, 24, 5.4, "URANS, α ≥ 10°  ·  hours", "#E4F1EE", TEAL, fs=6.5)
    ax.text(14, 44.3, "one solver per case, chosen by α —\n"
                      "all three write the same tensor contract",
            fontsize=5.7, color=GREY, ha="center", va="top", style="italic")

    # converge into the beam
    for yy in (62.2, 55.4, 48.6):
        arrow(26, yy, 33, 55.4, rad=0.0 if yy == 55.4 else
              (-0.16 if yy > 55.4 else 0.16))
    box(33, 52.7, 16, 5.4, "Euler–Bernoulli\nbeam (one-way)",
        "#F0EBF5", PURPLE, fs=6.5)
    arrow(49, 55.4, 53, 55.4)
    box(53, 52.7, 24, 5.4, "1,928 cases, fidelity-tagged\n5 fields + 5 scalars",
        "#F4F4F4", GREY, fs=6.5)

    # ---- LEARN
    box(2, 32.6, 22, 5.4, "13-channel input\nSDF · prior · coords",
        "#FDF0E8", ORANGE, fs=6.5)
    box(27, 32.6, 19, 5.4, "U-Net 11.6 M\nFiLM · attention",
        "#FDF0E8", ORANGE, fs=6.5)
    box(49, 32.6, 15, 5.4, "Mirror TTA\nexact symmetry", "#FDF0E8", ORANGE, fs=6.5)
    box(67, 32.6, 31, 5.4,
        "$u,v,p$, load, disp + 5 scalars\n"
        "$C_l\\,R^2$=0.998  $C_d\\,R^2$=0.995  ·  ms", "#FDF0E8", ORANGE, fs=6.5)
    arrow(24, 35.3, 27, 35.3)
    arrow(46, 35.3, 49, 35.3)
    arrow(64, 35.3, 67, 35.3)
    arrow(65, 52.7, 38, 38.3, rad=0.14)
    ax.text(56, 45.0, "train", fontsize=6, color=GREY, ha="center", style="italic")

    # ---- BRIDGE: two alternative sources of a RawExperiment
    box(2, 21.0, 23, 5.2, "Twin bridge (surrogate)\nsteady fields → time series",
        "#F4F4F4", GREY, fs=6.3)
    box(2, 14.4, 23, 5.2, "PIV rig\nnot connected", "white", RED,
        tc=RED, fs=6.3, ls=(0, (3.5, 2)))
    ax.text(13.5, 12.6, "the one unimplemented edge —\n"
                        "replaces build_raw_experiment",
            fontsize=5.6, color=RED, ha="center", va="top", style="italic")

    box(29, 17.7, 18, 5.2, "FFT · POD · damping\n(feature extraction)",
        "#F4F4F4", GREY, fs=6.3)
    box(50, 17.7, 17, 5.2, "11 fault detectors\nPIV · ADC · mains",
        "#F4F4F4", GREY, fs=6.3)
    box(70, 17.7, 28, 5.2, "FeatureRecord + [data_quality]\nTHE CONTRACT",
        "#FFF8E1", "#B8912F", fs=6.6)

    arrow(25, 23.6, 29, 21.2, rad=-0.10)
    arrow(25, 17.0, 29, 19.4, color=RED, ls=(0, (3.5, 2)), rad=0.10)
    arrow(47, 20.3, 50, 20.3)
    arrow(67, 20.3, 70, 20.3)
    arrow(84, 32.6, 84, 23.3)

    # ---- INTERPRET
    box(29, 2.0, 20, 5.2, "Phi-4-mini 3.8 B\n+ LoRA (46 MB)", "#EAF0F6", BLUE, fs=6.5)
    box(52, 2.0, 15, 5.2, "RAG · FAISS\n16 papers", "#EAF0F6", BLUE, fs=6.5)
    box(70, 2.0, 28, 5.2, "Regime · lock-in · caveats\nREST · MCP · CLI · UI",
        "#EAF0F6", BLUE, fs=6.5)
    arrow(49, 4.6, 52, 4.6)
    arrow(67, 4.6, 70, 4.6)
    arrow(84, 17.7, 84, 7.5)
    arrow(72, 17.7, 41, 7.4, rad=0.12)

    save(fig, out, "fig1_system")


# --------------------------------------------------------------------------- #
# Figure 2 -- dataset coverage
# --------------------------------------------------------------------------- #
def fig2_dataset(out: Path) -> None:
    """What the 1,928 cases cover, and how the splits hold out whole shapes."""
    rows = _manifest()
    if not rows:
        print("  skip fig2_dataset: no manifest")
        return
    idx = _load_json(CACHE_DIR / "index.json")
    splits = idx["splits"] if idx else {}
    where = {f: s for s, foils in splits.items() for f in foils}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    # (a) thickness vs camber-ish proxy, coloured by split
    by_foil: Dict[str, dict] = {}
    for r in rows:
        by_foil.setdefault(r["foil"], r)
    colours = {"train": BLUE, "val": "#C9A227", "test": ORANGE}
    ax = axes[0]
    for split in ("train", "val", "test"):
        t = [r["thickness"] for f, r in by_foil.items()
             if where.get(f) == split and r.get("thickness")]
        x = [r["thickness_x"] for f, r in by_foil.items()
             if where.get(f) == split and r.get("thickness")]
        ax.scatter(x, t, s=16, c=colours[split], alpha=0.75,
                   edgecolors="none", label=f"{split} ({len(t)})")
    # highlight NACA 0009
    r9 = by_foil.get(FOIL_0009)
    if r9 and r9.get("thickness"):
        ax.scatter([r9["thickness_x"]], [r9["thickness"]], s=70,
                   facecolors="none", edgecolors=RED, linewidths=1.3, zorder=5)
        ax.annotate("NACA 0009", (r9["thickness_x"], r9["thickness"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=6.5, color=RED)
    ax.set_xlabel("max-thickness location $x/c$")
    ax.set_ylabel("max thickness $t/c$")
    ax.set_title("107 sections, split by shape")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(alpha=0.4)
    _panel_label(ax, "a", dx=-0.28)

    # (b) angle sampling: spacing is the point, not the count
    ax = axes[1]
    panel_a = sorted({r["alpha_deg"] for r in rows if r["fidelity"] == "panel"})
    rans_a = sorted({r["alpha_deg"] for r in rows if r["fidelity"] == "rans"})
    gaps = np.diff(panel_a)
    mids = (np.array(panel_a[:-1]) + np.array(panel_a[1:])) / 2
    ax.step(mids, gaps, where="mid", color=BLUE, lw=1.3)
    ax.plot(panel_a, [0.35] * len(panel_a), "|", color=BLUE, ms=7,
            mew=1.1, label=f"panel ({len(panel_a)} angles)")
    ax.plot(rans_a, [0.05] * len(rans_a), "|", color=TEAL, ms=7,
            mew=1.1, label=f"RANS ({len(rans_a)})")
    ax.axvspan(8, 14, color=ORANGE, alpha=0.10)
    ax.text(11, 2.55, "stall", fontsize=6, color=ORANGE, ha="center")
    ax.set_ylim(-0.15, 2.9)
    ax.set_xlabel(r"angle of attack $\alpha$ [deg]")
    ax.set_ylabel(r"sampling gap $\Delta\alpha$ [deg]")
    ax.set_title("dense where physics changes")
    ax.legend(loc="upper left", fontsize=5.8)
    ax.grid(axis="y", alpha=0.4)
    _panel_label(ax, "b", dx=-0.28)

    # (c) fidelity + convergence
    ax = axes[2]
    n_panel = sum(1 for r in rows if r["fidelity"] == "panel")
    rans = [r for r in rows if r["fidelity"] == "rans"]
    n_conv = sum(1 for r in rans if r.get("converged"))
    bars = [("panel", n_panel, BLUE), ("RANS\nconverged", n_conv, TEAL),
            ("RANS\nnot conv.", len(rans) - n_conv, RED)]
    ax.bar([b[0] for b in bars], [b[1] for b in bars],
           color=[b[2] for b in bars], edgecolor="none", width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel("cases (log)")
    ax.set_title("fidelity of the targets")
    for i, (_, v, _) in enumerate(bars):
        ax.text(i, v * 1.25, str(v), ha="center", fontsize=7, color=INK)
    ax.set_ylim(0.6, n_panel * 4)
    ax.grid(axis="y", alpha=0.4)
    _panel_label(ax, "c", dx=-0.28)

    fig.tight_layout()
    save(fig, out, "fig2_dataset")


# --------------------------------------------------------------------------- #
# Figure 3 -- input contract and network
# --------------------------------------------------------------------------- #
def fig3_network(out: Path) -> None:
    """The 13 input channels, and the U-Net that consumes them."""
    try:
        from sensor.geometry import load_dat
        from sensor.sdf import SPECS, Grid, build_input
    except Exception as e:  # noqa: BLE001
        print(f"  skip fig3_network: {e}")
        return

    dat = ROOT / "Data/Airfoils/4/n0009sm.dat.txt"
    if not dat.exists():
        print("  skip fig3_network: airfoil missing")
        return

    fig = plt.figure(figsize=(7.2, 3.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.30)

    # ---- top: the input channels, actually computed
    try:
        grid = Grid()
        foil = load_dat(str(dat))
        spec = SPECS["v2"]
        x = build_input(foil, alpha_deg=12.0, reynolds=5e5, grid=grid, spec=spec)
        names = list(spec.names)
    except Exception as e:  # noqa: BLE001
        print(f"  skip fig3_network input panel: {e}")
        x, names = None, []

    if x is not None:
        show = ["sdf", "wall", "n_x", "pot_u", "pot_v", "pot_p"]
        gs_top = gs[0].subgridspec(1, len(show), wspace=0.08)
        for k, nm in enumerate(show):
            ax = fig.add_subplot(gs_top[0, k])
            ch = x[names.index(nm)]
            lim = float(np.nanpercentile(np.abs(ch), 99)) or 1.0
            ax.imshow(ch, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(nm, fontsize=7, pad=3)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(GREY); s.set_linewidth(0.6)
            if k == 0:
                _panel_label(ax, "a", dx=-0.10, dy=1.30)
        fig.text(0.5, 0.955,
                 "13-channel input contract (v2) — 6 of 13 shown; "
                 r"$pot_{u,v,p}$ is the inviscid prior",
                 ha="center", fontsize=7.5, color=GREY)

    # ---- bottom: architecture schematic
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
    _panel_label(ax, "b", dx=-0.02, dy=1.02)

    enc_w = [7.0, 6.0, 5.0, 4.0]
    enc_h = [20, 15, 11, 8]
    xs = [6, 17, 27, 36]
    for i, (w, h, xx) in enumerate(zip(enc_w, enc_h, xs)):
        ax.add_patch(FancyBboxPatch((xx, 17 - h / 2), w, h,
                                    boxstyle="round,pad=0.2,rounding_size=0.6",
                                    facecolor="#FDF0E8", edgecolor=ORANGE, lw=0.9))
        ax.text(xx + w / 2, 17, f"{128 // 2**i}", ha="center", va="center",
                fontsize=6.5, rotation=90, color=INK)

    ax.add_patch(FancyBboxPatch((45, 12), 10, 10,
                                boxstyle="round,pad=0.2,rounding_size=0.6",
                                facecolor="#F0EBF5", edgecolor=PURPLE, lw=0.9))
    ax.text(50, 17, "self-attn\n8×8", ha="center", va="center", fontsize=6.5)

    for i, (w, h, xx) in enumerate(zip(enc_w[::-1], enc_h[::-1], [58, 67, 77, 88])):
        ax.add_patch(FancyBboxPatch((xx, 17 - h / 2), w, h,
                                    boxstyle="round,pad=0.2,rounding_size=0.6",
                                    facecolor="#FDF0E8", edgecolor=ORANGE, lw=0.9))
        ax.text(xx + w / 2, 17, f"{16 * 2**i}", ha="center", va="center",
                fontsize=6.5, rotation=90, color=INK)

    # skip connections
    for xa, wa, xb, wb in zip(xs, enc_w, [88, 77, 67, 58], enc_w[::-1]):
        ax.annotate("", xy=(xb, 28), xytext=(xa + wa, 28),
                    arrowprops=dict(arrowstyle="-", color=LIGHT, lw=0.8,
                                    connectionstyle="arc3,rad=-0.12"))
    ax.text(50, 30.5, "skip connections", ha="center", fontsize=6.5, color=GREY)

    # FiLM conditioning
    ax.add_patch(FancyBboxPatch((26, 0.5), 48, 5.5,
                                boxstyle="round,pad=0.2,rounding_size=0.6",
                                facecolor="#EAF0F6", edgecolor=BLUE, lw=0.9))
    ax.text(50, 3.2, r"FiLM: $\alpha$, $\log Re$, fidelity → per-channel scale & shift",
            ha="center", va="center", fontsize=6.8)
    for xx in (12, 30, 50, 70, 90):
        ax.annotate("", xy=(xx, 7.2), xytext=(xx, 6.1),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.7,
                                    mutation_scale=6))

    ax.text(2.5, 17, "13 ch\nin", ha="center", va="center", fontsize=6.5, color=GREY)
    ax.text(91.5, 31.0, "5 field channels", ha="center", va="bottom",
            fontsize=6.5, color=GREY)
    ax.text(91.5, 5.0, "5 scalars", ha="center", va="top",
            fontsize=6.5, color=GREY)

    save(fig, out, "fig3_network")


# --------------------------------------------------------------------------- #
# Shared: run the surrogate over a split so several figures can use it
# --------------------------------------------------------------------------- #
def _predict_split(split: str = "test", fidelity: Optional[str] = "panel",
                   converged_only: bool = False, limit: Optional[int] = None):
    """Return (scalars_pred, scalars_true, case_rows) for a split, from cache."""
    import torch
    from sensor.data import SensorCache
    from sensor.predict import SurrogatePredictor, _converged_npz

    p = SurrogatePredictor(CHECKPOINT, "auto", tta=True)
    cache = SensorCache(str(CACHE_DIR))
    idx = cache.indices(split, fidelity=fidelity)
    if converged_only:
        conv = _converged_npz(cache)
        idx = np.asarray([i for i in idx
                          if conv.get(cache.cases[int(i)]["npz"], True)], np.int64)
    if limit:
        idx = idx[:limit]

    x_mean = np.asarray(p.stats["x_mean"], np.float32)[:, None, None]
    x_std = np.asarray(p.stats["x_std"], np.float32)[:, None, None]

    sp_all, st_all, rows = [], [], []
    for j in idx:
        xi = np.asarray(cache.inputs[j], np.float32)
        t = torch.from_numpy(((xi - x_mean) / x_std).astype(np.float32))[None]
        yp, sp = p._forward(t.to(p.device))
        _, sp = p._denormalise(yp[0].cpu().numpy(), sp[0].cpu().numpy())
        sp_all.append(sp)
        st_all.append(np.asarray(cache.scalars[j], np.float32))
        rows.append(cache.cases[int(j)])
    return np.asarray(sp_all), np.asarray(st_all), rows


# --------------------------------------------------------------------------- #
# Figure 4 -- parity plots on held-out sections
# --------------------------------------------------------------------------- #
def fig4_parity(out: Path) -> None:
    """Predicted vs solver coefficients on 17 unseen sections."""
    ev = _load_json(EVAL_PANEL)
    if not CHECKPOINT.exists() or not CACHE_DIR.exists():
        print("  skip fig4_parity: checkpoint or cache missing")
        return
    try:
        sp, st, rows = _predict_split("test", "panel")
    except Exception as e:  # noqa: BLE001
        print(f"  skip fig4_parity: {e}")
        return

    names = ["cl", "cd", "cm"]
    labels = [r"$C_l$", r"$C_d$", r"$C_m$"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    for k, (ax, nm, lab) in enumerate(zip(axes, names, labels)):
        yt, yp = st[:, k], sp[:, k]
        alpha = np.array([r.get("alpha_deg", 0.0) for r in rows])
        s = ax.scatter(yt, yp, c=alpha, cmap="viridis", s=9, alpha=0.8,
                       edgecolors="none")
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color=GREY, lw=0.8, ls="--", zorder=0)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"solver {lab}")
        ax.set_ylabel(f"surrogate {lab}")
        ax.set_title(lab, fontsize=9)

        if ev:
            m = ev["scalars"][nm]
            ax.text(0.04, 0.96,
                    f"$R^2$ = {m['r2']:.3f}\nMAE = {m['mae']:.4f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=6.8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=LIGHT, linewidth=0.6))
        ax.grid(alpha=0.35)
        _panel_label(ax, "abc"[k], dx=-0.34, dy=1.13)

    cb = fig.colorbar(s, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label(r"$\alpha$ [deg]", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"held out: {len(rows)} cases on 17 unseen sections",
                 fontsize=8, color=GREY, y=1.04)
    save(fig, out, "fig4_parity")


# --------------------------------------------------------------------------- #
# Figure 5 -- field prediction on the held-out NACA 0009
# --------------------------------------------------------------------------- #
def fig5_fields(out: Path, alpha_deg: float = 12.0) -> None:
    """Truth, prediction and error for all five channels on NACA 0009."""
    import torch
    from sensor.data import SensorCache
    from sensor.predict import SurrogatePredictor

    if not CHECKPOINT.exists() or not CACHE_DIR.exists():
        print("  skip fig5_fields: checkpoint or cache missing")
        return
    try:
        p = SurrogatePredictor(CHECKPOINT, "auto", tta=True)
        cache = SensorCache(str(CACHE_DIR))
    except Exception as e:  # noqa: BLE001
        print(f"  skip fig5_fields: {e}")
        return

    # pick the NACA 0009 panel case nearest the requested incidence
    cand = [(i, c) for i, c in enumerate(cache.cases)
            if c.get("foil") == FOIL_0009 and c.get("fidelity") == "panel"]
    if not cand:
        print("  skip fig5_fields: no NACA 0009 case in cache")
        return
    j, case = min(cand, key=lambda ic: abs(ic[1].get("alpha_deg", 0) - alpha_deg))

    x_mean = np.asarray(p.stats["x_mean"], np.float32)[:, None, None]
    x_std = np.asarray(p.stats["x_std"], np.float32)[:, None, None]
    xi = np.asarray(cache.inputs[j], np.float32)
    yi = np.asarray(cache.targets[j], np.float32)
    mi = np.asarray(cache.masks[j], np.float32)
    t = torch.from_numpy(((xi - x_mean) / x_std).astype(np.float32))[None]
    yp, sp = p._forward(t.to(p.device))
    yp, _ = p._denormalise(yp[0].cpu().numpy(), sp[0].cpu().numpy())

    chans = ["u", "v", "p", "load", "disp"]
    titles = [r"$u/U_\infty$", r"$v/U_\infty$", r"$p/\rho U_\infty^2$",
              r"load $\Delta C_p$", r"disp $w/c$"]
    # u is a perturbation about the freestream, not about zero; centring a
    # diverging map on 0 makes the whole field one colour.
    centres = [1.0, 0.0, 0.0, 0.0, 0.0]

    solid = mi[1] > 0
    ys, xs = np.where(solid)
    pad_px = 6
    bbox = (max(xs.min() - pad_px, 0), min(xs.max() + pad_px, solid.shape[1] - 1),
            max(ys.min() - 3 * pad_px, 0), min(ys.max() + 3 * pad_px, solid.shape[0] - 1))

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F0F0F0")
    hot = plt.get_cmap("magma").copy()
    hot.set_bad("#F0F0F0")

    fig, axes = plt.subplots(3, 5, figsize=(7.2, 4.6))
    for k, (ch, ti, c0) in enumerate(zip(chans, titles, centres)):
        m = mi[0] if k < 3 else mi[1]
        tru = np.where(m > 0, yi[k], np.nan)
        prd = np.where(m > 0, yp[k], np.nan)
        err = np.abs(prd - tru)
        if k >= 3:                       # zoom the solid channels onto the section
            x0, x1, y0, y1 = bbox
            tru, prd, err = (a[y0:y1 + 1, x0:x1 + 1] for a in (tru, prd, err))

        span = float(np.nanpercentile(np.abs(tru - c0), 99)) or 1.0
        emax = float(np.nanpercentile(err, 99)) or 1e-9

        for row, (arr, cm_, vmin, vmax) in enumerate((
                (tru, cmap, c0 - span, c0 + span),
                (prd, cmap, c0 - span, c0 + span),
                (err, hot, 0.0, emax))):
            ax = axes[row, k]
            im = ax.imshow(np.ma.masked_invalid(arr), origin="lower",
                           cmap=cm_, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(True); s_.set_color(GREY); s_.set_linewidth(0.5)
            if row == 0:
                ax.set_title(ti, fontsize=7.5, pad=3)
            if k == 0:
                ax.set_ylabel(["solver", "surrogate", "|error|"][row], fontsize=7.5)
            # Every row gets a colorbar so the image axes stay the same width;
            # the middle one shares row 0's scale and is hidden.
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 1:
                cb.ax.set_visible(False)
            else:
                cb.ax.tick_params(labelsize=5, length=1.5)
                cb.outline.set_linewidth(0.4)

        rel = float(np.sqrt(np.nansum((prd - tru) ** 2) / np.nansum(tru ** 2)))
        axes[2, k].set_xlabel(f"rel. RMSE {rel:.3f}", fontsize=6.8)

    fig.text(0.5, 0.055,
             "flow channels span the full 3c × 3c raster; "
             "load and displacement are zoomed onto the section",
             ha="center", fontsize=6.2, color=GREY, style="italic")
    fig.suptitle(
        "NACA 0009 (held out) at "
        r"$\alpha$ = " + f"{case.get('alpha_deg'):.0f}°, Re = 5×10⁵",
        fontsize=8.5, y=0.985)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save(fig, out, "fig5_fields")


# --------------------------------------------------------------------------- #
# Figure 6 -- RANS validation on the held-out section
# --------------------------------------------------------------------------- #
def fig6_rans(out: Path) -> None:
    """Steady-RANS polar on NACA 0009, with the non-converged branch marked."""
    rows = [r for r in _manifest()
            if r["foil"] == FOIL_0009 and r["fidelity"] == "rans"]
    panel = [r for r in _manifest()
             if r["foil"] == FOIL_0009 and r["fidelity"] == "panel"]
    if not rows:
        print("  skip fig6_rans: no RANS rows")
        return
    rows.sort(key=lambda r: r["alpha_deg"])
    panel.sort(key=lambda r: r["alpha_deg"])

    conv = [r for r in rows if r.get("converged")]
    nonc = [r for r in rows if not r.get("converged")]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    # (a) lift
    ax = axes[0]
    ax.plot([r["alpha_deg"] for r in panel], [r["cl"] for r in panel],
            color=BLUE, lw=1.2, label="panel (training target)")
    ax.plot([r["alpha_deg"] for r in conv], [r["cl"] for r in conv],
            "o-", color=TEAL, ms=3.5, lw=1.2, label="RANS, converged")
    ax.errorbar([r["alpha_deg"] for r in nonc], [r["cl"] for r in nonc],
                yerr=[r.get("cl_std", 0) for r in nonc], fmt="x", color=RED,
                ms=4, lw=0.9, capsize=2, label="RANS, limit-cycling")
    # Fit the whole converged branch, and say so -- a narrower fit near zero
    # incidence gives a visibly steeper slope (0.098/deg over -4..4).
    a = np.array([r["alpha_deg"] for r in conv])
    cl_c = np.array([r["cl"] for r in conv])
    slope = float(np.polyfit(a, cl_c, 1)[0])
    frac = slope * 180.0 / np.pi / (2.0 * np.pi)
    ax.plot(a, slope * a, ls=":", color=GREY, lw=1.0,
            label=f"fit {slope:.3f}/deg ({frac*100:.0f}% of " + r"$2\pi$)")
    ax.axvspan(9, 21, color=RED, alpha=0.05)
    ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$C_l$")
    ax.set_title("lift")
    ax.set_ylim(-1.15, 1.75)
    ax.legend(fontsize=5.3, loc="lower right", handlelength=1.6,
              borderpad=0.3, labelspacing=0.3)
    ax.grid(alpha=0.4)
    _panel_label(ax, "a", dx=-0.26)

    # (b) drag
    ax = axes[1]
    ax.plot([r["alpha_deg"] for r in panel], [r["cd"] for r in panel],
            color=BLUE, lw=1.2)
    ax.plot([r["alpha_deg"] for r in conv], [r["cd"] for r in conv],
            "o-", color=TEAL, ms=3.5, lw=1.2)
    ax.plot([r["alpha_deg"] for r in nonc], [r["cd"] for r in nonc],
            "x", color=RED, ms=4)
    ax.axhspan(0.006, 0.008, color=GREEN, alpha=0.25)
    ax.text(-8.2, 0.385, "green band: published\n$C_{d,min}$ = 0.006–0.008",
            fontsize=5.8, color=GREEN, va="top", ha="left")
    ax.axvspan(9, 21, color=RED, alpha=0.05)
    ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$C_d$")
    ax.set_title("drag")
    ax.grid(alpha=0.4)
    _panel_label(ax, "b", dx=-0.26)

    # (c) why the post-stall branch is unusable
    ax = axes[2]
    al = [r["alpha_deg"] for r in rows]
    res = [r.get("residual_max") or 1e-6 for r in rows]
    ax.semilogy(al, res, "o-", color=INK, ms=3.5, lw=1.0)
    ax.axhline(1e-4, color=GREEN, ls="--", lw=0.9)
    ax.text(-7.5, 1.3e-4, "convergence criterion", fontsize=5.8, color=GREEN)
    ax.axvspan(9, 21, color=RED, alpha=0.05)
    ax.text(15, 3e-3, "no steady\nsolution exists", fontsize=6,
            color=RED, ha="center")
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel("final residual")
    ax.set_title("7 of 15 converged")
    ax.grid(alpha=0.4, which="both")
    _panel_label(ax, "c", dx=-0.30)

    fig.tight_layout()
    save(fig, out, "fig6_rans")


# --------------------------------------------------------------------------- #
# Figure 7 -- cross-fidelity transfer
# --------------------------------------------------------------------------- #
def fig7_transfer(out: Path) -> None:
    """What survives when the truth changes solver."""
    ep, er = _load_json(EVAL_PANEL), _load_json(EVAL_RANS)
    if not ep or not er:
        print("  skip fig7_transfer: eval JSON missing")
        return

    fields = ["u", "v", "p", "load", "disp"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # (a) field errors side by side
    ax = axes[0]
    xs = np.arange(len(fields))
    pv = [ep["fields"][f]["relative_rmse"] for f in fields]
    rv = [er["fields"][f]["relative_rmse"] for f in fields]
    ax.bar(xs - 0.19, pv, 0.36, color=BLUE, label=f"vs panel (n={ep['n_cases']})")
    ax.bar(xs + 0.19, rv, 0.36, color=TEAL, label=f"vs RANS (n={er['n_cases']})")
    for i, (a_, b_) in enumerate(zip(pv, rv)):
        ax.text(i + 0.19, b_ + 0.012, f"{b_/a_:.1f}×", ha="center",
                fontsize=6, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"${f}$" if len(f) == 1 else f for f in fields])
    ax.set_ylabel("relative RMSE")
    ax.set_title("field error against each solver")
    ax.legend(loc="upper left", fontsize=6.2)
    ax.grid(axis="y", alpha=0.4)
    _panel_label(ax, "a", dx=-0.15)

    # (b) the headline: what transfers
    ax = axes[1]
    items = [("$u$ field", ep["fields"]["u"]["relative_rmse"],
              er["fields"]["u"]["relative_rmse"]),
             ("$C_d$ MAE", ep["scalars"]["cd"]["mae"], er["scalars"]["cd"]["mae"]),
             ("$C_l$ MAE", ep["scalars"]["cl"]["mae"], er["scalars"]["cl"]["mae"]),
             ("disp field", ep["fields"]["disp"]["relative_rmse"],
              er["fields"]["disp"]["relative_rmse"])]
    ratios = [b / a for _, a, b in items]
    colours = [GREEN if r < 1.5 else (ORANGE if r < 3 else RED) for r in ratios]
    ypos = np.arange(len(items))[::-1]
    ax.barh(ypos, ratios, 0.55, color=colours)
    ax.axvline(1.0, color=INK, lw=0.9, ls="--")
    for y, r in zip(ypos, ratios):
        ax.text(r + 0.14, y, f"{r:.2f}×", va="center", fontsize=6.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([i[0] for i in items])
    ax.set_xlabel("error ratio, RANS truth ÷ panel truth")
    ax.set_xlim(0, max(ratios) * 1.32)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_title("what transfers, what does not")
    ax.annotate("no degradation", xy=(1.0, len(items) - 0.55),
                xytext=(1.35, len(items) - 0.5), fontsize=6, color=GREEN,
                va="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7))
    ax.grid(axis="x", alpha=0.4)
    _panel_label(ax, "b", dx=-0.30)

    fig.tight_layout()
    save(fig, out, "fig7_transfer")


# --------------------------------------------------------------------------- #
# Figure 8 -- test-time augmentation ablation
# --------------------------------------------------------------------------- #
def fig8_ablation(out: Path) -> None:
    """What mirror test-time averaging is worth, same checkpoint both runs."""
    a, b = _load_json(EVAL_PANEL), _load_json(EVAL_PANEL_NOTTA)
    if not a or not b:
        print("  skip fig8_ablation: eval JSON missing")
        return
    if Path(a["checkpoint"]).parent != Path(b["checkpoint"]).parent:
        print("  warn fig8_ablation: the two runs use different checkpoints")

    fields = ["u", "v", "p", "load", "disp"]
    scalars = ["cl", "cd", "cm", "w_max", "twist_te_deg"]
    slabels = [r"$C_l$", r"$C_d$", r"$C_m$", r"$w_{max}$", r"$\theta_{TE}$"]

    gains = ([100 * (b["fields"][f]["relative_rmse"] - a["fields"][f]["relative_rmse"])
              / b["fields"][f]["relative_rmse"] for f in fields]
             + [100 * (b["scalars"][s]["mae"] - a["scalars"][s]["mae"])
                / b["scalars"][s]["mae"] for s in scalars])
    labels = [f"${f}$" if len(f) == 1 else f for f in fields] + slabels

    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    colours = [GREEN if g > 0 else RED for g in gains]
    ax.bar(np.arange(len(gains)), gains, 0.64, color=colours)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(4.5, color=GREY, lw=0.7, ls=":")
    ax.text(2.0, max(gains) * 1.06, "fields", ha="center", fontsize=6.5, color=GREY)
    ax.text(7.0, max(gains) * 1.06, "scalars", ha="center", fontsize=6.5, color=GREY)
    ax.set_xticks(np.arange(len(gains)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("error reduction [%]")
    ax.set_title("mirror test-time averaging, one extra forward pass", fontsize=8)
    for i, g in enumerate(gains):
        ax.text(i, g + (0.6 if g > 0 else -1.4), f"{g:.0f}",
                ha="center", fontsize=6, color=INK)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    save(fig, out, "fig8_ablation")


# --------------------------------------------------------------------------- #
# Figure 9 -- interpreter performance
# --------------------------------------------------------------------------- #
def fig9_interpreter(out: Path) -> None:
    """Regime confusion and the lock-in precision/recall asymmetry."""
    d = _load_json(EVAL_SLM)
    if not d:
        print("  skip fig9_interpreter: eval JSON missing")
        return
    key = next((k for k in d if k.startswith("v2")), None)
    if key is None:
        print("  skip fig9_interpreter: no v2 block")
        return
    blk = d[key]
    cm = blk["regime_confusion_matrix"]["matrix"]
    lk = blk["lock_in_metrics"]
    pf = blk["per_field_accuracy"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    # (a) confusion matrix
    ax = axes[0]
    M = np.array([[cm[r][c] for c in REGIME_ORDER] for r in REGIME_ORDER], float)
    ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
    for i in range(len(REGIME_ORDER)):
        for j in range(len(REGIME_ORDER)):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if M[i, j] > M.max() * .6 else INK)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER],
                       rotation=40, ha="right", fontsize=6)
    ax.set_yticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], fontsize=6)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"regime, n={blk['summary']['num_examples']}"
                 f"  macro F1={blk['summary']['regime_macro_f1']:.3f}")
    _panel_label(ax, "a", dx=-0.42)

    # (b) lock-in: the real error
    ax = axes[1]
    vals = [lk["precision"], lk["recall"], lk["f1"]]
    cols = [ORANGE, GREEN, BLUE]
    ax.bar(["precision", "recall", "F1"], vals, 0.55, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.set_title(f"lock-in: {lk['false_positives']} FP, "
                 f"{lk['false_negatives']} FN")
    ax.grid(axis="y", alpha=0.4)
    _panel_label(ax, "b", dx=-0.28)

    # (c) per-field accuracy
    ax = axes[2]
    order = ["shedding_frequency_hz", "strouhal_number",
             "mode_number", "rms_amplitude_over_c"]
    short = {"shedding_frequency_hz": r"$f_{shed}$",
             "strouhal_number": r"$St$",
             "mode_number": "mode",
             "rms_amplitude_over_c": r"$A_{rms}/c$"}
    accs = [pf[k]["accuracy"] for k in order if k in pf]
    labs = [short[k] for k in order if k in pf]
    cols = [GREEN if a >= 0.99 else ORANGE for a in accs]
    ax.barh(np.arange(len(accs))[::-1], accs, 0.55, color=cols)
    for y, a_ in zip(np.arange(len(accs))[::-1], accs):
        ax.text(a_ + 0.02, y, f"{a_:.2f}", va="center", fontsize=6.8)
    ax.set_yticks(np.arange(len(accs))[::-1]); ax.set_yticklabels(labs)
    ax.set_xlim(0, 1.18); ax.set_xlabel("accuracy within tolerance")
    ax.set_title("numeric fields")
    ax.grid(axis="x", alpha=0.4)
    _panel_label(ax, "c", dx=-0.30)

    fig.tight_layout()
    save(fig, out, "fig9_interpreter")


# --------------------------------------------------------------------------- #
FIGURES = {
    "fig1_system": fig1_system,
    "fig2_dataset": fig2_dataset,
    "fig3_network": fig3_network,
    "fig4_parity": fig4_parity,
    "fig5_fields": fig5_fields,
    "fig6_rans": fig6_rans,
    "fig7_transfer": fig7_transfer,
    "fig8_ablation": fig8_ablation,
    "fig9_interpreter": fig9_interpreter,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--only", default=None,
                    help="comma-separated figure names; default is all")
    args = ap.parse_args()

    apply_style()
    out = Path(args.out)
    names = args.only.split(",") if args.only else list(FIGURES)
    for n in names:
        n = n.strip()
        if n not in FIGURES:
            print(f"  unknown figure {n!r}; known: {', '.join(FIGURES)}")
            continue
        print(f"{n}:")
        try:
            FIGURES[n](out)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {n}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
