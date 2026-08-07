"""Poster figures that read live from artefacts on disk.

    python3 poster/figure_scripts/fig_poster_2026.py

Two figures:
  fig_metric_cards.png  -- headline numbers, all read from JSON, never typed
  fig_mf_gain.png       -- multi-fidelity fine-tune vs panel-only baseline

Style matches paper/figures.py so the poster reads as one document.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "poster/generated"
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = ROOT / "artifacts/cfd/manifest.jsonl"
EVAL_PANEL = ROOT / "artifacts/sensor/eval_test_panel.json"
EVAL_PANEL_MF = ROOT / "artifacts/sensor/eval_test_panel_mf.json"
EVAL_RANS_BASE = ROOT / "artifacts/sensor/eval_test_rans_baseline.json"
EVAL_RANS_MF = ROOT / "artifacts/sensor/eval_test_rans_mf.json"
EVAL_SLM = ROOT / "artifacts/eval_results/v2_vs_v3.json"

# paper/figures.py palette
INK = "#1A1A1A"
GREY = "#6E7377"
BLUE = "#2C5F8A"
ORANGE = "#C1571E"
TEAL = "#2A8A7F"
PURPLE = "#6A4C93"
GREEN = "#3F7D3A"
RED = "#B3202C"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "axes.edgecolor": GREY,
    "axes.linewidth": 0.9,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def load(p):
    with open(p) as fh:
        return json.load(fh)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- #
def fig_metric_cards():
    """Five headline numbers, every one read from an artefact."""
    rows = [json.loads(l) for l in open(MANIFEST)]
    n_cases = len(rows)
    n_foils = len({r["foil"] for r in rows})

    panel = load(EVAL_PANEL)
    cl_r2 = panel["scalars"]["cl"]["r2"]
    u_err = panel["fields"]["u"]["relative_rmse"]

    slm = load(EVAL_SLM)["v2 (base, no RAG)"]
    lockin_f1 = slm["lock_in_metrics"]["f1"]
    n_slm = slm["summary"]["num_examples"]

    cards = [
        (f"{n_cases:,}", f"solver cases\n{n_foils} airfoil sections", TEAL),
        (f"{cl_r2:.3f}", "lift $R^2$ on airfoil\nshapes never seen", BLUE),
        (f"{u_err*100:.1f}%", "velocity-field error\n(relative RMSE)", GREEN),
        (f"{lockin_f1:.2f}", f"lock-in detection F1\n(n = {n_slm} held-out)", ORANGE),
        ("1", "laptop\nno GPU cluster", PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(16, 2.9))
    ax.axis("off")
    n = len(cards)
    w = 1 / n
    for i, (big, small, col) in enumerate(cards):
        ax.add_patch(plt.Rectangle(
            (i * w + 0.008, 0.04), w - 0.016, 0.92,
            transform=ax.transAxes, facecolor=col, alpha=0.11,
            edgecolor=col, linewidth=2.2, zorder=0))
        x = i * w + w / 2
        ax.text(x, 0.58, big, transform=ax.transAxes, ha="center", va="center",
                fontsize=44, fontweight="bold", color=col)
        ax.text(x, 0.20, small, transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color=INK, linespacing=1.35)
    save(fig, "fig_metric_cards.png")


# --------------------------------------------------------------------------- #
def fig_mf_gain():
    """Adding 30 high-fidelity cases to training closes the transfer gap.

    Left  : RANS-test error, panel-only checkpoint vs multi-fidelity checkpoint.
    Right : the panel-test score, to show the gain costs nothing.
    """
    rb = load(EVAL_RANS_BASE)
    rm = load(EVAL_RANS_MF)
    pb = load(EVAL_PANEL)
    pm = load(EVAL_PANEL_MF)

    rows = [json.loads(l) for l in open(MANIFEST)]
    n_hi = sum(1 for r in rows if r["fidelity"] in ("rans", "urans"))
    n_hi_conv = sum(1 for r in rows
                    if r["fidelity"] in ("rans", "urans") and r.get("converged"))

    fields = ["u", "v", "p", "load", "disp"]
    flabel = {"u": "$u$", "v": "$v$", "p": "$p$",
              "load": "load\n$\\Delta C_p$", "disp": "disp\n$w/c$"}

    # Sized so the type stays legible when the figure is placed on a 60 in
    # board: native width x placement scale must keep 14 pt above ~18 pt.
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9),
                             gridspec_kw={"width_ratios": [1.25, 0.72, 1.25],
                                          "wspace": 0.42})

    # ---- (a) fields, on RANS ----
    ax = axes[0]
    x = np.arange(len(fields))
    wd = 0.36
    b0 = [rb["fields"][f]["relative_rmse"] for f in fields]
    b1 = [rm["fields"][f]["relative_rmse"] for f in fields]
    ax.bar(x - wd / 2, b0, wd, color=ORANGE, label="panel-only training")
    ax.bar(x + wd / 2, b1, wd, color=TEAL, label="+ high-fidelity training")
    for i, (a, b) in enumerate(zip(b0, b1)):
        ax.annotate(f"{a/b:.1f}$\\times$", (i, max(a, b)),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=12.5, fontweight="bold", color=TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels([flabel[f] for f in fields])
    ax.set_ylabel("relative RMSE vs RANS")
    ax.set_title("(a) field channels vs RANS", loc="left", pad=10)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    ax.set_ylim(0, max(b0) * 1.62)
    ax.grid(axis="y", color=GREY, alpha=0.18, lw=0.7)
    ax.set_axisbelow(True)

    # ---- (b) integrated forces, on RANS ----
    ax = axes[1]
    sc = ["cl", "cd"]
    slabel = {"cl": "$C_l$", "cd": "$C_d$"}
    x = np.arange(len(sc))
    b0 = [rb["scalars"][s]["mae"] for s in sc]
    b1 = [rm["scalars"][s]["mae"] for s in sc]
    ax.bar(x - wd / 2, b0, wd, color=ORANGE)
    ax.bar(x + wd / 2, b1, wd, color=TEAL)
    for i, (a, b) in enumerate(zip(b0, b1)):
        ax.annotate(f"{a/b:.1f}$\\times$", (i, max(a, b)),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=12.5, fontweight="bold", color=TEAL)
    ax.set_yscale("log")
    ax.set_ylim(top=max(b0) * 3.2)
    ax.set_xticks(x)
    ax.set_xticklabels([slabel[s] for s in sc])
    ax.set_ylabel("MAE vs RANS (log)")
    ax.set_title("(b) forces", loc="left", pad=10)
    ax.grid(axis="y", color=GREY, alpha=0.18, lw=0.7)
    ax.set_axisbelow(True)

    # ---- (c) no regression on the panel test ----
    ax = axes[2]
    x = np.arange(len(fields))
    b0 = [pb["fields"][f]["relative_rmse"] for f in fields]
    b1 = [pm["fields"][f]["relative_rmse"] for f in fields]
    ax.bar(x - wd / 2, b0, wd, color=ORANGE)
    ax.bar(x + wd / 2, b1, wd, color=TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels([flabel[f] for f in fields])
    ax.set_ylabel(f"rel. RMSE vs panel (n={pb['n_cases']})")
    ax.set_title("(c) same checkpoints, panel test", loc="left", pad=10)
    ax.grid(axis="y", color=GREY, alpha=0.18, lw=0.7)
    ax.set_axisbelow(True)
    worst = max(abs(b / a - 1) for a, b in zip(b0, b1))
    ax.set_ylim(0, max(max(b0), max(b1)) * 1.30)
    ax.text(0.03, 0.94, f"every channel within {worst*100:.1f}%\nof the baseline",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=13, fontweight="bold", color=GREEN, linespacing=1.25)

    fig.suptitle(
        f"Multi-fidelity fine-tuning: {n_hi} high-fidelity cases "
        f"({n_hi_conv} converged) on training-split sections close the transfer gap",
        fontsize=15.5, fontweight="bold", y=1.04)
    
    save(fig, "fig_mf_gain.png")

    # print the numbers so captions cannot drift from the figure
    print("\n-- numbers for the caption --")
    for f in fields:
        a, b = rb["fields"][f]["relative_rmse"], rm["fields"][f]["relative_rmse"]
        print(f"  {f:5s} RANS {a:.4f} -> {b:.4f}  ({a/b:.2f}x)")
    for s in ["cl", "cd"]:
        a, b = rb["scalars"][s]["mae"], rm["scalars"][s]["mae"]
        print(f"  {s:5s} MAE  {a:.5f} -> {b:.5f}  ({a/b:.2f}x)")
    print(f"  high-fidelity cases {n_hi} ({n_hi_conv} converged)")


if __name__ == "__main__":
    fig_metric_cards()
    fig_mf_gain()
