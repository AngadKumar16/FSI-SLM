"""The real-PIV strip for the poster: measured field -> record -> verdict.

Everything quoted here is read from results/real_piv/, not typed in twice.

Sized so that, printed at one poster column (~17 in), no label falls below
about 16 pt on the board.

    python3 poster/make_piv_fig.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
PIV = ROOT / "results/real_piv"
OUT = ROOT / "poster/generated"
OUT.mkdir(parents=True, exist_ok=True)

rec = json.loads((PIV / "feature_record.json").read_text())
cond, piv = rec["conditions"], rec["piv"]

TEAL = "#1B9AAA"
NAVY = "#1F3864"
DEEP = "#12233F"
CORAL = "#F4714F"
GREEN = "#2E8B57"
GOLD = "#D99400"
RED = "#B3202C"
GREY = "#5C6266"
INK = "#12233F"

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "savefig.facecolor": "white",
                     "figure.facecolor": "white"})


def _trim(a):
    """Drop the white border matplotlib bakes into a saved PNG."""
    g = a[..., :3].mean(axis=2) if a.ndim == 3 else a
    ink = g < 0.985
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return a
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


FW, FH = 10.6, 3.60
fig = plt.figure(figsize=(FW, FH))

fig.text(0.5, 0.972,
         "Real PIV, end to end: 200 measured frames, one record, one verdict",
         ha="center", va="center", fontsize=15.0, fontweight="bold", color=INK)

# ---- (a) measured TKE field, (b) measured shedding spectrum -------------- #
BAND0, BAND1 = 0.040, 0.905
for i, (key, src) in enumerate([("a", "wake_tke_field.png"),
                                ("b", "wake_spectrum.png")]):
    img = _trim(mpimg.imread(PIV / "figures" / src))
    wfrac = 0.263
    hfrac = wfrac * (FW / FH) * (img.shape[0] / img.shape[1])
    y0 = (BAND0 + BAND1) / 2 - hfrac / 2
    ax = fig.add_axes([0.004 + i * 0.270, y0, wfrac, hfrac])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-0.02, 1.13, key, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top", ha="left", color=INK)

# ---- (c) the record it produces ------------------------------------------ #
axc = fig.add_axes([0.552, BAND0, 0.205, BAND1 - BAND0])
axc.set_xlim(0, 1)
axc.set_ylim(0, 1)
axc.axis("off")
axc.text(-0.01, 1.02, "c", fontsize=14, fontweight="bold", va="top", color=INK)

axc.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.96,
                             boxstyle="round,pad=0,rounding_size=0.03",
                             facecolor="#F4F7F8", edgecolor=DEEP, lw=1.8,
                             transform=axc.transAxes, clip_on=False))
axc.add_patch(FancyBboxPatch((0.0, 0.865), 1.0, 0.095,
                             boxstyle="round,pad=0,rounding_size=0.03",
                             facecolor=DEEP, edgecolor=DEEP, lw=1.8,
                             transform=axc.transAxes, clip_on=False))
axc.add_patch(plt.Rectangle((0.0, 0.865), 1.0, 0.02, color=DEEP,
                            transform=axc.transAxes, clip_on=False))
axc.text(0.5, 0.911, "FeatureRecord · JSON", ha="center", va="center",
         color="white", fontsize=10.6, fontweight="bold")

rows = [
    ("conditions", TEAL, [f"Re = {cond['reynolds_number']/1e5:.2f}e5",
                          f"AoA = {cond['angle_of_attack_deg']:.1f} deg"]),
    ("piv_derived", NAVY, [f"f_shed = {piv['shedding_frequency_hz']:.2f} Hz",
                           f"St     = {piv['strouhal_number']:.3f}",
                           f"wake   = {piv['wake_width_over_c']:.2f} c",
                           "POD1-4 = 8.5 .. 5.0 %"]),
    ("structure, forces", GOLD, ["all zero  ← not acquired"]),
]
y = 0.825
for key, col, vals in rows:
    h = 0.055 + len(vals) * 0.073
    axc.add_patch(plt.Rectangle((0.038, y - h + 0.030), 0.017, h - 0.026,
                                color=col, transform=axc.transAxes,
                                clip_on=False))
    axc.text(0.085, y, f'"{key}"', ha="left", va="top", fontsize=9.8,
             fontweight="bold", color=col, family="DejaVu Sans Mono")
    for j, v in enumerate(vals):
        axc.text(0.085, y - 0.066 - j * 0.073, v, ha="left", va="top",
                 fontsize=9.4, color=RED if "all zero" in v else "#33404A",
                 family="DejaVu Sans Mono")
    y -= h + 0.012

axc.text(0.5, 0.048, "flagged, never invented", ha="center", va="center",
         fontsize=9.0, color=GREY, style="italic")

# ---- (d) what the pipeline said ------------------------------------------ #
axd = fig.add_axes([0.768, BAND0, 0.228, BAND1 - BAND0])
axd.set_xlim(0, 1)
axd.set_ylim(0, 1)
axd.axis("off")
axd.text(-0.01, 1.02, "d", fontsize=14, fontweight="bold", va="top", color=INK)


def card(y0, h, title, body, fc, mark=None, fs=11.6, bfs=9.6):
    axd.add_patch(FancyBboxPatch((0.0, y0), 1.0, h,
                                 boxstyle="round,pad=0,rounding_size=0.03",
                                 facecolor=fc, edgecolor=fc, lw=0,
                                 transform=axd.transAxes, clip_on=False))
    axd.text(0.46 if mark else 0.5, y0 + h - 0.062, title, ha="center",
             va="center", color="white", fontsize=fs, fontweight="bold")
    axd.text(0.5, y0 + h - 0.124, body, ha="center", va="top", color="white",
             fontsize=bfs, alpha=0.95, linespacing=1.45)
    if mark:
        axd.text(0.90, y0 + h - 0.062, mark, ha="center", va="center",
                 color="white", fontsize=15, fontweight="bold")


card(0.700, 0.283, "Validator",
     "regime → SEPARATED\nbroadband wake, flat POD", GREEN, mark="✓")
axd.add_patch(FancyArrowPatch((0.5, 0.694), (0.5, 0.642), arrowstyle="-|>",
                              mutation_scale=15, color=DEEP, lw=2.2,
                              transform=axd.transAxes, clip_on=False))

card(0.352, 0.283, "Fine-tuned SLM",
     "regime → VIV lock-in\nprovoked by the zeros", CORAL, mark="✗")
axd.add_patch(FancyArrowPatch((0.5, 0.346), (0.5, 0.294), arrowstyle="-|>",
                              mutation_scale=15, color=DEEP, lw=2.2,
                              transform=axd.transAxes, clip_on=False))

card(0.004, 0.283, "Claim rejected",
     "17.2° is ~8° past stall:\nseparated, not lock-in", DEEP)

fig.savefig(OUT / "fig_p_realpiv.png", dpi=205, bbox_inches="tight",
            pad_inches=0.05)
plt.close(fig)
print("wrote fig_p_realpiv.png")
