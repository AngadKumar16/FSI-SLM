"""Alternative slide figures — same two claims, told with flow fields.

Slide 1 v2: why the solver stops converging, shown as separated flow.
Slide 2 v2: the panel/RANS disagreement, shown as a field difference.

Fields come from `artifacts/cfd/npz/*.npz` (RANS, already solved). The panel
comparison case is re-solved live through `sensor.simulate.simulate_case`, which
takes about a second and needs numpy only.

Run:  python poster/figure_scripts/fig_slides_rans_v2.py
Out:  poster/generated/slide1_v2_separation.png
      poster/generated/slide2_v2_solver_gap.png
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import style

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "poster" / "generated"
sys.path.insert(0, str(ROOT))

FOIL = "n0009sm"
DAT = "Data/Airfoils/4/n0009sm.dat.txt"
XLIM, YLIM = (-0.6, 2.4), (-1.5, 1.5)  # sensor.sdf.Grid default
ZOOM_X, ZOOM_Y = (-0.45, 1.75), (-0.75, 0.75)
EXTENT = [XLIM[0], XLIM[1], YLIM[0], YLIM[1]]
U, V, P = 0, 1, 2


def load_npz(alpha: int) -> dict:
    tag = f"a{alpha}" if alpha != 0 else "a0"
    d = np.load(ROOT / "artifacts" / "cfd" / "npz" / f"{FOIL}_{tag}.npz", allow_pickle=True)
    return {"target": np.asarray(d["target"], dtype=np.float32),
            "solid": np.asarray(d["solid_mask"]),
            "meta": json.loads(str(d["meta"]))}


def rans_rows() -> list[dict]:
    rows = []
    with open(ROOT / "artifacts" / "cfd" / "manifest.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("fidelity") == "rans" and r.get("foil") == FOIL and r.get("ok"):
                rows.append(r)
    rows.sort(key=lambda r: r["alpha_deg"])
    return rows


def draw_solid(ax, solid) -> None:
    ax.contourf(np.linspace(*XLIM, solid.shape[1]), np.linspace(*YLIM, solid.shape[0]),
                solid, levels=[0.5, 1.5], colors=[style.INK], zorder=5)
    ax.contour(np.linspace(*XLIM, solid.shape[1]), np.linspace(*YLIM, solid.shape[0]),
               solid, levels=[0.5], colors=[style.INK], linewidths=1.4, zorder=6)


def field_ax(ax, data, solid, cmap, vmin, vmax, title):
    im = ax.imshow(data, origin="lower", extent=EXTENT, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal", interpolation="bilinear")
    draw_solid(ax, solid)
    ax.set_xlim(*ZOOM_X)
    ax.set_ylim(*ZOOM_Y)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=17)
    return im


# --------------------------------------------------------------- slide 1 v2


def slide1_v2_separation() -> None:
    """Attached flow converges; separated flow has no steady state to find."""
    good, bad = load_npz(8), load_npz(14)
    rows = rans_rows()

    fig = plt.figure(figsize=(16.0, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.22)

    vmin, vmax = 0.0, 1.5
    ax0 = fig.add_subplot(gs[0, 0])
    im = field_ax(ax0, good["target"][U], good["solid"], "RdYlBu_r", vmin, vmax,
                  "α = 8°  ·  converged")
    ax1 = fig.add_subplot(gs[0, 1])
    field_ax(ax1, bad["target"][U], bad["solid"], "RdYlBu_r", vmin, vmax,
             "α = 14°  ·  never converged")

    # outline the reverse-flow (separated) region on both
    xs = np.linspace(*XLIM, 128)
    ys = np.linspace(*YLIM, 128)
    for ax, case in ((ax0, good), (ax1, bad)):
        rev = (case["target"][U] < 0.02) & (case["solid"] == 0)
        ax.contour(xs, ys, rev.astype(float), levels=[0.5],
                   colors=[style.PURPLE], linewidths=2.6, zorder=7)

    ax0.text(-0.40, -0.68, "flow stays glued to the foil", fontsize=13.5,
             color=style.NAVY, fontweight="bold")
    ax1.text(-0.40, -0.68, "big stalled wake, shedding, never settles",
             fontsize=13.5, color=style.RED, fontweight="bold")

    cax = fig.add_axes([0.085, 0.06, 0.545, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("streamwise velocity  u / U∞", fontsize=14)

    # right: separated area fraction against angle
    ax2 = fig.add_subplot(gs[0, 2])
    alphas, frac, conv = [], [], []
    for r in rows:
        c = load_npz(int(r["alpha_deg"]))
        fluid = c["solid"] == 0
        rev = (c["target"][U] < 0.02) & fluid
        alphas.append(r["alpha_deg"])
        frac.append(100.0 * rev.sum() / fluid.sum())
        conv.append(bool(r["converged"]))
    alphas, frac = np.array(alphas), np.array(frac)
    conv = np.array(conv)
    ax2.bar(alphas[conv], frac[conv], width=1.4, color=style.TEAL,
            edgecolor=style.NAVY, lw=0.9)
    ax2.bar(alphas[~conv], frac[~conv], width=1.4, color=style.RED,
            edgecolor=style.NAVY, lw=0.9)
    ax2.set_xlabel("angle of attack  α  [deg]")
    ax2.set_ylabel("area of reversed flow  [% of domain]")
    ax2.set_title("Separation grows, then the solver quits", fontsize=17)
    ax2.grid(alpha=0.22, axis="y")
    ax2.legend(handles=[Patch(facecolor=style.TEAL, edgecolor=style.NAVY,
                              label="converged (7)"),
                        Patch(facecolor=style.RED, edgecolor=style.NAVY,
                              label="not converged (8)")],
               loc="upper left", fontsize=12.5)

    fig.suptitle("Convergence fails exactly where the flow separates",
                 fontsize=22, fontweight="bold", color=style.NAVY, y=0.99)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.86, bottom=0.16)
    fig.savefig(OUT / "slide1_v2_separation.png")
    plt.close(fig)


# --------------------------------------------------------------- slide 2 v2


def slide2_v2_solver_gap() -> None:
    """Where the cheap solver and the expensive one disagree, and by how much."""
    from sensor.geometry import load_dat
    from sensor.simulate import simulate_case

    rans = load_npz(8)
    panel = simulate_case(load_dat(ROOT / DAT), 8.0)
    pt = np.asarray(panel["target"], dtype=np.float32)
    rt = rans["target"]
    solid = rans["solid"]
    cl_p = float(panel["coeffs"]["cl"])
    cl_r = float(rans["meta"]["coeffs"]["cl"])

    # Panel boxes are sized to the zoom aspect ratio so an equal-aspect image
    # fills its box exactly; the colour bars can then be pinned to the box edge
    # and come out the same height as the image.
    ratio = (ZOOM_Y[1] - ZOOM_Y[0]) / (ZOOM_X[1] - ZOOM_X[0])
    pw, ph = 4.35, 4.35 * ratio
    fig = plt.figure(figsize=(3 * pw + 1.9, 2 * ph + 1.35))

    left, bottom = 0.020, 0.045
    top_pad, row_gap = 0.115, 0.085
    w = pw / fig.get_figwidth()
    h = ph / fig.get_figheight()
    col_x = [left, left + w + 0.075, left + 2 * w + 0.235]
    row_y = [1 - top_pad - h, 1 - top_pad - 2 * h - row_gap]

    def bar(ax, im, pad=0.012, width=0.013):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=12)
        cb.outline.set_edgecolor(style.GREY)
        return cb

    axes = np.empty((2, 3), dtype=object)
    for row, (ch, name, cmap) in enumerate(((U, "u / U∞", "RdYlBu_r"),
                                            (P, "pressure  Cp", "RdBu_r"))):
        a, b = pt[ch], rt[ch]
        if ch == U:
            vmin, vmax = 0.0, 1.5
        else:
            lim = float(np.percentile(np.abs(np.concatenate([a.ravel(), b.ravel()])), 99))
            vmin, vmax = -lim, lim
        for col in range(3):
            axes[row, col] = fig.add_axes([col_x[col], row_y[row], w, h])

        im = field_ax(axes[row, 0], a, solid, cmap, vmin, vmax,
                      f"panel method  ·  {name}")
        field_ax(axes[row, 1], b, solid, cmap, vmin, vmax, f"RANS  ·  {name}")
        bar(axes[row, 1], im).set_label(name, fontsize=13.5, labelpad=8)

        d = b - a
        dlim = float(np.percentile(np.abs(d[solid == 0]), 99))
        imd = field_ax(axes[row, 2], d, solid, "PuOr_r", -dlim, dlim,
                       "difference  (RANS − panel)")
        bar(axes[row, 2], imd).set_label(f"Δ {name}", fontsize=13.5, labelpad=8)

    axes[0, 0].text(-0.40, 0.60, f"$C_l$ = {cl_p:.2f}", fontsize=16,
                    fontweight="bold", color=style.NAVY)
    axes[0, 1].text(-0.40, 0.60, f"$C_l$ = {cl_r:.2f}", fontsize=16,
                    fontweight="bold", color=style.NAVY)
    axes[0, 2].text(-0.40, 0.58,
                    f"panel runs {100 * (cl_p - cl_r) / cl_r:+.0f}% high on lift",
                    fontsize=14.5, fontweight="bold", color=style.RED)
    axes[1, 2].text(-0.40, -0.68,
                    "disagreement sits on the suction side\n"
                    "— which is what sets lift and load",
                    fontsize=13.5, color=style.RED, fontweight="bold")

    fig.suptitle("NACA 0009 at α = 8°: the same case, two solvers",
                 fontsize=22, fontweight="bold", color=style.NAVY, y=0.985)
    fig.savefig(OUT / "slide2_v2_solver_gap.png", bbox_inches=None)
    plt.close(fig)


def main() -> None:
    style.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    slide1_v2_separation()
    slide2_v2_solver_gap()
    print(f"wrote 2 figures to {OUT}")


if __name__ == "__main__":
    main()
