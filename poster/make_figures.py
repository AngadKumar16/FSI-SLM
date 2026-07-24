"""Regenerate the two poster figures from real data.

Usage:
    python poster/make_figures.py <solved_case_dir> [out_dir]

Figure 1 resamples an OpenFOAM solution (default: NACA 0009, a=8) onto a fine
local grid. Figure 2 reads the organized airfoil dataset in Data/Airfoils/{4,5,6}.
Run from the repo root.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.path import Path as MPath

from sensor.geometry import load_dat, rotate

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 220})


def fig1_flowfields(case_dir: str, out: str, foil_dat: str, alpha=8.0):
    import pyvista as pv
    open(os.path.join(case_dir, "case.foam"), "a").close()
    r = pv.OpenFOAMReader(os.path.join(case_dir, "case.foam"))
    r.set_active_time_value(r.time_values[-1]); r.cell_to_point_creation = True
    internal = r.read()["internalMesh"]

    x0, x1, y0, y1, nx, ny = -0.6, 2.0, -0.85, 0.85, 650, 430
    zc = float(internal.points[:, 2].mean())
    img = pv.ImageData(dimensions=(nx, ny, 1),
                       spacing=((x1 - x0) / (nx - 1), (y1 - y0) / (ny - 1), 1.0),
                       origin=(x0, y0, zc))
    s = img.sample(internal)
    U = np.asarray(s["U"]).reshape(ny, nx, 3); P = np.asarray(s["p"]).reshape(ny, nx)
    xs = np.linspace(x0, x1, nx); ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    foil = rotate(load_dat(foil_dat), alpha, (0.25, 0)).xy
    inside = MPath(foil).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    u, v = U[..., 0], U[..., 1]
    sp = np.where(inside, np.nan, np.hypot(u, v))
    Us, Vs = np.where(inside, np.nan, u), np.where(inside, np.nan, v)
    Cp = np.where(inside, np.nan, 2 * P)  # Cp = p/(0.5 U^2), U=1

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for cm, a in ((plt.cm.turbo, ax[0]), (plt.cm.RdBu_r, ax[1])):
        cm = cm.copy(); cm.set_bad("#ededed")
    cm0 = plt.cm.turbo.copy(); cm0.set_bad("#ededed")
    im0 = ax[0].pcolormesh(xs, ys, sp, cmap=cm0, shading="gouraud", vmin=0, vmax=1.5, rasterized=True)
    ax[0].streamplot(xs, ys, Us, Vs, density=1.5, color="white", linewidth=0.5, arrowsize=0.6)
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.02).set_label("|U| / U∞")
    ax[0].set_title("Velocity magnitude + streamlines", fontsize=13, weight="bold")
    cm1 = plt.cm.RdBu_r.copy(); cm1.set_bad("#ededed")
    im1 = ax[1].contourf(xs, ys, Cp, levels=np.linspace(-1.2, 1.0, 25), cmap=cm1, extend="both")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.02).set_label("Cₚ")
    ax[1].set_title("Pressure coefficient", fontsize=13, weight="bold")
    for a in ax:
        a.fill(foil[:, 0], foil[:, 1], facecolor="0.13", edgecolor="k", lw=0.9, zorder=6)
        a.set_xlim(x0, x1); a.set_ylim(y0, y1); a.set_aspect("equal"); a.set_xlabel("x/c")
    ax[0].set_ylabel("y/c")
    fig.suptitle("U-Net target: OpenFOAM RANS flow around NACA 0009      "
                 "(Re = 5×10⁵,   α = 8°,   Cₗ = 0.73)",
                 fontsize=14, weight="bold")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def fig2_gallery(out: str, root="Data/Airfoils"):
    series = {"4-digit": (f"{root}/4", "#2166ac"),
              "5-digit": (f"{root}/5", "#1a9850"),
              "6-series": (f"{root}/6", "#d6604d")}
    data = {k: (sorted(glob.glob(os.path.join(d, "*.dat.txt"))), c)
            for k, (d, c) in series.items()}
    tot = sum(len(v[0]) for v in data.values())
    ncol = 12
    counts = [len(data[k][0]) for k in data]
    rows = [int(np.ceil(c / ncol)) for c in counts]
    fig = plt.figure(figsize=(15, 9))
    outer = gridspec.GridSpec(len(data), 1, height_ratios=[r + 0.6 for r in rows], hspace=0.28)
    for si, (name, (fs, col)) in enumerate(data.items()):
        inner = gridspec.GridSpecFromSubplotSpec(rows[si], ncol, subplot_spec=outer[si],
                                                 wspace=0.15, hspace=0.35)
        for i, fp in enumerate(fs):
            ax = fig.add_subplot(inner[i // ncol, i % ncol])
            try:
                xy = load_dat(fp).xy
                ax.fill(xy[:, 0], xy[:, 1], color=col, lw=0)
                ax.plot(xy[:, 0], xy[:, 1], color="k", lw=0.4)
            except Exception:
                pass
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.25, 0.25); ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(os.path.basename(fp).replace(".dat.txt", "").replace("naca", "").lstrip("n"),
                         fontsize=5, pad=1)
        lab = fig.add_subplot(outer[si]); lab.axis("off")
        lab.text(-0.01, 1.02, f"{name}  (n={len(fs)})", transform=lab.transAxes,
                 fontsize=15, weight="bold", color=col, va="bottom")
    fig.suptitle(f"Airfoil dataset — {tot} NACA sections organized by series family",
                 fontsize=17, weight="bold", y=0.995)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "artifacts/cfd/cases/n0009sm_a8"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "poster"
    os.makedirs(out_dir, exist_ok=True)
    fig1_flowfields(case, os.path.join(out_dir, "poster_fig1_flowfields.png"),
                    "Data/Airfoils/4/n0009sm.dat.txt")
    fig2_gallery(os.path.join(out_dir, "poster_fig2_gallery.png"))
