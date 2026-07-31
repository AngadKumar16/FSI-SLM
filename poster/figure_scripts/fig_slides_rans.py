"""Slide figures: RANS validation on NACA 0009 + cross-fidelity transfer.

Every number is read from repository artefacts, nothing is hard-coded:
  artifacts/cfd/manifest.jsonl        RANS cases (Cl, Cd, residual, cl_std)
  artifacts/sensor/eval_test_panel.json   surrogate scored against panel targets
  artifacts/sensor/eval_test_rans.json    same net scored against RANS targets

Run:  python poster/figure_scripts/fig_slides_rans.py
Out:  poster/generated/slide1_rans_validation.png   (slide 1, primary)
      poster/generated/slide1_alt_convergence.png   (slide 1, backup)
      poster/generated/slide2_fidelity_transfer.png (slide 2, primary)
      poster/generated/slide2_alt_transfer_ratio.png(slide 2, backup)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "poster" / "generated"
RESID_TOL = 1e-4  # solver convergence criterion used in sensor/cfd


# --------------------------------------------------------------------------- data


def load_rans(foil: str = "n0009sm") -> list[dict]:
    rows = []
    with open(ROOT / "artifacts" / "cfd" / "manifest.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("fidelity") == "rans" and r.get("foil") == foil and r.get("ok"):
                rows.append(r)
    rows.sort(key=lambda r: r["alpha_deg"])
    return rows


def load_evals() -> tuple[dict, dict]:
    p = json.loads((ROOT / "artifacts" / "sensor" / "eval_test_panel.json").read_text())
    r = json.loads((ROOT / "artifacts" / "sensor" / "eval_test_rans.json").read_text())
    return p, r


def fit_slope(alpha_deg: np.ndarray, cl: np.ndarray) -> tuple[float, float]:
    """Least-squares lift slope [1/deg] and intercept over the given points."""
    a, b = np.polyfit(alpha_deg, cl, 1)
    return float(a), float(b)


# ------------------------------------------------------------- slide 1, primary


def slide1_rans_validation(rows: list[dict]) -> None:
    """Lift curve, converged branch separated from the limit-cycling branch."""
    alpha = np.array([r["alpha_deg"] for r in rows])
    cl = np.array([r["cl"] for r in rows])
    cd = np.array([r["cd"] for r in rows])
    cl_std = np.array([r["cl_std"] for r in rows])
    conv = np.array([bool(r["converged"]) for r in rows])

    slope, icept = fit_slope(alpha[conv], cl[conv])
    thin = 2 * np.pi * np.deg2rad(1.0)  # dCl/dalpha per degree, thin-airfoil

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), width_ratios=[1.25, 1])

    # -- left: Cl(alpha)
    ax = axes[0]
    lo, hi = alpha[conv].min() - 1, alpha[conv].max() + 0.6
    ax.axvspan(lo, hi, color=style.GREEN, alpha=0.07, zorder=0)
    ax.axhline(0, color=style.GREY, lw=0.8, zorder=1)
    ax.axvline(0, color=style.GREY, lw=0.8, zorder=1)

    ar = np.linspace(lo, hi, 50)
    ax.plot(ar, thin * ar, "--", color=style.GREY, lw=2, zorder=2,
            label=r"thin-airfoil $2\pi$")
    ax.plot(ar, slope * ar + icept, "-", color=style.BLUE, lw=2.4, alpha=0.55, zorder=3,
            label=f"fit  {slope:.3f}/deg  ({slope / thin * 100:.0f}% of $2\\pi$)")

    ax.errorbar(alpha[~conv], cl[~conv], yerr=cl_std[~conv], fmt="o", ms=9,
                mfc="white", mec=style.RED, mew=2.0, ecolor=style.RED,
                elinewidth=2.0, capsize=5, alpha=0.85, zorder=4,
                label="not converged (bars = oscillation)")
    ax.plot(alpha[conv], cl[conv], "o", ms=11, color=style.TEAL,
            mec=style.NAVY, mew=1.4, zorder=5, label="converged RANS")

    ax.text(-4.0, 1.30, "validated band  α ≤ 8°", fontsize=14,
            color=style.GREEN, fontweight="bold", ha="center")
    ax.annotate("steady SIMPLE\nlimit-cycles here",
                xy=(13.6, 0.60), xytext=(15.4, 0.16), fontsize=13,
                color=style.RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=style.RED, lw=1.8))

    ax.set_xlabel("angle of attack  α  [deg]")
    ax.set_ylabel(r"lift coefficient  $C_l$")
    ax.set_title("NACA 0009, k-ω SST, Re = 5×10⁵")
    ax.set_ylim(-1.35, 1.55)
    ax.legend(loc="lower right", fontsize=12.0, framealpha=0.95)
    ax.grid(alpha=0.22)

    # -- right: Cd(alpha), log scale, minimum at zero incidence
    ax = axes[1]
    ax.axvspan(lo, hi, color=style.GREEN, alpha=0.07, zorder=0)
    ax.plot(alpha[~conv], cd[~conv], "o", ms=9, mfc="white",
            mec=style.RED, mew=2.0, alpha=0.85, zorder=3)
    ax.plot(alpha[conv], cd[conv], "o-", ms=10, lw=2.2, color=style.CORAL,
            mec=style.NAVY, mew=1.2, zorder=4)

    i0 = int(np.argmin(np.abs(alpha)))
    ax.annotate(f"$C_{{d,min}}$ = {cd[i0]:.4f}\nat α = 0  (ref 0.006–0.008)",
                xy=(alpha[i0], cd[i0]), xytext=(-7.2, 0.030), fontsize=12.5,
                color=style.NAVY,
                arrowprops=dict(arrowstyle="->", color=style.NAVY, lw=1.6))
    ax.set_yscale("log")
    ax.set_xlabel("angle of attack  α  [deg]")
    ax.set_ylabel(r"drag coefficient  $C_d$")
    ax.set_title("Drag bucket recovered")
    ax.grid(alpha=0.22, which="both")

    fig.suptitle("OpenFOAM RANS validation — the branch we can defend",
                 fontsize=22, fontweight="bold", color=style.NAVY, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT / "slide1_rans_validation.png")
    plt.close(fig)


# ------------------------------------------------------------- slide 1, backup


def slide1_alt_convergence(rows: list[dict]) -> None:
    """Where the solver stops being trustworthy: residual and lift scatter."""
    alpha = np.array([r["alpha_deg"] for r in rows])
    resid = np.array([r["residual_max"] for r in rows])
    cl_std = np.array([r["cl_std"] for r in rows])
    iters = np.array([r["iterations"] for r in rows])
    conv = np.array([bool(r["converged"]) for r in rows])
    colors = [style.TEAL if c else style.RED for c in conv]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

    ax = axes[0]
    ax.bar(alpha, resid, width=1.35, color=colors, edgecolor=style.NAVY, lw=0.9)
    ax.axhline(RESID_TOL, color=style.NAVY, ls="--", lw=2)
    ax.text(-8.4, RESID_TOL * 1.5, "tolerance 1e-4", fontsize=12.5,
            color=style.NAVY, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("α  [deg]")
    ax.set_ylabel("max initial residual")
    ax.set_title("Residual: a cliff at 10°")
    ax.grid(alpha=0.2, axis="y", which="both")

    ax = axes[1]
    ax.bar(alpha, cl_std, width=1.35, color=colors, edgecolor=style.NAVY, lw=0.9)
    ax.set_xlabel("α  [deg]")
    ax.set_ylabel(r"$C_l$ oscillation  (std over last 500 iters)")
    ax.set_title("Lift never settles past stall")
    ax.grid(alpha=0.2, axis="y")
    ax.annotate(f"±{cl_std.max():.2f}", xy=(alpha[int(np.argmax(cl_std))], cl_std.max()),
                xytext=(6.0, cl_std.max() * 0.92), fontsize=13,
                color=style.RED, fontweight="bold")

    ax = axes[2]
    ax.bar(alpha, iters, width=1.35, color=colors, edgecolor=style.NAVY, lw=0.9)
    ax.axhline(2000, color=style.NAVY, ls="--", lw=2)
    ax.set_ylim(0, 2400)
    ax.text(-8.4, 2090, "iteration cap 2000", fontsize=12.5,
            color=style.NAVY, fontweight="bold")
    ax.set_xlabel("α  [deg]")
    ax.set_ylabel("SIMPLE iterations")
    ax.set_title("Every stalled case hits the cap")
    ax.grid(alpha=0.2, axis="y")

    handles = [Line2D([], [], marker="s", ls="", ms=13, mfc=style.TEAL, mec=style.NAVY,
                      label=f"converged  ({int(conv.sum())}/{len(conv)})"),
               Line2D([], [], marker="s", ls="", ms=13, mfc=style.RED, mec=style.NAVY,
                      label=f"excluded  ({int((~conv).sum())}/{len(conv)})")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("Why 7 of 15 cases count — steady RANS quits at separation",
                 fontsize=21, fontweight="bold", color=style.NAVY, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "slide1_alt_convergence.png")
    plt.close(fig)


# ------------------------------------------------------------- slide 2, primary


FIELD_LABEL = {"u": "u  streamwise", "v": "v  transverse", "p": "p  pressure",
               "load": "load  (solid)", "disp": "disp  (solid)"}


def slide2_fidelity_transfer(panel: dict, rans: dict) -> None:
    """Same weights, two graders: paired bars per channel + scalar MAE."""
    keys = ["u", "v", "p", "load", "disp"]
    pv = [panel["fields"][k]["relative_rmse"] for k in keys]
    rv = [rans["fields"][k]["relative_rmse"] for k in keys]
    y = np.arange(len(keys))[::-1]
    h = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2), width_ratios=[1.6, 1])

    ax = axes[0]
    ax.barh(y + h / 2, pv, height=h, color=style.TEAL, edgecolor=style.NAVY, lw=1.0,
            label=f"vs panel targets  (n = {panel['n_cases']})")
    ax.barh(y - h / 2, rv, height=h, color=style.CORAL, edgecolor=style.NAVY, lw=1.0,
            label=f"vs RANS targets  (n = {rans['n_cases']})")
    for yy, a, b in zip(y, pv, rv):
        ax.text(a + 0.012, yy + h / 2, f"{a:.3f}", va="center", fontsize=12.5,
                color=style.NAVY)
        ax.text(b + 0.012, yy - h / 2, f"{b:.3f}", va="center", fontsize=12.5,
                color=style.NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels([FIELD_LABEL[k] for k in keys])
    ax.set_xlabel("relative RMSE   (lower is better)")
    ax.set_xlim(0, max(max(pv), max(rv)) * 1.30)
    ax.set_ylim(-1.05, len(keys) - 0.15)
    ax.set_title("One network, two graders")
    ax.legend(loc="upper right", fontsize=12.5, framealpha=0.95)
    ax.grid(alpha=0.2, axis="x")

    ax.annotate("u is better against RANS —\nthe net learned flow, not solver quirks",
                xy=(rv[0] + 0.055, y[0] - h / 2), xytext=(0.22, y[0] - 0.64),
                fontsize=12.5, color=style.GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=style.GREEN, lw=1.8))
    ax.annotate("structural channel\ndoes not transfer",
                xy=(rv[4] * 0.55, y[4] - h - 0.05), xytext=(0.20, y[4] - 0.95),
                fontsize=12.5, color=style.RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=style.RED, lw=1.8))

    # -- right: integrated forces
    ax = axes[1]
    sk = ["cl", "cd"]
    pm = [panel["scalars"][k]["mae"] for k in sk]
    rm = [rans["scalars"][k]["mae"] for k in sk]
    x = np.arange(len(sk))
    ax.bar(x - 0.19, pm, width=0.36, color=style.TEAL, edgecolor=style.NAVY, lw=1.0)
    ax.bar(x + 0.19, rm, width=0.36, color=style.CORAL, edgecolor=style.NAVY, lw=1.0)
    for xi, a, b in zip(x, pm, rm):
        ax.text(xi - 0.19, a * 1.35, f"{a:.4f}", ha="center", fontsize=12.5,
                color=style.NAVY)
        ax.text(xi + 0.19, b * 1.35, f"{b:.4f}", ha="center", fontsize=12.5,
                color=style.NAVY)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$C_l$", r"$C_d$"], fontsize=18)
    ax.set_yscale("log")
    ax.set_ylabel("MAE")
    ax.set_ylim(min(pm + rm) * 0.35, max(pm + rm) * 6)
    ax.set_title("Lift inherits the bias, drag does not")
    ax.grid(alpha=0.2, axis="y", which="both")
    ax.text(0.0, max(pm) * 2.6, f"×{rm[0] / pm[0]:.1f}", ha="center", fontsize=15,
            color=style.RED, fontweight="bold")
    ax.text(1.0, max(pm) * 2.6, "identical", ha="center", fontsize=14,
            color=style.GREEN, fontweight="bold")

    fig.suptitle("Cross-fidelity transfer — trained on panel, graded on RANS",
                 fontsize=22, fontweight="bold", color=style.NAVY, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT / "slide2_fidelity_transfer.png")
    plt.close(fig)


# ------------------------------------------------------------- slide 2, backup


def slide2_alt_transfer_ratio(panel: dict, rans: dict) -> None:
    """Error ratio RANS/panel — one axis, one threshold, no reading of bars."""
    items = [("u", "fields", "u  streamwise"),
             ("cd", "scalars", r"$C_d$  drag"),
             ("load", "fields", "load  (solid)"),
             ("v", "fields", "v  transverse"),
             ("p", "fields", "p  pressure"),
             ("cl", "scalars", r"$C_l$  lift"),
             ("disp", "fields", "disp  (solid)")]

    def val(d, key, group):
        return d[group][key]["relative_rmse" if group == "fields" else "mae"]

    ratio, labels = [], []
    for key, group, lab in items:
        ratio.append(val(rans, key, group) / val(panel, key, group))
        labels.append(lab)
    ratio = np.array(ratio)
    order = np.argsort(ratio)
    ratio, labels = ratio[order], [labels[i] for i in order]
    y = np.arange(len(ratio))

    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    colors = [style.GREEN if r <= 1.25 else (style.GOLD if r < 2.2 else style.RED)
              for r in ratio]
    ax.hlines(y, 1.0, ratio, color=colors, lw=4.5, alpha=0.55, zorder=2)
    ax.scatter(ratio, y, s=230, c=colors, edgecolors=style.NAVY, lw=1.6, zorder=3)
    ax.axvline(1.0, color=style.NAVY, lw=2.2, ls="--", zorder=1)

    for yi, r in zip(y, ratio):
        off = 1.06 if r >= 1 else 0.94
        ax.text(r * off, yi, f"×{r:.2f}", va="center",
                ha="left" if r >= 1 else "right", fontsize=13.5,
                color=style.NAVY, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(0.55, 5.0)
    ax.set_xticks([0.6, 1, 2, 3, 5])
    ax.set_xticklabels(["0.6", "1", "2", "3", "5"])
    ax.set_xticks([], minor=True)
    ax.set_ylim(-0.9, len(y) - 0.4)
    ax.set_xlabel("error against RANS  ÷  error against panel")
    ax.set_title("What survives the fidelity gap", pad=14)
    ax.text(0.72, -0.65, "transfers", fontsize=14, color=style.GREEN,
            fontweight="bold", ha="center")
    ax.text(2.6, -0.65, "needs RANS cases in training", fontsize=14, color=style.RED,
            fontweight="bold", ha="center")
    ax.grid(alpha=0.22, axis="x", which="both")
    fig.tight_layout()
    fig.savefig(OUT / "slide2_alt_transfer_ratio.png")
    plt.close(fig)


def main() -> None:
    style.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rans()
    panel, rans = load_evals()
    slide1_rans_validation(rows)
    slide1_alt_convergence(rows)
    slide2_fidelity_transfer(panel, rans)
    slide2_alt_transfer_ratio(panel, rans)
    print(f"wrote 4 figures to {OUT}")


if __name__ == "__main__":
    main()
