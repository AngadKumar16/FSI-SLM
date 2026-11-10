"""Physics / data figures for the poster."""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
import style, joukowski, panel

style.apply()
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT = ROOT + "/poster/generated"
AIRFOILS = ROOT + "/Data/Airfoils"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p); plt.close(fig); print("wrote", name)


# ---------- 1. Flow field: speed + streamlines, and Cp field ----------
def fig_flowfield(alpha=8.0):
    f = joukowski.flow(alpha, res=560)
    c = f["chord"]; x0 = f["xf"].min()
    Xn = (f["X"] - x0)/c; Yn = f["Y"]/c
    xfn = (f["xf"] - x0)/c; yfn = f["yf"]/c
    xlim = ((f["xlim"][0]-x0)/c, (f["xlim"][1]-x0)/c)
    ylim = (f["ylim"][0]/c, f["ylim"][1]/c)
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.2), constrained_layout=True)
    pc = ax[0].pcolormesh(Xn, Yn, f["speed"], cmap="viridis",
                          vmin=0, vmax=1.9, shading="auto")
    ax[0].streamplot(Xn[0], Yn[:, 0], f["u"], f["v"], density=1.4,
                     color="white", linewidth=0.7, arrowsize=0.8, broken_streamlines=True)
    ax[0].fill(xfn, yfn, color=style.INK, zorder=5)
    cb = fig.colorbar(pc, ax=ax[0], fraction=0.05, pad=0.02)
    cb.set_label("Speed  |U| / U$_\\infty$")
    ax[0].set_title(f"Velocity field & streamlines  (α = {alpha:.0f}°)")
    pc2 = ax[1].pcolormesh(Xn, Yn, f["Cp"], cmap="RdBu",
                           vmin=-2.5, vmax=1.0, shading="auto")
    ax[1].contour(Xn, Yn, f["Cp"], levels=[-2,-1.5,-1,-.5,0,.5],
                  colors="k", linewidths=0.3, alpha=0.4)
    ax[1].fill(xfn, yfn, color=style.INK, zorder=5)
    cb2 = fig.colorbar(pc2, ax=ax[1], fraction=0.05, pad=0.02)
    cb2.set_label("Pressure coefficient  C$_p$")
    ax[1].set_title(f"Pressure field  (α = {alpha:.0f}°)")
    for a in ax:
        a.set_xlim(xlim); a.set_ylim(ylim); a.set_aspect("equal")
        a.set_xlabel("x / c"); a.set_ylabel("y / c")
    fig.suptitle("Inviscid flow around NACA 0009 (Joukowski solution)",
                 fontsize=20, fontweight="bold")
    save(fig, "fig_flowfield.png")


# ---------- 2. Surface Cp at several AoA ----------
def fig_cp_surface():
    fig, ax = plt.subplots(figsize=(8.4, 6))
    for al, c in [(0, style.TEAL), (4, style.GOLD), (8, style.CORAL), (12, style.RED)]:
        f = joukowski.flow(al, res=200)
        x = f["xs_surf"]; cp = f["Cps"]
        xn = (x - x.min())/(x.max()-x.min())
        ax.plot(xn, cp, color=c, lw=2.6, label=f"α = {al}°")
    ax.invert_yaxis()
    ax.axhline(0, color=style.GREY, lw=0.8, ls="--")
    ax.set_xlabel("x / c"); ax.set_ylabel("C$_p$")
    ax.set_title("Surface pressure vs. angle of attack")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    save(fig, "fig_cp_surface.png")


# ---------- 3. Cl-alpha curve with stall model ----------
def fig_cl_alpha():
    a = np.linspace(0, 20, 200)
    # thin-airfoil linear
    cl_lin = 2*np.pi*np.radians(a)
    # Joukowski (thickness corrected)
    cl_jouk = np.array([joukowski.flow(x, res=60)["Cl"] for x in np.linspace(0, 20, 40)])
    aj = np.linspace(0, 20, 40)
    # simple stall model: linear until ~11 deg then Viterna-like drop
    a_stall = 11.0
    cl_slope = 2*np.pi*np.pi/180
    cl_max = cl_slope*a_stall
    cl_model = np.where(a <= a_stall, cl_slope*a,
                        cl_max*np.cos(np.radians((a-a_stall))*3.2)*np.exp(-(a-a_stall)/22))
    fig, ax = plt.subplots(figsize=(8.4, 6))
    ax.plot(a, cl_lin, color=style.GREY, lw=1.8, ls="--", label="Thin-airfoil  2πα")
    ax.plot(aj, cl_jouk, "o", color=style.TEAL, ms=5, label="Joukowski (inviscid)")
    ax.plot(a, cl_model, color=style.CORAL, lw=3, label="With stall model")
    ax.axvspan(a_stall, 20, color=style.RED, alpha=0.07)
    ax.text(15.3, 0.35, "stall\nregion", color=style.RED, ha="center", fontsize=13, fontweight="bold")
    ax.set_xlabel("Angle of attack  α  (deg)")
    ax.set_ylabel("Lift coefficient  C$_l$")
    ax.set_title("Lift curve: what the U-Net learns to predict")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25); ax.set_xlim(0, 20); ax.set_ylim(0, 2.2)
    save(fig, "fig_cl_alpha.png")


# ---------- 4. Airfoil gallery ----------
def fig_gallery():
    series_dirs = {"4-digit": "4", "5-digit": "5", "6-series": "6"}
    colors = {"4-digit": style.TEAL, "5-digit": style.GOLD, "6-series": style.CORAL}
    files = []
    for lab, d in series_dirs.items():
        for fp in sorted(glob.glob(os.path.join(AIRFOILS, d, "*.dat*"))):
            files.append((lab, fp))
    # subsample to a tidy grid
    ncol, nrow = 12, 8
    n = min(len(files), ncol*nrow)
    idx = np.linspace(0, len(files)-1, n).astype(int)
    files = [files[i] for i in idx]
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 9))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (lab, fp) in zip(axes.ravel(), files):
        try:
            x, y = panel.load_dat(fp)
            if len(x) < 10:
                continue
            ax.fill(x, y, color=colors[lab], alpha=0.9, lw=0)
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.35, 0.35); ax.set_aspect("equal")
        except Exception:
            pass
    handles = [plt.Line2D([0], [0], marker="s", ls="", ms=13, color=c, label=l)
               for l, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=16, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Training shape library: {len(files)} NACA sections across 3 families",
                 fontsize=20, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.93, bottom=0.06, wspace=0.1, hspace=0.1)
    save(fig, "fig_gallery.png")


# ---------- 5. SDF + input channels ----------
def sdf_field(x, y, res=320):
    poly = np.column_stack([x, y])
    xs = np.linspace(-0.6, 1.6, res); ys = np.linspace(-0.7, 0.7, int(res*0.63))
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()])
    inside = MPath(poly).contains_points(P)
    # distance to edges
    a = poly[:-1]; b = poly[1:]
    ab = b - a; ab2 = (ab**2).sum(1) + 1e-30
    ap = P[:, None, :] - a[None]
    t = np.clip((ap*ab[None]).sum(2)/ab2[None], 0, 1)
    proj = a[None] + t[..., None]*ab[None]
    d = np.linalg.norm(P[:, None, :] - proj, axis=2).min(1)
    sdf = np.where(inside, -d, d).reshape(X.shape)
    return X, Y, sdf


def fig_sdf_channels(alpha=8.0, Re=5e5):
    x, y = panel.load_dat(os.path.join(AIRFOILS, "4", "n0009sm.dat.txt"))
    X, Y, sdf = sdf_field(x, y)
    a = np.radians(alpha)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.9), constrained_layout=True)
    # ch0 SDF
    pc = ax[0].pcolormesh(X, Y, sdf, cmap="RdBu_r", vmin=-0.4, vmax=0.4, shading="auto")
    ax[0].contour(X, Y, sdf, levels=[0], colors="k", linewidths=1.5)
    cb0 = fig.colorbar(pc, ax=ax[0], fraction=0.046, pad=0.02, location="bottom")
    cb0.set_label("distance (chords)", fontsize=12)
    ax[0].set_title("Ch 0: signed distance")
    # ch1 cos a
    ax[1].pcolormesh(X, Y, np.full_like(sdf, np.cos(a)), cmap="viridis", vmin=-1, vmax=1, shading="auto")
    ax[1].fill(x, y, color="k"); ax[1].set_title(f"Ch 1: cos α = {np.cos(a):.2f}")
    # ch2 sin a
    ax[2].pcolormesh(X, Y, np.full_like(sdf, np.sin(a)), cmap="viridis", vmin=-1, vmax=1, shading="auto")
    ax[2].fill(x, y, color="k"); ax[2].set_title(f"Ch 2: sin α = {np.sin(a):.2f}")
    # ch3 log Re
    lr = np.log10(Re)
    ax[3].pcolormesh(X, Y, np.full_like(sdf, lr), cmap="magma", vmin=4, vmax=7, shading="auto")
    ax[3].fill(x, y, color="w"); ax[3].set_title(f"Ch 3: log$_{{10}}$Re = {lr:.1f}")
    for a_ in ax:
        a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("U-Net input tensor: geometry as a signed-distance field + flow-condition channels",
                 fontsize=18, fontweight="bold")
    save(fig, "fig_sdf_channels.png")


# ---------- 6. VIV lock-in explainer ----------
def fig_viv_lockin():
    Ur = np.linspace(2, 12, 400)
    St = 0.2
    f_shed = St*Ur                      # natural shedding grows with velocity
    f_n = 1.0                            # structural natural freq (normalized)
    lock_lo, lock_hi = 4.2, 7.0
    f_dom = f_shed.copy()
    mask = (Ur > lock_lo) & (Ur < lock_hi)
    f_dom[mask] = f_n
    fig, ax = plt.subplots(figsize=(8.6, 6))
    ax.plot(Ur, f_shed, color=style.GREY, ls="--", lw=2, label="Vortex shedding  f = St·U*")
    ax.axhline(f_n, color=style.BLUE, ls=":", lw=2, label="Structure  f$_n$")
    ax.plot(Ur, f_dom, color=style.CORAL, lw=3.4, label="Response frequency")
    ax.axvspan(lock_lo, lock_hi, color=style.CORAL, alpha=0.12)
    ax.text((lock_lo+lock_hi)/2, 0.35, "LOCK-IN\nwindow", color=style.CORAL,
            ha="center", fontweight="bold", fontsize=14)
    ax.set_xlabel("Reduced velocity  U*"); ax.set_ylabel("Frequency (normalized)")
    ax.set_title("VIV lock-in: shedding synchronizes to the structure")
    ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.25)
    ax.set_ylim(0, 2.4)
    save(fig, "fig_viv_lockin.png")


# ---------- 7. Dataset scatter: St vs Re by regime ----------
def fig_dataset_scatter():
    import json
    recs = []
    for split in ("train", "val", "test"):
        p = f"{ROOT}/artifacts/dataset/feature_records_{split}.jsonl"
        with open(p) as fh:
            for line in fh:
                recs.append(json.loads(line))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    for r in recs:
        reg = r.get("ground_truth_regime")
        c = style.REGIME_COLORS.get(reg, style.GREY)
        Re = r["conditions"]["reynolds_number"]; St = r["piv"]["strouhal_number"]
        aoa = r["conditions"]["angle_of_attack_deg"]; Ur = r["conditions"]["reduced_velocity"]
        ax[0].scatter(Re, St, color=c, s=26, alpha=0.65, edgecolor="none")
        ax[1].scatter(Ur, aoa, color=c, s=26, alpha=0.65, edgecolor="none")
    ax[0].set_xscale("log"); ax[0].set_xlabel("Reynolds number"); ax[0].set_ylabel("Strouhal number  St")
    ax[0].set_title("Flow regimes in (Re, St) space")
    ax[1].set_xlabel("Reduced velocity  U*"); ax[1].set_ylabel("Angle of attack (deg)")
    ax[1].set_title("Flow regimes in (U*, α) space")
    for a in ax:
        a.grid(alpha=0.22)
    handles = [plt.Line2D([0],[0], marker="o", ls="", ms=11, color=c,
               label=style.REGIME_LABELS[k]) for k, c in style.REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Synthetic training set: 5 physically-coupled FSI regimes",
                 fontsize=20, fontweight="bold", y=1.05)
    fig.subplots_adjust(bottom=0.14, top=0.86)
    save(fig, "fig_dataset_scatter.png")


if __name__ == "__main__":
    fig_flowfield()
    fig_cp_surface()
    fig_cl_alpha()
    fig_gallery()
    fig_sdf_channels()
    fig_viv_lockin()
    fig_dataset_scatter()
    print("science figures done")
