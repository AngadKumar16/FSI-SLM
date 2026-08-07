"""System / architecture diagrams for the poster."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Polygon
import style
from diagram_helpers import box, arrow, label, new
style.apply()
OUT = "/sessions/peaceful-focused-hamilton/mnt/FSI-SLM/poster/generated"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("wrote", name)


# ================= 1. End-to-end system =================
def fig_system():
    fig, ax = new((16, 8.4))
    ax.text(50, 95, "FSI-SLM: an end-to-end digital twin", ha="center",
            fontsize=24, fontweight="bold", color=style.NAVY)
    exp = box(ax, 1, 60, 21, 19, "Experiment", "PIV + hydrofoil", fc=style.TEAL, sub_fs=12)
    unet = box(ax, 1, 30, 21, 19, "U-Net flow\nsensor", "shape → fields + forces", fc=style.BLUE, sub_fs=11)
    feat = box(ax, 27, 45, 15, 19, "Feature\nextraction", "FFT · POD · St", fc=style.GREEN, sub_fs=11)
    rec  = box(ax, 48, 45, 15, 19, "FeatureRecord", "JSON summary", fc=style.GOLD, tc=style.INK, sub_fs=12)
    slm  = box(ax, 69, 45, 17, 19, "Small Language\nModel", "Phi-4-mini + LoRA", fc=style.CORAL, sub_fs=11)
    out  = box(ax, 69, 12, 17, 18, "Answers", "regime · lock-in · Q&A", fc=style.PURPLE, sub_fs=11)
    rag  = box(ax, 40, 12, 23, 18, "RAG literature", "~30 FSI papers · FAISS", fc=style.NAVY, sub_fs=11)
    # arrows
    arrow(ax, (22, 69), (27, 58))
    arrow(ax, (22, 39), (27, 51))
    arrow(ax, (42, 54.5), (48, 54.5))
    arrow(ax, (63, 54.5), (69, 54.5))
    arrow(ax, (77.5, 45), (77.5, 30))
    label(ax, 82, 37, "answers", fs=10, color=style.GREY)
    arrow(ax, (54, 30), (74, 45), color=style.NAVY, rad=-0.18, ls="--")
    label(ax, 58, 41, "retrieved\npassages", fs=10, color=style.NAVY)
    # bottom banner
    ax.add_patch(FancyBboxPatch((1, 1.5), 85, 7, boxstyle="round,pad=0.2,rounding_size=1.5",
                 facecolor=style.SAND, edgecolor=style.GREY, lw=1.5))
    ax.text(43.5, 5, "Served via  FastAPI REST API  +  MCP server   —   entire pipeline runs on one laptop, no GPU cluster",
            ha="center", va="center", fontsize=14, color=style.INK, fontweight="bold")
    save(fig, "fig_system.png")


# ================= 2. U-Net architecture =================
def fig_unet():
    fig, ax = new((16, 8))
    ax.text(50, 96, "U-Net flow sensor: convolutional encoder–decoder", ha="center",
            fontsize=22, fontweight="bold", color=style.NAVY)
    ax.text(50, 89, "shape (signed-distance field) + flow condition  →  velocity, pressure, forces",
            ha="center", fontsize=14, color=style.GREY)

    def blk(x, yc, h, w, color, txt=None, tc="white"):
        ax.add_patch(FancyBboxPatch((x, yc-h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.8",
                     facecolor=color, edgecolor="white", lw=1.2, zorder=3))
        if txt:
            ax.text(x+w/2, yc, txt, ha="center", va="center", color=tc,
                    fontsize=11, fontweight="bold", zorder=4, rotation=90)
        return x+w/2

    # input tensor (4 channels)
    for i, c in enumerate([style.RED, style.GOLD, style.GREEN, style.BLUE]):
        ax.add_patch(FancyBboxPatch((2+i*0.9, 46-i*0.9), 8, 26, boxstyle="round,pad=0.01,rounding_size=0.5",
                     facecolor=c, alpha=0.85, edgecolor="white", lw=1, zorder=2+i))
    ax.text(6.5, 32, "input tensor\n4 × 256 × 256", ha="center", fontsize=11, color=style.INK)

    enc_x = [16, 24, 32]; enc_y = [66, 54, 44]; enc_h = [22, 16, 11]
    bx = 42; by = 36; bh = 8
    dec_x = [52, 60, 68]; dec_y = [44, 54, 66]; dec_h = [11, 16, 22]
    encc = [style.TEAL, style.TEAL, style.TEAL]
    decc = [style.CORAL, style.CORAL, style.CORAL]
    ecent = []
    for x, y, h, c in zip(enc_x, enc_y, enc_h, encc):
        ecent.append((blk(x, y, h, 6, c), y, h))
    blk(bx, by, bh, 6, style.NAVY)
    ax.text(bx+3, 21, "bottleneck", ha="center", fontsize=11, color=style.NAVY, fontweight="bold")
    dcent = []
    for x, y, h, c in zip(dec_x, dec_y, dec_h, decc):
        dcent.append((blk(x, y, h, 6, c), y, h))
    # down/up arrows along the U
    arrow(ax, (10.5, 59), (16, 66), color=style.GREY, lw=2)
    for i in range(2):
        arrow(ax, (enc_x[i]+6, enc_y[i]), (enc_x[i+1], enc_y[i+1]), color=style.GREY, lw=2)
    arrow(ax, (enc_x[2]+6, enc_y[2]), (bx, by), color=style.GREY, lw=2)
    arrow(ax, (bx+6, by), (dec_x[0], dec_y[0]), color=style.GREY, lw=2)
    for i in range(2):
        arrow(ax, (dec_x[i]+6, dec_y[i]), (dec_x[i+1], dec_y[i+1]), color=style.GREY, lw=2)
    # skip connections (dashed)
    for (ex, ey, eh), (dx, dy, dh) in zip(ecent, reversed(dcent)):
        arrow(ax, (ex+3, ey+eh/2+0.5), (dx-3, dy+dh/2+0.5), color=style.GOLD,
              lw=1.8, ls=(0, (4, 3)), style_="-|>", rad=-0.12)
    ax.text(42, 74, "skip connections", ha="center", color=style.GOLD, fontsize=12, fontweight="bold")
    # outputs
    arrow(ax, (71, 66), (78, 62), color=style.GREY, lw=2)
    box(ax, 78, 54, 19, 15, "Field head", "u, v, p  (256×256)", fc=style.AQUA, tc=style.INK, sub_fs=11)
    box(ax, 78, 33, 19, 13, "Force head", "C$_l$ , C$_d$", fc=style.PURPLE, sub_fs=12)
    arrow(ax, (71, 64), (78, 47), color=style.GREY, lw=2, rad=-0.2)
    ax.text(23, 30, "encoder (downsample)", color=style.TEAL, fontsize=12, fontweight="bold", ha="center")
    ax.text(61, 30, "decoder (upsample)", color=style.CORAL, fontsize=12, fontweight="bold", ha="center")
    save(fig, "fig_unet.png")


# ================= 3. SLM training pipeline =================
def fig_training():
    fig, ax = new((16, 4.6))
    ax.text(50, 90, "How the language model is trained", ha="center",
            fontsize=21, fontweight="bold", color=style.NAVY)
    xs = [1, 21, 41, 61, 81]
    data = [("Synthetic\ngenerator", "5 coupled regimes", style.TEAL),
            ("Feature\nextraction", "FFT · POD · St", style.GREEN),
            ("SFT dataset", "prompt + target", style.GOLD),
            ("LoRA\nfine-tune", "MLX · rank 8", style.CORAL),
            ("Evaluation", "F1 · field acc.", style.PURPLE)]
    for i, (x, (t, s, c)) in enumerate(zip(xs, data)):
        tc = style.INK if c == style.GOLD else "white"
        box(ax, x, 30, 17, 34, t, s, fc=c, tc=tc, sub_fs=11)
        if i < 4:
            arrow(ax, (x+17, 47), (xs[i+1], 47))
    save(fig, "fig_training.png")


# ================= 4. RAG diagram =================
def fig_rag():
    fig, ax = new((15, 7.2))
    ax.text(50, 95, "Retrieval-augmented generation grounds answers in real papers",
            ha="center", fontsize=19, fontweight="bold", color=style.NAVY)
    # indexing row
    ax.text(2, 80, "Build the index (once)", fontsize=13, color=style.GREY, fontweight="bold")
    idx = [("PDF corpus", "~30 FSI papers", style.TEAL),
           ("Chunk", "256-token passages", style.GREEN),
           ("Embed", "all-MiniLM-L6-v2", style.BLUE),
           ("FAISS index", "vector search", style.NAVY)]
    xs = [1, 25, 49, 73]
    for i, (x, (t, s, c)) in enumerate(zip(xs, idx)):
        box(ax, x, 58, 21, 16, t, s, fc=c, sub_fs=11)
        if i < 3:
            arrow(ax, (x+21, 66), (xs[i+1], 66))
    # query row
    ax.text(2, 44, "Answer a question", fontsize=13, color=style.GREY, fontweight="bold")
    box(ax, 1, 18, 22, 18, "Question +\nFeatureRecord", None, fc=style.GOLD, tc=style.INK)
    box(ax, 30, 18, 22, 18, "Retrieve top-k", "nearest passages", fc=style.CORAL, sub_fs=11)
    box(ax, 59, 18, 15, 18, "Prompt\nassembly", None, fc=style.PURPLE)
    box(ax, 80, 18, 17, 18, "SLM answer", "with citations", fc=style.TEAL, sub_fs=11)
    arrow(ax, (23, 27), (30, 27))
    arrow(ax, (52, 27), (59, 27))
    arrow(ax, (74, 27), (80, 27))
    arrow(ax, (83, 50), (41, 36), color=style.NAVY, ls="--", rad=0.15)
    label(ax, 70, 45, "search", fs=11, color=style.NAVY)
    save(fig, "fig_rag.png")


# ================= 5. Regime explainer =================
def fig_regimes():
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2))
    fig.suptitle("Five fluid–structure interaction regimes the model distinguishes",
                 fontsize=19, fontweight="bold", color=style.NAVY, y=1.06)
    regs = [
        ("Attached", style.REGIME_COLORS["attached"], "quiet wake\ntiny vibration"),
        ("Separated", style.REGIME_COLORS["separated"], "broadband wake\nno limit cycle"),
        ("VIV lock-in", style.REGIME_COLORS["viv_lockin"], "shedding ≈ f$_n$\nlarge limit cycle"),
        ("Galloping", style.REGIME_COLORS["galloping"], "f$_n$ ≪ f$_{shed}$\ngrowing amplitude"),
        ("Stall flutter", style.REGIME_COLORS["stall_flutter"], "dynamic stall\ntorsional LCO"),
    ]

    def foil(ax, aoa=6, y0=0.5, color=style.INK):
        import panel
        x, yy = panel.naca4("0009", 80)
        th = np.radians(-aoa)
        xr = (x-0.25)*np.cos(th) - yy*np.sin(th)
        yr = (x-0.25)*np.sin(th) + yy*np.cos(th)
        ax.fill(0.32+xr*0.34, y0+yr*0.34, color=color, zorder=5)

    for ax, (name, col, sig) in zip(axes, regs):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.01,rounding_size=0.03",
                     facecolor=col, alpha=0.08, edgecolor=col, lw=2))
        ax.set_title(name, color=col, fontsize=15, fontweight="bold")
        if name == "Attached":
            foil(ax, 3)
            for yy in np.linspace(0.32, 0.7, 6):
                ax.plot([0.05, 0.95], [yy, yy], color=col, lw=1, alpha=0.6)
        elif name == "Separated":
            foil(ax, 12)
            rng = np.random.default_rng(3)
            for _ in range(28):
                ax.add_patch(Circle((rng.uniform(0.55, 0.95), rng.uniform(0.3, 0.72)),
                             rng.uniform(0.015, 0.05), color=col, alpha=0.5))
        elif name == "VIV lock-in":
            foil(ax, 4)
            for i, xx in enumerate(np.linspace(0.55, 0.92, 4)):
                ax.add_patch(Circle((xx, 0.6 if i % 2 else 0.4), 0.05, color=col, alpha=0.7))
            ax.annotate("", (0.32, 0.78), (0.32, 0.22),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=2.5))
        elif name == "Galloping":
            foil(ax, 5)
            ax.annotate("", (0.32, 0.85), (0.32, 0.15),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=4))
            ax.plot([0.6, 0.95], [0.5, 0.5], color=col, lw=1, ls="--")
        else:  # stall flutter
            foil(ax, 15)
            ax.annotate("", (0.5, 0.72), (0.14, 0.34),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=2.5,
                                        connectionstyle="arc3,rad=0.4"))
            rng = np.random.default_rng(1)
            for _ in range(14):
                ax.add_patch(Circle((rng.uniform(0.5, 0.9), rng.uniform(0.5, 0.8)),
                             rng.uniform(0.02, 0.045), color=col, alpha=0.45))
        ax.text(0.5, 0.11, sig, ha="center", va="center", fontsize=11.5,
                color=style.INK, transform=ax.transAxes)
    fig.tight_layout()
    save(fig, "fig_regimes.png")


# ================= 6. Digital twin concept =================
def fig_twin():
    fig, ax = new((14, 7))
    ax.text(50, 95, "The digital twin: measure once, then ask anything",
            ha="center", fontsize=20, fontweight="bold", color=style.NAVY)
    # physical panel
    ax.add_patch(FancyBboxPatch((3, 20), 38, 60, boxstyle="round,pad=0.4,rounding_size=2",
                 facecolor=style.TEAL, alpha=0.12, edgecolor=style.TEAL, lw=2.5))
    ax.text(22, 74, "PHYSICAL WORLD", ha="center", fontsize=15, fontweight="bold", color=style.TEAL)
    # water tunnel + foil + PIV
    ax.add_patch(FancyBboxPatch((8, 40), 28, 22, boxstyle="round,pad=0.1,rounding_size=1",
                 facecolor="#d8f0f2", edgecolor=style.TEAL, lw=1.5))
    import panel
    xf, yf = panel.naca4("0009", 80)
    th = np.radians(-8)
    xr = (xf-0.25)*np.cos(th)-yf*np.sin(th); yr = (xf-0.25)*np.sin(th)+yf*np.cos(th)
    ax.fill(20+xr*16, 51+yr*16, color=style.INK, zorder=6)
    for yy in np.linspace(43, 59, 6):
        arrow(ax, (9, yy), (16, yy), color=style.TEAL, lw=1.3, style_="-|>")
    ax.text(22, 35, "hydrofoil + PIV\nvibration sensors", ha="center", fontsize=12, color=style.INK)
    # digital panel
    ax.add_patch(FancyBboxPatch((59, 20), 38, 60, boxstyle="round,pad=0.4,rounding_size=2",
                 facecolor=style.CORAL, alpha=0.12, edgecolor=style.CORAL, lw=2.5))
    ax.text(78, 74, "DIGITAL TWIN", ha="center", fontsize=15, fontweight="bold", color=style.CORAL)
    box(ax, 64, 52, 28, 14, "FeatureRecord", "+ U-Net fields", fc=style.GOLD, tc=style.INK, sub_fs=11)
    box(ax, 64, 30, 28, 15, "Language model", "interprets & answers", fc=style.CORAL, sub_fs=11)
    arrow(ax, (78, 52), (78, 45), color=style.INK)
    # loop arrows between worlds
    arrow(ax, (41, 58), (64, 60), color=style.TEAL, lw=3, rad=-0.25)
    label(ax, 52, 68, "measurements", fs=12, color=style.TEAL, weight="bold")
    arrow(ax, (64, 30), (41, 33), color=style.CORAL, lw=3, rad=-0.25)
    label(ax, 52, 22, "interpretation · what-if sweeps", fs=12, color=style.CORAL, weight="bold")
    save(fig, "fig_twin.png")


if __name__ == "__main__":
    fig_system(); fig_unet(); fig_training(); fig_rag(); fig_regimes(); fig_twin()
    print("diagrams done")
