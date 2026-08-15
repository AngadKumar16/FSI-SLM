"""Two poster figures the repo did not have, drawn with the numbers that
are actually true today (16 papers / 1,840 passages / 12 MCP tools).

    python3 poster/make_new_figs.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "generated"
OUT.mkdir(parents=True, exist_ok=True)

TEAL = "#1B9AAA"
NAVY = "#1F3864"
DEEP = "#12233F"
CORAL = "#F4714F"
GREEN = "#2E8B57"
PURPLE = "#7B52C4"
GOLD = "#D99400"
GREY = "#5C6266"
INK = "#12233F"
FAINT = "#9AA3A9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def box(ax, x, y, w, h, label, sub=None, fc=TEAL, tc="white", fs=10.5,
        sub_fs=8.2, radius=0.010):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=fc, linewidth=0, zorder=2))
    nl = sub.count("\n") if sub else 0
    cy = y + h / 2 + (0.030 + nl * 0.014 if sub else 0)
    ax.text(x + w / 2, cy, label, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold", zorder=3, linespacing=1.25)
    if sub:
        ax.text(x + w / 2, cy - 0.052 - nl * 0.008, sub, ha="center",
                va="top", color=tc, fontsize=sub_fs, alpha=0.93, zorder=3,
                linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=DEEP, lw=2.0, ls="-", rad=0.0, ms=13):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
        color=color, linewidth=lw, linestyle=ls, zorder=4,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax


def kicker(ax, x, y, text, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=9.0,
            fontweight="bold", color=GREY)


# ===================================================================== #
# one experiment -> one machine-readable record
# ===================================================================== #
def fig_record():
    fig, ax = blank(11.6, 4.6)
    TITLE_Y = 0.955
    ax.text(0.5, TITLE_Y, "One experiment becomes one machine-readable record",
            ha="center", va="center", fontsize=15.5, fontweight="bold",
            color=INK)

    # ---------------- left: what comes in --------------------------- #
    kicker(ax, 0.098, 0.875, "MEASURED  OR  SOLVED", ha="center")
    ins = [("PIV velocity fields", None, TEAL),
           ("force / moment traces", None, NAVY),
           ("structural response", None, PURPLE),
           ("U-Net surrogate", "geometry → fields", GREEN)]
    bx, bw, bh, step = 0.008, 0.180, 0.105, 0.140
    top = 0.760
    for i, (t, s, c) in enumerate(ins):
        box(ax, bx, top - i * step, bw, bh, t, sub=s, fc=c, fs=9.8,
            sub_fs=8.0)
        arrow(ax, bx + bw + 0.004, top - i * step + bh / 2,
              0.246, 0.500, color=FAINT, lw=1.5, rad=0.12, ms=10)

    ax.text(0.098, 0.185, "the rig and the surrogate\nwrite the same fields",
            ha="center", va="center", fontsize=8.6, color=GREY,
            style="italic", linespacing=1.4)

    # ---------------- centre: the JSON card ------------------------- #
    CX, CW = 0.252, 0.300
    CY0, CY1 = 0.060, 0.880
    HDR = 0.086
    ax.add_patch(FancyBboxPatch(
        (CX, CY0), CW, CY1 - CY0,
        boxstyle="round,pad=0,rounding_size=0.012",
        facecolor="#F4F7F8", edgecolor=DEEP, linewidth=1.8, zorder=2))
    ax.add_patch(FancyBboxPatch(
        (CX, CY1 - HDR), CW, HDR,
        boxstyle="round,pad=0,rounding_size=0.012",
        facecolor=DEEP, edgecolor=DEEP, linewidth=1.8, zorder=3))
    ax.add_patch(plt.Rectangle((CX, CY1 - HDR), CW, 0.012, color=DEEP,
                               zorder=3))
    ax.text(CX + CW / 2, CY1 - HDR / 2, "FeatureRecord   ·   JSON",
            ha="center", va="center", color="white", fontsize=11.4,
            fontweight="bold", zorder=4)

    rows = [
        ("conditions", "Re, α, U∞, U*, chord", TEAL),
        ("piv_derived", "f_shed, St, wake width,\nseparation x/c, POD, TKE",
         NAVY),
        ("structure", "A_rms/c, f_dom, f_n,\nmode, camber", PURPLE),
        ("forces", "Cl, Cd, Cm — mean, rms, freq", CORAL),
        ("data_quality", "11 fault flags + severity", GOLD),
    ]
    y = CY1 - HDR - 0.030
    for key, val, col in rows:
        nl = val.count("\n")
        h = 0.086 + nl * 0.052
        ax.add_patch(plt.Rectangle((CX + 0.014, y - h), 0.009, h, color=col,
                                   zorder=4))
        ax.text(CX + 0.036, y - 0.024, f'"{key}"', ha="left", va="center",
                fontsize=10.0, fontweight="bold", color=col,
                family="DejaVu Sans Mono", zorder=4)
        ax.text(CX + 0.036, y - 0.052, val, ha="left", va="top",
                fontsize=8.6, color="#33404A", family="DejaVu Sans Mono",
                linespacing=1.5, zorder=4)
        y -= h + 0.020

    ax.text(CX + CW / 2, CY0 + 0.036,
            "the true regime is never in the prompt", ha="center",
            va="center", fontsize=8.4, color=GREY, style="italic", zorder=4)

    # ---------------- right: the gate and the reader ---------------- #
    RX, RW = 0.610, 0.372
    arrow(ax, CX + CW + 0.006, 0.700, RX - 0.006, 0.700, lw=2.2)

    box(ax, RX, 0.626, RW, 0.148, "11 instrumentation-fault detectors",
        sub="PIV dropout  ·  ADC clipping  ·  mains hum  ·  spikes",
        fc=GOLD, fs=10.6, sub_fs=8.6)
    ax.text(RX + RW / 2, 0.592, "noise is flagged before the model reads it",
            ha="center", va="center", fontsize=8.6, color=GREY,
            style="italic")

    arrow(ax, RX + RW / 2, 0.566, RX + RW / 2, 0.506, lw=2.2)

    box(ax, RX, 0.352, RW, 0.150, "Phi-4-mini  3.8 B  +  LoRA",
        sub="46 MB adapter, fine-tuned on this contract",
        fc=CORAL, fs=11.0, sub_fs=8.6)

    arrow(ax, RX + RW / 2, 0.344, RX + RW / 2, 0.292, lw=2.2)

    chips = ["regime  ·  lock-in", "f_shed,  St,  mode",
             "caveats  ·  citations"]
    cw = (RW - 2 * 0.014) / 3
    for i, t in enumerate(chips):
        ax.add_patch(FancyBboxPatch(
            (RX + i * (cw + 0.014), 0.196), cw, 0.092,
            boxstyle="round,pad=0,rounding_size=0.010",
            facecolor="#E9EEF1", edgecolor="#C6CFD5", linewidth=1.1,
            zorder=2))
        ax.text(RX + i * (cw + 0.014) + cw / 2, 0.242, t, ha="center",
                va="center", fontsize=8.4, color=INK, fontweight="bold",
                zorder=3)
    ax.text(RX + RW / 2, 0.148,
            "structured answer, physics-validated before it is returned",
            ha="center", va="center", fontsize=8.6, color=GREY,
            style="italic")

    fig.savefig(OUT / "fig_p_record.png", dpi=210, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)


# ===================================================================== #
# grounded by retrieval, served three ways
# ===================================================================== #
def fig_serve():
    fig, ax = blank(8.3, 3.5)

    ax.text(0.5, 0.962,
            "Grounded by retrieval, served three ways — on one laptop",
            ha="center", va="center", fontsize=15.5, fontweight="bold",
            color=INK)

    x, w, gap = 0.008, 0.226, 0.030
    ctr = lambda i: x + i * (w + gap) + w / 2          # noqa: E731

    # ---- band 1: build the index ----------------------------------- #
    kicker(ax, 0.008, 0.858, "BUILD  THE  INDEX   (once)")
    b1 = [("PDF corpus", "16 FSI papers", TEAL),
          ("Chunk", "256 tok, 50% overlap", GREEN),
          ("Embed", "all-MiniLM-L6-v2", NAVY),
          ("FAISS index", "1,840 passages", DEEP)]
    B1Y, BH = 0.712, 0.114
    for i, (t, s, c) in enumerate(b1):
        box(ax, x + i * (w + gap), B1Y, w, BH, t, sub=s, fc=c, fs=11.0,
            sub_fs=8.6)
        if i < 3:
            arrow(ax, x + i * (w + gap) + w + 0.003, B1Y + BH / 2,
                  x + (i + 1) * (w + gap) - 0.003, B1Y + BH / 2)

    # ---- band 2: answer a question --------------------------------- #
    kicker(ax, 0.008, 0.618, "ANSWER  A  QUESTION")
    b2 = [("Record + question", None, GOLD),
          ("Retrieve top-3", "score ≥ 0.30", CORAL),
          ("Phi-4-mini + LoRA", "3.8 B · 46 MB", PURPLE),
          ("Answer + citations", "physics-validated", TEAL)]
    B2Y = 0.472
    for i, (t, s, c) in enumerate(b2):
        box(ax, x + i * (w + gap), B2Y, w, BH, t, sub=s, fc=c, fs=10.6,
            sub_fs=8.6)
        if i < 3:
            arrow(ax, x + i * (w + gap) + w + 0.003, B2Y + BH / 2,
                  x + (i + 1) * (w + gap) - 0.003, B2Y + BH / 2)

    # retrieval feed: FAISS index -> retrieve
    arrow(ax, ctr(3), B1Y - 0.004, ctr(1), B2Y + BH + 0.004, color=DEEP,
          lw=1.7, ls=(0, (5, 4)), rad=-0.05, ms=11)
    ax.text(0.637, 0.649, "nearest passages", ha="center", va="center",
            fontsize=9.4, color=DEEP, style="italic",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none"))

    # ---- band 3: three front doors --------------------------------- #
    kicker(ax, 0.008, 0.386, "CALL  IT")
    doors = [("CLI", "cli.py twin", "#4A5A66"),
             ("REST · FastAPI", "POST /infer", NAVY),
             ("MCP server", "12 tools", CORAL)]
    dw, dgap, dx = 0.214, 0.028, 0.008
    D3Y = 0.204
    dctr = lambda i: dx + i * (dw + dgap) + dw / 2     # noqa: E731

    # bus from the pipeline down to the three doors
    BUS = 0.352
    ax.plot([ctr(0), ctr(3)], [B2Y - 0.030, B2Y - 0.030], color=FAINT,
            lw=1.8, zorder=1)
    ax.plot([0.503, 0.503], [B2Y - 0.030, BUS], color=FAINT, lw=1.8,
            zorder=1)
    ax.plot([dctr(0), dctr(2)], [BUS, BUS], color=FAINT, lw=1.8, zorder=1)
    for i in range(3):
        arrow(ax, dctr(i), BUS, dctr(i), D3Y + BH + 0.004, color=FAINT,
              lw=1.8, ms=11)

    for i, (t, s, c) in enumerate(doors):
        box(ax, dx + i * (dw + dgap), D3Y, dw, BH, t, sub=s, fc=c, fs=10.8,
            sub_fs=8.6)

    agx = dx + 3 * (dw + dgap) + 0.020
    arrow(ax, agx - 0.024, D3Y + BH / 2, agx - 0.004, D3Y + BH / 2, lw=2.2)
    box(ax, agx, D3Y, 0.992 - agx, BH, "Agentic LLM",
        sub="MCP client", fc=GREEN, fs=10.8, sub_fs=8.6)

    # ---- footer ----------------------------------------------------- #
    ax.add_patch(FancyBboxPatch(
        (0.008, 0.040), 0.984, 0.108,
        boxstyle="round,pad=0,rounding_size=0.010",
        facecolor="#EEF2F3", edgecolor="#D3D9DC", linewidth=1.2, zorder=2))
    ax.text(0.500, 0.094,
            "Training, retrieval, inference and serving all run on one "
            "M4 Pro MacBook — no GPU cluster in the loop.",
            ha="center", va="center", fontsize=10.6, color=INK, zorder=3)

    fig.savefig(OUT / "fig_p_serve.png", dpi=210, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    fig_record()
    fig_serve()
    print("wrote fig_p_record.png, fig_p_serve.png")
