"""Reusable box + arrow primitives for clean flow diagrams."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import style


def box(ax, x, y, w, h, title, sub=None, fc=style.TEAL, tc="white",
        fs=15, sub_fs=12, round=0.02, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={round}",
                       linewidth=0, facecolor=fc, alpha=alpha, zorder=2)
    ax.add_patch(p)
    cy = y + h/2 + (0.04*h if sub else 0)
    ax.text(x+w/2, cy if sub else y+h/2, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc, zorder=3)
    if sub:
        ax.text(x+w/2, y+h/2 - 0.30*h, sub, ha="center", va="center",
                fontsize=sub_fs, color=tc, zorder=3)
    return (x+w/2, y+h/2, x, y, w, h)


def arrow(ax, xy1, xy2, color=style.INK, lw=2.6, style_="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style_, mutation_scale=20,
                        color=color, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)


def label(ax, x, y, txt, fs=12, color=style.GREY, weight="normal", ha="center", style_i="normal"):
    ax.text(x, y, txt, fontsize=fs, color=color, ha=ha, va="center",
            fontweight=weight, fontstyle=style_i, zorder=4)


def new(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax
