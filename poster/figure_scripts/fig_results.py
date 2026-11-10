"""Results charts from the real evaluation JSON."""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import style
style.apply()

import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT = ROOT + "/poster/generated"
EVAL = ROOT + "/artifacts/eval_results/eval_results.json"
os.makedirs(OUT, exist_ok=True)
with open(EVAL) as fh:
    E = json.load(fh)

REG = ["attached", "separated", "viv_lockin", "galloping", "stall_flutter"]
LAB = [style.REGIME_LABELS[r] for r in REG]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("wrote", name)


# ---------- confusion matrix ----------
def fig_confusion():
    M = E["regime_confusion_matrix"]["matrix"]
    C = np.array([[M[t][p] for p in REG] for t in REG], float)
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=C.max())
    for i in range(5):
        for j in range(5):
            v = int(C[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > C.max()*0.55 else style.INK,
                    fontsize=17, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(LAB, rotation=35, ha="right"); ax.set_yticklabels(LAB)
    ax.set_xlabel("Predicted regime"); ax.set_ylabel("True regime")
    ax.set_title("Regime classification: confusion matrix\n(held-out test set, n = 20)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="count")
    save(fig, "fig_confusion.png")


# ---------- per-class precision/recall/F1 ----------
def fig_perclass():
    cm = E["regime_confusion_matrix"]["class_metrics"]
    prec = [cm[r]["precision"] for r in REG]
    rec = [cm[r]["recall"] for r in REG]
    f1 = [cm[r]["f1"] for r in REG]
    x = np.arange(5); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x-w, prec, w, label="Precision", color=style.TEAL)
    ax.bar(x, rec, w, label="Recall", color=style.GOLD)
    ax.bar(x+w, f1, w, label="F1", color=style.CORAL)
    ax.set_xticks(x); ax.set_xticklabels(LAB, rotation=20, ha="right")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.axhline(1.0, color=style.GREY, lw=0.8, ls=":")
    ax.set_title("Per-regime detection performance")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    for i, v in enumerate(f1):
        ax.text(x[i]+w, v+0.02, f"{v:.2f}", ha="center", fontsize=11)
    macro = np.mean(f1)
    ax.text(0.02, 1.07, f"Macro-F1 = {macro:.2f}", transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=style.NAVY)
    save(fig, "fig_perclass.png")


# ---------- per-field numeric accuracy ----------
def fig_perfield():
    pf = E["per_field_accuracy"]
    names = {"shedding_frequency_hz": "Shedding\nfrequency",
             "strouhal_number": "Strouhal\nnumber",
             "rms_amplitude_over_c": "RMS\namplitude",
             "mode_number": "Mode\nnumber"}
    keys = list(names)
    acc = [pf[k]["accuracy"]*100 for k in keys]
    err = [pf[k]["mean_relative_error"]*100 for k in keys]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    ax[0].bar(range(4), acc, color=style.GREEN, width=0.6)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels([names[k] for k in keys])
    ax[0].set_ylim(0, 112); ax[0].set_ylabel("Accuracy (%)")
    ax[0].set_title("Numeric field accuracy (within tolerance)")
    for i, v in enumerate(acc):
        ax[0].text(i, v+2, f"{v:.0f}%", ha="center", fontsize=13, fontweight="bold")
    ax[1].bar(range(4), err, color=style.BLUE, width=0.6)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([names[k] for k in keys])
    ax[1].set_ylabel("Mean relative error (%)")
    ax[1].set_title("Mean relative error per field")
    for i, v in enumerate(err):
        ax[1].text(i, v+0.05, f"{v:.2f}%", ha="center", fontsize=12)
    fig.suptitle("Feature extraction: how accurately the pipeline reads each quantity",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig_perfield.png")


# ---------- headline metric cards ----------
def fig_metric_cards():
    cm = E["regime_confusion_matrix"]["class_metrics"]
    macro_f1 = np.mean([cm[r]["f1"] for r in REG])
    cards = [("1,926", "CFD cases\n(107 shapes x 18 AoA)", style.TEAL),
             ("5", "FSI regimes\nclassified", style.CORAL),
             (f"{macro_f1*100:.0f}%", "Regime\nmacro-F1", style.GREEN),
             ("100%", "Lock-in\ndetection F1", style.GOLD),
             ("1", "laptop\nno GPU cluster", style.PURPLE)]
    fig, ax = plt.subplots(figsize=(15, 3.0)); ax.axis("off")
    n = len(cards); w = 1/n
    for i, (big, small, col) in enumerate(cards):
        x = i*w + w/2
        ax.add_patch(plt.Rectangle((i*w+0.01, 0.05), w-0.02, 0.9,
                     transform=ax.transAxes, facecolor=col, alpha=0.14,
                     edgecolor=col, linewidth=2, zorder=0))
        ax.text(x, 0.60, big, transform=ax.transAxes, ha="center",
                fontsize=40, fontweight="bold", color=col)
        ax.text(x, 0.24, small, transform=ax.transAxes, ha="center",
                fontsize=15, color=style.INK)
    save(fig, "fig_metric_cards.png")


if __name__ == "__main__":
    fig_confusion(); fig_perclass(); fig_perfield(); fig_metric_cards()
    print("results figures done")
