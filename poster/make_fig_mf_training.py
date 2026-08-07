import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

h = json.load(open("artifacts/sensor/run_mf/history.json"))
ep = [x["epoch"] for x in h]
tr = [x["train"]["loss"] for x in h]
va = [x["val"]["loss"] for x in h]
mae = [x["val"]["scalar_mae"] for x in h]

fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0))
fig.suptitle("Multi-fidelity U-Net training: one 11.59 M network, three solvers, 23 min on a laptop GPU",
             fontsize=13, fontweight="bold")

# (a) fidelity mix
mix = [("panel\n(inviscid)", 1881), ("steady RANS\n(OpenFOAM)", 35), ("URANS\n(deep stall)", 12)]
labels = [m[0] for m in mix]; vals = [m[1] for m in mix]
cols = ["#B8B8B8", "#C8552A", "#1F7A6E"]
b = ax[0].bar(labels, vals, color=cols)
ax[0].set_yscale("log"); ax[0].set_ylabel("training cases (log)")
ax[0].set_title("(a) 1,928-case corpus, 3 fidelities", fontsize=11)
for r, v in zip(b, vals):
    ax[0].text(r.get_x()+r.get_width()/2, v*1.25, f"{v:,}", ha="center", fontweight="bold", fontsize=10)
ax[0].set_ylim(5, 6000)
ax[0].text(0.5, 0.93, "loss weights  panel=1 · rans=8 · urans=8", transform=ax[0].transAxes,
           ha="center", fontsize=9.5, style="italic")

# (b) loss curves
ax[1].plot(ep, tr, color="#C8552A", lw=1.8, label="train")
ax[1].plot(ep, va, color="#1F7A6E", lw=2.2, label="val")
ax[1].axhline(min(va), ls=":", color="#1F7A6E", lw=1)
ax[1].annotate(f"best val {min(va):.4f}", xy=(len(ep)*0.55, min(va)),
               xytext=(len(ep)*0.42, min(va)+0.12), fontsize=9.5, color="#1F7A6E",
               arrowprops=dict(arrowstyle="->", color="#1F7A6E", lw=1))
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("total loss")
ax[1].set_title("(b) 40 epochs, no early stop", fontsize=11); ax[1].legend(frameon=False)
ax[1].grid(alpha=0.25)

# (c) val scalar MAE
ax[2].plot(ep, mae, color="#204070", lw=2.2)
ax[2].set_xlabel("epoch"); ax[2].set_ylabel("val scalar MAE")
ax[2].set_title("(c) force coefficients keep improving", fontsize=11)
ax[2].grid(alpha=0.25)
ax[2].annotate(f"{mae[0]:.3f} → {mae[-1]:.3f}", xy=(0.55, 0.75), xycoords="axes fraction",
               fontsize=11, fontweight="bold", color="#204070")

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("poster/generated/fig_mf_training.png", dpi=220)
print("ok", min(va), mae[0], mae[-1])
