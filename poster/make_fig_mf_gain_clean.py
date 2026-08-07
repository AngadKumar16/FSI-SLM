import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
RUST="#C8552A"; TEAL="#1F7A6E"; GREY="#9AA5A8"

fig, ax = plt.subplots(1, 3, figsize=(14.5, 3.9))
fig.suptitle("Multi-fidelity fine-tuning: 47 high-fidelity cases (30 converged) close the transfer gap",
             fontsize=13, fontweight="bold")

def grouped(a, labels, base, tuned, ylab, title, fmt="{:.2f}"):
    x = np.arange(len(labels)); w = 0.36
    a.bar(x-w/2, base,  w, color=RUST, label="panel-only")
    a.bar(x+w/2, tuned, w, color=TEAL, label="+ high-fidelity")
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=10.5)
    a.set_ylabel(ylab, fontsize=10.5); a.set_title(title, fontsize=11)
    a.grid(axis="y", alpha=0.25); a.set_axisbelow(True)
    top = max(max(base), max(tuned))
    a.set_ylim(0, top*1.30)
    for xi, b, t in zip(x, base, tuned):
        d = (t-b)/b*100
        a.text(xi, max(b, t)+top*0.05, f"{d:+.0f}%", ha="center", fontsize=9.5,
               fontweight="bold", color=TEAL if d < 0 else RUST)
    return a

grouped(ax[0], ["u","v","p","load\n$\\Delta C_p$","disp\n$w/c$"],
        [0.0259,0.3367,0.4973,0.1999,0.5764],
        [0.0216,0.2565,0.3504,0.0833,0.2344],
        "relative RMSE vs RANS", "(a) field channels, 7 RANS test cases")
ax[0].legend(frameon=False, fontsize=9.5, loc="upper left")

grouped(ax[1], ["$C_l$","$C_d$","$C_m$","$w_{max}$","twist"],
        [0.2373,0.3690,0.6051,1.3353,1.0452],
        [0.0711,0.3486,0.5118,0.6458,1.0980],
        "nRMSE vs RANS", "(b) scalars  (n = 7, use nRMSE not $R^2$)")

grouped(ax[2], ["$C_l$","$C_d$","$C_m$"],
        [0.0456,0.0677,0.0946],
        [0.0465,0.0798,0.0972],
        "nRMSE vs panel (n = 293)", "(c) cheap fidelity did not collapse")
ax[2].text(0.5, 0.88, "worst drift 0.012 absolute", transform=ax[2].transAxes,
           ha="center", fontsize=10, style="italic", color="#5B6B72")

fig.tight_layout(rect=[0,0,1,0.92])
fig.savefig("poster/generated/fig_mf_gain_clean.png", dpi=220)
print("ok")
