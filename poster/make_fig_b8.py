import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch
OUT="poster/generated/mf_alt/"
RUST="#C8552A"; TEAL="#1F7A6E"; GREY="#9AA5A8"; NAVY="#204070"; MUTE="#5B6B72"
plt.rcParams.update({"font.size":11})

# ---- B8 revised ----
fig,axs=plt.subplots(1,2,figsize=(12.6,4.2))
lab=["panel","steady RANS","URANS"]; col=[GREY,RUST,TEAL]
v1=[1881,35,12]
b=axs[0].bar(lab,v1,color=col); axs[0].set_yscale("log"); axs[0].set_ylim(5,7000)
axs[0].set_title("how many cases we have",fontsize=12.5)
for r,v in zip(b,v1): axs[0].text(r.get_x()+r.get_width()/2,v*1.45,f"{v:,}",ha="center",fontweight="bold",fontsize=13)
axs[0].set_ylabel("cases (log scale)")

v2=[0.6,189,9566]; txt=["under a second","3 minutes","2 h 40 min"]
b=axs[1].bar(lab,v2,color=col); axs[1].set_yscale("log"); axs[1].set_ylim(0.15,60000)
axs[1].set_title("what one case costs to solve",fontsize=12.5)
for r,v,t in zip(b,v2,txt): axs[1].text(r.get_x()+r.get_width()/2,v*1.9,t,ha="center",fontweight="bold",fontsize=12)
axs[1].set_ylabel("solver time per case (log scale)")
for a in axs: a.grid(axis="y",alpha=0.25); a.set_axisbelow(True); a.tick_params(labelsize=11)
fig.suptitle("Cheap data is everywhere. Accurate data is not. So we use both instead of picking one.",
             fontsize=13.5,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.90]); fig.savefig(OUT+"B8_cheap_vs_expensive.png",dpi=200); plt.close(fig)

# ---- B10 pyramid ----
fig,ax=plt.subplots(figsize=(11.5,5.0)); ax.set_xlim(0,11.5); ax.set_ylim(0,5.0); ax.axis("off")
tiers=[(0.35,1.30,4.30,GREY,"panel method","1,881 cases  ·  under a second each","rough but everywhere"),
       (1.55,1.15,2.60,RUST,"steady RANS","35 cases  ·  3 minutes each","accurate below stall"),
       (2.75,1.05,1.10,TEAL,"URANS","12 cases  ·  2 h 40 min each","the only thing that works past stall")]
for y,h,halfw,c,name,count,note in tiers:
    cx=3.4
    ax.add_patch(Polygon([[cx-halfw,y],[cx+halfw,y],[cx+halfw*0.72,y+h],[cx-halfw*0.72,y+h]],
                         closed=True, fc=c, ec="white", lw=2.5))
    ax.text(cx,y+h*0.62,name,ha="center",va="center",color="white",fontsize=14,fontweight="bold")
    ax.text(cx,y+h*0.26,count,ha="center",va="center",color="white",fontsize=11.5)
    ax.text(cx+halfw+0.35,y+h*0.5,note,ha="left",va="center",color=MUTE,fontsize=12,style="italic")
ax.add_patch(FancyArrowPatch((0.55,1.25),(0.55,3.85),arrowstyle="-|>",mutation_scale=18,lw=2,color=NAVY))
ax.text(0.30,2.55,"more accurate, far more expensive",rotation=90,ha="center",va="center",
        color=NAVY,fontsize=11.5,fontweight="bold")
ax.add_patch(FancyBboxPatch((7.9,1.35),3.35,2.0,boxstyle="round,pad=0.12,rounding_size=0.15",
                            fc="#EEF3F4",ec="none"))
ax.text(9.57,2.95,"One network, all three",ha="center",fontsize=13,fontweight="bold",color=NAVY)
ax.text(9.57,2.15,"We tell it which solver each\ncase came from, and count the\n47 expensive ones 8× as much.",
        ha="center",va="center",fontsize=11.5,color="#1A1A1A")
ax.add_patch(FancyArrowPatch((7.0,2.35),(7.8,2.35),arrowstyle="-|>",mutation_scale=18,lw=2,color=NAVY))
ax.text(5.75,4.65,"Three solvers, one training set",ha="center",fontsize=14.5,fontweight="bold")
fig.tight_layout(); fig.savefig(OUT+"B10_data_pyramid.png",dpi=200); plt.close(fig)
print("ok")
