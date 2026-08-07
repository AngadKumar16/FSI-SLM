import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch
OUT="poster/generated/mf_alt/"
RUST="#C8552A"; TEAL="#1F7A6E"; GREY="#9AA5A8"; NAVY="#204070"; MUTE="#5B6B72"

fig,ax=plt.subplots(figsize=(12.0,5.4)); ax.set_xlim(0,12); ax.set_ylim(0,5.4); ax.axis("off")
CX=3.55
tiers=[(0.45,1.15,2.70,GREY,"panel method","1,881 cases  ·  under a second each","rough, but everywhere"),
       (1.70,1.10,2.05,RUST,"steady RANS","35 cases  ·  3 minutes each","accurate below stall"),
       (2.90,1.05,1.45,TEAL,"URANS","12 cases  ·  2 h 40 min","the only thing that works past stall")]
for y,h,hw,c,name,count,note in tiers:
    ax.add_patch(Polygon([[CX-hw,y],[CX+hw,y],[CX+hw*0.70,y+h],[CX-hw*0.70,y+h]],
                         closed=True, fc=c, ec="white", lw=2.5))
    ax.text(CX,y+h*0.64,name,ha="center",va="center",color="white",fontsize=14,fontweight="bold")
    ax.text(CX,y+h*0.26,count,ha="center",va="center",color="white",fontsize=10.5)
    ax.text(CX+hw+0.22,y+h*0.5,note,ha="left",va="center",color=MUTE,fontsize=11.5,style="italic")

ax.add_patch(FancyArrowPatch((0.62,0.60),(0.62,3.85),arrowstyle="-|>",mutation_scale=16,lw=2,color=NAVY))
ax.text(0.36,2.2,"more accurate, far more expensive",rotation=90,ha="center",va="center",
        color=NAVY,fontsize=11,fontweight="bold")

ax.add_patch(FancyBboxPatch((8.15,1.25),3.55,2.30,boxstyle="round,pad=0.12,rounding_size=0.15",
                            fc="#EEF3F4",ec="none"))
ax.text(9.93,3.14,"One network, all three",ha="center",fontsize=13.5,fontweight="bold",color=NAVY)
ax.text(9.93,2.20,"We tell it which solver each case\ncame from, so it learns how they\ndisagree — and the 47 expensive\ncases count 8× as much.",
        ha="center",va="center",fontsize=11.5,color="#1A1A1A")
ax.add_patch(FancyArrowPatch((7.35,2.40),(8.05,2.40),arrowstyle="-|>",mutation_scale=18,lw=2,color=NAVY))
ax.text(6.0,5.05,"Three solvers, one training set",ha="center",fontsize=15,fontweight="bold")
fig.savefig(OUT+"B10_data_pyramid.png",dpi=200,bbox_inches="tight"); print("ok")
