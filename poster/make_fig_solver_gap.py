import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT="poster/generated/mf_alt/"
RUST="#C8552A"; TEAL="#1F7A6E"; GREY="#9AA5A8"; NAVY="#204070"
plt.rcParams.update({"font.size":11})
m=[json.loads(l) for l in open("artifacts/cfd/manifest.jsonl")]
foils=["naca0008","n64012a","naca23018","naca63206"]

def get(f,fid): return sorted([x for x in m if x["foil"]==f and x["fidelity"]==fid], key=lambda z:z["alpha_deg"])

# ---- B7: 4-panel solver disagreement ----
fig,axs=plt.subplots(1,4,figsize=(15.0,4.0),sharey=True)
for ax,f in zip(axs,foils):
    p=get(f,"panel"); r=[x for x in get(f,"rans") if x.get("converged")]
    ax.plot([x["alpha_deg"] for x in p],[x["cl"] for x in p],"o-",color=GREY,lw=2,ms=6,
            label="panel method (cheap)")
    ax.plot([x["alpha_deg"] for x in r],[x["cl"] for x in r],"s",color=RUST,ms=9,
            label="RANS (expensive)")
    pa=np.array([x["alpha_deg"] for x in p]); pc=np.array([x["cl"] for x in p])
    for x in r:
        pv=np.interp(x["alpha_deg"],pa,pc)
        ax.plot([x["alpha_deg"]]*2,[pv,x["cl"]],color=RUST,lw=1.2,alpha=0.6)
    ax.set_title(f,fontsize=12); ax.grid(alpha=0.25); ax.set_xlabel("α [deg]")
    ax.set_xlim(-10,12)
axs[0].set_ylabel("$C_l$"); axs[0].legend(frameon=False,fontsize=10,loc="upper left")
fig.suptitle("The two solvers disagree, and they disagree the same way every time — that gap is what the network learns",
             fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.90])
fig.savefig(OUT+"B7_solver_gap.png",dpi=200); plt.close(fig); print("B7")

# ---- B8: cheap vs expensive trade ----
fig,axs=plt.subplots(1,3,figsize=(13.0,3.9))
lab=["panel","steady RANS","URANS"]; col=[GREY,RUST,TEAL]
data=[("cases in training",[1881,35,12],"{:,}"),
      ("seconds per case",[0.4,189,9566],"{:,.0f}"),
      ("total solver time (h)",[0.2,1.8,31.9],"{:.1f}")]
for ax,(t,v,fm) in zip(axs,data):
    b=ax.bar(lab,v,color=col); ax.set_yscale("log"); ax.set_title(t,fontsize=12)
    ax.set_ylim(min(v)*0.25, max(v)*6)
    for rr,vv in zip(b,v):
        ax.text(rr.get_x()+rr.get_width()/2, vv*1.45, fm.format(vv), ha="center",
                fontweight="bold", fontsize=11.5)
    ax.grid(axis="y",alpha=0.25); ax.set_axisbelow(True); ax.tick_params(labelsize=10)
fig.suptitle("Cheap data is everywhere, accurate data is not — so we use both instead of picking one",
             fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.89])
fig.savefig(OUT+"B8_cheap_vs_expensive.png",dpi=200); plt.close(fig); print("B8")

# ---- B9: one foil, gap shaded ----
f="naca0008"; p=get(f,"panel"); r=[x for x in get(f,"rans") if x.get("converged")]
pa=np.array([x["alpha_deg"] for x in p]); pc=np.array([x["cl"] for x in p])
fig,ax=plt.subplots(figsize=(7.4,5.2))
g=np.linspace(-8,8,200)
ax.plot(g,np.interp(g,pa,pc),color=GREY,lw=3,label="panel method — fast, 1,881 cases")
ra=[x["alpha_deg"] for x in r]; rc=[x["cl"] for x in r]
ax.plot(ra,rc,"s-",color=RUST,lw=2.4,ms=10,label="RANS — accurate, 35 cases")
ax.fill_between(ra,[np.interp(a,pa,pc) for a in ra],rc,color=RUST,alpha=0.15)
ps=np.polyfit(pa[(pa>=-8)&(pa<=8)],pc[(pa>=-8)&(pa<=8)],1)[0]
rs=np.polyfit(ra,rc,1)[0]
ax.annotate(f"panel lift slope {ps:.3f}/deg\nRANS lift slope {rs:.3f}/deg\n→ panel runs {ps/rs-1:+.0%} high, consistently",
            xy=(-7.5,0.55), fontsize=11.5, color=NAVY,
            bbox=dict(boxstyle="round,pad=0.5",fc="#EEF3F4",ec="none"))
ax.set_xlabel("angle of attack  α  [deg]"); ax.set_ylabel("$C_l$")
ax.set_title("NACA 0008: the cheap solver is wrong in a predictable way")
ax.legend(frameon=False,loc="lower right"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT+"B9_gap_one_foil.png",dpi=200); plt.close(fig); print("B9")
