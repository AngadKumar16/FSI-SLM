import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="poster/generated/mf_alt/"
RUST="#C8552A"; TEAL="#1F7A6E"; GREY="#9AA5A8"; NAVY="#204070"
plt.rcParams.update({"font.size":11})
m=[json.loads(l) for l in open("artifacts/cfd/manifest.jsonl")]
foils=["naca0008","n64012a","naca23018","naca63206"]
cols=[RUST,NAVY,TEAL,"#B08A2E"]
g=lambda f,fid: sorted([x for x in m if x["foil"]==f and x["fidelity"]==fid],key=lambda z:z["alpha_deg"])

fig,axs=plt.subplots(1,2,figsize=(13.2,4.6))

# (a) one foil, curves + shaded gap
f="naca23018"; p=g(f,"panel"); r=[x for x in g(f,"rans") if x.get("converged")]
pa=np.array([x["alpha_deg"] for x in p]); pc=np.array([x["cl"] for x in p])
ra=np.array([x["alpha_deg"] for x in r]); rc=np.array([x["cl"] for x in r])
gg=np.linspace(-8,8,200)
axs[0].plot(gg,np.interp(gg,pa,pc),color=GREY,lw=3.2,label="panel method (cheap, 1,881 cases)")
axs[0].plot(ra,rc,"s-",color=RUST,lw=2.4,ms=10,label="RANS (accurate, 35 cases)")
axs[0].fill_between(ra,np.interp(ra,pa,pc),rc,color=RUST,alpha=0.18)
axs[0].annotate("past 0° the panel method\nover-predicts lift, and the gap\nwidens as the flow separates",
    xy=(6.6,0.62), xytext=(-7.4,0.78), fontsize=11.5, color=NAVY,
    arrowprops=dict(arrowstyle="->",color=NAVY,lw=1.4,connectionstyle="arc3,rad=-0.2"))
axs[0].set_xlabel("angle of attack  α  [deg]"); axs[0].set_ylabel("$C_l$")
axs[0].set_title("(a) NACA 23018 — the two solvers side by side",fontsize=12.5)
axs[0].legend(frameon=False,loc="lower right",fontsize=10.5); axs[0].grid(alpha=0.25)

# (b) gap vs alpha, all foils
axs[1].axhline(0,color="#333333",lw=1.2)
for f,c in zip(foils,cols):
    p=g(f,"panel"); r=[x for x in g(f,"rans") if x.get("converged")]
    pa=np.array([x["alpha_deg"] for x in p]); pc=np.array([x["cl"] for x in p])
    ra=np.array([x["alpha_deg"] for x in r]); rc=np.array([x["cl"] for x in r])
    axs[1].plot(ra, np.interp(ra,pa,pc)-rc, "o-", color=c, ms=8, lw=2, label=f)
axs[1].set_xlabel("angle of attack  α  [deg]")
axs[1].set_ylabel("panel $C_l$  −  RANS $C_l$")
axs[1].set_title("(b) the gap is systematic, not scatter — it widens with angle",fontsize=12.5)
axs[1].legend(frameon=False,fontsize=10.5,loc="upper left"); axs[1].grid(alpha=0.25)
axs[1].text(0.97,0.06,"this offset is what the network learns",transform=axs[1].transAxes,
            ha="right",fontsize=11.5,style="italic",color=NAVY)

fig.suptitle("Why we cannot just pool the data: the cheap solver is wrong in a predictable way",
             fontsize=13.5,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.91])
fig.savefig(OUT+"B11_solver_gap.png",dpi=200); print("ok")
