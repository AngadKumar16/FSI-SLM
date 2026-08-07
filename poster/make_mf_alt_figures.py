import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "poster/generated/mf_alt/"
RUST="#C8552A"; TEAL="#1F7A6E"; NAVY="#204070"; GREY="#9AA5A8"; CYAN="#3FA3C4"
plt.rcParams.update({"font.size":11, "axes.titlesize":12})

rows = [json.loads(l) for l in open("artifacts/cfd/rows/urans_1785895725.jsonl")]
rows.sort(key=lambda r:(r["foil"], r["alpha_deg"]))

def save(fig, name):
    fig.savefig(OUT+name, dpi=200, bbox_inches="tight"); plt.close(fig); print("  ", name)

# ---------------- SLIDE 1 ALTERNATES ----------------
# A2 Cl-alpha, steady + URANS
steady_a = [-8,-4,0,2,4,6,8]
steady_cl = [-0.755,-0.315,0.075,0.175,0.475,0.645,0.745]
fig,ax=plt.subplots(figsize=(7.2,5.0))
ax.plot(steady_a, steady_cl, "s-", color="#3B6EA5", lw=2, ms=7, label="steady RANS (converged)")
for r in rows:
    st = r["statistically_steady"]
    ax.errorbar(r["alpha_deg"], r["cl"], yerr=r["cl_rms"], fmt="o", color=RUST,
                mfc=RUST if st else "white", ms=8, capsize=4, lw=1.5)
ax.axvspan(10,21, color=RUST, alpha=0.07)
ax.text(15.5, -0.55, "no steady solution\nexists here", ha="center", color=RUST, fontsize=11, style="italic")
ax.plot([],[],"o",color=RUST,label="URANS (bar = RMS fluctuation)")
ax.set_xlabel("angle of attack  α  [deg]"); ax.set_ylabel("$C_l$")
ax.set_title("Lift curve: steady RANS stops at 8°, URANS carries on")
ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.25); ax.set_xlim(-10,22)
save(fig,"A2_cl_alpha.png")

# A3 Strouhal
fig,ax=plt.subplots(figsize=(7.2,5.0))
ax.axhspan(0.15,0.25,color=GREY,alpha=0.30, label="bluff-body band, St ≈ 0.2")
foils = sorted({r["foil"] for r in rows})
cols = ["#C8552A","#204070","#1F7A6E","#B08A2E"]
for f,c in zip(foils, cols):
    rs=[r for r in rows if r["foil"]==f]
    ax.plot([r["alpha_deg"] for r in rs],[r["strouhal_projected"] for r in rs],
            "o-", color=c, ms=8, lw=1.8, label=f)
ax.set_xlabel("angle of attack  α  [deg]")
ax.set_ylabel("$St_d = f\\,d/U$,   $d=c(\\sin|α|+t/c)$")
ax.set_title("Measured shedding frequency, projected height")
ax.set_xticks([12,16,20]); ax.legend(frameon=False, fontsize=10); ax.grid(alpha=0.25)
save(fig,"A3_strouhal.png")

# A4 URANS velocity fields montage
sel = [("n64012a",12.0),("n64012a",16.0),("n64012a",20.0)]
fig,axs=plt.subplots(1,3,figsize=(13.5,4.4))
for a,(f,al) in zip(axs,sel):
    r=[x for x in rows if x["foil"]==f and x["alpha_deg"]==al][0]
    d=np.load(r["npz"].split("FSI-SLM/")[-1])
    u=d["target"][0].astype(np.float32); solid=d["solid_mask"].astype(bool)
    u=np.ma.masked_where(solid,u)
    im=a.imshow(u, origin="lower", cmap="RdYlBu_r", vmin=-0.3, vmax=1.6)
    a.set_title(f"{f}  α = {al:.0f}°   $St_d$ = {r['strouhal_projected']:.2f}", fontsize=11.5)
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("URANS time-averaged streamwise velocity: the separated wake grows with incidence",
             fontsize=13, fontweight="bold")
fig.colorbar(im, ax=axs, fraction=0.020, pad=0.015, label="$u/U_\\infty$")
save(fig,"A4_urans_fields.png")

# A5 Cl RMS fluctuation
fig,ax=plt.subplots(figsize=(7.2,5.0))
w=0.2
for i,(f,c) in enumerate(zip(foils,cols)):
    rs=[r for r in rows if r["foil"]==f]
    xs=np.arange(3)+(i-1.5)*w
    ax.bar(xs,[r["cl_rms"] for r in rs], w, color=c, label=f)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(["α = 12°","α = 16°","α = 20°"])
ax.set_ylabel("$C_l$ RMS fluctuation")
ax.set_title("Unsteadiness the steady solver cannot represent")
ax.legend(frameon=False, fontsize=10); ax.grid(axis="y",alpha=0.25); ax.set_axisbelow(True)
save(fig,"A5_cl_rms.png")

# A6 convergence tally
fig,ax=plt.subplots(figsize=(7.2,4.6))
cats=["steady RANS\nα ≤ 8°","steady RANS\nα ≥ 10°","URANS\nα ≥ 12°"]
good=[20,0,12]; bad=[0,8,0]
ax.bar(cats,good,color=TEAL,label="usable")
ax.bar(cats,bad,bottom=good,color=RUST,label="hit iteration cap")
for i,(g,b) in enumerate(zip(good,bad)):
    if g: ax.text(i,g/2,str(g),ha="center",va="center",color="white",fontweight="bold",fontsize=15)
    if b: ax.text(i,g+b/2,str(b),ha="center",va="center",color="white",fontweight="bold",fontsize=15)
ax.set_ylabel("cases"); ax.set_title("Why URANS: the steady solver quits past stall")
ax.legend(frameon=False); ax.grid(axis="y",alpha=0.25); ax.set_axisbelow(True)
save(fig,"A6_convergence_tally.png")

# ---------------- SLIDE 2 ALTERNATES ----------------
h=json.load(open("artifacts/sensor/run_mf/history.json"))
ep=[x["epoch"] for x in h]; tr=[x["train"]["loss"] for x in h]
va=[x["val"]["loss"] for x in h]; mae=[x["val"]["scalar_mae"] for x in h]

# B2 fidelity mix
fig,ax=plt.subplots(figsize=(7.2,5.0))
lab=["panel\n(inviscid)","steady RANS\n(OpenFOAM)","URANS\n(deep stall)"]; val=[1881,35,12]
b=ax.bar(lab,val,color=[GREY,RUST,TEAL])
ax.set_yscale("log"); ax.set_ylim(5,7000); ax.set_ylabel("training cases (log scale)")
for r,v in zip(b,val): ax.text(r.get_x()+r.get_width()/2, v*1.3, f"{v:,}", ha="center", fontweight="bold", fontsize=13)
ax.set_title("1,928 cases, three fidelities\nloss weights  panel = 1 · RANS = 8 · URANS = 8")
save(fig,"B2_fidelity_mix.png")

# B3 loss curves
fig,ax=plt.subplots(figsize=(7.2,5.0))
ax.plot(ep,tr,color=RUST,lw=2,label="train"); ax.plot(ep,va,color=TEAL,lw=2.4,label="val")
ax.axhline(min(va),ls=":",color=TEAL); ax.set_xlabel("epoch"); ax.set_ylabel("total loss")
ax.set_title(f"40 epochs, 23 min on one laptop GPU\nbest val loss {min(va):.4f}")
ax.legend(frameon=False); ax.grid(alpha=0.25)
save(fig,"B3_loss_curves.png")

# B4 scalar MAE
fig,ax=plt.subplots(figsize=(7.2,5.0))
ax.plot(ep,mae,color=NAVY,lw=2.6)
ax.fill_between(ep,mae,min(mae)*0.97,color=NAVY,alpha=0.12)
ax.set_xlabel("epoch"); ax.set_ylabel("validation scalar MAE")
ax.set_title(f"Force coefficients keep improving:  {mae[0]:.3f} → {mae[-1]:.3f}")
ax.grid(alpha=0.25)
save(fig,"B4_scalar_mae.png")

# B5 schematic
fig,ax=plt.subplots(figsize=(12.0,4.6)); ax.set_xlim(0,12); ax.set_ylim(0,4.6); ax.axis("off")
def box(x,y,w,hh,txt,fc,tc="white",fs=11,bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,hh,boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=fc,ec="none"))
    ax.text(x+w/2,y+hh/2,txt,ha="center",va="center",color=tc,fontsize=fs,
            fontweight="bold" if bold else "normal")
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,
                                 lw=1.6,color="#4A5A61"))
box(0.2,3.3,2.5,0.85,"panel + BL\n1,881 cases",GREY)
box(0.2,2.1,2.5,0.85,"steady RANS\n35 cases",RUST)
box(0.2,0.9,2.5,0.85,"URANS\n12 cases",TEAL)
box(3.5,2.0,2.1,1.4,"one tensor\ncontract\n13 channels",NAVY,fs=11)
for y in (3.72,2.52,1.32): arrow(2.75,y,3.45,2.7)
box(6.4,1.9,2.4,1.6,"U-Net\n11.59 M par.\nattention @ 8×8",CYAN,fs=11)
arrow(5.65,2.7,6.35,2.7)
box(5.9,0.45,3.4,0.9,"FiLM:  α · Re · fidelity code","#7A5EA8",fs=11)
arrow(7.6,1.4,7.6,1.85)
box(9.6,1.9,2.2,1.6,"fields + $C_l$, $C_d$,\n$C_m$, $w$, twist",TEAL,fs=11)
arrow(8.85,2.7,9.55,2.7)
ax.text(6.0,4.25,"Fidelity is an input, not a separate model — one network learns the offset between solvers",
        ha="center",fontsize=12.5,fontweight="bold",color="#1A1A1A")
save(fig,"B5_schematic.png")

# B6 split composition
fig,ax=plt.subplots(figsize=(7.6,4.4))
splits=["train\n(76 sections)","val\n(14 sections)","test\n(17 sections)"]
n=[1368,252,308]
b=ax.barh(splits,n,color=[TEAL,CYAN,RUST])
for r,v in zip(b,n): ax.text(v+18,r.get_y()+r.get_height()/2,f"{v:,}",va="center",fontweight="bold")
ax.set_xlabel("cases"); ax.set_xlim(0,1550)
ax.set_title("Splits hold out whole shapes, not random cases\nNACA 0009 is pinned to test and never seen in training")
ax.grid(axis="x",alpha=0.25); ax.set_axisbelow(True)
save(fig,"B6_splits.png")

# ---------------- SLIDE 3 ALTERNATES ----------------
FLD=["u","v","p","load $\\Delta C_p$","disp $w/c$"]
FB=[0.0259,0.3367,0.4973,0.1999,0.5764]; FT=[0.0216,0.2565,0.3504,0.0833,0.2344]
SC=["$C_l$","$C_d$","$C_m$","$w_{max}$","twist"]
SB=[0.2373,0.3690,0.6051,1.3353,1.0452]; ST=[0.0711,0.3486,0.5118,0.6458,1.0980]
PN=["$C_l$","$C_d$","$C_m$"]; PB=[0.0456,0.0677,0.0946]; PT=[0.0465,0.0798,0.0972]

def pair(ax,labels,base,tuned,ylab,title):
    x=np.arange(len(labels)); w=0.36
    ax.bar(x-w/2,base,w,color=RUST,label="panel-only")
    ax.bar(x+w/2,tuned,w,color=TEAL,label="+ high-fidelity")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(ylab); ax.set_title(title)
    top=max(max(base),max(tuned)); ax.set_ylim(0,top*1.28)
    for xi,bb,tt in zip(x,base,tuned):
        d=(tt-bb)/bb*100
        ax.text(xi,max(bb,tt)+top*0.045,f"{d:+.0f}%",ha="center",fontsize=10.5,
                fontweight="bold",color=TEAL if d<0 else RUST)
    ax.grid(axis="y",alpha=0.25); ax.set_axisbelow(True)

fig,ax=plt.subplots(figsize=(7.4,5.0))
pair(ax,FLD,FB,FT,"relative RMSE vs RANS","Field channels, 7 held-out RANS test cases")
ax.legend(frameon=False,loc="upper left"); save(fig,"C3_fields.png")

fig,ax=plt.subplots(figsize=(7.4,5.0))
pair(ax,SC,SB,ST,"nRMSE vs RANS","Force + structure scalars  (n = 7, use nRMSE not $R^2$)")
ax.legend(frameon=False,loc="upper left"); save(fig,"C4_scalars.png")

fig,ax=plt.subplots(figsize=(7.4,5.0))
pair(ax,PN,PB,PT,"nRMSE vs panel  (n = 293)","The cheap fidelity did not collapse")
ax.legend(frameon=False,loc="upper left")
ax.text(0.5,0.80,"worst drift 0.012 absolute",transform=ax.transAxes,ha="center",
        fontsize=11,style="italic",color="#5B6B72")
save(fig,"C5_panel_preserved.png")

# C2 dumbbell
names=["$C_l$ nRMSE","displacement","surface load","$w_{max}$","p field","$C_m$","v field","u field","$C_d$","twist"]
b=[0.2373,0.5764,0.1999,1.3353,0.4973,0.6051,0.3367,0.0259,0.3690,1.0452]
t=[0.0711,0.2344,0.0833,0.6458,0.3504,0.5118,0.2565,0.0216,0.3486,1.0980]
rel=[(x/y) for x,y in zip(t,b)]
order=np.argsort(rel)
fig,ax=plt.subplots(figsize=(7.8,5.6))
for i,k in enumerate(order):
    good = rel[k] < 1
    ax.plot([b[k],t[k]],[i,i],color=TEAL if good else RUST,lw=2.5,zorder=1)
    ax.scatter(b[k],i,s=70,color=RUST,zorder=2)
    ax.scatter(t[k],i,s=70,color=TEAL,zorder=2)
    ax.text(max(b[k],t[k])+0.03,i,f"{rel[k]:.2f}×",va="center",fontsize=10,
            fontweight="bold",color=TEAL if good else RUST)
ax.set_yticks(range(len(order))); ax.set_yticklabels([names[k] for k in order])
ax.set_xlabel("error vs RANS (lower is better)")
ax.set_xlim(0,1.65)
ax.scatter([],[],color=RUST,label="panel-only"); ax.scatter([],[],color=TEAL,label="+ high-fidelity")
ax.legend(frameon=False,loc="lower right")
ax.set_title("Every quantity, baseline → fine-tuned")
ax.grid(axis="x",alpha=0.25); ax.set_axisbelow(True)
save(fig,"C2_dumbbell.png")

# C6 headline
fig,ax=plt.subplots(figsize=(7.6,4.6)); ax.axis("off")
cards=[("−70%","$C_l$ error vs RANS",TEAL),("−59%","tip displacement",TEAL),
       ("−58%","surface load",TEAL),("< 3%","panel drift",RUST)]
for i,(big,small,c) in enumerate(cards):
    x=(i%2)*0.52; y=0.52-(i//2)*0.52
    ax.add_patch(FancyBboxPatch((x,y),0.44,0.40,boxstyle="round,pad=0.01,rounding_size=0.03",
                                transform=ax.transAxes,fc="#EEF3F4",ec="none"))
    ax.text(x+0.04,y+0.24,big,transform=ax.transAxes,fontsize=38,fontweight="bold",color=c)
    ax.text(x+0.045,y+0.10,small,transform=ax.transAxes,fontsize=13,color="#5B6B72")
ax.text(0.5,1.02,"47 high-fidelity cases, 23 minutes of training",transform=ax.transAxes,
        ha="center",fontsize=14,fontweight="bold")
save(fig,"C6_headline.png")
print("done")
