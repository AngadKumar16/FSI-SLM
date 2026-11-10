const pptxgen = require("pptxgenjs");
const FIG = require("path").join(__dirname, "generated") + "/";

const NAVY = "0F2B3D", TEAL = "1F7A6E", RUST = "C8552A",
      INK = "1A1A1A", MUTE = "5B6B72", PAPER = "FFFFFF", TINT = "EEF3F4";

const p = new pptxgen();
p.layout = "LAYOUT_WIDE";           // 13.3 x 7.5
p.author = "Angad Kumar";
p.title = "FSI-SLM — URANS, multi-fidelity U-Net";

// ---------- helpers ----------
function header(s, kicker, title) {
  s.addShape(p.ShapeType.rect, { x:0, y:0, w:13.33, h:1.32, fill:{ color:NAVY } });
  s.addText(kicker, { x:0.55, y:0.16, w:8, h:0.3, fontSize:12, bold:true,
    color:"7FB8C4", charSpacing:2, fontFace:"Arial", margin:0 });
  s.addText(title, { x:0.55, y:0.46, w:12.2, h:0.7, fontSize:30, bold:true,
    color:PAPER, fontFace:"Cambria", margin:0 });
}

function stats(s, y, items) {
  const n = items.length, gap = 0.28, W = (12.23 - gap*(n-1)) / n;
  items.forEach((it, i) => {
    const x = 0.55 + i*(W+gap);
    s.addShape(p.ShapeType.roundRect, { x, y, w:W, h:1.28, rectRadius:0.08,
      fill:{ color:TINT }, line:{ color:TINT } });
    s.addText(it[0], { x:x+0.16, y:y+0.10, w:W-0.32, h:0.56, fontSize:26, bold:true,
      color: it[2] || TEAL, fontFace:"Arial", margin:0, valign:"middle" });
    s.addText(it[1], { x:x+0.16, y:y+0.70, w:W-0.32, h:0.48, fontSize:11.5,
      color:MUTE, fontFace:"Calibri", margin:0, valign:"top" });
  });
}

function caption(s, y, txt) {
  s.addText(txt, { x:0.55, y, w:12.23, h:0.5, fontSize:12.5, color:MUTE,
    fontFace:"Calibri", italic:true, margin:0 });
}

// ================= SLIDE 1 — URANS =================
{
  const s = p.addSlide();
  header(s, "STEP 1 — HIGH-FIDELITY DATA", "Past stall, the steady solver has nothing to converge to");

  s.addImage({ path: FIG+"fig_urans_shedding.png", x:0.55, y:1.62, w:12.23, h:3.35 });
  caption(s, 5.02, "12 URANS cases · pimpleFoam, k-ω SST · 4 training-split sections at α = 12°, 16°, 20° · 31.9 solver-hours. NACA 0009 stays in test.");

  stats(s, 5.62, [
    ["12 cases",   "URANS at deep stall, 4 of them statistically settled", RUST],
    ["St_d ≈ 0.2", "measured shedding lands in the bluff-body band — physics, not a fit"],
    ["α ≥ 10°", "where every steady RANS run hit the iteration cap", RUST],
    ["+20 RANS",   "steady cases below stall added in the same sweep"],
  ]);
  s.addNotes("Steady RANS is valid to alpha 8. Above that there is no fixed point, so every steady case stalled 2-3 decades above tolerance. URANS gives a time-resolved answer AND a measurable shedding frequency. St_d in the 0.15-0.25 bluff-body band is the sanity check that the unsteady runs are real.");
}

// ================= SLIDE 2 — TRAINING =================
{
  const s = p.addSlide();
  header(s, "STEP 2 — MULTI-FIDELITY TRAINING", "One U-Net, three solvers, fidelity as a conditioning input");

  s.addImage({ path: FIG+"fig_mf_training.png", x:0.55, y:1.62, w:12.23, h:3.37 });
  caption(s, 5.04, "13-channel input (SDF + panel prior); α, Re and the fidelity code enter by FiLM, not as image channels. High-fidelity rows carry 8× loss weight so 47 cases can move a 1,928-case fit.");

  stats(s, 5.62, [
    ["1,928",   "cases · 107 sections · 18 incidences"],
    ["11.59 M", "parameters, self-attention at the 8×8 bottleneck"],
    ["8×",      "loss weight on RANS + URANS rows", RUST],
    ["23 min",  "40 epochs, MPS, one laptop — no cluster"],
  ]);
  s.addNotes("The point of the fidelity code: the network is told which solver produced a row, so it can learn a systematic offset instead of averaging the two into mush. Val loss is flat because the val split is 93% panel - the panel task was already solved. The signal to watch is the scalar MAE, which halves.");
}

// ================= SLIDE 3 — RESULT =================
{
  const s = p.addSlide();
  header(s, "STEP 3 — RESULT", "47 high-fidelity cases close the transfer gap");

  s.addImage({ path: FIG+"fig_mf_gain_clean.png", x:0.55, y:1.62, w:12.23, h:3.29 });
  caption(s, 4.88, "Same checkpoint pair, evaluated on 7 converged RANS test cases (NACA 0009, held out by shape) and on 293 panel test cases.");

  stats(s, 5.42, [
    ["−70%",  "Cl nRMSE vs RANS: 0.237 → 0.071", TEAL],
    ["−59%",  "displacement rel. RMSE: 0.576 → 0.234", TEAL],
    ["−58%",  "surface load rel. RMSE: 0.200 → 0.083", TEAL],
    ["< 3%",   "panel accuracy drift — no catastrophic forgetting", RUST],
  ]);

  s.addText([
    { text:"Caveat: ", options:{ bold:true, color:RUST } },
    { text:"no URANS rows land in the test split — the URANS claim rests on training rows and the val split. Twist is the one channel that got worse (+5%).",
      options:{ color:MUTE } },
  ], { x:0.55, y:6.82, w:12.23, h:0.4, fontSize:11.5, fontFace:"Calibri", margin:0 });

  s.addNotes("Say the caveat before a judge finds it. Also: n=7, so R^2 is not meaningful on this sample - report nRMSE. Panel channels within 3 percent proves the fine-tune did not trade the cheap fidelity away.");
}

p.writeFile({ fileName: require("path").join(__dirname, "..", "docs", "slides", "FSI-SLM_multifidelity.pptx") })
 .then(f => console.log("wrote", f));
