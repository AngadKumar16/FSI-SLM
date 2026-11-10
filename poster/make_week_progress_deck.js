const pptxgen = require("pptxgenjs");
const G = require("path").join(__dirname, "generated") + "/";
const A = G + "mf_alt/";
const CY = "3FA3C4", INK = "1A1A1A";

const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
p.author = "Angad Kumar";
p.title = "URANS + multi-fidelity U-Net — Week progress";

function slide(title, bullets, nextStep, img, imgBox, notes) {
  const s = p.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText(title, { x:0.6, y:0.30, w:12.1, h:0.85, fontSize:40, color:CY,
    fontFace:"Calibri", margin:0, valign:"middle" });

  s.addText(bullets.map((b,i) => ({ text:"-  " + b,
    options:{ breakLine: i < bullets.length-1, paraSpaceAfter:13 } })),
    { x:0.7, y:1.40, w:5.05, h:4.35, fontSize:15.5, color:INK, fontFace:"Calibri",
      margin:0, valign:"top", lineSpacing:21 });

  s.addText([{ text:"Next Step: ", options:{ bold:true } }, { text:nextStep }],
    { x:0.7, y:5.90, w:5.05, h:1.2, fontSize:17, color:INK, fontFace:"Calibri",
      margin:0, valign:"top", lineSpacing:24 });

  s.addImage(Object.assign({ path: img }, imgBox));
  s.addNotes(notes);
  return s;
}

// ---- 1 ----
slide("Why We Needed Unsteady CFD",
  [
    "Below 8° the flow stays glued to the foil. It settles down, so the steady solver finds one answer and stops.",
    "Past 10° the flow rips off the back and flaps around. There is no single steady answer to find — so the solver never stops. That is physics, not a bad setup.",
    "Fix: stop looking for one answer. URANS marches the flow forward in time and averages what it sees.",
    "Ran 12 cases — 4 airfoils at 12°, 16° and 20°, about 32 hours of compute.",
    "Sanity check: the wake sheds at roughly the same rate as flow past a cylinder (St ≈ 0.2). Real physics, not solver noise.",
  ],
  "Feed all 47 high-fidelity cases into the U-Net alongside the panel data",
  A + "A2_cl_alpha.png", { x:6.35, y:1.35, w:6.35, h:4.40 },
  "Plain version: steady solvers assume the flow eventually stops changing. Past stall it never does - the separated wake keeps shedding - so the residual never drops and the run hits the iteration cap. URANS solves the same equations but in time, then averages. The Strouhal number check is how we know the shedding is real: 0.2 is the textbook value for flow past a bluff body."
);

// ---- 2 ----
slide("Teaching One Network Both Solvers",
  [
    "Two kinds of data. The panel method is fast and rough — 1,881 cases. RANS and URANS are accurate but slow — only 47 cases, weeks of compute.",
    "Instead of two models, one. We tell the network which solver each case came from, so it learns how the two disagree rather than blurring them together.",
    "The 47 expensive cases count 8× as much when the network is scored, otherwise 1,881 cheap ones drown them out.",
    "11.59 M weights, 40 passes over the data, 23 minutes on one laptop. No cluster.",
  ],
  "Test it on an airfoil it has never seen and compare against the slow solver",
  A + "B5_schematic.png", { x:6.20, y:1.60, w:6.65, h:2.55 },
  "Plain version: mixing cheap and expensive training data usually makes a model average the two into something wrong for both. Feeding the fidelity in as a conditioning signal (FiLM) lets the network keep them apart and learn the offset. The 8x weight is what stops 47 cases from being statistically invisible next to 1,881."
).addImage({ path: A + "B2_fidelity_mix.png", x:7.35, y:4.20, w:4.35, h:2.85 });

// ---- 3 ----
slide("Did the Expensive Data Pay Off?",
  [
    "Tested on NACA 0009 — an airfoil kept out of training entirely — against 7 slow-solver cases.",
    "Lift error dropped 70%. Tip deflection 59%. Surface pressure load 58%.",
    "It did not forget the cheap data: accuracy on the 293 panel test cases moved by under 3%.",
    "Honest bits: twist got 5% worse, and no unsteady cases landed in the test set, so that part is not proven yet.",
  ],
  "Swap the simulated experiment for real PIV data; run URANS on a held-out airfoil",
  A + "C2_dumbbell.png", { x:6.35, y:1.35, w:6.35, h:4.55 },
  "Each line is one quantity; the dot moves from panel-only to fine-tuned, and the number is the ratio. Below 1.0 is better. Only twist moved the wrong way. Note n = 7, so quote the error directly rather than R-squared - the sample is too small for a ratio like that to mean anything."
);

p.writeFile({ fileName: require("path").join(__dirname, "..", "docs", "slides", "FSI-SLM_week_progress_3slides.pptx") })
 .then(f => console.log("wrote", f));
