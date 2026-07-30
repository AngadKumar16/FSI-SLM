# AGU Bright STaRS Poster — Layout & Build Plan
### FSI-SLM: an end-to-end digital twin for fluid–structure interaction

---

## 1. Print specs (AGU Fall Meeting)

- **Board size:** 6 ft wide × 4 ft high (**landscape**). Usable area ≈ 70 in × 46 in.
- **Design your file at:** 72 in × 48 in (or 48 in × 36 in scaled up at print) — landscape.
- **Export:** PDF at 150–200 DPI at full size. All figures here are 200 DPI PNGs and hold up well when placed at the sizes below.
- **Readability rule:** body text ≥ 24 pt, section headers ≥ 48 pt, title ≥ 90 pt. A viewer should read everything from 5–6 feet away.
- **Golden rule for Bright STaRS judges:** the poster should tell the story in ~60 seconds of skimming, before you say a word.

---

## 2. Color scheme (used across all figures — match your text/boxes to these)

| Role | Hex | Use |
|---|---|---|
| Navy (primary) | `#0B2545` | Title bar, section headers |
| Blue | `#134074` | Accents |
| Teal | `#1B9AAA` | Experiment / physical side |
| Coral | `#EF6F45` | Model / AI side |
| Gold | `#F2B134` | Data / FeatureRecord |
| Green | `#3DA35D` | Feature extraction / good results |
| Purple | `#6C5CE7` | Outputs |
| Sand (bg) | `#F6F4EF` | Panel backgrounds |

Keep the background white or very light sand. Use color for the boxes/figures, not big color fields behind text.

---

## 3. Overall grid

```
┌───────────────────────────────────────────────────────────────────────┐
│  TITLE BAR  (full width)   — title, your name, school, mentor, QR code  │
├───────────────────────────────────────────────────────────────────────┤
│  METRIC CARDS STRIP (full width)  — fig_metric_cards.png                │
├────────────────────┬────────────────────┬─────────────────────────────┤
│  COLUMN 1           │  COLUMN 2          │  COLUMN 3                    │
│  Why it matters     │  How it works      │  Results & what's next      │
└────────────────────┴────────────────────┴─────────────────────────────┘
```

Three equal columns (~22 in each with gutters). Read top-to-bottom, left-to-right.

---

## 4. Section-by-section content

### TITLE BAR (full width, ~6 in tall)
- **Title (90–110 pt):** "FSI-SLM: An End-to-End Digital Twin that Reads Fluid–Structure Interaction Experiments"
- **Subtitle (44 pt):** "A U-Net flow sensor + a small language model that interprets the results — all on one laptop."
- **Line under it (28 pt):** Your name · Your school · Mentor name · Program. Add a **QR code** to the GitHub repo (github.com/AngadKumar16/FSI-SLM) in the top-right corner.

### METRIC CARDS STRIP (full width, ~5 in tall)
- **Figure:** `fig_metric_cards.png` (place edge-to-edge).
- These five numbers (1,926 cases · 5 regimes · 78% macro-F1 · 100% lock-in F1 · 1 laptop) are your headline. Judges remember numbers.

---

### COLUMN 1 — "Why this matters"

**1.1 The problem (header 48 pt, ~50 words)**
> Engineers care about fluid–structure interaction (FSI): how flowing water or air makes a structure vibrate, and when that vibration causes failure. Reading FSI data takes an expert. We built a system that reads it automatically and answers questions about it.

**1.2 Figure — the concept:** `fig_twin.png`
Caption: "Measure the hydrofoil once; then ask the digital twin anything."

**1.3 Figure — the five regimes:** `fig_regimes.png`
Caption: "The five flow/vibration regimes our system tells apart — from a quiet attached flow to violent stall flutter."

**1.4 One-line explainer of VIV (optional, if space):** `fig_viv_lockin.png`
Caption: "Lock-in: when vortex shedding synchronizes with the structure and vibration spikes."

---

### COLUMN 2 — "How it works"

**2.1 System overview (header, ~40 words)**
> Raw experiment and simulation data become a compact JSON summary. A small language model, fine-tuned on realistic simulated experiments, reads that summary, interprets the physics, and flags likely sensor noise — grounded in real papers via retrieval.

**2.2 Figure — the pipeline (hero of this column):** `fig_system.png`
Caption: "From hydrofoil to answer. Everything runs on one laptop, served over a REST API and an MCP server."

**2.3 Figure — the U-Net flow sensor:** `fig_unet.png`
Caption: "A convolutional U-Net predicts full velocity/pressure fields and lift & drag from just the shape and flow condition — replacing a slow CFD solve."

**2.4 Figure — geometry encoding:** `fig_sdf_channels.png`
Caption: "How the airfoil enters the network: a signed-distance field plus flow-condition channels."

**2.5 Figure — training + RAG (pick one or stack both if space):**
`fig_training.png` — "How the language model is trained."
`fig_rag.png` — "Retrieval keeps answers grounded in real literature."

---

### COLUMN 3 — "Results & what's next"

**3.1 Flow the sensor learns (header, ~30 words)**
> The U-Net targets come from real CFD. Validated on NACA 0009 (α = 8°, Re = 5×10⁵): correct leading-edge suction, stagnation, and lift (Cl ≈ 0.73).

**3.2 Figure — flow fields:** `fig_flowfield.png`
Caption: "Inviscid pressure & velocity around NACA 0009 — the flow structure the sensor reproduces."
(Alternative/companion: the real CFD figure `poster_fig1_flowfields.png` already in this folder.)

**3.3 Figure — lift curve:** `fig_cl_alpha.png`
Caption: "Lift vs. angle of attack, through stall — the U-Net's regression target."

**3.4 Figure — classification results:** `fig_confusion.png` + `fig_perclass.png`
Caption: "On held-out tests the interpreter nails attached, separated, and lock-in cases; galloping vs. stall-flutter is the current confusion (small sample)."

**3.5 Figure — the dataset:** `fig_dataset_scatter.png`
Caption: "The five regimes occupy distinct regions of (Re, St) and (U*, α) space — physically consistent by construction."

**3.6 Conclusions (header, bullet-free prose ~40 words)**
> A full FSI digital twin — sensor + interpreter — that runs without a GPU cluster. It turns raw experiments into plain-language answers and stays grounded in the literature.

**3.7 What's next (~30 words)**
> Train the U-Net across all 1,926 cases and check the predicted Cl–α curve against XFOIL through stall; connect the live PIV rig; add load & displacement prediction for two-way FSI.

**3.8 Footer:** QR to repo, acknowledgements, a one-line "Built with Python, MLX, OpenFOAM, FAISS."

---

## 5. Figure inventory (all in `poster/generated/`, 200 DPI)

| File | What it shows | Best placement |
|---|---|---|
| `fig_metric_cards.png` | 5 headline numbers | Title strip |
| `fig_twin.png` | Digital-twin concept loop | Col 1 |
| `fig_regimes.png` | 5 FSI regimes explained | Col 1 |
| `fig_viv_lockin.png` | Lock-in frequency plot | Col 1 (optional) |
| `fig_system.png` | End-to-end pipeline | Col 2 (hero) |
| `fig_unet.png` | U-Net architecture | Col 2 |
| `fig_sdf_channels.png` | SDF + input channels | Col 2 |
| `fig_training.png` | SLM training pipeline | Col 2 |
| `fig_rag.png` | RAG retrieval flow | Col 2 |
| `fig_flowfield.png` | Velocity + pressure fields | Col 3 (hero) |
| `fig_cp_surface.png` | Surface pressure vs α | Col 3 (optional) |
| `fig_cl_alpha.png` | Lift curve w/ stall | Col 3 |
| `fig_confusion.png` | Confusion matrix | Col 3 |
| `fig_perclass.png` | Precision/recall/F1 | Col 3 |
| `fig_perfield.png` | Numeric field accuracy | Col 3 (optional) |
| `fig_gallery.png` | 96-airfoil shape library | Col 2 or 3 |
| `fig_dataset_scatter.png` | Regimes in parameter space | Col 3 |

Existing real-CFD figures also available: `poster_fig1_flowfields.png`, `poster_fig3_mesh.png`, `poster_fig4_channels.png`.

---

## 6. Honest-claims checklist (so a judge can't trip you up)

- The **U-Net is specified and fed but not yet trained across the full set** — say "the data pipeline is built and validated on one case; training the full sensor is the current step." The `fig_cl_alpha` and `fig_flowfield` are analytic/CFD references for what it will predict, not U-Net outputs yet. Label them as such.
- The **classification results are from the rule-based baseline on a 20-case held-out set** — small sample. Present F1 = 0.78 honestly and note galloping/stall-flutter is the hard pair.
- **"Runs on a laptop"** is true and is your strongest, most quotable line.

---

## 7. Build tips

- Build in **PowerPoint or Canva** at 48×36 (then print at 200%) or directly at 72×48. Keep one master text style per level.
- Leave **whitespace** — do not fill every inch. Empty space reads as confidence.
- Number your figures F1…Fn and refer to them in the text so the eye has a path.
- Print a **letter-size handout** version of the same poster to give judges.
