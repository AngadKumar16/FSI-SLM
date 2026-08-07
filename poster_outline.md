# Poster Outline — FSI Digital Twin + SLM
**Format:** AGU 60×40 in, 3 columns (template layout)

**Title:** A Digital Twin for Fluid–Structure Interaction: Small Language Models + Surrogate Modeling for Airfoil Analysis

**Authors:** *Kumar, Angad (Obra Tompkins HS, TX) · Shukla, Khemraj (Brown University, RI)

**Banner strip (green bar):** One-line hook — "PIV + simulation + SLM + U-Net = end-to-end digital twin that runs on a laptop."

---

## COLUMN 1

### MOTIVATION
- FSI around structures drives structural failure. Interpreting FSI data needs expert.
- Expert bottleneck: raw PIV/CFD fields → meaning is slow, manual, inconsistent.
- Sensor noise vs real flow physics hard to separate.
- LLMs hallucinate on physics; big models need GPU clusters. Need small, grounded, local.

### OBJECTIVES
1. Turn raw FSI experiment (PIV) + simulation into machine-readable summary.
2. Fine-tune SLM to interpret flow physics + flag sensor noise.
3. Ground answers in published fluid-dynamics literature via RAG → cut hallucination.
4. Serve pipeline on ordinary laptop — REST API (FastAPI) + MCP server for agentic workflows.
5. Replace CFD/experiment with U-Net surrogate → full end-to-end digital twin.

### DATA
- **Experimental:** NACA 0009 airfoil, PIV system, multiple angles of attack (AoA).
- **Simulation:** matched CFD cases; realistic simulated experiments for SLM fine-tuning.
- **Surrogate training set:** 107 airfoil shapes × 18 AoA = **1,926 cases**.
- **Fields per case:** velocity field, C_L, C_D, load field, displacement field.
- **SLM summary artifact:** JSON — geometry, flow conditions, derived quantities, QC flags.
- *Figure:* PIV rig photo + sample velocity field + JSON snippet side by side.

---

## COLUMN 2

### METHODS
**Pipeline (left→right flow diagram, hero figure of poster):**

`PIV experiment + CFD → feature extraction → JSON summary → fine-tuned SLM (+RAG) → interpretation / noise flags → REST API + MCP → agent`

1. **Data → JSON.** Reduce raw fields to structured summary (conditions, wake metrics, separation indicators, coefficients).
2. **SLM fine-tune.** Train on realistic simulated experiments so model learns physics-consistent readings, not generic text.
3. **RAG.** Retrieve from published fluid-dynamics literature; answers cite retrieved passages → hallucination control.
4. **Noise detection.** Model flags physically implausible vectors / regions as likely sensor artifacts.
5. **U-Net surrogate.** Input = airfoil geometry + flow conditions (SDF/mask + AoA, Re). Output = velocity field, C_L, C_D, load + displacement fields. Replaces solver call.
6. **Serving.** FastAPI REST endpoints + MCP server → callable from agentic LLM workflows. CPU-only, laptop-grade.

*Figures:* U-Net architecture block; API/MCP call schematic.

### ANALOGY: WHY SMALL + GROUNDED
- Big LLM = giant reference library, no lab notebook → confident wrong answers, needs data center.
- Our SLM = junior lab partner trained on THIS rig, with the textbook open (RAG) on the bench.
- U-Net = "learned wind tunnel": once trained, a new case is a lookup, not an overnight solve.
- Net effect: expert-level reading of an experiment, on the machine already sitting on the bench.

---

## COLUMN 3

### RESULTS
*(fill with your numbers — suggested panels)*
- **R1.** U-Net predicted vs CFD velocity field, side-by-side + error map (held-out airfoils).
- **R2.** C_L and C_D: predicted vs ground truth scatter, R² / mean error vs AoA.
- **R3.** Load + displacement field accuracy for representative NACA 0009 AoA sweep.
- **R4.** SLM interpretation accuracy: correct physics call rate, with vs without RAG.
- **R5.** Noise-flagging: detection rate / false-alarm rate on injected sensor noise.
- **R6.** Cost: surrogate inference time vs CFD runtime (speedup ×), laptop CPU, peak RAM.
- **R7.** Example transcript — JSON in, SLM answer out, on a real PIV case.

### SUMMARY
- End-to-end digital twin: experiment + simulation + language interpretation in one pipeline.
- Fine-tuned SLM + RAG reads FSI data, explains flow physics, flags sensor noise — grounded, not hallucinated.
- U-Net trained on 1,926 cases predicts fields, C_L/C_D, load, displacement directly from geometry + conditions, removing the solver from the loop.
- Runs entirely on a laptop; exposed via FastAPI REST and MCP for agentic AI.

### FUTURE WORK
- Extend beyond NACA 0009 → cambered / thick / multi-element geometries; higher Re, transonic.
- Unsteady + time-resolved FSI (vortex shedding, flutter), not steady snapshots.
- Uncertainty quantification on U-Net outputs; conformal bounds on SLM claims.
- Closed-loop agent: SLM proposes next AoA / test condition, runs surrogate, updates twin.
- Real-time in-the-loop deployment during PIV acquisition.

### REFERENCES
*(replace template refs)*
- Ronneberger, Fischer, Brox — U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015.
- Thuerey et al. — Deep Learning Methods for RANS Simulations of Airfoil Flows, AIAA J. 2020.
- Lewis et al. — Retrieval-Augmented Generation for Knowledge-Intensive NLP, NeurIPS 2020.
- Raissi, Perdikaris, Karniadakis — Physics-Informed Neural Networks, JCP 2019.
- Anthropic — Model Context Protocol specification.
- Abbott & von Doenhoff — Theory of Wing Sections.

### ACKNOWLEDGEMENT
Advisor Khemraj Shukla (Brown University). Lab / facility providing PIV access. Funding source.

---

## LAYOUT NOTES
- Hero = pipeline diagram, top of Column 2, full column width.
- Results column: R1/R2 large; R4–R6 small stat callouts (big number + one-line label).
- Bullets ≤ 12 words. Body ≥ 28 pt. Section headers keep template coral.
- QR code bottom-right → repo / API docs / demo.
- Abstract text itself does NOT go on poster — it becomes Motivation + Methods bullets.
