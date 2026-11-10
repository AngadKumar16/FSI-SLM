# Poster build record — AGU 60 × 40 in

**Deliverable:** `poster/output/FSI-SLM_AGU_poster_60x40.pdf` (4320 × 2880 pt = exactly 60 × 40 in, 1 page)
**Screen preview:** `poster/output/FSI-SLM_AGU_poster_preview.png` (6000 × 4000, 100 dpi)

**Title:** *A Fine-Tuned Small Language Model for Interpreting Fluid–Structure
Interaction Experiments* — retitled to match the submitted abstract, which leads
with the language model rather than the twin.

Rebuild everything:

```bash
python3 -m paper.figures --out paper/figures        # F2, F4, F6, F8 sources
python3 poster/figure_scripts/fig_diagrams.py       # F1 concept figure
python3 poster/figure_scripts/fig_poster_2026.py    # F7, live from JSON
python3 poster/make_new_figs.py                     # F5  (new)
python3 poster/make_piv_fig.py                      # F3  (new)
python3 poster/build_poster.py                      # the board
```

---

## What's on the board

| Fig | File | Column | Shows |
|---|---|---|---|
| F1 | `poster/generated/fig_regimes.png` | 1 · Motivation | the five FSI regimes the interpreter separates |
| F2 | `paper/figures/fig2_dataset.png` | 1 · Data | 107 sections split by shape; angle sampling; fidelity mix |
| F3 | `poster/generated/fig_p_realpiv.png` | 1 · Real PIV | **new** — the measured NACA 0009 run, end to end |
| F4 | `paper/figures/fig1_system.png` | 2 · Methods | the twin end to end (hero figure) |
| F5 | `poster/generated/fig_p_serve.png` | 2 · Methods | **new** — RAG index, retrieval, and the three front doors |
| F6 | `paper/figures/fig5_fields.png` | 3 · Results | solver vs surrogate vs error, held-out NACA 0009 |
| F7 | `poster/generated/fig_mf_gain.png` | 3 · Results | multi-fidelity fine-tuning closes the transfer gap |
| F8 | `paper/figures/fig9_interpreter.png` | 3 · Results | LoRA Phi-4-mini on 50 held-out records |

Headline numbers live in the lime strip. A QR to the repository sits under the
AGU mark in the header; it decodes at print size.

### F3 is new, and it is the poster's answer to "where is the experiment?"

The title says *Experiments* and the abstract promises PIV. The previous draft
showed only the dashed `PIV rig / not connected` edge, which reads as though no
measurement exists. It does: `results/real_piv/` holds a worked NACA 0009 at
**AoA 17.2°** over 200 DaVis frames. F3 puts it on the board — measured wake TKE
and shedding spectrum, the `FeatureRecord` those produce, and the verdict chain
where the deterministic validator **overrules** the language model's
over-confident lock-in call. Every number in panel (c) is read from
`results/real_piv/feature_record.json` at build time, not retyped.

The dashed edge in F4 is now described precisely as the *synthetic-experiment
generator*, not "the PIV rig", so F3 and F4 no longer contradict each other.

### F5 is new
`fig_rag.png` and `fig_twin.png` carried stale numbers (~30 papers, 1,928 in a
metric card, a 4-channel 256² U-Net). F5 replaces them with the shipped
configuration: 16 papers → 1,840 passages, top-3 at a 0.30 threshold, and the
CLI / FastAPI REST / MCP-with-twelve-tools front doors the abstract names.

---

## Numbers audit

Re-checked against `README.md` and `results/real_piv/` after the final build.
Verified: **1,928 solved cases** (1,881 panel + 35 RANS + 12 URANS) against a
1,926-case 107 × 18 design grid — both figures now appear in the DATA text, so
F2(c) and F4 no longer look wrong · 107 sections · lift R² 0.998, drag 0.995,
moment 0.991, peak deflection 0.968, TE twist 0.786 · 2.3% velocity error ·
293 held-out cases on 17 unseen sections · lock-in F1 0.857 (recall 1.000,
precision 0.750, 8 FP, n = 50) · 47 high-fidelity cases, 30 converged · 11.59 M
parameters · 16 papers, 1,840 passages · 12 MCP tools · real PIV: AoA 17.2°,
f_shed 1.65 Hz, St 0.105, wake 0.65 c, POD₁ 8.5%.

**The abstract still says 1,926.** The poster now carries both numbers and
explains the gap, so nothing has to be corrected under time pressure — but the
abstract is the thing to fix if there is a chance to.

### Claims deliberately narrowed
- **R² is scored against the panel targets the network trained on.** F6 says so
  explicitly now, and reports the two weak scalars (0.968, 0.786) alongside the
  three strong ones.
- **F7 no longer claims "no cost to panel accuracy".** It claims what is true —
  every *field* channel within 3.1% — and states the cost: panel-test drag MAE
  regresses 0.00230 → 0.00314 while lift MAE improves 11%.
- **F8 concedes** that regime classification is saturated at n = 50.

---

## Dropped, and why

| File | Why it is not on the board |
|---|---|
| `paper/figures/fig6_rans.png` | The RANS-validation panel lost its slot to F3. The measured run is the stronger claim; keep this one in the handout for "how do you know the targets are right?" |
| `paper/figures/fig3_network.png` | Architecture detail. Its content (13 channels, potential-flow prior, FiLM, 8×8 attention, 11.59 M) is now carried by the METHODS paragraph. |
| `fig4_parity.png` | R² appears in the lime strip and F6. Handout material if anyone doubts 0.998. |
| `fig_rag.png`, `fig_twin.png`, `fig_metric_cards.png`, `fig_unet.png`, `fig_gallery.png` | Stale numbers (~30 papers, 96 sections, 4 × 256² input). Superseded by F5; do not reuse without regenerating. |
| `fig8_ablation.png`, `fig_confusion.png`, `fig_perclass.png`, `fig_urans_shedding.png` | Unchanged from the previous build — handout material. |

## Handout pocket
- `paper/figures/fig6_rans.png` — steady-RANS validation of the targets, and where it stops (α ≤ 8° only).
- `paper/figures/fig3_network.png` — the 13-channel input contract and the U-Net.
- `paper/figures/fig4_parity.png` — the parity scatter behind R² = 0.998.
- `results/real_piv/REPORT.md` + `figures/pod_modes.png` — the full real-PIV workup.
- `poster/generated/slide1_alt_convergence.png` — which RANS cases were discarded, and why.

---

## Layout notes

- Geometry sampled from the supplied Canva template: lime `#DDEB54`, coral
  `#F47051`, header `#292929`, column splits at 0.3111 / 0.6069 of the width.
- Header band trimmed from 0.1833 H to 400 pt; the template's 7.3 in band is
  more than this much content can afford.
- **Column 3 caps its text measure at 1,210 pt.** At the full 22 in the captions
  ran past 150 characters a line; the block is capped and centred instead.
- Column rules stop 56 pt above the trim edge rather than bleeding off it.
- `build_poster.py` solves figure scale per column numerically (`autofit`) and
  refuses to claim success if a column overflows. Final fill: 98.3 / 98.6 / 98.4 %.
- Type: Poppins for headers and title, Lato for body. Body 28 pt, captions 22 pt,
  section headers 52 pt, title fitted to the available measure (~76 pt).
- F3 and F5 are authored on deliberately small canvases (10.6 in and 8.3 in) so
  that, blown up to a poster column, their internal labels land at 12–14 pt on
  the board rather than 9.

### Before printing
1. Print one letter-size proof and read it at arm's length; that approximates
   the board at 5 ft. Pay attention to F3 and F5 — they are the smallest type.
2. Fix the 1,926 / 1,928 mismatch in the abstract if the deadline allows.
3. Export check: the PDF embeds fonts and 200–400 dpi PNGs; print at 100%, no
   scaling. Outer margins are 46 pt — confirm the shop's trim and bleed spec.
