# Poster build record — AGU 60 × 40 in

**Deliverable:** `FSI-SLM_AGU_poster_60x40.pdf` (4320 × 2880 pt = exactly 60 × 40 in, 1 page)
**Screen preview:** `FSI-SLM_AGU_poster_preview.png` (6000 × 4000, 100 dpi)

Rebuild everything:

```bash
python3 -m paper.figures --out paper/figures        # F2, F3, F4, F5, F8 sources
python3 poster/figure_scripts/fig_diagrams.py       # F1 concept figures
python3 poster/figure_scripts/fig_poster_2026.py    # F7 + metric cards, live from JSON
python3 poster/build_poster.py                      # the board
```

---

## What's on the board

| Fig | File | Column | Shows |
|---|---|---|---|
| F1 | `poster/generated/fig_regimes.png` | 1 · Motivation | the five FSI regimes the interpreter separates |
| F2 | `paper/figures/fig2_dataset.png` | 1 · Data | 107 sections, split by shape; angle sampling; fidelity mix |
| F3 | `paper/figures/fig6_rans.png` | 1 · Data | steady-RANS validation of the targets, and where it stops |
| F4 | `paper/figures/fig1_system.png` | 2 · Methods | the twin end to end (hero figure) |
| F5 | `paper/figures/fig3_network.png` | 2 · Methods | 13-channel input contract + U-Net |
| F6 | `paper/figures/fig5_fields.png` | 3 · Results | solver vs surrogate vs error, held-out NACA 0009 |
| F7 | `poster/generated/fig_mf_gain.png` | 3 · Results | **new** — multi-fidelity fine-tuning closes the transfer gap |
| F8 | `paper/figures/fig9_interpreter.png` | 3 · Results | LoRA Phi-4-mini on 50 held-out records |

Headline numbers live in the lime strip, not in a separate metric-card image.

### F7 is new
Nothing in the repo plotted the multi-fidelity result before this. `run_mf/best.pt`
(epoch 29) versus the panel-only `run3/best.pt` (epoch 59), scored on the same
converged RANS cases:

| quantity | panel-only | + high-fidelity | gain |
|---|---|---|---|
| displacement rel. RMSE | 0.576 | 0.234 | 2.5× |
| load rel. RMSE | 0.200 | 0.083 | 2.4× |
| C_l MAE | 0.0879 | 0.0283 | 3.1× |
| C_d MAE | 0.00194 | 0.00192 | 1.0× (unchanged) |
| panel-test field channels | — | — | within 3.1% |

Caveat to have ready: the panel-test **C_d MAE** does regress, 0.00230 → 0.00314
(+36%), while C_l MAE improves 11%. The figure and caption claim only the field
channels stay within 3.1%, which is exactly true. Don't widen the claim.

---

## Numbers audit

Every figure is rendered from artefacts, and every number quoted in body text or
captions was re-checked against those artefacts after the final build. **23 of 23
claims pass.** Verified: 1,928 cases · 107 sections · 18 panel angles · lift
R² 0.998 · drag R² 0.995 · 2.3% velocity error · lock-in F1 0.857 (recall 1.000,
precision 0.750, 8 false positives, n = 50) · 293 held-out cases on 17 unseen
sections · 47 high-fidelity cases, 30 converged · 15 RANS on NACA 0009, 7 converged
· the three multi-fidelity ratios · 11.59 M parameters.

### One thing to fix in the abstract
The abstract says **1,926 cases (107 shapes × 18 angles)**. The manifest now holds
**1,928**: 1,881 panel + 35 RANS + 12 URANS. The 107 × 18 grid is the panel
*design*; five sections have partial panel coverage because they were handed to the
high-fidelity solvers instead. The poster says 1,928, which is what is on disk.
Update the abstract to match, or say "≈1,900 cases across 107 sections".

---

## Dropped, and why

| File | Why it is not on the board |
|---|---|
| `fig4_parity.png` | R² = 0.998 already appears twice (lime strip, F6 caption). The scatter added no new claim for 650 pt of height. |
| `fig_twin.png` | Concept cartoon. First thing cut when the columns filled — it restates the title. Keep it for the talk's opening slide. |
| `fig_rag.png`, `fig_training.png` | Folded into the F4 caption and the analogy text. |
| `fig8_ablation.png` | Mirror-averaging is a technique detail. Handout material. |
| `fig_confusion.png`, `fig_perclass.png` | Rule-based baseline, superseded by F8. Showing both invites "which is it?" |
| `fig_flowfield.png`, `fig_cl_alpha.png`, `fig_cp_surface.png` | Analytic reference curves, **not** network output. Easy to misread as results. |
| `fig_urans_shedding.png` | Only 4 of 12 URANS cases have settled statistics. Handout. |
| `slide1_*`, `slide2_*` | Talk-styled duplicates of F3 and F7. |
| `fig_metric_cards.png` | Regenerated and current, but the lime strip carries the same numbers. Use it in slides. |

---

## Handout pocket
Print letter-size, keep behind the board:

- `poster/generated/slide1_alt_convergence.png` — "how did you decide which RANS cases to discard?" Residual, C_l oscillation and iteration count per case; the cutoff at 10° has no judgement call in the middle.
- `poster/generated/slide2_alt_transfer_ratio.png` — transfer as one ratio per quantity, readable across a room.
- `paper/figures/fig4_parity.png` — the parity scatter, if anyone doubts R² = 0.998.
- `paper/figures/fig8_ablation.png` — mirror test-time averaging.

---

## Layout notes

- Geometry sampled from the supplied Canva template: lime `#DDEB54`, coral
  `#F47051`, header `#292929`, column splits at 0.3111 / 0.6069 of the width.
  Header texture is the template's own aerial photo, re-extracted clean — the
  original render had the template's "TEMPLATE" title and logos baked in.
- Header band trimmed from 0.1833 H to 400 pt. The template's band is 7.3 in tall,
  which this much content could not afford.
- `build_poster.py` solves figure scale per column numerically (`autofit`) and
  refuses to claim success if a column overflows. Final fill: 97.7 / 97.9 / 98.6 %.
- References and acknowledgement sit at the foot of column 2, not column 3 —
  column 3 needed the height for three results figures.
- Type: Poppins for headers and title, Lato for body. Body 28 pt, captions 22 pt,
  section headers 52 pt, title 82 pt. Smallest type inside any figure lands at
  ≈14–18 pt on the printed board.

### Before printing
1. Print one letter-size proof and read it at arm's length; that approximates the board at 5 ft.
2. Fix the 1,926 / 1,928 mismatch in the abstract.
3. Add a QR code to the repo if there is room in the header's right third.
4. Export check: the PDF embeds fonts and 300–400 dpi PNGs; print at 100%, no scaling.
