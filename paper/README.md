# Paper figures

```bash
python -m paper.figures --out paper/figures                  # all nine
python -m paper.figures --out paper/figures --only fig4_parity
```

Each figure is written twice: a vector `.pdf` for the manuscript and a 400 dpi
`.png` for slides and previews. Every one is generated from artefacts on disk —
the manifest, the memmap cache, the checkpoint, the eval JSON — so a figure
cannot show a result this repository does not contain. If an input is missing
the figure is skipped with a message rather than faked.

## Inputs

| Artefact | Used by |
|---|---|
| `artifacts/cfd/manifest.jsonl` | 2, 6 |
| `artifacts/sensor/cache/` | 4, 5 |
| `artifacts/sensor/run3/best.pt` | 4, 5 |
| `artifacts/sensor/eval_test_panel.json` | 4, 7, 8 |
| `artifacts/sensor/eval_test_panel_notta.json` | 8 |
| `artifacts/sensor/eval_test_rans.json` | 7 |
| `artifacts/eval_results/v2_vs_v3.json` | 9 |
| `Data/Airfoils/4/n0009sm.dat.txt` | 3 |

Regenerate the inputs with the commands in [docs/results.md](../docs/results.md) §8.

## Draft captions

Numbers in these captions are read from the artefacts at render time; if you
change a checkpoint, re-render and re-check the caption text against the figure.

**Figure 1 — System architecture.** The digital twin end to end. Three solvers
write one tensor contract, chosen per case by incidence: a Hess–Smith panel
method with a Beddoes–Leishman stall correction at any angle, steady RANS below
stall, and URANS above it, where no steady solution exists. A chordwise
Euler–Bernoulli beam converts surface pressure to load and displacement. The
U-Net learns the map from geometry to fields and coefficients; the twin bridge
converts a steady field into a time-resolved experiment record, which a
LoRA-tuned Phi-4-mini interprets. The dashed edge is the only unimplemented one:
the PIV rig would replace `build_raw_experiment` and nothing downstream would
change.

**Figure 2 — Dataset.** (a) 107 NACA sections plotted by maximum thickness and
its chordwise location, coloured by split. Splits hold out whole *shapes*, never
individual angles — NACA 0009 (circled) is pinned to test, so every number
reported for it is a generalisation result. (b) Angle sampling: the gap between
successive incidences narrows from 2° to 1° through the stall region, where the
physics changes fastest. (c) Target fidelity, log scale. 1,913 panel cases and
15 steady-RANS cases on NACA 0009, of which 7 converged.

**Figure 3 — Input contract and network.** (a) Six of the thirteen input
channels, computed for NACA 0009 at α = 12°. `pot_u`, `pot_v` and `pot_p` are
the inviscid panel solution on the same raster: with the prior present the
network predicts the *correction* to potential flow rather than the flow, so the
stagnation point, circulation and far field arrive already exact. (b) U-Net,
four down/up steps, 11.59 M parameters, with self-attention at the 8 × 8
bottleneck because circulation is a global constraint. Incidence, Reynolds
number and the fidelity code enter by FiLM as a per-channel scale and shift
rather than as broadcast image channels.

**Figure 4 — Held-out coefficients.** Surrogate against solver for 293 cases on
17 sections held out by shape, coloured by incidence. Dashed line is parity.

**Figure 5 — Field prediction, NACA 0009.** Solver, surrogate and absolute error
for all five channels on a section held out of training entirely. Flow channels
span the full raster; load and displacement are zoomed onto the section. The
error in the flow channels concentrates in the near-wall layer and the wake,
which is where the boundary layer lives and where the prior is least
informative.

**Figure 6 — Steady-RANS validation, NACA 0009.** (a) Lift and (b) drag against
the panel targets, with the RANS branch split by convergence. Error bars on the
non-converged points are the standard deviation of the lift over the averaging
window. The converged branch recovers a lift slope of 0.095/deg — 86% of the
thin-airfoil value, fitted over all seven converged points — and a drag minimum of 0.0078 at α = 0, inside the published
0.006–0.008 band. (c) Final residual per case: every case at α ≥ 10° hit the
iteration cap two to three decades above the convergence criterion. This is not
a tuning failure — massively separated flow has no steady solution to converge
*to*, so `simpleFoam` limit-cycles. **RANS validation is claimed for α ≤ 8°
only.**

**Figure 7 — Cross-fidelity transfer.** One network, trained on panel targets
only, scored on the held-out NACA 0009 against each solver. (a) Field error
against each truth, with the ratio annotated. (b) The same as a ratio: the
streamwise velocity degrades by only 14% and drag does not degrade at all
(0.84×, n = 7), so the network learned flow structure rather than panel-solver
artefacts. Lift carries the panel model's documented near-stall bias (5.2×), and
the structural channels do not transfer (5.1×) because the chordwise beam
amplifies leading-edge pressure differences — exactly where the two solvers
disagree most.

**Figure 8 — Test-time mirror averaging.** Error reduction from averaging a
prediction with its exact mirror, same checkpoint in both runs. One extra
forward pass, no retraining. The two quantities that gain most — displacement
and lift — are the ones whose error is predominantly antisymmetric, which is
what the averaging cancels.

**Figure 9 — Interpreter.** LoRA-tuned Phi-4-mini on 50 held-out records.
(a) Regime confusion: 50/50, macro F1 = 1.000. This is a statement about the
difficulty of the generated test set as much as about the model. (b) The
discriminating metric is lock-in detection: recall is 1.000 and precision 0.750,
so the model never misses a lock-in and over-calls eight. The rule-based
classifier that generated the training labels shows the same bias, so the model
inherited it rather than inventing it. (c) Numeric fields within tolerance;
amplitude is judged at ±0.02 absolute against a quantity of order a few percent
of chord.

## Notes for the manuscript

- Figures 4, 5, 7 and 8 all describe the **panel-trained** checkpoint. It has
  never been trained on RANS; figure 7 is a *validation* result, not a transfer
  learning result.
- Do not quote R² for `Cm`, `w_max` or `twist_TE` against RANS. With n = 7 on a
  symmetric section the variance is near zero and R² describes the sample. The
  evaluator flags these itself with `"r2_meaningful": false`.
- Figure 9 is the trained adapter (`meta.predictor =
  "trained-model+adapter(no-rag)"`). A second, rule-based table exists in
  `artifacts/eval_results/eval_results.json` — check `meta.predictor` before
  quoting either.
