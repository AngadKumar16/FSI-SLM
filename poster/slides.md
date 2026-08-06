# Poster — U-Net flow sensor for hydrofoils (2 slides)

Figures (in this folder):
- `poster_fig1_flowfields.png` — CFD flow fields around NACA 0009 (Slide 2 hero)
- `poster_fig2_gallery.png` — 107-airfoil dataset by series (Slide 1)

---

## Slide 1 — Problem, method, data

**Title:** A U-Net flow sensor for hydrofoils: shape + angle → velocity, pressure, force

- **Task.** Given a 2D airfoil shape (`.dat`/STL → signed-distance field) and an
  angle of attack (0–17°, up to stall) in water, predict the full velocity and
  pressure fields, and integrated lift/drag (Cl, Cd) — a fast learned *sensor*
  in place of a CFD solve.
- **Representation.** Airfoil → signed-distance field (SDF) on a fixed grid;
  AoA and Reynolds as extra channels. Output = (u, v, p) images + force head.
- **Why it matters.** This is the upstream *sensor* for the FSI-SLM interpreter:
  its fields feed the same FeatureRecord schema the language model reads.
- **Data.** 107 NACA sections organized into 4-digit / 5-digit / 6-series
  families → generalization across shapes; NACA 0009 held out for validation.

→ Figure: `poster_fig2_gallery.png`

## Slide 2 — This week: working proof-of-concept

**Title:** End-to-end CFD data generation, validated on NACA 0009

- Built the full automated pipeline: `.dat → SDF → gmsh mesh → OpenFOAM
  simpleFoam (k-ω SST) → Cl/Cd → grid-sampled training pair`.
- Validated on NACA 0009 at α = 8°, Re = 5×10⁵: **Cl ≈ 0.73**, with the correct
  flow structure — leading-edge suction, stagnation, circulation, and wake.
- **Done:** all 1,926 cases (107 foils × 18 AoA) generated with the fast
  panel + Beddoes-Leishman backend, and the U-Net (11.6 M params, 128 × 128)
  trained on them. Held out 17 unseen sections (NACA 0009 among them):
  **Cl R² = 0.998, Cd R² = 0.995**, peak deflection R² = 0.97, and **2.3%**
  relative RMSE on the streamwise velocity field.
- **Done:** a steady-RANS validation set on the held-out NACA 0009 (k-ω SST,
  7 converged cases at α ≤ 8°). The velocity field transfers almost intact —
  2.6% error against a solver the network never trained on — and drag does not
  degrade (MAE 0.0019). Lift keeps the panel model's near-stall bias.
- Next: URANS past stall on *training* foils (running now, 8 of 12 solved) so
  high-fidelity rows become legal to train on, then fine-tune the
  panel-pretrained network on the mix.

→ Figures: `generated/fig_sensor_fields.png`, `generated/fig_sensor_separation.png`,
`generated/fig_sensor_accuracy.png`, `generated/fig_sensor_field_compare.png`

→ Figure: `poster_fig1_flowfields.png`

---

Regenerate both figures: `python poster/make_figures.py` (needs a solved case;
see `sensor/cfd/`). Figure 1 resamples the OpenFOAM solution onto a fine local
grid; Figure 2 reads `Data/Airfoils/{4,5,6}/`.
