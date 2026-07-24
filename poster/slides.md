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
- Next: generate ~1900 cases (107 foils × 18 AoA), train the U-Net, and check
  the predicted Cl–α curve against XFOIL through stall (~17°).

→ Figure: `poster_fig1_flowfields.png`

---

Regenerate both figures: `python poster/make_figures.py` (needs a solved case;
see `sensor/cfd/`). Figure 1 resamples the OpenFOAM solution onto a fine local
grid; Figure 2 reads `Data/Airfoils/{4,5,6}/`.
