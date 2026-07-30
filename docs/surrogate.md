# The `sensor` surrogate: geometry in, fields and loads out

This document covers the half of the system that produces data, as opposed to
the half that interprets it. It describes what is computed, how it is
validated, and — as precisely as possible — what it is *not*.

---

## 1. What it does

Given an airfoil section, an angle of attack and a Reynolds number, the
surrogate returns:

| Quantity | Symbol | Where it lives |
|---|---|---|
| Streamwise velocity | `u / U∞` | fluid |
| Transverse velocity | `v / U∞` | fluid |
| Kinematic pressure | `p / (ρ U∞²)` | fluid |
| Net chordwise load | `ΔCp(x)` | solid |
| Transverse displacement | `w / c` | solid |
| Lift, drag, moment | `Cl, Cd, Cm` | scalars |
| Peak deflection, TE twist | `w_max/c`, `θ_TE` | scalars |

Fields are on a fixed 128 × 128 Cartesian raster spanning
`x ∈ [-0.6, 2.4] c`, `y ∈ [-1.5, 1.5] c` — about 42 pixels per chord, with 1.4
chords of wake. The flow channels are zero inside the section and the structural
channels are zero outside it, so the two regions partition the image exactly.

---

## 2. Two backends, one contract

```
sensor/flow/   panel method + viscous correction + wake model   ~0.13 s/case
sensor/cfd/    gmsh + OpenFOAM steady RANS                      ~minutes/case
```

Both write the same `.npz`. Each dataset row carries a `fidelity` tag
(`"panel"` or `"rans"`), so the two can be mixed, weighted differently in the
loss, or split. The 1,926-case sweep that trains the network is entirely
`panel`; the OpenFOAM path exists to regenerate a validation subset — NACA 0009
is held in the test split precisely so it can be re-run at higher fidelity and
compared.

### 2.1 Inviscid core

Hess–Smith panel method: constant-strength sources on each of 180 panels plus a
single constant vortex, closed with an explicit Kutta condition at the trailing
edge. The geometry is rotated to incidence and the freestream held horizontal,
so every field is pixel-aligned with the SDF the network is given.

Two details that matter and are easy to get wrong:

- **Panel clustering.** Cosine spacing is applied so panels bunch at *both* the
  trailing edge and the leading edge. Clustering only at the TE leaves the
  stagnation point unresolved and `Cp_max` lands near 0.64 instead of 1.0.
- **Blunt trailing edges.** Most tabulated sections (NACA 0012, the whole
  6-series) are published with a finite TE gap. The Kutta condition is imposed
  on the first and last panels; with a blunt TE those panels lie on the *base*,
  and a symmetric section then reports a zero-lift angle of −2.6°. Every section
  is passed through `close_trailing_edge` first.

### 2.2 Viscous correction

Beddoes–Leishman static stall, the model used in rotor and wind-turbine
aeroelastic codes:

1. Separation point `f(α)` — the attached chord fraction — decays from 1 through
   0.7 at the stall break `α₁` to 0.04 in deep stall.
2. Kirchhoff–Helmholtz flat-plate theory converts that to a normal-force loss,
   `Cn = Cn_inviscid · ((1 + √f)/2)²`.
3. Leading-edge suction supplies a chord force that dies with separation, so
   pressure drag grows as `Cn sin α − Cc cos α`.

`α₁ = 7 + 55·(t/c)` degrees, plus 2° per decade of Reynolds number and a
camber term. Skin friction is Prandtl–Schlichting with a Hoerner thickness form
factor and a partial-laminar correction below Re = 10⁶.

### 2.3 Field construction

1. Panel velocity field, with the bound circulation scaled by the Kirchhoff
   factor so the far field carries the *stalled* lift.
2. A near-wall layer: within about two cells of the surface the field is
   replaced by the panel's own surface tangential velocity carrying a 1/7-power
   profile. Without it the first cell off the wall sits on the 1/r singularity
   and reports `Cp ≈ −7`.
3. A separated region on the suction side, anchored at the wall, inside which
   the flow is blended toward weak reverse flow.
4. A Gaussian far wake whose integral is exactly `Cd/2` — the momentum-integral
   theorem holds by construction, and a test asserts it to within 5%.
5. Bernoulli pressure, minus the total-pressure loss implied by the local
   deficit, minus a base-pressure suction inside the recirculation bubble.

### 2.4 Structure

The section is a chordwise Euler–Bernoulli beam, supported at the quarter-chord
spar and free at both ends, loaded by `ΔCp(x)` and stiffened by the local
thickness (`EI(x) = E t(x)³/12`). Solved with 2-node cubic-Hermite elements,
which are exact for piecewise-constant `EI` and piecewise-linear load.

Deflection scale is set by one dimensionless stiffness, `E* = E/(½ρU²)`,
default 2 × 10⁴, giving peak deflections around 0.5% of chord. The value is
written into every dataset row.

A separate spanwise cantilever model supplies the modal quantities the FSI
interpreter needs — dry and wet natural frequency (with a strip-theory added
mass), mass ratio, reduced velocity.

**Coupling is one-way and static.** Pressure drives deflection; deflection does
not feed back into the flow. Two-way static iteration is a short extension of
`solve_structure` and is deliberately not claimed.

---

## 3. Accuracy

Against Abbott & von Doenhoff / Sheldahl & Klimas at Re = 5 × 10⁵:

| Quantity | Model | Reference | |
|---|---|---|---|
| NACA 0012 `Cd_min` | 0.011 | 0.010 | |
| NACA 0012 `Cl(8°)` | 0.96 | ~0.85 | |
| NACA 0012 `Cl_max` | 1.36 @ 12° | 1.2–1.35 @ 13–14° | |
| NACA 0012 `Cd(20°)` | 0.27 | ~0.30 | |
| NACA 4412 `Cl(0°)` | 0.51 | ~0.45 | |
| NACA 63-415 `α_L0` | −3.2° | −3.0° | |
| Lift slope | 2π(1 + 0.77 t/c) | thin-airfoil theory | within 3% |
| Inviscid drag | < 10⁻² | 0 (d'Alembert) | |

Lift is 5–15% high near stall and deep-stall drag is ~10% low. These are
correlation-model errors, not solver errors, and they are systematic — which is
what matters for a training target.

**What this is not.** It is not RANS. It has no turbulence model, no laminar
separation bubble, no transition prediction, no compressibility, no
three-dimensionality, and no unsteadiness. Cases with `f < 0.15` are tagged
`deep_stall` in the manifest (219 of 1,926) so they can be down-weighted or
regenerated with OpenFOAM.

---

## 4. Dataset

107 sections × 18 angles = **1,926 cases**, Re = 5 × 10⁵.

Angles are dense where the physics changes fastest and coarse in the post-stall
plateau: −8, −6, −4, −2, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20.

Splits **hold out whole foils**, never individual angles — splitting by case
would put the same shape in train and test at neighbouring incidences and
inflate the score without measuring generalisation to an unseen shape. NACA 0009
is pinned into the test set. Result: 76 train / 14 val / 17 test foils,
stratified across the 4-digit, 5-digit and 6-series families.

```bash
python cli.py sensor-generate                 # 1,926 cases, ~2 min on 4 cores
python cli.py sensor-cache                    # flatten to memmaps
```

---

## 5. Network

U-Net, four down/up steps, GroupNorm and SiLU, residual conv blocks. One
decoder produces all five field channels; a small MLP reads the bottleneck for
the five scalars.

Three departures from the standard airfoil-surrogate setup, each for a reason:

- **Shared trunk for flow and structure.** The load field *is* the surface
  pressure, so the structural head should reuse those features rather than
  relearn them.
- **Direct scalar head.** Integrating `Cl` from the predicted pressure field
  would compound field error and, at 42 px/chord, quantise the surface integral
  badly.
- **GroupNorm.** Batches are small and every case is a different geometry, so
  batch statistics are noisy.

Loss is L1 in the fluid, L1 in the solid, MSE on the scalars — L1 rather than
L2 on the fields because the targets contain a genuine suction peak and a
genuine reverse-flow region, and a squared penalty lets those few extreme pixels
dominate the gradient and wash out the wake, which is what carries the drag.

```bash
python cli.py sensor-train --epochs 60 --batch-size 16          # full 128x128
python cli.py sensor-train --downsample 2 --base 16 --epochs 45 # fast 64x64
python cli.py sensor-eval                                       # held-out score
```

Training is resumable and wall-clock aware (`--time-budget-s`); the checkpoint
carries its own normalisation statistics, fingerprinted against the dataset so a
changed dataset forces a recompute rather than silently rescaling the loss.

---

## 6. Closing the loop

`sensor/bridge.py` turns a steady field into a time-resolved experiment record:

- wake width from the half-deficit points 1.5 chords downstream;
- separation point from the first chord station where near-wall flow reverses;
- shedding frequency from Roshko's universal Strouhal number on the
  Fage–Johansen projected height `c(sin|α| + t/c)`;
- structural amplitude from a single-degree-of-freedom transfer function at
  `r = f_shed/f_n` — which is what makes a lock-in case actually show a large
  response rather than one asserted by a label;
- regime by rule: attached/separated by separation point, lock-in by frequency
  ratio, galloping by the Den Hartog criterion `dCl/dα + Cd < 0` (evaluated with
  two extra solver calls), stall flutter by proximity to the stall break.

The resulting `RawExperiment` goes through the existing `extract_features` and
the new `features/noise.py` screen, and the `FeatureRecord` plus a
`[data_quality]` block go to the language model.

**The time series are a model of the instrument, not a measurement.** Every
record says so in its `notes` and its `provenance` block. When the PIV rig comes
online, its ingest replaces `build_raw_experiment` and nothing downstream
changes — that is the point of the `RawExperiment` contract.

```bash
python cli.py twin --airfoil Data/Airfoils/4/n0009sm.dat.txt --alpha 14
python cli.py twin --airfoil ... --alpha 14 --interpret --use-rag
python cli.py twin --airfoil ... --alpha 14 --inject-faults "mains,dropout"
```

---

## 7. Instrumentation screening

`features/noise.py` answers "is this a rig fault?", which is a different
question from `check_physical_consistency`'s "is this physically coherent?".

Detectors: PIV dropout, spurious vectors (universal median test, Westerweel &
Scarano 2005), frozen pixels, ADC clipping and quantisation, spikes, drift,
flatlines, spectral signal-to-noise, aliasing, and 50/60 Hz mains pickup.

Two design points worth stating:

- The spectral-SNR bar **scales with record length**. A flat periodogram of pure
  noise still has a largest bin: for `m` bins the expected peak-to-median ratio
  is `ln(m)/ln 2`, about 10 for a 2,048-sample record. A fixed threshold of 3
  would never fire.
- Mains pickup is only flagged when **no physical frequency in the record
  explains the peak**. The wake-probe and lift frequencies are computed first
  and passed in as expected; otherwise every experiment whose shedding lands
  near 50 Hz gets flagged, which at laboratory scale is common.

---

## 8. Interfaces

| Surface | Entry points |
|---|---|
| CLI | `sensor-generate`, `sensor-cache`, `sensor-train`, `sensor-eval`, `sensor-predict`, `twin` |
| REST | `GET /sensor/status`, `POST /sensor/predict`, `POST /sensor/polar`, `POST /twin` |
| MCP | `predict_flow`, `predict_polar`, `digital_twin_record`, `check_data_quality`, `surrogate_status` |

All three call `sensor/api.py`, so they cannot drift apart.
