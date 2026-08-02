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

## 2. Three backends, one contract

```
sensor/flow/            panel method + viscous correction + wake  ~0.13 s/case
sensor/cfd/dataset.py   gmsh + OpenFOAM steady RANS               ~4 min/case
sensor/cfd/unsteady.py  gmsh + OpenFOAM URANS (pimpleFoam)        ~hours/case
```

All three write the same `.npz`. Each dataset row carries a `fidelity` tag
(`"panel"`, `"rans"` or `"urans"`), so they can be mixed, conditioned on,
weighted differently in the loss, or split.

Which solver for which incidence is not a preference, it is a measurement.
Every steady case at α ≥ 10° in the first RANS sweep hit the iteration cap with
initial residuals of order 10⁻² and lift oscillating by up to ±0.43. That is not
a convergence failure to tune away: massively separated flow has no steady
solution to converge *to*, `simpleFoam` limit-cycles, and whichever iteration it
stops on is an arbitrary point on the oscillation.

| α | solver | why |
|---|---|---|
| ≤ 8° | steady RANS | converges to 10⁻⁵, and is ~100× cheaper than transient |
| ≥ 10° | URANS | the only one that has an answer |

The merge in `finalize_manifest` ranks `urans > rans > panel` for the same
(foil, angle), so re-solving a case upgrades it in place.

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

### 5.1 What the network is given

The input contract is named and versioned (`sensor/sdf.py`, `InputSpec`).
The default, `v2`, is 13 channels:

| Channels | | Why |
|---|---|---|
| `sdf` | signed distance, chord units | smooth geometry encoding with a usable gradient everywhere |
| `x`, `y` | grid coordinates | the flow is *not* translation-invariant — the wake is downstream — and a convolution has no other way to know where it is |
| `n_x`, `n_y` | ∇(sdf) | the wall normal; near the surface it is what decides which side of the body a pixel is on |
| `wall` | exp(−d/0.05c) | near-wall emphasis, and the weight the scalar head pools with |
| `pot_u`, `pot_v`, `pot_p` | **inviscid solution** | the physics prior |
| `fs_x`, `fs_y`, `log_re`, `fidelity` | constants | global conditioning |

The prior is the change that matters most. With `pot_*` present the network is
not predicting the flow, it is predicting the *correction* to potential flow:
the stagnation point, the circulation, the suction-peak location and the whole
far field arrive already exact, and what is left to learn is the boundary layer,
the separation and the wake. It costs one panel solve (~40 ms) per case at cache
build time and, crucially, is available at every fidelity — the panel geometry
does not care which solver produced the target, so a RANS row gets the same
prior and the two stay interchangeable.

`v2_noprior` (10 channels) exists to ablate exactly that:

```bash
python cli.py sensor-cache --spec v2_noprior && python cli.py sensor-train
```

The four constant channels are stored as full maps so the on-disk tensor stays
one array, but they are flagged `constant` and read back by the model as a
global vector — see FiLM below.

### 5.2 Architecture

U-Net, four down/up steps, GroupNorm and SiLU, residual conv blocks, ~11 M
parameters at `--base 32`. Seven departures from the standard setup, each for a
reason:

- **Shared trunk for flow and structure.** The load field *is* the surface
  pressure, so the structural head should reuse those features rather than
  relearn them.
- **FiLM conditioning rather than constant image channels.** Incidence,
  Reynolds number and the fidelity code are scalars. Broadcasting them into
  feature maps and asking 3×3 convolutions to carry them through four
  downsamples spends capacity reproducing a constant. They are read out of the
  constant channels as a global vector and injected as a per-channel scale and
  shift at every block, initialised to the identity.
- **A scalar head that pools where the forces are.** `Cl` is a surface integral;
  a global average over the bottleneck weights a pixel three chords downstream
  exactly like a pixel on the suction peak. The head now pools every encoder
  level with both mean and max, *and* takes a wall-weighted average of the
  finest decoder features using the `wall` channel as the weight.
- **Bilinear upsampling, not transposed convolution.** A stride-2 transposed
  convolution with an even kernel lays down a checkerboard, and on a pressure
  field that pattern lands in the wake, which is where the drag is read.
- **Replicate padding.** Zero padding tells the network the flow stops at the
  image border. It does not; the border is an interior slice of a much larger
  domain.
- **Self-attention at the bottleneck.** Circulation is global — the Kutta
  condition at the trailing edge sets the lift, which sets the far field
  everywhere. The bottleneck map is 8 × 8, so attention over 64 tokens is cheap.
  Initialised to zero, so it can only help.
- **GroupNorm.** Batches are small and every case is a different geometry, so
  batch statistics are noisy.

### 5.3 Loss

L1 in the fluid and L1 in the solid — rather than L2, because the targets
contain a genuine suction peak and a genuine reverse-flow region and a squared
penalty lets those few extreme pixels dominate the gradient and wash out the
wake. Huber on the scalars, so one badly-predicted deep-stall case does not own
the batch. Then four additions, each aimed at a measured failure:

| Term | Weight | What it fixes |
|---|---|---|
| `gradient` | 0.25 | A value-only L1 is minimised by a blurred field. `v` and `p` — the two worst channels — are defined by gradients, not by their own magnitude. |
| `near_wall` | 3.0 | Near-wall pixels carry the load and the suction peak and are ~2% of the fluid area; unweighted they are averaged away. |
| `divergence` | 0.02 | The flow is incompressible: ∂u/∂x + ∂v/∂y = 0 exactly. Two finite differences, and a constraint the network otherwise cannot know. |
| `circulation` | 0.05 | Kutta–Joukowski ties the scalar head to the velocity field. |

The circulation term deserves a note. The contour integral of velocity around
the image border is the enclosed circulation, hence `Cl/2` — measured on the
solver's own rasters it recovers `Cl` to within 0.3–5% (`tests/test_physics_loss.py`),
one-signed, because the wake deficit leaves through the downstream edge and
takes a little circulation with it. So the loss is written as a *difference of
differences*: the error in the raster circulation must move with the error in
`Cl`. Any systematic offset cancels instead of being learned as a bias.

All four are switchable (`--w-gradient`, `--w-divergence`, `--w-circulation`,
`--w-near-wall`), and `masked_loss` still exists as the plain-L1 ablation.

### 5.4 Symmetry

Reflecting a case about `y = 0` gives another physically valid case: the
mirrored section at the negated incidence. Not an approximation — for a
symmetric section, mirroring the solution at +8° reproduces the solution at −8°
to float32 rounding outside the painted wall layer, which is what
`tests/test_symmetry.py` asserts against the solver rather than assuming.

It is used twice. As **augmentation**, doubling the effective dataset and
balancing an incidence sweep that runs −8 to +20. And as **test-time
averaging**: predict the case and its mirror, fold the second back, average.
One extra forward pass, no retraining, and it cancels the antisymmetric part of
the error.

The one trap: mirroring happens in *normalised* space, where a sign flip alone
is wrong. `(parity·s − mean)/std ≠ parity·(s − mean)/std` unless the mean is
zero, and `Cd`, `w_max` and the pressure channel all have means far from zero.
Every transform is therefore an affine (scale, shift) pair, and a test pins that
it stays an involution — otherwise test-time averaging silently biases every
channel.

### 5.5 Training

```bash
python cli.py sensor-train --epochs 80 --batch-size 16          # full 128x128
python cli.py sensor-train --downsample 2 --base 16 --epochs 45 # fast 64x64
python cli.py sensor-eval                                       # held-out score
python cli.py sensor-eval --tta=False                           # what TTA is worth
```

Weights are exponentially averaged (`--ema-decay`, default 0.999) and it is the
*averaged* weights that are validated, selected on and released: a raw SGD
iterate on a 16-case batch is a noisier estimate of the same model, and picking
"best" on it selects for lucky steps.

Training is resumable and wall-clock aware (`--time-budget-s`); the checkpoint
carries its own normalisation statistics and input spec, fingerprinted against
the dataset so a changed dataset forces a recompute rather than silently
rescaling the loss, and a mismatched cache is refused with a message rather than
a broadcast error.

---

## 5b. Multi-fidelity

Two mechanisms, and they compose.

**Conditioning.** The fidelity code is an input channel, so one network learns
all three solvers and can be *asked* which to emit:

```python
p.predict(foil, alpha_deg=12, fidelity="urans")
```

This is strictly more useful than a weighted average of the two: the 1,913 cheap
cases teach the shared structure, and the expensive ones only have to teach the
difference. Asking for a fidelity that was not in the checkpoint's training
split returns the prediction *and* a warning saying the conditioning channel is
being extrapolated.

**Weighting.** A hundred RANS cases against 1,900 panel cases are 5% of the
gradient at equal weight, which does not move the network off the cheap targets:

```bash
python cli.py sensor-train --fidelity-weights "panel=1,rans=8,urans=8"
```

**Two-stage.** Pretrain on panel, fine-tune on the mix at a tenth the rate:

```bash
python cli.py sensor-train --run-dir artifacts/sensor/run
python cli.py sensor-train --run-dir artifacts/sensor/run_ft --lr 2e-4 \
    --init-from artifacts/sensor/run/best.pt \
    --fidelity-weights "panel=1,rans=8,urans=8"
```

### Getting high-fidelity rows that are legal to train on

The RANS cases that exist are all on NACA 0009, which is pinned to the **test**
split — using them for training would be training on test, which is why the
multi-fidelity path was wired and tested but never actually used. What it needs
is high-fidelity rows on foils the network is allowed to see:

```bash
python cli.py sensor-multifidelity                # print the plan, spend nothing
python cli.py sensor-multifidelity --run --max-workers 4
python cli.py sensor-finalize && python cli.py sensor-cache
```

The plan picks train-split foils one per series (seeded, so the choice is
reproducible and reportable) and solves each below stall with steady RANS and
above stall with URANS.

---

## 5c. Unsteady RANS

`sensor/cfd/unsteady.py`. `pimpleFoam`, k-ω SST, second-order `backward` time
scheme, Courant-limited adaptive stepping, `fieldAverage` started after the
transient. Times are in convective times `c/U∞`, so the defaults —
`t_end = 80`, averaging from `t = 20` — leave roughly ten shedding cycles in the
window.

Four settings that decide whether the run is worth anything:

- **`ddtSchemes backward`.** Euler damps the shedding amplitude badly at the
  steps an adaptive run actually takes.
- **No under-relaxation on the final outer corrector.** Otherwise the time
  derivative is quietly damped, and the shedding with it.
- **Force coefficients written every time step.** The lift signal *is* the
  measurement; a decimated record aliases the shedding peak.
- **`fieldAverage timeStart` after the transient.** Averaging from `t = 0` folds
  the impulsive start into the mean and no amount of run time removes it.

### What comes out that a steady solve cannot give

The vortex-shedding frequency, **measured**. `shedding_frequency` interpolates
the adaptive-step record onto a uniform grid (an FFT of unequally-spaced samples
smears the peak and can move it), removes the mean, applies a Hann window, and
fits a parabola through the peak bin and its neighbours. Sub-bin interpolation
matters here: the bin spacing is 1/(record length) ≈ 0.0125, about 6% of a
typical Strouhal number, and lock-in is decided by comparing `f_shed` to `f_n`
at the few-percent level. On synthetic records the recovered frequency is within
0.1% of truth.

Because the case is nondimensional (`c = 1`, `U∞ = 1`), a frequency in cycles
per unit time *is* the chord-based Strouhal number — no conversion, no chance of
a units slip. The projected-height number `St_d = f·c(sin|α| + t/c)` is the one
the bluff-body literature quotes.

A peak is only reported as shedding if it stands more than 5× clear of the local
spectral background. An attached foil has no organised street, and returning the
largest noise bin as a frequency would be worse than admitting the correlation
is all there is.

### Two convergence questions, not one

`pimpleFoam` exits 0 whether or not the statistics settled, exactly as
`simpleFoam` exits 0 whether or not the residuals fell. `statistics_converged`
requires both that the window covers at least six shedding periods (averaging
over two and a half returns a number that depends on the starting phase) and
that the mean over the first half of the window agrees with the second half to
within 5% of the signal's own RMS. The flag travels with the case, and
`--converged-only` reads it.

### Where it lands downstream

`sensor/bridge.py` used the Fage–Johansen `St ≈ 0.2` correlation for the
shedding frequency — a correlation, applied to an airfoil, in the regime where
it is least reliable — and that frequency is what the lock-in classification is
built on. It now prefers a measured value when the case carries one, and says
which it used. The difference between a measured and a correlated frequency is
the difference between a lock-in call that means something and one that restates
an assumption.

```bash
python cli.py sensor-urans --foils Data/Airfoils/4/n0012.dat.txt --alphas 12,16,20
```

### What 2-D URANS is not

It resolves the primary von Kármán shedding and nothing smaller. Real post-stall
flow is three-dimensional and broadband; 2-D URANS is known to overpredict the
coherence of the shedding and therefore the lift fluctuation amplitude, often by
tens of percent. It gets the **frequency** far better than the **amplitude**.
Cases are tagged `fidelity="urans"` so the distinction survives into training.

---

## 6. Closing the loop

`sensor/bridge.py` turns a steady field into a time-resolved experiment record:

- wake width from the half-deficit points 1.5 chords downstream;
- separation point from the first chord station where near-wall flow reverses;
- shedding frequency, **measured** from the URANS lift signal when the case has
  one, otherwise from Roshko's universal Strouhal number on the Fage–Johansen
  projected height `c(sin|α| + t/c)` — the record says which;
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
| CLI | `sensor-generate`, `sensor-cache`, `sensor-finalize`, `sensor-rans`, `sensor-urans`, `sensor-multifidelity`, `sensor-train`, `sensor-eval`, `sensor-predict`, `twin` |
| REST | `GET /sensor/status`, `POST /sensor/predict`, `POST /sensor/polar`, `POST /twin` |
| MCP | `predict_flow`, `predict_polar`, `digital_twin_record`, `check_data_quality`, `surrogate_status` |

All three call `sensor/api.py`, so they cannot drift apart.
