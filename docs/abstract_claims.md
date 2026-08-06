# Abstract claims → implementation

A sentence-by-sentence map from the paper abstract to what exists in this
repository, with the file that implements it and the honest limitation.

Status key: **implemented** · **implemented, scoped** (real but narrower than the
sentence implies) · **not implemented**

---

### "We build a digital twin that integrates experimental FSI obtained using a particle image velocimetry (PIV) system…"

**Not implemented — the rig is not connected.**

There is no PIV hardware ingest. `Data/schema.py` defines the `RawExperiment`
contract a real rig would populate, and `sensor/bridge.py` populates it from the
surrogate instead. Every generated record carries
`provenance.signals = "synthesised by a 1-DOF instrument model, not measured"`.

Until the rig is online, the abstract should say *simulated* PIV, or state that
the PIV integration is the contract plus the ingest point, not the data.

### "…and simulations with a small-language model (SLM) that can interpret and respond to questions based on the data."

**Implemented.** Phi-4-mini-instruct (3.8 B) with MLX LoRA adapters in
`artifacts/adapters/`, trained by `train/sft.py`.

### "Our work focuses specifically on the NACA 0009 airfoil at various angles of attack."

**Implemented.** NACA 0009 is pinned into the surrogate test split
(`sensor/generate.py:TEST_FOILS`) and is the section used in every validation
figure. It is held out of training entirely, so every number reported for it is
a generalisation result.

### "We turn the raw FSI data into a JSON file summary of the experiment."

**Implemented.** `Data/schema.py` → `FeatureRecord`, serialised verbatim into the
prompt. `features/extract.py` does the FFT / POD / damping extraction.

### "A small language model, fine-tuned on realistic simulated experiments, reads that summary to interpret the fluid dynamics…"

**Implemented.** `Data/synthetic.py` generates physically coupled experiments
(Strouhal, lock-in, galloping, stall flutter); `inference/runner.py` runs the
tuned model.

Measured, released adapter, 50 held-out records
(`artifacts/eval_results/v2_vs_v3.json`, `meta.predictor =
"trained-model+adapter(no-rag)"`):

| Metric | Value |
|---|---|
| Regime macro F1 | 1.000 (50/50) |
| Lock-in F1 | 0.857 (precision 0.750, recall 1.000) |
| `f_shed`, `St`, `mode_number` accuracy | 1.000 |
| `rms_amplitude_over_c` accuracy | 0.780 |

**Scope to state:** regime classification is *saturated* at this sample size —
50/50 and 100/100 correct. That is a statement about the difficulty of the
generated test set as much as about the model, and the paper should say so
rather than presenting 1.000 as a headline. The discriminating number is
lock-in F1 at 0.857, where recall is perfect and precision is 0.750: the model
never misses a lock-in and calls 8 that are not there.

### "…and flag likely sensor noise."

**Implemented.** `features/noise.py`, eleven detectors:

| Detector | What it catches |
|---|---|
| `piv_dropout`, `piv_outliers`, `piv_frozen_pixels` | correlation failure, spurious vectors (universal median test), dead sensor regions |
| `clipping`, `quantisation` | load cell past range, ADC too coarse |
| `spikes`, `drift`, `flatline` | interference, thermal drift, unresponsive channel |
| `spectral_snr`, `aliasing`, `mains_interference` | unresolvable frequency, fold-down, 50/60 Hz pickup |

Flags are injected into the prompt as a `[data_quality]` block
(`rag/prompt_assembly.py`) with an instruction not to attribute an
instrumentation artefact to a flow mechanism. 19 tests in `tests/test_noise.py`
cover both firing and *not* firing.

### "The model also pulls from published fluid-dynamics research through retrieval-augmented generation (RAG)…"

**Implemented.** FAISS index over **16** papers in `Data/Papers/`, chunked to
1,840 passages (256 tokens, 50% overlap), embedded with `all-MiniLM-L6-v2`,
top-3 retrieval at a 0.3 score threshold. See `rag/`.

One honest qualifier on the *purpose* clause "to reduce the chance for
hallucinations": retrieval at inference time is implemented and available via
`--use-rag`, but **it has not been shown to reduce hallucination**, because no
hallucination-rate measurement exists in the repository. What was measured is
RAG in the *training* mix, and it came out marginally worse (§7b.1 of
`results.md`: lock-in F1 0.840 vs 0.857). The grounding argument that *is*
supported is architectural rather than empirical: the system prompt pins the
`FeatureRecord` as the only source of truth for measured numbers, and
`eval/validate_output.py` physics-checks every generated interpretation.

### "The full pipeline can be run on an ordinary laptop, with no specialized hardware required."

**Implemented.** MLX on Apple Silicon for the SLM; the surrogate dataset
generates in about two minutes on four CPU cores and the U-Net trains on CPU or
MPS.

### "It is accessible through a REST API built with FastAPI, and through an MCP server…"

**Implemented.** `serve/fastapi_app.py` — 8 routes (`GET /health`,
`GET /schema`, `POST /infer`, `POST /infer/stream`, `GET /sensor/status`,
`POST /sensor/predict`, `POST /sensor/polar`, `POST /twin`) — and
`serve/mcp_server.py` — **12** tools plus the `fsi://schema/feature_record`
resource. Both call the same `sensor/api.py` and `inference/runner.py`, so the
surfaces cannot drift apart.

### "We also train a U-Net architecture using 107 airfoil shapes and 18 angles of attack (1,926 cases)…"

**Implemented.** `sensor/model.py`, `sensor/train.py`. Dataset is exactly
107 × 18 = 1,926 cases, generated by `sensor/generate.py`, foil-disjoint splits
76 / 14 / 17.

**Scope to state in the paper:** the *training* targets are not RANS. They come
from a Hess–Smith panel method with a Beddoes–Leishman stall correction and a
momentum-consistent wake model (`sensor/flow/`), calibrated against published
NACA data to roughly 10% on `Cl_max` and `Cd`.

A RANS validation set now exists. The OpenFOAM path (`sensor/cfd/`) writes the
identical tensor contract through the shared packer in `sensor/cfd/adapt.py`,
every row carries a `fidelity` tag (`"panel"` | `"rans"`), and the manifest
merge prefers the higher fidelity when the same (foil, angle) exists at both.
15 k-ω SST cases were run on NACA 0009 at Re = 5 × 10⁵ (2-D, 324 k prisms,
`simpleFoam`), giving a 1,928-case dataset.

Only 7 of the 15 converged. Steady SIMPLE limit-cycles on separated flow; every
case at α ≥ 10° hit the iteration cap with initial residuals of order 10⁻² and
lift oscillating by up to ±0.43. Convergence is recorded per case
(`converged`, `residual_max`, `cl_std`, `cd_std`) and `sensor-eval
--converged-only` excludes the rest. **Claim RANS validation for α ≤ 8° only.**

Converged branch, against published NACA 0009 data:

| | RANS | reference |
|---|---|---|
| lift slope | 0.095 /deg (5.43 /rad, 86% of 2π) | thin-airfoil theory |
| `Cd_min` at α = 0 | 0.0078 | 0.006–0.008 |
| drag minimum location | α = 0 | α = 0, symmetric section |
| ±4° lift symmetry | 2.3% | 0 |

### "…to predict the velocity fields, lift and drag coefficients, and the airfoil's load and displacement fields directly from airfoil geometry and flow conditions."

**Implemented.** Five field channels (`u`, `v`, `p`, `load`, `disp`) plus five
scalars (`Cl`, `Cd`, `Cm`, `w_max`, `twist_TE`). Held-out results, **293 cases
across 17 unseen sections**, full 128 × 128, 11.59 M parameters, mirror TTA:

| Scalar | R² | MAE |
|---|---|---|
| `Cl` | 0.998 | 0.0168 |
| `Cd` | 0.995 | 0.0023 |
| `Cm` | 0.991 | 0.0037 |
| `w_max / c` | 0.968 | 2.7e-4 |
| `twist_TE` | 0.786 | 0.227° |

| Field | relative RMSE |
|---|---|
| `u` | 0.023 |
| `v` | 0.077 |
| `p` | 0.108 |
| `load` | 0.079 |
| `disp` | 0.113 |

### Cross-fidelity transfer

The same network, scored on the held-out NACA 0009 against RANS truth
(7 converged cases) versus against panel truth (293 cases). Scoring a mixed
split gives a number that is neither, so they are reported apart.

| | vs panel (n=293) | vs RANS (n=7) | ratio |
|---|---|---|---|
| `u` rel-RMSE | 0.0227 | 0.0259 | 1.14× |
| `v` rel-RMSE | 0.0770 | 0.3367 | 4.37× |
| `p` rel-RMSE | 0.1079 | 0.4973 | 4.61× |
| `load` rel-RMSE | 0.0788 | 0.1999 | 2.54× |
| `disp` rel-RMSE | 0.1125 | 0.5764 | 5.12× |
| `Cl` MAE | 0.0168 | 0.0879 | 5.22× |
| `Cd` MAE | 0.00230 | **0.00194** | **0.84×** |

Three findings, in order of how much they matter:

1. **The velocity field and drag transfer.** `u` degrades by only 14% when the
   truth changes solver, against 2.5–5× for every other channel, and `Cd` MAE is
   *lower* against RANS. The network learned flow structure, not solver
   artefacts. At n = 7 the honest claim for `Cd` is "does not degrade", not
   "improves".
2. **Lift carries the panel model's bias** — 5.2× the error. The panel backend
   is documented as running 5–15% high near stall, and the surrogate inherited
   it.
3. **Structural and transverse channels do not transfer** (`disp` 0.58, `p`
   0.50, `v` 0.34). The chordwise beam amplifies leading-edge pressure
   differences, and that is exactly where the panel method and RANS disagree
   most. Predicting load and displacement at RANS fidelity requires RANS cases
   in *training*, not only in test.

Do **not** quote R² for `Cm`, `w_max` or `twist` on the RANS rows. With n = 7 on
a symmetric section those quantities have near-zero variance, so
`R² = 1 − SS_res/SS_tot` is dominated by a vanishing denominator and returns
large negative values (−0.78 for `w_max`, −0.09 for `twist_TE`) that describe
the sample, not the model. The evaluator flags this itself — those rows carry
`"r2_meaningful": false` and a note naming the spread and n. Report RMSE/nRMSE
and state n.

**Scope to state:** the released checkpoint is the full 128 × 128 configuration
(11,589,130 parameters, `--base 32`, depth 4, 60 epochs, EMA weights), trained
on **panel targets only**. Multi-fidelity training is wired
(`--fidelity-weights "panel=1,rans=8,urans=8"`) but has not been used for a
released checkpoint, because the only RANS rows in the manifest are on NACA 0009,
which is in the test split — training on them would be training on test. A URANS
sweep on *training*-split foils is currently running (8 of 12 cases solved); its
rows are not merged into the manifest yet and affect no number reported here.

Structural coupling is **one-way and static** — pressure drives deflection,
deflection does not feed back into the flow. The section is a chordwise
Euler–Bernoulli beam at a fixed dimensionless stiffness, and a separate spanwise
cantilever model supplies natural frequency, mass ratio and reduced velocity.

### "This approach replaces computational simulations and experiments, thus building a complete end-to-end digital twin with Agentic AI."

**Overstated. Recommend rewording.**

What is true: the network replaces the *panel-and-beam solver* it was trained
on, at roughly 10⁴× lower cost, and the loop from geometry to interpreted
experiment closes without a human (`cli.py twin`, `POST /twin`,
`digital_twin_record`).

What is not true: it does not replace experiments (there are none in the loop
yet), and it has never been *trained* on RANS. It has now been *validated*
against RANS on the held-out section, which is a weaker but real claim. A
defensible version:

> This replaces the solver in the loop, giving an end-to-end path from geometry
> to interpreted experiment record. On a held-out NACA 0009 the surrogate
> reproduces steady RANS streamwise velocity fields to 2.6% and drag to a mean
> absolute error of 0.002 without having seen RANS in training; lift and the
> structural channels retain the bias of the potential-flow targets. Validation
> against the PIV rig is in progress.

---

## Suggested abstract edits

1. "experimental FSI obtained using a PIV system" → "an FSI dataset from a
   validated potential-flow-plus-viscous-correction solver, with a PIV ingest
   contract in place for the physical rig".
2. Name the target fidelity for the U-Net rather than letting "1,926 cases"
   imply RANS. They are panel-method targets, RANS-validated on one held-out
   section below stall.
3. Say the structural coupling is one-way and static.
4. Soften "to reduce the chance for hallucinations" to what was actually built —
   retrieval plus a record-is-ground-truth prompt contract plus an output
   physics-checker — since no hallucination rate was measured, and RAG in the
   *training* mix measured slightly worse.
5. Replace "replaces computational simulations and experiments" with the
   narrower claim above.
6. If regime F1 is quoted, state n. At n = 50 it is 1.000, which says the test
   set is not discriminative; lock-in F1 (0.857) is the number that separates
   models.

## Reproducing every number here

```bash
python cli.py sensor-generate
python cli.py sensor-cache
python cli.py sensor-train --epochs 80 --batch-size 16          # full 128x128
python cli.py sensor-eval --checkpoint artifacts/sensor/run3/best.pt \
    --split test --fidelity panel
python cli.py eval --use-model                                   # interpreter
python -m sensor.figures --out poster/generated
python -m pytest tests/ -q                                       # 266 tests
```

The RANS validation set (needs OpenFOAM v2606 + gmsh; ~3 h on an M4 Pro, and
most of that is the eight post-stall cases that never converge):

```bash
python cli.py sensor-rans --foils Data/Airfoils/4/n0009sm.dat.txt \
    --alphas="-8,-4,0,2,4,6,8" --max-workers 4
python cli.py sensor-finalize
python cli.py sensor-cache
python cli.py sensor-eval --checkpoint artifacts/sensor/run3/best.pt \
    --split test --fidelity rans --converged-only
```

Two shell traps, both of which bite silently:

- Interactive zsh does not treat `#` as a comment, so a trailing note on these
  commands arrives as positional arguments. `cli.py` detects and rejects that
  rather than binding it to a parameter.
- `--alphas="-8,..."` needs the quotes and the `=`. A bare leading minus is
  parsed as a flag.
