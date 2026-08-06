# Schema Documentation

## Overview

The FSI-SLM project defines two core data schemas in `Data/schema.py`:

1. **`RawExperiment`** — The raw output of an experimental run or simulation. Contains time-resolved signals (structural displacement, force coefficients, wake velocity probe) and PIV snapshot data. This is the input to feature extraction.

2. **`FeatureRecord`** — The compact, model-facing record produced by feature extraction. This is serialized *verbatim* into the language model prompt and is the source of truth for an experiment's measured numbers.

Both schemas use Pydantic `BaseModel` with strict field validation and `extra="forbid"` to prevent silent field name typos.

---

## `RawExperiment`

The closest thing to "what a rig/solver outputs". A future real-data pipeline should produce objects matching this schema.

| Field | Type | Description | Valid Range |
|-------|------|-------------|-------------|
| `experiment_id` | `str` | Unique run identifier; the train/val/test split is BY this id | — |
| `conditions` | `FlowConditions` | Free-stream / geometric operating point | — |
| `sampling_rate_hz` | `float` | Acquisition sampling rate of the time signals | > 0 |
| `structural_displacement_signal` | `List[float]` | Structural tip displacement vs time, normalized by chord | — |
| `wake_velocity_probe_signal` | `List[float]` | Streamwise velocity at a fixed wake probe vs time (m/s) | — |
| `cl_signal` | `List[float]` | Lift-coefficient time series (dimensionless) | — |
| `cd_signal` | `List[float]` | Drag-coefficient time series (dimensionless) | — |
| `cm_signal` | `List[float]` | Moment-coefficient time series (dimensionless) | — |
| `piv_snapshots` | `List[List[float]]` | Fluctuating-velocity PIV snapshots, shape `(n_snapshots, n_points)` | — |
| `natural_frequency_hz` | `float` | Structural natural frequency f_n in Hz | [0.1, 1000] |
| `mode_number` | `int` | Dominant structural mode number | [1, 6] |
| `camber` | `float` | Mean camber fraction of chord | [0.0, 0.25] |
| `youngs_modulus_pa` | `Optional[float]` | Young's modulus of the section material | — |
| `pressure` | `float` | Ambient pressure in Pa (default 101325) | [0, 500000] |
| `strain` | `float` | Strain, dimensionless (default 0) | [0, 10] |
| `wake_width_over_c` | `float` | Wake width normalized by chord | [0.0, 3.0] |
| `separation_location_x_over_c` | `float` | Separation point x/c (0=LE, 1=TE) | [0, 1] |
| `turbulent_kinetic_energy` | `float` | Wake TKE / U² (dimensionless) | [0.0, 0.5] |
| `ground_truth_regime` | `Optional[FlowRegime]` | True regime (synthetic data only); MUST be None for real experiments | — |
| `notes` | `Optional[str]` | Free-text provenance / caveats | — |

---

## `FeatureRecord`

The compact, model-facing record. This is the object serialized *verbatim* into the language-model prompt via `to_prompt_block()`.

| Field | Type | Description | Valid Range |
|-------|------|-------------|-------------|
| `experiment_id` | `str` | Unique run identifier | — |
| `conditions` | `FlowConditions` | Free-stream / geometric operating point | — |
| `pressure_strain` | `PressureStrain` | Ambient pressure and strain | — |
| `piv` | `PIVDerived` | Quantities derived from PIV velocity fields | — |
| `structure` | `StructuralSignal` | Foil/membrane displacement response | — |
| `forces` | `ForceCoefficients` | Mean / RMS and dominant frequency of force coefficients | — |
| `classification_confidence` | `Optional[float]` | Confidence in the regime call; higher = clearer decision boundary | [0.0, 1.0] |
| `supporting_features` | `Optional[dict]` | Feature-importance weights behind the classification (interpretability aid) | — |
| `ground_truth_regime` | `Optional[FlowRegime]` | Answer key for evaluation only; excluded from the prompt | — |

### `FlowConditions`

| Field | Type | Description | Valid Range |
|-------|------|-------------|-------------|
| `reynolds_number` | `float` | Chord-based Re = U·c/ν | [1e3, 1e7] |
| `angle_of_attack_deg` | `float` | Geometric angle of attack in DEGREES | [-20, 30] |
| `freestream_velocity_m_s` | `float` | Free-stream velocity U in m/s | [0.1, 100] |
| `reduced_velocity` | `float` | U* = U / (f_n · c) | [0.5, 30] |
| `chord_m` | `float` | Chord length c in meters | [0.01, 2.0] |

### `PIVDerived`

| Field | Type | Description | Valid Range |
|-------|------|-------------|-------------|
| `shedding_frequency_hz` | `float` | Vortex-shedding frequency f_shed (largest peak of wake probe spectrum) | [0, 1000] |
| `strouhal_number` | `float` | St = f_shed · c / U (chord-based) | [0.0, 1.0] |
| `wake_width_over_c` | `float` | Wake half/full-width normalized by chord | [0.0, 3.0] |
| `separation_location_x_over_c` | `float` | Chordwise separation point (0=LE, 1=TE) | [0, 1] |
| `pod_mode_amplitudes` | `List[float]` | Energy fraction of leading POD modes (most-energetic first). Need **not** sum to 1 — only the leading modes are kept | Each in [0,1], 1-6 entries |
| `turbulent_kinetic_energy` | `float` | Wake TKE = 0.5⟨u′²+v′²+w′²⟩ / U² (dimensionless) | [0.0, 0.5] |
| `coherence` | `Optional[AerodynamicCoherence]` | Structure-to-force cross-spectral coherence | — |

### `AerodynamicCoherence` (optional, nested in `PIVDerived`)

| Field | Type | Description | Valid Range |
|-------|------|-------------|-------------|
| `cl_coherence_peak` | `float` | Peak coherence between displacement and `Cl` at the dominant frequency; ≈1 means phase-locked energy transfer | [0.0, 1.0] |
| `cd_coherence_peak` | `float` | Peak coherence between displacement and `Cd` | [0.0, 1.0] |
| `phase_angle_deg` | `float` | Phase between displacement and lift at the dominant frequency; ≈90° is maximum energy transfer in VIV | [-180, 180] |

This block is what separates a genuine lock-in from a coincidence of
frequencies. Two signals can share a peak frequency without exchanging energy;
coherence near 1 at a phase near 90° is the physical signature of VIV, and
`features.extract.check_physical_consistency()` cross-checks the lock-in label
against it.

### `StructuralSignal`

| Field | Type | Description | Valid Range |
|-------|------|-------------|-------------|
| `rms_amplitude_over_c` | `float` | RMS structural displacement / chord | [0.0, 1.0] |
| `dominant_frequency_hz` | `float` | Dominant frequency of structural displacement signal | [0.0, 1000] |
| `natural_frequency_hz` | `float` | Structural natural frequency f_n (in vacuo/still-fluid) | [0.1, 1000] |
| `mode_number` | `int` | Dominant structural mode number | [1, 6] |
| `camber` | `float` | Mean camber as fraction of chord | [0.0, 0.25] |
| `damping` | `Optional[ModalDamping]` | Modal damping block | — |
| `spectrum` | `Optional[StructuralSpectrum]` | Frequency-domain descriptors | — |

### `ModalDamping` (optional, nested in `StructuralSignal`)

| Field | Type | Description | Valid Range |
|-------|------|-------------|-------------|
| `damping_ratio` | `float` | Log-decrement-derived damping ratio ζ | [0.0, 0.5]; typical FSI 0.01–0.1 |
| `q_factor` | `float` | Quality factor Q = 1/(2ζ) of the structural mode | [1.0, 500.0]; higher Q = lower damping |

### `StructuralSpectrum` (optional, nested in `StructuralSignal`)

| Field | Type | Description | Valid Range |
|-------|------|-------------|-------------|
| `spectral_bandwidth_hz` | `float` | Half-power bandwidth of the dominant structural peak | [0.0, 100.0] |
| `spectral_kurtosis` | `float` | Kurtosis of the displacement *spectrum*; higher = less sinusoidal | [-1.0, 10.0] |
| `skewness` | `float` | Skewness of the displacement *time series*; zero for symmetric oscillation | [-3.0, 3.0] |
| `kurtosis` | `float` | Kurtosis of the displacement *time series*; > 3 indicates intermittent/impacting behaviour | [-1.0, 10.0] |

Both blocks are `None` unless feature extraction had enough signal to compute
them. When present they are appended to `to_prompt_block()`, so the model sees
them — which means an interpretation may legitimately cite damping on one record
and not another.

### `PressureStrain`

| Field | Type | Description | Valid Range |
|-------|------|-------------|-------------|
| `pressure` | `float` | Ambient pressure in Pa | [0, 500000] |
| `strain` | `float` | Strain (dimensionless / engineering strain) | [0, 10] |

### `ForceCoefficients`

| Field | Type | Description | Valid Range |
|-------|------|-------------|-------------|
| `cl_mean` | `float` | Mean lift coefficient | [-2.5, 2.5] |
| `cl_rms` | `float` | RMS of lift-coefficient fluctuation | [0, 2] |
| `cl_dominant_frequency_hz` | `float` | Dominant frequency of Cl(t) | [0, 1000] |
| `cd_mean` | `float` | Mean drag coefficient | [0, 3] |
| `cd_rms` | `float` | RMS of drag-coefficient fluctuation | [0, 2] |
| `cd_dominant_frequency_hz` | `float` | Dominant frequency of Cd(t) | [0, 1000] |
| `cm_mean` | `float` | Mean pitching-moment coefficient | [-1.5, 1.5] |
| `cm_rms` | `float` | RMS of moment-coefficient fluctuation | [0, 1.5] |
| `cm_dominant_frequency_hz` | `float` | Dominant frequency of Cm(t) | [0, 1000] |

---

## Enums

### `FlowRegime`

The five mutually exclusive flow/aeroelastic regimes:

| Value | Description |
|-------|-------------|
| `"attached"` | Boundary layer attached, low unsteadiness, small structural response |
| `"separated"` | Massive/leading-edge separation, broadband wake, no sustained limit cycle |
| `"viv_lockin"` | Vortex-induced vibration with frequency lock-in |
| `"galloping"` | Low-frequency, large-amplitude movement-induced instability |
| `"stall_flutter"` | SDOF limit-cycle oscillation driven by dynamic stall |

### `LockInStatus`

Relationship between vortex-shedding frequency and structural natural frequency:

| Value | Description |
|-------|-------------|
| `"locked"` | \|f_shed - f_n\| / f_n ≤ tolerance (default 0.10) |
| `"near_lockin"` | Within twice the tolerance |
| `"unlocked"` | Frequencies are independent |

### `AmplitudeTrend`

Qualitative trend of structural response amplitude:

| Value | Description |
|-------|-------------|
| `"growing"` | Amplitude increasing with reduced velocity |
| `"saturated"` | Limit-cycle amplitude reached |
| `"decaying"` | Amplitude decreasing |
| `"negligible"` | No significant structural response |

---

## Key Methods

### `FeatureRecord.to_prompt_block()`

Serializes the measured features into a deterministic, flat `KEY: value` block (not JSON) that is injected verbatim into the model input. The ground-truth regime is deliberately excluded — the model must infer it.

### `FeatureRecord.lock_in_status(tol=0.10)`

Classifies lock-in by comparing shedding frequency to structural natural frequency.

### `FeatureRecord.range_warnings()`

Returns human-readable warnings for any field outside its documented valid range. A future real-data pipeline should run this on every record; many out-of-range values at once almost always means a units mismatch upstream.

---

## Three Layers of Checking

These answer different questions and are deliberately not merged. A record can
pass all of one and fail another.

| Layer | Question | Where |
|---|---|---|
| **Range** | Is any single field outside its documented bounds? | `FeatureRecord.range_warnings()` |
| **Consistency** | Do the fields agree *with each other*? | `features.extract.check_physical_consistency()` |
| **Instrumentation** | Is this a rig fault rather than physics? | `features/noise.py` |

**Range** is the units-mismatch detector. One field out of range is usually a
real outlier; many at once is almost always Hz vs. rad/s, or mm vs. m.

**Consistency** cross-validates the record against itself: reported `St` against
`f·c/U` computed from the record's own frequency and velocity, the lock-in label
against the actual frequency ratio, the damping ratio against the spectral
bandwidth, and the coherence phase against VIV physics. Surfaces as
`record_consistency_warnings` in the FastAPI `/infer` response, the Gradio UI,
and the MCP `interpret_experiment` tool.

**Instrumentation** screening (11 detectors) runs on the *raw signals*, not the
record, and covers PIV dropout, spurious vectors (universal median test), frozen
pixels, ADC clipping and quantisation, spikes, drift, flatlines, spectral SNR,
aliasing, and 50/60 Hz mains pickup. Its output becomes a `[data_quality]` block
in the prompt with an explicit instruction not to attribute an instrumentation
artefact to a flow mechanism.

Two design points in that screen are worth knowing when reading a record:

- The spectral-SNR bar **scales with record length**. A flat periodogram of pure
  noise still has a largest bin — for `m` bins the expected peak-to-median ratio
  is `ln(m)/ln 2`, about 10 for a 2,048-sample record — so a fixed threshold of
  3 would never fire.
- Mains pickup is flagged only when **no physical frequency in the record
  explains the peak**. The wake-probe and lift frequencies are computed first and
  passed in as expected; otherwise every experiment whose shedding lands near
  50 Hz gets flagged, which at laboratory scale is common.

---

## Provenance

`provenance` is **not** a field on `FeatureRecord`. It is a sibling key on the
dict `sensor.bridge.digital_twin()` returns, alongside `feature_record`,
`raw_experiment`, `noise_report`, `case`, `fields` and `den_hartog_margin`. The
record stays exactly the shape a rig would produce; the provenance travels
beside it.

| Key | Meaning |
|---|---|
| `source` | Which path built the case — solver or surrogate |
| `checkpoint` | The surrogate checkpoint used, when there was one |
| `grid` | `[ny, nx]` of the field raster |
| `rig` | Assumed physical scale: chord, span, `U∞`, Re, damping ratio |
| `signals` | Always `"synthesised by a 1-DOF instrument model, not measured"` |
| `shedding_frequency` | `"measured (URANS)"` or `"Fage-Johansen correlation"` |
| `unsteady_source` | Which manifest row supplied the measured frequency |

Two of these carry more weight than the rest.

`signals` is the standing admission that no time series in this system was ever
measured — they are generated from a steady field by a single-degree-of-freedom
instrument model. Nothing downstream should be read as experimental data until
the PIV rig replaces `build_raw_experiment`.

`shedding_frequency` decides what a lock-in call means. Lock-in is judged by
comparing `f_shed` to `f_n` at the few-percent level, so a **measured** frequency
makes the classification a result, while a **correlated** one makes it a
restatement of the assumption that went in. Prefer records where this reads
`"measured (URANS)"` when the distinction matters.

---

## Conventions

- **Frequencies** are in Hz (cycles/second), never angular (rad/s).
- **Strouhal number** uses the chord `c` as the length scale and the freestream velocity `U` as the velocity scale: `St = f_shed * c / U`.
- **Displacement amplitudes** are non-dimensionalized by chord `c` where a `_over_c` suffix is present.
- **"Dominant frequency"** of a signal means the frequency of its largest spectral peak (excluding the DC/mean component).

---

## PIV to FeatureRecord Mapping

| RawExperiment field | FeatureRecord field | Notes |
|---------------------|-------------------|-------|
| `wake_velocity_probe_signal` | `piv.shedding_frequency_hz` | FFT peak |
| `piv_snapshots` | `piv.pod_mode_amplitudes` | SVD energy fractions |
| `structural_displacement_signal` | `structure.rms_amplitude_over_c` | std of signal |
| `structural_displacement_signal` | `structure.dominant_frequency_hz` | FFT peak |
| `cl_signal` | `forces.cl_mean`, `cl_rms`, `cl_dominant_frequency_hz` | mean, std, FFT |
| `cd_signal` | `forces.cd_mean`, `cd_rms`, `cd_dominant_frequency_hz` | mean, std, FFT |
| `cm_signal` | `forces.cm_mean`, `cm_rms`, `cm_dominant_frequency_hz` | mean, std, FFT |
| `wake_width_over_c` | `piv.wake_width_over_c` | Passthrough |
| `separation_location_x_over_c` | `piv.separation_location_x_over_c` | Passthrough |
| `turbulent_kinetic_energy` | `piv.turbulent_kinetic_energy` | Passthrough |
| `natural_frequency_hz` | `structure.natural_frequency_hz` | Passthrough |
| `mode_number` | `structure.mode_number` | Passthrough |
| `camber` | `structure.camber` | Passthrough |