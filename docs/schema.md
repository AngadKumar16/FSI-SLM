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
| `piv` | `PIVDerived` | Quantities derived from PIV velocity fields | — |
| `structure` | `StructuralSignal` | Foil/membrane displacement response | — |
| `forces` | `ForceCoefficients` | Mean / RMS and dominant frequency of force coefficients | — |
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
| `pod_mode_amplitudes` | `List[float]` | Energy fraction of leading POD modes (most-energetic first) | Each in [0,1], 1-6 entries |
| `turbulent_kinetic_energy` | `float` | Wake TKE / U² (dimensionless) | [0.0, 0.5] |

### `StructuralSignal`

| Field | Type | Description | Valid Range |
|-------|------|-------------|-------------|
| `rms_amplitude_over_c` | `float` | RMS structural displacement / chord | [0.0, 1.0] |
| `dominant_frequency_hz` | `float` | Dominant frequency of structural displacement signal | [0.0, 1000] |
| `natural_frequency_hz` | `float` | Structural natural frequency f_n (in vacuo/still-fluid) | [0.1, 1000] |
| `mode_number` | `int` | Dominant structural mode number | [1, 6] |
| `camber` | `float` | Mean camber as fraction of chord | [0.0, 0.25] |

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