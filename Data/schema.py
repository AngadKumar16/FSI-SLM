"""Data contract for the FSI airfoil-experiment interpreter.

This module is the **interface specification** between this repository and any
upstream data source. The current repo only produces *synthetic* experiments,
but a future project will convert real PIV (Particle Image Velocimetry) +
structural + force measurements into objects that match these schemas. That
project does not need to read any other file in this repo -- if it emits
``RawExperiment`` objects (or, better, ``FeatureRecord`` objects) that validate
against the models below, everything downstream (feature extraction, dataset
building, training, RAG, evaluation) will work unchanged.

Two schemas are defined:

* :class:`RawExperiment` -- the closest thing to "what a rig/solver outputs".
  It holds the test conditions, the time-resolved signals (structural
  displacement, force coefficients, a wake velocity probe) and the raw PIV
  products (a snapshot stack for POD, plus scalar wake statistics). Feature
  extraction (:mod:`features.extract`) consumes this and produces a
  ``FeatureRecord``.

* :class:`FeatureRecord` -- the compact, model-facing record. This is the object
  that is serialized *verbatim* into the language-model prompt. It is the source
  of truth for an experiment's own measured numbers; RAG passages may add domain
  context but never override these values.

UNITS AND RANGES
----------------
Every field below documents its physical unit and a *valid range*. Ranges are
sanity bounds for low-speed water/wind-tunnel airfoil/hydrofoil/membrane FSI
experiments, not hard physical limits; values outside are allowed but will emit
a validation warning via :func:`FeatureRecord.range_warnings`. A future
real-data pipeline should treat persistent out-of-range values as a units bug
(e.g. Hz vs rad/s, mm vs m) before trusting them.

CONVENTIONS
-----------
* Frequencies are in Hz (cycles/second), never angular (rad/s).
* The Strouhal number uses the chord ``c`` as the length scale and the
  freestream velocity ``U`` as the velocity scale: ``St = f_shed * c / U``.
  (If your community uses thickness or diameter instead, convert before
  populating this schema and note it in ``notes``.)
* Displacement amplitudes are non-dimensionalized by chord ``c`` where a
  ``_over_c`` suffix is present.
* "Dominant frequency" of a signal means the frequency of its largest spectral
  peak (excluding the DC/mean component).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #
class FlowRegime(str, Enum):
    """The five flow/aeroelastic regimes this interpreter distinguishes.

    These are mutually exclusive top-level labels for the experiment as a whole.

    * ``ATTACHED``      -- boundary layer attached, low unsteadiness, small
                           structural response. Forces dominated by mean loading.
    * ``SEPARATED``     -- massive/leading-edge separation, broadband wake, but
                           *not* a sustained structural limit cycle.
    * ``VIV_LOCKIN``    -- vortex-induced vibration with frequency lock-in: the
                           shedding frequency is captured by the structural
                           natural frequency over a band of reduced velocity.
    * ``GALLOPING``     -- low-frequency, large-amplitude movement-induced
                           instability (translational, f << f_shed), typical of
                           non-circular bluff/stalled sections.
    * ``STALL_FLUTTER`` -- single-degree-of-freedom (often torsional) limit-cycle
                           oscillation driven by dynamic stall / negative
                           aerodynamic damping near/above static stall.
    """

    ATTACHED = "attached"
    SEPARATED = "separated"
    VIV_LOCKIN = "viv_lockin"
    GALLOPING = "galloping"
    STALL_FLUTTER = "stall_flutter"


class LockInStatus(str, Enum):
    """Relationship between vortex-shedding frequency and structural natural freq.

    * ``LOCKED``        -- |f_shed - f_n| / f_n <= lock-in tolerance (default 0.10):
                           the two frequencies have synchronized.
    * ``NEAR_LOCKIN``   -- within twice the tolerance: approaching synchronization.
    * ``UNLOCKED``      -- shedding and structural frequencies are independent.
    """

    LOCKED = "locked"
    NEAR_LOCKIN = "near_lockin"
    UNLOCKED = "unlocked"


class AmplitudeTrend(str, Enum):
    """Qualitative trend of structural response amplitude vs. reduced velocity.

    Describes how the RMS structural amplitude is changing across the operating
    point relative to neighbouring conditions (or, for a single synthetic point,
    the trend implied by its regime).
    """

    GROWING = "growing"
    SATURATED = "saturated"
    DECAYING = "decaying"
    NEGLIGIBLE = "negligible"


# --------------------------------------------------------------------------- #
# Sub-models
# --------------------------------------------------------------------------- #
class FlowConditions(BaseModel):
    """Free-stream / geometric operating point of the experiment."""

    model_config = ConfigDict(extra="forbid")

    reynolds_number: float = Field(
        ...,
        description="Chord-based Reynolds number Re = U*c/nu (dimensionless). "
        "Valid range ~ [1e3, 1e7] for typical low/medium-speed FSI rigs.",
    )
    angle_of_attack_deg: float = Field(
        ...,
        description="Geometric angle of attack in DEGREES. Valid range [-20, 30].",
    )
    freestream_velocity_m_s: float = Field(
        ...,
        description="Free-stream velocity U in METERS/SECOND. Valid range [0.1, 100].",
    )
    reduced_velocity: float = Field(
        ...,
        description="Reduced velocity U* = U / (f_n * c), dimensionless, where "
        "f_n is the structural natural frequency and c the chord. Lock-in for a "
        "VIV system typically occurs around U* ~ 5-8. Valid range [0.5, 30].",
    )
    chord_m: float = Field(
        ...,
        description="Chord length c in METERS (length scale for Re and St). "
        "Valid range [0.01, 2.0].",
    )


class StructuralSignal(BaseModel):
    """Foil/membrane displacement response."""

    model_config = ConfigDict(extra="forbid")

    rms_amplitude_over_c: float = Field(
        ...,
        description="RMS structural displacement normalized by chord (A_rms/c), "
        "dimensionless. Valid range [0.0, 1.0]; >~0.3 is a strong limit cycle.",
    )
    dominant_frequency_hz: float = Field(
        ...,
        description="Dominant frequency of the structural displacement signal in "
        "HERTZ. Valid range [0.0, 1000].",
    )
    natural_frequency_hz: float = Field(
        ...,
        description="Structural natural frequency f_n (in vacuo/still-fluid) in "
        "HERTZ. Used for the lock-in comparison. Valid range [0.1, 1000].",
    )
    mode_number: int = Field(
        ...,
        description="Dominant structural mode number (1 = first bending/heave, "
        "2 = second, ...). Valid range [1, 6].",
    )
    camber: float = Field(
        ...,
        description="Mean camber as a fraction of chord (max-camber/chord), "
        "dimensionless. Valid range [0.0, 0.25] (membranes inflate to high camber).",
    )


class PressureStrain(BaseModel):
    """Pressure and strain measurements from the experiment."""

    model_config = ConfigDict(extra="forbid")

    pressure: float = Field(
        ...,
        description="Ambient pressure in Pa. Valid range [0, 500000].",
    )
    strain: float = Field(
        ...,
        description="Strain (dimensionless or engineering strain). Valid range [0, 10].",
    )


class ForceCoefficients(BaseModel):
    """Mean / RMS and dominant frequency of the aerodynamic force coefficients."""

    model_config = ConfigDict(extra="forbid")

    cl_mean: float = Field(..., description="Mean lift coefficient (dimensionless). Range [-2.5, 2.5].")
    cl_rms: float = Field(..., description="RMS of lift-coefficient fluctuation (dimensionless). Range [0, 2].")
    cl_dominant_frequency_hz: float = Field(..., description="Dominant frequency of Cl(t) in HERTZ. Range [0, 1000].")
    cd_mean: float = Field(..., description="Mean drag coefficient (dimensionless). Range [0, 3].")
    cd_rms: float = Field(..., description="RMS of drag-coefficient fluctuation (dimensionless). Range [0, 2].")
    cd_dominant_frequency_hz: float = Field(..., description="Dominant frequency of Cd(t) in HERTZ. Range [0, 1000].")
    cm_mean: float = Field(..., description="Mean pitching-moment coefficient (dimensionless). Range [-1.5, 1.5].")
    cm_rms: float = Field(..., description="RMS of moment-coefficient fluctuation (dimensionless). Range [0, 1.5].")
    cm_dominant_frequency_hz: float = Field(..., description="Dominant frequency of Cm(t) in HERTZ. Range [0, 1000].")


class PIVDerived(BaseModel):
    """Quantities derived from the PIV velocity fields (scalar summaries)."""

    model_config = ConfigDict(extra="forbid")

    shedding_frequency_hz: float = Field(
        ...,
        description="Vortex-shedding frequency f_shed (largest peak of a wake "
        "velocity probe spectrum) in HERTZ. Valid range [0, 1000].",
    )
    strouhal_number: float = Field(
        ...,
        description="St = f_shed * c / U (chord-based), dimensionless. For airfoils "
        "near stall St ~ 0.15-0.25 is typical. Valid range [0.0, 1.0].",
    )
    wake_width_over_c: float = Field(
        ...,
        description="Wake half/full-width normalized by chord, dimensionless. "
        "Valid range [0.0, 3.0]; grows with separation.",
    )
    separation_location_x_over_c: float = Field(
        ...,
        description="Chordwise separation point as a fraction of chord from the "
        "leading edge (0 = LE, 1 = TE). 1.0 means effectively attached. Range [0, 1].",
    )
    pod_mode_amplitudes: List[float] = Field(
        ...,
        description="Energy fraction captured by the leading POD modes (from SVD "
        "of the PIV snapshot stack), most-energetic first. Each in [0,1], the "
        "list need not sum to 1 (only the leading modes are kept). 1-6 entries.",
    )
    turbulent_kinetic_energy: float = Field(
        ...,
        description="Wake turbulent kinetic energy TKE = 0.5*<u'^2+v'^2+w'^2>, "
        "non-dimensionalized by U^2 (so dimensionless). Valid range [0.0, 0.5].",
    )


# --------------------------------------------------------------------------- #
# Raw experiment (upstream of feature extraction)
# --------------------------------------------------------------------------- #
class RawExperiment(BaseModel):
    """Time-resolved raw outputs of a single experimental run (or simulation).

    This is what a rig/solver -- or the future real-data pipeline -- should
    produce *before* feature extraction. The arrays are stored as plain Python
    lists so the object is trivially JSON-serializable.

    The future PIV pipeline only needs to populate the fields it actually
    measures; any field it cannot provide should be passed through, and
    :mod:`features.extract` documents which raw fields each feature depends on.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(..., description="Unique run identifier; the split is BY this id.")
    conditions: FlowConditions

    sampling_rate_hz: float = Field(
        ..., description="Acquisition sampling rate of the time signals in HERTZ."
    )
    # Time-resolved signals (1-D, all sharing the same sampling_rate_hz timebase).
    structural_displacement_signal: List[float] = Field(
        ..., description="Structural tip displacement vs time, normalized by chord."
    )
    wake_velocity_probe_signal: List[float] = Field(
        ...,
        description="Streamwise velocity at a fixed wake probe vs time (m/s); its "
        "spectrum gives the shedding frequency.",
    )
    cl_signal: List[float] = Field(..., description="Lift-coefficient time series (dimensionless).")
    cd_signal: List[float] = Field(..., description="Drag-coefficient time series (dimensionless).")
    cm_signal: List[float] = Field(..., description="Moment-coefficient time series (dimensionless).")

    # PIV snapshot stack for POD: shape (n_snapshots, n_points). Flattened fields.
    piv_snapshots: List[List[float]] = Field(
        ...,
        description="Fluctuating-velocity PIV snapshots, list of n_snapshots rows, "
        "each row a flattened spatial field (n_points). Used for POD via SVD.",
    )

    # Static structural / geometric properties needed downstream.
    natural_frequency_hz: float = Field(..., description="Structural natural frequency f_n in HERTZ.")
    mode_number: int = Field(..., description="Dominant structural mode number.")
    camber: float = Field(..., description="Mean camber fraction of chord.")

    # Scalar PIV summaries that are measured, not FFT-derived.
    wake_width_over_c: float = Field(..., description="Wake width normalized by chord.")
    separation_location_x_over_c: float = Field(..., description="Separation point x/c (0=LE,1=TE).")
    turbulent_kinetic_energy: float = Field(..., description="Wake TKE / U^2 (dimensionless).")

    # Pressure and strain measurements.
    pressure: float = Field(default=101325.0, description="Ambient pressure in Pa. Default = standard atmospheric pressure.")
    strain: float = Field(default=0.0, description="Strain (dimensionless or engineering strain). Default = 0.")

    # Optional ground-truth label carried alongside synthetic data ONLY. Real
    # experiments leave this None; it is never an input to the model.
    ground_truth_regime: Optional[FlowRegime] = Field(
        default=None,
        description="True regime, populated by the synthetic generator for "
        "evaluation. MUST be None for real experiments (it is the answer key).",
    )
    notes: Optional[str] = Field(default=None, description="Free-text provenance / caveats.")


# --------------------------------------------------------------------------- #
# Feature record (model-facing)
# --------------------------------------------------------------------------- #
class FeatureRecord(BaseModel):
    """Compact, model-facing feature record -- the prompt's source of truth.

    Produced by :mod:`features.extract` from a :class:`RawExperiment`. This is the
    object serialized verbatim into the language-model prompt. Keep it stable: the
    SFT model is trained against :meth:`to_prompt_block`, so changing that format
    requires re-building the dataset and re-training.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    conditions: FlowConditions
    pressure_strain: PressureStrain
    piv: PIVDerived
    structure: StructuralSignal
    forces: ForceCoefficients

    # Ground truth carried only for synthetic data / evaluation. Never serialized
    # into the model input (see to_prompt_block). None for real experiments.
    ground_truth_regime: Optional[FlowRegime] = Field(
        default=None,
        description="Answer key for evaluation only; excluded from the prompt.",
    )

    # ------------------------------------------------------------------ #
    # Derived convenience properties
    # ------------------------------------------------------------------ #
    def lock_in_status(self, tol: float = 0.10) -> LockInStatus:
        """Classify lock-in by comparing shedding freq to structural natural freq.

        ``tol`` is the relative-frequency tolerance for "locked" (default 10%).
        """
        f_shed = self.piv.shedding_frequency_hz
        f_n = self.structure.natural_frequency_hz
        if f_n <= 0:
            return LockInStatus.UNLOCKED
        rel = abs(f_shed - f_n) / f_n
        if rel <= tol:
            return LockInStatus.LOCKED
        if rel <= 2 * tol:
            return LockInStatus.NEAR_LOCKIN
        return LockInStatus.UNLOCKED

    # ------------------------------------------------------------------ #
    # Canonical serialization used VERBATIM in the model prompt
    # ------------------------------------------------------------------ #
    def to_prompt_block(self) -> str:
        """Serialize the measured features into the deterministic text block that
        is injected verbatim into the model input.

        This is intentionally a flat, units-annotated ``KEY: value`` block (not
        JSON) so the SFT target format is easy for the model to attend to and the
        evaluator can locate fields. The ground-truth regime is deliberately
        excluded -- the model must infer it.
        """
        c = self.conditions
        p = self.piv
        s = self.structure
        f = self.forces
        pod = ", ".join(f"{a:.3f}" for a in p.pod_mode_amplitudes)
        lines = [
            "FSI_FEATURE_RECORD v1",
            f"experiment_id: {self.experiment_id}",
            "[conditions]",
            f"reynolds_number: {c.reynolds_number:.4g}",
            f"angle_of_attack_deg: {c.angle_of_attack_deg:.3f}",
            f"freestream_velocity_m_s: {c.freestream_velocity_m_s:.4g}",
            f"reduced_velocity_Ustar: {c.reduced_velocity:.3f}",
            f"chord_m: {c.chord_m:.4g}",
            "[piv_derived]",
            f"shedding_frequency_hz: {p.shedding_frequency_hz:.4g}",
            f"strouhal_number_St: {p.strouhal_number:.4f}",
            f"wake_width_over_c: {p.wake_width_over_c:.3f}",
            f"separation_location_x_over_c: {p.separation_location_x_over_c:.3f}",
            f"pod_mode_amplitudes: [{pod}]",
            f"turbulent_kinetic_energy: {p.turbulent_kinetic_energy:.4f}",
            "[structure]",
            f"rms_amplitude_over_c: {s.rms_amplitude_over_c:.4f}",
            f"dominant_frequency_hz: {s.dominant_frequency_hz:.4g}",
            f"natural_frequency_hz: {s.natural_frequency_hz:.4g}",
            f"mode_number: {s.mode_number}",
            f"camber: {s.camber:.4f}",
            "[forces]",
            f"cl_mean: {f.cl_mean:.4f}",
            f"cl_rms: {f.cl_rms:.4f}",
            f"cl_dominant_frequency_hz: {f.cl_dominant_frequency_hz:.4g}",
            f"cd_mean: {f.cd_mean:.4f}",
            f"cd_rms: {f.cd_rms:.4f}",
            f"cd_dominant_frequency_hz: {f.cd_dominant_frequency_hz:.4g}",
            f"cm_mean: {f.cm_mean:.4f}",
            f"cm_rms: {f.cm_rms:.4f}",
            f"cm_dominant_frequency_hz: {f.cm_dominant_frequency_hz:.4g}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Lightweight range validation (units-bug detector for real data)
    # ------------------------------------------------------------------ #
    def range_warnings(self) -> List[str]:
        """Return human-readable warnings for any field outside its valid range.

        Returns an empty list when everything is in range. A future real-data
        pipeline should run this on every record; many out-of-range values at
        once almost always means a units mismatch upstream.
        """
        warns: List[str] = []

        def chk(name: str, value: float, lo: float, hi: float) -> None:
            if not (lo <= value <= hi):
                warns.append(f"{name}={value:g} outside expected [{lo:g}, {hi:g}]")

        c, p, s, f = self.conditions, self.piv, self.structure, self.forces
        chk("reynolds_number", c.reynolds_number, 1e3, 1e7)
        chk("angle_of_attack_deg", c.angle_of_attack_deg, -20, 30)
        chk("freestream_velocity_m_s", c.freestream_velocity_m_s, 0.1, 100)
        chk("reduced_velocity", c.reduced_velocity, 0.5, 30)
        chk("chord_m", c.chord_m, 0.01, 2.0)
        chk("strouhal_number", p.strouhal_number, 0.0, 1.0)
        chk("separation_location_x_over_c", p.separation_location_x_over_c, 0.0, 1.0)
        chk("turbulent_kinetic_energy", p.turbulent_kinetic_energy, 0.0, 0.5)
        chk("rms_amplitude_over_c", s.rms_amplitude_over_c, 0.0, 1.0)
        chk("mode_number", s.mode_number, 1, 6)
        chk("camber", s.camber, 0.0, 0.25)
        return warns


# --------------------------------------------------------------------------- #
# Helpers for (de)serializing numpy-heavy raw experiments
# --------------------------------------------------------------------------- #
def snapshots_to_array(raw: RawExperiment) -> np.ndarray:
    """Return the PIV snapshot stack as a (n_snapshots, n_points) float array."""
    return np.asarray(raw.piv_snapshots, dtype=float)


__all__ = [
    "FlowRegime",
    "LockInStatus",
    "AmplitudeTrend",
    "FlowConditions",
    "StructuralSignal",
    "ForceCoefficients",
    "PIVDerived",
    "RawExperiment",
    "FeatureRecord",
    "snapshots_to_array",
]