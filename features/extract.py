"""Feature extraction: RawExperiment -> FeatureRecord.

This module turns a time-resolved :class:`~data.schema.RawExperiment` into the
compact, model-facing :class:`~data.schema.FeatureRecord`. It is deliberately
generic signal processing (FFT for dominant frequencies, St from the shedding
peak, POD via SVD of the snapshot stack) so the *same code path* will work on a
future real-data ``RawExperiment`` without modification.

Feature -> raw-field dependency map (useful for the future PIV pipeline):

* shedding_frequency_hz   <- FFT peak of ``wake_velocity_probe_signal``
* strouhal_number         <- f_shed * chord / U
* pod_mode_amplitudes     <- SVD of ``piv_snapshots`` (energy fractions)
* structure.dominant_freq <- FFT peak of ``structural_displacement_signal``
* forces.*_dominant_freq  <- FFT peak of cl/cd/cm signals
* rms_amplitude_over_c    <- std of ``structural_displacement_signal``
* cl/cd/cm mean & rms     <- mean / std of the corresponding signals
* wake_width / sep / tke / camber / mode_number -> passed through from raw
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from Data.schema import (
    FeatureRecord,
    ForceCoefficients,
    PIVDerived,
    PressureStrain,
    RawExperiment,
    StructuralSignal,
    snapshots_to_array,
)


def dominant_frequency(signal: np.ndarray, fs: float) -> Tuple[float, float]:
    """Return (dominant_frequency_hz, amplitude) of a real signal via FFT.

    The DC/mean component is removed first so the dominant frequency is the
    largest *fluctuation* peak. Returns (0.0, 0.0) for a flat signal.
    """
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 4 or not np.any(np.abs(x) > 0):
        return 0.0, 0.0
    # Hann window reduces spectral leakage for these finite records.
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Ignore the zero-frequency bin.
    if spectrum.size <= 1:
        return 0.0, 0.0
    k = int(np.argmax(spectrum[1:])) + 1
    return float(freqs[k]), float(spectrum[k])


def pod_energy_fractions(snapshots: np.ndarray, n_modes: int = 4) -> List[float]:
    """POD via SVD: return the energy fraction of the leading ``n_modes`` modes.

    The snapshot stack is (n_snapshots, n_points). We mean-subtract along the time
    axis and take the SVD; squared singular values give modal energy, normalized
    by total energy to yield fractions in [0, 1] (most-energetic first).
    """
    s = np.asarray(snapshots, dtype=float)
    if s.ndim != 2 or s.size == 0:
        return [0.0]
    s = s - s.mean(axis=0, keepdims=True)
    # economy SVD; singular values squared ~ POD modal energy
    try:
        sv = np.linalg.svd(s, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return [0.0]
    energy = sv**2
    total = energy.sum()
    if total <= 0:
        return [0.0]
    fracs = (energy / total)[:n_modes]
    return [float(round(f, 4)) for f in fracs]


def extract_features(raw: RawExperiment) -> FeatureRecord:
    """Extract a :class:`FeatureRecord` from a :class:`RawExperiment`."""
    fs = raw.sampling_rate_hz
    c = raw.conditions

    # --- PIV-derived ---
    f_shed, _ = dominant_frequency(np.asarray(raw.wake_velocity_probe_signal), fs)
    st = f_shed * c.chord_m / c.freestream_velocity_m_s if c.freestream_velocity_m_s > 0 else 0.0
    pod = pod_energy_fractions(snapshots_to_array(raw), n_modes=4)

    piv = PIVDerived(
        shedding_frequency_hz=f_shed,
        strouhal_number=float(np.clip(st, 0.0, 1.0)),
        wake_width_over_c=raw.wake_width_over_c,
        separation_location_x_over_c=raw.separation_location_x_over_c,
        pod_mode_amplitudes=pod,
        turbulent_kinetic_energy=raw.turbulent_kinetic_energy,
    )

    # --- structure ---
    disp = np.asarray(raw.structural_displacement_signal, dtype=float)
    f_struct, _ = dominant_frequency(disp, fs)
    structure = StructuralSignal(
        rms_amplitude_over_c=float(disp.std()),
        dominant_frequency_hz=f_struct,
        natural_frequency_hz=raw.natural_frequency_hz,
        mode_number=raw.mode_number,
        camber=raw.camber,
    )

    # --- forces ---
    cl = np.asarray(raw.cl_signal, dtype=float)
    cd = np.asarray(raw.cd_signal, dtype=float)
    cm = np.asarray(raw.cm_signal, dtype=float)
    f_cl, _ = dominant_frequency(cl, fs)
    f_cd, _ = dominant_frequency(cd, fs)
    f_cm, _ = dominant_frequency(cm, fs)
    forces = ForceCoefficients(
        cl_mean=float(cl.mean()),
        cl_rms=float(cl.std()),
        cl_dominant_frequency_hz=f_cl,
        cd_mean=float(cd.mean()),
        cd_rms=float(cd.std()),
        cd_dominant_frequency_hz=f_cd,
        cm_mean=float(cm.mean()),
        cm_rms=float(cm.std()),
        cm_dominant_frequency_hz=f_cm,
    )

    return FeatureRecord(
        experiment_id=raw.experiment_id,
        conditions=c,
        pressure_strain=PressureStrain(
            pressure=raw.pressure,
            strain=raw.strain,
        ),
        piv=piv,
        structure=structure,
        forces=forces,
        ground_truth_regime=raw.ground_truth_regime,
    )


__all__ = ["extract_features", "dominant_frequency", "pod_energy_fractions"]
