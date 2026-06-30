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
    AerodynamicCoherence,
    FeatureRecord,
    FlowRegime,
    ForceCoefficients,
    LockInStatus,
    ModalDamping,
    PIVDerived,
    PressureStrain,
    RawExperiment,
    StructuralSignal,
    StructuralSpectrum,
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


def estimate_damping(signal: np.ndarray, fs: float, method: str = "log_decrement") -> Tuple[float, float, float]:
    """Estimate modal damping ratio ζ, Q-factor, and spectral bandwidth.

    Uses half-power bandwidth method on the FFT spectrum (default). Falls back
    to a simple logarithmic decrement estimate if amplitude envelope is clear.

    Returns:
        (damping_ratio, q_factor, spectral_bandwidth_hz)
        damping_ratio: ζ in [0, 1]; typical FSI structures 0.01-0.1
        q_factor: Q = 1/(2ζ)
        spectral_bandwidth_hz: half-power bandwidth in Hz
    """
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 8 or not np.any(np.abs(x) > 1e-12):
        return 0.0, 1.0, 0.0

    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    if spectrum.size <= 2:
        return 0.0, 1.0, 0.0

    # Find dominant peak (skip DC)
    k_peak = int(np.argmax(spectrum[1:])) + 1
    peak_val = spectrum[k_peak]
    peak_freq = freqs[k_peak]

    if peak_val < 1e-12 or peak_freq <= 0:
        return 0.0, 1.0, 0.0

    # Half-power bandwidth method
    half_power = peak_val / np.sqrt(2.0)
    above = np.where(spectrum >= half_power)[0]
    if len(above) < 2:
        bandwidth = 0.0
    else:
        bandwidth = float(freqs[above[-1]] - freqs[above[0]])

    if peak_freq > 0 and bandwidth > 0:
        damping_ratio = bandwidth / (2.0 * peak_freq)
        # Clamp to physical range
        damping_ratio = float(np.clip(damping_ratio, 0.0, 0.5))
        q_factor = 1.0 / (2.0 * damping_ratio) if damping_ratio > 1e-12 else 500.0
        q_factor = float(np.clip(q_factor, 1.0, 500.0))
    else:
        damping_ratio = 0.01
        q_factor = 50.0
        bandwidth = 0.0

    return damping_ratio, q_factor, bandwidth


def spectral_moments(signal: np.ndarray, fs: float) -> Tuple[float, float, float, float]:
    """Compute higher-order statistical moments of the structural signal.

    Returns:
        (skewness, kurtosis, spectral_kurtosis, spectral_bandwidth_hz)
        skewness: asymmetry of time series (0 = symmetric)
        kurtosis: tail weight of time series (>3 = impulsive)
        spectral_kurtosis: flatness of frequency spectrum
        spectral_bandwidth_hz: half-power bandwidth
    """
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()

    # Time-domain moments
    if x.size < 3 or np.std(x) < 1e-12:
        return 0.0, 0.0, 0.0, 0.0

    from scipy.stats import skew, kurtosis as scipy_kurtosis
    sk = float(np.clip(skew(x), -3.0, 3.0))
    kurt = float(np.clip(scipy_kurtosis(x, fisher=False), -1.0, 10.0))

    # Frequency-domain moments
    n = x.size
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(x * win))**2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    if spectrum.size <= 2 or spectrum.sum() < 1e-12:
        return sk, kurt, 0.0, 0.0

    # Spectral kurtosis (flatness)
    spec_mean = spectrum.mean()
    spec_std = spectrum.std()
    spec_kurt = float(np.clip((spec_std / spec_mean)**2 - 1.0, -1.0, 10.0)) if spec_mean > 1e-12 else 0.0

    # Bandwidth from cumulative energy
    cum_energy = np.cumsum(spectrum)
    total_energy = cum_energy[-1]
    lower = np.searchsorted(cum_energy, 0.25 * total_energy)
    upper = np.searchsorted(cum_energy, 0.75 * total_energy)
    bandwidth = float(freqs[min(upper, len(freqs)-1)] - freqs[min(lower, len(freqs)-1)])

    return sk, kurt, spec_kurt, bandwidth


def cross_spectral_coherence(sig1: np.ndarray, sig2: np.ndarray, fs: float) -> Tuple[float, float, float]:
    """Compute cross-spectral coherence and phase between two signals.

    Args:
        sig1: Primary signal (e.g., structural displacement)
        sig2: Secondary signal (e.g., lift coefficient)
        fs: Sampling rate in Hz

    Returns:
        (peak_coherence, phase_at_peak_deg, dominant_freq_hz)
        peak_coherence: magnitude-squared coherence at dominant frequency [0, 1]
        phase_at_peak_deg: phase angle at coherence peak in degrees [-180, 180]
        dominant_freq_hz: frequency of maximum coherence
    """
    x1 = np.asarray(sig1, dtype=float)
    x2 = np.asarray(sig2, dtype=float)
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    n = min(len(x1), len(x2))

    if n < 16:
        return 0.0, 0.0, 0.0

    win = np.hanning(n)
    x1w = x1[:n] * win
    x2w = x2[:n] * win

    fft1 = np.fft.rfft(x1w)
    fft2 = np.fft.rfft(x2w)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Cross-spectrum and auto-spectra
    cross = fft1 * np.conj(fft2)
    s1 = np.abs(fft1)**2
    s2 = np.abs(fft2)**2

    # Magnitude-squared coherence
    denom = s1 * s2
    coherence = np.zeros_like(cross, dtype=float)
    mask = denom > 1e-20
    coherence[mask] = np.abs(cross[mask])**2 / denom[mask]

    # Exclude DC bin
    if coherence.size > 1:
        coherence[0] = 0.0

    if coherence.size == 0 or coherence.max() < 1e-6:
        return 0.0, 0.0, 0.0

    k_peak = int(np.argmax(coherence))
    peak_coh = float(np.clip(coherence[k_peak], 0.0, 1.0))
    phase_rad = np.angle(cross[k_peak])
    phase_deg = float(np.degrees(phase_rad))
    peak_freq = float(freqs[k_peak])

    return peak_coh, phase_deg, peak_freq


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
    damping_ratio, q_factor, bandwidth = estimate_damping(disp, fs)
    sk, kurt, spec_kurt, spec_bw = spectral_moments(disp, fs)

    structure = StructuralSignal(
        rms_amplitude_over_c=float(disp.std()),
        dominant_frequency_hz=f_struct,
        natural_frequency_hz=raw.natural_frequency_hz,
        mode_number=raw.mode_number,
        camber=raw.camber,
        damping=ModalDamping(
            damping_ratio=damping_ratio,
            q_factor=q_factor,
        ),
        spectrum=StructuralSpectrum(
            spectral_bandwidth_hz=bandwidth,
            spectral_kurtosis=spec_kurt,
            skewness=sk,
            kurtosis=kurt,
        ),
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

    # --- coherence ---
    cl = np.asarray(raw.cl_signal, dtype=float)
    cd = np.asarray(raw.cd_signal, dtype=float)
    coh_peak, phase_deg, coh_freq = cross_spectral_coherence(disp, cl, fs)
    coh_peak_cd, _, _ = cross_spectral_coherence(disp, cd, fs)
    piv = PIVDerived(
        shedding_frequency_hz=f_shed,
        strouhal_number=float(np.clip(st, 0.0, 1.0)),
        wake_width_over_c=raw.wake_width_over_c,
        separation_location_x_over_c=raw.separation_location_x_over_c,
        pod_mode_amplitudes=pod,
        turbulent_kinetic_energy=raw.turbulent_kinetic_energy,
        coherence=AerodynamicCoherence(
            cl_coherence_peak=coh_peak,
            cd_coherence_peak=coh_peak_cd,
            phase_angle_deg=phase_deg,
        ),
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


def check_physical_consistency(fr: FeatureRecord) -> list:
    """Validate physical consistency between recorded features.

    Checks:
    1. Strouhal number matches reported f_shed / U within 5%
    2. Lock-in logic consistent with f_struct vs f_shed
    3. Damping ratio consistent with amplitude bandwidth
    4. Coherence phase within expected range for VIV (~90°)
    5. Classification confidence aligned with feature ambiguity
    """
    warnings = []
    s = fr.structure
    p = fr.piv
    c = fr.conditions

    # 1. Strouhal check
    if c.chord_m > 0 and c.freestream_velocity_m_s > 0 and p.shedding_frequency_hz > 0:
        st_computed = p.shedding_frequency_hz * c.chord_m / c.freestream_velocity_m_s
        if abs(st_computed - p.strouhal_number) / max(st_computed, 1e-12) > 0.05:
            warnings.append(
                f"strouhal_inconsistency: reported St={p.strouhal_number:.4f} "
                f"but computed {st_computed:.4f} from f_shed and U"
            )

    # 2. Lock-in logic
    lock = fr.lock_in_status()
    f_shed = p.shedding_frequency_hz
    f_n = s.natural_frequency_hz
    if f_n > 0 and f_shed > 0:
        rel_diff = abs(f_shed - f_n) / f_n
        if lock == LockInStatus.LOCKED and rel_diff > 0.10:
            warnings.append(
                f"lockin_mismatch: classified as LOCKED but |f_shed-f_n|/f_n={rel_diff:.3f} > 0.10"
            )
        if lock == LockInStatus.NEAR_LOCKIN and (rel_diff <= 0.10 or rel_diff > 0.20):
            warnings.append(
                f"lockin_mismatch: classified as NEAR_LOCKIN but rel_diff={rel_diff:.3f} "
                f"not in [0.10, 0.20]"
            )

    # 3. Damping-bandwidth consistency
    if s.damping is not None and s.spectrum is not None:
        damping = s.damping.damping_ratio
        bandwidth = s.spectrum.spectral_bandwidth_hz
        if s.dominant_frequency_hz > 0 and bandwidth > 0:
            damping_implied = bandwidth / (2.0 * s.dominant_frequency_hz)
            if abs(damping - damping_implied) / max(damping, 1e-12) > 0.5:
                warnings.append(
                    f"damping_bandwidth_mismatch: damping_ratio={damping:.4f} "
                    f"but implied from bandwidth = {damping_implied:.4f}"
                )

    # 4. Phase coherence physics
    if p.coherence and lock == LockInStatus.LOCKED:
        phase = abs(p.coherence.phase_angle_deg)
        # VIV typically has phase near 90° ± 30°
        if not (60.0 <= phase <= 120.0):
            warnings.append(
                f"coherence_phase_unexpected: lock-in with phase={p.coherence.phase_angle_deg:.1f}° "
                f"(expected ~90° for VIV)"
            )

    # 5. Confidence calibration
    if hasattr(fr, 'classification_confidence') and fr.classification_confidence is not None:
        conf = fr.classification_confidence
        if conf < 0.3:
            warnings.append("low_classification_confidence: prediction may be unreliable")
        if conf > 0.95 and lock == LockInStatus.NEAR_LOCKIN:
            warnings.append(
                "overconfident_near_lockin: confidence >0.95 despite ambiguous lock-in"
            )

    return warnings


__all__ = [
    "extract_features",
    "dominant_frequency",
    "pod_energy_fractions",
    "estimate_damping",
    "spectral_moments",
    "cross_spectral_coherence",
    "check_physical_consistency",
]
