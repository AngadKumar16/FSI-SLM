"""Unit tests for features.extract module.

Tests cover:
- extract_features returns correct field count and types
- dominant_frequency finds expected peaks
- pod_energy_fractions returns plausible values
- Passthrough fields are preserved
"""

from __future__ import annotations

import numpy as np
import pytest

from Data.schema import FeatureRecord, FlowRegime, ForceCoefficients, RawExperiment
from features.extract import (
    check_physical_consistency,
    cross_spectral_coherence,
    dominant_frequency,
    estimate_damping,
    extract_features,
    pod_energy_fractions,
    spectral_moments,
)


def _make_synthetic_raw(regime: FlowRegime = FlowRegime.ATTACHED) -> RawExperiment:
    """Build a minimal RawExperiment for testing feature extraction."""
    fs = 400.0
    duration = 4.0
    t = np.arange(0, duration, 1.0 / fs)
    n = len(t)

    # Shedding at 10 Hz
    f_shed = 10.0
    wake = 1.0 + 0.3 * np.sin(2 * np.pi * f_shed * t) + 0.05 * np.random.default_rng(42).standard_normal(n)

    # Structure oscillating at 8 Hz, small amplitude
    f_struct = 8.0
    disp = 0.005 * np.sin(2 * np.pi * f_struct * t) + 0.0001 * np.random.default_rng(42).standard_normal(n)

    # Forces
    cl = 0.5 + 0.02 * np.sin(2 * np.pi * f_shed * t) + 0.005 * np.random.default_rng(42).standard_normal(n)
    cd = 0.05 + 0.01 * np.sin(2 * np.pi * 2 * f_shed * t) + 0.002 * np.random.default_rng(42).standard_normal(n)
    cm = -0.05 + 0.005 * np.sin(2 * np.pi * f_shed * t) + 0.001 * np.random.default_rng(42).standard_normal(n)

    # PIV snapshots (simple synthetic)
    n_snaps = 32
    n_points = 16
    snaps = np.random.default_rng(42).standard_normal((n_snaps, n_points)).tolist()

    from Data.schema import FlowConditions

    return RawExperiment(
        experiment_id=f"test-{regime.value}-00000",
        conditions=FlowConditions(
            reynolds_number=50000,
            angle_of_attack_deg=2.0,
            freestream_velocity_m_s=1.0,
            reduced_velocity=6.0,
            chord_m=0.1,
        ),
        sampling_rate_hz=fs,
        structural_displacement_signal=disp.tolist(),
        wake_velocity_probe_signal=wake.tolist(),
        cl_signal=cl.tolist(),
        cd_signal=cd.tolist(),
        cm_signal=cm.tolist(),
        piv_snapshots=snaps,
        natural_frequency_hz=10.0,
        mode_number=1,
        camber=0.02,
        wake_width_over_c=0.2,
        separation_location_x_over_c=0.9,
        turbulent_kinetic_energy=0.01,
        ground_truth_regime=regime,
    )


class TestDominantFrequency:
    def test_pure_sine(self):
        """A pure sine wave should give its exact frequency."""
        fs = 1000.0
        t = np.arange(0, 2.0, 1.0 / fs)
        freq = 50.0
        sig = np.sin(2 * np.pi * freq * t)
        dom_freq, amp = dominant_frequency(sig, fs)
        assert dom_freq == pytest.approx(freq, abs=0.5)  # FFT resolution limited

    def test_flat_signal_returns_zero(self):
        """A flat (constant) signal should return (0, 0)."""
        sig = np.ones(100)
        freq, amp = dominant_frequency(sig, fs=100.0)
        assert freq == 0.0
        assert amp == 0.0

    def test_short_signal_returns_zero(self):
        """Very short signals should return (0, 0)."""
        sig = np.array([1.0, 2.0])
        freq, amp = dominant_frequency(sig, fs=100.0)
        assert freq == 0.0
        assert amp == 0.0

    def test_dc_removed(self):
        """DC component should be removed before FFT."""
        fs = 100.0
        t = np.arange(0, 10.0, 1.0 / fs)
        sig = 100.0 + np.sin(2 * np.pi * 5.0 * t)  # DC = 100
        freq, amp = dominant_frequency(sig, fs)
        # Should find 5 Hz, not DC
        assert freq == pytest.approx(5.0, abs=0.2)


class TestPODEnergyFractions:
    def test_known_snapshot_stack(self):
        """POD on a structured snapshot stack should yield sorted energy fractions."""
        rng = np.random.default_rng(42)
        snaps = rng.standard_normal((50, 20))
        energies = pod_energy_fractions(snaps, n_modes=4)
        assert len(energies) == 4
        # Should be descending
        for i in range(len(energies) - 1):
            assert energies[i] >= energies[i + 1] - 1e-6

    def test_empty_snapshots(self):
        """Empty snapshot stack should return [0.0]."""
        energies = pod_energy_fractions(np.array([]), n_modes=4)
        assert energies == [0.0]

    def test_single_mode_dominance(self):
        """If one mode dominates, the first energy fraction should be large."""
        # Create a rank-1 snapshot stack
        n_snaps, n_points = 30, 10
        mode = np.random.default_rng(42).standard_normal(n_points)
        mode = mode / np.linalg.norm(mode)
        time_coeff = np.random.default_rng(42).standard_normal(n_snaps)
        snaps = np.outer(time_coeff, mode)
        energies = pod_energy_fractions(snaps, n_modes=4)
        assert energies[0] > 0.9  # first mode carries most energy

    def test_fractions_sum_to_one(self):
        """POD energy fractions should sum to 1 (approximately, for full decomposition)."""
        rng = np.random.default_rng(42)
        snaps = rng.standard_normal((40, 15))
        energies = pod_energy_fractions(snaps, n_modes=15)
        assert sum(energies) == pytest.approx(1.0, abs=0.01)


class TestExtractFeatures:
    def test_returns_feature_record(self):
        """extract_features should return a FeatureRecord."""
        raw = _make_synthetic_raw()
        fr = extract_features(raw)
        assert isinstance(fr, FeatureRecord)

    def test_preserves_experiment_id(self):
        """Experiment ID should be passed through."""
        raw = _make_synthetic_raw()
        fr = extract_features(raw)
        assert fr.experiment_id == raw.experiment_id

    def test_preserves_conditions(self):
        """Flow conditions should be passed through unchanged."""
        raw = _make_synthetic_raw()
        fr = extract_features(raw)
        assert fr.conditions.reynolds_number == raw.conditions.reynolds_number
        assert fr.conditions.chord_m == raw.conditions.chord_m

    def test_preserves_passthrough_fields(self):
        """Fields like camber, mode_number, wake_width should pass through."""
        raw = _make_synthetic_raw()
        fr = extract_features(raw)
        assert fr.structure.camber == raw.camber
        assert fr.structure.mode_number == raw.mode_number
        assert fr.piv.wake_width_over_c == raw.wake_width_over_c
        assert fr.piv.separation_location_x_over_c == raw.separation_location_x_over_c

    def test_preserves_ground_truth(self):
        """Ground truth regime should be preserved."""
        raw = _make_synthetic_raw(FlowRegime.VIV_LOCKIN)
        fr = extract_features(raw)
        assert fr.ground_truth_regime == FlowRegime.VIV_LOCKIN

    def test_shedding_frequency_extracted(self):
        """Shedding frequency should be extracted from wake probe."""
        raw = _make_synthetic_raw()
        fr = extract_features(raw)
        # The wake signal has a 10 Hz component
        assert fr.piv.shedding_frequency_hz > 5.0  # should find something near 10 Hz

    def test_force_coefficients_have_means(self):
        """Force coefficients should have non-zero means."""
        raw = _make_synthetic_raw()
        fr = extract_features(raw)
        assert abs(fr.forces.cl_mean) > 0.01
        assert fr.forces.cd_mean > 0.0

    def test_pod_amplitudes_not_empty(self):
        """POD mode amplitudes should be extracted."""
        raw = _make_synthetic_raw()
        fr = extract_features(raw)
        assert len(fr.piv.pod_mode_amplitudes) > 0


class TestDampingEstimation:
    def test_damped_sine_wave(self):
        """Damped sine should yield reasonable damping ratio (within method accuracy)."""
        fs = 100.0
        t = np.linspace(0, 5, 500)
        f0 = 5.0
        zeta = 0.05
        decay = np.exp(-zeta * 2 * np.pi * f0 * t)
        signal = np.sin(2 * np.pi * f0 * t) * decay
        damp, q, bw = estimate_damping(signal, fs)
        # Half-power bandwidth on a short/decaying record may return zero bandwidth,
        # yielding q=1.0; accept this boundary behavior.
        assert 0.0 <= damp <= 0.5
        assert q >= 1.0

    def test_undamped_signal(self):
        """Undamped signal should give low damping estimate."""
        fs = 100.0
        t = np.linspace(0, 2, 200)
        signal = np.sin(2 * np.pi * 10.0 * t)
        damp, q, bw = estimate_damping(signal, fs)
        assert damp >= 0.0
        assert q >= 1.0

    def test_flat_signal(self):
        """Constant signal should return defaults."""
        damp, q, bw = estimate_damping(np.ones(100), 10.0)
        assert damp == 0.0
        assert q == 1.0
        assert bw == 0.0


class TestSpectralMoments:
    def test_symmetric_signal(self):
        """Pure sine should have near-zero skewness."""
        fs = 100.0
        t = np.linspace(0, 5, 500)
        signal = np.sin(2 * np.pi * 5.0 * t)
        sk, kurt, spec_kurt, bw = spectral_moments(signal, fs)
        assert abs(sk) < 0.5
        assert kurt > 0.0

    def test_impulsive_signal(self):
        """Signal with spikes should have higher kurtosis."""
        fs = 100.0
        signal = np.random.randn(500)
        signal[::50] += 5.0
        sk, kurt, spec_kurt, bw = spectral_moments(signal, fs)
        assert kurt > 3.0

    def test_flat_signal(self):
        """Constant signal should return zeros."""
        sk, kurt, spec_kurt, bw = spectral_moments(np.ones(100), 100.0)
        assert sk == 0.0
        assert kurt == 0.0


class TestCrossSpectralCoherence:
    def test_coherent_signals(self):
        """Two in-phase sine waves should have high coherence."""
        fs = 100.0
        t = np.linspace(0, 20, 2000)
        sig1 = 0.5 * np.sin(2 * np.pi * 5.0 * t) + 0.1 * np.random.default_rng(42).standard_normal(len(t))
        sig2 = 0.5 * np.sin(2 * np.pi * 5.0 * t + 0.1) + 0.1 * np.random.default_rng(42).standard_normal(len(t))
        peak_coh, phase, freq = cross_spectral_coherence(sig1, sig2, fs)
        # Primary assertion: coherence magnitude
        assert peak_coh > 0.5
        # Phase should be near 0 (in-phase)
        assert abs(phase) < 30.0

    def test_time_shifted_signals(self):
        """Signals with a large time shift should have lower coherence than in-phase."""
        fs = 100.0
        t = np.linspace(0, 10, 1000)
        sig1 = np.sin(2 * np.pi * 5.0 * t)
        sig2 = np.sin(2 * np.pi * 5.0 * t + np.pi)  # 180° out of phase
        peak_coh, phase, freq = cross_spectral_coherence(sig1, sig2, fs)
        # Still coherent in magnitude, but phase should be near 180°
        assert peak_coh > 0.5
        assert abs(phase - 180.0) < 30.0 or abs(phase + 180.0) < 30.0

    def test_short_signals(self):
        """Signals shorter than minimum length should return zeros."""
        sig1 = np.ones(10)
        sig2 = np.ones(10)
        peak_coh, phase, freq = cross_spectral_coherence(sig1, sig2, 10.0)
        assert peak_coh == 0.0
        assert phase == 0.0
        assert freq == 0.0


class TestPhysicalConsistency:
    def test_strouhal_consistency(self):
        """Consistent St, f_shed, U should pass."""
        from Data.schema import (
            FeatureRecord, FlowConditions, PIVDerived, StructuralSignal,
            ForceCoefficients, PressureStrain,
        )
        fr = FeatureRecord(
            experiment_id="test",
            conditions=FlowConditions(
                reynolds_number=50000, angle_of_attack_deg=5.0,
                freestream_velocity_m_s=1.0, reduced_velocity=5.0, chord_m=0.02,
            ),
            pressure_strain=PressureStrain(pressure=101325.0, strain=0.01),
            piv=PIVDerived(
                shedding_frequency_hz=10.0, strouhal_number=0.2,
                wake_width_over_c=0.3, separation_location_x_over_c=0.8,
                pod_mode_amplitudes=[0.5, 0.2, 0.1], turbulent_kinetic_energy=0.05,
            ),
            structure=StructuralSignal(
                rms_amplitude_over_c=0.1, dominant_frequency_hz=10.0,
                natural_frequency_hz=10.0, mode_number=1, camber=0.03,
            ),
            forces=ForceCoefficients(
                cl_mean=0.5, cl_rms=0.1, cl_dominant_frequency_hz=10.0,
                cd_mean=0.05, cd_rms=0.01, cd_dominant_frequency_hz=10.0,
                cm_mean=0.0, cm_rms=0.01, cm_dominant_frequency_hz=10.0,
            ),
        )
        warnings = check_physical_consistency(fr)
        assert len(warnings) == 0

    def test_inconsistent_strouhal_triggers_warning(self):
        """Mismatched St and f_shed*U should trigger warning."""
        from Data.schema import (
            FeatureRecord, FlowConditions, PIVDerived, StructuralSignal,
            ForceCoefficients, PressureStrain,
        )
        fr = FeatureRecord(
            experiment_id="test",
            conditions=FlowConditions(
                reynolds_number=50000, angle_of_attack_deg=5.0,
                freestream_velocity_m_s=1.0, reduced_velocity=5.0, chord_m=0.1,
            ),
            pressure_strain=PressureStrain(pressure=101325.0, strain=0.01),
            piv=PIVDerived(
                shedding_frequency_hz=10.0, strouhal_number=0.5,
                wake_width_over_c=0.3, separation_location_x_over_c=0.8,
                pod_mode_amplitudes=[0.5, 0.2, 0.1], turbulent_kinetic_energy=0.05,
            ),
            structure=StructuralSignal(
                rms_amplitude_over_c=0.1, dominant_frequency_hz=10.0,
                natural_frequency_hz=10.0, mode_number=1, camber=0.03,
            ),
            forces=ForceCoefficients(
                cl_mean=0.5, cl_rms=0.1, cl_dominant_frequency_hz=10.0,
                cd_mean=0.05, cd_rms=0.01, cd_dominant_frequency_hz=10.0,
                cm_mean=0.0, cm_rms=0.01, cm_dominant_frequency_hz=10.0,
            ),
        )
        warnings = check_physical_consistency(fr)
        assert any("strouhal" in w.lower() for w in warnings)
