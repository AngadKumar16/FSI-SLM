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

from data.schema import FeatureRecord, FlowRegime, ForceCoefficients, RawExperiment
from features.extract import (
    dominant_frequency,
    extract_features,
    pod_energy_fractions,
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

    from data.schema import FlowConditions

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