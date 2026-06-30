"""Unit tests for data.describe module.

Tests cover:
- describe returns deterministic output for canonical FeatureRecord
- classify_regime matches ground_truth_regime
- amplitude_trend returns correct enum
- mode_shape_phrase formatting
- Mechanism note presence
"""

from __future__ import annotations

import numpy as np
import pytest

from Data.schema import (
    AmplitudeTrend,
    FeatureRecord,
    FlowConditions,
    FlowRegime,
    ForceCoefficients,
    PIVDerived,
    PressureStrain,
    StructuralSignal,
)
from Data.describe import (
    amplitude_trend,
    classify_regime,
    classify_regime_confidence,
    describe,
)


def _make_fr(regime: FlowRegime = FlowRegime.ATTACHED) -> FeatureRecord:
    """Build a FeatureRecord matching a given regime."""
    base = {
        "experiment_id": f"test-{regime.value}-00000",
        "conditions": FlowConditions(
            reynolds_number=50000,
            angle_of_attack_deg=2.0,
            freestream_velocity_m_s=1.0,
            reduced_velocity=6.0,
            chord_m=0.1,
        ),
        "pressure_strain": PressureStrain(
            pressure=101325.0,
            strain=0.01,
        ),
        "forces": ForceCoefficients(
            cl_mean=0.5, cl_rms=0.02, cl_dominant_frequency_hz=10.0,
            cd_mean=0.05, cd_rms=0.01, cd_dominant_frequency_hz=20.0,
            cm_mean=-0.05, cm_rms=0.005, cm_dominant_frequency_hz=10.0,
        ),
    }

    if regime == FlowRegime.ATTACHED:
        base["piv"] = PIVDerived(
            shedding_frequency_hz=10.0, strouhal_number=0.2,
            wake_width_over_c=0.2, separation_location_x_over_c=0.95,
            pod_mode_amplitudes=[0.5, 0.2, 0.1, 0.05], turbulent_kinetic_energy=0.01,
        )
        base["structure"] = StructuralSignal(
            rms_amplitude_over_c=0.003, dominant_frequency_hz=8.0,
            natural_frequency_hz=30.0, mode_number=1, camber=0.02,
        )
    elif regime == FlowRegime.VIV_LOCKIN:
        base["conditions"].angle_of_attack_deg = 4.0
        base["piv"] = PIVDerived(
            shedding_frequency_hz=10.0, strouhal_number=0.2,
            wake_width_over_c=0.6, separation_location_x_over_c=0.6,
            pod_mode_amplitudes=[0.7, 0.15, 0.05, 0.03], turbulent_kinetic_energy=0.08,
        )
        base["structure"] = StructuralSignal(
            rms_amplitude_over_c=0.35, dominant_frequency_hz=10.0,
            natural_frequency_hz=10.0, mode_number=1, camber=0.05,
        )
    elif regime == FlowRegime.GALLOPING:
        base["conditions"].angle_of_attack_deg = 8.0
        base["piv"] = PIVDerived(
            shedding_frequency_hz=20.0, strouhal_number=0.15,
            wake_width_over_c=1.0, separation_location_x_over_c=0.4,
            pod_mode_amplitudes=[0.65, 0.18, 0.06, 0.03], turbulent_kinetic_energy=0.1,
        )
        base["structure"] = StructuralSignal(
            rms_amplitude_over_c=0.5, dominant_frequency_hz=5.0,
            natural_frequency_hz=4.0, mode_number=1, camber=0.03,
        )
    elif regime == FlowRegime.STALL_FLUTTER:
        base["conditions"].angle_of_attack_deg = 16.0
        base["piv"] = PIVDerived(
            shedding_frequency_hz=12.0, strouhal_number=0.18,
            wake_width_over_c=1.2, separation_location_x_over_c=0.25,
            pod_mode_amplitudes=[0.4, 0.2, 0.1, 0.05], turbulent_kinetic_energy=0.15,
        )
        base["structure"] = StructuralSignal(
            rms_amplitude_over_c=0.3, dominant_frequency_hz=10.0,
            natural_frequency_hz=14.0, mode_number=2, camber=0.04,
        )
    else:  # SEPARATED
        base["conditions"].angle_of_attack_deg = 14.0
        base["piv"] = PIVDerived(
            shedding_frequency_hz=10.0, strouhal_number=0.2,
            wake_width_over_c=1.0, separation_location_x_over_c=0.3,
            pod_mode_amplitudes=[0.35, 0.2, 0.1, 0.05], turbulent_kinetic_energy=0.12,
        )
        base["structure"] = StructuralSignal(
            rms_amplitude_over_c=0.015, dominant_frequency_hz=8.0,
            natural_frequency_hz=25.0, mode_number=1, camber=0.03,
        )

    base["ground_truth_regime"] = regime
    return FeatureRecord(**base)


class TestDescribe:
    def test_deterministic_output(self):
        """describe should produce identical output for the same input."""
        fr = _make_fr(FlowRegime.ATTACHED)
        d1 = describe(fr)
        d2 = describe(fr)
        assert d1 == d2

    def test_contains_regime_keyword(self):
        """Output should contain the regime name."""
        fr = _make_fr(FlowRegime.VIV_LOCKIN)
        text = describe(fr)
        assert "VIV lock-in" in text

    def test_contains_lockin_status(self):
        """Output should contain the lock-in status."""
        fr = _make_fr(FlowRegime.VIV_LOCKIN)
        text = describe(fr)
        assert "locked" in text.lower() or "unlocked" in text.lower()

    def test_contains_frequency_values(self):
        """Output should contain specific frequency values."""
        fr = _make_fr(FlowRegime.ATTACHED)
        text = describe(fr)
        assert "Hz" in text
        assert "St =" in text

    def test_contains_amplitude_trend(self):
        """Output should contain the amplitude trend word."""
        fr = _make_fr(FlowRegime.ATTACHED)
        text = describe(fr)
        assert "Amplitude trend" in text

    def test_contains_mechanism(self):
        """Output should contain a mechanism note."""
        fr = _make_fr(FlowRegime.VIV_LOCKIN)
        text = describe(fr)
        assert "vortex" in text.lower() or "synchron" in text.lower()

    def test_contains_mode_shape(self):
        """Output should mention the mode shape."""
        fr = _make_fr(FlowRegime.ATTACHED)
        text = describe(fr)
        assert "mode" in text.lower()
        assert "structural mode" in text.lower()

    def test_paraphrase_fallback(self):
        """Without API key, paraphrase should return the same template text."""
        fr = _make_fr(FlowRegime.SEPARATED)
        text = describe(fr, paraphrase_with_api=True)
        assert "separated" in text.lower()


class TestClassifyRegime:
    def test_attached_flow(self):
        """Attached flow with low amplitude should classify as ATTACHED."""
        fr = _make_fr(FlowRegime.ATTACHED)
        fr.ground_truth_regime = None  # Remove ground truth
        regime = classify_regime(fr)
        assert regime == FlowRegime.ATTACHED

    def test_viv_lockin(self):
        """VIV lock-in with large amplitude and locked frequencies."""
        fr = _make_fr(FlowRegime.VIV_LOCKIN)
        fr.ground_truth_regime = None
        regime = classify_regime(fr)
        assert regime == FlowRegime.VIV_LOCKIN

    def test_galloping(self):
        """Galloping with large amplitude and f_n << f_shed."""
        fr = _make_fr(FlowRegime.GALLOPING)
        fr.ground_truth_regime = None
        regime = classify_regime(fr)
        assert regime == FlowRegime.GALLOPING

    def test_stall_flutter(self):
        """Stall flutter with high AoA and large amplitude."""
        fr = _make_fr(FlowRegime.STALL_FLUTTER)
        fr.ground_truth_regime = None
        regime = classify_regime(fr)
        assert regime == FlowRegime.STALL_FLUTTER

    def test_separated_flow(self):
        """Separated flow with wide wake and low amplitude."""
        fr = _make_fr(FlowRegime.SEPARATED)
        fr.ground_truth_regime = None
        regime = classify_regime(fr)
        assert regime == FlowRegime.SEPARATED


class TestAmplitudeTrend:
    def test_negligible_for_small_amplitude(self):
        """Very small amplitudes should give NEGLIGIBLE."""
        fr = _make_fr(FlowRegime.ATTACHED)
        trend = amplitude_trend(fr, FlowRegime.ATTACHED)
        assert trend == AmplitudeTrend.NEGLIGIBLE

    def test_growing_for_galloping(self):
        """Galloping regimes should give GROWING."""
        fr = _make_fr(FlowRegime.GALLOPING)
        trend = amplitude_trend(fr, FlowRegime.GALLOPING)
        assert trend == AmplitudeTrend.GROWING

    def test_saturated_for_viv(self):
        """VIV lock-in regimes should give SATURATED."""
        fr = _make_fr(FlowRegime.VIV_LOCKIN)
        trend = amplitude_trend(fr, FlowRegime.VIV_LOCKIN)
        assert trend == AmplitudeTrend.SATURATED

    def test_saturated_for_stall_flutter(self):
        """Stall flutter regimes should give SATURATED."""
        fr = _make_fr(FlowRegime.STALL_FLUTTER)
        trend = amplitude_trend(fr, FlowRegime.STALL_FLUTTER)
        assert trend == AmplitudeTrend.SATURATED

    def test_decaying_for_separated(self):
        """Separated flow should give DECAYING."""
        fr = _make_fr(FlowRegime.SEPARATED)
        trend = amplitude_trend(fr, FlowRegime.SEPARATED)
        assert trend == AmplitudeTrend.DECAYING


class TestProbabilisticClassifier:
    def test_confidence_range(self):
        fr = _make_fr(FlowRegime.ATTACHED)
        regime, conf, weights = classify_regime_confidence(fr)
        assert 0.0 <= conf <= 1.0
        assert isinstance(weights, dict)

    def test_weights_normalized(self):
        fr = _make_fr(FlowRegime.ATTACHED)
        _, _, weights = classify_regime_confidence(fr)
        if weights:
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6

    def test_galloping_classification(self):
        fr = _make_fr(FlowRegime.GALLOPING)
        fr.ground_truth_regime = None
        regime, conf, _ = classify_regime_confidence(fr)
        assert regime == FlowRegime.GALLOPING
        assert conf > 0.7

    def test_viv_lockin_high_confidence(self):
        from Data.schema import AerodynamicCoherence
        fr = _make_fr(FlowRegime.VIV_LOCKIN)
        fr.ground_truth_regime = None
        fr.piv.coherence = AerodynamicCoherence(
            cl_coherence_peak=0.95, cd_coherence_peak=0.3, phase_angle_deg=90.0
        )
        regime, conf, _ = classify_regime_confidence(fr)
        assert regime == FlowRegime.VIV_LOCKIN
        assert conf > 0.8

    def test_stall_flutter_classification(self):
        fr = _make_fr(FlowRegime.STALL_FLUTTER)
        fr.ground_truth_regime = None
        regime, _, _ = classify_regime_confidence(fr)
        assert regime == FlowRegime.STALL_FLUTTER


class TestEnhancedDescription:
    def test_description_includes_confidence(self):
        fr = _make_fr(FlowRegime.ATTACHED)
        text = describe(fr)
        assert "confidence:" in text

    def test_description_includes_damping(self):
        from Data.schema import ModalDamping
        fr = _make_fr(FlowRegime.ATTACHED)
        fr.structure.damping = ModalDamping(damping_ratio=0.04, q_factor=12.5)
        text = describe(fr)
        assert "ζ = 0.04" in text
        assert "Q = 12.5" in text

    def test_description_includes_coherence(self):
        from Data.schema import AerodynamicCoherence
        fr = _make_fr(FlowRegime.VIV_LOCKIN)
        fr.piv.coherence = AerodynamicCoherence(
            cl_coherence_peak=0.85, cd_coherence_peak=0.2, phase_angle_deg=-30.0
        )
        text = describe(fr)
        assert "coherence" in text

    def test_description_includes_moments(self):
        from Data.schema import StructuralSpectrum
        fr = _make_fr(FlowRegime.ATTACHED)
        fr.structure.spectrum = StructuralSpectrum(
            spectral_bandwidth_hz=0.5,
            spectral_kurtosis=2.0,
            skewness=0.1,
            kurtosis=3.2,
        )
        text = describe(fr)
        assert "skewness" in text
        assert "kurtosis" in text
