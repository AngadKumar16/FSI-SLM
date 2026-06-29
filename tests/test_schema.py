"""Unit tests for data.schema module.

Tests cover:
- Valid FeatureRecord construction
- Invalid ranges / values trigger validation errors
- range_warnings triggers correctly
- lock_in_status logic
- to_prompt_block format
- Enum values
"""

from __future__ import annotations

import pytest
from data.schema import (
    AmplitudeTrend,
    FeatureRecord,
    FlowConditions,
    FlowRegime,
    ForceCoefficients,
    LockInStatus,
    PIVDerived,
    StructuralSignal,
)


def _make_minimal_fr(
    regime: FlowRegime = FlowRegime.ATTACHED,
) -> FeatureRecord:
    """Build a minimal valid FeatureRecord."""
    return FeatureRecord(
        experiment_id="test-00000",
        conditions=FlowConditions(
            reynolds_number=50000,
            angle_of_attack_deg=2.0,
            freestream_velocity_m_s=1.0,
            reduced_velocity=6.0,
            chord_m=0.1,
        ),
        piv=PIVDerived(
            shedding_frequency_hz=10.0,
            strouhal_number=0.2,
            wake_width_over_c=0.2,
            separation_location_x_over_c=0.9,
            pod_mode_amplitudes=[0.7, 0.2, 0.05, 0.03],
            turbulent_kinetic_energy=0.01,
        ),
        structure=StructuralSignal(
            rms_amplitude_over_c=0.005,
            dominant_frequency_hz=8.0,
            natural_frequency_hz=10.0,
            mode_number=1,
            camber=0.02,
        ),
        forces=ForceCoefficients(
            cl_mean=0.5,
            cl_rms=0.01,
            cl_dominant_frequency_hz=10.0,
            cd_mean=0.05,
            cd_rms=0.005,
            cd_dominant_frequency_hz=20.0,
            cm_mean=-0.05,
            cm_rms=0.003,
            cm_dominant_frequency_hz=10.0,
        ),
        ground_truth_regime=regime,
    )


class TestFeatureRecord:
    def test_valid_construction(self):
        """A valid FeatureRecord should construct without error."""
        fr = _make_minimal_fr()
        assert fr.experiment_id == "test-00000"
        assert fr.ground_truth_regime == FlowRegime.ATTACHED

    def test_invalid_range_reynolds_number(self):
        """Reynolds number outside valid range should still construct (no Pydantic
        validation error) because ranges are advisory, but should appear in warnings."""
        fr = _make_minimal_fr()
        fr.conditions.reynolds_number = 1e10  # way out of range
        warns = fr.range_warnings()
        assert any("reynolds_number" in w for w in warns)

    def test_invalid_strouhal_number_triggers_warning(self):
        """Strouhal number > 1 should trigger a range warning."""
        fr = _make_minimal_fr()
        fr.piv.strouhal_number = 1.5
        warns = fr.range_warnings()
        assert any("strouhal_number" in w for w in warns)

    def test_range_warnings_empty_for_valid_record(self):
        """A record with all values in range should have no warnings."""
        fr = _make_minimal_fr()
        warns = fr.range_warnings()
        assert len(warns) == 0

    def test_lock_in_status_locked(self):
        """Lock-in should be LOCKED when shedding freq ≈ natural freq."""
        fr = _make_minimal_fr()
        fr.piv.shedding_frequency_hz = 10.0
        fr.structure.natural_frequency_hz = 10.0
        assert fr.lock_in_status() == LockInStatus.LOCKED

    def test_lock_in_status_near_lockin(self):
        """Lock-in should be NEAR_LOCKIN within 2x tolerance."""
        fr = _make_minimal_fr()
        fr.piv.shedding_frequency_hz = 10.0
        fr.structure.natural_frequency_hz = 9.0  # rel diff = 0.111 -> near_lockin
        assert fr.lock_in_status(tol=0.10) == LockInStatus.NEAR_LOCKIN

    def test_lock_in_status_unlocked(self):
        """Lock-in should be UNLOCKED when frequencies are far apart."""
        fr = _make_minimal_fr()
        fr.piv.shedding_frequency_hz = 10.0
        fr.structure.natural_frequency_hz = 50.0  # rel diff = 0.8
        assert fr.lock_in_status() == LockInStatus.UNLOCKED

    def test_lock_in_status_zero_natural_freq(self):
        """If natural frequency is <= 0, lock-in should be UNLOCKED."""
        fr = _make_minimal_fr()
        fr.structure.natural_frequency_hz = 0.0
        assert fr.lock_in_status() == LockInStatus.UNLOCKED

    def test_to_prompt_block_contains_experiment_id(self):
        """to_prompt_block should contain the experiment id."""
        fr = _make_minimal_fr()
        block = fr.to_prompt_block()
        assert "test-00000" in block

    def test_to_prompt_block_headers(self):
        """to_prompt_block should contain all section headers."""
        fr = _make_minimal_fr()
        block = fr.to_prompt_block()
        assert "[conditions]" in block
        assert "[piv_derived]" in block
        assert "[structure]" in block
        assert "[forces]" in block

    def test_to_prompt_block_no_ground_truth(self):
        """to_prompt_block should NOT contain the ground truth regime."""
        fr = _make_minimal_fr()
        block = fr.to_prompt_block()
        assert "ground_truth" not in block
        assert "attached" not in block.lower()  # regime keyword not in prompt block


class TestEnums:
    def test_flow_regime_values(self):
        assert FlowRegime.ATTACHED.value == "attached"
        assert FlowRegime.SEPARATED.value == "separated"
        assert FlowRegime.VIV_LOCKIN.value == "viv_lockin"
        assert FlowRegime.GALLOPING.value == "galloping"
        assert FlowRegime.STALL_FLUTTER.value == "stall_flutter"

    def test_lock_in_status_values(self):
        assert LockInStatus.LOCKED.value == "locked"
        assert LockInStatus.NEAR_LOCKIN.value == "near_lockin"
        assert LockInStatus.UNLOCKED.value == "unlocked"

    def test_amplitude_trend_values(self):
        assert AmplitudeTrend.GROWING.value == "growing"
        assert AmplitudeTrend.SATURATED.value == "saturated"
        assert AmplitudeTrend.DECAYING.value == "decaying"
        assert AmplitudeTrend.NEGLIGIBLE.value == "negligible"