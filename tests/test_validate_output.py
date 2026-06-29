"""Unit tests for eval.validate_output module.

Tests cover:
- validate_output accepts a correctly formatted string
- validate_output rejects malformed output
- parse_structured_fields extracts numeric fields correctly
- Lock-in logic detection
- Numeric consistency checks
"""

from __future__ import annotations

import json
import pytest

from eval.validate_output import (
    ValidationResult,
    check_lock_in_logic,
    check_numeric_consistency,
    parse_structured_fields,
    validate_output,
)


# A valid output string matching the describe() format
_VALID_OUTPUT = (
    "Regime: VIV lock-in. "
    "The wake sheds at a dominant frequency of 10.00 Hz, "
    "giving a Strouhal number St = 0.200. "
    "Lock-in status: locked "
    "(shedding 10.00 Hz vs. structural natural frequency 10.00 Hz). "
    "Amplitude trend: saturated "
    "(RMS displacement 0.350c). "
    "Mode shape: first structural mode (mode 1, bending/heave) with mild camber (0.05c). "
    "Mechanism: Vortex shedding has synchronized with the structural natural frequency, "
    "driving a saturated vortex-induced limit cycle."
)

_MALFORMED_OUTPUT = "This is not a valid FSI interpretation."


class TestParseStructuredFields:
    def test_parses_regime(self):
        """Should extract the regime name."""
        fields = parse_structured_fields(_VALID_OUTPUT)
        assert fields.get("regime") == "viv lock-in"

    def test_parses_frequency(self):
        """Should extract the shedding frequency."""
        fields = parse_structured_fields(_VALID_OUTPUT)
        assert fields.get("shedding_frequency_hz") == 10.0

    def test_parses_strouhal(self):
        """Should extract the Strouhal number."""
        fields = parse_structured_fields(_VALID_OUTPUT)
        assert fields.get("strouhal_number") == 0.2

    def test_parses_lockin_status(self):
        """Should extract the lock-in status."""
        fields = parse_structured_fields(_VALID_OUTPUT)
        assert fields.get("lock_in_status") == "locked"

    def test_parses_amplitude_trend(self):
        """Should extract the amplitude trend."""
        fields = parse_structured_fields(_VALID_OUTPUT)
        assert fields.get("amplitude_trend") == "saturated"

    def test_parses_natural_frequency(self):
        """Should extract the structural natural frequency."""
        fields = parse_structured_fields(_VALID_OUTPUT)
        assert fields.get("natural_frequency_hz") == 10.0

    def test_parses_mode_number(self):
        """Should extract the mode number."""
        fields = parse_structured_fields(_VALID_OUTPUT)
        assert fields.get("mode_number") == 1

    def test_parses_rms_amplitude(self):
        """Should extract the RMS amplitude."""
        fields = parse_structured_fields(_VALID_OUTPUT)
        assert fields.get("rms_amplitude_over_c") == 0.35

    def test_empty_text_returns_empty_dict(self):
        """Empty text should return an empty dict."""
        fields = parse_structured_fields("")
        assert fields == {}

    def test_malformed_text_returns_partial(self):
        """Malformed text should return whatever fields can be parsed."""
        fields = parse_structured_fields(_MALFORMED_OUTPUT)
        assert fields == {}  # no fields match


class TestCheckNumericConsistency:
    def test_valid_fields_no_errors(self):
        """Valid fields should produce no errors."""
        fields = {
            "shedding_frequency_hz": 10.0,
            "strouhal_number": 0.2,
            "natural_frequency_hz": 10.0,
            "lock_in_status": "locked",
            "rms_amplitude_over_c": 0.35,
        }
        errors = check_numeric_consistency(fields)
        assert len(errors) == 0

    def test_strouhal_out_of_range(self):
        """Strouhal number > 1 should produce an error."""
        fields = {"strouhal_number": 1.5}
        errors = check_numeric_consistency(fields)
        assert len(errors) > 0
        assert any("Strouhal" in e for e in errors)

    def test_locked_with_distant_frequencies(self):
        """Claiming 'locked' with distant frequencies should produce an error."""
        fields = {
            "shedding_frequency_hz": 10.0,
            "natural_frequency_hz": 50.0,
            "lock_in_status": "locked",
        }
        errors = check_numeric_consistency(fields)
        assert len(errors) > 0
        assert any("locked" in e.lower() for e in errors)

    def test_negative_amplitude(self):
        """Negative amplitude should produce an error."""
        fields = {"rms_amplitude_over_c": -0.1}
        errors = check_numeric_consistency(fields)
        assert len(errors) > 0
        assert any("amplitude" in e.lower() for e in errors)

    def test_amplitude_exceeds_bound(self):
        """Amplitude > 1 should produce an error."""
        fields = {"rms_amplitude_over_c": 1.5}
        errors = check_numeric_consistency(fields)
        assert len(errors) > 0
        assert any("amplitude" in e.lower() for e in errors)


class TestCheckLockInLogic:
    def test_locked_with_close_frequencies_no_warning(self):
        """Locked with close frequencies should produce no warnings."""
        fields = {
            "shedding_frequency_hz": 10.0,
            "natural_frequency_hz": 10.5,
            "lock_in_status": "locked",
        }
        warnings = check_lock_in_logic(fields)
        assert len(warnings) == 0

    def test_locked_with_distant_frequencies_warning(self):
        """Locked with distant frequencies should produce a warning."""
        fields = {
            "shedding_frequency_hz": 10.0,
            "natural_frequency_hz": 20.0,
            "lock_in_status": "locked",
        }
        warnings = check_lock_in_logic(fields)
        assert len(warnings) > 0
        assert any("locked" in w.lower() for w in warnings)

    def test_unlocked_with_close_frequencies_warning(self):
        """Unlocked with very close frequencies should produce a warning."""
        fields = {
            "shedding_frequency_hz": 10.0,
            "natural_frequency_hz": 10.1,
            "lock_in_status": "unlocked",
        }
        warnings = check_lock_in_logic(fields)
        assert len(warnings) > 0
        assert any("unlocked" in w.lower() for w in warnings)


class TestValidateOutput:
    def test_valid_output_passes(self):
        """A correctly formatted output should pass validation."""
        result = validate_output(_VALID_OUTPUT)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_output_has_parsed_fields(self):
        """A valid output should have parsed fields."""
        result = validate_output(_VALID_OUTPUT)
        assert len(result.parsed_fields) > 0
        assert result.parsed_fields.get("regime") == "viv lock-in"

    def test_malformed_output_fails(self):
        """Malformed output should fail validation."""
        result = validate_output(_MALFORMED_OUTPUT)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_empty_output_fails(self):
        """Empty output should fail validation."""
        result = validate_output("")
        assert result.valid is False
        assert "Empty" in result.errors[0]

    def test_missing_required_fields(self):
        """Output missing required fields should fail."""
        text = "Some text without regime or lock-in status."
        result = validate_output(text)
        assert result.valid is False
        assert any("regime" in e for e in result.errors)
        assert any("lock_in_status" in e for e in result.errors)

    def test_validation_result_serializable(self):
        """ValidationResult should be JSON-serializable."""
        result = validate_output(_VALID_OUTPUT)
        d = {
            "valid": result.valid,
            "parsed_fields": result.parsed_fields,
            "errors": result.errors,
            "warnings": result.warnings,
        }
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        assert "viv lock-in" in json_str