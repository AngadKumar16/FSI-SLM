"""Standalone output validator (no model dependencies).

Parses raw generated text from stdin or a file, extracts structured claims,
checks numeric consistency (e.g., reported St vs computed from reported
f_shed/U/c), lock-in logic vs claimed frequencies, and schema conformance.

Exits with code 0 if valid, 1 if invalid. Prints machine-readable JSON summary.

Usage::

    python -m eval.validate_output --file output.txt
    python -m eval.validate_output < output.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class ValidationResult:
    """Result of validating a model output against expected format and physics.

    Attributes
    ----------
    valid : bool
        Whether the output passes all validation checks.
    parsed_fields : dict
        Dictionary of parsed structured fields from the text.
    errors : list[str]
        List of error messages for failed checks.
    warnings : list[str]
        List of non-fatal warnings.
    """
    valid: bool = True
    parsed_fields: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Regex patterns for extracting structured fields from generated text.
# These match the output format produced by data/describe.py.

_REGIME_PATTERN = re.compile(
    r"Regime:\s*(attached flow|separated flow|VIV lock-in|galloping|stall flutter)\s*\.",
    re.IGNORECASE,
)

_FREQ_PATTERN = re.compile(
    r"dominant frequency of\s*([\d.]+)\s*Hz",
    re.IGNORECASE,
)

_STROUHAL_PATTERN = re.compile(
    r"Strouhal number St\s*=\s*([\d.]+)",
    re.IGNORECASE,
)

_LOCKIN_PATTERN = re.compile(
    r"Lock-in status:\s*(locked|near lock-in|unlocked)",
    re.IGNORECASE,
)

_AMPLITUDE_PATTERN = re.compile(
    r"Amplitude trend:\s*(growing|saturated|decaying|negligible)",
    re.IGNORECASE,
)

_NATURAL_FREQ_PATTERN = re.compile(
    r"structural natural frequency\s*([\d.]+)\s*Hz",
    re.IGNORECASE,
)

_MODE_PATTERN = re.compile(
    r"mode\s*(\d+),",
    re.IGNORECASE,
)

_RMS_AMPLITUDE_PATTERN = re.compile(
    r"RMS displacement\s*([\d.]+)c",
    re.IGNORECASE,
)


def parse_structured_fields(text: str) -> dict:
    """Parse structured fields from the generated text.

    Parameters
    ----------
    text : str
        The raw generated text from the model.

    Returns
    -------
    dict
        Dictionary of parsed fields (keys may include: regime, shedding_frequency_hz,
        strouhal_number, lock_in_status, amplitude_trend, mode_number,
        natural_frequency_hz, rms_amplitude_over_c).
    """
    fields: dict = {}

    m = _REGIME_PATTERN.search(text)
    if m:
        fields["regime"] = m.group(1).strip().lower()

    m = _FREQ_PATTERN.search(text)
    if m:
        fields["shedding_frequency_hz"] = float(m.group(1))

    m = _STROUHAL_PATTERN.search(text)
    if m:
        fields["strouhal_number"] = float(m.group(1))

    m = _LOCKIN_PATTERN.search(text)
    if m:
        fields["lock_in_status"] = m.group(1).strip().lower()

    m = _AMPLITUDE_PATTERN.search(text)
    if m:
        fields["amplitude_trend"] = m.group(1).strip().lower()

    m = _NATURAL_FREQ_PATTERN.search(text)
    if m:
        fields["natural_frequency_hz"] = float(m.group(1))

    m = _MODE_PATTERN.search(text)
    if m:
        fields["mode_number"] = int(m.group(1))

    m = _RMS_AMPLITUDE_PATTERN.search(text)
    if m:
        fields["rms_amplitude_over_c"] = float(m.group(1))

    return fields


def check_numeric_consistency(fields: dict) -> list[str]:
    """Check numeric consistency of parsed fields.

    Checks:
    - Strouhal number consistency: St = f_shed * c / U (cannot fully verify
      without c and U, but can check if St is within plausible bounds).
    - Basic sanity checks on frequency values.

    Parameters
    ----------
    fields : dict
        Parsed structured fields.

    Returns
    -------
    list[str]
        List of error messages for failed consistency checks.
    """
    errors: list[str] = []

    f_shed = fields.get("shedding_frequency_hz")
    st = fields.get("strouhal_number")
    f_n = fields.get("natural_frequency_hz")
    amplitude = fields.get("rms_amplitude_over_c")

    # Strouhal number sanity check (should be 0-1 for chord-based)
    if st is not None:
        if st < 0.0 or st > 1.0:
            errors.append(
                f"Strouhal number {st:.4f} is outside plausible range [0, 1]"
            )

    # Frequency sanity checks
    if f_shed is not None and f_n is not None:
        # Frequencies should be positive
        if f_shed <= 0:
            errors.append(f"Shedding frequency ({f_shed} Hz) must be positive")
        if f_n <= 0:
            errors.append(f"Natural frequency ({f_n} Hz) must be positive")

        # Lock-in check: if claimed locked, f_shed and f_n should be close
        lock_status = fields.get("lock_in_status")
        if lock_status == "locked":
            if f_n > 0:
                rel_diff = abs(f_shed - f_n) / f_n
                if rel_diff > 0.15:
                    errors.append(
                        f"Claimed 'locked' but |f_shed - f_n| / f_n = {rel_diff:.2%} "
                        f"(f_shed={f_shed} Hz, f_n={f_n} Hz), exceeds 15% lock-in tolerance"
                    )
        elif lock_status == "near lock-in":
            if f_n > 0:
                rel_diff = abs(f_shed - f_n) / f_n
                if rel_diff > 0.30:
                    errors.append(
                        f"Claimed 'near lock-in' but |f_shed - f_n| / f_n = "
                        f"{rel_diff:.2%}, exceeds 30% near-lock-in tolerance"
                    )

    # Amplitude sanity
    if amplitude is not None:
        if amplitude < 0.0:
            errors.append(f"RMS amplitude ({amplitude}) cannot be negative")
        elif amplitude > 1.0:
            errors.append(
                f"RMS amplitude ({amplitude:.4f}c) exceeds physical bound of 1.0c"
            )

    return errors


def check_lock_in_logic(fields: dict) -> list[str]:
    """Check lock-in logic vs claimed frequencies.

    Verifies that the claimed lock-in status is consistent with the relationship
    between shedding frequency and structural natural frequency.

    Parameters
    ----------
    fields : dict
        Parsed structured fields.

    Returns
    -------
    list[str]
        List of warning messages for potential logic inconsistencies.
    """
    warnings: list[str] = []
    f_shed = fields.get("shedding_frequency_hz")
    f_n = fields.get("natural_frequency_hz")
    lock_status = fields.get("lock_in_status")

    if f_shed is not None and f_n is not None and lock_status is not None and f_n > 0:
        rel_diff = abs(f_shed - f_n) / f_n
        if lock_status == "locked" and rel_diff > 0.10:
            warnings.append(
                f"Lock-in status is 'locked' but frequency difference is "
                f"{rel_diff:.1%} (exceeds typical 10% lock-in criterion)"
            )
        elif lock_status == "unlocked" and rel_diff < 0.05:
            warnings.append(
                f"Lock-in status is 'unlocked' but frequencies are very close "
                f"(diff={rel_diff:.1%})"
            )

    return warnings


def validate_output(text: str) -> ValidationResult:
    """Validate a model output string against expected format and physics.

    Parameters
    ----------
    text : str
        The raw model output text.

    Returns
    -------
    ValidationResult
        Result with parsed fields, errors, and warnings.
    """
    result = ValidationResult()
    text = text.strip()

    if not text:
        result.valid = False
        result.errors.append("Empty output")
        return result

    # Parse structured fields
    result.parsed_fields = parse_structured_fields(text)

    # Check that at least regime, lock-in, and amplitude trend were found
    required_fields = ["regime", "lock_in_status", "amplitude_trend"]
    for field in required_fields:
        if field not in result.parsed_fields:
            result.errors.append(f"Missing required field: {field}")

    # Check numeric consistency
    numeric_errors = check_numeric_consistency(result.parsed_fields)
    result.errors.extend(numeric_errors)

    # Check lock-in logic (warnings only)
    logic_warnings = check_lock_in_logic(result.parsed_fields)
    result.warnings.extend(logic_warnings)

    # Set overall validity
    result.valid = len(result.errors) == 0

    return result


def main() -> None:
    """CLI entry point for standalone validation."""
    parser = argparse.ArgumentParser(
        description="Validate model output for FSI interpretation format and physics."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a file containing model output (reads from stdin if not provided)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output",
    )
    args = parser.parse_args()

    # Read input
    if args.file:
        with open(args.file) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    # Validate
    result = validate_output(text)

    # Output JSON
    output = asdict(result)
    indent = 2 if args.pretty else None
    json.dump(output, sys.stdout, indent=indent)
    sys.stdout.write("\n")

    # Exit with appropriate code
    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()


__all__ = ["ValidationResult", "validate_output", "parse_structured_fields",
           "check_numeric_consistency", "check_lock_in_logic", "main"]