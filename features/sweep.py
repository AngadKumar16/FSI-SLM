"""What-if parameter sweeps over a single FeatureRecord.

Given one experiment, vary a single operating parameter across a range while
holding the rest of the record fixed (rescaling the physically coupled
quantities), and classify the regime at every point. This turns one measured
operating point into a predicted regime map -- e.g. "at what reduced velocity
does this rig enter lock-in?" -- using the transparent rule-based classifier,
so it runs instantly with no model load.

Physical couplings applied per swept parameter:

* ``reduced_velocity`` (U* = U / (f_n c), with f_n and c held fixed):
  freestream velocity scales with U*, the shedding frequency follows
  f_shed = St * U / c (St held fixed), and Re scales with U.
* ``angle_of_attack_deg``: varied directly; no other field is rescaled.
* ``natural_frequency_hz``: f_n varied; U* = U / (f_n c) recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from Data.describe import classify_regime_confidence
from Data.schema import FeatureRecord, FlowRegime

SWEEPABLE = ("reduced_velocity", "angle_of_attack_deg", "natural_frequency_hz")


@dataclass
class SweepPoint:
    """Classification outcome at one swept parameter value."""

    value: float
    regime: FlowRegime
    confidence: float
    shedding_frequency_hz: float
    lock_in: str


def _record_at(fr: FeatureRecord, param: str, value: float) -> FeatureRecord:
    """Return a copy of ``fr`` with ``param`` set to ``value`` and coupled
    quantities rescaled consistently."""
    rec = fr.model_copy(deep=True)
    c = rec.conditions
    p = rec.piv

    if param == "reduced_velocity":
        base_ustar = fr.conditions.reduced_velocity
        scale = value / base_ustar if base_ustar > 0 else 1.0
        c.reduced_velocity = value
        c.freestream_velocity_m_s = fr.conditions.freestream_velocity_m_s * scale
        c.reynolds_number = fr.conditions.reynolds_number * scale
        # St fixed => f_shed scales with U.
        p.shedding_frequency_hz = fr.piv.shedding_frequency_hz * scale
    elif param == "angle_of_attack_deg":
        c.angle_of_attack_deg = value
    elif param == "natural_frequency_hz":
        rec.structure.natural_frequency_hz = value
        if value > 0 and c.chord_m > 0:
            c.reduced_velocity = c.freestream_velocity_m_s / (value * c.chord_m)
    else:
        raise ValueError(f"Unsupported sweep parameter '{param}'. Choose from {SWEEPABLE}.")
    return rec


def sweep_parameter(
    fr: FeatureRecord,
    param: str = "reduced_velocity",
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    n: int = 15,
) -> List[SweepPoint]:
    """Classify the regime across ``n`` values of ``param`` in [lo, hi].

    Defaults span a physically sensible band around the record's own value:
    U* in [2, 12], AoA in [-4, 24] deg, f_n in [0.25x, 4x] of the current
    natural frequency.
    """
    if param not in SWEEPABLE:
        raise ValueError(f"Unsupported sweep parameter '{param}'. Choose from {SWEEPABLE}.")

    if lo is None or hi is None:
        if param == "reduced_velocity":
            lo, hi = 2.0, 12.0
        elif param == "angle_of_attack_deg":
            lo, hi = -4.0, 24.0
        else:
            base = fr.structure.natural_frequency_hz
            lo, hi = 0.25 * base, 4.0 * base

    points: List[SweepPoint] = []
    for value in np.linspace(lo, hi, n):
        rec = _record_at(fr, param, float(value))
        regime, confidence, _ = classify_regime_confidence(rec)
        points.append(
            SweepPoint(
                value=float(value),
                regime=regime,
                confidence=confidence,
                shedding_frequency_hz=rec.piv.shedding_frequency_hz,
                lock_in=rec.lock_in_status().value,
            )
        )
    return points


def find_transitions(points: List[SweepPoint]) -> List[tuple]:
    """Return (value, from_regime, to_regime) at each regime boundary."""
    out = []
    for prev, cur in zip(points, points[1:]):
        if cur.regime != prev.regime:
            out.append((cur.value, prev.regime, cur.regime))
    return out


__all__ = ["SweepPoint", "sweep_parameter", "find_transitions", "SWEEPABLE"]
