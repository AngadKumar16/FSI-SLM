"""Signal analysis for the URANS backend, checked on signals of known content.

None of this needs OpenFOAM. The spectral estimator and the stationarity check
are the parts that decide whether a six-hour transient run produced a number or
produced noise, so they are tested against synthetic records whose answers are
known exactly -- including the two ways they can be fooled: a record that has
not settled, and a record with no organised tone at all.

The case writer is checked structurally: the dictionaries have to contain the
entries the physics depends on (second-order time scheme, averaging that starts
after the transient, force sampling every step), because getting any of those
wrong produces a run that completes and is silently worthless.
"""
from __future__ import annotations

import numpy as np
import pytest

from sensor.cfd.unsteady import (
    UnsteadyParams,
    analyse_case,
    shedding_frequency,
    statistics_converged,
    strouhal_projected,
    write_unsteady_case,
)

ST_TRUE = 0.22  # cycles per convective time


def _record(seed=0, t_end=100.0, amp=0.3, drift=0.0, noise=0.02, tone=True):
    """A force history with an adaptive time step, like the solver writes."""
    rng = np.random.default_rng(seed)
    dt = rng.uniform(0.004, 0.012, 20000)
    t = np.cumsum(dt)
    t = t / t[-1] * t_end
    cl = 1.0 + noise * rng.normal(size=t.size) + drift * t
    if tone:
        cl = cl + amp * np.sin(2 * np.pi * ST_TRUE * t)
    return t, cl


def test_shedding_frequency_recovers_a_known_tone():
    t, cl = _record()
    out = shedding_frequency(t, cl, t_start=20.0)
    assert out["f_shed"] == pytest.approx(ST_TRUE, rel=0.02)
    assert out["organised"]


def test_nonuniform_sampling_does_not_move_the_peak():
    """The solver's time step is adaptive; an FFT that ignores that is wrong.

    The two records below contain the same tone, one sampled uniformly and one
    with a 3x spread in step size. They have to return the same frequency.
    """
    t, cl = _record(seed=1)
    tu = np.linspace(t[0], t[-1], t.size)
    clu = 1.0 + 0.3 * np.sin(2 * np.pi * ST_TRUE * tu)
    a = shedding_frequency(t, cl, 20.0)["f_shed"]
    b = shedding_frequency(tu, clu, 20.0)["f_shed"]
    assert a == pytest.approx(b, rel=0.02)


def test_broadband_noise_is_not_reported_as_shedding():
    t, cl = _record(seed=2, tone=False, noise=0.05)
    out = shedding_frequency(t, cl, 20.0)
    assert not out["organised"]
    assert "background" in out["reason"]


def test_drifting_record_is_not_statistically_steady():
    t, cl = _record(seed=3, drift=0.004)
    out = statistics_converged(t, cl, 20.0, f_shed=ST_TRUE)
    assert not out["statistically_steady"]
    assert "drifts" in out["reason"]


def test_settled_record_is_accepted():
    t, cl = _record(seed=4)
    out = statistics_converged(t, cl, 20.0, f_shed=ST_TRUE)
    assert out["statistically_steady"], out["reason"]
    assert out["n_cycles"] > 6


def test_short_window_is_rejected_even_if_flat():
    """Few cycles means the mean depends on the starting phase, flat or not."""
    t, cl = _record(seed=5, t_end=30.0)
    out = statistics_converged(t, cl, 20.0, f_shed=ST_TRUE, min_cycles=6.0)
    assert not out["statistically_steady"]
    assert "cycles" in out["reason"]


def test_strouhal_projected_uses_the_fage_johansen_height():
    # d/c = sin(16 deg) + 0.12 = 0.3957
    assert strouhal_projected(0.5, 16.0, 0.12) == pytest.approx(0.5 * 0.3957, rel=1e-3)
    # A symmetric section at zero incidence projects only its thickness.
    assert strouhal_projected(1.0, 0.0, 0.09) == pytest.approx(0.09, rel=1e-6)


def test_case_dictionaries_carry_the_entries_the_physics_needs(tmp_path):
    up = UnsteadyParams(t_end=60.0, t_start_average=15.0)
    d = write_unsteady_case(tmp_path / "case", up)

    control = (d / "system/controlDict").read_text()
    schemes = (d / "system/fvSchemes").read_text()
    solution = (d / "system/fvSolution").read_text()

    assert "application pimpleFoam" in control
    assert "adjustTimeStep yes" in control
    # Force history every step: the lift signal is the measurement, and a
    # decimated record aliases the shedding peak.
    assert "writeControl timeStep; writeInterval 1" in control
    # Averaging must start after the transient, not at t = 0.
    assert "timeStart 15" in control
    assert "fieldAverage" in control
    # Euler damps the shedding amplitude at the steps an adaptive run takes.
    assert "ddtSchemes { default backward; }" in schemes
    # No under-relaxation, or the time derivative is quietly damped.
    assert 'relaxationFactors { equations { ".*" 1.0; } }' in solution
    assert "nOuterCorrectors 2" in solution


def test_analyse_case_reads_a_written_force_history(tmp_path):
    """End-to-end over the file format OpenFOAM actually writes."""
    case = tmp_path / "case"
    d = case / "postProcessing/forceCoeffs/0"
    d.mkdir(parents=True)
    t, cl = _record(seed=6)
    cd = 0.3 + 0.05 * np.sin(4 * np.pi * ST_TRUE * t)
    cm = -0.02 + 0.01 * np.sin(2 * np.pi * ST_TRUE * t)
    lines = ["# Force coefficients", "# Time Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) CmPitch"]
    for i in range(t.size):
        lines.append(
            f"{t[i]:.6g} {cd[i]:.6g} 0 0 {cl[i]:.6g} 0 0 {cm[i]:.6g}"
        )
    (d / "coefficient.dat").write_text("\n".join(lines) + "\n")

    out = analyse_case(case, UnsteadyParams(t_end=100.0, t_start_average=20.0),
                       alpha_deg=16.0, thickness=0.12)
    assert out["cl_mean"] == pytest.approx(1.0, abs=0.02)
    assert out["cl_rms"] == pytest.approx(0.3 / np.sqrt(2), rel=0.1)
    assert out["strouhal_chord"] == pytest.approx(ST_TRUE, rel=0.02)
    assert out["strouhal_projected"] == pytest.approx(
        ST_TRUE * (np.sin(np.deg2rad(16.0)) + 0.12), rel=0.03
    )
    assert out["statistically_steady"], out["stationarity_note"]


def test_bridge_prefers_a_measured_shedding_frequency():
    """A URANS row overrides the Fage-Johansen correlation, and says so."""
    from sensor.bridge import TwinConfig, _measured_shedding

    cfg = TwinConfig()
    case = {"unsteady": {"strouhal_chord": 0.25, "organised_shedding": True,
                         "statistically_steady": True}}
    f, note = _measured_shedding(case, cfg)
    assert f == pytest.approx(0.25 * cfg.u_inf_m_s / cfg.chord_m)
    assert "measured" in note

    # No organised tone -> fall back rather than dress up a noise bin.
    case["unsteady"]["organised_shedding"] = False
    assert _measured_shedding(case, cfg) is None

    # A panel or steady-RANS case has nothing to offer.
    assert _measured_shedding({}, cfg) is None
