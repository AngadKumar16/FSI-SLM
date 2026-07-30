"""Physics acceptance tests for the panel solver.

These are not regression tests against saved numbers -- they check the solver
against results that were known before this repository existed: thin-airfoil
theory, d'Alembert's paradox, and the stagnation condition.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sensor.flow.panel import PanelSolver, resample_polygon
from sensor.geometry import close_trailing_edge, load_dat, rotate

FOIL_0009 = "Data/Airfoils/4/n0009sm.dat.txt"
FOIL_0012 = "Data/Airfoils/4/n0012.dat.txt"
FOIL_4412 = "Data/Airfoils/4/naca4412.dat.txt"


def _solve(path, alpha_deg, n=180):
    af = close_trailing_edge(load_dat(path))
    xy = resample_polygon(rotate(af, alpha_deg).xy, n)
    return PanelSolver(xy).solve(0.0)


def test_lift_slope_matches_thin_airfoil_theory():
    """dCl/dalpha should be 2*pi*(1 + 0.77 t/c) to within a couple of percent."""
    sol0 = _solve(FOIL_0012, 0.0)
    sol4 = _solve(FOIL_0012, 4.0)
    slope = (sol4.cl - sol0.cl) / math.radians(4.0)
    expected = 2 * math.pi * (1 + 0.77 * 0.12)
    assert abs(slope - expected) / expected < 0.03


def test_symmetric_section_has_zero_lift_at_zero_incidence():
    """NACA 0012's table is exactly symmetric, so its Cl(0) must be numerically
    zero. The "smoothed" 0009 table carries about 2.5e-4 c of residual camber
    near the trailing edge, which is real geometry, not solver error -- it shows
    up as a zero-lift angle of a few hundredths of a degree, and the test
    asserts that rather than pretending the section is perfect."""
    assert abs(_solve(FOIL_0012, 0.0).cl) < 1e-3

    cl0 = _solve(FOIL_0009, 0.0).cl
    alpha_l0_deg = -math.degrees(cl0 / (2 * math.pi * 1.07))
    assert abs(alpha_l0_deg) < 0.1


def test_blunt_trailing_edge_must_be_closed_first():
    """The raw NACA 0012 table has a finite TE gap; leaving it open breaks Kutta."""
    af_raw = load_dat(FOIL_0012)
    xy_raw = resample_polygon(af_raw.xy, 180)
    cl_open = PanelSolver(xy_raw).solve(0.0).cl
    cl_closed = _solve(FOIL_0012, 0.0).cl
    assert abs(cl_open) > 0.05          # the bug this guards against
    assert abs(cl_closed) < 1e-3


def test_dalembert_paradox():
    """Inviscid pressure drag must vanish; a large residual means bad panelling."""
    for alpha in (0.0, 4.0, 8.0, 12.0):
        sol = _solve(FOIL_0009, alpha)
        assert sol.dalembert_residual < 1e-2, alpha


def test_kutta_joukowski_agrees_with_pressure_integration():
    for alpha in (0.0, 5.0, 10.0):
        sol = _solve(FOIL_4412, alpha)
        assert abs(sol.cl - sol.cl_pressure) < 0.03 * max(abs(sol.cl), 0.1)


def test_stagnation_pressure_reaches_one():
    """Cp = 1 at the stagnation point -- and only there."""
    for alpha in (0.0, 6.0):
        sol = _solve(FOIL_0009, alpha)
        assert 0.97 < sol.cp.max() <= 1.0 + 1e-6
        assert sol.cp.max() <= 1.0 + 1e-6


def test_cambered_section_zero_lift_angle_is_negative():
    """NACA 4412's published alpha_L0 is about -4 degrees."""
    sol = _solve(FOIL_4412, 0.0)
    slope = 2 * math.pi * 1.1
    alpha_l0 = -math.degrees(sol.cl / slope)
    assert -6.0 < alpha_l0 < -2.5


def test_panel_count_convergence():
    """Cl should be converged to well under a percent by 180 panels."""
    cl = [_solve(FOIL_0009, 6.0, n).cl for n in (120, 180, 300)]
    assert abs(cl[1] - cl[2]) / abs(cl[2]) < 0.01
    assert abs(cl[0] - cl[2]) / abs(cl[2]) < 0.03


def test_resample_preserves_geometry():
    af = close_trailing_edge(load_dat(FOIL_0009))
    xy = resample_polygon(af.xy, 200)
    assert np.allclose(xy[0], xy[-1])
    assert abs(xy[:, 0].max() - af.xy[:, 0].max()) < 1e-3
    thickness = xy[:, 1].max() - xy[:, 1].min()
    assert abs(thickness - af.thickness) < 2e-3
