"""Calibration tests for the stall and drag model.

Reference values are the standard low-Reynolds NACA data (Abbott & von
Doenhoff; Sheldahl & Klimas). Tolerances are deliberately loose -- this is a
correlation-based model feeding a surrogate, not a RANS replacement -- but they
are tight enough to catch a sign error, a units slip, or a stall that never
happens.
"""
from __future__ import annotations

import math

import pytest

from sensor.flow.solve import foil_constants, solve_case
from sensor.flow.viscous import ViscousModel
from sensor.geometry import load_dat
from sensor.sdf import Grid

GRID = Grid(nx=64, ny=64)


def _polar(path, alphas, reynolds=5.0e5):
    af = load_dat(path)
    consts = foil_constants(af)
    return [solve_case(af, a, reynolds, GRID, constants=consts)["coeffs"]
            for a in alphas]


def test_separation_point_is_monotonic_in_incidence():
    m = ViscousModel(thickness=0.12, reynolds=5e5)
    f = [m.separation_point(a) for a in range(0, 26, 2)]
    assert all(b <= a + 1e-9 for a, b in zip(f, f[1:]))
    assert f[0] > 0.95      # attached at zero incidence
    assert f[-1] < 0.10     # fully separated in deep stall


def test_stall_angle_grows_with_thickness():
    thin = ViscousModel(0.06, 5e5).alpha_stall_deg
    thick = ViscousModel(0.18, 5e5).alpha_stall_deg
    assert thick > thin + 3.0


def test_naca0012_polar_matches_reference_data():
    alphas = [0, 4, 8, 12, 16, 20]
    polar = _polar("Data/Airfoils/4/n0012.dat.txt", alphas)
    cl = [p["cl"] for p in polar]
    cd = [p["cd"] for p in polar]

    assert abs(cl[0]) < 0.02                       # symmetric
    assert 0.75 < cl[2] < 1.05                     # Cl(8 deg) ~ 0.85
    assert 1.05 < max(cl) < 1.55                   # Cl_max ~ 1.2-1.35
    assert 0.008 < cd[0] < 0.016                   # Cd_min ~ 0.010
    assert cd[-1] > 8 * cd[0]                      # deep-stall drag rise
    assert cl[-1] < max(cl)                        # lift falls after stall


def test_drag_never_negative_and_rises_with_incidence():
    polar = _polar("Data/Airfoils/4/naca4412.dat.txt", [0, 4, 8, 12, 16])
    cd = [p["cd"] for p in polar]
    assert all(c > 0 for c in cd)
    assert cd == sorted(cd)


def test_cambered_section_lifts_at_zero_incidence():
    polar = _polar("Data/Airfoils/4/naca4412.dat.txt", [0.0])
    assert 0.35 < polar[0]["cl"] < 0.65            # published ~0.45


def test_reynolds_raises_the_stall_angle():
    low = ViscousModel(0.12, 1e5).alpha_stall_deg
    high = ViscousModel(0.12, 1e6).alpha_stall_deg
    assert high > low + 1.0


def test_negative_incidence_mirrors_positive_for_symmetric_section():
    plus = _polar("Data/Airfoils/4/n0012.dat.txt", [8.0])[0]
    minus = _polar("Data/Airfoils/4/n0012.dat.txt", [-8.0])[0]
    assert abs(plus["cl"] + minus["cl"]) < 0.02
    assert abs(plus["cd"] - minus["cd"]) < 0.005
