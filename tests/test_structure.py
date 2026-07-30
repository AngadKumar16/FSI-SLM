"""Tests for the chordwise beam and the spanwise modal model.

The beam solver is checked against closed-form Euler-Bernoulli results, which
is possible because a uniform-thickness section under a uniform load has an
exact answer.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sensor.geometry import load_dat
from sensor.structure.beam import (
    ALUMINIUM,
    STEEL,
    modal_properties,
    section_properties,
    solve_structure,
)


def _uniform_section(n=201, thickness=0.1):
    """A constant-thickness 'section' so the beam has an analytic solution."""
    xs = np.linspace(0.0, 1.0, n)
    # Deliberately *not* closed: appending the closing point would put a second
    # sample at x = 1 on the lower-surface branch and np.interp would pick up
    # the upper-surface value there, collapsing the trailing-edge thickness.
    return section_properties(
        np.vstack([
            np.column_stack([xs[::-1], np.full(n, thickness / 2)]),
            np.column_stack([xs, np.full(n, -thickness / 2)]),
        ])
    )


def test_cantilever_tip_deflection_matches_closed_form():
    """Clamp at x=0, uniform load q: w(L) = q L^4 / (8 EI)."""
    t = 0.1
    sec = _uniform_section(thickness=t)
    e_star = 1.0e4
    q_dyn = 0.5
    dcp = np.ones_like(sec.x)

    res = solve_structure(sec, dcp, e_star=e_star, spar_x=0.0, q_dyn=q_dyn)
    ei = e_star * t ** 3 / 12.0
    expected = (dcp[0] * q_dyn) * 1.0 ** 4 / (8.0 * ei)
    assert abs(res.w[-1] - expected) / expected < 0.02


def test_clamped_point_has_zero_deflection_and_slope():
    sec = _uniform_section()
    res = solve_structure(sec, np.ones_like(sec.x), spar_x=0.25)
    i = int(np.argmin(np.abs(sec.x - 0.25)))
    assert abs(res.w[i]) < 1e-12
    assert abs(res.slope[i]) < 1e-12


def test_deflection_scales_inversely_with_stiffness():
    sec = _uniform_section()
    dcp = np.ones_like(sec.x)
    soft = solve_structure(sec, dcp, e_star=1.0e4).w_max
    stiff = solve_structure(sec, dcp, e_star=2.0e4).w_max
    assert abs(soft / stiff - 2.0) < 0.02


def test_deflection_scales_linearly_with_load():
    sec = _uniform_section()
    one = solve_structure(sec, np.ones_like(sec.x)).w_max
    two = solve_structure(sec, 2 * np.ones_like(sec.x)).w_max
    assert abs(two / one - 2.0) < 1e-6


def test_thicker_section_deflects_less_as_cube_of_thickness():
    thin = solve_structure(_uniform_section(thickness=0.10),
                           np.ones(121), e_star=1e4).w_max
    thick = solve_structure(_uniform_section(thickness=0.20),
                            np.ones(121), e_star=1e4).w_max
    assert abs((thin / thick) / 8.0 - 1.0) < 0.05


def test_section_properties_of_a_real_foil():
    sec = section_properties(load_dat("Data/Airfoils/4/n0012.dat.txt").xy)
    assert 0.11 < sec.max_thickness < 0.13
    assert 0.20 < sec.max_thickness_x < 0.40      # 4-digit series: t_max at 0.30 c
    assert 0.06 < sec.area < 0.10                 # ~0.685 * t * c
    assert sec.i_zz > 0


def test_cantilever_natural_frequency_matches_hand_calculation():
    """f1 = (1.8751^2 / 2 pi L^2) sqrt(EI / m'), checked independently."""
    sec = section_properties(load_dat("Data/Airfoils/4/n0012.dat.txt").xy)
    m = modal_properties(sec, chord_m=0.1, span_m=0.3, material=ALUMINIUM,
                         fluid_density=0.0)
    area = sec.area * 0.1 ** 2
    i_zz = sec.i_zz * 0.1 ** 4
    ei = ALUMINIUM.flexural_modulus * i_zz
    mass = ALUMINIUM.density_kg_m3 * area
    expected = (1.8751 ** 2 / (2 * math.pi * 0.3 ** 2)) * math.sqrt(ei / mass)
    assert abs(m["natural_frequency_dry_hz"] - expected) / expected < 1e-6
    # with no fluid there is no added mass, so wet == dry
    assert abs(m["natural_frequency_hz"] - m["natural_frequency_dry_hz"]) < 1e-6


def test_added_mass_lowers_the_wet_frequency():
    sec = section_properties(load_dat("Data/Airfoils/4/n0012.dat.txt").xy)
    m = modal_properties(sec, fluid_density=998.0)
    assert m["natural_frequency_hz"] < m["natural_frequency_dry_hz"]
    assert m["added_mass_kg_m"] > 0


def test_material_frequency_ratio_follows_specific_stiffness():
    """Dry frequency scales as sqrt(E' / rho). Steel and aluminium have nearly
    the same specific stiffness, so their dry frequencies should be within a few
    percent -- a classic result, and a good check that the material properties
    are being used where they belong. The *wet* frequencies differ much more,
    because added mass penalises the lighter section, so the comparison is made
    dry."""
    sec = section_properties(load_dat("Data/Airfoils/4/n0012.dat.txt").xy)
    al = modal_properties(sec, material=ALUMINIUM)["natural_frequency_dry_hz"]
    st = modal_properties(sec, material=STEEL)["natural_frequency_dry_hz"]
    expected = math.sqrt(
        (STEEL.flexural_modulus / STEEL.density_kg_m3)
        / (ALUMINIUM.flexural_modulus / ALUMINIUM.density_kg_m3)
    )
    assert abs(st / al - expected) < 1e-6
    assert abs(expected - 1.0) < 0.10
