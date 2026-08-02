"""The mirror parity tables are checked against the solver, not asserted.

A symmetric section at ``-alpha`` *is* the mirror of the same section at
``+alpha``. Nothing about that depends on the surrogate, the raster or the
normalisation, so it is a statement the panel + beam solver has to satisfy to
solver precision -- and if any sign in :mod:`sensor.symmetry` is wrong, it will
not.

This is the test that makes the augmentation safe to switch on. Without it the
augmentation is a pile of sign conventions applied to a loss function, and a
wrong sign there does not crash: it trains, converges, and quietly halves the
accuracy of every antisymmetric quantity.
"""
from __future__ import annotations

import numpy as np
import pytest

from sensor.geometry import load_dat
from sensor.sdf import DEFAULT_SPEC, SPEC_V1, Grid, build_input
from sensor.simulate import CasePrep, simulate_case
from sensor.symmetry import (
    MASK_PARITY,
    SCALAR_PARITY,
    TARGET_PARITY,
    mirror_np,
    mirror_scalars_np,
    normalised_field_parity,
    normalised_scalar_parity,
)

SYMMETRIC_FOIL = "Data/Airfoils/4/n0012.dat.txt"
ALPHA = 8.0


@pytest.fixture(scope="module")
def pair():
    """The same symmetric section solved at +alpha and -alpha."""
    prep = CasePrep.build(load_dat(SYMMETRIC_FOIL))
    grid = Grid()
    return (
        simulate_case(prep, +ALPHA, grid=grid, with_modal=False),
        simulate_case(prep, -ALPHA, grid=grid, with_modal=False),
        grid,
    )


def test_grid_is_mirror_symmetric():
    g = Grid()
    assert g.y_symmetric, "a row flip is only the mirror if ylim is symmetric"


def _off_wall(grid, alpha_deg=-ALPHA):
    """Pixels outside the solver's painted near-wall band.

    Inside that band each pixel adopts the tangential velocity of its *nearest
    panel*, and within a cell of the stagnation point that lookup is a near-tie
    which resolves to different panels at ``+alpha`` and ``-alpha``. It is a
    discretisation tie-break, not a parity error, and it is the only place in
    the whole field where the two cases disagree at all.
    """
    from sensor.geometry import rotate
    from sensor.sdf import signed_distance

    sdf = signed_distance(rotate(load_dat(SYMMETRIC_FOIL), alpha_deg).xy, grid)
    return (sdf > 2.5 * max(grid.dx, grid.dy)).astype(np.float32)


def test_target_fields_mirror(pair):
    """Every field channel mirrors, exactly, outside the painted wall layer.

    The grid flip is exact -- ``ny = 128`` over a symmetric ``ylim`` maps row
    ``i`` to ``ny-1-i`` with ``y -> -y`` to machine precision -- so any residual
    at all is the solver's own.
    """
    plus, minus, grid = pair
    got = mirror_np(plus["target"], TARGET_PARITY)
    want = minus["target"]
    off_wall = _off_wall(grid)

    for i, name in enumerate(("u", "v", "p", "load", "disp")):
        scale = np.abs(want[i]).max() + 1e-9
        diff = np.abs(got[i] - want[i]) / scale
        assert diff.mean() < 1e-4, f"channel {name}: mean mirror error {diff.mean():.3g}"
        region = off_wall if i < 3 else np.ones_like(diff)
        worst = float((diff * region).max())
        # Float32 rounding, four orders of magnitude below the physics.
        assert worst < 1e-4, (
            f"channel {name}: mirror error {worst:.3g} outside the wall band, "
            f"which should be exact to rounding"
        )


def test_masks_mirror(pair):
    plus, minus, _ = pair
    m_plus = np.stack([plus["mask"], plus["solid_mask"]])
    m_minus = np.stack([minus["mask"], minus["solid_mask"]])
    got = mirror_np(m_plus, MASK_PARITY)
    # A handful of boundary pixels can differ; the body is ~1,100 pixels.
    assert np.abs(got - m_minus).sum() < 0.01 * m_minus[1].sum() + 5


def test_scalars_mirror(pair):
    plus, minus, _ = pair
    names = ("cl", "cd", "cm", "w_max", "twist_te_deg")
    s_plus = np.array([plus["coeffs"][n] for n in names])
    s_minus = np.array([minus["coeffs"][n] for n in names])
    got = mirror_scalars_np(s_plus)
    scale = np.maximum(np.abs(s_minus), 1e-6)
    err = np.abs(got - s_minus) / scale
    for n, e in zip(names, err):
        assert e < 0.02, f"scalar {n}: relative mirror error {e:.3g}"


def test_cd_and_w_max_do_not_flip_sign():
    """The two invariant scalars, called out because they are the easy mistake."""
    assert SCALAR_PARITY[1] == +1.0, "cd is a magnitude and must not negate"
    assert SCALAR_PARITY[3] == +1.0, "w_max is max|w| and must not negate"
    assert SCALAR_PARITY[4] == -1.0, "twist_te_deg is a signed slope and must negate"


@pytest.mark.parametrize("spec", [DEFAULT_SPEC, SPEC_V1])
def test_input_tensor_mirrors(pair, spec):
    plus, minus, grid = pair
    af = CasePrep.build(load_dat(SYMMETRIC_FOIL)).airfoil
    x_plus = build_input(af, +ALPHA, 5e5, grid, spec)
    x_minus = build_input(af, -ALPHA, 5e5, grid, spec)
    got = mirror_np(x_plus, spec.parity)
    off_wall = _off_wall(grid)
    for i, name in enumerate(spec.names):
        scale = np.abs(x_minus[i]).max() + 1e-9
        diff = np.abs(got[i] - x_minus[i]) / scale
        assert diff.mean() < 1e-4, (
            f"input channel {name}: mean mirror error {diff.mean():.3g}"
        )
        # The prior channels carry the same near-wall band as the targets do.
        region = off_wall if name.startswith("pot_") else np.ones_like(diff)
        worst = float((diff * region).max())
        assert worst < 1e-4, f"input channel {name}: mirror error {worst:.3g}"


def test_normalised_mirror_is_an_involution():
    """Applying the normalised-space mirror twice has to be the identity.

    It is only an involution because the shift is nonzero exactly where the
    parity is -1, and there the second application cancels it. If that ever
    stops holding, test-time mirror averaging silently biases every channel.
    """
    mean = [0.4, -0.1, -0.3, 0.0, 0.02]
    std = [0.5, 0.2, 0.4, 0.1, 0.01]
    par, shift = normalised_field_parity(mean, std)
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 8, 8)).astype(np.float32)

    def M(z):
        return z[:, ::-1, :] * par[:, None, None] + shift[:, None, None]

    assert np.allclose(M(M(a)), a, atol=1e-6)

    s_par, s_shift = normalised_scalar_parity(mean, std)
    s = rng.normal(size=5).astype(np.float32)
    assert np.allclose((s * s_par + s_shift) * s_par + s_shift, s, atol=1e-6)
