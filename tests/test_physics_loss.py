"""The physics terms in the loss are checked against the solver's own fields.

A physics-informed penalty is only informative if the quantity it penalises is
actually near zero on the *targets*. If the target fields carry a divergence of
0.5 then a divergence penalty is teaching the network to disagree with its own
training data. Each term below is therefore measured on real solver output
before it is trusted as a loss.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sensor.geometry import load_dat  # noqa: E402
from sensor.model import LossConfig, UNet, UNetConfig, border_circulation, surrogate_loss  # noqa: E402
from sensor.sdf import DEFAULT_SPEC, Grid  # noqa: E402
from sensor.simulate import CasePrep, simulate_case  # noqa: E402

FOILS = ("n0012", "n2414", "n6409")
ALPHAS = (-4.0, 0.0, 4.0, 8.0, 12.0)


@pytest.fixture(scope="module")
def cases():
    grid = Grid()
    out = []
    for foil in FOILS:
        prep = CasePrep.build(load_dat(f"Data/Airfoils/4/{foil}.dat.txt"))
        for a in ALPHAS:
            out.append(simulate_case(prep, a, grid=grid, with_modal=False))
    return out, grid


def test_border_circulation_recovers_lift(cases):
    """``2 * Gamma`` around the window border reproduces ``Cl`` to a few percent.

    Kelvin plus Kutta-Joukowski: the contour encloses the bound vortex and
    nothing else, so the circulation is ``Cl / 2``. The residual is systematic
    and one-signed -- the wake deficit leaves through the downstream edge and
    takes a little circulation with it -- which is precisely why the loss uses a
    difference of differences rather than the raw identity. It is the *sign* and
    the *scale* that this test protects.
    """
    rows, grid = cases
    ratios = []
    for c in rows:
        t = torch.from_numpy(c["target"][None])
        gamma = float(border_circulation(t[:, 0], t[:, 1], grid.dx, grid.dy)[0])
        cl = c["coeffs"]["cl"]
        if abs(cl) < 0.05:  # a symmetric section at zero lift says nothing
            assert abs(2 * gamma) < 0.05
            continue
        ratios.append(2 * gamma / cl)
    ratios = np.asarray(ratios)
    assert ratios.min() > 0.90, f"worst Cl recovery {ratios.min():.3f}"
    assert ratios.max() < 1.05, f"best Cl recovery {ratios.max():.3f}"


def test_target_velocity_is_nearly_divergence_free(cases):
    """The divergence penalty has to be near zero on the data it is applied to."""
    from sensor.model import _dx, _dy, _erode

    rows, grid = cases
    worst = 0.0
    for c in rows:
        t = torch.from_numpy(c["target"][None])
        m = torch.from_numpy(c["mask"][None, None])
        inner = _erode(m)
        div = _dx(t[:, 0:1]) / grid.dx + _dy(t[:, 1:2]) / grid.dy
        mean_div = float((div.abs() * inner).sum() / (inner.sum() + 1e-6))
        worst = max(worst, mean_div)
    # Units are 1/chord against a freestream of 1, so O(1) is the scale of a
    # velocity gradient and this is two orders below it.
    assert worst < 0.5, f"mean |div u| on the targets is {worst:.3f}"


def test_loss_terms_are_finite_and_reduce_on_a_perfect_prediction():
    rng = np.random.default_rng(0)
    b, h, w = 2, 32, 32
    y = torch.from_numpy(rng.normal(size=(b, 5, h, w)).astype(np.float32))
    mask = torch.zeros(b, 2, h, w)
    mask[:, 0, :, :] = 1.0
    mask[:, 0, 12:20, 12:20] = 0.0
    mask[:, 1] = 1.0 - mask[:, 0]
    s = torch.from_numpy(rng.normal(size=(b, 5)).astype(np.float32))
    wall = torch.rand(b, 1, h, w)
    kw = dict(cfg=LossConfig(), wall=wall, y_std=[0.4] * 5, s_std=[0.5] * 5,
              dx=0.024, dy=0.024)

    exact, log_exact = surrogate_loss(y, y, mask, s, s, **kw)
    noisy, log_noisy = surrogate_loss(y + 0.3, y, mask, s + 0.3, s, **kw)

    assert torch.isfinite(exact) and torch.isfinite(noisy)
    assert float(exact) < float(noisy)
    for key in ("flow", "struct", "scalar", "grad", "div", "circ"):
        assert key in log_noisy, f"{key} term missing from the loss log"
    # A constant offset has zero gradient and zero divergence, so those two
    # terms must be blind to it -- which is what makes them independent
    # information rather than a rescaled L1.
    assert log_noisy["grad"] == pytest.approx(log_exact["grad"], abs=1e-5)


def test_forward_and_backward_shapes():
    cfg = UNetConfig.for_spec(DEFAULT_SPEC, base=8, depth=3)
    model = UNet(cfg)
    x = torch.randn(2, DEFAULT_SPEC.n_channels, 64, 64)
    fields, scalars = model(x)
    assert fields.shape == (2, 5, 64, 64)
    assert scalars.shape == (2, 5)
    fields.sum().add_(scalars.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing[:5]}"


def test_condition_vector_is_read_exactly():
    """FiLM only works if the constant channels round-trip through the mean."""
    cfg = UNetConfig.for_spec(DEFAULT_SPEC, base=8, depth=2)
    model = UNet(cfg)
    x = torch.zeros(1, DEFAULT_SPEC.n_channels, 16, 16)
    for k, i in enumerate(cfg.global_idx):
        x[:, i] = 0.5 * (k + 1)
    cond = model._condition(x)
    assert torch.allclose(
        cond, torch.tensor([[0.5 * (k + 1) for k in range(len(cfg.global_idx))]])
    )


def test_r2_is_flagged_when_the_target_barely_varies():
    """A ratio against a vanishing denominator has to announce itself.

    The foil-disjoint split is stratified by series, not by response range, and
    the test foils turned out to be stiff in torsion: their trailing-edge twist
    varies about six times less than the validation foils'. The same checkpoint
    then scores R^2 = 0.96 on val and -0.08 on test while its RMSE hardly moves,
    and the second number reads as a broken model rather than a narrow split.
    """
    import numpy as np

    from sensor.predict import _r2_is_meaningful

    # The arithmetic the reporting relies on, stated directly: with residuals
    # the size of the spread, R^2 lands at zero no matter how good the model is
    # in absolute terms.
    true = np.array([1.0, 1.2, 0.9, 1.1, 1.05])
    pred = true + 0.15
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(((pred - true) ** 2).mean()))
    assert r2 < 0.0
    assert rmse == pytest.approx(0.15)
    # nrmse is the number that survives a narrow split.
    assert rmse / true.std() > 1.0

    # The flag itself, on the split the commit is about: the test foils have a
    # real spread (std ~ 1.46 over n = 308) but the error is the size of it, so
    # nrmse >= 1 and R^2 must NOT be reported as meaningful. A bare std > 0 test
    # -- what the first fix shipped -- would wrongly pass this case.
    assert _r2_is_meaningful(std=1.461, n=308, nrmse=1.05) is False
    # Same spread, a model that clears the mean baseline: R^2 is informative.
    assert _r2_is_meaningful(std=1.461, n=308, nrmse=0.19) is True
    # Near-constant target (the RANS rows): degenerate denominator, still flagged.
    assert _r2_is_meaningful(std=1e-9, n=308, nrmse=0.1) is False
