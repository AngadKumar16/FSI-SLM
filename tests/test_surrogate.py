"""End-to-end checks on the surrogate pipeline: case solve, tensors, dataset,
model shapes, and the digital-twin bridge.

Torch-dependent tests are skipped when torch is absent so the physics suite
still runs in a minimal environment.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sensor.bridge import (
    TwinConfig,
    classify_regime,
    sdof_amplification,
    separation_point,
    shedding_frequency,
    wake_features,
)
from sensor.flow.solve import solve_case
from sensor.geometry import load_dat
from sensor.sdf import Grid, signed_distance_and_edge
from sensor.simulate import CasePrep, TARGET_CHANNELS, pack_npz, simulate_case

FOIL = "Data/Airfoils/4/n0009sm.dat.txt"
GRID = Grid(nx=64, ny=64)

# ------------------------------------------------------------------ case solve


@pytest.fixture(scope="module")
def prep():
    return CasePrep.build(load_dat(FOIL))


@pytest.fixture(scope="module")
def case(prep):
    return simulate_case(prep, 8.0, 5.0e5, GRID)


def test_tensor_contract(case):
    assert case["input"].shape == (4, GRID.ny, GRID.nx)
    assert case["target"].shape == (5, GRID.ny, GRID.nx)
    assert case["mask"].shape == (GRID.ny, GRID.nx)
    assert case["input"].dtype == np.float32
    assert set(TARGET_CHANNELS) == {"u", "v", "p", "load", "disp"}


def test_channels_live_in_their_own_region(case):
    fluid = case["mask"] > 0
    solid = case["solid_mask"] > 0
    for i in range(3):
        assert np.all(case["target"][i][solid] == 0.0)
    for i in (3, 4):
        assert np.all(case["target"][i][fluid] == 0.0)
    assert not np.any(fluid & solid)


def test_masks_partition_the_grid(case):
    assert np.all((case["mask"] + case["solid_mask"]) == 1.0)


def test_sdf_sign_convention(prep):
    from sensor.flow.panel import resample_polygon
    from sensor.geometry import rotate

    xy = resample_polygon(rotate(prep.airfoil, 0.0).xy, 120)
    sdf, edge = signed_distance_and_edge(xy, GRID)
    xg, yg = GRID.meshgrid()
    # a point at mid-chord inside the section must be negative
    i = np.argmin(np.abs(yg[:, 0] - 0.0))
    j = np.argmin(np.abs(xg[0] - 0.3))
    assert sdf[i, j] < 0
    # far field must be positive and large
    assert sdf[0, 0] > 0.5
    assert edge.min() >= 0 and edge.max() < xy.shape[0]


def test_fields_are_bounded(case):
    u, v, p = case["target"][:3]
    assert np.abs(u).max() <= 2.0 + 1e-6
    assert np.abs(v).max() <= 2.0 + 1e-6
    assert p.min() >= -1.5 - 1e-6


def test_wake_momentum_equals_drag(prep):
    """The momentum-integral theorem, satisfied by construction -- so a failure
    means the wake model and the reported Cd have drifted apart."""
    grid = Grid()
    for alpha in (4.0, 12.0):
        r = solve_case(prep.airfoil, alpha, 5.0e5, grid,
                       constants=(prep.alpha_L0_deg, prep.cl_alpha_per_rad))
        xs = np.linspace(*grid.xlim, grid.nx)
        col = int(np.argmin(np.abs(xs - 1.5)))
        integral = 2.0 * float(r["wake_deficit"][:, col].sum()) * grid.dy
        assert abs(integral - r["coeffs"]["cd"]) / r["coeffs"]["cd"] < 0.05


def test_load_channel_vanishes_for_symmetric_foil_at_zero_incidence(prep):
    c = simulate_case(prep, 0.0, 5.0e5, GRID)
    assert np.abs(c["target"][3]).max() < 0.05
    assert np.abs(c["target"][4]).max() < 1e-3


def test_deflection_grows_with_load(prep):
    small = simulate_case(prep, 2.0, 5.0e5, GRID)["coeffs"]["w_max"]
    large = simulate_case(prep, 8.0, 5.0e5, GRID)["coeffs"]["w_max"]
    assert large > small


def test_pack_npz_roundtrip(case, tmp_path):
    packed = pack_npz(case)
    path = tmp_path / "case.npz"
    np.savez_compressed(path, **packed)
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        assert abs(float(z["scalars"][0]) - case["coeffs"]["cl"]) < 1e-5
        assert meta["coeffs"]["cl"] == pytest.approx(case["coeffs"]["cl"], rel=1e-6)
        assert z["target"].dtype == np.float16
        assert meta["fidelity"] == "panel"


# ------------------------------------------------------------------ bridge maths


def test_sdof_amplification_peaks_at_resonance():
    zeta = 0.01
    off = sdof_amplification(10.0, 30.0, zeta)
    on = sdof_amplification(30.0, 30.0, zeta)
    assert on > 20 * off
    assert abs(on - 1.0 / (2 * zeta)) / (1.0 / (2 * zeta)) < 0.01


def test_shedding_frequency_falls_as_the_wake_widens():
    cfg = TwinConfig()
    narrow, _ = shedding_frequency(0.1, cfg, 2.0, 0.09)
    wide, _ = shedding_frequency(0.8, cfg, 18.0, 0.09)
    assert narrow > wide


def test_shedding_frequency_warns_when_flow_is_attached():
    _, note = shedding_frequency(0.05, TwinConfig(), 2.0, 0.09)
    assert "attached" in note
    _, note2 = shedding_frequency(0.6, TwinConfig(), 16.0, 0.09)
    assert note2 == ""


def test_regime_rules():
    from Data.schema import FlowRegime

    assert classify_regime(1.0, 0.3, 5.0, 0.001, False) == FlowRegime.ATTACHED
    assert classify_regime(0.5, 0.3, 5.0, 0.001, False) == FlowRegime.SEPARATED
    assert classify_regime(0.9, 1.02, 5.0, 0.05, False) == FlowRegime.VIV_LOCKIN
    assert classify_regime(0.5, 0.3, -1.0, 0.05, False) == FlowRegime.GALLOPING
    assert classify_regime(0.5, 0.3, 5.0, 0.05, True) == FlowRegime.STALL_FLUTTER


def test_separation_point_detects_reverse_flow(prep):
    grid = Grid()
    attached = solve_case(prep.airfoil, 2.0, 5.0e5, grid,
                          constants=(prep.alpha_L0_deg, prep.cl_alpha_per_rad))
    stalled = solve_case(prep.airfoil, 18.0, 5.0e5, grid,
                         constants=(prep.alpha_L0_deg, prep.cl_alpha_per_rad))
    x_att = separation_point(attached["u"], attached["sdf"], grid)
    x_stall = separation_point(stalled["u"], stalled["sdf"], grid)
    assert x_att > 0.9
    assert x_stall < 0.6


def test_wake_features_widen_with_incidence(prep):
    grid = Grid()
    out = []
    for alpha in (2.0, 16.0):
        r = solve_case(prep.airfoil, alpha, 5.0e5, grid,
                       constants=(prep.alpha_L0_deg, prep.cl_alpha_per_rad))
        out.append(wake_features(r["u"], r["v"], r["sdf"], grid))
    assert out[1]["wake_width_over_c"] > 2 * out[0]["wake_width_over_c"]
    assert out[1]["turbulent_kinetic_energy"] > out[0]["turbulent_kinetic_energy"]


# ----------------------------------------------------------------- digital twin


def test_digital_twin_produces_a_valid_feature_record():
    from Data.schema import FeatureRecord
    from sensor.bridge import digital_twin

    out = digital_twin(FOIL, 12.0, TwinConfig(), grid=Grid())
    record = out["feature_record"]
    assert isinstance(record, FeatureRecord)
    assert record.conditions.angle_of_attack_deg == 12.0
    assert record.piv.shedding_frequency_hz > 0
    assert record.structure.natural_frequency_hz > 0
    assert out["provenance"]["source"] == "solver"
    assert "not measured" in out["provenance"]["signals"]


def test_digital_twin_flags_injected_faults():
    from sensor.bridge import digital_twin

    clean = digital_twin(FOIL, 10.0, TwinConfig(), grid=Grid())
    assert clean["noise_report"].verdict in {"clean", "minor_issues"}

    dirty = digital_twin(FOIL, 10.0, TwinConfig(), grid=Grid(),
                         inject_faults=("mains", "dropout"))
    names = {f.name for f in dirty["noise_report"].flags}
    assert "mains_interference" in names
    assert "piv_dropout" in names
    assert dirty["noise_report"].quality_score < clean["noise_report"].quality_score


def test_twin_record_is_json_serialisable():
    from sensor.api import twin_record

    json.dumps(twin_record(FOIL, 6.0, use_surrogate=False))


# ------------------------------------------------------------------ model shapes


def test_unet_shapes_and_parameter_count():
    torch = pytest.importorskip("torch")
    from sensor.model import UNet, UNetConfig, masked_loss

    model = UNet(UNetConfig(base=8, depth=3))
    x = torch.randn(2, 4, 64, 64)
    y, s = model(x)
    assert y.shape == (2, 5, 64, 64)
    assert s.shape == (2, 5)
    assert model.n_params > 0

    mask = torch.zeros(2, 2, 64, 64)
    mask[:, 0, :32] = 1.0
    mask[:, 1, 32:] = 1.0
    loss, parts = masked_loss(y, torch.randn_like(y), mask, s, torch.randn(2, 5))
    loss.backward()
    assert math.isfinite(parts["loss"])
    assert all(math.isfinite(v) for v in parts.values())


def test_masked_loss_ignores_the_other_region():
    torch = pytest.importorskip("torch")
    from sensor.model import masked_loss

    pred = torch.zeros(1, 5, 8, 8)
    target = torch.zeros(1, 5, 8, 8)
    target[:, :3] = 99.0          # fluid channels, but mask says solid-only
    mask = torch.zeros(1, 2, 8, 8)
    mask[:, 1] = 1.0
    _, parts = masked_loss(pred, target, mask, torch.zeros(1, 5), torch.zeros(1, 5))
    assert parts["flow"] == 0.0
    assert parts["struct"] == 0.0


def test_normalisation_guards_constant_channels():
    """A channel with zero variance must not get a ~0 scale.

    Reynolds is constant across the base sweep. With ``std + 1e-6`` as the
    scale, the float16 rounding in the cached inputs versus the float32 value
    computed at inference time became an input of order 100, and the predictor
    returned a constant wrong answer while the cache-based evaluation still
    reported R^2 = 0.99. This is the guard for that.
    """
    from sensor.data import _guard_std

    std = _guard_std(np.array([0.5, 0.0, 1e-9]), np.array([1.0, 0.7, 0.0]))
    assert std[0] == pytest.approx(0.5, abs=1e-6)
    assert std[1] == 1.0
    assert std[2] == 1.0


@pytest.mark.skipif(
    not Path("artifacts/sensor/run/best.pt").exists(),
    reason="no trained checkpoint",
)
def test_predictor_agrees_with_the_cached_evaluation_path():
    """predict() and evaluate_split() must see the same normalised input.

    These are two separate code paths onto the same network -- one builds the
    input tensor on the fly, one reads it from the float16 cache. They diverging
    is exactly the failure the guard above addresses, so it is pinned here.
    """
    pytest.importorskip("torch")
    import torch

    from sensor.data import SensorCache
    from sensor.predict import SurrogatePredictor

    p = SurrogatePredictor("artifacts/sensor/run/best.pt")
    cache = SensorCache("artifacts/sensor/cache")
    idx = cache.indices("test")
    if idx.size == 0:
        pytest.skip("empty test split")
    j = int(idx[len(idx) // 2])
    case = cache.cases[j]

    out = p.predict(case["dat"], case["alpha_deg"], 5.0e5)
    truth = np.asarray(cache.scalars[j], np.float32)
    assert abs(out["coeffs"]["cl"] - float(truth[0])) < 0.25
    assert abs(out["coeffs"]["cd"] - float(truth[1])) < 0.10


def test_downsampled_mask_never_invents_fluid():
    torch = pytest.importorskip("torch")
    from sensor.train import _downsample_batch

    mask = torch.ones(1, 2, 8, 8)
    mask[0, 0, 0, 0] = 0.0
    out = _downsample_batch(
        {"x": torch.zeros(1, 4, 8, 8), "y": torch.zeros(1, 5, 8, 8), "mask": mask}, 2
    )
    assert out["mask"][0, 0, 0, 0] == 0.0
