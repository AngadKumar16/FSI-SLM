"""Close the loop: surrogate fields -> ``FeatureRecord`` -> language model.

The U-Net predicts a *steady* flow and a *static* deflection. The interpreter
consumes a record of a *time-resolved* experiment. This module is the joint
between them, and it is where the physics that turns one into the other lives:

``wake_features``
    Reads the predicted velocity field: wake width from the half-deficit points
    one and a half chords downstream, separation point from the first chord
    station where the near-wall flow reverses, turbulence level from the deficit
    magnitude.

``shedding_frequency``
    Roshko's universal Strouhal number, ``St_d = f d / U ~ 0.2`` on the *wake
    width* rather than the chord. That is the scaling that holds across bluff
    and stalled bodies, and it correctly makes the frequency rise as the wake
    narrows at low incidence.

``structural_response``
    A single-degree-of-freedom transfer function. The static deflection comes
    from the surrogate; the dynamic amplitude is that deflection times
    ``1 / sqrt((1 - r^2)^2 + (2 zeta r)^2)`` at ``r = f_shed / f_n``. This is
    what makes a lock-in case actually show a large response instead of one
    asserted by a label.

``classify_regime``
    Rule-based, from the quantities above: attached / separated by separation
    point, lock-in by frequency ratio, galloping by the Den Hartog criterion
    ``dCl/dalpha + Cd < 0`` evaluated with two extra solver calls, stall flutter
    by proximity to the stall break with a torsional response.

The synthesised signals are explicitly a *model of the instrument*, not a
measurement, and every record produced here carries ``notes`` saying so and a
``provenance`` block naming the surrogate checkpoint. When the PIV rig comes
online its ingest replaces :func:`build_raw_experiment` and nothing downstream
changes -- that is the point of the ``RawExperiment`` contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .geometry import Airfoil, load_dat
from .sdf import Grid
from .simulate import CasePrep, simulate_case
from .structure.beam import modal_properties

# Roshko's universal Strouhal number, based on wake width.
ST_UNIVERSAL = 0.20
# Sampling model for the synthesised instrument.
FS_HZ = 400.0
DURATION_S = 8.0
N_SNAPSHOTS = 64
PIV_GRID = 12


@dataclass
class TwinConfig:
    """Rig parameters that turn nondimensional surrogate output into SI."""

    chord_m: float = 0.1
    span_m: float = 0.3
    u_inf_m_s: float = 5.0
    fluid_density: float = 998.0
    kinematic_viscosity: float = 1.0e-6
    damping_ratio: float = 0.01
    material: str = "aluminium"
    sampling_rate_hz: float = FS_HZ
    duration_s: float = DURATION_S

    @property
    def reynolds(self) -> float:
        return self.u_inf_m_s * self.chord_m / self.kinematic_viscosity


# ------------------------------------------------------------------ field reads


def wake_features(
    u: np.ndarray,
    v: np.ndarray,
    sdf: np.ndarray,
    grid: Grid,
    x_station: float = 1.5,
) -> Dict[str, float]:
    """Wake width, peak deficit and turbulence level from a velocity field."""
    xs = np.linspace(*grid.xlim, u.shape[1])
    ys = np.linspace(*grid.ylim, u.shape[0])
    col = int(np.argmin(np.abs(xs - x_station)))
    profile = u[:, col]
    fluid = sdf[:, col] > 0
    if fluid.sum() < 8:
        return {"wake_width_over_c": 0.1, "peak_deficit": 0.0,
                "turbulent_kinetic_energy": 0.0}

    deficit = np.clip(1.0 - profile, 0.0, None) * fluid
    peak = float(deficit.max())
    if peak <= 1e-4:
        width = 0.05
    else:
        half = deficit >= 0.5 * peak
        idx = np.where(half)[0]
        width = float(abs(ys[idx[-1]] - ys[idx[0]])) if idx.size > 1 else float(grid.dy)
        width = max(width, 2.0 * grid.dy)

    # Turbulence level: a plane wake's streamwise fluctuation is roughly a third
    # of the local mean deficit, and TKE ~ 1.5 u'^2 for near-isotropic wake
    # turbulence.
    u_rms = 0.33 * peak
    tke = 1.5 * u_rms ** 2
    return {
        "wake_width_over_c": float(np.clip(width, 0.01, 2.0)),
        "peak_deficit": peak,
        "turbulent_kinetic_energy": float(np.clip(tke, 1e-5, 0.5)),
    }


def separation_point(
    u: np.ndarray,
    sdf: np.ndarray,
    grid: Grid,
    suction_side: float = 1.0,
    band: float = 2.0,
) -> float:
    """First chord station where the flow just off the suction surface reverses.

    ``band`` is the search depth in cells: close enough to be inside the shear
    layer, far enough not to sit on the wall value itself.
    """
    ny, nx = u.shape
    xs = np.linspace(*grid.xlim, nx)
    ys = np.linspace(*grid.ylim, ny)
    inside_cols = np.where((xs >= 0.0) & (xs <= 1.0))[0]
    if inside_cols.size == 0:
        return 1.0

    for c in inside_cols:
        solid_rows = np.where(sdf[:, c] <= 0)[0]
        if solid_rows.size == 0:
            continue
        # step out from the surface on the suction side
        edge = solid_rows.max() if suction_side > 0 else solid_rows.min()
        probe = int(edge + band) if suction_side > 0 else int(edge - band)
        if 0 <= probe < ny and sdf[probe, c] > 0 and u[probe, c] < 0.0:
            return float(np.clip(xs[c], 0.0, 1.0))
    return 1.0


# ------------------------------------------------------------- dynamics model


def shedding_frequency(
    wake_width_over_c: float,
    cfg: TwinConfig,
    alpha_deg: float = 0.0,
    thickness_over_c: float = 0.09,
) -> Tuple[float, str]:
    """Shedding frequency and a note on how far to trust it.

    The length scale is the Fage-Johansen projected height
    ``d = c (sin|alpha| + t/c)`` -- the standard correlation for airfoil wakes --
    or the wake width actually measured in the field, whichever is larger.

    Two honest caveats come back with the number. Below roughly 6 degrees an
    attached foil has no organised von Karman street at all: what sheds is a
    trailing-edge-scale instability, so the frequency is high and the
    chord-based Strouhal number is correspondingly larger than the 0.1-0.4 band
    that bluff-body and stalled-airfoil records occupy. Reporting 0.2 there
    would be tidier and wrong.
    """
    projected = abs(math.sin(math.radians(alpha_deg))) + thickness_over_c
    d_over_c = max(projected, wake_width_over_c)
    d = max(d_over_c * cfg.chord_m, 1e-4)
    f = float(ST_UNIVERSAL * cfg.u_inf_m_s / d)
    note = ""
    if wake_width_over_c < 0.12 and abs(alpha_deg) < 6.0:
        note = (
            "flow is attached: no organised vortex street, the reported "
            "frequency is trailing-edge-scale and chord-based St is above the "
            "usual bluff-body band"
        )
    return f, note


def sdof_amplification(f_forcing: float, f_natural: float, zeta: float) -> float:
    """Magnitude of the single-degree-of-freedom transfer function."""
    if f_natural <= 0:
        return 1.0
    r = f_forcing / f_natural
    return float(1.0 / math.sqrt((1.0 - r ** 2) ** 2 + (2.0 * zeta * r) ** 2))


def den_hartog_margin(
    prep: CasePrep,
    alpha_deg: float,
    reynolds: float,
    grid: Grid,
    delta_deg: float = 1.0,
) -> float:
    """``dCl/dalpha + Cd``. Negative means galloping-unstable.

    Two extra solver calls at plus and minus one degree; cheap next to the
    field solve and the only honest way to get the local lift slope, which is
    what the criterion actually needs (it goes negative past stall).
    """
    lo = simulate_case(prep, alpha_deg - delta_deg, reynolds, grid, with_modal=False)
    hi = simulate_case(prep, alpha_deg + delta_deg, reynolds, grid, with_modal=False)
    dcl_dalpha = (hi["coeffs"]["cl"] - lo["coeffs"]["cl"]) / math.radians(2 * delta_deg)
    cd = 0.5 * (hi["coeffs"]["cd"] + lo["coeffs"]["cd"])
    return float(dcl_dalpha + cd)


def classify_regime(
    sep_x: float,
    f_ratio: float,
    den_hartog: Optional[float],
    amplitude_over_c: float,
    near_stall: bool,
):
    """Rule-based regime label. Returns a :class:`Data.schema.FlowRegime`."""
    from Data.schema import FlowRegime

    if den_hartog is not None and den_hartog < 0.0 and amplitude_over_c > 0.02:
        return FlowRegime.GALLOPING
    if abs(f_ratio - 1.0) <= 0.10 and amplitude_over_c > 0.01:
        return FlowRegime.VIV_LOCKIN
    if near_stall and amplitude_over_c > 0.015:
        return FlowRegime.STALL_FLUTTER
    if sep_x < 0.9:
        return FlowRegime.SEPARATED
    return FlowRegime.ATTACHED


# ------------------------------------------------------ record synthesis


def _piv_snapshots(
    rng: np.random.Generator,
    f_shed: float,
    energy: Sequence[float],
    tke: float,
) -> np.ndarray:
    """Two coherent wake modes oscillating in time, plus a turbulence floor."""
    xs = np.linspace(0.0, 1.0, PIV_GRID)
    ys = np.linspace(0.0, 1.0, PIV_GRID)
    gx, gy = np.meshgrid(xs, ys)
    m1 = np.sin(np.pi * gx) * np.cos(np.pi * gy)
    m2 = np.sin(2 * np.pi * gx) * np.cos(np.pi * gy)
    m1 = (m1 / np.linalg.norm(m1)).ravel()
    m2 = (m2 / np.linalg.norm(m2)).ravel()
    t = np.arange(N_SNAPSHOTS) * (6.0 / FS_HZ)
    a1 = math.sqrt(max(energy[0], 1e-6)) * np.sin(2 * np.pi * f_shed * t)
    a2 = math.sqrt(max(energy[1], 1e-6)) * np.sin(4 * np.pi * f_shed * t + 0.5)
    coherent = np.outer(a1, m1) + np.outer(a2, m2)
    return coherent + math.sqrt(tke) * 0.25 * rng.standard_normal(coherent.shape)


def _signal(rng, t, freq, amp, mean, noise, harmonics=False):
    phase = rng.uniform(0.0, 2 * np.pi)
    sig = mean + amp * np.sin(2 * np.pi * freq * t + phase)
    if harmonics:
        sig = sig + 0.3 * amp * np.sin(4 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return sig + noise * rng.standard_normal(t.shape)


def build_raw_experiment(
    case: dict,
    cfg: TwinConfig,
    experiment_id: str,
    seed: int = 0,
    den_hartog: Optional[float] = None,
    inject_faults: Sequence[str] = (),
):
    """Assemble a ``RawExperiment`` from one surrogate/solver case.

    ``inject_faults`` deliberately corrupts the synthesised signals -- one of
    ``clipping``, ``dropout``, ``mains``, ``drift``, ``spike``, ``flatline`` --
    so the noise detector and the model's response to it can be exercised
    end-to-end. Nothing injects faults by default.
    """
    from Data.schema import FlowConditions, RawExperiment

    rng = np.random.default_rng(seed)
    coeffs = case["coeffs"]
    grid = case["grid"]
    u, v, sdf = case["u"], case["v"], case["sdf"]

    wf = wake_features(u, v, sdf, grid)
    suction = 1.0 if case["alpha_deg"] >= 0 else -1.0
    sep_x = separation_point(u, sdf, grid, suction)
    f_shed, shed_note = shedding_frequency(
        wf["wake_width_over_c"], cfg, case["alpha_deg"],
        float(coeffs.get("max_thickness", 0.09)),
    )

    modal = case["modal"]
    f_n = modal["natural_frequency_hz"]
    zeta = cfg.damping_ratio
    ratio = f_shed / max(f_n, 1e-6)

    # Static deflection in chord units, then the SDOF response at f_shed.
    w_static = abs(coeffs.get("w_max", 0.0))
    amp = w_static * sdof_amplification(f_shed, f_n, zeta)
    amp = float(np.clip(amp, 1e-5, 0.35))

    near_stall = bool(coeffs.get("f_sep", 1.0) < 0.7)
    regime = classify_regime(sep_x, ratio, den_hartog, amp, near_stall)

    n = int(cfg.duration_s * cfg.sampling_rate_hz)
    t = np.arange(n) / cfg.sampling_rate_hz

    # Structural response rides at f_shed when locked in, otherwise at f_n.
    f_struct = f_shed if abs(ratio - 1.0) <= 0.15 else f_n
    disp = _signal(rng, t, f_struct, amp, 0.0, 0.05 * amp + 1e-5,
                   harmonics=near_stall)
    wake = _signal(rng, t, f_shed, wf["peak_deficit"] * cfg.u_inf_m_s * 0.5,
                   cfg.u_inf_m_s * (1.0 - 0.5 * wf["peak_deficit"]),
                   cfg.u_inf_m_s * wf["turbulent_kinetic_energy"] ** 0.5 * 0.5,
                   harmonics=True)

    # Force fluctuation grows with separation; attached foils are quiet.
    cl_rms = 0.02 + 0.60 * (1.0 - float(coeffs.get("f_sep", 1.0)))
    cd_rms = 0.3 * cl_rms
    cl_sig = _signal(rng, t, f_shed, cl_rms * math.sqrt(2), coeffs["cl"], 0.1 * cl_rms)
    cd_sig = _signal(rng, t, 2 * f_shed, cd_rms * math.sqrt(2), coeffs["cd"],
                     0.1 * cd_rms + 1e-4)
    cm_sig = _signal(rng, t, f_shed, 0.5 * cl_rms * math.sqrt(2), coeffs["cm"],
                     0.05 * cl_rms + 1e-4)

    energy = [0.45 + 0.25 * (1.0 - coeffs.get("f_sep", 1.0)), 0.20, 0.08, 0.04]
    snaps = _piv_snapshots(rng, f_shed, energy, wf["turbulent_kinetic_energy"])

    signals = {
        "disp": disp, "wake": wake, "cl": cl_sig, "cd": cd_sig, "cm": cm_sig,
    }
    signals, snaps, fault_notes = _inject_faults(signals, snaps, inject_faults,
                                                 cfg, rng)

    return RawExperiment(
        experiment_id=experiment_id,
        conditions=FlowConditions(
            reynolds_number=cfg.reynolds,
            angle_of_attack_deg=float(case["alpha_deg"]),
            freestream_velocity_m_s=cfg.u_inf_m_s,
            reduced_velocity=float(modal["reduced_velocity"]),
            chord_m=cfg.chord_m,
        ),
        sampling_rate_hz=cfg.sampling_rate_hz,
        structural_displacement_signal=signals["disp"].tolist(),
        wake_velocity_probe_signal=signals["wake"].tolist(),
        cl_signal=signals["cl"].tolist(),
        cd_signal=signals["cd"].tolist(),
        cm_signal=signals["cm"].tolist(),
        piv_snapshots=snaps.tolist(),
        natural_frequency_hz=float(f_n),
        mode_number=1,
        camber=float(case.get("camber", 0.0)),
        youngs_modulus_pa=float(modal.get("flexural_rigidity_nm2", 0.0)) or None,
        density_kg_m3=cfg.fluid_density,
        wake_width_over_c=wf["wake_width_over_c"],
        separation_location_x_over_c=float(sep_x),
        turbulent_kinetic_energy=wf["turbulent_kinetic_energy"],
        ground_truth_regime=regime,
        notes=(
            f"surrogate-derived ({case.get('source', 'solver')}); signals are a "
            f"1-DOF instrument model driven by steady surrogate output, not a "
            f"measurement"
            + (f"; {shed_note}" if shed_note else "")
            + (f"; injected faults: {fault_notes}" if fault_notes else "")
        ),
    )


def _inject_faults(signals, snaps, faults, cfg: TwinConfig, rng):
    """Corrupt the synthesised record in the ways real instruments fail."""
    faults = list(faults)
    if not faults:
        return signals, snaps, ""
    out = {k: np.array(v, dtype=float) for k, v in signals.items()}
    snaps = np.array(snaps, dtype=float)
    n = out["cl"].size
    t = np.arange(n) / cfg.sampling_rate_hz
    for f in faults:
        if f == "clipping":
            # Rail about the signal's own mean. Clipping symmetrically about
            # zero on a channel with a large offset (Cl at incidence) flattens
            # the entire trace into a constant, which is a different fault.
            mu = float(np.mean(out["cl"]))
            amp = float(np.max(np.abs(out["cl"] - mu))) or 1.0
            out["cl"] = np.clip(out["cl"], mu - 0.6 * amp, mu + 0.6 * amp)
        elif f == "dropout":
            k = max(1, int(0.08 * snaps.size))
            flat = snaps.ravel()
            flat[rng.choice(flat.size, k, replace=False)] = 0.0
            snaps = flat.reshape(snaps.shape)
        elif f == "mains":
            amp = 3.0 * float(np.std(out["disp"]))
            out["disp"] = out["disp"] + amp * np.sin(2 * np.pi * 50.0 * t)
        elif f == "drift":
            out["cd"] = out["cd"] + np.linspace(0.0, 4.0 * float(np.std(out["cd"])), n)
        elif f == "spike":
            idx = rng.choice(n, max(2, n // 150), replace=False)
            out["wake"][idx] += 10.0 * float(np.std(out["wake"]))
        elif f == "flatline":
            out["disp"][n // 3: n // 3 + 60] = out["disp"][n // 3]
    return out, snaps, ",".join(faults)


# ------------------------------------------------------------------ top level


def solver_case(
    airfoil: Union[str, Path, Airfoil],
    alpha_deg: float,
    cfg: Optional[TwinConfig] = None,
    grid: Optional[Grid] = None,
) -> dict:
    """Run the physics solver and shape the result for :func:`digital_twin`."""
    cfg = cfg or TwinConfig()
    grid = grid or Grid()
    af = airfoil if isinstance(airfoil, Airfoil) else load_dat(airfoil)
    prep = CasePrep.build(af)
    case = simulate_case(
        prep, alpha_deg, cfg.reynolds, grid,
        chord_m=cfg.chord_m, span_m=cfg.span_m, u_inf_m_s=cfg.u_inf_m_s,
        fluid_density=cfg.fluid_density,
    )
    return {
        "u": case["target"][0], "v": case["target"][1], "p": case["target"][2],
        "load": case["target"][3], "disp": case["target"][4],
        "sdf": case["input"][0], "grid": grid,
        "coeffs": case["coeffs"], "modal": case["modal"],
        "alpha_deg": alpha_deg, "airfoil": af.name, "source": "solver",
        "camber": float(np.mean(case["section"].camber)),
        "_prep": prep,
    }


def surrogate_case(
    predictor,
    airfoil: Union[str, Path, Airfoil],
    alpha_deg: float,
    cfg: Optional[TwinConfig] = None,
    grid: Optional[Grid] = None,
) -> dict:
    """Run the trained U-Net and shape the result for :func:`digital_twin`."""
    from .structure.beam import ALUMINIUM, section_properties

    cfg = cfg or TwinConfig()
    grid = grid or Grid()
    af = airfoil if isinstance(airfoil, Airfoil) else load_dat(airfoil)
    pred = predictor.predict(af, alpha_deg, cfg.reynolds, grid)

    section = section_properties(af.xy)
    modal = modal_properties(
        section, chord_m=cfg.chord_m, span_m=cfg.span_m, material=ALUMINIUM,
        fluid_density=cfg.fluid_density, u_inf_m_s=cfg.u_inf_m_s,
        damping_ratio=cfg.damping_ratio,
    )
    # The network predicts the separation state implicitly; recover a scalar
    # f_sep from where the near-wall flow reverses so the downstream rules have
    # the same quantity the solver would have given them.
    sdf = pred["sdf"]
    g = Grid(nx=sdf.shape[1], ny=sdf.shape[0], xlim=grid.xlim, ylim=grid.ylim)
    sep = separation_point(pred["fields"]["u"], sdf, g,
                           1.0 if alpha_deg >= 0 else -1.0)
    coeffs = dict(pred["coeffs"])
    coeffs["f_sep"] = float(sep)
    return {
        "u": pred["fields"]["u"], "v": pred["fields"]["v"], "p": pred["fields"]["p"],
        "load": pred["fields"]["load"], "disp": pred["fields"]["disp"],
        "sdf": sdf, "grid": g,
        "coeffs": coeffs, "modal": modal,
        "alpha_deg": alpha_deg, "airfoil": af.name, "source": "unet",
        "camber": float(np.mean(section.camber)),
        "checkpoint": pred.get("checkpoint"),
    }


def digital_twin(
    airfoil: Union[str, Path, Airfoil],
    alpha_deg: float,
    cfg: Optional[TwinConfig] = None,
    grid: Optional[Grid] = None,
    predictor=None,
    experiment_id: Optional[str] = None,
    seed: int = 0,
    inject_faults: Sequence[str] = (),
    with_den_hartog: bool = True,
) -> dict:
    """Geometry + incidence -> fields -> ``FeatureRecord`` + data-quality report.

    Pass ``predictor`` (a :class:`sensor.predict.SurrogatePredictor`) to run the
    network; leave it ``None`` to run the physics solver directly. Both paths
    produce the same record type, which is what makes the surrogate swappable.
    """
    from features.extract import extract_features
    from features.noise import analyse_raw_experiment

    cfg = cfg or TwinConfig()
    grid = grid or Grid()
    case = (
        surrogate_case(predictor, airfoil, alpha_deg, cfg, grid)
        if predictor is not None
        else solver_case(airfoil, alpha_deg, cfg, grid)
    )

    dh = None
    if with_den_hartog:
        prep = case.get("_prep")
        if prep is None:
            af = airfoil if isinstance(airfoil, Airfoil) else load_dat(airfoil)
            prep = CasePrep.build(af)
        dh = den_hartog_margin(prep, alpha_deg, cfg.reynolds, grid)

    exp_id = experiment_id or f"twin_{case['airfoil'].replace(' ', '')}_a{alpha_deg:g}"
    raw = build_raw_experiment(case, cfg, exp_id, seed=seed, den_hartog=dh,
                               inject_faults=inject_faults)
    record = extract_features(raw)
    noise = analyse_raw_experiment(raw)

    return {
        "feature_record": record,
        "raw_experiment": raw,
        "noise_report": noise,
        "case": {k: v for k, v in case.items() if not k.startswith("_")
                 and not isinstance(v, np.ndarray)},
        "fields": {k: case[k] for k in ("u", "v", "p", "load", "disp", "sdf")},
        "den_hartog_margin": dh,
        "provenance": {
            "source": case["source"],
            "checkpoint": case.get("checkpoint"),
            "grid": [case["grid"].ny, case["grid"].nx],
            "rig": {
                "chord_m": cfg.chord_m, "span_m": cfg.span_m,
                "u_inf_m_s": cfg.u_inf_m_s, "reynolds": cfg.reynolds,
                "damping_ratio": cfg.damping_ratio,
            },
            "signals": "synthesised by a 1-DOF instrument model, not measured",
        },
    }
