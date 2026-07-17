"""Physically-plausible synthetic FSI experiment generator.

This produces :class:`~data.schema.RawExperiment` objects -- time-resolved
signals plus a PIV snapshot stack -- whose statistics are *correlated the way a
real airfoil/hydrofoil/membrane FSI experiment would be*, not white noise. The
key physical couplings encoded here:

* Strouhal number sits in a realistic band (St ~ 0.15-0.25) and sets the wake
  shedding frequency via ``f_shed = St * U / c``.
* VIV lock-in is created by *placing the structural natural frequency near the
  shedding frequency* and giving the structure a large, saturated, sinusoidal
  limit cycle at that shared frequency.
* Galloping uses a low structural frequency (f_n << f_shed), large translational
  amplitude, growing trend, low St relevance.
* Stall flutter sits near/above static stall (high AoA), torsional-ish mode,
  shedding and structural frequencies close-ish but driven by dynamic stall.
* Separated flows have a broadband, energetic wake but only a weak structural
  response (no sustained limit cycle).
* Attached flows are quiet: small forces fluctuation, negligible structure.

Everything is seeded for reproducibility. The generator returns ``RawExperiment``
objects carrying ``ground_truth_regime`` so downstream evaluation has an answer
key; feature extraction discards nothing it shouldn't.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from Data.schema import FlowConditions, FlowRegime, RawExperiment

# Global default; every public entry point also accepts an explicit seed.
DEFAULT_SEED = 1234

# How long / how finely we sample the synthetic signals.
_DURATION_S = 8.0
_FS_HZ = 400.0  # sampling rate of time signals
_N_SNAPSHOTS = 64  # PIV snapshots for POD
_GRID = 12  # PIV spatial grid is _GRID x _GRID -> 144 points


@dataclass
class _RegimePlan:
    """Internal blueprint of the per-regime sampling distributions."""

    regime: FlowRegime
    re_log10_range: tuple
    aoa_range: tuple
    st_range: tuple
    # structural natural freq expressed as a multiple of the shedding freq
    fn_over_fshed_range: tuple
    struct_rms_range: tuple  # A_rms / c
    sep_xc_range: tuple
    wake_width_range: tuple
    tke_range: tuple
    cl_rms_range: tuple
    cd_mean_range: tuple
    camber_range: tuple
    mode_choices: tuple


# Per-regime physical blueprints. Numbers chosen to be individually plausible and
# jointly consistent (e.g. lock-in => fn ~ fshed; galloping => fn << fshed).
_PLANS: Dict[FlowRegime, _RegimePlan] = {
    FlowRegime.ATTACHED: _RegimePlan(
        regime=FlowRegime.ATTACHED,
        re_log10_range=(4.0, 5.5),
        aoa_range=(-4.0, 6.0),
        st_range=(0.16, 0.20),
        fn_over_fshed_range=(2.5, 6.0),  # structure far stiffer than shedding
        struct_rms_range=(0.0005, 0.006),
        sep_xc_range=(0.9, 1.0),
        wake_width_range=(0.1, 0.3),
        tke_range=(0.002, 0.02),
        cl_rms_range=(0.005, 0.03),
        cd_mean_range=(0.01, 0.05),
        camber_range=(0.0, 0.06),
        mode_choices=(1,),
    ),
    FlowRegime.SEPARATED: _RegimePlan(
        regime=FlowRegime.SEPARATED,
        re_log10_range=(4.0, 6.0),
        aoa_range=(10.0, 18.0),
        st_range=(0.15, 0.22),
        fn_over_fshed_range=(1.8, 4.0),
        struct_rms_range=(0.004, 0.03),
        sep_xc_range=(0.15, 0.5),
        wake_width_range=(0.6, 1.6),
        tke_range=(0.05, 0.2),
        cl_rms_range=(0.08, 0.25),
        cd_mean_range=(0.2, 0.8),
        camber_range=(0.0, 0.08),
        mode_choices=(1, 2),
    ),
    FlowRegime.VIV_LOCKIN: _RegimePlan(
        regime=FlowRegime.VIV_LOCKIN,
        re_log10_range=(3.5, 5.5),
        aoa_range=(0.0, 8.0),
        st_range=(0.17, 0.22),
        fn_over_fshed_range=(0.95, 1.05),  # synchronized
        struct_rms_range=(0.15, 0.6),  # large saturated limit cycle
        sep_xc_range=(0.4, 0.8),
        wake_width_range=(0.4, 0.9),
        tke_range=(0.03, 0.12),
        cl_rms_range=(0.15, 0.5),
        cd_mean_range=(0.1, 0.4),
        camber_range=(0.0, 0.12),
        mode_choices=(1, 2),
    ),
    FlowRegime.GALLOPING: _RegimePlan(
        regime=FlowRegime.GALLOPING,
        re_log10_range=(4.0, 6.0),
        aoa_range=(4.0, 14.0),
        st_range=(0.12, 0.18),
        fn_over_fshed_range=(0.1, 0.35),  # f_n << f_shed
        struct_rms_range=(0.2, 0.8),  # very large, low-freq
        sep_xc_range=(0.2, 0.6),
        wake_width_range=(0.7, 1.5),
        tke_range=(0.04, 0.15),
        cl_rms_range=(0.2, 0.6),
        cd_mean_range=(0.3, 1.0),
        camber_range=(0.0, 0.1),
        mode_choices=(1,),
    ),
    FlowRegime.STALL_FLUTTER: _RegimePlan(
        regime=FlowRegime.STALL_FLUTTER,
        re_log10_range=(4.0, 6.0),
        aoa_range=(12.0, 22.0),  # near/above static stall
        st_range=(0.14, 0.20),
        fn_over_fshed_range=(0.5, 0.9),  # SDOF torsional below shedding
        struct_rms_range=(0.1, 0.45),
        sep_xc_range=(0.1, 0.4),
        wake_width_range=(0.8, 1.8),
        tke_range=(0.06, 0.2),
        cl_rms_range=(0.2, 0.6),
        cd_mean_range=(0.25, 0.9),
        camber_range=(0.0, 0.08),
        mode_choices=(2, 3),  # often torsional / higher mode
    ),
}

ALL_REGIMES: List[FlowRegime] = list(_PLANS.keys())


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _make_signal(
    rng: np.random.Generator,
    t: np.ndarray,
    freq: float,
    amp: float,
    mean: float,
    noise: float,
    harmonics: bool = False,
) -> List[float]:
    """A mean + sinusoid (+ optional 2nd harmonic) + Gaussian noise signal."""
    phase = rng.uniform(0, 2 * math.pi)
    sig = mean + amp * np.sin(2 * math.pi * freq * t + phase)
    if harmonics:
        sig = sig + 0.3 * amp * np.sin(2 * math.pi * 2 * freq * t + rng.uniform(0, 2 * math.pi))
    sig = sig + noise * rng.standard_normal(t.shape)
    return sig.astype(float).tolist()


def _make_piv_snapshots(
    rng: np.random.Generator,
    shed_freq: float,
    energy_fracs: List[float],
    tke: float,
) -> List[List[float]]:
    """Build a PIV snapshot stack as a few spatial modes oscillating in time.

    Two coherent spatial modes (a shedding mode + its harmonic) carry most of the
    energy, scaled by ``energy_fracs``; broadband noise sets the TKE floor. SVD of
    this stack (in features.extract) recovers POD energy fractions close to
    ``energy_fracs``.
    """
    n_pts = _GRID * _GRID
    xs = np.linspace(0, 1, _GRID)
    ys = np.linspace(0, 1, _GRID)
    gx, gy = np.meshgrid(xs, ys)
    # Spatial mode shapes (flattened).
    mode1 = np.sin(math.pi * gx) * np.cos(math.pi * gy)
    mode2 = np.sin(2 * math.pi * gx) * np.cos(math.pi * gy)
    mode1 = (mode1 / np.linalg.norm(mode1)).ravel()
    mode2 = (mode2 / np.linalg.norm(mode2)).ravel()

    dt = 1.0 / _FS_HZ
    t = np.arange(_N_SNAPSHOTS) * dt * 6  # stride snapshots over several periods
    a1 = math.sqrt(max(energy_fracs[0], 1e-6)) * np.sin(2 * math.pi * shed_freq * t)
    a2 = math.sqrt(max(energy_fracs[1] if len(energy_fracs) > 1 else 0.0, 1e-6)) * np.sin(
        2 * math.pi * 2 * shed_freq * t + 0.5
    )
    coherent = np.outer(a1, mode1) + np.outer(a2, mode2)
    noise = math.sqrt(tke) * 0.25 * rng.standard_normal((_N_SNAPSHOTS, n_pts))
    snaps = coherent + noise
    return snaps.astype(float).tolist()


def generate_experiment(
    regime: FlowRegime,
    rng: np.random.Generator,
    index: int = 0,
) -> RawExperiment:
    """Generate a single physically-plausible ``RawExperiment`` for ``regime``."""
    plan = _PLANS[regime]

    re = 10.0 ** _u(rng, *plan.re_log10_range)
    aoa = _u(rng, *plan.aoa_range)
    chord = _u(rng, 0.05, 0.4)
    # Velocity from Reynolds (water-ish nu) but clamped to the schema's range.
    nu = 1.0e-6
    u_inf = re * nu / chord
    u_inf = float(np.clip(u_inf, 0.2, 40.0))

    st = _u(rng, *plan.st_range)
    f_shed = st * u_inf / chord
    f_shed = float(np.clip(f_shed, 0.5, 400.0))

    fn = float(np.clip(f_shed * _u(rng, *plan.fn_over_fshed_range), 0.5, 400.0))
    u_star = u_inf / (fn * chord)

    struct_rms = _u(rng, *plan.struct_rms_range)
    # The structure oscillates at f_n; in lock-in f_n ~ f_shed by construction.
    struct_freq = fn
    mode_number = int(rng.choice(plan.mode_choices))
    camber = _u(rng, *plan.camber_range)

    wake_width = _u(rng, *plan.wake_width_range)
    sep_xc = _u(rng, *plan.sep_xc_range)
    tke = _u(rng, *plan.tke_range)

    # Force coefficient statistics.
    cl_mean = 2 * math.pi * math.radians(aoa) * 0.9  # thin-airfoil-ish, attenuated
    if regime in (FlowRegime.SEPARATED, FlowRegime.STALL_FLUTTER):
        cl_mean *= 0.6  # lift loss past stall
    cl_rms = _u(rng, *plan.cl_rms_range)
    cd_mean = _u(rng, *plan.cd_mean_range)
    cd_rms = _u(rng, 0.3, 0.8) * cl_rms
    cm_mean = -0.05 - 0.02 * camber * 100
    cm_rms = _u(rng, 0.2, 0.6) * cl_rms

    # Time base and signals.
    t = np.arange(0, _DURATION_S, 1.0 / _FS_HZ)

    # Structural displacement oscillates at struct_freq.
    disp = _make_signal(
        rng, t, struct_freq, amp=struct_rms * math.sqrt(2), mean=0.0,
        noise=0.05 * struct_rms + 1e-4, harmonics=(regime == FlowRegime.STALL_FLUTTER),
    )
    # Wake probe oscillates at shedding freq.
    wake = _make_signal(
        rng, t, f_shed, amp=0.3 * u_inf * (0.3 + tke), mean=u_inf,
        noise=u_inf * tke * 0.5, harmonics=True,
    )
    # Forces: lift dominated by shedding freq; in lock-in this equals struct freq.
    f_force = f_shed if regime != FlowRegime.GALLOPING else struct_freq
    cl_sig = _make_signal(rng, t, f_force, amp=cl_rms * math.sqrt(2), mean=cl_mean, noise=0.1 * cl_rms)
    cd_sig = _make_signal(rng, t, 2 * f_force, amp=cd_rms * math.sqrt(2), mean=cd_mean, noise=0.1 * cd_rms + 1e-4)
    cm_sig = _make_signal(rng, t, f_force, amp=cm_rms * math.sqrt(2), mean=cm_mean, noise=0.1 * cm_rms + 1e-4)

    # POD energy fractions: more coherent (concentrated) for lock-in/galloping.
    if regime in (FlowRegime.VIV_LOCKIN, FlowRegime.GALLOPING):
        energy = [_u(rng, 0.55, 0.75), _u(rng, 0.12, 0.22)]
    elif regime == FlowRegime.ATTACHED:
        energy = [_u(rng, 0.35, 0.5), _u(rng, 0.15, 0.25)]
    else:
        energy = [_u(rng, 0.3, 0.45), _u(rng, 0.15, 0.25)]
    snaps = _make_piv_snapshots(rng, f_shed, energy, tke)

    exp_id = f"{regime.value}-{index:05d}"
    return RawExperiment(
        experiment_id=exp_id,
        conditions=FlowConditions(
            reynolds_number=re,
            angle_of_attack_deg=aoa,
            freestream_velocity_m_s=u_inf,
            reduced_velocity=u_star,
            chord_m=chord,
        ),
        sampling_rate_hz=_FS_HZ,
        structural_displacement_signal=disp,
        wake_velocity_probe_signal=wake,
        cl_signal=cl_sig,
        cd_signal=cd_sig,
        cm_signal=cm_sig,
        piv_snapshots=snaps,
        natural_frequency_hz=fn,
        mode_number=mode_number,
        camber=camber,
        wake_width_over_c=wake_width,
        separation_location_x_over_c=sep_xc,
        turbulent_kinetic_energy=tke,
        ground_truth_regime=regime,
        notes="synthetic",
    )


def generate_dataset(
    n_per_regime: int = 40,
    seed: int = DEFAULT_SEED,
    regimes: Optional[List[FlowRegime]] = None,
) -> List[RawExperiment]:
    """Generate a balanced list of raw experiments across all regimes.

    Parameters
    ----------
    n_per_regime: number of experiments per regime.
    seed: master seed (reproducible).
    regimes: subset of regimes to generate; defaults to all five.
    """
    regimes = regimes or ALL_REGIMES
    rng = np.random.default_rng(seed)
    experiments: List[RawExperiment] = []
    counter = 0
    for regime in regimes:
        for _ in range(n_per_regime):
            experiments.append(generate_experiment(regime, rng, index=counter))
            counter += 1
    # Shuffle so train/val/test splits are not regime-ordered.
    rng.shuffle(experiments)
    return experiments


__all__ = ["generate_experiment", "generate_dataset", "ALL_REGIMES", "DEFAULT_SEED"]
