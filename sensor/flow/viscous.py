"""Viscous correction on top of the inviscid panel solution.

The panel method gives an exact potential-flow answer, which is right up to
about 8-10 degrees and then keeps climbing linearly forever. Real foils stall.
This module applies the standard Beddoes-Leishman *static* separation model,
which is the workhorse used in rotor and wind-turbine aeroelastic codes:

1. A separation-point function ``f(alpha)`` -- the fraction of chord with
   attached flow on the suction side -- decays from 1 through 0.7 at the stall
   break ``alpha_1`` down to 0.04 in deep stall.

2. Kirchhoff-Helmholtz flat-plate theory relates that separation point to the
   loss of normal force::

       Cn = Cn_inviscid * ((1 + sqrt(f)) / 2)^2

3. Leading-edge suction supplies a chord force ``Cc`` that vanishes as the flow
   separates, so pressure drag grows as ``Cn sin(alpha) - Cc cos(alpha)``.

Skin friction is a Prandtl-Schlichting flat plate with a Hoerner thickness form
factor and a partial-laminar-run correction below Re = 1e6.

Calibration check (NACA 0012, Re = 5e5, from Sheldahl & Klimas / Abbott & von
Doenhoff): model gives Cl(10 deg) ~ 1.05, Cl_max ~ 1.2 near 14 deg,
Cl(20 deg) ~ 0.85, Cd_min ~ 0.010, Cd(20 deg) ~ 0.30. See ``tests/test_viscous.py``.

Everything here is *static*: no dynamic stall, no hysteresis. The dataset is a
steady sweep, and the ``sensor`` package documents itself as a steady-RANS-class
surrogate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

# Beddoes-Leishman shape constants. S1 governs the gentle pre-stall departure,
# S2 the collapse after the break. Values are the commonly cited defaults,
# nudged with thickness (thick foils stall more gradually).
_S1_BASE = 3.0  # deg
_S2_BASE = 1.6  # deg
_ETA_SUCTION = 0.95  # leading-edge suction recovery factor


@dataclass
class ViscousState:
    """Everything the wake model and the dataset writer need."""

    alpha_deg: float
    alpha_eff_deg: float  # relative to the zero-lift line
    alpha_L0_deg: float
    alpha_stall_deg: float
    f_sep: float  # attached-flow chord fraction, 0.04 .. 1
    x_sep: float  # separation point, chord fraction from LE
    stalled: bool
    cl: float
    cl_inviscid: float
    cd: float
    cd_friction: float
    cd_pressure: float
    cm_quarter: float
    cn: float
    cc: float
    gamma_scale: float  # multiplier on bound circulation for the field
    theta_te: float  # wake momentum thickness at the TE, chord units
    reynolds: float
    thickness: float

    def to_dict(self) -> dict:
        return {k: (bool(v) if isinstance(v, bool) else float(v))
                for k, v in asdict(self).items()}


class ViscousModel:
    """Thickness- and Reynolds-parameterised static stall + drag model."""

    def __init__(self, thickness: float, reynolds: float = 5.0e5):
        self.thickness = float(max(thickness, 0.02))
        self.reynolds = float(max(reynolds, 1.0e4))

    # ------------------------------------------------------------ ingredients

    @property
    def alpha_stall_deg(self) -> float:
        """Stall-break angle: thicker foil and higher Re stall later.

        ``7 + 55 t/c`` reproduces the classic NACA 4-digit trend (0006 ~ 10 deg,
        0012 ~ 14 deg, 0018 ~ 17 deg) and the log-Re term captures the ~2 deg
        gain per decade seen between Re = 1e5 and 1e6.
        """
        base = 7.0 + 55.0 * self.thickness
        re_shift = 2.0 * math.log10(self.reynolds / 5.0e5)
        return float(np.clip(base + re_shift, 6.0, 20.0))

    @property
    def cf(self) -> float:
        """Prandtl-Schlichting turbulent flat-plate skin friction."""
        return 0.455 / (math.log10(self.reynolds) ** 2.58)

    @property
    def cd_friction(self) -> float:
        ff = 1.0 + 2.0 * self.thickness + 60.0 * self.thickness ** 4  # Hoerner
        # partial laminar run below Re = 1e6 lowers the effective Cf
        lam = 0.80 + 0.20 * min(1.0, self.reynolds / 1.0e6)
        return 2.0 * self.cf * ff * lam

    def separation_point(self, alpha_eff_deg: float, camber_shift_deg: float = 0.0) -> float:
        """Beddoes-Leishman ``f``: attached chord fraction, in [0.04, 1].

        ``camber_shift_deg`` delays the break for cambered sections: camber
        unloads the leading edge at a given effective incidence, so a 4412
        holds attached flow a couple of degrees past a 0012.
        """
        a = abs(alpha_eff_deg)
        a1 = self.alpha_stall_deg + max(camber_shift_deg, 0.0)
        s1 = _S1_BASE * (1.0 + 2.0 * (self.thickness - 0.09))
        s2 = _S2_BASE * (1.0 + 2.0 * (self.thickness - 0.09))
        s1 = max(s1, 1.0)
        s2 = max(s2, 0.6)
        if a <= a1:
            f = 1.0 - 0.3 * math.exp((a - a1) / s1)
        else:
            f = 0.04 + 0.66 * math.exp((a1 - a) / s2)
        return float(np.clip(f, 0.04, 1.0))

    # ----------------------------------------------------------------- solve

    def apply(
        self,
        alpha_deg: float,
        cl_inviscid: float,
        cm_inviscid: float,
        cl_alpha_per_rad: float = 2.0 * math.pi,
        alpha_L0_deg: float = 0.0,
    ) -> ViscousState:
        """Correct one inviscid solution.

        Parameters
        ----------
        alpha_deg
            Geometric angle of attack.
        cl_inviscid, cm_inviscid
            Panel-method lift and quarter-chord moment.
        cl_alpha_per_rad
            Inviscid lift slope; the panel value is ~2*pi*(1 + 0.77 t/c).
        alpha_L0_deg
            Zero-lift angle from the panel solve at alpha = 0.
        """
        a_rad = math.radians(alpha_deg)
        ae_deg = alpha_deg - alpha_L0_deg
        ae_rad = math.radians(ae_deg)

        camber_shift = 0.35 * abs(alpha_L0_deg)
        f = self.separation_point(ae_deg, camber_shift)
        kirchhoff = ((1.0 + math.sqrt(f)) / 2.0) ** 2

        # Normal force in the body frame, then Kirchhoff-attenuated.
        cn_inv = cl_inviscid / max(math.cos(ae_rad), 0.2)
        cn = cn_inv * kirchhoff

        # Leading-edge suction chord force -- a thrust that dies as the flow
        # separates. Scaling it with the *achieved* normal force rather than
        # with alpha^2 keeps it bounded in deep stall; the alpha^2 form
        # over-predicts suction past 15 deg and cancels most of the pressure
        # drag, which is exactly where the drag should be largest.
        cc = _ETA_SUCTION * cn * ae_rad * math.sqrt(f)

        # Resolve into wind axes about the zero-lift line. Using alpha_e rather
        # than the geometric alpha matters for cambered sections: the suction
        # term is built from alpha_e, so resolving with alpha would mix two
        # different reference lines and drive the pressure drag negative.
        cl = cn * math.cos(ae_rad) + cc * math.sin(ae_rad)
        cd_press = cn * math.sin(ae_rad) - cc * math.cos(ae_rad)
        cd_press = float(max(cd_press, 0.0))
        cd = self.cd_friction + cd_press

        # Centre of pressure marches aft toward mid-chord as the foil stalls.
        x_cp = 0.25 + 0.22 * (1.0 - f) ** 1.5
        cm = cm_inviscid * kirchhoff - (x_cp - 0.25) * cn

        gamma_scale = kirchhoff
        # Momentum thickness at the TE from the total drag (Squire-Young inverse,
        # ue/U ~ 1 at the TE for an attached foil).
        theta_te = 0.5 * cd

        return ViscousState(
            alpha_deg=float(alpha_deg),
            alpha_eff_deg=float(ae_deg),
            alpha_L0_deg=float(alpha_L0_deg),
            alpha_stall_deg=self.alpha_stall_deg + camber_shift,
            f_sep=f,
            x_sep=f,
            stalled=bool(f < 0.7),
            cl=float(cl),
            cl_inviscid=float(cl_inviscid),
            cd=float(cd),
            cd_friction=float(self.cd_friction),
            cd_pressure=float(cd_press),
            cm_quarter=float(cm),
            cn=float(cn),
            cc=float(cc),
            gamma_scale=float(gamma_scale),
            theta_te=float(theta_te),
            reynolds=self.reynolds,
            thickness=self.thickness,
        )
