"""Assemble one full case: inviscid panel field + viscous corrections + wake.

``solve_case`` is the single entry point the dataset generator calls. It returns
the pixel-aligned ``(u, v, p)`` fields on :class:`sensor.sdf.Grid`, the fluid
mask, the surface pressure distribution (which the structural solver consumes)
and the integrated coefficients.

Field construction, in order:

1. **Inviscid velocity** from the panel solution, with the bound circulation
   scaled by the Kirchhoff factor so the far field carries the *stalled* lift
   rather than the potential-flow lift.
2. **Separated region.** For ``f < 1`` a smooth blob hugging the suction side
   from the separation point to roughly one bubble length past the trailing
   edge, inside which the velocity is blended toward a weak reverse flow. This
   is what makes a stalled case *look* stalled to the network.
3. **Far wake.** A Gaussian velocity deficit on the trailing streamline whose
   integral is exactly ``Cd/2``, i.e. the momentum-integral theorem is satisfied
   by construction, with turbulent ``sqrt(theta * x)`` spreading.
4. **Pressure.** Bernoulli outside the wake; inside, the total-pressure loss
   implied by the local deficit is subtracted so the far wake relaxes to
   ``p = 0``, plus a base-pressure suction in the recirculation bubble.

Limitations, stated plainly: this is a steady, incompressible, 2-D,
attached-or-mildly-separated model. It is not a substitute for RANS in deep
stall, and cases with ``f < 0.15`` are tagged ``deep_stall`` in the manifest so
they can be down-weighted or regenerated with OpenFOAM.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from ..geometry import Airfoil, close_trailing_edge, rotate
from ..sdf import Grid, signed_distance_and_edge
from .panel import PanelSolver, resample_polygon
from .viscous import ViscousModel, ViscousState

N_PANELS = 180
# Velocity ceiling, in units of Uinf. |V| = 2 <=> Cp = -3.
_VMAX = 2.0


@dataclass
class FlowCase:
    """Inputs that define one dataset row."""

    airfoil: Airfoil
    alpha_deg: float
    reynolds: float = 5.0e5
    n_panels: int = N_PANELS


def foil_constants(af: Airfoil, n_panels: int = N_PANELS) -> Tuple[float, float]:
    """Zero-lift angle (deg) and inviscid lift slope (per rad) for one foil.

    Two extra panel solves; the batch driver caches this per foil rather than
    per (foil, alpha) case.
    """
    af = close_trailing_edge(af)
    xy0 = resample_polygon(af.xy, n_panels)
    cl0 = PanelSolver(xy0).solve(0.0).cl
    xy5 = resample_polygon(rotate(af, 5.0).xy, n_panels)
    cl5 = PanelSolver(xy5).solve(0.0).cl
    cl_alpha = (cl5 - cl0) / np.deg2rad(5.0)
    if not np.isfinite(cl_alpha) or cl_alpha < 1.0:
        cl_alpha = 2.0 * np.pi
    alpha_L0 = -cl0 / cl_alpha * 180.0 / np.pi
    return float(alpha_L0), float(cl_alpha)


def _trailing_edge(xy: np.ndarray) -> Tuple[float, float]:
    i = int(np.argmax(xy[:, 0]))
    return float(xy[i, 0]), float(xy[i, 1])


def _suction_side_sign(vs: ViscousState) -> float:
    """+1 if the suction side is the upper surface (positive effective alpha)."""
    return 1.0 if vs.alpha_eff_deg >= 0.0 else -1.0


def _separated_weight(
    xg: np.ndarray,
    yg: np.ndarray,
    xy: np.ndarray,
    vs: ViscousState,
    t_floor: float = 0.02,
) -> np.ndarray:
    """Smooth 0..1 field marking the recirculating separated region.

    The weight is anchored *at the wall* and decays outward. An earlier version
    tapered the bubble to zero thickness at the separation point, which meant
    the first cells off the surface stayed attached-looking and no reverse flow
    ever appeared in the raster -- the downstream separation detector then saw
    every case as attached. A separation bubble reverses the flow at the wall
    first; ``t_floor`` (a couple of grid cells) keeps that representable.
    """
    sep = 1.0 - vs.f_sep
    # Below a few percent there is no bubble worth representing: the model would
    # otherwise plant a one-cell blob just aft of the trailing edge on foils
    # whose flow is attached to 99% of chord, which is noise for the network to
    # learn and visibly wrong in the rendered field.
    if sep <= 0.04:
        return np.zeros_like(xg)

    side = _suction_side_sign(vs)
    x_te, y_te = _trailing_edge(xy)
    x_sep = float(np.clip(vs.x_sep, 0.02, 0.98)) * x_te
    # bubble runs from the separation point to ~1 bubble length past the TE
    bubble_len = 0.15 + 1.3 * sep
    x0, x1 = x_sep, x_te + bubble_len
    t_max = 0.05 + 0.55 * sep

    xi = (xg - x0) / max(x1 - x0, 1e-6)
    # Rise to full strength within the first 12% of the bubble, hold, then fade
    # over the last 40% -- rather than a sine hump that is weak exactly where
    # the separation begins.
    ramp = np.clip(xi / 0.12, 0.0, 1.0)
    fade = np.clip((1.0 - xi) / 0.40, 0.0, 1.0)
    w_x = np.where((xi < 0.0) | (xi > 1.0), 0.0, ramp * fade)

    surf = _surface_height(xg, xy, side)
    t_local = np.maximum(t_max * np.clip(xi, 0.0, 1.5) ** 0.5, t_floor)
    dn = side * (yg - surf)  # normal distance out from the suction surface
    # Two-sided, both sides smooth. Outward the bubble decays over its own
    # thickness; inward it decays over a couple of cells so it cannot leak
    # through the body -- but as a taper, not a hard cut. A step here is
    # extremely visible downstream of the trailing edge, where the "surface"
    # is just the clipped TE height and the cut ran straight across the wake.
    t_in = np.minimum(t_local, 0.06)
    w_y = (
        np.exp(-np.clip(dn, 0.0, None) ** 2 / (2.0 * t_local ** 2))
        * np.exp(-np.clip(-dn, 0.0, None) ** 2 / (2.0 * t_in ** 2))
    )

    return np.clip(w_x * w_y, 0.0, 1.0)


def _surface_height(xg: np.ndarray, xy: np.ndarray, side: float) -> np.ndarray:
    """y of the suction (side=+1) or pressure (side=-1) surface at each grid x.

    The two surfaces have to be separated by which side of the *chord line* a
    point falls on, not by the sign of its ``y``. On a foil rotated to 12
    degrees a good part of the lower surface sits above ``y = 0`` near the
    leading edge and part of the upper surface sits below it near the trailing
    edge, so a sign-of-y test interleaves the two and the interpolated surface
    jumps between them -- which showed up in the rendered fields as a
    rectangular step downstream of the trailing edge and a separation bubble
    detached from the body.
    """
    i_le = int(np.argmin(xy[:, 0]))
    i_te = int(np.argmax(xy[:, 0]))
    le, te = xy[i_le], xy[i_te]
    chord = te - le
    length = float(np.hypot(*chord))
    if length < 1e-9:
        return np.zeros_like(xg)
    e_c = chord / length
    e_n = np.array([-e_c[1], e_c[0]])

    dn = (xy - le) @ e_n
    sel = (side * dn) >= 0.0
    xs, ys = xy[sel, 0], xy[sel, 1]
    if xs.size < 3:
        xs, ys = xy[:, 0], xy[:, 1]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    xq = np.clip(xg, float(xy[:, 0].min()), float(xy[:, 0].max()))
    return np.interp(xq, xs, ys)


def _wake_deficit(
    xg: np.ndarray,
    yg: np.ndarray,
    xy: np.ndarray,
    vs: ViscousState,
) -> np.ndarray:
    """Gaussian far-wake deficit with integral exactly Cd/2."""
    x_te, y_te = _trailing_edge(xy)
    xi = xg - x_te
    downstream = xi > 0.0
    if not np.any(downstream):
        return np.zeros_like(xg)

    sep = 1.0 - vs.f_sep
    b0 = 0.5 * vs.theta_te + 0.01 + 0.35 * sep  # initial half-width
    # Self-similar turbulent plane wake: b ~ C sqrt(theta x) with C ~ 0.5, i.e.
    # b^2 ~ 0.25 theta x. (An earlier coefficient of 1.44 spread the wake about
    # 2.4x too fast, which showed up downstream as a wake width of 0.17 c behind
    # an attached foil where the momentum thickness only supports ~0.07 c.)
    sigma = np.sqrt(b0 ** 2 + 0.25 * max(vs.theta_te, 1e-4) * np.clip(xi, 0.0, None))
    # wake sheds roughly along the freestream from the TE
    eta = yg - y_te
    amp = (0.5 * vs.cd) / (sigma * np.sqrt(2.0 * np.pi))
    amp = np.clip(amp, 0.0, 0.98)
    # Fade the far-wake model in over the first 0.15 c. Switching it on at the
    # trailing edge leaves a visible vertical step for high-drag cases, and in
    # that first stretch the near-wake bubble is already carrying the deficit.
    # The momentum integral is unaffected downstream of the ramp, which is where
    # it is defined and where the test measures it.
    ramp = np.clip((xg - x_te) / 0.15, 0.0, 1.0)
    d = amp * np.exp(-eta ** 2 / (2.0 * sigma ** 2)) * ramp
    return np.where(downstream, d, 0.0)


def _regularise_near_wall(
    u: np.ndarray,
    v: np.ndarray,
    sdf: np.ndarray,
    edge: np.ndarray,
    solver: "PanelSolver",
    cp_wall: np.ndarray,
    fluid: np.ndarray,
    delta: float,
) -> None:
    """Clip the panel singularity and paint a no-slip layer. In place.

    Factored out because the *prior* raster (the inviscid field handed to the
    network as an input channel) needs exactly the same treatment as the target
    raster. Without it the first pixel off the wall sits on the source/vortex
    singularity and reports ``|V| ~ 2.5``, and a physics prior that is wrong
    precisely at the surface -- where the load comes from -- is worse than none.
    """
    np.clip(u, -_VMAX, _VMAX, out=u)
    np.clip(v, -_VMAX, _VMAX, out=v)
    vt_wall = np.sign(solver._sol.vt) * np.sqrt(np.clip(1.0 - cp_wall, 0.0, None))
    near = fluid & (sdf < delta)
    if not np.any(near):
        return
    e = edge[near]
    prof = np.clip(sdf[near] / delta, 0.0, 1.0) ** (1.0 / 7.0)
    blend = 1.0 - np.clip(sdf[near] / delta, 0.0, 1.0) ** 2
    u[near] = (1.0 - blend) * u[near] + blend * (vt_wall[e] * solver.tx[e] * prof)
    v[near] = (1.0 - blend) * v[near] + blend * (vt_wall[e] * solver.ty[e] * prof)


def potential_prior(
    af: Airfoil,
    alpha_deg: float,
    grid: Optional[Grid] = None,
    n_panels: int = N_PANELS,
    sdf: Optional[np.ndarray] = None,
    edge: Optional[np.ndarray] = None,
) -> dict:
    """Inviscid ``(u, v, p)`` raster for one case, standalone.

    This is the physics prior the surrogate is conditioned on. It is the same
    computation :func:`solve_case` performs internally, exposed separately for
    the OpenFOAM path -- a RANS row has no panel solve of its own, but the
    network still has to be given the same input channels, or panel-generated
    and RANS-generated rows would not be interchangeable.
    """
    grid = grid or Grid()
    af = close_trailing_edge(af)
    xy = resample_polygon(rotate(af, alpha_deg).xy, n_panels)
    solver = PanelSolver(xy)
    solver.solve(0.0)
    if sdf is None or edge is None:
        sdf, edge = signed_distance_and_edge(solver.xy, grid)
    xg, yg = grid.meshgrid()
    (u, v), = solver.velocity_multi(xg, yg, (1.0,))
    fluid = sdf > 0.0
    _regularise_near_wall(
        u, v, sdf, edge, solver, solver._sol.cp, fluid,
        max(2.0 * grid.dx, 0.012),
    )
    p = 0.5 * (1.0 - (u * u + v * v))
    np.clip(p, -0.5 * (_VMAX ** 2 - 1.0), 0.5, out=p)
    return {
        "u": np.where(fluid, u, 0.0).astype(np.float32),
        "v": np.where(fluid, v, 0.0).astype(np.float32),
        "p": np.where(fluid, p, 0.0).astype(np.float32),
        "sdf": np.asarray(sdf, dtype=np.float32),
        "cl": float(solver._sol.cl),
    }


def solve_case(
    af: Airfoil,
    alpha_deg: float,
    reynolds: float = 5.0e5,
    grid: Optional[Grid] = None,
    n_panels: int = N_PANELS,
    constants: Optional[Tuple[float, float]] = None,
) -> dict:
    """Solve one case. Returns fields, masks, surface data and coefficients."""
    grid = grid or Grid()
    af = close_trailing_edge(af)  # idempotent; a blunt TE breaks the Kutta condition
    alpha_L0, cl_alpha = constants if constants is not None else foil_constants(af, n_panels)

    rot = rotate(af, alpha_deg)
    xy = resample_polygon(rot.xy, n_panels)
    solver = PanelSolver(xy)
    inviscid = solver.solve(0.0)  # geometry already rotated; freestream is +x

    visc = ViscousModel(thickness=af.thickness, reynolds=reynolds)
    state = visc.apply(
        alpha_deg=alpha_deg,
        cl_inviscid=inviscid.cl,
        cm_inviscid=inviscid.cm_quarter,
        cl_alpha_per_rad=cl_alpha,
        alpha_L0_deg=alpha_L0,
    )

    xg, yg = grid.meshgrid()
    # Use the solver's own polygon: PanelSolver may reverse the point order to
    # force clockwise panels, and the near-wall model indexes panels by number.
    sdf, edge = signed_distance_and_edge(solver.xy, grid)
    fluid = sdf > 0.0

    # One influence pass, two circulation scalings: the inviscid field (the
    # physics prior the network is given) and the circulation-corrected field
    # the viscous model builds the target from. Evaluating them separately
    # doubled the most expensive part of a case for no reason.
    (u_pot, v_pot), (u, v) = solver.velocity_multi(
        xg, yg, (1.0, state.gamma_scale)
    )

    # --- near-wall layer ----------------------------------------------------
    # Bound the raw potential field, then paint a no-slip layer. A pixel that
    # lands within a panel length of the surface sees the 1/r singularity;
    # |V| = 2 is Cp = -3, already at the top of what a real suction peak reaches
    # on these sections, so anything past it is numerical, not physical. Inside
    # a band of a couple of cells the field is replaced by the panel's own
    # surface tangential velocity carrying a 1/7-power turbulent profile, which
    # both removes the singularity and gives the network a resolved no-slip
    # layer. The band width is set by the raster, not by Re: at 42 px/chord a
    # real Re = 5e5 boundary layer is thinner than one cell, so this is the
    # finest wall structure the target grid can represent.
    delta = max(2.0 * grid.dx, 0.012)
    cp_wall = surf_cp = solver._sol.cp * state.gamma_scale + (1.0 - state.gamma_scale) * (-0.2)
    _regularise_near_wall(u, v, sdf, edge, solver, cp_wall, fluid, delta)
    _regularise_near_wall(
        u_pot, v_pot, sdf, edge, solver, solver._sol.cp, fluid, delta
    )
    p_pot = 0.5 * (1.0 - (u_pot * u_pot + v_pot * v_pot))
    np.clip(p_pot, -0.5 * (_VMAX ** 2 - 1.0), 0.5, out=p_pot)

    # --- separated region: blend toward weak reverse flow -------------------
    w_sep = _separated_weight(xg, yg, xy, state, t_floor=2.5 * grid.dy)
    u_rev = -0.12  # mild reverse flow inside the bubble
    u = (1.0 - w_sep) * u + w_sep * u_rev
    v = (1.0 - w_sep) * v + w_sep * (0.02 * _suction_side_sign(state))

    # --- far wake: momentum-consistent Gaussian deficit ---------------------
    d = _wake_deficit(xg, yg, xy, state)
    u = u - d

    # --- pressure ------------------------------------------------------------
    vmag2 = u * u + v * v
    p = 0.5 * (1.0 - vmag2)
    # total-pressure loss so the far wake relaxes to p = 0 rather than a
    # spurious Bernoulli suction
    p = p - (d - 0.5 * d * d)
    # base pressure in the recirculation bubble drives the pressure drag
    p = p - w_sep * (0.10 + 0.45 * (1.0 - state.f_sep))

    np.clip(u, -_VMAX, _VMAX, out=u)
    np.clip(v, -_VMAX, _VMAX, out=v)
    np.clip(p, -0.5 * (_VMAX ** 2 - 1.0), 0.5, out=p)

    u = np.where(fluid, u, 0.0).astype(np.float32)
    v = np.where(fluid, v, 0.0).astype(np.float32)
    p = np.where(fluid, p, 0.0).astype(np.float32)

    surf = solver.surface()
    cp_visc = cp_wall  # already viscous-scaled above

    return {
        "u": u,
        "v": v,
        "p": p,
        # Inviscid prior on the same raster: input channels, not targets.
        "u_pot": np.where(fluid, u_pot, 0.0).astype(np.float32),
        "v_pot": np.where(fluid, v_pot, 0.0).astype(np.float32),
        "p_pot": np.where(fluid, p_pot, 0.0).astype(np.float32),
        "sdf": sdf.astype(np.float32),
        "fluid_mask": fluid.astype(np.float32),
        "solid_mask": (~fluid).astype(np.float32),
        "separated_weight": w_sep.astype(np.float32),
        "wake_deficit": d.astype(np.float32),
        "surface": {
            "xc": surf["xc"],
            "yc": surf["yc"],
            "nx": surf["nx"],
            "ny": surf["ny"],
            "s": surf["s"],
            "cp": cp_visc,
            "cp_inviscid": surf["cp"],
        },
        "polygon": xy,
        "state": state,
        "inviscid": inviscid,
        "coeffs": {
            "cl": state.cl,
            "cd": state.cd,
            "cm": state.cm_quarter,
            "cl_inviscid": inviscid.cl,
            "cd_friction": state.cd_friction,
            "cd_pressure": state.cd_pressure,
            "f_sep": state.f_sep,
            "alpha_L0_deg": alpha_L0,
            "cl_alpha_per_rad": cl_alpha,
            "stalled": state.stalled,
        },
    }
