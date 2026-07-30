"""Chordwise Euler-Bernoulli beam + spanwise cantilever modal model.

Two separate structural models, because a hydrofoil experiment has two distinct
deformation stories and the abstract needs both:

**Chordwise (the field model).** The section is a beam running from leading to
trailing edge, supported at the spar (quarter chord by default) and free at both
ends, loaded by the net pressure difference ``dCp(x)``. Local bending stiffness
comes from the local thickness, ``EI(x) = E t(x)^3 / 12`` per unit span. Solved
with 2-node cubic-Hermite beam elements -- exact for piecewise-constant ``EI``
and piecewise-linear load, and unconditionally stable, unlike a naive
fourth-order finite-difference stencil on a variable-coefficient operator.
This produces the per-pixel **load** and **displacement** target channels.

**Spanwise (the modal model).** The same section swept into a cantilever of
span ``L``, giving the first bending frequency, modal mass, mass ratio and
reduced velocity. Those are what :mod:`sensor.bridge` needs to hand the SLM a
``FeatureRecord`` with a physically consistent structural side -- lock-in is
defined by ``f_shed / f_n``, so the surrogate has to produce ``f_n``.

Nondimensionalisation. The flow solver works in ``Uinf = 1, c = 1, rho = 1``
units, so a single dimensionless stiffness

    E* = E / (0.5 rho U^2)

fixes the deflection scale. ``E* = 2e4`` gives peak deflections of order 1% of
chord for a 12%-thick section -- the regime a flexible-hydrofoil PIV rig
actually operates in, and a well-conditioned target for a network. The value
used is written into every dataset row so it is never implicit.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

# Dimensionless bending stiffness E / (0.5 rho U^2). See module docstring.
DEFAULT_E_STAR = 2.0e4
DEFAULT_SPAR_X = 0.25


@dataclass(frozen=True)
class Material:
    """SI material properties, used only by the dimensional modal model."""

    name: str
    youngs_modulus_pa: float
    density_kg_m3: float
    poisson: float = 0.33

    @property
    def flexural_modulus(self) -> float:
        """Plate flexural modulus E / (1 - nu^2); a wide foil bends as a plate."""
        return self.youngs_modulus_pa / (1.0 - self.poisson ** 2)


ALUMINIUM = Material("aluminium_6061", 69.0e9, 2700.0, 0.33)
STEEL = Material("stainless_316", 193.0e9, 8000.0, 0.30)
PVC = Material("pvc", 3.0e9, 1400.0, 0.38)
POLYCARBONATE = Material("polycarbonate", 2.3e9, 1200.0, 0.37)


@dataclass
class SectionProperties:
    """Geometric properties of the (unrotated) section, chord normalised to 1."""

    x: np.ndarray  # (M,) chord stations, 0..1
    thickness: np.ndarray  # (M,) local thickness t(x)/c
    camber: np.ndarray  # (M,) mean line y(x)/c
    area: float  # enclosed area / c^2
    i_zz: float  # second moment about the chordwise centroidal axis / c^4
    max_thickness: float
    max_thickness_x: float


@dataclass
class StructuralResult:
    """Chordwise beam solution plus derived scalars."""

    x: np.ndarray  # (M,) chord stations
    load: np.ndarray  # (M,) net dCp(x) = Cp_lower - Cp_upper
    q: np.ndarray  # (M,) load per unit chord in nondimensional units
    w: np.ndarray  # (M,) transverse displacement / c
    slope: np.ndarray  # (M,) dw/dx (radians)
    moment: np.ndarray  # (M,) bending moment, nondimensional
    stress: np.ndarray  # (M,) peak fibre stress / (0.5 rho U^2)
    w_max: float
    w_te: float  # trailing-edge deflection / c
    w_le: float
    twist_te_deg: float  # effective twist from the TE slope
    stress_max: float
    e_star: float
    spar_x: float

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.astype(float).tolist()
        return d


# --------------------------------------------------------------------- section


def section_properties(xy: np.ndarray, n: int = 121) -> SectionProperties:
    """Thickness and camber distributions plus area moments from a polygon.

    ``xy`` must be the *unrotated* closed section, chord along +x in [0, 1].
    """
    xy = np.asarray(xy, float)
    x = xy[:, 0]
    i_le = int(np.argmin(x))
    upper = xy[: i_le + 1][::-1]  # LE -> TE along the upper surface
    lower = xy[i_le:]  # LE -> TE along the lower surface

    xs = np.linspace(0.0, 1.0, n)
    yu = np.interp(xs, upper[:, 0], upper[:, 1])
    yl = np.interp(xs, lower[:, 0], lower[:, 1])
    if float(np.mean(yu - yl)) < 0.0:  # surfaces swapped
        yu, yl = yl, yu
    t = np.clip(yu - yl, 1e-4, None)
    camber = 0.5 * (yu + yl)

    # Shoelace area and second moment about the horizontal centroidal axis,
    # both from the strip decomposition (exact for the interpolated surfaces).
    dx = np.gradient(xs)
    area = float(np.sum(t * dx))
    y_bar = float(np.sum(camber * t * dx) / max(area, 1e-12))
    # I = integral over x of [ (y_u - y_bar)^3 - (y_l - y_bar)^3 ] / 3
    i_zz = float(np.sum(((yu - y_bar) ** 3 - (yl - y_bar) ** 3) / 3.0 * dx))

    k = int(np.argmax(t))
    return SectionProperties(
        x=xs,
        thickness=t,
        camber=camber,
        area=area,
        i_zz=abs(i_zz),
        max_thickness=float(t[k]),
        max_thickness_x=float(xs[k]),
    )


# ------------------------------------------------------------------ beam solve


def _hermite_beam_system(
    x: np.ndarray,
    ei: np.ndarray,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble K and F for a chain of 2-node cubic-Hermite beam elements.

    DOFs are ``[w_0, theta_0, w_1, theta_1, ...]``. ``ei`` and ``q`` are given at
    the nodes and averaged onto elements.
    """
    n_node = x.size
    n_el = n_node - 1
    ndof = 2 * n_node
    k_glob = np.zeros((ndof, ndof))
    f_glob = np.zeros(ndof)

    for e in range(n_el):
        le = float(x[e + 1] - x[e])
        if le <= 0:
            continue
        ei_e = 0.5 * float(ei[e] + ei[e + 1])
        q_e = 0.5 * float(q[e] + q[e + 1])
        c = ei_e / le ** 3
        ke = c * np.array([
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, 4.0 * le ** 2, -6.0 * le, 2.0 * le ** 2],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, 2.0 * le ** 2, -6.0 * le, 4.0 * le ** 2],
        ])
        fe = q_e * le * np.array([0.5, le / 12.0, 0.5, -le / 12.0])
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        k_glob[np.ix_(idx, idx)] += ke
        f_glob[idx] += fe

    return k_glob, f_glob


def solve_structure(
    section: SectionProperties,
    dcp: np.ndarray,
    e_star: float = DEFAULT_E_STAR,
    spar_x: float = DEFAULT_SPAR_X,
    q_dyn: float = 0.5,
) -> StructuralResult:
    """Static chordwise deflection under the net pressure load.

    Parameters
    ----------
    section
        From :func:`section_properties`, sampled on the same stations as ``dcp``.
    dcp
        Net pressure coefficient ``Cp_lower - Cp_upper`` at each station. A
        positive value pushes the section toward the suction side.
    e_star
        Dimensionless Young's modulus ``E / (0.5 rho U^2)``.
    spar_x
        Chord fraction of the clamped support.
    q_dyn
        Dynamic pressure in solver units (``0.5 * rho * U^2`` = 0.5 for the
        nondimensional flow solve).
    """
    x = section.x
    t = section.thickness
    dcp = np.asarray(dcp, float)
    if dcp.shape != x.shape:
        dcp = np.interp(x, np.linspace(0.0, 1.0, dcp.size), dcp)

    ei = e_star * t ** 3 / 12.0  # per unit span
    # A cusped trailing edge drives t -> 0 and EI -> 0, which turns the last
    # elements into hinges and the system into a near-singular one. Floor EI
    # relative to its own maximum: physically the TE is not a zero-stiffness
    # membrane, and numerically this is far safer than a ridge term on the
    # assembled matrix, which acts as a Winkler elastic foundation. (With this
    # beam's proportions a ridge of 1e-10 * trace was enough to stiffen the
    # cantilever by a factor of two -- the foundation's characteristic length
    # came out comparable to the chord.)
    ei = np.maximum(ei, 1e-6 * float(ei.max()))
    q = dcp * q_dyn  # load per unit chord per unit span

    k_glob, f_glob = _hermite_beam_system(x, ei, q)

    # Clamp at the spar: nearest node gets w = 0 and dw/dx = 0.
    i_spar = int(np.argmin(np.abs(x - spar_x)))
    fixed = [2 * i_spar, 2 * i_spar + 1]
    free = np.setdiff1d(np.arange(k_glob.shape[0]), fixed)

    u = np.zeros(k_glob.shape[0])
    k_ff = k_glob[np.ix_(free, free)]
    u[free] = np.linalg.solve(k_ff, f_glob[free])

    w = u[0::2]
    slope = u[1::2]

    # Curvature by central differences on the slope -> moment and fibre stress.
    curvature = np.gradient(slope, x)
    moment = ei * curvature
    stress = np.abs(moment) * (t / 2.0) / np.maximum(t ** 3 / 12.0, 1e-12)

    return StructuralResult(
        x=x,
        load=dcp,
        q=q,
        w=w,
        slope=slope,
        moment=moment,
        stress=stress,
        w_max=float(np.max(np.abs(w))),
        w_te=float(w[-1]),
        w_le=float(w[0]),
        twist_te_deg=float(np.rad2deg(slope[-1])),
        stress_max=float(np.max(stress)),
        e_star=float(e_star),
        spar_x=float(spar_x),
    )


# ----------------------------------------------------------------- modal model


def modal_properties(
    section: SectionProperties,
    chord_m: float = 0.1,
    span_m: float = 0.3,
    material: Material = ALUMINIUM,
    fluid_density: float = 998.0,
    u_inf_m_s: float = 5.0,
    damping_ratio: float = 0.01,
) -> dict:
    """First-bending-mode properties of the foil as a spanwise cantilever.

    Uses the standard Euler-Bernoulli cantilever eigenvalue
    ``beta_1 L = 1.8751`` and a strip-theory added-mass correction, which for a
    thin foil in water is the dominant effect on the wet natural frequency:
    ``m_added = pi/4 * rho_f * c^2`` per unit span.

    Returns the quantities the FSI feature record needs: natural frequency in
    air and in the working fluid, mass ratio, and reduced velocity.
    """
    beta_l = 1.8751
    # Dimensional section properties.
    area = section.area * chord_m ** 2
    i_zz = section.i_zz * chord_m ** 4
    mass_per_len = material.density_kg_m3 * area
    ei = material.flexural_modulus * i_zz

    f_dry = (beta_l ** 2 / (2.0 * np.pi * span_m ** 2)) * np.sqrt(
        ei / max(mass_per_len, 1e-12)
    )

    m_added = (np.pi / 4.0) * fluid_density * chord_m ** 2
    f_wet = f_dry * np.sqrt(mass_per_len / (mass_per_len + m_added))

    # fluid_density = 0 is a legitimate request (a vacuum/dry reference case),
    # so it must not divide by zero; the mass ratio is then unbounded.
    denom = fluid_density * chord_m ** 2
    mass_ratio = float("inf") if denom <= 0 else mass_per_len / denom
    reduced_velocity = u_inf_m_s / (max(f_wet, 1e-9) * chord_m)

    return {
        "material": material.name,
        "chord_m": float(chord_m),
        "span_m": float(span_m),
        "mass_per_length_kg_m": float(mass_per_len),
        "added_mass_kg_m": float(m_added),
        "flexural_rigidity_nm2": float(ei),
        "natural_frequency_dry_hz": float(f_dry),
        "natural_frequency_hz": float(f_wet),
        "mass_ratio": float(mass_ratio),
        "reduced_velocity": float(reduced_velocity),
        "damping_ratio": float(damping_ratio),
        "u_inf_m_s": float(u_inf_m_s),
        "fluid_density": float(fluid_density),
    }


# ------------------------------------------------------------ surface -> load


def surface_dcp(
    xc: np.ndarray,
    yc: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    cp: np.ndarray,
    polygon: np.ndarray,
    stations: np.ndarray,
) -> np.ndarray:
    """Project panel Cp onto chord stations and difference the two surfaces.

    Works on the *rotated* geometry: the chord line is recovered from the
    polygon's own extreme points, so no angle bookkeeping is needed.
    """
    x_le_i = int(np.argmin(polygon[:, 0]))
    x_te_i = int(np.argmax(polygon[:, 0]))
    le = polygon[x_le_i]
    te = polygon[x_te_i]
    chord_vec = te - le
    length = float(np.hypot(*chord_vec))
    if length < 1e-9:
        raise ValueError("degenerate chord")
    e_c = chord_vec / length
    e_n = np.array([-e_c[1], e_c[0]])  # chord normal, points to the +y side

    pts = np.column_stack([xc, yc]) - le
    s = (pts @ e_c) / length
    side = (np.column_stack([nx, ny]) @ e_n)  # >0 on the +n (upper) surface

    up = side > 0
    lo = ~up
    if up.sum() < 3 or lo.sum() < 3:
        return np.zeros_like(stations)

    def _interp(mask):
        order = np.argsort(s[mask])
        return np.interp(stations, np.clip(s[mask][order], 0.0, 1.0), cp[mask][order])

    cp_u = _interp(up)
    cp_l = _interp(lo)
    return cp_l - cp_u


def rasterize_structure(
    result: StructuralResult,
    polygon: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    solid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Paint ``load`` and ``displacement`` onto the solid pixels of the grid.

    Beam kinematics say both quantities are functions of the chord station
    alone, so every pixel inside the section takes the value at its own
    station. The fluid region stays zero, which makes the structural channels
    the exact complement of the flow channels -- the network never has to guess
    which region a channel lives in.
    """
    x_le_i = int(np.argmin(polygon[:, 0]))
    x_te_i = int(np.argmax(polygon[:, 0]))
    le = polygon[x_le_i]
    te = polygon[x_te_i]
    chord_vec = te - le
    length = float(np.hypot(*chord_vec))
    e_c = chord_vec / length

    pts = np.stack([grid_x - le[0], grid_y - le[1]], axis=-1)
    s = (pts @ e_c) / length
    s = np.clip(s, 0.0, 1.0)

    load = np.interp(s, result.x, result.load)
    disp = np.interp(s, result.x, result.w)
    return (load * solid_mask).astype(np.float32), (disp * solid_mask).astype(np.float32)
