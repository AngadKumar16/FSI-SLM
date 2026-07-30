"""Turn a solved flow field from *any* backend into the training contract.

The panel backend (:mod:`sensor.simulate`) and the OpenFOAM backend both have to
emit byte-identical ``.npz`` structure, or the dataset cannot mix fidelities.
Rather than duplicate the packing logic in two places and let it drift, both
funnel through :func:`build_case_arrays` here.

What the caller has to supply is only what a flow solver actually produces:
``u``, ``v``, ``p`` sampled on :class:`sensor.sdf.Grid`, and optionally the
integrated force coefficients. Everything else -- the SDF, the masks, the
surface pressure distribution, the structural solve, the rasterised load and
displacement channels, the modal properties, the metadata block -- is derived
here, identically for both backends.

Surface pressure has two routes:

``surface_cp_from_case``
    Read the ``airfoil`` patch out of the OpenFOAM solution with pyvista. Exact,
    but needs the case directory and the optional dependency.
``surface_cp_from_field``
    Probe the rasterised pressure a couple of cells off each panel along its
    outward normal. Backend-agnostic, needs nothing but the arrays, and is what
    keeps this module testable with no solver installed. Resolution-limited at
    42 px/chord, so it is the fallback rather than the default.

The structural side is deliberately *not* re-derived per backend: the same
chordwise beam runs on whatever surface pressure it is handed, so a RANS row and
a panel row differ only in where their pressure came from.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..flow.panel import PanelSolver, resample_polygon
from ..flow.solve import N_PANELS
from ..geometry import Airfoil, close_trailing_edge, rotate
from ..sdf import Grid, input_tensor, signed_distance_and_edge
from ..structure.beam import (
    ALUMINIUM,
    DEFAULT_E_STAR,
    DEFAULT_SPAR_X,
    Material,
    modal_properties,
    rasterize_structure,
    section_properties,
    solve_structure,
    surface_dcp,
)

# Probe depth for the raster fallback, in grid cells. Two and a half cells is
# outside the wall layer the panel backend paints on and still inside the region
# where the pressure has not yet relaxed toward freestream.
PROBE_CELLS = 2.5


def surface_cp_from_field(
    p: np.ndarray,
    grid: Grid,
    solver: PanelSolver,
    fluid: np.ndarray,
    probe_cells: float = PROBE_CELLS,
) -> np.ndarray:
    """Cp at each panel control point, probed from the rasterised pressure.

    ``p`` is kinematic pressure in units of ``U^2``, so ``Cp = 2 p``.
    """
    d = probe_cells * max(grid.dx, grid.dy)
    px = solver.xc + d * solver.nx
    py = solver.yc + d * solver.ny

    j = np.clip(np.round((px - grid.xlim[0]) / grid.dx).astype(int), 0, grid.nx - 1)
    i = np.clip(np.round((py - grid.ylim[0]) / grid.dy).astype(int), 0, grid.ny - 1)

    cp = 2.0 * p[i, j]
    # Panels whose probe landed inside the body or outside the valid region get
    # the mean of their neighbours rather than a zero, which would read as a
    # stagnation-free patch and corrupt the load distribution.
    bad = ~fluid[i, j].astype(bool)
    if bad.any() and (~bad).any():
        idx = np.arange(cp.size)
        cp[bad] = np.interp(idx[bad], idx[~bad], cp[~bad])
    return cp


def surface_cp_from_case(case_dir, solver: PanelSolver) -> Optional[np.ndarray]:
    """Cp on the ``airfoil`` patch of an OpenFOAM case, or ``None`` if unavailable.

    Returns values interpolated onto the panel control points so the structural
    solve sees the same abscissa either way.
    """
    try:
        from pathlib import Path

        import pyvista as pv

        case_dir = Path(case_dir)
        foam = case_dir / "case.foam"
        foam.touch()
        reader = pv.OpenFOAMReader(str(foam))
        reader.set_active_time_value(reader.time_values[-1])
        reader.cell_to_point_creation = True
        mesh = reader.read()
        patches = mesh["boundary"] if "boundary" in mesh.keys() else None
        if patches is None:
            return None
        block = None
        for name in patches.keys():
            if "airfoil" in str(name).lower():
                block = patches[name]
                break
        if block is None:
            return None
        pts = np.asarray(block.points)[:, :2]
        pvals = np.asarray(block["p"]).ravel()
    except Exception:
        return None

    # nearest control point for each patch node, then average per panel
    d2 = (
        (pts[:, None, 0] - solver.xc[None, :]) ** 2
        + (pts[:, None, 1] - solver.yc[None, :]) ** 2
    )
    owner = np.argmin(d2, axis=1)
    cp = np.zeros(solver.n)
    count = np.zeros(solver.n)
    np.add.at(cp, owner, 2.0 * pvals)
    np.add.at(count, owner, 1.0)
    seen = count > 0
    cp[seen] /= count[seen]
    if not seen.any():
        return None
    if (~seen).any():
        idx = np.arange(solver.n)
        cp[~seen] = np.interp(idx[~seen], idx[seen], cp[seen])
    return cp


def integrate_coefficients(solver: PanelSolver, cp: np.ndarray) -> Tuple[float, float, float]:
    """Cl, pressure-Cd and quarter-chord Cm from a surface Cp distribution.

    Used only when the backend did not report its own force coefficients.
    OpenFOAM's ``forceCoeffs`` function object is preferred because it includes
    the viscous contribution, which a pressure integral cannot see.
    """
    dfx = -cp * solver.nx * solver.s
    dfy = -cp * solver.ny * solver.s
    cl = float(dfy.sum())
    cd = float(dfx.sum())
    cm = float(np.sum((solver.xc - 0.25) * dfy - solver.yc * dfx))
    return cl, cd, cm


def build_case_arrays(
    af: Airfoil,
    alpha_deg: float,
    reynolds: float,
    grid: Grid,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    valid: Optional[np.ndarray] = None,
    surface_cp: Optional[np.ndarray] = None,
    case_dir=None,
    cl: Optional[float] = None,
    cd: Optional[float] = None,
    cm: Optional[float] = None,
    fidelity: str = "rans",
    e_star: float = DEFAULT_E_STAR,
    spar_x: float = DEFAULT_SPAR_X,
    n_panels: int = N_PANELS,
    material: Material = ALUMINIUM,
    chord_m: float = 0.1,
    span_m: float = 0.3,
    u_inf_m_s: float = 5.0,
    fluid_density: float = 998.0,
    with_modal: bool = True,
) -> dict:
    """Assemble the full contract from solved fields.

    Output keys and shapes match :func:`sensor.simulate.simulate_case` exactly,
    so :func:`sensor.simulate.pack_npz` writes an identical file layout.
    """
    af = close_trailing_edge(af)
    section = section_properties(af.xy)
    poly = resample_polygon(rotate(af, alpha_deg).xy, n_panels)
    solver = PanelSolver(poly)
    # Solved only to populate the panel geometry (control points, normals,
    # lengths) that the surface probe and the Cp integration index into. The
    # inviscid velocities are discarded -- the fields come from the caller.
    solver.solve(0.0)

    sdf, _edge = signed_distance_and_edge(solver.xy, grid)
    fluid = sdf > 0.0
    if valid is not None:
        fluid = fluid & np.asarray(valid, dtype=bool)
    solid = ~fluid & (sdf <= 0.0)

    u = np.where(fluid, np.nan_to_num(u), 0.0).astype(np.float32)
    v = np.where(fluid, np.nan_to_num(v), 0.0).astype(np.float32)
    p = np.where(fluid, np.nan_to_num(p), 0.0).astype(np.float32)

    if surface_cp is None and case_dir is not None:
        surface_cp = surface_cp_from_case(case_dir, solver)
    if surface_cp is None:
        surface_cp = surface_cp_from_field(p, grid, solver, fluid)
    surface_cp = np.asarray(surface_cp, dtype=float)

    dcp = surface_dcp(
        solver.xc, solver.yc, solver.nx, solver.ny, surface_cp, solver.xy, section.x
    )
    struct = solve_structure(section, dcp, e_star=e_star, spar_x=spar_x)

    xg, yg = grid.meshgrid()
    solid_f = solid.astype(np.float32)
    load_field, disp_field = rasterize_structure(struct, solver.xy, xg, yg, solid_f)

    cl_i, cd_i, cm_i = integrate_coefficients(solver, surface_cp)
    coeffs = {
        "cl": float(cl if cl is not None else cl_i),
        "cd": float(cd if cd is not None else max(cd_i, 0.0)),
        "cm": float(cm if cm is not None else cm_i),
        "cl_pressure_integral": cl_i,
        "cd_pressure_integral": cd_i,
        "w_max": struct.w_max,
        "w_te": struct.w_te,
        "twist_te_deg": struct.twist_te_deg,
        "stress_max": struct.stress_max,
        "e_star": struct.e_star,
        "spar_x": struct.spar_x,
        "max_thickness": section.max_thickness,
        "max_thickness_x": section.max_thickness_x,
        "section_area": section.area,
        "section_i_zz": section.i_zz,
    }

    out = {
        "input": input_tensor(af, alpha_deg, reynolds, grid, sdf=sdf),
        "target": np.stack([u, v, p, load_field, disp_field]).astype(np.float32),
        "mask": fluid.astype(np.float32),
        "solid_mask": solid_f,
        "alpha_deg": np.float32(alpha_deg),
        "reynolds": np.float32(reynolds),
        "coeffs": coeffs,
        "structure": struct,
        "section": section,
        "polygon": solver.xy,
        "surface_dcp": dcp,
        "surface_cp": surface_cp,
        "fidelity": fidelity,
        # The panel backend attaches its viscous state here; a RANS row has no
        # equivalent scalar model, and downstream code treats it as optional.
        "state": None,
    }
    if with_modal:
        out["modal"] = modal_properties(
            section, chord_m=chord_m, span_m=span_m, material=material,
            fluid_density=fluid_density, u_inf_m_s=u_inf_m_s,
        )
    return out
