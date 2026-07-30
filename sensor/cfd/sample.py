"""Interpolate an OpenFOAM solution onto the U-Net grid and pack the contract.

The sampling grid is exactly :class:`sensor.sdf.Grid`, so the SDF input channel
and the CFD target fields are pixel-aligned. Cells inside the foil are masked.
Targets are nondimensional: velocity in units of Uinf (= 1 in the case setup)
and pressure as the kinematic pressure p (units Uinf^2), matching ``simpleFoam``
output.

Only the *reading* of the OpenFOAM solution lives here. Everything after that --
masks, surface pressure, the structural solve, the load and displacement
channels, the metadata -- goes through :mod:`sensor.cfd.adapt`, shared with the
panel backend, so the two backends cannot produce different file layouts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..geometry import Airfoil
from ..sdf import Grid
from ..simulate import pack_npz
from .adapt import build_case_arrays


def _read_latest(case_dir: Path):
    import pyvista as pv

    foam = case_dir / "case.foam"
    foam.touch()
    reader = pv.OpenFOAMReader(str(foam))
    reader.set_active_time_value(reader.time_values[-1])
    reader.cell_to_point_creation = True
    mesh = reader.read()
    internal = mesh["internalMesh"] if "internalMesh" in mesh.keys() else mesh[0]
    return internal


def read_fields(case_dir: str | Path, grid: Grid):
    """Sample ``U`` and ``p`` from an OpenFOAM case onto the Cartesian grid.

    Returns ``(u, v, p, valid)``; ``valid`` marks grid points that fell inside
    the CFD domain.
    """
    import pyvista as pv

    case_dir = Path(case_dir)
    internal = _read_latest(case_dir)

    zc = float(internal.points[:, 2].mean())
    dx = (grid.xlim[1] - grid.xlim[0]) / (grid.nx - 1)
    dy = (grid.ylim[1] - grid.ylim[0]) / (grid.ny - 1)
    img = pv.ImageData(
        dimensions=(grid.nx, grid.ny, 1),
        spacing=(dx, dy, 1.0),
        origin=(grid.xlim[0], grid.ylim[0], zc),
    )
    sampled = img.sample(internal)

    vel = np.asarray(sampled["U"]).reshape(grid.ny, grid.nx, 3)
    p = np.asarray(sampled["p"]).reshape(grid.ny, grid.nx)
    valid = np.asarray(sampled["vtkValidPointMask"]).reshape(grid.ny, grid.nx).astype(bool)
    return vel[..., 0], vel[..., 1], p, valid


def sample_case(
    case_dir: str | Path,
    af: Airfoil,
    alpha_deg: float,
    reynolds: float,
    grid: Optional[Grid] = None,
    cl: Optional[float] = None,
    cd: Optional[float] = None,
    cm: Optional[float] = None,
    **kwargs,
) -> dict:
    """Full RANS case in the training contract.

    ``cl``/``cd``/``cm`` should come from OpenFOAM's ``forceCoeffs`` function
    object (see :func:`sensor.cfd.run.read_force_coeffs`); they include the
    viscous contribution that a surface-pressure integral cannot see. Omit them
    and the pressure integral is used instead, which underestimates drag.
    """
    grid = grid or Grid()
    u, v, p, valid = read_fields(case_dir, grid)
    return build_case_arrays(
        af, alpha_deg, reynolds, grid, u, v, p,
        valid=valid, case_dir=case_dir, cl=cl, cd=cd, cm=cm,
        fidelity="rans", **kwargs,
    )


def save_npz(out: str | Path, sample: dict) -> Path:
    """Write the packed contract. Identical layout to the panel backend."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **pack_npz(sample))
    return out
