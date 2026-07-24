"""Interpolate an OpenFOAM solution onto the fixed Cartesian grid used by the
UNet, and pack (input, target) into a ``.npz``.

The sampling grid is exactly :class:`sensor.sdf.Grid`, so the SDF input channel
and the CFD target fields are pixel-aligned. Cells inside the foil (SDF < 0)
are masked to zero (there is no fluid there). Targets are stored
nondimensional: velocity in units of Uinf (= 1 in the case setup) and pressure
as the kinematic pressure p (units Uinf^2), matching ``simpleFoam`` output.

Target channels: [u, v, p]. A binary fluid ``mask`` is saved alongside so the
training loss can ignore the interior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry import Airfoil
from ..sdf import Grid, input_tensor


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


def sample_case(
    case_dir: str | Path,
    af: Airfoil,
    alpha_deg: float,
    reynolds: float,
    grid: Grid | None = None,
) -> dict:
    import pyvista as pv

    grid = grid or Grid()
    case_dir = Path(case_dir)
    internal = _read_latest(case_dir)

    # target Cartesian grid as an ImageData at z = mid-span of the 2D case
    zc = float(internal.points[:, 2].mean())
    dx = (grid.xlim[1] - grid.xlim[0]) / (grid.nx - 1)
    dy = (grid.ylim[1] - grid.ylim[0]) / (grid.ny - 1)
    img = pv.ImageData(
        dimensions=(grid.nx, grid.ny, 1),
        spacing=(dx, dy, 1.0),
        origin=(grid.xlim[0], grid.ylim[0], zc),
    )
    sampled = img.sample(internal)  # interpolate CFD onto the grid points

    U = np.asarray(sampled["U"]).reshape(grid.ny, grid.nx, 3)
    p = np.asarray(sampled["p"]).reshape(grid.ny, grid.nx)
    valid = np.asarray(sampled["vtkValidPointMask"]).reshape(grid.ny, grid.nx).astype(bool)

    inp = input_tensor(af, alpha_deg, reynolds, grid)   # (4, ny, nx); ch0 = SDF
    fluid = (inp[0] > 0) & valid                          # outside foil & in domain
    u = np.where(fluid, U[..., 0], 0.0)
    v = np.where(fluid, U[..., 1], 0.0)
    pp = np.where(fluid, p, 0.0)
    target = np.stack([u, v, pp]).astype(np.float32)

    return {
        "input": inp,                       # (4, ny, nx)
        "target": target,                   # (3, ny, nx)
        "mask": fluid.astype(np.float32),   # (ny, nx)
        "alpha_deg": np.float32(alpha_deg),
        "reynolds": np.float32(reynolds),
    }


def save_npz(out: str | Path, sample: dict) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **sample)
    return out
