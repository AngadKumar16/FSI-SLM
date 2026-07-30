"""Rasterize an airfoil polygon to a signed-distance field + flow-condition
channels -> the UNet input tensor.

Following Guo (2016), Bhatnagar (2019), Thuerey (2020): SDF (not a binary mask)
is the geometry encoding -- it is smooth, encodes wall distance, and gives the
network a well-conditioned gradient everywhere. Sign convention: negative
inside the foil, positive outside.

Input tensor channels (C, H, W), all nondimensional:
  0: SDF (chord units)
  1: freestream x-component (cos alpha), broadcast constant
  2: freestream y-component (sin alpha), broadcast constant
  3: log10(Re) scaled, broadcast constant   [reserved flow scalar]

Channels 1-3 are broadcast constants here but kept as full maps so a later
2-way / transient extension can make them spatially varying (e.g. moving-wall
BC, time index) without changing the tensor contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .geometry import Airfoil, rotate


@dataclass
class Grid:
    """Fixed Cartesian sampling window, in chord units, shared by every backend.

    The window is deliberately *not* the CFD far field. A solver needs a domain
    tens of chords across so the boundary conditions do not contaminate the
    solution, but a convolutional network needs the body to occupy a decent
    share of the image: on a 128 x 128 raster spanning 20 chords the foil is six
    pixels wide and the surface layer is unresolvable. The default here spans 3
    chords, putting the section at ~42 px and keeping 1.4 chords of wake, which
    matches the framing used in the airfoil-surrogate literature (Thuerey et al.
    2020 sample [-0.5, 1.5] x [-1, 1] at 128^2).

    The OpenFOAM path still solves on its own large mesh and only *samples* onto
    this window, so both backends stay pixel-aligned.
    """

    nx: int = 128
    ny: int = 128
    xlim: Tuple[float, float] = (-0.6, 2.4)  # chords: upstream, downstream
    ylim: Tuple[float, float] = (-1.5, 1.5)

    @property
    def dx(self) -> float:
        return (self.xlim[1] - self.xlim[0]) / (self.nx - 1)

    @property
    def dy(self) -> float:
        return (self.ylim[1] - self.ylim[0]) / (self.ny - 1)

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(*self.xlim, self.nx)
        ys = np.linspace(*self.ylim, self.ny)
        return np.meshgrid(xs, ys)  # (ny, nx)


def _point_in_poly(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd ray casting, looped over edges.

    Looping over the ~180 edges and vectorising over the ~16k points keeps the
    working set at O(M) instead of the O(M*K) broadcast, which for a 128^2 grid
    is the difference between a 130 kB temporary and a 24 MB one.
    """
    inside = np.zeros(px.size, dtype=bool)
    x1, y1 = poly[:-1, 0], poly[:-1, 1]
    x2, y2 = poly[1:, 0], poly[1:, 1]
    for i in range(x1.size):
        cond = (y1[i] > py) != (y2[i] > py)
        if not cond.any():
            continue
        xint = (x2[i] - x1[i]) * (py - y1[i]) / (y2[i] - y1[i] + 1e-30) + x1[i]
        inside ^= cond & (px < xint)
    return inside


def _dist_to_poly(
    px: np.ndarray,
    py: np.ndarray,
    poly: np.ndarray,
    want_index: bool = False,
):
    """Min distance from each point to the polygon, edge-by-edge.

    Returns ``d`` or ``(d, edge_index)``. The edge index is what the near-wall
    model needs to look up the local surface tangential velocity.
    """
    a = poly[:-1]
    b = poly[1:]
    best = np.full(px.size, np.inf)
    idx = np.zeros(px.size, dtype=np.int32) if want_index else None
    for i in range(a.shape[0]):
        ax, ay = a[i]
        abx, aby = b[i, 0] - ax, b[i, 1] - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-30:
            continue
        t = ((px - ax) * abx + (py - ay) * aby) / ab2
        np.clip(t, 0.0, 1.0, out=t)
        dx = px - (ax + t * abx)
        dy = py - (ay + t * aby)
        d2 = dx * dx + dy * dy
        if want_index:
            closer = d2 < best
            idx[closer] = i
            np.minimum(best, d2, out=best)
        else:
            np.minimum(best, d2, out=best)
    d = np.sqrt(best)
    return (d, idx) if want_index else d


def signed_distance(poly: np.ndarray, grid: Grid) -> np.ndarray:
    X, Y = grid.meshgrid()
    px, py = X.ravel(), Y.ravel()
    d = _dist_to_poly(px, py, poly)
    inside = _point_in_poly(px, py, poly)
    sdf = np.where(inside, -d, d)
    return sdf.reshape(X.shape)


def signed_distance_and_edge(poly: np.ndarray, grid: Grid):
    """``(sdf, nearest_edge_index)`` on the grid, both shaped ``(ny, nx)``."""
    X, Y = grid.meshgrid()
    px, py = X.ravel(), Y.ravel()
    d, idx = _dist_to_poly(px, py, poly, want_index=True)
    inside = _point_in_poly(px, py, poly)
    sdf = np.where(inside, -d, d)
    return sdf.reshape(X.shape), idx.reshape(X.shape)


def input_tensor(
    af: Airfoil,
    alpha_deg: float,
    reynolds: float,
    grid: Grid | None = None,
    sdf: np.ndarray | None = None,
) -> np.ndarray:
    """Build (4, ny, nx) UNet input for one (shape, AoA, Re) case.

    Pass ``sdf`` to reuse a field the caller already computed -- the distance
    transform is the most expensive part of a case, and the solver needs the
    same one.
    """
    grid = grid or Grid()
    if sdf is None:
        rot = rotate(af, alpha_deg)
        sdf = signed_distance(rot.xy, grid)
    a = np.deg2rad(alpha_deg)
    ones = np.ones_like(sdf)
    re_scaled = (np.log10(max(reynolds, 1.0)) - 5.0)  # ~0 at Re=1e5
    return np.stack([
        sdf,
        ones * np.cos(a),
        ones * np.sin(a),
        ones * re_scaled,
    ]).astype(np.float32)
