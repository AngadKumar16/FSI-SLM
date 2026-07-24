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
    nx: int = 256
    ny: int = 256
    xlim: Tuple[float, float] = (-5.0, 15.0)  # chords: upstream, downstream
    ylim: Tuple[float, float] = (-8.0, 8.0)

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(*self.xlim, self.nx)
        ys = np.linspace(*self.ylim, self.ny)
        return np.meshgrid(xs, ys)  # (ny, nx)


def _point_in_poly(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized even-odd ray casting. px,py shape (M,); poly (K,2) closed."""
    x1 = poly[:-1, 0][:, None]
    y1 = poly[:-1, 1][:, None]
    x2 = poly[1:, 0][:, None]
    y2 = poly[1:, 1][:, None]
    cond = ((y1 > py) != (y2 > py))
    xint = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-30) + x1
    inside = np.sum(cond & (px < xint), axis=0) % 2 == 1
    return inside


def _dist_to_poly(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Min distance from each (px,py) to polygon edges. Vectorized over edges."""
    a = poly[:-1]  # (K,2)
    b = poly[1:]
    ab = b - a  # (K,2)
    ab2 = np.sum(ab * ab, axis=1) + 1e-30  # (K,)
    P = np.stack([px, py], axis=1)  # (M,2)
    # t = clamp( (P-a).ab / ab2 )  -> (M,K)
    ap = P[:, None, :] - a[None, :, :]  # (M,K,2)
    t = np.clip(np.sum(ap * ab[None], axis=2) / ab2[None], 0.0, 1.0)
    proj = a[None] + t[:, :, None] * ab[None]  # (M,K,2)
    d = np.linalg.norm(P[:, None, :] - proj, axis=2)  # (M,K)
    return d.min(axis=1)


def signed_distance(poly: np.ndarray, grid: Grid) -> np.ndarray:
    X, Y = grid.meshgrid()
    px, py = X.ravel(), Y.ravel()
    d = _dist_to_poly(px, py, poly)
    inside = _point_in_poly(px, py, poly)
    sdf = np.where(inside, -d, d)
    return sdf.reshape(X.shape)


def input_tensor(
    af: Airfoil,
    alpha_deg: float,
    reynolds: float,
    grid: Grid | None = None,
) -> np.ndarray:
    """Build (4, ny, nx) UNet input for one (shape, AoA, Re) case."""
    grid = grid or Grid()
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
