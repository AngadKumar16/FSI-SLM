"""Rasterize an airfoil polygon into the U-Net input tensor.

Following Guo (2016), Bhatnagar (2019), Thuerey (2020): a signed distance field
(not a binary mask) is the geometry encoding -- smooth, encodes wall distance,
well-conditioned gradient everywhere. Sign convention: negative inside the foil,
positive outside.

Two things this module does that the cited work does not.

**A physics prior.** Channels ``pot_u``, ``pot_v``, ``pot_p`` carry the
*inviscid* solution on the same raster. The network then has to learn the
viscous and rotational correction to potential flow rather than the whole field
from scratch: the stagnation point, the circulation, the suction-peak location
and the entire far field are handed to it, already exact. The prior costs one
panel solve (~30 ms) and it is available for a RANS row too -- the panel
geometry does not care which solver produced the target -- so panel-generated
and RANS-generated rows stay interchangeable.

**An explicit channel contract.** :class:`InputSpec` names every channel and
records its parity under a mirror about ``y = 0``, which is what makes the
symmetry augmentation in :mod:`sensor.symmetry` checkable instead of a pile of
sign conventions in a loop body. Checkpoints store the spec they were trained
with, so an old four-channel model still loads and still runs.

Channel sets
------------

``v1`` (4 channels, the original contract)
    ``sdf, fs_x, fs_y, log_re``

``v2`` (13 channels, the default)
    ``sdf, fs_x, fs_y, log_re, x, y, n_x, n_y, wall, fidelity, pot_u, pot_v,
    pot_p``

``v2_noprior`` (10 channels)
    ``v2`` without the three prior channels -- for ablating what the prior is
    actually worth.

Constant channels (``fs_x``, ``fs_y``, ``log_re``, ``fidelity``) are kept as
full maps so the on-disk tensor stays one array, but they are also flagged
``constant`` so the network can read them back as a global conditioning vector
(:mod:`sensor.model` FiLM) instead of spending convolution capacity on
reproducing a constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .geometry import Airfoil, rotate

# Width of the near-wall emphasis channel, in chords. About two grid cells at
# the default 42 px/chord raster.
WALL_BAND = 0.05

# Numeric code per fidelity, broadcast into the ``fidelity`` channel. This is
# what makes one network able to emit either solver's answer: the code is an
# input, so it can be *asked* for RANS at inference time.
FIDELITY_CODES: Dict[str, float] = {"panel": 0.0, "rans": 1.0, "urans": 2.0}


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

    ``ylim`` is symmetric about zero on purpose: the mirror augmentation is an
    exact operation only if flipping the row index maps the grid onto itself.

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

    @property
    def y_symmetric(self) -> bool:
        """True if a row flip is an exact mirror about ``y = 0``."""
        return abs(self.ylim[0] + self.ylim[1]) < 1e-9

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(*self.xlim, self.nx)
        ys = np.linspace(*self.ylim, self.ny)
        return np.meshgrid(xs, ys)  # (ny, nx)


# --------------------------------------------------------------- channel spec


@dataclass(frozen=True)
class Channel:
    """One input channel: what it is, and how it behaves under a y-mirror."""

    name: str
    parity: int  # +1 keeps its value under the mirror, -1 negates it
    constant: bool = False  # spatially uniform => readable as a global scalar
    needs_prior: bool = False  # requires the inviscid panel raster


_CHANNELS: Dict[str, Channel] = {
    c.name: c
    for c in (
        Channel("sdf", +1),
        Channel("fs_x", +1, constant=True),
        Channel("fs_y", -1, constant=True),
        Channel("log_re", +1, constant=True),
        Channel("x", +1),
        Channel("y", -1),
        Channel("n_x", +1),
        Channel("n_y", -1),
        Channel("wall", +1),
        Channel("fidelity", +1, constant=True),
        Channel("pot_u", +1, needs_prior=True),
        Channel("pot_v", -1, needs_prior=True),
        Channel("pot_p", +1, needs_prior=True),
    )
}


@dataclass(frozen=True)
class InputSpec:
    """An ordered, named set of input channels."""

    name: str
    names: Tuple[str, ...]

    def __post_init__(self):
        unknown = [n for n in self.names if n not in _CHANNELS]
        if unknown:
            raise ValueError(f"unknown input channels: {unknown}")

    @property
    def n_channels(self) -> int:
        return len(self.names)

    @property
    def parity(self) -> np.ndarray:
        return np.array([_CHANNELS[n].parity for n in self.names], dtype=np.float32)

    @property
    def constant_idx(self) -> Tuple[int, ...]:
        return tuple(i for i, n in enumerate(self.names) if _CHANNELS[n].constant)

    @property
    def needs_prior(self) -> bool:
        return any(_CHANNELS[n].needs_prior for n in self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def to_dict(self) -> dict:
        return {"name": self.name, "names": list(self.names)}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "InputSpec":
        """Rebuild a spec from a checkpoint, defaulting to the old contract.

        A checkpoint written before this module existed has no spec at all; it
        was necessarily trained on the four-channel tensor, so that is the right
        default rather than an error.
        """
        if not d:
            return SPEC_V1
        if isinstance(d, str):
            return SPECS[d]
        return cls(name=d.get("name", "custom"), names=tuple(d["names"]))


SPEC_V1 = InputSpec("v1", ("sdf", "fs_x", "fs_y", "log_re"))
SPEC_V2 = InputSpec(
    "v2",
    ("sdf", "fs_x", "fs_y", "log_re", "x", "y", "n_x", "n_y", "wall",
     "fidelity", "pot_u", "pot_v", "pot_p"),
)
SPEC_V2_NOPRIOR = InputSpec(
    "v2_noprior",
    ("sdf", "fs_x", "fs_y", "log_re", "x", "y", "n_x", "n_y", "wall", "fidelity"),
)
SPECS: Dict[str, InputSpec] = {s.name: s for s in (SPEC_V1, SPEC_V2, SPEC_V2_NOPRIOR)}
DEFAULT_SPEC = SPEC_V2

# Kept for callers that import the old name.
INPUT_CHANNELS = SPEC_V1.names


# ------------------------------------------------------------------ distances


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
    """``(sdf, nearest_edge_index)``, both shaped ``(ny, nx)``."""
    X, Y = grid.meshgrid()
    px, py = X.ravel(), Y.ravel()
    d, idx = _dist_to_poly(px, py, poly, want_index=True)
    inside = _point_in_poly(px, py, poly)
    sdf = np.where(inside, -d, d)
    return sdf.reshape(X.shape), idx.reshape(X.shape)


# --------------------------------------------------------------- input tensor


def surface_normals(sdf: np.ndarray, grid: Grid) -> Tuple[np.ndarray, np.ndarray]:
    """``grad(sdf)``, which is the outward unit normal wherever the SDF is exact.

    A convolution can of course differentiate its own input, but it has to spend
    a 3x3 kernel and a nonlinearity doing it in every case, and near the wall --
    where the field is one pixel wide and the load lives -- it is exactly the
    quantity that decides whether a pixel is above or below the surface.
    """
    gy, gx = np.gradient(sdf, grid.dy, grid.dx)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-8
    return gx / mag, gy / mag


def build_input(
    af: Airfoil,
    alpha_deg: float,
    reynolds: float,
    grid: Optional[Grid] = None,
    spec: Optional[InputSpec] = None,
    sdf: Optional[np.ndarray] = None,
    prior: Optional[Dict[str, np.ndarray]] = None,
    fidelity: str | float = "panel",
) -> np.ndarray:
    """Build the ``(C, ny, nx)`` input tensor for one case.

    Pass ``sdf`` to reuse a field the caller already computed -- the distance
    transform is the most expensive part of a case and the solver needs the same
    one. Pass ``prior`` (keys ``u``/``v``/``p``) to reuse the inviscid raster;
    if the spec needs it and it was not supplied it is computed here, which
    costs one panel solve.
    """
    grid = grid or Grid()
    spec = spec or DEFAULT_SPEC
    if sdf is None:
        sdf = signed_distance(rotate(af, alpha_deg).xy, grid)
    sdf = np.asarray(sdf, dtype=np.float32)

    if spec.needs_prior and prior is None:
        from .flow.solve import potential_prior

        prior = potential_prior(af, alpha_deg, grid, sdf=None)

    a = np.deg2rad(alpha_deg)
    ones = np.ones_like(sdf)
    code = (
        FIDELITY_CODES.get(fidelity, 0.0) if isinstance(fidelity, str)
        else float(fidelity)
    )

    built: Dict[str, np.ndarray] = {}

    def need(name: str) -> bool:
        return name in spec.names

    built["sdf"] = sdf
    if need("fs_x"):
        built["fs_x"] = ones * np.cos(a)
    if need("fs_y"):
        built["fs_y"] = ones * np.sin(a)
    if need("log_re"):
        # ~0 at Re = 1e5, so the channel is centred over the range we sweep.
        built["log_re"] = ones * (np.log10(max(reynolds, 1.0)) - 5.0)
    if need("fidelity"):
        built["fidelity"] = ones * code
    if need("x") or need("y"):
        X, Y = grid.meshgrid()
        built["x"] = X.astype(np.float32)
        built["y"] = Y.astype(np.float32)
    if need("n_x") or need("n_y"):
        nx_, ny_ = surface_normals(sdf, grid)
        built["n_x"] = nx_.astype(np.float32)
        built["n_y"] = ny_.astype(np.float32)
    if need("wall"):
        built["wall"] = np.exp(-np.clip(sdf, 0.0, None) / WALL_BAND).astype(np.float32)
    if spec.needs_prior:
        built["pot_u"] = np.asarray(prior["u"], dtype=np.float32)
        built["pot_v"] = np.asarray(prior["v"], dtype=np.float32)
        built["pot_p"] = np.asarray(prior["p"], dtype=np.float32)

    return np.stack([built[n] for n in spec.names]).astype(np.float32)


def input_tensor(
    af: Airfoil,
    alpha_deg: float,
    reynolds: float,
    grid: Grid | None = None,
    sdf: np.ndarray | None = None,
) -> np.ndarray:
    """The original four-channel tensor. Kept so ``v1`` checkpoints still run."""
    return build_input(af, alpha_deg, reynolds, grid, SPEC_V1, sdf=sdf)
