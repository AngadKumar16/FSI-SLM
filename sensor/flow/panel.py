"""Hess-Smith panel method: constant-strength sources per panel + one constant
vortex over the whole body, closed with an explicit Kutta condition.

Unknowns are ``N`` source strengths ``sigma_i`` plus a single circulation
density ``gamma``. Flow tangency is enforced at every panel control point and
the Kutta condition sets equal-and-opposite tangential velocity on the two
trailing-edge panels. The resulting linear system is (N+1) x (N+1) and dense;
for N ~ 160 panels this solves in well under a millisecond.

Conventions
-----------
* The polygon is the *already rotated* airfoil (see :func:`sensor.geometry.rotate`),
  so the freestream is always horizontal, left to right, and every field this
  module returns is pixel-aligned with :class:`sensor.sdf.Grid`.
* Panels are ordered clockwise so that panel normals ``(-sin t, cos t)`` point
  into the fluid.
* All output is nondimensional: velocities in units of ``Uinf``, pressure as the
  kinematic coefficient ``p = (1 - |V|^2) / 2`` (i.e. ``Cp / 2``), which is what
  ``simpleFoam`` writes for an incompressible case and what
  :mod:`sensor.cfd.sample` stores. Chord is 1.

Validation (see ``tests/test_panel.py``): lift slope matches thin-airfoil theory
``dCl/dalpha = 2*pi`` to better than 2%, the stagnation point reaches ``Cp = 1``,
and inviscid drag satisfies d'Alembert (``|Cd| < 1e-3``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# Chunk size for field evaluation: influence matrices are (M, N) float64, so
# 4096 points x 200 panels x 8 B x ~8 temporaries ~ 50 MB peak. Keeps the
# generator inside a 3 GB container.
_CHUNK = 4096


@dataclass
class PanelSolution:
    """Converged inviscid solution for one (geometry, alpha) pair."""

    sigma: np.ndarray  # (N,) source strengths
    gamma: float  # constant vortex density
    cp: np.ndarray  # (N,) surface pressure coefficient at control points
    vt: np.ndarray  # (N,) surface tangential velocity / Uinf
    cl: float  # from Kutta-Joukowski
    cl_pressure: float  # from Cp integration (consistency check)
    cd_pressure: float  # ~0 by d'Alembert; nonzero => panel resolution issue
    cm_quarter: float  # moment about (0.25, 0), nose-up positive
    circulation: float  # total bound circulation
    alpha_deg: float
    uinf: float

    @property
    def dalembert_residual(self) -> float:
        """Inviscid drag that *should* be zero. Use as a solver health check."""
        return float(abs(self.cd_pressure))


class PanelSolver:
    """Geometry-bound solver. Build once per airfoil+alpha, then query fields.

    Parameters
    ----------
    xy
        Closed polygon ``(N+1, 2)``, first point == last point.
    """

    def __init__(self, xy: np.ndarray):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"expected (N+1, 2) polygon, got {xy.shape}")
        if not np.allclose(xy[0], xy[-1]):
            xy = np.vstack([xy, xy[0]])
        xy = _dedupe(xy)

        # Hess-Smith wants clockwise ordering (fluid on the left of travel).
        signed_area2 = float(np.sum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1]))
        if signed_area2 > 0:  # counter-clockwise -> flip
            xy = xy[::-1]
        self.xy = xy

        self.x, self.y = xy[:, 0], xy[:, 1]
        self.xc = 0.5 * (self.x[:-1] + self.x[1:])
        self.yc = 0.5 * (self.y[:-1] + self.y[1:])
        dx = self.x[1:] - self.x[:-1]
        dy = self.y[1:] - self.y[:-1]
        self.s = np.hypot(dx, dy)
        self.theta = np.arctan2(dy, dx)
        self.cos_t = np.cos(self.theta)
        self.sin_t = np.sin(self.theta)
        # Outward normal for clockwise ordering.
        self.nx = -self.sin_t
        self.ny = self.cos_t
        self.tx = self.cos_t
        self.ty = self.sin_t
        self.n = self.s.size
        self._sol: PanelSolution | None = None

    # ------------------------------------------------------------------ core

    def _influence(self, px: np.ndarray, py: np.ndarray):
        """Unit-strength global-frame velocity influence of every panel.

        Returns four ``(M, N)`` arrays: ``us, vs`` from unit sources and
        ``uv, vv`` from unit vortices.
        """
        px = np.ravel(px)
        py = np.ravel(py)
        dx = px[:, None] - self.x[None, :-1]
        dy = py[:, None] - self.y[None, :-1]
        # rotate into each panel's local frame
        xt = dx * self.cos_t[None] + dy * self.sin_t[None]
        yt = -dx * self.sin_t[None] + dy * self.cos_t[None]

        r1sq = xt * xt + yt * yt
        xt2 = xt - self.s[None]
        r2sq = xt2 * xt2 + yt * yt
        # 0.5*log(r1^2/r2^2) == log(r1/r2), one fewer sqrt
        lnr = 0.5 * np.log(np.clip(r1sq, 1e-24, None) / np.clip(r2sq, 1e-24, None))
        dth = np.arctan2(yt, xt2) - np.arctan2(yt, xt)

        inv2pi = 1.0 / (2.0 * np.pi)
        us_l = lnr * inv2pi
        ws_l = dth * inv2pi
        uv_l = ws_l  # vortex local u == source local w
        wv_l = -us_l  # and vortex local w == -source local u

        c = self.cos_t[None]
        sn = self.sin_t[None]
        us = us_l * c - ws_l * sn
        vs = us_l * sn + ws_l * c
        uv = uv_l * c - wv_l * sn
        vv = uv_l * sn + wv_l * c
        return us, vs, uv, vv

    def solve(self, alpha_deg: float = 0.0, uinf: float = 1.0) -> PanelSolution:
        """Assemble and solve the (N+1) linear system."""
        a = np.deg2rad(alpha_deg)
        vx, vy = uinf * np.cos(a), uinf * np.sin(a)

        us, vs, uv, vv = self._influence(self.xc, self.yc)
        # Analytic self-influence. A control point sits *on* its own panel where
        # yt is exactly zero, and arctan2(-0.0, -x) returns -pi while
        # arctan2(+0.0, -x) returns +pi -- a sign that flips with round-off. Pin
        # the diagonals to the closed-form values instead: a constant source
        # panel induces (0, 1/2) on itself in local coords, a constant vortex
        # panel induces (1/2, 0).
        idx = np.arange(self.n)
        us[idx, idx] = 0.5 * self.nx
        vs[idx, idx] = 0.5 * self.ny
        uv[idx, idx] = 0.5 * self.tx
        vv[idx, idx] = 0.5 * self.ty

        an = us * self.nx[:, None] + vs * self.ny[:, None]
        at = us * self.tx[:, None] + vs * self.ty[:, None]
        if not (np.all(np.isfinite(an)) and np.all(np.isfinite(at))):
            bad = np.argwhere(~np.isfinite(at))
            i, j = (int(bad[0][0]), int(bad[0][1])) if bad.size else (-1, -1)
            raise RuntimeError(
                f"influence matrix is not finite: first bad entry at panel "
                f"({i}, {j}), panel lengths {self.s[i]:.3e}/{self.s[j]:.3e}, "
                f"control point ({self.xc[i]:.5f}, {self.yc[i]:.5f}). "
                f"Usually a duplicated or near-duplicated coordinate in the "
                f".dat file that survived deduplication."
            )

        av_n = (uv * self.nx[:, None] + vv * self.ny[:, None]).sum(axis=1)
        av_t = (uv * self.tx[:, None] + vv * self.ty[:, None]).sum(axis=1)

        a_mat = np.zeros((self.n + 1, self.n + 1))
        a_mat[: self.n, : self.n] = an
        a_mat[: self.n, self.n] = av_n
        a_mat[self.n, : self.n] = at[0] + at[-1]
        a_mat[self.n, self.n] = av_t[0] + av_t[-1]

        rhs = np.zeros(self.n + 1)
        rhs[: self.n] = -(vx * self.nx + vy * self.ny)
        rhs[self.n] = -(
            (vx * self.tx[0] + vy * self.ty[0]) + (vx * self.tx[-1] + vy * self.ty[-1])
        )

        sol = np.linalg.solve(a_mat, rhs)
        # A near-singular system does not raise: LAPACK returns values that are
        # finite but astronomically large, and they only surface later as
        # overflow warnings inside the surface-velocity matmul, naming nothing.
        # Checking finiteness alone misses that case -- the magnitude is what
        # gives it away. Source strengths are O(1) for a unit-chord foil in a
        # unit freestream.
        worst = float(np.max(np.abs(sol))) if sol.size else 0.0
        if not np.all(np.isfinite(sol)) or worst > 1e6:
            k = int(np.argmin(self.s))
            raise RuntimeError(
                f"panel system is ill-conditioned: max|solution| = {worst:.3e} "
                f"(expected O(1)), {int((~np.isfinite(sol)).sum())} non-finite. "
                f"n_panels={self.n}, cond={np.linalg.cond(a_mat):.3e}, "
                f"shortest panel {self.s[k]:.3e} at index {k} "
                f"({self.xc[k]:.4f}, {self.yc[k]:.4f}), "
                f"chord={self.x.max() - self.x.min():.4f}"
            )
        sigma = sol[: self.n]
        gamma = float(sol[self.n])

        # Apple's Accelerate BLAS raises the divide-by-zero and overflow FPE
        # flags during some GEMM calls even when every input and every output
        # is finite and correctly valued. NumPy 2 surfaces those flags as
        # RuntimeWarnings attributed to this line. Both operands were already
        # checked finite above and the solution magnitude was bounded, so the
        # flags carry no information here -- suppress them, then verify the
        # result rather than trusting the suppression.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            vt = at @ sigma + av_t * gamma + (vx * self.tx + vy * self.ty)
        if not np.all(np.isfinite(vt)):
            raise RuntimeError(
                f"surface velocity is not finite on {int((~np.isfinite(vt)).sum())} "
                f"of {vt.size} panels: max|sigma| = {np.max(np.abs(sigma)):.3e}, "
                f"gamma = {gamma:.3e}"
            )
        cp = 1.0 - (vt / uinf) ** 2

        # Kutta-Joukowski. Panels were forced clockwise in __init__, which fixes
        # the sign of the circulation integral, so no extra sign juggling here.
        circulation = gamma * float(self.s.sum())
        cl_kj = 2.0 * circulation / uinf  # chord = 1

        # Cp integration, in the *freestream* frame (freestream is horizontal
        # after geometry rotation, so lift is +y and drag is +x).
        dfx = -cp * self.nx * self.s
        dfy = -cp * self.ny * self.s
        cl_p = float(dfy.sum())
        cd_p = float(dfx.sum())
        cm = float(np.sum((self.xc - 0.25) * dfy - self.yc * dfx))

        self._sol = PanelSolution(
            sigma=sigma,
            gamma=gamma,
            cp=cp,
            vt=vt / uinf,
            cl=float(cl_kj),
            cl_pressure=cl_p,
            cd_pressure=cd_p,
            cm_quarter=cm,
            circulation=float(circulation),
            alpha_deg=float(alpha_deg),
            uinf=float(uinf),
        )
        return self._sol

    # --------------------------------------------------------------- queries

    def velocity(
        self,
        px: np.ndarray,
        py: np.ndarray,
        gamma_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Induced + freestream velocity at arbitrary points.

        ``gamma_scale`` multiplies the bound circulation only (sources, i.e. the
        thickness effect, are untouched). Separation reduces circulation without
        changing displacement thickness to first order, so the viscous model
        uses this to make the far field consistent with the stalled ``Cl``.
        """
        if self._sol is None:
            raise RuntimeError("call solve() first")
        sol = self._sol
        shape = np.asarray(px).shape
        px = np.ravel(px)
        py = np.ravel(py)
        u = np.empty(px.size)
        v = np.empty(px.size)
        g = sol.gamma * gamma_scale
        for i in range(0, px.size, _CHUNK):
            sl = slice(i, min(i + _CHUNK, px.size))
            us, vs, uv, vv = self._influence(px[sl], py[sl])
            u[sl] = us @ sol.sigma + g * uv.sum(axis=1)
            v[sl] = vs @ sol.sigma + g * vv.sum(axis=1)
        a = np.deg2rad(sol.alpha_deg)
        u += sol.uinf * np.cos(a)
        v += sol.uinf * np.sin(a)
        return u.reshape(shape), v.reshape(shape)

    def surface(self) -> dict:
        """Control points, normals, panel lengths and Cp as plain arrays."""
        if self._sol is None:
            raise RuntimeError("call solve() first")
        return {
            "xc": self.xc,
            "yc": self.yc,
            "nx": self.nx,
            "ny": self.ny,
            "s": self.s,
            "cp": self._sol.cp,
            "vt": self._sol.vt,
        }


def _dedupe(xy: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Drop zero-length panels; they make the influence matrix singular."""
    keep = [0]
    for i in range(1, xy.shape[0]):
        if np.hypot(*(xy[i] - xy[keep[-1]])) > tol:
            keep.append(i)
    out = xy[keep]
    if not np.allclose(out[0], out[-1]):
        out = np.vstack([out, out[0]])
    return out


def resample_polygon(xy: np.ndarray, n: int = 180) -> np.ndarray:
    """Cosine-cluster a polygon onto ``n`` panels (dense at LE and TE).

    Raw ``.dat`` files carry anywhere from 30 to 300 points with wildly
    different clustering; the panel method is far better conditioned on a
    uniform panel count, and the U-Net dataset needs every foil treated alike.
    """
    xy = np.asarray(xy, float)
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    # cosine spacing in arc length: clusters both ends of the loop, which after
    # the TE->upper->LE->lower->TE ordering means TE *and* LE get fine panels
    # Spacing density d(t)/d(beta) = (1 - cos 2*beta) / 2*pi vanishes at
    # beta = 0, pi, 2*pi -- i.e. panels bunch at the trailing edge (start and
    # end of the loop) *and* at the leading edge (mid-loop). Clustering only at
    # the TE leaves the stagnation point unresolved and Cp_max lands well short
    # of 1.
    beta = np.linspace(0.0, 2.0 * np.pi, n + 1)
    t = (beta - 0.5 * np.sin(2.0 * beta)) / (2.0 * np.pi) * total
    out = np.column_stack([
        np.interp(t, arc, xy[:, 0]),
        np.interp(t, arc, xy[:, 1]),
    ])
    out[-1] = out[0]
    return _dedupe(out)
