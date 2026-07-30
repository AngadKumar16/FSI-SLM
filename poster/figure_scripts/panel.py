"""Hess-Smith constant-strength source + single vortex panel method.
Used to render inviscid flow fields / Cp / Cl-alpha for the poster.
Validated against thin-airfoil theory (Cl ~ 2*pi*alpha).
"""
from __future__ import annotations
import numpy as np


def naca4(code="0009", n=160):
    """Return closed airfoil coords (TE -> upper -> LE -> lower -> TE), Selig-like."""
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    # cosine spacing
    beta = np.linspace(0, np.pi, n // 2 + 1)
    x = (1 - np.cos(beta)) / 2
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                  + 0.2843*x**3 - 0.1015*x**4)
    if m == 0:
        yc = np.zeros_like(x); dyc = np.zeros_like(x)
    else:
        yc = np.where(x < p, m/p**2*(2*p*x - x**2),
                      m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
        dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu = x - yt*np.sin(th); yu = yc + yt*np.cos(th)
    xl = x + yt*np.sin(th); yl = yc - yt*np.cos(th)
    # upper from TE->LE reversed, then lower LE->TE
    X = np.concatenate([xu[::-1], xl[1:]])
    Y = np.concatenate([yu[::-1], yl[1:]])
    return X, Y


def load_dat(path):
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                try:
                    a, b = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                if abs(a) <= 2 and abs(b) <= 2:
                    xs.append(a); ys.append(b)
    return np.array(xs), np.array(ys)


class Panels:
    def __init__(self, X, Y):
        # ensure clockwise (interior on right) for Hess-Smith: reverse if CCW
        area2 = np.sum(X[:-1]*Y[1:] - X[1:]*Y[:-1])
        if area2 > 0:  # CCW -> make CW
            X = X[::-1]; Y = Y[::-1]
        self.X, self.Y = X, Y
        self.xc = 0.5*(X[:-1] + X[1:])
        self.yc = 0.5*(Y[:-1] + Y[1:])
        dx = X[1:] - X[:-1]; dy = Y[1:] - Y[:-1]
        self.S = np.hypot(dx, dy)
        self.theta = np.arctan2(dy, dx)
        self.sin = np.sin(self.theta); self.cos = np.cos(self.theta)
        self.N = len(self.S)

    def _local_uw(self, px, py):
        """Unit source & vortex local-frame velocities at points (px,py) from
        each panel. Returns arrays shaped (M, N) for us,ws (source) uv,wv(vortex)
        already rotated into GLOBAL frame components (u,v)."""
        px = np.atleast_1d(px).ravel(); py = np.atleast_1d(py).ravel()
        M = px.size; N = self.N
        # local coords
        dX = px[:, None] - self.X[None, :-1]
        dY = py[:, None] - self.Y[None, :-1]
        xt = dX*self.cos[None] + dY*self.sin[None]
        yt = -dX*self.sin[None] + dY*self.cos[None]
        r1 = np.hypot(xt, yt)
        r2 = np.hypot(xt - self.S[None], yt)
        th1 = np.arctan2(yt, xt)
        th2 = np.arctan2(yt, xt - self.S[None])
        lnr = np.log(np.clip(r1/np.clip(r2, 1e-12, None), 1e-12, None))
        dth = th2 - th1
        # local-frame velocities (unit strength)
        us_l = lnr/(2*np.pi);  ws_l = dth/(2*np.pi)          # source
        uv_l = dth/(2*np.pi);  wv_l = -lnr/(2*np.pi)         # vortex
        # rotate local (along-panel, normal) -> global (x,y)
        c = self.cos[None]; s = self.sin[None]
        us = us_l*c - ws_l*s; vs = us_l*s + ws_l*c
        uv = uv_l*c - wv_l*s; vv = uv_l*s + wv_l*c
        return us, vs, uv, vv

    def solve(self, alpha_deg, Uinf=1.0):
        a = np.radians(alpha_deg)
        Vx, Vy = Uinf*np.cos(a), Uinf*np.sin(a)
        us, vs, uv, vv = self._local_uw(self.xc, self.yc)  # (N,N)
        nx, ny = -self.sin, self.cos      # panel normals (M,)
        tx, ty = self.cos, self.sin       # tangents
        # normal-velocity influence
        An = us*nx[:, None] + vs*ny[:, None]      # source -> normal, (N,N)
        Av = (uv*nx[:, None] + vv*ny[:, None]).sum(axis=1)  # vortex col (N,)
        A = np.zeros((self.N+1, self.N+1))
        A[:self.N, :self.N] = An
        A[:self.N, self.N] = Av
        b = -(Vx*nx + Vy*ny)
        # Kutta: tangential at panel 1 and N equal & opposite
        At = us*tx[:, None] + vs*ty[:, None]
        Avt = (uv*tx[:, None] + vv*ty[:, None]).sum(axis=1)
        A[self.N, :self.N] = At[0] + At[-1]
        A[self.N, self.N] = Avt[0] + Avt[-1]
        rhs = np.zeros(self.N+1)
        rhs[:self.N] = b
        rhs[self.N] = -((Vx*tx[0] + Vy*ty[0]) + (Vx*tx[-1] + Vy*ty[-1]))
        sol = np.linalg.solve(A, rhs)
        self.sigma = sol[:self.N]; self.gamma = sol[self.N]
        self.alpha = alpha_deg; self.Uinf = Uinf
        # surface tangential velocity & Cp
        Vt = (At @ self.sigma) + Avt*self.gamma + (Vx*tx + Vy*ty)
        self.Cp = 1 - (Vt/Uinf)**2
        Gamma = self.gamma * self.S.sum()
        self.Cl = -2*Gamma/(Uinf*1.0)   # chord=1 (sign: +lift for +alpha)
        return self.Cl

    def velocity(self, px, py):
        shape = np.asarray(px).shape
        us, vs, uv, vv = self._local_uw(px, py)
        u = us @ self.sigma + (uv @ np.full(self.N, self.gamma))
        v = vs @ self.sigma + (vv @ np.full(self.N, self.gamma))
        a = np.radians(self.alpha)
        u = u + self.Uinf*np.cos(a)
        v = v + self.Uinf*np.sin(a)
        return u.reshape(shape), v.reshape(shape)


if __name__ == "__main__":
    X, Y = naca4("0009", 200)
    p = Panels(X, Y)
    print("N panels", p.N)
    for al in [0, 3, 5, 8, 10]:
        cl = p.solve(al)
        print(f"alpha={al:5.1f}  Cl_panel={cl:+.4f}  2pi*a={2*np.pi*np.radians(al):+.4f}")
