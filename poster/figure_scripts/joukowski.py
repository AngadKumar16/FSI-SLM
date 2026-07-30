"""Exact inviscid flow via the Joukowski conformal map. Analytic, stable.
Gives streamlines, velocity magnitude, and pressure coefficient around a
symmetric ~9%-thick foil (NACA 0009-like) with the Kutta condition enforced.
"""
from __future__ import annotations
import numpy as np

B = 1.0  # TE maps to z = 2B


def circle_params(thickness=0.09):
    # center shift on real axis controls thickness (symmetric foil, muy=0)
    mux = -thickness * B * 1.30
    mu = complex(mux, 0.0)
    R = abs(B - mu)          # circle passes through zeta=B (TE)
    return mu, R


def zeta_circle(mu, R, n=400):
    t = np.linspace(0, 2*np.pi, n)
    return mu + R*np.exp(1j*t)


def joukowski(zeta):
    return zeta + B**2/zeta


def flow(alpha_deg, thickness=0.09, Uinf=1.0, grid=(-2.2, 3.2, -1.6, 1.6), res=520):
    a = np.radians(alpha_deg)
    mu, R = circle_params(thickness)
    beta = np.arcsin(mu.imag/R) if R else 0.0
    Gamma = 4*np.pi*Uinf*R*np.sin(a + beta)   # Kutta

    # airfoil surface (z-plane)
    zc = joukowski(zeta_circle(mu, R, 600))
    xf, yf = zc.real, zc.imag
    chord = xf.max() - xf.min()

    # grid in z-plane -> invert map to zeta -> evaluate cylinder flow
    xs = np.linspace(grid[0], grid[1], res)
    ys = np.linspace(grid[2], grid[3], int(res*(grid[3]-grid[2])/(grid[1]-grid[0])))
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j*Y
    # invert z = zeta + B^2/zeta  ->  zeta = (z +/- sqrt(z^2-4B^2))/2, pick outside
    root = np.sqrt(Z**2 - 4*B**2)
    z1 = (Z + root)/2
    z2 = (Z - root)/2
    zeta = np.where(np.abs(z1 - mu) >= np.abs(z2 - mu), z1, z2)

    # complex velocity in zeta-plane (cylinder + circulation)
    dwdzeta = (Uinf*np.exp(-1j*a)
               - Uinf*R**2*np.exp(1j*a)/(zeta - mu)**2
               + 1j*Gamma/(2*np.pi*(zeta - mu)))
    dzdzeta = 1 - B**2/zeta**2
    W = dwdzeta/dzdzeta          # = u - i v
    u, v = W.real, -W.imag
    speed = np.hypot(u, v)
    Cp = 1 - (speed/Uinf)**2

    # mask points inside the airfoil (|zeta-mu| < R)
    inside = np.abs(zeta - mu) < R*1.001
    for arr in (u, v, speed, Cp):
        arr[inside] = np.nan

    # stream function (for clean streamlines): psi from cylinder flow
    zt = zeta - mu
    psi = (Uinf*(zt*np.exp(-1j*a)).imag
           + Uinf*R**2*(np.exp(1j*a)/zt).imag
           - Gamma/(2*np.pi)*np.log(np.abs(zt)))
    psi = np.asarray(psi, float)
    psi[inside] = np.nan

    # surface Cp
    zetas = zeta_circle(mu, R, 600)
    dwc = (Uinf*np.exp(-1j*a) - Uinf*R**2*np.exp(1j*a)/(zetas-mu)**2
           + 1j*Gamma/(2*np.pi*(zetas-mu)))
    dzc = 1 - B**2/zetas**2
    # avoid the TE singularity
    Wc = np.where(np.abs(dzc) > 1e-6, dwc/np.where(np.abs(dzc)>1e-6,dzc,1), np.nan)
    Cps = 1 - np.abs(Wc)**2/Uinf**2

    Cl = 2*Gamma/(Uinf*chord)
    return dict(X=X, Y=Y, u=u, v=v, speed=speed, Cp=Cp, psi=psi,
                xf=xf, yf=yf, xs_surf=zc.real, Cps=Cps, chord=chord, Cl=Cl,
                xlim=(grid[0], grid[1]), ylim=(grid[2], grid[3]))


if __name__ == "__main__":
    for al in (0, 4, 8, 12):
        f = flow(al)
        print(f"alpha={al:5.1f}  Cl={f['Cl']:+.3f}  2*pi*a={2*np.pi*np.radians(al):+.3f}  chord={f['chord']:.3f}")
