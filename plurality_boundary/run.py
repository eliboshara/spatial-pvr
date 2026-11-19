# plurality on the unit disc with uniform voters and candidates (figure 4 and appendix a.4).
# computes the normalized expected plurality score p(x)/(n/m) along a radius for several m,
# and its boundary profile p(d) in the limit m -> infinity, with the distance d to the rim
# in units of m^(-1/2). writes data/plurality_boundary.npz and prints the peak of p(d).
import os

import numpy as np
from scipy import integrate

M_VALUES = [5, 25, 50, 100, 250, 500]
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "plurality_boundary.npz")


def lens_area(d, r, R=1.0):
    # area of the intersection of the unit disc about 0 and the disc of radius r about a point at distance d
    d, r = np.broadcast_arrays(np.asarray(d, float), np.asarray(r, float))
    out = np.where(d >= R + r, 0.0, np.where(d <= R - r, np.pi * r ** 2, np.pi * R ** 2))
    part = (d < R + r) & (d > np.abs(R - r))
    dp, rp = d[part], r[part]
    a1 = rp ** 2 * np.arccos(np.clip((dp ** 2 + rp ** 2 - R ** 2) / (2 * dp * rp), -1, 1))
    a2 = R ** 2 * np.arccos(np.clip((dp ** 2 + R ** 2 - rp ** 2) / (2 * dp * R), -1, 1))
    a3 = 0.5 * np.sqrt(np.clip((-dp + rp + R) * (dp - rp + R) * (dp + rp - R) * (dp + rp + R), 0, None))
    out = out.copy()
    out[part] = a1 + a2 - a3
    return out


def normalized_plurality(x0, m, n_rho=1500, n_phi=720):
    # m times the integral over the disc of (1 - |d cap b(z, |z - x|)|/pi)^(m-1) dz/pi, around x = (x0, 0)
    u = (np.arange(n_rho) + 0.5) / n_rho
    rho, drho = 2 * u ** 2, 4 * u / n_rho
    phi = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    R, PH = np.meshgrid(rho, phi)
    Z1, Z2 = x0 + R * np.cos(PH), R * np.sin(PH)
    inside = Z1 ** 2 + Z2 ** 2 <= 1
    beat = np.clip(1 - lens_area(np.hypot(Z1, Z2), R) / np.pi, 0, 1)
    integrand = np.where(inside, beat ** (m - 1), 0.0) * R * drho[None, :] * (2 * np.pi / n_phi)
    return m * integrand.sum() / np.pi


def boundary_area(u, v, d):
    # 1/pi times the area of the upper half-plane inside the disc about (u, v) through (0, d)
    r2 = u * u + (v - d) ** 2
    rr = np.sqrt(r2)
    seg = ((0.5 + np.arcsin(np.clip(v / np.maximum(rr, 1e-300), -1, 1)) / np.pi) * r2
           + v * np.sqrt(np.maximum(r2 - v * v, 0)) / np.pi)
    return np.where(rr <= v, r2, seg)


def profile(d):
    U = np.linspace(-8, 8, 1201)
    V = np.linspace(0, 8 + d, int(601 + 80 * d))
    uu, vv = np.meshgrid(U, V)
    return integrate.simpson(integrate.simpson(np.exp(-boundary_area(uu, vv, d)), x=U, axis=1), x=V) / np.pi


if __name__ == "__main__":
    xs = np.linspace(0, 1, 100)
    scores = np.array([[normalized_plurality(x, m) for x in xs] for m in M_VALUES])
    ds = np.linspace(0, 5, 81)
    Ps = np.array([profile(d) for d in ds])
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, m_values=M_VALUES, xs=xs, scores=scores, ds=ds, profile=Ps)
    i = int(np.argmax(Ps))
    print(f"p(0) = {Ps[0]:.3f}, max p = {Ps[i]:.3f} at d = {ds[i]:.2f}")
