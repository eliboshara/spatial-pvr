# expected borda score for uniform voters and candidates (figures 1 and 2), up to the constant
# n(m-1)/pi^2 on the unit disc and n(m-1)/16 on the square [-1,1]^2.
# disc: the integral over y of delta sqrt(1 - delta^2) + asin(delta) + pi/2, by a riemann sum over a grid.
# square: the integral over y of the area of the part of the square closer to x than to y,
# with that area in closed form. writes data/borda_surfaces.npz.
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "borda_surfaces.npz")


def disc_kernel(x, y):
    dist = np.linalg.norm(x - y, axis=-1)
    d = np.clip(((y ** 2).sum(-1) - (x ** 2).sum(-1)) / (2 * np.maximum(dist, 1e-300)), -1, 1)
    return d * np.sqrt(1 - d ** 2) + np.arcsin(d) + np.pi / 2


def square_halfplane_area(e, t):
    # area of {z in [-1,1]^2 : <z, e> < t} for unit vectors e and offsets t
    a = np.maximum(np.maximum(np.abs(e[:, 0]), np.abs(e[:, 1])), 1e-12)
    b = np.maximum(np.minimum(np.abs(e[:, 0]), np.abs(e[:, 1])), 1e-12)
    s = np.clip(t, -(a + b), a + b)
    # <z, e> = a u1 + b u2 with u1, u2 uniform on [-1,1] has a trapezoidal distribution
    F = np.where(s <= b - a, (s + a + b) ** 2 / (8 * a * b),
                 np.where(s <= a - b, 0.5 + s / (2 * a), 1 - (a + b - s) ** 2 / (8 * a * b)))
    return 4 * F


def disc(N=150):
    g = np.linspace(-1, 1, N)
    X1, X2 = np.meshgrid(g, g)
    mask = X1 ** 2 + X2 ** 2 <= 1
    P = np.stack([X1[mask], X2[mask]], 1)
    F = np.empty(len(P))
    for s in range(0, len(P), 200):
        K = disc_kernel(P[s:s + 200, None, :], P[None, :, :])
        K[np.arange(len(K)), np.arange(s, s + len(K))] = 0.0
        F[s:s + 200] = K.sum(1) * np.pi / len(P)
    grid = np.full(X1.shape, np.nan)
    grid[mask] = F
    return g, grid


def square(N=75):
    g = np.linspace(-1, 1, N)
    X1, X2 = np.meshgrid(g, g)
    P = np.stack([X1.ravel(), X2.ravel()], 1)
    w1 = np.full(N, 2 / (N - 1))
    w1[[0, -1]] /= 2
    W = np.outer(w1, w1).ravel()
    F = np.empty(len(P))
    for i, x in enumerate(P):
        d = P - x
        nd = np.linalg.norm(d, axis=1)
        same = nd < 1e-12
        nd[same] = 1.0
        area = square_halfplane_area(d / nd[:, None], ((P ** 2).sum(1) - x @ x) / (2 * nd))
        area[same] = 2.0
        F[i] = (W * area).sum()
    return g, F.reshape(N, N)


if __name__ == "__main__":
    g_disc, F_disc = disc()
    g_square, F_square = square()
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, g_disc=g_disc, F_disc=F_disc, g_square=g_square, F_square=F_square)
