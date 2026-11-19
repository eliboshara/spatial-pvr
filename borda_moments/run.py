# mean and variance of the borda score of a candidate at a fixed position (figures 11 and 12).
# monte carlo estimates, with the voters and the m - 1 opponents resampled in every replicate,
# for n in {100, 500, 1000} and m in {2, 5, 10}. on the unit disc the candidate moves along
# [0,1] x {0} and the estimates are compared with the exact mean and variance of proposition 3.1;
# on the square [-1,1]^2 it moves over a 40 x 40 grid (monte carlo only).
# writes data/borda_moments.npz.
import os
import sys

import numpy as np
from scipy import integrate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.elections import sample_disc, sample_square

N_VALUES, M_VALUES = [100, 500, 1000], [2, 5, 10]
DISC_REPS, SQUARE_REPS, POINTS = 5000, 1000, 150
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "borda_moments.npz")


def exact_mean_integral(num=POINTS):
    # the disc integral for the expected borda score along the ray, in polar coordinates
    xs = np.linspace(0, 1, num)
    vals = np.zeros(num)
    for i, x1 in enumerate(xs):
        def f(r, th):
            dist = np.sqrt(r ** 2 + x1 ** 2 - 2 * r * x1 * np.cos(th))
            d = 0.0 if dist < 1e-12 else np.clip((r ** 2 - x1 ** 2) / (2 * dist), -1, 1)
            return 2 * r * (d * np.sqrt(1 - d ** 2) + np.arcsin(d) + np.pi / 2)
        vals[i] = integrate.dblquad(f, 0, np.pi, 0.0, 1.0, epsabs=1e-5, epsrel=1e-5)[0]
    return xs, vals


def exact_p123(num=POINTS, res=500):
    # p1, p2 and p3 of the variance formula for uniform voters and candidates on the disc
    r_vals = np.linspace(0, 1, num)
    s_vals, th_vals = np.linspace(1e-5, 1, res), np.linspace(0, np.pi, res)
    S, Th = np.meshgrid(s_vals, th_vals)
    p1, p2, p3 = np.zeros(num), np.zeros(num), np.zeros(num)
    for i, r in enumerate(r_vals):
        dist = np.maximum(np.sqrt(r ** 2 + S ** 2 - 2 * r * S * np.cos(Th)), 1e-10)
        h = np.clip((r ** 2 - S ** 2) / (2 * dist), -1, 1)
        # area of the part of the disc closer to x than to a point at polar coordinates (s, th)
        A_xy = np.arccos(h) - h * np.sqrt(1 - h ** 2)
        # area of the disc inside the ball around (s, th) that passes through x
        inter = np.zeros_like(S)
        inside, none = S <= np.abs(1 - dist), S >= 1 + dist
        part = ~(inside | none)
        inter[inside] = np.pi * np.minimum(1.0, dist[inside]) ** 2
        Sp, Rp = S[part], dist[part]
        x = np.clip((Sp ** 2 - Rp ** 2 + 1) / (2 * Sp), -1, 1)
        y = (Sp ** 2 + Rp ** 2 - 1) / (2 * Sp)
        inter[part] = (np.arccos(x) - x * np.sqrt(np.maximum(0, 1 - x ** 2))
                       + Rp ** 2 * np.arccos(np.clip(y / Rp, -1, 1)) - y * np.sqrt(np.maximum(0, Rp ** 2 - y ** 2)))
        C_xz = np.pi - inter
        integ = lambda F: 2 * np.trapezoid(np.trapezoid(F * S, s_vals, axis=1), th_vals)
        p1[i], p2[i], p3[i] = integ(A_xy) / np.pi ** 2, integ(A_xy ** 2) / np.pi ** 3, integ(C_xz ** 2) / np.pi ** 3
    return r_vals, p1, p2, p3


def monte_carlo(sampler, test_points, reps, rng):
    mean, var = {}, {}
    for m in M_VALUES:
        for n in N_VALUES:
            scores = np.zeros((reps, len(test_points)))
            for i in range(reps):
                V, Y = sampler(n, rng), sampler(m - 1, rng)
                dx = np.linalg.norm(V[:, None, :] - test_points[None, :, :], axis=2)
                dy = np.linalg.norm(V[:, None, :] - Y[None, :, :], axis=2)
                scores[i] = (dx[:, :, None] < dy[:, None, :]).sum(axis=(0, 2))
            mean[(m, n)], var[(m, n)] = scores.mean(0), scores.var(0)
    return mean, var


def run():
    rng = np.random.default_rng(41)
    radii = np.linspace(0, 1, POINTS)
    mean, var = monte_carlo(sample_disc, np.c_[radii, np.zeros(POINTS)], DISC_REPS, rng)
    xs, base = exact_mean_integral()
    r_vals, p1, p2, p3 = exact_p123()
    g = np.linspace(-1, 1, 40)
    X1, X2 = np.meshgrid(g, g)
    mean_sq, var_sq = monte_carlo(sample_square, np.c_[X1.ravel(), X2.ravel()], SQUARE_REPS, rng)
    data = dict(radii=radii, xs=xs, base=base, r_vals=r_vals, p1=p1, p2=p2, p3=p3)
    for name, d in [("mean", mean), ("var", var), ("mean_sq", mean_sq), ("var_sq", var_sq)]:
        data.update({f"{name}_{m}_{n}": v for (m, n), v in d.items()})
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, **data)


if __name__ == "__main__":
    run()
