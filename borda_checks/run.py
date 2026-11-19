# numerical checks on the expected borda score quoted in the appendix.
# beta(x) is the probability that a random voter prefers a candidate at x to a random opponent,
# so the expected borda score is n(m-1) beta(x).
# appendix a.3: voters and candidates uniform on [-1,1]^2; beta along seven rays and four
# horizontal segments, by gauss-legendre quadrature in polar coordinates around x.
# appendix b.3: voters at (+-1/2, 0) and candidates uniform on the unit disc; the range of s
# where beta((s,0)) > beta(0), and where the gain is largest.
# writes data/borda_checks.json and prints the results.
import json
import os

import numpy as np
from numpy.polynomial.legendre import leggauss

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "borda_checks.json")


def square_halfplane(e, t):
    # share of the uniform distribution on [-1,1]^2 in {<z, e> < t}
    a = np.maximum(np.maximum(np.abs(e[..., 0]), np.abs(e[..., 1])), 1e-15)
    b = np.maximum(np.minimum(np.abs(e[..., 0]), np.abs(e[..., 1])), 1e-15)
    s = np.clip(t, -(a + b), a + b)
    return np.where(s <= b - a, (s + a + b) ** 2 / (8 * a * b),
                    np.where(s <= a - b, 0.5 + s / (2 * a), 1 - (a + b - s) ** 2 / (8 * a * b)))


def beta_square(x, n_rho=96, n_phi=1440):
    x = np.asarray(x, float)
    phi = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    e = np.c_[np.cos(phi), np.sin(phi)]
    with np.errstate(divide="ignore"):
        exits = np.where(e > 0, (1 - x) / e, np.where(e < 0, (-1 - x) / e, np.inf))
    rmax = exits.min(1)
    u, wq = leggauss(n_rho)
    rho = np.maximum(0.5 * (u[None, :] + 1) * rmax[:, None], 1e-12)
    wr = 0.5 * wq[None, :] * rmax[:, None]
    # offset of the bisector of x and x + rho e along e
    t = (x @ x + 2 * rho * (e @ x)[:, None] + rho ** 2 - x @ x) / (2 * rho)
    vals = square_halfplane(e[:, None, :], t)
    return (vals * rho * wr).sum() * (2 * np.pi / n_phi) / 4


def square_check():
    worst_ray, worst_segment = -np.inf, -np.inf
    for th in np.linspace(0, np.pi / 4, 7):
        w = np.array([np.cos(th), np.sin(th)])
        v = [beta_square(s * w) for s in np.linspace(0, 1 / max(w), 21)]
        worst_ray = max(worst_ray, np.diff(v).max())
    for x2 in [0.0, 0.3, 0.6, 0.9]:
        v = [beta_square((x1, x2)) for x1 in np.linspace(0, 1, 21)]
        worst_segment = max(worst_segment, np.diff(v).max())
    return {"beta_at_center": beta_square((0, 0)), "largest_step_along_rays": worst_ray,
            "largest_step_along_segments": worst_segment}


def disc_ball_mass(r):
    # mass of the uniform distribution on the unit disc in the ball of radius r about (1/2, 0)
    if r <= 0.5:
        return r * r
    a1 = r * r * np.arccos((0.25 + r * r - 1) / r)
    a2 = np.arccos(0.25 + 1 - r * r)
    a3 = 0.5 * np.sqrt((-0.5 + r + 1) * (0.5 + r - 1) * (0.5 - r + 1) * (0.5 + r + 1))
    return (a1 + a2 - a3) / np.pi


def counterexample_gain(s):
    # beta((s,0)) - beta(0) for two equal blocs of voters at (+-1/2, 0)
    return 0.25 - 0.5 * (disc_ball_mass(0.5 - s) + disc_ball_mass(0.5 + s))


def counterexample_check():
    lo, hi = 0.1, 0.2
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if counterexample_gain(mid) > 0 else (lo, mid)
    s = np.linspace(1e-4, lo, 2000)
    gains = np.array([counterexample_gain(v) for v in s])
    return {"gain_positive_up_to": lo, "largest_gain_at": float(s[gains.argmax()]), "largest_gain": float(gains.max())}


if __name__ == "__main__":
    res = {"square": square_check(), "counterexample": counterexample_check()}
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    json.dump(res, open(DATA, "w"), indent=1)
    print(json.dumps(res, indent=1))
