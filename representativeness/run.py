# representativeness of small committees (figures 8 and 9).
# n = 500 voters uniform on the unit disc; m in {20, 100, 250, 500} candidates, uniform or
# gaussian conditioned on the disc; k = m/10 seats; 2500 elections per setting. the benchmark
# is k points drawn from the voter distribution. for every committee we record the average and
# minimum pairwise distance, the dissatisfaction (mean distance from a voter to the nearest
# winner) and the smallest wedge at the origin that contains all winners, and, for m = 100,
# the number of winners in each quadrant.
# writes data/representativeness.npz.
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import cdist, pdist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.elections import k_approval, k_borda, rankings, sample_disc, sample_gaussian_disc, sntv

N, M_VALUES, TRIALS, QUAD_M = 500, [20, 100, 250, 500], 2500, 100
RULES = ["random_voters", "kborda", "kapproval_m10", "sntv"]
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "representativeness.npz")
# quadrature points for the dissatisfaction integral against the voter distribution
MC_POINTS = sample_disc(20000, np.random.default_rng(7))


def metrics(W):
    dissat = cdist(MC_POINTS, W).min(1).mean()
    if len(W) < 2:
        return 0.0, 0.0, dissat, 0.0
    pd = pdist(W)
    ang = np.sort(np.arctan2(W[:, 1], W[:, 0]))
    max_gap = max(np.diff(ang).max(), 2 * np.pi - ang[-1] + ang[0])
    return pd.mean(), pd.min(), dissat, 2 * np.pi - max_gap


def quadrant_counts(W):
    x, y = W[:, 0], W[:, 1]
    return [np.sum((x >= 0) & (y >= 0)), np.sum((x < 0) & (y >= 0)),
            np.sum((x < 0) & (y < 0)), np.sum((x >= 0) & (y < 0))]


def trial(args):
    cand, m, seed = args
    rng = np.random.default_rng(seed)
    k = max(1, int(0.1 * m))
    V = sample_disc(N, rng)
    C = sample_disc(m, rng) if cand == "unif" else sample_gaussian_disc(m, rng)
    order = rankings(V, C)
    won = {"sntv": sntv(order, k, rng), "kapproval_m10": k_approval(order, k, k, rng), "kborda": k_borda(order, k, rng)}
    committees = {"random_voters": sample_disc(k, rng), **{r: C[w] for r, w in won.items()}}
    return ([metrics(committees[r]) for r in RULES],
            [quadrant_counts(committees[r]) for r in RULES] if m == QUAD_M else None)


def run():
    out = {}
    with Pool() as pool:
        for cand, base in [("unif", 11), ("gauss", 12)]:
            for m in M_VALUES:
                seeds = np.random.SeedSequence([base, m]).generate_state(TRIALS)
                res = pool.map(trial, [(cand, m, int(s)) for s in seeds], chunksize=20)
                # arrays of shape (elections, rules, metrics) and (elections, rules, quadrants)
                out[f"{cand}_{m}"] = np.array([r[0] for r in res]).astype(np.float32)
                if m == QUAD_M:
                    out[f"{cand}_quad"] = np.array([r[1] for r in res]).astype(np.int8)
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, rules=np.array(RULES), **out)


if __name__ == "__main__":
    run()
