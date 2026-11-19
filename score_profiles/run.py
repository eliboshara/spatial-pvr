# mean normalized score against the distance from the center (figure 5).
# n = 500 voters uniform on the unit disc; m in {10, 100, 500, 2000} candidates, uniform or
# gaussian conditioned on the disc. in every election the scores of sntv, 5-approval,
# approval of the m/10 nearest candidates and borda are normalized so that each rule's
# weights sum to one, which makes (m/n) v_m average to one over the candidates, and are
# averaged over the candidates in 40 equal-area annuli.
# writes data/score_profiles.npz.
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.elections import rankings, sample_disc, sample_gaussian_disc

N = 500
M_TRIALS = {10: 40000, 100: 20000, 500: 5000, 2000: 2000}
BINS = 40
RULES = ["sntv", "kapproval_5", "kapproval_m10", "kborda"]
EDGES = np.sqrt(np.linspace(0, 1, BINS + 1))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "score_profiles.npz")


def normalized_scores(V, C):
    n, m = len(V), len(C)
    k = max(1, m // 10)
    order = rankings(V, C)
    sntv = np.bincount(order[:, 0], minlength=m)
    kapp_m10 = np.bincount(order[:, :k].ravel(), minlength=m)
    kapp_5 = np.bincount(order[:, :5].ravel(), minlength=m)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(m)[None, :], axis=1)
    borda = (m - 1 - ranks).sum(0)
    return {"sntv": sntv * m / n, "kapproval_5": kapp_5 * m / (n * 5), "kapproval_m10": kapp_m10 * m / (n * k),
            "kborda": borda * 2 / (n * (m - 1))}


def batch(args):
    # sums of the normalized scores and of their squares, and candidate counts, per annulus
    cand, m, seeds = args
    out = {f"{r}_{s}": np.zeros(BINS) for r in RULES for s in ("sum", "sq")}
    out["count"] = np.zeros(BINS)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        V = sample_disc(N, rng)
        C = sample_disc(m, rng) if cand == "unif" else sample_gaussian_disc(m, rng)
        b = np.clip(np.searchsorted(EDGES, np.linalg.norm(C, axis=1), side="right") - 1, 0, BINS - 1)
        out["count"] += np.bincount(b, minlength=BINS)
        for rule, s in normalized_scores(V, C).items():
            out[f"{rule}_sum"] += np.bincount(b, weights=s, minlength=BINS)
            out[f"{rule}_sq"] += np.bincount(b, weights=s ** 2, minlength=BINS)
    return out


def run():
    res = {"n": N, "edges": EDGES}
    with Pool() as pool:
        for cand, base in [("unif", 41), ("gauss", 42)]:
            for m, trials in M_TRIALS.items():
                seeds = [int(s) for s in np.random.SeedSequence([base, m]).generate_state(trials)]
                parts = pool.map(batch, [(cand, m, seeds[i::200]) for i in range(200)])
                tot = {key: sum(p[key] for p in parts) for key in parts[0]}
                res[f"{cand}_{m}_count"] = tot["count"]
                for rule in RULES:
                    mean = tot[f"{rule}_sum"] / tot["count"]
                    var = tot[f"{rule}_sq"] / tot["count"] - mean ** 2
                    res[f"{cand}_{m}_{rule}_mean"] = mean
                    res[f"{cand}_{m}_{rule}_se"] = np.sqrt(var / tot["count"])
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, **res)


if __name__ == "__main__":
    run()
