# where the winners lie (figure 7).
# n = 500 voters uniform on the unit disc; m in {10, 100, 500} candidates, uniform or gaussian
# conditioned on the disc; k = max(1, m/10) seats; 10000 elections per setting.
# every election elects four committees from the same voters and candidates: sntv,
# approval of each voter's m/10 nearest candidates, k-borda, and k candidates drawn at random.
# writes the distances of the winners from the center to data/winner_density.npz.
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.elections import k_approval, k_borda, rankings, sample_disc, sample_gaussian_disc, sntv

N, M_VALUES, TRIALS = 500, [10, 100, 500], 10000
RULES = ["sntv", "kapproval_m10", "kborda", "random_committee"]
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "winner_density.npz")


def trial(args):
    cand, m, seed = args
    rng = np.random.default_rng(seed)
    k = max(1, m // 10)
    V = sample_disc(N, rng)
    C = sample_disc(m, rng) if cand == "unif" else sample_gaussian_disc(m, rng)
    order = rankings(V, C)
    won = {"sntv": sntv(order, k, rng), "kapproval_m10": k_approval(order, k, k, rng), "kborda": k_borda(order, k, rng),
           "random_committee": rng.choice(m, size=k, replace=False)}
    radius = np.linalg.norm(C, axis=1)
    return {rule: radius[w] for rule, w in won.items()}


def run():
    out = {}
    with Pool() as pool:
        for cand, base in [("unif", 31), ("gauss", 32)]:
            for m in M_VALUES:
                seeds = np.random.SeedSequence([base, m]).generate_state(TRIALS)
                res = pool.map(trial, [(cand, m, int(s)) for s in seeds], chunksize=25)
                for rule in RULES:
                    out[f"{cand}_{m}_{rule}"] = np.concatenate([r[rule] for r in res]).astype(np.float32)
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, n=N, trials=TRIALS, **out)


if __name__ == "__main__":
    run()
