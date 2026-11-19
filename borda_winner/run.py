# where the single borda winner lies (figure 3).
# n = 1000 voters and m in {1, 5, 10, 25, 50, 100} candidates uniform on the unit disc,
# 1000 elections per m. writes the distance of each winner from the center to data/borda_winner.npz.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.elections import borda_scores, rankings, sample_disc, top_k

N, TRIALS, M_VALUES = 1000, 1000, [1, 5, 10, 25, 50, 100]
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "borda_winner.npz")


def run():
    rng = np.random.default_rng(31)
    data = {}
    for m in M_VALUES:
        radii = np.empty(TRIALS)
        for t in range(TRIALS):
            V, C = sample_disc(N, rng), sample_disc(m, rng)
            radii[t] = np.linalg.norm(C[top_k(borda_scores(rankings(V, C)), 1, rng)[0]])
        data[f"radii_{m}"] = radii
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, **data)


if __name__ == "__main__":
    run()
