# seats-votes curves in a polarized two-bloc electorate (figure 6).
# n = 6000 voters on [-1,1]^2 in two gaussian clusters, a fraction w around (-0.5, 0);
# m = 300 candidates uniform on the square; k = 30 seats; 10 elections per w.
# the seat share of the larger bloc is the fraction of winners with x < 0.
# writes data/seats_votes.json.
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.elections import approval_scores, k_approval, k_borda, rankings, sample_square, sample_two_blocs, sntv, top_k

WS, M, K, N, TRIALS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8), 300, 30, 6000, 10
RULES = ["sntv", "kapproval_5", "kapproval_m10", "kborda", "random_committee"]
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "seats_votes.json")


def trial(args):
    w, t = args
    seed = 1000 + int(round(100 * w)) * 50 + t
    rng = np.random.default_rng(seed)
    V = sample_two_blocs(N, w, rng)
    C = sample_square(M, rng)
    order = rankings(V, C)
    won = {"sntv": sntv(order, K, rng), "kapproval_m10": k_approval(order, K, K, rng), "kborda": k_borda(order, K, rng)}
    won["random_committee"] = rng.choice(M, K, replace=False)
    # 5-approval breaks seat ties with its own random stream
    won["kapproval_5"] = top_k(approval_scores(order, 5), K, np.random.default_rng([seed, 5]))
    return np.mean(V[:, 0] < 0), {r: float(np.mean(C[won[r], 0] < 0)) for r in RULES}


def run():
    with Pool() as pool:
        out = pool.map(trial, [(w, t) for w in WS for t in range(TRIALS)])
    res = []
    for i, w in enumerate(WS):
        block = out[i * TRIALS:(i + 1) * TRIALS]
        res.append(dict(w=w, vote_share=float(np.mean([b[0] for b in block])),
                        seat_share={r: float(np.mean([b[1][r] for b in block])) for r in RULES},
                        seat_share_sd={r: float(np.std([b[1][r] for b in block])) for r in RULES}))
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    json.dump(res, open(DATA, "w"), indent=1)


if __name__ == "__main__":
    run()
