# spatial elections: sampling voters and candidates, distance rankings, and positional rules.
# every voter ranks the candidates by euclidean distance. a committee rule elects the k
# candidates with the highest scores, with ties broken uniformly at random.
import numpy as np

# 95% of the mass of this gaussian lies in the unit disc, so its density at the rim is 5% of the center
GAUSS_SIGMA = np.sqrt(-1.0 / (2.0 * np.log(0.05)))


def sample_disc(n, rng):
    r, th = np.sqrt(rng.random(n)), 2 * np.pi * rng.random(n)
    return np.c_[r * np.cos(th), r * np.sin(th)]


def sample_gaussian_disc(n, rng, sigma=GAUSS_SIGMA):
    # rejection sampling of the gaussian conditioned on the unit disc
    out, tot = [], 0
    while tot < n:
        P = rng.normal(0.0, sigma, size=(2 * n + 10, 2))
        P = P[(P ** 2).sum(1) <= 1.0]
        out.append(P)
        tot += len(P)
    return np.concatenate(out)[:n]


def sample_square(n, rng):
    return rng.uniform(-1.0, 1.0, size=(n, 2))


def sample_two_blocs(n, w, rng, centers=(-0.5, 0.5), sigma=0.2):
    # a fraction w of the voters around (-0.5, 0), the rest around (0.5, 0), restricted to [-1,1]^2
    out, tot = [], 0
    while tot < n:
        left = rng.random(2 * n) < w
        P = rng.normal(0.0, sigma, size=(2 * n, 2))
        P[:, 0] += np.where(left, centers[0], centers[1])
        P = P[(np.abs(P) <= 1).all(1)]
        out.append(P)
        tot += len(P)
    return np.concatenate(out)[:n]


def candidate_density(r, cand):
    # density per unit area at distance r from the center, for "unif" or "gauss" candidates
    if cand == "unif":
        return np.full_like(r, 1 / np.pi)
    s2 = GAUSS_SIGMA ** 2
    return np.exp(-r ** 2 / (2 * s2)) / (2 * np.pi * s2 * (1 - np.exp(-1 / (2 * s2))))


def rankings(V, C):
    # rankings[i] lists the candidates in the order of preference of voter i
    D = ((V[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return np.argsort(D, axis=1)


def top_k(scores, k, rng):
    return np.lexsort((rng.random(len(scores)), -scores))[:k]


def approval_scores(order, K):
    # each voter approves their nearest candidates, as many as the approval count
    return np.bincount(order[:, :K].ravel(), minlength=order.shape[1]).astype(float)


def borda_scores(order):
    m = order.shape[1]
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(m)[None, :], axis=1)
    return (m - 1 - ranks).sum(0).astype(float)


def sntv(order, k, rng):
    return top_k(approval_scores(order, 1), k, rng)


def k_approval(order, K, k, rng):
    return top_k(approval_scores(order, K), k, rng)


def k_borda(order, k, rng):
    return top_k(borda_scores(order), k, rng)
