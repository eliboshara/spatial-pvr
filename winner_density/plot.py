# draws figure 7 from data/winner_density.npz into figures/winner_density.pdf:
# winners per unit area on 50 equal-area annuli, rows by candidate distribution, columns by rule.
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import plotting as pl
from common.elections import GAUSS_SIGMA, candidate_density

RULES = ["sntv", "kapproval_m10", "kborda", "random_committee"]
M_VALUES = [10, 100, 500]
M_STYLE = {10: (0, (1, 1.3)), 100: (0, (4, 1.5)), 500: "-"}
BINS = 50
DASH = (0, (4, 2))


def pv_over_pc(r, cand):
    # voter density over candidate density, normalized to integrate to one over the disc
    if cand == "unif":
        return np.full_like(r, 1 / np.pi)
    s2 = GAUSS_SIGMA ** 2
    return np.exp(r ** 2 / (2 * s2)) / (2 * np.pi * s2 * (np.exp(1 / (2 * s2)) - 1))


def density(radii):
    edges = np.sqrt(np.linspace(0, 1, BINS + 1))
    counts, _ = np.histogram(radii, bins=edges)
    return edges, counts / (len(radii) * np.pi / BINS)


def plot(res):
    plt.rcdefaults()
    r = np.linspace(0, 1, 400)
    with plt.rc_context({**pl.multipanel_rc(), "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "axes.titlesize": 7}):
        fig, axes = plt.subplots(2, 4, figsize=(pl.TEXTWIDTH, 2.8), sharex=True)
        for j, rule in enumerate(RULES):
            cols = pl.ramp(pl.RULE_COLOR[rule])
            for i, cand in enumerate(["unif", "gauss"]):
                ax = axes[i, j]
                ax.axhline(1 / np.pi, color=pl.GRAY, linestyle=(0, (1, 1.5)), linewidth=0.8)
                if cand == "gauss" and rule in ("sntv", "random_committee"):
                    ref = pv_over_pc(r, cand) if rule == "sntv" else candidate_density(r, cand)
                    ax.plot(r, ref, color=pl.GRAY, linestyle=DASH, linewidth=0.9)
                for m, col in zip(M_VALUES, cols):
                    edges, dens = density(res[f"{cand}_{m}_{rule}"])
                    ax.stairs(dens, edges, color=col, linewidth=1.05, baseline=None, linestyle=M_STYLE[m])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, None)
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels(["0", "0.5", "1"])
                pl.light_grid(ax, linewidth=0.5)
                if i == 0:
                    ax.set_title(pl.RULE_LABEL[rule])
        axes[0, 0].set_ylabel("Uniform candidates")
        axes[1, 0].set_ylabel("Gaussian candidates")
        fig.supylabel("Winners per unit area", fontsize=7, x=0.0)
        fig.supxlabel("Distance $r$ from the center", fontsize=7, y=0.0)
        handles = [Line2D([], [], color=pl.MUTED, linestyle=M_STYLE[m], linewidth=1.1,
                          label=f"$m = {m}$, $k = {max(1, m // 10)}$") for m in M_VALUES]
        handles += [Line2D([], [], color=pl.GRAY, linestyle=(0, (1, 1.5)), linewidth=0.8, label=r"voters, $1/\pi$"),
                    Line2D([], [], color=pl.GRAY, linestyle=DASH, linewidth=0.9, label=r"$p_v/p_c$ (SNTV), $p_c$ (random)")]
        fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0),
                   handlelength=2.2, columnspacing=1.2, fontsize=6)
        fig.tight_layout(pad=0.3, rect=(0.01, 0.02, 1, 0.93), w_pad=0.6, h_pad=0.6)
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        fig.savefig(os.path.join(HERE, "figures", "winner_density.pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


if __name__ == "__main__":
    plot(dict(np.load(os.path.join(HERE, "data", "winner_density.npz"))))
