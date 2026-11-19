# draws figure 3 from data/borda_winner.npz into figures/borda_winner.pdf: the probability that
# the borda winner lies within distance r of the center, against the approximation 1 - (1 - r^2)^m.
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import plotting as pl

N, TRIALS, M_VALUES = 1000, 1000, [1, 5, 10, 25, 50, 100]
WIDTH = 0.5 * pl.TEXTWIDTH


def plot(data):
    r_eval = np.linspace(0, 1, 100)
    with plt.rc_context(pl.print_rc()):
        fig, ax = plt.subplots(figsize=(WIDTH, WIDTH))
        colors = plt.cm.viridis(np.linspace(0, 1, len(M_VALUES)))
        for m, color in zip(M_VALUES, colors):
            radii = data[f"radii_{m}"]
            cdf = np.array([np.mean(radii <= r) for r in r_eval])
            p = cdf[::4]
            ax.plot(r_eval, 1 - (1 - r_eval ** 2) ** m, color=color, zorder=1)
            ax.errorbar(r_eval[::4], p, yerr=np.sqrt(p * (1 - p) / TRIALS), fmt="o", color=color,
                        markersize=2.2, capsize=1.2, elinewidth=0.6, capthick=0.6, alpha=0.8, zorder=2)
        legend = [mpatches.Patch(color=c, label=f"m = {m}") for m, c in zip(M_VALUES, colors)]
        legend += [Line2D([0], [0], color="black", lw=1, label="Theoretical"),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=3.5, label="Simulated ±1 SD")]
        ax.legend(handles=legend, title="Number of candidates", loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=4, fontsize=5.5, title_fontsize=6, handlelength=1.2, handleheight=0.6, labelspacing=0.3,
                  columnspacing=0.9, borderpad=0.4)
        ax.set_xlabel("Radius $r$")
        ax.set_ylabel("Probability Borda Winner is within $B_r(0)$")
        ax.set_title(f"Borda Winning Probability:\nTheoretical vs. Simulated ($n={N}$ voters)")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.6)
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        fig.savefig(os.path.join(HERE, "figures", "borda_winner.pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


if __name__ == "__main__":
    plot(dict(np.load(os.path.join(HERE, "data", "borda_winner.npz"))))
