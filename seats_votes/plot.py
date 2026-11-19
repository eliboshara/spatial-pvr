# draws figure 6 from data/seats_votes.json into figures/seats_votes.pdf.
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import plotting as pl


def plot(res):
    plt.rcdefaults()
    pl.vector_fonts()
    width = 0.6 * pl.TEXTWIDTH
    rc = {**pl.print_rc(), "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": pl.MUTED}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(width, 0.74 * width))
        x = np.array([r["vote_share"] for r in res])
        ax.plot([0.45, 0.85], [0.45, 0.85], color=pl.GRAY, linestyle="--", linewidth=0.8)
        # both approval variants in violet, 5-approval dashed with hollow markers
        look = [("sntv", pl.TERRA, "s", "-", True, "SNTV"),
                ("kapproval_5", pl.VIOLET, "^", (0, (3, 1.5)), False, "$K$-approval, $K = 5$"),
                ("kapproval_m10", pl.VIOLET, "^", "-", True, "$K$-approval, $K = k = 30$"),
                ("kborda", pl.OCHRE, "D", "-", True, "$k$-Borda"),
                ("random_committee", pl.GRAY, "x", "-", True, "Random committee")]
        for rule, col, mk, ls, filled, label in look:
            y = np.array([r["seat_share"][rule] for r in res])
            e = np.array([r["seat_share_sd"][rule] for r in res])
            ax.errorbar(x, y, yerr=e, color=col, marker=mk, markersize=3.2, capsize=1.5, linewidth=1.0,
                        elinewidth=0.6, capthick=0.6, linestyle=ls, label=label,
                        markerfacecolor=col if filled else "white", markeredgewidth=0.8)
        ax.set_xlim(0.47, 0.83)
        ax.set_ylim(0.4, 1.03)
        p0, p1 = ax.transData.transform([(0.6, 0.6), (0.7, 0.7)])
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.text(0.725, 0.68, "Seats = votes", rotation=angle, rotation_mode="anchor", ha="center", va="top",
                fontsize=6, color=pl.MUTED)
        ax.set_xlabel("Vote share of the larger bloc")
        ax.set_ylabel("Seat share of the larger bloc")
        ax.legend(loc="upper left", frameon=False, fontsize=5.8, handlelength=2.2, borderaxespad=0.2)
        pl.light_grid(ax)
        fig.tight_layout(pad=0.2)
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        fig.savefig(os.path.join(HERE, "figures", "seats_votes.pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


if __name__ == "__main__":
    plot(json.load(open(os.path.join(HERE, "data", "seats_votes.json"))))
