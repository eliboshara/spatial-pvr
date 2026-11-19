# draws figure 5 from data/score_profiles.npz into figures/score_profiles.pdf.
# the wide gray band is the limit p_v/p_c of the local rules theorem.
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import plotting as pl
from common.elections import candidate_density

RULES = ["sntv", "kapproval_5", "kapproval_m10", "kborda"]
M_VALUES = [10, 100, 500, 2000]
M_STYLE = {10: (0, (1, 1.3)), 100: (0, (5, 1.5, 1, 1.5)), 500: (0, (4, 1.5)), 2000: "-"}
LIMIT = "#c9c8c3"
YTICKS = [0.25, 0.5, 1, 2, 4]
LIGHTNESS = (0.72, 0.61, 0.50, 0.36)


def limit(r, cand):
    return (1 / np.pi) / candidate_density(r, cand)


def panel(ax, res, cand, rule, cols, show_limit):
    r = np.linspace(0, 1, 400)
    if show_limit:
        ax.plot(r, limit(r, cand), color=LIMIT, linewidth=3.2, solid_capstyle="butt", zorder=1)
    ax.axhline(1, color=pl.MUTED, linewidth=0.5, zorder=1)
    for m, col in zip(M_VALUES, cols):
        ax.stairs(res[f"{cand}_{m}_{rule}_mean"], res["edges"], color=col, linestyle=M_STYLE[m], linewidth=1.1,
                  baseline=None, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 7)
    ax.yaxis.set_major_locator(FixedLocator(YTICKS))
    ax.set_yticklabels([f"{t:g}" for t in YTICKS])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0, 1)
    pl.light_grid(ax, linewidth=0.5)


def plot(res):
    plt.rcdefaults()
    with plt.rc_context({**pl.multipanel_rc(), "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "axes.titlesize": 7}):
        fig, axes = plt.subplots(2, 4, figsize=(pl.TEXTWIDTH, 3.0), sharex=True, sharey=True)
        for j, rule in enumerate(RULES):
            cols = pl.ramp(pl.RULE_COLOR[rule], lightness=LIGHTNESS)
            for i, cand in enumerate(["unif", "gauss"]):
                panel(axes[i, j], res, cand, rule, cols, show_limit=rule in ("sntv", "kapproval_5"))
                axes[i, j].set_xticks([0, 0.5, 1])
                axes[i, j].set_xticklabels(["0", "0.5", "1"])
            axes[0, j].set_title(pl.RULE_LABEL[rule])
        axes[0, 0].set_ylabel("Uniform candidates")
        axes[1, 0].set_ylabel("Gaussian candidates")
        fig.supylabel("Mean normalized score $(m/n)\\,V_m$", fontsize=7, x=0.0)
        fig.supxlabel("Distance $r$ of the candidate from the center", fontsize=7, y=0.0)
        gray = pl.ramp("#454440", lightness=LIGHTNESS)
        handles = [Line2D([], [], color=c, linestyle=M_STYLE[m], linewidth=1.15, label=f"$m = {m}$")
                   for m, c in zip(M_VALUES, gray)]
        handles.append(Line2D([], [], color=LIMIT, linewidth=3.2, solid_capstyle="butt", label=r"$p_v/p_c$"))
        fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0),
                   handlelength=2.4, columnspacing=1.6, fontsize=6)
        fig.tight_layout(pad=0.3, rect=(0.01, 0.02, 1, 0.93), w_pad=0.5, h_pad=0.6)
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        fig.savefig(os.path.join(HERE, "figures", "score_profiles.pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


if __name__ == "__main__":
    plot(dict(np.load(os.path.join(HERE, "data", "score_profiles.npz"))))
