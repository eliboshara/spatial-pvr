# draws figures 8 and 9 from data/representativeness.npz into figures/:
# metrics_uniform.pdf and metrics_gaussian.pdf (figure 8), quadrant_histograms.pdf and
# quadrant_variance.pdf (figure 9).
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import plotting as pl

LABELS = {"random_voters": "Random Points", "kborda": "$k$-Borda", "kapproval_m10": "$K$-approval, $K = m/10$",
          "sntv": "SNTV"}
COLORS = {"random_voters": "gray", "kborda": "#e9c46a", "kapproval_m10": "#e76f51", "sntv": "#264653"}
METRICS = ["Average Pairwise Distance", "Minimum Pairwise Distance", "Dissatisfaction Score", "Smallest Wedge (Radians)"]
N, M_VALUES, TRIALS, QUAD_M = 500, [20, 100, 250, 500], 2500, 100
METRICS_WIDTH, HIST_WIDTH, VARIANCE_WIDTH = 0.48 * pl.TEXTWIDTH, 0.3 * pl.TEXTWIDTH, 0.62 * pl.TEXTWIDTH
SMALL = 5.5
FIGURES = os.path.join(HERE, "figures")


def plot_metrics(data, rules, cand):
    plt.rcdefaults()
    pl.vector_fonts()
    rc = {**pl.print_rc(), "axes.titlesize": 6, "axes.labelsize": 6, "xtick.labelsize": SMALL, "ytick.labelsize": SMALL}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(METRICS_WIDTH, 0.95 * METRICS_WIDTH))
        for idx, ax in enumerate(axes.flatten()):
            for rule in rules:
                vals = [data[f"{cand}_{m}"][:, rules.index(rule), idx] for m in M_VALUES]
                means = np.array([v.mean() for v in vals])
                lo = means - np.array([np.percentile(v, 1) for v in vals])
                hi = np.array([np.percentile(v, 99) for v in vals]) - means
                ax.errorbar(M_VALUES, means, yerr=[lo, hi], label=LABELS[rule], color=COLORS[rule], marker="o",
                            markersize=2.2, capsize=1.5, linewidth=0.9, elinewidth=0.6, capthick=0.6)
            ax.set_title(METRICS[idx], pad=2)
            if idx >= 2:
                ax.set_xlabel("Number of Candidates (m)", labelpad=1)
            ax.grid(True, linestyle="--", alpha=0.6)
        fig.suptitle(f"Spatial Grouping Metrics: {'Uniform' if cand == 'unif' else 'Gaussian'} Candidates", fontsize=7)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=SMALL,
                   handlelength=1.5, columnspacing=1.0, handletextpad=0.4)
        fig.tight_layout(rect=[0, 0.11, 1, 0.97], pad=0.2, h_pad=0.5, w_pad=0.5)
        fig.savefig(os.path.join(FIGURES, f"metrics_{'uniform' if cand == 'unif' else 'gaussian'}.pdf"),
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def plot_quadrant_histograms(data, rules):
    plt.rcdefaults()
    pl.vector_fonts()
    q = data["unif_quad"]
    order = ["random_voters", "sntv", "kapproval_m10", "kborda"]
    names = {**LABELS, "random_voters": "Random"}
    rc = {**pl.print_rc(), "axes.titlesize": 6, "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": SMALL}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(nrows=len(order), ncols=1, figsize=(HIST_WIDTH, 2.4), sharex=True)
        x_bins = np.arange(11)
        for ax, rule in zip(axes, order):
            pct = 100 * np.bincount(q[:, rules.index(rule), :].ravel(), minlength=11)[:11] / q[:, 0, :].size
            ax.bar(x_bins, pct, width=0.8, color="gray" if rule == "random_voters" else "lightskyblue",
                   edgecolor="none", alpha=0.85)
            ax.set_title(names[rule], pad=1.5)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            ax.set_axisbelow(True)
            ax.set_ylim(0, pct.max() + 5)
            ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        axes[-1].set_xlabel("Number of Winning\nCandidates per Quadrant", labelpad=1)
        axes[-1].set_xticks(x_bins)
        fig.supylabel("Percent (%)", fontsize=6, x=0.01)
        fig.suptitle(f"Quadrant Distribution of\nWinning Candidates\n(n={N}, m={QUAD_M}, k={QUAD_M // 10}, {TRIALS} Trials)",
                     fontsize=6.5)
        fig.tight_layout(rect=[0.04, 0, 1, 1], pad=0.2, h_pad=0.3)
        fig.savefig(os.path.join(FIGURES, "quadrant_histograms.pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def quadrant_variance(q):
    # variance of the four quadrant counts within an election, averaged over elections
    return q.var(axis=2).mean(axis=0)


def plot_quadrant_variance(data, rules):
    plt.rcdefaults()
    pl.vector_fonts()
    vu, vg = quadrant_variance(data["unif_quad"]), quadrant_variance(data["gauss_quad"])
    order = ["sntv", "kborda", "kapproval_m10"]
    x = np.arange(len(order))
    with plt.rc_context(pl.print_rc()):
        fig, ax = plt.subplots(figsize=(VARIANCE_WIDTH, 2.3))
        ax.axhline(vu[rules.index("random_voters")], color="gray", linestyle="--", linewidth=1.2,
                   label="Random Points Benchmark")
        ax.bar(x - 0.175, [vu[rules.index(r)] for r in order], width=0.35, color="#2a9d8f", label="Uniform Candidates")
        ax.bar(x + 0.175, [vg[rules.index(r)] for r in order], width=0.35, color="#e9c46a", label="Gaussian Candidates")
        ax.set_xticks(x, [LABELS[r] for r in order], fontsize=7)
        ax.set_ylabel("Variance")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"Average Variance of Winning Candidates in each Quadrant\n(n={N}, m={QUAD_M}, k={QUAD_M // 10}, {TRIALS} Trials)")
        ax.legend(loc="upper left")
        fig.tight_layout(pad=0.2)
        fig.savefig(os.path.join(FIGURES, "quadrant_variance.pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


if __name__ == "__main__":
    data = dict(np.load(os.path.join(HERE, "data", "representativeness.npz")))
    rules = list(data["rules"])
    os.makedirs(FIGURES, exist_ok=True)
    for cand in ["unif", "gauss"]:
        plot_metrics(data, rules, cand)
    plot_quadrant_histograms(data, rules)
    plot_quadrant_variance(data, rules)
