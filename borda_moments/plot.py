# draws figures 11 and 12 from data/borda_moments.npz into figures/: borda_mean.pdf and
# borda_variance.pdf (disc, simulated against exact), square_mean.pdf and square_variance.pdf.
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter, MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import plotting as pl

N_VALUES, M_VALUES = [100, 500, 1000], [2, 5, 10]
CURVES_WIDTH, GRID_WIDTH = 0.92 * pl.TEXTWIDTH, 0.48 * pl.TEXTWIDTH
FIGURES = os.path.join(HERE, "figures")


def exact_variance(p1, p2, p3, n, m):
    return n * (m - 1) * (p1 + (n - 1) * p2 + (m - 2) * p3 - (n + m - 2) * p1 ** 2)


def plot_curves(radii, sim, exact_fn, x_exact, ylabel, title, fname):
    with plt.rc_context(pl.print_rc()):
        fig, axes = plt.subplots(1, len(M_VALUES), figsize=(CURVES_WIDTH, 2.05), sharey=True)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(N_VALUES)))
        for ax, m in zip(axes, M_VALUES):
            for c, n in zip(colors, N_VALUES):
                ax.plot(radii, sim[(m, n)], label=rf"Simulated ($n={n}$)", color=c, linestyle="-")
                ax.plot(x_exact, exact_fn(n, m), label=rf"True ($n={n}$)", color=c, linestyle="--")
            ax.set_title(rf"$m = {m}$")
            ax.set_xlabel(r"$r$", labelpad=1)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.set_xlim(0, 1)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(title, fontweight="bold", y=1.03)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(N_VALUES), fancybox=True,
                   handlelength=2.2, columnspacing=1.5)
        fig.tight_layout(rect=[0, 0.16, 1, 1], pad=0.3, w_pad=0.6)
        fig.savefig(os.path.join(FIGURES, fname), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def plot_square_grid(results, title, fname, res=40):
    with plt.rc_context(pl.print_rc()):
        fig, axes = plt.subplots(len(M_VALUES), len(N_VALUES), figsize=(GRID_WIDTH, 0.93 * GRID_WIDTH),
                                 sharex=True, sharey=True)
        fig.suptitle(title, fontweight="bold", fontsize=7)
        for col, m in enumerate(M_VALUES):
            for row, n in enumerate(N_VALUES):
                ax = axes[row, col]
                im = ax.imshow(results[(m, n)].reshape(res, res), extent=[-1, 1, -1, 1], origin="lower", cmap="viridis")
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.locator = MaxNLocator(3)
                cb.formatter = EngFormatter(sep="")
                cb.update_ticks()
                cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1)
                cb.outline.set_linewidth(0.4)
                ax.set_xticks([-1, 0, 1])
                ax.set_yticks([-1, 0, 1])
                ax.tick_params(labelsize=5.5, length=1.5, pad=1)
                if col == 0:
                    ax.set_ylabel(rf"$n = {n}$", labelpad=1)
                if row == 0:
                    ax.set_title(rf"$m = {m}$", pad=2)
        fig.supxlabel(r"$x_1$", fontsize=7, y=0.005)
        fig.supylabel(r"$x_2$", fontsize=7, x=0.005)
        fig.tight_layout(rect=[0.03, 0.03, 1, 1], pad=0.2, h_pad=0.3, w_pad=0.1)
        fig.savefig(os.path.join(FIGURES, fname), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def plot(data):
    unpack = lambda name: {(m, n): data[f"{name}_{m}_{n}"] for m in M_VALUES for n in N_VALUES}
    radii, xs, base, r_vals = data["radii"], data["xs"], data["base"], data["r_vals"]
    p1, p2, p3 = data["p1"], data["p2"], data["p3"]
    os.makedirs(FIGURES, exist_ok=True)
    plot_curves(radii, unpack("mean"), lambda n, m: n * (m - 1) / np.pi ** 2 * base, xs, "Expectation",
                "Monte Carlo Estimation vs. Theoretical Expectation", "borda_mean.pdf")
    plot_curves(radii, unpack("var"), lambda n, m: exact_variance(p1, p2, p3, n, m), r_vals, "Variance",
                "Monte Carlo Estimation vs. Theoretical Variance", "borda_variance.pdf")
    plot_square_grid(unpack("mean_sq"), "Monte Carlo Estimation of Expected Borda Score\nin the Unit Square",
                     "square_mean.pdf")
    plot_square_grid(unpack("var_sq"), "Monte Carlo Estimation of Variance of Borda Score\nin the Unit Square",
                     "square_variance.pdf")


if __name__ == "__main__":
    plot(dict(np.load(os.path.join(HERE, "data", "borda_moments.npz"))))
