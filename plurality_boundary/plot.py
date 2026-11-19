# draws figure 4 (normalized_plurality.pdf and boundary_profile.pdf) and figure 10
# (boundary_level_sets.pdf, level sets of the area function in the boundary profile) into figures/.
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import plotting as pl
from run import boundary_area

SCORE_WIDTH, PROFILE_WIDTH, LEVEL_SETS_WIDTH = 0.52 * pl.TEXTWIDTH, 0.46 * pl.TEXTWIDTH, 0.6 * pl.TEXTWIDTH
PANEL_HEIGHT = 1.85
FIGURES = os.path.join(HERE, "figures")


def plot_normalized_score(data):
    plt.style.use("default")
    pl.vector_fonts()
    m_values = data["m_values"]
    with plt.rc_context(pl.print_rc()):
        fig, ax = plt.subplots(figsize=(SCORE_WIDTH, PANEL_HEIGHT))
        for m, row, color in zip(m_values, data["scores"], plt.cm.viridis(np.linspace(0, 1, len(m_values)))):
            ax.plot(data["xs"], row, label=f"m = {m}", color=color, linewidth=1.2)
        ax.set_xlabel("Distance from center $r$")
        ax.set_ylabel(r"$P(r) / (n/m)$")
        ax.set_title("Share of Plurality Points Relative to Mean")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.legend(title="Number of Candidates", loc="lower left", frameon=True, fontsize=5.5, title_fontsize=5.5,
                  labelspacing=0.2, handlelength=1.5, borderpad=0.4)
        fig.tight_layout(pad=0.2)
        fig.savefig(os.path.join(FIGURES, "normalized_plurality.pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def plot_boundary_profile(data):
    pl.paper_style()
    ds, Ps = data["ds"], data["profile"]
    with plt.rc_context({**pl.print_rc(), "axes.edgecolor": pl.MUTED}):
        fig, ax = plt.subplots(figsize=(PROFILE_WIDTH, PANEL_HEIGHT))
        ax.plot(ds, Ps, color=pl.TEAL, linewidth=1.2)
        ax.axhline(1, color=pl.GRAY, linestyle="--", linewidth=0.8)
        i = int(np.argmax(Ps))
        ax.plot(ds[i], Ps[i], "o", color=pl.TEAL, markersize=3)
        ax.annotate(rf"Max $\approx {Ps[i]:.2f}$ at $d \approx {ds[i]:.1f}$", (ds[i], Ps[i]), xytext=(2.2, 0.8),
                    fontsize=6.5, arrowprops=dict(arrowstyle="-", color=pl.MUTED, linewidth=0.6))
        ax.text(4.95, 1.015, "Bulk value", ha="right", va="bottom", fontsize=6.5, color=pl.MUTED)
        ax.set_xlabel(r"Distance $d$ to the boundary (rescaled)")
        ax.set_ylabel(r"Normalized plurality share $P(d)$")
        pl.light_grid(ax)
        fig.tight_layout(pad=0.2)
        fig.savefig(os.path.join(FIGURES, "boundary_profile.pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    plt.rcdefaults()
    pl.vector_fonts()


def plot_level_sets():
    uu, vv = np.meshgrid(np.linspace(-4, 4, 801), np.linspace(0, 4, 401))
    with plt.rc_context(pl.print_rc()):
        fig, axes = plt.subplots(2, 2, figsize=(LEVEL_SETS_WIDTH, 0.7 * LEVEL_SETS_WIDTH))
        for ax, d in zip(axes.flatten(), [0.5, 1.0, 1.5, 2.0]):
            ax.contour(uu, vv, boundary_area(uu, vv, d), levels=0.35 * np.arange(1, 15), cmap="viridis", linewidths=0.8)
            ax.set_title(f"d = {d}")
            ax.set_xlabel("x", labelpad=1)
            ax.set_ylabel("y", labelpad=1)
            ax.set_aspect("equal")
            ax.grid(True, color="#e6e5e0", linewidth=0.5)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
        fig.tight_layout(pad=0.2, h_pad=1.2, w_pad=0.6)
        fig.savefig(os.path.join(FIGURES, "boundary_level_sets.pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


if __name__ == "__main__":
    data = np.load(os.path.join(HERE, "data", "plurality_boundary.npz"))
    os.makedirs(FIGURES, exist_ok=True)
    plot_normalized_score(data)
    plot_boundary_profile(data)
    plot_level_sets()
