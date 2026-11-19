# draws figures 1 and 2 from data/borda_surfaces.npz into figures/: borda_disc.pdf,
# borda_square.pdf and square_contours.pdf.
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import plotting as pl

DISC_WIDTH, SQUARE_WIDTH, CONTOUR_WIDTH = 0.62 * pl.TEXTWIDTH, 0.55 * pl.TEXTWIDTH, 0.4 * pl.TEXTWIDTH
FIGURES = os.path.join(HERE, "figures")


def surface_plot(X1, X2, F, fname, width):
    # the 3d panel's labels stick out, so the figure comes out about 10% wider than width and
    # latex shrinks it; fonts are 10% larger to compensate
    rc = pl.print_rc()
    rc.update({key: 1.1 * rc[key] for key in ("xtick.labelsize", "ytick.labelsize", "axes.labelsize", "axes.titlesize")})
    with plt.rc_context(rc):
        height = 0.5 * width
        fig = plt.figure(figsize=(width, height))
        ax1 = fig.add_axes([-0.07, -0.09, 0.62, 0.99], projection="3d")
        ax1.plot_surface(X1, X2, F, cmap="viridis", edgecolor="none", antialiased=True, rasterized=True)
        ax1.set_xticks([-1, 0, 1])
        ax1.set_yticks([-1, 0, 1])
        ax1.zaxis.set_major_locator(MaxNLocator(4))
        for axis in ("x", "y", "z"):
            ax1.tick_params(axis=axis, pad=-2)
        ax1.set_xlabel("$x_1$", labelpad=-5)
        ax1.set_ylabel("$x_2$", labelpad=-5)
        side = 0.64 * height
        x0, y0 = 0.655, 0.16
        ax2 = fig.add_axes([x0, y0, side / width, side / height])
        im = ax2.imshow(F, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis")
        cax = fig.add_axes([x0 + side / width + 0.025, y0, 0.022, side / height])
        cb = fig.colorbar(im, cax=cax)
        cb.locator = MaxNLocator(5)
        cb.update_ticks()
        cb.outline.set_linewidth(0.5)
        ax2.set_xticks([-1, 0, 1])
        ax2.set_yticks([-1, 0, 1])
        ax2.set_xlabel("$x_1$", labelpad=1)
        ax2.set_ylabel("$x_2$", labelpad=0)
        title_y = y0 + side / height + 0.04
        for x, title in [(0.25, "3D"), (x0 + side / width / 2, "Heatmap")]:
            fig.text(x, title_y, title, ha="center", va="bottom", fontsize=plt.rcParams["axes.titlesize"])
        fig.savefig(os.path.join(FIGURES, fname), dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def contour_plot(X1, X2, F):
    with plt.rc_context(pl.print_rc()):
        fig, ax = plt.subplots(figsize=(CONTOUR_WIDTH, CONTOUR_WIDTH))
        CS = ax.contour(X1, X2, F, levels=14, cmap="viridis", linewidths=0.8)
        ax.clabel(CS, CS.levels[::2], inline=True, fontsize=5, fmt="%1.2f", inline_spacing=2)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(True)
        ax.set_aspect("equal")
        fig.savefig(os.path.join(FIGURES, "square_contours.pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


if __name__ == "__main__":
    data = np.load(os.path.join(HERE, "data", "borda_surfaces.npz"))
    os.makedirs(FIGURES, exist_ok=True)
    X1, X2 = np.meshgrid(data["g_disc"], data["g_disc"])
    surface_plot(X1, X2, data["F_disc"], "borda_disc.pdf", DISC_WIDTH)
    X1, X2 = np.meshgrid(data["g_square"], data["g_square"])
    surface_plot(X1, X2, data["F_square"], "borda_square.pdf", SQUARE_WIDTH)
    contour_plot(X1, X2, data["F_square"])
