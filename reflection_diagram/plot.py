# schematic for the proof of theorem 4.1 (figure 13): (a) the partition of the disc by reflection
# across the bisector of two candidate positions x, y on a common ray, (b) and (c) the annuli
# compared for a reflected pair of voters z, z'. writes figures/reflection.pdf; there is no data.
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import plotting as pl

WIDTH = pl.TEXTWIDTH
ANGLE = np.deg2rad(35)
T, TP = 0.25, 0.65
C = (T + TP) / 2
ZP_LOCAL = np.array([0.74, 0.56])

# positions are given in local coordinates (along the ray, across the ray)
U = np.array([np.cos(ANGLE), np.sin(ANGLE)])
V = np.array([-np.sin(ANGLE), np.cos(ANGLE)])


def world(p):
    p = np.atleast_2d(p)
    return p[:, :1] * U + p[:, 1:2] * V


def reflect_local(p):
    p = np.array(p, float).copy()
    p[..., 0] = 2 * C - p[..., 0]
    return p


def cap_local(n=200):
    # boundary of the part of the disc beyond the bisector
    phi = np.arccos(C)
    th = np.linspace(-phi, phi, n)
    return np.c_[np.cos(th), np.sin(th)]


def base(ax):
    ax.add_patch(Circle((0, 0), 1, fill=False, lw=0.9, color=pl.INK, zorder=3))
    h = np.sqrt(1 - C ** 2)
    ends = world([[C, -h], [C, h]])
    ax.plot(ends[:, 0], ends[:, 1], ls=(0, (4, 2.5)), lw=0.8, color=pl.INK, zorder=3)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")


def point(ax, p, label, offset, **kw):
    q = world(p)[0]
    ax.plot(*q, "o", ms=2.6, color=kw.pop("color", pl.INK), zorder=5)
    ax.annotate(label, q, xytext=offset, textcoords="offset points", fontsize=7.5, zorder=6, **kw)


def panel_partition(ax, zp, z):
    base(ax)
    ax.add_patch(Polygon(world(cap_local()), closed=True, fc=pl.TERRA, alpha=0.18, lw=0, zorder=1))
    ax.add_patch(Polygon(world(reflect_local(cap_local())), closed=True, fc=pl.TEAL, alpha=0.18, lw=0, zorder=1))
    arc = world(reflect_local(cap_local()))
    ax.plot(arc[:, 0], arc[:, 1], ls=(0, (4, 2.5)), lw=0.8, color=pl.INK, zorder=3)
    ray = world([[0, 0], [1, 0]])
    ax.plot(ray[:, 0], ray[:, 1], ls=":", lw=0.7, color=pl.MUTED, zorder=2)
    pair = world([z, zp])
    ax.plot(pair[:, 0], pair[:, 1], lw=0.6, color=pl.MUTED, zorder=2)
    point(ax, [0, 0], "$0$", (-9, 2))
    point(ax, [T, 0], "$x$", (3, -8))
    point(ax, [TP, 0], "$y$", (3, -8))
    point(ax, z, "$z$", (-8, 2), color=pl.TEAL)
    point(ax, zp, "$z'$", (3, 2), color=pl.TERRA)
    for lab, p in [("$R_y$", [0.86, -0.18]), ("$R_x$", [0.12, -0.48]), ("$Q$", [-0.62, -0.08]),
                   ("$L_{xy}$", [C + 0.13, -np.sqrt(1 - C ** 2) - 0.08])]:
        ax.text(*world(p)[0], lab, fontsize=7.5, ha="center", va="center", zorder=6)
    ax.set_title("(a)", fontsize=7.5)


def panel_annulus(ax, center, r1, r2, color, label, title):
    base(ax)
    q = world(center)[0]
    ring = Wedge(q, r2, 0, 360, width=r2 - r1, fc=color, alpha=0.35, lw=0, zorder=1)
    ax.add_patch(ring)
    ring.set_clip_path(Circle((0, 0), 1, transform=ax.transData))
    for r in (r1, r2):
        inside = Circle(q, r, fill=False, lw=0.7, color=color, zorder=2)
        ax.add_patch(inside)
        inside.set_clip_path(Circle((0, 0), 1, transform=ax.transData))
        ax.add_patch(Circle(q, r, fill=False, lw=0.5, ls=(0, (1, 1.5)), color=color, alpha=0.6, zorder=2))
    point(ax, [T, 0], "$x$", (3, -8))
    point(ax, [TP, 0], "$y$", (3, -8))
    point(ax, center, label, (3, 2), color=color)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_title(title, fontsize=7.5)


if __name__ == "__main__":
    zp = ZP_LOCAL
    z = reflect_local(zp)
    r1, r2 = np.linalg.norm(z - np.array([T, 0])), np.linalg.norm(z - np.array([TP, 0]))
    with plt.rc_context(pl.print_rc()):
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH / 3 * 1.08))
        panel_partition(axes[0], zp, z)
        panel_annulus(axes[1], z, r1, r2, pl.TEAL, "$z$", "(b)")
        panel_annulus(axes[2], zp, r1, r2, pl.TERRA, "$z'$", "(c)")
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.92, wspace=0.06)
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        fig.savefig(os.path.join(HERE, "figures", "reflection.pdf"))
    plt.close(fig)
