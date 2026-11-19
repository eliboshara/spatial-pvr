# shared figure settings. figures are drawn at the width at which the paper prints them
# (a fraction of the acmsmall text width), so font sizes survive inclusion in latex.
# fonts are embedded as truetype, which acm production requires.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEXTWIDTH = 395.8225 / 72

TEAL, TERRA, VIOLET, OCHRE = "#00969e", "#bf4b2a", "#6a4ba3", "#a5821a"
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
# rule hues re-stepped so that every pair stays distinguishable under color-vision deficiency
RULE_TERRA, RULE_OCHRE, RULE_CHARCOAL = "#b5401f", "#b8921c", "#454440"

RULE_LABEL = {"sntv": "SNTV", "kapproval_5": "$K$-approval, $K = 5$", "kapproval_m10": "$K$-approval, $K = m/10$",
              "kborda": "$k$-Borda", "random_committee": "Random committee"}
RULE_COLOR = {"sntv": RULE_TERRA, "kapproval_5": VIOLET, "kapproval_m10": VIOLET, "kborda": RULE_OCHRE,
              "random_committee": RULE_CHARCOAL}


def vector_fonts():
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


vector_fonts()


def print_rc():
    return {"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "figure.titlesize": 8, "lines.linewidth": 1.0,
            "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
            "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2, "grid.linewidth": 0.5,
            "axes.titlepad": 3, "axes.labelpad": 2, "legend.handlelength": 1.8, "pdf.fonttype": 42,
            "ps.fonttype": 42}


def multipanel_rc():
    # print_rc with minor ticks and no top or right spines
    return {**print_rc(), "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED}


def paper_style():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "lines.linewidth": 1.6, "savefig.bbox": "tight", "pdf.fonttype": 42})


def light_grid(ax, linewidth=0.6):
    ax.grid(True, color="#e6e5e0", linewidth=linewidth)
    ax.set_axisbelow(True)


def _srgb_to_oklab(hexcol):
    c = np.array([int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lms = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]]) @ c
    return np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                     [0.0259040371, 0.7827717662, -0.8086757660]]) @ np.cbrt(lms)


def _oklab_to_hex(lab):
    lms = np.array([[1, 0.3963377774, 0.2158037573], [1, -0.1055613458, -0.0638541728],
                    [1, -0.0894841775, -1.2914855480]]) @ lab
    c = np.array([[4.0767416621, -3.3077115913, 0.2309699292], [-1.2684380046, 2.6097574011, -0.3413193965],
                  [-0.0041960863, -0.7034186147, 1.7076147010]]) @ lms ** 3
    c = np.clip(np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055), 0, 1)
    return "#" + "".join(f"{round(255 * v):02x}" for v in c)


def ramp(hexcol, lightness=(0.76, 0.57, 0.40)):
    # one hue from light to dark, at the given oklab lightness values
    L0, a, b = _srgb_to_oklab(hexcol)
    scale = np.interp(lightness, [0.40, 0.57, 0.76], [0.85, 1.0, 0.8])
    return [_oklab_to_hex(np.array([L, a * c, b * c])) for L, c in zip(lightness, scale)]
