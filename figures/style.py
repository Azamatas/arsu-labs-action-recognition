"""Shared look of the paper figures and helpers to read results/. Figures are saved as EPS in figures/out/ (transparent
background); --png DIR writes a preview as well."""
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
RESULTS = os.path.join(ROOT, "results")
CM = 1 / 2.54
BLUE, ORANGE, GREEN, PURPLE, GRAY, INK = "#2a78d6", "#eb6834", "#82b366", "#7a4fc9", "#8a8a86", "#3d3d3a"
GRID = "#e1e0d9"


def setup(**overrides):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False, "ps.fonttype": 42, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "lines.linewidth": 1.2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, **overrides})
    return plt


def save(fig, name, png_dir=None, **kw):
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{name}.eps")
    fig.savefig(out, facecolor="none", **kw)
    print("saved", out)
    if png_dir:
        fig.savefig(os.path.join(png_dir, f"{name}.png"), dpi=250, **kw)
    plt.close(fig)


def runs(protocol, variant, model):
    """[(json dict, npz arrays or None)] for every seed of one model."""
    out = []
    import json

    for f in sorted(glob.glob(os.path.join(RESULTS, protocol, variant, model, "seed*.json"))):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        npz = f[:-5] + ".npz"
        out.append((r, dict(np.load(npz)) if os.path.exists(npz) else None))
    return out


def bands(ax, x, items, alpha=None):
    """Overlapping mean +- s.d. regions; where bands overlap their colours multiply. items: (low, high, colour)."""
    from itertools import combinations

    from matplotlib.colors import to_rgb

    for k in range(1, len(items) + 1):
        for subset in combinations(items, k):
            low = np.max([b[0] for b in subset], axis=0)
            high = np.min([b[1] for b in subset], axis=0)
            colour = np.prod([to_rgb(b[2]) for b in subset], axis=0)
            ax.fill_between(x, low, high, where=high > low, interpolate=True, color=colour, linewidth=0, zorder=0.5)
