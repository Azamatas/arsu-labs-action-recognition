import argparse
import os
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.join(HERE, "data_scarcity_curves.npz")
OUT = os.path.join(HERE, "scarcity_panel_cm.eps")
CM = 1 / 2.54
SIZE = (5.86 * CM, 5.46 * 1.15 * CM)
XMAX = 50

CURVE = [("fin40_pretrained", "FIN + 40, pretrained", "#2a78d6", "#dae8fc"),
         ("ignatov_cnn", "CNN (+40 stat.)", "#82b366", "#d5e8d4"),
         ("fin_random", "FIN, random init", "#8a8a86", "#eaeae9")]


def bands(ax, x, items):
    from matplotlib.colors import to_rgb

    for k in range(1, len(items) + 1):
        for subset in combinations(items, k):
            low = np.max([b[0] for b in subset], axis=0)
            high = np.min([b[1] for b in subset], axis=0)
            colour = np.prod([to_rgb(b[2]) for b in subset], axis=0)
            ax.fill_between(x, low, high, where=high > low, interpolate=True, color=colour, linewidth=0, zorder=0.5)


def plot(curves=CURVES, out=OUT, png_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False, "ps.fonttype": 42,
        "font.size": 8, "axes.titlesize": 10, "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    })
    data = np.load(curves)
    x = np.arange(1, XMAX + 1)
    stats = [(data[k][:, :XMAX].mean(0), data[k][:, :XMAX].std(0, ddof=1)) for k, *_ in CURVE]

    fig, ax = plt.subplots(figsize=SIZE)
    bands(ax, x, [(m - s, m + s, band) for (m, s), (*_, band) in zip(stats, CURVE)])
    for (m, _), (_, label, line, _) in zip(stats, CURVE):
        ax.plot(x, m, color=line, label=label)
    ax.set_title("Test accuracy on 27 unseen users", pad=5)
    ax.set_xlabel("epoch")
    ax.set_xlim(1, XMAX)
    ax.set_ylim(0.6, 0.92)
    ax.set_xticks([1, *range(10, XMAX + 1, 10)])
    ax.set_yticks(np.arange(0.6, 0.91, 0.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".lstrip("0")))
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.4, zorder=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    fig.tight_layout(pad=0.15)
    fig.savefig(out)
    print("saved", out)
    if png_dir:
        png = os.path.join(png_dir, "scarcity_panel_cm.png")
        fig.savefig(png, dpi=300)
        print("saved", png)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="directory for a PNG preview")
    plot(png_dir=parser.parse_args().png)


if __name__ == "__main__":
    main()
