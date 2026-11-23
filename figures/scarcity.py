"""Data scarcity: test accuracy on the unseen users after every epoch, mean +- s.d. over the seeds.

    python figures/scarcity.py                     # figures/out/scarcity.eps
    python figures/scarcity.py --models bifin-w64-k61-e64-h128-acc40 cnn_ignatov-acc40 --png DIR
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from figures import style  # noqa: E402

DEFAULT = [  # (model id in results/scarcity/users7, label, line colour, band colour)
    ("bifin-w64-k61-e64-h128-acc40-keras", "BiFIN + 40, pretrained", style.BLUE, "#dae8fc"),
    ("cnn_ignatov-acc40-keras", "CNN (+40 stat.)", style.GREEN, "#d5e8d4"),
    ("bifin-w64-k61-e64-h128-none-random-keras", "BiFIN, random init", style.GRAY, "#eaeae9"),
]  # as in the ESANN paper; the like-for-like comparison is bifin-w64-k61-e64-h128-acc40-random-keras
XMAX = 50


def plot(curves=DEFAULT, variant="users7", png_dir=None):
    plt = style.setup(**{"axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
    import matplotlib

    x = np.arange(1, XMAX + 1)
    stats = []
    for model, *_ in curves:
        c = np.array([a["curve|unseen users"][:XMAX] for _, a in style.runs("scarcity", variant, model)])
        if not len(c):
            raise SystemExit(f"no results for {model} in results/scarcity/{variant}")
        stats.append((c.mean(0), c.std(0, ddof=1) if len(c) > 1 else np.zeros(XMAX)))
    fig, ax = plt.subplots(figsize=(5.86 * style.CM, 6.3 * style.CM))
    style.bands(ax, x, [(m - s, m + s, band) for (m, s), (*_, band) in zip(stats, curves)])
    for (m, _), (_, label, line, _) in zip(stats, curves):
        ax.plot(x, m, color=line, label=label)
    ax.set_title("Test accuracy on 27 unseen users", pad=5)
    ax.set_xlabel("epoch")
    ax.set_xlim(1, XMAX)
    ax.set_ylim(0.6, 0.92)
    ax.set_xticks([1, *range(10, XMAX + 1, 10)])
    ax.set_yticks(np.arange(0.6, 0.91, 0.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".lstrip("0")))
    ax.grid(axis="y", color=style.GRID, linewidth=0.4, zorder=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    fig.tight_layout(pad=0.15)
    style.save(fig, "scarcity", png_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="model ids; default: the paper's three curves")
    parser.add_argument("--png")
    args = parser.parse_args()
    palette = [(style.BLUE, "#dae8fc"), (style.GREEN, "#d5e8d4"), (style.GRAY, "#eaeae9"), (style.ORANGE, "#fbe1d4")]
    chosen = DEFAULT if not args.models else [(m, m, *palette[i % 4]) for i, m in enumerate(args.models)]
    plot(chosen, png_dir=args.png)
