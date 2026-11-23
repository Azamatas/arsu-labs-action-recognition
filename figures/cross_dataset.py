"""Cross-dataset figures for one direction: test accuracy on the other dataset after every epoch (mean +- s.d. over the
seeds) and confusion matrices (counts pooled over the seeds).

    python figures/cross_dataset.py m2w            # figures/out/m2w_curves.eps, m2w_confusion.eps
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from data_processing.activities import PAPER, SHORT  # noqa: E402
from figures import style  # noqa: E402

MODELS = [("bifin-w64-k61-e64-h128-acc40-keras", "FIN + 40, pretrained", style.BLUE),
          ("bifin-w64-k61-e64-h128-acc40-random-keras", "FIN + 40, random init", style.ORANGE),
          ("cnn_ignatov-acc40-keras", "CNN + 40 statistical", style.PURPLE)]
TARGET = {"w2m": "MotionSense", "m2w": "WISDM"}


def plot(direction, models=MODELS, png_dir=None):
    plt = style.setup(**{"font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "lines.linewidth": 1.3})
    target = TARGET[direction]
    data = {m: style.runs("cross_dataset", direction, m) for m, *_ in models}

    fig, ax = plt.subplots(figsize=(3.15, 1.95))
    curves = {m: np.array([a[f"curve|{target}"] for _, a in rs]) for m, rs in data.items()}
    x = np.arange(1, next(iter(curves.values())).shape[1] + 1)
    def sd(c):
        return c.std(0, ddof=1) if len(c) > 1 else np.zeros(c.shape[1])

    style.bands(ax, x, [(c.mean(0) - sd(c), c.mean(0) + sd(c), col) for (m, _, col), c in
                        zip(models, curves.values())])
    for (m, label, col) in models:
        ax.plot(x, curves[m].mean(0), color=col, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"test accuracy ({target})")
    ax.set_xlim(1, x[-1])
    ax.grid(axis="y", color=style.GRID, linewidth=0.4, zorder=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    style.save(fig, f"{direction}_curves", png_dir, bbox_inches="tight")

    short = [SHORT[a] for a in PAPER]
    k = len(PAPER)
    fig, axes = plt.subplots(1, len(models), figsize=(4.72, 1.75), squeeze=False)
    for ax, (m, label, _) in zip(axes.ravel(), models):
        counts = np.zeros((k, k))
        for _, a in data[m]:
            np.add.at(counts, (a[f"truth|{target}"], a[f"pred|{target}"]), 1)
        share = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{100 * share[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if share[i, j] > 0.6 else "black")
        ax.set_title(label)
        ax.set_xticks(range(k), short, rotation=90)
        ax.set_yticks(range(k), short)
        ax.set_xlabel("predicted")
        ax.tick_params(length=0)
    axes[0, 0].set_ylabel("true")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    style.save(fig, f"{direction}_confusion", png_dir, bbox_inches="tight")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("direction", choices=list(TARGET))
    parser.add_argument("--png")
    args = parser.parse_args()
    plot(args.direction, png_dir=args.png)
