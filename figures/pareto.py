"""DAGHAR accuracy vs. cost per model family: x = MFLOPs per 6 x 60 window (log, reversed: cheaper to the right),
marker area grows with the number of parameters, the points of one family joined by a dashed line, each family named
in a coloured box. Mean target accuracy over the six targets (mean over seeds first).

    python figures/pareto.py           # figures/out/pareto.eps
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from data_processing import daghar  # noqa: E402
from figures import style  # noqa: E402

FAMILIES = [  # (label, colour, text colour, model ids in order along the dashed line, label position)
    ("BiTFIN (convolutions)", "#c7a5d6", "#2b2135",
     ["bifin-w64-k31-e64-h128-acc40-keras", "bifin-w32-k31-e64-h128-acc40-keras",
      "bifin-w16-k31-e64-h128-acc40-keras", "bifin-w16-k15-e64-h128-acc40-keras"], (0.95, 75.85)),
    ("BiTFIN as FFT + matrix", "#4a7fb0", "white",
     ["bifin-w32-k31-e64-h128-circ-acc40-keras", "bifin-w32-k31-e64-h64-circ-acc40-keras"], (0.105, 75.85)),
    ("TinyHAR", "#8fcca3", "#16301f", ["tinyhar-old"], (1.45, 73.72)),
    ("TinierHAR", "#f2c46b", "#3a2a06", ["tinierhar-old"], (0.19, 70.3)),
]
TINY_MFLOPS = {"tinyhar": 2.49, "tinierhar": 0.3011}  # per 6 x 60 window (TinierHAR: 150.6k multiply-adds)


def point(model):
    """(MFLOPs, mean target accuracy %, parameters) or None if a target is missing."""
    acc, info = [], None
    for t in daghar.DOMAINS:
        rs = style.runs("daghar_lodo", t, model)
        if not rs:
            return None
        acc.append(100 * np.mean([r["scores"][t]["accuracy"] for r, _ in rs]))
        info = rs[0][0]
    mflops = info.get("mflops_spectral", info.get("mflops", TINY_MFLOPS.get(model.split("-")[0])))
    params = info.get("params_spectral", info.get("params"))
    return (mflops, float(np.mean(acc)), params) if mflops else None


def plot(png_dir=None):
    plt = style.setup(**{"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.linewidth": 0.8})
    from matplotlib.ticker import FixedLocator, NullFormatter

    fig, ax = plt.subplots(figsize=(12 * style.CM, 8 * style.CM))
    for label, colour, ink, models, (lx, ly) in FAMILIES:
        pts = [p for p in map(point, models) if p]
        if not pts:
            print("skipped (missing results):", label)
            continue
        xs, ys, ps = zip(*pts)
        if len(pts) > 1:
            ax.plot(xs, ys, ls=(0, (3, 2)), lw=0.9, color=style.INK, zorder=2)
        ax.scatter(xs, ys, s=[25 + 260 * p / 64000 for p in ps], color=colour, edgecolor="#2b2b29", linewidth=0.9,
                   zorder=3)
        ax.text(lx, ly, label, fontsize=8.5, fontweight="bold", color=ink, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.25", facecolor=colour, edgecolor="none"))
    ax.set_xscale("log")
    ax.set_xlim(4.5, 0.035)
    ticks = [3, 2, 1, 0.5, 0.2, 0.1, 0.05]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(69.5, 77.5)
    ax.set_xlabel("MFLOPs per window (log, cheaper to the right)")
    ax.set_ylabel("DAGHAR target accuracy (%)")
    ax.grid(True, which="major", color="#dedcd6", lw=0.6, zorder=0)
    ax.grid(True, which="minor", axis="x", color="#eeede9", lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(1, 0, ">", color="black", ms=4, transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^", color="black", ms=4, transform=ax.transAxes, clip_on=False)
    fig.tight_layout()
    style.save(fig, "pareto", png_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--png")
    plot(parser.parse_args().png)
