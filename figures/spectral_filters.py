"""What does the pretrained FIN look at? With circular convolutions the FIN is exactly emb = ReLU(P L + c), P = power
spectrum of the z-scored axis (bifin/spectral.py), so every embedding unit is one learned weighting of the frequencies,
a column of L. The paper's FIN (64 filters, kernel 61, 10 s windows at 20 Hz, circular) is pretrained on the WISDM
training users 1-26 (cross-dataset protocol, without the flipped copies: a flip does not change a power spectrum):

  (a) mean power spectrum of each activity (training windows, all three axes)
  (b) the weights of the units that are ever active, each scaled to max |weight| = 1, sorted by the frequency of
      their largest weight
  (c) mean activation of each unit per activity (each unit scaled to its maximum), units in the order of (b)

    python figures/spectral_filters.py           # figures/out/spectral_filters.eps (+ .npz with the numbers)
    python figures/spectral_filters.py --plot    # redraw from the .npz
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from data_processing.activities import PAPER  # noqa: E402
from figures import style  # noqa: E402

NPZ = os.path.join(style.OUT, "spectral_filters.npz")
RATE = 20
COLOURS = ["#2a78d6", "#eb6834", "#1f9e89", "#c23b8a", "#8a8a86", "#d9a400"]


def compute(seed=0):
    from bifin import config, model, spectral, train
    from protocols import cross_dataset

    x, y, _ = cross_dataset.data("w2m")
    x, y = x[: len(x) // 2], y[: len(y) // 2]  # drop the flipped copies
    axes = x.reshape(-1, x.shape[-1])
    labels = np.repeat(y, x.shape[1])
    cfg = config.preset("bifin", circular=True)
    train.seed_all(seed)
    fin = model.FIN(cfg.width, cfg.kernel, cfg.emb, circular=True).to(train.DEV)
    mse = train.pretrain_fin(fin, axes, max_lag=60, seed=seed)
    lmat, c = spectral.matrix(fin, axes.shape[1])
    power = spectral.power(axes)
    emb = np.maximum(power @ lmat + c, 0)
    err = spectral.max_relative_error(fin, axes[:2000])
    print(f"imitation MSE {mse:.3f}; circular FIN vs FFT + matrix: max rel. diff {err:.1e}")
    os.makedirs(style.OUT, exist_ok=True)
    np.savez(NPZ, L=lmat, c=c, freqs=np.fft.rfftfreq(axes.shape[1], 1 / RATE),
             class_power=np.stack([power[labels == k].mean(0) for k in range(len(PAPER))]),
             class_emb=np.stack([emb[labels == k].mean(0) for k in range(len(PAPER))]),
             active=(emb > 0).mean(0), activities=np.array(PAPER), imitation_mse=mse, equivalence_error=err)


def plot(png_dir=None):
    plt = style.setup(**{"lines.linewidth": 1.0})
    d = np.load(NPZ)
    f, lmat, acts = d["freqs"], d["L"], [str(a) for a in d["activities"]]
    keep = np.where(d["active"] > 0)[0]
    rows = lmat[:, keep].T
    rows = rows / np.abs(rows).max(1, keepdims=True)
    order = np.argsort(f[rows.argmax(1)], kind="stable")
    rows = rows[order]
    sel = d["class_emb"][:, keep][:, order].T
    sel = sel / np.maximum(sel.max(1, keepdims=True), 1e-12)

    fig = plt.figure(figsize=(14 * style.CM, 9 * style.CM))
    grid = fig.add_gridspec(2, 2, width_ratios=[4, 1.1], height_ratios=[1.3, 2.2], wspace=0.08, hspace=0.28)
    top, heat, cls = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    for k, a in enumerate(acts):
        top.plot(f[1:], d["class_power"][k, 1:] / d["class_power"][k, 1:].sum(), color=COLOURS[k], label=a)
    top.set_yscale("log")
    top.set_xlim(0, f[-1])
    top.set_xticklabels([])
    top.set_ylabel("power share")
    top.set_title("(a) mean power spectrum per activity", loc="left")
    top.spines[["top", "right"]].set_visible(False)
    top.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=1.2,
               labelspacing=0.25, borderaxespad=0)

    step = f[1] - f[0]
    im = heat.imshow(rows, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
                     extent=[f[0] - step / 2, f[-1] + step / 2, len(rows) - 0.5, -0.5])
    heat.set_xlim(0, f[-1])
    heat.set_xlabel("frequency (Hz)")
    heat.set_ylabel(f"embedding unit ({len(rows)} active of {lmat.shape[1]})")
    heat.set_yticks([])
    heat.set_title("(b) rows of $L$: weight of each frequency", loc="left")

    cls.imshow(sel, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    cls.set_xticks(range(len(acts)))
    cls.set_xticklabels(acts, rotation=90)
    cls.set_yticks([])
    cls.set_title("(c) mean activation", loc="left")
    cb = fig.colorbar(im, cax=heat.inset_axes([0.3, -0.33, 0.4, 0.035]), orientation="horizontal")
    cb.set_ticks([-1, 0, 1])
    cb.ax.tick_params(labelsize=6)
    style.save(fig, "spectral_filters", png_dir, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="only redraw from the .npz")
    parser.add_argument("--png", help="directory for a PNG preview")
    args = parser.parse_args()
    if not args.plot:
        compute()
    plot(args.png)
