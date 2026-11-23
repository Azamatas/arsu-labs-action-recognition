"""What does the pretrained FIN look at? With circular convolutions the FIN is exactly emb = ReLU(L P + c), P = power
spectrum of the z-scored axis (efficiency/spectral_equivalence.py), so every embedding unit is one learned weighting of
the frequencies, a row of L. This script pretrains the paper's FIN (64 filters, kernel 61, 10 s windows at 20 Hz,
circular) on the WISDM training users 1-26 exactly as cross_dataset.py does, and draws:

  (a) the mean power spectrum of each activity (training windows, all three axes)
  (b) the rows of L (units that are ever active on the training windows, each scaled to max |weight| = 1),
      sorted by the frequency of their largest positive weight
  (c) the mean activation of each unit per activity (each unit scaled to its maximum), units in the order of (b)

    python interpretability/spectral_filters.py           # interpretability/spectral_filters.eps (+ .npz)
    python interpretability/spectral_filters.py --plot    # redraw from the .npz
"""
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "data-processing"), os.path.join(ROOT, "efficiency"),
                os.path.join(ROOT, "cross-dataset")]

import numpy as np  # noqa: E402

NPZ = os.path.join(HERE, "spectral_filters.npz")
EPS = os.path.join(HERE, "spectral_filters.eps")
RATE = 20
CM = 1 / 2.54
COLOURS = ["#2a78d6", "#eb6834", "#1f9e89", "#c23b8a", "#8a8a86", "#d9a400"]


def compute(seed=0):
    import keras

    import bilin_fin_temporal_plus_original as bfin
    import config as cfg
    import cross_dataset as cd
    import spectral_equivalence as se
    from bilin_fin_circular import circular

    x, y, _ = cd.load("w2m")  # WISDM users 1-26 and their flipped copies, as in the paper
    x, y = x[: len(x) // 2], y[: len(y) // 2]  # the flip does not change any power spectrum
    axes = x.reshape(-1, cfg.SEGMENT)
    labels = np.repeat(y, len(cfg.AXES))

    keras.utils.set_random_seed(seed)
    fin = circular(bfin.RhythmFIN)()
    t = bfin.targets(axes)
    fin.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
    hist = fin.fit(axes, ((t - t.mean(0)) / t.std(0)).astype(np.float32), epochs=cfg.FIN_EPOCHS,
                   batch_size=cfg.BATCH_SIZE, verbose=0)

    wts = se.weights(fin)
    lmat = se.spectral_matrix(wts, cfg.SEGMENT)  # (F, units)
    power = np.abs(np.fft.rfft(se.zscore(axes.astype(np.float64)), axis=1)) ** 2
    emb = np.maximum(power @ lmat + wts[3], 0)
    keras_emb = keras.ops.convert_to_numpy(fin.embed(keras.ops.convert_to_tensor(axes[:2000].astype(np.float32))))
    err = np.abs(keras_emb - emb[:2000]).max() / np.abs(keras_emb).max()
    print(f"imitation MSE {hist.history['loss'][-1]:.3f}; circular FIN vs FFT + matrix: max rel. diff {err:.1e}")

    np.savez(NPZ, L=lmat, c=wts[3], freqs=np.fft.rfftfreq(cfg.SEGMENT, 1 / RATE),
             class_power=np.stack([power[labels == k].mean(0) for k in range(cfg.N_CLASSES)]),
             class_emb=np.stack([emb[labels == k].mean(0) for k in range(cfg.N_CLASSES)]),
             active=(emb > 0).mean(0), activities=np.array(cfg.ACTIVITIES),
             imitation_mse=hist.history["loss"][-1], equivalence_error=err)


def plot(png_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False, "ps.fonttype": 42, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "lines.linewidth": 1.0, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })
    d = np.load(NPZ)
    f, lmat, acts = d["freqs"], d["L"], [str(a) for a in d["activities"]]
    keep = np.where(d["active"] > 0)[0]
    rows = lmat[:, keep].T
    rows = rows / np.abs(rows).max(1, keepdims=True)
    order = np.argsort(f[rows.argmax(1)], kind="stable")
    rows = rows[order]
    sel = d["class_emb"][:, keep][:, order].T
    sel = sel / np.maximum(sel.max(1, keepdims=True), 1e-12)

    fig = plt.figure(figsize=(14 * CM, 9 * CM))
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
    top.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=1.2, labelspacing=0.25, borderaxespad=0)

    extent = [f[0] - (f[1] - f[0]) / 2, f[-1] + (f[1] - f[0]) / 2, len(rows) - 0.5, -0.5]
    im = heat.imshow(rows, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, extent=extent, interpolation="nearest")
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
    fig.savefig(EPS, bbox_inches="tight", pad_inches=0.02)
    print("saved", EPS)
    if png_dir:
        png = os.path.join(png_dir, "spectral_filters.png")
        fig.savefig(png, dpi=250, bbox_inches="tight", pad_inches=0.02)
        print("saved", png)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="only redraw from the .npz")
    parser.add_argument("--png", help="directory for a PNG preview")
    args = parser.parse_args()
    if not args.plot:
        compute()
    plot(args.png)
