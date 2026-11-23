"""Check that the bilinear FIN equals a linear map of the power spectrum (Parseval).

For one z-scored axis x of length T, the FIN computes m_k = mean_t (a_k * x)_t (b_k * x)_t and emb = ReLU(W m + c).
With CIRCULAR convolutions, Parseval gives m_k = sum_f G_k(f) P(f), where P(f) = |X(f)|^2 over the T // 2 + 1
one-sided frequencies and G_k(f) = Re(H_ak(f) conj(H_bk(f))) * w_f / T^2 (H = frequency response of the kernel as
Keras applies it, w_f = 1 at f = 0 and Nyquist, 2 elsewhere). Hence emb = ReLU(L P + c) with L = W G, of size
width x (T // 2 + 1): two convolutions, product, mean and dense layer collapse into one matrix on the spectrum.

Checks, on real DAGHAR windows (T = 60, kernel 31) and WISDM-length windows (T = 200, kernel 61):
  1. our numpy zero-padded convolution reproduces the Keras FIN (sanity of the kernel orientation)
  2. circular FIN (same weights) vs spectral form: should agree to float precision
  3. the FIN as trained now (zero padding) vs spectral form: size of the approximation
and the FLOPs of both forms.

    python efficiency/spectral_equivalence.py        # writes efficiency/results_spectral.md
"""
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "daghar_benchmark"), ROOT, os.path.join(ROOT, "data-processing")]

import keras  # noqa: E402
import numpy as np  # noqa: E402

import fin_daghar_acc_only as fd  # noqa: E402  (kernel 31, lags 5-30)

bfin, cfg = fd.bfin, fd.cfg
N_WINDOWS, PRETRAIN_WINDOWS = 5000, 20000


def zscore(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def weights(fin):
    a = fin.a.get_weights()[0][:, 0, :]  # (K, width)
    b = fin.b.get_weights()[0][:, 0, :]
    w, c = fin.proj.get_weights()
    return a, b, w, c


def conv(x, k, circular):
    """Keras Conv1D 'same' (cross-correlation): y_t = sum_j k_j x_{t + j - (K - 1) // 2}; zeros or wrap-around."""
    n, t = x.shape
    kk, width = k.shape
    left = (kk - 1) // 2
    idx = np.arange(t)[:, None] + np.arange(kk)[None, :] - left  # (T, K)
    if circular:
        win = x[:, idx % t]
    else:
        pad = np.concatenate([np.zeros((n, left)), x, np.zeros((n, kk - 1 - left))], axis=1)
        win = pad[:, idx + left]
    return np.einsum("ntk,kw->ntw", win, k)


def fin_embed(x, wts, circular):
    a, b, w, c = wts
    xs = zscore(x.astype(np.float64))
    m = (conv(xs, a, circular) * conv(xs, b, circular)).mean(1)
    return np.maximum(m @ w + c, 0)


def spectral_matrix(wts, t):
    a, b, w, _ = wts
    kk = a.shape[0]
    f = np.arange(t // 2 + 1)
    phase = np.exp(2j * np.pi * np.outer(np.arange(kk) - (kk - 1) // 2, f) / t)  # (K, F)
    ha, hb = a.T @ phase, b.T @ phase  # (width, F)
    wf = np.full(len(f), 2.0)
    wf[0] = 1.0
    if t % 2 == 0:
        wf[-1] = 1.0
    g = np.real(ha * np.conj(hb)) * wf / t ** 2  # (width, F)
    return g.T @ w  # (F, emb): emb = ReLU(P @ L + c)


def spectral_embed(x, wts, lmat):
    p = np.abs(np.fft.rfft(zscore(x.astype(np.float64)), axis=1)) ** 2
    return np.maximum(p @ lmat + wts[3], 0)


def flops(t, kk, width, emb):
    direct = 2 * 2 * width * kk * t + width * t + width * t + 2 * width * emb
    spectral = 2.5 * t * np.log2(t) + 3 * (t // 2 + 1) + 2 * (t // 2 + 1) * emb  # real FFT, |X|^2, matrix
    return direct, spectral


def check(axes, kernel, seed=0):
    t = axes.shape[1]
    bfin.KERNEL = kernel
    keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    pre = axes[rng.choice(len(axes), min(PRETRAIN_WINDOWS, len(axes)), replace=False)]
    fin = bfin.RhythmFIN()
    y = bfin.targets(pre)
    fin.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
    fin.fit(pre, ((y - y.mean(0)) / y.std(0)).astype(np.float32), epochs=cfg.FIN_EPOCHS, batch_size=cfg.BATCH_SIZE,
            verbose=0)
    x = axes[rng.choice(len(axes), min(N_WINDOWS, len(axes)), replace=False)]
    wts = weights(fin)
    keras_emb = keras.ops.convert_to_numpy(fin.embed(keras.ops.convert_to_tensor(x.astype(np.float32))))
    zero_emb, circ_emb = fin_embed(x, wts, False), fin_embed(x, wts, True)
    spec_emb = spectral_embed(x, wts, spectral_matrix(wts, t))
    scale = np.abs(circ_emb).max()
    rel = lambda u, v: np.abs(u - v).max() / scale
    d, s = flops(t, kernel, bfin.WIDTH, bfin.WIDTH)
    return {"T": t, "kernel": kernel, "windows": len(x), "scale": scale,
            "keras_vs_numpy": rel(keras_emb, zero_emb), "circular_vs_spectral": rel(circ_emb, spec_emb),
            "zero_vs_spectral": rel(zero_emb, spec_emb),
            "zero_vs_spectral_mean": np.abs(zero_emb - spec_emb).mean() / np.abs(zero_emb).mean(),
            "flops_direct": d, "flops_spectral": s}


def main():
    import wisdm_windowing as ww

    rows = [("DAGHAR MotionSense train, accelerometer", check(fd.read("MotionSense", "train")[0].reshape(-1, fd.WINDOW), 31)),
            ("WISDM training split (paper windows)", check(ww.load_splits()["train"][0].reshape(-1, 200), 61))]
    lines = ["# Bilinear FIN = linear map of the power spectrum: numerical check", "",
             "Written by `efficiency/spectral_equivalence.py`. FIN pretrained (60 epochs, imitation targets) on "
             f"{PRETRAIN_WINDOWS:,} single-axis windows, evaluated on {N_WINDOWS:,} windows drawn from the same set. Errors are max |difference| of "
             "the 64-d embedding divided by its max |value|.", "",
             "| windows | T | kernel | numpy vs Keras (zero pad) | circular FIN vs spectral form | zero-pad FIN vs spectral form (max / mean) | FLOPs per axis, direct | FLOPs per axis, spectral | ratio |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in rows:
        lines.append(f"| {name} | {r['T']} | {r['kernel']} | {r['keras_vs_numpy']:.1e} | {r['circular_vs_spectral']:.1e} | "
                     f"{r['zero_vs_spectral']:.2f} / {r['zero_vs_spectral_mean']:.2f} | {r['flops_direct']:,.0f} | "
                     f"{r['flops_spectral']:,.0f} | {r['flops_direct'] / r['flops_spectral']:.0f}x |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "results_spectral.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
