"""The circular FIN as one linear map of the power spectrum (Parseval).

For a z-scored axis z of length T and circular convolutions, m_k = mean_t (a_k * z)_t (b_k * z)_t = sum_f G_k(f) P(f),
with P(f) = |Z(f)|^2 over the T // 2 + 1 one-sided frequencies and G_k(f) = Re(H_ak(f) conj H_bk(f)) w_f / T^2
(H = frequency response of the kernel, w_f = 1 at 0 and Nyquist, 2 elsewhere). Hence

    emb = ReLU(P L + c),  L = G^T W  of size (T // 2 + 1) x emb,

and the trained FIN runs as one real FFT and one matrix per axis. Exact for circular convolutions; for zero padding it
is an approximation.
"""
import numpy as np


def weights(fin):
    """(a (K, width), b (K, width), W (width, emb), c (emb,)) of a model.FIN, float64."""
    a = fin.a.weight.detach().cpu().double().numpy()[:, 0, :].T
    b = fin.b.weight.detach().cpu().double().numpy()[:, 0, :].T
    w = fin.proj.weight.detach().cpu().double().numpy().T
    c = fin.proj.bias.detach().cpu().double().numpy()
    return a, b, w, c


def matrix(fin, t):
    """L (T // 2 + 1, emb) and c (emb,) such that emb = ReLU(|rfft(zscore(x))|^2 @ L + c)."""
    a, b, w, c = weights(fin)
    k = a.shape[0]
    f = np.arange(t // 2 + 1)
    phase = np.exp(2j * np.pi * np.outer(np.arange(k) - (k - 1) // 2, f) / t)  # (K, F)
    ha, hb = a.T @ phase, b.T @ phase  # (width, F)
    wf = np.full(len(f), 2.0)
    wf[0] = 1.0
    if t % 2 == 0:
        wf[-1] = 1.0
    g = np.real(ha * np.conj(hb)) * wf / t ** 2
    return g.T @ w, c


def zscore(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def power(x):
    return np.abs(np.fft.rfft(zscore(np.asarray(x, np.float64)), axis=1)) ** 2


def embed(x, lmat, c):
    """Spectral form of the FIN for single axes x (n, T)."""
    return np.maximum(power(x) @ lmat + c, 0)


def max_relative_error(fin, x):
    """max |FIN(x) - spectral form(x)| / max |FIN(x)| on single axes x (n, T)."""
    import torch

    with torch.no_grad():
        conv = fin.embed(torch.as_tensor(np.asarray(x, np.float32), device=next(fin.parameters()).device))
    conv = conv.cpu().double().numpy()
    lmat, c = matrix(fin, x.shape[1])
    return float(np.abs(conv - embed(x, lmat, c)).max() / max(np.abs(conv).max(), 1e-12))


def share_above(lmat, t, fs, rate):
    """Share of the total |L| weight on frequencies above rate / 2: what a device sampling at `rate` Hz cannot see."""
    freqs = np.fft.rfftfreq(t, 1 / fs)
    weight = np.abs(lmat).sum(1)
    return float(weight[freqs > rate / 2].sum() / weight.sum())
