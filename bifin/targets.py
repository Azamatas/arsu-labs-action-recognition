"""Temporal features of single-axis windows. Three of them (IMITATED) are the targets the FIN is pretrained to imitate;
all eleven (TEMPORAL) are the "33 temporal features" (11 per axis) of the classical baselines.
"""
import numpy as np

FS = 20.0
MIN_LAG = 5
BANDS = {"band_lo": (0.0, 3.0), "band_mid": (3.0, 6.0), "band_hi": (6.0, 10.0)}
TEMPORAL = ("mcr", "ac_peak", "ac_lag", "dom_freq", "spec_peak", "spec_entropy", "jerk_std", "jerk_mad",
            "band_lo", "band_mid", "band_hi")
IMITATED = ("spec_entropy", "ac_peak", "spec_peak")


def temporal(x, max_lag, min_lag=MIN_LAG, fs=FS):
    """x: (n, T) single-axis windows (float64). Returns {name: (n,)} for every name in TEMPORAL."""
    c = x - x.mean(axis=1, keepdims=True)
    n = c.shape[1]
    out = {}

    s = np.signbit(c)
    out["mcr"] = (s[:, :-1] != s[:, 1:]).mean(axis=1)  # mean-crossing rate

    spec = np.fft.rfft(c, n=2 * n, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=2 * n, axis=1).real  # autocorrelation
    ac = ac[:, min_lag:max_lag + 1] / np.where(np.abs(ac[:, :1]) < 1e-12, 1.0, ac[:, :1])
    out["ac_peak"] = ac.max(axis=1)
    out["ac_lag"] = (ac.argmax(axis=1) + min_lag).astype(np.float64)

    d = np.diff(x, axis=1)
    out["jerk_std"] = d.std(axis=1, ddof=1)
    out["jerk_mad"] = np.abs(d - d.mean(axis=1, keepdims=True)).mean(axis=1)

    p = np.abs(np.fft.rfft(c, axis=1)[:, 1:]) ** 2
    total = p.sum(axis=1, keepdims=True)
    share = p / np.where(total < 1e-12, 1.0, total)
    freqs = (np.arange(p.shape[1]) + 1) * fs / n
    out["dom_freq"] = freqs[p.argmax(axis=1)]
    out["spec_peak"] = share.max(axis=1)
    out["spec_entropy"] = -(share * np.log(np.where(share < 1e-12, 1.0, share))).sum(axis=1)
    for name, (lo, hi) in BANDS.items():
        out[name] = share[:, (freqs > lo) & (freqs <= hi)].sum(axis=1)
    return out


def imitated(axes, max_lag):
    """(n, 3) imitation targets of single-axis windows (n, T): spectral entropy, autocorrelation-peak height,
    spectral peak. max_lag is half the window in our protocols (60 for 10 s windows, 30 for DAGHAR's 3 s)."""
    t = temporal(axes.astype(np.float64), max_lag)
    return np.stack([t[name] for name in IMITATED], axis=1)


def temporal33(windows, max_lag):
    """(n, 33) for (n, 3, T) windows: the 11 TEMPORAL features of x, y and z, ordered feature-major."""
    w = windows.astype(np.float64)
    axes = [temporal(w[:, a], max_lag) for a in range(w.shape[1])]
    return np.stack([axes[a][f] for f in TEMPORAL for a in range(len(axes))], axis=1)
