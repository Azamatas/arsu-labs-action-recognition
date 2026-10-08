import os

import numpy as np
import pandas as pd

from config import ACTIVITIES, AXES, BANDS, FS, MAX_LAG, MIN_LAG, SEGMENT, SPLITS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "..", "ClassicML", "data_processing", "datasets", "wisdm_raw_data.txt")
CACHE_PATH = os.path.join(BASE_DIR, "windows_cache.npz")


def window_raw(rebuild=False):
    if not rebuild and os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH)
        if int(z["segment"]) == SEGMENT:
            return z["windows"], z["users"], z["labels"]

    df = pd.read_csv(
        RAW_PATH,
        header=None,
        names=["user", "activity", "time", "x", "y", "z"],
        usecols=[0, 1, 2, 3, 4, 5],
        engine="python",
        on_bad_lines="skip",
    )
    df = df[df["activity"].isin(ACTIVITIES)]

    user = df["user"].to_numpy()
    act = df["activity"].to_numpy()
    sig = np.stack([df[a].to_numpy(dtype=np.float64) for a in AXES])
    label = df["activity"].map({a: i for i, a in enumerate(ACTIVITIES)}).to_numpy()

    change = np.flatnonzero((act[:-1] != act[1:]) | (user[:-1] != user[1:])) + 1
    bounds = np.concatenate(([0], change, [len(act)]))

    starts = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        k = lo
        while k + SEGMENT < hi:
            starts.append(k)
            k += SEGMENT // 2
    starts = np.array(starts)

    windows = np.stack([sig[:, s : s + SEGMENT] for s in starts]).astype(np.float32)
    users, labels = user[starts], label[starts]
    np.savez_compressed(
        CACHE_PATH, windows=windows, users=users, labels=labels, segment=SEGMENT
    )
    return windows, users, labels


def load_splits(rebuild=False):
    windows, users, labels = window_raw(rebuild)
    out = {}
    for name, group in SPLITS.items():
        m = np.isin(users, list(group))
        out[name] = (windows[m], labels[m])
    return out


def per_axis(windows):
    return windows.reshape(-1, SEGMENT)


def compute_feature(w, name):
    if name == "mean":
        return w.mean(axis=1)
    if name == "std":
        return w.std(axis=1, ddof=1)
    if name == "mad":
        m = w.mean(axis=1, keepdims=True)
        return np.abs(w - m).mean(axis=1)
    raise KeyError(name)


def compute_targets(w, names=("mean", "std", "mad")):
    return np.stack([compute_feature(w, n) for n in names], axis=1)


def temporal_axis(x):
    c = x - x.mean(axis=1, keepdims=True)
    n = c.shape[1]
    out = {}

    s = np.signbit(c)
    out["mcr"] = (s[:, :-1] != s[:, 1:]).mean(axis=1)

    spec = np.fft.rfft(c, n=2 * n, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=2 * n, axis=1).real
    ac = ac[:, MIN_LAG:MAX_LAG + 1] / np.where(np.abs(ac[:, :1]) < 1e-12, 1.0, ac[:, :1])
    out["ac_peak"] = ac.max(axis=1)
    out["ac_lag"] = (ac.argmax(axis=1) + MIN_LAG).astype(np.float64)

    d = np.diff(x, axis=1)
    out["jerk_std"] = d.std(axis=1, ddof=1)
    out["jerk_mad"] = np.abs(d - d.mean(axis=1, keepdims=True)).mean(axis=1)

    p = np.abs(np.fft.rfft(c, axis=1)[:, 1:]) ** 2
    total = p.sum(axis=1, keepdims=True)
    share = p / np.where(total < 1e-12, 1.0, total)
    freqs = (np.arange(p.shape[1]) + 1) * FS / n
    out["dom_freq"] = freqs[p.argmax(axis=1)]
    out["spec_peak"] = share.max(axis=1)
    out["spec_entropy"] = -(share * np.log(np.where(share < 1e-12, 1.0, share))).sum(axis=1)
    for name, (lo, hi) in BANDS.items():
        out[name] = share[:, (freqs > lo) & (freqs <= hi)].sum(axis=1)
    return out
