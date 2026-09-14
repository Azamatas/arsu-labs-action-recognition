
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "..", "data_processing", "datasets", "wisdm_raw_data.txt")
CACHE_PATH = os.path.join(BASE_DIR, "windows_cache.npz")

SEGMENT = 200
AXES = ("x", "y", "z")
ACTIVITIES = ("Jogging", "Walking", "Upstairs", "Downstairs", "Sitting", "Standing")
SPLITS = {"train": range(1, 21), "val": range(21, 27), "test": range(27, 37)}


def window_raw(rebuild=False):
    """(n, 3, SEGMENT) float32 windows, their user ids, and their activity labels."""
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
    """{"train"/"val"/"test": (windows, labels)}, subject-disjoint."""
    windows, users, labels = window_raw(rebuild)
    out = {}
    for name, group in SPLITS.items():
        m = np.isin(users, list(group))
        out[name] = (windows[m], labels[m])
    return out


def per_axis(windows):
    """(n, 3, SEGMENT) -> (3n, SEGMENT): every axis becomes its own single-axis example."""
    return windows.reshape(-1, SEGMENT)


def compute_feature(w, name):
    """Definitions match ClassicML/data_processing/generate_wisdm_data.py."""
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
