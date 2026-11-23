"""WISDM v1.1 (Kwapisz et al. 2011): 36 users, phone in the front pocket, accelerometer at 20 Hz, m/s^2.

Windows of 200 samples (10 s) with a step of 100, cut inside every unbroken run of one user doing one activity.
"""
import io
import os

import numpy as np
import pandas as pd

from . import DATASETS
from .activities import PAPER
from .cache import cached

RAW = os.path.join(DATASETS, "WISDM_ar_v1.1", "WISDM_ar_v1.1_raw.txt")
AXES = ("x", "y", "z")


def read_raw():
    """Every record of the raw file ends with ';', some lines hold several records."""
    with open(RAW) as f:
        lines = [line for line in f.read().replace(";", "\n").splitlines() if line.strip()]
    df = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=["user", "activity", "time", *AXES],
                     usecols=range(6), engine="python", on_bad_lines="skip")
    df = df[df["activity"].isin(PAPER)]
    df[list(AXES)] = df[list(AXES)].fillna(0.0)
    return df


def _build(length):
    df = read_raw()
    user, act = df["user"].to_numpy(), df["activity"].to_numpy()
    sig = np.stack([df[a].to_numpy(np.float64) for a in AXES])
    label = df["activity"].map({a: i for i, a in enumerate(PAPER)}).to_numpy()
    change = np.flatnonzero((act[:-1] != act[1:]) | (user[:-1] != user[1:])) + 1
    bounds = np.concatenate(([0], change, [len(act)]))
    starts = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        k = lo
        while k + length < hi:  # strict: a window ending exactly at the end of a run is dropped (as in the paper)
            starts.append(k)
            k += length // 2
    starts = np.array(starts)
    x = np.stack([sig[:, s:s + length] for s in starts]).astype(np.float32)
    return {"x": x, "y": label[starts], "user": user[starts]}


def windows(length=200, rebuild=False):
    """(x (n, 3, length) float32, y (n,), user (n,)) for all 36 users."""
    d = cached(f"wisdm_{length}", lambda: _build(length), rebuild)
    return d["x"], d["y"], d["user"]
