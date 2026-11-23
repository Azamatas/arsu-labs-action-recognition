"""DAGHAR standardized view (Napoli & Borin): six smartphone datasets harmonised to 20 Hz, m/s^2, gravity removed,
3 s windows (60 samples), accelerometer and gyroscope, labels in the DAGHAR standard activity code
(activities.DAGHAR), each dataset split into train / validation / test by user.
"""
import os

import numpy as np
import pandas as pd

from . import DATASETS
from .cache import cached

DIR = os.path.join(DATASETS, "DAGHAR", "standardized_view")
DOMAINS = ("KuHar", "MotionSense", "RealWorld_thigh", "RealWorld_waist", "UCI", "WISDM")
SPLITS = ("train", "validation", "test")
WINDOW = 60
CHANNELS = ("accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z")


def _build(domain, split):
    df = pd.read_csv(os.path.join(DIR, domain, f"{split}.csv"))
    x = np.stack([df[[f"{c}-{i}" for i in range(WINDOW)]].to_numpy(np.float32) for c in CHANNELS], axis=1)
    return {"x": x, "y": df["standard activity code"].to_numpy(int)}


def read(domain, split, accel_only=False):
    """(x (n, 6, 60) float32: accelerometer x y z, gyroscope x y z; y (n,)). accel_only keeps the first 3 channels."""
    d = cached(f"daghar_{domain}_{split}", lambda: _build(domain, split))
    return (d["x"][:, :3] if accel_only else d["x"]), d["y"]


def read_many(domains, splits, accel_only=False):
    parts = [read(d, s, accel_only) for d in domains for s in splits]
    return np.concatenate([x for x, _ in parts]), np.concatenate([y for _, y in parts])
