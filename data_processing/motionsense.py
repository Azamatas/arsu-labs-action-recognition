"""MotionSense (Malekzadeh et al. 2019): 24 users, iPhone in the front pocket, 50 Hz, accelerometer in g.

Converted to m/s^2, resampled to 20 Hz (polyphase 2/5), clipped to +-19.6 m/s^2 (the range of WISDM), and cut into
windows of 200 samples with a step of 100 inside every recording.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from . import DATASETS
from .activities import PAPER
from .cache import cached

RAW = os.path.join(DATASETS, "MotionSense", "B_Accelerometer_data")
CODES = {"jog": "Jogging", "wlk": "Walking", "ups": "Upstairs", "dws": "Downstairs", "sit": "Sitting",
         "std": "Standing"}
G = 9.81
CLIP = 19.6


def _build(length):
    x, users, labels = [], [], []
    for path in sorted(glob.glob(os.path.join(RAW, "*_*", "sub_*.csv"))):
        activity = os.path.basename(os.path.dirname(path)).split("_")[0]
        user = int(re.search(r"sub_(\d+)", path).group(1))
        xyz = pd.read_csv(path)[["x", "y", "z"]].to_numpy(np.float64) * G
        xyz = np.clip(resample_poly(xyz, 2, 5, axis=0), -CLIP, CLIP).T  # 50 Hz -> 20 Hz, (3, time)
        for start in range(0, xyz.shape[1] - length, length // 2):
            x.append(xyz[:, start:start + length])
            users.append(user)
            labels.append(PAPER.index(CODES[activity]))
    return {"x": np.stack(x).astype(np.float32), "y": np.array(labels), "user": np.array(users)}


def windows(length=200, rebuild=False):
    """(x (n, 3, length) float32, y (n,), user (n,)) for all 24 users."""
    d = cached(f"motionsense_{length}", lambda: _build(length), rebuild)
    return d["x"], d["y"], d["user"]
