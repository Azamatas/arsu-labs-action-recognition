import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from config import ACTIVITIES, SEGMENT

RAW_DIR = os.path.join(HERE, "..", "..", "datasets", "MotionSense", "B_Accelerometer_data")
CACHE_PATH = os.path.join(HERE, "motionsense_cache.npz")
CODES = {"jog": "Jogging", "wlk": "Walking", "ups": "Upstairs", "dws": "Downstairs", "sit": "Sitting", "std": "Standing"}
G = 9.81
CLIP = 19.6


def window_raw(rebuild=False):
    if not rebuild and os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH)
        if int(z["segment"]) == SEGMENT:
            return z["windows"], z["users"], z["labels"]

    windows, users, labels = [], [], []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*_*", "sub_*.csv"))):
        activity = os.path.basename(os.path.dirname(path)).split("_")[0]
        user = int(re.search(r"sub_(\d+)", path).group(1))
        xyz = pd.read_csv(path)[["x", "y", "z"]].to_numpy(np.float64) * G
        xyz = np.clip(resample_poly(xyz, 2, 5, axis=0), -CLIP, CLIP).T  # 50 Hz -> 20 Hz, (3, time)
        for start in range(0, xyz.shape[1] - SEGMENT, SEGMENT // 2):
            windows.append(xyz[:, start:start + SEGMENT])
            users.append(user)
            labels.append(ACTIVITIES.index(CODES[activity]))

    windows = np.stack(windows).astype(np.float32)
    users, labels = np.array(users), np.array(labels)
    np.savez(CACHE_PATH, windows=windows, users=users, labels=labels, segment=SEGMENT)
    return windows, users, labels


if __name__ == "__main__":
    windows, users, labels = window_raw(rebuild=True)
    print(f"{len(windows)} windows of {windows.shape[1:]} from {len(np.unique(users))} users")
    print(pd.crosstab(users, np.array(ACTIVITIES)[labels]))
